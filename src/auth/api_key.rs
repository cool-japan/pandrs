//! API Key Authentication
//!
//! This module provides API key generation, validation, and management
//! for service-to-service authentication and programmatic access.

use crate::error::{Error, Result};
use crate::multitenancy::{Permission, TenantId};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// API Key information
#[derive(Debug, Clone)]
pub struct ApiKeyInfo {
    /// Unique key identifier
    pub key_id: String,
    /// Human-readable name
    pub name: String,
    /// Hash of the actual key (for secure storage)
    pub key_hash: String,
    /// Associated user ID
    pub user_id: String,
    /// Associated tenant ID
    pub tenant_id: TenantId,
    /// Permissions granted to this key
    pub permissions: Vec<Permission>,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Expiration timestamp (None = never expires)
    pub expires_at: Option<SystemTime>,
    /// Last usage timestamp
    pub last_used: Option<SystemTime>,
    /// Usage count
    pub usage_count: u64,
    /// Whether the key is active
    pub active: bool,
    /// Rate limit (requests per minute)
    pub rate_limit: Option<u32>,
    /// Current rate limit counter
    pub rate_limit_counter: u32,
    /// Rate limit window start
    pub rate_limit_window_start: Option<SystemTime>,
    /// IP whitelist (empty = allow all)
    pub ip_whitelist: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ApiKeyInfo {
    /// Create a new API key info
    pub fn new(
        key_id: impl Into<String>,
        name: impl Into<String>,
        key_hash: impl Into<String>,
        user_id: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Self {
        ApiKeyInfo {
            key_id: key_id.into(),
            name: name.into(),
            key_hash: key_hash.into(),
            user_id: user_id.into(),
            tenant_id: tenant_id.into(),
            permissions: vec![Permission::Read],
            created_at: SystemTime::now(),
            expires_at: None,
            last_used: None,
            usage_count: 0,
            active: true,
            rate_limit: None,
            rate_limit_counter: 0,
            rate_limit_window_start: None,
            ip_whitelist: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set permissions
    pub fn with_permissions(mut self, permissions: Vec<Permission>) -> Self {
        self.permissions = permissions;
        self
    }

    /// Set expiration time
    pub fn with_expiration(mut self, expires_at: SystemTime) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Set expiration duration from now
    pub fn expires_in(mut self, duration: Duration) -> Self {
        self.expires_at = Some(SystemTime::now() + duration);
        self
    }

    /// Set rate limit
    pub fn with_rate_limit(mut self, requests_per_minute: u32) -> Self {
        self.rate_limit = Some(requests_per_minute);
        self
    }

    /// Add IP to whitelist
    pub fn with_ip_whitelist(mut self, ips: Vec<String>) -> Self {
        self.ip_whitelist = ips;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if the key has expired
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| exp < SystemTime::now())
            .unwrap_or(false)
    }

    /// Check if IP is allowed
    pub fn is_ip_allowed(&self, ip: &str) -> bool {
        if self.ip_whitelist.is_empty() {
            return true;
        }
        self.ip_whitelist.contains(&ip.to_string())
    }

    /// Check rate limit
    pub fn check_rate_limit(&mut self) -> bool {
        let Some(limit) = self.rate_limit else {
            return true;
        };

        let now = SystemTime::now();
        let window_start = self.rate_limit_window_start.unwrap_or(now);

        // Check if we're in a new window (1 minute)
        if now.duration_since(window_start).unwrap_or(Duration::ZERO) > Duration::from_secs(60) {
            self.rate_limit_counter = 1;
            self.rate_limit_window_start = Some(now);
            return true;
        }

        // Check if under limit
        if self.rate_limit_counter < limit {
            self.rate_limit_counter += 1;
            return true;
        }

        false
    }

    /// Record usage
    pub fn record_usage(&mut self) {
        self.last_used = Some(SystemTime::now());
        self.usage_count += 1;
    }
}

/// API Key manager for storing and validating keys
#[derive(Debug)]
pub struct ApiKeyManager {
    /// Keys stored by their hash (not the actual key)
    keys_by_hash: HashMap<String, ApiKeyInfo>,
    /// Keys stored by key ID
    keys_by_id: HashMap<String, String>, // key_id -> key_hash
    /// Keys per user
    user_keys: HashMap<String, Vec<String>>, // user_id -> [key_ids]
    /// Maximum keys per user
    max_keys_per_user: usize,
    /// Default expiration
    default_expiration: Option<Duration>,
    /// Key prefix
    key_prefix: String,
}

impl ApiKeyManager {
    /// Create a new API key manager
    pub fn new() -> Self {
        ApiKeyManager {
            keys_by_hash: HashMap::new(),
            keys_by_id: HashMap::new(),
            user_keys: HashMap::new(),
            max_keys_per_user: 10,
            default_expiration: None,
            key_prefix: "pk".to_string(),
        }
    }

    /// Set maximum keys per user
    pub fn with_max_keys(mut self, max: usize) -> Self {
        self.max_keys_per_user = max;
        self
    }

    /// Set default expiration
    pub fn with_default_expiration(mut self, duration: Duration) -> Self {
        self.default_expiration = Some(duration);
        self
    }

    /// Set key prefix
    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    /// Generate a new API key
    pub fn generate_key(
        &mut self,
        name: &str,
        user_id: &str,
        tenant_id: &str,
        permissions: Vec<Permission>,
    ) -> Result<String> {
        // Check max keys per user
        let user_key_count = self
            .user_keys
            .get(user_id)
            .map(|keys| keys.len())
            .unwrap_or(0);

        if user_key_count >= self.max_keys_per_user {
            return Err(Error::InvalidOperation(format!(
                "Maximum API keys ({}) reached for user",
                self.max_keys_per_user
            )));
        }

        // Generate the actual key
        let key = generate_api_key(&self.key_prefix);
        let key_hash = hash_api_key(&key);
        let key_id = generate_key_id();

        let mut key_info = ApiKeyInfo::new(&key_id, name, &key_hash, user_id, tenant_id)
            .with_permissions(permissions);

        // Apply default expiration
        if let Some(duration) = self.default_expiration {
            key_info = key_info.expires_in(duration);
        }

        // Store key info
        self.keys_by_hash.insert(key_hash.clone(), key_info);
        self.keys_by_id.insert(key_id.clone(), key_hash);

        // Track per-user
        self.user_keys
            .entry(user_id.to_string())
            .or_insert_with(Vec::new)
            .push(key_id);

        // Return the actual key (only time it's available in plaintext)
        Ok(key)
    }

    /// Validate an API key
    pub fn validate_key(&mut self, key: &str) -> Result<&ApiKeyInfo> {
        let key_hash = hash_api_key(key);

        let key_info = self
            .keys_by_hash
            .get(&key_hash)
            .ok_or_else(|| Error::InvalidInput("Invalid API key".to_string()))?;

        if !key_info.active {
            return Err(Error::InvalidOperation(
                "API key is deactivated".to_string(),
            ));
        }

        if key_info.is_expired() {
            return Err(Error::InvalidOperation("API key has expired".to_string()));
        }

        Ok(key_info)
    }

    /// Validate and record usage of an API key
    pub fn validate_and_use(&mut self, key: &str, ip_address: Option<&str>) -> Result<&ApiKeyInfo> {
        let key_hash = hash_api_key(key);

        let key_info = self
            .keys_by_hash
            .get_mut(&key_hash)
            .ok_or_else(|| Error::InvalidInput("Invalid API key".to_string()))?;

        if !key_info.active {
            return Err(Error::InvalidOperation(
                "API key is deactivated".to_string(),
            ));
        }

        if key_info.is_expired() {
            return Err(Error::InvalidOperation("API key has expired".to_string()));
        }

        // Check IP whitelist
        if let Some(ip) = ip_address {
            if !key_info.is_ip_allowed(ip) {
                return Err(Error::InvalidOperation(format!(
                    "IP address {} not in whitelist",
                    ip
                )));
            }
        }

        // Check rate limit
        if !key_info.check_rate_limit() {
            return Err(Error::InvalidOperation("Rate limit exceeded".to_string()));
        }

        // Record usage
        key_info.record_usage();

        Ok(self.keys_by_hash.get(&key_hash).unwrap())
    }

    /// Get key info by ID
    pub fn get_key(&self, key_id: &str) -> Option<&ApiKeyInfo> {
        self.keys_by_id
            .get(key_id)
            .and_then(|hash| self.keys_by_hash.get(hash))
    }

    /// Get mutable key info by ID
    pub fn get_key_mut(&mut self, key_id: &str) -> Option<&mut ApiKeyInfo> {
        let hash = self.keys_by_id.get(key_id)?.clone();
        self.keys_by_hash.get_mut(&hash)
    }

    /// List keys for a user
    pub fn list_user_keys(&self, user_id: &str) -> Vec<&ApiKeyInfo> {
        self.user_keys
            .get(user_id)
            .map(|key_ids| key_ids.iter().filter_map(|id| self.get_key(id)).collect())
            .unwrap_or_default()
    }

    /// Revoke a key by ID
    pub fn revoke_key(&mut self, key_id: &str) -> Result<()> {
        let key_info = self
            .get_key_mut(key_id)
            .ok_or_else(|| Error::InvalidInput("Key not found".to_string()))?;

        key_info.active = false;
        Ok(())
    }

    /// Delete a key by ID
    pub fn delete_key(&mut self, key_id: &str) -> Result<ApiKeyInfo> {
        let key_hash = self
            .keys_by_id
            .remove(key_id)
            .ok_or_else(|| Error::InvalidInput("Key not found".to_string()))?;

        let key_info = self
            .keys_by_hash
            .remove(&key_hash)
            .ok_or_else(|| Error::InvalidInput("Key info not found".to_string()))?;

        // Remove from user's key list
        if let Some(user_keys) = self.user_keys.get_mut(&key_info.user_id) {
            user_keys.retain(|id| id != key_id);
        }

        Ok(key_info)
    }

    /// Revoke all keys for a user
    pub fn revoke_user_keys(&mut self, user_id: &str) {
        if let Some(key_ids) = self.user_keys.get(user_id) {
            for key_id in key_ids.clone() {
                if let Some(key_info) = self.get_key_mut(&key_id) {
                    key_info.active = false;
                }
            }
        }
    }

    /// Update key permissions
    pub fn update_permissions(&mut self, key_id: &str, permissions: Vec<Permission>) -> Result<()> {
        let key_info = self
            .get_key_mut(key_id)
            .ok_or_else(|| Error::InvalidInput("Key not found".to_string()))?;

        key_info.permissions = permissions;
        Ok(())
    }

    /// Update key expiration
    pub fn update_expiration(
        &mut self,
        key_id: &str,
        expires_at: Option<SystemTime>,
    ) -> Result<()> {
        let key_info = self
            .get_key_mut(key_id)
            .ok_or_else(|| Error::InvalidInput("Key not found".to_string()))?;

        key_info.expires_at = expires_at;
        Ok(())
    }

    /// Cleanup expired keys
    pub fn cleanup_expired(&mut self) {
        let expired_ids: Vec<String> = self
            .keys_by_id
            .iter()
            .filter_map(|(id, hash)| {
                self.keys_by_hash
                    .get(hash)
                    .filter(|info| info.is_expired())
                    .map(|_| id.clone())
            })
            .collect();

        for key_id in expired_ids {
            let _ = self.delete_key(&key_id);
        }
    }

    /// Get total key count
    pub fn key_count(&self) -> usize {
        self.keys_by_hash.len()
    }

    /// Get active key count
    pub fn active_key_count(&self) -> usize {
        self.keys_by_hash
            .values()
            .filter(|k| k.active && !k.is_expired())
            .count()
    }

    /// Get key statistics
    pub fn get_stats(&self) -> ApiKeyStats {
        let total = self.keys_by_hash.len();
        let active = self.active_key_count();
        let expired = self
            .keys_by_hash
            .values()
            .filter(|k| k.is_expired())
            .count();
        let revoked = self.keys_by_hash.values().filter(|k| !k.active).count();

        let total_usage: u64 = self.keys_by_hash.values().map(|k| k.usage_count).sum();

        ApiKeyStats {
            total_keys: total,
            active_keys: active,
            expired_keys: expired,
            revoked_keys: revoked,
            total_usage,
        }
    }
}

impl Default for ApiKeyManager {
    fn default() -> Self {
        Self::new()
    }
}

/// API key statistics
#[derive(Debug, Clone)]
pub struct ApiKeyStats {
    /// Total number of keys
    pub total_keys: usize,
    /// Number of active keys
    pub active_keys: usize,
    /// Number of expired keys
    pub expired_keys: usize,
    /// Number of revoked keys
    pub revoked_keys: usize,
    /// Total usage count across all keys
    pub total_usage: u64,
}

/// Scoped API key for limited access
#[derive(Debug, Clone)]
pub struct ScopedApiKey {
    /// Key info
    pub key_info: ApiKeyInfo,
    /// Allowed resources (dataset IDs, etc.)
    pub allowed_resources: Vec<String>,
    /// Allowed operations
    pub allowed_operations: Vec<String>,
    /// Time-based restrictions (start, end)
    pub time_restrictions: Option<(SystemTime, SystemTime)>,
}

impl ScopedApiKey {
    /// Create a new scoped API key
    pub fn new(key_info: ApiKeyInfo) -> Self {
        ScopedApiKey {
            key_info,
            allowed_resources: Vec::new(),
            allowed_operations: Vec::new(),
            time_restrictions: None,
        }
    }

    /// Add allowed resource
    pub fn allow_resource(mut self, resource: impl Into<String>) -> Self {
        self.allowed_resources.push(resource.into());
        self
    }

    /// Add allowed operation
    pub fn allow_operation(mut self, operation: impl Into<String>) -> Self {
        self.allowed_operations.push(operation.into());
        self
    }

    /// Set time restrictions
    pub fn with_time_restrictions(mut self, start: SystemTime, end: SystemTime) -> Self {
        self.time_restrictions = Some((start, end));
        self
    }

    /// Check if resource access is allowed
    pub fn can_access_resource(&self, resource: &str) -> bool {
        if self.allowed_resources.is_empty() {
            return true;
        }
        self.allowed_resources.contains(&resource.to_string())
    }

    /// Check if operation is allowed
    pub fn can_perform_operation(&self, operation: &str) -> bool {
        if self.allowed_operations.is_empty() {
            return true;
        }
        self.allowed_operations.contains(&operation.to_string())
    }

    /// Check if access is within time restrictions
    pub fn is_within_time_restrictions(&self) -> bool {
        match self.time_restrictions {
            None => true,
            Some((start, end)) => {
                let now = SystemTime::now();
                now >= start && now <= end
            }
        }
    }
}

// Helper functions

/// Generate an API key
fn generate_api_key(prefix: &str) -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    format!(
        "{}_{}",
        prefix,
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Generate a key ID
fn generate_key_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    format!(
        "key_{}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Hash API key for storage
fn hash_api_key(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_generation() {
        let mut manager = ApiKeyManager::new();

        let key = manager
            .generate_key(
                "test-key",
                "user1",
                "tenant_a",
                vec![Permission::Read, Permission::Write],
            )
            .unwrap();

        assert!(key.starts_with("pk_"));
        assert_eq!(key.len(), 3 + 64); // "pk_" + 32 bytes as hex
    }

    #[test]
    fn test_api_key_validation() {
        let mut manager = ApiKeyManager::new();

        let key = manager
            .generate_key("test-key", "user1", "tenant_a", vec![Permission::Read])
            .unwrap();

        // Valid key
        let result = manager.validate_key(&key);
        assert!(result.is_ok());

        // Invalid key
        let result = manager.validate_key("invalid_key");
        assert!(result.is_err());
    }

    #[test]
    fn test_api_key_expiration() {
        let mut manager = ApiKeyManager::new().with_default_expiration(Duration::from_millis(1));

        let key = manager
            .generate_key("test-key", "user1", "tenant_a", vec![Permission::Read])
            .unwrap();

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        let result = manager.validate_key(&key);
        assert!(result.is_err());
    }

    #[test]
    fn test_api_key_revocation() {
        let mut manager = ApiKeyManager::new();

        let key = manager
            .generate_key("test-key", "user1", "tenant_a", vec![Permission::Read])
            .unwrap();

        // Get key ID
        let key_id = manager.list_user_keys("user1")[0].key_id.clone();

        // Revoke key
        manager.revoke_key(&key_id).unwrap();

        // Validation should fail
        let result = manager.validate_key(&key);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_keys_per_user() {
        let mut manager = ApiKeyManager::new().with_max_keys(2);

        manager
            .generate_key("key1", "user1", "tenant_a", vec![])
            .unwrap();
        manager
            .generate_key("key2", "user1", "tenant_a", vec![])
            .unwrap();

        // Third key should fail
        let result = manager.generate_key("key3", "user1", "tenant_a", vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ip_whitelist() {
        let mut manager = ApiKeyManager::new();

        let key = manager
            .generate_key("test-key", "user1", "tenant_a", vec![Permission::Read])
            .unwrap();

        // Get key ID and update whitelist
        let key_id = manager.list_user_keys("user1")[0].key_id.clone();
        if let Some(key_info) = manager.get_key_mut(&key_id) {
            key_info.ip_whitelist = vec!["192.168.1.1".to_string()];
        }

        // Allowed IP
        let result = manager.validate_and_use(&key, Some("192.168.1.1"));
        assert!(result.is_ok());

        // Blocked IP
        let result = manager.validate_and_use(&key, Some("10.0.0.1"));
        assert!(result.is_err());
    }

    #[test]
    fn test_rate_limiting() {
        let mut manager = ApiKeyManager::new();

        let key = manager
            .generate_key("test-key", "user1", "tenant_a", vec![Permission::Read])
            .unwrap();

        // Set rate limit
        let key_id = manager.list_user_keys("user1")[0].key_id.clone();
        if let Some(key_info) = manager.get_key_mut(&key_id) {
            key_info.rate_limit = Some(2);
        }

        // First two requests should succeed
        assert!(manager.validate_and_use(&key, None).is_ok());
        assert!(manager.validate_and_use(&key, None).is_ok());

        // Third request should fail
        let result = manager.validate_and_use(&key, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_api_key_stats() {
        let mut manager = ApiKeyManager::new();

        manager
            .generate_key("key1", "user1", "tenant_a", vec![])
            .unwrap();
        manager
            .generate_key("key2", "user1", "tenant_a", vec![])
            .unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_keys, 2);
        assert_eq!(stats.active_keys, 2);
        assert_eq!(stats.revoked_keys, 0);
    }

    #[test]
    fn test_scoped_api_key() {
        let key_info = ApiKeyInfo::new("key_123", "test-key", "hash", "user1", "tenant_a");

        let scoped = ScopedApiKey::new(key_info)
            .allow_resource("dataset_a")
            .allow_resource("dataset_b")
            .allow_operation("read")
            .allow_operation("query");

        assert!(scoped.can_access_resource("dataset_a"));
        assert!(!scoped.can_access_resource("dataset_c"));
        assert!(scoped.can_perform_operation("read"));
        assert!(!scoped.can_perform_operation("write"));
    }
}
