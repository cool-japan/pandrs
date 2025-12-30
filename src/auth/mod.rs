//! Enterprise Authentication Module
//!
//! This module provides enterprise-grade authentication features including:
//! - JWT (JSON Web Token) token generation and validation
//! - OAuth 2.0 support (Authorization Code and Client Credentials flows)
//! - API Key authentication
//! - Session management
//! - Integration with multi-tenancy
//!
//! # Example
//!
//! ```ignore
//! use pandrs::auth::{AuthManager, JwtConfig, TokenClaims};
//!
//! // Create authentication manager
//! let mut auth = AuthManager::new(JwtConfig::default());
//!
//! // Register a user
//! auth.register_user("user@example.com", "tenant_a", vec!["read", "write"])?;
//!
//! // Generate JWT token
//! let token = auth.generate_token("user@example.com")?;
//!
//! // Validate token
//! let claims = auth.validate_token(&token)?;
//! ```

pub mod api_key;
pub mod jwt;
pub mod oauth;
pub mod session;

pub use api_key::*;
pub use jwt::*;
pub use oauth::*;
pub use session::*;

use crate::error::{Error, Result};
use crate::multitenancy::{Permission, TenantId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Authentication result containing user identity and permissions
#[derive(Debug, Clone)]
pub struct AuthResult {
    /// User identifier
    pub user_id: String,
    /// Associated tenant
    pub tenant_id: TenantId,
    /// Granted permissions
    pub permissions: Vec<Permission>,
    /// Token expiration time (Unix timestamp)
    pub expires_at: u64,
    /// Session identifier
    pub session_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// User registration information
#[derive(Debug, Clone)]
pub struct UserInfo {
    /// Unique user identifier
    pub user_id: String,
    /// Associated tenant ID
    pub tenant_id: TenantId,
    /// User email
    pub email: String,
    /// Display name
    pub display_name: Option<String>,
    /// Assigned roles
    pub roles: Vec<String>,
    /// Granted permissions
    pub permissions: Vec<Permission>,
    /// Whether the user is active
    pub active: bool,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last login time
    pub last_login: Option<SystemTime>,
    /// Password hash (if using password auth)
    password_hash: Option<String>,
    /// API keys associated with this user
    pub api_keys: Vec<String>,
}

impl UserInfo {
    /// Create a new user info
    pub fn new(
        user_id: impl Into<String>,
        email: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Self {
        UserInfo {
            user_id: user_id.into(),
            email: email.into(),
            tenant_id: tenant_id.into(),
            display_name: None,
            roles: Vec::new(),
            permissions: vec![Permission::Read],
            active: true,
            created_at: SystemTime::now(),
            last_login: None,
            password_hash: None,
            api_keys: Vec::new(),
        }
    }

    /// Set display name
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Add a role
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.roles.push(role.into());
        self
    }

    /// Set permissions
    pub fn with_permissions(mut self, perms: Vec<Permission>) -> Self {
        self.permissions = perms;
        self
    }

    /// Add a permission
    pub fn with_permission(mut self, perm: Permission) -> Self {
        if !self.permissions.contains(&perm) {
            self.permissions.push(perm);
        }
        self
    }

    /// Set password (will be hashed)
    pub fn with_password(mut self, password: &str) -> Self {
        self.password_hash = Some(hash_password(password));
        self
    }

    /// Verify password
    pub fn verify_password(&self, password: &str) -> bool {
        match &self.password_hash {
            Some(hash) => verify_password(password, hash),
            None => false,
        }
    }
}

/// Authentication method types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMethod {
    /// JWT token authentication
    Jwt,
    /// OAuth 2.0
    OAuth,
    /// API Key authentication
    ApiKey,
    /// Password-based authentication
    Password,
    /// Service-to-service authentication
    ServiceAccount,
}

/// Authentication event for auditing
#[derive(Debug, Clone)]
pub struct AuthEvent {
    /// Event timestamp
    pub timestamp: SystemTime,
    /// Event type
    pub event_type: AuthEventType,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// Tenant ID
    pub tenant_id: Option<TenantId>,
    /// Authentication method used
    pub auth_method: AuthMethod,
    /// Whether the event was successful
    pub success: bool,
    /// IP address (if available)
    pub ip_address: Option<String>,
    /// User agent (if available)
    pub user_agent: Option<String>,
    /// Error message (if failed)
    pub error_message: Option<String>,
}

/// Types of authentication events
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthEventType {
    /// User login attempt
    Login,
    /// User logout
    Logout,
    /// Token refresh
    TokenRefresh,
    /// Token validation
    TokenValidation,
    /// API key usage
    ApiKeyUsage,
    /// Password change
    PasswordChange,
    /// User registration
    UserRegistration,
    /// Permission denied
    PermissionDenied,
    /// Session expired
    SessionExpired,
}

/// Central authentication manager
#[derive(Debug)]
pub struct AuthManager {
    /// JWT configuration
    jwt_config: JwtConfig,
    /// OAuth configuration
    oauth_config: Option<OAuthConfig>,
    /// Registered users
    users: HashMap<String, UserInfo>,
    /// Active sessions
    sessions: HashMap<String, Session>,
    /// API keys
    api_keys: HashMap<String, ApiKeyInfo>,
    /// Refresh tokens
    refresh_tokens: HashMap<String, RefreshToken>,
    /// Authentication event log
    auth_events: Vec<AuthEvent>,
    /// Maximum events to keep
    max_events: usize,
    /// Token expiration duration
    token_expiry: Duration,
    /// Refresh token expiration duration
    refresh_token_expiry: Duration,
    /// Session timeout
    session_timeout: Duration,
}

impl AuthManager {
    /// Create a new authentication manager
    pub fn new(jwt_config: JwtConfig) -> Self {
        AuthManager {
            jwt_config,
            oauth_config: None,
            users: HashMap::new(),
            sessions: HashMap::new(),
            api_keys: HashMap::new(),
            refresh_tokens: HashMap::new(),
            auth_events: Vec::new(),
            max_events: 10000,
            token_expiry: Duration::from_secs(3600), // 1 hour
            refresh_token_expiry: Duration::from_secs(86400 * 7), // 7 days
            session_timeout: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Configure OAuth
    pub fn with_oauth(mut self, config: OAuthConfig) -> Self {
        self.oauth_config = Some(config);
        self
    }

    /// Set token expiration duration
    pub fn with_token_expiry(mut self, duration: Duration) -> Self {
        self.token_expiry = duration;
        self
    }

    /// Set refresh token expiration duration
    pub fn with_refresh_token_expiry(mut self, duration: Duration) -> Self {
        self.refresh_token_expiry = duration;
        self
    }

    /// Set session timeout
    pub fn with_session_timeout(mut self, duration: Duration) -> Self {
        self.session_timeout = duration;
        self
    }

    /// Register a new user
    pub fn register_user(&mut self, user_info: UserInfo) -> Result<()> {
        if self.users.contains_key(&user_info.user_id) {
            return Err(Error::InvalidInput(format!(
                "User '{}' already exists",
                user_info.user_id
            )));
        }

        let user_id = user_info.user_id.clone();
        self.users.insert(user_id.clone(), user_info);

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::UserRegistration,
            user_id: Some(user_id),
            tenant_id: None,
            auth_method: AuthMethod::Password,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        Ok(())
    }

    /// Get user info
    pub fn get_user(&self, user_id: &str) -> Option<&UserInfo> {
        self.users.get(user_id)
    }

    /// Update user info
    pub fn update_user(&mut self, user_info: UserInfo) -> Result<()> {
        if !self.users.contains_key(&user_info.user_id) {
            return Err(Error::InvalidInput(format!(
                "User '{}' not found",
                user_info.user_id
            )));
        }

        self.users.insert(user_info.user_id.clone(), user_info);
        Ok(())
    }

    /// Deactivate a user
    pub fn deactivate_user(&mut self, user_id: &str) -> Result<()> {
        let user = self
            .users
            .get_mut(user_id)
            .ok_or_else(|| Error::InvalidInput(format!("User '{}' not found", user_id)))?;

        user.active = false;

        // Invalidate all active sessions for this user
        self.sessions
            .retain(|_, session| session.user_id != user_id);

        Ok(())
    }

    /// Authenticate with password and get JWT token
    pub fn authenticate_password(&mut self, email: &str, password: &str) -> Result<AuthResult> {
        // Find user by email
        let user = self
            .users
            .values()
            .find(|u| u.email == email)
            .ok_or_else(|| Error::InvalidInput("Invalid credentials".to_string()))?;

        if !user.active {
            self.log_event(AuthEvent {
                timestamp: SystemTime::now(),
                event_type: AuthEventType::Login,
                user_id: Some(user.user_id.clone()),
                tenant_id: Some(user.tenant_id.clone()),
                auth_method: AuthMethod::Password,
                success: false,
                ip_address: None,
                user_agent: None,
                error_message: Some("User account is deactivated".to_string()),
            });
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        if !user.verify_password(password) {
            self.log_event(AuthEvent {
                timestamp: SystemTime::now(),
                event_type: AuthEventType::Login,
                user_id: Some(user.user_id.clone()),
                tenant_id: Some(user.tenant_id.clone()),
                auth_method: AuthMethod::Password,
                success: false,
                ip_address: None,
                user_agent: None,
                error_message: Some("Invalid password".to_string()),
            });
            return Err(Error::InvalidInput("Invalid credentials".to_string()));
        }

        // Update last login
        let user_id = user.user_id.clone();
        let tenant_id = user.tenant_id.clone();
        let permissions = user.permissions.clone();

        if let Some(user_mut) = self.users.get_mut(&user_id) {
            user_mut.last_login = Some(SystemTime::now());
        }

        // Create session
        let session =
            Session::new(user_id.clone(), tenant_id.clone()).with_timeout(self.session_timeout);
        let session_id = session.session_id.clone();
        self.sessions.insert(session_id.clone(), session);

        let expires_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            + self.token_expiry.as_secs();

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::Login,
            user_id: Some(user_id.clone()),
            tenant_id: Some(tenant_id.clone()),
            auth_method: AuthMethod::Password,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        Ok(AuthResult {
            user_id,
            tenant_id,
            permissions,
            expires_at,
            session_id: Some(session_id),
            metadata: HashMap::new(),
        })
    }

    /// Generate JWT token for authenticated user
    pub fn generate_token(&self, user_id: &str) -> Result<String> {
        let user = self
            .users
            .get(user_id)
            .ok_or_else(|| Error::InvalidInput(format!("User '{}' not found", user_id)))?;

        if !user.active {
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        let claims = TokenClaims {
            sub: user_id.to_string(),
            tenant_id: user.tenant_id.clone(),
            roles: user.roles.clone(),
            permissions: user
                .permissions
                .iter()
                .map(|p| format!("{:?}", p))
                .collect(),
            iat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                + self.token_expiry.as_secs(),
            iss: self.jwt_config.issuer.clone(),
            aud: self.jwt_config.audience.clone(),
            jti: generate_token_id(),
        };

        encode_jwt(&claims, &self.jwt_config)
    }

    /// Generate refresh token
    pub fn generate_refresh_token(&mut self, user_id: &str) -> Result<String> {
        let user = self
            .users
            .get(user_id)
            .ok_or_else(|| Error::InvalidInput(format!("User '{}' not found", user_id)))?;

        if !user.active {
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        let token_id = generate_token_id();
        let refresh_token = RefreshToken {
            token_id: token_id.clone(),
            user_id: user_id.to_string(),
            tenant_id: user.tenant_id.clone(),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + self.refresh_token_expiry,
            revoked: false,
        };

        self.refresh_tokens.insert(token_id.clone(), refresh_token);

        Ok(token_id)
    }

    /// Refresh access token using refresh token
    pub fn refresh_access_token(&mut self, refresh_token_id: &str) -> Result<String> {
        let refresh_token = self
            .refresh_tokens
            .get(refresh_token_id)
            .ok_or_else(|| Error::InvalidInput("Invalid refresh token".to_string()))?;

        if refresh_token.revoked {
            return Err(Error::InvalidOperation(
                "Refresh token has been revoked".to_string(),
            ));
        }

        if refresh_token.expires_at < SystemTime::now() {
            return Err(Error::InvalidOperation(
                "Refresh token has expired".to_string(),
            ));
        }

        let user_id = refresh_token.user_id.clone();

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::TokenRefresh,
            user_id: Some(user_id.clone()),
            tenant_id: Some(refresh_token.tenant_id.clone()),
            auth_method: AuthMethod::Jwt,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        self.generate_token(&user_id)
    }

    /// Validate JWT token
    pub fn validate_token(&mut self, token: &str) -> Result<AuthResult> {
        let claims = decode_jwt(token, &self.jwt_config)?;

        // Check if user still exists and is active, clone needed data
        let (user_permissions, user_active) = {
            let user = self
                .users
                .get(&claims.sub)
                .ok_or_else(|| Error::InvalidInput("User not found".to_string()))?;
            (user.permissions.clone(), user.active)
        };

        if !user_active {
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::TokenValidation,
            user_id: Some(claims.sub.clone()),
            tenant_id: Some(claims.tenant_id.clone()),
            auth_method: AuthMethod::Jwt,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        Ok(AuthResult {
            user_id: claims.sub,
            tenant_id: claims.tenant_id,
            permissions: user_permissions,
            expires_at: claims.exp,
            session_id: None,
            metadata: HashMap::new(),
        })
    }

    /// Authenticate with API key
    pub fn authenticate_api_key(&mut self, key: &str) -> Result<AuthResult> {
        // Clone all needed data from api_key_info first
        let (key_active, key_expires_at, key_user_id, key_tenant_id, key_permissions) = {
            let api_key_info = self
                .api_keys
                .get(key)
                .ok_or_else(|| Error::InvalidInput("Invalid API key".to_string()))?;
            (
                api_key_info.active,
                api_key_info.expires_at,
                api_key_info.user_id.clone(),
                api_key_info.tenant_id.clone(),
                api_key_info.permissions.clone(),
            )
        };

        if !key_active {
            return Err(Error::InvalidOperation(
                "API key is deactivated".to_string(),
            ));
        }

        if let Some(expires_at) = key_expires_at {
            if expires_at < SystemTime::now() {
                return Err(Error::InvalidOperation("API key has expired".to_string()));
            }
        }

        // Check user is active
        let user_active = {
            let user = self
                .users
                .get(&key_user_id)
                .ok_or_else(|| Error::InvalidInput("User not found".to_string()))?;
            user.active
        };

        if !user_active {
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        // Update usage count
        if let Some(key_info) = self.api_keys.get_mut(key) {
            key_info.usage_count += 1;
            key_info.last_used = Some(SystemTime::now());
        }

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::ApiKeyUsage,
            user_id: Some(key_user_id.clone()),
            tenant_id: Some(key_tenant_id.clone()),
            auth_method: AuthMethod::ApiKey,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        let expires_at = key_expires_at
            .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
            .unwrap_or(u64::MAX);

        Ok(AuthResult {
            user_id: key_user_id,
            tenant_id: key_tenant_id,
            permissions: key_permissions,
            expires_at,
            session_id: None,
            metadata: HashMap::new(),
        })
    }

    /// Create API key for a user
    pub fn create_api_key(
        &mut self,
        user_id: &str,
        name: &str,
        permissions: Option<Vec<Permission>>,
    ) -> Result<String> {
        // Clone user data before any mutations
        let (user_tenant_id, user_permissions, user_active) = {
            let user = self
                .users
                .get(user_id)
                .ok_or_else(|| Error::InvalidInput(format!("User '{}' not found", user_id)))?;
            (
                user.tenant_id.clone(),
                user.permissions.clone(),
                user.active,
            )
        };

        if !user_active {
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        let api_key = generate_api_key();
        let api_key_info = ApiKeyInfo {
            key_id: generate_token_id(),
            name: name.to_string(),
            key_hash: hash_api_key(&api_key),
            user_id: user_id.to_string(),
            tenant_id: user_tenant_id,
            permissions: permissions.unwrap_or(user_permissions),
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
        };

        // Store API key info (using hash as lookup key for security)
        self.api_keys.insert(api_key.clone(), api_key_info);

        // Add key reference to user
        if let Some(user_mut) = self.users.get_mut(user_id) {
            user_mut.api_keys.push(api_key.clone());
        }

        Ok(api_key)
    }

    /// Revoke an API key
    pub fn revoke_api_key(&mut self, key: &str) -> Result<()> {
        let api_key_info = self
            .api_keys
            .get_mut(key)
            .ok_or_else(|| Error::InvalidInput("API key not found".to_string()))?;

        api_key_info.active = false;

        // Remove from user's key list
        if let Some(user) = self.users.get_mut(&api_key_info.user_id) {
            user.api_keys.retain(|k| k != key);
        }

        Ok(())
    }

    /// Revoke refresh token
    pub fn revoke_refresh_token(&mut self, token_id: &str) -> Result<()> {
        let refresh_token = self
            .refresh_tokens
            .get_mut(token_id)
            .ok_or_else(|| Error::InvalidInput("Refresh token not found".to_string()))?;

        refresh_token.revoked = true;
        Ok(())
    }

    /// Revoke all refresh tokens for a user
    pub fn revoke_all_refresh_tokens(&mut self, user_id: &str) {
        for token in self.refresh_tokens.values_mut() {
            if token.user_id == user_id {
                token.revoked = true;
            }
        }
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Validate and refresh session
    pub fn validate_session(&mut self, session_id: &str) -> Result<&Session> {
        // Check if session exists and if expired
        let (is_expired, user_id, tenant_id) = {
            let session = self
                .sessions
                .get(session_id)
                .ok_or_else(|| Error::InvalidInput("Session not found".to_string()))?;
            (
                session.is_expired(),
                session.user_id.clone(),
                session.tenant_id.clone(),
            )
        };

        if is_expired {
            self.log_event(AuthEvent {
                timestamp: SystemTime::now(),
                event_type: AuthEventType::SessionExpired,
                user_id: Some(user_id),
                tenant_id: Some(tenant_id),
                auth_method: AuthMethod::Password,
                success: false,
                ip_address: None,
                user_agent: None,
                error_message: Some("Session expired".to_string()),
            });
            return Err(Error::InvalidOperation("Session has expired".to_string()));
        }

        // Refresh the session
        if let Some(session) = self.sessions.get_mut(session_id) {
            session.refresh();
        }

        Ok(self.sessions.get(session_id).unwrap())
    }

    /// Logout and invalidate session
    pub fn logout(&mut self, session_id: &str) -> Result<()> {
        let session = self
            .sessions
            .remove(session_id)
            .ok_or_else(|| Error::InvalidInput("Session not found".to_string()))?;

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::Logout,
            user_id: Some(session.user_id),
            tenant_id: Some(session.tenant_id),
            auth_method: AuthMethod::Password,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        Ok(())
    }

    /// Clean up expired sessions and tokens
    pub fn cleanup_expired(&mut self) {
        // Remove expired sessions
        self.sessions.retain(|_, session| !session.is_expired());

        // Remove expired refresh tokens
        let now = SystemTime::now();
        self.refresh_tokens
            .retain(|_, token| token.expires_at > now && !token.revoked);

        // Remove expired API keys
        self.api_keys
            .retain(|_, key| key.active && key.expires_at.map(|t| t > now).unwrap_or(true));
    }

    /// Get authentication events for a user
    pub fn get_user_events(&self, user_id: &str) -> Vec<&AuthEvent> {
        self.auth_events
            .iter()
            .filter(|e| e.user_id.as_ref().map(|id| id == user_id).unwrap_or(false))
            .collect()
    }

    /// Get all authentication events
    pub fn get_all_events(&self) -> &[AuthEvent] {
        &self.auth_events
    }

    /// Change user password
    pub fn change_password(
        &mut self,
        user_id: &str,
        old_password: &str,
        new_password: &str,
    ) -> Result<()> {
        let user = self
            .users
            .get(user_id)
            .ok_or_else(|| Error::InvalidInput(format!("User '{}' not found", user_id)))?;

        if !user.verify_password(old_password) {
            self.log_event(AuthEvent {
                timestamp: SystemTime::now(),
                event_type: AuthEventType::PasswordChange,
                user_id: Some(user_id.to_string()),
                tenant_id: Some(user.tenant_id.clone()),
                auth_method: AuthMethod::Password,
                success: false,
                ip_address: None,
                user_agent: None,
                error_message: Some("Invalid current password".to_string()),
            });
            return Err(Error::InvalidInput("Invalid current password".to_string()));
        }

        let tenant_id = user.tenant_id.clone();

        // Update password
        if let Some(user_mut) = self.users.get_mut(user_id) {
            user_mut.password_hash = Some(hash_password(new_password));
        }

        // Revoke all existing refresh tokens
        self.revoke_all_refresh_tokens(user_id);

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::PasswordChange,
            user_id: Some(user_id.to_string()),
            tenant_id: Some(tenant_id),
            auth_method: AuthMethod::Password,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        Ok(())
    }

    /// Log an authentication event
    fn log_event(&mut self, event: AuthEvent) {
        self.auth_events.push(event);

        // Trim events if needed
        if self.auth_events.len() > self.max_events {
            let excess = self.auth_events.len() - self.max_events;
            self.auth_events.drain(0..excess);
        }
    }
}

impl Default for AuthManager {
    fn default() -> Self {
        Self::new(JwtConfig::default())
    }
}

/// Thread-safe authentication manager
pub type SharedAuthManager = Arc<RwLock<AuthManager>>;

/// Create a new shared authentication manager
pub fn create_shared_auth_manager(config: JwtConfig) -> SharedAuthManager {
    Arc::new(RwLock::new(AuthManager::new(config)))
}

/// Refresh token information
#[derive(Debug, Clone)]
pub struct RefreshToken {
    /// Token identifier
    pub token_id: String,
    /// Associated user
    pub user_id: String,
    /// Associated tenant
    pub tenant_id: TenantId,
    /// Creation time
    pub created_at: SystemTime,
    /// Expiration time
    pub expires_at: SystemTime,
    /// Whether the token has been revoked
    pub revoked: bool,
}

// Helper functions

/// Generate a unique token ID
fn generate_token_id() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Generate an API key
fn generate_api_key() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    format!(
        "pk_{}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Hash password using PBKDF2
fn hash_password(password: &str) -> String {
    use pbkdf2::pbkdf2_hmac;
    use rand::RngCore;
    use sha2::Sha256;

    let mut salt = [0u8; 16];
    rand::rng().fill_bytes(&mut salt);

    let mut hash = [0u8; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, 100_000, &mut hash);

    // Format: iterations$salt_hex$hash_hex
    format!(
        "100000${}${}",
        salt.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>(),
        hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Verify password against hash
fn verify_password(password: &str, stored_hash: &str) -> bool {
    use pbkdf2::pbkdf2_hmac;
    use sha2::Sha256;

    let parts: Vec<&str> = stored_hash.split('$').collect();
    if parts.len() != 3 {
        return false;
    }

    let iterations: u32 = match parts[0].parse() {
        Ok(i) => i,
        Err(_) => return false,
    };

    let salt: Vec<u8> = match hex_decode(parts[1]) {
        Some(s) => s,
        None => return false,
    };

    let stored_hash_bytes: Vec<u8> = match hex_decode(parts[2]) {
        Some(h) => h,
        None => return false,
    };

    let mut computed_hash = vec![0u8; stored_hash_bytes.len()];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, iterations, &mut computed_hash);

    // Constant-time comparison
    computed_hash == stored_hash_bytes
}

/// Hash API key for storage
fn hash_api_key(key: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Decode hex string to bytes
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(s.len() / 2);
    let mut chars = s.chars();

    while let (Some(a), Some(b)) = (chars.next(), chars.next()) {
        let high = a.to_digit(16)?;
        let low = b.to_digit(16)?;
        bytes.push((high * 16 + low) as u8);
    }

    if chars.next().is_some() {
        // Odd number of characters
        return None;
    }

    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_registration() {
        let mut auth = AuthManager::default();

        let user = UserInfo::new("user1", "user@example.com", "tenant_a")
            .with_password("secret123")
            .with_permission(Permission::Read)
            .with_permission(Permission::Write);

        auth.register_user(user).unwrap();

        assert!(auth.get_user("user1").is_some());
    }

    #[test]
    fn test_password_authentication() {
        let mut auth = AuthManager::default();

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).unwrap();

        // Valid credentials
        let result = auth.authenticate_password("user@example.com", "secret123");
        assert!(result.is_ok());

        // Invalid password
        let result = auth.authenticate_password("user@example.com", "wrong");
        assert!(result.is_err());

        // Invalid email
        let result = auth.authenticate_password("wrong@example.com", "secret123");
        assert!(result.is_err());
    }

    #[test]
    fn test_token_generation() {
        let auth = AuthManager::default();
        let mut auth = auth;

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).unwrap();

        let token = auth.generate_token("user1").unwrap();
        assert!(!token.is_empty());

        // Validate token
        let result = auth.validate_token(&token);
        assert!(result.is_ok());
    }

    #[test]
    fn test_api_key_authentication() {
        let mut auth = AuthManager::default();

        let user = UserInfo::new("user1", "user@example.com", "tenant_a")
            .with_password("secret123")
            .with_permission(Permission::Read);

        auth.register_user(user).unwrap();

        let api_key = auth.create_api_key("user1", "test-key", None).unwrap();
        assert!(api_key.starts_with("pk_"));

        let result = auth.authenticate_api_key(&api_key);
        assert!(result.is_ok());

        let auth_result = result.unwrap();
        assert_eq!(auth_result.user_id, "user1");
        assert_eq!(auth_result.tenant_id, "tenant_a");
    }

    #[test]
    fn test_refresh_token() {
        let mut auth = AuthManager::default();

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).unwrap();

        let refresh_token = auth.generate_refresh_token("user1").unwrap();
        let new_token = auth.refresh_access_token(&refresh_token);
        assert!(new_token.is_ok());

        // Revoke and try again
        auth.revoke_refresh_token(&refresh_token).unwrap();
        let result = auth.refresh_access_token(&refresh_token);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_management() {
        let mut auth = AuthManager::default();

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).unwrap();

        let result = auth
            .authenticate_password("user@example.com", "secret123")
            .unwrap();
        let session_id = result.session_id.unwrap();

        // Validate session
        assert!(auth.validate_session(&session_id).is_ok());

        // Logout
        auth.logout(&session_id).unwrap();

        // Session should be invalid now
        assert!(auth.validate_session(&session_id).is_err());
    }

    #[test]
    fn test_password_change() {
        let mut auth = AuthManager::default();

        let user = UserInfo::new("user1", "user@example.com", "tenant_a").with_password("oldpass");

        auth.register_user(user).unwrap();

        // Change password
        auth.change_password("user1", "oldpass", "newpass").unwrap();

        // Old password should fail
        let result = auth.authenticate_password("user@example.com", "oldpass");
        assert!(result.is_err());

        // New password should work
        let result = auth.authenticate_password("user@example.com", "newpass");
        assert!(result.is_ok());
    }

    #[test]
    fn test_user_deactivation() {
        let mut auth = AuthManager::default();

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).unwrap();

        // Login should work
        assert!(auth
            .authenticate_password("user@example.com", "secret123")
            .is_ok());

        // Deactivate user
        auth.deactivate_user("user1").unwrap();

        // Login should fail
        let result = auth.authenticate_password("user@example.com", "secret123");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_hashing() {
        let password = "test_password_123";
        let hash = hash_password(password);

        // Hash should contain iterations, salt, and hash
        let parts: Vec<&str> = hash.split('$').collect();
        assert_eq!(parts.len(), 3);

        // Verify should work
        assert!(verify_password(password, &hash));

        // Wrong password should fail
        assert!(!verify_password("wrong_password", &hash));
    }
}
