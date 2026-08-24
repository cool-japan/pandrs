//! Enterprise Authentication Module
//!
//! This module provides enterprise-grade authentication features including:
//! - JWT (JSON Web Token) token generation and validation
//! - OAuth 2.0 support (Authorization Code and Client Credentials flows)
//! - API Key authentication
//! - Session management
//! - ReBAC (Relationship-Based Access Control)
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
//!
//! # ReBAC Example
//!
//! ```ignore
//! use pandrs::auth::rebac::RebacManager;
//!
//! let rebac = RebacManager::new();
//!
//! // Grant permission
//! rebac.grant("user:alice", "owner", "document:123")?;
//!
//! // Check permission
//! let can_access = rebac.check_access("user:alice", "owner", "document:123")?;
//! ```

pub mod api_key;
pub mod jwt;
pub mod oauth;
pub mod rebac;
pub mod session;

pub use api_key::*;
pub use jwt::*;
pub use oauth::*;
pub use rebac::*;
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
#[derive(Clone)]
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
    /// Hashes of API keys associated with this user (never plaintext).
    pub api_keys: Vec<String>,
}

impl std::fmt::Debug for UserInfo {
    /// Redacting `Debug`: never print the password hash or API-key hashes.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserInfo")
            .field("user_id", &self.user_id)
            .field("tenant_id", &self.tenant_id)
            .field("email", &self.email)
            .field("display_name", &self.display_name)
            .field("roles", &self.roles)
            .field("permissions", &self.permissions)
            .field("active", &self.active)
            .field(
                "password_hash",
                &self.password_hash.as_ref().map(|_| "<redacted>"),
            )
            .field("api_keys", &self.api_keys.len())
            .finish_non_exhaustive()
    }
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

/// Maximum consecutive failed password attempts (per email) before a temporary
/// lockout kicks in.
const MAX_FAILED_ATTEMPTS: u32 = 5;
/// Duration of the lockout / sliding failure window.
const LOCKOUT_WINDOW: Duration = Duration::from_secs(900); // 15 minutes

/// Central authentication manager
pub struct AuthManager {
    /// JWT configuration
    jwt_config: JwtConfig,
    /// OAuth configuration
    oauth_config: Option<OAuthConfig>,
    /// Registered users
    users: HashMap<String, UserInfo>,
    /// Active sessions
    sessions: HashMap<String, Session>,
    /// API keys — delegated to the single hash-keyed [`ApiKeyManager`] so keys
    /// are never stored in plaintext and there is one source of truth.
    api_key_manager: ApiKeyManager,
    /// Refresh tokens
    refresh_tokens: HashMap<String, RefreshToken>,
    /// Per-user access-token validity floor (unix seconds). Tokens issued at or
    /// before this instant are rejected — this is how a password change kills
    /// already-issued access tokens.
    tokens_valid_after: HashMap<String, u64>,
    /// Failed password attempts per email: (count, window start).
    failed_attempts: HashMap<String, (u32, std::time::Instant)>,
    /// Authentication event log
    auth_events: Vec<AuthEvent>,
    /// Optional shared security audit sink.
    audit: Option<crate::audit::SharedAuditLogger>,
    /// Maximum events to keep
    max_events: usize,
    /// Token expiration duration
    token_expiry: Duration,
    /// Refresh token expiration duration
    refresh_token_expiry: Duration,
    /// Session timeout
    session_timeout: Duration,
}

impl std::fmt::Debug for AuthManager {
    /// Redacting `Debug`: never print the JWT signing key, password hashes,
    /// API-key material, or refresh-token secrets — only shapes/counts.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthManager")
            .field("issuer", &self.jwt_config.issuer)
            .field("oauth_configured", &self.oauth_config.is_some())
            .field("users", &self.users.len())
            .field("sessions", &self.sessions.len())
            .field("api_keys", &self.api_key_manager.key_count())
            .field("refresh_tokens", &self.refresh_tokens.len())
            .field("auth_events", &self.auth_events.len())
            .field("secret_key", &"<redacted>")
            .finish()
    }
}

impl AuthManager {
    /// Create a new authentication manager
    pub fn new(jwt_config: JwtConfig) -> Self {
        AuthManager {
            jwt_config,
            oauth_config: None,
            users: HashMap::new(),
            sessions: HashMap::new(),
            api_key_manager: ApiKeyManager::new(),
            refresh_tokens: HashMap::new(),
            tokens_valid_after: HashMap::new(),
            failed_attempts: HashMap::new(),
            auth_events: Vec::new(),
            audit: None,
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

    /// Attach a shared security audit logger. Authentication failures and
    /// permission denials will be emitted as `Security` audit entries.
    pub fn with_audit_logger(mut self, logger: crate::audit::SharedAuditLogger) -> Self {
        self.audit = Some(logger);
        self
    }

    /// Emit a `Security` audit entry (best-effort; never blocks auth).
    fn audit_security(&self, operation: &str, user: Option<&str>, success: bool, message: &str) {
        if let Some(ref logger) = self.audit {
            let level = if success {
                crate::audit::LogLevel::Info
            } else {
                crate::audit::LogLevel::Warn
            };
            let mut entry = crate::audit::AuditEntry::new(
                level,
                crate::audit::EventCategory::Security,
                operation,
                "auth",
                message,
            );
            if let Some(u) = user {
                entry = entry.with_user(u);
            }
            if !success {
                entry = entry.with_error(message);
            }
            logger.log(entry);
        }
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

    /// Whether an email is currently locked out from password authentication.
    fn is_locked_out(&self, email: &str) -> bool {
        matches!(
            self.failed_attempts.get(email),
            Some((count, start)) if *count >= MAX_FAILED_ATTEMPTS && start.elapsed() < LOCKOUT_WINDOW
        )
    }

    /// Record a failed password attempt for an email, rolling the window.
    fn record_failure(&mut self, email: &str) {
        let now = std::time::Instant::now();
        let entry = self
            .failed_attempts
            .entry(email.to_string())
            .or_insert((0, now));
        if entry.1.elapsed() >= LOCKOUT_WINDOW {
            *entry = (1, now);
        } else {
            entry.0 = entry.0.saturating_add(1);
        }
    }

    /// Clear failed-attempt state for an email after a success.
    fn clear_failures(&mut self, email: &str) {
        self.failed_attempts.remove(email);
    }

    /// Authenticate with password and get JWT token.
    ///
    /// Hardened against user enumeration: an unknown email performs the same
    /// PBKDF2 work as a real verification (via `dummy_verify`) and returns
    /// the identical generic error. Repeated failures trip a per-email lockout,
    /// and every failure is logged (and audited when a logger is attached).
    pub fn authenticate_password(&mut self, email: &str, password: &str) -> Result<AuthResult> {
        if self.is_locked_out(email) {
            self.audit_security("auth.login", None, false, "account temporarily locked");
            self.log_login_failure(None, None, "Account temporarily locked");
            return Err(Error::InvalidOperation(
                "Too many failed attempts; try again later".to_string(),
            ));
        }

        // Clone the matched user so the immutable borrow of `self.users` ends
        // before we take mutable borrows (failure counters, logging).
        let user = self.users.values().find(|u| u.email == email).cloned();

        let user = match user {
            Some(u) => u,
            None => {
                // Equalize timing with the real path, then fail generically.
                dummy_verify(password);
                self.record_failure(email);
                self.audit_security("auth.login", None, false, "unknown email");
                self.log_login_failure(None, None, "Invalid credentials");
                return Err(Error::InvalidInput("Invalid credentials".to_string()));
            }
        };

        if !user.active {
            self.record_failure(email);
            self.audit_security(
                "auth.login",
                Some(&user.user_id),
                false,
                "account deactivated",
            );
            self.log_login_failure(
                Some(user.user_id.clone()),
                Some(user.tenant_id.clone()),
                "User account is deactivated",
            );
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        if !user.verify_password(password) {
            self.record_failure(email);
            self.audit_security("auth.login", Some(&user.user_id), false, "invalid password");
            self.log_login_failure(
                Some(user.user_id.clone()),
                Some(user.tenant_id.clone()),
                "Invalid password",
            );
            return Err(Error::InvalidInput("Invalid credentials".to_string()));
        }

        // Success: clear failure state.
        self.clear_failures(email);

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

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let claims = TokenClaims {
            sub: user_id.to_string(),
            tenant_id: user.tenant_id.clone(),
            roles: user.roles.clone(),
            permissions: user
                .permissions
                .iter()
                .map(|p| format!("{:?}", p))
                .collect(),
            iat: now,
            nbf: Some(now),
            exp: now.saturating_add(self.token_expiry.as_secs()),
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

        // Reject tokens issued before the user's validity floor (set on
        // password change / explicit revocation) — this is how a password
        // change kills already-issued access tokens.
        if let Some(&valid_after) = self.tokens_valid_after.get(&claims.sub) {
            if claims.iat < valid_after {
                self.audit_security(
                    "auth.token",
                    Some(&claims.sub),
                    false,
                    "token issued before validity floor",
                );
                return Err(Error::InvalidOperation(
                    "Token has been invalidated".to_string(),
                ));
            }
        }

        // Check if user still exists and is active; take tenant_id from the
        // authoritative user record rather than trusting the (attacker-mintable
        // if the key ever leaked) token claim.
        let (user_permissions, user_active, user_tenant_id) = {
            let user = self
                .users
                .get(&claims.sub)
                .ok_or_else(|| Error::InvalidInput("User not found".to_string()))?;
            (
                user.permissions.clone(),
                user.active,
                user.tenant_id.clone(),
            )
        };

        if !user_active {
            self.audit_security(
                "auth.token",
                Some(&claims.sub),
                false,
                "account deactivated",
            );
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
        }

        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::TokenValidation,
            user_id: Some(claims.sub.clone()),
            tenant_id: Some(user_tenant_id.clone()),
            auth_method: AuthMethod::Jwt,
            success: true,
            ip_address: None,
            user_agent: None,
            error_message: None,
        });

        Ok(AuthResult {
            user_id: claims.sub,
            tenant_id: user_tenant_id,
            permissions: user_permissions,
            expires_at: claims.exp,
            session_id: None,
            metadata: HashMap::new(),
        })
    }

    /// Authenticate with API key.
    ///
    /// Delegates storage/validation to the hash-keyed [`ApiKeyManager`]: the
    /// presented key is hashed before lookup, so plaintext keys are never held
    /// in memory or used as map keys. `validate_and_use` enforces
    /// active/expiry/IP/rate-limit and records usage in one place.
    pub fn authenticate_api_key(&mut self, key: &str) -> Result<AuthResult> {
        let (key_expires_at, key_user_id, key_tenant_id, key_permissions) = {
            let api_key_info = match self.api_key_manager.validate_and_use(key, None) {
                Ok(info) => info,
                Err(e) => {
                    self.audit_security("auth.api_key", None, false, "invalid API key");
                    return Err(e);
                }
            };
            (
                api_key_info.expires_at,
                api_key_info.user_id.clone(),
                api_key_info.tenant_id.clone(),
                api_key_info.permissions.clone(),
            )
        };

        // Check user is active
        let user_active = {
            let user = self
                .users
                .get(&key_user_id)
                .ok_or_else(|| Error::InvalidInput("User not found".to_string()))?;
            user.active
        };

        if !user_active {
            self.audit_security(
                "auth.api_key",
                Some(&key_user_id),
                false,
                "account deactivated",
            );
            return Err(Error::InvalidOperation(
                "User account is deactivated".to_string(),
            ));
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

        let perms = permissions.unwrap_or(user_permissions);
        // The manager generates the key, hashes it, and stores only the hash.
        // The plaintext is returned here and nowhere retained.
        let api_key = self
            .api_key_manager
            .generate_key(name, user_id, &user_tenant_id, perms)?;

        // Track the key on the user by its HASH, never the plaintext.
        if let Some(user_mut) = self.users.get_mut(user_id) {
            user_mut.api_keys.push(hash_api_key(&api_key));
        }

        Ok(api_key)
    }

    /// Revoke an API key (by its plaintext value; hashed before lookup).
    pub fn revoke_api_key(&mut self, key: &str) -> Result<()> {
        self.api_key_manager.revoke_by_key(key)?;

        // Remove the hash reference from whichever user holds it.
        let key_hash = hash_api_key(key);
        for user in self.users.values_mut() {
            user.api_keys.retain(|k| k != &key_hash);
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

        self.sessions
            .get(session_id)
            .ok_or_else(|| Error::InvalidInput("Session not found".to_string()))
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

        // Remove expired API keys via the manager.
        self.api_key_manager.cleanup_expired();
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

        // Raise the access-token validity floor so tokens minted before this
        // password change are rejected on their next validation. `+1` ensures
        // tokens issued in the same second are also invalidated.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.tokens_valid_after
            .insert(user_id.to_string(), now.saturating_add(1));
        self.audit_security(
            "auth.password_change",
            Some(user_id),
            true,
            "password changed",
        );

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

    /// Convenience: record a failed `Login` authentication event.
    fn log_login_failure(
        &mut self,
        user_id: Option<String>,
        tenant_id: Option<TenantId>,
        message: &str,
    ) {
        self.log_event(AuthEvent {
            timestamp: SystemTime::now(),
            event_type: AuthEventType::Login,
            user_id,
            tenant_id,
            auth_method: AuthMethod::Password,
            success: false,
            ip_address: None,
            user_agent: None,
            error_message: Some(message.to_string()),
        });
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
    use scirs2_core::random::Rng;
    let mut bytes = [0u8; 32];
    scirs2_core::random::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Password-hashing parameters. The iteration count is tagged into the stored
/// hash so it can evolve, but is clamped on verification (see below).
const PWHASH_ALGORITHM: &str = "pbkdf2-sha256";
const PWHASH_VERSION: &str = "v1";
const PWHASH_ITERATIONS: u32 = 100_000;
/// Accepted iteration bounds when verifying a stored hash. An attacker who can
/// influence the stored artifact must not be able to downgrade the work factor
/// to 1 (instant crack) or inflate it to `u32::MAX` (verification DoS).
const PWHASH_MIN_ITERATIONS: u32 = 100_000;
const PWHASH_MAX_ITERATIONS: u32 = 1_000_000;

/// Constant-time byte-slice equality.
///
/// Pure-Rust manual implementation (the `subtle` crate would be the idiomatic
/// choice, but adding a dependency is outside this change's file ownership; a
/// volatile xor-fold is the sanctioned fallback and is itself pure Rust). The
/// `black_box` on the accumulator prevents the optimizer from introducing an
/// early-out. Length is compared up front — that only leaks the length, which
/// for fixed-size MACs/hashes is not secret.
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= *x ^ *y;
    }
    std::hint::black_box(diff) == 0
}

/// Hash password using PBKDF2-HMAC-SHA256, tagging the algorithm, version, and
/// iteration count into the stored string.
fn hash_password(password: &str) -> String {
    use pbkdf2::pbkdf2_hmac;
    use scirs2_core::random::Rng;
    use sha2::Sha256;

    let mut salt = [0u8; 16];
    scirs2_core::random::rng().fill_bytes(&mut salt);

    let mut hash = [0u8; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, PWHASH_ITERATIONS, &mut hash);

    // Format: algorithm$version$iterations$salt_hex$hash_hex
    format!(
        "{}${}${}${}${}",
        PWHASH_ALGORITHM,
        PWHASH_VERSION,
        PWHASH_ITERATIONS,
        salt.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>(),
        hash.iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Verify password against a stored hash.
///
/// Rejects any hash whose tagged iteration count falls outside
/// `[PWHASH_MIN_ITERATIONS, PWHASH_MAX_ITERATIONS]` rather than clamping and
/// recomputing (which would silently derive a different key and muddy the
/// failure). Comparison of the derived digest is constant-time.
fn verify_password(password: &str, stored_hash: &str) -> bool {
    use pbkdf2::pbkdf2_hmac;
    use sha2::Sha256;

    let parts: Vec<&str> = stored_hash.split('$').collect();
    if parts.len() != 5 {
        return false;
    }
    if parts[0] != PWHASH_ALGORITHM {
        return false;
    }
    // parts[1] is the version tag; only v1 is defined today.
    if parts[1] != PWHASH_VERSION {
        return false;
    }

    let iterations: u32 = match parts[2].parse() {
        Ok(i) => i,
        Err(_) => return false,
    };
    if !(PWHASH_MIN_ITERATIONS..=PWHASH_MAX_ITERATIONS).contains(&iterations) {
        return false;
    }

    let salt: Vec<u8> = match hex_decode(parts[3]) {
        Some(s) => s,
        None => return false,
    };

    let stored_hash_bytes: Vec<u8> = match hex_decode(parts[4]) {
        Some(h) => h,
        None => return false,
    };

    let mut computed_hash = vec![0u8; stored_hash_bytes.len()];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, iterations, &mut computed_hash);

    constant_time_eq(&computed_hash, &stored_hash_bytes)
}

/// Run a PBKDF2 derivation against a fixed dummy hash to consume time
/// comparable to a real verification. Used to close the user-enumeration timing
/// oracle: authenticating an unknown email must take roughly as long as
/// authenticating a known one with a wrong password.
fn dummy_verify(password: &str) {
    use pbkdf2::pbkdf2_hmac;
    use sha2::Sha256;
    let salt = [0u8; 16];
    let mut out = [0u8; 32];
    pbkdf2_hmac::<Sha256>(password.as_bytes(), &salt, PWHASH_ITERATIONS, &mut out);
    let _ = std::hint::black_box(out);
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

        auth.register_user(user).expect("operation should succeed");

        assert!(auth.get_user("user1").is_some());
    }

    #[test]
    fn test_password_authentication() {
        let mut auth = AuthManager::default();

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).expect("operation should succeed");

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

        auth.register_user(user).expect("operation should succeed");

        let token = auth
            .generate_token("user1")
            .expect("operation should succeed");
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

        auth.register_user(user).expect("operation should succeed");

        let api_key = auth
            .create_api_key("user1", "test-key", None)
            .expect("operation should succeed");
        assert!(api_key.starts_with("pk_"));

        let result = auth.authenticate_api_key(&api_key);
        assert!(result.is_ok());

        let auth_result = result.expect("operation should succeed");
        assert_eq!(auth_result.user_id, "user1");
        assert_eq!(auth_result.tenant_id, "tenant_a");
    }

    #[test]
    fn test_refresh_token() {
        let mut auth = AuthManager::default();

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).expect("operation should succeed");

        let refresh_token = auth
            .generate_refresh_token("user1")
            .expect("operation should succeed");
        let new_token = auth.refresh_access_token(&refresh_token);
        assert!(new_token.is_ok());

        // Revoke and try again
        auth.revoke_refresh_token(&refresh_token)
            .expect("operation should succeed");
        let result = auth.refresh_access_token(&refresh_token);
        assert!(result.is_err());
    }

    #[test]
    fn test_session_management() {
        let mut auth = AuthManager::default();

        let user =
            UserInfo::new("user1", "user@example.com", "tenant_a").with_password("secret123");

        auth.register_user(user).expect("operation should succeed");

        let result = auth
            .authenticate_password("user@example.com", "secret123")
            .expect("operation should succeed");
        let session_id = result.session_id.expect("operation should succeed");

        // Validate session
        assert!(auth.validate_session(&session_id).is_ok());

        // Logout
        auth.logout(&session_id).expect("operation should succeed");

        // Session should be invalid now
        assert!(auth.validate_session(&session_id).is_err());
    }

    #[test]
    fn test_password_change() {
        let mut auth = AuthManager::default();

        let user = UserInfo::new("user1", "user@example.com", "tenant_a").with_password("oldpass");

        auth.register_user(user).expect("operation should succeed");

        // Change password
        auth.change_password("user1", "oldpass", "newpass")
            .expect("operation should succeed");

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

        auth.register_user(user).expect("operation should succeed");

        // Login should work
        assert!(auth
            .authenticate_password("user@example.com", "secret123")
            .is_ok());

        // Deactivate user
        auth.deactivate_user("user1")
            .expect("operation should succeed");

        // Login should fail
        let result = auth.authenticate_password("user@example.com", "secret123");
        assert!(result.is_err());
    }

    #[test]
    fn test_password_hashing() {
        let password = "test_password_123";
        let hash = hash_password(password);

        // Hash should be algorithm$version$iterations$salt$hash
        let parts: Vec<&str> = hash.split('$').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0], "pbkdf2-sha256");

        // Verify should work
        assert!(verify_password(password, &hash));

        // Wrong password should fail
        assert!(!verify_password("wrong_password", &hash));
    }
}
