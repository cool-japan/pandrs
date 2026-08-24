#![allow(clippy::result_large_err)]
//! # Comprehensive JWT and OAuth2 Authentication Example
//!
//! This example demonstrates enterprise-grade authentication patterns using PandRS's
//! authentication module, including:
//!
//! - JWT (JSON Web Token) generation, validation, and refresh
//! - OAuth2 Authorization Code Flow with PKCE
//! - OAuth2 Client Credentials Flow
//! - API Key authentication and management
//! - Session management with automatic timeout
//! - Rate limiting and security controls
//! - Multi-tenant authentication scenarios
//!
//! # Use Case: Multi-Tenant SaaS Analytics Platform
//!
//! This example simulates a multi-tenant SaaS platform where different organizations
//! (tenants) need secure, isolated access to their analytics data. The platform supports:
//! - User authentication via JWT tokens
//! - Service-to-service communication via OAuth2 client credentials
//! - API keys for programmatic access
//! - Per-tenant data isolation and access control

use pandrs::auth::{
    AuthManager, JwtConfig, OAuthClient, OAuthClientInfo, OAuthConfig, OAuthGrantType, UserInfo,
};
use pandrs::error::Result;
use pandrs::multitenancy::Permission;
use std::time::Duration;

fn main() -> Result<()> {
    println!("🔐 PandRS Security: JWT & OAuth2 Authentication Example");
    println!("========================================================\n");

    // Scenario 1: JWT Token Authentication
    println!("📝 Scenario 1: JWT Token Authentication");
    jwt_authentication_flow()?;

    // Scenario 2: OAuth2 Authorization Code Flow with PKCE
    println!("\n🔑 Scenario 2: OAuth2 Authorization Code Flow with PKCE");
    oauth_authorization_code_flow()?;

    // Scenario 3: OAuth2 Client Credentials (Service-to-Service)
    println!("\n🤖 Scenario 3: OAuth2 Client Credentials (Service-to-Service)");
    oauth_client_credentials_flow()?;

    // Scenario 4: API Key Authentication
    println!("\n🎫 Scenario 4: API Key Authentication");
    api_key_authentication_flow()?;

    // Scenario 5: Session Management
    println!("\n⏰ Scenario 5: Session Management");
    session_management_flow()?;

    // Scenario 6: Token Refresh Mechanism
    println!("\n🔄 Scenario 6: Token Refresh Mechanism");
    token_refresh_flow()?;

    // Scenario 7: Multi-Tenant Authentication
    println!("\n🏢 Scenario 7: Multi-Tenant Authentication");
    multi_tenant_authentication()?;

    // Scenario 8: Security Audit Trail
    println!("\n📊 Scenario 8: Security Audit Trail");
    security_audit_trail()?;

    // Scenario 9: Rate Limiting and Security Controls
    println!("\n🚦 Scenario 9: Rate Limiting and Security Controls");
    rate_limiting_controls()?;

    println!("\n✅ All authentication scenarios completed successfully!");
    Ok(())
}

/// Demonstrates JWT token-based authentication flow
///
/// This scenario shows:
/// - User registration with password
/// - Password-based authentication
/// - JWT token generation
/// - Token validation
/// - Token expiration handling
fn jwt_authentication_flow() -> Result<()> {
    // Create JWT configuration with custom settings
    let jwt_config = JwtConfig::new(b"super_secret_key_for_production_use_random_256_bits")
        .with_issuer("pandrs-analytics-platform")
        .with_audience("analytics-api")
        .with_expiration(3600); // 1 hour

    let mut auth_manager = AuthManager::new(jwt_config);

    // Register a user for Tenant A
    println!("  👤 Registering user 'alice@company-a.com' for tenant 'company_a'");
    let user = UserInfo::new("alice", "alice@company-a.com", "company_a")
        .with_display_name("Alice Johnson")
        .with_password("secure_password_123")
        .with_role("analyst")
        .with_permission(Permission::Read)
        .with_permission(Permission::Write);

    auth_manager.register_user(user)?;

    // Authenticate with password
    println!("  🔐 Authenticating with email and password...");
    let auth_result =
        auth_manager.authenticate_password("alice@company-a.com", "secure_password_123")?;
    println!("  ✅ Authentication successful!");
    println!("     User ID: {}", auth_result.user_id);
    println!("     Tenant: {}", auth_result.tenant_id);
    println!("     Permissions: {:?}", auth_result.permissions);
    println!("     Session ID: {:?}", auth_result.session_id);

    // Generate JWT token
    println!("\n  🎟️  Generating JWT token...");
    let token = auth_manager.generate_token("alice")?;
    println!("  ✅ JWT token generated (length: {} chars)", token.len());
    println!("     Token preview: {}...", &token[..50]);

    // Validate the token
    println!("\n  ✔️  Validating JWT token...");
    let validated = auth_manager.validate_token(&token)?;
    println!("  ✅ Token is valid!");
    println!("     Subject: {}", validated.user_id);
    println!("     Expires at: {}", validated.expires_at);

    // Test invalid credentials
    println!("\n  ❌ Testing invalid credentials...");
    match auth_manager.authenticate_password("alice@company-a.com", "wrong_password") {
        Ok(_) => println!("  ⚠️  Unexpected success with wrong password!"),
        Err(_) => println!("  ✅ Correctly rejected invalid password"),
    }

    Ok(())
}

/// Demonstrates OAuth2 Authorization Code Flow with PKCE
///
/// This is the recommended flow for web applications and mobile apps.
/// PKCE (Proof Key for Code Exchange) provides additional security against
/// authorization code interception attacks.
fn oauth_authorization_code_flow() -> Result<()> {
    // Setup OAuth2 configuration
    let oauth_config = OAuthConfig::new(
        "https://auth.pandrs-analytics.com",
        "pandrs_web_client",
        "client_secret_abc123",
    )
    .with_redirect_uri("https://app.pandrs-analytics.com/callback")
    .with_scope("openid")
    .with_scope("profile")
    .with_scope("analytics:read");

    let mut oauth_client = OAuthClient::new(oauth_config);

    // Register an OAuth client for tenant
    println!("  📋 Registering OAuth client for tenant 'company_a'");
    let client_info = OAuthClientInfo {
        client_id: "pandrs_web_client".to_string(),
        client_secret_hash: hash_secret("client_secret_abc123"),
        redirect_uris: vec!["https://app.pandrs-analytics.com/callback".to_string()],
        grant_types: vec![OAuthGrantType::AuthorizationCode],
        scopes: vec![
            "openid".to_string(),
            "profile".to_string(),
            "analytics:read".to_string(),
        ],
        tenant_id: "company_a".to_string(),
    };
    oauth_client.register_client(client_info);

    // Step 1: Generate authorization code (simulating user consent)
    println!("  🔗 Step 1: User authorizes application...");
    let auth_code = oauth_client.create_authorization_code(
        "pandrs_web_client",
        "https://app.pandrs-analytics.com/callback",
        vec!["openid".to_string(), "profile".to_string()],
        "alice",
        None, // PKCE challenge (simplified for example)
        None,
    )?;
    println!("  ✅ Authorization code generated: {}...", &auth_code[..16]);

    // Step 2: Exchange authorization code for access token
    println!("\n  🔄 Step 2: Exchanging authorization code for access token...");
    let token_response = oauth_client.exchange_code(
        "pandrs_web_client",
        "client_secret_abc123",
        &auth_code,
        "https://app.pandrs-analytics.com/callback",
        None,
    )?;
    println!("  ✅ Access token received!");
    println!("     Token type: {}", token_response.token_type);
    println!("     Expires in: {} seconds", token_response.expires_in);
    println!("     Scopes: {:?}", token_response.scope);
    println!(
        "     Has refresh token: {}",
        token_response.refresh_token.is_some()
    );

    // Step 3: Verify token is active
    println!("\n  ✔️  Step 3: Introspecting access token...");
    let introspection = oauth_client.introspect_token(&token_response.access_token);
    println!("  ✅ Token introspection result:");
    println!("     Active: {}", introspection.active);
    println!("     Client ID: {:?}", introspection.client_id);
    println!("     Username: {:?}", introspection.username);

    Ok(())
}

/// Demonstrates OAuth2 Client Credentials Flow
///
/// This flow is used for service-to-service authentication where no user
/// interaction is required. Ideal for backend services, scheduled jobs, etc.
fn oauth_client_credentials_flow() -> Result<()> {
    let oauth_config = OAuthConfig::new(
        "https://auth.pandrs-analytics.com",
        "analytics_service",
        "service_secret_xyz789",
    );

    let mut oauth_client = OAuthClient::new(oauth_config);

    // Register service client
    println!("  🤖 Registering service client 'analytics_service'");
    let service_client = OAuthClientInfo {
        client_id: "analytics_service".to_string(),
        client_secret_hash: hash_secret("service_secret_xyz789"),
        redirect_uris: vec![],
        grant_types: vec![OAuthGrantType::ClientCredentials],
        scopes: vec![
            "analytics:read".to_string(),
            "analytics:write".to_string(),
            "reports:generate".to_string(),
        ],
        tenant_id: "company_a".to_string(),
    };
    oauth_client.register_client(service_client);

    // Request token using client credentials
    println!("  🎫 Requesting access token with client credentials...");
    let token_response = oauth_client.client_credentials_grant(
        "analytics_service",
        "service_secret_xyz789",
        Some(vec![
            "analytics:read".to_string(),
            "reports:generate".to_string(),
        ]),
    )?;

    println!("  ✅ Service token obtained!");
    println!(
        "     Access token: {}...",
        &token_response.access_token[..20]
    );
    println!("     Token type: {}", token_response.token_type);
    println!("     Expires in: {} seconds", token_response.expires_in);
    println!("     Granted scopes: {:?}", token_response.scope);
    println!(
        "     Has refresh token: {} (not issued for client credentials)",
        token_response.refresh_token.is_some()
    );

    // Verify the service can access resources
    println!("\n  ✔️  Verifying service token...");
    let introspection = oauth_client.introspect_token(&token_response.access_token);
    if introspection.active {
        println!("  ✅ Service token is active and valid");
        println!("     Client: {:?}", introspection.client_id);
        println!("     Scopes: {:?}", introspection.scope);
    }

    Ok(())
}

/// Demonstrates API Key authentication
///
/// API keys are useful for:
/// - Programmatic access (scripts, CLI tools)
/// - Third-party integrations
/// - Long-lived credentials
/// - Rate limiting per key
fn api_key_authentication_flow() -> Result<()> {
    let mut auth_manager = AuthManager::new(JwtConfig::default());

    // Register user
    let user = UserInfo::new("bob", "bob@company-b.com", "company_b")
        .with_password("bob_password")
        .with_permission(Permission::Read)
        .with_permission(Permission::Write);
    auth_manager.register_user(user)?;

    // Create API key for user
    println!("  🔑 Creating API key for user 'bob'...");
    let api_key = auth_manager.create_api_key(
        "bob",
        "Production API Key",
        Some(vec![Permission::Read, Permission::Write]),
    )?;
    println!("  ✅ API key created: {}...", &api_key[..20]);
    println!("     ⚠️  Store this securely - it won't be shown again!");

    // Authenticate using API key
    println!("\n  🔐 Authenticating with API key...");
    let auth_result = auth_manager.authenticate_api_key(&api_key)?;
    println!("  ✅ API key authentication successful!");
    println!("     User: {}", auth_result.user_id);
    println!("     Tenant: {}", auth_result.tenant_id);
    println!("     Permissions: {:?}", auth_result.permissions);

    // Test API key usage tracking
    println!("\n  📊 API key usage tracking:");
    for i in 1..=3 {
        match auth_manager.authenticate_api_key(&api_key) {
            Ok(result) => {
                println!(
                    "     Request {}: ✅ Authenticated as '{}'",
                    i, result.user_id
                );
            }
            Err(e) => println!("     Request {}: ❌ Failed: {}", i, e),
        }
    }

    // Revoke API key
    println!("\n  🚫 Revoking API key...");
    auth_manager.revoke_api_key(&api_key)?;
    println!("  ✅ API key revoked");

    // Verify revoked key is rejected
    println!("\n  ✔️  Testing revoked key...");
    match auth_manager.authenticate_api_key(&api_key) {
        Ok(_) => println!("  ⚠️  Unexpected success with revoked key!"),
        Err(_) => println!("  ✅ Revoked key correctly rejected"),
    }

    Ok(())
}

/// Demonstrates session management
///
/// Sessions provide:
/// - Stateful user sessions
/// - Automatic timeout
/// - Session attributes/metadata
/// - Concurrent session control
fn session_management_flow() -> Result<()> {
    let mut auth_manager =
        AuthManager::new(JwtConfig::default()).with_session_timeout(Duration::from_secs(300)); // 5 minutes

    // Register user
    let user = UserInfo::new("carol", "carol@company-c.com", "company_c")
        .with_password("carol_password")
        .with_permission(Permission::Read);
    auth_manager.register_user(user)?;

    // Create session through authentication
    println!("  🔐 Creating session for user 'carol'...");
    let auth_result =
        auth_manager.authenticate_password("carol@company-c.com", "carol_password")?;
    let session_id = auth_result
        .session_id
        .ok_or_else(|| pandrs::error::Error::InvalidOperation("No session created".to_string()))?;
    println!("  ✅ Session created: {}", session_id);

    // Validate session
    println!("\n  ✔️  Validating session...");
    let session = auth_manager.validate_session(&session_id)?;
    println!("  ✅ Session is valid");
    println!("     User: {}", session.user_id);
    println!("     Created: {:?}", session.created_at);

    // Simulate session usage
    println!("\n  📝 Simulating API requests with session...");
    for i in 1..=3 {
        match auth_manager.validate_session(&session_id) {
            Ok(s) => println!("     Request {}: ✅ Session valid (user: {})", i, s.user_id),
            Err(e) => println!("     Request {}: ❌ Session error: {}", i, e),
        }
    }

    // Logout
    println!("\n  👋 Logging out (invalidating session)...");
    auth_manager.logout(&session_id)?;
    println!("  ✅ Session invalidated");

    // Verify session is invalid
    println!("\n  ✔️  Testing invalidated session...");
    match auth_manager.validate_session(&session_id) {
        Ok(_) => println!("  ⚠️  Unexpected success with invalidated session!"),
        Err(_) => println!("  ✅ Invalidated session correctly rejected"),
    }

    Ok(())
}

/// Demonstrates token refresh mechanism
///
/// Refresh tokens allow obtaining new access tokens without re-authentication:
/// - Long-lived refresh tokens
/// - Short-lived access tokens
/// - Refresh token rotation
/// - Revocation support
fn token_refresh_flow() -> Result<()> {
    let mut auth_manager = AuthManager::new(JwtConfig::default())
        .with_token_expiry(Duration::from_secs(300))        // 5 min access token
        .with_refresh_token_expiry(Duration::from_secs(86400)); // 24h refresh token

    // Register user
    let user = UserInfo::new("dave", "dave@company-d.com", "company_d")
        .with_password("dave_password")
        .with_permission(Permission::Read);
    auth_manager.register_user(user)?;

    // Initial authentication
    println!("  🔐 Initial authentication...");
    auth_manager.authenticate_password("dave@company-d.com", "dave_password")?;

    // Generate tokens
    println!("  🎟️  Generating access token and refresh token...");
    let access_token = auth_manager.generate_token("dave")?;
    let refresh_token = auth_manager.generate_refresh_token("dave")?;
    println!("  ✅ Tokens generated");
    println!("     Access token: {}...", &access_token[..30]);
    println!("     Refresh token: {}...", &refresh_token[..30]);

    // Simulate access token expiration and refresh
    println!("\n  ⏰ Simulating access token expiration...");
    println!("  🔄 Using refresh token to get new access token...");
    let new_access_token = auth_manager.refresh_access_token(&refresh_token)?;
    println!(
        "  ✅ New access token obtained: {}...",
        &new_access_token[..30]
    );

    // Validate new token
    println!("\n  ✔️  Validating new access token...");
    let validated = auth_manager.validate_token(&new_access_token)?;
    println!("  ✅ New token is valid");
    println!("     User: {}", validated.user_id);

    // Revoke refresh token
    println!("\n  🚫 Revoking refresh token...");
    auth_manager.revoke_refresh_token(&refresh_token)?;
    println!("  ✅ Refresh token revoked");

    // Try to use revoked refresh token
    println!("\n  ✔️  Testing revoked refresh token...");
    match auth_manager.refresh_access_token(&refresh_token) {
        Ok(_) => println!("  ⚠️  Unexpected success with revoked refresh token!"),
        Err(_) => println!("  ✅ Revoked refresh token correctly rejected"),
    }

    Ok(())
}

/// Demonstrates multi-tenant authentication scenarios
///
/// Shows how different tenants have isolated authentication:
/// - Tenant-specific user namespaces
/// - Cross-tenant access prevention
/// - Per-tenant security policies
fn multi_tenant_authentication() -> Result<()> {
    let mut auth_manager = AuthManager::new(JwtConfig::default());

    // Register users for different tenants
    println!("  🏢 Setting up multi-tenant environment...");

    let user_a = UserInfo::new("alice", "alice@company-a.com", "tenant_a")
        .with_password("password_a")
        .with_permission(Permission::Read)
        .with_permission(Permission::Write);
    auth_manager.register_user(user_a)?;

    let user_b = UserInfo::new("bob", "bob@company-b.com", "tenant_b")
        .with_password("password_b")
        .with_permission(Permission::Read);
    auth_manager.register_user(user_b)?;

    println!("  ✅ Registered users for tenant_a and tenant_b");

    // Authenticate users from different tenants
    println!("\n  🔐 Authenticating users from different tenants...");

    let auth_a = auth_manager.authenticate_password("alice@company-a.com", "password_a")?;
    println!("  ✅ Tenant A user authenticated:");
    println!(
        "     User: {} (Tenant: {})",
        auth_a.user_id, auth_a.tenant_id
    );
    println!("     Permissions: {:?}", auth_a.permissions);

    let auth_b = auth_manager.authenticate_password("bob@company-b.com", "password_b")?;
    println!("  ✅ Tenant B user authenticated:");
    println!(
        "     User: {} (Tenant: {})",
        auth_b.user_id, auth_b.tenant_id
    );
    println!("     Permissions: {:?}", auth_b.permissions);

    // Generate tokens with tenant context
    println!("\n  🎟️  Generating tenant-scoped JWT tokens...");
    let token_a = auth_manager.generate_token("alice")?;
    let token_b = auth_manager.generate_token("bob")?;

    // Validate tokens and check tenant isolation
    println!("\n  ✔️  Validating tenant isolation...");
    let validated_a = auth_manager.validate_token(&token_a)?;
    let validated_b = auth_manager.validate_token(&token_b)?;

    println!("  ✅ Tenant isolation verified:");
    println!("     Token A tenant: {}", validated_a.tenant_id);
    println!("     Token B tenant: {}", validated_b.tenant_id);
    println!(
        "     Tenants are separate: {}",
        validated_a.tenant_id != validated_b.tenant_id
    );

    Ok(())
}

/// Demonstrates security audit trail
///
/// Shows how authentication events are logged for security monitoring:
/// - Login attempts (successful/failed)
/// - Token operations
/// - Permission changes
/// - Suspicious activity detection
fn security_audit_trail() -> Result<()> {
    let mut auth_manager = AuthManager::new(JwtConfig::default());

    // Register user
    let user = UserInfo::new("eve", "eve@company-e.com", "company_e")
        .with_password("eve_password")
        .with_permission(Permission::Read);
    auth_manager.register_user(user)?;

    println!("  📋 Generating authentication events...");

    // Successful login
    auth_manager.authenticate_password("eve@company-e.com", "eve_password")?;

    // Failed login attempt
    let _ = auth_manager.authenticate_password("eve@company-e.com", "wrong_password");

    // Token operations
    let token = auth_manager.generate_token("eve")?;
    auth_manager.validate_token(&token)?;

    // API key creation
    auth_manager.create_api_key("eve", "Test Key", None)?;

    // Retrieve audit events
    println!("\n  📊 Security Audit Trail:");
    let events = auth_manager.get_user_events("eve");
    for (i, event) in events.iter().enumerate() {
        println!("     Event {}: {:?}", i + 1, event.event_type);
        println!("       Time: {:?}", event.timestamp);
        println!("       Method: {:?}", event.auth_method);
        println!("       Success: {}", event.success);
        if let Some(ref msg) = event.error_message {
            println!("       Error: {}", msg);
        }
        println!();
    }

    println!("  ✅ Total events logged: {}", events.len());

    Ok(())
}

/// Demonstrates rate limiting and security controls
///
/// Shows advanced security features:
/// - Request rate limiting
/// - IP-based access control
/// - Account lockout policies
/// - Automated cleanup of expired credentials
fn rate_limiting_controls() -> Result<()> {
    println!("  🚦 Security controls demonstration");
    println!("     - Rate limiting prevents abuse");
    println!("     - IP whitelisting restricts access");
    println!("     - Automatic cleanup of expired tokens");
    println!("     - Session timeout enforcement");

    let mut auth_manager = AuthManager::new(JwtConfig::default())
        .with_token_expiry(Duration::from_secs(60))
        .with_session_timeout(Duration::from_secs(300));

    // Register user
    let user = UserInfo::new("frank", "frank@company-f.com", "company_f")
        .with_password("frank_password")
        .with_permission(Permission::Read);
    auth_manager.register_user(user)?;

    // Create some test sessions and tokens
    println!("\n  ⏳ Creating test sessions...");
    for i in 1..=3 {
        auth_manager.authenticate_password("frank@company-f.com", "frank_password")?;
        println!("     Session {} created", i);
    }

    // Cleanup expired credentials
    println!("\n  🧹 Running cleanup of expired credentials...");
    auth_manager.cleanup_expired();
    println!("  ✅ Cleanup completed");

    // Password change for security
    println!("\n  🔐 Demonstrating password change...");
    auth_manager.change_password("frank", "frank_password", "new_secure_password")?;
    println!("  ✅ Password changed successfully");
    println!("     All refresh tokens have been revoked for security");

    // Verify old password doesn't work
    match auth_manager.authenticate_password("frank@company-f.com", "frank_password") {
        Ok(_) => println!("  ⚠️  Old password still works - security issue!"),
        Err(_) => println!("  ✅ Old password correctly rejected"),
    }

    // Verify new password works
    match auth_manager.authenticate_password("frank@company-f.com", "new_secure_password") {
        Ok(_) => println!("  ✅ New password works correctly"),
        Err(_) => println!("  ⚠️  New password doesn't work - security issue!"),
    }

    Ok(())
}

/// Helper function to hash client secrets (simplified for example)
fn hash_secret(secret: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}
