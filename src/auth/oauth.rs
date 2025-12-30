//! OAuth 2.0 Implementation
//!
//! This module provides OAuth 2.0 support including:
//! - Authorization Code flow
//! - Client Credentials flow
//! - Token introspection
//! - Token revocation

use crate::error::{Error, Result};
use crate::multitenancy::{Permission, TenantId};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// OAuth 2.0 configuration
#[derive(Debug, Clone)]
pub struct OAuthConfig {
    /// OAuth provider URL
    pub provider_url: String,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Redirect URI for authorization code flow
    pub redirect_uri: Option<String>,
    /// Requested scopes
    pub scopes: Vec<String>,
    /// Token endpoint
    pub token_endpoint: String,
    /// Authorization endpoint
    pub authorization_endpoint: String,
    /// Introspection endpoint
    pub introspection_endpoint: Option<String>,
    /// Revocation endpoint
    pub revocation_endpoint: Option<String>,
    /// Token expiration
    pub token_expiry: Duration,
}

impl OAuthConfig {
    /// Create a new OAuth configuration
    pub fn new(
        provider_url: impl Into<String>,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
    ) -> Self {
        let provider = provider_url.into();
        OAuthConfig {
            provider_url: provider.clone(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            redirect_uri: None,
            scopes: vec!["openid".to_string(), "profile".to_string()],
            token_endpoint: format!("{}/oauth/token", provider),
            authorization_endpoint: format!("{}/oauth/authorize", provider),
            introspection_endpoint: Some(format!("{}/oauth/introspect", provider)),
            revocation_endpoint: Some(format!("{}/oauth/revoke", provider)),
            token_expiry: Duration::from_secs(3600),
        }
    }

    /// Set redirect URI
    pub fn with_redirect_uri(mut self, uri: impl Into<String>) -> Self {
        self.redirect_uri = Some(uri.into());
        self
    }

    /// Set scopes
    pub fn with_scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = scopes;
        self
    }

    /// Add a scope
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scopes.push(scope.into());
        self
    }

    /// Set custom token endpoint
    pub fn with_token_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.token_endpoint = endpoint.into();
        self
    }

    /// Set custom authorization endpoint
    pub fn with_authorization_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.authorization_endpoint = endpoint.into();
        self
    }
}

/// OAuth 2.0 grant types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OAuthGrantType {
    /// Authorization code grant
    AuthorizationCode,
    /// Client credentials grant
    ClientCredentials,
    /// Refresh token grant
    RefreshToken,
    /// Password grant (not recommended)
    Password,
}

impl OAuthGrantType {
    /// Get the grant type string for OAuth requests
    pub fn as_str(&self) -> &str {
        match self {
            OAuthGrantType::AuthorizationCode => "authorization_code",
            OAuthGrantType::ClientCredentials => "client_credentials",
            OAuthGrantType::RefreshToken => "refresh_token",
            OAuthGrantType::Password => "password",
        }
    }
}

/// OAuth 2.0 authorization request
#[derive(Debug, Clone)]
pub struct AuthorizationRequest {
    /// Client ID
    pub client_id: String,
    /// Redirect URI
    pub redirect_uri: String,
    /// Response type (typically "code")
    pub response_type: String,
    /// Requested scopes
    pub scopes: Vec<String>,
    /// State parameter for CSRF protection
    pub state: String,
    /// PKCE code challenge (optional)
    pub code_challenge: Option<String>,
    /// PKCE code challenge method
    pub code_challenge_method: Option<String>,
    /// Additional parameters
    pub extra_params: HashMap<String, String>,
}

impl AuthorizationRequest {
    /// Create a new authorization request
    pub fn new(client_id: impl Into<String>, redirect_uri: impl Into<String>) -> Self {
        AuthorizationRequest {
            client_id: client_id.into(),
            redirect_uri: redirect_uri.into(),
            response_type: "code".to_string(),
            scopes: vec!["openid".to_string()],
            state: generate_state(),
            code_challenge: None,
            code_challenge_method: None,
            extra_params: HashMap::new(),
        }
    }

    /// Set scopes
    pub fn with_scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = scopes;
        self
    }

    /// Enable PKCE with S256 method
    pub fn with_pkce(mut self) -> (Self, String) {
        let (verifier, challenge) = generate_pkce_pair();
        self.code_challenge = Some(challenge);
        self.code_challenge_method = Some("S256".to_string());
        (self, verifier)
    }

    /// Build the authorization URL
    pub fn build_url(&self, base_url: &str) -> String {
        let mut params = vec![
            ("client_id", self.client_id.as_str()),
            ("redirect_uri", self.redirect_uri.as_str()),
            ("response_type", self.response_type.as_str()),
            ("state", self.state.as_str()),
        ];

        let scopes = self.scopes.join(" ");
        params.push(("scope", &scopes));

        if let Some(ref challenge) = self.code_challenge {
            params.push(("code_challenge", challenge));
        }
        if let Some(ref method) = self.code_challenge_method {
            params.push(("code_challenge_method", method));
        }

        let query: Vec<String> = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, url_encode(v)))
            .collect();

        format!("{}?{}", base_url, query.join("&"))
    }
}

/// OAuth 2.0 token request
#[derive(Debug, Clone)]
pub struct TokenRequest {
    /// Grant type
    pub grant_type: OAuthGrantType,
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Authorization code (for authorization_code grant)
    pub code: Option<String>,
    /// Redirect URI (for authorization_code grant)
    pub redirect_uri: Option<String>,
    /// PKCE code verifier
    pub code_verifier: Option<String>,
    /// Refresh token (for refresh_token grant)
    pub refresh_token: Option<String>,
    /// Username (for password grant)
    pub username: Option<String>,
    /// Password (for password grant)
    pub password: Option<String>,
    /// Requested scopes
    pub scopes: Option<Vec<String>>,
}

impl TokenRequest {
    /// Create a client credentials token request
    pub fn client_credentials(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
    ) -> Self {
        TokenRequest {
            grant_type: OAuthGrantType::ClientCredentials,
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            code: None,
            redirect_uri: None,
            code_verifier: None,
            refresh_token: None,
            username: None,
            password: None,
            scopes: None,
        }
    }

    /// Create an authorization code token request
    pub fn authorization_code(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        code: impl Into<String>,
        redirect_uri: impl Into<String>,
    ) -> Self {
        TokenRequest {
            grant_type: OAuthGrantType::AuthorizationCode,
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            code: Some(code.into()),
            redirect_uri: Some(redirect_uri.into()),
            code_verifier: None,
            refresh_token: None,
            username: None,
            password: None,
            scopes: None,
        }
    }

    /// Create a refresh token request
    pub fn refresh_token(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        refresh_token: impl Into<String>,
    ) -> Self {
        TokenRequest {
            grant_type: OAuthGrantType::RefreshToken,
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            code: None,
            redirect_uri: None,
            code_verifier: None,
            refresh_token: Some(refresh_token.into()),
            username: None,
            password: None,
            scopes: None,
        }
    }

    /// Add PKCE code verifier
    pub fn with_code_verifier(mut self, verifier: impl Into<String>) -> Self {
        self.code_verifier = Some(verifier.into());
        self
    }

    /// Set scopes
    pub fn with_scopes(mut self, scopes: Vec<String>) -> Self {
        self.scopes = Some(scopes);
        self
    }

    /// Build request body for token endpoint
    pub fn build_body(&self) -> HashMap<String, String> {
        let mut body = HashMap::new();

        body.insert(
            "grant_type".to_string(),
            self.grant_type.as_str().to_string(),
        );
        body.insert("client_id".to_string(), self.client_id.clone());
        body.insert("client_secret".to_string(), self.client_secret.clone());

        if let Some(ref code) = self.code {
            body.insert("code".to_string(), code.clone());
        }
        if let Some(ref redirect_uri) = self.redirect_uri {
            body.insert("redirect_uri".to_string(), redirect_uri.clone());
        }
        if let Some(ref verifier) = self.code_verifier {
            body.insert("code_verifier".to_string(), verifier.clone());
        }
        if let Some(ref refresh_token) = self.refresh_token {
            body.insert("refresh_token".to_string(), refresh_token.clone());
        }
        if let Some(ref scopes) = self.scopes {
            body.insert("scope".to_string(), scopes.join(" "));
        }

        body
    }
}

/// OAuth 2.0 token response
#[derive(Debug, Clone)]
pub struct TokenResponse {
    /// Access token
    pub access_token: String,
    /// Token type (typically "Bearer")
    pub token_type: String,
    /// Expires in (seconds)
    pub expires_in: u64,
    /// Refresh token (optional)
    pub refresh_token: Option<String>,
    /// Granted scopes
    pub scope: Option<String>,
    /// ID token (for OpenID Connect)
    pub id_token: Option<String>,
}

impl TokenResponse {
    /// Create a new token response
    pub fn new(access_token: impl Into<String>, expires_in: u64) -> Self {
        TokenResponse {
            access_token: access_token.into(),
            token_type: "Bearer".to_string(),
            expires_in,
            refresh_token: None,
            scope: None,
            id_token: None,
        }
    }

    /// Set refresh token
    pub fn with_refresh_token(mut self, token: impl Into<String>) -> Self {
        self.refresh_token = Some(token.into());
        self
    }

    /// Set scope
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Set ID token
    pub fn with_id_token(mut self, token: impl Into<String>) -> Self {
        self.id_token = Some(token.into());
        self
    }

    /// Check if token is expired
    pub fn is_expired(&self, issued_at: SystemTime) -> bool {
        issued_at + Duration::from_secs(self.expires_in) < SystemTime::now()
    }

    /// Parse from JSON response
    pub fn from_json(json: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| Error::InvalidInput(format!("Invalid JSON: {}", e)))?;

        let access_token = value
            .get("access_token")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::InvalidInput("Missing access_token".to_string()))?
            .to_string();

        let token_type = value
            .get("token_type")
            .and_then(|v| v.as_str())
            .unwrap_or("Bearer")
            .to_string();

        let expires_in = value
            .get("expires_in")
            .and_then(|v| v.as_u64())
            .unwrap_or(3600);

        let refresh_token = value
            .get("refresh_token")
            .and_then(|v| v.as_str())
            .map(String::from);

        let scope = value
            .get("scope")
            .and_then(|v| v.as_str())
            .map(String::from);

        let id_token = value
            .get("id_token")
            .and_then(|v| v.as_str())
            .map(String::from);

        Ok(TokenResponse {
            access_token,
            token_type,
            expires_in,
            refresh_token,
            scope,
            id_token,
        })
    }
}

/// OAuth 2.0 token introspection response
#[derive(Debug, Clone)]
pub struct IntrospectionResponse {
    /// Whether the token is active
    pub active: bool,
    /// Token scopes
    pub scope: Option<String>,
    /// Client ID
    pub client_id: Option<String>,
    /// Username/Subject
    pub username: Option<String>,
    /// Token type
    pub token_type: Option<String>,
    /// Expiration time (Unix timestamp)
    pub exp: Option<u64>,
    /// Issued at (Unix timestamp)
    pub iat: Option<u64>,
    /// Not before (Unix timestamp)
    pub nbf: Option<u64>,
    /// Subject
    pub sub: Option<String>,
    /// Audience
    pub aud: Option<String>,
    /// Issuer
    pub iss: Option<String>,
    /// JWT ID
    pub jti: Option<String>,
}

impl IntrospectionResponse {
    /// Parse from JSON response
    pub fn from_json(json: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| Error::InvalidInput(format!("Invalid JSON: {}", e)))?;

        let active = value
            .get("active")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(IntrospectionResponse {
            active,
            scope: value
                .get("scope")
                .and_then(|v| v.as_str())
                .map(String::from),
            client_id: value
                .get("client_id")
                .and_then(|v| v.as_str())
                .map(String::from),
            username: value
                .get("username")
                .and_then(|v| v.as_str())
                .map(String::from),
            token_type: value
                .get("token_type")
                .and_then(|v| v.as_str())
                .map(String::from),
            exp: value.get("exp").and_then(|v| v.as_u64()),
            iat: value.get("iat").and_then(|v| v.as_u64()),
            nbf: value.get("nbf").and_then(|v| v.as_u64()),
            sub: value.get("sub").and_then(|v| v.as_str()).map(String::from),
            aud: value.get("aud").and_then(|v| v.as_str()).map(String::from),
            iss: value.get("iss").and_then(|v| v.as_str()).map(String::from),
            jti: value.get("jti").and_then(|v| v.as_str()).map(String::from),
        })
    }
}

/// OAuth 2.0 client for local testing/simulation
#[derive(Debug)]
pub struct OAuthClient {
    /// Configuration
    config: OAuthConfig,
    /// Registered clients for simulation
    clients: HashMap<String, OAuthClientInfo>,
    /// Authorization codes (for simulation)
    auth_codes: HashMap<String, AuthCode>,
    /// Active tokens
    tokens: HashMap<String, OAuthToken>,
}

/// OAuth client registration info
#[derive(Debug, Clone)]
pub struct OAuthClientInfo {
    /// Client ID
    pub client_id: String,
    /// Client secret hash
    pub client_secret_hash: String,
    /// Allowed redirect URIs
    pub redirect_uris: Vec<String>,
    /// Allowed grant types
    pub grant_types: Vec<OAuthGrantType>,
    /// Allowed scopes
    pub scopes: Vec<String>,
    /// Associated tenant
    pub tenant_id: TenantId,
}

/// Authorization code
#[derive(Debug, Clone)]
struct AuthCode {
    code: String,
    client_id: String,
    redirect_uri: String,
    scopes: Vec<String>,
    user_id: String,
    code_challenge: Option<String>,
    code_challenge_method: Option<String>,
    expires_at: SystemTime,
}

/// OAuth token
#[derive(Debug, Clone)]
struct OAuthToken {
    access_token: String,
    refresh_token: Option<String>,
    client_id: String,
    user_id: Option<String>,
    scopes: Vec<String>,
    expires_at: SystemTime,
    revoked: bool,
}

impl OAuthClient {
    /// Create a new OAuth client
    pub fn new(config: OAuthConfig) -> Self {
        OAuthClient {
            config,
            clients: HashMap::new(),
            auth_codes: HashMap::new(),
            tokens: HashMap::new(),
        }
    }

    /// Register an OAuth client
    pub fn register_client(&mut self, info: OAuthClientInfo) {
        self.clients.insert(info.client_id.clone(), info);
    }

    /// Validate client credentials
    pub fn validate_client(
        &self,
        client_id: &str,
        client_secret: &str,
    ) -> Result<&OAuthClientInfo> {
        let client = self
            .clients
            .get(client_id)
            .ok_or_else(|| Error::InvalidInput("Invalid client_id".to_string()))?;

        let secret_hash = hash_client_secret(client_secret);
        if client.client_secret_hash != secret_hash {
            return Err(Error::InvalidInput("Invalid client_secret".to_string()));
        }

        Ok(client)
    }

    /// Create authorization code
    pub fn create_authorization_code(
        &mut self,
        client_id: &str,
        redirect_uri: &str,
        scopes: Vec<String>,
        user_id: &str,
        code_challenge: Option<String>,
        code_challenge_method: Option<String>,
    ) -> Result<String> {
        let client = self
            .clients
            .get(client_id)
            .ok_or_else(|| Error::InvalidInput("Invalid client_id".to_string()))?;

        // Validate redirect URI
        if !client.redirect_uris.contains(&redirect_uri.to_string()) {
            return Err(Error::InvalidInput("Invalid redirect_uri".to_string()));
        }

        // Validate scopes
        for scope in &scopes {
            if !client.scopes.contains(scope) {
                return Err(Error::InvalidInput(format!("Invalid scope: {}", scope)));
            }
        }

        let code = generate_authorization_code();
        let auth_code = AuthCode {
            code: code.clone(),
            client_id: client_id.to_string(),
            redirect_uri: redirect_uri.to_string(),
            scopes,
            user_id: user_id.to_string(),
            code_challenge,
            code_challenge_method,
            expires_at: SystemTime::now() + Duration::from_secs(600), // 10 minutes
        };

        self.auth_codes.insert(code.clone(), auth_code);

        Ok(code)
    }

    /// Exchange authorization code for tokens
    pub fn exchange_code(
        &mut self,
        client_id: &str,
        client_secret: &str,
        code: &str,
        redirect_uri: &str,
        code_verifier: Option<&str>,
    ) -> Result<TokenResponse> {
        // Validate client
        self.validate_client(client_id, client_secret)?;

        // Get and validate authorization code
        let auth_code = self
            .auth_codes
            .remove(code)
            .ok_or_else(|| Error::InvalidInput("Invalid authorization code".to_string()))?;

        if auth_code.expires_at < SystemTime::now() {
            return Err(Error::InvalidOperation(
                "Authorization code expired".to_string(),
            ));
        }

        if auth_code.client_id != client_id {
            return Err(Error::InvalidInput("Client ID mismatch".to_string()));
        }

        if auth_code.redirect_uri != redirect_uri {
            return Err(Error::InvalidInput("Redirect URI mismatch".to_string()));
        }

        // Validate PKCE if present
        if let Some(challenge) = &auth_code.code_challenge {
            let verifier = code_verifier
                .ok_or_else(|| Error::InvalidInput("Missing code_verifier".to_string()))?;

            let expected_challenge = compute_code_challenge(verifier);
            if &expected_challenge != challenge {
                return Err(Error::InvalidInput("Invalid code_verifier".to_string()));
            }
        }

        // Generate tokens
        let access_token = generate_access_token();
        let refresh_token = generate_refresh_token();

        let token = OAuthToken {
            access_token: access_token.clone(),
            refresh_token: Some(refresh_token.clone()),
            client_id: client_id.to_string(),
            user_id: Some(auth_code.user_id),
            scopes: auth_code.scopes.clone(),
            expires_at: SystemTime::now() + self.config.token_expiry,
            revoked: false,
        };

        self.tokens.insert(access_token.clone(), token);

        Ok(
            TokenResponse::new(access_token, self.config.token_expiry.as_secs())
                .with_refresh_token(refresh_token)
                .with_scope(auth_code.scopes.join(" ")),
        )
    }

    /// Client credentials grant
    pub fn client_credentials_grant(
        &mut self,
        client_id: &str,
        client_secret: &str,
        scopes: Option<Vec<String>>,
    ) -> Result<TokenResponse> {
        let client = self.validate_client(client_id, client_secret)?;

        // Check if client_credentials grant is allowed
        if !client
            .grant_types
            .contains(&OAuthGrantType::ClientCredentials)
        {
            return Err(Error::InvalidOperation(
                "Client credentials grant not allowed".to_string(),
            ));
        }

        // Validate scopes
        let requested_scopes = scopes.unwrap_or_else(|| client.scopes.clone());
        for scope in &requested_scopes {
            if !client.scopes.contains(scope) {
                return Err(Error::InvalidInput(format!("Invalid scope: {}", scope)));
            }
        }

        // Generate token
        let access_token = generate_access_token();

        let token = OAuthToken {
            access_token: access_token.clone(),
            refresh_token: None, // No refresh token for client credentials
            client_id: client_id.to_string(),
            user_id: None,
            scopes: requested_scopes.clone(),
            expires_at: SystemTime::now() + self.config.token_expiry,
            revoked: false,
        };

        self.tokens.insert(access_token.clone(), token);

        Ok(
            TokenResponse::new(access_token, self.config.token_expiry.as_secs())
                .with_scope(requested_scopes.join(" ")),
        )
    }

    /// Introspect token
    pub fn introspect_token(&self, token: &str) -> IntrospectionResponse {
        match self.tokens.get(token) {
            Some(oauth_token)
                if !oauth_token.revoked && oauth_token.expires_at > SystemTime::now() =>
            {
                IntrospectionResponse {
                    active: true,
                    scope: Some(oauth_token.scopes.join(" ")),
                    client_id: Some(oauth_token.client_id.clone()),
                    username: oauth_token.user_id.clone(),
                    token_type: Some("Bearer".to_string()),
                    exp: Some(
                        oauth_token
                            .expires_at
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    ),
                    iat: None,
                    nbf: None,
                    sub: oauth_token.user_id.clone(),
                    aud: None,
                    iss: None,
                    jti: None,
                }
            }
            _ => IntrospectionResponse {
                active: false,
                scope: None,
                client_id: None,
                username: None,
                token_type: None,
                exp: None,
                iat: None,
                nbf: None,
                sub: None,
                aud: None,
                iss: None,
                jti: None,
            },
        }
    }

    /// Revoke token
    pub fn revoke_token(&mut self, token: &str) -> bool {
        if let Some(oauth_token) = self.tokens.get_mut(token) {
            oauth_token.revoked = true;
            true
        } else {
            false
        }
    }
}

// Helper functions

/// Generate a state parameter for CSRF protection
fn generate_state() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Generate PKCE code verifier and challenge pair
fn generate_pkce_pair() -> (String, String) {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);

    // URL-safe base64 encoding for verifier
    let verifier = bytes
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>();

    let challenge = compute_code_challenge(&verifier);

    (verifier, challenge)
}

/// Compute PKCE code challenge from verifier
fn compute_code_challenge(verifier: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let hash = hasher.finalize();

    // Base64 URL encoding
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    let mut result = String::with_capacity(43); // 256 bits / 6 = ~43 characters

    for chunk in hash.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(ALPHABET[b0 >> 2] as char);
        result.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            result.push(ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)] as char);
        }
        if chunk.len() > 2 {
            result.push(ALPHABET[b2 & 0x3f] as char);
        }
    }

    result
}

/// Generate authorization code
fn generate_authorization_code() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Generate access token
fn generate_access_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    format!(
        "at_{}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Generate refresh token
fn generate_refresh_token() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    format!(
        "rt_{}",
        bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()
    )
}

/// Hash client secret for storage
fn hash_client_secret(secret: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

/// URL encode a string
fn url_encode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oauth_config() {
        let config = OAuthConfig::new("https://auth.example.com", "client_id", "client_secret")
            .with_redirect_uri("https://app.example.com/callback")
            .with_scope("email");

        assert_eq!(config.provider_url, "https://auth.example.com");
        assert!(config.scopes.contains(&"email".to_string()));
    }

    #[test]
    fn test_authorization_request() {
        let request = AuthorizationRequest::new("client_id", "https://app.example.com/callback")
            .with_scopes(vec!["openid".to_string(), "profile".to_string()]);

        let url = request.build_url("https://auth.example.com/authorize");
        assert!(url.contains("client_id=client_id"));
        assert!(url.contains("redirect_uri="));
        assert!(url.contains("response_type=code"));
    }

    #[test]
    fn test_pkce() {
        let (verifier, challenge) = generate_pkce_pair();
        assert!(!verifier.is_empty());
        assert!(!challenge.is_empty());

        // Challenge should be derived from verifier
        let computed_challenge = compute_code_challenge(&verifier);
        assert_eq!(challenge, computed_challenge);
    }

    #[test]
    fn test_token_request() {
        let request = TokenRequest::client_credentials("client_id", "client_secret")
            .with_scopes(vec!["read".to_string()]);

        let body = request.build_body();
        assert_eq!(
            body.get("grant_type"),
            Some(&"client_credentials".to_string())
        );
        assert_eq!(body.get("client_id"), Some(&"client_id".to_string()));
    }

    #[test]
    fn test_token_response_parsing() {
        let json = r#"{"access_token":"abc123","token_type":"Bearer","expires_in":3600,"refresh_token":"xyz789"}"#;
        let response = TokenResponse::from_json(json).unwrap();

        assert_eq!(response.access_token, "abc123");
        assert_eq!(response.token_type, "Bearer");
        assert_eq!(response.expires_in, 3600);
        assert_eq!(response.refresh_token, Some("xyz789".to_string()));
    }

    #[test]
    fn test_oauth_client_credentials_flow() {
        let config = OAuthConfig::new("https://auth.example.com", "test", "test");
        let mut client = OAuthClient::new(config);

        // Register client
        let client_info = OAuthClientInfo {
            client_id: "test_client".to_string(),
            client_secret_hash: hash_client_secret("test_secret"),
            redirect_uris: vec!["https://app.example.com/callback".to_string()],
            grant_types: vec![
                OAuthGrantType::ClientCredentials,
                OAuthGrantType::AuthorizationCode,
            ],
            scopes: vec!["read".to_string(), "write".to_string()],
            tenant_id: "tenant_a".to_string(),
        };
        client.register_client(client_info);

        // Get token
        let response = client
            .client_credentials_grant("test_client", "test_secret", Some(vec!["read".to_string()]))
            .unwrap();

        assert!(!response.access_token.is_empty());
        assert!(response.access_token.starts_with("at_"));

        // Introspect token
        let introspection = client.introspect_token(&response.access_token);
        assert!(introspection.active);
        assert_eq!(introspection.client_id, Some("test_client".to_string()));
    }

    #[test]
    fn test_oauth_authorization_code_flow() {
        let config = OAuthConfig::new("https://auth.example.com", "test", "test");
        let mut client = OAuthClient::new(config);

        // Register client
        let client_info = OAuthClientInfo {
            client_id: "test_client".to_string(),
            client_secret_hash: hash_client_secret("test_secret"),
            redirect_uris: vec!["https://app.example.com/callback".to_string()],
            grant_types: vec![OAuthGrantType::AuthorizationCode],
            scopes: vec!["read".to_string(), "write".to_string()],
            tenant_id: "tenant_a".to_string(),
        };
        client.register_client(client_info);

        // Create authorization code
        let code = client
            .create_authorization_code(
                "test_client",
                "https://app.example.com/callback",
                vec!["read".to_string()],
                "user123",
                None,
                None,
            )
            .unwrap();

        // Exchange code for tokens
        let response = client
            .exchange_code(
                "test_client",
                "test_secret",
                &code,
                "https://app.example.com/callback",
                None,
            )
            .unwrap();

        assert!(!response.access_token.is_empty());
        assert!(response.refresh_token.is_some());
    }

    #[test]
    fn test_token_revocation() {
        let config = OAuthConfig::new("https://auth.example.com", "test", "test");
        let mut client = OAuthClient::new(config);

        let client_info = OAuthClientInfo {
            client_id: "test_client".to_string(),
            client_secret_hash: hash_client_secret("test_secret"),
            redirect_uris: vec![],
            grant_types: vec![OAuthGrantType::ClientCredentials],
            scopes: vec!["read".to_string()],
            tenant_id: "tenant_a".to_string(),
        };
        client.register_client(client_info);

        let response = client
            .client_credentials_grant("test_client", "test_secret", None)
            .unwrap();

        // Token should be active
        assert!(client.introspect_token(&response.access_token).active);

        // Revoke token
        assert!(client.revoke_token(&response.access_token));

        // Token should be inactive
        assert!(!client.introspect_token(&response.access_token).active);
    }
}
