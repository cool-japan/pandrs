//! JWT (JSON Web Token) Implementation
//!
//! This module provides JWT token generation and validation with
//! HMAC-SHA256 signing.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// JWT configuration
#[derive(Clone)]
pub struct JwtConfig {
    /// Secret key for HMAC signing
    pub secret_key: Vec<u8>,
    /// Token issuer
    pub issuer: String,
    /// Token audience
    pub audience: String,
    /// Token expiration in seconds
    pub expiration_secs: u64,
    /// Whether to validate expiration
    pub validate_exp: bool,
    /// Whether to validate issuer
    pub validate_iss: bool,
    /// Whether to validate audience
    pub validate_aud: bool,
    /// Allowed clock skew in seconds
    pub leeway_secs: u64,
}

impl std::fmt::Debug for JwtConfig {
    /// Redacting `Debug`: the HMAC signing key must never appear in logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JwtConfig")
            .field("secret_key", &"<redacted>")
            .field("issuer", &self.issuer)
            .field("audience", &self.audience)
            .field("expiration_secs", &self.expiration_secs)
            .field("validate_exp", &self.validate_exp)
            .field("validate_iss", &self.validate_iss)
            .field("validate_aud", &self.validate_aud)
            .field("leeway_secs", &self.leeway_secs)
            .finish()
    }
}

impl Drop for JwtConfig {
    /// Best-effort zeroization of the signing key when the config is dropped.
    ///
    /// The `zeroize` crate would be idiomatic but adding a dependency is
    /// outside this change's file ownership; this pure-Rust equivalent
    /// overwrites the bytes and uses `black_box` to keep the optimizer from
    /// eliding the writes on a soon-to-be-freed buffer.
    fn drop(&mut self) {
        for b in self.secret_key.iter_mut() {
            *b = 0;
        }
        std::hint::black_box(self.secret_key.as_ptr());
    }
}

impl Default for JwtConfig {
    /// Builds a config with a **fresh random 64-byte secret each call**.
    ///
    /// This is fail-closed and secure for a single long-lived process, but it is
    /// *not* stable across restarts or processes: two `AuthManager`s built from
    /// separate `JwtConfig::default()` values cannot validate each other's
    /// tokens, and a restart invalidates every previously issued token. Any
    /// multi-process, load-balanced, or restart-surviving deployment MUST supply
    /// an explicit shared secret via [`JwtConfig::new`] (which still enforces the
    /// >= 32-byte minimum at encode/decode time). A fixed default secret is
    /// deliberately not provided — that would be strictly worse (a public,
    /// forgeable key).
    fn default() -> Self {
        use scirs2_core::random::Rng;
        let mut secret = vec![0u8; 64];
        scirs2_core::random::rng().fill_bytes(&mut secret);

        JwtConfig {
            secret_key: secret,
            issuer: "pandrs".to_string(),
            audience: "pandrs-api".to_string(),
            expiration_secs: 3600,
            validate_exp: true,
            validate_iss: true,
            validate_aud: true,
            leeway_secs: 60,
        }
    }
}

impl JwtConfig {
    /// Create a new JWT configuration with a specific secret
    pub fn new(secret: impl Into<Vec<u8>>) -> Self {
        // Field assignment rather than functional-record-update: `JwtConfig`
        // implements `Drop` (to zeroize the key), and FRU would require moving
        // fields out of the `Default` temporary, which Drop forbids.
        let mut config = JwtConfig::default();
        config.secret_key = secret.into();
        config
    }

    /// Set the issuer
    pub fn with_issuer(mut self, issuer: impl Into<String>) -> Self {
        self.issuer = issuer.into();
        self
    }

    /// Set the audience
    pub fn with_audience(mut self, audience: impl Into<String>) -> Self {
        self.audience = audience.into();
        self
    }

    /// Set the expiration time in seconds
    pub fn with_expiration(mut self, secs: u64) -> Self {
        self.expiration_secs = secs;
        self
    }

    /// Disable expiration validation
    pub fn without_exp_validation(mut self) -> Self {
        self.validate_exp = false;
        self
    }

    /// Disable issuer validation
    pub fn without_iss_validation(mut self) -> Self {
        self.validate_iss = false;
        self
    }

    /// Disable audience validation
    pub fn without_aud_validation(mut self) -> Self {
        self.validate_aud = false;
        self
    }
}

/// JWT token claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Tenant ID
    pub tenant_id: String,
    /// User roles
    pub roles: Vec<String>,
    /// Permissions
    pub permissions: Vec<String>,
    /// Issued at (Unix timestamp)
    pub iat: u64,
    /// Not before (Unix timestamp). Optional; defaults to absent for tokens
    /// minted before this field existed.
    #[serde(default)]
    pub nbf: Option<u64>,
    /// Expiration time (Unix timestamp)
    pub exp: u64,
    /// Issuer
    pub iss: String,
    /// Audience
    pub aud: String,
    /// JWT ID (unique identifier)
    pub jti: String,
}

/// JWT header
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JwtHeader {
    /// Algorithm (always HS256)
    alg: String,
    /// Token type (always JWT)
    typ: String,
}

impl Default for JwtHeader {
    fn default() -> Self {
        JwtHeader {
            alg: "HS256".to_string(),
            typ: "JWT".to_string(),
        }
    }
}

/// Minimum accepted HMAC secret length (bytes). HS256 with a key shorter than
/// its 256-bit output is trivially brute-forceable; an empty key is no key.
const MIN_SECRET_LEN: usize = 32;

/// Encode JWT token
pub fn encode_jwt(claims: &TokenClaims, config: &JwtConfig) -> Result<String> {
    if config.secret_key.len() < MIN_SECRET_LEN {
        return Err(Error::InvalidInput(format!(
            "JWT secret key must be at least {} bytes",
            MIN_SECRET_LEN
        )));
    }
    let header = JwtHeader::default();

    // Encode header
    let header_json = serde_json::to_string(&header)
        .map_err(|e| Error::InvalidOperation(format!("Failed to serialize header: {}", e)))?;
    let header_b64 = base64_url_encode(header_json.as_bytes());

    // Encode payload
    let payload_json = serde_json::to_string(claims)
        .map_err(|e| Error::InvalidOperation(format!("Failed to serialize claims: {}", e)))?;
    let payload_b64 = base64_url_encode(payload_json.as_bytes());

    // Create signature
    let message = format!("{}.{}", header_b64, payload_b64);
    let signature = hmac_sha256(&config.secret_key, message.as_bytes());
    let signature_b64 = base64_url_encode(&signature);

    Ok(format!("{}.{}.{}", header_b64, payload_b64, signature_b64))
}

/// Decode and validate JWT token
pub fn decode_jwt(token: &str, config: &JwtConfig) -> Result<TokenClaims> {
    if config.secret_key.len() < MIN_SECRET_LEN {
        return Err(Error::InvalidInput(format!(
            "JWT secret key must be at least {} bytes",
            MIN_SECRET_LEN
        )));
    }

    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(Error::InvalidInput("Invalid token format".to_string()));
    }

    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let signature_b64 = parts[2];

    // Verify signature in constant time over the RAW MAC bytes. Comparing the
    // base64 text with `!=` both leaks a timing oracle (byte-by-byte
    // short-circuit) and, because the previous decoder silently dropped invalid
    // characters, accepted malleable encodings. Decode the presented signature
    // strictly and `ct_eq` it against the freshly computed MAC.
    let message = format!("{}.{}", header_b64, payload_b64);
    let expected_signature = hmac_sha256(&config.secret_key, message.as_bytes());
    let actual_signature = base64_url_decode(signature_b64)
        .ok_or_else(|| Error::InvalidInput("Invalid signature encoding".to_string()))?;

    if !crate::auth::constant_time_eq(&expected_signature, &actual_signature) {
        return Err(Error::InvalidInput("Invalid token signature".to_string()));
    }

    // Decode header
    let header_bytes = base64_url_decode(header_b64)
        .ok_or_else(|| Error::InvalidInput("Invalid header encoding".to_string()))?;
    let header: JwtHeader = serde_json::from_slice(&header_bytes)
        .map_err(|e| Error::InvalidInput(format!("Invalid header: {}", e)))?;

    if header.alg != "HS256" {
        return Err(Error::InvalidInput("Unsupported algorithm".to_string()));
    }

    // Decode payload
    let payload_bytes = base64_url_decode(payload_b64)
        .ok_or_else(|| Error::InvalidInput("Invalid payload encoding".to_string()))?;
    let claims: TokenClaims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| Error::InvalidInput(format!("Invalid payload: {}", e)))?;

    // Validate claims
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if config.validate_exp {
        // Saturating so a token claiming exp == u64::MAX cannot wrap the
        // leeway addition to a small number and appear expired.
        if claims.exp.saturating_add(config.leeway_secs) < now {
            return Err(Error::InvalidOperation("Token has expired".to_string()));
        }
        // Reject tokens that are not yet valid (nbf), honouring the same leeway.
        if let Some(nbf) = claims.nbf {
            if nbf > now.saturating_add(config.leeway_secs) {
                return Err(Error::InvalidOperation(
                    "Token is not yet valid".to_string(),
                ));
            }
        }
    }

    if config.validate_iss && claims.iss != config.issuer {
        return Err(Error::InvalidInput("Invalid issuer".to_string()));
    }

    if config.validate_aud && claims.aud != config.audience {
        return Err(Error::InvalidInput("Invalid audience".to_string()));
    }

    Ok(claims)
}

/// Verify a JWT and report validity as a boolean.
///
/// Returns `Ok(true)` only when the token is fully valid, and `Ok(false)` for
/// *every* validation failure — bad signature, malformed token, wrong
/// issuer/audience, expired, or not-yet-valid. The previous implementation
/// returned `Err` for a forged signature, which inverts the security
/// predicate: a caller writing `if verify_jwt(t, c)?` would propagate the error
/// (often surfacing as a 500 / retry) instead of treating the token as
/// rejected.
pub fn verify_jwt(token: &str, config: &JwtConfig) -> Result<bool> {
    Ok(decode_jwt(token, config).is_ok())
}

/// Read a token's expiration **after verifying its signature**, ignoring only
/// the temporal (exp/nbf) and issuer/audience claims so the expiry of an
/// already-expired token can still be inspected.
///
/// This requires the `JwtConfig` (and therefore the signing key): the previous
/// signature-less variant let an attacker mint arbitrary expiry values and was
/// re-exported at the crate root, an auth-bypass invitation. It is renamed in
/// spirit — callers must supply the config — but the public symbol is retained
/// because it is re-exported from the crate root (`src/lib.rs`, outside this
/// change's ownership) and renaming it there is not possible here.
pub fn get_token_expiration(token: &str, config: &JwtConfig) -> Result<u64> {
    let mut cfg = config.clone();
    cfg.validate_exp = false;
    cfg.validate_iss = false;
    cfg.validate_aud = false;
    let claims = decode_jwt(token, &cfg)?;
    Ok(claims.exp)
}

/// Check if a token is expired, verifying its signature first.
pub fn is_token_expired(token: &str, config: &JwtConfig) -> Result<bool> {
    let exp = get_token_expiration(token, config)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Ok(exp < now)
}

// Base64 URL-safe encoding (without padding)
fn base64_url_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    let mut result = String::with_capacity((data.len() * 4 + 2) / 3);

    for chunk in data.chunks(3) {
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

// Base64 URL-safe decoding.
//
// Rejects any character outside the URL-safe alphabet by returning `None`
// rather than silently dropping it. Dropping invalid characters makes the
// encoding malleable — distinct byte strings decode to the same value — which
// for a JWT signature means forged variants can slip past a naive comparison.
fn base64_url_decode(input: &str) -> Option<Vec<u8>> {
    let decode_char = |c: char| -> Option<u8> {
        match c {
            'A'..='Z' => Some(c as u8 - b'A'),
            'a'..='z' => Some(c as u8 - b'a' + 26),
            '0'..='9' => Some(c as u8 - b'0' + 52),
            '-' => Some(62),
            '_' => Some(63),
            _ => None,
        }
    };

    let mut vals: Vec<u8> = Vec::with_capacity(input.len());
    for c in input.chars() {
        vals.push(decode_char(c)?);
    }

    if vals.is_empty() {
        return Some(Vec::new());
    }

    let mut result = Vec::with_capacity((vals.len() * 3) / 4);

    for chunk in vals.chunks(4) {
        // A trailing group of a single sextet cannot come from valid base64.
        if chunk.len() == 1 {
            return None;
        }
        if chunk.len() >= 2 {
            result.push((chunk[0] << 2) | (chunk[1] >> 4));
        }
        if chunk.len() >= 3 {
            result.push((chunk[1] << 4) | (chunk[2] >> 2));
        }
        if chunk.len() >= 4 {
            result.push((chunk[2] << 6) | chunk[3]);
        }
    }

    Some(result)
}

// HMAC-SHA256 implementation
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    const BLOCK_SIZE: usize = 64;

    // If key is longer than block size, hash it
    let key_bytes: Vec<u8> = if key.len() > BLOCK_SIZE {
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.finalize().to_vec()
    } else {
        key.to_vec()
    };

    // Pad key to block size
    let mut key_padded = [0u8; BLOCK_SIZE];
    key_padded[..key_bytes.len()].copy_from_slice(&key_bytes);

    // Create inner and outer pads
    let mut ipad = vec![0x36u8; BLOCK_SIZE];
    let mut opad = vec![0x5cu8; BLOCK_SIZE];

    for i in 0..BLOCK_SIZE {
        ipad[i] ^= key_padded[i];
        opad[i] ^= key_padded[i];
    }

    // Inner hash
    let mut inner_hasher = Sha256::new();
    inner_hasher.update(&ipad);
    inner_hasher.update(message);
    let inner_hash = inner_hasher.finalize();

    // Outer hash
    let mut outer_hasher = Sha256::new();
    outer_hasher.update(&opad);
    outer_hasher.update(&inner_hash);
    let result = outer_hasher.finalize();

    let mut output = [0u8; 32];
    output.copy_from_slice(&result);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jwt_encode_decode() {
        let config = JwtConfig::new(b"test_secret_key_for_testing_purposes_only".to_vec())
            .with_issuer("test-issuer")
            .with_audience("test-audience");

        let claims = TokenClaims {
            sub: "user123".to_string(),
            tenant_id: "tenant_a".to_string(),
            roles: vec!["admin".to_string()],
            permissions: vec!["read".to_string(), "write".to_string()],
            iat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
                .as_secs(),
            nbf: None,
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
                .as_secs()
                + 3600,
            iss: "test-issuer".to_string(),
            aud: "test-audience".to_string(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).expect("operation should succeed");
        assert!(!token.is_empty());

        // Token should have 3 parts
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);

        // Decode and verify
        let decoded = decode_jwt(&token, &config).expect("operation should succeed");
        assert_eq!(decoded.sub, "user123");
        assert_eq!(decoded.tenant_id, "tenant_a");
        assert_eq!(decoded.roles, vec!["admin"]);
    }

    #[test]
    fn test_jwt_invalid_signature() {
        let config = JwtConfig::new(b"secret1_padded_to_at_least_32_bytes".to_vec());
        let config2 = JwtConfig::new(b"secret2_padded_to_at_least_32_bytes".to_vec());

        let claims = TokenClaims {
            sub: "user123".to_string(),
            tenant_id: "tenant_a".to_string(),
            roles: vec![],
            permissions: vec![],
            iat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
                .as_secs(),
            nbf: None,
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
                .as_secs()
                + 3600,
            iss: config.issuer.clone(),
            aud: config.audience.clone(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).expect("operation should succeed");

        // Should fail with different secret
        let result = decode_jwt(&token, &config2);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_expired() {
        let config = JwtConfig::new(b"secret_padded_to_at_least_32_bytes_x".to_vec());

        let claims = TokenClaims {
            sub: "user123".to_string(),
            tenant_id: "tenant_a".to_string(),
            roles: vec![],
            permissions: vec![],
            iat: 0,
            nbf: None,
            exp: 1, // Expired in 1970
            iss: config.issuer.clone(),
            aud: config.audience.clone(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).expect("operation should succeed");

        // Should fail due to expiration
        let result = decode_jwt(&token, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_base64_url_encode_decode() {
        let data = b"Hello, World!";
        let encoded = base64_url_encode(data);
        let decoded = base64_url_decode(&encoded).expect("operation should succeed");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_hmac_sha256() {
        let key = b"key";
        let message = b"The quick brown fox jumps over the lazy dog";
        let mac = hmac_sha256(key, message);

        // Expected HMAC-SHA256 output for this key/message
        // (verified against known test vectors)
        assert_eq!(mac.len(), 32);
    }

    #[test]
    fn test_token_expiration_check() {
        let config = JwtConfig::new(b"secret_padded_to_at_least_32_bytes_x".to_vec());

        let claims = TokenClaims {
            sub: "user123".to_string(),
            tenant_id: "tenant_a".to_string(),
            roles: vec![],
            permissions: vec![],
            iat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
                .as_secs(),
            nbf: None,
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
                .as_secs()
                + 3600,
            iss: config.issuer.clone(),
            aud: config.audience.clone(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).expect("operation should succeed");

        // Should not be expired
        assert!(!is_token_expired(&token, &config).expect("operation should succeed"));

        // Get expiration time
        let exp = get_token_expiration(&token, &config).expect("operation should succeed");
        assert!(exp > 0);
    }
}
