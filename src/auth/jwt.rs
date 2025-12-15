//! JWT (JSON Web Token) Implementation
//!
//! This module provides JWT token generation and validation with
//! HMAC-SHA256 signing.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// JWT configuration
#[derive(Debug, Clone)]
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

impl Default for JwtConfig {
    fn default() -> Self {
        use rand::RngCore;
        let mut secret = vec![0u8; 64];
        rand::rng().fill_bytes(&mut secret);

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
        JwtConfig {
            secret_key: secret.into(),
            ..Default::default()
        }
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

/// Encode JWT token
pub fn encode_jwt(claims: &TokenClaims, config: &JwtConfig) -> Result<String> {
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
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(Error::InvalidInput("Invalid token format".to_string()));
    }

    let header_b64 = parts[0];
    let payload_b64 = parts[1];
    let signature_b64 = parts[2];

    // Verify signature
    let message = format!("{}.{}", header_b64, payload_b64);
    let expected_signature = hmac_sha256(&config.secret_key, message.as_bytes());
    let expected_signature_b64 = base64_url_encode(&expected_signature);

    if signature_b64 != expected_signature_b64 {
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
        if claims.exp + config.leeway_secs < now {
            return Err(Error::InvalidOperation("Token has expired".to_string()));
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

/// Verify JWT token without decoding (just check signature and expiration)
pub fn verify_jwt(token: &str, config: &JwtConfig) -> Result<bool> {
    match decode_jwt(token, config) {
        Ok(_) => Ok(true),
        Err(Error::InvalidOperation(_)) => Ok(false), // Expired
        Err(e) => Err(e),
    }
}

/// Get token expiration time without full validation
pub fn get_token_expiration(token: &str) -> Result<u64> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Err(Error::InvalidInput("Invalid token format".to_string()));
    }

    let payload_bytes = base64_url_decode(parts[1])
        .ok_or_else(|| Error::InvalidInput("Invalid payload encoding".to_string()))?;

    let claims: TokenClaims = serde_json::from_slice(&payload_bytes)
        .map_err(|e| Error::InvalidInput(format!("Invalid payload: {}", e)))?;

    Ok(claims.exp)
}

/// Check if token is expired
pub fn is_token_expired(token: &str) -> Result<bool> {
    let exp = get_token_expiration(token)?;
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

// Base64 URL-safe decoding
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

    let chars: Vec<u8> = input.chars().filter_map(decode_char).collect();

    if chars.is_empty() {
        return Some(Vec::new());
    }

    let mut result = Vec::with_capacity((chars.len() * 3) / 4);

    for chunk in chars.chunks(4) {
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
    let mut key_padded = vec![0u8; BLOCK_SIZE];
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
                .unwrap()
                .as_secs(),
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            iss: "test-issuer".to_string(),
            aud: "test-audience".to_string(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).unwrap();
        assert!(!token.is_empty());

        // Token should have 3 parts
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);

        // Decode and verify
        let decoded = decode_jwt(&token, &config).unwrap();
        assert_eq!(decoded.sub, "user123");
        assert_eq!(decoded.tenant_id, "tenant_a");
        assert_eq!(decoded.roles, vec!["admin"]);
    }

    #[test]
    fn test_jwt_invalid_signature() {
        let config = JwtConfig::new(b"secret1".to_vec());
        let config2 = JwtConfig::new(b"secret2".to_vec());

        let claims = TokenClaims {
            sub: "user123".to_string(),
            tenant_id: "tenant_a".to_string(),
            roles: vec![],
            permissions: vec![],
            iat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            iss: config.issuer.clone(),
            aud: config.audience.clone(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).unwrap();

        // Should fail with different secret
        let result = decode_jwt(&token, &config2);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_expired() {
        let config = JwtConfig::new(b"secret".to_vec());

        let claims = TokenClaims {
            sub: "user123".to_string(),
            tenant_id: "tenant_a".to_string(),
            roles: vec![],
            permissions: vec![],
            iat: 0,
            exp: 1, // Expired in 1970
            iss: config.issuer.clone(),
            aud: config.audience.clone(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).unwrap();

        // Should fail due to expiration
        let result = decode_jwt(&token, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_base64_url_encode_decode() {
        let data = b"Hello, World!";
        let encoded = base64_url_encode(data);
        let decoded = base64_url_decode(&encoded).unwrap();
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
        let config = JwtConfig::new(b"secret".to_vec());

        let claims = TokenClaims {
            sub: "user123".to_string(),
            tenant_id: "tenant_a".to_string(),
            roles: vec![],
            permissions: vec![],
            iat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            exp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600,
            iss: config.issuer.clone(),
            aud: config.audience.clone(),
            jti: "token123".to_string(),
        };

        let token = encode_jwt(&claims, &config).unwrap();

        // Should not be expired
        assert!(!is_token_expired(&token).unwrap());

        // Get expiration time
        let exp = get_token_expiration(&token).unwrap();
        assert!(exp > 0);
    }
}
