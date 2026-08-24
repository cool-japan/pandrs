//! Secure credential management for PandRS
//!
//! This module provides secure storage, encryption, and management of
//! sensitive configuration data like API keys, passwords, and tokens.

use crate::core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

/// PBKDF2 iteration count for credential key derivation (OWASP 2023 guidance
/// for PBKDF2-HMAC-SHA256).
const CREDENTIAL_KDF_ITERATIONS: u32 = 600_000;
/// Accepted bounds when a config is loaded from an untrusted file: reject a
/// work factor low enough to be trivial or high enough to be a verification
/// DoS (e.g. `u32::MAX`).
const CREDENTIAL_KDF_MIN_ITERATIONS: u32 = 100_000;
const CREDENTIAL_KDF_MAX_ITERATIONS: u32 = 10_000_000;
/// AAD version tag bound into every credential's authenticated encryption.
const CREDENTIAL_AAD_VERSION: &str = "pandrs-cred-v1";

/// Credential store for managing sensitive data
#[derive(Clone)]
pub struct CredentialStore {
    /// Encrypted credential storage
    credentials: HashMap<String, EncryptedCredential>,
    /// Encryption key for credential protection
    encryption_key: Option<Vec<u8>>,
    /// Store configuration
    config: CredentialStoreConfig,
}

impl std::fmt::Debug for CredentialStore {
    /// Redacting `Debug`: never print the derived encryption key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CredentialStore")
            .field("credentials", &self.credentials.len())
            .field(
                "encryption_key",
                &self.encryption_key.as_ref().map(|_| "<redacted>"),
            )
            .field("config", &self.config)
            .finish()
    }
}

impl Drop for CredentialStore {
    /// Best-effort zeroization of the derived key on drop (pure-Rust equivalent
    /// of `zeroize`, which is outside this change's dependency ownership).
    fn drop(&mut self) {
        if let Some(ref mut key) = self.encryption_key {
            for b in key.iter_mut() {
                *b = 0;
            }
            std::hint::black_box(key.as_ptr());
        }
    }
}

/// Configuration for credential store
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialStoreConfig {
    /// Enable encryption for stored credentials
    pub encrypt_at_rest: bool,
    /// Encryption algorithm to use
    pub encryption_algorithm: String,
    /// Key derivation function
    pub key_derivation: String,
    /// Salt for key derivation
    pub salt: Vec<u8>,
    /// Iterations for key derivation
    pub iterations: u32,
    /// Credential file path
    pub file_path: Option<String>,
    /// Auto-save credentials to file
    pub auto_save: bool,
}

/// Encrypted credential container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedCredential {
    /// Encrypted credential data
    pub data: Vec<u8>,
    /// Initialization vector for encryption
    pub iv: Vec<u8>,
    /// Authentication tag for encryption
    pub tag: Vec<u8>,
    /// Credential metadata
    pub metadata: CredentialMetadata,
}

/// Metadata for credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialMetadata {
    /// Credential type (database, cloud, api_key, etc.)
    pub credential_type: String,
    /// When the credential was created
    pub created_at: String,
    /// When the credential was last accessed
    pub last_accessed: Option<String>,
    /// When the credential expires (if applicable)
    pub expires_at: Option<String>,
    /// Tags for organization
    pub tags: Vec<String>,
    /// Whether this credential is active
    pub active: bool,
}

/// Credential types supported by the system
#[derive(Clone, Serialize, Deserialize)]
pub enum CredentialType {
    /// Database connection credentials
    Database {
        username: String,
        password: String,
        host: String,
        port: u16,
        database: String,
    },
    /// Cloud storage credentials
    Cloud {
        provider: String,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
        region: Option<String>,
    },
    /// API key credentials
    ApiKey {
        key: String,
        secret: Option<String>,
        endpoint: Option<String>,
    },
    /// SSH key credentials
    SshKey {
        private_key: String,
        public_key: String,
        passphrase: Option<String>,
    },
    /// Generic credentials
    Generic { fields: HashMap<String, String> },
}

impl std::fmt::Debug for CredentialType {
    /// Redacting `Debug`: print only the credential category, never the
    /// secret material (passwords, keys, tokens).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            CredentialType::Database { .. } => "Database",
            CredentialType::Cloud { .. } => "Cloud",
            CredentialType::ApiKey { .. } => "ApiKey",
            CredentialType::SshKey { .. } => "SshKey",
            CredentialType::Generic { .. } => "Generic",
        };
        write!(f, "CredentialType::{} {{ <redacted> }}", kind)
    }
}

impl Default for CredentialStoreConfig {
    fn default() -> Self {
        Self {
            encrypt_at_rest: true,
            encryption_algorithm: "AES-256-GCM".to_string(),
            key_derivation: "PBKDF2".to_string(),
            salt: generate_random_bytes(32),
            iterations: CREDENTIAL_KDF_ITERATIONS,
            file_path: None,
            auto_save: false,
        }
    }
}

impl CredentialStore {
    /// Create a new credential store
    pub fn new(config: CredentialStoreConfig) -> Self {
        Self {
            credentials: HashMap::new(),
            encryption_key: None,
            config,
        }
    }

    /// Create credential store with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CredentialStoreConfig::default())
    }

    /// Initialize encryption key from password
    pub fn init_encryption(&mut self, password: &str) -> Result<()> {
        let key = derive_key(
            password.as_bytes(),
            &self.config.salt,
            self.config.iterations,
        )?;
        self.encryption_key = Some(key);
        Ok(())
    }

    /// Initialize encryption key from environment variable
    pub fn init_encryption_from_env(&mut self, env_var: &str) -> Result<()> {
        let password = env::var(env_var).map_err(|_| {
            Error::ConfigurationError(format!("Environment variable {} not found", env_var))
        })?;
        self.init_encryption(&password)
    }

    /// Store a credential
    pub fn store_credential(&mut self, name: &str, credential: CredentialType) -> Result<()> {
        let encrypted = self.encrypt_credential(name, &credential)?;
        self.credentials.insert(name.to_string(), encrypted);

        if self.config.auto_save {
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Retrieve a credential
    pub fn get_credential(&mut self, name: &str) -> Result<CredentialType> {
        let encrypted = self
            .credentials
            .get(name)
            .ok_or_else(|| Error::ConfigurationError(format!("Credential '{}' not found", name)))?;

        let credential = self.decrypt_credential(name, encrypted)?;

        // Update last accessed time
        if let Some(encrypted_mut) = self.credentials.get_mut(name) {
            encrypted_mut.metadata.last_accessed = Some(current_timestamp());
        }

        Ok(credential)
    }

    /// List all credential names
    pub fn list_credentials(&self) -> Vec<String> {
        self.credentials.keys().cloned().collect()
    }

    /// Check if credential exists
    pub fn has_credential(&self, name: &str) -> bool {
        self.credentials.contains_key(name)
    }

    /// Remove a credential
    pub fn remove_credential(&mut self, name: &str) -> Result<()> {
        self.credentials
            .remove(name)
            .ok_or_else(|| Error::ConfigurationError(format!("Credential '{}' not found", name)))?;

        if self.config.auto_save {
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Get credential metadata
    pub fn get_credential_metadata(&self, name: &str) -> Result<&CredentialMetadata> {
        let encrypted = self
            .credentials
            .get(name)
            .ok_or_else(|| Error::ConfigurationError(format!("Credential '{}' not found", name)))?;
        Ok(&encrypted.metadata)
    }

    /// Update credential metadata
    pub fn update_credential_metadata(
        &mut self,
        name: &str,
        metadata: CredentialMetadata,
    ) -> Result<()> {
        let encrypted = self
            .credentials
            .get_mut(name)
            .ok_or_else(|| Error::ConfigurationError(format!("Credential '{}' not found", name)))?;
        encrypted.metadata = metadata;

        if self.config.auto_save {
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Rotate the encryption key.
    ///
    /// Rebuilt to be all-or-nothing: the complete re-encrypted map is
    /// constructed under a freshly-derived key held in a local before any store
    /// state is mutated. The previous implementation cleared `self.credentials`
    /// and swapped in the new key *before* re-encrypting, so a failure midway
    /// (or a single credential that failed to re-encrypt) permanently destroyed
    /// every remaining credential.
    pub fn rotate_encryption_key(&mut self, new_password: &str) -> Result<()> {
        // 1. Decrypt everything with the CURRENT key.
        let mut decrypted: Vec<(String, CredentialType)> = Vec::new();
        for (name, encrypted) in &self.credentials {
            decrypted.push((name.clone(), self.decrypt_credential(name, encrypted)?));
        }

        // 2. Derive the NEW key locally without touching `self` yet.
        let new_salt = generate_random_bytes(32);
        let new_key = derive_key(new_password.as_bytes(), &new_salt, self.config.iterations)?;

        // 3. Re-encrypt everything into a LOCAL map under the new key.
        let mut new_credentials = HashMap::new();
        for (name, credential) in &decrypted {
            let encrypted = encrypt_credential_with(
                name,
                credential,
                Some(&new_key),
                self.config.encrypt_at_rest,
            )?;
            new_credentials.insert(name.clone(), encrypted);
        }

        // 4. Only now, after every step has succeeded, commit atomically.
        self.config.salt = new_salt;
        self.encryption_key = Some(new_key);
        self.credentials = new_credentials;

        if self.config.auto_save {
            self.save_to_file()?;
        }

        Ok(())
    }

    /// Save credentials to file
    pub fn save_to_file(&self) -> Result<()> {
        if let Some(file_path) = &self.config.file_path {
            let data = CredentialFileData {
                config: self.config.clone(),
                credentials: self.credentials.clone(),
            };

            let json = serde_json::to_string_pretty(&data).map_err(|e| {
                Error::ConfigurationError(format!("Failed to serialize credentials: {}", e))
            })?;

            // Create parent directory if needed, owner-only (0700) on unix.
            if let Some(parent) = Path::new(file_path).parent() {
                if !parent.as_os_str().is_empty() && !parent.exists() {
                    let mut builder = fs::DirBuilder::new();
                    builder.recursive(true);
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::DirBuilderExt;
                        builder.mode(0o700);
                    }
                    builder.create(parent).map_err(|e| {
                        Error::ConfigurationError(format!(
                            "Failed to create credential directory: {}",
                            e
                        ))
                    })?;
                }
            }

            write_secret_file(file_path, json.as_bytes())?;
        }

        Ok(())
    }

    /// Load credentials from file
    pub fn load_from_file(file_path: &str) -> Result<Self> {
        if !Path::new(file_path).exists() {
            return Err(Error::ConfigurationError(format!(
                "Credential file not found: {}",
                file_path
            )));
        }

        let json = fs::read_to_string(file_path).map_err(|e| {
            Error::ConfigurationError(format!("Failed to read credential file: {}", e))
        })?;

        let data: CredentialFileData = serde_json::from_str(&json).map_err(|e| {
            Error::ConfigurationError(format!("Failed to parse credential file: {}", e))
        })?;

        // Do not trust security parameters from the file blindly.
        validate_loaded_config(&data.config)?;

        Ok(Self {
            credentials: data.credentials,
            encryption_key: None,
            config: data.config,
        })
    }

    /// Export credentials (for backup/migration)
    pub fn export_credentials(&self, password: &str) -> Result<String> {
        // Create a temporary store for export
        let mut export_config = self.config.clone();
        export_config.salt = generate_random_bytes(32);

        let mut export_store = Self::new(export_config);
        export_store.init_encryption(password)?;

        // Re-encrypt all credentials with export key
        for (name, encrypted) in &self.credentials {
            let credential = self.decrypt_credential(name, encrypted)?;
            export_store.store_credential(name, credential)?;
        }

        // Clone rather than move: `CredentialStore` implements `Drop` (to
        // zeroize its key), which forbids moving fields out by value.
        let export_data = CredentialFileData {
            config: export_store.config.clone(),
            credentials: export_store.credentials.clone(),
        };

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| Error::ConfigurationError(format!("Failed to export credentials: {}", e)))
    }

    /// Import credentials from export
    pub fn import_credentials(&mut self, export_data: &str, password: &str) -> Result<()> {
        let data: CredentialFileData = serde_json::from_str(export_data).map_err(|e| {
            Error::ConfigurationError(format!("Failed to parse import data: {}", e))
        })?;

        // Do not trust security parameters from imported data blindly.
        validate_loaded_config(&data.config)?;

        // Create temporary store to decrypt import data
        let mut import_store = Self {
            credentials: data.credentials,
            encryption_key: None,
            config: data.config,
        };
        import_store.init_encryption(password)?;

        // Decrypt and re-encrypt with current store's key
        for name in import_store.list_credentials() {
            let credential = import_store.get_credential(&name)?;
            self.store_credential(&name, credential)?;
        }

        Ok(())
    }

    /// Encrypt a credential, binding the credential `name` as AAD.
    fn encrypt_credential(
        &self,
        name: &str,
        credential: &CredentialType,
    ) -> Result<EncryptedCredential> {
        encrypt_credential_with(
            name,
            credential,
            self.encryption_key.as_deref(),
            self.config.encrypt_at_rest,
        )
    }

    /// Decrypt a credential, verifying the credential `name` bound as AAD. A
    /// blob whose stored name does not match `name` fails authentication, so
    /// ciphertexts cannot be swapped between credential entries.
    fn decrypt_credential(
        &self,
        name: &str,
        encrypted: &EncryptedCredential,
    ) -> Result<CredentialType> {
        if !self.config.encrypt_at_rest {
            // Data is stored unencrypted
            let credential: CredentialType =
                serde_json::from_slice(&encrypted.data).map_err(|e| {
                    Error::ConfigurationError(format!("Failed to deserialize credential: {}", e))
                })?;
            return Ok(credential);
        }

        let key = self.encryption_key.as_ref().ok_or_else(|| {
            Error::ConfigurationError("Encryption key not initialized".to_string())
        })?;

        let aad = credential_aad(name);
        let plaintext = decrypt_data(&encrypted.data, &encrypted.iv, &encrypted.tag, key, &aad)?;

        let credential: CredentialType = serde_json::from_slice(&plaintext).map_err(|e| {
            Error::ConfigurationError(format!("Failed to deserialize credential: {}", e))
        })?;

        Ok(credential)
    }
}

/// Encrypt a credential with an explicit key (used by rotation, which must
/// encrypt under a freshly-derived key without mutating the store first).
fn encrypt_credential_with(
    name: &str,
    credential: &CredentialType,
    key: Option<&[u8]>,
    encrypt_at_rest: bool,
) -> Result<EncryptedCredential> {
    let metadata = CredentialMetadata {
        credential_type: get_credential_type_name(credential),
        created_at: current_timestamp(),
        last_accessed: None,
        expires_at: None,
        tags: Vec::new(),
        active: true,
    };

    if !encrypt_at_rest {
        let data = serde_json::to_vec(credential).map_err(|e| {
            Error::ConfigurationError(format!("Failed to serialize credential: {}", e))
        })?;
        return Ok(EncryptedCredential {
            data,
            iv: Vec::new(),
            tag: Vec::new(),
            metadata,
        });
    }

    let key =
        key.ok_or_else(|| Error::ConfigurationError("Encryption key not initialized".to_string()))?;

    let plaintext = serde_json::to_vec(credential)
        .map_err(|e| Error::ConfigurationError(format!("Failed to serialize credential: {}", e)))?;

    let aad = credential_aad(name);
    let (ciphertext, iv, tag) = encrypt_data(&plaintext, key, &aad)?;

    Ok(EncryptedCredential {
        data: ciphertext,
        iv,
        tag,
        metadata,
    })
}

/// Data structure for credential file storage
#[derive(Debug, Serialize, Deserialize)]
struct CredentialFileData {
    config: CredentialStoreConfig,
    credentials: HashMap<String, EncryptedCredential>,
}

// Helper functions for credential management

/// Generate random bytes for salt/IV generation
fn generate_random_bytes(len: usize) -> Vec<u8> {
    use scirs2_core::random::Rng;
    let mut bytes = vec![0u8; len];
    scirs2_core::random::rng().fill_bytes(&mut bytes);
    bytes
}

/// Validate a config loaded from an untrusted file. Security parameters must
/// not be taken on faith from the artifact: the PBKDF2 iteration count is
/// bounded so a crafted file cannot trigger a verification DoS (`u32::MAX`) or
/// a downgrade to a trivial work factor.
fn validate_loaded_config(config: &CredentialStoreConfig) -> Result<()> {
    if !(CREDENTIAL_KDF_MIN_ITERATIONS..=CREDENTIAL_KDF_MAX_ITERATIONS).contains(&config.iterations)
    {
        return Err(Error::ConfigurationError(format!(
            "Credential file specifies an unsafe PBKDF2 iteration count ({}); refusing to load",
            config.iterations
        )));
    }
    Ok(())
}

/// Write secret file contents with owner-only permissions.
///
/// On unix the file is created `0600` and, for a pre-existing file, its
/// permissions are tightened *while it is still empty* (before any secret bytes
/// are written), so there is no window in which the secret content is readable
/// under a looser umask.
fn write_secret_file(path: &str, contents: &[u8]) -> Result<()> {
    use std::io::Write;

    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let mut file = options
        .open(path)
        .map_err(|e| Error::ConfigurationError(format!("Failed to open credential file: {}", e)))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = file.set_permissions(fs::Permissions::from_mode(0o600));
    }

    file.write_all(contents).map_err(|e| {
        Error::ConfigurationError(format!("Failed to write credential file: {}", e))
    })?;

    Ok(())
}

/// Derive encryption key from password using PBKDF2
fn derive_key(password: &[u8], salt: &[u8], iterations: u32) -> Result<Vec<u8>> {
    use pbkdf2::pbkdf2_hmac;
    use sha2::Sha256;

    let mut key = [0u8; 32]; // 256-bit key
    pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut key);
    Ok(key.to_vec())
}

/// Additional authenticated data binding a ciphertext to its credential name
/// and a format version. Without this, AES-GCM ciphertext blobs are
/// interchangeable between credential names — an attacker with write access to
/// the store file could swap the ciphertext of "readonly-db" for "admin-db" and
/// the tag would still verify.
fn credential_aad(name: &str) -> Vec<u8> {
    format!("{}:{}", CREDENTIAL_AAD_VERSION, name).into_bytes()
}

/// Encrypt data using AES-256-GCM with the given additional authenticated data.
fn encrypt_data(plaintext: &[u8], key: &[u8], aad: &[u8]) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    use aes_gcm::aead::inout::InOutBuf;
    use aes_gcm::{AeadInOut, Aes256Gcm, Key, KeyInit, Nonce};

    let key_arr = Key::<Aes256Gcm>::try_from(key)
        .map_err(|e| Error::ConfigurationError(format!("Invalid key length: {}", e)))?;
    let cipher = Aes256Gcm::new(&key_arr);
    let nonce_bytes = generate_random_bytes(12); // 96-bit nonce for GCM
    let nonce = Nonce::try_from(nonce_bytes.as_slice())
        .map_err(|e| Error::ConfigurationError(format!("Invalid nonce length: {}", e)))?;

    let mut buffer = plaintext.to_vec();
    let tag = cipher
        .encrypt_inout_detached(&nonce, aad, InOutBuf::from(buffer.as_mut_slice()))
        .map_err(|e| Error::ConfigurationError(format!("Encryption failed: {}", e)))?;

    Ok((buffer, nonce_bytes, tag.to_vec()))
}

/// Decrypt data using AES-256-GCM, verifying the additional authenticated data.
fn decrypt_data(
    ciphertext: &[u8],
    iv: &[u8],
    tag: &[u8],
    key: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>> {
    use aes_gcm::aead::inout::InOutBuf;
    use aes_gcm::{AeadInOut, Aes256Gcm, Key, KeyInit, Nonce, Tag};

    let key_arr = Key::<Aes256Gcm>::try_from(key)
        .map_err(|e| Error::ConfigurationError(format!("Invalid key length: {}", e)))?;
    let cipher = Aes256Gcm::new(&key_arr);
    let nonce = Nonce::try_from(iv)
        .map_err(|e| Error::ConfigurationError(format!("Invalid nonce length: {}", e)))?;
    let tag = Tag::try_from(tag)
        .map_err(|e| Error::ConfigurationError(format!("Invalid tag length: {}", e)))?;

    let mut buffer = ciphertext.to_vec();
    cipher
        .decrypt_inout_detached(&nonce, aad, InOutBuf::from(buffer.as_mut_slice()), &tag)
        .map_err(|e| Error::ConfigurationError(format!("Decryption failed: {}", e)))?;

    Ok(buffer)
}

/// Get credential type name for metadata
fn get_credential_type_name(credential: &CredentialType) -> String {
    match credential {
        CredentialType::Database { .. } => "database".to_string(),
        CredentialType::Cloud { .. } => "cloud".to_string(),
        CredentialType::ApiKey { .. } => "api_key".to_string(),
        CredentialType::SshKey { .. } => "ssh_key".to_string(),
        CredentialType::Generic { .. } => "generic".to_string(),
    }
}

/// Get current timestamp as ISO string
fn current_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("operation should succeed")
        .as_secs();

    // Simple timestamp format (in production, use proper datetime formatting)
    format!("{}", timestamp)
}

/// Credential builder for easy credential creation
pub struct CredentialBuilder {
    credential_type: Option<CredentialType>,
    metadata: CredentialMetadata,
}

impl CredentialBuilder {
    pub fn new() -> Self {
        Self {
            credential_type: None,
            metadata: CredentialMetadata {
                credential_type: String::new(),
                created_at: current_timestamp(),
                last_accessed: None,
                expires_at: None,
                tags: Vec::new(),
                active: true,
            },
        }
    }

    pub fn database(
        mut self,
        username: &str,
        password: &str,
        host: &str,
        port: u16,
        database: &str,
    ) -> Self {
        self.credential_type = Some(CredentialType::Database {
            username: username.to_string(),
            password: password.to_string(),
            host: host.to_string(),
            port,
            database: database.to_string(),
        });
        self.metadata.credential_type = "database".to_string();
        self
    }

    pub fn cloud_aws(mut self, access_key: &str, secret_key: &str, region: Option<&str>) -> Self {
        self.credential_type = Some(CredentialType::Cloud {
            provider: "aws".to_string(),
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            session_token: None,
            region: region.map(|s| s.to_string()),
        });
        self.metadata.credential_type = "cloud".to_string();
        self
    }

    pub fn api_key(mut self, key: &str, secret: Option<&str>, endpoint: Option<&str>) -> Self {
        self.credential_type = Some(CredentialType::ApiKey {
            key: key.to_string(),
            secret: secret.map(|s| s.to_string()),
            endpoint: endpoint.map(|s| s.to_string()),
        });
        self.metadata.credential_type = "api_key".to_string();
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.metadata.tags = tags;
        self
    }

    pub fn with_expiry(mut self, expires_at: &str) -> Self {
        self.metadata.expires_at = Some(expires_at.to_string());
        self
    }

    pub fn build(self) -> Result<CredentialType> {
        self.credential_type
            .ok_or_else(|| Error::ConfigurationError("Credential type not specified".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credential_store_basic_operations() {
        let mut store = CredentialStore::with_defaults();
        store
            .init_encryption("test_password")
            .expect("operation should succeed");

        let credential = CredentialBuilder::new()
            .database("user", "pass", "localhost", 5432, "mydb")
            .build()
            .expect("operation should succeed");

        // Store credential
        store
            .store_credential("db1", credential)
            .expect("operation should succeed");

        // Check existence
        assert!(store.has_credential("db1"));
        assert!(!store.has_credential("db2"));

        // Retrieve credential
        let retrieved = store
            .get_credential("db1")
            .expect("operation should succeed");
        match retrieved {
            CredentialType::Database {
                username,
                password,
                host,
                port,
                database,
            } => {
                assert_eq!(username, "user");
                assert_eq!(password, "pass");
                assert_eq!(host, "localhost");
                assert_eq!(port, 5432);
                assert_eq!(database, "mydb");
            }
            _ => panic!("Wrong credential type retrieved"),
        }

        // Remove credential
        store
            .remove_credential("db1")
            .expect("operation should succeed");
        assert!(!store.has_credential("db1"));
    }

    #[test]
    fn test_credential_builder() {
        let db_cred = CredentialBuilder::new()
            .database("admin", "secret123", "db.example.com", 5432, "production")
            .with_tags(vec!["production".to_string(), "primary".to_string()])
            .build()
            .expect("operation should succeed");

        match db_cred {
            CredentialType::Database {
                username,
                password,
                host,
                port,
                database,
            } => {
                assert_eq!(username, "admin");
                assert_eq!(password, "secret123");
                assert_eq!(host, "db.example.com");
                assert_eq!(port, 5432);
                assert_eq!(database, "production");
            }
            _ => panic!("Wrong credential type"),
        }

        let aws_cred = CredentialBuilder::new()
            .cloud_aws("AKIATEST", "secret", Some("us-west-2"))
            .build()
            .expect("operation should succeed");

        match aws_cred {
            CredentialType::Cloud {
                provider,
                access_key,
                secret_key,
                region,
                ..
            } => {
                assert_eq!(provider, "aws");
                assert_eq!(access_key, "AKIATEST");
                assert_eq!(secret_key, "secret");
                assert_eq!(region, Some("us-west-2".to_string()));
            }
            _ => panic!("Wrong credential type"),
        }
    }
}
