//! Configuration loading utilities
//!
//! This module handles loading configuration from various sources with proper
//! precedence and validation.

use super::*;
use crate::core::error::{Error, Result};
use std::env;
use std::fs;
use std::path::Path;

/// Load configuration from environment variables
pub fn load_from_env() -> Result<PandRSConfig> {
    let mut config = PandRSConfig::default();

    // Database configuration
    if let Ok(url) = env::var("PANDRS_DB_URL") {
        config.database.default_url = Some(url);
    }

    if let Ok(pool_size) = env::var("PANDRS_DB_POOL_SIZE") {
        config.database.pool.max_connections = pool_size.parse().map_err(|e| {
            Error::ConfigurationError(format!("Invalid PANDRS_DB_POOL_SIZE: {}", e))
        })?;
    }

    if let Ok(timeout) = env::var("PANDRS_DB_TIMEOUT") {
        config.database.timeouts.query = timeout
            .parse()
            .map_err(|e| Error::ConfigurationError(format!("Invalid PANDRS_DB_TIMEOUT: {}", e)))?;
    }

    if let Ok(ssl_mode) = env::var("PANDRS_DB_SSL_MODE") {
        config.database.ssl.mode = ssl_mode;
        config.database.ssl.enabled = true;
    }

    // AWS configuration
    if let Ok(region) = env::var("AWS_DEFAULT_REGION").or_else(|_| env::var("AWS_REGION")) {
        config.cloud.aws.region = Some(region);
    }

    if let Ok(access_key) = env::var("AWS_ACCESS_KEY_ID") {
        config.cloud.aws.access_key_id = Some(access_key);
    }

    if let Ok(secret_key) = env::var("AWS_SECRET_ACCESS_KEY") {
        config.cloud.aws.secret_access_key = Some(secret_key);
    }

    if let Ok(session_token) = env::var("AWS_SESSION_TOKEN") {
        config.cloud.aws.session_token = Some(session_token);
    }

    if let Ok(profile) = env::var("AWS_PROFILE") {
        config.cloud.aws.profile = Some(profile);
    }

    // GCP configuration
    if let Ok(project_id) = env::var("GOOGLE_CLOUD_PROJECT").or_else(|_| env::var("GCP_PROJECT_ID"))
    {
        config.cloud.gcp.project_id = Some(project_id);
    }

    if let Ok(credentials) = env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        config.cloud.gcp.service_account_key = Some(credentials);
    }

    // Azure configuration
    if let Ok(account) = env::var("AZURE_STORAGE_ACCOUNT") {
        config.cloud.azure.account_name = Some(account);
    }

    if let Ok(key) = env::var("AZURE_STORAGE_KEY") {
        config.cloud.azure.account_key = Some(key);
    }

    if let Ok(sas) = env::var("AZURE_STORAGE_SAS_TOKEN") {
        config.cloud.azure.sas_token = Some(sas);
    }

    // Performance configuration
    if let Ok(threads) = env::var("PANDRS_WORKER_THREADS") {
        config.performance.threading.worker_threads = threads.parse().map_err(|e| {
            Error::ConfigurationError(format!("Invalid PANDRS_WORKER_THREADS: {}", e))
        })?;
    }

    if let Ok(parallel) = env::var("PANDRS_PARALLEL_ENABLED") {
        config.performance.threading.parallel_enabled = parallel.parse().map_err(|e| {
            Error::ConfigurationError(format!("Invalid PANDRS_PARALLEL_ENABLED: {}", e))
        })?;
    }

    if let Ok(memory_limit) = env::var("PANDRS_MEMORY_LIMIT") {
        config.performance.memory.limit = memory_limit.parse().map_err(|e| {
            Error::ConfigurationError(format!("Invalid PANDRS_MEMORY_LIMIT: {}", e))
        })?;
    }

    if let Ok(jit_enabled) = env::var("PANDRS_JIT_ENABLED") {
        config.performance.jit.enabled = jit_enabled
            .parse()
            .map_err(|e| Error::ConfigurationError(format!("Invalid PANDRS_JIT_ENABLED: {}", e)))?;
    }

    // Logging configuration
    if let Ok(log_level) = env::var("PANDRS_LOG_LEVEL").or_else(|_| env::var("RUST_LOG")) {
        config.logging.level = log_level;
    }

    if let Ok(log_file) = env::var("PANDRS_LOG_FILE") {
        config.logging.file_path = Some(log_file);
    }

    // Security configuration
    if let Ok(encryption) = env::var("PANDRS_ENCRYPTION_ENABLED") {
        config.security.encryption.enabled = encryption.parse().map_err(|e| {
            Error::ConfigurationError(format!("Invalid PANDRS_ENCRYPTION_ENABLED: {}", e))
        })?;
    }

    if let Ok(audit) = env::var("PANDRS_AUDIT_ENABLED") {
        config.security.audit.enabled = audit.parse().map_err(|e| {
            Error::ConfigurationError(format!("Invalid PANDRS_AUDIT_ENABLED: {}", e))
        })?;
    }

    Ok(config)
}

/// Load configuration from a file (YAML or TOML based on extension)
pub fn load_from_file(path: &Path) -> Result<PandRSConfig> {
    if !path.exists() {
        return Err(Error::ConfigurationError(format!(
            "Configuration file not found: {}",
            path.display()
        )));
    }

    let contents = fs::read_to_string(path).map_err(|e| {
        Error::ConfigurationError(format!(
            "Failed to read config file {}: {}",
            path.display(),
            e
        ))
    })?;

    match path.extension().and_then(|ext| ext.to_str()) {
        Some("yaml") | Some("yml") => load_from_yaml(&contents),
        Some("toml") => load_from_toml(&contents),
        Some(ext) => Err(Error::ConfigurationError(format!(
            "Unsupported config file format: {}",
            ext
        ))),
        None => {
            // Try to parse as YAML first, then TOML
            load_from_yaml(&contents).or_else(|_| load_from_toml(&contents))
        }
    }
}

/// Load configuration from YAML string
pub fn load_from_yaml(yaml: &str) -> Result<PandRSConfig> {
    serde_yaml::from_str(yaml)
        .map_err(|e| Error::ConfigurationError(format!("Failed to parse YAML config: {}", e)))
}

/// Load configuration from TOML string
pub fn load_from_toml(toml: &str) -> Result<PandRSConfig> {
    toml::from_str(toml)
        .map_err(|e| Error::ConfigurationError(format!("Failed to parse TOML config: {}", e)))
}

/// Load configuration with precedence: defaults -> file -> environment
pub fn load_with_precedence<P: AsRef<Path>>(config_file: Option<P>) -> Result<PandRSConfig> {
    // Start with defaults
    let mut config = PandRSConfig::default();

    // Load from file if provided
    if let Some(file_path) = config_file {
        let file_config = load_from_file(file_path.as_ref())?;
        config.merge(&file_config);
    }

    // Load from environment (highest precedence)
    let env_config = load_from_env()?;
    config.merge(&env_config);

    // Validate final configuration
    config.validate()?;

    Ok(config)
}

/// Save configuration to a file
pub fn save_to_file(config: &PandRSConfig, path: &Path) -> Result<()> {
    let contents = match path.extension().and_then(|ext| ext.to_str()) {
        Some("yaml") | Some("yml") => config.to_yaml()?,
        Some("toml") => config.to_toml()?,
        Some(ext) => {
            return Err(Error::ConfigurationError(format!(
                "Unsupported config file format: {}",
                ext
            )))
        }
        None => config.to_yaml()?, // Default to YAML
    };

    // Create parent directory if it doesn't exist
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| {
                Error::ConfigurationError(format!(
                    "Failed to create config directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
    }

    fs::write(path, contents).map_err(|e| {
        Error::ConfigurationError(format!(
            "Failed to write config file {}: {}",
            path.display(),
            e
        ))
    })
}

/// Get configuration file paths in order of precedence
pub fn get_config_file_paths() -> Vec<std::path::PathBuf> {
    let mut paths = Vec::new();

    // Current directory
    paths.push("pandrs.yml".into());
    paths.push("pandrs.yaml".into());
    paths.push("pandrs.toml".into());

    // User config directory
    if let Some(config_dir) = dirs::config_dir() {
        let pandrs_dir = config_dir.join("pandrs");
        paths.push(pandrs_dir.join("config.yml"));
        paths.push(pandrs_dir.join("config.yaml"));
        paths.push(pandrs_dir.join("config.toml"));
    }

    // System config directory
    paths.push("/etc/pandrs/config.yml".into());
    paths.push("/etc/pandrs/config.yaml".into());
    paths.push("/etc/pandrs/config.toml".into());

    // Environment variable override
    if let Ok(config_path) = env::var("PANDRS_CONFIG_FILE") {
        paths.insert(0, config_path.into());
    }

    paths
}

/// Auto-discover and load configuration file
pub fn auto_load() -> Result<PandRSConfig> {
    let config_paths = get_config_file_paths();

    for path in config_paths {
        if path.exists() {
            return load_with_precedence(Some(path));
        }
    }

    // No config file found, load from environment and defaults
    load_with_precedence::<&Path>(None)
}

/// Create a sample configuration file
pub fn create_sample_config<P: AsRef<Path>>(path: P) -> Result<()> {
    let sample_config = create_sample_config_content();
    save_to_file(&sample_config, path.as_ref())
}

/// Create sample configuration with documentation
fn create_sample_config_content() -> PandRSConfig {
    let mut config = PandRSConfig::default();

    // Set some reasonable defaults for a sample config
    config.database.default_url = Some("postgresql://localhost:5432/mydb".to_string());
    config.database.pool.max_connections = 20;
    config.database.ssl.enabled = true;
    config.database.ssl.mode = "require".to_string();

    config.cloud.aws.region = Some("us-west-2".to_string());
    config.cloud.aws.use_instance_metadata = true;
    config.cloud.gcp.use_default_credentials = true;
    config.cloud.azure.use_managed_identity = true;

    config.performance.threading.worker_threads = 0; // Auto-detect
    config.performance.memory.limit = 2 * 1024 * 1024 * 1024; // 2GB
    config.performance.jit.enabled = true;

    config.logging.level = "info".to_string();
    config.logging.console = true;

    config.security.encryption.enabled = true;
    config.security.audit.enabled = true;

    config
}

/// Reload configuration dynamically
pub fn reload_config(current_config: &mut PandRSConfig) -> Result<bool> {
    let new_config = auto_load()?;

    // Check if configuration has changed
    let current_yaml = current_config.to_yaml()?;
    let new_yaml = new_config.to_yaml()?;

    if current_yaml != new_yaml {
        *current_config = new_config;
        Ok(true) // Configuration changed
    } else {
        Ok(false) // No change
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;
    use tempfile::tempdir;

    // Mutex to serialize tests that modify environment variables
    static ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_load_from_env() {
        // Lock to prevent concurrent environment variable tests
        let _lock = ENV_TEST_MUTEX.lock().unwrap();

        // Save original environment variables
        let orig_db_url = env::var("PANDRS_DB_URL").ok();
        let orig_pool_size = env::var("PANDRS_DB_POOL_SIZE").ok();
        let orig_aws_region = env::var("AWS_DEFAULT_REGION").ok();
        let orig_log_level = env::var("PANDRS_LOG_LEVEL").ok();

        // Clean up any pre-existing environment variables first
        env::remove_var("PANDRS_DB_URL");
        env::remove_var("PANDRS_DB_POOL_SIZE");
        env::remove_var("AWS_DEFAULT_REGION");
        env::remove_var("PANDRS_LOG_LEVEL");

        // Set test environment variables
        env::set_var("PANDRS_DB_URL", "postgresql://test:test@localhost/test");
        env::set_var("PANDRS_DB_POOL_SIZE", "15");
        env::set_var("AWS_DEFAULT_REGION", "us-east-1");
        env::set_var("PANDRS_LOG_LEVEL", "debug");

        let config = load_from_env().unwrap();

        assert_eq!(
            config.database.default_url,
            Some("postgresql://test:test@localhost/test".to_string())
        );
        assert_eq!(config.database.pool.max_connections, 15);
        assert_eq!(config.cloud.aws.region, Some("us-east-1".to_string()));
        assert_eq!(config.logging.level, "debug");

        // Clean up and restore original values
        env::remove_var("PANDRS_DB_URL");
        env::remove_var("PANDRS_DB_POOL_SIZE");
        env::remove_var("AWS_DEFAULT_REGION");
        env::remove_var("PANDRS_LOG_LEVEL");

        // Restore original environment variables if they existed
        if let Some(val) = orig_db_url {
            env::set_var("PANDRS_DB_URL", val);
        }
        if let Some(val) = orig_pool_size {
            env::set_var("PANDRS_DB_POOL_SIZE", val);
        }
        if let Some(val) = orig_aws_region {
            env::set_var("AWS_DEFAULT_REGION", val);
        }
        if let Some(val) = orig_log_level {
            env::set_var("PANDRS_LOG_LEVEL", val);
        }
    }

    #[test]
    fn test_load_from_yaml() {
        let yaml = r#"
database:
  default_url: "postgresql://localhost/test"
  pool:
    max_connections: 25
    min_idle: 2
    max_lifetime: 3600
    idle_timeout: 600
    acquire_timeout: 30
  timeouts:
    query: 30
    connection: 10
    transaction: 60
  ssl:
    enabled: false
    mode: "prefer"
  parameters: {}
cloud:
  aws:
    region: "us-west-1"
    endpoint_url: ""
    access_key_id: ""
    secret_access_key: ""
    session_token: ""
    profile: "default"
    use_instance_metadata: false
  gcp:
    project_id: ""
    service_account_key: ""
    use_default_credentials: true
    endpoint_url: ""
  azure:
    account_name: ""
    account_key: ""
    sas_token: ""
    use_managed_identity: false
    endpoint_url: ""
  default_provider: "aws"
  global:
    timeout: 300
    max_retries: 3
    retry_backoff: 2.0
    compression: true
    buffer_size: 65536
performance:
  threading:
    worker_threads: 0
    parallel_enabled: true
    parallel_batch_size: 1000
    stack_size: 2097152
  memory:
    limit: 0
    enable_mmap: true
    string_pool:
      enabled: true
      max_size: 104857600
      cleanup_threshold: 0.8
    gc:
      auto_gc: true
      trigger_mb: 512
      aggressive: false
  caching:
    enabled: true
    size_limit: 104857600
    ttl: 3600
    cleanup_interval: 300
  jit:
    enabled: false
    threshold: 100
    simd_enabled: true
    cache_functions: true
security:
  encryption:
    enabled: false
    algorithm: "aes-256-gcm"
    key_derivation: "pbkdf2"
    salt: ""
  audit:
    enabled: false
    level: "info"
    file_path: ""
    include_sensitive: false
  access_control:
    enabled: false
    allowed_operations: []
    restricted_patterns: []
logging:
  level: "info"
  format: "text"
  file_path: ""
  console: true
  rotation:
    enabled: false
    max_size: 10485760
    max_files: 10
    compress: true
"#;

        let config = load_from_yaml(yaml).unwrap();
        assert_eq!(
            config.database.default_url,
            Some("postgresql://localhost/test".to_string())
        );
        assert_eq!(config.database.pool.max_connections, 25);
        assert_eq!(config.cloud.aws.region, Some("us-west-1".to_string()));
        assert!(!config.performance.jit.enabled);
    }

    #[test]
    fn test_save_and_load_file() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.yml");

        let original_config = create_sample_config_content();
        save_to_file(&original_config, &config_path).unwrap();

        let loaded_config = load_from_file(&config_path).unwrap();

        // Compare serialized versions to check equality
        assert_eq!(
            original_config.to_yaml().unwrap(),
            loaded_config.to_yaml().unwrap()
        );
    }

    #[test]
    fn test_precedence() {
        // Lock to prevent concurrent environment variable tests
        let _lock = ENV_TEST_MUTEX.lock().unwrap();

        // Save original environment variables
        let orig_pool_size = env::var("PANDRS_DB_POOL_SIZE").ok();
        let orig_aws_region = env::var("AWS_DEFAULT_REGION").ok();

        // Clean up environment variables first to avoid interference from other tests
        env::remove_var("PANDRS_DB_POOL_SIZE");
        env::remove_var("AWS_DEFAULT_REGION");

        let dir = tempdir().unwrap();
        let config_path = dir.path().join("precedence_test.yml");

        // Create a config file
        let yaml = r#"
database:
  default_url: "postgresql://localhost/test"
  pool:
    max_connections: 30
    min_idle: 2
    max_lifetime: 3600
    idle_timeout: 600
    acquire_timeout: 30
  timeouts:
    query: 30
    connection: 10
    transaction: 60
  ssl:
    enabled: false
    mode: "prefer"
  parameters: {}
cloud:
  aws:
    region: "eu-west-1"
    endpoint_url: ""
    access_key_id: ""
    secret_access_key: ""
    session_token: ""
    profile: "default"
    use_instance_metadata: false
  gcp:
    project_id: ""
    service_account_key: ""
    use_default_credentials: true
    endpoint_url: ""
  azure:
    account_name: ""
    account_key: ""
    sas_token: ""
    use_managed_identity: false
    endpoint_url: ""
  default_provider: "aws"
  global:
    timeout: 300
    max_retries: 3
    retry_backoff: 2.0
    compression: true
    buffer_size: 65536
performance:
  threading:
    worker_threads: 0
    parallel_enabled: true
    parallel_batch_size: 1000
    stack_size: 2097152
  memory:
    limit: 0
    enable_mmap: true
    string_pool:
      enabled: true
      max_size: 104857600
      cleanup_threshold: 0.8
    gc:
      auto_gc: true
      trigger_mb: 512
      aggressive: false
  caching:
    enabled: true
    size_limit: 104857600
    ttl: 3600
    cleanup_interval: 300
  jit:
    enabled: false
    threshold: 100
    simd_enabled: true
    cache_functions: true
security:
  encryption:
    enabled: false
    algorithm: "aes-256-gcm"
    key_derivation: "pbkdf2"
    salt: ""
  audit:
    enabled: false
    level: "info"
    file_path: ""
    include_sensitive: false
  access_control:
    enabled: false
    allowed_operations: []
    restricted_patterns: []
logging:
  level: "info"
  format: "text"
  file_path: ""
  console: true
  rotation:
    enabled: false
    max_size: 10485760
    max_files: 10
    compress: true
"#;
        std::fs::write(&config_path, yaml).unwrap();

        // Set environment variable that should override file
        env::set_var("PANDRS_DB_POOL_SIZE", "50");
        env::set_var("AWS_DEFAULT_REGION", "ap-southeast-1");

        let config = load_with_precedence(Some(&config_path)).unwrap();

        // Environment should override file
        assert_eq!(config.database.pool.max_connections, 50);
        assert_eq!(config.cloud.aws.region, Some("ap-southeast-1".to_string()));

        // Clean up and restore original values
        env::remove_var("PANDRS_DB_POOL_SIZE");
        env::remove_var("AWS_DEFAULT_REGION");

        // Restore original environment variables if they existed
        if let Some(val) = orig_pool_size {
            env::set_var("PANDRS_DB_POOL_SIZE", val);
        }
        if let Some(val) = orig_aws_region {
            env::set_var("AWS_DEFAULT_REGION", val);
        }
    }
}
