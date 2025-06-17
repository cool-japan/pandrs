//! Configuration validation utilities
//!
//! This module provides comprehensive validation for PandRS configuration,
//! ensuring that all settings are valid and compatible.

use super::*;
use crate::core::error::{Error, Result};
use std::net::SocketAddr;
use url::Url;

/// Validate the entire configuration
pub fn validate_config(config: &PandRSConfig) -> Result<()> {
    validate_database_config(&config.database)?;
    validate_cloud_config(&config.cloud)?;
    validate_performance_config(&config.performance)?;
    validate_security_config(&config.security)?;
    validate_logging_config(&config.logging)?;

    Ok(())
}

/// Validate database configuration
pub fn validate_database_config(config: &DatabaseConfig) -> Result<()> {
    // Validate connection URL if provided
    if let Some(url) = &config.default_url {
        validate_database_url(url)?;
    }

    // Validate connection pool settings
    validate_connection_pool(&config.pool)?;

    // Validate timeout settings
    validate_timeouts(&config.timeouts)?;

    // Validate SSL configuration
    validate_ssl_config(&config.ssl)?;

    Ok(())
}

/// Validate database URL format
fn validate_database_url(url: &str) -> Result<()> {
    let parsed_url = Url::parse(url)
        .map_err(|e| Error::ConfigurationError(format!("Invalid database URL '{}': {}", url, e)))?;

    match parsed_url.scheme() {
        "postgresql" | "postgres" | "sqlite" | "mysql" => Ok(()),
        scheme => Err(Error::ConfigurationError(format!(
            "Unsupported database scheme: {}",
            scheme
        ))),
    }
}

/// Validate connection pool configuration
fn validate_connection_pool(config: &ConnectionPoolConfig) -> Result<()> {
    if config.max_connections == 0 {
        return Err(Error::ConfigurationError(
            "max_connections must be greater than 0".to_string(),
        ));
    }

    if config.min_idle > config.max_connections {
        return Err(Error::ConfigurationError(
            "min_idle cannot be greater than max_connections".to_string(),
        ));
    }

    if config.acquire_timeout == 0 {
        return Err(Error::ConfigurationError(
            "acquire_timeout must be greater than 0".to_string(),
        ));
    }

    if config.max_lifetime == 0 {
        return Err(Error::ConfigurationError(
            "max_lifetime must be greater than 0".to_string(),
        ));
    }

    Ok(())
}

/// Validate timeout configuration
fn validate_timeouts(config: &TimeoutConfig) -> Result<()> {
    if config.query == 0 {
        return Err(Error::ConfigurationError(
            "query timeout must be greater than 0".to_string(),
        ));
    }

    if config.connection == 0 {
        return Err(Error::ConfigurationError(
            "connection timeout must be greater than 0".to_string(),
        ));
    }

    if config.transaction == 0 {
        return Err(Error::ConfigurationError(
            "transaction timeout must be greater than 0".to_string(),
        ));
    }

    // Warn about potentially problematic timeout values
    if config.query > 3600 {
        eprintln!(
            "Warning: Query timeout of {} seconds is very high",
            config.query
        );
    }

    if config.connection > 300 {
        eprintln!(
            "Warning: Connection timeout of {} seconds is very high",
            config.connection
        );
    }

    Ok(())
}

/// Validate SSL configuration
fn validate_ssl_config(config: &SslConfig) -> Result<()> {
    let valid_modes = [
        "disable",
        "allow",
        "prefer",
        "require",
        "verify-ca",
        "verify-full",
    ];

    if !valid_modes.contains(&config.mode.as_str()) {
        return Err(Error::ConfigurationError(format!(
            "Invalid SSL mode '{}'. Valid modes: {}",
            config.mode,
            valid_modes.join(", ")
        )));
    }

    // Validate certificate files if provided
    if let Some(cert_file) = &config.cert_file {
        validate_file_path(cert_file, "SSL certificate")?;
    }

    if let Some(key_file) = &config.key_file {
        validate_file_path(key_file, "SSL key")?;
    }

    if let Some(ca_file) = &config.ca_file {
        validate_file_path(ca_file, "SSL CA certificate")?;
    }

    Ok(())
}

/// Validate cloud configuration
pub fn validate_cloud_config(config: &CloudConfig) -> Result<()> {
    validate_aws_config(&config.aws)?;
    validate_gcp_config(&config.gcp)?;
    validate_azure_config(&config.azure)?;
    validate_global_cloud_config(&config.global)?;

    // Validate default provider if specified
    if let Some(provider) = &config.default_provider {
        match provider.to_lowercase().as_str() {
            "aws" | "s3" | "gcp" | "gcs" | "azure" => Ok(()),
            _ => Err(Error::ConfigurationError(format!(
                "Invalid default cloud provider: {}",
                provider
            ))),
        }?;
    }

    Ok(())
}

/// Validate AWS configuration
fn validate_aws_config(config: &AwsConfig) -> Result<()> {
    // Validate region format if provided
    if let Some(region) = &config.region {
        validate_aws_region(region)?;
    }

    // Validate endpoint URL if provided and not empty
    if let Some(endpoint) = &config.endpoint_url {
        if !endpoint.is_empty() {
            validate_url(endpoint, "AWS endpoint")?;
        }
    }

    // Check for credential completeness
    let has_keys = config.access_key_id.is_some() && config.secret_access_key.is_some();
    let has_profile = config.profile.is_some();
    let has_instance_metadata = config.use_instance_metadata;

    if !has_keys && !has_profile && !has_instance_metadata {
        eprintln!("Warning: No AWS credentials configured. Set access keys, profile, or enable instance metadata.");
    }

    Ok(())
}

/// Validate AWS region format
fn validate_aws_region(region: &str) -> Result<()> {
    // Basic AWS region format validation
    let parts: Vec<&str> = region.split('-').collect();
    if parts.len() < 3 {
        return Err(Error::ConfigurationError(format!(
            "Invalid AWS region format: {}",
            region
        )));
    }

    Ok(())
}

/// Validate GCP configuration
fn validate_gcp_config(config: &GcpConfig) -> Result<()> {
    // Validate service account key file if provided and not empty
    if let Some(key_file) = &config.service_account_key {
        if !key_file.is_empty() {
            validate_file_path(key_file, "GCP service account key")?;
        }
    }

    // Validate endpoint URL if provided and not empty
    if let Some(endpoint) = &config.endpoint_url {
        if !endpoint.is_empty() {
            validate_url(endpoint, "GCP endpoint")?;
        }
    }

    // Check for credential configuration
    let has_key_file = config.service_account_key.is_some();
    let has_default_creds = config.use_default_credentials;

    if !has_key_file && !has_default_creds {
        eprintln!("Warning: No GCP credentials configured. Set service account key or enable default credentials.");
    }

    Ok(())
}

/// Validate Azure configuration
fn validate_azure_config(config: &AzureConfig) -> Result<()> {
    // Validate endpoint URL if provided and not empty
    if let Some(endpoint) = &config.endpoint_url {
        if !endpoint.is_empty() {
            validate_url(endpoint, "Azure endpoint")?;
        }
    }

    // Check for credential configuration
    let has_account_key = config.account_name.is_some() && config.account_key.is_some();
    let has_sas_token = config.sas_token.is_some();
    let has_managed_identity = config.use_managed_identity;

    if !has_account_key && !has_sas_token && !has_managed_identity {
        eprintln!("Warning: No Azure credentials configured. Set account key, SAS token, or enable managed identity.");
    }

    Ok(())
}

/// Validate global cloud configuration
fn validate_global_cloud_config(config: &GlobalCloudConfig) -> Result<()> {
    if config.timeout == 0 {
        return Err(Error::ConfigurationError(
            "Cloud timeout must be greater than 0".to_string(),
        ));
    }

    if config.retry_backoff <= 0.0 {
        return Err(Error::ConfigurationError(
            "Retry backoff must be greater than 0".to_string(),
        ));
    }

    if config.buffer_size == 0 {
        return Err(Error::ConfigurationError(
            "Buffer size must be greater than 0".to_string(),
        ));
    }

    // Warn about potentially problematic values
    if config.timeout > 3600 {
        eprintln!(
            "Warning: Cloud timeout of {} seconds is very high",
            config.timeout
        );
    }

    if config.max_retries > 10 {
        eprintln!(
            "Warning: Max retries of {} is very high",
            config.max_retries
        );
    }

    Ok(())
}

/// Validate performance configuration
pub fn validate_performance_config(config: &PerformanceConfig) -> Result<()> {
    validate_threading_config(&config.threading)?;
    validate_memory_config(&config.memory)?;
    validate_caching_config(&config.caching)?;
    validate_jit_config(&config.jit)?;

    Ok(())
}

/// Validate threading configuration
fn validate_threading_config(config: &ThreadingConfig) -> Result<()> {
    if config.parallel_batch_size == 0 {
        return Err(Error::ConfigurationError(
            "Parallel batch size must be greater than 0".to_string(),
        ));
    }

    if config.stack_size < 64 * 1024 {
        return Err(Error::ConfigurationError(
            "Stack size must be at least 64KB".to_string(),
        ));
    }

    // Warn about potentially problematic values
    if config.worker_threads > 128 {
        eprintln!(
            "Warning: Worker threads count of {} is very high",
            config.worker_threads
        );
    }

    Ok(())
}

/// Validate memory configuration
fn validate_memory_config(config: &MemoryConfig) -> Result<()> {
    validate_string_pool_config(&config.string_pool)?;
    validate_gc_config(&config.gc)?;

    // Warn about memory limit
    if config.limit > 0 && config.limit < 1024 * 1024 * 1024 {
        eprintln!(
            "Warning: Memory limit of {} bytes is very low",
            config.limit
        );
    }

    Ok(())
}

/// Validate string pool configuration
fn validate_string_pool_config(config: &StringPoolConfig) -> Result<()> {
    if config.max_size == 0 {
        return Err(Error::ConfigurationError(
            "String pool max size must be greater than 0".to_string(),
        ));
    }

    if config.cleanup_threshold <= 0.0 || config.cleanup_threshold > 1.0 {
        return Err(Error::ConfigurationError(
            "String pool cleanup threshold must be between 0 and 1".to_string(),
        ));
    }

    Ok(())
}

/// Validate garbage collection configuration
fn validate_gc_config(config: &GcConfig) -> Result<()> {
    if config.trigger_mb == 0 {
        return Err(Error::ConfigurationError(
            "GC trigger threshold must be greater than 0".to_string(),
        ));
    }

    if config.trigger_mb > 10 * 1024 {
        eprintln!(
            "Warning: GC trigger of {}MB is very high",
            config.trigger_mb
        );
    }

    Ok(())
}

/// Validate caching configuration
fn validate_caching_config(config: &CachingConfig) -> Result<()> {
    if config.size_limit == 0 {
        return Err(Error::ConfigurationError(
            "Cache size limit must be greater than 0".to_string(),
        ));
    }

    if config.ttl == 0 {
        return Err(Error::ConfigurationError(
            "Cache TTL must be greater than 0".to_string(),
        ));
    }

    if config.cleanup_interval == 0 {
        return Err(Error::ConfigurationError(
            "Cache cleanup interval must be greater than 0".to_string(),
        ));
    }

    Ok(())
}

/// Validate JIT configuration
fn validate_jit_config(config: &JitConfig) -> Result<()> {
    if config.threshold == 0 {
        return Err(Error::ConfigurationError(
            "JIT threshold must be greater than 0".to_string(),
        ));
    }

    if config.threshold > 10000 {
        eprintln!(
            "Warning: JIT threshold of {} is very high",
            config.threshold
        );
    }

    Ok(())
}

/// Validate security configuration
pub fn validate_security_config(config: &SecurityConfig) -> Result<()> {
    validate_encryption_config(&config.encryption)?;
    validate_audit_config(&config.audit)?;
    validate_access_control_config(&config.access_control)?;

    Ok(())
}

/// Validate encryption configuration
fn validate_encryption_config(config: &EncryptionConfig) -> Result<()> {
    let valid_algorithms = ["AES-256-GCM", "AES-128-GCM", "ChaCha20Poly1305"];

    if !valid_algorithms.contains(&config.algorithm.as_str()) {
        return Err(Error::ConfigurationError(format!(
            "Invalid encryption algorithm '{}'. Valid algorithms: {}",
            config.algorithm,
            valid_algorithms.join(", ")
        )));
    }

    let valid_kdf = ["PBKDF2", "Argon2", "scrypt"];

    if !valid_kdf.contains(&config.key_derivation.as_str()) {
        return Err(Error::ConfigurationError(format!(
            "Invalid key derivation method '{}'. Valid methods: {}",
            config.key_derivation,
            valid_kdf.join(", ")
        )));
    }

    Ok(())
}

/// Validate audit configuration
fn validate_audit_config(config: &AuditConfig) -> Result<()> {
    let valid_levels = ["trace", "debug", "info", "warn", "error"];

    if !valid_levels.contains(&config.level.as_str()) {
        return Err(Error::ConfigurationError(format!(
            "Invalid audit log level '{}'. Valid levels: {}",
            config.level,
            valid_levels.join(", ")
        )));
    }

    // Validate audit log file path if provided
    if let Some(file_path) = &config.file_path {
        validate_parent_dir(file_path, "audit log")?;
    }

    Ok(())
}

/// Validate access control configuration
fn validate_access_control_config(config: &AccessControlConfig) -> Result<()> {
    // Validate operation patterns
    for operation in &config.allowed_operations {
        if operation.is_empty() {
            return Err(Error::ConfigurationError(
                "Empty operation in allowed_operations".to_string(),
            ));
        }
    }

    for pattern in &config.restricted_patterns {
        if pattern.is_empty() {
            return Err(Error::ConfigurationError(
                "Empty pattern in restricted_patterns".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate logging configuration
pub fn validate_logging_config(config: &LoggingConfig) -> Result<()> {
    let valid_levels = ["trace", "debug", "info", "warn", "error"];

    if !valid_levels.contains(&config.level.as_str()) {
        return Err(Error::ConfigurationError(format!(
            "Invalid log level '{}'. Valid levels: {}",
            config.level,
            valid_levels.join(", ")
        )));
    }

    let valid_formats = ["json", "text", "compact"];

    if !valid_formats.contains(&config.format.as_str()) {
        return Err(Error::ConfigurationError(format!(
            "Invalid log format '{}'. Valid formats: {}",
            config.format,
            valid_formats.join(", ")
        )));
    }

    // Validate log file path if provided
    if let Some(file_path) = &config.file_path {
        validate_parent_dir(file_path, "log")?;
    }

    validate_log_rotation_config(&config.rotation)?;

    Ok(())
}

/// Validate log rotation configuration
fn validate_log_rotation_config(config: &LogRotationConfig) -> Result<()> {
    if config.max_size == 0 {
        return Err(Error::ConfigurationError(
            "Log rotation max size must be greater than 0".to_string(),
        ));
    }

    if config.max_files == 0 {
        return Err(Error::ConfigurationError(
            "Log rotation max files must be greater than 0".to_string(),
        ));
    }

    if config.max_files > 1000 {
        eprintln!(
            "Warning: Keeping {} log files is very high",
            config.max_files
        );
    }

    Ok(())
}

/// Validate URL format
fn validate_url(url: &str, context: &str) -> Result<()> {
    Url::parse(url).map_err(|e| {
        Error::ConfigurationError(format!("Invalid {} URL '{}': {}", context, url, e))
    })?;
    Ok(())
}

/// Validate file path exists
fn validate_file_path(path: &str, context: &str) -> Result<()> {
    if !std::path::Path::new(path).exists() {
        return Err(Error::ConfigurationError(format!(
            "{} file not found: {}",
            context, path
        )));
    }
    Ok(())
}

/// Validate parent directory exists or can be created
fn validate_parent_dir(path: &str, context: &str) -> Result<()> {
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            return Err(Error::ConfigurationError(format!(
                "{} directory does not exist: {}",
                context,
                parent.display()
            )));
        }
    }
    Ok(())
}

/// Validate network address format
fn _validate_network_address(addr: &str, context: &str) -> Result<()> {
    addr.parse::<SocketAddr>().map_err(|e| {
        Error::ConfigurationError(format!("Invalid {} address '{}': {}", context, addr, e))
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_database_url() {
        assert!(validate_database_url("postgresql://localhost:5432/test").is_ok());
        assert!(validate_database_url("sqlite:///path/to/db.sqlite").is_ok());
        assert!(validate_database_url("mysql://user:pass@localhost/db").is_ok());
        assert!(validate_database_url("invalid-scheme://localhost").is_err());
        assert!(validate_database_url("not-a-url").is_err());
    }

    #[test]
    fn test_validate_connection_pool() {
        let mut config = ConnectionPoolConfig::default();
        assert!(validate_connection_pool(&config).is_ok());

        config.max_connections = 0;
        assert!(validate_connection_pool(&config).is_err());

        config.max_connections = 10;
        config.min_idle = 15;
        assert!(validate_connection_pool(&config).is_err());
    }

    #[test]
    fn test_validate_aws_region() {
        assert!(validate_aws_region("us-west-2").is_ok());
        assert!(validate_aws_region("eu-central-1").is_ok());
        assert!(validate_aws_region("ap-southeast-1").is_ok());
        assert!(validate_aws_region("invalid").is_err());
        assert!(validate_aws_region("us-west").is_err());
    }

    #[test]
    fn test_validate_ssl_config() {
        let mut config = SslConfig::default();
        assert!(validate_ssl_config(&config).is_ok());

        config.mode = "invalid-mode".to_string();
        assert!(validate_ssl_config(&config).is_err());

        config.mode = "require".to_string();
        assert!(validate_ssl_config(&config).is_ok());
    }
}
