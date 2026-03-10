//! # Cloud Storage Connectors
//!
//! This module provides direct connectivity to cloud storage services
//! including AWS S3, Google Cloud Storage, and Azure Blob Storage.
//!
//! When the `cloud-storage` feature is enabled, real implementations backed by
//! the `object_store` crate are used. Without the feature, the connectors still
//! compile but every operation returns `Error::NotImplemented`.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use std::collections::HashMap;

#[cfg(feature = "cloud-storage")]
use std::sync::Arc;

#[cfg(feature = "cloud-storage")]
use bytes::Bytes;

#[cfg(feature = "cloud-storage")]
use futures::StreamExt;

#[cfg(feature = "cloud-storage")]
use object_store::{
    path::Path as ObjectPath, GetResult, ObjectMeta, ObjectStore, ObjectStoreExt, PutPayload,
};

#[cfg(feature = "cloud-storage")]
use object_store::aws::AmazonS3Builder;

#[cfg(feature = "cloud-storage")]
use object_store::gcp::GoogleCloudStorageBuilder;

#[cfg(feature = "cloud-storage")]
use object_store::azure::MicrosoftAzureBuilder;

// ──────────────────────────────────────────────────────────────────────────────
// Configuration types (always available, no feature gate required)
// ──────────────────────────────────────────────────────────────────────────────

/// Cloud storage configuration
#[derive(Debug, Clone)]
pub struct CloudConfig {
    /// Service provider
    pub provider: CloudProvider,
    /// Authentication credentials
    pub credentials: CloudCredentials,
    /// Region or location
    pub region: Option<String>,
    /// Endpoint URL (for custom endpoints or MinIO)
    pub endpoint: Option<String>,
    /// Connection timeout in seconds
    pub timeout: Option<u64>,
    /// Additional parameters
    pub parameters: HashMap<String, String>,
}

/// Cloud storage providers
#[derive(Debug, Clone)]
pub enum CloudProvider {
    /// Amazon Web Services S3
    AWS,
    /// Google Cloud Storage
    GCS,
    /// Microsoft Azure Blob Storage
    Azure,
    /// MinIO (S3-compatible)
    MinIO,
}

/// Cloud authentication credentials
#[derive(Debug, Clone)]
pub enum CloudCredentials {
    /// AWS credentials
    AWS {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    /// Google Cloud Service Account
    GCS {
        service_account_key: String,
        project_id: String,
    },
    /// Azure credentials
    Azure {
        account_name: String,
        account_key: String,
    },
    /// Environment-based authentication (reads from env vars)
    Environment,
    /// Anonymous access
    Anonymous,
}

impl CloudConfig {
    /// Create new cloud configuration
    pub fn new(provider: CloudProvider, credentials: CloudCredentials) -> Self {
        Self {
            provider,
            credentials,
            region: None,
            endpoint: None,
            timeout: Some(300), // 5 minutes default
            parameters: HashMap::new(),
        }
    }

    /// Set region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set custom endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add parameter
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared data types
// ──────────────────────────────────────────────────────────────────────────────

/// Cloud object information
#[derive(Debug, Clone)]
pub struct CloudObject {
    pub key: String,
    pub size: u64,
    pub last_modified: Option<String>,
    pub etag: Option<String>,
    pub content_type: Option<String>,
}

/// Object metadata
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    pub size: u64,
    pub last_modified: Option<String>,
    pub content_type: Option<String>,
    pub etag: Option<String>,
    pub custom_metadata: HashMap<String, String>,
}

/// Supported file formats for cloud storage I/O
#[derive(Debug, Clone)]
pub enum FileFormat {
    CSV { delimiter: char, has_header: bool },
    Parquet,
    JSON,
    JSONL,
}

impl FileFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &str) -> Option<Self> {
        let extension = path.split('.').last()?.to_lowercase();
        match extension.as_str() {
            "csv" => Some(FileFormat::CSV {
                delimiter: ',',
                has_header: true,
            }),
            "parquet" | "pq" => Some(FileFormat::Parquet),
            "json" => Some(FileFormat::JSON),
            "jsonl" | "ndjson" => Some(FileFormat::JSONL),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Trait
// ──────────────────────────────────────────────────────────────────────────────

/// Generic cloud storage connector trait
#[allow(async_fn_in_trait)]
pub trait CloudConnector: Send + Sync {
    /// Connect to cloud storage
    async fn connect(&mut self, config: &CloudConfig) -> Result<()>;

    /// List objects in bucket/container
    async fn list_objects(&self, bucket: &str, prefix: Option<&str>) -> Result<Vec<CloudObject>>;

    /// Read DataFrame from cloud object (CSV, Parquet, etc.)
    async fn read_dataframe(
        &self,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<DataFrame>;

    /// Write DataFrame to cloud storage
    async fn write_dataframe(
        &self,
        df: &DataFrame,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<()>;

    /// Download object to local file
    async fn download_object(&self, bucket: &str, key: &str, local_path: &str) -> Result<()>;

    /// Upload local file to cloud storage
    async fn upload_object(&self, local_path: &str, bucket: &str, key: &str) -> Result<()>;

    /// Delete object
    async fn delete_object(&self, bucket: &str, key: &str) -> Result<()>;

    /// Get object metadata
    async fn get_object_metadata(&self, bucket: &str, key: &str) -> Result<ObjectMetadata>;

    /// Check if object exists
    async fn object_exists(&self, bucket: &str, key: &str) -> Result<bool>;

    /// Create bucket/container
    async fn create_bucket(&self, bucket: &str) -> Result<()>;

    /// Delete bucket/container
    async fn delete_bucket(&self, bucket: &str) -> Result<()>;
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared helpers (feature-gated)
// ──────────────────────────────────────────────────────────────────────────────

/// Convert an `ObjectMeta` returned by `object_store` into our `CloudObject`.
#[cfg(feature = "cloud-storage")]
fn meta_to_cloud_object(meta: ObjectMeta) -> CloudObject {
    CloudObject {
        key: meta.location.to_string(),
        size: meta.size as u64,
        last_modified: Some(meta.last_modified.to_rfc3339()),
        etag: meta.e_tag,
        content_type: None,
    }
}

/// Convert an `ObjectMeta` into our `ObjectMetadata`.
#[cfg(feature = "cloud-storage")]
fn meta_to_object_metadata(meta: ObjectMeta) -> ObjectMetadata {
    ObjectMetadata {
        size: meta.size as u64,
        last_modified: Some(meta.last_modified.to_rfc3339()),
        content_type: None,
        etag: meta.e_tag,
        custom_metadata: HashMap::new(),
    }
}

/// Serialize a `DataFrame` into a `PutPayload` using the requested `FileFormat`.
#[cfg(feature = "cloud-storage")]
fn df_to_payload(df: &DataFrame, format: &FileFormat) -> Result<PutPayload> {
    use std::io::Write as IoWrite;

    match format {
        FileFormat::CSV { has_header, .. } => {
            let tmp = tempfile::NamedTempFile::new().map_err(|e| Error::IoError(e.to_string()))?;
            let path = tmp.path().to_owned();
            let _ = has_header; // write_csv always writes header
            crate::io::write_csv(df, &path)?;
            let data = std::fs::read(&path).map_err(|e| Error::IoError(e.to_string()))?;
            Ok(PutPayload::from(Bytes::from(data)))
        }
        FileFormat::JSON | FileFormat::JSONL => {
            let tmp = tempfile::NamedTempFile::new().map_err(|e| Error::IoError(e.to_string()))?;
            let path = tmp.path().to_owned();
            crate::io::write_json(df, &path, crate::io::json::JsonOrient::Records)?;
            let data = std::fs::read(&path).map_err(|e| Error::IoError(e.to_string()))?;
            Ok(PutPayload::from(Bytes::from(data)))
        }
        FileFormat::Parquet => Err(Error::NotImplemented(
            "Parquet write to cloud requires the 'parquet' feature".to_string(),
        )),
    }
}

/// Download `Bytes` to a temp file and parse it as a `DataFrame`.
#[cfg(feature = "cloud-storage")]
fn bytes_to_df(data: Bytes, format: &FileFormat) -> Result<DataFrame> {
    use std::io::Write as IoWrite;

    let mut tmp = tempfile::NamedTempFile::new().map_err(|e| Error::IoError(e.to_string()))?;
    tmp.write_all(&data)
        .map_err(|e| Error::IoError(e.to_string()))?;
    tmp.flush().map_err(|e| Error::IoError(e.to_string()))?;
    let path = tmp.path().to_owned();

    match format {
        FileFormat::CSV { has_header, .. } => crate::io::read_csv(&path, *has_header),
        FileFormat::JSON | FileFormat::JSONL => crate::io::read_json(&path),
        FileFormat::Parquet => Err(Error::NotImplemented(
            "Parquet read from cloud requires the 'parquet' feature".to_string(),
        )),
    }
}

/// Build the full object path from `key`.
///
/// `object_store` paths do NOT include a bucket component — the store is
/// already scoped to a single bucket.
#[cfg(feature = "cloud-storage")]
fn make_path(key: &str) -> ObjectPath {
    ObjectPath::from(key)
}

// ──────────────────────────────────────────────────────────────────────────────
// S3 Connector
// ──────────────────────────────────────────────────────────────────────────────

/// AWS S3 connector
///
/// With the `cloud-storage` feature enabled this uses `object_store::aws`.
/// Without the feature every method returns `Error::NotImplemented`.
pub struct S3Connector {
    #[cfg(feature = "cloud-storage")]
    store: Option<Arc<dyn ObjectStore>>,
    #[cfg(not(feature = "cloud-storage"))]
    _phantom: std::marker::PhantomData<()>,
}

impl S3Connector {
    /// Create a new, unconnected S3 connector.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cloud-storage")]
            store: None,
            #[cfg(not(feature = "cloud-storage"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Build and connect immediately using the provided configuration.
    #[cfg(feature = "cloud-storage")]
    pub async fn connect_with_config(config: CloudConfig) -> Result<Self> {
        let mut connector = Self::new();
        connector.connect(&config).await?;
        Ok(connector)
    }

    /// Stub for no-feature builds.
    #[cfg(not(feature = "cloud-storage"))]
    pub async fn connect_with_config(_config: CloudConfig) -> Result<Self> {
        Err(Error::NotImplemented(
            "cloud-storage feature required for S3 connectivity".to_string(),
        ))
    }

    #[cfg(feature = "cloud-storage")]
    fn build_store(config: &CloudConfig, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        let mut builder = AmazonS3Builder::new().with_bucket_name(bucket);

        match &config.credentials {
            CloudCredentials::AWS {
                access_key_id,
                secret_access_key,
                session_token,
            } => {
                builder = builder
                    .with_access_key_id(access_key_id)
                    .with_secret_access_key(secret_access_key);
                if let Some(token) = session_token {
                    builder = builder.with_token(token);
                }
            }
            CloudCredentials::Environment => {
                let key_id = std::env::var("AWS_ACCESS_KEY_ID").unwrap_or_default();
                let secret = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap_or_default();
                builder = builder
                    .with_access_key_id(key_id)
                    .with_secret_access_key(secret);
                if let Ok(token) = std::env::var("AWS_SESSION_TOKEN") {
                    builder = builder.with_token(token);
                }
            }
            _ => {
                return Err(Error::InvalidOperation(
                    "Invalid credentials for S3: expected AWS or Environment credentials"
                        .to_string(),
                ));
            }
        }

        if let Some(region) = &config.region {
            builder = builder.with_region(region);
        }
        if let Some(endpoint) = &config.endpoint {
            builder = builder.with_endpoint(endpoint);
            // MinIO / custom endpoints require HTTP and path-style access
            builder = builder.with_allow_http(true);
        }

        let store = builder
            .build()
            .map_err(|e| Error::ConnectionError(format!("S3 build error: {e}")))?;
        Ok(Arc::new(store))
    }
}

impl Default for S3Connector {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudConnector for S3Connector {
    async fn connect(&mut self, config: &CloudConfig) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let placeholder_bucket = config
                .parameters
                .get("bucket")
                .cloned()
                .unwrap_or_else(|| "default".to_string());
            let store = Self::build_store(config, &placeholder_bucket)?;
            self.store = Some(store);
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = config;
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn list_objects(&self, bucket: &str, prefix: Option<&str>) -> Result<Vec<CloudObject>> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError(
                    "S3Connector not connected — call connect() first".to_string(),
                )
            })?;
            let prefix_path = prefix.map(ObjectPath::from);
            let results: Vec<_> = store.list(prefix_path.as_ref()).collect::<Vec<_>>().await;

            let mut objects = Vec::with_capacity(results.len());
            for item in results {
                let meta = item.map_err(|e| {
                    Error::IoError(format!("S3 list error for bucket '{}': {}", bucket, e))
                })?;
                objects.push(meta_to_cloud_object(meta));
            }
            Ok(objects)
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, prefix);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn read_dataframe(
        &self,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<DataFrame> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("S3Connector not connected".to_string()))?;
            let path = make_path(key);
            let result = store
                .get(&path)
                .await
                .map_err(|e| Error::IoError(format!("S3 get '{}/{}': {}", bucket, key, e)))?;
            let data: Bytes = result.bytes().await.map_err(|e| {
                Error::IoError(format!("S3 read bytes '{}/{}': {}", bucket, key, e))
            })?;
            bytes_to_df(data, &format)
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key, format);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn write_dataframe(
        &self,
        df: &DataFrame,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("S3Connector not connected".to_string()))?;
            let data = df_to_payload(df, &format)?;
            let path = make_path(key);
            store
                .put(&path, data.into())
                .await
                .map_err(|e| Error::IoError(format!("S3 put '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (df, bucket, key, format);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn download_object(&self, bucket: &str, key: &str, local_path: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("S3Connector not connected".to_string()))?;
            let path = make_path(key);
            let result = store
                .get(&path)
                .await
                .map_err(|e| Error::IoError(format!("S3 get '{}/{}': {}", bucket, key, e)))?;
            let data: Bytes = result
                .bytes()
                .await
                .map_err(|e| Error::IoError(format!("S3 read bytes: {}", e)))?;
            std::fs::write(local_path, &data).map_err(|e| Error::IoError(e.to_string()))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key, local_path);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn upload_object(&self, local_path: &str, bucket: &str, key: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("S3Connector not connected".to_string()))?;
            let data = std::fs::read(local_path).map_err(|e| Error::IoError(e.to_string()))?;
            let path = make_path(key);
            store
                .put(&path, Bytes::from(data).into())
                .await
                .map_err(|e| Error::IoError(format!("S3 put '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (local_path, bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("S3Connector not connected".to_string()))?;
            let path = make_path(key);
            store
                .delete(&path)
                .await
                .map_err(|e| Error::IoError(format!("S3 delete '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn get_object_metadata(&self, bucket: &str, key: &str) -> Result<ObjectMetadata> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("S3Connector not connected".to_string()))?;
            let path = make_path(key);
            let meta = store
                .head(&path)
                .await
                .map_err(|e| Error::IoError(format!("S3 head '{}/{}': {}", bucket, key, e)))?;
            Ok(meta_to_object_metadata(meta))
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn object_exists(&self, bucket: &str, key: &str) -> Result<bool> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("S3Connector not connected".to_string()))?;
            let path = make_path(key);
            match store.head(&path).await {
                Ok(_) => Ok(true),
                Err(object_store::Error::NotFound { .. }) => Ok(false),
                Err(e) => Err(Error::IoError(format!(
                    "S3 head '{}/{}': {}",
                    bucket, key, e
                ))),
            }
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn create_bucket(&self, _bucket: &str) -> Result<()> {
        // object_store does not expose a create-bucket API; bucket creation is
        // typically done via the cloud provider's console or CLI.
        Err(Error::NotImplemented(
            "Bucket creation is not supported via object_store — use the cloud provider CLI or \
             console"
                .to_string(),
        ))
    }

    async fn delete_bucket(&self, _bucket: &str) -> Result<()> {
        Err(Error::NotImplemented(
            "Bucket deletion is not supported via object_store — use the cloud provider CLI or \
             console"
                .to_string(),
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GCS Connector
// ──────────────────────────────────────────────────────────────────────────────

/// Google Cloud Storage connector
///
/// With the `cloud-storage` feature enabled this uses `object_store::gcp`.
pub struct GCSConnector {
    #[cfg(feature = "cloud-storage")]
    store: Option<Arc<dyn ObjectStore>>,
    #[cfg(not(feature = "cloud-storage"))]
    _phantom: std::marker::PhantomData<()>,
}

impl GCSConnector {
    /// Create a new, unconnected GCS connector.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cloud-storage")]
            store: None,
            #[cfg(not(feature = "cloud-storage"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Build and connect immediately using the provided configuration.
    #[cfg(feature = "cloud-storage")]
    pub async fn connect_with_config(config: CloudConfig) -> Result<Self> {
        let mut connector = Self::new();
        connector.connect(&config).await?;
        Ok(connector)
    }

    /// Stub for no-feature builds.
    #[cfg(not(feature = "cloud-storage"))]
    pub async fn connect_with_config(_config: CloudConfig) -> Result<Self> {
        Err(Error::NotImplemented(
            "cloud-storage feature required for GCS connectivity".to_string(),
        ))
    }

    #[cfg(feature = "cloud-storage")]
    fn build_store(config: &CloudConfig, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        let mut builder = GoogleCloudStorageBuilder::new().with_bucket_name(bucket);

        match &config.credentials {
            CloudCredentials::GCS {
                service_account_key,
                ..
            } => {
                builder = builder.with_service_account_key(service_account_key);
            }
            CloudCredentials::Environment => {
                // Use Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS env var)
                // object_store picks this up automatically.
            }
            _ => {
                return Err(Error::InvalidOperation(
                    "Invalid credentials for GCS: expected GCS or Environment credentials"
                        .to_string(),
                ));
            }
        }

        if let Some(endpoint) = &config.endpoint {
            builder = builder.with_url(endpoint);
        }

        let store = builder
            .build()
            .map_err(|e| Error::ConnectionError(format!("GCS build error: {e}")))?;
        Ok(Arc::new(store))
    }
}

impl Default for GCSConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudConnector for GCSConnector {
    async fn connect(&mut self, config: &CloudConfig) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let placeholder_bucket = config
                .parameters
                .get("bucket")
                .cloned()
                .unwrap_or_else(|| "default".to_string());
            let store = Self::build_store(config, &placeholder_bucket)?;
            self.store = Some(store);
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = config;
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn list_objects(&self, bucket: &str, prefix: Option<&str>) -> Result<Vec<CloudObject>> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let prefix_path = prefix.map(ObjectPath::from);
            let results: Vec<_> = store.list(prefix_path.as_ref()).collect::<Vec<_>>().await;

            let mut objects = Vec::with_capacity(results.len());
            for item in results {
                let meta = item.map_err(|e| {
                    Error::IoError(format!("GCS list error for bucket '{}': {}", bucket, e))
                })?;
                objects.push(meta_to_cloud_object(meta));
            }
            Ok(objects)
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, prefix);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn read_dataframe(
        &self,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<DataFrame> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let path = make_path(key);
            let result = store
                .get(&path)
                .await
                .map_err(|e| Error::IoError(format!("GCS get '{}/{}': {}", bucket, key, e)))?;
            let data: Bytes = result
                .bytes()
                .await
                .map_err(|e| Error::IoError(format!("GCS read bytes: {}", e)))?;
            bytes_to_df(data, &format)
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key, format);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn write_dataframe(
        &self,
        df: &DataFrame,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let data = df_to_payload(df, &format)?;
            let path = make_path(key);
            store
                .put(&path, data.into())
                .await
                .map_err(|e| Error::IoError(format!("GCS put '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (df, bucket, key, format);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn download_object(&self, bucket: &str, key: &str, local_path: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let path = make_path(key);
            let result = store
                .get(&path)
                .await
                .map_err(|e| Error::IoError(format!("GCS get '{}/{}': {}", bucket, key, e)))?;
            let data: Bytes = result
                .bytes()
                .await
                .map_err(|e| Error::IoError(format!("GCS read bytes: {}", e)))?;
            std::fs::write(local_path, &data).map_err(|e| Error::IoError(e.to_string()))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key, local_path);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn upload_object(&self, local_path: &str, bucket: &str, key: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let data = std::fs::read(local_path).map_err(|e| Error::IoError(e.to_string()))?;
            let path = make_path(key);
            store
                .put(&path, Bytes::from(data).into())
                .await
                .map_err(|e| Error::IoError(format!("GCS put '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (local_path, bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let path = make_path(key);
            store
                .delete(&path)
                .await
                .map_err(|e| Error::IoError(format!("GCS delete '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn get_object_metadata(&self, bucket: &str, key: &str) -> Result<ObjectMetadata> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let path = make_path(key);
            let meta = store
                .head(&path)
                .await
                .map_err(|e| Error::IoError(format!("GCS head '{}/{}': {}", bucket, key, e)))?;
            Ok(meta_to_object_metadata(meta))
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn object_exists(&self, bucket: &str, key: &str) -> Result<bool> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self
                .store
                .as_ref()
                .ok_or_else(|| Error::ConnectionError("GCSConnector not connected".to_string()))?;
            let path = make_path(key);
            match store.head(&path).await {
                Ok(_) => Ok(true),
                Err(object_store::Error::NotFound { .. }) => Ok(false),
                Err(e) => Err(Error::IoError(format!(
                    "GCS head '{}/{}': {}",
                    bucket, key, e
                ))),
            }
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn create_bucket(&self, _bucket: &str) -> Result<()> {
        Err(Error::NotImplemented(
            "Bucket creation is not supported via object_store".to_string(),
        ))
    }

    async fn delete_bucket(&self, _bucket: &str) -> Result<()> {
        Err(Error::NotImplemented(
            "Bucket deletion is not supported via object_store".to_string(),
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Azure Blob Storage Connector
// ──────────────────────────────────────────────────────────────────────────────

/// Azure Blob Storage connector
///
/// With the `cloud-storage` feature enabled this uses `object_store::azure`.
pub struct AzureConnector {
    #[cfg(feature = "cloud-storage")]
    store: Option<Arc<dyn ObjectStore>>,
    #[cfg(not(feature = "cloud-storage"))]
    _phantom: std::marker::PhantomData<()>,
}

impl AzureConnector {
    /// Create a new, unconnected Azure connector.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cloud-storage")]
            store: None,
            #[cfg(not(feature = "cloud-storage"))]
            _phantom: std::marker::PhantomData,
        }
    }

    /// Build and connect immediately using the provided configuration.
    #[cfg(feature = "cloud-storage")]
    pub async fn connect_with_config(config: CloudConfig) -> Result<Self> {
        let mut connector = Self::new();
        connector.connect(&config).await?;
        Ok(connector)
    }

    /// Stub for no-feature builds.
    #[cfg(not(feature = "cloud-storage"))]
    pub async fn connect_with_config(_config: CloudConfig) -> Result<Self> {
        Err(Error::NotImplemented(
            "cloud-storage feature required for Azure connectivity".to_string(),
        ))
    }

    #[cfg(feature = "cloud-storage")]
    fn build_store(config: &CloudConfig, container: &str) -> Result<Arc<dyn ObjectStore>> {
        let mut builder = MicrosoftAzureBuilder::new().with_container_name(container);

        match &config.credentials {
            CloudCredentials::Azure {
                account_name,
                account_key,
            } => {
                builder = builder
                    .with_account(account_name)
                    .with_access_key(account_key);
            }
            CloudCredentials::Environment => {
                // Uses AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY, or
                // AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID.
                // object_store picks these up automatically.
            }
            _ => {
                return Err(Error::InvalidOperation(
                    "Invalid credentials for Azure: expected Azure or Environment credentials"
                        .to_string(),
                ));
            }
        }

        if let Some(endpoint) = &config.endpoint {
            builder = builder.with_endpoint(endpoint.clone());
        }

        let store = builder
            .build()
            .map_err(|e| Error::ConnectionError(format!("Azure build error: {e}")))?;
        Ok(Arc::new(store))
    }
}

impl Default for AzureConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudConnector for AzureConnector {
    async fn connect(&mut self, config: &CloudConfig) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let placeholder_container = config
                .parameters
                .get("container")
                .or_else(|| config.parameters.get("bucket"))
                .cloned()
                .unwrap_or_else(|| "default".to_string());
            let store = Self::build_store(config, &placeholder_container)?;
            self.store = Some(store);
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = config;
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn list_objects(&self, bucket: &str, prefix: Option<&str>) -> Result<Vec<CloudObject>> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let prefix_path = prefix.map(ObjectPath::from);
            let results: Vec<_> = store.list(prefix_path.as_ref()).collect::<Vec<_>>().await;

            let mut objects = Vec::with_capacity(results.len());
            for item in results {
                let meta = item.map_err(|e| {
                    Error::IoError(format!(
                        "Azure list error for container '{}': {}",
                        bucket, e
                    ))
                })?;
                objects.push(meta_to_cloud_object(meta));
            }
            Ok(objects)
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, prefix);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn read_dataframe(
        &self,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<DataFrame> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let path = make_path(key);
            let result = store
                .get(&path)
                .await
                .map_err(|e| Error::IoError(format!("Azure get '{}/{}': {}", bucket, key, e)))?;
            let data: Bytes = result
                .bytes()
                .await
                .map_err(|e| Error::IoError(format!("Azure read bytes: {}", e)))?;
            bytes_to_df(data, &format)
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key, format);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn write_dataframe(
        &self,
        df: &DataFrame,
        bucket: &str,
        key: &str,
        format: FileFormat,
    ) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let data = df_to_payload(df, &format)?;
            let path = make_path(key);
            store
                .put(&path, data.into())
                .await
                .map_err(|e| Error::IoError(format!("Azure put '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (df, bucket, key, format);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn download_object(&self, bucket: &str, key: &str, local_path: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let path = make_path(key);
            let result = store
                .get(&path)
                .await
                .map_err(|e| Error::IoError(format!("Azure get '{}/{}': {}", bucket, key, e)))?;
            let data: Bytes = result
                .bytes()
                .await
                .map_err(|e| Error::IoError(format!("Azure read bytes: {}", e)))?;
            std::fs::write(local_path, &data).map_err(|e| Error::IoError(e.to_string()))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key, local_path);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn upload_object(&self, local_path: &str, bucket: &str, key: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let data = std::fs::read(local_path).map_err(|e| Error::IoError(e.to_string()))?;
            let path = make_path(key);
            store
                .put(&path, Bytes::from(data).into())
                .await
                .map_err(|e| Error::IoError(format!("Azure put '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (local_path, bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let path = make_path(key);
            store
                .delete(&path)
                .await
                .map_err(|e| Error::IoError(format!("Azure delete '{}/{}': {}", bucket, key, e)))?;
            Ok(())
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn get_object_metadata(&self, bucket: &str, key: &str) -> Result<ObjectMetadata> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let path = make_path(key);
            let meta = store
                .head(&path)
                .await
                .map_err(|e| Error::IoError(format!("Azure head '{}/{}': {}", bucket, key, e)))?;
            Ok(meta_to_object_metadata(meta))
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn object_exists(&self, bucket: &str, key: &str) -> Result<bool> {
        #[cfg(feature = "cloud-storage")]
        {
            let store = self.store.as_ref().ok_or_else(|| {
                Error::ConnectionError("AzureConnector not connected".to_string())
            })?;
            let path = make_path(key);
            match store.head(&path).await {
                Ok(_) => Ok(true),
                Err(object_store::Error::NotFound { .. }) => Ok(false),
                Err(e) => Err(Error::IoError(format!(
                    "Azure head '{}/{}': {}",
                    bucket, key, e
                ))),
            }
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            let _ = (bucket, key);
            Err(Error::NotImplemented(
                "cloud-storage feature required".to_string(),
            ))
        }
    }

    async fn create_bucket(&self, _bucket: &str) -> Result<()> {
        Err(Error::NotImplemented(
            "Container creation is not supported via object_store".to_string(),
        ))
    }

    async fn delete_bucket(&self, _bucket: &str) -> Result<()> {
        Err(Error::NotImplemented(
            "Container deletion is not supported via object_store".to_string(),
        ))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Factory
// ──────────────────────────────────────────────────────────────────────────────

/// Cloud connector factory — convenience constructors
pub struct CloudConnectorFactory;

impl CloudConnectorFactory {
    /// Create an S3 connector
    pub fn s3() -> S3Connector {
        S3Connector::new()
    }

    /// Create a GCS connector
    pub fn gcs() -> GCSConnector {
        GCSConnector::new()
    }

    /// Create an Azure connector
    pub fn azure() -> AzureConnector {
        AzureConnector::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Unit tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_config_builder() {
        let config = CloudConfig::new(
            CloudProvider::AWS,
            CloudCredentials::AWS {
                access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
                secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
                session_token: None,
            },
        )
        .with_region("us-west-2")
        .with_timeout(600);

        assert!(matches!(config.provider, CloudProvider::AWS));
        assert_eq!(config.region, Some("us-west-2".to_string()));
        assert_eq!(config.timeout, Some(600));
    }

    #[test]
    fn test_file_format_detection() {
        assert!(matches!(
            FileFormat::from_extension("data.csv").expect("should detect CSV"),
            FileFormat::CSV { .. }
        ));
        assert!(matches!(
            FileFormat::from_extension("data.parquet").expect("should detect Parquet"),
            FileFormat::Parquet
        ));
        assert!(matches!(
            FileFormat::from_extension("data.json").expect("should detect JSON"),
            FileFormat::JSON
        ));
        assert!(matches!(
            FileFormat::from_extension("data.jsonl").expect("should detect JSONL"),
            FileFormat::JSONL
        ));
        assert!(FileFormat::from_extension("data.unknown").is_none());
    }

    #[test]
    fn test_connectors_instantiate() {
        let _s3 = S3Connector::new();
        let _gcs = GCSConnector::new();
        let _azure = AzureConnector::new();
    }

    #[test]
    fn test_factory() {
        let _s3 = CloudConnectorFactory::s3();
        let _gcs = CloudConnectorFactory::gcs();
        let _azure = CloudConnectorFactory::azure();
    }

    #[test]
    fn test_gcs_config() {
        let config = CloudConfig::new(
            CloudProvider::GCS,
            CloudCredentials::GCS {
                service_account_key: "{}".to_string(),
                project_id: "my-project".to_string(),
            },
        )
        .with_parameter("bucket", "my-bucket");

        assert!(matches!(config.provider, CloudProvider::GCS));
        assert_eq!(
            config.parameters.get("bucket"),
            Some(&"my-bucket".to_string())
        );
    }

    #[test]
    fn test_azure_config() {
        let config = CloudConfig::new(
            CloudProvider::Azure,
            CloudCredentials::Azure {
                account_name: "myaccount".to_string(),
                account_key: "base64key==".to_string(),
            },
        )
        .with_parameter("container", "mycontainer");

        assert!(matches!(config.provider, CloudProvider::Azure));
        assert_eq!(
            config.parameters.get("container"),
            Some(&"mycontainer".to_string())
        );
    }
}
