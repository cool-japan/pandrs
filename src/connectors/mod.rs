//! # Data Connectors
//!
//! This module provides connectivity to various data sources including
//! cloud storage and other external systems.

pub mod cloud;
pub mod local;

// Re-export commonly used types
pub use cloud::{
    AzureConnector, CloudConfig, CloudConnector, CloudConnectorFactory, CloudCredentials,
    CloudObject, CloudProvider, FileFormat, GCSConnector, ObjectMetadata, S3Connector,
};

pub use local::LocalConnector;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;

/// High-level data connector that can connect to various sources
pub enum DataConnector {
    S3(cloud::S3Connector),
    GCS(cloud::GCSConnector),
    Azure(cloud::AzureConnector),
    Local(local::LocalConnector),
}

impl DataConnector {
    /// Create S3 connector
    pub fn s3() -> Self {
        Self::S3(cloud::S3Connector::new())
    }

    /// Create GCS connector
    pub fn gcs() -> Self {
        Self::GCS(cloud::GCSConnector::new())
    }

    /// Create Azure connector
    pub fn azure() -> Self {
        Self::Azure(cloud::AzureConnector::new())
    }

    /// Create local filesystem connector (useful for testing and development)
    pub fn local(base_path: impl Into<std::path::PathBuf>) -> Self {
        Self::Local(local::LocalConnector::new(base_path))
    }

    /// Create connector from connection string
    pub fn from_connection_string(connection_string: &str) -> Result<Self> {
        if connection_string.starts_with("s3://") {
            Ok(Self::s3())
        } else if connection_string.starts_with("gs://") {
            Ok(Self::gcs())
        } else if connection_string.starts_with("azure://") {
            Ok(Self::azure())
        } else if connection_string.starts_with("local://") {
            let base = connection_string.trim_start_matches("local://");
            Ok(Self::local(base))
        } else {
            Err(Error::InvalidOperation(format!(
                "Unsupported connection string: {}",
                connection_string
            )))
        }
    }
}

/// Unified data source for reading DataFrames from various sources
pub struct DataSource {
    connector: DataConnector,
}

impl DataSource {
    /// Create new data source
    pub fn new(connector: DataConnector) -> Self {
        Self { connector }
    }

    /// Read DataFrame from cloud storage path
    pub async fn read_cloud(&self, bucket: &str, key: &str) -> Result<DataFrame> {
        use cloud::CloudConnector;

        let format = FileFormat::from_extension(key).unwrap_or(FileFormat::CSV {
            delimiter: ',',
            has_header: true,
        });

        match &self.connector {
            DataConnector::S3(cloud) => cloud.read_dataframe(bucket, key, format).await,
            DataConnector::GCS(cloud) => cloud.read_dataframe(bucket, key, format).await,
            DataConnector::Azure(cloud) => cloud.read_dataframe(bucket, key, format).await,
            DataConnector::Local(cloud) => cloud.read_dataframe(bucket, key, format).await,
        }
    }

    /// Write DataFrame to cloud storage
    pub async fn write_cloud(&self, df: &DataFrame, bucket: &str, key: &str) -> Result<()> {
        use cloud::CloudConnector;

        let format = FileFormat::from_extension(key).unwrap_or(FileFormat::CSV {
            delimiter: ',',
            has_header: true,
        });

        match &self.connector {
            DataConnector::S3(cloud) => cloud.write_dataframe(df, bucket, key, format).await,
            DataConnector::GCS(cloud) => cloud.write_dataframe(df, bucket, key, format).await,
            DataConnector::Azure(cloud) => cloud.write_dataframe(df, bucket, key, format).await,
            DataConnector::Local(cloud) => cloud.write_dataframe(df, bucket, key, format).await,
        }
    }
}

/// Convenience functions for DataFrame connectivity
impl DataFrame {
    /// Read from any data source using connection string
    pub async fn read_from(connection_string: &str, query_or_path: &str) -> Result<Self> {
        let connector = DataConnector::from_connection_string(connection_string)?;
        let source = DataSource::new(connector);

        // Extract bucket and key from path
        let parts: Vec<&str> = query_or_path.split('/').collect();
        if parts.len() >= 2 {
            let bucket = parts[0];
            let key = parts[1..].join("/");
            source.read_cloud(bucket, &key).await
        } else {
            Err(Error::InvalidOperation(
                "Invalid cloud storage path".to_string(),
            ))
        }
    }

    /// Write to any data source using connection string
    pub async fn write_to(&self, connection_string: &str, table_or_path: &str) -> Result<()> {
        let connector = DataConnector::from_connection_string(connection_string)?;
        let source = DataSource::new(connector);

        // Extract bucket and key from path
        let parts: Vec<&str> = table_or_path.split('/').collect();
        if parts.len() >= 2 {
            let bucket = parts[0];
            let key = parts[1..].join("/");
            source.write_cloud(self, bucket, &key).await
        } else {
            Err(Error::InvalidOperation(
                "Invalid cloud storage path".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_connector_from_connection_string() {
        // Test cloud storage URLs
        let s3_connector = DataConnector::from_connection_string("s3://bucket/path");
        assert!(s3_connector.is_ok());
        assert!(matches!(
            s3_connector.expect("operation should succeed"),
            DataConnector::S3(_)
        ));

        let gcs_connector = DataConnector::from_connection_string("gs://bucket/path");
        assert!(gcs_connector.is_ok());
        assert!(matches!(
            gcs_connector.expect("operation should succeed"),
            DataConnector::GCS(_)
        ));

        // Test local filesystem URL
        let local_connector = DataConnector::from_connection_string("local:///tmp/test");
        assert!(local_connector.is_ok());
        assert!(matches!(
            local_connector.expect("operation should succeed"),
            DataConnector::Local(_)
        ));
    }

    #[test]
    fn test_data_source_creation() {
        let connector = DataConnector::s3();
        let _source = DataSource::new(connector);
        // DataSource should be created successfully
    }
}
