//! # Data Connectors
//!
//! This module provides connectivity to various data sources including
//! databases, cloud storage, and other external systems.

pub mod cloud;
pub mod database;

// Re-export commonly used types
pub use database::{
    ColumnInfo, DatabaseConfig, DatabaseConnector, DatabaseConnectorFactory, SQLiteConnector,
    TableInfo, WriteMode,
};

#[cfg(feature = "sql")]
pub use database::PostgreSQLConnector;

pub use cloud::{
    AzureConnector, CloudConfig, CloudConnector, CloudConnectorFactory, CloudCredentials,
    CloudObject, CloudProvider, FileFormat, GCSConnector, ObjectMetadata, S3Connector,
};

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;

/// High-level data connector that can connect to various sources
pub enum DataConnector {
    SQLite(database::SQLiteConnector),
    #[cfg(feature = "sql")]
    PostgreSQL(database::PostgreSQLConnector),
    S3(cloud::S3Connector),
    GCS(cloud::GCSConnector),
    Azure(cloud::AzureConnector),
}

impl DataConnector {
    /// Create SQLite connector
    pub fn sqlite() -> Self {
        Self::SQLite(database::SQLiteConnector::new())
    }

    /// Create PostgreSQL connector
    #[cfg(feature = "sql")]
    pub fn postgresql() -> Self {
        Self::PostgreSQL(database::PostgreSQLConnector::new())
    }

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

    /// Create connector from connection string
    pub fn from_connection_string(connection_string: &str) -> Result<Self> {
        if connection_string.starts_with("s3://") {
            Ok(Self::s3())
        } else if connection_string.starts_with("gs://") {
            Ok(Self::gcs())
        } else if connection_string.starts_with("azure://") {
            Ok(Self::azure())
        } else if connection_string.starts_with("sqlite:") {
            Ok(Self::sqlite())
        } else if connection_string.starts_with("postgresql://")
            || connection_string.starts_with("postgres://")
        {
            #[cfg(feature = "sql")]
            {
                Ok(Self::postgresql())
            }
            #[cfg(not(feature = "sql"))]
            {
                Err(Error::FeatureNotAvailable(
                    "SQL feature not enabled".to_string(),
                ))
            }
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

    /// Read DataFrame from SQL query (database sources only)
    pub async fn read_sql(&self, query: &str) -> Result<DataFrame> {
        use database::DatabaseConnector;

        match &self.connector {
            #[cfg(feature = "sql")]
            DataConnector::SQLite(db) => db.query(query).await,
            #[cfg(feature = "sql")]
            DataConnector::PostgreSQL(db) => db.query(query).await,
            #[cfg(not(feature = "sql"))]
            DataConnector::SQLite(_) => Err(Error::FeatureNotAvailable(
                "SQL feature not enabled".to_string(),
            )),
            _ => Err(Error::InvalidOperation(
                "SQL queries not supported for cloud storage connectors".to_string(),
            )),
        }
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
            _ => Err(Error::InvalidOperation(
                "Cloud storage operations not supported for database connectors".to_string(),
            )),
        }
    }

    /// Write DataFrame to database table
    pub async fn write_sql(&self, df: &DataFrame, table_name: &str) -> Result<()> {
        use database::DatabaseConnector;

        match &self.connector {
            #[cfg(feature = "sql")]
            DataConnector::SQLite(db) => db.write_table(df, table_name, WriteMode::Replace).await,
            #[cfg(feature = "sql")]
            DataConnector::PostgreSQL(db) => {
                db.write_table(df, table_name, WriteMode::Replace).await
            }
            #[cfg(not(feature = "sql"))]
            DataConnector::SQLite(_) => Err(Error::FeatureNotAvailable(
                "SQL feature not enabled".to_string(),
            )),
            _ => Err(Error::InvalidOperation(
                "Database operations not supported for cloud storage connectors".to_string(),
            )),
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
            _ => Err(Error::InvalidOperation(
                "Cloud storage operations not supported for database connectors".to_string(),
            )),
        }
    }
}

/// Convenience functions for DataFrame connectivity
impl DataFrame {
    /// Read from any data source using connection string
    pub async fn read_from(connection_string: &str, query_or_path: &str) -> Result<Self> {
        let connector = DataConnector::from_connection_string(connection_string)?;
        let source = DataSource::new(connector);

        if connection_string.starts_with("s3://")
            || connection_string.starts_with("gs://")
            || connection_string.starts_with("azure://")
        {
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
        } else {
            // Database query
            source.read_sql(query_or_path).await
        }
    }

    /// Write to any data source using connection string
    pub async fn write_to(&self, connection_string: &str, table_or_path: &str) -> Result<()> {
        let connector = DataConnector::from_connection_string(connection_string)?;
        let source = DataSource::new(connector);

        if connection_string.starts_with("s3://")
            || connection_string.starts_with("gs://")
            || connection_string.starts_with("azure://")
        {
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
        } else {
            // Database table
            source.write_sql(self, table_or_path).await
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
        assert!(matches!(s3_connector.unwrap(), DataConnector::S3(_)));

        let gcs_connector = DataConnector::from_connection_string("gs://bucket/path");
        assert!(gcs_connector.is_ok());
        assert!(matches!(gcs_connector.unwrap(), DataConnector::GCS(_)));

        // Test database URLs
        let sqlite_connector = DataConnector::from_connection_string("sqlite::memory:");
        assert!(sqlite_connector.is_ok());
        assert!(matches!(
            sqlite_connector.unwrap(),
            DataConnector::SQLite(_)
        ));
    }

    #[test]
    fn test_data_source_creation() {
        let connector = DataConnector::s3();
        let source = DataSource::new(connector);

        // DataSource should be created successfully
        // (We can't test async methods in a simple unit test)
    }
}
