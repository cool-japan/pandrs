//! Integration tests for PandRS ecosystem connectors
//! 
//! This module provides integration testing infrastructure with mock services
//! for database and cloud storage connectors.

use pandrs::core::error::Result;
use pandrs::dataframe::DataFrame;
use pandrs::series::base::Series;
use std::collections::HashMap;

/// Test utilities for integration testing
pub mod test_utils {
    use super::*;
    
    /// Create a sample DataFrame for testing
    pub fn create_test_dataframe() -> Result<DataFrame> {
        let mut df = DataFrame::new();
        
        // Create test data
        let ids: Vec<String> = (1..=100).map(|i| i.to_string()).collect();
        let names: Vec<String> = (1..=100).map(|i| format!("User_{}", i)).collect();
        let ages: Vec<String> = (20..120).map(|i| i.to_string()).collect();
        let active: Vec<String> = (1..=100).map(|i| (i % 2 == 0).to_string()).collect();
        
        // Add columns
        df.add_column("id".to_string(), Series::new(ids, Some("id".to_string()))?)?;
        df.add_column("name".to_string(), Series::new(names, Some("name".to_string()))?)?;
        df.add_column("age".to_string(), Series::new(ages, Some("age".to_string()))?)?;
        df.add_column("active".to_string(), Series::new(active, Some("active".to_string()))?)?;
        
        Ok(df)
    }
    
    /// Create a large DataFrame for performance testing
    pub fn create_large_test_dataframe(rows: usize) -> Result<DataFrame> {
        let mut df = DataFrame::new();
        
        let ids: Vec<String> = (1..=rows).map(|i| i.to_string()).collect();
        let values: Vec<String> = (1..=rows).map(|i| (i as f64 * 1.5).to_string()).collect();
        let categories: Vec<String> = (1..=rows).map(|i| {
            match i % 4 {
                0 => "A",
                1 => "B", 
                2 => "C",
                _ => "D",
            }.to_string()
        }).collect();
        
        df.add_column("id".to_string(), Series::new(ids, Some("id".to_string()))?)?;
        df.add_column("value".to_string(), Series::new(values, Some("value".to_string()))?)?;
        df.add_column("category".to_string(), Series::new(categories, Some("category".to_string()))?)?;
        
        Ok(df)
    }
    
    /// Assert DataFrames are equal (for testing)
    pub fn assert_dataframes_equal(df1: &DataFrame, df2: &DataFrame) {
        assert_eq!(df1.row_count(), df2.row_count(), "Row counts differ");
        assert_eq!(df1.column_names(), df2.column_names(), "Column names differ");
        
        for col_name in df1.column_names() {
            let values1 = df1.get_column_string_values(&col_name).unwrap();
            let values2 = df2.get_column_string_values(&col_name).unwrap();
            assert_eq!(values1, values2, "Column '{}' values differ", col_name);
        }
    }
}

/// Mock database connector for testing
pub mod mock_database {
    use super::*;
    use pandrs::connectors::database::*;
    use std::collections::HashMap;
    use async_trait::async_trait;
    
    /// Mock database connector that simulates database operations
    pub struct MockDatabaseConnector {
        pub connected: bool,
        pub tables: HashMap<String, DataFrame>,
        pub query_log: Vec<String>,
    }
    
    impl MockDatabaseConnector {
        pub fn new() -> Self {
            Self {
                connected: false,
                tables: HashMap::new(),
                query_log: Vec::new(),
            }
        }
        
        pub fn with_table(mut self, name: &str, df: DataFrame) -> Self {
            self.tables.insert(name.to_string(), df);
            self
        }
    }
    
    impl DatabaseConnector for MockDatabaseConnector {
        async fn connect(&mut self, _config: &DatabaseConfig) -> Result<()> {
            self.connected = true;
            Ok(())
        }
        
        async fn query(&self, sql: &str) -> Result<DataFrame> {
            if !self.connected {
                return Err(pandrs::core::error::Error::ConnectionError(
                    "Not connected to database".to_string()
                ));
            }
            
            // Simple mock: return test data for any SELECT query
            if sql.to_uppercase().starts_with("SELECT") {
                test_utils::create_test_dataframe()
            } else {
                Ok(DataFrame::new())
            }
        }
        
        async fn query_with_params(&self, sql: &str, _params: &[&dyn std::fmt::Display]) -> Result<DataFrame> {
            self.query(sql).await
        }
        
        async fn write_table(&self, _df: &DataFrame, table_name: &str, _if_exists: WriteMode) -> Result<()> {
            if !self.connected {
                return Err(pandrs::core::error::Error::ConnectionError(
                    "Not connected to database".to_string()
                ));
            }
            
            // Simulate successful write
            println!("Mock: Written data to table '{}'", table_name);
            Ok(())
        }
        
        async fn list_tables(&self) -> Result<Vec<String>> {
            Ok(self.tables.keys().cloned().collect())
        }
        
        async fn get_table_info(&self, table_name: &str) -> Result<TableInfo> {
            if let Some(df) = self.tables.get(table_name) {
                let columns = df.column_names().into_iter().map(|name| ColumnInfo {
                    name: name.clone(),
                    data_type: "TEXT".to_string(),
                    nullable: true,
                    primary_key: name == "id",
                    default_value: None,
                }).collect();
                
                Ok(TableInfo {
                    name: table_name.to_string(),
                    columns,
                    row_count: Some(df.row_count() as u64),
                    schema: Some("public".to_string()),
                })
            } else {
                Err(pandrs::core::error::Error::InvalidOperation(
                    format!("Table '{}' not found", table_name)
                ))
            }
        }
        
        async fn execute(&self, sql: &str) -> Result<u64> {
            if !self.connected {
                return Err(pandrs::core::error::Error::ConnectionError(
                    "Not connected to database".to_string()
                ));
            }
            
            // Simulate affected rows
            if sql.to_uppercase().contains("INSERT") {
                Ok(1) // Simulated insert
            } else if sql.to_uppercase().contains("UPDATE") {
                Ok(5) // Simulated update
            } else {
                Ok(0)
            }
        }
        
        async fn begin_transaction(&self) -> Result<String> {
            Ok(format!("mock_tx_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()))
        }
        
        async fn close(&mut self) -> Result<()> {
            self.connected = false;
            Ok(())
        }
    }
}

/// Mock cloud storage connector for testing
pub mod mock_cloud {
    use super::*;
    use pandrs::connectors::cloud::*;
    use std::collections::HashMap;
    use async_trait::async_trait;
    
    /// Mock cloud storage that simulates object operations
    pub struct MockCloudConnector {
        pub connected: bool,
        pub objects: HashMap<String, Vec<u8>>, // bucket/key -> data
        pub metadata: HashMap<String, ObjectMetadata>,
    }
    
    impl MockCloudConnector {
        pub fn new() -> Self {
            Self {
                connected: false,
                objects: HashMap::new(),
                metadata: HashMap::new(),
            }
        }
        
        pub fn with_object(mut self, bucket: &str, key: &str, data: Vec<u8>) -> Self {
            let full_key = format!("{}/{}", bucket, key);
            self.objects.insert(full_key.clone(), data);
            self.metadata.insert(full_key, ObjectMetadata {
                size: data.len() as u64,
                last_modified: Some("2024-12-16T10:00:00Z".to_string()),
                content_type: Some("application/octet-stream".to_string()),
                etag: Some("\"mock-etag\"".to_string()),
                custom_metadata: HashMap::new(),
            });
            self
        }
        
        fn get_full_key(&self, bucket: &str, key: &str) -> String {
            format!("{}/{}", bucket, key)
        }
    }
    
    impl CloudConnector for MockCloudConnector {
        async fn connect(&mut self, _config: &CloudConfig) -> Result<()> {
            self.connected = true;
            Ok(())
        }
        
        async fn list_objects(&self, bucket: &str, prefix: Option<&str>) -> Result<Vec<CloudObject>> {
            if !self.connected {
                return Err(pandrs::core::error::Error::ConnectionError(
                    "Not connected to cloud storage".to_string()
                ));
            }
            
            let bucket_prefix = format!("{}/", bucket);
            let full_prefix = match prefix {
                Some(p) => format!("{}{}", bucket_prefix, p),
                None => bucket_prefix,
            };
            
            let objects = self.objects.keys()
                .filter(|key| key.starts_with(&full_prefix))
                .map(|full_key| {
                    let key = full_key.strip_prefix(&format!("{}/", bucket)).unwrap_or(full_key);
                    let metadata = self.metadata.get(full_key).unwrap();
                    CloudObject {
                        key: key.to_string(),
                        size: metadata.size,
                        last_modified: metadata.last_modified.clone(),
                        etag: metadata.etag.clone(),
                        content_type: metadata.content_type.clone(),
                    }
                })
                .collect();
            
            Ok(objects)
        }
        
        async fn read_dataframe(&self, bucket: &str, key: &str, _format: FileFormat) -> Result<DataFrame> {
            if !self.connected {
                return Err(pandrs::core::error::Error::ConnectionError(
                    "Not connected to cloud storage".to_string()
                ));
            }
            
            let full_key = self.get_full_key(bucket, key);
            if self.objects.contains_key(&full_key) {
                // Return mock DataFrame for any existing object
                test_utils::create_test_dataframe()
            } else {
                Err(pandrs::core::error::Error::InvalidOperation(
                    format!("Object {}/{} not found", bucket, key)
                ))
            }
        }
        
        async fn write_dataframe(&self, _df: &DataFrame, bucket: &str, key: &str, _format: FileFormat) -> Result<()> {
            if !self.connected {
                return Err(pandrs::core::error::Error::ConnectionError(
                    "Not connected to cloud storage".to_string()
                ));
            }
            
            // Simulate successful write
            println!("Mock: Written DataFrame to {}/{}", bucket, key);
            Ok(())
        }
        
        async fn download_object(&self, bucket: &str, key: &str, local_path: &str) -> Result<()> {
            let full_key = self.get_full_key(bucket, key);
            if let Some(data) = self.objects.get(&full_key) {
                // Simulate file write
                println!("Mock: Downloaded {}/{} to {}", bucket, key, local_path);
                Ok(())
            } else {
                Err(pandrs::core::error::Error::InvalidOperation(
                    format!("Object {}/{} not found", bucket, key)
                ))
            }
        }
        
        async fn upload_object(&self, local_path: &str, bucket: &str, key: &str) -> Result<()> {
            if !self.connected {
                return Err(pandrs::core::error::Error::ConnectionError(
                    "Not connected to cloud storage".to_string()
                ));
            }
            
            // Simulate successful upload
            println!("Mock: Uploaded {} to {}/{}", local_path, bucket, key);
            Ok(())
        }
        
        async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
            println!("Mock: Deleted {}/{}", bucket, key);
            Ok(())
        }
        
        async fn get_object_metadata(&self, bucket: &str, key: &str) -> Result<ObjectMetadata> {
            let full_key = self.get_full_key(bucket, key);
            if let Some(metadata) = self.metadata.get(&full_key) {
                Ok(metadata.clone())
            } else {
                Err(pandrs::core::error::Error::InvalidOperation(
                    format!("Object {}/{} not found", bucket, key)
                ))
            }
        }
        
        async fn object_exists(&self, bucket: &str, key: &str) -> Result<bool> {
            let full_key = self.get_full_key(bucket, key);
            Ok(self.objects.contains_key(&full_key))
        }
        
        async fn create_bucket(&self, bucket: &str) -> Result<()> {
            println!("Mock: Created bucket {}", bucket);
            Ok(())
        }
        
        async fn delete_bucket(&self, bucket: &str) -> Result<()> {
            println!("Mock: Deleted bucket {}", bucket);
            Ok(())
        }
    }
}