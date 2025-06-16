//! Integration tests for cloud storage connectors
//! 
//! These tests use mock cloud services to validate cloud connector
//! functionality without requiring external cloud storage accounts.

use super::{test_utils, mock_cloud::MockCloudConnector};
use pandrs::connectors::cloud::*;
use pandrs::core::error::Result;

#[tokio::test]
async fn test_cloud_connection() -> Result<()> {
    let mut connector = MockCloudConnector::new();
    let config = CloudConfig::new(
        CloudProvider::AWS,
        CloudCredentials::Environment
    );
    
    // Initially not connected
    assert!(!connector.connected);
    
    // Connect
    connector.connect(&config).await?;
    assert!(connector.connected);
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_configuration() -> Result<()> {
    // Test AWS configuration
    let aws_config = CloudConfig::new(
        CloudProvider::AWS,
        CloudCredentials::AWS {
            access_key_id: "AKIATEST".to_string(),
            secret_access_key: "test-secret".to_string(),
            session_token: Some("test-token".to_string()),
        }
    )
    .with_region("us-west-2")
    .with_timeout(300)
    .with_parameter("max_retries", "3");
    
    assert!(matches!(aws_config.provider, CloudProvider::AWS));
    assert_eq!(aws_config.region, Some("us-west-2".to_string()));
    assert_eq!(aws_config.timeout, Some(300));
    
    // Test GCS configuration
    let gcs_config = CloudConfig::new(
        CloudProvider::GCS,
        CloudCredentials::GCS {
            project_id: "my-project".to_string(),
            service_account_key: "/path/to/key.json".to_string(),
        }
    );
    
    assert!(matches!(gcs_config.provider, CloudProvider::GCS));
    
    // Test Azure configuration
    let azure_config = CloudConfig::new(
        CloudProvider::Azure,
        CloudCredentials::Azure {
            account_name: "myaccount".to_string(),
            account_key: "base64-key".to_string(),
        }
    );
    
    assert!(matches!(azure_config.provider, CloudProvider::Azure));
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_list_objects() -> Result<()> {
    let test_data = vec![1, 2, 3, 4, 5];
    let mut connector = MockCloudConnector::new()
        .with_object("test-bucket", "data/file1.csv", test_data.clone())
        .with_object("test-bucket", "data/file2.parquet", test_data.clone())
        .with_object("test-bucket", "logs/app.log", test_data);
    
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    // List all objects in bucket
    let objects = connector.list_objects("test-bucket", None).await?;
    assert_eq!(objects.len(), 3);
    
    // List objects with prefix
    let data_objects = connector.list_objects("test-bucket", Some("data/")).await?;
    assert_eq!(data_objects.len(), 2);
    
    let log_objects = connector.list_objects("test-bucket", Some("logs/")).await?;
    assert_eq!(log_objects.len(), 1);
    
    // Verify object properties
    let file1 = data_objects.iter().find(|obj| obj.key == "file1.csv").unwrap();
    assert_eq!(file1.size, 5);
    assert!(file1.last_modified.is_some());
    assert!(file1.etag.is_some());
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_read_dataframe() -> Result<()> {
    let test_data = b"id,name,age\n1,John,25\n2,Jane,30".to_vec();
    let mut connector = MockCloudConnector::new()
        .with_object("test-bucket", "data/users.csv", test_data);
    
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    // Read CSV file
    let df = connector.read_dataframe(
        "test-bucket",
        "data/users.csv",
        FileFormat::CSV { delimiter: ',', has_header: true }
    ).await?;
    
    assert!(df.row_count() > 0);
    assert!(!df.column_names().is_empty());
    
    // Test reading non-existent file
    let result = connector.read_dataframe(
        "test-bucket",
        "non-existent.csv",
        FileFormat::CSV { delimiter: ',', has_header: true }
    ).await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_write_dataframe() -> Result<()> {
    let mut connector = MockCloudConnector::new();
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    let df = test_utils::create_test_dataframe()?;
    
    // Test writing different formats
    connector.write_dataframe(
        &df,
        "output-bucket",
        "results/data.csv",
        FileFormat::CSV { delimiter: ',', has_header: true }
    ).await?;
    
    connector.write_dataframe(
        &df,
        "output-bucket",
        "results/data.parquet",
        FileFormat::Parquet
    ).await?;
    
    connector.write_dataframe(
        &df,
        "output-bucket",
        "results/data.json",
        FileFormat::JSON
    ).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_object_operations() -> Result<()> {
    let test_data = b"test file content".to_vec();
    let mut connector = MockCloudConnector::new()
        .with_object("test-bucket", "data/test.txt", test_data);
    
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    // Test object existence check
    let exists = connector.object_exists("test-bucket", "data/test.txt").await?;
    assert!(exists);
    
    let not_exists = connector.object_exists("test-bucket", "data/missing.txt").await?;
    assert!(!not_exists);
    
    // Test get object metadata
    let metadata = connector.get_object_metadata("test-bucket", "data/test.txt").await?;
    assert_eq!(metadata.size, 17); // Length of "test file content"
    assert!(metadata.last_modified.is_some());
    
    // Test download object
    connector.download_object("test-bucket", "data/test.txt", "/tmp/downloaded.txt").await?;
    
    // Test upload object
    connector.upload_object("/tmp/upload.txt", "test-bucket", "data/uploaded.txt").await?;
    
    // Test delete object
    connector.delete_object("test-bucket", "data/test.txt").await?;
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_bucket_operations() -> Result<()> {
    let mut connector = MockCloudConnector::new();
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    // Test create bucket
    connector.create_bucket("new-bucket").await?;
    
    // Test delete bucket
    connector.delete_bucket("old-bucket").await?;
    
    Ok(())
}

#[tokio::test]
async fn test_file_format_detection() -> Result<()> {
    // Test automatic format detection
    let csv_format = FileFormat::from_extension("data.csv").unwrap();
    assert!(matches!(csv_format, FileFormat::CSV { .. }));
    
    let parquet_format = FileFormat::from_extension("data.parquet").unwrap();
    assert!(matches!(parquet_format, FileFormat::Parquet));
    
    let json_format = FileFormat::from_extension("data.json").unwrap();
    assert!(matches!(json_format, FileFormat::JSON));
    
    let jsonl_format = FileFormat::from_extension("logs.jsonl").unwrap();
    assert!(matches!(jsonl_format, FileFormat::JSONL));
    
    // Test unknown extension
    let unknown = FileFormat::from_extension("data.unknown");
    assert!(unknown.is_none());
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_error_handling() -> Result<()> {
    let mut connector = MockCloudConnector::new();
    
    // Test operations without connection
    let result = connector.list_objects("bucket", None).await;
    assert!(result.is_err());
    
    let result = connector.read_dataframe(
        "bucket",
        "key",
        FileFormat::CSV { delimiter: ',', has_header: true }
    ).await;
    assert!(result.is_err());
    
    let df = test_utils::create_test_dataframe()?;
    let result = connector.write_dataframe(&df, "bucket", "key", FileFormat::JSON).await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_large_dataset_handling() -> Result<()> {
    let mut connector = MockCloudConnector::new();
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    // Create large DataFrame
    let large_df = test_utils::create_large_test_dataframe(100000)?;
    
    // Test writing large dataset
    connector.write_dataframe(
        &large_df,
        "big-data-bucket",
        "datasets/large_data.parquet",
        FileFormat::Parquet
    ).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_concurrent_operations() -> Result<()> {
    use tokio::task;
    
    let test_data = vec![1, 2, 3, 4, 5];
    let mut connector = MockCloudConnector::new();
    
    // Add multiple objects
    for i in 0..10 {
        connector = connector.with_object(
            "test-bucket",
            &format!("data/file_{}.csv", i),
            test_data.clone()
        );
    }
    
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    // Test concurrent reads
    let handles: Vec<_> = (0..5).map(|i| {
        task::spawn(async move {
            // Create new connector for each task
            let test_data = vec![1, 2, 3, 4, 5];
            let mut local_connector = MockCloudConnector::new()
                .with_object("test-bucket", &format!("data/file_{}.csv", i), test_data);
            
            let local_config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
            local_connector.connect(&local_config).await.unwrap();
            
            local_connector.read_dataframe(
                "test-bucket",
                &format!("data/file_{}.csv", i),
                FileFormat::CSV { delimiter: ',', has_header: true }
            ).await
        })
    }).collect();
    
    // Wait for all operations to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_provider_specific_features() -> Result<()> {
    // Test AWS-specific configuration
    let aws_config = CloudConfig::new(
        CloudProvider::AWS,
        CloudCredentials::Environment
    )
    .with_region("us-east-1")
    .with_parameter("max_keys", "1000");
    
    let mut aws_connector = MockCloudConnector::new();
    aws_connector.connect(&aws_config).await?;
    
    // Test GCS-specific configuration
    let gcs_config = CloudConfig::new(
        CloudProvider::GCS,
        CloudCredentials::GCS {
            project_id: "test-project".to_string(),
            service_account_key: "/path/to/key.json".to_string(),
        }
    );
    
    let mut gcs_connector = MockCloudConnector::new();
    gcs_connector.connect(&gcs_config).await?;
    
    // Test Azure-specific configuration
    let azure_config = CloudConfig::new(
        CloudProvider::Azure,
        CloudCredentials::Azure {
            account_name: "testaccount".to_string(),
            account_key: "test-key".to_string(),
        }
    );
    
    let mut azure_connector = MockCloudConnector::new();
    azure_connector.connect(&azure_config).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_cloud_performance_characteristics() -> Result<()> {
    use std::time::Instant;
    
    let mut connector = MockCloudConnector::new();
    let config = CloudConfig::new(CloudProvider::AWS, CloudCredentials::Environment);
    connector.connect(&config).await?;
    
    let df = test_utils::create_test_dataframe()?;
    
    // Measure write performance
    let start = Instant::now();
    connector.write_dataframe(
        &df,
        "perf-bucket",
        "test/performance.parquet",
        FileFormat::Parquet
    ).await?;
    let write_duration = start.elapsed();
    
    // Mock operations should be fast
    assert!(write_duration.as_millis() < 100);
    
    // Add test object for read test
    connector = connector.with_object("perf-bucket", "test/data.csv", vec![1, 2, 3]);
    connector.connect(&config).await?;
    
    // Measure read performance
    let start = Instant::now();
    let _read_df = connector.read_dataframe(
        "perf-bucket",
        "test/data.csv",
        FileFormat::CSV { delimiter: ',', has_header: true }
    ).await?;
    let read_duration = start.elapsed();
    
    assert!(read_duration.as_millis() < 100);
    
    Ok(())
}