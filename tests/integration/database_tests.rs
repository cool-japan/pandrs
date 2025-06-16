//! Integration tests for database connectors
//! 
//! These tests use mock database services to validate database connector
//! functionality without requiring external database instances.

use super::{test_utils, mock_database::MockDatabaseConnector};
use pandrs::connectors::database::*;
use pandrs::core::error::Result;

#[tokio::test]
async fn test_database_connection() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    // Initially not connected
    assert!(!connector.connected);
    
    // Connect
    connector.connect(&config).await?;
    assert!(connector.connected);
    
    // Close connection
    connector.close().await?;
    assert!(!connector.connected);
    
    Ok(())
}

#[tokio::test]
async fn test_database_query() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Test basic query
    let df = connector.query("SELECT * FROM users").await?;
    assert!(df.row_count() > 0);
    assert!(df.column_names().contains(&"id".to_string()));
    assert!(df.column_names().contains(&"name".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_database_query_without_connection() -> Result<()> {
    let connector = MockDatabaseConnector::new();
    
    // Should fail when not connected
    let result = connector.query("SELECT * FROM users").await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_database_parameterized_query() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Test parameterized query
    let df = connector.query_with_params(
        "SELECT * FROM users WHERE age > $1 AND active = $2",
        &[&25, &true]
    ).await?;
    
    assert!(df.row_count() > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_database_write_operations() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Create test DataFrame
    let df = test_utils::create_test_dataframe()?;
    
    // Test different write modes
    connector.write_table(&df, "test_table", WriteMode::Replace).await?;
    connector.write_table(&df, "test_table", WriteMode::Append).await?;
    
    // WriteMode::Fail would normally fail if table exists, but mock allows it
    connector.write_table(&df, "new_table", WriteMode::Fail).await?;
    
    Ok(())
}

#[tokio::test]
async fn test_database_metadata_operations() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let df = test_utils::create_test_dataframe()?;
    
    // Set up mock with test table
    connector = connector.with_table("users", df);
    
    let config = DatabaseConfig::new("mock://test");
    connector.connect(&config).await?;
    
    // Test list tables
    let tables = connector.list_tables().await?;
    assert!(tables.contains(&"users".to_string()));
    
    // Test get table info
    let table_info = connector.get_table_info("users").await?;
    assert_eq!(table_info.name, "users");
    assert!(!table_info.columns.is_empty());
    assert!(table_info.row_count.is_some());
    
    // Test non-existent table
    let result = connector.get_table_info("non_existent").await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_database_execute_operations() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Test INSERT
    let affected = connector.execute("INSERT INTO users (name, age) VALUES ('John', 30)").await?;
    assert_eq!(affected, 1);
    
    // Test UPDATE
    let affected = connector.execute("UPDATE users SET age = 31 WHERE name = 'John'").await?;
    assert_eq!(affected, 5); // Mock returns 5 for UPDATE operations
    
    // Test other operations
    let affected = connector.execute("CREATE TABLE test (id INTEGER)").await?;
    assert_eq!(affected, 0);
    
    Ok(())
}

#[tokio::test]
async fn test_database_transaction_management() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Test transaction creation
    let tx_id = connector.begin_transaction().await?;
    assert!(tx_id.starts_with("mock_tx_"));
    
    // In a real implementation, you would use the transaction ID
    // to track and manage the transaction state
    
    Ok(())
}

#[tokio::test]
async fn test_database_configuration_options() -> Result<()> {
    // Test various configuration options
    let config = DatabaseConfig::new("postgresql://localhost/test")
        .with_pool_size(20)
        .with_timeout(60)
        .with_ssl()
        .with_parameter("sslmode", "require")
        .with_parameter("application_name", "pandrs_test");
    
    assert_eq!(config.pool_size, Some(20));
    assert_eq!(config.timeout, Some(60));
    assert!(config.ssl);
    assert_eq!(config.parameters.get("sslmode"), Some(&"require".to_string()));
    assert_eq!(config.parameters.get("application_name"), Some(&"pandrs_test".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_database_error_handling() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    
    // Test operations without connection
    let result = connector.query("SELECT 1").await;
    assert!(result.is_err());
    
    let result = connector.execute("INSERT INTO test VALUES (1)").await;
    assert!(result.is_err());
    
    let result = connector.write_table(&test_utils::create_test_dataframe()?, "test", WriteMode::Replace).await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_database_large_dataset_handling() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Create large DataFrame
    let large_df = test_utils::create_large_test_dataframe(10000)?;
    
    // Test writing large dataset
    connector.write_table(&large_df, "large_table", WriteMode::Replace).await?;
    
    // Test querying (mock will return standard test data)
    let result_df = connector.query("SELECT * FROM large_table").await?;
    assert!(result_df.row_count() > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_database_concurrent_operations() -> Result<()> {
    use tokio::task;
    
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Test concurrent queries (note: in real implementation, you'd need
    // connection pooling to handle concurrent operations properly)
    let handles: Vec<_> = (0..5).map(|i| {
        let query = format!("SELECT * FROM table_{}", i);
        task::spawn(async move {
            // Create new connector for each task in real scenario
            let mut local_connector = MockDatabaseConnector::new();
            let local_config = DatabaseConfig::new("mock://test");
            local_connector.connect(&local_config).await.unwrap();
            local_connector.query(&query).await
        })
    }).collect();
    
    // Wait for all queries to complete
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_database_data_type_handling() -> Result<()> {
    let mut connector = MockDatabaseConnector::new();
    let config = DatabaseConfig::new("mock://test");
    
    connector.connect(&config).await?;
    
    // Create DataFrame with various data types
    let df = test_utils::create_test_dataframe()?;
    
    // Test writing and reading back
    connector.write_table(&df, "type_test", WriteMode::Replace).await?;
    let result_df = connector.query("SELECT * FROM type_test").await?;
    
    // Verify basic structure (mock returns standard test data)
    assert_eq!(result_df.column_names().len(), 4);
    assert!(result_df.column_names().contains(&"id".to_string()));
    assert!(result_df.column_names().contains(&"name".to_string()));
    
    Ok(())
}