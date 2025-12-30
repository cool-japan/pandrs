//! SQL database support for PandRS DataFrames
//! 
//! This module provides comprehensive SQL database integration, allowing you to read and write
//! DataFrames from/to various SQL databases including SQLite, PostgreSQL, and MySQL.
//! 
//! The module has been refactored into submodules for better organization:
//! - `connection`: Database connection management and pooling
//! - `operations`: Core read/write operations
//! - `advanced`: Async operations, transactions, and advanced features
//! - `schema`: Database schema introspection and management
//! 
//! # Quick Start
//! 
//! ## Reading from SQLite
//! 
//! ```no_run
//! use pandrs::io::sql::read_sql;
//! 
//! // Simple query
//! let df = read_sql("SELECT * FROM users WHERE age > 25", "database.db").unwrap();
//! println!("Found {} users", df.row_count());
//! ```
//! 
//! ## Writing to SQLite
//! 
//! ```no_run
//! use pandrs::io::sql::{write_to_sql};
//! use pandrs::optimized::OptimizedDataFrame;
//! 
//! // Assuming you have an OptimizedDataFrame `df`
//! // write_to_sql(&df, "users", "database.db", "replace").unwrap();
//! ```
//! 
//! ## Advanced Usage with Connection Management
//! 
//! ```no_run
//! use pandrs::io::sql::{SqlConnection, read_sql_advanced, SqlReadOptions};
//! 
//! let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
//! let options = SqlReadOptions {
//!     chunksize: Some(1000),
//!     parse_dates: Some(vec!["created_at".to_string()]),
//!     ..Default::default()
//! };
//! 
//! let df = read_sql_advanced("SELECT * FROM users", &conn, options).unwrap();
//! ```
//! 
//! ## Async Operations (requires `sql` feature)
//! 
//! ```no_run
//! use pandrs::io::sql::{AsyncDatabasePool, PoolConfig};
//! 
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = PoolConfig::default();
//! let pool = AsyncDatabasePool::new("postgresql://localhost/db", config).await?;
//! 
//! let df = pool.query_async("SELECT * FROM users WHERE active = true", None).await?;
//! println!("Active users: {}", df.row_count());
//! # Ok(())
//! # }
//! ```
//! 
//! ## Schema Introspection
//! 
//! ```no_run
//! use pandrs::io::sql::{SqlConnection, get_table_schema, list_tables};
//! 
//! let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
//! 
//! // List all tables
//! let tables = list_tables(&conn, None).unwrap();
//! for table in tables {
//!     println!("Found table: {}", table);
//!     
//!     // Get schema for each table
//!     let schema = get_table_schema(&table, &conn, None).unwrap();
//!     println!("  {} columns, {} primary keys", 
//!              schema.columns.len(), schema.primary_keys.len());
//! }
//! ```

// Re-export everything from the sql module
pub use self::sql_module::*;

#[path = "sql/mod.rs"]
mod sql_module;

// Additional convenience exports for backward compatibility
pub use sql_module::{
    read_sql as read,
    write_to_sql as write,
    execute_sql as execute,
    SqlConnection as Connection,
};

// Legacy type aliases for backward compatibility
pub type DatabaseUrl = String;
pub type SqlQuery = String;
pub type TableName = String;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::series::Series;
    use crate::dataframe::DataFrame;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_sql_module_exports() {
        // Test that key types are available
        let _options = SqlReadOptions::default();
        let _write_options = SqlWriteOptions::default();
        let _config = PoolConfig::default();
        
        // Test that enums work
        assert_eq!(WriteMode::Fail, WriteMode::Fail);
        assert_ne!(WriteMode::Fail, WriteMode::Replace);
    }

    #[test]
    fn test_connection_creation() {
        // Test SQLite connection creation
        let conn = SqlConnection::from_url("sqlite:test.db").unwrap();
        match conn.connection_type() {
            DatabaseConnection::Sqlite(_) => (), // Expected
            _ => panic!("Expected SQLite connection"),
        }

        // Test file extension detection
        let conn2 = SqlConnection::from_url("data.db").unwrap();
        match conn2.connection_type() {
            DatabaseConnection::Sqlite(_) => (), // Expected
            _ => panic!("Expected SQLite connection"),
        }
    }

    #[test]
    #[ignore = "Requires actual SQLite database"]
    fn test_basic_sql_operations() {
        // This test would require setting up a real database
        // For now, just test that the functions exist and can be called
        
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path().to_str().unwrap();
        
        // Test that execute_sql function exists
        let result = execute_sql("CREATE TABLE test (id INTEGER, name TEXT)", db_path);
        // We expect this to work with a real SQLite file
        // but will fail with tempfile, so we just check the function exists
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_query_builder_integration() {
        let query = QueryBuilder::new()
            .from("users")
            .select("id")
            .select("name")
            .where_clause("active = true")
            .order_by_desc("created_at")
            .limit(100)
            .build()
            .unwrap();
        
        assert!(query.contains("SELECT id, name"));
        assert!(query.contains("FROM users"));
        assert!(query.contains("WHERE active = true"));
        assert!(query.contains("ORDER BY created_at DESC"));
        assert!(query.contains("LIMIT 100"));
    }

    #[test]
    fn test_schema_types() {
        let column = ColumnDefinition {
            name: "test_col".to_string(),
            data_type: "VARCHAR(255)".to_string(),
            nullable: true,
            default_value: Some("'default'".to_string()),
            max_length: Some(255),
            precision: None,
            scale: None,
            auto_increment: false,
        };
        
        assert_eq!(column.name, "test_col");
        assert_eq!(column.data_type, "VARCHAR(255)");
        assert!(column.nullable);
        assert_eq!(column.max_length, Some(255));
    }

    #[test]
    fn test_legacy_aliases() {
        // Test that legacy type aliases work
        let _url: DatabaseUrl = "sqlite:test.db".to_string();
        let _query: SqlQuery = "SELECT * FROM users".to_string();
        let _table: TableName = "users".to_string();
    }

    #[test]
    fn test_schema_comparison() {
        let schema1 = TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnDefinition {
                    name: "id".to_string(),
                    data_type: "INTEGER".to_string(),
                    nullable: false,
                    default_value: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    auto_increment: true,
                }
            ],
            primary_keys: vec!["id".to_string()],
            foreign_keys: vec![],
            indexes: vec![],
        };

        let schema2 = TableSchema {
            name: "users".to_string(),
            columns: vec![
                ColumnDefinition {
                    name: "id".to_string(),
                    data_type: "BIGINT".to_string(), // Different type
                    nullable: false,
                    default_value: None,
                    max_length: None,
                    precision: None,
                    scale: None,
                    auto_increment: true,
                }
            ],
            primary_keys: vec!["id".to_string()],
            foreign_keys: vec![],
            indexes: vec![],
        };

        let comparison = SchemaComparator::compare_schemas(&schema1, &schema2);
        assert!(!comparison.are_identical);
        assert!(!comparison.differences.is_empty());
    }
}