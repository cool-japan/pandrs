//! SQL module for PandRS
//!
//! This module provides comprehensive SQL database support for reading and writing DataFrames.
//! It is organized into several submodules for better maintainability:
//!
//! - `connection`: Database connection management and configuration
//! - `operations`: Core SQL operations and DataFrame conversion
//! - `advanced`: Advanced features including async operations and transactions
//! - `schema`: Schema management and introspection utilities
//!
//! # Basic Usage
//!
//! ```no_run
//! use pandrs::io::sql::{read_sql, write_to_sql, SqlConnection};
//!
//! // Read from database
//! let df = read_sql("SELECT * FROM users", "database.db").unwrap();
//!
//! // Write to database with connection
//! let conn = SqlConnection::from_url("sqlite:database.db").unwrap();
//! // write_sql_advanced(&df, "users", &conn, SqlWriteOptions::default()).unwrap();
//! ```
//!
//! # Advanced Usage
//!
//! ```no_run
//! use pandrs::io::sql::{AsyncDatabasePool, PoolConfig, SqlReadOptions};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Async operations with connection pooling
//! let config = PoolConfig::default();
//! let pool = AsyncDatabasePool::new("postgresql://localhost/db", config).await?;
//! let df = pool.query_async("SELECT * FROM users WHERE active = true", None).await?;
//! # Ok(())
//! # }
//! ```

pub mod advanced;
pub mod connection;
pub mod operations;
pub mod schema;

// Re-export commonly used types and functions from each module

// Connection management
pub use connection::{
    AsyncDatabasePool, ConnectionStats, DatabaseConnection, DatabaseOperation, IsolationLevel,
    PoolConfig, SqlConnection, SqlValue, TransactionManager,
};

// Core operations
pub use operations::{
    execute_sql, read_sql, read_sql_advanced, read_sql_table, write_sql_advanced, write_to_sql,
    InsertMethod, SqlReadOptions, SqlWriteOptions, WriteMode,
};

// Schema management
pub use schema::{
    get_create_table_sql, get_table_schema, has_table, list_tables, ColumnDefinition, ColumnStats,
    ForeignKey, IndexDefinition, SchemaComparator, SchemaComparison, SchemaDifference,
    SchemaIntrospector, TableAnalysis, TableSchema,
};

// Advanced features
pub use advanced::{AdvancedConnectionPool, HealthCheckResult, QueryBuilder, RetrySettings};

// Convenience re-exports for backward compatibility and ease of use
pub use connection::SqlConnection as Connection;
pub use operations::{read_sql as read, write_to_sql as write};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;
    use std::collections::HashMap;

    #[test]
    fn test_sql_read_options_default() {
        let options = SqlReadOptions::default();
        assert!(options.chunksize.is_none());
        assert!(options.index_col.is_none());
        assert!(options.parse_dates.is_none());
        assert!(options.dtype.is_none());
        assert!(options.params.is_none());
        assert!(options.coerce_float);
    }

    #[test]
    fn test_sql_write_options_default() {
        let options = SqlWriteOptions::default();
        assert!(options.schema.is_none());
        assert!(options.index);
        assert!(options.index_label.is_none());
        assert_eq!(options.chunksize, Some(10000));
        assert!(options.dtype.is_none());
        assert_eq!(options.if_exists, WriteMode::Fail);
        assert!(matches!(options.method, InsertMethod::Multi));
    }

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.max_connections, 10);
        assert_eq!(config.min_connections, 1);
        assert_eq!(config.connect_timeout.as_secs(), 30);
        assert!(config.idle_timeout.is_some());
        assert_eq!(config.idle_timeout.unwrap().as_secs(), 600);
    }

    #[test]
    fn test_sql_value_to_sql() {
        use rusqlite::ToSql;

        let null_val = SqlValue::Null;
        let int_val = SqlValue::Integer(42);
        let real_val = SqlValue::Real(3.14);
        let text_val = SqlValue::Text("hello".to_string());
        let bool_val = SqlValue::Boolean(true);

        // Test that conversion doesn't panic
        assert!(null_val.to_sql().is_ok());
        assert!(int_val.to_sql().is_ok());
        assert!(real_val.to_sql().is_ok());
        assert!(text_val.to_sql().is_ok());
        assert!(bool_val.to_sql().is_ok());
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new()
            .from("users")
            .select("name")
            .select("age")
            .where_clause("age > 18")
            .order_by_desc("created_at")
            .limit(10)
            .build()
            .unwrap();

        assert!(query.contains("SELECT name, age"));
        assert!(query.contains("FROM users"));
        assert!(query.contains("WHERE age > 18"));
        assert!(query.contains("ORDER BY created_at DESC"));
        assert!(query.contains("LIMIT 10"));
    }

    #[test]
    fn test_query_builder_complex() {
        let query = QueryBuilder::new()
            .from("users")
            .select_many(&["u.name", "u.email", "p.title"])
            .inner_join("profiles p ON u.id = p.user_id")
            .where_clause("u.active = true")
            .and_where("p.public = true")
            .group_by("u.id")
            .having("COUNT(p.id) > 0")
            .order_by_asc("u.name")
            .limit(50)
            .offset(10)
            .build()
            .unwrap();

        assert!(query.contains("SELECT u.name, u.email, p.title"));
        assert!(query.contains("INNER JOIN profiles p ON u.id = p.user_id"));
        assert!(query.contains("WHERE u.active = true AND p.public = true"));
        assert!(query.contains("GROUP BY u.id"));
        assert!(query.contains("HAVING COUNT(p.id) > 0"));
        assert!(query.contains("ORDER BY u.name ASC"));
        assert!(query.contains("LIMIT 50"));
        assert!(query.contains("OFFSET 10"));
    }

    #[test]
    fn test_database_connection_from_url() {
        // Test SQLite URL
        let conn = SqlConnection::from_url("sqlite:test.db").unwrap();
        match conn.connection_type() {
            DatabaseConnection::Sqlite(path) => assert!(path.contains("test.db")),
            _ => panic!("Expected SQLite connection"),
        }

        // Test file extension detection
        let conn = SqlConnection::from_url("data.db").unwrap();
        match conn.connection_type() {
            DatabaseConnection::Sqlite(path) => assert!(path.contains("data.db")),
            _ => panic!("Expected SQLite connection"),
        }
    }

    #[test]
    fn test_write_mode_equality() {
        assert_eq!(WriteMode::Fail, WriteMode::Fail);
        assert_eq!(WriteMode::Replace, WriteMode::Replace);
        assert_eq!(WriteMode::Append, WriteMode::Append);
        assert_ne!(WriteMode::Fail, WriteMode::Replace);
    }

    #[test]
    fn test_column_definition() {
        let col_def = ColumnDefinition {
            name: "id".to_string(),
            data_type: "INTEGER".to_string(),
            nullable: false,
            default_value: None,
            max_length: None,
            precision: None,
            scale: None,
            auto_increment: true,
        };

        assert_eq!(col_def.name, "id");
        assert_eq!(col_def.data_type, "INTEGER");
        assert!(!col_def.nullable);
        assert!(col_def.auto_increment);
    }

    #[test]
    fn test_foreign_key() {
        let fk = ForeignKey {
            column: "user_id".to_string(),
            referenced_table: "users".to_string(),
            referenced_column: "id".to_string(),
            on_delete: Some("CASCADE".to_string()),
            on_update: Some("RESTRICT".to_string()),
        };

        assert_eq!(fk.column, "user_id");
        assert_eq!(fk.referenced_table, "users");
        assert_eq!(fk.on_delete, Some("CASCADE".to_string()));
    }

    #[test]
    fn test_index_definition() {
        let index = IndexDefinition {
            name: "idx_user_email".to_string(),
            columns: vec!["email".to_string()],
            unique: true,
            index_type: Some("BTREE".to_string()),
        };

        assert_eq!(index.name, "idx_user_email");
        assert_eq!(index.columns.len(), 1);
        assert!(index.unique);
    }

    #[test]
    fn test_table_schema() {
        let schema = TableSchema {
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
                },
                ColumnDefinition {
                    name: "name".to_string(),
                    data_type: "TEXT".to_string(),
                    nullable: false,
                    default_value: None,
                    max_length: Some(255),
                    precision: None,
                    scale: None,
                    auto_increment: false,
                },
            ],
            primary_keys: vec!["id".to_string()],
            foreign_keys: vec![],
            indexes: vec![],
        };

        assert_eq!(schema.name, "users");
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.primary_keys.len(), 1);
        assert_eq!(schema.primary_keys[0], "id");
    }

    #[test]
    fn test_retry_settings_default() {
        let settings = RetrySettings::default();
        assert_eq!(settings.max_retries, 3);
        assert_eq!(settings.base_delay.as_millis(), 100);
        assert_eq!(settings.max_delay.as_secs(), 5);
        assert_eq!(settings.backoff_factor, 2.0);
    }
}
