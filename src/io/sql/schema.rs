use std::collections::HashMap;

// SQLite support
use rusqlite::Connection as SqliteConnection;

// Multi-database support via sqlx (optional)
#[cfg(feature = "sql")]
use sqlx::{AnyPool, Pool, Row as SqlxRow};

use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;

use super::connection::{DatabaseConnection, SqlConnection};

/// Database table schema information
#[derive(Debug, Clone)]
pub struct TableSchema {
    /// Table name
    pub name: String,
    /// Column definitions
    pub columns: Vec<ColumnDefinition>,
    /// Primary key columns
    pub primary_keys: Vec<String>,
    /// Foreign key constraints
    pub foreign_keys: Vec<ForeignKey>,
    /// Indexes on the table
    pub indexes: Vec<IndexDefinition>,
}

/// Column definition in database table
#[derive(Debug, Clone)]
pub struct ColumnDefinition {
    /// Column name
    pub name: String,
    /// SQL data type
    pub data_type: String,
    /// Whether column allows NULL values
    pub nullable: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Maximum length (for varchar/char types)
    pub max_length: Option<u32>,
    /// Numeric precision (for decimal types)
    pub precision: Option<u32>,
    /// Numeric scale (for decimal types)
    pub scale: Option<u32>,
    /// Whether column is auto-increment
    pub auto_increment: bool,
}

/// Foreign key constraint information
#[derive(Debug, Clone)]
pub struct ForeignKey {
    /// Local column name
    pub column: String,
    /// Referenced table name
    pub referenced_table: String,
    /// Referenced column name
    pub referenced_column: String,
    /// ON DELETE action
    pub on_delete: Option<String>,
    /// ON UPDATE action
    pub on_update: Option<String>,
}

/// Index definition
#[derive(Debug, Clone)]
pub struct IndexDefinition {
    /// Index name
    pub name: String,
    /// Columns in the index
    pub columns: Vec<String>,
    /// Whether index is unique
    pub unique: bool,
    /// Index type (e.g., BTREE, HASH)
    pub index_type: Option<String>,
}

/// Check if table exists in database
///
/// # Arguments
///
/// * `table_name` - Name of the table to check
/// * `connection` - Database connection
/// * `schema` - Optional schema name
///
/// # Returns
///
/// * `Result<bool>` - True if table exists, false otherwise, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::sql::{has_table, SqlConnection};
///
/// let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
/// let exists = has_table("users", &conn, None).unwrap();
/// println!("Table exists: {}", exists);
/// ```
pub fn has_table(
    table_name: &str,
    connection: &SqlConnection,
    schema: Option<&str>,
) -> Result<bool> {
    match connection.connection_type() {
        DatabaseConnection::Sqlite(path) => {
            let conn = SqliteConnection::open(path)
                .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

            let exists = conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
                .map_err(|e| Error::IoError(format!("Failed to prepare query: {}", e)))?
                .exists(&[&table_name])
                .map_err(|e| Error::IoError(format!("Failed to check table existence: {}", e)))?;

            Ok(exists)
        }
        #[cfg(feature = "sql")]
        _ => {
            // Implementation for other databases would go here
            Err(Error::IoError(
                "Table existence check not yet implemented for non-SQLite databases".to_string(),
            ))
        }
    }
}

/// List all tables in database
///
/// # Arguments
///
/// * `connection` - Database connection
/// * `schema` - Optional schema name to filter by
///
/// # Returns
///
/// * `Result<Vec<String>>` - List of table names, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::sql::{list_tables, SqlConnection};
///
/// let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
/// let tables = list_tables(&conn, None).unwrap();
/// for table in tables {
///     println!("Table: {}", table);
/// }
/// ```
pub fn list_tables(connection: &SqlConnection, schema: Option<&str>) -> Result<Vec<String>> {
    match connection.connection_type() {
        DatabaseConnection::Sqlite(path) => {
            let conn = SqliteConnection::open(path)
                .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

            let mut stmt = conn
                .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                .map_err(|e| Error::IoError(format!("Failed to prepare query: {}", e)))?;

            let table_names = stmt
                .query_map([], |row| Ok(row.get::<_, String>(0)?))
                .map_err(|e| Error::IoError(format!("Failed to execute query: {}", e)))?
                .collect::<std::result::Result<Vec<String>, _>>()
                .map_err(|e| Error::IoError(format!("Failed to collect results: {}", e)))?;

            Ok(table_names)
        }
        #[cfg(feature = "sql")]
        _ => {
            // Implementation for other databases would go here
            Err(Error::IoError(
                "Table listing not yet implemented for non-SQLite databases".to_string(),
            ))
        }
    }
}

/// Get schema information for a table
///
/// # Arguments
///
/// * `table_name` - Name of the table
/// * `connection` - Database connection
/// * `schema` - Optional schema name
///
/// # Returns
///
/// * `Result<TableSchema>` - Table schema information, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::sql::{get_table_schema, SqlConnection};
///
/// let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
/// let schema = get_table_schema("users", &conn, None).unwrap();
/// println!("Table {} has {} columns", schema.name, schema.columns.len());
/// ```
pub fn get_table_schema(
    table_name: &str,
    connection: &SqlConnection,
    schema: Option<&str>,
) -> Result<TableSchema> {
    match connection.connection_type() {
        DatabaseConnection::Sqlite(path) => get_sqlite_table_schema(table_name, path),
        #[cfg(feature = "sql")]
        _ => {
            // Implementation for other databases would go here
            Err(Error::IoError(
                "Schema inspection not yet implemented for non-SQLite databases".to_string(),
            ))
        }
    }
}

/// Generate CREATE TABLE SQL for DataFrame
///
/// # Arguments
///
/// * `df` - DataFrame to generate SQL for
/// * `table_name` - Target table name
/// * `connection` - Database connection (for dialect-specific SQL)
///
/// # Returns
///
/// * `Result<String>` - CREATE TABLE SQL statement, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::sql::{get_create_table_sql, SqlConnection};
///
/// let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
/// let sql = get_create_table_sql(&df, "users", &conn).unwrap();
/// println!("CREATE SQL: {}", sql);
/// ```
pub fn get_create_table_sql(
    df: &OptimizedDataFrame,
    table_name: &str,
    connection: &SqlConnection,
) -> Result<String> {
    let mut columns = Vec::new();

    for col_name in df.column_names() {
        if let Ok(column) = df.column(col_name) {
            let sql_type = match column.column_type() {
                crate::column::ColumnType::Int64 => "INTEGER",
                crate::column::ColumnType::Float64 => "REAL",
                crate::column::ColumnType::Boolean => match connection.connection_type() {
                    DatabaseConnection::Sqlite(_) => "INTEGER",
                    #[cfg(feature = "sql")]
                    DatabaseConnection::PostgreSQL(_) => "BOOLEAN",
                    #[cfg(feature = "sql")]
                    DatabaseConnection::MySQL(_) => "BOOLEAN",
                    #[cfg(feature = "sql")]
                    DatabaseConnection::Generic(_) => "BOOLEAN",
                },
                crate::column::ColumnType::String => "TEXT",
            };
            columns.push(format!("{} {}", col_name, sql_type));
        }
    }

    Ok(format!(
        "CREATE TABLE {} ({})",
        table_name,
        columns.join(", ")
    ))
}

/// Get detailed table schema for SQLite database
fn get_sqlite_table_schema(table_name: &str, db_path: &str) -> Result<TableSchema> {
    let conn = SqliteConnection::open(db_path)
        .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

    // Get column information
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({})", table_name))
        .map_err(|e| Error::IoError(format!("Failed to prepare query: {}", e)))?;

    let columns = stmt
        .query_map([], |row| {
            Ok(ColumnDefinition {
                name: row.get::<_, String>(1)?,
                data_type: row.get::<_, String>(2)?,
                nullable: row.get::<_, i32>(3)? == 0,
                default_value: row.get::<_, Option<String>>(4)?,
                max_length: None, // SQLite doesn't enforce length constraints
                precision: None,
                scale: None,
                auto_increment: false, // Would need additional logic to detect
            })
        })
        .map_err(|e| Error::IoError(format!("Failed to execute query: {}", e)))?
        .collect::<std::result::Result<Vec<ColumnDefinition>, _>>()
        .map_err(|e| Error::IoError(format!("Failed to collect results: {}", e)))?;

    // Get primary key information
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({})", table_name))
        .map_err(|e| Error::IoError(format!("Failed to prepare query: {}", e)))?;

    let primary_keys = stmt
        .query_map([], |row| {
            let is_pk: i32 = row.get(5)?;
            if is_pk > 0 {
                Ok(Some(row.get::<_, String>(1)?))
            } else {
                Ok(None)
            }
        })
        .map_err(|e| Error::IoError(format!("Failed to execute query: {}", e)))?
        .filter_map(|result| result.transpose())
        .collect::<std::result::Result<Vec<String>, _>>()
        .map_err(|e| Error::IoError(format!("Failed to collect results: {}", e)))?;

    // Get foreign key information
    let foreign_keys = get_sqlite_foreign_keys(&conn, table_name)?;

    // Get index information
    let indexes = get_sqlite_indexes(&conn, table_name)?;

    Ok(TableSchema {
        name: table_name.to_string(),
        columns,
        primary_keys,
        foreign_keys,
        indexes,
    })
}

/// Get foreign key constraints for SQLite table
fn get_sqlite_foreign_keys(conn: &SqliteConnection, table_name: &str) -> Result<Vec<ForeignKey>> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA foreign_key_list({})", table_name))
        .map_err(|e| Error::IoError(format!("Failed to prepare foreign key query: {}", e)))?;

    let foreign_keys = stmt
        .query_map([], |row| {
            Ok(ForeignKey {
                column: row.get::<_, String>(3)?,
                referenced_table: row.get::<_, String>(2)?,
                referenced_column: row.get::<_, String>(4)?,
                on_update: row.get::<_, Option<String>>(5)?,
                on_delete: row.get::<_, Option<String>>(6)?,
            })
        })
        .map_err(|e| Error::IoError(format!("Failed to execute foreign key query: {}", e)))?
        .collect::<std::result::Result<Vec<ForeignKey>, _>>()
        .map_err(|e| Error::IoError(format!("Failed to collect foreign key results: {}", e)))?;

    Ok(foreign_keys)
}

/// Get index information for SQLite table
fn get_sqlite_indexes(conn: &SqliteConnection, table_name: &str) -> Result<Vec<IndexDefinition>> {
    // Get list of indexes
    let mut stmt = conn
        .prepare(&format!("PRAGMA index_list({})", table_name))
        .map_err(|e| Error::IoError(format!("Failed to prepare index list query: {}", e)))?;

    let index_names_and_unique = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(1)?,   // index name
                row.get::<_, i32>(2)? == 1, // unique flag
            ))
        })
        .map_err(|e| Error::IoError(format!("Failed to execute index list query: {}", e)))?
        .collect::<std::result::Result<Vec<(String, bool)>, _>>()
        .map_err(|e| Error::IoError(format!("Failed to collect index list results: {}", e)))?;

    let mut indexes = Vec::new();

    // Get column information for each index
    for (index_name, unique) in index_names_and_unique {
        let mut stmt = conn
            .prepare(&format!("PRAGMA index_info({})", index_name))
            .map_err(|e| Error::IoError(format!("Failed to prepare index info query: {}", e)))?;

        let columns = stmt
            .query_map([], |row| {
                Ok(row.get::<_, String>(2)?) // column name
            })
            .map_err(|e| Error::IoError(format!("Failed to execute index info query: {}", e)))?
            .collect::<std::result::Result<Vec<String>, _>>()
            .map_err(|e| Error::IoError(format!("Failed to collect index info results: {}", e)))?;

        indexes.push(IndexDefinition {
            name: index_name,
            columns,
            unique,
            index_type: None, // SQLite doesn't provide this information easily
        });
    }

    Ok(indexes)
}

/// Database schema introspection utilities
pub struct SchemaIntrospector {
    /// Connection pool
    #[cfg(feature = "sql")]
    pool: AnyPool,
}

impl SchemaIntrospector {
    /// Create new schema introspector
    #[cfg(feature = "sql")]
    pub fn new(pool: AnyPool) -> Self {
        Self { pool }
    }

    /// Get all table names in the database
    ///
    /// # Returns
    ///
    /// * `Result<Vec<String>>` - List of table names
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::SchemaIntrospector;
    ///
    /// async fn example(inspector: &SchemaIntrospector) {
    ///     let tables = inspector.list_tables().await.unwrap();
    ///     for table in tables {
    ///         println!("Table: {}", table);
    ///     }
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn list_tables(&self) -> Result<Vec<String>> {
        use sqlx::Row;

        let query = r#"
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        "#;

        let rows = sqlx::query(query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::IoError(format!("Failed to list tables: {}", e)))?;

        let mut tables = Vec::new();
        for row in rows {
            let table_name: String = row.get(0);
            tables.push(table_name);
        }

        Ok(tables)
    }

    /// Get detailed schema information for a table
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the table to inspect
    ///
    /// # Returns
    ///
    /// * `Result<TableSchema>` - Detailed table schema
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::SchemaIntrospector;
    ///
    /// async fn example(inspector: &SchemaIntrospector) {
    ///     let schema = inspector.describe_table("users").await.unwrap();
    ///     println!("Table has {} columns", schema.columns.len());
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn describe_table(&self, table_name: &str) -> Result<TableSchema> {
        use sqlx::Row;

        let query = r#"
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = $1
            ORDER BY ordinal_position
        "#;

        let rows = sqlx::query(query)
            .bind(table_name)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::IoError(format!("Failed to describe table: {}", e)))?;

        let mut columns = Vec::new();
        for row in rows {
            let column_name: String = row.get(0);
            let data_type: String = row.get(1);
            let is_nullable: String = row.get(2);
            let column_default: Option<String> = row.get(3);

            columns.push(ColumnDefinition {
                name: column_name,
                data_type,
                nullable: is_nullable == "YES",
                default_value: column_default,
                max_length: None,      // Would need additional query
                precision: None,       // Would need additional query
                scale: None,           // Would need additional query
                auto_increment: false, // Would need additional query
            });
        }

        Ok(TableSchema {
            name: table_name.to_string(),
            columns,
            primary_keys: Vec::new(), // Would need additional query
            foreign_keys: Vec::new(), // Would need additional query
            indexes: Vec::new(),      // Would need additional query
        })
    }

    /// Get all schemas in the database
    ///
    /// # Returns
    ///
    /// * `Result<Vec<String>>` - List of schema names
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::SchemaIntrospector;
    ///
    /// async fn example(inspector: &SchemaIntrospector) {
    ///     let schemas = inspector.list_schemas().await.unwrap();
    ///     for schema in schemas {
    ///         println!("Schema: {}", schema);
    ///     }
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn list_schemas(&self) -> Result<Vec<String>> {
        use sqlx::Row;

        let query = r#"
            SELECT schema_name 
            FROM information_schema.schemata
            ORDER BY schema_name
        "#;

        let rows = sqlx::query(query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::IoError(format!("Failed to list schemas: {}", e)))?;

        let mut schemas = Vec::new();
        for row in rows {
            let schema_name: String = row.get(0);
            schemas.push(schema_name);
        }

        Ok(schemas)
    }

    /// Get column statistics for a table
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the table
    /// * `schema_name` - Optional schema name
    ///
    /// # Returns
    ///
    /// * `Result<HashMap<String, ColumnStats>>` - Column statistics by column name
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::SchemaIntrospector;
    ///
    /// async fn example(inspector: &SchemaIntrospector) {
    ///     let stats = inspector.get_column_stats("users", None).await.unwrap();
    ///     for (column, stat) in stats {
    ///         println!("Column {}: {} rows", column, stat.row_count);
    ///     }
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn get_column_stats(
        &self,
        table_name: &str,
        schema_name: Option<&str>,
    ) -> Result<HashMap<String, ColumnStats>> {
        use sqlx::Row;

        let full_table_name = if let Some(schema) = schema_name {
            format!("{}.{}", schema, table_name)
        } else {
            table_name.to_string()
        };

        // Get column names first
        let column_query = r#"
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = $1
        "#;

        let mut query_builder = sqlx::query(column_query).bind(table_name);
        if let Some(schema) = schema_name {
            query_builder = query_builder.bind(schema);
        }

        let column_rows = query_builder
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::IoError(format!("Failed to get columns: {}", e)))?;

        let mut stats = HashMap::new();

        for row in column_rows {
            let column_name: String = row.get(0);

            // Get statistics for this column
            let stats_query = format!(
                "SELECT COUNT(*), COUNT({}), MIN({}), MAX({}) FROM {}",
                column_name, column_name, column_name, full_table_name
            );

            if let Ok(stat_rows) = sqlx::query(&stats_query).fetch_all(&self.pool).await {
                if let Some(stat_row) = stat_rows.first() {
                    let total_count: i64 = stat_row.try_get(0).unwrap_or(0);
                    let non_null_count: i64 = stat_row.try_get(1).unwrap_or(0);

                    stats.insert(
                        column_name,
                        ColumnStats {
                            row_count: total_count as usize,
                            non_null_count: non_null_count as usize,
                            null_count: (total_count - non_null_count) as usize,
                            min_value: stat_row.try_get::<String, _>(2).ok(),
                            max_value: stat_row.try_get::<String, _>(3).ok(),
                        },
                    );
                }
            }
        }

        Ok(stats)
    }

    /// Analyze table for performance insights
    ///
    /// # Arguments
    ///
    /// * `table_name` - Name of the table to analyze
    ///
    /// # Returns
    ///
    /// * `Result<TableAnalysis>` - Performance analysis results
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::SchemaIntrospector;
    ///
    /// async fn example(inspector: &SchemaIntrospector) {
    ///     let analysis = inspector.analyze_table("users").await.unwrap();
    ///     println!("Table size: {} MB", analysis.size_mb);
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn analyze_table(&self, table_name: &str) -> Result<TableAnalysis> {
        use sqlx::Row;

        // Get basic table statistics
        let stats_query = format!("SELECT COUNT(*) FROM {}", table_name);
        let rows = sqlx::query(&stats_query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::IoError(format!("Failed to analyze table: {}", e)))?;

        let row_count = if let Some(row) = rows.first() {
            row.get::<i64, _>(0) as usize
        } else {
            0
        };

        // Estimate table size (simplified calculation)
        let estimated_size_mb = (row_count as f64 * 100.0) / (1024.0 * 1024.0); // Very rough estimate

        Ok(TableAnalysis {
            table_name: table_name.to_string(),
            row_count,
            size_mb: estimated_size_mb,
            estimated_query_time_ms: estimate_query_time(row_count),
            recommended_indexes: Vec::new(), // Would need more complex analysis
        })
    }
}

/// Column statistics
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Total number of rows
    pub row_count: usize,
    /// Number of non-null values
    pub non_null_count: usize,
    /// Number of null values
    pub null_count: usize,
    /// Minimum value (as string)
    pub min_value: Option<String>,
    /// Maximum value (as string)
    pub max_value: Option<String>,
}

/// Table performance analysis
#[derive(Debug, Clone)]
pub struct TableAnalysis {
    /// Table name
    pub table_name: String,
    /// Number of rows
    pub row_count: usize,
    /// Estimated size in MB
    pub size_mb: f64,
    /// Estimated query time in milliseconds
    pub estimated_query_time_ms: f64,
    /// Recommended indexes for performance
    pub recommended_indexes: Vec<String>,
}

/// Schema comparison utilities
pub struct SchemaComparator;

impl SchemaComparator {
    /// Compare two table schemas
    ///
    /// # Arguments
    ///
    /// * `schema1` - First table schema
    /// * `schema2` - Second table schema
    ///
    /// # Returns
    ///
    /// * `SchemaComparison` - Detailed comparison results
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{SchemaComparator, TableSchema};
    ///
    /// let comparison = SchemaComparator::compare_schemas(&schema1, &schema2);
    /// if !comparison.differences.is_empty() {
    ///     println!("Schemas differ: {:?}", comparison.differences);
    /// }
    /// ```
    pub fn compare_schemas(schema1: &TableSchema, schema2: &TableSchema) -> SchemaComparison {
        let mut differences = Vec::new();

        // Compare table names
        if schema1.name != schema2.name {
            differences.push(SchemaDifference::TableNameDifferent {
                left: schema1.name.clone(),
                right: schema2.name.clone(),
            });
        }

        // Compare columns
        let mut schema1_columns: HashMap<String, &ColumnDefinition> = HashMap::new();
        let mut schema2_columns: HashMap<String, &ColumnDefinition> = HashMap::new();

        for col in &schema1.columns {
            schema1_columns.insert(col.name.clone(), col);
        }

        for col in &schema2.columns {
            schema2_columns.insert(col.name.clone(), col);
        }

        // Check for missing columns
        for col_name in schema1_columns.keys() {
            if !schema2_columns.contains_key(col_name) {
                differences.push(SchemaDifference::ColumnMissing {
                    column_name: col_name.clone(),
                    missing_from: "right".to_string(),
                });
            }
        }

        for col_name in schema2_columns.keys() {
            if !schema1_columns.contains_key(col_name) {
                differences.push(SchemaDifference::ColumnMissing {
                    column_name: col_name.clone(),
                    missing_from: "left".to_string(),
                });
            }
        }

        // Compare existing columns
        for (col_name, col1) in &schema1_columns {
            if let Some(col2) = schema2_columns.get(col_name) {
                if col1.data_type != col2.data_type {
                    differences.push(SchemaDifference::ColumnTypeDifferent {
                        column_name: col_name.clone(),
                        left_type: col1.data_type.clone(),
                        right_type: col2.data_type.clone(),
                    });
                }

                if col1.nullable != col2.nullable {
                    differences.push(SchemaDifference::ColumnNullabilityDifferent {
                        column_name: col_name.clone(),
                        left_nullable: col1.nullable,
                        right_nullable: col2.nullable,
                    });
                }
            }
        }

        SchemaComparison {
            are_identical: differences.is_empty(),
            differences,
        }
    }

    /// Generate SQL to migrate from one schema to another
    ///
    /// # Arguments
    ///
    /// * `from_schema` - Source schema
    /// * `to_schema` - Target schema
    ///
    /// # Returns
    ///
    /// * `Result<Vec<String>>` - List of SQL statements to perform migration
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{SchemaComparator, TableSchema};
    ///
    /// let migration_sql = SchemaComparator::generate_migration(&old_schema, &new_schema).unwrap();
    /// for sql in migration_sql {
    ///     println!("Migration SQL: {}", sql);
    /// }
    /// ```
    pub fn generate_migration(
        from_schema: &TableSchema,
        to_schema: &TableSchema,
    ) -> Result<Vec<String>> {
        let comparison = Self::compare_schemas(from_schema, to_schema);
        let mut sql_statements = Vec::new();

        for difference in &comparison.differences {
            match difference {
                SchemaDifference::ColumnMissing {
                    column_name,
                    missing_from,
                } => {
                    if missing_from == "right" {
                        // Column exists in left but not right - need to add it
                        if let Some(col_def) =
                            from_schema.columns.iter().find(|c| &c.name == column_name)
                        {
                            let mut add_column_sql = format!(
                                "ALTER TABLE {} ADD COLUMN {} {}",
                                to_schema.name, col_def.name, col_def.data_type
                            );

                            if !col_def.nullable {
                                add_column_sql.push_str(" NOT NULL");
                            }

                            if let Some(ref default) = col_def.default_value {
                                add_column_sql.push_str(&format!(" DEFAULT {}", default));
                            }

                            sql_statements.push(add_column_sql);
                        }
                    }
                    // Note: Dropping columns is more complex and dangerous, so we don't auto-generate those
                }
                SchemaDifference::ColumnTypeDifferent {
                    column_name,
                    right_type,
                    ..
                } => {
                    // Generate ALTER COLUMN statement (syntax varies by database)
                    sql_statements.push(format!(
                        "ALTER TABLE {} ALTER COLUMN {} TYPE {}",
                        to_schema.name, column_name, right_type
                    ));
                }
                _ => {
                    // Other differences would require more complex handling
                }
            }
        }

        Ok(sql_statements)
    }
}

/// Schema comparison results
#[derive(Debug)]
pub struct SchemaComparison {
    /// Whether the schemas are identical
    pub are_identical: bool,
    /// List of differences found
    pub differences: Vec<SchemaDifference>,
}

/// Types of schema differences
#[derive(Debug)]
pub enum SchemaDifference {
    /// Table names are different
    TableNameDifferent { left: String, right: String },
    /// Column is missing from one schema
    ColumnMissing {
        column_name: String,
        missing_from: String,
    },
    /// Column types are different
    ColumnTypeDifferent {
        column_name: String,
        left_type: String,
        right_type: String,
    },
    /// Column nullability is different
    ColumnNullabilityDifferent {
        column_name: String,
        left_nullable: bool,
        right_nullable: bool,
    },
    /// Primary key differences
    PrimaryKeyDifferent {
        left_keys: Vec<String>,
        right_keys: Vec<String>,
    },
    /// Index differences
    IndexDifferent {
        index_name: String,
        difference_type: String,
    },
}

/// Estimate query time based on row count (very rough estimation)
fn estimate_query_time(row_count: usize) -> f64 {
    // Very simple estimation: 1ms per 1000 rows for a basic SELECT
    (row_count as f64 / 1000.0).max(1.0)
}
