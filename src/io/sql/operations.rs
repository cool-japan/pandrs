use std::collections::HashMap;
use std::path::Path;

// SQLite support
use rusqlite::{Connection as SqliteConnection, Row, Statement};

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use crate::series::Series;

use super::connection::{DatabaseConnection, SqlConnection, SqlValue};

/// Advanced SQL reading options
#[derive(Debug, Clone)]
pub struct SqlReadOptions {
    /// Read data in chunks of this size (for memory efficiency)
    pub chunksize: Option<usize>,
    /// Column(s) to use as index
    pub index_col: Option<Vec<String>>,
    /// Parse specific columns as dates
    pub parse_dates: Option<Vec<String>>,
    /// Override data types for specific columns
    pub dtype: Option<HashMap<String, String>>,
    /// Parameters for parameterized queries
    pub params: Option<Vec<SqlValue>>,
    /// Coerce floating point numbers
    pub coerce_float: bool,
}

impl Default for SqlReadOptions {
    fn default() -> Self {
        Self {
            chunksize: None,
            index_col: None,
            parse_dates: None,
            dtype: None,
            params: None,
            coerce_float: true,
        }
    }
}

/// Advanced SQL writing options
#[derive(Debug, Clone)]
pub struct SqlWriteOptions {
    /// Database schema name
    pub schema: Option<String>,
    /// Whether to write DataFrame index
    pub index: bool,
    /// Label for index column(s)
    pub index_label: Option<String>,
    /// Write in chunks of this size
    pub chunksize: Option<usize>,
    /// Data types for columns
    pub dtype: Option<HashMap<String, String>>,
    /// How to handle existing table
    pub if_exists: WriteMode,
    /// Insertion method
    pub method: InsertMethod,
}

impl Default for SqlWriteOptions {
    fn default() -> Self {
        Self {
            schema: None,
            index: true,
            index_label: None,
            chunksize: Some(10000),
            dtype: None,
            if_exists: WriteMode::Fail,
            method: InsertMethod::Multi,
        }
    }
}

/// How to handle existing tables when writing
#[derive(Debug, Clone, PartialEq)]
pub enum WriteMode {
    /// Raise error if table exists
    Fail,
    /// Drop and recreate table
    Replace,
    /// Append to existing table
    Append,
}

/// Method for inserting data
#[derive(Debug, Clone)]
pub enum InsertMethod {
    /// Single INSERT statement per row
    Single,
    /// Multi-value INSERT statements
    Multi,
    /// Custom insertion logic
    Custom,
}

/// Create a DataFrame from SQL query results
///
/// # Arguments
///
/// * `query` - SQL query to execute
/// * `db_path` - Path to the database file (for SQLite)
///
/// # Returns
///
/// * `Result<DataFrame>` - DataFrame containing query results, or an error
///
/// # Example
///
/// ```no_run
/// use pandrs::io::sql::read_sql;
///
/// let df = read_sql("SELECT name, age FROM users WHERE age > 30", "users.db").unwrap();
/// ```
pub fn read_sql<P: AsRef<Path>>(query: &str, db_path: P) -> Result<DataFrame> {
    // Connect to database
    let conn = SqliteConnection::open(db_path)
        .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

    // Prepare query
    let mut stmt = conn
        .prepare(query)
        .map_err(|e| Error::IoError(format!("Failed to prepare SQL query: {}", e)))?;

    // Get column names
    let column_names: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|&name| name.to_string())
        .collect();

    // Map to store data for each column
    let mut column_data: HashMap<String, Vec<String>> = HashMap::new();
    for name in &column_names {
        column_data.insert(name.clone(), Vec::new());
    }

    // Execute query and get results
    let mut rows = stmt
        .query([])
        .map_err(|e| Error::IoError(format!("Failed to execute SQL query: {}", e)))?;

    // Process each row of data
    while let Some(row) = rows
        .next()
        .map_err(|e| Error::IoError(format!("Failed to retrieve SQL query results: {}", e)))?
    {
        for (idx, name) in column_names.iter().enumerate() {
            let value = get_row_value(row, idx)?;
            if let Some(data) = column_data.get_mut(name) {
                data.push(value);
            }
        }
    }

    // Create DataFrame
    let mut df = DataFrame::new();

    // Create series from column data and add to DataFrame
    for name in column_names {
        if let Some(data) = column_data.get(&name) {
            if let Some(series) = infer_series_from_strings(&name, data)? {
                df.add_column(name.clone(), series)?;
            }
        }
    }

    Ok(df)
}

/// Execute an SQL statement (without returning results)
///
/// # Arguments
///
/// * `sql` - SQL statement to execute
/// * `db_path` - Path to the database file (for SQLite)
///
/// # Returns
///
/// * `Result<usize>` - Number of affected rows, or an error
///
/// # Example
///
/// ```no_run
/// use pandrs::io::sql::execute_sql;
///
/// let affected_rows = execute_sql("UPDATE users SET status = 'active' WHERE last_login > '2023-01-01'", "users.db").unwrap();
/// println!("Affected rows: {}", affected_rows);
/// ```
pub fn execute_sql<P: AsRef<Path>>(sql: &str, db_path: P) -> Result<usize> {
    // Connect to database
    let conn = SqliteConnection::open(db_path)
        .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

    // Execute SQL statement
    let affected_rows = conn
        .execute(sql, [])
        .map_err(|e| Error::IoError(format!("Failed to execute SQL statement: {}", e)))?;

    Ok(affected_rows)
}

/// Write a DataFrame to an SQL database table
///
/// # Arguments
///
/// * `df` - DataFrame to write
/// * `table_name` - Table name
/// * `db_path` - Path to the database file (for SQLite)
/// * `if_exists` - How to handle existing tables ("fail", "replace", "append")
///
/// # Returns
///
/// * `Result<()>` - Ok(()) on success, or an error
///
/// # Example
///
/// ```ignore
/// // DOC test disabled
/// ```
pub fn write_to_sql<P: AsRef<Path>>(
    df: &OptimizedDataFrame,
    table_name: &str,
    db_path: P,
    if_exists: &str,
) -> Result<()> {
    // Connect to database
    let mut conn = SqliteConnection::open(db_path)
        .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

    // Check if table exists
    let table_exists = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
        .map_err(|e| Error::IoError(format!("Failed to prepare table verification query: {}", e)))?
        .exists(&[&table_name])
        .map_err(|e| Error::IoError(format!("Failed to verify table existence: {}", e)))?;

    // Process table based on if_exists
    if table_exists {
        match if_exists {
            "fail" => {
                return Err(Error::IoError(format!(
                    "Table '{}' already exists",
                    table_name
                )));
            }
            "replace" => {
                // Drop the table
                conn.execute(&format!("DROP TABLE IF EXISTS {}", table_name), [])
                    .map_err(|e| Error::IoError(format!("Failed to drop table: {}", e)))?;

                // Create new table
                create_table_from_df(&conn, df, table_name)?;
            }
            "append" => {
                // Table already exists, proceed to append data
            }
            _ => {
                return Err(Error::IoError(format!(
                    "Unknown if_exists value: {}",
                    if_exists
                )));
            }
        }
    } else {
        // Create new table if it doesn't exist
        create_table_from_df(&conn, df, table_name)?;
    }

    // Insert data
    // List of column names
    let column_names = df.column_names();
    let columns = column_names.join(", ");

    // List of placeholders
    let placeholders: Vec<String> = (0..column_names.len()).map(|_| "?".to_string()).collect();
    let placeholders = placeholders.join(", ");

    // Prepare INSERT statement
    let insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        table_name, columns, placeholders
    );

    // Start transaction
    {
        let tx = conn
            .transaction()
            .map_err(|e| Error::IoError(format!("Failed to start transaction: {}", e)))?;

        // Insert each row of data (prepare and execute statement within transaction)
        for row_idx in 0..df.row_count() {
            // Get row data
            let mut row_values: Vec<String> = Vec::new();
            for col_name in column_names.iter() {
                // Get column value as string
                if let Ok(column) = df.column(col_name) {
                    // Extract actual data from the column based on its type
                    let value = match column.column_type() {
                        crate::column::ColumnType::Int64 => {
                            if let Some(int_col) = column.as_int64() {
                                match int_col.get(row_idx) {
                                    Ok(Some(val)) => val.to_string(),
                                    Ok(None) => "NULL".to_string(),
                                    Err(_) => "NULL".to_string(),
                                }
                            } else {
                                "NULL".to_string()
                            }
                        }
                        crate::column::ColumnType::Float64 => {
                            if let Some(float_col) = column.as_float64() {
                                match float_col.get(row_idx) {
                                    Ok(Some(val)) => val.to_string(),
                                    Ok(None) => "NULL".to_string(),
                                    Err(_) => "NULL".to_string(),
                                }
                            } else {
                                "NULL".to_string()
                            }
                        }
                        crate::column::ColumnType::Boolean => {
                            if let Some(bool_col) = column.as_boolean() {
                                match bool_col.get(row_idx) {
                                    Ok(Some(val)) => {
                                        if val {
                                            "1".to_string()
                                        } else {
                                            "0".to_string()
                                        }
                                    }
                                    Ok(None) => "NULL".to_string(),
                                    Err(_) => "NULL".to_string(),
                                }
                            } else {
                                "NULL".to_string()
                            }
                        }
                        crate::column::ColumnType::String => {
                            if let Some(str_col) = column.as_string() {
                                match str_col.get(row_idx) {
                                    Ok(Some(val)) => val.to_string(),
                                    Ok(None) => "NULL".to_string(),
                                    Err(_) => "NULL".to_string(),
                                }
                            } else {
                                "NULL".to_string()
                            }
                        }
                    };
                    row_values.push(value);
                } else {
                    row_values.push("NULL".to_string());
                }
            }

            // Execute INSERT
            let params: Vec<&dyn rusqlite::ToSql> = row_values
                .iter()
                .map(|s| s as &dyn rusqlite::ToSql)
                .collect();

            tx.execute(&insert_sql, params.as_slice())
                .map_err(|e| Error::IoError(format!("Failed to insert data: {}", e)))?;
        }

        // Commit transaction
        tx.commit()
            .map_err(|e| Error::IoError(format!("Failed to commit transaction: {}", e)))?;
    }

    Ok(())
}

/// Read SQL query with advanced options
///
/// # Arguments
///
/// * `sql` - SQL query to execute
/// * `connection` - Database connection
/// * `options` - Advanced reading options
///
/// # Returns
///
/// * `Result<DataFrame>` - DataFrame containing query results, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::sql::{read_sql_advanced, SqlConnection, SqlReadOptions};
///
/// let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
/// let options = SqlReadOptions {
///     chunksize: Some(1000),
///     parse_dates: Some(vec!["created_at".to_string()]),
///     ..Default::default()
/// };
/// let df = read_sql_advanced("SELECT * FROM users", &conn, options).unwrap();
/// ```
pub fn read_sql_advanced(
    sql: &str,
    connection: &SqlConnection,
    options: SqlReadOptions,
) -> Result<DataFrame> {
    match connection.connection_type() {
        DatabaseConnection::Sqlite(path) => read_sql_sqlite_advanced(sql, path, options),
        #[cfg(feature = "sql")]
        _ => {
            // For now, fall back to basic implementation for other databases
            // Full sqlx implementation would go here
            Err(Error::IoError(
                "Advanced SQL features not yet implemented for non-SQLite databases".to_string(),
            ))
        }
    }
}

/// Read SQL table with advanced options
///
/// # Arguments
///
/// * `table_name` - Name of the table to read
/// * `connection` - Database connection
/// * `options` - Advanced reading options
///
/// # Returns
///
/// * `Result<DataFrame>` - DataFrame containing table data, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::sql::{read_sql_table, SqlConnection, SqlReadOptions};
///
/// let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
/// let options = SqlReadOptions {
///     index_col: Some(vec!["id".to_string()]),
///     ..Default::default()
/// };
/// let df = read_sql_table("users", &conn, options).unwrap();
/// ```
pub fn read_sql_table(
    table_name: &str,
    connection: &SqlConnection,
    options: SqlReadOptions,
) -> Result<DataFrame> {
    let sql = format!("SELECT * FROM {}", table_name);
    read_sql_advanced(&sql, connection, options)
}

/// Write DataFrame to SQL with advanced options
///
/// # Arguments
///
/// * `df` - DataFrame to write
/// * `table_name` - Target table name
/// * `connection` - Database connection
/// * `options` - Advanced writing options
///
/// # Returns
///
/// * `Result<usize>` - Number of rows written, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::sql::{write_sql_advanced, SqlConnection, SqlWriteOptions, WriteMode};
/// use pandrs::optimized::dataframe::OptimizedDataFrame;
///
/// // Create sample dataframe
/// let df = OptimizedDataFrame::new();
///
/// let conn = SqlConnection::from_url("sqlite:data.db").unwrap();
/// let options = SqlWriteOptions {
///     if_exists: WriteMode::Replace,
///     chunksize: Some(5000),
///     index: false,
///     ..Default::default()
/// };
/// let rows_written = write_sql_advanced(&df, "users", &conn, options).unwrap();
/// ```
pub fn write_sql_advanced(
    df: &OptimizedDataFrame,
    table_name: &str,
    connection: &SqlConnection,
    options: SqlWriteOptions,
) -> Result<usize> {
    match connection.connection_type() {
        DatabaseConnection::Sqlite(path) => {
            write_sql_sqlite_advanced(df, table_name, path, options)
        }
        #[cfg(feature = "sql")]
        _ => {
            // For now, fall back to basic implementation for other databases
            Err(Error::IoError(
                "Advanced SQL features not yet implemented for non-SQLite databases".to_string(),
            ))
        }
    }
}

/// Internal helper function to create a new table from an OptimizedDataFrame
fn create_table_from_df(
    conn: &SqliteConnection,
    df: &OptimizedDataFrame,
    table_name: &str,
) -> Result<()> {
    // Create list of column names and types
    let mut columns = Vec::new();

    for col_name in df.column_names() {
        // Get each column and determine data type
        if let Ok(column) = df.column(col_name) {
            let sql_type = match column.column_type() {
                crate::column::ColumnType::Int64 => "INTEGER",
                crate::column::ColumnType::Float64 => "REAL",
                crate::column::ColumnType::Boolean => "INTEGER", // SQLite stores booleans as integers
                crate::column::ColumnType::String => "TEXT",
            };
            columns.push(format!("{} {}", col_name, sql_type));
        }
    }

    // Create and execute CREATE TABLE statement
    let create_sql = format!("CREATE TABLE {} ({})", table_name, columns.join(", "));
    conn.execute(&create_sql, [])
        .map_err(|e| Error::IoError(format!("Failed to create table: {}", e)))?;

    Ok(())
}

/// Infer SQL data type from a series
fn series_to_sql_type(series: &Series<String>) -> Result<String> {
    // Infer data type from series name
    let series_type = series
        .name()
        .map_or("unknown", |s| s)
        .split_whitespace()
        .next()
        .unwrap_or("");

    match series_type {
        "i64" | "Int64" => Ok("INTEGER".to_string()),
        "f64" | "Float64" => Ok("REAL".to_string()),
        "bool" | "Boolean" => Ok("INTEGER".to_string()), // In SQLite, boolean values are stored as integers
        _ => Ok("TEXT".to_string()),                     // Default is TEXT
    }
}

/// Get value from SQL row
fn get_row_value(row: &Row, idx: usize) -> Result<String> {
    // Try to get as different types since SQLite is dynamic
    if let Ok(value) = row.get::<_, Option<String>>(idx) {
        Ok(value.unwrap_or_else(|| "NULL".to_string()))
    } else if let Ok(value) = row.get::<_, Option<i64>>(idx) {
        Ok(value
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NULL".to_string()))
    } else if let Ok(value) = row.get::<_, Option<f64>>(idx) {
        Ok(value
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NULL".to_string()))
    } else if let Ok(value) = row.get::<_, Option<bool>>(idx) {
        Ok(value
            .map(|v| v.to_string())
            .unwrap_or_else(|| "NULL".to_string()))
    } else {
        // Fallback: try to get as raw value and convert to string
        match row.get_ref(idx) {
            Ok(value_ref) => {
                use rusqlite::types::ValueRef;
                match value_ref {
                    ValueRef::Null => Ok("NULL".to_string()),
                    ValueRef::Integer(i) => Ok(i.to_string()),
                    ValueRef::Real(r) => Ok(r.to_string()),
                    ValueRef::Text(t) => Ok(String::from_utf8_lossy(t).to_string()),
                    ValueRef::Blob(_) => Ok("BLOB".to_string()),
                }
            }
            Err(e) => Err(Error::IoError(format!(
                "Failed to retrieve row data: {}",
                e
            ))),
        }
    }
}

/// Infer data type from vector of strings and create a series
fn infer_series_from_strings(name: &str, data: &[String]) -> Result<Option<Series<String>>> {
    if data.is_empty() {
        return Ok(None);
    }

    // Check if all values are integers
    let all_integers = data
        .iter()
        .all(|s| s.trim().parse::<i64>().is_ok() || s.trim().is_empty() || s.trim() == "NULL");

    if all_integers {
        let values: Vec<i64> = data
            .iter()
            .map(|s| {
                if s.trim().is_empty() || s.trim() == "NULL" {
                    0
                } else {
                    s.trim().parse::<i64>().unwrap_or(0)
                }
            })
            .collect();
        let series = Series::new(values, Some(name.to_string()))?;
        let string_series = series.to_string_series()?;
        return Ok(Some(string_series));
    }

    // Check if all values are floating point numbers
    let all_floats = data
        .iter()
        .all(|s| s.trim().parse::<f64>().is_ok() || s.trim().is_empty() || s.trim() == "NULL");

    if all_floats {
        let values: Vec<f64> = data
            .iter()
            .map(|s| {
                if s.trim().is_empty() || s.trim() == "NULL" {
                    0.0
                } else {
                    s.trim().parse::<f64>().unwrap_or(0.0)
                }
            })
            .collect();
        let series = Series::new(values, Some(name.to_string()))?;
        let string_series = series.to_string_series()?;
        return Ok(Some(string_series));
    }

    // Check if all values are booleans
    let all_booleans = data.iter().all(|s| {
        let s = s.trim().to_lowercase();
        s == "true" || s == "false" || s == "1" || s == "0" || s.is_empty() || s == "null"
    });

    if all_booleans {
        let values: Vec<bool> = data
            .iter()
            .map(|s| {
                let s = s.trim().to_lowercase();
                s == "true" || s == "1"
            })
            .collect();
        let series = Series::new(values, Some(name.to_string()))?;
        let string_series = series.to_string_series()?;
        return Ok(Some(string_series));
    }

    // Treat everything else as strings
    let values: Vec<String> = data
        .iter()
        .map(|s| {
            if s.trim() == "NULL" {
                "".to_string()
            } else {
                s.clone()
            }
        })
        .collect();

    Ok(Some(Series::new(values, Some(name.to_string()))?))
}

/// Internal implementation for SQLite advanced reading
fn read_sql_sqlite_advanced(
    sql: &str,
    db_path: &str,
    options: SqlReadOptions,
) -> Result<DataFrame> {
    let conn = SqliteConnection::open(db_path)
        .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

    // Prepare statement
    let mut stmt = conn
        .prepare(sql)
        .map_err(|e| Error::IoError(format!("Failed to prepare SQL query: {}", e)))?;

    // Get column names
    let column_names: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|&name| name.to_string())
        .collect();

    // Execute query with parameters if provided
    let param_values: Vec<&dyn rusqlite::ToSql> = if let Some(ref params) = options.params {
        params.iter().map(|p| p as &dyn rusqlite::ToSql).collect()
    } else {
        Vec::new()
    };

    let mut rows = stmt
        .query(param_values.as_slice())
        .map_err(|e| Error::IoError(format!("Failed to execute SQL query: {}", e)))?;

    // Collect data
    let mut column_data: HashMap<String, Vec<String>> = HashMap::new();
    for name in &column_names {
        column_data.insert(name.clone(), Vec::new());
    }

    let mut row_count = 0;
    while let Some(row) = rows
        .next()
        .map_err(|e| Error::IoError(format!("Failed to retrieve SQL query results: {}", e)))?
    {
        // Check chunk size limit
        if let Some(chunksize) = options.chunksize {
            if row_count >= chunksize {
                break;
            }
        }

        for (idx, name) in column_names.iter().enumerate() {
            let value = get_row_value(row, idx)?;
            if let Some(data) = column_data.get_mut(name) {
                data.push(value);
            }
        }
        row_count += 1;
    }

    // Create DataFrame with enhanced type inference
    let mut df = DataFrame::new();

    for name in column_names {
        if let Some(data) = column_data.get(&name) {
            // Check if this column should be parsed as date
            let is_date_column = options
                .parse_dates
                .as_ref()
                .map(|dates| dates.contains(&name))
                .unwrap_or(false);

            // Check for explicit dtype
            let explicit_dtype = options.dtype.as_ref().and_then(|dtypes| dtypes.get(&name));

            if let Some(series) = infer_series_from_strings_advanced(
                &name,
                data,
                is_date_column,
                explicit_dtype,
                options.coerce_float,
            )? {
                df.add_column(name.clone(), series)?;
            }
        }
    }

    Ok(df)
}

/// Internal implementation for SQLite advanced writing
fn write_sql_sqlite_advanced(
    df: &OptimizedDataFrame,
    table_name: &str,
    db_path: &str,
    options: SqlWriteOptions,
) -> Result<usize> {
    let mut conn = SqliteConnection::open(db_path)
        .map_err(|e| Error::IoError(format!("Failed to connect to database: {}", e)))?;

    // Check if table exists
    let table_exists = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
        .map_err(|e| Error::IoError(format!("Failed to prepare table verification query: {}", e)))?
        .exists(&[&table_name])
        .map_err(|e| Error::IoError(format!("Failed to verify table existence: {}", e)))?;

    // Handle existing table based on if_exists option
    if table_exists {
        match options.if_exists {
            WriteMode::Fail => {
                return Err(Error::IoError(format!(
                    "Table '{}' already exists",
                    table_name
                )));
            }
            WriteMode::Replace => {
                conn.execute(&format!("DROP TABLE IF EXISTS {}", table_name), [])
                    .map_err(|e| Error::IoError(format!("Failed to drop table: {}", e)))?;
                create_table_from_df_advanced(&conn, df, table_name, &options)?;
            }
            WriteMode::Append => {
                // Table exists, proceed to append
            }
        }
    } else {
        create_table_from_df_advanced(&conn, df, table_name, &options)?;
    }

    // Insert data with chunking
    let chunk_size = options.chunksize.unwrap_or(df.row_count());
    let mut total_inserted = 0;

    for chunk_start in (0..df.row_count()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(df.row_count());
        let rows_inserted =
            insert_data_chunk(&mut conn, df, table_name, chunk_start, chunk_end, &options)?;
        total_inserted += rows_inserted;
    }

    Ok(total_inserted)
}

/// Create table with advanced options
fn create_table_from_df_advanced(
    conn: &SqliteConnection,
    df: &OptimizedDataFrame,
    table_name: &str,
    options: &SqlWriteOptions,
) -> Result<()> {
    let mut columns = Vec::new();

    // Add index column if requested
    if options.index {
        let index_name = options
            .index_label
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("index");
        columns.push(format!("{} INTEGER", index_name));
    }

    // Add data columns
    for col_name in df.column_names() {
        if let Ok(column) = df.column(col_name) {
            let sql_type = if let Some(ref dtype_map) = options.dtype {
                dtype_map
                    .get(col_name)
                    .map(|s| s.as_str())
                    .unwrap_or_else(|| match column.column_type() {
                        crate::column::ColumnType::Int64 => "INTEGER",
                        crate::column::ColumnType::Float64 => "REAL",
                        crate::column::ColumnType::Boolean => "INTEGER",
                        crate::column::ColumnType::String => "TEXT",
                    })
            } else {
                match column.column_type() {
                    crate::column::ColumnType::Int64 => "INTEGER",
                    crate::column::ColumnType::Float64 => "REAL",
                    crate::column::ColumnType::Boolean => "INTEGER",
                    crate::column::ColumnType::String => "TEXT",
                }
            };
            columns.push(format!("{} {}", col_name, sql_type));
        }
    }

    let create_sql = format!("CREATE TABLE {} ({})", table_name, columns.join(", "));
    conn.execute(&create_sql, [])
        .map_err(|e| Error::IoError(format!("Failed to create table: {}", e)))?;

    Ok(())
}

/// Insert data chunk with different insertion methods
fn insert_data_chunk(
    conn: &mut SqliteConnection,
    df: &OptimizedDataFrame,
    table_name: &str,
    start_row: usize,
    end_row: usize,
    options: &SqlWriteOptions,
) -> Result<usize> {
    let column_names = df.column_names();
    let mut all_columns = Vec::new();

    // Add index column if requested
    if options.index {
        let index_name = options
            .index_label
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("index");
        all_columns.push(index_name.to_string());
    }
    all_columns.extend(column_names.iter().cloned());

    match options.method {
        InsertMethod::Single => insert_single_rows(
            conn,
            df,
            table_name,
            start_row,
            end_row,
            &all_columns,
            options,
        ),
        InsertMethod::Multi => insert_multi_rows(
            conn,
            df,
            table_name,
            start_row,
            end_row,
            &all_columns,
            options,
        ),
        InsertMethod::Custom => {
            // For now, fall back to multi-row insertion
            insert_multi_rows(
                conn,
                df,
                table_name,
                start_row,
                end_row,
                &all_columns,
                options,
            )
        }
    }
}

/// Insert rows one by one
fn insert_single_rows(
    conn: &mut SqliteConnection,
    df: &OptimizedDataFrame,
    table_name: &str,
    start_row: usize,
    end_row: usize,
    all_columns: &[String],
    options: &SqlWriteOptions,
) -> Result<usize> {
    let columns = all_columns.join(", ");
    let placeholders: Vec<String> = (0..all_columns.len()).map(|_| "?".to_string()).collect();
    let placeholders = placeholders.join(", ");
    let insert_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        table_name, columns, placeholders
    );

    let tx = conn
        .transaction()
        .map_err(|e| Error::IoError(format!("Failed to start transaction: {}", e)))?;

    let mut inserted = 0;
    for row_idx in start_row..end_row {
        let mut row_values: Vec<String> = Vec::new();

        // Add index value if requested
        if options.index {
            row_values.push(row_idx.to_string());
        }

        // Add data values
        for col_name in df.column_names().iter() {
            if let Ok(column) = df.column(col_name) {
                let value = extract_column_value(column, row_idx)?;
                row_values.push(value);
            } else {
                row_values.push("NULL".to_string());
            }
        }

        let params: Vec<&dyn rusqlite::ToSql> = row_values
            .iter()
            .map(|s| s as &dyn rusqlite::ToSql)
            .collect();

        tx.execute(&insert_sql, params.as_slice())
            .map_err(|e| Error::IoError(format!("Failed to insert data: {}", e)))?;

        inserted += 1;
    }

    tx.commit()
        .map_err(|e| Error::IoError(format!("Failed to commit transaction: {}", e)))?;

    Ok(inserted)
}

/// Insert multiple rows in single statement
fn insert_multi_rows(
    conn: &mut SqliteConnection,
    df: &OptimizedDataFrame,
    table_name: &str,
    start_row: usize,
    end_row: usize,
    all_columns: &[String],
    options: &SqlWriteOptions,
) -> Result<usize> {
    let columns = all_columns.join(", ");

    // Build values for all rows
    let mut all_values = Vec::new();
    let mut all_params = Vec::new();

    for row_idx in start_row..end_row {
        let mut row_values = Vec::new();

        // Add index value if requested
        if options.index {
            row_values.push(row_idx.to_string());
        }

        // Add data values
        for col_name in df.column_names().iter() {
            if let Ok(column) = df.column(col_name) {
                let value = extract_column_value(column, row_idx)?;
                row_values.push(value);
            } else {
                row_values.push("NULL".to_string());
            }
        }

        let placeholders: Vec<String> = (0..row_values.len()).map(|_| "?".to_string()).collect();
        all_values.push(format!("({})", placeholders.join(", ")));

        for value in row_values {
            all_params.push(value);
        }
    }

    let insert_sql = format!(
        "INSERT INTO {} ({}) VALUES {}",
        table_name,
        columns,
        all_values.join(", ")
    );

    let params: Vec<&dyn rusqlite::ToSql> = all_params
        .iter()
        .map(|s| s as &dyn rusqlite::ToSql)
        .collect();

    let inserted = conn
        .execute(&insert_sql, params.as_slice())
        .map_err(|e| Error::IoError(format!("Failed to insert data: {}", e)))?;

    Ok(inserted)
}

/// Extract value from column at specific row index
fn extract_column_value(column: crate::optimized::ColumnView, row_idx: usize) -> Result<String> {
    match column.column_type() {
        crate::column::ColumnType::Int64 => {
            if let Some(int_col) = column.as_int64() {
                match int_col.get(row_idx) {
                    Ok(Some(val)) => Ok(val.to_string()),
                    Ok(None) => Ok("NULL".to_string()),
                    Err(_) => Ok("NULL".to_string()),
                }
            } else {
                Ok("NULL".to_string())
            }
        }
        crate::column::ColumnType::Float64 => {
            if let Some(float_col) = column.as_float64() {
                match float_col.get(row_idx) {
                    Ok(Some(val)) => Ok(val.to_string()),
                    Ok(None) => Ok("NULL".to_string()),
                    Err(_) => Ok("NULL".to_string()),
                }
            } else {
                Ok("NULL".to_string())
            }
        }
        crate::column::ColumnType::Boolean => {
            if let Some(bool_col) = column.as_boolean() {
                match bool_col.get(row_idx) {
                    Ok(Some(val)) => Ok(if val {
                        "1".to_string()
                    } else {
                        "0".to_string()
                    }),
                    Ok(None) => Ok("NULL".to_string()),
                    Err(_) => Ok("NULL".to_string()),
                }
            } else {
                Ok("NULL".to_string())
            }
        }
        crate::column::ColumnType::String => {
            if let Some(str_col) = column.as_string() {
                match str_col.get(row_idx) {
                    Ok(Some(val)) => Ok(val.to_string()),
                    Ok(None) => Ok("NULL".to_string()),
                    Err(_) => Ok("NULL".to_string()),
                }
            } else {
                Ok("NULL".to_string())
            }
        }
    }
}

/// Enhanced type inference with date parsing and explicit types
fn infer_series_from_strings_advanced(
    name: &str,
    data: &[String],
    is_date_column: bool,
    explicit_dtype: Option<&String>,
    coerce_float: bool,
) -> Result<Option<Series<String>>> {
    if data.is_empty() {
        return Ok(None);
    }

    // Handle explicit data type specification
    if let Some(dtype) = explicit_dtype {
        return match dtype.to_lowercase().as_str() {
            "int64" | "integer" => {
                let values: Vec<i64> = data.iter().map(|s| s.parse().unwrap_or(0)).collect();
                let series = Series::new(values, Some(name.to_string()))?;
                Ok(Some(series.to_string_series()?))
            }
            "float64" | "float" => {
                let values: Vec<f64> = data.iter().map(|s| s.parse().unwrap_or(0.0)).collect();
                let series = Series::new(values, Some(name.to_string()))?;
                Ok(Some(series.to_string_series()?))
            }
            "bool" | "boolean" => {
                let values: Vec<bool> = data
                    .iter()
                    .map(|s| {
                        let s = s.trim().to_lowercase();
                        s == "true" || s == "1"
                    })
                    .collect();
                let series = Series::new(values, Some(name.to_string()))?;
                Ok(Some(series.to_string_series()?))
            }
            _ => {
                // Default to string
                Ok(Some(Series::new(data.to_vec(), Some(name.to_string()))?))
            }
        };
    }

    // Handle date parsing
    if is_date_column {
        // For now, keep as strings but could parse to date types in future
        return Ok(Some(Series::new(data.to_vec(), Some(name.to_string()))?));
    }

    // Automatic type inference (existing logic enhanced)
    infer_series_from_strings(name, data)
}
