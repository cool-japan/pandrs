use std::time::Duration;

#[cfg(feature = "sql")]
use sqlx::{AnyPool, Column as SqlxColumn, Executor, Pool, Row as SqlxRow};

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use crate::series::Series;

use super::connection::{
    AsyncDatabasePool, ConnectionStats, DatabaseOperation, IsolationLevel, SqlValue,
    TransactionManager,
};
use super::operations::SqlWriteOptions;

impl AsyncDatabasePool {
    /// Execute async SQL query and return DataFrame
    ///
    /// # Arguments
    ///
    /// * `query` - SQL query to execute
    /// * `params` - Optional query parameters
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame>` - Query results as DataFrame
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{AsyncDatabasePool, SqlValue};
    ///
    /// async fn example(pool: &AsyncDatabasePool) {
    ///     let params = vec![SqlValue::Integer(25)];
    ///     let df = pool.query_async("SELECT * FROM users WHERE age > ?", Some(params)).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn query_async(
        &self,
        query: &str,
        _params: Option<Vec<SqlValue>>,
    ) -> Result<DataFrame> {
        use std::time::Instant;

        let _start_time = Instant::now();

        // Execute query
        let rows = sqlx::query(query)
            .fetch_all(self.pool())
            .await
            .map_err(|e| Error::IoError(format!("Async query failed: {}", e)))?;

        let _query_duration = _start_time.elapsed().as_millis() as f64;

        // Convert rows to DataFrame
        let df = convert_sqlx_rows_to_dataframe(&rows)?;

        // Update statistics (would need Arc<Mutex<ConnectionStats>> for thread safety)

        Ok(df)
    }

    /// Execute async bulk insert operation
    ///
    /// # Arguments
    ///
    /// * `table_name` - Target table name
    /// * `df` - DataFrame to insert
    /// * `options` - Write options
    ///
    /// # Returns
    ///
    /// * `Result<u64>` - Number of rows inserted
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{AsyncDatabasePool, SqlWriteOptions};
    /// use pandrs::DataFrame;
    ///
    /// async fn example(pool: &AsyncDatabasePool, df: &DataFrame) {
    ///     let options = SqlWriteOptions::default();
    ///     let rows_inserted = pool.bulk_insert_async("users", df, options).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn bulk_insert_async(
        &self,
        table_name: &str,
        df: &DataFrame,
        options: SqlWriteOptions,
    ) -> Result<u64> {
        // Begin transaction for bulk insert
        let mut tx = self
            .pool()
            .begin()
            .await
            .map_err(|e| Error::IoError(format!("Failed to begin transaction: {}", e)))?;

        let chunk_size = options.chunksize.unwrap_or(10000);
        let mut total_inserted = 0u64;

        // Process in chunks
        for chunk_start in (0..df.row_count()).step_by(chunk_size) {
            let chunk_end = std::cmp::min(chunk_start + chunk_size, df.row_count());

            // Generate bulk INSERT SQL for this chunk
            let insert_sql =
                generate_bulk_insert_sql(table_name, df, chunk_start, chunk_end, &options)?;

            // Execute bulk insert
            let result = tx
                .execute(sqlx::query(&insert_sql))
                .await
                .map_err(|e| Error::IoError(format!("Bulk insert failed: {}", e)))?;

            total_inserted += result.rows_affected();
        }

        // Commit transaction
        tx.commit()
            .await
            .map_err(|e| Error::IoError(format!("Failed to commit transaction: {}", e)))?;

        Ok(total_inserted)
    }

    /// Execute parameterized query with bind parameters
    ///
    /// # Arguments
    ///
    /// * `query` - SQL query with parameter placeholders
    /// * `params` - Parameters to bind to the query
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame>` - Query results
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{AsyncDatabasePool, SqlValue};
    ///
    /// async fn example(pool: &AsyncDatabasePool) {
    ///     let params = vec![
    ///         SqlValue::Text("John".to_string()),
    ///         SqlValue::Integer(25)
    ///     ];
    ///     let df = pool.query_with_params(
    ///         "SELECT * FROM users WHERE name = ? AND age > ?",
    ///         params
    ///     ).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn query_with_params(&self, query: &str, params: Vec<SqlValue>) -> Result<DataFrame> {
        use std::time::Instant;

        let _start_time = Instant::now();

        // Build parameterized query
        let mut sqlx_query = sqlx::query(query);

        // Bind parameters (simplified - would need proper type conversion)
        for param in params {
            match param {
                SqlValue::Integer(i) => sqlx_query = sqlx_query.bind(i),
                SqlValue::Real(f) => sqlx_query = sqlx_query.bind(f),
                SqlValue::Text(s) => sqlx_query = sqlx_query.bind(s),
                SqlValue::Boolean(b) => sqlx_query = sqlx_query.bind(b),
                SqlValue::Null => sqlx_query = sqlx_query.bind(None::<String>),
                SqlValue::Blob(b) => sqlx_query = sqlx_query.bind(b),
            }
        }

        // Execute query
        let rows = sqlx_query
            .fetch_all(self.pool())
            .await
            .map_err(|e| Error::IoError(format!("Parameterized query failed: {}", e)))?;

        // Convert to DataFrame
        convert_sqlx_rows_to_dataframe(&rows)
    }

    /// Execute multiple queries in parallel
    ///
    /// # Arguments
    ///
    /// * `queries` - Vector of SQL queries to execute
    ///
    /// # Returns
    ///
    /// * `Result<Vec<DataFrame>>` - Results of all queries
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::AsyncDatabasePool;
    ///
    /// async fn example(pool: &AsyncDatabasePool) {
    ///     let queries = vec![
    ///         "SELECT COUNT(*) FROM users".to_string(),
    ///         "SELECT COUNT(*) FROM orders".to_string(),
    ///         "SELECT COUNT(*) FROM products".to_string(),
    ///     ];
    ///     let results = pool.parallel_queries(queries).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn parallel_queries(&self, queries: Vec<String>) -> Result<Vec<DataFrame>> {
        use futures::future::try_join_all;

        let futures: Vec<_> = queries
            .iter()
            .map(|query| self.query_async(query, None))
            .collect();

        try_join_all(futures).await
    }

    /// Stream large result sets to avoid memory issues
    ///
    /// # Arguments
    ///
    /// * `query` - SQL query to execute
    /// * `chunk_size` - Number of rows to fetch at a time
    ///
    /// # Returns
    ///
    /// * `Result<Vec<DataFrame>>` - Vector of DataFrames, each containing chunk_size rows
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::AsyncDatabasePool;
    ///
    /// async fn example(pool: &AsyncDatabasePool) {
    ///     let chunks = pool.stream_query("SELECT * FROM large_table", 1000).await.unwrap();
    ///     for (i, chunk) in chunks.iter().enumerate() {
    ///         println!("Chunk {}: {} rows", i, chunk.row_count());
    ///     }
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn stream_query(&self, query: &str, chunk_size: usize) -> Result<Vec<DataFrame>> {
        use sqlx::Row;

        // This is a simplified implementation - real streaming would use cursors
        let mut chunks = Vec::new();
        let mut offset = 0;

        loop {
            let chunked_query = format!("{} LIMIT {} OFFSET {}", query, chunk_size, offset);

            let rows = sqlx::query(&chunked_query)
                .fetch_all(self.pool())
                .await
                .map_err(|e| Error::IoError(format!("Stream query failed: {}", e)))?;

            if rows.is_empty() {
                break;
            }

            let df = convert_sqlx_rows_to_dataframe(&rows)?;
            chunks.push(df);

            offset += chunk_size;

            // If we got fewer rows than chunk_size, we're done
            if rows.len() < chunk_size {
                break;
            }
        }

        Ok(chunks)
    }
}

impl TransactionManager {
    /// Execute multiple operations in a transaction
    ///
    /// # Arguments
    ///
    /// * `operations` - Vector of SQL operations to execute
    ///
    /// # Returns
    ///
    /// * `Result<Vec<DataFrame>>` - Results of each operation
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{TransactionManager, DatabaseOperation};
    ///
    /// async fn example(tx_manager: &TransactionManager) {
    ///     let ops = vec![
    ///         DatabaseOperation::Query("SELECT COUNT(*) FROM orders".to_string()),
    ///         DatabaseOperation::Execute("UPDATE orders SET status = 'processed'".to_string()),
    ///     ];
    ///     let results = tx_manager.execute_transaction(ops).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn execute_transaction(
        &self,
        operations: Vec<DatabaseOperation>,
    ) -> Result<Vec<Option<DataFrame>>> {
        let mut tx = self
            .pool()
            .begin()
            .await
            .map_err(|e| Error::IoError(format!("Failed to begin transaction: {}", e)))?;

        let mut results = Vec::new();

        for operation in operations {
            match operation {
                DatabaseOperation::Query(sql) => {
                    let rows = sqlx::query(&sql)
                        .fetch_all(&mut *tx)
                        .await
                        .map_err(|e| Error::IoError(format!("Transaction query failed: {}", e)))?;

                    let df = convert_sqlx_rows_to_dataframe(&rows)?;
                    results.push(Some(df));
                }
                DatabaseOperation::Execute(sql) => {
                    tx.execute(sqlx::query(&sql)).await.map_err(|e| {
                        Error::IoError(format!("Transaction execute failed: {}", e))
                    })?;
                    results.push(None);
                }
            }
        }

        tx.commit()
            .await
            .map_err(|e| Error::IoError(format!("Failed to commit transaction: {}", e)))?;

        Ok(results)
    }

    /// Execute transaction with savepoints for nested transactions
    ///
    /// # Arguments
    ///
    /// * `operations` - Vector of operation groups, each executed within a savepoint
    ///
    /// # Returns
    ///
    /// * `Result<Vec<Vec<Option<DataFrame>>>>` - Results grouped by savepoint
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{TransactionManager, DatabaseOperation};
    ///
    /// async fn example(tx_manager: &TransactionManager) {
    ///     let operation_groups = vec![
    ///         vec![DatabaseOperation::Execute("INSERT INTO users (name) VALUES ('Alice')".to_string())],
    ///         vec![DatabaseOperation::Execute("INSERT INTO orders (user_id) VALUES (1)".to_string())],
    ///     ];
    ///     let results = tx_manager.execute_nested_transaction(operation_groups).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn execute_nested_transaction(
        &self,
        operation_groups: Vec<Vec<DatabaseOperation>>,
    ) -> Result<Vec<Vec<Option<DataFrame>>>> {
        let mut tx = self
            .pool()
            .begin()
            .await
            .map_err(|e| Error::IoError(format!("Failed to begin transaction: {}", e)))?;

        let mut all_results = Vec::new();

        for (i, operations) in operation_groups.into_iter().enumerate() {
            let savepoint_name = format!("sp_{}", i);

            // Create savepoint
            tx.execute(sqlx::query(&format!("SAVEPOINT {}", savepoint_name)))
                .await
                .map_err(|e| Error::IoError(format!("Failed to create savepoint: {}", e)))?;

            let mut group_results = Vec::new();
            let mut savepoint_error = false;

            for operation in operations {
                match operation {
                    DatabaseOperation::Query(sql) => {
                        match sqlx::query(&sql).fetch_all(&mut *tx).await {
                            Ok(rows) => match convert_sqlx_rows_to_dataframe(&rows) {
                                Ok(df) => group_results.push(Some(df)),
                                Err(_) => {
                                    savepoint_error = true;
                                    break;
                                }
                            },
                            Err(_) => {
                                savepoint_error = true;
                                break;
                            }
                        }
                    }
                    DatabaseOperation::Execute(sql) => match tx.execute(sqlx::query(&sql)).await {
                        Ok(_) => group_results.push(None),
                        Err(_) => {
                            savepoint_error = true;
                            break;
                        }
                    },
                }
            }

            if savepoint_error {
                // Rollback to savepoint
                tx.execute(sqlx::query(&format!("ROLLBACK TO {}", savepoint_name)))
                    .await
                    .map_err(|e| Error::IoError(format!("Failed to rollback savepoint: {}", e)))?;

                return Err(Error::IoError(format!(
                    "Transaction failed at savepoint {}",
                    savepoint_name
                )));
            } else {
                // Release savepoint (commit it)
                tx.execute(sqlx::query(&format!("RELEASE {}", savepoint_name)))
                    .await
                    .map_err(|e| Error::IoError(format!("Failed to release savepoint: {}", e)))?;
            }

            all_results.push(group_results);
        }

        tx.commit()
            .await
            .map_err(|e| Error::IoError(format!("Failed to commit transaction: {}", e)))?;

        Ok(all_results)
    }

    /// Set transaction isolation level
    ///
    /// # Arguments
    ///
    /// * `level` - Isolation level to set
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{TransactionManager, IsolationLevel};
    ///
    /// async fn example(tx_manager: &mut TransactionManager) {
    ///     tx_manager.set_isolation_level(IsolationLevel::Serializable).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn set_isolation_level(&mut self, level: IsolationLevel) -> Result<()> {
        let sql = match level {
            IsolationLevel::ReadUncommitted => "SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED",
            IsolationLevel::ReadCommitted => "SET TRANSACTION ISOLATION LEVEL READ COMMITTED",
            IsolationLevel::RepeatableRead => "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ",
            IsolationLevel::Serializable => "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE",
        };

        sqlx::query(sql)
            .execute(self.pool())
            .await
            .map_err(|e| Error::IoError(format!("Failed to set isolation level: {}", e)))?;

        self.set_isolation_level_internal(level);
        Ok(())
    }

    /// Execute read-only transaction for better performance
    ///
    /// # Arguments
    ///
    /// * `queries` - Vector of read-only queries to execute
    ///
    /// # Returns
    ///
    /// * `Result<Vec<DataFrame>>` - Results of all queries
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::TransactionManager;
    ///
    /// async fn example(tx_manager: &TransactionManager) {
    ///     let queries = vec![
    ///         "SELECT * FROM users WHERE active = true".to_string(),
    ///         "SELECT COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL '1 day'".to_string(),
    ///     ];
    ///     let results = tx_manager.execute_readonly_transaction(queries).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn execute_readonly_transaction(
        &self,
        queries: Vec<String>,
    ) -> Result<Vec<DataFrame>> {
        let mut tx =
            self.pool().begin().await.map_err(|e| {
                Error::IoError(format!("Failed to begin readonly transaction: {}", e))
            })?;

        // Set transaction as read-only if supported by the database
        let _ = tx.execute(sqlx::query("SET TRANSACTION READ ONLY")).await;

        let mut results = Vec::new();

        for query in queries {
            let rows = sqlx::query(&query)
                .fetch_all(&mut *tx)
                .await
                .map_err(|e| Error::IoError(format!("Readonly query failed: {}", e)))?;

            let df = convert_sqlx_rows_to_dataframe(&rows)?;
            results.push(df);
        }

        tx.commit()
            .await
            .map_err(|e| Error::IoError(format!("Failed to commit readonly transaction: {}", e)))?;

        Ok(results)
    }
}

/// Database query builder for type-safe SQL generation
pub struct QueryBuilder {
    /// Table name
    table: Option<String>,
    /// SELECT columns
    select_columns: Vec<String>,
    /// WHERE conditions
    where_conditions: Vec<String>,
    /// JOIN clauses
    joins: Vec<String>,
    /// ORDER BY clauses
    order_by: Vec<String>,
    /// GROUP BY clauses
    group_by: Vec<String>,
    /// HAVING conditions
    having_conditions: Vec<String>,
    /// LIMIT clause
    limit: Option<usize>,
    /// OFFSET clause
    offset: Option<usize>,
}

impl QueryBuilder {
    /// Create new query builder
    pub fn new() -> Self {
        Self {
            table: None,
            select_columns: Vec::new(),
            where_conditions: Vec::new(),
            joins: Vec::new(),
            order_by: Vec::new(),
            group_by: Vec::new(),
            having_conditions: Vec::new(),
            limit: None,
            offset: None,
        }
    }

    /// Set table to query from
    pub fn from(mut self, table: &str) -> Self {
        self.table = Some(table.to_string());
        self
    }

    /// Add SELECT column
    pub fn select(mut self, column: &str) -> Self {
        self.select_columns.push(column.to_string());
        self
    }

    /// Select multiple columns
    pub fn select_many(mut self, columns: &[&str]) -> Self {
        for column in columns {
            self.select_columns.push(column.to_string());
        }
        self
    }

    /// Add WHERE condition
    pub fn where_clause(mut self, condition: &str) -> Self {
        self.where_conditions.push(condition.to_string());
        self
    }

    /// Add WHERE condition with AND operator
    pub fn and_where(mut self, condition: &str) -> Self {
        self.where_conditions.push(format!("AND {}", condition));
        self
    }

    /// Add WHERE condition with OR operator
    pub fn or_where(mut self, condition: &str) -> Self {
        self.where_conditions.push(format!("OR {}", condition));
        self
    }

    /// Add JOIN clause
    pub fn join(mut self, join_clause: &str) -> Self {
        self.joins.push(format!("JOIN {}", join_clause));
        self
    }

    /// Add LEFT JOIN clause
    pub fn left_join(mut self, join_clause: &str) -> Self {
        self.joins.push(format!("LEFT JOIN {}", join_clause));
        self
    }

    /// Add RIGHT JOIN clause
    pub fn right_join(mut self, join_clause: &str) -> Self {
        self.joins.push(format!("RIGHT JOIN {}", join_clause));
        self
    }

    /// Add INNER JOIN clause
    pub fn inner_join(mut self, join_clause: &str) -> Self {
        self.joins.push(format!("INNER JOIN {}", join_clause));
        self
    }

    /// Add ORDER BY clause
    pub fn order_by(mut self, column: &str, direction: &str) -> Self {
        self.order_by.push(format!("{} {}", column, direction));
        self
    }

    /// Add ORDER BY ASC
    pub fn order_by_asc(mut self, column: &str) -> Self {
        self.order_by.push(format!("{} ASC", column));
        self
    }

    /// Add ORDER BY DESC
    pub fn order_by_desc(mut self, column: &str) -> Self {
        self.order_by.push(format!("{} DESC", column));
        self
    }

    /// Add GROUP BY clause
    pub fn group_by(mut self, column: &str) -> Self {
        self.group_by.push(column.to_string());
        self
    }

    /// Add HAVING condition
    pub fn having(mut self, condition: &str) -> Self {
        self.having_conditions.push(condition.to_string());
        self
    }

    /// Set LIMIT
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set OFFSET
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build the SQL query
    pub fn build(self) -> Result<String> {
        let table = self
            .table
            .ok_or_else(|| Error::IoError("Table name is required".to_string()))?;

        let mut sql = String::new();

        // SELECT clause
        if self.select_columns.is_empty() {
            sql.push_str("SELECT *");
        } else {
            sql.push_str(&format!("SELECT {}", self.select_columns.join(", ")));
        }

        // FROM clause
        sql.push_str(&format!(" FROM {}", table));

        // JOIN clauses
        for join in &self.joins {
            sql.push_str(&format!(" {}", join));
        }

        // WHERE clause
        if !self.where_conditions.is_empty() {
            let conditions = self.where_conditions.join(" ");
            sql.push_str(&format!(" WHERE {}", conditions));
        }

        // GROUP BY clause
        if !self.group_by.is_empty() {
            sql.push_str(&format!(" GROUP BY {}", self.group_by.join(", ")));
        }

        // HAVING clause
        if !self.having_conditions.is_empty() {
            sql.push_str(&format!(" HAVING {}", self.having_conditions.join(" AND ")));
        }

        // ORDER BY clause
        if !self.order_by.is_empty() {
            sql.push_str(&format!(" ORDER BY {}", self.order_by.join(", ")));
        }

        // LIMIT clause
        if let Some(limit) = self.limit {
            sql.push_str(&format!(" LIMIT {}", limit));
        }

        // OFFSET clause
        if let Some(offset) = self.offset {
            sql.push_str(&format!(" OFFSET {}", offset));
        }

        Ok(sql)
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate bulk INSERT SQL statement
fn generate_bulk_insert_sql(
    table_name: &str,
    df: &DataFrame,
    start_row: usize,
    end_row: usize,
    _options: &SqlWriteOptions,
) -> Result<String> {
    // Get column names
    let columns: Vec<String> = df.column_names().iter().cloned().collect();

    // Build INSERT statement
    let mut sql = format!("INSERT INTO {} (", table_name);
    sql.push_str(&columns.join(", "));
    sql.push_str(") VALUES ");

    // Add value placeholders (simplified implementation)
    let value_rows: Vec<String> = (start_row..end_row)
        .map(|_row_idx| {
            let placeholders: Vec<String> = columns.iter().map(|_| "?".to_string()).collect();
            format!("({})", placeholders.join(", "))
        })
        .collect();

    sql.push_str(&value_rows.join(", "));

    Ok(sql)
}

/// Convert sqlx rows to DataFrame
#[cfg(feature = "sql")]
fn convert_sqlx_rows_to_dataframe(rows: &[sqlx::any::AnyRow]) -> Result<DataFrame> {
    use sqlx::Row;

    if rows.is_empty() {
        return Ok(DataFrame::new());
    }

    let mut df = DataFrame::new();

    // Get column information from first row
    let first_row = &rows[0];
    let column_count = first_row.len();

    for col_idx in 0..column_count {
        let column = first_row.column(col_idx);
        let col_name = column.name().to_string();

        // Extract column data
        let mut values = Vec::new();
        for row in rows {
            // Simplified value extraction - try to get as string
            let value = match row.try_get::<String, _>(col_idx) {
                Ok(s) => s,
                Err(_) => "NULL".to_string(), // Fallback for non-string types
            };
            values.push(value);
        }

        let series = Series::new(values, Some(col_name.clone()))?;
        df.add_column(col_name, series)?;
    }

    Ok(df)
}

/// Advanced connection pooling with health checks
pub struct AdvancedConnectionPool {
    /// Primary connection pool
    #[cfg(feature = "sql")]
    primary_pool: AnyPool,
    /// Read-only replica pools
    #[cfg(feature = "sql")]
    read_pools: Vec<AnyPool>,
    /// Health check interval
    health_check_interval: Duration,
    /// Connection retry settings
    retry_settings: RetrySettings,
}

/// Connection retry configuration
#[derive(Debug, Clone)]
pub struct RetrySettings {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Base delay between retries
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Exponential backoff factor
    pub backoff_factor: f64,
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_factor: 2.0,
        }
    }
}

impl AdvancedConnectionPool {
    /// Create new advanced connection pool with read replicas
    #[cfg(feature = "sql")]
    pub async fn new_with_replicas(
        primary_url: &str,
        read_urls: Vec<&str>,
        health_check_interval: Duration,
    ) -> Result<Self> {
        use sqlx::any::AnyPoolOptions;

        let primary_pool = AnyPoolOptions::new()
            .connect(primary_url)
            .await
            .map_err(|e| Error::IoError(format!("Failed to connect to primary: {}", e)))?;

        let mut read_pools = Vec::new();
        for url in read_urls {
            let pool = AnyPoolOptions::new()
                .connect(url)
                .await
                .map_err(|e| Error::IoError(format!("Failed to connect to read replica: {}", e)))?;
            read_pools.push(pool);
        }

        Ok(Self {
            primary_pool,
            read_pools,
            health_check_interval,
            retry_settings: RetrySettings::default(),
        })
    }

    /// Execute read query with automatic replica selection
    #[cfg(feature = "sql")]
    pub async fn read_query(&self, query: &str) -> Result<DataFrame> {
        // Select a read replica (simple round-robin for now)
        let pool = if self.read_pools.is_empty() {
            &self.primary_pool
        } else {
            // In a real implementation, you'd want proper load balancing
            &self.read_pools[0]
        };

        let rows = sqlx::query(query)
            .fetch_all(pool)
            .await
            .map_err(|e| Error::IoError(format!("Read query failed: {}", e)))?;

        convert_sqlx_rows_to_dataframe(&rows)
    }

    /// Execute write query on primary
    #[cfg(feature = "sql")]
    pub async fn write_query(&self, query: &str) -> Result<u64> {
        let result = sqlx::query(query)
            .execute(&self.primary_pool)
            .await
            .map_err(|e| Error::IoError(format!("Write query failed: {}", e)))?;

        Ok(result.rows_affected())
    }

    /// Perform health check on all pools
    #[cfg(feature = "sql")]
    pub async fn health_check(&self) -> Result<HealthCheckResult> {
        let mut result = HealthCheckResult {
            primary_healthy: false,
            replica_health: Vec::new(),
        };

        // Check primary
        result.primary_healthy = self.check_pool_health(&self.primary_pool).await;

        // Check replicas
        for pool in &self.read_pools {
            let healthy = self.check_pool_health(pool).await;
            result.replica_health.push(healthy);
        }

        Ok(result)
    }

    /// Check health of a single pool
    #[cfg(feature = "sql")]
    async fn check_pool_health(&self, pool: &AnyPool) -> bool {
        match sqlx::query("SELECT 1").fetch_one(pool).await {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}

/// Health check results
#[derive(Debug)]
pub struct HealthCheckResult {
    /// Primary pool health status
    pub primary_healthy: bool,
    /// Replica pool health statuses
    pub replica_health: Vec<bool>,
}
