use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

// SQLite support
use rusqlite::{Connection as SqliteConnection, Row, Statement};

// Multi-database support via sqlx (optional)
#[cfg(feature = "sql")]
use sqlx::{AnyPool, Column as SqlxColumn, Pool, Row as SqlxRow};

#[cfg(feature = "sql")]
use sqlx::any::{AnyArguments, AnyConnectOptions};

#[cfg(feature = "sql")]
use sqlx::ConnectOptions;

use crate::error::{Error, Result};

/// Database connection types supported by PandRS
#[derive(Debug, Clone)]
pub enum DatabaseConnection {
    /// SQLite file-based database
    Sqlite(String),
    /// PostgreSQL connection string
    #[cfg(feature = "sql")]
    PostgreSQL(String),
    /// MySQL connection string  
    #[cfg(feature = "sql")]
    MySQL(String),
    /// Generic database URL
    #[cfg(feature = "sql")]
    Generic(String),
}

/// Database connection pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections in pool
    pub max_connections: u32,
    /// Minimum number of connections to maintain
    pub min_connections: u32,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Maximum idle time before closing connection
    pub idle_timeout: Option<Duration>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            min_connections: 1,
            connect_timeout: Duration::from_secs(30),
            idle_timeout: Some(Duration::from_secs(600)),
        }
    }
}

/// SQL parameter value types
#[derive(Debug, Clone)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
    Boolean(bool),
}

/// Database connection manager
pub struct SqlConnection {
    connection_type: DatabaseConnection,
    #[cfg(feature = "sql")]
    pool: Option<AnyPool>,
}

impl SqlConnection {
    /// Create new database connection from URL
    pub fn from_url(url: &str) -> Result<Self> {
        let connection_type = if url.starts_with("sqlite:") || url.ends_with(".db") {
            DatabaseConnection::Sqlite(url.to_string())
        } else {
            #[cfg(feature = "sql")]
            {
                if url.starts_with("postgresql:") || url.starts_with("postgres:") {
                    DatabaseConnection::PostgreSQL(url.to_string())
                } else if url.starts_with("mysql:") {
                    DatabaseConnection::MySQL(url.to_string())
                } else {
                    DatabaseConnection::Generic(url.to_string())
                }
            }
            #[cfg(not(feature = "sql"))]
            {
                return Err(Error::IoError(
                    "Multi-database support requires 'sqlx' feature".to_string(),
                ));
            }
        };

        Ok(Self {
            connection_type,
            #[cfg(feature = "sql")]
            pool: None,
        })
    }

    /// Create connection with pooling
    #[cfg(feature = "sql")]
    pub async fn with_pool(url: &str, config: PoolConfig) -> Result<Self> {
        let options: AnyConnectOptions = url
            .parse()
            .map_err(|e| Error::IoError(format!("Invalid connection URL: {}", e)))?;

        let pool = Pool::connect_with(options)
            .await
            .map_err(|e| Error::IoError(format!("Failed to create connection pool: {}", e)))?;

        let connection_type = if url.starts_with("postgresql:") || url.starts_with("postgres:") {
            DatabaseConnection::PostgreSQL(url.to_string())
        } else if url.starts_with("mysql:") {
            DatabaseConnection::MySQL(url.to_string())
        } else {
            DatabaseConnection::Generic(url.to_string())
        };

        Ok(Self {
            connection_type,
            pool: Some(pool),
        })
    }

    /// Get the connection type
    pub fn connection_type(&self) -> &DatabaseConnection {
        &self.connection_type
    }

    /// Get the connection pool (if available)
    #[cfg(feature = "sql")]
    pub fn pool(&self) -> Option<&AnyPool> {
        self.pool.as_ref()
    }
}

/// Connection pool statistics
#[derive(Debug, Clone, Default)]
pub struct ConnectionStats {
    /// Total connections created
    pub total_connections: u64,
    /// Active connections
    pub active_connections: u32,
    /// Idle connections
    pub idle_connections: u32,
    /// Total queries executed
    pub total_queries: u64,
    /// Average query duration (milliseconds)
    pub avg_query_duration: f64,
}

/// Enhanced database connection pool manager with async support
pub struct AsyncDatabasePool {
    /// Connection pool
    #[cfg(feature = "sql")]
    pool: AnyPool,
    /// Pool configuration
    config: PoolConfig,
    /// Connection statistics
    stats: ConnectionStats,
}

impl AsyncDatabasePool {
    /// Create a new async database pool
    ///
    /// # Arguments
    ///
    /// * `url` - Database connection URL
    /// * `config` - Pool configuration
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - New async pool or error
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::sql::{AsyncDatabasePool, PoolConfig};
    /// use tokio;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let config = PoolConfig::default();
    ///     let pool = AsyncDatabasePool::new("postgresql://user:pass@localhost/db", config).await.unwrap();
    /// }
    /// ```
    #[cfg(feature = "sql")]
    pub async fn new(url: &str, _config: PoolConfig) -> Result<Self> {
        use sqlx::any::AnyPoolOptions;

        let pool = AnyPoolOptions::new()
            .max_connections(_config.max_connections)
            .min_connections(_config.min_connections)
            .acquire_timeout(_config.connect_timeout)
            .idle_timeout(_config.idle_timeout)
            .connect(url)
            .await
            .map_err(|e| Error::IoError(format!("Failed to create async pool: {}", e)))?;

        Ok(Self {
            pool,
            config: _config,
            stats: ConnectionStats::default(),
        })
    }

    /// Get connection pool statistics
    pub fn get_stats(&self) -> &ConnectionStats {
        &self.stats
    }

    /// Get the underlying pool
    #[cfg(feature = "sql")]
    pub fn pool(&self) -> &AnyPool {
        &self.pool
    }

    /// Close the connection pool
    #[cfg(feature = "sql")]
    pub async fn close(self) {
        self.pool.close().await;
    }
}

/// Implementation of ToSql for SqlValue
impl rusqlite::ToSql for SqlValue {
    fn to_sql(&self) -> rusqlite::Result<rusqlite::types::ToSqlOutput<'_>> {
        use rusqlite::types::{ToSqlOutput, Value};

        match self {
            SqlValue::Null => Ok(ToSqlOutput::Owned(Value::Null)),
            SqlValue::Integer(i) => Ok(ToSqlOutput::Owned(Value::Integer(*i))),
            SqlValue::Real(r) => Ok(ToSqlOutput::Owned(Value::Real(*r))),
            SqlValue::Text(s) => Ok(ToSqlOutput::Owned(Value::Text(s.clone()))),
            SqlValue::Blob(b) => Ok(ToSqlOutput::Owned(Value::Blob(b.clone()))),
            SqlValue::Boolean(b) => Ok(ToSqlOutput::Owned(Value::Integer(if *b { 1 } else { 0 }))),
        }
    }
}

/// Transaction isolation levels
#[derive(Debug, Clone)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

/// Advanced database transaction manager
pub struct TransactionManager {
    /// Connection pool
    #[cfg(feature = "sql")]
    pool: AnyPool,
    /// Transaction timeout
    timeout: Duration,
    /// Isolation level
    isolation_level: IsolationLevel,
}

impl TransactionManager {
    /// Create new transaction manager
    #[cfg(feature = "sql")]
    pub fn new(pool: AnyPool) -> Self {
        Self {
            pool,
            timeout: Duration::from_secs(30),
            isolation_level: IsolationLevel::ReadCommitted,
        }
    }

    /// Get the underlying pool
    #[cfg(feature = "sql")]
    pub fn pool(&self) -> &AnyPool {
        &self.pool
    }
}

/// Database operation for transactions
#[derive(Debug, Clone)]
pub enum DatabaseOperation {
    /// Query that returns results
    Query(String),
    /// Execute statement (no results)
    Execute(String),
}
