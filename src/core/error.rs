use crate::core::error_context::{ErrorContext, ErrorRecovery, ErrorRecoveryHelper};
use thiserror::Error;

/// Error type definitions
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    IoError(String),

    #[error("IO error")]
    Io(#[source] std::io::Error),

    #[error("CSV error: {0}")]
    CsvError(String),

    #[error("CSV error")]
    Csv(#[source] csv::Error),

    #[error("JSON error: {0}")]
    JsonError(String),

    #[error("JSON error")]
    Json(#[source] serde_json::Error),

    #[error("Parquet error: {0}")]
    ParquetError(String),

    #[error("Index out of bounds: index {index}, size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Index out of bounds: {0}")]
    IndexOutOfBoundsStr(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Column error: {0}")]
    Column(String),

    #[error("Duplicate column name: {0}")]
    DuplicateColumnName(String),

    #[error("Inconsistent row count: expected {expected}, found {found}")]
    InconsistentRowCount { expected: usize, found: usize },

    #[error("Column type mismatch: column {name}, expected {expected:?}, found {found:?}")]
    ColumnTypeMismatch {
        name: String,
        expected: crate::core::column::ColumnType,
        found: crate::core::column::ColumnType,
    },

    #[error("Invalid regex: {0}")]
    InvalidRegex(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    #[error("Empty data: {0}")]
    EmptyData(String),

    #[error("Empty DataFrame: {0}")]
    EmptyDataFrame(String),

    #[error("Empty data error: {0}")]
    Empty(String),

    #[error("Operation failed: {0}")]
    OperationFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Consistency error: {0}")]
    Consistency(String),

    #[error("Length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    // We already have IoError, CsvError, JsonError, etc. above
    // Removed duplicated error types
    #[error("Cast error: {0}")]
    Cast(String),

    #[error("Type error: {0}")]
    Type(String),

    #[error("Format error: {0}")]
    Format(String),

    #[error("Visualization error: {0}")]
    Visualization(String),

    #[error("Parallel processing error: {0}")]
    Parallel(String),

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Operation error: {0}")]
    Operation(String),

    #[error("Computation error: {0}")]
    Computation(String),

    #[error("Dimension mismatch error: {0}")]
    DimensionMismatch(String),

    #[error("Insufficient data error: {0}")]
    InsufficientData(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Empty column list")]
    EmptyColumnList,

    #[error("Empty series")]
    EmptySeries,

    #[error("Inconsistent array lengths: expected {expected}, found {found}")]
    InconsistentArrayLengths { expected: usize, found: usize },

    // Distributed processing errors
    #[error("Distributed processing error: {0}")]
    DistributedProcessing(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    #[error("Timeout error: {0}")]
    TimeoutError(String),

    #[error("Executor error: {0}")]
    ExecutorError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Distributed configuration error: {0}")]
    DistributedConfigError(String),

    #[error("Data partitioning error: {0}")]
    PartitioningError(String),

    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    #[error("Other error: {0}")]
    Other(String),

    /// Lock poisoned error - occurs when a thread panics while holding a lock
    #[error("Lock poisoned: {context}")]
    LockPoisoned { context: String },

    /// Type mismatch error
    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    /// Parse error
    #[error("Parse error: failed to parse '{value}' as {target_type}: {error_msg}")]
    ParseError {
        value: String,
        target_type: String,
        error_msg: String,
    },

    /// Enhanced error with context
    #[error("Enhanced error: {message}")]
    Enhanced {
        message: String,
        context: ErrorContext,
    },
}

// Maintaining backward compatibility with PandRSError
pub type PandRSError = Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Create a lock poisoned error
    ///
    /// # Arguments
    /// * `context` - Context string describing where the lock poison occurred
    ///
    /// # Returns
    /// * `Error::LockPoisoned` - The lock poisoned error
    pub fn lock_poisoned(context: impl Into<String>) -> Self {
        Error::LockPoisoned {
            context: context.into(),
        }
    }

    /// Create an index out of bounds error
    ///
    /// # Arguments
    /// * `index` - The index that was out of bounds
    /// * `size` - The size of the collection
    ///
    /// # Returns
    /// * `Error::IndexOutOfBounds` - The index out of bounds error
    pub fn index_out_of_bounds(index: usize, size: usize) -> Self {
        Error::IndexOutOfBounds { index, size }
    }

    /// Create a parse error
    ///
    /// # Arguments
    /// * `value` - The string value that failed to parse
    /// * `target_type` - The target type we tried to parse into
    /// * `err` - The underlying error
    ///
    /// # Returns
    /// * `Error::ParseError` - The parse error
    pub fn parse_error<T: std::fmt::Display>(value: &str, target_type: &str, err: T) -> Self {
        Error::ParseError {
            value: value.to_string(),
            target_type: target_type.to_string(),
            error_msg: err.to_string(),
        }
    }
}

/// Extension trait for Option to provide ergonomic error conversion
pub trait OptionExt<T> {
    /// Convert None to an index out of bounds error
    ///
    /// # Arguments
    /// * `index` - The index that was out of bounds
    /// * `size` - The size of the collection
    ///
    /// # Returns
    /// * `Result<T>` - Ok if Some, Err with index error if None
    fn ok_or_index_error(self, index: usize, size: usize) -> Result<T>;

    /// Convert None to a column not found error
    ///
    /// # Arguments
    /// * `column` - The column name that was not found
    ///
    /// # Returns
    /// * `Result<T>` - Ok if Some, Err with column error if None
    fn ok_or_column_error(self, column: impl Into<String>) -> Result<T>;

    /// Convert None to an empty data error
    ///
    /// # Arguments
    /// * `context` - Context string for the error
    ///
    /// # Returns
    /// * `Result<T>` - Ok if Some, Err with empty data error if None
    fn ok_or_empty_error(self, context: impl Into<String>) -> Result<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_index_error(self, index: usize, size: usize) -> Result<T> {
        self.ok_or_else(|| Error::index_out_of_bounds(index, size))
    }

    fn ok_or_column_error(self, column: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| Error::ColumnNotFound(column.into()))
    }

    fn ok_or_empty_error(self, context: impl Into<String>) -> Result<T> {
        self.ok_or_else(|| Error::EmptyData(context.into()))
    }
}

// Conversions from standard errors (From implementations are automatically implemented by #[derive(Error)])
impl From<csv::Error> for Error {
    fn from(err: csv::Error) -> Self {
        Error::Csv(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Json(err)
    }
}

impl From<regex::Error> for Error {
    fn from(err: regex::Error) -> Self {
        Error::InvalidRegex(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

// Implement ErrorRecovery trait for Error
impl ErrorRecovery for Error {
    fn suggest_fixes(&self) -> Vec<String> {
        match self {
            Error::ColumnNotFound(name) => {
                vec![
                    format!("Column '{}' not found", name),
                    "Use .columns() to list available columns".to_string(),
                    "Check for typos in column name".to_string(),
                ]
            }
            Error::ColumnTypeMismatch {
                name,
                expected,
                found,
            } => ErrorRecoveryHelper::type_mismatch_suggestions(
                name,
                &format!("{:?}", expected),
                &format!("{:?}", found),
            ),
            Error::InconsistentRowCount { expected, found } => {
                ErrorRecoveryHelper::shape_mismatch_suggestions((*expected, 0), (*found, 0))
            }
            Error::IndexOutOfBounds { index, size } => {
                vec![
                    format!("Index {} is out of bounds for size {}", index, size),
                    format!("Valid range is 0 to {}", size.saturating_sub(1)),
                    "Use .len() to check size before indexing".to_string(),
                ]
            }
            Error::InvalidInput(msg) => {
                vec![
                    format!("Invalid input: {}", msg),
                    "Check input data format and types".to_string(),
                    "Refer to documentation for expected input format".to_string(),
                ]
            }
            Error::LockPoisoned { context } => {
                vec![
                    format!("Lock poisoned: {}", context),
                    "A thread panicked while holding this lock".to_string(),
                    "Consider restarting the operation or reinitializing the data structure"
                        .to_string(),
                ]
            }
            Error::ParseError {
                value,
                target_type,
                error_msg,
            } => {
                vec![
                    format!(
                        "Failed to parse '{}' as {}: {}",
                        value, target_type, error_msg
                    ),
                    "Check the input format matches the expected type".to_string(),
                    "Verify the data source provides valid values".to_string(),
                ]
            }
            Error::Enhanced { context, .. } => context.suggested_fixes.clone(),
            _ => vec!["Refer to PandRS documentation for error resolution".to_string()],
        }
    }

    fn can_auto_recover(&self) -> bool {
        match self {
            Error::ColumnTypeMismatch { .. } => true, // Could auto-convert types
            Error::IndexOutOfBounds { .. } => false,  // Cannot auto-fix bounds
            Error::ColumnNotFound(_) => false,        // Cannot guess column names
            Error::Enhanced { context, .. } => {
                // Could implement more sophisticated recovery logic
                !context.suggested_fixes.is_empty()
            }
            _ => false,
        }
    }

    fn attempt_recovery(&self) -> std::result::Result<Option<Box<dyn std::any::Any>>, Error> {
        match self {
            Error::ColumnTypeMismatch { .. } => {
                // In a real implementation, this would attempt type conversion
                Err(Error::NotImplemented(
                    "Auto-recovery for type mismatch".to_string(),
                ))
            }
            _ => Err(Error::NotImplemented("Auto-recovery".to_string())),
        }
    }

    fn error_context(&self) -> Option<&ErrorContext> {
        match self {
            Error::Enhanced { context, .. } => Some(context),
            _ => None,
        }
    }
}

// Add support for JIT errors
impl From<crate::optimized::jit::JitError> for Error {
    fn from(err: crate::optimized::jit::JitError) -> Self {
        Error::InvalidOperation(err.to_string())
    }
}

// Helper function to generate std::io::Error from String error messages
pub fn io_error<T: AsRef<str>>(msg: T) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, msg.as_ref())
}

// Conversion for Plotters errors
#[cfg(feature = "plotters")]
impl<E: std::error::Error + Send + Sync + 'static> From<plotters::drawing::DrawingAreaErrorKind<E>>
    for Error
{
    fn from(err: plotters::drawing::DrawingAreaErrorKind<E>) -> Self {
        Error::Visualization(format!("Plot drawing error: {}", err))
    }
}

// Conversion for WASM/JS errors
#[cfg(feature = "wasm")]
impl From<Error> for wasm_bindgen::JsValue {
    fn from(err: Error) -> Self {
        wasm_bindgen::JsValue::from_str(&err.to_string())
    }
}
