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
}

// Maintaining backward compatibility with PandRSError
pub type PandRSError = Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

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
