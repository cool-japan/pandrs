/// CSV (Comma-Separated Values) file format support.
///
/// Read and write CSV files with customizable options:
/// - Custom delimiters
/// - Header row handling
/// - Type inference
/// - Missing value handling
///
/// # Examples
///
/// ```rust,no_run
/// use pandrs::io;
///
/// // Read CSV with headers
/// let df = io::read_csv("data.csv", true).expect("Failed to read CSV");
///
/// // Write CSV
/// io::write_csv(&df, "output.csv").expect("Failed to write CSV");
/// ```
///
/// # Performance Tips
///
/// - For large files, consider using chunked reading
/// - Specify column types explicitly when known
/// - Use appropriate buffer sizes for better I/O performance
pub mod csv;

/// Excel file format support (requires `excel` feature).
///
/// Read and write Microsoft Excel files (.xlsx, .xls) with:
/// - Multiple sheet support
/// - Cell formatting
/// - Formula evaluation
/// - Named ranges
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "excel")]
/// # {
/// use pandrs::io;
///
/// // Read specific sheet
/// let df = io::read_excel("workbook.xlsx", Some("Sheet1"), true, 0, None)
///     .expect("Failed to read Excel");
///
/// // Write to Excel (requires OptimizedDataFrame)
/// // let odf = pandrs::OptimizedDataFrame::from_dataframe(&df).expect("convert");
/// // io::write_excel(&odf, "output.xlsx", Some("Data"), false)
/// //     .expect("Failed to write Excel");
/// # }
/// ```
#[cfg(feature = "excel")]
pub mod excel;

/// Format trait definitions for extensible I/O.
///
/// Defines traits and types for implementing custom file format handlers.
pub mod format_traits;

/// JSON (JavaScript Object Notation) file format support.
///
/// Read and write JSON files with:
/// - Records orientation
/// - Columns orientation
/// - Split orientation
/// - Pretty printing
///
/// # Examples
///
/// ```rust,no_run
/// use pandrs::io;
/// use pandrs::io::json::JsonOrient;
///
/// // Read JSON
/// let df = io::read_json("data.json").expect("Failed to read JSON");
///
/// // Write JSON
/// io::write_json(&df, "output.json", JsonOrient::Records).expect("Failed to write JSON");
/// ```
pub mod json;

/// Parquet columnar file format support (requires `parquet` feature).
///
/// Read and write Apache Parquet files for efficient columnar storage:
/// - Columnar compression
/// - Predicate pushdown
/// - Schema evolution
/// - Row group statistics
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "parquet")]
/// # {
/// use pandrs::io;
///
/// // Read Parquet file
/// let df = io::read_parquet("data.parquet")
///     .expect("Failed to read Parquet");
///
/// // Write with compression (requires OptimizedDataFrame)
/// // let odf = pandrs::OptimizedDataFrame::from_dataframe(&df).expect("convert");
/// // io::write_parquet(&odf, "output.parquet", None)
/// //     .expect("Failed to write Parquet");
/// # }
/// ```
///
/// # Performance Tips
///
/// - Parquet is optimized for columnar operations
/// - Use predicate filters to read only needed data
/// - Choose appropriate compression (snappy, gzip, lz4)
#[cfg(feature = "parquet")]
pub mod parquet;

/// SQL database connectivity (requires `sql` feature).
///
/// Connect to relational databases and execute queries:
/// - PostgreSQL
/// - MySQL
/// - SQLite
/// - Connection pooling
/// - Transaction support
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "sql")]
/// # {
/// use pandrs::io;
///
/// // Read from database (SQLite)
/// let db_path = "data.db";
/// let df = io::read_sql("SELECT * FROM users", db_path)
///     .expect("Failed to read from SQL");
///
/// // Write to database (requires OptimizedDataFrame)
/// // let odf = pandrs::OptimizedDataFrame::from_dataframe(&df).expect("convert");
/// // io::write_to_sql(&odf, "users_backup", db_path, "replace")
/// //     .expect("Failed to write to SQL");
/// # }
/// ```
///
/// # Security Best Practices
///
/// - Use connection pooling for better performance
/// - Always use parameterized queries to prevent SQL injection
/// - Store credentials securely using the config module
#[cfg(feature = "sql")]
pub mod sql;

/// Streaming I/O for processing data in chunks.
///
/// Process large datasets that don't fit in memory:
/// - Chunked reading and writing
/// - Pipeline processing
/// - Backpressure handling
#[cfg(feature = "streaming")]
pub mod streaming;

// Re-export commonly used functions
pub use csv::{read_csv, write_csv};
#[cfg(feature = "excel")]
pub use excel::{
    analyze_excel_file, get_sheet_info, get_workbook_info, list_sheet_names, optimize_excel_file,
    read_excel, read_excel_enhanced, read_excel_sheets, read_excel_with_info, write_excel,
    write_excel_enhanced, write_excel_sheets, ExcelCell, ExcelCellFormat, ExcelFileAnalysis,
    ExcelReadOptions, ExcelSheetInfo, ExcelWorkbookInfo, ExcelWriteOptions, NamedRange,
};
pub use format_traits::{
    ColumnConstraint, ColumnDefinition as FormatColumnDefinition, DataDestination, DataOperations,
    DataSource, FileFormat, ForeignKeyConstraint, FormatCapabilities, FormatDataType,
    FormatRegistry, IndexDefinition, IndexType, JoinType, ReferentialAction, SerializationFormat,
    SqlCapabilities, SqlDataType, SqlOps, SqlStandard, StreamingCapabilities, StreamingOps,
    TableSchema as FormatTableSchema, TransformPipeline, TransformStage,
};
pub use json::{read_json, write_json};
#[cfg(feature = "parquet")]
pub use parquet::{
    analyze_parquet_schema, get_column_statistics, get_parquet_metadata, get_row_group_info,
    read_parquet, read_parquet_advanced, read_parquet_enhanced, read_parquet_with_predicates,
    read_parquet_with_schema_evolution, write_parquet, write_parquet_advanced,
    write_parquet_streaming, AdvancedParquetReadOptions, ColumnStats, ParquetCompression,
    ParquetMetadata, ParquetReadOptions, ParquetSchemaAnalysis, ParquetWriteOptions,
    PredicateFilter, RowGroupInfo, SchemaEvolution, StreamingParquetReader,
};
#[cfg(feature = "sql")]
pub use sql::{
    execute_sql, get_create_table_sql, get_table_schema, has_table, list_tables, read_sql,
    read_sql_advanced, read_sql_table, write_sql_advanced, write_to_sql, AsyncDatabasePool,
    ColumnDefinition, ConnectionStats, DatabaseConnection, DatabaseOperation, ForeignKey,
    InsertMethod, IsolationLevel, PoolConfig, QueryBuilder, SchemaIntrospector, SqlConnection,
    SqlReadOptions, SqlValue, SqlWriteOptions, TableSchema, TransactionManager, WriteMode,
};
#[cfg(feature = "streaming")]
pub use streaming::{
    DataFrameStreaming, ErrorAction, ErrorHandler, ErrorStats, ErrorStrategy, MemoryStreamSink,
    MemoryStreamSource, PipelineConfig, PipelineStage, PipelineStats, ProcessorConfig,
    ProcessorMetadata, ProcessorStats, ProcessorType, SinkMetadata, SinkType, StageStats,
    StreamDataType, StreamField, StreamMetadata, StreamProcessor, StreamSchema, StreamType,
    StreamWindow, StreamingDataSink, StreamingDataSource, StreamingPipeline, WindowStats,
    WindowType,
};
