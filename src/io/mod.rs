pub mod csv;
#[cfg(feature = "excel")]
pub mod excel;
pub mod format_traits;
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;
#[cfg(feature = "sql")]
pub mod sql;
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
