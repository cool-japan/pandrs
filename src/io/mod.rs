pub mod csv;
#[cfg(feature = "excel")]
pub mod excel;
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;
#[cfg(feature = "sql")]
pub mod sql;
#[cfg(feature = "streaming")]
pub mod streaming;
pub mod format_traits;

// Re-export commonly used functions
pub use csv::{read_csv, write_csv};
#[cfg(feature = "excel")]
pub use excel::{
    read_excel, write_excel, list_sheet_names, get_workbook_info, get_sheet_info,
    read_excel_sheets, read_excel_with_info, write_excel_sheets,
    ExcelWorkbookInfo, ExcelSheetInfo
};
pub use json::{read_json, write_json};
#[cfg(feature = "parquet")]
pub use parquet::{
    read_parquet, write_parquet, read_parquet_advanced, write_parquet_advanced,
    get_parquet_metadata, get_row_group_info, get_column_statistics,
    ParquetCompression, ParquetMetadata, RowGroupInfo, ColumnStats,
    ParquetReadOptions, ParquetWriteOptions
};
#[cfg(feature = "sql")]
pub use sql::{
    execute_sql, read_sql, write_to_sql, read_sql_advanced, read_sql_table, write_sql_advanced,
    has_table, list_tables, get_table_schema, get_create_table_sql,
    SqlConnection, DatabaseConnection, PoolConfig, SqlReadOptions, SqlWriteOptions,
    WriteMode, InsertMethod, SqlValue, TableSchema, ColumnDefinition, ForeignKey
};
#[cfg(feature = "streaming")]
pub use streaming::{
    StreamingDataSource, StreamingDataSink, StreamProcessor, StreamingPipeline, DataFrameStreaming,
    StreamMetadata, SinkMetadata, StreamSchema, StreamField, StreamDataType, StreamType, SinkType,
    ProcessorMetadata, ProcessorType, ProcessorConfig, ProcessorStats, PipelineConfig, PipelineStats,
    StreamWindow, WindowType, WindowStats, ErrorStrategy, ErrorHandler, ErrorAction, ErrorStats,
    MemoryStreamSource, MemoryStreamSink, StageStats, PipelineStage,
};
pub use format_traits::{
    FileFormat, SqlOps, StreamingOps, DataOperations, FormatRegistry,
    FormatCapabilities, SqlCapabilities, StreamingCapabilities,
    DataSource, DataDestination, TransformPipeline, TransformStage, JoinType,
    TableSchema, ColumnDefinition, SqlDataType, ColumnConstraint,
    ForeignKeyConstraint, ReferentialAction, IndexDefinition, IndexType,
    FormatDataType, SqlStandard, SerializationFormat,
};
