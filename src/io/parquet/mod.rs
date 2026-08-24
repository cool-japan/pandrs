//! Parquet file I/O.
//!
//! Split into focused submodules (kept under the project's 2000-line-per-file
//! policy): [`core`](crate::io::parquet::core) (compression validation and the basic read/write entry
//! points), [`convert`](crate::io::parquet::convert) (Arrow `RecordBatch` <-> `DataFrame` value
//! conversion), [`metadata`](crate::io::parquet::metadata) (file/row-group/column statistics and schema
//! analysis), [`advanced`](crate::io::parquet::advanced) (projection-aware reading and enhanced-type-aware
//! writing), [`streaming`](crate::io::parquet::streaming) (chunked/streaming reads), and [`evolution`](crate::io::parquet::evolution)
//! (schema evolution and post-read predicate pushdown). Every public item is
//! re-exported here so `pandrs::io::parquet::*` and `pandrs::io::*` are
//! unchanged by this split.

pub mod advanced;
pub mod convert;
pub mod core;
pub mod evolution;
pub mod metadata;
pub mod streaming;

#[cfg(test)]
mod tests;

pub use advanced::{read_parquet_advanced, write_parquet_advanced};
pub use core::{
    read_parquet, write_parquet, ColumnStats, ParquetCompression, ParquetMetadata,
    ParquetReadOptions, ParquetWriteOptions, RowGroupInfo,
};
pub use evolution::{
    read_parquet_with_predicates, read_parquet_with_schema_evolution, PredicateFilter,
    SchemaEvolution,
};
pub use metadata::{
    analyze_parquet_schema, get_column_statistics, get_parquet_metadata, get_row_group_info,
    ParquetSchemaAnalysis,
};
pub use streaming::{
    read_parquet_enhanced, write_parquet_streaming, AdvancedParquetReadOptions,
    StreamingParquetReader,
};
