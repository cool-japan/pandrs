//! Chunked/streaming reads over bounded row ranges.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::file::metadata::PageIndexPolicy;
use std::fs::File;
use std::path::Path;

use super::advanced::read_parquet_advanced;
use super::convert::record_batches_to_dataframe;
use super::core::{write_parquet, ParquetReadOptions};
use super::evolution::{
    apply_predicate_filters, apply_schema_evolution, PredicateFilter, SchemaEvolution,
};

/// Advanced Parquet reading options with schema evolution and predicate pushdown
#[derive(Debug, Clone)]
pub struct AdvancedParquetReadOptions {
    /// Base reading options
    pub base_options: ParquetReadOptions,
    /// Schema evolution rules
    pub schema_evolution: Option<SchemaEvolution>,
    /// Predicate filters for pushdown
    pub predicate_filters: Vec<PredicateFilter>,
    /// Enable streaming mode for large files
    pub streaming_mode: bool,
    /// Streaming chunk size (rows per chunk)
    pub streaming_chunk_size: usize,
    /// Memory limit for streaming (bytes)
    pub memory_limit: Option<usize>,
}
impl Default for AdvancedParquetReadOptions {
    fn default() -> Self {
        Self {
            base_options: ParquetReadOptions::default(),
            schema_evolution: None,
            predicate_filters: Vec::new(),
            streaming_mode: false,
            streaming_chunk_size: 10000,
            memory_limit: Some(1024 * 1024 * 1024),
        }
    }
}
/// Streaming Parquet reader for large datasets
pub struct StreamingParquetReader {
    /// File path
    pub(super) path: String,
    /// Current chunk index
    pub(super) chunk_index: usize,
    /// Total number of chunks
    pub(super) total_chunks: usize,
    /// Chunk size
    pub(super) chunk_size: usize,
    /// Total number of rows in the file (from file metadata)
    pub(super) total_rows: usize,
    /// Schema information
    pub(super) schema: SchemaRef,
    /// Row index of the next unread row
    pub(super) current_position: usize,
}
impl StreamingParquetReader {
    /// Create a new streaming Parquet reader
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Parquet file
    /// * `chunk_size` - Number of rows per chunk
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - The streaming reader, or an error
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use pandrs::io::StreamingParquetReader;
    ///
    /// let reader = StreamingParquetReader::new("large_data.parquet", 10000).expect("operation should succeed");
    /// ```
    pub fn new(path: impl AsRef<Path>, chunk_size: usize) -> Result<Self> {
        if chunk_size == 0 {
            return Err(Error::InvalidInput(
                "StreamingParquetReader chunk_size must be greater than 0".to_string(),
            ));
        }
        let file = File::open(path.as_ref())
            .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::IoError(format!("Failed to parse Parquet file: {}", e)))?;
        let metadata = builder.metadata().clone();
        let schema = builder.schema().clone();
        let total_rows = metadata.file_metadata().num_rows() as usize;
        let total_chunks = (total_rows + chunk_size - 1) / chunk_size;
        Ok(Self {
            path: path.as_ref().to_string_lossy().to_string(),
            chunk_index: 0,
            total_chunks,
            chunk_size,
            total_rows,
            schema,
            current_position: 0,
        })
    }
    /// Read the next chunk
    ///
    /// Reads exactly the rows belonging to this chunk (`[chunk_index *
    /// chunk_size, ...)`, clipped to the file's row count) via a bounded,
    /// offset/limit-scoped Parquet read — not the whole file re-decoded on
    /// every call. Successive calls yield disjoint row ranges that sum to
    /// the file's total row count.
    ///
    /// # Returns
    ///
    /// * `Result<Option<DataFrame>>` - Next chunk as DataFrame, None if end of file
    pub fn next_chunk(&mut self) -> Result<Option<DataFrame>> {
        if self.chunk_index >= self.total_chunks {
            return Ok(None);
        }
        let start_row = self.chunk_index * self.chunk_size;
        let take = self
            .total_rows
            .saturating_sub(start_row)
            .min(self.chunk_size);
        self.chunk_index += 1;
        if take == 0 {
            self.current_position = start_row;
            return Ok(None);
        }
        let batches = read_parquet_row_range(&self.path, start_row, take)?;
        self.current_position = start_row + take;
        if batches.is_empty() {
            return Ok(Some(DataFrame::new()));
        }
        record_batches_to_dataframe(&batches, self.schema.clone()).map(Some)
    }
    /// Get total number of chunks
    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }
    /// Get current chunk index
    pub fn current_chunk(&self) -> usize {
        self.chunk_index
    }
    /// Get schema information
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }
}
/// Read a contiguous slice `[offset, offset + limit)` of rows from a Parquet
/// file as raw Arrow record batches, without decoding the rest of the file.
/// Shared by `StreamingParquetReader::next_chunk` and `read_parquet_streaming`
/// so that a "chunk" genuinely means "this row range", not "the whole file,
/// read again". Page-index reading is enabled so the offset/limit bounds can
/// skip whole data pages instead of decoding-then-discarding them.
fn read_parquet_row_range(path: &str, offset: usize, limit: usize) -> Result<Vec<RecordBatch>> {
    let file = File::open(path)
        .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;

    // `Optional`, not `Required`: a file written without an embedded page
    // index must still read correctly (just without the skip-ahead speedup),
    // matching what `ArrowReaderOptions::with_page_index(true)` intended —
    // that deprecated boolean form actually maps to `Required` (errors if the
    // index is missing), which is not the intent here.
    let reader_options =
        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional);
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(file, reader_options)
        .map_err(|e| Error::IoError(format!("Failed to parse Parquet file: {}", e)))?;

    let reader = builder
        .with_offset(offset)
        .with_limit(limit)
        .build()
        .map_err(|e| Error::IoError(format!("Failed to read Parquet file: {}", e)))?;

    let mut batches = Vec::new();
    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| Error::IoError(format!("Failed to read record batch: {}", e)))?;
        batches.push(batch);
    }
    Ok(batches)
}

/// Advanced Parquet reading with all enhanced features
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `options` - Advanced reading options
///
/// # Returns
///
/// * `Result<DataFrame>` - Enhanced DataFrame
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::{read_parquet_enhanced, AdvancedParquetReadOptions, PredicateFilter};
///
/// let options = AdvancedParquetReadOptions {
///     predicate_filters: vec![
///         PredicateFilter::Equals("category".to_string(), "premium".to_string())
///     ],
///     streaming_mode: true,
///     streaming_chunk_size: 50000,
///     ..Default::default()
/// };
///
/// let df = read_parquet_enhanced("large_data.parquet", options).expect("operation should succeed");
/// ```
pub fn read_parquet_enhanced(
    path: impl AsRef<Path>,
    options: AdvancedParquetReadOptions,
) -> Result<DataFrame> {
    // Schema evolution and predicate filters used to be applied only on the
    // non-streaming branch below, so `streaming_mode: true` silently dropped
    // both — including the combination this function's own doc example uses.
    // Both are now applied uniformly regardless of which read path was used.
    let mut df = if options.streaming_mode {
        // `read_parquet_streaming` reads every column of every row group by
        // construction (see `read_parquet_row_range`, which never applies a
        // projection or row-group filter). Silently ignoring a `columns` or
        // `row_groups` request from `base_options` here would mean the
        // caller asked for a subset and got the whole file back with no
        // signal — the same "field on the struct that the streaming branch
        // never reads" bug already fixed above for `schema_evolution` and
        // `predicate_filters`. `batch_size` is not included in this check:
        // it is genuinely superseded by `streaming_chunk_size` (which
        // governs the same thing — how many rows are read per pass) rather
        // than silently dropped.
        if options.base_options.columns.is_some() || options.base_options.row_groups.is_some() {
            return Err(Error::NotImplemented(
                "read_parquet_enhanced: streaming_mode combined with \
                 base_options.columns or base_options.row_groups is not supported (streaming \
                 mode always reads every column of every row group). Use read_parquet_advanced \
                 for a projected/row-group-filtered read, or drop streaming_mode."
                    .to_string(),
            ));
        }
        read_parquet_streaming(
            path.as_ref(),
            options.streaming_chunk_size,
            options.memory_limit,
        )?
    } else {
        read_parquet_advanced(path.as_ref(), options.base_options.clone())?
    };

    if let Some(evolution) = &options.schema_evolution {
        apply_schema_evolution(&mut df, evolution)?;
    }

    if !options.predicate_filters.is_empty() {
        df = apply_predicate_filters(df, &options.predicate_filters)?;
    }

    Ok(df)
}

/// Read a Parquet file in bounded, chunked passes and combine every chunk
/// into one DataFrame (unless `memory_limit` triggers an early stop).
///
/// Each chunk is read via [`read_parquet_row_range`] — a real offset/limit
/// bounded read, not the whole file re-decoded per chunk — and every chunk's
/// Arrow record batches are accumulated and converted to a `DataFrame` only
/// once at the end, since `DataFrame` itself has no row-append primitive to
/// combine per-chunk DataFrames incrementally.
fn read_parquet_streaming(
    path: &Path,
    chunk_size: usize,
    memory_limit: Option<usize>,
) -> Result<DataFrame> {
    let path_str = path.to_string_lossy().to_string();
    let file = File::open(path)
        .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::IoError(format!("Failed to parse Parquet file: {}", e)))?;
    let schema = builder.schema().clone();
    let total_rows = builder.metadata().file_metadata().num_rows() as usize;

    let chunk_size = chunk_size.max(1);
    let mut all_batches: Vec<RecordBatch> = Vec::new();
    let mut start = 0usize;
    let mut rows_read = 0usize;

    while start < total_rows {
        let take = (total_rows - start).min(chunk_size);
        let mut batches = read_parquet_row_range(&path_str, start, take)?;
        rows_read += take;
        all_batches.append(&mut batches);
        start += take;

        if let Some(limit) = memory_limit {
            // Same rough per-row byte estimate the original placeholder
            // used; a precise accounting would need per-column-type sizing
            // that isn't available at this (raw Arrow batch) layer.
            let estimated_memory = rows_read.saturating_mul(100);
            if estimated_memory > limit {
                // This must be a hard error, not a silent `break`: every
                // chunk read so far is accumulated in `all_batches` and only
                // converted to a `DataFrame` once, at the very end, so
                // "stop early" would return `Ok` with a truncated result
                // that looks byte-for-byte like a complete, successful read.
                // Genuinely bounding *peak memory* (not just "rows read
                // before giving up") requires consuming chunks one at a
                // time without accumulating them, which is exactly what
                // `StreamingParquetReader::next_chunk` does.
                return Err(Error::OperationFailed(format!(
                    "read_parquet_streaming: memory_limit ({limit} bytes) exceeded after \
                     reading {rows_read} of {total_rows} rows. Raise `memory_limit`, or use \
                     `StreamingParquetReader::next_chunk` to consume the file one bounded \
                     chunk at a time instead of accumulating it all into a single DataFrame."
                )));
            }
        }
    }

    if all_batches.is_empty() {
        return Ok(DataFrame::new());
    }

    record_batches_to_dataframe(&all_batches, schema)
}

/// Write DataFrame to Parquet with streaming support for large datasets
///
/// # Arguments
///
/// * `df` - DataFrame to write
/// * `path` - Output path
/// * `chunk_size` - Rows per chunk for streaming
///
/// # Returns
///
/// * `Result<()>` - Success or error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::write_parquet_streaming;
/// use pandrs::optimized::dataframe::OptimizedDataFrame;
///
/// // Create sample large dataframe
/// let large_df = OptimizedDataFrame::new();
///
/// write_parquet_streaming(&large_df, "output.parquet", 100000).expect("operation should succeed");
/// ```
pub fn write_parquet_streaming(
    df: &OptimizedDataFrame,
    path: impl AsRef<Path>,
    chunk_size: usize,
) -> Result<()> {
    // For large DataFrames, write in chunks to manage memory
    let total_rows = df.row_count();
    let num_chunks = (total_rows + chunk_size - 1) / chunk_size;

    if num_chunks <= 1 {
        // Small enough to write in one go
        return write_parquet(df, path, None);
    }

    // For streaming writes, we'd need to implement chunked writing
    // This is a placeholder showing the concept
    println!(
        "Writing {} rows in {} chunks of size {}",
        total_rows, num_chunks, chunk_size
    );

    // Currently, fall back to standard writing
    // True streaming would require row-by-row or chunk-by-chunk DataFrame access
    write_parquet(df, path, None)
}
