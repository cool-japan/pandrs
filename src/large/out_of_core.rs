//! Out-of-core DataFrame processing for datasets larger than RAM.
//!
//! This module provides streaming chunk-based processing so that arbitrarily
//! large CSV/JSON datasets can be processed without loading them entirely into
//! memory.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use csv::{ReaderBuilder, WriterBuilder};
use rayon::prelude::*;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::Series;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for out-of-core processing.
#[derive(Debug, Clone)]
pub struct OutOfCoreConfig {
    /// Number of data rows per chunk (not counting the CSV header).
    pub chunk_size: usize,
    /// Maximum bytes of RAM to use at a time (informational; actual enforcement
    /// is done by keeping only `chunk_size` rows in memory at once).
    pub max_memory_bytes: usize,
    /// Directory where temporary chunk files are written.
    pub temp_dir: PathBuf,
    /// Whether to use gzip compression for temporary files.
    pub compression: bool,
    /// Number of chunks to process in parallel (via rayon).
    pub parallelism: usize,
}

impl Default for OutOfCoreConfig {
    fn default() -> Self {
        OutOfCoreConfig {
            chunk_size: 100_000,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            temp_dir: std::env::temp_dir(),
            compression: false,
            parallelism: num_cpus::get(),
        }
    }
}

// ---------------------------------------------------------------------------
// DataFormat
// ---------------------------------------------------------------------------

/// Supported source file formats for out-of-core processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataFormat {
    Csv,
    Json,
    Parquet,
}

// ---------------------------------------------------------------------------
// AggOp
// ---------------------------------------------------------------------------

/// Aggregation operations supported across chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

// ---------------------------------------------------------------------------
// Internal helper: chunk writer
// ---------------------------------------------------------------------------

/// Write a slice of CSV rows (with header) to a temp file, returning its path.
fn write_chunk_file(
    rows: &[Vec<String>],
    headers: &[String],
    dir: &Path,
    index: usize,
) -> Result<PathBuf> {
    let path = dir.join(format!("pandrs_ooc_chunk_{}.csv", index));
    let file = File::create(&path).map_err(|e| Error::IoError(e.to_string()))?;
    let mut wtr = WriterBuilder::new().from_writer(BufWriter::new(file));

    wtr.write_record(headers)
        .map_err(|e| Error::CsvError(e.to_string()))?;
    for row in rows {
        wtr.write_record(row)
            .map_err(|e| Error::CsvError(e.to_string()))?;
    }
    wtr.flush().map_err(|e| Error::IoError(e.to_string()))?;
    Ok(path)
}

/// Read a temp chunk file back into a DataFrame.
fn read_chunk_file(path: &Path) -> Result<DataFrame> {
    crate::io::csv::read_csv(path, true)
}

/// Convert a DataFrame to CSV rows (excluding header row).
fn dataframe_to_rows(df: &DataFrame) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let col_names = df.column_names();
    let row_count = df.row_count();
    let mut rows: Vec<Vec<String>> = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let mut row: Vec<String> = Vec::with_capacity(col_names.len());
        for col in &col_names {
            let val = df.get_string_value(col, i).unwrap_or("").to_string();
            row.push(val);
        }
        rows.push(row);
    }
    Ok((col_names, rows))
}

/// Write a full DataFrame to a CSV file.
fn write_dataframe_csv(df: &DataFrame, path: &Path) -> Result<()> {
    let (headers, rows) = dataframe_to_rows(df)?;
    let file = File::create(path).map_err(|e| Error::IoError(e.to_string()))?;
    let mut wtr = WriterBuilder::new().from_writer(BufWriter::new(file));
    wtr.write_record(&headers)
        .map_err(|e| Error::CsvError(e.to_string()))?;
    for row in &rows {
        wtr.write_record(row)
            .map_err(|e| Error::CsvError(e.to_string()))?;
    }
    wtr.flush().map_err(|e| Error::IoError(e.to_string()))
}

// ---------------------------------------------------------------------------
// OutOfCoreReader
// ---------------------------------------------------------------------------

/// Iterator-based out-of-core DataFrame processor.
///
/// Reads a large source file in chunks of `config.chunk_size` rows.
pub struct OutOfCoreReader {
    source_path: PathBuf,
    pub(crate) format: DataFormat,
    pub(crate) config: OutOfCoreConfig,
    /// Cached total row count (excluding header), filled lazily.
    total_rows: Option<usize>,
}

impl OutOfCoreReader {
    /// Create a reader for a CSV source.
    pub fn from_csv(path: impl AsRef<Path>, config: OutOfCoreConfig) -> Result<Self> {
        let source_path = path.as_ref().to_path_buf();
        if !source_path.exists() {
            return Err(Error::IoError(format!(
                "File not found: {}",
                source_path.display()
            )));
        }
        Ok(OutOfCoreReader {
            source_path,
            format: DataFormat::Csv,
            config,
            total_rows: None,
        })
    }

    // -----------------------------------------------------------------------
    // count
    // -----------------------------------------------------------------------

    /// Count the total number of data rows in the source file (excluding header).
    pub fn count(&self) -> Result<usize> {
        match self.format {
            DataFormat::Csv => self.count_csv_rows(),
            DataFormat::Json => Err(Error::NotImplemented("count() for JSON format".into())),
            DataFormat::Parquet => Err(Error::NotImplemented("count() for Parquet format".into())),
        }
    }

    fn count_csv_rows(&self) -> Result<usize> {
        let file = File::open(&self.source_path).map_err(|e| Error::IoError(e.to_string()))?;
        let reader = BufReader::new(file);
        // subtract 1 for the header line
        let mut count: usize = 0;
        let mut first = true;
        for line_result in reader.lines() {
            let _ = line_result.map_err(|e| Error::IoError(e.to_string()))?;
            if first {
                first = false;
                continue; // skip header
            }
            count += 1;
        }
        Ok(count)
    }

    // -----------------------------------------------------------------------
    // chunk iteration helpers
    // -----------------------------------------------------------------------

    /// Call `f` with each chunk in sequence. The chunks are `DataFrame` values
    /// of at most `config.chunk_size` rows.
    fn for_each_chunk<F>(&self, mut f: F) -> Result<()>
    where
        F: FnMut(DataFrame) -> Result<()>,
    {
        match self.format {
            DataFormat::Csv => self.for_each_csv_chunk(&mut f),
            DataFormat::Json => Err(Error::NotImplemented(
                "chunked iteration for JSON format".into(),
            )),
            DataFormat::Parquet => Err(Error::NotImplemented(
                "chunked iteration for Parquet format".into(),
            )),
        }
    }

    fn for_each_csv_chunk<F>(&self, f: &mut F) -> Result<()>
    where
        F: FnMut(DataFrame) -> Result<()>,
    {
        let file = File::open(&self.source_path).map_err(|e| Error::IoError(e.to_string()))?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        let headers: Vec<String> = rdr
            .headers()
            .map_err(|e| Error::CsvError(e.to_string()))?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let chunk_size = self.config.chunk_size;
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(chunk_size);

        let flush_chunk = |rows: &mut Vec<Vec<String>>, headers: &[String]| -> Result<DataFrame> {
            let mut df = DataFrame::new();
            let num_cols = headers.len();
            let mut col_data: Vec<Vec<String>> = vec![Vec::with_capacity(rows.len()); num_cols];
            for row in rows.iter() {
                for (ci, cell) in row.iter().enumerate() {
                    if ci < num_cols {
                        col_data[ci].push(cell.clone());
                    }
                }
            }
            for (ci, col_name) in headers.iter().enumerate() {
                let series = Series::new(col_data[ci].clone(), Some(col_name.clone()))
                    .map_err(|e| Error::Operation(e.to_string()))?;
                df.add_column(col_name.clone(), series)
                    .map_err(|e| Error::Operation(e.to_string()))?;
            }
            rows.clear();
            Ok(df)
        };

        for record_result in rdr.records() {
            let record = record_result.map_err(|e| Error::CsvError(e.to_string()))?;
            let row: Vec<String> = record.iter().map(|f| f.to_string()).collect();
            rows.push(row);
            if rows.len() >= chunk_size {
                let df = flush_chunk(&mut rows, &headers)?;
                f(df)?;
            }
        }

        // flush remaining rows
        if !rows.is_empty() {
            let df = flush_chunk(&mut rows, &headers)?;
            f(df)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // map
    // -----------------------------------------------------------------------

    /// Apply a transformation function to each chunk, writing intermediate
    /// results to temp files and returning an `OutOfCoreWriter`.
    pub fn map<F>(self, f: F) -> Result<OutOfCoreWriter>
    where
        F: Fn(DataFrame) -> Result<DataFrame> + Send + Sync,
    {
        // Collect all chunks into temp files first so rayon can parallelise
        let mut chunk_input_paths: Vec<PathBuf> = Vec::new();
        let mut chunk_index = 0usize;
        self.for_each_chunk(|chunk_df| {
            let path = write_chunk_file(
                &dataframe_to_rows(&chunk_df)?.1,
                &chunk_df.column_names(),
                &self.config.temp_dir,
                chunk_index,
            )?;
            chunk_input_paths.push(path);
            chunk_index += 1;
            Ok(())
        })?;

        // Process in parallel
        let config = self.config.clone();
        let results: Vec<Result<PathBuf>> = chunk_input_paths
            .par_iter()
            .enumerate()
            .map(|(i, input_path)| {
                let chunk_df = read_chunk_file(input_path)?;
                let transformed = f(chunk_df)?;
                let out_path = config.temp_dir.join(format!("pandrs_ooc_mapped_{}.csv", i));
                write_dataframe_csv(&transformed, &out_path)?;
                Ok(out_path)
            })
            .collect();

        let mut output_chunks: Vec<PathBuf> = Vec::with_capacity(results.len());
        for r in results {
            output_chunks.push(r?);
        }

        // Clean up input chunk files
        for p in &chunk_input_paths {
            let _ = std::fs::remove_file(p);
        }

        Ok(OutOfCoreWriter {
            chunks: output_chunks,
            config: self.config,
        })
    }

    // -----------------------------------------------------------------------
    // collect
    // -----------------------------------------------------------------------

    /// Load and concatenate all chunks into a single in-memory DataFrame.
    ///
    /// Only use this when the total data fits in RAM.
    pub fn collect(self) -> Result<DataFrame> {
        let mut all_dfs: Vec<DataFrame> = Vec::new();
        self.for_each_chunk(|chunk| {
            all_dfs.push(chunk);
            Ok(())
        })?;
        concat_dataframes(all_dfs)
    }

    // -----------------------------------------------------------------------
    // foreach
    // -----------------------------------------------------------------------

    /// Apply a function to each chunk without collecting results.
    pub fn foreach<F>(&self, f: F) -> Result<()>
    where
        F: Fn(DataFrame) -> Result<()> + Send + Sync,
    {
        self.for_each_chunk(f)
    }

    // -----------------------------------------------------------------------
    // aggregate
    // -----------------------------------------------------------------------

    /// Compute per-column aggregates across all chunks.
    ///
    /// Returns a one-row `DataFrame`.  The output column names are
    /// `"{column}_{op}"` (e.g. `"value_sum"`, `"id_count"`), so multiple
    /// aggregation operations on the same source column do not collide.
    ///
    /// `ops` is a slice of `(column_name, AggOp)` pairs.
    pub fn aggregate(&self, ops: &[(&str, AggOp)]) -> Result<DataFrame> {
        struct ColState {
            /// Source column name.
            source_col: String,
            /// Output column name, e.g. "value_sum".
            output_col: String,
            sum: f64,
            count: usize,
            min: Option<f64>,
            max: Option<f64>,
            op: AggOp,
        }

        let mut states: Vec<ColState> = ops
            .iter()
            .map(|(col, op)| {
                let op_str = match op {
                    AggOp::Sum => "sum",
                    AggOp::Mean => "mean",
                    AggOp::Min => "min",
                    AggOp::Max => "max",
                    AggOp::Count => "count",
                };
                ColState {
                    source_col: col.to_string(),
                    output_col: format!("{}_{}", col, op_str),
                    sum: 0.0,
                    count: 0,
                    min: None,
                    max: None,
                    op: *op,
                }
            })
            .collect();

        self.for_each_chunk(|chunk| {
            for state in states.iter_mut() {
                if !chunk.contains_column(&state.source_col) {
                    continue;
                }
                let row_count = chunk.row_count();
                for row_idx in 0..row_count {
                    let val_str = chunk
                        .get_string_value(&state.source_col, row_idx)
                        .unwrap_or("0");
                    if let Ok(val) = val_str.parse::<f64>() {
                        state.sum += val;
                        state.count += 1;
                        state.min = Some(state.min.map_or(val, |m: f64| m.min(val)));
                        state.max = Some(state.max.map_or(val, |m: f64| m.max(val)));
                    }
                }
            }
            Ok(())
        })?;

        // Build result DataFrame (one row per aggregate)
        let mut result_df = DataFrame::new();
        for state in &states {
            let value = match state.op {
                AggOp::Sum => state.sum,
                AggOp::Mean => {
                    if state.count > 0 {
                        state.sum / state.count as f64
                    } else {
                        f64::NAN
                    }
                }
                AggOp::Min => state.min.unwrap_or(f64::NAN),
                AggOp::Max => state.max.unwrap_or(f64::NAN),
                AggOp::Count => state.count as f64,
            };
            let series = Series::new(vec![value.to_string()], Some(state.output_col.clone()))
                .map_err(|e| Error::Operation(e.to_string()))?;
            result_df
                .add_column(state.output_col.clone(), series)
                .map_err(|e| Error::Operation(e.to_string()))?;
        }
        Ok(result_df)
    }
}

// ---------------------------------------------------------------------------
// OutOfCoreWriter
// ---------------------------------------------------------------------------

/// Holds a list of temporary chunk files resulting from an out-of-core
/// `map` or join operation.
pub struct OutOfCoreWriter {
    /// Paths to temporary chunk CSV files.
    pub(crate) chunks: Vec<PathBuf>,
    pub(crate) config: OutOfCoreConfig,
}

impl OutOfCoreWriter {
    /// Write all chunks to a single CSV output file.
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        let out_path = path.as_ref();
        let out_file = File::create(out_path).map_err(|e| Error::IoError(e.to_string()))?;
        let mut out = BufWriter::new(out_file);

        let mut header_written = false;
        for chunk_path in &self.chunks {
            let chunk_file = File::open(chunk_path).map_err(|e| Error::IoError(e.to_string()))?;
            let reader = BufReader::new(chunk_file);
            let mut lines = reader.lines();

            // First line is header
            if let Some(header_result) = lines.next() {
                let header = header_result.map_err(|e| Error::IoError(e.to_string()))?;
                if !header_written {
                    out.write_all(header.as_bytes())
                        .map_err(|e| Error::IoError(e.to_string()))?;
                    out.write_all(b"\n")
                        .map_err(|e| Error::IoError(e.to_string()))?;
                    header_written = true;
                }
            }

            for line_result in lines {
                let line = line_result.map_err(|e| Error::IoError(e.to_string()))?;
                if !line.trim().is_empty() {
                    out.write_all(line.as_bytes())
                        .map_err(|e| Error::IoError(e.to_string()))?;
                    out.write_all(b"\n")
                        .map_err(|e| Error::IoError(e.to_string()))?;
                }
            }
        }
        out.flush().map_err(|e| Error::IoError(e.to_string()))?;
        Ok(())
    }

    /// Write all chunks to a JSON file (records orientation).
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let df = self.collect()?;
        crate::io::json::write_json(&df, path, crate::io::json::JsonOrient::Records)
    }

    /// Merge all chunks into a single in-memory DataFrame.
    pub fn collect(&self) -> Result<DataFrame> {
        let mut all_dfs: Vec<DataFrame> = Vec::new();
        for chunk_path in &self.chunks {
            let df = read_chunk_file(chunk_path)?;
            all_dfs.push(df);
        }
        concat_dataframes(all_dfs)
    }
}

impl Drop for OutOfCoreWriter {
    fn drop(&mut self) {
        for p in &self.chunks {
            let _ = std::fs::remove_file(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: concatenate DataFrames
// ---------------------------------------------------------------------------

/// Concatenate a list of DataFrames vertically (same columns required).
pub(crate) fn concat_dataframes(dfs: Vec<DataFrame>) -> Result<DataFrame> {
    if dfs.is_empty() {
        return Ok(DataFrame::new());
    }

    let col_names = dfs[0].column_names();
    let total_rows: usize = dfs.iter().map(|df| df.row_count()).sum();

    // Build per-column data
    let mut col_data: HashMap<String, Vec<String>> = col_names
        .iter()
        .map(|n| (n.clone(), Vec::with_capacity(total_rows)))
        .collect();

    for df in &dfs {
        let row_count = df.row_count();
        for col in &col_names {
            for row_idx in 0..row_count {
                let val = df.get_string_value(col, row_idx).unwrap_or("").to_string();
                if let Some(vec) = col_data.get_mut(col) {
                    vec.push(val);
                }
            }
        }
    }

    let mut result = DataFrame::new();
    for col_name in &col_names {
        let values = col_data.remove(col_name).unwrap_or_default();
        let series = Series::new(values, Some(col_name.clone()))
            .map_err(|e| Error::Operation(e.to_string()))?;
        result
            .add_column(col_name.clone(), series)
            .map_err(|e| Error::Operation(e.to_string()))?;
    }

    Ok(result)
}
