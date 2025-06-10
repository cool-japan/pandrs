//! Module for handling large datasets
//!
//! This module provides functionality for working with datasets that are too large
//! to fit in memory. It implements disk-based processing techniques including:
//! - Memory-mapped files for efficient data access
//! - Chunked processing for large datasets
//! - Spill-to-disk operations when memory limits are reached

use memmap2::{Mmap, MmapMut, MmapOptions};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

use crate::dataframe::DataFrame;
use crate::error::{Error, PandRSError, Result};
use crate::optimized::dataframe::OptimizedDataFrame;

/// Configuration for disk-based processing
#[derive(Debug, Clone)]
pub struct DiskConfig {
    /// Maximum memory usage in bytes before spilling to disk
    pub memory_limit: usize,
    /// Temporary directory for spilled data
    pub temp_dir: Option<PathBuf>,
    /// Chunk size for processing large datasets
    pub chunk_size: usize,
    /// Whether to use memory mapping for file access
    pub use_memory_mapping: bool,
}

impl Default for DiskConfig {
    fn default() -> Self {
        DiskConfig {
            // Default to 1GB memory limit
            memory_limit: 1024 * 1024 * 1024,
            // Default to system temp directory
            temp_dir: None,
            // Default to 100,000 rows per chunk
            chunk_size: 100_000,
            // Use memory mapping by default
            use_memory_mapping: true,
        }
    }
}

/// MemoryTracker keeps track of memory usage to determine when to spill to disk
#[derive(Debug)]
struct MemoryTracker {
    /// Current memory usage in bytes
    current_usage: usize,
    /// Maximum allowed memory usage in bytes
    limit: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker with specified limit
    fn new(limit: usize) -> Self {
        MemoryTracker {
            current_usage: 0,
            limit,
        }
    }

    /// Register memory allocation
    fn allocate(&mut self, bytes: usize) -> bool {
        let new_usage = self.current_usage + bytes;
        if new_usage <= self.limit {
            self.current_usage = new_usage;
            true
        } else {
            false
        }
    }

    /// Register memory deallocation
    fn deallocate(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_sub(bytes);
    }

    /// Get current memory usage
    fn usage(&self) -> usize {
        self.current_usage
    }

    /// Check if memory limit is reached
    fn is_limit_reached(&self) -> bool {
        self.current_usage >= self.limit
    }
}

/// ChunkedDataFrame processes a large DataFrame in manageable chunks
#[derive(Debug)]
pub struct ChunkedDataFrame {
    /// Path to the data source
    source_path: PathBuf,
    /// Configuration for chunked processing
    config: DiskConfig,
    /// Current chunk being processed
    current_chunk: Option<DataFrame>,
    /// Current chunk index
    chunk_index: usize,
    /// Total number of chunks
    total_chunks: Option<usize>,
    /// Temporary directory for intermediate results
    temp_dir: tempfile::TempDir,
    /// Memory tracker
    memory_tracker: MemoryTracker,
}

impl ChunkedDataFrame {
    /// Create a new chunked DataFrame from a file path
    pub fn new<P: AsRef<Path>>(path: P, config: Option<DiskConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let source_path = path.as_ref().to_path_buf();

        // Ensure the file exists
        if !source_path.exists() {
            return Err(Error::IoError(format!("File not found: {:?}", source_path)));
        }

        // Create temporary directory
        let temp_dir = match &config.temp_dir {
            Some(dir) => tempdir_in(dir)?,
            None => tempdir()?,
        };

        let memory_tracker = MemoryTracker::new(config.memory_limit);

        Ok(ChunkedDataFrame {
            source_path,
            config,
            current_chunk: None,
            chunk_index: 0,
            total_chunks: None,
            temp_dir,
            memory_tracker,
        })
    }

    /// Load the next chunk of data
    pub fn next_chunk(&mut self) -> Result<Option<&DataFrame>> {
        // Calculate total chunks if not already calculated
        if self.total_chunks.is_none() {
            self.calculate_total_chunks()?;
        }

        // Check if we've reached the end
        if let Some(total) = self.total_chunks {
            if self.chunk_index >= total {
                return Ok(None);
            }
        }

        // Load the chunk
        if self.config.use_memory_mapping {
            self.load_chunk_mmap()?;
        } else {
            self.load_chunk_standard()?;
        }

        self.chunk_index += 1;
        Ok(self.current_chunk.as_ref())
    }

    /// Calculate the total number of chunks in the dataset
    fn calculate_total_chunks(&mut self) -> Result<()> {
        // This is a rough estimate based on file size and chunk size
        let file = File::open(&self.source_path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;

        // Estimate bytes per row (assuming CSV for now)
        let sample_size = file_size.min(1024 * 1024); // Read at most 1MB for estimation
        let mut buffer = vec![0; sample_size];
        let mut file = File::open(&self.source_path)?;
        let bytes_read = file.read(&mut buffer)?;
        buffer.truncate(bytes_read);

        // Count newlines to estimate rows in the sample
        let newlines = buffer.iter().filter(|&&b| b == b'\n').count();
        if newlines == 0 {
            return Err(Error::Consistency(
                "Could not determine row count in file".into(),
            ));
        }

        let bytes_per_row = sample_size / newlines;
        let estimated_rows = file_size / bytes_per_row;

        // Calculate total chunks
        let total_chunks = (estimated_rows + self.config.chunk_size - 1) / self.config.chunk_size;
        self.total_chunks = Some(total_chunks);

        Ok(())
    }

    /// Load a chunk using memory mapping
    fn load_chunk_mmap(&mut self) -> Result<()> {
        let file = File::open(&self.source_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Find the start and end positions for this chunk
        let (start_pos, end_pos) = self.find_chunk_boundaries(&mmap)?;

        // Extract chunk data
        let chunk_data = if end_pos > start_pos {
            &mmap[start_pos..end_pos]
        } else {
            &[]
        };

        // Parse the chunk data (assuming CSV for now)
        let mut reader = csv::Reader::from_reader(chunk_data);
        let df = DataFrame::from_csv_reader(&mut reader, true)?;

        // Track memory usage
        let estimated_memory = estimate_dataframe_memory(&df);
        if !self.memory_tracker.allocate(estimated_memory) {
            // Spill previous chunk to disk if needed
            if let Some(prev_chunk) = self.current_chunk.take() {
                self.spill_to_disk(prev_chunk)?;
                // Now we should have enough memory
                self.memory_tracker.allocate(estimated_memory);
            }
        }

        self.current_chunk = Some(df);
        Ok(())
    }

    /// Load a chunk using standard file I/O
    fn load_chunk_standard(&mut self) -> Result<()> {
        let file = File::open(&self.source_path)?;
        let mut reader = io::BufReader::new(file);

        // Skip to the start of this chunk
        self.skip_to_chunk(&mut reader)?;

        // Read the appropriate number of lines
        let mut lines = Vec::new();
        let mut line_count = 0;
        let mut line = String::new();

        while line_count < self.config.chunk_size {
            line.clear();
            let bytes = reader.read_line(&mut line)?;
            if bytes == 0 {
                break; // End of file
            }

            lines.push(line.clone());
            line_count += 1;
        }

        // Parse the chunk data (assuming CSV for now)
        let csv_data = lines.join("");
        let mut csv_reader = csv::Reader::from_reader(csv_data.as_bytes());
        let df = DataFrame::from_csv_reader(&mut csv_reader, true)?;

        // Track memory usage
        let estimated_memory = estimate_dataframe_memory(&df);
        if !self.memory_tracker.allocate(estimated_memory) {
            // Spill previous chunk to disk if needed
            if let Some(prev_chunk) = self.current_chunk.take() {
                self.spill_to_disk(prev_chunk)?;
                // Now we should have enough memory
                self.memory_tracker.allocate(estimated_memory);
            }
        }

        self.current_chunk = Some(df);
        Ok(())
    }

    /// Find the start and end positions of the current chunk in the memory-mapped file
    fn find_chunk_boundaries(&self, mmap: &Mmap) -> Result<(usize, usize)> {
        let file_size = mmap.len();

        // Skip header line for first chunk
        let start_offset = if self.chunk_index == 0 {
            // Find the first newline
            mmap.iter().position(|&b| b == b'\n').unwrap_or(0) + 1
        } else {
            0
        };

        // Find the position to start from based on chunk index
        let mut start_pos = start_offset;
        let mut lines_skipped = 0;
        let lines_to_skip = self.chunk_index * self.config.chunk_size;

        for i in start_offset..file_size {
            if mmap[i] == b'\n' {
                lines_skipped += 1;
                if lines_skipped == lines_to_skip {
                    start_pos = i + 1;
                    break;
                }
            }
        }

        // Find the end position for this chunk
        let mut end_pos = start_pos;
        let mut lines_read = 0;

        for i in start_pos..file_size {
            if mmap[i] == b'\n' {
                lines_read += 1;
                end_pos = i + 1;
                if lines_read == self.config.chunk_size {
                    break;
                }
            }
        }

        Ok((start_pos, end_pos))
    }

    /// Skip to the start of the current chunk using a reader
    fn skip_to_chunk<R: Read + BufRead>(&self, reader: &mut R) -> Result<()> {
        let mut line = String::new();

        // Skip header for first chunk
        if self.chunk_index == 0 {
            reader.read_line(&mut line)?;
        }

        // Skip lines to get to the current chunk
        for _ in 0..(self.chunk_index * self.config.chunk_size) {
            line.clear();
            let bytes = reader.read_line(&mut line)?;
            if bytes == 0 {
                return Err(Error::IndexOutOfBoundsStr(
                    "Chunk index exceeds file size".into(),
                ));
            }
        }

        Ok(())
    }

    /// Spill a DataFrame to disk to free up memory
    fn spill_to_disk(&mut self, df: DataFrame) -> Result<()> {
        // Create a temp file in the temp directory
        let file_path = self
            .temp_dir
            .path()
            .join(format!("chunk_{}.csv", self.chunk_index));

        // Save DataFrame to the temp file
        df.to_csv(&file_path)?;

        // Free memory
        let estimated_memory = estimate_dataframe_memory(&df);
        self.memory_tracker.deallocate(estimated_memory);

        Ok(())
    }

    /// Process the entire dataset with a function that operates on chunks
    pub fn process_with<F, T>(&mut self, mut func: F) -> Result<Vec<T>>
    where
        F: FnMut(&DataFrame) -> Result<T>,
    {
        let mut results = Vec::new();

        while let Some(chunk) = self.next_chunk()? {
            let result = func(chunk)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Apply a parallel operation to each chunk and combine the results
    pub fn parallel_process<F, T, C>(&mut self, chunk_func: F, combiner: C) -> Result<T>
    where
        F: Fn(&DataFrame) -> Result<T> + Send + Sync,
        T: Send + 'static,
        C: FnOnce(Vec<T>) -> Result<T>,
    {
        use rayon::prelude::*;

        // First load all chunks and spill to disk if needed
        let mut chunk_paths = Vec::new();

        while let Some(_) = self.next_chunk()? {
            // If we're using in-memory processing and the chunk hasn't been spilled,
            // spill it now so we can process in parallel
            if let Some(chunk) = self.current_chunk.take() {
                let file_path = self
                    .temp_dir
                    .path()
                    .join(format!("chunk_{}.csv", self.chunk_index - 1));
                chunk.to_csv(&file_path)?;
                chunk_paths.push(file_path);

                // Free memory
                let estimated_memory = estimate_dataframe_memory(&chunk);
                self.memory_tracker.deallocate(estimated_memory);
            }
        }

        // Process chunks in parallel
        let results: Vec<Result<T>> = chunk_paths
            .par_iter()
            .map(|path| {
                let df = DataFrame::from_csv(path, true)?;
                chunk_func(&df)
            })
            .collect();

        // Check for errors
        let mut unwrapped_results = Vec::new();
        for result in results {
            unwrapped_results.push(result?);
        }

        // Combine results
        combiner(unwrapped_results)
    }
}

/// Estimate memory usage of a DataFrame (rough approximation)
fn estimate_dataframe_memory(df: &DataFrame) -> usize {
    let row_count = df.row_count();
    let col_count = df.column_count();

    // Rough estimate: 8 bytes per numeric cell, 24 bytes per string cell on average
    // Add overhead for data structures
    let avg_bytes_per_cell = 16;
    let base_overhead = 1000;

    base_overhead + (row_count * col_count * avg_bytes_per_cell)
}

/// Create a temporary directory inside another directory
fn tempdir_in<P: AsRef<Path>>(dir: P) -> io::Result<tempfile::TempDir> {
    tempfile::Builder::new().prefix("pandrs_").tempdir_in(dir)
}

/// DiskBasedDataFrame provides an interface similar to DataFrame but uses disk storage
#[derive(Debug)]
pub struct DiskBasedDataFrame {
    /// Path to the data source
    source_path: PathBuf,
    /// Configuration
    config: DiskConfig,
    /// Schema information
    schema: DataFrame,
    /// Memory-mapped file if being used
    mmap: Option<Mmap>,
    /// Temporary directory for spilled data
    temp_dir: tempfile::TempDir,
    /// Memory tracker
    memory_tracker: Arc<Mutex<MemoryTracker>>,
}

impl DiskBasedDataFrame {
    /// Create a new disk-based DataFrame from a file path
    pub fn new<P: AsRef<Path>>(path: P, config: Option<DiskConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let source_path = path.as_ref().to_path_buf();

        // Ensure the file exists
        if !source_path.exists() {
            return Err(Error::IoError(format!("File not found: {:?}", source_path)));
        }

        // Create temporary directory
        let temp_dir = match &config.temp_dir {
            Some(dir) => tempdir_in(dir)?,
            None => tempdir()?,
        };

        // Read just the header to get schema information
        let file = File::open(&source_path)?;
        let mut reader = csv::Reader::from_reader(io::BufReader::new(file));
        let headers = reader.headers()?.clone();

        // Create a sample DataFrame with the schema
        let mut schema = DataFrame::new();
        for header in headers.iter() {
            schema.add_column(
                header.to_string(),
                crate::series::Series::new(Vec::<String>::new(), Some(header.to_string()))?,
            )?;
        }

        let memory_tracker = Arc::new(Mutex::new(MemoryTracker::new(config.memory_limit)));

        // Set up memory mapping if enabled
        let mmap = if config.use_memory_mapping {
            let file = File::open(&source_path)?;
            Some(unsafe { MmapOptions::new().map(&file)? })
        } else {
            None
        };

        Ok(DiskBasedDataFrame {
            source_path,
            config,
            schema,
            mmap,
            temp_dir,
            memory_tracker,
        })
    }

    /// Get the schema (column structure) of the DataFrame
    pub fn schema(&self) -> &DataFrame {
        &self.schema
    }

    /// Get a chunked view of this DataFrame for efficient processing
    pub fn chunked(&self) -> Result<ChunkedDataFrame> {
        ChunkedDataFrame::new(&self.source_path, Some(self.config.clone()))
    }

    /// Apply a function to the entire DataFrame by processing it in chunks
    pub fn apply<F, T>(&self, function: F) -> Result<T>
    where
        F: FnMut(&DataFrame) -> Result<T>,
        T: Send + 'static,
    {
        let mut chunked = self.chunked()?;

        // Define a combiner function that keeps only the last result
        // This is for operations that don't need to combine results
        let results = chunked.process_with(function)?;

        if results.is_empty() {
            return Err(Error::EmptyDataFrame("No data to process".into()));
        }

        Ok(results.into_iter().last().unwrap())
    }

    /// Apply an aggregation function that combines results from all chunks
    pub fn aggregate<F, C, T>(&self, chunk_func: F, combiner: C) -> Result<T>
    where
        F: Fn(&DataFrame) -> Result<T> + Send + Sync,
        T: Send + 'static,
        C: FnOnce(Vec<T>) -> Result<T>,
    {
        let mut chunked = self.chunked()?;
        chunked.parallel_process(chunk_func, combiner)
    }
}

/// Operations available on both in-memory and disk-based DataFrames
pub trait DataFrameOperations {
    /// Apply a filter to the DataFrame
    fn filter(
        &self,
        condition: impl Fn(&str, usize) -> bool + Send + Sync,
    ) -> Result<Vec<HashMap<String, String>>>;

    /// Select specific columns
    fn select(&self, columns: &[&str]) -> Result<Vec<HashMap<String, String>>>;

    /// Apply a transformation to each value
    fn transform(
        &self,
        transformation: impl Fn(&str, &str, usize) -> Result<String> + Send + Sync,
    ) -> Result<Vec<HashMap<String, String>>>;

    /// Group by a column and aggregate
    fn group_by(
        &self,
        group_column: &str,
        agg_column: &str,
        agg_func: impl Fn(Vec<String>) -> Result<String> + Send + Sync,
    ) -> Result<HashMap<String, Vec<String>>>;
}

impl DataFrameOperations for DiskBasedDataFrame {
    fn filter(
        &self,
        condition: impl Fn(&str, usize) -> bool + Send + Sync,
    ) -> Result<Vec<HashMap<String, String>>> {
        self.aggregate(
            // Process each chunk
            |chunk| {
                // Apply filter to this chunk
                let mut result_rows = Vec::new();

                for row_idx in 0..chunk.row_count() {
                    let mut keep_row = true;

                    for col_name in chunk.column_names() {
                        let value = chunk.get_string_value(&col_name, row_idx)?;
                        if !condition(value, row_idx) {
                            keep_row = false;
                            break;
                        }
                    }

                    if keep_row {
                        let mut row = HashMap::new();
                        for col_name in chunk.column_names() {
                            let value = chunk.get_string_value(&col_name, row_idx)?;
                            row.insert(col_name.clone(), value.to_string());
                        }
                        result_rows.push(row);
                    }
                }

                Ok(result_rows)
            },
            // Combine results from all chunks
            |all_rows: Vec<Vec<HashMap<String, String>>>| {
                let mut combined_rows = Vec::new();

                for chunk_rows in all_rows {
                    combined_rows.extend(chunk_rows);
                }

                Ok(combined_rows)
            },
        )
    }

    fn select(&self, columns: &[&str]) -> Result<Vec<HashMap<String, String>>> {
        // Validate that all columns exist in the schema
        for &col in columns {
            if !self.schema.contains_column(col) {
                return Err(Error::Column(format!("Column '{}' does not exist", col)));
            }
        }

        self.aggregate(
            // Process each chunk
            |chunk| {
                let mut selected_rows = Vec::new();

                for row_idx in 0..chunk.row_count() {
                    let mut row = HashMap::new();
                    for &col in columns {
                        if chunk.contains_column(col) {
                            let value = chunk.get_string_value(&col, row_idx)?;
                            row.insert(col.to_string(), value.to_string());
                        }
                    }
                    selected_rows.push(row);
                }

                Ok(selected_rows)
            },
            // Combine results from all chunks
            |all_rows: Vec<Vec<HashMap<String, String>>>| {
                let mut combined_rows = Vec::new();

                for chunk_rows in all_rows {
                    combined_rows.extend(chunk_rows);
                }

                Ok(combined_rows)
            },
        )
    }

    fn transform(
        &self,
        transformation: impl Fn(&str, &str, usize) -> Result<String> + Send + Sync,
    ) -> Result<Vec<HashMap<String, String>>> {
        self.aggregate(
            // Process each chunk
            |chunk| {
                let mut transformed_rows = Vec::new();

                for row_idx in 0..chunk.row_count() {
                    let mut row = HashMap::new();

                    for col_name in chunk.column_names() {
                        let value = chunk.get_string_value(&col_name, row_idx)?;
                        let new_value = transformation(&col_name, value, row_idx)?;
                        row.insert(col_name.clone(), new_value);
                    }

                    transformed_rows.push(row);
                }

                Ok(transformed_rows)
            },
            // Combine results from all chunks
            |all_rows: Vec<Vec<HashMap<String, String>>>| {
                let mut combined_rows = Vec::new();

                for chunk_rows in all_rows {
                    combined_rows.extend(chunk_rows);
                }

                Ok(combined_rows)
            },
        )
    }

    fn group_by(
        &self,
        group_column: &str,
        agg_column: &str,
        agg_func: impl Fn(Vec<String>) -> Result<String> + Send + Sync,
    ) -> Result<HashMap<String, Vec<String>>> {
        if !self.schema.contains_column(group_column) {
            return Err(Error::Column(format!(
                "Group column '{}' does not exist",
                group_column
            )));
        }

        if !self.schema.contains_column(agg_column) {
            return Err(Error::Column(format!(
                "Aggregation column '{}' does not exist",
                agg_column
            )));
        }

        self.aggregate(
            // Process each chunk
            |chunk| {
                let mut grouped_data: HashMap<String, Vec<String>> = HashMap::new();

                for row_idx in 0..chunk.row_count() {
                    let group_value = chunk.get_string_value(group_column, row_idx)?;
                    let agg_value = chunk.get_string_value(agg_column, row_idx)?;

                    grouped_data
                        .entry(group_value.to_string())
                        .or_insert_with(Vec::new)
                        .push(agg_value.to_string());
                }

                Ok(grouped_data)
            },
            // Combine results by merging the HashMaps
            |chunk_maps| {
                let mut result_map: HashMap<String, Vec<String>> = HashMap::new();

                for chunk_map in chunk_maps {
                    for (key, values) in chunk_map {
                        result_map
                            .entry(key)
                            .or_insert_with(Vec::new)
                            .extend(values);
                    }
                }

                // Return the result map directly
                Ok(result_map)
            },
        )
    }
}

/// Optimized disk-based processing with the OptimizedDataFrame structure
#[derive(Debug)]
pub struct DiskBasedOptimizedDataFrame {
    /// Internal representation
    inner: DiskBasedDataFrame,
}

impl DiskBasedOptimizedDataFrame {
    /// Create a new optimized disk-based DataFrame from a file path
    pub fn new<P: AsRef<Path>>(path: P, config: Option<DiskConfig>) -> Result<Self> {
        Ok(DiskBasedOptimizedDataFrame {
            inner: DiskBasedDataFrame::new(path, config)?,
        })
    }

    /// Convert to an in-memory OptimizedDataFrame
    pub fn to_optimized_dataframe(&self) -> Result<OptimizedDataFrame> {
        self.inner.aggregate(
            // Convert each chunk to an OptimizedDataFrame
            |chunk| Ok(OptimizedDataFrame::from_dataframe(chunk)?),
            // Combine by concatenating rows
            |chunk_results| {
                if chunk_results.is_empty() {
                    return Ok(OptimizedDataFrame::new());
                }

                let mut result = chunk_results[0].clone();

                for chunk in chunk_results.iter().skip(1) {
                    result = result.concat_rows(chunk)?;
                }

                Ok(result)
            },
        )
    }

    /// Apply an aggregation function to the entire dataset
    pub fn aggregate<F, C, T>(&self, chunk_func: F, combiner: C) -> Result<T>
    where
        F: Fn(&OptimizedDataFrame) -> Result<T> + Send + Sync,
        T: Send + 'static,
        C: FnOnce(Vec<T>) -> Result<T>,
    {
        self.inner.aggregate(
            // Convert each chunk to OptimizedDataFrame and apply function
            |chunk| {
                let opt_chunk = OptimizedDataFrame::from_dataframe(chunk)?;
                chunk_func(&opt_chunk)
            },
            combiner,
        )
    }
}
