use crate::core::error::{Error, Result};
use crate::storage::traits::{
    AccessPattern, DataChunk, Efficiency, PerformanceProfile, Speed, StorageConfig, StorageEngine,
    StorageStatistics,
};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, RwLock};

/// Compression strategies for columnar data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    RunLength,
    Dictionary,
    BitPacked,
}

/// Storage metadata for a column
#[derive(Debug, Clone)]
pub struct ColumnMetadata {
    pub name: String,
    pub data_type: String,
    pub row_count: usize,
    pub compression: CompressionType,
    pub null_count: usize,
    pub size_bytes: usize,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
}

/// Compressed column data storage
#[derive(Debug, Clone)]
pub enum CompressedColumnData {
    /// Uncompressed raw data
    Raw(Vec<u8>),
    /// Run-length encoded data (value, count) pairs
    RunLength(Vec<(Vec<u8>, usize)>),
    /// Dictionary encoded data (dictionary, indices)
    Dictionary {
        dictionary: Vec<Vec<u8>>,
        indices: Vec<u32>,
    },
    /// Bit-packed data for low cardinality integers
    BitPacked { data: Vec<u8>, bits_per_value: u8 },
}

impl CompressedColumnData {
    /// Get the approximate size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            CompressedColumnData::Raw(data) => data.len(),
            CompressedColumnData::RunLength(runs) => {
                runs.iter().map(|(value, _)| value.len() + 8).sum()
            }
            CompressedColumnData::Dictionary {
                dictionary,
                indices,
            } => dictionary.iter().map(|v| v.len()).sum::<usize>() + indices.len() * 4,
            CompressedColumnData::BitPacked { data, .. } => data.len(),
        }
    }

    /// Decompress the data to raw bytes
    pub fn decompress(&self) -> Vec<u8> {
        match self {
            CompressedColumnData::Raw(data) => data.clone(),
            CompressedColumnData::RunLength(runs) => {
                let mut result = Vec::new();
                for (value, count) in runs {
                    for _ in 0..*count {
                        result.extend_from_slice(value);
                    }
                }
                result
            }
            CompressedColumnData::Dictionary {
                dictionary,
                indices,
            } => {
                let mut result = Vec::new();
                for &index in indices {
                    if let Some(value) = dictionary.get(index as usize) {
                        result.extend_from_slice(value);
                    }
                }
                result
            }
            CompressedColumnData::BitPacked {
                data,
                bits_per_value,
            } => {
                // Simple decompression for demonstration
                // In production, this would be more sophisticated
                data.clone()
            }
        }
    }
}

/// A column-oriented storage engine for data
#[derive(Debug)]
pub struct ColumnStore {
    /// Stored columns indexed by name
    columns: Arc<RwLock<HashMap<String, CompressedColumnData>>>,
    /// Metadata for each column
    metadata: Arc<RwLock<HashMap<String, ColumnMetadata>>>,
    /// Total number of rows across all columns
    row_count: Arc<RwLock<usize>>,
    /// Storage statistics
    stats: Arc<RwLock<StorageStats>>,
}

/// Storage statistics for performance monitoring
#[derive(Debug, Default, Clone)]
pub struct StorageStats {
    pub total_columns: usize,
    pub total_size_bytes: usize,
    pub total_rows: usize,
    pub compression_ratio: f64,
    pub read_operations: usize,
    pub write_operations: usize,
}

impl ColumnStore {
    /// Creates a new column store
    pub fn new() -> Self {
        Self {
            columns: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            row_count: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(StorageStats::default())),
        }
    }

    /// Add a column to the store with automatic compression selection
    pub fn add_column<T: AsRef<[u8]>>(
        &self,
        name: String,
        data: &[T],
        data_type: String,
    ) -> Result<()> {
        if data.is_empty() {
            return Err(Error::InvalidInput("Cannot add empty column".into()));
        }

        // Choose optimal compression strategy
        let compression = self.select_compression_strategy(data);

        // Compress the data
        let compressed_data = self.compress_data(data, compression)?;

        // Calculate metadata
        let metadata = ColumnMetadata {
            name: name.clone(),
            data_type,
            row_count: data.len(),
            compression,
            null_count: 0, // Would be calculated based on actual null detection
            size_bytes: compressed_data.size_bytes(),
            min_value: None, // Would be calculated for numeric types
            max_value: None, // Would be calculated for numeric types
        };

        // Store the column and metadata
        {
            let mut columns = self.columns.write().unwrap();
            let mut metadata_map = self.metadata.write().unwrap();
            let mut stats = self.stats.write().unwrap();

            let size_bytes = metadata.size_bytes; // Extract before moving
            columns.insert(name.clone(), compressed_data);
            metadata_map.insert(name, metadata);

            // Update statistics
            stats.total_columns += 1;
            stats.total_size_bytes += size_bytes;
            stats.write_operations += 1;
        }

        // Update row count
        {
            let mut row_count = self.row_count.write().unwrap();
            if *row_count == 0 {
                *row_count = data.len();
            } else if *row_count != data.len() {
                return Err(Error::DimensionMismatch(
                    "Column length doesn't match existing row count".into(),
                ));
            }
        }

        Ok(())
    }

    /// Get a column from the store
    pub fn get_column(&self, name: &str) -> Result<Vec<u8>> {
        let columns = self.columns.read().unwrap();
        let mut stats = self.stats.write().unwrap();

        stats.read_operations += 1;

        match columns.get(name) {
            Some(compressed_data) => Ok(compressed_data.decompress()),
            None => Err(Error::ColumnNotFound(name.to_string())),
        }
    }

    /// Get column metadata
    pub fn get_metadata(&self, name: &str) -> Result<ColumnMetadata> {
        let metadata = self.metadata.read().unwrap();
        match metadata.get(name) {
            Some(meta) => Ok(meta.clone()),
            None => Err(Error::ColumnNotFound(name.to_string())),
        }
    }

    /// List all column names
    pub fn column_names(&self) -> Vec<String> {
        let columns = self.columns.read().unwrap();
        columns.keys().cloned().collect()
    }

    /// Get the number of rows
    pub fn row_count(&self) -> usize {
        *self.row_count.read().unwrap()
    }

    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        let stats = self.stats.read().unwrap();
        (*stats).clone()
    }

    /// Remove a column from the store
    pub fn remove_column(&self, name: &str) -> Result<()> {
        let mut columns = self.columns.write().unwrap();
        let mut metadata_map = self.metadata.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        if let Some(compressed_data) = columns.remove(name) {
            metadata_map.remove(name);
            stats.total_columns -= 1;
            stats.total_size_bytes -= compressed_data.size_bytes();
            Ok(())
        } else {
            Err(Error::ColumnNotFound(name.to_string()))
        }
    }

    /// Optimize storage by recompressing all columns
    pub fn optimize(&self) -> Result<()> {
        let column_names: Vec<String> = self.column_names();

        for name in column_names {
            let data = self.get_column(&name)?;
            let metadata = self.get_metadata(&name)?;

            // Remove and re-add with potentially better compression
            self.remove_column(&name)?;

            // For now, just re-add the raw data as a single chunk
            // In a production system, this would intelligently reconstruct the original format
            let single_chunk = vec![data.as_slice()];
            self.add_column(name, &single_chunk, metadata.data_type)?;
        }

        Ok(())
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f64 {
        let columns = self.columns.read().unwrap();
        if columns.is_empty() {
            return 1.0;
        }

        let compressed_size: usize = columns.values().map(|data| data.size_bytes()).sum();

        let uncompressed_size: usize = columns.values().map(|data| data.decompress().len()).sum();

        if compressed_size == 0 {
            1.0
        } else {
            uncompressed_size as f64 / compressed_size as f64
        }
    }

    // Private helper methods

    fn select_compression_strategy<T: AsRef<[u8]>>(&self, data: &[T]) -> CompressionType {
        if data.len() < 10 {
            return CompressionType::None;
        }

        // Check for run-length encoding potential
        let mut consecutive_count = 1;
        let mut max_consecutive = 1;

        for i in 1..data.len() {
            if data[i].as_ref() == data[i - 1].as_ref() {
                consecutive_count += 1;
                max_consecutive = max_consecutive.max(consecutive_count);
            } else {
                consecutive_count = 1;
            }
        }

        // If we have long runs, use run-length encoding
        if max_consecutive > data.len() / 4 {
            return CompressionType::RunLength;
        }

        // Check for dictionary encoding potential
        let unique_count = {
            let mut unique = std::collections::HashSet::new();
            for item in data {
                unique.insert(item.as_ref());
                if unique.len() > data.len() / 2 {
                    break; // Too many unique values for dictionary encoding
                }
            }
            unique.len()
        };

        if unique_count < data.len() / 4 {
            CompressionType::Dictionary
        } else {
            CompressionType::None
        }
    }

    fn compress_data<T: AsRef<[u8]>>(
        &self,
        data: &[T],
        compression: CompressionType,
    ) -> Result<CompressedColumnData> {
        match compression {
            CompressionType::None => {
                let mut raw_data = Vec::new();
                for item in data {
                    raw_data.extend_from_slice(item.as_ref());
                }
                Ok(CompressedColumnData::Raw(raw_data))
            }
            CompressionType::RunLength => {
                let mut runs = Vec::new();
                if !data.is_empty() {
                    let mut current_value = data[0].as_ref().to_vec();
                    let mut count = 1;

                    for item in data.iter().skip(1) {
                        if item.as_ref() == current_value {
                            count += 1;
                        } else {
                            runs.push((current_value, count));
                            current_value = item.as_ref().to_vec();
                            count = 1;
                        }
                    }
                    runs.push((current_value, count));
                }
                Ok(CompressedColumnData::RunLength(runs))
            }
            CompressionType::Dictionary => {
                let mut dictionary = Vec::new();
                let mut value_to_index = HashMap::new();
                let mut indices = Vec::new();

                for item in data {
                    let bytes = item.as_ref().to_vec();
                    if let Some(&index) = value_to_index.get(&bytes) {
                        indices.push(index);
                    } else {
                        let index = dictionary.len() as u32;
                        dictionary.push(bytes.clone());
                        value_to_index.insert(bytes, index);
                        indices.push(index);
                    }
                }

                Ok(CompressedColumnData::Dictionary {
                    dictionary,
                    indices,
                })
            }
            CompressionType::BitPacked => {
                // Simplified bit-packing implementation
                // In production, this would be more sophisticated
                let mut packed_data = Vec::new();
                for item in data {
                    packed_data.extend_from_slice(item.as_ref());
                }
                Ok(CompressedColumnData::BitPacked {
                    data: packed_data,
                    bits_per_value: 8, // Simplified
                })
            }
        }
    }
}

impl Default for ColumnStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for column store operations
#[derive(Debug, Clone)]
pub struct ColumnStoreHandle {
    /// Unique identifier for the storage instance
    pub id: usize,
    /// Reference to the column store
    pub store: Arc<ColumnStore>,
}

impl ColumnStoreHandle {
    /// Create a new handle
    pub fn new(id: usize, store: Arc<ColumnStore>) -> Self {
        Self { id, store }
    }
}

impl StorageEngine for ColumnStore {
    type Handle = ColumnStoreHandle;
    type Error = Error;

    fn create_storage(&mut self, _config: &StorageConfig) -> Result<Self::Handle> {
        // Create a new handle for this storage instance
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        Ok(ColumnStoreHandle::new(id, Arc::new(ColumnStore::new())))
    }

    fn read_chunk(&self, handle: &Self::Handle, range: Range<usize>) -> Result<DataChunk> {
        // For column store, we'll concatenate all column data within the range
        let columns = handle.store.columns.read().unwrap();
        let mut chunk_data = Vec::new();
        let mut total_rows = 0;

        for (_name, compressed_data) in columns.iter() {
            let decompressed = compressed_data.decompress();
            // Apply range if applicable
            let start = range.start.min(decompressed.len());
            let end = range.end.min(decompressed.len());
            if start < end {
                chunk_data.extend_from_slice(&decompressed[start..end]);
                total_rows = (end - start) / 8; // Assume 8 bytes per value for simplicity
            }
        }

        let metadata = crate::storage::traits::ChunkMetadata {
            row_count: total_rows,
            column_count: columns.len(),
            compression: crate::storage::traits::CompressionPreference::Auto,
            uncompressed_size: chunk_data.len(),
            compressed_size: chunk_data.len(), // Would be different with compression
        };

        Ok(DataChunk::new(chunk_data, metadata))
    }

    fn write_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        // For simplicity, we'll add this as a new column with a generated name
        let column_name = format!("chunk_{}", chunk.metadata.row_count);

        // Convert chunk data to the format expected by add_column
        let data: Vec<Vec<u8>> = chunk.data
            .chunks(8) // Assume 8 bytes per value
            .map(|chunk| chunk.to_vec())
            .collect();

        handle
            .store
            .add_column(column_name, &data, "bytes".to_string())?;
        Ok(())
    }

    fn append_chunk(&mut self, handle: &Self::Handle, chunk: DataChunk) -> Result<()> {
        // For column store, append is similar to write for new data
        self.write_chunk(handle, chunk)
    }

    fn flush(&mut self, _handle: &Self::Handle) -> Result<()> {
        // Column store is in-memory, so flush is a no-op
        Ok(())
    }

    fn delete_storage(&mut self, _handle: &Self::Handle) -> Result<()> {
        // For in-memory storage, this would clear the data
        // Implementation would depend on handle management
        Ok(())
    }

    fn performance_profile(&self) -> PerformanceProfile {
        PerformanceProfile {
            read_speed: Speed::Fast,
            write_speed: Speed::Medium,
            memory_efficiency: Efficiency::Good,
            compression_ratio: 0.7, // Estimate based on compression strategies
            random_access_speed: Speed::Fast,
            sequential_access_speed: Speed::VeryFast,
        }
    }

    fn storage_stats(&self, handle: &Self::Handle) -> Result<StorageStatistics> {
        let stats = handle.store.stats();
        Ok(StorageStatistics {
            total_size: stats.total_size_bytes,
            chunk_count: stats.total_columns,
            avg_compression_ratio: stats.compression_ratio,
            read_operations: stats.read_operations as u64,
            write_operations: stats.write_operations as u64,
            cache_hit_rate: 0.9, // Assume high cache hit rate for in-memory storage
        })
    }

    fn supports_random_access(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        false // Column store is batch-oriented
    }

    fn supports_compression(&self) -> bool {
        true
    }

    fn optimal_chunk_size(&self) -> usize {
        64 * 1024 // 64KB chunks
    }

    fn memory_overhead(&self) -> usize {
        1024 // Approximate overhead per chunk
    }

    fn optimize_for_pattern(&mut self, pattern: AccessPattern) -> Result<()> {
        match pattern {
            AccessPattern::Columnar | AccessPattern::ReadHeavy => {
                // Already optimized for these patterns
                Ok(())
            }
            AccessPattern::Sequential => {
                // Could pre-load adjacent data
                Ok(())
            }
            _ => {
                // Other patterns might not be optimal for column store
                Ok(())
            }
        }
    }

    fn compact(&mut self, handle: &Self::Handle) -> Result<()> {
        // Trigger optimization to recompress all columns
        handle.store.optimize()
    }
}
