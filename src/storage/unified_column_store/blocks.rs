//! Physical block management for the unified column store.
//!
//! A block is the unit of compression, encoding and I/O. Block metadata now
//! records the row range and chunk layout a block covers, which is what lets
//! `read_chunk` honour a row-indexed [`ChunkRange`](crate::storage::unified_memory::ChunkRange)
//! instead of always returning every block.

use crate::core::error::{Error, Result};
use crate::storage::checksum::checksum64;
use crate::storage::unified_column_store::encoding::EncodingType;
use crate::storage::unified_memory::{ChunkLayout, CompressionType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u64);

/// Block location information
#[derive(Debug, Clone)]
pub struct BlockLocation {
    pub offset: u64,
    pub size: usize,
}

/// Block metadata for columnar storage
#[derive(Debug, Clone)]
pub struct BlockMetadata {
    pub id: BlockId,
    pub location: BlockLocation,
    pub compressed_size: usize,
    pub uncompressed_size: usize,
    pub compression_type: CompressionType,
    pub encoding_type: EncodingType,
    pub checksum: u64,
    pub created_at: Instant,
    /// First row of the logical stream covered by this block (inclusive)
    pub row_start: usize,
    /// Number of logical rows covered by this block
    pub row_count: usize,
    /// Layout of the rows this block stores
    pub layout: ChunkLayout,
    pub min_value: Option<Vec<u8>>,
    pub max_value: Option<Vec<u8>>,
    pub null_count: u64,
    pub distinct_count: Option<u64>,
}

impl BlockMetadata {
    /// Row range covered by this block, half-open.
    pub fn row_range(&self) -> std::ops::Range<usize> {
        self.row_start..self.row_start + self.row_count
    }
}

/// Compressed block data
#[derive(Debug, Clone)]
pub struct CompressedBlock {
    pub data: Vec<u8>,
    pub compression_type: CompressionType,
    pub encoding_type: EncodingType,
    pub metadata: BlockMetadata,
}

impl CompressedBlock {
    pub fn new(
        data: Vec<u8>,
        compression_type: CompressionType,
        encoding_type: EncodingType,
    ) -> Self {
        let metadata = BlockMetadata {
            id: BlockId(0), // assigned by the block manager on write
            location: BlockLocation {
                offset: 0,
                size: data.len(),
            },
            compressed_size: data.len(),
            uncompressed_size: data.len(), // updated by the caller after encoding
            compression_type,
            encoding_type,
            checksum: Self::compute_checksum(&data),
            created_at: Instant::now(),
            row_start: 0,
            row_count: 0,
            layout: ChunkLayout::Opaque,
            min_value: None,
            max_value: None,
            null_count: 0,
            distinct_count: None,
        };

        Self {
            data,
            compression_type,
            encoding_type,
            metadata,
        }
    }

    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    pub fn uncompressed_size(&self) -> usize {
        self.metadata.uncompressed_size
    }

    pub fn checksum(&self) -> u64 {
        self.metadata.checksum
    }

    /// CRC-32C based integrity checksum.
    ///
    /// This replaced a plain byte-sum, which could not detect reordered or
    /// swapped bytes at all.
    pub fn compute_checksum(data: &[u8]) -> u64 {
        checksum64(data)
    }
}

/// Physical storage trait
pub trait PhysicalStorage: Send + Sync {
    fn write_at_location(&mut self, location: &BlockLocation, data: &[u8]) -> Result<()>;
    fn read_at_location(&self, location: &BlockLocation, size: usize) -> Result<Vec<u8>>;
    fn delete_at_location(&mut self, location: &BlockLocation) -> Result<()>;
    fn total_size(&self) -> u64;
    fn available_space(&self) -> u64;
    /// Grow the backing store so that `required` bytes are addressable.
    fn ensure_capacity(&mut self, required: usize) -> Result<()>;
}

/// In-memory physical storage.
///
/// The buffer grows on demand up to `max_size` instead of being pinned at a
/// fixed 100 MB, so a store no longer becomes permanently unwritable once the
/// bump allocator has walked past the end.
pub struct InMemoryPhysicalStorage {
    data: Vec<u8>,
    max_size: usize,
    live_bytes: usize,
}

impl InMemoryPhysicalStorage {
    pub fn new(capacity: usize) -> Self {
        Self::with_limit(capacity, usize::MAX)
    }

    pub fn with_limit(initial_capacity: usize, max_size: usize) -> Self {
        Self {
            data: vec![0; initial_capacity.min(max_size)],
            max_size,
            live_bytes: 0,
        }
    }
}

impl PhysicalStorage for InMemoryPhysicalStorage {
    fn write_at_location(&mut self, location: &BlockLocation, data: &[u8]) -> Result<()> {
        let start = location.offset as usize;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| Error::InvalidOperation("Block location overflow".to_string()))?;

        self.ensure_capacity(end)?;
        self.data[start..end].copy_from_slice(data);
        self.live_bytes = self.live_bytes.saturating_add(data.len());
        Ok(())
    }

    fn read_at_location(&self, location: &BlockLocation, size: usize) -> Result<Vec<u8>> {
        let start = location.offset as usize;
        let end = start
            .checked_add(size)
            .ok_or_else(|| Error::InvalidOperation("Block location overflow".to_string()))?;

        if end > self.data.len() {
            return Err(Error::InvalidOperation(
                "Read beyond storage bounds".to_string(),
            ));
        }

        Ok(self.data[start..end].to_vec())
    }

    fn delete_at_location(&mut self, location: &BlockLocation) -> Result<()> {
        self.live_bytes = self.live_bytes.saturating_sub(location.size);
        Ok(())
    }

    fn total_size(&self) -> u64 {
        self.data.len() as u64
    }

    fn available_space(&self) -> u64 {
        (self.max_size.saturating_sub(self.live_bytes)) as u64
    }

    fn ensure_capacity(&mut self, required: usize) -> Result<()> {
        if required <= self.data.len() {
            return Ok(());
        }
        if required > self.max_size {
            return Err(Error::InvalidOperation(format!(
                "Column store backing storage exhausted: need {} bytes, limit is {}",
                required, self.max_size
            )));
        }
        // Grow geometrically to keep amortised cost low.
        let new_len = required
            .max(self.data.len().saturating_mul(2))
            .min(self.max_size);
        self.data.resize(new_len, 0);
        Ok(())
    }
}

/// Free space tracker
#[derive(Debug, Default)]
pub struct FreeSpaceTracker {
    free_blocks: Vec<(u64, usize)>,
}

impl FreeSpaceTracker {
    pub fn new() -> Self {
        Self {
            free_blocks: Vec::new(),
        }
    }

    pub fn add_free_space(&mut self, offset: u64, size: usize) {
        if size == 0 {
            return;
        }
        self.free_blocks.push((offset, size));
        self.free_blocks.sort_by_key(|(offset, _)| *offset);
        self.merge_adjacent_blocks();
    }

    /// Merge adjacent free blocks to reduce fragmentation
    pub fn merge_adjacent_blocks(&mut self) {
        if self.free_blocks.len() <= 1 {
            return;
        }

        let mut merged = Vec::with_capacity(self.free_blocks.len());
        let mut current = self.free_blocks[0];

        for &next in self.free_blocks.iter().skip(1) {
            if current.0 + current.1 as u64 == next.0 {
                current.1 += next.1;
            } else {
                merged.push(current);
                current = next;
            }
        }
        merged.push(current);

        self.free_blocks = merged;
    }

    pub fn find_space(&self, required_size: usize) -> Option<u64> {
        self.free_blocks
            .iter()
            .find(|(_, size)| *size >= required_size)
            .map(|(offset, _)| *offset)
    }

    /// Claim `size` bytes from the free block starting at `offset`.
    fn claim(&mut self, offset: u64, size: usize) {
        if let Some(pos) = self.free_blocks.iter().position(|(o, _)| *o == offset) {
            let (o, s) = self.free_blocks[pos];
            if s > size {
                self.free_blocks[pos] = (o + size as u64, s - size);
            } else {
                self.free_blocks.remove(pos);
            }
        }
    }

    /// Total number of bytes currently reclaimable.
    pub fn free_bytes(&self) -> usize {
        self.free_blocks.iter().map(|(_, size)| *size).sum()
    }
}

/// Block allocator: reuses reclaimed space before extending the store.
#[derive(Debug, Default)]
pub struct BlockAllocator {
    next_offset: u64,
    free_space: FreeSpaceTracker,
}

impl BlockAllocator {
    pub fn new() -> Self {
        Self {
            next_offset: 0,
            free_space: FreeSpaceTracker::new(),
        }
    }

    /// Allocate `size` bytes, preferring a reclaimed region.
    ///
    /// The previous implementation was a pure bump allocator that never
    /// consulted the free-space tracker, so deleted blocks leaked their space
    /// forever.
    pub fn allocate_space(&mut self, size: usize) -> Result<BlockLocation> {
        if size == 0 {
            return Ok(BlockLocation {
                offset: self.next_offset,
                size: 0,
            });
        }
        if let Some(offset) = self.free_space.find_space(size) {
            self.free_space.claim(offset, size);
            return Ok(BlockLocation { offset, size });
        }
        let offset = self.next_offset;
        self.next_offset = self
            .next_offset
            .checked_add(size as u64)
            .ok_or_else(|| Error::InvalidOperation("Block offset overflow".to_string()))?;
        Ok(BlockLocation { offset, size })
    }

    pub fn release(&mut self, location: &BlockLocation) {
        self.free_space
            .add_free_space(location.offset, location.size);
    }

    pub fn free_space(&mut self) -> &mut FreeSpaceTracker {
        &mut self.free_space
    }

    pub fn high_water_mark(&self) -> u64 {
        self.next_offset
    }
}

/// Bounded LRU cache of decompressed-on-demand blocks.
///
/// The old cache was an unbounded `HashMap` that every read and write inserted
/// into and nothing ever evicted, so it duplicated the entire store in RAM and
/// `metadata_cache_size` was never used for anything.
#[derive(Debug, Default)]
struct BlockCache {
    entries: HashMap<BlockId, (CompressedBlock, u64)>,
    capacity_bytes: usize,
    used_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
}

impl BlockCache {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity_bytes,
            used_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, id: BlockId) -> Option<CompressedBlock> {
        self.clock += 1;
        let clock = self.clock;
        match self.entries.get_mut(&id) {
            Some((block, stamp)) => {
                *stamp = clock;
                self.hits += 1;
                Some(block.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn put(&mut self, id: BlockId, block: CompressedBlock) {
        let size = block.data.len();
        if size > self.capacity_bytes {
            return;
        }
        if let Some((old, _)) = self.entries.remove(&id) {
            self.used_bytes = self.used_bytes.saturating_sub(old.data.len());
        }
        while self.used_bytes + size > self.capacity_bytes && !self.entries.is_empty() {
            if let Some((&victim, _)) = self.entries.iter().min_by_key(|(_, (_, stamp))| *stamp) {
                if let Some((evicted, _)) = self.entries.remove(&victim) {
                    self.used_bytes = self.used_bytes.saturating_sub(evicted.data.len());
                }
            } else {
                break;
            }
        }
        self.clock += 1;
        self.used_bytes += size;
        self.entries.insert(id, (block, self.clock));
    }

    fn remove(&mut self, id: BlockId) {
        if let Some((block, _)) = self.entries.remove(&id) {
            self.used_bytes = self.used_bytes.saturating_sub(block.data.len());
        }
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Block manager for handling physical storage
pub struct BlockManager {
    storage: Box<dyn PhysicalStorage>,
    allocator: BlockAllocator,
    block_cache: BlockCache,
    metadata_index: HashMap<BlockId, BlockMetadata>,
    next_block_id: AtomicU64,
}

impl std::fmt::Debug for BlockManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockManager")
            .field("blocks", &self.metadata_index.len())
            .field("high_water_mark", &self.allocator.high_water_mark())
            .finish()
    }
}

impl BlockManager {
    pub fn new(storage: Box<dyn PhysicalStorage>) -> Self {
        Self::with_cache_capacity(storage, 10 * 1024 * 1024)
    }

    pub fn with_cache_capacity(storage: Box<dyn PhysicalStorage>, cache_bytes: usize) -> Self {
        Self {
            storage,
            allocator: BlockAllocator::new(),
            block_cache: BlockCache::new(cache_bytes),
            metadata_index: HashMap::new(),
            next_block_id: AtomicU64::new(1),
        }
    }

    pub fn write_block(&mut self, mut block: CompressedBlock) -> Result<BlockId> {
        let block_id = BlockId(self.next_block_id.fetch_add(1, Ordering::SeqCst));
        block.metadata.id = block_id;

        let location = self.allocator.allocate_space(block.compressed_size())?;
        block.metadata.location = location.clone();
        block.metadata.compressed_size = block.data.len();
        block.metadata.checksum = CompressedBlock::compute_checksum(&block.data);

        self.storage.write_at_location(&location, &block.data)?;
        self.metadata_index.insert(block_id, block.metadata.clone());
        self.block_cache.put(block_id, block);

        Ok(block_id)
    }

    pub fn read_block(&mut self, block_id: BlockId) -> Result<CompressedBlock> {
        if let Some(block) = self.block_cache.get(block_id) {
            return Ok(block);
        }

        let metadata = self
            .metadata_index
            .get(&block_id)
            .ok_or_else(|| Error::InvalidOperation(format!("Block {:?} not found", block_id)))?
            .clone();

        let data = self
            .storage
            .read_at_location(&metadata.location, metadata.compressed_size)?;

        let computed_checksum = CompressedBlock::compute_checksum(&data);
        if computed_checksum != metadata.checksum {
            return Err(Error::InvalidOperation(format!(
                "Checksum mismatch for block {:?}: expected {}, got {}",
                block_id, metadata.checksum, computed_checksum
            )));
        }

        let block = CompressedBlock {
            data,
            compression_type: metadata.compression_type,
            encoding_type: metadata.encoding_type,
            metadata,
        };

        self.block_cache.put(block_id, block.clone());
        Ok(block)
    }

    /// Replace a block's payload in place, keeping metadata consistent.
    ///
    /// The compaction path used to write new ZSTD bytes but leave the stale
    /// `compression_type` in the metadata index, so post-compaction reads fed
    /// ZSTD bytes to whatever codec the old tag named.
    pub fn replace_block(
        &mut self,
        block_id: BlockId,
        data: Vec<u8>,
        compression_type: CompressionType,
    ) -> Result<()> {
        let old_location = self
            .metadata_index
            .get(&block_id)
            .map(|m| m.location.clone())
            .ok_or_else(|| Error::InvalidOperation(format!("Block {:?} not found", block_id)))?;

        // The recompressed payload may be a different size, so allocate fresh
        // space and release the old extent rather than overwriting in place.
        let location = self.allocator.allocate_space(data.len())?;
        self.storage.write_at_location(&location, &data)?;
        if location.offset != old_location.offset {
            self.storage.delete_at_location(&old_location)?;
            self.allocator.release(&old_location);
        }

        let checksum = CompressedBlock::compute_checksum(&data);
        let metadata = self
            .metadata_index
            .get_mut(&block_id)
            .ok_or_else(|| Error::InvalidOperation(format!("Block {:?} not found", block_id)))?;
        metadata.location = location;
        metadata.compressed_size = data.len();
        metadata.compression_type = compression_type;
        metadata.checksum = checksum;
        let metadata = metadata.clone();

        self.block_cache.put(
            block_id,
            CompressedBlock {
                data,
                compression_type,
                encoding_type: metadata.encoding_type,
                metadata,
            },
        );
        Ok(())
    }

    /// Delete a block, reclaiming its space.
    pub fn delete_block(&mut self, block_id: BlockId) -> Result<bool> {
        let metadata = match self.metadata_index.remove(&block_id) {
            Some(metadata) => metadata,
            None => return Ok(false),
        };
        self.storage.delete_at_location(&metadata.location)?;
        self.allocator.release(&metadata.location);
        self.block_cache.remove(block_id);
        Ok(true)
    }

    pub fn metadata(&self, block_id: BlockId) -> Option<&BlockMetadata> {
        self.metadata_index.get(&block_id)
    }

    pub fn block_count(&self) -> usize {
        self.metadata_index.len()
    }

    /// Total compressed bytes currently held by live blocks.
    pub fn stored_bytes(&self) -> u64 {
        self.metadata_index
            .values()
            .map(|m| m.compressed_size as u64)
            .sum()
    }

    pub fn cache_hit_rate(&self) -> f64 {
        self.block_cache.hit_rate()
    }

    pub fn reclaimable_bytes(&mut self) -> usize {
        self.allocator.free_space().free_bytes()
    }

    pub fn merge_free_space(&mut self) {
        self.allocator.free_space().merge_adjacent_blocks();
    }

    pub fn backing_size(&self) -> u64 {
        self.storage.total_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_write_read_roundtrip() {
        let storage = Box::new(InMemoryPhysicalStorage::new(1024 * 1024));
        let mut manager = BlockManager::new(storage);

        let block = CompressedBlock::new(
            vec![1, 2, 3, 4, 5],
            CompressionType::None,
            EncodingType::None,
        );

        let block_id = manager.write_block(block).expect("write");
        let read_block = manager.read_block(block_id).expect("read");
        assert_eq!(read_block.data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn deleted_space_is_reused() {
        let storage = Box::new(InMemoryPhysicalStorage::new(4096));
        let mut manager = BlockManager::new(storage);

        let first = manager
            .write_block(CompressedBlock::new(
                vec![7u8; 512],
                CompressionType::None,
                EncodingType::None,
            ))
            .expect("write");
        let watermark_before = manager.allocator.high_water_mark();
        assert!(manager.delete_block(first).expect("delete"));
        assert_eq!(manager.reclaimable_bytes(), 512);

        let second = manager
            .write_block(CompressedBlock::new(
                vec![9u8; 512],
                CompressionType::None,
                EncodingType::None,
            ))
            .expect("write");
        // Reused the freed extent instead of bumping past it.
        assert_eq!(manager.allocator.high_water_mark(), watermark_before);
        assert_eq!(
            manager.read_block(second).expect("read").data,
            vec![9u8; 512]
        );
    }

    #[test]
    fn backing_storage_grows_past_initial_capacity() {
        let storage = Box::new(InMemoryPhysicalStorage::with_limit(64, 1024 * 1024));
        let mut manager = BlockManager::new(storage);
        for _ in 0..8 {
            manager
                .write_block(CompressedBlock::new(
                    vec![1u8; 256],
                    CompressionType::None,
                    EncodingType::None,
                ))
                .expect("write must grow the buffer, not fail");
        }
        assert_eq!(manager.block_count(), 8);
    }

    #[test]
    fn corrupted_bytes_fail_the_checksum() {
        let storage = Box::new(InMemoryPhysicalStorage::new(4096));
        let mut manager = BlockManager::new(storage);
        let id = manager
            .write_block(CompressedBlock::new(
                vec![1, 2, 3, 4],
                CompressionType::None,
                EncodingType::None,
            ))
            .expect("write");
        // Drop it from the cache so the read goes to physical storage, then
        // corrupt the stored bytes.
        manager.block_cache.remove(id);
        let location = manager.metadata(id).expect("metadata").location.clone();
        manager
            .storage
            .write_at_location(&location, &[4, 3, 2, 1])
            .expect("corrupt");
        assert!(manager.read_block(id).is_err());
    }

    #[test]
    fn cache_is_bounded() {
        let storage = Box::new(InMemoryPhysicalStorage::new(1024 * 1024));
        let mut manager = BlockManager::with_cache_capacity(storage, 1024);
        for _ in 0..32 {
            manager
                .write_block(CompressedBlock::new(
                    vec![3u8; 256],
                    CompressionType::None,
                    EncodingType::None,
                ))
                .expect("write");
        }
        assert!(
            manager.block_cache.used_bytes <= 1024,
            "cache grew past its capacity: {}",
            manager.block_cache.used_bytes
        );
    }
}
