//! Memory-mapped file access.
//!
//! This used to be an empty struct whose constructor discarded the path and
//! returned `Ok(Self {})`, shadowing the working machinery in
//! [`crate::storage::zero_copy`]. It is now a thin, safe front-end over
//! [`zero_copy::MemoryMappedView`](crate::storage::zero_copy::MemoryMappedView), which owns the actual `memmap2` mapping.

use crate::core::error::{Error, Result};
use crate::storage::zero_copy::MemoryMappedView;
use std::fs::File;
use std::path::{Path, PathBuf};

/// Memory-mapped file for efficient read access to large on-disk datasets.
///
/// The mapping is read-only and lives for as long as the `MemoryMappedFile`.
/// Typed element views are obtained with [`MemoryMappedFile::view`], which
/// delegates to [`zero_copy::MemoryMappedView`](crate::storage::zero_copy::MemoryMappedView).
pub struct MemoryMappedFile {
    /// Path the mapping was created from
    path: PathBuf,
    /// Byte length of the mapped file at open time
    byte_len: usize,
    /// Byte-granular mapping of the whole file
    bytes: MemoryMappedView<u8>,
}

impl MemoryMappedFile {
    /// Memory-map the file at `path` for reading.
    ///
    /// Returns an error for missing files and for empty files (an empty
    /// mapping has no valid address to hand out).
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| {
            Error::IoError(format!(
                "Failed to open {} for memory mapping: {}",
                path.display(),
                e
            ))
        })?;
        let byte_len = file
            .metadata()
            .map_err(|e| Error::IoError(format!("Failed to stat {}: {}", path.display(), e)))?
            .len() as usize;
        if byte_len == 0 {
            return Err(Error::InvalidOperation(format!(
                "Cannot memory-map empty file {}",
                path.display()
            )));
        }
        let bytes = MemoryMappedView::<u8>::from_file(file, byte_len)?;
        Ok(Self {
            path,
            byte_len,
            bytes,
        })
    }

    /// Path this mapping was created from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Length of the mapped region in bytes.
    pub fn len(&self) -> usize {
        self.byte_len
    }

    /// Whether the mapped region is empty.
    pub fn is_empty(&self) -> bool {
        self.byte_len == 0
    }

    /// Borrow the mapped file as raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Map the same file again as a typed element view.
    ///
    /// # Safety contract
    ///
    /// The caller must guarantee that the file's bytes are a valid array of
    /// `T`. This is only sound for plain-old-data element types whose every
    /// bit pattern is valid (`u8`, `i32`, `f64`, ...); it is **not** sound for
    /// `bool`, `char`, references or any type with invalid bit patterns, and
    /// the mapping must not be truncated by another process while alive.
    pub unsafe fn view<T>(&self) -> Result<MemoryMappedView<T>> {
        let element_size = std::mem::size_of::<T>();
        if element_size == 0 {
            return Err(Error::InvalidOperation(
                "Cannot build a memory-mapped view of a zero-sized type".to_string(),
            ));
        }
        let file = File::open(&self.path).map_err(|e| {
            Error::IoError(format!(
                "Failed to reopen {} for typed mapping: {}",
                self.path.display(),
                e
            ))
        })?;
        MemoryMappedView::<T>::from_file(file, self.byte_len / element_size)
    }
}

impl std::fmt::Debug for MemoryMappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryMappedFile")
            .field("path", &self.path)
            .field("byte_len", &self.byte_len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_file(name: &str, contents: &[u8]) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("pandrs_mmap_{}_{}.bin", name, std::process::id()));
        let mut file = std::fs::File::create(&path).expect("create temp file");
        file.write_all(contents).expect("write temp file");
        file.sync_all().expect("sync temp file");
        path
    }

    #[test]
    fn maps_real_file_contents() {
        let path = temp_file("contents", b"pandrs mapped bytes");
        let mapped = MemoryMappedFile::new(&path).expect("map");
        assert_eq!(mapped.len(), 19);
        assert_eq!(mapped.as_bytes(), b"pandrs mapped bytes");
        assert_eq!(mapped.path(), path.as_path());
        drop(mapped);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn typed_view_reads_elements() {
        let values: [u32; 4] = [1, 2, 3, 4];
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        let path = temp_file("typed", &bytes);
        let mapped = MemoryMappedFile::new(&path).expect("map");
        // SAFETY: the file was written as a native-endian [u32; 4] above and is
        // not modified while the view is alive.
        let view = unsafe { mapped.view::<u32>() }.expect("typed view");
        assert_eq!(view.as_slice(), &values[..]);
        drop(view);
        drop(mapped);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_an_error() {
        let mut path = std::env::temp_dir();
        path.push("pandrs_mmap_definitely_absent.bin");
        let _ = std::fs::remove_file(&path);
        assert!(MemoryMappedFile::new(&path).is_err());
    }

    #[test]
    fn empty_file_is_rejected() {
        let path = temp_file("empty", b"");
        assert!(MemoryMappedFile::new(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
