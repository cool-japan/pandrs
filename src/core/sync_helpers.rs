//! Synchronization helpers for safe lock operations
//!
//! This module provides macros and helper functions for handling locks safely
//! without using unwrap(), which can panic on poisoned locks or other errors.

use crate::core::error::{Error, Result};
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Safely acquire a Mutex lock, converting poison errors to Result
///
/// # Arguments
/// * `$lock` - The Mutex to lock
/// * `$context` - Context string for error messages
///
/// # Returns
/// * `Result<MutexGuard>` - The lock guard or an error
///
/// # Example
/// ```ignore
/// let data = lock_safe!(self.cache, "cache access")?;
/// ```
#[macro_export]
macro_rules! lock_safe {
    ($lock:expr, $context:expr) => {
        $lock
            .lock()
            .map_err(|e| $crate::core::error::Error::lock_poisoned(format!("{}: {}", $context, e)))
    };
}

/// Safely acquire a RwLock read lock
///
/// # Arguments
/// * `$lock` - The RwLock to read lock
/// * `$context` - Context string for error messages
///
/// # Returns
/// * `Result<RwLockReadGuard>` - The read lock guard or an error
///
/// # Example
/// ```ignore
/// let data = read_lock_safe!(self.buffer, "buffer read")?;
/// ```
#[macro_export]
macro_rules! read_lock_safe {
    ($lock:expr, $context:expr) => {
        $lock
            .read()
            .map_err(|e| $crate::core::error::Error::lock_poisoned(format!("{}: {}", $context, e)))
    };
}

/// Safely acquire a RwLock write lock
///
/// # Arguments
/// * `$lock` - The RwLock to write lock
/// * `$context` - Context string for error messages
///
/// # Returns
/// * `Result<RwLockWriteGuard>` - The write lock guard or an error
///
/// # Example
/// ```ignore
/// let mut data = write_lock_safe!(self.state, "state update")?;
/// ```
#[macro_export]
macro_rules! write_lock_safe {
    ($lock:expr, $context:expr) => {
        $lock
            .write()
            .map_err(|e| $crate::core::error::Error::lock_poisoned(format!("{}: {}", $context, e)))
    };
}

/// Lock a Mutex and clone the contained value
///
/// # Arguments
/// * `$lock` - The Mutex to lock
/// * `$context` - Context string for error messages
///
/// # Returns
/// * `Result<T>` - The cloned value or an error
///
/// # Example
/// ```ignore
/// let config = lock_and_clone!(self.config, "config access")?;
/// ```
#[macro_export]
macro_rules! lock_and_clone {
    ($lock:expr, $context:expr) => {
        $lock
            .lock()
            .map_err(|e| $crate::core::error::Error::lock_poisoned(format!("{}: {}", $context, e)))
            .map(|guard| guard.clone())
    };
}

/// Helper function to perform an operation with a write lock
///
/// This function acquires a write lock, performs the operation, and releases the lock.
///
/// # Arguments
/// * `lock` - The RwLock to write lock
/// * `context` - Context string for error messages
/// * `f` - Function to execute with mutable access to the locked data
///
/// # Returns
/// * `Result<R>` - Result of the operation or a lock error
pub fn with_write<T, F, R>(lock: &RwLock<T>, context: &str, f: F) -> Result<R>
where
    F: FnOnce(&mut T) -> R,
{
    let mut guard = lock
        .write()
        .map_err(|e| Error::lock_poisoned(format!("{}: {}", context, e)))?;
    Ok(f(&mut *guard))
}

/// Helper function to perform an operation with a read lock
///
/// This function acquires a read lock, performs the operation, and releases the lock.
///
/// # Arguments
/// * `lock` - The RwLock to read lock
/// * `context` - Context string for error messages
/// * `f` - Function to execute with read access to the locked data
///
/// # Returns
/// * `Result<R>` - Result of the operation or a lock error
pub fn with_read<T, F, R>(lock: &RwLock<T>, context: &str, f: F) -> Result<R>
where
    F: FnOnce(&T) -> R,
{
    let guard = lock
        .read()
        .map_err(|e| Error::lock_poisoned(format!("{}: {}", context, e)))?;
    Ok(f(&*guard))
}

/// Helper function to perform an operation with a mutex lock
///
/// This function acquires a mutex lock, performs the operation, and releases the lock.
///
/// # Arguments
/// * `lock` - The Mutex to lock
/// * `context` - Context string for error messages
/// * `f` - Function to execute with mutable access to the locked data
///
/// # Returns
/// * `Result<R>` - Result of the operation or a lock error
pub fn with_mutex<T, F, R>(lock: &Mutex<T>, context: &str, f: F) -> Result<R>
where
    F: FnOnce(&mut T) -> R,
{
    let mut guard = lock
        .lock()
        .map_err(|e| Error::lock_poisoned(format!("{}: {}", context, e)))?;
    Ok(f(&mut *guard))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, RwLock};

    #[test]
    fn test_lock_safe_macro() {
        let mutex = Arc::new(Mutex::new(42));
        let result = lock_safe!(mutex, "test context");
        assert!(result.is_ok());
        assert_eq!(*result.expect("operation should succeed"), 42);
    }

    #[test]
    fn test_read_lock_safe_macro() {
        let rwlock = Arc::new(RwLock::new(42));
        let result = read_lock_safe!(rwlock, "test read");
        assert!(result.is_ok());
        assert_eq!(*result.expect("operation should succeed"), 42);
    }

    #[test]
    fn test_write_lock_safe_macro() {
        let rwlock = Arc::new(RwLock::new(42));
        let result = write_lock_safe!(rwlock, "test write");
        assert!(result.is_ok());
        let mut guard = result.expect("operation should succeed");
        *guard = 100;
        drop(guard);
        assert_eq!(*rwlock.read().expect("operation should succeed"), 100);
    }

    #[test]
    fn test_lock_and_clone_macro() {
        let mutex = Arc::new(Mutex::new(String::from("test")));
        let result = lock_and_clone!(mutex, "clone test");
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed"), "test");
    }

    #[test]
    fn test_with_write_helper() {
        let rwlock = RwLock::new(42);
        let result = with_write(&rwlock, "test write", |value| {
            *value = 100;
            *value
        });
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed"), 100);
        assert_eq!(*rwlock.read().expect("operation should succeed"), 100);
    }

    #[test]
    fn test_with_read_helper() {
        let rwlock = RwLock::new(42);
        let result = with_read(&rwlock, "test read", |value| *value * 2);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed"), 84);
    }

    #[test]
    fn test_with_mutex_helper() {
        let mutex = Mutex::new(42);
        let result = with_mutex(&mutex, "test mutex", |value| {
            *value = 100;
            *value
        });
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed"), 100);
        assert_eq!(*mutex.lock().expect("operation should succeed"), 100);
    }
}
