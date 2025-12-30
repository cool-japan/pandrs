//! # Core Fault Tolerance Types
//!
//! This module provides core types and structures for fault tolerance in
//! distributed processing.

use crate::error::{Error, Result};
use std::time::{Duration, Instant};

/// Retry policy for failed operations
#[derive(Debug, Clone, Copy)]
pub enum RetryPolicy {
    /// No retry attempts
    None,
    /// Fixed interval between retry attempts
    Fixed {
        /// Maximum number of retry attempts
        max_retries: usize,
        /// Delay between retry attempts in milliseconds
        delay_ms: u64,
    },
    /// Exponential backoff between retry attempts
    Exponential {
        /// Maximum number of retry attempts
        max_retries: usize,
        /// Initial delay in milliseconds
        initial_delay_ms: u64,
        /// Maximum delay in milliseconds
        max_delay_ms: u64,
        /// Backoff factor
        backoff_factor: f64,
    },
}

impl RetryPolicy {
    /// Creates a default retry policy with 3 retries and 1 second delay
    pub fn default_fixed() -> Self {
        Self::Fixed {
            max_retries: 3,
            delay_ms: 1000,
        }
    }

    /// Creates a default exponential backoff retry policy
    pub fn default_exponential() -> Self {
        Self::Exponential {
            max_retries: 5,
            initial_delay_ms: 100,
            max_delay_ms: 10000,
            backoff_factor: 2.0,
        }
    }

    /// Gets the maximum number of retries
    pub fn max_retries(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Fixed { max_retries, .. } => *max_retries,
            Self::Exponential { max_retries, .. } => *max_retries,
        }
    }

    /// Gets the delay for a specific retry attempt
    pub fn delay_for_attempt(&self, attempt: usize) -> Duration {
        match self {
            Self::None => Duration::from_millis(0),
            Self::Fixed { delay_ms, .. } => Duration::from_millis(*delay_ms),
            Self::Exponential {
                initial_delay_ms,
                max_delay_ms,
                backoff_factor,
                ..
            } => {
                let delay = (*initial_delay_ms as f64 * backoff_factor.powi(attempt as i32)) as u64;
                Duration::from_millis(delay.min(*max_delay_ms))
            }
        }
    }
}

/// Type of operation failures
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FailureType {
    /// Network failure (communication error)
    Network,
    /// Node failure (a compute node went down)
    Node,
    /// Memory error (out of memory)
    Memory,
    /// Timeout (operation took too long)
    Timeout,
    /// Data error (corrupted data, schema mismatch, etc.)
    Data,
    /// Unknown error
    Unknown,
}

impl FailureType {
    /// Determines if the failure is retriable
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::Network | Self::Node | Self::Timeout => true,
            Self::Memory | Self::Data | Self::Unknown => false,
        }
    }

    /// Gets a failure type from an error
    pub fn from_error(error: &Error) -> Self {
        match error {
            Error::IoError(_) => Self::Network,
            Error::Timeout(_) => Self::Timeout,
            Error::OutOfMemory(_) => Self::Memory,
            Error::DataError(_) => Self::Data,
            _ => Self::Unknown,
        }
    }
}

/// Information about a query failure
#[derive(Debug, Clone)]
pub struct FailureInfo {
    /// Type of failure
    pub failure_type: FailureType,
    /// Time of failure
    pub failure_time: Instant,
    /// Node ID (if applicable)
    pub node_id: Option<String>,
    /// Specific error message
    pub error_message: String,
    /// Whether the failure has been recovered
    pub recovered: bool,
    /// Number of retry attempts
    pub retry_attempts: usize,
}

impl FailureInfo {
    /// Creates a new failure info
    pub fn new(failure_type: FailureType, error_message: impl Into<String>) -> Self {
        Self {
            failure_type,
            failure_time: Instant::now(),
            node_id: None,
            error_message: error_message.into(),
            recovered: false,
            retry_attempts: 0,
        }
    }

    /// Sets the node ID
    pub fn with_node_id(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Marks the failure as recovered
    pub fn mark_recovered(&mut self) {
        self.recovered = true;
    }

    /// Increments the retry attempts
    pub fn increment_retry(&mut self) {
        self.retry_attempts += 1;
    }
}

/// Recovery strategy for failed operations
#[derive(Debug, Clone, Copy)]
pub enum RecoveryStrategy {
    /// Retry the entire query
    RetryQuery,
    /// Retry only the failed partitions
    RetryFailedPartitions,
    /// Reroute the query to different nodes
    Reroute,
    /// Fallback to a local execution engine
    LocalFallback,
}
