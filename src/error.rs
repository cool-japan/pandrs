// Stable public alias of the canonical error module (`crate::core::error`).
//
// This path is intentionally kept as a supported, non-deprecated re-export:
// `pandrs::error` and `pandrs::PandRSError` are used throughout the examples
// and downstream code. It forwards to `crate::core::error`, which remains the
// canonical home for the error types.
pub use crate::core::error::{io_error, Error, PandRSError, Result};

// For backward compatibility, these From impls are implemented in the core::error module
