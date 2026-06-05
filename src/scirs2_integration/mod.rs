//! SciRS2 integration for PandRS
//!
//! Provides bridge between PandRS DataFrames/Series and SciRS2's
//! scientific computing capabilities.
//!
//! Enable with the `scirs2` feature flag:
//!
//! ```toml
//! [dependencies]
//! pandrs = { version = "0.3.0", features = ["scirs2"] }
//! ```
//!
//! ## Overview
//!
//! This module provides:
//! - Type conversions between PandRS and ndarray types
//! - Advanced statistical functions via SciRS2
//! - Linear algebra operations via SciRS2
//! - Extension traits for DataFrame

pub mod conversion;
pub mod dataframe_ext;
pub mod linalg;
pub mod stats;

// Re-export key types at module level
#[cfg(feature = "scirs2")]
pub use stats::{
    AnovaResult, Chi2TestResult, NormalityTestResult, PcaResult, SciRS2Stats, TTestResult,
};

#[cfg(feature = "scirs2")]
pub use linalg::{EigResult, LstsqDataFrameResult, LuResult, QrResult, SciRS2LinAlg, SvdResult};

#[cfg(feature = "scirs2")]
pub use dataframe_ext::SciRS2Ext;
