//! Pandas API Compatibility Extension
//!
//! Provides additional methods to improve pandas API compatibility.

pub mod concat;
pub mod functions;
pub mod groupby;
pub mod helpers;
pub mod merge;
pub mod rankmethod_traits;
pub mod seriesvalue_traits;
pub mod trait_def;
pub mod types;

// Re-export all types and traits
pub use concat::*;
pub use functions::*;
pub use groupby::*;
pub use merge::*;
pub use rankmethod_traits::*;
pub use seriesvalue_traits::*;
pub use trait_def::*;
pub use types::*;
