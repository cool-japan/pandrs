//! Module for time series data manipulation

// Module structure
pub mod core;
pub mod date_range;
pub mod frequency;
pub mod resample;
pub mod window;

// GPU-accelerated time series operations (conditionally compiled)
#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export public items from submodules
pub use self::core::{Temporal, TimeSeries};
pub use self::date_range::date_range;
pub use self::frequency::Frequency;
pub use self::resample::Resample;
pub use self::window::{Window, WindowOperation, WindowType};
