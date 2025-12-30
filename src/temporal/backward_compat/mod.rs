//! Backward compatibility layer for temporal module
//! Provides deprecated re-exports of the old module structure

// Re-export files from backward compatibility directory
mod date_range;
mod frequency;
mod resample;
mod window;

// Re-export core types from legacy files
#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::core::Temporal` instead"
)]
pub use super::core::Temporal;

#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::core::TimeSeries` instead"
)]
pub use super::core::TimeSeries;

// Re-export frequency types
#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::frequency::Frequency` instead"
)]
pub use self::frequency::Frequency;

// Re-export window operations
#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::window::Window` instead"
)]
pub use self::window::Window;

#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::window::WindowType` instead"
)]
pub use self::window::WindowType;

#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::window::WindowOperation` instead"
)]
pub use self::window::WindowOperation;

// Re-export date range functionality
#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::date_range::date_range` instead"
)]
pub use self::date_range::date_range;

#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::date_range::DateRange` instead"
)]
pub use self::date_range::DateRange;

// Re-export resample functionality
#[allow(deprecated)]
#[deprecated(
    since = "0.1.0",
    note = "Use `pandrs::temporal::resample::Resample` instead"
)]
pub use self::resample::Resample;