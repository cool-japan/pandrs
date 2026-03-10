/// Series implementations module.
///
/// This module provides the core Series data structure, a one-dimensional labeled array
/// capable of holding any data type with integer or string labels. Series is the building
/// block for DataFrames and provides many operations for data manipulation.

/// Base Series implementation with core operations.
///
/// The fundamental Series structure with essential methods for:
/// - Element access and slicing
/// - Arithmetic and comparison operations
/// - Sorting and ranking
/// - Type conversions
///
/// # Examples
///
/// ```rust
/// use pandrs::Series;
///
/// // Create a numeric series
/// let s = Series::new(vec![1, 2, 3, 4, 5], Some("numbers".to_string())).expect("Failed to create series");
///
/// // Basic operations
/// assert_eq!(s.len(), 5);
/// assert_eq!(s.name(), Some(&"numbers".to_string()));
/// ```
pub mod base;

/// Categorical data type for memory-efficient string handling.
///
/// Provides categorical encoding for string data with:
/// - Memory-efficient storage
/// - Fast equality comparisons
/// - Ordered and unordered categories
///
/// # Examples
///
/// ```rust
/// use pandrs::Categorical;
///
/// let data = vec!["small", "medium", "large", "small", "large"];
/// let categories = vec!["small", "medium", "large"];
/// let cat = Categorical::new(data, Some(categories), true).expect("Failed to create");
/// ```
pub mod categorical;

/// DateTime accessor for time-based operations on Series.
///
/// Provides specialized methods for datetime Series:
/// - Extract date components (year, month, day)
/// - Extract time components (hour, minute, second)
/// - Timezone conversions
/// - Date arithmetic
///
/// # Examples
///
/// ```rust,no_run
/// use pandrs::Series;
/// use chrono::NaiveDate;
///
/// let dates = vec![
///     NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
///     NaiveDate::from_ymd_opt(2024, 6, 15).unwrap(),
/// ];
/// let s = Series::new(dates, Some("dates".to_string())).expect("Failed to create");
///
/// // Access datetime methods through .dt accessor
/// // let years = s.dt().year();
/// ```
pub mod datetime_accessor;

/// Utility functions for Series operations.
///
/// Helper functions for common Series manipulations.
pub mod functions;

/// NA (Not Available) handling for missing data.
///
/// Provides methods for:
/// - Detecting missing values
/// - Filling missing values
/// - Dropping missing values
/// - Interpolation
pub mod na;

/// String accessor for text operations on Series.
///
/// Provides vectorized string operations:
/// - Case conversions
/// - Pattern matching
/// - String splitting and joining
/// - Regular expressions
///
/// # Examples
///
/// ```rust
/// use pandrs::Series;
///
/// let s = Series::new(
///     vec!["hello", "world", "rust"],
///     Some("text".to_string())
/// ).expect("Failed to create");
///
/// // Access string methods through .str accessor
/// // let upper = s.str().upper();
/// ```
pub mod string_accessor;

/// Window operations for rolling computations.
///
/// Provides windowing functionality:
/// - Rolling windows
/// - Expanding windows
/// - Exponentially weighted windows
///
/// # Examples
///
/// ```rust
/// use pandrs::Series;
///
/// let s = Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], None).expect("Failed to create");
///
/// // Rolling mean with window size 3
/// // let rolling_mean = s.rolling(3).mean();
/// ```
pub mod window;

#[cfg(cuda_available)]
pub mod gpu;

// Re-exports for convenience
pub use base::Series;
pub use categorical::{Categorical, CategoricalOrder, StringCategorical};
pub use datetime_accessor::{DateTimeAccessor, DateTimeAccessorTz};
pub use na::NASeries;
pub use string_accessor::StringAccessor;
pub use window::{Expanding, Rolling, WindowClosed, WindowExt, WindowOps, EWM};

// Optional feature re-exports
#[cfg(cuda_available)]
pub use gpu::SeriesGpuExt;

// Legacy exports for backward compatibility
pub use categorical::{
    Categorical as LegacyCategorical, CategoricalOrder as LegacyCategoricalOrder,
    StringCategorical as LegacyStringCategorical,
};
// For backward compatibility (old NASeries)
#[allow(deprecated)]
pub use na::NASeries as LegacyNASeries;
