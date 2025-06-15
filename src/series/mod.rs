// Series implementations module
pub mod base;
pub mod categorical;
pub mod datetime_accessor;
pub mod functions;
pub mod na;
pub mod string_accessor;
pub mod window;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports for convenience
pub use base::Series;
pub use categorical::{Categorical, CategoricalOrder, StringCategorical};
pub use datetime_accessor::{DateTimeAccessor, DateTimeAccessorTz};
pub use na::NASeries;
pub use string_accessor::StringAccessor;
pub use window::{Expanding, Rolling, WindowClosed, WindowExt, WindowOps, EWM};

// Optional feature re-exports
#[cfg(feature = "cuda")]
pub use gpu::SeriesGpuExt;

// Legacy exports for backward compatibility
pub use categorical::{
    Categorical as LegacyCategorical, CategoricalOrder as LegacyCategoricalOrder,
    StringCategorical as LegacyStringCategorical,
};
// For backward compatibility (old NASeries)
#[allow(deprecated)]
pub use na::NASeries as LegacyNASeries;
