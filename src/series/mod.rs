// Series implementations module
pub mod base;
pub mod categorical;
pub mod functions;
pub mod na;

#[cfg(feature = "cuda")]
pub mod gpu;

// Re-exports for convenience
pub use base::Series;
pub use categorical::{Categorical, CategoricalOrder, StringCategorical};
pub use na::NASeries;

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
