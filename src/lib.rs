pub mod dataframe;
pub mod series;
pub mod io;
pub mod error;
pub mod index;
pub mod na;
pub mod groupby;
pub mod temporal;
pub mod pivot;
pub mod vis;
pub mod parallel;

// Re-export commonly used types
pub use dataframe::DataFrame;
pub use series::{Series, NASeries};
pub use index::Index;
pub use error::PandRSError;
pub use na::NA;
pub use groupby::GroupBy;
pub use vis::{PlotConfig, PlotType, OutputFormat};
pub use parallel::ParallelUtils;

// Export version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");