// 特定の警告を無効化
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod dataframe;
pub mod error;
pub mod groupby;
pub mod index;
pub mod io;
pub mod na;
pub mod parallel;
pub mod pivot;
pub mod series;
pub mod temporal;
pub mod vis;

// Re-export commonly used types
pub use dataframe::DataFrame;
pub use error::PandRSError;
pub use groupby::GroupBy;
pub use index::Index;
pub use na::NA;
pub use parallel::ParallelUtils;
pub use series::{NASeries, Series};
pub use vis::{OutputFormat, PlotConfig, PlotType};

// Export version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
