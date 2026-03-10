//! Module structure for pandas-compatible functions
//!
//! Split implementation of PandasCompatExt trait across multiple files
//! to comply with <2000 lines policy.

pub mod functions;
pub mod functions_2;
pub mod functions_2_impl_part1;
pub mod functions_2_impl_part2;
pub mod functions_2_impl_part3;
pub mod functions_3;
pub mod functions_4;
#[cfg(test)]
mod functions_tests_2;
#[cfg(test)]
mod functions_tests_3;

// Re-export all types
pub use functions::*;
pub use functions_2::*;
