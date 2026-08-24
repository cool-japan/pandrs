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

// Re-export the companion extension traits (see functions_4's module docs)
// so they're reachable at `pandrs::dataframe::pandas_compat::AstypeErrorsExt`
// etc. via `pandas_compat::mod.rs`'s existing `pub use functions::*;`,
// instead of only at the buried `...::functions::functions_4::AstypeErrorsExt`.
pub use functions_4::*;
