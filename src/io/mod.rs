pub mod csv;
#[cfg(feature = "excel")]
pub mod excel;
pub mod json;
#[cfg(feature = "parquet")]
pub mod parquet;
#[cfg(feature = "sql")]
pub mod sql;

// Re-export commonly used functions
pub use csv::{read_csv, write_csv};
#[cfg(feature = "excel")]
pub use excel::{read_excel, write_excel};
pub use json::{read_json, write_json};
#[cfg(feature = "parquet")]
pub use parquet::{read_parquet, write_parquet, ParquetCompression};
#[cfg(feature = "sql")]
pub use sql::{execute_sql, read_sql, write_to_sql};
