//! I/O operations for OptimizedDataFrame
//!
//! This module handles file I/O operations including:
//! - CSV reading/writing
//! - Excel reading/writing
//! - Parquet reading/writing  
//! - JSON reading/writing

use std::collections::HashMap;
use std::path::Path;

use crate::column::{BooleanColumn, Column, Float64Column, Int64Column, StringColumn};
use crate::error::{Error, Result};
#[cfg(feature = "parquet")]
use crate::optimized::split_dataframe::io::ParquetCompression;

use super::core::{ColumnView, JsonOrient, OptimizedDataFrame};

#[cfg(feature = "excel")]
use simple_excel_writer::{Sheet, Workbook};

impl OptimizedDataFrame {
    /// Create a DataFrame from a CSV file (high-performance implementation)
    /// # Arguments
    /// * `path` - Path to the CSV file
    /// * `has_header` - Whether the file has a header
    /// # Returns
    /// * `Result<Self>` - DataFrame on success, error on failure
    pub fn from_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<Self> {
        // Using implementation from split_dataframe/io.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Call from_csv from SplitDataFrame
        let split_df = SplitDataFrame::from_csv(path, has_header)?;

        // Convert to StandardDataFrame (for compatibility)
        let mut df = Self::new();

        // Copy column data
        for name in split_df.column_names() {
            let column_result = split_df.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column;
                // Same as original code
                df.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_df.get_index() {
            df.index = Some(index.clone());
        }

        Ok(df)
    }

    /// Save DataFrame to a CSV file
    /// # Arguments
    /// * `path` - Path to save the file
    /// * `write_header` - Whether to write the header
    /// # Returns
    /// * `Result<()>` - Ok on success, error on failure
    pub fn to_csv<P: AsRef<Path>>(&self, path: P, write_header: bool) -> Result<()> {
        // Using implementation from split_dataframe/io.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Call to_csv from SplitDataFrame
        split_df.to_csv(path, write_header)
    }

    /// Read DataFrame from an Excel file
    #[cfg(feature = "excel")]
    pub fn from_excel<P: AsRef<Path>>(
        path: P,
        sheet_name: Option<&str>,
        header: bool,
        skip_rows: usize,
        use_cols: Option<&[&str]>,
    ) -> Result<Self> {
        // Using implementation from split_dataframe/io.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Call from_excel from SplitDataFrame
        let split_df = SplitDataFrame::from_excel(path, sheet_name, header, skip_rows, use_cols)?;

        // Convert to OptimizedDataFrame
        let mut df = Self::new();

        // Copy column data
        for name in split_df.column_names() {
            let column_result = split_df.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column;
                df.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_df.get_index() {
            df.index = Some(index.clone());
        }

        Ok(df)
    }

    /// Write DataFrame to an Excel file
    #[cfg(feature = "excel")]
    pub fn to_excel<P: AsRef<Path>>(
        &self,
        path: P,
        sheet_name: Option<&str>,
        index: bool,
    ) -> Result<()> {
        // Using implementation from split_dataframe/io.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            let column_result = self.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column();
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            let _ = split_df.set_index(index.clone());
        }

        // Call to_excel from SplitDataFrame
        split_df.to_excel(path, sheet_name, index)
    }

    /// Write DataFrame to a Parquet file
    #[cfg(feature = "parquet")]
    pub fn to_parquet<P: AsRef<Path>>(
        &self,
        path: P,
        compression: Option<ParquetCompression>,
    ) -> Result<()> {
        // Using implementation from split_dataframe/io.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::io::ParquetCompression as SplitParquetCompression;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Convert compression settings
        let split_compression = compression.map(|c| match c {
            ParquetCompression::None => SplitParquetCompression::None,
            ParquetCompression::Snappy => SplitParquetCompression::Snappy,
            ParquetCompression::Gzip => SplitParquetCompression::Gzip,
            ParquetCompression::Lzo => SplitParquetCompression::Lzo,
            ParquetCompression::Brotli => SplitParquetCompression::Brotli,
            ParquetCompression::Lz4 => SplitParquetCompression::Lz4,
            ParquetCompression::Zstd => SplitParquetCompression::Zstd,
        });

        // Call to_parquet from SplitDataFrame
        split_df.to_parquet(path, split_compression)
    }

    /// Read DataFrame from a Parquet file
    #[cfg(feature = "parquet")]
    pub fn from_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Using implementation from split_dataframe/io.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Call from_parquet from SplitDataFrame
        let split_df = SplitDataFrame::from_parquet(path)?;

        // Convert to StandardDataFrame (for compatibility)
        let mut df = Self::new();

        // Copy column data
        for name in split_df.column_names() {
            let column_result = split_df.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column;
                // Same as original code
                df.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_df.get_index() {
            df.index = Some(index.clone());
        }

        Ok(df)
    }

    /// Read DataFrame from a JSON file
    ///
    /// # Arguments
    /// * `path` - Path to the JSON file
    ///
    /// # Returns
    /// * `Result<Self>` - DataFrame read from the file
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Using implementation from split_dataframe/serialize.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::serialize::JsonOrient as SplitJsonOrient;

        // Call from_json from SplitDataFrame
        let split_df = SplitDataFrame::from_json(path)?;

        // Convert to OptimizedDataFrame
        let mut df = Self::new();

        // Copy column data
        for name in split_df.column_names() {
            let column_result = split_df.column(name);
            if let Ok(column_view) = column_result {
                let column = column_view.column;
                df.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_df.get_index() {
            df.index = Some(index.clone());
        }

        Ok(df)
    }

    /// Write DataFrame to a JSON file
    ///
    /// # Arguments
    /// * `path` - Path to the JSON file
    /// * `orient` - JSON output format (Records or Columns)
    ///
    /// # Returns
    /// * `Result<()>` - Ok on success
    pub fn to_json<P: AsRef<Path>>(&self, path: P, orient: JsonOrient) -> Result<()> {
        // Using implementation from split_dataframe/serialize.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::serialize::JsonOrient as SplitJsonOrient;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Convert JSON output format
        let split_orient = match orient {
            JsonOrient::Records => SplitJsonOrient::Records,
            JsonOrient::Columns => SplitJsonOrient::Columns,
        };

        // Call to_json from SplitDataFrame
        split_df.to_json(path, split_orient)
    }

    /// Infer data type and create the optimal column (internal helper)
    pub(super) fn infer_and_create_column(data: &[String], name: &str) -> Column {
        // Return a string column for empty data
        if data.is_empty() {
            return Column::String(StringColumn::new(Vec::new()));
        }

        // Check for integer values
        let is_int64 = data
            .iter()
            .all(|s| s.parse::<i64>().is_ok() || s.trim().is_empty());

        if is_int64 {
            let int_data: Vec<i64> = data.iter().map(|s| s.parse::<i64>().unwrap_or(0)).collect();
            return Column::Int64(Int64Column::new(int_data));
        }

        // Check for floating-point values
        let is_float64 = data
            .iter()
            .all(|s| s.parse::<f64>().is_ok() || s.trim().is_empty());

        if is_float64 {
            let float_data: Vec<f64> = data
                .iter()
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            return Column::Float64(Float64Column::new(float_data));
        }

        // Check for boolean values
        let bool_values = ["true", "false", "0", "1", "yes", "no", "t", "f"];
        let is_boolean = data
            .iter()
            .all(|s| bool_values.contains(&s.to_lowercase().trim()) || s.trim().is_empty());

        if is_boolean {
            let bool_data: Vec<bool> = data
                .iter()
                .map(|s| {
                    let lower = s.to_lowercase();
                    let trimmed = lower.trim();
                    match trimmed {
                        "true" | "1" | "yes" | "t" => true,
                        "false" | "0" | "no" | "f" => false,
                        _ => false, // Empty string, etc.
                    }
                })
                .collect();
            return Column::Boolean(BooleanColumn::new(bool_data));
        }

        // Handle all other cases as strings
        Column::String(StringColumn::new(data.to_vec()))
    }

    /// Create an OptimizedDataFrame from a standard DataFrame
    pub(super) fn from_standard_dataframe(df: &crate::dataframe::DataFrame) -> Result<Self> {
        // Use functions from the convert module
        crate::optimized::convert::from_standard_dataframe(df)
    }

    /// Create a DataFrame from standard DataFrame (alias for from_standard_dataframe)
    ///
    /// This is a public alias provided for backward compatibility with existing code
    pub fn from_dataframe(df: &crate::dataframe::DataFrame) -> Result<Self> {
        Self::from_standard_dataframe(df)
    }

    /// Convert an OptimizedDataFrame to a standard DataFrame
    pub(super) fn to_standard_dataframe(&self) -> Result<crate::dataframe::DataFrame> {
        // Use functions from the convert module
        crate::optimized::convert::to_standard_dataframe(self)
    }
}
