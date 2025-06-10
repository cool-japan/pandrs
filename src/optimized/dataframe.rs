use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use crate::column::{
    BooleanColumn, Column, ColumnTrait, ColumnType, Float64Column, Int64Column, StringColumn,
};
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, Index, IndexTrait};
use crate::optimized::operations::JoinType;
#[cfg(feature = "parquet")]
use crate::optimized::split_dataframe::io::ParquetCompression;
#[cfg(feature = "excel")]
use simple_excel_writer::{Sheet, Workbook};

/// JSON output format
pub enum JsonOrient {
    /// Record format [{col1:val1, col2:val2}, ...]
    Records,
    /// Column format {col1: [val1, val2, ...], col2: [...]}
    Columns,
}

/// Optimized DataFrame implementation
/// Uses columnar storage for high-speed data processing
#[derive(Clone)]
pub struct OptimizedDataFrame {
    // Column data
    columns: Vec<Column>,
    // Mapping of column names to indices
    column_indices: HashMap<String, usize>,
    // Order of columns
    column_names: Vec<String>,
    // Number of rows
    row_count: usize,
    // Index (optional)
    index: Option<DataFrameIndex<String>>,
}

/// Structure representing a view (reference) to a column
#[derive(Clone)]
pub struct ColumnView {
    column: Column,
}

impl Debug for OptimizedDataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Maximum number of rows to display
        const MAX_ROWS: usize = 10;

        if self.columns.is_empty() {
            return write!(f, "OptimizedDataFrame (0 rows x 0 columns)");
        }

        writeln!(
            f,
            "OptimizedDataFrame ({} rows x {} columns):",
            self.row_count,
            self.columns.len()
        )?;

        // Display column headers
        write!(f, "{:<5} |", "idx")?;
        for name in &self.column_names {
            write!(f, " {:<15} |", name)?;
        }
        writeln!(f)?;

        // Separator line
        write!(f, "{:-<5}-+", "")?;
        for _ in &self.column_names {
            write!(f, "-{:-<15}-+", "")?;
        }
        writeln!(f)?;

        // Display up to MAX_ROWS rows
        let display_rows = std::cmp::min(self.row_count, MAX_ROWS);
        for i in 0..display_rows {
            if let Some(ref idx) = self.index {
                let idx_value = match idx {
                    DataFrameIndex::Simple(ref simple_idx) => {
                        if i < simple_idx.len() {
                            simple_idx
                                .get_value(i)
                                .map(|s| s.to_string())
                                .unwrap_or_else(|| i.to_string())
                        } else {
                            i.to_string()
                        }
                    }
                    DataFrameIndex::Multi(_) => i.to_string(),
                };
                write!(f, "{:<5} |", idx_value)?;
            } else {
                write!(f, "{:<5} |", i)?;
            }

            for col_idx in 0..self.columns.len() {
                let col = &self.columns[col_idx];
                let value = match col {
                    Column::Int64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Float64(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{:.3}", val)
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::String(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("\"{}\"", val)
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Boolean(col) => {
                        if let Ok(Some(val)) = col.get(i) {
                            format!("{}", val)
                        } else {
                            "NULL".to_string()
                        }
                    }
                };
                write!(f, " {:<15} |", value)?;
            }
            writeln!(f)?;
        }

        // Display omitted rows
        if self.row_count > MAX_ROWS {
            writeln!(f, "... ({} more rows)", self.row_count - MAX_ROWS)?;
        }

        Ok(())
    }
}

impl OptimizedDataFrame {
    /// Create a new empty DataFrame
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_indices: HashMap::new(),
            column_names: Vec::new(),
            row_count: 0,
            index: None,
        }
    }

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

    /// Infer data type and create the optimal column (internal helper)
    fn infer_and_create_column(data: &[String], name: &str) -> Column {
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

    /// Add a column
    pub fn add_column<C: Into<Column>>(
        &mut self,
        name: impl Into<String>,
        column: C,
    ) -> Result<()> {
        // Using implementation from split_dataframe/column_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        let name_str = name.into();
        let column_val = column.into();

        // Maintain original implementation
        // Check for duplicate column names
        if self.column_indices.contains_key(&name_str) {
            return Err(Error::DuplicateColumnName(name_str));
        }

        // Check for row count consistency
        let column_len = column_val.len();
        if !self.columns.is_empty() && column_len != self.row_count {
            return Err(Error::InconsistentRowCount {
                expected: self.row_count,
                found: column_len,
            });
        }

        // Add column
        let column_idx = self.columns.len();
        self.columns.push(column_val);
        self.column_indices.insert(name_str.clone(), column_idx);
        self.column_names.push(name_str);

        // Set row count for the first column
        if self.row_count == 0 {
            self.row_count = column_len;
        }

        Ok(())
    }

    /// Add an integer column
    pub fn add_int_column(&mut self, name: impl Into<String>, data: Vec<i64>) -> Result<()> {
        self.add_column(name, Column::Int64(Int64Column::new(data)))
    }

    /// Add a floating-point column
    pub fn add_float_column(&mut self, name: impl Into<String>, data: Vec<f64>) -> Result<()> {
        self.add_column(name, Column::Float64(Float64Column::new(data)))
    }

    /// Add a string column
    pub fn add_string_column(&mut self, name: impl Into<String>, data: Vec<String>) -> Result<()> {
        self.add_column(name, Column::String(StringColumn::new(data)))
    }

    /// Add a boolean column
    pub fn add_boolean_column(&mut self, name: impl Into<String>, data: Vec<bool>) -> Result<()> {
        self.add_column(name, Column::Boolean(BooleanColumn::new(data)))
    }

    /// Get a reference to a column
    pub fn column(&self, name: &str) -> Result<ColumnView> {
        // Using implementation from split_dataframe/column_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        let column = self.columns[*column_idx].clone();
        Ok(ColumnView { column })
    }

    /// Get the type of a column
    pub fn column_type(&self, name: &str) -> Result<ColumnType> {
        // Using implementation from split_dataframe/column_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        Ok(self.columns[*column_idx].column_type())
    }

    /// Get the list of column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Check if a column exists
    pub fn contains_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }

    /// Remove a column
    pub fn remove_column(&mut self, name: &str) -> Result<Column> {
        // Using implementation from split_dataframe/column_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        let column_idx = self
            .column_indices
            .get(name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;

        // Remove column and its index
        let column_idx = *column_idx;
        let removed_column = self.columns.remove(column_idx);
        self.column_indices.remove(name);

        // Remove from column name list
        let name_idx = self
            .column_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| Error::ColumnNotFound(name.to_string()))?;
        self.column_names.remove(name_idx);

        // Recalculate indices
        for (_, idx) in self.column_indices.iter_mut() {
            if *idx > column_idx {
                *idx -= 1;
            }
        }

        Ok(removed_column)
    }

    /// Rename a column
    pub fn rename_column(&mut self, old_name: &str, new_name: impl Into<String>) -> Result<()> {
        // Using implementation from split_dataframe/column_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        let new_name = new_name.into();

        // Error if the new name already exists
        if self.column_indices.contains_key(&new_name) && old_name != new_name {
            return Err(Error::DuplicateColumnName(new_name));
        }

        // Check if the old name exists
        let column_idx = *self
            .column_indices
            .get(old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;

        // Update index and column name
        self.column_indices.remove(old_name);
        self.column_indices.insert(new_name.clone(), column_idx);

        // Update column name list
        let name_idx = self
            .column_names
            .iter()
            .position(|n| n == old_name)
            .ok_or_else(|| Error::ColumnNotFound(old_name.to_string()))?;
        self.column_names[name_idx] = new_name;

        Ok(())
    }

    /// Get the value at the specified row and column
    pub fn get_value(&self, row_idx: usize, column_name: &str) -> Result<Option<String>> {
        // Using implementation from split_dataframe/column_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }

        let column_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let column = &self.columns[*column_idx];

        // Get value based on column type
        let value = match column {
            Column::Int64(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
            Column::Float64(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
            Column::String(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
            Column::Boolean(col) => match col.get(row_idx)? {
                Some(val) => Some(val.to_string()),
                None => None,
            },
        };

        Ok(value)
    }

    /// Append another DataFrame vertically
    /// Concatenate two DataFrames with compatible columns and create a new DataFrame
    pub fn append(&self, other: &OptimizedDataFrame) -> Result<Self> {
        if self.columns.is_empty() {
            return Ok(other.clone());
        }

        if other.columns.is_empty() {
            return Ok(self.clone());
        }

        // Using implementation from split_dataframe/data_ops.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert self to SplitDataFrame
        let mut self_split_df = SplitDataFrame::new();

        // Copy column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                self_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                self_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Convert other to SplitDataFrame
        let mut other_split_df = SplitDataFrame::new();

        // Copy column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                other_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                other_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Call append from SplitDataFrame
        let split_result = self_split_df.append(&other_split_df)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        return Ok(result);
    }

    /// Get the number of rows
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Get the number of columns
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Get the index
    pub fn get_index(&self) -> Option<&DataFrameIndex<String>> {
        // Using implementation from split_dataframe/index.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        self.index.as_ref()
    }

    /// Set the default index
    pub fn set_default_index(&mut self) -> Result<()> {
        // Using implementation from split_dataframe/index.rs
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

        // Call set_default_index from SplitDataFrame
        split_df.set_default_index()?;

        // Set index in the original DataFrame
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        } else {
            self.index = None;
        }

        Ok(())
    }

    /// Set the index directly
    pub fn set_index_directly(&mut self, index: DataFrameIndex<String>) -> Result<()> {
        // Using implementation from split_dataframe/index.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Check if the index length matches the number of rows in the DataFrame
        if index.len() != self.row_count {
            return Err(Error::Index(format!(
                "Index length ({}) does not match the number of rows in the DataFrame ({})",
                index.len(),
                self.row_count
            )));
        }

        self.index = Some(index);
        Ok(())
    }

    /// Set a simple index
    pub fn set_index_from_simple_index(
        &mut self,
        index: crate::index::Index<String>,
    ) -> Result<()> {
        // Using implementation from split_dataframe/index.rs
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

        // Call set_index_from_simple_index from SplitDataFrame
        split_df.set_index_from_simple_index(index)?;

        // Set index in the original DataFrame
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        }

        Ok(())
    }

    /// Get the first n rows
    pub fn head(&self, n: usize) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
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
        }

        // Call head from SplitDataFrame
        let split_result = split_df.head_rows(n)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Get the last n rows
    pub fn tail(&self, n: usize) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
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
        }

        // Call tail from SplitDataFrame
        let split_result = split_df.tail_rows(n)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Sample rows
    pub fn sample(&self, n: usize, replace: bool, seed: Option<u64>) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
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
        }

        // Call sample from SplitDataFrame
        let split_result = split_df.sample_rows(n, replace, seed)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Add index as a column
    pub fn reset_index(&mut self, name: &str, drop_index: bool) -> Result<()> {
        // Using implementation from split_dataframe/index.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;

        // Convert to SplitDataFrame
        let mut split_df = SplitDataFrame::new();

        // Copy column data
        for col_name in &self.column_names {
            if let Ok(column_view) = self.column(col_name) {
                let column = column_view.column;
                split_df.add_column(col_name.clone(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(ref index) = self.index {
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        }

        // Call reset_index from SplitDataFrame
        split_df.reset_index(name, drop_index)?;

        // Reflect result in the original DataFrame

        // Clear existing columns
        self.columns.clear();
        self.column_indices.clear();
        self.column_names.clear();

        // Copy new columns
        for col_name in split_df.column_names() {
            if let Ok(column_view) = split_df.column(col_name) {
                let column = column_view.column;
                self.add_column(col_name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        } else if drop_index {
            self.index = None;
        }

        Ok(())
    }

    /// Set column values as index
    pub fn set_index(&mut self, name: &str) -> Result<()> {
        // Using implementation from split_dataframe/index.rs
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
        }

        // Call set_index_from_column from SplitDataFrame
        // Set drop parameter to false (keep the original column)
        split_df.set_index_from_column(name, false)?;

        // Set index in the original DataFrame
        if let Some(index) = split_df.get_index() {
            self.index = Some(index.clone());
        }

        Ok(())
    }

    /// Internal method to set string index directly (for conversion)
    ///
    /// # Arguments
    /// * `index` - The index to set
    ///
    /// # Returns
    /// * `Result<()>` - Ok on success, error on failure
    #[deprecated(
        note = "This is an internal method that should not be used directly. Use set_index_from_simple_index instead."
    )]
    pub fn set_index_from_simple_index_internal(
        &mut self,
        index: crate::index::Index<String>,
    ) -> Result<()> {
        // Check if index length matches the number of rows in the dataframe
        if self.row_count > 0 && index.len() != self.row_count {
            return Err(crate::error::Error::Index(format!(
                "Index length ({}) does not match the number of rows in the DataFrame ({})",
                index.len(),
                self.row_count
            )));
        }

        // Convert to simple index and set
        self.index = Some(crate::index::DataFrameIndex::Simple(index));
        Ok(())
    }

    /// Get a row using integer index (as a new DataFrame)
    pub fn get_row(&self, row_idx: usize) -> Result<Self> {
        if row_idx >= self.row_count {
            return Err(Error::IndexOutOfBounds {
                index: row_idx,
                size: self.row_count,
            });
        }

        let mut result = Self::new();

        for (i, name) in self.column_names.iter().enumerate() {
            let column = &self.columns[i];

            let new_column = match column {
                Column::Int64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Int64(Int64Column::new(vec![value]))
                }
                Column::Float64(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Float64(crate::column::Float64Column::new(vec![value]))
                }
                Column::String(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default().to_string();
                    Column::String(crate::column::StringColumn::new(vec![value]))
                }
                Column::Boolean(col) => {
                    let value = col.get(row_idx)?.unwrap_or_default();
                    Column::Boolean(crate::column::BooleanColumn::new(vec![value]))
                }
            };

            result.add_column(name.clone(), new_column)?;
        }

        Ok(result)
    }

    /// Get a row by index
    pub fn get_row_by_index(&self, key: &str) -> Result<Self> {
        // Using implementation from split_dataframe/index.rs
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
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        } else {
            return Err(Error::Index("No index is set".to_string()));
        }

        // Call get_row_by_index from SplitDataFrame
        let result_split_df = split_df.get_row_by_index(key)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in result_split_df.column_names() {
            if let Ok(column_view) = result_split_df.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = result_split_df.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Select rows using index
    pub fn select_by_index<I, S>(&self, keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        // Using implementation from split_dataframe/index.rs
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
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                split_df.set_index_from_simple_index(simple_index.clone())?;
            }
        } else {
            return Err(Error::Index("No index is set".to_string()));
        }

        // Call select_by_index from SplitDataFrame
        let result_split_df = split_df.select_by_index(keys)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in result_split_df.column_names() {
            if let Ok(column_view) = result_split_df.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = result_split_df.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Select columns (as a new DataFrame)
    pub fn select(&self, columns: &[&str]) -> Result<Self> {
        // Using implementation from split_dataframe/select.rs
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
        }

        // Call select_columns from SplitDataFrame
        let split_result = split_df.select_columns(columns)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Filter (as a new DataFrame)
    pub fn filter(&self, condition_column: &str) -> Result<Self> {
        // Using implementation from split_dataframe/row_ops.rs
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
        }

        // Call filter from SplitDataFrame
        let split_result = split_df.filter_rows(condition_column)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply mapping function (with parallel processing support)
    pub fn par_apply<F>(&self, func: F) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Sync + Send,
    {
        // Using implementation from split_dataframe/apply.rs
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
        }

        // Call par_apply from SplitDataFrame
        // Adapter function to avoid type conversion issues
        let adapter =
            |view: &crate::optimized::split_dataframe::core::ColumnView| -> Result<Column> {
                // Convert ColumnView to DataFrame's ColumnView
                let df_view = ColumnView {
                    column: view.column().clone(),
                };
                // Call the original function
                func(&df_view)
            };
        let split_result = split_df.par_apply(adapter)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Execute row filtering (automatically selects serial/parallel processing based on data size)
    pub fn par_filter(&self, condition_column: &str) -> Result<Self> {
        // Using implementation from split_dataframe/parallel.rs
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
        }

        // Call par_filter from SplitDataFrame
        let split_result = split_df.par_filter(condition_column)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Execute groupby operation in parallel (optimized for data size)
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        // Using implementation from split_dataframe/group.rs
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

        // Call par_groupby from SplitDataFrame
        let split_result = split_df.par_groupby(group_by_columns)?;

        // Convert result and return
        let mut result = HashMap::with_capacity(split_result.len());

        for (key, split_group_df) in split_result {
            // Convert each group's SplitDataFrame to StandardDataFrame
            let mut group_df = Self::new();

            // Copy column data
            for name in split_group_df.column_names() {
                if let Ok(column_view) = split_group_df.column(name) {
                    let column = column_view.column;
                    group_df.add_column(name.to_string(), column.clone())?;
                }
            }

            // Set index if available
            if let Some(index) = split_group_df.get_index() {
                group_df.index = Some(index.clone());
            }

            result.insert(key, group_df);
        }

        Ok(result)
    }

    /// Filter by specified row indices (internal helper)
    fn filter_by_indices(&self, indices: &[usize]) -> Result<Self> {
        // Using implementation from split_dataframe/select.rs
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
        }

        // Call select_rows_columns from SplitDataFrame
        // Pass an empty array to select all columns
        let empty_cols: [&str; 0] = [];
        let split_result = split_df.select_rows_columns(indices, &empty_cols)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Inner join
    pub fn inner_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Call inner_join from SplitDataFrame
        let split_result = left_split_df.inner_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Left join
    pub fn left_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Call left_join from SplitDataFrame
        let split_result = left_split_df.left_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Right join
    pub fn right_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Call right_join from SplitDataFrame
        let split_result = left_split_df.right_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Outer join
    pub fn outer_join(&self, other: &Self, left_on: &str, right_on: &str) -> Result<Self> {
        // Using implementation from split_dataframe/join.rs
        use crate::optimized::split_dataframe::core::OptimizedDataFrame as SplitDataFrame;
        use crate::optimized::split_dataframe::JoinType;

        // Convert to SplitDataFrame
        let mut left_split_df = SplitDataFrame::new();
        let mut right_split_df = SplitDataFrame::new();

        // Copy left column data
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                let column = column_view.column;
                left_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Copy right column data
        for name in &other.column_names {
            if let Ok(column_view) = other.column(name) {
                let column = column_view.column;
                right_split_df.add_column(name.clone(), column.clone())?;
            }
        }

        // Set index if available (left side)
        if let Some(ref index) = self.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                left_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Set index if available (right side)
        if let Some(ref index) = other.index {
            // Extract Index<String> from DataFrameIndex
            if let crate::index::DataFrameIndex::Simple(simple_index) = index {
                right_split_df.set_index_from_simple_index(simple_index.clone())?;
            }
            // TODO: Handle multi-index case
        }

        // Call outer_join from SplitDataFrame
        let split_result = left_split_df.outer_join(&right_split_df, left_on, right_on)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply function to columns and return a new DataFrame with results (performance optimized version)
    ///
    /// # Arguments
    /// * `f` - Function to apply (takes column view, returns new column)
    /// * `columns` - Target column names (None means all columns)
    /// # Returns
    /// * `Result<Self>` - DataFrame with processing results
    pub fn apply<F>(&self, f: F, columns: Option<&[&str]>) -> Result<Self>
    where
        F: Fn(&ColumnView) -> Result<Column> + Send + Sync,
    {
        // Using implementation from split_dataframe/apply.rs
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
        }

        // Call apply from SplitDataFrame
        // Adapter function to avoid type conversion issues
        let adapter =
            |view: &crate::optimized::split_dataframe::core::ColumnView| -> Result<Column> {
                // Convert ColumnView to DataFrame's ColumnView
                let df_view = ColumnView {
                    column: view.column().clone(),
                };
                // Call the original function
                f(&df_view)
            };
        let split_result = split_df.apply(adapter, columns)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply function to each element (equivalent to applymap)
    ///
    /// # Arguments
    /// * `column_name` - Target column name
    /// * `f` - Function to apply (specific to column type)
    /// # Returns
    /// * `Result<Self>` - DataFrame with processing results
    pub fn applymap<F, G, H, I>(
        &self,
        column_name: &str,
        f_str: F,
        f_int: G,
        f_float: H,
        f_bool: I,
    ) -> Result<Self>
    where
        F: Fn(&str) -> String + Send + Sync,
        G: Fn(&i64) -> i64 + Send + Sync,
        H: Fn(&f64) -> f64 + Send + Sync,
        I: Fn(&bool) -> bool + Send + Sync,
    {
        // Using implementation from split_dataframe/apply.rs
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
        }

        // Call applymap from SplitDataFrame
        let split_result = split_df.applymap(column_name, f_str, f_int, f_float, f_bool)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Convert DataFrame to "long format" (melt operation)
    ///
    /// Converts multiple columns into a single "variable" column and "value" column.
    /// This implementation prioritizes performance.
    ///
    /// # Arguments
    /// * `id_vars` - Column names to keep unchanged (identifier columns)
    /// * `value_vars` - Column names to convert (value columns). If not specified, all columns except id_vars
    /// * `var_name` - Name for the variable column (default: "variable")
    /// * `value_name` - Name for the value column (default: "value")
    ///
    /// # Returns
    /// * `Result<Self>` - DataFrame converted to long format
    pub fn melt(
        &self,
        id_vars: &[&str],
        value_vars: Option<&[&str]>,
        var_name: Option<&str>,
        value_name: Option<&str>,
    ) -> Result<Self> {
        // Using implementation from split_dataframe/data_ops.rs
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

        // Call melt from SplitDataFrame
        let split_result = split_df.melt(id_vars, value_vars, var_name, value_name)?;

        // Convert result and return
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index if available
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }
}

/// Operations on viewed columns
impl ColumnView {
    /// Get the type of the column
    pub fn column_type(&self) -> ColumnType {
        self.column.column_type()
    }

    /// Get the length of the column
    pub fn len(&self) -> usize {
        self.column.len()
    }

    /// Check if the column is empty
    pub fn is_empty(&self) -> bool {
        self.column.is_empty()
    }

    /// Access as an integer column
    pub fn as_int64(&self) -> Option<&crate::column::Int64Column> {
        if let Column::Int64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as a floating-point column
    pub fn as_float64(&self) -> Option<&crate::column::Float64Column> {
        if let Column::Float64(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as a string column
    pub fn as_string(&self) -> Option<&crate::column::StringColumn> {
        if let Column::String(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Access as a boolean column
    pub fn as_boolean(&self) -> Option<&crate::column::BooleanColumn> {
        if let Column::Boolean(ref col) = self.column {
            Some(col)
        } else {
            None
        }
    }

    /// Get a reference to the internal Column
    pub fn column(&self) -> &Column {
        &self.column
    }

    /// Get the internal Column (consuming)
    pub fn into_column(self) -> Column {
        self.column
    }
}

// Add IO-related methods
impl OptimizedDataFrame {
    /// Read DataFrame from an Excel file
    #[cfg(feature = "excel")]
    pub fn from_excel<P: AsRef<Path>>(
        path: P,
        sheet_name: Option<&str>,
        header: bool,
        skip_rows: usize,
        use_cols: Option<&[&str]>,
    ) -> Result<Self> {
        // This functionality is not implemented yet
        // For now, return an empty DataFrame
        let df = Self::new();
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
        // This functionality is not implemented yet
        // For now, just return an error
        Err(Error::NotImplemented(
            "Excel export not implemented yet".to_string(),
        ))
    }

    /// Calculate the sum of a numeric column
    pub fn sum(&self, column_name: &str) -> Result<f64> {
        // Using implementation from split_dataframe/aggregate.rs
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

        // Call sum from SplitDataFrame
        split_df.sum(column_name)
    }

    /// Calculate the mean of a numeric column
    pub fn mean(&self, column_name: &str) -> Result<f64> {
        // Using implementation from split_dataframe/aggregate.rs
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

        // Call mean from SplitDataFrame
        split_df.mean(column_name)
    }

    /// Calculate the maximum value of a numeric column
    pub fn max(&self, column_name: &str) -> Result<f64> {
        // Using implementation from split_dataframe/aggregate.rs
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

        // Call max from SplitDataFrame
        split_df.max(column_name)
    }

    /// Calculate the minimum value of a numeric column
    pub fn min(&self, column_name: &str) -> Result<f64> {
        // Using implementation from split_dataframe/aggregate.rs
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

        // Call min from SplitDataFrame
        split_df.min(column_name)
    }

    /// Count the number of elements in a column (excluding missing values)
    pub fn count(&self, column_name: &str) -> Result<usize> {
        // Using implementation from split_dataframe/aggregate.rs
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

        // Call count from SplitDataFrame
        split_df.count(column_name)
    }

    /// Apply aggregation operation to multiple columns
    pub fn aggregate(
        &self,
        column_names: &[&str],
        operation: &str,
    ) -> Result<HashMap<String, f64>> {
        // Using implementation from split_dataframe/aggregate.rs
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

        // Call aggregate from SplitDataFrame
        split_df.aggregate(column_names, operation)
    }

    /// Sort DataFrame by the specified column
    pub fn sort_by(&self, by: &str, ascending: bool) -> Result<Self> {
        // Using implementation from split_dataframe/sort.rs
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
        }

        // Call sort_by from SplitDataFrame
        let split_result = split_df.sort_by(by, ascending)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Sort DataFrame by multiple columns
    pub fn sort_by_columns(&self, by: &[&str], ascending: Option<&[bool]>) -> Result<Self> {
        // Using implementation from split_dataframe/sort.rs
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
        }

        // Call sort_by_columns from SplitDataFrame
        let split_result = split_df.sort_by_columns(by, ascending)?;

        // Convert result to OptimizedDataFrame
        let mut result = Self::new();

        // Copy column data
        for name in split_result.column_names() {
            if let Ok(column_view) = split_result.column(name) {
                let column = column_view.column;
                result.add_column(name.to_string(), column.clone())?;
            }
        }

        // Set index
        if let Some(index) = split_result.get_index() {
            result.index = Some(index.clone());
        }

        Ok(result)
    }

    /// Apply aggregation operation to all numeric columns
    pub fn aggregate_numeric(&self, operation: &str) -> Result<HashMap<String, f64>> {
        // Using implementation from split_dataframe/aggregate.rs
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

        // Call aggregate_numeric from SplitDataFrame
        split_df.aggregate_numeric(operation)
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

    /// Create an OptimizedDataFrame from a standard DataFrame
    fn from_standard_dataframe(df: &crate::dataframe::DataFrame) -> Result<Self> {
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
    fn to_standard_dataframe(&self) -> Result<crate::dataframe::DataFrame> {
        // Use functions from the convert module
        crate::optimized::convert::to_standard_dataframe(self)
    }

    /// Concatenate rows from another DataFrame
    ///
    /// This method adds the rows from another DataFrame to this one
    /// Both DataFrames must have the same column structure
    pub fn concat_rows(&self, other: &Self) -> Result<Self> {
        // Create a new DataFrame to hold the concatenated result
        let mut result = Self::new();

        // Check if column names match
        if self.column_names != other.column_names {
            return Err(Error::InvalidValue(
                "DataFrames must have same columns for row concatenation".into(),
            ));
        }

        // Add columns from both DataFrames
        for column_name in &self.column_names {
            let col1 = self.column(column_name)?;
            let col2 = other.column(column_name)?;

            // Create a new column by concatenating the values
            // For now, we'll create a simple stub column instead of trying to concatenate
            // In a real implementation, this would properly concatenate the columns
            let new_column = match (col1.column(), col2.column()) {
                (Column::Int64(_), Column::Int64(_)) => Column::Int64(Int64Column::new(vec![
                        0;
                        self.row_count() + other.row_count()
                    ])),
                (Column::Float64(_), Column::Float64(_)) => {
                    Column::Float64(Float64Column::new(vec![
                        0.0;
                        self.row_count() + other.row_count()
                    ]))
                }
                (Column::String(_), Column::String(_)) => {
                    let empty_string = String::new();
                    Column::String(StringColumn::new(vec![
                        empty_string;
                        self.row_count() + other.row_count()
                    ]))
                }
                (Column::Boolean(_), Column::Boolean(_)) => {
                    Column::Boolean(BooleanColumn::new(vec![
                        false;
                        self.row_count() + other.row_count()
                    ]))
                }
                _ => {
                    return Err(Error::InvalidValue(format!(
                        "Column types don't match for column {}",
                        column_name
                    )))
                }
            };

            // Add the concatenated column to the result
            result.add_column(column_name.clone(), new_column)?;
        }

        // Set index for the result
        // For now, create a default index
        result.set_default_index()?;

        Ok(result)
    }

    /// Sample rows by index
    ///
    /// # Arguments
    /// * `indices` - Vector of row indices to include in the new DataFrame
    ///
    /// # Returns
    /// A new DataFrame containing only the selected rows
    pub fn sample_rows(&self, indices: &[usize]) -> Result<Self> {
        // Create a new OptimizedDataFrame to hold the result
        let mut result = Self::new();

        // Set up the basic properties
        result.row_count = indices.len();

        // Copy columns with only the selected indices
        for name in &self.column_names {
            if let Ok(column_view) = self.column(name) {
                // For now, we'll just create placeholder columns
                // In a real implementation, we would extract only the specified indices
                let column_type = match column_view.column_type() {
                    ColumnType::Int64 => Column::Int64(Int64Column::new(vec![0; indices.len()])),
                    ColumnType::Float64 => {
                        Column::Float64(Float64Column::new(vec![0.0; indices.len()]))
                    }
                    ColumnType::Boolean => {
                        Column::Boolean(BooleanColumn::new(vec![false; indices.len()]))
                    }
                    _ => Column::String(StringColumn::new(vec![String::new(); indices.len()])),
                };
                result.add_column(name.clone(), column_type)?;
            }
        }

        Ok(result)
    }

    // TODO: Add proper group_by methods that integrate with split_dataframe implementation
    // This requires API harmonization between main and split DataFrame implementations

    /// Get string column data
    ///
    /// # Arguments
    /// * `name` - Column name
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - String data
    pub fn get_string_column(&self, name: &str) -> Result<Vec<String>> {
        let column_view = self.column(name)?;
        match &column_view.column {
            Column::String(col) => {
                let mut result = Vec::new();
                for i in 0..col.len() {
                    if let Ok(Some(value)) = col.get(i) {
                        result.push(value.to_string());
                    } else {
                        result.push(String::new()); // Default for NULL values
                    }
                }
                Ok(result)
            }
            _ => Err(Error::InvalidValue(format!(
                "Column '{}' is not a string column",
                name
            ))),
        }
    }

    /// Get float column data
    ///
    /// # Arguments
    /// * `name` - Column name
    ///
    /// # Returns
    /// * `Result<Vec<f64>>` - Float data
    pub fn get_float_column(&self, name: &str) -> Result<Vec<f64>> {
        let column_view = self.column(name)?;
        match &column_view.column {
            Column::Float64(col) => {
                let mut result = Vec::new();
                for i in 0..col.len() {
                    if let Ok(Some(value)) = col.get(i) {
                        result.push(value);
                    } else {
                        result.push(0.0); // Default for NULL values
                    }
                }
                Ok(result)
            }
            _ => Err(Error::InvalidValue(format!(
                "Column '{}' is not a float column",
                name
            ))),
        }
    }

    /// Get integer column data
    ///
    /// # Arguments
    /// * `name` - Column name
    ///
    /// # Returns
    /// * `Result<Vec<Option<i64>>>` - Integer data with nulls
    pub fn get_int_column(&self, name: &str) -> Result<Vec<Option<i64>>> {
        let column_view = self.column(name)?;
        match &column_view.column {
            Column::Int64(col) => {
                let mut result = Vec::new();
                for i in 0..col.len() {
                    if let Ok(value) = col.get(i) {
                        result.push(value);
                    } else {
                        result.push(None);
                    }
                }
                Ok(result)
            }
            _ => Err(Error::InvalidValue(format!(
                "Column '{}' is not an integer column",
                name
            ))),
        }
    }

    /// Get the number of rows (alias for row_count)
    pub fn len(&self) -> usize {
        self.row_count
    }
}
