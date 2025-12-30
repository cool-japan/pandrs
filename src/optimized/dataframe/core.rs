//! Core data structures and basic functionality for OptimizedDataFrame

use std::collections::HashMap;
use std::fmt::{self, Debug};

use crate::column::{Column, ColumnTrait, ColumnType};
use crate::error::{Error, Result};
use crate::index::DataFrameIndex;

/// JSON output format
#[derive(Debug, Clone)]
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
    pub(super) columns: Vec<Column>,
    // Mapping of column names to indices
    pub(super) column_indices: HashMap<String, usize>,
    // Order of columns
    pub(super) column_names: Vec<String>,
    // Number of rows
    pub(super) row_count: usize,
    // Index (optional)
    pub(super) index: Option<DataFrameIndex<String>>,
}

/// Structure representing a view (reference) to a column
#[derive(Clone)]
pub struct ColumnView {
    pub(super) column: Column,
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

    /// Get the number of rows in the DataFrame
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Get the number of columns in the DataFrame
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }

    /// Check if the DataFrame is empty
    pub fn is_empty(&self) -> bool {
        self.row_count == 0 || self.columns.is_empty()
    }

    /// Get the column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Check if a column exists
    pub fn contains_column(&self, name: &str) -> bool {
        self.column_indices.contains_key(name)
    }
}

impl Default for OptimizedDataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnView {
    /// Get the column reference
    pub fn column(&self) -> &Column {
        &self.column
    }

    /// Get the column type
    pub fn column_type(&self) -> ColumnType {
        match &self.column {
            Column::Int64(_) => ColumnType::Int64,
            Column::Float64(_) => ColumnType::Float64,
            Column::String(_) => ColumnType::String,
            Column::Boolean(_) => ColumnType::Boolean,
        }
    }

    /// Get the length of the column
    pub fn len(&self) -> usize {
        self.column.len()
    }

    /// Check if the column is empty
    pub fn is_empty(&self) -> bool {
        self.column.len() == 0
    }
}
