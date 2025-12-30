//! Index operation functionality for OptimizedDataFrame

use super::core::OptimizedDataFrame;
use crate::error::{Error, Result};
use crate::index::{DataFrameIndex, Index, IndexTrait};

impl OptimizedDataFrame {
    /// Set index
    ///
    /// # Arguments
    /// * `index` - New index
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn set_index(&mut self, index: DataFrameIndex<String>) -> Result<()> {
        // Check if index length matches DataFrame row count
        if index.len() != self.row_count {
            return Err(Error::Index(format!(
                "Index length ({}) does not match DataFrame row count ({})",
                index.len(),
                self.row_count
            )));
        }

        self.index = Some(index);
        Ok(())
    }

    /// Get index
    ///
    /// # Returns
    /// * `Option<&DataFrameIndex<String>>` - Some if index exists, None otherwise
    pub fn get_index(&self) -> Option<&DataFrameIndex<String>> {
        self.index.as_ref()
    }

    /// Create and set default index
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn set_default_index(&mut self) -> Result<()> {
        if self.row_count == 0 {
            self.index = None;
            return Ok(());
        }

        let index = DataFrameIndex::<String>::default_with_len(self.row_count)?;
        self.index = Some(index);
        Ok(())
    }

    /// Set column as index
    ///
    /// # Arguments
    /// * `column_name` - Name of column to set as index
    /// * `drop` - Whether to drop the original column
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise

    /// Set string index (from existing index object)
    ///
    /// # Arguments
    /// * `index` - Index to set
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn set_index_from_simple_index(
        &mut self,
        index: crate::index::Index<String>,
    ) -> Result<()> {
        // Check index length
        if !self.column_names.is_empty() && index.len() != self.row_count {
            return Err(Error::Consistency(format!(
                "New index length ({}) does not match DataFrame row count ({})",
                index.len(),
                self.row_count
            )));
        }

        // Set index
        self.index = Some(DataFrameIndex::<String>::from_simple(index));

        Ok(())
    }

    /// Set multi-index (from existing multi-index object)
    ///
    /// # Arguments
    /// * `index` - Multi-index to set
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn set_index_from_multi_index(
        &mut self,
        index: crate::index::MultiIndex<String>,
    ) -> Result<()> {
        // Check index length
        if !self.column_names.is_empty() && index.len() != self.row_count {
            return Err(Error::Consistency(format!(
                "New multi-index length ({}) does not match DataFrame row count ({})",
                index.len(),
                self.row_count
            )));
        }

        // Set index
        self.index = Some(DataFrameIndex::<String>::from_multi(index));

        Ok(())
    }
    pub fn set_index_from_column(&mut self, column_name: &str, drop: bool) -> Result<()> {
        // Check if column exists
        let col_idx = self
            .column_indices
            .get(column_name)
            .ok_or_else(|| Error::ColumnNotFound(column_name.to_string()))?;

        let col = &self.columns[*col_idx];

        // Get column values converted to String
        let values: Vec<String> = (0..self.row_count)
            .filter_map(|i| match col {
                crate::column::Column::Int64(c) => {
                    if let Ok(Some(v)) = c.get(i) {
                        Some(v.to_string())
                    } else {
                        Some("NULL".to_string())
                    }
                }
                crate::column::Column::Float64(c) => {
                    if let Ok(Some(v)) = c.get(i) {
                        Some(v.to_string())
                    } else {
                        Some("NULL".to_string())
                    }
                }
                crate::column::Column::String(c) => {
                    if let Ok(Some(v)) = c.get(i) {
                        Some(v.to_string())
                    } else {
                        Some("NULL".to_string())
                    }
                }
                crate::column::Column::Boolean(c) => {
                    if let Ok(Some(v)) = c.get(i) {
                        Some(v.to_string())
                    } else {
                        Some("NULL".to_string())
                    }
                }
            })
            .collect();

        // Create index that allows duplicates (using vector, not map)
        let name = Some(column_name.to_string());
        let index = Index::with_name(values, name)?;
        self.index = Some(DataFrameIndex::Simple(index));

        // Whether to drop the original column
        if drop {
            // Column deletion functionality needs to be implemented
            // self.drop_column(column_name)?;
            return Err(Error::NotImplemented(
                "Column deletion functionality not yet implemented".to_string(),
            ));
        }

        Ok(())
    }

    /// Add index as column
    ///
    /// # Arguments
    /// * `name` - Name of new column
    /// * `drop_index` - Whether to drop the index
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn reset_index(&mut self, name: &str, drop_index: bool) -> Result<()> {
        if let Some(ref index) = self.index {
            let values = match index {
                DataFrameIndex::Simple(idx) => idx.values().iter().map(|v| v.clone()).collect(),
                DataFrameIndex::Multi(midx) => {
                    // For multi-index, join values as strings
                    midx.tuples().iter().map(|tuple| tuple.join(", ")).collect()
                }
            };

            // Add index as a new column
            let col = crate::column::Column::String(crate::column::StringColumn::new(values));
            self.add_column(name.to_string(), col)?;

            // Drop index if requested
            if drop_index {
                self.index = None;
            }

            Ok(())
        } else {
            // If no index exists, add default index as column
            let values: Vec<String> = (0..self.row_count).map(|i| i.to_string()).collect();

            let col = crate::column::Column::String(crate::column::StringColumn::new(values));
            self.add_column(name.to_string(), col)?;

            Ok(())
        }
    }

    /// Get row by index
    ///
    /// # Arguments
    /// * `key` - Index value to search for
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame containing the matching row
    pub fn get_row_by_index(&self, key: &str) -> Result<Self> {
        if let Some(ref index) = self.index {
            let pos = match index {
                DataFrameIndex::Simple(idx) => idx.get_loc(&key.to_string()),
                DataFrameIndex::Multi(_) => None, // Multi-index not currently supported
            };

            if let Some(row_idx) = pos {
                // Extract row using row index
                // Create DataFrame with 1 row
                let indices = vec![row_idx];
                self.filter_by_indices(&indices)
            } else {
                Err(Error::Index(format!("Index '{}' not found", key)))
            }
        } else {
            Err(Error::Index("No index is set".to_string()))
        }
    }

    /// Select rows by index
    ///
    /// # Arguments
    /// * `keys` - List of index values to select
    ///
    /// # Returns
    /// * `Result<Self>` - New DataFrame with selected rows
    pub fn select_by_index<I, S>(&self, keys: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if self.index.is_none() {
            return Err(Error::Index("No index is set".to_string()));
        }

        let index = self.index.as_ref().unwrap();

        // Get row numbers from index
        let mut indices = Vec::new();

        for key in keys {
            let key_str = key.as_ref().to_string();
            let pos = match index {
                DataFrameIndex::Simple(idx) => idx.get_loc(&key_str),
                DataFrameIndex::Multi(_) => None, // Multi-index not currently supported
            };

            if let Some(idx) = pos {
                indices.push(idx);
            } else {
                return Err(Error::Index(format!("Index '{}' not found", key.as_ref())));
            }
        }

        // Select rows using row indices
        self.filter_by_indices(&indices)
    }
}
