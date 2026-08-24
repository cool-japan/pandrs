//! Row operations functionality for OptimizedDataFrame

use super::core::OptimizedDataFrame;
use super::select::take_rows;
use crate::column::{Column, ColumnTrait};
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// Filter rows (as a new DataFrame)
    ///
    /// Extracts only rows where the value in the condition column (boolean type) is true.
    /// NULL and `false` are both treated as "not selected", and the values of the
    /// surviving rows keep their NULL status.
    ///
    /// # Arguments
    /// * `condition_column` - Name of the boolean column to use as filter condition
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with filtered rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `filter_rows`.
    pub fn filter_rows(&self, condition_column: &str) -> Result<Self> {
        // Get condition column
        let column_idx = self
            .column_indices
            .get(condition_column)
            .ok_or_else(|| Error::ColumnNotFound(condition_column.to_string()))?;

        let condition = &self.columns[*column_idx];

        // Verify that the condition column is boolean type
        if let Column::Boolean(bool_col) = condition {
            // Collect indices of rows where the value is true
            let mut indices = Vec::new();
            for i in 0..bool_col.len() {
                if let Ok(Some(true)) = bool_col.get(i) {
                    indices.push(i);
                }
            }

            take_rows(self, &indices)
        } else {
            Err(Error::ColumnTypeMismatch {
                name: condition_column.to_string(),
                expected: crate::column::ColumnType::Boolean,
                found: condition.column_type(),
            })
        }
    }

    /// Filter by specified row indices
    ///
    /// # Arguments
    /// * `indices` - Array of row indices to extract
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with filtered rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `filter_rows_by_indices`.
    pub fn filter_rows_by_indices(&self, indices: &[usize]) -> Result<Self> {
        take_rows(self, indices)
    }

    /// Get the first n rows
    ///
    /// # Arguments
    /// * `n` - Number of rows to retrieve
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with the first n rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `head_rows`.
    pub fn head_rows(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let indices: Vec<usize> = (0..n).collect();
        self.filter_rows_by_indices(&indices)
    }

    /// Get the last n rows
    ///
    /// # Arguments
    /// * `n` - Number of rows to retrieve
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with the last n rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `tail_rows`.
    pub fn tail_rows(&self, n: usize) -> Result<Self> {
        let n = std::cmp::min(n, self.row_count);
        let start = self.row_count.saturating_sub(n);
        let indices: Vec<usize> = (start..self.row_count).collect();
        self.filter_rows_by_indices(&indices)
    }

    /// Sample rows from the DataFrame
    ///
    /// # Arguments
    /// * `n` - Number of rows to sample
    /// * `replace` - Whether to sample with replacement
    /// * `seed` - Random seed value (for reproducibility)
    ///
    /// # Returns
    /// * `Result<Self>` - A new DataFrame with sampled rows
    ///
    /// # Note
    /// This function has the same signature as the one in the data operations module,
    /// so the actual implementation is provided as `sample_rows`.
    pub fn sample_rows(&self, n: usize, replace: bool, seed: Option<u64>) -> Result<Self> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::{Rng, RngExt, SeedableRng, SliceRandom};

        if self.row_count == 0 {
            return Ok(Self::new());
        }

        let row_indices: Vec<usize> = (0..self.row_count).collect();

        // Initialize random number generator
        let mut rng = if let Some(seed_val) = seed {
            StdRng::seed_from_u64(seed_val)
        } else {
            // API changed due to dependency updates, so using a method to generate seed
            let mut seed_bytes = [0u8; 32];
            scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
            StdRng::from_seed(seed_bytes)
        };

        // Sample row indices
        let sampled_indices = if replace {
            // Sampling with replacement
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.random_range(0..self.row_count);
                samples.push(idx);
            }
            samples
        } else {
            // Sampling without replacement
            let sample_size = std::cmp::min(n, self.row_count);
            let mut indices_copy = row_indices.clone();
            indices_copy.shuffle(&mut rng);
            indices_copy[0..sample_size].to_vec()
        };

        self.filter_rows_by_indices(&sampled_indices)
    }
}
