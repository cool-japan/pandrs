//! Group creation logic and parallel grouping operations

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use super::super::core::OptimizedDataFrame;
use super::types::GroupBy;
use crate::column::Column;
use crate::error::{Error, Result};

impl OptimizedDataFrame {
    /// Group DataFrame
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by<I, S>(&self, columns: I) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.group_by_with_options(columns, true)
    }

    /// Group DataFrame with options
    ///
    /// # Arguments
    /// * `columns` - Column names for grouping
    /// * `as_multi_index` - Whether to create a multi-index for the result (when multiple columns)
    ///
    /// # Returns
    /// * `Result<GroupBy>` - Grouping results
    pub fn group_by_with_options<I, S>(
        &self,
        columns: I,
        as_multi_index: bool,
    ) -> Result<GroupBy<'_>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let group_by_columns: Vec<String> = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();

        // Verify existence of each column
        for column in &group_by_columns {
            if !self.column_indices.contains_key(column) {
                return Err(Error::ColumnNotFound(column.clone()));
            }
        }

        // Create grouping keys
        let mut groups: HashMap<Vec<String>, Vec<usize>> = HashMap::new();

        for row_idx in 0..self.row_count {
            let mut key = Vec::with_capacity(group_by_columns.len());

            for col_name in &group_by_columns {
                let col_idx = self.column_indices[col_name];
                let col = &self.columns[col_idx];

                let key_part = match col {
                    Column::Int64(int_col) => {
                        if let Ok(Some(val)) = int_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Float64(float_col) => {
                        if let Ok(Some(val)) = float_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::String(str_col) => {
                        if let Ok(Some(val)) = str_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                    Column::Boolean(bool_col) => {
                        if let Ok(Some(val)) = bool_col.get(row_idx) {
                            val.to_string()
                        } else {
                            "NULL".to_string()
                        }
                    }
                };

                key.push(key_part);
            }

            groups.entry(key).or_default().push(row_idx);
        }

        // Only use multi-index when we have multiple grouping columns and option is enabled
        let create_multi_index = as_multi_index && group_by_columns.len() > 1;

        Ok(GroupBy {
            df: self,
            group_by_columns,
            groups,
            create_multi_index,
        })
    }

    /// Group DataFrame using parallel processing
    ///
    /// # Arguments
    /// * `group_by_columns` - Column names for grouping
    ///
    /// # Returns
    /// * `Result<HashMap<String, Self>>` - Grouping results (map of keys and DataFrames)
    pub fn par_groupby(&self, group_by_columns: &[&str]) -> Result<HashMap<String, Self>> {
        use rayon::prelude::*;
        use std::collections::hash_map::Entry;
        use std::sync::{Arc, Mutex};

        // Optimization threshold based on data size
        const PARALLEL_THRESHOLD: usize = 50_000;

        // Get column indices for grouping keys
        let mut group_col_indices = Vec::with_capacity(group_by_columns.len());
        for &col_name in group_by_columns {
            let col_idx = self
                .column_indices
                .get(col_name)
                .ok_or_else(|| Error::ColumnNotFound(col_name.to_string()))?;
            group_col_indices.push(*col_idx);
        }

        // Generate group keys and group each row's index
        let groups: HashMap<String, Vec<usize>> = if self.row_count < PARALLEL_THRESHOLD {
            // Serial processing is more efficient for small data
            let mut groups = HashMap::new();

            for row_idx in 0..self.row_count {
                // Generate group key for this row
                let mut key_parts = Vec::with_capacity(group_col_indices.len());

                for &col_idx in &group_col_indices {
                    let column = &self.columns[col_idx];
                    let part = match column {
                        Column::Int64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                        Column::Float64(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                        Column::String(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                        Column::Boolean(col) => {
                            if let Ok(Some(val)) = col.get(row_idx) {
                                val.to_string()
                            } else {
                                "NA".to_string()
                            }
                        }
                    };
                    key_parts.push(part);
                }

                let group_key = key_parts.join("_");

                match groups.entry(group_key) {
                    Entry::Vacant(e) => {
                        e.insert(vec![row_idx]);
                    }
                    Entry::Occupied(mut e) => {
                        e.get_mut().push(row_idx);
                    }
                }
            }

            groups
        } else {
            // For large data, use parallel processing + lock-free approach
            // 1. Create local group maps in parallel
            // 2. Merge them
            let chunk_size = (self.row_count / rayon::current_num_threads()).max(1000);

            // Step 1: Create local intermediate group maps in parallel
            let local_maps: Vec<HashMap<String, Vec<usize>>> = (0..self.row_count)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut local_groups = HashMap::new();

                    for &row_idx in chunk {
                        // Generate group key for this row
                        let mut key_parts = Vec::with_capacity(group_col_indices.len());

                        for &col_idx in &group_col_indices {
                            let column = &self.columns[col_idx];
                            let part = match column {
                                Column::Int64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                                Column::Float64(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                                Column::String(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                                Column::Boolean(col) => {
                                    if let Ok(Some(val)) = col.get(row_idx) {
                                        val.to_string()
                                    } else {
                                        "NA".to_string()
                                    }
                                }
                            };
                            key_parts.push(part);
                        }

                        let group_key = key_parts.join("_");

                        match local_groups.entry(group_key) {
                            Entry::Vacant(e) => {
                                e.insert(vec![row_idx]);
                            }
                            Entry::Occupied(mut e) => {
                                e.get_mut().push(row_idx);
                            }
                        }
                    }

                    local_groups
                })
                .collect();

            // Step 2: Merge intermediate maps
            let mut merged_groups = HashMap::new();
            for local_map in local_maps {
                for (key, indices) in local_map {
                    match merged_groups.entry(key) {
                        Entry::Vacant(e) => {
                            e.insert(indices);
                        }
                        Entry::Occupied(mut e) => {
                            e.get_mut().extend(indices);
                        }
                    }
                }
            }

            merged_groups
        };

        // Efficiently create DataFrames for each group
        let result = if groups.len() < 100 || self.row_count < PARALLEL_THRESHOLD {
            // Use serial processing for small data or when group count is small
            let mut result = HashMap::with_capacity(groups.len());
            for (key, indices) in groups {
                let group_df = self.filter_by_indices(&indices)?;
                result.insert(key, group_df);
            }
            result
        } else {
            // Parallelize group processing for large data
            // Process each group in parallel and safely aggregate results
            let result_mutex = Arc::new(Mutex::new(HashMap::with_capacity(groups.len())));

            // Adjust chunk size to minimize overhead
            let chunk_size = (groups.len() / rayon::current_num_threads()).max(10);

            // Create list of groups and split into chunks for parallel processing
            let group_items: Vec<(String, Vec<usize>)> = groups.into_iter().collect();

            group_items.par_chunks(chunk_size).for_each(|chunk| {
                // Temporarily store processing results for each chunk
                let mut local_results = HashMap::new();

                for (key, indices) in chunk {
                    if let Ok(group_df) = self.filter_by_indices(indices) {
                        local_results.insert(key.clone(), group_df);
                    }
                }

                // Merge results into the main HashMap
                if let Ok(mut result_map) = result_mutex.lock() {
                    for (key, df) in local_results {
                        result_map.insert(key, df);
                    }
                }
            });

            // Get final results
            match Arc::try_unwrap(result_mutex) {
                Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                Err(_) => HashMap::new(), // If failed to unwrap arc
            }
        };

        Ok(result)
    }
}
