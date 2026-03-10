//! Helper functions for functions2 module

use crate::core::error::Result;
use crate::dataframe::base::DataFrame;
use crate::series::Series;

/// Helper function for selecting rows by indices
pub(super) fn select_rows_by_indices(df: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
    let mut new_df = DataFrame::new();
    for col_name in df.column_names() {
        if let Ok(values) = df.get_column_numeric_values(&col_name) {
            let selected: Vec<f64> = indices
                .iter()
                .filter_map(|&i| values.get(i).copied())
                .collect();
            new_df.add_column(
                col_name.clone(),
                Series::new(selected, Some(col_name.clone()))?,
            )?;
        } else if let Ok(values) = df.get_column_string_values(&col_name) {
            let selected: Vec<String> = indices
                .iter()
                .filter_map(|&i| values.get(i).cloned())
                .collect();
            new_df.add_column(
                col_name.clone(),
                Series::new(selected, Some(col_name.clone()))?,
            )?;
        }
    }
    Ok(new_df)
}
