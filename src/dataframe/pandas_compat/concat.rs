//! DataFrame concatenation operations
//!
//! Provides pandas-compatible concat functionality for combining DataFrames.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;

/// Axis for concatenation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcatAxis {
    /// Concatenate along rows (stack vertically, axis=0 in pandas)
    Rows,
    /// Concatenate along columns (stack horizontally, axis=1 in pandas)
    Columns,
}

/// Concatenate DataFrames along an axis
///
/// # Arguments
/// * `dfs` - Slice of DataFrames to concatenate
/// * `axis` - Axis along which to concatenate (Rows or Columns)
/// * `ignore_index` - If true, do not use index values along the concatenation axis
///
/// # Returns
/// Concatenated DataFrame
///
/// # Example
/// ```ignore
/// use pandrs::dataframe::pandas_compat::{concat, ConcatAxis};
///
/// let df1 = DataFrame::new(); // ... populate
/// let df2 = DataFrame::new(); // ... populate
/// let result = concat(&[&df1, &df2], ConcatAxis::Rows, true)?;
/// ```
pub fn concat(dfs: &[&DataFrame], axis: ConcatAxis, _ignore_index: bool) -> Result<DataFrame> {
    if dfs.is_empty() {
        return Ok(DataFrame::new());
    }

    if dfs.len() == 1 {
        return Ok(dfs[0].clone());
    }

    match axis {
        ConcatAxis::Rows => concat_rows(dfs),
        ConcatAxis::Columns => concat_columns(dfs),
    }
}

/// Concatenate DataFrames vertically (row-wise)
fn concat_rows(dfs: &[&DataFrame]) -> Result<DataFrame> {
    // Collect all unique column names in order
    let mut all_columns: Vec<String> = Vec::new();
    for df in dfs {
        for col in df.column_names() {
            if !all_columns.contains(&col) {
                all_columns.push(col);
            }
        }
    }

    if all_columns.is_empty() {
        return Ok(DataFrame::new());
    }

    // Determine column types from first DataFrame that has each column
    let mut column_types: std::collections::HashMap<String, ColumnType> =
        std::collections::HashMap::new();

    for col in &all_columns {
        for df in dfs {
            if df.contains_column(col) {
                if df.get_column_numeric_values(col).is_ok() {
                    column_types.insert(col.clone(), ColumnType::Numeric);
                } else if df.get_column_string_values(col).is_ok() {
                    column_types.insert(col.clone(), ColumnType::String);
                }
                break;
            }
        }
    }

    // Build result DataFrame
    let mut result = DataFrame::new();

    for col in &all_columns {
        let col_type = column_types
            .get(col)
            .cloned()
            .unwrap_or(ColumnType::Numeric);

        match col_type {
            ColumnType::Numeric => {
                let mut values: Vec<f64> = Vec::new();
                for df in dfs {
                    if let Ok(col_values) = df.get_column_numeric_values(col) {
                        values.extend(col_values);
                    } else {
                        // Column doesn't exist in this DataFrame, fill with NaN
                        for _ in 0..df.row_count() {
                            values.push(f64::NAN);
                        }
                    }
                }
                result.add_column(col.clone(), Series::new(values, Some(col.clone()))?)?;
            }
            ColumnType::String => {
                let mut values: Vec<String> = Vec::new();
                for df in dfs {
                    if let Ok(col_values) = df.get_column_string_values(col) {
                        values.extend(col_values);
                    } else {
                        // Column doesn't exist in this DataFrame, fill with empty string
                        for _ in 0..df.row_count() {
                            values.push(String::new());
                        }
                    }
                }
                result.add_column(col.clone(), Series::new(values, Some(col.clone()))?)?;
            }
        }
    }

    Ok(result)
}

/// Concatenate DataFrames horizontally (column-wise)
fn concat_columns(dfs: &[&DataFrame]) -> Result<DataFrame> {
    // Verify all DataFrames have the same number of rows
    let row_counts: Vec<usize> = dfs.iter().map(|df| df.row_count()).collect();
    if !row_counts.windows(2).all(|w| w[0] == w[1]) {
        return Err(Error::InvalidValue(
            "All DataFrames must have the same number of rows for column-wise concatenation"
                .to_string(),
        ));
    }

    let mut result = DataFrame::new();

    // Track column names to handle duplicates
    let mut seen_columns: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for df in dfs {
        for col_name in df.column_names() {
            let final_name = if seen_columns.contains_key(&col_name) {
                let count = seen_columns.get_mut(&col_name).unwrap();
                *count += 1;
                format!("{}_{}", col_name, count)
            } else {
                seen_columns.insert(col_name.clone(), 0);
                col_name.clone()
            };

            // Copy column data
            if let Ok(values) = df.get_column_numeric_values(&col_name) {
                result.add_column(
                    final_name.clone(),
                    Series::new(values, Some(final_name.clone()))?,
                )?;
            } else if let Ok(values) = df.get_column_string_values(&col_name) {
                result.add_column(
                    final_name.clone(),
                    Series::new(values, Some(final_name.clone()))?,
                )?;
            }
        }
    }

    Ok(result)
}

#[derive(Debug, Clone, Copy)]
enum ColumnType {
    Numeric,
    String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_rows_same_columns() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df1.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 4.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "b".to_string(),
            Series::new(vec![30.0, 40.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, true).unwrap();

        assert_eq!(result.row_count(), 4);
        let a_values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(a_values, vec![1.0, 2.0, 3.0, 4.0]);
        let b_values = result.get_column_numeric_values("b").unwrap();
        assert_eq!(b_values, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_concat_rows_different_columns() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "b".to_string(),
            Series::new(vec![30.0, 40.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, true).unwrap();

        assert_eq!(result.row_count(), 4);

        // Column 'a' should have values from df1, then NaN for df2
        let a_values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(a_values[0], 1.0);
        assert_eq!(a_values[1], 2.0);
        assert!(a_values[2].is_nan());
        assert!(a_values[3].is_nan());

        // Column 'b' should have NaN for df1, then values from df2
        let b_values = result.get_column_numeric_values("b").unwrap();
        assert!(b_values[0].is_nan());
        assert!(b_values[1].is_nan());
        assert_eq!(b_values[2], 30.0);
        assert_eq!(b_values[3], 40.0);
    }

    #[test]
    fn test_concat_rows_string_columns() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "name".to_string(),
            Series::new(
                vec!["Alice".to_string(), "Bob".to_string()],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "name".to_string(),
            Series::new(
                vec!["Charlie".to_string(), "David".to_string()],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, true).unwrap();

        assert_eq!(result.row_count(), 4);
        let names = result.get_column_string_values("name").unwrap();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie", "David"]);
    }

    #[test]
    fn test_concat_columns() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = concat(&[&df1, &df2], ConcatAxis::Columns, true).unwrap();

        assert_eq!(result.row_count(), 2);
        assert!(result.contains_column("a"));
        assert!(result.contains_column("b"));

        let a_values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(a_values, vec![1.0, 2.0]);
        let b_values = result.get_column_numeric_values("b").unwrap();
        assert_eq!(b_values, vec![10.0, 20.0]);
    }

    #[test]
    fn test_concat_columns_duplicate_names() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "value".to_string(),
            Series::new(vec![1.0, 2.0], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "value".to_string(),
            Series::new(vec![10.0, 20.0], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        let result = concat(&[&df1, &df2], ConcatAxis::Columns, true).unwrap();

        assert_eq!(result.row_count(), 2);
        // Should have renamed duplicate column
        assert!(result.contains_column("value"));
        assert!(result.contains_column("value_1"));

        let v1 = result.get_column_numeric_values("value").unwrap();
        assert_eq!(v1, vec![1.0, 2.0]);
        let v2 = result.get_column_numeric_values("value_1").unwrap();
        assert_eq!(v2, vec![10.0, 20.0]);
    }

    #[test]
    fn test_concat_columns_mismatched_rows() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("b".to_string())).unwrap(),
        )
        .unwrap();

        let result = concat(&[&df1, &df2], ConcatAxis::Columns, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_empty_input() {
        let result = concat(&[], ConcatAxis::Rows, true).unwrap();
        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn test_concat_single_df() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = concat(&[&df], ConcatAxis::Rows, true).unwrap();
        assert_eq!(result.row_count(), 2);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_concat_multiple_dfs() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![2.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let mut df3 = DataFrame::new();
        df3.add_column(
            "a".to_string(),
            Series::new(vec![3.0], Some("a".to_string())).unwrap(),
        )
        .unwrap();

        let result = concat(&[&df1, &df2, &df3], ConcatAxis::Rows, true).unwrap();
        assert_eq!(result.row_count(), 3);
        let values = result.get_column_numeric_values("a").unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
}
