//! Merge and join operations for DataFrames
//!
//! Provides pandas-compatible merge and join functionality.

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;
use std::collections::HashMap;

/// Join type for merge operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join - only matching rows from both DataFrames
    Inner,
    /// Left join - all rows from left, matching from right (with NaN for non-matches)
    Left,
    /// Right join - all rows from right, matching from left (with NaN for non-matches)
    Right,
    /// Outer join - all rows from both DataFrames (with NaN for non-matches)
    Outer,
}

/// Merge two DataFrames on a common column
///
/// # Arguments
/// * `left` - Left DataFrame
/// * `right` - Right DataFrame
/// * `on` - Column name to join on (must exist in both DataFrames)
/// * `how` - Join type (inner, left, right, outer)
/// * `suffixes` - Tuple of suffixes to add to overlapping column names (left_suffix, right_suffix)
///
/// # Returns
/// Merged DataFrame
pub fn merge(
    left: &DataFrame,
    right: &DataFrame,
    on: &str,
    how: JoinType,
    suffixes: (&str, &str),
) -> Result<DataFrame> {
    // Validate that join column exists in both DataFrames
    if !left.contains_column(on) {
        return Err(Error::InvalidValue(format!(
            "Join column '{}' not found in left DataFrame",
            on
        )));
    }
    if !right.contains_column(on) {
        return Err(Error::InvalidValue(format!(
            "Join column '{}' not found in right DataFrame",
            on
        )));
    }

    // Get join column values (try numeric first, then string)
    let left_join_values = if let Ok(vals) = left.get_column_numeric_values(on) {
        vals.into_iter()
            .map(|v| v.to_bits().to_string())
            .collect::<Vec<_>>()
    } else if let Ok(vals) = left.get_column_string_values(on) {
        vals
    } else {
        return Err(Error::InvalidValue(format!(
            "Cannot read join column '{}' from left DataFrame",
            on
        )));
    };

    let right_join_values = if let Ok(vals) = right.get_column_numeric_values(on) {
        vals.into_iter()
            .map(|v| v.to_bits().to_string())
            .collect::<Vec<_>>()
    } else if let Ok(vals) = right.get_column_string_values(on) {
        vals
    } else {
        return Err(Error::InvalidValue(format!(
            "Cannot read join column '{}' from right DataFrame",
            on
        )));
    };

    // Build index maps for right DataFrame
    let mut right_index: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, val) in right_join_values.iter().enumerate() {
        right_index
            .entry(val.clone())
            .or_insert_with(Vec::new)
            .push(i);
    }

    // Collect matching row pairs based on join type
    let mut matched_pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();
    let mut left_matched = vec![false; left_join_values.len()];
    let mut right_matched = vec![false; right_join_values.len()];

    // Process left DataFrame rows
    for (left_idx, left_val) in left_join_values.iter().enumerate() {
        if let Some(right_indices) = right_index.get(left_val) {
            // Matching rows found
            for &right_idx in right_indices {
                matched_pairs.push((Some(left_idx), Some(right_idx)));
                left_matched[left_idx] = true;
                right_matched[right_idx] = true;
            }
        } else if matches!(how, JoinType::Left | JoinType::Outer) {
            // No match, but include for left/outer join
            matched_pairs.push((Some(left_idx), None));
            left_matched[left_idx] = true;
        }
    }

    // Add unmatched right rows for right/outer joins
    if matches!(how, JoinType::Right | JoinType::Outer) {
        for (right_idx, matched) in right_matched.iter().enumerate() {
            if !matched {
                matched_pairs.push((None, Some(right_idx)));
            }
        }
    }

    // Build result DataFrame
    let mut result = DataFrame::new();

    // Get column names
    let left_cols = left.column_names();
    let right_cols = right.column_names();

    // Identify overlapping columns (excluding join column)
    let mut overlapping: Vec<String> = Vec::new();
    for col in &right_cols {
        if col != on && left_cols.contains(col) {
            overlapping.push(col.clone());
        }
    }

    // Add columns from left DataFrame
    for col_name in &left_cols {
        // Try numeric first
        if let Ok(values) = left.get_column_numeric_values(col_name) {
            let merged: Vec<f64> = if col_name == on {
                // For join key, use right value when left is None
                let right_values = right.get_column_numeric_values(on).unwrap();
                matched_pairs
                    .iter()
                    .map(|(left_idx, right_idx)| {
                        left_idx
                            .map(|i| values.get(i).copied().unwrap_or(f64::NAN))
                            .or_else(|| {
                                right_idx.map(|i| right_values.get(i).copied().unwrap_or(f64::NAN))
                            })
                            .unwrap_or(f64::NAN)
                    })
                    .collect()
            } else {
                matched_pairs
                    .iter()
                    .map(|(left_idx, _)| {
                        left_idx
                            .map(|i| values.get(i).copied().unwrap_or(f64::NAN))
                            .unwrap_or(f64::NAN)
                    })
                    .collect()
            };
            result.add_column(
                col_name.clone(),
                Series::new(merged, Some(col_name.clone()))?,
            )?;
        } else if let Ok(values) = left.get_column_string_values(col_name) {
            let merged: Vec<String> = if col_name == on {
                // For join key, use right value when left is None
                let right_values = right.get_column_string_values(on).unwrap();
                matched_pairs
                    .iter()
                    .map(|(left_idx, right_idx)| {
                        left_idx
                            .and_then(|i| values.get(i).cloned())
                            .or_else(|| right_idx.and_then(|i| right_values.get(i).cloned()))
                            .unwrap_or_else(|| "".to_string())
                    })
                    .collect()
            } else {
                matched_pairs
                    .iter()
                    .map(|(left_idx, _)| {
                        left_idx
                            .and_then(|i| values.get(i).cloned())
                            .unwrap_or_else(|| "".to_string())
                    })
                    .collect()
            };
            result.add_column(
                col_name.clone(),
                Series::new(merged, Some(col_name.clone()))?,
            )?;
        }
    }

    // Add columns from right DataFrame (with suffix handling)
    for col_name in &right_cols {
        if col_name == on {
            // Skip join column (already included from left)
            continue;
        }

        let final_name = if overlapping.contains(col_name) {
            format!("{}{}", col_name, suffixes.1)
        } else {
            col_name.clone()
        };

        // Try numeric first
        if let Ok(values) = right.get_column_numeric_values(col_name) {
            let merged: Vec<f64> = matched_pairs
                .iter()
                .map(|(_, right_idx)| {
                    right_idx
                        .map(|i| values.get(i).copied().unwrap_or(f64::NAN))
                        .unwrap_or(f64::NAN)
                })
                .collect();
            result.add_column(
                final_name.clone(),
                Series::new(merged, Some(final_name.clone()))?,
            )?;
        } else if let Ok(values) = right.get_column_string_values(col_name) {
            let merged: Vec<String> = matched_pairs
                .iter()
                .map(|(_, right_idx)| {
                    right_idx
                        .and_then(|i| values.get(i).cloned())
                        .unwrap_or_else(|| "".to_string())
                })
                .collect();
            result.add_column(
                final_name.clone(),
                Series::new(merged, Some(final_name.clone()))?,
            )?;
        }
    }

    // Rename overlapping columns from left DataFrame with suffix
    if !overlapping.is_empty() {
        let mut rename_map = HashMap::new();
        for col in &overlapping {
            rename_map.insert(col.clone(), format!("{}{}", col, suffixes.0));
        }

        // Rebuild DataFrame with renamed columns
        let mut renamed_df = DataFrame::new();
        for col_name in result.column_names() {
            let new_name = rename_map.get(&col_name).unwrap_or(&col_name).clone();

            if let Ok(vals) = result.get_column_numeric_values(&col_name) {
                renamed_df
                    .add_column(new_name.clone(), Series::new(vals, Some(new_name.clone()))?)?;
            } else if let Ok(vals) = result.get_column_string_values(&col_name) {
                renamed_df
                    .add_column(new_name.clone(), Series::new(vals, Some(new_name.clone()))?)?;
            }
        }
        result = renamed_df;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_left_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "key".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "C".to_string(),
                    "D".to_string(),
                ],
                Some("key".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("value1".to_string())).unwrap(),
        )
        .unwrap();
        df
    }

    fn create_right_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "key".to_string(),
            Series::new(
                vec![
                    "B".to_string(),
                    "C".to_string(),
                    "D".to_string(),
                    "E".to_string(),
                ],
                Some("key".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value2".to_string(),
            Series::new(vec![20.0, 30.0, 40.0, 50.0], Some("value2".to_string())).unwrap(),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_merge_inner() {
        let left = create_left_df();
        let right = create_right_df();

        let result = merge(&left, &right, "key", JoinType::Inner, ("_x", "_y")).unwrap();

        // Inner join should have 3 rows (B, C, D)
        assert_eq!(result.row_count(), 3);

        let keys = result.get_column_string_values("key").unwrap();
        assert_eq!(keys, vec!["B", "C", "D"]);

        let val1 = result.get_column_numeric_values("value1").unwrap();
        assert_eq!(val1, vec![2.0, 3.0, 4.0]);

        let val2 = result.get_column_numeric_values("value2").unwrap();
        assert_eq!(val2, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_merge_left() {
        let left = create_left_df();
        let right = create_right_df();

        let result = merge(&left, &right, "key", JoinType::Left, ("_x", "_y")).unwrap();

        // Left join should have 4 rows (all from left: A, B, C, D)
        assert_eq!(result.row_count(), 4);

        let keys = result.get_column_string_values("key").unwrap();
        assert_eq!(keys, vec!["A", "B", "C", "D"]);

        let val1 = result.get_column_numeric_values("value1").unwrap();
        assert_eq!(val1, vec![1.0, 2.0, 3.0, 4.0]);

        let val2 = result.get_column_numeric_values("value2").unwrap();
        assert!(val2[0].is_nan()); // A has no match
        assert_eq!(val2[1], 20.0);
        assert_eq!(val2[2], 30.0);
        assert_eq!(val2[3], 40.0);
    }

    #[test]
    fn test_merge_right() {
        let left = create_left_df();
        let right = create_right_df();

        let result = merge(&left, &right, "key", JoinType::Right, ("_x", "_y")).unwrap();

        // Right join should have 4 rows (all from right: B, C, D, E)
        assert_eq!(result.row_count(), 4);

        let keys = result.get_column_string_values("key").unwrap();
        assert_eq!(keys, vec!["B", "C", "D", "E"]);

        let val1 = result.get_column_numeric_values("value1").unwrap();
        assert_eq!(val1[0], 2.0);
        assert_eq!(val1[1], 3.0);
        assert_eq!(val1[2], 4.0);
        assert!(val1[3].is_nan()); // E has no match

        let val2 = result.get_column_numeric_values("value2").unwrap();
        assert_eq!(val2, vec![20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_merge_outer() {
        let left = create_left_df();
        let right = create_right_df();

        let result = merge(&left, &right, "key", JoinType::Outer, ("_x", "_y")).unwrap();

        // Outer join should have 5 rows (A, B, C, D, E)
        assert_eq!(result.row_count(), 5);

        let keys = result.get_column_string_values("key").unwrap();
        assert_eq!(keys, vec!["A", "B", "C", "D", "E"]);

        let val1 = result.get_column_numeric_values("value1").unwrap();
        assert_eq!(val1[0], 1.0); // A
        assert_eq!(val1[1], 2.0); // B
        assert_eq!(val1[2], 3.0); // C
        assert_eq!(val1[3], 4.0); // D
        assert!(val1[4].is_nan()); // E (no match in left)

        let val2 = result.get_column_numeric_values("value2").unwrap();
        assert!(val2[0].is_nan()); // A (no match in right)
        assert_eq!(val2[1], 20.0); // B
        assert_eq!(val2[2], 30.0); // C
        assert_eq!(val2[3], 40.0); // D
        assert_eq!(val2[4], 50.0); // E
    }

    #[test]
    fn test_merge_with_overlapping_columns() {
        let mut left = DataFrame::new();
        left.add_column(
            "key".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string()],
                Some("key".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        left.add_column(
            "value".to_string(),
            Series::new(vec![1.0, 2.0], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        let mut right = DataFrame::new();
        right
            .add_column(
                "key".to_string(),
                Series::new(
                    vec!["A".to_string(), "B".to_string()],
                    Some("key".to_string()),
                )
                .unwrap(),
            )
            .unwrap();
        right
            .add_column(
                "value".to_string(),
                Series::new(vec![10.0, 20.0], Some("value".to_string())).unwrap(),
            )
            .unwrap();

        let result = merge(&left, &right, "key", JoinType::Inner, ("_left", "_right")).unwrap();

        // Should have renamed overlapping 'value' column
        assert!(result.contains_column("value_left"));
        assert!(result.contains_column("value_right"));

        let val_left = result.get_column_numeric_values("value_left").unwrap();
        assert_eq!(val_left, vec![1.0, 2.0]);

        let val_right = result.get_column_numeric_values("value_right").unwrap();
        assert_eq!(val_right, vec![10.0, 20.0]);
    }

    #[test]
    fn test_merge_numeric_key() {
        let mut left = DataFrame::new();
        left.add_column(
            "id".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        left.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                ],
                Some("name".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let mut right = DataFrame::new();
        right
            .add_column(
                "id".to_string(),
                Series::new(vec![2.0, 3.0, 4.0], Some("id".to_string())).unwrap(),
            )
            .unwrap();
        right
            .add_column(
                "score".to_string(),
                Series::new(vec![85.0, 90.0, 95.0], Some("score".to_string())).unwrap(),
            )
            .unwrap();

        let result = merge(&left, &right, "id", JoinType::Inner, ("_x", "_y")).unwrap();

        assert_eq!(result.row_count(), 2); // Only 2.0 and 3.0 match

        let names = result.get_column_string_values("name").unwrap();
        assert_eq!(names, vec!["Bob", "Charlie"]);

        let scores = result.get_column_numeric_values("score").unwrap();
        assert_eq!(scores, vec![85.0, 90.0]);
    }
}
