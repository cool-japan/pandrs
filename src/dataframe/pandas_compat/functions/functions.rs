//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::core::error::Result;
use crate::dataframe::base::DataFrame;
#[cfg(test)]
use crate::series::Series;

// Helper function for selecting rows by indices.
//
// Delegates to `DataFrame::sample`, which downcasts each column's boxed
// storage to its concrete element type (f64/i64/i32/f32/String/bool) before
// selecting rows. This preserves the column's real dtype instead of
// re-inferring it from stringified values -- the previous implementation
// routed every column through `get_column_numeric_values` first, which
// happily *parses* a `Series<String>` column of e.g. `"007"` into `7.0`,
// silently turning text data into floats (and losing leading zeros / any
// non-numeric strings entirely, since those rows were dropped by the
// `filter_map`). `sample` tries the `String` downcast before ever attempting
// numeric parsing, so text columns stay text.
//
// The row index (when the source frame carries an explicit one) is
// preserved too: `sample` only rebuilds the columns, so we additionally
// slice the index labels the same way and re-attach them. If the same
// source row is selected more than once (e.g. `sample(n, replace=true)`),
// the sliced labels would contain duplicates, which `Index` rejects -- in
// that case we leave the result on the default positional index rather
// than fail the whole selection, matching the previous (index-dropping)
// behavior for that one case.
pub(super) fn select_rows_by_indices(df: &DataFrame, indices: &[usize]) -> Result<DataFrame> {
    let mut new_df = df.sample(indices)?;

    if let Some(labels) = df.get_index().string_values() {
        if labels.len() == df.row_count() && !labels.is_empty() {
            let selected: Vec<String> = indices
                .iter()
                .filter_map(|&i| labels.get(i).cloned())
                .collect();
            if selected.len() == indices.len() {
                if let Ok(new_index) = crate::index::Index::<String>::new(selected) {
                    new_df.set_index(new_index)?;
                }
            }
        }
    }

    Ok(new_df)
}

#[cfg(test)]
mod tests {
    use super::super::super::trait_def::PandasCompatExt;
    use super::super::super::types::{Axis, RankMethod};
    use super::*;
    use std::collections::HashMap;
    fn create_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![5.0, 4.0, 3.0, 2.0, 1.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                    "David".to_string(),
                    "Eve".to_string(),
                ],
                Some("name".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df
    }
    #[test]
    fn test_pipe() {
        let df = create_test_df();
        let result = df.pipe(|d| d.row_count());
        assert_eq!(result, 5);
    }
    #[test]
    fn test_isin() {
        let df = create_test_df();
        let mask = df
            .isin("name", &["Alice", "Bob", "Unknown"])
            .expect("test should succeed");
        assert_eq!(mask, vec![true, true, false, false, false]);
    }
    #[test]
    fn test_nlargest() {
        let df = create_test_df();
        let result = df.nlargest(3, "a").expect("test should succeed");
        assert_eq!(result.row_count(), 3);
    }
    #[test]
    fn test_nsmallest() {
        let df = create_test_df();
        let result = df.nsmallest(2, "a").expect("test should succeed");
        assert_eq!(result.row_count(), 2);
    }
    #[test]
    fn test_idxmax() {
        let df = create_test_df();
        let idx = df.idxmax("a").expect("test should succeed");
        assert_eq!(idx, Some(4));
    }
    #[test]
    fn test_idxmin() {
        let df = create_test_df();
        let idx = df.idxmin("a").expect("test should succeed");
        assert_eq!(idx, Some(0));
    }
    #[test]
    fn test_rank_average() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("x".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let ranks = df
            .rank("x", RankMethod::Average)
            .expect("test should succeed");
        assert_eq!(ranks[1], 1.5);
        assert_eq!(ranks[3], 1.5);
        assert_eq!(ranks[0], 3.0);
    }
    #[test]
    fn test_between() {
        let df = create_test_df();
        let mask = df.between("a", 2.0, 4.0).expect("test should succeed");
        assert_eq!(mask, vec![false, true, true, true, false]);
    }
    #[test]
    fn test_cumsum() {
        let df = create_test_df();
        let result = df.cumsum("a").expect("test should succeed");
        assert_eq!(result, vec![1.0, 3.0, 6.0, 10.0, 15.0]);
    }
    #[test]
    fn test_cumprod() {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("x".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.cumprod("x").expect("test should succeed");
        assert_eq!(result, vec![1.0, 2.0, 6.0, 24.0]);
    }
    #[test]
    fn test_cummax() {
        let df = create_test_df();
        let result = df.cummax("b").expect("test should succeed");
        assert_eq!(result, vec![5.0, 5.0, 5.0, 5.0, 5.0]);
    }
    #[test]
    fn test_cummin() {
        let df = create_test_df();
        let result = df.cummin("b").expect("test should succeed");
        assert_eq!(result, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }
    #[test]
    fn test_shift() {
        let df = create_test_df();
        let result = df.shift("a", 1).expect("test should succeed");
        assert_eq!(result[0], None);
        assert_eq!(result[1], Some(1.0));
        assert_eq!(result[2], Some(2.0));
    }
    #[test]
    fn test_nunique() {
        let df = create_test_df();
        let result = df.nunique().expect("test should succeed");
        for (_, count) in &result {
            assert_eq!(*count, 5);
        }
    }
    #[test]
    fn test_memory_usage() {
        let df = create_test_df();
        let mem = df.memory_usage();
        assert!(mem > 0);
    }
    #[test]
    fn test_assign_many() {
        let df = create_test_df();
        let result = df
            .assign_many(vec![("c", vec![10.0, 20.0, 30.0, 40.0, 50.0])])
            .expect("test should succeed");
        assert!(result.contains_column("c"));
    }
    #[test]
    fn test_value_counts() {
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "C".to_string(),
                    "A".to_string(),
                ],
                Some("category".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let counts = PandasCompatExt::value_counts(&df, "category").expect("test should succeed");
        assert_eq!(&counts[0].0, "A");
        assert_eq!(counts[0].1, 3);
        assert_eq!(counts[1].1, 1);
        assert_eq!(counts[2].1, 1);
    }
    #[test]
    fn test_value_counts_numeric() {
        let df = create_test_df();
        let result = df.value_counts_numeric("a").expect("test should succeed");
        assert_eq!(result.len(), 5);
        for (_, count) in &result {
            assert_eq!(*count, 1);
        }
    }
    #[test]
    fn test_describe() {
        let df = create_test_df();
        let stats = df.describe("a").expect("test should succeed");
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 0.0001);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.q50, 3.0);
    }
    #[test]
    fn test_apply_rows() {
        let df = create_test_df();
        let result: Vec<f64> = df
            .apply(|row| row.iter().sum::<f64>(), Axis::Rows)
            .expect("test should succeed");
        assert!((result[0] - 6.0).abs() < 0.0001);
        assert!((result[4] - 6.0).abs() < 0.0001);
    }
    #[test]
    fn test_apply_columns() {
        let df = create_test_df();
        let result: Vec<f64> = df
            .apply(|col| col.iter().sum::<f64>(), Axis::Columns)
            .expect("test should succeed");
        assert!((result[0] - 15.0).abs() < 0.0001);
        assert!((result[1] - 15.0).abs() < 0.0001);
    }
    #[test]
    fn test_corr() {
        let df = create_test_df();
        let corr_matrix = df.corr().expect("test should succeed");
        assert!((corr_matrix.values[0][0] - 1.0).abs() < 0.0001);
        assert!((corr_matrix.values[1][1] - 1.0).abs() < 0.0001);
        assert!(corr_matrix.values[0][1] < -0.99);
        assert!(corr_matrix.values[1][0] < -0.99);
    }
    #[test]
    fn test_cov() {
        let df = create_test_df();
        let cov_matrix = df.cov().expect("test should succeed");
        assert!((cov_matrix.values[0][1] - cov_matrix.values[1][0]).abs() < 0.0001);
        let (rows, cols) = cov_matrix.shape();
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);
    }
    #[test]
    fn test_pct_change() {
        let df = create_test_df();
        let result = df.pct_change("a", 1).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!((result[1] - 1.0).abs() < 0.0001);
        assert!((result[2] - 0.5).abs() < 0.0001);
    }
    #[test]
    fn test_diff() {
        let df = create_test_df();
        let result = df.diff("a", 1).expect("test should succeed");
        assert!(result[0].is_nan());
        for i in 1..result.len() {
            assert!((result[i] - 1.0).abs() < 0.0001);
        }
    }
    #[test]
    fn test_diff_periods() {
        let df = create_test_df();
        let result = df.diff("a", 2).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        for i in 2..result.len() {
            assert!((result[i] - 2.0).abs() < 0.0001);
        }
    }
    #[test]
    fn test_replace() {
        let mut df = DataFrame::new();
        df.add_column(
            "status".to_string(),
            Series::new(
                vec!["ok".to_string(), "fail".to_string(), "ok".to_string()],
                Some("status".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .replace("status", &["ok", "fail"], &["success", "error"])
            .expect("test should succeed");
        let values = result
            .get_column_string_values("status")
            .expect("test should succeed");
        assert_eq!(values[0], "success");
        assert_eq!(values[1], "error");
        assert_eq!(values[2], "success");
    }
    #[test]
    fn test_replace_numeric() {
        let df = create_test_df();
        let result = df
            .replace_numeric("a", &[1.0, 2.0], &[10.0, 20.0])
            .expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 10.0);
        assert_eq!(values[1], 20.0);
        assert_eq!(values[2], 3.0);
    }
    #[test]
    fn test_correlation_matrix_get() {
        let df = create_test_df();
        let corr = df.corr().expect("test should succeed");
        let val = corr.get("a", "b").expect("test should succeed");
        assert!(val < -0.99);
        let self_corr = corr.get("a", "a").expect("test should succeed");
        assert!((self_corr - 1.0).abs() < 0.0001);
    }
    #[test]
    fn test_sample() {
        let df = create_test_df();
        let result = PandasCompatExt::sample(&df, 3, false).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let result2 = PandasCompatExt::sample(&df, 10, true).expect("test should succeed");
        assert_eq!(result2.row_count(), 10);
    }
    #[test]
    fn test_sample_too_many() {
        let df = create_test_df();
        let result = PandasCompatExt::sample(&df, 10, false);
        assert!(result.is_err());
    }
    #[test]
    fn test_drop_columns() {
        let df = create_test_df();
        let result = df.drop_columns(&["a"]).expect("test should succeed");
        assert!(!result.contains_column("a"));
        assert!(result.contains_column("b"));
        assert!(result.contains_column("name"));
    }
    #[test]
    fn test_rename_columns() {
        let df = create_test_df();
        let mut mapper = HashMap::new();
        mapper.insert("a".to_string(), "alpha".to_string());
        mapper.insert("b".to_string(), "beta".to_string());
        let result = df.rename_columns(&mapper).expect("test should succeed");
        assert!(result.contains_column("alpha"));
        assert!(result.contains_column("beta"));
        assert!(!result.contains_column("a"));
        assert!(!result.contains_column("b"));
    }
    #[test]
    fn test_abs() {
        let mut df = DataFrame::new();
        df.add_column(
            "values".to_string(),
            Series::new(vec![-1.0, -2.0, 3.0, -4.0, 5.0], Some("values".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.abs("values").expect("test should succeed");
        let values = result
            .get_column_numeric_values("values")
            .expect("test should succeed");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 5.0);
    }
    #[test]
    fn test_round() {
        let mut df = DataFrame::new();
        df.add_column(
            "values".to_string(),
            Series::new(vec![1.123, 2.567, 3.999], Some("values".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.round("values", 2).expect("test should succeed");
        let values = result
            .get_column_numeric_values("values")
            .expect("test should succeed");
        assert!((values[0] - 1.12).abs() < 0.001);
        assert!((values[1] - 2.57).abs() < 0.001);
        assert!((values[2] - 4.00).abs() < 0.001);
    }
    #[test]
    fn test_quantile() {
        let df = create_test_df();
        let median = df.quantile("a", 0.5).expect("test should succeed");
        assert_eq!(median, 3.0);
        let q25 = df.quantile("a", 0.25).expect("test should succeed");
        assert_eq!(q25, 2.0);
        let q75 = df.quantile("a", 0.75).expect("test should succeed");
        assert_eq!(q75, 4.0);
    }
    #[test]
    fn test_quantile_invalid() {
        let df = create_test_df();
        assert!(df.quantile("a", 1.5).is_err());
        assert!(df.quantile("a", -0.5).is_err());
    }
    #[test]
    fn test_head() {
        let df = create_test_df();
        let result = PandasCompatExt::head(&df, 3).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_head_more_than_available() {
        let df = create_test_df();
        let result = PandasCompatExt::head(&df, 100).expect("test should succeed");
        assert_eq!(result.row_count(), 5);
    }
    #[test]
    fn test_tail() {
        let df = create_test_df();
        let result = PandasCompatExt::tail(&df, 3).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_unique() {
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "C".to_string(),
                    "A".to_string(),
                ],
                Some("category".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.unique("category").expect("test should succeed");
        assert_eq!(result.len(), 3);
        assert!(result.contains(&"A".to_string()));
        assert!(result.contains(&"B".to_string()));
        assert!(result.contains(&"C".to_string()));
    }
    #[test]
    fn test_unique_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "nums".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0, 1.0], Some("nums".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.unique_numeric("nums").expect("test should succeed");
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_fillna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna("a", 0.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    }
    #[test]
    fn test_fillna_with_negative() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna("a", -999.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, -999.0, 3.0]);
    }
    #[test]
    fn test_fillna_ffill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna_method("a", "ffill").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!(values[0] == 1.0);
        assert_eq!(values[1], 1.0);
        assert_eq!(values[2], 1.0);
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 4.0);
        assert_eq!(values[5], 6.0);
    }
    #[test]
    fn test_fillna_bfill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna_method("a", "bfill").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 4.0);
        assert_eq!(values[2], 4.0);
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 6.0);
        assert_eq!(values[5], 6.0);
    }
    #[test]
    fn test_fillna_ffill_leading_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna_method("a", "ffill").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 3.0);
        assert_eq!(values[4], 5.0);
    }
    #[test]
    fn test_fillna_bfill_trailing_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, f64::NAN],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna_method("a", "bfill").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 3.0);
        assert_eq!(values[2], 3.0);
        assert!(values[3].is_nan());
        assert!(values[4].is_nan());
    }
    #[test]
    fn test_fillna_invalid_method() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna_method("a", "invalid");
        assert!(result.is_err());
    }
    #[test]
    fn test_interpolate_linear() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN, 6.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.interpolate("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 5.0);
        assert_eq!(values[5], 6.0);
    }
    #[test]
    fn test_interpolate_single_gap() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, f64::NAN, 8.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.interpolate("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 2.0);
        assert_eq!(values[1], 5.0);
        assert_eq!(values[2], 8.0);
    }
    #[test]
    fn test_interpolate_leading_trailing_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, 2.0, f64::NAN, 4.0, f64::NAN],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.interpolate("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!(values[0].is_nan());
        assert_eq!(values[1], 2.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 4.0);
        assert!(values[4].is_nan());
    }
    #[test]
    fn test_interpolate_no_gaps() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.interpolate("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_dropna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0, 40.0, 50.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.dropna("a").expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values_a = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        let values_b = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(values_a, vec![1.0, 3.0, 5.0]);
        assert_eq!(values_b, vec![10.0, 30.0, 50.0]);
    }
    #[test]
    fn test_isna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.isna("a").expect("test should succeed");
        assert_eq!(result, vec![false, true, false, true, false]);
    }
    #[test]
    fn test_sum_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.sum_all().expect("test should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 6.0));
        assert_eq!(result[1], ("b".to_string(), 60.0));
    }
    #[test]
    fn test_sum_all_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.sum_all().expect("test should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("a".to_string(), 4.0));
    }
    #[test]
    fn test_mean_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.mean_all().expect("test should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 2.0));
        assert_eq!(result[1], ("b".to_string(), 20.0));
    }
    #[test]
    fn test_std_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.std_all().expect("test should succeed");
        assert_eq!(result.len(), 1);
        assert!((result[0].1 - 1.5811).abs() < 0.001);
    }
    #[test]
    fn test_var_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.var_all().expect("test should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("a".to_string(), 2.5));
    }
    #[test]
    fn test_min_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![5.0, 2.0, 8.0, 1.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 5.0, 15.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.min_all().expect("test should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 1.0));
        assert_eq!(result[1], ("b".to_string(), 5.0));
    }
    #[test]
    fn test_max_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![5.0, 2.0, 8.0, 1.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 5.0, 15.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.max_all().expect("test should succeed");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], ("a".to_string(), 8.0));
        assert_eq!(result[1], ("b".to_string(), 20.0));
    }
    #[test]
    fn test_sort_values_ascending() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 2.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![30.0, 10.0, 40.0, 20.0, 50.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.sort_values("a", true).expect("test should succeed");
        let values_a = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        let values_b = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(values_a, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(values_b, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }
    #[test]
    fn test_sort_values_descending() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 2.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.sort_values("a", false).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }
    #[test]
    fn test_sort_by_columns_single() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .sort_by_columns(&["a"], &[true])
            .expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_sort_by_columns_multiple() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![20.0, 10.0, 10.0, 20.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .sort_by_columns(&["a", "b"], &[true, true])
            .expect("test should succeed");
        let values_a = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        let values_b = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(values_a, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(values_b, vec![10.0, 20.0, 10.0, 20.0]);
    }
    #[test]
    fn test_sort_by_columns_mixed_order() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![20.0, 10.0, 10.0, 20.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .sort_by_columns(&["a", "b"], &[true, false])
            .expect("test should succeed");
        let values_a = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        let values_b = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(values_a, vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(values_b, vec![20.0, 10.0, 20.0, 10.0]);
    }
    #[test]
    fn test_sort_by_columns_error_mismatch() {
        let df = create_test_df();
        let result = df.sort_by_columns(&["a", "b"], &[true]);
        assert!(result.is_err());
    }
    #[test]
    fn test_where_cond() {
        let df = create_test_df();
        let condition = vec![true, false, true, false, true];
        let result = df
            .where_cond("a", &condition, -1.0)
            .expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, -1.0, 3.0, -1.0, 5.0]);
    }
    #[test]
    fn test_where_cond_all_true() {
        let df = create_test_df();
        let condition = vec![true, true, true, true, true];
        let result = df
            .where_cond("a", &condition, 0.0)
            .expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_where_cond_all_false() {
        let df = create_test_df();
        let condition = vec![false, false, false, false, false];
        let result = df
            .where_cond("a", &condition, 0.0)
            .expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    #[test]
    fn test_where_cond_length_mismatch() {
        let df = create_test_df();
        let condition = vec![true, false];
        let result = df.where_cond("a", &condition, -1.0);
        assert!(result.is_err());
    }
    #[test]
    fn test_mask() {
        let df = create_test_df();
        let condition = vec![true, false, true, false, true];
        let result = df.mask("a", &condition, -1.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![-1.0, 2.0, -1.0, 4.0, -1.0]);
    }
    #[test]
    fn test_mask_all_true() {
        let df = create_test_df();
        let condition = vec![true, true, true, true, true];
        let result = df.mask("a", &condition, 0.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    #[test]
    fn test_mask_all_false() {
        let df = create_test_df();
        let condition = vec![false, false, false, false, false];
        let result = df.mask("a", &condition, 0.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_mask_length_mismatch() {
        let df = create_test_df();
        let condition = vec![true, false];
        let result = df.mask("a", &condition, -1.0);
        assert!(result.is_err());
    }
    #[test]
    fn test_drop_duplicates_keep_first() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 10.0, 30.0, 20.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .drop_duplicates(None, "first")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values_a = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!(values_a.contains(&1.0));
        assert!(values_a.contains(&2.0));
        assert!(values_a.contains(&3.0));
    }
    #[test]
    fn test_drop_duplicates_keep_last() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .drop_duplicates(Some(&["a"]), "last")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 3);
    }
    #[test]
    fn test_drop_duplicates_keep_none() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 4.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .drop_duplicates(Some(&["a"]), "none")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!(values.contains(&2.0));
        assert!(values.contains(&3.0));
        assert!(values.contains(&4.0));
        assert!(!values.contains(&1.0));
    }
    #[test]
    fn test_drop_duplicates_subset() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 1.0, 2.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 10.0, 20.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .drop_duplicates(Some(&["a"]), "first")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 2);
    }
    #[test]
    fn test_drop_duplicates_invalid_keep() {
        let df = create_test_df();
        let result = df.drop_duplicates(None, "invalid");
        assert!(result.is_err());
    }
    #[test]
    fn test_drop_duplicates_no_duplicates() {
        let df = create_test_df();
        let result = df
            .drop_duplicates(None, "first")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 5);
    }
    #[test]
    fn test_select_dtypes_numeric() {
        let df = create_test_df();
        let result = df.select_dtypes(&["numeric"]).expect("test should succeed");
        assert!(result.contains_column("a"));
        assert!(result.contains_column("b"));
        assert!(!result.contains_column("name"));
    }
    #[test]
    fn test_select_dtypes_string() {
        let df = create_test_df();
        let result = df.select_dtypes(&["string"]).expect("test should succeed");
        assert!(!result.contains_column("a"));
        assert!(!result.contains_column("b"));
        assert!(result.contains_column("name"));
    }
    #[test]
    fn test_select_dtypes_both() {
        let df = create_test_df();
        let result = df
            .select_dtypes(&["numeric", "string"])
            .expect("test should succeed");
        assert!(result.contains_column("a"));
        assert!(result.contains_column("b"));
        assert!(result.contains_column("name"));
    }
    #[test]
    fn test_select_dtypes_aliases() {
        let df = create_test_df();
        let result = df.select_dtypes(&["number"]).expect("test should succeed");
        assert!(result.contains_column("a"));
        assert!(!result.contains_column("name"));
        let result2 = df.select_dtypes(&["object"]).expect("test should succeed");
        assert!(!result2.contains_column("a"));
        assert!(result2.contains_column("name"));
    }
    #[test]
    fn test_any_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "all_zero".to_string(),
            Series::new(vec![0.0, 0.0, 0.0], Some("all_zero".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "has_nonzero".to_string(),
            Series::new(vec![0.0, 1.0, 0.0], Some("has_nonzero".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "has_nan".to_string(),
            Series::new(vec![f64::NAN, f64::NAN, 1.0], Some("has_nan".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.any_numeric().expect("test should succeed");
        assert_eq!(result.len(), 3);
        for (name, has_any) in &result {
            match name.as_str() {
                "all_zero" => assert!(!has_any),
                "has_nonzero" => assert!(*has_any),
                "has_nan" => assert!(*has_any),
                _ => panic!("Unexpected column"),
            }
        }
    }
    #[test]
    fn test_all_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "all_nonzero".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("all_nonzero".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "has_zero".to_string(),
            Series::new(vec![1.0, 0.0, 3.0], Some("has_zero".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "has_nan".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("has_nan".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.all_numeric().expect("test should succeed");
        assert_eq!(result.len(), 3);
        for (name, all_true) in &result {
            match name.as_str() {
                "all_nonzero" => assert!(*all_true),
                "has_zero" => assert!(!all_true),
                "has_nan" => assert!(!all_true),
                _ => panic!("Unexpected column"),
            }
        }
    }
    #[test]
    fn test_count_valid_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.count_valid().expect("test should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("a".to_string(), 3));
    }
    #[test]
    fn test_count_valid_string() {
        let mut df = DataFrame::new();
        df.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "".to_string(),
                    "Charlie".to_string(),
                    "".to_string(),
                ],
                Some("name".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.count_valid().expect("test should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("name".to_string(), 2));
    }
    #[test]
    fn test_count_valid_mixed() {
        let df = create_test_df();
        let result = df.count_valid().expect("test should succeed");
        assert_eq!(result.len(), 3);
        for (_, count) in &result {
            assert_eq!(*count, 5);
        }
    }
    #[test]
    fn test_reverse_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![3.0, 4.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "c".to_string(),
            Series::new(vec![5.0, 6.0], Some("c".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.reverse_columns().expect("test should succeed");
        let cols = result.column_names();
        assert_eq!(
            cols,
            vec!["c".to_string(), "b".to_string(), "a".to_string()]
        );
    }
    #[test]
    fn test_reverse_columns_single() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.reverse_columns().expect("test should succeed");
        let cols = result.column_names();
        assert_eq!(cols, vec!["a".to_string()]);
    }
    #[test]
    fn test_reverse_rows() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.reverse_rows().expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }
    #[test]
    fn test_reverse_rows_with_strings() {
        let mut df = DataFrame::new();
        df.add_column(
            "name".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string(), "C".to_string()],
                Some("name".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.reverse_rows().expect("test should succeed");
        let values = result
            .get_column_string_values("name")
            .expect("test should succeed");
        assert_eq!(
            values,
            vec!["C".to_string(), "B".to_string(), "A".to_string()]
        );
    }
    #[test]
    fn test_reverse_rows_preserves_columns() {
        let df = create_test_df();
        let result = df.reverse_rows().expect("test should succeed");
        assert_eq!(result.row_count(), 5);
        let a_values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(a_values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let b_values = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(b_values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let names = result
            .get_column_string_values("name")
            .expect("test should succeed");
        assert_eq!(names[0], "Eve");
        assert_eq!(names[4], "Alice");
    }
    #[test]
    fn test_notna() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.notna("a").expect("test should succeed");
        assert_eq!(result, vec![true, false, true, false, true]);
    }
    #[test]
    fn test_melt_basic() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string()],
                Some("id".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0], Some("x".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "y".to_string(),
            Series::new(vec![3.0, 4.0], Some("y".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .melt(&["id"], None, "variable", "value")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 4);
        assert!(result.contains_column("id"));
        assert!(result.contains_column("variable"));
        assert!(result.contains_column("value"));
    }
}
