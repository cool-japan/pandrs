//! Test module part 2 for pandas compatibility functions
//!
//! Split from functions.rs to comply with <2000 lines policy.

#[cfg(test)]
mod tests {
    // `0.7071` in the expanding-std assertions is a rounded expected value
    // (close to FRAC_1_SQRT_2), i.e. test-fixture data, not a use of the
    // constant; silence `approx_constant` for the test module rather than
    // rewriting the asserted value.
    #![allow(clippy::approx_constant)]
    use crate::dataframe::base::DataFrame;
    use crate::dataframe::pandas_compat::trait_def::PandasCompatExt;
    use crate::series::Series;

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
    fn test_melt_with_value_vars() {
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
        df.add_column(
            "z".to_string(),
            Series::new(vec![5.0, 6.0], Some("z".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .melt(&["id"], Some(&["x", "y"]), "var", "val")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 4);
        assert!(result.contains_column("var"));
        assert!(result.contains_column("val"));
    }
    #[test]
    fn test_explode() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1.0, 2.0], Some("id".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "tags".to_string(),
            Series::new(
                vec!["a,b,c".to_string(), "x,y".to_string()],
                Some("tags".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.explode("tags", ",").expect("test should succeed");
        assert_eq!(result.row_count(), 5);
        let tags = result
            .get_column_string_values("tags")
            .expect("test should succeed");
        assert_eq!(tags, vec!["a", "b", "c", "x", "y"]);
    }
    #[test]
    fn test_explode_preserves_other_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1.0, 2.0], Some("id".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "items".to_string(),
            Series::new(
                vec!["a,b".to_string(), "c".to_string()],
                Some("items".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.explode("items", ",").expect("test should succeed");
        let ids = result
            .get_column_numeric_values("id")
            .expect("test should succeed");
        assert_eq!(ids, vec![1.0, 1.0, 2.0]);
    }
    #[test]
    fn test_duplicated_keep_first() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .duplicated(Some(&["a"]), "first")
            .expect("test should succeed");
        assert_eq!(result, vec![false, false, true, false, true]);
    }
    #[test]
    fn test_duplicated_keep_last() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .duplicated(Some(&["a"]), "last")
            .expect("test should succeed");
        assert_eq!(result, vec![true, true, false, false, false]);
    }
    #[test]
    fn test_duplicated_keep_none() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .duplicated(Some(&["a"]), "none")
            .expect("test should succeed");
        assert_eq!(result, vec![true, false, true, false]);
    }
    #[test]
    fn test_copy() {
        let df = create_test_df();
        let copied = df.copy();
        assert_eq!(df.row_count(), copied.row_count());
        assert_eq!(df.column_names(), copied.column_names());
    }
    #[test]
    fn test_to_dict() {
        let df = create_test_df();
        let dict = df.to_dict().expect("test should succeed");
        assert!(dict.contains_key("a"));
        assert!(dict.contains_key("b"));
        assert!(dict.contains_key("name"));
        assert_eq!(dict.get("name").expect("test should succeed").len(), 5);
    }
    #[test]
    fn test_first_valid_index() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, f64::NAN, 3.0, 4.0, 5.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.first_valid_index("a").expect("test should succeed");
        assert_eq!(result, Some(2));
    }
    #[test]
    fn test_first_valid_index_no_nan() {
        let df = create_test_df();
        let result = df.first_valid_index("a").expect("test should succeed");
        assert_eq!(result, Some(0));
    }
    #[test]
    fn test_first_valid_index_all_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![f64::NAN, f64::NAN, f64::NAN], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.first_valid_index("a").expect("test should succeed");
        assert_eq!(result, None);
    }
    #[test]
    fn test_last_valid_index() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, 2.0, 3.0, f64::NAN, f64::NAN],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.last_valid_index("a").expect("test should succeed");
        assert_eq!(result, Some(2));
    }
    #[test]
    fn test_product_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.product_all().expect("test should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, 24.0);
    }
    #[test]
    fn test_product_all_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.product_all().expect("test should succeed");
        assert_eq!(result[0].1, 6.0);
    }
    #[test]
    fn test_median_all() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.median_all().expect("test should succeed");
        assert_eq!(result[0].1, 3.0);
    }
    #[test]
    fn test_median_all_even() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.median_all().expect("test should succeed");
        assert_eq!(result[0].1, 2.5);
    }
    #[test]
    fn test_skew() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.skew("a").expect("test should succeed");
        assert!((result - 0.0).abs() < 0.1);
    }
    #[test]
    fn test_skew_insufficient_data() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.skew("a").expect("test should succeed");
        assert!(result.is_nan());
    }
    #[test]
    fn test_kurtosis() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.kurtosis("a").expect("test should succeed");
        assert!(!result.is_nan());
    }
    #[test]
    fn test_kurtosis_insufficient_data() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.kurtosis("a").expect("test should succeed");
        assert!(result.is_nan());
    }
    #[test]
    fn test_add_prefix() {
        let df = create_test_df();
        let result = df.add_prefix("col_").expect("test should succeed");
        assert!(result.contains_column("col_a"));
        assert!(result.contains_column("col_b"));
        assert!(result.contains_column("col_name"));
        assert!(!result.contains_column("a"));
    }
    #[test]
    fn test_add_suffix() {
        let df = create_test_df();
        let result = df.add_suffix("_new").expect("test should succeed");
        assert!(result.contains_column("a_new"));
        assert!(result.contains_column("b_new"));
        assert!(result.contains_column("name_new"));
        assert!(!result.contains_column("a"));
    }
    #[test]
    fn test_filter_by_mask() {
        let df = create_test_df();
        let mask = vec![true, false, true, false, true];
        let result = df.filter_by_mask(&mask).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }
    #[test]
    fn test_filter_by_mask_length_mismatch() {
        let df = create_test_df();
        let mask = vec![true, false];
        let result = df.filter_by_mask(&mask);
        assert!(result.is_err());
    }
    #[test]
    fn test_mode_numeric() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.mode_numeric("a").expect("test should succeed");
        assert_eq!(result, vec![3.0]);
    }
    #[test]
    fn test_mode_numeric_multiple() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 1.0, 2.0, 2.0, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.mode_numeric("a").expect("test should succeed");
        assert_eq!(result, vec![1.0, 2.0]);
    }
    #[test]
    fn test_mode_string() {
        let mut df = DataFrame::new();
        df.add_column(
            "cat".to_string(),
            Series::new(
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "a".to_string(),
                    "c".to_string(),
                ],
                Some("cat".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.mode_string("cat").expect("test should succeed");
        assert_eq!(result, vec!["a".to_string()]);
    }
    #[test]
    fn test_percentile() {
        let df = create_test_df();
        let p50 = df.percentile("a", 50.0).expect("test should succeed");
        assert_eq!(p50, 3.0);
    }
    #[test]
    fn test_ewma() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.ewma("a", 3).expect("test should succeed");
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 1.0);
        assert!(result[4] > result[3]);
    }
    #[test]
    fn test_ewma_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, 4.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.ewma("a", 2).expect("test should succeed");
        assert_eq!(result[0], 1.0);
        assert!(result[1].is_nan());
    }
    #[test]
    fn test_iloc() {
        let df = create_test_df();
        let row = df.iloc(2).expect("test should succeed");
        assert_eq!(row.get("a").expect("test should succeed"), "3");
        assert_eq!(row.get("name").expect("test should succeed"), "Charlie");
    }
    #[test]
    fn test_iloc_out_of_bounds() {
        let df = create_test_df();
        let result = df.iloc(100);
        assert!(result.is_err());
    }
    #[test]
    fn test_iloc_range() {
        let df = create_test_df();
        let result = df.iloc_range(1, 4).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_iloc_range_clamped() {
        let df = create_test_df();
        let result = df.iloc_range(3, 100).expect("test should succeed");
        assert_eq!(result.row_count(), 2);
    }
    #[test]
    fn test_iloc_range_invalid() {
        let df = create_test_df();
        let result = df.iloc_range(4, 2);
        assert!(result.is_err());
    }
    #[test]
    fn test_info() {
        let df = create_test_df();
        let info = df.info();
        assert!(info.contains("<DataFrame>"));
        assert!(info.contains("RangeIndex: 5 entries"));
        assert!(info.contains("Data columns (total 3 columns)"));
        assert!(info.contains("float64"));
        assert!(info.contains("object"));
        assert!(info.contains("memory usage:"));
    }
    #[test]
    fn test_info_empty() {
        let df = DataFrame::new();
        let info = df.info();
        assert!(info.contains("RangeIndex: 0 entries"));
        assert!(info.contains("Data columns (total 0 columns)"));
    }
    #[test]
    fn test_equals_same() {
        let df = create_test_df();
        assert!(df.equals(&df));
    }
    #[test]
    fn test_equals_identical() {
        let df1 = create_test_df();
        let df2 = create_test_df();
        assert!(df1.equals(&df2));
    }
    #[test]
    fn test_equals_different_values() {
        let df1 = create_test_df();
        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 99.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df2.add_column(
            "b".to_string(),
            Series::new(vec![5.0, 4.0, 3.0, 2.0, 1.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df2.add_column(
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
        assert!(!df1.equals(&df2));
    }
    #[test]
    fn test_equals_different_columns() {
        let df1 = create_test_df();
        let mut df2 = DataFrame::new();
        df2.add_column(
            "x".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        assert!(!df1.equals(&df2));
    }
    #[test]
    fn test_equals_nan_handling() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        assert!(df1.equals(&df2));
    }
    #[test]
    fn test_compare() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 5.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df1.compare(&df2).expect("test should succeed");
        let diff = result
            .get_column_numeric_values("a_diff")
            .expect("test should succeed");
        assert_eq!(diff[0], 0.0);
        assert_eq!(diff[1], -3.0);
        assert_eq!(diff[2], 0.0);
    }
    #[test]
    fn test_compare_different_rows() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df1.compare(&df2);
        assert!(result.is_err());
    }
    #[test]
    fn test_keys() {
        let df = create_test_df();
        let keys = df.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(keys.contains(&"name".to_string()));
    }
    #[test]
    fn test_pop_column() {
        let df = create_test_df();
        let (new_df, values) = df.pop_column("a").expect("test should succeed");
        assert!(!new_df.contains_column("a"));
        assert!(new_df.contains_column("b"));
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_pop_column_nonexistent() {
        let df = create_test_df();
        let result = df.pop_column("nonexistent");
        assert!(result.is_err());
    }
    #[test]
    fn test_insert_column() {
        let df = create_test_df();
        let new_df = df
            .insert_column(1, "new_col", vec![10.0, 20.0, 30.0, 40.0, 50.0])
            .expect("test should succeed");
        let cols = new_df.column_names();
        assert_eq!(cols.len(), 4);
        assert!(cols.contains(&"new_col".to_string()));
        let values = new_df
            .get_column_numeric_values("new_col")
            .expect("test should succeed");
        assert_eq!(values, vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    }
    #[test]
    fn test_insert_column_wrong_length() {
        let df = create_test_df();
        let result = df.insert_column(1, "bad_col", vec![1.0, 2.0]);
        assert!(result.is_err());
    }
    #[test]
    fn test_rolling_sum() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.rolling_sum("a", 3, None).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 9.0);
        assert_eq!(result[4], 12.0);
    }
    #[test]
    fn test_rolling_sum_min_periods() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .rolling_sum("a", 3, Some(1))
            .expect("test should succeed");
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 9.0);
        assert_eq!(result[4], 12.0);
    }
    #[test]
    fn test_rolling_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.rolling_mean("a", 3, None).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0);
        assert_eq!(result[3], 3.0);
        assert_eq!(result[4], 4.0);
    }
    #[test]
    fn test_rolling_mean_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .rolling_mean("a", 3, Some(2))
            .expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0);
        assert!((result[3] - 3.5).abs() < 0.001);
        assert_eq!(result[4], 4.0);
    }
    #[test]
    fn test_rolling_std() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.rolling_std("a", 3, None).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 1.0).abs() < 0.001);
        assert!((result[3] - 1.0).abs() < 0.001);
        assert!((result[4] - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_rolling_min() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.rolling_min("a", 3, None).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 1.0);
    }
    #[test]
    fn test_rolling_max() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.rolling_max("a", 3, None).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 4.0);
        assert_eq!(result[3], 4.0);
        assert_eq!(result[4], 5.0);
    }
    #[test]
    fn test_rolling_var() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.rolling_var("a", 3, None).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert!((result[2] - 1.0).abs() < 0.0001);
        assert!((result[3] - 1.0).abs() < 0.0001);
        assert!((result[4] - 1.0).abs() < 0.0001);
    }
    #[test]
    fn test_rolling_median() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .rolling_median("a", 3, None)
            .expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 4.0);
    }
    #[test]
    fn test_rolling_count() {
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
        let result = df.rolling_count("a", 3).expect("test should succeed");
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 2);
        assert_eq!(result[3], 1);
        assert_eq!(result[4], 2);
    }
    #[test]
    fn test_rolling_apply() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .rolling_apply("a", 3, |vals| vals.iter().map(|v| v * v).sum(), None)
            .expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 14.0);
        assert_eq!(result[3], 29.0);
        assert_eq!(result[4], 50.0);
    }
    #[test]
    fn test_expanding_var() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.expanding_var("a", 2).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!((result[1] - 0.5).abs() < 0.0001);
        assert!((result[2] - 1.0).abs() < 0.0001);
        assert!((result[3] - 1.6667).abs() < 0.001);
        assert!((result[4] - 2.5).abs() < 0.0001);
    }
    #[test]
    fn test_expanding_apply() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .expanding_apply("a", |vals| vals.iter().product(), 1)
            .expect("test should succeed");
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 24.0);
        assert_eq!(result[4], 120.0);
    }
    #[test]
    fn test_cumcount() {
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
        let result = df.cumcount("a").expect("test should succeed");
        assert_eq!(result, vec![1, 1, 2, 2, 3]);
    }
    #[test]
    fn test_nth_positive() {
        let df = create_test_df();
        let row = df.nth(2).expect("test should succeed");
        assert_eq!(row.get("a").expect("test should succeed"), "3");
        assert_eq!(row.get("name").expect("test should succeed"), "Charlie");
    }
    #[test]
    fn test_nth_negative() {
        let df = create_test_df();
        let row = df.nth(-1).expect("test should succeed");
        assert_eq!(row.get("a").expect("test should succeed"), "5");
        assert_eq!(row.get("name").expect("test should succeed"), "Eve");
    }
    #[test]
    fn test_nth_negative_second() {
        let df = create_test_df();
        let row = df.nth(-2).expect("test should succeed");
        assert_eq!(row.get("a").expect("test should succeed"), "4");
        assert_eq!(row.get("name").expect("test should succeed"), "David");
    }
    #[test]
    fn test_nth_out_of_bounds() {
        let df = create_test_df();
        let result = df.nth(100);
        assert!(result.is_err());
    }
    #[test]
    fn test_transform() {
        let df = create_test_df();
        let result = df.transform("a", |x| x * 2.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }
    #[test]
    fn test_transform_preserves_other_columns() {
        let df = create_test_df();
        let result = df.transform("a", |x| x * 2.0).expect("test should succeed");
        let b_values = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(b_values, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }
    #[test]
    fn test_crosstab() {
        let mut df = DataFrame::new();
        df.add_column(
            "gender".to_string(),
            Series::new(
                vec![
                    "M".to_string(),
                    "F".to_string(),
                    "M".to_string(),
                    "F".to_string(),
                    "M".to_string(),
                ],
                Some("gender".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "category".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                ],
                Some("category".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .crosstab("gender", "category")
            .expect("test should succeed");
        assert!(result.contains_column("gender"));
        assert!(result.contains_column("A"));
        assert!(result.contains_column("B"));
    }
    #[test]
    fn test_expanding_sum() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.expanding_sum("a", 1).expect("test should succeed");
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 10.0);
        assert_eq!(result[4], 15.0);
    }
    #[test]
    fn test_expanding_sum_min_periods() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.expanding_sum("a", 3).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 6.0);
        assert_eq!(result[3], 10.0);
        assert_eq!(result[4], 15.0);
    }
    #[test]
    fn test_expanding_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, 4.0, 6.0, 8.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.expanding_mean("a", 1).expect("test should succeed");
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
        assert_eq!(result[3], 5.0);
    }
    #[test]
    fn test_expanding_std() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.expanding_std("a", 2).expect("test should succeed");
        assert!(result[0].is_nan());
        assert!((result[1] - 0.7071).abs() < 0.001);
    }
    #[test]
    fn test_expanding_min() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.expanding_min("a", 1).expect("test should succeed");
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 1.0);
    }
    #[test]
    fn test_expanding_max() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.expanding_max("a", 1).expect("test should succeed");
        assert_eq!(result[0], 3.0);
        assert_eq!(result[1], 3.0);
        assert_eq!(result[2], 4.0);
        assert_eq!(result[3], 4.0);
        assert_eq!(result[4], 5.0);
    }
    #[test]
    fn test_align() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let mut df2 = DataFrame::new();
        df2.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let (aligned1, aligned2) = df1.align(&df2).expect("test should succeed");
        assert!(aligned1.contains_column("a") || aligned1.contains_column("b"));
        assert!(aligned2.contains_column("a") || aligned2.contains_column("b"));
    }
    #[test]
    fn test_reindex_columns() {
        let df = create_test_df();
        let result = df
            .reindex_columns(&["b", "a"])
            .expect("test should succeed");
        let cols = result.column_names();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0], "b");
        assert_eq!(cols[1], "a");
    }
    #[test]
    fn test_reindex_columns_with_missing() {
        let df = create_test_df();
        let result = df
            .reindex_columns(&["a", "nonexistent", "b"])
            .expect("test should succeed");
        let cols = result.column_names();
        assert_eq!(cols.len(), 3);
        assert!(result.contains_column("nonexistent"));
    }
    #[test]
    fn test_value_range() {
        let df = create_test_df();
        let (min, max) = df.value_range("a").expect("test should succeed");
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }
    #[test]
    fn test_value_range_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, 2.0, 5.0, f64::NAN, 1.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let (min, max) = df.value_range("a").expect("test should succeed");
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }
    #[test]
    fn test_zscore() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.zscore("a").expect("test should succeed");
        assert!(result[0] < 0.0);
        assert!((result[2]).abs() < 0.001);
        assert!(result[4] > 0.0);
    }
    #[test]
    fn test_zscore_insufficient_values() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.zscore("a");
        assert!(result.is_err());
    }
    #[test]
    fn test_normalize() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 50.0, 100.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.normalize("a").expect("test should succeed");
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.5);
        assert_eq!(result[2], 1.0);
    }
    #[test]
    fn test_normalize_with_negative() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-10.0, 0.0, 10.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.normalize("a").expect("test should succeed");
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.5);
        assert_eq!(result[2], 1.0);
    }
    #[test]
    fn test_normalize_constant_values() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![5.0, 5.0, 5.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.normalize("a");
        assert!(result.is_err());
    }
    #[test]
    fn test_cut() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.5, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.cut("a", 2).expect("test should succeed");
        assert_eq!(result.len(), 4);
        assert!(result[0].contains("1.00") || result[0].contains("("));
    }
    #[test]
    fn test_cut_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.cut("a", 2).expect("test should succeed");
        assert_eq!(result[1], "NaN");
    }
    #[test]
    fn test_cut_zero_bins() {
        let df = create_test_df();
        let result = df.cut("a", 0);
        assert!(result.is_err());
    }
    #[test]
    fn test_qcut() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.qcut("a", 4).expect("test should succeed");
        assert_eq!(result.len(), 8);
        assert!(result.iter().any(|s| s.starts_with("Q")));
    }
    #[test]
    fn test_qcut_with_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, 4.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.qcut("a", 2).expect("test should succeed");
        assert_eq!(result[1], "NaN");
    }
    #[test]
    fn test_qcut_zero_quantiles() {
        let df = create_test_df();
        let result = df.qcut("a", 0);
        assert!(result.is_err());
    }
    #[test]
    fn test_stack() {
        let df = create_test_df();
        let result = df.stack(Some(&["a", "b"])).expect("test should succeed");
        assert!(result.contains_column("row_index"));
        assert!(result.contains_column("variable"));
        assert!(result.contains_column("value"));
        assert_eq!(result.row_count(), 10);
    }
    #[test]
    fn test_stack_all_numeric() {
        let mut df = DataFrame::new();
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
        let result = df.stack(None).expect("test should succeed");
        assert_eq!(result.row_count(), 4);
    }
    #[test]
    fn test_unstack() {
        let mut df = DataFrame::new();
        df.add_column(
            "idx".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                    "B".to_string(),
                ],
                Some("idx".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "col".to_string(),
            Series::new(
                vec![
                    "X".to_string(),
                    "Y".to_string(),
                    "X".to_string(),
                    "Y".to_string(),
                ],
                Some("col".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "val".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("val".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .unstack("idx", "col", "val")
            .expect("test should succeed");
        assert!(result.contains_column("idx"));
        assert!(result.contains_column("X"));
        assert!(result.contains_column("Y"));
        assert_eq!(result.row_count(), 2);
    }
    #[test]
    fn test_pivot() {
        let mut df = DataFrame::new();
        df.add_column(
            "date".to_string(),
            Series::new(
                vec![
                    "Mon".to_string(),
                    "Mon".to_string(),
                    "Tue".to_string(),
                    "Tue".to_string(),
                ],
                Some("date".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "type".to_string(),
            Series::new(
                vec![
                    "A".to_string(),
                    "B".to_string(),
                    "A".to_string(),
                    "B".to_string(),
                ],
                Some("type".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "value".to_string(),
            Series::new(vec![10.0, 20.0, 30.0, 40.0], Some("value".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .pivot("date", "type", "value")
            .expect("test should succeed");
        assert!(result.contains_column("date"));
        assert!(result.contains_column("A"));
        assert!(result.contains_column("B"));
    }
    #[test]
    fn test_astype_to_int() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.7, 2.3, 3.9], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.astype("a", "int64").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_astype_to_string() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.astype("a", "string").expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], "1");
        assert_eq!(values[1], "2");
        assert_eq!(values[2], "3");
    }
    #[test]
    fn test_astype_to_bool() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 1.0, 5.0, 0.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.astype("a", "bool").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![0.0, 1.0, 1.0, 0.0]);
    }
    #[test]
    fn test_applymap() {
        let df = create_test_df();
        let result = df.applymap(|x| x * 2.0).expect("test should succeed");
        let a_values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(a_values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
        let b_values = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(b_values, vec![10.0, 8.0, 6.0, 4.0, 2.0]);
    }
}
