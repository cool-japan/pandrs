//! Test module part 3 for pandas compatibility functions
//!
//! Split from functions.rs to comply with <2000 lines policy.

#[cfg(test)]
mod tests {
    use crate::dataframe::base::DataFrame;
    use crate::dataframe::pandas_compat::trait_def::PandasCompatExt;
    use crate::dataframe::pandas_compat::types::SeriesValue;
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
    fn test_agg_multiple() {
        let df = create_test_df();
        let result = df
            .agg("a", &["sum", "mean", "min", "max", "count"])
            .expect("test should succeed");
        assert_eq!(result.get("sum").expect("test should succeed"), &15.0);
        assert_eq!(result.get("mean").expect("test should succeed"), &3.0);
        assert_eq!(result.get("min").expect("test should succeed"), &1.0);
        assert_eq!(result.get("max").expect("test should succeed"), &5.0);
        assert_eq!(result.get("count").expect("test should succeed"), &5.0);
    }
    #[test]
    fn test_agg_statistics() {
        let df = create_test_df();
        let result = df
            .agg("a", &["std", "var", "median"])
            .expect("test should succeed");
        assert!((result.get("std").expect("test should succeed") - 1.5811).abs() < 0.001);
        assert_eq!(result.get("var").expect("test should succeed"), &2.5);
        assert_eq!(result.get("median").expect("test should succeed"), &3.0);
    }
    #[test]
    fn test_dtypes() {
        let df = create_test_df();
        let dtypes = df.dtypes();
        assert!(dtypes
            .iter()
            .any(|(name, dtype)| name == "a" && dtype == "float64"));
        assert!(dtypes
            .iter()
            .any(|(name, dtype)| name == "name" && dtype == "object"));
    }
    #[test]
    fn test_set_values() {
        let df = create_test_df();
        let result = df
            .set_values("a", &[0, 2, 4], &[100.0, 300.0, 500.0])
            .expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![100.0, 2.0, 300.0, 4.0, 500.0]);
    }
    #[test]
    fn test_set_values_mismatch() {
        let df = create_test_df();
        let result = df.set_values("a", &[0, 1], &[100.0]);
        assert!(result.is_err());
    }
    #[test]
    fn test_query_eq() {
        let df = create_test_df();
        let result = df.query_eq("a", 3.0).expect("test should succeed");
        assert_eq!(result.row_count(), 1);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 3.0);
    }
    #[test]
    fn test_query_gt() {
        let df = create_test_df();
        let result = df.query_gt("a", 3.0).expect("test should succeed");
        assert_eq!(result.row_count(), 2);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!(values.iter().all(|&v| v > 3.0));
    }
    #[test]
    fn test_query_lt() {
        let df = create_test_df();
        let result = df.query_lt("a", 3.0).expect("test should succeed");
        assert_eq!(result.row_count(), 2);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!(values.iter().all(|&v| v < 3.0));
    }
    #[test]
    fn test_query_contains() {
        let df = create_test_df();
        let result = df
            .query_contains("name", "li")
            .expect("test should succeed");
        assert_eq!(result.row_count(), 2);
    }
    #[test]
    fn test_select_columns() {
        let df = create_test_df();
        let result = df
            .select_columns(&["a", "name"])
            .expect("test should succeed");
        let a_values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(a_values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let name_values = result
            .get_column_string_values("name")
            .expect("test should succeed");
        assert_eq!(name_values[0], "Alice");
        assert!(result.get_column_numeric_values("b").is_err());
    }
    #[test]
    fn test_select_columns_nonexistent() {
        let df = create_test_df();
        let result = df.select_columns(&["a", "nonexistent"]);
        assert!(result.is_err());
    }
    #[test]
    fn test_add_scalar() {
        let df = create_test_df();
        let result = df.add_scalar("a", 10.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    }
    #[test]
    fn test_mul_scalar() {
        let df = create_test_df();
        let result = df.mul_scalar("a", 2.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0, 10.0]);
    }
    #[test]
    fn test_sub_scalar() {
        let df = create_test_df();
        let result = df.sub_scalar("a", 1.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_div_scalar() {
        let df = create_test_df();
        let result = df.div_scalar("a", 2.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    }
    #[test]
    fn test_pow() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![2.0, 3.0, 4.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.pow("a", 2.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![4.0, 9.0, 16.0]);
    }
    #[test]
    fn test_sqrt() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![4.0, 9.0, 16.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.sqrt("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }
    #[test]
    fn test_log() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![
                    1.0,
                    std::f64::consts::E,
                    std::f64::consts::E * std::f64::consts::E,
                ],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.log("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!((values[0] - 0.0).abs() < 0.001);
        assert!((values[1] - 1.0).abs() < 0.001);
        assert!((values[2] - 2.0).abs() < 0.001);
    }
    #[test]
    fn test_exp() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.exp("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!((values[0] - 1.0).abs() < 0.001);
        assert!((values[1] - std::f64::consts::E).abs() < 0.001);
    }
    #[test]
    fn test_col_add() {
        let df = create_test_df();
        let result = df.col_add("a", "b", "sum").expect("test should succeed");
        assert!(result.contains_column("sum"));
        let values = result
            .get_column_numeric_values("sum")
            .expect("test should succeed");
        assert_eq!(values, vec![6.0, 6.0, 6.0, 6.0, 6.0]);
    }
    #[test]
    fn test_col_mul() {
        let df = create_test_df();
        let result = df
            .col_mul("a", "b", "product")
            .expect("test should succeed");
        assert!(result.contains_column("product"));
        let values = result
            .get_column_numeric_values("product")
            .expect("test should succeed");
        assert_eq!(values, vec![5.0, 8.0, 9.0, 8.0, 5.0]);
    }
    #[test]
    fn test_col_sub() {
        let df = create_test_df();
        let result = df.col_sub("a", "b", "diff").expect("test should succeed");
        assert!(result.contains_column("diff"));
        let values = result
            .get_column_numeric_values("diff")
            .expect("test should succeed");
        assert_eq!(values, vec![-4.0, -2.0, 0.0, 2.0, 4.0]);
    }
    #[test]
    fn test_col_div() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![2.0, 4.0, 5.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.col_div("a", "b", "ratio").expect("test should succeed");
        let values = result
            .get_column_numeric_values("ratio")
            .expect("test should succeed");
        assert_eq!(values, vec![5.0, 5.0, 6.0]);
    }
    #[test]
    fn test_iterrows() {
        let df = create_test_df();
        let rows = df.iterrows();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].0, 0);
        let first_row = &rows[0].1;
        assert!(first_row.contains_key("a"));
        if let SeriesValue::Float(val) = first_row.get("a").expect("test should succeed") {
            assert_eq!(*val, 1.0);
        }
    }
    #[test]
    fn test_at() {
        let df = create_test_df();
        let val = df.at(2, "a").expect("test should succeed");
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 3.0);
        }
        let val = df.at(0, "name").expect("test should succeed");
        if let SeriesValue::String(s) = val {
            assert_eq!(s, "Alice");
        }
        assert!(df.at(100, "a").is_err());
    }
    #[test]
    fn test_iat() {
        let df = create_test_df();
        let val = df.iat(2, 0).expect("test should succeed");
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 3.0);
        }
        assert!(df.iat(0, 100).is_err());
    }
    #[test]
    fn test_drop_rows() {
        let df = create_test_df();
        let result = df.drop_rows(&[1, 3]).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }
    #[test]
    fn test_set_index() {
        let df = create_test_df();
        let (result, index) = df.set_index("name", true).expect("test should succeed");
        assert_eq!(index.len(), 5);
        assert_eq!(index[0], "Alice");
        assert!(!result.contains_column("name"));
    }
    #[test]
    fn test_reset_index() {
        let df = create_test_df();
        let index_vals: Vec<String> = vec![
            "idx0".to_string(),
            "idx1".to_string(),
            "idx2".to_string(),
            "idx3".to_string(),
            "idx4".to_string(),
        ];
        let result = df
            .reset_index(Some(&index_vals), "index")
            .expect("test should succeed");
        assert!(result.contains_column("index"));
        let vals = result
            .get_column_string_values("index")
            .expect("test should succeed");
        assert_eq!(vals[0], "idx0");
    }
    #[test]
    fn test_to_records() {
        let df = create_test_df();
        let records = df.to_records();
        assert_eq!(records.len(), 5);
        assert!(records[0].contains_key("a"));
    }
    #[test]
    fn test_items() {
        let df = create_test_df();
        let items = df.items();
        assert!(!items.is_empty());
        let a_item = items
            .iter()
            .find(|(name, _)| name == "a")
            .expect("test should succeed");
        assert_eq!(a_item.1.len(), 5);
    }
    #[test]
    fn test_update() {
        let df = create_test_df();
        let mut other = DataFrame::new();
        other
            .add_column(
                "a".to_string(),
                Series::new(vec![100.0, 200.0], Some("a".to_string()))
                    .expect("test should succeed"),
            )
            .expect("test should succeed");
        let result = df.update(&other).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 100.0);
        assert_eq!(values[1], 200.0);
        assert_eq!(values[2], 3.0);
    }
    #[test]
    fn test_shape() {
        let df = create_test_df();
        let (rows, cols) = df.shape();
        assert_eq!(rows, 5);
        assert_eq!(cols, 3);
    }
    #[test]
    fn test_size() {
        let df = create_test_df();
        assert_eq!(df.size(), 15);
    }
    #[test]
    fn test_empty() {
        let df = create_test_df();
        assert!(!df.empty());
        let empty_df = DataFrame::new();
        assert!(empty_df.empty());
    }
    #[test]
    fn test_first_last_row() {
        let df = create_test_df();
        let first = df.first_row().expect("test should succeed");
        if let SeriesValue::Float(v) = first.get("a").expect("test should succeed") {
            assert_eq!(*v, 1.0);
        }
        let last = df.last_row().expect("test should succeed");
        if let SeriesValue::Float(v) = last.get("a").expect("test should succeed") {
            assert_eq!(*v, 5.0);
        }
    }
    #[test]
    fn test_get_value() {
        let df = create_test_df();
        let val = df.get_value(0, "a", SeriesValue::Float(0.0));
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 1.0);
        }
        let val = df.get_value(100, "a", SeriesValue::Float(999.0));
        if let SeriesValue::Float(v) = val {
            assert_eq!(v, 999.0);
        }
    }
    #[test]
    fn test_get_column_by_index() {
        let df = create_test_df();
        let (name, values) = df.get_column_by_index(0).expect("test should succeed");
        assert!(!name.is_empty());
        assert_eq!(values.len(), 5);
        assert!(df.get_column_by_index(100).is_err());
    }
    #[test]
    fn test_swap_columns() {
        let df = create_test_df();
        let original_a = df
            .get_column_numeric_values("a")
            .expect("test should succeed");
        let original_b = df
            .get_column_numeric_values("b")
            .expect("test should succeed");
        let result = df.swap_columns("a", "b").expect("test should succeed");
        let new_a = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        let new_b = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(new_a, original_b);
        assert_eq!(new_b, original_a);
    }
    #[test]
    fn test_sort_columns() {
        let mut df = DataFrame::new();
        df.add_column(
            "c".to_string(),
            Series::new(vec![1.0, 2.0], Some("c".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 4.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![5.0, 6.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.sort_columns(true).expect("test should succeed");
        let cols = result.column_names();
        assert_eq!(cols[0], "a");
        assert_eq!(cols[1], "b");
        assert_eq!(cols[2], "c");
    }
    #[test]
    fn test_rename_column() {
        let df = create_test_df();
        let result = df.rename_column("a", "new_a").expect("test should succeed");
        assert!(result.contains_column("new_a"));
        assert!(!result.contains_column("a"));
    }
    #[test]
    fn test_to_categorical() {
        let df = create_test_df();
        let (result, mapping) = df.to_categorical("name").expect("test should succeed");
        let codes = result
            .get_column_numeric_values("name")
            .expect("test should succeed");
        assert_eq!(codes.len(), 5);
        assert!(mapping.contains_key("Alice"));
    }
    #[test]
    fn test_row_hash() {
        let df = create_test_df();
        let hashes = df.row_hash();
        assert_eq!(hashes.len(), 5);
        let unique: std::collections::HashSet<u64> = hashes.iter().copied().collect();
        assert_eq!(unique.len(), 5);
    }
    #[test]
    fn test_sample_frac() {
        let df = create_test_df();
        let result = df.sample_frac(0.4, false).expect("test should succeed");
        assert_eq!(result.row_count(), 2);
    }
    #[test]
    fn test_take() {
        let df = create_test_df();
        let result = df.take(&[0, 2, 4]).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 3.0, 5.0]);
    }
    #[test]
    fn test_duplicated_rows() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let dupes = df
            .duplicated_rows(None, "first")
            .expect("test should succeed");
        assert_eq!(dupes.len(), 5);
    }
    #[test]
    fn test_get_column_as_f64() {
        let df = create_test_df();
        let values: Vec<f64> =
            PandasCompatExt::get_column_as_f64(&df, "a").expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_get_column_as_string() {
        let df = create_test_df();
        let values = df
            .get_column_as_string("name")
            .expect("test should succeed");
        assert_eq!(values[0], "Alice");
        let values = df.get_column_as_string("a").expect("test should succeed");
        assert_eq!(values[0], "1");
    }
    #[test]
    fn test_corr_columns() {
        let df = create_test_df();
        let corr = df.corr_columns("a", "b").expect("test should succeed");
        assert!((corr - (-1.0)).abs() < 0.0001);
    }
    #[test]
    fn test_cov_columns() {
        let df = create_test_df();
        let cov = df.cov_columns("a", "b").expect("test should succeed");
        assert!(cov < 0.0);
    }
    #[test]
    fn test_var_column() {
        let df = create_test_df();
        let var = df.var_column("a", 0).expect("test should succeed");
        assert_eq!(var, 2.0);
    }
    #[test]
    fn test_std_column() {
        let df = create_test_df();
        let std = df.std_column("a", 0).expect("test should succeed");
        assert!((std - std::f64::consts::SQRT_2).abs() < 0.0001);
    }
    #[test]
    fn test_str_lower() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["HELLO".to_string(), "WORLD".to_string()],
                Some("text".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.str_lower("text").expect("test should succeed");
        let values = result
            .get_column_string_values("text")
            .expect("test should succeed");
        assert_eq!(values[0], "hello");
        assert_eq!(values[1], "world");
    }
    #[test]
    fn test_str_upper() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["hello".to_string(), "world".to_string()],
                Some("text".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.str_upper("text").expect("test should succeed");
        let values = result
            .get_column_string_values("text")
            .expect("test should succeed");
        assert_eq!(values[0], "HELLO");
        assert_eq!(values[1], "WORLD");
    }
    #[test]
    fn test_str_strip() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["  hello  ".to_string(), "  world  ".to_string()],
                Some("text".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.str_strip("text").expect("test should succeed");
        let values = result
            .get_column_string_values("text")
            .expect("test should succeed");
        assert_eq!(values[0], "hello");
        assert_eq!(values[1], "world");
    }
    #[test]
    fn test_str_contains() {
        let df = create_test_df();
        let contains = df.str_contains("name", "li").expect("test should succeed");
        assert_eq!(contains.iter().filter(|&&b| b).count(), 2);
    }
    #[test]
    fn test_str_replace() {
        let df = create_test_df();
        let result = df
            .str_replace("name", "Alice", "Alicia")
            .expect("test should succeed");
        let values = result
            .get_column_string_values("name")
            .expect("test should succeed");
        assert_eq!(values[0], "Alicia");
    }
    #[test]
    fn test_str_split() {
        let mut df = DataFrame::new();
        df.add_column(
            "text".to_string(),
            Series::new(
                vec!["a,b,c".to_string(), "d,e,f".to_string()],
                Some("text".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let splits = df.str_split("text", ",").expect("test should succeed");
        assert_eq!(splits[0], vec!["a", "b", "c"]);
        assert_eq!(splits[1], vec!["d", "e", "f"]);
    }
    #[test]
    fn test_str_len() {
        let df = create_test_df();
        let lengths = df.str_len("name").expect("test should succeed");
        assert_eq!(lengths[0], 5);
        assert_eq!(lengths[1], 3);
    }
    #[test]
    fn test_combine() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df1
            .combine(&df2, |v1, v2| v1.unwrap_or(0.0) + v2.unwrap_or(0.0))
            .expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![11.0, 22.0, 33.0]);
    }
    #[test]
    fn test_lookup() {
        let df = create_test_df();
        let mut lookup_df = DataFrame::new();
        lookup_df
            .add_column(
                "key".to_string(),
                Series::new(
                    vec!["Alice".to_string(), "Bob".to_string()],
                    Some("key".to_string()),
                )
                .expect("test should succeed"),
            )
            .expect("test should succeed");
        lookup_df
            .add_column(
                "value".to_string(),
                Series::new(
                    vec!["A".to_string(), "B".to_string()],
                    Some("value".to_string()),
                )
                .expect("test should succeed"),
            )
            .expect("test should succeed");
        let result = df
            .lookup("name", &lookup_df, "key", "value")
            .expect("test should succeed");
        assert!(result.contains_column("value_result"));
    }
    #[test]
    fn test_sem() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.sem("a", 1).expect("test should succeed");
        assert!((result - 0.707).abs() < 0.01);
    }
    #[test]
    fn test_mad() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.mad("a").expect("test should succeed");
        assert!((result - 1.2).abs() < 0.0001);
    }
    #[test]
    fn test_ffill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::NAN, f64::NAN, 4.0, f64::NAN],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.ffill("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 1.0);
        assert_eq!(values[2], 1.0);
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 4.0);
    }
    #[test]
    fn test_bfill() {
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
        let result = df.bfill("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values[0], 3.0);
        assert_eq!(values[1], 3.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 5.0);
        assert_eq!(values[4], 5.0);
    }
    #[test]
    fn test_pct_rank() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.pct_rank("a").expect("test should succeed");
        assert!((result[1] - 0.0).abs() < 0.01);
        assert!((result[4] - 1.0).abs() < 0.01);
    }
    #[test]
    fn test_abs_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-1.0, -2.0, 3.0, -4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.abs_column("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    #[test]
    fn test_round_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.234, 2.567, 3.891], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.round_column("a", 2).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.23, 2.57, 3.89]);
    }
    #[test]
    fn test_argmax_argmin() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 1.0, 4.0, 1.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        assert_eq!(df.argmax("a").expect("test should succeed"), 4);
        assert_eq!(df.argmin("a").expect("test should succeed"), 1);
    }
    #[test]
    fn test_gt_lt_ge_le() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let gt = df.gt("a", 3.0).expect("test should succeed");
        assert_eq!(gt, vec![false, false, false, true, true]);
        let ge = df.ge("a", 3.0).expect("test should succeed");
        assert_eq!(ge, vec![false, false, true, true, true]);
        let lt = df.lt("a", 3.0).expect("test should succeed");
        assert_eq!(lt, vec![true, true, false, false, false]);
        let le = df.le("a", 3.0).expect("test should succeed");
        assert_eq!(le, vec![true, true, true, false, false]);
    }
    #[test]
    fn test_eq_ne_value() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 2.0, 1.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let eq = df.eq_value("a", 2.0).expect("test should succeed");
        assert_eq!(eq, vec![false, true, false, true, false]);
        let ne = df.ne_value("a", 2.0).expect("test should succeed");
        assert_eq!(ne, vec![true, false, true, false, true]);
    }
    #[test]
    fn test_clip_lower_upper() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let clipped = df.clip_lower("a", 2.5).expect("test should succeed");
        let values = clipped
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.5, 2.5, 3.0, 4.0, 5.0]);
        let clipped = df.clip_upper("a", 3.5).expect("test should succeed");
        let values = clipped
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 3.5, 3.5]);
    }
    #[test]
    fn test_any_all_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![0.0, 1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "c".to_string(),
            Series::new(vec![0.0, 0.0, 0.0], Some("c".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        assert!(df.any_column("a").expect("test should succeed"));
        assert!(!df.all_column("a").expect("test should succeed"));
        assert!(df.all_column("b").expect("test should succeed"));
        assert!(!df.any_column("c").expect("test should succeed"));
    }
    #[test]
    fn test_count_na() {
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
        assert_eq!(df.count_na("a").expect("test should succeed"), 3);
    }
    #[test]
    fn test_prod() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        assert_eq!(df.prod("a").expect("test should succeed"), 24.0);
    }
    #[test]
    fn test_coalesce() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, f64::NAN], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0, 30.0, 40.0], Some("b".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.coalesce("a", "b", "c").expect("test should succeed");
        let values = result
            .get_column_numeric_values("c")
            .expect("test should succeed");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 20.0);
        assert_eq!(values[2], 3.0);
        assert_eq!(values[3], 40.0);
    }
    #[test]
    fn test_first_last_valid() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![f64::NAN, f64::NAN, 3.0, 4.0, f64::NAN],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        assert_eq!(df.first_valid("a").expect("test should succeed"), 3.0);
        assert_eq!(df.last_valid("a").expect("test should succeed"), 4.0);
    }
    #[test]
    fn test_column_arithmetic() {
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
        let result = df
            .add_columns("a", "b", "sum")
            .expect("test should succeed");
        assert_eq!(
            result
                .get_column_numeric_values("sum")
                .expect("test should succeed"),
            vec![11.0, 22.0, 33.0]
        );
        let result = df
            .sub_columns("b", "a", "diff")
            .expect("test should succeed");
        assert_eq!(
            result
                .get_column_numeric_values("diff")
                .expect("test should succeed"),
            vec![9.0, 18.0, 27.0]
        );
        let result = df
            .mul_columns("a", "b", "prod")
            .expect("test should succeed");
        assert_eq!(
            result
                .get_column_numeric_values("prod")
                .expect("test should succeed"),
            vec![10.0, 40.0, 90.0]
        );
        let result = df
            .div_columns("b", "a", "quot")
            .expect("test should succeed");
        assert_eq!(
            result
                .get_column_numeric_values("quot")
                .expect("test should succeed"),
            vec![10.0, 10.0, 10.0]
        );
    }
    #[test]
    fn test_mod_floordiv() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![7.0, 10.0, 15.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.mod_column("a", 3.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 1.0, 0.0]);
        let result = df.floordiv("a", 3.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.0, 3.0, 5.0]);
    }
    #[test]
    fn test_neg_sign() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-2.0, 0.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.neg("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.0, 0.0, -3.0]);
        let signs = df.sign("a").expect("test should succeed");
        assert_eq!(signs, vec![-1, 0, 1]);
    }
    #[test]
    fn test_is_finite_infinite() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let finite = df.is_finite("a").expect("test should succeed");
        assert_eq!(finite, vec![true, false, false, false]);
        let infinite = df.is_infinite("a").expect("test should succeed");
        assert_eq!(infinite, vec![false, true, true, false]);
    }
    #[test]
    fn test_replace_inf() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.replace_inf("a", 0.0).expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 0.0, 3.0, 0.0]);
    }
    #[test]
    fn test_str_startswith_endswith() {
        let df = create_test_df();
        let starts = df.str_startswith("name", "A").expect("test should succeed");
        assert_eq!(starts[0], true);
        assert_eq!(starts[1], false);
        let ends = df.str_endswith("name", "e").expect("test should succeed");
        assert_eq!(ends[0], true);
        assert_eq!(ends[1], false);
    }
    #[test]
    fn test_str_pad() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["a".to_string(), "bb".to_string(), "ccc".to_string()],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.str_pad_left("a", 5, '0').expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec!["0000a", "000bb", "00ccc"]);
        let result = df.str_pad_right("a", 5, '-').expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec!["a----", "bb---", "ccc--"]);
    }
    #[test]
    fn test_str_slice() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["hello".to_string(), "world".to_string()],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.str_slice("a", 0, Some(3)).expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec!["hel", "wor"]);
        let result = df.str_slice("a", 2, None).expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec!["llo", "rld"]);
    }
    #[test]
    fn test_floor_ceil_trunc() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.7, -1.7, 2.3, -2.3], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.floor("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, -2.0, 2.0, -3.0]);
        let result = df.ceil("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![2.0, -1.0, 3.0, -2.0]);
        let result = df.trunc("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, -1.0, 2.0, -2.0]);
    }
    #[test]
    fn test_fract_reciprocal() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.5, 2.7, 4.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fract("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!((values[0] - 0.5).abs() < 0.0001);
        assert!((values[1] - 0.7).abs() < 0.0001);
        assert!((values[2] - 0.0).abs() < 0.0001);
        let result = df.reciprocal("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert!((values[0] - 0.6667).abs() < 0.001);
        assert!((values[1] - 0.3704).abs() < 0.001);
        assert!((values[2] - 0.25).abs() < 0.0001);
    }
    #[test]
    fn test_count_value() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 1.0, 3.0, 1.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        assert_eq!(df.count_value("a", 1.0).expect("test should succeed"), 3);
        assert_eq!(df.count_value("a", 2.0).expect("test should succeed"), 1);
        assert_eq!(df.count_value("a", 5.0).expect("test should succeed"), 0);
    }
    #[test]
    fn test_fillna_zero() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0, f64::NAN], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.fillna_zero("a").expect("test should succeed");
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 0.0, 3.0, 0.0]);
    }
    #[test]
    fn test_nunique_all() {
        let df = create_test_df();
        let result = df.nunique_all().expect("test should succeed");
        assert_eq!(result.get("a").expect("test should succeed"), &5);
        assert_eq!(result.get("name").expect("test should succeed"), &5);
    }
    #[test]
    fn test_is_between() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df
            .is_between("a", 2.0, 4.0, true)
            .expect("test should succeed");
        assert_eq!(result, vec![false, true, true, true, false]);
        let result = df
            .is_between("a", 2.0, 4.0, false)
            .expect("test should succeed");
        assert_eq!(result, vec![false, false, true, false, false]);
    }
    #[test]
    fn test_str_count() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["aaa".to_string(), "aba".to_string(), "bbb".to_string()],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let counts = df.str_count("a", "a").expect("test should succeed");
        assert_eq!(counts, vec![3, 2, 0]);
    }
    #[test]
    fn test_str_repeat() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["ab".to_string(), "cd".to_string()],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.str_repeat("a", 3).expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec!["ababab", "cdcdcd"]);
    }
    #[test]
    fn test_str_center_zfill() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(
                vec!["ab".to_string(), "c".to_string()],
                Some("a".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.str_center("a", 5, '-').expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec!["-ab--", "--c--"]);
        let result = df.str_zfill("a", 5).expect("test should succeed");
        let values = result
            .get_column_string_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec!["000ab", "0000c"]);
    }
    #[test]
    fn test_column_type_detection() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "name".to_string(),
            Series::new(
                vec![
                    "Alice".to_string(),
                    "Bob".to_string(),
                    "Charlie".to_string(),
                ],
                Some("name".to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        let numeric_result = df.get_column_numeric_values("a");
        assert!(numeric_result.is_ok(), "Should get numeric values from 'a'");
        let string_result = df.get_column_string_values("name");
        assert!(
            string_result.is_ok(),
            "Should get string values from 'name'"
        );
    }
    #[test]
    fn test_has_nulls() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, f64::NAN, 3.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        assert!(df.has_nulls("a").expect("test should succeed"));
        assert!(!df.has_nulls("b").expect("test should succeed"));
    }
    #[test]
    fn test_describe_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let stats = df.describe_column("a").expect("test should succeed");
        assert_eq!(stats.get("count").expect("test should succeed"), &5.0);
        assert!((stats.get("mean").expect("test should succeed") - 3.0).abs() < 0.0001);
        assert!((stats.get("min").expect("test should succeed") - 1.0).abs() < 0.0001);
        assert!((stats.get("max").expect("test should succeed") - 5.0).abs() < 0.0001);
    }
    #[test]
    fn test_range_abs_sum() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![-2.0, 1.0, 5.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        assert_eq!(df.range("a").expect("test should succeed"), 7.0);
        assert_eq!(df.abs_sum("a").expect("test should succeed"), 8.0);
    }
    #[test]
    fn test_is_unique() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df.add_column(
            "b".to_string(),
            Series::new(vec![1.0, 1.0, 2.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        assert!(df.is_unique("a").expect("test should succeed"));
        assert!(!df.is_unique("b").expect("test should succeed"));
    }
    #[test]
    fn test_mode_with_count() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 2.0, 3.0, 2.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let (mode, count) = df.mode_with_count("a").expect("test should succeed");
        assert_eq!(mode, 2.0);
        assert_eq!(count, 3);
    }
    #[test]
    fn test_geometric_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 4.0, 8.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.geometric_mean("a").expect("test should succeed");
        assert!((result - 2.828).abs() < 0.01);
    }
    #[test]
    fn test_harmonic_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 4.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.harmonic_mean("a").expect("test should succeed");
        assert!((result - 1.714).abs() < 0.01);
    }
    #[test]
    fn test_iqr() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.iqr("a").expect("test should succeed");
        assert!(result > 1.0 && result < 3.0);
    }
    #[test]
    fn test_cv() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![10.0, 20.0, 30.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.cv("a").expect("test should succeed");
        assert!(result > 0.0 && result < 1.0);
    }
    #[test]
    fn test_percentile_value() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let p50 = df.percentile_value("a", 0.5).expect("test should succeed");
        assert!((p50 - 3.0).abs() < 0.5);
        let p0 = df.percentile_value("a", 0.0).expect("test should succeed");
        assert_eq!(p0, 1.0);
        let p100 = df.percentile_value("a", 1.0).expect("test should succeed");
        assert_eq!(p100, 5.0);
    }
    #[test]
    fn test_trimmed_mean() {
        let mut df = DataFrame::new();
        df.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 100.0], Some("a".to_string()))
                .expect("test should succeed"),
        )
        .expect("test should succeed");
        let result = df.trimmed_mean("a", 0.2).expect("test should succeed");
        assert!((result - 3.0).abs() < 0.1);
    }
}
