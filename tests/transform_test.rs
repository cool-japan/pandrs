#[cfg(test)]
mod tests {
    use pandrs::{DataFrame, MeltOptions, Series, StackOptions, UnstackOptions};

    // Helper function to clean DataBox string values
    fn clean_databox_value(value: &str) -> String {
        let trimmed = value.trim_start_matches("DataBox(\"").trim_end_matches("\")");
        let value_str = if trimmed.starts_with("DataBox(") {
            trimmed.trim_start_matches("DataBox(").trim_end_matches(")")
        } else {
            trimmed
        };
        value_str.trim_matches('"').to_string()
    }

    #[test]
    fn test_melt() {
        // Create test DataFrame
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "A".to_string(),
            Series::new(vec!["a1", "a2"], Some("A".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "B".to_string(),
            Series::new(vec!["b1", "b2"], Some("B".to_string())).unwrap(),
        )
        .unwrap();

        // Melt operation
        let options = MeltOptions {
            id_vars: Some(vec!["id".to_string()]),
            value_vars: Some(vec!["A".to_string(), "B".to_string()]),
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
        };

        let melted = df.melt(&options).unwrap();

        // Verification
        assert_eq!(melted.column_count(), 3); // id, variable, value
        assert_eq!(melted.row_count(), 4);    // 2 rows x 2 columns = 4 rows

        // Check column names
        let columns = melted.column_names();
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"variable".to_string()));
        assert!(columns.contains(&"value".to_string()));

        // Check data
        let id_col = melted.get_column("id").unwrap();
        let var_col = melted.get_column("variable").unwrap();
        let val_col = melted.get_column("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[0].to_string()), "A");
        assert_eq!(clean_databox_value(&val_col.values()[0].to_string()), "a1");

        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[1].to_string()), "B");
        assert_eq!(clean_databox_value(&val_col.values()[1].to_string()), "b1");

        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "2");
        assert_eq!(clean_databox_value(&var_col.values()[2].to_string()), "A");
        assert_eq!(clean_databox_value(&val_col.values()[2].to_string()), "a2");

        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "2");
        assert_eq!(clean_databox_value(&var_col.values()[3].to_string()), "B");
        assert_eq!(clean_databox_value(&val_col.values()[3].to_string()), "b2");
    }

    #[test]
    fn test_stack() {
        // Create test DataFrame
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "A".to_string(),
            Series::new(vec!["a1", "a2"], Some("A".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "B".to_string(),
            Series::new(vec!["b1", "b2"], Some("B".to_string())).unwrap(),
        )
        .unwrap();

        // Stack operation
        let options = StackOptions {
            columns: Some(vec!["A".to_string(), "B".to_string()]),
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
            dropna: false,
        };

        let stacked = df.stack(&options).unwrap();

        // Validation
        assert_eq!(stacked.column_count(), 3); // id, variable, value
        assert_eq!(stacked.row_count(), 4);    // 2 rows x 2 columns = 4 rows

        // Check data
        let id_col = stacked.get_column("id").unwrap();
        let var_col = stacked.get_column("variable").unwrap();
        let val_col = stacked.get_column("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[0].to_string()), "A");
        assert_eq!(clean_databox_value(&val_col.values()[0].to_string()), "a1");

        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "1");
        assert_eq!(clean_databox_value(&var_col.values()[1].to_string()), "B");
        assert_eq!(clean_databox_value(&val_col.values()[1].to_string()), "b1");
    }

    #[test]
    fn test_unstack() {
        // Create test long-format DataFrame
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(
                vec!["1", "1", "2", "2"],
                Some("id".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "variable".to_string(),
            Series::new(
                vec!["A", "B", "A", "B"],
                Some("variable".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "value".to_string(),
            Series::new(
                vec!["a1", "b1", "a2", "b2"],
                Some("value".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        // Unstack operation
        let options = UnstackOptions {
            var_column: "variable".to_string(),
            value_column: "value".to_string(),
            index_columns: Some(vec!["id".to_string()]),
            fill_value: None,
        };

        let unstacked = df.unstack(&options).unwrap();

        // Validation
        assert_eq!(unstacked.column_count(), 3); // id, A, B
        assert_eq!(unstacked.row_count(), 2);    // 2 rows (by id)

        // Check column names
        let columns = unstacked.column_names();
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"A".to_string()));
        assert!(columns.contains(&"B".to_string()));

        // Check data
        let id_col = unstacked.get_column("id").unwrap();
        let a_col = unstacked.get_column("A").unwrap();
        let b_col = unstacked.get_column("B").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&a_col.values()[0].to_string()), "a1");
        assert_eq!(clean_databox_value(&b_col.values()[0].to_string()), "b1");

        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "2");
        assert_eq!(clean_databox_value(&a_col.values()[1].to_string()), "a2");
        assert_eq!(clean_databox_value(&b_col.values()[1].to_string()), "b2");
    }

    #[test]
    fn test_conditional_aggregate() {
        // Create test DataFrame
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec!["Food", "Electronics", "Food", "Clothing"],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "sales".to_string(),
            Series::new(
                vec!["1000", "1500", "800", "1200"],
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        // Conditional aggregation: calculate totals by category for sales >= 1000
        let result = df
            .conditional_aggregate(
                "category",
                "sales",
                |row| {
                    if let Some(sales_str) = row.get("sales") {
                        if let Ok(sales) = sales_str.parse::<i32>() {
                            return sales >= 1000;
                        }
                    }
                    false
                },
                |values| {
                    let sum: i32 = values
                        .iter()
                        .filter_map(|v| v.parse::<i32>().ok())
                        .sum();
                    sum.to_string()
                },
            )
            .unwrap();

        // Verification
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 3); // Food, Electronics, Clothing

        // Check aggregate results
        let cat_col = result.get_column("category").unwrap();
        let agg_col = result.get_column("sales_agg").unwrap();

        // Check each category's aggregate value
        // Note: Order may depend on implementation, so check each category individually
        for i in 0..result.row_count() {
            let category = clean_databox_value(&cat_col.values()[i].to_string());
            let agg_value = clean_databox_value(&agg_col.values()[i].to_string());

            if category == "Food" {
                assert_eq!(agg_value, "1000"); // Only one Food item is >= 1000
            } else if category == "Electronics" {
                assert_eq!(agg_value, "1500");
            } else if category == "Clothing" {
                assert_eq!(agg_value, "1200");
            } else {
                panic!("Unexpected category: {}", category);
            }
        }
    }

    #[test]
    fn test_concat() {
        // First dataframe
        let mut df1 = DataFrame::new();
        df1.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df1.add_column(
            "value".to_string(),
            Series::new(vec!["a", "b"], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        // Second dataframe
        let mut df2 = DataFrame::new();
        df2.add_column(
            "id".to_string(),
            Series::new(vec!["3", "4"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "value".to_string(),
            Series::new(vec!["c", "d"], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        // Concatenation operation
        let concat_df = DataFrame::concat(&[&df1, &df2], true).unwrap();

        // Validation
        assert_eq!(concat_df.column_count(), 2);
        assert_eq!(concat_df.row_count(), 4);

        // Check columns
        let id_col = concat_df.get_column("id").unwrap();
        let value_col = concat_df.get_column("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&value_col.values()[0].to_string()), "a");
        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "2");
        assert_eq!(clean_databox_value(&value_col.values()[1].to_string()), "b");
        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "3");
        assert_eq!(clean_databox_value(&value_col.values()[2].to_string()), "c");
        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "4");
        assert_eq!(clean_databox_value(&value_col.values()[3].to_string()), "d");
    }

    #[test]
    fn test_concat_different_columns() {
        // First dataframe
        let mut df1 = DataFrame::new();
        df1.add_column(
            "id".to_string(),
            Series::new(vec!["1", "2"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df1.add_column(
            "A".to_string(),
            Series::new(vec!["a1", "a2"], Some("A".to_string())).unwrap(),
        )
        .unwrap();

        // Second dataframe (with different columns)
        let mut df2 = DataFrame::new();
        df2.add_column(
            "id".to_string(),
            Series::new(vec!["3", "4"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "B".to_string(),
            Series::new(vec!["b3", "b4"], Some("B".to_string())).unwrap(),
        )
        .unwrap();

        // Concatenation operation
        let concat_df = DataFrame::concat(&[&df1, &df2], true).unwrap();

        // Validation
        assert_eq!(concat_df.column_count(), 3); // id, A, B
        assert_eq!(concat_df.row_count(), 4);

        // Check columns
        let id_col = concat_df.get_column("id").unwrap();
        let a_col = concat_df.get_column("A").unwrap();
        let b_col = concat_df.get_column("B").unwrap();

        // id column
        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "2");
        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "3");
        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "4");

        // A column (df2 doesn't have column A, so empty values are used)
        assert_eq!(clean_databox_value(&a_col.values()[0].to_string()), "a1");
        assert_eq!(clean_databox_value(&a_col.values()[1].to_string()), "a2");
        assert_eq!(clean_databox_value(&a_col.values()[2].to_string()), "");
        assert_eq!(clean_databox_value(&a_col.values()[3].to_string()), "");

        // B column (df1 doesn't have column B, so empty values are used)
        assert_eq!(clean_databox_value(&b_col.values()[0].to_string()), "");
        assert_eq!(clean_databox_value(&b_col.values()[1].to_string()), "");
        assert_eq!(clean_databox_value(&b_col.values()[2].to_string()), "b3");
        assert_eq!(clean_databox_value(&b_col.values()[3].to_string()), "b4");
    }
}