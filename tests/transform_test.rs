#[cfg(test)]
mod tests {
    use pandrs::dataframe::TransformExt;
    use pandrs::{DataFrame, Series};

    // Helper function to clean DataBox string values
    fn clean_databox_value(value: &str) -> String {
        let trimmed = value
            .trim_start_matches("DataBox(\"")
            .trim_end_matches("\")");
        let value_str = if trimmed.starts_with("DataBox(") {
            trimmed.trim_start_matches("DataBox(").trim_end_matches(")")
        } else {
            trimmed
        };
        value_str.trim_matches('"').to_string()
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
                    let sum: i32 = values.iter().filter_map(|v| v.parse::<i32>().ok()).sum();
                    sum.to_string()
                },
            )
            .unwrap();

        // Verification
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 3); // Food, Electronics, Clothing

        // Check aggregate results
        let cat_col = result.get_column::<String>("category").unwrap();
        let agg_col = result.get_column::<String>("sales_agg").unwrap();

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
        let id_col = concat_df.get_column::<String>("id").unwrap();
        let value_col = concat_df.get_column::<String>("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "1");
        assert_eq!(clean_databox_value(&value_col.values()[0].to_string()), "a");
        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "2");
        assert_eq!(clean_databox_value(&value_col.values()[1].to_string()), "b");
        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "3");
        assert_eq!(clean_databox_value(&value_col.values()[2].to_string()), "c");
        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "4");
        assert_eq!(clean_databox_value(&value_col.values()[3].to_string()), "d");
    }
}
