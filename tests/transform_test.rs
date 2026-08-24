#![allow(clippy::result_large_err)]
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
        // Create test DataFrame. These categories/values are deliberately
        // different from pandrs' old hardcoded fabricated output
        // (Food/Electronics/Clothing with totals 1000/1500/1200): a
        // regression back to that stub would make the assertions below
        // fail instead of accidentally passing.
        let mut df = DataFrame::new();
        df.add_column(
            "category".to_string(),
            Series::new(
                vec!["Books", "Toys", "Books", "Garden", "Toys"],
                Some("category".to_string()),
            )
            .unwrap(),
        )
        .unwrap();
        df.add_column(
            "sales".to_string(),
            Series::new(
                vec!["300", "1200", "900", "1600", "400"],
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

        // "Books" (300, 900) never clears the >=1000 filter, so it must not
        // appear at all: only "Toys" (1200) and "Garden" (1600) survive.
        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 2);

        // Check aggregate results
        let cat_col = result.get_column::<String>("category").unwrap();
        let agg_col = result.get_column::<String>("sales_agg").unwrap();

        // Check each category's aggregate value
        // Note: Order may depend on implementation, so check each category individually
        for i in 0..result.row_count() {
            let category = clean_databox_value(&cat_col.values()[i].to_string());
            let agg_value = clean_databox_value(&agg_col.values()[i].to_string());

            if category == "Toys" {
                assert_eq!(agg_value, "1200"); // Only the 1200 Toys row is >= 1000
            } else if category == "Garden" {
                assert_eq!(agg_value, "1600");
            } else {
                panic!("Unexpected category (should have been filtered out): {category}");
            }
        }
    }

    #[test]
    fn test_concat() {
        // Ids/values deliberately different from pandrs' old hardcoded
        // fabricated concat output ("1".."4" / "a".."d"): a regression back
        // to that stub would fail these assertions instead of accidentally
        // passing.
        let mut df1 = DataFrame::new();
        df1.add_column(
            "id".to_string(),
            Series::new(vec!["10", "20"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df1.add_column(
            "value".to_string(),
            Series::new(vec!["p", "q"], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        // Second dataframe
        let mut df2 = DataFrame::new();
        df2.add_column(
            "id".to_string(),
            Series::new(vec!["30", "40", "50"], Some("id".to_string())).unwrap(),
        )
        .unwrap();
        df2.add_column(
            "value".to_string(),
            Series::new(vec!["r", "s", "t"], Some("value".to_string())).unwrap(),
        )
        .unwrap();

        // Concatenation operation
        let concat_df = DataFrame::concat(&[&df1, &df2], true).unwrap();

        // Validation: 2 + 3 = 5 rows (the old stub always emitted exactly 4
        // regardless of input).
        assert_eq!(concat_df.column_count(), 2);
        assert_eq!(concat_df.row_count(), 5);

        // Check columns
        let id_col = concat_df.get_column::<String>("id").unwrap();
        let value_col = concat_df.get_column::<String>("value").unwrap();

        assert_eq!(clean_databox_value(&id_col.values()[0].to_string()), "10");
        assert_eq!(clean_databox_value(&value_col.values()[0].to_string()), "p");
        assert_eq!(clean_databox_value(&id_col.values()[1].to_string()), "20");
        assert_eq!(clean_databox_value(&value_col.values()[1].to_string()), "q");
        assert_eq!(clean_databox_value(&id_col.values()[2].to_string()), "30");
        assert_eq!(clean_databox_value(&value_col.values()[2].to_string()), "r");
        assert_eq!(clean_databox_value(&id_col.values()[3].to_string()), "40");
        assert_eq!(clean_databox_value(&value_col.values()[3].to_string()), "s");
        assert_eq!(clean_databox_value(&id_col.values()[4].to_string()), "50");
        assert_eq!(clean_databox_value(&value_col.values()[4].to_string()), "t");
    }
}
