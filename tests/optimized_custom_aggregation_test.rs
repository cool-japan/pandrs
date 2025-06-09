#[cfg(test)]
mod optimized_custom_aggregation_tests {
    use pandrs::error::Result;
    use pandrs::optimized::{AggregateOp, CustomAggregation, OptimizedDataFrame};
    use std::sync::Arc;

    /// Set up a test DataFrame for grouping
    fn setup_test_df() -> Result<OptimizedDataFrame> {
        let mut df = OptimizedDataFrame::new();

        // Add columns for grouping
        let groups = vec!["A", "B", "A", "B", "A", "C", "B", "C", "C", "A"];
        df.add_string_column("group", groups)?;

        // Numeric data for aggregation
        let values = vec![10, 25, 15, 30, 22, 18, 24, 12, 16, 20];
        df.add_int_column("value", values)?;

        Ok(df)
    }

    #[test]
    fn test_custom_aggregation_method() -> Result<()> {
        let df = setup_test_df()?;
        let grouped = df.group_by(["group"])?;

        // Define a custom aggregation function - calculate harmonic mean
        let result = grouped.custom("value", "value_harmonic_mean", |values| {
            if values.is_empty() {
                return 0.0;
            }

            let sum_of_reciprocals: f64 =
                values.iter().filter(|&&x| x != 0.0).map(|&x| 1.0 / x).sum();
            if sum_of_reciprocals == 0.0 {
                0.0
            } else {
                values.len() as f64 / sum_of_reciprocals
            }
        })?;

        // Check the result
        assert!(result.contains_column("value_harmonic_mean"));

        // Calculate expected harmonic means for each group
        // Group A: [10, 15, 22, 20]
        // Group B: [25, 30, 24]
        // Group C: [18, 12, 16]

        // Harmonic mean = n / (1/x1 + 1/x2 + ... + 1/xn)
        let expected_a = 4.0 / (1.0 / 10.0 + 1.0 / 15.0 + 1.0 / 22.0 + 1.0 / 20.0);
        let expected_b = 3.0 / (1.0 / 25.0 + 1.0 / 30.0 + 1.0 / 24.0);
        let expected_c = 3.0 / (1.0 / 18.0 + 1.0 / 12.0 + 1.0 / 16.0);

        // Get the actual values from the result
        let a_mean = find_group_value(&result, "A", "value_harmonic_mean")?;
        let b_mean = find_group_value(&result, "B", "value_harmonic_mean")?;
        let c_mean = find_group_value(&result, "C", "value_harmonic_mean")?;

        // Check with a small tolerance
        assert!((a_mean - expected_a).abs() < 0.001);
        assert!((b_mean - expected_b).abs() < 0.001);
        assert!((c_mean - expected_c).abs() < 0.001);

        Ok(())
    }

    #[test]
    fn test_aggregate_custom_method() -> Result<()> {
        let df = setup_test_df()?;
        let grouped = df.group_by(["group"])?;

        // Define multiple custom aggregation functions
        let median_absolute_deviation = Arc::new(|values: &[f64]| -> f64 {
            if values.is_empty() {
                return 0.0;
            }

            // Get the median
            let mut sorted_values = values.to_vec();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median = if values.len() % 2 == 0 {
                let mid = values.len() / 2;
                (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
            } else {
                sorted_values[values.len() / 2]
            };

            // Calculate absolute deviations from the median
            let mut deviations: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();

            // Get the median of the deviations
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if deviations.len() % 2 == 0 {
                let mid = deviations.len() / 2;
                (deviations[mid - 1] + deviations[mid]) / 2.0
            } else {
                deviations[deviations.len() / 2]
            }
        });

        let coefficient_of_variation = Arc::new(|values: &[f64]| -> f64 {
            if values.is_empty() || values.iter().sum::<f64>() == 0.0 {
                return 0.0;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;

            let variance = values
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (values.len() as f64);

            let std_dev = variance.sqrt();

            // CV = std_dev / mean
            std_dev / mean
        });

        // Create aggregation specifications
        let aggregations = vec![
            CustomAggregation {
                column: "value".to_string(),
                op: AggregateOp::Custom,
                result_name: "value_mad".to_string(),
                custom_fn: Some(median_absolute_deviation),
            },
            CustomAggregation {
                column: "value".to_string(),
                op: AggregateOp::Custom,
                result_name: "value_cv".to_string(),
                custom_fn: Some(coefficient_of_variation),
            },
            // Add a standard aggregation
            CustomAggregation {
                column: "value".to_string(),
                op: AggregateOp::Mean,
                result_name: "value_mean".to_string(),
                custom_fn: None,
            },
        ];

        let result = grouped.aggregate_custom(aggregations)?;

        // Check the result
        assert!(result.contains_column("value_mad"));
        assert!(result.contains_column("value_cv"));
        assert!(result.contains_column("value_mean"));

        // Get the actual values from the result
        let a_mean = find_group_value(&result, "A", "value_mean")?;
        let b_mean = find_group_value(&result, "B", "value_mean")?;
        let c_mean = find_group_value(&result, "C", "value_mean")?;

        // Check mean values for each group to verify standard aggregations still work
        assert!((a_mean - 16.75).abs() < 0.001); // (10 + 15 + 22 + 20) / 4 = 16.75
        assert!((b_mean - 26.33).abs() < 0.01); // (25 + 30 + 24) / 3 = 26.33
        assert!((c_mean - 15.33).abs() < 0.01); // (18 + 12 + 16) / 3 = 15.33

        Ok(())
    }

    // Helper function to find values for specific groups in the result DataFrame
    fn find_group_value(df: &OptimizedDataFrame, group_value: &str, column: &str) -> Result<f64> {
        let groups = df.get_string_column("group")?;
        let values = df.get_float_column(column)?;

        for i in 0..df.row_count() {
            if let (Some(grp), Some(val)) =
                (groups.get(i).ok().flatten(), values.get(i).ok().flatten())
            {
                if grp == group_value {
                    return Ok(val);
                }
            }
        }

        Err(pandrs::error::Error::DataNotFound(format!(
            "Group {} not found",
            group_value
        )))
    }
}
