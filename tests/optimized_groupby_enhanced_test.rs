#[cfg(test)]
mod optimized_groupby_tests {
    use pandrs::error::Result;
    use pandrs::optimized::{AggregateOp, OptimizedDataFrame};

    /// Set up a test DataFrame for grouping
    fn setup_test_df() -> Result<OptimizedDataFrame> {
        let mut df = OptimizedDataFrame::new();
        
        // Add columns for grouping
        let groups = vec!["A", "B", "A", "B", "A", "C", "B", "C", "C", "A"];
        df.add_string_column("group", groups)?;
        
        // Numeric data for aggregation
        let values = vec![10, 25, 15, 30, 22, 18, 24, 12, 16, 20];
        df.add_int_column("value", values)?;
        
        let floats = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0];
        df.add_float_column("float", floats)?;
        
        Ok(df)
    }

    #[test]
    fn test_basic_aggregation() -> Result<()> {
        let df = setup_test_df()?;
        let grouped = df.group_by(["group"])?;
        
        // Test sum
        let sum_result = grouped.sum("value")?;
        
        // Sum expectations: A=[10, 15, 22, 20]=67, B=[25, 30, 24]=79, C=[18, 12, 16]=46
        let a_sum = find_group_value(&sum_result, "A", "value_sum")?;
        let b_sum = find_group_value(&sum_result, "B", "value_sum")?;
        let c_sum = find_group_value(&sum_result, "C", "value_sum")?;
        
        assert!((a_sum - 67.0).abs() < 0.001);
        assert!((b_sum - 79.0).abs() < 0.001);
        assert!((c_sum - 46.0).abs() < 0.001);
        
        // Test mean
        let mean_result = grouped.mean("value")?;
        
        // Mean expectations: A=[10, 15, 22, 20]/4=16.75, B=[25, 30, 24]/3=26.33, C=[18, 12, 16]/3=15.33
        let a_mean = find_group_value(&mean_result, "A", "value_mean")?;
        let b_mean = find_group_value(&mean_result, "B", "value_mean")?;
        let c_mean = find_group_value(&mean_result, "C", "value_mean")?;
        
        assert!((a_mean - 16.75).abs() < 0.001);
        assert!((b_mean - 26.33).abs() < 0.01);
        assert!((c_mean - 15.33).abs() < 0.01);
        
        Ok(())
    }
    
    #[test]
    fn test_advanced_aggregation() -> Result<()> {
        let df = setup_test_df()?;
        let grouped = df.group_by(["group"])?;
        
        // Test standard deviation
        let std_result = grouped.std("value")?;
        
        // Verify that standard deviation is calculated
        let a_std = find_group_value(&std_result, "A", "value_std")?;
        let b_std = find_group_value(&std_result, "B", "value_std")?;
        let c_std = find_group_value(&std_result, "C", "value_std")?;
        
        // Manual verification for group A
        let a_values = [10.0, 15.0, 22.0, 20.0];
        let mean = a_values.iter().sum::<f64>() / a_values.len() as f64;
        let sum_sq_diff = a_values.iter().map(|v| (*v - mean).powi(2)).sum::<f64>();
        let manual_std = (sum_sq_diff / (a_values.len() - 1) as f64).sqrt();
        
        assert!((a_std - manual_std).abs() < 0.001);
        
        // Test median
        let median_result = grouped.median("value")?;
        
        // Median expectations: A=[10, 15, 20, 22] -> (15+20)/2=17.5, B=[24, 25, 30] -> 25, C=[12, 16, 18] -> 16
        let a_median = find_group_value(&median_result, "A", "value_median")?;
        let b_median = find_group_value(&median_result, "B", "value_median")?;
        let c_median = find_group_value(&median_result, "C", "value_median")?;
        
        assert!((a_median - 17.5).abs() < 0.001);
        assert!((b_median - 25.0).abs() < 0.001);
        assert!((c_median - 16.0).abs() < 0.001);
        
        // Test first/last
        let first_result = grouped.first("value")?;
        let a_first = find_group_value(&first_result, "A", "value_first")?;
        assert!((a_first - 10.0).abs() < 0.001);
        
        let last_result = grouped.last("value")?;
        let a_last = find_group_value(&last_result, "A", "value_last")?;
        assert!((a_last - 20.0).abs() < 0.001);
        
        Ok(())
    }
    
    #[test]
    fn test_multiple_aggregations() -> Result<()> {
        let df = setup_test_df()?;
        let grouped = df.group_by(["group"])?;
        
        // Test multiple aggregations
        let result = grouped.agg(&[
            ("value", AggregateOp::Sum),
            ("value", AggregateOp::Mean),
            ("float", AggregateOp::Max)
        ])?;
        
        // Check that we have all the expected columns
        assert!(result.contains_column("value_sum"));
        assert!(result.contains_column("value_mean"));
        assert!(result.contains_column("float_max"));
        
        // Check values for group A
        let a_sum = find_group_value(&result, "A", "value_sum")?;
        let a_mean = find_group_value(&result, "A", "value_mean")?;
        let a_float_max = find_group_value(&result, "A", "float_max")?;
        
        assert!((a_sum - 67.0).abs() < 0.001);
        assert!((a_mean - 16.75).abs() < 0.001);
        assert!((a_float_max - 10.0).abs() < 0.001);
        
        Ok(())
    }
    
    #[test]
    fn test_filter() -> Result<()> {
        let df = setup_test_df()?;
        let grouped = df.group_by(["group"])?;
        
        // Filter for groups with mean value > 20
        let filtered = grouped.filter(|group_df| {
            let values = group_df.get_int_column("value").unwrap();
            let sum: i64 = values.iter().filter_map(|v| v).sum();
            let count = values.iter().filter_map(|v| v).count();
            
            if count == 0 {
                false
            } else {
                (sum as f64 / count as f64) > 20.0
            }
        })?;
        
        // Only group B has a mean value > 20
        // Check that we have only rows from group B
        let remaining_groups = filtered.get_string_column("group")?
            .iter()
            .filter_map(|g| g)
            .collect::<Vec<_>>();
            
        // Only group B should remain (which has 3 rows)
        assert_eq!(remaining_groups.len(), 3);
        assert!(remaining_groups.iter().all(|&g| g == "B"));
        
        Ok(())
    }
    
    #[test]
    fn test_transform() -> Result<()> {
        let df = setup_test_df()?;
        let grouped = df.group_by(["group"])?;
        
        // Transform - calculate percentages of value relative to group total
        let transformed = grouped.transform(|group_df| {
            let mut result = group_df.clone();
            
            let values = group_df.get_int_column("value").unwrap();
            let total: i64 = values.iter().filter_map(|v| v).sum();
            
            if total > 0 {
                // Calculate percentages
                let percentages: Vec<f64> = values
                    .iter()
                    .filter_map(|v| v.map(|val| (val as f64 / total as f64) * 100.0))
                    .collect();
                
                // Add a new column with percentages
                result.add_float_column("pct", percentages)?;
            }
            
            Ok(result)
        })?;
        
        // Check that the transform created the new column
        assert!(transformed.contains_column("pct"));
        
        // For group A, values are 10, 15, 22, 20 with total 67
        // Percentages should be approx 14.93%, 22.39%, 32.84%, 29.85%
        let values = transformed.get_int_column("value")?;
        let groups = transformed.get_string_column("group")?;
        let percentages = transformed.get_float_column("pct")?;
        
        for i in 0..transformed.row_count() {
            if let (Some(grp), Some(val), Some(pct)) = (groups.get(i).ok().flatten(), 
                                                     values.get(i).ok().flatten(), 
                                                     percentages.get(i).ok().flatten()) {
                // Calculate expected percentage
                let expected = match grp {
                    "A" => (val as f64 / 67.0) * 100.0,
                    "B" => (val as f64 / 79.0) * 100.0,
                    "C" => (val as f64 / 46.0) * 100.0,
                    _ => panic!("Unexpected group"),
                };
                
                assert!((pct - expected).abs() < 0.001);
            }
        }
        
        Ok(())
    }
    
    // Helper function to find values for specific groups in the result DataFrame
    fn find_group_value(df: &OptimizedDataFrame, group_value: &str, column: &str) -> Result<f64> {
        let groups = df.get_string_column("group")?;
        let values = df.get_float_column(column)?;
        
        for i in 0..df.row_count() {
            if let (Some(grp), Some(val)) = (groups.get(i).ok().flatten(), values.get(i).ok().flatten()) {
                if grp == group_value {
                    return Ok(val);
                }
            }
        }
        
        Err(pandrs::error::Error::DataNotFound(format!("Group {} not found", group_value)))
    }
}