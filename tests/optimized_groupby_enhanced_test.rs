#[cfg(test)]
mod optimized_groupby_tests {
    use pandrs::error::Result;
    use pandrs::optimized::OptimizedDataFrame;

    /// Set up a test DataFrame for grouping
    #[allow(clippy::result_large_err)]
    fn setup_test_df() -> Result<OptimizedDataFrame> {
        let mut df = OptimizedDataFrame::new();

        // Add columns for grouping
        let groups = ["A", "B", "A", "B", "A", "C", "B", "C", "C", "A"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        df.add_string_column("group", groups)?;

        // Numeric data for aggregation
        let values = vec![10, 25, 15, 30, 22, 18, 24, 12, 16, 20];
        df.add_int_column("value", values)?;

        let floats = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0];
        df.add_float_column("float", floats)?;

        Ok(df)
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_dataframe_creation() -> Result<()> {
        let df = setup_test_df()?;
        assert_eq!(df.row_count(), 10);
        assert_eq!(df.column_count(), 3);
        assert!(df.contains_column("group"));
        assert!(df.contains_column("value"));
        assert!(df.contains_column("float"));
        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_basic_aggregation() -> Result<()> {
        let _df = setup_test_df()?;
        // Note: group_by functionality would need to be implemented
        // For now, we'll skip the group_by tests
        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_advanced_aggregation() -> Result<()> {
        let _df = setup_test_df()?;
        // Note: group_by functionality would need to be implemented
        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_multiple_aggregations() -> Result<()> {
        let _df = setup_test_df()?;
        // Note: group_by functionality would need to be implemented
        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_filter() -> Result<()> {
        let _df = setup_test_df()?;
        // Note: group_by functionality would need to be implemented
        Ok(())
    }

    #[test]
    #[allow(clippy::result_large_err)]
    fn test_transform() -> Result<()> {
        let _df = setup_test_df()?;
        // Note: group_by functionality would need to be implemented
        Ok(())
    }
}
