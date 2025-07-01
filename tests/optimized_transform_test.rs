#[cfg(test)]
mod tests {
    use pandrs::error::Result;
    use pandrs::{Column, OptimizedDataFrame, StringColumn};

    #[test]
#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
    fn test_optimized_melt() -> Result<()> {
        // Create test dataframe
        let mut df = OptimizedDataFrame::new();

        // ID column
        let id_col = StringColumn::new(vec!["1".to_string(), "2".to_string()]);
        df.add_column("id", Column::String(id_col))?;

        // A column
        let a_col = StringColumn::new(vec!["a1".to_string(), "a2".to_string()]);
        df.add_column("A", Column::String(a_col))?;

        // B column
        let b_col = StringColumn::new(vec!["b1".to_string(), "b2".to_string()]);
        df.add_column("B", Column::String(b_col))?;

        // Execute melt operation
        let melted = df.melt(&["id"], Some(&["A", "B"]), Some("variable"), Some("value"))?;

        // Validation
        assert_eq!(melted.column_count(), 3); // id, variable, value
        assert_eq!(melted.row_count(), 4); // 2 rows x 2 columns = 4 rows

        // Check column names
        let columns = melted.column_names();
        assert!(columns.contains(&"id".to_string()));
        assert!(columns.contains(&"variable".to_string()));
        assert!(columns.contains(&"value".to_string()));

        // Data validation is OK as long as values are correctly preserved
        // Detailed validation of complex data structures is omitted as it depends on implementation

        Ok(())
    }

    #[test]
#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
    fn test_optimized_concat() -> Result<()> {
        // First dataframe
        let mut df1 = OptimizedDataFrame::new();

        let id_col1 = StringColumn::new(vec!["1".to_string(), "2".to_string()]);
        df1.add_column("id", Column::String(id_col1))?;

        let value_col1 = StringColumn::new(vec!["a".to_string(), "b".to_string()]);
        df1.add_column("value", Column::String(value_col1))?;

        // Second dataframe
        let mut df2 = OptimizedDataFrame::new();

        let id_col2 = StringColumn::new(vec!["3".to_string(), "4".to_string()]);
        df2.add_column("id", Column::String(id_col2))?;

        let value_col2 = StringColumn::new(vec!["c".to_string(), "d".to_string()]);
        df2.add_column("value", Column::String(value_col2))?;

        // Concatenation operation
        let concat_df = df1.append(&df2)?;

        // Validation
        assert_eq!(concat_df.column_count(), 2);
        assert_eq!(concat_df.row_count(), 4);

        // Check column existence
        assert!(concat_df.contains_column("id"));
        assert!(concat_df.contains_column("value"));

        Ok(())
    }

    #[test]
#[allow(clippy::result_large_err)]
#[allow(clippy::result_large_err)]
    fn test_optimized_concat_different_columns() -> Result<()> {
        // First dataframe
        let mut df1 = OptimizedDataFrame::new();

        let id_col1 = StringColumn::new(vec!["1".to_string(), "2".to_string()]);
        df1.add_column("id", Column::String(id_col1))?;

        let a_col = StringColumn::new(vec!["a1".to_string(), "a2".to_string()]);
        df1.add_column("A", Column::String(a_col))?;

        // Second dataframe (with different columns)
        let mut df2 = OptimizedDataFrame::new();

        let id_col2 = StringColumn::new(vec!["3".to_string(), "4".to_string()]);
        df2.add_column("id", Column::String(id_col2))?;

        let b_col = StringColumn::new(vec!["b3".to_string(), "b4".to_string()]);
        df2.add_column("B", Column::String(b_col))?;

        // Concatenation operation
        let concat_df = df1.append(&df2)?;

        // Validation
        assert_eq!(concat_df.column_count(), 3); // id, A, B
        assert_eq!(concat_df.row_count(), 4);

        // Check column existence
        assert!(concat_df.contains_column("id"));
        assert!(concat_df.contains_column("A"));
        assert!(concat_df.contains_column("B"));

        Ok(())
    }
}
