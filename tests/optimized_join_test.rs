use pandrs::error::Result;
use pandrs::{Column, Int64Column, OptimizedDataFrame, StringColumn};

#[test]
#[allow(clippy::result_large_err)]
fn test_inner_join() -> Result<()> {
    // Left dataframe
    let mut left_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;

    // Name column
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;

    // Right dataframe
    let mut right_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;

    // Value column
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;

    // Inner join
    let joined = left_df.inner_join(&right_df, "id", "id")?;

    // Validation - inner join only includes matching rows (id=1 and 2)
    assert_eq!(joined.row_count(), 2);
    assert_eq!(joined.column_count(), 3); // id, name, value

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_left_join() -> Result<()> {
    // Left dataframe
    let mut left_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;

    // Name column
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;

    // Right dataframe
    let mut right_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;

    // Value column
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;

    // Left join
    let joined = left_df.left_join(&right_df, "id", "id")?;

    // Validation - left join includes all rows from left (id=1,2,3,4) and matching right rows
    assert_eq!(joined.row_count(), 4);
    assert_eq!(joined.column_count(), 3); // id, name, value

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_right_join() -> Result<()> {
    // Left dataframe
    let mut left_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;

    // Name column
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;

    // Right dataframe
    let mut right_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;

    // Value column
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;

    // Right join
    let joined = left_df.right_join(&right_df, "id", "id")?;

    // Validation - right join includes all rows from right (id=1,2,5,6) and matching left rows
    assert_eq!(joined.row_count(), 4);
    assert_eq!(joined.column_count(), 3); // id, name, value

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_outer_join() -> Result<()> {
    // Left dataframe
    let mut left_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;

    // Name column
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;

    // Right dataframe
    let mut right_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("id", Column::Int64(id_col))?;

    // Value column
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;

    // Outer join
    let joined = left_df.outer_join(&right_df, "id", "id")?;

    // Validation - outer join includes all rows (id=1,2,3,4,5,6)
    assert_eq!(joined.row_count(), 6);
    assert_eq!(joined.column_count(), 3); // id, name, value

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_join_different_column_names() -> Result<()> {
    // Left dataframe
    let mut left_df = OptimizedDataFrame::new();

    // Left ID column
    let left_id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("left_id", Column::Int64(left_id_col))?;

    // Name column
    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
        "Dave".to_string(),
    ]);
    left_df.add_column("name", Column::String(name_col))?;

    // Right dataframe
    let mut right_df = OptimizedDataFrame::new();

    // Right ID column
    let right_id_col = Int64Column::new(vec![1, 2, 5, 6]);
    right_df.add_column("right_id", Column::Int64(right_id_col))?;

    // Value column
    let value_col = Int64Column::new(vec![100, 200, 500, 600]);
    right_df.add_column("value", Column::Int64(value_col))?;

    // Inner join on different column names
    let joined = left_df.inner_join(&right_df, "left_id", "right_id")?;

    // Validation - inner join only includes matching rows (id=1 and 2)
    assert_eq!(joined.row_count(), 2);
    assert_eq!(joined.column_count(), 3); // left_id, name, value

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_empty_join() -> Result<()> {
    // Join with empty DataFrames
    let empty_df = OptimizedDataFrame::new();
    let result = empty_df.inner_join(&empty_df, "id", "id");

    // Should error because trying to join on non-existent columns
    assert!(result.is_err());

    // Left dataframe
    let mut left_df = OptimizedDataFrame::new();

    // ID column
    let id_col = Int64Column::new(vec![1, 2, 3, 4]);
    left_df.add_column("id", Column::Int64(id_col))?;

    // Right dataframe
    let mut right_df = OptimizedDataFrame::new();

    // Non-matching ID column
    let id_col = Int64Column::new(vec![5, 6, 7, 8]);
    right_df.add_column("id", Column::Int64(id_col))?;

    // Inner join (no matching rows)
    let joined = left_df.inner_join(&right_df, "id", "id")?;

    // Validation - empty dataframe as no matching rows
    assert_eq!(joined.row_count(), 0);
    // Implementation might vary: could be an empty DataFrame or a DataFrame with just columns
    // So we only verify row count is 0

    Ok(())
}
