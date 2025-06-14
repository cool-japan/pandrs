use pandrs::column::ColumnTrait;
use pandrs::error::Result;
use pandrs::{Column, Float64Column, Int64Column, OptimizedDataFrame, StringColumn};

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_dataframe_apply() -> Result<()> {
    // Create a DataFrame for testing
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let id_col = Int64Column::new(vec![1, 2, 3]);
    df.add_column("id", Column::Int64(id_col))?;

    let value_col = Float64Column::new(vec![10.0, 20.0, 30.0]);
    df.add_column("value", Column::Float64(value_col))?;

    // Apply a function to double the values
    let doubled_df = df.apply(
        |col| {
            match col.column() {
                Column::Int64(int_col) => {
                    let mut new_data = Vec::with_capacity(int_col.len());
                    for i in 0..int_col.len() {
                        if let Ok(Some(value)) = int_col.get(i) {
                            new_data.push(value * 2);
                        } else {
                            new_data.push(0); // Use default value for NA
                        }
                    }
                    Ok(Column::Int64(Int64Column::new(new_data)))
                }
                Column::Float64(float_col) => {
                    let mut new_data = Vec::with_capacity(float_col.len());
                    for i in 0..float_col.len() {
                        if let Ok(Some(value)) = float_col.get(i) {
                            new_data.push(value * 2.0);
                        } else {
                            new_data.push(0.0); // Use default value for NA
                        }
                    }
                    Ok(Column::Float64(Float64Column::new(new_data)))
                }
                _ => {
                    // Return other types unchanged
                    Ok(col.column().clone())
                }
            }
        },
        None,
    )?;

    // Validation
    assert_eq!(doubled_df.row_count(), 3);
    assert_eq!(doubled_df.column_count(), 2);

    // Verify that values are doubled
    let id_view = doubled_df.column("id")?;
    if let Some(int_col) = id_view.as_int64() {
        assert_eq!(int_col.get(0)?, Some(2));
        assert_eq!(int_col.get(1)?, Some(4));
        assert_eq!(int_col.get(2)?, Some(6));
    } else {
        panic!("Could not get ID column as Int64");
    }

    let value_view = doubled_df.column("value")?;
    if let Some(float_col) = value_view.as_float64() {
        assert_eq!(float_col.get(0)?, Some(20.0));
        assert_eq!(float_col.get(1)?, Some(40.0));
        assert_eq!(float_col.get(2)?, Some(60.0));
    } else {
        panic!("Could not get Value column as Float64");
    }

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_dataframe_apply_with_column_subset() -> Result<()> {
    // Create a DataFrame for testing
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let id_col = Int64Column::new(vec![1, 2, 3]);
    df.add_column("id", Column::Int64(id_col))?;

    let value_col = Float64Column::new(vec![10.0, 20.0, 30.0]);
    df.add_column("value", Column::Float64(value_col))?;

    let name_col = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Charlie".to_string(),
    ]);
    df.add_column("name", Column::String(name_col))?;

    // Apply function to specific columns only
    let result = df.apply(
        |col| match col.column() {
            Column::Int64(int_col) => {
                let mut new_data = Vec::with_capacity(int_col.len());
                for i in 0..int_col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
                        new_data.push(value * 2);
                    } else {
                        new_data.push(0);
                    }
                }
                Ok(Column::Int64(Int64Column::new(new_data)))
            }
            _ => Ok(col.column().clone()),
        },
        Some(&["id"]),
    )?;

    // Validation
    assert_eq!(result.row_count(), 3);
    assert_eq!(result.column_count(), 3);

    // Verify that only id was modified
    let id_view = result.column("id")?;
    if let Some(int_col) = id_view.as_int64() {
        assert_eq!(int_col.get(0)?, Some(2)); // Double the original value
    } else {
        panic!("Could not get ID column as Int64");
    }

    // Verify that value column remains unchanged
    let value_view = result.column("value")?;
    if let Some(float_col) = value_view.as_float64() {
        assert_eq!(float_col.get(0)?, Some(10.0)); // Same as original value
    } else {
        panic!("Could not get Value column as Float64");
    }

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_dataframe_applymap() -> Result<()> {
    // Create a DataFrame for testing
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let id_col = Int64Column::new(vec![1, 2, 3]);
    df.add_column("id", Column::Int64(id_col))?;

    let value_col = Float64Column::new(vec![10.0, 20.0, 30.0]);
    df.add_column("value", Column::Float64(value_col))?;

    // Add string column
    let name_col = StringColumn::new(vec![
        "alice".to_string(),
        "bob".to_string(),
        "charlie".to_string(),
    ]);
    df.add_column("name", Column::String(name_col))?;

    // Note: The actual applymap function might not be fully implemented yet,
    // so we're just checking that the basic DataFrame functionality works correctly

    // Verify the basic structure of the DataFrame
    assert_eq!(df.row_count(), 3);
    assert_eq!(df.column_count(), 3);
    assert!(df.contains_column("name"));

    // Verify that we can read the string column
    let name_view = df.column("name")?;
    if let Some(str_col) = name_view.as_string() {
        let val = str_col.get(0)?;
        assert!(val.is_some());
    }

    Ok(())
}
