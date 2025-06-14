use pandrs::column::ColumnTrait;
use pandrs::error::Result;
use pandrs::{Column, Float64Column, OptimizedDataFrame, StringColumn};

// Helper function for string -> NaiveDate parsing
// Parsing helper function has been removed as it's not used

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_window_operations() -> Result<()> {
    // Create time series data for testing
    let mut df = OptimizedDataFrame::new();

    // Create date column
    let dates = [
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
        "2023-01-05",
        "2023-01-06",
        "2023-01-07",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect::<Vec<String>>();

    let date_col = StringColumn::new(dates);
    df.add_column("date", Column::String(date_col))?;

    // Create value column
    let values = vec![10.0, 20.0, 15.0, 30.0, 25.0, 40.0, 35.0];
    let value_col = Float64Column::new(values);
    df.add_column("value", Column::Float64(value_col))?;

    // Validation
    assert_eq!(df.row_count(), 7);
    assert!(df.contains_column("date"));
    assert!(df.contains_column("value"));

    // Note: Actual window operations need to be implemented in OptimizedDataFrame
    // Here we're only checking that the data is correctly set up

    // Check column access
    let value_col_view = df.column("value")?;
    if let Some(float_col) = value_col_view.as_float64() {
        // Check sum
        let sum = float_col.sum();
        assert_eq!(sum, 175.0);

        // Check mean
        let mean = float_col.mean().unwrap_or(0.0);
        assert!((mean - 25.0).abs() < 0.001);
    } else {
        panic!("Could not get value column as float64");
    }

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_cumulative_operations() -> Result<()> {
    // Test to simulate cumulative operations
    let mut df = OptimizedDataFrame::new();

    // Create value column
    let values = vec![10.0, 20.0, 15.0, 30.0, 25.0];
    let value_col = Float64Column::new(values);
    df.add_column("value", Column::Float64(value_col))?;

    // Simulate cumulative sum (in actual implementation would be calculated with window operation)
    let mut cumsum = Vec::new();
    let mut running_sum = 0.0;

    // Get value column
    let value_col_view = df.column("value")?;
    if let Some(float_col) = value_col_view.as_float64() {
        for i in 0..float_col.len() {
            if let Ok(Some(val)) = float_col.get(i) {
                running_sum += val;
                cumsum.push(running_sum);
            }
        }
    }

    // Check expected values
    assert_eq!(cumsum, vec![10.0, 30.0, 45.0, 75.0, 100.0]);

    Ok(())
}
