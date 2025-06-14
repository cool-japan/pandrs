use pandrs::error::Result;
use pandrs::{Column, Float64Column, Int64Column, OptimizedDataFrame};

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_series_representation() -> Result<()> {
    // OptimizedDataFrame is column-oriented, and Series are implemented as columns
    let mut df = OptimizedDataFrame::new();

    // Create integer column
    let values = vec![1, 2, 3, 4, 5];
    let int_col = Int64Column::new(values);
    df.add_column("test", Column::Int64(int_col))?;

    // Validation
    assert_eq!(df.row_count(), 5);
    assert!(df.contains_column("test"));

    // Check column access
    let col = df.column("test")?;
    if let Some(int64_col) = col.as_int64() {
        assert_eq!(int64_col.get(0)?, Some(1));
        assert_eq!(int64_col.get(4)?, Some(5));
        assert!(int64_col.get(5).is_err()); // Out of range access
    } else {
        panic!("Could not get as integer column");
    }

    Ok(())
}

#[test]
#[allow(clippy::result_large_err)]
fn test_optimized_series_numeric_operations() -> Result<()> {
    // Check numeric operations in OptimizedDataFrame
    let mut df = OptimizedDataFrame::new();

    // Create integer column
    let values = vec![1, 2, 3, 4, 5];
    let int_col = Int64Column::new(values);
    df.add_column("int_values", Column::Int64(int_col))?;

    // Create floating point column
    let float_values = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let float_col = Float64Column::new(float_values);
    df.add_column("float_values", Column::Float64(float_col))?;

    // Check aggregation operations on integer column
    let int_series = df.column("int_values")?;
    if let Some(int64_col) = int_series.as_int64() {
        // Sum
        assert_eq!(int64_col.sum(), 15);

        // Mean
        assert_eq!(int64_col.mean().unwrap_or(0.0), 3.0);

        // Minimum
        assert_eq!(int64_col.min().unwrap_or(0), 1);

        // Maximum
        assert_eq!(int64_col.max().unwrap_or(0), 5);
    } else {
        panic!("Could not get as integer column");
    }

    // Check aggregation operations on floating point column
    let float_series = df.column("float_values")?;
    if let Some(float64_col) = float_series.as_float64() {
        // Sum
        assert!((float64_col.sum() - 17.5).abs() < 0.001);

        // Mean
        assert!((float64_col.mean().unwrap_or(0.0) - 3.5).abs() < 0.001);

        // Minimum
        assert!((float64_col.min().unwrap_or(0.0) - 1.5).abs() < 0.001);

        // Maximum
        assert!((float64_col.max().unwrap_or(0.0) - 5.5).abs() < 0.001);
    } else {
        panic!("Could not get as floating point column");
    }

    Ok(())
}
