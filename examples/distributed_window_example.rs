//! Example of using Window Functions with Distributed Processing
//!
//! This example demonstrates how to use window functions for analytics
//! in the distributed processing framework.
//! Note: Requires the "distributed" feature flag to be enabled.

#[cfg(feature = "distributed")]
use pandrs::distributed::{window_functions, WindowFunctionExt};
#[cfg(feature = "distributed")]
use pandrs::distributed::{WindowFrame, WindowFrameBoundary};
#[cfg(feature = "distributed")]
use pandrs::error::Result;
#[cfg(feature = "distributed")]
use pandrs::{
    distributed::{DistributedConfig, ToDistributed},
    DataFrame,
};

#[cfg(feature = "distributed")]
#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS Distributed Window Functions Example");

    // Create a data frame
    let mut df = create_test_data()?;
    println!("Original DataFrame:\n{:?}\n", df);

    // Configure distributed processing
    let config = DistributedConfig::default()
        .with_executor_count(2)
        .with_partition_size(5);

    // Create a context
    let ctx = pandrs::distributed::datafusion::DataFusionContext::new(config);

    // Convert to distributed DataFrame
    let dist_df = df.to_distributed(&ctx)?;

    // Basic window function: row_number
    println!("\n-- Row Number Example --");
    let window_df = dist_df.window_function(
        window_functions::row_number(),
        "row_num",
        None, // No partition by
        None, // No order by
    )?;

    let result_df = window_df.collect()?;
    println!("Row Number Result:\n{:?}\n", result_df);

    // Window function with partition by
    println!("\n-- Rank with Partition By Example --");
    let window_df = dist_df.window_function(
        window_functions::rank(),
        "rank",
        Some(vec!["category"]),      // Partition by category
        Some(vec![("value", true)]), // Order by value (ascending)
    )?;

    let result_df = window_df.collect()?;
    println!("Rank Result:\n{:?}\n", result_df);

    // Window function with custom frame
    println!("\n-- Custom Window Frame Example --");

    // Define a window frame: 2 rows preceding to current row
    let window_frame = WindowFrame {
        start_bound: WindowFrameBoundary::Preceding(2),
        end_bound: WindowFrameBoundary::CurrentRow,
    };

    let window_df = dist_df.window_function_with_frame(
        window_functions::sum("value"), // Sum of value column
        "sum_3_rows",
        Some(vec!["category"]),   // Partition by category
        Some(vec![("id", true)]), // Order by id (ascending)
        window_frame,
    )?;

    let result_df = window_df.collect()?;
    println!("Window Frame Result:\n{:?}\n", result_df);

    // Multiple window functions
    println!("\n-- Multiple Window Functions Example --");

    // First window function: rank within category
    let window_df1 = dist_df.window_function(
        window_functions::rank(),
        "rank_in_category",
        Some(vec!["category"]),       // Partition by category
        Some(vec![("value", false)]), // Order by value (descending)
    )?;

    // Second window function: average value within category
    let window_df2 = window_df1.window_function(
        window_functions::avg("value"),
        "avg_value_in_category",
        Some(vec!["category"]), // Partition by category
        None,                   // No specific order
    )?;

    // Third window function: row number over entire dataset
    let window_df3 = window_df2.window_function(
        window_functions::row_number(),
        "overall_row_num",
        None,                         // No partition by
        Some(vec![("value", false)]), // Order by value (descending)
    )?;

    let result_df = window_df3.collect()?;
    println!("Multiple Window Functions Result:\n{:?}\n", result_df);

    // Running totals example
    println!("\n-- Running Totals Example --");

    // Define a window frame: unbounded preceding to current row
    let running_frame = WindowFrame {
        start_bound: WindowFrameBoundary::UnboundedPreceding,
        end_bound: WindowFrameBoundary::CurrentRow,
    };

    let window_df = dist_df.window_function_with_frame(
        window_functions::sum("value"), // Sum of value column
        "running_total",
        None,                     // No partition by
        Some(vec![("id", true)]), // Order by id (ascending)
        running_frame,
    )?;

    let result_df = window_df.collect()?;
    println!("Running Totals Result:\n{:?}\n", result_df);

    Ok(())
}

#[cfg(feature = "distributed")]
/// Create a test DataFrame for the example
#[allow(clippy::result_large_err)]
fn create_test_data() -> Result<DataFrame> {
    use pandrs::column::{Column, Float64Column, Int64Column, StringColumn};
    use pandrs::optimized::OptimizedDataFrame;

    // Create a test DataFrame
    let mut df = OptimizedDataFrame::new();

    // Add ID column
    let ids = Int64Column::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    df.add_column("id".to_string(), Column::Int64(ids))?;

    // Add Value column
    let values = Float64Column::new(vec![
        55.0, 30.0, 40.0, 85.0, 60.0, 75.0, 45.0, 90.0, 25.0, 50.0,
    ]);
    df.add_column("value".to_string(), Column::Float64(values))?;

    // Add Category column
    let categories = StringColumn::new(vec![
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string(),
        "C".to_string(),
        "C".to_string(),
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "A".to_string(),
    ]);
    df.add_column("category".to_string(), Column::String(categories))?;

    Ok(df)
}

#[cfg(not(feature = "distributed"))]
fn main() {
    println!("This example requires the 'distributed' feature flag to be enabled.");
    println!("Please recompile with 'cargo run --example distributed_window_example --features distributed'");
}
