use pandrs::error::Result;
use pandrs::optimized::{AggregateOp, CustomAggregation, OptimizedDataFrame};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("PandRS Custom Aggregation Example");
    println!("=================================");

    // Create a sample DataFrame with various data
    let mut df = OptimizedDataFrame::new();

    // Add columns
    let categories = vec!["A", "B", "A", "B", "A", "C", "B", "C", "C", "A"];
    df.add_string_column("category", categories)?;

    let values = vec![10, 25, 15, 30, 22, 18, 24, 12, 16, 20];
    df.add_int_column("value", values)?;

    // Display the DataFrame
    println!("Original DataFrame:");
    println!("{:?}", df);

    // Group by category
    let grouped = df.group_by(["category"])?;

    // 1. Simple custom aggregation using the custom method
    println!("\n1. Simple Custom Aggregation");
    println!("----------------------------");

    // Calculate coefficient of variation (CV = standard deviation / mean)
    let cv_result = grouped.custom("value", "value_cv", |values| {
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
            / (values.len() - 1) as f64;

        let std_dev = variance.sqrt();

        // CV = std_dev / mean
        std_dev / mean
    })?;

    println!("Coefficient of Variation by category:");
    println!("{:?}", cv_result);

    // 2. Multiple custom aggregations using aggregate_custom
    println!("\n2. Multiple Custom Aggregations");
    println!("------------------------------");

    // Create custom functions
    let geometric_mean = Arc::new(|values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let product = values.iter().fold(1.0, |acc, &x| acc * x);
        product.powf(1.0 / values.len() as f64)
    });

    let range = Arc::new(|values: &[f64]| -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        max - min
    });

    // Create custom aggregation specifications
    let aggregations = vec![
        CustomAggregation {
            column: "value".to_string(),
            op: AggregateOp::Custom,
            result_name: "value_geomean".to_string(),
            custom_fn: Some(geometric_mean),
        },
        CustomAggregation {
            column: "value".to_string(),
            op: AggregateOp::Custom,
            result_name: "value_range".to_string(),
            custom_fn: Some(range),
        },
        // We can mix custom and standard aggregations
        CustomAggregation {
            column: "value".to_string(),
            op: AggregateOp::Mean,
            result_name: "value_mean".to_string(),
            custom_fn: None,
        },
    ];

    let multi_result = grouped.aggregate_custom(aggregations)?;
    println!("Multiple custom aggregations:");
    println!("{:?}", multi_result);

    // 3. Combining standard and custom aggregations
    println!("\n3. Standard and Advanced Statistics");
    println!("---------------------------------");

    // First, get standard statistics
    let std_stats = grouped.agg(&[
        ("value", AggregateOp::Mean),
        ("value", AggregateOp::Std),
        ("value", AggregateOp::Min),
        ("value", AggregateOp::Max),
    ])?;

    println!("Standard statistics:");
    println!("{:?}", std_stats);

    Ok(())
}
