// PandRS Optimized Statistical Functions Sample
// Example of using statistical functions with optimized implementation

use pandrs::error::Result;
use pandrs::{LazyFrame, OptimizedDataFrame, TTestResult};
use rand::{rng, Rng};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS Optimized Statistical Module Sample\n");

    // Example of descriptive statistics
    optimized_descriptive_stats_example()?;

    // Example of inferential statistics and hypothesis testing
    optimized_ttest_example()?;

    // Example of regression analysis
    optimized_regression_example()?;

    Ok(())
}

#[allow(clippy::result_large_err)]
fn optimized_descriptive_stats_example() -> Result<()> {
    println!("1. Example of Descriptive Statistics with Optimized Implementation");
    println!("--------------------------");

    // Create an optimized DataFrame
    let mut df = OptimizedDataFrame::new();

    // Add a column with type specialization
    let values = vec![10.5, 12.3, 15.2, 9.8, 11.5, 13.7, 14.3, 12.9, 8.5, 10.2];
    df.add_column(
        "Values".to_string(),
        pandrs::Column::Float64(pandrs::column::Float64Column::new(values.clone())),
    )?;

    // Speed up descriptive statistics calculation
    let stats = pandrs::stats::describe(&values)?;

    // Display results
    println!("Count: {}", stats.count);
    println!("Mean: {:.2}", stats.mean);
    println!("Standard Deviation: {:.2}", stats.std);
    println!("Minimum: {:.2}", stats.min);
    println!("First Quartile: {:.2}", stats.q1);
    println!("Median: {:.2}", stats.median);
    println!("Third Quartile: {:.2}", stats.q3);
    println!("Maximum: {:.2}", stats.max);

    // Use parallel computation for covariance and correlation coefficient calculation
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data2 = vec![1.5, 3.1, 4.2, 5.8, 7.1];

    let cov = pandrs::stats::covariance(&data1, &data2)?;
    let corr = pandrs::stats::correlation(&data1, &data2)?;

    println!("\nCovariance and Correlation Coefficient:");
    println!("Covariance: {:.4}", cov);
    println!("Correlation Coefficient: {:.4}", corr);

    println!();
    Ok(())
}

#[allow(clippy::result_large_err)]
fn optimized_ttest_example() -> Result<()> {
    println!("2. Example of t-test with Optimized Implementation");
    println!("-------------------------");

    // Use an optimized DataFrame
    let mut df = OptimizedDataFrame::new();

    // Prepare data for two groups
    let group1 = vec![5.2, 5.8, 6.1, 5.5, 5.9, 6.2, 5.7, 6.0, 5.6, 5.8];
    let group2 = vec![4.8, 5.1, 5.3, 4.9, 5.0, 5.2, 4.7, 5.1, 4.9, 5.0];

    // Add columns with type specialization
    df.add_column(
        "Group 1".to_string(),
        pandrs::Column::Float64(pandrs::column::Float64Column::new(group1.clone())),
    )?;
    df.add_column(
        "Group 2".to_string(),
        pandrs::Column::Float64(pandrs::column::Float64Column::new(group2.clone())),
    )?;

    // Perform t-test with significance level 0.05 (5%)
    let alpha = 0.05;

    // t-test assuming equal variances
    let result_equal = pandrs::stats::ttest(&group1, &group2, alpha, true)?;

    println!("t-test Result Assuming Equal Variances:");
    print_ttest_result(&result_equal);

    // Welch's t-test (not assuming equal variances)
    let result_welch = pandrs::stats::ttest(&group1, &group2, alpha, false)?;

    println!("\nWelch's t-test Result (Not Assuming Equal Variances):");
    print_ttest_result(&result_welch);

    println!();
    Ok(())
}

fn print_ttest_result(result: &TTestResult) {
    println!("t-statistic: {:.4}", result.statistic);
    println!("p-value: {:.4}", result.pvalue);
    println!("Degrees of Freedom: {}", result.df);
    println!(
        "Significant: {}",
        if result.significant { "Yes" } else { "No" }
    );
}

#[allow(clippy::result_large_err)]
fn optimized_regression_example() -> Result<()> {
    println!("3. Example of Regression Analysis with Optimized Implementation");
    println!("--------------------------");

    // Create an optimized DataFrame
    let mut opt_df = OptimizedDataFrame::new();

    // Add explanatory variables as type-specialized columns
    let x1: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let x2: Vec<f64> = (1..=10).map(|i| 5.0 + 3.0 * i as f64).collect();

    opt_df.add_column(
        "x1".to_string(),
        pandrs::Column::Float64(pandrs::column::Float64Column::new(x1.clone())),
    )?;
    opt_df.add_column(
        "x2".to_string(),
        pandrs::Column::Float64(pandrs::column::Float64Column::new(x2.clone())),
    )?;

    // Dependent variable (y = 2*x1 + 1.5*x2 + 3 + noise)
    let mut y_values = Vec::with_capacity(10);
    let mut rng = rng();

    for i in 0..10 {
        let noise = rng.random_range(-1.0..1.0);
        let y_val = 2.0 * x1[i] + 1.5 * x2[i] + 3.0 + noise;
        y_values.push(y_val);
    }

    opt_df.add_column(
        "y".to_string(),
        pandrs::Column::Float64(pandrs::column::Float64Column::new(y_values.clone())),
    )?;

    // Convert to a regular DataFrame (to match the interface of regression analysis functions)
    let mut df = pandrs::DataFrame::new();

    // Add x1, x2, y columns to the regular DataFrame
    df.add_column(
        "x1".to_string(),
        pandrs::Series::new(x1.clone(), Some("x1".to_string()))?,
    )?;
    df.add_column(
        "x2".to_string(),
        pandrs::Series::new(x2.clone(), Some("x2".to_string()))?,
    )?;
    df.add_column(
        "y".to_string(),
        pandrs::Series::new(y_values.clone(), Some("y".to_string()))?,
    )?;

    // Perform regression analysis
    let model = pandrs::stats::linear_regression(&df, "y", &["x1", "x2"])?;

    // Display results
    println!(
        "Linear Regression Model: y = {:.4} + {:.4} × x1 + {:.4} × x2",
        model.intercept, model.coefficients[0], model.coefficients[1]
    );
    println!("R²: {:.4}", model.r_squared);
    println!("Adjusted R²: {:.4}", model.adj_r_squared);
    println!("p-values of regression coefficients: {:?}", model.p_values);

    // Example using LazyFrame (leveraging lazy evaluation)
    println!("\nOperations with LazyFrame:");

    let lazy_df = LazyFrame::new(opt_df);

    // Perform column selection and filtering by condition at once
    // These operations are not executed until actually needed
    let filtered = lazy_df
        .select(&["x1", "x2", "y"])
        // Filter condition for LazyFrame is specified as a simple string expression
        .filter("x1 > 5.0")
        .execute()?;

    println!("Number of rows after filtering: {}", filtered.row_count());

    // Example of simple regression
    println!("\nSimple Regression Model (x1 only):");
    let model_simple = pandrs::stats::linear_regression(&df, "y", &["x1"])?;
    println!(
        "Linear Regression Model: y = {:.4} + {:.4} × x1",
        model_simple.intercept, model_simple.coefficients[0]
    );
    println!("R²: {:.4}", model_simple.r_squared);

    Ok(())
}
