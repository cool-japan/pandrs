// PandRS statistical functions sample

use pandrs::error::Result;
use pandrs::{DataFrame, Series, TTestResult};
use rand::Rng;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("PandRS Statistical Module Sample\n");

    // Example of descriptive statistics
    descriptive_stats_example()?;

    // Example of inferential statistics and hypothesis testing
    ttest_example()?;

    // Example of regression analysis
    regression_example()?;

    Ok(())
}

#[allow(clippy::result_large_err)]
fn descriptive_stats_example() -> Result<()> {
    println!("1. Descriptive Statistics Sample");
    println!("-----------------");

    // Create dataset
    let mut df = DataFrame::new();
    let values = Series::new(
        vec![10.5, 12.3, 15.2, 9.8, 11.5, 13.7, 14.3, 12.9, 8.5, 10.2],
        Some("Values".to_string()),
    )?;

    df.add_column("Values".to_string(), values)?;

    // Descriptive statistics
    let stats = pandrs::stats::describe(
        df.get_column("Values")
            .unwrap()
            .values()
            .iter()
            .map(|v: &String| v.parse::<f64>().unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    )?;

    // Display results
    println!("Count: {}", stats.count);
    println!("Mean: {:.2}", stats.mean);
    println!("Standard Deviation: {:.2}", stats.std);
    println!("Min: {:.2}", stats.min);
    println!("First Quartile: {:.2}", stats.q1);
    println!("Median: {:.2}", stats.median);
    println!("Third Quartile: {:.2}", stats.q3);
    println!("Max: {:.2}", stats.max);

    // Covariance and correlation coefficient
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
fn ttest_example() -> Result<()> {
    println!("2. t-test Sample");
    println!("--------------");

    // Create sample data
    let group1 = vec![5.2, 5.8, 6.1, 5.5, 5.9, 6.2, 5.7, 6.0, 5.6, 5.8];
    let group2 = vec![4.8, 5.1, 5.3, 4.9, 5.0, 5.2, 4.7, 5.1, 4.9, 5.0];

    // Perform t-test with significance level 0.05 (5%)
    let alpha = 0.05;

    // t-test assuming equal variances
    let result_equal = pandrs::stats::ttest(&group1, &group2, alpha, true)?;

    println!("t-test result assuming equal variances:");
    print_ttest_result(&result_equal);

    // Welch's t-test (not assuming equal variances)
    let result_welch = pandrs::stats::ttest(&group1, &group2, alpha, false)?;

    println!("\nWelch's t-test result (not assuming equal variances):");
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
fn regression_example() -> Result<()> {
    println!("3. Regression Analysis Sample");
    println!("-----------------");

    // Create dataset
    let mut df = DataFrame::new();

    // Explanatory variables
    let x1 = Series::new(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        Some("x1".to_string()),
    )?;
    let x2 = Series::new(
        vec![5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0],
        Some("x2".to_string()),
    )?;

    // Dependent variable (y = 2*x1 + 1.5*x2 + 3 + noise)
    let mut y_values = Vec::with_capacity(10);
    let mut rng = rand::rng();

    for i in 0..10 {
        let noise = rng.random_range(-1.0..1.0);
        let y_val = 2.0 * (i as f64 + 1.0) + 1.5 * (5.0 + 3.0 * i as f64) + 3.0 + noise;
        y_values.push(y_val);
    }

    let y = Series::new(y_values, Some("y".to_string()))?;

    // Add to DataFrame
    df.add_column("x1".to_string(), x1)?;
    df.add_column("x2".to_string(), x2)?;
    df.add_column("y".to_string(), y)?;

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

    // Simple regression example
    println!("\nSimple Regression Model (x1 only):");
    let model_simple = pandrs::stats::linear_regression(&df, "y", &["x1"])?;
    println!(
        "Linear Regression Model: y = {:.4} + {:.4} × x1",
        model_simple.intercept, model_simple.coefficients[0]
    );
    println!("R²: {:.4}", model_simple.r_squared);

    Ok(())
}
