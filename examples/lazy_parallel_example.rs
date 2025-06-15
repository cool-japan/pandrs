use std::error::Error;
use std::time::Instant;

use pandrs::{
    AggregateOp, Column, Float64Column, Int64Column, LazyFrame, OptimizedDataFrame, StringColumn,
};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Performance Evaluation with Lazy Evaluation and Parallel Processing\n");

    // Generate a large DataFrame
    println!("Generating a large dataset...");
    let rows = 100_000;
    let df = generate_large_dataframe(rows)?;
    println!("Generated {} rows of data", rows);

    // Display a portion of the DataFrame
    println!("\nFirst row of data:");
    println!("{:?}\n", df);

    // Filtering and aggregation with the standard approach
    println!("Executing data processing with the standard approach...");
    let start = Instant::now();

    // Create an age filter (30 years and older)
    let age_col = df.column("Age")?;
    let mut age_filter = vec![false; df.row_count()];
    if let Some(int_col) = age_col.as_int64() {
        for (i, filter_val) in age_filter.iter_mut().enumerate().take(df.row_count()) {
            if let Ok(Some(age)) = int_col.get(i) {
                *filter_val = age >= 30;
            }
        }
    }

    // Add the filter to the DataFrame
    let bool_data = pandrs::BooleanColumn::new(age_filter);
    let mut df_with_filter = df.clone();
    df_with_filter.add_column("30 and older", Column::Boolean(bool_data))?;

    // Execute filtering
    let filtered_df = df_with_filter.filter("30 and older")?;

    // Manually aggregate by department
    let dept_col = filtered_df.column("Department")?;
    let salary_col = filtered_df.column("Salary")?;

    // Aggregation by department
    let mut dept_totals: std::collections::HashMap<String, (f64, i32)> =
        std::collections::HashMap::new();

    if let (Some(str_col), Some(float_col)) = (dept_col.as_string(), salary_col.as_float64()) {
        for i in 0..filtered_df.row_count() {
            if let (Ok(Some(dept)), Ok(Some(salary))) = (str_col.get(i), float_col.get(i)) {
                let entry = dept_totals.entry(dept.to_string()).or_insert((0.0, 0));
                entry.0 += salary;
                entry.1 += 1;
            }
        }
    }

    // Construct the result
    let mut result_depts = Vec::new();
    let mut result_totals = Vec::new();
    let mut result_avgs = Vec::new();
    let mut result_counts = Vec::new();

    for (dept, (total, count)) in dept_totals {
        result_depts.push(dept);
        result_totals.push(total);
        result_avgs.push(total / count as f64);
        result_counts.push(count as f64);
    }

    // Create the result DataFrame
    let mut result_df = OptimizedDataFrame::new();
    result_df.add_column(
        "Department",
        Column::String(StringColumn::new(result_depts)),
    )?;
    result_df.add_column(
        "Total Salary",
        Column::Float64(Float64Column::new(result_totals)),
    )?;
    result_df.add_column(
        "Average Salary",
        Column::Float64(Float64Column::new(result_avgs)),
    )?;
    result_df.add_column("Count", Column::Float64(Float64Column::new(result_counts)))?;

    let standard_duration = start.elapsed();
    println!(
        "Processing time with the standard approach: {:?}",
        standard_duration
    );
    println!("\nResults of the standard approach:");
    println!("{:?}\n", result_df);

    // Approach using LazyFrame and parallel processing
    println!("Approach using LazyFrame and parallel processing...");
    let start = Instant::now();

    // First, create a boolean column for the filter
    let age_col = df.column("Age")?;
    let mut age_filter = vec![false; df.row_count()];
    if let Some(int_col) = age_col.as_int64() {
        for (i, filter_val) in age_filter.iter_mut().enumerate().take(df.row_count()) {
            if let Ok(Some(age)) = int_col.get(i) {
                *filter_val = age >= 30;
            }
        }
    }

    // Add the filter to the DataFrame
    let mut df_with_age_filter = df.clone();
    let bool_data = pandrs::BooleanColumn::new(age_filter);
    df_with_age_filter.add_column("Age Filter", Column::Boolean(bool_data))?;

    // Recreate the LazyFrame
    let lazy_df = LazyFrame::new(df_with_age_filter);
    let result_lazy = lazy_df
        .filter("Age Filter") // Use the newly added boolean column for filtering
        .aggregate(
            vec!["Department".to_string()],
            vec![
                ("Salary".to_string(), AggregateOp::Sum, "Total Salary".to_string()),
                ("Salary".to_string(), AggregateOp::Mean, "Average Salary".to_string()),
                ("Salary".to_string(), AggregateOp::Count, "Count".to_string()),
            ]
        );

    // Display the execution plan
    println!("\nExecution Plan:");
    println!("{}", result_lazy.explain());

    // Execute
    let lazy_result = result_lazy.execute()?;

    let lazy_duration = start.elapsed();
    println!(
        "Processing time with the LazyFrame approach: {:?}",
        lazy_duration
    );
    println!("\nResults of the LazyFrame approach:");
    println!("{:?}\n", lazy_result);

    // Performance comparison
    let speedup = standard_duration.as_secs_f64() / lazy_duration.as_secs_f64();
    println!(
        "The LazyFrame approach is {:.2} times faster than the standard approach",
        speedup
    );

    Ok(())
}

// Function to generate a large DataFrame
fn generate_large_dataframe(rows: usize) -> Result<OptimizedDataFrame, Box<dyn Error>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility

    // Generate data
    let mut ids = Vec::with_capacity(rows);
    let mut ages = Vec::with_capacity(rows);
    let mut depts = Vec::with_capacity(rows);
    let mut salaries = Vec::with_capacity(rows);

    // List of departments
    let departments = [
        "Sales".to_string(),
        "Development".to_string(),
        "HR".to_string(),
        "Finance".to_string(),
        "Marketing".to_string(),
    ];

    for i in 0..rows {
        ids.push(i as i64 + 1000); // ID
        ages.push(rng.random_range(20..60)); // Age
        depts.push(departments[rng.random_range(0..departments.len())].clone()); // Department

        // Salary (generated based on department and age)
        let base_salary = match depts.last().unwrap().as_str() {
            "Sales" => 350_000.0,
            "Development" => 400_000.0,
            "HR" => 320_000.0,
            "Finance" => 380_000.0,
            "Marketing" => 360_000.0,
            _ => 300_000.0,
        };

        let age_factor = *ages.last().unwrap() as f64 / 30.0;
        let variation = rng.random_range(0.8..1.2);

        salaries.push(base_salary * age_factor * variation);
    }

    // Create the DataFrame
    let mut df = OptimizedDataFrame::new();
    df.add_column("ID", Column::Int64(Int64Column::new(ids)))?;
    df.add_column("Age", Column::Int64(Int64Column::new(ages)))?;
    df.add_column("Department", Column::String(StringColumn::new(depts)))?;
    df.add_column("Salary", Column::Float64(Float64Column::new(salaries)))?;

    Ok(df)
}
