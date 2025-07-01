use pandrs::{DataFrame, NASeries, ParallelUtils, Series, NA};
use std::error::Error;

#[allow(clippy::result_large_err)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Example of Parallel Processing Features ===\n");

    // Create sample data
    let numbers = Series::new(
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        Some("numbers".to_string()),
    )?;

    // Parallel map: square each number
    println!("Example of parallel map processing:");
    let squared = numbers.par_map(|x| x * x);
    println!("Original values: {:?}", numbers.values());
    println!("Squared values: {:?}", squared.values());

    // Parallel filter: keep only even numbers
    println!("\nExample of parallel filtering:");
    let even_numbers = numbers.par_filter(|x| x % 2 == 0);
    println!("Even Numbers: {:?}", even_numbers.values());

    // Processing data containing NA
    let na_data = vec![
        NA::Value(10),
        NA::Value(20),
        NA::NA,
        NA::Value(40),
        NA::NA,
        NA::Value(60),
    ];
    let na_series = NASeries::new(na_data, Some("na_numbers".to_string()))?;

    println!("\nParallel processing of data containing NA:");
    let na_tripled = na_series.par_map(|x| x * 3);
    println!("Original values: {:?}", na_series.values());
    println!("Tripled values: {:?}", na_tripled.values());

    // Parallel processing of DataFrame
    println!("\nParallel processing of DataFrame:");

    // Creating a sample DataFrame
    let mut df = DataFrame::new();
    let names = Series::new(
        vec!["Alice", "Bob", "Charlie", "David", "Eve"],
        Some("name".to_string()),
    )?;
    let ages = Series::new(vec![25, 30, 35, 40, 45], Some("age".to_string()))?;
    let scores = Series::new(vec![85, 90, 78, 92, 88], Some("score".to_string()))?;

    df.add_column("name".to_string(), names)?;
    df.add_column("age".to_string(), ages)?;
    df.add_column("score".to_string(), scores)?;

    // Parallel transformation of DataFrame
    println!("Example of DataFrame.par_apply:");
    let transformed_df = df.par_apply(|col, _row, val| {
        match col {
            "age" => {
                // Add 1 to age
                let age: i32 = val.parse().unwrap_or(0);
                (age + 1).to_string()
            }
            "score" => {
                // Add 5 to score
                let score: i32 = val.parse().unwrap_or(0);
                (score + 5).to_string()
            }
            _ => val.to_string(),
        }
    })?;

    println!(
        "Original DF row count: {}, column count: {}",
        df.row_count(),
        df.column_count()
    );
    println!(
        "Transformed DF row count: {}, column count: {}",
        transformed_df.row_count(),
        transformed_df.column_count()
    );

    // Filtering rows
    println!("\nExample of DataFrame.par_filter_rows:");
    let filtered_df = df.par_filter_rows(|row| {
        // Keep only rows where score > 85
        if let Ok(values) = df.get_column_numeric_values("score") {
            if row < values.len() {
                return values[row] > 85.0;
            }
        }
        false
    })?;

    println!("Row count after filtering: {}", filtered_df.row_count());

    // Example of using ParallelUtils
    println!("\nExample of ParallelUtils features:");

    let unsorted = vec![5, 3, 8, 1, 9, 4, 7, 2, 6];
    let sorted = ParallelUtils::par_sort(unsorted.clone());
    println!("Before sorting: {:?}", unsorted);
    println!("After sorting: {:?}", sorted);

    let numbers_vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let sum = ParallelUtils::par_sum(&numbers_vec);
    let mean = ParallelUtils::par_mean(&numbers_vec);
    let min = ParallelUtils::par_min(&numbers_vec);
    let max = ParallelUtils::par_max(&numbers_vec);

    println!("Sum: {}", sum);
    println!("Mean: {}", mean.unwrap());
    println!("Min: {}", min.unwrap());
    println!("Max: {}", max.unwrap());

    println!("\n=== Example of Parallel Processing Features Complete ===");
    Ok(())
}
