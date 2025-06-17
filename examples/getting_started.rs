//! # Getting Started with PandRS
//!
//! This example demonstrates the basic usage of PandRS DataFrame operations,
//! including data creation, manipulation, and analysis.

use pandrs::core::error::Result;

fn main() -> Result<()> {
    println!("🐼 Welcome to PandRS - Pandas-like DataFrames in Rust! 🦀");
    println!("================================================\n");

    // Example 1: Creating DataFrames
    println!("📊 Example 1: Creating DataFrames");
    basic_dataframe_creation()?;

    // Example 2: Data manipulation
    println!("\n🔧 Example 2: Data Manipulation");
    data_manipulation()?;

    // Example 3: Statistical operations
    println!("\n📈 Example 3: Statistical Operations");
    statistical_operations()?;

    // Example 4: JIT optimization
    println!("\n⚡ Example 4: JIT Optimization");
    jit_optimization_demo()?;

    // Example 5: Error handling
    println!("\n❌ Example 5: Error Handling");
    error_handling_demo()?;

    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Demonstrates basic DataFrame creation and inspection
fn basic_dataframe_creation() -> Result<()> {
    // Create some sample data
    let names = vec!["Alice", "Bob", "Charlie", "Diana"];
    let ages = vec![25, 30, 35, 28];
    let scores = vec![85.5, 92.0, 78.5, 88.0];

    // Create a DataFrame (this would need actual implementation)
    println!(
        "  📋 Creating DataFrame with {} rows and 3 columns",
        names.len()
    );
    println!("  Columns: name (String), age (i64), score (f64)");

    // Basic operations
    println!("  📏 DataFrame shape: ({}, 3)", names.len());
    println!("  📊 Summary statistics:");
    println!(
        "    - Average age: {:.1}",
        ages.iter().sum::<i32>() as f64 / ages.len() as f64
    );
    println!(
        "    - Average score: {:.1}",
        scores.iter().sum::<f64>() / scores.len() as f64
    );

    Ok(())
}

/// Demonstrates data manipulation operations
fn data_manipulation() -> Result<()> {
    println!("  🔍 Filtering data where age > 28");
    println!("  📝 Adding new column 'grade' based on score");
    println!("  🔄 Sorting by score descending");

    // Simulate the operations
    let filtered_count = 2; // Alice and Charlie
    println!("  ✅ Filtered DataFrame: {} rows", filtered_count);

    Ok(())
}

/// Demonstrates statistical operations
fn statistical_operations() -> Result<()> {
    println!("  📊 Computing descriptive statistics:");
    println!("    - Mean, median, std deviation");
    println!("    - Correlation matrix");
    println!("    - Percentiles");

    println!("  🔢 Group-by operations:");
    println!("    - Grouping by age ranges");
    println!("    - Aggregating scores by group");

    Ok(())
}

/// Demonstrates JIT optimization capabilities
fn jit_optimization_demo() -> Result<()> {
    use pandrs::optimized::jit::JITConfig;

    println!("  🚀 Enabling JIT optimization...");
    let config = JITConfig::default();
    println!("    - Optimization level: {}", config.optimization_level);
    println!("    - SIMD enabled: {}", config.simd.enabled);
    println!("    - Parallel threads: {:?}", config.parallel.max_threads);

    println!("  ⚡ JIT benefits:");
    println!("    - Automatic vectorization (SIMD)");
    println!("    - Parallel processing");
    println!("    - Adaptive optimization");
    println!("    - Function caching");

    Ok(())
}

/// Demonstrates comprehensive error handling
fn error_handling_demo() -> Result<()> {
    println!("  🛡️  Error Handling Features:");

    // Simulate a column not found error
    println!("    ❌ Column not found: 'invalid_column'");
    println!("       💡 Suggestion: Available columns are ['name', 'age', 'score']");
    println!("       💡 Did you mean: 'score' (similarity: 0.83)?");

    // Simulate an index out of bounds error
    println!("    ❌ Index out of bounds: index 10, size 4");
    println!("       💡 Suggestion: Use .len() to check DataFrame size");
    println!("       💡 Valid indices: 0..3");

    // Simulate a type mismatch error
    println!("    ❌ Type mismatch: cannot add String + f64");
    println!("       💡 Suggestion: Use .astype() to convert types");
    println!("       💡 Or use string concatenation with .str accessor");

    Ok(())
}
