use pandrs::error::Result;
use pandrs::{GroupBy, Series};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of GroupBy Operations ===");

    // Create sample data
    let values = Series::new(vec![10, 20, 15, 30, 25, 15], Some("values".to_string()))?;

    // Keys for grouping
    let keys = ["A", "B", "A", "C", "B", "A"];

    // Create GroupBy
    let group_by = GroupBy::new(
        keys.iter().map(|s| s.to_string()).collect(),
        &values,
        Some("by_category".to_string()),
    )?;

    println!("Number of groups: {}", group_by.group_count());

    // Get group sizes
    let sizes = group_by.size();
    println!("\n--- Group Sizes ---");
    for (key, size) in &sizes {
        println!("Group '{}': {} elements", key, size);
    }

    // Calculate sum for each group
    let sums = group_by.sum()?;
    println!("\n--- Sum by Group ---");
    for (key, sum) in &sums {
        println!("Sum of group '{}': {}", key, sum);
    }

    // Calculate mean for each group
    let means = group_by.mean()?;
    println!("\n--- Mean by Group ---");
    for (key, mean) in &means {
        println!("Mean of group '{}': {:.2}", key, mean);
    }

    // Grouping with different data types
    println!("\n--- Grouping with Different Data Types ---");
    let ages = Series::new(vec![25, 30, 25, 40, 30, 25], Some("ages".to_string()))?;

    let age_group_by = GroupBy::new(ages.values().to_vec(), &values, Some("by_age".to_string()))?;

    let age_means = age_group_by.mean()?;
    for (key, mean) in &age_means {
        println!("Mean of age group {}: {:.2}", key, mean);
    }

    println!("=== GroupBy Operations Example Complete ===");
    Ok(())
}
