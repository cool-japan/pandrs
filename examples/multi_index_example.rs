use pandrs::error::Result;
use pandrs::{DataFrame, Index, MultiIndex};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of Using MultiIndex ===\n");

    // =========================================
    // Creating a MultiIndex
    // =========================================

    println!("--- Creating MultiIndex from Tuples ---");

    // Create MultiIndex from tuples (vector of vectors)
    let tuples = vec![
        vec!["A".to_string(), "a".to_string()],
        vec!["A".to_string(), "b".to_string()],
        vec!["B".to_string(), "a".to_string()],
        vec!["B".to_string(), "b".to_string()],
    ];

    let names = Some(vec![Some("first".to_string()), Some("second".to_string())]);
    let multi_idx = MultiIndex::from_tuples(tuples, names)?;

    println!("MultiIndex: {:?}\n", multi_idx);
    println!("Number of Levels: {}", multi_idx.n_levels());
    println!("Number of Rows: {}\n", multi_idx.len());

    // =========================================
    // Operations on MultiIndex
    // =========================================

    println!("--- Retrieving Level Values ---");
    let level0_values = multi_idx.get_level_values(0)?;
    println!("Values in Level 0: {:?}", level0_values);

    let level1_values = multi_idx.get_level_values(1)?;
    println!("Values in Level 1: {:?}", level1_values);

    println!("--- Swapping Levels ---");
    let swapped = multi_idx.swaplevel(0, 1)?;
    println!("After Swapping Levels: {:?}\n", swapped);

    // =========================================
    // DataFrame with MultiIndex
    // =========================================

    println!("--- DataFrame with MultiIndex ---");

    // Create DataFrame
    let mut df = DataFrame::with_multi_index(multi_idx.clone());

    // Add data
    let data = vec![
        "data1".to_string(),
        "data2".to_string(),
        "data3".to_string(),
        "data4".to_string(),
    ];
    df.add_column(
        "data".to_string(),
        pandrs::Series::new(data, Some("data".to_string()))?,
    )?;

    println!("DataFrame: {:?}\n", df);
    println!("Number of Rows: {}", df.row_count());
    println!("Number of Columns: {}", df.column_count());

    // =========================================
    // Conversion Between Simple Index and MultiIndex
    // =========================================

    println!("\n--- Example of Index Conversion ---");

    // Create DataFrame from simple index
    let simple_idx = Index::new(vec!["X".to_string(), "Y".to_string(), "Z".to_string()])?;
    let mut simple_df = DataFrame::with_index(simple_idx);

    // Add data
    let values = [100, 200, 300];
    let str_values: Vec<String> = values.iter().map(|v| v.to_string()).collect();
    simple_df.add_column(
        "values".to_string(),
        pandrs::Series::new(str_values, Some("values".to_string()))?,
    )?;

    println!("Simple Index DataFrame: {:?}", simple_df);

    // Prepare for conversion to MultiIndex
    let tuples = vec![
        vec!["Category".to_string(), "X".to_string()],
        vec!["Category".to_string(), "Y".to_string()],
        vec!["Category".to_string(), "Z".to_string()],
    ];

    // Create and set MultiIndex
    let new_multi_idx = MultiIndex::from_tuples(tuples, None)?;
    simple_df.set_multi_index(new_multi_idx)?;

    println!("After Conversion to MultiIndex: {:?}", simple_df);

    println!("\n=== Sample Complete ===");
    Ok(())
}
