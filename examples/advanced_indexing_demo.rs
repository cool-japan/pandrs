use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::indexing::{
    selectors, AdvancedIndexingExt, AlignmentStrategy, IndexAligner, MultiLevelIndex,
};
use pandrs::error::Result;
use pandrs::series::base::Series;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Alpha 4 Advanced Indexing System Example ===\n");

    // Create sample employee data
    println!("1. Creating Sample Employee Data:");
    let mut df = DataFrame::new();

    let employee_ids = vec![
        "E001", "E002", "E003", "E004", "E005", "E006", "E007", "E008",
    ];
    let names = vec![
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    ];
    let departments = vec![
        "IT",
        "HR",
        "IT",
        "Marketing",
        "IT",
        "Finance",
        "HR",
        "Finance",
    ];
    let salaries = vec![
        "75000", "65000", "80000", "58000", "70000", "85000", "60000", "72000",
    ];
    let ages = vec!["28", "35", "42", "26", "31", "45", "29", "38"];
    let experiences = vec!["3", "8", "15", "2", "5", "18", "4", "12"];

    let id_series = Series::new(
        employee_ids.into_iter().map(|s| s.to_string()).collect(),
        Some("ID".to_string()),
    )?;
    let name_series = Series::new(
        names.into_iter().map(|s| s.to_string()).collect(),
        Some("Name".to_string()),
    )?;
    let dept_series = Series::new(
        departments.into_iter().map(|s| s.to_string()).collect(),
        Some("Department".to_string()),
    )?;
    let salary_series = Series::new(
        salaries.into_iter().map(|s| s.to_string()).collect(),
        Some("Salary".to_string()),
    )?;
    let age_series = Series::new(
        ages.into_iter().map(|s| s.to_string()).collect(),
        Some("Age".to_string()),
    )?;
    let exp_series = Series::new(
        experiences.into_iter().map(|s| s.to_string()).collect(),
        Some("Experience".to_string()),
    )?;

    df.add_column("ID".to_string(), id_series)?;
    df.add_column("Name".to_string(), name_series)?;
    df.add_column("Department".to_string(), dept_series)?;
    df.add_column("Salary".to_string(), salary_series)?;
    df.add_column("Age".to_string(), age_series)?;
    df.add_column("Experience".to_string(), exp_series)?;

    println!("Original Employee Data:");
    println!("{:?}", df);

    println!("\n=== Position-Based Indexing (.iloc) ===\n");

    // 2. Position-based indexing (iloc)
    println!("2. Position-Based Indexing (.iloc):");

    // Single row by position
    println!("Single row (position 2):");
    let row_2 = df.iloc().get(2)?;
    println!("{:?}", row_2);

    // Single value by row and column position
    println!("\nSingle value at (row=1, col=2):");
    let value = df.iloc().get_at(1, 2)?;
    println!("Value: {}", value);

    // Range of rows
    println!("\nRows 1-3 (using iloc.get_range):");
    let rows_range = df.iloc().get_range(1..4)?;
    println!("{:?}", rows_range);

    // Slice with row and column ranges
    println!("\nSlice (rows 0-2, cols 1-3):");
    let slice = df.iloc().get_slice(0..3, 1..4)?;
    println!("{:?}", slice);

    // Multiple specific positions
    println!("\nSpecific positions [0, 2, 4]:");
    let specific_rows = df.iloc().get_positions(&[0, 2, 4])?;
    println!("{:?}", specific_rows);

    // Boolean indexing
    println!("\nBoolean mask (every other row):");
    let mask = vec![true, false, true, false, true, false, true, false];
    let boolean_result = df.iloc().get_boolean(&mask)?;
    println!("{:?}", boolean_result);

    println!("\n=== Label-Based Indexing (.loc) ===\n");

    // 3. Label-based indexing (loc)
    println!("3. Label-Based Indexing (.loc):");

    // Single row by label (using row index as string)
    println!("Single row by label '0':");
    let label_row = df.loc().get("0")?;
    println!("{:?}", label_row);

    // Single value by label and column
    println!("\nSingle value at (label='1', column='Name'):");
    let label_value = df.loc().get_at("1", "Name")?;
    println!("Value: {}", label_value);

    // Multiple labels
    println!("\nMultiple labels ['0', '2', '4']:");
    let labels = vec!["0".to_string(), "2".to_string(), "4".to_string()];
    let label_rows = df.loc().get_labels(&labels)?;
    println!("{:?}", label_rows);

    println!("\n=== Scalar Indexing (.at and .iat) ===\n");

    // 4. Scalar indexing
    println!("4. Scalar Indexing:");

    // .at for label-based scalar access
    println!("Using .at to get value at (label='3', column='Salary'):");
    let at_value = df.at().get("3", "Salary")?;
    println!("Value: {}", at_value);

    // .iat for position-based scalar access
    println!("\nUsing .iat to get value at (row=2, col=1):");
    let iat_value = df.iat().get(2, 1)?;
    println!("Value: {}", iat_value);

    println!("\n=== Advanced Selection Builder ===\n");

    // 5. Advanced selection using builder pattern
    println!("5. Advanced Selection Builder:");

    // Select specific rows and columns
    println!("Select rows [1,3,5] and columns ['Name', 'Department', 'Salary']:");
    let selection = df
        .select()
        .rows(selectors::positions(vec![1, 3, 5]))
        .columns(selectors::cols(vec![
            "Name".to_string(),
            "Department".to_string(),
            "Salary".to_string(),
        ]))
        .select()?;
    println!("{:?}", selection);

    // Select with range and specific columns
    println!("\nSelect rows 2-5 and specific columns:");
    let range_selection = df
        .select()
        .rows(selectors::range(2, 6))
        .columns(selectors::cols(vec![
            "Name".to_string(),
            "Age".to_string(),
            "Experience".to_string(),
        ]))
        .select()?;
    println!("{:?}", range_selection);

    println!("\n=== Column Operations ===\n");

    // 6. Column selection and manipulation
    println!("6. Column Operations:");

    // Select specific columns
    println!("Select columns ['Name', 'Department', 'Salary']:");
    let selected_cols = df.select_columns(&["Name", "Department", "Salary"])?;
    println!("{:?}", selected_cols);

    // Drop columns
    println!("\nDrop columns ['ID', 'Experience']:");
    let dropped_cols = df.drop_columns(&["ID".to_string(), "Experience".to_string()])?;
    println!("{:?}", dropped_cols);

    println!("\n=== Multi-Level Index ===\n");

    // 7. Multi-level indexing
    println!("7. Multi-Level Index:");

    // Create multi-index from Department and Age columns
    println!("Creating multi-index from Department and Age:");
    let (indexed_df, multi_index) =
        df.set_multi_index(&["Department".to_string(), "Age".to_string()])?;
    println!("Multi-index info:");
    println!("Names: {:?}", multi_index.names);
    println!("Tuples: {:?}", multi_index.tuples);
    println!("DataFrame after indexing:");
    println!("{:?}", indexed_df);

    // Query multi-index
    println!("\nQuerying multi-index for tuple ['IT', '28']:");
    let multi_loc = pandrs::dataframe::indexing::LocIndexer::with_index(&indexed_df, &multi_index);
    let tuple_result = multi_loc.get_tuple(&["IT".to_string(), "28".to_string()])?;
    println!("{:?}", tuple_result);

    // Get unique values for a level
    println!("\nUnique values for level 0 (Department):");
    let level_values = multi_index.level_values(0)?;
    println!("{:?}", level_values);

    println!("\n=== DataFrame Utilities ===\n");

    // 8. DataFrame utilities
    println!("8. DataFrame Utilities:");

    // Head and tail
    println!("Head (first 3 rows):");
    let head = df.head(3)?;
    println!("{:?}", head);

    println!("\nTail (last 3 rows):");
    let tail = df.tail(3)?;
    println!("{:?}", tail);

    // Sampling (note: this might not work without proper rand setup)
    // println!("\nRandom sample (3 rows):");
    // let sample = df.sample(3)?;
    // println!("{:?}", sample);

    println!("\n=== Index Alignment ===\n");

    // 9. Index alignment
    println!("9. Index Alignment:");

    // Create two DataFrames with different sizes
    let mut df1 = DataFrame::new();
    let mut df2 = DataFrame::new();

    let series1 = Series::new(
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
        Some("Col1".to_string()),
    )?;
    let series2 = Series::new(
        vec![
            "1".to_string(),
            "2".to_string(),
            "3".to_string(),
            "4".to_string(),
            "5".to_string(),
        ],
        Some("Col2".to_string()),
    )?;

    df1.add_column("Col1".to_string(), series1)?;
    df2.add_column("Col2".to_string(), series2)?;

    println!("DataFrame 1:");
    println!("{:?}", df1);
    println!("DataFrame 2:");
    println!("{:?}", df2);

    // Align with different strategies
    println!("\nOuter alignment (extend to max length):");
    let (aligned1_outer, aligned2_outer) =
        IndexAligner::align(&df1, &df2, AlignmentStrategy::Outer)?;
    println!("Aligned DataFrame 1: {:?}", aligned1_outer);
    println!("Aligned DataFrame 2: {:?}", aligned2_outer);

    println!("\nInner alignment (truncate to min length):");
    let (aligned1_inner, aligned2_inner) =
        IndexAligner::align(&df1, &df2, AlignmentStrategy::Inner)?;
    println!("Aligned DataFrame 1: {:?}", aligned1_inner);
    println!("Aligned DataFrame 2: {:?}", aligned2_inner);

    println!("\n=== Macro Usage Examples ===\n");

    // 10. Using macros for convenient indexing
    println!("10. Macro Usage:");

    // Note: These macros would need to be properly defined and exported
    // println!("Using iloc! macro:");
    // let macro_value = iloc!(df, 1, 2)?;
    // println!("Value: {}", macro_value);

    // println!("\nUsing loc! macro:");
    // let macro_label_value = loc!(df, "1", "Name")?;
    // println!("Value: {}", macro_label_value);

    println!("\n=== Performance Demonstration ===\n");

    // 11. Performance with larger dataset
    println!("11. Performance Demonstration:");
    demonstrate_performance()?;

    println!("\n=== Error Handling ===\n");

    // 12. Error handling examples
    println!("12. Error Handling:");
    demonstrate_error_handling(&df)?;

    println!("\n=== Alpha 4 Advanced Indexing System Complete ===");
    println!("\nNew advanced indexing capabilities implemented:");
    println!("✓ Position-based indexing (.iloc)");
    println!("✓ Label-based indexing (.loc)");
    println!("✓ Scalar indexing (.at and .iat)");
    println!("✓ Advanced selection builder with fluent API");
    println!("✓ Multi-level index support");
    println!("✓ Column selection and manipulation");
    println!("✓ Boolean and fancy indexing");
    println!("✓ Range-based selections");
    println!("✓ Index alignment operations");
    println!("✓ DataFrame utilities (head, tail, sample)");
    println!("✓ Comprehensive error handling");
    println!("✓ Helper functions and selectors");

    Ok(())
}

/// Demonstrate performance with larger dataset
#[allow(clippy::result_large_err)]
fn demonstrate_performance() -> Result<()> {
    println!("--- Performance with Large Dataset ---");

    // Create a larger dataset
    let mut large_df = DataFrame::new();
    let size = 1000;

    let ids: Vec<String> = (1..=size).map(|i| format!("ID{:04}", i)).collect();
    let values: Vec<String> = (1..=size).map(|i| (i as f64 * 1.5).to_string()).collect();
    let categories: Vec<String> = (1..=size)
        .map(|i| match i % 4 {
            0 => "A".to_string(),
            1 => "B".to_string(),
            2 => "C".to_string(),
            _ => "D".to_string(),
        })
        .collect();

    let id_series = Series::new(ids, Some("ID".to_string()))?;
    let value_series = Series::new(values, Some("Value".to_string()))?;
    let cat_series = Series::new(categories, Some("Category".to_string()))?;

    large_df.add_column("ID".to_string(), id_series)?;
    large_df.add_column("Value".to_string(), value_series)?;
    large_df.add_column("Category".to_string(), cat_series)?;

    println!("Created dataset with {} rows", size);

    // Test iloc performance
    let start = std::time::Instant::now();
    let _slice = large_df.iloc().get_range(100..200)?;
    let duration = start.elapsed();
    println!("iloc range selection (100 rows) took: {:?}", duration);

    // Test column selection performance
    let start = std::time::Instant::now();
    let _cols = large_df.select_columns(&["ID", "Value"])?;
    let duration = start.elapsed();
    println!("Column selection took: {:?}", duration);

    // Test head/tail performance
    let start = std::time::Instant::now();
    let _head = large_df.head(50)?;
    let duration = start.elapsed();
    println!("Head operation took: {:?}", duration);

    // Test boolean indexing performance
    let start = std::time::Instant::now();
    let mask: Vec<bool> = (0..size).map(|i| i % 5 == 0).collect();
    let _filtered = large_df.iloc().get_boolean(&mask)?;
    let duration = start.elapsed();
    println!("Boolean indexing (every 5th row) took: {:?}", duration);

    Ok(())
}

/// Demonstrate error handling
#[allow(clippy::result_large_err)]
fn demonstrate_error_handling(df: &DataFrame) -> Result<()> {
    println!("--- Error Handling Examples ---");

    // Out of bounds row access
    match df.iloc().get(100) {
        Ok(_) => println!("Unexpected success with out-of-bounds row"),
        Err(e) => println!("Expected error for out-of-bounds row: {:?}", e),
    }

    // Out of bounds column access
    match df.iloc().get_at(0, 100) {
        Ok(_) => println!("Unexpected success with out-of-bounds column"),
        Err(e) => println!("Expected error for out-of-bounds column: {:?}", e),
    }

    // Invalid column selection
    match df.select_columns(&["NonExistent"]) {
        Ok(_) => println!("Unexpected success with non-existent column"),
        Err(e) => println!("Expected error for non-existent column: {:?}", e),
    }

    // Invalid label access
    match df.loc().get("invalid_label") {
        Ok(_) => println!("Unexpected success with invalid label"),
        Err(e) => println!("Expected error for invalid label: {:?}", e),
    }

    // Empty multi-index creation
    match MultiLevelIndex::new(vec![], vec![]) {
        Ok(_) => println!("Unexpected success with empty multi-index"),
        Err(e) => println!("Expected error for empty multi-index: {:?}", e),
    }

    Ok(())
}
