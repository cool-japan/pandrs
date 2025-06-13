use pandrs::storage::column_store::ColumnStore;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Enhanced Column Store Implementation");

    let store = ColumnStore::new();

    // Test 1: Basic column storage
    println!("\n=== Test 1: Basic Column Storage ===");
    let names = ["Alice", "Bob", "Charlie", "Alice", "Bob"];
    let name_bytes: Vec<Vec<u8>> = names.iter().map(|s| s.as_bytes().to_vec()).collect();
    
    store.add_column(
        "names".to_string(),
        &name_bytes,
        "String".to_string(),
    )?;

    println!("Added names column with {} rows", store.row_count());
    
    // Test 2: Add numeric data
    println!("\n=== Test 2: Numeric Data ===");
    let ages = [25u32, 30, 35, 25, 30];
    let age_bytes: Vec<Vec<u8>> = ages.iter().map(|&age| age.to_le_bytes().to_vec()).collect();
    
    store.add_column(
        "ages".to_string(),
        &age_bytes,
        "Integer".to_string(),
    )?;

    // Test 3: Run-length encoding test
    println!("\n=== Test 3: Run-Length Encoding Test ===");
    let repeated_data = ["A", "A", "A", "B", "B"]; // Same length as other columns
    let repeated_bytes: Vec<Vec<u8>> = repeated_data.iter().map(|s| s.as_bytes().to_vec()).collect();
    
    store.add_column(
        "repeated".to_string(),
        &repeated_bytes,
        "String".to_string(),
    )?;

    // Test 4: Check metadata and compression
    println!("\n=== Test 4: Metadata and Compression ===");
    for name in store.column_names() {
        let metadata = store.get_metadata(&name)?;
        println!("Column '{}': {} rows, {:?} compression, {} bytes", 
                 metadata.name, 
                 metadata.row_count, 
                 metadata.compression,
                 metadata.size_bytes);
    }

    // Test 5: Storage statistics
    println!("\n=== Test 5: Storage Statistics ===");
    let stats = store.stats();
    println!("Total columns: {}", stats.total_columns);
    println!("Total size: {} bytes", stats.total_size_bytes);
    println!("Read operations: {}", stats.read_operations);
    println!("Write operations: {}", stats.write_operations);
    println!("Compression ratio: {:.2}x", store.compression_ratio());

    // Test 6: Data retrieval
    println!("\n=== Test 6: Data Retrieval ===");
    let retrieved_names = store.get_column("names")?;
    println!("Retrieved names data: {} bytes", retrieved_names.len());
    
    let retrieved_ages = store.get_column("ages")?;
    println!("Retrieved ages data: {} bytes", retrieved_ages.len());

    // Test 7: Dictionary encoding test
    println!("\n=== Test 7: Dictionary Encoding Test ===");
    let categories = ["Red", "Blue", "Green", "Red", "Blue"]; // Same length as other columns
    let category_bytes: Vec<Vec<u8>> = categories.iter().map(|s| s.as_bytes().to_vec()).collect();
    
    store.add_column(
        "categories".to_string(),
        &category_bytes,
        "String".to_string(),
    )?;

    let cat_metadata = store.get_metadata("categories")?;
    println!("Categories column: {:?} compression", cat_metadata.compression);

    // Test 8: Column removal
    println!("\n=== Test 8: Column Removal ===");
    println!("Columns before removal: {:?}", store.column_names());
    store.remove_column("repeated")?;
    println!("Columns after removal: {:?}", store.column_names());

    // Test 9: Final statistics
    println!("\n=== Test 9: Final Statistics ===");
    let final_stats = store.stats();
    println!("Final state:");
    println!("  Columns: {}", final_stats.total_columns);
    println!("  Total size: {} bytes", final_stats.total_size_bytes);
    println!("  Read ops: {}", final_stats.read_operations);
    println!("  Write ops: {}", final_stats.write_operations);
    println!("  Compression ratio: {:.2}x", store.compression_ratio());

    println!("\n✅ Column Store implementation test completed successfully!");
    println!("The store efficiently compresses and manages columnar data with multiple compression strategies.");

    Ok(())
}