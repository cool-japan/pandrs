use pandrs::column::{Column, ColumnTrait, StringColumn};
use pandrs::optimized::dataframe::OptimizedDataFrame;
use std::time::Instant;

fn main() {
    println!("=== Test of Extended Features of Optimized DataFrame ===\n");

    // Create test CSV file
    create_test_csv(100000);

    // Read DataFrame from CSV file
    let start = Instant::now();
    let df = OptimizedDataFrame::from_csv("test_data.csv", true).unwrap();
    let duration = start.elapsed();
    println!("CSV Read Time: {duration:?}");
    println!(
        "Row Count: {}, Column Count: {}",
        df.row_count(),
        df.column_count()
    );

    // Test melt operation
    let start = Instant::now();
    let melted_df = df
        .melt(
            &["id"],
            Some(&["name", "age", "score"]),
            Some("variable"),
            Some("value"),
        )
        .unwrap();
    let duration = start.elapsed();
    println!("\nMelt Operation Time: {duration:?}");
    println!(
        "Post-Transformation Row Count: {}, Column Count: {}",
        melted_df.row_count(),
        melted_df.column_count()
    );

    // Test apply operation
    let start = Instant::now();
    let applied_df = df
        .apply(
            |col| {
                if col.column_type() == pandrs::column::ColumnType::String {
                    // Convert all string columns to uppercase
                    if let Some(str_col) = col.as_string() {
                        let mut new_data = Vec::with_capacity(str_col.len());
                        for i in 0..str_col.len() {
                            if let Ok(Some(val)) = str_col.get(i) {
                                new_data.push(val.to_uppercase());
                            } else {
                                new_data.push(String::new());
                            }
                        }
                        Ok(Column::String(StringColumn::new(new_data)))
                    } else {
                        // Handle type mismatch (unlikely)
                        Ok(col.column().clone())
                    }
                } else {
                    // Leave other columns unchanged
                    Ok(col.column().clone())
                }
            },
            Some(&["name"]),
        )
        .unwrap();
    let duration = start.elapsed();
    println!("\nApply Operation Time: {duration:?}");
    println!(
        "Post-Processing Row Count: {}, Column Count: {}",
        applied_df.row_count(),
        applied_df.column_count()
    );

    // Test CSV write operation
    let start = Instant::now();
    applied_df.to_csv("test_output.csv", true).unwrap();
    let duration = start.elapsed();
    println!("\nCSV Write Time: {duration:?}");

    println!("\n=== Test of Optimized DataFrame Features Complete ===");
}

// Function to create test CSV file
fn create_test_csv(rows: usize) {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    println!("Creating Test CSV File ({rows} rows)...");

    let file = File::create("test_data.csv").unwrap();
    let mut writer = BufWriter::new(file);

    // Header
    writeln!(writer, "id,name,age,score,category").unwrap();

    // Generate data
    let categories = ["A", "B", "C", "D", "E"];
    let names = [
        "Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Hannah", "Ian", "Julia",
    ];

    for i in 0..rows {
        let name = names[i % names.len()];
        let age = 20 + (i % 50);
        let score = (i % 100) as f64 / 10.0;
        let category = categories[i % categories.len()];

        writeln!(writer, "{i},{name},{age},{score},{category}").unwrap();
    }

    println!("Test CSV File Creation Complete");
}
