//! Comprehensive integration tests for PandRS alpha.4 release
//!
//! This test suite validates that all major alpha.4 features work correctly
//! both individually and in combination with each other.

use std::collections::HashMap;
use std::fs::remove_file;
use std::path::Path;

use pandrs::dataframe::DataFrame;
use pandrs::error::Result;
use pandrs::{BooleanColumn, Column, Float64Column, Int64Column, OptimizedDataFrame, StringColumn};

#[cfg(feature = "distributed")]
use pandrs::distributed::DistributedContext;

#[cfg(feature = "parquet")]
use pandrs::io::parquet::{read_parquet, write_parquet, ParquetCompression};

#[cfg(feature = "sql")]
use pandrs::io::sql::{read_sql, write_to_sql};

/// Test the new DataFrame operations introduced in alpha.4
#[test]
#[allow(clippy::result_large_err)]
fn test_alpha4_dataframe_operations() -> Result<()> {
    // Create a test DataFrame
    let mut df = DataFrame::new();

    // Add some test data
    df.add_column(
        "name".to_string(),
        pandrs::series::Series::new(
            vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()],
            Some("name".to_string()),
        )?,
    )?;

    df.add_column(
        "age".to_string(),
        pandrs::series::Series::new(vec![25, 30, 35], Some("age".to_string()))?,
    )?;

    df.add_column(
        "salary".to_string(),
        pandrs::series::Series::new(vec![50000, 60000, 70000], Some("salary".to_string()))?,
    )?;

    // Test initial state
    assert_eq!(df.column_names(), vec!["name", "age", "salary"]);
    assert_eq!(df.row_count(), 3);

    // Test rename_columns (alpha.4 feature)
    let mut rename_map = HashMap::new();
    rename_map.insert("name".to_string(), "employee_name".to_string());
    rename_map.insert("age".to_string(), "employee_age".to_string());

    df.rename_columns(&rename_map)?;

    // Verify rename worked
    assert_eq!(
        df.column_names(),
        vec!["employee_name", "employee_age", "salary"]
    );
    assert!(df.contains_column("employee_name"));
    assert!(df.contains_column("employee_age"));
    assert!(!df.contains_column("name")); // Old name should be gone
    assert!(!df.contains_column("age")); // Old name should be gone

    // Test set_column_names (alpha.4 feature)
    let new_names = vec![
        "worker_name".to_string(),
        "worker_age".to_string(),
        "worker_salary".to_string(),
    ];

    df.set_column_names(new_names)?;

    // Verify set_column_names worked
    assert_eq!(
        df.column_names(),
        vec!["worker_name", "worker_age", "worker_salary"]
    );

    // Test data integrity after operations
    let name_data = df.get_column_string_values("worker_name")?;
    assert_eq!(name_data, vec!["Alice", "Bob", "Carol"]);

    let age_data = df.get_column_string_values("worker_age")?;
    assert_eq!(age_data, vec!["25", "30", "35"]);

    Ok(())
}

/// Test OptimizedDataFrame with alpha.4 features
#[test]
#[allow(clippy::result_large_err)]
fn test_alpha4_optimized_dataframe_operations() -> Result<()> {
    let mut df = OptimizedDataFrame::new();

    // Add test data
    let names = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Carol".to_string(),
    ]);

    let ages = Int64Column::new(vec![25, 30, 35]);
    let salaries = Float64Column::new(vec![50000.0, 60000.0, 70000.0]);
    let active = BooleanColumn::new(vec![true, false, true]);

    df.add_column("name", Column::String(names))?;
    df.add_column("age", Column::Int64(ages))?;
    df.add_column("salary", Column::Float64(salaries))?;
    df.add_column("active", Column::Boolean(active))?;

    // Test initial state
    assert_eq!(df.column_count(), 4);
    assert_eq!(df.row_count(), 3);

    // Test rename_columns
    let mut rename_map = HashMap::new();
    rename_map.insert("name".to_string(), "employee_name".to_string());
    rename_map.insert("active".to_string(), "is_active".to_string());

    df.rename_columns(&rename_map)?;

    // Verify renames
    assert!(df.contains_column("employee_name"));
    assert!(df.contains_column("is_active"));
    assert!(!df.contains_column("name"));
    assert!(!df.contains_column("active"));

    // Test set_column_names
    let new_names = vec![
        "emp_name".to_string(),
        "emp_age".to_string(),
        "emp_salary".to_string(),
        "emp_active".to_string(),
    ];

    df.set_column_names(new_names)?;

    // Verify all names changed
    assert_eq!(
        df.column_names(),
        vec!["emp_name", "emp_age", "emp_salary", "emp_active"]
    );

    // Test data integrity
    let name_col = df.column("emp_name")?;
    assert!(name_col.as_string().is_some());

    let age_col = df.column("emp_age")?;
    assert!(age_col.as_int64().is_some());

    let salary_col = df.column("emp_salary")?;
    assert!(salary_col.as_float64().is_some());

    let active_col = df.column("emp_active")?;
    assert!(active_col.as_boolean().is_some());

    Ok(())
}

/// Test enhanced Parquet I/O with real data (alpha.4 feature)
#[cfg(feature = "parquet")]
#[test]
fn test_alpha4_enhanced_parquet_io() -> Result<()> {
    let test_file = "test_alpha4_parquet.parquet";

    // Clean up any existing test file
    if Path::new(test_file).exists() {
        let _ = remove_file(test_file);
    }

    // Create test data with multiple data types
    let mut df = OptimizedDataFrame::new();

    let names = StringColumn::new(vec![
        "Product_A".to_string(),
        "Product_B".to_string(),
        "Product_C".to_string(),
        "Product_D".to_string(),
    ]);

    let quantities = Int64Column::new(vec![100, 250, 75, 300]);
    let prices = Float64Column::new(vec![19.99, 49.99, 9.99, 99.99]);
    let in_stock = BooleanColumn::new(vec![true, false, true, true]);

    df.add_column("product_name", Column::String(names))?;
    df.add_column("quantity", Column::Int64(quantities))?;
    df.add_column("price", Column::Float64(prices))?;
    df.add_column("in_stock", Column::Boolean(in_stock))?;

    // Test writing with different compression options

    // Write with Snappy compression
    write_parquet(&df, test_file, Some(ParquetCompression::Snappy))?;
    assert!(Path::new(test_file).exists());

    // Read back the data
    let loaded_df = read_parquet(test_file)?;

    // Verify data integrity
    assert_eq!(loaded_df.row_count(), 4);
    assert_eq!(loaded_df.column_names().len(), 4);
    assert!(loaded_df.contains_column("product_name"));
    assert!(loaded_df.contains_column("quantity"));
    assert!(loaded_df.contains_column("price"));
    assert!(loaded_df.contains_column("in_stock"));

    // Test data values (convert to string for comparison)
    let product_data = loaded_df.get_column_string_values("product_name")?;
    assert!(product_data.contains(&"Product_A".to_string()));
    assert!(product_data.contains(&"Product_B".to_string()));

    // Clean up
    let _ = remove_file(test_file);

    // Test with Gzip compression
    let gzip_file = "test_alpha4_gzip.parquet";
    write_parquet(&df, gzip_file, Some(ParquetCompression::Gzip))?;
    assert!(Path::new(gzip_file).exists());

    let gzip_loaded_df = read_parquet(gzip_file)?;
    assert_eq!(gzip_loaded_df.row_count(), 4);

    // Clean up
    let _ = remove_file(gzip_file);

    Ok(())
}

/// Test enhanced SQL I/O with real data (alpha.4 feature)
#[cfg(feature = "sql")]
#[test]
fn test_alpha4_enhanced_sql_io() -> Result<()> {
    let db_file = "test_alpha4.db";

    // Clean up any existing test file
    if Path::new(db_file).exists() {
        let _ = remove_file(db_file);
    }

    // Create test data
    let mut df = OptimizedDataFrame::new();

    let customer_ids = Int64Column::new(vec![1, 2, 3, 4]);
    let customer_names = StringColumn::new(vec![
        "John Doe".to_string(),
        "Jane Smith".to_string(),
        "Bob Johnson".to_string(),
        "Alice Brown".to_string(),
    ]);
    let order_amounts = Int64Column::new(vec![150, 200, 75, 300]);

    df.add_column("customer_id", Column::Int64(customer_ids))?;
    df.add_column("customer_name", Column::String(customer_names))?;
    df.add_column("order_amount", Column::Int64(order_amounts))?;

    // Test write to SQL with different modes

    // Write initial data
    write_to_sql(&df, "customers", db_file, "replace")?;

    // Verify file was created
    assert!(Path::new(db_file).exists());

    // Read back data
    let loaded_df = read_sql("SELECT * FROM customers ORDER BY customer_id", db_file)?;

    // Verify data integrity
    assert_eq!(loaded_df.row_count(), 4);
    assert!(loaded_df.contains_column("customer_id"));
    assert!(loaded_df.contains_column("customer_name"));
    assert!(loaded_df.contains_column("order_amount"));

    // Test specific query results
    let specific_df = read_sql(
        "SELECT customer_name, order_amount FROM customers WHERE order_amount > 100",
        db_file,
    )?;

    assert_eq!(specific_df.row_count(), 3); // 3 customers with orders > 100

    // Test aggregation query
    let agg_df = read_sql(
        "SELECT COUNT(*) as total_customers, AVG(CAST(order_amount AS REAL)) as avg_amount FROM customers",
        db_file
    )?;

    assert_eq!(agg_df.row_count(), 1);
    assert!(agg_df.contains_column("total_customers"));
    assert!(agg_df.contains_column("avg_amount"));

    // Clean up
    let _ = remove_file(db_file);

    Ok(())
}

/// Test distributed processing integration (alpha.4 feature)
#[cfg(feature = "distributed")]
#[test]
fn test_alpha4_distributed_processing_integration() -> Result<()> {
    // Create test data
    let mut df = DataFrame::new();

    df.add_column(
        "region".to_string(),
        pandrs::series::Series::new(
            vec![
                "North".to_string(),
                "South".to_string(),
                "East".to_string(),
                "West".to_string(),
                "North".to_string(),
                "South".to_string(),
            ],
            Some("region".to_string()),
        )?,
    )?;

    df.add_column(
        "sales".to_string(),
        pandrs::series::Series::new(
            vec![1000, 1500, 800, 1200, 900, 1100],
            Some("sales".to_string()),
        )?,
    )?;

    df.add_column(
        "quarter".to_string(),
        pandrs::series::Series::new(vec![1, 1, 1, 1, 2, 2], Some("quarter".to_string()))?,
    )?;

    // Create distributed context
    let mut context = DistributedContext::new_local(2)?;
    context.register_dataframe("sales_data", &df)?;

    // Test distributed operations
    let sales_df = context.dataset("sales_data")?;

    // Test basic distributed select
    let selected = sales_df.select(&["region", "sales"])?.collect()?;
    assert_eq!(selected.row_count(), 6);
    assert_eq!(selected.column_names().len(), 2);

    // Test distributed filtering
    let filtered = sales_df.filter("sales > 1000")?.collect()?;

    assert!(filtered.row_count() < df.row_count());

    // Test distributed aggregation
    let aggregated = sales_df
        .aggregate(&["region"], &[("sales", "sum", "total_sales")])?
        .collect()?;

    assert_eq!(aggregated.column_names().len(), 2); // region + total_sales
    assert!(aggregated.row_count() <= 4); // Max 4 unique regions

    Ok(())
}

/// Test cross-feature integration: DataFusion + Parquet + new DataFrame operations
#[cfg(all(feature = "distributed", feature = "parquet"))]
#[test]
fn test_alpha4_cross_feature_integration() -> Result<()> {
    let test_file = "test_cross_feature.parquet";

    // Clean up any existing test file
    if Path::new(test_file).exists() {
        let _ = remove_file(test_file);
    }

    // Step 1: Create data with OptimizedDataFrame and new operations
    let mut df = OptimizedDataFrame::new();

    let departments = StringColumn::new(vec![
        "Engineering".to_string(),
        "Sales".to_string(),
        "Marketing".to_string(),
        "Engineering".to_string(),
        "Sales".to_string(),
    ]);

    let employees = StringColumn::new(vec![
        "Alice".to_string(),
        "Bob".to_string(),
        "Carol".to_string(),
        "David".to_string(),
        "Eve".to_string(),
    ]);

    let salaries = Int64Column::new(vec![75000, 65000, 60000, 80000, 70000]);

    df.add_column("dept", Column::String(departments))?;
    df.add_column("emp", Column::String(employees))?;
    df.add_column("sal", Column::Int64(salaries))?;

    // Step 2: Use new DataFrame operations
    let mut rename_map = HashMap::new();
    rename_map.insert("dept".to_string(), "department".to_string());
    rename_map.insert("emp".to_string(), "employee".to_string());
    rename_map.insert("sal".to_string(), "salary".to_string());

    df.rename_columns(&rename_map)?;

    // Step 3: Write to Parquet with compression
    write_parquet(&df, test_file, Some(ParquetCompression::Snappy))?;

    // Step 4: Read back using regular DataFrame
    let loaded_df = read_parquet(test_file)?;

    // Step 5: Use distributed processing on the loaded data
    let mut context = DistributedContext::new_local(2)?;
    context.register_dataframe("employee_data", &loaded_df)?;

    let emp_df = context.dataset("employee_data")?;

    // Test complex distributed query on Parquet-loaded data
    let high_earners = emp_df
        .filter("salary > 65000")?
        .select(&["department", "employee", "salary"])?
        .collect()?;

    // Verify results
    assert!(high_earners.row_count() >= 3); // Should have at least 3 high earners
    assert_eq!(high_earners.column_names().len(), 3);

    // Test aggregation on Parquet-loaded data
    let dept_summary = emp_df
        .aggregate(
            &["department"],
            &[
                ("salary", "avg", "avg_salary"),
                ("salary", "count", "emp_count"),
            ],
        )?
        .collect()?;

    assert!(dept_summary.row_count() >= 2); // At least 2 departments
    assert_eq!(dept_summary.column_names().len(), 3); // department + avg_salary + emp_count

    // Clean up
    let _ = remove_file(test_file);

    Ok(())
}

/// Test error handling and edge cases in alpha.4 features
#[test]
#[allow(clippy::result_large_err)]
fn test_alpha4_error_handling() -> Result<()> {
    let mut df = DataFrame::new();

    // Add test data
    df.add_column(
        "col1".to_string(),
        pandrs::series::Series::new(vec![1, 2, 3], Some("col1".to_string()))?,
    )?;

    // Test rename_columns with non-existent column
    let mut bad_rename_map = HashMap::new();
    bad_rename_map.insert("nonexistent".to_string(), "new_name".to_string());

    let result = df.rename_columns(&bad_rename_map);
    assert!(result.is_err()); // Should fail

    // Test set_column_names with wrong number of names
    let bad_names = vec!["name1".to_string(), "name2".to_string()]; // Only 2 names for 1 column
    let result = df.set_column_names(bad_names);
    assert!(result.is_err()); // Should fail

    // Test empty rename map (should succeed)
    let empty_map = HashMap::new();
    let result = df.rename_columns(&empty_map);
    assert!(result.is_ok()); // Should succeed with no changes

    Ok(())
}

/// Test performance characteristics of alpha.4 features
#[test]
#[allow(clippy::result_large_err)]
fn test_alpha4_performance_characteristics() -> Result<()> {
    // Create a larger dataset for performance testing
    let size = 1000;

    let mut df = OptimizedDataFrame::new();

    // Generate test data
    let mut ids = Vec::with_capacity(size);
    let mut values = Vec::with_capacity(size);
    let mut categories = Vec::with_capacity(size);
    let mut flags = Vec::with_capacity(size);

    for i in 0..size {
        ids.push(i as i64);
        values.push((i as f64) * 1.5);
        categories.push(format!("Category_{}", i % 10));
        flags.push(i % 2 == 0);
    }

    let id_col = Int64Column::new(ids);
    let value_col = Float64Column::new(values);
    let cat_col = StringColumn::new(categories);
    let flag_col = BooleanColumn::new(flags);

    df.add_column("id", Column::Int64(id_col))?;
    df.add_column("value", Column::Float64(value_col))?;
    df.add_column("category", Column::String(cat_col))?;
    df.add_column("flag", Column::Boolean(flag_col))?;

    // Test rename performance
    let start = std::time::Instant::now();

    let mut rename_map = HashMap::new();
    rename_map.insert("id".to_string(), "identifier".to_string());
    rename_map.insert("value".to_string(), "metric".to_string());
    rename_map.insert("category".to_string(), "group".to_string());
    rename_map.insert("flag".to_string(), "active".to_string());

    df.rename_columns(&rename_map)?;

    let rename_duration = start.elapsed();

    // Should be reasonably fast (under 100ms for 1000 rows)
    assert!(rename_duration.as_millis() < 100);

    // Test set_column_names performance
    let start = std::time::Instant::now();

    let new_names = vec![
        "id_new".to_string(),
        "value_new".to_string(),
        "category_new".to_string(),
        "flag_new".to_string(),
    ];

    df.set_column_names(new_names)?;

    let set_names_duration = start.elapsed();

    // Should also be reasonably fast
    assert!(set_names_duration.as_millis() < 100);

    // Verify data integrity after operations
    assert_eq!(df.row_count(), size);
    assert_eq!(df.column_count(), 4);

    Ok(())
}
