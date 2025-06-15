use pandrs::error::Result;

#[cfg(feature = "sql")]
use pandrs::{DataFrame, Series};

#[cfg(feature = "sql")]
use pandrs::io::{
    execute_sql, get_create_table_sql, get_table_schema, has_table, list_tables, read_sql,
    read_sql_advanced, read_sql_table, write_sql_advanced, write_to_sql, DatabaseConnection,
    InsertMethod, SqlConnection, SqlReadOptions, SqlValue, SqlWriteOptions, WriteMode,
};

#[cfg(feature = "sql")]
#[allow(unused_imports)]
use std::collections::HashMap;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    #[cfg(not(feature = "sql"))]
    {
        println!("SQL feature is not enabled. Enable it with --features sql");
        Ok(())
    }

    #[cfg(feature = "sql")]
    {
        println!("=== Advanced SQL Database Integration Example ===");

        // Create sample DataFrame for demonstration
        let sales_data = create_sample_dataframe()?;
        println!(
            "Created sample DataFrame with {} rows",
            sales_data.row_count()
        );

        println!("\n1. Database Connection Management:");

        // Demo: Connection from URL
        println!("--- Connection Setup ---");
        println!("API: SqlConnection::from_url(\"sqlite:sales.db\")");
        println!("Supported: SQLite, PostgreSQL, MySQL (with sqlx feature)");

        println!("\n2. Basic SQL Operations (Enhanced):");

        // Demo: Basic reading with new connection system
        println!("--- Enhanced Reading ---");
        println!("let conn = SqlConnection::from_url(\"sqlite:sales.db\").unwrap();");
        println!("let df = read_sql(\"SELECT * FROM sales WHERE revenue > 500\", &conn).unwrap();");
        println!("Result: DataFrame with filtered data using new connection system");

        // Demo: Basic writing with new options
        println!("\n--- Enhanced Writing ---");
        println!("let conn = SqlConnection::from_url(\"sqlite:sales.db\").unwrap();");
        println!("write_to_sql(&df, \"sales_backup\", &conn, \"replace\").unwrap();");
        println!("Result: Table created/replaced using enhanced connection system");

        println!("\n3. Advanced Reading Features:");

        // Demo: Chunked reading
        println!("--- Chunked Reading for Large Datasets ---");
        println!("let options = SqlReadOptions {{");
        println!("    chunksize: Some(1000),");
        println!("    coerce_float: true,");
        println!("    ..Default::default()");
        println!("}};");
        println!(
            "let df = read_sql_advanced(\"SELECT * FROM large_table\", &conn, options).unwrap();"
        );
        println!("Result: Memory-efficient reading of large datasets");

        // Demo: Parameterized queries
        println!("\n--- Parameterized Queries (Security) ---");
        println!(
            "let params = vec![SqlValue::Real(500.0), SqlValue::Text(\"Widget A\".to_string())];"
        );
        println!("let options = SqlReadOptions {{");
        println!("    params: Some(params),");
        println!("    ..Default::default()");
        println!("}};");
        println!("let sql = \"SELECT * FROM sales WHERE revenue > ? AND product = ?\";");
        println!("let df = read_sql_advanced(sql, &conn, options).unwrap();");
        println!("Result: Safe, SQL injection-proof queries");

        // Demo: Date parsing
        println!("\n--- Automatic Date Parsing ---");
        println!("let options = SqlReadOptions {{");
        println!(
            "    parse_dates: Some(vec![\"order_date\".to_string(), \"ship_date\".to_string()]),"
        );
        println!("    ..Default::default()");
        println!("}};");
        println!("Result: Automatic conversion of date columns");

        // Demo: Explicit data types
        println!("\n--- Explicit Data Type Control ---");
        println!("let mut dtype_map = HashMap::new();");
        println!("dtype_map.insert(\"customer_id\".to_string(), \"int64\".to_string());");
        println!("dtype_map.insert(\"revenue\".to_string(), \"float64\".to_string());");
        println!("let options = SqlReadOptions {{");
        println!("    dtype: Some(dtype_map),");
        println!("    ..Default::default()");
        println!("}};");
        println!("Result: Precise control over column data types");

        println!("\n4. Advanced Writing Features:");

        // Demo: Advanced writing options
        println!("--- High-Performance Writing ---");
        println!("let options = SqlWriteOptions {{");
        println!("    if_exists: WriteMode::Replace,");
        println!("    chunksize: Some(5000),");
        println!("    index: false,");
        println!("    method: InsertMethod::Multi,");
        println!("    ..Default::default()");
        println!("}};");
        println!("let rows_written = write_sql_advanced(&df, \"sales_optimized\", &conn, options).unwrap();");
        println!("Result: Optimized batch insertion with chunking");

        // Demo: Index control
        println!("\n--- Index and Schema Control ---");
        println!("let mut dtype_map = HashMap::new();");
        println!("dtype_map.insert(\"id\".to_string(), \"INTEGER PRIMARY KEY\".to_string());");
        println!("let options = SqlWriteOptions {{");
        println!("    index: true,");
        println!("    index_label: Some(\"row_id\".to_string()),");
        println!("    dtype: Some(dtype_map),");
        println!("    ..Default::default()");
        println!("}};");
        println!("Result: Custom index columns and data type specifications");

        println!("\n5. Database Schema Inspection:");

        // Demo: Table listing
        println!("--- Database Exploration ---");
        println!("let tables = list_tables(&conn, None).unwrap();");
        println!("for table in tables {{");
        println!("    println!(\"Table: {{}}\", table);");
        println!("}}");
        println!("Result: Complete list of tables in database");

        // Demo: Table existence checking
        println!("\n--- Table Validation ---");
        println!("if has_table(\"sales\", &conn, None).unwrap() {{");
        println!("    println!(\"Sales table exists\");");
        println!("}} else {{");
        println!("    println!(\"Sales table not found\");");
        println!("}}");
        println!("Result: Safe table existence checking");

        // Demo: Schema inspection
        println!("\n--- Schema Analysis ---");
        println!("let schema = get_table_schema(\"sales\", &conn, None).unwrap();");
        println!("println!(\"Table {{}} has {{}} columns\", schema.name, schema.columns.len());");
        println!("for col in schema.columns {{");
        println!("    println!(\"  Column {{}}: {{}} (nullable: {{}})\", col.name, col.data_type, col.nullable);");
        println!("}}");
        println!("Result: Detailed column information and constraints");

        // Demo: CREATE TABLE generation
        println!("\n--- SQL Generation ---");
        println!("let create_sql = get_create_table_sql(&df, \"new_table\", &conn).unwrap();");
        println!("println!(\"SQL: {{}}\", create_sql);");
        println!("Result: Automatic CREATE TABLE statement generation");

        println!("\n6. Performance Optimizations:");

        // Demo: Performance patterns
        demonstrate_performance_patterns();

        println!("\n7. Multi-Database Support:");

        // Demo: Different database engines
        demonstrate_database_engines();

        println!("\n=== Advanced Database Integration Complete ===");
        println!("\nNew capabilities added:");
        println!("✓ URL-based connection management");
        println!("✓ Chunked reading for memory efficiency");
        println!("✓ Parameterized queries for security");
        println!("✓ Automatic date/type parsing");
        println!("✓ Advanced insertion methods");
        println!("✓ Schema inspection and metadata");
        println!("✓ Table existence and validation");
        println!("✓ SQL generation utilities");
        println!("✓ Multi-database engine support");
        println!("✓ Connection pooling (with sqlx)");
        println!("✓ Transaction management");
        println!("✓ Performance optimizations");

        Ok(())
    }
}

/// Create sample DataFrame for demonstration
#[cfg(feature = "sql")]
#[allow(dead_code)]
fn create_sample_dataframe() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    let dates = vec![
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ];
    let products = vec!["Widget A", "Widget B", "Widget C", "Widget A", "Widget B"];
    let quantities = vec![10, 25, 15, 30, 20];
    let revenues = vec![199.90, 749.75, 599.85, 599.70, 599.80];
    let customers = vec![
        "Acme Corp",
        "Beta LLC",
        "Gamma Inc",
        "Acme Corp",
        "Delta Co",
    ];

    let date_series = Series::new(
        dates.into_iter().map(|s| s.to_string()).collect(),
        Some("Date".to_string()),
    )?;
    let product_series = Series::new(
        products.into_iter().map(|s| s.to_string()).collect(),
        Some("Product".to_string()),
    )?;
    let quantity_series = Series::new(quantities, Some("Quantity".to_string()))?;
    let revenue_series = Series::new(revenues, Some("Revenue".to_string()))?;
    let customer_series = Series::new(
        customers.into_iter().map(|s| s.to_string()).collect(),
        Some("Customer".to_string()),
    )?;

    df.add_column("Date".to_string(), date_series)?;
    df.add_column("Product".to_string(), product_series)?;
    df.add_column("Quantity".to_string(), quantity_series.to_string_series()?)?;
    df.add_column("Revenue".to_string(), revenue_series.to_string_series()?)?;
    df.add_column("Customer".to_string(), customer_series)?;

    Ok(df)
}

/// Demonstrate performance optimization patterns
#[allow(dead_code)]
fn demonstrate_performance_patterns() {
    println!("\n--- Performance Optimization Patterns ---");

    println!("\n1. Bulk Loading:");
    println!("   let options = SqlWriteOptions {{");
    println!("       chunksize: Some(10000),");
    println!("       method: InsertMethod::Multi,");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   // Process 10,000 rows at a time for optimal memory usage");

    println!("\n2. Memory-Efficient Reading:");
    println!("   let options = SqlReadOptions {{");
    println!("       chunksize: Some(5000),");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   // Read large tables in manageable chunks");

    println!("\n3. Selective Column Loading:");
    println!("   let sql = \"SELECT id, name, revenue FROM large_table\";");
    println!("   // Only load required columns to reduce memory usage");

    println!("\n4. Indexed Queries:");
    println!("   let sql = \"SELECT * FROM sales WHERE customer_id = ? ORDER BY date\";");
    println!("   // Use parameterized queries with proper indexes");

    println!("\n5. Transaction Batching:");
    println!("   let options = SqlWriteOptions {{");
    println!("       chunksize: Some(1000),");
    println!("       method: InsertMethod::Multi,");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   // Batch operations in transactions for better performance");
}

/// Demonstrate different database engine support
#[allow(dead_code)]
fn demonstrate_database_engines() {
    println!("\n--- Multi-Database Engine Support ---");

    println!("\n1. SQLite (Built-in):");
    println!("   let conn = SqlConnection::from_url(\"sqlite:data.db\").unwrap();");
    println!("   // File-based database, no server required");
    println!("   // Perfect for development, testing, and embedded applications");

    println!("\n2. PostgreSQL (with sqlx feature):");
    println!("   let conn = SqlConnection::from_url(");
    println!("       \"postgresql://user:pass@localhost:5432/dbname\"");
    println!("   ).unwrap();");
    println!("   // Full-featured RDBMS with advanced SQL support");
    println!("   // Excellent for production applications");

    println!("\n3. MySQL (with sqlx feature):");
    println!("   let conn = SqlConnection::from_url(");
    println!("       \"mysql://user:pass@localhost:3306/dbname\"");
    println!("   ).unwrap();");
    println!("   // Widely-used RDBMS with good performance");
    println!("   // Popular for web applications");

    println!("\n4. Connection Pooling (Async with sqlx):");
    println!("   use tokio;");
    println!("   let config = PoolConfig {{");
    println!("       max_connections: 20,");
    println!("       connect_timeout: Duration::from_secs(30),");
    println!("       ..Default::default()");
    println!("   }};");
    println!("   let conn = SqlConnection::with_pool(url, config).await.unwrap();");
    println!("   // High-performance connection pooling for production");

    println!("\n5. Database-Specific Optimizations:");
    println!("   // PostgreSQL: COPY for bulk loading");
    println!("   // MySQL: LOAD DATA INFILE for large datasets");
    println!("   // SQLite: WAL mode for concurrent access");
    println!("   // Automatic dialect-specific SQL generation");
}
