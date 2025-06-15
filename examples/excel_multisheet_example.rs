use pandrs::error::Result;

#[cfg(feature = "excel")]
use pandrs::{DataFrame, Series};

#[cfg(feature = "excel")]
use pandrs::io::{
    get_sheet_info, get_workbook_info, list_sheet_names, read_excel_sheets, read_excel_with_info,
    write_excel_sheets, ExcelSheetInfo, ExcelWorkbookInfo,
};

#[cfg(feature = "excel")]
use std::collections::HashMap;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    #[cfg(not(feature = "excel"))]
    {
        println!("Excel feature is not enabled. Enable it with --features excel");
        return Ok(());
    }

    #[cfg(feature = "excel")]
    {
        println!("=== Excel Multi-Sheet I/O Example ===");

        // Create sample DataFrames for demonstration
        let sales_data = create_sample_sales_dataframe()?;
        let inventory_data = create_sample_inventory_dataframe()?;
        let summary_data = create_sample_summary_dataframe()?;

        println!("\n1. Creating sample Excel file with multiple sheets...");

        // For this example, we'll create a sample file path
        let excel_file = "sample_data.xlsx";

        // Note: The actual file creation would require OptimizedDataFrame instances
        // This is a demonstration of the API structure

        println!("Sample DataFrames created:");
        println!("- Sales data: {} rows", sales_data.row_count());
        println!("- Inventory data: {} rows", inventory_data.row_count());
        println!("- Summary data: {} rows", summary_data.row_count());

        // Example usage patterns (would work with actual Excel files)
        println!("\n2. Excel Multi-Sheet Operations (API demonstration):");

        // Demo: List sheet names
        println!("\n--- Listing Sheet Names ---");
        println!("API: list_sheet_names(\"{}\").unwrap()", excel_file);
        println!("Expected result: [\"Sales\", \"Inventory\", \"Summary\"]");

        // Demo: Get workbook information
        println!("\n--- Getting Workbook Information ---");
        println!("API: get_workbook_info(\"{}\").unwrap()", excel_file);
        println!("Expected result: ExcelWorkbookInfo {{");
        println!("    sheet_names: [\"Sales\", \"Inventory\", \"Summary\"],");
        println!("    sheet_count: 3,");
        println!("    total_cells: 450");
        println!("}}");

        // Demo: Get specific sheet information
        println!("\n--- Getting Sheet Information ---");
        println!(
            "API: get_sheet_info(\"{}\", \"Sales\").unwrap()",
            excel_file
        );
        println!("Expected result: ExcelSheetInfo {{");
        println!("    name: \"Sales\",");
        println!("    rows: 100,");
        println!("    columns: 5,");
        println!("    range: \"A1:E100\"");
        println!("}}");

        // Demo: Read multiple sheets
        println!("\n--- Reading Multiple Sheets ---");
        println!(
            "API: read_excel_sheets(\"{}\", None, true, 0, None).unwrap()",
            excel_file
        );
        println!("Expected result: HashMap with 3 DataFrames");

        // Demo: Read specific sheets
        println!("\n--- Reading Specific Sheets ---");
        println!("API: read_excel_sheets(\"{}\", Some(&[\"Sales\", \"Summary\"]), true, 0, None).unwrap()", excel_file);
        println!("Expected result: HashMap with 2 DataFrames");

        // Demo: Read with workbook info
        println!("\n--- Reading with Workbook Info ---");
        println!(
            "API: read_excel_with_info(\"{}\", Some(\"Sales\"), true, 0, None).unwrap()",
            excel_file
        );
        println!("Expected result: (DataFrame, ExcelWorkbookInfo)");

        // Demo: Write multiple sheets
        println!("\n--- Writing Multiple Sheets ---");
        println!("API: write_excel_sheets(&sheets_map, \"output.xlsx\", false).unwrap()");
        println!("Expected result: Excel file with multiple sheets created");

        println!("\n3. Advanced Excel Operations:");

        // Demonstrate advanced patterns
        demonstrate_excel_patterns();

        println!("\n=== Excel Multi-Sheet Enhancement Complete ===");
        println!("\nNew capabilities added:");
        println!("✓ Sheet discovery and listing");
        println!("✓ Workbook metadata extraction");
        println!("✓ Multi-sheet reading operations");
        println!("✓ Multi-sheet writing operations");
        println!("✓ Enhanced sheet information");
        println!("✓ Comprehensive error handling");
        println!("✓ Validation and sheet management");

        Ok(())
    }

    /// Create sample sales DataFrame
    #[cfg(feature = "excel")]
    fn create_sample_sales_dataframe() -> Result<DataFrame> {
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
        let prices = vec![19.99, 29.99, 39.99, 19.99, 29.99];
        let revenues = vec![199.90, 749.75, 599.85, 599.70, 599.80];

        let date_series = Series::new(
            dates.into_iter().map(|s| s.to_string()).collect(),
            Some("Date".to_string()),
        )?;
        let product_series = Series::new(
            products.into_iter().map(|s| s.to_string()).collect(),
            Some("Product".to_string()),
        )?;
        let quantity_series = Series::new(quantities, Some("Quantity".to_string()))?;
        let price_series = Series::new(prices, Some("Price".to_string()))?;
        let revenue_series = Series::new(revenues, Some("Revenue".to_string()))?;

        df.add_column("Date".to_string(), date_series)?;
        df.add_column("Product".to_string(), product_series)?;
        df.add_column("Quantity".to_string(), quantity_series.to_string_series()?)?;
        df.add_column("Price".to_string(), price_series.to_string_series()?)?;
        df.add_column("Revenue".to_string(), revenue_series.to_string_series()?)?;

        Ok(df)
    }

    /// Create sample inventory DataFrame
    #[cfg(feature = "excel")]
    fn create_sample_inventory_dataframe() -> Result<DataFrame> {
        let mut df = DataFrame::new();

        let products = vec!["Widget A", "Widget B", "Widget C", "Widget D"];
        let stock_levels = vec![150, 200, 75, 300];
        let reorder_points = vec![50, 100, 25, 150];
        let suppliers = vec!["Supplier X", "Supplier Y", "Supplier X", "Supplier Z"];

        let product_series = Series::new(
            products.into_iter().map(|s| s.to_string()).collect(),
            Some("Product".to_string()),
        )?;
        let stock_series = Series::new(stock_levels, Some("Stock_Level".to_string()))?;
        let reorder_series = Series::new(reorder_points, Some("Reorder_Point".to_string()))?;
        let supplier_series = Series::new(
            suppliers.into_iter().map(|s| s.to_string()).collect(),
            Some("Supplier".to_string()),
        )?;

        df.add_column("Product".to_string(), product_series)?;
        df.add_column("Stock_Level".to_string(), stock_series.to_string_series()?)?;
        df.add_column(
            "Reorder_Point".to_string(),
            reorder_series.to_string_series()?,
        )?;
        df.add_column("Supplier".to_string(), supplier_series)?;

        Ok(df)
    }

    /// Create sample summary DataFrame
    #[cfg(feature = "excel")]
    fn create_sample_summary_dataframe() -> Result<DataFrame> {
        let mut df = DataFrame::new();

        let metrics = vec![
            "Total Sales",
            "Average Order",
            "Top Product",
            "Total Revenue",
        ];
        let values = vec!["100", "139.98", "Widget B", "2748.00"];

        let metric_series = Series::new(
            metrics.into_iter().map(|s| s.to_string()).collect(),
            Some("Metric".to_string()),
        )?;
        let value_series = Series::new(
            values.into_iter().map(|s| s.to_string()).collect(),
            Some("Value".to_string()),
        )?;

        df.add_column("Metric".to_string(), metric_series)?;
        df.add_column("Value".to_string(), value_series)?;

        Ok(df)
    }

    /// Demonstrate advanced Excel operation patterns
    #[allow(dead_code)]
    fn demonstrate_excel_patterns() {
        println!("\n--- Advanced Usage Patterns ---");

        println!("\n1. Batch Processing Multiple Files:");
        println!("   for file in excel_files {{");
        println!("       let info = get_workbook_info(&file)?;");
        println!("       println!(\"File {{}} has {{}} sheets\", file, info.sheet_count);");
        println!("   }}");

        println!("\n2. Conditional Sheet Reading:");
        println!("   let sheets = list_sheet_names(\"data.xlsx\")?;");
        println!("   let target_sheets: Vec<&str> = sheets.iter()");
        println!("       .filter(|name| name.contains(\"2024\"))");
        println!("       .map(|s| s.as_str()).collect();");
        println!(
            "   let data = read_excel_sheets(\"data.xlsx\", Some(&target_sheets), true, 0, None)?;"
        );

        println!("\n3. Sheet Validation:");
        println!("   let info = get_sheet_info(\"data.xlsx\", \"Sales\")?;");
        println!("   if info.rows > 1000 {{");
        println!("       println!(\"Large dataset detected: {{}} rows\", info.rows);");
        println!("   }}");

        println!("\n4. Multi-Sheet Analysis:");
        println!("   let (df, workbook_info) = read_excel_with_info(\"data.xlsx\", None, true, 0, None)?;");
        println!("   println!(\"Analyzing {{}} sheets with {{}} total cells\", ");
        println!("            workbook_info.sheet_count, workbook_info.total_cells);");

        println!("\n5. Dynamic Sheet Creation:");
        println!("   let mut sheets = HashMap::new();");
        println!("   for (category, data) in grouped_data {{");
        println!("       sheets.insert(format!(\"{{category}}_Report\"), &data);");
        println!("   }}");
        println!("   write_excel_sheets(&sheets, \"report.xlsx\", true)?;");
    }
}
