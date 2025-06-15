use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::groupby::{AggFunc, ColumnAggBuilder, GroupByExt, NamedAgg};
use pandrs::error::Result;
use pandrs::series::base::Series;
use std::collections::HashMap;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Alpha 4 Enhanced GroupBy with Named Aggregations Example ===\n");

    // Create sample sales data
    println!("1. Creating Sample Sales Data:");
    let mut df = DataFrame::new();

    let regions = vec![
        "North", "South", "North", "East", "South", "West", "East", "North", "West", "South",
    ];
    let categories = vec!["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"];
    let sales = vec![
        "1000", "1500", "1200", "800", "1800", "1100", "900", "1300", "1400", "1600",
    ];
    let quantities = vec!["10", "15", "12", "8", "18", "11", "9", "13", "14", "16"];
    let costs = vec![
        "600", "900", "720", "480", "1080", "660", "540", "780", "840", "960",
    ];

    let region_series = Series::new(
        regions.into_iter().map(|s| s.to_string()).collect(),
        Some("Region".to_string()),
    )?;
    let category_series = Series::new(
        categories.into_iter().map(|s| s.to_string()).collect(),
        Some("Category".to_string()),
    )?;
    let sales_series = Series::new(
        sales.into_iter().map(|s| s.to_string()).collect(),
        Some("Sales".to_string()),
    )?;
    let quantity_series = Series::new(
        quantities.into_iter().map(|s| s.to_string()).collect(),
        Some("Quantity".to_string()),
    )?;
    let cost_series = Series::new(
        costs.into_iter().map(|s| s.to_string()).collect(),
        Some("Cost".to_string()),
    )?;

    df.add_column("Region".to_string(), region_series)?;
    df.add_column("Category".to_string(), category_series)?;
    df.add_column("Sales".to_string(), sales_series)?;
    df.add_column("Quantity".to_string(), quantity_series)?;
    df.add_column("Cost".to_string(), cost_series)?;

    println!("Original Data:");
    println!("{:?}", df);

    println!("\n=== Basic GroupBy Operations ===\n");

    // 2. Basic GroupBy operations
    println!("2. Basic GroupBy by Region:");
    let region_groupby = GroupByExt::groupby_single(&df, "Region")?;

    // Simple aggregations
    let sales_sum = region_groupby.sum("Sales")?;
    println!("Sales Sum by Region:");
    println!("{:?}", sales_sum);

    let sales_mean = region_groupby.mean("Sales")?;
    println!("\nSales Mean by Region:");
    println!("{:?}", sales_mean);

    let quantity_count = region_groupby.count("Quantity")?;
    println!("\nQuantity Count by Region:");
    println!("{:?}", quantity_count);

    println!("\n=== Named Aggregations ===\n");

    // 3. Named Aggregations (pandas-like)
    println!("3. Named Aggregations:");

    let named_aggs = vec![
        NamedAgg::new("Sales".to_string(), AggFunc::Sum, "Total_Sales".to_string()),
        NamedAgg::new("Sales".to_string(), AggFunc::Mean, "Avg_Sales".to_string()),
        NamedAgg::new(
            "Quantity".to_string(),
            AggFunc::Sum,
            "Total_Quantity".to_string(),
        ),
        NamedAgg::new("Cost".to_string(), AggFunc::Mean, "Avg_Cost".to_string()),
    ];

    let named_result = region_groupby.agg(named_aggs)?;
    println!("Named Aggregations Result:");
    println!("{:?}", named_result);

    println!("\n=== Column Aggregation Builder ===\n");

    // 4. Multiple aggregations per column using builder pattern
    println!("4. Multiple Aggregations per Column:");

    let sales_aggs = ColumnAggBuilder::new("Sales".to_string())
        .agg(AggFunc::Sum, "Sales_Total".to_string())
        .agg(AggFunc::Mean, "Sales_Average".to_string())
        .agg(AggFunc::Min, "Sales_Min".to_string())
        .agg(AggFunc::Max, "Sales_Max".to_string())
        .agg(AggFunc::Std, "Sales_StdDev".to_string());

    let quantity_aggs = ColumnAggBuilder::new("Quantity".to_string())
        .agg(AggFunc::Sum, "Qty_Total".to_string())
        .agg(AggFunc::Mean, "Qty_Average".to_string());

    let multi_result = region_groupby.agg_multi(vec![sales_aggs, quantity_aggs])?;
    println!("Multiple Aggregations Result:");
    println!("{:?}", multi_result);

    println!("\n=== Dictionary-Style Aggregations ===\n");

    // 5. Dictionary-style aggregations (pandas .agg({'col': ['func1', 'func2']}) equivalent)
    println!("5. Dictionary-Style Aggregations:");

    let mut agg_dict = HashMap::new();
    agg_dict.insert(
        "Sales".to_string(),
        vec![
            (AggFunc::Sum, "sales_sum".to_string()),
            (AggFunc::Mean, "sales_mean".to_string()),
            (AggFunc::Std, "sales_std".to_string()),
        ],
    );
    agg_dict.insert(
        "Cost".to_string(),
        vec![
            (AggFunc::Sum, "cost_total".to_string()),
            (AggFunc::Mean, "cost_avg".to_string()),
        ],
    );

    let dict_result = region_groupby.agg_dict(agg_dict)?;
    println!("Dictionary Aggregations Result:");
    println!("{:?}", dict_result);

    println!("\n=== Custom Aggregation Functions ===\n");

    // 6. Custom aggregation functions
    println!("6. Custom Aggregation Functions:");

    // Custom function: Range (max - min)
    let range_result = region_groupby.apply("Sales", "sales_range", |values| {
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        max_val - min_val
    })?;
    println!("Sales Range by Region:");
    println!("{:?}", range_result);

    // Custom function: Coefficient of variation
    let cv_result = region_groupby.apply("Sales", "sales_cv", |values| {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std_dev = variance.sqrt();
        std_dev / mean
    })?;
    println!("Sales Coefficient of Variation by Region:");
    println!("{:?}", cv_result);

    println!("\n=== Multi-Column GroupBy ===\n");

    // 7. Multi-column groupby
    println!("7. Multi-Column GroupBy (Region + Category):");

    let multi_groupby = GroupByExt::groupby(&df, &["Region", "Category"])?;

    let multi_aggs = vec![
        NamedAgg::new("Sales".to_string(), AggFunc::Sum, "Total_Sales".to_string()),
        NamedAgg::new(
            "Quantity".to_string(),
            AggFunc::Mean,
            "Avg_Quantity".to_string(),
        ),
    ];

    let multi_group_result = multi_groupby.agg(multi_aggs)?;
    println!("Multi-Column GroupBy Result:");
    println!("{:?}", multi_group_result);

    println!("\n=== Advanced Aggregations ===\n");

    // 8. Advanced statistical aggregations
    println!("8. Advanced Statistical Aggregations:");

    let stats_aggs = vec![
        NamedAgg::new("Sales".to_string(), AggFunc::Mean, "sales_mean".to_string()),
        NamedAgg::new("Sales".to_string(), AggFunc::Std, "sales_std".to_string()),
        NamedAgg::new("Sales".to_string(), AggFunc::Var, "sales_var".to_string()),
        NamedAgg::new(
            "Sales".to_string(),
            AggFunc::Median,
            "sales_median".to_string(),
        ),
        NamedAgg::new("Sales".to_string(), AggFunc::Min, "sales_min".to_string()),
        NamedAgg::new("Sales".to_string(), AggFunc::Max, "sales_max".to_string()),
        NamedAgg::new(
            "Sales".to_string(),
            AggFunc::Count,
            "sales_count".to_string(),
        ),
        NamedAgg::new(
            "Sales".to_string(),
            AggFunc::Nunique,
            "sales_nunique".to_string(),
        ),
    ];

    let stats_result = region_groupby.agg(stats_aggs)?;
    println!("Statistical Aggregations Result:");
    println!("{:?}", stats_result);

    println!("\n=== Group Information ===\n");

    // 9. Group information and metadata
    println!("9. Group Information:");

    println!("Number of groups: {}", region_groupby.ngroups());

    let group_sizes = region_groupby.size()?;
    println!("Group sizes:");
    println!("{:?}", group_sizes);

    println!("\n=== Custom Named Aggregations with Builder ===\n");

    // 10. Complex example with custom functions using builder
    println!("10. Complex Custom Aggregations:");

    let complex_aggs = ColumnAggBuilder::new("Sales".to_string())
        .agg(AggFunc::Sum, "total".to_string())
        .agg(AggFunc::Mean, "average".to_string())
        .custom("geometric_mean".to_string(), |values| {
            let product: f64 = values.iter().product();
            product.powf(1.0 / values.len() as f64)
        })
        .custom("harmonic_mean".to_string(), |values| {
            let sum_reciprocals: f64 = values.iter().map(|&x| 1.0 / x).sum();
            values.len() as f64 / sum_reciprocals
        });

    let complex_result = region_groupby.agg_multi(vec![complex_aggs])?;
    println!("Complex Custom Aggregations Result:");
    println!("{:?}", complex_result);

    println!("\n=== Macro Usage Examples ===\n");

    // 11. Using helper macros
    println!("11. Using Helper Macros:");

    // Using named_agg! macro
    let macro_aggs = vec![
        pandrs::named_agg!("Sales", AggFunc::Sum, "macro_sum"),
        pandrs::named_agg!("Sales", AggFunc::Mean, "macro_mean"),
        pandrs::named_agg!("Quantity", AggFunc::Max, "macro_qty_max"),
    ];

    let macro_result = region_groupby.agg(macro_aggs)?;
    println!("Macro Aggregations Result:");
    println!("{:?}", macro_result);

    // Using column_aggs! macro
    let column_macro_agg = pandrs::column_aggs!(
        "Cost",
        (AggFunc::Sum, "cost_total"),
        (AggFunc::Mean, "cost_avg"),
        (AggFunc::Std, "cost_std")
    );

    let column_macro_result = region_groupby.agg_multi(vec![column_macro_agg])?;
    println!("Column Macro Aggregations Result:");
    println!("{:?}", column_macro_result);

    // Using agg_spec! macro
    let spec_dict = pandrs::agg_spec! {
        "Sales" => [(AggFunc::Sum, "sales_total"), (AggFunc::Mean, "sales_avg")],
        "Quantity" => [(AggFunc::Max, "qty_max"), (AggFunc::Min, "qty_min")]
    };

    let spec_result = region_groupby.agg_dict(spec_dict)?;
    println!("Spec Macro Aggregations Result:");
    println!("{:?}", spec_result);

    println!("\n=== Alpha 4 Enhanced GroupBy Complete ===");
    println!("\nNew GroupBy capabilities implemented:");
    println!("✓ Named aggregations (pandas-like syntax)");
    println!("✓ Multiple aggregations per column");
    println!("✓ Custom aggregation functions");
    println!("✓ Dictionary-style aggregation specifications");
    println!("✓ Builder pattern for fluent API");
    println!("✓ Multi-column grouping");
    println!("✓ Comprehensive statistical functions");
    println!("✓ Helper macros for ergonomic usage");
    println!("✓ Group metadata and information");
    println!("✓ Advanced mathematical aggregations");

    Ok(())
}
