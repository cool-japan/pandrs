#![allow(clippy::result_large_err)]
#![allow(unused_mut)]
#![allow(clippy::useless_vec)]
#![allow(unused_imports)]

use pandrs::dataframe::base::DataFrame;
use pandrs::dataframe::groupby::AggFunc;
use pandrs::dataframe::hierarchical_groupby::{
    utils as hierarchical_utils, HierarchicalAggBuilder, HierarchicalGroupByExt,
};
use pandrs::error::Result;
use pandrs::series::base::Series;

fn main() -> Result<()> {
    println!("=== Business Analytics with Hierarchical GroupBy ===\n");
    println!("Real-world business analytics scenarios demonstrating:");
    println!("• Corporate sales performance analysis across organizational hierarchy");
    println!("• Financial reporting with geographic and temporal hierarchies");
    println!("• Retail analytics with product category breakdowns");
    println!("• Employee performance analysis across departments");
    println!("• Customer segmentation and lifetime value analysis\n");

    // Scenario 1: Corporate Sales Performance Analysis
    println!("1. Corporate Sales Performance Analysis");
    println!("   Analyzing sales across Company → Division → Region → Territory");
    let corporate_df = create_corporate_sales_dataset()?;

    let corporate_gb = corporate_df.hierarchical_groupby(vec![
        "company".to_string(),
        "division".to_string(),
        "region".to_string(),
        "territory".to_string(),
    ])?;

    println!("   Corporate Hierarchy Statistics:");
    let stats = corporate_gb.hierarchy_stats();
    println!("     • Total organizational levels: {}", stats.total_levels);
    println!("     • Total groups: {}", stats.total_groups);
    println!(
        "     • Performance units (territories): {}",
        stats.leaf_groups
    );

    // Executive summary (company level)
    let mut exec_summary = corporate_df.hierarchical_groupby(vec![
        "company".to_string(),
        "division".to_string(),
        "region".to_string(),
        "territory".to_string(),
    ])?;
    let exec_results = exec_summary
        .level(0)
        .with_children()
        .agg_hierarchical(vec![
            HierarchicalAggBuilder::new("revenue".to_string())
                .at_level(0, AggFunc::Sum, "total_revenue".to_string())
                .build(),
            HierarchicalAggBuilder::new("units_sold".to_string())
                .at_level(0, AggFunc::Sum, "total_units".to_string())
                .build(),
        ])?;
    display_results(&exec_results, "Executive Summary (Company Level)", 5)?;

    // Division performance comparison
    let division_comparison = corporate_gb.cross_level_comparison(
        "revenue",
        1,             // Division level
        vec![0, 2, 3], // Compare with company, region, territory
    )?;
    display_results(&division_comparison, "Division Performance Comparison", 8)?;

    // Scenario 2: Financial Reporting with Geographic Hierarchy
    println!("\n2. Financial Reporting Analysis");
    println!("   Multi-dimensional analysis: Geographic × Product × Time");

    let financial_df = create_financial_dataset()?;
    let financial_gb = financial_df.hierarchical_groupby(vec![
        "country".to_string(),
        "state".to_string(),
        "city".to_string(),
        "product_line".to_string(),
    ])?;

    // Geographic rollup for CFO reporting
    let geographic_rollup = financial_gb.nested_rollup("profit", AggFunc::Sum)?;
    display_results(&geographic_rollup, "Geographic Profit Rollup", 12)?;

    // Market performance by geographic concentration
    let market_performance = financial_gb.inter_level_ratio("profit", 0, 2)?; // Country to City
    display_results(
        &market_performance,
        "Market Performance (City vs Country)",
        10,
    )?;

    // Scenario 3: Retail Analytics with Product Hierarchy
    println!("\n3. Retail Product Analytics");
    println!("   Category → Subcategory → Brand → SKU analysis");

    let retail_df = create_retail_dataset()?;
    let retail_gb = retail_df.hierarchical_groupby(vec![
        "category".to_string(),
        "subcategory".to_string(),
        "brand".to_string(),
        "sku".to_string(),
    ])?;

    // Category performance with nested transformations (market share within category)
    let market_share_analysis = retail_gb.nested_transform(
        "sales",
        |values: &[f64]| {
            let total: f64 = values.iter().sum();
            if total > 0.0 {
                values.iter().map(|&v| v / total * 100.0).collect() // Convert to percentage
            } else {
                vec![0.0; values.len()]
            }
        },
        0, // Category level for market share calculation
    )?;

    println!("   Market share analysis (first 8 products):");
    display_sample_data(&market_share_analysis, 8)?;

    // Top-performing subcategories filtering
    let top_subcategories = retail_gb.hierarchical_filter(
        "sales",
        1, // Subcategory level
        |total_sales| total_sales > 15000.0,
    )?;
    let top_subcategory_sizes = top_subcategories.size()?;
    display_results(
        &top_subcategory_sizes,
        "Top-Performing Subcategories (>15K sales)",
        10,
    )?;

    // Scenario 4: Employee Performance Analysis
    println!("\n4. Employee Performance Analysis");
    println!("   Company → Department → Team → Employee hierarchy");

    let employee_df = create_employee_performance_dataset()?;
    let employee_gb = employee_df.hierarchical_groupby(vec![
        "company".to_string(),
        "department".to_string(),
        "team".to_string(),
        "employee".to_string(),
    ])?;

    // Department efficiency analysis
    let dept_efficiency = employee_gb.cross_level_agg("performance_score", AggFunc::Mean, 3, 1)?;
    display_results(
        &dept_efficiency,
        "Department Efficiency (Employee Avg to Dept)",
        6,
    )?;

    // Team vs Department performance ratios
    let team_performance = employee_gb.inter_level_ratio("performance_score", 1, 2)?;
    display_results(&team_performance, "Team Performance vs Department", 12)?;

    // Scenario 5: Customer Segmentation Analysis
    println!("\n5. Customer Segmentation & Lifetime Value Analysis");
    println!("   Segment → SubSegment → Customer → Transaction hierarchy");

    let customer_df = create_customer_segmentation_dataset()?;
    let customer_gb = customer_df.hierarchical_groupby(vec![
        "segment".to_string(),
        "subsegment".to_string(),
        "customer_id".to_string(),
    ])?;

    // Customer lifetime value by segment
    let segment_clv = customer_gb.cross_level_comparison(
        "clv",
        0,          // Segment level
        vec![1, 2], // Compare with subsegment and individual customers
    )?;
    display_results(&segment_clv, "Customer Lifetime Value by Segment", 8)?;

    // High-value customer identification
    let high_value_customers = customer_gb.hierarchical_filter(
        "clv",
        2, // Customer level
        |clv| clv > 5000.0,
    )?;
    let hvc_distribution = high_value_customers.size()?;
    display_results(&hvc_distribution, "High-Value Customer Distribution", 10)?;

    // Scenario 6: Advanced Business Intelligence
    println!("\n6. Advanced Business Intelligence Analysis");
    println!("   Complex cross-level analytics for strategic insights");

    // Multi-metric comprehensive analysis
    let comprehensive_analysis = corporate_gb.agg_hierarchical(vec![
        HierarchicalAggBuilder::new("revenue".to_string())
            .at_level(0, AggFunc::Sum, "company_revenue".to_string())
            .at_level(1, AggFunc::Sum, "division_revenue".to_string())
            .at_level(2, AggFunc::Sum, "region_revenue".to_string())
            .with_propagation()
            .build(),
        HierarchicalAggBuilder::new("profit_margin".to_string())
            .at_level(0, AggFunc::Mean, "company_margin".to_string())
            .at_level(1, AggFunc::Mean, "division_margin".to_string())
            .at_level(2, AggFunc::Mean, "region_margin".to_string())
            .build(),
        HierarchicalAggBuilder::new("units_sold".to_string())
            .at_level(1, AggFunc::Count, "division_territories".to_string())
            .build(),
    ])?;
    display_results(
        &comprehensive_analysis,
        "Comprehensive Business Analysis",
        15,
    )?;

    // Performance variance analysis (custom aggregation)
    let variance_analysis =
        corporate_gb.agg_hierarchical(vec![HierarchicalAggBuilder::new("revenue".to_string())
            .at_level(1, AggFunc::Custom, "revenue_cv".to_string())
            .with_custom(|values: &[f64]| {
                if values.len() < 2 {
                    return 0.0;
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                let std_dev = variance.sqrt();
                if mean != 0.0 {
                    std_dev / mean
                } else {
                    0.0
                } // Coefficient of variation
            })
            .build()])?;
    display_results(
        &variance_analysis,
        "Revenue Coefficient of Variation by Division",
        8,
    )?;

    println!("\n=== Business Analytics Complete ===");
    println!("\nKey Business Insights Demonstrated:");
    println!("✓ Executive reporting with organizational hierarchy rollups");
    println!("✓ Geographic performance analysis across multiple dimensions");
    println!("✓ Product portfolio analysis with market share calculations");
    println!("✓ Employee performance tracking across organizational levels");
    println!("✓ Customer segmentation with lifetime value analysis");
    println!("✓ Advanced business intelligence with custom metrics");
    println!("✓ Risk analysis through performance variance measurements");
    println!("✓ Strategic filtering for high-value segments identification");
    println!("✓ Cross-level comparative analysis for decision making");
    println!("✓ Production-ready hierarchical analytics for enterprise use");

    Ok(())
}

/// Create corporate sales dataset with organizational hierarchy
fn create_corporate_sales_dataset() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    let companies = vec!["TechCorp"; 16];
    let divisions = vec![
        "Cloud", "Cloud", "Cloud", "Cloud", "Software", "Software", "Software", "Software",
        "Hardware", "Hardware", "Hardware", "Hardware", "Services", "Services", "Services",
        "Services",
    ];
    let regions = vec![
        "North", "North", "South", "South", "North", "North", "South", "South", "North", "North",
        "South", "South", "North", "North", "South", "South",
    ];
    let territories = vec![
        "NY", "CA", "TX", "FL", "NY", "CA", "TX", "FL", "NY", "CA", "TX", "FL", "NY", "CA", "TX",
        "FL",
    ];
    let revenues = vec![
        2500000.0, 3200000.0, 1800000.0, 2100000.0, 1500000.0, 1900000.0, 1200000.0, 1400000.0,
        2200000.0, 2800000.0, 1600000.0, 1900000.0, 1100000.0, 1300000.0, 900000.0, 1050000.0,
    ];
    let units_sold = vec![
        500, 640, 360, 420, 750, 950, 600, 700, 220, 280, 160, 190, 550, 650, 450, 525,
    ];
    let profit_margins = vec![
        25.5, 30.2, 22.1, 27.8, 35.0, 38.5, 32.0, 34.2, 28.0, 31.5, 25.5, 29.0, 40.0, 42.5, 38.0,
        39.5,
    ];

    df.add_column(
        "company".to_string(),
        Series::new(
            companies.iter().map(|s| s.to_string()).collect(),
            Some("company".to_string()),
        )?,
    )?;
    df.add_column(
        "division".to_string(),
        Series::new(
            divisions.iter().map(|s| s.to_string()).collect(),
            Some("division".to_string()),
        )?,
    )?;
    df.add_column(
        "region".to_string(),
        Series::new(
            regions.iter().map(|s| s.to_string()).collect(),
            Some("region".to_string()),
        )?,
    )?;
    df.add_column(
        "territory".to_string(),
        Series::new(
            territories.iter().map(|s| s.to_string()).collect(),
            Some("territory".to_string()),
        )?,
    )?;
    df.add_column(
        "revenue".to_string(),
        Series::new(
            revenues.iter().map(|&r| r.to_string()).collect(),
            Some("revenue".to_string()),
        )?,
    )?;
    df.add_column(
        "units_sold".to_string(),
        Series::new(
            units_sold.iter().map(|&u| u.to_string()).collect(),
            Some("units_sold".to_string()),
        )?,
    )?;
    df.add_column(
        "profit_margin".to_string(),
        Series::new(
            profit_margins.iter().map(|&p| p.to_string()).collect(),
            Some("profit_margin".to_string()),
        )?,
    )?;

    Ok(df)
}

/// Create financial dataset with geographic hierarchy
fn create_financial_dataset() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    let countries = vec![
        "USA", "USA", "USA", "USA", "Canada", "Canada", "Canada", "Canada",
    ];
    let states = vec!["CA", "CA", "NY", "NY", "ON", "ON", "BC", "BC"];
    let cities = vec![
        "LA",
        "SF",
        "NYC",
        "Albany",
        "Toronto",
        "Ottawa",
        "Vancouver",
        "Victoria",
    ];
    let product_lines = vec![
        "Premium", "Standard", "Premium", "Standard", "Premium", "Standard", "Premium", "Standard",
    ];
    let profits = vec![
        850000.0, 620000.0, 940000.0, 580000.0, 720000.0, 480000.0, 680000.0, 420000.0,
    ];
    let expenses = vec![
        320000.0, 280000.0, 360000.0, 240000.0, 290000.0, 210000.0, 270000.0, 180000.0,
    ];

    df.add_column(
        "country".to_string(),
        Series::new(
            countries.iter().map(|s| s.to_string()).collect(),
            Some("country".to_string()),
        )?,
    )?;
    df.add_column(
        "state".to_string(),
        Series::new(
            states.iter().map(|s| s.to_string()).collect(),
            Some("state".to_string()),
        )?,
    )?;
    df.add_column(
        "city".to_string(),
        Series::new(
            cities.iter().map(|s| s.to_string()).collect(),
            Some("city".to_string()),
        )?,
    )?;
    df.add_column(
        "product_line".to_string(),
        Series::new(
            product_lines.iter().map(|s| s.to_string()).collect(),
            Some("product_line".to_string()),
        )?,
    )?;
    df.add_column(
        "profit".to_string(),
        Series::new(
            profits.iter().map(|&p| p.to_string()).collect(),
            Some("profit".to_string()),
        )?,
    )?;
    df.add_column(
        "expenses".to_string(),
        Series::new(
            expenses.iter().map(|&e| e.to_string()).collect(),
            Some("expenses".to_string()),
        )?,
    )?;

    Ok(df)
}

/// Create retail dataset with product hierarchy
fn create_retail_dataset() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    let categories = vec![
        "Electronics",
        "Electronics",
        "Electronics",
        "Electronics",
        "Apparel",
        "Apparel",
        "Apparel",
        "Apparel",
    ];
    let subcategories = vec![
        "Phones", "Phones", "Laptops", "Laptops", "Men", "Men", "Women", "Women",
    ];
    let brands = vec![
        "Apple", "Samsung", "Dell", "HP", "Nike", "Adidas", "Zara", "H&M",
    ];
    let skus = vec![
        "IP14",
        "S23",
        "XPS13",
        "Elitebook",
        "AirMax",
        "Ultraboost",
        "Blazer",
        "Dress",
    ];
    let sales = vec![
        45000.0, 38000.0, 32000.0, 28000.0, 22000.0, 24000.0, 18000.0, 16000.0,
    ];
    let quantities = vec![450, 760, 160, 140, 550, 600, 360, 400];

    df.add_column(
        "category".to_string(),
        Series::new(
            categories.iter().map(|s| s.to_string()).collect(),
            Some("category".to_string()),
        )?,
    )?;
    df.add_column(
        "subcategory".to_string(),
        Series::new(
            subcategories.iter().map(|s| s.to_string()).collect(),
            Some("subcategory".to_string()),
        )?,
    )?;
    df.add_column(
        "brand".to_string(),
        Series::new(
            brands.iter().map(|s| s.to_string()).collect(),
            Some("brand".to_string()),
        )?,
    )?;
    df.add_column(
        "sku".to_string(),
        Series::new(
            skus.iter().map(|s| s.to_string()).collect(),
            Some("sku".to_string()),
        )?,
    )?;
    df.add_column(
        "sales".to_string(),
        Series::new(
            sales.iter().map(|&s| s.to_string()).collect(),
            Some("sales".to_string()),
        )?,
    )?;
    df.add_column(
        "quantity".to_string(),
        Series::new(
            quantities.iter().map(|&q| q.to_string()).collect(),
            Some("quantity".to_string()),
        )?,
    )?;

    Ok(df)
}

/// Create employee performance dataset
fn create_employee_performance_dataset() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    let companies = vec!["InnovateCorp"; 12];
    let departments = vec![
        "Engineering",
        "Engineering",
        "Engineering",
        "Sales",
        "Sales",
        "Sales",
        "Marketing",
        "Marketing",
        "Marketing",
        "HR",
        "HR",
        "HR",
    ];
    let teams = vec![
        "Backend",
        "Frontend",
        "DevOps",
        "Enterprise",
        "SMB",
        "Partners",
        "Digital",
        "Content",
        "Events",
        "Talent",
        "Ops",
        "Culture",
    ];
    let employees = vec![
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Kate", "Liam",
    ];
    let performance_scores = vec![
        94.5, 88.2, 91.7, 85.3, 92.1, 87.8, 90.4, 86.9, 89.3, 93.2, 88.7, 91.0,
    ];
    let projects_completed = vec![12, 8, 10, 15, 18, 14, 6, 9, 7, 4, 11, 5];

    df.add_column(
        "company".to_string(),
        Series::new(
            companies.iter().map(|s| s.to_string()).collect(),
            Some("company".to_string()),
        )?,
    )?;
    df.add_column(
        "department".to_string(),
        Series::new(
            departments.iter().map(|s| s.to_string()).collect(),
            Some("department".to_string()),
        )?,
    )?;
    df.add_column(
        "team".to_string(),
        Series::new(
            teams.iter().map(|s| s.to_string()).collect(),
            Some("team".to_string()),
        )?,
    )?;
    df.add_column(
        "employee".to_string(),
        Series::new(
            employees.iter().map(|s| s.to_string()).collect(),
            Some("employee".to_string()),
        )?,
    )?;
    df.add_column(
        "performance_score".to_string(),
        Series::new(
            performance_scores.iter().map(|&p| p.to_string()).collect(),
            Some("performance_score".to_string()),
        )?,
    )?;
    df.add_column(
        "projects_completed".to_string(),
        Series::new(
            projects_completed.iter().map(|&p| p.to_string()).collect(),
            Some("projects_completed".to_string()),
        )?,
    )?;

    Ok(df)
}

/// Create customer segmentation dataset
fn create_customer_segmentation_dataset() -> Result<DataFrame> {
    let mut df = DataFrame::new();

    let segments = vec![
        "Premium", "Premium", "Premium", "Standard", "Standard", "Standard", "Basic", "Basic",
    ];
    let subsegments = vec![
        "VIP", "Gold", "Silver", "Plus", "Regular", "Starter", "Entry", "Trial",
    ];
    let customer_ids = vec![
        "C001", "C002", "C003", "C004", "C005", "C006", "C007", "C008",
    ];
    let clvs = vec![
        12500.0, 8900.0, 6200.0, 4500.0, 3200.0, 2100.0, 1200.0, 800.0,
    ];
    let avg_order_values = vec![450.0, 320.0, 280.0, 180.0, 150.0, 120.0, 80.0, 60.0];
    let purchase_frequency = vec![28, 28, 22, 25, 21, 18, 15, 13];

    df.add_column(
        "segment".to_string(),
        Series::new(
            segments.iter().map(|s| s.to_string()).collect(),
            Some("segment".to_string()),
        )?,
    )?;
    df.add_column(
        "subsegment".to_string(),
        Series::new(
            subsegments.iter().map(|s| s.to_string()).collect(),
            Some("subsegment".to_string()),
        )?,
    )?;
    df.add_column(
        "customer_id".to_string(),
        Series::new(
            customer_ids.iter().map(|s| s.to_string()).collect(),
            Some("customer_id".to_string()),
        )?,
    )?;
    df.add_column(
        "clv".to_string(),
        Series::new(
            clvs.iter().map(|&c| c.to_string()).collect(),
            Some("clv".to_string()),
        )?,
    )?;
    df.add_column(
        "avg_order_value".to_string(),
        Series::new(
            avg_order_values.iter().map(|&a| a.to_string()).collect(),
            Some("avg_order_value".to_string()),
        )?,
    )?;
    df.add_column(
        "purchase_frequency".to_string(),
        Series::new(
            purchase_frequency.iter().map(|&p| p.to_string()).collect(),
            Some("purchase_frequency".to_string()),
        )?,
    )?;

    Ok(df)
}

/// Display sample data from DataFrame
fn display_sample_data(df: &DataFrame, rows: usize) -> Result<()> {
    let display_rows = rows.min(df.row_count());

    for i in 0..display_rows {
        let row_data: Vec<String> = df
            .column_names()
            .iter()
            .map(|col| {
                df.get_column_string_values(col)
                    .map(|values| values.get(i).cloned().unwrap_or_else(|| "NULL".to_string()))
                    .unwrap_or_else(|_| "ERROR".to_string())
            })
            .collect();

        println!("     {}: {}", i, row_data.join(" | "));
    }

    Ok(())
}

/// Display a sample of DataFrame with title
fn display_results(df: &DataFrame, title: &str, max_rows: usize) -> Result<()> {
    println!("   {}:", title);
    if df.row_count() == 0 {
        println!("     (No data)");
        return Ok(());
    }

    // Display column headers
    let headers = df.column_names();
    println!("     Headers: {}", headers.join(" | "));

    // Display data rows
    let display_rows = max_rows.min(df.row_count());
    for i in 0..display_rows {
        let row_data: Vec<String> = headers
            .iter()
            .map(|col| {
                df.get_column_string_values(col)
                    .map(|values| values.get(i).cloned().unwrap_or_else(|| "NULL".to_string()))
                    .unwrap_or_else(|_| "ERROR".to_string())
            })
            .collect();

        println!("     {}: {}", i, row_data.join(" | "));
    }

    if df.row_count() > display_rows {
        println!("     ... and {} more rows", df.row_count() - display_rows);
    }

    println!();
    Ok(())
}
