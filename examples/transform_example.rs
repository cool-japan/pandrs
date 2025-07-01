use pandrs::dataframe::TransformExt;
use pandrs::{DataFrame, MeltOptions, Series, StackOptions, UnstackOptions};
use std::error::Error;

#[allow(clippy::result_large_err)]
fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Example of Data Transformation ===\n");

    // Create sample data
    let mut df = DataFrame::new();

    // ID column
    df.add_column(
        "id".to_string(),
        Series::new(vec!["1", "2", "3"], Some("id".to_string()))?,
    )?;

    // Product category
    df.add_column(
        "category".to_string(),
        Series::new(
            vec!["Food", "Electronics", "Clothing"],
            Some("category".to_string()),
        )?,
    )?;

    // Monthly sales data
    df.add_column(
        "January".to_string(),
        Series::new(vec!["1000", "1500", "800"], Some("January".to_string()))?,
    )?;

    df.add_column(
        "February".to_string(),
        Series::new(vec!["1200", "1300", "1100"], Some("February".to_string()))?,
    )?;

    df.add_column(
        "March".to_string(),
        Series::new(vec!["900", "1800", "1400"], Some("March".to_string()))?,
    )?;

    println!("Original DataFrame:");
    println!("{:?}", df);

    // melt operation - convert from wide to long format
    println!("\n----- melt operation (convert from wide to long format) -----");
    let melt_options = MeltOptions {
        id_vars: Some(vec!["id".to_string(), "category".to_string()]),
        value_vars: Some(vec![
            "January".to_string(),
            "February".to_string(),
            "March".to_string(),
        ]),
        var_name: Some("Month".to_string()),
        value_name: Some("Sales".to_string()),
    };

    let melted_df = df.melt(&melt_options)?;
    println!("{:?}", melted_df);

    // stack operation
    println!("\n----- stack operation (stack columns to rows) -----");
    let stack_options = StackOptions {
        columns: Some(vec![
            "January".to_string(),
            "February".to_string(),
            "March".to_string(),
        ]),
        var_name: Some("Month".to_string()),
        value_name: Some("Sales".to_string()),
        dropna: false,
    };

    let stacked_df = df.stack(&stack_options)?;
    println!("{:?}", stacked_df);

    // unstack operation (using melted_df)
    println!("\n----- unstack operation (convert rows to columns) -----");
    let unstack_options = UnstackOptions {
        var_column: "Month".to_string(),
        value_column: "Sales".to_string(),
        index_columns: Some(vec!["id".to_string(), "category".to_string()]),
        fill_value: None,
    };

    let unstacked_df = melted_df.unstack(&unstack_options)?;
    println!("{:?}", unstacked_df);

    // Concatenate DataFrames
    println!("\n----- Concatenate DataFrames (concat) -----");

    // Create additional DataFrame
    let mut df2 = DataFrame::new();
    df2.add_column(
        "id".to_string(),
        Series::new(vec!["4", "5"], Some("id".to_string()))?,
    )?;
    df2.add_column(
        "category".to_string(),
        Series::new(
            vec!["Stationery", "Furniture"],
            Some("category".to_string()),
        )?,
    )?;
    df2.add_column(
        "January".to_string(),
        Series::new(vec!["500", "2000"], Some("January".to_string()))?,
    )?;
    df2.add_column(
        "February".to_string(),
        Series::new(vec!["600", "2200"], Some("February".to_string()))?,
    )?;
    df2.add_column(
        "March".to_string(),
        Series::new(vec!["700", "1900"], Some("March".to_string()))?,
    )?;

    println!("Additional DataFrame:");
    println!("{:?}", df2);

    // Concatenate DataFrames
    let concat_df = DataFrame::concat(&[&df, &df2], true)?;
    println!("Concatenated DataFrame:");
    println!("{:?}", concat_df);

    // Conditional aggregation
    println!("\n----- Conditional Aggregation -----");

    // Condition: Calculate the total sales for March by category, only for rows where February sales are 1000 or more
    let result = df.conditional_aggregate(
        "category",
        "March",
        |row| {
            if let Some(sales_str) = row.get("February") {
                if let Ok(sales) = sales_str.parse::<i32>() {
                    return sales >= 1000;
                }
            }
            false
        },
        |values| {
            let sum: i32 = values.iter().filter_map(|v| v.parse::<i32>().ok()).sum();
            sum.to_string()
        },
    )?;

    println!("Conditional Aggregation Result (Total March sales by category for rows where February sales are 1000 or more):");
    println!("{:?}", result);

    Ok(())
}
