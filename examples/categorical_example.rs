use pandrs::error::Result;
use pandrs::series::categorical::Categorical;
use pandrs::{DataFrame, Series};

// For compatibility with the new API
type StringCategorical = Categorical<String>;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of Using Categorical Data Type ===\n");

    // ===========================================================
    // Creating Basic Categorical Data
    // ===========================================================

    println!("--- Creating Basic Categorical Data ---");
    let values = ["Tokyo", "Osaka", "Tokyo", "Nagoya", "Osaka", "Tokyo"];
    let values_str: Vec<String> = values.iter().map(|s| s.to_string()).collect();

    // Create categorical data (unique values are automatically extracted)
    // Changed: Now using boolean instead of Some(CategoricalOrder::Unordered)
    let cat = StringCategorical::new(
        values_str, None,  // Automatically detect categories
        false, // Unordered
    )?;

    println!("Original Data: {:?}", values);
    println!("Categories: {:?}", cat.categories());
    println!("Order Type: {:?}", cat.ordered());
    println!("Data Length: {}", cat.len());

    // Retrieve actual values from categorical data
    println!(
        "\nFirst 3 values: {} {} {}",
        cat.get(0).unwrap_or(&"None".to_string()),
        cat.get(1).unwrap_or(&"None".to_string()),
        cat.get(2).unwrap_or(&"None".to_string())
    );
    println!("\nValues stored internally as codes: {:?}", cat.codes());

    // ===========================================================
    // Creating with Explicit Category List
    // ===========================================================

    println!("\n--- Creating with Explicit Category List ---");
    let values2 = ["Red", "Blue", "Red"];
    let values2_str: Vec<String> = values2.iter().map(|s| s.to_string()).collect();

    // Define all categories beforehand
    let categories = ["Red", "Blue", "Green", "Yellow"];
    let categories_str: Vec<String> = categories.iter().map(|s| s.to_string()).collect();

    // Create ordered categorical data
    // Changed: Now using boolean instead of Some(CategoricalOrder::Ordered)
    let cat2 = StringCategorical::new(
        values2_str,
        Some(categories_str), // Explicit category list
        true,                 // Ordered
    )?;

    println!("Categories: {:?}", cat2.categories()); // Red, Blue, Green, Yellow
    println!("Codes: {:?}", cat2.codes());

    // ===========================================================
    // Operations on Categorical Data
    // ===========================================================

    println!("\n--- Example of Categorical Operations ---");

    // Base categorical data
    // Changed: Using false instead of None for the ordered parameter
    let fruits = ["Apple", "Banana", "Apple", "Orange"];
    let fruits_str: Vec<String> = fruits.iter().map(|s| s.to_string()).collect();
    let mut fruit_cat = StringCategorical::new(fruits_str, None, false)?;

    println!("Original Categories: {:?}", fruit_cat.categories());

    // Add categories
    let new_cats = ["Grape", "Strawberry"];
    let new_cats_str: Vec<String> = new_cats.iter().map(|s| s.to_string()).collect();
    fruit_cat.add_categories(new_cats_str)?;

    println!("Categories after addition: {:?}", fruit_cat.categories());

    // Change category order
    let reordered = ["Banana", "Strawberry", "Orange", "Apple", "Grape"];
    let reordered_str: Vec<String> = reordered.iter().map(|s| s.to_string()).collect();
    fruit_cat.reorder_categories(reordered_str)?;

    println!("Categories after reordering: {:?}", fruit_cat.categories());
    println!("Codes: {:?}", fruit_cat.codes());

    // ===========================================================
    // Integration with DataFrame
    // ===========================================================

    println!("\n--- Integration of Categorical Data with DataFrame ---");

    // Create a basic DataFrame
    let mut df = DataFrame::new();

    // Add regular columns
    let regions = ["Hokkaido", "Kanto", "Kansai", "Kyushu", "Kanto", "Kansai"];
    let regions_str: Vec<String> = regions.iter().map(|s| s.to_string()).collect();
    let pop = ["Low", "High", "High", "Medium", "High", "High"];
    let pop_str: Vec<String> = pop.iter().map(|s| s.to_string()).collect();

    df.add_column(
        "Region".to_string(),
        Series::new(regions_str, Some("Region".to_string()))?,
    )?;
    df.add_column(
        "Population".to_string(),
        Series::new(pop_str, Some("Population".to_string()))?,
    )?;

    println!("Original DataFrame:\n{:?}", df);

    // ===========================================================
    // Creating Simplified Categorical DataFrame
    // ===========================================================

    // Create a DataFrame directly from categorical data
    println!("\n--- Creating DataFrame with Categorical Data ---");

    // Create categorical data
    // Changed: Using boolean instead of Some(CategoricalOrder::Ordered)
    let populations = ["Low", "Medium", "High"];
    let populations_str: Vec<String> = populations.iter().map(|s| s.to_string()).collect();
    let pop_cat = StringCategorical::new(
        populations_str,
        None, // Automatically detect
        true, // Ordered
    )?;

    // Region data
    let regions = ["Hokkaido", "Kanto", "Kansai"];
    let regions_str: Vec<String> = regions.iter().map(|s| s.to_string()).collect();

    // Create DataFrame from both categorical data
    let categoricals = [("Population".to_string(), pop_cat)];

    let mut df_cat = DataFrame::from_categoricals(categoricals.to_vec())?;

    // Add region column
    df_cat.add_column(
        "Region".to_string(),
        Series::new(regions_str, Some("Region".to_string()))?,
    )?;

    println!("\nDataFrame after adding categorical data:\n{:?}", df_cat);

    // Check if columns are categorical
    println!(
        "\nIs 'Population' column categorical: {}",
        df_cat.is_categorical("Population")
    );
    println!(
        "Is 'Region' column categorical: {}",
        df_cat.is_categorical("Region")
    );

    // ===========================================================
    // Example of Multi-Categorical DataFrame
    // ===========================================================

    println!("\n--- Example of Multi-Categorical DataFrame ---");

    // Create product and color data as separate categories
    // Changed: Using false instead of None for the ordered parameter
    let products = ["A", "B", "C"];
    let products_str: Vec<String> = products.iter().map(|s| s.to_string()).collect();
    let product_cat = StringCategorical::new(products_str, None, false)?;

    let colors = ["Red", "Blue", "Green"];
    let colors_str: Vec<String> = colors.iter().map(|s| s.to_string()).collect();
    let color_cat = StringCategorical::new(colors_str, None, false)?;

    // Create a DataFrame containing both categories
    let multi_categoricals = [
        ("Product".to_string(), product_cat),
        ("Color".to_string(), color_cat),
    ];

    let multi_df = DataFrame::from_categoricals(multi_categoricals.to_vec())?;

    println!("Multi-Categorical DataFrame:\n{:?}", multi_df);
    println!(
        "\nIs 'Product' column categorical: {}",
        multi_df.is_categorical("Product")
    );
    println!(
        "Is 'Color' column categorical: {}",
        multi_df.is_categorical("Color")
    );

    // ===========================================================
    // Aggregation and Analysis of Categorical Data
    // ===========================================================

    println!("\n--- Aggregation and Grouping of Categorical Data ---");

    // Start with a simple DataFrame
    let mut df_simple = DataFrame::new();

    // Add product data
    let products = ["A", "B", "C", "A", "B"];
    let products_str: Vec<String> = products.iter().map(|s| s.to_string()).collect();
    let sales = ["100", "150", "200", "120", "180"];
    let sales_str: Vec<String> = sales.iter().map(|s| s.to_string()).collect();

    df_simple.add_column(
        "Product".to_string(),
        Series::new(products_str.clone(), Some("Product".to_string()))?,
    )?;
    df_simple.add_column(
        "Sales".to_string(),
        Series::new(sales_str, Some("Sales".to_string()))?,
    )?;

    println!("Original DataFrame:\n{:?}", df_simple);

    // Aggregate by product
    let product_counts = df_simple.value_counts("Product")?;
    println!("\nProduct Counts:\n{:?}", product_counts);

    // Transformation and interaction between categorical and series
    println!("\n--- Interaction between Categorical and Series ---");

    // Create a simple categorical series
    // Changed: Using false instead of None for the ordered parameter
    let letter_cat = StringCategorical::new(
        ["A".to_string(), "B".to_string(), "C".to_string()].to_vec(),
        None,
        false,
    )?;

    // Convert to series
    let letter_series = letter_cat.to_series(Some("Letter".to_string()))?;
    println!("Converted from categorical to series: {:?}", letter_series);

    // Additional information about categorical data
    println!("\n--- Characteristics of Categorical Data ---");
    println!(
        "Categorical data is stored in memory only once, regardless of repeated string values."
    );
    println!(
        "This makes it particularly efficient for datasets with many duplicate string values."
    );
    println!("Additionally, ordered categorical data allows meaningful sorting of data.");

    println!("\n=== Sample Complete ===");
    Ok(())
}
