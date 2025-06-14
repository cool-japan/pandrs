use pandrs::error::Result;
use pandrs::series::{CategoricalOrder, StringCategorical};
use pandrs::{DataFrame, Series, NA};
use std::path::Path;

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Example of Categorical Data with Missing Values ===\n");

    // 1. Create categorical data
    println!("1. Create categorical data");

    // Create a vector with NA values
    let values = vec![
        NA::Value("Red".to_string()),
        NA::Value("Blue".to_string()),
        NA::NA, // Missing value
        NA::Value("Green".to_string()),
        NA::Value("Red".to_string()), // Duplicate value
    ];

    // Create categorical data type from vector
    // Create as unordered category
    let cat = StringCategorical::from_na_vec(
        values.clone(),
        None,                              // Auto-detect categories
        Some(CategoricalOrder::Unordered), // Unordered
    )?;

    println!("Categories: {:?}", cat.categories());
    println!("Number of categories: {}", cat.categories().len());
    println!("Number of data: {}", cat.len());

    // Display category codes
    println!("Internal codes: {:?}", cat.codes());
    println!();

    // 2. Create ordered categorical data
    println!("2. Create ordered categorical data");

    // Explicitly ordered category list
    let ordered_categories = vec!["Low".to_string(), "Medium".to_string(), "High".to_string()];

    // Create a vector with NA values
    let values = vec![
        NA::Value("Medium".to_string()),
        NA::Value("Low".to_string()),
        NA::NA, // Missing value
        NA::Value("High".to_string()),
        NA::Value("Medium".to_string()), // Duplicate value
    ];

    // Create as ordered category
    let ordered_cat = StringCategorical::from_na_vec(
        values.clone(),
        Some(ordered_categories),        // Explicit category list
        Some(CategoricalOrder::Ordered), // Ordered
    )?;

    println!("Ordered categories: {:?}", ordered_cat.categories());
    println!("Number of categories: {}", ordered_cat.categories().len());
    println!("Number of data: {}", ordered_cat.len());

    // Display category codes
    println!("Internal codes: {:?}", ordered_cat.codes());
    println!();

    // 3. Operations on categorical data
    println!("3. Operations on categorical data");

    // Create two categorical data
    let values1 = vec![
        NA::Value("A".to_string()),
        NA::Value("B".to_string()),
        NA::NA,
        NA::Value("C".to_string()),
    ];

    let values2 = vec![
        NA::Value("B".to_string()),
        NA::Value("C".to_string()),
        NA::Value("D".to_string()),
        NA::NA,
    ];

    let cat1 = StringCategorical::from_na_vec(values1, None, None)?;
    let cat2 = StringCategorical::from_na_vec(values2, None, None)?;

    // Set operations
    let union = cat1.union(&cat2)?; // Union
    let intersection = cat1.intersection(&cat2)?; // Intersection
    let difference = cat1.difference(&cat2)?; // Difference

    println!("Categories of set 1: {:?}", cat1.categories());
    println!("Categories of set 2: {:?}", cat2.categories());
    println!("Union: {:?}", union.categories());
    println!("Intersection: {:?}", intersection.categories());
    println!("Difference (set 1 - set 2): {:?}", difference.categories());
    println!();

    // 4. Using categorical columns in DataFrame
    println!("4. Using categorical columns in DataFrame");

    // Create a vector with NA values (first create for categorical)
    let values = vec![
        NA::Value("High".to_string()),
        NA::Value("Medium".to_string()),
        NA::NA,
        NA::Value("Low".to_string()),
    ];

    // Simplified for sample code
    let order_cats = vec!["Low".to_string(), "Medium".to_string(), "High".to_string()];

    // Create categorical data
    let cat_eval = StringCategorical::from_na_vec(
        values.clone(), // Clone it
        Some(order_cats),
        Some(CategoricalOrder::Ordered),
    )?;

    // Output the size of the created categorical data
    println!("Size of created categorical data: {}", cat_eval.len());

    // Add as categorical column
    let categoricals = vec![("Evaluation".to_string(), cat_eval)];
    let mut df = DataFrame::from_categoricals(categoricals)?;

    // Check the number of rows in the data and match it
    println!("Number of rows in DataFrame: {}", df.row_count());
    println!("Note: NA values are excluded when creating DataFrame");

    // Add numeric column (match the number of rows)
    let scores = vec![95, 80, 0]; // Match the number of rows in DataFrame
    println!("Size of scores: {}", scores.len());

    df.add_column(
        "Score".to_string(),
        Series::new(scores, Some("Score".to_string()))?,
    )?;

    println!("DataFrame: ");
    println!("{:#?}", df);

    // Retrieve and verify categorical data
    println!(
        "Is 'Evaluation' column categorical: {}",
        df.is_categorical("Evaluation")
    );

    // Explicitly handle errors
    match df.get_categorical::<String>("Evaluation") {
        Ok(cat_col) => println!(
            "Categories of 'Evaluation' column: {:?}",
            cat_col.categories()
        ),
        Err(_) => println!("Failed to retrieve categories of 'Evaluation' column"),
    }
    println!();

    // 5. Input and output with CSV file
    println!("5. Input and output with CSV file");

    // Save to temporary file
    let temp_path = Path::new("/tmp/categorical_example.csv");
    df.to_csv(temp_path)?;

    println!("Saved to CSV file: {}", temp_path.display());

    // Load from file
    let df_loaded = DataFrame::from_csv(temp_path, true)?;

    // After loading from CSV, categorical information is lost (loaded as regular string column)
    println!("Data loaded from CSV:");
    println!("{:#?}", df_loaded);

    // Check data loaded from CSV

    // Note that data loaded from CSV is in a special format
    println!("Example of data format loaded from CSV:");
    println!(
        "First value of 'Evaluation' column: {:?}",
        df_loaded
            .get_column::<String>("Evaluation")
            .unwrap()
            .values()[0]
    );

    // To reconstruct categorical data from this CSV loaded data,
    // more complex processing is required, so the following is a simple example

    // Create new categorical data as an example
    let new_values = vec![
        NA::Value("High".to_string()),
        NA::Value("Medium".to_string()),
        NA::NA,
        NA::Value("Low".to_string()),
    ];

    let new_cat =
        StringCategorical::from_na_vec(new_values, None, Some(CategoricalOrder::Ordered))?;

    println!("Example of newly created categorical data:");
    println!("Categories: {:?}", new_cat.categories());
    println!("Order: {:?}", new_cat.ordered());

    println!("\nTo actually convert data loaded from CSV to categorical data,");
    println!("parsing processing according to the format and string escaping method of the CSV is required.");

    println!("\n=== Sample End ===");
    Ok(())
}
