use pandrs::error::Result;
use pandrs::{DataFrame, Series};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== String Accessor Example ===");

    // Create a string series
    let data = vec![
        "Hello World".to_string(),
        "RUST Programming".to_string(),
        "  pandas-like  ".to_string(),
        "Data Analysis".to_string(),
    ];

    let series = Series::new(data, Some("text_data".to_string()))?;
    println!("Original Series: {:?}", series.values());

    // Test string accessor methods
    let str_accessor = series.str()?;

    // Test uppercase
    let upper_result = str_accessor.upper()?;
    println!("Uppercase: {:?}", upper_result.values());

    // Test lowercase
    let lower_result = str_accessor.lower()?;
    println!("Lowercase: {:?}", lower_result.values());

    // Test contains
    let contains_result = str_accessor.contains("rust", false, false)?;
    println!(
        "Contains 'rust' (case insensitive): {:?}",
        contains_result.values()
    );

    // Test startswith
    let startswith_result = str_accessor.startswith("hello", false)?;
    println!(
        "Starts with 'hello' (case insensitive): {:?}",
        startswith_result.values()
    );

    // Test string length
    let len_result = str_accessor.len()?;
    println!("String lengths: {:?}", len_result.values());

    // Test strip
    let strip_result = str_accessor.strip(None)?;
    println!("Stripped: {:?}", strip_result.values());

    // Test replace
    let replace_result = str_accessor.replace("a", "X", false, false)?;
    println!("Replace 'a' with 'X': {:?}", replace_result.values());

    // Test regex operations
    let regex_contains = str_accessor.contains(r"\b[A-Z]+\b", true, true)?;
    println!(
        "Contains uppercase words (regex): {:?}",
        regex_contains.values()
    );

    // Test extract
    let extract_result = str_accessor.extract(r"([A-Z][a-z]+)", None)?;
    println!(
        "Extract first capitalized word: {:?}",
        extract_result.values()
    );

    println!("\n=== DataFrame with String Operations ===");

    // Create a DataFrame with string column
    let mut df = DataFrame::new();
    df.add_column("text".to_string(), series)?;

    println!("DataFrame created successfully with string column!");

    Ok(())
}
