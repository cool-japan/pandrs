use pandrs::error::Result;
use pandrs::{NASeries, NA};

#[allow(clippy::result_large_err)]
fn main() -> Result<()> {
    println!("=== Support for NA Values (Missing Data) ===");

    // Create data with missing values
    let data = vec![
        NA::Value(10),
        NA::Value(20),
        NA::NA, // Missing value
        NA::Value(40),
        NA::NA, // Missing value
    ];

    // Create NASeries
    let series = NASeries::new(data, Some("numbers".to_string()))?;

    println!("Series with NA: {:?}", series);
    println!("Number of NAs: {}", series.na_count());
    println!("Number of values: {}", series.value_count());
    println!("Contains NA: {}", series.has_na());

    // Behavior of aggregation functions
    println!("\n--- Handling of NA Values ---");
    println!("Sum (ignoring NAs): {:?}", series.sum());
    println!("Mean (ignoring NAs): {:?}", series.mean());
    println!("Minimum (ignoring NAs): {:?}", series.min());
    println!("Maximum (ignoring NAs): {:?}", series.max());

    // Handling NAs
    println!("\n--- Handling of NA Values ---");
    let dropped = series.dropna()?;
    println!("Series after dropping NAs: {:?}", dropped);
    println!("Length after dropping NAs: {}", dropped.len());

    let filled = series.fillna(0)?;
    println!("Series after filling NAs with 0: {:?}", filled);

    // Conversion from Option
    println!("\n--- Conversion from Option ---");
    let option_data = vec![Some(100), Some(200), None, Some(400), None];
    let option_series = NASeries::from_options(option_data, Some("from_options".to_string()))?;
    println!("Series from Option: {:?}", option_series);

    // Numerical operations
    println!("\n--- Numerical Operations with NA Values ---");
    let a = NA::Value(10);
    let b = NA::Value(5);
    let na = NA::<i32>::NA;

    println!("{:?} + {:?} = {:?}", a, b, a + b);
    println!("{:?} - {:?} = {:?}", a, b, a - b);
    println!("{:?} * {:?} = {:?}", a, b, a * b);
    println!("{:?} / {:?} = {:?}", a, b, a / b);

    println!("{:?} + {:?} = {:?}", a, na, a + na);
    println!("{:?} * {:?} = {:?}", b, na, b * na);

    println!("=== NA Value Sample Complete ===");
    Ok(())
}
