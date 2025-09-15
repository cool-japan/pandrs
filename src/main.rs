use std::cmp::PartialOrd;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::iter::Sum;
use std::path::Path;
use std::str::FromStr;

// Import the cast function from the num-traits crate
// You need to add `num-traits = "0.2"` to Cargo.toml
use num_traits::cast;

// DataFrame structure
#[derive(Debug, Clone)]
pub struct DataFrame<T: Debug + Clone> {
    data: HashMap<String, Vec<T>>,
    columns: Vec<String>, // Maintain column order
    row_count: usize,
}

// Basic method implementation for DataFrame
impl<T: Debug + Clone> Default for DataFrame<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Debug + Clone> DataFrame<T> {
    pub fn new() -> Self {
        DataFrame {
            data: HashMap::new(),
            columns: Vec::new(),
            row_count: 0,
        }
    }

    // Generate DataFrame with specified column names and data
    pub fn from_vec(data: HashMap<String, Vec<T>>) -> Result<Self, Box<dyn Error>> {
        if data.is_empty() {
            return Ok(DataFrame::new());
        }
        // Get column names from HashMap keys (order is not guaranteed)
        // Removed `mut` (warning: variable does not need to be mutable)
        let columns: Vec<String> = data.keys().cloned().collect();
        // Stabilize column order if needed (e.g., sort)
        // let mut columns = data.keys().cloned().collect::<Vec<_>>(); // mut would be needed
        // columns.sort();

        let first_column_name = columns
            .first()
            .ok_or("Data map is empty after collecting keys")?;
        let row_count = data[first_column_name].len();

        // Check if all columns have the same length
        for column_name in &columns {
            let column_data = data
                .get(column_name)
                .ok_or("Internal error: Column name mismatch")?;
            if column_data.len() != row_count {
                return Err(format!(
                    "Column '{}' has length {} but expected {}",
                    column_name,
                    column_data.len(),
                    row_count
                )
                .into());
            }
        }
        // Construct DataFrame (using the obtained column list)
        Ok(DataFrame {
            data,
            columns,
            row_count,
        })
    }

    // Add a row to the DataFrame
    pub fn add_row(&mut self, mut row: HashMap<String, T>) -> Result<(), Box<dyn Error>> {
        // If the DataFrame is empty, insert the data as the first row and set the column order
        if self.columns.is_empty() {
            // Since the order of HashMap keys is indeterminate, we need to determine the order here
            self.columns = row.keys().cloned().collect();
            // Sort if necessary:
            // self.columns.sort();
            for column_name in &self.columns {
                let value = row.remove(column_name).ok_or_else(|| {
                    format!("Internal error: value for column '{column_name}' disappeared")
                })?;
                self.data.insert(column_name.clone(), vec![value]);
            }
            self.row_count = 1;
            return Ok(());
        }

        // Check if the number of columns matches the DataFrame
        if row.len() != self.columns.len() {
            return Err(format!(
                "Row has {} columns, but DataFrame expects {}",
                row.len(),
                self.columns.len()
            )
            .into());
        }

        // Add values according to the column order of the DataFrame
        for column_name in &self.columns {
            match row.remove(column_name) {
                // Consume the value from the HashMap
                Some(value) => {
                    // The corresponding Vec in the data HashMap should be guaranteed to exist
                    self.data.get_mut(column_name).unwrap().push(value);
                }
                None => {
                    // If the row does not contain a column of the DataFrame
                    return Err(format!(
                        "Missing value for column '{column_name}' in the provided row"
                    )
                    .into());
                }
            }
        }
        self.row_count += 1;
        Ok(())
    }

    // Return the data of the specified column
    pub fn get_column(&self, column_name: &str) -> Option<&Vec<T>> {
        self.data.get(column_name)
    }

    // Return a mutable reference to the data of the specified column
    pub fn get_column_mut(&mut self, column_name: &str) -> Option<&mut Vec<T>> {
        self.data.get_mut(column_name)
    }

    // Check if the specified column exists
    pub fn contains_column(&self, column_name: &str) -> bool {
        self.data.contains_key(column_name)
    }

    // Return the column names of the DataFrame (in the defined order)
    pub fn get_columns(&self) -> &Vec<String> {
        &self.columns
    }

    // Return the number of rows in the DataFrame
    pub fn get_row_count(&self) -> usize {
        self.row_count
    }

    // Filter the DataFrame
    pub fn filter<F>(&self, predicate: F) -> Result<Self, Box<dyn Error>>
    where
        F: Fn(&HashMap<&String, &T>) -> bool,
    {
        let mut filtered_data: HashMap<String, Vec<T>> = HashMap::new();
        for column_name in &self.columns {
            filtered_data.insert(column_name.clone(), Vec::new());
        }
        let mut new_row_count = 0;

        for i in 0..self.row_count {
            // Create row data as a HashMap (using references)
            let mut row_map: HashMap<&String, &T> = HashMap::with_capacity(self.columns.len());
            for column_name in &self.columns {
                row_map.insert(column_name, &self.data[column_name][i]);
            }
            // Check if the condition is met
            if predicate(&row_map) {
                // Add the data of the rows that meet the condition to the new DataFrame
                for column_name in &self.columns {
                    filtered_data
                        .get_mut(column_name)
                        .unwrap() // The key should exist
                        .push(self.data[column_name][i].clone()); // clone is needed
                }
                new_row_count += 1;
            }
        }
        // Construct and return the new DataFrame
        Ok(DataFrame {
            data: filtered_data,
            columns: self.columns.clone(), // Maintain column order
            row_count: new_row_count,
        })
    }

    // Sort the DataFrame (the original DataFrame is not changed, a new DataFrame is returned)
    pub fn sort_by<F>(&self, compare: F) -> Self
    where
        F: Fn(&HashMap<&String, &T>, &HashMap<&String, &T>) -> std::cmp::Ordering,
    {
        if self.row_count == 0 {
            return self.clone();
        } // Return a clone if empty

        // Prepare a HashMap to store the sorted data
        let mut sorted_data: HashMap<String, Vec<T>> = HashMap::new();
        for column_name in &self.columns {
            sorted_data.insert(column_name.clone(), Vec::with_capacity(self.row_count));
        }

        // Create row indices (0..row_count)
        let mut row_indices: Vec<usize> = (0..self.row_count).collect();

        // Helper function to sort indices (using references to the original data)
        let get_row_map = |index: usize| -> HashMap<&String, &T> {
            let mut row_map = HashMap::with_capacity(self.columns.len());
            for column_name in &self.columns {
                row_map.insert(column_name, &self.data[column_name][index]);
            }
            row_map
        };

        // Sort the indices using the comparison function
        row_indices.sort_unstable_by(|&a, &b| {
            // Unstable is often sufficient for performance
            let row_map_a = get_row_map(a);
            let row_map_b = get_row_map(b);
            compare(&row_map_a, &row_map_b)
        });

        // Move the data to the new HashMap in the order of the sorted indices
        for index in row_indices {
            for column_name in &self.columns {
                sorted_data
                    .get_mut(column_name)
                    .unwrap()
                    .push(self.data[column_name][index].clone()); // clone is needed
            }
        }
        // Construct the new DataFrame
        DataFrame {
            data: sorted_data,
            columns: self.columns.clone(),
            row_count: self.row_count,
        }
    }

    // Join two DataFrames (Inner Join)
    pub fn join(&self, other: &Self, join_column: &str) -> Result<Self, Box<dyn Error>>
    where
        T: Eq + std::hash::Hash, // Hash is also needed to use as a key in HashMap
    {
        // Check if the join column exists
        if !self.contains_column(join_column) {
            return Err(format!("Join column '{join_column}' not found in left DataFrame").into());
        }
        if !other.contains_column(join_column) {
            return Err(format!("Join column '{join_column}' not found in right DataFrame").into());
        }

        // Prepare a HashMap and column list to store the joined data
        let mut joined_data: HashMap<String, Vec<T>> = HashMap::new();
        let mut joined_columns = self.columns.clone(); // Maintain the order of the left columns
        let mut right_col_rename_map = HashMap::new(); // For renaming duplicate column names on the right

        // Create the list of columns after the join (add the right columns, rename duplicates)
        for column_name in &other.columns {
            if column_name != join_column {
                let new_col_name = if self.contains_column(column_name) {
                    // Rename if duplicate (e.g., "col" -> "col_right")
                    let mut rename = format!("{column_name}_right");
                    let mut count = 1;
                    // Add a serial number if the renamed name is still a duplicate (rare case)
                    while self.contains_column(&rename)
                        || right_col_rename_map.contains_key(&rename)
                    {
                        rename = format!("{column_name}_right{count}");
                        count += 1;
                    }
                    rename
                } else {
                    column_name.clone() // Use as is if not a duplicate
                };
                // Save the mapping from the original name to the new name
                right_col_rename_map.insert(column_name.clone(), new_col_name.clone());
                joined_columns.push(new_col_name); // Add to the list of columns after the join
            }
        }
        // Initialize the HashMap for the joined data
        for column_name in &joined_columns {
            joined_data.insert(column_name.clone(), Vec::new());
        }

        // Get the data of the join column (unwrap is safe because existence is checked)
        let left_join_col_data = self.get_column(join_column).unwrap();
        let right_join_col_data = other.get_column(join_column).unwrap();

        // Create a map of row indices for the join key in the right DataFrame for performance
        let mut right_indices_map: HashMap<&T, Vec<usize>> = HashMap::new();
        for (j, right_val) in right_join_col_data.iter().enumerate() {
            right_indices_map.entry(right_val).or_default().push(j);
        }

        // Join process
        let mut new_row_count = 0;
        // Process each row of the left DataFrame
        for (i, left_value) in left_join_col_data.iter().enumerate().take(self.row_count) {
            // Use the index map of the right side to search for rows with matching join keys
            if let Some(matching_indices) = right_indices_map.get(left_value) {
                // If matching rows are found on the right side, generate join results for each row
                for &j in matching_indices {
                    new_row_count += 1;
                    // Add the data from the left side to the joined data
                    for column_name in &self.columns {
                        joined_data
                            .get_mut(column_name)
                            .unwrap()
                            .push(self.data[column_name][i].clone());
                    }
                    // Add the data from the right side (excluding the join column) to the joined data
                    for original_right_col in &other.columns {
                        if original_right_col != join_column {
                            // Add using the renamed column name
                            let final_col_name =
                                right_col_rename_map.get(original_right_col).unwrap();
                            joined_data
                                .get_mut(final_col_name)
                                .unwrap()
                                .push(other.data[original_right_col][j].clone());
                        }
                    }
                }
            }
        }
        // Construct the joined DataFrame
        Ok(DataFrame {
            data: joined_data,
            columns: joined_columns,
            row_count: new_row_count,
        })
    }
}

// Extension for DataFrame limited to numeric types
impl<T> DataFrame<T>
where
    T: FromStr
        + Clone
        + Debug
        + Sum<T>
        + Default
        + Copy
        + std::ops::Div<Output = T>
        + PartialOrd
        + 'static,
    T: num_traits::NumCast, // NumCast trait bound (used in mean)
{
    // Calculate the sum of the specified column
    pub fn sum(&self, column_name: &str) -> Result<T, Box<dyn Error>> {
        let column = self
            .get_column(column_name)
            .ok_or_else(|| format!("Column '{column_name}' not found for sum()"))?;
        if column.is_empty() {
            Ok(T::default()) // Return default value (usually 0) if empty
        } else {
            Ok(column.iter().copied().sum()) // Use the sum of the iterator (T: Sum<T> + Copy)
        }
    }

    // Calculate the mean of the specified column
    pub fn mean(&self, column_name: &str) -> Result<T, Box<dyn Error>> {
        let column = self
            .get_column(column_name)
            .ok_or_else(|| format!("Column '{column_name}' not found for mean()"))?;
        let n = column.len(); // Number of rows (usize)
        if n == 0 {
            return Err(format!("Cannot calculate mean of an empty column '{column_name}'").into());
        }
        // Calculate the sum
        let sum_val = self.sum(column_name)?;

        // Safely cast the number of rows (usize) to type T (T: num_traits::NumCast)
        let n_t = cast::<usize, T>(n).ok_or_else(|| {
            format!(
                "Failed to cast column length {} (usize) to the required numeric type '{}'",
                n,
                std::any::type_name::<T>() // T: 'static
            )
        })?;

        // Optional: Check if the cast value is zero (T: num_traits::Zero is needed)
        // use num_traits::Zero;
        // if n_t.is_zero() && n != 0 {
        //     return Err("Division by zero occurred after casting column length to type T".into());
        // }

        // Sum / Number of rows (T: Div<Output = T>)
        Ok(sum_val / n_t)
    }
}

// Extension for DataFrame limited to string type (String)
impl DataFrame<String> {
    // String satisfies Debug + Clone + Eq + Hash + FromStr + 'static
    // Concatenate the strings of the specified column
    pub fn concat(&self, column_name: &str, separator: &str) -> Result<String, Box<dyn Error>> {
        let column = self
            .get_column(column_name)
            .ok_or_else(|| format!("Column '{column_name}' not found for concat()"))?;
        Ok(column.join(separator)) // Use the join method of Vec<String>
    }
}

// Function to read DataFrame from a file
pub fn read_csv<T: Debug + Clone + FromStr + 'static>(
    file_path: &Path,
    has_header: bool,
) -> Result<DataFrame<T>, Box<dyn Error>> {
    // Open the file
    let file = File::open(file_path)
        .map_err(|e| format!("Failed to open file '{}': {}", file_path.display(), e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines(); // Line iterator

    // For storing data
    let mut data: HashMap<String, Vec<T>> = HashMap::new();
    let mut columns: Vec<String> = Vec::new(); // Maintain column order
    let mut row_count = 0;
    let mut expected_columns = 0; // Expected number of columns

    // Process the header line
    if has_header {
        if let Some(header_line_result) = lines.next() {
            let header_line = header_line_result.map_err(|e| {
                format!(
                    "Failed to read header line from '{}': {}",
                    file_path.display(),
                    e
                )
            })?;
            // Split the header by commas, trim, and create a list of column names
            columns = header_line
                .split(',')
                .map(|s| s.trim().to_string())
                .collect();
            // Return an error if the header is empty or invalid
            if columns.is_empty() || columns.iter().all(String::is_empty) {
                return Err(format!(
                    "CSV file '{}' has an empty or invalid header",
                    file_path.display()
                )
                .into());
            }
            expected_columns = columns.len();
            // Initialize Vec for storing data for each column
            for column in &columns {
                data.insert(column.clone(), Vec::new());
            }
        } else {
            // If the file is empty but a header is expected
            return Err(format!(
                "CSV file '{}' is empty or failed to read header",
                file_path.display()
            )
            .into());
        }
    }

    // Process the data lines
    for (line_idx, line_result) in lines.enumerate() {
        let line_number = line_idx + if has_header { 2 } else { 1 }; // Line number for error display (1-based)
        let line = line_result.map_err(|e| {
            format!(
                "Failed to read line {} from '{}': {}",
                line_number,
                file_path.display(),
                e
            )
        })?;

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        // Split the line by commas and trim
        let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

        // If there is no header, set column information from the first data line
        if !has_header && row_count == 0 {
            expected_columns = values.len();
            if expected_columns == 0 {
                return Err(format!(
                    "First data line (line {line_number}) is empty, cannot determine columns"
                )
                .into());
            }
            // Generate column names as "column_0", "column_1", ...
            columns = (0..expected_columns)
                .map(|i| format!("column_{i}"))
                .collect();
            // Initialize Vec for storing data
            for column_name in &columns {
                data.insert(column_name.clone(), Vec::new());
            }
        }

        // Check if the number of columns matches the expected value
        if values.len() != expected_columns {
            return Err(format!(
                "Row {} has {} values, but expected {} columns",
                line_number,
                values.len(),
                expected_columns
            )
            .into());
        }

        // Parse each value and add it to the corresponding column's Vec
        for (col_idx, value_str) in values.iter().enumerate() {
            let column_name = &columns[col_idx];
            // Parse the string to type T (T: FromStr)
            let parsed_value = value_str.parse::<T>().map_err(|_| {
                format!(
                    "Failed to parse value '{}' in column '{}' (row {}) as type '{}'",
                    value_str,
                    column_name,
                    line_number,
                    std::any::type_name::<T>()
                ) // T: 'static
            })?;
            // Add the value to the Vec in the data HashMap (unwrap is safe because the key should exist)
            data.get_mut(column_name).unwrap().push(parsed_value);
        }
        row_count += 1; // Count the number of processed rows
    }

    // Final check after reading the CSV
    if has_header && row_count == 0 {
        // If there is a header but no data rows -> Return an empty DataFrame with only the header
    }
    if !has_header && row_count == 0 {
        // If there is no header and no data rows -> Return a completely empty DataFrame
        return Ok(DataFrame::new());
    }

    // Construct and return the final DataFrame
    Ok(DataFrame {
        data,
        columns,
        row_count,
    })
}

// --- main function ---
fn main() -> Result<(), Box<dyn Error>> {
    // === i32 DataFrame ===
    println!("--- i32 DataFrame Example ---");
    let mut df_i32: DataFrame<i32> = DataFrame::new();
    df_i32.add_row(HashMap::from([
        ("id".to_string(), 1),
        ("age".to_string(), 30),
        ("height".to_string(), 180),
    ]))?;
    df_i32.add_row(HashMap::from([
        ("id".to_string(), 2),
        ("age".to_string(), 25),
        ("height".to_string(), 175),
    ]))?;
    df_i32.add_row(HashMap::from([
        ("id".to_string(), 3),
        ("age".to_string(), 30),
        ("height".to_string(), 185),
    ]))?;
    println!("DataFrame (i32):\n{df_i32:?}");
    println!("Sum of age: {:?}", df_i32.sum("age")?);
    println!("Mean of age (using num_cast): {:?}", df_i32.mean("age")?); // Uses safe cast
    let filtered_df_i32 =
        df_i32.filter(|row| **row.get(&"age".to_string()).expect("Age missing") > 25)?;
    println!("Filtered DataFrame (age > 25):\n{filtered_df_i32:?}");
    let sorted_df_i32 = df_i32.sort_by(|a, b| {
        let age_a = **a.get(&"age".to_string()).expect("Age missing");
        let age_b = **b.get(&"age".to_string()).expect("Age missing");
        match age_a.cmp(&age_b) {
            std::cmp::Ordering::Equal => {
                let height_a = **a.get(&"height".to_string()).expect("Height missing");
                let height_b = **b.get(&"height".to_string()).expect("Height missing");
                height_b.cmp(&height_a) // height desc
            }
            other => other,
        }
    });
    println!("Sorted DataFrame (by age asc, height desc):\n{sorted_df_i32:?}");
    println!("-----------------------------");
    println!();

    // === String DataFrame ===
    println!("--- String DataFrame Example ---");
    let mut df_string = DataFrame::<String>::new();
    df_string.add_row(HashMap::from([
        ("name".to_string(), "Alice".to_string()),
        ("city".to_string(), "New York".to_string()),
    ]))?;
    df_string.add_row(HashMap::from([
        ("name".to_string(), "Bob".to_string()),
        ("city".to_string(), "Los Angeles".to_string()),
    ]))?;
    df_string.add_row(HashMap::from([
        ("name".to_string(), "Charlie".to_string()),
        ("city".to_string(), "New York".to_string()),
    ]))?;
    let mut df_string2 = DataFrame::<String>::new();
    df_string2.add_row(HashMap::from([
        ("name".to_string(), "Alice".to_string()),
        ("state".to_string(), "NY".to_string()),
        ("zip".to_string(), "10001".to_string()),
    ]))?;
    df_string2.add_row(HashMap::from([
        ("name".to_string(), "Bob".to_string()),
        ("state".to_string(), "CA".to_string()),
        ("zip".to_string(), "90001".to_string()),
    ]))?;
    df_string2.add_row(HashMap::from([
        ("name".to_string(), "David".to_string()),
        ("state".to_string(), "TX".to_string()),
        ("zip".to_string(), "75001".to_string()),
    ]))?;
    println!("DataFrame 1 (String):\n{df_string:?}");
    println!("DataFrame 2 (String):\n{df_string2:?}");
    let joined_df_string = df_string.join(&df_string2, "name")?;
    println!("Joined DataFrame (inner join on name):\n{joined_df_string:?}");
    println!("Concatenated names: {:?}", df_string.concat("name", ", ")?);
    println!("-----------------------------");
    println!();

    // === CSV Reading (f64) ===
    println!("--- CSV Reading Example (f64) ---");
    let file_path_str = "data_temp.csv";
    let file_path = Path::new(file_path_str);
    // Create dummy CSV
    {
        let mut file = File::create(file_path)?;
        writeln!(file, "value_a,value_b,label")?;
        writeln!(file, "1.1, 2.2, apple")?;
        writeln!(file, "3.3, 4.4, banana")?;
        writeln!(file, " 5.5 ,6.6, apple")?;
        writeln!(file)?; // empty line test
    }
    println!("Attempting to read: {file_path_str}");
    // Try reading as f64 (should fail on 'label' column)
    let df_from_csv_f64_result: Result<DataFrame<f64>, Box<dyn Error>> = read_csv(file_path, true);
    match df_from_csv_f64_result {
        Ok(df) => {
            println!("Successfully read CSV as f64 (This shouldn't happen):\n{df:?}");
        }
        Err(e) => {
            println!("Correctly failed to read CSV as f64:");
            eprintln!("Error: {e}");
        } // Expected
    }
    println!();
    // Try reading as String (should succeed)
    let df_from_csv_string_result: Result<DataFrame<String>, Box<dyn Error>> =
        read_csv(file_path, true);
    match df_from_csv_string_result {
        Ok(df) => {
            println!("Successfully read CSV as String:\n{df:?}");
            if df.contains_column("label") {
                println!("Concatenated labels: {:?}", df.concat("label", " | ")?);
            }
        }
        Err(e) => {
            eprintln!("Failed to read CSV as String (This shouldn't happen): {e}");
        }
    }
    // Clean up dummy file
    if let Err(e) = std::fs::remove_file(file_path) {
        eprintln!("Warning: Failed to remove temporary file '{file_path_str}': {e}");
    }
    println!("-----------------------------");

    Ok(()) // Main finishes successfully
}
