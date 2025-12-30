use csv::{ReaderBuilder, Writer};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::error::{PandRSError, Result};
use crate::series::{CategoricalOrder, Series, StringCategorical};
use crate::DataFrame;

/// Read a DataFrame from a CSV file
pub fn read_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<DataFrame> {
    let file = File::open(path.as_ref()).map_err(PandRSError::Io)?;

    // Set up the CSV reader
    let mut rdr = ReaderBuilder::new()
        .has_headers(has_header)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(file);

    let mut df = DataFrame::new();

    // Get the header row
    let headers: Vec<String> = if has_header {
        rdr.headers()
            .map_err(PandRSError::Csv)?
            .iter()
            .map(|h| h.to_string())
            .collect()
    } else {
        // If there is no header, infer from the first row and use "column_0", "column_1", etc.
        if let Some(first_record_result) = rdr.records().next() {
            let first_record = first_record_result.map_err(PandRSError::Csv)?;
            (0..first_record.len())
                .map(|i| format!("column_{}", i))
                .collect()
        } else {
            // If the file is empty
            return Ok(DataFrame::new());
        }
    };

    // Collect data for each column
    let mut columns: HashMap<String, Vec<String>> = HashMap::new();
    for header in &headers {
        columns.insert(header.clone(), Vec::new());
    }

    // Process each row
    for result in rdr.records() {
        let record = result.map_err(PandRSError::Csv)?;
        for (i, header) in headers.iter().enumerate() {
            if i < record.len() {
                columns.get_mut(header).unwrap().push(record[i].to_string());
            } else {
                // If the row is shorter, add an empty string
                columns.get_mut(header).unwrap().push(String::new());
            }
        }
    }

    // Add columns to the DataFrame
    for header in headers {
        if let Some(values) = columns.remove(&header) {
            let series = Series::new(values, Some(header.clone()))?;
            df.add_column(header, series)?;
        }
    }

    Ok(df)
}

/// Write a DataFrame to a CSV file
pub fn write_csv<P: AsRef<Path>>(df: &DataFrame, path: P) -> Result<()> {
    let file = File::create(path.as_ref()).map_err(PandRSError::Io)?;
    let mut wtr = Writer::from_writer(file);

    // Write the header row
    wtr.write_record(df.column_names())
        .map_err(PandRSError::Csv)?;

    // Write each row of data
    let row_count = df.row_count();

    // If there are no rows, flush and return
    if row_count == 0 {
        wtr.flush().map_err(PandRSError::Io)?;
        return Ok(());
    }

    for i in 0..row_count {
        let mut row = Vec::new();

        for col_name in df.column_names() {
            // This is a stub implementation since we don't have the full Series functionality yet
            // In a real implementation, we would get values from the series
            row.push(String::new());
        }

        wtr.write_record(&row).map_err(PandRSError::Csv)?;
    }

    wtr.flush().map_err(PandRSError::Io)?;
    Ok(())
}
