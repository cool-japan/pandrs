//! Out-of-core join operations for large datasets.
//!
//! Implements a hash-join where the right side must fit in memory and the
//! left side is streamed through in chunks from disk.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use csv::ReaderBuilder;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::large::out_of_core::{concat_dataframes, OutOfCoreConfig, OutOfCoreWriter};
use crate::series::Series;

// ---------------------------------------------------------------------------
// JoinType
// ---------------------------------------------------------------------------

/// Join strategy for out-of-core hash join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Keep only rows that match in both tables.
    Inner,
    /// Keep all rows from the left table; fill missing right values with empty
    /// strings.
    Left,
    /// Keep all rows from the right table; fill missing left values with empty
    /// strings.
    Right,
}

// ---------------------------------------------------------------------------
// hash_join_out_of_core
// ---------------------------------------------------------------------------

/// Hash join where `right` must fit in memory.
///
/// The left side is read from `left_path` in chunks; each chunk is joined
/// against the in-memory hash table built from `right`.  Result chunks are
/// written to temporary CSV files which are returned in an `OutOfCoreWriter`.
pub fn hash_join_out_of_core(
    left_path: &str,
    right: &DataFrame,
    left_key: &str,
    right_key: &str,
    join_type: JoinType,
    config: &OutOfCoreConfig,
) -> Result<OutOfCoreWriter> {
    // Validate right key exists
    if !right.contains_column(right_key) {
        return Err(Error::Column(format!(
            "Right key column '{}' does not exist",
            right_key
        )));
    }

    // Build hash table: right_key_value -> list of right rows (as Vec<String>)
    let right_col_names = right.column_names();
    let right_row_count = right.row_count();
    // Map: key value -> Vec<(col_name, value)>
    let mut right_map: HashMap<String, Vec<Vec<String>>> = HashMap::new();

    for row_idx in 0..right_row_count {
        let key_val = right
            .get_string_value(right_key, row_idx)
            .unwrap_or("")
            .to_string();
        let row: Vec<String> = right_col_names
            .iter()
            .map(|col| {
                right
                    .get_string_value(col, row_idx)
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        right_map.entry(key_val).or_default().push(row);
    }

    // Track which right keys were matched (for Right join)
    let mut right_matched: HashMap<String, bool> =
        right_map.keys().map(|k| (k.clone(), false)).collect();

    // Open left CSV
    let left_file = File::open(left_path).map_err(|e| Error::IoError(e.to_string()))?;
    let mut left_rdr = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(BufReader::new(left_file));

    let left_headers: Vec<String> = left_rdr
        .headers()
        .map_err(|e| Error::CsvError(e.to_string()))?
        .iter()
        .map(|h| h.to_string())
        .collect();

    // Validate left key exists
    if !left_headers.contains(&left_key.to_string()) {
        return Err(Error::Column(format!(
            "Left key column '{}' does not exist",
            left_key
        )));
    }

    let left_key_idx = left_headers
        .iter()
        .position(|h| h == left_key)
        .ok_or_else(|| Error::Column(format!("Left key '{}' not found", left_key)))?;

    // Build output column names: left cols + right cols (excluding right key to
    // avoid duplication unless left key != right key)
    let right_non_key_cols: Vec<String> = right_col_names
        .iter()
        .filter(|c| *c != right_key)
        .cloned()
        .collect();

    let mut output_col_names: Vec<String> = left_headers.clone();
    for c in &right_non_key_cols {
        // Avoid collision with left col names by suffixing if needed
        if output_col_names.contains(c) {
            output_col_names.push(format!("{}_right", c));
        } else {
            output_col_names.push(c.clone());
        }
    }

    let right_non_key_indices: Vec<usize> = right_col_names
        .iter()
        .enumerate()
        .filter(|(_, c)| *c != right_key)
        .map(|(i, _)| i)
        .collect();

    // Process left in chunks
    let chunk_size = config.chunk_size;
    let temp_dir = &config.temp_dir;
    let mut chunk_index = 0usize;
    let mut output_chunk_paths: Vec<PathBuf> = Vec::new();

    // Buffer left rows
    let mut left_rows: Vec<Vec<String>> = Vec::with_capacity(chunk_size);

    let process_chunk = |left_rows: &[Vec<String>],
                         right_map: &HashMap<String, Vec<Vec<String>>>,
                         right_matched: &mut HashMap<String, bool>,
                         left_key_idx: usize,
                         right_non_key_indices: &[usize],
                         output_col_names: &[String],
                         join_type: JoinType,
                         temp_dir: &Path,
                         chunk_index: &mut usize|
     -> Result<PathBuf> {
        let mut output_rows: Vec<Vec<String>> = Vec::new();
        let n_right_extra = right_non_key_indices.len();

        for left_row in left_rows {
            let key_val = left_row.get(left_key_idx).map(|s| s.as_str()).unwrap_or("");

            match right_map.get(key_val) {
                Some(matching_right_rows) => {
                    if let Some(matched) = right_matched.get_mut(key_val) {
                        *matched = true;
                    }
                    for right_row in matching_right_rows {
                        let mut out_row = left_row.clone();
                        for &ri in right_non_key_indices {
                            out_row.push(right_row.get(ri).cloned().unwrap_or_default());
                        }
                        output_rows.push(out_row);
                    }
                }
                None => {
                    if join_type == JoinType::Left {
                        // Keep left row, fill right with empty
                        let mut out_row = left_row.clone();
                        for _ in 0..n_right_extra {
                            out_row.push(String::new());
                        }
                        output_rows.push(out_row);
                    }
                    // Inner: skip
                }
            }
        }

        // Write chunk to temp file
        let path = temp_dir.join(format!("pandrs_join_chunk_{}.csv", *chunk_index));
        let f = File::create(&path).map_err(|e| Error::IoError(e.to_string()))?;
        let mut wtr = csv::WriterBuilder::new().from_writer(std::io::BufWriter::new(f));
        wtr.write_record(output_col_names)
            .map_err(|e| Error::CsvError(e.to_string()))?;
        for row in &output_rows {
            wtr.write_record(row)
                .map_err(|e| Error::CsvError(e.to_string()))?;
        }
        wtr.flush().map_err(|e| Error::IoError(e.to_string()))?;
        *chunk_index += 1;
        Ok(path)
    };

    for record_result in left_rdr.records() {
        let record = record_result.map_err(|e| Error::CsvError(e.to_string()))?;
        let row: Vec<String> = record.iter().map(|f| f.to_string()).collect();
        left_rows.push(row);
        if left_rows.len() >= chunk_size {
            let p = process_chunk(
                &left_rows,
                &right_map,
                &mut right_matched,
                left_key_idx,
                &right_non_key_indices,
                &output_col_names,
                join_type,
                temp_dir,
                &mut chunk_index,
            )?;
            output_chunk_paths.push(p);
            left_rows.clear();
        }
    }

    if !left_rows.is_empty() {
        let p = process_chunk(
            &left_rows,
            &right_map,
            &mut right_matched,
            left_key_idx,
            &right_non_key_indices,
            &output_col_names,
            join_type,
            temp_dir,
            &mut chunk_index,
        )?;
        output_chunk_paths.push(p);
        left_rows.clear();
    }

    // Right join: append unmatched right rows
    if join_type == JoinType::Right {
        let mut unmatched_rows: Vec<Vec<String>> = Vec::new();
        let n_left_cols = left_headers.len();

        for (key_val, matched) in &right_matched {
            if !matched {
                if let Some(right_rows) = right_map.get(key_val) {
                    for right_row in right_rows {
                        let mut out_row: Vec<String> = vec![String::new(); n_left_cols];
                        // Fill left key column with the key value
                        out_row[left_key_idx] = key_val.clone();
                        for &ri in &right_non_key_indices {
                            out_row.push(right_row.get(ri).cloned().unwrap_or_default());
                        }
                        unmatched_rows.push(out_row);
                    }
                }
            }
        }

        if !unmatched_rows.is_empty() {
            let path = temp_dir.join(format!("pandrs_join_chunk_{}.csv", chunk_index));
            let f = File::create(&path).map_err(|e| Error::IoError(e.to_string()))?;
            let mut wtr = csv::WriterBuilder::new().from_writer(std::io::BufWriter::new(f));
            wtr.write_record(&output_col_names)
                .map_err(|e| Error::CsvError(e.to_string()))?;
            for row in &unmatched_rows {
                wtr.write_record(row)
                    .map_err(|e| Error::CsvError(e.to_string()))?;
            }
            wtr.flush().map_err(|e| Error::IoError(e.to_string()))?;
            output_chunk_paths.push(path);
        }
    }

    Ok(OutOfCoreWriter {
        chunks: output_chunk_paths,
        config: config.clone(),
    })
}
