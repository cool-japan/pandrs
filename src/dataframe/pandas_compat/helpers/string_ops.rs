//! String operation helper functions

use crate::core::error::Result;
use crate::dataframe::DataFrame;
use crate::series::Series;

/// Convert string column to lowercase
pub fn str_lower(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let vals = df.get_column_string_values(column)?;
    let lowered: Vec<String> = vals.iter().map(|s| s.to_lowercase()).collect();

    let mut result = DataFrame::new();
    for col in df.column_names() {
        if col == column {
            result.add_column(
                col.clone(),
                Series::new(lowered.clone(), Some(col.clone()))?,
            )?;
        } else if let Ok(v) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        } else if let Ok(v) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        }
    }

    Ok(result)
}

/// Convert string column to uppercase
pub fn str_upper(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let vals = df.get_column_string_values(column)?;
    let uppered: Vec<String> = vals.iter().map(|s| s.to_uppercase()).collect();

    let mut result = DataFrame::new();
    for col in df.column_names() {
        if col == column {
            result.add_column(
                col.clone(),
                Series::new(uppered.clone(), Some(col.clone()))?,
            )?;
        } else if let Ok(v) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        } else if let Ok(v) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        }
    }

    Ok(result)
}

/// Strip whitespace from string column
pub fn str_strip(df: &DataFrame, column: &str) -> Result<DataFrame> {
    let vals = df.get_column_string_values(column)?;
    let stripped: Vec<String> = vals.iter().map(|s| s.trim().to_string()).collect();

    let mut result = DataFrame::new();
    for col in df.column_names() {
        if col == column {
            result.add_column(
                col.clone(),
                Series::new(stripped.clone(), Some(col.clone()))?,
            )?;
        } else if let Ok(v) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        } else if let Ok(v) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        }
    }

    Ok(result)
}

/// Check if string column contains pattern (returns boolean column)
pub fn str_contains(df: &DataFrame, column: &str, pattern: &str) -> Result<Vec<bool>> {
    let vals = df.get_column_string_values(column)?;
    Ok(vals.iter().map(|s| s.contains(pattern)).collect())
}

/// Replace pattern in string column
pub fn str_replace(
    df: &DataFrame,
    column: &str,
    pattern: &str,
    replacement: &str,
) -> Result<DataFrame> {
    let vals = df.get_column_string_values(column)?;
    let replaced: Vec<String> = vals
        .iter()
        .map(|s| s.replace(pattern, replacement))
        .collect();

    let mut result = DataFrame::new();
    for col in df.column_names() {
        if col == column {
            result.add_column(
                col.clone(),
                Series::new(replaced.clone(), Some(col.clone()))?,
            )?;
        } else if let Ok(v) = df.get_column_numeric_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        } else if let Ok(v) = df.get_column_string_values(&col) {
            result.add_column(col.clone(), Series::new(v, Some(col.clone()))?)?;
        }
    }

    Ok(result)
}

/// Split string column on delimiter
pub fn str_split(df: &DataFrame, column: &str, delimiter: &str) -> Result<Vec<Vec<String>>> {
    let vals = df.get_column_string_values(column)?;
    Ok(vals
        .iter()
        .map(|s| s.split(delimiter).map(|p| p.to_string()).collect())
        .collect())
}

/// Get length of strings in column
pub fn str_len(df: &DataFrame, column: &str) -> Result<Vec<usize>> {
    let vals = df.get_column_string_values(column)?;
    Ok(vals.iter().map(|s| s.len()).collect())
}

/// Check if string column starts with prefix
pub fn str_startswith(df: &DataFrame, column: &str, prefix: &str) -> Result<Vec<bool>> {
    let values = df.get_column_string_values(column)?;
    Ok(values.iter().map(|s| s.starts_with(prefix)).collect())
}

/// Check if string column ends with suffix
pub fn str_endswith(df: &DataFrame, column: &str, suffix: &str) -> Result<Vec<bool>> {
    let values = df.get_column_string_values(column)?;
    Ok(values.iter().map(|s| s.ends_with(suffix)).collect())
}

/// Pad strings on the left to specified width
pub fn str_pad_left(
    df: &DataFrame,
    column: &str,
    width: usize,
    fillchar: char,
) -> Result<DataFrame> {
    let values = df.get_column_string_values(column)?;
    let padded: Vec<String> = values
        .iter()
        .map(|s| {
            if s.len() >= width {
                s.clone()
            } else {
                format!("{}{}", fillchar.to_string().repeat(width - s.len()), s)
            }
        })
        .collect();

    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(padded.clone(), Some(col_name))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

/// Pad strings on the right to specified width
pub fn str_pad_right(
    df: &DataFrame,
    column: &str,
    width: usize,
    fillchar: char,
) -> Result<DataFrame> {
    let values = df.get_column_string_values(column)?;
    let padded: Vec<String> = values
        .iter()
        .map(|s| {
            if s.len() >= width {
                s.clone()
            } else {
                format!("{}{}", s, fillchar.to_string().repeat(width - s.len()))
            }
        })
        .collect();

    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(padded.clone(), Some(col_name))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

/// Slice strings from start to end position
pub fn str_slice(
    df: &DataFrame,
    column: &str,
    start: usize,
    end: Option<usize>,
) -> Result<DataFrame> {
    let values = df.get_column_string_values(column)?;
    let sliced: Vec<String> = values
        .iter()
        .map(|s| {
            let end_idx = end.unwrap_or(s.len()).min(s.len());
            let start_idx = start.min(s.len());
            s.chars()
                .skip(start_idx)
                .take(end_idx - start_idx)
                .collect()
        })
        .collect();

    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(sliced.clone(), Some(col_name))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

/// Count occurrences of a pattern in string column
pub fn str_count(df: &DataFrame, column: &str, pattern: &str) -> Result<Vec<usize>> {
    let values = df.get_column_string_values(column)?;
    Ok(values.iter().map(|s| s.matches(pattern).count()).collect())
}

/// Repeat strings n times
pub fn str_repeat(df: &DataFrame, column: &str, n: usize) -> Result<DataFrame> {
    let values = df.get_column_string_values(column)?;
    let repeated: Vec<String> = values.iter().map(|s| s.repeat(n)).collect();

    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(repeated.clone(), Some(col_name))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

/// Center strings in width with fillchar
pub fn str_center(df: &DataFrame, column: &str, width: usize, fillchar: char) -> Result<DataFrame> {
    let values = df.get_column_string_values(column)?;
    let centered: Vec<String> = values
        .iter()
        .map(|s| {
            if s.len() >= width {
                s.clone()
            } else {
                let padding = width - s.len();
                let left = padding / 2;
                let right = padding - left;
                format!(
                    "{}{}{}",
                    fillchar.to_string().repeat(left),
                    s,
                    fillchar.to_string().repeat(right)
                )
            }
        })
        .collect();

    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(centered.clone(), Some(col_name))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}

/// Zero-fill strings to width
pub fn str_zfill(df: &DataFrame, column: &str, width: usize) -> Result<DataFrame> {
    let values = df.get_column_string_values(column)?;
    let filled: Vec<String> = values
        .iter()
        .map(|s| {
            if s.len() >= width {
                s.clone()
            } else {
                format!("{}{}", "0".repeat(width - s.len()), s)
            }
        })
        .collect();

    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        if &col_name == column {
            result.add_column(
                col_name.clone(),
                Series::new(filled.clone(), Some(col_name))?,
            )?;
        } else if let Ok(vals) = df.get_column_numeric_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        } else if let Ok(vals) = df.get_column_string_values(&col_name) {
            result.add_column(col_name.clone(), Series::new(vals, Some(col_name.clone()))?)?;
        }
    }
    Ok(result)
}
