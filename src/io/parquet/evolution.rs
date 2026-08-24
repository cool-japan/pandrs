//! Schema evolution (rename/add/remove columns) and post-read predicate pushdown.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::series::Series;
use std::collections::HashMap;
use std::path::Path;

use super::core::read_parquet;

/// Predicate pushdown filters for efficient reading
#[derive(Debug, Clone)]
pub enum PredicateFilter {
    /// Equality filter: column = value
    Equals(String, String),
    /// Range filter: min <= column <= max
    Range(String, String, String),
    /// IN filter: column IN (values)
    In(String, Vec<String>),
    /// NOT NULL filter
    NotNull(String),
    /// Custom filter expression
    Custom(String),
}
/// Schema evolution support for Parquet files
#[derive(Debug, Clone)]
pub struct SchemaEvolution {
    /// Source schema (original)
    pub source_schema: String,
    /// Target schema (desired)
    pub target_schema: String,
    /// Column mappings (old_name -> new_name)
    pub column_mappings: HashMap<String, String>,
    /// Columns to add with default values
    pub columns_to_add: HashMap<String, String>,
    /// Columns to remove
    pub columns_to_remove: Vec<String>,
    /// Data type conversions
    pub type_conversions: HashMap<String, String>,
}
impl Default for SchemaEvolution {
    fn default() -> Self {
        Self {
            source_schema: String::new(),
            target_schema: String::new(),
            column_mappings: HashMap::new(),
            columns_to_add: HashMap::new(),
            columns_to_remove: Vec::new(),
            type_conversions: HashMap::new(),
        }
    }
}
/// Read Parquet file with schema evolution support
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `schema_evolution` - Schema evolution rules
///
/// # Returns
///
/// * `Result<DataFrame>` - DataFrame with evolved schema
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::{read_parquet_with_schema_evolution, SchemaEvolution};
/// use std::collections::HashMap;
///
/// let mut evolution = SchemaEvolution::default();
/// evolution.column_mappings.insert("old_name".to_string(), "new_name".to_string());
/// evolution.columns_to_add.insert("new_column".to_string(), "default_value".to_string());
///
/// let df = read_parquet_with_schema_evolution("data.parquet", evolution).expect("operation should succeed");
/// ```
pub fn read_parquet_with_schema_evolution(
    path: impl AsRef<Path>,
    schema_evolution: SchemaEvolution,
) -> Result<DataFrame> {
    // Read the original file
    let mut df = read_parquet(path.as_ref())?;

    // Apply schema evolution transformations
    apply_schema_evolution(&mut df, &schema_evolution)?;

    Ok(df)
}

/// Apply schema evolution rules to a DataFrame
pub(super) fn apply_schema_evolution(
    df: &mut DataFrame,
    evolution: &SchemaEvolution,
) -> Result<()> {
    // Apply column renames, preserving the source column's real data and type.
    for (old_name, new_name) in &evolution.column_mappings {
        if old_name == new_name {
            continue;
        }
        if df.contains_column(old_name) {
            let mut mapping = HashMap::new();
            mapping.insert(old_name.clone(), new_name.clone());
            df.rename_columns(&mapping)?;
        }
    }

    // Drop columns, preserving the type and order of every remaining column.
    // Column names not present are ignored (best-effort, matching the
    // existing rename/add rules above).
    if !evolution.columns_to_remove.is_empty() {
        *df = drop_columns_preserving_type(df, &evolution.columns_to_remove)?;
    }

    // Add new columns with default values
    for (col_name, default_value) in &evolution.columns_to_add {
        let row_count = df.row_count();
        let default_values = vec![default_value.clone(); row_count];
        let series = Series::new(default_values, Some(col_name.clone()))?;
        df.add_column(col_name.clone(), series)?;
    }

    // Type conversions would require DataFrame API extensions
    // This is a placeholder for enhanced type conversion support

    Ok(())
}

/// Remove the named columns from a DataFrame, preserving the type and
/// left-to-right order of every remaining column.
///
/// `DataFrame` has no built-in "drop column" primitive: the crate's own
/// `select_columns` silently widens `i64` columns to `f64` (it always reads
/// through `get_column_numeric_values`), which would corrupt the type of
/// every *untouched* column just because some other column was removed.
/// This instead re-inserts each kept column through the crate's typed
/// `get_column::<T>` accessor, trying every concrete type a Parquet-sourced
/// `DataFrame` can actually contain.
fn drop_columns_preserving_type(df: &DataFrame, names: &[String]) -> Result<DataFrame> {
    let mut result = DataFrame::new();

    for col_name in df.column_names() {
        if names.iter().any(|n| n == col_name) {
            continue;
        }

        if let Ok(series) = df.get_column::<i64>(col_name) {
            result.add_column(col_name.clone(), series.clone())?;
        } else if let Ok(series) = df.get_column::<f64>(col_name) {
            result.add_column(col_name.clone(), series.clone())?;
        } else if let Ok(series) = df.get_column::<bool>(col_name) {
            result.add_column(col_name.clone(), series.clone())?;
        } else if let Ok(series) = df.get_column::<String>(col_name) {
            result.add_column(col_name.clone(), series.clone())?;
        } else {
            return Err(Error::Type(format!(
                "Column '{}' has a type not supported by schema-evolution column removal",
                col_name
            )));
        }
    }

    Ok(result)
}

/// Read Parquet file with predicate pushdown for efficient filtering
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
/// * `predicates` - Predicate filters to apply
///
/// # Returns
///
/// * `Result<DataFrame>` - Filtered DataFrame
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::{read_parquet_with_predicates, PredicateFilter};
///
/// let predicates = vec![
///     PredicateFilter::Equals("status".to_string(), "active".to_string()),
///     PredicateFilter::Range("age".to_string(), "18".to_string(), "65".to_string()),
/// ];
///
/// let df = read_parquet_with_predicates("data.parquet", predicates).expect("operation should succeed");
/// ```
pub fn read_parquet_with_predicates(
    path: impl AsRef<Path>,
    predicates: Vec<PredicateFilter>,
) -> Result<DataFrame> {
    // For now, read the full file and apply filters post-read
    // True predicate pushdown would require deeper Arrow integration
    let df = read_parquet(path.as_ref())?;

    // Apply filters (simplified implementation)
    apply_predicate_filters(df, &predicates)
}

/// Numeric-aware cell equality: equal as strings, or equal once both parse as
/// floating-point numbers (so `"30"` matches `"30.0"`).
fn predicate_cell_equals(cell: &str, target: &str) -> bool {
    if cell == target {
        return true;
    }
    matches!(
        (cell.trim().parse::<f64>(), target.trim().parse::<f64>()),
        (Ok(a), Ok(b)) if a == b
    )
}

/// Apply predicate filters to an already-read DataFrame.
///
/// This is a post-read filter (not Parquet-level pushdown): every predicate is
/// evaluated against the materialised rows and only the matching rows are kept.
/// `Custom` expression predicates are not supported and are reported honestly.
pub(super) fn apply_predicate_filters(
    df: DataFrame,
    predicates: &[PredicateFilter],
) -> Result<DataFrame> {
    if predicates.is_empty() {
        return Ok(df);
    }

    let row_count = df.row_count();
    let mut keep = vec![true; row_count];

    for predicate in predicates {
        match predicate {
            PredicateFilter::Equals(column, value) => {
                let values = df.get_column_string_values(column)?;
                for (i, cell) in values.iter().enumerate() {
                    if !predicate_cell_equals(cell, value) {
                        keep[i] = false;
                    }
                }
            }
            PredicateFilter::Range(column, min, max) => {
                let values = df.get_column_string_values(column)?;
                let min_f = min.trim().parse::<f64>().ok();
                let max_f = max.trim().parse::<f64>().ok();
                for (i, cell) in values.iter().enumerate() {
                    let in_range = match cell.trim().parse::<f64>() {
                        Ok(v) => {
                            min_f.map_or(true, |lo| v >= lo) && max_f.map_or(true, |hi| v <= hi)
                        }
                        // Non-numeric cell: fall back to lexicographic comparison.
                        Err(_) => cell.as_str() >= min.as_str() && cell.as_str() <= max.as_str(),
                    };
                    if !in_range {
                        keep[i] = false;
                    }
                }
            }
            PredicateFilter::In(column, allowed) => {
                let values = df.get_column_string_values(column)?;
                for (i, cell) in values.iter().enumerate() {
                    if !allowed.iter().any(|a| predicate_cell_equals(cell, a)) {
                        keep[i] = false;
                    }
                }
            }
            PredicateFilter::NotNull(column) => {
                let values = df.get_column_string_values(column)?;
                for (i, cell) in values.iter().enumerate() {
                    let trimmed = cell.trim();
                    if trimmed.is_empty()
                        || trimmed.eq_ignore_ascii_case("null")
                        || trimmed.eq_ignore_ascii_case("nan")
                    {
                        keep[i] = false;
                    }
                }
            }
            PredicateFilter::Custom(expression) => {
                return Err(Error::NotImplemented(format!(
                    "Custom predicate expression '{}' is not supported",
                    expression
                )));
            }
        }
    }

    let kept_indices: Vec<usize> = (0..row_count).filter(|&i| keep[i]).collect();
    df.sample(&kept_indices)
}
