use std::collections::HashMap;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::Series;

use crate::plugins::traits::{PluginMetadata, PluginType, TransformPlugin};

// ---------------------------------------------------------------------------
// FilterTransformPlugin
// ---------------------------------------------------------------------------

/// Transform plugin that filters rows based on a column condition.
///
/// Options:
/// - `column`: column name to filter on (required)
/// - `operator`: one of "gt", "lt", "eq", "ne", "gte", "lte", "contains" (required)
/// - `value`: the comparison value as a string (required)
pub struct FilterTransformPlugin {
    metadata: PluginMetadata,
}

impl FilterTransformPlugin {
    pub fn new() -> Self {
        FilterTransformPlugin {
            metadata: PluginMetadata {
                name: "filter".to_string(),
                version: "1.0.0".to_string(),
                description: "Filter DataFrame rows based on a column condition".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::Transform,
                capabilities: vec![
                    "filter".to_string(),
                    "gt".to_string(),
                    "lt".to_string(),
                    "eq".to_string(),
                    "contains".to_string(),
                ],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for FilterTransformPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformPlugin for FilterTransformPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn transform(&self, df: DataFrame, options: &HashMap<String, String>) -> Result<DataFrame> {
        let column = options.get("column").ok_or_else(|| {
            Error::InvalidInput("filter: 'column' option is required".to_string())
        })?;
        let operator = options.get("operator").ok_or_else(|| {
            Error::InvalidInput("filter: 'operator' option is required".to_string())
        })?;
        let value = options
            .get("value")
            .ok_or_else(|| Error::InvalidInput("filter: 'value' option is required".to_string()))?;

        if !df.contains_column(column) {
            return Err(Error::ColumnNotFound(column.clone()));
        }

        let row_count = df.row_count();
        if row_count == 0 {
            return Ok(df);
        }

        // Build a boolean mask of which rows to keep
        let keep_indices = build_filter_mask(&df, column, operator, value)?;

        if keep_indices.is_empty() {
            // Return empty DataFrame with same schema
            return build_empty_like(&df);
        }

        df.sample(&keep_indices)
    }
}

/// Operators `build_filter_mask` understands.
const KNOWN_FILTER_OPERATORS: &[&str] = &["gt", "lt", "gte", "lte", "eq", "ne", "contains"];

/// Build a list of row indices that satisfy the filter condition.
///
/// The operator is validated up front against [`KNOWN_FILTER_OPERATORS`]:
/// previously an unknown operator reaching the *numeric* comparison branch
/// fell through its `match`'s `_ => false` arm for every row, so the filter
/// silently returned zero rows for a typo'd operator instead of erroring
/// (the string-comparison branch, reached only when `value` didn't parse as
/// a number, already errored correctly on the same case -- the two branches
/// disagreed). `contains` is also routed to the string comparison
/// unconditionally now, even when `value` happens to parse as a number
/// (e.g. filtering a phone-number-like column for the substring `"5"`):
/// `contains` is inherently a string operation, so silently taking the
/// numeric branch for it always returned zero rows too.
fn build_filter_mask(
    df: &DataFrame,
    column: &str,
    operator: &str,
    value: &str,
) -> Result<Vec<usize>> {
    if !KNOWN_FILTER_OPERATORS.contains(&operator) {
        return Err(Error::InvalidInput(format!(
            "filter: unknown operator '{}' (expected one of {:?})",
            operator, KNOWN_FILTER_OPERATORS
        )));
    }

    let mut indices = Vec::new();

    // Try numeric comparison first (never for `contains`, which is
    // string-only regardless of what `value` looks like).
    if operator != "contains" {
        if let Ok(num_val) = value.parse::<f64>() {
            if let Ok(col_values) = df.get_column_numeric_values(column) {
                for (i, v) in col_values.iter().enumerate() {
                    let keep = match operator {
                        "gt" => *v > num_val,
                        "lt" => *v < num_val,
                        "gte" => *v >= num_val,
                        "lte" => *v <= num_val,
                        "eq" => (*v - num_val).abs() < f64::EPSILON,
                        "ne" => (*v - num_val).abs() >= f64::EPSILON,
                        _ => false, // unreachable: operator already validated above
                    };
                    if keep {
                        indices.push(i);
                    }
                }
                return Ok(indices);
            }
        }
    }

    // Fall back to string comparison
    let col_values = df.get_column_string_values(column)?;
    let value_str: &str = value;
    for (i, v) in col_values.iter().enumerate() {
        let v_str: &str = v;
        let keep = match operator {
            "eq" => v_str == value_str,
            "ne" => v_str != value_str,
            "contains" => v_str.contains(value_str),
            "gt" => v_str > value_str,
            "lt" => v_str < value_str,
            "gte" => v_str >= value_str,
            "lte" => v_str <= value_str,
            _ => false, // unreachable: operator already validated above
        };
        if keep {
            indices.push(i);
        }
    }

    Ok(indices)
}

/// Copy a single column from `df` into `result`, preserving its concrete
/// element type (`i64`/`f64`/`i32`/`f32`/`bool`/`String`) exactly.
///
/// [`NormalizePlugin`] and [`FillNaPlugin`] both need to pass columns they
/// aren't touching straight through unchanged; routing every column through
/// `get_column_string_values` to do that (as they previously did) silently
/// turned every untouched column -- numeric or boolean included -- into a
/// `String` column, even when nothing about that column was normalized or
/// filled.
fn copy_column_typed(df: &DataFrame, col_name: &str, result: &mut DataFrame) -> Result<()> {
    macro_rules! try_copy {
        ($ty:ty) => {
            if let Ok(s) = df.get_column::<$ty>(col_name) {
                let series = Series::new(s.to_vec(), Some(col_name.to_string()))?;
                result.add_column(col_name.to_string(), series)?;
                return Ok(());
            }
        };
    }
    try_copy!(i64);
    try_copy!(f64);
    try_copy!(i32);
    try_copy!(f32);
    try_copy!(bool);
    try_copy!(String);

    Err(Error::NotImplemented(format!(
        "Column '{}' has an element type this plugin does not support copying \
         (supported: i64, f64, i32, f32, bool, String)",
        col_name
    )))
}

/// Build a single zero-row column in `result` under `col_name`, matching
/// `df`'s column of the same name's concrete element type exactly
/// (`i64`/`f64`/`i32`/`f32`/`bool`/`String`).
///
/// Sibling of [`copy_column_typed`], which copies *values*; this copies only
/// the type, producing an empty `Series<T>` of the same `T`.
fn empty_column_like(df: &DataFrame, col_name: &str, result: &mut DataFrame) -> Result<()> {
    macro_rules! try_empty {
        ($ty:ty) => {
            if df.get_column::<$ty>(col_name).is_ok() {
                let series: Series<$ty> = Series::new(Vec::new(), Some(col_name.to_string()))?;
                result.add_column(col_name.to_string(), series)?;
                return Ok(());
            }
        };
    }
    try_empty!(i64);
    try_empty!(f64);
    try_empty!(i32);
    try_empty!(f32);
    try_empty!(bool);
    try_empty!(String);

    Err(Error::NotImplemented(format!(
        "Column '{}' has an element type this plugin does not support copying \
         (supported: i64, f64, i32, f32, bool, String)",
        col_name
    )))
}

/// Build an empty DataFrame with the same column structure, preserving each
/// column's concrete element type exactly (see [`empty_column_like`]).
///
/// [`FilterTransformPlugin::transform`] uses this when a filter matches zero
/// rows. The previous implementation always produced `f64` or `String`
/// columns regardless of the source column's real type -- the same
/// blanket-recasting bug [`copy_column_typed`] exists to avoid on the
/// non-empty path. That meant an `i64`/`bool` column that happened to match
/// zero rows came back downcastable only as `f64`/`String`: "no rows
/// matched" silently became indistinguishable from "the column's type
/// changed" to any caller inspecting the result's schema.
fn build_empty_like(df: &DataFrame) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        empty_column_like(df, col_name.as_str(), &mut result)?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// SelectColumnsPlugin
// ---------------------------------------------------------------------------

/// Transform plugin that selects or drops specific columns.
///
/// Options:
/// - `columns`: comma-separated list of columns to select (mutually exclusive with `drop_columns`)
/// - `drop_columns`: comma-separated list of columns to drop
pub struct SelectColumnsPlugin {
    metadata: PluginMetadata,
}

impl SelectColumnsPlugin {
    pub fn new() -> Self {
        SelectColumnsPlugin {
            metadata: PluginMetadata {
                name: "select_columns".to_string(),
                version: "1.0.0".to_string(),
                description: "Select or drop columns from a DataFrame".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::Transform,
                capabilities: vec!["select".to_string(), "drop".to_string()],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for SelectColumnsPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformPlugin for SelectColumnsPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn transform(&self, df: DataFrame, options: &HashMap<String, String>) -> Result<DataFrame> {
        let all_cols = df.column_names();

        let select_cols: Vec<&str> = if let Some(cols_str) = options.get("columns") {
            cols_str.split(',').map(|s| s.trim()).collect()
        } else if let Some(drop_str) = options.get("drop_columns") {
            let to_drop: Vec<&str> = drop_str.split(',').map(|s| s.trim()).collect();
            all_cols
                .iter()
                .filter(|c| !to_drop.contains(&c.as_str()))
                .map(|c| c.as_str())
                .collect()
        } else {
            return Err(Error::InvalidInput(
                "select_columns: either 'columns' or 'drop_columns' option is required".to_string(),
            ));
        };

        df.select_columns(&select_cols)
    }
}

// ---------------------------------------------------------------------------
// NormalizePlugin
// ---------------------------------------------------------------------------

/// Transform plugin that applies min-max normalization to numeric columns.
///
/// Options:
/// - `columns`: comma-separated list of columns to normalize (default: all numeric columns)
pub struct NormalizePlugin {
    metadata: PluginMetadata,
}

impl NormalizePlugin {
    pub fn new() -> Self {
        NormalizePlugin {
            metadata: PluginMetadata {
                name: "normalize".to_string(),
                version: "1.0.0".to_string(),
                description: "Min-max normalize numeric columns".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::Transform,
                capabilities: vec!["normalize".to_string(), "min_max".to_string()],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for NormalizePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformPlugin for NormalizePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn transform(&self, df: DataFrame, options: &HashMap<String, String>) -> Result<DataFrame> {
        // Determine which columns to normalize
        let target_cols: Vec<String> = if let Some(cols_str) = options.get("columns") {
            cols_str.split(',').map(|s| s.trim().to_string()).collect()
        } else {
            // Default: all numeric columns
            df.column_names()
                .iter()
                .filter(|c| df.get_column_numeric_values(c.as_str()).is_ok())
                .cloned()
                .collect()
        };

        // Build the new DataFrame, normalizing target columns
        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            if target_cols.contains(col_name) {
                let values = df.get_column_numeric_values(col_name.as_str())?;
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                let normalized: Vec<f64> = if (max - min).abs() < f64::EPSILON {
                    // All values are the same; map to 0.0
                    vec![0.0; values.len()]
                } else {
                    values.iter().map(|v| (v - min) / (max - min)).collect()
                };

                let series = Series::new(normalized, Some(col_name.clone()))?;
                result.add_column(col_name.clone(), series)?;
            } else {
                // Preserve non-target columns exactly as they are.
                copy_column_typed(&df, col_name.as_str(), &mut result)?;
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// FillNaPlugin
// ---------------------------------------------------------------------------

/// Transform plugin that fills NA/missing values in a DataFrame.
///
/// Options:
/// - `value`: the fill value (required)
/// - `columns`: comma-separated list of columns to fill (default: all columns)
pub struct FillNaPlugin {
    metadata: PluginMetadata,
}

impl FillNaPlugin {
    pub fn new() -> Self {
        FillNaPlugin {
            metadata: PluginMetadata {
                name: "fill_na".to_string(),
                version: "1.0.0".to_string(),
                description: "Fill NA/missing values in DataFrame columns".to_string(),
                author: "PandRS".to_string(),
                plugin_type: PluginType::Transform,
                capabilities: vec!["fill".to_string(), "impute".to_string()],
            },
        }
    }

    pub fn arc() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for FillNaPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformPlugin for FillNaPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn transform(&self, df: DataFrame, options: &HashMap<String, String>) -> Result<DataFrame> {
        let fill_value = options
            .get("value")
            .ok_or_else(|| Error::InvalidInput("fill_na: 'value' option is required".to_string()))?
            .clone();

        let target_cols: Option<Vec<String>> = options
            .get("columns")
            .map(|cols_str| cols_str.split(',').map(|s| s.trim().to_string()).collect());

        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            let should_fill = target_cols
                .as_ref()
                .map(|cols| cols.contains(col_name))
                .unwrap_or(true);

            if !should_fill {
                copy_column_typed(&df, col_name.as_str(), &mut result)?;
                continue;
            }

            if let Ok(s) = df.get_column::<f64>(col_name.as_str()) {
                // `f64` is the only numeric column type in this DataFrame
                // model that can hold a missing value (`f64::NAN`); fill
                // NaN entries in place, keeping the column `f64`.
                let parsed_fill: f64 = fill_value.parse().map_err(|_| {
                    Error::InvalidInput(format!(
                        "fill_na: fill value '{}' is not a valid number for numeric column '{}'",
                        fill_value, col_name
                    ))
                })?;
                let filled: Vec<f64> = s
                    .to_vec()
                    .into_iter()
                    .map(|v| if v.is_nan() { parsed_fill } else { v })
                    .collect();
                let series = Series::new(filled, Some(col_name.clone()))?;
                result.add_column(col_name.clone(), series)?;
            } else if df.get_column::<i64>(col_name.as_str()).is_ok()
                || df.get_column::<i32>(col_name.as_str()).is_ok()
                || df.get_column::<f32>(col_name.as_str()).is_ok()
                || df.get_column::<bool>(col_name.as_str()).is_ok()
            {
                // None of these types can represent a missing value in this
                // DataFrame model (no NaN-equivalent for i64/i32/f32/bool),
                // so there is nothing to fill: copy through unchanged
                // rather than losing the column's real type by stringifying
                // a column that had no NA to begin with.
                copy_column_typed(&df, col_name.as_str(), &mut result)?;
            } else {
                let values: Vec<String> = df
                    .get_column_string_values(col_name.as_str())?
                    .into_iter()
                    .map(|v| {
                        if v.is_empty() || v == "null" || v == "NULL" || v == "NA" || v == "NaN" {
                            fill_value.clone()
                        } else {
                            v
                        }
                    })
                    .collect();
                let series = Series::new(values, Some(col_name.clone()))?;
                result.add_column(col_name.clone(), series)?;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mixed_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1i64, 2, 3], Some("id".to_string())).expect("series"),
        )
        .expect("add");
        df.add_column(
            "active".to_string(),
            Series::new(vec![true, false, true], Some("active".to_string())).expect("series"),
        )
        .expect("add");
        df.add_column(
            "score".to_string(),
            Series::new(vec![1.5f64, f64::NAN, 3.5], Some("score".to_string())).expect("series"),
        )
        .expect("add");
        df
    }

    #[test]
    fn test_filter_unknown_operator_errors_instead_of_keeping_nothing() {
        let df = make_mixed_df();
        let mut opts = HashMap::new();
        opts.insert("column".to_string(), "id".to_string());
        opts.insert("operator".to_string(), "typo_op".to_string());
        opts.insert("value".to_string(), "1".to_string());

        let result = FilterTransformPlugin::new().transform(df, &opts);
        assert!(
            result.is_err(),
            "an unknown operator must error, not silently return zero rows"
        );
    }

    #[test]
    fn test_filter_contains_on_numeric_looking_value_still_uses_string_match() {
        let mut df = DataFrame::new();
        df.add_column(
            "phone".to_string(),
            Series::new(
                vec!["555-0100".to_string(), "555-0250".to_string()],
                Some("phone".to_string()),
            )
            .expect("series"),
        )
        .expect("add");

        let mut opts = HashMap::new();
        opts.insert("column".to_string(), "phone".to_string());
        opts.insert("operator".to_string(), "contains".to_string());
        // "0" parses as a number, but `contains` must still do a substring
        // match rather than silently taking the (wrong) numeric branch.
        opts.insert("value".to_string(), "0".to_string());

        let result = FilterTransformPlugin::new()
            .transform(df, &opts)
            .expect("filter");
        assert_eq!(result.row_count(), 2, "both phone numbers contain '0'");
    }

    #[test]
    fn test_normalize_preserves_untouched_column_types() {
        let df = make_mixed_df();
        let mut opts = HashMap::new();
        opts.insert("columns".to_string(), "score".to_string());

        let result = NormalizePlugin::new()
            .transform(df, &opts)
            .expect("normalize");

        // "id" and "active" were never targeted -- they must keep their
        // real types, not become String columns.
        assert!(result.get_column::<i64>("id").is_ok());
        assert!(result.get_column::<bool>("active").is_ok());
    }

    #[test]
    fn test_fill_na_preserves_types_and_fills_real_nan() {
        let df = make_mixed_df();
        let mut opts = HashMap::new();
        opts.insert("value".to_string(), "0".to_string());
        // No "columns" option: every column is a fill target.

        let result = FillNaPlugin::new().transform(df, &opts).expect("fill_na");

        // "id" (i64) and "active" (bool) have no NA to fill -- they must
        // stay their real types, not become String columns.
        let ids = result.get_column::<i64>("id").expect("still i64");
        assert_eq!(ids.values(), &[1, 2, 3]);
        let active = result.get_column::<bool>("active").expect("still bool");
        assert_eq!(active.values(), &[true, false, true]);

        // "score" (f64) had a real NaN -- it must be filled in place and
        // remain f64, not be stringified.
        let scores = result.get_column::<f64>("score").expect("still f64");
        assert_eq!(scores.values(), &[1.5, 0.0, 3.5]);
    }

    #[test]
    fn test_filter_zero_matches_preserves_concrete_column_types() {
        // A filter matching zero rows must go through `build_empty_like`,
        // which previously collapsed every column to `f64`/`String`
        // regardless of its real type (the exact defect `copy_column_typed`
        // exists to avoid on the non-empty path). "No rows matched" must
        // stay distinguishable from "the schema changed".
        let df = make_mixed_df();
        let mut opts = HashMap::new();
        opts.insert("column".to_string(), "id".to_string());
        opts.insert("operator".to_string(), "eq".to_string());
        opts.insert("value".to_string(), "999".to_string()); // matches nothing

        let result = FilterTransformPlugin::new()
            .transform(df, &opts)
            .expect("filter");
        assert_eq!(result.row_count(), 0);

        let ids = result.get_column::<i64>("id").expect("still i64, not f64");
        assert_eq!(ids.values(), &[] as &[i64]);
        let active = result
            .get_column::<bool>("active")
            .expect("still bool, not String");
        assert_eq!(active.values(), &[] as &[bool]);
        let scores = result.get_column::<f64>("score").expect("still f64");
        assert_eq!(scores.values(), &[] as &[f64]);
    }
}
