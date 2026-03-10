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

/// Build a list of row indices that satisfy the filter condition
fn build_filter_mask(
    df: &DataFrame,
    column: &str,
    operator: &str,
    value: &str,
) -> Result<Vec<usize>> {
    let row_count = df.row_count();
    let mut indices = Vec::new();

    // Try numeric comparison first
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
                    _ => false,
                };
                if keep {
                    indices.push(i);
                }
            }
            return Ok(indices);
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
            other => {
                return Err(Error::InvalidInput(format!(
                    "filter: unknown operator '{}'",
                    other
                )))
            }
        };
        if keep {
            indices.push(i);
        }
    }

    let _ = row_count; // suppress warning
    Ok(indices)
}

/// Build an empty DataFrame with the same column structure
fn build_empty_like(df: &DataFrame) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in df.column_names() {
        // Try numeric first, fall back to string
        if df.get_column_numeric_values(&col_name).is_ok() {
            let series: Series<f64> = Series::new(vec![], Some(col_name.clone()))?;
            result.add_column(col_name, series)?;
        } else {
            let series: Series<String> = Series::new(vec![], Some(col_name.clone()))?;
            result.add_column(col_name, series)?;
        }
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
                .into_iter()
                .filter(|c| df.get_column_numeric_values(c).is_ok())
                .collect()
        };

        // Build the new DataFrame, normalizing target columns
        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            if target_cols.contains(&col_name) {
                let values = df.get_column_numeric_values(&col_name)?;
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                let normalized: Vec<f64> = if (max - min).abs() < f64::EPSILON {
                    // All values are the same; map to 0.0
                    vec![0.0; values.len()]
                } else {
                    values.iter().map(|v| (v - min) / (max - min)).collect()
                };

                let series = Series::new(normalized, Some(col_name.clone()))?;
                result.add_column(col_name, series)?;
            } else {
                // Preserve non-target columns as strings
                let values = df.get_column_string_values(&col_name)?;
                let series = Series::new(values, Some(col_name.clone()))?;
                result.add_column(col_name, series)?;
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
                .map(|cols| cols.contains(&col_name))
                .unwrap_or(true);

            if should_fill {
                let values: Vec<String> = df
                    .get_column_string_values(&col_name)?
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
                result.add_column(col_name, series)?;
            } else {
                let values = df.get_column_string_values(&col_name)?;
                let series = Series::new(values, Some(col_name.clone()))?;
                result.add_column(col_name, series)?;
            }
        }

        Ok(result)
    }
}
