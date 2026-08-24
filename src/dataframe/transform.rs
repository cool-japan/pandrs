use std::collections::{HashMap, HashSet};

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::na::NA;
use crate::series::base::Series;

/// DataFrame shape transformation options - melt operation
#[derive(Debug, Clone)]
pub struct MeltOptions {
    /// Names of columns to keep fixed (identifier columns)
    pub id_vars: Option<Vec<String>>,
    /// Names of columns to unpivot (value columns)
    pub value_vars: Option<Vec<String>>,
    /// Name of the column for variable names
    pub var_name: Option<String>,
    /// Name of the column for values
    pub value_name: Option<String>,
}

impl Default for MeltOptions {
    fn default() -> Self {
        Self {
            id_vars: None,
            value_vars: None,
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
        }
    }
}

/// DataFrame shape transformation options - stack operation
#[derive(Debug, Clone)]
pub struct StackOptions {
    /// List of columns to stack
    pub columns: Option<Vec<String>>,
    /// Name of the column for variable names after stacking
    pub var_name: Option<String>,
    /// Name of the column for values after stacking
    pub value_name: Option<String>,
    /// Whether to drop NaN values
    pub dropna: bool,
}

impl Default for StackOptions {
    fn default() -> Self {
        Self {
            columns: None,
            var_name: Some("variable".to_string()),
            value_name: Some("value".to_string()),
            dropna: false,
        }
    }
}

/// DataFrame shape transformation options - unstack operation
#[derive(Debug, Clone)]
pub struct UnstackOptions {
    /// Column containing variable names to unstack
    pub var_column: String,
    /// Column containing values to unstack
    pub value_column: String,
    /// Columns to use as index (can be multiple)
    pub index_columns: Option<Vec<String>>,
    /// Value to fill NA values
    pub fill_value: Option<NA<String>>,
}

/// Shape transformation functionality for DataFrames
pub trait TransformExt {
    /// Transform DataFrame to long format (wide to long)
    fn melt(&self, options: &MeltOptions) -> Result<Self>
    where
        Self: Sized;

    /// Stack DataFrame (columns to rows)
    fn stack(&self, options: &StackOptions) -> Result<Self>
    where
        Self: Sized;

    /// Unstack DataFrame (rows to columns)
    fn unstack(&self, options: &UnstackOptions) -> Result<Self>
    where
        Self: Sized;

    /// Aggregate values based on conditions (combination of pivot and filtering)
    fn conditional_aggregate<F, G>(
        &self,
        group_by: &str,
        agg_column: &str,
        filter_fn: F,
        agg_fn: G,
    ) -> Result<Self>
    where
        Self: Sized,
        F: Fn(&HashMap<String, String>) -> bool,
        G: Fn(&[String]) -> String;

    /// Concatenate multiple DataFrames along rows
    fn concat(dfs: &[&Self], ignore_index: bool) -> Result<Self>
    where
        Self: Sized;
}

/// Implementation of TransformExt for DataFrame
impl TransformExt for DataFrame {
    /// Unpivot `self` from wide to long format.
    ///
    /// `id_vars` columns are repeated once per melted value column (in
    /// `value_vars` order); the produced `var_name` column names which
    /// original column each row came from and `value_name` holds that
    /// column's value, rendered as text so columns of different original
    /// types can share one output column without corrupting either (mirrors
    /// pandas' object-dtype upcast when melting heterogeneous columns).
    fn melt(&self, options: &MeltOptions) -> Result<Self> {
        let id_vars: Vec<String> = match &options.id_vars {
            Some(vars) => {
                for v in vars {
                    if !self.contains_column(v) {
                        return Err(Error::ColumnNotFound(v.clone()));
                    }
                }
                vars.clone()
            }
            None => Vec::new(),
        };
        let id_set: HashSet<&str> = id_vars.iter().map(|s| s.as_str()).collect();

        let value_vars: Vec<String> = match &options.value_vars {
            Some(vars) => {
                for v in vars {
                    if !self.contains_column(v) {
                        return Err(Error::ColumnNotFound(v.clone()));
                    }
                }
                vars.clone()
            }
            None => self
                .column_names()
                .iter()
                .filter(|c| !id_set.contains(c.as_str()))
                .cloned()
                .collect(),
        };

        let var_name = options
            .var_name
            .clone()
            .unwrap_or_else(|| "variable".to_string());
        let value_name = options
            .value_name
            .clone()
            .unwrap_or_else(|| "value".to_string());

        let n_value_cols = value_vars.len();
        let total_rows = self.row_count() * n_value_cols;

        let mut result = DataFrame::new();

        for id_var in &id_vars {
            append_repeated_column(self, id_var, n_value_cols, &mut result)?;
        }

        let mut var_values: Vec<String> = Vec::with_capacity(total_rows);
        let mut val_values: Vec<String> = Vec::with_capacity(total_rows);
        for col in &value_vars {
            let text = column_as_strings(self, col)?;
            for v in &text {
                var_values.push(col.clone());
                val_values.push(v.clone());
            }
        }

        result.add_column(var_name.clone(), Series::new(var_values, Some(var_name))?)?;
        result.add_column(
            value_name.clone(),
            Series::new(val_values, Some(value_name))?,
        )?;

        Ok(result)
    }

    /// Stack `self`'s columns into rows: `columns` (default: every column)
    /// are visited in row-major order, each producing one output row of
    /// `(id, variable, value)` where `id` is that row's real index label
    /// (positional if `self` has no explicit index) and `value` is the
    /// column's value at that row, rendered as text. When `dropna` is set,
    /// a cell is skipped only when its *source* column is one of this
    /// crate's genuine numeric scalar types and the value is NaN -- a
    /// column with no numeric-NA representation (e.g. `String`) never has
    /// anything to drop.
    fn stack(&self, options: &StackOptions) -> Result<Self> {
        let cols_to_stack: Vec<String> = match &options.columns {
            Some(cols) => {
                for c in cols {
                    if !self.contains_column(c) {
                        return Err(Error::ColumnNotFound(c.clone()));
                    }
                }
                cols.clone()
            }
            None => self.column_names().to_vec(),
        };

        let var_name = options
            .var_name
            .clone()
            .unwrap_or_else(|| "variable".to_string());
        let value_name = options
            .value_name
            .clone()
            .unwrap_or_else(|| "value".to_string());

        let n_rows = self.row_count();
        let labels = row_labels(self);

        let mut col_text: Vec<Vec<String>> = Vec::with_capacity(cols_to_stack.len());
        let mut col_numeric: Vec<Option<Vec<f64>>> = Vec::with_capacity(cols_to_stack.len());
        for c in &cols_to_stack {
            col_text.push(column_as_strings(self, c)?);
            col_numeric.push(if is_numeric_typed(self, c) {
                self.get_column_numeric_values(c).ok()
            } else {
                None
            });
        }

        let mut id_values: Vec<String> = Vec::new();
        let mut var_values: Vec<String> = Vec::new();
        let mut val_values: Vec<String> = Vec::new();

        for row in 0..n_rows {
            for (col_idx, col_name) in cols_to_stack.iter().enumerate() {
                let is_na = col_numeric[col_idx]
                    .as_ref()
                    .map(|values| values[row].is_nan())
                    .unwrap_or(false);
                if options.dropna && is_na {
                    continue;
                }
                id_values.push(labels[row].clone());
                var_values.push(col_name.clone());
                val_values.push(col_text[col_idx][row].clone());
            }
        }

        let mut result = DataFrame::new();
        result.add_column(
            "id".to_string(),
            Series::new(id_values, Some("id".to_string()))?,
        )?;
        result.add_column(var_name.clone(), Series::new(var_values, Some(var_name))?)?;
        result.add_column(
            value_name.clone(),
            Series::new(val_values, Some(value_name))?,
        )?;
        Ok(result)
    }

    /// Pivot `self` from long to wide format: each unique value of
    /// `var_column` becomes a new column, populated from `value_column`,
    /// with rows keyed by `index_columns` (default: every other column).
    /// Both the emitted row keys and the new column names are sorted for a
    /// deterministic result. Because a pivot can invent index/variable
    /// combinations absent from the input, every produced value column is
    /// `Series<NA<String>>`: a present cell is `NA::Value(text)`, and a
    /// genuinely missing one is `options.fill_value.clone()` if given, else
    /// `NA::NA` -- never a fabricated empty string.
    fn unstack(&self, options: &UnstackOptions) -> Result<Self> {
        if !self.contains_column(&options.var_column) {
            return Err(Error::ColumnNotFound(options.var_column.clone()));
        }
        if !self.contains_column(&options.value_column) {
            return Err(Error::ColumnNotFound(options.value_column.clone()));
        }

        let index_columns: Vec<String> = match &options.index_columns {
            Some(cols) => {
                for c in cols {
                    if !self.contains_column(c) {
                        return Err(Error::ColumnNotFound(c.clone()));
                    }
                }
                cols.clone()
            }
            None => self
                .column_names()
                .iter()
                .filter(|c| c.as_str() != options.var_column && c.as_str() != options.value_column)
                .cloned()
                .collect(),
        };

        let n_rows = self.row_count();
        let var_values = column_as_strings(self, &options.var_column)?;
        let value_values = column_as_strings(self, &options.value_column)?;

        let mut index_key_columns: Vec<Vec<String>> = Vec::with_capacity(index_columns.len());
        for c in &index_columns {
            index_key_columns.push(column_as_strings(self, c)?);
        }
        let row_key = |row: usize| -> Vec<String> {
            index_key_columns
                .iter()
                .map(|col| col[row].clone())
                .collect()
        };

        let mut unique_keys: Vec<Vec<String>> = (0..n_rows).map(row_key).collect();
        unique_keys.sort();
        unique_keys.dedup();

        let mut unique_vars: Vec<String> = var_values.clone();
        unique_vars.sort();
        unique_vars.dedup();

        let mut cell_map: HashMap<(Vec<String>, String), String> = HashMap::with_capacity(n_rows);
        for row in 0..n_rows {
            cell_map.insert(
                (row_key(row), var_values[row].clone()),
                value_values[row].clone(),
            );
        }

        let mut result = DataFrame::new();
        for (level, name) in index_columns.iter().enumerate() {
            let values: Vec<String> = unique_keys.iter().map(|k| k[level].clone()).collect();
            result.add_column(name.clone(), Series::new(values, Some(name.clone()))?)?;
        }

        for var in &unique_vars {
            let values: Vec<NA<String>> = unique_keys
                .iter()
                .map(|key| match cell_map.get(&(key.clone(), var.clone())) {
                    Some(v) => NA::Value(v.clone()),
                    None => options.fill_value.clone().unwrap_or(NA::NA),
                })
                .collect();
            result.add_column(var.clone(), Series::new(values, Some(var.clone()))?)?;
        }

        Ok(result)
    }

    /// Filter rows to those where `filter_fn` returns true (given every
    /// column's value at that row, rendered as text -- not just
    /// `group_by`/`agg_column`, since a filter legitimately inspects other
    /// columns), then group the surviving rows by `group_by` (first-seen
    /// key order, not hash order) and reduce each group's `agg_column`
    /// values with `agg_fn`.
    fn conditional_aggregate<F, G>(
        &self,
        group_by: &str,
        agg_column: &str,
        filter_fn: F,
        agg_fn: G,
    ) -> Result<Self>
    where
        F: Fn(&HashMap<String, String>) -> bool,
        G: Fn(&[String]) -> String,
    {
        if !self.contains_column(group_by) {
            return Err(Error::ColumnNotFound(group_by.to_string()));
        }
        if !self.contains_column(agg_column) {
            return Err(Error::ColumnNotFound(agg_column.to_string()));
        }

        let n_rows = self.row_count();
        let column_names = self.column_names().to_vec();
        let mut column_text: Vec<(String, Vec<String>)> = Vec::with_capacity(column_names.len());
        for name in &column_names {
            column_text.push((name.clone(), column_as_strings(self, name)?));
        }

        let mut group_order: Vec<String> = Vec::new();
        let mut group_positions: HashMap<String, usize> = HashMap::new();
        let mut group_values: Vec<Vec<String>> = Vec::new();

        for row in 0..n_rows {
            let mut row_map: HashMap<String, String> = HashMap::with_capacity(column_text.len());
            for (name, values) in &column_text {
                row_map.insert(name.clone(), values[row].clone());
            }
            if !filter_fn(&row_map) {
                continue;
            }
            let key = row_map.get(group_by).cloned().unwrap_or_default();
            let value = row_map.get(agg_column).cloned().unwrap_or_default();
            let idx = *group_positions.entry(key.clone()).or_insert_with(|| {
                group_order.push(key.clone());
                group_values.push(Vec::new());
                group_order.len() - 1
            });
            group_values[idx].push(value);
        }

        let mut cat_values: Vec<String> = Vec::with_capacity(group_order.len());
        let mut agg_values: Vec<String> = Vec::with_capacity(group_order.len());
        for (idx, key) in group_order.iter().enumerate() {
            cat_values.push(key.clone());
            agg_values.push(agg_fn(&group_values[idx]));
        }

        let mut result = DataFrame::new();
        result.add_column(
            group_by.to_string(),
            Series::new(cat_values, Some(group_by.to_string()))?,
        )?;
        let agg_col_name = format!("{}_agg", agg_column);
        result.add_column(
            agg_col_name.clone(),
            Series::new(agg_values, Some(agg_col_name))?,
        )?;
        Ok(result)
    }

    /// Concatenate `dfs` along rows. Columns are aligned by name (the union
    /// across all frames, first-seen order); a column present in every
    /// frame keeps its exact type when they all agree (falling back to a
    /// shared text rendering when they don't), and a column missing from
    /// some frame(s) is NaN-widened to `f64` when it is numeric in the
    /// frame(s) that do have it, or otherwise rejected with
    /// `Error::NotImplemented` naming the column -- mirroring
    /// `DataFrame::concat_rows`'s handling of the same situation, so the two
    /// concatenation entry points in this crate agree. When `ignore_index`
    /// is false the original row labels are concatenated and used as the
    /// result's index; if that would contain duplicate labels (this crate's
    /// `Index` requires uniqueness), an error explains the conflict rather
    /// than silently ignoring the request.
    fn concat(dfs: &[&Self], ignore_index: bool) -> Result<Self> {
        if dfs.is_empty() {
            return Ok(DataFrame::new());
        }

        let mut union_names: Vec<String> = Vec::new();
        for df in dfs {
            for name in df.column_names() {
                if !union_names.contains(name) {
                    union_names.push(name.clone());
                }
            }
        }

        let mut result = DataFrame::new();
        for name in &union_names {
            append_concatenated_column(dfs, name, &mut result)?;
        }

        // Only attempt to carry the index forward when at least one input
        // frame actually has one: `row_labels` synthesizes positional
        // "0", "1", ... for a frame with none, and treating those as real
        // labels to preserve would reject ordinary concatenation (two
        // index-less frames both starting at "0") over a collision this
        // function invented, not one the caller's data has.
        if !ignore_index && dfs.iter().any(|df| has_explicit_index(df)) {
            let mut labels: Vec<String> = Vec::new();
            for df in dfs {
                labels.extend(row_labels(df));
            }
            let idx = crate::index::Index::new(labels).map_err(|e| {
                Error::InvalidValue(format!(
                    "concat: cannot preserve the original index with ignore_index=false \
                     because the combined row labels are not unique ({e}); pass \
                     ignore_index=true to reset the index instead"
                ))
            })?;
            result.set_index(idx)?;
        }

        Ok(result)
    }
}

/// Render a DataFrame column as owned strings, tolerating a `&'static
/// str`-backed column (e.g. built from `Series::new(vec!["a", "b"], ..)`, as
/// this crate's own examples and tests commonly do) that
/// [`DataFrame::get_column_string_values`] does not special-case.
fn column_as_strings(df: &DataFrame, col_name: &str) -> Result<Vec<String>> {
    if let Ok(series) = df.get_column::<&'static str>(col_name) {
        return Ok(series.values().iter().map(|s| s.to_string()).collect());
    }
    df.get_column_string_values(col_name)
}

/// Row labels for `df`: its real index when one has been set and its length
/// actually matches the row count, else a positional `"0", "1", ...`
/// sequence. (An unset index reports as an *empty* [`crate::index::Index`],
/// not a positional range, so that case is detected rather than trusted.)
fn row_labels(df: &DataFrame) -> Vec<String> {
    let n = df.row_count();
    match df.get_index().string_values() {
        Some(labels) if labels.len() == n => labels,
        _ => (0..n).map(|i| i.to_string()).collect(),
    }
}

/// True when `df` actually carries an explicit index (set via `set_index`,
/// `set_multi_index`, `with_index`, or `with_multi_index`), as opposed to
/// the implicit default: `get_index()` reports an *empty* index rather than
/// a positional range for a `DataFrame` that never had one set, so this
/// checks the real signal (`row_labels` returning something of the right
/// length from `get_index()` itself, not its own positional fallback).
fn has_explicit_index(df: &DataFrame) -> bool {
    df.row_count() > 0
        && df
            .get_index()
            .string_values()
            .is_some_and(|labels| labels.len() == df.row_count())
}

/// True only when `name` is stored as one of the concrete numeric scalar
/// types (`f64`/`i64`/`i32`/`f32`) that this module widens to `f64` with a
/// NaN fill when concatenating a column absent from some input frames.
/// Deliberately narrower than `DataFrame::get_column_numeric_values` (which
/// also *parses* `String` columns): treating a parseable string column as
/// numeric here would silently retype it (e.g. turning a zero-padded code
/// column into floats) instead of concatenating it as text.
fn is_numeric_typed(df: &DataFrame, name: &str) -> bool {
    df.get_column::<f64>(name).is_ok()
        || df.get_column::<i64>(name).is_ok()
        || df.get_column::<i32>(name).is_ok()
        || df.get_column::<f32>(name).is_ok()
}

/// Append `source`'s `col_name` column to `target`, repeated `repeat` times
/// back-to-back (`melt` tiles each id column once per value column this
/// way). Preserves the column's exact element type when it is one of this
/// crate's common scalar types; anything else (including a
/// `&'static str`-backed column) is bridged through [`column_as_strings`] so
/// the result is always a concrete, readable `Series<String>` rather than
/// being silently dropped.
fn append_repeated_column(
    source: &DataFrame,
    col_name: &str,
    repeat: usize,
    target: &mut DataFrame,
) -> Result<()> {
    macro_rules! try_repeat {
        ($ty:ty) => {
            if let Ok(series) = source.get_column::<$ty>(col_name) {
                let mut values: Vec<$ty> = Vec::with_capacity(series.len() * repeat);
                for _ in 0..repeat {
                    values.extend(series.values().iter().cloned());
                }
                target.add_column(
                    col_name.to_string(),
                    Series::new(values, Some(col_name.to_string()))?,
                )?;
                return Ok(());
            }
        };
    }
    try_repeat!(String);
    try_repeat!(i64);
    try_repeat!(f64);
    try_repeat!(i32);
    try_repeat!(f32);
    try_repeat!(bool);
    try_repeat!(chrono::NaiveDate);
    try_repeat!(chrono::NaiveDateTime);

    let text = column_as_strings(source, col_name)?;
    let mut values: Vec<String> = Vec::with_capacity(text.len() * repeat);
    for _ in 0..repeat {
        values.extend(text.iter().cloned());
    }
    target.add_column(
        col_name.to_string(),
        Series::new(values, Some(col_name.to_string()))?,
    )?;
    Ok(())
}

/// Build the concatenated `name` column across `dfs` into `result`; see the
/// [`TransformExt::concat`] doc comment for the type-handling rules this
/// implements.
fn append_concatenated_column(
    dfs: &[&DataFrame],
    name: &str,
    result: &mut DataFrame,
) -> Result<()> {
    let all_have = dfs.iter().all(|df| df.contains_column(name));

    if all_have {
        macro_rules! try_uniform {
            ($ty:ty) => {
                if dfs.iter().all(|df| df.get_column::<$ty>(name).is_ok()) {
                    let mut values: Vec<$ty> = Vec::new();
                    for df in dfs {
                        if let Ok(series) = df.get_column::<$ty>(name) {
                            values.extend(series.values().iter().cloned());
                        }
                    }
                    result.add_column(
                        name.to_string(),
                        Series::new(values, Some(name.to_string()))?,
                    )?;
                    return Ok(());
                }
            };
        }
        try_uniform!(String);
        try_uniform!(i64);
        try_uniform!(f64);
        try_uniform!(i32);
        try_uniform!(f32);
        try_uniform!(bool);
        try_uniform!(chrono::NaiveDate);
        try_uniform!(chrono::NaiveDateTime);

        // The frames disagree on this column's concrete type (or share one
        // this function doesn't special-case, such as `&'static str`): fall
        // back to a shared text rendering rather than corrupting or
        // dropping either side.
        let mut values: Vec<String> = Vec::new();
        for df in dfs {
            values.extend(column_as_strings(df, name)?);
        }
        result.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
        return Ok(());
    }

    // Missing from at least one frame: only a genuinely numeric column (in
    // every frame that does have it) has a safe NA representation
    // (`f64::NAN`) to fill the gap with.
    let numeric_uniform = dfs
        .iter()
        .filter(|df| df.contains_column(name))
        .all(|df| is_numeric_typed(df, name));

    if numeric_uniform {
        let mut values: Vec<f64> = Vec::new();
        for df in dfs {
            if df.contains_column(name) {
                values.extend(df.get_column_numeric_values(name)?);
            } else {
                values.extend(std::iter::repeat(f64::NAN).take(df.row_count()));
            }
        }
        result.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string()))?,
        )?;
        return Ok(());
    }

    Err(Error::NotImplemented(format!(
        "concat: column '{}' is missing from at least one input DataFrame and is not a \
         numeric type in the frame(s) that do have it, so the rows contributed by the \
         DataFrame(s) lacking it cannot be NA-filled for it; provide the column in every \
         DataFrame being concatenated",
        name
    )))
}

/// Helper function to clean DataBox values into plain strings
#[allow(dead_code)] // reserved for future use: cleans DataBox-wrapped values for full transform impls
fn clean_databox_value(value: &str) -> String {
    let trimmed = value
        .trim_start_matches("DataBox(\"")
        .trim_end_matches("\")");
    let value_str = if trimmed.starts_with("DataBox(") {
        trimmed.trim_start_matches("DataBox(").trim_end_matches(")")
    } else {
        trimmed
    };
    value_str.trim_matches('"').to_string()
}

/// Re-export transformation options for backward compatibility
#[deprecated(since = "0.1.0", note = "Use crate::dataframe::transform::MeltOptions")]
pub use crate::dataframe::transform::MeltOptions as LegacyMeltOptions;

#[deprecated(
    since = "0.1.0",
    note = "Use crate::dataframe::transform::StackOptions"
)]
pub use crate::dataframe::transform::StackOptions as LegacyStackOptions;

#[deprecated(
    since = "0.1.0",
    note = "Use crate::dataframe::transform::UnstackOptions"
)]
pub use crate::dataframe::transform::UnstackOptions as LegacyUnstackOptions;
