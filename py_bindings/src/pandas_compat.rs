//! # Pandas Compatibility Layer
//!
//! Pandas-shaped wrappers (`PandasDataFrame`, `PandasSeries`, `GroupBy`,
//! `read_csv`, `concat`, ...) that drive the **real** pandrs `DataFrame`
//! engine. Every method here either wires straight through to a genuine
//! pandrs implementation or raises a Python exception when a requested
//! behaviour cannot be honoured -- none of them fabricate data.
//!
//! ## Data model & known limitations
//!
//! `PandasDataFrame` wraps the string-typed [`crate::PyDataFrame`], so every
//! column is stored as text and a missing value is the marker `"NA"`. Numeric
//! operations (`describe`, `groupby().mean()`, ...) parse those strings on the
//! fly (NA/empty cells become `NaN`); a column is treated as numeric only when
//! every non-missing cell parses as a float. Because values are text, a genuine
//! string equal to `"NA"` is indistinguishable from a missing value. For fully
//! typed, high-performance columns use the `OptimizedDataFrame` class instead.

use crate::PyDataFrame;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PyString, PyStringMethods, PyType};
use std::collections::HashMap;

use ::pandrs::dataframe::pandas_compat::concat::{concat as pandrs_concat, ConcatAxis};
use ::pandrs::dataframe::pandas_compat::groupby::DataFrameGroupBy as PandrsGroupBy;
use ::pandrs::dataframe::pandas_compat::helpers::aggregations::describe_column;
use ::pandrs::dataframe::pandas_compat::merge::{merge as pandrs_merge, JoinType};
use ::pandrs::dataframe::QueryExt;
use ::pandrs::{DataFrame, Series};

/// The stored string form of a missing value in the string-typed compat frame.
const NA_MARKER: &str = "NA";

// ---------------------------------------------------------------------------
// Pure-Rust helpers (unit-tested below; no Python interpreter involved)
// ---------------------------------------------------------------------------

/// Map any displayable error into a Python `ValueError`.
fn to_py_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Format a statistic for display, rendering `NaN` explicitly.
fn fmt_stat(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else {
        format!("{}", v)
    }
}

/// Rebuild `df`, converting each fully numeric-parseable column into a real
/// `Series<f64>` so the genuine numeric aggregators (which dispatch on the
/// concrete column dtype) can see it.
///
/// * Columns named in `keep_as_string` are always left as text -- used to keep
///   group-key columns verbatim (e.g. zero-padded codes that must not become
///   floats).
/// * A missing (`"NA"`) or empty cell becomes `f64::NAN` rather than dropping
///   the whole column, so NA semantics survive.
/// * A column is converted only when **every** non-missing cell parses; a
///   single un-parseable value keeps the entire column as text (so a
///   categorical column is never silently coerced).
fn numeric_typed(df: &DataFrame, keep_as_string: &[String]) -> PyResult<DataFrame> {
    let mut out = DataFrame::new();
    for col in df.column_names() {
        let strings = df.get_column_string_values(col).map_err(to_py_value_err)?;

        if keep_as_string.iter().any(|k| k == col) {
            let series = Series::new(strings, Some(col.clone())).map_err(to_py_value_err)?;
            out.add_column(col.clone(), series)
                .map_err(to_py_value_err)?;
            continue;
        }

        let mut numeric: Vec<f64> = Vec::with_capacity(strings.len());
        let mut is_numeric = true;
        for cell in &strings {
            let trimmed = cell.trim();
            if trimmed.is_empty() || trimmed == NA_MARKER {
                numeric.push(f64::NAN);
                continue;
            }
            match trimmed.parse::<f64>() {
                Ok(v) => numeric.push(v),
                Err(_) => {
                    is_numeric = false;
                    break;
                }
            }
        }

        // The two branches build different concrete Series types
        // (`Series<f64>` vs `Series<String>`), so they add their columns
        // separately rather than through a shared binding.
        if is_numeric {
            let series = Series::new(numeric, Some(col.clone())).map_err(to_py_value_err)?;
            out.add_column(col.clone(), series)
                .map_err(to_py_value_err)?;
        } else {
            let series = Series::new(strings, Some(col.clone())).map_err(to_py_value_err)?;
            out.add_column(col.clone(), series)
                .map_err(to_py_value_err)?;
        }
    }
    Ok(out)
}

/// Serialise `df` to CSV text, honouring the pandas-style `sep`, `na_rep`,
/// `index` and `header` options. Fields containing the separator, a quote, or a
/// newline are quoted with doubled inner quotes (RFC-4180). The stored `"NA"`
/// marker is emitted as `na_rep`.
fn build_csv(
    df: &DataFrame,
    sep: &str,
    na_rep: &str,
    index: bool,
    header: bool,
) -> PyResult<String> {
    let columns = df.column_names().to_vec();
    let mut string_columns: Vec<Vec<String>> = Vec::with_capacity(columns.len());
    for col in &columns {
        string_columns.push(df.get_column_string_values(col).map_err(to_py_value_err)?);
    }
    let row_count = df.row_count();

    let quote = |field: &str| -> String {
        if field.contains(sep)
            || field.contains('"')
            || field.contains('\n')
            || field.contains('\r')
        {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.to_string()
        }
    };

    let mut out = String::new();
    if header {
        let mut fields: Vec<String> = Vec::with_capacity(columns.len() + 1);
        if index {
            fields.push(String::new());
        }
        for col in &columns {
            fields.push(quote(col));
        }
        out.push_str(&fields.join(sep));
        out.push('\n');
    }
    for r in 0..row_count {
        let mut fields: Vec<String> = Vec::with_capacity(columns.len() + 1);
        if index {
            fields.push(r.to_string());
        }
        for column in &string_columns {
            let cell = column.get(r).map(|s| s.as_str()).unwrap_or("");
            let rendered = if cell == NA_MARKER { na_rep } else { cell };
            fields.push(quote(rendered));
        }
        out.push_str(&fields.join(sep));
        out.push('\n');
    }
    Ok(out)
}

/// Whether `df`'s row index is the default positional `0..n` index (so that
/// label-based and position-based selection coincide).
fn index_is_positional(df: &DataFrame) -> bool {
    let idx = df.get_index();
    if idx.is_multi() {
        return false;
    }
    match idx.string_values() {
        Some(values) => {
            values.is_empty() || values.iter().enumerate().all(|(i, v)| v == &i.to_string())
        }
        None => true,
    }
}

/// Build a new DataFrame from the rows of `df` at the given positions, in the
/// requested order. Positions must already be validated `< row_count`. The
/// result carries a fresh default positional index (`0..len`).
fn select_rows_by_position(df: &DataFrame, positions: &[usize]) -> PyResult<DataFrame> {
    let mut data: HashMap<String, Vec<String>> = HashMap::new();
    for col in df.column_names() {
        let values = df.get_column_string_values(col).map_err(to_py_value_err)?;
        let selected: Vec<String> = positions.iter().map(|&p| values[p].clone()).collect();
        data.insert(col.clone(), selected);
    }
    DataFrame::from_map(data, None).map_err(to_py_value_err)
}

/// Resolve a single (possibly negative) position against a row count.
fn resolve_one(p: isize, n: usize) -> PyResult<usize> {
    let idx = if p < 0 { p + n as isize } else { p };
    if idx < 0 || idx as usize >= n {
        return Err(PyIndexError::new_err(format!(
            "position {} out of range for {} rows",
            p, n
        )));
    }
    Ok(idx as usize)
}

/// Resolve an iloc/loc key (int, list of ints, or slice) to concrete row
/// positions.
fn resolve_positions(py: Python<'_>, key: &Py<PyAny>, n: usize) -> PyResult<Vec<usize>> {
    let bound = key.bind(py);

    if let Ok(slice) = bound.cast::<PySlice>() {
        let indices = slice.indices(n as isize)?;
        let (start, stop, step) = (indices.start, indices.stop, indices.step);
        let mut out = Vec::new();
        let mut i = start;
        if step > 0 {
            while i < stop {
                out.push(i as usize);
                i += step;
            }
        } else if step < 0 {
            while i > stop {
                out.push(i as usize);
                i += step;
            }
        }
        return Ok(out);
    }

    if let Ok(list) = bound.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(resolve_one(item.extract::<isize>()?, n)?);
        }
        return Ok(out);
    }

    if let Ok(p) = bound.extract::<isize>() {
        return Ok(vec![resolve_one(p, n)?]);
    }

    Err(PyIndexError::new_err(
        "indexer key must be an int, a list of ints, or a slice",
    ))
}

/// Extract a column name or list of names from a Python object.
fn extract_string_or_list(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Vec<String>> {
    let bound = obj.bind(py);
    if let Ok(s) = bound.extract::<String>() {
        return Ok(vec![s]);
    }
    if let Ok(v) = bound.extract::<Vec<String>>() {
        return Ok(v);
    }
    Err(PyValueError::new_err(
        "expected a column name or a list of column names",
    ))
}

/// Extract an integer row label or list of integer labels from a Python object.
fn extract_usize_or_list(py: Python<'_>, obj: &Py<PyAny>) -> PyResult<Vec<usize>> {
    let bound = obj.bind(py);
    if let Ok(i) = bound.extract::<usize>() {
        return Ok(vec![i]);
    }
    if let Ok(v) = bound.extract::<Vec<usize>>() {
        return Ok(v);
    }
    Err(PyValueError::new_err(
        "expected an integer row label or a list of integers",
    ))
}

// ---------------------------------------------------------------------------
// PandasDataFrame
// ---------------------------------------------------------------------------

/// Pandas-compatible DataFrame view over the real pandrs engine.
///
/// A thin, pandas-shaped wrapper around the string-typed `DataFrame` class.
/// Every operation is executed by genuine pandrs code (`head`/`tail`,
/// `describe`, `query`, `drop`, `merge`, `groupby`, CSV I/O, ...); unsupported
/// options raise a Python exception rather than being silently ignored.
/// Columns are stored as strings -- for typed, high-performance columns use the
/// `OptimizedDataFrame` class.
// `FromPyObject` is never needed for this type (it's never used as a bare
// `#[pymethods]`/`#[pyfunction]` parameter type -- only via `Py<T>`/`PyRef<T>`
// or constructed directly in Rust), so opt out of the deprecated implicit
// by-value `FromPyObject` blanket impl for `Clone`-able `#[pyclass]` types.
#[pyclass(name = "PandasDataFrame", skip_from_py_object)]
#[derive(Clone)]
pub struct PandasCompatibleDataFrame {
    inner: PyDataFrame,
}

#[pymethods]
impl PandasCompatibleDataFrame {
    /// Create a DataFrame from a dictionary of column -> list of values.
    ///
    /// `index`, `columns` and `dtype` are rejected (rather than silently
    /// ignored): this string-typed frame always uses a default positional
    /// index and stores columns as text.
    #[new]
    #[pyo3(signature = (data=None, index=None, columns=None, dtype=None))]
    fn new(
        py: Python<'_>,
        data: Option<Py<PyAny>>,
        index: Option<Py<PyAny>>,
        columns: Option<Py<PyAny>>,
        dtype: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if index.is_some() {
            return Err(PyNotImplementedError::new_err(
                "PandasDataFrame(index=...) is not supported; this frame uses a default positional index",
            ));
        }
        if columns.is_some() {
            return Err(PyNotImplementedError::new_err(
                "PandasDataFrame(columns=...) is not supported; pass a dict whose keys are the column names, or use .select for a subset",
            ));
        }
        if dtype.is_some() {
            return Err(PyNotImplementedError::new_err(
                "PandasDataFrame(dtype=...) is not supported; columns are stored as strings (see OptimizedDataFrame for typed columns)",
            ));
        }
        let inner = PyDataFrame::new(py, data)?;
        Ok(Self { inner })
    }

    /// Return the first `n` rows (real slice of the underlying DataFrame).
    #[pyo3(signature = (n=5))]
    fn head(&self, n: Option<usize>) -> PyResult<Self> {
        let n = n.unwrap_or(5);
        let sub = self.inner.rs_df().head(n).map_err(to_py_value_err)?;
        Ok(Self {
            inner: PyDataFrame::from_df(sub),
        })
    }

    /// Return the last `n` rows (real slice of the underlying DataFrame).
    #[pyo3(signature = (n=5))]
    fn tail(&self, n: Option<usize>) -> PyResult<Self> {
        let n = n.unwrap_or(5);
        let sub = self.inner.rs_df().tail(n).map_err(to_py_value_err)?;
        Ok(Self {
            inner: PyDataFrame::from_df(sub),
        })
    }

    /// Print a pandas-style structural summary (real column count, per-column
    /// non-null counts, and `object` dtype -- every column is string-backed).
    fn info(&self, py: Python<'_>) -> PyResult<()> {
        let df = self.inner.rs_df();
        let columns = df.column_names().to_vec();
        let rows = df.row_count();

        let mut lines: Vec<String> = Vec::new();
        lines.push("<class 'pandrs.PandasDataFrame'>".to_string());
        lines.push(format!(
            "RangeIndex: {} entries, 0 to {}",
            rows,
            rows.saturating_sub(1)
        ));
        lines.push(format!("Data columns (total {} columns):", columns.len()));
        lines.push(" #   Column           Non-Null Count  Dtype".to_string());
        lines.push(" --- ------           --------------  -----".to_string());
        for (i, col) in columns.iter().enumerate() {
            let non_null = df
                .get_column_string_values(col)
                .map(|values| {
                    values
                        .iter()
                        .filter(|s| !s.is_empty() && s.as_str() != NA_MARKER)
                        .count()
                })
                .unwrap_or(0);
            lines.push(format!(
                " {:<3} {:<16} {:<15} object",
                i,
                col,
                format!("{} non-null", non_null)
            ));
        }

        let info_str = lines.join("\n");
        // Route through Python's print so redirects / Jupyter capture still work.
        py.import("builtins")?
            .getattr("print")?
            .call1((info_str,))?;
        Ok(())
    }

    /// Compute descriptive statistics for every numeric column.
    ///
    /// Real statistics (count/mean/std/min/25%/50%/75%/max) computed by pandrs.
    /// Only columns whose non-missing cells all parse as numbers are included
    /// (like pandas' default `describe()` over numeric columns). The result is
    /// a frame with a `statistic` column plus one column per numeric input
    /// column (a transpose of pandas' stats-as-index layout); columns are in
    /// sorted order. A frame with no numeric columns yields an empty frame.
    fn describe(&self) -> PyResult<Self> {
        let df = self.inner.rs_df();
        // NA-aware typing so a column with a missing value is not dropped and
        // "NA" cells do not make the whole column un-parseable.
        let typed = numeric_typed(df, &[])?;
        let numeric_cols: Vec<String> = typed
            .column_names()
            .iter()
            .filter(|c| typed.is_numeric_column(c))
            .cloned()
            .collect();

        if numeric_cols.is_empty() {
            return Ok(Self {
                inner: PyDataFrame::from_df(DataFrame::new()),
            });
        }

        let stat_order = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
        let mut data: HashMap<String, Vec<String>> = HashMap::new();
        data.insert(
            "statistic".to_string(),
            stat_order.iter().map(|s| s.to_string()).collect(),
        );

        for col in &numeric_cols {
            let stats = describe_column(&typed, col).map_err(to_py_value_err)?;
            let get = |k: &str| stats.get(k).copied().unwrap_or(f64::NAN);
            let row = vec![
                format!("{}", get("count") as i64),
                fmt_stat(get("mean")),
                fmt_stat(get("std")),
                fmt_stat(get("min")),
                fmt_stat(get("25%")),
                fmt_stat(get("50%")),
                fmt_stat(get("75%")),
                fmt_stat(get("max")),
            ];
            data.insert(col.clone(), row);
        }

        let inner = DataFrame::from_map(data, None).map_err(to_py_value_err)?;
        Ok(Self {
            inner: PyDataFrame::from_df(inner),
        })
    }

    /// Position-based indexer (`df.iloc()[key]`).
    fn iloc(&self) -> ILocIndexer {
        ILocIndexer {
            dataframe: self.clone(),
        }
    }

    /// Label-based indexer (`df.loc()[key]`). For the default positional index
    /// this coincides with `iloc`; an explicit non-positional index raises.
    fn loc(&self) -> LocIndexer {
        LocIndexer {
            dataframe: self.clone(),
        }
    }

    /// Column access: `df["col"]` -> Series, `df[["a", "b"]]` -> DataFrame.
    fn __getitem__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(column_name) = key.cast_bound::<PyString>(py) {
            let column_str = column_name.to_cow()?.into_owned();
            self.get_column(py, &column_str)
        } else if let Ok(column_list) = key.cast_bound::<PyList>(py) {
            let mut column_names: Vec<String> = Vec::with_capacity(column_list.len());
            for item in column_list.iter() {
                let py_str = item
                    .cast::<PyString>()
                    .map_err(|_| PyKeyError::new_err("Invalid column selection"))?;
                column_names.push(py_str.to_cow()?.into_owned());
            }
            self.select_columns(py, column_names)
        } else {
            Err(PyKeyError::new_err("Unsupported key type"))
        }
    }

    /// Filter rows with a pandas-style boolean expression, evaluated by the
    /// real pandrs query engine (e.g. `"age > 30 and city == 'NYC'"`).
    #[pyo3(signature = (expr, inplace=false))]
    fn query(&self, expr: &str, inplace: Option<bool>) -> PyResult<Self> {
        if inplace.unwrap_or(false) {
            return Err(PyNotImplementedError::new_err(
                "query(inplace=True) is not supported; use the returned DataFrame",
            ));
        }
        let result = self.inner.rs_df().query(expr).map_err(to_py_value_err)?;
        Ok(Self {
            inner: PyDataFrame::from_df(result),
        })
    }

    /// Group by one or more columns. Returns a `GroupBy` object whose
    /// aggregations are computed by real pandrs code.
    fn groupby(&self, py: Python<'_>, by: Py<PyAny>) -> PyResult<GroupBy> {
        let keys = extract_string_or_list(py, &by)?;
        if keys.is_empty() {
            return Err(PyValueError::new_err(
                "groupby requires at least one column",
            ));
        }
        let df = self.inner.rs_df();
        for key in &keys {
            if !df.column_names().iter().any(|c| c == key) {
                return Err(PyKeyError::new_err(format!(
                    "groupby column '{}' not found",
                    key
                )));
            }
        }
        Ok(GroupBy {
            dataframe: self.clone(),
            group_keys: keys,
        })
    }

    /// Drop columns and/or rows.
    ///
    /// * `columns=` (or `labels=` with `axis=1`) drops columns -- preserves the
    ///   remaining columns' types and the index.
    /// * `index=` (or `labels=` with `axis=0`) drops rows by integer position
    ///   (this frame's labels are positional); row-dropping reindexes the
    ///   result to `0..len`.
    #[pyo3(signature = (labels=None, axis=0, index=None, columns=None, inplace=false))]
    fn drop(
        &self,
        py: Python<'_>,
        labels: Option<Py<PyAny>>,
        axis: Option<i32>,
        index: Option<Py<PyAny>>,
        columns: Option<Py<PyAny>>,
        inplace: Option<bool>,
    ) -> PyResult<Self> {
        if inplace.unwrap_or(false) {
            return Err(PyNotImplementedError::new_err(
                "drop(inplace=True) is not supported; use the returned DataFrame",
            ));
        }
        let axis = axis.unwrap_or(0);

        let mut drop_cols: Vec<String> = Vec::new();
        let mut drop_rows: Vec<usize> = Vec::new();

        if let Some(cols) = columns {
            drop_cols.extend(extract_string_or_list(py, &cols)?);
        }
        if let Some(idx) = index {
            drop_rows.extend(extract_usize_or_list(py, &idx)?);
        }
        if let Some(lab) = labels {
            if axis == 1 {
                drop_cols.extend(extract_string_or_list(py, &lab)?);
            } else {
                drop_rows.extend(extract_usize_or_list(py, &lab)?);
            }
        }

        if drop_cols.is_empty() && drop_rows.is_empty() {
            return Err(PyValueError::new_err(
                "drop requires `labels`, `index`, or `columns`",
            ));
        }

        let df = self.inner.rs_df();
        for col in &drop_cols {
            if !df.column_names().iter().any(|c| c == col) {
                return Err(PyKeyError::new_err(format!("column '{}' not found", col)));
            }
        }

        // Column drop via select_columns of the survivors (keeps types + index).
        let survivors: Vec<String> = df
            .column_names()
            .iter()
            .filter(|c| !drop_cols.contains(c))
            .cloned()
            .collect();
        let survivor_refs: Vec<&str> = survivors.iter().map(|s| s.as_str()).collect();
        let mut result = df.select_columns(&survivor_refs).map_err(to_py_value_err)?;

        // Row drop by position.
        if !drop_rows.is_empty() {
            let n = result.row_count();
            for &p in &drop_rows {
                if p >= n {
                    return Err(PyIndexError::new_err(format!(
                        "row label {} out of range for {} rows",
                        p, n
                    )));
                }
            }
            let keep: Vec<usize> = (0..n).filter(|p| !drop_rows.contains(p)).collect();
            result = select_rows_by_position(&result, &keep)?;
        }

        Ok(Self {
            inner: PyDataFrame::from_df(result),
        })
    }

    /// Merge with another DataFrame on a shared key column (real join).
    ///
    /// The underlying join takes a single key that is present under the same
    /// name in both frames. Unsupported pandas options (`left_index`,
    /// `right_index`, `sort`, differing `left_on`/`right_on`, multi-key `on`)
    /// raise rather than being ignored. When `on` is omitted it is inferred
    /// from the single shared column, if unambiguous.
    #[pyo3(signature = (
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=false,
        right_index=false,
        sort=false,
        suffixes=("_x".to_string(), "_y".to_string())
    ))]
    #[allow(clippy::too_many_arguments)]
    fn merge(
        &self,
        py: Python<'_>,
        right: &Self,
        how: Option<&str>,
        on: Option<Py<PyAny>>,
        left_on: Option<Py<PyAny>>,
        right_on: Option<Py<PyAny>>,
        left_index: Option<bool>,
        right_index: Option<bool>,
        sort: Option<bool>,
        suffixes: Option<(String, String)>,
    ) -> PyResult<Self> {
        if left_index.unwrap_or(false) || right_index.unwrap_or(false) {
            return Err(PyNotImplementedError::new_err(
                "merge on index (left_index/right_index) is not supported; merge on a shared key column",
            ));
        }
        if sort.unwrap_or(false) {
            return Err(PyNotImplementedError::new_err(
                "merge(sort=True) is not supported",
            ));
        }

        let join = match how.unwrap_or("inner") {
            "inner" => JoinType::Inner,
            "left" => JoinType::Left,
            "right" => JoinType::Right,
            "outer" => JoinType::Outer,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown join type '{}'",
                    other
                )))
            }
        };

        let left_df = self.inner.rs_df();
        let right_df = right.inner.rs_df();

        let key: String = if let Some(on_obj) = on {
            let names = extract_string_or_list(py, &on_obj)?;
            if names.len() != 1 {
                return Err(PyNotImplementedError::new_err(
                    "merge supports a single `on` key column",
                ));
            }
            names
                .into_iter()
                .next()
                .ok_or_else(|| PyValueError::new_err("empty `on` key"))?
        } else {
            match (left_on, right_on) {
                (Some(l), Some(r)) => {
                    let left_names = extract_string_or_list(py, &l)?;
                    let right_names = extract_string_or_list(py, &r)?;
                    if left_names.len() != 1
                        || right_names.len() != 1
                        || left_names[0] != right_names[0]
                    {
                        return Err(PyNotImplementedError::new_err(
                            "merge requires a single shared key; left_on/right_on with differing names is not supported",
                        ));
                    }
                    left_names
                        .into_iter()
                        .next()
                        .ok_or_else(|| PyValueError::new_err("empty key"))?
                }
                (None, None) => {
                    let common: Vec<String> = left_df
                        .column_names()
                        .iter()
                        .filter(|c| right_df.column_names().iter().any(|x| x == *c))
                        .cloned()
                        .collect();
                    match common.len() {
                        1 => common
                            .into_iter()
                            .next()
                            .ok_or_else(|| PyValueError::new_err("empty key"))?,
                        0 => {
                            return Err(PyValueError::new_err(
                                "no common columns to merge on; specify `on`",
                            ))
                        }
                        _ => {
                            return Err(PyValueError::new_err(format!(
                                "ambiguous merge: multiple common columns {:?}; specify `on`",
                                common
                            )))
                        }
                    }
                }
                _ => {
                    return Err(PyNotImplementedError::new_err(
                        "merge requires both left_on and right_on, or neither",
                    ))
                }
            }
        };

        let (left_suffix, right_suffix) =
            suffixes.unwrap_or_else(|| ("_x".to_string(), "_y".to_string()));
        let result = pandrs_merge(
            left_df,
            right_df,
            &key,
            join,
            (left_suffix.as_str(), right_suffix.as_str()),
        )
        .map_err(to_py_value_err)?;

        Ok(Self {
            inner: PyDataFrame::from_df(result),
        })
    }

    /// Write to CSV, honouring `sep`, `na_rep`, `index` and `header`.
    ///
    /// With a path the real CSV is written to disk; without one the CSV text is
    /// returned. (A genuine value equal to the `"NA"` marker is emitted as
    /// `na_rep`, since it is indistinguishable from missing in this model.)
    #[pyo3(signature = (path_or_buf=None, sep=",", na_rep="", index=true, header=true))]
    fn to_csv(
        &self,
        path_or_buf: Option<&str>,
        sep: Option<&str>,
        na_rep: Option<&str>,
        index: Option<bool>,
        header: Option<bool>,
    ) -> PyResult<Option<String>> {
        let content = build_csv(
            self.inner.rs_df(),
            sep.unwrap_or(","),
            na_rep.unwrap_or(""),
            index.unwrap_or(true),
            header.unwrap_or(true),
        )?;

        if let Some(path) = path_or_buf {
            std::fs::write(path, &content)
                .map_err(|e| PyValueError::new_err(format!("Failed to write CSV: {}", e)))?;
            Ok(None)
        } else {
            Ok(Some(content))
        }
    }

    /// Read a CSV file into a DataFrame (real parser).
    ///
    /// Only the default `,` separator and header inference are supported;
    /// `sep != ","`, `names`, and `index_col` raise rather than being ignored.
    #[classmethod]
    #[pyo3(signature = (filepath_or_buffer, sep=",", header="infer", names=None, index_col=None))]
    fn read_csv(
        _cls: &Bound<'_, PyType>,
        filepath_or_buffer: &str,
        sep: Option<&str>,
        header: Option<&str>,
        names: Option<Py<PyAny>>,
        index_col: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if let Some(s) = sep {
            if s != "," {
                return Err(PyNotImplementedError::new_err(
                    "read_csv only supports the default ',' separator",
                ));
            }
        }
        if names.is_some() {
            return Err(PyNotImplementedError::new_err(
                "read_csv(names=...) is not supported",
            ));
        }
        if index_col.is_some() {
            return Err(PyNotImplementedError::new_err(
                "read_csv(index_col=...) is not supported",
            ));
        }
        let has_header = !matches!(header, Some("none") | Some("None"));

        let df = DataFrame::from_csv(filepath_or_buffer, has_header)
            .map_err(|e| PyValueError::new_err(format!("Failed to read CSV: {}", e)))?;
        Ok(Self {
            inner: PyDataFrame::from_df(df),
        })
    }

    /// DataFrame shape `(rows, columns)`.
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.rs_len(), self.inner.rs_column_names().len())
    }

    /// Column names.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.rs_column_names()
    }

    /// Rename every column (validates that the count matches).
    #[setter]
    fn set_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        self.inner.rs_set_column_names(columns)
    }

    /// Row index. This frame always carries a default positional index (the
    /// constructor rejects a custom `index=`), so this is `0..n`.
    #[getter]
    fn index(&self) -> Vec<usize> {
        (0..self.inner.rs_len()).collect()
    }

    /// Return one column as a `PandasSeries` of its real string values.
    fn get_column(&self, py: Python<'_>, column_name: &str) -> PyResult<Py<PyAny>> {
        let df = self.inner.rs_df();
        if !df.column_names().iter().any(|c| c == column_name) {
            return Err(PyKeyError::new_err(format!(
                "column '{}' not found",
                column_name
            )));
        }
        let values = df
            .get_column_string_values(column_name)
            .map_err(to_py_value_err)?;
        let series = PandasSeries::new(values, Some(column_name.to_string()))?;
        Ok(Py::new(py, series)?.into_any())
    }

    /// Return a DataFrame with only the selected columns (real projection).
    fn select_columns(&self, py: Python<'_>, column_names: Vec<String>) -> PyResult<Py<PyAny>> {
        let refs: Vec<&str> = column_names.iter().map(|s| s.as_str()).collect();
        let sub = self
            .inner
            .rs_df()
            .select_columns(&refs)
            .map_err(|e| PyKeyError::new_err(e.to_string()))?;
        let compat = Self {
            inner: PyDataFrame::from_df(sub),
        };
        Ok(Py::new(py, compat)?.into_any())
    }
}

// ---------------------------------------------------------------------------
// PandasSeries
// ---------------------------------------------------------------------------

/// Pandas-compatible Series (a labelled 1-D string array).
#[pyclass(name = "PandasSeries")]
pub struct PandasSeries {
    data: Vec<String>,
    name: Option<String>,
    /// Optional index labels (e.g. the distinct values from `value_counts`).
    /// `None` means a default positional index.
    index: Option<Vec<String>>,
}

#[pymethods]
impl PandasSeries {
    #[new]
    #[pyo3(signature = (data, name=None))]
    fn new(data: Vec<String>, name: Option<String>) -> PyResult<Self> {
        Ok(Self {
            data,
            name,
            index: None,
        })
    }

    /// First `n` values (index labels sliced alongside, if present).
    #[pyo3(signature = (n=5))]
    fn head(&self, n: Option<usize>) -> PyResult<Self> {
        let n = n.unwrap_or(5);
        let data = self.data.iter().take(n).cloned().collect();
        let index = self
            .index
            .as_ref()
            .map(|ix| ix.iter().take(n).cloned().collect());
        Ok(Self {
            data,
            name: self.name.clone(),
            index,
        })
    }

    /// Count occurrences of each distinct value.
    ///
    /// Returns a Series of counts indexed by the counted value (like pandas),
    /// ordered by descending count then value so the result is deterministic.
    fn value_counts(&self) -> PyResult<Self> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for value in &self.data {
            *counts.entry(value.clone()).or_insert(0) += 1;
        }
        let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let index: Vec<String> = pairs.iter().map(|(value, _)| value.clone()).collect();
        let data: Vec<String> = pairs.iter().map(|(_, count)| count.to_string()).collect();
        Ok(Self {
            data,
            name: Some("count".to_string()),
            index: Some(index),
        })
    }

    /// Series length.
    fn __len__(&self) -> usize {
        self.data.len()
    }

    /// Positional element access.
    fn __getitem__(&self, index: usize) -> PyResult<String> {
        self.data
            .get(index)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("Index out of bounds"))
    }

    /// Index labels (default positional if none were set).
    #[getter]
    fn index(&self) -> Vec<String> {
        self.index
            .clone()
            .unwrap_or_else(|| (0..self.data.len()).map(|i| i.to_string()).collect())
    }

    /// Underlying values.
    #[getter]
    fn values(&self) -> Vec<String> {
        self.data.clone()
    }

    /// Series name.
    #[getter]
    fn name(&self) -> Option<String> {
        self.name.clone()
    }

    /// String representation (shows index labels when present).
    fn __str__(&self) -> String {
        match &self.index {
            Some(labels) => {
                let body: Vec<String> = labels
                    .iter()
                    .zip(&self.data)
                    .map(|(label, value)| format!("{}    {}", label, value))
                    .collect();
                format!(
                    "{}\nName: {}",
                    body.join("\n"),
                    self.name.clone().unwrap_or_default()
                )
            }
            None => format!("PandasSeries({:?}, name={:?})", self.data, self.name),
        }
    }
}

// ---------------------------------------------------------------------------
// iloc / loc indexers
// ---------------------------------------------------------------------------

/// Position-based indexer for `PandasDataFrame`.
#[pyclass(name = "ILocIndexer")]
pub struct ILocIndexer {
    dataframe: PandasCompatibleDataFrame,
}

#[pymethods]
impl ILocIndexer {
    /// Select rows by integer position: an int, a list of ints, or a slice.
    /// Returns a DataFrame of the selected rows.
    fn __getitem__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let df = self.dataframe.inner.rs_df();
        let positions = resolve_positions(py, &key, df.row_count())?;
        let selected = select_rows_by_position(df, &positions)?;
        let compat = PandasCompatibleDataFrame {
            inner: PyDataFrame::from_df(selected),
        };
        Ok(Py::new(py, compat)?.into_any())
    }
}

/// Label-based indexer for `PandasDataFrame`.
#[pyclass(name = "LocIndexer")]
pub struct LocIndexer {
    dataframe: PandasCompatibleDataFrame,
}

#[pymethods]
impl LocIndexer {
    /// Select rows by label. For the default positional index this coincides
    /// with position-based selection; an explicit non-positional index raises
    /// (there is no public label-based row-selection API to wire to).
    fn __getitem__(&self, py: Python<'_>, key: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let df = self.dataframe.inner.rs_df();
        if !index_is_positional(df) {
            return Err(PyNotImplementedError::new_err(
                "label-based .loc is only supported on a default positional index; \
                 this frame has an explicit index -- use .iloc for positional access",
            ));
        }
        let positions = resolve_positions(py, &key, df.row_count())?;
        let selected = select_rows_by_position(df, &positions)?;
        let compat = PandasCompatibleDataFrame {
            inner: PyDataFrame::from_df(selected),
        };
        Ok(Py::new(py, compat)?.into_any())
    }
}

// ---------------------------------------------------------------------------
// GroupBy
// ---------------------------------------------------------------------------

/// GroupBy object whose aggregations are computed by real pandrs code.
#[pyclass(name = "GroupBy")]
pub struct GroupBy {
    dataframe: PandasCompatibleDataFrame,
    group_keys: Vec<String>,
}

impl GroupBy {
    fn key_refs(&self) -> Vec<&str> {
        self.group_keys.iter().map(|s| s.as_str()).collect()
    }

    /// Run a numeric aggregation via the real pandrs GroupBy. Non-key columns
    /// are numeric-typed first so the aggregators can see them.
    fn aggregate_numeric(&self, op: &str) -> PyResult<PandasCompatibleDataFrame> {
        let typed = numeric_typed(self.dataframe.inner.rs_df(), &self.group_keys)?;
        let groupby = PandrsGroupBy::new(&typed, &self.key_refs()).map_err(to_py_value_err)?;
        let result = match op {
            "sum" => groupby.sum(),
            "mean" => groupby.mean(),
            "min" => groupby.min(),
            "max" => groupby.max(),
            "std" => groupby.std(),
            "var" => groupby.var(),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown aggregation '{}'",
                    other
                )))
            }
        }
        .map_err(to_py_value_err)?;
        Ok(PandasCompatibleDataFrame {
            inner: PyDataFrame::from_df(result),
        })
    }
}

#[pymethods]
impl GroupBy {
    /// Group sizes (row count per group) as a `(group..., size)` DataFrame.
    fn size(&self) -> PyResult<PandasCompatibleDataFrame> {
        let df = self.dataframe.inner.rs_df();
        let groupby = PandrsGroupBy::new(df, &self.key_refs()).map_err(to_py_value_err)?;
        let result = groupby.size().map_err(to_py_value_err)?;
        Ok(PandasCompatibleDataFrame {
            inner: PyDataFrame::from_df(result),
        })
    }

    /// Non-null counts per group and column.
    fn count(&self) -> PyResult<PandasCompatibleDataFrame> {
        let df = self.dataframe.inner.rs_df();
        let groupby = PandrsGroupBy::new(df, &self.key_refs()).map_err(to_py_value_err)?;
        let result = groupby.count().map_err(to_py_value_err)?;
        Ok(PandasCompatibleDataFrame {
            inner: PyDataFrame::from_df(result),
        })
    }

    /// Sum of numeric columns per group.
    fn sum(&self) -> PyResult<PandasCompatibleDataFrame> {
        self.aggregate_numeric("sum")
    }

    /// Mean of numeric columns per group.
    fn mean(&self) -> PyResult<PandasCompatibleDataFrame> {
        self.aggregate_numeric("mean")
    }

    /// Minimum of numeric columns per group.
    fn min(&self) -> PyResult<PandasCompatibleDataFrame> {
        self.aggregate_numeric("min")
    }

    /// Maximum of numeric columns per group.
    fn max(&self) -> PyResult<PandasCompatibleDataFrame> {
        self.aggregate_numeric("max")
    }

    /// Sample standard deviation of numeric columns per group.
    fn std(&self) -> PyResult<PandasCompatibleDataFrame> {
        self.aggregate_numeric("std")
    }

    /// Sample variance of numeric columns per group.
    fn var(&self) -> PyResult<PandasCompatibleDataFrame> {
        self.aggregate_numeric("var")
    }

    /// Aggregate with a `{column: func}` (or `{column: [func, ...]}`) mapping,
    /// where each `func` is an aggregation name (`"sum"`, `"mean"`, ...).
    fn agg(&self, py: Python<'_>, func: Py<PyAny>) -> PyResult<PandasCompatibleDataFrame> {
        let bound = func.bind(py);
        let dict = bound.cast::<PyDict>().map_err(|_| {
            PyNotImplementedError::new_err(
                "groupby.agg currently accepts a dict {column: aggregation_name}",
            )
        })?;

        let mut pairs: Vec<(String, String)> = Vec::new();
        for (key, value) in dict.iter() {
            let column: String = key.extract()?;
            if let Ok(single) = value.extract::<String>() {
                pairs.push((column.clone(), single));
            } else if let Ok(many) = value.extract::<Vec<String>>() {
                for function in many {
                    pairs.push((column.clone(), function));
                }
            } else {
                return Err(PyNotImplementedError::new_err(
                    "agg values must be an aggregation name or a list of names",
                ));
            }
        }
        if pairs.is_empty() {
            return Err(PyValueError::new_err(
                "agg requires at least one column:function mapping",
            ));
        }

        let typed = numeric_typed(self.dataframe.inner.rs_df(), &self.group_keys)?;
        let groupby = PandrsGroupBy::new(&typed, &self.key_refs()).map_err(to_py_value_err)?;
        let pair_refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(column, function)| (column.as_str(), function.as_str()))
            .collect();
        let result = groupby.agg(&pair_refs).map_err(to_py_value_err)?;
        Ok(PandasCompatibleDataFrame {
            inner: PyDataFrame::from_df(result),
        })
    }
}

// ---------------------------------------------------------------------------
// Module registration & free functions
// ---------------------------------------------------------------------------

/// Register pandas compatibility types with the Python module.
pub fn register_pandas_types(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PandasCompatibleDataFrame>()?;
    m.add_class::<PandasSeries>()?;
    m.add_class::<ILocIndexer>()?;
    m.add_class::<LocIndexer>()?;
    m.add_class::<GroupBy>()?;

    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(concat, m)?)?;

    Ok(())
}

/// Pandas-style `read_csv` free function (real parser).
///
/// Options that cannot be honoured (`sep != ","`, `names`, `index_col`) raise
/// rather than being silently ignored; `header=None` (or `"none"`) disables the
/// header row.
#[pyfunction]
#[pyo3(signature = (filepath_or_buffer, **kwargs))]
fn read_csv(
    filepath_or_buffer: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PandasCompatibleDataFrame> {
    let mut has_header = true;

    if let Some(kw) = kwargs {
        if let Some(sep) = kw.get_item("sep")? {
            let sep: String = sep.extract()?;
            if sep != "," {
                return Err(PyNotImplementedError::new_err(
                    "read_csv only supports the default ',' separator",
                ));
            }
        }
        if kw.get_item("names")?.is_some() {
            return Err(PyNotImplementedError::new_err(
                "read_csv(names=...) is not supported",
            ));
        }
        if kw.get_item("index_col")?.is_some() {
            return Err(PyNotImplementedError::new_err(
                "read_csv(index_col=...) is not supported",
            ));
        }
        if let Some(header) = kw.get_item("header")? {
            if header.is_none() {
                has_header = false;
            } else if let Ok(header_str) = header.extract::<String>() {
                if header_str == "none" || header_str == "None" {
                    has_header = false;
                }
            }
        }
    }

    let df = DataFrame::from_csv(filepath_or_buffer, has_header)
        .map_err(|e| PyValueError::new_err(format!("Failed to read CSV: {}", e)))?;
    Ok(PandasCompatibleDataFrame {
        inner: PyDataFrame::from_df(df),
    })
}

/// Pandas-style `concat` free function (real concatenation).
///
/// `axis=0` stacks rows, `axis=1` stacks columns.
#[pyfunction]
#[pyo3(signature = (objs, axis=0, ignore_index=false))]
fn concat(
    py: Python<'_>,
    objs: Vec<Py<PandasCompatibleDataFrame>>,
    axis: Option<i32>,
    ignore_index: Option<bool>,
) -> PyResult<PandasCompatibleDataFrame> {
    if objs.is_empty() {
        return Err(PyValueError::new_err("No objects to concatenate"));
    }
    let axis = match axis.unwrap_or(0) {
        0 => ConcatAxis::Rows,
        1 => ConcatAxis::Columns,
        other => return Err(PyValueError::new_err(format!("invalid axis {}", other))),
    };

    // Keep each borrow alive for the duration of the concat call.
    let borrows: Vec<PyRef<'_, PandasCompatibleDataFrame>> =
        objs.iter().map(|o| o.bind(py).borrow()).collect();
    let dfs: Vec<&DataFrame> = borrows.iter().map(|b| b.inner.rs_df()).collect();

    let result =
        pandrs_concat(&dfs, axis, ignore_index.unwrap_or(false)).map_err(to_py_value_err)?;
    Ok(PandasCompatibleDataFrame {
        inner: PyDataFrame::from_df(result),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_from(pairs: &[(&str, &[&str])]) -> DataFrame {
        let mut data: HashMap<String, Vec<String>> = HashMap::new();
        for (name, values) in pairs {
            data.insert(
                name.to_string(),
                values.iter().map(|s| s.to_string()).collect(),
            );
        }
        DataFrame::from_map(data, None).expect("failed to build test frame")
    }

    #[test]
    fn numeric_typed_preserves_na_and_converts() {
        // A missing ("NA") cell must NOT drop the column; it becomes NaN.
        let df = frame_from(&[("a", &["1", "NA", "3"])]);
        let typed = numeric_typed(&df, &[]).expect("numeric_typed");
        assert!(typed.is_numeric_column("a"), "column should become numeric");

        let stats = describe_column(&typed, "a").expect("describe");
        assert_eq!(stats.get("count").copied(), Some(2.0));
        assert_eq!(stats.get("mean").copied(), Some(2.0));
    }

    #[test]
    fn numeric_typed_keeps_non_numeric_and_keys() {
        let df = frame_from(&[("k", &["001", "002"]), ("txt", &["x", "y"])]);
        // Keep "k" as string (group key): a zero-padded code must stay text.
        let typed = numeric_typed(&df, &["k".to_string()]).expect("numeric_typed");
        assert!(!typed.is_numeric_column("k"), "kept key must stay string");
        assert!(!typed.is_numeric_column("txt"), "text column stays string");
        assert_eq!(
            typed.get_column_string_values("k").unwrap(),
            vec!["001", "002"]
        );
    }

    #[test]
    fn build_csv_emits_real_values_and_honours_options() {
        let df = frame_from(&[("a", &["1", "2"]), ("b", &["x", "y"])]);

        let with_index = build_csv(&df, ",", "", true, true).expect("csv");
        // Header has a leading empty index cell; rows are prefixed by position.
        assert_eq!(with_index, ",a,b\n0,1,x\n1,2,y\n");

        let no_index = build_csv(&df, ";", "", false, true).expect("csv");
        assert_eq!(no_index, "a;b\n1;x\n2;y\n");

        let no_header = build_csv(&df, ",", "", false, false).expect("csv");
        assert_eq!(no_header, "1,x\n2,y\n");
    }

    #[test]
    fn build_csv_maps_na_marker_and_quotes() {
        let df = frame_from(&[("a", &["NA", "b,c"])]);
        let csv = build_csv(&df, ",", "NULL", false, true).expect("csv");
        // "NA" -> na_rep; a value containing the separator is quoted.
        assert_eq!(csv, "a\nNULL\n\"b,c\"\n");
    }

    #[test]
    fn index_is_positional_detects_default() {
        let df = frame_from(&[("a", &["1", "2", "3"])]);
        assert!(index_is_positional(&df));
    }

    #[test]
    fn query_engine_filters_numeric_predicate_on_string_frame() {
        // The real query engine handles numeric predicates over the
        // string-typed columns these bindings store (it parses on the fly), so
        // `PandasDataFrame.query` genuinely filters rather than erroring.
        let df = frame_from(&[("a", &["1", "2", "3"]), ("b", &["x", "y", "z"])]);
        let out = df.query("a > 2").expect("numeric query over string column");
        assert_eq!(out.row_count(), 1);
        assert_eq!(out.get_column_string_values("b").unwrap(), vec!["z"]);
    }

    #[test]
    fn select_rows_by_position_selects_and_reorders() {
        let df = frame_from(&[("a", &["10", "20", "30"]), ("b", &["p", "q", "r"])]);
        let sub = select_rows_by_position(&df, &[2, 0]).expect("select");
        assert_eq!(sub.row_count(), 2);
        assert_eq!(sub.get_column_string_values("a").unwrap(), vec!["30", "10"]);
        assert_eq!(sub.get_column_string_values("b").unwrap(), vec!["r", "p"]);
    }
}
