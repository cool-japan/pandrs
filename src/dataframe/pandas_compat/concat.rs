//! DataFrame concatenation operations
//!
//! Provides pandas-compatible concat functionality for combining DataFrames.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::join::{element_type, gather_column, numeric_values, ElemType, NA_STRING};
use crate::index::{DataFrameIndex, Index, MultiIndex};
use crate::series::Series;

/// Axis for concatenation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcatAxis {
    /// Concatenate along rows (stack vertically, axis=0 in pandas)
    Rows,
    /// Concatenate along columns (stack horizontally, axis=1 in pandas)
    Columns,
}

/// Concatenate DataFrames along an axis
///
/// # Arguments
/// * `dfs` - Slice of DataFrames to concatenate
/// * `axis` - Axis along which to concatenate (Rows or Columns)
/// * `ignore_index` - If true, do not use the existing labels along the
///   concatenation axis
///
/// # Column types (axis = Rows)
///
/// The output type of a column is decided from **every** input frame, never
/// from whichever frame happens to come first:
///
/// * all frames agree on the concrete element type -> that type is kept exactly
///   (so a text column of `"007"` stays `"007"` instead of being re-inferred
///   into the float `7.0`);
/// * the frames disagree but every one of them is numeric -> `f64`
///   (pandas' int/float upcast);
/// * the frames disagree otherwise -> text, with each frame's own values
///   rendered from its own dtype.
///
/// Because the decision uses all frames, `concat([a, b])` and `concat([b, a])`
/// now produce the same column types.
///
/// # Missing columns (axis = Rows)
///
/// Rows contributed by a frame that lacks a column are missing, not zero and
/// not empty text: float columns use `NaN`, integer and boolean columns widen
/// to `f64` with `NaN` (as pandas does), and text/date columns use the textual
/// NA marker (`crate::series::Series` has no nullable string representation).
///
/// # `ignore_index`
///
/// * `axis = Rows`: `true` renumbers the rows (the result carries the implicit
///   positional index). `false` concatenates the input frames' row labels; when
///   no input frame carries an explicit index there is nothing to preserve and
///   the result likewise keeps the implicit positional index. Once at least one
///   frame has real labels they are all concatenated (frames without an index
///   contribute their positional labels), and because
///   [`crate::index::Index`] rejects duplicate labels a collision is reported
///   as an error naming the label rather than silently renumbering; this is the
///   one place where pandas differs (it permits duplicate row labels).
/// * `axis = Columns`: the concatenation axis is the column labels, so `true`
///   replaces them with their positions (`"0"`, `"1"`, ...) and `false` keeps
///   the original names, disambiguating repeats as `name_1`, `name_2`, ... The
///   row index is carried over from the first frame that has one.
///
/// # Returns
/// Concatenated DataFrame
///
/// # Example
/// ```ignore
/// use pandrs::dataframe::pandas_compat::{concat, ConcatAxis};
///
/// let df1 = DataFrame::new(); // ... populate
/// let df2 = DataFrame::new(); // ... populate
/// let result = concat(&[&df1, &df2], ConcatAxis::Rows, true)?;
/// ```
pub fn concat(dfs: &[&DataFrame], axis: ConcatAxis, ignore_index: bool) -> Result<DataFrame> {
    if dfs.is_empty() {
        return Ok(DataFrame::new());
    }

    // A single frame is returned unchanged only when its labels are kept; with
    // `ignore_index` the general path rebuilds it so the labels are dropped.
    if dfs.len() == 1 && !ignore_index {
        return Ok(dfs[0].clone());
    }

    match axis {
        ConcatAxis::Rows => concat_rows(dfs, ignore_index),
        ConcatAxis::Columns => concat_columns(dfs, ignore_index),
    }
}

/// Concatenate DataFrames vertically (row-wise)
fn concat_rows(dfs: &[&DataFrame], ignore_index: bool) -> Result<DataFrame> {
    // Collect all unique column names in order
    let mut all_columns: Vec<String> = Vec::new();
    for df in dfs {
        for col in df.column_names() {
            if !all_columns.contains(col) {
                all_columns.push(col.clone());
            }
        }
    }

    let mut result = DataFrame::new();
    for col in &all_columns {
        stack_column(&mut result, dfs, col)?;
    }

    if !ignore_index && result.row_count() > 0 {
        apply_concatenated_index(&mut result, dfs)?;
    }

    Ok(result)
}

/// Append one column, stacked across every frame, to `out`.
fn stack_column(out: &mut DataFrame, dfs: &[&DataFrame], column: &str) -> Result<()> {
    // Element type of the column in every frame that has it, plus whether any
    // frame that actually contributes rows is missing it. A frame with no rows
    // contributes nothing, so it cannot force an NA fill.
    let mut present: Vec<ElemType> = Vec::new();
    let mut missing_somewhere = false;
    for df in dfs {
        if df.contains_column(column) {
            present.push(element_type(df, column)?);
        } else if df.row_count() > 0 {
            missing_somewhere = true;
        }
    }

    let first = match present.first() {
        Some(elem) => *elem,
        None => {
            return Err(Error::ColumnNotFound(format!(
                "Internal error: column '{}' was collected for concatenation but no DataFrame \
                 contains it",
                column
            )))
        }
    };
    let all_same = present.iter().all(|elem| *elem == first);

    if all_same && !missing_somewhere {
        return stack_exact_dispatch(out, dfs, column, first);
    }

    if all_same {
        return match first {
            ElemType::F64 => stack_with_na(out, dfs, column, f64::NAN),
            ElemType::F32 => stack_with_na(out, dfs, column, f32::NAN),
            elem if elem.is_integer() || elem == ElemType::Bool => stack_widened(out, dfs, column),
            // String / date / time have no in-band NA representation.
            _ => stack_rendered(out, dfs, column),
        };
    }

    // The frames disagree about this column's element type.
    if present.iter().all(|elem| elem.is_numeric()) {
        return stack_widened(out, dfs, column);
    }
    stack_rendered(out, dfs, column)
}

/// Stack a column that every contributing frame holds with the same concrete
/// element type, preserving that type exactly.
fn stack_exact_dispatch(
    out: &mut DataFrame,
    dfs: &[&DataFrame],
    column: &str,
    elem: ElemType,
) -> Result<()> {
    match elem {
        ElemType::Str => stack_exact::<String>(out, dfs, column),
        ElemType::Bool => stack_exact::<bool>(out, dfs, column),
        ElemType::I8 => stack_exact::<i8>(out, dfs, column),
        ElemType::I16 => stack_exact::<i16>(out, dfs, column),
        ElemType::I32 => stack_exact::<i32>(out, dfs, column),
        ElemType::I64 => stack_exact::<i64>(out, dfs, column),
        ElemType::I128 => stack_exact::<i128>(out, dfs, column),
        ElemType::ISize => stack_exact::<isize>(out, dfs, column),
        ElemType::U8 => stack_exact::<u8>(out, dfs, column),
        ElemType::U16 => stack_exact::<u16>(out, dfs, column),
        ElemType::U32 => stack_exact::<u32>(out, dfs, column),
        ElemType::U64 => stack_exact::<u64>(out, dfs, column),
        ElemType::U128 => stack_exact::<u128>(out, dfs, column),
        ElemType::USize => stack_exact::<usize>(out, dfs, column),
        ElemType::F32 => stack_exact::<f32>(out, dfs, column),
        ElemType::F64 => stack_exact::<f64>(out, dfs, column),
        ElemType::Date => stack_exact::<chrono::NaiveDate>(out, dfs, column),
        ElemType::DateTime => stack_exact::<chrono::NaiveDateTime>(out, dfs, column),
        ElemType::DateTimeUtc => stack_exact::<chrono::DateTime<chrono::Utc>>(out, dfs, column),
    }
}

fn stack_exact<T>(out: &mut DataFrame, dfs: &[&DataFrame], column: &str) -> Result<()>
where
    T: 'static + Debug + Clone + Send + Sync,
{
    let mut values: Vec<T> = Vec::new();
    for df in dfs {
        if df.contains_column(column) {
            values.extend_from_slice(df.get_column::<T>(column)?.values());
        }
        // Frames without the column contribute no rows on this path: the
        // caller only selects it when no frame with rows lacks the column.
    }
    out.add_column(
        column.to_string(),
        Series::new(values, Some(column.to_string()))?,
    )
}

/// Stack a column whose element type carries NA in-band (`f32`/`f64`).
fn stack_with_na<T>(
    out: &mut DataFrame,
    dfs: &[&DataFrame],
    column: &str,
    na_value: T,
) -> Result<()>
where
    T: 'static + Debug + Clone + Send + Sync,
{
    let mut values: Vec<T> = Vec::new();
    for df in dfs {
        if df.contains_column(column) {
            values.extend_from_slice(df.get_column::<T>(column)?.values());
        } else {
            values.extend((0..df.row_count()).map(|_| na_value.clone()));
        }
    }
    out.add_column(
        column.to_string(),
        Series::new(values, Some(column.to_string()))?,
    )
}

/// Stack a numeric column as `f64` (used for integer/boolean columns that need
/// an NA fill, and for frames that disagree between integer and float dtypes).
fn stack_widened(out: &mut DataFrame, dfs: &[&DataFrame], column: &str) -> Result<()> {
    let mut values: Vec<f64> = Vec::new();
    for df in dfs {
        if df.contains_column(column) {
            let elem = element_type(df, column)?;
            values.extend(numeric_values(df, column, elem)?);
        } else {
            values.extend((0..df.row_count()).map(|_| f64::NAN));
        }
    }
    out.add_column(
        column.to_string(),
        Series::new(values, Some(column.to_string()))?,
    )
}

/// Stack a column as text. Each frame renders its own values from its own
/// dtype, so a text frame's values survive verbatim.
fn stack_rendered(out: &mut DataFrame, dfs: &[&DataFrame], column: &str) -> Result<()> {
    let mut values: Vec<String> = Vec::new();
    for df in dfs {
        if df.contains_column(column) {
            values.extend(df.get_column_string_values(column)?);
        } else {
            values.extend((0..df.row_count()).map(|_| NA_STRING.to_string()));
        }
    }
    out.add_column(
        column.to_string(),
        Series::new(values, Some(column.to_string()))?,
    )
}

/// The frame's own row labels, if it carries an explicit index of the right
/// length. `None` means the frame uses the implicit positional index.
fn explicit_labels(df: &DataFrame) -> Option<Vec<String>> {
    df.get_index()
        .string_values()
        .filter(|values| !values.is_empty() && values.len() == df.row_count())
}

/// Row labels of one frame: its explicit index when it has one, otherwise the
/// implicit positional labels.
fn row_labels(df: &DataFrame) -> Vec<String> {
    explicit_labels(df).unwrap_or_else(|| (0..df.row_count()).map(|i| i.to_string()).collect())
}

/// Build the concatenated row index for `ignore_index = false`.
fn apply_concatenated_index(result: &mut DataFrame, dfs: &[&DataFrame]) -> Result<()> {
    // A multi-level index can only be concatenated with other multi-level
    // indices of the same depth.
    let multi_indices: Vec<Option<MultiIndex<String>>> = dfs
        .iter()
        .map(|df| match df.get_index() {
            DataFrameIndex::Multi(index) if index.len() == df.row_count() => Some(index),
            _ => None,
        })
        .collect();
    let with_rows: Vec<usize> = (0..dfs.len()).filter(|i| dfs[*i].row_count() > 0).collect();

    if with_rows.iter().any(|i| multi_indices[*i].is_some()) {
        let depths: Vec<usize> = with_rows
            .iter()
            .map(|i| multi_indices[*i].as_ref().map(MultiIndex::n_levels))
            .collect::<Option<Vec<usize>>>()
            .ok_or_else(|| {
                Error::InvalidValue(
                    "concat(ignore_index = false): cannot combine a multi-level row index with a \
                     single-level one; give every DataFrame the same kind of index or pass \
                     ignore_index = true"
                        .to_string(),
                )
            })?;
        if depths.windows(2).any(|w| w[0] != w[1]) {
            return Err(Error::InvalidValue(
                "concat(ignore_index = false): the multi-level row indices have different numbers \
                 of levels and cannot be concatenated; pass ignore_index = true"
                    .to_string(),
            ));
        }

        let mut names: Vec<Option<String>> = Vec::new();
        let mut tuples: Vec<Vec<String>> = Vec::new();
        for i in &with_rows {
            if let Some(index) = &multi_indices[*i] {
                if names.is_empty() {
                    names = index.names().to_vec();
                }
                tuples.extend(index.tuples());
            }
        }
        let multi = MultiIndex::from_tuples(tuples, Some(names))?;
        return result.set_multi_index(multi);
    }

    // None of the frames carries labels of its own, so there is nothing to
    // preserve: leave the result on the implicit positional index instead of
    // synthesising per-frame labels that would immediately collide.
    if !dfs.iter().any(|df| explicit_labels(df).is_some()) {
        return Ok(());
    }

    let mut labels: Vec<String> = Vec::new();
    for df in dfs {
        labels.extend(row_labels(df));
    }

    let mut seen: HashSet<&str> = HashSet::with_capacity(labels.len());
    for label in &labels {
        if !seen.insert(label.as_str()) {
            return Err(Error::InvalidValue(format!(
                "concat(ignore_index = false) would produce the duplicate row label '{}'. This \
                 DataFrame's index requires unique labels (pandas allows duplicates here); pass \
                 ignore_index = true to renumber the rows, or give the inputs disjoint index \
                 labels.",
                label
            )));
        }
    }

    result.set_index(Index::new(labels)?)
}

/// Concatenate DataFrames horizontally (column-wise)
fn concat_columns(dfs: &[&DataFrame], ignore_index: bool) -> Result<DataFrame> {
    // Verify all DataFrames have the same number of rows
    let row_counts: Vec<usize> = dfs.iter().map(|df| df.row_count()).collect();
    if !row_counts.windows(2).all(|w| w[0] == w[1]) {
        return Err(Error::InvalidValue(
            "All DataFrames must have the same number of rows for column-wise concatenation"
                .to_string(),
        ));
    }
    let row_count = row_counts.first().copied().unwrap_or(0);
    let rows: Vec<Option<usize>> = (0..row_count).map(Some).collect();

    let mut result = DataFrame::new();

    // Track column names to handle duplicates
    let mut seen_columns: HashMap<String, usize> = HashMap::new();
    let mut position = 0usize;

    for df in dfs {
        for col_name in df.column_names() {
            let final_name = if ignore_index {
                position.to_string()
            } else if let Some(count) = seen_columns.get_mut(col_name.as_str()) {
                *count += 1;
                format!("{}_{}", col_name, count)
            } else {
                seen_columns.insert(col_name.to_string(), 0);
                col_name.to_string()
            };
            position += 1;

            // Copy the column with its concrete element type intact; going
            // through a numeric-then-string re-inference used to turn a text
            // column of "007" into the float 7.0.
            gather_column(&mut result, &final_name, df, col_name, &rows)?;
        }
    }

    // The row labels are not the concatenation axis here, so they survive:
    // adopt them from the first frame that carries an index.
    if result.row_count() > 0 {
        for df in dfs {
            match df.get_index() {
                DataFrameIndex::Simple(index) if index.len() == row_count && !index.is_empty() => {
                    result.set_index(index)?;
                    break;
                }
                DataFrameIndex::Multi(index) if index.len() == row_count && !index.is_empty() => {
                    result.set_multi_index(index)?;
                    break;
                }
                _ => {}
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_df(name: &str, values: Vec<f64>) -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            name.to_string(),
            Series::new(values, Some(name.to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df
    }

    fn string_df(name: &str, values: Vec<&str>) -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            name.to_string(),
            Series::new(
                values.into_iter().map(str::to_string).collect::<Vec<_>>(),
                Some(name.to_string()),
            )
            .expect("test should succeed"),
        )
        .expect("test should succeed");
        df
    }

    #[test]
    fn test_concat_rows_same_columns() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "a".to_string(),
            Series::new(vec![1.0, 2.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df1.add_column(
            "b".to_string(),
            Series::new(vec![10.0, 20.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");

        let mut df2 = DataFrame::new();
        df2.add_column(
            "a".to_string(),
            Series::new(vec![3.0, 4.0], Some("a".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        df2.add_column(
            "b".to_string(),
            Series::new(vec![30.0, 40.0], Some("b".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, true).expect("test should succeed");

        assert_eq!(result.row_count(), 4);
        let a_values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(a_values, vec![1.0, 2.0, 3.0, 4.0]);
        let b_values = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(b_values, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_concat_rows_different_columns() {
        let df1 = numeric_df("a", vec![1.0, 2.0]);
        let df2 = numeric_df("b", vec![30.0, 40.0]);

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, true).expect("test should succeed");

        assert_eq!(result.row_count(), 4);

        // Column 'a' should have values from df1, then NaN for df2
        let a_values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(a_values[0], 1.0);
        assert_eq!(a_values[1], 2.0);
        assert!(a_values[2].is_nan());
        assert!(a_values[3].is_nan());

        // Column 'b' should have NaN for df1, then values from df2
        let b_values = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert!(b_values[0].is_nan());
        assert!(b_values[1].is_nan());
        assert_eq!(b_values[2], 30.0);
        assert_eq!(b_values[3], 40.0);
    }

    #[test]
    fn test_concat_rows_string_columns() {
        let df1 = string_df("name", vec!["Alice", "Bob"]);
        let df2 = string_df("name", vec!["Charlie", "David"]);

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, true).expect("test should succeed");

        assert_eq!(result.row_count(), 4);
        let names = result
            .get_column_string_values("name")
            .expect("test should succeed");
        assert_eq!(names, vec!["Alice", "Bob", "Charlie", "David"]);
    }

    #[test]
    fn test_concat_columns_keeps_names_when_index_kept() {
        let df1 = numeric_df("a", vec![1.0, 2.0]);
        let df2 = numeric_df("b", vec![10.0, 20.0]);

        let result =
            concat(&[&df1, &df2], ConcatAxis::Columns, false).expect("test should succeed");

        assert_eq!(result.row_count(), 2);
        assert!(result.contains_column("a"));
        assert!(result.contains_column("b"));

        let a_values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(a_values, vec![1.0, 2.0]);
        let b_values = result
            .get_column_numeric_values("b")
            .expect("test should succeed");
        assert_eq!(b_values, vec![10.0, 20.0]);
    }

    #[test]
    fn test_concat_columns_ignore_index_renumbers_labels() {
        let df1 = numeric_df("a", vec![1.0, 2.0]);
        let df2 = numeric_df("b", vec![10.0, 20.0]);

        let result = concat(&[&df1, &df2], ConcatAxis::Columns, true).expect("test should succeed");

        // On axis = Columns the concatenation axis is the column labels, so
        // ignore_index replaces them with their positions.
        assert_eq!(result.column_names(), &["0", "1"]);
        assert_eq!(
            result
                .get_column_numeric_values("0")
                .expect("test should succeed"),
            vec![1.0, 2.0]
        );
        assert_eq!(
            result
                .get_column_numeric_values("1")
                .expect("test should succeed"),
            vec![10.0, 20.0]
        );
    }

    #[test]
    fn test_concat_columns_duplicate_names() {
        let df1 = numeric_df("value", vec![1.0, 2.0]);
        let df2 = numeric_df("value", vec![10.0, 20.0]);

        let result =
            concat(&[&df1, &df2], ConcatAxis::Columns, false).expect("test should succeed");

        assert_eq!(result.row_count(), 2);
        // Should have renamed duplicate column
        assert!(result.contains_column("value"));
        assert!(result.contains_column("value_1"));

        let v1 = result
            .get_column_numeric_values("value")
            .expect("test should succeed");
        assert_eq!(v1, vec![1.0, 2.0]);
        let v2 = result
            .get_column_numeric_values("value_1")
            .expect("test should succeed");
        assert_eq!(v2, vec![10.0, 20.0]);
    }

    #[test]
    fn test_concat_columns_mismatched_rows() {
        let df1 = numeric_df("a", vec![1.0, 2.0]);
        let df2 = numeric_df("b", vec![10.0, 20.0, 30.0]);

        let result = concat(&[&df1, &df2], ConcatAxis::Columns, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_empty_input() {
        let result = concat(&[], ConcatAxis::Rows, true).expect("test should succeed");
        assert_eq!(result.row_count(), 0);
    }

    #[test]
    fn test_concat_single_df() {
        let df = numeric_df("a", vec![1.0, 2.0]);

        let result = concat(&[&df], ConcatAxis::Rows, true).expect("test should succeed");
        assert_eq!(result.row_count(), 2);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_concat_multiple_dfs() {
        let df1 = numeric_df("a", vec![1.0]);
        let df2 = numeric_df("a", vec![2.0]);
        let df3 = numeric_df("a", vec![3.0]);

        let result =
            concat(&[&df1, &df2, &df3], ConcatAxis::Rows, true).expect("test should succeed");
        assert_eq!(result.row_count(), 3);
        let values = result
            .get_column_numeric_values("a")
            .expect("test should succeed");
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_concat_rows_preserves_integer_dtype() {
        let mut df1 = DataFrame::new();
        df1.add_column(
            "id".to_string(),
            Series::new(vec![1i64, 2], Some("id".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");
        let mut df2 = DataFrame::new();
        df2.add_column(
            "id".to_string(),
            Series::new(vec![3i64, 4], Some("id".to_string())).expect("test should succeed"),
        )
        .expect("test should succeed");

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, true).expect("test should succeed");
        assert_eq!(
            result
                .get_column::<i64>("id")
                .expect("test should succeed")
                .values(),
            &[1i64, 2, 3, 4]
        );
    }

    #[test]
    fn test_concat_rows_index_is_preserved_when_not_ignored() {
        let mut df1 = numeric_df("a", vec![1.0, 2.0]);
        df1.set_index(Index::new(vec!["r1".to_string(), "r2".to_string()]).unwrap())
            .expect("test should succeed");
        let mut df2 = numeric_df("a", vec![3.0]);
        df2.set_index(Index::new(vec!["r3".to_string()]).unwrap())
            .expect("test should succeed");

        let result = concat(&[&df1, &df2], ConcatAxis::Rows, false).expect("test should succeed");
        assert_eq!(
            result.get_index().string_values(),
            Some(vec!["r1".to_string(), "r2".to_string(), "r3".to_string()])
        );

        // ignore_index renumbers instead: no explicit index survives.
        let renumbered =
            concat(&[&df1, &df2], ConcatAxis::Rows, true).expect("test should succeed");
        assert_eq!(renumbered.get_index().string_values(), Some(Vec::new()));
    }

    #[test]
    fn test_concat_rows_duplicate_labels_are_reported() {
        let mut df1 = numeric_df("a", vec![1.0, 2.0]);
        df1.set_index(Index::new(vec!["x".to_string(), "y".to_string()]).unwrap())
            .expect("test should succeed");
        let mut df2 = numeric_df("a", vec![3.0]);
        df2.set_index(Index::new(vec!["x".to_string()]).unwrap())
            .expect("test should succeed");

        // Label "x" appears in both frames and this index cannot hold
        // duplicates, so the collision is reported instead of hidden.
        assert!(concat(&[&df1, &df2], ConcatAxis::Rows, false).is_err());
    }

    #[test]
    fn test_concat_rows_without_explicit_indexes_keeps_positional_index() {
        let df1 = numeric_df("a", vec![1.0, 2.0]);
        let df2 = numeric_df("a", vec![3.0, 4.0]);

        // Neither frame carries labels, so `ignore_index = false` has nothing
        // to preserve and must not fail.
        let result = concat(&[&df1, &df2], ConcatAxis::Rows, false).expect("test should succeed");
        assert_eq!(result.row_count(), 4);
        assert_eq!(result.get_index().string_values(), Some(Vec::new()));
    }
}
