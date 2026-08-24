use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::series::Series;

/// Enum for join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join (only rows that match in both tables)
    Inner,
    /// Left join (all rows from the left table and matching rows from the right table)
    Left,
    /// Right join (all rows from the right table and matching rows from the left table)
    Right,
    /// Outer join (all rows from both tables)
    Outer,
}

/// Suffixes applied to non-key columns whose names collide between the two
/// operands: nothing is appended on the left, `_right` on the right.
///
/// pandas' default is `("_x", "_y")`; this crate keeps `("", "_right")` for
/// API compatibility with earlier releases. Use [`join_with_suffixes`] to
/// choose different suffixes.
pub const DEFAULT_SUFFIXES: (&str, &str) = ("", "_right");

/// Placeholder written into a column whose element type has **no in-band NA
/// representation** when a joined/concatenated row has no source value.
///
/// [`crate::series::Series`] stores a plain `Vec<T>` with no validity bitmap,
/// so `String`, date and time columns cannot express "missing". Numeric
/// columns can (`f64::NAN`), and integer/boolean columns are therefore widened
/// to `f64` when a NA fill is required (mirroring pandas' `int64 -> float64`
/// upcast). For the remaining types this crate writes the same text a missing
/// float renders as, so that one logical NA reads back identically regardless
/// of the source dtype.
///
/// Known limitation: a genuine `"NaN"` string in the input is indistinguishable
/// from a filled NA. Representing this properly requires a nullable string
/// column in the DataFrame column model.
pub(crate) const NA_STRING: &str = "NaN";

/// The concrete element type of a DataFrame column.
///
/// [`DataFrame`] stores each column as a `Series<T>` behind a `dyn Any`, so the
/// only reliable way to learn `T` is an exact downcast. Deciding a column's
/// type by "does `get_column_numeric_values` succeed?" is *not* equivalent:
/// that helper parses `Series<String>` values, so a text column of `"007"`
/// masquerades as numeric (and a sibling text column of `"x"` then does not) --
/// the root cause of the merge key-space and concat type-inference bugs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ElemType {
    /// `Series<String>`
    Str,
    /// `Series<bool>`
    Bool,
    /// `Series<i8>`
    I8,
    /// `Series<i16>`
    I16,
    /// `Series<i32>`
    I32,
    /// `Series<i64>`
    I64,
    /// `Series<i128>`
    I128,
    /// `Series<isize>`
    ISize,
    /// `Series<u8>`
    U8,
    /// `Series<u16>`
    U16,
    /// `Series<u32>`
    U32,
    /// `Series<u64>`
    U64,
    /// `Series<u128>`
    U128,
    /// `Series<usize>`
    USize,
    /// `Series<f32>`
    F32,
    /// `Series<f64>`
    F64,
    /// `Series<chrono::NaiveDate>`
    Date,
    /// `Series<chrono::NaiveDateTime>`
    DateTime,
    /// `Series<chrono::DateTime<chrono::Utc>>`
    DateTimeUtc,
}

impl ElemType {
    /// True for the integer element types (which have no NA representation and
    /// therefore widen to `f64` when a NA fill is needed).
    pub(crate) fn is_integer(self) -> bool {
        matches!(
            self,
            ElemType::I8
                | ElemType::I16
                | ElemType::I32
                | ElemType::I64
                | ElemType::I128
                | ElemType::ISize
                | ElemType::U8
                | ElemType::U16
                | ElemType::U32
                | ElemType::U64
                | ElemType::U128
                | ElemType::USize
        )
    }

    /// True for `f32`/`f64`, which carry NA in-band as `NaN`.
    pub(crate) fn is_float(self) -> bool {
        matches!(self, ElemType::F32 | ElemType::F64)
    }

    /// True for every type that has a meaningful `f64` value (booleans map to
    /// `1.0`/`0.0`, exactly as [`DataFrame::get_column_numeric_values`] does).
    pub(crate) fn is_numeric(self) -> bool {
        self.is_integer() || self.is_float() || self == ElemType::Bool
    }

    /// Human-readable dtype name, used in error messages.
    pub(crate) fn name(self) -> &'static str {
        match self {
            ElemType::Str => "String",
            ElemType::Bool => "bool",
            ElemType::I8 => "i8",
            ElemType::I16 => "i16",
            ElemType::I32 => "i32",
            ElemType::I64 => "i64",
            ElemType::I128 => "i128",
            ElemType::ISize => "isize",
            ElemType::U8 => "u8",
            ElemType::U16 => "u16",
            ElemType::U32 => "u32",
            ElemType::U64 => "u64",
            ElemType::U128 => "u128",
            ElemType::USize => "usize",
            ElemType::F32 => "f32",
            ElemType::F64 => "f64",
            ElemType::Date => "NaiveDate",
            ElemType::DateTime => "NaiveDateTime",
            ElemType::DateTimeUtc => "DateTime<Utc>",
        }
    }
}

/// Determine a column's concrete element type by exact downcast.
pub(crate) fn element_type(df: &DataFrame, column: &str) -> Result<ElemType> {
    if !df.contains_column(column) {
        return Err(Error::ColumnNotFound(column.to_string()));
    }

    macro_rules! probe {
        ($ty:ty, $variant:expr) => {
            if df.get_column::<$ty>(column).is_ok() {
                return Ok($variant);
            }
        };
    }

    probe!(String, ElemType::Str);
    probe!(i64, ElemType::I64);
    probe!(f64, ElemType::F64);
    probe!(i32, ElemType::I32);
    probe!(f32, ElemType::F32);
    probe!(bool, ElemType::Bool);
    probe!(u64, ElemType::U64);
    probe!(u32, ElemType::U32);
    probe!(usize, ElemType::USize);
    probe!(isize, ElemType::ISize);
    probe!(i8, ElemType::I8);
    probe!(i16, ElemType::I16);
    probe!(i128, ElemType::I128);
    probe!(u8, ElemType::U8);
    probe!(u16, ElemType::U16);
    probe!(u128, ElemType::U128);
    probe!(chrono::NaiveDate, ElemType::Date);
    probe!(chrono::NaiveDateTime, ElemType::DateTime);
    probe!(chrono::DateTime<chrono::Utc>, ElemType::DateTimeUtc);

    Err(Error::InvalidValue(format!(
        "Column '{}' has an element type that join/merge/concat cannot handle \
         (supported: String, integers, floats, bool, NaiveDate, NaiveDateTime, DateTime<Utc>)",
        column
    )))
}

/// Read a numeric column as `f64` without going through the string-parsing
/// fallback of [`DataFrame::get_column_numeric_values`] (so a text column is
/// never silently reinterpreted as numeric) and while supporting every integer
/// width.
pub(crate) fn numeric_values(df: &DataFrame, column: &str, elem: ElemType) -> Result<Vec<f64>> {
    fn cast<T>(df: &DataFrame, column: &str, convert: fn(&T) -> f64) -> Result<Vec<f64>>
    where
        T: 'static + Debug + Clone + Send + Sync,
    {
        Ok(df
            .get_column::<T>(column)?
            .values()
            .iter()
            .map(convert)
            .collect())
    }

    match elem {
        ElemType::F64 => cast::<f64>(df, column, |v| *v),
        ElemType::F32 => cast::<f32>(df, column, |v| *v as f64),
        ElemType::I8 => cast::<i8>(df, column, |v| *v as f64),
        ElemType::I16 => cast::<i16>(df, column, |v| *v as f64),
        ElemType::I32 => cast::<i32>(df, column, |v| *v as f64),
        ElemType::I64 => cast::<i64>(df, column, |v| *v as f64),
        ElemType::I128 => cast::<i128>(df, column, |v| *v as f64),
        ElemType::ISize => cast::<isize>(df, column, |v| *v as f64),
        ElemType::U8 => cast::<u8>(df, column, |v| *v as f64),
        ElemType::U16 => cast::<u16>(df, column, |v| *v as f64),
        ElemType::U32 => cast::<u32>(df, column, |v| *v as f64),
        ElemType::U64 => cast::<u64>(df, column, |v| *v as f64),
        ElemType::U128 => cast::<u128>(df, column, |v| *v as f64),
        ElemType::USize => cast::<usize>(df, column, |v| *v as f64),
        ElemType::Bool => cast::<bool>(df, column, |v| if *v { 1.0 } else { 0.0 }),
        other => Err(Error::InvalidValue(format!(
            "Column '{}' has element type {} and cannot be read as numeric values",
            column,
            other.name()
        ))),
    }
}

/// Gather `rows` (all of which must be `Some`) out of a column, preserving its
/// concrete element type exactly.
fn gather_exact<T>(
    out: &mut DataFrame,
    name: &str,
    source: &DataFrame,
    column: &str,
    rows: &[Option<usize>],
) -> Result<()>
where
    T: 'static + Debug + Clone + Send + Sync,
{
    let values = source.get_column::<T>(column)?.values();
    let mut gathered: Vec<T> = Vec::with_capacity(rows.len());
    for row in rows {
        let idx = row.ok_or_else(|| {
            Error::InvalidValue(format!(
                "Internal error: exact gather requested for column '{}' although some rows have \
                 no source value",
                column
            ))
        })?;
        let value = values.get(idx).ok_or_else(|| Error::IndexOutOfBounds {
            index: idx,
            size: values.len(),
        })?;
        gathered.push(value.clone());
    }
    out.add_column(
        name.to_string(),
        Series::new(gathered, Some(name.to_string()))?,
    )
}

/// Gather `rows` out of a column whose element type carries NA in-band,
/// substituting `na_value` for rows without a source value.
fn gather_with_na<T>(
    out: &mut DataFrame,
    name: &str,
    source: &DataFrame,
    column: &str,
    rows: &[Option<usize>],
    na_value: T,
) -> Result<()>
where
    T: 'static + Debug + Clone + Send + Sync,
{
    let values = source.get_column::<T>(column)?.values();
    let mut gathered: Vec<T> = Vec::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(idx) => {
                let value = values.get(*idx).ok_or_else(|| Error::IndexOutOfBounds {
                    index: *idx,
                    size: values.len(),
                })?;
                gathered.push(value.clone());
            }
            None => gathered.push(na_value.clone()),
        }
    }
    out.add_column(
        name.to_string(),
        Series::new(gathered, Some(name.to_string()))?,
    )
}

/// Gather an integer or boolean column, widening it to `f64` so that rows
/// without a source value can be represented as `NaN` (pandas performs the same
/// `int64 -> float64` upcast when a join introduces missing values).
fn gather_widened(
    out: &mut DataFrame,
    name: &str,
    source: &DataFrame,
    column: &str,
    rows: &[Option<usize>],
    elem: ElemType,
) -> Result<()> {
    let values = numeric_values(source, column, elem)?;
    let mut gathered: Vec<f64> = Vec::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(idx) => {
                let value = values.get(*idx).ok_or_else(|| Error::IndexOutOfBounds {
                    index: *idx,
                    size: values.len(),
                })?;
                gathered.push(*value);
            }
            None => gathered.push(f64::NAN),
        }
    }
    out.add_column(
        name.to_string(),
        Series::new(gathered, Some(name.to_string()))?,
    )
}

/// Gather a column with no in-band NA representation by rendering it to text
/// and writing [`NA_STRING`] for rows without a source value.
fn gather_rendered(
    out: &mut DataFrame,
    name: &str,
    source: &DataFrame,
    column: &str,
    rows: &[Option<usize>],
) -> Result<()> {
    let values = source.get_column_string_values(column)?;
    let mut gathered: Vec<String> = Vec::with_capacity(rows.len());
    for row in rows {
        match row {
            Some(idx) => {
                let value = values.get(*idx).ok_or_else(|| Error::IndexOutOfBounds {
                    index: *idx,
                    size: values.len(),
                })?;
                gathered.push(value.clone());
            }
            None => gathered.push(NA_STRING.to_string()),
        }
    }
    out.add_column(
        name.to_string(),
        Series::new(gathered, Some(name.to_string()))?,
    )
}

/// Append `source[column]`, restricted to `rows`, to `out` under `name`.
///
/// * When every row has a source value the column's concrete element type is
///   preserved exactly (an `i64` column stays `Series<i64>`, a `String` column
///   stays `Series<String>` -- no re-inference, so `"007"` stays `"007"`).
/// * When some rows are missing, the output dtype follows pandas' NA-upcast
///   rules: `f32`/`f64` keep their type and use `NaN`; integers and booleans
///   widen to `f64` with `NaN`; text and date/time columns render to text and
///   use [`NA_STRING`].
pub(crate) fn gather_column(
    out: &mut DataFrame,
    name: &str,
    source: &DataFrame,
    column: &str,
    rows: &[Option<usize>],
) -> Result<()> {
    let elem = element_type(source, column)?;

    if !rows.iter().any(Option::is_none) {
        return match elem {
            ElemType::Str => gather_exact::<String>(out, name, source, column, rows),
            ElemType::Bool => gather_exact::<bool>(out, name, source, column, rows),
            ElemType::I8 => gather_exact::<i8>(out, name, source, column, rows),
            ElemType::I16 => gather_exact::<i16>(out, name, source, column, rows),
            ElemType::I32 => gather_exact::<i32>(out, name, source, column, rows),
            ElemType::I64 => gather_exact::<i64>(out, name, source, column, rows),
            ElemType::I128 => gather_exact::<i128>(out, name, source, column, rows),
            ElemType::ISize => gather_exact::<isize>(out, name, source, column, rows),
            ElemType::U8 => gather_exact::<u8>(out, name, source, column, rows),
            ElemType::U16 => gather_exact::<u16>(out, name, source, column, rows),
            ElemType::U32 => gather_exact::<u32>(out, name, source, column, rows),
            ElemType::U64 => gather_exact::<u64>(out, name, source, column, rows),
            ElemType::U128 => gather_exact::<u128>(out, name, source, column, rows),
            ElemType::USize => gather_exact::<usize>(out, name, source, column, rows),
            ElemType::F32 => gather_exact::<f32>(out, name, source, column, rows),
            ElemType::F64 => gather_exact::<f64>(out, name, source, column, rows),
            ElemType::Date => gather_exact::<chrono::NaiveDate>(out, name, source, column, rows),
            ElemType::DateTime => {
                gather_exact::<chrono::NaiveDateTime>(out, name, source, column, rows)
            }
            ElemType::DateTimeUtc => {
                gather_exact::<chrono::DateTime<chrono::Utc>>(out, name, source, column, rows)
            }
        };
    }

    match elem {
        ElemType::F64 => gather_with_na(out, name, source, column, rows, f64::NAN),
        ElemType::F32 => gather_with_na(out, name, source, column, rows, f32::NAN),
        elem if elem.is_integer() || elem == ElemType::Bool => {
            gather_widened(out, name, source, column, rows, elem)
        }
        // String / date / time: no in-band NA, so render and use NA_STRING.
        _ => gather_rendered(out, name, source, column, rows),
    }
}

/// A join key normalised so that both operands are compared in one key space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinKey {
    /// Canonical IEEE-754 bits of a numeric key. `-0.0` is normalised to `0.0`
    /// so the two compare equal, and `NaN` is never stored here -- it is NA.
    Num(u64),
    /// A textual key (compared byte-for-byte, like pandas' object dtype).
    Text(String),
}

/// Decide, from **both** operands, whether the join key is compared numerically
/// or textually.
///
/// Choosing the representation per side (as the old merge implementation did)
/// puts the two frames in different key spaces: a `f64` key encoded as its bit
/// pattern can never equal the same value rendered as text, so every row
/// silently failed to match, and the mixed case additionally panicked.
fn key_space_is_numeric(left: &DataFrame, right: &DataFrame, on: &str) -> Result<bool> {
    let left_elem = element_type(left, on)?;
    let right_elem = element_type(right, on)?;

    match (left_elem.is_numeric(), right_elem.is_numeric()) {
        (true, true) => Ok(true),
        (false, false) => Ok(false),
        _ => Err(Error::InvalidValue(format!(
            "Cannot join on column '{}': the left operand has dtype {} and the right operand has \
             dtype {}. Numeric and textual key columns are not comparable (pandas raises here \
             too); convert one side so both key columns have the same kind of dtype.",
            on,
            left_elem.name(),
            right_elem.name()
        ))),
    }
}

/// Normalise one side's join keys. `None` marks a missing (NA) key, which never
/// matches anything -- including another NA -- exactly as in pandas.
fn normalized_keys(df: &DataFrame, on: &str, numeric: bool) -> Result<Vec<Option<JoinKey>>> {
    if numeric {
        let elem = element_type(df, on)?;
        Ok(numeric_values(df, on, elem)?
            .into_iter()
            .map(|value| {
                if value.is_nan() {
                    None
                } else if value == 0.0 {
                    // Collapse -0.0 and 0.0 onto one key.
                    Some(JoinKey::Num(0.0f64.to_bits()))
                } else {
                    Some(JoinKey::Num(value.to_bits()))
                }
            })
            .collect())
    } else {
        Ok(df
            .get_column_string_values(on)?
            .into_iter()
            .map(|value| Some(JoinKey::Text(value)))
            .collect())
    }
}

/// Emit the single key column, taking each value from the left operand when the
/// pair has a left row and from the right operand otherwise.
fn gather_key_column(
    out: &mut DataFrame,
    left: &DataFrame,
    right: &DataFrame,
    on: &str,
    pairs: &[(Option<usize>, Option<usize>)],
) -> Result<()> {
    fn coalesce_exact<T>(
        out: &mut DataFrame,
        left: &DataFrame,
        right: &DataFrame,
        on: &str,
        pairs: &[(Option<usize>, Option<usize>)],
    ) -> Result<()>
    where
        T: 'static + Debug + Clone + Send + Sync,
    {
        let left_values = left.get_column::<T>(on)?.values();
        let right_values = right.get_column::<T>(on)?.values();
        let mut gathered: Vec<T> = Vec::with_capacity(pairs.len());
        for (l, r) in pairs {
            let value = match (l, r) {
                (Some(i), _) => left_values.get(*i),
                (None, Some(j)) => right_values.get(*j),
                (None, None) => None,
            }
            .ok_or_else(|| {
                Error::InvalidValue(format!(
                    "Internal error: join produced a row with no source value for key column '{}'",
                    on
                ))
            })?;
            gathered.push(value.clone());
        }
        out.add_column(on.to_string(), Series::new(gathered, Some(on.to_string()))?)
    }

    let left_elem = element_type(left, on)?;
    let right_elem = element_type(right, on)?;

    if left_elem == right_elem {
        return match left_elem {
            ElemType::Str => coalesce_exact::<String>(out, left, right, on, pairs),
            ElemType::Bool => coalesce_exact::<bool>(out, left, right, on, pairs),
            ElemType::I8 => coalesce_exact::<i8>(out, left, right, on, pairs),
            ElemType::I16 => coalesce_exact::<i16>(out, left, right, on, pairs),
            ElemType::I32 => coalesce_exact::<i32>(out, left, right, on, pairs),
            ElemType::I64 => coalesce_exact::<i64>(out, left, right, on, pairs),
            ElemType::I128 => coalesce_exact::<i128>(out, left, right, on, pairs),
            ElemType::ISize => coalesce_exact::<isize>(out, left, right, on, pairs),
            ElemType::U8 => coalesce_exact::<u8>(out, left, right, on, pairs),
            ElemType::U16 => coalesce_exact::<u16>(out, left, right, on, pairs),
            ElemType::U32 => coalesce_exact::<u32>(out, left, right, on, pairs),
            ElemType::U64 => coalesce_exact::<u64>(out, left, right, on, pairs),
            ElemType::U128 => coalesce_exact::<u128>(out, left, right, on, pairs),
            ElemType::USize => coalesce_exact::<usize>(out, left, right, on, pairs),
            ElemType::F32 => coalesce_exact::<f32>(out, left, right, on, pairs),
            ElemType::F64 => coalesce_exact::<f64>(out, left, right, on, pairs),
            ElemType::Date => coalesce_exact::<chrono::NaiveDate>(out, left, right, on, pairs),
            ElemType::DateTime => {
                coalesce_exact::<chrono::NaiveDateTime>(out, left, right, on, pairs)
            }
            ElemType::DateTimeUtc => {
                coalesce_exact::<chrono::DateTime<chrono::Utc>>(out, left, right, on, pairs)
            }
        };
    }

    // The two key columns have different (but compatible) dtypes: widen both
    // sides to one common representation instead of taking either side's.
    if left_elem.is_numeric() && right_elem.is_numeric() {
        let left_values = numeric_values(left, on, left_elem)?;
        let right_values = numeric_values(right, on, right_elem)?;
        let mut gathered: Vec<f64> = Vec::with_capacity(pairs.len());
        for (l, r) in pairs {
            let value = match (l, r) {
                (Some(i), _) => left_values.get(*i).copied(),
                (None, Some(j)) => right_values.get(*j).copied(),
                (None, None) => None,
            };
            gathered.push(value.unwrap_or(f64::NAN));
        }
        return out.add_column(on.to_string(), Series::new(gathered, Some(on.to_string()))?);
    }

    let left_values = left.get_column_string_values(on)?;
    let right_values = right.get_column_string_values(on)?;
    let mut gathered: Vec<String> = Vec::with_capacity(pairs.len());
    for (l, r) in pairs {
        let value = match (l, r) {
            (Some(i), _) => left_values.get(*i).cloned(),
            (None, Some(j)) => right_values.get(*j).cloned(),
            (None, None) => None,
        };
        gathered.push(value.unwrap_or_else(|| NA_STRING.to_string()));
    }
    out.add_column(on.to_string(), Series::new(gathered, Some(on.to_string()))?)
}

/// Pair up left and right row positions according to `join_type`.
fn build_pairs(
    left_keys: &[Option<JoinKey>],
    right_keys: &[Option<JoinKey>],
    join_type: JoinType,
) -> Vec<(Option<usize>, Option<usize>)> {
    let mut pairs: Vec<(Option<usize>, Option<usize>)> = Vec::new();

    if join_type == JoinType::Right {
        // A right join is driven by the *right* frame: its row order is the
        // output row order (pandas' `how="right"`). The column layout is
        // unchanged -- left columns first, then right -- which is why this is a
        // real join arm and not `other.left_join(self)`: swapping the operands
        // also swaps the output columns and their suffixes, so `v` and
        // `v_right` came out exactly inverted.
        let mut index: HashMap<&JoinKey, Vec<usize>> = HashMap::new();
        for (i, key) in left_keys.iter().enumerate() {
            if let Some(key) = key {
                index.entry(key).or_default().push(i);
            }
        }
        for (r, key) in right_keys.iter().enumerate() {
            match key.as_ref().and_then(|key| index.get(key)) {
                Some(left_rows) if !left_rows.is_empty() => {
                    for &l in left_rows {
                        pairs.push((Some(l), Some(r)));
                    }
                }
                _ => pairs.push((None, Some(r))),
            }
        }
        return pairs;
    }

    let mut index: HashMap<&JoinKey, Vec<usize>> = HashMap::new();
    for (i, key) in right_keys.iter().enumerate() {
        if let Some(key) = key {
            index.entry(key).or_default().push(i);
        }
    }

    let mut right_matched = vec![false; right_keys.len()];
    for (l, key) in left_keys.iter().enumerate() {
        match key.as_ref().and_then(|key| index.get(key)) {
            Some(right_rows) if !right_rows.is_empty() => {
                for &r in right_rows {
                    right_matched[r] = true;
                    pairs.push((Some(l), Some(r)));
                }
            }
            // No match -- either the key is genuinely absent from the right
            // frame or it is NA (which never matches, not even another NA).
            // The row itself is still emitted for left/outer joins.
            _ => {
                if matches!(join_type, JoinType::Left | JoinType::Outer) {
                    pairs.push((Some(l), None));
                }
            }
        }
    }

    if join_type == JoinType::Outer {
        for (r, matched) in right_matched.iter().enumerate() {
            if !matched {
                pairs.push((None, Some(r)));
            }
        }
    }

    pairs
}

/// Build the joined DataFrame for the given key column and join type using a
/// hash join.
///
/// Output layout follows pandas: the left operand's columns in their original
/// order (the key column keeps its position and appears once), followed by the
/// right operand's non-key columns. Non-key column names present on both sides
/// get `suffixes.0` / `suffixes.1` appended.
///
/// Each column keeps its source dtype whenever every output row has a source
/// value; see [`gather_column`] for the NA-upcast rules that apply otherwise.
fn hash_join(
    left: &DataFrame,
    right: &DataFrame,
    on: &str,
    join_type: JoinType,
    suffixes: (&str, &str),
) -> Result<DataFrame> {
    if !left.contains_column(on) {
        return Err(Error::ColumnNotFound(format!(
            "Join column '{}' does not exist in the left DataFrame",
            on
        )));
    }
    if !right.contains_column(on) {
        return Err(Error::ColumnNotFound(format!(
            "Join column '{}' does not exist in the right DataFrame",
            on
        )));
    }

    let numeric_key_space = key_space_is_numeric(left, right, on)?;
    let left_keys = normalized_keys(left, on, numeric_key_space)?;
    let right_keys = normalized_keys(right, on, numeric_key_space)?;

    let pairs = build_pairs(&left_keys, &right_keys, join_type);
    let left_rows: Vec<Option<usize>> = pairs.iter().map(|pair| pair.0).collect();
    let right_rows: Vec<Option<usize>> = pairs.iter().map(|pair| pair.1).collect();

    let left_cols = left.column_names();
    let right_cols = right.column_names();
    let right_name_set: HashSet<&String> = right_cols.iter().collect();
    let overlapping: HashSet<&String> = left_cols
        .iter()
        .filter(|col| col.as_str() != on && right_name_set.contains(*col))
        .collect();

    let mut result = DataFrame::new();

    for col in left_cols {
        if col == on {
            gather_key_column(&mut result, left, right, on, &pairs)?;
            continue;
        }
        let name = if overlapping.contains(col) {
            format!("{}{}", col, suffixes.0)
        } else {
            col.clone()
        };
        gather_column(&mut result, &name, left, col, &left_rows)?;
    }

    for col in right_cols {
        if col == on {
            continue;
        }
        let name = if overlapping.contains(col) {
            format!("{}{}", col, suffixes.1)
        } else {
            col.clone()
        };
        gather_column(&mut result, &name, right, col, &right_rows)?;
    }

    Ok(result)
}

/// Join two DataFrames on `on`, choosing the suffixes applied to overlapping
/// non-key column names.
///
/// [`JoinExt`]'s methods use [`DEFAULT_SUFFIXES`]; this function exists so
/// callers that need pandas' `("_x", "_y")` (or any other pair) do not have to
/// rename columns afterwards.
pub fn join_with_suffixes(
    left: &DataFrame,
    right: &DataFrame,
    on: &str,
    join_type: JoinType,
    suffixes: (&str, &str),
) -> Result<DataFrame> {
    hash_join(left, right, on, join_type, suffixes)
}

/// Join functionality for DataFrames
pub trait JoinExt {
    /// Join two DataFrames
    fn join(&self, other: &Self, on: &str, join_type: JoinType) -> Result<Self>
    where
        Self: Sized;

    /// Perform inner join
    fn inner_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;

    /// Perform left join
    fn left_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;

    /// Perform right join
    fn right_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;

    /// Perform outer join
    fn outer_join(&self, other: &Self, on: &str) -> Result<Self>
    where
        Self: Sized;
}

/// Implementation of JoinExt for DataFrame
impl JoinExt for DataFrame {
    fn join(&self, other: &Self, on: &str, join_type: JoinType) -> Result<Self> {
        hash_join(self, other, on, join_type, DEFAULT_SUFFIXES)
    }

    fn inner_join(&self, other: &Self, on: &str) -> Result<Self> {
        hash_join(self, other, on, JoinType::Inner, DEFAULT_SUFFIXES)
    }

    fn left_join(&self, other: &Self, on: &str) -> Result<Self> {
        hash_join(self, other, on, JoinType::Left, DEFAULT_SUFFIXES)
    }

    fn right_join(&self, other: &Self, on: &str) -> Result<Self> {
        hash_join(self, other, on, JoinType::Right, DEFAULT_SUFFIXES)
    }

    fn outer_join(&self, other: &Self, on: &str) -> Result<Self> {
        hash_join(self, other, on, JoinType::Outer, DEFAULT_SUFFIXES)
    }
}

/// Re-export JoinType for backward compatibility
#[deprecated(since = "0.1.0", note = "Use crate::dataframe::join::JoinType")]
pub use crate::dataframe::join::JoinType as LegacyJoinType;

#[cfg(test)]
mod tests {
    use super::*;

    fn left_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1i64, 2, 3], None).unwrap(),
        )
        .unwrap();
        df.add_column(
            "lval".to_string(),
            Series::new(
                vec!["a".to_string(), "b".to_string(), "c".to_string()],
                None,
            )
            .unwrap(),
        )
        .unwrap();
        df
    }

    fn right_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![2i64, 3, 4], None).unwrap(),
        )
        .unwrap();
        df.add_column(
            "rval".to_string(),
            Series::new(
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
                None,
            )
            .unwrap(),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_inner_join() {
        let result = left_df().inner_join(&right_df(), "id").unwrap();
        assert_eq!(result.row_count(), 2);
        assert_eq!(
            result.get_column_string_values("id").unwrap(),
            vec!["2", "3"]
        );
        assert_eq!(
            result.get_column_string_values("lval").unwrap(),
            vec!["b", "c"]
        );
        assert_eq!(
            result.get_column_string_values("rval").unwrap(),
            vec!["x", "y"]
        );
        // The key column keeps its i64 dtype (it used to become String).
        assert_eq!(result.get_column::<i64>("id").unwrap().values(), &[2i64, 3]);
    }

    #[test]
    fn test_left_join() {
        let result = left_df().left_join(&right_df(), "id").unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(
            result.get_column_string_values("id").unwrap(),
            vec!["1", "2", "3"]
        );
        // The unmatched left row has no right value: it reads back as NA, not
        // as an empty string that is indistinguishable from real data.
        assert_eq!(
            result.get_column_string_values("rval").unwrap(),
            vec![NA_STRING, "x", "y"]
        );
    }

    #[test]
    fn test_outer_join() {
        let result = left_df().outer_join(&right_df(), "id").unwrap();
        assert_eq!(result.row_count(), 4);
        assert_eq!(
            result.get_column_string_values("id").unwrap(),
            vec!["1", "2", "3", "4"]
        );
        assert_eq!(
            result.get_column_string_values("lval").unwrap(),
            vec!["a", "b", "c", NA_STRING]
        );
        assert_eq!(
            result.get_column_string_values("rval").unwrap(),
            vec![NA_STRING, "x", "y", "z"]
        );
    }

    #[test]
    fn test_right_join() {
        let result = left_df().right_join(&right_df(), "id").unwrap();
        assert_eq!(result.row_count(), 3);
        assert_eq!(
            result.get_column_string_values("id").unwrap(),
            vec!["2", "3", "4"]
        );
        // Column layout is the left frame's columns first, then the right
        // frame's -- identical to a left join, only the row set differs.
        assert_eq!(result.column_names(), &["id", "lval", "rval"]);
        assert_eq!(
            result.get_column_string_values("lval").unwrap(),
            vec!["b", "c", NA_STRING]
        );
        assert_eq!(
            result.get_column_string_values("rval").unwrap(),
            vec!["x", "y", "z"]
        );
    }

    #[test]
    fn test_join_missing_column_errors() {
        assert!(left_df().inner_join(&right_df(), "missing").is_err());
    }

    #[test]
    fn test_join_overlapping_columns_suffixed() {
        // Both sides have a non-key column named "v"; the right one is suffixed.
        let mut left = DataFrame::new();
        left.add_column("id".to_string(), Series::new(vec![1i64, 2], None).unwrap())
            .unwrap();
        left.add_column(
            "v".to_string(),
            Series::new(vec!["l1".to_string(), "l2".to_string()], None).unwrap(),
        )
        .unwrap();
        let mut right = DataFrame::new();
        right
            .add_column("id".to_string(), Series::new(vec![1i64, 2], None).unwrap())
            .unwrap();
        right
            .add_column(
                "v".to_string(),
                Series::new(vec!["r1".to_string(), "r2".to_string()], None).unwrap(),
            )
            .unwrap();

        let result = left.inner_join(&right, "id").unwrap();
        assert!(result.contains_column("v"));
        assert!(result.contains_column("v_right"));
        assert_eq!(
            result.get_column_string_values("v").unwrap(),
            vec!["l1", "l2"]
        );
        assert_eq!(
            result.get_column_string_values("v_right").unwrap(),
            vec!["r1", "r2"]
        );
    }

    #[test]
    fn test_join_with_custom_suffixes() {
        let mut left = DataFrame::new();
        left.add_column("id".to_string(), Series::new(vec![1i64], None).unwrap())
            .unwrap();
        left.add_column(
            "v".to_string(),
            Series::new(vec!["l".to_string()], None).unwrap(),
        )
        .unwrap();
        let mut right = DataFrame::new();
        right
            .add_column("id".to_string(), Series::new(vec![1i64], None).unwrap())
            .unwrap();
        right
            .add_column(
                "v".to_string(),
                Series::new(vec!["r".to_string()], None).unwrap(),
            )
            .unwrap();

        let result =
            join_with_suffixes(&left, &right, "id", JoinType::Inner, ("_x", "_y")).unwrap();
        assert_eq!(result.column_names(), &["id", "v_x", "v_y"]);
        assert_eq!(result.get_column_string_values("v_x").unwrap(), vec!["l"]);
        assert_eq!(result.get_column_string_values("v_y").unwrap(), vec!["r"]);
    }

    #[test]
    fn test_join_mixed_key_dtypes_error_not_panic() {
        let mut left = DataFrame::new();
        left.add_column("id".to_string(), Series::new(vec![1i64, 2], None).unwrap())
            .unwrap();
        let mut right = DataFrame::new();
        right
            .add_column(
                "id".to_string(),
                Series::new(vec!["a".to_string(), "b".to_string()], None).unwrap(),
            )
            .unwrap();

        assert!(left.inner_join(&right, "id").is_err());
    }
}
