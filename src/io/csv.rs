use csv::{ReaderBuilder, StringRecord};
use std::path::Path;

use crate::error::{PandRSError, Result};
use crate::series::Series;
use crate::DataFrame;

/// UTF-8 byte-order mark, as sometimes emitted by Excel and other Windows
/// tools at the start of a CSV file.
const UTF8_BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// Parse a CSV file into its header row and raw string columns.
///
/// This is the shared core used by both [`read_csv`] (always-String
/// columns) and [`read_csv_typed`] (per-column type inference). It takes
/// care of the tricky parts so callers don't have to:
///
/// - Strips a leading UTF-8 byte-order mark, if present, so the first
///   column name doesn't pick up a stray `\u{feff}`.
/// - Trims whitespace from header names only (`Trim::Headers`); field
///   values, quoted or not, are preserved byte-for-byte. `Trim::All` (the
///   previous setting) trims *after* quotes are stripped, so `" x "` in a
///   quoted field would silently lose its intentional padding.
/// - Reads every record in a single pass. Peeking at the first record (to
///   count columns for a headerless file) and then iterating `.records()`
///   again loses that first record: the `csv` crate only re-yields a
///   peeked row when it hasn't been read yet, and a first `.records()`
///   call already marks it as read. Collecting once, up front, avoids the
///   trap entirely.
/// - Rejects ragged rows (a data row whose field count does not match the
///   header) with a descriptive error instead of silently truncating extra
///   fields or zero-padding short ones.
fn read_csv_raw<P: AsRef<Path>>(
    path: P,
    has_header: bool,
) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut bytes = std::fs::read(path.as_ref()).map_err(PandRSError::Io)?;
    if bytes.starts_with(&UTF8_BOM) {
        bytes.drain(0..UTF8_BOM.len());
    }

    let mut rdr = ReaderBuilder::new()
        .has_headers(has_header)
        .flexible(true)
        .trim(csv::Trim::Headers)
        .from_reader(bytes.as_slice());

    let header_row: Option<Vec<String>> = if has_header {
        Some(
            rdr.headers()
                .map_err(PandRSError::Csv)?
                .iter()
                .map(|h| h.to_string())
                .collect(),
        )
    } else {
        None
    };

    let all_records: Vec<StringRecord> = rdr
        .records()
        .collect::<std::result::Result<Vec<_>, csv::Error>>()
        .map_err(PandRSError::Csv)?;

    let headers: Vec<String> = match header_row {
        Some(headers) => headers,
        None => match all_records.first() {
            Some(first) => (0..first.len()).map(|i| format!("column_{}", i)).collect(),
            None => return Ok((Vec::new(), Vec::new())),
        },
    };

    let mut columns: Vec<Vec<String>> = (0..headers.len())
        .map(|_| Vec::with_capacity(all_records.len()))
        .collect();

    for (row_idx, record) in all_records.iter().enumerate() {
        if record.len() != headers.len() {
            return Err(PandRSError::Format(format!(
                "CSV data row {} has {} field(s) but the header defines {} column(s); \
                 ragged rows are not supported",
                row_idx + 1,
                record.len(),
                headers.len()
            )));
        }
        for (col, field) in columns.iter_mut().zip(record.iter()) {
            col.push(field.to_string());
        }
    }

    Ok((headers, columns))
}

/// Read a DataFrame from a CSV file.
///
/// Every column comes back as `Series<String>`, matching
/// `DataFrame::from_csv`'s documented contract of not inferring dtypes
/// unless asked. Missing/short fields become an empty string. For
/// per-column numeric/boolean type inference, use [`read_csv_typed`]
/// instead.
///
/// A UTF-8 byte-order mark at the start of the file is stripped
/// automatically. A data row whose field count does not match the header
/// (or the inferred column count, for headerless files) is a hard error
/// rather than silently truncated or zero-padded data.
pub fn read_csv<P: AsRef<Path>>(path: P, has_header: bool) -> Result<DataFrame> {
    let (headers, columns) = read_csv_raw(path, has_header)?;

    let mut df = DataFrame::new();
    for (header, values) in headers.into_iter().zip(columns) {
        let series = Series::new(values, Some(header.clone()))?;
        df.add_column(header, series)?;
    }

    Ok(df)
}

/// Read a DataFrame from a CSV file, inferring a numeric or boolean type
/// for each column when it is safe to do so.
///
/// Per column, in order:
///
/// 1. **Int64**: every field is non-empty and parses as `i64`.
/// 2. **Float64**: every non-empty field parses as `f64`. Empty cells
///    become `f64::NAN` (matching pandas' own upcast-to-float behaviour
///    for an otherwise-integer column that has missing values -- `i64` has
///    no way to represent "missing", so a column that would otherwise be
///    Int64 but has even one empty cell is inferred as Float64 instead).
/// 3. **Bool**: every field is non-empty and, case-insensitively, one of
///    `true`/`false`/`1`/`0`/`yes`/`no`/`t`/`f`.
/// 4. **String** (fallback): the raw field text, unchanged -- this is also
///    what any column with missing cells falls back to once Int64/Bool are
///    ruled out (i.e. non-numeric columns with blanks stay String, keeping
///    the empty string as the missing-value marker instead of fabricating
///    a placeholder value).
///
/// Note the resulting text is not always stable across a
/// `read_csv_typed` -> `write_csv` -> `read_csv_typed` round trip: an
/// originally-empty float cell becomes `NaN` in memory, which `write_csv`
/// renders as the literal text `NaN` (not an empty string). The *value*
/// stays a faithfully-missing `NaN` throughout -- it never turns into a
/// fabricated `0.0` -- but the on-disk text changes on the first
/// round-trip.
pub fn read_csv_typed<P: AsRef<Path>>(path: P, has_header: bool) -> Result<DataFrame> {
    let (headers, columns) = read_csv_raw(path, has_header)?;

    let mut df = DataFrame::new();
    for (header, values) in headers.into_iter().zip(columns) {
        add_inferred_column(&mut df, header, values)?;
    }

    Ok(df)
}

/// Boolean tokens accepted by [`read_csv_typed`]'s type inference,
/// compared case-insensitively.
fn is_bool_token(s: &str) -> bool {
    matches!(
        s.to_lowercase().as_str(),
        "true" | "false" | "1" | "0" | "yes" | "no" | "t" | "f"
    )
}

fn parse_bool_token(s: &str) -> bool {
    matches!(s.to_lowercase().as_str(), "true" | "1" | "yes" | "t")
}

/// Infer a type for one CSV column and add it to `df`. See
/// [`read_csv_typed`] for the exact rules.
fn add_inferred_column(df: &mut DataFrame, header: String, values: Vec<String>) -> Result<()> {
    let has_missing = values.iter().any(|v| v.is_empty());
    let non_empty: Vec<&str> = values
        .iter()
        .filter(|v| !v.is_empty())
        .map(String::as_str)
        .collect();

    if non_empty.is_empty() {
        let series = Series::new(values, Some(header.clone()))?;
        return df.add_column(header, series);
    }

    if !has_missing && non_empty.iter().all(|s| s.parse::<i64>().is_ok()) {
        let mut parsed = Vec::with_capacity(values.len());
        for s in &values {
            let v = s.parse::<i64>().map_err(|e| {
                PandRSError::Format(format!(
                    "CSV column '{}': failed to parse '{}' as an integer: {}",
                    header, s, e
                ))
            })?;
            parsed.push(v);
        }
        let series = Series::new(parsed, Some(header.clone()))?;
        return df.add_column(header, series);
    }

    if non_empty.iter().all(|s| s.parse::<f64>().is_ok()) {
        let mut parsed = Vec::with_capacity(values.len());
        for s in &values {
            if s.is_empty() {
                parsed.push(f64::NAN);
                continue;
            }
            let v = s.parse::<f64>().map_err(|e| {
                PandRSError::Format(format!(
                    "CSV column '{}': failed to parse '{}' as a float: {}",
                    header, s, e
                ))
            })?;
            parsed.push(v);
        }
        let series = Series::new(parsed, Some(header.clone()))?;
        return df.add_column(header, series);
    }

    if !has_missing && non_empty.iter().all(|s| is_bool_token(s)) {
        let parsed: Vec<bool> = values.iter().map(|s| parse_bool_token(s)).collect();
        let series = Series::new(parsed, Some(header.clone()))?;
        return df.add_column(header, series);
    }

    let series = Series::new(values, Some(header.clone()))?;
    df.add_column(header, series)
}

/// Write a DataFrame to a CSV file.
///
/// Delegates to the real (inherent) CSV writer on `DataFrame`
/// ([`DataFrame::to_csv`]): every column's actual values are written,
/// using each element's string representation, with a header row of
/// column names.
pub fn write_csv<P: AsRef<Path>>(df: &DataFrame, path: P) -> Result<()> {
    DataFrame::to_csv(df, path)
}
