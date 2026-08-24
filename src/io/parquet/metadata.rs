//! File/row-group/column metadata, real column statistics, and schema analysis.

use crate::error::{Error, Result};
use parquet::data_type::ByteArray;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::statistics::{Statistics, ValueStatistics};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::core::{ColumnStats, ParquetMetadata, RowGroupInfo};

/// Parquet schema analysis structure
#[derive(Debug, Clone)]
pub struct ParquetSchemaAnalysis {
    /// Number of columns
    pub column_count: usize,
    /// Column names and types
    pub columns: HashMap<String, String>,
    /// Schema complexity score
    pub complexity_score: f64,
    /// Nested structure depth
    pub max_nesting_depth: usize,
    /// Estimated schema evolution difficulty
    pub evolution_difficulty: String,
}
/// A single column-chunk statistics bound (min or max), carrying both its
/// natively-typed, comparable value and its display string.
pub(super) struct StatBound {
    pub(super) value: StatValue,
    pub(super) text: String,
}
/// A comparable representation of a Parquet statistics bound, widened to a
/// small common set of native types so bounds from different row groups of
/// the *same* column (always the same physical type) can be ordered
/// correctly instead of lexicographically.
pub(super) enum StatValue {
    Bool(bool),
    I64(i64),
    F64(f64),
    Bytes(Vec<u8>),
    /// INT96 (deprecated legacy timestamp): no native ordering is
    /// implemented here, so bounds of this kind never replace one another —
    /// the first row group's value is kept, which is honest (it is a real,
    /// present-in-the-file value) even though it is not guaranteed to be the
    /// true file-wide extreme.
    Unordered,
}
impl StatValue {
    pub(super) fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Self::Bool(a), Self::Bool(b)) => a.partial_cmp(b),
            (Self::I64(a), Self::I64(b)) => a.partial_cmp(b),
            (Self::F64(a), Self::F64(b)) => a.partial_cmp(b),
            (Self::Bytes(a), Self::Bytes(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}
/// Get comprehensive metadata about a Parquet file
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
///
/// # Returns
///
/// * `Result<ParquetMetadata>` - File metadata information, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::get_parquet_metadata;
///
/// let metadata = get_parquet_metadata("data.parquet").expect("operation should succeed");
/// println!("File has {} rows in {} row groups", metadata.num_rows, metadata.num_row_groups);
/// ```
pub fn get_parquet_metadata(path: impl AsRef<Path>) -> Result<ParquetMetadata> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;

    let reader = SerializedFileReader::new(file)
        .map_err(|e| Error::IoError(format!("Failed to create Parquet reader: {}", e)))?;

    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();

    Ok(ParquetMetadata {
        num_rows: file_metadata.num_rows(),
        num_row_groups: metadata.num_row_groups(),
        schema: format!("{:?}", file_metadata.schema()),
        file_size: None,                    // Would need additional file stats
        compression: "Various".to_string(), // Row groups can have different compression
        created_by: file_metadata.created_by().map(|s| s.to_string()),
    })
}

/// Get information about all row groups in a Parquet file
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
///
/// # Returns
///
/// * `Result<Vec<RowGroupInfo>>` - Vector of row group information, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::get_row_group_info;
///
/// let row_groups = get_row_group_info("data.parquet").expect("operation should succeed");
/// for (i, rg) in row_groups.iter().enumerate() {
///     println!("Row group {}: {} rows, {} bytes", i, rg.num_rows, rg.total_byte_size);
/// }
/// ```
pub fn get_row_group_info(path: impl AsRef<Path>) -> Result<Vec<RowGroupInfo>> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;

    let reader = SerializedFileReader::new(file)
        .map_err(|e| Error::IoError(format!("Failed to create Parquet reader: {}", e)))?;

    let metadata = reader.metadata();
    let mut row_groups = Vec::new();

    for i in 0..metadata.num_row_groups() {
        let rg_metadata = metadata.row_group(i);
        row_groups.push(RowGroupInfo {
            index: i,
            num_rows: rg_metadata.num_rows(),
            total_byte_size: rg_metadata.total_byte_size(),
            num_columns: rg_metadata.num_columns(),
        });
    }

    Ok(row_groups)
}

/// Get column statistics for all columns in a Parquet file
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
///
/// # Returns
///
/// * `Result<Vec<ColumnStats>>` - Vector of column statistics, or an error
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::get_column_statistics;
///
/// let stats = get_column_statistics("data.parquet").expect("operation should succeed");
/// for stat in stats {
///     println!("{}: {} nulls, min={:?}, max={:?}",
///              stat.name, stat.null_count.unwrap_or(0), stat.min_value, stat.max_value);
/// }
/// ```
pub fn get_column_statistics(path: impl AsRef<Path>) -> Result<Vec<ColumnStats>> {
    let file = File::open(path.as_ref())
        .map_err(|e| Error::IoError(format!("Failed to open Parquet file: {}", e)))?;

    let reader = SerializedFileReader::new(file)
        .map_err(|e| Error::IoError(format!("Failed to create Parquet reader: {}", e)))?;

    let metadata = reader.metadata();
    let schema = metadata.file_metadata().schema_descr();
    let mut column_stats = Vec::new();

    // Collect statistics from all row groups, merging min/max across row
    // groups by the value's own natural ordering (not a string comparison,
    // which would rank "9" above "10").
    let mut null_counts: HashMap<String, u64> = HashMap::new();
    let mut distinct_counts: HashMap<String, u64> = HashMap::new();
    let mut min_stats: HashMap<String, StatBound> = HashMap::new();
    let mut max_stats: HashMap<String, StatBound> = HashMap::new();

    for rg_idx in 0..metadata.num_row_groups() {
        let rg_metadata = metadata.row_group(rg_idx);

        for col_idx in 0..rg_metadata.num_columns() {
            let col_metadata = rg_metadata.column(col_idx);
            let col_name = schema.column(col_idx).name().to_string();

            let Some(statistics) = col_metadata.statistics() else {
                continue;
            };

            if let Some(null_count) = statistics.null_count_opt() {
                *null_counts.entry(col_name.clone()).or_insert(0) += null_count;
            }
            if let Some(distinct_count) = statistics.distinct_count_opt() {
                // Distinct counts can't be summed across row groups (values
                // may repeat between groups); surface the largest reported
                // per-row-group figure as a lower bound on the true count.
                let entry = distinct_counts.entry(col_name.clone()).or_insert(0);
                *entry = (*entry).max(distinct_count);
            }

            if let Some(candidate) = stat_bound(statistics, true) {
                update_bound(min_stats.entry(col_name.clone()), candidate, Ordering::Less);
            }
            if let Some(candidate) = stat_bound(statistics, false) {
                update_bound(
                    max_stats.entry(col_name.clone()),
                    candidate,
                    Ordering::Greater,
                );
            }
        }
    }

    // Create column stats
    for col_idx in 0..schema.num_columns() {
        let column = schema.column(col_idx);
        let col_name = column.name().to_string();

        column_stats.push(ColumnStats {
            name: col_name.clone(),
            data_type: format!("{:?}", column.physical_type()),
            null_count: null_counts.get(&col_name).map(|&n| n as i64),
            distinct_count: distinct_counts.get(&col_name).map(|&n| n as i64),
            min_value: min_stats.get(&col_name).map(|b| b.text.clone()),
            max_value: max_stats.get(&col_name).map(|b| b.text.clone()),
        });
    }

    Ok(column_stats)
}

/// Replace `slot` with `candidate` when `candidate` compares as `direction`
/// (`Less` for a min-tracking slot, `Greater` for a max-tracking slot)
/// relative to what's already there. An empty slot always takes the
/// candidate; an incomparable pair (mismatched/`Unordered` variants) keeps
/// whichever value was seen first.
fn update_bound(
    slot: std::collections::hash_map::Entry<'_, String, StatBound>,
    candidate: StatBound,
    direction: Ordering,
) {
    use std::collections::hash_map::Entry;
    match slot {
        Entry::Vacant(v) => {
            v.insert(candidate);
        }
        Entry::Occupied(mut o) => {
            if candidate.value.partial_cmp(&o.get().value) == Some(direction) {
                o.insert(candidate);
            }
        }
    }
}

/// Decode the min (or max) bound of a Parquet column-chunk `Statistics` into
/// a `StatBound`, dispatching on the physical type actually stored. Returns
/// `None` when the bound isn't present (or, for floats, is NaN — Parquet
/// writers should not use NaN as a min/max bound, and treating it as "no
/// information" avoids a NaN poisoning every later, valid comparison).
fn stat_bound(statistics: &Statistics, want_min: bool) -> Option<StatBound> {
    fn pick<T>(s: &ValueStatistics<T>, want_min: bool) -> Option<&T> {
        if want_min {
            s.min_opt()
        } else {
            s.max_opt()
        }
    }
    fn byte_array_text(bytes: &ByteArray) -> String {
        match bytes.as_utf8() {
            Ok(s) => s.to_string(),
            Err(_) => bytes.to_string(),
        }
    }

    match statistics {
        Statistics::Boolean(s) => pick(s, want_min).map(|&v| StatBound {
            value: StatValue::Bool(v),
            text: v.to_string(),
        }),
        Statistics::Int32(s) => pick(s, want_min).map(|&v| StatBound {
            value: StatValue::I64(v as i64),
            text: v.to_string(),
        }),
        Statistics::Int64(s) => pick(s, want_min).map(|&v| StatBound {
            value: StatValue::I64(v),
            text: v.to_string(),
        }),
        Statistics::Float(s) => pick(s, want_min).and_then(|&v| {
            (!v.is_nan()).then_some(StatBound {
                value: StatValue::F64(v as f64),
                text: v.to_string(),
            })
        }),
        Statistics::Double(s) => pick(s, want_min).and_then(|&v| {
            (!v.is_nan()).then_some(StatBound {
                value: StatValue::F64(v),
                text: v.to_string(),
            })
        }),
        Statistics::ByteArray(s) => pick(s, want_min).map(|v| StatBound {
            value: StatValue::Bytes(v.data().to_vec()),
            text: byte_array_text(v),
        }),
        Statistics::FixedLenByteArray(s) => pick(s, want_min).map(|v| StatBound {
            value: StatValue::Bytes(v.data().to_vec()),
            text: byte_array_text(v),
        }),
        Statistics::Int96(s) => pick(s, want_min).map(|v| StatBound {
            value: StatValue::Unordered,
            text: format!("{:?}", v),
        }),
    }
}

/// Analyze Parquet file schema for evolution planning
///
/// # Arguments
///
/// * `path` - Path to the Parquet file
///
/// # Returns
///
/// * `Result<ParquetSchemaAnalysis>` - Schema analysis
///
/// # Examples
///
/// ```no_run
/// use pandrs::io::analyze_parquet_schema;
///
/// let analysis = analyze_parquet_schema("data.parquet").expect("operation should succeed");
/// println!("Schema has {} columns", analysis.column_count);
/// ```
pub fn analyze_parquet_schema(path: impl AsRef<Path>) -> Result<ParquetSchemaAnalysis> {
    let metadata = get_parquet_metadata(path.as_ref())?;
    let column_stats = get_column_statistics(path.as_ref())?;

    let column_count = column_stats.len();
    let mut columns = HashMap::new();

    for stat in &column_stats {
        columns.insert(stat.name.clone(), stat.data_type.clone());
    }

    // Calculate complexity score
    let complexity_score = (column_count as f64 * 1.0) + (metadata.num_row_groups as f64 * 0.1);

    // Determine evolution difficulty
    let evolution_difficulty = if column_count < 10 {
        "Easy".to_string()
    } else if column_count < 50 {
        "Medium".to_string()
    } else {
        "Hard".to_string()
    };

    Ok(ParquetSchemaAnalysis {
        column_count,
        columns,
        complexity_score,
        max_nesting_depth: 1, // Simplified for now
        evolution_difficulty,
    })
}
