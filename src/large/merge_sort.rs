//! External merge sort for large CSV datasets.
//!
//! Implements a classic external sort that:
//! 1. Reads the input in sorted chunks that fit in RAM.
//! 2. Writes each sorted chunk to a temporary file.
//! 3. K-way merges the sorted chunk files into the final output.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use csv::{ReaderBuilder, WriterBuilder};

use crate::core::error::{Error, Result};
use crate::large::out_of_core::OutOfCoreConfig;

// ---------------------------------------------------------------------------
// External sort entry point
// ---------------------------------------------------------------------------

/// Sort a large CSV file using external merge sort.
///
/// The file is processed in chunks of `config.chunk_size` rows.  Each chunk
/// is sorted in memory by the `sort_column` and written to a temporary file.
/// The temporary files are then k-way merged into `output_path`.
///
/// The sort is purely lexicographic over the string representation of the
/// column value.  For numeric-aware sorting, convert the column to a
/// zero-padded or sign-magnitude string representation before calling this.
pub fn external_sort(
    input_path: &str,
    output_path: &str,
    sort_column: &str,
    ascending: bool,
    config: &OutOfCoreConfig,
) -> Result<()> {
    // Phase 1 – create sorted run files
    let run_paths = create_sorted_runs(input_path, sort_column, ascending, config)?;

    // Phase 2 – k-way merge
    merge_sorted_chunks(&run_paths, output_path, sort_column, ascending)?;

    // Clean up temporary run files
    for p in &run_paths {
        let _ = std::fs::remove_file(p);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 1: create sorted runs
// ---------------------------------------------------------------------------

fn create_sorted_runs(
    input_path: &str,
    sort_column: &str,
    ascending: bool,
    config: &OutOfCoreConfig,
) -> Result<Vec<PathBuf>> {
    let file = File::open(input_path).map_err(|e| Error::IoError(e.to_string()))?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(BufReader::new(file));

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| Error::CsvError(e.to_string()))?
        .iter()
        .map(|h| h.to_string())
        .collect();

    // Find sort column index
    let sort_col_idx = headers
        .iter()
        .position(|h| h == sort_column)
        .ok_or_else(|| {
            Error::Column(format!(
                "Sort column '{}' not found in headers",
                sort_column
            ))
        })?;

    let chunk_size = config.effective_chunk_size(headers.len());
    let temp_dir = &config.temp_dir;
    let mut run_index = 0usize;
    let mut run_paths: Vec<PathBuf> = Vec::new();
    let mut current_rows: Vec<Vec<String>> = Vec::with_capacity(chunk_size);

    let flush_run = |rows: &mut Vec<Vec<String>>,
                     headers: &[String],
                     sort_idx: usize,
                     asc: bool,
                     dir: &Path,
                     idx: &mut usize|
     -> Result<PathBuf> {
        // Sort
        rows.sort_by(|a, b| {
            let va = a.get(sort_idx).map(|s| s.as_str()).unwrap_or("");
            let vb = b.get(sort_idx).map(|s| s.as_str()).unwrap_or("");
            // Try numeric comparison first
            let ord = compare_values(va, vb);
            if asc {
                ord
            } else {
                ord.reverse()
            }
        });

        // Write
        let path = dir.join(format!("pandrs_sort_run_{}.csv", *idx));
        let f = File::create(&path).map_err(|e| Error::IoError(e.to_string()))?;
        let mut wtr = WriterBuilder::new().from_writer(BufWriter::new(f));
        wtr.write_record(headers)
            .map_err(|e| Error::CsvError(e.to_string()))?;
        for row in rows.iter() {
            wtr.write_record(row)
                .map_err(|e| Error::CsvError(e.to_string()))?;
        }
        wtr.flush().map_err(|e| Error::IoError(e.to_string()))?;

        rows.clear();
        *idx += 1;
        Ok(path)
    };

    for record_result in rdr.records() {
        let record = record_result.map_err(|e| Error::CsvError(e.to_string()))?;
        let row: Vec<String> = record.iter().map(|f| f.to_string()).collect();
        current_rows.push(row);
        if current_rows.len() >= chunk_size {
            let p = flush_run(
                &mut current_rows,
                &headers,
                sort_col_idx,
                ascending,
                temp_dir,
                &mut run_index,
            )?;
            run_paths.push(p);
        }
    }

    if !current_rows.is_empty() {
        let p = flush_run(
            &mut current_rows,
            &headers,
            sort_col_idx,
            ascending,
            temp_dir,
            &mut run_index,
        )?;
        run_paths.push(p);
    }

    Ok(run_paths)
}

// ---------------------------------------------------------------------------
// Phase 2: k-way merge
// ---------------------------------------------------------------------------

/// Item for the min-heap used during k-way merge.
struct HeapItem {
    /// The sort key (string) for comparison.
    key: String,
    /// The complete CSV row.
    row: Vec<String>,
    /// Which run file this came from.
    run_index: usize,
    /// Whether we are sorting ascending.
    ascending: bool,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; invert comparison for min-heap behaviour.
        let ord = compare_values(&self.key, &other.key);
        let ord = if self.ascending { ord.reverse() } else { ord };
        ord
    }
}

/// Compare two CSV cell values for external-sort/merge ordering.
///
/// Values that parse as `f64` sort before values that don't, and are
/// compared numerically via `f64::total_cmp` (a real total order, unlike
/// `partial_cmp(..).unwrap_or(Equal)`, which treats every NaN as equal to
/// everything and breaks transitivity). Values that don't parse are
/// compared lexicographically against each other.
///
/// This decides per-*value* rather than per-column: a previous version
/// tried `(a.parse(), b.parse())` per PAIR and fell back to comparing the
/// raw text whenever either side failed to parse. That is not a total
/// order -- for "2" < "10" (both numeric) and "10" < "1a" (text fallback,
/// since "1a" doesn't parse) it also judged "1a" < "2" (also a text
/// fallback), a genuine three-way cycle. Bucketing by "parses as a number"
/// per value first, then comparing consistently within each bucket, keeps
/// the order transitive even for a column that mixes numeric and
/// non-numeric text (e.g. missing values stored as an empty string among
/// otherwise-numeric IDs): numeric values first in numeric order,
/// non-numeric values after in lexicographic order.
fn compare_values(a: &str, b: &str) -> Ordering {
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(fa), Ok(fb)) => fa.total_cmp(&fb),
        (Ok(_), Err(_)) => Ordering::Less,
        (Err(_), Ok(_)) => Ordering::Greater,
        (Err(_), Err(_)) => a.cmp(b),
    }
}

/// Merge multiple sorted chunk CSV files into a single sorted CSV file.
///
/// This performs a k-way merge using a binary min-heap, so memory usage is
/// proportional to the number of runs (one row buffered per run).
pub fn merge_sorted_chunks(
    chunk_paths: &[PathBuf],
    output_path: &str,
    sort_column: &str,
    ascending: bool,
) -> Result<()> {
    if chunk_paths.is_empty() {
        // Write an empty file
        File::create(output_path).map_err(|e| Error::IoError(e.to_string()))?;
        return Ok(());
    }

    // Open all run files and read their headers
    let mut readers: Vec<csv::Reader<BufReader<File>>> = Vec::with_capacity(chunk_paths.len());
    let mut headers_per_run: Vec<Vec<String>> = Vec::with_capacity(chunk_paths.len());

    for path in chunk_paths {
        let f = File::open(path).map_err(|e| Error::IoError(e.to_string()))?;
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(BufReader::new(f));
        let hdrs: Vec<String> = rdr
            .headers()
            .map_err(|e| Error::CsvError(e.to_string()))?
            .iter()
            .map(|h| h.to_string())
            .collect();
        headers_per_run.push(hdrs);
        readers.push(rdr);
    }

    // Use headers from first run as canonical
    let headers = headers_per_run[0].clone();
    let sort_col_idx = headers
        .iter()
        .position(|h| h == sort_column)
        .ok_or_else(|| {
            Error::Column(format!(
                "Sort column '{}' not found during merge",
                sort_column
            ))
        })?;

    // Open output file
    let out_file = File::create(output_path).map_err(|e| Error::IoError(e.to_string()))?;
    let mut wtr = WriterBuilder::new().from_writer(BufWriter::new(out_file));
    wtr.write_record(&headers)
        .map_err(|e| Error::CsvError(e.to_string()))?;

    // Wrap readers in iterators we can advance independently
    // We need to be able to call next() per run; use record iterators stored as
    // boxed dynamic iterators to avoid recursive type issues.
    let mut record_iters: Vec<csv::StringRecordsIntoIter<BufReader<File>>> =
        readers.into_iter().map(|r| r.into_records()).collect();

    // Seed the heap
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();

    for (run_idx, iter) in record_iters.iter_mut().enumerate() {
        if let Some(result) = iter.next() {
            let record = result.map_err(|e| Error::CsvError(e.to_string()))?;
            let row: Vec<String> = record.iter().map(|f| f.to_string()).collect();
            let key = row.get(sort_col_idx).cloned().unwrap_or_default();
            heap.push(HeapItem {
                key,
                row,
                run_index: run_idx,
                ascending,
            });
        }
    }

    // Merge loop
    while let Some(item) = heap.pop() {
        wtr.write_record(&item.row)
            .map_err(|e| Error::CsvError(e.to_string()))?;

        let run_idx = item.run_index;
        if let Some(result) = record_iters[run_idx].next() {
            let record = result.map_err(|e| Error::CsvError(e.to_string()))?;
            let row: Vec<String> = record.iter().map(|f| f.to_string()).collect();
            let key = row.get(sort_col_idx).cloned().unwrap_or_default();
            heap.push(HeapItem {
                key,
                row,
                run_index: run_idx,
                ascending,
            });
        }
    }

    wtr.flush().map_err(|e| Error::IoError(e.to_string()))?;
    Ok(())
}
