//! Real CPU math shared by the GPU-dispatch entry points.
//!
//! None of these functions run on a GPU: cudarc 0.19.x exposes no
//! cuBLAS/cuSOLVER bindings this crate uses (see the module-level honesty
//! notes throughout `crate::gpu`), so every `gpu_*` entry point in this
//! crate computes on the CPU regardless of device availability. This module
//! exists so that real math is written and tested exactly **once** and
//! shared by every caller that needs it — `dataframe::gpu::DataFrameGpuExt`,
//! `series::gpu::SeriesGpuExt`, and the Python bindings
//! (`py_bindings::py_gpu`) — instead of each surface hand-rolling its own
//! (and risking diverging) copy. `py_bindings/src/py_gpu.rs::gpu_corr` used
//! to have exactly this bug: it dropped nulls independently, one column at a
//! time, before computing pairwise correlations, silently misaligning rows
//! whenever two columns didn't have nulls at exactly the same positions.

use scirs2_core::ndarray::Array2;

use crate::error::{Error, Result};
use crate::stats::descriptive::pearson_correlation;

/// Compute a Pearson correlation matrix over columns using **listwise
/// deletion**: a row contributes to *every* pairwise correlation only when
/// all columns hold a value at that row.
///
/// Filtering each column's nulls independently (pairwise-complete deletion
/// done incorrectly, e.g. `column.iter().filter_map(|v| *v).collect()` per
/// column before zipping) shifts the surviving observations of one column
/// against another whenever they don't have nulls at exactly the same row
/// positions — silently correlating misaligned pairs of numbers instead of
/// the pairs that were actually observed together. Listwise deletion keeps
/// every pairwise computation aligned to the same set of complete
/// observations.
///
/// A pair of columns whose complete-case overlap is too small to define a
/// correlation (fewer than 2 shared complete rows) reports `NaN` for that
/// specific cell rather than failing the whole matrix — matching how
/// [`pearson_correlation`] itself is used elsewhere for an
/// under-determined pair, and how a real-world pairwise-correlation report
/// (e.g. pandas' `DataFrame.corr()`) surfaces the same situation.
///
/// # Errors
/// Returns `Error::InvalidValue` if `columns` is empty, and
/// `Error::DimensionMismatch` if the columns don't all have the same number
/// of rows.
pub fn correlation_matrix_listwise(columns: &[Vec<Option<f64>>]) -> Result<Array2<f64>> {
    let n = columns.len();
    if n == 0 {
        return Err(Error::InvalidValue(
            "correlation matrix requires at least one column".to_string(),
        ));
    }

    let n_rows = columns[0].len();
    for (i, col) in columns.iter().enumerate() {
        if col.len() != n_rows {
            return Err(Error::DimensionMismatch(format!(
                "column {} has {} rows, expected {} (from column 0)",
                i,
                col.len(),
                n_rows
            )));
        }
    }

    // Rows where every column holds a value.
    let complete_rows: Vec<usize> = (0..n_rows)
        .filter(|&row| columns.iter().all(|col| col[row].is_some()))
        .collect();

    // Each column's complete-case values, in the same row order, so that
    // `complete[i][r]` and `complete[j][r]` are always the same original
    // row for every `i`, `j`.
    let complete: Vec<Vec<f64>> = columns
        .iter()
        .map(|col| complete_rows.iter().filter_map(|&row| col[row]).collect())
        .collect();

    let mut matrix = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        matrix[[i, i]] = 1.0;
        for j in (i + 1)..n {
            let r = pearson_correlation(&complete[i], &complete[j]).unwrap_or(f64::NAN);
            matrix[[i, j]] = r;
            matrix[[j, i]] = r;
        }
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listwise_deletion_keeps_pairs_aligned() {
        // Column b is a perfect 2x of column a, but only at the rows where
        // NEITHER column is null. If nulls were dropped independently per
        // column instead of by row, the surviving values would be
        // misaligned and the correlation would not measure 1.0.
        let a = vec![Some(1.0), None, Some(3.0), Some(4.0), Some(5.0)];
        let b = vec![Some(2.0), Some(100.0), None, Some(8.0), Some(10.0)];

        let matrix = correlation_matrix_listwise(&[a, b]).expect("operation should succeed");
        assert_eq!(matrix.shape(), &[2, 2]);
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-12);
        assert!((matrix[[1, 1]] - 1.0).abs() < 1e-12);
        // Complete rows are indices 0, 3, 4: a = [1,4,5], b = [2,8,10] = 2*a.
        assert!(
            (matrix[[0, 1]] - 1.0).abs() < 1e-9,
            "expected perfect correlation on the complete rows, got {}",
            matrix[[0, 1]]
        );
        assert!(
            (matrix[[1, 0]] - matrix[[0, 1]]).abs() < 1e-15,
            "matrix must be symmetric"
        );
    }

    #[test]
    fn empty_columns_errs() {
        assert!(correlation_matrix_listwise(&[]).is_err());
    }

    #[test]
    fn mismatched_lengths_err() {
        let a = vec![Some(1.0), Some(2.0)];
        let b = vec![Some(1.0)];
        assert!(correlation_matrix_listwise(&[a, b]).is_err());
    }
}
