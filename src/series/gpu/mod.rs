use std::fmt::Debug;

use crate::core::error::{Error, Result};
use crate::series::base::Series;

/// GPU acceleration extensions for Series
pub trait SeriesGpuExt<T: Debug + Clone> {
    /// Apply GPU acceleration to a series
    fn gpu_accelerate(&self) -> Result<Self>
    where
        Self: Sized;

    /// GPU-accelerated sum
    fn gpu_sum(&self) -> Result<T>;

    /// GPU-accelerated mean
    fn gpu_mean(&self) -> Result<T>;

    /// GPU-accelerated standard deviation
    fn gpu_std(&self) -> Result<T>;

    /// GPU-accelerated correlation
    fn gpu_corr(&self, other: &Self) -> Result<f64>
    where
        Self: Sized;
}

// Implement for f64 series
//
// HONESTY NOTE: there is no real CUDA kernel behind any of these entry
// points (cudarc 0.19.x exposes no cuBLAS/cuSOLVER bindings this crate uses
// — see the module-level notes throughout `crate::gpu`). `gpu_sum`,
// `gpu_mean`, `gpu_std` and `gpu_corr` used to all honestly return
// `NotImplemented`, but `Series<f64>` already has real, correct
// implementations of exactly these reductions (`Series::sum`/`Series::mean`)
// or one call away (`stats::descriptive::std_dev`/`pearson_correlation`), so
// returning an error for them was refusing to do arithmetic the crate is
// perfectly capable of, not refusing to fake a GPU. These now route through
// that same real CPU math — the same functions the Python bindings and the
// rest of the crate use — rather than duplicating (and risking diverging
// from) it. `gpu_accelerate` alone stays `NotImplemented`: unlike the
// reductions above, there is no CPU computation it could honestly perform
// instead (there is nothing to "accelerate" ahead of time; see
// `dataframe::gpu::DataFrameGpuExt::gpu_accelerate`).
impl SeriesGpuExt<f64> for Series<f64> {
    fn gpu_accelerate(&self) -> Result<Self> {
        // No real GPU kernel exists here. Returning a clone while claiming
        // "GPU acceleration" would be dishonest, so report it honestly.
        Err(Error::NotImplemented(
            "GPU acceleration for Series not implemented (no real CUDA kernel)".into(),
        ))
    }

    fn gpu_sum(&self) -> Result<f64> {
        // `Series::<f64>::sum` already skips `NaN` (this crate's
        // missing-value sentinel for a plain float series; see its doc
        // comment) and is infallible.
        Ok(self.sum())
    }

    fn gpu_mean(&self) -> Result<f64> {
        self.mean()
    }

    fn gpu_std(&self) -> Result<f64> {
        // Sample standard deviation (`ddof = 1`, matching pandas'
        // `Series.std()` default and this crate's ddof=1 convention
        // elsewhere), skipping `NaN` for consistency with `sum`/`mean`
        // above (`stats::descriptive::std_dev` itself does not skip `NaN`:
        // it is a generic `&[f64]` statistic used in contexts where `NaN`
        // is a real value, not a missing-data sentinel).
        let clean: Vec<f64> = self
            .values()
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .collect();
        crate::stats::descriptive::std_dev(&clean, 1)
    }

    fn gpu_corr(&self, other: &Self) -> Result<f64> {
        if self.len() != other.len() {
            return Err(Error::DimensionMismatch(format!(
                "Series lengths differ: {} vs {}",
                self.len(),
                other.len()
            )));
        }

        // Listwise deletion: a pair of observations only contributes when
        // *both* series have a real value at that position. Filtering each
        // series independently before zipping (dropping `NaN`s one column
        // at a time) would shift the surviving observations against each
        // other whenever the two series don't have `NaN` at exactly the
        // same positions, silently correlating misaligned pairs — the same
        // class of bug fixed in `py_gpu.rs`'s `gpu_corr` and
        // `dataframe::gpu`'s `gpu_corr` (see `gpu::cpu_math::
        // correlation_matrix_listwise`, which this mirrors for the
        // two-series case).
        let (xs, ys): (Vec<f64>, Vec<f64>) = self
            .values()
            .iter()
            .zip(other.values().iter())
            .filter(|(&x, &y)| !x.is_nan() && !y.is_nan())
            .map(|(&x, &y)| (x, y))
            .unzip();

        crate::stats::descriptive::pearson_correlation(&xs, &ys)
    }
}
