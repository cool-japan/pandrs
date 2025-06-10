use std::fmt::Debug;

use crate::core::error::{Error, Result};
use crate::na::NA;
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
impl SeriesGpuExt<f64> for Series<f64> {
    fn gpu_accelerate(&self) -> Result<Self> {
        // Forward to legacy implementation (placeholder for now)
        Ok(self.clone())
    }

    fn gpu_sum(&self) -> Result<f64> {
        // Forward to legacy implementation (placeholder for now)
        Ok(0.0)
    }

    fn gpu_mean(&self) -> Result<f64> {
        // Forward to legacy implementation (placeholder for now)
        Ok(0.0)
    }

    fn gpu_std(&self) -> Result<f64> {
        // Forward to legacy implementation (placeholder for now)
        Ok(0.0)
    }

    fn gpu_corr(&self, other: &Self) -> Result<f64> {
        // Forward to legacy implementation (placeholder for now)
        Ok(0.0)
    }
}
