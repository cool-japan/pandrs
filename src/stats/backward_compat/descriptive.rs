//! Descriptive statistics module
//!
//! This module provides functions for calculating descriptive statistics
//! such as mean, variance, standard deviation, quantiles, and correlation.

use crate::error::{Result, Error};
use crate::stats::DescriptiveStats;

/// Internal implementation for calculating descriptive statistics
pub(crate) fn describe_impl(data: &[f64]) -> Result<DescriptiveStats> {
    if data.is_empty() {
        return Err(Error::EmptyData("At least one data point is required for descriptive statistics".into()));
    }

    let count = data.len();
    
    // Calculate mean
    let mean = data.iter().sum::<f64>() / count as f64;
    
    // Calculate standard deviation (unbiased estimator)
    let variance = if count > 1 {
        let sum_squared_diff = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>();
        sum_squared_diff / (count - 1) as f64
    } else {
        0.0
    };
    let std = variance.sqrt();
    
    // Sort data and calculate quantiles
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let min = sorted[0];
    let max = sorted[count - 1];
    
    // Calculate quantiles
    let median = if count % 2 == 0 {
        (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
    } else {
        sorted[count / 2]
    };
    
    // First quartile (25%)
    let q1 = percentile(&sorted, 0.25);
    
    // Third quartile (75%)
    let q3 = percentile(&sorted, 0.75);
    
    Ok(DescriptiveStats {
        count,
        mean,
        std,
        min,
        q1,
        median,
        q3,
        max,
    })
}

/// Calculate percentile
pub(crate) fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    
    let n = sorted_data.len();
    let idx = p * (n - 1) as f64;
    let idx_floor = idx.floor() as usize;
    let idx_ceil = idx.ceil() as usize;
    
    if idx_floor == idx_ceil {
        return sorted_data[idx_floor];
    }
    
    let weight_ceil = idx - idx_floor as f64;
    let weight_floor = 1.0 - weight_ceil;
    
    sorted_data[idx_floor] * weight_floor + sorted_data[idx_ceil] * weight_ceil
}

/// Internal implementation for calculating covariance
pub(crate) fn covariance_impl(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            format!("Data lengths do not match for covariance calculation: x={}, y={}", x.len(), y.len())
        ));
    }
    
    if x.is_empty() {
        return Err(Error::EmptyData("Covariance calculation requires data".into()));
    }
    
    let n = x.len();
    
    if n <= 1 {
        return Err(Error::InsufficientData("Covariance calculation requires at least 2 data points".into()));
    }
    
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    
    let cov = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>() / (n - 1) as f64;
    
    Ok(cov)
}

/// Internal implementation for calculating correlation coefficient
pub(crate) fn correlation_impl(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            format!("Data lengths do not match for correlation calculation: x={}, y={}", x.len(), y.len())
        ));
    }
    
    if x.is_empty() {
        return Err(Error::EmptyData("Correlation calculation requires data".into()));
    }
    
    let n = x.len();
    
    if n <= 1 {
        return Err(Error::InsufficientData("Correlation calculation requires at least 2 data points".into()));
    }
    
    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    
    // Numerator: Σ(xi - x̄)(yi - ȳ)
    let numerator = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<f64>();
    
    // Denominator: √[Σ(xi - x̄)² * Σ(yi - ȳ)²]
    let sum_squared_diff_x = x.iter()
        .map(|&xi| (xi - mean_x).powi(2))
        .sum::<f64>();
    
    let sum_squared_diff_y = y.iter()
        .map(|&yi| (yi - mean_y).powi(2))
        .sum::<f64>();
    
    let denominator = (sum_squared_diff_x * sum_squared_diff_y).sqrt();
    
    if denominator.abs() < std::f64::EPSILON {
        return Err(Error::Computation("Correlation calculation: zero variance".into()));
    }
    
    Ok(numerator / denominator)
}

/// Calculate variance of the data
///
/// # Description
/// Calculates the variance of the data. By default, uses Bessel's correction
/// (n-1 in the denominator) for unbiased estimation of population variance.
///
/// # Arguments
/// * `data` - The data to calculate variance for
/// * `ddof` - Delta degrees of freedom (typically 1 for unbiased sample variance, 0 for population variance)
///
/// # Example
/// ```
/// use pandrs::stats::descriptive;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let var = descriptive::variance(&data, 1).unwrap();
/// println!("Variance: {}", var);
/// ```
pub fn variance(data: &[f64], ddof: usize) -> Result<f64> {
    if data.is_empty() {
        return Err(Error::EmptyData("Variance calculation requires data".into()));
    }
    
    let n = data.len();
    
    if n <= ddof {
        return Err(Error::InsufficientData("Insufficient data points for variance calculation".into()));
    }
    
    let mean = data.iter().sum::<f64>() / n as f64;
    
    let sum_squared_diff = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>();
    
    Ok(sum_squared_diff / (n - ddof) as f64)
}

/// Calculate standard deviation of the data
///
/// # Description
/// Calculates the standard deviation of the data. By default, uses Bessel's correction
/// for unbiased estimation of population standard deviation.
///
/// # Arguments
/// * `data` - The data to calculate standard deviation for
/// * `ddof` - Delta degrees of freedom (typically 1 for unbiased sample std, 0 for population std)
///
/// # Example
/// ```
/// use pandrs::stats::descriptive;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let std_dev = descriptive::std_dev(&data, 1).unwrap();
/// println!("Standard deviation: {}", std_dev);
/// ```
pub fn std_dev(data: &[f64], ddof: usize) -> Result<f64> {
    let var = variance(data, ddof)?;
    Ok(var.sqrt())
}

/// Calculate the correlation matrix for a set of variables
///
/// # Description
/// Calculates the Pearson correlation coefficients between all pairs of variables.
///
/// # Arguments
/// * `data` - A 2D array where each row represents a variable and each column an observation
///
/// # Example
/// ```
/// use pandrs::stats::descriptive;
///
/// let data = vec![
///     vec![1.0, 2.0, 3.0, 4.0, 5.0],
///     vec![2.0, 3.0, 4.0, 5.0, 6.0],
///     vec![5.0, 4.0, 3.0, 2.0, 1.0]
/// ];
/// let corr_matrix = descriptive::correlation_matrix(&data).unwrap();
/// ```
pub fn correlation_matrix(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    if data.is_empty() {
        return Err(Error::EmptyData("Correlation matrix calculation requires data".into()));
    }
    
    let n_vars = data.len();
    let mut result = vec![vec![0.0; n_vars]; n_vars];
    
    // Diagonal elements (correlation of a variable with itself) are always 1.0
    for i in 0..n_vars {
        result[i][i] = 1.0;
    }
    
    // Calculate correlation for each pair of variables
    for i in 0..n_vars {
        for j in (i+1)..n_vars {
            let corr = correlation_impl(&data[i], &data[j])?;
            result[i][j] = corr;
            result[j][i] = corr; // Correlation matrix is symmetric
        }
    }
    
    Ok(result)
}

/// Calculate specified quantile of the data
///
/// # Description
/// Calculates the specified quantile (percentile) of the data.
///
/// # Arguments
/// * `data` - The data to calculate quantile for
/// * `q` - The quantile to calculate (between 0 and 1)
///
/// # Example
/// ```
/// use pandrs::stats::descriptive;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let median = descriptive::quantile(&data, 0.5).unwrap();
/// println!("Median: {}", median);
/// ```
pub fn quantile(data: &[f64], q: f64) -> Result<f64> {
    if data.is_empty() {
        return Err(Error::EmptyData("Quantile calculation requires data".into()));
    }
    
    if q < 0.0 || q > 1.0 {
        return Err(Error::InvalidInput("Quantile must be between 0 and 1".into()));
    }
    
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    Ok(percentile(&sorted, q))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_describe_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = describe_impl(&data).unwrap();
        
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!((stats.std - 1.5811388300841898).abs() < 1e-10);
        assert!((stats.min - 1.0).abs() < 1e-10);
        assert!((stats.max - 5.0).abs() < 1e-10);
        assert!((stats.median - 3.0).abs() < 1e-10);
        assert!((stats.q1 - 2.0).abs() < 1e-10);
        assert!((stats.q3 - 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_describe_empty() {
        let data: Vec<f64> = vec![];
        let result = describe_impl(&data);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cov = covariance_impl(&x, &y).unwrap();
        assert!((cov - 2.5).abs() < 1e-10);
        
        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let cov_neg = covariance_impl(&x, &y_neg).unwrap();
        assert!((cov_neg + 2.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = correlation_impl(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
        
        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = correlation_impl(&x, &y_neg).unwrap();
        assert!((corr_neg + 1.0).abs() < 1e-10);
        
        let y_uncorr = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let result = correlation_impl(&x, &y_uncorr);
        assert!(result.is_err());
    }

    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var = variance(&data, 1).unwrap();
        assert!((var - 2.5).abs() < 1e-10);
        
        let pop_var = variance(&data, 0).unwrap();
        assert!((pop_var - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let std = std_dev(&data, 1).unwrap();
        assert!((std - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix() {
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0]
        ];
        
        let corr_matrix = correlation_matrix(&data).unwrap();
        assert_eq!(corr_matrix.len(), 3);
        assert_eq!(corr_matrix[0].len(), 3);
        
        // Diagonal elements should be 1.0
        assert!((corr_matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[1][1] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[2][2] - 1.0).abs() < 1e-10);
        
        // Correlation between first and second variables should be 1.0 (perfect correlation)
        assert!((corr_matrix[0][1] - 1.0).abs() < 1e-10);
        
        // Correlation between first and third variables should be -1.0 (perfect negative correlation)
        assert!((corr_matrix[0][2] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let median = quantile(&data, 0.5).unwrap();
        assert!((median - 3.0).abs() < 1e-10);
        
        let q1 = quantile(&data, 0.25).unwrap();
        assert!((q1 - 2.0).abs() < 1e-10);
        
        let q3 = quantile(&data, 0.75).unwrap();
        assert!((q3 - 4.0).abs() < 1e-10);
    }
}