//! Descriptive statistics and summary measures
//!
//! This module provides comprehensive descriptive statistics including measures
//! of central tendency, dispersion, shape, and advanced statistical summaries.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::Series;
use crate::stats::distributions::{Distribution, Normal, TDistribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Comprehensive statistical summary for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalSummary {
    /// Sample size
    pub count: usize,
    /// Mean (average)
    pub mean: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// Mode (most frequent value)
    pub mode: Option<f64>,
    /// Standard deviation
    pub std: f64,
    /// Variance
    pub variance: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Range (max - min)
    pub range: f64,
    /// Interquartile range (Q3 - Q1)
    pub iqr: f64,
    /// Skewness (measure of asymmetry)
    pub skewness: f64,
    /// Kurtosis (measure of tail heaviness)
    pub kurtosis: f64,
    /// Standard error of the mean
    pub standard_error: f64,
    /// Coefficient of variation
    pub coefficient_of_variation: f64,
    /// Quartiles and percentiles
    pub quartiles: Quartiles,
    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,
    /// Outlier information
    pub outliers: OutlierAnalysis,
}

/// Quartile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quartiles {
    /// First quartile (25th percentile)
    pub q1: f64,
    /// Second quartile (50th percentile / median)
    pub q2: f64,
    /// Third quartile (75th percentile)
    pub q3: f64,
    /// Selected percentiles
    pub percentiles: HashMap<u8, f64>,
}

/// Confidence intervals for mean
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// 90% confidence interval
    pub ci_90: (f64, f64),
    /// 95% confidence interval
    pub ci_95: (f64, f64),
    /// 99% confidence interval
    pub ci_99: (f64, f64),
}

/// Outlier detection and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    /// Outliers detected using IQR method
    pub iqr_outliers: Vec<f64>,
    /// Outliers detected using Z-score method (|z| > 3)
    pub z_score_outliers: Vec<f64>,
    /// Outliers detected using modified Z-score method
    pub modified_z_outliers: Vec<f64>,
    /// Number of outliers by each method
    pub outlier_counts: HashMap<String, usize>,
}

/// Calculate comprehensive statistical summary for a numeric array
pub fn describe(data: &[f64]) -> Result<StatisticalSummary> {
    if data.is_empty() {
        return Err(Error::InvalidValue(
            "Cannot compute statistics for empty data".into(),
        ));
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let count = data.len();
    let mean = data.iter().sum::<f64>() / count as f64;
    let median = percentile(&sorted_data, 50.0)?;
    let mode = calculate_mode(data);

    // Variance and standard deviation
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (count - 1) as f64;
    let std = variance.sqrt();

    let min = sorted_data[0];
    let max = sorted_data[count - 1];
    let range = max - min;

    let q1 = percentile(&sorted_data, 25.0)?;
    let q3 = percentile(&sorted_data, 75.0)?;
    let iqr = q3 - q1;

    let skewness = calculate_skewness(data, mean, std)?;
    let kurtosis = calculate_kurtosis(data, mean, std)?;

    let standard_error = std / (count as f64).sqrt();
    let coefficient_of_variation = if mean != 0.0 {
        std / mean.abs()
    } else {
        f64::NAN
    };

    // Calculate percentiles
    let mut percentiles = HashMap::new();
    for p in [5, 10, 25, 50, 75, 90, 95] {
        percentiles.insert(p, percentile(&sorted_data, p as f64)?);
    }

    let quartiles = Quartiles {
        q1,
        q2: median,
        q3,
        percentiles,
    };

    // Calculate confidence intervals
    let confidence_intervals = calculate_confidence_intervals(mean, standard_error, count)?;

    // Outlier analysis
    let outliers = detect_outliers(data, mean, std, q1, q3)?;

    Ok(StatisticalSummary {
        count,
        mean,
        median,
        mode,
        std,
        variance,
        min,
        max,
        range,
        iqr,
        skewness,
        kurtosis,
        standard_error,
        coefficient_of_variation,
        quartiles,
        confidence_intervals,
        outliers,
    })
}

/// Calculate percentile of sorted data
pub fn percentile(sorted_data: &[f64], p: f64) -> Result<f64> {
    if sorted_data.is_empty() {
        return Err(Error::InvalidValue(
            "Cannot compute percentile for empty data".into(),
        ));
    }

    if p < 0.0 || p > 100.0 {
        return Err(Error::InvalidValue(
            "Percentile must be between 0 and 100".into(),
        ));
    }

    if p == 0.0 {
        return Ok(sorted_data[0]);
    }
    if p == 100.0 {
        return Ok(sorted_data[sorted_data.len() - 1]);
    }

    let n = sorted_data.len();
    let index = (p / 100.0) * (n - 1) as f64;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        Ok(sorted_data[lower_index])
    } else {
        let weight = index - lower_index as f64;
        Ok(sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight)
    }
}

/// Calculate mode (most frequent value)
fn calculate_mode(data: &[f64]) -> Option<f64> {
    let mut frequency_map: HashMap<String, (f64, usize)> = HashMap::new();

    // Use string representation to handle floating point precision
    for &value in data {
        let key = format!("{:.10}", value);
        let entry = frequency_map.entry(key).or_insert((value, 0));
        entry.1 += 1;
    }

    // Find the value(s) with maximum frequency
    let max_frequency = frequency_map.values().map(|(_, freq)| *freq).max()?;

    // Only return mode if it appears more than once
    if max_frequency <= 1 {
        return None;
    }

    // Return the first mode found (in case of ties)
    frequency_map
        .values()
        .find(|(_, freq)| *freq == max_frequency)
        .map(|(value, _)| *value)
}

/// Calculate skewness (third moment)
fn calculate_skewness(data: &[f64], mean: f64, std: f64) -> Result<f64> {
    if std == 0.0 {
        return Ok(0.0);
    }

    let n = data.len() as f64;
    let third_moment = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(3))
        .sum::<f64>()
        / n;

    // Sample skewness with bias correction
    let sample_skewness = (n / ((n - 1.0) * (n - 2.0))).sqrt() * third_moment;

    Ok(sample_skewness)
}

/// Calculate kurtosis (fourth moment)
fn calculate_kurtosis(data: &[f64], mean: f64, std: f64) -> Result<f64> {
    if std == 0.0 {
        return Ok(0.0);
    }

    let n = data.len() as f64;
    let fourth_moment = data
        .iter()
        .map(|&x| ((x - mean) / std).powi(4))
        .sum::<f64>()
        / n;

    // Excess kurtosis (subtract 3 for normal distribution baseline)
    let excess_kurtosis = fourth_moment - 3.0;

    // Sample kurtosis with bias correction
    let sample_kurtosis =
        ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * excess_kurtosis + 6.0);

    Ok(sample_kurtosis)
}

/// Calculate confidence intervals for the mean
fn calculate_confidence_intervals(
    mean: f64,
    standard_error: f64,
    n: usize,
) -> Result<ConfidenceIntervals> {
    let df = (n - 1) as f64;
    let t_dist = TDistribution::new(df)?;

    // Critical values for different confidence levels
    let t_90 = t_dist.inverse_cdf(0.95); // 90% CI
    let t_95 = t_dist.inverse_cdf(0.975); // 95% CI
    let t_99 = t_dist.inverse_cdf(0.995); // 99% CI

    let margin_90 = t_90 * standard_error;
    let margin_95 = t_95 * standard_error;
    let margin_99 = t_99 * standard_error;

    Ok(ConfidenceIntervals {
        ci_90: (mean - margin_90, mean + margin_90),
        ci_95: (mean - margin_95, mean + margin_95),
        ci_99: (mean - margin_99, mean + margin_99),
    })
}

/// Detect outliers using multiple methods
fn detect_outliers(data: &[f64], mean: f64, std: f64, q1: f64, q3: f64) -> Result<OutlierAnalysis> {
    let iqr = q3 - q1;

    // IQR method outliers
    let iqr_lower = q1 - 1.5 * iqr;
    let iqr_upper = q3 + 1.5 * iqr;
    let iqr_outliers: Vec<f64> = data
        .iter()
        .filter(|&&x| x < iqr_lower || x > iqr_upper)
        .copied()
        .collect();

    // Z-score method outliers (|z| > 3)
    let z_score_outliers: Vec<f64> = data
        .iter()
        .filter(|&&x| ((x - mean) / std).abs() > 3.0)
        .copied()
        .collect();

    // Modified Z-score method using median absolute deviation
    let median = percentile(
        &{
            let mut sorted = data.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted
        },
        50.0,
    )?;

    let mad = {
        let deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        percentile(&sorted_deviations, 50.0)?
    };

    let modified_z_outliers: Vec<f64> = if mad > 0.0 {
        data.iter()
            .filter(|&&x| (0.6745 * (x - median) / mad).abs() > 3.5)
            .copied()
            .collect()
    } else {
        Vec::new()
    };

    let mut outlier_counts = HashMap::new();
    outlier_counts.insert("IQR".to_string(), iqr_outliers.len());
    outlier_counts.insert("Z-score".to_string(), z_score_outliers.len());
    outlier_counts.insert("Modified Z-score".to_string(), modified_z_outliers.len());

    Ok(OutlierAnalysis {
        iqr_outliers,
        z_score_outliers,
        modified_z_outliers,
        outlier_counts,
    })
}

/// Calculate correlation matrix for multiple variables
pub fn correlation_matrix(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n_vars = data.len();
    if n_vars == 0 {
        return Err(Error::InvalidValue("No variables provided".into()));
    }

    // Check that all variables have the same length
    let n_obs = data[0].len();
    for (i, var) in data.iter().enumerate() {
        if var.len() != n_obs {
            return Err(Error::DimensionMismatch(format!(
                "Variable {} has different length than variable 0",
                i
            )));
        }
    }

    if n_obs < 2 {
        return Err(Error::InvalidValue(
            "At least 2 observations required".into(),
        ));
    }

    let mut correlation_matrix = vec![vec![0.0; n_vars]; n_vars];

    for i in 0..n_vars {
        for j in 0..n_vars {
            if i == j {
                correlation_matrix[i][j] = 1.0;
            } else {
                correlation_matrix[i][j] = pearson_correlation(&data[i], &data[j])?;
            }
        }
    }

    Ok(correlation_matrix)
}

/// Calculate Pearson correlation coefficient
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            "Variables must have same length".into(),
        ));
    }

    if x.len() < 2 {
        return Err(Error::InvalidValue(
            "At least 2 observations required".into(),
        ));
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    let denominator = (sum_xx * sum_yy).sqrt();
    if denominator < 1e-10 {
        Ok(0.0) // Zero correlation when no variance
    } else {
        Ok(sum_xy / denominator)
    }
}

/// Calculate Spearman rank correlation
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            "Variables must have same length".into(),
        ));
    }

    // Convert to ranks
    let ranks_x = calculate_ranks(x);
    let ranks_y = calculate_ranks(y);

    // Calculate Pearson correlation on ranks
    pearson_correlation(&ranks_x, &ranks_y)
}

/// Calculate ranks for Spearman correlation
fn calculate_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed_data: Vec<(usize, f64)> =
        data.iter().enumerate().map(|(i, &val)| (i, val)).collect();

    // Sort by value
    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];

    // Assign ranks, handling ties by averaging
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && indexed_data[j].1 == indexed_data[i].1 {
            j += 1;
        }

        // Average rank for tied values
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0; // 1-based ranking

        for k in i..j {
            ranks[indexed_data[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Calculate covariance matrix
pub fn covariance_matrix(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let n_vars = data.len();
    if n_vars == 0 {
        return Err(Error::InvalidValue("No variables provided".into()));
    }

    let n_obs = data[0].len();
    for (i, var) in data.iter().enumerate() {
        if var.len() != n_obs {
            return Err(Error::DimensionMismatch(format!(
                "Variable {} has different length than variable 0",
                i
            )));
        }
    }

    if n_obs < 2 {
        return Err(Error::InvalidValue(
            "At least 2 observations required".into(),
        ));
    }

    // Calculate means
    let means: Vec<f64> = data
        .iter()
        .map(|var| var.iter().sum::<f64>() / n_obs as f64)
        .collect();

    let mut cov_matrix = vec![vec![0.0; n_vars]; n_vars];

    for i in 0..n_vars {
        for j in 0..n_vars {
            let cov = data[i]
                .iter()
                .zip(data[j].iter())
                .map(|(&xi, &xj)| (xi - means[i]) * (xj - means[j]))
                .sum::<f64>()
                / (n_obs - 1) as f64;

            cov_matrix[i][j] = cov;
        }
    }

    Ok(cov_matrix)
}

/// Summary statistics for grouped data
#[derive(Debug, Clone)]
pub struct GroupedStatistics {
    pub group_summaries: HashMap<String, StatisticalSummary>,
    pub overall_summary: StatisticalSummary,
    pub between_group_variance: f64,
    pub within_group_variance: f64,
    pub f_statistic: f64,
}

/// Calculate summary statistics by groups
pub fn describe_by_groups(values: &[f64], groups: &[String]) -> Result<GroupedStatistics> {
    if values.len() != groups.len() {
        return Err(Error::DimensionMismatch(
            "Values and groups must have same length".into(),
        ));
    }

    if values.is_empty() {
        return Err(Error::InvalidValue(
            "Cannot compute statistics for empty data".into(),
        ));
    }

    // Group data
    let mut grouped_data: HashMap<String, Vec<f64>> = HashMap::new();
    for (value, group) in values.iter().zip(groups.iter()) {
        grouped_data
            .entry(group.clone())
            .or_insert_with(Vec::new)
            .push(*value);
    }

    // Calculate summary for each group
    let mut group_summaries = HashMap::new();
    for (group, group_values) in &grouped_data {
        let summary = describe(group_values)?;
        group_summaries.insert(group.clone(), summary);
    }

    // Overall summary
    let overall_summary = describe(values)?;

    // Calculate between and within group variance for ANOVA
    let overall_mean = overall_summary.mean;
    let mut between_group_ss = 0.0;
    let mut within_group_ss = 0.0;
    let mut total_n = 0;

    for (_, group_values) in &grouped_data {
        let group_mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
        let group_n = group_values.len();
        total_n += group_n;

        // Between group sum of squares
        between_group_ss += group_n as f64 * (group_mean - overall_mean).powi(2);

        // Within group sum of squares
        for &value in group_values {
            within_group_ss += (value - group_mean).powi(2);
        }
    }

    let k = grouped_data.len() as f64; // number of groups
    let between_group_variance = between_group_ss / (k - 1.0);
    let within_group_variance = within_group_ss / (total_n as f64 - k);

    let f_statistic = if within_group_variance > 0.0 {
        between_group_variance / within_group_variance
    } else {
        f64::INFINITY
    };

    Ok(GroupedStatistics {
        group_summaries,
        overall_summary,
        between_group_variance,
        within_group_variance,
        f_statistic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_describe() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let summary = describe(&data).unwrap();

        assert_eq!(summary.count, 5);
        assert_eq!(summary.mean, 3.0);
        assert_eq!(summary.median, 3.0);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 5.0);
        assert_eq!(summary.range, 4.0);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile(&data, 0.0).unwrap(), 1.0);
        assert_eq!(percentile(&data, 50.0).unwrap(), 3.0);
        assert_eq!(percentile(&data, 100.0).unwrap(), 5.0);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = pearson_correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation
    }

    #[test]
    fn test_correlation_matrix() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![1.0, 3.0, 2.0],
        ];

        let matrix = correlation_matrix(&data).unwrap();

        // Diagonal should be 1.0
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        assert!((matrix[1][1] - 1.0).abs() < 1e-10);
        assert!((matrix[2][2] - 1.0).abs() < 1e-10);

        // Matrix should be symmetric
        assert!((matrix[0][1] - matrix[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_detection() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is an outlier
        let summary = describe(&data).unwrap();

        assert!(!summary.outliers.iqr_outliers.is_empty());
        assert!(summary.outliers.iqr_outliers.contains(&100.0));
    }

    #[test]
    fn test_mode_calculation() {
        let data = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let mode = calculate_mode(&data);

        assert_eq!(mode, Some(2.0));

        // Test no mode (all unique)
        let unique_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let no_mode = calculate_mode(&unique_data);
        assert_eq!(no_mode, None);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // Monotonic but not linear

        let spearman = spearman_correlation(&x, &y).unwrap();
        assert!((spearman - 1.0).abs() < 1e-10); // Perfect rank correlation
    }
}
