//! Inferential statistics and hypothesis testing module
//!
//! This module provides functions for statistical inference and hypothesis testing,
//! including t-tests, ANOVA, chi-square tests, and non-parametric tests.

use crate::error::{Result, Error};
use crate::stats::{TTestResult, AnovaResult, MannWhitneyResult, ChiSquareResult};
use std::f64::consts::PI;
use std::collections::HashMap;

/// Calculate standard normal distribution CDF (cumulative distribution function)
pub(crate) fn normal_cdf(z: f64) -> f64 {
    // Calculate approximation of error function (pure Rust implementation)
    // Approximation calculation for standard normal distribution CDF (Abramowitz and Stegun)
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / (2.0_f64).sqrt();
    
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    
    0.5 * (1.0 + sign * y)
}

/// Calculate t-distribution CDF (cumulative distribution function)
pub(crate) fn t_distribution_cdf(t: f64, df: usize) -> f64 {
    // Use normal distribution approximation (for large degrees of freedom)
    if df > 30 {
        return normal_cdf(t);
    }
    
    // Here we use a simplified approximation
    // In a real implementation, higher precision calculation would be needed
    let df_f64 = df as f64;
    let x = df_f64 / (df_f64 + t * t);
    let a = 0.5 * df_f64;
    let b = 0.5;
    
    // Approximation calculation for incomplete beta function (for accurate t-distribution CDF)
    // This part should use a numerical calculation library in practice
    let beta_approx = if t > 0.0 {
        1.0 - 0.5 * x.powf(a)
    } else {
        0.5 * x.powf(a)
    };
    
    beta_approx
}

/// Internal implementation for two-sample t-test
pub(crate) fn ttest_impl(
    sample1: &[f64],
    sample2: &[f64],
    alpha: f64,
    equal_var: bool,
) -> Result<TTestResult> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(Error::EmptyData("t-test requires data".into()));
    }
    
    let n1 = sample1.len();
    let n2 = sample2.len();
    
    if n1 < 2 || n2 < 2 {
        return Err(Error::InsufficientData("t-test requires at least 2 data points in each group".into()));
    }
    
    // Calculate means
    let mean1 = sample1.iter().sum::<f64>() / n1 as f64;
    let mean2 = sample2.iter().sum::<f64>() / n2 as f64;
    
    // Calculate variances
    let var1 = sample1.iter()
        .map(|&x| (x - mean1).powi(2))
        .sum::<f64>() / (n1 - 1) as f64;
    
    let var2 = sample2.iter()
        .map(|&x| (x - mean2).powi(2))
        .sum::<f64>() / (n2 - 1) as f64;
    
    let (t_stat, df) = if equal_var {
        // Equal variance assumption t-statistic
        let pooled_var = ((n1 - 1) as f64 * var1 + (n2 - 1) as f64 * var2) / 
                          (n1 + n2 - 2) as f64;
        let std_err = (pooled_var * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
        let t_value = (mean1 - mean2) / std_err;
        (t_value, n1 + n2 - 2)
    } else {
        // Welch's t-test (no equal variance assumption)
        let std_err = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();
        let t_value = (mean1 - mean2) / std_err;
        
        // Welch-Satterthwaite approximation for degrees of freedom
        let df_num = (var1 / n1 as f64 + var2 / n2 as f64).powi(2);
        let df_denom = (var1 / n1 as f64).powi(2) / (n1 - 1) as f64 +
                       (var2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df_welch = df_num / df_denom;
        (t_value, df_welch.floor() as usize)
    };
    
    // Two-tailed test p-value calculation
    let p_value = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));
    
    Ok(TTestResult {
        statistic: t_stat,
        pvalue: p_value,
        significant: p_value < alpha,
        df,
    })
}

/// Convert chi-square value to p-value
pub(crate) fn chi2_to_pvalue(chi2: f64, df: usize) -> f64 {
    // Simplified implementation (more accurate calculation needed in practice)
    // Should use special function library in real implementation
    let k = df as f64 / 2.0;
    let x = chi2 / 2.0;
    
    // Approximation calculation for gamma function
    let gamma_k = if df % 2 == 0 {
        1.0 // k is integer
    } else {
        (PI * 2.0).sqrt() // k + 0.5 is integer
    };
    
    // Approximation calculation for lower incomplete gamma function
    let p = if chi2 > df as f64 + 2.0 {
        1.0 - gamma_k * (1.0 - x.exp() * (1.0 + x + 0.5 * x.powi(2)))
    } else {
        gamma_k * x.exp() * x.powf(k - 1.0)
    };
    
    1.0 - p.min(1.0).max(0.0)
}

/// F-distribution cumulative distribution function (CDF)
/// Calculate p-value for F-distribution (approximate)
pub(crate) fn f_distribution_cdf(f: f64, df1: usize, df2: usize) -> f64 {
    // Approximation calculation for F-distribution (higher precision implementation needed for large degrees of freedom)
    // Real library implementation should use special function library
    
    // Use relationship between F-distribution and beta distribution
    let df1_f64 = df1 as f64;
    let df2_f64 = df2 as f64;
    let x = df1_f64 * f / (df1_f64 * f + df2_f64);
    
    // Approximation calculation for incomplete beta function
    let a = df1_f64 / 2.0;
    let b = df2_f64 / 2.0;
    
    // Simplified approximation
    let beta_approx = if x > 0.5 {
        // Approximation for x > 0.5
        1.0 - (1.0 - x).powf(b) * (1.0 + (1.0 - x) * a / b + 
                                (1.0 - x).powi(2) * a * (a + 1.0) / (b * (b + 1.0)) / 2.0)
    } else {
        // Approximation for x <= 0.5
        x.powf(a) * (1.0 + x * b / a + 
                    x.powi(2) * b * (b + 1.0) / (a * (a + 1.0)) / 2.0)
    };
    
    beta_approx.min(1.0).max(0.0)
}

/// Implementation for one-way ANOVA
pub(crate) fn anova_impl(
    groups: &HashMap<&str, &[f64]>, 
    alpha: f64
) -> Result<AnovaResult> {
    // Check number of groups and sample size for each group
    if groups.is_empty() {
        return Err(Error::EmptyData("ANOVA requires at least one group".into()));
    }
    
    if groups.len() < 2 {
        return Err(Error::InsufficientData("ANOVA requires at least two groups".into()));
    }
    
    // Calculate total number of data points, group means, and overall mean
    let mut total_n = 0;
    let mut global_sum = 0.0;
    
    for (_, values) in groups.iter() {
        if values.is_empty() {
            return Err(Error::EmptyData("There is an empty group".into()));
        }
        
        total_n += values.len();
        global_sum += values.iter().sum::<f64>();
    }
    
    let global_mean = global_sum / total_n as f64;
    
    // Calculate sum of squares between groups (SSB), within groups (SSW), and total (SST)
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    let mut ss_total = 0.0;
    
    for (_, values) in groups.iter() {
        let group_n = values.len();
        let group_mean = values.iter().sum::<f64>() / group_n as f64;
        
        // Calculate sum of squares between groups
        ss_between += group_n as f64 * (group_mean - global_mean).powi(2);
        
        // Calculate sum of squares within groups
        for &value in *values {
            // Within-group variation
            ss_within += (value - group_mean).powi(2);
            
            // Total variation (for verification)
            ss_total += (value - global_mean).powi(2);
        }
    }
    
    // Calculate degrees of freedom
    let df_between = groups.len() - 1;
    let df_within = total_n - groups.len();
    let df_total = total_n - 1;
    
    // Calculate mean squares (MS)
    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;
    
    // Calculate F-statistic
    let f_statistic = ms_between / ms_within;
    
    // Calculate p-value (using F-distribution)
    let p_value = 1.0 - f_distribution_cdf(f_statistic, df_between, df_within);
    
    // Return result
    Ok(AnovaResult {
        f_statistic,
        p_value,
        ss_between,
        ss_within,
        ss_total,
        df_between,
        df_within,
        df_total,
        ms_between,
        ms_within,
        significant: p_value < alpha,
    })
}

/// Implementation for Mann-Whitney U test (non-parametric test)
pub(crate) fn mann_whitney_u_impl(
    sample1: &[f64],
    sample2: &[f64],
    alpha: f64
) -> Result<MannWhitneyResult> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(Error::EmptyData("Mann-Whitney U test requires data".into()));
    }
    
    let n1 = sample1.len();
    let n2 = sample2.len();
    
    // Combine both samples and rank
    let mut combined: Vec<(f64, usize, usize)> = Vec::with_capacity(n1 + n2);
    
    // Add group 1 data
    for (i, &val) in sample1.iter().enumerate() {
        combined.push((val, 0, i)); // Group 0, index i
    }
    
    // Add group 2 data
    for (i, &val) in sample2.iter().enumerate() {
        combined.push((val, 1, i)); // Group 1, index i
    }
    
    // Sort by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // Rank
    let mut ranks = vec![0.0; n1 + n2];
    let mut i = 0;
    
    while i < n1 + n2 {
        let mut j = i;
        // Find data with the same value
        while j < n1 + n2 - 1 && (combined[j].0 - combined[j + 1].0).abs() < f64::EPSILON {
            j += 1;
        }
        
        // Assign average rank for ties
        if j > i {
            let rank_avg = (i + 1 + j + 1) as f64 / 2.0;
            for k in i..=j {
                let (_, group, idx) = combined[k];
                if group == 0 {
                    ranks[idx] = rank_avg;
                } else {
                    ranks[idx + n1] = rank_avg;
                }
            }
        } else {
            let (_, group, idx) = combined[i];
            if group == 0 {
                ranks[idx] = (i + 1) as f64;
            } else {
                ranks[idx + n1] = (i + 1) as f64;
            }
        }
        
        i = j + 1;
    }
    
    // Calculate rank sum for group 1
    let r1: f64 = ranks.iter().take(n1).sum();
    
    // Calculate U statistic
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    
    // Use the smaller U value
    let u_statistic = u1.min(u2);
    
    // Calculate mean and standard deviation
    let mean_u = (n1 * n2) as f64 / 2.0;
    let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
    
    // Calculate p-value using normal approximation
    let z = (u_statistic - mean_u) / std_u;
    let p_value = 2.0 * normal_cdf(-z.abs()); // Two-tailed test
    
    Ok(MannWhitneyResult {
        u_statistic,
        p_value,
        significant: p_value < alpha,
    })
}

/// Implementation for chi-square test
pub(crate) fn chi_square_test_impl(
    observed: &[Vec<f64>],
    alpha: f64
) -> Result<ChiSquareResult> {
    // Validate observed data
    if observed.is_empty() {
        return Err(Error::EmptyData("Chi-square test requires observed data".into()));
    }
    
    let rows = observed.len();
    if rows < 2 {
        return Err(Error::InsufficientData("Chi-square test requires at least 2 rows of data".into()));
    }
    
    let cols = observed[0].len();
    if cols < 2 {
        return Err(Error::InsufficientData("Chi-square test requires at least 2 columns of data".into()));
    }
    
    // Ensure all rows have the same number of columns
    for row in observed.iter() {
        if row.len() != cols {
            return Err(Error::InvalidInput("All rows must have the same number of columns".into()));
        }
    }
    
    // Calculate row and column sums
    let mut row_sums = vec![0.0; rows];
    let mut col_sums = vec![0.0; cols];
    let mut total_sum = 0.0;
    
    for i in 0..rows {
        for j in 0..cols {
            let value = observed[i][j];
            if value < 0.0 {
                return Err(Error::InvalidInput("Observed values must not be negative".into()));
            }
            row_sums[i] += value;
            col_sums[j] += value;
            total_sum += value;
        }
    }
    
    if total_sum < 1.0 {
        return Err(Error::InvalidInput("Sum of observed data is zero".into()));
    }
    
    // Calculate expected frequencies
    let mut expected = vec![vec![0.0; cols]; rows];
    let mut chi2_statistic = 0.0;
    
    for i in 0..rows {
        for j in 0..cols {
            // Expected frequency = (row sum * column sum) / total sum
            expected[i][j] = row_sums[i] * col_sums[j] / total_sum;
            
            // Warning if expected frequency is less than 5 (Yates' correction may be needed)
            if expected[i][j] < 5.0 {
                // Here we just show a warning (in a real library, log output or similar)
                // println!("Warning: There are cells with expected frequency less than 5. Interpret results with caution.");
            }
            
            // Calculate chi-square statistic
            let diff = observed[i][j] - expected[i][j];
            chi2_statistic += diff * diff / expected[i][j];
        }
    }
    
    // Calculate degrees of freedom
    let df = (rows - 1) * (cols - 1);
    
    // Calculate p-value
    let p_value = chi2_to_pvalue(chi2_statistic, df);
    
    Ok(ChiSquareResult {
        chi2_statistic,
        p_value,
        df,
        significant: p_value < alpha,
        expected_freq: expected,
    })
}

/// Perform one-sample t-test
///
/// # Description
/// Tests if the mean of a sample differs from a hypothesized population mean.
///
/// # Arguments
/// * `sample` - The sample data
/// * `pop_mean` - The hypothesized population mean to test against
/// * `alpha` - Significance level (e.g., 0.05 for 95% confidence)
///
/// # Example
/// ```
/// use pandrs::stats::inference;
///
/// let sample = vec![5.1, 5.3, 5.6, 5.2, 5.5];
/// let pop_mean = 5.0;
/// let result = inference::one_sample_ttest(&sample, pop_mean, 0.05).unwrap();
/// println!("t-statistic: {}", result.statistic);
/// println!("p-value: {}", result.pvalue);
/// println!("Significant difference: {}", result.significant);
/// ```
pub fn one_sample_ttest(
    sample: &[f64],
    pop_mean: f64, 
    alpha: f64
) -> Result<TTestResult> {
    if sample.is_empty() {
        return Err(Error::EmptyData("t-test requires data".into()));
    }
    
    let n = sample.len();
    
    if n < 2 {
        return Err(Error::InsufficientData("t-test requires at least 2 data points".into()));
    }
    
    // Calculate sample mean
    let sample_mean = sample.iter().sum::<f64>() / n as f64;
    
    // Calculate sample variance
    let sample_var = sample.iter()
        .map(|&x| (x - sample_mean).powi(2))
        .sum::<f64>() / (n - 1) as f64;
    
    // Calculate standard error
    let std_err = (sample_var / n as f64).sqrt();
    
    // Calculate t-statistic
    let t_stat = (sample_mean - pop_mean) / std_err;
    
    // Degrees of freedom = n - 1
    let df = n - 1;
    
    // Two-tailed test p-value calculation
    let p_value = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));
    
    Ok(TTestResult {
        statistic: t_stat,
        pvalue: p_value,
        significant: p_value < alpha,
        df,
    })
}

/// Perform paired t-test
///
/// # Description
/// Tests if the mean difference between paired samples is zero.
///
/// # Arguments
/// * `sample1` - First sample (before/condition 1)
/// * `sample2` - Second sample (after/condition 2) with values paired with sample1
/// * `alpha` - Significance level (e.g., 0.05 for 95% confidence)
///
/// # Example
/// ```
/// use pandrs::stats::inference;
///
/// // Before and after measurements
/// let before = vec![120.0, 115.0, 118.0, 125.0, 122.0];
/// let after = vec![115.0, 110.0, 112.0, 118.0, 119.0];
///
/// let result = inference::paired_ttest(&before, &after, 0.05).unwrap();
/// println!("t-statistic: {}", result.statistic);
/// println!("p-value: {}", result.pvalue);
/// println!("Significant difference: {}", result.significant);
/// ```
pub fn paired_ttest(
    sample1: &[f64],
    sample2: &[f64],
    alpha: f64
) -> Result<TTestResult> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(Error::EmptyData("Paired t-test requires data".into()));
    }
    
    if sample1.len() != sample2.len() {
        return Err(Error::DimensionMismatch(
            format!("Paired t-test requires equal sample sizes: sample1={}, sample2={}", 
                   sample1.len(), sample2.len())
        ));
    }
    
    let n = sample1.len();
    
    if n < 2 {
        return Err(Error::InsufficientData("Paired t-test requires at least 2 pairs of data points".into()));
    }
    
    // Calculate differences
    let differences: Vec<f64> = sample1.iter()
        .zip(sample2.iter())
        .map(|(&x, &y)| x - y)
        .collect();
    
    // Use one-sample t-test on the differences (H0: mean difference = 0)
    one_sample_ttest(&differences, 0.0, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ttest_equal_means() {
        let sample1 = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        
        let result = ttest_impl(&sample1, &sample2, 0.05, true).unwrap();
        
        // The difference in means is 1.0, but due to large variance it should not be significant
        assert!((result.statistic + 1.0).abs() < 1.0); // t-value should be negative
        assert!(result.pvalue > 0.05); // should not be significant
        assert!(!result.significant);
    }
    
    #[test]
    fn test_ttest_different_means() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        
        let result = ttest_impl(&sample1, &sample2, 0.05, true).unwrap();
        
        // The difference in means is large, should be significant
        assert!(result.statistic < -5.0); // t-value should be a large negative value
        assert!(result.pvalue < 0.05); // should be significant
        assert!(result.significant);
    }
    
    #[test]
    fn test_ttest_welch() {
        // Data with different variances
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![11.0, 13.0, 15.0, 17.0, 19.0];
        
        let result_equal_var = ttest_impl(&sample1, &sample2, 0.05, true).unwrap();
        let result_welch = ttest_impl(&sample1, &sample2, 0.05, false).unwrap();
        
        // Both should be significant, but degrees of freedom and exact statistics should differ
        assert!(result_equal_var.significant);
        assert!(result_welch.significant);
        assert!(result_equal_var.df != result_welch.df);
    }
    
    #[test]
    fn test_ttest_empty() {
        let sample1 = vec![1.0, 2.0, 3.0];
        let sample2: Vec<f64> = vec![];
        
        let result = ttest_impl(&sample1, &sample2, 0.05, true);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_anova_basic() {
        let mut groups = HashMap::new();
        let a_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b_values = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let c_values = vec![3.0, 4.0, 5.0, 6.0, 7.0];
        
        groups.insert("A", a_values.as_slice());
        groups.insert("B", b_values.as_slice());
        groups.insert("C", c_values.as_slice());
        
        let result = anova_impl(&groups, 0.05).unwrap();
        
        // The means of each group are 3, 4, and 5 respectively, with clear differences but large variance
        // F-value should be positive, with a difference of 1.0 between adjacent groups
        assert!(result.f_statistic > 0.0);
        // 15 data points, 3 groups, so degrees of freedom are 2, 12
        assert_eq!(result.df_between, 2);
        assert_eq!(result.df_within, 12);
        assert_eq!(result.df_total, 14);
    }
    
    #[test]
    fn test_anova_significant_difference() {
        let mut groups = HashMap::new();
        let a_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b_values = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let c_values = vec![21.0, 22.0, 23.0, 24.0, 25.0];
        
        groups.insert("A", a_values.as_slice());
        groups.insert("B", b_values.as_slice());
        groups.insert("C", c_values.as_slice());
        
        let result = anova_impl(&groups, 0.05).unwrap();
        
        // With large differences, F-value should be large
        assert!(result.f_statistic > 100.0);
        assert!(result.p_value < 0.05);
        assert!(result.significant);
    }
    
    #[test]
    fn test_mann_whitney_u() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        
        let result = mann_whitney_u_impl(&sample1, &sample2, 0.05).unwrap();
        
        // Completely separated samples should show significant difference
        assert!(result.u_statistic == 0.0); // Minimum U value
        assert!(result.p_value < 0.05);
        assert!(result.significant);
    }
    
    #[test]
    fn test_chi_square() {
        // 2x2 chi-square test (test of independence)
        let observed = vec![
            vec![10.0, 10.0],
            vec![10.0, 20.0]
        ];
        
        let result = chi_square_test_impl(&observed, 0.05).unwrap();
        
        assert!(result.chi2_statistic > 0.0);
        assert_eq!(result.df, 1); // (2-1) * (2-1) = 1
        
        // Check expected frequencies
        assert_eq!(result.expected_freq.len(), 2);
        assert_eq!(result.expected_freq[0].len(), 2);
    }

    #[test]
    fn test_one_sample_ttest() {
        let sample = vec![5.1, 5.3, 5.6, 5.2, 5.5];
        
        // Test sample against population mean of 5.0
        let result_significant = one_sample_ttest(&sample, 5.0, 0.05).unwrap();
        
        // Test sample against population mean equal to sample mean
        let sample_mean = sample.iter().sum::<f64>() / sample.len() as f64;
        let result_nonsignificant = one_sample_ttest(&sample, sample_mean, 0.05).unwrap();
        
        // When testing against sample_mean, t-statistic should be close to 0
        assert!(result_nonsignificant.statistic.abs() < 1e-10);
        assert!(result_nonsignificant.pvalue > 0.99);
        assert!(!result_nonsignificant.significant);
    }

    #[test]
    fn test_paired_ttest() {
        let before = vec![120.0, 115.0, 118.0, 125.0, 122.0];
        let after = vec![115.0, 110.0, 112.0, 118.0, 119.0];
        
        let result = paired_ttest(&before, &after, 0.05).unwrap();
        
        // Systematic decrease should show a positive t-statistic
        // Mean difference is about 5.2
        assert!(result.statistic > 0.0);
        
        // Test with identical samples
        let result_identical = paired_ttest(&before, &before, 0.05).unwrap();
        
        // With identical data, t-statistic should be 0
        assert!(result_identical.statistic.abs() < 1e-10);
        assert!(!result_identical.significant);
    }
}