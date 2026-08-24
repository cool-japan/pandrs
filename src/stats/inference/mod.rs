//! Inferential statistics and hypothesis testing module

use crate::error::{Error, Result};
use crate::stats::{AnovaResult, ChiSquareResult, MannWhitneyResult, TTestResult};
use std::collections::HashMap;

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
        return Err(Error::InsufficientData(
            "t-test requires at least 2 data points in each group".into(),
        ));
    }

    // Calculate means
    let mean1 = sample1.iter().sum::<f64>() / n1 as f64;
    let mean2 = sample2.iter().sum::<f64>() / n2 as f64;

    // Calculate variances
    let var1 = sample1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1) as f64;

    let var2 = sample2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

    let (t_stat, df) = if equal_var {
        // Equal variance assumption t-statistic
        let pooled_var = ((n1 - 1) as f64 * var1 + (n2 - 1) as f64 * var2) / (n1 + n2 - 2) as f64;
        let std_err = (pooled_var * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
        let t_value = (mean1 - mean2) / std_err;
        (t_value, n1 + n2 - 2)
    } else {
        // Welch's t-test (no equal variance assumption)
        let std_err = (var1 / n1 as f64 + var2 / n2 as f64).sqrt();
        let t_value = (mean1 - mean2) / std_err;

        // Welch-Satterthwaite approximation for degrees of freedom
        let df_num = (var1 / n1 as f64 + var2 / n2 as f64).powi(2);
        let df_denom = (var1 / n1 as f64).powi(2) / (n1 - 1) as f64
            + (var2 / n2 as f64).powi(2) / (n2 - 1) as f64;
        let df_welch = df_num / df_denom;
        (t_value, df_welch.floor() as usize)
    };

    // Two-tailed test p-value, via `special::student_t_two_sided_p`'s direct
    // incomplete-beta evaluation — never `1.0 - student_t_cdf(...)`, which
    // double-cancels to exactly `0.0` for any strongly-significant (large
    // |t|) result (see `special::student_t_sf`'s doc comment for the exact
    // mechanism).
    let p_value = crate::stats::special::student_t_two_sided_p(t_stat, df as f64);

    Ok(TTestResult {
        statistic: t_stat,
        pvalue: p_value,
        significant: p_value < alpha,
        df,
    })
}

/// Convert a chi-square statistic to its upper-tail p-value `P(X > chi2 | df)`.
///
/// The previous closed-form was numerically divergent (used `x.exp()` where
/// `(-x).exp()` was meant) and returned ≈0 for every input, so every
/// chi-square test reported "significant". Now routed through the correct
/// regularized incomplete gamma in `stats::special`.
fn chi2_to_pvalue(chi2: f64, df: usize) -> f64 {
    crate::stats::special::chi2_sf(chi2, df as f64)
}

/// Implementation for one-way ANOVA
pub(crate) fn anova_impl(groups: &HashMap<&str, &[f64]>, alpha: f64) -> Result<AnovaResult> {
    // Check number of groups and sample size for each group
    if groups.is_empty() {
        return Err(Error::EmptyData("ANOVA requires at least one group".into()));
    }

    if groups.len() < 2 {
        return Err(Error::InsufficientData(
            "ANOVA requires at least two groups".into(),
        ));
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

    // Calculate p-value directly via `special::f_sf` — never
    // `1.0 - f_cdf(...)`, which silently reports `p_value = 0.0` for any
    // strongly-significant F-statistic (see `special::f_sf`'s doc comment
    // for the exact cancellation mechanism this avoids).
    let p_value = crate::stats::special::f_sf(f_statistic, df_between as f64, df_within as f64);

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
    alpha: f64,
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

    let n_total = (n1 + n2) as f64;
    let mean_u = (n1 * n2) as f64 / 2.0;

    // Tie correction (Hollander & Wolfe): σ² = n1·n2/12 · [(N+1) −
    // Σ(t_j³ − t_j) / (N(N−1))], where t_j is each tied-value group's size.
    // The previous `(n1·n2·(N+1))/12` no-tie formula overstates the
    // variance — and understates significance — whenever the combined
    // sample has ties.
    let tie_term: f64 = {
        let mut sorted: Vec<f64> = combined.iter().map(|(v, _, _)| *v).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut sum = 0.0;
        let mut i = 0;
        while i < sorted.len() {
            let mut j = i;
            while j < sorted.len() && (sorted[j] - sorted[i]).abs() < f64::EPSILON {
                j += 1;
            }
            let t = (j - i) as f64;
            sum += t * t * t - t;
            i = j;
        }
        sum
    };
    let var_u =
        (n1 * n2) as f64 / 12.0 * ((n_total + 1.0) - tie_term / (n_total * (n_total - 1.0)));
    let std_u = var_u.sqrt();

    // Two-sided normal approximation with continuity correction, matching
    // the reference formula in `nonparametric::mann_whitney_u_test` (and
    // `scipy.stats.mannwhitneyu`'s asymptotic method): test the larger of
    // U1/U2 against its null mean via the survival function, subtracting
    // the 0.5 continuity correction — rather than the previous
    // uncorrected `min(u1, u2)` against a non-tie-adjusted standard
    // deviation.
    let u_statistic = u1.min(u2);
    let z = (u1.max(u2) - mean_u - 0.5) / std_u;
    let p_value = (2.0 * crate::stats::special::normal_sf(z)).clamp(0.0, 1.0);

    Ok(MannWhitneyResult {
        u_statistic,
        p_value,
        significant: p_value < alpha,
    })
}

/// Implementation for chi-square test
pub(crate) fn chi_square_test_impl(observed: &[Vec<f64>], alpha: f64) -> Result<ChiSquareResult> {
    // Validate observed data
    if observed.is_empty() {
        return Err(Error::EmptyData(
            "Chi-square test requires observed data".into(),
        ));
    }

    let rows = observed.len();
    if rows < 2 {
        return Err(Error::InsufficientData(
            "Chi-square test requires at least 2 rows of data".into(),
        ));
    }

    let cols = observed[0].len();
    if cols < 2 {
        return Err(Error::InsufficientData(
            "Chi-square test requires at least 2 columns of data".into(),
        ));
    }

    // Ensure all rows have the same number of columns
    for row in observed.iter() {
        if row.len() != cols {
            return Err(Error::InvalidInput(
                "All rows must have the same number of columns".into(),
            ));
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
                return Err(Error::InvalidInput(
                    "Observed values must not be negative".into(),
                ));
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

/// Chi-square test with pre-calculated statistic
pub fn chi_square_test_with_statistic(
    chi2: f64,
    df: usize,
    alpha: f64,
    expected: Vec<Vec<f64>>,
) -> Result<ChiSquareResult> {
    // Calculate p-value
    let p_value = chi2_to_pvalue(chi2, df);

    Ok(ChiSquareResult {
        chi2_statistic: chi2,
        p_value,
        df,
        significant: p_value < alpha,
        expected_freq: expected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttest_equal_means() {
        let sample1 = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result = ttest_impl(&sample1, &sample2, 0.05, true).expect("operation should succeed");

        // The difference in means is 1.0, but due to large variance it should not be significant
        assert!((result.statistic + 1.0).abs() < 1.0); // t-value should be negative
        assert!(result.pvalue > 0.05); // should not be significant
        assert!(!result.significant);
    }

    #[test]
    fn test_ttest_different_means() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![11.0, 12.0, 13.0, 14.0, 15.0];

        let result = ttest_impl(&sample1, &sample2, 0.05, true).expect("operation should succeed");

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

        let result_equal_var =
            ttest_impl(&sample1, &sample2, 0.05, true).expect("operation should succeed");
        let result_welch =
            ttest_impl(&sample1, &sample2, 0.05, false).expect("operation should succeed");

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

        let result = anova_impl(&groups, 0.05).expect("operation should succeed");

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

        let result = anova_impl(&groups, 0.05).expect("operation should succeed");

        // With large differences, F-value should be large
        assert!(result.f_statistic > 100.0);
        assert!(result.p_value < 0.05);
        assert!(result.significant);
    }

    #[test]
    fn test_mann_whitney_u() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result =
            mann_whitney_u_impl(&sample1, &sample2, 0.05).expect("operation should succeed");

        // Completely separated samples should show significant difference
        assert!(result.u_statistic == 0.0); // Minimum U value
        assert!(result.p_value < 0.05);
        assert!(result.significant);
    }

    #[test]
    fn test_chi_square() {
        // 2x2 chi-square test (test of independence)
        let observed = vec![vec![10.0, 10.0], vec![10.0, 20.0]];

        let result = chi_square_test_impl(&observed, 0.05).expect("operation should succeed");

        assert!(result.chi2_statistic > 0.0);
        assert_eq!(result.df, 1); // (2-1) * (2-1) = 1

        // Check expected frequencies
        assert_eq!(result.expected_freq.len(), 2);
        assert_eq!(result.expected_freq[0].len(), 2);
    }
}
