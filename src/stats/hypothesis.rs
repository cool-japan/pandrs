//! Comprehensive hypothesis testing framework
//!
//! This module provides a complete suite of statistical hypothesis tests including
//! parametric and non-parametric tests, effect size calculations, and multiple
//! comparison corrections for robust statistical analysis.

use crate::core::error::{Error, Result};
use crate::stats::distributions::{ChiSquared, Distribution, FDistribution, TDistribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Statistical hypothesis test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value of the test
    pub p_value: f64,
    /// Degrees of freedom (if applicable)
    pub degrees_of_freedom: Option<f64>,
    /// Critical value at α = 0.05
    pub critical_value: Option<f64>,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
    /// Effect size interpretation
    pub effect_size_interpretation: Option<String>,
    /// Confidence interval for the effect
    pub confidence_interval: Option<(f64, f64)>,
    /// Test description
    pub test_name: String,
    /// Alternative hypothesis
    pub alternative: AlternativeHypothesis,
    /// Whether to reject null hypothesis at α = 0.05
    pub reject_null: bool,
    /// Additional test-specific information
    pub additional_info: HashMap<String, f64>,
}

/// Alternative hypothesis specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlternativeHypothesis {
    /// Two-sided test (≠)
    TwoSided,
    /// Greater than test (>)
    Greater,
    /// Less than test (<)
    Less,
}

/// Effect size measures
#[derive(Debug, Clone)]
pub enum EffectSize {
    /// Cohen's d for t-tests
    CohensD(f64),
    /// Pearson's r for correlation
    PearsonR(f64),
    /// Eta squared for ANOVA
    EtaSquared(f64),
    /// Partial eta squared
    PartialEtaSquared(f64),
    /// Omega squared
    OmegaSquared(f64),
    /// Cramer's V for chi-square
    CramersV(f64),
    /// Glass's delta
    GlassDelta(f64),
    /// Hedges' g
    HedgesG(f64),
}

impl EffectSize {
    /// Get the numeric value of the effect size
    pub fn value(&self) -> f64 {
        match self {
            EffectSize::CohensD(d)
            | EffectSize::PearsonR(d)
            | EffectSize::EtaSquared(d)
            | EffectSize::PartialEtaSquared(d)
            | EffectSize::OmegaSquared(d)
            | EffectSize::CramersV(d)
            | EffectSize::GlassDelta(d)
            | EffectSize::HedgesG(d) => *d,
        }
    }

    /// Get interpretation of effect size magnitude
    pub fn interpretation(&self) -> String {
        let val = self.value().abs();
        match self {
            EffectSize::CohensD(_) | EffectSize::GlassDelta(_) | EffectSize::HedgesG(_) => {
                if val < 0.2 {
                    "Negligible".to_string()
                } else if val < 0.5 {
                    "Small".to_string()
                } else if val < 0.8 {
                    "Medium".to_string()
                } else {
                    "Large".to_string()
                }
            }
            EffectSize::PearsonR(_) => {
                if val < 0.1 {
                    "Negligible".to_string()
                } else if val < 0.3 {
                    "Small".to_string()
                } else if val < 0.5 {
                    "Medium".to_string()
                } else {
                    "Large".to_string()
                }
            }
            EffectSize::EtaSquared(_)
            | EffectSize::PartialEtaSquared(_)
            | EffectSize::OmegaSquared(_) => {
                if val < 0.01 {
                    "Small".to_string()
                } else if val < 0.06 {
                    "Medium".to_string()
                } else {
                    "Large".to_string()
                }
            }
            EffectSize::CramersV(_) => {
                if val < 0.1 {
                    "Negligible".to_string()
                } else if val < 0.3 {
                    "Small".to_string()
                } else if val < 0.5 {
                    "Medium".to_string()
                } else {
                    "Large".to_string()
                }
            }
        }
    }
}

/// One-sample t-test
pub fn one_sample_ttest(
    data: &[f64],
    hypothesized_mean: f64,
    alternative: AlternativeHypothesis,
) -> Result<TestResult> {
    if data.is_empty() {
        return Err(Error::InvalidValue("Data cannot be empty".into()));
    }

    let n = data.len() as f64;
    let sample_mean = data.iter().sum::<f64>() / n;
    let sample_std = {
        let variance = data.iter().map(|&x| (x - sample_mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    };

    let standard_error = sample_std / n.sqrt();
    let t_statistic = (sample_mean - hypothesized_mean) / standard_error;
    let df = n - 1.0;

    let t_dist = TDistribution::new(df)?;
    // `TwoSided`/`Greater` route through `stats::special`'s direct survival
    // functions, never `1.0 - t_dist.cdf(...)`: `cdf` itself already returns
    // `1.0 - half_tail` internally for a nonnegative argument (see
    // `special::student_t_cdf`'s doc comment), so subtracting that
    // already-rounded result from `1.0` *again* here double-cancels and
    // silently reports `p_value = 0.0` for any strongly-significant (large
    // |t|) result — see `special::student_t_sf`'s doc comment for the exact
    // mechanism. `Less` needs no such fix: `cdf` at a negative argument is
    // already a direct, well-conditioned tail computation.
    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => {
            crate::stats::special::student_t_two_sided_p(t_statistic, df)
        }
        AlternativeHypothesis::Greater => crate::stats::special::student_t_sf(t_statistic, df),
        AlternativeHypothesis::Less => t_dist.cdf(t_statistic),
    };

    let critical_value = match alternative {
        AlternativeHypothesis::TwoSided => t_dist.inverse_cdf(0.975),
        AlternativeHypothesis::Greater => t_dist.inverse_cdf(0.95),
        AlternativeHypothesis::Less => t_dist.inverse_cdf(0.05),
    };

    // Calculate Cohen's d effect size
    let cohens_d = (sample_mean - hypothesized_mean) / sample_std;
    let effect_size = EffectSize::CohensD(cohens_d);

    // Confidence interval for the mean difference
    let margin_of_error = critical_value * standard_error;
    let ci = (
        (sample_mean - hypothesized_mean) - margin_of_error,
        (sample_mean - hypothesized_mean) + margin_of_error,
    );

    let mut additional_info = HashMap::new();
    additional_info.insert("sample_mean".to_string(), sample_mean);
    additional_info.insert("sample_std".to_string(), sample_std);
    additional_info.insert("standard_error".to_string(), standard_error);
    additional_info.insert("hypothesized_mean".to_string(), hypothesized_mean);

    Ok(TestResult {
        statistic: t_statistic,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: Some(critical_value),
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: Some(ci),
        test_name: "One-sample t-test".to_string(),
        alternative,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Independent samples t-test (Welch's t-test)
pub fn independent_ttest(
    group1: &[f64],
    group2: &[f64],
    alternative: AlternativeHypothesis,
    equal_variances: bool,
) -> Result<TestResult> {
    if group1.is_empty() || group2.is_empty() {
        return Err(Error::InvalidValue("Both groups must contain data".into()));
    }

    let n1 = group1.len() as f64;
    let n2 = group2.len() as f64;

    let mean1 = group1.iter().sum::<f64>() / n1;
    let mean2 = group2.iter().sum::<f64>() / n2;

    let var1 = group1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = group2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let (t_statistic, df, standard_error) = if equal_variances {
        // Pooled variance t-test
        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        let se = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
        let t = (mean1 - mean2) / se;
        let degrees_freedom = n1 + n2 - 2.0;
        (t, degrees_freedom, se)
    } else {
        // Welch's t-test (unequal variances)
        let se = (var1 / n1 + var2 / n2).sqrt();
        let t = (mean1 - mean2) / se;

        // Welch-Satterthwaite equation for degrees of freedom
        let numerator = (var1 / n1 + var2 / n2).powi(2);
        let denominator = (var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0);
        let degrees_freedom = numerator / denominator;
        (t, degrees_freedom, se)
    };

    let t_dist = TDistribution::new(df)?;
    // See `one_sample_ttest`'s doc comment above for why `TwoSided`/`Greater`
    // must route through `special`'s direct survival functions rather than
    // `1.0 - t_dist.cdf(...)`.
    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => {
            crate::stats::special::student_t_two_sided_p(t_statistic, df)
        }
        AlternativeHypothesis::Greater => crate::stats::special::student_t_sf(t_statistic, df),
        AlternativeHypothesis::Less => t_dist.cdf(t_statistic),
    };

    let critical_value = match alternative {
        AlternativeHypothesis::TwoSided => t_dist.inverse_cdf(0.975),
        AlternativeHypothesis::Greater => t_dist.inverse_cdf(0.95),
        AlternativeHypothesis::Less => t_dist.inverse_cdf(0.05),
    };

    // Calculate Cohen's d effect size
    let pooled_std = if equal_variances {
        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
        pooled_var.sqrt()
    } else {
        ((var1 + var2) / 2.0).sqrt()
    };

    let cohens_d = (mean1 - mean2) / pooled_std;
    let effect_size = EffectSize::CohensD(cohens_d);

    // Confidence interval for the mean difference
    let margin_of_error = critical_value * standard_error;
    let ci = (
        (mean1 - mean2) - margin_of_error,
        (mean1 - mean2) + margin_of_error,
    );

    let mut additional_info = HashMap::new();
    additional_info.insert("mean1".to_string(), mean1);
    additional_info.insert("mean2".to_string(), mean2);
    additional_info.insert("var1".to_string(), var1);
    additional_info.insert("var2".to_string(), var2);
    additional_info.insert("n1".to_string(), n1);
    additional_info.insert("n2".to_string(), n2);
    additional_info.insert("pooled_std".to_string(), pooled_std);

    let test_name = if equal_variances {
        "Independent samples t-test (equal variances)".to_string()
    } else {
        "Welch's t-test (unequal variances)".to_string()
    };

    Ok(TestResult {
        statistic: t_statistic,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: Some(critical_value),
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: Some(ci),
        test_name,
        alternative,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Paired samples t-test
pub fn paired_ttest(
    before: &[f64],
    after: &[f64],
    alternative: AlternativeHypothesis,
) -> Result<TestResult> {
    if before.len() != after.len() {
        return Err(Error::DimensionMismatch(
            "Before and after groups must have same length".into(),
        ));
    }

    if before.is_empty() {
        return Err(Error::InvalidValue("Data cannot be empty".into()));
    }

    // Calculate differences
    let differences: Vec<f64> = before
        .iter()
        .zip(after.iter())
        .map(|(&b, &a)| b - a)
        .collect();

    // Perform one-sample t-test on differences against 0
    one_sample_ttest(&differences, 0.0, alternative).map(|mut result| {
        result.test_name = "Paired samples t-test".to_string();

        // Add paired-specific information
        result.additional_info.insert(
            "mean_before".to_string(),
            before.iter().sum::<f64>() / before.len() as f64,
        );
        result.additional_info.insert(
            "mean_after".to_string(),
            after.iter().sum::<f64>() / after.len() as f64,
        );
        result.additional_info.insert(
            "mean_difference".to_string(),
            differences.iter().sum::<f64>() / differences.len() as f64,
        );

        result
    })
}

/// One-way ANOVA
pub fn one_way_anova(groups: &[&[f64]]) -> Result<TestResult> {
    if groups.is_empty() {
        return Err(Error::InvalidValue("At least one group is required".into()));
    }

    if groups.len() < 2 {
        return Err(Error::InvalidValue(
            "At least two groups are required for ANOVA".into(),
        ));
    }

    // Check that all groups have data
    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(Error::InvalidValue(format!("Group {} is empty", i)));
        }
    }

    let k = groups.len() as f64; // number of groups
    let n_total: usize = groups.iter().map(|g| g.len()).sum(); // total sample size

    // Calculate group means and overall mean
    let group_means: Vec<f64> = groups
        .iter()
        .map(|group| group.iter().sum::<f64>() / group.len() as f64)
        .collect();

    let overall_mean = groups.iter().flat_map(|group| group.iter()).sum::<f64>() / n_total as f64;

    // Calculate sum of squares
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;

    for (i, group) in groups.iter().enumerate() {
        let group_mean = group_means[i];
        let n_group = group.len() as f64;

        // Between-group sum of squares
        ss_between += n_group * (group_mean - overall_mean).powi(2);

        // Within-group sum of squares
        for &value in group.iter() {
            ss_within += (value - group_mean).powi(2);
        }
    }

    let ss_total = ss_between + ss_within;

    // Degrees of freedom
    let df_between = k - 1.0;
    let df_within = n_total as f64 - k;
    let _df_total = n_total as f64 - 1.0;

    // Mean squares
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    // F-statistic
    let f_statistic = ms_between / ms_within;

    // P-value, via `special::f_sf` directly — not `1.0 - f_dist.cdf(...)`.
    // `cdf`'s own `betai` evaluation already collapses to a single `f64`
    // indistinguishable from `1.0` once the true tail probability is small
    // enough (the same cancellation `special::f_sf`'s doc comment describes
    // for `f_sf(1e4, 10, 10)`), so re-subtracting it from `1.0` here would
    // silently report `p_value = 0.0` for any highly significant ANOVA
    // result instead of the true (still nonzero) tail probability.
    let f_dist = FDistribution::new(df_between, df_within)?;
    let p_value = crate::stats::special::f_sf(f_statistic, df_between, df_within);

    // Critical value
    let critical_value = f_dist.inverse_cdf(0.95);

    // Effect sizes
    let eta_squared = ss_between / ss_total;
    let omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within);
    let effect_size = EffectSize::EtaSquared(eta_squared);

    let mut additional_info = HashMap::new();
    additional_info.insert("ss_between".to_string(), ss_between);
    additional_info.insert("ss_within".to_string(), ss_within);
    additional_info.insert("ss_total".to_string(), ss_total);
    additional_info.insert("ms_between".to_string(), ms_between);
    additional_info.insert("ms_within".to_string(), ms_within);
    additional_info.insert("df_between".to_string(), df_between);
    additional_info.insert("df_within".to_string(), df_within);
    additional_info.insert("eta_squared".to_string(), eta_squared);
    additional_info.insert("omega_squared".to_string(), omega_squared);
    additional_info.insert("n_groups".to_string(), k);
    additional_info.insert("n_total".to_string(), n_total as f64);

    Ok(TestResult {
        statistic: f_statistic,
        p_value,
        degrees_of_freedom: Some(df_between), // Primary df
        critical_value: Some(critical_value),
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: None, // Not typically reported for ANOVA
        test_name: "One-way ANOVA".to_string(),
        alternative: AlternativeHypothesis::Greater, // F-test is always one-sided
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Chi-square test of independence
pub fn chi_square_independence(observed: &[Vec<f64>]) -> Result<TestResult> {
    if observed.is_empty() || observed[0].is_empty() {
        return Err(Error::InvalidValue(
            "Contingency table cannot be empty".into(),
        ));
    }

    let rows = observed.len();
    let cols = observed[0].len();

    // Check that all rows have the same length
    for row in observed.iter() {
        if row.len() != cols {
            return Err(Error::DimensionMismatch(
                "All rows must have the same length".into(),
            ));
        }
    }

    // Calculate row and column totals
    let mut row_totals = vec![0.0; rows];
    let mut col_totals = vec![0.0; cols];
    let mut grand_total = 0.0;

    for (i, row) in observed.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value < 0.0 {
                return Err(Error::InvalidValue(
                    "All frequencies must be non-negative".into(),
                ));
            }
            row_totals[i] += value;
            col_totals[j] += value;
            grand_total += value;
        }
    }

    if grand_total == 0.0 {
        return Err(Error::InvalidValue("Total frequency cannot be zero".into()));
    }

    // Calculate expected frequencies and chi-square statistic
    let mut chi_square = 0.0;
    let mut min_expected = f64::INFINITY;

    for i in 0..rows {
        for j in 0..cols {
            let expected = (row_totals[i] * col_totals[j]) / grand_total;
            if expected < min_expected {
                min_expected = expected;
            }

            if expected > 0.0 {
                chi_square += (observed[i][j] - expected).powi(2) / expected;
            }
        }
    }

    // Degrees of freedom
    let df = (rows - 1) * (cols - 1);

    // P-value, via `special::chi2_sf` directly — see `one_way_anova`'s doc
    // comment above for why `1.0 - chi_sq_dist.cdf(...)` would silently
    // report `p_value = 0.0` for a strongly-significant chi-square statistic.
    let chi_sq_dist = ChiSquared::new(df as f64)?;
    let p_value = crate::stats::special::chi2_sf(chi_square, df as f64);

    // Critical value
    let critical_value = chi_sq_dist.inverse_cdf(0.95);

    // Cramer's V effect size
    let cramers_v = (chi_square / (grand_total * ((rows.min(cols) - 1) as f64))).sqrt();
    let effect_size = EffectSize::CramersV(cramers_v);

    let mut additional_info = HashMap::new();
    additional_info.insert("degrees_of_freedom".to_string(), df as f64);
    additional_info.insert("grand_total".to_string(), grand_total);
    additional_info.insert("min_expected_frequency".to_string(), min_expected);
    additional_info.insert("cramers_v".to_string(), cramers_v);
    additional_info.insert("n_rows".to_string(), rows as f64);
    additional_info.insert("n_cols".to_string(), cols as f64);

    // Add warning if expected frequencies are too low
    if min_expected < 5.0 {
        additional_info.insert("warning_low_expected".to_string(), 1.0);
    }

    Ok(TestResult {
        statistic: chi_square,
        p_value,
        degrees_of_freedom: Some(df as f64),
        critical_value: Some(critical_value),
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: None,
        test_name: "Chi-square test of independence".to_string(),
        alternative: AlternativeHypothesis::Greater,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Pearson correlation test
pub fn correlation_test(
    x: &[f64],
    y: &[f64],
    alternative: AlternativeHypothesis,
) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::DimensionMismatch(
            "X and Y must have the same length".into(),
        ));
    }

    if x.len() < 3 {
        return Err(Error::InvalidValue(
            "At least 3 data points are required".into(),
        ));
    }

    let n = x.len() as f64;

    // Calculate correlation coefficient
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
        return Err(Error::InvalidValue(
            "Cannot compute correlation: zero variance".into(),
        ));
    }

    let r = sum_xy / denominator;

    // t-statistic for testing correlation
    let denominator_t = 1.0 - r.powi(2);
    let t_statistic = if denominator_t < 1e-10 {
        // For perfect correlation, t-statistic approaches infinity
        if r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        r * ((n - 2.0) / denominator_t).sqrt()
    };
    let df = n - 2.0;

    let t_dist = TDistribution::new(df)?;
    let p_value = if t_statistic.is_infinite() {
        // For perfect correlation, p-value is essentially 0
        match alternative {
            AlternativeHypothesis::TwoSided => 0.0,
            AlternativeHypothesis::Greater => {
                if t_statistic > 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
            AlternativeHypothesis::Less => {
                if t_statistic < 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    } else {
        // See `one_sample_ttest`'s doc comment for why `TwoSided`/`Greater`
        // must route through `special`'s direct survival functions rather
        // than `1.0 - t_dist.cdf(...)`.
        match alternative {
            AlternativeHypothesis::TwoSided => {
                crate::stats::special::student_t_two_sided_p(t_statistic, df)
            }
            AlternativeHypothesis::Greater => crate::stats::special::student_t_sf(t_statistic, df),
            AlternativeHypothesis::Less => t_dist.cdf(t_statistic),
        }
    };

    let critical_value = match alternative {
        AlternativeHypothesis::TwoSided => t_dist.inverse_cdf(0.975),
        AlternativeHypothesis::Greater => t_dist.inverse_cdf(0.95),
        AlternativeHypothesis::Less => t_dist.inverse_cdf(0.05),
    };

    let effect_size = EffectSize::PearsonR(r);

    // Fisher's z-transformation for confidence interval
    let ci = if r.abs() >= 0.999999 {
        // For near-perfect correlation, CI is very narrow around r
        let margin = 1e-6;
        let r_clamped = r.clamp(-0.999999, 0.999999);
        (r_clamped - margin, r_clamped + margin)
    } else {
        let z_r = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
        let se_z = 1.0 / (n - 3.0).sqrt();
        let z_critical = 1.96; // for 95% CI

        let z_lower = z_r - z_critical * se_z;
        let z_upper = z_r + z_critical * se_z;

        let r_lower = z_lower.tanh();
        let r_upper = z_upper.tanh();

        (r_lower, r_upper)
    };

    let mut additional_info = HashMap::new();
    additional_info.insert("correlation".to_string(), r);
    additional_info.insert("n".to_string(), n);
    additional_info.insert("mean_x".to_string(), mean_x);
    additional_info.insert("mean_y".to_string(), mean_y);
    additional_info.insert("r_squared".to_string(), r.powi(2));

    Ok(TestResult {
        statistic: t_statistic,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: Some(critical_value),
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: Some(ci),
        test_name: "Pearson correlation test".to_string(),
        alternative,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Shapiro-Wilk test for normality
pub fn shapiro_wilk_test(data: &[f64]) -> Result<TestResult> {
    let n = data.len();

    if n < 3 {
        return Err(Error::InvalidValue(
            "At least 3 observations required for Shapiro-Wilk test".into(),
        ));
    }

    if n > 5000 {
        return Err(Error::InvalidValue(
            "Shapiro-Wilk test not reliable for n > 5000".into(),
        ));
    }

    // Sort the data.
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let an = n as f64;
    let mean = sorted_data.iter().sum::<f64>() / an;
    let ss: f64 = sorted_data.iter().map(|x| (x - mean).powi(2)).sum();
    let std_dev = (ss / (n - 1) as f64).sqrt();

    if ss <= 0.0 {
        // Degenerate case: all values identical → W = 1, p = 1.
        let mut additional_info = HashMap::new();
        additional_info.insert("n".to_string(), an);
        additional_info.insert("mean".to_string(), mean);
        additional_info.insert("std_dev".to_string(), std_dev);
        return Ok(TestResult {
            statistic: 1.0,
            p_value: 1.0,
            degrees_of_freedom: None,
            critical_value: Some(0.95),
            effect_size: None,
            effect_size_interpretation: None,
            confidence_interval: None,
            test_name: "Shapiro-Wilk normality test (approximation)".to_string(),
            alternative: AlternativeHypothesis::Greater,
            reject_null: false,
            additional_info,
        });
    }

    // Royston (1992) AS R94: compute normal-score weights.
    let n2 = n / 2;
    let mut m = vec![0.0_f64; n2 + 1]; // 1-indexed; m[0] unused
    let mut summ2 = 0.0;
    for i in 1..=n2 {
        let mi = sw_inv_normal_cdf((i as f64 - 0.375) / (an + 0.25));
        m[i] = mi;
        summ2 += mi * mi;
    }
    summ2 *= 2.0;
    let ssumm2 = summ2.sqrt();
    let rsn = 1.0 / an.sqrt();

    // Polynomial corrections for the two extreme weights.
    let c1 = [0.0, 0.221157, -0.147981, -2.071190, 4.434685, -2.706056];
    let c2 = [0.0, 0.042981, -0.293762, -1.752461, 5.682633, -3.582633];

    let mut a = vec![0.0_f64; n2 + 1];
    let a1 = sw_poly(&c1, rsn) - m[1] / ssumm2;
    let (i1, fac) = if n > 5 {
        let a2 = sw_poly(&c2, rsn) - m[2] / ssumm2;
        a[2] = a2;
        let f = ((summ2 - 2.0 * m[1] * m[1] - 2.0 * m[2] * m[2])
            / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2))
            .sqrt();
        (3_usize, f)
    } else {
        let f = ((summ2 - 2.0 * m[1] * m[1]) / (1.0 - 2.0 * a1 * a1)).sqrt();
        (2_usize, f)
    };
    a[1] = a1;
    if fac.is_finite() && fac > 0.0 {
        for i in i1..=n2 {
            a[i] = -m[i] / fac;
        }
    }

    // W = (Σ aᵢ (x₍n+1−i₎ − x₍ᵢ₎))² / SS
    let mut numerator = 0.0;
    for i in 1..=n2 {
        numerator += a[i] * (sorted_data[n - i] - sorted_data[i - 1]);
    }
    let w_statistic = (numerator * numerator / ss).min(1.0);

    // Royston's p-value transform (multi-branch, AS R94).
    //
    // Both branches clamp only to the valid `[0, 1]` probability range, not
    // to an arbitrary `1e-10` floor (as this used to): the `n == 3` branch
    // is an *exact* closed form (not an asymptotic approximation), for
    // which a `W` at its theoretical minimum for `n = 3` (`W = 0.75`)
    // legitimately computes `p = 0.0` exactly, and the general branch's
    // `normal_sf` is itself accurate deep into the tail (see `special.rs`),
    // so there is no numerical-safety reason to launder a genuinely tiny
    // (or exactly zero) computed p-value into a fabricated `1e-10` —
    // exactly the anti-pattern already removed from the Kolmogorov-Smirnov
    // test's old `.max(0.001)` floor in `nonparametric.rs`.
    let p_value = if n == 3 {
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = (0.75_f64).sqrt().asin();
        (pi6 * (w_statistic.sqrt().asin() - stqr)).clamp(0.0, 1.0)
    } else {
        let w1 = 1.0 - w_statistic;
        let z = if n <= 11 {
            let gamma = -2.273 + 0.459 * an;
            let mu = sw_poly(&[0.5440, -0.39978, 0.025054, -6.714e-4], an);
            let sigma = sw_poly(&[1.3822, -0.77857, 0.062767, -0.0020322], an).exp();
            let y = -(gamma - w1.ln()).ln();
            (y - mu) / sigma
        } else {
            let ln_an = an.ln();
            let mu = sw_poly(&[-1.5861, -0.31082, -0.083751, 0.0038915], ln_an);
            let sigma = sw_poly(&[-0.4803, -0.082676, 0.0030302], ln_an).exp();
            let y = w1.ln();
            (y - mu) / sigma
        };
        crate::stats::special::normal_sf(z).clamp(0.0, 1.0)
    };

    let mut additional_info = HashMap::new();
    additional_info.insert("n".to_string(), an);
    additional_info.insert("mean".to_string(), mean);
    additional_info.insert("std_dev".to_string(), std_dev);
    additional_info.insert("note".to_string(), 1.0_f64);

    Ok(TestResult {
        statistic: w_statistic,
        p_value,
        degrees_of_freedom: None,
        critical_value: Some(0.95),
        effect_size: None,
        effect_size_interpretation: None,
        confidence_interval: None,
        test_name: "Shapiro-Wilk normality test (approximation)".to_string(),
        alternative: AlternativeHypothesis::Greater,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

// ─── Royston (1992) Shapiro-Wilk helpers ────────────────────────────────────

/// Horner evaluation of a polynomial `c[0] + c[1]·x + c[2]·x² + …`.
fn sw_poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Inverse standard-normal CDF (probit) via Acklam's rational approximation
/// (relative error < 1.2e-9). Used for the Shapiro-Wilk normal scores.
fn sw_inv_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;

    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Multiple comparison correction methods
#[derive(Debug, Clone)]
pub enum MultipleComparisonCorrection {
    /// No correction
    None,
    /// Bonferroni correction
    Bonferroni,
    /// Holm-Bonferroni method
    HolmBonferroni,
    /// Benjamini-Hochberg (FDR)
    BenjaminiHochberg,
    /// Benjamini-Yekutieli (FDR under dependence)
    BenjaminiYekutieli,
}

/// Apply multiple comparison correction to p-values
pub fn adjust_p_values(p_values: &[f64], method: MultipleComparisonCorrection) -> Result<Vec<f64>> {
    if p_values.is_empty() {
        return Ok(Vec::new());
    }

    let n = p_values.len();

    match method {
        MultipleComparisonCorrection::None => Ok(p_values.to_vec()),

        MultipleComparisonCorrection::Bonferroni => {
            Ok(p_values.iter().map(|&p| (p * n as f64).min(1.0)).collect())
        }

        MultipleComparisonCorrection::HolmBonferroni => {
            let mut indexed_p: Vec<(usize, f64)> =
                p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut adjusted = vec![0.0; n];
            let mut max_adj = 0.0;

            for (rank, (original_index, p)) in indexed_p.iter().enumerate() {
                let multiplier = n - rank;
                let adj_p = (p * multiplier as f64).min(1.0);
                let adj_p = adj_p.max(max_adj);
                adjusted[*original_index] = adj_p;
                max_adj = adj_p;
            }

            Ok(adjusted)
        }

        MultipleComparisonCorrection::BenjaminiHochberg => {
            let mut indexed_p: Vec<(usize, f64)> =
                p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed_p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); // Descending order

            let mut adjusted = vec![0.0; n];
            let mut min_adj = 1.0;

            for (rank, (original_index, p)) in indexed_p.iter().enumerate() {
                let adj_p = (p * n as f64 / (n - rank) as f64).min(1.0);
                let adj_p = adj_p.min(min_adj);
                adjusted[*original_index] = adj_p;
                min_adj = adj_p;
            }

            Ok(adjusted)
        }

        MultipleComparisonCorrection::BenjaminiYekutieli => {
            // Similar to BH but with additional correction factor
            let correction_factor: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();

            let mut indexed_p: Vec<(usize, f64)> =
                p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed_p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut adjusted = vec![0.0; n];
            let mut min_adj = 1.0;

            for (rank, (original_index, p)) in indexed_p.iter().enumerate() {
                let adj_p = (p * n as f64 * correction_factor / (n - rank) as f64).min(1.0);
                let adj_p = adj_p.min(min_adj);
                adjusted[*original_index] = adj_p;
                min_adj = adj_p;
            }

            Ok(adjusted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_sample_ttest() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = one_sample_ttest(&data, 3.0, AlternativeHypothesis::TwoSided)
            .expect("operation should succeed");

        assert_eq!(result.test_name, "One-sample t-test");
        assert!(result.p_value > 0.05); // Should not reject null hypothesis
        assert!(!result.reject_null);
    }

    #[test]
    fn test_independent_ttest() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = independent_ttest(&group1, &group2, AlternativeHypothesis::TwoSided, true)
            .expect("operation should succeed");

        assert!(result.test_name.contains("t-test"));
        assert!(result.effect_size.is_some());
        assert!(result.confidence_interval.is_some());
    }

    #[test]
    fn test_correlation_test() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let result = correlation_test(&x, &y, AlternativeHypothesis::TwoSided)
            .expect("operation should succeed");

        assert_eq!(result.test_name, "Pearson correlation test");
        assert!((result.additional_info["correlation"] - 1.0).abs() < 1e-10);
        assert!(result.reject_null); // Should reject null of no correlation
    }

    #[test]
    fn test_chi_square_independence() {
        let observed = vec![vec![10.0, 15.0, 25.0], vec![20.0, 10.0, 15.0]];

        let result = chi_square_independence(&observed).expect("operation should succeed");

        assert_eq!(result.test_name, "Chi-square test of independence");
        assert!(result.degrees_of_freedom.is_some());
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_multiple_comparison_bonferroni() {
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let adjusted = adjust_p_values(&p_values, MultipleComparisonCorrection::Bonferroni)
            .expect("operation should succeed");

        // All should be multiplied by 5
        assert!((adjusted[0] - 0.05).abs() < 1e-10);
        assert!((adjusted[1] - 0.10).abs() < 1e-10);
        assert!((adjusted[4] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_effect_size_interpretation() {
        let small_effect = EffectSize::CohensD(0.3);
        assert_eq!(small_effect.interpretation(), "Small");

        let large_effect = EffectSize::CohensD(1.0);
        assert_eq!(large_effect.interpretation(), "Large");

        let medium_correlation = EffectSize::PearsonR(0.4);
        assert_eq!(medium_correlation.interpretation(), "Medium");
    }

    #[test]
    fn test_shapiro_wilk_known_sample() {
        // Sample: [6.0, 1.0, -1.0, 3.0, 2.0]
        // Royston AS R94 normal-score weights for n=5 give W ≈ 0.984.
        // (The old fabricated-coefficient code returned a different, incorrect value.)
        let data = [6.0_f64, 1.0, -1.0, 3.0, 2.0];
        let result = shapiro_wilk_test(&data).expect("shapiro_wilk_test should succeed");
        // W should be close to 0.984 (within 0.01 tolerance)
        assert!(
            (result.statistic - 0.984).abs() < 0.01,
            "W = {}, expected ≈ 0.984",
            result.statistic
        );
        // p-value should be > 0.05 (cannot reject normality for this small, unremarkable sample)
        assert!(
            result.p_value > 0.05,
            "p = {}, expected > 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_shapiro_wilk_normal_data() {
        // Hard-coded normal-ish sample (mean≈0, std≈1), large enough for stable p.
        let data = [
            0.1_f64, 0.5, 1.2, -0.3, -0.8, 0.6, 1.1, -0.2, 0.4, 0.9, -0.5, 0.3, -1.0, 0.7, -0.4,
            1.5, -0.6, 0.2, 0.8, -0.1, -0.7, 1.3, -0.9, 0.0, 0.4, -0.3, 0.6, -0.2, 0.1, 0.5,
        ];
        let result = shapiro_wilk_test(&data).expect("shapiro_wilk_test should succeed");
        // Normal data → p-value should be > 0.05
        assert!(
            result.p_value > 0.05,
            "Normal data: p = {}, expected > 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_shapiro_wilk_skewed_data() {
        // Heavily right-skewed (exponential-like) → should reject normality.
        let data = [
            0.01_f64, 0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.5, 5.0, 8.0,
        ];
        let result = shapiro_wilk_test(&data).expect("shapiro_wilk_test should succeed");
        // Skewed data → p-value should be < 0.05
        assert!(
            result.p_value < 0.05,
            "Skewed data: p = {}, expected < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_shapiro_wilk_real_w_coefficients() {
        // Near-normal data: linear-spaced quantiles are approximately normal.
        // Real W for approximately-normal data should be > 0.85.
        let normal_data = vec![-1.5, -0.8, -0.3, 0.1, 0.4, 0.7, 1.0, 1.3, 1.8, 2.2f64];
        let result = shapiro_wilk_test(&normal_data).expect("shapiro_wilk_test should succeed");
        assert!(
            result.statistic > 0.85,
            "W={} should be > 0.85 for near-normal data",
            result.statistic
        );
        assert!(
            result.p_value > 0.05,
            "p={} should be > 0.05 for near-normal data",
            result.p_value
        );
    }

    #[test]
    fn test_shapiro_wilk_rejects_skewed() {
        // Cubic-growth data is clearly non-normal (heavy right skew).
        let skewed: Vec<f64> = (1..=10).map(|i| (i as f64).powi(3)).collect();
        let result = shapiro_wilk_test(&skewed).expect("shapiro_wilk_test should succeed");
        assert!(
            result.statistic < 0.9,
            "W={} should be < 0.9 for skewed data",
            result.statistic
        );
    }
}
