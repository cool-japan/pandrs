//! Comprehensive hypothesis testing framework
//!
//! This module provides a complete suite of statistical hypothesis tests including
//! parametric and non-parametric tests, effect size calculations, and multiple
//! comparison corrections for robust statistical analysis.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::Series;
use crate::stats::distributions::{ChiSquared, Distribution, FDistribution, Normal, TDistribution};
use crate::utils::rand_compat::{thread_rng, GenRangeCompat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

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
    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_statistic.abs())),
        AlternativeHypothesis::Greater => 1.0 - t_dist.cdf(t_statistic),
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
    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_statistic.abs())),
        AlternativeHypothesis::Greater => 1.0 - t_dist.cdf(t_statistic),
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
    let df_total = n_total as f64 - 1.0;

    // Mean squares
    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;

    // F-statistic
    let f_statistic = ms_between / ms_within;

    // P-value
    let f_dist = FDistribution::new(df_between, df_within)?;
    let p_value = 1.0 - f_dist.cdf(f_statistic);

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

    // P-value
    let chi_sq_dist = ChiSquared::new(df as f64)?;
    let p_value = 1.0 - chi_sq_dist.cdf(chi_square);

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
        match alternative {
            AlternativeHypothesis::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_statistic.abs())),
            AlternativeHypothesis::Greater => 1.0 - t_dist.cdf(t_statistic),
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

    // Sort the data
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate sample mean and standard deviation
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();

    // This is a simplified implementation
    // Full Shapiro-Wilk requires complex coefficients that depend on sample size
    // For a complete implementation, we'd need lookup tables or algorithms for these coefficients

    // Simplified W statistic calculation (approximation)
    let mut w_numerator = 0.0;
    let k = n / 2;

    // Approximate coefficients (this is a simplification)
    for i in 0..k {
        let coeff = if i == 0 {
            0.7071
        } else {
            0.5 / (i as f64 + 1.0)
        };
        w_numerator += coeff * (sorted_data[n - 1 - i] - sorted_data[i]);
    }

    let w_denominator = (n - 1) as f64 * variance;
    let w_statistic = w_numerator.powi(2) / w_denominator;

    // Approximate p-value calculation (very simplified)
    // In practice, this would use complex transformations and lookup tables
    let log_w = w_statistic.ln();
    let normalized = (log_w + 1.0) * (n as f64).sqrt();

    // Very rough approximation - would need proper implementation
    let p_value = if w_statistic > 0.95 {
        1.0 - Normal::new(0.0, 1.0)?.cdf(normalized)
    } else {
        0.01
    };

    let mut additional_info = HashMap::new();
    additional_info.insert("n".to_string(), n as f64);
    additional_info.insert("mean".to_string(), mean);
    additional_info.insert("std_dev".to_string(), std_dev);
    additional_info.insert("note".to_string(), 1.0); // Indicates this is an approximation

    Ok(TestResult {
        statistic: w_statistic,
        p_value: p_value.max(0.001).min(1.0), // Clamp p-value
        degrees_of_freedom: None,
        critical_value: Some(0.95), // Rough threshold
        effect_size: None,
        effect_size_interpretation: None,
        confidence_interval: None,
        test_name: "Shapiro-Wilk normality test (approximation)".to_string(),
        alternative: AlternativeHypothesis::Greater,
        reject_null: p_value < 0.05,
        additional_info,
    })
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
            indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
            indexed_p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Descending order

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
            indexed_p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
        let result = one_sample_ttest(&data, 3.0, AlternativeHypothesis::TwoSided).unwrap();

        assert_eq!(result.test_name, "One-sample t-test");
        assert!(result.p_value > 0.05); // Should not reject null hypothesis
        assert!(!result.reject_null);
    }

    #[test]
    fn test_independent_ttest() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let result =
            independent_ttest(&group1, &group2, AlternativeHypothesis::TwoSided, true).unwrap();

        assert!(result.test_name.contains("t-test"));
        assert!(result.effect_size.is_some());
        assert!(result.confidence_interval.is_some());
    }

    #[test]
    fn test_correlation_test() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let result = correlation_test(&x, &y, AlternativeHypothesis::TwoSided).unwrap();

        assert_eq!(result.test_name, "Pearson correlation test");
        assert!((result.additional_info["correlation"] - 1.0).abs() < 1e-10);
        assert!(result.reject_null); // Should reject null of no correlation
    }

    #[test]
    fn test_chi_square_independence() {
        let observed = vec![vec![10.0, 15.0, 25.0], vec![20.0, 10.0, 15.0]];

        let result = chi_square_independence(&observed).unwrap();

        assert_eq!(result.test_name, "Chi-square test of independence");
        assert!(result.degrees_of_freedom.is_some());
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_multiple_comparison_bonferroni() {
        let p_values = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let adjusted =
            adjust_p_values(&p_values, MultipleComparisonCorrection::Bonferroni).unwrap();

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
}
