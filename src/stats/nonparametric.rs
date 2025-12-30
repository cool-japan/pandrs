//! Non-parametric statistical tests
//!
//! This module provides distribution-free statistical tests that don't assume
//! specific probability distributions for the underlying data.

use crate::core::error::{Error, Result};
use crate::stats::distributions::{ChiSquared, Distribution, Normal};
use crate::stats::hypothesis::{AlternativeHypothesis, EffectSize, TestResult};
use crate::utils::rand_compat::{thread_rng, GenRangeCompat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mann-Whitney U test (Wilcoxon rank-sum test)
pub fn mann_whitney_u_test(
    group1: &[f64],
    group2: &[f64],
    alternative: AlternativeHypothesis,
) -> Result<TestResult> {
    if group1.is_empty() || group2.is_empty() {
        return Err(Error::InvalidValue("Both groups must contain data".into()));
    }

    let n1 = group1.len();
    let n2 = group2.len();
    let n_total = n1 + n2;

    // Combine and rank all values
    let mut combined: Vec<(f64, usize)> = Vec::new();

    // Add group 1 values (marked as group 0)
    for &value in group1 {
        combined.push((value, 0));
    }

    // Add group 2 values (marked as group 1)
    for &value in group2 {
        combined.push((value, 1));
    }

    // Sort by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks, handling ties
    let ranks = assign_ranks(&combined.iter().map(|(val, _)| *val).collect::<Vec<_>>());

    // Sum ranks for group 1
    let mut r1 = 0.0;
    for (i, &(_, group)) in combined.iter().enumerate() {
        if group == 0 {
            r1 += ranks[i];
        }
    }

    // Calculate U statistics
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;

    // Use smaller U as test statistic
    let u_statistic = u1.min(u2);

    // Calculate z-score for normal approximation (valid for large samples)
    let mean_u = (n1 * n2) as f64 / 2.0;
    let var_u = (n1 * n2 * (n_total + 1)) as f64 / 12.0;

    // Continuity correction
    let z_statistic = if u_statistic > mean_u {
        (u_statistic - 0.5 - mean_u) / var_u.sqrt()
    } else {
        (u_statistic + 0.5 - mean_u) / var_u.sqrt()
    };

    let normal = Normal::new(0.0, 1.0)?;
    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => 2.0 * (1.0 - normal.cdf(z_statistic.abs())),
        AlternativeHypothesis::Greater => 1.0 - normal.cdf(z_statistic),
        AlternativeHypothesis::Less => normal.cdf(z_statistic),
    };

    // Effect size (rank-biserial correlation)
    let effect_size_r = 2.0 * u1 / (n1 * n2) as f64 - 1.0;
    let effect_size = EffectSize::PearsonR(effect_size_r);

    let mut additional_info = HashMap::new();
    additional_info.insert("u1".to_string(), u1);
    additional_info.insert("u2".to_string(), u2);
    additional_info.insert("n1".to_string(), n1 as f64);
    additional_info.insert("n2".to_string(), n2 as f64);
    additional_info.insert("mean_u".to_string(), mean_u);
    additional_info.insert("var_u".to_string(), var_u);
    additional_info.insert("rank_sum_group1".to_string(), r1);

    Ok(TestResult {
        statistic: u_statistic,
        p_value,
        degrees_of_freedom: None,
        critical_value: None,
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: None,
        test_name: "Mann-Whitney U test".to_string(),
        alternative,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Wilcoxon signed-rank test for paired samples
pub fn wilcoxon_signed_rank_test(
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

    // Calculate differences and remove zeros
    let differences: Vec<f64> = before
        .iter()
        .zip(after.iter())
        .map(|(&b, &a)| b - a)
        .filter(|&diff| diff != 0.0)
        .collect();

    if differences.is_empty() {
        return Err(Error::InvalidValue("No non-zero differences found".into()));
    }

    let n = differences.len();

    // Get absolute differences and their ranks
    let abs_differences: Vec<f64> = differences.iter().map(|&d| d.abs()).collect();
    let ranks = assign_ranks(&abs_differences);

    // Sum of positive and negative ranks
    let mut w_plus = 0.0;
    let mut w_minus = 0.0;

    for (i, &diff) in differences.iter().enumerate() {
        if diff > 0.0 {
            w_plus += ranks[i];
        } else {
            w_minus += ranks[i];
        }
    }

    // Test statistic (smaller of the two rank sums)
    let w_statistic = w_plus.min(w_minus);

    // Normal approximation for large samples
    let mean_w = n as f64 * (n + 1) as f64 / 4.0;
    let var_w = n as f64 * (n + 1) as f64 * (2 * n + 1) as f64 / 24.0;

    // Continuity correction
    let z_statistic = if w_statistic > mean_w {
        (w_statistic - 0.5 - mean_w) / var_w.sqrt()
    } else {
        (w_statistic + 0.5 - mean_w) / var_w.sqrt()
    };

    let normal = Normal::new(0.0, 1.0)?;
    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => 2.0 * (1.0 - normal.cdf(z_statistic.abs())),
        AlternativeHypothesis::Greater => 1.0 - normal.cdf(z_statistic),
        AlternativeHypothesis::Less => normal.cdf(z_statistic),
    };

    // Effect size (matched-pairs rank-biserial correlation)
    let effect_size_r = w_plus / (n * (n + 1) / 2) as f64 - 0.5;
    let effect_size = EffectSize::PearsonR(effect_size_r);

    let mut additional_info = HashMap::new();
    additional_info.insert("w_plus".to_string(), w_plus);
    additional_info.insert("w_minus".to_string(), w_minus);
    additional_info.insert("n_differences".to_string(), n as f64);
    additional_info.insert("mean_w".to_string(), mean_w);
    additional_info.insert("var_w".to_string(), var_w);

    Ok(TestResult {
        statistic: w_statistic,
        p_value,
        degrees_of_freedom: None,
        critical_value: None,
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: None,
        test_name: "Wilcoxon signed-rank test".to_string(),
        alternative,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Kruskal-Wallis test (non-parametric ANOVA)
pub fn kruskal_wallis_test(groups: &[&[f64]]) -> Result<TestResult> {
    if groups.is_empty() {
        return Err(Error::InvalidValue("At least one group is required".into()));
    }

    if groups.len() < 2 {
        return Err(Error::InvalidValue(
            "At least two groups are required".into(),
        ));
    }

    // Check that all groups have data
    for (i, group) in groups.iter().enumerate() {
        if group.is_empty() {
            return Err(Error::InvalidValue(format!("Group {} is empty", i)));
        }
    }

    let k = groups.len();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();

    // Combine all values with group labels
    let mut combined: Vec<(f64, usize)> = Vec::new();
    for (group_idx, group) in groups.iter().enumerate() {
        for &value in group.iter() {
            combined.push((value, group_idx));
        }
    }

    // Sort by value
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks
    let values: Vec<f64> = combined.iter().map(|(val, _)| *val).collect();
    let ranks = assign_ranks(&values);

    // Calculate rank sums for each group
    let mut rank_sums = vec![0.0; k];
    let mut group_sizes = vec![0; k];

    for (i, &(_, group_idx)) in combined.iter().enumerate() {
        rank_sums[group_idx] += ranks[i];
        group_sizes[group_idx] += 1;
    }

    // Calculate H statistic
    let mut h_statistic = 0.0;
    for i in 0..k {
        let ni = group_sizes[i] as f64;
        let ri = rank_sums[i];
        h_statistic += ri * ri / ni;
    }

    h_statistic =
        12.0 / (n_total * (n_total + 1)) as f64 * h_statistic - 3.0 * (n_total + 1) as f64;

    // Degrees of freedom
    let df = (k - 1) as f64;

    // P-value using chi-squared distribution
    let chi_sq = ChiSquared::new(df)?;
    let p_value = 1.0 - chi_sq.cdf(h_statistic);

    // Effect size (eta-squared)
    let eta_squared = (h_statistic - k as f64 + 1.0) / (n_total as f64 - k as f64);
    let effect_size = EffectSize::EtaSquared(eta_squared);

    let mut additional_info = HashMap::new();
    additional_info.insert("k_groups".to_string(), k as f64);
    additional_info.insert("n_total".to_string(), n_total as f64);
    additional_info.insert("eta_squared".to_string(), eta_squared);

    for (i, &rank_sum) in rank_sums.iter().enumerate() {
        additional_info.insert(format!("rank_sum_group_{}", i), rank_sum);
        additional_info.insert(format!("n_group_{}", i), group_sizes[i] as f64);
    }

    Ok(TestResult {
        statistic: h_statistic,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: Some(chi_sq.inverse_cdf(0.95)),
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: None,
        test_name: "Kruskal-Wallis test".to_string(),
        alternative: AlternativeHypothesis::Greater,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Friedman test for repeated measures
pub fn friedman_test(data: &[Vec<f64>]) -> Result<TestResult> {
    if data.is_empty() {
        return Err(Error::InvalidValue("Data cannot be empty".into()));
    }

    let n_subjects = data.len();
    let k_conditions = data[0].len();

    if k_conditions < 2 {
        return Err(Error::InvalidValue(
            "At least two conditions required".into(),
        ));
    }

    // Check that all subjects have same number of conditions
    for (i, subject) in data.iter().enumerate() {
        if subject.len() != k_conditions {
            return Err(Error::DimensionMismatch(format!(
                "Subject {} has different number of conditions",
                i
            )));
        }
    }

    // Rank within each subject (row)
    let mut rank_sums = vec![0.0; k_conditions];

    for subject_data in data.iter() {
        let ranks = assign_ranks(subject_data);
        for (condition, &rank) in ranks.iter().enumerate() {
            rank_sums[condition] += rank;
        }
    }

    // Calculate Friedman statistic
    let mut q_statistic = 0.0;
    for &rank_sum in &rank_sums {
        q_statistic += rank_sum * rank_sum;
    }

    let n = n_subjects as f64;
    let k = k_conditions as f64;

    q_statistic = 12.0 / (n * k * (k + 1.0)) * q_statistic - 3.0 * n * (k + 1.0);

    // Degrees of freedom
    let df = k - 1.0;

    // P-value using chi-squared distribution
    let chi_sq = ChiSquared::new(df)?;
    let p_value = 1.0 - chi_sq.cdf(q_statistic);

    // Effect size (Kendall's W)
    let kendalls_w = q_statistic / (n * (k - 1.0));
    let effect_size = EffectSize::EtaSquared(kendalls_w);

    let mut additional_info = HashMap::new();
    additional_info.insert("n_subjects".to_string(), n);
    additional_info.insert("k_conditions".to_string(), k);
    additional_info.insert("kendalls_w".to_string(), kendalls_w);

    for (i, &rank_sum) in rank_sums.iter().enumerate() {
        additional_info.insert(format!("rank_sum_condition_{}", i), rank_sum);
    }

    Ok(TestResult {
        statistic: q_statistic,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: Some(chi_sq.inverse_cdf(0.95)),
        effect_size: Some(effect_size.value()),
        effect_size_interpretation: Some(effect_size.interpretation()),
        confidence_interval: None,
        test_name: "Friedman test".to_string(),
        alternative: AlternativeHypothesis::Greater,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Kolmogorov-Smirnov two-sample test
pub fn ks_two_sample_test(
    sample1: &[f64],
    sample2: &[f64],
    alternative: AlternativeHypothesis,
) -> Result<TestResult> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(Error::InvalidValue("Both samples must contain data".into()));
    }

    let n1 = sample1.len();
    let n2 = sample2.len();

    // Sort both samples
    let mut sorted1 = sample1.to_vec();
    let mut sorted2 = sample2.to_vec();
    sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Find all unique values
    let mut all_values = Vec::new();
    all_values.extend_from_slice(&sorted1);
    all_values.extend_from_slice(&sorted2);
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    // Calculate empirical CDFs at each unique value
    let mut max_diff: f64 = 0.0;
    let mut max_diff_positive: f64 = 0.0;
    let mut max_diff_negative: f64 = 0.0;

    for &value in &all_values {
        let cdf1 = empirical_cdf(&sorted1, value);
        let cdf2 = empirical_cdf(&sorted2, value);
        let diff = cdf1 - cdf2;

        max_diff = max_diff.max(diff.abs());
        max_diff_positive = max_diff_positive.max(diff);
        max_diff_negative = max_diff_negative.max(-diff);
    }

    let d_statistic = match alternative {
        AlternativeHypothesis::TwoSided => max_diff,
        AlternativeHypothesis::Greater => max_diff_positive,
        AlternativeHypothesis::Less => max_diff_negative,
    };

    // Approximate p-value calculation
    let effective_n = (n1 * n2) as f64 / (n1 + n2) as f64;
    let z = d_statistic * effective_n.sqrt();

    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => {
            // Approximation for two-sided test
            2.0 * (-2.0 * z * z).exp()
        }
        AlternativeHypothesis::Greater | AlternativeHypothesis::Less => {
            // Approximation for one-sided test
            (-2.0 * z * z).exp()
        }
    };

    let mut additional_info = HashMap::new();
    additional_info.insert("n1".to_string(), n1 as f64);
    additional_info.insert("n2".to_string(), n2 as f64);
    additional_info.insert("max_diff_positive".to_string(), max_diff_positive);
    additional_info.insert("max_diff_negative".to_string(), max_diff_negative);
    additional_info.insert("effective_n".to_string(), effective_n);

    Ok(TestResult {
        statistic: d_statistic,
        p_value: p_value.max(0.001).min(1.0), // Clamp p-value
        degrees_of_freedom: None,
        critical_value: None,
        effect_size: None,
        effect_size_interpretation: None,
        confidence_interval: None,
        test_name: "Kolmogorov-Smirnov two-sample test".to_string(),
        alternative,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Runs test for randomness
pub fn runs_test(sequence: &[bool]) -> Result<TestResult> {
    if sequence.is_empty() {
        return Err(Error::InvalidValue("Sequence cannot be empty".into()));
    }

    if sequence.len() < 2 {
        return Err(Error::InvalidValue(
            "Sequence must have at least 2 elements".into(),
        ));
    }

    // Count runs
    let mut runs = 1;
    for i in 1..sequence.len() {
        if sequence[i] != sequence[i - 1] {
            runs += 1;
        }
    }

    // Count number of each type
    let n1 = sequence.iter().filter(|&&x| x).count();
    let n2 = sequence.len() - n1;

    if n1 == 0 || n2 == 0 {
        return Err(Error::InvalidValue(
            "Sequence must contain both true and false values".into(),
        ));
    }

    // Expected runs and variance
    let n_total = sequence.len() as f64;
    let n1_f = n1 as f64;
    let n2_f = n2 as f64;

    let expected_runs = (2.0 * n1_f * n2_f) / n_total + 1.0;
    let variance_runs =
        (2.0 * n1_f * n2_f * (2.0 * n1_f * n2_f - n_total)) / (n_total * n_total * (n_total - 1.0));

    // Z-statistic with continuity correction
    let z_statistic = if runs as f64 > expected_runs {
        (runs as f64 - 0.5 - expected_runs) / variance_runs.sqrt()
    } else {
        (runs as f64 + 0.5 - expected_runs) / variance_runs.sqrt()
    };

    let normal = Normal::new(0.0, 1.0)?;
    let p_value = 2.0 * (1.0 - normal.cdf(z_statistic.abs()));

    let mut additional_info = HashMap::new();
    additional_info.insert("n_runs".to_string(), runs as f64);
    additional_info.insert("n_true".to_string(), n1_f);
    additional_info.insert("n_false".to_string(), n2_f);
    additional_info.insert("expected_runs".to_string(), expected_runs);
    additional_info.insert("variance_runs".to_string(), variance_runs);

    Ok(TestResult {
        statistic: runs as f64,
        p_value,
        degrees_of_freedom: None,
        critical_value: None,
        effect_size: None,
        effect_size_interpretation: None,
        confidence_interval: None,
        test_name: "Runs test for randomness".to_string(),
        alternative: AlternativeHypothesis::TwoSided,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

/// Helper function to assign ranks (handling ties by averaging)
fn assign_ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();

    // Create indexed data for sorting
    let mut indexed_data: Vec<(usize, f64)> =
        data.iter().enumerate().map(|(i, &val)| (i, val)).collect();

    // Sort by value
    indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];

    // Assign ranks, handling ties by averaging
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed_data[j].1 - indexed_data[i].1).abs() < 1e-10 {
            j += 1;
        }

        // Average rank for tied values (1-based ranking)
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;

        for k in i..j {
            ranks[indexed_data[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Calculate empirical CDF at a given value
fn empirical_cdf(sorted_data: &[f64], value: f64) -> f64 {
    let count = sorted_data.iter().filter(|&&x| x <= value).count();
    count as f64 / sorted_data.len() as f64
}

/// Bootstrap confidence interval for a statistic
pub fn bootstrap_confidence_interval<F>(
    data: &[f64],
    statistic_fn: F,
    confidence_level: f64,
    n_bootstrap: usize,
) -> Result<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if data.is_empty() {
        return Err(Error::InvalidValue("Data cannot be empty".into()));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(Error::InvalidValue(
            "Confidence level must be between 0 and 1".into(),
        ));
    }

    let mut rng = thread_rng();
    let mut bootstrap_stats = Vec::with_capacity(n_bootstrap);

    // Generate bootstrap samples
    for _ in 0..n_bootstrap {
        let mut bootstrap_sample = Vec::with_capacity(data.len());

        // Sample with replacement
        for _ in 0..data.len() {
            let idx = rng.gen_range(0..data.len());
            bootstrap_sample.push(data[idx]);
        }

        // Calculate statistic for this bootstrap sample
        let stat = statistic_fn(&bootstrap_sample);
        bootstrap_stats.push(stat);
    }

    // Sort bootstrap statistics
    bootstrap_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Calculate percentiles for confidence interval
    let alpha = 1.0 - confidence_level;
    let lower_percentile = alpha / 2.0;
    let upper_percentile = 1.0 - alpha / 2.0;

    let lower_idx = (lower_percentile * n_bootstrap as f64) as usize;
    let upper_idx = (upper_percentile * n_bootstrap as f64) as usize;

    let lower_idx = lower_idx.min(n_bootstrap - 1);
    let upper_idx = upper_idx.min(n_bootstrap - 1);

    Ok((bootstrap_stats[lower_idx], bootstrap_stats[upper_idx]))
}

/// Permutation test for comparing two groups
pub fn permutation_test<F>(
    group1: &[f64],
    group2: &[f64],
    test_statistic_fn: F,
    n_permutations: usize,
    alternative: AlternativeHypothesis,
) -> Result<TestResult>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    if group1.is_empty() || group2.is_empty() {
        return Err(Error::InvalidValue("Both groups must contain data".into()));
    }

    // Calculate observed test statistic
    let observed_statistic = test_statistic_fn(group1, group2);

    // Combine all data
    let mut combined_data = Vec::new();
    combined_data.extend_from_slice(group1);
    combined_data.extend_from_slice(group2);

    let n1 = group1.len();
    let n2 = group2.len();
    let mut rng = thread_rng();

    // Generate permutation distribution
    let mut permutation_stats = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Randomly shuffle combined data
        let mut shuffled = combined_data.clone();
        for i in (1..shuffled.len()).rev() {
            let j = rng.gen_range(0..=i);
            shuffled.swap(i, j);
        }

        // Split into two groups
        let perm_group1 = &shuffled[..n1];
        let perm_group2 = &shuffled[n1..];

        // Calculate test statistic for this permutation
        let perm_stat = test_statistic_fn(perm_group1, perm_group2);
        permutation_stats.push(perm_stat);
    }

    // Calculate p-value
    let p_value = match alternative {
        AlternativeHypothesis::TwoSided => {
            let count = permutation_stats
                .iter()
                .filter(|&&stat| stat.abs() >= observed_statistic.abs())
                .count();
            count as f64 / n_permutations as f64
        }
        AlternativeHypothesis::Greater => {
            let count = permutation_stats
                .iter()
                .filter(|&&stat| stat >= observed_statistic)
                .count();
            count as f64 / n_permutations as f64
        }
        AlternativeHypothesis::Less => {
            let count = permutation_stats
                .iter()
                .filter(|&&stat| stat <= observed_statistic)
                .count();
            count as f64 / n_permutations as f64
        }
    };

    let mut additional_info = HashMap::new();
    additional_info.insert("n_permutations".to_string(), n_permutations as f64);
    additional_info.insert("n1".to_string(), n1 as f64);
    additional_info.insert("n2".to_string(), n2 as f64);

    Ok(TestResult {
        statistic: observed_statistic,
        p_value,
        degrees_of_freedom: None,
        critical_value: None,
        effect_size: None,
        effect_size_interpretation: None,
        confidence_interval: None,
        test_name: "Permutation test".to_string(),
        alternative,
        reject_null: p_value < 0.05,
        additional_info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mann_whitney_u() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result =
            mann_whitney_u_test(&group1, &group2, AlternativeHypothesis::TwoSided).unwrap();

        assert_eq!(result.test_name, "Mann-Whitney U test");
        assert!(result.reject_null); // Groups are clearly different
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_wilcoxon_signed_rank() {
        let before = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let after = vec![2.0, 3.0, 4.0, 5.0, 6.0]; // All increased by 1

        let result =
            wilcoxon_signed_rank_test(&before, &after, AlternativeHypothesis::TwoSided).unwrap();

        assert_eq!(result.test_name, "Wilcoxon signed-rank test");
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_kruskal_wallis() {
        let group1 = vec![1.0, 2.0, 3.0];
        let group2 = vec![4.0, 5.0, 6.0];
        let group3 = vec![7.0, 8.0, 9.0];
        let groups = vec![group1.as_slice(), group2.as_slice(), group3.as_slice()];

        let result = kruskal_wallis_test(&groups).unwrap();

        assert_eq!(result.test_name, "Kruskal-Wallis test");
        assert!(result.degrees_of_freedom.is_some());
        assert!(result.effect_size.is_some());
    }

    #[test]
    fn test_assign_ranks() {
        let data = vec![1.0, 3.0, 2.0, 3.0, 5.0];
        let ranks = assign_ranks(&data);

        // Expected ranks: [1, 2.5, 2, 2.5, 5] (ties get averaged ranks)
        assert_eq!(ranks[0], 1.0); // 1.0 is rank 1
        assert_eq!(ranks[1], 3.5); // 3.0 appears twice, ranks 3 and 4, average = 3.5
        assert_eq!(ranks[2], 2.0); // 2.0 is rank 2
        assert_eq!(ranks[3], 3.5); // 3.0 appears twice, ranks 3 and 4, average = 3.5
        assert_eq!(ranks[4], 5.0); // 5.0 is rank 5
    }

    #[test]
    fn test_runs_test() {
        let sequence = vec![true, false, true, false, true, false];
        let result = runs_test(&sequence).unwrap();

        assert_eq!(result.test_name, "Runs test for randomness");
        assert!(result.additional_info.contains_key("n_runs"));
    }

    #[test]
    fn test_ks_two_sample() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result =
            ks_two_sample_test(&sample1, &sample2, AlternativeHypothesis::TwoSided).unwrap();

        assert_eq!(result.test_name, "Kolmogorov-Smirnov two-sample test");
        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_bootstrap_confidence_interval() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Bootstrap CI for the mean
        let (lower, upper) = bootstrap_confidence_interval(
            &data,
            |sample| sample.iter().sum::<f64>() / sample.len() as f64,
            0.95,
            1000,
        )
        .unwrap();

        let actual_mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!(lower < actual_mean && actual_mean < upper);
    }
}
