//! Sampling methods module
//!
//! This module provides functions for statistical sampling, including
//! random sampling, stratified sampling, bootstrap resampling, and more.

use crate::error::{Result, Error};
use crate::dataframe::DataFrame;
use rand::Rng;
use rand::seq::SliceRandom;
// Import compatibility utils
use crate::utils::rand_compat::{thread_rng, GenRangeCompat};
use std::collections::HashMap;

/// Internal implementation for sampling DataFrame rows
pub(crate) fn sample_impl(
    df: &DataFrame,
    fraction: f64,
    replace: bool
) -> Result<DataFrame> {
    if fraction <= 0.0 || fraction > 1.0 {
        return Err(Error::InvalidInput("Sampling fraction must be between 0 and 1".into()));
    }
    
    let n_rows = df.nrows();
    if n_rows == 0 {
        return Err(Error::EmptyData("Cannot sample from empty DataFrame".into()));
    }
    
    let sample_size = (n_rows as f64 * fraction).ceil() as usize;
    
    // Generate random indices
    let mut indices = Vec::with_capacity(sample_size);
    let mut rng = thread_rng();
    
    if replace {
        // Sampling with replacement
        for _ in 0..sample_size {
            indices.push(rng.gen_range(0..n_rows));
        }
    } else {
        // Sampling without replacement
        if sample_size > n_rows {
            return Err(Error::InvalidInput(
                format!("Sample size ({}) cannot be larger than DataFrame size ({}) when sampling without replacement", 
                       sample_size, n_rows)
            ));
        }
        
        // Create a vector of indices and shuffle it
        let mut all_indices: Vec<usize> = (0..n_rows).collect();
        all_indices.shuffle(&mut rng);
        
        // Take the first sample_size indices
        indices = all_indices.into_iter().take(sample_size).collect();
    }
    
    // Create sample DataFrame
    let mut sample_df = DataFrame::new();
    
    // Add each column from the original DataFrame
    for col_name in df.columns() {
        let col = df.get_column(&col_name)?;
        let sampled_col = col.sample(&indices)?;
        sample_df.add_column(col_name.clone(), sampled_col)?;
    }
    
    Ok(sample_df)
}

/// Internal implementation for bootstrap resampling
pub(crate) fn bootstrap_impl(
    data: &[f64],
    n_samples: usize
) -> Result<Vec<Vec<f64>>> {
    if data.is_empty() {
        return Err(Error::EmptyData("Cannot bootstrap from empty data".into()));
    }
    
    if n_samples == 0 {
        return Err(Error::InvalidInput("Number of bootstrap samples must be greater than 0".into()));
    }
    
    let n = data.len();
    let mut rng = thread_rng();
    let mut bootstrap_samples = Vec::with_capacity(n_samples);
    
    for _ in 0..n_samples {
        // Generate a bootstrap sample (sampling with replacement)
        let mut sample = Vec::with_capacity(n);
        
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sample.push(data[idx]);
        }
        
        bootstrap_samples.push(sample);
    }
    
    Ok(bootstrap_samples)
}

/// Perform stratified sampling
///
/// # Description
/// Samples data from each stratum with either the same proportion or custom proportions.
///
/// # Arguments
/// * `df` - The DataFrame to sample from
/// * `strata_column` - The column name that defines the strata
/// * `fractions` - A map of stratum values to sampling fractions
/// * `replace` - Whether to sample with replacement
///
/// # Example
/// ```
/// use pandrs::stats::sampling;
/// use pandrs::dataframe::DataFrame;
/// use std::collections::HashMap;
///
/// let df = DataFrame::new(); // DataFrame with data
/// 
/// // Sample 20% from stratum A and 30% from stratum B
/// let mut fractions = HashMap::new();
/// fractions.insert("A".to_string(), 0.2);
/// fractions.insert("B".to_string(), 0.3);
/// 
/// let stratified_sample = sampling::stratified_sample(&df, "group", &fractions, false).unwrap();
/// ```
pub fn stratified_sample(
    df: &DataFrame,
    strata_column: &str,
    fractions: &HashMap<String, f64>,
    replace: bool
) -> Result<DataFrame> {
    // Check if strata column exists
    if !df.has_column(strata_column) {
        return Err(Error::InvalidColumn(format!("Column '{}' does not exist", strata_column)));
    }
    
    let strata_series = df.get_column(strata_column)?;
    
    // Convert strata column to strings
    let strata_values = strata_series.as_str()?;
    
    // Check for valid fractions
    for (stratum, &fraction) in fractions {
        if fraction <= 0.0 || fraction > 1.0 {
            return Err(Error::InvalidInput(
                format!("Sampling fraction for stratum '{}' must be between 0 and 1", stratum)
            ));
        }
    }
    
    // Create a map of strata to row indices
    let mut strata_indices: HashMap<String, Vec<usize>> = HashMap::new();
    
    for (i, stratum) in strata_values.iter().enumerate() {
        strata_indices.entry(stratum.to_string())
            .or_insert_with(Vec::new)
            .push(i);
    }
    
    // Sample from each stratum
    let mut all_sampled_indices = Vec::new();
    let mut rng = thread_rng();
    
    for (stratum, indices) in strata_indices {
        // Skip if no fraction specified for this stratum
        if !fractions.contains_key(&stratum) {
            continue;
        }
        
        let fraction = fractions[&stratum];
        let n_rows = indices.len();
        let sample_size = (n_rows as f64 * fraction).ceil() as usize;
        
        if n_rows == 0 {
            continue;
        }
        
        if !replace && sample_size > n_rows {
            return Err(Error::InvalidInput(
                format!("Sample size ({}) cannot be larger than stratum size ({}) when sampling without replacement", 
                       sample_size, n_rows)
            ));
        }
        
        if replace {
            // Sampling with replacement
            for _ in 0..sample_size {
                let idx = indices[rng.gen_range(0..n_rows)];
                all_sampled_indices.push(idx);
            }
        } else {
            // Sampling without replacement
            let mut stratum_indices = indices.clone();
            stratum_indices.shuffle(&mut rng);
            
            for i in 0..sample_size {
                all_sampled_indices.push(stratum_indices[i]);
            }
        }
    }
    
    // Check if any rows were sampled
    if all_sampled_indices.is_empty() {
        return Err(Error::InvalidInput("No rows were sampled. Check if strata names match.".into()));
    }
    
    // Create sample DataFrame
    let mut sample_df = DataFrame::new();
    
    // Add each column from the original DataFrame
    for col_name in df.columns() {
        let col = df.get_column(&col_name)?;
        let sampled_col = col.sample(&all_sampled_indices)?;
        sample_df.add_column(col_name.clone(), sampled_col)?;
    }
    
    Ok(sample_df)
}

/// Generate bootstrap confidence interval
///
/// # Description
/// Computes bootstrap confidence intervals for a parameter estimate.
///
/// # Arguments
/// * `data` - The data to bootstrap
/// * `n_samples` - Number of bootstrap samples to generate
/// * `statistic_fn` - Function to compute the statistic of interest
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95% confidence)
///
/// # Example
/// ```
/// use pandrs::stats::sampling;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// 
/// // Compute bootstrap confidence interval for the mean
/// let mean_fn = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
/// 
/// let (lower, upper) = sampling::bootstrap_confidence_interval(
///     &data, 1000, &mean_fn, 0.95
/// ).unwrap();
/// 
/// println!("95% CI for mean: ({}, {})", lower, upper);
/// ```
pub fn bootstrap_confidence_interval<F>(
    data: &[f64],
    n_samples: usize,
    statistic_fn: &F,
    confidence_level: f64
) -> Result<(f64, f64)>
where
    F: Fn(&[f64]) -> f64
{
    if data.is_empty() {
        return Err(Error::EmptyData("Bootstrap requires data".into()));
    }
    
    if n_samples < 100 {
        return Err(Error::InvalidInput("Recommended to use at least 100 bootstrap samples".into()));
    }
    
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(Error::InvalidInput("Confidence level must be between 0 and 1".into()));
    }
    
    // Generate bootstrap samples
    let bootstrap_samples = bootstrap_impl(data, n_samples)?;
    
    // Calculate statistic for each bootstrap sample
    let mut bootstrap_statistics = Vec::with_capacity(n_samples);
    
    for sample in bootstrap_samples {
        let stat = statistic_fn(&sample);
        bootstrap_statistics.push(stat);
    }
    
    // Sort statistics to find percentiles
    bootstrap_statistics.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Calculate confidence interval bounds
    let alpha = 1.0 - confidence_level;
    let lower_index = (alpha / 2.0 * n_samples as f64).ceil() as usize;
    let upper_index = (n_samples as f64 - alpha / 2.0 * n_samples as f64).floor() as usize - 1;
    
    let lower_bound = bootstrap_statistics.get(lower_index)
        .ok_or_else(|| Error::ComputationError("Failed to compute lower bound".into()))?;
    
    let upper_bound = bootstrap_statistics.get(upper_index)
        .ok_or_else(|| Error::ComputationError("Failed to compute upper bound".into()))?;
    
    Ok((*lower_bound, *upper_bound))
}

/// Perform systematic sampling
///
/// # Description
/// Selects samples at regular intervals.
///
/// # Arguments
/// * `data` - Data to sample from
/// * `k` - Sampling interval (select every kth element)
/// * `offset` - Starting point (0 to k-1)
///
/// # Example
/// ```
/// use pandrs::stats::sampling;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// 
/// // Select every 3rd element starting from the first
/// let sample = sampling::systematic_sample(&data, 3, 0).unwrap();
/// // Result should be [1.0, 4.0, 7.0, 10.0]
/// ```
pub fn systematic_sample<T: Clone>(
    data: &[T],
    k: usize,
    offset: usize
) -> Result<Vec<T>> {
    if data.is_empty() {
        return Err(Error::EmptyData("Cannot sample from empty data".into()));
    }
    
    if k == 0 {
        return Err(Error::InvalidInput("Sampling interval must be greater than 0".into()));
    }
    
    if offset >= k {
        return Err(Error::InvalidInput(
            format!("Offset must be between 0 and k-1 (k={})", k)
        ));
    }
    
    let mut sample = Vec::new();
    let mut i = offset;
    
    while i < data.len() {
        sample.push(data[i].clone());
        i += k;
    }
    
    if sample.is_empty() {
        return Err(Error::InvalidInput("No samples were selected".into()));
    }
    
    Ok(sample)
}

/// Perform weighted sampling
///
/// # Description
/// Samples data with probabilities proportional to weights.
///
/// # Arguments
/// * `data` - Data to sample from
/// * `weights` - Sampling weights (higher values increase selection probability)
/// * `size` - Number of samples to draw
/// * `replace` - Whether to sample with replacement
///
/// # Example
/// ```
/// use pandrs::stats::sampling;
///
/// let data = vec![10, 20, 30, 40, 50];
/// let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Higher weights = higher probability
/// 
/// // Sample 3 items with weights
/// let sample = sampling::weighted_sample(&data, &weights, 3, true).unwrap();
/// ```
pub fn weighted_sample<T: Clone>(
    data: &[T],
    weights: &[f64],
    size: usize,
    replace: bool
) -> Result<Vec<T>> {
    if data.is_empty() {
        return Err(Error::EmptyData("Cannot sample from empty data".into()));
    }
    
    if weights.len() != data.len() {
        return Err(Error::DimensionMismatch(
            format!("Weights length ({}) must match data length ({})", weights.len(), data.len())
        ));
    }
    
    // Check for negative weights
    for &w in weights {
        if w < 0.0 {
            return Err(Error::InvalidInput("Weights must be non-negative".into()));
        }
    }
    
    // Check if all weights are zero
    if weights.iter().all(|&w| w == 0.0) {
        return Err(Error::InvalidInput("At least one weight must be positive".into()));
    }
    
    // Calculate cumulative weights for weighted selection
    let sum_weights: f64 = weights.iter().sum();
    let mut cum_weights = Vec::with_capacity(weights.len());
    let mut cumulative = 0.0;
    
    for &w in weights {
        cumulative += w / sum_weights;
        cum_weights.push(cumulative);
    }
    
    // Ensure last cumulative weight is exactly 1.0
    if let Some(last) = cum_weights.last_mut() {
        *last = 1.0;
    }
    
    let mut rng = thread_rng();
    let mut sample = Vec::with_capacity(size);
    let mut used_indices = std::collections::HashSet::new();
    
    // Sample with weights
    for _ in 0..size {
        if !replace && used_indices.len() == data.len() {
            // All items already used (when sampling without replacement)
            break;
        }
        
        let r = rng.gen::<f64>(); // Random number between 0 and 1
        
        // Find the index where r falls in the cumulative weights
        let mut selected_idx = 0;
        while selected_idx < cum_weights.len() - 1 && r > cum_weights[selected_idx] {
            selected_idx += 1;
        }
        
        if !replace {
            // Sampling without replacement: ensure each item is selected at most once
            if used_indices.contains(&selected_idx) {
                // Find the next unused index
                let mut found = false;
                for i in (selected_idx + 1)..data.len() {
                    if !used_indices.contains(&i) {
                        selected_idx = i;
                        found = true;
                        break;
                    }
                }
                
                if !found {
                    for i in 0..selected_idx {
                        if !used_indices.contains(&i) {
                            selected_idx = i;
                            found = true;
                            break;
                        }
                    }
                }
                
                if !found {
                    // All items used, shouldn't normally reach here
                    break;
                }
            }
            
            used_indices.insert(selected_idx);
        }
        
        sample.push(data[selected_idx].clone());
    }
    
    // Check if sample size matches requested size
    if !replace && size > data.len() {
        // Warning: without replacement, can't sample more than the data size
    }
    
    if sample.is_empty() {
        return Err(Error::InvalidInput("No samples were selected".into()));
    }
    
    Ok(sample)
}

/// Calculate bootstrap standard error
///
/// # Description
/// Computes the standard error of a statistic using bootstrap resampling.
///
/// # Arguments
/// * `data` - The data to bootstrap
/// * `n_samples` - Number of bootstrap samples to generate
/// * `statistic_fn` - Function to compute the statistic of interest
///
/// # Example
/// ```
/// use pandrs::stats::sampling;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// 
/// // Compute bootstrap standard error for the mean
/// let mean_fn = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
/// 
/// let std_error = sampling::bootstrap_standard_error(&data, 1000, &mean_fn).unwrap();
/// println!("Standard error of the mean: {}", std_error);
/// ```
pub fn bootstrap_standard_error<F>(
    data: &[f64],
    n_samples: usize,
    statistic_fn: &F
) -> Result<f64>
where
    F: Fn(&[f64]) -> f64
{
    if data.is_empty() {
        return Err(Error::EmptyData("Bootstrap requires data".into()));
    }
    
    if n_samples < 100 {
        return Err(Error::InvalidInput("Recommended to use at least 100 bootstrap samples".into()));
    }
    
    // Generate bootstrap samples
    let bootstrap_samples = bootstrap_impl(data, n_samples)?;
    
    // Calculate statistic for each bootstrap sample
    let mut bootstrap_statistics = Vec::with_capacity(n_samples);
    
    for sample in bootstrap_samples {
        let stat = statistic_fn(&sample);
        bootstrap_statistics.push(stat);
    }
    
    // Calculate mean of bootstrap statistics
    let mean: f64 = bootstrap_statistics.iter().sum::<f64>() / n_samples as f64;
    
    // Calculate standard deviation (standard error)
    let variance: f64 = bootstrap_statistics.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n_samples as f64;
    
    let std_error = variance.sqrt();
    
    Ok(std_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;
    
    #[test]
    fn test_sample_impl() {
        // Create a DataFrame for testing
        let mut df = DataFrame::new();
        
        // Add some columns
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        df.add_column("values".to_string(), 
                     Series::new(values, Some("values".to_string())).unwrap()).unwrap();
        
        // Test sampling with replacement
        let sample_with_replacement = sample_impl(&df, 0.5, true).unwrap();
        assert_eq!(sample_with_replacement.nrows(), 5);
        
        // Test sampling without replacement
        let sample_without_replacement = sample_impl(&df, 0.5, false).unwrap();
        assert_eq!(sample_without_replacement.nrows(), 5);
        
        // Test invalid fraction
        let result_invalid = sample_impl(&df, 1.5, false);
        assert!(result_invalid.is_err());
    }
    
    #[test]
    fn test_bootstrap_impl() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test bootstrap resampling
        let bootstrap_samples = bootstrap_impl(&data, 10).unwrap();
        
        assert_eq!(bootstrap_samples.len(), 10);
        for sample in bootstrap_samples {
            assert_eq!(sample.len(), data.len());
            
            // Each value in sample should be from the original data
            for &value in &sample {
                assert!(data.contains(&value));
            }
        }
        
        // Test invalid parameters
        let result_empty = bootstrap_impl(&[], 10);
        assert!(result_empty.is_err());
        
        let result_zero = bootstrap_impl(&data, 0);
        assert!(result_zero.is_err());
    }
    
    #[test]
    fn test_stratified_sample() {
        // Create a DataFrame for testing
        let mut df = DataFrame::new();
        
        // Add a stratification column and a values column
        let strata = vec!["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"]
            .into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
            
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        df.add_column("strata".to_string(), 
                     Series::new(strata, Some("strata".to_string())).unwrap()).unwrap();
        df.add_column("values".to_string(), 
                     Series::new(values, Some("values".to_string())).unwrap()).unwrap();
        
        // Create sampling fractions
        let mut fractions = HashMap::new();
        fractions.insert("A".to_string(), 0.5); // 2 out of 4
        fractions.insert("B".to_string(), 0.75); // 3 out of 4
        fractions.insert("C".to_string(), 1.0); // 2 out of 2
        
        // Test stratified sampling
        let stratified_sample = stratified_sample(&df, "strata", &fractions, false).unwrap();
        
        // Count number of samples from each stratum
        let sampled_strata = stratified_sample.get_column("strata").unwrap();
        let strata_values = sampled_strata.as_str().unwrap();
        
        let mut counts = HashMap::new();
        for stratum in strata_values {
            *counts.entry(stratum.to_string()).or_insert(0) += 1;
        }
        
        // Check if counts match expected
        assert_eq!(*counts.get("A").unwrap_or(&0), 2);
        assert_eq!(*counts.get("B").unwrap_or(&0), 3);
        assert_eq!(*counts.get("C").unwrap_or(&0), 2);
    }
    
    #[test]
    fn test_systematic_sample() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        // Test systematic sampling with k=3, offset=0
        let sys_sample = systematic_sample(&data, 3, 0).unwrap();
        assert_eq!(sys_sample, vec![1.0, 4.0, 7.0, 10.0]);
        
        // Test with offset=1
        let sys_sample_offset = systematic_sample(&data, 3, 1).unwrap();
        assert_eq!(sys_sample_offset, vec![2.0, 5.0, 8.0]);
        
        // Test invalid parameters
        let result_invalid_k = systematic_sample(&data, 0, 0);
        assert!(result_invalid_k.is_err());
        
        let result_invalid_offset = systematic_sample(&data, 3, 3);
        assert!(result_invalid_offset.is_err());
    }
    
    #[test]
    fn test_weighted_sample() {
        let data = vec![1, 2, 3, 4, 5];
        
        // All items equal weight
        let equal_weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let sample_equal = weighted_sample(&data, &equal_weights, 10, true).unwrap();
        assert_eq!(sample_equal.len(), 10);
        
        // Weighted sample - all weight on first item
        let biased_weights = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let sample_biased = weighted_sample(&data, &biased_weights, 5, true).unwrap();
        assert_eq!(sample_biased, vec![1, 1, 1, 1, 1]);
        
        // Test sampling without replacement
        let sample_no_replace = weighted_sample(&data, &equal_weights, 5, false).unwrap();
        assert_eq!(sample_no_replace.len(), 5);
        
        // Check that all items are unique when sampling without replacement
        let mut unique_items = std::collections::HashSet::new();
        for item in &sample_no_replace {
            unique_items.insert(item);
        }
        assert_eq!(unique_items.len(), 5);
        
        // Test invalid parameters
        let invalid_weights = vec![1.0, 1.0]; // Length mismatch
        let result_invalid = weighted_sample(&data, &invalid_weights, 3, true);
        assert!(result_invalid.is_err());
        
        let negative_weights = vec![-1.0, 1.0, 1.0, 1.0, 1.0]; // Negative weight
        let result_negative = weighted_sample(&data, &negative_weights, 3, true);
        assert!(result_negative.is_err());
    }
    
    #[test]
    fn test_bootstrap_confidence_interval() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        // Function to compute the mean
        let mean_fn = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        
        // Compute 95% confidence interval for the mean
        let (lower, upper) = bootstrap_confidence_interval(&data, 1000, &mean_fn, 0.95).unwrap();
        
        // True mean is 5.5
        assert!(lower <= 5.5 && upper >= 5.5);
        assert!(lower < upper); // Lower bound should be less than upper bound
        
        // Test invalid parameters
        let result_empty = bootstrap_confidence_interval(&[], 1000, &mean_fn, 0.95);
        assert!(result_empty.is_err());
        
        let result_invalid_cl = bootstrap_confidence_interval(&data, 1000, &mean_fn, 1.1);
        assert!(result_invalid_cl.is_err());
    }
    
    #[test]
    fn test_bootstrap_standard_error() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        // Function to compute the mean
        let mean_fn = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
        
        // Compute bootstrap standard error for the mean
        let std_error = bootstrap_standard_error(&data, 1000, &mean_fn).unwrap();
        
        // Standard error should be positive
        assert!(std_error > 0.0);
        
        // Theoretical standard error of the mean: σ/√n
        // σ ≈ 2.87 for this data, n = 10, so σ/√n ≈ 0.91
        assert!((std_error - 0.91).abs() < 0.2);
        
        // Test invalid parameters
        let result_empty = bootstrap_standard_error(&[], 1000, &mean_fn);
        assert!(result_empty.is_err());
    }
}