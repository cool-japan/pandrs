//! Categorical data statistics module
//!
//! This module provides functions for analyzing categorical data, including
//! contingency tables, association measures, and categorical hypothesis tests.

use crate::error::{Result, Error};
use crate::dataframe::DataFrame;
use crate::stats::{ChiSquareResult, AnovaResult};
use std::collections::{HashMap, HashSet};
use std::f64::consts::LN_2;
use std::fmt::Debug;

/// ContingencyTable represents a cross-tabulation of categorical data
#[derive(Debug, Clone)]
pub struct ContingencyTable {
    /// Observed frequencies
    pub observed: Vec<Vec<f64>>,
    /// Row labels
    pub row_labels: Vec<String>,
    /// Column labels
    pub col_labels: Vec<String>,
    /// Row totals
    pub row_totals: Vec<f64>,
    /// Column totals
    pub col_totals: Vec<f64>,
    /// Grand total
    pub total: f64,
}

impl ContingencyTable {
    /// Creates a new empty contingency table
    pub fn new() -> Self {
        ContingencyTable {
            observed: Vec::new(),
            row_labels: Vec::new(),
            col_labels: Vec::new(),
            row_totals: Vec::new(),
            col_totals: Vec::new(),
            total: 0.0,
        }
    }
    
    /// Creates a contingency table from observed data
    pub fn from_observed(
        observed: Vec<Vec<f64>>,
        row_labels: Vec<String>,
        col_labels: Vec<String>,
    ) -> Result<Self> {
        if observed.is_empty() {
            return Err(Error::EmptyData("Contingency table requires data".into()));
        }
        
        let rows = observed.len();
        let cols = observed[0].len();
        
        if row_labels.len() != rows {
            return Err(Error::DimensionMismatch(
                format!("Number of row labels ({}) must match number of rows ({})", 
                       row_labels.len(), rows)
            ));
        }
        
        if col_labels.len() != cols {
            return Err(Error::DimensionMismatch(
                format!("Number of column labels ({}) must match number of columns ({})", 
                       col_labels.len(), cols)
            ));
        }
        
        // Ensure all rows have the same number of columns
        for (i, row) in observed.iter().enumerate() {
            if row.len() != cols {
                return Err(Error::DimensionMismatch(
                    format!("Row {} has {} columns, expected {}", i, row.len(), cols)
                ));
            }
        }
        
        // Calculate row and column totals
        let mut row_totals = vec![0.0; rows];
        let mut col_totals = vec![0.0; cols];
        let mut total = 0.0;
        
        for i in 0..rows {
            for j in 0..cols {
                let value = observed[i][j];
                
                if value < 0.0 {
                    return Err(Error::InvalidInput("Contingency table cannot contain negative values".into()));
                }
                
                row_totals[i] += value;
                col_totals[j] += value;
                total += value;
            }
        }
        
        Ok(ContingencyTable {
            observed,
            row_labels,
            col_labels,
            row_totals,
            col_totals,
            total,
        })
    }
    
    /// Calculates the expected frequencies under independence assumption
    pub fn expected_frequencies(&self) -> Vec<Vec<f64>> {
        let rows = self.observed.len();
        let cols = self.observed[0].len();
        
        let mut expected = vec![vec![0.0; cols]; rows];
        
        for i in 0..rows {
            for j in 0..cols {
                expected[i][j] = self.row_totals[i] * self.col_totals[j] / self.total;
            }
        }
        
        expected
    }
    
    /// Performs chi-square test of independence
    pub fn chi_square_test(&self, alpha: f64) -> Result<ChiSquareResult> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(Error::InvalidInput("Significance level (alpha) must be between 0 and 1".into()));
        }
        
        // Calculate expected frequencies
        let expected = self.expected_frequencies();
        
        // Calculate chi-square statistic
        let mut chi2_stat = 0.0;
        let rows = self.observed.len();
        let cols = self.observed[0].len();
        
        for i in 0..rows {
            for j in 0..cols {
                if expected[i][j] < 1e-10 {
                    return Err(Error::ComputationError("Expected frequency too small".into()));
                }
                
                let diff = self.observed[i][j] - expected[i][j];
                chi2_stat += diff * diff / expected[i][j];
            }
        }
        
        // Degrees of freedom = (rows-1) * (cols-1)
        let df = (rows - 1) * (cols - 1);
        
        // Calculate p-value
        let p_value = chi2_to_pvalue(chi2_stat, df);
        
        Ok(ChiSquareResult {
            chi2_statistic: chi2_stat,
            p_value,
            df,
            significant: p_value < alpha,
            expected_freq: expected,
        })
    }
    
    /// Calculates Cramer's V (measure of association between categorical variables)
    pub fn cramers_v(&self) -> Result<f64> {
        let rows = self.observed.len();
        let cols = self.observed[0].len();
        
        if rows < 2 || cols < 2 {
            return Err(Error::InvalidInput("Cramer's V requires at least 2x2 contingency table".into()));
        }
        
        // Run chi-square test (using alpha=0.05, but it doesn't matter for V calculation)
        let chi_square_result = self.chi_square_test(0.05)?;
        
        // Cramer's V = sqrt(χ² / (n * min(r-1, c-1)))
        let n = self.total;
        let min_dim = (rows - 1).min(cols - 1);
        
        if n < 1.0 || min_dim == 0 {
            return Err(Error::ComputationError("Invalid dimensions for Cramer's V".into()));
        }
        
        let v = (chi_square_result.chi2_statistic / (n * min_dim as f64)).sqrt();
        
        Ok(v)
    }
    
    /// Calculates mutual information (measure of statistical dependence)
    pub fn mutual_information(&self) -> Result<f64> {
        let rows = self.observed.len();
        let cols = self.observed[0].len();
        
        if rows < 2 || cols < 2 {
            return Err(Error::InvalidInput("Mutual information requires at least 2x2 contingency table".into()));
        }
        
        // Calculate expected frequencies
        let expected = self.expected_frequencies();
        
        // Calculate mutual information: I(X;Y) = Σₓ Σᵧ P(x,y) log(P(x,y) / (P(x) P(y)))
        let mut mi = 0.0;
        
        for i in 0..rows {
            for j in 0..cols {
                let p_xy = self.observed[i][j] / self.total;
                
                // Skip zero probabilities (0 * log(0) = 0)
                if p_xy < 1e-10 {
                    continue;
                }
                
                let p_x = self.row_totals[i] / self.total;
                let p_y = self.col_totals[j] / self.total;
                
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
        
        // Convert from natural log to base 2 (bits)
        mi /= LN_2;
        
        Ok(mi)
    }
    
    /// Calculates normalized mutual information (ranges from 0 to 1)
    pub fn normalized_mutual_information(&self) -> Result<f64> {
        let mi = self.mutual_information()?;
        
        // Calculate entropies
        let mut h_x = 0.0;
        let mut h_y = 0.0;
        
        for i in 0..self.row_totals.len() {
            let p = self.row_totals[i] / self.total;
            if p > 0.0 {
                h_x -= p * p.ln();
            }
        }
        
        for j in 0..self.col_totals.len() {
            let p = self.col_totals[j] / self.total;
            if p > 0.0 {
                h_y -= p * p.ln();
            }
        }
        
        // Convert from natural log to base 2 (bits)
        h_x /= LN_2;
        h_y /= LN_2;
        
        // Normalize by arithmetic mean of entropies
        if h_x < 1e-10 || h_y < 1e-10 {
            return Ok(0.0);
        }
        
        let nmi = 2.0 * mi / (h_x + h_y);
        
        Ok(nmi)
    }
}

/// Implementation for DataFrame column contingency table
pub(crate) fn dataframe_contingency_table(
    df: &DataFrame,
    col1: &str,
    col2: &str
) -> Result<ContingencyTable> {
    // Check if columns exist
    if !df.has_column(col1) {
        return Err(Error::InvalidColumn(format!("Column '{}' does not exist", col1)));
    }
    
    if !df.has_column(col2) {
        return Err(Error::InvalidColumn(format!("Column '{}' does not exist", col2)));
    }
    
    // Get columns as string
    let series1 = df.get_column(col1)?;
    let series2 = df.get_column(col2)?;
    
    let values1 = series1.as_str()?;
    let values2 = series2.as_str()?;
    
    if values1.len() != values2.len() {
        return Err(Error::DimensionMismatch(
            format!("Column lengths do not match: {} has length {}, {} has length {}", 
                   col1, values1.len(), col2, values2.len())
        ));
    }
    
    // Get unique values for each column
    let mut unique1 = HashSet::new();
    let mut unique2 = HashSet::new();
    
    for value in &values1 {
        unique1.insert(value.clone());
    }
    
    for value in &values2 {
        unique2.insert(value.clone());
    }
    
    // Convert to sorted vectors for consistent ordering
    let mut row_labels: Vec<_> = unique1.iter().cloned().collect();
    let mut col_labels: Vec<_> = unique2.iter().cloned().collect();
    
    row_labels.sort();
    col_labels.sort();
    
    // Create maps for faster lookup
    let row_map: HashMap<_, _> = row_labels.iter()
        .enumerate()
        .map(|(i, label)| (label.clone(), i))
        .collect();
    
    let col_map: HashMap<_, _> = col_labels.iter()
        .enumerate()
        .map(|(i, label)| (label.clone(), i))
        .collect();
    
    // Create and fill contingency table
    let rows = row_labels.len();
    let cols = col_labels.len();
    let mut observed = vec![vec![0.0; cols]; rows];
    
    for (val1, val2) in values1.iter().zip(values2.iter()) {
        let row = row_map[val1];
        let col = col_map[val2];
        observed[row][col] += 1.0;
    }
    
    // Create contingency table
    ContingencyTable::from_observed(observed, row_labels, col_labels)
}

/// Implementation for chi-square test of independence between DataFrame columns
pub(crate) fn dataframe_chi_square_test(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    alpha: f64
) -> Result<ChiSquareResult> {
    let contingency_table = dataframe_contingency_table(df, col1, col2)?;
    contingency_table.chi_square_test(alpha)
}

/// Implementation for Cramer's V between DataFrame columns
pub(crate) fn dataframe_cramers_v(
    df: &DataFrame,
    col1: &str,
    col2: &str
) -> Result<f64> {
    let contingency_table = dataframe_contingency_table(df, col1, col2)?;
    contingency_table.cramers_v()
}

/// Implementation for normalized mutual information between DataFrame columns
pub(crate) fn dataframe_normalized_mutual_information(
    df: &DataFrame,
    col1: &str,
    col2: &str
) -> Result<f64> {
    let contingency_table = dataframe_contingency_table(df, col1, col2)?;
    contingency_table.normalized_mutual_information()
}

/// Implementation for ANOVA test between categorical and numeric columns
pub(crate) fn dataframe_categorical_anova(
    df: &DataFrame,
    cat_col: &str,
    numeric_col: &str,
    alpha: f64
) -> Result<AnovaResult> {
    // Check if columns exist
    if !df.has_column(cat_col) {
        return Err(Error::InvalidColumn(format!("Column '{}' does not exist", cat_col)));
    }
    
    if !df.has_column(numeric_col) {
        return Err(Error::InvalidColumn(format!("Column '{}' does not exist", numeric_col)));
    }
    
    // Get categorical column as string
    let cat_series = df.get_column(cat_col)?;
    let cat_values = cat_series.as_str()?;
    
    // Get numeric column as float
    let num_series = df.get_column(numeric_col)?;
    let num_values = num_series.as_f64()?;
    
    if cat_values.len() != num_values.len() {
        return Err(Error::DimensionMismatch(
            format!("Column lengths do not match: {} has length {}, {} has length {}", 
                   cat_col, cat_values.len(), numeric_col, num_values.len())
        ));
    }
    
    // Group numeric values by category
    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
    
    for (cat, num) in cat_values.iter().zip(num_values.iter()) {
        groups.entry(cat.clone())
            .or_insert_with(Vec::new)
            .push(*num);
    }
    
    // Prepare groups for ANOVA
    let mut anova_groups: HashMap<&str, &[f64]> = HashMap::new();
    for (cat, values) in &groups {
        anova_groups.insert(cat.as_str(), values.as_slice());
    }
    
    // Perform ANOVA
    anova_impl(&anova_groups, alpha)
}

/// Simple chi-square to p-value conversion (similar to the one in inference.rs)
fn chi2_to_pvalue(chi2: f64, df: usize) -> f64 {
    // Simplified implementation (more accurate calculation needed in practice)
    let k = df as f64 / 2.0;
    let x = chi2 / 2.0;
    
    // Approximation calculation for gamma function
    let gamma_k = if df % 2 == 0 {
        1.0 // k is integer
    } else {
        std::f64::consts::PI.sqrt() // k + 0.5 is integer
    };
    
    // Approximation calculation for lower incomplete gamma function
    let p = if chi2 > df as f64 + 2.0 {
        1.0 - gamma_k * (1.0 - x.exp() * (1.0 + x + 0.5 * x.powi(2)))
    } else {
        gamma_k * x.exp() * x.powf(k - 1.0)
    };
    
    1.0 - p.min(1.0).max(0.0)
}

/// Implementation for one-way ANOVA (copy from inference.rs to avoid circular dependency)
fn anova_impl(
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

/// F-distribution CDF approximation (copy from inference.rs to avoid circular dependency)
fn f_distribution_cdf(f: f64, df1: usize, df2: usize) -> f64 {
    // Approximation calculation for F-distribution
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

/// Calculate mode (most frequent value) for categorical data
///
/// # Description
/// Finds the most frequent value(s) in a categorical variable.
///
/// # Arguments
/// * `data` - The categorical data
///
/// # Returns
/// * A vector of the most frequent values (can be multiple in case of ties)
///
/// # Example
/// ```
/// use pandrs::stats::categorical;
///
/// let data = vec!["A", "B", "A", "C", "B", "A"];
/// let mode = categorical::mode(&data).unwrap();
/// println!("Mode: {:?}", mode); // Should be ["A"]
/// ```
pub fn mode<T: AsRef<str> + Clone>(data: &[T]) -> Result<Vec<String>> {
    if data.is_empty() {
        return Err(Error::EmptyData("Mode calculation requires data".into()));
    }
    
    // Count frequencies
    let mut counts: HashMap<String, usize> = HashMap::new();
    
    for item in data {
        let key = item.as_ref().to_string();
        *counts.entry(key).or_insert(0) += 1;
    }
    
    // Find maximum frequency
    let max_count = counts.values().max().unwrap_or(&0);
    
    // Find all categories with maximum frequency
    let modes: Vec<String> = counts.iter()
        .filter(|(_, &count)| count == *max_count)
        .map(|(category, _)| category.clone())
        .collect();
    
    Ok(modes)
}

/// Calculate entropy of categorical data
///
/// # Description
/// Calculates the Shannon entropy of a categorical variable.
///
/// # Arguments
/// * `data` - The categorical data
///
/// # Returns
/// * Entropy value in bits (base 2)
///
/// # Example
/// ```
/// use pandrs::stats::categorical;
///
/// let data = vec!["A", "B", "A", "C", "B", "A"];
/// let entropy = categorical::entropy(&data).unwrap();
/// println!("Entropy: {}", entropy);
/// ```
pub fn entropy<T: AsRef<str>>(data: &[T]) -> Result<f64> {
    if data.is_empty() {
        return Err(Error::EmptyData("Entropy calculation requires data".into()));
    }
    
    // Count frequencies
    let mut counts: HashMap<String, usize> = HashMap::new();
    let n = data.len();
    
    for item in data {
        let key = item.as_ref().to_string();
        *counts.entry(key).or_insert(0) += 1;
    }
    
    // Calculate entropy
    let mut entropy = 0.0;
    
    for &count in counts.values() {
        let p = count as f64 / n as f64;
        entropy -= p * p.ln();
    }
    
    // Convert from natural log to base 2 (bits)
    entropy /= LN_2;
    
    Ok(entropy)
}

/// Calculate frequency distribution of categorical data
///
/// # Description
/// Computes the frequency and relative frequency of each category.
///
/// # Arguments
/// * `data` - The categorical data
///
/// # Returns
/// * A map with category as key and (frequency, relative frequency) as value
///
/// # Example
/// ```
/// use pandrs::stats::categorical;
///
/// let data = vec!["A", "B", "A", "C", "B", "A"];
/// let freq = categorical::frequency_distribution(&data).unwrap();
/// 
/// for (category, (count, percentage)) in &freq {
///     println!("{}: {} ({:.1}%)", category, count, percentage * 100.0);
/// }
/// ```
pub fn frequency_distribution<T: AsRef<str>>(
    data: &[T]
) -> Result<HashMap<String, (usize, f64)>> {
    if data.is_empty() {
        return Err(Error::EmptyData("Frequency calculation requires data".into()));
    }
    
    // Count frequencies
    let mut counts: HashMap<String, usize> = HashMap::new();
    let n = data.len();
    
    for item in data {
        let key = item.as_ref().to_string();
        *counts.entry(key).or_insert(0) += 1;
    }
    
    // Calculate relative frequencies
    let mut distribution = HashMap::new();
    
    for (category, count) in counts {
        let relative_freq = count as f64 / n as f64;
        distribution.insert(category, (count, relative_freq));
    }
    
    Ok(distribution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;
    
    #[test]
    fn test_contingency_table() {
        // Create a 2x3 contingency table
        let observed = vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ];
        
        let row_labels = vec!["Row1".to_string(), "Row2".to_string()];
        let col_labels = vec!["Col1".to_string(), "Col2".to_string(), "Col3".to_string()];
        
        let table = ContingencyTable::from_observed(observed, row_labels, col_labels).unwrap();
        
        // Check row totals
        assert_eq!(table.row_totals, vec![60.0, 150.0]);
        
        // Check column totals
        assert_eq!(table.col_totals, vec![50.0, 70.0, 90.0]);
        
        // Check total
        assert_eq!(table.total, 210.0);
        
        // Check expected frequencies
        let expected = table.expected_frequencies();
        assert_eq!(expected.len(), 2);
        assert_eq!(expected[0].len(), 3);
        
        // Expected[0][0] = (60.0 * 50.0) / 210.0 = 14.29
        assert!((expected[0][0] - 14.29).abs() < 0.01);
    }
    
    #[test]
    fn test_chi_square_test() {
        // Create a contingency table with clear association
        let observed = vec![
            vec![40.0, 10.0],
            vec![10.0, 40.0],
        ];
        
        let row_labels = vec!["Row1".to_string(), "Row2".to_string()];
        let col_labels = vec!["Col1".to_string(), "Col2".to_string()];
        
        let table = ContingencyTable::from_observed(observed, row_labels, col_labels).unwrap();
        
        // Chi-square test
        let result = table.chi_square_test(0.05).unwrap();
        
        // Should be significant (p < 0.05)
        assert!(result.chi2_statistic > 0.0);
        assert!(result.p_value < 0.05);
        assert!(result.significant);
        assert_eq!(result.df, 1); // (2-1) * (2-1) = 1
    }
    
    #[test]
    fn test_cramers_v() {
        // Create a contingency table with perfect association
        let observed = vec![
            vec![10.0, 0.0],
            vec![0.0, 10.0],
        ];
        
        let row_labels = vec!["Row1".to_string(), "Row2".to_string()];
        let col_labels = vec!["Col1".to_string(), "Col2".to_string()];
        
        let table = ContingencyTable::from_observed(observed, row_labels, col_labels).unwrap();
        
        // Cramer's V
        let v = table.cramers_v().unwrap();
        
        // Perfect association should give V = 1
        assert!((v - 1.0).abs() < 0.01);
        
        // Create a contingency table with no association
        let observed_no_assoc = vec![
            vec![10.0, 10.0],
            vec![10.0, 10.0],
        ];
        
        let table_no_assoc = ContingencyTable::from_observed(
            observed_no_assoc, 
            row_labels.clone(), 
            col_labels.clone()
        ).unwrap();
        
        let v_no_assoc = table_no_assoc.cramers_v().unwrap();
        
        // No association should give V = 0
        assert!(v_no_assoc < 0.01);
    }
    
    #[test]
    fn test_mutual_information() {
        // Create a contingency table with perfect association
        let observed = vec![
            vec![10.0, 0.0],
            vec![0.0, 10.0],
        ];
        
        let row_labels = vec!["Row1".to_string(), "Row2".to_string()];
        let col_labels = vec!["Col1".to_string(), "Col2".to_string()];
        
        let table = ContingencyTable::from_observed(observed, row_labels, col_labels).unwrap();
        
        // Mutual information
        let mi = table.mutual_information().unwrap();
        
        // Perfect association should give MI = 1 bit for balanced 2x2 table
        assert!((mi - 1.0).abs() < 0.01);
        
        // Normalized mutual information should be 1
        let nmi = table.normalized_mutual_information().unwrap();
        assert!((nmi - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_dataframe_contingency_table() {
        // Create a DataFrame for testing
        let mut df = DataFrame::new();
        
        // Add two categorical columns
        let cat1 = vec!["A", "A", "A", "B", "B", "B"]
            .into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
            
        let cat2 = vec!["X", "X", "Y", "X", "Y", "Y"]
            .into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
        
        df.add_column("cat1".to_string(), 
                     Series::new(cat1, Some("cat1".to_string())).unwrap()).unwrap();
        df.add_column("cat2".to_string(), 
                     Series::new(cat2, Some("cat2".to_string())).unwrap()).unwrap();
        
        // Create contingency table
        let table = dataframe_contingency_table(&df, "cat1", "cat2").unwrap();
        
        // Check table dimensions
        assert_eq!(table.observed.len(), 2); // 2 rows (A, B)
        assert_eq!(table.observed[0].len(), 2); // 2 columns (X, Y)
        
        // Check frequencies
        assert_eq!(table.observed[0][0], 2.0); // A, X
        assert_eq!(table.observed[0][1], 1.0); // A, Y
        assert_eq!(table.observed[1][0], 1.0); // B, X
        assert_eq!(table.observed[1][1], 2.0); // B, Y
    }
    
    #[test]
    fn test_categorical_anova() {
        // Create a DataFrame for testing
        let mut df = DataFrame::new();
        
        // Add categorical and numeric columns
        let cat = vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"]
            .into_iter().map(|s| s.to_string()).collect::<Vec<_>>();
            
        // Values depend on category (A: ~1, B: ~5, C: ~10)
        let num = vec![1.0, 1.1, 0.9, 4.9, 5.1, 5.0, 9.8, 10.1, 10.1];
        
        df.add_column("category".to_string(), 
                     Series::new(cat, Some("category".to_string())).unwrap()).unwrap();
        df.add_column("value".to_string(), 
                     Series::new(num, Some("value".to_string())).unwrap()).unwrap();
        
        // Perform ANOVA
        let result = dataframe_categorical_anova(&df, "category", "value", 0.05).unwrap();
        
        // The categories have clearly different means, so F should be large and p small
        assert!(result.f_statistic > 100.0);
        assert!(result.p_value < 0.001);
        assert!(result.significant);
    }
    
    #[test]
    fn test_mode() {
        // Single mode
        let data = vec!["A", "B", "A", "C", "B", "A"];
        let result = mode(&data).unwrap();
        assert_eq!(result, vec!["A"]);
        
        // Multiple modes (tie)
        let data_tie = vec!["A", "B", "A", "B", "C", "C"];
        let result_tie = mode(&data_tie).unwrap();
        assert_eq!(result_tie.len(), 2);
        assert!(result_tie.contains(&"A".to_string()));
        assert!(result_tie.contains(&"B".to_string()));
    }
    
    #[test]
    fn test_entropy() {
        // Uniform distribution (maximum entropy)
        let uniform = vec!["A", "B", "C", "D"];
        let entropy_uniform = entropy(&uniform).unwrap();
        assert!((entropy_uniform - 2.0).abs() < 0.01); // log2(4) = 2 bits
        
        // Single value (minimum entropy)
        let single = vec!["A", "A", "A", "A"];
        let entropy_single = entropy(&single).unwrap();
        assert!(entropy_single < 0.01); // Should be 0
        
        // Mixed distribution
        let mixed = vec!["A", "A", "B", "C", "C", "C", "C"];
        let entropy_mixed = entropy(&mixed).unwrap();
        assert!(entropy_mixed > 0.0 && entropy_mixed < 2.0);
    }
    
    #[test]
    fn test_frequency_distribution() {
        let data = vec!["A", "B", "A", "C", "B", "A"];
        let freq = frequency_distribution(&data).unwrap();
        
        assert_eq!(freq.len(), 3); // Three categories: A, B, C
        
        // Check frequencies
        let (a_count, a_freq) = freq.get("A").unwrap();
        assert_eq!(*a_count, 3);
        assert!((a_freq - 0.5).abs() < 0.01); // 3/6 = 0.5
        
        let (b_count, b_freq) = freq.get("B").unwrap();
        assert_eq!(*b_count, 2);
        assert!((b_freq - 0.333).abs() < 0.01); // 2/6 = 0.333
        
        let (c_count, c_freq) = freq.get("C").unwrap();
        assert_eq!(*c_count, 1);
        assert!((c_freq - 0.167).abs() < 0.01); // 1/6 = 0.167
    }
}