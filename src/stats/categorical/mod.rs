//! Statistics module for categorical data
//!
//! This module provides specialized statistical functions for working with categorical data.
//! It includes contingency tables, chi-square tests, and other categorical data analyses.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::dataframe::DataFrame;
use crate::error::{Error, PandRSError, Result};
use crate::series::{Categorical, CategoricalOrder, StringCategorical};
use crate::stats::ChiSquareResult;

/// ContingencyTable represents a cross-tabulation of categorical data
#[derive(Debug, Clone)]
pub struct ContingencyTable {
    /// The observed frequencies in the table
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
    /// Create a new contingency table from observed frequencies
    pub fn new(
        observed: Vec<Vec<f64>>,
        row_labels: Option<Vec<String>>,
        col_labels: Option<Vec<String>>,
    ) -> Result<Self> {
        if observed.is_empty() {
            return Err(Error::InsufficientData(
                "Observed frequencies cannot be empty".into(),
            ));
        }

        let rows = observed.len();
        let cols = observed[0].len();

        // Check if all rows have the same number of columns
        for row in &observed {
            if row.len() != cols {
                return Err(Error::Consistency(
                    "All rows must have the same number of columns".into(),
                ));
            }
        }

        // Generate default row labels if not provided
        let row_labels = match row_labels {
            Some(labels) => {
                if labels.len() != rows {
                    return Err(Error::Consistency(format!(
                        "Number of row labels ({}) does not match number of rows ({})",
                        labels.len(),
                        rows
                    )));
                }
                labels
            }
            None => (0..rows).map(|i| format!("Row_{}", i)).collect(),
        };

        // Generate default column labels if not provided
        let col_labels = match col_labels {
            Some(labels) => {
                if labels.len() != cols {
                    return Err(Error::Consistency(format!(
                        "Number of column labels ({}) does not match number of columns ({})",
                        labels.len(),
                        cols
                    )));
                }
                labels
            }
            None => (0..cols).map(|i| format!("Col_{}", i)).collect(),
        };

        // Calculate row totals
        let mut row_totals = vec![0.0; rows];
        for i in 0..rows {
            row_totals[i] = observed[i].iter().sum();
        }

        // Calculate column totals
        let mut col_totals = vec![0.0; cols];
        for j in 0..cols {
            col_totals[j] = (0..rows).map(|i| observed[i][j]).sum();
        }

        // Calculate grand total
        let total = row_totals.iter().sum();

        Ok(ContingencyTable {
            observed,
            row_labels,
            col_labels,
            row_totals,
            col_totals,
            total,
        })
    }

    /// Calculate expected frequencies under independence assumption
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

    /// Calculate the chi-square statistic for this contingency table
    pub fn chi_square(&self, alpha: f64) -> Result<ChiSquareResult> {
        let expected = self.expected_frequencies();

        // Calculate chi-square statistic
        let mut chi2 = 0.0;
        for i in 0..self.observed.len() {
            for j in 0..self.observed[0].len() {
                let o = self.observed[i][j];
                let e = expected[i][j];

                // Skip cells with expected frequency of 0
                if e == 0.0 {
                    continue;
                }

                chi2 += (o - e).powi(2) / e;
            }
        }

        // Calculate degrees of freedom
        let df = (self.observed.len() - 1) * (self.observed[0].len() - 1);

        // Use the existing chi_square_test function to calculate p-value and significance
        crate::stats::inference::chi_square_test_with_statistic(chi2, df, alpha, expected)
    }

    /// Calculate Cramer's V - a measure of association between categorical variables
    pub fn cramers_v(&self) -> Result<f64> {
        let chi2_result = self.chi_square(0.05)?;
        let n = self.total;
        let min_dim = (self.observed.len() - 1).min(self.observed[0].len() - 1);

        if min_dim == 0 {
            return Err(Error::Consistency(
                "Cannot calculate Cramer's V with only one row or column".into(),
            ));
        }

        let v = (chi2_result.chi2_statistic / (n * min_dim as f64)).sqrt();

        Ok(v)
    }
}

/// Create a contingency table from two categorical variables
pub fn contingency_table<T, U>(
    cat1: &Categorical<T>,
    cat2: &Categorical<U>,
) -> Result<ContingencyTable>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
    U: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    // Get the categories for each variable
    let cat1_categories = cat1.categories();
    let cat2_categories = cat2.categories();

    let rows = cat1_categories.len();
    let cols = cat2_categories.len();

    // Initialize the frequency table
    let mut observed = vec![vec![0.0; cols]; rows];

    // Ensure both categorical variables have the same length
    if cat1.len() != cat2.len() {
        return Err(Error::Consistency(format!(
            "Categorical variables must have the same length, got {} and {}",
            cat1.len(),
            cat2.len()
        )));
    }

    // Count frequencies
    for i in 0..cat1.len() {
        // Skip if either value is missing (NA)
        let cat1_code = cat1.codes()[i];
        let cat2_code = cat2.codes()[i];

        if cat1_code == -1 || cat2_code == -1 {
            continue;
        }

        observed[cat1_code as usize][cat2_code as usize] += 1.0;
    }

    // Convert categories to strings for labels
    let row_labels = cat1_categories.iter().map(|c| format!("{}", c)).collect();
    let col_labels = cat2_categories.iter().map(|c| format!("{}", c)).collect();

    ContingencyTable::new(observed, Some(row_labels), Some(col_labels))
}

/// Create a contingency table from two categorical columns in a DataFrame
pub fn dataframe_contingency_table(
    df: &DataFrame,
    col1: &str,
    col2: &str,
) -> Result<ContingencyTable> {
    // Check if both columns exist and are categorical
    if !df.contains_column(col1) {
        return Err(Error::Column(format!("Column '{}' does not exist", col1)));
    }

    if !df.contains_column(col2) {
        return Err(Error::Column(format!("Column '{}' does not exist", col2)));
    }

    if !df.is_categorical(col1) {
        return Err(Error::Consistency(format!(
            "Column '{}' is not categorical",
            col1
        )));
    }

    if !df.is_categorical(col2) {
        return Err(Error::Consistency(format!(
            "Column '{}' is not categorical",
            col2
        )));
    }

    // Get categorical data
    let cat1 = df.get_categorical::<String>(col1)?;
    let cat2 = df.get_categorical::<String>(col2)?;

    // Create contingency table
    contingency_table(&cat1, &cat2)
}

/// Calculate the chi-square test for independence between two categorical variables
pub fn chi_square_test_for_independence<T, U>(
    cat1: &Categorical<T>,
    cat2: &Categorical<U>,
    alpha: f64,
) -> Result<ChiSquareResult>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
    U: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    let table = contingency_table(cat1, cat2)?;
    table.chi_square(alpha)
}

/// Calculate the chi-square test for independence between two categorical columns in a DataFrame
pub fn dataframe_chi_square_test(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    alpha: f64,
) -> Result<ChiSquareResult> {
    let table = dataframe_contingency_table(df, col1, col2)?;
    table.chi_square(alpha)
}

/// Calculate Cramer's V (measure of association between categorical variables)
pub fn cramers_v<T, U>(cat1: &Categorical<T>, cat2: &Categorical<U>) -> Result<f64>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
    U: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    let table = contingency_table(cat1, cat2)?;
    table.cramers_v()
}

/// Calculate Cramer's V between two categorical columns in a DataFrame
pub fn dataframe_cramers_v(df: &DataFrame, col1: &str, col2: &str) -> Result<f64> {
    let table = dataframe_contingency_table(df, col1, col2)?;
    table.cramers_v()
}

/// Test for association between categorical variable and numeric variable using ANOVA
pub fn categorical_anova<T>(
    cat: &Categorical<T>,
    numeric_values: &[f64],
    alpha: f64,
) -> Result<crate::stats::AnovaResult>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    // Ensure categorical and numeric variables have the same length
    if cat.len() != numeric_values.len() {
        return Err(Error::Consistency(format!(
            "Categorical and numeric variables must have the same length, got {} and {}",
            cat.len(),
            numeric_values.len()
        )));
    }

    // Group numeric values by category
    let mut groups: HashMap<&T, Vec<f64>> = HashMap::new();

    for i in 0..cat.len() {
        let code = cat.codes()[i];

        // Skip if the categorical value is missing (NA)
        if code == -1 {
            continue;
        }

        let category = &cat.categories()[code as usize];
        groups
            .entry(category)
            .or_insert_with(Vec::new)
            .push(numeric_values[i]);
    }

    // Convert to the format required by the ANOVA function
    // First collect all the strings, then build the HashMap
    let mut cat_strings: Vec<String> = Vec::new();
    let mut groups_vec: Vec<(String, Vec<f64>)> = Vec::new();

    for (cat_val, values) in groups {
        let cat_str = format!("{}", cat_val);
        groups_vec.push((cat_str, values));
    }

    // Now create the HashMap with string refs
    let mut anova_groups: HashMap<&str, Vec<f64>> = HashMap::new();

    // Add strings to cat_strings first
    for (cat_str, _) in &groups_vec {
        cat_strings.push(cat_str.clone());
    }

    // Now we can create the HashMap with references to the strings
    for i in 0..groups_vec.len() {
        anova_groups.insert(&cat_strings[i], groups_vec[i].1.clone());
    }

    // Perform ANOVA
    crate::stats::anova(&anova_groups, alpha)
}

/// Test for association between categorical column and numeric column in a DataFrame using ANOVA
pub fn dataframe_categorical_anova(
    df: &DataFrame,
    cat_col: &str,
    numeric_col: &str,
    alpha: f64,
) -> Result<crate::stats::AnovaResult> {
    // Check if columns exist
    if !df.contains_column(cat_col) {
        return Err(Error::Column(format!(
            "Column '{}' does not exist",
            cat_col
        )));
    }

    if !df.contains_column(numeric_col) {
        return Err(Error::Column(format!(
            "Column '{}' does not exist",
            numeric_col
        )));
    }

    // Check if categorical column is categorical
    if !df.is_categorical(cat_col) {
        return Err(Error::Consistency(format!(
            "Column '{}' is not categorical",
            cat_col
        )));
    }

    // Get categorical data
    let cat = df.get_categorical::<String>(cat_col)?;

    // Get numeric values
    let numeric_values = df.get_column_numeric_values(numeric_col)?;

    // Simplified numeric column handling
    if !df.is_numeric_column(numeric_col) {
        return Err(Error::InvalidValue(format!(
            "Column '{}' is not numeric",
            numeric_col
        )));
    }

    // Perform ANOVA
    categorical_anova(&cat, &numeric_values, alpha)
}

/// Calculate mode (most frequent category) for categorical data
pub fn mode<T>(cat: &Categorical<T>) -> Result<Option<T>>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    let mut counts = vec![0; cat.categories().len()];

    // Count occurrences of each category
    for &code in cat.codes() {
        if code != -1 {
            // Skip NA values
            counts[code as usize] += 1;
        }
    }

    // Find the category with the highest count
    let mut max_count = 0;
    let mut mode_index = None;

    for (i, &count) in counts.iter().enumerate() {
        if count > max_count {
            max_count = count;
            mode_index = Some(i);
        }
    }

    // Return the mode category
    match mode_index {
        Some(index) => Ok(Some(cat.categories()[index].clone())),
        None => Ok(None), // No mode found (all values are NA)
    }
}

/// Calculate entropy of a categorical variable
///
/// Entropy measures the uncertainty or randomness in a categorical variable.
/// Higher entropy indicates more uniformly distributed categories.
pub fn entropy<T>(cat: &Categorical<T>) -> Result<f64>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    let len = cat.len() as f64;

    // Count occurrences of each category
    let mut counts = vec![0; cat.categories().len()];
    let mut valid_count = 0;

    for &code in cat.codes() {
        if code != -1 {
            // Skip NA values
            counts[code as usize] += 1;
            valid_count += 1;
        }
    }

    // Calculate entropy
    if valid_count == 0 {
        return Ok(0.0); // All values are NA
    }

    let valid_count_f64 = valid_count as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / valid_count_f64;
            entropy -= p * p.log2();
        }
    }

    Ok(entropy)
}

/// Calculate mutual information between two categorical variables
///
/// Mutual information measures the amount of information obtained about one random
/// variable through observing the other random variable.
pub fn mutual_information<T, U>(cat1: &Categorical<T>, cat2: &Categorical<U>) -> Result<f64>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
    U: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    // Get contingency table
    let table = contingency_table(cat1, cat2)?;

    // Calculate joint probabilities and marginal probabilities
    let mut mutual_info = 0.0;
    let n = table.total;

    for i in 0..table.observed.len() {
        for j in 0..table.observed[0].len() {
            let p_xy = table.observed[i][j] / n;

            if p_xy > 0.0 {
                let p_x = table.row_totals[i] / n;
                let p_y = table.col_totals[j] / n;

                mutual_info += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }
    }

    Ok(mutual_info)
}

/// Calculate normalized mutual information between two categorical variables
///
/// Normalized mutual information is mutual information divided by the square root
/// of the product of the entropies, resulting in a value between 0 and 1.
pub fn normalized_mutual_information<T, U>(
    cat1: &Categorical<T>,
    cat2: &Categorical<U>,
) -> Result<f64>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
    U: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    let mi = mutual_information(cat1, cat2)?;
    let h1 = entropy(cat1)?;
    let h2 = entropy(cat2)?;

    if h1 == 0.0 || h2 == 0.0 {
        return Ok(0.0);
    }

    Ok(mi / (h1 * h2).sqrt())
}

/// Calculate normalized mutual information between two categorical columns in a DataFrame
pub fn dataframe_normalized_mutual_information(
    df: &DataFrame,
    col1: &str,
    col2: &str,
) -> Result<f64> {
    let cat1 = df.get_categorical::<String>(col1)?;
    let cat2 = df.get_categorical::<String>(col2)?;

    normalized_mutual_information(&cat1, &cat2)
}

/// Calculate the frequency distribution of a categorical variable
///
/// Returns a HashMap with categories as keys and frequencies (counts) as values
pub fn frequency_distribution<T>(cat: &Categorical<T>) -> Result<HashMap<T, usize>>
where
    T: Debug + Clone + Eq + Hash + std::fmt::Display + Ord,
{
    let mut freq_dist = HashMap::new();

    // Count occurrences of each category
    for i in 0..cat.len() {
        let code = cat.codes()[i];

        // Skip NA values
        if code == -1 {
            continue;
        }

        let category = &cat.categories()[code as usize];
        *freq_dist.entry(category.clone()).or_insert(0) += 1;
    }

    Ok(freq_dist)
}
