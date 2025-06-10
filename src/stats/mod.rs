//! PandRS Statistics Module
//!
//! This module provides statistical functionality for data analysis.
//! It implements a wide range of statistical methods including descriptive statistics,
//! inferential statistics, hypothesis testing, and regression analysis.

// Feature modules
pub mod categorical;
pub mod descriptive;
pub mod inference;
pub mod regression;
pub mod sampling;

// GPU-accelerated statistical functions (conditionally compiled)
#[cfg(feature = "cuda")]
pub mod gpu;

// Re-export public types and functions
use crate::dataframe::DataFrame;
use crate::error::{Error, PandRSError, Result};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Structure holding descriptive statistics results
#[derive(Debug, Clone)]
pub struct DescriptiveStats {
    /// Number of data points
    pub count: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation (unbiased estimator)
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// 25% quantile
    pub q1: f64,
    /// Median (50% quantile)
    pub median: f64,
    /// 75% quantile
    pub q3: f64,
    /// Maximum value
    pub max: f64,
}

/// T-test result
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// t-statistic
    pub statistic: f64,
    /// p-value
    pub pvalue: f64,
    /// Whether significant at given significance level
    pub significant: bool,
    /// Degrees of freedom
    pub df: usize,
}

/// Linear regression model results
#[derive(Debug, Clone)]
pub struct LinearRegressionResult {
    /// Intercept
    pub intercept: f64,
    /// Coefficients (multiple for multivariate regression)
    pub coefficients: Vec<f64>,
    /// Coefficient of determination (R²)
    pub r_squared: f64,
    /// Adjusted coefficient of determination
    pub adj_r_squared: f64,
    /// p-values for each coefficient
    pub p_values: Vec<f64>,
    /// Fitted values
    pub fitted_values: Vec<f64>,
    /// Residuals
    pub residuals: Vec<f64>,
}

/// One-way ANOVA results
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// F-statistic
    pub f_statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Between-groups sum of squares
    pub ss_between: f64,
    /// Within-groups sum of squares
    pub ss_within: f64,
    /// Total sum of squares
    pub ss_total: f64,
    /// Between-groups degrees of freedom
    pub df_between: usize,
    /// Within-groups degrees of freedom
    pub df_within: usize,
    /// Total degrees of freedom
    pub df_total: usize,
    /// Between-groups mean square
    pub ms_between: f64,
    /// Within-groups mean square
    pub ms_within: f64,
    /// Whether significant at given significance level
    pub significant: bool,
}

/// Mann-Whitney U test (non-parametric test) results
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// U-statistic
    pub u_statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Whether significant at given significance level
    pub significant: bool,
}

/// Chi-square test results
#[derive(Debug, Clone)]
pub struct ChiSquareResult {
    /// Chi-square statistic
    pub chi2_statistic: f64,
    /// p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Whether significant at given significance level
    pub significant: bool,
    /// Expected frequencies
    pub expected_freq: Vec<Vec<f64>>,
}

// Public API functions

/// Calculate basic statistics for data
///
/// # Description
/// This function calculates basic descriptive statistics (mean, standard deviation,
/// minimum, maximum, etc.) for a Series, DataFrame, or other numeric data.
///
/// # Example
/// ```rust
/// use pandrs::stats;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let stats = stats::describe(&data).unwrap();
/// println!("Mean: {}", stats.mean);
/// println!("Standard deviation: {}", stats.std);
/// ```
pub fn describe<T: AsRef<[f64]>>(data: T) -> Result<DescriptiveStats> {
    descriptive::describe_impl(data.as_ref())
}

/// Calculate correlation coefficient
///
/// # Description
/// Calculates the Pearson correlation coefficient between two numeric arrays.
/// The correlation coefficient ranges from -1 to 1, where 1 indicates perfect positive correlation,
/// -1 indicates perfect negative correlation, and 0 indicates no correlation.
///
/// # Example
/// ```rust
/// use pandrs::stats;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
/// let corr = stats::correlation(&x, &y).unwrap();
/// println!("Correlation coefficient: {}", corr);
/// ```
pub fn correlation<T: AsRef<[f64]>, U: AsRef<[f64]>>(x: T, y: U) -> Result<f64> {
    descriptive::correlation_impl(x.as_ref(), y.as_ref())
}

/// Calculate covariance
///
/// # Description
/// Calculates the covariance between two numeric arrays.
/// Covariance is a measure of how much two variables change together.
///
/// # Example
/// ```rust
/// use pandrs::stats;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![2.0, 4.0, 5.0, 4.0, 5.0];
/// let cov = stats::covariance(&x, &y).unwrap();
/// println!("Covariance: {}", cov);
/// ```
pub fn covariance<T: AsRef<[f64]>, U: AsRef<[f64]>>(x: T, y: U) -> Result<f64> {
    descriptive::covariance_impl(x.as_ref(), y.as_ref())
}

/// Perform two-sample t-test
///
/// # Description
/// Tests if there is a significant difference between the means of two independent samples.
/// The test can be performed assuming equal or unequal variances.
///
/// # Example
/// ```rust
/// use pandrs::stats;
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
/// // t-test assuming equal variances, significance level 0.05
/// let result = stats::ttest(&sample1, &sample2, 0.05, true).unwrap();
/// println!("t-statistic: {}", result.statistic);
/// println!("p-value: {}", result.pvalue);
/// println!("Significant difference: {}", result.significant);
/// ```
pub fn ttest<T: AsRef<[f64]>, U: AsRef<[f64]>>(
    sample1: T,
    sample2: U,
    alpha: f64,
    equal_var: bool,
) -> Result<TTestResult> {
    inference::ttest_impl(sample1.as_ref(), sample2.as_ref(), alpha, equal_var)
}

/// Perform linear regression analysis
///
/// # Description
/// Performs simple or multiple linear regression analysis to build a least squares linear model.
///
/// # Example
/// ```rust,no_run
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
/// use pandrs::series::Series;
///
/// // Create DataFrame
/// let mut df = DataFrame::new();
/// // Add data
/// df.add_column("x1".to_string(), Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("x1".to_string())).unwrap()).unwrap();
/// df.add_column("x2".to_string(), Series::new(vec![2.0, 3.0, 4.0, 5.0, 6.0], Some("x2".to_string())).unwrap()).unwrap();
/// df.add_column("y".to_string(), Series::new(vec![3.0, 5.0, 7.0, 9.0, 11.0], Some("y".to_string())).unwrap()).unwrap();
///
/// // Regression analysis with y as target, x1 and x2 as predictors
/// let model = stats::linear_regression(&df, "y", &["x1", "x2"]).unwrap();
/// println!("Intercept: {}", model.intercept);
/// println!("Coefficients: {:?}", model.coefficients);
/// println!("R-squared: {}", model.r_squared);
/// ```
pub fn linear_regression(
    df: &DataFrame,
    y_column: &str,
    x_columns: &[&str],
) -> Result<LinearRegressionResult> {
    regression::linear_regression_impl(df, y_column, x_columns)
}

/// Perform random sampling
///
/// # Description
/// Gets a random sample of specified size.
///
/// # Example
/// ```rust
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// let df = DataFrame::new(); // DataFrame with data
/// // Get a 10% random sample
/// let sampled_df = stats::sample(&df, 0.1, true).unwrap();
/// ```
pub fn sample(df: &DataFrame, fraction: f64, replace: bool) -> Result<DataFrame> {
    sampling::sample_impl(df, fraction, replace)
}

/// Generate bootstrap samples
///
/// # Description
/// Performs bootstrap resampling.
/// Used for statistical inference such as confidence interval estimation.
///
/// # Example
/// ```rust
/// use pandrs::stats;
///
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// // 1000 bootstrap samples
/// let bootstrap_samples = stats::bootstrap(&data, 1000).unwrap();
/// ```
pub fn bootstrap<T: AsRef<[f64]>>(data: T, n_samples: usize) -> Result<Vec<Vec<f64>>> {
    sampling::bootstrap_impl(data.as_ref(), n_samples)
}

/// Perform one-way ANOVA
///
/// # Description
/// Tests if there are significant differences between the means of three or more groups.
/// Uses the ratio of between-group variance to within-group variance (F-value).
///
/// # Example
/// ```rust
/// use pandrs::stats;
/// use std::collections::HashMap;
///
/// let mut groups = HashMap::new();
/// groups.insert("Group A", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// groups.insert("Group B", vec![2.0, 3.0, 4.0, 5.0, 6.0]);
/// groups.insert("Group C", vec![3.0, 4.0, 5.0, 6.0, 7.0]);
///
/// // ANOVA test with significance level 0.05
/// let result = stats::anova(&groups, 0.05).unwrap();
/// println!("F-statistic: {}", result.f_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Significant difference: {}", result.significant);
/// ```
pub fn anova<T: AsRef<[f64]>>(groups: &HashMap<&str, T>, alpha: f64) -> Result<AnovaResult> {
    if groups.len() < 2 {
        return Err(Error::InsufficientData(
            "At least 2 groups are needed for ANOVA".into(),
        ));
    }

    // Delegate implementation to inference module
    let groups_converted: HashMap<&str, &[f64]> =
        groups.iter().map(|(k, v)| (*k, v.as_ref())).collect();

    inference::anova_impl(&groups_converted, alpha)
}

/// Perform Mann-Whitney U test (non-parametric test)
///
/// # Description
/// A non-parametric test that can be used instead of t-test when the data doesn't follow
/// a normal distribution. Compares the distributions of two independent samples.
///
/// # Example
/// ```rust
/// use pandrs::stats;
///
/// let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let sample2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];
/// // Mann-Whitney U test with significance level 0.05
/// let result = stats::mann_whitney_u(&sample1, &sample2, 0.05).unwrap();
/// println!("U-statistic: {}", result.u_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Significant difference: {}", result.significant);
/// ```
pub fn mann_whitney_u<T: AsRef<[f64]>, U: AsRef<[f64]>>(
    sample1: T,
    sample2: U,
    alpha: f64,
) -> Result<MannWhitneyResult> {
    inference::mann_whitney_u_impl(sample1.as_ref(), sample2.as_ref(), alpha)
}

/// Perform chi-square test
///
/// # Description
/// Tests the association between categorical variables.
/// Evaluates whether variables are independent based on the difference between
/// observed and expected values.
///
/// # Example
/// ```rust
/// use pandrs::stats;
///
/// // 2x2 contingency table (observed values)
/// let observed = vec![
///     vec![20.0, 30.0],
///     vec![25.0, 25.0]
/// ];
/// // Chi-square test with significance level 0.05
/// let result = stats::chi_square_test(&observed, 0.05).unwrap();
/// println!("Chi-square statistic: {}", result.chi2_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Significant difference: {}", result.significant);
/// ```
pub fn chi_square_test(observed: &[Vec<f64>], alpha: f64) -> Result<ChiSquareResult> {
    inference::chi_square_test_impl(observed, alpha)
}

// Categorical data statistical functions

/// ContingencyTable represents a cross-tabulation of categorical data
pub use categorical::ContingencyTable;

/// Create a contingency table from two categorical columns in a DataFrame
///
/// # Description
/// Creates a contingency table showing the joint distribution of two categorical variables.
/// The table contains observed frequencies for each combination of categories.
///
/// # Example
/// ```rust,no_run
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// let df = DataFrame::new(); // DataFrame with categorical data
/// let table = stats::contingency_table_from_df(&df, "category1", "category2").unwrap();
/// println!("Observed frequencies: {:?}", table.observed);
/// ```
pub fn contingency_table_from_df(
    df: &DataFrame,
    col1: &str,
    col2: &str,
) -> Result<ContingencyTable> {
    categorical::dataframe_contingency_table(df, col1, col2)
}

/// Calculate chi-square test for independence between two categorical columns
///
/// # Description
/// Tests if there is a significant association between two categorical variables.
///
/// # Example
/// ```rust,no_run
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// let df = DataFrame::new(); // DataFrame with categorical data
/// let result = stats::chi_square_independence(&df, "category1", "category2", 0.05).unwrap();
/// println!("Chi-square statistic: {}", result.chi2_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Significant association: {}", result.significant);
/// ```
pub fn chi_square_independence(
    df: &DataFrame,
    col1: &str,
    col2: &str,
    alpha: f64,
) -> Result<ChiSquareResult> {
    categorical::dataframe_chi_square_test(df, col1, col2, alpha)
}

/// Calculate Cramer's V measure of association between categorical columns
///
/// # Description
/// Cramer's V is a measure of association between categorical variables, based on chi-square.
/// It ranges from 0 (no association) to 1 (perfect association).
///
/// # Example
/// ```rust,no_run
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// let df = DataFrame::new(); // DataFrame with categorical data
/// let v = stats::cramers_v_from_df(&df, "category1", "category2").unwrap();
/// println!("Cramer's V: {}", v);
/// ```
pub fn cramers_v_from_df(df: &DataFrame, col1: &str, col2: &str) -> Result<f64> {
    categorical::dataframe_cramers_v(df, col1, col2)
}

/// Test association between categorical and numeric variables using ANOVA
///
/// # Description
/// Tests if different categories in a categorical variable have significantly different
/// mean values in a numeric variable.
///
/// # Example
/// ```rust,no_run
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// let df = DataFrame::new(); // DataFrame with categorical and numeric data
/// let result = stats::categorical_anova_from_df(&df, "category", "numeric_val", 0.05).unwrap();
/// println!("F-statistic: {}", result.f_statistic);
/// println!("p-value: {}", result.p_value);
/// println!("Significant difference: {}", result.significant);
/// ```
pub fn categorical_anova_from_df(
    df: &DataFrame,
    cat_col: &str,
    numeric_col: &str,
    alpha: f64,
) -> Result<AnovaResult> {
    categorical::dataframe_categorical_anova(df, cat_col, numeric_col, alpha)
}

/// Calculate mutual information between categorical variables
///
/// # Description
/// Mutual information measures how much information is shared between two categorical variables.
/// Higher values indicate stronger association.
///
/// # Example
/// ```rust,no_run
/// use pandrs::stats;
/// use pandrs::dataframe::DataFrame;
///
/// let df = DataFrame::new(); // DataFrame with categorical data
/// let nmi = stats::normalized_mutual_info(&df, "category1", "category2").unwrap();
/// println!("Normalized Mutual Information: {}", nmi);
/// ```
pub fn normalized_mutual_info(df: &DataFrame, col1: &str, col2: &str) -> Result<f64> {
    categorical::dataframe_normalized_mutual_information(df, col1, col2)
}

// TODO: Re-export functions once they are implemented
// For now, we'll comment these out since the implementation files might have been reorganized

// Descriptive statistics functions
// pub use descriptive::variance;
// pub use descriptive::std_dev;
// pub use descriptive::quantile;
// These are probably implemented under different names or in different modules
pub use descriptive::correlation_impl as correlation_matrix;

// Inference statistics functions
// pub use inference::one_sample_ttest;
// pub use inference::paired_ttest;

// Regression functions - using existing functions instead
pub use regression::linear_regression as simple_linear_regression;
// pub use regression::polynomial_regression;
// pub use regression::residual_diagnostics;

// Sampling functions - existing implementation
pub use sampling::stratified_sample_impl as stratified_sample;
// pub use sampling::bootstrap_confidence_interval;
// pub use sampling::systematic_sample;
// pub use sampling::weighted_sample;
// pub use sampling::bootstrap_standard_error;

pub use categorical::entropy;
pub use categorical::frequency_distribution;
pub use categorical::mode;

// Re-export GPU-accelerated functions when CUDA is enabled
#[cfg(feature = "cuda")]
pub use gpu::{
    correlation_matrix as gpu_correlation_matrix, covariance_matrix as gpu_covariance_matrix,
    describe_gpu, feature_importance, kmeans, linear_regression as gpu_linear_regression, pca,
};
