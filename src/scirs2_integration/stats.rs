//! Advanced statistical functions using SciRS2's implementations.
//!
//! All types and functions in this module are gated behind the `scirs2` feature flag.

#[cfg(feature = "scirs2")]
use ndarray::Array2;
#[cfg(feature = "scirs2")]
use scirs2_core::ndarray::ArrayView1;

#[cfg(feature = "scirs2")]
use crate::core::error::{Error, Result};
#[cfg(feature = "scirs2")]
use crate::dataframe::DataFrame;
#[cfg(feature = "scirs2")]
use crate::scirs2_integration::conversion::{array2_to_dataframe, dataframe_to_array2};
#[cfg(feature = "scirs2")]
use crate::series::Series;

/// Result of a PCA decomposition.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Principal components as a DataFrame (columns are components)
    pub components: DataFrame,
    /// Variance explained by each component
    pub explained_variance: Vec<f64>,
    /// Fraction of variance explained by each component
    pub explained_variance_ratio: Vec<f64>,
}

/// Result of a t-test.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct TTestResult {
    /// The t-statistic
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: f64,
}

/// Result of a one-way ANOVA.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct AnovaResult {
    /// The F-statistic
    pub f_statistic: f64,
    /// The p-value
    pub p_value: f64,
}

/// Advanced statistical functions using SciRS2's implementations.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "scirs2")]
/// # {
/// use pandrs::{DataFrame, Series};
/// use pandrs::scirs2_integration::stats::SciRS2Stats;
///
/// let mut df = DataFrame::new();
/// df.add_column("a".to_string(),
///     Series::new(vec![1.0f64, 2.0, 3.0, 4.0, 5.0], Some("a".to_string())).expect("ok"))
///     .expect("ok");
/// df.add_column("b".to_string(),
///     Series::new(vec![2.0f64, 4.0, 6.0, 8.0, 10.0], Some("b".to_string())).expect("ok"))
///     .expect("ok");
///
/// let corr = SciRS2Stats::correlation_matrix(&df, &["a", "b"]).expect("corr");
/// # }
/// ```
#[cfg(feature = "scirs2")]
pub struct SciRS2Stats;

#[cfg(feature = "scirs2")]
impl SciRS2Stats {
    /// Compute descriptive statistics for selected columns using SciRS2.
    ///
    /// Returns a DataFrame with statistics as rows and columns as columns.
    /// Statistics computed: count, mean, std, min, 25%, 50%, 75%, max.
    ///
    /// # Arguments
    ///
    /// * `df` - The source DataFrame
    /// * `columns` - Column names to include in the description
    ///
    /// # Errors
    ///
    /// Returns an error if any column cannot be converted to numeric values.
    pub fn describe(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        use scirs2_stats::{mean, median, std, var};

        let stat_names = vec![
            "count".to_string(),
            "mean".to_string(),
            "std".to_string(),
            "min".to_string(),
            "25%".to_string(),
            "50%".to_string(),
            "75%".to_string(),
            "max".to_string(),
        ];

        let mut result_df = DataFrame::new();

        // Add stat labels column
        let stat_series = Series::new(stat_names.clone(), Some("statistic".to_string()))?;
        result_df.add_column("statistic".to_string(), stat_series)?;

        for &col_name in columns {
            let values = df.get_column_numeric_values(col_name)?;
            if values.is_empty() {
                return Err(Error::EmptyData(format!(
                    "Column '{}' has no numeric values",
                    col_name
                )));
            }

            let arr = scirs2_core::ndarray::Array1::from(values.clone());
            let view = arr.view();

            let count = values.len() as f64;
            let mean_val = mean(&view)
                .map_err(|e| Error::OperationFailed(format!("SciRS2 mean failed: {}", e)))?;
            let std_val = std(&view, 1, None)
                .map_err(|e| Error::OperationFailed(format!("SciRS2 std failed: {}", e)))?;

            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let min_val = sorted[0];
            let max_val = sorted[sorted.len() - 1];
            let q1 = Self::percentile_sorted(&sorted, 25.0);
            let q2 = Self::percentile_sorted(&sorted, 50.0);
            let q3 = Self::percentile_sorted(&sorted, 75.0);

            let stat_values = vec![count, mean_val, std_val, min_val, q1, q2, q3, max_val];
            let col_series = Series::new(stat_values, Some(col_name.to_string()))?;
            result_df.add_column(col_name.to_string(), col_series)?;
        }

        Ok(result_df)
    }

    /// Compute the Pearson correlation matrix for selected columns using SciRS2.
    ///
    /// Returns a DataFrame where both rows (as a "column" label column) and columns
    /// correspond to the input column names.
    ///
    /// # Arguments
    ///
    /// * `df` - The source DataFrame
    /// * `columns` - Column names to include in the correlation matrix
    ///
    /// # Errors
    ///
    /// Returns an error if any column cannot be converted to numeric values.
    pub fn correlation_matrix(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        use scirs2_stats::corrcoef;

        let arr = dataframe_to_array2(df, columns)?;
        let arr_t = arr.t().to_owned(); // corrcoef expects (n_vars, n_obs) in some implementations

        // corrcoef expects rows = observations, columns = variables
        let corr = corrcoef::<f64, _>(&arr, "pearson")
            .map_err(|e| Error::OperationFailed(format!("SciRS2 corrcoef failed: {}", e)))?;

        let col_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();

        // Build result DataFrame: first add a "column" label column, then each correlation column
        let mut result_df = DataFrame::new();
        let label_series = Series::new(col_names.clone(), Some("column".to_string()))?;
        result_df.add_column("column".to_string(), label_series)?;

        let n = columns.len();
        for (col_idx, col_name) in columns.iter().enumerate() {
            let corr_values: Vec<f64> = (0..n).map(|row| corr[[row, col_idx]]).collect();
            let series = Series::new(corr_values, Some(col_name.to_string()))?;
            result_df.add_column(col_name.to_string(), series)?;
        }

        Ok(result_df)
    }

    /// Perform Principal Component Analysis using SciRS2.
    ///
    /// # Arguments
    ///
    /// * `df` - The source DataFrame
    /// * `columns` - Numeric columns to use for PCA
    /// * `n_components` - Number of principal components to extract
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be converted or the PCA fails.
    pub fn pca(df: &DataFrame, columns: &[&str], n_components: usize) -> Result<PcaResult> {
        use scirs2_stats::{pca_memory_efficient, AdvancedMemoryManager, MemoryConstraints};

        let arr = dataframe_to_array2(df, columns)?;
        let (n_rows, n_cols) = arr.dim();

        if n_components > n_cols.min(n_rows) {
            return Err(Error::InvalidInput(format!(
                "n_components ({}) cannot exceed min(n_rows={}, n_cols={})",
                n_components, n_rows, n_cols
            )));
        }

        let constraints = MemoryConstraints {
            max_memory_bytes: 1024 * 1024 * 256, // 256 MB
            ..MemoryConstraints::default()
        };
        let mut manager = AdvancedMemoryManager::new(constraints);

        let pca_result = pca_memory_efficient(&arr.view(), Some(n_components), &mut manager)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 PCA failed: {}", e)))?;

        // Extract explained variance
        let explained_var: Vec<f64> = pca_result.explained_variance.iter().copied().collect();
        let total_var: f64 = explained_var.iter().sum();
        let explained_var_ratio: Vec<f64> = if total_var > 0.0 {
            explained_var.iter().map(|v| v / total_var).collect()
        } else {
            vec![0.0; explained_var.len()]
        };

        // Convert components matrix to DataFrame
        let component_names: Vec<String> =
            (0..n_components).map(|i| format!("PC{}", i + 1)).collect();

        let components_df = array2_to_dataframe(&pca_result.components, component_names)?;

        Ok(PcaResult {
            components: components_df,
            explained_variance: explained_var,
            explained_variance_ratio: explained_var_ratio,
        })
    }

    /// Perform a one-sample t-test using SciRS2.
    ///
    /// Tests if the mean of `data` differs from `popmean`.
    ///
    /// # Arguments
    ///
    /// * `data` - The sample data
    /// * `popmean` - The hypothesized population mean
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty or the test fails.
    pub fn ttest_1samp(data: &[f64], popmean: f64) -> Result<TTestResult> {
        use scirs2_stats::tests::ttest::{ttest_1samp, Alternative};

        if data.is_empty() {
            return Err(Error::EmptyData(
                "t-test requires non-empty data".to_string(),
            ));
        }

        let arr = scirs2_core::ndarray::Array1::from(data.to_vec());
        let result = ttest_1samp(&arr.view(), popmean, Alternative::TwoSided, "propagate")
            .map_err(|e| Error::OperationFailed(format!("SciRS2 ttest_1samp failed: {}", e)))?;

        Ok(TTestResult {
            statistic: result.statistic,
            p_value: result.pvalue,
            df: result.df,
        })
    }

    /// Perform an independent two-sample t-test using SciRS2.
    ///
    /// Tests if the means of two independent samples differ.
    ///
    /// # Arguments
    ///
    /// * `a` - First sample
    /// * `b` - Second sample
    ///
    /// # Errors
    ///
    /// Returns an error if either sample is empty or the test fails.
    pub fn ttest_ind(a: &[f64], b: &[f64]) -> Result<TTestResult> {
        use scirs2_stats::tests::ttest::{ttest_ind, Alternative};

        if a.is_empty() || b.is_empty() {
            return Err(Error::EmptyData(
                "t-test requires non-empty samples".to_string(),
            ));
        }

        let arr_a = scirs2_core::ndarray::Array1::from(a.to_vec());
        let arr_b = scirs2_core::ndarray::Array1::from(b.to_vec());

        // equal_var=true uses Student's t-test, false uses Welch's t-test
        let result = ttest_ind(
            &arr_a.view(),
            &arr_b.view(),
            false, // use Welch's t-test by default
            Alternative::TwoSided,
            "propagate",
        )
        .map_err(|e| Error::OperationFailed(format!("SciRS2 ttest_ind failed: {}", e)))?;

        Ok(TTestResult {
            statistic: result.statistic,
            p_value: result.pvalue,
            df: result.df,
        })
    }

    /// Perform a one-way ANOVA using SciRS2.
    ///
    /// Tests if the means of multiple groups are equal.
    ///
    /// # Arguments
    ///
    /// * `groups` - Slice of groups (each group is a slice of f64)
    ///
    /// # Errors
    ///
    /// Returns an error if any group is empty or the test fails.
    pub fn f_oneway(groups: &[&[f64]]) -> Result<AnovaResult> {
        use scirs2_stats::tests::anova::one_way_anova;

        if groups.is_empty() {
            return Err(Error::InvalidInput(
                "ANOVA requires at least one group".to_string(),
            ));
        }
        for (i, g) in groups.iter().enumerate() {
            if g.is_empty() {
                return Err(Error::EmptyData(format!("Group {} is empty", i)));
            }
        }

        let arrays: Vec<scirs2_core::ndarray::Array1<f64>> = groups
            .iter()
            .map(|g| scirs2_core::ndarray::Array1::from(g.to_vec()))
            .collect();

        let views: Vec<scirs2_core::ndarray::ArrayView1<f64>> =
            arrays.iter().map(|a| a.view()).collect();
        let group_views: Vec<&scirs2_core::ndarray::ArrayView1<f64>> = views.iter().collect();

        let result = one_way_anova(&group_views)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 one_way_anova failed: {}", e)))?;

        Ok(AnovaResult {
            f_statistic: result.f_statistic,
            p_value: result.p_value,
        })
    }

    // --- Internal helpers ---

    /// Compute a percentile from pre-sorted data using linear interpolation.
    fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return f64::NAN;
        }
        let n = sorted.len();
        if n == 1 {
            return sorted[0];
        }
        let index = p / 100.0 * (n - 1) as f64;
        let lo = index.floor() as usize;
        let hi = index.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let frac = index - lo as f64;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }
}
