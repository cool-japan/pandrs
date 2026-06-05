//! Advanced statistical functions using SciRS2's implementations.
//!
//! All types and functions in this module are gated behind the `scirs2` feature flag.

#[cfg(feature = "scirs2")]
use scirs2_core::ndarray::Array2;
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

/// Result of a chi-square test.
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct Chi2TestResult {
    /// The chi-square statistic
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
    /// Degrees of freedom
    pub dof: usize,
}

/// Result of a normality test (Shapiro-Wilk, KS two-sample).
#[cfg(feature = "scirs2")]
#[derive(Debug, Clone)]
pub struct NormalityTestResult {
    /// The test statistic
    pub statistic: f64,
    /// The p-value
    pub p_value: f64,
    /// Whether the data appears normally distributed (p >= 0.05)
    pub is_normal: bool,
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

    /// Compute the Spearman rank correlation matrix for selected columns.
    ///
    /// Returns a DataFrame where both rows (as a "column" label column) and columns
    /// correspond to the input column names.  Uses SciRS2's `spearmanr` function
    /// for each pair.
    ///
    /// # Arguments
    ///
    /// * `df` - The source DataFrame
    /// * `columns` - Column names to include in the correlation matrix
    ///
    /// # Errors
    ///
    /// Returns an error if any column cannot be converted to numeric values.
    pub fn spearman_correlation_matrix(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        use scirs2_stats::spearmanr;

        let n = columns.len();
        if n < 1 {
            return Err(Error::InvalidInput(
                "Need at least 1 column for Spearman correlation matrix".to_string(),
            ));
        }

        // Collect all column data upfront
        let mut col_data: Vec<Vec<f64>> = Vec::with_capacity(n);
        for &col_name in columns {
            let vals = df.get_column_numeric_values(col_name)?;
            if vals.is_empty() {
                return Err(Error::EmptyData(format!(
                    "Column '{}' has no numeric values",
                    col_name
                )));
            }
            col_data.push(vals);
        }

        // Build the n×n Spearman matrix
        let mut matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let arr_i = scirs2_core::ndarray::Array1::from(col_data[i].clone());
                let arr_j = scirs2_core::ndarray::Array1::from(col_data[j].clone());
                let (rho, _p) = spearmanr::<f64, _>(&arr_i, &arr_j, "two-sided").map_err(|e| {
                    Error::OperationFailed(format!("SciRS2 spearmanr failed: {}", e))
                })?;
                matrix[i][j] = rho;
                matrix[j][i] = rho;
            }
        }

        let col_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();

        let mut result_df = DataFrame::new();
        let label_series = Series::new(col_names.clone(), Some("column".to_string()))?;
        result_df.add_column("column".to_string(), label_series)?;

        for (col_idx, col_name) in columns.iter().enumerate() {
            let corr_values: Vec<f64> = (0..n).map(|row| matrix[row][col_idx]).collect();
            let series = Series::new(corr_values, Some(col_name.to_string()))?;
            result_df.add_column(col_name.to_string(), series)?;
        }

        Ok(result_df)
    }

    /// Compute the sample covariance matrix for selected columns using SciRS2 SIMD.
    ///
    /// Returns a DataFrame with the same column structure as `correlation_matrix`.
    /// Uses unbiased estimator (divides by n-1).
    ///
    /// # Arguments
    ///
    /// * `df` - The source DataFrame
    /// * `columns` - Column names to include
    ///
    /// # Errors
    ///
    /// Returns an error if any column cannot be converted to numeric values.
    pub fn covariance_matrix(df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        use scirs2_stats::covariance_matrix_simd;

        // arr is (n_obs × n_vars); rowvar=false means variables are columns
        let arr = dataframe_to_array2(df, columns)?;
        let cov = covariance_matrix_simd::<f64, _>(&arr, false, 1).map_err(|e| {
            Error::OperationFailed(format!("SciRS2 covariance_matrix_simd failed: {}", e))
        })?;

        let col_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        let n = columns.len();

        let mut result_df = DataFrame::new();
        let label_series = Series::new(col_names.clone(), Some("column".to_string()))?;
        result_df.add_column("column".to_string(), label_series)?;

        for (col_idx, col_name) in columns.iter().enumerate() {
            let cov_values: Vec<f64> = (0..n).map(|row| cov[[row, col_idx]]).collect();
            let series = Series::new(cov_values, Some(col_name.to_string()))?;
            result_df.add_column(col_name.to_string(), series)?;
        }

        Ok(result_df)
    }

    /// Perform a paired-sample t-test using SciRS2.
    ///
    /// Tests if the mean difference between paired observations differs from zero.
    ///
    /// # Arguments
    ///
    /// * `a` - First sample (must be same length as `b`)
    /// * `b` - Paired second sample
    ///
    /// # Errors
    ///
    /// Returns an error if samples are empty, unequal length, or the test fails.
    pub fn ttest_paired(a: &[f64], b: &[f64]) -> Result<TTestResult> {
        use scirs2_stats::tests::ttest::{ttest_rel, Alternative};

        if a.is_empty() || b.is_empty() {
            return Err(Error::EmptyData(
                "Paired t-test requires non-empty samples".to_string(),
            ));
        }
        if a.len() != b.len() {
            return Err(Error::InvalidInput(format!(
                "Paired t-test requires equal-length samples: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        let arr_a = scirs2_core::ndarray::Array1::from(a.to_vec());
        let arr_b = scirs2_core::ndarray::Array1::from(b.to_vec());
        let result = ttest_rel(
            &arr_a.view(),
            &arr_b.view(),
            Alternative::TwoSided,
            "propagate",
        )
        .map_err(|e| Error::OperationFailed(format!("SciRS2 ttest_rel failed: {}", e)))?;

        Ok(TTestResult {
            statistic: result.statistic,
            p_value: result.pvalue,
            df: result.df,
        })
    }

    /// Perform a chi-square goodness-of-fit test using SciRS2.
    ///
    /// Tests whether observed frequencies differ from expected frequencies.
    ///
    /// # Arguments
    ///
    /// * `observed` - Observed count frequencies (converted internally to integer counts)
    /// * `expected` - Expected frequencies (must be same length as `observed`)
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are empty, unequal, or the test fails.
    pub fn chi2_goodness_of_fit(observed: &[f64], expected: &[f64]) -> Result<Chi2TestResult> {
        use scirs2_core::ndarray::Array1;
        use scirs2_stats::tests::chi2_test::chi2_gof;

        if observed.is_empty() {
            return Err(Error::EmptyData(
                "Chi-square GOF test requires non-empty data".to_string(),
            ));
        }
        if observed.len() != expected.len() {
            return Err(Error::InvalidInput(format!(
                "Observed and expected must have same length: {} vs {}",
                observed.len(),
                expected.len()
            )));
        }

        // chi2_gof expects integer observed; convert via rounding
        let obs_int: Vec<i64> = observed.iter().map(|&x| x.round() as i64).collect();
        let arr_obs = Array1::from(obs_int);
        let arr_exp = Array1::from(expected.to_vec());

        let result = chi2_gof::<f64, i64>(&arr_obs.view(), Some(arr_exp.view()))
            .map_err(|e| Error::OperationFailed(format!("SciRS2 chi2_gof failed: {}", e)))?;

        Ok(Chi2TestResult {
            statistic: result.statistic,
            p_value: result.p_value,
            dof: result.df,
        })
    }

    /// Perform a chi-square test of independence on a contingency table.
    ///
    /// # Arguments
    ///
    /// * `contingency_table` - Observed frequencies as a slice of rows; each row is a `&[f64]`
    ///
    /// # Errors
    ///
    /// Returns an error if the table is empty, non-rectangular, or the test fails.
    pub fn chi2_independence(contingency_table: &[&[f64]]) -> Result<Chi2TestResult> {
        use scirs2_core::ndarray::Array2;
        use scirs2_stats::tests::chi2_test::chi2_independence;

        if contingency_table.is_empty() {
            return Err(Error::EmptyData(
                "Chi-square independence test requires a non-empty contingency table".to_string(),
            ));
        }
        let rows = contingency_table.len();
        let cols = contingency_table[0].len();
        for (i, row) in contingency_table.iter().enumerate() {
            if row.len() != cols {
                return Err(Error::InvalidInput(format!(
                    "Row {} has {} columns but expected {}",
                    i,
                    row.len(),
                    cols
                )));
            }
        }

        // chi2_independence expects integer observations
        let flat: Vec<i64> = contingency_table
            .iter()
            .flat_map(|row| row.iter().map(|&x| x.round() as i64))
            .collect();
        let arr2 = Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
            Error::OperationFailed(format!(
                "Failed to build Array2 for chi2_independence: {}",
                e
            ))
        })?;

        let result = chi2_independence::<f64, i64>(&arr2.view()).map_err(|e| {
            Error::OperationFailed(format!("SciRS2 chi2_independence failed: {}", e))
        })?;

        Ok(Chi2TestResult {
            statistic: result.statistic,
            p_value: result.p_value,
            dof: result.df,
        })
    }

    /// Perform the Mann-Whitney U test for two independent samples.
    ///
    /// Non-parametric alternative to the independent t-test.
    ///
    /// # Arguments
    ///
    /// * `a` - First sample
    /// * `b` - Second sample
    ///
    /// # Errors
    ///
    /// Returns an error if either sample is empty or the test fails.
    pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> Result<Chi2TestResult> {
        use scirs2_stats::mann_whitney;

        if a.is_empty() || b.is_empty() {
            return Err(Error::EmptyData(
                "Mann-Whitney U test requires non-empty samples".to_string(),
            ));
        }

        let arr_a = scirs2_core::ndarray::Array1::from(a.to_vec());
        let arr_b = scirs2_core::ndarray::Array1::from(b.to_vec());
        let (u_stat, p_val) = mann_whitney::<f64>(&arr_a.view(), &arr_b.view(), "two-sided", true)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 mann_whitney failed: {}", e)))?;

        Ok(Chi2TestResult {
            statistic: u_stat,
            p_value: p_val,
            dof: 0, // Mann-Whitney U does not have a fixed DOF
        })
    }

    /// Perform the Wilcoxon signed-rank test on paired samples.
    ///
    /// Tests whether the median difference between paired observations is zero.
    ///
    /// # Arguments
    ///
    /// * `x` - First sample
    /// * `y` - Paired second sample (same length as `x`)
    ///
    /// # Errors
    ///
    /// Returns an error if samples are empty, unequal length, or the test fails.
    pub fn wilcoxon_signed_rank(x: &[f64], y: &[f64]) -> Result<Chi2TestResult> {
        use scirs2_stats::wilcoxon;

        if x.is_empty() || y.is_empty() {
            return Err(Error::EmptyData(
                "Wilcoxon signed-rank test requires non-empty samples".to_string(),
            ));
        }
        if x.len() != y.len() {
            return Err(Error::InvalidInput(format!(
                "Wilcoxon requires equal-length paired samples: {} vs {}",
                x.len(),
                y.len()
            )));
        }

        let arr_x = scirs2_core::ndarray::Array1::from(x.to_vec());
        let arr_y = scirs2_core::ndarray::Array1::from(y.to_vec());
        let (w_stat, p_val) = wilcoxon::<f64>(&arr_x.view(), &arr_y.view(), "wilcox", true)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 wilcoxon failed: {}", e)))?;

        Ok(Chi2TestResult {
            statistic: w_stat,
            p_value: p_val,
            dof: 0,
        })
    }

    /// Perform the Kruskal-Wallis H test for k independent groups.
    ///
    /// Non-parametric generalisation of one-way ANOVA.
    ///
    /// # Arguments
    ///
    /// * `groups` - Slice of groups (each group is a `&[f64]`)
    ///
    /// # Errors
    ///
    /// Returns an error if there are fewer than 2 groups or any group is empty.
    pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<AnovaResult> {
        use scirs2_stats::kruskal_wallis;

        if groups.len() < 2 {
            return Err(Error::InvalidInput(
                "Kruskal-Wallis test requires at least 2 groups".to_string(),
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

        let (h_stat, p_val) = kruskal_wallis::<f64>(&views)
            .map_err(|e| Error::OperationFailed(format!("SciRS2 kruskal_wallis failed: {}", e)))?;

        Ok(AnovaResult {
            f_statistic: h_stat,
            p_value: p_val,
        })
    }

    /// Perform the Shapiro-Wilk test for normality.
    ///
    /// Tests whether `data` was drawn from a normal distribution.
    /// Sample size must be in [3, 5000].
    ///
    /// # Arguments
    ///
    /// * `data` - Sample data
    ///
    /// # Errors
    ///
    /// Returns an error if the sample is too small/large or the test fails.
    pub fn shapiro_wilk_test(data: &[f64]) -> Result<NormalityTestResult> {
        use scirs2_stats::shapiro_wilk;

        if data.is_empty() {
            return Err(Error::EmptyData(
                "Shapiro-Wilk test requires non-empty data".to_string(),
            ));
        }

        let arr = scirs2_core::ndarray::Array1::from(data.to_vec());
        let (w_stat, p_val) = shapiro_wilk::<f64>(&arr.view())
            .map_err(|e| Error::OperationFailed(format!("SciRS2 shapiro_wilk failed: {}", e)))?;

        Ok(NormalityTestResult {
            statistic: w_stat,
            p_value: p_val,
            is_normal: p_val >= 0.05,
        })
    }

    /// Perform the Kolmogorov-Smirnov two-sample test.
    ///
    /// Tests whether two samples come from the same continuous distribution.
    ///
    /// # Arguments
    ///
    /// * `data1` - First sample
    /// * `data2` - Second sample
    ///
    /// # Errors
    ///
    /// Returns an error if either sample is empty or the test fails.
    pub fn ks_two_sample(data1: &[f64], data2: &[f64]) -> Result<NormalityTestResult> {
        use scirs2_stats::tests::normality::ks_2samp;

        if data1.is_empty() || data2.is_empty() {
            return Err(Error::EmptyData(
                "KS two-sample test requires non-empty samples".to_string(),
            ));
        }

        let arr1 = scirs2_core::ndarray::Array1::from(data1.to_vec());
        let arr2 = scirs2_core::ndarray::Array1::from(data2.to_vec());
        let (ks_stat, p_val) = ks_2samp::<f64>(&arr1.view(), &arr2.view(), "two-sided")
            .map_err(|e| Error::OperationFailed(format!("SciRS2 ks_2samp failed: {}", e)))?;

        Ok(NormalityTestResult {
            statistic: ks_stat,
            p_value: p_val,
            is_normal: p_val >= 0.05,
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
