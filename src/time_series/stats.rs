//! Time Series Statistical Tests Module
//!
//! This module provides comprehensive statistical tests specifically designed for
//! time series analysis, including tests for stationarity, seasonality, autocorrelation,
//! and other time series properties.

use crate::core::error::{Error, Result};
use crate::stats::special::{
    chi2_sf, f_sf, normal_cdf, normal_sf, student_t_ppf, student_t_two_sided_p,
};
use crate::time_series::analysis::ols_with_std_errors;
use crate::time_series::core::TimeSeries;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Normality / outlier tests live in a submodule to keep this file under
// the 2000-line guideline (see stats_normality.rs).
#[path = "stats_normality.rs"]
mod normality;

// Shared numerical helpers (Dickey-Fuller regression, Newey-West long-run
// variance, the normal quantile, and the published critical-value tables) live
// in a submodule to keep this file under the 2000-line guideline.
#[path = "stats_numeric.rs"]
mod numeric;

use numeric::{
    adf_p_value, adf_regression_statistic, average_ranks, durbin_watson_bounds, tie_sum,
};
pub(crate) use numeric::{
    inv_normal_cdf, kpss_critical_values, kpss_p_value_from_table, newey_west_bandwidth,
    newey_west_long_run_variance, normal_critical_value, poly,
};

/// Comprehensive time series statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesStats {
    /// Basic descriptive statistics
    pub descriptive: DescriptiveStats,
    /// Stationarity test results
    pub stationarity_tests: StationarityTestResults,
    /// Seasonality test results
    pub seasonality_tests: SeasonalityTestResults,
    /// Autocorrelation tests
    pub autocorrelation_tests: AutocorrelationTestResults,
    /// Normality tests
    pub normality_tests: NormalityTestResults,
    /// Outlier detection results
    pub outlier_tests: OutlierTestResults,
}

/// Descriptive statistics for time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    /// Number of observations
    pub count: usize,
    /// Mean
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// 25th percentile
    pub q25: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// 75th percentile
    pub q75: f64,
    /// Maximum value
    pub max: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Coefficient of variation
    pub cv: f64,
    /// Interquartile range
    pub iqr: f64,
}

/// Stationarity test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTestResults {
    /// Augmented Dickey-Fuller test
    pub adf_test: AugmentedDickeyFullerTest,
    /// KPSS test
    pub kpss_test: KwiatkowskiPhillipsSchmidtShinTest,
    /// Phillips-Perron test
    pub pp_test: PhillipsPerronTest,
    /// Overall stationarity assessment
    pub is_stationary: bool,
    /// Recommendation for differencing
    pub differencing_recommendation: DifferencingRecommendation,
}

/// Seasonality test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityTestResults {
    /// Seasonal test
    pub seasonal_test: SeasonalTest,
    /// Friedman test for seasonality, or `None` when the series is too short
    /// for the assumed 12-period cycle to have two complete blocks. It is an
    /// `Option` rather than a zero-filled `FriedmanTest` so that "not enough
    /// data" cannot be mistaken for "tested, found no seasonality".
    pub friedman_test: Option<FriedmanTest>,
    /// Kruskal-Wallis test
    pub kruskal_wallis_test: KruskalWallisTest,
    /// Overall seasonality assessment
    pub has_seasonality: bool,
    /// Detected seasonal periods
    pub seasonal_periods: Vec<usize>,
}

/// Autocorrelation test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationTestResults {
    /// Ljung-Box test
    pub ljung_box_test: LjungBoxTest,
    /// Box-Pierce test
    pub box_pierce_test: BoxPierceTest,
    /// Durbin-Watson test
    pub durbin_watson_test: DurbinWatsonTest,
    /// Breusch-Godfrey test
    pub breusch_godfrey_test: BreuschGodfreyTest,
    /// White noise assessment
    pub is_white_noise: bool,
}

/// Normality test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalityTestResults {
    /// Jarque-Bera test
    pub jarque_bera_test: JarqueBeraTest,
    /// Shapiro-Wilk test
    pub shapiro_wilk_test: ShapiroWilkTest,
    /// Anderson-Darling test
    pub anderson_darling_test: AndersonDarlingTest,
    /// Overall normality assessment
    pub is_normal: bool,
}

/// Outlier test results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierTestResults {
    /// Grubbs test
    pub grubbs_test: GrubbsTest,
    /// Modified Z-score test
    pub modified_z_score_test: ModifiedZScoreTest,
    /// IQR-based outlier detection
    pub iqr_outlier_test: IQROutlierTest,
    /// Detected outliers
    pub outlier_indices: Vec<usize>,
    /// Outlier percentage
    pub outlier_percentage: f64,
}

/// Augmented Dickey-Fuller test for unit roots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentedDickeyFullerTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Number of lags used
    pub n_lags: usize,
    /// Critical values
    pub critical_values: HashMap<String, f64>,
    /// Whether unit root is rejected (series is stationary)
    pub is_stationary: bool,
    /// Trend component included
    pub trend: String,
}

/// KPSS test for stationarity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KwiatkowskiPhillipsSchmidtShinTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical values
    pub critical_values: HashMap<String, f64>,
    /// Whether series is stationary
    pub is_stationary: bool,
    /// Trend specification
    pub trend: String,
    /// Number of lags for long-run variance estimation
    pub n_lags: usize,
}

/// Phillips-Perron test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhillipsPerronTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical values
    pub critical_values: HashMap<String, f64>,
    /// Whether unit root is rejected
    pub is_stationary: bool,
    /// Trend specification
    pub trend: String,
}

/// Differencing recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferencingRecommendation {
    /// Recommended number of differences
    pub recommended_d: usize,
    /// Recommended seasonal differences
    pub recommended_seasonal_d: usize,
    /// Reason for recommendation
    pub reason: String,
}

/// Seasonal test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Detected period
    pub period: Option<usize>,
    /// Seasonal strength
    pub seasonal_strength: f64,
    /// Whether seasonality is significant
    pub is_seasonal: bool,
}

/// Friedman test for seasonality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FriedmanTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: f64,
    /// Whether seasonality is detected
    pub is_seasonal: bool,
    /// Tested period
    pub period: usize,
}

/// Kruskal-Wallis test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KruskalWallisTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: f64,
    /// Whether groups are significantly different
    pub is_significant: bool,
    /// Tested period
    pub period: usize,
}

/// Ljung-Box test for autocorrelation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LjungBoxTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Number of lags tested
    pub n_lags: usize,
    /// Whether autocorrelation is present
    pub has_autocorrelation: bool,
}

/// Box-Pierce test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxPierceTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Number of lags tested
    pub n_lags: usize,
    /// Whether autocorrelation is present
    pub has_autocorrelation: bool,
}

/// Durbin-Watson test for autocorrelation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurbinWatsonTest {
    /// Test statistic
    pub statistic: f64,
    /// Savin-White lower bound `d_L` at the 5% level for this sample size
    /// (one regressor besides the intercept). `d < d_L` rejects in favour of
    /// positive autocorrelation; `d > 4 − d_L` rejects in favour of negative
    /// autocorrelation.
    pub lower_critical: f64,
    /// Savin-White upper bound `d_U` at the 5% level. `d_L < d < d_U` (and
    /// symmetrically `4 − d_U < d < 4 − d_L`) is the *inconclusive* region of
    /// the bounds test.
    pub upper_critical: f64,
    /// Test result interpretation: one of `"Positive autocorrelation"`,
    /// `"Negative autocorrelation"`, `"Inconclusive"`,
    /// `"No significant autocorrelation"`, or
    /// `"Undefined (no residual variation)"` for a constant series.
    pub result: String,
    /// Whether positive autocorrelation is detected
    pub has_positive_autocorr: bool,
    /// Whether negative autocorrelation is detected
    pub has_negative_autocorr: bool,
}

/// Breusch-Godfrey test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreuschGodfreyTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Number of lags tested
    pub n_lags: usize,
    /// Whether serial correlation is present
    pub has_serial_correlation: bool,
}

/// Jarque-Bera test for normality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JarqueBeraTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Skewness component
    pub skewness_stat: f64,
    /// Kurtosis component
    pub kurtosis_stat: f64,
    /// Whether data is normally distributed
    pub is_normal: bool,
}

/// Shapiro-Wilk test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapiroWilkTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether data is normally distributed
    pub is_normal: bool,
}

/// Anderson-Darling test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndersonDarlingTest {
    /// Test statistic
    pub statistic: f64,
    /// Critical values
    pub critical_values: HashMap<String, f64>,
    /// P-value (approximate)
    pub p_value: f64,
    /// Whether data is normally distributed
    pub is_normal: bool,
}

/// Grubbs test for outliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrubbsTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value
    pub critical_value: f64,
    /// Index of potential outlier
    pub outlier_index: Option<usize>,
    /// Whether outlier is detected
    pub has_outlier: bool,
}

/// Modified Z-score test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifiedZScoreTest {
    /// Modified Z-scores
    pub modified_z_scores: Vec<f64>,
    /// Threshold used
    pub threshold: f64,
    /// Outlier indices
    pub outlier_indices: Vec<usize>,
    /// Whether outliers are detected
    pub has_outliers: bool,
}

/// IQR-based outlier test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQROutlierTest {
    /// Q1 (25th percentile)
    pub q1: f64,
    /// Q3 (75th percentile)
    pub q3: f64,
    /// IQR
    pub iqr: f64,
    /// Lower fence
    pub lower_fence: f64,
    /// Upper fence
    pub upper_fence: f64,
    /// Outlier indices
    pub outlier_indices: Vec<usize>,
    /// Whether outliers are detected
    pub has_outliers: bool,
}

/// White noise test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteNoiseTest {
    /// Ljung-Box test for multiple lags
    pub ljung_box_tests: Vec<LjungBoxTest>,
    /// Variance ratio test
    pub variance_ratio_test: VarianceRatioTest,
    /// Runs test
    pub runs_test: RunsTest,
    /// Overall white noise assessment
    pub is_white_noise: bool,
}

/// Variance ratio test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceRatioTest {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Variance ratio
    pub variance_ratio: f64,
    /// Whether series follows random walk
    pub is_random_walk: bool,
}

/// Runs test for randomness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunsTest {
    /// Number of runs
    pub n_runs: usize,
    /// Expected number of runs
    pub expected_runs: f64,
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether sequence is random
    pub is_random: bool,
}

impl TimeSeriesStats {
    /// Compute comprehensive statistics for time series
    pub fn compute(ts: &TimeSeries) -> Result<Self> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.is_empty() {
            return Err(Error::InvalidInput(
                "No valid values in time series".to_string(),
            ));
        }

        let descriptive = Self::compute_descriptive_stats(&values)?;
        let stationarity_tests = Self::compute_stationarity_tests(&values)?;
        let seasonality_tests = Self::compute_seasonality_tests(&values)?;
        let autocorrelation_tests = Self::compute_autocorrelation_tests(&values)?;
        let normality_tests = Self::compute_normality_tests(&values)?;
        let outlier_tests = Self::compute_outlier_tests(&values)?;

        Ok(Self {
            descriptive,
            stationarity_tests,
            seasonality_tests,
            autocorrelation_tests,
            normality_tests,
            outlier_tests,
        })
    }

    /// Compute descriptive statistics
    fn compute_descriptive_stats(values: &[f64]) -> Result<DescriptiveStats> {
        let count = values.len();
        let sum = values.iter().sum::<f64>();
        let mean = sum / count as f64;

        // Variance and standard deviation
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std = variance.sqrt();

        // Sorted values for quantiles
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let min = sorted[0];
        let max = sorted[count - 1];
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let q1_idx = count / 4;
        let q3_idx = 3 * count / 4;
        let q25 = sorted[q1_idx];
        let q75 = sorted[q3_idx];
        let iqr = q75 - q25;

        // Higher order moments
        let skewness = if std > 0.0 {
            values
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>()
                / count as f64
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 {
            values
                .iter()
                .map(|x| ((x - mean) / std).powi(4))
                .sum::<f64>()
                / count as f64
                - 3.0
        } else {
            0.0
        };

        let cv = if mean != 0.0 { std / mean.abs() } else { 0.0 };

        Ok(DescriptiveStats {
            count,
            mean,
            std,
            min,
            q25,
            median,
            q75,
            max,
            skewness,
            kurtosis,
            cv,
            iqr,
        })
    }

    /// Compute stationarity tests
    fn compute_stationarity_tests(values: &[f64]) -> Result<StationarityTestResults> {
        let adf_test = AugmentedDickeyFullerTest::compute(values)?;
        let kpss_test = KwiatkowskiPhillipsSchmidtShinTest::compute(values, "constant")?;
        let pp_test = PhillipsPerronTest::compute(values)?;

        // Overall assessment
        let is_stationary = adf_test.is_stationary && kpss_test.is_stationary;

        let differencing_recommendation = if !adf_test.is_stationary {
            DifferencingRecommendation {
                recommended_d: 1,
                recommended_seasonal_d: 0,
                reason: "ADF test suggests non-stationarity".to_string(),
            }
        } else {
            DifferencingRecommendation {
                recommended_d: 0,
                recommended_seasonal_d: 0,
                reason: "Series appears stationary".to_string(),
            }
        };

        Ok(StationarityTestResults {
            adf_test,
            kpss_test,
            pp_test,
            is_stationary,
            differencing_recommendation,
        })
    }

    /// Compute seasonality tests
    fn compute_seasonality_tests(values: &[f64]) -> Result<SeasonalityTestResults> {
        let seasonal_test = SeasonalTest::compute(values)?;
        // Assume monthly data. Under 24 observations there are not two
        // complete cycles to rank against each other, so the test is reported
        // as unavailable rather than as a fabricated "not seasonal".
        let friedman_test = FriedmanTest::compute(values, 12).ok();
        let kruskal_wallis_test = KruskalWallisTest::compute(values, 7)?; // Assume weekly seasonality

        let has_seasonality = seasonal_test.is_seasonal
            || friedman_test.as_ref().is_some_and(|t| t.is_seasonal)
            || kruskal_wallis_test.is_significant;

        let mut seasonal_periods = Vec::new();
        if let Some(period) = seasonal_test.period {
            seasonal_periods.push(period);
        }
        if let Some(friedman) = friedman_test.as_ref().filter(|t| t.is_seasonal) {
            seasonal_periods.push(friedman.period);
        }
        if kruskal_wallis_test.is_significant {
            seasonal_periods.push(kruskal_wallis_test.period);
        }
        seasonal_periods.sort_unstable();
        seasonal_periods.dedup();

        Ok(SeasonalityTestResults {
            seasonal_test,
            friedman_test,
            kruskal_wallis_test,
            has_seasonality,
            seasonal_periods,
        })
    }

    /// Compute autocorrelation tests
    fn compute_autocorrelation_tests(values: &[f64]) -> Result<AutocorrelationTestResults> {
        let ljung_box_test = LjungBoxTest::compute(values, 10)?;
        let box_pierce_test = BoxPierceTest::compute(values, 10)?;
        let durbin_watson_test = DurbinWatsonTest::compute(values)?;
        let breusch_godfrey_test = BreuschGodfreyTest::compute(values, 5)?;

        let is_white_noise = !ljung_box_test.has_autocorrelation
            && !box_pierce_test.has_autocorrelation
            && !durbin_watson_test.has_positive_autocorr
            && !breusch_godfrey_test.has_serial_correlation;

        Ok(AutocorrelationTestResults {
            ljung_box_test,
            box_pierce_test,
            durbin_watson_test,
            breusch_godfrey_test,
            is_white_noise,
        })
    }

    /// Compute normality tests
    fn compute_normality_tests(values: &[f64]) -> Result<NormalityTestResults> {
        let jarque_bera_test = JarqueBeraTest::compute(values)?;
        let shapiro_wilk_test = ShapiroWilkTest::compute(values)?;
        let anderson_darling_test = AndersonDarlingTest::compute(values)?;

        let is_normal = jarque_bera_test.is_normal
            && shapiro_wilk_test.is_normal
            && anderson_darling_test.is_normal;

        Ok(NormalityTestResults {
            jarque_bera_test,
            shapiro_wilk_test,
            anderson_darling_test,
            is_normal,
        })
    }

    /// Compute outlier tests
    fn compute_outlier_tests(values: &[f64]) -> Result<OutlierTestResults> {
        let grubbs_test = GrubbsTest::compute(values)?;
        let modified_z_score_test = ModifiedZScoreTest::compute(values, 3.5)?;
        let iqr_outlier_test = IQROutlierTest::compute(values)?;

        let mut all_outliers = Vec::new();
        if let Some(idx) = grubbs_test.outlier_index {
            all_outliers.push(idx);
        }
        all_outliers.extend(&modified_z_score_test.outlier_indices);
        all_outliers.extend(&iqr_outlier_test.outlier_indices);
        all_outliers.sort_unstable();
        all_outliers.dedup();

        let outlier_percentage = all_outliers.len() as f64 / values.len() as f64 * 100.0;

        Ok(OutlierTestResults {
            grubbs_test,
            modified_z_score_test,
            iqr_outlier_test,
            outlier_indices: all_outliers,
            outlier_percentage,
        })
    }
}

impl AugmentedDickeyFullerTest {
    /// Compute ADF test
    pub fn compute(values: &[f64]) -> Result<Self> {
        if values.len() < 10 {
            return Err(Error::InvalidInput(
                "Need at least 10 observations for ADF test".to_string(),
            ));
        }

        let n = values.len();
        let n_lags = ((n as f64).cbrt() * 12.0 / 100.0) as usize;

        // Real ADF regression: Δyₜ = α + β·yₜ₋₁ + Σⱼ γⱼ·Δyₜ₋ⱼ + εₜ. The test
        // statistic is the t-ratio on the lagged-level coefficient β.
        let statistic = adf_regression_statistic(values, n_lags)?;

        let mut critical_values = HashMap::new();
        critical_values.insert("1%".to_string(), -3.43);
        critical_values.insert("5%".to_string(), -2.86);
        critical_values.insert("10%".to_string(), -2.57);

        // Approximate p-value by monotone interpolation of the MacKinnon
        // constant-only critical-value surface (no closed form exists for the
        // Dickey-Fuller distribution); the verdict uses the tabulated values.
        let p_value = adf_p_value(statistic);

        let is_stationary = statistic < critical_values["5%"];

        Ok(Self {
            statistic,
            p_value,
            n_lags,
            critical_values,
            is_stationary,
            trend: "constant".to_string(),
        })
    }
}

impl KwiatkowskiPhillipsSchmidtShinTest {
    /// Compute KPSS test
    pub fn compute(values: &[f64], trend: &str) -> Result<Self> {
        if values.len() < 10 {
            return Err(Error::InvalidInput(
                "Need at least 10 observations for KPSS test".to_string(),
            ));
        }

        // Detrend the series
        let detrended = match trend {
            "constant" => Self::detrend_constant(values)?,
            "linear" => Self::detrend_linear(values)?,
            _ => {
                return Err(Error::InvalidInput(
                    "Invalid trend specification".to_string(),
                ))
            }
        };

        // Calculate partial sums
        let mut partial_sums = vec![0.0; detrended.len()];
        partial_sums[0] = detrended[0];
        for i in 1..detrended.len() {
            partial_sums[i] = partial_sums[i - 1] + detrended[i];
        }

        // Long-run variance via a Bartlett-kernel Newey-West estimator with the
        // Schwert lag rule l = floor(4 * (n/100)^(1/4)). Using only the
        // contemporaneous variance (the previous behaviour) ignores the serial
        // correlation the KPSS statistic is explicitly built to account for.
        let n_obs = detrended.len();
        let bandwidth = (4.0 * (n_obs as f64 / 100.0).powf(0.25)).floor() as usize;
        let long_run_variance = newey_west_long_run_variance(&detrended, bandwidth);

        // KPSS statistic
        let n = values.len() as f64;
        let sum_of_squares: f64 = partial_sums.iter().map(|x| x * x).sum();
        let statistic = if long_run_variance > 0.0 {
            sum_of_squares / (n * n * long_run_variance)
        } else {
            0.0
        };

        let (c1, c5, c10) = kpss_critical_values(trend)?;
        let mut critical_values = HashMap::new();
        critical_values.insert("1%".to_string(), c1);
        critical_values.insert("5%".to_string(), c5);
        critical_values.insert("10%".to_string(), c10);

        // Approximate p-value by interpolating the KPSS critical-value table
        // (Kwiatkowski et al., 1992). The KPSS statistic follows a
        // non-standard distribution with no elementary closed form, so this is
        // an honest table-based approximation, clamped to [0.01, 0.10] outside
        // the tabulated range (matching the convention used by statsmodels).
        let p_value = kpss_p_value_from_table(statistic, c10, c5, c1);

        let is_stationary = statistic < c5;

        Ok(Self {
            statistic,
            p_value,
            critical_values,
            is_stationary,
            trend: trend.to_string(),
            n_lags: bandwidth,
        })
    }

    fn detrend_constant(values: &[f64]) -> Result<Vec<f64>> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Ok(values.iter().map(|x| x - mean).collect())
    }

    fn detrend_linear(values: &[f64]) -> Result<Vec<f64>> {
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = x_values.iter().zip(values).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        let detrended: Vec<f64> = x_values
            .iter()
            .zip(values)
            .map(|(x, y)| y - (slope * x + intercept))
            .collect();

        Ok(detrended)
    }
}

impl PhillipsPerronTest {
    /// Compute the Phillips-Perron `Z_τ` test for a unit root (constant, no
    /// trend).
    ///
    /// Unlike the ADF test, which soaks up residual serial correlation by
    /// adding lagged differences to the regression, PP runs the *unaugmented*
    /// regression
    ///
    /// ```text
    /// Δyₜ = α + ρ·yₜ₋₁ + eₜ
    /// ```
    ///
    /// and corrects its t-statistic non-parametrically with a Bartlett-kernel
    /// long-run variance `λ²` of the residuals (Hamilton 1994, eq. 17.6.8):
    ///
    /// ```text
    /// Z_τ = √(γ̂₀/λ̂²)·t_ρ  −  (λ̂² − γ̂₀)·T·σ̂_ρ / (2·λ̂·s)
    /// ```
    ///
    /// with `γ̂₀ = Σê²/T`, `s² = Σê²/(T−k)`, `λ̂ = √λ̂²` and `σ̂_ρ` the OLS
    /// standard error of `ρ̂`. Both terms are invariant to rescaling the series
    /// (`λ̂² − γ̂₀ ∝ c²` against `λ̂·s ∝ c²`), as `Z_τ` must be. Bandwidth
    /// follows Newey & West (1994), `l = ⌊4·(T/100)^{2/9}⌋`.
    ///
    /// This replaces a version that simply re-ran the ADF test and multiplied
    /// its statistic by `√((n−1)/n)` — a factor of at most 1.005 that has
    /// nothing to do with the Phillips-Perron correction, so the "PP test" was
    /// numerically the ADF test under a different name and inherited ADF's lag
    /// augmentation rather than doing the non-parametric correction PP is for.
    ///
    /// `Z_τ` has the same Dickey-Fuller asymptotic distribution as the ADF
    /// statistic, so the same critical-value table and (approximate,
    /// interpolated) p-value apply.
    pub fn compute(values: &[f64]) -> Result<Self> {
        let n = values.len();
        if n < 10 {
            return Err(Error::InvalidInput(
                "Need at least 10 observations for the Phillips-Perron test".to_string(),
            ));
        }

        // Unaugmented Dickey-Fuller regression: Δyₜ = α + ρ·yₜ₋₁ + eₜ.
        let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(n - 1);
        let mut response: Vec<f64> = Vec::with_capacity(n - 1);
        for t in 1..n {
            x_rows.push(vec![1.0, values[t - 1]]);
            response.push(values[t] - values[t - 1]);
        }

        let (coefficients, std_errors) =
            ols_with_std_errors(&x_rows, &response).ok_or_else(|| {
                Error::InvalidInput("Phillips-Perron regression matrix is singular".to_string())
            })?;

        let rho = coefficients[1];
        let se_rho = std_errors[1];
        if !se_rho.is_finite() || se_rho <= 0.0 {
            return Err(Error::InvalidInput(
                "Phillips-Perron regression produced a degenerate standard error".to_string(),
            ));
        }
        let t_rho = rho / se_rho;

        // Residuals of the unaugmented regression.
        let residuals: Vec<f64> = x_rows
            .iter()
            .zip(response.iter())
            .map(|(row, &y)| y - (coefficients[0] * row[0] + coefficients[1] * row[1]))
            .collect();

        let t_obs = residuals.len();
        let t_f = t_obs as f64;
        let n_regressors = 2.0; // constant + lagged level
        let ssr: f64 = residuals.iter().map(|e| e * e).sum();

        let gamma0 = ssr / t_f;
        let s_squared = ssr / (t_f - n_regressors);
        let s = s_squared.sqrt();

        let bandwidth = newey_west_bandwidth(t_obs);
        let lambda_squared = newey_west_long_run_variance(&residuals, bandwidth);

        let statistic = if gamma0 > 0.0 && lambda_squared > 0.0 && s > 0.0 {
            let lambda = lambda_squared.sqrt();
            (gamma0 / lambda_squared).sqrt() * t_rho
                - (lambda_squared - gamma0) * t_f * se_rho / (2.0 * lambda * s)
        } else {
            // No residual variation: the regression is degenerate.
            f64::NAN
        };

        let mut critical_values = HashMap::new();
        critical_values.insert("1%".to_string(), -3.43);
        critical_values.insert("5%".to_string(), -2.86);
        critical_values.insert("10%".to_string(), -2.57);

        let is_stationary = statistic < -2.86;
        let p_value = if statistic.is_finite() {
            adf_p_value(statistic)
        } else {
            f64::NAN
        };

        Ok(Self {
            statistic,
            p_value,
            critical_values,
            is_stationary,
            trend: "constant".to_string(),
        })
    }
}

impl SeasonalTest {
    /// Compute seasonal test
    pub fn compute(values: &[f64]) -> Result<Self> {
        if values.len() < 20 {
            return Ok(Self {
                statistic: 0.0,
                p_value: 1.0,
                period: None,
                seasonal_strength: 0.0,
                is_seasonal: false,
            });
        }

        let mut max_strength = 0.0;
        let mut best_period = None;

        // Test common periods
        for period in 2..=std::cmp::min(values.len() / 3, 365) {
            let strength = Self::calculate_seasonal_strength(values, period)?;
            if strength > max_strength {
                max_strength = strength;
                best_period = Some(period);
            }
        }

        // For the strongest candidate period, run a one-way ANOVA across the
        // `period` seasonal phases (groups formed by index mod period). The
        // F-statistic tests whether the per-phase means differ; its p-value
        // comes from the real F distribution rather than a 2-bucket ladder.
        let (statistic, p_value, is_seasonal) = match best_period {
            Some(period) => {
                let n = values.len();
                let k = period;
                let grand_mean = values.iter().sum::<f64>() / n as f64;

                let mut group_sum = vec![0.0_f64; k];
                let mut group_count = vec![0usize; k];
                for (i, &v) in values.iter().enumerate() {
                    group_sum[i % k] += v;
                    group_count[i % k] += 1;
                }

                let mut ss_between = 0.0;
                for g in 0..k {
                    if group_count[g] > 0 {
                        let gm = group_sum[g] / group_count[g] as f64;
                        ss_between += group_count[g] as f64 * (gm - grand_mean).powi(2);
                    }
                }

                let mut ss_within = 0.0;
                for (i, &v) in values.iter().enumerate() {
                    let g = i % k;
                    if group_count[g] > 0 {
                        let gm = group_sum[g] / group_count[g] as f64;
                        ss_within += (v - gm).powi(2);
                    }
                }

                let df1 = (k - 1) as f64;
                let df2 = (n - k) as f64;
                if df1 > 0.0 && df2 > 0.0 && ss_within > 0.0 {
                    let f_stat = (ss_between / df1) / (ss_within / df2);
                    let p = f_sf(f_stat, df1, df2);
                    (f_stat, p, p < 0.05)
                } else {
                    (0.0, 1.0, false)
                }
            }
            None => (0.0, 1.0, false),
        };

        Ok(Self {
            statistic,
            p_value,
            period: best_period,
            seasonal_strength: max_strength,
            is_seasonal,
        })
    }

    fn calculate_seasonal_strength(values: &[f64], period: usize) -> Result<f64> {
        if values.len() < period * 2 {
            return Ok(0.0);
        }

        let mut seasonal_means = vec![0.0; period];
        let mut counts = vec![0; period];

        for (i, &value) in values.iter().enumerate() {
            let season_idx = i % period;
            seasonal_means[season_idx] += value;
            counts[season_idx] += 1;
        }

        for i in 0..period {
            if counts[i] > 0 {
                seasonal_means[i] /= counts[i] as f64;
            }
        }

        let overall_mean = values.iter().sum::<f64>() / values.len() as f64;
        let seasonal_variance = seasonal_means
            .iter()
            .map(|&mean| (mean - overall_mean).powi(2))
            .sum::<f64>()
            / period as f64;

        let total_variance = values
            .iter()
            .map(|&value| (value - overall_mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;

        if total_variance > 0.0 {
            Ok((seasonal_variance / total_variance).min(1.0))
        } else {
            Ok(0.0)
        }
    }
}

impl FriedmanTest {
    /// Friedman rank test for seasonality.
    ///
    /// The series is laid out as `b = ⌊n/period⌋` blocks (complete seasonal
    /// cycles) of `k = period` treatments (positions within the cycle). Values
    /// are **ranked within each block** (average ranks for ties), the rank sums
    /// `R_j` per within-cycle position are formed, and
    ///
    /// ```text
    /// Q = 12 / (b·k·(k+1)) · Σ R_j²  −  3·b·(k+1)
    /// ```
    ///
    /// is compared against `χ²(k−1)`, with the standard tie correction
    /// `Q / (1 − Σ(t³−t) / (b·k·(k²−1)))`.
    ///
    /// The previous implementation never ranked anything: it summed the **raw
    /// values** per within-cycle position and divided the sum of squared
    /// deviations by their mean, i.e. a Pearson goodness-of-fit statistic on
    /// level sums. That is not distributed as `χ²(k−1)`, it is not
    /// scale-invariant (adding a constant to the whole series changes it), and
    /// it is undefined for series that can go negative.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] when `period < 2` or fewer than two
    /// complete cycles are available.
    pub fn compute(values: &[f64], period: usize) -> Result<Self> {
        if period < 2 {
            return Err(Error::InvalidInput(format!(
                "Friedman test needs a seasonal period of at least 2, got {period}"
            )));
        }
        if values.len() < period * 2 {
            return Err(Error::InvalidInput(format!(
                "Friedman test needs at least 2 complete cycles ({} observations), got {}",
                period * 2,
                values.len()
            )));
        }

        let n_blocks = values.len() / period;
        let b = n_blocks as f64;
        let k = period as f64;

        let mut rank_sums = vec![0.0; period];
        // Σ (t³ − t) over every tie group in every block, for the correction.
        let mut tie_correction = 0.0;

        for block in 0..n_blocks {
            let row = &values[block * period..(block + 1) * period];
            let ranks = average_ranks(row);
            for (j, rank) in ranks.iter().enumerate() {
                rank_sums[j] += rank;
            }
            tie_correction += tie_sum(row);
        }

        let statistic_raw = 12.0 / (b * k * (k + 1.0))
            * rank_sums.iter().map(|r| r * r).sum::<f64>()
            - 3.0 * b * (k + 1.0);

        let tie_denominator = 1.0 - tie_correction / (b * k * (k * k - 1.0));
        let statistic = if tie_denominator > 0.0 {
            statistic_raw / tie_denominator
        } else {
            // Every block is completely tied: no ranking information at all.
            f64::NAN
        };

        let df = k - 1.0;
        let p_value = if statistic.is_finite() {
            chi2_sf(statistic, df)
        } else {
            f64::NAN
        };
        let is_seasonal = p_value < 0.05;

        Ok(Self {
            statistic,
            p_value,
            df,
            is_seasonal,
            period,
        })
    }
}

impl KruskalWallisTest {
    /// Compute Kruskal-Wallis H test.
    ///
    /// The data is divided into `period` groups by cycling modulo `period`.
    /// H = (12 / (n*(n+1))) * Σ(R_j²/n_j) − 3*(n+1)
    /// Under H₀, H ~ χ²(k−1) where k = period.
    pub fn compute(values: &[f64], period: usize) -> Result<Self> {
        if period < 2 {
            return Err(Error::InvalidValue(
                "KruskalWallis requires at least 2 groups (period >= 2)".into(),
            ));
        }
        let n = values.len();
        if n < period {
            return Err(Error::InvalidValue(
                "Insufficient observations for Kruskal-Wallis test".into(),
            ));
        }

        // Build ranks of the pooled data (average tied ranks).
        let mut indexed: Vec<(usize, f64)> = values.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut ranks = vec![0.0_f64; n];
        let mut i = 0usize;
        while i < n {
            let mut j = i + 1;
            while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
                j += 1;
            }
            // Average rank for tied group (1-based)
            let avg_rank = ((i + 1 + j) as f64) / 2.0;
            for &(orig_idx, _) in &indexed[i..j] {
                ranks[orig_idx] = avg_rank;
            }
            i = j;
        }

        // Accumulate rank sums per group and group sizes.
        let k = period;
        let mut rank_sums = vec![0.0_f64; k];
        let mut group_sizes = vec![0usize; k];
        for (idx, &r) in ranks.iter().enumerate() {
            let g = idx % k;
            rank_sums[g] += r;
            group_sizes[g] += 1;
        }

        let nf = n as f64;
        let h_raw: f64 = rank_sums
            .iter()
            .zip(group_sizes.iter())
            .filter(|(_, &sz)| sz > 0)
            .map(|(&rs, &sz)| rs * rs / sz as f64)
            .sum();

        let statistic = 12.0 / (nf * (nf + 1.0)) * h_raw - 3.0 * (nf + 1.0);
        let df = (k - 1) as f64;
        let p_value = chi2_sf(statistic.max(0.0), df);
        let is_significant = p_value < 0.05;

        Ok(Self {
            statistic,
            p_value,
            df,
            is_significant,
            period,
        })
    }
}

impl LjungBoxTest {
    /// Compute Ljung-Box test
    pub fn compute(values: &[f64], n_lags: usize) -> Result<Self> {
        let n = values.len() as f64;
        let mut statistic = 0.0;

        let mean = values.iter().sum::<f64>() / n;

        for lag in 1..=n_lags {
            let autocorr = Self::calculate_autocorrelation(values, lag, mean)?;
            statistic += autocorr * autocorr / (n - lag as f64);
        }

        statistic *= n * (n + 2.0);

        let df = n_lags;
        let p_value = chi2_sf(statistic, df as f64);
        let has_autocorrelation = p_value < 0.05;

        Ok(Self {
            statistic,
            p_value,
            df,
            n_lags,
            has_autocorrelation,
        })
    }

    fn calculate_autocorrelation(values: &[f64], lag: usize, mean: f64) -> Result<f64> {
        if lag >= values.len() {
            return Ok(0.0);
        }

        let n = values.len() - lag;
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let dev1 = values[i] - mean;
            let dev2 = values[i + lag] - mean;
            numerator += dev1 * dev2;
        }

        for &val in values {
            let dev = val - mean;
            denominator += dev * dev;
        }

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

impl BoxPierceTest {
    /// Compute Box-Pierce Q statistic.
    ///
    /// Q = n * Σ_{k=1}^{m} r̂_k²
    /// Under H₀, Q ~ χ²(m).  Distinct from Ljung-Box which uses the
    /// weighted form n*(n+2)*Σ r̂_k²/(n-k).
    pub fn compute(values: &[f64], n_lags: usize) -> Result<Self> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let mut statistic = 0.0;
        for lag in 1..=n_lags {
            let autocorr = LjungBoxTest::calculate_autocorrelation(values, lag, mean)?;
            statistic += autocorr * autocorr;
        }
        statistic *= n;

        let df = n_lags;
        let p_value = chi2_sf(statistic, df as f64);
        let has_autocorrelation = p_value < 0.05;

        Ok(Self {
            statistic,
            p_value,
            df,
            n_lags,
            has_autocorrelation,
        })
    }
}

impl DurbinWatsonTest {
    /// Compute Durbin-Watson test
    pub fn compute(values: &[f64]) -> Result<Self> {
        if values.len() < 3 {
            return Err(Error::InvalidInput(
                "Need at least 3 observations for DW test".to_string(),
            ));
        }

        let mut sum_diff_sq = 0.0;
        let mut sum_sq = 0.0;

        let mean = values.iter().sum::<f64>() / values.len() as f64;

        for i in 1..values.len() {
            sum_diff_sq += (values[i] - values[i - 1]).powi(2);
        }

        for &val in values {
            sum_sq += (val - mean).powi(2);
        }

        let statistic = if sum_sq > 0.0 {
            sum_diff_sq / sum_sq
        } else {
            // A constant series has no residual variation at all: the ratio is
            // 0/0, not the "no autocorrelation" value of 2.
            f64::NAN
        };

        // Savin-White (1977) 5% bounds for k' = 1 (one regressor beyond the
        // intercept — here the demeaned level, which is the regression the
        // statistic above corresponds to). The previous 1.5 / 2.5 pair was not
        // a critical value of anything: at n = 20 the true bounds are
        // d_L = 1.201 / d_U = 1.411, so a d of 1.45 was reported as "positive
        // autocorrelation" when the bounds test does not reject at all.
        let (lower_critical, upper_critical) = durbin_watson_bounds(values.len());

        let (result, has_positive_autocorr, has_negative_autocorr) = if !statistic.is_finite() {
            ("Undefined (no residual variation)", false, false)
        } else if statistic < lower_critical {
            ("Positive autocorrelation", true, false)
        } else if statistic > 4.0 - lower_critical {
            ("Negative autocorrelation", false, true)
        } else if statistic < upper_critical || statistic > 4.0 - upper_critical {
            // Between d_L and d_U (or its mirror) the bounds test cannot decide.
            ("Inconclusive", false, false)
        } else {
            ("No significant autocorrelation", false, false)
        };

        Ok(Self {
            statistic,
            lower_critical,
            upper_critical,
            result: result.to_string(),
            has_positive_autocorr,
            has_negative_autocorr,
        })
    }
}

impl BreuschGodfreyTest {
    /// Compute Breusch-Godfrey LM test for serial correlation up to `n_lags`.
    ///
    /// The BG statistic is approximated as n*R² from the auxiliary regression
    /// of residuals on lagged residuals. Under H₀ it is asymptotically χ²(n_lags).
    /// Here we estimate R² via the sum of squared autocorrelations (a standard
    /// asymptotic approximation valid for large n).
    pub fn compute(values: &[f64], n_lags: usize) -> Result<Self> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let mut r_sq_sum = 0.0;
        for lag in 1..=n_lags {
            let rk = LjungBoxTest::calculate_autocorrelation(values, lag, mean)?;
            r_sq_sum += rk * rk;
        }
        // BG statistic ≈ n * R² (asymptotic approximation)
        let statistic = n * r_sq_sum;
        let df = n_lags;
        let p_value = chi2_sf(statistic, df as f64);
        let has_serial_correlation = p_value < 0.05;

        Ok(Self {
            statistic,
            p_value,
            df,
            n_lags,
            has_serial_correlation,
        })
    }
}

impl JarqueBeraTest {
    /// Compute Jarque-Bera test
    pub fn compute(values: &[f64]) -> Result<Self> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        // Calculate skewness and kurtosis
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let skewness = if std > 0.0 {
            values
                .iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 {
            values
                .iter()
                .map(|x| ((x - mean) / std).powi(4))
                .sum::<f64>()
                / n
                - 3.0
        } else {
            0.0
        };

        let skewness_stat = n * skewness.powi(2) / 6.0;
        let kurtosis_stat = n * kurtosis.powi(2) / 24.0;
        let statistic = skewness_stat + kurtosis_stat;

        // JB ~ χ²(2) under H₀
        let p_value = chi2_sf(statistic, 2.0);
        let is_normal = p_value > 0.05;

        Ok(Self {
            statistic,
            p_value,
            skewness_stat,
            kurtosis_stat,
            is_normal,
        })
    }
}

impl WhiteNoiseTest {
    /// Compute comprehensive white noise test
    pub fn compute(values: &[f64]) -> Result<Self> {
        let mut ljung_box_tests = Vec::new();

        // Test multiple lag values
        for &n_lags in &[5, 10, 15, 20] {
            if n_lags < values.len() / 4 {
                ljung_box_tests.push(LjungBoxTest::compute(values, n_lags)?);
            }
        }

        let variance_ratio_test = VarianceRatioTest::compute(values)?;
        let runs_test = RunsTest::compute(values)?;

        let is_white_noise = ljung_box_tests.iter().all(|test| !test.has_autocorrelation)
            && variance_ratio_test.is_random_walk
            && runs_test.is_random;

        Ok(Self {
            ljung_box_tests,
            variance_ratio_test,
            runs_test,
            is_white_noise,
        })
    }
}

impl VarianceRatioTest {
    /// Compute variance ratio test
    pub fn compute(values: &[f64]) -> Result<Self> {
        if values.len() < 10 {
            return Ok(Self {
                statistic: 0.0,
                p_value: 0.5,
                variance_ratio: 1.0,
                is_random_walk: true,
            });
        }

        // Calculate first differences
        let mut diff_values = Vec::new();
        for i in 1..values.len() {
            diff_values.push(values[i] - values[i - 1]);
        }

        // Calculate variance of first differences
        let mean_diff = diff_values.iter().sum::<f64>() / diff_values.len() as f64;
        let var_1 = diff_values
            .iter()
            .map(|x| (x - mean_diff).powi(2))
            .sum::<f64>()
            / diff_values.len() as f64;

        // Calculate variance of k-period differences (k=2)
        let k = 2;
        let mut k_diff_values = Vec::new();
        for i in k..values.len() {
            k_diff_values.push(values[i] - values[i - k]);
        }

        let mean_k_diff = k_diff_values.iter().sum::<f64>() / k_diff_values.len() as f64;
        let var_k = k_diff_values
            .iter()
            .map(|x| (x - mean_k_diff).powi(2))
            .sum::<f64>()
            / k_diff_values.len() as f64;

        let variance_ratio = if var_1 > 0.0 {
            var_k / (k as f64 * var_1)
        } else {
            1.0
        };

        // Lo-MacKinlay (1988) homoskedastic standardized statistic. Under the
        // random-walk null, VR(k) is asymptotically normal with variance
        //   Var(VR) = 2(2k − 1)(k − 1) / (3·k·N),
        // where N is the number of one-period observations. The two-sided
        // p-value comes from the real normal tail (not a fixed ladder).
        let big_n = diff_values.len() as f64;
        let kf = k as f64;
        let vr_var = 2.0 * (2.0 * kf - 1.0) * (kf - 1.0) / (3.0 * kf * big_n);
        let statistic = if vr_var > 0.0 {
            (variance_ratio - 1.0) / vr_var.sqrt()
        } else {
            0.0
        };
        let p_value = (2.0 * normal_sf(statistic.abs())).clamp(0.0, 1.0);
        let is_random_walk = p_value > 0.05;

        Ok(Self {
            statistic,
            p_value,
            variance_ratio,
            is_random_walk,
        })
    }
}

impl RunsTest {
    /// Compute runs test for randomness
    pub fn compute(values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            return Ok(Self {
                n_runs: 0,
                expected_runs: 0.0,
                statistic: 0.0,
                p_value: 0.5,
                is_random: true,
            });
        }

        // Convert to binary sequence (above/below median)
        let median = {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            }
        };

        let binary: Vec<bool> = values.iter().map(|&x| x >= median).collect();

        // Count runs
        let mut n_runs = 1;
        for i in 1..binary.len() {
            if binary[i] != binary[i - 1] {
                n_runs += 1;
            }
        }

        // Count positive and negative values
        let n_pos = binary.iter().filter(|&&x| x).count() as f64;
        let n_neg = binary.len() as f64 - n_pos;
        let n = binary.len() as f64;

        // Expected number of runs
        let expected_runs = if n > 0.0 {
            (2.0 * n_pos * n_neg) / n + 1.0
        } else {
            0.0
        };

        // Test statistic
        let variance = if n > 1.0 {
            (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n * n * (n - 1.0))
        } else {
            1.0
        };

        let statistic = if variance > 0.0 {
            (n_runs as f64 - expected_runs) / variance.sqrt()
        } else {
            0.0
        };

        // Wald-Wolfowitz runs test: the standardized run count is asymptotically
        // standard normal, so the two-sided p-value is 2·(1 − Φ(|z|)).
        let p_value = (2.0 * normal_sf(statistic.abs())).clamp(0.0, 1.0);
        let is_random = p_value > 0.05;

        Ok(Self {
            n_runs,
            expected_runs,
            statistic,
            p_value,
            is_random,
        })
    }
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;
