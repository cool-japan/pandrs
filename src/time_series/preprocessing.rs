//! Time Series Preprocessing Module
//!
//! This module provides comprehensive preprocessing capabilities for time series data,
//! including missing value handling, outlier detection and treatment, normalization,
//! differencing, and data transformation methods.

use crate::core::error::{Error, Result};
use crate::series::Series;
use crate::time_series::core::{DateTimeIndex, TimeSeries, TimeSeriesData};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Time series preprocessor with configurable options
#[derive(Debug, Clone)]
pub struct TimeSeriesPreprocessor {
    /// Missing value strategy
    pub missing_value_strategy: MissingValueStrategy,
    /// Outlier detection method
    pub outlier_detection: OutlierDetection,
    /// Normalization method
    pub normalization: Option<Normalization>,
    /// Differencing configuration
    pub differencing: Option<Differencing>,
    /// Smoothing configuration
    pub smoothing: Option<SmoothingConfig>,
    /// Resampling configuration
    pub resampling: Option<ResamplingConfig>,
}

/// Strategies for handling missing values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingValueStrategy {
    /// Remove rows with missing values
    DropNA,
    /// Forward fill missing values
    ForwardFill,
    /// Backward fill missing values
    BackwardFill,
    /// Linear interpolation
    LinearInterpolation,
    /// Spline interpolation
    SplineInterpolation,
    /// Fill with mean value
    FillMean,
    /// Fill with median value
    FillMedian,
    /// Fill with constant value
    FillConstant(f64),
    /// Seasonal decomposition and fill
    SeasonalFill,
}

/// Outlier detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierDetection {
    /// No outlier detection
    None,
    /// Z-score method
    ZScore { threshold: f64 },
    /// Modified Z-score method
    ModifiedZScore { threshold: f64 },
    /// IQR method
    IQR { multiplier: f64 },
    /// Isolation Forest
    IsolationForest { contamination: f64 },
    /// Statistical process control
    SPC { window: usize, sigma: f64 },
}

/// Outlier treatment methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierTreatment {
    /// Remove outliers
    Remove,
    /// Cap outliers at threshold
    Cap,
    /// Replace with median
    ReplaceWithMedian,
    /// Replace with mean
    ReplaceWithMean,
    /// Interpolate outliers
    Interpolate,
    /// Winsorize outliers
    Winsorize { percentile: f64 },
}

/// Normalization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Normalization {
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Min-max normalization to custom range
    MinMaxRange { min: f64, max: f64 },
    /// Z-score normalization (standardization)
    ZScore,
    /// Robust normalization using median and MAD
    Robust,
    /// Unit vector normalization
    UnitVector,
    /// Quantile normalization
    Quantile,
    /// Box-Cox transformation
    BoxCox { lambda: Option<f64> },
    /// Log transformation
    Log { base: f64 },
    /// Square root transformation
    Sqrt,
}

/// Differencing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Differencing {
    /// Regular differencing order
    pub order: usize,
    /// Seasonal differencing order
    pub seasonal_order: Option<usize>,
    /// Seasonal period
    pub seasonal_period: Option<usize>,
}

/// Smoothing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothingConfig {
    /// Smoothing method
    pub method: SmoothingMethod,
    /// Method-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Smoothing methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmoothingMethod {
    /// Moving average
    MovingAverage { window: usize },
    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64 },
    /// Savitzky-Golay filter
    SavitzkyGolay { window: usize, order: usize },
    /// LOWESS smoothing
    Lowess { fraction: f64 },
    /// Kalman filter
    KalmanFilter,
    /// Hodrick-Prescott filter
    HodrickPrescott { lambda: f64 },
}

/// Resampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResamplingConfig {
    /// Target frequency
    pub frequency: crate::time_series::core::Frequency,
    /// Aggregation method
    pub aggregation: AggregationMethod,
}

/// Aggregation methods for resampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Mean value
    Mean,
    /// Median value
    Median,
    /// Sum of values
    Sum,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// First value
    First,
    /// Last value
    Last,
    /// Standard deviation
    Std,
    /// Count of non-null values
    Count,
}

/// Preprocessing result containing the processed time series and metadata
#[derive(Debug, Clone)]
pub struct PreprocessingResult {
    /// Processed time series
    pub processed_series: TimeSeries,
    /// Applied transformations
    pub transformations: Vec<TransformationInfo>,
    /// Preprocessing statistics
    pub statistics: PreprocessingStatistics,
    /// Outlier information
    pub outlier_info: OutlierInfo,
}

/// Information about applied transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationInfo {
    /// Transformation type
    pub transformation_type: String,
    /// Parameters used
    pub parameters: HashMap<String, f64>,
    /// Number of affected values
    pub affected_values: usize,
    /// Transformation order
    pub order: usize,
}

/// Preprocessing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStatistics {
    /// Original series length
    pub original_length: usize,
    /// Final series length
    pub final_length: usize,
    /// Number of missing values handled
    pub missing_values_handled: usize,
    /// Number of outliers detected
    pub outliers_detected: usize,
    /// Original value range
    pub original_range: (f64, f64),
    /// Final value range
    pub final_range: (f64, f64),
    /// Mean before/after processing
    pub mean_before_after: (f64, f64),
    /// Std before/after processing
    pub std_before_after: (f64, f64),
}

/// Outlier detection and treatment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierInfo {
    /// Outlier indices in original series
    pub outlier_indices: Vec<usize>,
    /// Outlier values
    pub outlier_values: Vec<f64>,
    /// Detection method used
    pub detection_method: String,
    /// Treatment method used
    pub treatment_method: String,
    /// Detection threshold
    pub threshold: f64,
}

impl Default for TimeSeriesPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesPreprocessor {
    /// Create a new preprocessor with default settings
    pub fn new() -> Self {
        Self {
            missing_value_strategy: MissingValueStrategy::LinearInterpolation,
            outlier_detection: OutlierDetection::ModifiedZScore { threshold: 3.5 },
            normalization: None,
            differencing: None,
            smoothing: None,
            resampling: None,
        }
    }

    /// Set missing value strategy
    pub fn with_missing_value_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.missing_value_strategy = strategy;
        self
    }

    /// Set outlier detection method
    pub fn with_outlier_detection(mut self, detection: OutlierDetection) -> Self {
        self.outlier_detection = detection;
        self
    }

    /// Set normalization method
    pub fn with_normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = Some(normalization);
        self
    }

    /// Set differencing configuration
    pub fn with_differencing(mut self, differencing: Differencing) -> Self {
        self.differencing = Some(differencing);
        self
    }

    /// Set smoothing configuration
    pub fn with_smoothing(mut self, smoothing: SmoothingConfig) -> Self {
        self.smoothing = Some(smoothing);
        self
    }

    /// Set resampling configuration
    pub fn with_resampling(mut self, resampling: ResamplingConfig) -> Self {
        self.resampling = Some(resampling);
        self
    }

    /// Preprocess the time series
    pub fn preprocess(&self, ts: &TimeSeries) -> Result<PreprocessingResult> {
        let mut processed_series = ts.clone();
        let mut transformations = Vec::new();

        let original_stats = self.calculate_basic_stats(&processed_series)?;
        let original_length = processed_series.len();

        // Step 1: Handle missing values
        let (series_after_missing, missing_transform) =
            self.handle_missing_values(&processed_series)?;
        processed_series = series_after_missing;
        if let Some(transform) = missing_transform {
            transformations.push(transform);
        }

        // Step 2: Resample if configured
        if let Some(resampling_config) = &self.resampling {
            let (resampled_series, resample_transform) =
                self.resample_series(&processed_series, resampling_config)?;
            processed_series = resampled_series;
            transformations.push(resample_transform);
        }

        // Step 3: Detect and treat outliers
        let (series_after_outliers, outlier_transform, outlier_detection_info) =
            self.handle_outliers(&processed_series)?;
        processed_series = series_after_outliers;
        if let Some(transform) = outlier_transform {
            transformations.push(transform);
        }
        let outlier_info = outlier_detection_info;

        // Step 4: Apply smoothing if configured
        if let Some(smoothing_config) = &self.smoothing {
            let (smoothed_series, smooth_transform) =
                self.apply_smoothing(&processed_series, smoothing_config)?;
            processed_series = smoothed_series;
            transformations.push(smooth_transform);
        }

        // Step 5: Apply differencing if configured
        if let Some(differencing_config) = &self.differencing {
            let (differenced_series, diff_transform) =
                self.apply_differencing(&processed_series, differencing_config)?;
            processed_series = differenced_series;
            transformations.push(diff_transform);
        }

        // Step 6: Apply normalization if configured
        if let Some(normalization_method) = &self.normalization {
            let (normalized_series, norm_transform) =
                self.apply_normalization(&processed_series, normalization_method)?;
            processed_series = normalized_series;
            transformations.push(norm_transform);
        }

        let final_stats = self.calculate_basic_stats(&processed_series)?;
        let final_length = processed_series.len();

        let statistics = PreprocessingStatistics {
            original_length,
            final_length,
            missing_values_handled: transformations
                .iter()
                .find(|t| t.transformation_type.contains("missing"))
                .map(|t| t.affected_values)
                .unwrap_or(0),
            outliers_detected: outlier_info.outlier_indices.len(),
            original_range: (original_stats.0, original_stats.1),
            final_range: (final_stats.0, final_stats.1),
            mean_before_after: (original_stats.2, final_stats.2),
            std_before_after: (original_stats.3, final_stats.3),
        };

        Ok(PreprocessingResult {
            processed_series,
            transformations,
            statistics,
            outlier_info,
        })
    }

    /// Handle missing values
    fn handle_missing_values(
        &self,
        ts: &TimeSeries,
    ) -> Result<(TimeSeries, Option<TransformationInfo>)> {
        let missing_count = self.count_missing_values(ts);

        if missing_count == 0 {
            return Ok((ts.clone(), None));
        }

        let processed_series = match &self.missing_value_strategy {
            MissingValueStrategy::DropNA => self.drop_missing_values(ts)?,
            MissingValueStrategy::ForwardFill => ts.fillna_forward()?,
            MissingValueStrategy::BackwardFill => ts.fillna_backward()?,
            MissingValueStrategy::LinearInterpolation => self.linear_interpolation(ts)?,
            MissingValueStrategy::SplineInterpolation => self.spline_interpolation(ts)?,
            MissingValueStrategy::FillMean => self.fill_with_mean(ts)?,
            MissingValueStrategy::FillMedian => self.fill_with_median(ts)?,
            MissingValueStrategy::FillConstant(value) => self.fill_with_constant(ts, *value)?,
            MissingValueStrategy::SeasonalFill => self.seasonal_fill(ts)?,
        };

        let transform_info = TransformationInfo {
            transformation_type: format!("missing_value_{:?}", self.missing_value_strategy),
            parameters: HashMap::new(),
            affected_values: missing_count,
            order: 1,
        };

        Ok((processed_series, Some(transform_info)))
    }

    /// Handle outliers
    fn handle_outliers(
        &self,
        ts: &TimeSeries,
    ) -> Result<(TimeSeries, Option<TransformationInfo>, OutlierInfo)> {
        let outlier_indices = match &self.outlier_detection {
            OutlierDetection::None => Vec::new(),
            OutlierDetection::ZScore { threshold } => {
                self.detect_outliers_zscore(ts, *threshold)?
            }
            OutlierDetection::ModifiedZScore { threshold } => {
                self.detect_outliers_modified_zscore(ts, *threshold)?
            }
            OutlierDetection::IQR { multiplier } => self.detect_outliers_iqr(ts, *multiplier)?,
            OutlierDetection::IsolationForest { contamination } => {
                self.detect_outliers_isolation_forest(ts, *contamination)?
            }
            OutlierDetection::SPC { window, sigma } => {
                self.detect_outliers_spc(ts, *window, *sigma)?
            }
        };

        let outlier_values: Vec<f64> = outlier_indices
            .iter()
            .filter_map(|&idx| ts.values.get_f64(idx))
            .collect();

        let outlier_info = OutlierInfo {
            outlier_indices: outlier_indices.clone(),
            outlier_values,
            detection_method: format!("{:?}", self.outlier_detection),
            treatment_method: "Remove".to_string(), // Simplified for now
            threshold: self.get_outlier_threshold(),
        };

        if outlier_indices.is_empty() {
            return Ok((ts.clone(), None, outlier_info));
        }

        // For now, just remove outliers (could be extended to other treatments)
        let processed_series = self.remove_outliers(ts, &outlier_indices)?;

        let transform_info = TransformationInfo {
            transformation_type: "outlier_removal".to_string(),
            parameters: HashMap::new(),
            affected_values: outlier_indices.len(),
            order: 2,
        };

        Ok((processed_series, Some(transform_info), outlier_info))
    }

    /// Apply normalization
    fn apply_normalization(
        &self,
        ts: &TimeSeries,
        method: &Normalization,
    ) -> Result<(TimeSeries, TransformationInfo)> {
        // Collect valid values and their corresponding indices
        let mut valid_values = Vec::new();
        let mut valid_indices = Vec::new();

        for i in 0..ts.len() {
            if let Some(value) = ts.values.get_f64(i) {
                if value.is_finite() {
                    valid_values.push(value);
                    if let Some(timestamp) = ts.index.get(i) {
                        valid_indices.push(*timestamp);
                    }
                }
            }
        }

        let normalized_values = match method {
            Normalization::MinMax => self.minmax_normalize(&valid_values, 0.0, 1.0)?,
            Normalization::MinMaxRange { min, max } => {
                self.minmax_normalize(&valid_values, *min, *max)?
            }
            Normalization::ZScore => self.zscore_normalize(&valid_values)?,
            Normalization::Robust => self.robust_normalize(&valid_values)?,
            Normalization::UnitVector => self.unit_vector_normalize(&valid_values)?,
            Normalization::Quantile => self.quantile_normalize(&valid_values)?,
            Normalization::BoxCox { lambda } => self.boxcox_transform(&valid_values, *lambda)?,
            Normalization::Log { base } => self.log_transform(&valid_values, *base)?,
            Normalization::Sqrt => self.sqrt_transform(&valid_values)?,
        };

        // Create new index for the valid values
        let new_index = DateTimeIndex::new(valid_indices);
        let normalized_series = TimeSeriesData::from_vec(normalized_values);
        let processed_ts = TimeSeries::new(new_index, normalized_series)?;

        let mut parameters = HashMap::new();
        match method {
            Normalization::MinMaxRange { min, max } => {
                parameters.insert("min".to_string(), *min);
                parameters.insert("max".to_string(), *max);
            }
            Normalization::Log { base } => {
                parameters.insert("base".to_string(), *base);
            }
            Normalization::BoxCox { lambda } => {
                if let Some(l) = lambda {
                    parameters.insert("lambda".to_string(), *l);
                }
            }
            _ => {}
        }

        let transform_info = TransformationInfo {
            transformation_type: format!("normalization_{:?}", method),
            parameters,
            affected_values: valid_values.len(),
            order: 6,
        };

        Ok((processed_ts, transform_info))
    }

    /// Apply differencing
    fn apply_differencing(
        &self,
        ts: &TimeSeries,
        config: &Differencing,
    ) -> Result<(TimeSeries, TransformationInfo)> {
        let mut processed_series = ts.clone();

        // Apply regular differencing
        for _ in 0..config.order {
            processed_series = processed_series.diff(1)?;
        }

        // Apply seasonal differencing if configured
        if let (Some(seasonal_order), Some(seasonal_period)) =
            (config.seasonal_order, config.seasonal_period)
        {
            for _ in 0..seasonal_order {
                processed_series = processed_series.diff(seasonal_period)?;
            }
        }

        let mut parameters = HashMap::new();
        parameters.insert("order".to_string(), config.order as f64);
        if let Some(seasonal_order) = config.seasonal_order {
            parameters.insert("seasonal_order".to_string(), seasonal_order as f64);
        }
        if let Some(seasonal_period) = config.seasonal_period {
            parameters.insert("seasonal_period".to_string(), seasonal_period as f64);
        }

        let transform_info = TransformationInfo {
            transformation_type: "differencing".to_string(),
            parameters,
            affected_values: ts.len(),
            order: 5,
        };

        Ok((processed_series, transform_info))
    }

    /// Resample the time series to a new frequency, aggregating each bucket
    /// according to `config.aggregation`.
    ///
    /// The original points are grouped into the half-open intervals defined by
    /// the target-frequency grid and each bucket is reduced with the requested
    /// aggregation (Mean/Median/Sum/Min/Max/First/Last/Std/Count). The previous
    /// implementation hardcoded `ResampleMethod::Mean` and ignored
    /// `config.aggregation` entirely.
    fn resample_series(
        &self,
        ts: &TimeSeries,
        config: &ResamplingConfig,
    ) -> Result<(TimeSeries, TransformationInfo)> {
        let resampled = self.aggregate_resample(ts, config)?;

        let mut parameters = HashMap::new();
        parameters.insert("buckets".to_string(), resampled.len() as f64);

        let transform_info = TransformationInfo {
            transformation_type: format!("resampling_{:?}", config.aggregation),
            parameters,
            affected_values: ts.len(),
            order: 2,
        };

        Ok((resampled, transform_info))
    }

    /// Bucket-aggregate `ts` onto the target-frequency grid.
    fn aggregate_resample(&self, ts: &TimeSeries, config: &ResamplingConfig) -> Result<TimeSeries> {
        if ts.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot resample an empty time series".to_string(),
            ));
        }

        let start = *ts.index.start().ok_or_else(|| {
            Error::InvalidInput("Time series index has no start date".to_string())
        })?;
        let end = *ts
            .index
            .end()
            .ok_or_else(|| Error::InvalidInput("Time series index has no end date".to_string()))?;

        let grid = DateTimeIndex::date_range(start, end, config.frequency.clone())?;
        let m = grid.len();

        let mut new_values = Vec::with_capacity(m);
        for i in 0..m {
            let left = *grid.get(i).ok_or_else(|| {
                Error::InvalidInput("Resampling grid index out of range".to_string())
            })?;
            // Half-open bucket [left, right); the final bucket is closed.
            let right = if i + 1 < m {
                grid.get(i + 1).copied()
            } else {
                None
            };

            let mut bucket = Vec::new();
            for j in 0..ts.len() {
                if let Some(t) = ts.index.get(j) {
                    let in_bucket = *t >= left && right.map(|r| *t < r).unwrap_or(true);
                    if in_bucket {
                        if let Some(v) = ts.values.get_f64(j) {
                            if v.is_finite() {
                                bucket.push(v);
                            }
                        }
                    }
                }
            }

            new_values.push(aggregate_bucket(&bucket, &config.aggregation));
        }

        let series = TimeSeriesData::from_vec(new_values);
        TimeSeries::new(grid, series)
    }

    // Helper methods for missing value handling

    fn count_missing_values(&self, ts: &TimeSeries) -> usize {
        (0..ts.len())
            .map(|i| ts.values.get_f64(i).unwrap_or(f64::NAN))
            .filter(|v| !v.is_finite())
            .count()
    }

    fn drop_missing_values(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let mut valid_indices = Vec::new();

        for i in 0..ts.len() {
            if let Some(val) = ts.values.get_f64(i) {
                if val.is_finite() {
                    valid_indices.push(i);
                }
            }
        }

        if valid_indices.is_empty() {
            return Err(Error::InvalidInput(
                "No valid values remaining after dropping NAs".to_string(),
            ));
        }

        let new_timestamps: Vec<_> = valid_indices
            .iter()
            .filter_map(|&i| ts.index.get(i))
            .cloned()
            .collect();

        let new_values: Vec<f64> = valid_indices
            .iter()
            .filter_map(|&i| ts.values.get_f64(i))
            .collect();

        TimeSeries::from_vecs(new_timestamps, new_values)
    }

    fn linear_interpolation(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let mut interpolated_values = Vec::new();

        for i in 0..ts.len() {
            if let Some(val) = ts.values.get_f64(i) {
                if val.is_finite() {
                    interpolated_values.push(val);
                } else {
                    // Find previous and next valid values
                    let prev_valid = (0..i)
                        .rev()
                        .find_map(|j| ts.values.get_f64(j).filter(|v| v.is_finite()));
                    let next_valid = ((i + 1)..ts.len())
                        .find_map(|j| ts.values.get_f64(j).filter(|v| v.is_finite()));

                    let interpolated = match (prev_valid, next_valid) {
                        (Some(prev), Some(next)) => {
                            // Simple linear interpolation
                            (prev + next) / 2.0
                        }
                        (Some(prev), None) => prev,
                        (None, Some(next)) => next,
                        (None, None) => 0.0, // Fallback
                    };

                    interpolated_values.push(interpolated);
                }
            } else {
                interpolated_values.push(0.0); // Fallback
            }
        }

        let interpolated_series = TimeSeriesData::from_vec(interpolated_values);
        TimeSeries::new(ts.index.clone(), interpolated_series)
    }

    /// Fill missing values by **natural cubic spline** interpolation.
    ///
    /// Builds a natural cubic spline (second derivative zero at both ends)
    /// through the finite observations and evaluates it at the missing indices.
    /// Falls back to linear interpolation when fewer than three knots are
    /// available (a cubic spline is underdetermined there).
    fn spline_interpolation(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let n = ts.len();

        // Knots = the finite observations (x = position index, y = value).
        let mut knot_x = Vec::new();
        let mut knot_y = Vec::new();
        for i in 0..n {
            if let Some(v) = ts.values.get_f64(i) {
                if v.is_finite() {
                    knot_x.push(i as f64);
                    knot_y.push(v);
                }
            }
        }

        if knot_x.len() < 3 {
            return self.linear_interpolation(ts);
        }

        let second_derivs = natural_cubic_spline_second_derivatives(&knot_x, &knot_y);

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            match ts.values.get_f64(i) {
                Some(v) if v.is_finite() => out.push(v),
                _ => out.push(eval_cubic_spline(
                    &knot_x,
                    &knot_y,
                    &second_derivs,
                    i as f64,
                )),
            }
        }

        let series = TimeSeriesData::from_vec(out);
        TimeSeries::new(ts.index.clone(), series)
    }

    fn fill_with_mean(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let valid_values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if valid_values.is_empty() {
            return Err(Error::InvalidInput(
                "No valid values to calculate mean".to_string(),
            ));
        }

        let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
        self.fill_with_constant(ts, mean)
    }

    fn fill_with_median(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let mut valid_values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if valid_values.is_empty() {
            return Err(Error::InvalidInput(
                "No valid values to calculate median".to_string(),
            ));
        }

        valid_values.sort_by(|a, b| a.total_cmp(b));
        let median = if valid_values.len() % 2 == 0 {
            (valid_values[valid_values.len() / 2 - 1] + valid_values[valid_values.len() / 2]) / 2.0
        } else {
            valid_values[valid_values.len() / 2]
        };

        self.fill_with_constant(ts, median)
    }

    fn fill_with_constant(&self, ts: &TimeSeries, value: f64) -> Result<TimeSeries> {
        let filled_values: Vec<f64> = (0..ts.len())
            .map(|i| {
                if let Some(val) = ts.values.get_f64(i) {
                    if val.is_finite() {
                        val
                    } else {
                        value
                    }
                } else {
                    value
                }
            })
            .collect();

        let filled_series = TimeSeriesData::from_vec(filled_values);
        TimeSeries::new(ts.index.clone(), filled_series)
    }

    /// Fill missing values using the seasonal period.
    ///
    /// Each gap is imputed with the mean of the finite observations that share
    /// its seasonal phase (`index mod period`); phases with no observed value
    /// fall back to the global mean. The period is inferred from the series'
    /// frequency. This actually uses the seasonal structure rather than
    /// silently forward-filling.
    fn seasonal_fill(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        let n = ts.len();
        let period = self.infer_seasonal_period(ts).clamp(1, n.max(1));

        let observed: Vec<Option<f64>> = (0..n)
            .map(|i| ts.values.get_f64(i).filter(|v| v.is_finite()))
            .collect();

        // Per-phase sums/counts and the global mean fallback.
        let mut phase_sum = vec![0.0_f64; period];
        let mut phase_count = vec![0usize; period];
        let mut total_sum = 0.0;
        let mut total_count = 0usize;
        for (i, v) in observed.iter().enumerate() {
            if let Some(x) = v {
                phase_sum[i % period] += x;
                phase_count[i % period] += 1;
                total_sum += x;
                total_count += 1;
            }
        }
        let global_mean = if total_count > 0 {
            total_sum / total_count as f64
        } else {
            0.0
        };

        let mut out = Vec::with_capacity(n);
        for (i, v) in observed.iter().enumerate() {
            match v {
                Some(x) => out.push(*x),
                None => {
                    let phase = i % period;
                    let fill = if phase_count[phase] > 0 {
                        phase_sum[phase] / phase_count[phase] as f64
                    } else {
                        global_mean
                    };
                    out.push(fill);
                }
            }
        }

        let series = TimeSeriesData::from_vec(out);
        TimeSeries::new(ts.index.clone(), series)
    }

    /// Infer a seasonal period from the series frequency (defaults to 7).
    fn infer_seasonal_period(&self, ts: &TimeSeries) -> usize {
        use crate::time_series::core::Frequency;
        match &ts.index.frequency {
            Some(Frequency::Daily) => 7,
            Some(Frequency::Weekly) => 52,
            Some(Frequency::Monthly) => 12,
            Some(Frequency::Quarterly) => 4,
            Some(Frequency::Hour) => 24,
            Some(Frequency::Minute) => 60,
            _ => 7,
        }
    }

    // Helper methods for outlier detection

    fn detect_outliers_zscore(&self, ts: &TimeSeries, threshold: f64) -> Result<Vec<usize>> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        let mut outliers = Vec::new();

        for i in 0..ts.len() {
            if let Some(val) = ts.values.get_f64(i) {
                if val.is_finite() {
                    let z_score = if std > 0.0 {
                        (val - mean).abs() / std
                    } else {
                        0.0
                    };
                    if z_score > threshold {
                        outliers.push(i);
                    }
                }
            }
        }

        Ok(outliers)
    }

    fn detect_outliers_modified_zscore(
        &self,
        ts: &TimeSeries,
        threshold: f64,
    ) -> Result<Vec<usize>> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        // Calculate median
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate MAD
        let deviations: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.total_cmp(b));
        let mad = if sorted_deviations.len() % 2 == 0 {
            (sorted_deviations[sorted_deviations.len() / 2 - 1]
                + sorted_deviations[sorted_deviations.len() / 2])
                / 2.0
        } else {
            sorted_deviations[sorted_deviations.len() / 2]
        };

        let mut outliers = Vec::new();

        for i in 0..ts.len() {
            if let Some(val) = ts.values.get_f64(i) {
                if val.is_finite() {
                    let modified_z = if mad > 0.0 {
                        0.6745 * (val - median).abs() / mad
                    } else {
                        0.0
                    };
                    if modified_z > threshold {
                        outliers.push(i);
                    }
                }
            }
        }

        Ok(outliers)
    }

    fn detect_outliers_iqr(&self, ts: &TimeSeries, multiplier: f64) -> Result<Vec<usize>> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        let mut sorted = values;
        sorted.sort_by(|a, b| a.total_cmp(b));

        let n = sorted.len();
        let q1 = sorted[n / 4];
        let q3 = sorted[3 * n / 4];
        let iqr = q3 - q1;

        let lower_bound = q1 - multiplier * iqr;
        let upper_bound = q3 + multiplier * iqr;

        let mut outliers = Vec::new();

        for i in 0..ts.len() {
            if let Some(val) = ts.values.get_f64(i) {
                if val.is_finite() && (val < lower_bound || val > upper_bound) {
                    outliers.push(i);
                }
            }
        }

        Ok(outliers)
    }

    /// Detect outliers with a real Isolation Forest (`crate::ml::anomaly`).
    ///
    /// The finite values are fed into an `IsolationForest` (univariate feature);
    /// indices whose anomaly label is `-1` are returned, mapped back to their
    /// positions in the original series. This replaces the previous stub that
    /// silently returned an empty vector (i.e. claimed "no outliers").
    fn detect_outliers_isolation_forest(
        &self,
        ts: &TimeSeries,
        contamination: f64,
    ) -> Result<Vec<usize>> {
        use crate::dataframe::DataFrame;
        use crate::ml::{IsolationForest, UnsupervisedModel};

        // Collect finite values along with their original indices.
        let mut orig_indices = Vec::new();
        let mut feature = Vec::new();
        for i in 0..ts.len() {
            if let Some(v) = ts.values.get_f64(i) {
                if v.is_finite() {
                    orig_indices.push(i);
                    feature.push(v);
                }
            }
        }

        if feature.is_empty() {
            return Ok(Vec::new());
        }

        let contamination =
            if contamination.is_finite() && contamination > 0.0 && contamination < 0.5 {
                contamination
            } else {
                0.1
            };

        let mut df = DataFrame::new();
        df.add_column(
            "value".to_string(),
            Series::new(feature, Some("value".to_string()))?,
        )?;

        // Fixed seed for reproducible labelling.
        let mut forest = IsolationForest::new()
            .contamination(contamination)
            .random_seed(42);
        forest.fit(&df)?;

        let outliers = forest
            .labels()
            .iter()
            .enumerate()
            .filter_map(|(k, &label)| {
                if label == -1 {
                    orig_indices.get(k).copied()
                } else {
                    None
                }
            })
            .collect();

        Ok(outliers)
    }

    fn detect_outliers_spc(
        &self,
        ts: &TimeSeries,
        window: usize,
        sigma: f64,
    ) -> Result<Vec<usize>> {
        let mut outliers = Vec::new();

        for i in window..ts.len() {
            let window_values: Vec<f64> = (i.saturating_sub(window)..i)
                .filter_map(|j| ts.values.get_f64(j))
                .filter(|v| v.is_finite())
                .collect();

            if window_values.len() >= 3 {
                let mean = window_values.iter().sum::<f64>() / window_values.len() as f64;
                let variance = window_values
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / window_values.len() as f64;
                let std = variance.sqrt();

                if let Some(current_val) = ts.values.get_f64(i) {
                    if current_val.is_finite() {
                        let z_score = if std > 0.0 {
                            (current_val - mean).abs() / std
                        } else {
                            0.0
                        };
                        if z_score > sigma {
                            outliers.push(i);
                        }
                    }
                }
            }
        }

        Ok(outliers)
    }

    fn remove_outliers(&self, ts: &TimeSeries, outlier_indices: &[usize]) -> Result<TimeSeries> {
        let mut valid_indices = Vec::new();

        for i in 0..ts.len() {
            if !outlier_indices.contains(&i) {
                valid_indices.push(i);
            }
        }

        if valid_indices.is_empty() {
            return Err(Error::InvalidInput(
                "No valid values remaining after outlier removal".to_string(),
            ));
        }

        let new_timestamps: Vec<_> = valid_indices
            .iter()
            .filter_map(|&i| ts.index.get(i))
            .cloned()
            .collect();

        let new_values: Vec<f64> = valid_indices
            .iter()
            .filter_map(|&i| ts.values.get_f64(i))
            .collect();

        TimeSeries::from_vecs(new_timestamps, new_values)
    }

    fn get_outlier_threshold(&self) -> f64 {
        match &self.outlier_detection {
            OutlierDetection::ZScore { threshold } => *threshold,
            OutlierDetection::ModifiedZScore { threshold } => *threshold,
            OutlierDetection::IQR { multiplier } => *multiplier,
            OutlierDetection::IsolationForest { contamination } => *contamination,
            OutlierDetection::SPC { sigma, .. } => *sigma,
            _ => 0.0,
        }
    }

    // Helper methods for normalization

    fn minmax_normalize(
        &self,
        values: &[f64],
        target_min: f64,
        target_max: f64,
    ) -> Result<Vec<f64>> {
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if min_val == max_val {
            return Ok(vec![target_min; values.len()]);
        }

        let range = max_val - min_val;
        let target_range = target_max - target_min;

        Ok(values
            .iter()
            .map(|&x| target_min + (x - min_val) / range * target_range)
            .collect())
    }

    fn zscore_normalize(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return Ok(vec![0.0; values.len()]);
        }

        Ok(values.iter().map(|&x| (x - mean) / std).collect())
    }

    fn robust_normalize(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let deviations: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.total_cmp(b));
        let mad = if sorted_deviations.len() % 2 == 0 {
            (sorted_deviations[sorted_deviations.len() / 2 - 1]
                + sorted_deviations[sorted_deviations.len() / 2])
                / 2.0
        } else {
            sorted_deviations[sorted_deviations.len() / 2]
        };

        if mad == 0.0 {
            return Ok(vec![0.0; values.len()]);
        }

        Ok(values.iter().map(|&x| (x - median) / mad).collect())
    }

    fn unit_vector_normalize(&self, values: &[f64]) -> Result<Vec<f64>> {
        let norm = values.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm == 0.0 {
            return Ok(vec![0.0; values.len()]);
        }

        Ok(values.iter().map(|&x| x / norm).collect())
    }

    fn quantile_normalize(&self, values: &[f64]) -> Result<Vec<f64>> {
        let mut sorted_with_indices: Vec<(f64, usize)> = values
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        sorted_with_indices.sort_by(|a, b| a.0.total_cmp(&b.0));

        let n = values.len() as f64;
        let mut normalized = vec![0.0; values.len()];

        for (rank, &(_, original_idx)) in sorted_with_indices.iter().enumerate() {
            normalized[original_idx] = rank as f64 / (n - 1.0);
        }

        Ok(normalized)
    }

    fn boxcox_transform(&self, values: &[f64], lambda: Option<f64>) -> Result<Vec<f64>> {
        // Check for non-positive values
        if values.iter().any(|&x| x <= 0.0) {
            return Err(Error::InvalidInput(
                "Box-Cox transformation requires positive values".to_string(),
            ));
        }

        let lambda = lambda.unwrap_or(0.0); // Default to log transformation

        if lambda == 0.0 {
            Ok(values.iter().map(|&x| x.ln()).collect())
        } else {
            Ok(values
                .iter()
                .map(|&x| (x.powf(lambda) - 1.0) / lambda)
                .collect())
        }
    }

    fn log_transform(&self, values: &[f64], base: f64) -> Result<Vec<f64>> {
        if values.iter().any(|&x| x <= 0.0) {
            return Err(Error::InvalidInput(
                "Log transformation requires positive values".to_string(),
            ));
        }

        let log_base = base.ln();
        Ok(values.iter().map(|&x| x.ln() / log_base).collect())
    }

    fn sqrt_transform(&self, values: &[f64]) -> Result<Vec<f64>> {
        if values.iter().any(|&x| x < 0.0) {
            return Err(Error::InvalidInput(
                "Square root transformation requires non-negative values".to_string(),
            ));
        }

        Ok(values.iter().map(|&x| x.sqrt()).collect())
    }

    fn calculate_basic_stats(&self, ts: &TimeSeries) -> Result<(f64, f64, f64, f64)> {
        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.is_empty() {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        Ok((min, max, mean, std))
    }
}

/// Solve the linear system `A·x = b` by Gauss-Jordan elimination with partial
/// pivoting. Returns `None` if `A` is (numerically) singular. Used by the
/// Savitzky-Golay local polynomial fits.
pub(super) fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n {
        return None;
    }

    for col in 0..n {
        // Partial pivot.
        let mut pivot = col;
        let mut max_abs = a[col][col].abs();
        for r in (col + 1)..n {
            let v = a[r][col].abs();
            if v > max_abs {
                max_abs = v;
                pivot = r;
            }
        }
        if max_abs < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let diag = a[col][col];
        for j in col..n {
            a[col][j] /= diag;
        }
        b[col] /= diag;

        for r in 0..n {
            if r != col {
                let factor = a[r][col];
                if factor != 0.0 {
                    for j in col..n {
                        a[r][j] -= factor * a[col][j];
                    }
                    b[r] -= factor * b[col];
                }
            }
        }
    }

    Some(b)
}

/// Second derivatives of the natural cubic spline through `(x, y)` (knots must
/// be strictly increasing in `x`), obtained by the Thomas algorithm with the
/// natural boundary condition `M₀ = M_{n-1} = 0`.
fn natural_cubic_spline_second_derivatives(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut m = vec![0.0_f64; n];
    if n < 3 {
        return m;
    }

    // Tridiagonal system for the interior second derivatives.
    let mut sub = vec![0.0_f64; n]; // lower diagonal
    let mut diag = vec![0.0_f64; n];
    let mut sup = vec![0.0_f64; n]; // upper diagonal
    let mut rhs = vec![0.0_f64; n];

    for i in 1..n - 1 {
        let h_prev = x[i] - x[i - 1];
        let h_curr = x[i + 1] - x[i];
        sub[i] = h_prev;
        diag[i] = 2.0 * (h_prev + h_curr);
        sup[i] = h_curr;
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h_curr - (y[i] - y[i - 1]) / h_prev);
    }

    // Forward elimination (Thomas) over the interior rows 1..=n-2.
    for i in 2..n - 1 {
        if diag[i - 1].abs() < 1e-12 {
            return vec![0.0_f64; n];
        }
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }

    // Back substitution.
    if n >= 3 && diag[n - 2].abs() > 1e-12 {
        m[n - 2] = rhs[n - 2] / diag[n - 2];
        for i in (1..n - 2).rev() {
            if diag[i].abs() < 1e-12 {
                return vec![0.0_f64; n];
            }
            m[i] = (rhs[i] - sup[i] * m[i + 1]) / diag[i];
        }
    }

    m
}

/// Evaluate the natural cubic spline defined by knots `(x, y)` and second
/// derivatives `m` at the query point `xq`. Queries outside `[x₀, x_{n-1}]` are
/// clamped to the nearest knot value.
fn eval_cubic_spline(x: &[f64], y: &[f64], m: &[f64], xq: f64) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    if xq <= x[0] {
        return y[0];
    }
    if xq >= x[n - 1] {
        return y[n - 1];
    }

    // Locate the segment [x[k], x[k+1]] containing xq (binary search).
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if x[mid] <= xq {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let h = x[hi] - x[lo];
    if h.abs() < 1e-12 {
        return y[lo];
    }
    let a = (x[hi] - xq) / h;
    let b = (xq - x[lo]) / h;
    a * y[lo] + b * y[hi] + ((a * a * a - a) * m[lo] + (b * b * b - b) * m[hi]) * h * h / 6.0
}

/// Reduce a resampling bucket to a single value per the aggregation method.
/// Empty buckets yield `0` for Sum/Count and `NaN` otherwise.
fn aggregate_bucket(values: &[f64], method: &AggregationMethod) -> f64 {
    if values.is_empty() {
        return match method {
            AggregationMethod::Sum | AggregationMethod::Count => 0.0,
            _ => f64::NAN,
        };
    }

    match method {
        AggregationMethod::Mean => values.iter().sum::<f64>() / values.len() as f64,
        AggregationMethod::Median => {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let len = sorted.len();
            if len % 2 == 0 {
                (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
            } else {
                sorted[len / 2]
            }
        }
        AggregationMethod::Sum => values.iter().sum(),
        AggregationMethod::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
        AggregationMethod::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        AggregationMethod::First => values[0],
        AggregationMethod::Last => values[values.len() - 1],
        AggregationMethod::Std => {
            if values.len() < 2 {
                0.0
            } else {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / (values.len() - 1) as f64;
                var.sqrt()
            }
        }
        AggregationMethod::Count => values.len() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};

    fn create_test_series_with_outliers() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..50 {
            let timestamp = Utc
                .timestamp_opt(1640995200 + (i * 86400) as i64, 0)
                .single()
                .expect("timestamp should be unambiguous");
            let value = if i == 10 || i == 30 {
                100.0 // Outliers
            } else {
                10.0 + (i as f64 * 0.1).sin()
            };
            builder = builder.add_point(timestamp, value);
        }

        builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed")
    }

    #[test]
    fn test_missing_value_handling() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_missing_value_strategy(MissingValueStrategy::LinearInterpolation)
            .with_outlier_detection(OutlierDetection::None); // Disable outlier detection

        let result = preprocessor
            .preprocess(&ts)
            .expect("operation should succeed");
        assert_eq!(result.processed_series.len(), ts.len());
    }

    #[test]
    fn test_outlier_detection() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_outlier_detection(OutlierDetection::ModifiedZScore { threshold: 2.0 });

        let result = preprocessor
            .preprocess(&ts)
            .expect("operation should succeed");
        assert!(result.outlier_info.outlier_indices.len() > 0);
        assert!(result.processed_series.len() < ts.len()); // Outliers removed
    }

    #[test]
    fn test_normalization() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_normalization(Normalization::MinMax)
            .with_outlier_detection(OutlierDetection::None);

        let result = preprocessor
            .preprocess(&ts)
            .expect("operation should succeed");

        // Check that values are normalized to [0, 1]
        let values: Vec<f64> = (0..result.processed_series.len())
            .filter_map(|i| result.processed_series.values.get_f64(i))
            .collect();

        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(min_val >= -0.01); // Allow small numerical errors
        assert!(max_val <= 1.01);
    }

    #[test]
    fn test_differencing() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_differencing(Differencing {
                order: 1,
                seasonal_order: None,
                seasonal_period: None,
            })
            .with_outlier_detection(OutlierDetection::None);

        let result = preprocessor
            .preprocess(&ts)
            .expect("operation should succeed");

        // After differencing, the series should be shorter
        assert!(result.processed_series.len() <= ts.len());

        // Check that differencing transformation was applied
        assert!(result
            .transformations
            .iter()
            .any(|t| t.transformation_type.contains("differencing")));
    }

    #[test]
    fn test_smoothing() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_smoothing(SmoothingConfig {
                method: SmoothingMethod::MovingAverage { window: 5 },
                parameters: HashMap::new(),
            })
            .with_outlier_detection(OutlierDetection::None);

        let result = preprocessor
            .preprocess(&ts)
            .expect("operation should succeed");

        assert_eq!(result.processed_series.len(), ts.len());

        // Check that smoothing transformation was applied
        assert!(result
            .transformations
            .iter()
            .any(|t| t.transformation_type.contains("smoothing")));
    }

    #[test]
    fn test_comprehensive_preprocessing() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_missing_value_strategy(MissingValueStrategy::LinearInterpolation)
            .with_outlier_detection(OutlierDetection::ModifiedZScore { threshold: 2.0 })
            .with_smoothing(SmoothingConfig {
                method: SmoothingMethod::MovingAverage { window: 3 },
                parameters: HashMap::new(),
            })
            .with_normalization(Normalization::ZScore);

        let result = preprocessor
            .preprocess(&ts)
            .expect("operation should succeed");

        // Check that multiple transformations were applied
        assert!(result.transformations.len() > 1);

        // Check statistics
        assert!(result.statistics.outliers_detected > 0);
        assert!(result.statistics.final_length <= result.statistics.original_length);
    }

    #[test]
    fn test_zscore_normalization() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let preprocessor = TimeSeriesPreprocessor::new();
        let normalized = preprocessor
            .zscore_normalize(&values)
            .expect("operation should succeed");

        // Check that mean is approximately 0 and std is approximately 1
        let mean = normalized.iter().sum::<f64>() / normalized.len() as f64;
        let variance =
            normalized.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / normalized.len() as f64;
        let std = variance.sqrt();

        assert!((mean.abs()) < 1e-10);
        assert!((std - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_detection_methods() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        values.push(100.0); // Clear outlier

        let mut builder = TimeSeriesBuilder::new();
        for (i, &val) in values.iter().enumerate() {
            let timestamp = Utc
                .timestamp_opt(1640995200 + (i * 86400) as i64, 0)
                .single()
                .expect("timestamp should be unambiguous");
            builder = builder.add_point(timestamp, val);
        }
        let ts = builder.build().expect("operation should succeed");

        let preprocessor = TimeSeriesPreprocessor::new();

        // Test Z-score detection
        let zscore_outliers = preprocessor
            .detect_outliers_zscore(&ts, 2.0)
            .expect("operation should succeed");
        assert!(!zscore_outliers.is_empty());

        // Test Modified Z-score detection
        let modified_outliers = preprocessor
            .detect_outliers_modified_zscore(&ts, 2.0)
            .expect("operation should succeed");
        assert!(!modified_outliers.is_empty());

        // Test IQR detection
        let iqr_outliers = preprocessor
            .detect_outliers_iqr(&ts, 1.5)
            .expect("operation should succeed");
        assert!(!iqr_outliers.is_empty());
    }

    #[test]
    fn test_savitzky_golay_preserves_linear() {
        // A degree-2 Savitzky-Golay filter reproduces linear data exactly
        // (interior and boundary), confirming it honours `order` rather than
        // collapsing to a moving average.
        let pre = TimeSeriesPreprocessor::new();
        let mut builder = TimeSeriesBuilder::new();
        for i in 0..10 {
            let t = Utc
                .timestamp_opt(1_640_995_200 + (i * 86_400) as i64, 0)
                .single()
                .expect("timestamp should be unambiguous");
            builder = builder.add_point(t, 3.0 * i as f64 + 2.0);
        }
        let ts = builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed");

        let out = pre
            .savitzky_golay_smooth(&ts, 5, 2)
            .expect("operation should succeed");
        for i in 0..ts.len() {
            let expected = 3.0 * i as f64 + 2.0;
            let got = out.values.get_f64(i).expect("value");
            assert!(
                (got - expected).abs() < 1e-6,
                "SG should preserve a line at {i}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn test_spline_fills_linear_gap_exactly() {
        // A natural cubic spline through collinear knots is the line itself, so
        // a gap in linear data is recovered exactly.
        let pre = TimeSeriesPreprocessor::new();
        let mut builder = TimeSeriesBuilder::new();
        for i in 0..7 {
            let v = if i == 3 { f64::NAN } else { 2.0 * i as f64 };
            let t = Utc
                .timestamp_opt(1_640_995_200 + (i * 86_400) as i64, 0)
                .single()
                .expect("timestamp should be unambiguous");
            builder = builder.add_point(t, v);
        }
        let ts = builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed");

        let filled = pre
            .spline_interpolation(&ts)
            .expect("operation should succeed");
        let got = filled.values.get_f64(3).expect("value");
        assert!(
            (got - 6.0).abs() < 1e-6,
            "spline of collinear knots should give 6.0, got {got}"
        );
    }

    #[test]
    fn test_aggregate_bucket_methods() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::Sum), 6.0);
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::Min), 1.0);
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::Max), 3.0);
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::Mean), 2.0);
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::Median), 2.0);
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::First), 1.0);
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::Last), 3.0);
        assert_eq!(aggregate_bucket(&v, &AggregationMethod::Count), 3.0);
        assert!((aggregate_bucket(&v, &AggregationMethod::Std) - 1.0).abs() < 1e-9);

        // Empty buckets: Sum/Count yield 0, the rest NaN.
        assert_eq!(aggregate_bucket(&[], &AggregationMethod::Sum), 0.0);
        assert_eq!(aggregate_bucket(&[], &AggregationMethod::Count), 0.0);
        assert!(aggregate_bucket(&[], &AggregationMethod::Mean).is_nan());
    }
}
