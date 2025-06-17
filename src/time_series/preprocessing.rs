//! Time Series Preprocessing Module
//!
//! This module provides comprehensive preprocessing capabilities for time series data,
//! including missing value handling, outlier detection and treatment, normalization,
//! differencing, and data transformation methods.

use crate::core::error::{Error, Result};
use crate::series::Series;
use crate::time_series::core::{DateTimeIndex, TimeSeries, TimeSeriesBuilder, TimeSeriesData};
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

    /// Apply smoothing
    fn apply_smoothing(
        &self,
        ts: &TimeSeries,
        config: &SmoothingConfig,
    ) -> Result<(TimeSeries, TransformationInfo)> {
        let smoothed_series = match &config.method {
            SmoothingMethod::MovingAverage { window } => self.moving_average_smooth(ts, *window)?,
            SmoothingMethod::ExponentialSmoothing { alpha } => {
                self.exponential_smooth(ts, *alpha)?
            }
            SmoothingMethod::SavitzkyGolay { window, order } => {
                self.savitzky_golay_smooth(ts, *window, *order)?
            }
            SmoothingMethod::Lowess { fraction } => self.lowess_smooth(ts, *fraction)?,
            SmoothingMethod::KalmanFilter => self.kalman_smooth(ts)?,
            SmoothingMethod::HodrickPrescott { lambda } => {
                self.hodrick_prescott_smooth(ts, *lambda)?
            }
        };

        let transform_info = TransformationInfo {
            transformation_type: format!("smoothing_{:?}", config.method),
            parameters: config.parameters.clone(),
            affected_values: ts.len(),
            order: 4,
        };

        Ok((smoothed_series, transform_info))
    }

    /// Resample time series
    fn resample_series(
        &self,
        ts: &TimeSeries,
        config: &ResamplingConfig,
    ) -> Result<(TimeSeries, TransformationInfo)> {
        // For now, use the existing resample method from TimeSeries
        let resampled = ts.resample(
            config.frequency.clone(),
            crate::time_series::core::ResampleMethod::Mean,
        )?;

        let mut parameters = HashMap::new();
        parameters.insert("frequency".to_string(), 0.0); // Simplified

        let transform_info = TransformationInfo {
            transformation_type: "resampling".to_string(),
            parameters,
            affected_values: ts.len(),
            order: 2,
        };

        Ok((resampled, transform_info))
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

    fn spline_interpolation(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        // Simplified spline interpolation (cubic)
        self.linear_interpolation(ts) // For now, fall back to linear
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

        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

    fn seasonal_fill(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        // Simplified seasonal fill - use forward fill for now
        ts.fillna_forward()
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
        let mut value_idx = 0;

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
                    value_idx += 1;
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate MAD
        let deviations: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

    fn detect_outliers_isolation_forest(
        &self,
        _ts: &TimeSeries,
        _contamination: f64,
    ) -> Result<Vec<usize>> {
        // Simplified implementation - in practice would use proper isolation forest
        Ok(Vec::new())
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let deviations: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        sorted_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

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

    // Helper methods for smoothing

    fn moving_average_smooth(&self, ts: &TimeSeries, window: usize) -> Result<TimeSeries> {
        ts.rolling_mean(window)
    }

    fn exponential_smooth(&self, ts: &TimeSeries, alpha: f64) -> Result<TimeSeries> {
        let mut smoothed_values = Vec::with_capacity(ts.len());

        if let Some(first_val) = ts.values.get_f64(0) {
            smoothed_values.push(first_val);

            for i in 1..ts.len() {
                if let Some(current_val) = ts.values.get_f64(i) {
                    let prev_smooth = smoothed_values[i - 1];
                    let new_smooth = alpha * current_val + (1.0 - alpha) * prev_smooth;
                    smoothed_values.push(new_smooth);
                } else {
                    smoothed_values.push(smoothed_values[i - 1]);
                }
            }
        }

        let smoothed_series = TimeSeriesData::from_vec(smoothed_values);
        TimeSeries::new(ts.index.clone(), smoothed_series)
    }

    fn savitzky_golay_smooth(
        &self,
        ts: &TimeSeries,
        _window: usize,
        _order: usize,
    ) -> Result<TimeSeries> {
        // Simplified implementation - fall back to moving average
        self.moving_average_smooth(ts, _window)
    }

    fn lowess_smooth(&self, ts: &TimeSeries, _fraction: f64) -> Result<TimeSeries> {
        // Simplified implementation - fall back to moving average
        self.moving_average_smooth(ts, 5)
    }

    fn kalman_smooth(&self, ts: &TimeSeries) -> Result<TimeSeries> {
        // Simplified Kalman filter implementation
        self.exponential_smooth(ts, 0.3)
    }

    fn hodrick_prescott_smooth(&self, ts: &TimeSeries, _lambda: f64) -> Result<TimeSeries> {
        // Simplified HP filter - fall back to moving average
        self.moving_average_smooth(ts, 10)
    }

    // Helper method for calculating basic statistics

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
                .unwrap();
            let value = if i == 10 || i == 30 {
                100.0 // Outliers
            } else {
                10.0 + (i as f64 * 0.1).sin()
            };
            builder = builder.add_point(timestamp, value);
        }

        builder.frequency(Frequency::Daily).build().unwrap()
    }

    #[test]
    fn test_missing_value_handling() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_missing_value_strategy(MissingValueStrategy::LinearInterpolation)
            .with_outlier_detection(OutlierDetection::None); // Disable outlier detection

        let result = preprocessor.preprocess(&ts).unwrap();
        assert_eq!(result.processed_series.len(), ts.len());
    }

    #[test]
    fn test_outlier_detection() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_outlier_detection(OutlierDetection::ModifiedZScore { threshold: 2.0 });

        let result = preprocessor.preprocess(&ts).unwrap();
        assert!(result.outlier_info.outlier_indices.len() > 0);
        assert!(result.processed_series.len() < ts.len()); // Outliers removed
    }

    #[test]
    fn test_normalization() {
        let ts = create_test_series_with_outliers();
        let preprocessor = TimeSeriesPreprocessor::new()
            .with_normalization(Normalization::MinMax)
            .with_outlier_detection(OutlierDetection::None);

        let result = preprocessor.preprocess(&ts).unwrap();

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

        let result = preprocessor.preprocess(&ts).unwrap();

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

        let result = preprocessor.preprocess(&ts).unwrap();

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

        let result = preprocessor.preprocess(&ts).unwrap();

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
        let normalized = preprocessor.zscore_normalize(&values).unwrap();

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
                .unwrap();
            builder = builder.add_point(timestamp, val);
        }
        let ts = builder.build().unwrap();

        let preprocessor = TimeSeriesPreprocessor::new();

        // Test Z-score detection
        let zscore_outliers = preprocessor.detect_outliers_zscore(&ts, 2.0).unwrap();
        assert!(!zscore_outliers.is_empty());

        // Test Modified Z-score detection
        let modified_outliers = preprocessor
            .detect_outliers_modified_zscore(&ts, 2.0)
            .unwrap();
        assert!(!modified_outliers.is_empty());

        // Test IQR detection
        let iqr_outliers = preprocessor.detect_outliers_iqr(&ts, 1.5).unwrap();
        assert!(!iqr_outliers.is_empty());
    }
}
