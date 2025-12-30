//! Time Series Feature Extraction Module
//!
//! This module provides comprehensive feature extraction capabilities for time series data,
//! including statistical features, window-based features, and domain-specific features
//! for machine learning applications.

use crate::core::error::{Error, Result};
use crate::time_series::core::TimeSeries;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Feature set containing extracted features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    /// Statistical features
    pub statistical: StatisticalFeatures,
    /// Window-based features
    pub window: WindowFeatures,
    /// Frequency domain features
    pub frequency: FrequencyFeatures,
    /// Entropy and complexity features
    pub complexity: ComplexityFeatures,
    /// Custom features
    pub custom: HashMap<String, f64>,
}

/// Statistical features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFeatures {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Variance
    pub variance: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Range (max - min)
    pub range: f64,
    /// Median
    pub median: f64,
    /// Interquartile range
    pub iqr: f64,
    /// Coefficient of variation
    pub cv: f64,
    /// Mean absolute deviation
    pub mad: f64,
    /// Number of zero crossings
    pub zero_crossings: usize,
    /// Number of peaks
    pub peaks: usize,
    /// Number of valleys
    pub valleys: usize,
}

/// Window-based features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowFeatures {
    /// Moving average features for different window sizes
    pub moving_averages: HashMap<usize, Vec<f64>>,
    /// Moving standard deviation features
    pub moving_stds: HashMap<usize, Vec<f64>>,
    /// Rolling correlation with lag
    pub rolling_correlations: HashMap<usize, Vec<f64>>,
    /// Exponential moving averages (alpha, values)
    pub ema_features: Vec<(f64, Vec<f64>)>,
    /// Bollinger band features
    pub bollinger_bands: BollingerBandFeatures,
    /// Trend strength by window
    pub trend_strengths: HashMap<usize, f64>,
}

/// Bollinger band features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerBandFeatures {
    /// Upper band
    pub upper_band: Vec<f64>,
    /// Lower band
    pub lower_band: Vec<f64>,
    /// Middle band (moving average)
    pub middle_band: Vec<f64>,
    /// Percentage within bands
    pub pct_within_bands: f64,
    /// Band width
    pub band_width: Vec<f64>,
    /// %B indicator
    pub percent_b: Vec<f64>,
}

/// Frequency domain features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyFeatures {
    /// Dominant frequency
    pub dominant_frequency: f64,
    /// Power spectral density
    pub psd: Vec<f64>,
    /// Frequencies corresponding to PSD
    pub frequencies: Vec<f64>,
    /// Spectral centroid
    pub spectral_centroid: f64,
    /// Spectral bandwidth
    pub spectral_bandwidth: f64,
    /// Spectral rolloff
    pub spectral_rolloff: f64,
    /// Spectral flux
    pub spectral_flux: f64,
    /// Harmonic-to-noise ratio
    pub hnr: f64,
}

/// Complexity and entropy features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFeatures {
    /// Approximate entropy
    pub approximate_entropy: f64,
    /// Sample entropy
    pub sample_entropy: f64,
    /// Permutation entropy
    pub permutation_entropy: f64,
    /// Spectral entropy
    pub spectral_entropy: f64,
    /// Lempel-Ziv complexity
    pub lempel_ziv_complexity: f64,
    /// Fractal dimension
    pub fractal_dimension: f64,
    /// Hurst exponent
    pub hurst_exponent: f64,
    /// Detrended fluctuation analysis
    pub dfa_alpha: f64,
}

/// Time series feature extractor
pub struct TimeSeriesFeatureExtractor {
    /// Window sizes for rolling features
    pub window_sizes: Vec<usize>,
    /// EMA alpha values
    pub ema_alphas: Vec<f64>,
    /// Extract frequency features
    pub include_frequency: bool,
    /// Extract complexity features
    pub include_complexity: bool,
    /// Custom feature extractors
    pub custom_extractors: Vec<Box<dyn FeatureExtractor>>,
}

/// Trait for custom feature extractors
pub trait FeatureExtractor {
    /// Extract features from time series
    fn extract(&self, ts: &TimeSeries) -> Result<HashMap<String, f64>>;

    /// Get feature names
    fn feature_names(&self) -> Vec<String>;
}

impl Default for TimeSeriesFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesFeatureExtractor {
    /// Create a new feature extractor with default settings
    pub fn new() -> Self {
        Self {
            window_sizes: vec![5, 10, 20, 50],
            ema_alphas: vec![0.1, 0.3, 0.5],
            include_frequency: true,
            include_complexity: false, // Expensive to compute
            custom_extractors: Vec::new(),
        }
    }

    /// Set window sizes for rolling features
    pub fn with_window_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.window_sizes = sizes;
        self
    }

    /// Set EMA alpha values
    pub fn with_ema_alphas(mut self, alphas: Vec<f64>) -> Self {
        self.ema_alphas = alphas;
        self
    }

    /// Enable/disable frequency domain features
    pub fn with_frequency_features(mut self, include: bool) -> Self {
        self.include_frequency = include;
        self
    }

    /// Enable/disable complexity features
    pub fn with_complexity_features(mut self, include: bool) -> Self {
        self.include_complexity = include;
        self
    }

    /// Add custom feature extractor
    pub fn add_custom_extractor(mut self, extractor: Box<dyn FeatureExtractor>) -> Self {
        self.custom_extractors.push(extractor);
        self
    }

    /// Extract all features from time series
    pub fn extract_features(&self, ts: &TimeSeries) -> Result<FeatureSet> {
        if ts.is_empty() {
            return Err(Error::InvalidInput("Empty time series".to_string()));
        }

        let values: Vec<f64> = (0..ts.len())
            .filter_map(|i| ts.values.get_f64(i))
            .filter(|v| v.is_finite())
            .collect();

        if values.is_empty() {
            return Err(Error::InvalidInput(
                "No valid values in time series".to_string(),
            ));
        }

        // Extract statistical features
        let statistical = self.extract_statistical_features(&values)?;

        // Extract window-based features
        let window = self.extract_window_features(&values)?;

        // Extract frequency domain features
        let frequency = if self.include_frequency {
            self.extract_frequency_features(&values)?
        } else {
            FrequencyFeatures::default()
        };

        // Extract complexity features
        let complexity = if self.include_complexity {
            self.extract_complexity_features(&values)?
        } else {
            ComplexityFeatures::default()
        };

        // Extract custom features
        let mut custom = HashMap::new();
        for extractor in &self.custom_extractors {
            let features = extractor.extract(ts)?;
            custom.extend(features);
        }

        Ok(FeatureSet {
            statistical,
            window,
            frequency,
            complexity,
            custom,
        })
    }

    /// Extract statistical features
    fn extract_statistical_features(&self, values: &[f64]) -> Result<StatisticalFeatures> {
        let n = values.len() as f64;

        // Basic statistics
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        // Min, max, range
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        // Median and quantiles
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        let iqr = sorted[q3_idx] - sorted[q1_idx];

        // Higher order moments
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

        // Coefficient of variation
        let cv = if mean != 0.0 { std / mean.abs() } else { 0.0 };

        // Mean absolute deviation
        let mad = values.iter().map(|x| (x - mean).abs()).sum::<f64>() / n;

        // Zero crossings
        let zero_crossings = self.count_zero_crossings(values);

        // Peaks and valleys
        let (peaks, valleys) = self.count_peaks_valleys(values);

        Ok(StatisticalFeatures {
            mean,
            std,
            variance,
            skewness,
            kurtosis,
            min,
            max,
            range,
            median,
            iqr,
            cv,
            mad,
            zero_crossings,
            peaks,
            valleys,
        })
    }

    /// Extract window-based features
    fn extract_window_features(&self, values: &[f64]) -> Result<WindowFeatures> {
        let mut moving_averages = HashMap::new();
        let mut moving_stds = HashMap::new();
        let mut rolling_correlations = HashMap::new();
        let mut trend_strengths = HashMap::new();

        // Calculate features for each window size
        for &window_size in &self.window_sizes {
            if window_size < values.len() {
                moving_averages.insert(window_size, self.moving_average(values, window_size)?);
                moving_stds.insert(window_size, self.moving_std(values, window_size)?);
                rolling_correlations
                    .insert(window_size, self.rolling_correlation(values, window_size)?);
                trend_strengths.insert(window_size, self.trend_strength(values, window_size)?);
            }
        }

        // EMA features
        let mut ema_features = Vec::new();
        for &alpha in &self.ema_alphas {
            ema_features.push((alpha, self.exponential_moving_average(values, alpha)?));
        }

        // Bollinger bands
        let bollinger_bands = self.bollinger_bands(values, 20, 2.0)?;

        Ok(WindowFeatures {
            moving_averages,
            moving_stds,
            rolling_correlations,
            ema_features,
            bollinger_bands,
            trend_strengths,
        })
    }

    /// Extract frequency domain features
    fn extract_frequency_features(&self, values: &[f64]) -> Result<FrequencyFeatures> {
        // Simplified FFT implementation (in practice, would use a proper FFT library)
        let n = values.len();
        let mut psd = Vec::new();
        let mut frequencies = Vec::new();

        // Calculate power spectral density using autocorrelation method
        for k in 0..n / 2 {
            let freq = k as f64 / n as f64;
            frequencies.push(freq);

            let mut power = 0.0;
            for lag in 0..std::cmp::min(n / 4, 50) {
                let autocorr = self.calculate_autocorrelation(values, lag)?;
                power += autocorr * (2.0 * PI * freq * lag as f64).cos();
            }
            psd.push(power.abs());
        }

        // Find dominant frequency
        let dominant_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let dominant_frequency = frequencies[dominant_idx];

        // Spectral centroid
        let total_power: f64 = psd.iter().sum();
        let spectral_centroid = if total_power > 0.0 {
            frequencies
                .iter()
                .zip(&psd)
                .map(|(freq, power)| freq * power)
                .sum::<f64>()
                / total_power
        } else {
            0.0
        };

        // Spectral bandwidth
        let spectral_bandwidth = if total_power > 0.0 {
            frequencies
                .iter()
                .zip(&psd)
                .map(|(freq, power)| (freq - spectral_centroid).powi(2) * power)
                .sum::<f64>()
                / total_power
        } else {
            0.0
        }
        .sqrt();

        // Spectral rolloff (frequency below which 85% of energy is contained)
        let mut cumulative_power = 0.0;
        let rolloff_threshold = 0.85 * total_power;
        let spectral_rolloff = frequencies
            .iter()
            .zip(&psd)
            .find(|(_, power)| {
                cumulative_power += *power;
                cumulative_power >= rolloff_threshold
            })
            .map(|(freq, _)| *freq)
            .unwrap_or(0.5);

        // Spectral flux (simplified)
        let spectral_flux = psd
            .windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .sum::<f64>()
            / psd.len() as f64;

        // Harmonic-to-noise ratio (simplified)
        let hnr = if psd.len() > 1 {
            let signal_power = psd[1..].iter().sum::<f64>();
            let noise_power = psd[0];
            if noise_power > 0.0 {
                10.0 * (signal_power / noise_power).log10()
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(FrequencyFeatures {
            dominant_frequency,
            psd,
            frequencies,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_flux,
            hnr,
        })
    }

    /// Extract complexity features
    fn extract_complexity_features(&self, values: &[f64]) -> Result<ComplexityFeatures> {
        // Approximate entropy
        let approximate_entropy = self.approximate_entropy(values, 2, 0.2)?;

        // Sample entropy
        let sample_entropy = self.sample_entropy(values, 2, 0.2)?;

        // Permutation entropy
        let permutation_entropy = self.permutation_entropy(values, 3)?;

        // Spectral entropy (simplified)
        let spectral_entropy = self.spectral_entropy(values)?;

        // Lempel-Ziv complexity
        let lempel_ziv_complexity = self.lempel_ziv_complexity(values)?;

        // Fractal dimension (box-counting method, simplified)
        let fractal_dimension = self.fractal_dimension(values)?;

        // Hurst exponent
        let hurst_exponent = self.hurst_exponent(values)?;

        // Detrended fluctuation analysis
        let dfa_alpha = self.detrended_fluctuation_analysis(values)?;

        Ok(ComplexityFeatures {
            approximate_entropy,
            sample_entropy,
            permutation_entropy,
            spectral_entropy,
            lempel_ziv_complexity,
            fractal_dimension,
            hurst_exponent,
            dfa_alpha,
        })
    }

    /// Count zero crossings (crossings around the mean of the series)
    fn count_zero_crossings(&self, values: &[f64]) -> usize {
        if values.len() < 2 {
            return 0;
        }

        // Calculate mean
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        // Count sign changes in mean-centered values
        values
            .windows(2)
            .filter(|window| (window[0] - mean) * (window[1] - mean) < 0.0)
            .count()
    }

    /// Count peaks and valleys
    fn count_peaks_valleys(&self, values: &[f64]) -> (usize, usize) {
        let mut peaks = 0;
        let mut valleys = 0;

        for i in 1..(values.len() - 1) {
            if values[i] > values[i - 1] && values[i] > values[i + 1] {
                peaks += 1;
            } else if values[i] < values[i - 1] && values[i] < values[i + 1] {
                valleys += 1;
            }
        }

        (peaks, valleys)
    }

    /// Calculate moving average
    fn moving_average(&self, values: &[f64], window: usize) -> Result<Vec<f64>> {
        let mut ma = Vec::new();

        for i in 0..values.len() {
            if i < window - 1 {
                ma.push(f64::NAN);
            } else {
                let sum: f64 = values[i + 1 - window..=i].iter().sum();
                ma.push(sum / window as f64);
            }
        }

        Ok(ma)
    }

    /// Calculate moving standard deviation
    fn moving_std(&self, values: &[f64], window: usize) -> Result<Vec<f64>> {
        let mut std = Vec::new();

        for i in 0..values.len() {
            if i < window - 1 {
                std.push(f64::NAN);
            } else {
                let window_values = &values[i + 1 - window..=i];
                let mean = window_values.iter().sum::<f64>() / window as f64;
                let variance = window_values
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / window as f64;
                std.push(variance.sqrt());
            }
        }

        Ok(std)
    }

    /// Calculate rolling correlation with lag 1
    fn rolling_correlation(&self, values: &[f64], window: usize) -> Result<Vec<f64>> {
        let mut corr = Vec::new();

        for i in 0..values.len() {
            if i < window {
                corr.push(f64::NAN);
            } else {
                let window_values = &values[i + 1 - window..=i];
                let lagged_values = &values[i - window..i];

                if window_values.len() == lagged_values.len() {
                    let correlation = self.calculate_correlation(window_values, lagged_values)?;
                    corr.push(correlation);
                } else {
                    corr.push(f64::NAN);
                }
            }
        }

        Ok(corr)
    }

    /// Calculate exponential moving average
    fn exponential_moving_average(&self, values: &[f64], alpha: f64) -> Result<Vec<f64>> {
        let mut ema = Vec::with_capacity(values.len());

        if values.is_empty() {
            return Ok(ema);
        }

        ema.push(values[0]);

        for &value in &values[1..] {
            let prev_ema = ema[ema.len() - 1];
            ema.push(alpha * value + (1.0 - alpha) * prev_ema);
        }

        Ok(ema)
    }

    /// Calculate Bollinger bands
    fn bollinger_bands(
        &self,
        values: &[f64],
        window: usize,
        std_dev: f64,
    ) -> Result<BollingerBandFeatures> {
        let ma = self.moving_average(values, window)?;
        let std = self.moving_std(values, window)?;

        let mut upper_band = Vec::new();
        let mut lower_band = Vec::new();
        let mut band_width = Vec::new();
        let mut percent_b = Vec::new();

        for i in 0..values.len() {
            if ma[i].is_finite() && std[i].is_finite() {
                let upper = ma[i] + std_dev * std[i];
                let lower = ma[i] - std_dev * std[i];
                upper_band.push(upper);
                lower_band.push(lower);
                band_width.push(upper - lower);

                // %B indicator
                if upper != lower {
                    percent_b.push((values[i] - lower) / (upper - lower));
                } else {
                    percent_b.push(0.5);
                }
            } else {
                upper_band.push(f64::NAN);
                lower_band.push(f64::NAN);
                band_width.push(f64::NAN);
                percent_b.push(f64::NAN);
            }
        }

        // Percentage within bands
        let within_bands = values
            .iter()
            .zip(&upper_band)
            .zip(&lower_band)
            .filter(|((value, upper), lower)| {
                value.is_finite()
                    && upper.is_finite()
                    && lower.is_finite()
                    && **value >= **lower
                    && **value <= **upper
            })
            .count();

        let pct_within_bands = within_bands as f64 / values.len() as f64;

        Ok(BollingerBandFeatures {
            upper_band,
            lower_band,
            middle_band: ma,
            pct_within_bands,
            band_width,
            percent_b,
        })
    }

    /// Calculate trend strength in window
    fn trend_strength(&self, values: &[f64], window: usize) -> Result<f64> {
        if values.len() < window {
            return Ok(0.0);
        }

        let window_values = &values[values.len() - window..];
        let x_values: Vec<f64> = (0..window).map(|i| i as f64).collect();

        // Linear regression
        let n = window as f64;
        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = window_values.iter().sum::<f64>();
        let sum_xy = x_values
            .iter()
            .zip(window_values)
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x * x).sum::<f64>();
        let sum_y2 = window_values.iter().map(|y| y * y).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        // Calculate R-squared
        let y_mean = sum_y / n;
        let ss_tot = window_values
            .iter()
            .map(|y| (y - y_mean).powi(2))
            .sum::<f64>();
        let ss_res = x_values
            .iter()
            .zip(window_values)
            .map(|(x, y)| {
                let predicted = slope * x + (sum_y - slope * sum_x) / n;
                (y - predicted).powi(2)
            })
            .sum::<f64>();

        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        Ok(r_squared)
    }

    /// Calculate autocorrelation
    fn calculate_autocorrelation(&self, values: &[f64], lag: usize) -> Result<f64> {
        if lag >= values.len() {
            return Ok(0.0);
        }

        let n = values.len() - lag;
        let mean = values.iter().sum::<f64>() / values.len() as f64;

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

    /// Calculate correlation between two series
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(0.0);
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Calculate approximate entropy
    fn approximate_entropy(&self, values: &[f64], m: usize, r: f64) -> Result<f64> {
        if values.len() < m + 1 {
            return Ok(0.0);
        }

        let n = values.len();
        let mut phi = Vec::new();

        for pattern_len in m..=m + 1 {
            let mut c = vec![0.0; n - pattern_len + 1];

            for i in 0..(n - pattern_len + 1) {
                for j in 0..(n - pattern_len + 1) {
                    let mut max_diff: f64 = 0.0;
                    for k in 0..pattern_len {
                        max_diff = max_diff.max((values[i + k] - values[j + k]).abs());
                    }
                    if max_diff <= r {
                        c[i] += 1.0;
                    }
                }
                c[i] /= (n - pattern_len + 1) as f64;
            }

            let phi_val = c.iter().filter(|&&x| x > 0.0).map(|&x| x.ln()).sum::<f64>()
                / (n - pattern_len + 1) as f64;

            phi.push(phi_val);
        }

        if phi.len() == 2 {
            Ok(phi[0] - phi[1])
        } else {
            Ok(0.0)
        }
    }

    /// Calculate sample entropy
    fn sample_entropy(&self, values: &[f64], m: usize, r: f64) -> Result<f64> {
        if values.len() < m + 1 {
            return Ok(0.0);
        }

        let n = values.len();
        let mut a: f64 = 0.0;
        let mut b: f64 = 0.0;

        for i in 0..(n - m) {
            for j in (i + 1)..(n - m) {
                let mut match_m = true;

                // Check pattern of length m
                for k in 0..m {
                    if (values[i + k] - values[j + k]).abs() > r {
                        match_m = false;
                        break;
                    }
                }

                if match_m {
                    b += 1.0;

                    // Check pattern of length m+1
                    if (values[i + m] - values[j + m]).abs() <= r {
                        a += 1.0;
                    }
                }
            }
        }

        if b == 0.0 {
            Ok(0.0)
        } else {
            Ok(-(a / b).ln())
        }
    }

    /// Calculate permutation entropy
    fn permutation_entropy(&self, values: &[f64], order: usize) -> Result<f64> {
        if values.len() < order {
            return Ok(0.0);
        }

        let mut permutation_counts = HashMap::new();
        let total_patterns = values.len() - order + 1;

        for i in 0..total_patterns {
            let pattern = &values[i..i + order];
            let mut indices: Vec<usize> = (0..order).collect();
            indices.sort_by(|&a, &b| pattern[a].partial_cmp(&pattern[b]).unwrap());

            let permutation = indices
                .into_iter()
                .enumerate()
                .map(|(rank, _)| rank)
                .collect::<Vec<_>>();

            *permutation_counts.entry(permutation).or_insert(0) += 1;
        }

        let entropy = permutation_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total_patterns as f64;
                -p * p.ln()
            })
            .sum::<f64>();

        Ok(entropy)
    }

    /// Calculate spectral entropy
    fn spectral_entropy(&self, values: &[f64]) -> Result<f64> {
        // Calculate power spectral density
        let mut psd = Vec::new();
        for k in 0..values.len() / 2 {
            let mut power = 0.0;
            for lag in 0..std::cmp::min(values.len() / 4, 20) {
                let autocorr = self.calculate_autocorrelation(values, lag)?;
                power += autocorr * (2.0 * PI * k as f64 * lag as f64 / values.len() as f64).cos();
            }
            psd.push(power.abs());
        }

        // Normalize PSD
        let total_power: f64 = psd.iter().sum();
        if total_power == 0.0 {
            return Ok(0.0);
        }

        let entropy = psd
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| {
                let normalized = p / total_power;
                -normalized * normalized.ln()
            })
            .sum::<f64>();

        Ok(entropy)
    }

    /// Calculate Lempel-Ziv complexity (simplified)
    fn lempel_ziv_complexity(&self, values: &[f64]) -> Result<f64> {
        // Convert to binary sequence (simplified)
        let median = values.iter().sum::<f64>() / values.len() as f64;
        let binary: Vec<u8> = values
            .iter()
            .map(|&x| if x >= median { 1 } else { 0 })
            .collect();

        let mut complexity = 1;
        let mut i = 0;

        while i < binary.len() {
            let mut j = 1;
            while i + j <= binary.len() {
                let pattern = &binary[i..i + j];
                let mut found = false;

                for k in 0..i {
                    if k + j <= i && &binary[k..k + j] == pattern {
                        found = true;
                        break;
                    }
                }

                if !found {
                    break;
                }
                j += 1;
            }

            complexity += 1;
            i += j;
        }

        Ok(complexity as f64 / values.len() as f64)
    }

    /// Calculate fractal dimension (simplified box-counting)
    fn fractal_dimension(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Ok(0.0);
        }

        // Simplified fractal dimension calculation
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range == 0.0 {
            return Ok(1.0);
        }

        let mut log_scales = Vec::new();
        let mut log_counts = Vec::new();

        for &scale in &[2, 4, 8, 16, 32] {
            if scale < values.len() {
                let box_size = range / scale as f64;
                let mut occupied_boxes = std::collections::HashSet::new();

                for &value in values {
                    let box_index = ((value - min_val) / box_size).floor() as i32;
                    occupied_boxes.insert(box_index);
                }

                log_scales.push((1.0 / scale as f64).ln());
                log_counts.push((occupied_boxes.len() as f64).ln());
            }
        }

        if log_scales.len() < 2 {
            return Ok(1.5); // Default fractal dimension
        }

        // Linear regression to find slope
        let n = log_scales.len() as f64;
        let sum_x = log_scales.iter().sum::<f64>();
        let sum_y = log_counts.iter().sum::<f64>();
        let sum_xy = log_scales
            .iter()
            .zip(&log_counts)
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = log_scales.iter().map(|x| x * x).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        Ok(slope.abs())
    }

    /// Calculate Hurst exponent using R/S analysis
    fn hurst_exponent(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 10 {
            return Ok(0.5);
        }

        let mut log_rs = Vec::new();
        let mut log_n = Vec::new();

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let cumulative_devs: Vec<f64> = values
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x - mean;
                Some(*acc)
            })
            .collect();

        for &n in &[10, 20, 50, 100] {
            if n < values.len() {
                let range = cumulative_devs[..n]
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
                    - cumulative_devs[..n]
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min);

                let std_dev =
                    values[..n].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
                let std_dev = std_dev.sqrt();

                if std_dev > 0.0 {
                    let rs = range / std_dev;
                    log_rs.push(rs.ln());
                    log_n.push((n as f64).ln());
                }
            }
        }

        if log_n.len() < 2 {
            return Ok(0.5);
        }

        // Linear regression
        let n = log_n.len() as f64;
        let sum_x = log_n.iter().sum::<f64>();
        let sum_y = log_rs.iter().sum::<f64>();
        let sum_xy = log_n.iter().zip(&log_rs).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = log_n.iter().map(|x| x * x).sum::<f64>();

        let hurst = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        Ok(hurst.max(0.0).min(1.0))
    }

    /// Detrended fluctuation analysis
    fn detrended_fluctuation_analysis(&self, values: &[f64]) -> Result<f64> {
        if values.len() < 20 {
            return Ok(1.0);
        }

        // Create integrated series
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let integrated: Vec<f64> = values
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x - mean;
                Some(*acc)
            })
            .collect();

        let mut log_box_sizes = Vec::new();
        let mut log_fluctuations = Vec::new();

        for &box_size in &[4, 8, 16, 32, 64] {
            if box_size < values.len() / 4 {
                let num_boxes = integrated.len() / box_size;
                let mut fluctuations = Vec::new();

                for i in 0..num_boxes {
                    let start = i * box_size;
                    let end = start + box_size;
                    let box_data = &integrated[start..end];

                    // Linear detrending
                    let x_vals: Vec<f64> = (0..box_size).map(|j| j as f64).collect();
                    let n = box_size as f64;
                    let sum_x = x_vals.iter().sum::<f64>();
                    let sum_y = box_data.iter().sum::<f64>();
                    let sum_xy = x_vals.iter().zip(box_data).map(|(x, y)| x * y).sum::<f64>();
                    let sum_x2 = x_vals.iter().map(|x| x * x).sum::<f64>();

                    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
                    let intercept = (sum_y - slope * sum_x) / n;

                    let fluctuation = x_vals
                        .iter()
                        .zip(box_data)
                        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
                        .sum::<f64>()
                        / n;

                    fluctuations.push(fluctuation.sqrt());
                }

                if !fluctuations.is_empty() {
                    let avg_fluctuation =
                        fluctuations.iter().sum::<f64>() / fluctuations.len() as f64;
                    log_box_sizes.push((box_size as f64).ln());
                    log_fluctuations.push(avg_fluctuation.ln());
                }
            }
        }

        if log_box_sizes.len() < 2 {
            return Ok(1.0);
        }

        // Linear regression
        let n = log_box_sizes.len() as f64;
        let sum_x = log_box_sizes.iter().sum::<f64>();
        let sum_y = log_fluctuations.iter().sum::<f64>();
        let sum_xy = log_box_sizes
            .iter()
            .zip(&log_fluctuations)
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = log_box_sizes.iter().map(|x| x * x).sum::<f64>();

        let alpha = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        Ok(alpha)
    }
}

impl Default for FrequencyFeatures {
    fn default() -> Self {
        Self {
            dominant_frequency: 0.0,
            psd: Vec::new(),
            frequencies: Vec::new(),
            spectral_centroid: 0.0,
            spectral_bandwidth: 0.0,
            spectral_rolloff: 0.0,
            spectral_flux: 0.0,
            hnr: 0.0,
        }
    }
}

impl Default for ComplexityFeatures {
    fn default() -> Self {
        Self {
            approximate_entropy: 0.0,
            sample_entropy: 0.0,
            permutation_entropy: 0.0,
            spectral_entropy: 0.0,
            lempel_ziv_complexity: 0.0,
            fractal_dimension: 1.5,
            hurst_exponent: 0.5,
            dfa_alpha: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};

    fn create_test_series() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..100 {
            let timestamp = Utc.timestamp_opt(1640995200 + i * 86400, 0).unwrap();
            let value = 10.0 + i as f64 * 0.1 + (2.0 * PI * i as f64 / 10.0).sin() * 2.0;
            builder = builder.add_point(timestamp, value);
        }

        builder.frequency(Frequency::Daily).build().unwrap()
    }

    #[test]
    fn test_statistical_features() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new();
        let features = extractor.extract_features(&ts).unwrap();

        assert!(features.statistical.mean > 0.0);
        assert!(features.statistical.std > 0.0);
        assert!(features.statistical.variance > 0.0);
        assert!(features.statistical.min < features.statistical.max);
        assert!(features.statistical.range > 0.0);
    }

    #[test]
    fn test_window_features() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new().with_window_sizes(vec![5, 10]);
        let features = extractor.extract_features(&ts).unwrap();

        assert!(features.window.moving_averages.contains_key(&5));
        assert!(features.window.moving_averages.contains_key(&10));
        assert!(features.window.moving_stds.contains_key(&5));
        assert!(features.window.moving_stds.contains_key(&10));
    }

    #[test]
    fn test_frequency_features() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new().with_frequency_features(true);
        let features = extractor.extract_features(&ts).unwrap();

        assert!(features.frequency.dominant_frequency >= 0.0);
        assert!(!features.frequency.psd.is_empty());
        assert_eq!(
            features.frequency.psd.len(),
            features.frequency.frequencies.len()
        );
        assert!(features.frequency.spectral_centroid >= 0.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new();
        let features = extractor.extract_features(&ts).unwrap();

        let bb = &features.window.bollinger_bands;
        assert_eq!(bb.upper_band.len(), ts.len());
        assert_eq!(bb.lower_band.len(), ts.len());
        assert_eq!(bb.middle_band.len(), ts.len());
        assert!(bb.pct_within_bands >= 0.0 && bb.pct_within_bands <= 1.0);
    }

    #[test]
    fn test_complexity_features() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new().with_complexity_features(true);
        let features = extractor.extract_features(&ts).unwrap();

        assert!(features.complexity.approximate_entropy >= 0.0);
        assert!(features.complexity.sample_entropy >= 0.0);
        assert!(features.complexity.permutation_entropy >= 0.0);
        assert!(features.complexity.fractal_dimension > 0.0);
        assert!(
            features.complexity.hurst_exponent >= 0.0 && features.complexity.hurst_exponent <= 1.0
        );
    }

    #[test]
    fn test_zero_crossings() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new();
        let features = extractor.extract_features(&ts).unwrap();

        // Should have some zero crossings due to sinusoidal component
        assert!(features.statistical.zero_crossings > 0);
    }

    #[test]
    fn test_peaks_valleys() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new();
        let features = extractor.extract_features(&ts).unwrap();

        // Should detect peaks and valleys from sinusoidal component
        assert!(features.statistical.peaks > 0);
        assert!(features.statistical.valleys > 0);
    }
}
