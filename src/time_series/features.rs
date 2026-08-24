//! Time Series Feature Extraction Module
//!
//! This module provides comprehensive feature extraction capabilities for time series data,
//! including statistical features, window-based features, and domain-specific features
//! for machine learning applications.

use crate::core::error::{Error, Result};
use crate::time_series::core::TimeSeries;
use crate::time_series::spectral::{periodogram, Detrend, Periodogram};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
        sorted.sort_by(|a, b| a.total_cmp(b));

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

    /// Extract frequency domain features from the real periodogram.
    ///
    /// The power spectral density comes from [`crate::time_series::spectral::periodogram`],
    /// a genuine OxiFFT-backed DFT of the mean-removed series (previously an
    /// `O(n²)` sum of at most 50 autocorrelation lags against `cos(2π f l)`,
    /// whose frequency resolution was fixed at `1/50` regardless of series
    /// length and whose peak therefore did not track the true dominant
    /// oscillation). Frequencies are in cycles per sample, `k / n` for
    /// `k = 0..=n/2`.
    ///
    /// `dominant_frequency` skips the DC bin: after mean removal DC carries no
    /// information, and reporting `f = 0` as "the dominant frequency" would be
    /// meaningless.
    fn extract_frequency_features(&self, values: &[f64]) -> Result<FrequencyFeatures> {
        let spectrum = periodogram(values, Detrend::Mean)?;
        let Periodogram { frequencies, psd } = spectrum.clone();

        // Dominant frequency: the strongest non-DC bin. `None` only when the
        // series is too short to have one, which `periodogram` already
        // rejects, so this is a defensive NaN rather than a fabricated 0.0.
        let dominant_frequency = spectrum
            .dominant_bin()
            .map(|idx| frequencies[idx])
            .unwrap_or(f64::NAN);

        let total_power = spectrum.total_power();

        // Spectral centroid: the power-weighted mean frequency. Undefined
        // (NaN) for a series with no spectral power at all, i.e. a constant.
        let spectral_centroid = if total_power > 0.0 {
            frequencies
                .iter()
                .zip(&psd)
                .map(|(freq, power)| freq * power)
                .sum::<f64>()
                / total_power
        } else {
            f64::NAN
        };

        // Spectral bandwidth: the power-weighted spread about the centroid.
        let spectral_bandwidth = if total_power > 0.0 {
            (frequencies
                .iter()
                .zip(&psd)
                .map(|(freq, power)| (freq - spectral_centroid).powi(2) * power)
                .sum::<f64>()
                / total_power)
                .sqrt()
        } else {
            f64::NAN
        };

        // Spectral rolloff: the frequency below which 85% of the power lies.
        let spectral_rolloff = if total_power > 0.0 {
            let rolloff_threshold = 0.85 * total_power;
            let mut cumulative_power = 0.0;
            frequencies
                .iter()
                .zip(&psd)
                .find(|(_, power)| {
                    cumulative_power += *power;
                    cumulative_power >= rolloff_threshold
                })
                .map(|(freq, _)| *freq)
                // The cumulative sum reaches the threshold by construction
                // unless rounding leaves it a hair short; fall back to Nyquist.
                .unwrap_or_else(|| frequencies.last().copied().unwrap_or(f64::NAN))
        } else {
            f64::NAN
        };

        // Spectral flux across neighbouring bins (spectral roughness): the
        // mean absolute bin-to-bin change of the PSD.
        let spectral_flux = if psd.len() > 1 {
            psd.windows(2)
                .map(|window| (window[1] - window[0]).abs())
                .sum::<f64>()
                / (psd.len() - 1) as f64
        } else {
            f64::NAN
        };

        // Harmonic-to-noise ratio: power in the dominant frequency and its
        // integer harmonics (each bin plus its immediate neighbours, to absorb
        // spectral leakage) against everything else. The previous version
        // divided the whole spectrum by the DC bin, which after mean removal
        // is ~0 and carries no harmonic meaning at all.
        let hnr = match spectrum.dominant_bin() {
            Some(base) if base > 0 && total_power > 0.0 => {
                let mut harmonic_power = 0.0;
                let mut counted = vec![false; psd.len()];
                let mut harmonic = base;
                while harmonic < psd.len() {
                    let lo = harmonic.saturating_sub(1);
                    let hi = (harmonic + 1).min(psd.len() - 1);
                    for (bin, seen) in counted.iter_mut().enumerate().take(hi + 1).skip(lo) {
                        if !*seen {
                            *seen = true;
                            harmonic_power += psd[bin];
                        }
                    }
                    harmonic += base;
                }
                let noise_power = total_power - harmonic_power;
                if noise_power > 0.0 {
                    10.0 * (harmonic_power / noise_power).log10()
                } else {
                    // A pure tone: no residual noise floor to compare against.
                    f64::INFINITY
                }
            }
            _ => f64::NAN,
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

    /// Extract complexity features.
    ///
    /// Approximate and sample entropy are computed at the conventional
    /// tolerance `r = 0.2 · σ` (Pincus 1991; Richman & Moorman 2000), where σ
    /// is the sample standard deviation of the series. The call sites
    /// previously passed the *relative* factor `0.2` straight through as the
    /// absolute Chebyshev radius, so both statistics were scale-dependent
    /// artefacts: a series in millivolts and the same series in volts got
    /// wildly different "complexity".
    fn extract_complexity_features(&self, values: &[f64]) -> Result<ComplexityFeatures> {
        let tolerance = Self::entropy_tolerance(values, 0.2);

        // Approximate entropy
        let approximate_entropy = self.approximate_entropy(values, 2, tolerance)?;

        // Sample entropy
        let sample_entropy = self.sample_entropy(values, 2, tolerance)?;

        // Permutation entropy
        let permutation_entropy = self.permutation_entropy(values, 3)?;

        // Spectral entropy
        let spectral_entropy = self.spectral_entropy(values)?;

        // Lempel-Ziv complexity
        let lempel_ziv_complexity = self.lempel_ziv_complexity(values)?;

        // Fractal dimension (Higuchi)
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

    /// Chebyshev matching radius for the regularity entropies: `factor · σ`
    /// with σ the sample (`ddof = 1`) standard deviation.
    ///
    /// Returns `NaN` for a series that has no spread (fewer than two points,
    /// or a constant), which propagates through [`Self::approximate_entropy`]
    /// and [`Self::sample_entropy`] as "not estimable" instead of silently
    /// collapsing every comparison onto an all-match / no-match degenerate.
    fn entropy_tolerance(values: &[f64], factor: f64) -> f64 {
        let n = values.len();
        if n < 2 {
            return f64::NAN;
        }
        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev > 0.0 {
            factor * std_dev
        } else {
            f64::NAN
        }
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

                // %B indicator. A zero-width band (a perfectly flat window)
                // leaves %B undefined — reporting the band midpoint `0.5`
                // would claim the price sits exactly between two coincident
                // bands, so this is NaN.
                if upper != lower {
                    percent_b.push((values[i] - lower) / (upper - lower));
                } else {
                    percent_b.push(f64::NAN);
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
        let _sum_y2 = window_values.iter().map(|y| y * y).sum::<f64>();

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

    /// Calculate the approximate entropy `ApEn(m, r)` (Pincus, 1991).
    ///
    /// `r` is the **absolute** Chebyshev matching radius; callers that want the
    /// conventional relative tolerance should pass
    /// `Self::entropy_tolerance(values, 0.2)` (i.e. `0.2 · σ`).
    ///
    /// Returns `NaN` when the series is shorter than `m + 1` or `r` is not a
    /// usable radius (non-finite or non-positive — e.g. a constant series has
    /// no scale to normalize against), because ApEn is genuinely undefined
    /// there; the previous `Ok(0.0)` reported "perfectly regular" instead.
    fn approximate_entropy(&self, values: &[f64], m: usize, r: f64) -> Result<f64> {
        if values.len() < m + 1 || !r.is_finite() || r <= 0.0 {
            return Ok(f64::NAN);
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
            Ok(f64::NAN)
        }
    }

    /// Calculate the sample entropy `SampEn(m, r)` (Richman & Moorman, 2000).
    ///
    /// `r` is the **absolute** Chebyshev matching radius (see
    /// [`Self::approximate_entropy`] and [`Self::entropy_tolerance`]).
    ///
    /// Returns `NaN` when the series is shorter than `m + 1`, `r` is unusable,
    /// or there are no length-`m` template matches at all (`B = 0`, so the
    /// conditional probability `A/B` has no denominator). Returns `+∞` when
    /// there are `m`-matches but no `(m+1)`-matches (`A = 0`): that is the
    /// genuine value of `−ln(A/B)`, and the standard "no estimate available at
    /// this series length" signal, not a regularity of `0.0`.
    fn sample_entropy(&self, values: &[f64], m: usize, r: f64) -> Result<f64> {
        if values.len() < m + 1 || !r.is_finite() || r <= 0.0 {
            return Ok(f64::NAN);
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
            Ok(f64::NAN)
        } else {
            Ok(-(a / b).ln())
        }
    }

    /// Calculate the Bandt-Pompe permutation entropy of embedding dimension
    /// `order` (Bandt & Pompe, 2002), in nats.
    ///
    /// Each length-`order` window is mapped to its **ordinal pattern**: the
    /// argsort permutation `π` such that `x[i+π₀] ≤ x[i+π₁] ≤ … ≤ x[i+π_{m-1}]`.
    /// The entropy is the Shannon entropy of the empirical distribution over
    /// the `order!` possible patterns.
    ///
    /// The previous implementation computed the argsort but then discarded it
    /// (`indices.into_iter().enumerate().map(|(rank, _)| rank)` rebuilds
    /// `0, 1, …, order-1` no matter what the data was), so every window mapped
    /// to the same key, the distribution was always a single atom, and the
    /// function returned exactly `0.0` for every input.
    ///
    /// # Errors
    /// Returns [`Error::InvalidInput`] when `order < 2` (no ordering exists)
    /// or the series is shorter than `order`.
    fn permutation_entropy(&self, values: &[f64], order: usize) -> Result<f64> {
        if order < 2 {
            return Err(Error::InvalidInput(format!(
                "permutation entropy needs an embedding order of at least 2, got {order}"
            )));
        }
        if values.len() < order {
            return Err(Error::InvalidInput(format!(
                "permutation entropy of order {order} needs at least {order} observations, got {}",
                values.len()
            )));
        }

        let mut permutation_counts: HashMap<Vec<usize>, usize> = HashMap::new();
        let total_patterns = values.len() - order + 1;

        for i in 0..total_patterns {
            let pattern = &values[i..i + order];
            let mut permutation: Vec<usize> = (0..order).collect();
            // Ties break on position, which keeps the pattern well defined and
            // deterministic (the Bandt-Pompe convention for equal values).
            permutation.sort_by(|&a, &b| pattern[a].total_cmp(&pattern[b]).then(a.cmp(&b)));

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

    /// Calculate the spectral entropy: the Shannon entropy (in nats) of the
    /// power spectral density normalized to a probability distribution.
    ///
    /// Built on the same OxiFFT periodogram as the frequency features, rather
    /// than on the truncated 20-lag autocorrelation cosine sum this replaced.
    fn spectral_entropy(&self, values: &[f64]) -> Result<f64> {
        let spectrum = periodogram(values, Detrend::Mean)?;
        let total_power = spectrum.total_power();
        if !(total_power > 0.0) {
            // A constant series has no spectral content; its normalized
            // spectrum is undefined rather than "zero entropy".
            return Ok(f64::NAN);
        }

        let entropy = spectrum
            .psd
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| {
                let normalized = p / total_power;
                -normalized * normalized.ln()
            })
            .sum::<f64>();

        Ok(entropy)
    }

    /// Calculate the normalized Lempel-Ziv complexity (LZ76).
    ///
    /// The series is binarized against its **median** (a robust threshold), then
    /// parsed with the Kaspar-Schuster (1987) realization of the Lempel-Ziv
    /// (1976) production complexity `c(n)`. The result is normalized by the
    /// asymptotic upper bound for a `b`-symbol alphabet, `b(n) = n / log_b(n)`
    /// (here `b = 2`), so a random binary sequence tends to `1.0`:
    ///     `C = c(n) * log_2(n) / n`.
    fn lempel_ziv_complexity(&self, values: &[f64]) -> Result<f64> {
        let n = values.len();
        if n < 2 {
            return Ok(0.0);
        }

        // Robust binarization against the true median (not the mean).
        let median = {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| a.total_cmp(b));
            if sorted.len() % 2 == 0 {
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
            } else {
                sorted[sorted.len() / 2]
            }
        };
        let s: Vec<u8> = values.iter().map(|&x| u8::from(x >= median)).collect();

        // LZ76 production-complexity count (Kaspar & Schuster, 1987).
        let mut c = 1usize; // number of distinct productions
        let mut u = 1usize; // length of the already-reconstructed prefix
        let mut v = 1usize; // current candidate substring length
        let mut v_max = 1usize; // longest reproducible substring at this prefix
        let mut i = 0usize; // pointer scanning the prefix
        while u + v <= n {
            if s[i + v - 1] == s[u + v - 1] {
                v += 1;
            } else {
                if v > v_max {
                    v_max = v;
                }
                i += 1;
                if i == u {
                    // No prefix substring reproduces the next block: new production.
                    c += 1;
                    u += v_max;
                    i = 0;
                    v = 1;
                    v_max = 1;
                } else {
                    v = 1;
                }
            }
        }
        if v != 1 {
            c += 1;
        }

        // Normalize by the b(n) = n / log_2(n) upper bound (binary alphabet).
        let normalized = c as f64 * (n as f64).log2() / n as f64;
        Ok(normalized)
    }

    /// Estimate the fractal dimension with Higuchi's method (HFD, 1988).
    ///
    /// For each scale `k` the mean curve length `L(k)` of the `k` interleaved
    /// sub-series is computed; `L(k) ∝ k^{-D}`, so the dimension `D` is the slope
    /// of `ln L(k)` against `ln(1/k)`. This operates on the temporal structure of
    /// the signal (unlike amplitude-only box counting) and returns a value in the
    /// usual `[1, 2]` range for a 1-D time series.
    ///
    /// Returns `NaN` when fewer than four observations are available or fewer
    /// than two scales produce a usable curve length, since the log-log slope
    /// is then undefined; a **constant** series is the one honest special case
    /// (`L(k) ≡ 0` at every scale) and is reported as the mathematically
    /// correct `D = 1.0` for a flat line.
    fn fractal_dimension(&self, values: &[f64]) -> Result<f64> {
        let n = values.len();
        if n < 4 {
            return Ok(f64::NAN);
        }
        let is_constant = values.windows(2).all(|w| w[0] == w[1]);
        if is_constant {
            return Ok(1.0);
        }

        // Scales 1..=k_max; cap so every sub-series has at least one increment.
        let k_max = std::cmp::max(2, std::cmp::min(10, n / 4));

        let mut ln_inv_k = Vec::new();
        let mut ln_len = Vec::new();

        for k in 1..=k_max {
            let mut lk_sum = 0.0;
            let mut m_count = 0usize;

            // m is 1-based per Higuchi; arrays are 0-based (X(j) == values[j-1]).
            for m in 1..=k {
                let num_steps = (n - m) / k; // floor((N - m) / k)
                if num_steps < 1 {
                    continue;
                }

                let mut length = 0.0;
                for step in 1..=num_steps {
                    let idx_curr = m + step * k - 1;
                    let idx_prev = m + (step - 1) * k - 1;
                    length += (values[idx_curr] - values[idx_prev]).abs();
                }

                // Normalization factor (N-1) / (num_steps * k), then the 1/k mean.
                let norm = (n - 1) as f64 / (num_steps as f64 * k as f64);
                lk_sum += length * norm / k as f64;
                m_count += 1;
            }

            if m_count > 0 {
                let lk = lk_sum / m_count as f64;
                if lk > 0.0 {
                    ln_inv_k.push((1.0 / k as f64).ln());
                    ln_len.push(lk.ln());
                }
            }
        }

        if ln_inv_k.len() < 2 {
            return Ok(f64::NAN);
        }

        // L(k) ∝ k^{-D} ⇒ ln L = D·ln(1/k) + const, so the slope is D directly.
        let count = ln_inv_k.len() as f64;
        let sum_x = ln_inv_k.iter().sum::<f64>();
        let sum_y = ln_len.iter().sum::<f64>();
        let sum_xy = ln_inv_k
            .iter()
            .zip(&ln_len)
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = ln_inv_k.iter().map(|x| x * x).sum::<f64>();

        let denom = count * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return Ok(f64::NAN);
        }
        let slope = (count * sum_xy - sum_x * sum_y) / denom;

        Ok(slope)
    }

    /// Estimate the Hurst exponent by rescaled-range (R/S) analysis
    /// (Hurst 1951; Mandelbrot & Wallis 1969).
    ///
    /// For every scale `s` the series is cut into `⌊N/s⌋` **non-overlapping**
    /// windows; within each window the cumulative deviation from that window's
    /// own mean gives the range `R`, which is divided by the window's own
    /// standard deviation `S`. The rescaled ranges are averaged per scale and
    /// `ln E[R/S]` is regressed on `ln s`; the slope is `H`.
    ///
    /// The previous implementation used a hardcoded scale ladder
    /// `[10, 20, 50, 100]`, evaluated only the **first** window at each scale,
    /// and rescaled by the *global* mean and standard deviation instead of the
    /// window's — so it measured how far the prefix drifted from the global
    /// mean rather than a rescaled range, and produced no estimate at all for
    /// series shorter than 20 points (falling back to a literal `0.5`).
    ///
    /// Returns `NaN` when fewer than two scales yield a usable `R/S` (a series
    /// under 16 points, or one that is constant inside every window). The
    /// estimate is clamped to `[0, 1]`, the range in which `H` is defined.
    fn hurst_exponent(&self, values: &[f64]) -> Result<f64> {
        let len = values.len();
        if len < 16 {
            return Ok(f64::NAN);
        }

        let mut log_rs = Vec::new();
        let mut log_scale = Vec::new();

        // Geometric scale ladder from 8 up to N/2, so that every scale has at
        // least two windows to average over.
        let mut scale = 8usize;
        while scale <= len / 2 {
            let n_windows = len / scale;
            let mut rs_values = Vec::with_capacity(n_windows);

            for w in 0..n_windows {
                let window = &values[w * scale..(w + 1) * scale];
                let mean = window.iter().sum::<f64>() / scale as f64;

                let mut cumulative = 0.0;
                let mut max_dev = f64::NEG_INFINITY;
                let mut min_dev = f64::INFINITY;
                for &x in window {
                    cumulative += x - mean;
                    max_dev = max_dev.max(cumulative);
                    min_dev = min_dev.min(cumulative);
                }
                let range = max_dev - min_dev;

                let std_dev =
                    (window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / scale as f64).sqrt();

                if std_dev > 0.0 && range > 0.0 {
                    rs_values.push(range / std_dev);
                }
            }

            if !rs_values.is_empty() {
                let mean_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
                log_rs.push(mean_rs.ln());
                log_scale.push((scale as f64).ln());
            }

            // Roughly √2 spacing keeps the ladder dense without repeating scales.
            let next = (scale as f64 * std::f64::consts::SQRT_2).round() as usize;
            scale = next.max(scale + 1);
        }

        if log_scale.len() < 2 {
            return Ok(f64::NAN);
        }

        // Linear regression of ln E[R/S] on ln s.
        let n = log_scale.len() as f64;
        let sum_x = log_scale.iter().sum::<f64>();
        let sum_y = log_rs.iter().sum::<f64>();
        let sum_xy = log_scale
            .iter()
            .zip(&log_rs)
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = log_scale.iter().map(|x| x * x).sum::<f64>();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return Ok(f64::NAN);
        }
        let hurst = (n * sum_xy - sum_x * sum_y) / denom;

        Ok(hurst.clamp(0.0, 1.0))
    }

    /// Detrended fluctuation analysis (Peng et al., 1994): the scaling
    /// exponent `α` of the detrended fluctuation `F(s) ∝ s^α`.
    ///
    /// The series is integrated (cumulative sum of deviations from the mean),
    /// cut into boxes of size `s` **from both ends** so the tail left over by
    /// `N mod s` still contributes, linearly detrended inside each box, and
    /// `F(s)` is the root-mean-square residual pooled over all boxes. `ln F(s)`
    /// is then regressed on `ln s`.
    ///
    /// Scales run over a geometric ladder from 4 to `N/4` derived from the
    /// series length, rather than the fixed `[4, 8, 16, 32, 64]` ladder this
    /// replaced (which silently produced a single usable scale — and therefore
    /// the hardcoded `1.0` — for any series shorter than 33 points, and never
    /// looked past `s = 64` however long the series was).
    ///
    /// Returns `NaN` when fewer than two scales are usable.
    fn detrended_fluctuation_analysis(&self, values: &[f64]) -> Result<f64> {
        let len = values.len();
        if len < 16 {
            return Ok(f64::NAN);
        }

        // Create integrated series (the "profile")
        let mean = values.iter().sum::<f64>() / len as f64;
        let integrated: Vec<f64> = values
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x - mean;
                Some(*acc)
            })
            .collect();

        // Root-mean-square residual of a least-squares line fitted to one box.
        let box_residual_ms = |box_data: &[f64]| -> Option<f64> {
            let n = box_data.len() as f64;
            let sum_x = (0..box_data.len()).map(|j| j as f64).sum::<f64>();
            let sum_y = box_data.iter().sum::<f64>();
            let sum_xy = box_data
                .iter()
                .enumerate()
                .map(|(j, y)| j as f64 * y)
                .sum::<f64>();
            let sum_x2 = (0..box_data.len()).map(|j| (j * j) as f64).sum::<f64>();

            let denom = n * sum_x2 - sum_x * sum_x;
            if denom.abs() < 1e-12 {
                return None;
            }
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            let intercept = (sum_y - slope * sum_x) / n;

            Some(
                box_data
                    .iter()
                    .enumerate()
                    .map(|(j, y)| (y - (slope * j as f64 + intercept)).powi(2))
                    .sum::<f64>()
                    / n,
            )
        };

        let mut log_box_sizes = Vec::new();
        let mut log_fluctuations = Vec::new();

        let max_box = len / 4;
        let mut box_size = 4usize;
        while box_size <= max_box {
            let num_boxes = len / box_size;
            let mut mean_squares = Vec::with_capacity(2 * num_boxes);

            // Forward pass, then a backward pass offset by the remainder so the
            // trailing `len % box_size` samples are not discarded.
            let offset = len - num_boxes * box_size;
            for i in 0..num_boxes {
                let start = i * box_size;
                if let Some(ms) = box_residual_ms(&integrated[start..start + box_size]) {
                    mean_squares.push(ms);
                }
                if offset > 0 {
                    let start = offset + i * box_size;
                    if let Some(ms) = box_residual_ms(&integrated[start..start + box_size]) {
                        mean_squares.push(ms);
                    }
                }
            }

            if !mean_squares.is_empty() {
                let f_s = (mean_squares.iter().sum::<f64>() / mean_squares.len() as f64).sqrt();
                if f_s > 0.0 {
                    log_box_sizes.push((box_size as f64).ln());
                    log_fluctuations.push(f_s.ln());
                }
            }

            let next = (box_size as f64 * std::f64::consts::SQRT_2).round() as usize;
            box_size = next.max(box_size + 1);
        }

        if log_box_sizes.len() < 2 {
            return Ok(f64::NAN);
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

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return Ok(f64::NAN);
        }
        let alpha = (n * sum_xy - sum_x * sum_y) / denom;

        Ok(alpha)
    }
}

impl Default for FrequencyFeatures {
    /// An *empty* feature set: no spectrum has been computed, so every scalar
    /// is `NaN` ("not measured") rather than a plausible-looking `0.0`.
    fn default() -> Self {
        Self {
            dominant_frequency: f64::NAN,
            psd: Vec::new(),
            frequencies: Vec::new(),
            spectral_centroid: f64::NAN,
            spectral_bandwidth: f64::NAN,
            spectral_rolloff: f64::NAN,
            spectral_flux: f64::NAN,
            hnr: f64::NAN,
        }
    }
}

impl Default for ComplexityFeatures {
    /// An *empty* feature set: nothing has been estimated yet, so every field
    /// is `NaN`. The previous defaults (`fractal_dimension: 1.5`,
    /// `hurst_exponent: 0.5`, `dfa_alpha: 1.0`) were indistinguishable from a
    /// real measurement of a fractional Brownian motion.
    fn default() -> Self {
        Self {
            approximate_entropy: f64::NAN,
            sample_entropy: f64::NAN,
            permutation_entropy: f64::NAN,
            spectral_entropy: f64::NAN,
            lempel_ziv_complexity: f64::NAN,
            fractal_dimension: f64::NAN,
            hurst_exponent: f64::NAN,
            dfa_alpha: f64::NAN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::core::{Frequency, TimeSeriesBuilder};
    use chrono::{TimeZone, Utc};
    use std::f64::consts::PI;

    fn create_test_series() -> TimeSeries {
        let mut builder = TimeSeriesBuilder::new();

        for i in 0..100 {
            let timestamp = Utc
                .timestamp_opt(1640995200 + i * 86400, 0)
                .single()
                .expect("operation should succeed");
            let value = 10.0 + i as f64 * 0.1 + (2.0 * PI * i as f64 / 10.0).sin() * 2.0;
            builder = builder.add_point(timestamp, value);
        }

        builder
            .frequency(Frequency::Daily)
            .build()
            .expect("operation should succeed")
    }

    #[test]
    fn test_statistical_features() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new();
        let features = extractor
            .extract_features(&ts)
            .expect("operation should succeed");

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
        let features = extractor
            .extract_features(&ts)
            .expect("operation should succeed");

        assert!(features.window.moving_averages.contains_key(&5));
        assert!(features.window.moving_averages.contains_key(&10));
        assert!(features.window.moving_stds.contains_key(&5));
        assert!(features.window.moving_stds.contains_key(&10));
    }

    #[test]
    fn test_frequency_features() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new().with_frequency_features(true);
        let features = extractor
            .extract_features(&ts)
            .expect("operation should succeed");

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
        let features = extractor
            .extract_features(&ts)
            .expect("operation should succeed");

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
        let features = extractor
            .extract_features(&ts)
            .expect("operation should succeed");

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
        let features = extractor
            .extract_features(&ts)
            .expect("operation should succeed");

        // Should have some zero crossings due to sinusoidal component
        assert!(features.statistical.zero_crossings > 0);
    }

    #[test]
    fn test_peaks_valleys() {
        let ts = create_test_series();
        let extractor = TimeSeriesFeatureExtractor::new();
        let features = extractor
            .extract_features(&ts)
            .expect("operation should succeed");

        // Should detect peaks and valleys from sinusoidal component
        assert!(features.statistical.peaks > 0);
        assert!(features.statistical.valleys > 0);
    }

    #[test]
    fn test_higuchi_fractal_dimension_of_line() {
        // A perfectly linear ramp is a smooth 1-D curve: Higuchi's L(k) ∝ 1/k,
        // so the fractal dimension must be ≈ 1.0.
        let extractor = TimeSeriesFeatureExtractor::new();
        let line: Vec<f64> = (0..64).map(|i| 2.0 * i as f64 + 1.0).collect();
        let fd = extractor
            .fractal_dimension(&line)
            .expect("operation should succeed");
        assert!(
            (fd - 1.0).abs() < 0.05,
            "FD of a line should be ~1.0, got {fd}"
        );
    }

    #[test]
    fn test_lempel_ziv_orders_constant_below_alternating() {
        // A constant sequence has minimal LZ complexity; an alternating
        // sequence is far less compressible, so it must score higher.
        let extractor = TimeSeriesFeatureExtractor::new();
        let constant = vec![5.0_f64; 40];
        let alternating: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { 4.0 } else { 6.0 })
            .collect();

        let lz_const = extractor
            .lempel_ziv_complexity(&constant)
            .expect("operation should succeed");
        let lz_alt = extractor
            .lempel_ziv_complexity(&alternating)
            .expect("operation should succeed");

        assert!(lz_const >= 0.0);
        assert!(
            lz_alt > lz_const,
            "alternating LZ ({lz_alt}) should exceed constant LZ ({lz_const})"
        );
    }
}
