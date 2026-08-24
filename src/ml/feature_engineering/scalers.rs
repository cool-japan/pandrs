//! Individual [`FeatureScaler`] implementations used by [`super::AutoFeatureEngineer`]:
//! `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `QuantileTransformer`, and
//! `PowerTransformer` (Yeo-Johnson).

use super::FeatureScaler;
use crate::core::error::{Error, Result};

/// Standard scaler implementation
#[derive(Debug, Clone)]
pub struct StandardScaler {
    mean: Option<f64>,
    std: Option<f64>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }
}

impl FeatureScaler for StandardScaler {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::InvalidValue("Cannot fit on empty data".into()));
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();

        self.mean = Some(mean);
        self.std = Some(if std > 1e-10 { std } else { 1.0 });

        Ok(())
    }

    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mean = self
            .mean
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;
        let std = self
            .std
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;

        Ok(data.iter().map(|&x| (x - mean) / std).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mean = self
            .mean
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;
        let std = self
            .std
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;

        Ok(data.iter().map(|&x| x * std + mean).collect())
    }
}

/// MinMax scaler implementation
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    min: Option<f64>,
    max: Option<f64>,
    feature_range: (f64, f64),
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            feature_range: (0.0, 1.0),
        }
    }

    pub fn with_range(min: f64, max: f64) -> Self {
        Self {
            min: None,
            max: None,
            feature_range: (min, max),
        }
    }
}

impl FeatureScaler for MinMaxScaler {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::InvalidValue("Cannot fit on empty data".into()));
        }

        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        self.min = Some(min);
        self.max = Some(max);

        Ok(())
    }

    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let min = self
            .min
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;
        let max = self
            .max
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;
        let (feature_min, feature_max) = self.feature_range;

        let range = max - min;
        let feature_range = feature_max - feature_min;

        if range < 1e-10 {
            Ok(vec![feature_min; data.len()])
        } else {
            Ok(data
                .iter()
                .map(|&x| feature_min + ((x - min) / range) * feature_range)
                .collect())
        }
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let min = self
            .min
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;
        let max = self
            .max
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;
        let (feature_min, feature_max) = self.feature_range;

        let range = max - min;
        let feature_range = feature_max - feature_min;

        if feature_range < 1e-10 || range < 1e-10 {
            Ok(vec![min; data.len()])
        } else {
            Ok(data
                .iter()
                .map(|&x| min + ((x - feature_min) / feature_range) * range)
                .collect())
        }
    }
}

/// Robust scaler using median and interquartile range (IQR)
///
/// Centers data around the median and scales by the IQR, making it robust
/// to outliers. Transform: `(x - median) / IQR`.
#[derive(Debug, Clone)]
pub struct RobustScaler {
    median: Option<f64>,
    iqr: Option<f64>,
}

impl RobustScaler {
    pub fn new() -> Self {
        RobustScaler {
            median: None,
            iqr: None,
        }
    }
}

impl FeatureScaler for RobustScaler {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::InvalidValue("Cannot fit on empty data".into()));
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();

        // Compute median
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };

        // Compute Q1 (25th percentile) and Q3 (75th percentile)
        // Using linear interpolation for accurate quantile computation
        let q1 = {
            let pos = 0.25 * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let frac = pos - lo as f64;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        };
        let q3 = {
            let pos = 0.75 * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let frac = pos - lo as f64;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        };
        // sklearn's RobustScaler special-cases a degenerate (zero) IQR to a
        // unit scale rather than an epsilon floor: `(x - median) / 1e-10`
        // would blow tiny floating-point noise up to ~1e10, whereas
        // `(x - median) / 1.0` just centers the (constant) data at zero.
        let iqr_raw = q3 - q1;
        let iqr = if iqr_raw > 1e-10 { iqr_raw } else { 1.0 };

        self.median = Some(median);
        self.iqr = Some(iqr);
        Ok(())
    }

    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let median = self
            .median
            .ok_or_else(|| Error::InvalidOperation("RobustScaler not fitted".into()))?;
        let iqr = self
            .iqr
            .ok_or_else(|| Error::InvalidOperation("RobustScaler not fitted".into()))?;
        Ok(data.iter().map(|&x| (x - median) / iqr).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let median = self
            .median
            .ok_or_else(|| Error::InvalidOperation("RobustScaler not fitted".into()))?;
        let iqr = self
            .iqr
            .ok_or_else(|| Error::InvalidOperation("RobustScaler not fitted".into()))?;
        Ok(data.iter().map(|&x| x * iqr + median).collect())
    }
}

/// Quantile transformer that maps feature values to a uniform `[0,1]` distribution
///
/// Uses rank-based normalization: each value is mapped to its empirical quantile
/// in the training data using linear interpolation between stored training values.
#[derive(Debug, Clone)]
pub struct QuantileTransformer {
    /// Sorted training values used as reference quantiles
    reference_values: Option<Vec<f64>>,
}

impl QuantileTransformer {
    pub fn new() -> Self {
        QuantileTransformer {
            reference_values: None,
        }
    }

    /// Linearly interpolate the quantile rank of `x` relative to sorted `reference`
    fn interpolate_quantile(x: f64, reference: &[f64]) -> f64 {
        let n = reference.len();
        if n == 0 {
            return 0.5;
        }
        if x <= reference[0] {
            return 0.0;
        }
        if x >= reference[n - 1] {
            return 1.0;
        }
        // Binary search for insertion point
        let pos = reference.partition_point(|&v| v <= x);
        // pos is the index where x would be inserted: reference[pos-1] <= x < reference[pos]
        let lo = pos.saturating_sub(1);
        let hi = pos.min(n - 1);
        if reference[hi] == reference[lo] {
            return lo as f64 / (n - 1) as f64;
        }
        let t = (x - reference[lo]) / (reference[hi] - reference[lo]);
        let q_lo = lo as f64 / (n - 1) as f64;
        let q_hi = hi as f64 / (n - 1) as f64;
        q_lo + t * (q_hi - q_lo)
    }

    /// Invert a quantile value back to a data value using reference distribution
    fn inverse_quantile(q: f64, reference: &[f64]) -> f64 {
        let n = reference.len();
        if n == 0 {
            return 0.0;
        }
        let q = q.clamp(0.0, 1.0);
        let pos = q * (n - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = (pos.ceil() as usize).min(n - 1);
        let frac = pos - lo as f64;
        reference[lo] + frac * (reference[hi] - reference[lo])
    }
}

impl FeatureScaler for QuantileTransformer {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::InvalidValue("Cannot fit on empty data".into()));
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.reference_values = Some(sorted);
        Ok(())
    }

    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let reference = self
            .reference_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("QuantileTransformer not fitted".into()))?;
        Ok(data
            .iter()
            .map(|&x| Self::interpolate_quantile(x, reference))
            .collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let reference = self
            .reference_values
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("QuantileTransformer not fitted".into()))?;
        Ok(data
            .iter()
            .map(|&q| Self::inverse_quantile(q, reference))
            .collect())
    }
}

/// Power transformer using the Yeo-Johnson transformation
///
/// Applies the Yeo-Johnson power transform to make the distribution more
/// Gaussian-like. The optimal lambda parameter is chosen by scanning a grid
/// of lambda values and selecting the one that minimizes |skewness|.
///
/// Yeo-Johnson transform for lambda != 0, x >= 0: ((x + 1)^lambda - 1) / lambda
/// Yeo-Johnson transform for lambda == 0, x >= 0: ln(x + 1)
/// Yeo-Johnson transform for lambda != 2, x < 0:  -((-x + 1)^(2-lambda) - 1) / (2 - lambda)
/// Yeo-Johnson transform for lambda == 2, x < 0:  -ln(-x + 1)
#[derive(Debug, Clone)]
pub struct PowerTransformer {
    lambda: Option<f64>,
}

impl PowerTransformer {
    pub fn new() -> Self {
        PowerTransformer { lambda: None }
    }

    /// Apply the Yeo-Johnson transform to a single value given lambda
    fn yeo_johnson(x: f64, lambda: f64) -> f64 {
        if x >= 0.0 {
            if (lambda - 0.0).abs() < 1e-10 {
                (x + 1.0_f64).ln()
            } else {
                ((x + 1.0_f64).powf(lambda) - 1.0) / lambda
            }
        } else if (lambda - 2.0).abs() < 1e-10 {
            -(-x + 1.0_f64).ln()
        } else {
            -((-x + 1.0_f64).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
        }
    }

    /// Compute skewness of a vector (used to pick optimal lambda)
    fn skewness(vals: &[f64]) -> f64 {
        let n = vals.len() as f64;
        if n < 3.0 {
            return 0.0;
        }
        let mean = vals.iter().sum::<f64>() / n;
        let m2 = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        let m3 = vals.iter().map(|&v| (v - mean).powi(3)).sum::<f64>() / n;
        let std = m2.sqrt();
        if std < 1e-10 {
            0.0
        } else {
            m3 / std.powi(3)
        }
    }

    /// Inverse Yeo-Johnson: recover x from transformed value y given lambda.
    ///
    /// The forward transform's image is bounded whenever `lambda < 0` (for
    /// `y >= 0`) or `lambda > 2` (for `y < 0`): e.g. for `x >= 0, lambda < 0`,
    /// `y = ((x+1)^lambda - 1)/lambda` approaches `-1/lambda` as `x -> inf`
    /// but never reaches it, so no real `x` maps to a `y` at or beyond that
    /// bound. Attempting the inverse there requires raising a non-positive
    /// base to a non-integer power, which is undefined; rather than let that
    /// silently produce `NaN`/`inf`, report it as an out-of-domain input.
    fn yeo_johnson_inverse(y: f64, lambda: f64) -> Result<f64> {
        if y >= 0.0 {
            if (lambda - 0.0).abs() < 1e-10 {
                Ok(y.exp() - 1.0)
            } else {
                let base = y * lambda + 1.0;
                if base <= 0.0 {
                    return Err(Error::InvalidValue(format!(
                        "yeo_johnson_inverse: y={} is outside the invertible range for \
                         lambda={} (y*lambda + 1 = {} must be positive)",
                        y, lambda, base
                    )));
                }
                Ok(base.powf(1.0 / lambda) - 1.0)
            }
        } else if (lambda - 2.0).abs() < 1e-10 {
            Ok(1.0 - (-y).exp())
        } else {
            let two_minus_lambda = 2.0 - lambda;
            let base = -y * two_minus_lambda + 1.0;
            if base <= 0.0 {
                return Err(Error::InvalidValue(format!(
                    "yeo_johnson_inverse: y={} is outside the invertible range for lambda={} \
                     (-y*(2-lambda) + 1 = {} must be positive)",
                    y, lambda, base
                )));
            }
            Ok(1.0 - base.powf(1.0 / two_minus_lambda))
        }
    }
}

impl FeatureScaler for PowerTransformer {
    fn fit(&mut self, data: &[f64]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::InvalidValue("Cannot fit on empty data".into()));
        }
        // Grid search over lambda in [-2, 2] with step 0.25 to minimize |skewness|
        let candidates: Vec<f64> = (-8..=8).map(|i| i as f64 * 0.25).collect();
        let mut best_lambda = 0.0_f64;
        let mut best_skew_abs = f64::INFINITY;

        for &lambda in &candidates {
            let transformed: Vec<f64> =
                data.iter().map(|&x| Self::yeo_johnson(x, lambda)).collect();
            // Skip if any non-finite values (happens with extreme lambdas)
            if transformed.iter().any(|v| !v.is_finite()) {
                continue;
            }
            let skew = Self::skewness(&transformed).abs();
            if skew < best_skew_abs {
                best_skew_abs = skew;
                best_lambda = lambda;
            }
        }

        self.lambda = Some(best_lambda);
        Ok(())
    }

    fn transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let lambda = self
            .lambda
            .ok_or_else(|| Error::InvalidOperation("PowerTransformer not fitted".into()))?;
        Ok(data.iter().map(|&x| Self::yeo_johnson(x, lambda)).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let lambda = self
            .lambda
            .ok_or_else(|| Error::InvalidOperation("PowerTransformer not fitted".into()))?;
        data.iter()
            .map(|&y| Self::yeo_johnson_inverse(y, lambda))
            .collect()
    }
}
