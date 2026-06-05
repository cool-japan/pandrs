//! Automated feature engineering and selection
//!
//! This module provides comprehensive feature engineering capabilities including
//! automated feature generation, selection, and transformation pipelines.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::model_selection::{ScoreFunction, SelectKBest};
use crate::ml::models::linear::LinearRegression;
use crate::ml::models::SupervisedModel;
use crate::ml::sklearn_compat::{SklearnEstimator, SklearnTransformer};
use crate::series::Series;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Automated feature engineering pipeline
#[derive(Debug)]
pub struct AutoFeatureEngineer {
    /// Whether to generate polynomial features
    pub generate_polynomial: bool,
    /// Maximum degree for polynomial features
    pub poly_degree: usize,
    /// Whether to generate interaction features
    pub generate_interactions: bool,
    /// Maximum number of features to interact
    pub max_interaction_features: usize,
    /// Whether to generate aggregation features
    pub generate_aggregations: bool,
    /// Aggregation functions to use
    pub aggregation_functions: Vec<AggregationFunction>,
    /// Whether to generate time-based features
    pub generate_temporal: bool,
    /// Whether to perform feature selection
    pub perform_selection: bool,
    /// Number of features to select (if None, use automatic selection)
    pub n_features_to_select: Option<usize>,
    /// Feature selection method
    pub selection_method: FeatureSelectionMethod,
    /// Whether to scale features
    pub scale_features: bool,
    /// Scaling method
    pub scaling_method: ScalingMethod,
    /// Generated feature names
    generated_features_: Option<Vec<String>>,
    /// Feature importance scores
    feature_scores_: Option<HashMap<String, f64>>,
    /// Selected feature indices
    selected_features_: Option<Vec<usize>>,
    /// Fitted scalers
    scalers_: Option<HashMap<String, Box<dyn FeatureScaler + Send + Sync>>>,
}

/// Aggregation functions for feature engineering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Mean,
    Median,
    Sum,
    Min,
    Max,
    Std,
    Var,
    Skew,
    Kurt,
    Count,
    Quantile(f64),
}

/// Feature selection methods
#[derive(Clone)]
pub enum FeatureSelectionMethod {
    /// Select k best features using univariate statistical tests
    KBest(ScoreFunction),
    /// Recursive feature elimination
    RecursiveElimination,
    /// L1-based feature selection (Lasso)
    L1Based,
    /// Tree-based feature importance
    TreeBased,
    /// Mutual information
    MutualInformation,
    /// Variance threshold
    VarianceThreshold(f64),
    /// Custom selection function
    Custom(Arc<dyn Fn(&DataFrame, &DataFrame) -> Result<Vec<usize>> + Send + Sync>),
}

impl std::fmt::Debug for FeatureSelectionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KBest(score_func) => f.debug_tuple("KBest").field(score_func).finish(),
            Self::RecursiveElimination => write!(f, "RecursiveElimination"),
            Self::L1Based => write!(f, "L1Based"),
            Self::TreeBased => write!(f, "TreeBased"),
            Self::MutualInformation => write!(f, "MutualInformation"),
            Self::VarianceThreshold(threshold) => {
                f.debug_tuple("VarianceThreshold").field(threshold).finish()
            }
            Self::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

/// Scaling methods for features
#[derive(Debug, Clone)]
pub enum ScalingMethod {
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    None,
}

/// Trait for feature scalers
pub trait FeatureScaler: std::fmt::Debug {
    fn fit(&mut self, data: &[f64]) -> Result<()>;
    fn transform(&self, data: &[f64]) -> Result<Vec<f64>>;
    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>>;
}

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
        let iqr = (q3 - q1).max(1e-10);

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

    /// Inverse Yeo-Johnson: recover x from transformed value y given lambda
    fn yeo_johnson_inverse(y: f64, lambda: f64) -> f64 {
        if y >= 0.0 {
            if (lambda - 0.0).abs() < 1e-10 {
                y.exp() - 1.0
            } else {
                (y * lambda + 1.0).powf(1.0 / lambda) - 1.0
            }
        } else if (lambda - 2.0).abs() < 1e-10 {
            1.0 - (-y).exp()
        } else {
            let two_minus_lambda = 2.0 - lambda;
            1.0 - (-y * two_minus_lambda + 1.0).powf(1.0 / two_minus_lambda)
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
        Ok(data
            .iter()
            .map(|&y| Self::yeo_johnson_inverse(y, lambda))
            .collect())
    }
}

impl AutoFeatureEngineer {
    /// Create a new AutoFeatureEngineer with default settings
    pub fn new() -> Self {
        Self {
            generate_polynomial: true,
            poly_degree: 2,
            generate_interactions: true,
            max_interaction_features: 5,
            generate_aggregations: true,
            aggregation_functions: vec![
                AggregationFunction::Mean,
                AggregationFunction::Std,
                AggregationFunction::Min,
                AggregationFunction::Max,
                AggregationFunction::Median,
            ],
            generate_temporal: false,
            perform_selection: true,
            n_features_to_select: None,
            selection_method: FeatureSelectionMethod::KBest(ScoreFunction::FRegression),
            scale_features: true,
            scaling_method: ScalingMethod::StandardScaler,
            generated_features_: None,
            feature_scores_: None,
            selected_features_: None,
            scalers_: None,
        }
    }

    /// Configure polynomial feature generation
    pub fn with_polynomial(mut self, degree: usize) -> Self {
        self.generate_polynomial = true;
        self.poly_degree = degree;
        self
    }

    /// Configure interaction feature generation
    pub fn with_interactions(mut self, max_features: usize) -> Self {
        self.generate_interactions = true;
        self.max_interaction_features = max_features;
        self
    }

    /// Configure aggregation feature generation
    pub fn with_aggregations(mut self, functions: Vec<AggregationFunction>) -> Self {
        self.generate_aggregations = true;
        self.aggregation_functions = functions;
        self
    }

    /// Configure feature selection
    pub fn with_selection(
        mut self,
        method: FeatureSelectionMethod,
        n_features: Option<usize>,
    ) -> Self {
        self.perform_selection = true;
        self.selection_method = method;
        self.n_features_to_select = n_features;
        self
    }

    /// Configure feature scaling
    pub fn with_scaling(mut self, method: ScalingMethod) -> Self {
        self.scale_features = true;
        self.scaling_method = method;
        self
    }

    /// Disable feature scaling
    pub fn without_scaling(mut self) -> Self {
        self.scale_features = false;
        self
    }

    /// Fit the feature engineering pipeline
    pub fn fit(&mut self, x: &DataFrame, y: Option<&DataFrame>) -> Result<()> {
        let start_time = Instant::now();

        // Start with original features
        let mut engineered_df = x.clone();
        let mut generated_features = x.column_names();

        // Generate polynomial features
        if self.generate_polynomial {
            let poly_features = self.generate_polynomial_features(&engineered_df)?;
            for (name, series) in poly_features {
                engineered_df.add_column(name.clone(), series)?;
                generated_features.push(name);
            }
        }

        // Generate interaction features
        if self.generate_interactions {
            let interaction_features = self.generate_interaction_features(&engineered_df)?;
            for (name, series) in interaction_features {
                engineered_df.add_column(name.clone(), series)?;
                generated_features.push(name);
            }
        }

        // Generate aggregation features
        if self.generate_aggregations {
            let agg_features = self.generate_aggregation_features(&engineered_df)?;
            for (name, series) in agg_features {
                engineered_df.add_column(name.clone(), series)?;
                generated_features.push(name);
            }
        }

        // Generate temporal features (if applicable)
        if self.generate_temporal {
            let temporal_features = self.generate_temporal_features(&engineered_df)?;
            for (name, series) in temporal_features {
                engineered_df.add_column(name.clone(), series)?;
                generated_features.push(name);
            }
        }

        // Perform feature selection
        if self.perform_selection {
            if let Some(y_data) = y {
                let selected_indices = self.select_features(&engineered_df, y_data)?;
                self.selected_features_ = Some(selected_indices);
            }
        }

        // Fit scalers
        if self.scale_features {
            let mut scalers = HashMap::new();

            for feature_name in &generated_features {
                let col = engineered_df.get_column::<f64>(feature_name)?;
                let values = col.as_f64()?;

                let mut scaler = self.create_scaler();
                scaler.fit(&values)?;
                scalers.insert(feature_name.clone(), scaler);
            }

            self.scalers_ = Some(scalers);
        }

        self.generated_features_ = Some(generated_features);

        println!(
            "Feature engineering completed in {:.2}s",
            start_time.elapsed().as_secs_f64()
        );
        println!(
            "Generated {} features",
            self.generated_features_
                .as_ref()
                .ok_or_else(|| Error::InvalidOperation(
                    "Model not fitted. Call fit() first.".into()
                ))?
                .len()
        );

        Ok(())
    }

    /// Transform data using the fitted feature engineering pipeline
    pub fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let generated_features = self.generated_features_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("AutoFeatureEngineer must be fitted before transform".into())
        })?;

        // Start with original features
        let mut result = x.clone();

        // Generate polynomial features
        if self.generate_polynomial {
            let poly_features = self.generate_polynomial_features(&result)?;
            for (name, series) in poly_features {
                result.add_column(name, series)?;
            }
        }

        // Generate interaction features
        if self.generate_interactions {
            let interaction_features = self.generate_interaction_features(&result)?;
            for (name, series) in interaction_features {
                result.add_column(name, series)?;
            }
        }

        // Generate aggregation features
        if self.generate_aggregations {
            let agg_features = self.generate_aggregation_features(&result)?;
            for (name, series) in agg_features {
                result.add_column(name, series)?;
            }
        }

        // Generate temporal features
        if self.generate_temporal {
            let temporal_features = self.generate_temporal_features(&result)?;
            for (name, series) in temporal_features {
                result.add_column(name, series)?;
            }
        }

        // Apply feature selection
        if let Some(selected_indices) = &self.selected_features_ {
            let all_feature_names = result.column_names();
            let mut selected_df = DataFrame::new();

            for &idx in selected_indices {
                if idx < all_feature_names.len() {
                    let feature_name = &all_feature_names[idx];
                    let col = result.get_column::<f64>(feature_name)?;
                    selected_df.add_column(feature_name.clone(), col.clone())?;
                }
            }

            result = selected_df;
        }

        // Apply scaling
        if let Some(scalers) = &self.scalers_ {
            let mut scaled_df = DataFrame::new();

            for feature_name in result.column_names() {
                let col = result.get_column::<f64>(&feature_name)?;
                let values = col.as_f64()?;

                if let Some(scaler) = scalers.get(&feature_name) {
                    let scaled_values = scaler.transform(&values)?;
                    scaled_df.add_column(
                        feature_name.clone(),
                        Series::new(scaled_values, Some(feature_name))?,
                    )?;
                } else {
                    scaled_df.add_column(feature_name, col.clone())?;
                }
            }

            result = scaled_df;
        }

        Ok(result)
    }

    /// Generate polynomial features
    fn generate_polynomial_features(&self, df: &DataFrame) -> Result<Vec<(String, Series<f64>)>> {
        let mut poly_features = Vec::new();
        let feature_names = df.column_names();
        let numeric_features: Vec<String> = feature_names
            .into_iter()
            .filter(|name| df.get_column::<f64>(name).is_ok())
            .collect();

        // Generate single-feature polynomial terms
        for feature_name in &numeric_features {
            let col = df.get_column::<f64>(feature_name)?;
            let values = col.as_f64()?;

            for degree in 2..=self.poly_degree {
                let poly_values: Vec<f64> = values.iter().map(|&x| x.powi(degree as i32)).collect();
                let poly_name = format!("{}^{}", feature_name, degree);
                poly_features.push((
                    poly_name.clone(),
                    Series::new(poly_values, Some(poly_name))?,
                ));
            }
        }

        // Generate cross-terms for degree 2
        if self.poly_degree >= 2 {
            for i in 0..numeric_features.len() {
                for j in (i + 1)..numeric_features.len() {
                    let feature1 = &numeric_features[i];
                    let feature2 = &numeric_features[j];

                    let col1 = df.get_column::<f64>(feature1)?;
                    let values1 = col1.as_f64()?;
                    let col2 = df.get_column::<f64>(feature2)?;
                    let values2 = col2.as_f64()?;

                    if values1.len() == values2.len() {
                        let cross_values: Vec<f64> = values1
                            .iter()
                            .zip(values2.iter())
                            .map(|(&x1, &x2)| x1 * x2)
                            .collect();
                        let cross_name = format!("{}*{}", feature1, feature2);
                        poly_features.push((
                            cross_name.clone(),
                            Series::new(cross_values, Some(cross_name))?,
                        ));
                    }
                }
            }
        }

        Ok(poly_features)
    }

    /// Generate interaction features
    fn generate_interaction_features(&self, df: &DataFrame) -> Result<Vec<(String, Series<f64>)>> {
        let mut interaction_features = Vec::new();
        let feature_names = df.column_names();
        let numeric_features: Vec<String> = feature_names
            .into_iter()
            .filter(|name| df.get_column::<f64>(name).is_ok())
            .take(self.max_interaction_features)
            .collect();

        // Generate pairwise interactions
        for i in 0..numeric_features.len() {
            for j in (i + 1)..numeric_features.len() {
                let feature1 = &numeric_features[i];
                let feature2 = &numeric_features[j];

                let col1 = df.get_column::<f64>(feature1)?;
                let values1 = col1.as_f64()?;
                let col2 = df.get_column::<f64>(feature2)?;
                let values2 = col2.as_f64()?;

                if values1.len() == values2.len() {
                    // Multiplication
                    let mult_values: Vec<f64> = values1
                        .iter()
                        .zip(values2.iter())
                        .map(|(&x1, &x2)| x1 * x2)
                        .collect();
                    let mult_name = format!("{}_mult_{}", feature1, feature2);
                    interaction_features.push((
                        mult_name.clone(),
                        Series::new(mult_values, Some(mult_name))?,
                    ));

                    // Division (with safety check)
                    let div_values: Vec<f64> = values1
                        .iter()
                        .zip(values2.iter())
                        .map(|(&x1, &x2)| if x2.abs() > 1e-10 { x1 / x2 } else { 0.0 })
                        .collect();
                    let div_name = format!("{}_div_{}", feature1, feature2);
                    interaction_features
                        .push((div_name.clone(), Series::new(div_values, Some(div_name))?));

                    // Addition
                    let add_values: Vec<f64> = values1
                        .iter()
                        .zip(values2.iter())
                        .map(|(&x1, &x2)| x1 + x2)
                        .collect();
                    let add_name = format!("{}_add_{}", feature1, feature2);
                    interaction_features
                        .push((add_name.clone(), Series::new(add_values, Some(add_name))?));

                    // Subtraction
                    let sub_values: Vec<f64> = values1
                        .iter()
                        .zip(values2.iter())
                        .map(|(&x1, &x2)| x1 - x2)
                        .collect();
                    let sub_name = format!("{}_sub_{}", feature1, feature2);
                    interaction_features
                        .push((sub_name.clone(), Series::new(sub_values, Some(sub_name))?));
                }
            }
        }

        Ok(interaction_features)
    }

    /// Generate aggregation features
    fn generate_aggregation_features(&self, df: &DataFrame) -> Result<Vec<(String, Series<f64>)>> {
        let mut agg_features = Vec::new();
        let feature_names = df.column_names();
        let numeric_features: Vec<String> = feature_names
            .into_iter()
            .filter(|name| df.get_column::<f64>(name).is_ok())
            .collect();

        // Generate aggregations across all numeric features for each row
        if numeric_features.len() > 1 {
            let n_rows = df.nrows();

            for agg_func in &self.aggregation_functions {
                let mut agg_values = Vec::with_capacity(n_rows);

                for row_idx in 0..n_rows {
                    let mut row_values = Vec::new();

                    for feature_name in &numeric_features {
                        let col = df.get_column::<f64>(feature_name)?;
                        let values = col.as_f64()?;
                        if row_idx < values.len() {
                            row_values.push(values[row_idx]);
                        }
                    }

                    let agg_value = self.calculate_aggregation(&row_values, agg_func)?;
                    agg_values.push(agg_value);
                }

                let agg_name = format!("row_{:?}", agg_func).to_lowercase();
                agg_features.push((agg_name.clone(), Series::new(agg_values, Some(agg_name))?));
            }
        }

        Ok(agg_features)
    }

    /// Generate temporal features (placeholder)
    fn generate_temporal_features(&self, _df: &DataFrame) -> Result<Vec<(String, Series<f64>)>> {
        // Placeholder for temporal feature generation
        // In a real implementation, this would detect datetime columns and generate
        // features like hour, day of week, month, etc.
        Ok(Vec::new())
    }

    /// Calculate aggregation value
    pub fn calculate_aggregation(&self, values: &[f64], func: &AggregationFunction) -> Result<f64> {
        if values.is_empty() {
            return Ok(0.0);
        }

        match func {
            AggregationFunction::Mean => Ok(values.iter().sum::<f64>() / values.len() as f64),
            AggregationFunction::Median => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                Ok(if sorted.len() % 2 == 0 {
                    (sorted[mid - 1] + sorted[mid]) / 2.0
                } else {
                    sorted[mid]
                })
            }
            AggregationFunction::Sum => Ok(values.iter().sum()),
            AggregationFunction::Min => Ok(values.iter().copied().fold(f64::INFINITY, f64::min)),
            AggregationFunction::Max => {
                Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            }
            AggregationFunction::Std => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                Ok(variance.sqrt())
            }
            AggregationFunction::Var => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                Ok(variance)
            }
            AggregationFunction::Skew => {
                // Simplified skewness calculation
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = {
                    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    variance.sqrt()
                };
                if std < 1e-10 {
                    Ok(0.0)
                } else {
                    let skew = values
                        .iter()
                        .map(|&x| ((x - mean) / std).powi(3))
                        .sum::<f64>()
                        / values.len() as f64;
                    Ok(skew)
                }
            }
            AggregationFunction::Kurt => {
                // Simplified kurtosis calculation
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = {
                    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    variance.sqrt()
                };
                if std < 1e-10 {
                    Ok(0.0)
                } else {
                    let kurt = values
                        .iter()
                        .map(|&x| ((x - mean) / std).powi(4))
                        .sum::<f64>()
                        / values.len() as f64
                        - 3.0;
                    Ok(kurt)
                }
            }
            AggregationFunction::Count => Ok(values.len() as f64),
            AggregationFunction::Quantile(q) => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((*q) * (sorted.len() - 1) as f64).round() as usize;
                Ok(sorted[idx.min(sorted.len() - 1)])
            }
        }
    }

    /// Select features using the configured selection method
    fn select_features(&mut self, x: &DataFrame, y: &DataFrame) -> Result<Vec<usize>> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();

        let selected_indices = match &self.selection_method {
            FeatureSelectionMethod::KBest(score_func) => {
                let k = self.n_features_to_select.unwrap_or(n_features.min(20));
                let mut selector = SelectKBest::new(score_func.clone(), k);
                selector.fit(x, y)?;
                selector.get_selected_features().unwrap_or(&[]).to_vec()
            }
            FeatureSelectionMethod::VarianceThreshold(threshold) => {
                self.select_by_variance_threshold(x, *threshold)?
            }
            FeatureSelectionMethod::RecursiveElimination => {
                let k = self.n_features_to_select.unwrap_or(n_features.min(10));
                self.select_recursive_elimination(x, y, k)?
            }
            FeatureSelectionMethod::L1Based => {
                let k = self.n_features_to_select.unwrap_or(n_features.min(15));
                self.select_l1_based(x, y, k)?
            }
            FeatureSelectionMethod::TreeBased => {
                let k = self.n_features_to_select.unwrap_or(n_features.min(12));
                self.select_tree_based(x, y, k)?
            }
            FeatureSelectionMethod::MutualInformation => {
                let k = self.n_features_to_select.unwrap_or(n_features.min(18));
                self.select_mutual_information(x, y, k)?
            }
            FeatureSelectionMethod::Custom(func) => func(x, y)?,
        };

        Ok(selected_indices)
    }

    /// Recursive Feature Elimination (RFE): iteratively removes the feature
    /// with the smallest absolute LinearRegression coefficient until `k` remain.
    fn select_recursive_elimination(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        k: usize,
    ) -> Result<Vec<usize>> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        // Extract target column from y (use first column)
        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?.to_vec();

        // Track which original feature indices are still active (by position in feature_names)
        let mut active_indices: Vec<usize> = (0..n_features).collect();

        while active_indices.len() > k {
            // Build a temporary DataFrame with active features + target
            let mut train_df = DataFrame::new();
            for &idx in &active_indices {
                let fname = &feature_names[idx];
                let col = x.get_column::<f64>(fname)?;
                train_df.add_column(fname.clone(), col.clone())?;
            }
            train_df.add_column(
                target_col_name.clone(),
                Series::new(target_vals.clone(), Some(target_col_name.clone()))?,
            )?;

            // Fit LinearRegression on the current active feature subset
            let mut lr = LinearRegression::new();
            match lr.fit(&train_df, target_col_name) {
                Ok(()) => {}
                Err(_) => {
                    // If fit fails (e.g., singular matrix), just drop the last active feature
                    active_indices.pop();
                    continue;
                }
            }

            let coefs = lr.coefficients.as_ref().ok_or_else(|| {
                Error::InvalidOperation("LinearRegression fit produced no coefficients".into())
            })?;

            // Find the active feature with the smallest |coefficient|
            let mut min_importance = f64::INFINITY;
            let mut min_pos = 0usize; // position in active_indices to remove

            for (pos, &orig_idx) in active_indices.iter().enumerate() {
                let fname = &feature_names[orig_idx];
                let importance = coefs.get(fname).map(|c| c.abs()).unwrap_or(0.0);
                if importance < min_importance {
                    min_importance = importance;
                    min_pos = pos;
                }
            }

            active_indices.remove(min_pos);
        }

        active_indices.sort_unstable();
        Ok(active_indices)
    }

    /// L1-Based selection: fit LinearRegression on all features at once, then
    /// select the `k` features with the largest |coefficient| (Lasso approximation).
    fn select_l1_based(&self, x: &DataFrame, y: &DataFrame, k: usize) -> Result<Vec<usize>> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        // Build training DataFrame
        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?.to_vec();

        let mut train_df = x.clone();
        train_df.add_column(
            target_col_name.clone(),
            Series::new(target_vals, Some(target_col_name.clone()))?,
        )?;

        // Fit a single LinearRegression on all features
        let mut lr = LinearRegression::new();
        lr.fit(&train_df, target_col_name).map_err(|e| {
            Error::Computation(format!("L1Based feature selection: fit failed: {}", e))
        })?;

        let coefs = lr.coefficients.as_ref().ok_or_else(|| {
            Error::InvalidOperation("LinearRegression fit produced no coefficients".into())
        })?;

        // Compute |coef| / sum(|coef|) importance per feature
        let importances: Vec<f64> = feature_names
            .iter()
            .map(|name| coefs.get(name).map(|c| c.abs()).unwrap_or(0.0))
            .collect();

        let total: f64 = importances.iter().sum();
        let importances: Vec<f64> = if total > 1e-10 {
            importances.iter().map(|&v| v / total).collect()
        } else {
            importances
        };

        // Sort feature indices by importance descending, take top-k
        let mut indexed: Vec<(usize, f64)> = importances.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
        selected.sort_unstable();
        Ok(selected)
    }

    /// Tree-Based approximation: rank features by variance × |correlation with target|.
    ///
    /// This approximates tree-based feature importance without requiring a decision
    /// tree implementation. High-variance features that strongly correlate with the
    /// target receive the highest scores.
    fn select_tree_based(&self, x: &DataFrame, y: &DataFrame, k: usize) -> Result<Vec<usize>> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?;

        let n = target_vals.len() as f64;
        let target_mean = target_vals.iter().sum::<f64>() / n;

        let mut scores: Vec<f64> = Vec::with_capacity(n_features);

        for feat_name in &feature_names {
            let col = x.get_column::<f64>(feat_name)?;
            let feat_vals = col.as_f64()?;

            if feat_vals.len() != target_vals.len() {
                scores.push(0.0);
                continue;
            }

            let feat_mean = feat_vals.iter().sum::<f64>() / n;

            // Variance of the feature
            let variance = feat_vals
                .iter()
                .map(|&v| (v - feat_mean).powi(2))
                .sum::<f64>()
                / n;

            // Pearson correlation with target
            let mut sum_xy = 0.0_f64;
            let mut sum_xx = 0.0_f64;
            let mut sum_yy = 0.0_f64;
            for (&fv, &tv) in feat_vals.iter().zip(target_vals.iter()) {
                let dx = fv - feat_mean;
                let dy = tv - target_mean;
                sum_xy += dx * dy;
                sum_xx += dx * dx;
                sum_yy += dy * dy;
            }
            let corr = if sum_xx > 1e-10 && sum_yy > 1e-10 {
                (sum_xy / (sum_xx * sum_yy).sqrt()).abs()
            } else {
                0.0
            };

            scores.push(variance * corr);
        }

        // Select top-k indices by score descending
        let mut indexed: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
        selected.sort_unstable();
        Ok(selected)
    }

    /// Mutual Information selection: estimate MI between each feature and the target
    /// using histogram-based density estimation, then select top-k features.
    ///
    /// Bins are computed per-feature using the Sturges rule: k = ceil(log2(n) + 1).
    fn select_mutual_information(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        k: usize,
    ) -> Result<Vec<usize>> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();
        let k = k.min(n_features).max(1);

        let y_col_names = y.column_names();
        if y_col_names.is_empty() {
            return Err(Error::InvalidValue(
                "Target DataFrame has no columns".into(),
            ));
        }
        let target_col_name = &y_col_names[0];
        let target_col = y.get_column::<f64>(target_col_name)?;
        let target_vals = target_col.as_f64()?;

        let n = target_vals.len();
        if n == 0 {
            return Ok((0..k).collect());
        }

        // Sturges rule for number of histogram bins
        let n_bins = ((n as f64).log2().ceil() as usize + 1).max(2).min(50);

        let mut mi_scores: Vec<f64> = Vec::with_capacity(n_features);

        // Precompute target histogram range
        let t_min = target_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = target_vals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let t_range = (t_max - t_min).max(1e-10);

        for feat_name in &feature_names {
            let col = x.get_column::<f64>(feat_name)?;
            let feat_vals = col.as_f64()?;

            if feat_vals.len() != n {
                mi_scores.push(0.0);
                continue;
            }

            let f_min = feat_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let f_max = feat_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let f_range = (f_max - f_min).max(1e-10);

            // Build 2D joint histogram: n_bins × n_bins
            let mut joint_hist = vec![vec![0usize; n_bins]; n_bins];
            let mut feat_hist = vec![0usize; n_bins];
            let mut target_hist = vec![0usize; n_bins];

            for (&fv, &tv) in feat_vals.iter().zip(target_vals.iter()) {
                let fi = (((fv - f_min) / f_range) * (n_bins as f64 - 1.0)).floor() as usize;
                let ti = (((tv - t_min) / t_range) * (n_bins as f64 - 1.0)).floor() as usize;
                let fi = fi.min(n_bins - 1);
                let ti = ti.min(n_bins - 1);
                joint_hist[fi][ti] += 1;
                feat_hist[fi] += 1;
                target_hist[ti] += 1;
            }

            // Compute MI = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
            let n_f = n as f64;
            let mut mi = 0.0_f64;
            for fi in 0..n_bins {
                for ti in 0..n_bins {
                    let pxy = joint_hist[fi][ti] as f64 / n_f;
                    let px = feat_hist[fi] as f64 / n_f;
                    let py = target_hist[ti] as f64 / n_f;
                    if pxy > 1e-12 && px > 1e-12 && py > 1e-12 {
                        mi += pxy * (pxy / (px * py)).ln();
                    }
                }
            }
            mi_scores.push(mi.max(0.0));
        }

        // Select top-k by MI score descending
        let mut indexed: Vec<(usize, f64)> = mi_scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut selected: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
        selected.sort_unstable();
        Ok(selected)
    }

    /// Select features by variance threshold
    fn select_by_variance_threshold(&self, x: &DataFrame, threshold: f64) -> Result<Vec<usize>> {
        let feature_names = x.column_names();
        let mut selected_indices = Vec::new();

        for (i, feature_name) in feature_names.iter().enumerate() {
            let col = x.get_column::<f64>(feature_name)?;
            let values = col.as_f64()?;

            if values.is_empty() {
                continue;
            }

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

            if variance > threshold {
                selected_indices.push(i);
            }
        }

        Ok(selected_indices)
    }

    /// Create a scaler based on the scaling method
    fn create_scaler(&self) -> Box<dyn FeatureScaler + Send + Sync> {
        match self.scaling_method {
            ScalingMethod::StandardScaler => Box::new(StandardScaler::new()),
            ScalingMethod::MinMaxScaler => Box::new(MinMaxScaler::new()),
            ScalingMethod::RobustScaler => Box::new(RobustScaler::new()),
            ScalingMethod::QuantileTransformer => Box::new(QuantileTransformer::new()),
            ScalingMethod::PowerTransformer => Box::new(PowerTransformer::new()),
            ScalingMethod::None => Box::new(StandardScaler::new()),
        }
    }

    /// Get generated feature names
    pub fn get_feature_names(&self) -> Option<&[String]> {
        self.generated_features_.as_ref().map(|f| f.as_slice())
    }

    /// Get feature importance scores
    pub fn get_feature_scores(&self) -> Option<&HashMap<String, f64>> {
        self.feature_scores_.as_ref()
    }

    /// Get selected feature indices
    pub fn get_selected_features(&self) -> Option<&[usize]> {
        self.selected_features_.as_ref().map(|f| f.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_auto_feature_engineer() {
        let mut engineer = AutoFeatureEngineer::new()
            .with_polynomial(2)
            .with_interactions(3)
            .with_scaling(ScalingMethod::StandardScaler);

        // Create test data
        let mut x = DataFrame::new();
        x.add_column(
            "feature1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");
        x.add_column(
            "feature2".to_string(),
            Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("feature2".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");

        let mut y = DataFrame::new();
        y.add_column(
            "target".to_string(),
            Series::new(vec![3.0, 6.0, 9.0, 12.0, 15.0], Some("target".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");

        // Fit and transform
        engineer
            .fit(&x, Some(&y))
            .expect("operation should succeed");
        let transformed = engineer.transform(&x).expect("operation should succeed");

        // Should have more features than original
        assert!(transformed.column_names().len() > x.column_names().len());
    }

    #[test]
    fn test_standard_scaler() {
        let mut scaler = StandardScaler::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        scaler.fit(&data).expect("operation should succeed");
        let transformed = scaler.transform(&data).expect("operation should succeed");

        // Check that mean is approximately zero
        let mean = transformed.iter().sum::<f64>() / transformed.len() as f64;
        assert!((mean).abs() < 1e-10);

        // Check that std is approximately one
        let variance =
            transformed.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / transformed.len() as f64;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_scaler() {
        let mut scaler = MinMaxScaler::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        scaler.fit(&data).expect("operation should succeed");
        let transformed = scaler.transform(&data).expect("operation should succeed");

        // Check range
        let min = transformed.iter().copied().fold(f64::INFINITY, f64::min);
        let max = transformed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!((min - 0.0).abs() < 1e-10);
        assert!((max - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregation_functions() {
        let engineer = AutoFeatureEngineer::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let mean = engineer
            .calculate_aggregation(&values, &AggregationFunction::Mean)
            .expect("operation should succeed");
        assert!((mean - 3.0).abs() < 1e-10);

        let sum = engineer
            .calculate_aggregation(&values, &AggregationFunction::Sum)
            .expect("operation should succeed");
        assert!((sum - 15.0).abs() < 1e-10);

        let min = engineer
            .calculate_aggregation(&values, &AggregationFunction::Min)
            .expect("operation should succeed");
        assert!((min - 1.0).abs() < 1e-10);

        let max = engineer
            .calculate_aggregation(&values, &AggregationFunction::Max)
            .expect("operation should succeed");
        assert!((max - 5.0).abs() < 1e-10);
    }

    /// Build a test DataFrame with named f64 columns
    fn make_df(cols: &[(&str, Vec<f64>)]) -> DataFrame {
        let mut df = DataFrame::new();
        for (name, vals) in cols {
            df.add_column(
                name.to_string(),
                Series::new(vals.clone(), Some(name.to_string()))
                    .expect("series creation should succeed"),
            )
            .expect("add_column should succeed");
        }
        df
    }

    #[test]
    fn test_recursive_elimination_selects_k() {
        // feature0 = 2 * target (highly important), features 1-4 = noise-like constants/small
        let target: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let feat0: Vec<f64> = target.iter().map(|&t| 2.0 * t).collect();
        let feat1: Vec<f64> = (1..=20).map(|i| (i as f64 * 0.01).sin()).collect();
        let feat2: Vec<f64> = vec![0.1; 20];
        let feat3: Vec<f64> = vec![0.2; 20];
        let feat4: Vec<f64> = vec![0.3; 20];

        let x = make_df(&[
            ("feat0", feat0),
            ("feat1", feat1),
            ("feat2", feat2),
            ("feat3", feat3),
            ("feat4", feat4),
        ]);
        let y = make_df(&[("target", target)]);

        let mut engineer = AutoFeatureEngineer::new()
            .with_selection(FeatureSelectionMethod::RecursiveElimination, Some(2))
            .without_scaling();
        // Disable auto feature generation to keep the feature set small
        engineer.generate_polynomial = false;
        engineer.generate_interactions = false;
        engineer.generate_aggregations = false;

        engineer.fit(&x, Some(&y)).expect("fit should succeed");
        let selected = engineer
            .get_selected_features()
            .expect("selected features should exist");

        // feature0 (index 0) should be among the top-2 selected
        assert!(
            selected.contains(&0),
            "feat0 (index 0) should be selected; got {:?}",
            selected
        );
        assert_eq!(selected.len(), 2, "should select exactly 2 features");
    }

    #[test]
    fn test_l1_based_selects_k() {
        // All features are non-constant, non-collinear to avoid singular matrix
        // feat0 = 3 * target (dominant), others are orthogonal-ish patterns
        let target: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let feat0: Vec<f64> = target.iter().map(|&t| 3.0 * t).collect();
        let feat1: Vec<f64> = (1..=20)
            .map(|i| (i as f64 * 0.3).sin() + i as f64 * 0.001)
            .collect();
        let feat2: Vec<f64> = (1..=20)
            .map(|i| (i as f64 * 0.7).cos() * 0.1 + i as f64 * 0.002)
            .collect();
        let feat3: Vec<f64> = (1..=20)
            .map(|i| (i as f64 * 1.1).sin() * 0.05 - i as f64 * 0.0005)
            .collect();
        let feat4: Vec<f64> = (1..=20)
            .map(|i| (i as f64 * 1.5).cos() * 0.02 + i as f64 * 0.0001)
            .collect();

        let x = make_df(&[
            ("feat0", feat0),
            ("feat1", feat1),
            ("feat2", feat2),
            ("feat3", feat3),
            ("feat4", feat4),
        ]);
        let y = make_df(&[("target", target)]);

        let mut engineer = AutoFeatureEngineer::new()
            .with_selection(FeatureSelectionMethod::L1Based, Some(2))
            .without_scaling();
        engineer.generate_polynomial = false;
        engineer.generate_interactions = false;
        engineer.generate_aggregations = false;

        engineer.fit(&x, Some(&y)).expect("fit should succeed");
        let selected = engineer
            .get_selected_features()
            .expect("selected features should exist");

        assert!(
            selected.contains(&0),
            "feat0 should be selected by L1Based; got {:?}",
            selected
        );
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_robust_scaler_median_zero() {
        let mut scaler = RobustScaler::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        scaler.fit(&data).expect("fit should succeed");
        let transformed = scaler.transform(&data).expect("transform should succeed");

        // median of [1,2,3,4,5] is 3; after subtracting median the middle element maps to 0
        let middle = transformed[2]; // was 3.0, should be (3-3)/IQR = 0.0
        assert!(
            middle.abs() < 1e-10,
            "median element should transform to 0.0, got {}",
            middle
        );

        // Round-trip: inverse_transform should recover original values
        let recovered = scaler
            .inverse_transform(&transformed)
            .expect("inverse_transform should succeed");
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-10,
                "inverse transform should recover {}, got {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_mutual_info_selection() {
        // feat0 = constant (zero MI), feat1 = perfectly correlated with target (high MI)
        let target: Vec<f64> = (1..=30).map(|i| i as f64).collect();
        let feat0: Vec<f64> = vec![5.0; 30]; // constant → zero variance → zero MI
        let feat1: Vec<f64> = target.iter().map(|&t| t * 2.0 + 1.0).collect();

        let x = make_df(&[("constant", feat0), ("correlated", feat1)]);
        let y = make_df(&[("target", target)]);

        let mut engineer = AutoFeatureEngineer::new()
            .with_selection(FeatureSelectionMethod::MutualInformation, Some(1))
            .without_scaling();
        engineer.generate_polynomial = false;
        engineer.generate_interactions = false;
        engineer.generate_aggregations = false;

        engineer.fit(&x, Some(&y)).expect("fit should succeed");
        let selected = engineer
            .get_selected_features()
            .expect("selected features should exist");

        // The correlated feature (index 1) should be preferred over the constant (index 0)
        assert!(
            selected.contains(&1),
            "correlated feature (index 1) should be selected; got {:?}",
            selected
        );
        assert_eq!(selected.len(), 1);
    }
}
