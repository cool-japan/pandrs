//! Automated feature engineering and selection
//!
//! This module provides comprehensive feature engineering capabilities including
//! automated feature generation, selection, and transformation pipelines.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::model_selection::{ScoreFunction, SelectKBest};
use crate::ml::sklearn_compat::{SklearnEstimator, SklearnTransformer};
use crate::series::Series;
use crate::utils::rand_compat::{thread_rng, GenRangeCompat};
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
        let std = self.std.unwrap();

        Ok(data.iter().map(|&x| (x - mean) / std).collect())
    }

    fn inverse_transform(&self, data: &[f64]) -> Result<Vec<f64>> {
        let mean = self
            .mean
            .ok_or_else(|| Error::InvalidOperation("Scaler not fitted".into()))?;
        let std = self.std.unwrap();

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
        let max = self.max.unwrap();
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
        let max = self.max.unwrap();
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
        if self.perform_selection && y.is_some() {
            let selected_indices = self.select_features(&engineered_df, y.unwrap())?;
            self.selected_features_ = Some(selected_indices);
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
            self.generated_features_.as_ref().unwrap().len()
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
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
                // Placeholder for RFE implementation
                (0..n_features.min(10)).collect()
            }
            FeatureSelectionMethod::L1Based => {
                // Placeholder for L1-based selection
                (0..n_features.min(15)).collect()
            }
            FeatureSelectionMethod::TreeBased => {
                // Placeholder for tree-based selection
                (0..n_features.min(12)).collect()
            }
            FeatureSelectionMethod::MutualInformation => {
                // Placeholder for mutual information selection
                (0..n_features.min(18)).collect()
            }
            FeatureSelectionMethod::Custom(func) => func(x, y)?,
        };

        Ok(selected_indices)
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
            ScalingMethod::RobustScaler => Box::new(StandardScaler::new()), // Placeholder
            ScalingMethod::QuantileTransformer => Box::new(StandardScaler::new()), // Placeholder
            ScalingMethod::PowerTransformer => Box::new(StandardScaler::new()), // Placeholder
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
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string())).unwrap(),
        )
        .unwrap();
        x.add_column(
            "feature2".to_string(),
            Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("feature2".to_string())).unwrap(),
        )
        .unwrap();

        let mut y = DataFrame::new();
        y.add_column(
            "target".to_string(),
            Series::new(vec![3.0, 6.0, 9.0, 12.0, 15.0], Some("target".to_string())).unwrap(),
        )
        .unwrap();

        // Fit and transform
        engineer.fit(&x, Some(&y)).unwrap();
        let transformed = engineer.transform(&x).unwrap();

        // Should have more features than original
        assert!(transformed.column_names().len() > x.column_names().len());
    }

    #[test]
    fn test_standard_scaler() {
        let mut scaler = StandardScaler::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        scaler.fit(&data).unwrap();
        let transformed = scaler.transform(&data).unwrap();

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

        scaler.fit(&data).unwrap();
        let transformed = scaler.transform(&data).unwrap();

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
            .unwrap();
        assert!((mean - 3.0).abs() < 1e-10);

        let sum = engineer
            .calculate_aggregation(&values, &AggregationFunction::Sum)
            .unwrap();
        assert!((sum - 15.0).abs() < 1e-10);

        let min = engineer
            .calculate_aggregation(&values, &AggregationFunction::Min)
            .unwrap();
        assert!((min - 1.0).abs() < 1e-10);

        let max = engineer
            .calculate_aggregation(&values, &AggregationFunction::Max)
            .unwrap();
        assert!((max - 5.0).abs() < 1e-10);
    }
}
