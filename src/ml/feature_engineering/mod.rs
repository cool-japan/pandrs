//! Automated feature engineering and selection
//!
//! This module provides comprehensive feature engineering capabilities including
//! automated feature generation, selection, and transformation pipelines.
//!
//! Split across submodules to stay under the project's file-size policy:
//!   - `scalers`: the individual [`FeatureScaler`] implementations.
//!   - `polynomial`: the shared full-polynomial-basis helpers (also used by
//!     [`pipeline_extended`](crate::ml::pipeline_extended)).
//!   - `selection`: `select_features` and each `FeatureSelectionMethod`'s implementation.
//!   - `tests`: unit tests for all of the above.

mod polynomial;
mod scalers;
mod selection;
#[cfg(test)]
mod tests;

pub(crate) use polynomial::{combinations_with_replacement, monomial_name};
pub use scalers::{
    MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler,
};

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::model_selection::ScoreFunction;
use crate::series::Series;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
        let mut generated_features = x.column_names().to_vec();

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

        // Fit scalers over the FULL generated feature set BEFORE selection.
        // Several selection criteria (variance threshold, tree-based
        // importances, L1-based coefficient magnitude) are scale-sensitive:
        // running them on unscaled data lets a feature dominate purely
        // because of its native units rather than its real relationship
        // with the target (measured: a noise feature scoring 5613 vs. a
        // real signal's 0.726 under an unscaled variance*correlation
        // ranking). Scaling first -- exactly as a scikit-learn
        // `Pipeline([("scaler", ...), ("select", ...)])` would -- fixes
        // this at the source instead of downstream.
        let selection_input = if self.scale_features {
            let mut scalers = HashMap::new();
            let mut scaled = DataFrame::new();

            for feature_name in &generated_features {
                let col = engineered_df.get_column::<f64>(feature_name)?;
                let values = col.as_f64()?;

                let mut scaler = self.create_scaler();
                scaler.fit(&values)?;
                let scaled_values = scaler.transform(&values)?;
                scaled.add_column(
                    feature_name.clone(),
                    Series::new(scaled_values, Some(feature_name.clone()))?,
                )?;
                scalers.insert(feature_name.clone(), scaler);
            }

            self.scalers_ = Some(scalers);
            scaled
        } else {
            engineered_df
        };

        // Perform feature selection on the (possibly scaled) data
        if self.perform_selection {
            if let Some(y_data) = y {
                let selected_indices = self.select_features(&selection_input, y_data)?;
                self.selected_features_ = Some(selected_indices);
            }
        }

        self.generated_features_ = Some(generated_features);

        log::info!(
            "Feature engineering completed in {:.2}s, generated {} features",
            start_time.elapsed().as_secs_f64(),
            self.generated_features_
                .as_ref()
                .map(|f| f.len())
                .unwrap_or(0)
        );

        Ok(())
    }

    /// Transform data using the fitted feature engineering pipeline
    pub fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let _generated_features = self.generated_features_.as_ref().ok_or_else(|| {
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

        // Apply scaling BEFORE selection, mirroring the order feature
        // selection scores were actually computed in during `fit` (scale,
        // then select) -- see the comment there.
        if let Some(scalers) = &self.scalers_ {
            let mut scaled_df = DataFrame::new();

            for feature_name in result.column_names() {
                let col = result.get_column::<f64>(&feature_name)?;
                let values = col.as_f64()?;

                if let Some(scaler) = scalers.get(feature_name.as_str()) {
                    let scaled_values = scaler.transform(&values)?;
                    scaled_df.add_column(
                        feature_name.clone(),
                        Series::new(scaled_values, Some(feature_name.clone()))?,
                    )?;
                } else {
                    scaled_df.add_column(feature_name.clone(), col.clone())?;
                }
            }

            result = scaled_df;
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

        Ok(result)
    }

    /// Generate the full polynomial feature basis up to `self.poly_degree`:
    /// every distinct monomial of total degree `2..=poly_degree` formed
    /// from the numeric feature columns -- e.g. for degree 3 this includes
    /// not just pure powers (`x^3`) and pairwise products (`x*y`), but also
    /// mixed-degree cross terms (`x^2*y`) and three-way products
    /// (`x*y*z`). This mirrors the basis
    /// `sklearn.preprocessing.PolynomialFeatures` generates, minus the
    /// bias term and the degree-1 columns (already present in `df`).
    fn generate_polynomial_features(&self, df: &DataFrame) -> Result<Vec<(String, Series<f64>)>> {
        let mut poly_features = Vec::new();
        let feature_names = df.column_names();
        let numeric_features: Vec<String> = feature_names
            .iter()
            .filter(|name| df.get_column::<f64>(name.as_str()).is_ok())
            .cloned()
            .collect();

        if numeric_features.is_empty() || self.poly_degree < 2 {
            return Ok(poly_features);
        }

        // Fetch every numeric column's values once; monomials at every
        // degree reuse the same underlying vectors.
        let columns: Vec<Vec<f64>> = numeric_features
            .iter()
            .map(|name| {
                let col = df.get_column::<f64>(name)?;
                col.as_f64().map(|v| v.to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        let n_rows = columns.first().map(|c| c.len()).unwrap_or(0);

        for degree in 2..=self.poly_degree {
            for combo in combinations_with_replacement(numeric_features.len(), degree) {
                let name = monomial_name(&combo, &numeric_features);
                let values: Vec<f64> = (0..n_rows)
                    .map(|row| combo.iter().map(|&idx| columns[idx][row]).product())
                    .collect();
                poly_features.push((name.clone(), Series::new(values, Some(name))?));
            }
        }

        Ok(poly_features)
    }

    /// Generate pairwise interaction features: division, addition, and
    /// subtraction between every pair of (the first
    /// `max_interaction_features`) numeric columns.
    ///
    /// Multiplication (`a*b`) is intentionally NOT generated here: it is
    /// exactly the degree-2 cross term `generate_polynomial_features`
    /// already produces when polynomial generation is enabled (the
    /// default), and generating it in both places previously created two
    /// exactly-collinear columns under different names (`"a*b"` and
    /// `"a_mult_b"`) -- pure multicollinearity with no added information.
    fn generate_interaction_features(&self, df: &DataFrame) -> Result<Vec<(String, Series<f64>)>> {
        let mut interaction_features = Vec::new();
        let feature_names = df.column_names();
        let numeric_features: Vec<String> = feature_names
            .iter()
            .filter(|name| df.get_column::<f64>(name.as_str()).is_ok())
            .take(self.max_interaction_features)
            .cloned()
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
            .iter()
            .filter(|name| df.get_column::<f64>(name.as_str()).is_ok())
            .cloned()
            .collect();

        // Generate aggregations across all numeric features for each row
        if numeric_features.len() > 1 {
            let n_rows = df.nrows();

            // Fetch every numeric column's values once up front. The
            // row/aggregation-function loops below previously re-fetched
            // every column on every row for every aggregation function --
            // O(rows * features * agg_funcs) column lookups for what only
            // needs O(features) lookups total.
            let columns: Vec<Vec<f64>> = numeric_features
                .iter()
                .map(|name| {
                    let col = df.get_column::<f64>(name)?;
                    col.as_f64().map(|v| v.to_vec())
                })
                .collect::<Result<Vec<_>>>()?;

            for agg_func in &self.aggregation_functions {
                let mut agg_values = Vec::with_capacity(n_rows);

                for row_idx in 0..n_rows {
                    let row_values: Vec<f64> = columns
                        .iter()
                        .filter_map(|col| col.get(row_idx).copied())
                        .collect();

                    let agg_value = self.calculate_aggregation(&row_values, agg_func)?;
                    agg_values.push(agg_value);
                }

                let agg_name = format!("row_{:?}", agg_func).to_lowercase();
                agg_features.push((agg_name.clone(), Series::new(agg_values, Some(agg_name))?));
            }
        }

        Ok(agg_features)
    }

    /// Generate temporal features (year, month, day, hour, weekday) from
    /// every column whose values are fully parseable as datetimes.
    ///
    /// A column is a datetime candidate when it is not itself numeric
    /// (`f64`) and every one of its stringified values parses via the same
    /// fallback chain used elsewhere in this crate to build
    /// `Series<NaiveDateTime>` from strings (RFC 3339, then
    /// `"%Y-%m-%d %H:%M:%S"`, then `"%Y-%m-%d"`; see
    /// `series::datetime_accessor::datetime_constructors::parse_datetime_series`).
    /// Columns that are numeric, or where any single value fails to parse,
    /// contribute nothing: this is best-effort auto-detection over an
    /// arbitrary feature `DataFrame`, not a hard requirement that every
    /// column be a datetime.
    fn generate_temporal_features(&self, df: &DataFrame) -> Result<Vec<(String, Series<f64>)>> {
        use chrono::{Datelike, Timelike};

        let mut temporal_features = Vec::new();

        for col_name in df.column_names() {
            // Numeric columns are never datetime candidates.
            if df.get_column::<f64>(col_name.as_str()).is_ok() {
                continue;
            }

            let string_values = match df.get_column_string_values(col_name.as_str()) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if string_values.is_empty() {
                continue;
            }

            let parsed =
                crate::series::datetime_accessor::datetime_constructors::parse_datetime_series(
                    string_values,
                    None,
                    None,
                );
            let datetime_series = match parsed {
                Ok(series) => series,
                Err(_) => continue, // not a (fully) parseable datetime column
            };
            let datetimes = datetime_series.values();

            let years: Vec<f64> = datetimes.iter().map(|dt| dt.year() as f64).collect();
            let months: Vec<f64> = datetimes.iter().map(|dt| dt.month() as f64).collect();
            let days: Vec<f64> = datetimes.iter().map(|dt| dt.day() as f64).collect();
            let hours: Vec<f64> = datetimes.iter().map(|dt| dt.hour() as f64).collect();
            let weekdays: Vec<f64> = datetimes
                .iter()
                .map(|dt| dt.weekday().num_days_from_monday() as f64)
                .collect();

            for (suffix, values) in [
                ("year", years),
                ("month", months),
                ("day", days),
                ("hour", hours),
                ("weekday", weekdays),
            ] {
                let name = format!("{}_{}", col_name, suffix);
                temporal_features.push((name.clone(), Series::new(values, Some(name))?));
            }
        }

        Ok(temporal_features)
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
                // Linear interpolation between order statistics, matching
                // the convention used everywhere else in this module
                // (`RobustScaler`'s Q1/Q3, `QuantileTransformer`) instead of
                // nearest-rank rounding.
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = sorted.len();
                if n == 1 {
                    return Ok(sorted[0]);
                }
                let q = q.clamp(0.0, 1.0);
                let pos = q * (n - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = (pos.ceil() as usize).min(n - 1);
                let frac = pos - lo as f64;
                Ok(sorted[lo] + frac * (sorted[hi] - sorted[lo]))
            }
        }
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
