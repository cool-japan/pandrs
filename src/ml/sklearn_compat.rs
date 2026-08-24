//! Scikit-learn compatibility layer for PandRS ML
//!
//! This module provides comprehensive scikit-learn compatible interfaces,
//! allowing PandRS to be used as a drop-in replacement for many sklearn workflows.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::models::ensemble::{
    GradientBoostingConfig, GradientBoostingRegressor, RandomForestConfig, RandomForestRegressor,
};
use crate::ml::models::linear::{LinearRegression, LogisticRegression};
use crate::ml::models::neural::{MLPClassifier, MLPRegressor};
use crate::ml::models::tree::{DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeRegressor};
use crate::ml::models::SupervisedModel;
use crate::series::Series;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;

// ─── Hyperparameter application for wrapped models ──────────────────────────────────────────
//
// `SupervisedModel` (the trait concrete models implement) intentionally has no generic
// `set_params`-style method of its own — each model's hyperparameters live in a
// model-specific, privately-held config struct with no public setters. To let
// `SupervisedAdapter::set_params` actually tune the wrapped model — instead of discarding every
// key but `target_col`, which was the root cause of grid/randomized search tuning nothing at
// all — the functions below downcast to each concrete model type pandrs ships and rebuild it
// from a fresh `Config::default()` overlaid with the recognized keys in `params`.
//
// Rebuilding (rather than mutating fields in place) is necessary because these models store
// their config privately with no setters, and it is *correct* to do here because `set_params`
// is always called on a freshly-cloned, not-yet-fitted estimator immediately before `fit()` in
// every search loop in this crate (`GridSearchCV`/`RandomizedSearchCV`); it is never called on a
// model whose fitted state must be preserved.
//
// Unknown parameter keys — and model types this function doesn't recognize at all — return an
// `Err` (matching scikit-learn's `set_params`, which raises `ValueError` on invalid parameters)
// instead of silently accepting hyperparameters that go nowhere.

fn parse_bool_param(key: &str, value: &str) -> Result<bool> {
    value
        .parse::<bool>()
        .map_err(|_| Error::InvalidValue(format!("Invalid boolean value for {key}: {value}")))
}

fn parse_usize_param(key: &str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| Error::InvalidValue(format!("Invalid usize value for {key}: {value}")))
}

fn parse_u64_param(key: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| Error::InvalidValue(format!("Invalid u64 value for {key}: {value}")))
}

fn parse_f64_param(key: &str, value: &str) -> Result<f64> {
    value
        .parse::<f64>()
        .map_err(|_| Error::InvalidValue(format!("Invalid f64 value for {key}: {value}")))
}

/// Parse an `Option<usize>`-valued parameter where the sentinel string `"none"`
/// (case-insensitive) or an empty string means `None` — the convention this crate's own
/// `ParameterDistribution::Choice` search spaces use for "unlimited", e.g.
/// `["3", "5", "10", "None"]` for `max_depth`.
fn parse_optional_usize_param(key: &str, value: &str) -> Result<Option<usize>> {
    if value.is_empty() || value.eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        Ok(Some(parse_usize_param(key, value)?))
    }
}

fn err_on_unknown_params(model_type: &str, unknown: Vec<String>) -> Result<()> {
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(Error::InvalidValue(format!(
            "set_params: unknown parameter(s) {:?} for {}",
            unknown, model_type
        )))
    }
}

fn set_linear_regression_params(
    model: &mut LinearRegression,
    params: &HashMap<String, String>,
) -> Result<()> {
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "fit_intercept" => model.fit_intercept = parse_bool_param(key, value)?,
            "normalize" => model.normalize = parse_bool_param(key, value)?,
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("LinearRegression", unknown)
}

fn set_logistic_regression_params(
    model: &mut LogisticRegression,
    params: &HashMap<String, String>,
) -> Result<()> {
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            // sklearn's inverse-regularization-strength hyperparameter is spelled "C"; accept
            // the lowercase form too since it matches the model's own field name.
            "C" | "c" => model.c = parse_f64_param(key, value)?,
            "fit_intercept" => model.fit_intercept = parse_bool_param(key, value)?,
            "max_iter" => model.max_iter = parse_usize_param(key, value)?,
            "tol" => model.tol = parse_f64_param(key, value)?,
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("LogisticRegression", unknown)
}

fn set_decision_tree_classifier_params(
    model: &mut DecisionTreeClassifier,
    params: &HashMap<String, String>,
) -> Result<()> {
    let mut config = DecisionTreeConfig::default();
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "max_depth" => config.max_depth = parse_optional_usize_param(key, value)?,
            "min_samples_split" => config.min_samples_split = parse_usize_param(key, value)?,
            "min_samples_leaf" => config.min_samples_leaf = parse_usize_param(key, value)?,
            "max_features" => config.max_features = parse_optional_usize_param(key, value)?,
            "random_seed" => config.random_seed = Some(parse_u64_param(key, value)?),
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("DecisionTreeClassifier", unknown)?;
    *model = DecisionTreeClassifier::new(config);
    Ok(())
}

fn set_decision_tree_regressor_params(
    model: &mut DecisionTreeRegressor,
    params: &HashMap<String, String>,
) -> Result<()> {
    let mut config = DecisionTreeConfig::default();
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "max_depth" => config.max_depth = parse_optional_usize_param(key, value)?,
            "min_samples_split" => config.min_samples_split = parse_usize_param(key, value)?,
            "min_samples_leaf" => config.min_samples_leaf = parse_usize_param(key, value)?,
            "max_features" => config.max_features = parse_optional_usize_param(key, value)?,
            "random_seed" => config.random_seed = Some(parse_u64_param(key, value)?),
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("DecisionTreeRegressor", unknown)?;
    *model = DecisionTreeRegressor::new(config);
    Ok(())
}

fn set_random_forest_regressor_params(
    model: &mut RandomForestRegressor,
    params: &HashMap<String, String>,
) -> Result<()> {
    let mut config = RandomForestConfig::default();
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "n_estimators" => config.n_estimators = parse_usize_param(key, value)?,
            "max_depth" => config.max_depth = parse_optional_usize_param(key, value)?,
            "min_samples_split" => config.min_samples_split = parse_usize_param(key, value)?,
            "min_samples_leaf" => config.min_samples_leaf = parse_usize_param(key, value)?,
            "max_features" => config.max_features = parse_optional_usize_param(key, value)?,
            "bootstrap" => config.bootstrap = parse_bool_param(key, value)?,
            "max_samples" => config.max_samples = parse_optional_usize_param(key, value)?,
            "random_seed" => config.random_seed = Some(parse_u64_param(key, value)?),
            "oob_score" => config.oob_score = parse_bool_param(key, value)?,
            "n_jobs" => config.n_jobs = parse_usize_param(key, value)?,
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("RandomForestRegressor", unknown)?;
    *model = RandomForestRegressor::new(config);
    Ok(())
}

fn set_gradient_boosting_regressor_params(
    model: &mut GradientBoostingRegressor,
    params: &HashMap<String, String>,
) -> Result<()> {
    let mut config = GradientBoostingConfig::default();
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "n_estimators" => config.n_estimators = parse_usize_param(key, value)?,
            "learning_rate" => config.learning_rate = parse_f64_param(key, value)?,
            "max_depth" => config.max_depth = parse_usize_param(key, value)?,
            "min_samples_split" => config.min_samples_split = parse_usize_param(key, value)?,
            "min_samples_leaf" => config.min_samples_leaf = parse_usize_param(key, value)?,
            "subsample" => config.subsample = parse_f64_param(key, value)?,
            "random_seed" => config.random_seed = Some(parse_u64_param(key, value)?),
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("GradientBoostingRegressor", unknown)?;
    *model = GradientBoostingRegressor::new(config);
    Ok(())
}

fn set_mlp_regressor_params(
    model: &mut MLPRegressor,
    params: &HashMap<String, String>,
) -> Result<()> {
    // MLPRegressor stores its config privately with no accessor either, so — like the
    // tree/ensemble models above — tuning rebuilds a fresh (unfitted) instance from
    // `MLPConfig::default()` overlaid with the recognized keys.
    let mut config = crate::ml::models::neural::MLPConfig::default();
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "learning_rate" => config.learning_rate = parse_f64_param(key, value)?,
            "n_epochs" => config.n_epochs = parse_usize_param(key, value)?,
            "batch_size" => config.batch_size = parse_usize_param(key, value)?,
            "random_seed" => config.random_seed = parse_u64_param(key, value)?,
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("MLPRegressor", unknown)?;
    *model = MLPRegressor::new(config);
    Ok(())
}

fn set_mlp_classifier_params(
    model: &mut MLPClassifier,
    params: &HashMap<String, String>,
) -> Result<()> {
    let mut config = crate::ml::models::neural::MLPConfig::default();
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "learning_rate" => config.learning_rate = parse_f64_param(key, value)?,
            "n_epochs" => config.n_epochs = parse_usize_param(key, value)?,
            "batch_size" => config.batch_size = parse_usize_param(key, value)?,
            "random_seed" => config.random_seed = parse_u64_param(key, value)?,
            _ => unknown.push(key.clone()),
        }
    }
    err_on_unknown_params("MLPClassifier", unknown)?;
    *model = MLPClassifier::new(config);
    Ok(())
}

/// Apply `params` to `model` by downcasting to a concrete type pandrs ships.
///
/// Returns `Ok(())` when the type is recognized and every key in `params` was applied (or
/// `params` is empty); `Err` when the type is recognized but `params` contains keys it doesn't
/// support, OR when `model`'s concrete type isn't one this function knows how to tune at all —
/// rather than silently accepting hyperparameters that go nowhere.
///
/// `RandomForestClassifier`/`GradientBoostingClassifier` are deliberately not among the
/// recognized types: they derive only `Debug` (not `Clone`) in `models/ensemble.rs`, so they can
/// never actually instantiate `SupervisedAdapter<M>` (which requires `M: Clone`) in the first
/// place — downcasting to them here would be unreachable dead code.
fn apply_model_hyperparams<M: 'static>(
    model: &mut M,
    params: &HashMap<String, String>,
) -> Result<()> {
    if params.is_empty() {
        return Ok(());
    }
    let any_model: &mut dyn Any = model;
    if let Some(m) = any_model.downcast_mut::<LinearRegression>() {
        return set_linear_regression_params(m, params);
    }
    if let Some(m) = any_model.downcast_mut::<LogisticRegression>() {
        return set_logistic_regression_params(m, params);
    }
    if let Some(m) = any_model.downcast_mut::<DecisionTreeClassifier>() {
        return set_decision_tree_classifier_params(m, params);
    }
    if let Some(m) = any_model.downcast_mut::<DecisionTreeRegressor>() {
        return set_decision_tree_regressor_params(m, params);
    }
    if let Some(m) = any_model.downcast_mut::<RandomForestRegressor>() {
        return set_random_forest_regressor_params(m, params);
    }
    if let Some(m) = any_model.downcast_mut::<GradientBoostingRegressor>() {
        return set_gradient_boosting_regressor_params(m, params);
    }
    if let Some(m) = any_model.downcast_mut::<MLPRegressor>() {
        return set_mlp_regressor_params(m, params);
    }
    if let Some(m) = any_model.downcast_mut::<MLPClassifier>() {
        return set_mlp_classifier_params(m, params);
    }
    Err(Error::InvalidValue(format!(
        "set_params: hyperparameter tuning is not implemented for model type `{}`; \
         unrecognized parameter(s): {:?}",
        std::any::type_name::<M>(),
        params.keys().collect::<Vec<_>>()
    )))
}

/// Adapter that bridges SupervisedModel (fit by target column name) into SklearnPredictor
/// (fit by separate X and Y DataFrames).
///
/// This adapter merges the X and Y DataFrames before calling `SupervisedModel::fit`, and
/// implements the full `SklearnPredictor` interface including `score`, `set_params`, and
/// `feature_importances` delegation.
#[derive(Debug, Clone)]
pub struct SupervisedAdapter<M: SupervisedModel + Clone + Send + Sync + fmt::Debug> {
    /// Wrapped model
    model: M,
    /// Name of the target column (used when merging X+Y DataFrames)
    target_col: String,
}

impl<M: SupervisedModel + Clone + Send + Sync + fmt::Debug> SupervisedAdapter<M> {
    /// Create a new adapter wrapping the given model.
    ///
    /// `target_col` is the column name that will be used for the target when the X and Y
    /// DataFrames are merged during `fit`.  It must match the column name present in the Y
    /// DataFrame passed to `SklearnPredictor::fit`.
    pub fn new(model: M, target_col: &str) -> Self {
        Self {
            model,
            target_col: target_col.to_string(),
        }
    }

    /// Compute R² score from two slices — 1 – SS_res / SS_tot.
    fn r2_score_slices(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(Error::DimensionMismatch(format!(
                "r2_score_slices: y_true has {} rows but predictions have {} rows",
                y_true.len(),
                y_pred.len()
            )));
        }
        if y_true.is_empty() {
            return Ok(0.0);
        }
        let mean_y = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_tot: f64 = y_true.iter().map(|&y| (y - mean_y).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
            .sum();
        Ok(if ss_tot == 0.0 {
            if ss_res == 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            1.0 - ss_res / ss_tot
        })
    }

    /// Merge the feature DataFrame `x` with the target DataFrame `y` into one combined
    /// DataFrame, with the target column named `self.target_col`.  The target column is
    /// added last so that `SupervisedModel::fit` can identify it by name.
    fn merge_x_y(&self, x: &DataFrame, y: &DataFrame) -> Result<DataFrame> {
        let mut merged = DataFrame::new();

        // Copy all feature columns from X
        for col_name in x.column_names() {
            let col = x.get_column::<f64>(&col_name)?;
            merged.add_column(col_name.clone(), col.clone())?;
        }

        // Locate target column in Y (prefer self.target_col name, fall back to first column)
        let y_col_name = if y.has_column(&self.target_col) {
            self.target_col.clone()
        } else {
            y.column_names()
                .iter()
                .next()
                .cloned()
                .ok_or_else(|| Error::InvalidValue("Y DataFrame has no columns".into()))?
        };

        let y_col = y.get_column::<f64>(&y_col_name)?;

        // Add target column with the canonical target_col name, unless X already contains it.
        if !merged.has_column(&self.target_col) {
            merged.add_column(self.target_col.clone(), y_col.clone())?;
        } else if !merged.has_column(&y_col_name) {
            // X already occupies self.target_col — fall back to the original y column name.
            merged.add_column(y_col_name.clone(), y_col.clone())?;
        }

        Ok(merged)
    }
}

impl<M: SupervisedModel + Clone + Send + Sync + fmt::Debug + 'static> SklearnEstimator
    for SupervisedAdapter<M>
{
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("target_col".to_string(), self.target_col.clone());
        params
    }

    /// Set hyperparameters on the wrapped model.
    ///
    /// `"target_col"` is handled here directly (it belongs to the adapter, not the wrapped
    /// model). Every other key is forwarded to `apply_model_hyperparams`, which downcasts to
    /// the wrapped model's concrete type and actually applies it — this is what makes
    /// `GridSearchCV`/`RandomizedSearchCV` tune real hyperparameters instead of fitting the same
    /// untuned model on every trial. Unknown keys return `Err`, matching scikit-learn's
    /// `set_params` (which raises on invalid parameter names) rather than silently discarding
    /// them.
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        let mut remaining: HashMap<String, String> = HashMap::new();
        for (key, value) in params {
            if key == "target_col" {
                self.target_col = value;
            } else {
                remaining.insert(key, value);
            }
        }
        apply_model_hyperparams(&mut self.model, &remaining)
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        input_features.map(|f| f.to_vec())
    }
}

impl<M: SupervisedModel + Clone + Send + Sync + fmt::Debug + 'static> SklearnPredictor
    for SupervisedAdapter<M>
{
    /// Fit the wrapped SupervisedModel by merging X and Y into a single DataFrame.
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let merged = self.merge_x_y(x, y)?;
        self.model.fit(&merged, &self.target_col)
    }

    /// Delegate predictions to the wrapped model, passing only the feature columns.
    fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        self.model.predict(x)
    }

    /// Score by computing R² (coefficient of determination) on the test set.
    fn score(&self, x: &DataFrame, y: &DataFrame) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Extract true values from y
        let y_col_name = if y.has_column(&self.target_col) {
            self.target_col.clone()
        } else {
            y.column_names()
                .iter()
                .next()
                .cloned()
                .ok_or_else(|| Error::InvalidValue("Y DataFrame has no columns".into()))?
        };
        let y_col = y.get_column::<f64>(&y_col_name)?;
        let y_true = y_col.as_f64()?;

        Self::r2_score_slices(&y_true, &predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        self.model.feature_importances()
    }

    fn clone_predictor(&self) -> Box<dyn SklearnPredictor + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Trait for all scikit-learn compatible estimators
pub trait SklearnEstimator: fmt::Debug {
    /// Get parameters of the estimator
    fn get_params(&self) -> HashMap<String, String>;

    /// Set parameters of the estimator
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()>;

    /// Get feature names output by this estimator
    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>>;
}

/// Trait for transformers that fit to data and transform it
pub trait SklearnTransformer: SklearnEstimator {
    /// Fit transformer to training data
    fn fit(&mut self, x: &DataFrame, y: Option<&DataFrame>) -> Result<()>;

    /// Transform data using fitted transformer
    fn transform(&self, x: &DataFrame) -> Result<DataFrame>;

    /// Fit to data, then transform it
    fn fit_transform(&mut self, x: &DataFrame, y: Option<&DataFrame>) -> Result<DataFrame> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Inverse transform data (if supported)
    fn inverse_transform(&self, _x: &DataFrame) -> Result<DataFrame> {
        Err(Error::NotImplemented(
            "inverse_transform not supported".into(),
        ))
    }

    /// Create a clone of this transformer for cross-validation
    fn clone_transformer(&self) -> Box<dyn SklearnTransformer + Send + Sync>;
}

/// Trait for predictors (classifiers and regressors)
pub trait SklearnPredictor: SklearnEstimator {
    /// Fit predictor to training data
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()>;

    /// Make predictions on data
    fn predict(&self, x: &DataFrame) -> Result<Vec<f64>>;

    /// Get prediction confidence scores (if supported)
    fn predict_proba(&self, _x: &DataFrame) -> Result<Vec<Vec<f64>>> {
        Err(Error::NotImplemented("predict_proba not supported".into()))
    }

    /// Score the model on test data
    fn score(&self, x: &DataFrame, y: &DataFrame) -> Result<f64>;

    /// Get feature importances (if supported by the underlying model).
    ///
    /// Returns `None` if the model type does not support feature importances, or if the
    /// model has not been fitted yet.
    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        None
    }

    /// Create a clone of this predictor for cross-validation
    fn clone_predictor(&self) -> Box<dyn SklearnPredictor + Send + Sync>;
}

/// Enhanced StandardScaler with full scikit-learn compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScalerCompat {
    /// Whether to center the data at 0
    pub with_mean: bool,
    /// Whether to scale the data to unit variance
    pub with_std: bool,
    /// Whether to remove a copy of the data
    pub copy: bool,
    /// Mean values for each feature (fitted)
    mean_: Option<HashMap<String, f64>>,
    /// Scale values for each feature (fitted)
    scale_: Option<HashMap<String, f64>>,
    /// Variance values for each feature (fitted)
    var_: Option<HashMap<String, f64>>,
    /// Number of samples seen during fit
    n_samples_seen_: Option<usize>,
    /// Feature names seen during fit
    feature_names_in_: Option<Vec<String>>,
    /// Number of features seen during fit
    n_features_in_: Option<usize>,
}

impl StandardScalerCompat {
    /// Create new StandardScaler with default parameters
    pub fn new() -> Self {
        Self {
            with_mean: true,
            with_std: true,
            copy: true,
            mean_: None,
            scale_: None,
            var_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }

    /// Create StandardScaler with custom parameters
    pub fn with_params(with_mean: bool, with_std: bool) -> Self {
        Self {
            with_mean,
            with_std,
            copy: true,
            mean_: None,
            scale_: None,
            var_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }
}

impl SklearnEstimator for StandardScalerCompat {
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("with_mean".to_string(), self.with_mean.to_string());
        params.insert("with_std".to_string(), self.with_std.to_string());
        params.insert("copy".to_string(), self.copy.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "with_mean" => {
                    self.with_mean = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid boolean value for with_mean: {}",
                            value
                        ))
                    })?
                }
                "with_std" => {
                    self.with_std = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid boolean value for with_std: {}",
                            value
                        ))
                    })?
                }
                "copy" => {
                    self.copy = value.parse().map_err(|_| {
                        Error::InvalidValue(format!("Invalid boolean value for copy: {}", value))
                    })?
                }
                _ => return Err(Error::InvalidValue(format!("Unknown parameter: {}", key))),
            }
        }
        Ok(())
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        if let Some(features) = input_features {
            Some(features.to_vec())
        } else {
            self.feature_names_in_.clone()
        }
    }
}

impl SklearnTransformer for StandardScalerCompat {
    fn fit(&mut self, x: &DataFrame, _y: Option<&DataFrame>) -> Result<()> {
        let feature_names: Vec<String> = x.column_names().to_vec();
        let n_features = feature_names.len();
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(Error::InvalidValue("Cannot fit on empty dataset".into()));
        }

        let mut means = HashMap::new();
        let mut vars = HashMap::new();
        let mut scales = HashMap::new();

        for feature_name in &feature_names {
            // Try to get numeric column
            let col = x.get_column::<f64>(feature_name)?;
            let values = col.as_f64()?;

            if values.is_empty() {
                continue;
            }

            // The true mean is always needed to compute variance correctly (Var(x) = E[(x -
            // mean)^2]), even when `with_mean` is false and centering itself is skipped at
            // transform time below. Using 0.0 here when `with_mean` is false would compute
            // E[x^2] instead of Var(x) whenever the feature isn't already zero-centered.
            let mean = values.iter().sum::<f64>() / values.len() as f64;

            // Calculate variance
            let variance = if self.with_std {
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
            } else {
                1.0
            };

            // Calculate scale (standard deviation)
            let scale = if self.with_std && variance > 1e-10 {
                variance.sqrt()
            } else {
                1.0
            };

            means.insert(feature_name.clone(), mean);
            vars.insert(feature_name.clone(), variance);
            scales.insert(feature_name.clone(), scale);
        }

        self.mean_ = Some(means);
        self.var_ = Some(vars);
        self.scale_ = Some(scales);
        self.n_samples_seen_ = Some(n_samples);
        self.feature_names_in_ = Some(feature_names);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let means = self.mean_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("StandardScaler must be fitted before transform".into())
        })?;
        let scales = self
            .scale_
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(feature_name.as_str())?;
            let values = col.as_f64()?;

            let mean = means.get(feature_name.as_str()).copied().unwrap_or(0.0);
            let scale = scales.get(feature_name.as_str()).copied().unwrap_or(1.0);

            let transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    let centered = if self.with_mean { val - mean } else { val };
                    if self.with_std && scale > 1e-10 {
                        centered / scale
                    } else {
                        centered
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(transformed_values, Some(feature_name.clone()))?,
            )?;
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let means = self.mean_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("StandardScaler must be fitted before inverse_transform".into())
        })?;
        let scales = self
            .scale_
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(feature_name.as_str())?;
            let values = col.as_f64()?;

            let mean = means.get(feature_name.as_str()).copied().unwrap_or(0.0);
            let scale = scales.get(feature_name.as_str()).copied().unwrap_or(1.0);

            let inverse_transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    let scaled = if self.with_std && scale > 1e-10 {
                        val * scale
                    } else {
                        val
                    };
                    if self.with_mean {
                        scaled + mean
                    } else {
                        scaled
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(inverse_transformed_values, Some(feature_name.clone()))?,
            )?;
        }

        Ok(result)
    }

    fn clone_transformer(&self) -> Box<dyn SklearnTransformer + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Enhanced MinMaxScaler with full scikit-learn compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinMaxScalerCompat {
    /// Desired range of transformed data
    pub feature_range: (f64, f64),
    /// Whether to remove a copy of the data
    pub copy: bool,
    /// Whether to clip transformed values to feature_range
    pub clip: bool,
    /// Minimum values for each feature (fitted)
    data_min_: Option<HashMap<String, f64>>,
    /// Maximum values for each feature (fitted)
    data_max_: Option<HashMap<String, f64>>,
    /// Range of each feature (fitted)
    data_range_: Option<HashMap<String, f64>>,
    /// Scaling factor for each feature (fitted)
    scale_: Option<HashMap<String, f64>>,
    /// Minimum bound for each feature (fitted)
    min_: Option<HashMap<String, f64>>,
    /// Number of samples seen during fit
    n_samples_seen_: Option<usize>,
    /// Feature names seen during fit
    feature_names_in_: Option<Vec<String>>,
    /// Number of features seen during fit
    n_features_in_: Option<usize>,
}

impl MinMaxScalerCompat {
    /// Create new MinMaxScaler with default range [0, 1]
    pub fn new() -> Self {
        Self {
            feature_range: (0.0, 1.0),
            copy: true,
            clip: false,
            data_min_: None,
            data_max_: None,
            data_range_: None,
            scale_: None,
            min_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }

    /// Create MinMaxScaler with custom range
    pub fn with_range(min: f64, max: f64) -> Self {
        Self {
            feature_range: (min, max),
            copy: true,
            clip: false,
            data_min_: None,
            data_max_: None,
            data_range_: None,
            scale_: None,
            min_: None,
            n_samples_seen_: None,
            feature_names_in_: None,
            n_features_in_: None,
        }
    }
}

impl SklearnEstimator for MinMaxScalerCompat {
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "feature_range_min".to_string(),
            self.feature_range.0.to_string(),
        );
        params.insert(
            "feature_range_max".to_string(),
            self.feature_range.1.to_string(),
        );
        params.insert("copy".to_string(), self.copy.to_string());
        params.insert("clip".to_string(), self.clip.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "feature_range_min" => {
                    let min_val: f64 = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid float value for feature_range_min: {}",
                            value
                        ))
                    })?;
                    self.feature_range.0 = min_val;
                }
                "feature_range_max" => {
                    let max_val: f64 = value.parse().map_err(|_| {
                        Error::InvalidValue(format!(
                            "Invalid float value for feature_range_max: {}",
                            value
                        ))
                    })?;
                    self.feature_range.1 = max_val;
                }
                "copy" => {
                    self.copy = value.parse().map_err(|_| {
                        Error::InvalidValue(format!("Invalid boolean value for copy: {}", value))
                    })?
                }
                "clip" => {
                    self.clip = value.parse().map_err(|_| {
                        Error::InvalidValue(format!("Invalid boolean value for clip: {}", value))
                    })?
                }
                _ => return Err(Error::InvalidValue(format!("Unknown parameter: {}", key))),
            }
        }
        Ok(())
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        if let Some(features) = input_features {
            Some(features.to_vec())
        } else {
            self.feature_names_in_.clone()
        }
    }
}

impl SklearnTransformer for MinMaxScalerCompat {
    fn fit(&mut self, x: &DataFrame, _y: Option<&DataFrame>) -> Result<()> {
        let feature_names: Vec<String> = x.column_names().to_vec();
        let n_features = feature_names.len();
        let n_samples = x.nrows();

        if n_samples == 0 {
            return Err(Error::InvalidValue("Cannot fit on empty dataset".into()));
        }

        let mut data_mins = HashMap::new();
        let mut data_maxs = HashMap::new();
        let mut data_ranges = HashMap::new();
        let mut scales = HashMap::new();
        let mut mins = HashMap::new();

        let (feature_min, feature_max) = self.feature_range;
        let feature_range = feature_max - feature_min;

        for feature_name in &feature_names {
            let col = x.get_column::<f64>(feature_name)?;
            let values = col.as_f64()?;

            if values.is_empty() {
                continue;
            }

            let data_min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let data_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let data_range = data_max - data_min;

            let scale = if data_range > 1e-10 {
                feature_range / data_range
            } else {
                1.0
            };

            let min = feature_min - data_min * scale;

            data_mins.insert(feature_name.clone(), data_min);
            data_maxs.insert(feature_name.clone(), data_max);
            data_ranges.insert(feature_name.clone(), data_range);
            scales.insert(feature_name.clone(), scale);
            mins.insert(feature_name.clone(), min);
        }

        self.data_min_ = Some(data_mins);
        self.data_max_ = Some(data_maxs);
        self.data_range_ = Some(data_ranges);
        self.scale_ = Some(scales);
        self.min_ = Some(mins);
        self.n_samples_seen_ = Some(n_samples);
        self.feature_names_in_ = Some(feature_names);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let scales = self.scale_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("MinMaxScaler must be fitted before transform".into())
        })?;
        let mins = self
            .min_
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(feature_name.as_str())?;
            let values = col.as_f64()?;

            let scale = scales.get(feature_name.as_str()).copied().unwrap_or(1.0);
            let min = mins.get(feature_name.as_str()).copied().unwrap_or(0.0);

            let transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    let transformed = val * scale + min;
                    if self.clip {
                        transformed
                            .max(self.feature_range.0)
                            .min(self.feature_range.1)
                    } else {
                        transformed
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(transformed_values, Some(feature_name.clone()))?,
            )?;
        }

        Ok(result)
    }

    fn inverse_transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let scales = self.scale_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("MinMaxScaler must be fitted before inverse_transform".into())
        })?;
        let mins = self
            .min_
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;

        let mut result = DataFrame::new();

        for feature_name in x.column_names() {
            let col = x.get_column::<f64>(feature_name.as_str())?;
            let values = col.as_f64()?;

            let scale = scales.get(feature_name.as_str()).copied().unwrap_or(1.0);
            let min = mins.get(feature_name.as_str()).copied().unwrap_or(0.0);

            let inverse_transformed_values: Vec<f64> = values
                .iter()
                .map(|&val| {
                    if scale > 1e-10 {
                        (val - min) / scale
                    } else {
                        val
                    }
                })
                .collect();

            result.add_column(
                feature_name.clone(),
                Series::new(inverse_transformed_values, Some(feature_name.clone()))?,
            )?;
        }

        Ok(result)
    }

    fn clone_transformer(&self) -> Box<dyn SklearnTransformer + Send + Sync> {
        Box::new(self.clone())
    }
}

/// Enhanced Pipeline with full scikit-learn compatibility
///
/// Note: unlike scikit-learn's `Pipeline`, there is deliberately no `memory` /
/// joblib-`Memory`-style caching option here. A previous version accepted and echoed back a
/// `memory` parameter but never actually cached anything — a no-op that silently did nothing
/// while looking configured. Real caching of intermediate transformer outputs would need a
/// content-addressed on-disk cache (real infrastructure, not a quick shim), so the honest
/// choice was to remove the false affordance rather than leave a setting that does nothing.
#[derive(Debug)]
pub struct Pipeline {
    /// List of pipeline steps (name, transformer/estimator)
    pub steps: Vec<(String, PipelineStep)>,
    /// Verbose output
    pub verbose: bool,
}

/// A step in the pipeline that can be either a transformer or predictor
#[derive(Debug)]
pub enum PipelineStep {
    /// A transformer step
    Transformer(Box<dyn SklearnTransformer + Send + Sync>),
    /// A predictor step (must be the last step)
    Predictor(Box<dyn SklearnPredictor + Send + Sync>),
}

impl Clone for PipelineStep {
    fn clone(&self) -> Self {
        match self {
            PipelineStep::Transformer(transformer) => {
                PipelineStep::Transformer(transformer.clone_transformer())
            }
            PipelineStep::Predictor(predictor) => {
                PipelineStep::Predictor(predictor.clone_predictor())
            }
        }
    }
}

impl Pipeline {
    /// Create a new pipeline
    pub fn new(steps: Vec<(String, PipelineStep)>) -> Self {
        Self {
            steps,
            verbose: false,
        }
    }

    /// Add a step to the pipeline
    pub fn add_step(&mut self, name: String, step: PipelineStep) {
        self.steps.push((name, step));
    }

    /// Get a step by name
    pub fn get_step(&self, name: &str) -> Option<&PipelineStep> {
        self.steps
            .iter()
            .find(|(step_name, _)| step_name == name)
            .map(|(_, step)| step)
    }

    /// Get step names
    pub fn get_step_names(&self) -> Vec<&String> {
        self.steps.iter().map(|(name, _)| name).collect()
    }

    /// Set pipeline parameters.
    ///
    /// Step-scoped keys use the `"<step_name>__<param_name>"` convention (e.g.
    /// `"scaler__with_mean"`), checked *before* any other interpretation of the key — a
    /// previous version checked `key.starts_with("memory")` first, which meant a step literally
    /// named `memory...` (e.g. `"memory_cache__enabled"`) would be misrouted into the
    /// pipeline-level memory setting instead of reaching its step. A `"__"`-bearing key whose
    /// step name doesn't exist, or a top-level key that isn't `"verbose"`, is an error rather
    /// than being silently dropped — matching scikit-learn's `set_params`, which raises on
    /// invalid parameter names.
    pub fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            if let Some(sep_idx) = key.find("__") {
                let step_name = &key[..sep_idx];
                let param_name = &key[sep_idx + 2..];

                let mut matched = false;
                for (name, step) in &mut self.steps {
                    if name == step_name {
                        let mut step_params = HashMap::new();
                        step_params.insert(param_name.to_string(), value.clone());

                        match step {
                            PipelineStep::Transformer(transformer) => {
                                transformer.set_params(step_params)?;
                            }
                            PipelineStep::Predictor(predictor) => {
                                predictor.set_params(step_params)?;
                            }
                        }
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return Err(Error::InvalidValue(format!(
                        "Pipeline::set_params: no step named '{}' (parameter '{}')",
                        step_name, key
                    )));
                }
            } else if key == "verbose" {
                self.verbose = value.parse().map_err(|_| {
                    Error::InvalidValue(format!("Invalid boolean value for verbose: {}", value))
                })?;
            } else {
                return Err(Error::InvalidValue(format!(
                    "Pipeline::set_params: unknown parameter '{}'",
                    key
                )));
            }
        }
        Ok(())
    }
}

impl SklearnEstimator for Pipeline {
    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("verbose".to_string(), self.verbose.to_string());

        // Add step-specific parameters
        for (step_name, step) in &self.steps {
            let step_params = match step {
                PipelineStep::Transformer(transformer) => transformer.get_params(),
                PipelineStep::Predictor(predictor) => predictor.get_params(),
            };

            for (param_name, param_value) in step_params {
                params.insert(format!("{}__{}", step_name, param_name), param_value);
            }
        }

        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        self.set_params(params)
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        let mut current_features = input_features.map(|f| f.to_vec());

        for (_, step) in &self.steps {
            match step {
                PipelineStep::Transformer(transformer) => {
                    current_features = transformer
                        .get_feature_names_out(current_features.as_ref().map(|f| f.as_slice()));
                }
                PipelineStep::Predictor(predictor) => {
                    current_features = predictor
                        .get_feature_names_out(current_features.as_ref().map(|f| f.as_slice()));
                }
            }
        }

        current_features
    }
}

impl SklearnPredictor for Pipeline {
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let mut current_x = x.clone();
        let steps_len = self.steps.len();

        for (i, (step_name, step)) in self.steps.iter_mut().enumerate() {
            if self.verbose {
                println!("Fitting step {}: {}", i, step_name);
            }

            match step {
                PipelineStep::Transformer(transformer) => {
                    transformer.fit(&current_x, Some(y))?;
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    // Predictor should be the last step
                    if i != steps_len - 1 {
                        return Err(Error::InvalidOperation(
                            "Predictor must be the last step in pipeline".into(),
                        ));
                    }
                    predictor.fit(&current_x, y)?;
                }
            }
        }

        Ok(())
    }

    fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        let mut current_x = x.clone();

        for (i, (step_name, step)) in self.steps.iter().enumerate() {
            if self.verbose {
                println!("Transforming step {}: {}", i, step_name);
            }

            match step {
                PipelineStep::Transformer(transformer) => {
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    return predictor.predict(&current_x);
                }
            }
        }

        Err(Error::InvalidOperation(
            "Pipeline has no predictor step".into(),
        ))
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Vec<Vec<f64>>> {
        let mut current_x = x.clone();

        for (_step_name, step) in &self.steps {
            match step {
                PipelineStep::Transformer(transformer) => {
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    return predictor.predict_proba(&current_x);
                }
            }
        }

        Err(Error::InvalidOperation(
            "Pipeline has no predictor step".into(),
        ))
    }

    fn score(&self, x: &DataFrame, y: &DataFrame) -> Result<f64> {
        let mut current_x = x.clone();

        for (_step_name, step) in &self.steps {
            match step {
                PipelineStep::Transformer(transformer) => {
                    current_x = transformer.transform(&current_x)?;
                }
                PipelineStep::Predictor(predictor) => {
                    return predictor.score(&current_x, y);
                }
            }
        }

        Err(Error::InvalidOperation(
            "Pipeline has no predictor step".into(),
        ))
    }

    fn clone_predictor(&self) -> Box<dyn SklearnPredictor + Send + Sync> {
        let cloned_steps = self
            .steps
            .iter()
            .map(|(name, step)| (name.clone(), step.clone()))
            .collect();

        Box::new(Pipeline {
            steps: cloned_steps,
            verbose: self.verbose,
        })
    }
}

/// Helper functions for creating common pipeline configurations
pub mod pipeline_builders {
    use super::*;

    /// Create a standard preprocessing pipeline
    pub fn standard_preprocessing_pipeline() -> Pipeline {
        let steps = vec![(
            "scaler".to_string(),
            PipelineStep::Transformer(Box::new(StandardScalerCompat::new())),
        )];

        Pipeline::new(steps)
    }

    /// Create a minmax preprocessing pipeline
    pub fn minmax_preprocessing_pipeline() -> Pipeline {
        let steps = vec![(
            "scaler".to_string(),
            PipelineStep::Transformer(Box::new(MinMaxScalerCompat::new())),
        )];

        Pipeline::new(steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_standard_scaler_compat() {
        let mut scaler = StandardScalerCompat::new();

        // Create test data
        let mut df = DataFrame::new();
        df.add_column(
            "feature1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");
        df.add_column(
            "feature2".to_string(),
            Series::new(
                vec![10.0, 20.0, 30.0, 40.0, 50.0],
                Some("feature2".to_string()),
            )
            .expect("operation should succeed"),
        )
        .expect("operation should succeed");

        // Fit and transform
        scaler.fit(&df, None).expect("operation should succeed");
        let transformed = scaler.transform(&df).expect("operation should succeed");

        // Check that means are approximately zero
        let feature1_col = transformed
            .get_column::<f64>("feature1")
            .expect("operation should succeed");
        let feature1_values = feature1_col.as_f64().expect("operation should succeed");
        let feature1_mean = feature1_values.iter().sum::<f64>() / feature1_values.len() as f64;

        assert!(
            (feature1_mean).abs() < 1e-10,
            "Mean should be approximately zero"
        );

        // Test inverse transform
        let inverse_transformed = scaler
            .inverse_transform(&transformed)
            .expect("operation should succeed");
        let original_feature1 = df
            .get_column::<f64>("feature1")
            .expect("operation should succeed")
            .as_f64()
            .expect("operation should succeed");
        let restored_feature1 = inverse_transformed
            .get_column::<f64>("feature1")
            .expect("operation should succeed")
            .as_f64()
            .expect("operation should succeed");

        for (original, restored) in original_feature1.iter().zip(restored_feature1.iter()) {
            assert!(
                (original - restored).abs() < 1e-10,
                "Inverse transform should restore original values"
            );
        }
    }

    #[test]
    fn test_standard_scaler_with_mean_false_uses_true_variance() {
        // Values [1,2,3,4,5]: true mean=3, true variance=Σ(x-3)²/5=2.0 (scale=sqrt(2)).
        // The previous bug forced mean=0.0 before computing variance whenever `with_mean` was
        // false, so it computed E[x²]=11.0 (scale=sqrt(11)) instead of the true variance — wrong
        // for any feature that isn't already zero-centered.
        let mut scaler = StandardScalerCompat::with_params(false, true);
        let mut df = DataFrame::new();
        df.add_column(
            "feature1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");

        scaler.fit(&df, None).expect("fit should succeed");
        let transformed = scaler.transform(&df).expect("transform should succeed");
        let values = transformed
            .get_column::<f64>("feature1")
            .expect("column should exist")
            .as_f64()
            .expect("column should be f64");

        let true_scale = 2.0f64.sqrt();
        let buggy_scale = 11.0f64.sqrt();
        assert!(
            (values[0] - 1.0 / true_scale).abs() < 1e-9,
            "with_mean=false must still divide by the TRUE standard deviation (sqrt(2) ≈ \
             {:.4}), not sqrt(E[x^2]) (sqrt(11) ≈ {:.4}); got {}",
            true_scale,
            buggy_scale,
            values[0]
        );

        // Sanity: with_mean=false means values are NOT centered (no subtraction), only scaled —
        // val=3 (the true mean) maps to 3/sqrt(2), not 0.
        assert!(
            (values[2] - 3.0 / true_scale).abs() < 1e-9,
            "with_mean=false must not subtract the mean before scaling"
        );
    }

    #[test]
    fn test_minmax_scaler_compat() {
        let mut scaler = MinMaxScalerCompat::new();

        // Create test data
        let mut df = DataFrame::new();
        df.add_column(
            "feature1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string()))
                .expect("operation should succeed"),
        )
        .expect("operation should succeed");

        // Fit and transform
        scaler.fit(&df, None).expect("operation should succeed");
        let transformed = scaler.transform(&df).expect("operation should succeed");

        // Check that values are in range [0, 1]
        let feature1_col = transformed
            .get_column::<f64>("feature1")
            .expect("operation should succeed");
        let feature1_values = feature1_col.as_f64().expect("operation should succeed");

        for &value in &feature1_values {
            assert!(
                value >= 0.0 && value <= 1.0,
                "Values should be in range [0, 1]"
            );
        }

        // Check that min is 0 and max is 1
        let min_val = feature1_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_val = feature1_values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!((min_val - 0.0).abs() < 1e-10, "Minimum should be 0");
        assert!((max_val - 1.0).abs() < 1e-10, "Maximum should be 1");
    }

    #[test]
    fn test_pipeline_parameters() {
        let pipeline = pipeline_builders::standard_preprocessing_pipeline();
        let params = pipeline.get_params();

        assert!(params.contains_key("verbose"));
        assert!(params.contains_key("scaler__with_mean"));
        assert!(params.contains_key("scaler__with_std"));
    }

    #[test]
    fn test_pipeline_set_params_step_scoped_and_errors() {
        let mut pipeline = pipeline_builders::standard_preprocessing_pipeline();

        // Step-scoped parameter actually reaches the step.
        let mut params = HashMap::new();
        params.insert("scaler__with_mean".to_string(), "false".to_string());
        pipeline
            .set_params(params)
            .expect("step-scoped set_params should succeed");
        let updated = pipeline.get_params();
        assert_eq!(
            updated.get("scaler__with_mean").map(|s| s.as_str()),
            Some("false")
        );

        // A "__"-bearing key whose step doesn't exist must error, not be silently dropped.
        let mut bad_step_params = HashMap::new();
        bad_step_params.insert("no_such_step__with_mean".to_string(), "true".to_string());
        assert!(
            pipeline.set_params(bad_step_params).is_err(),
            "set_params referencing a nonexistent step must return Err"
        );

        // An unknown top-level (non-"__", non-"verbose") key must also error — including the
        // removed "memory" setting, which used to be silently accepted and echoed back despite
        // never caching anything.
        let mut unknown_top_level = HashMap::new();
        unknown_top_level.insert("memory".to_string(), "/tmp/cache".to_string());
        assert!(
            pipeline.set_params(unknown_top_level).is_err(),
            "unknown top-level parameters (including the removed 'memory' option) must error"
        );

        // A step literally prefixed "memory" must still route through step-scoped handling
        // rather than being misrouted into a pipeline-level "memory" setting (the root cause of
        // the original bug: `key.starts_with(\"memory\")` was checked before the \"__\" split).
        pipeline.add_step(
            "memory_cache".to_string(),
            PipelineStep::Transformer(Box::new(StandardScalerCompat::new())),
        );
        let mut step_like_memory = HashMap::new();
        step_like_memory.insert("memory_cache__with_std".to_string(), "false".to_string());
        pipeline
            .set_params(step_like_memory)
            .expect("a step named 'memory_cache' must be reachable via '__' routing");
    }

    #[test]
    fn test_supervised_adapter_set_params_rejects_unknown_key() {
        let mut adapter: Box<dyn SklearnPredictor + Send + Sync> =
            Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));
        let mut params = HashMap::new();
        params.insert("not_a_real_hyperparameter".to_string(), "1".to_string());
        assert!(
            adapter.set_params(params).is_err(),
            "set_params must error on an unrecognized key instead of silently discarding it"
        );

        // Recognized keys still succeed.
        let mut good_params = HashMap::new();
        good_params.insert("fit_intercept".to_string(), "false".to_string());
        good_params.insert("normalize".to_string(), "true".to_string());
        assert!(adapter.set_params(good_params).is_ok());
    }
}
