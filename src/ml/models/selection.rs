//! Model selection utilities
//!
//! This module provides tools for model selection, including grid search and
//! randomized search for hyperparameter optimization.
//!
//! Both searches are *real*: every candidate parameter combination is applied to a
//! fresh clone of the base model (via [`TunableModel::set_params`]), cross-validated
//! on its own, and scored independently. The reported `best_params` / `best_score`
//! are the genuine arg-best and best of those per-combination scores, and
//! `cv_results` carries one row per combination with its own mean/std/per-fold
//! scores and rank.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::{ModelMetrics, SupervisedModel};
use crate::series::Series;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TunableModel
// ---------------------------------------------------------------------------

/// A model whose hyperparameters can be set from a `name -> value` string map.
///
/// This is the contract that makes a hyperparameter search *search anything at all*:
/// the search clones the base model once per candidate combination and calls
/// [`set_params`](TunableModel::set_params) on the clone before cross-validating it.
///
/// # Implementor contract
///
/// * Every key in `params` that the model recognises must be applied to `self`,
///   **on top of the model's existing configuration** (keys absent from `params`
///   keep the base model's value — a search over `learning_rate` must not reset
///   `hidden_layers` to a default).
/// * Any key the model does not recognise, and any value it cannot parse, must
///   return `Err`. Silently ignoring an unknown key makes a search report scores
///   for a configuration it never actually evaluated.
/// * An empty `params` map is a no-op that returns `Ok(())`.
///
/// pandrs implements this for [`crate::ml::models::linear::LinearRegression`],
/// [`crate::ml::models::linear::LogisticRegression`],
/// [`crate::ml::models::neural::MLPRegressor`] and
/// [`crate::ml::models::neural::MLPClassifier`]. Model types that store their
/// configuration privately without setters (the tree and ensemble models) do not
/// implement it yet, so they cannot be passed to [`GridSearchCV::fit`] — that is a
/// compile-time error rather than a search that silently tunes nothing. Those
/// estimators can be tuned today through
/// [`crate::ml::model_selection::GridSearchCV`], which drives them through
/// [`crate::ml::sklearn_compat::SupervisedAdapter`].
pub trait TunableModel: SupervisedModel + Clone {
    /// Apply one hyperparameter combination to this model instance.
    ///
    /// Returns `Err` for unknown parameter names and unparsable values.
    fn set_params(&mut self, params: &HashMap<String, String>) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Parameter parsing helpers shared by the `TunableModel` implementations
// ---------------------------------------------------------------------------

/// Parse a boolean hyperparameter value, reporting the offending key on failure.
pub(crate) fn parse_param_bool(key: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        _ => Err(Error::InvalidValue(format!(
            "Invalid boolean value for hyperparameter '{}': '{}'",
            key, value
        ))),
    }
}

/// Parse a floating-point hyperparameter value (accepts `inf` / `infinity`).
pub(crate) fn parse_param_f64(key: &str, value: &str) -> Result<f64> {
    value.trim().parse::<f64>().map_err(|_| {
        Error::InvalidValue(format!(
            "Invalid floating-point value for hyperparameter '{}': '{}'",
            key, value
        ))
    })
}

/// Parse an unsigned-integer hyperparameter value.
pub(crate) fn parse_param_usize(key: &str, value: &str) -> Result<usize> {
    value.trim().parse::<usize>().map_err(|_| {
        Error::InvalidValue(format!(
            "Invalid unsigned integer value for hyperparameter '{}': '{}'",
            key, value
        ))
    })
}

/// Parse a `u64` hyperparameter value (random seeds).
pub(crate) fn parse_param_u64(key: &str, value: &str) -> Result<u64> {
    value.trim().parse::<u64>().map_err(|_| {
        Error::InvalidValue(format!(
            "Invalid u64 value for hyperparameter '{}': '{}'",
            key, value
        ))
    })
}

/// Turn a collected list of unrecognised parameter names into an error.
///
/// Returns `Ok(())` only when nothing was left over: a search must never report a
/// score for a configuration whose parameters were silently dropped.
pub(crate) fn err_on_unknown_params(model_type: &str, unknown: Vec<String>) -> Result<()> {
    if unknown.is_empty() {
        Ok(())
    } else {
        let mut unknown = unknown;
        unknown.sort();
        Err(Error::InvalidValue(format!(
            "set_params: unknown hyperparameter(s) {:?} for {}",
            unknown, model_type
        )))
    }
}

// ---------------------------------------------------------------------------
// Hyperparameter grid
// ---------------------------------------------------------------------------

/// A grid of hyperparameters for model selection
#[derive(Debug, Clone)]
pub struct HyperparameterGrid {
    /// Map of parameter names to possible values
    pub params: HashMap<String, Vec<String>>,
}

impl HyperparameterGrid {
    /// Create a new empty hyperparameter grid
    pub fn new() -> Self {
        HyperparameterGrid {
            params: HashMap::new(),
        }
    }

    /// Add a parameter with its possible values
    pub fn add_param<T: ToString>(&mut self, name: &str, values: Vec<T>) -> &mut Self {
        let string_values = values.into_iter().map(|v| v.to_string()).collect();
        self.params.insert(name.to_string(), string_values);
        self
    }

    /// Get all parameter combinations as a full Cartesian product.
    ///
    /// Keys are sorted for deterministic, reproducible ordering across runs.
    ///
    /// # Returns
    /// * Vector of parameter dictionaries, where each dictionary is one combination.
    pub fn parameter_combinations(&self) -> Vec<HashMap<String, String>> {
        if self.params.is_empty() {
            return vec![HashMap::new()];
        }

        // Sort keys for deterministic output
        let keys: Vec<String> = {
            let mut k: Vec<String> = self.params.keys().cloned().collect();
            k.sort();
            k
        };

        let mut result: Vec<HashMap<String, String>> = vec![HashMap::new()];

        for key in &keys {
            let values = &self.params[key];
            let mut new_result = Vec::with_capacity(result.len() * values.len());
            for existing in &result {
                for value in values {
                    let mut combo = existing.clone();
                    combo.insert(key.clone(), value.clone());
                    new_result.push(combo);
                }
            }
            result = new_result;
        }

        result
    }

    /// Sorted list of the parameter names in this grid.
    fn sorted_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.params.keys().cloned().collect();
        keys.sort();
        keys
    }
}

impl Default for HyperparameterGrid {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Shared search machinery
// ---------------------------------------------------------------------------

/// Result of cross-validating one parameter combination.
#[derive(Debug, Clone)]
struct CombinationOutcome {
    /// The parameter combination that produced these scores
    params: HashMap<String, String>,
    /// Per-fold scores for the requested metric
    fold_scores: Vec<f64>,
    /// Mean of `fold_scores`
    mean_score: f64,
    /// Population standard deviation of `fold_scores`
    std_score: f64,
}

/// Whether a larger value of `metric` denotes a better model.
///
/// Returns `Err` for metrics whose optimisation direction this module does not know:
/// assuming "higher is better" for an error metric would report the *worst*
/// configuration in the grid as the best one.
fn metric_higher_is_better(metric: &str) -> Result<bool> {
    match metric.trim().to_ascii_lowercase().as_str() {
        "r2" | "r2_score" | "accuracy" | "precision" | "recall" | "f1" | "auc" | "roc_auc"
        | "explained_variance" => Ok(true),
        "mse" | "rmse" | "mae" | "mape" | "msle" | "log_loss" | "error" => Ok(false),
        other => Err(Error::InvalidInput(format!(
            "Unknown scoring metric '{}': cannot determine whether higher or lower is better. \
             Supported metrics are r2, accuracy, precision, recall, f1, auc/roc_auc, \
             explained_variance (higher is better) and mse, rmse, mae, mape, msle, log_loss \
             (lower is better)",
            other
        ))),
    }
}

/// Render a parameter combination deterministically for error messages.
fn sorted_pairs(params: &HashMap<String, String>) -> Vec<String> {
    let mut pairs: Vec<String> = params
        .iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect();
    pairs.sort();
    pairs
}

/// Extract the requested `scoring` metric from one fold's metrics.
fn fold_score(metrics: &ModelMetrics, scoring: &str, fold_idx: usize) -> Result<f64> {
    let value = metrics.get_metric(scoring).copied().ok_or_else(|| {
        let mut available: Vec<&String> = metrics.metrics.keys().collect();
        available.sort();
        Error::InvalidInput(format!(
            "Scoring metric '{}' was not reported by the model for CV fold {}; \
             metrics reported by this model: {:?}",
            scoring, fold_idx, available
        ))
    })?;

    if !value.is_finite() {
        return Err(Error::Computation(format!(
            "Scoring metric '{}' was {} on CV fold {}; a non-finite score cannot be \
             compared against other parameter combinations",
            scoring, value, fold_idx
        )));
    }

    Ok(value)
}

/// Apply one parameter combination to a clone of `base_model` and cross-validate it.
///
/// Every step here can fail loudly: unknown hyperparameters (`set_params`), a model
/// whose `cross_validate` does not actually run the folds, and folds that do not
/// report the requested metric all return `Err` instead of contributing a
/// placeholder score to the search. A failing combination therefore aborts the whole
/// search — deliberately stricter than scikit-learn, which by default records a failed
/// fit as `NaN` (`error_score`) and warns. A combination that cannot be evaluated is a
/// hole in the search, and reporting the remaining winner as "best" would hide it. The
/// error names the offending combination so the caller can drop or fix it.
fn evaluate_combination<T: TunableModel>(
    base_model: &T,
    params: &HashMap<String, String>,
    data: &DataFrame,
    target: &str,
    cv: usize,
    scoring: &str,
) -> Result<CombinationOutcome> {
    let mut candidate = base_model.clone();
    candidate.set_params(params).map_err(|e| {
        Error::InvalidValue(format!(
            "parameter combination {:?} could not be applied to the model: {}",
            sorted_pairs(params),
            e
        ))
    })?;

    let fold_metrics = candidate.cross_validate(data, target, cv).map_err(|e| {
        Error::Computation(format!(
            "parameter combination {:?} failed cross-validation: {}",
            sorted_pairs(params),
            e
        ))
    })?;

    if fold_metrics.len() != cv {
        return Err(Error::InvalidOperation(format!(
            "parameter combination {:?}: cross_validate returned {} fold result(s) for a \
             {}-fold cross-validation; the model's cross_validate implementation must \
             produce exactly one ModelMetrics per fold",
            sorted_pairs(params),
            fold_metrics.len(),
            cv
        )));
    }

    let mut fold_scores = Vec::with_capacity(fold_metrics.len());
    for (fold_idx, metrics) in fold_metrics.iter().enumerate() {
        fold_scores.push(fold_score(metrics, scoring, fold_idx).map_err(|e| {
            Error::InvalidInput(format!(
                "parameter combination {:?}: {}",
                sorted_pairs(params),
                e
            ))
        })?);
    }

    let n = fold_scores.len() as f64;
    let mean_score = fold_scores.iter().sum::<f64>() / n;
    let variance = fold_scores
        .iter()
        .map(|&s| (s - mean_score).powi(2))
        .sum::<f64>()
        / n;

    Ok(CombinationOutcome {
        params: params.clone(),
        fold_scores,
        mean_score,
        std_score: variance.sqrt(),
    })
}

/// Evaluate every candidate combination and return the outcomes in candidate order.
fn evaluate_all<T: TunableModel>(
    base_model: &T,
    combinations: &[HashMap<String, String>],
    data: &DataFrame,
    target: &str,
    cv: usize,
    scoring: &str,
) -> Result<Vec<CombinationOutcome>> {
    // Validate the metric direction once, before spending any CV time.
    metric_higher_is_better(scoring)?;

    let mut outcomes = Vec::with_capacity(combinations.len());
    for params in combinations {
        outcomes.push(evaluate_combination(
            base_model, params, data, target, cv, scoring,
        )?);
    }
    Ok(outcomes)
}

/// Index of the best outcome, honouring the metric's optimisation direction.
///
/// Ties are broken by candidate order, which is deterministic (the grid's Cartesian
/// product visits sorted parameter names).
fn best_outcome_index(outcomes: &[CombinationOutcome], scoring: &str) -> Result<usize> {
    let higher_is_better = metric_higher_is_better(scoring)?;

    let mut best_idx = 0usize;
    let mut best = outcomes
        .first()
        .ok_or_else(|| Error::InvalidInput("No parameter combination was evaluated".into()))?
        .mean_score;

    for (idx, outcome) in outcomes.iter().enumerate().skip(1) {
        let better = if higher_is_better {
            outcome.mean_score > best
        } else {
            outcome.mean_score < best
        };
        if better {
            best = outcome.mean_score;
            best_idx = idx;
        }
    }

    Ok(best_idx)
}

/// Competition ranking (rank 1 = best, ties share the smaller rank) of every outcome.
fn rank_outcomes(outcomes: &[CombinationOutcome], scoring: &str) -> Result<Vec<i64>> {
    let higher_is_better = metric_higher_is_better(scoring)?;

    let mut ranks = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        let better_count = outcomes
            .iter()
            .filter(|other| {
                if higher_is_better {
                    other.mean_score > outcome.mean_score
                } else {
                    other.mean_score < outcome.mean_score
                }
            })
            .count();
        ranks.push(better_count as i64 + 1);
    }
    Ok(ranks)
}

/// Build the `cv_results` DataFrame: one row per evaluated combination.
///
/// Columns: `param_<name>` for every searched hyperparameter, `split{i}_test_score`
/// for every fold, plus `mean_test_score`, `std_test_score` and `rank_test_score`.
fn build_cv_results(
    outcomes: &[CombinationOutcome],
    param_names: &[String],
    scoring: &str,
) -> Result<DataFrame> {
    let mut df = DataFrame::new();

    for name in param_names {
        let column: Vec<String> = outcomes
            .iter()
            .map(|o| {
                o.params.get(name).cloned().ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "cv_results: evaluated combination {:?} is missing searched \
                         parameter '{}'",
                        o.params, name
                    ))
                })
            })
            .collect::<Result<Vec<String>>>()?;
        let col_name = format!("param_{}", name);
        df.add_column(
            col_name.clone(),
            Series::new(column, Some(col_name.clone()))?,
        )?;
    }

    let n_folds = outcomes.first().map(|o| o.fold_scores.len()).unwrap_or(0);
    for fold_idx in 0..n_folds {
        let column: Vec<f64> = outcomes
            .iter()
            .map(|o| o.fold_scores.get(fold_idx).copied().unwrap_or(f64::NAN))
            .collect();
        let col_name = format!("split{}_test_score", fold_idx);
        df.add_column(
            col_name.clone(),
            Series::new(column, Some(col_name.clone()))?,
        )?;
    }

    let means: Vec<f64> = outcomes.iter().map(|o| o.mean_score).collect();
    df.add_column(
        "mean_test_score".to_string(),
        Series::new(means, Some("mean_test_score".to_string()))?,
    )?;

    let stds: Vec<f64> = outcomes.iter().map(|o| o.std_score).collect();
    df.add_column(
        "std_test_score".to_string(),
        Series::new(stds, Some("std_test_score".to_string()))?,
    )?;

    let ranks = rank_outcomes(outcomes, scoring)?;
    df.add_column(
        "rank_test_score".to_string(),
        Series::new(ranks, Some("rank_test_score".to_string()))?,
    )?;

    Ok(df)
}

// ---------------------------------------------------------------------------
// GridSearchCV
// ---------------------------------------------------------------------------

/// Grid search for hyperparameter optimization
///
/// Exhaustively evaluates *every* parameter combination in the grid: each combination
/// is applied to a fresh clone of the base model with [`TunableModel::set_params`] and
/// scored by its own k-fold cross-validation run. `best_params` is the combination with
/// the best mean CV score (respecting the metric's optimisation direction), and
/// `cv_results` holds one row per combination.
///
/// Parameter combinations are evaluated serially, in the deterministic order produced
/// by [`HyperparameterGrid::parameter_combinations`]. (An `n_jobs` knob used to be
/// accepted here and never read; it was removed rather than left as configuration that
/// does nothing.)
pub struct GridSearchCV<T: SupervisedModel> {
    /// Base model to tune
    pub base_model: T,
    /// Parameter grid to search
    pub param_grid: HyperparameterGrid,
    /// Scoring metric to optimize
    pub scoring: String,
    /// Number of cross-validation folds
    pub cv: usize,
    /// Whether to refit the best configuration on the full dataset during `fit`
    pub refit: bool,
    /// Best parameters found
    pub best_params: Option<HashMap<String, String>>,
    /// Best score found
    pub best_score: Option<f64>,
    /// All results from the search
    pub cv_results: Option<DataFrame>,
    /// Best estimator: the base model with `best_params` applied, refit on the full
    /// dataset when `refit` is enabled (otherwise tuned but left unfitted).
    pub best_estimator_: Option<T>,
}

impl<T: SupervisedModel + Clone> GridSearchCV<T> {
    /// Create a new GridSearchCV instance
    ///
    /// # Arguments
    /// * `base_model` - The model to tune
    /// * `param_grid` - Grid of hyperparameters to search
    /// * `scoring` - Metric to optimize (e.g. "r2", "mse")
    /// * `cv` - Number of cross-validation folds (must be >= 2)
    pub fn new(base_model: T, param_grid: HyperparameterGrid, scoring: &str, cv: usize) -> Self {
        GridSearchCV {
            base_model,
            param_grid,
            scoring: scoring.to_string(),
            cv,
            refit: true,
            best_params: None,
            best_score: None,
            cv_results: None,
            best_estimator_: None,
        }
    }

    /// Set whether the best configuration is refit on the full dataset by `fit`.
    pub fn with_refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }

    /// Get the best estimator found by the search.
    ///
    /// The returned model has `best_params` applied and — unless `refit` was disabled —
    /// is already fitted on the full dataset passed to [`GridSearchCV::fit`].
    pub fn best_estimator(&self) -> Result<T> {
        self.best_estimator_
            .clone()
            .ok_or_else(|| Error::InvalidValue("Grid search not fitted".into()))
    }
}

impl<T: TunableModel> GridSearchCV<T> {
    /// Run the search: cross-validate every parameter combination and keep the best.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `target` - Target column name
    ///
    /// # Errors
    /// Returns `Err` when the target column is missing, `cv < 2`, a parameter cannot be
    /// applied to the model, a model's `cross_validate` does not produce one result per
    /// fold, or a fold does not report the requested scoring metric.
    pub fn fit(&mut self, data: &DataFrame, target: &str) -> Result<()> {
        if !data.has_column(target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target
            )));
        }

        if self.cv < 2 {
            return Err(Error::InvalidInput(
                "Number of CV folds must be at least 2".into(),
            ));
        }

        let param_combinations = self.param_grid.parameter_combinations();
        if param_combinations.is_empty() {
            return Err(Error::InvalidInput(
                "No parameter combinations to search".into(),
            ));
        }

        let outcomes = evaluate_all(
            &self.base_model,
            &param_combinations,
            data,
            target,
            self.cv,
            &self.scoring,
        )?;

        let best_idx = best_outcome_index(&outcomes, &self.scoring)?;
        let best = &outcomes[best_idx];

        self.best_params = Some(best.params.clone());
        self.best_score = Some(best.mean_score);
        self.cv_results = Some(build_cv_results(
            &outcomes,
            &self.param_grid.sorted_keys(),
            &self.scoring,
        )?);

        let mut best_model = self.base_model.clone();
        best_model.set_params(&best.params)?;
        if self.refit {
            best_model.fit(data, target)?;
        }
        self.best_estimator_ = Some(best_model);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RandomizedSearchCV
// ---------------------------------------------------------------------------

/// Randomized search for hyperparameter optimization
///
/// Samples `n_iter` distinct parameter combinations from the grid (without replacement,
/// using a seeded RNG) and cross-validates *each* of them independently, exactly like
/// [`GridSearchCV`] does for the full grid.
pub struct RandomizedSearchCV<T: SupervisedModel> {
    /// Base model to tune
    pub base_model: T,
    /// Parameter grid to sample from
    pub param_grid: HyperparameterGrid,
    /// Number of parameter combinations to try
    pub n_iter: usize,
    /// Scoring metric to optimize
    pub scoring: String,
    /// Number of cross-validation folds
    pub cv: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to refit the best configuration on the full dataset during `fit`
    pub refit: bool,
    /// Best parameters found
    pub best_params: Option<HashMap<String, String>>,
    /// Best score found
    pub best_score: Option<f64>,
    /// All results from the search
    pub cv_results: Option<DataFrame>,
    /// Best estimator: the base model with `best_params` applied, refit on the full
    /// dataset when `refit` is enabled (otherwise tuned but left unfitted).
    pub best_estimator_: Option<T>,
}

impl<T: SupervisedModel + Clone> RandomizedSearchCV<T> {
    /// Create a new RandomizedSearchCV instance
    ///
    /// # Arguments
    /// * `base_model` - The model to tune
    /// * `param_grid` - Grid of hyperparameters to sample from
    /// * `n_iter` - Number of parameter combinations to try
    /// * `scoring` - Metric to optimize (e.g. "r2", "mse")
    /// * `cv` - Number of cross-validation folds (must be >= 2)
    pub fn new(
        base_model: T,
        param_grid: HyperparameterGrid,
        n_iter: usize,
        scoring: &str,
        cv: usize,
    ) -> Self {
        RandomizedSearchCV {
            base_model,
            param_grid,
            n_iter,
            scoring: scoring.to_string(),
            cv,
            random_seed: None,
            refit: true,
            best_params: None,
            best_score: None,
            cv_results: None,
            best_estimator_: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set whether the best configuration is refit on the full dataset by `fit`.
    pub fn with_refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }

    /// Get the best estimator found by the search.
    ///
    /// The returned model has `best_params` applied and — unless `refit` was disabled —
    /// is already fitted on the full dataset passed to [`RandomizedSearchCV::fit`].
    pub fn best_estimator(&self) -> Result<T> {
        self.best_estimator_
            .clone()
            .ok_or_else(|| Error::InvalidValue("Randomized search not fitted".into()))
    }

    /// Sample the candidate combinations to evaluate (without replacement).
    fn sample_combinations(&self) -> Vec<HashMap<String, String>> {
        let all_combinations = self.param_grid.parameter_combinations();
        let n_to_try = self.n_iter.min(all_combinations.len());

        if n_to_try >= all_combinations.len() {
            return all_combinations;
        }

        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        use scirs2_core::random::SliceRandom;

        let mut rng: StdRng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(scirs2_core::random::random::<u64>()),
        };

        let mut indices: Vec<usize> = (0..all_combinations.len()).collect();
        indices.shuffle(&mut rng);
        let mut selected: Vec<usize> = indices[..n_to_try].to_vec();
        // Keep grid order among the sampled combinations so `cv_results` rows and
        // score ties are reported in the same deterministic order as GridSearchCV.
        selected.sort_unstable();
        selected
            .into_iter()
            .map(|i| all_combinations[i].clone())
            .collect()
    }
}

impl<T: TunableModel> RandomizedSearchCV<T> {
    /// Run the search: sample `n_iter` combinations, cross-validate each, keep the best.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `target` - Target column name
    pub fn fit(&mut self, data: &DataFrame, target: &str) -> Result<()> {
        if !data.has_column(target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target
            )));
        }

        if self.cv < 2 {
            return Err(Error::InvalidInput(
                "Number of CV folds must be at least 2".into(),
            ));
        }

        if self.n_iter == 0 {
            return Err(Error::InvalidInput(
                "n_iter must be at least 1: a randomized search that samples no \
                 parameter combination cannot report a best configuration"
                    .into(),
            ));
        }

        let selected_combos = self.sample_combinations();
        if selected_combos.is_empty() {
            return Err(Error::InvalidInput(
                "No parameter combinations to search".into(),
            ));
        }

        let outcomes = evaluate_all(
            &self.base_model,
            &selected_combos,
            data,
            target,
            self.cv,
            &self.scoring,
        )?;

        let best_idx = best_outcome_index(&outcomes, &self.scoring)?;
        let best = &outcomes[best_idx];

        self.best_params = Some(best.params.clone());
        self.best_score = Some(best.mean_score);
        self.cv_results = Some(build_cv_results(
            &outcomes,
            &self.param_grid.sorted_keys(),
            &self.scoring,
        )?);

        let mut best_model = self.base_model.clone();
        best_model.set_params(&best.params)?;
        if self.refit {
            best_model.fit(data, target)?;
        }
        self.best_estimator_ = Some(best_model);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::ml::models::linear::LinearRegression;
    use crate::series::Series;

    /// Build a simple y = 2x + 1 dataset with `n` rows.
    fn make_linear_df(n: usize) -> DataFrame {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(x, Some("x".to_string())).expect("Series::new"),
        )
        .expect("add x");
        df.add_column(
            "y".to_string(),
            Series::new(y, Some("y".to_string())).expect("Series::new"),
        )
        .expect("add y");
        df
    }

    #[test]
    fn test_cartesian_product() {
        let mut grid = HyperparameterGrid::new();
        grid.add_param("a", vec!["1", "2"]);
        grid.add_param("b", vec!["x", "y"]);
        let combos = grid.parameter_combinations();
        assert_eq!(
            combos.len(),
            4,
            "2x2 Cartesian product must yield exactly 4 combinations"
        );
        for combo in &combos {
            assert!(combo.contains_key("a"), "combo missing key 'a'");
            assert!(combo.contains_key("b"), "combo missing key 'b'");
        }
    }

    #[test]
    fn test_cartesian_empty() {
        let grid = HyperparameterGrid::new();
        let combos = grid.parameter_combinations();
        assert_eq!(
            combos.len(),
            1,
            "empty grid must return exactly one (empty) combination"
        );
        assert!(combos[0].is_empty(), "the single combination must be empty");
    }

    #[test]
    fn test_gridsearch_cv_real() {
        let df = make_linear_df(10);
        let model = LinearRegression::new();
        let grid = HyperparameterGrid::new(); // empty grid -> one combo
        let mut gs = GridSearchCV::new(model, grid, "r2", 2);
        gs.fit(&df, "y").expect("GridSearchCV::fit should succeed");

        let best_score = gs.best_score.expect("best_score must be set after fit");
        // LinearRegression on y=2x+1 gives near-perfect R², well above 0.
        assert!(
            best_score > 0.0,
            "best_score must be a real positive CV score, got {}",
            best_score
        );
        assert!(gs.cv_results.is_some(), "cv_results must be set after fit");
    }

    #[test]
    fn test_gridsearch_scores_every_combination() {
        let df = make_linear_df(12);
        let mut grid = HyperparameterGrid::new();
        grid.add_param("fit_intercept", vec!["true", "false"]);

        let mut gs = GridSearchCV::new(LinearRegression::new(), grid, "r2", 3);
        gs.fit(&df, "y").expect("GridSearchCV::fit should succeed");

        let results = gs.cv_results.as_ref().expect("cv_results");
        assert_eq!(results.nrows(), 2, "one row per parameter combination");

        let scores = results
            .get_column::<f64>("mean_test_score")
            .expect("mean_test_score column")
            .values()
            .to_vec();
        assert!(
            (scores[0] - scores[1]).abs() > 1e-9,
            "each combination must be scored on its own; got identical scores {:?}",
            scores
        );

        // y = 2x + 1 needs an intercept: fit_intercept=true must win.
        let best = gs.best_params.as_ref().expect("best_params");
        assert_eq!(
            best.get("fit_intercept").map(String::as_str),
            Some("true"),
            "the genuinely best combination must be reported as best"
        );
    }

    #[test]
    fn test_unknown_scoring_metric_errors() {
        let df = make_linear_df(10);
        let mut gs = GridSearchCV::new(
            LinearRegression::new(),
            HyperparameterGrid::new(),
            "not_a_metric",
            2,
        );
        assert!(
            gs.fit(&df, "y").is_err(),
            "an unknown scoring metric must be rejected, not silently ranked"
        );
    }

    #[test]
    fn test_unknown_parameter_errors() {
        let df = make_linear_df(10);
        let mut grid = HyperparameterGrid::new();
        grid.add_param("no_such_param", vec!["1", "2"]);
        let mut gs = GridSearchCV::new(LinearRegression::new(), grid, "r2", 2);
        assert!(
            gs.fit(&df, "y").is_err(),
            "a parameter the model cannot accept must error, not be skipped"
        );
    }

    #[test]
    fn test_randomized_search_samples_and_scores() {
        let df = make_linear_df(12);
        let mut grid = HyperparameterGrid::new();
        grid.add_param("fit_intercept", vec!["true", "false"]);

        let mut rs =
            RandomizedSearchCV::new(LinearRegression::new(), grid, 2, "r2", 3).with_random_seed(42);
        rs.fit(&df, "y")
            .expect("RandomizedSearchCV::fit should succeed");

        let results = rs.cv_results.as_ref().expect("cv_results");
        assert_eq!(results.nrows(), 2, "one row per sampled combination");
        assert_eq!(
            rs.best_params
                .as_ref()
                .and_then(|p| p.get("fit_intercept"))
                .map(String::as_str),
            Some("true")
        );
    }
}
