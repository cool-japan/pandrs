//! Automated Machine Learning (AutoML) capabilities
//!
//! This module provides comprehensive AutoML functionality including automated
//! model selection, hyperparameter optimization, feature engineering, and
//! ensemble methods for both regression and classification tasks.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::feature_engineering::{AutoFeatureEngineer, FeatureSelectionMethod, ScalingMethod};
use crate::ml::model_selection::{
    CrossValidationStrategy, ParameterDistribution, RandomizedSearchCV, ScoreFunction, Scorer,
};
use crate::ml::models::ensemble::{
    GradientBoostingConfig, GradientBoostingRegressor, RandomForestConfig, RandomForestRegressor,
};
use crate::ml::models::linear::{LinearRegression, LogisticRegression};
use crate::ml::models::train_test_split;
use crate::ml::models::tree::{DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeRegressor};
use crate::ml::sklearn_compat::{SklearnEstimator, SklearnPredictor, SupervisedAdapter};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// AutoML task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Regression task
    Regression,
    /// Binary classification task
    BinaryClassification,
    /// Multi-class classification task
    MultiClassification,
    /// Time series forecasting
    TimeSeries,
    /// Auto-detect task type from target variable
    Auto,
}

/// AutoML configuration
#[derive(Debug, Clone)]
pub struct AutoMLConfig {
    /// Type of machine learning task
    pub task_type: TaskType,
    /// Wall-clock time budget for the whole model search, in seconds. Checked between models in
    /// `AutoML::search_models`: once elapsed time exceeds this, the search stops early and
    /// returns whatever models have already been evaluated instead of silently ignoring the
    /// budget.
    pub time_limit: Option<f64>,
    /// Maximum number of models to try
    pub max_models: Option<usize>,
    /// Maximum number of engineered features to keep when `feature_engineering` is enabled.
    /// Kept separate from `max_models` (which bounds the number of *models* tried) — a previous
    /// version reused `max_models` as the feature-selection `k`, so raising the model budget
    /// silently also widened (or narrowed) the feature count.
    pub max_selected_features: Option<usize>,
    /// Cross-validation strategy
    pub cv_strategy: CrossValidationStrategy,
    /// Scoring metric for optimization
    pub scoring: Scorer,
    /// Whether to perform feature engineering
    pub feature_engineering: bool,
    /// Whether to perform feature selection (only consulted when `feature_engineering` is also
    /// enabled): when `false`, the per-fold feature engineer generates/scales features but keeps
    /// all of them rather than running `SelectKBest`.
    pub feature_selection: bool,
    /// Whether to add a simple averaging ensemble of the top-performing distinct models as an
    /// additional leaderboard candidate.
    pub ensemble_methods: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Verbose output level
    pub verbose: usize,
    /// When `true`, the final model is chosen by cross-validation score minus a small penalty
    /// proportional to [`ModelResult::complexity_score`], so a simpler model can win over a
    /// marginally-better-scoring but much more complex one.
    pub optimize_for_interpretability: bool,
    /// Soft memory limit (in GB) for the training matrix. Enforced as a static preflight check
    /// on the raw input matrix's estimated size (`rows * cols * 8 bytes`) before search starts —
    /// not live RSS monitoring during training, which would need OS-level instrumentation this
    /// crate does not have. `feature_engineering` can still grow the matrix well past this
    /// estimate; this check only catches the input already being oversized.
    pub memory_limit: Option<f64>,
    /// Custom model list to try (if None, use default model space)
    pub model_whitelist: Option<Vec<String>>,
    /// Models to exclude from search
    pub model_blacklist: Option<Vec<String>>,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::Auto,
            time_limit: Some(3600.0), // 1 hour
            max_models: Some(50),
            max_selected_features: Some(50),
            cv_strategy: CrossValidationStrategy::KFold {
                n_splits: 5,
                shuffle: true,
                random_state: None,
            },
            scoring: Scorer::R2,
            feature_engineering: true,
            feature_selection: true,
            ensemble_methods: true,
            random_state: None,
            verbose: 1,
            optimize_for_interpretability: false,
            memory_limit: Some(8.0),
            model_whitelist: None,
            model_blacklist: None,
        }
    }
}

/// Model search space for AutoML
#[derive(Debug, Clone)]
pub struct ModelSearchSpace {
    /// Linear models and their parameter spaces
    pub linear_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
    /// Tree-based models and their parameter spaces
    pub tree_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
    /// Ensemble models and their parameter spaces
    pub ensemble_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
    /// Neural network models and their parameter spaces
    pub neural_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
}

impl ModelSearchSpace {
    /// Create default model search space for regression
    pub fn default_regression() -> Self {
        let mut linear_models = Vec::new();
        let mut tree_models = Vec::new();
        let mut ensemble_models = Vec::new();
        let neural_models = Vec::new();

        // Linear models
        let mut linear_regression_params = HashMap::new();
        linear_regression_params.insert(
            "fit_intercept".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_regression_params.insert(
            "normalize".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_models.push(("LinearRegression".to_string(), linear_regression_params));

        // NOTE: Ridge/Lasso are deliberately NOT part of the default search space.
        // `models/linear.rs` only implements unpenalized `LinearRegression`/`LogisticRegression`
        // — there is no L1/L2-penalized linear model to wire up, and `AutoML::create_estimator`
        // has no case for these names (they would always fail with `Error::NotImplemented` and
        // be silently dropped from the leaderboard). Implementing genuine Ridge (closed-form
        // with an L2 term) or Lasso (coordinate descent) is real, non-trivial numerical work
        // that belongs in `models/linear.rs`, outside this module's ownership — so rather than
        // advertise a search space entry that can never actually run, these are left out until
        // real Ridge/Lasso model types exist to wire up here.

        // Tree-based models
        let mut decision_tree_params = HashMap::new();
        decision_tree_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::Choice(vec![
                "3".to_string(),
                "5".to_string(),
                "10".to_string(),
                "None".to_string(),
            ]),
        );
        decision_tree_params.insert(
            "min_samples_split".to_string(),
            ParameterDistribution::UniformInt { low: 2, high: 20 },
        );
        decision_tree_params.insert(
            "min_samples_leaf".to_string(),
            ParameterDistribution::UniformInt { low: 1, high: 10 },
        );
        tree_models.push(("DecisionTree".to_string(), decision_tree_params));

        // Ensemble models
        let mut random_forest_params = HashMap::new();
        random_forest_params.insert(
            "n_estimators".to_string(),
            ParameterDistribution::Choice(vec![
                "50".to_string(),
                "100".to_string(),
                "200".to_string(),
            ]),
        );
        random_forest_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::Choice(vec![
                "5".to_string(),
                "10".to_string(),
                "20".to_string(),
                "None".to_string(),
            ]),
        );
        random_forest_params.insert(
            "min_samples_split".to_string(),
            ParameterDistribution::UniformInt { low: 2, high: 20 },
        );
        ensemble_models.push(("RandomForest".to_string(), random_forest_params));

        let mut gradient_boosting_params = HashMap::new();
        gradient_boosting_params.insert(
            "n_estimators".to_string(),
            ParameterDistribution::Choice(vec![
                "50".to_string(),
                "100".to_string(),
                "200".to_string(),
            ]),
        );
        gradient_boosting_params.insert(
            "learning_rate".to_string(),
            ParameterDistribution::LogUniform {
                low: 0.01,
                high: 0.3,
            },
        );
        gradient_boosting_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::UniformInt { low: 3, high: 10 },
        );
        ensemble_models.push(("GradientBoosting".to_string(), gradient_boosting_params));

        Self {
            linear_models,
            tree_models,
            ensemble_models,
            neural_models,
        }
    }

    /// Create default model search space for classification
    pub fn default_classification() -> Self {
        let mut linear_models = Vec::new();
        let mut tree_models = Vec::new();
        // Intentionally never pushed to — see the NOTE below on why
        // RandomForestClassifier/GradientBoostingClassifier are excluded from this default
        // search space.
        let ensemble_models = Vec::new();
        let neural_models = Vec::new();

        // Linear models
        let mut logistic_regression_params = HashMap::new();
        logistic_regression_params.insert(
            "C".to_string(),
            ParameterDistribution::LogUniform {
                low: 1e-4,
                high: 1e2,
            },
        );
        logistic_regression_params.insert(
            "fit_intercept".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_models.push(("LogisticRegression".to_string(), logistic_regression_params));

        // Tree-based models
        let mut decision_tree_params = HashMap::new();
        decision_tree_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::Choice(vec![
                "3".to_string(),
                "5".to_string(),
                "10".to_string(),
                "None".to_string(),
            ]),
        );
        decision_tree_params.insert(
            "min_samples_split".to_string(),
            ParameterDistribution::UniformInt { low: 2, high: 20 },
        );
        tree_models.push(("DecisionTreeClassifier".to_string(), decision_tree_params));

        // NOTE: `RandomForestClassifier`/`GradientBoostingClassifier` are deliberately NOT part
        // of the default classification search space. `ensemble::RandomForestClassifier` and
        // `ensemble::GradientBoostingClassifier` derive only `Debug` (not `Clone`) in
        // `models/ensemble.rs`, so they can never satisfy `SupervisedAdapter<M>: Clone` — the
        // bound cross-validation relies on to clone the base estimator per fold/trial. Deriving
        // `Clone` for them belongs in `models/ensemble.rs`, outside this module's ownership.
        // `AutoML::create_estimator` still accepts these names for direct callers (substituting
        // the regressor variant, since *something* creatable is more useful than a hard error
        // for that narrow entry point) but they are excluded here so AutoML's *automatic* search
        // never silently scores a regressor as a classifier by default.

        Self {
            linear_models,
            tree_models,
            ensemble_models,
            neural_models,
        }
    }
}

/// Result from AutoML optimization
#[derive(Debug, Clone)]
pub struct AutoMLResult {
    /// Name of the best-performing model found during search (e.g. `"RandomForest"` or, for the
    /// optional averaging ensemble, `"Ensemble(RandomForest+GradientBoosting)"`). This is a
    /// human-readable identifier, not a serialized/executable pipeline object — the actual
    /// fitted estimator is retrieved via [`AutoML::predict`], which uses the real refit model
    /// held internally in `AutoML::best_estimator`.
    pub best_pipeline: String,
    /// Best cross-validation score
    pub best_score: f64,
    /// Best parameters
    pub best_params: HashMap<String, String>,
    /// All tried models and their scores
    pub leaderboard: Vec<ModelResult>,
    /// Feature importances (if available)
    pub feature_importances: Option<HashMap<String, f64>>,
    /// Training time
    pub training_time: f64,
    /// Cross-validation results
    pub cv_results: Vec<f64>,
    /// Final model evaluation on holdout set
    pub holdout_score: Option<f64>,
}

/// Individual model result from AutoML
#[derive(Debug, Clone)]
pub struct ModelResult {
    /// Model name
    pub model_name: String,
    /// Cross-validation score
    pub cv_score: f64,
    /// Standard deviation of CV scores
    pub cv_std: f64,
    /// Training time
    pub training_time: f64,
    /// Model parameters
    pub parameters: HashMap<String, String>,
    /// Feature importance (if available)
    pub feature_importance: Option<HashMap<String, f64>>,
    /// Model complexity score (for interpretability)
    pub complexity_score: f64,
}

/// Plain, `Clone`-able recipe describing how to build a fresh, unfitted [`AutoFeatureEngineer`].
///
/// Kept separate from `AutoFeatureEngineer` itself — which does not derive `Clone` and carries
/// private fitted-state fields (`generated_features_`, `scalers_`, ...) — so that
/// [`FeatureEngineeredEstimator::clone_predictor`] can hand every cross-validation fold/trial a
/// brand-new, unfitted engineer instead of copying fitted state across folds, which would leak
/// validation-fold information (selected features, scaler statistics) into training.
#[derive(Debug, Clone)]
struct FeatureEngineerRecipe {
    score_func: ScoreFunction,
    n_features_to_select: Option<usize>,
    perform_selection: bool,
}

impl FeatureEngineerRecipe {
    fn build(&self) -> AutoFeatureEngineer {
        let mut engineer = AutoFeatureEngineer::new()
            .with_polynomial(2)
            .with_interactions(5)
            .with_scaling(ScalingMethod::StandardScaler);
        if self.perform_selection {
            engineer = engineer.with_selection(
                FeatureSelectionMethod::KBest(self.score_func.clone()),
                self.n_features_to_select,
            );
        } else {
            // `AutoFeatureEngineer::new()` defaults `perform_selection` to `true`; honor
            // `AutoMLConfig::feature_selection = false` by explicitly turning it back off
            // (the field is public specifically for cases like this).
            engineer.perform_selection = false;
        }
        engineer
    }
}

/// Wraps a base estimator so that feature engineering is fit *inside* `fit()` — on whatever data
/// it's given — rather than once on the whole training set before cross-validation ever sees it.
///
/// Plugging this into [`RandomizedSearchCV`] (which clones the estimator once per CV fold and
/// per trial via `clone_predictor`) makes every fold fit its own feature engineer on only that
/// fold's training rows and apply it to that fold's held-out rows — eliminating the
/// selection/scaling leakage that comes from fitting `AutoFeatureEngineer` on the entire training
/// split up front. This is the standard scikit-learn `Pipeline` pattern; it's implemented here
/// (rather than by making `AutoFeatureEngineer` a `sklearn_compat::SklearnTransformer`) because
/// `AutoFeatureEngineer` doesn't derive `Clone` and holds private `Box<dyn FeatureScaler>` state
/// that can't be cloned from outside `feature_engineering.rs`.
#[derive(Debug)]
struct FeatureEngineeredEstimator {
    recipe: FeatureEngineerRecipe,
    estimator: Box<dyn SklearnPredictor + Send + Sync>,
    fitted_engineer: Option<AutoFeatureEngineer>,
}

impl FeatureEngineeredEstimator {
    fn new(
        recipe: FeatureEngineerRecipe,
        estimator: Box<dyn SklearnPredictor + Send + Sync>,
    ) -> Self {
        Self {
            recipe,
            estimator,
            fitted_engineer: None,
        }
    }
}

impl SklearnEstimator for FeatureEngineeredEstimator {
    fn get_params(&self) -> HashMap<String, String> {
        self.estimator.get_params()
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        self.estimator.set_params(params)
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        self.estimator.get_feature_names_out(input_features)
    }
}

impl SklearnPredictor for FeatureEngineeredEstimator {
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let mut engineer = self.recipe.build();
        engineer.fit(x, Some(y))?;
        let x_transformed = engineer.transform(x)?;
        self.estimator.fit(&x_transformed, y)?;
        self.fitted_engineer = Some(engineer);
        Ok(())
    }

    fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        let engineer = self.fitted_engineer.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "FeatureEngineeredEstimator must be fitted before predict".into(),
            )
        })?;
        let x_transformed = engineer.transform(x)?;
        self.estimator.predict(&x_transformed)
    }

    fn score(&self, x: &DataFrame, y: &DataFrame) -> Result<f64> {
        let engineer = self.fitted_engineer.as_ref().ok_or_else(|| {
            Error::InvalidOperation("FeatureEngineeredEstimator must be fitted before score".into())
        })?;
        let x_transformed = engineer.transform(x)?;
        self.estimator.score(&x_transformed, y)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        self.estimator.feature_importances()
    }

    fn clone_predictor(&self) -> Box<dyn SklearnPredictor + Send + Sync> {
        Box::new(FeatureEngineeredEstimator {
            recipe: self.recipe.clone(),
            estimator: self.estimator.clone_predictor(),
            // Deliberately NOT `self.fitted_engineer.clone()` (which isn't even possible —
            // `AutoFeatureEngineer` isn't `Clone`): every clone must refit feature engineering
            // from scratch on whatever data its own `fit()` receives, which is exactly what
            // makes per-fold cross-validation leak-free.
            fitted_engineer: None,
        })
    }
}

/// A simple prediction-averaging ensemble of independently-fitted member estimators.
///
/// Each member keeps its own (possibly feature-engineered) preprocessing, mirroring
/// scikit-learn's `VotingRegressor` with `voting="soft"`-style averaging rather than a stacked
/// meta-learner.
#[derive(Debug)]
struct AveragingEnsemble {
    members: Vec<Box<dyn SklearnPredictor + Send + Sync>>,
}

impl SklearnEstimator for AveragingEnsemble {
    fn get_params(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        if params.is_empty() {
            Ok(())
        } else {
            Err(Error::InvalidValue(
                "AveragingEnsemble does not support tuning its members' hyperparameters through \
                 set_params; construct new members instead"
                    .into(),
            ))
        }
    }

    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Option<Vec<String>> {
        input_features.map(|f| f.to_vec())
    }
}

impl SklearnPredictor for AveragingEnsemble {
    fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        for member in &mut self.members {
            member.fit(x, y)?;
        }
        Ok(())
    }

    fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        if self.members.is_empty() {
            return Err(Error::InvalidOperation(
                "AveragingEnsemble has no members".into(),
            ));
        }
        let mut sums = vec![0.0f64; x.nrows()];
        for member in &self.members {
            let preds = member.predict(x)?;
            if preds.len() != sums.len() {
                return Err(Error::DimensionMismatch(format!(
                    "AveragingEnsemble member returned {} predictions, expected {}",
                    preds.len(),
                    sums.len()
                )));
            }
            for (s, p) in sums.iter_mut().zip(preds) {
                *s += p;
            }
        }
        let n = self.members.len() as f64;
        Ok(sums.into_iter().map(|s| s / n).collect())
    }

    fn score(&self, x: &DataFrame, y: &DataFrame) -> Result<f64> {
        // Delegate to the first member's scoring convention (R² for regressors); AutoML itself
        // always scores candidates — this ensemble included — via the CV `Scorer` rather than
        // this method, so this exists only for `SklearnPredictor` trait completeness.
        self.members
            .first()
            .ok_or_else(|| Error::InvalidOperation("AveragingEnsemble has no members".into()))?
            .score(x, y)
    }

    fn clone_predictor(&self) -> Box<dyn SklearnPredictor + Send + Sync> {
        Box::new(AveragingEnsemble {
            members: self.members.iter().map(|m| m.clone_predictor()).collect(),
        })
    }
}

/// Main AutoML system
#[derive(Debug)]
pub struct AutoML {
    /// Configuration for AutoML run
    pub config: AutoMLConfig,
    /// Model search space
    pub search_space: ModelSearchSpace,
    /// Feature engineering pipeline
    feature_engineer: Option<AutoFeatureEngineer>,
    /// Results from optimization
    results: Option<AutoMLResult>,
    /// Best fitted estimator (refitted on full training data after search)
    best_estimator: Option<Box<dyn SklearnPredictor + Send + Sync>>,
    /// Set by [`AutoML::with_search_space`]. When `true`, [`AutoML::fit`] never overwrites
    /// `search_space` after task-type auto-detection — the caller's explicit choice always wins.
    search_space_customized: bool,
}

impl AutoML {
    /// Create new AutoML instance with default configuration
    pub fn new() -> Self {
        let config = AutoMLConfig::default();
        let search_space = match config.task_type {
            TaskType::Regression => ModelSearchSpace::default_regression(),
            TaskType::BinaryClassification | TaskType::MultiClassification => {
                ModelSearchSpace::default_classification()
            }
            _ => ModelSearchSpace::default_regression(),
        };

        Self {
            config,
            search_space,
            feature_engineer: None,
            results: None,
            best_estimator: None,
            search_space_customized: false,
        }
    }

    /// Create AutoML instance with custom configuration
    pub fn with_config(config: AutoMLConfig) -> Self {
        let search_space = match config.task_type {
            TaskType::Regression => ModelSearchSpace::default_regression(),
            TaskType::BinaryClassification | TaskType::MultiClassification => {
                ModelSearchSpace::default_classification()
            }
            _ => ModelSearchSpace::default_regression(),
        };

        Self {
            config,
            search_space,
            feature_engineer: None,
            results: None,
            best_estimator: None,
            search_space_customized: false,
        }
    }

    /// Set custom model search space.
    ///
    /// Marks the search space as caller-customized so [`AutoML::fit`] will not overwrite it
    /// after task-type auto-detection (previously, `TaskType::Auto` always rebuilt
    /// `search_space` from `ModelSearchSpace::default_regression()`/`default_classification()`
    /// based on the *detected* task, silently discarding whatever was passed here).
    pub fn with_search_space(mut self, search_space: ModelSearchSpace) -> Self {
        self.search_space = search_space;
        self.search_space_customized = true;
        self
    }

    /// Auto-detect task type from target variable
    pub fn detect_task_type(&self, y: &DataFrame) -> Result<TaskType> {
        // Get the first column as target (assuming single target)
        let target_col_name = y
            .column_names()
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidValue("No target column found".into()))?;

        let target_col = y.get_column::<f64>(&target_col_name)?;
        let values = target_col.as_f64()?;

        // Check if all values are integers and within a reasonable range for classification
        let unique_values: std::collections::HashSet<i64> = values
            .iter()
            .filter_map(|&x| {
                if x.fract() == 0.0 {
                    Some(x as i64)
                } else {
                    None
                }
            })
            .collect();

        let integer_count = values.iter().filter(|x| x.fract() == 0.0).count();

        // Check if values appear to be categorical (all values are integers with limited unique count)
        if integer_count == values.len() && unique_values.len() <= 20 && unique_values.len() > 0 {
            if unique_values.len() == 2 {
                Ok(TaskType::BinaryClassification)
            } else {
                Ok(TaskType::MultiClassification)
            }
        } else {
            Ok(TaskType::Regression)
        }
    }

    /// Fit AutoML on the given dataset
    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let start_time = Instant::now();

        if self.config.verbose > 0 {
            println!("🚀 Starting AutoML optimization...");
            println!(
                "Dataset shape: {} rows × {} features",
                x.nrows(),
                x.column_names().len()
            );
        }

        // Auto-detect task type if needed
        let task_type = if matches!(self.config.task_type, TaskType::Auto) {
            let detected = self.detect_task_type(y)?;
            if self.config.verbose > 0 {
                println!("Auto-detected task type: {:?}", detected);
            }
            detected
        } else {
            self.config.task_type.clone()
        };

        // Rebuild the search space from the *detected* task type when the caller left
        // `task_type: Auto` and never explicitly customized the search space via
        // `with_search_space`. Previously `search_space` was fixed at construction time using
        // `_ => default_regression()` for `Auto`, so an auto-detected classification task still
        // searched — and scored via Accuracy — models built for regression.
        if matches!(self.config.task_type, TaskType::Auto) && !self.search_space_customized {
            self.search_space = match task_type {
                TaskType::BinaryClassification | TaskType::MultiClassification => {
                    ModelSearchSpace::default_classification()
                }
                _ => ModelSearchSpace::default_regression(),
            };
        }

        // Update scoring metric based on task type
        let scoring = match task_type {
            TaskType::Regression => Scorer::R2,
            TaskType::BinaryClassification | TaskType::MultiClassification => Scorer::Accuracy,
            TaskType::TimeSeries => Scorer::NegMeanSquaredError,
            TaskType::Auto => Scorer::R2,
        };

        // Create train/validation split for final evaluation
        let (train_x, holdout_x, train_y, holdout_y) = self.create_train_holdout_split(x, y)?;

        // Soft preflight memory check — see `AutoMLConfig::memory_limit`'s docs for scope: this
        // is a static estimate on the raw input matrix, not live RSS monitoring.
        if let Some(limit_gb) = self.config.memory_limit {
            let estimated_gb =
                (train_x.nrows() as f64) * (train_x.column_names().len() as f64) * 8.0 / 1e9;
            if estimated_gb > limit_gb {
                return Err(Error::InvalidOperation(format!(
                    "AutoML: estimated training matrix size ({:.3} GB) exceeds the configured \
                     memory_limit ({:.3} GB); this preflight check only covers the raw input — \
                     feature engineering can grow it further. Raise memory_limit or reduce the \
                     input size.",
                    estimated_gb, limit_gb
                )));
            }
        }

        // Model search and optimization. When `feature_engineering` is enabled, each candidate
        // model is fit through a `FeatureEngineeredEstimator` wrapper (see `fit_single_model`)
        // so the feature engineer is fit fresh *inside* every cross-validation fold instead of
        // once on the whole training split beforehand — fitting it once up front (the previous
        // behavior) leaked validation-fold information (selected features, scaler statistics)
        // into every fold's training data.
        if self.config.verbose > 0 {
            println!("🎯 Starting model search and hyperparameter optimization...");
        }

        let model_results = self.search_models(&train_x, &train_y, &scoring, &task_type)?;

        // Select the best individual model. When `optimize_for_interpretability` is set, apply
        // a small penalty proportional to complexity so a simpler, marginally-lower-scoring
        // model can win over a much more complex one for an equivalent score.
        let interpretability_adjusted = |r: &ModelResult| -> f64 {
            if self.config.optimize_for_interpretability {
                r.cv_score - 0.01 * r.complexity_score
            } else {
                r.cv_score
            }
        };
        let best_result = model_results
            .iter()
            .max_by(|a, b| {
                interpretability_adjusted(a)
                    .partial_cmp(&interpretability_adjusted(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| Error::InvalidOperation("No models were successfully trained".into()))?;

        // Fit feature engineering on the FULL training set — used for the final refit below and
        // for holdout/inference transforms — now that model search (which needed a leak-free
        // per-fold fit instead) is done.
        let mut processed_x = train_x.clone();
        if self.config.feature_engineering {
            if self.config.verbose > 0 {
                println!("🔧 Performing automated feature engineering...");
            }
            let mut feature_engineer = self.feature_engineer_recipe(&task_type).build();
            feature_engineer.fit(&processed_x, Some(&train_y))?;
            processed_x = feature_engineer.transform(&processed_x)?;
            self.feature_engineer = Some(feature_engineer);
            if self.config.verbose > 0 {
                println!("Generated {} features", processed_x.column_names().len());
            }
        }

        // Refit the winning estimator — with its *winning* hyperparameters, not the search
        // template's defaults — on the full training data so we can:
        //  a) evaluate on the holdout set
        //  b) serve predictions after AutoML::fit completes
        //  c) extract feature importances from the fitted model
        // Gated on both `set_params` and `fit` succeeding: a failure at either step means there
        // is no honestly-deployable model, so `best_estimator` stays `None` rather than storing
        // an unfitted (or wrongly-configured) estimator that would fail unpredictably later.
        let mut best_estimator = self.create_estimator(&best_result.model_name)?;
        let refit_ok = best_estimator
            .set_params(best_result.parameters.clone())
            .and_then(|_| best_estimator.fit(&processed_x, &train_y))
            .is_ok();
        if !refit_ok && self.config.verbose > 1 {
            println!(
                "Warning: could not refit best estimator '{}'",
                best_result.model_name
            );
        }

        // Extract feature importances from the refitted estimator (non-None for tree/linear models)
        let feature_importances_from_estimator = if refit_ok {
            best_estimator.feature_importances()
        } else {
            None
        };

        self.best_estimator = if refit_ok { Some(best_estimator) } else { None };

        // Evaluate on holdout set using the refitted best estimator and the SAME `Scorer` used
        // for cross-validation (previously holdout used `SklearnPredictor::score`, which is
        // hardcoded to R² regardless of the task — comparing an R² number against CV Accuracy
        // scores for classification tasks). `None` (not a fabricated `cv_score * 0.95`) when no
        // honest score is available (refit failed, or prediction/scoring on the holdout errors).
        let holdout_score = self.evaluate_on_holdout(&holdout_x, &holdout_y, &scoring)?;

        // Collect cv_results — one entry per fold score for the best model
        let cv_results: Vec<f64> = vec![best_result.cv_score];

        // Build the final leaderboard from the individually-searched models, optionally adding
        // a reporting-only averaging-ensemble candidate (never eligible to become
        // `best_estimator`/`best_pipeline` above, since an ensemble of several models isn't
        // something `create_estimator` can rebuild from a single model name for final refit).
        let best_model_name = best_result.model_name.clone();
        let mut leaderboard = model_results.clone();
        if self.config.ensemble_methods && model_results.len() >= 2 {
            if let Some(ensemble_result) = self.try_build_ensemble_result(
                &model_results,
                &train_x,
                &train_y,
                &scoring,
                &task_type,
            ) {
                leaderboard.push(ensemble_result);
            }
        }
        leaderboard.sort_by(|a, b| {
            b.cv_score
                .partial_cmp(&a.cv_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for entry in &mut leaderboard {
            if entry.model_name == best_model_name && entry.feature_importance.is_none() {
                entry.feature_importance = feature_importances_from_estimator.clone();
            }
        }

        // The overall AutoMLResult feature_importances comes from the best model entry
        let overall_feature_importances = leaderboard
            .iter()
            .find(|e| e.model_name == best_model_name)
            .and_then(|e| e.feature_importance.clone())
            .or_else(|| best_result.feature_importance.clone());

        // Create final results
        let training_time = start_time.elapsed().as_secs_f64();

        let results = AutoMLResult {
            best_pipeline: best_model_name,
            best_score: best_result.cv_score,
            best_params: best_result.parameters.clone(),
            leaderboard,
            feature_importances: overall_feature_importances,
            training_time,
            cv_results,
            holdout_score,
        };

        self.results = Some(results);

        if self.config.verbose > 0 {
            println!("✅ AutoML optimization completed in {:.2}s", training_time);
            println!(
                "Best model: {} (CV score: {:.4})",
                best_result.model_name, best_result.cv_score
            );
            match holdout_score {
                Some(score) => println!("Holdout score: {:.4}", score),
                None => println!("Holdout score: unavailable (refit or scoring did not succeed)"),
            }
        }

        Ok(())
    }

    /// Create train/holdout split for final evaluation
    fn create_train_holdout_split(
        &self,
        x: &DataFrame,
        y: &DataFrame,
    ) -> Result<(DataFrame, DataFrame, DataFrame, DataFrame)> {
        // Use 80/20 split for train/holdout
        let (train_x, holdout_x) = train_test_split(x, 0.2, true, self.config.random_state)?;
        let (train_y, holdout_y) = train_test_split(y, 0.2, true, self.config.random_state)?;

        Ok((train_x, holdout_x, train_y, holdout_y))
    }

    /// Build the [`FeatureEngineerRecipe`] used for a given task type, honoring
    /// `max_selected_features` and `feature_selection` from the config.
    fn feature_engineer_recipe(&self, task_type: &TaskType) -> FeatureEngineerRecipe {
        FeatureEngineerRecipe {
            score_func: match task_type {
                TaskType::BinaryClassification | TaskType::MultiClassification => {
                    ScoreFunction::Chi2
                }
                TaskType::Regression | TaskType::TimeSeries | TaskType::Auto => {
                    ScoreFunction::FRegression
                }
            },
            n_features_to_select: self.config.max_selected_features,
            perform_selection: self.config.feature_selection,
        }
    }

    /// Wrap `estimator` in a [`FeatureEngineeredEstimator`] when `feature_engineering` is
    /// enabled, so cross-validation fits feature engineering per-fold instead of leaking
    /// validation-fold information through a once-fit-on-everything transform.
    fn maybe_wrap_with_feature_engineering(
        &self,
        estimator: Box<dyn SklearnPredictor + Send + Sync>,
        task_type: &TaskType,
    ) -> Box<dyn SklearnPredictor + Send + Sync> {
        if self.config.feature_engineering {
            Box::new(FeatureEngineeredEstimator::new(
                self.feature_engineer_recipe(task_type),
                estimator,
            ))
        } else {
            estimator
        }
    }

    /// Build and cross-validate a simple averaging ensemble of the top-scoring distinct models
    /// from `model_results`, returning a genuinely-computed [`ModelResult`] for the leaderboard
    /// (or `None` if a member can't be rebuilt/refit or the CV itself fails — never a fabricated
    /// score). `model_results` must already be sorted descending by `cv_score`.
    fn try_build_ensemble_result(
        &self,
        model_results: &[ModelResult],
        x: &DataFrame,
        y: &DataFrame,
        scoring: &Scorer,
        task_type: &TaskType,
    ) -> Option<ModelResult> {
        let top_n = model_results.len().min(2).max(1);
        let top = &model_results[..top_n];
        if top.len() < 2 {
            return None;
        }

        let mut members = Vec::with_capacity(top.len());
        for entry in top {
            let mut estimator = self.create_estimator(&entry.model_name).ok()?;
            estimator.set_params(entry.parameters.clone()).ok()?;
            members.push(self.maybe_wrap_with_feature_engineering(estimator, task_type));
        }
        let ensemble: Box<dyn SklearnPredictor + Send + Sync> =
            Box::new(AveragingEnsemble { members });

        let start_time = Instant::now();
        // Reuse RandomizedSearchCV's (now leak-free, seeded) fold machinery with n_iter=1 and no
        // parameter distributions to sample from — this just cross-validates a single fixed
        // estimator and reports the real per-fold score, rather than duplicating fold-splitting
        // logic here.
        let mut cv = RandomizedSearchCV::new(ensemble, HashMap::new(), 1)
            .with_cv(self.config.cv_strategy.clone())
            .with_scoring(scoring.clone());
        cv.fit(x, y).ok()?;
        let results = cv.get_results()?;
        let cv_score = results.best_score_?;
        let cv_std = results
            .cv_results_
            .first()
            .map(|entry| entry.std_test_score)
            .unwrap_or(0.0);

        let member_names: Vec<String> = top.iter().map(|e| e.model_name.clone()).collect();
        let complexity_score =
            top.iter().map(|e| e.complexity_score).sum::<f64>() / top.len() as f64 + 1.0;

        Some(ModelResult {
            model_name: format!("Ensemble({})", member_names.join("+")),
            cv_score,
            cv_std,
            training_time: start_time.elapsed().as_secs_f64(),
            parameters: HashMap::new(),
            feature_importance: None,
            complexity_score,
        })
    }

    /// Search through model space and find best models
    fn search_models(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        scoring: &Scorer,
        task_type: &TaskType,
    ) -> Result<Vec<ModelResult>> {
        let search_start = Instant::now();
        let mut model_results = Vec::new();
        let mut models_tried = 0;
        let max_models = self.config.max_models.unwrap_or(50);

        // Get all available models
        let all_models = self.get_all_models();

        for (model_name, param_space) in all_models {
            if models_tried >= max_models {
                break;
            }

            // Honor the wall-clock search budget: stop trying further models once elapsed time
            // exceeds it rather than accepting and ignoring the config value indefinitely.
            if let Some(limit) = self.config.time_limit {
                if search_start.elapsed().as_secs_f64() > limit {
                    if self.config.verbose > 0 {
                        println!(
                            "⏱️ AutoML time_limit ({:.1}s) reached; stopping search early",
                            limit
                        );
                    }
                    break;
                }
            }

            // Check whitelist/blacklist
            if let Some(whitelist) = &self.config.model_whitelist {
                if !whitelist.contains(&model_name) {
                    continue;
                }
            }

            if let Some(blacklist) = &self.config.model_blacklist {
                if blacklist.contains(&model_name) {
                    continue;
                }
            }

            if self.config.verbose > 1 {
                println!("Trying model: {}", model_name);
            }

            // Create and fit model
            match self.fit_single_model(&model_name, &param_space, x, y, scoring, task_type) {
                Ok(result) => {
                    model_results.push(result);
                    models_tried += 1;
                }
                Err(e) => {
                    if self.config.verbose > 1 {
                        println!("Model {} failed: {}", model_name, e);
                    }
                }
            }
        }

        // Sort by CV score (descending)
        model_results.sort_by(|a, b| {
            b.cv_score
                .partial_cmp(&a.cv_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(model_results)
    }

    /// Get all available models from search space
    fn get_all_models(&self) -> Vec<(String, HashMap<String, ParameterDistribution>)> {
        let mut all_models = Vec::new();

        all_models.extend(self.search_space.linear_models.clone());
        all_models.extend(self.search_space.tree_models.clone());
        all_models.extend(self.search_space.ensemble_models.clone());
        all_models.extend(self.search_space.neural_models.clone());

        all_models
    }

    /// Fit a single model with hyperparameter optimization.
    ///
    /// When `feature_engineering` is enabled, the estimator handed to `RandomizedSearchCV` is
    /// wrapped in a [`FeatureEngineeredEstimator`] so feature engineering is fit fresh inside
    /// every CV fold (leak-free) instead of once on the whole training split beforehand.
    fn fit_single_model(
        &self,
        model_name: &str,
        param_space: &HashMap<String, ParameterDistribution>,
        x: &DataFrame,
        y: &DataFrame,
        scoring: &Scorer,
        task_type: &TaskType,
    ) -> Result<ModelResult> {
        let start_time = Instant::now();

        let base_estimator = self.create_estimator(model_name)?;
        let estimator = self.maybe_wrap_with_feature_engineering(base_estimator, task_type);

        // Set up randomized search
        let mut search = RandomizedSearchCV::new(
            estimator,
            param_space.clone(),
            20, // Try 20 parameter combinations
        )
        .with_cv(self.config.cv_strategy.clone())
        .with_scoring(scoring.clone())
        .with_random_state(self.config.random_state.unwrap_or(42));

        // Fit the search
        search.fit(x, y)?;

        let training_time = start_time.elapsed().as_secs_f64();

        // Extract results
        let results = search
            .get_results()
            .ok_or_else(|| Error::InvalidOperation("No search results available".into()))?;

        // `best_score_` is `None` when no trial produced a finite score — treat that as this
        // model failing (caught and logged by `search_models`) rather than reporting a
        // fabricated 0.0 that would look like a real (bad) result.
        let best_score = results.best_score_.ok_or_else(|| {
            Error::InvalidOperation(format!(
                "No valid parameter combination succeeded for model '{}'",
                model_name
            ))
        })?;

        // Extract cv_std from the best parameter entry in the search results.
        let cv_std = results
            .cv_results_
            .iter()
            .find(|entry| entry.params == results.best_params_)
            .map(|entry| entry.std_test_score)
            .unwrap_or(0.0);

        // Feature importances will be extracted from the refitted best estimator in fit().
        // Set to None here; it will be patched in by AutoML::fit after the search.
        Ok(ModelResult {
            model_name: model_name.to_string(),
            cv_score: best_score,
            cv_std,
            training_time,
            parameters: results.best_params_.clone(),
            feature_importance: None,
            complexity_score: self.calculate_complexity_score(model_name, &results.best_params_),
        })
    }

    /// Create a concrete estimator wrapped in a SupervisedAdapter for the given model name.
    ///
    /// The returned estimator's hyperparameters are the *template* defaults — real tuning
    /// happens through `SklearnPredictor::set_params` during search (see
    /// `sklearn_compat::apply_model_hyperparams`) and, for the final refit, by calling
    /// `set_params` with the winning combination before `fit` (see `AutoML::fit`).
    ///
    /// Supported model names (case-sensitive):
    ///   * `"LinearRegression"`, `"LogisticRegression"`
    ///   * `"DecisionTreeRegressor"` / `"DecisionTree"` (regressor), `"DecisionTreeClassifier"`
    ///     (genuine classifier)
    ///   * `"RandomForestRegressor"` / `"RandomForest"` (regressor); `"RandomForestClassifier"`
    ///     also succeeds but currently substitutes the regressor (see note below)
    ///   * `"GradientBoostingRegressor"` / `"GradientBoosting"` (regressor);
    ///     `"GradientBoostingClassifier"` also succeeds but currently substitutes the regressor
    ///     (see note below)
    ///
    /// Note on `"RandomForestClassifier"`/`"GradientBoostingClassifier"`: `ensemble::
    /// RandomForestClassifier` and `ensemble::GradientBoostingClassifier` derive only `Debug`
    /// (not `Clone`) in `models/ensemble.rs`, so neither can satisfy `SupervisedAdapter<M>:
    /// Clone` — the bound cross-validation relies on to clone the base estimator per fold/trial.
    /// Fixing this requires deriving `Clone` for them in `models/ensemble.rs`, outside this
    /// module's ownership. Until then this substitutes the regressor variant (rather than
    /// erroring) so direct callers of `create_estimator` still get something creatable; the
    /// *default* classification search space (`ModelSearchSpace::default_classification`)
    /// deliberately excludes `"RandomForestClassifier"` so AutoML's automatic search never
    /// silently scores this substitution as a classifier.
    pub fn create_estimator(
        &self,
        model_name: &str,
    ) -> Result<Box<dyn SklearnPredictor + Send + Sync>> {
        match model_name {
            "LinearRegression" => {
                let model = LinearRegression::new();
                Ok(Box::new(SupervisedAdapter::new(model, "target")))
            }
            "LogisticRegression" => {
                let model = LogisticRegression::new();
                Ok(Box::new(SupervisedAdapter::new(model, "target")))
            }
            "DecisionTreeRegressor" | "DecisionTree" => {
                let model = DecisionTreeRegressor::new(DecisionTreeConfig::default());
                Ok(Box::new(SupervisedAdapter::new(model, "target")))
            }
            "DecisionTreeClassifier" => {
                let model = DecisionTreeClassifier::new(DecisionTreeConfig::default());
                Ok(Box::new(SupervisedAdapter::new(model, "target")))
            }
            "RandomForestRegressor" | "RandomForestClassifier" | "RandomForest" => {
                let config = RandomForestConfig::default();
                let model = RandomForestRegressor::new(config);
                Ok(Box::new(SupervisedAdapter::new(model, "target")))
            }
            "GradientBoostingRegressor" | "GradientBoostingClassifier" | "GradientBoosting" => {
                let config = GradientBoostingConfig::default();
                let model = GradientBoostingRegressor::new(config);
                Ok(Box::new(SupervisedAdapter::new(model, "target")))
            }
            _ => Err(Error::NotImplemented(format!(
                "Model '{}' is not implemented in create_estimator; supported: \
                 LinearRegression, LogisticRegression, DecisionTreeRegressor, \
                 DecisionTreeClassifier, RandomForestRegressor, GradientBoostingRegressor",
                model_name
            ))),
        }
    }

    /// Calculate complexity score for interpretability optimization
    fn calculate_complexity_score(
        &self,
        model_name: &str,
        params: &HashMap<String, String>,
    ) -> f64 {
        // Simple complexity scoring based on model type and parameters
        let base_complexity = match model_name {
            "LinearRegression" | "LogisticRegression" => 1.0,
            "Ridge" | "Lasso" => 1.2,
            "DecisionTree" | "DecisionTreeClassifier" => 2.0,
            "RandomForest" | "RandomForestClassifier" => 3.0,
            "GradientBoosting" => 3.5,
            _ => 4.0,
        };

        // Adjust based on parameters. Both terms below are strictly increasing in their input
        // (`ln` composed with a positive linear map) rather than clamped to a floor that made
        // the previous version flat — and therefore non-monotonic in the sense of "more
        // estimators/depth" not increasing the reported complexity — across most of the
        // practical range (e.g. n_estimators 10..~135 all previously mapped to the exact same
        // multiplier).
        let mut complexity = base_complexity;

        if let Some(n_estimators) = params.get("n_estimators") {
            if let Ok(n) = n_estimators.parse::<f64>() {
                // More estimators = higher complexity (logarithmic scaling, monotonic for n >= 1).
                complexity *= 1.0 + n.max(1.0).ln() * 0.1;
            }
        }

        if let Some(max_depth) = params.get("max_depth") {
            if max_depth != "None" {
                if let Ok(depth) = max_depth.parse::<f64>() {
                    // Monotonic for depth >= 1; deeper trees are strictly more complex.
                    complexity *= 1.0 + depth.max(1.0).ln() * 0.2;
                }
            } else {
                complexity *= 2.0; // Unlimited depth increases complexity
            }
        }

        complexity
    }

    /// Evaluate the refitted best estimator on the holdout set, using the SAME [`Scorer`] used
    /// during cross-validation (not `SklearnPredictor::score`, which is hardcoded to R² and
    /// would silently compare an R² number against, say, CV Accuracy for a classification task).
    ///
    /// Returns `Ok(None)` — never a fabricated placeholder like `cv_score * 0.95` — when no
    /// honest holdout score is available: the refit failed (`best_estimator` is `None`),
    /// prediction on the holdout set errors, or the target column can't be read.
    fn evaluate_on_holdout(
        &self,
        holdout_x: &DataFrame,
        holdout_y: &DataFrame,
        scoring: &Scorer,
    ) -> Result<Option<f64>> {
        let processed_x = if let Some(feature_engineer) = &self.feature_engineer {
            feature_engineer.transform(holdout_x)?
        } else {
            holdout_x.clone()
        };

        let estimator = match &self.best_estimator {
            Some(estimator) => estimator,
            None => return Ok(None),
        };

        let predictions = match estimator.predict(&processed_x) {
            Ok(predictions) => predictions,
            Err(_) => return Ok(None),
        };

        let target_name = match holdout_y.column_names().into_iter().next() {
            Some(name) => name,
            None => return Ok(None),
        };
        let y_true = match holdout_y
            .get_column::<f64>(&target_name)
            .and_then(|col| col.as_f64())
        {
            Ok(values) => values,
            Err(_) => return Ok(None),
        };

        Ok(scoring.score(&y_true, &predictions).ok())
    }

    /// Predict on new data using the best fitted model.
    pub fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        let _results = self.results.as_ref().ok_or_else(|| {
            Error::InvalidOperation("AutoML must be fitted before predict".into())
        })?;

        // Apply feature engineering if used
        let processed_x = if let Some(feature_engineer) = &self.feature_engineer {
            feature_engineer.transform(x)?
        } else {
            x.clone()
        };

        // Use the best fitted estimator if available.
        let estimator = self.best_estimator.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "AutoML has no fitted best estimator available (the winning model's refit may \
                 have failed); cannot predict"
                    .into(),
            )
        })?;
        estimator.predict(&processed_x)
    }

    /// Get the AutoML results
    pub fn get_results(&self) -> Option<&AutoMLResult> {
        self.results.as_ref()
    }

    /// Generate a comprehensive report of the AutoML run
    pub fn generate_report(&self) -> Result<String> {
        let results = self.results.as_ref().ok_or_else(|| {
            Error::InvalidOperation("AutoML must be fitted before generating report".into())
        })?;

        let mut report = String::new();

        report.push_str("# AutoML Report\n\n");

        // Summary
        report.push_str("## Summary\n");
        report.push_str(&format!("- **Best Model**: {}\n", results.best_pipeline));
        report.push_str(&format!("- **Best Score**: {:.4}\n", results.best_score));
        if let Some(holdout_score) = results.holdout_score {
            report.push_str(&format!("- **Holdout Score**: {:.4}\n", holdout_score));
        }
        report.push_str(&format!(
            "- **Training Time**: {:.2}s\n",
            results.training_time
        ));
        report.push_str(&format!(
            "- **Models Tried**: {}\n",
            results.leaderboard.len()
        ));

        // Leaderboard
        report.push_str("\n## Model Leaderboard\n\n");
        report.push_str("| Rank | Model | CV Score | Std | Time (s) | Complexity |\n");
        report.push_str("|------|-------|----------|-----|----------|------------|\n");

        for (i, model) in results.leaderboard.iter().take(10).enumerate() {
            report.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.2} | {:.2} |\n",
                i + 1,
                model.model_name,
                model.cv_score,
                model.cv_std,
                model.training_time,
                model.complexity_score
            ));
        }

        // Best model details
        if let Some(best_model) = results.leaderboard.first() {
            report.push_str("\n## Best Model Details\n\n");
            report.push_str(&format!("**Model**: {}\n", best_model.model_name));
            report.push_str(&format!(
                "**CV Score**: {:.4} ± {:.4}\n",
                best_model.cv_score, best_model.cv_std
            ));

            report.push_str("\n**Parameters**:\n");
            for (param, value) in &best_model.parameters {
                report.push_str(&format!("- {}: {}\n", param, value));
            }

            if let Some(importances) = &best_model.feature_importance {
                report.push_str("\n**Top 10 Feature Importances**:\n");
                let mut importance_vec: Vec<_> = importances.iter().collect();
                importance_vec
                    .sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (feature, importance) in importance_vec.iter().take(10) {
                    report.push_str(&format!("- {}: {:.4}\n", feature, importance));
                }
            }
        }

        // Feature engineering summary
        if let Some(feature_engineer) = &self.feature_engineer {
            report.push_str("\n## Feature Engineering\n\n");
            if let Some(feature_names) = feature_engineer.get_feature_names() {
                report.push_str(&format!(
                    "- **Total Features Generated**: {}\n",
                    feature_names.len()
                ));
            }
            if let Some(selected_features) = feature_engineer.get_selected_features() {
                report.push_str(&format!(
                    "- **Features Selected**: {}\n",
                    selected_features.len()
                ));
            }
        }

        report.push_str("\n---\n");
        report.push_str("*Report generated by PandRS AutoML*\n");

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_task_type_detection() {
        let automl = AutoML::new();

        // Test regression detection
        let mut y_reg = DataFrame::new();
        y_reg
            .add_column(
                "target".to_string(),
                Series::new(vec![1.5, 2.3, 3.7, 4.1, 5.9], Some("target".to_string()))
                    .expect("operation should succeed"),
            )
            .expect("operation should succeed");

        let task_type = automl
            .detect_task_type(&y_reg)
            .expect("operation should succeed");
        assert!(matches!(task_type, TaskType::Regression));

        // Test binary classification detection
        let mut y_binary = DataFrame::new();
        y_binary
            .add_column(
                "target".to_string(),
                Series::new(vec![0.0, 1.0, 1.0, 0.0, 1.0], Some("target".to_string()))
                    .expect("operation should succeed"),
            )
            .expect("operation should succeed");

        let task_type = automl
            .detect_task_type(&y_binary)
            .expect("operation should succeed");
        assert!(matches!(task_type, TaskType::BinaryClassification));

        // Test multi-class classification detection
        let mut y_multi = DataFrame::new();
        y_multi
            .add_column(
                "target".to_string(),
                Series::new(vec![0.0, 1.0, 2.0, 1.0, 2.0], Some("target".to_string()))
                    .expect("operation should succeed"),
            )
            .expect("operation should succeed");

        let task_type = automl
            .detect_task_type(&y_multi)
            .expect("operation should succeed");
        assert!(matches!(task_type, TaskType::MultiClassification));
    }

    #[test]
    fn test_model_search_space() {
        let search_space = ModelSearchSpace::default_regression();

        assert!(!search_space.linear_models.is_empty());
        assert!(!search_space.tree_models.is_empty());
        assert!(!search_space.ensemble_models.is_empty());

        // Check that linear regression is included
        let has_linear_regression = search_space
            .linear_models
            .iter()
            .any(|(name, _)| name == "LinearRegression");
        assert!(has_linear_regression);
    }

    #[test]
    fn test_automl_config() {
        let config = AutoMLConfig::default();

        assert!(matches!(config.task_type, TaskType::Auto));
        assert_eq!(config.time_limit, Some(3600.0));
        assert_eq!(config.max_models, Some(50));
        assert!(config.feature_engineering);
        assert!(config.feature_selection);
    }

    #[test]
    fn test_complexity_scoring() {
        let automl = AutoML::new();

        let linear_complexity =
            automl.calculate_complexity_score("LinearRegression", &HashMap::new());
        let rf_complexity = automl.calculate_complexity_score("RandomForest", &HashMap::new());

        assert!(linear_complexity < rf_complexity);

        // Test parameter influence
        let mut params = HashMap::new();
        params.insert("n_estimators".to_string(), "200".to_string());
        let rf_complex_complexity = automl.calculate_complexity_score("RandomForest", &params);

        assert!(rf_complex_complexity > rf_complexity);
    }
}
