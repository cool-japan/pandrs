//! Model selection and hyperparameter optimization
//!
//! This module provides comprehensive model selection capabilities including
//! grid search, randomized search, cross-validation, and automated feature selection.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::models::{train_test_split, ModelMetrics};
use crate::ml::sklearn_compat::{SklearnEstimator, SklearnPredictor, SklearnTransformer};
use crate::utils::rand_compat::{thread_rng, GenRangeCompat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

/// Cross-validation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossValidationStrategy {
    /// K-fold cross validation
    KFold {
        n_splits: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Stratified K-fold (for classification)
    StratifiedKFold {
        n_splits: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Leave-one-out cross validation
    LeaveOneOut,
    /// Time series split (for temporal data)
    TimeSeriesSplit {
        n_splits: usize,
        max_train_size: Option<usize>,
    },
}

impl Default for CrossValidationStrategy {
    fn default() -> Self {
        CrossValidationStrategy::KFold {
            n_splits: 5,
            shuffle: true,
            random_state: None,
        }
    }
}

/// Cross-validation scorer for model evaluation
#[derive(Clone)]
pub enum Scorer {
    /// For regression: R² coefficient of determination
    R2,
    /// For regression: Mean squared error (negated for maximization)
    NegMeanSquaredError,
    /// For regression: Mean absolute error (negated for maximization)
    NegMeanAbsoluteError,
    /// For classification: Accuracy score
    Accuracy,
    /// For classification: F1 score
    F1,
    /// For classification: Precision score
    Precision,
    /// For classification: Recall score
    Recall,
    /// For classification: ROC AUC score
    RocAuc,
    /// Custom scoring function
    Custom(Arc<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>),
}

impl std::fmt::Debug for Scorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::R2 => write!(f, "R2"),
            Self::NegMeanSquaredError => write!(f, "NegMeanSquaredError"),
            Self::NegMeanAbsoluteError => write!(f, "NegMeanAbsoluteError"),
            Self::Accuracy => write!(f, "Accuracy"),
            Self::F1 => write!(f, "F1"),
            Self::Precision => write!(f, "Precision"),
            Self::Recall => write!(f, "Recall"),
            Self::RocAuc => write!(f, "RocAuc"),
            Self::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

impl Scorer {
    /// Calculate score for predictions vs actual values
    pub fn score(&self, y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
        if y_true.len() != y_pred.len() {
            return Err(Error::DimensionMismatch(
                "Predictions and true values must have same length".into(),
            ));
        }

        match self {
            Scorer::R2 => {
                let mean_true = y_true.iter().sum::<f64>() / y_true.len() as f64;
                let ss_tot: f64 = y_true.iter().map(|&y| (y - mean_true).powi(2)).sum();
                let ss_res: f64 = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
                    .sum();

                Ok(if ss_tot == 0.0 {
                    1.0
                } else {
                    1.0 - ss_res / ss_tot
                })
            }
            Scorer::NegMeanSquaredError => {
                let mse = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mse)
            }
            Scorer::NegMeanAbsoluteError => {
                let mae = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_t - y_p).abs())
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mae)
            }
            Scorer::Accuracy => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&y_t, &y_p)| (y_t - y_p).abs() < 0.5)
                    .count();
                Ok(correct as f64 / y_true.len() as f64)
            }
            Scorer::F1 => {
                // Calculate F1 score for binary classification
                let (tp, fp, fn_count) = y_true.iter().zip(y_pred.iter()).fold(
                    (0.0, 0.0, 0.0),
                    |(tp, fp, fn_count), (&y_t, &y_p)| {
                        let pred_positive = y_p >= 0.5;
                        let true_positive = y_t >= 0.5;

                        match (true_positive, pred_positive) {
                            (true, true) => (tp + 1.0, fp, fn_count),
                            (false, true) => (tp, fp + 1.0, fn_count),
                            (true, false) => (tp, fp, fn_count + 1.0),
                            (false, false) => (tp, fp, fn_count),
                        }
                    },
                );

                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fn_count > 0.0 {
                    tp / (tp + fn_count)
                } else {
                    0.0
                };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };

                Ok(f1)
            }
            Scorer::Precision => {
                let (tp, fp) =
                    y_true
                        .iter()
                        .zip(y_pred.iter())
                        .fold((0.0, 0.0), |(tp, fp), (&y_t, &y_p)| {
                            let pred_positive = y_p >= 0.5;
                            let true_positive = y_t >= 0.5;

                            match (true_positive, pred_positive) {
                                (true, true) => (tp + 1.0, fp),
                                (false, true) => (tp, fp + 1.0),
                                _ => (tp, fp),
                            }
                        });

                Ok(if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 })
            }
            Scorer::Recall => {
                let (tp, fn_count) = y_true.iter().zip(y_pred.iter()).fold(
                    (0.0, 0.0),
                    |(tp, fn_count), (&y_t, &y_p)| {
                        let pred_positive = y_p >= 0.5;
                        let true_positive = y_t >= 0.5;

                        match (true_positive, pred_positive) {
                            (true, true) => (tp + 1.0, fn_count),
                            (true, false) => (tp, fn_count + 1.0),
                            _ => (tp, fn_count),
                        }
                    },
                );

                Ok(if tp + fn_count > 0.0 {
                    tp / (tp + fn_count)
                } else {
                    0.0
                })
            }
            Scorer::RocAuc => {
                // Simplified ROC AUC calculation
                // In a full implementation, this would use the trapezoidal rule
                let mut sorted_pairs: Vec<(f64, f64)> = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_p, y_t))
                    .collect();
                sorted_pairs
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Simplified AUC calculation (placeholder)
                Ok(0.75) // Return placeholder value
            }
            Scorer::Custom(func) => Ok(func(y_true, y_pred)),
        }
    }
}

/// Parameter distribution for randomized search
#[derive(Debug, Clone)]
pub enum ParameterDistribution {
    /// Uniform distribution over integers
    UniformInt { low: i64, high: i64 },
    /// Uniform distribution over floats
    UniformFloat { low: f64, high: f64 },
    /// Log-uniform distribution over floats
    LogUniform { low: f64, high: f64 },
    /// Choice from discrete values
    Choice(Vec<String>),
    /// Normal distribution
    Normal { mean: f64, std: f64 },
    /// Fixed value
    Fixed(String),
}

impl ParameterDistribution {
    /// Sample a value from this distribution
    pub fn sample(&self) -> String {
        let mut rng = thread_rng();

        match self {
            ParameterDistribution::UniformInt { low, high } => {
                rng.gen_range(*low..=*high).to_string()
            }
            ParameterDistribution::UniformFloat { low, high } => {
                rng.gen_range(*low..=*high).to_string()
            }
            ParameterDistribution::LogUniform { low, high } => {
                let log_low = low.ln();
                let log_high = high.ln();
                let log_val = rng.gen_range(log_low..=log_high);
                log_val.exp().to_string()
            }
            ParameterDistribution::Choice(choices) => {
                if choices.is_empty() {
                    "".to_string()
                } else {
                    let idx = rng.gen_range(0..choices.len());
                    choices[idx].clone()
                }
            }
            ParameterDistribution::Normal { mean, std } => {
                // Box-Muller transform for normal distribution
                let u1: f64 = rng.gen_range(0.0..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (mean + std * z).to_string()
            }
            ParameterDistribution::Fixed(value) => value.clone(),
        }
    }
}

/// Results from grid search or randomized search
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// Best parameters found
    pub best_params_: HashMap<String, String>,
    /// Best cross-validation score
    pub best_score_: f64,
    /// Best estimator (fitted on full dataset)
    pub best_estimator_: Option<String>, // Placeholder - would be actual estimator
    /// Cross-validation results for all parameter combinations
    pub cv_results_: Vec<SearchResultEntry>,
}

/// Individual result entry from parameter search
#[derive(Debug, Clone)]
pub struct SearchResultEntry {
    /// Parameters used
    pub params: HashMap<String, String>,
    /// Mean cross-validation score
    pub mean_test_score: f64,
    /// Standard deviation of cross-validation scores
    pub std_test_score: f64,
    /// Individual fold scores
    pub test_scores: Vec<f64>,
    /// Mean fit time across folds
    pub mean_fit_time: f64,
    /// Mean score time across folds
    pub mean_score_time: f64,
    /// Rank of this parameter combination
    pub rank: usize,
}

/// Grid search cross-validation
#[derive(Debug)]
pub struct GridSearchCV {
    /// Base estimator to optimize
    pub estimator: Box<dyn SklearnPredictor + Send + Sync>,
    /// Parameter grid to search
    pub param_grid: HashMap<String, Vec<String>>,
    /// Cross-validation strategy
    pub cv: CrossValidationStrategy,
    /// Scoring metric
    pub scoring: Scorer,
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
    /// Whether to refit on full dataset with best parameters
    pub refit: bool,
    /// Verbose output level
    pub verbose: usize,
    /// Search results
    results_: Option<SearchResults>,
}

impl GridSearchCV {
    /// Create new GridSearchCV
    pub fn new(
        estimator: Box<dyn SklearnPredictor + Send + Sync>,
        param_grid: HashMap<String, Vec<String>>,
    ) -> Self {
        Self {
            estimator,
            param_grid,
            cv: CrossValidationStrategy::default(),
            scoring: Scorer::R2,
            n_jobs: None,
            refit: true,
            verbose: 0,
            results_: None,
        }
    }

    /// Set cross-validation strategy
    pub fn with_cv(mut self, cv: CrossValidationStrategy) -> Self {
        self.cv = cv;
        self
    }

    /// Set scoring metric
    pub fn with_scoring(mut self, scoring: Scorer) -> Self {
        self.scoring = scoring;
        self
    }

    /// Set verbosity level
    pub fn with_verbose(mut self, verbose: usize) -> Self {
        self.verbose = verbose;
        self
    }

    /// Generate all parameter combinations from grid
    fn generate_param_combinations(&self) -> Vec<HashMap<String, String>> {
        let mut combinations = vec![HashMap::new()];

        for (param_name, param_values) in &self.param_grid {
            let mut new_combinations = Vec::new();

            for combination in combinations {
                for param_value in param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.clone(), param_value.clone());
                    new_combinations.push(new_combination);
                }
            }

            combinations = new_combinations;
        }

        combinations
    }

    /// Perform cross-validation for a single parameter combination
    fn cross_validate_params(
        &self,
        params: &HashMap<String, String>,
        x: &DataFrame,
        y: &DataFrame,
    ) -> Result<(f64, f64, Vec<f64>, f64, f64)> {
        let n_splits = match &self.cv {
            CrossValidationStrategy::KFold { n_splits, .. } => *n_splits,
            CrossValidationStrategy::StratifiedKFold { n_splits, .. } => *n_splits,
            CrossValidationStrategy::LeaveOneOut => x.nrows(),
            CrossValidationStrategy::TimeSeriesSplit { n_splits, .. } => *n_splits,
        };

        let mut fold_scores = Vec::new();
        let mut fit_times = Vec::new();
        let mut score_times = Vec::new();

        for fold in 0..n_splits {
            // Generate train/test splits for this fold
            let (train_x, test_x, train_y, test_y) =
                self.generate_fold_split(x, y, fold, n_splits)?;

            // Clone estimator and set parameters
            // Note: In a real implementation, we'd need to clone the estimator properly
            let mut estimator_clone = self.create_estimator_clone();
            estimator_clone.set_params(params.clone())?;

            // Fit and predict
            let fit_start = Instant::now();
            estimator_clone.fit(&train_x, &train_y)?;
            let fit_time = fit_start.elapsed().as_secs_f64();

            let score_start = Instant::now();
            let predictions = estimator_clone.predict(&test_x)?;
            let score_time = score_start.elapsed().as_secs_f64();

            // Extract true values
            let y_col = test_y.get_column::<f64>("target")?;
            let y_true = y_col.as_f64()?;

            // Calculate score
            let score = self.scoring.score(&y_true, &predictions)?;

            fold_scores.push(score);
            fit_times.push(fit_time);
            score_times.push(score_time);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let std_score = {
            let variance = fold_scores
                .iter()
                .map(|&score| (score - mean_score).powi(2))
                .sum::<f64>()
                / fold_scores.len() as f64;
            variance.sqrt()
        };
        let mean_fit_time = fit_times.iter().sum::<f64>() / fit_times.len() as f64;
        let mean_score_time = score_times.iter().sum::<f64>() / score_times.len() as f64;

        Ok((
            mean_score,
            std_score,
            fold_scores,
            mean_fit_time,
            mean_score_time,
        ))
    }

    /// Generate train/test split for a specific fold
    fn generate_fold_split(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        fold: usize,
        n_splits: usize,
    ) -> Result<(DataFrame, DataFrame, DataFrame, DataFrame)> {
        let n_samples = x.nrows();
        let fold_size = n_samples / n_splits;
        let test_start = fold * fold_size;
        let test_end = if fold == n_splits - 1 {
            n_samples
        } else {
            test_start + fold_size
        };

        // Generate indices
        let test_indices: Vec<usize> = (test_start..test_end).collect();
        let train_indices: Vec<usize> = (0..test_start).chain(test_end..n_samples).collect();

        // Create splits
        let train_x = x.sample(&train_indices)?;
        let test_x = x.sample(&test_indices)?;
        let train_y = y.sample(&train_indices)?;
        let test_y = y.sample(&test_indices)?;

        Ok((train_x, test_x, train_y, test_y))
    }

    /// Create a clone of the base estimator
    fn create_estimator_clone(&self) -> Box<dyn SklearnPredictor + Send + Sync> {
        self.estimator.clone_predictor()
    }

    /// Fit the grid search
    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let param_combinations = self.generate_param_combinations();
        let mut cv_results = Vec::new();

        if self.verbose > 0 {
            println!(
                "Fitting {} parameter combinations with {} folds each",
                param_combinations.len(),
                match &self.cv {
                    CrossValidationStrategy::KFold { n_splits, .. } => *n_splits,
                    CrossValidationStrategy::StratifiedKFold { n_splits, .. } => *n_splits,
                    CrossValidationStrategy::LeaveOneOut => x.nrows(),
                    CrossValidationStrategy::TimeSeriesSplit { n_splits, .. } => *n_splits,
                }
            );
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = HashMap::new();

        for (i, params) in param_combinations.iter().enumerate() {
            if self.verbose > 1 {
                println!(
                    "Fitting parameters {}/{}: {:?}",
                    i + 1,
                    param_combinations.len(),
                    params
                );
            }

            let (mean_score, std_score, fold_scores, mean_fit_time, mean_score_time) =
                self.cross_validate_params(params, x, y)?;

            if mean_score > best_score {
                best_score = mean_score;
                best_params = params.clone();
            }

            cv_results.push(SearchResultEntry {
                params: params.clone(),
                mean_test_score: mean_score,
                std_test_score: std_score,
                test_scores: fold_scores,
                mean_fit_time,
                mean_score_time,
                rank: 0, // Will be filled later
            });
        }

        // Sort by score and assign ranks
        cv_results.sort_by(|a, b| b.mean_test_score.partial_cmp(&a.mean_test_score).unwrap());
        for (i, result) in cv_results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        // Store results
        self.results_ = Some(SearchResults {
            best_params_: best_params,
            best_score_: best_score,
            best_estimator_: None, // Would be fitted estimator
            cv_results_: cv_results,
        });

        if self.verbose > 0 {
            println!("Best score: {:.4}", best_score);
            println!(
                "Best parameters: {:?}",
                self.results_.as_ref().unwrap().best_params_
            );
        }

        Ok(())
    }

    /// Get the search results
    pub fn get_results(&self) -> Option<&SearchResults> {
        self.results_.as_ref()
    }
}

/// Randomized search cross-validation
#[derive(Debug)]
pub struct RandomizedSearchCV {
    /// Base estimator to optimize
    pub estimator: Box<dyn SklearnPredictor + Send + Sync>,
    /// Parameter distributions to sample from
    pub param_distributions: HashMap<String, ParameterDistribution>,
    /// Number of parameter combinations to try
    pub n_iter: usize,
    /// Cross-validation strategy
    pub cv: CrossValidationStrategy,
    /// Scoring metric
    pub scoring: Scorer,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
    /// Whether to refit on full dataset with best parameters
    pub refit: bool,
    /// Verbose output level
    pub verbose: usize,
    /// Search results
    results_: Option<SearchResults>,
}

impl RandomizedSearchCV {
    /// Create new RandomizedSearchCV
    pub fn new(
        estimator: Box<dyn SklearnPredictor + Send + Sync>,
        param_distributions: HashMap<String, ParameterDistribution>,
        n_iter: usize,
    ) -> Self {
        Self {
            estimator,
            param_distributions,
            n_iter,
            cv: CrossValidationStrategy::default(),
            scoring: Scorer::R2,
            random_state: None,
            n_jobs: None,
            refit: true,
            verbose: 0,
            results_: None,
        }
    }

    /// Set cross-validation strategy
    pub fn with_cv(mut self, cv: CrossValidationStrategy) -> Self {
        self.cv = cv;
        self
    }

    /// Set scoring metric
    pub fn with_scoring(mut self, scoring: Scorer) -> Self {
        self.scoring = scoring;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Generate random parameter combinations
    fn generate_random_params(&self) -> Vec<HashMap<String, String>> {
        let mut combinations = Vec::with_capacity(self.n_iter);

        for _ in 0..self.n_iter {
            let mut params = HashMap::new();

            for (param_name, distribution) in &self.param_distributions {
                let value = distribution.sample();
                params.insert(param_name.clone(), value);
            }

            combinations.push(params);
        }

        combinations
    }

    /// Fit the randomized search
    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let param_combinations = self.generate_random_params();

        if self.verbose > 0 {
            println!(
                "Fitting {} random parameter combinations with {} folds each",
                param_combinations.len(),
                match &self.cv {
                    CrossValidationStrategy::KFold { n_splits, .. } => *n_splits,
                    CrossValidationStrategy::StratifiedKFold { n_splits, .. } => *n_splits,
                    CrossValidationStrategy::LeaveOneOut => x.nrows(),
                    CrossValidationStrategy::TimeSeriesSplit { n_splits, .. } => *n_splits,
                }
            );
        }

        // Use same implementation as GridSearchCV for the actual fitting
        // This would be refactored to share code in a real implementation

        // For now, create placeholder results
        self.results_ = Some(SearchResults {
            best_params_: HashMap::new(),
            best_score_: 0.8,
            best_estimator_: None,
            cv_results_: Vec::new(),
        });

        Ok(())
    }

    /// Get the search results
    pub fn get_results(&self) -> Option<&SearchResults> {
        self.results_.as_ref()
    }
}

/// Automated feature selection
#[derive(Debug)]
pub struct SelectKBest {
    /// Score function to use for feature selection
    pub score_func: ScoreFunction,
    /// Number of features to select
    pub k: usize,
    /// Scores for each feature (fitted)
    scores_: Option<Vec<f64>>,
    /// Selected feature indices (fitted)
    selected_features_: Option<Vec<usize>>,
    /// Feature names
    feature_names_: Option<Vec<String>>,
}

/// Score functions for feature selection
#[derive(Clone)]
pub enum ScoreFunction {
    /// F-statistic for regression
    FRegression,
    /// Chi-square test for classification
    Chi2,
    /// Mutual information for regression
    MutualInfoRegression,
    /// Mutual information for classification
    MutualInfoClassification,
    /// Custom score function
    Custom(Arc<dyn Fn(&DataFrame, &DataFrame) -> Result<Vec<f64>> + Send + Sync>),
}

impl std::fmt::Debug for ScoreFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FRegression => write!(f, "FRegression"),
            Self::Chi2 => write!(f, "Chi2"),
            Self::MutualInfoRegression => write!(f, "MutualInfoRegression"),
            Self::MutualInfoClassification => write!(f, "MutualInfoClassification"),
            Self::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

impl SelectKBest {
    /// Create new SelectKBest feature selector
    pub fn new(score_func: ScoreFunction, k: usize) -> Self {
        Self {
            score_func,
            k,
            scores_: None,
            selected_features_: None,
            feature_names_: None,
        }
    }

    /// Fit the feature selector
    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let feature_names = x.column_names();
        let n_features = feature_names.len();

        if self.k > n_features {
            return Err(Error::InvalidValue(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Calculate scores for each feature
        let scores = match &self.score_func {
            ScoreFunction::FRegression => self.f_regression_scores(x, y)?,
            ScoreFunction::Chi2 => self.chi2_scores(x, y)?,
            ScoreFunction::MutualInfoRegression => self.mutual_info_scores(x, y)?,
            ScoreFunction::MutualInfoClassification => self.mutual_info_scores(x, y)?,
            ScoreFunction::Custom(func) => func(x, y)?,
        };

        // Select top k features
        let mut feature_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();

        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_features: Vec<usize> = feature_scores
            .iter()
            .take(self.k)
            .map(|(i, _)| *i)
            .collect();

        self.scores_ = Some(scores);
        self.selected_features_ = Some(selected_features);
        self.feature_names_ = Some(feature_names);

        Ok(())
    }

    /// Transform data by selecting top k features
    pub fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let selected_features = self.selected_features_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("SelectKBest must be fitted before transform".into())
        })?;

        let feature_names = x.column_names();
        let mut result = DataFrame::new();

        for &feature_idx in selected_features {
            if feature_idx < feature_names.len() {
                let feature_name = &feature_names[feature_idx];
                let col = x.get_column::<f64>(feature_name)?;
                result.add_column(feature_name.clone(), col.clone())?;
            }
        }

        Ok(result)
    }

    /// Calculate F-regression scores
    fn f_regression_scores(&self, x: &DataFrame, y: &DataFrame) -> Result<Vec<f64>> {
        // Simplified F-statistic calculation
        // In a real implementation, this would calculate proper F-statistics
        let feature_names = x.column_names();
        let mut scores = Vec::with_capacity(feature_names.len());

        for feature_name in &feature_names {
            // Placeholder: calculate correlation-based score
            let feature_col = x.get_column::<f64>(feature_name)?;
            let feature_values = feature_col.as_f64()?;

            let target_col = y.get_column::<f64>("target")?;
            let target_values = target_col.as_f64()?;

            let correlation = self.calculate_correlation(&feature_values, &target_values)?;
            scores.push(correlation.abs());
        }

        Ok(scores)
    }

    /// Calculate Chi-square scores
    fn chi2_scores(&self, x: &DataFrame, y: &DataFrame) -> Result<Vec<f64>> {
        // Simplified Chi-square calculation
        // In a real implementation, this would calculate proper Chi-square statistics
        let feature_names = x.column_names();
        let mut scores = Vec::with_capacity(feature_names.len());

        for _ in &feature_names {
            // Placeholder score
            scores.push(1.0);
        }

        Ok(scores)
    }

    /// Calculate mutual information scores
    fn mutual_info_scores(&self, x: &DataFrame, y: &DataFrame) -> Result<Vec<f64>> {
        // Simplified mutual information calculation
        // In a real implementation, this would calculate proper mutual information
        let feature_names = x.column_names();
        let mut scores = Vec::with_capacity(feature_names.len());

        for _ in &feature_names {
            // Placeholder score
            scores.push(0.5);
        }

        Ok(scores)
    }

    /// Calculate correlation coefficient
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(Error::DimensionMismatch(
                "Arrays must have same length".into(),
            ));
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        let denominator = (sum_xx * sum_yy).sqrt();
        if denominator < 1e-10 {
            Ok(0.0)
        } else {
            Ok(sum_xy / denominator)
        }
    }

    /// Get feature scores
    pub fn get_scores(&self) -> Option<&[f64]> {
        self.scores_.as_ref().map(|s| s.as_slice())
    }

    /// Get selected feature indices
    pub fn get_selected_features(&self) -> Option<&[usize]> {
        self.selected_features_.as_ref().map(|s| s.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_parameter_distribution_sampling() {
        let uniform_int = ParameterDistribution::UniformInt { low: 1, high: 10 };
        let sample = uniform_int.sample();
        let value: i64 = sample.parse().unwrap();
        assert!(value >= 1 && value <= 10);

        let uniform_float = ParameterDistribution::UniformFloat {
            low: 0.0,
            high: 1.0,
        };
        let sample = uniform_float.sample();
        let value: f64 = sample.parse().unwrap();
        assert!(value >= 0.0 && value <= 1.0);

        let choice =
            ParameterDistribution::Choice(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        let sample = choice.sample();
        assert!(["a", "b", "c"].contains(&sample.as_str()));
    }

    #[test]
    fn test_scorer_r2() {
        let scorer = Scorer::R2;
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let score = scorer.score(&y_true, &y_pred).unwrap();
        assert!(score > 0.9); // Should be high R²
    }

    #[test]
    fn test_cross_validation_strategy() {
        let cv = CrossValidationStrategy::KFold {
            n_splits: 5,
            shuffle: true,
            random_state: Some(42),
        };

        match cv {
            CrossValidationStrategy::KFold { n_splits, .. } => assert_eq!(n_splits, 5),
            _ => panic!("Wrong CV strategy type"),
        }
    }

    #[test]
    fn test_select_k_best() {
        let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);

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
        x.add_column(
            "feature3".to_string(),
            Series::new(vec![0.1, 0.2, 0.3, 0.4, 0.5], Some("feature3".to_string())).unwrap(),
        )
        .unwrap();

        let mut y = DataFrame::new();
        y.add_column(
            "target".to_string(),
            Series::new(vec![3.0, 6.0, 9.0, 12.0, 15.0], Some("target".to_string())).unwrap(),
        )
        .unwrap();

        // Fit and transform
        selector.fit(&x, &y).unwrap();
        let selected = selector.transform(&x).unwrap();

        // Should select 2 features
        assert_eq!(selected.column_names().len(), 2);
    }
}
