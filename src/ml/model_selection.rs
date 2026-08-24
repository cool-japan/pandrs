//! Model selection and hyperparameter optimization
//!
//! This module provides comprehensive model selection capabilities including
//! grid search, randomized search, cross-validation, and automated feature selection.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::sklearn_compat::SklearnPredictor;
use scirs2_core::random::{Random, StdRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Build a seeded RNG for reproducible-when-requested randomness.
///
/// When `random_state` is `Some(seed)` the returned generator is fully deterministic across
/// runs. When it is `None` a fresh generator is still returned (seeded from OS entropy via
/// [`scirs2_core::random::random`]) so callers always get a *real* shuffle/sample rather than a
/// silently-skipped one — only reproducibility, not randomness itself, depends on the seed.
fn seeded_or_entropy_rng(random_state: Option<u64>) -> StdRng {
    let seed = random_state.unwrap_or_else(|| scirs2_core::random::random::<u64>());
    Random::seed(seed)
}

/// Resolve the target column name in a `y` DataFrame used for cross-validation scoring.
///
/// Prefers a column literally named `"target"`; otherwise falls back to the first (typically
/// only) column. This is shared by [`GridSearchCV`] and [`RandomizedSearchCV`] so both resolve
/// the target column identically instead of one hardcoding `"target"`.
fn resolve_target_column(y: &DataFrame) -> Result<String> {
    if y.has_column("target") {
        return Ok("target".to_string());
    }
    y.column_names()
        .into_iter()
        .next()
        .cloned()
        .ok_or_else(|| Error::InvalidInput("y DataFrame has no columns".into()))
}

/// Compute the `(train_indices, test_indices)` pair for one fold of a cross-validation split.
///
/// Shared by [`GridSearchCV`] and [`RandomizedSearchCV`] so `shuffle`/`random_state` are honored
/// identically everywhere, and so `StratifiedKFold`/`TimeSeriesSplit` each have exactly one
/// correct implementation instead of being duplicated (and drifting) per caller.
fn compute_cv_fold(
    cv: &CrossValidationStrategy,
    n_samples: usize,
    y: &DataFrame,
    fold: usize,
    n_splits: usize,
) -> Result<(Vec<usize>, Vec<usize>)> {
    if n_splits == 0 {
        return Err(Error::InvalidValue(
            "Cross-validation requires n_splits >= 1".into(),
        ));
    }

    match cv {
        CrossValidationStrategy::KFold {
            shuffle,
            random_state,
            ..
        } => {
            if n_splits > n_samples {
                return Err(Error::InvalidValue(format!(
                    "Cannot have n_splits={} greater than n_samples={}",
                    n_splits, n_samples
                )));
            }
            let mut order: Vec<usize> = (0..n_samples).collect();
            if *shuffle {
                let mut rng = seeded_or_entropy_rng(*random_state);
                rng.shuffle(&mut order);
            }
            let fold_size = n_samples / n_splits;
            let test_start = fold * fold_size;
            let test_end = if fold == n_splits - 1 {
                n_samples
            } else {
                test_start + fold_size
            };
            let test_indices: Vec<usize> = order[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = order[..test_start]
                .iter()
                .chain(order[test_end..].iter())
                .copied()
                .collect();
            Ok((train_indices, test_indices))
        }
        CrossValidationStrategy::StratifiedKFold {
            shuffle,
            random_state,
            ..
        } => {
            if n_splits > n_samples {
                return Err(Error::InvalidValue(format!(
                    "Cannot have n_splits={} greater than n_samples={}",
                    n_splits, n_samples
                )));
            }
            let target_name = resolve_target_column(y)?;
            let labels = y.get_column::<f64>(&target_name)?.as_f64()?;
            if labels.len() != n_samples {
                return Err(Error::DimensionMismatch(format!(
                    "y has {} rows but x has {} rows",
                    labels.len(),
                    n_samples
                )));
            }

            // Group sample indices by class label (rounded to the nearest integer), preserving
            // first-seen class order for determinism.
            let mut class_order: Vec<i64> = Vec::new();
            let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();
            for (i, &v) in labels.iter().enumerate() {
                let class = v.round() as i64;
                groups.entry(class).or_insert_with(Vec::new).push(i);
                if !class_order.contains(&class) {
                    class_order.push(class);
                }
            }

            // Optionally shuffle *within* each class before the round-robin fold assignment so
            // repeated fits with `shuffle: false` stay perfectly stable while `shuffle: true`
            // still stratifies (equal per-class proportions in every fold) rather than always
            // handing fold 0 the first-seen rows of each class.
            if *shuffle {
                let mut rng = seeded_or_entropy_rng(*random_state);
                for class in &class_order {
                    if let Some(members) = groups.get_mut(class) {
                        rng.shuffle(members);
                    }
                }
            }

            // Round-robin: the i-th member of each class goes to fold (i % n_splits). This keeps
            // each fold's class proportions close to the overall proportions (true
            // stratification), unlike plain contiguous slicing which can hand an entire fold a
            // single class when the input is class-sorted.
            let mut test_indices = Vec::new();
            for class in &class_order {
                if let Some(members) = groups.get(class) {
                    for (i, &idx) in members.iter().enumerate() {
                        if i % n_splits == fold {
                            test_indices.push(idx);
                        }
                    }
                }
            }
            test_indices.sort_unstable();
            let test_set: std::collections::HashSet<usize> = test_indices.iter().copied().collect();
            let train_indices: Vec<usize> =
                (0..n_samples).filter(|i| !test_set.contains(i)).collect();
            Ok((train_indices, test_indices))
        }
        CrossValidationStrategy::LeaveOneOut => {
            if fold >= n_samples {
                return Err(Error::InvalidValue(
                    "fold index out of range for LeaveOneOut".into(),
                ));
            }
            let test_indices = vec![fold];
            let train_indices: Vec<usize> = (0..n_samples).filter(|&i| i != fold).collect();
            Ok((train_indices, test_indices))
        }
        CrossValidationStrategy::TimeSeriesSplit { max_train_size, .. } => {
            // Expanding-window split: fold `k`'s test block is the (k+1)-th contiguous chunk of
            // the series; train is *strictly* everything before it (never future rows), so there
            // is no look-ahead leakage. `max_train_size`, when set, caps training to the most
            // recent rows instead of always using the full history.
            if n_splits == 0 || n_splits >= n_samples {
                return Err(Error::InvalidValue(format!(
                    "TimeSeriesSplit requires 0 < n_splits < n_samples (got n_splits={}, n_samples={})",
                    n_splits, n_samples
                )));
            }
            let test_fold_size = n_samples / (n_splits + 1);
            if test_fold_size == 0 {
                return Err(Error::InvalidValue(
                    "TimeSeriesSplit: not enough samples for the requested n_splits".into(),
                ));
            }
            let test_start = test_fold_size * (fold + 1);
            let test_end = if fold == n_splits - 1 {
                n_samples
            } else {
                test_fold_size * (fold + 2)
            };
            let train_start = match max_train_size {
                Some(max) => test_start.saturating_sub(*max),
                None => 0,
            };
            let train_indices: Vec<usize> = (train_start..test_start).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();
            Ok((train_indices, test_indices))
        }
    }
}

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
                    // Constant target: mirror metrics/regression.rs's r2_score — a perfect
                    // (zero-residual) prediction still scores 1.0, but a constant target with
                    // ANY residual error scores 0.0 rather than being fabricated as a perfect
                    // 1.0 regardless of how wrong the predictions are.
                    if ss_res == 0.0 {
                        1.0
                    } else {
                        0.0
                    }
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
                // AUC via rank-sum / Mann-Whitney U statistic (O(n log n), handles ties).
                //
                // Algorithm:
                //   1. Sort by predicted score descending.
                //   2. Process the sorted list in *tied groups* (same score value).
                //      Within a tied group, every positive–negative pair contributes 0.5
                //      (expected rank for tied items), while positives ahead of negatives
                //      contribute 1.0 and negatives ahead of positives contribute 0.0.
                //   3. AUC = total_contribution / (n_pos * n_neg).

                let mut sorted_pairs: Vec<(f64, f64)> = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_p, y_t))
                    .collect();
                sorted_pairs
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                let n_pos: f64 = y_true.iter().filter(|&&y| y > 0.5).count() as f64;
                let n_neg = y_true.len() as f64 - n_pos;

                // Degenerate case: only one class present. sklearn's `roc_auc_score` raises
                // `ValueError` here because AUC is genuinely undefined with a single class —
                // returning a fabricated 0.5 would silently misreport a fold as "random" instead
                // of surfacing that this fold cannot be scored with ROC AUC at all (e.g. an
                // unstratified split of imbalanced data handing one fold a single class).
                if n_pos == 0.0 || n_neg == 0.0 {
                    return Err(Error::InvalidValue(
                        "RocAuc is undefined when y_true contains only one class".into(),
                    ));
                }

                // Walk sorted pairs, processing tied score groups together.
                // For each group: n_pos_before * n_neg_group + 0.5 * n_pos_group * n_neg_group
                let mut auc = 0.0f64;
                let mut positives_before = 0.0f64; // positives in all preceding (higher) groups
                let n = sorted_pairs.len();
                let mut i = 0usize;
                while i < n {
                    let current_score = sorted_pairs[i].0;
                    // Collect all items in this tied group
                    let group_start = i;
                    while i < n
                        && (sorted_pairs[i].0 - current_score).abs()
                            < f64::EPSILON * current_score.abs().max(1.0)
                    {
                        i += 1;
                    }
                    // Count positives and negatives in this group
                    let group_pos: f64 = sorted_pairs[group_start..i]
                        .iter()
                        .filter(|&&(_, label)| label > 0.5)
                        .count() as f64;
                    let group_neg: f64 = sorted_pairs[group_start..i]
                        .iter()
                        .filter(|&&(_, label)| label <= 0.5)
                        .count() as f64;

                    // Each negative in this group is beaten by all positives before this group,
                    // and tied with all positives in this same group (contributes 0.5 each).
                    auc += positives_before * group_neg + 0.5 * group_pos * group_neg;
                    positives_before += group_pos;
                }
                Ok(auc / (n_pos * n_neg))
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
    /// Sample a value from this distribution using the given random generator.
    ///
    /// Taking `rng` as a parameter (rather than reaching for a fresh thread-local generator on
    /// every call) lets [`RandomizedSearchCV`] thread a single seeded generator through an
    /// entire search so that `random_state` actually makes the sampled trials reproducible.
    ///
    /// Returns an error for [`ParameterDistribution::LogUniform`] bounds that are not both
    /// strictly positive (`ln` of a non-positive number is undefined/NaN, which would otherwise
    /// silently propagate into a `"NaN"` parameter value).
    pub fn sample(&self, rng: &mut StdRng) -> Result<String> {
        match self {
            ParameterDistribution::UniformInt { low, high } => {
                Ok(rng.random_range(*low..=*high).to_string())
            }
            ParameterDistribution::UniformFloat { low, high } => {
                Ok(rng.random_range(*low..=*high).to_string())
            }
            ParameterDistribution::LogUniform { low, high } => {
                if !(*low > 0.0) || !(*high > 0.0) {
                    return Err(Error::InvalidValue(format!(
                        "LogUniform requires low > 0 and high > 0, got low={}, high={}",
                        low, high
                    )));
                }
                let log_low = low.ln();
                let log_high = high.ln();
                let log_val = rng.random_range(log_low..=log_high);
                Ok(log_val.exp().to_string())
            }
            ParameterDistribution::Choice(choices) => {
                if choices.is_empty() {
                    Err(Error::InvalidValue(
                        "Choice distribution has no options to sample from".into(),
                    ))
                } else {
                    let idx = rng.random_range(0..choices.len());
                    Ok(choices[idx].clone())
                }
            }
            ParameterDistribution::Normal { mean, std } => {
                let u1: f64 = rng.random_range(1e-300_f64..1.0_f64);
                let u2: f64 = rng.random_f64_raw();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                Ok((mean + std * z).to_string())
            }
            ParameterDistribution::Fixed(value) => Ok(value.clone()),
        }
    }
}

/// Results from grid search or randomized search
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// Best parameters found
    pub best_params_: HashMap<String, String>,
    /// Best cross-validation score, or `None` if no parameter combination produced a finite
    /// score (e.g. every trial errored or every fold was degenerate) — never a fabricated
    /// placeholder value.
    pub best_score_: Option<f64>,
    /// Textual description (parameters) of the estimator refit on the full dataset.
    ///
    /// This is a human-readable summary; the live fitted estimator itself is available via
    /// [`GridSearchCV::best_estimator`] / [`RandomizedSearchCV::best_estimator`]. It is `None`
    /// only when `refit` is disabled or no valid parameter combination was found.
    pub best_estimator_: Option<String>,
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
    /// Best estimator refit on the full dataset (present after `fit` when `refit` is true)
    best_estimator_: Option<Box<dyn SklearnPredictor + Send + Sync>>,
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
            best_estimator_: None,
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

    /// Generate all parameter combinations from grid, in a deterministic order.
    ///
    /// Parameter names are visited in sorted order rather than `HashMap`'s unspecified
    /// iteration order, so the resulting combination list — and therefore `cv_results_`'s order
    /// and any score ties broken by "first seen" — is reproducible across runs instead of
    /// depending on hash-map iteration order.
    fn generate_param_combinations(&self) -> Vec<HashMap<String, String>> {
        let mut combinations = vec![HashMap::new()];

        let mut param_names: Vec<&String> = self.param_grid.keys().collect();
        param_names.sort();

        for param_name in param_names {
            let param_values = &self.param_grid[param_name];
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

        // `n_splits == 0` (an explicit `KFold { n_splits: 0, .. }`, or `LeaveOneOut` on an
        // empty `x`) would otherwise skip the fold loop below entirely, leaving `fold_scores`
        // empty and computing `0.0 / 0.0 = NaN` as the "mean" score — a silently fabricated
        // result stored straight into `cv_results_` rather than a clear error. Fail fast
        // instead, matching `compute_cv_fold`'s own "n_splits >= 1" requirement (which this
        // guard front-runs, since the fold loop would otherwise never call it at all).
        if n_splits == 0 {
            return Err(Error::InvalidValue(
                "Cross-validation requires at least one fold (n_splits >= 1 for \
                 KFold/StratifiedKFold/TimeSeriesSplit; LeaveOneOut requires a non-empty \
                 dataset); got 0 folds"
                    .into(),
            ));
        }

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

            // Extract true values (resolves "target" when present, else the first column —
            // consistent with RandomizedSearchCV instead of hardcoding "target").
            let target_name = resolve_target_column(&test_y)?;
            let y_col = test_y.get_column::<f64>(&target_name)?;
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

    /// Generate train/test split for a specific fold.
    ///
    /// Delegates to [`compute_cv_fold`] so `shuffle`/`random_state` are honored and
    /// `StratifiedKFold`/`TimeSeriesSplit` are handled correctly instead of being treated as
    /// plain contiguous `KFold` slicing.
    fn generate_fold_split(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        fold: usize,
        n_splits: usize,
    ) -> Result<(DataFrame, DataFrame, DataFrame, DataFrame)> {
        let (train_indices, test_indices) =
            compute_cv_fold(&self.cv, x.nrows(), y, fold, n_splits)?;

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
        cv_results.sort_by(|a, b| {
            b.mean_test_score
                .partial_cmp(&a.mean_test_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, result) in cv_results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        // Refit the best parameters on the full dataset and keep the real fitted estimator.
        let mut best_estimator_desc = None;
        if self.refit && best_score.is_finite() {
            let mut estimator = self.create_estimator_clone();
            estimator.set_params(best_params.clone())?;
            estimator.fit(x, y)?;
            best_estimator_desc = Some(format!("{:?}", estimator.get_params()));
            self.best_estimator_ = Some(estimator);
        }

        // Store results. `best_score_` is `None` (never a fabricated 0.0) when no combination
        // produced a finite score.
        self.results_ = Some(SearchResults {
            best_params_: best_params,
            best_score_: if best_score.is_finite() {
                Some(best_score)
            } else {
                None
            },
            best_estimator_: best_estimator_desc,
            cv_results_: cv_results,
        });

        if self.verbose > 0 {
            println!("Best score: {:.4}", best_score);
            if let Some(results) = self.results_.as_ref() {
                println!("Best parameters: {:?}", results.best_params_);
            }
        }

        Ok(())
    }

    /// Get the search results
    pub fn get_results(&self) -> Option<&SearchResults> {
        self.results_.as_ref()
    }

    /// Get the best estimator, refit on the full dataset (available when `refit` is true).
    pub fn best_estimator(&self) -> Option<&(dyn SklearnPredictor + Send + Sync)> {
        self.best_estimator_.as_deref()
    }

    /// Predict using the best estimator refit on the full dataset.
    pub fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        let estimator = self.best_estimator_.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "No fitted best estimator available; call fit() with refit enabled first".into(),
            )
        })?;
        estimator.predict(x)
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
    /// Best estimator refit on the full dataset (present after `fit` when `refit` is true)
    best_estimator_: Option<Box<dyn SklearnPredictor + Send + Sync>>,
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
            best_estimator_: None,
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

    /// Generate random parameter combinations by sampling from `param_distributions`.
    ///
    /// Threads a single RNG (seeded from `self.random_state` when set) through every sample so
    /// that a given `random_state` reproduces the exact same trials run after run — previously
    /// each `ParameterDistribution::sample()` call reached for an unseeded thread-local
    /// generator, so `random_state` had no effect at all.
    ///
    /// Parameter names are visited in sorted order (not `HashMap`'s unspecified order) so the
    /// sequence of RNG draws — and therefore the sampled combinations for a given seed — is
    /// reproducible independent of hash-map iteration order.
    fn generate_random_params(&self, rng: &mut StdRng) -> Result<Vec<HashMap<String, String>>> {
        let mut combinations = Vec::with_capacity(self.n_iter);

        let mut param_names: Vec<&String> = self.param_distributions.keys().collect();
        param_names.sort();

        for _ in 0..self.n_iter {
            let mut params = HashMap::new();

            for param_name in &param_names {
                let distribution = &self.param_distributions[*param_name];
                let value = distribution.sample(rng)?;
                params.insert((*param_name).clone(), value);
            }

            combinations.push(params);
        }

        Ok(combinations)
    }

    /// Fit the randomized search
    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let mut sampling_rng = seeded_or_entropy_rng(self.random_state);
        let param_combinations = self.generate_random_params(&mut sampling_rng)?;

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

        let n_splits = match &self.cv {
            CrossValidationStrategy::KFold { n_splits, .. } => *n_splits,
            CrossValidationStrategy::StratifiedKFold { n_splits, .. } => *n_splits,
            CrossValidationStrategy::LeaveOneOut => x.nrows(),
            CrossValidationStrategy::TimeSeriesSplit { n_splits, .. } => *n_splits,
        };

        let mut cv_results = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_params = HashMap::new();

        for (combo_idx, params) in param_combinations.iter().enumerate() {
            let mut fold_scores = Vec::new();
            let mut fit_times = Vec::new();
            let mut score_times = Vec::new();

            for fold in 0..n_splits {
                // Delegate to the same fold-index logic GridSearchCV uses so `shuffle` /
                // `random_state` / `StratifiedKFold` / `TimeSeriesSplit` behave identically
                // everywhere. Randomized search stays lenient (skips a fold that can't be
                // formed, e.g. n_splits > n_samples) rather than failing the whole trial.
                let (train_indices, test_indices) =
                    match compute_cv_fold(&self.cv, x.nrows(), y, fold, n_splits) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                if train_indices.len() < 2 || test_indices.is_empty() {
                    continue;
                }

                let train_x = x.sample(&train_indices)?;
                let test_x = x.sample(&test_indices)?;
                let train_y = y.sample(&train_indices)?;
                let test_y = y.sample(&test_indices)?;

                let mut estimator_clone = self.create_estimator_clone();
                if estimator_clone.set_params(params.clone()).is_err() {
                    continue;
                }

                let fit_start = Instant::now();
                if estimator_clone.fit(&train_x, &train_y).is_err() {
                    continue;
                }
                let fit_time = fit_start.elapsed().as_secs_f64();

                let score_start = Instant::now();
                let predictions = match estimator_clone.predict(&test_x) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let score_time = score_start.elapsed().as_secs_f64();

                let target_col_name = match resolve_target_column(&test_y) {
                    Ok(name) => name,
                    Err(_) => continue,
                };
                if let Ok(y_col) = test_y.get_column::<f64>(&target_col_name) {
                    if let Ok(y_true) = y_col.as_f64() {
                        if let Ok(score) = self.scoring.score(&y_true, &predictions) {
                            fold_scores.push(score);
                            fit_times.push(fit_time);
                            score_times.push(score_time);
                        }
                    }
                }
            }

            if fold_scores.is_empty() {
                continue;
            }

            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            let variance = fold_scores
                .iter()
                .map(|&s| (s - mean_score).powi(2))
                .sum::<f64>()
                / fold_scores.len() as f64;
            let std_score = variance.sqrt();
            let mean_fit_time = fit_times.iter().sum::<f64>() / fit_times.len() as f64;
            let mean_score_time = score_times.iter().sum::<f64>() / score_times.len() as f64;

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
                rank: combo_idx + 1,
            });
        }

        // Sort by score and assign ranks
        cv_results.sort_by(|a, b| {
            b.mean_test_score
                .partial_cmp(&a.mean_test_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, entry) in cv_results.iter_mut().enumerate() {
            entry.rank = i + 1;
        }

        // Refit the best parameters on the full dataset and keep the real fitted estimator.
        let mut best_estimator_desc = None;
        if self.refit && best_score.is_finite() {
            let mut estimator = self.create_estimator_clone();
            estimator.set_params(best_params.clone())?;
            estimator.fit(x, y)?;
            best_estimator_desc = Some(format!("{:?}", estimator.get_params()));
            self.best_estimator_ = Some(estimator);
        }

        // `best_score_` is `None` (never a fabricated 0.0) when no trial produced a finite score.
        self.results_ = Some(SearchResults {
            best_params_: best_params,
            best_score_: if best_score.is_finite() {
                Some(best_score)
            } else {
                None
            },
            best_estimator_: best_estimator_desc,
            cv_results_: cv_results,
        });

        Ok(())
    }

    /// Create a clone of the estimator for cross-validation
    fn create_estimator_clone(&self) -> Box<dyn SklearnPredictor + Send + Sync> {
        self.estimator.clone_predictor()
    }

    /// Get the search results
    pub fn get_results(&self) -> Option<&SearchResults> {
        self.results_.as_ref()
    }

    /// Get the best estimator, refit on the full dataset (available when `refit` is true).
    pub fn best_estimator(&self) -> Option<&(dyn SklearnPredictor + Send + Sync)> {
        self.best_estimator_.as_deref()
    }

    /// Predict using the best estimator refit on the full dataset.
    pub fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        let estimator = self.best_estimator_.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "No fitted best estimator available; call fit() with refit enabled first".into(),
            )
        })?;
        estimator.predict(x)
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

        feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep the *selected* indices in ascending (original-column) order — matching
        // scikit-learn's `SelectKBest.transform`, which preserves each surviving feature's
        // original position rather than reordering columns by score rank.
        let mut selected_features: Vec<usize> = feature_scores
            .iter()
            .take(self.k)
            .map(|(i, _)| *i)
            .collect();
        selected_features.sort_unstable();

        self.scores_ = Some(scores);
        self.selected_features_ = Some(selected_features);
        self.feature_names_ = Some(feature_names.to_vec());

        Ok(())
    }

    /// Transform data by selecting top k features (in their original column order).
    pub fn transform(&self, x: &DataFrame) -> Result<DataFrame> {
        let selected_features = self.selected_features_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("SelectKBest must be fitted before transform".into())
        })?;
        let fitted_feature_names = self.feature_names_.as_ref().ok_or_else(|| {
            Error::InvalidOperation("SelectKBest must be fitted before transform".into())
        })?;

        // `selected_features_` holds *positional* indices captured at fit time. If `x`'s
        // columns don't match what was fitted (different count, names, or order), those
        // positions would silently address the wrong columns — so require an exact match
        // instead of transforming garbage.
        let feature_names: Vec<String> = x.column_names().to_vec();
        if &feature_names != fitted_feature_names {
            return Err(Error::InvalidValue(format!(
                "SelectKBest::transform: input columns {:?} do not match the columns seen \
                 during fit {:?}",
                feature_names, fitted_feature_names
            )));
        }

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

    /// Calculate the univariate linear-regression F-statistic for each feature.
    ///
    /// This is the genuine `f_regression` statistic (as in scikit-learn): for each feature the
    /// Pearson correlation `r` with the target is converted to an F-value via
    ///
    /// ```text
    /// F = r² / (1 - r²) · (n - 2)
    /// ```
    ///
    /// where `n - 2` is the residual degrees of freedom. Larger `F` means a stronger linear
    /// relationship. A perfect fit (`r² → 1`) yields `+∞`, and fewer than three samples yields a
    /// zero score (the statistic is undefined).
    fn f_regression_scores(&self, x: &DataFrame, y: &DataFrame) -> Result<Vec<f64>> {
        let feature_names = x.column_names();
        let mut scores = Vec::with_capacity(feature_names.len());

        // Resolve the target column (first column of y, falling back to "target").
        let target_name = y
            .column_names()
            .iter()
            .find(|name| name.as_str() == "target")
            .or_else(|| y.column_names().iter().next())
            .cloned()
            .ok_or_else(|| Error::InvalidInput("y DataFrame has no columns".into()))?;
        let target_col = y.get_column::<f64>(&target_name)?;
        let target_values = target_col.as_f64()?;

        for feature_name in feature_names {
            let feature_col = x.get_column::<f64>(feature_name)?;
            let feature_values = feature_col.as_f64()?;

            let n = feature_values.len();
            if n < 3 {
                // Degrees of freedom (n - 2) must be positive for the statistic to exist.
                scores.push(0.0);
                continue;
            }

            let correlation = self.calculate_correlation(&feature_values, &target_values)?;
            let r_squared = correlation * correlation;
            let degrees_of_freedom = n as f64 - 2.0;

            let f_statistic = if r_squared >= 1.0 {
                f64::INFINITY
            } else {
                (r_squared / (1.0 - r_squared)) * degrees_of_freedom
            };
            scores.push(f_statistic);
        }

        Ok(scores)
    }

    /// Calculate Chi-square scores between each feature and the target using contingency tables.
    ///
    /// Features are binned into equal-width bins; the chi-square statistic measures how
    /// non-uniform the distribution of classes is across bins.
    fn chi2_scores(&self, x: &DataFrame, y: &DataFrame) -> Result<Vec<f64>> {
        // Obtain target column (first column of y)
        let target_name = y
            .column_names()
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidInput("y DataFrame has no columns".into()))?;
        let target_col = y
            .get_column::<f64>(&target_name)
            .map_err(|_| Error::InvalidInput("Target column must be numeric".into()))?;
        let target_vals = target_col
            .as_f64()
            .map_err(|_| Error::InvalidInput("Target values must be numeric".into()))?;

        // Collect unique integer class labels (round to nearest int)
        let mut classes: Vec<i64> = target_vals
            .iter()
            .map(|&v| v.round() as i64)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();
        let n_classes = classes.len();

        let feature_names = x.column_names();
        let mut scores = Vec::with_capacity(feature_names.len());

        for feat_name in feature_names {
            let feat_col = match x.get_column::<f64>(feat_name) {
                Ok(c) => c,
                Err(_) => {
                    scores.push(0.0);
                    continue;
                }
            };
            let feat_vals = feat_col.as_f64()?;
            let n = feat_vals.len();

            // Determine bin boundaries (5 equal-width bins)
            let n_bins = 5usize;
            let feat_min = feat_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let feat_max = feat_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = feat_max - feat_min;
            let bin_width = if range.abs() < 1e-10 {
                1.0
            } else {
                range / n_bins as f64
            };

            // Build contingency table: rows = bins, cols = classes
            let mut contingency = vec![vec![0usize; n_classes]; n_bins];
            let mut row_totals = vec![0usize; n_bins];
            let mut col_totals = vec![0usize; n_classes];

            for (&fv, &tv) in feat_vals.iter().zip(target_vals.iter()) {
                let bin_idx = if range.abs() < 1e-10 {
                    0
                } else {
                    (((fv - feat_min) / bin_width) as usize).min(n_bins - 1)
                };
                let class_idx = classes.binary_search(&(tv.round() as i64)).unwrap_or(0);
                contingency[bin_idx][class_idx] += 1;
                row_totals[bin_idx] += 1;
                col_totals[class_idx] += 1;
            }

            // χ² = Σ (O - E)² / E,  E = row_total * col_total / n
            let mut chi2 = 0.0f64;
            let n_f = n as f64;
            for r in 0..n_bins {
                for c in 0..n_classes {
                    let observed = contingency[r][c] as f64;
                    let expected = (row_totals[r] as f64 * col_totals[c] as f64) / n_f.max(1.0);
                    let e = expected.max(1e-10);
                    chi2 += (observed - e).powi(2) / e;
                }
            }
            scores.push(chi2);
        }

        Ok(scores)
    }

    /// Calculate mutual information I(X; Y) via histogram-based estimation.
    ///
    /// Uses equal-width bins for both X and Y.  MI is guaranteed non-negative by
    /// clamping the result to 0.
    fn mutual_info_scores(&self, x: &DataFrame, y: &DataFrame) -> Result<Vec<f64>> {
        // Obtain target column
        let target_name = y
            .column_names()
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidInput("y DataFrame has no columns".into()))?;
        let target_col = y
            .get_column::<f64>(&target_name)
            .map_err(|_| Error::InvalidInput("Target column must be numeric".into()))?;
        let target_vals = target_col
            .as_f64()
            .map_err(|_| Error::InvalidInput("Target values must be numeric".into()))?;
        let n = target_vals.len();

        // Discretize Y into bins: prefer unique values when there are few, else sqrt(n) bins
        let y_min = target_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = target_vals
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let y_range = y_max - y_min;

        let n_y_bins = if y_range.abs() < 1e-10 {
            1usize
        } else {
            let n_unique = target_vals
                .iter()
                .map(|&v| (v * 1000.0) as i64)
                .collect::<std::collections::HashSet<_>>()
                .len();
            if n_unique <= 10 {
                n_unique
            } else {
                ((n as f64).sqrt() as usize).max(2)
            }
        };
        let y_bin_width = if n_y_bins == 1 || y_range.abs() < 1e-10 {
            1.0
        } else {
            y_range / n_y_bins as f64
        };

        let y_bins: Vec<usize> = target_vals
            .iter()
            .map(|&v| {
                if n_y_bins == 1 {
                    0
                } else {
                    (((v - y_min) / y_bin_width) as usize).min(n_y_bins - 1)
                }
            })
            .collect();

        // Number of X bins: max(5, floor(sqrt(n)))
        let n_x_bins = ((n as f64).sqrt() as usize).max(5);

        let feature_names = x.column_names();
        let mut scores = Vec::with_capacity(feature_names.len());

        for feat_name in feature_names {
            let feat_col = match x.get_column::<f64>(feat_name) {
                Ok(c) => c,
                Err(_) => {
                    scores.push(0.0);
                    continue;
                }
            };
            let feat_vals = feat_col.as_f64()?;
            let x_min = feat_vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let x_max = feat_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let x_range = x_max - x_min;
            let x_bin_width = if x_range.abs() < 1e-10 {
                1.0
            } else {
                x_range / n_x_bins as f64
            };

            let x_bins_vec: Vec<usize> = feat_vals
                .iter()
                .map(|&v| {
                    if x_range.abs() < 1e-10 {
                        0
                    } else {
                        (((v - x_min) / x_bin_width) as usize).min(n_x_bins - 1)
                    }
                })
                .collect();

            // Accumulate joint and marginal counts
            let mut joint = vec![0u64; n_x_bins * n_y_bins];
            let mut px = vec![0u64; n_x_bins];
            let mut py = vec![0u64; n_y_bins];

            for (&xi, &yi) in x_bins_vec.iter().zip(y_bins.iter()) {
                joint[xi * n_y_bins + yi] += 1;
                px[xi] += 1;
                py[yi] += 1;
            }

            // MI = Σ P(x,y) * ln( P(x,y) / (P(x)*P(y)) )
            let n_f = n as f64;
            let mut mi = 0.0f64;
            for xi in 0..n_x_bins {
                for yi in 0..n_y_bins {
                    let count_xy = joint[xi * n_y_bins + yi];
                    if count_xy == 0 {
                        continue;
                    }
                    let pxy = count_xy as f64 / n_f;
                    let px_v = px[xi] as f64 / n_f;
                    let py_v = py[yi] as f64 / n_f;
                    mi += pxy * (pxy / (px_v * py_v)).ln();
                }
            }
            // MI is theoretically non-negative; clamp floating-point noise
            scores.push(mi.max(0.0));
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

// The unit test module for this file lives in `model_selection_tests.rs` (kept as a separate
// physical file, still compiled as a child `mod tests` with full access to private items via
// `use super::*;`) so this file itself stays under the project's 2000-line-per-file limit. This
// mirrors the existing `#[path]` convention used by e.g. `dataframe/base.rs` +
// `dataframe/base_tests.rs`.
#[cfg(test)]
#[path = "model_selection_tests.rs"]
mod tests;
