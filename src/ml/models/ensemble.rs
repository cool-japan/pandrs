//! Ensemble Methods for Machine Learning
//!
//! This module provides ensemble learning algorithms including:
//! - Random Forest (Classifier and Regressor)
//! - Gradient Boosting (Classifier and Regressor)
//! - Bagging
//! - AdaBoost

use crate::dataframe::DataFrame;
use crate::dataframe::PandasCompatExt;
use crate::error::{Error, Result};
use crate::ml::models::tree::{
    median_of, DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeConfigBuilder,
    DecisionTreeRegressor, SplitCriterion,
};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use rayon::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::RngExt;
use scirs2_core::random::SeedableRng;
use scirs2_core::random::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Build a thread count from a `RandomForestConfig::n_jobs` value (`0` means
/// "use all available cores", matching the field's documented convention).
fn resolve_n_jobs(n_jobs: usize) -> usize {
    if n_jobs == 0 {
        num_cpus::get().max(1)
    } else {
        n_jobs
    }
}

/// Run `build_one(tree_idx)` for every `0..n_estimators`, honoring
/// `n_jobs`: sequentially when `n_jobs == 1` (the default — identical to the
/// pre-parallel behavior, so single-threaded callers see no change), or via
/// a dedicated rayon thread pool sized to `resolve_n_jobs(n_jobs)` when
/// `n_jobs != 1`. Each tree's bootstrap sample and split search already draw
/// from a seed derived independently per `tree_idx`
/// (`config.random_seed.unwrap_or(42).wrapping_add(tree_idx)`), so fitting
/// them concurrently changes only wall-clock time, never which trees get
/// built. Previously `n_jobs` was accepted into the config and never read
/// anywhere.
fn build_trees_honoring_n_jobs<T, F>(
    n_estimators: usize,
    n_jobs: usize,
    build_one: F,
) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(usize) -> Result<T> + Sync + Send,
{
    if n_jobs == 1 {
        return (0..n_estimators).map(build_one).collect();
    }

    let threads = resolve_n_jobs(n_jobs);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| {
            Error::InvalidOperation(format!(
                "Failed to build a {}-thread pool for n_jobs={}: {}",
                threads, n_jobs, e
            ))
        })?;
    pool.install(|| (0..n_estimators).into_par_iter().map(build_one).collect())
}

/// Configuration for Random Forest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestConfig {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Maximum depth of each tree (None = no limit)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required at a leaf node
    pub min_samples_leaf: usize,
    /// Number of features to consider at each split (None = sqrt(n_features))
    pub max_features: Option<usize>,
    /// Whether to bootstrap samples
    pub bootstrap: bool,
    /// Maximum number of samples to use for each tree (None = n_samples)
    pub max_samples: Option<usize>,
    /// Random seed
    pub random_seed: Option<u64>,
    /// Whether to use out-of-bag samples for estimation
    pub oob_score: bool,
    /// Number of parallel jobs (0 = use all cores)
    pub n_jobs: usize,
}

impl Default for RandomForestConfig {
    fn default() -> Self {
        RandomForestConfig {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            bootstrap: true,
            max_samples: None,
            random_seed: None,
            oob_score: false,
            n_jobs: 1,
        }
    }
}

/// Builder for RandomForestConfig
pub struct RandomForestConfigBuilder {
    config: RandomForestConfig,
}

impl RandomForestConfigBuilder {
    pub fn new() -> Self {
        RandomForestConfigBuilder {
            config: RandomForestConfig::default(),
        }
    }

    pub fn n_estimators(mut self, n: usize) -> Self {
        self.config.n_estimators = n;
        self
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = Some(depth);
        self
    }

    pub fn min_samples_split(mut self, samples: usize) -> Self {
        self.config.min_samples_split = samples;
        self
    }

    pub fn min_samples_leaf(mut self, samples: usize) -> Self {
        self.config.min_samples_leaf = samples;
        self
    }

    pub fn max_features(mut self, features: usize) -> Self {
        self.config.max_features = Some(features);
        self
    }

    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }

    pub fn max_samples(mut self, samples: usize) -> Self {
        self.config.max_samples = Some(samples);
        self
    }

    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    pub fn oob_score(mut self, oob: bool) -> Self {
        self.config.oob_score = oob;
        self
    }

    /// Number of parallel jobs to use when fitting trees (`0` = use all
    /// available cores, matching scikit-learn's `n_jobs` convention).
    pub fn n_jobs(mut self, n_jobs: usize) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    pub fn build(self) -> RandomForestConfig {
        self.config
    }
}

impl Default for RandomForestConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Random Forest Classifier
#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    config: RandomForestConfig,
    trees: Vec<DecisionTreeClassifier>,
    feature_names: Vec<String>,
    n_classes: usize,
    classes: Vec<f64>,
    feature_importances_: Option<HashMap<String, f64>>,
    oob_score_: Option<f64>,
    is_fitted: bool,
}

impl RandomForestClassifier {
    /// Create a new random forest classifier
    pub fn new(config: RandomForestConfig) -> Self {
        RandomForestClassifier {
            config,
            trees: Vec::new(),
            feature_names: Vec::new(),
            n_classes: 0,
            classes: Vec::new(),
            feature_importances_: None,
            oob_score_: None,
            is_fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(RandomForestConfig::default())
    }

    /// Get the number of trees
    pub fn n_estimators(&self) -> usize {
        self.trees.len()
    }

    /// Get OOB score
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_
    }

    /// Bootstrap sample indices: `max_samples` (default: `n_samples`) i.i.d.
    /// draws *with replacement* from `0..n_samples`, seeded deterministically
    /// per tree so the whole forest is reproducible from `config.random_seed`
    /// (default base seed `42`, matching the per-tree `DecisionTreeConfig`
    /// seeding already used in `fit`). This is a real bootstrap: each row has
    /// an independent `1/n` chance per draw, so about `1 - (1-1/n)^n ≈ 0.632`
    /// of the distinct rows are expected to appear at least once. Previously
    /// this was an arithmetic progression (`seed*1103515245 + i*12345) % n`)
    /// that produced the same handful of rows for every `i`, and even
    /// overflowed `usize` multiplication for large seeds — there was no
    /// data-level variance between trees at all.
    fn bootstrap_indices(&self, n_samples: usize, tree_idx: usize) -> Vec<usize> {
        if n_samples == 0 {
            return Vec::new();
        }
        let seed = self
            .config
            .random_seed
            .unwrap_or(42)
            .wrapping_add(tree_idx as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let max_samples = self.config.max_samples.unwrap_or(n_samples);
        (0..max_samples)
            .map(|_| rng.random_range(0..n_samples))
            .collect()
    }

    /// Estimate accuracy from out-of-bag predictions: for each training row,
    /// average the class-probability vectors of only the trees whose
    /// bootstrap sample (see [`bootstrap_indices`](Self::bootstrap_indices))
    /// did *not* include that row, then compare the resulting argmax to the
    /// true label. Rows that happened to be in-bag for every tree contribute
    /// no estimate and are excluded from the denominator (matching
    /// scikit-learn's handling of the same situation), rather than being
    /// silently counted as correct or incorrect.
    fn compute_oob_accuracy(&self, train_data: &DataFrame, y: &[f64]) -> Result<f64> {
        let n_samples = y.len();
        let mut in_bag: Vec<HashSet<usize>> = Vec::with_capacity(self.trees.len());
        let mut tree_probs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(self.trees.len());
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            in_bag.push(
                self.bootstrap_indices(n_samples, tree_idx)
                    .into_iter()
                    .collect(),
            );
            tree_probs.push(tree.predict_proba(train_data)?);
        }

        let mut correct = 0usize;
        let mut scored = 0usize;
        for i in 0..n_samples {
            let mut avg_prob = vec![0.0f64; self.n_classes];
            let mut n_oob_trees = 0usize;
            for (tree_idx, probs) in tree_probs.iter().enumerate() {
                if !in_bag[tree_idx].contains(&i) {
                    for (acc, &p) in avg_prob.iter_mut().zip(&probs[i]) {
                        *acc += p;
                    }
                    n_oob_trees += 1;
                }
            }
            if n_oob_trees == 0 {
                continue;
            }
            scored += 1;
            let predicted_idx = avg_prob
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let predicted = self.classes.get(predicted_idx).cloned().unwrap_or(0.0);
            if (predicted - y[i]).abs() < 1e-10 {
                correct += 1;
            }
        }

        if scored == 0 {
            return Err(Error::InvalidOperation(
                "oob_score requested but every training row was in-bag for all trees; \
                 increase n_estimators or decrease max_samples"
                    .to_string(),
            ));
        }
        Ok(correct as f64 / scored as f64)
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        // Collect predictions from all trees
        let mut all_probs: Vec<Vec<Vec<f64>>> = Vec::new();
        for tree in &self.trees {
            let probs = tree.predict_proba(data)?;
            all_probs.push(probs);
        }

        // Average probabilities
        let n_samples = all_probs[0].len();
        let mut avg_probs = vec![vec![0.0; self.n_classes]; n_samples];

        for tree_probs in &all_probs {
            for (i, sample_probs) in tree_probs.iter().enumerate() {
                for (j, &prob) in sample_probs.iter().enumerate() {
                    if j < self.n_classes {
                        avg_probs[i][j] += prob;
                    }
                }
            }
        }

        let n_trees = self.trees.len() as f64;
        for sample_probs in &mut avg_probs {
            for prob in sample_probs.iter_mut() {
                *prob /= n_trees;
            }
        }

        Ok(avg_probs)
    }

    /// Calculate feature importances by averaging across trees
    fn calculate_feature_importances(&mut self) {
        let mut importances: HashMap<String, f64> = HashMap::new();

        for tree in &self.trees {
            if let Some(tree_importances) = tree.feature_importances() {
                for (feature, importance) in tree_importances {
                    *importances.entry(feature).or_insert(0.0) += importance;
                }
            }
        }

        // Average
        let n_trees = self.trees.len() as f64;
        for importance in importances.values_mut() {
            *importance /= n_trees;
        }

        self.feature_importances_ = Some(importances);
    }
}

impl SupervisedModel for RandomForestClassifier {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.feature_names = train_data
            .column_names()
            .iter()
            .filter(|c| c.as_str() != target_column)
            .cloned()
            .collect();

        if self.feature_names.is_empty() {
            return Err(Error::InvalidInput("No feature columns found".to_string()));
        }

        // Get target values to find classes
        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let mut classes: Vec<f64> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
        self.n_classes = self.classes.len();

        if self.config.oob_score && !self.config.bootstrap {
            return Err(Error::InvalidInput(
                "oob_score=true requires bootstrap=true (out-of-bag rows only exist when \
                 trees are fit on bootstrap resamples)"
                    .to_string(),
            ));
        }

        let n_samples = train_data.row_count();
        let n_features = self.feature_names.len();

        // Default max_features to sqrt(n_features) for classification
        let max_features = self
            .config
            .max_features
            .unwrap_or((n_features as f64).sqrt().ceil() as usize);

        // Build trees, honoring `n_jobs` (sequential when 1, a scoped rayon
        // pool otherwise). Each tree's bootstrap draw and split search are
        // seeded solely from `tree_idx`, so building them out of order or
        // concurrently does not change the forest that results.
        let bootstrap = self.config.bootstrap;
        let random_seed_base = self.config.random_seed.unwrap_or(42);
        let max_depth = self.config.max_depth.unwrap_or(usize::MAX);
        let min_samples_split = self.config.min_samples_split;
        let min_samples_leaf = self.config.min_samples_leaf;
        let n_estimators = self.config.n_estimators;
        let n_jobs = self.config.n_jobs;
        self.trees = build_trees_honoring_n_jobs(n_estimators, n_jobs, |tree_idx| {
            let tree_config = DecisionTreeConfigBuilder::new()
                .max_depth(max_depth)
                .min_samples_split(min_samples_split)
                .min_samples_leaf(min_samples_leaf)
                .max_features(max_features)
                .random_seed(random_seed_base.wrapping_add(tree_idx as u64))
                .build();

            let mut tree = DecisionTreeClassifier::new(tree_config);

            let indices = if bootstrap {
                self.bootstrap_indices(n_samples, tree_idx)
            } else {
                (0..n_samples).collect()
            };

            let bootstrap_data = train_data.sample(&indices)?;
            tree.fit(&bootstrap_data, target_column)?;
            Ok(tree)
        })?;

        self.calculate_feature_importances();
        self.is_fitted = true;

        self.oob_score_ = if self.config.oob_score {
            Some(self.compute_oob_accuracy(train_data, &y)?)
        } else {
            None
        };

        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        let probs = self.predict_proba(data)?;

        let predictions: Vec<f64> = probs
            .iter()
            .map(|sample_probs| {
                let max_idx = sample_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.classes.get(max_idx).cloned().unwrap_or(0.0)
            })
            .collect();

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        self.feature_importances_.clone()
    }
}

impl ModelEvaluator for RandomForestClassifier {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        let mut metrics = ModelMetrics::new();

        let correct = predictions
            .iter()
            .zip(&actual)
            .filter(|(p, a)| (*p - *a).abs() < 1e-10)
            .count();
        let accuracy = correct as f64 / predictions.len() as f64;
        metrics.add_metric("accuracy", accuracy);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        crate::ml::models::contiguous_kfold_cross_validate(self, data, target, folds)
    }
}

/// Random Forest Regressor
#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    config: RandomForestConfig,
    trees: Vec<DecisionTreeRegressor>,
    feature_names: Vec<String>,
    feature_importances_: Option<HashMap<String, f64>>,
    oob_score_: Option<f64>,
    is_fitted: bool,
}

impl RandomForestRegressor {
    /// Create a new random forest regressor
    pub fn new(config: RandomForestConfig) -> Self {
        RandomForestRegressor {
            config,
            trees: Vec::new(),
            feature_names: Vec::new(),
            feature_importances_: None,
            oob_score_: None,
            is_fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(RandomForestConfig::default())
    }

    /// Get OOB score (R²), if `config.oob_score` was set before fitting.
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_
    }

    /// Bootstrap sample indices: see
    /// [`RandomForestClassifier::bootstrap_indices`] for the rationale — real
    /// i.i.d. draws with replacement, seeded per tree from `config.random_seed`.
    fn bootstrap_indices(&self, n_samples: usize, tree_idx: usize) -> Vec<usize> {
        if n_samples == 0 {
            return Vec::new();
        }
        let seed = self
            .config
            .random_seed
            .unwrap_or(42)
            .wrapping_add(tree_idx as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let max_samples = self.config.max_samples.unwrap_or(n_samples);
        (0..max_samples)
            .map(|_| rng.random_range(0..n_samples))
            .collect()
    }

    /// Calculate feature importances by averaging across trees (mirrors
    /// [`RandomForestClassifier::calculate_feature_importances`]; the
    /// regressor previously never populated this field at all, so
    /// `feature_importances()` always returned `None` regardless of fit).
    fn calculate_feature_importances(&mut self) {
        let mut importances: HashMap<String, f64> = HashMap::new();

        for tree in &self.trees {
            if let Some(tree_importances) = tree.feature_importances() {
                for (feature, importance) in tree_importances {
                    *importances.entry(feature).or_insert(0.0) += importance;
                }
            }
        }

        let n_trees = self.trees.len() as f64;
        for importance in importances.values_mut() {
            *importance /= n_trees;
        }

        self.feature_importances_ = Some(importances);
    }

    /// Estimate R² from out-of-bag predictions, analogous to
    /// [`RandomForestClassifier::compute_oob_accuracy`]: for each training
    /// row, average the predictions of only the trees that did not see that
    /// row in their bootstrap sample, then score the resulting predictions
    /// against the true targets. Rows in-bag for every tree are excluded.
    fn compute_oob_r2(&self, train_data: &DataFrame, y: &[f64]) -> Result<f64> {
        let n_samples = y.len();
        let mut in_bag: Vec<HashSet<usize>> = Vec::with_capacity(self.trees.len());
        let mut tree_preds: Vec<Vec<f64>> = Vec::with_capacity(self.trees.len());
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            in_bag.push(
                self.bootstrap_indices(n_samples, tree_idx)
                    .into_iter()
                    .collect(),
            );
            tree_preds.push(tree.predict(train_data)?);
        }

        let mut oob_pred: Vec<f64> = Vec::new();
        let mut oob_true: Vec<f64> = Vec::new();
        for i in 0..n_samples {
            let mut sum = 0.0f64;
            let mut n_oob_trees = 0usize;
            for (tree_idx, preds) in tree_preds.iter().enumerate() {
                if !in_bag[tree_idx].contains(&i) {
                    sum += preds[i];
                    n_oob_trees += 1;
                }
            }
            if n_oob_trees == 0 {
                continue;
            }
            oob_pred.push(sum / n_oob_trees as f64);
            oob_true.push(y[i]);
        }

        if oob_true.is_empty() {
            return Err(Error::InvalidOperation(
                "oob_score requested but every training row was in-bag for all trees; \
                 increase n_estimators or decrease max_samples"
                    .to_string(),
            ));
        }

        Ok(crate::ml::models::r2_score_guarded(&oob_pred, &oob_true))
    }
}

impl SupervisedModel for RandomForestRegressor {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.feature_names = train_data
            .column_names()
            .iter()
            .filter(|c| c.as_str() != target_column)
            .cloned()
            .collect();

        if self.config.oob_score && !self.config.bootstrap {
            return Err(Error::InvalidInput(
                "oob_score=true requires bootstrap=true (out-of-bag rows only exist when \
                 trees are fit on bootstrap resamples)"
                    .to_string(),
            ));
        }

        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let n_samples = train_data.row_count();
        let n_features = self.feature_names.len();

        // Default max_features to n_features/3 for regression
        let max_features = self
            .config
            .max_features
            .unwrap_or((n_features as f64 / 3.0).ceil() as usize);

        let bootstrap = self.config.bootstrap;
        let random_seed_base = self.config.random_seed.unwrap_or(42);
        let max_depth = self.config.max_depth;
        let min_samples_split = self.config.min_samples_split;
        let min_samples_leaf = self.config.min_samples_leaf;
        let n_estimators = self.config.n_estimators;
        let n_jobs = self.config.n_jobs;
        self.trees = build_trees_honoring_n_jobs(n_estimators, n_jobs, |tree_idx| {
            let tree_config = DecisionTreeConfig {
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features: Some(max_features),
                criterion: SplitCriterion::MSE,
                random_seed: Some(random_seed_base.wrapping_add(tree_idx as u64)),
            };

            let mut tree = DecisionTreeRegressor::new(tree_config);

            let indices = if bootstrap {
                self.bootstrap_indices(n_samples, tree_idx)
            } else {
                (0..n_samples).collect()
            };

            let bootstrap_data = train_data.sample(&indices)?;
            tree.fit(&bootstrap_data, target_column)?;
            Ok(tree)
        })?;

        self.calculate_feature_importances();
        self.is_fitted = true;

        self.oob_score_ = if self.config.oob_score {
            Some(self.compute_oob_r2(train_data, &y)?)
        } else {
            None
        };

        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        // Collect predictions from all trees
        let mut all_predictions: Vec<Vec<f64>> = Vec::new();
        for tree in &self.trees {
            let preds = tree.predict(data)?;
            all_predictions.push(preds);
        }

        // Average predictions
        let n_samples = all_predictions[0].len();
        let mut avg_predictions = vec![0.0; n_samples];

        for tree_preds in &all_predictions {
            for (i, &pred) in tree_preds.iter().enumerate() {
                avg_predictions[i] += pred;
            }
        }

        let n_trees = self.trees.len() as f64;
        for pred in &mut avg_predictions {
            *pred /= n_trees;
        }

        Ok(avg_predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        self.feature_importances_.clone()
    }
}

impl ModelEvaluator for RandomForestRegressor {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        let mut metrics = ModelMetrics::new();

        let mse = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        metrics.add_metric("mse", mse);
        metrics.add_metric("rmse", mse.sqrt());

        let r2 = crate::ml::models::r2_score_guarded(&predictions, &actual);
        metrics.add_metric("r2", r2);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        crate::ml::models::contiguous_kfold_cross_validate(self, data, target, folds)
    }
}

/// Configuration for Gradient Boosting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingConfig {
    /// Number of boosting stages
    pub n_estimators: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Maximum depth of each tree
    pub max_depth: usize,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required at a leaf node
    pub min_samples_leaf: usize,
    /// Fraction of samples to use for each tree
    pub subsample: f64,
    /// Random seed
    pub random_seed: Option<u64>,
    /// Loss function
    pub loss: GBLoss,
}

/// Loss functions for Gradient Boosting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GBLoss {
    /// Squared error (for regression)
    SquaredError,
    /// Absolute error (for regression)
    AbsoluteError,
    /// Deviance / Log loss (for classification)
    Deviance,
    /// Exponential loss (for classification)
    Exponential,
}

impl Default for GBLoss {
    fn default() -> Self {
        GBLoss::SquaredError
    }
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        GradientBoostingConfig {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            random_seed: None,
            loss: GBLoss::SquaredError,
        }
    }
}

/// Builder for GradientBoostingConfig
pub struct GradientBoostingConfigBuilder {
    config: GradientBoostingConfig,
}

impl GradientBoostingConfigBuilder {
    pub fn new() -> Self {
        GradientBoostingConfigBuilder {
            config: GradientBoostingConfig::default(),
        }
    }

    pub fn n_estimators(mut self, n: usize) -> Self {
        self.config.n_estimators = n;
        self
    }

    pub fn learning_rate(mut self, rate: f64) -> Self {
        self.config.learning_rate = rate;
        self
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.config.max_depth = depth;
        self
    }

    pub fn subsample(mut self, subsample: f64) -> Self {
        self.config.subsample = subsample.clamp(0.0, 1.0);
        self
    }

    pub fn loss(mut self, loss: GBLoss) -> Self {
        self.config.loss = loss;
        self
    }

    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    pub fn build(self) -> GradientBoostingConfig {
        self.config
    }
}

impl Default for GradientBoostingConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient Boosting Regressor
#[derive(Debug, Clone)]
pub struct GradientBoostingRegressor {
    config: GradientBoostingConfig,
    trees: Vec<DecisionTreeRegressor>,
    initial_prediction: f64,
    feature_names: Vec<String>,
    feature_importances_: Option<HashMap<String, f64>>,
    train_scores_: Vec<f64>,
    is_fitted: bool,
}

impl GradientBoostingRegressor {
    /// Create a new gradient boosting regressor
    pub fn new(config: GradientBoostingConfig) -> Self {
        GradientBoostingRegressor {
            config,
            trees: Vec::new(),
            initial_prediction: 0.0,
            feature_names: Vec::new(),
            feature_importances_: None,
            train_scores_: Vec::new(),
            is_fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(GradientBoostingConfig::default())
    }

    /// Get training scores
    pub fn train_scores(&self) -> &[f64] {
        &self.train_scores_
    }

    /// Subsample indices: a real random subset *without* replacement of
    /// `ceil(n_samples * subsample)` rows (stochastic gradient boosting, as
    /// in Friedman 1999 and scikit-learn's `subsample` parameter — unlike a
    /// bootstrap, each boosting iteration should see each row at most once),
    /// reseeded per iteration from `config.random_seed`. Previously this was
    /// the same broken arithmetic-progression generator as the Random Forest
    /// bootstrap (see `RandomForestClassifier::bootstrap_indices`): no real
    /// row-level variance between boosting iterations.
    fn subsample_indices(&self, n_samples: usize, iteration: usize) -> Vec<usize> {
        if self.config.subsample >= 1.0 || n_samples == 0 {
            return (0..n_samples).collect();
        }

        let n_subsample =
            ((n_samples as f64 * self.config.subsample).ceil() as usize).clamp(1, n_samples);
        let seed = self
            .config
            .random_seed
            .unwrap_or(42)
            .wrapping_add(iteration as u64);
        let mut rng = StdRng::seed_from_u64(seed);

        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n_subsample);
        indices
    }

    /// Calculate negative gradient (residuals for squared error)
    fn negative_gradient(&self, y: &[f64], predictions: &[f64]) -> Vec<f64> {
        match self.config.loss {
            GBLoss::SquaredError => y.iter().zip(predictions).map(|(yi, pi)| yi - pi).collect(),
            GBLoss::AbsoluteError => y
                .iter()
                .zip(predictions)
                .map(|(yi, pi)| {
                    if yi > pi {
                        1.0
                    } else if yi < pi {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
            _ => y.iter().zip(predictions).map(|(yi, pi)| yi - pi).collect(),
        }
    }

    /// Calculate loss
    fn calculate_loss(&self, y: &[f64], predictions: &[f64]) -> f64 {
        let n = y.len() as f64;
        match self.config.loss {
            GBLoss::SquaredError => {
                y.iter()
                    .zip(predictions)
                    .map(|(yi, pi)| (yi - pi).powi(2))
                    .sum::<f64>()
                    / n
            }
            GBLoss::AbsoluteError => {
                y.iter()
                    .zip(predictions)
                    .map(|(yi, pi)| (yi - pi).abs())
                    .sum::<f64>()
                    / n
            }
            _ => {
                y.iter()
                    .zip(predictions)
                    .map(|(yi, pi)| (yi - pi).powi(2))
                    .sum::<f64>()
                    / n
            }
        }
    }
}

impl SupervisedModel for GradientBoostingRegressor {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.feature_names = train_data
            .column_names()
            .iter()
            .filter(|c| c.as_str() != target_column)
            .cloned()
            .collect();

        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let n_samples = y.len();

        // Initialize with the constant that minimizes the configured loss:
        // the mean minimizes squared error, but for LAD/absolute-error the
        // minimizer is the median. Using the mean unconditionally for
        // absolute-error was inconsistent with the loss the model claims to
        // optimize (and with the per-leaf medians computed below).
        self.initial_prediction = match self.config.loss {
            GBLoss::AbsoluteError => median_of(&y),
            _ => y.iter().sum::<f64>() / n_samples as f64,
        };
        let mut predictions = vec![self.initial_prediction; n_samples];

        self.trees.clear();
        self.train_scores_.clear();

        let subsampling = self.config.subsample < 1.0;

        for iteration in 0..self.config.n_estimators {
            // Calculate negative gradient (residuals)
            let residuals = self.negative_gradient(&y, &predictions);

            // Create DataFrame with residuals as target. The original target
            // column is dropped first: previously it stayed in `residual_data`
            // and, since `DecisionTreeRegressor::fit` treats every non-target
            // column as a feature, `target_column` itself became a (perfectly
            // predictive) splitting feature for the residual tree — a direct
            // target leak that also made `predict` on unlabeled data fail
            // with a "column not found" error.
            let mut residual_data = train_data.drop_columns(&[target_column])?;
            let residual_series =
                crate::series::Series::new(residuals.clone(), Some("_residual".to_string()))?;
            residual_data.add_column("_residual".to_string(), residual_series)?;

            // Subsample if needed. The fast path is gated on the *config*
            // flag rather than on `indices.len() < n_samples`: `subsample`
            // rounds up (`ceil`), so a subsample fraction just under 1.0 can
            // still produce a full-length (but shuffled) index list, and
            // comparing lengths would then wrongly take the "no subsampling"
            // branch and silently discard the shuffle, desynchronizing
            // `indices` from `subsample_data`'s actual row order.
            let indices = self.subsample_indices(n_samples, iteration);
            let subsample_data = if subsampling {
                residual_data.sample(&indices)?
            } else {
                residual_data
            };

            // Fit tree to residuals. For the LAD/absolute-error loss the
            // target used here (`sign(y - F)`) only determines the split
            // *structure*; the leaf values CART fits from it (means of ±1/0)
            // are meaningless as prediction updates and are overwritten
            // below with the true per-leaf median residual.
            let tree_config = DecisionTreeConfig {
                max_depth: Some(self.config.max_depth),
                min_samples_split: self.config.min_samples_split,
                min_samples_leaf: self.config.min_samples_leaf,
                max_features: None,
                criterion: SplitCriterion::MSE,
                random_seed: self
                    .config
                    .random_seed
                    .map(|s| s.wrapping_add(iteration as u64)),
            };

            let mut tree = DecisionTreeRegressor::new(tree_config);
            tree.fit(&subsample_data, "_residual")?;

            if self.config.loss == GBLoss::AbsoluteError {
                // Terminal-region update: the constant that minimizes
                // absolute-error loss within a leaf is the *median* of the
                // true residuals `y - F_{m-1}` routed to it, not the mean of
                // the sign-valued pseudo-residuals the split search used.
                // `predictions` still holds `F_{m-1}` here (the update loop
                // below runs after this block), and `indices[k]` names the
                // original row that `subsample_data` row `k` came from.
                let leaf_idx_per_row = tree.leaf_indices(&subsample_data)?;
                let mut leaf_residuals: HashMap<usize, Vec<f64>> = HashMap::new();
                for (&row_idx, &leaf_idx) in indices.iter().zip(&leaf_idx_per_row) {
                    leaf_residuals
                        .entry(leaf_idx)
                        .or_default()
                        .push(y[row_idx] - predictions[row_idx]);
                }
                for (leaf_idx, leaf_values) in leaf_residuals {
                    tree.set_leaf_prediction(leaf_idx, median_of(&leaf_values))?;
                }
            }

            // Update predictions
            let tree_predictions = tree.predict(train_data)?;
            for (pred, tree_pred) in predictions.iter_mut().zip(&tree_predictions) {
                *pred += self.config.learning_rate * tree_pred;
            }

            self.trees.push(tree);

            // Calculate and store training loss
            let loss = self.calculate_loss(&y, &predictions);
            self.train_scores_.push(loss);
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        let n_samples = data.row_count();
        let mut predictions = vec![self.initial_prediction; n_samples];

        for tree in &self.trees {
            let tree_preds = tree.predict(data)?;
            for (pred, tree_pred) in predictions.iter_mut().zip(&tree_preds) {
                *pred += self.config.learning_rate * tree_pred;
            }
        }

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        self.feature_importances_.clone()
    }
}

impl ModelEvaluator for GradientBoostingRegressor {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        let mut metrics = ModelMetrics::new();

        let mse = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        metrics.add_metric("mse", mse);
        metrics.add_metric("rmse", mse.sqrt());

        let r2 = crate::ml::models::r2_score_guarded(&predictions, &actual);
        metrics.add_metric("r2", r2);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        crate::ml::models::contiguous_kfold_cross_validate(self, data, target, folds)
    }
}

/// Gradient Boosting Classifier
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier {
    config: GradientBoostingConfig,
    trees: Vec<Vec<DecisionTreeRegressor>>,
    initial_predictions: Vec<f64>,
    feature_names: Vec<String>,
    n_classes: usize,
    classes: Vec<f64>,
    is_fitted: bool,
}

impl GradientBoostingClassifier {
    /// Create a new gradient boosting classifier
    pub fn new(config: GradientBoostingConfig) -> Self {
        let mut config = config;
        config.loss = GBLoss::Deviance;
        GradientBoostingClassifier {
            config,
            trees: Vec::new(),
            initial_predictions: Vec::new(),
            feature_names: Vec::new(),
            n_classes: 0,
            classes: Vec::new(),
            is_fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(GradientBoostingConfig {
            loss: GBLoss::Deviance,
            ..Default::default()
        })
    }

    /// Softmax function
    fn softmax(scores: &[f64]) -> Vec<f64> {
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        exp_scores.iter().map(|s| s / sum).collect()
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        let n_samples = data.row_count();
        let mut scores = vec![self.initial_predictions.clone(); n_samples];

        // Add predictions from all trees
        for (class_idx, class_trees) in self.trees.iter().enumerate() {
            for tree in class_trees {
                let tree_preds = tree.predict(data)?;
                for (i, &pred) in tree_preds.iter().enumerate() {
                    scores[i][class_idx] += self.config.learning_rate * pred;
                }
            }
        }

        // Apply softmax
        let probs: Vec<Vec<f64>> = scores.iter().map(|s| Self::softmax(s)).collect();

        Ok(probs)
    }
}

impl SupervisedModel for GradientBoostingClassifier {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.feature_names = train_data
            .column_names()
            .iter()
            .filter(|c| c.as_str() != target_column)
            .cloned()
            .collect();

        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        // Find classes
        let mut classes: Vec<f64> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
        self.n_classes = self.classes.len();

        let n_samples = y.len();

        // Initialize predictions (uniform distribution in log space)
        let init_pred = 0.0; // log(1/n_classes) for each class
        self.initial_predictions = vec![init_pred; self.n_classes];
        let mut predictions = vec![self.initial_predictions.clone(); n_samples];

        // One-hot encode target
        let y_onehot: Vec<Vec<f64>> = y
            .iter()
            .map(|yi| {
                let mut oh = vec![0.0; self.n_classes];
                if let Some(idx) = self.classes.iter().position(|c| (c - yi).abs() < 1e-10) {
                    oh[idx] = 1.0;
                }
                oh
            })
            .collect();

        // Initialize trees for each class
        self.trees = vec![Vec::new(); self.n_classes];

        for _iteration in 0..self.config.n_estimators {
            // Calculate probabilities
            let probs: Vec<Vec<f64>> = predictions.iter().map(|s| Self::softmax(s)).collect();

            // Fit a tree for each class
            for class_idx in 0..self.n_classes {
                // Calculate residuals (negative gradient)
                let residuals: Vec<f64> = probs
                    .iter()
                    .zip(&y_onehot)
                    .map(|(p, y)| y[class_idx] - p[class_idx])
                    .collect();

                // Create DataFrame with residuals. `target_column` is dropped
                // first -- see the identical fix (and rationale) in
                // `GradientBoostingRegressor::fit` -- otherwise it stays
                // present as an (perfectly predictive) feature column and
                // leaks the label into every per-class residual tree.
                let mut residual_data = train_data.drop_columns(&[target_column])?;
                let residual_series =
                    crate::series::Series::new(residuals.clone(), Some("_residual".to_string()))?;
                residual_data.add_column("_residual".to_string(), residual_series)?;

                // Fit tree
                let tree_config = DecisionTreeConfig {
                    max_depth: Some(self.config.max_depth),
                    min_samples_split: self.config.min_samples_split,
                    min_samples_leaf: self.config.min_samples_leaf,
                    max_features: None,
                    criterion: SplitCriterion::MSE,
                    random_seed: self.config.random_seed,
                };

                let mut tree = DecisionTreeRegressor::new(tree_config);
                tree.fit(&residual_data, "_residual")?;

                // Update predictions
                let tree_preds = tree.predict(train_data)?;
                for (pred, tree_pred) in predictions.iter_mut().zip(&tree_preds) {
                    pred[class_idx] += self.config.learning_rate * tree_pred;
                }

                self.trees[class_idx].push(tree);
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        let probs = self.predict_proba(data)?;

        let predictions: Vec<f64> = probs
            .iter()
            .map(|p| {
                let max_idx = p
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.classes.get(max_idx).cloned().unwrap_or(0.0)
            })
            .collect();

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        None
    }
}

impl ModelEvaluator for GradientBoostingClassifier {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        let mut metrics = ModelMetrics::new();

        let correct = predictions
            .iter()
            .zip(&actual)
            .filter(|(p, a)| (*p - *a).abs() < 1e-10)
            .count();
        let accuracy = correct as f64 / predictions.len() as f64;
        metrics.add_metric("accuracy", accuracy);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        crate::ml::models::contiguous_kfold_cross_validate(self, data, target, folds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    fn create_classification_data() -> DataFrame {
        let mut df = DataFrame::new();

        let x1 = Series::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            Some("x1".to_string()),
        )
        .expect("operation should succeed");
        let x2 = Series::new(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            Some("x2".to_string()),
        )
        .expect("operation should succeed");
        let y = Series::new(
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            Some("y".to_string()),
        )
        .expect("operation should succeed");

        df.add_column("x1".to_string(), x1)
            .expect("operation should succeed");
        df.add_column("x2".to_string(), x2)
            .expect("operation should succeed");
        df.add_column("y".to_string(), y)
            .expect("operation should succeed");

        df
    }

    fn create_regression_data() -> DataFrame {
        let mut df = DataFrame::new();

        let x1 = Series::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            Some("x1".to_string()),
        )
        .expect("operation should succeed");
        let y = Series::new(
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            Some("y".to_string()),
        )
        .expect("operation should succeed");

        df.add_column("x1".to_string(), x1)
            .expect("operation should succeed");
        df.add_column("y".to_string(), y)
            .expect("operation should succeed");

        df
    }

    #[test]
    fn test_random_forest_classifier() {
        let data = create_classification_data();
        let config = RandomForestConfigBuilder::new()
            .n_estimators(10)
            .max_depth(3)
            .build();

        let mut rf = RandomForestClassifier::new(config);
        rf.fit(&data, "y").expect("operation should succeed");

        let predictions = rf.predict(&data).expect("operation should succeed");
        assert_eq!(predictions.len(), 10);

        let metrics = rf.evaluate(&data, "y").expect("operation should succeed");
        let accuracy = metrics
            .get_metric("accuracy")
            .expect("operation should succeed");
        assert!(*accuracy > 0.7);
    }

    #[test]
    fn test_random_forest_regressor() {
        let data = create_regression_data();
        let config = RandomForestConfigBuilder::new()
            .n_estimators(50)
            .max_depth(10)
            .build();

        let mut rf = RandomForestRegressor::new(config);
        rf.fit(&data, "y").expect("operation should succeed");

        let predictions = rf.predict(&data).expect("operation should succeed");
        assert_eq!(predictions.len(), 10);

        let metrics = rf.evaluate(&data, "y").expect("operation should succeed");
        let r2 = metrics.get_metric("r2").expect("operation should succeed");
        // Random forest may not perfectly fit linear data, so use reasonable threshold
        assert!(*r2 > 0.5, "R² should be positive (got {})", r2);
    }

    #[test]
    fn test_gradient_boosting_regressor() {
        let data = create_regression_data();
        let config = GradientBoostingConfigBuilder::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_depth(3)
            .build();

        let mut gb = GradientBoostingRegressor::new(config);
        gb.fit(&data, "y").expect("operation should succeed");

        let predictions = gb.predict(&data).expect("operation should succeed");
        assert_eq!(predictions.len(), 10);

        let metrics = gb.evaluate(&data, "y").expect("operation should succeed");
        let r2 = metrics.get_metric("r2").expect("operation should succeed");
        assert!(*r2 > 0.9);
    }

    #[test]
    fn test_gradient_boosting_classifier() {
        let data = create_classification_data();
        let config = GradientBoostingConfigBuilder::new()
            .n_estimators(20)
            .learning_rate(0.1)
            .max_depth(2)
            .build();

        let mut gb = GradientBoostingClassifier::new(config);
        gb.fit(&data, "y").expect("operation should succeed");

        let predictions = gb.predict(&data).expect("operation should succeed");
        assert_eq!(predictions.len(), 10);

        let metrics = gb.evaluate(&data, "y").expect("operation should succeed");
        let accuracy = metrics
            .get_metric("accuracy")
            .expect("operation should succeed");
        assert!(*accuracy > 0.7);
    }

    #[test]
    fn test_random_forest_predict_proba() {
        let data = create_classification_data();
        let config = RandomForestConfigBuilder::new().n_estimators(10).build();

        let mut rf = RandomForestClassifier::new(config);
        rf.fit(&data, "y").expect("operation should succeed");

        let probs = rf.predict_proba(&data).expect("operation should succeed");
        assert_eq!(probs.len(), 10);

        // Probabilities should sum to 1
        for prob in &probs {
            let sum: f64 = prob.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_gb_training_scores() {
        let data = create_regression_data();
        let config = GradientBoostingConfigBuilder::new()
            .n_estimators(20)
            .build();

        let mut gb = GradientBoostingRegressor::new(config);
        gb.fit(&data, "y").expect("operation should succeed");

        let scores = gb.train_scores();
        assert_eq!(scores.len(), 20);

        // Training loss should generally decrease
        assert!(
            scores.last().expect("operation should succeed")
                < scores.first().expect("operation should succeed")
        );
    }

    // -- Wave-2 regression tests for `bootstrap_indices` -------------------
    //
    // These check a private method directly (white-box), rather than from
    // `tests/ml_ensemble_trees_w2_regression_test.rs`, because the
    // distinct-row-fraction invariant they verify isn't observable through
    // any public API: it's a property of exactly which row indices a single
    // tree's bootstrap draw contains, and `RandomForestClassifier`/
    // `RandomForestRegressor` never expose that. This mirrors the existing
    // precedent of testing `model_selection::compute_cv_fold` directly for
    // the same reason (see `tests/ml_selection_compat_w2_regression_test.rs`).

    /// A real i.i.d.-with-replacement bootstrap of `n` draws from `n` items
    /// is expected to include about `1 - (1-1/n)^n -> 1 - 1/e ≈ 63.2%` of the
    /// distinct rows at least once. The previous "bootstrap" was an
    /// arithmetic progression (`(seed*1103515245 + i*12345) % n`), which had
    /// no such statistical property at all (e.g. the audited case `n=10`
    /// only ever produced rows `{0, 5}`, an ~20% distinct fraction that does
    /// not budge no matter how many draws `i` are taken).
    #[test]
    fn test_classifier_bootstrap_indices_distinct_fraction_matches_theory() {
        let n = 500usize;
        let rf = RandomForestClassifier::new(RandomForestConfig {
            random_seed: Some(7),
            ..RandomForestConfig::default()
        });

        let trials = 60;
        let mut total_fraction = 0.0;
        for tree_idx in 0..trials {
            let idx = rf.bootstrap_indices(n, tree_idx);
            assert_eq!(idx.len(), n, "default max_samples should draw n samples");
            assert!(
                idx.iter().all(|&i| i < n),
                "every drawn index must be a valid row index"
            );
            let distinct: HashSet<usize> = idx.into_iter().collect();
            total_fraction += distinct.len() as f64 / n as f64;
        }
        let avg_fraction = total_fraction / trials as f64;
        let theoretical = 1.0 - std::f64::consts::E.recip();

        assert!(
            (avg_fraction - theoretical).abs() < 0.02,
            "expected ~{:.4} (1 - 1/e) distinct rows from a real bootstrap, got {:.4} \
             averaged over {trials} independent tree_idx draws",
            theoretical,
            avg_fraction
        );
    }

    /// Same statistical property, for `RandomForestRegressor`'s independent
    /// `bootstrap_indices` implementation (the audit flagged this as a
    /// separate fake-bootstrap site from the classifier's).
    #[test]
    fn test_regressor_bootstrap_indices_distinct_fraction_matches_theory() {
        let n = 500usize;
        let rf = RandomForestRegressor::new(RandomForestConfig {
            random_seed: Some(11),
            ..RandomForestConfig::default()
        });

        let trials = 60;
        let mut total_fraction = 0.0;
        for tree_idx in 0..trials {
            let idx = rf.bootstrap_indices(n, tree_idx);
            assert_eq!(idx.len(), n);
            let distinct: HashSet<usize> = idx.into_iter().collect();
            total_fraction += distinct.len() as f64 / n as f64;
        }
        let avg_fraction = total_fraction / trials as f64;
        let theoretical = 1.0 - std::f64::consts::E.recip();

        assert!(
            (avg_fraction - theoretical).abs() < 0.02,
            "expected ~{:.4} distinct rows, got {:.4}",
            theoretical,
            avg_fraction
        );
    }

    /// Different `tree_idx` values must draw genuinely different bootstrap
    /// samples (the concrete "forest trees differ" property at the sampling
    /// level): under the old arithmetic-progression generator, per-tree
    /// index sets were a fixed, low-diversity function of `tree_idx` with no
    /// real independence between trees.
    #[test]
    fn test_bootstrap_indices_vary_across_trees() {
        let n = 200usize;
        let rf = RandomForestClassifier::new(RandomForestConfig {
            random_seed: Some(3),
            ..RandomForestConfig::default()
        });

        let sample_0 = rf.bootstrap_indices(n, 0);
        let sample_1 = rf.bootstrap_indices(n, 1);
        let sample_2 = rf.bootstrap_indices(n, 2);

        assert_ne!(
            sample_0, sample_1,
            "consecutive tree_idx values must not draw identical bootstrap samples"
        );
        assert_ne!(sample_1, sample_2);
        assert_ne!(sample_0, sample_2);
    }
}
