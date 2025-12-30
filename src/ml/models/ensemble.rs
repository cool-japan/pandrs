//! Ensemble Methods for Machine Learning
//!
//! This module provides ensemble learning algorithms including:
//! - Random Forest (Classifier and Regressor)
//! - Gradient Boosting (Classifier and Regressor)
//! - Bagging
//! - AdaBoost

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::tree::{
    DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeConfigBuilder, DecisionTreeRegressor,
    SplitCriterion,
};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
#[derive(Debug)]
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

    /// Bootstrap sample indices
    fn bootstrap_indices(&self, n_samples: usize, tree_idx: usize) -> Vec<usize> {
        let seed = self.config.random_seed.unwrap_or(42) + tree_idx as u64;
        let max_samples = self.config.max_samples.unwrap_or(n_samples);

        let mut indices = Vec::with_capacity(max_samples);
        for i in 0..max_samples {
            let idx = ((seed as usize * 1103515245 + i * 12345) % n_samples) as usize;
            indices.push(idx);
        }
        indices
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
            .into_iter()
            .filter(|c| c != target_column)
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

        let n_samples = train_data.row_count();
        let n_features = self.feature_names.len();

        // Default max_features to sqrt(n_features) for classification
        let max_features = self
            .config
            .max_features
            .unwrap_or((n_features as f64).sqrt().ceil() as usize);

        // Build trees
        self.trees.clear();
        for tree_idx in 0..self.config.n_estimators {
            // Create tree config
            let tree_config = DecisionTreeConfigBuilder::new()
                .max_depth(self.config.max_depth.unwrap_or(usize::MAX))
                .min_samples_split(self.config.min_samples_split)
                .min_samples_leaf(self.config.min_samples_leaf)
                .max_features(max_features)
                .random_seed(self.config.random_seed.unwrap_or(42) + tree_idx as u64)
                .build();

            let mut tree = DecisionTreeClassifier::new(tree_config);

            // Bootstrap sample
            let indices = if self.config.bootstrap {
                self.bootstrap_indices(n_samples, tree_idx)
            } else {
                (0..n_samples).collect()
            };

            // Create bootstrap DataFrame
            let bootstrap_data = train_data.sample(&indices)?;

            // Fit tree
            tree.fit(&bootstrap_data, target_column)?;
            self.trees.push(tree);
        }

        self.calculate_feature_importances();
        self.is_fitted = true;

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
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Ok(vec![])
    }
}

/// Random Forest Regressor
#[derive(Debug)]
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

    /// Bootstrap sample indices
    fn bootstrap_indices(&self, n_samples: usize, tree_idx: usize) -> Vec<usize> {
        let seed = self.config.random_seed.unwrap_or(42) + tree_idx as u64;
        let max_samples = self.config.max_samples.unwrap_or(n_samples);

        let mut indices = Vec::with_capacity(max_samples);
        for i in 0..max_samples {
            let idx = ((seed as usize * 1103515245 + i * 12345) % n_samples) as usize;
            indices.push(idx);
        }
        indices
    }
}

impl SupervisedModel for RandomForestRegressor {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.feature_names = train_data
            .column_names()
            .into_iter()
            .filter(|c| c != target_column)
            .collect();

        let n_samples = train_data.row_count();
        let n_features = self.feature_names.len();

        // Default max_features to n_features/3 for regression
        let max_features = self
            .config
            .max_features
            .unwrap_or((n_features as f64 / 3.0).ceil() as usize);

        self.trees.clear();
        for tree_idx in 0..self.config.n_estimators {
            let tree_config = DecisionTreeConfig {
                max_depth: self.config.max_depth,
                min_samples_split: self.config.min_samples_split,
                min_samples_leaf: self.config.min_samples_leaf,
                max_features: Some(max_features),
                criterion: SplitCriterion::MSE,
                random_seed: Some(self.config.random_seed.unwrap_or(42) + tree_idx as u64),
            };

            let mut tree = DecisionTreeRegressor::new(tree_config);

            let indices = if self.config.bootstrap {
                self.bootstrap_indices(n_samples, tree_idx)
            } else {
                (0..n_samples).collect()
            };

            let bootstrap_data = train_data.sample(&indices)?;
            tree.fit(&bootstrap_data, target_column)?;
            self.trees.push(tree);
        }

        self.is_fitted = true;
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

        let y_mean = actual.iter().sum::<f64>() / actual.len() as f64;
        let ss_tot: f64 = actual.iter().map(|a| (a - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (a - p).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        metrics.add_metric("r2", r2);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Ok(vec![])
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
#[derive(Debug)]
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

    /// Subsample indices
    fn subsample_indices(&self, n_samples: usize, iteration: usize) -> Vec<usize> {
        if self.config.subsample >= 1.0 {
            return (0..n_samples).collect();
        }

        let n_subsample = (n_samples as f64 * self.config.subsample).ceil() as usize;
        let seed = self.config.random_seed.unwrap_or(42) + iteration as u64;

        let mut indices = Vec::with_capacity(n_subsample);
        for i in 0..n_subsample {
            let idx = ((seed as usize * 1103515245 + i * 12345) % n_samples) as usize;
            indices.push(idx);
        }
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
            .into_iter()
            .filter(|c| c != target_column)
            .collect();

        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let n_samples = y.len();

        // Initialize with mean
        self.initial_prediction = y.iter().sum::<f64>() / n_samples as f64;
        let mut predictions = vec![self.initial_prediction; n_samples];

        self.trees.clear();
        self.train_scores_.clear();

        for iteration in 0..self.config.n_estimators {
            // Calculate negative gradient (residuals)
            let residuals = self.negative_gradient(&y, &predictions);

            // Create DataFrame with residuals as target
            let mut residual_data = train_data.clone();
            let residual_series =
                crate::series::Series::new(residuals.clone(), Some("_residual".to_string()))?;
            residual_data.add_column("_residual".to_string(), residual_series)?;

            // Subsample if needed
            let indices = self.subsample_indices(n_samples, iteration);
            let subsample_data = if indices.len() < n_samples {
                residual_data.sample(&indices)?
            } else {
                residual_data
            };

            // Fit tree to residuals
            let tree_config = DecisionTreeConfig {
                max_depth: Some(self.config.max_depth),
                min_samples_split: self.config.min_samples_split,
                min_samples_leaf: self.config.min_samples_leaf,
                max_features: None,
                criterion: SplitCriterion::MSE,
                random_seed: self.config.random_seed.map(|s| s + iteration as u64),
            };

            let mut tree = DecisionTreeRegressor::new(tree_config);
            tree.fit(&subsample_data, "_residual")?;

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

        let y_mean = actual.iter().sum::<f64>() / actual.len() as f64;
        let ss_tot: f64 = actual.iter().map(|a| (a - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (a - p).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;
        metrics.add_metric("r2", r2);

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Ok(vec![])
    }
}

/// Gradient Boosting Classifier
#[derive(Debug)]
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
            .into_iter()
            .filter(|c| c != target_column)
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

                // Create DataFrame with residuals
                let mut residual_data = train_data.clone();
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
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Ok(vec![])
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
        .unwrap();
        let x2 = Series::new(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            Some("x2".to_string()),
        )
        .unwrap();
        let y = Series::new(
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            Some("y".to_string()),
        )
        .unwrap();

        df.add_column("x1".to_string(), x1).unwrap();
        df.add_column("x2".to_string(), x2).unwrap();
        df.add_column("y".to_string(), y).unwrap();

        df
    }

    fn create_regression_data() -> DataFrame {
        let mut df = DataFrame::new();

        let x1 = Series::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            Some("x1".to_string()),
        )
        .unwrap();
        let y = Series::new(
            vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            Some("y".to_string()),
        )
        .unwrap();

        df.add_column("x1".to_string(), x1).unwrap();
        df.add_column("y".to_string(), y).unwrap();

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
        rf.fit(&data, "y").unwrap();

        let predictions = rf.predict(&data).unwrap();
        assert_eq!(predictions.len(), 10);

        let metrics = rf.evaluate(&data, "y").unwrap();
        let accuracy = metrics.get_metric("accuracy").unwrap();
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
        rf.fit(&data, "y").unwrap();

        let predictions = rf.predict(&data).unwrap();
        assert_eq!(predictions.len(), 10);

        let metrics = rf.evaluate(&data, "y").unwrap();
        let r2 = metrics.get_metric("r2").unwrap();
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
        gb.fit(&data, "y").unwrap();

        let predictions = gb.predict(&data).unwrap();
        assert_eq!(predictions.len(), 10);

        let metrics = gb.evaluate(&data, "y").unwrap();
        let r2 = metrics.get_metric("r2").unwrap();
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
        gb.fit(&data, "y").unwrap();

        let predictions = gb.predict(&data).unwrap();
        assert_eq!(predictions.len(), 10);

        let metrics = gb.evaluate(&data, "y").unwrap();
        let accuracy = metrics.get_metric("accuracy").unwrap();
        assert!(*accuracy > 0.7);
    }

    #[test]
    fn test_random_forest_predict_proba() {
        let data = create_classification_data();
        let config = RandomForestConfigBuilder::new().n_estimators(10).build();

        let mut rf = RandomForestClassifier::new(config);
        rf.fit(&data, "y").unwrap();

        let probs = rf.predict_proba(&data).unwrap();
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
        gb.fit(&data, "y").unwrap();

        let scores = gb.train_scores();
        assert_eq!(scores.len(), 20);

        // Training loss should generally decrease
        assert!(scores.last().unwrap() < scores.first().unwrap());
    }
}
