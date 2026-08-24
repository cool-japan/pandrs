//! Decision Tree implementation
//!
//! This module provides a decision tree classifier and regressor using
//! the CART (Classification and Regression Trees) algorithm.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::random::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Build a seeded RNG when `seed` is given, otherwise one seeded from the
/// system entropy source — matching the pattern established at
/// `stats::sampling::seeded_rng` / `optimized::split_dataframe::row_ops::sample_rows`.
fn seeded_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed_val) => StdRng::seed_from_u64(seed_val),
        None => {
            let mut seed_bytes = [0u8; 32];
            scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
            StdRng::from_seed(seed_bytes)
        }
    }
}

/// Median of a slice of values (sorts a scratch copy; does not mutate `values`).
/// Used for the MAE splitting criterion, whose impurity- and leaf-minimizing
/// constant is the median rather than the mean. `pub(crate)` so the gradient
/// boosting ensemble (`models::ensemble`) can reuse it for LAD/absolute-error
/// terminal-region updates and initial-prediction computation, which are
/// minimized by the same statistic.
pub(crate) fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Criterion for splitting nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitCriterion {
    /// Gini impurity (for classification)
    Gini,
    /// Entropy / Information Gain (for classification)
    Entropy,
    /// Mean Squared Error (for regression)
    MSE,
    /// Mean Absolute Error (for regression)
    MAE,
}

impl Default for SplitCriterion {
    fn default() -> Self {
        SplitCriterion::Gini
    }
}

/// Configuration for decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeConfig {
    /// Maximum depth of the tree (None = no limit)
    pub max_depth: Option<usize>,
    /// Minimum samples required to split a node
    pub min_samples_split: usize,
    /// Minimum samples required at a leaf node
    pub min_samples_leaf: usize,
    /// Maximum number of features to consider for splits (None = all features)
    pub max_features: Option<usize>,
    /// Splitting criterion
    pub criterion: SplitCriterion,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        DecisionTreeConfig {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            criterion: SplitCriterion::Gini,
            random_seed: None,
        }
    }
}

/// Builder for DecisionTreeConfig
pub struct DecisionTreeConfigBuilder {
    config: DecisionTreeConfig,
}

impl DecisionTreeConfigBuilder {
    pub fn new() -> Self {
        DecisionTreeConfigBuilder {
            config: DecisionTreeConfig::default(),
        }
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

    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }

    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    pub fn build(self) -> DecisionTreeConfig {
        self.config
    }
}

impl Default for DecisionTreeConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A node in the decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Feature index used for splitting
    pub feature_index: Option<usize>,
    /// Threshold for the split
    pub threshold: Option<f64>,
    /// Prediction value (for leaf nodes)
    pub prediction: f64,
    /// Class probabilities (for classification)
    pub class_probs: Option<Vec<f64>>,
    /// Left child node index
    pub left_child: Option<usize>,
    /// Right child node index
    pub right_child: Option<usize>,
    /// Number of samples at this node
    pub n_samples: usize,
    /// Impurity at this node
    pub impurity: f64,
    /// Depth of this node
    pub depth: usize,
    /// Whether this is a leaf node
    pub is_leaf: bool,
}

impl TreeNode {
    fn new_leaf(
        prediction: f64,
        class_probs: Option<Vec<f64>>,
        n_samples: usize,
        impurity: f64,
        depth: usize,
    ) -> Self {
        TreeNode {
            feature_index: None,
            threshold: None,
            prediction,
            class_probs,
            left_child: None,
            right_child: None,
            n_samples,
            impurity,
            depth,
            is_leaf: true,
        }
    }

    fn new_split(
        feature_index: usize,
        threshold: f64,
        n_samples: usize,
        impurity: f64,
        depth: usize,
    ) -> Self {
        TreeNode {
            feature_index: Some(feature_index),
            threshold: Some(threshold),
            prediction: 0.0,
            class_probs: None,
            left_child: None,
            right_child: None,
            n_samples,
            impurity,
            depth,
            is_leaf: false,
        }
    }
}

/// Decision Tree Classifier
#[derive(Debug, Clone)]
pub struct DecisionTreeClassifier {
    config: DecisionTreeConfig,
    nodes: Vec<TreeNode>,
    feature_names: Vec<String>,
    n_classes: usize,
    classes: Vec<f64>,
    feature_importances_: Option<HashMap<String, f64>>,
    is_fitted: bool,
}

impl DecisionTreeClassifier {
    /// Create a new decision tree classifier
    pub fn new(config: DecisionTreeConfig) -> Self {
        DecisionTreeClassifier {
            config,
            nodes: Vec::new(),
            feature_names: Vec::new(),
            n_classes: 0,
            classes: Vec::new(),
            feature_importances_: None,
            is_fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(DecisionTreeConfig::default())
    }

    /// Get the tree nodes
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// Get the tree depth
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Get the number of leaves
    pub fn n_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf).count()
    }

    /// Calculate Gini impurity
    fn gini_impurity(class_counts: &[usize], total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f64;
        1.0 - class_counts
            .iter()
            .map(|&c| (c as f64 / total_f).powi(2))
            .sum::<f64>()
    }

    /// Calculate entropy.
    ///
    /// Uses the natural logarithm (nats), not `log2` (bits) as sklearn's
    /// default `criterion="entropy"` documentation states. This is a
    /// deliberate, harmless convention difference: for split *selection*
    /// the two are equivalent because `log2(p) = ln(p) / ln(2)`, i.e. the
    /// nats-based entropy is exactly `ln(2)` times the bits-based one, a
    /// uniform positive rescaling that never changes which candidate split
    /// maximizes information gain. Only the raw magnitude reported via
    /// `impurity`/feature importances differs from a bits-based
    /// implementation, not any decision made from it.
    fn entropy(class_counts: &[usize], total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let total_f = total as f64;
        -class_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total_f;
                p * p.ln()
            })
            .sum::<f64>()
    }

    /// Calculate impurity based on criterion
    fn calculate_impurity(&self, class_counts: &[usize], total: usize) -> f64 {
        match self.config.criterion {
            SplitCriterion::Gini => Self::gini_impurity(class_counts, total),
            SplitCriterion::Entropy => Self::entropy(class_counts, total),
            _ => Self::gini_impurity(class_counts, total),
        }
    }

    /// Find the best split for a node.
    ///
    /// When `max_features` is configured, the candidate feature subset is
    /// redrawn from `rng` at *every* call (i.e. at every node), matching the
    /// standard Random Forest node-level feature resampling scheme rather
    /// than a single subset fixed for the whole tree.
    fn find_best_split(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        indices: &[usize],
        n_features: usize,
        rng: &mut StdRng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        if indices.len() < self.config.min_samples_split {
            return None;
        }

        // Calculate current class counts
        let mut class_counts = vec![0usize; self.n_classes];
        for &idx in indices {
            let class_idx = self.classes.iter().position(|&c| c == y[idx]).unwrap_or(0);
            class_counts[class_idx] += 1;
        }
        let current_impurity = self.calculate_impurity(&class_counts, indices.len());

        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> = None;

        // Select features to consider: a fresh random subset drawn from `rng`
        // for this node (real per-node resampling, not one fixed permutation
        // reused at every node of the tree).
        let features_to_consider: Vec<usize> = if let Some(max_features) = self.config.max_features
        {
            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            feature_indices.shuffle(rng);
            feature_indices
                .into_iter()
                .take(max_features.min(n_features))
                .collect()
        } else {
            (0..n_features).collect()
        };

        for &feature_idx in &features_to_consider {
            // Get unique values for this feature
            let mut values: Vec<f64> = indices
                .iter()
                .map(|&idx| x[idx][feature_idx])
                .filter(|v| v.is_finite())
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Try different thresholds
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Split indices
                let mut left_indices = Vec::new();
                let mut right_indices = Vec::new();
                let mut left_counts = vec![0usize; self.n_classes];
                let mut right_counts = vec![0usize; self.n_classes];

                for &idx in indices {
                    let class_idx = self.classes.iter().position(|&c| c == y[idx]).unwrap_or(0);
                    if x[idx][feature_idx] <= threshold {
                        left_indices.push(idx);
                        left_counts[class_idx] += 1;
                    } else {
                        right_indices.push(idx);
                        right_counts[class_idx] += 1;
                    }
                }

                // Check minimum samples constraint
                if left_indices.len() < self.config.min_samples_leaf
                    || right_indices.len() < self.config.min_samples_leaf
                {
                    continue;
                }

                // Calculate information gain
                let left_impurity = self.calculate_impurity(&left_counts, left_indices.len());
                let right_impurity = self.calculate_impurity(&right_counts, right_indices.len());

                let n = indices.len() as f64;
                let weighted_impurity = (left_indices.len() as f64 * left_impurity
                    + right_indices.len() as f64 * right_impurity)
                    / n;

                let gain = current_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices, gain));
                }
            }
        }

        best_split
    }

    /// Build the tree recursively. `rng` is threaded through the whole
    /// recursion so that `max_features` feature subsampling is resampled at
    /// every node from a single real random stream, rather than fixed once
    /// per tree.
    fn build_tree(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        indices: Vec<usize>,
        depth: usize,
        rng: &mut StdRng,
    ) -> usize {
        // Calculate class counts for this node
        let mut class_counts = vec![0usize; self.n_classes];
        for &idx in &indices {
            let class_idx = self.classes.iter().position(|&c| c == y[idx]).unwrap_or(0);
            class_counts[class_idx] += 1;
        }

        let total = indices.len();
        let impurity = self.calculate_impurity(&class_counts, total);

        // Calculate prediction (majority class) and probabilities
        let (prediction, class_probs) = {
            let max_idx = class_counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let probs: Vec<f64> = class_counts
                .iter()
                .map(|&c| c as f64 / total as f64)
                .collect();
            (self.classes[max_idx], probs)
        };

        // Check stopping conditions
        let should_stop = self.config.max_depth.map(|d| depth >= d).unwrap_or(false)
            || total < self.config.min_samples_split
            || class_counts.iter().filter(|&&c| c > 0).count() <= 1;

        if should_stop {
            // Create leaf node
            let node = TreeNode::new_leaf(prediction, Some(class_probs), total, impurity, depth);
            let node_idx = self.nodes.len();
            self.nodes.push(node);
            return node_idx;
        }

        // Find best split
        let n_features = x[0].len();
        if let Some((feature_idx, threshold, left_indices, right_indices, _gain)) =
            self.find_best_split(x, y, &indices, n_features, rng)
        {
            // Create split node
            let mut node = TreeNode::new_split(feature_idx, threshold, total, impurity, depth);
            node.prediction = prediction;
            node.class_probs = Some(class_probs);

            let node_idx = self.nodes.len();
            self.nodes.push(node);

            // Build children
            let left_child_idx = self.build_tree(x, y, left_indices, depth + 1, rng);
            let right_child_idx = self.build_tree(x, y, right_indices, depth + 1, rng);

            self.nodes[node_idx].left_child = Some(left_child_idx);
            self.nodes[node_idx].right_child = Some(right_child_idx);

            node_idx
        } else {
            // No valid split found, create leaf
            let node = TreeNode::new_leaf(prediction, Some(class_probs), total, impurity, depth);
            let node_idx = self.nodes.len();
            self.nodes.push(node);
            node_idx
        }
    }

    /// Predict class probabilities for a single sample
    pub fn predict_proba_single(&self, sample: &[f64]) -> Option<Vec<f64>> {
        if self.nodes.is_empty() {
            return None;
        }

        let mut node_idx = 0;
        loop {
            let node = &self.nodes[node_idx];

            if node.is_leaf {
                return node.class_probs.clone();
            }

            let feature_idx = node.feature_index?;
            let threshold = node.threshold?;

            if sample[feature_idx] <= threshold {
                node_idx = node.left_child?;
            } else {
                node_idx = node.right_child?;
            }
        }
    }

    /// Predict class probabilities for multiple samples
    pub fn predict_proba(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        // Get feature matrix
        let x = self.get_feature_matrix(data)?;

        let probs: Vec<Vec<f64>> = x
            .iter()
            .map(|sample| {
                self.predict_proba_single(sample)
                    .unwrap_or_else(|| vec![1.0 / self.n_classes as f64; self.n_classes])
            })
            .collect();

        Ok(probs)
    }

    /// Get feature matrix from DataFrame
    fn get_feature_matrix(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        let n_rows = data.row_count();

        // Pre-fetch all column values
        let column_values: Vec<Vec<f64>> = self
            .feature_names
            .iter()
            .map(|col_name| {
                data.get_column_numeric_values(col_name).map_err(|_| {
                    Error::Column(format!("Column '{}' not found or not numeric", col_name))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Build feature matrix row by row
        let mut x = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<f64> = column_values.iter().map(|col| col[i]).collect();
            x.push(row);
        }

        Ok(x)
    }

    /// Calculate feature importances
    fn calculate_feature_importances(&mut self) {
        let mut importances = vec![0.0f64; self.feature_names.len()];
        let total_samples = self.nodes.get(0).map(|n| n.n_samples).unwrap_or(1) as f64;

        for node in &self.nodes {
            if !node.is_leaf {
                if let (Some(feature_idx), Some(left_idx), Some(right_idx)) =
                    (node.feature_index, node.left_child, node.right_child)
                {
                    let left_node = &self.nodes[left_idx];
                    let right_node = &self.nodes[right_idx];

                    let weighted_impurity_decrease = (node.n_samples as f64 / total_samples)
                        * (node.impurity
                            - (left_node.n_samples as f64 / node.n_samples as f64)
                                * left_node.impurity
                            - (right_node.n_samples as f64 / node.n_samples as f64)
                                * right_node.impurity);

                    if feature_idx < importances.len() {
                        importances[feature_idx] += weighted_impurity_decrease;
                    }
                }
            }
        }

        // Normalize
        let sum: f64 = importances.iter().sum();
        if sum > 0.0 {
            for imp in &mut importances {
                *imp /= sum;
            }
        }

        let importance_map: HashMap<String, f64> = self
            .feature_names
            .iter()
            .zip(importances.iter())
            .map(|(name, &imp)| (name.clone(), imp))
            .collect();

        self.feature_importances_ = Some(importance_map);
    }
}

impl SupervisedModel for DecisionTreeClassifier {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        // Get feature columns
        self.feature_names = train_data
            .column_names()
            .iter()
            .filter(|c| c.as_str() != target_column)
            .cloned()
            .collect();

        if self.feature_names.is_empty() {
            return Err(Error::InvalidInput("No feature columns found".to_string()));
        }

        // Get feature matrix
        let x = self.get_feature_matrix(train_data)?;

        // Get target values
        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| {
                Error::Column(format!(
                    "Target column '{}' not found or not numeric",
                    target_column
                ))
            })?;

        // Find unique classes
        let mut classes: Vec<f64> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
        self.n_classes = self.classes.len();

        // Build tree
        let indices: Vec<usize> = (0..x.len()).collect();
        self.nodes.clear();
        let mut rng = seeded_rng(self.config.random_seed);
        self.build_tree(&x, &y, indices, 0, &mut rng);

        // Calculate feature importances
        self.calculate_feature_importances();
        self.is_fitted = true;

        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        let x = self.get_feature_matrix(data)?;

        let predictions: Vec<f64> = x
            .iter()
            .map(|sample| {
                let probs = self
                    .predict_proba_single(sample)
                    .unwrap_or_else(|| vec![1.0 / self.n_classes as f64; self.n_classes]);
                let max_idx = probs
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

impl ModelEvaluator for DecisionTreeClassifier {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        let mut metrics = ModelMetrics::new();

        // Accuracy
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
        if folds < 2 {
            return Err(Error::InvalidInput(
                "Number of folds must be at least 2".into(),
            ));
        }

        let n = data.nrows();
        if n < folds {
            return Err(Error::InvalidInput(
                "Number of samples must be at least equal to the number of folds".into(),
            ));
        }

        let fold_size = n / folds;
        let mut all_metrics: Vec<ModelMetrics> = Vec::with_capacity(folds);

        for fold_idx in 0..folds {
            let test_start = fold_idx * fold_size;
            let test_end = if fold_idx == folds - 1 {
                n
            } else {
                (fold_idx + 1) * fold_size
            };

            let test_indices: Vec<usize> = (test_start..test_end).collect();
            let train_indices: Vec<usize> = (0..n)
                .filter(|&i| i < test_start || i >= test_end)
                .collect();

            if train_indices.is_empty() || test_indices.is_empty() {
                return Err(Error::InvalidInput(
                    "A fold resulted in empty train or test set".into(),
                ));
            }

            let train_df = data.sample(&train_indices)?;
            let test_df = data.sample(&test_indices)?;

            let mut model = self.clone();
            model.fit(&train_df, target)?;
            let fold_metrics = model.evaluate(&test_df, target)?;
            all_metrics.push(fold_metrics);
        }

        Ok(all_metrics)
    }
}

/// Decision Tree Regressor
#[derive(Debug, Clone)]
pub struct DecisionTreeRegressor {
    config: DecisionTreeConfig,
    nodes: Vec<TreeNode>,
    feature_names: Vec<String>,
    feature_importances_: Option<HashMap<String, f64>>,
    is_fitted: bool,
}

impl DecisionTreeRegressor {
    /// Create a new decision tree regressor
    pub fn new(config: DecisionTreeConfig) -> Self {
        let mut config = config;
        config.criterion = SplitCriterion::MSE;
        DecisionTreeRegressor {
            config,
            nodes: Vec::new(),
            feature_names: Vec::new(),
            feature_importances_: None,
            is_fitted: false,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(DecisionTreeConfig {
            criterion: SplitCriterion::MSE,
            ..Default::default()
        })
    }

    /// Calculate MSE for a set of values
    fn calculate_mse(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Calculate MAE (mean absolute deviation) for a set of values.
    ///
    /// CART's MAE criterion measures dispersion around the *median*, since
    /// the median (not the mean) is the constant that minimizes mean
    /// absolute deviation. This matches `calculate_mae`'s dispersion measure
    /// with the leaf/impurity-minimizing statistic actually used by
    /// [`build_tree`] when `criterion == SplitCriterion::MAE`.
    fn calculate_mae(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let median = median_of(values);
        values.iter().map(|v| (v - median).abs()).sum::<f64>() / values.len() as f64
    }

    /// Calculate impurity based on criterion
    fn calculate_impurity(&self, values: &[f64]) -> f64 {
        match self.config.criterion {
            SplitCriterion::MSE => Self::calculate_mse(values),
            SplitCriterion::MAE => Self::calculate_mae(values),
            _ => Self::calculate_mse(values),
        }
    }

    /// Calculate feature importances (mean impurity decrease, weighted by the
    /// fraction of samples routed through each split), normalized to sum to
    /// 1. Mirrors [`DecisionTreeClassifier::calculate_feature_importances`]
    /// exactly: the formula only depends on `TreeNode::{impurity,n_samples}`,
    /// which mean the same thing for a regression tree's MSE/MAE impurity as
    /// for a classification tree's Gini/entropy impurity. Previously this had
    /// no counterpart on the regressor at all, so `feature_importances()`
    /// unconditionally returned `None` and `RandomForestRegressor` had no
    /// real importances to average across trees.
    fn calculate_feature_importances(&mut self) {
        let mut importances = vec![0.0f64; self.feature_names.len()];
        let total_samples = self.nodes.first().map(|n| n.n_samples).unwrap_or(1) as f64;

        for node in &self.nodes {
            if !node.is_leaf {
                if let (Some(feature_idx), Some(left_idx), Some(right_idx)) =
                    (node.feature_index, node.left_child, node.right_child)
                {
                    let left_node = &self.nodes[left_idx];
                    let right_node = &self.nodes[right_idx];

                    let weighted_impurity_decrease = (node.n_samples as f64 / total_samples)
                        * (node.impurity
                            - (left_node.n_samples as f64 / node.n_samples as f64)
                                * left_node.impurity
                            - (right_node.n_samples as f64 / node.n_samples as f64)
                                * right_node.impurity);

                    if feature_idx < importances.len() {
                        importances[feature_idx] += weighted_impurity_decrease;
                    }
                }
            }
        }

        let sum: f64 = importances.iter().sum();
        if sum > 0.0 {
            for imp in &mut importances {
                *imp /= sum;
            }
        }

        let importance_map: HashMap<String, f64> = self
            .feature_names
            .iter()
            .zip(importances.iter())
            .map(|(name, &imp)| (name.clone(), imp))
            .collect();

        self.feature_importances_ = Some(importance_map);
    }

    /// Find the best split for a node.
    ///
    /// When `max_features` is configured, the candidate feature subset is
    /// redrawn from `rng` at *every* call (i.e. at every node), matching the
    /// standard Random Forest node-level feature resampling scheme rather
    /// than considering all features unconditionally.
    fn find_best_split(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        indices: &[usize],
        n_features: usize,
        rng: &mut StdRng,
    ) -> Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> {
        if indices.len() < self.config.min_samples_split {
            return None;
        }

        let values: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        let current_impurity = self.calculate_impurity(&values);

        let mut best_gain = 0.0;
        let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>, f64)> = None;

        // Select features to consider: a fresh random subset drawn from
        // `rng` for this node (real per-node resampling). Previously this
        // loop ignored `max_features` entirely, which made every tree in a
        // RandomForestRegressor consider all features and (combined with the
        // formerly-fake bootstrap) produce provably identical trees.
        let features_to_consider: Vec<usize> = if let Some(max_features) = self.config.max_features
        {
            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            feature_indices.shuffle(rng);
            feature_indices
                .into_iter()
                .take(max_features.min(n_features))
                .collect()
        } else {
            (0..n_features).collect()
        };

        for &feature_idx in &features_to_consider {
            let mut feature_values: Vec<f64> = indices
                .iter()
                .map(|&idx| x[idx][feature_idx])
                .filter(|v| v.is_finite())
                .collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            feature_values.dedup();

            for i in 0..feature_values.len().saturating_sub(1) {
                let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                let mut left_indices = Vec::new();
                let mut right_indices = Vec::new();
                let mut left_values = Vec::new();
                let mut right_values = Vec::new();

                for &idx in indices {
                    if x[idx][feature_idx] <= threshold {
                        left_indices.push(idx);
                        left_values.push(y[idx]);
                    } else {
                        right_indices.push(idx);
                        right_values.push(y[idx]);
                    }
                }

                if left_indices.len() < self.config.min_samples_leaf
                    || right_indices.len() < self.config.min_samples_leaf
                {
                    continue;
                }

                let left_impurity = self.calculate_impurity(&left_values);
                let right_impurity = self.calculate_impurity(&right_values);

                let n = indices.len() as f64;
                let weighted_impurity = (left_indices.len() as f64 * left_impurity
                    + right_indices.len() as f64 * right_impurity)
                    / n;

                let gain = current_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_split = Some((feature_idx, threshold, left_indices, right_indices, gain));
                }
            }
        }

        best_split
    }

    /// Build the tree recursively.
    ///
    /// `rng` is threaded through the recursion for per-node `max_features`
    /// resampling (see [`find_best_split`]). `root_impurity` is the impurity
    /// of the *entire* training set, computed once before the top-level
    /// call; the stopping rule below compares each node's impurity against
    /// it (a relative threshold) rather than a fixed absolute constant, so
    /// that trees fit on large-magnitude targets (where MSE routinely sits
    /// far above `1e-10`) and trees fit on near-constant targets both stop
    /// splitting at a consistent, scale-appropriate point.
    fn build_tree(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
        indices: Vec<usize>,
        depth: usize,
        rng: &mut StdRng,
        root_impurity: f64,
    ) -> usize {
        let values: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
        // The constant that minimizes each criterion's own loss: the mean
        // minimizes MSE, the median minimizes MAE. Using the mean
        // unconditionally previously made the MAE criterion's leaf values
        // inconsistent with the impurity it was supposedly minimizing.
        let prediction = match self.config.criterion {
            SplitCriterion::MAE => median_of(&values),
            _ => values.iter().sum::<f64>() / values.len() as f64,
        };
        let impurity = self.calculate_impurity(&values);
        let total = indices.len();

        let relative_eps = (root_impurity * 1e-10).max(0.0);
        let should_stop = self.config.max_depth.map(|d| depth >= d).unwrap_or(false)
            || total < self.config.min_samples_split
            || impurity <= relative_eps;

        if should_stop {
            let node = TreeNode::new_leaf(prediction, None, total, impurity, depth);
            let node_idx = self.nodes.len();
            self.nodes.push(node);
            return node_idx;
        }

        let n_features = x[0].len();
        if let Some((feature_idx, threshold, left_indices, right_indices, _gain)) =
            self.find_best_split(x, y, &indices, n_features, rng)
        {
            let mut node = TreeNode::new_split(feature_idx, threshold, total, impurity, depth);
            node.prediction = prediction;

            let node_idx = self.nodes.len();
            self.nodes.push(node);

            let left_child_idx = self.build_tree(x, y, left_indices, depth + 1, rng, root_impurity);
            let right_child_idx =
                self.build_tree(x, y, right_indices, depth + 1, rng, root_impurity);

            self.nodes[node_idx].left_child = Some(left_child_idx);
            self.nodes[node_idx].right_child = Some(right_child_idx);

            node_idx
        } else {
            let node = TreeNode::new_leaf(prediction, None, total, impurity, depth);
            let node_idx = self.nodes.len();
            self.nodes.push(node);
            node_idx
        }
    }

    /// Resolve the leaf-node index a sample would route to.
    ///
    /// Bounded to `nodes.len() + 1` iterations: a well-formed tree always
    /// reaches a leaf in at most `nodes.len()` hops, so exceeding that bound
    /// means the node links are corrupt (dangling `left_child`/`right_child`
    /// producing a cycle). Previously this traversal fell back to node `0`
    /// via `unwrap_or(0)` on any missing link, which could spin forever if
    /// node 0 was itself an internal node; it now surfaces a `Result`
    /// instead of ever looping unboundedly.
    fn leaf_index_single(&self, sample: &[f64]) -> Result<usize> {
        if self.nodes.is_empty() {
            return Err(Error::InvalidOperation(
                "Decision tree has no nodes (model not fitted)".to_string(),
            ));
        }

        let mut node_idx = 0usize;
        for _ in 0..=self.nodes.len() {
            let node = self.nodes.get(node_idx).ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "Corrupt decision tree: node index {} out of range",
                    node_idx
                ))
            })?;

            if node.is_leaf {
                return Ok(node_idx);
            }

            let feature_idx = node.feature_index.ok_or_else(|| {
                Error::InvalidOperation(
                    "Corrupt decision tree: internal node missing feature_index".to_string(),
                )
            })?;
            let threshold = node.threshold.ok_or_else(|| {
                Error::InvalidOperation(
                    "Corrupt decision tree: internal node missing threshold".to_string(),
                )
            })?;

            node_idx = if sample.get(feature_idx).copied().unwrap_or(f64::NAN) <= threshold {
                node.left_child.ok_or_else(|| {
                    Error::InvalidOperation(
                        "Corrupt decision tree: internal node missing left_child".to_string(),
                    )
                })?
            } else {
                node.right_child.ok_or_else(|| {
                    Error::InvalidOperation(
                        "Corrupt decision tree: internal node missing right_child".to_string(),
                    )
                })?
            };
        }

        Err(Error::InvalidOperation(
            "Decision tree traversal exceeded the node count; the tree structure is corrupt \
             (cyclic child links)"
                .to_string(),
        ))
    }

    /// Predict for a single sample
    fn predict_single(&self, sample: &[f64]) -> Result<f64> {
        let leaf_idx = self.leaf_index_single(sample)?;
        Ok(self.nodes[leaf_idx].prediction)
    }

    /// Return the leaf-node index each row of `data` routes to (analogous to
    /// scikit-learn's `tree.apply`). Exposed so ensembles (e.g. gradient
    /// boosting with the LAD/absolute-error loss) can recompute a
    /// terminal-region statistic — such as the median residual — that
    /// differs from the value the tree stored while fitting.
    pub fn leaf_indices(&self, data: &DataFrame) -> Result<Vec<usize>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }
        let x = self.get_feature_matrix(data)?;
        x.iter()
            .map(|sample| self.leaf_index_single(sample))
            .collect()
    }

    /// Overwrite the stored prediction value of a leaf node.
    ///
    /// Returns `Err` if `node_idx` is out of range or does not name a leaf,
    /// since silently mutating a split node's placeholder `prediction`
    /// field (which routing never reads) would look like it worked while
    /// actually doing nothing.
    pub fn set_leaf_prediction(&mut self, node_idx: usize, value: f64) -> Result<()> {
        let n_nodes = self.nodes.len();
        let node = self.nodes.get_mut(node_idx).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Node index {} is out of range (tree has {} nodes)",
                node_idx, n_nodes
            ))
        })?;
        if !node.is_leaf {
            return Err(Error::InvalidInput(format!(
                "Node {} is not a leaf; refusing to overwrite an internal split node's \
                 placeholder prediction value",
                node_idx
            )));
        }
        node.prediction = value;
        Ok(())
    }

    /// Get the tree nodes.
    pub fn nodes(&self) -> &[TreeNode] {
        &self.nodes
    }

    /// Get feature matrix from DataFrame
    fn get_feature_matrix(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        let n_rows = data.row_count();

        // Pre-fetch all column values
        let column_values: Vec<Vec<f64>> = self
            .feature_names
            .iter()
            .map(|col_name| {
                data.get_column_numeric_values(col_name).map_err(|_| {
                    Error::Column(format!("Column '{}' not found or not numeric", col_name))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Build feature matrix row by row
        let mut x = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<f64> = column_values.iter().map(|col| col[i]).collect();
            x.push(row);
        }

        Ok(x)
    }
}

impl SupervisedModel for DecisionTreeRegressor {
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

        let x = self.get_feature_matrix(train_data)?;
        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let indices: Vec<usize> = (0..x.len()).collect();
        self.nodes.clear();
        let mut rng = seeded_rng(self.config.random_seed);
        let root_impurity = self.calculate_impurity(&y);
        self.build_tree(&x, &y, indices, 0, &mut rng, root_impurity);
        self.calculate_feature_importances();
        self.is_fitted = true;

        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        let x = self.get_feature_matrix(data)?;
        let predictions: Vec<f64> = x
            .iter()
            .map(|sample| self.predict_single(sample))
            .collect::<Result<Vec<f64>>>()?;

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        self.feature_importances_.clone()
    }
}

impl ModelEvaluator for DecisionTreeRegressor {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        let mut metrics = ModelMetrics::new();

        // MSE
        let mse = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        metrics.add_metric("mse", mse);
        metrics.add_metric("rmse", mse.sqrt());

        // R²: see `models::r2_score_guarded` for the zero-`ss_tot` edge case.
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
    fn test_decision_tree_classifier() {
        let data = create_classification_data();
        let mut tree = DecisionTreeClassifier::new(DecisionTreeConfig::default());

        tree.fit(&data, "y").expect("operation should succeed");

        let predictions = tree.predict(&data).expect("operation should succeed");
        assert_eq!(predictions.len(), 10);

        // Check accuracy
        let metrics = tree.evaluate(&data, "y").expect("operation should succeed");
        let accuracy = metrics
            .get_metric("accuracy")
            .expect("operation should succeed");
        assert!(*accuracy > 0.8);
    }

    #[test]
    fn test_decision_tree_regressor() {
        let data = create_regression_data();
        let mut tree = DecisionTreeRegressor::default_config();

        tree.fit(&data, "y").expect("operation should succeed");

        let predictions = tree.predict(&data).expect("operation should succeed");
        assert_eq!(predictions.len(), 10);

        let metrics = tree.evaluate(&data, "y").expect("operation should succeed");
        let r2 = metrics.get_metric("r2").expect("operation should succeed");
        assert!(*r2 > 0.9);
    }

    #[test]
    fn test_tree_depth_limit() {
        let data = create_classification_data();
        let config = DecisionTreeConfigBuilder::new().max_depth(2).build();

        let mut tree = DecisionTreeClassifier::new(config);
        tree.fit(&data, "y").expect("operation should succeed");

        assert!(tree.depth() <= 2);
    }

    #[test]
    fn test_feature_importances() {
        let data = create_classification_data();
        let mut tree = DecisionTreeClassifier::default_config();
        tree.fit(&data, "y").expect("operation should succeed");

        let importances = tree
            .feature_importances()
            .expect("operation should succeed");
        assert!(!importances.is_empty());

        // Sum should be approximately 1
        let sum: f64 = importances.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_predict_proba() {
        let data = create_classification_data();
        let mut tree = DecisionTreeClassifier::default_config();
        tree.fit(&data, "y").expect("operation should succeed");

        let probs = tree.predict_proba(&data).expect("operation should succeed");
        assert_eq!(probs.len(), 10);

        // Each probability vector should sum to 1
        for prob in &probs {
            let sum: f64 = prob.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }
}
