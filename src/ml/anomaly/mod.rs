//! Anomaly detection algorithms
//!
//! This module provides real implementations of anomaly detection algorithms
//! for identifying outliers and unusual patterns in data.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::models::ModelEvaluator;
use crate::ml::models::ModelMetrics;
use crate::ml::models::UnsupervisedModel;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::RngExt;
use scirs2_core::random::SeedableRng;
use scirs2_core::random::SliceRandom;

// ─── Private helpers ────────────────────────────────────────────────────────

/// Extract numeric feature matrix from a DataFrame.
/// Returns (matrix, column_names) where matrix is outer=samples, inner=features.
fn extract_features(
    data: &DataFrame,
    feature_columns: &Option<Vec<String>>,
) -> Result<(Vec<Vec<f64>>, Vec<String>)> {
    let col_names: Vec<String> = match feature_columns {
        Some(cols) => cols.clone(),
        None => data.column_names(),
    };

    let n_samples = data.nrows();
    let n_features = col_names.len();

    if n_features == 0 {
        return Err(Error::InvalidOperation(
            "No feature columns available".into(),
        ));
    }

    let mut matrix = vec![vec![0.0f64; n_features]; n_samples];

    for (feat_idx, col_name) in col_names.iter().enumerate() {
        let col = data
            .get_column::<f64>(col_name)
            .map_err(|_| Error::InvalidInput(format!("Column '{}' is not numeric", col_name)))?;
        let vals = col.values();
        for row_idx in 0..n_samples {
            if row_idx < vals.len() {
                matrix[row_idx][feat_idx] = vals[row_idx];
            } else {
                return Err(Error::IndexOutOfBounds {
                    index: row_idx,
                    size: vals.len(),
                });
            }
        }
    }

    Ok((matrix, col_names))
}

/// Squared Euclidean distance between two feature vectors.
#[inline]
fn euclidean_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Euler–Mascheroni constant used in the c(n) computation.
const EULER_MASCHERONI: f64 = 0.577_215_664_9;

/// Expected average path length for n samples (isolation forest normalisation).
/// c(n) = 2 * H(n-1) - 2*(n-1)/n  where H(k) = ln(k) + γ
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n = n as f64;
    let h = (n - 1.0).ln() + EULER_MASCHERONI;
    2.0 * h - 2.0 * (n - 1.0) / n
}

// ─── IsolationTree (private) ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum IsolationNode {
    Leaf {
        size: usize,
    },
    Split {
        feature_idx: usize,
        split_value: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
}

#[derive(Debug, Clone)]
struct IsolationTree {
    root: Option<Box<IsolationNode>>,
    max_depth: usize,
}

impl IsolationTree {
    fn build(
        data: &[Vec<f64>],
        rng: &mut StdRng,
        max_depth: usize,
        depth: usize,
    ) -> Box<IsolationNode> {
        let n = data.len();
        let n_features = if n == 0 { 0 } else { data[0].len() };

        if depth >= max_depth || n <= 1 || n_features == 0 {
            return Box::new(IsolationNode::Leaf { size: n });
        }

        let feat = rng.random_range(0..n_features);

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for row in data {
            let v = row[feat];
            if v < min_val {
                min_val = v;
            }
            if v > max_val {
                max_val = v;
            }
        }

        if (max_val - min_val).abs() < f64::EPSILON {
            return Box::new(IsolationNode::Leaf { size: n });
        }

        let split: f64 = min_val + rng.random::<f64>() * (max_val - min_val);

        let mut left_data: Vec<Vec<f64>> = Vec::new();
        let mut right_data: Vec<Vec<f64>> = Vec::new();
        for row in data {
            if row[feat] < split {
                left_data.push(row.clone());
            } else {
                right_data.push(row.clone());
            }
        }

        let left = Self::build(&left_data, rng, max_depth, depth + 1);
        let right = Self::build(&right_data, rng, max_depth, depth + 1);

        Box::new(IsolationNode::Split {
            feature_idx: feat,
            split_value: split,
            left,
            right,
        })
    }

    fn path_length(node: &IsolationNode, sample: &[f64], depth: usize) -> f64 {
        match node {
            IsolationNode::Leaf { size } => depth as f64 + c_factor(*size),
            IsolationNode::Split {
                feature_idx,
                split_value,
                left,
                right,
            } => {
                if sample[*feature_idx] < *split_value {
                    Self::path_length(left, sample, depth + 1)
                } else {
                    Self::path_length(right, sample, depth + 1)
                }
            }
        }
    }

    fn score_sample(&self, sample: &[f64]) -> f64 {
        match &self.root {
            Some(node) => Self::path_length(node, sample, 0),
            None => 0.0,
        }
    }
}

// ─── IsolationForest ─────────────────────────────────────────────────────────

/// Isolation Forest for anomaly detection.
///
/// Detects anomalies by isolating samples through random recursive partitioning.
/// Points that require fewer splits to isolate are flagged as anomalies.
#[derive(Debug, Clone)]
pub struct IsolationForest {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Maximum number of samples to draw for each tree
    pub max_samples: Option<usize>,
    /// Maximum depth of the trees
    pub max_depth: Option<usize>,
    /// Contamination: expected proportion of outliers in the data
    pub contamination: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Raw anomaly scores after fit (lower = more anomalous, range 0..1)
    pub scores: Option<Vec<f64>>,
    /// Anomaly labels after fit (-1 anomaly, 1 normal)
    pub labels: Option<Vec<i64>>,
    /// Feature columns used for anomaly detection
    pub feature_columns: Option<Vec<String>>,
    // Private: stored trees for scoring new data
    trees: Vec<IsolationTree>,
    // Private: max_samples actually used during fit (for score normalisation)
    fitted_max_samples: usize,
}

impl IsolationForest {
    /// Create a new IsolationForest with sensible defaults.
    pub fn new() -> Self {
        IsolationForest {
            n_estimators: 100,
            max_samples: None,
            max_depth: None,
            contamination: 0.1,
            random_seed: None,
            scores: None,
            labels: None,
            feature_columns: None,
            trees: Vec::new(),
            fitted_max_samples: 256,
        }
    }

    /// Get raw anomaly scores (lower means more anomalous).
    pub fn anomaly_scores(&self) -> &[f64] {
        match &self.scores {
            Some(s) => s,
            None => &[],
        }
    }

    /// Get anomaly labels: -1 for anomaly, 1 for normal.
    pub fn labels(&self) -> &[i64] {
        match &self.labels {
            Some(l) => l,
            None => &[],
        }
    }

    /// Set number of trees in the forest.
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set maximum number of samples drawn per tree.
    pub fn max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = Some(max_samples);
        self
    }

    /// Set maximum depth of each tree.
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Set expected contamination proportion.
    pub fn contamination(mut self, contamination: f64) -> Self {
        self.contamination = contamination;
        self
    }

    /// Set random seed for reproducibility.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Specify feature columns to use.
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = Some(columns);
        self
    }

    /// Compute anomaly score for a single sample using all fitted trees.
    fn score_sample_internal(&self, sample: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.5;
        }
        let mean_path: f64 = self
            .trees
            .iter()
            .map(|t| t.score_sample(sample))
            .sum::<f64>()
            / self.trees.len() as f64;
        2.0_f64.powf(-mean_path / c_factor(self.fitted_max_samples).max(1.0))
    }

    /// Compute anomaly scores for all samples in a feature matrix.
    fn scores_for_matrix(&self, matrix: &[Vec<f64>]) -> Vec<f64> {
        matrix
            .iter()
            .map(|row| self.score_sample_internal(row))
            .collect()
    }

    /// Predict anomaly labels (-1.0 for anomaly, 1.0 for normal) for new data.
    /// Higher raw IF score = more anomalous; threshold is at the (1-contamination) quantile.
    pub fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        let (matrix, _) = extract_features(data, &self.feature_columns)?;
        let raw = self.scores_for_matrix(&matrix);
        let threshold = self.anomaly_threshold(&raw);
        let labels: Vec<f64> = raw
            .iter()
            .map(|&s| if s >= threshold { -1.0 } else { 1.0 })
            .collect();
        Ok(labels)
    }

    /// Return raw anomaly scores for new data.
    /// Higher score = more anomalous (easier to isolate = shorter path length).
    pub fn decision_function(&self, data: &DataFrame) -> Result<Vec<f64>> {
        let (matrix, _) = extract_features(data, &self.feature_columns)?;
        Ok(self.scores_for_matrix(&matrix))
    }

    /// Compute the contamination-th highest score as anomaly threshold.
    /// Points scoring at or above this threshold are flagged as anomalies.
    fn anomaly_threshold(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.5;
        }
        let mut sorted = scores.to_vec();
        // Sort descending: highest scores first (most anomalous first)
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((self.contamination * scores.len() as f64).ceil() as usize)
            .min(sorted.len() - 1)
            .max(0);
        sorted[idx]
    }
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsupervisedModel for IsolationForest {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        let (feature_data, col_names) = extract_features(data, &self.feature_columns)?;
        let n_samples = feature_data.len();

        if n_samples == 0 {
            return Err(Error::InvalidOperation("Empty dataset".into()));
        }

        let sub_size = self.max_samples.unwrap_or(256).min(n_samples).max(1);
        self.fitted_max_samples = sub_size;

        let tree_depth = self
            .max_depth
            .unwrap_or_else(|| (sub_size as f64).log2() as usize + 1);

        let mut rng: StdRng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut seed_bytes = [0u8; 32];
                scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
                StdRng::from_seed(seed_bytes)
            }
        };

        self.trees.clear();
        self.trees.reserve(self.n_estimators);

        let mut indices: Vec<usize> = (0..n_samples).collect();

        for _ in 0..self.n_estimators {
            indices.shuffle(&mut rng);
            let subsample: Vec<Vec<f64>> = indices[..sub_size]
                .iter()
                .map(|&i| feature_data[i].clone())
                .collect();

            let root = IsolationTree::build(&subsample, &mut rng, tree_depth, 0);
            self.trees.push(IsolationTree {
                root: Some(root),
                max_depth: tree_depth,
            });
        }

        let scores = self.scores_for_matrix(&feature_data);
        // Higher IF score = more anomalous; threshold at (1-contamination) quantile
        let threshold = self.anomaly_threshold(&scores);

        let labels: Vec<i64> = scores
            .iter()
            .map(|&s| if s >= threshold { -1 } else { 1 })
            .collect();

        self.scores = Some(scores);
        self.labels = Some(labels);
        self.feature_columns = Some(col_names);

        Ok(())
    }

    fn transform(&self, data: &DataFrame) -> Result<DataFrame> {
        let scores = self.decision_function(data)?;
        let mut result = data.clone();
        result.add_column(
            "anomaly_score".to_string(),
            crate::series::Series::new(scores, Some("anomaly_score".to_string()))?,
        )?;
        Ok(result)
    }
}

impl ModelEvaluator for IsolationForest {
    fn evaluate(&self, _test_data: &DataFrame, _test_target: &str) -> Result<ModelMetrics> {
        let mut metrics = ModelMetrics::new();
        metrics.add_metric("anomaly_ratio", self.contamination);
        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for anomaly detection".into(),
        ))
    }
}

// ─── LocalOutlierFactor ───────────────────────────────────────────────────────

/// Local Outlier Factor for anomaly detection.
///
/// Computes the local density deviation of each point relative to its
/// k nearest neighbours. LOF near 1 indicates an inlier; LOF much greater
/// than 1 indicates an outlier.
#[derive(Debug, Clone)]
pub struct LocalOutlierFactor {
    /// Number of neighbours to consider
    pub n_neighbors: usize,
    /// Contamination: expected proportion of outliers in the data
    pub contamination: f64,
    /// Algorithm hint (kept for API compatibility)
    pub algorithm: String,
    /// LOF scores for training points (higher = more anomalous)
    pub scores: Option<Vec<f64>>,
    /// Anomaly labels (-1 anomaly, 1 normal)
    pub labels: Option<Vec<i64>>,
    /// Feature columns used for anomaly detection
    pub feature_columns: Option<Vec<String>>,
    // Private: stored training data
    train_data: Option<Vec<Vec<f64>>>,
}

impl LocalOutlierFactor {
    /// Create a new LocalOutlierFactor instance.
    pub fn new(n_neighbors: usize) -> Self {
        LocalOutlierFactor {
            n_neighbors,
            contamination: 0.1,
            algorithm: "auto".to_string(),
            scores: None,
            labels: None,
            feature_columns: None,
            train_data: None,
        }
    }

    /// Get LOF anomaly scores (higher = more anomalous).
    pub fn anomaly_scores(&self) -> &[f64] {
        match &self.scores {
            Some(s) => s,
            None => &[],
        }
    }

    /// Get anomaly labels: -1 for anomaly, 1 for normal.
    pub fn labels(&self) -> &[i64] {
        match &self.labels {
            Some(l) => l,
            None => &[],
        }
    }

    /// Set contamination (expected proportion of outliers).
    pub fn contamination(mut self, contamination: f64) -> Self {
        self.contamination = contamination;
        self
    }

    /// Set algorithm hint.
    pub fn algorithm(mut self, algorithm: &str) -> Self {
        self.algorithm = algorithm.to_string();
        self
    }

    /// Specify feature columns to use.
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = Some(columns);
        self
    }

    /// Find k nearest neighbour indices and distances for point at `idx`.
    fn knn(data: &[Vec<f64>], idx: usize, k: usize) -> (Vec<usize>, Vec<f64>) {
        let n = data.len();
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != idx)
            .map(|j| (j, euclidean_sq(&data[idx], &data[j]).sqrt()))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        let indices: Vec<usize> = dists.iter().map(|(i, _)| *i).collect();
        let distances: Vec<f64> = dists.iter().map(|(_, d)| *d).collect();
        (indices, distances)
    }
}

impl UnsupervisedModel for LocalOutlierFactor {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        let (feature_data, col_names) = extract_features(data, &self.feature_columns)?;
        let n_samples = feature_data.len();

        if n_samples == 0 {
            return Err(Error::InvalidOperation("Empty dataset".into()));
        }

        let k = self.n_neighbors.min(n_samples - 1).max(1);

        // Step 1: k-nearest neighbours for every point
        let mut all_nn_indices: Vec<Vec<usize>> = Vec::with_capacity(n_samples);
        let mut all_nn_dists: Vec<Vec<f64>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let (nn_idx, nn_dist) = Self::knn(&feature_data, i, k);
            all_nn_indices.push(nn_idx);
            all_nn_dists.push(nn_dist);
        }

        // Step 2: k-distance(i) = distance to its k-th nearest neighbour
        let k_dist: Vec<f64> = all_nn_dists
            .iter()
            .map(|dists| dists.last().copied().unwrap_or(0.0))
            .collect();

        // Step 3: Local reachability density
        //   reach_dist(A,B) = max(k_dist(B), dist(A,B))
        //   lrd(A) = k / sum( reach_dist(A, B_i) for B_i in kNN(A) )
        let mut lrd: Vec<f64> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let reach_sum: f64 = all_nn_indices[i]
                .iter()
                .zip(all_nn_dists[i].iter())
                .map(|(&j, &dist_ij)| k_dist[j].max(dist_ij))
                .sum();
            let k_actual = all_nn_indices[i].len() as f64;
            let density = if reach_sum < f64::EPSILON {
                f64::INFINITY
            } else {
                k_actual / reach_sum
            };
            lrd.push(density);
        }

        // Step 4: LOF(A) = mean(lrd(neighbour) / lrd(A)) over kNN(A)
        let mut lof_scores: Vec<f64> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let k_actual = all_nn_indices[i].len();
            let lof_i = if k_actual == 0 || lrd[i].is_infinite() {
                1.0
            } else {
                let sum_ratio: f64 = all_nn_indices[i].iter().map(|&j| lrd[j] / lrd[i]).sum();
                sum_ratio / k_actual as f64
            };
            lof_scores.push(lof_i);
        }

        // Step 5: threshold at contamination-th percentile (LOF: higher = anomalous)
        let mut sorted = lof_scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((self.contamination * n_samples as f64).ceil() as usize)
            .min(sorted.len() - 1)
            .max(0);
        let threshold = sorted[threshold_idx];

        let labels: Vec<i64> = lof_scores
            .iter()
            .map(|&s| if s > threshold { -1 } else { 1 })
            .collect();

        self.scores = Some(lof_scores);
        self.labels = Some(labels);
        self.feature_columns = Some(col_names);
        self.train_data = Some(feature_data);

        Ok(())
    }

    fn transform(&self, _data: &DataFrame) -> Result<DataFrame> {
        Err(Error::InvalidOperation(
            "LocalOutlierFactor does not support transform on unseen data (transductive method)"
                .into(),
        ))
    }
}

impl ModelEvaluator for LocalOutlierFactor {
    fn evaluate(&self, _test_data: &DataFrame, _test_target: &str) -> Result<ModelMetrics> {
        let mut metrics = ModelMetrics::new();
        metrics.add_metric("anomaly_ratio", self.contamination);
        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for anomaly detection".into(),
        ))
    }
}

// ─── OneClassSVM (SVDD / kernel-distance approach) ───────────────────────────

/// One-Class SVM for anomaly detection using an RBF kernel distance approach.
///
/// Implements a simplified Support Vector Data Description (SVDD): the anomaly
/// score for each point is 1 minus its mean RBF kernel similarity to the
/// training set. Higher score means farther from the training data distribution.
#[derive(Debug, Clone)]
pub struct OneClassSVM {
    /// Kernel type (only "rbf" is currently supported)
    pub kernel: String,
    /// Regularisation parameter (fraction of outliers upper bound)
    pub nu: f64,
    /// RBF kernel bandwidth: K(x,y) = exp(-gamma * ||x-y||^2)
    pub gamma: Option<f64>,
    /// Anomaly scores for training points (higher = more anomalous)
    pub scores: Option<Vec<f64>>,
    /// Anomaly labels (-1 anomaly, 1 normal)
    pub labels: Option<Vec<i64>>,
    /// Feature columns used for anomaly detection
    pub feature_columns: Option<Vec<String>>,
    // Private: stored training data for scoring new points
    train_data: Option<Vec<Vec<f64>>>,
    // Private: gamma actually used (resolved from data if not set)
    fitted_gamma: f64,
}

impl OneClassSVM {
    /// Create a new OneClassSVM with sensible defaults.
    pub fn new() -> Self {
        OneClassSVM {
            kernel: "rbf".to_string(),
            nu: 0.1,
            gamma: None,
            scores: None,
            labels: None,
            feature_columns: None,
            train_data: None,
            fitted_gamma: 1.0,
        }
    }

    /// Get anomaly scores (higher = more anomalous).
    pub fn anomaly_scores(&self) -> &[f64] {
        match &self.scores {
            Some(s) => s,
            None => &[],
        }
    }

    /// Get anomaly labels: -1 for anomaly, 1 for normal.
    pub fn labels(&self) -> &[i64] {
        match &self.labels {
            Some(l) => l,
            None => &[],
        }
    }

    /// Set kernel type.
    pub fn kernel(mut self, kernel: &str) -> Self {
        self.kernel = kernel.to_string();
        self
    }

    /// Set regularisation parameter nu.
    pub fn nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// Set RBF kernel coefficient gamma.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Specify feature columns to use.
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = Some(columns);
        self
    }

    /// Compute RBF kernel value between two vectors.
    #[inline]
    fn rbf_kernel(a: &[f64], b: &[f64], gamma: f64) -> f64 {
        (-gamma * euclidean_sq(a, b)).exp()
    }

    /// Compute anomaly score for a test point against the training set.
    /// score = 1 - mean_j K(x, x_j)  (higher = farther from training mass)
    fn score_point(&self, x: &[f64]) -> f64 {
        match &self.train_data {
            None => 0.5,
            Some(train) => {
                if train.is_empty() {
                    return 0.5;
                }
                let mean_k: f64 = train
                    .iter()
                    .map(|t| Self::rbf_kernel(x, t, self.fitted_gamma))
                    .sum::<f64>()
                    / train.len() as f64;
                1.0 - mean_k
            }
        }
    }
}

impl Default for OneClassSVM {
    fn default() -> Self {
        Self::new()
    }
}

impl UnsupervisedModel for OneClassSVM {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        let (feature_data, col_names) = extract_features(data, &self.feature_columns)?;
        let n_samples = feature_data.len();

        if n_samples == 0 {
            return Err(Error::InvalidOperation("Empty dataset".into()));
        }

        let n_features = feature_data[0].len();
        self.fitted_gamma = self.gamma.unwrap_or_else(|| {
            if n_features == 0 {
                1.0
            } else {
                1.0 / n_features as f64
            }
        });

        self.train_data = Some(feature_data.clone());

        let scores: Vec<f64> = feature_data.iter().map(|x| self.score_point(x)).collect();

        // Threshold at nu-th percentile from the top (higher = anomalous)
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((self.nu * n_samples as f64).ceil() as usize)
            .min(sorted.len() - 1)
            .max(0);
        let threshold = sorted[threshold_idx];

        let labels: Vec<i64> = scores
            .iter()
            .map(|&s| if s > threshold { -1 } else { 1 })
            .collect();

        self.scores = Some(scores);
        self.labels = Some(labels);
        self.feature_columns = Some(col_names);

        Ok(())
    }

    fn transform(&self, data: &DataFrame) -> Result<DataFrame> {
        if self.train_data.is_none() {
            return Err(Error::InvalidOperation(
                "OneClassSVM has not been fitted yet. Call fit() first.".into(),
            ));
        }

        let (feature_data, _) = extract_features(data, &self.feature_columns)?;
        let scores: Vec<f64> = feature_data.iter().map(|x| self.score_point(x)).collect();

        let mut result = data.clone();
        result.add_column(
            "anomaly_score".to_string(),
            crate::series::Series::new(scores, Some("anomaly_score".to_string()))?,
        )?;
        Ok(result)
    }
}

impl ModelEvaluator for OneClassSVM {
    fn evaluate(&self, _test_data: &DataFrame, _test_target: &str) -> Result<ModelMetrics> {
        let mut metrics = ModelMetrics::new();
        metrics.add_metric("anomaly_ratio", self.nu);
        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for anomaly detection".into(),
        ))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;

    /// Build a test DataFrame with 20 inliers near the origin and 3 extreme outliers.
    fn make_test_df() -> DataFrame {
        let inlier_x: Vec<f64> = vec![
            0.1, -0.3, 0.5, -0.7, 0.2, 0.8, -0.5, 0.6, -0.2, 0.4, -0.9, 0.3, 0.7, -0.4, 0.1, -0.6,
            0.9, -0.1, 0.4, -0.8,
        ];
        let inlier_y: Vec<f64> = vec![
            0.2, 0.5, -0.3, 0.8, -0.6, 0.1, -0.4, 0.7, -0.5, 0.3, 0.6, -0.2, -0.8, 0.4, -0.9, 0.2,
            -0.7, 0.5, -0.1, 0.3,
        ];

        let outlier_x: Vec<f64> = vec![10.0, -10.0, 10.0];
        let outlier_y: Vec<f64> = vec![10.0, 10.0, -10.0];

        let mut xs = inlier_x;
        xs.extend(outlier_x);
        let mut ys = inlier_y;
        ys.extend(outlier_y);

        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(xs, Some("x".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "y".to_string(),
            Series::new(ys, Some("y".to_string())).unwrap(),
        )
        .unwrap();
        df
    }

    #[test]
    fn test_unfitted_anomaly_score_empty() {
        let ifo = IsolationForest::new();
        assert_eq!(ifo.anomaly_scores().len(), 0);
        assert_eq!(ifo.labels().len(), 0);
    }

    #[test]
    fn test_isolation_forest_labels_not_all_same() {
        let df = make_test_df();
        let mut ifo = IsolationForest::new()
            .n_estimators(100)
            .contamination(0.15)
            .random_seed(42);

        ifo.fit(&df).unwrap();

        let labels = ifo.labels();
        assert!(!labels.is_empty(), "Labels must not be empty after fit");

        let has_anomaly = labels.iter().any(|&l| l == -1);
        let has_normal = labels.iter().any(|&l| l == 1);
        assert!(has_anomaly, "Should have at least one anomaly label (-1)");
        assert!(has_normal, "Should have at least one normal label (1)");
    }

    #[test]
    fn test_isolation_forest_detects_outliers() {
        let df = make_test_df();
        let mut ifo = IsolationForest::new()
            .n_estimators(100)
            .contamination(0.15)
            .random_seed(42);

        ifo.fit(&df).unwrap();

        // Outlier points should score lower (more anomalous) than inliers in
        // decision_function (lower isolation-forest score = more anomalous)
        let mut outlier_df = DataFrame::new();
        let ox = vec![10.0_f64, -10.0, 10.0];
        let oy = vec![10.0_f64, 10.0, -10.0];
        outlier_df
            .add_column(
                "x".to_string(),
                Series::new(ox, Some("x".to_string())).unwrap(),
            )
            .unwrap();
        outlier_df
            .add_column(
                "y".to_string(),
                Series::new(oy, Some("y".to_string())).unwrap(),
            )
            .unwrap();

        let mut inlier_df = DataFrame::new();
        let ix: Vec<f64> = vec![0.1, -0.3, 0.5, -0.7, 0.2];
        let iy: Vec<f64> = vec![0.2, 0.5, -0.3, 0.8, -0.6];
        inlier_df
            .add_column(
                "x".to_string(),
                Series::new(ix, Some("x".to_string())).unwrap(),
            )
            .unwrap();
        inlier_df
            .add_column(
                "y".to_string(),
                Series::new(iy, Some("y".to_string())).unwrap(),
            )
            .unwrap();

        let outlier_scores = ifo.decision_function(&outlier_df).unwrap();
        let inlier_scores = ifo.decision_function(&inlier_df).unwrap();

        let mean_outlier: f64 = outlier_scores.iter().sum::<f64>() / outlier_scores.len() as f64;
        let mean_inlier: f64 = inlier_scores.iter().sum::<f64>() / inlier_scores.len() as f64;

        // In Isolation Forest, higher raw score = easier to isolate = more anomalous.
        // True outliers far from origin should be isolated faster → higher score.
        assert!(
            mean_outlier > mean_inlier,
            "Outlier mean score ({:.4}) should be greater than inlier mean score ({:.4})",
            mean_outlier,
            mean_inlier
        );
    }

    #[test]
    fn test_lof_detects_outliers() {
        let df = make_test_df();
        let mut lof = LocalOutlierFactor::new(3).contamination(0.15);

        lof.fit(&df).unwrap();

        let labels = lof.labels();
        assert!(!labels.is_empty(), "Labels must not be empty after fit");

        let has_anomaly = labels.iter().any(|&l| l == -1);
        let has_normal = labels.iter().any(|&l| l == 1);
        assert!(has_anomaly, "LOF should detect at least one anomaly");
        assert!(
            has_normal,
            "LOF should classify at least one point as normal"
        );
    }

    #[test]
    fn test_one_class_svm_detects_outliers() {
        let df = make_test_df();
        let mut svm = OneClassSVM::new().nu(0.15);

        svm.fit(&df).unwrap();

        let scores = svm.anomaly_scores();
        assert!(!scores.is_empty(), "Scores must not be empty after fit");

        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_score - min_score) > 1e-9,
            "Anomaly scores must vary across samples (not all identical)"
        );

        let result = svm.transform(&df).unwrap();
        assert!(
            result.column_names().contains(&"anomaly_score".to_string()),
            "transform() must produce anomaly_score column"
        );
    }
}
