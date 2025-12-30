//! Anomaly detection module
//!
//! Provides algorithms for detecting outliers and abnormal patterns in datasets.

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::column::{Float64Column, Int64Column, Column, ColumnTrait};
use crate::error::{Result, Error};
use crate::ml::pipeline::Transformer;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::utils::rand_compat::GenRangeCompat;
use std::collections::{HashMap, HashSet};

/// Isolation Forest anomaly detection algorithm
pub struct IsolationForest {
    /// Number of decision trees
    n_estimators: usize,
    /// Subsampling size
    max_samples: Option<usize>,
    /// Feature subsampling (ratio)
    max_features: Option<f64>,
    /// Random seed
    random_seed: Option<u64>,
    /// Contamination rate (expected proportion of anomalies)
    contamination: f64,
    /// Anomaly scores for each sample
    anomaly_scores: Vec<f64>,
    /// Anomaly threshold
    threshold: f64,
    /// Anomaly flags (1: anomaly, -1: normal)
    labels: Vec<i64>,
    /// Feature names
    feature_names: Vec<String>,
    /// Whether the model is fitted
    fitted: bool,
    /// Collection of trees
    trees: Vec<ITree>,
}

/// Decision tree for Isolation Forest
struct ITree {
    /// Maximum tree depth
    height_limit: usize,
    /// Root node of the tree
    root: Option<Box<ITreeNode>>,
}

/// Node of a decision tree in Isolation Forest
struct ITreeNode {
    /// Index of the split feature
    split_feature: Option<usize>,
    /// Split threshold
    split_threshold: Option<f64>,
    /// Left child node
    left: Option<Box<ITreeNode>>,
    /// Right child node
    right: Option<Box<ITreeNode>>,
    /// Depth of the tree if no further splits are made
    depth: usize,
    /// Number of samples in this node
    size: usize,
}

impl IsolationForest {
    /// Create a new IsolationForest instance
    pub fn new(
        n_estimators: usize,
        max_samples: Option<usize>,
        max_features: Option<f64>,
        contamination: f64,
        random_seed: Option<u64>,
    ) -> Self {
        if contamination <= 0.0 || contamination >= 0.5 {
            panic!("Contamination must be in (0, 0.5)");
        }
        
        IsolationForest {
            n_estimators,
            max_samples,
            max_features,
            random_seed,
            contamination,
            anomaly_scores: Vec::new(),
            threshold: 0.0,
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
            trees: Vec::new(),
        }
    }
    
    /// Get anomaly scores
    pub fn anomaly_scores(&self) -> &[f64] {
        &self.anomaly_scores
    }
    
    /// Get anomaly flags (1: anomaly, -1: normal)
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }
    
    /// Build decision tree
    fn build_tree(
        &self,
        data: &[Vec<f64>],
        indices: &[usize],
        height_limit: usize,
        depth: usize,
        rng: &mut StdRng,
    ) -> Option<Box<ITreeNode>> {
        // Termination condition
        if indices.is_empty() {
            return None;
        }
        
        if depth >= height_limit || indices.len() <= 1 {
            return Some(Box::new(ITreeNode {
                split_feature: None,
                split_threshold: None,
                left: None,
                right: None,
                depth,
                size: indices.len(),
            }));
        }
        
        // Feature sampling
        let n_features = data[0].len();
        let n_features_to_use = match self.max_features {
            Some(ratio) => (ratio * n_features as f64).round() as usize,
            None => n_features,
        };
        
        let feature_indices: Vec<usize> = (0..n_features).collect();
        let sampled_features: Vec<usize> = feature_indices
            .iter()
            .copied()
            .filter(|_| rng.gen_bool(n_features_to_use as f64 / n_features as f64))
            .collect();
        
        if sampled_features.is_empty() {
            // Select at least one feature
            return Some(Box::new(ITreeNode {
                split_feature: Some(rng.gen_range(0..n_features)),
                split_threshold: Some(rng.random()),
                left: None,
                right: None,
                depth,
                size: indices.len(),
            }));
        }
        
        // Randomly select feature and threshold
        let split_feature = sampled_features[rng.gen_range(0..sampled_features.len())];
        
        // Find min and max values of the selected feature
        let min_val = indices.iter().map(|&i| data[i][split_feature]).fold(f64::INFINITY, f64::min);
        let max_val = indices.iter().map(|&i| data[i][split_feature]).fold(f64::NEG_INFINITY, f64::max);
        
        // If min and max values are the same, cannot split
        if (max_val - min_val).abs() < f64::EPSILON {
            return Some(Box::new(ITreeNode {
                split_feature: None,
                split_threshold: None,
                left: None,
                right: None,
                depth,
                size: indices.len(),
            }));
        }
        
        // Randomly select threshold
        let split_threshold = min_val + rng.random::<f64>() * (max_val - min_val);
        
        // Split data
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for &idx in indices {
            if data[idx][split_feature] < split_threshold {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }
        
        // Recursively build left and right child nodes
        let left = self.build_tree(data, &left_indices, height_limit, depth + 1, rng);
        let right = self.build_tree(data, &right_indices, height_limit, depth + 1, rng);
        
        Some(Box::new(ITreeNode {
            split_feature: Some(split_feature),
            split_threshold: Some(split_threshold),
            left,
            right,
            depth,
            size: indices.len(),
        }))
    }
    
    /// Compute path length for a sample
    fn compute_path_length(node: &Option<Box<ITreeNode>>, x: &[f64], current_height: usize) -> usize {
        match node {
            None => current_height,
            Some(node) => {
                match (node.split_feature, node.split_threshold) {
                    (Some(feature), Some(threshold)) => {
                        if x[feature] < threshold {
                            Self::compute_path_length(&node.left, x, current_height + 1)
                        } else {
                            Self::compute_path_length(&node.right, x, current_height + 1)
                        }
                    }
                    _ => current_height + Self::c_factor(node.size),
                }
            }
        }
    }
    
    /// Calculate the adjustment factor c(n)
    fn c_factor(n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        
        let n = n as f64;
        let h = 2.0 * (n - 1.0).ln() + 0.5772156649; // Euler's constant
        let c = 2.0 * h - (2.0 * (n - 1.0) / n);
        
        c.round() as usize
    }
    
    /// Helper method to extract numeric values from a column
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // Treat NA as 0 (or implement appropriate strategy)
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
                        values.push(value as f64);
                    } else {
                        values.push(0.0); // Treat NA as 0
                    }
                }
                Ok(values)
            },
            _ => Err(Error::Type(format!("Column type {:?} cannot be converted to numeric", col.column_type())))
        }
    }
}

impl Transformer for IsolationForest {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Extract only numeric columns
        let numeric_columns: Vec<String> = df.column_names()
            .into_iter()
            .filter(|col_name| {
                if let Ok(col_view) = df.column(col_name) {
                    col_view.as_float64().is_some() || col_view.as_int64().is_some()
                } else {
                    false
                }
            })
            .map(|s| s.to_string())
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for IsolationForest".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // Prepare data
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // Load data
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let column = df.column(col_name)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            for row_idx in 0..n_samples {
                if row_idx < values.len() {
                    data[row_idx][col_idx] = values[row_idx];
                }
            }
        }
        
        // Initialize random number generator
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random()),
        };
        
        // Determine subsampling size
        let sub_sample_size = match self.max_samples {
            Some(size) => size.min(n_samples),
            None => (n_samples as f64 * 0.632).min(256.0).max(1.0) as usize, // Empirical rule (paper recommendation)
        };
        
        // Calculate height limit
        let height_limit = (sub_sample_size as f64).log2().ceil() as usize;
        
        // Build decision trees
        self.trees.clear();
        for _ in 0..self.n_estimators {
            // Subsampling
            let mut indices: Vec<usize> = (0..n_samples).collect();
            
            // Shuffle indices
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
            
            indices.truncate(sub_sample_size);
            
            // Build tree
            let mut tree = ITree {
                height_limit,
                root: None,
            };
            
            tree.root = self.build_tree(&data, &indices, height_limit, 0, &mut rng);
            self.trees.push(tree);
        }
        
        // Calculate anomaly scores
        self.anomaly_scores = vec![0.0; n_samples];
        
        for i in 0..n_samples {
            let mut path_length_sum = 0.0;
            
            for tree in &self.trees {
                let path_length = Self::compute_path_length(&tree.root, &data[i], 0) as f64;
                path_length_sum += path_length;
            }
            
            let avg_path_length = path_length_sum / self.n_estimators as f64;
            let expected_path_length = Self::c_factor(sub_sample_size) as f64;
            
            // Normalized anomaly score
            // Higher score indicates more anomalous (range 0-1)
            self.anomaly_scores[i] = 2.0_f64.powf(-avg_path_length / expected_path_length);
        }
        
        // Calculate threshold based on contamination rate
        let mut sorted_scores = self.anomaly_scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending order
        
        let threshold_idx = (self.contamination * n_samples as f64).round() as usize;
        self.threshold = sorted_scores.get(threshold_idx.max(1) - 1).copied().unwrap_or(0.5);
        
        // Assign labels
        self.labels = self.anomaly_scores
            .iter()
            .map(|&score| if score >= self.threshold { 1 } else { -1 })
            .collect();
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "IsolationForest has not been fitted yet".to_string()
            ));
        }
        
        // Copy the original DataFrame
        let mut result = df.clone();
        
        // Add anomaly scores and predicted labels to the DataFrame
        let mut scores_float_col = Float64Column::new(self.anomaly_scores.clone());
        let mut labels_int_col = Int64Column::new(self.labels.clone());
        
        scores_float_col.set_name("anomaly_score");
        labels_int_col.set_name("anomaly");
        
        let scores_column = Column::Float64(scores_float_col);
        let labels_column = Column::Int64(labels_int_col);
        
        result.add_column("anomaly_score".to_string(), scores_column)?;
        result.add_column("anomaly".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// LOF (Local Outlier Factor) anomaly detection algorithm
pub struct LocalOutlierFactor {
    /// Number of neighbors
    n_neighbors: usize,
    /// Contamination rate (expected proportion of anomalies)
    contamination: f64,
    /// Metric
    metric: DistanceMetric,
    /// LOF scores for each sample
    lof_scores: Vec<f64>,
    /// Anomaly threshold
    threshold: f64,
    /// Anomaly flags (1: anomaly, -1: normal)
    labels: Vec<i64>,
    /// Feature names
    feature_names: Vec<String>,
    /// Training data
    data: Vec<Vec<f64>>,
    /// Whether the model has been fitted
    fitted: bool,
}

/// Distance metric
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

impl LocalOutlierFactor {
    /// Create a new LocalOutlierFactor instance
    pub fn new(
        n_neighbors: usize,
        contamination: f64,
        metric: DistanceMetric,
    ) -> Self {
        if contamination <= 0.0 || contamination >= 0.5 {
            panic!("Contamination must be in (0, 0.5)");
        }
        
        LocalOutlierFactor {
            n_neighbors,
            contamination,
            metric,
            lof_scores: Vec::new(),
            threshold: 0.0,
            labels: Vec::new(),
            feature_names: Vec::new(),
            data: Vec::new(),
            fitted: false,
        }
    }
    
    /// Get LOF scores
    pub fn lof_scores(&self) -> &[f64] {
        &self.lof_scores
    }
    
    /// Get anomaly flags (1: anomaly, -1: normal)
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }
    
    /// Compute distance between two data points
    fn compute_distance(&self, x: &[f64], y: &[f64]) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => {
                // Euclidean distance
                x.iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| (xi - yi).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            DistanceMetric::Manhattan => {
                // Manhattan distance
                x.iter()
                    .zip(y.iter())
                    .map(|(&xi, &yi)| (xi - yi).abs())
                    .sum()
            }
            DistanceMetric::Cosine => {
                // Cosine distance
                let dot_product: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();
                let norm_x: f64 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>().sqrt();
                let norm_y: f64 = y.iter().map(|&yi| yi.powi(2)).sum::<f64>().sqrt();
                
                if norm_x > 0.0 && norm_y > 0.0 {
                    1.0 - dot_product / (norm_x * norm_y)
                } else {
                    1.0 // Maximum distance
                }
            }
        }
    }
    
    /// Find k-nearest neighbors
    fn find_neighbors(&self, point_idx: usize, k: usize) -> Vec<(usize, f64)> {
        let n_samples = self.data.len();
        let mut distances = Vec::with_capacity(n_samples - 1);
        
        for i in 0..n_samples {
            if i != point_idx {
                let dist = self.compute_distance(&self.data[point_idx], &self.data[i]);
                distances.push((i, dist));
            }
        }
        
        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Return top k
        distances.into_iter().take(k.min(n_samples - 1)).collect()
    }
    
    /// Calculate reachability distance
    fn reachability_distance(&self, point_a_idx: usize, point_b_idx: usize, k_distance: f64) -> f64 {
        let direct_distance = self.compute_distance(&self.data[point_a_idx], &self.data[point_b_idx]);
        direct_distance.max(k_distance)
    }
    
    /// Helper method to extract numeric values from a column
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // Treat NA as 0 (or implement appropriate strategy)
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
                        values.push(value as f64);
                    } else {
                        values.push(0.0); // Treat NA as 0
                    }
                }
                Ok(values)
            },
            _ => Err(Error::Type(format!("Column type {:?} cannot be converted to numeric", col.column_type())))
        }
    }
}

impl Transformer for LocalOutlierFactor {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Extract only numeric columns
        let numeric_columns: Vec<String> = df.column_names()
            .into_iter()
            .filter(|col_name| {
                if let Ok(col_view) = df.column(col_name) {
                    col_view.as_float64().is_some() || col_view.as_int64().is_some()
                } else {
                    false
                }
            })
            .map(|s| s.to_string())
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for LocalOutlierFactor".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // Prepare data
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        self.data = vec![vec![0.0; n_features]; n_samples];
        
        // Load data
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let column = df.column(col_name)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            for row_idx in 0..n_samples {
                if row_idx < values.len() {
                    self.data[row_idx][col_idx] = values[row_idx];
                }
            }
        }
        
        // 1. Find k-nearest neighbors for each point
        let mut neighbors = Vec::with_capacity(n_samples);
        let mut k_distances = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            let k_neighbors = self.find_neighbors(i, self.n_neighbors);
            // Distance to the k-th point
            let k_dist = k_neighbors.last().map(|&(_, dist)| dist).unwrap_or(0.0);
            
            neighbors.push(k_neighbors);
            k_distances.push(k_dist);
        }
        
        // 2. Calculate local reachability density
        let mut lrd = vec![0.0; n_samples];
        
        for i in 0..n_samples {
            let mut sum_reachability = 0.0;
            
            for &(neighbor_idx, _) in &neighbors[i] {
                let reach_dist = self.reachability_distance(i, neighbor_idx, k_distances[neighbor_idx]);
                sum_reachability += reach_dist;
            }
            
            if !neighbors[i].is_empty() {
                lrd[i] = neighbors[i].len() as f64 / sum_reachability;
            } else {
                lrd[i] = 0.0;
            }
        }
        
        // 3. Calculate LOF scores
        self.lof_scores = vec![0.0; n_samples];
        
        for i in 0..n_samples {
            let mut lof_sum = 0.0;
            
            for &(neighbor_idx, _) in &neighbors[i] {
                lof_sum += lrd[neighbor_idx] / lrd[i];
            }
            
            if !neighbors[i].is_empty() {
                self.lof_scores[i] = lof_sum / neighbors[i].len() as f64;
            } else {
                self.lof_scores[i] = 1.0;  // Default value
            }
        }
        
        // Calculate threshold based on contamination rate
        let mut sorted_scores = self.lof_scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Descending order
        
        let threshold_idx = (self.contamination * n_samples as f64).round() as usize;
        self.threshold = sorted_scores.get(threshold_idx.max(1) - 1).copied().unwrap_or(1.0);
        
        // Assign labels (LOF above threshold indicates anomaly)
        self.labels = self.lof_scores
            .iter()
            .map(|&score| if score >= self.threshold { 1 } else { -1 })
            .collect();
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "LocalOutlierFactor has not been fitted yet".to_string()
            ));
        }
        
        // Check if data size matches
        if df.row_count() != self.lof_scores.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // Copy the original DataFrame
        let mut result = df.clone();
        
        // Add LOF scores and predicted labels to the DataFrame
        let mut scores_float_col = Float64Column::new(self.lof_scores.clone());
        let mut labels_int_col = Int64Column::new(self.labels.clone());
        
        scores_float_col.set_name("lof_score");
        labels_int_col.set_name("anomaly");
        
        let scores_column = Column::Float64(scores_float_col);
        let labels_column = Column::Int64(labels_int_col);
        
        result.add_column("lof_score".to_string(), scores_column)?;
        result.add_column("anomaly".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// One-Class SVM anomaly detection algorithm
pub struct OneClassSVM {
    /// Nu (threshold adjustment parameter)
    nu: f64,
    /// Kernel coefficient
    gamma: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence threshold
    tol: f64,
    /// Support vectors
    support_vectors: Vec<Vec<f64>>,
    /// Lagrange multipliers
    alphas: Vec<f64>,
    /// Bias
    rho: f64,
    /// Decision scores
    decision_scores: Vec<f64>,
    /// Anomaly flags (1: anomaly, -1: normal)
    labels: Vec<i64>,
    /// Feature names
    feature_names: Vec<String>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl OneClassSVM {
    /// Create a new OneClassSVM instance
    pub fn new(
        nu: f64,
        gamma: f64,
        max_iter: usize,
        tol: f64,
    ) -> Self {
        if nu <= 0.0 || nu >= 1.0 {
            panic!("Nu must be in (0, 1)");
        }
        
        OneClassSVM {
            nu,
            gamma,
            max_iter,
            tol,
            support_vectors: Vec::new(),
            alphas: Vec::new(),
            rho: 0.0,
            decision_scores: Vec::new(),
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// Get decision scores
    pub fn decision_scores(&self) -> &[f64] {
        &self.decision_scores
    }
    
    /// Get anomaly flags (1: anomaly, -1: normal)
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }
    
    /// Calculate RBF kernel
    fn rbf_kernel(&self, x: &[f64], y: &[f64]) -> f64 {
        let squared_distance = x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi).powi(2))
            .sum::<f64>();
        
        (-self.gamma * squared_distance).exp()
    }
    
    /// Calculate decision function value for a new sample
    fn decision_function(&self, x: &[f64]) -> f64 {
        let mut sum = 0.0;
        
        for (i, support_vector) in self.support_vectors.iter().enumerate() {
            sum += self.alphas[i] * self.rbf_kernel(x, support_vector);
        }
        
        sum - self.rho
    }
    
    /// Helper method to extract numeric values from a column
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // Treat NA as 0 (or implement appropriate strategy)
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
                let mut values = Vec::with_capacity(col.len());
                for i in 0..col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
                        values.push(value as f64);
                    } else {
                        values.push(0.0); // Treat NA as 0
                    }
                }
                Ok(values)
            },
            _ => Err(Error::Type(format!("Column type {:?} cannot be converted to numeric", col.column_type())))
        }
    }
}

impl Transformer for OneClassSVM {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Extract only numeric columns
        let numeric_columns: Vec<String> = df.column_names()
            .into_iter()
            .filter(|col_name| {
                if let Ok(col_view) = df.column(col_name) {
                    col_view.as_float64().is_some() || col_view.as_int64().is_some()
                } else {
                    false
                }
            })
            .map(|s| s.to_string())
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for OneClassSVM".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // Prepare data
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // Load data
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let column = df.column(col_name)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            for row_idx in 0..n_samples {
                if row_idx < values.len() {
                    data[row_idx][col_idx] = values[row_idx];
                }
            }
        }
        
        // This is a simplified implementation
        // A full implementation would use more efficient methods like the SMO algorithm
        
        // Calculate kernel matrix
        let mut kernel_matrix = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in 0..=i {
                let k_ij = self.rbf_kernel(&data[i], &data[j]);
                kernel_matrix[i][j] = k_ij;
                kernel_matrix[j][i] = k_ij;
            }
        }
        
        // Simplified optimization procedure (should use SMO algorithm in practice)
        let mut alphas = vec![0.0; n_samples];
        let mut g = vec![0.0; n_samples]; // Gradient
        
        // Initialize
        for i in 0..n_samples {
            alphas[i] = 1.0 / (n_samples as f64 * self.nu); // Uniform initialization
            g[i] = 0.0;
            for j in 0..n_samples {
                g[i] -= alphas[j] * kernel_matrix[i][j];
            }
        }
        
        // Optimization
        for _ in 0..self.max_iter {
            let mut max_diff = 0.0;
            
            for i in 0..n_samples {
                let old_alpha_i = alphas[i];
                
                // Gradient descent step
                let new_alpha_i = old_alpha_i - g[i] / kernel_matrix[i][i];
                
                // Clipping
                alphas[i] = new_alpha_i.max(0.0).min(1.0 / (n_samples as f64 * self.nu));
                
                let diff = alphas[i] - old_alpha_i;
                if diff.abs() > max_diff {
                    max_diff = diff.abs();
                }
                
                // Update gradient
                for j in 0..n_samples {
                    g[j] -= diff * kernel_matrix[i][j];
                }
            }
            
            // Convergence check
            if max_diff < self.tol {
                break;
            }
        }
        
        // Extract support vectors and Lagrange multipliers
        let mut support_vector_indices = Vec::new();
        for i in 0..n_samples {
            if alphas[i] > 1e-5 {
                support_vector_indices.push(i);
            }
        }
        
        self.support_vectors = support_vector_indices
            .iter()
            .map(|&i| data[i].clone())
            .collect();
        
        self.alphas = support_vector_indices
            .iter()
            .map(|&i| alphas[i])
            .collect();
        
        // Calculate bias (rho)
        let mut rho_sum = 0.0;
        let mut count = 0;
        
        for &i in &support_vector_indices {
            let mut f_i = 0.0;
            for (j, &sv_j) in support_vector_indices.iter().enumerate() {
                f_i += self.alphas[j] * kernel_matrix[i][sv_j];
            }
            
            if alphas[i] < (1.0 / (n_samples as f64 * self.nu)) - 1e-5 {
                rho_sum += f_i;
                count += 1;
            }
        }
        
        if count > 0 {
            self.rho = rho_sum / count as f64;
        } else {
            // Backup plan
            let mut f_sum = 0.0;
            for i in 0..n_samples {
                let mut f_i = 0.0;
                for (j, &sv_j) in support_vector_indices.iter().enumerate() {
                    f_i += self.alphas[j] * kernel_matrix[i][sv_j];
                }
                f_sum += f_i;
            }
            self.rho = f_sum / n_samples as f64;
        }
        
        // Calculate decision scores
        self.decision_scores = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            let mut score = 0.0;
            for (j, &sv_j) in support_vector_indices.iter().enumerate() {
                score += self.alphas[j] * kernel_matrix[i][sv_j];
            }
            score -= self.rho;
            self.decision_scores.push(score);
        }
        
        // Assign labels (score < 0 indicates anomaly)
        self.labels = self.decision_scores
            .iter()
            .map(|&score| if score < 0.0 { 1 } else { -1 })
            .collect();
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "OneClassSVM has not been fitted yet".to_string()
            ));
        }
        
        // Check if data size matches
        if df.row_count() != self.decision_scores.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // Copy the original DataFrame
        let mut result = df.clone();
        
        // Add decision scores and predicted labels to the DataFrame
        let mut scores_float_col = Float64Column::new(self.decision_scores.clone());
        let mut labels_int_col = Int64Column::new(self.labels.clone());
        
        scores_float_col.set_name("decision_score");
        labels_int_col.set_name("anomaly");
        
        let scores_column = Column::Float64(scores_float_col);
        let labels_column = Column::Int64(labels_int_col);
        
        result.add_column("decision_score".to_string(), scores_column)?;
        result.add_column("anomaly".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}