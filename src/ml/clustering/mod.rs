//! Clustering algorithms
//!
//! This module provides implementations of clustering algorithms for
//! unsupervised learning, such as K-means, hierarchical clustering,
//! and density-based clustering.

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
use std::collections::{HashMap, HashSet, VecDeque};

/// Linkage method for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    /// Single linkage (minimum distance between clusters)
    Single,
    /// Complete linkage (maximum distance between clusters)
    Complete,
    /// Average linkage (average distance between clusters)
    Average,
    /// Ward linkage (minimize variance increase)
    Ward,
}

/// Distance metric for clustering algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
}

/// K-means clustering algorithm
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Random seed for initialization
    pub random_seed: Option<u64>,
    /// Cluster assignments for each sample
    pub labels: Option<Vec<usize>>,
    /// Cluster centers
    pub centroids: Option<Vec<Vec<f64>>>,
    /// Inertia (within-cluster sum of squares)
    pub inertia: Option<f64>,
    /// Column names used for clustering
    pub feature_columns: Option<Vec<String>>,
}

impl KMeans {
    /// Create a new K-means instance
    pub fn new(n_clusters: usize) -> Self {
        KMeans {
            n_clusters,
            max_iter: 100,
            tol: 1e-4,
            random_seed: None,
            labels: None,
            centroids: None,
            inertia: None,
            feature_columns: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for initialization
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Specify feature columns to use
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = Some(columns);
        self
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, data: &DataFrame) -> Result<Vec<usize>> {
        if self.centroids.is_none() {
            return Err(Error::InvalidValue("KMeans not fitted".into()));
        }

        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;
        let feature_columns = match &self.feature_columns {
            Some(cols) => cols,
            None => return Err(Error::InvalidValue("Feature columns not specified".into())),
        };

        let n_samples = data.nrows();
        let mut labels = vec![0; n_samples];

        // Extract feature data
        let mut feature_data = Vec::with_capacity(n_samples);

        for row_idx in 0..n_samples {
            let mut row_data = Vec::with_capacity(feature_columns.len());

            for col_name in feature_columns {
                // Try to get column as f64 or convert appropriately
                if let Ok(col_f64) = data.get_column::<f64>(col_name) {
                    let numeric_col = col_f64.values();
                    if row_idx < numeric_col.len() {
                        row_data.push(numeric_col[row_idx]);
                    } else {
                        return Err(Error::IndexOutOfBounds {
                            index: row_idx,
                            size: numeric_col.len(),
                        });
                    }
                } else {
                    return Err(Error::InvalidInput(format!(
                        "Column {} is not numeric",
                        col_name
                    )));
                }
            }

            feature_data.push(row_data);
        }

        // Assign each sample to nearest centroid
        for (i, sample) in feature_data.iter().enumerate() {
            let mut min_dist = f64::MAX;
            let mut min_cluster = 0;

            for (j, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance(sample, centroid);

                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = j;
                }
            }

            labels[i] = min_cluster;
        }

        Ok(labels)
    }
}

impl UnsupervisedModel for KMeans {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        // Determine feature columns
        let feature_columns = match &self.feature_columns {
            Some(cols) => cols.clone(),
            None => data.column_names(),
        };

        // Extract feature data
        let n_samples = data.nrows();
        let n_features = feature_columns.len();

        let mut feature_data = Vec::with_capacity(n_samples);

        for row_idx in 0..n_samples {
            let mut row_data = Vec::with_capacity(n_features);

            for col_name in &feature_columns {
                // Try to get column as f64 or convert appropriately
                if let Ok(col_f64) = data.get_column::<f64>(col_name) {
                    let numeric_col = col_f64.values();
                    if row_idx < numeric_col.len() {
                        row_data.push(numeric_col[row_idx]);
                    } else {
                        return Err(Error::IndexOutOfBounds {
                            index: row_idx,
                            size: numeric_col.len(),
                        });
                    }
                } else {
                    return Err(Error::InvalidInput(format!(
                        "Column {} is not numeric",
                        col_name
                    )));
                }
            }

            feature_data.push(row_data);
        }

        // Initialize centroids (randomly select k samples)
        // A real implementation would use k-means++ or similar
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut seed_bytes = [0u8; 32];
                scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
                StdRng::from_seed(seed_bytes)
            }
        };

        let mut centroid_indices = Vec::with_capacity(self.n_clusters);
        let indices: Vec<usize> = (0..n_samples).collect();

        // Sample without replacement
        // In rand 0.9, we need to use slice_choose instead of choose_multiple
        let mut indices_copy = indices.clone();
        indices_copy.shuffle(&mut rng);
        for idx in indices_copy.iter().take(self.n_clusters.min(n_samples)) {
            centroid_indices.push(*idx);
        }

        // Initialize centroids with selected samples
        let mut centroids = Vec::with_capacity(self.n_clusters);
        for &idx in &centroid_indices {
            centroids.push(feature_data[idx].clone());
        }

        // Perform k-means clustering iterations
        let mut labels = vec![0; n_samples];
        let mut prev_inertia = f64::MAX;
        let mut inertia = 0.0;

        for _ in 0..self.max_iter {
            // Assign samples to nearest centroid
            inertia = 0.0;

            for (i, sample) in feature_data.iter().enumerate() {
                let mut min_dist = f64::MAX;
                let mut min_cluster = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(sample, centroid);

                    if dist < min_dist {
                        min_dist = dist;
                        min_cluster = j;
                    }
                }

                labels[i] = min_cluster;
                inertia += min_dist;
            }

            // Check convergence
            if (prev_inertia - inertia).abs() < self.tol {
                break;
            }

            prev_inertia = inertia;

            // Update centroids
            let mut new_centroids = vec![vec![0.0; n_features]; self.n_clusters];
            let mut counts = vec![0; self.n_clusters];

            for (i, sample) in feature_data.iter().enumerate() {
                let cluster = labels[i];
                counts[cluster] += 1;

                for (j, &val) in sample.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
            }

            // Calculate new centroids as mean of assigned points
            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[i] as f64;
                    }
                }
            }

            // Handle empty clusters by reinitializing them
            for i in 0..self.n_clusters {
                if counts[i] == 0 {
                    // Find the point furthest from its centroid
                    let mut max_dist = 0.0;
                    let mut max_idx = 0;

                    for (j, sample) in feature_data.iter().enumerate() {
                        let cluster = labels[j];
                        let dist = euclidean_distance(sample, &centroids[cluster]);

                        if dist > max_dist {
                            max_dist = dist;
                            max_idx = j;
                        }
                    }

                    // Assign this point to the empty cluster
                    new_centroids[i] = feature_data[max_idx].clone();
                }
            }

            centroids = new_centroids;
        }

        // Store results
        self.labels = Some(labels);
        self.centroids = Some(centroids);
        self.inertia = Some(inertia);
        self.feature_columns = Some(feature_columns);

        Ok(())
    }

    fn transform(&self, data: &DataFrame) -> Result<DataFrame> {
        // K-means transform returns the distance to each centroid
        if self.centroids.is_none() {
            return Err(Error::InvalidValue("KMeans not fitted".into()));
        }

        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| Error::InvalidOperation("Model not fitted. Call fit() first.".into()))?;
        let feature_columns = match &self.feature_columns {
            Some(cols) => cols,
            None => return Err(Error::InvalidValue("Feature columns not specified".into())),
        };

        let n_samples = data.nrows();
        let n_clusters = centroids.len();

        // Extract feature data
        let mut feature_data = Vec::with_capacity(n_samples);

        for row_idx in 0..n_samples {
            let mut row_data = Vec::with_capacity(feature_columns.len());

            for col_name in feature_columns {
                // Try to get column as f64 or convert appropriately
                if let Ok(col_f64) = data.get_column::<f64>(col_name) {
                    let numeric_col = col_f64.values();
                    if row_idx < numeric_col.len() {
                        row_data.push(numeric_col[row_idx]);
                    } else {
                        return Err(Error::IndexOutOfBounds {
                            index: row_idx,
                            size: numeric_col.len(),
                        });
                    }
                } else {
                    return Err(Error::InvalidInput(format!(
                        "Column {} is not numeric",
                        col_name
                    )));
                }
            }

            feature_data.push(row_data);
        }

        // Compute distances to centroids
        let mut result = DataFrame::new();

        for c in 0..n_clusters {
            let mut distances = Vec::with_capacity(n_samples);

            for sample in &feature_data {
                let dist = euclidean_distance(sample, &centroids[c]);
                distances.push(dist);
            }

            result.add_column(
                format!("distance_to_cluster_{}", c),
                crate::series::Series::new(distances, Some(format!("distance_to_cluster_{}", c)))?,
            )?;
        }

        Ok(result)
    }
}

impl ModelEvaluator for KMeans {
    fn evaluate(&self, test_data: &DataFrame, _test_target: &str) -> Result<ModelMetrics> {
        // K-means evaluation metrics include inertia (within-cluster sum of squares)
        let mut metrics = ModelMetrics::new();

        if let Some(inertia) = self.inertia {
            metrics.add_metric("inertia", inertia);
        }

        // Compute silhouette score for test data
        if let Some(labels) = &self.labels {
            if let Some(centroids) = &self.centroids {
                let silhouette =
                    compute_silhouette(test_data, labels, centroids, &self.feature_columns)?;
                metrics.add_metric("silhouette_score", silhouette);
            }
        }

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        // K-means doesn't typically use cross-validation in the same way as supervised models
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for K-means clustering".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calculate Manhattan distance between two vectors
fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
}

/// Calculate Cosine distance between two vectors (1 - cosine_similarity)
fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    let similarity = dot / (norm_a * norm_b);
    // Clamp to [-1,1] to guard against floating-point rounding
    1.0 - similarity.clamp(-1.0, 1.0)
}

/// Dispatch distance computation according to the chosen metric
fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::Manhattan => manhattan_distance(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
    }
}

// ---------------------------------------------------------------------------
// Feature extraction helper
// ---------------------------------------------------------------------------

/// Extract feature matrix from a DataFrame given optional column names.
/// If `feature_columns` is None, all columns in the frame are used.
fn extract_features(
    data: &DataFrame,
    feature_columns: &Option<Vec<String>>,
) -> Result<(Vec<Vec<f64>>, Vec<String>)> {
    let columns: Vec<String> = match feature_columns {
        Some(cols) => cols.clone(),
        None => data.column_names(),
    };

    let n_samples = data.nrows();
    let mut feature_data: Vec<Vec<f64>> = vec![Vec::with_capacity(columns.len()); n_samples];

    for col_name in &columns {
        match data.get_column::<f64>(col_name) {
            Ok(col) => {
                let values = col.values();
                for (row_idx, row) in feature_data.iter_mut().enumerate() {
                    if row_idx < values.len() {
                        row.push(values[row_idx]);
                    } else {
                        return Err(Error::IndexOutOfBounds {
                            index: row_idx,
                            size: values.len(),
                        });
                    }
                }
            }
            Err(_) => {
                return Err(Error::InvalidInput(format!(
                    "Column {} is not numeric",
                    col_name
                )));
            }
        }
    }

    Ok((feature_data, columns))
}

// ---------------------------------------------------------------------------
// Silhouette coefficient
// ---------------------------------------------------------------------------

/// Compute the mean silhouette coefficient.
///
/// For each point *i* with cluster label *c_i*:
///   a_i = mean intra-cluster distance to all other points in c_i
///   b_i = min over clusters k ≠ c_i of mean distance to all points in k
///   s_i = (b_i − a_i) / max(a_i, b_i)   (0 if singleton cluster)
///
/// Returns the mean of all s_i values.  Returns 0.0 if fewer than 2 clusters
/// are present or if all points belong to a single cluster.
fn compute_silhouette(
    data: &DataFrame,
    labels: &[usize],
    _centroids: &[Vec<f64>],
    feature_columns: &Option<Vec<String>>,
) -> Result<f64> {
    if labels.is_empty() {
        return Ok(0.0);
    }

    // Determine the set of active cluster ids (exclude noise sentinel u32::MAX)
    let cluster_ids: HashSet<usize> = labels.iter().cloned().collect();
    let n_clusters = cluster_ids.len();
    if n_clusters < 2 {
        return Ok(0.0);
    }

    let (feature_data, _) = extract_features(data, feature_columns)?;
    let n_samples = feature_data.len();

    if n_samples != labels.len() {
        return Err(Error::InvalidValue(
            "labels length does not match data row count".into(),
        ));
    }

    // Build a map: cluster_id -> list of point indices
    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &lbl) in labels.iter().enumerate() {
        cluster_members.entry(lbl).or_default().push(i);
    }

    let mut silhouette_sum = 0.0;
    let mut count = 0usize;

    for i in 0..n_samples {
        let c_i = labels[i];
        let members_ci = &cluster_members[&c_i];

        // a_i: mean distance to other points in same cluster
        let a_i = if members_ci.len() <= 1 {
            // Singleton — s_i = 0 by definition
            0.0
        } else {
            let sum: f64 = members_ci
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| euclidean_distance(&feature_data[i], &feature_data[j]))
                .sum();
            sum / (members_ci.len() - 1) as f64
        };

        if members_ci.len() <= 1 {
            // Singleton contributes 0
            silhouette_sum += 0.0;
            count += 1;
            continue;
        }

        // b_i: min mean distance over all other clusters
        let mut b_i = f64::MAX;
        for (&other_cluster, other_members) in &cluster_members {
            if other_cluster == c_i {
                continue;
            }
            if other_members.is_empty() {
                continue;
            }
            let mean_dist: f64 = other_members
                .iter()
                .map(|&j| euclidean_distance(&feature_data[i], &feature_data[j]))
                .sum::<f64>()
                / other_members.len() as f64;
            if mean_dist < b_i {
                b_i = mean_dist;
            }
        }

        let s_i = if b_i == f64::MAX {
            // No other cluster found (degenerate case)
            0.0
        } else {
            let denom = a_i.max(b_i);
            if denom == 0.0 {
                0.0
            } else {
                (b_i - a_i) / denom
            }
        };

        silhouette_sum += s_i;
        count += 1;
    }

    if count == 0 {
        return Ok(0.0);
    }

    Ok(silhouette_sum / count as f64)
}

// ---------------------------------------------------------------------------
// Agglomerative Hierarchical Clustering
// ---------------------------------------------------------------------------

/// Agglomerative hierarchical clustering
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering {
    /// Number of clusters
    pub n_clusters: usize,
    /// Linkage method
    pub linkage: Linkage,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Cluster assignments for each sample
    pub labels: Option<Vec<usize>>,
    /// Feature columns used for clustering
    pub feature_columns: Option<Vec<String>>,
}

impl AgglomerativeClustering {
    /// Create a new AgglomerativeClustering instance
    pub fn new(n_clusters: usize) -> Self {
        AgglomerativeClustering {
            n_clusters,
            linkage: Linkage::Ward,
            metric: DistanceMetric::Euclidean,
            labels: None,
            feature_columns: None,
        }
    }

    /// Set linkage method
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Set distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Specify feature columns to use
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = Some(columns);
        self
    }
}

impl UnsupervisedModel for AgglomerativeClustering {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        let (feature_data, used_columns) = extract_features(data, &self.feature_columns)?;
        let n_samples = feature_data.len();

        if n_samples == 0 {
            return Err(Error::InvalidValue("Empty dataset".into()));
        }

        let target_clusters = self.n_clusters.min(n_samples);

        // Precompute full pairwise distance matrix
        let mut dist_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let d = compute_distance(&feature_data[i], &feature_data[j], self.metric);
                dist_matrix[i][j] = d;
                dist_matrix[j][i] = d;
            }
        }

        // Each point starts in its own cluster; we represent clusters as sets of
        // original point indices.
        let mut clusters: Vec<Option<Vec<usize>>> = (0..n_samples).map(|i| Some(vec![i])).collect();
        // Track which cluster slots are active
        let mut active: Vec<usize> = (0..n_samples).collect();

        while active.len() > target_clusters {
            // Find the two active clusters with the smallest linkage distance
            let mut best_dist = f64::MAX;
            let mut best_a = 0usize;
            let mut best_b = 0usize;

            for ai in 0..active.len() {
                for bi in (ai + 1)..active.len() {
                    let idx_a = active[ai];
                    let idx_b = active[bi];
                    let pts_a = clusters[idx_a]
                        .as_ref()
                        .expect("active cluster slot is always Some");
                    let pts_b = clusters[idx_b]
                        .as_ref()
                        .expect("active cluster slot is always Some");

                    let d =
                        linkage_distance(pts_a, pts_b, &dist_matrix, &feature_data, self.linkage);

                    if d < best_dist {
                        best_dist = d;
                        best_a = idx_a;
                        best_b = idx_b;
                    }
                }
            }

            // Merge cluster best_b into cluster best_a
            let pts_b = clusters[best_b]
                .take()
                .expect("active cluster slot is always Some");
            let pts_a = clusters[best_a]
                .as_mut()
                .expect("active cluster slot is always Some");
            pts_a.extend(pts_b);

            // Remove best_b from the active list
            active.retain(|&idx| idx != best_b);
        }

        // Assign final integer labels (0 .. target_clusters-1)
        let mut labels = vec![0usize; n_samples];
        for (cluster_label, &slot) in active.iter().enumerate() {
            for &pt in clusters[slot]
                .as_ref()
                .expect("active cluster slot is always Some")
            {
                labels[pt] = cluster_label;
            }
        }

        self.labels = Some(labels);
        self.feature_columns = Some(used_columns);

        Ok(())
    }

    fn transform(&self, _data: &DataFrame) -> Result<DataFrame> {
        // AgglomerativeClustering doesn't support transform
        Err(Error::InvalidOperation(
            "AgglomerativeClustering does not support transform".into(),
        ))
    }
}

/// Compute inter-cluster linkage distance according to the requested criterion.
fn linkage_distance(
    pts_a: &[usize],
    pts_b: &[usize],
    dist_matrix: &[Vec<f64>],
    feature_data: &[Vec<f64>],
    linkage: Linkage,
) -> f64 {
    match linkage {
        Linkage::Single => {
            // min over all cross pairs
            let mut min_d = f64::MAX;
            for &i in pts_a {
                for &j in pts_b {
                    let d = dist_matrix[i][j];
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
            min_d
        }
        Linkage::Complete => {
            // max over all cross pairs
            let mut max_d: f64 = 0.0;
            for &i in pts_a {
                for &j in pts_b {
                    let d = dist_matrix[i][j];
                    if d > max_d {
                        max_d = d;
                    }
                }
            }
            max_d
        }
        Linkage::Average => {
            // mean over all cross pairs
            let mut sum = 0.0;
            let count = pts_a.len() * pts_b.len();
            for &i in pts_a {
                for &j in pts_b {
                    sum += dist_matrix[i][j];
                }
            }
            if count == 0 {
                0.0
            } else {
                sum / count as f64
            }
        }
        Linkage::Ward => {
            // Ward's criterion: Δvariance = (n_a * n_b / (n_a + n_b)) * ||centroid_a - centroid_b||²
            let n_a = pts_a.len();
            let n_b = pts_b.len();
            if n_a == 0 || n_b == 0 {
                return 0.0;
            }
            let n_features = feature_data[0].len();
            let mut centroid_a = vec![0.0f64; n_features];
            let mut centroid_b = vec![0.0f64; n_features];
            for &i in pts_a {
                for (k, val) in centroid_a.iter_mut().enumerate() {
                    *val += feature_data[i][k];
                }
            }
            for &j in pts_b {
                for (k, val) in centroid_b.iter_mut().enumerate() {
                    *val += feature_data[j][k];
                }
            }
            for val in centroid_a.iter_mut() {
                *val /= n_a as f64;
            }
            for val in centroid_b.iter_mut() {
                *val /= n_b as f64;
            }
            let sq_dist: f64 = centroid_a
                .iter()
                .zip(centroid_b.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum();
            (n_a as f64 * n_b as f64 / (n_a + n_b) as f64) * sq_dist
        }
    }
}

impl ModelEvaluator for AgglomerativeClustering {
    fn evaluate(&self, test_data: &DataFrame, _test_target: &str) -> Result<ModelMetrics> {
        let mut metrics = ModelMetrics::new();

        if let Some(labels) = &self.labels {
            // Build dummy centroids vector (silhouette ignores centroids, uses labels only)
            let dummy_centroids: Vec<Vec<f64>> = Vec::new();
            let silhouette =
                compute_silhouette(test_data, labels, &dummy_centroids, &self.feature_columns)?;
            metrics.add_metric("silhouette_score", silhouette);
        }

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for hierarchical clustering".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// DBSCAN
// ---------------------------------------------------------------------------

/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
#[derive(Debug, Clone)]
pub struct DBSCAN {
    /// Neighborhood radius epsilon
    pub eps: f64,
    /// Minimum number of points to form a core point
    pub min_samples: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Cluster assignments for each sample (-1 for noise points)
    pub labels: Option<Vec<i32>>,
    /// Feature columns used for clustering
    pub feature_columns: Option<Vec<String>>,
}

impl DBSCAN {
    /// Create a new DBSCAN instance
    pub fn new(eps: f64, min_samples: usize) -> Self {
        DBSCAN {
            eps,
            min_samples,
            metric: DistanceMetric::Euclidean,
            labels: None,
            feature_columns: None,
        }
    }

    /// Set distance metric
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Specify feature columns to use
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = Some(columns);
        self
    }
}

impl UnsupervisedModel for DBSCAN {
    fn fit(&mut self, data: &DataFrame) -> Result<()> {
        let (feature_data, used_columns) = extract_features(data, &self.feature_columns)?;
        let n_samples = feature_data.len();

        if n_samples == 0 {
            self.labels = Some(Vec::new());
            self.feature_columns = Some(used_columns);
            return Ok(());
        }

        // Precompute pairwise distances
        let mut dist_matrix: Vec<Vec<f64>> = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let d = compute_distance(&feature_data[i], &feature_data[j], self.metric);
                dist_matrix[i][j] = d;
                dist_matrix[j][i] = d;
            }
        }

        // For each point determine its eps-neighborhood
        let mut neighborhoods: Vec<Vec<usize>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let nbrs: Vec<usize> = (0..n_samples)
                .filter(|&j| j != i && dist_matrix[i][j] <= self.eps)
                .collect();
            neighborhoods.push(nbrs);
        }

        // -1 = unvisited; label assignment happens during BFS
        let mut labels: Vec<i32> = vec![-1; n_samples];
        let mut cluster_id: i32 = -1;

        for i in 0..n_samples {
            // Already visited
            if labels[i] != -1 {
                continue;
            }

            // Check if core point
            if neighborhoods[i].len() < self.min_samples {
                // Mark as noise for now (may be updated later as border point)
                labels[i] = -1;
                continue;
            }

            // Start a new cluster
            cluster_id += 1;
            labels[i] = cluster_id;

            // BFS expansion
            let mut queue: VecDeque<usize> = neighborhoods[i].iter().cloned().collect();
            while let Some(q) = queue.pop_front() {
                if labels[q] == -1 {
                    // Was noise — promote to border point of this cluster
                    labels[q] = cluster_id;
                } else {
                    // Already assigned to a cluster — skip
                    continue;
                }

                // If q is itself a core point, expand its neighborhood
                if neighborhoods[q].len() >= self.min_samples {
                    for &nbr in &neighborhoods[q] {
                        if labels[nbr] == -1 {
                            queue.push_back(nbr);
                        }
                    }
                }
            }
        }

        self.labels = Some(labels);
        self.feature_columns = Some(used_columns);

        Ok(())
    }

    fn transform(&self, _data: &DataFrame) -> Result<DataFrame> {
        // DBSCAN doesn't support transform
        Err(Error::InvalidOperation(
            "DBSCAN does not support transform".into(),
        ))
    }
}

impl ModelEvaluator for DBSCAN {
    fn evaluate(&self, test_data: &DataFrame, _test_target: &str) -> Result<ModelMetrics> {
        let mut metrics = ModelMetrics::new();

        if let Some(i32_labels) = &self.labels {
            // Convert i32 labels to usize labels for silhouette, skipping noise points (-1).
            // Noise points are excluded from the silhouette computation entirely.
            //
            // Build a sub-set of the data without noise points, then compute silhouette.
            // Because compute_silhouette receives the full DataFrame we instead remap
            // non-noise cluster ids to contiguous usize and pass ALL labels, but noise
            // points will end up in a "cluster" keyed by usize::MAX, which we handle by
            // not counting singletons in the silhouette computation.  The cleaner
            // approach is to derive usize labels for silhouette and pass only non-noise
            // rows — but since compute_silhouette takes a &DataFrame we use an
            // alternative: map -1 to a unique large id so all "noise" points share one
            // cluster, then let compute_silhouette handle it (it will give that cluster
            // s_i ≈ 0 since noise points are far from each other).
            //
            // Simplest correct approach: compute silhouette only on non-noise points by
            // building a temporary in-memory Vec and calling the inner silhouette logic
            // directly rather than through the DataFrame wrapper.
            let non_noise: Vec<(usize, usize)> = i32_labels
                .iter()
                .enumerate()
                .filter(|(_, &lbl)| lbl >= 0)
                .map(|(i, &lbl)| (i, lbl as usize))
                .collect();

            if non_noise.len() >= 2 {
                let (feature_data, _) = extract_features(test_data, &self.feature_columns)?;
                // Re-index to contiguous labels starting from 0
                let mut label_remap: HashMap<usize, usize> = HashMap::new();
                let mut next_id = 0usize;
                let mut subset_labels: Vec<usize> = Vec::with_capacity(non_noise.len());
                let mut subset_data: Vec<Vec<f64>> = Vec::with_capacity(non_noise.len());

                for (orig_idx, orig_lbl) in &non_noise {
                    let new_lbl = *label_remap.entry(*orig_lbl).or_insert_with(|| {
                        let id = next_id;
                        next_id += 1;
                        id
                    });
                    subset_labels.push(new_lbl);
                    if *orig_idx < feature_data.len() {
                        subset_data.push(feature_data[*orig_idx].clone());
                    }
                }

                let n_unique = label_remap.len();
                if n_unique >= 2 && !subset_data.is_empty() {
                    let silhouette = compute_silhouette_raw(&subset_data, &subset_labels, n_unique);
                    metrics.add_metric("silhouette_score", silhouette);
                } else {
                    metrics.add_metric("silhouette_score", 0.0);
                }
            } else {
                metrics.add_metric("silhouette_score", 0.0);
            }
        }

        Ok(metrics)
    }

    fn cross_validate(
        &self,
        _data: &DataFrame,
        _target: &str,
        _folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for DBSCAN clustering".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Raw silhouette (works directly on a feature matrix, no DataFrame required)
// ---------------------------------------------------------------------------

/// Compute silhouette coefficient directly from a feature matrix and label vector.
/// `n_clusters` is the number of distinct cluster ids (0 .. n_clusters-1).
fn compute_silhouette_raw(feature_data: &[Vec<f64>], labels: &[usize], n_clusters: usize) -> f64 {
    if n_clusters < 2 || labels.is_empty() {
        return 0.0;
    }

    let n_samples = feature_data.len();
    // Build cluster membership map
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl < n_clusters {
            cluster_members[lbl].push(i);
        }
    }

    let mut silhouette_sum = 0.0;
    let mut count = 0usize;

    for i in 0..n_samples {
        let c_i = labels[i];
        if c_i >= n_clusters {
            continue;
        }
        let members_ci = &cluster_members[c_i];

        let a_i = if members_ci.len() <= 1 {
            0.0
        } else {
            let sum: f64 = members_ci
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| euclidean_distance(&feature_data[i], &feature_data[j]))
                .sum();
            sum / (members_ci.len() - 1) as f64
        };

        if members_ci.len() <= 1 {
            silhouette_sum += 0.0;
            count += 1;
            continue;
        }

        let mut b_i = f64::MAX;
        for k in 0..n_clusters {
            if k == c_i {
                continue;
            }
            let other = &cluster_members[k];
            if other.is_empty() {
                continue;
            }
            let mean_d: f64 = other
                .iter()
                .map(|&j| euclidean_distance(&feature_data[i], &feature_data[j]))
                .sum::<f64>()
                / other.len() as f64;
            if mean_d < b_i {
                b_i = mean_d;
            }
        }

        let s_i = if b_i == f64::MAX {
            0.0
        } else {
            let denom = a_i.max(b_i);
            if denom == 0.0 {
                0.0
            } else {
                (b_i - a_i) / denom
            }
        };

        silhouette_sum += s_i;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        silhouette_sum / count as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::series::Series;

    /// Build a DataFrame from two equal-length column slices
    fn make_df(xs: &[f64], ys: &[f64]) -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(xs.to_vec(), Some("x".to_string())).unwrap(),
        )
        .unwrap();
        df.add_column(
            "y".to_string(),
            Series::new(ys.to_vec(), Some("y".to_string())).unwrap(),
        )
        .unwrap();
        df
    }

    // -----------------------------------------------------------------------
    // DBSCAN tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dbscan_two_blobs() {
        // Two well-separated blobs: 5 points near (0,0) and 5 near (10,10)
        let xs = [0.1, -0.1, 0.2, -0.2, 0.0, 9.9, 10.1, 9.8, 10.2, 10.0];
        let ys = [0.1, -0.1, -0.2, 0.2, 0.0, 9.9, 10.1, 10.2, 9.8, 10.0];
        let df = make_df(&xs, &ys);

        let mut dbscan = DBSCAN::new(2.0, 2).with_columns(vec!["x".to_string(), "y".to_string()]);
        dbscan.fit(&df).unwrap();

        let labels = dbscan.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 10);

        // All points should be assigned (no noise)
        assert!(
            labels.iter().all(|&l| l >= 0),
            "Expected no noise points, got: {:?}",
            labels
        );

        // Exactly 2 clusters (max label == 1 when cluster ids are 0 and 1)
        let max_label = *labels.iter().max().unwrap();
        assert_eq!(
            max_label, 1,
            "Expected exactly 2 clusters, got max label = {}",
            max_label
        );
    }

    #[test]
    fn test_dbscan_noise() {
        // A tight cluster of 4 points plus 3 clear outliers
        let xs = [0.0, 0.1, -0.1, 0.05, 100.0, -100.0, 50.0];
        let ys = [0.0, 0.1, -0.1, 0.05, 100.0, -100.0, 50.0];
        let df = make_df(&xs, &ys);

        let mut dbscan = DBSCAN::new(1.0, 2).with_columns(vec!["x".to_string(), "y".to_string()]);
        dbscan.fit(&df).unwrap();

        let labels = dbscan.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 7);

        // At least one noise point (-1) must exist
        assert!(
            labels.iter().any(|&l| l == -1),
            "Expected at least one noise point, got: {:?}",
            labels
        );
    }

    // -----------------------------------------------------------------------
    // Agglomerative tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_agglomerative_two_blobs() {
        // Two well-separated blobs
        let xs = [0.1, -0.1, 0.2, -0.2, 0.0, 9.9, 10.1, 9.8, 10.2, 10.0];
        let ys = [0.1, -0.1, -0.2, 0.2, 0.0, 9.9, 10.1, 10.2, 9.8, 10.0];
        let df = make_df(&xs, &ys);

        let mut agg =
            AgglomerativeClustering::new(2).with_columns(vec!["x".to_string(), "y".to_string()]);
        agg.fit(&df).unwrap();

        let labels = agg.labels.as_ref().unwrap();
        assert_eq!(labels.len(), 10);

        // Exactly 2 distinct labels, both in range [0, 2)
        let unique: HashSet<usize> = labels.iter().cloned().collect();
        assert_eq!(
            unique.len(),
            2,
            "Expected exactly 2 distinct cluster labels, got: {:?}",
            unique
        );
        assert!(
            unique.iter().all(|&l| l < 2),
            "Labels out of range: {:?}",
            unique
        );
    }

    // -----------------------------------------------------------------------
    // Silhouette tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_silhouette_perfect() {
        // Two clusters with near-perfect separation
        let xs: Vec<f64> = [
            0.0, 0.05, -0.05, 0.02, -0.02, 100.0, 100.05, 99.95, 100.02, 99.98,
        ]
        .to_vec();
        let ys: Vec<f64> = [
            0.0, 0.05, -0.05, -0.02, 0.02, 100.0, 100.05, 99.95, 99.98, 100.02,
        ]
        .to_vec();
        let df = make_df(&xs, &ys);

        // Labels: first 5 → cluster 0, last 5 → cluster 1
        let labels: Vec<usize> = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1].to_vec();
        let dummy_centroids: Vec<Vec<f64>> = Vec::new();
        let feature_cols = Some(vec!["x".to_string(), "y".to_string()]);

        let score = compute_silhouette(&df, &labels, &dummy_centroids, &feature_cols).unwrap();
        assert!(
            score > 0.8,
            "Expected silhouette score > 0.8 for perfect separation, got {}",
            score
        );
    }

    // -----------------------------------------------------------------------
    // KMeans silhouette via evaluate()
    // -----------------------------------------------------------------------

    #[test]
    fn test_kmeans_silhouette_evaluates() {
        // Two well-separated blobs
        let xs = [0.1_f64, -0.1, 0.2, -0.2, 0.0, 9.9, 10.1, 9.8, 10.2, 10.0];
        let ys = [0.1_f64, -0.1, -0.2, 0.2, 0.0, 9.9, 10.1, 10.2, 9.8, 10.0];
        let df = make_df(&xs, &ys);

        let mut km = KMeans::new(2)
            .random_seed(42)
            .with_columns(vec!["x".to_string(), "y".to_string()]);
        km.fit(&df).unwrap();

        let metrics = km.evaluate(&df, "").unwrap();
        let score = metrics
            .get_metric("silhouette_score")
            .copied()
            .unwrap_or(0.0);
        assert!(
            score > 0.8,
            "Expected KMeans silhouette > 0.8 for well-separated blobs, got {}",
            score
        );
    }
}
