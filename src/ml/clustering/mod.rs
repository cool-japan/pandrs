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

/// Centroid initialization strategy for [`KMeans`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KMeansInit {
    /// k-means++ (Arthur & Vassilvitskii, 2007): spreads initial centroids out
    /// by sampling with probability proportional to squared distance from the
    /// nearest already-chosen centroid. This is scikit-learn's default and
    /// reliably produces lower-inertia results than uniform random picks.
    KMeansPlusPlus,
    /// Uniform random selection of `n_clusters` distinct data points.
    Random,
}

/// K-means clustering algorithm
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (compared against the change in inertia,
    /// i.e. the sum of squared distances to the assigned centroid)
    pub tol: f64,
    /// Random seed for initialization
    pub random_seed: Option<u64>,
    /// Centroid initialization strategy (default: k-means++)
    pub init: KMeansInit,
    /// Number of independent initializations to run; the run with the lowest
    /// inertia is kept (default: 10, matching scikit-learn's historical
    /// default for both `random` and `k-means++` initialization)
    pub n_init: usize,
    /// Cluster assignments for each sample
    pub labels: Option<Vec<usize>>,
    /// Cluster centers
    pub centroids: Option<Vec<Vec<f64>>>,
    /// Inertia: sum of squared distances of samples to their closest
    /// cluster center (matches scikit-learn's definition)
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
            init: KMeansInit::KMeansPlusPlus,
            n_init: 10,
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

    /// Set the centroid initialization strategy
    pub fn init(mut self, init: KMeansInit) -> Self {
        self.init = init;
        self
    }

    /// Set the number of independent initializations to run (the run with
    /// the lowest inertia is kept). Must be at least 1.
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init.max(1);
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
                let dist = squared_euclidean_distance(sample, centroid)?;

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
            None => data.column_names().to_vec(),
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

        let k = self.n_clusters.min(n_samples);
        if k == 0 {
            return Err(Error::InvalidValue(
                "KMeans requires n_clusters >= 1 and at least one sample".into(),
            ));
        }

        let mut rng: StdRng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut seed_bytes = [0u8; 32];
                scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
                StdRng::from_seed(seed_bytes)
            }
        };

        // Run `n_init` independent initializations and keep the one with the
        // lowest final inertia (scikit-learn semantics). Each run draws its
        // own centroid seed from the shared, already-seeded RNG so the whole
        // fit is reproducible given `random_seed`.
        let n_init = self.n_init.max(1);
        let mut best: Option<(Vec<usize>, Vec<Vec<f64>>, f64)> = None;

        for _ in 0..n_init {
            let init_centroids = match self.init {
                KMeansInit::KMeansPlusPlus => kmeans_plusplus_init(&feature_data, k, &mut rng)?,
                KMeansInit::Random => random_init(&feature_data, k, &mut rng),
            };

            let run = run_single_kmeans(
                &feature_data,
                n_features,
                k,
                self.max_iter,
                self.tol,
                init_centroids,
            )?;

            let is_better = match &best {
                None => true,
                Some((_, _, best_inertia)) => run.2 < *best_inertia,
            };
            if is_better {
                best = Some(run);
            }
        }

        let (labels, centroids, inertia) = best.ok_or_else(|| {
            Error::InvalidOperation("KMeans produced no runs (n_init must be >= 1)".into())
        })?;

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

        // Compute distances to centroids (real Euclidean distance, matching
        // scikit-learn's `KMeans.transform`, which is unrelated to the
        // squared-distance objective used internally for fitting)
        let mut result = DataFrame::new();

        for c in 0..n_clusters {
            let mut distances = Vec::with_capacity(n_samples);

            for sample in &feature_data {
                let dist = euclidean_distance(sample, &centroids[c])?;
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

        // Compute silhouette score for `test_data` using labels PREDICTED for
        // it (not the labels recorded during training, which correspond to
        // the training set's row count/order and may not line up with
        // `test_data` at all).
        if let Some(centroids) = &self.centroids {
            let predicted_labels = self.predict(test_data)?;
            let silhouette = compute_silhouette(
                test_data,
                &predicted_labels,
                centroids,
                &self.feature_columns,
            )?;
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
        // K-means doesn't typically use cross-validation in the same way as supervised models
        Err(Error::InvalidOperation(
            "Cross-validation is not applicable for K-means clustering".into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// K-means initialization and Lloyd's-algorithm helpers
// ---------------------------------------------------------------------------

/// Uniform-random centroid initialization: `k` distinct data points chosen
/// without replacement.
fn random_init(feature_data: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let n = feature_data.len();
    if k == 0 || n == 0 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices
        .into_iter()
        .take(k)
        .map(|idx| feature_data[idx].clone())
        .collect()
}

/// k-means++ seeding (Arthur & Vassilvitskii, 2007).
///
/// Chooses the first centroid uniformly at random, then repeatedly samples
/// each subsequent centroid from the remaining points with probability
/// proportional to the squared distance to the nearest already-chosen
/// centroid. This spreads the initial centroids apart and, in expectation,
/// yields an initialization within `O(log k)` of the optimal inertia —
/// materially better starting points than uniform random selection.
fn kmeans_plusplus_init(
    feature_data: &[Vec<f64>],
    k: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f64>>> {
    let n = feature_data.len();
    if k == 0 || n == 0 {
        return Ok(Vec::new());
    }

    let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);

    let first_idx = rng.random_range(0..n);
    centroids.push(feature_data[first_idx].clone());

    // closest_sq_dist[i] = squared distance from point i to the nearest
    // centroid chosen so far.
    let mut closest_sq_dist: Vec<f64> = Vec::with_capacity(n);
    for point in feature_data {
        closest_sq_dist.push(squared_euclidean_distance(point, &centroids[0])?);
    }

    while centroids.len() < k {
        let total: f64 = closest_sq_dist.iter().sum();

        let next_idx = if total <= 0.0 {
            // Every remaining point already coincides with a chosen centroid
            // (e.g. duplicate rows, or k > number of distinct points); fall
            // back to a uniform pick so `k` centroids are still returned.
            rng.random_range(0..n)
        } else {
            let target = rng.random::<f64>() * total;
            let mut cumulative = 0.0_f64;
            let mut chosen = n - 1;
            for (idx, &d) in closest_sq_dist.iter().enumerate() {
                cumulative += d;
                if cumulative >= target {
                    chosen = idx;
                    break;
                }
            }
            chosen
        };

        let new_centroid = feature_data[next_idx].clone();
        for (idx, point) in feature_data.iter().enumerate() {
            let d = squared_euclidean_distance(point, &new_centroid)?;
            if d < closest_sq_dist[idx] {
                closest_sq_dist[idx] = d;
            }
        }
        centroids.push(new_centroid);
    }

    Ok(centroids)
}

/// Perform one complete run of Lloyd's algorithm from the given initial
/// centroids.
///
/// Returns `(labels, centroids, inertia)` where `inertia` is the sum of
/// SQUARED distances from each point to its assigned centroid — matching
/// scikit-learn's definition, and consistent with the units `tol`-based
/// convergence is checked against.
fn run_single_kmeans(
    feature_data: &[Vec<f64>],
    n_features: usize,
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    mut centroids: Vec<Vec<f64>>,
) -> Result<(Vec<usize>, Vec<Vec<f64>>, f64)> {
    let n_samples = feature_data.len();
    let mut labels = vec![0usize; n_samples];
    let mut prev_inertia = f64::MAX;
    let mut inertia = 0.0;

    for _ in 0..max_iter {
        // Assign samples to nearest centroid; inertia accumulates the
        // SQUARED distance (the actual K-means objective), not the raw
        // distance — squaring is monotonic so the assignment itself is
        // unaffected, but the reported inertia and the `tol` convergence
        // check now match scikit-learn's semantics.
        inertia = 0.0;

        for (i, sample) in feature_data.iter().enumerate() {
            let mut min_dist = f64::MAX;
            let mut min_cluster = 0;

            for (j, centroid) in centroids.iter().enumerate() {
                let dist = squared_euclidean_distance(sample, centroid)?;

                if dist < min_dist {
                    min_dist = dist;
                    min_cluster = j;
                }
            }

            labels[i] = min_cluster;
            inertia += min_dist;
        }

        // Check convergence
        if (prev_inertia - inertia).abs() < tol {
            break;
        }

        prev_inertia = inertia;

        // Update centroids
        let mut new_centroids = vec![vec![0.0; n_features]; n_clusters];
        let mut counts = vec![0usize; n_clusters];

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

        // Handle empty clusters by reinitializing them with DISTINCT points.
        //
        // Each empty cluster is reseeded with a different point, chosen in
        // decreasing order of squared distance to that point's own (current)
        // centroid — the single furthest point overall, then the next
        // furthest, and so on. Consuming a shared, pre-sorted iterator
        // guarantees no point is used twice, fixing the earlier behaviour
        // where every empty cluster received an identical copy of the same
        // "furthest point" (computed independently, and hence redundantly,
        // for each empty cluster).
        let empty_clusters: Vec<usize> = (0..n_clusters).filter(|&c| counts[c] == 0).collect();
        if !empty_clusters.is_empty() {
            let mut point_dists: Vec<(usize, f64)> = Vec::with_capacity(n_samples);
            for (j, sample) in feature_data.iter().enumerate() {
                let cluster = labels[j];
                let dist = squared_euclidean_distance(sample, &centroids[cluster])?;
                point_dists.push((j, dist));
            }
            point_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut dist_iter = point_dists.into_iter();
            for &empty_c in &empty_clusters {
                if let Some((idx, _)) = dist_iter.next() {
                    new_centroids[empty_c] = feature_data[idx].clone();
                }
                // If the iterator is exhausted (more empty clusters than
                // samples, only possible when n_clusters > n_samples), the
                // placeholder all-zero centroid is left as-is.
            }
        }

        centroids = new_centroids;
    }

    Ok((labels, centroids, inertia))
}

// ---------------------------------------------------------------------------
// Distance helpers
// ---------------------------------------------------------------------------

/// Calculate Euclidean distance between two vectors
fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::InvalidValue(format!(
            "Vectors must have the same length: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt())
}

/// Calculate squared Euclidean distance between two vectors (avoids the
/// `sqrt` round-trip; used wherever only relative ordering or a
/// sum-of-squares objective is needed, e.g. K-means assignment/inertia).
fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::InvalidValue(format!(
            "Vectors must have the same length: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum())
}

/// Calculate Manhattan distance between two vectors
fn manhattan_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::InvalidValue(format!(
            "Vectors must have the same length: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum())
}

/// Calculate Cosine distance between two vectors (1 - cosine_similarity)
fn cosine_distance(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(Error::InvalidValue(format!(
            "Vectors must have the same length: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(1.0);
    }
    let similarity = dot / (norm_a * norm_b);
    // Clamp to [-1,1] to guard against floating-point rounding
    Ok(1.0 - similarity.clamp(-1.0, 1.0))
}

/// Dispatch distance computation according to the chosen metric
fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> Result<f64> {
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
        None => data.column_names().to_vec(),
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

        // Singleton cluster — s_i = 0 by definition
        if members_ci.len() <= 1 {
            silhouette_sum += 0.0;
            count += 1;
            continue;
        }

        // a_i: mean distance to other points in same cluster
        let mut a_sum = 0.0_f64;
        for &j in members_ci {
            if j != i {
                a_sum += euclidean_distance(&feature_data[i], &feature_data[j])?;
            }
        }
        let a_i = a_sum / (members_ci.len() - 1) as f64;

        // b_i: min mean distance over all other clusters
        let mut b_i = f64::MAX;
        for (&other_cluster, other_members) in &cluster_members {
            if other_cluster == c_i || other_members.is_empty() {
                continue;
            }
            let mut sum_d = 0.0_f64;
            for &j in other_members {
                sum_d += euclidean_distance(&feature_data[i], &feature_data[j])?;
            }
            let mean_dist = sum_d / other_members.len() as f64;
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
        // Ward's criterion minimizes the increase in within-cluster variance
        // assuming squared Euclidean distances; it is not a coherent
        // objective under other metrics. scikit-learn raises for this
        // combination, and so do we.
        if self.linkage == Linkage::Ward && self.metric != DistanceMetric::Euclidean {
            return Err(Error::InvalidInput(
                "Ward linkage requires the Euclidean distance metric".into(),
            ));
        }

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
                let d = compute_distance(&feature_data[i], &feature_data[j], self.metric)?;
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
                    // `active` only ever holds indices of slots that have not
                    // yet been merged away, so `clusters[idx]` is always
                    // `Some` here — but that invariant is enforced by this
                    // loop's own bookkeeping rather than the type system, so
                    // a violation is surfaced as an `Error` (never a panic).
                    let pts_a = clusters[idx_a].as_ref().ok_or_else(|| {
                        Error::InvalidOperation(
                            "Internal error: active cluster slot was unexpectedly empty".into(),
                        )
                    })?;
                    let pts_b = clusters[idx_b].as_ref().ok_or_else(|| {
                        Error::InvalidOperation(
                            "Internal error: active cluster slot was unexpectedly empty".into(),
                        )
                    })?;

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
            let pts_b = clusters[best_b].take().ok_or_else(|| {
                Error::InvalidOperation(
                    "Internal error: active cluster slot was unexpectedly empty".into(),
                )
            })?;
            let pts_a = clusters[best_a].as_mut().ok_or_else(|| {
                Error::InvalidOperation(
                    "Internal error: active cluster slot was unexpectedly empty".into(),
                )
            })?;
            pts_a.extend(pts_b);

            // Remove best_b from the active list
            active.retain(|&idx| idx != best_b);
        }

        // Assign final integer labels (0 .. target_clusters-1)
        let mut labels = vec![0usize; n_samples];
        for (cluster_label, &slot) in active.iter().enumerate() {
            let members = clusters[slot].as_ref().ok_or_else(|| {
                Error::InvalidOperation(
                    "Internal error: active cluster slot was unexpectedly empty".into(),
                )
            })?;
            for &pt in members {
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
    /// Minimum number of points to form a core point (a point counts as its
    /// own neighbor, matching scikit-learn: a point with `min_samples - 1`
    /// OTHER points within `eps` is a core point)
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
                let d = compute_distance(&feature_data[i], &feature_data[j], self.metric)?;
                dist_matrix[i][j] = d;
                dist_matrix[j][i] = d;
            }
        }

        // For each point determine its eps-neighborhood, INCLUDING the point
        // itself (dist_matrix[i][i] == 0.0 <= eps). This matches
        // scikit-learn's DBSCAN, where `min_samples` counts the point itself:
        // a point needs `min_samples - 1` OTHER points within `eps` to be a
        // core point. Excluding self here would make `min_samples` off by
        // one relative to sklearn (this module previously excluded self).
        let mut neighborhoods: Vec<Vec<usize>> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let nbrs: Vec<usize> = (0..n_samples)
                .filter(|&j| dist_matrix[i][j] <= self.eps)
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
                    // Already assigned to a cluster (including `i` itself,
                    // which is now in its own neighborhood) — skip
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
                    let silhouette =
                        compute_silhouette_raw(&subset_data, &subset_labels, n_unique)?;
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
fn compute_silhouette_raw(
    feature_data: &[Vec<f64>],
    labels: &[usize],
    n_clusters: usize,
) -> Result<f64> {
    if n_clusters < 2 || labels.is_empty() {
        return Ok(0.0);
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

        if members_ci.len() <= 1 {
            silhouette_sum += 0.0;
            count += 1;
            continue;
        }

        let mut a_sum = 0.0_f64;
        for &j in members_ci {
            if j != i {
                a_sum += euclidean_distance(&feature_data[i], &feature_data[j])?;
            }
        }
        let a_i = a_sum / (members_ci.len() - 1) as f64;

        let mut b_i = f64::MAX;
        for k in 0..n_clusters {
            if k == c_i {
                continue;
            }
            let other = &cluster_members[k];
            if other.is_empty() {
                continue;
            }
            let mut sum_d = 0.0_f64;
            for &j in other {
                sum_d += euclidean_distance(&feature_data[i], &feature_data[j])?;
            }
            let mean_d = sum_d / other.len() as f64;
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

    Ok(if count == 0 {
        0.0
    } else {
        silhouette_sum / count as f64
    })
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

    #[test]
    fn test_dbscan_core_point_includes_self() {
        // sklearn semantics: a point is core when it has `min_samples - 1`
        // OTHER points within `eps` (i.e. `min_samples` including itself).
        // Three points within eps=1.0 of each other and min_samples=3: each
        // point's neighborhood (including itself) has size 3, so all three
        // must be core points forming a single cluster — with the old
        // "exclude self" behaviour each neighborhood would have size 2,
        // which is < min_samples, and every point would be noise.
        let xs = [0.0_f64, 0.3, 0.6];
        let ys = [0.0_f64, 0.0, 0.0];
        let df = make_df(&xs, &ys);

        let mut dbscan = DBSCAN::new(1.0, 3).with_columns(vec!["x".to_string(), "y".to_string()]);
        dbscan.fit(&df).unwrap();

        let labels = dbscan.labels.as_ref().unwrap();
        assert!(
            labels.iter().all(|&l| l == 0),
            "Expected all 3 points in a single cluster (core-point self-inclusion), got {:?}",
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

    #[test]
    fn test_agglomerative_ward_rejects_non_euclidean() {
        let xs = [0.0_f64, 1.0, 2.0];
        let ys = [0.0_f64, 1.0, 2.0];
        let df = make_df(&xs, &ys);

        let mut agg = AgglomerativeClustering::new(2)
            .with_linkage(Linkage::Ward)
            .with_metric(DistanceMetric::Manhattan)
            .with_columns(vec!["x".to_string(), "y".to_string()]);

        let result = agg.fit(&df);
        assert!(
            result.is_err(),
            "Ward linkage with a non-Euclidean metric must be rejected"
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

    #[test]
    fn test_kmeans_evaluate_on_disjoint_test_data_uses_prediction() {
        // Fit on one set of blobs, evaluate on a DIFFERENT set of points with
        // a different row count than the training data. Before the fix this
        // scored `test_data` against the (wrong-length/training) `self.labels`
        // directly; now `evaluate` must call `predict` on `test_data` first.
        let train_xs = [0.0_f64, 0.1, -0.1, 10.0, 10.1, 9.9];
        let train_ys = [0.0_f64, 0.1, -0.1, 10.0, 10.1, 9.9];
        let train_df = make_df(&train_xs, &train_ys);

        let mut km = KMeans::new(2)
            .random_seed(7)
            .with_columns(vec!["x".to_string(), "y".to_string()]);
        km.fit(&train_df).unwrap();

        // Test set has a DIFFERENT number of rows than the training set.
        let test_xs = [0.05_f64, 9.95, -0.05, 10.05];
        let test_ys = [0.05_f64, 9.95, -0.05, 10.05];
        let test_df = make_df(&test_xs, &test_ys);

        // Must not error out due to a labels/row-count mismatch, and must
        // produce a well-separated silhouette score since the test points
        // are themselves two clean, well-separated blobs.
        let metrics = km.evaluate(&test_df, "").unwrap();
        let score = metrics
            .get_metric("silhouette_score")
            .copied()
            .unwrap_or(0.0);
        assert!(
            score > 0.5,
            "Expected a well-separated silhouette on freshly predicted labels, got {}",
            score
        );
    }

    #[test]
    fn test_kmeans_empty_cluster_reseed_is_distinct() {
        // Directly exercise `run_single_kmeans` (the private per-run Lloyd's
        // algorithm helper) with HAND-CRAFTED starting centroids that
        // deterministically force multiple empty clusters after a single
        // assignment pass — no RNG or convergence luck involved.
        //
        // Three of the four starting centroids are placed at the exact same
        // location (0, 0). Nearest-centroid assignment breaks ties with a
        // strict `<` comparison, so the FIRST matching index always wins:
        // every point nearest that shared location is captured by centroid
        // 0 alone, leaving centroids 1 and 2 with zero points (empty), while
        // centroid 3 — far away at (10, 10) — captures its own local group.
        //
        // Under the old bug, both empty clusters were reseeded independently
        // by searching for "the single globally-furthest point", so they
        // would end up as EXACT DUPLICATES of each other ((-25, -25) in both
        // slots). The fix must give them DISTINCT points instead.
        let feature_data: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![-0.1, -0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![25.0, 25.0],
            vec![-25.0, -25.0],
        ];
        let initial_centroids = vec![
            vec![0.0, 0.0],   // captures the near-origin group (wins ties as index 0)
            vec![0.0, 0.0],   // duplicate starting position -> ends up EMPTY
            vec![0.0, 0.0],   // duplicate starting position -> ends up EMPTY
            vec![10.0, 10.0], // captures the (10,10) group
        ];

        let (labels, centroids, _inertia) =
            run_single_kmeans(&feature_data, 2, 4, 1, 1e-4, initial_centroids).unwrap();

        // Sanity-check the hand-traced assignment: clusters 1 and 2 must
        // indeed have received zero points (i.e. the fixture actually
        // exercises the empty-cluster path this test is about).
        for empty_cluster in [1usize, 2usize] {
            assert!(
                !labels.contains(&empty_cluster),
                "expected cluster {} to be empty after the first assignment pass \
                 (fixture assumption violated); got labels {:?}",
                empty_cluster,
                labels
            );
        }

        assert_eq!(centroids.len(), 4);
        // Collect centroids as bit-pattern-comparable tuples to check for
        // exact duplicates (reseeded-from-the-same-point centroids would be
        // bit-identical).
        let mut seen: Vec<(u64, u64)> = Vec::new();
        for c in &centroids {
            let key = (c[0].to_bits(), c[1].to_bits());
            assert!(
                !seen.contains(&key),
                "Found duplicate centroid {:?} among {:?} — empty-cluster reseed must \
                 assign DISTINCT points to different empty clusters, not the same point \
                 to all of them",
                c,
                centroids
            );
            seen.push(key);
        }
    }
}
