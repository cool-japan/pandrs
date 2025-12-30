//! Clustering module
//!
//! Provides clustering algorithms for grouping data points into clusters.

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::error::{Result, Error};
use crate::ml::pipeline::Transformer;
use crate::column::{Float64Column, Column, ColumnTrait};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::utils::rand_compat::GenRangeCompat;
use std::collections::{HashMap, HashSet};

/// K-Means clustering algorithm
pub struct KMeans {
    /// Number of clusters
    k: usize,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence threshold
    tol: f64,
    /// Random seed
    random_seed: Option<u64>,
    /// Cluster centroids
    centroids: Vec<Vec<f64>>,
    /// Labels for each data point
    labels: Vec<usize>,
    /// Feature names
    feature_names: Vec<String>,
    /// Inertia (sum of squared distances within clusters)
    inertia: f64,
    /// Number of iterations until convergence
    n_iter: usize,
    /// Whether the model has been fitted
    fitted: bool,
}

impl KMeans {
    /// Create a new KMeans instance
    pub fn new(k: usize, max_iter: usize, tol: f64, random_seed: Option<u64>) -> Self {
        KMeans {
            k,
            max_iter,
            tol,
            random_seed,
            centroids: Vec::new(),
            labels: Vec::new(),
            feature_names: Vec::new(),
            inertia: 0.0,
            n_iter: 0,
            fitted: false,
        }
    }
    
    /// Get cluster labels
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }
    
    /// Get cluster centroids
    pub fn centroids(&self) -> &[Vec<f64>] {
        &self.centroids
    }
    
    /// Get inertia
    pub fn inertia(&self) -> f64 {
        self.inertia
    }
    
    /// Get the number of iterations until convergence
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
    
    /// Compute squared Euclidean distance
    fn squared_euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi).powi(2))
            .sum()
    }
    
    /// Initialize cluster centroids using k-means++
    fn kmeans_plus_plus_init(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let n_features = data[0].len();
        
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random()),
        };
        
        // Select the first centroid randomly
        let first_idx = rng.gen_range(0..n_samples);
        let mut centroids = vec![data[first_idx].clone()];
        
        // Select the remaining k-1 centroids
        for _ in 1..self.k {
            // Compute squared distances to the nearest centroid for each data point
            let mut distances = vec![0.0; n_samples];
            let mut sum_distances = 0.0;
            
            for (i, point) in data.iter().enumerate() {
                // Squared distance to the nearest centroid
                let closest_dist = centroids
                    .iter()
                    .map(|c| Self::squared_euclidean_distance(point, c))
                    .fold(f64::INFINITY, |a, b| a.min(b));
                
                distances[i] = closest_dist;
                sum_distances += closest_dist;
            }
            
            // Select the next centroid with probability proportional to squared distance
            let mut cumsum = 0.0;
            let threshold = rng.gen_range(0.0..sum_distances);
            
            for (i, &dist) in distances.iter().enumerate() {
                cumsum += dist;
                if cumsum >= threshold {
                    centroids.push(data[i].clone());
                    break;
                }
            }
        }
        
        centroids
    }
    
    /// Helper method to extract numeric values from column
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let mut values = Vec::with_capacity(col.len());
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // Treat NA as 0 (or implement an appropriate strategy)
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let mut values = Vec::with_capacity(col.len());
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
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

impl Transformer for KMeans {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Extract numeric columns only
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
                "DataFrame must contain at least one numeric column for KMeans".to_string()
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
        
        // Initialize cluster centroids using k-means++
        self.centroids = self.kmeans_plus_plus_init(&data);
        
        // Main loop of k-means algorithm
        let mut prev_inertia = f64::INFINITY;
        self.labels = vec![0; n_samples];
        
        for iter in 0..self.max_iter {
            // Assign each data point to the nearest cluster
            let mut new_labels = vec![0; n_samples];
            let mut inertia = 0.0;
            
            for (i, point) in data.iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                let mut closest_centroid = 0;
                
                for (j, centroid) in self.centroids.iter().enumerate() {
                    let dist = Self::squared_euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        closest_centroid = j;
                    }
                }
                
                new_labels[i] = closest_centroid;
                inertia += min_dist;
            }
            
            self.labels = new_labels;
            self.inertia = inertia;
            
            // Update cluster centroids
            let mut new_centroids = vec![vec![0.0; n_features]; self.k];
            let mut counts = vec![0; self.k];
            
            for (i, point) in data.iter().enumerate() {
                let cluster = self.labels[i];
                counts[cluster] += 1;
                
                for j in 0..n_features {
                    new_centroids[cluster][j] += point[j];
                }
            }
            
            for i in 0..self.k {
                if counts[i] > 0 {
                    for j in 0..n_features {
                        new_centroids[i][j] /= counts[i] as f64;
                    }
                }
            }
            
            // Convergence check
            let mut centroid_shift = 0.0;
            for (old, new) in self.centroids.iter().zip(new_centroids.iter()) {
                centroid_shift += Self::squared_euclidean_distance(old, new);
            }
            
            self.centroids = new_centroids;
            
            // If the change in inertia is below the threshold, convergence is achieved
            let inertia_change = (prev_inertia - self.inertia).abs();
            if inertia_change / prev_inertia < self.tol {
                self.n_iter = iter + 1;
                break;
            }
            
            if centroid_shift < self.tol {
                self.n_iter = iter + 1;
                break;
            }
            
            prev_inertia = self.inertia;
            
            // If this is the last iteration
            if iter == self.max_iter - 1 {
                self.n_iter = self.max_iter;
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "KMeans has not been fitted yet".to_string()
            ));
        }
        
        // Copy the original DataFrame
        let mut result = df.clone();
        
        // Assign each data point to the nearest cluster
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
        
        // Assign each data point to the nearest cluster
        let mut labels = Vec::with_capacity(n_samples);
        let mut distances = Vec::with_capacity(n_samples);
        
        for point in &data {
            let mut min_dist = f64::INFINITY;
            let mut closest_centroid = 0;
            
            for (j, centroid) in self.centroids.iter().enumerate() {
                let dist = Self::squared_euclidean_distance(point, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    closest_centroid = j;
                }
            }
            
            labels.push(closest_centroid as i64);
            distances.push(min_dist.sqrt()); // Euclidean distance
        }
        
        // Add cluster labels and distances as new columns
        let labels_column = Column::Int64(crate::column::Int64Column::with_name(labels, "cluster".to_string()));
        let distances_column = Column::Float64(Float64Column::with_name(distances, "distance_to_centroid".to_string()));
        
        result.add_column("cluster".to_string(), labels_column)?;
        result.add_column("distance_to_centroid".to_string(), distances_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// Hierarchical clustering algorithm
pub struct AgglomerativeClustering {
    /// Number of clusters
    n_clusters: usize,
    /// Linkage method
    linkage: Linkage,
    /// Distance metric
    metric: DistanceMetric,
    /// Cluster labels
    labels: Vec<usize>,
    /// Feature names
    feature_names: Vec<String>,
    /// Whether the model has been fitted
    fitted: bool,
}

/// Linkage method
pub enum Linkage {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage
    Average,
    /// Ward linkage
    Ward,
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

impl AgglomerativeClustering {
    /// Create a new AgglomerativeClustering instance
    pub fn new(n_clusters: usize, linkage: Linkage, metric: DistanceMetric) -> Self {
        AgglomerativeClustering {
            n_clusters,
            linkage,
            metric,
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// Get cluster labels
    pub fn labels(&self) -> &[usize] {
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
    
    /// Compute distance between two clusters
    fn compute_cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        data: &[Vec<f64>],
        distances: &HashMap<(usize, usize), f64>,
    ) -> f64 {
        match self.linkage {
            Linkage::Single => {
                // Single linkage: minimum distance
                let mut min_dist = f64::INFINITY;
                
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = if i < j {
                            *distances.get(&(i, j)).unwrap_or(&f64::INFINITY)
                        } else {
                            *distances.get(&(j, i)).unwrap_or(&f64::INFINITY)
                        };
                        
                        min_dist = min_dist.min(dist);
                    }
                }
                
                min_dist
            }
            Linkage::Complete => {
                // Complete linkage: maximum distance
                let mut max_dist: f64 = 0.0;
                
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = if i < j {
                            *distances.get(&(i, j)).unwrap_or(&0.0)
                        } else {
                            *distances.get(&(j, i)).unwrap_or(&0.0)
                        };
                        
                        max_dist = max_dist.max(dist);
                    }
                }
                
                max_dist
            }
            Linkage::Average => {
                // Average linkage: average distance
                let mut sum_dist = 0.0;
                let mut count = 0;
                
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = if i < j {
                            *distances.get(&(i, j)).unwrap_or(&0.0)
                        } else {
                            *distances.get(&(j, i)).unwrap_or(&0.0)
                        };
                        
                        sum_dist += dist;
                        count += 1;
                    }
                }
                
                if count > 0 {
                    sum_dist / count as f64
                } else {
                    f64::INFINITY
                }
            }
            Linkage::Ward => {
                // Ward linkage: increase in variance
                let n1 = cluster1.len();
                let n2 = cluster2.len();
                
                if n1 == 0 || n2 == 0 {
                    return f64::INFINITY;
                }
                
                // Centroid of cluster 1
                let mut centroid1 = vec![0.0; data[0].len()];
                for &i in cluster1 {
                    for j in 0..data[0].len() {
                        centroid1[j] += data[i][j];
                    }
                }
                for j in 0..centroid1.len() {
                    centroid1[j] /= n1 as f64;
                }
                
                // Centroid of cluster 2
                let mut centroid2 = vec![0.0; data[0].len()];
                for &i in cluster2 {
                    for j in 0..data[0].len() {
                        centroid2[j] += data[i][j];
                    }
                }
                for j in 0..centroid2.len() {
                    centroid2[j] /= n2 as f64;
                }
                
                // Distance between centroids
                let mut dist = 0.0;
                for j in 0..centroid1.len() {
                    dist += (centroid1[j] - centroid2[j]).powi(2);
                }
                dist = dist.sqrt();
                
                // Ward linkage distance
                (n1 * n2) as f64 * dist / (n1 + n2) as f64
            }
        }
    }
    
    /// Helper method to extract numeric values from column
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let mut values = Vec::with_capacity(col.len());
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // Treat NA as 0 (or implement an appropriate strategy)
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let mut values = Vec::with_capacity(col.len());
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
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

impl Transformer for AgglomerativeClustering {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Extract numeric columns only
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
                "DataFrame must contain at least one numeric column for AgglomerativeClustering".to_string()
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
        
        // Compute distances between all pairs
        let mut distances = HashMap::new();
        for i in 0..n_samples {
            for j in i+1..n_samples {
                let dist = self.compute_distance(&data[i], &data[j]);
                distances.insert((i, j), dist);
            }
        }
        
        // Initialize each data point as its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        
        // Repeat until the number of clusters reaches the target
        while clusters.len() > self.n_clusters {
            // Find the two closest clusters
            let mut min_dist = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 0;
            
            for i in 0..clusters.len() {
                for j in i+1..clusters.len() {
                    let dist = self.compute_cluster_distance(&clusters[i], &clusters[j], &data, &distances);
                    
                    if dist < min_dist {
                        min_dist = dist;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }
            
            // Merge clusters
            let cluster_j = clusters.remove(merge_j); // Remove from higher index
            let cluster_i = &mut clusters[merge_i];
            cluster_i.extend(cluster_j);
        }
        
        // Assign final cluster labels
        self.labels = vec![0; n_samples];
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            for &sample_idx in cluster {
                self.labels[sample_idx] = cluster_idx;
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "AgglomerativeClustering has not been fitted yet".to_string()
            ));
        }
        
        // Check if the data size matches
        if df.row_count() != self.labels.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // Copy the original DataFrame
        let mut result = df.clone();
        
        // Add cluster labels as a new column
        let labels: Vec<i64> = self.labels.iter().map(|&l| l as i64).collect();
        let labels_column = Column::Int64(crate::column::Int64Column::with_name(labels, "cluster".to_string()));
        
        result.add_column("cluster".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm
pub struct DBSCAN {
    /// Epsilon (neighborhood radius)
    eps: f64,
    /// Minimum number of points
    min_samples: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// Cluster labels
    labels: Vec<i64>,
    /// Feature names
    feature_names: Vec<String>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl DBSCAN {
    /// Create a new DBSCAN instance
    pub fn new(eps: f64, min_samples: usize, metric: DistanceMetric) -> Self {
        DBSCAN {
            eps,
            min_samples,
            metric,
            labels: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// Get cluster labels
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
    
    /// Find the neighbors of a given point
    fn region_query(&self, point_idx: usize, data: &[Vec<f64>]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        
        for (i, point) in data.iter().enumerate() {
            if i != point_idx && self.compute_distance(&data[point_idx], point) <= self.eps {
                neighbors.push(i);
            }
        }
        
        neighbors
    }
    
    /// Expand the cluster from a given point
    fn expand_cluster(
        &self,
        point_idx: usize,
        neighbors: &[usize],
        cluster_id: i64,
        labels: &mut [i64],
        data: &[Vec<f64>],
        visited: &mut HashSet<usize>,
    ) {
        labels[point_idx] = cluster_id;
        
        let mut i = 0;
        let mut neighbors_vec = neighbors.to_vec();
        
        while i < neighbors_vec.len() {
            let current_point = neighbors_vec[i];
            
            // Process unvisited points
            if !visited.contains(&current_point) {
                visited.insert(current_point);
                
                let current_neighbors = self.region_query(current_point, data);
                
                if current_neighbors.len() >= self.min_samples {
                    // Add density-reachable points
                    for &neighbor in &current_neighbors {
                        if !neighbors_vec.contains(&neighbor) {
                            neighbors_vec.push(neighbor);
                        }
                    }
                }
            }
            
            // If the label has not been assigned yet, add to this cluster
            if labels[current_point] == -1 {
                labels[current_point] = cluster_id;
            }
            
            i += 1;
        }
    }
    
    /// Helper method to extract numeric values from column
    fn extract_numeric_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        match col.column_type() {
            crate::column::ColumnType::Float64 => {
                let mut values = Vec::with_capacity(col.len());
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(value);
                    } else {
                        values.push(0.0); // Treat NA as 0 (or implement an appropriate strategy)
                    }
                }
                Ok(values)
            },
            crate::column::ColumnType::Int64 => {
                let mut values = Vec::with_capacity(col.len());
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
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

impl Transformer for DBSCAN {
    fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()> {
        // Extract numeric columns only
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
                "DataFrame must contain at least one numeric column for DBSCAN".to_string()
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
        
        // Initialize labels (unassigned: -1)
        self.labels = vec![-1; n_samples];
        let mut visited = HashSet::new();
        let mut cluster_id = 0;
        
        // Process each point
        for i in 0..n_samples {
            if !visited.contains(&i) {
                visited.insert(i);
                
                // Find neighbors
                let neighbors = self.region_query(i, &data);
                
                if neighbors.len() < self.min_samples {
                    // Treat as noise
                    self.labels[i] = -1;
                } else {
                    // Start a new cluster
                    // Expand the cluster (manually implemented to avoid self-referencing issues)
                    self.labels[i] = cluster_id;
                    
                    // Copy the neighbor list for processing
                    let mut process_queue = neighbors.clone();
                    let mut index = 0;
                    
                    // Process all points in the queue
                    while index < process_queue.len() {
                        let current_point = process_queue[index];
                        index += 1;
                        
                        // Process unvisited points
                        if !visited.contains(&current_point) {
                            visited.insert(current_point);
                            
                            // Get neighbors of the current point
                            let current_neighbors: Vec<usize> = data.iter().enumerate()
                                .filter(|(idx, point)| {
                                    *idx != current_point && 
                                    self.compute_distance(&data[current_point], point) <= self.eps
                                })
                                .map(|(idx, _)| idx)
                                .collect::<Vec<_>>();
                            
                            // If the point is a core point, process its edges as well
                            if current_neighbors.len() >= self.min_samples {
                                for &neighbor in &current_neighbors {
                                    if !process_queue.contains(&neighbor) {
                                        process_queue.push(neighbor);
                                    }
                                }
                            }
                        }
                        
                        // Assign label
                        if self.labels[current_point] == -1 {
                            self.labels[current_point] = cluster_id;
                        }
                    }
                    cluster_id += 1;
                }
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "DBSCAN has not been fitted yet".to_string()
            ));
        }
        
        // Check if the data size matches
        if df.row_count() != self.labels.len() {
            return Err(Error::InvalidOperation(
                "Number of samples in the input DataFrame does not match the number of samples used during fitting".to_string()
            ));
        }
        
        // Copy the original DataFrame
        let mut result = df.clone();
        
        // Add cluster labels as a new column
        let int_column = crate::column::Int64Column::with_name(self.labels.clone(), "cluster".to_string());
        let labels_column = Column::Int64(int_column);
        
        result.add_column("cluster".to_string(), labels_column)?;
        
        Ok(result)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}