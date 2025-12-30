//! Dimension Reduction Module
//!
//! Provides dimension reduction algorithms for visualization and analysis of high-dimensional data.

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::column::{Float64Column, Int64Column, Column};
use crate::column::ColumnTrait;
use crate::error::{Result, Error};
use crate::ml::pipeline::Transformer;
use std::collections::HashMap;
use rand::Rng;
use rand::SeedableRng;
use crate::utils::rand_compat::GenRangeCompat;

/// Principal Component Analysis (PCA) implementation
#[derive(Debug)]
pub struct PCA {
    /// Number of dimensions after reduction
    n_components: usize,
    /// Explained variance ratio of each principal component
    explained_variance_ratio: Vec<f64>,
    /// Cumulative explained variance
    cumulative_explained_variance: Vec<f64>,
    /// Eigenvectors of principal components
    components: Vec<Vec<f64>>,
    /// Mean of each feature
    mean: Vec<f64>,
    /// Standard deviation of each feature
    std: Vec<f64>,
    /// Feature names
    feature_names: Vec<String>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl PCA {
    /// Create a new PCA instance
    pub fn new(n_components: usize) -> Self {
        PCA {
            n_components,
            explained_variance_ratio: Vec::new(),
            cumulative_explained_variance: Vec::new(),
            components: Vec::new(),
            mean: Vec::new(),
            std: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> &[f64] {
        &self.explained_variance_ratio
    }
    
    /// Get cumulative explained variance
    pub fn cumulative_explained_variance(&self) -> &[f64] {
        &self.cumulative_explained_variance
    }
    
    /// Get principal component eigenvectors
    pub fn components(&self) -> &[Vec<f64>] {
        &self.components
    }
    
    /// Calculate covariance matrix from data matrix
    fn compute_covariance_matrix(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let n_features = data[0].len();
        
        // Calculate mean of each feature
        let mut mean = vec![0.0; n_features];
        for sample in data {
            for (j, &val) in sample.iter().enumerate() {
                mean[j] += val;
            }
        }
        
        for j in 0..n_features {
            mean[j] /= n_samples as f64;
        }
        
        // Create centered data
        let centered_data: Vec<Vec<f64>> = data
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .enumerate()
                    .map(|(j, &val)| val - mean[j])
                    .collect()
            })
            .collect();
        
        // Calculate covariance matrix
        let mut cov = vec![vec![0.0; n_features]; n_features];
        
        for i in 0..n_features {
            for j in 0..n_features {
                let mut sum = 0.0;
                for sample in &centered_data {
                    sum += sample[i] * sample[j];
                }
                cov[i][j] = sum / (n_samples as f64 - 1.0);
            }
        }
        
        cov
    }
    
    /// Calculate the largest eigenvalue and corresponding eigenvector using power iteration method
    fn power_iteration(matrix: &[Vec<f64>], tol: f64, max_iter: usize) -> (f64, Vec<f64>) {
        let n = matrix.len();
        
        // Random initial vector (unit vector)
        let mut vec = vec![1.0 / (n as f64).sqrt(); n];
        
        // Power iteration
        for _ in 0..max_iter {
            // Matrix-vector multiplication
            let mut new_vec = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_vec[i] += matrix[i][j] * vec[j];
                }
            }
            
            // Calculate norm
            let norm: f64 = new_vec.iter().map(|&x| x * x).sum::<f64>().sqrt();
            
            // Convergence check
            let mut converged = true;
            for i in 0..n {
                let v = new_vec[i] / norm;
                if (v - vec[i]).abs() > tol {
                    converged = false;
                }
                vec[i] = v;
            }
            
            if converged {
                break;
            }
        }
        
        // Calculate eigenvalue (Rayleigh quotient)
        let mut eigenvalue = 0.0;
        let mut denom = 0.0;
        
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += matrix[i][j] * vec[j];
            }
            eigenvalue += vec[i] * sum;
            denom += vec[i] * vec[i];
        }
        
        eigenvalue /= denom;
        
        (eigenvalue, vec)
    }
    
    /// Deflation process: remove eigenvector contribution from matrix
    fn deflate(matrix: &mut [Vec<f64>], eigenvalue: f64, eigenvector: &[f64]) {
        let n = matrix.len();
        
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
            }
        }
    }
    
    // Calculate standard deviation (from column)
    fn compute_std(values: &[f64], mean: f64) -> f64 {
        if values.is_empty() {
            return 1.0;  // Default value
        }
        
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() as f64);
        
        variance.sqrt()
    }
}

impl Transformer for PCA {
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
            .map(|s| s.to_string())  // Fixed: clone &String to get ownership
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for PCA".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        let n_features = self.feature_names.len();
        
        // Adjust n_components to not exceed the number of features
        let n_components = self.n_components.min(n_features);
        self.n_components = n_components;
        
        // Prepare data
        let n_samples = df.row_count();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        self.mean = vec![0.0; n_features];
        self.std = vec![1.0; n_features];
        
        // Load data and standardize
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let col_view = df.column(col_name)?;
            
            if let Some(float_col) = col_view.as_float64() {
                // Load data
                let mut values = Vec::with_capacity(n_samples);
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = float_col.get(row_idx) {
                        values.push(value);
                        data[row_idx][col_idx] = value;
                    }
                }
                
                // Calculate mean and standard deviation
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = Self::compute_std(&values, mean);
                
                self.mean[col_idx] = mean;
                self.std[col_idx] = std;
                
                // Standardize data
                for row_idx in 0..n_samples {
                    if std > 0.0 {
                        data[row_idx][col_idx] = (data[row_idx][col_idx] - mean) / std;
                    }
                }
            } else if let Some(int_col) = col_view.as_int64() {
                // Load data
                let mut values = Vec::with_capacity(n_samples);
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = int_col.get(row_idx) {
                        values.push(value as f64);
                        data[row_idx][col_idx] = value as f64;
                    }
                }
                
                // Calculate mean and standard deviation
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std = Self::compute_std(&values, mean);
                
                self.mean[col_idx] = mean;
                self.std[col_idx] = std;
                
                // Standardize data
                for row_idx in 0..n_samples {
                    if std > 0.0 {
                        data[row_idx][col_idx] = (data[row_idx][col_idx] - mean) / std;
                    }
                }
            }
        }
        
        // Calculate covariance matrix
        let mut cov_matrix = Self::compute_covariance_matrix(&data);
        
        // Eigenvalue decomposition
        let mut eigenvalues = Vec::with_capacity(n_components);
        self.components = Vec::with_capacity(n_components);
        
        // Calculate top n_components eigenvalues and eigenvectors using power iteration
        for _ in 0..n_components {
            let (eigenvalue, eigenvector) = Self::power_iteration(&cov_matrix, 1e-10, 100);
            eigenvalues.push(eigenvalue);
            self.components.push(eigenvector.clone());
            
            // Deflation
            Self::deflate(&mut cov_matrix, eigenvalue, &eigenvector);
        }
        
        // Total eigenvalues (total variance)
        let total_variance: f64 = eigenvalues.iter().sum();
        
        // Calculate explained variance ratio
        self.explained_variance_ratio = eigenvalues
            .iter()
            .map(|&val| val / total_variance)
            .collect();
        
        // Calculate cumulative explained variance
        self.cumulative_explained_variance = Vec::with_capacity(n_components);
        let mut cum_sum = 0.0;
        for &ratio in &self.explained_variance_ratio {
            cum_sum += ratio;
            self.cumulative_explained_variance.push(cum_sum);
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "PCA has not been fitted yet".to_string()
            ));
        }
        
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        
        // Matrix to store standardized data
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // Load data and standardize
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let col_view = df.column(col_name).map_err(|_| {
                Error::InvalidOperation(
                    format!("Column '{}' not found in DataFrame", col_name)
                )
            })?;
            
            if let Some(float_col) = col_view.as_float64() {
                // Load and standardize data
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = float_col.get(row_idx) {
                        if self.std[col_idx] > 0.0 {
                            data[row_idx][col_idx] = (value - self.mean[col_idx]) / self.std[col_idx];
                        }
                    }
                }
            } else if let Some(int_col) = col_view.as_int64() {
                // Load and standardize data
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = int_col.get(row_idx) {
                        if self.std[col_idx] > 0.0 {
                            data[row_idx][col_idx] = ((value as f64) - self.mean[col_idx]) / self.std[col_idx];
                        }
                    }
                }
            }
        }
        
        // Transform to principal components
        let mut transformed_data = vec![vec![0.0; self.n_components]; n_samples];
        
        for i in 0..n_samples {
            for j in 0..self.n_components {
                let mut pc_value = 0.0;
                for k in 0..n_features {
                    pc_value += data[i][k] * self.components[j][k];
                }
                transformed_data[i][j] = pc_value;
            }
        }
        
        // Convert to OptimizedDataFrame
        let mut result_df = OptimizedDataFrame::new();
        
        // Add principal component columns
        for j in 0..self.n_components {
            let mut pc_values = Vec::with_capacity(n_samples);
            
            for i in 0..n_samples {
                pc_values.push(transformed_data[i][j]);
            }
            
            let pc_col = Float64Column::new(pc_values);
            result_df.add_column(format!("PC{}", j + 1), Column::Float64(pc_col))?;
        }
        
        // Add non-numeric columns if any
        for col_name in df.column_names() {
            if !self.feature_names.contains(&col_name) {
                if let Ok(col_view) = df.column(&col_name) {
                    if let Some(str_col) = col_view.as_string() {
                        // String column
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = str_col.get(i) {
                                values.push(value.to_string());
                            } else {
                                values.push("".to_string());
                            }
                        }
                        let string_col = Column::String(crate::column::StringColumn::new(values));
                        result_df.add_column(col_name.clone(), string_col)?;
                    } else if let Some(bool_col) = col_view.as_boolean() {
                        // Boolean column
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = bool_col.get(i) {
                                values.push(value);
                            } else {
                                values.push(false);
                            }
                        }
                        let bool_col = Column::Boolean(crate::column::BooleanColumn::new(values));
                        result_df.add_column(col_name.clone(), bool_col)?;
                    }
                }
            }
        }
        
        Ok(result_df)
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// t-SNE (t-distributed Stochastic Neighbor Embedding) implementation
#[derive(Debug)]
pub struct TSNE {
    /// Number of dimensions after reduction (typically 2 or 3)
    n_components: usize,
    /// Learning rate
    learning_rate: f64,
    /// Perplexity (parameter controlling the neighborhood size)
    perplexity: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Initialization method
    init: TSNEInit,
    /// Coordinates after embedding
    embedding: Vec<Vec<f64>>,
    /// Feature names
    feature_names: Vec<String>,
    /// Whether the model has been fitted
    fitted: bool,
}

/// t-SNE initialization method
#[derive(Debug)]
pub enum TSNEInit {
    /// Random initialization
    Random,
    /// PCA initialization
    PCA,
}

impl TSNE {
    /// Create a new TSNE instance
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        max_iter: usize,
        init: TSNEInit,
    ) -> Self {
        TSNE {
            n_components,
            learning_rate,
            perplexity,
            max_iter,
            init,
            embedding: Vec::new(),
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// Get embedding
    pub fn embedding(&self) -> &[Vec<f64>] {
        &self.embedding
    }
    
    /// Calculate squared Euclidean distance
    fn squared_euclidean_distance(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - yi).powi(2))
            .sum()
    }
    
    /// Calculate pairwise affinities matrix P (similarity in high-dimensional space)
    fn compute_pairwise_affinities(data: &[Vec<f64>], perplexity: f64) -> Vec<Vec<f64>> {
        let n_samples = data.len();
        let mut p = vec![vec![0.0; n_samples]; n_samples];
        
        // Calculate conditional probabilities
        for i in 0..n_samples {
            // Binary search to find sigma
            let mut beta = 1.0; // Beta = 1 / (2 * sigma^2)
            let mut beta_min = 0.0;
            let mut beta_max = f64::INFINITY;
            
            let target_entropy = perplexity.ln();
            let tol = 1e-5;
            let max_iter = 50;
            
            for _ in 0..max_iter {
                // Calculate conditional probabilities
                let mut sum_pi = 0.0;
                for j in 0..n_samples {
                    if i != j {
                        let dist = Self::squared_euclidean_distance(&data[i], &data[j]);
                        p[i][j] = (-beta * dist).exp();
                        sum_pi += p[i][j];
                    }
                }
                
                // Normalize probability distribution
                for j in 0..n_samples {
                    if i != j && sum_pi > 0.0 {
                        p[i][j] /= sum_pi;
                    }
                }
                
                // Calculate entropy
                let mut entropy = 0.0;
                for j in 0..n_samples {
                    if i != j && p[i][j] > 1e-7 {
                        entropy -= p[i][j] * p[i][j].ln();
                    }
                }
                
                // Difference from target perplexity
                let entropy_diff = entropy - target_entropy;
                
                if entropy_diff.abs() < tol {
                    break;
                }
                
                // Update beta
                if entropy_diff > 0.0 {
                    beta_min = beta;
                    if beta_max == f64::INFINITY {
                        beta *= 2.0;
                    } else {
                        beta = (beta + beta_max) / 2.0;
                    }
                } else {
                    beta_max = beta;
                    beta = (beta + beta_min) / 2.0;
                }
            }
        }
        
        // Symmetrize and scale
        let mut symmetric_p = vec![vec![0.0; n_samples]; n_samples];
        for i in 0..n_samples {
            for j in 0..n_samples {
                symmetric_p[i][j] = (p[i][j] + p[j][i]) / (2.0 * n_samples as f64);
            }
        }
        
        symmetric_p
    }
    
    /// Calculate Q matrix (similarity in low-dimensional space using t-distribution)
    fn compute_q_matrix(embedding: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = embedding.len();
        let mut q = vec![vec![0.0; n_samples]; n_samples];
        let mut sum_q = 0.0;
        
        for i in 0..n_samples {
            for j in 0..i {
                let dist = Self::squared_euclidean_distance(&embedding[i], &embedding[j]);
                let q_ij = 1.0 / (1.0 + dist);
                q[i][j] = q_ij;
                q[j][i] = q_ij;
                sum_q += 2.0 * q_ij;
            }
        }
        
        // Normalize
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    q[i][j] /= sum_q;
                }
            }
        }
        
        q
    }
    
    /// Calculate gradient
    fn compute_gradient(p: &[Vec<f64>], q: &[Vec<f64>], embedding: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n_samples = embedding.len();
        let n_components = embedding[0].len();
        let mut grad = vec![vec![0.0; n_components]; n_samples];
        
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    // (p_ij - q_ij) * (1 + ||y_i - y_j||^2)^-1
                    let factor = (p[i][j] - q[i][j]) * (1.0 + Self::squared_euclidean_distance(&embedding[i], &embedding[j])).powi(-1);
                    
                    for k in 0..n_components {
                        grad[i][k] += 4.0 * factor * (embedding[i][k] - embedding[j][k]);
                    }
                }
            }
        }
        
        grad
    }
}

impl Transformer for TSNE {
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
            .map(|s| s.to_string())  // Fixed: clone &String to get ownership
            .collect();
        
        if numeric_columns.is_empty() {
            return Err(Error::InvalidOperation(
                "DataFrame must contain at least one numeric column for t-SNE".to_string()
            ));
        }
        
        self.feature_names = numeric_columns.clone();
        
        // Prepare data
        let n_samples = df.row_count();
        let n_features = self.feature_names.len();
        let mut data = vec![vec![0.0; n_features]; n_samples];
        
        // Load data
        for (col_idx, col_name) in self.feature_names.iter().enumerate() {
            let col_view = df.column(col_name)?;
            
            if let Some(float_col) = col_view.as_float64() {
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = float_col.get(row_idx) {
                        data[row_idx][col_idx] = value;
                    }
                }
            } else if let Some(int_col) = col_view.as_int64() {
                for row_idx in 0..n_samples {
                    if let Ok(Some(value)) = int_col.get(row_idx) {
                        data[row_idx][col_idx] = value as f64;
                    }
                }
            }
        }
        
        // Calculate conditional probability matrix
        let p = Self::compute_pairwise_affinities(&data, self.perplexity);
        
        // Generate initial embedding
        self.embedding = match self.init {
            TSNEInit::Random => {
                // Random initialization
                let mut rng = rand::rngs::StdRng::seed_from_u64(rand::random());
                (0..n_samples)
                    .map(|_| {
                        (0..self.n_components)
                            .map(|_| 1e-4 * rng.gen_range(-1.0..1.0))
                            .collect()
                    })
                    .collect()
            }
            TSNEInit::PCA => {
                // PCA initialization
                let mut pca = PCA::new(self.n_components);
                let pca_result = pca.fit_transform(df)?;
                
                let mut embedding = vec![vec![0.0; self.n_components]; n_samples];
                for i in 0..n_samples {
                    for j in 0..self.n_components {
                        let pc_col = format!("PC{}", j + 1);
                        let col_view = pca_result.column(&pc_col)?;
                        
                        if let Some(float_col) = col_view.as_float64() {
                            if let Ok(Some(value)) = float_col.get(i) {
                                embedding[i][j] = value * 1e-4;
                            }
                        }
                    }
                }
                embedding
            }
        };
        
        // Gradient descent optimization for t-SNE
        let mut gains = vec![vec![1.0; self.n_components]; n_samples];
        let mut velocities = vec![vec![0.0; self.n_components]; n_samples];
        let mut momentum = 0.5;
        
        for iter in 0..self.max_iter {
            // Calculate similarity Q in the low-dimensional space using t-distribution
            let q = Self::compute_q_matrix(&self.embedding);
            
            // Calculate gradient
            let grad = Self::compute_gradient(&p, &q, &self.embedding);
            
            // Update
            if iter == 20 {
                momentum = 0.8;  // Increase momentum after 20 iterations
            }
            
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    // Update adjusted learning rate (gains)
                    if grad[i][j] * velocities[i][j] > 0.0 {
                        gains[i][j] = gains[i][j] * 0.8;
                    } else {
                        gains[i][j] = gains[i][j] + 0.2;
                    }
                    
                    gains[i][j] = f64::max(gains[i][j], 0.01);
                    
                    // Update velocity
                    velocities[i][j] = momentum * velocities[i][j] - 
                                      self.learning_rate * gains[i][j] * grad[i][j];
                    
                    // Update embedding
                    self.embedding[i][j] += velocities[i][j];
                }
            }
            
            // Normalization of the embedding (center at origin)
            let mut mean = vec![0.0; self.n_components];
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    mean[j] += self.embedding[i][j];
                }
            }
            
            for j in 0..self.n_components {
                mean[j] /= n_samples as f64;
            }
            
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    self.embedding[i][j] -= mean[j];
                }
            }
        }
        
        self.fitted = true;
        
        Ok(())
    }
    
    fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "t-SNE has not been fitted yet".to_string()
            ));
        }
        
        // t-SNE cannot add new data points to existing embeddings,
        // so return an error if it's not the same data as training
        return Err(Error::InvalidOperation(
            "t-SNE does not support the transform method on new data. Use fit_transform instead.".to_string()
        ));
    }
    
    fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        self.fit(df)?;
        
        // Convert to OptimizedDataFrame
        let n_samples = df.row_count();
        let mut result_df = OptimizedDataFrame::new();
        
        // Add TSNE dimension columns
        for j in 0..self.n_components {
            let mut values = Vec::with_capacity(n_samples);
            
            for i in 0..n_samples {
                values.push(self.embedding[i][j]);
            }
            
            let col = Float64Column::new(values);
            result_df.add_column(format!("TSNE{}", j + 1), Column::Float64(col))?;
        }
        
        // Add non-numeric columns if any
        for col_name in df.column_names() {
            if !self.feature_names.contains(&col_name) {
                if let Ok(col_view) = df.column(&col_name) {
                    if let Some(str_col) = col_view.as_string() {
                        // String column
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = str_col.get(i) {
                                values.push(value.to_string());
                            } else {
                                values.push("".to_string());
                            }
                        }
                        let string_col = Column::String(crate::column::StringColumn::new(values));
                        result_df.add_column(col_name.clone(), string_col)?;
                    } else if let Some(bool_col) = col_view.as_boolean() {
                        // Boolean column
                        let mut values = Vec::with_capacity(n_samples);
                        for i in 0..n_samples {
                            if let Ok(Some(value)) = bool_col.get(i) {
                                values.push(value);
                            } else {
                                values.push(false);
                            }
                        }
                        let bool_col = Column::Boolean(crate::column::BooleanColumn::new(values));
                        result_df.add_column(col_name.clone(), bool_col)?;
                    }
                }
            }
        }
        
        Ok(result_df)
    }
}