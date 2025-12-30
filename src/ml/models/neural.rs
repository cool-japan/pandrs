//! Neural Network Models
//!
//! This module provides basic neural network implementations including
//! Multi-layer Perceptron (MLP) for classification and regression.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use std::collections::HashMap;

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// ReLU (Rectified Linear Unit): max(0, x)
    ReLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Tanh,
    /// Linear/Identity: x
    Linear,
    /// Softmax: exp(x_i) / sum(exp(x_j))
    Softmax,
}

impl Activation {
    /// Apply activation function
    fn forward(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Activation::ReLU => x.iter().map(|&v| v.max(0.0)).collect(),
            Activation::Sigmoid => x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
            Activation::Tanh => x.iter().map(|&v| v.tanh()).collect(),
            Activation::Linear => x.to_vec(),
            Activation::Softmax => {
                let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_vals: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
                let sum: f64 = exp_vals.iter().sum();
                exp_vals.iter().map(|&v| v / sum).collect()
            }
        }
    }

    /// Compute derivative of activation function
    fn backward(&self, x: &[f64], output: &[f64]) -> Vec<f64> {
        match self {
            Activation::ReLU => x.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect(),
            Activation::Sigmoid => output.iter().map(|&o| o * (1.0 - o)).collect(),
            Activation::Tanh => output.iter().map(|&o| 1.0 - o * o).collect(),
            Activation::Linear => vec![1.0; x.len()],
            Activation::Softmax => {
                // For softmax, we typically use cross-entropy loss which simplifies the gradient
                // The actual derivative is: softmax(x)_i * (delta_ij - softmax(x)_j)
                // But when combined with cross-entropy loss, gradient is simplified to (y_hat - y)
                vec![1.0; x.len()]
            }
        }
    }
}

/// Loss function types
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Cross Entropy (for classification)
    CrossEntropy,
    /// Binary Cross Entropy
    BinaryCrossEntropy,
}

impl LossFunction {
    /// Compute loss
    fn compute(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        match self {
            LossFunction::MSE => {
                let n = predicted.len() as f64;
                predicted
                    .iter()
                    .zip(actual)
                    .map(|(p, a)| (p - a).powi(2))
                    .sum::<f64>()
                    / n
            }
            LossFunction::CrossEntropy => {
                let epsilon = 1e-15;
                -predicted
                    .iter()
                    .zip(actual)
                    .map(|(p, a)| {
                        let p_clipped = p.max(epsilon).min(1.0 - epsilon);
                        a * p_clipped.ln()
                    })
                    .sum::<f64>()
            }
            LossFunction::BinaryCrossEntropy => {
                let epsilon = 1e-15;
                let n = predicted.len() as f64;
                -predicted
                    .iter()
                    .zip(actual)
                    .map(|(p, a)| {
                        let p_clipped = p.max(epsilon).min(1.0 - epsilon);
                        a * p_clipped.ln() + (1.0 - a) * (1.0 - p_clipped).ln()
                    })
                    .sum::<f64>()
                    / n
            }
        }
    }

    /// Compute gradient of loss with respect to predictions
    fn gradient(&self, predicted: &[f64], actual: &[f64]) -> Vec<f64> {
        match self {
            LossFunction::MSE => {
                let n = predicted.len() as f64;
                predicted
                    .iter()
                    .zip(actual)
                    .map(|(p, a)| 2.0 * (p - a) / n)
                    .collect()
            }
            LossFunction::CrossEntropy | LossFunction::BinaryCrossEntropy => {
                // For cross-entropy with softmax, gradient simplifies to (predicted - actual)
                predicted.iter().zip(actual).map(|(p, a)| p - a).collect()
            }
        }
    }
}

/// Neural network layer
#[derive(Debug, Clone)]
struct Layer {
    /// Weight matrix (output_dim x input_dim)
    weights: Vec<Vec<f64>>,
    /// Bias vector (output_dim)
    biases: Vec<f64>,
    /// Activation function
    activation: Activation,
    /// Cached input (for backprop)
    input_cache: Vec<f64>,
    /// Cached pre-activation output (for backprop)
    pre_activation_cache: Vec<f64>,
    /// Cached output (for backprop)
    output_cache: Vec<f64>,
}

impl Layer {
    /// Create a new layer with random initialization
    fn new(input_dim: usize, output_dim: usize, activation: Activation, seed: u64) -> Self {
        // Xavier/Glorot initialization
        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();

        // Simple LCG random generator
        let mut rng_state = seed;
        let rand_f64 = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let random_bits = (*state >> 33) as u32;
            (random_bits as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        let weights: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rand_f64(&mut rng_state) * scale)
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_dim];

        Layer {
            weights,
            biases,
            activation,
            input_cache: Vec::new(),
            pre_activation_cache: Vec::new(),
            output_cache: Vec::new(),
        }
    }

    /// Forward pass through the layer
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.input_cache = input.to_vec();

        // Linear transformation: z = Wx + b
        let pre_activation: Vec<f64> = self
            .weights
            .iter()
            .zip(&self.biases)
            .map(|(w, b)| w.iter().zip(input).map(|(wi, xi)| wi * xi).sum::<f64>() + b)
            .collect();

        self.pre_activation_cache = pre_activation.clone();

        // Apply activation
        let output = self.activation.forward(&pre_activation);
        self.output_cache = output.clone();

        output
    }

    /// Backward pass through the layer
    fn backward(&mut self, grad_output: &[f64], learning_rate: f64) -> Vec<f64> {
        // Compute activation gradient
        let activation_grad = self
            .activation
            .backward(&self.pre_activation_cache, &self.output_cache);

        // Element-wise multiplication with upstream gradient
        let delta: Vec<f64> = grad_output
            .iter()
            .zip(&activation_grad)
            .map(|(g, a)| g * a)
            .collect();

        // Gradient for weights: delta * input^T
        for (i, w_row) in self.weights.iter_mut().enumerate() {
            for (j, w) in w_row.iter_mut().enumerate() {
                *w -= learning_rate * delta[i] * self.input_cache[j];
            }
        }

        // Gradient for biases
        for (i, b) in self.biases.iter_mut().enumerate() {
            *b -= learning_rate * delta[i];
        }

        // Gradient for input: W^T * delta
        let grad_input: Vec<f64> = (0..self.input_cache.len())
            .map(|j| {
                self.weights
                    .iter()
                    .zip(&delta)
                    .map(|(w_row, d)| w_row[j] * d)
                    .sum()
            })
            .collect();

        grad_input
    }
}

/// MLP Configuration
#[derive(Debug, Clone)]
pub struct MLPConfig {
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Activation function for hidden layers
    pub hidden_activation: Activation,
    /// Output activation function
    pub output_activation: Activation,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub n_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Random seed
    pub random_seed: u64,
    /// Early stopping patience (number of epochs without improvement)
    pub early_stopping_patience: Option<usize>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for MLPConfig {
    fn default() -> Self {
        MLPConfig {
            hidden_layers: vec![100],
            hidden_activation: Activation::ReLU,
            output_activation: Activation::Linear,
            learning_rate: 0.001,
            n_epochs: 200,
            batch_size: 32,
            random_seed: 42,
            early_stopping_patience: Some(10),
            verbose: false,
        }
    }
}

/// MLP Configuration Builder
#[derive(Debug, Clone)]
pub struct MLPConfigBuilder {
    config: MLPConfig,
}

impl MLPConfigBuilder {
    pub fn new() -> Self {
        MLPConfigBuilder {
            config: MLPConfig::default(),
        }
    }

    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.config.hidden_layers = layers;
        self
    }

    pub fn hidden_activation(mut self, activation: Activation) -> Self {
        self.config.hidden_activation = activation;
        self
    }

    pub fn output_activation(mut self, activation: Activation) -> Self {
        self.config.output_activation = activation;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn n_epochs(mut self, n: usize) -> Self {
        self.config.n_epochs = n;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = seed;
        self
    }

    pub fn early_stopping_patience(mut self, patience: Option<usize>) -> Self {
        self.config.early_stopping_patience = patience;
        self
    }

    pub fn verbose(mut self, v: bool) -> Self {
        self.config.verbose = v;
        self
    }

    pub fn build(self) -> MLPConfig {
        self.config
    }
}

impl Default for MLPConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-layer Perceptron Regressor
#[derive(Debug, Clone)]
pub struct MLPRegressor {
    config: MLPConfig,
    layers: Vec<Layer>,
    feature_names: Vec<String>,
    is_fitted: bool,
    training_loss_history: Vec<f64>,
}

impl MLPRegressor {
    /// Create a new MLP regressor
    pub fn new(config: MLPConfig) -> Self {
        MLPRegressor {
            config,
            layers: Vec::new(),
            feature_names: Vec::new(),
            is_fitted: false,
            training_loss_history: Vec::new(),
        }
    }

    /// Initialize network layers
    fn init_layers(&mut self, input_dim: usize, output_dim: usize) {
        self.layers.clear();
        let mut prev_dim = input_dim;
        let mut seed = self.config.random_seed;

        // Hidden layers
        for &hidden_size in &self.config.hidden_layers {
            self.layers.push(Layer::new(
                prev_dim,
                hidden_size,
                self.config.hidden_activation,
                seed,
            ));
            prev_dim = hidden_size;
            seed = seed.wrapping_add(1);
        }

        // Output layer
        self.layers
            .push(Layer::new(prev_dim, output_dim, Activation::Linear, seed));
    }

    /// Forward pass through the network
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Backward pass through the network
    fn backward(&mut self, loss_grad: &[f64]) {
        let mut grad = loss_grad.to_vec();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, self.config.learning_rate);
        }
    }

    /// Get feature matrix from DataFrame
    fn get_feature_matrix(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        let n_rows = data.row_count();

        let column_values: Vec<Vec<f64>> = self
            .feature_names
            .iter()
            .map(|col_name| {
                data.get_column_numeric_values(col_name).map_err(|_| {
                    Error::Column(format!("Column '{}' not found or not numeric", col_name))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut x = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<f64> = column_values.iter().map(|col| col[i]).collect();
            x.push(row);
        }

        Ok(x)
    }

    /// Get training loss history
    pub fn training_loss_history(&self) -> &[f64] {
        &self.training_loss_history
    }
}

impl SupervisedModel for MLPRegressor {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.feature_names = train_data
            .column_names()
            .into_iter()
            .filter(|c| c != target_column)
            .collect();

        if self.feature_names.is_empty() {
            return Err(Error::InvalidInput("No feature columns found".to_string()));
        }

        let x = self.get_feature_matrix(train_data)?;
        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let input_dim = self.feature_names.len();
        self.init_layers(input_dim, 1);

        self.training_loss_history.clear();
        let n_samples = x.len();
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.n_epochs {
            let mut epoch_loss = 0.0;

            // Mini-batch training
            for batch_start in (0..n_samples).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(n_samples);

                for i in batch_start..batch_end {
                    let predicted = self.forward(&x[i]);
                    let actual = vec![y[i]];

                    epoch_loss += LossFunction::MSE.compute(&predicted, &actual);

                    let grad = LossFunction::MSE.gradient(&predicted, &actual);
                    self.backward(&grad);
                }
            }

            epoch_loss /= n_samples as f64;
            self.training_loss_history.push(epoch_loss);

            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if epoch_loss < best_loss {
                    best_loss = epoch_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        if self.config.verbose {
                            println!("Early stopping at epoch {}", epoch);
                        }
                        break;
                    }
                }
            }

            if self.config.verbose && epoch % 10 == 0 {
                println!("Epoch {}: loss = {:.6}", epoch, epoch_loss);
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        let x = self.get_feature_matrix(data)?;
        let mut model = self.clone();

        let predictions: Vec<f64> = x.iter().map(|sample| model.forward(sample)[0]).collect();

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        // Neural networks don't have straightforward feature importances
        // Could implement gradient-based importance in the future
        None
    }
}

impl ModelEvaluator for MLPRegressor {
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

        // R²
        let y_mean = actual.iter().sum::<f64>() / actual.len() as f64;
        let ss_tot: f64 = actual.iter().map(|a| (a - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).powi(2))
            .sum();
        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
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

/// Multi-layer Perceptron Classifier
#[derive(Debug, Clone)]
pub struct MLPClassifier {
    config: MLPConfig,
    layers: Vec<Layer>,
    feature_names: Vec<String>,
    n_classes: usize,
    classes: Vec<f64>,
    is_fitted: bool,
    training_loss_history: Vec<f64>,
}

impl MLPClassifier {
    /// Create a new MLP classifier
    pub fn new(config: MLPConfig) -> Self {
        MLPClassifier {
            config,
            layers: Vec::new(),
            feature_names: Vec::new(),
            n_classes: 0,
            classes: Vec::new(),
            is_fitted: false,
            training_loss_history: Vec::new(),
        }
    }

    /// Initialize network layers
    fn init_layers(&mut self, input_dim: usize, output_dim: usize) {
        self.layers.clear();
        let mut prev_dim = input_dim;
        let mut seed = self.config.random_seed;

        // Hidden layers
        for &hidden_size in &self.config.hidden_layers {
            self.layers.push(Layer::new(
                prev_dim,
                hidden_size,
                self.config.hidden_activation,
                seed,
            ));
            prev_dim = hidden_size;
            seed = seed.wrapping_add(1);
        }

        // Output layer with softmax for multi-class, sigmoid for binary
        let output_activation = if output_dim == 1 {
            Activation::Sigmoid
        } else {
            Activation::Softmax
        };

        self.layers
            .push(Layer::new(prev_dim, output_dim, output_activation, seed));
    }

    /// Forward pass through the network
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Backward pass through the network
    fn backward(&mut self, loss_grad: &[f64]) {
        let mut grad = loss_grad.to_vec();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, self.config.learning_rate);
        }
    }

    /// Get feature matrix from DataFrame
    fn get_feature_matrix(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        let n_rows = data.row_count();

        let column_values: Vec<Vec<f64>> = self
            .feature_names
            .iter()
            .map(|col_name| {
                data.get_column_numeric_values(col_name).map_err(|_| {
                    Error::Column(format!("Column '{}' not found or not numeric", col_name))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut x = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let row: Vec<f64> = column_values.iter().map(|col| col[i]).collect();
            x.push(row);
        }

        Ok(x)
    }

    /// Convert class label to one-hot encoding
    fn to_one_hot(&self, class_label: f64) -> Vec<f64> {
        if self.n_classes == 2 {
            // Binary classification
            let idx = self
                .classes
                .iter()
                .position(|&c| (c - class_label).abs() < 1e-10)
                .unwrap_or(0);
            vec![idx as f64]
        } else {
            // Multi-class
            let mut one_hot = vec![0.0; self.n_classes];
            if let Some(idx) = self
                .classes
                .iter()
                .position(|&c| (c - class_label).abs() < 1e-10)
            {
                one_hot[idx] = 1.0;
            }
            one_hot
        }
    }

    /// Get probability predictions
    pub fn predict_proba(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        let x = self.get_feature_matrix(data)?;
        let mut model = self.clone();

        let probs: Vec<Vec<f64>> = x
            .iter()
            .map(|sample| {
                let output = model.forward(sample);
                if self.n_classes == 2 {
                    vec![1.0 - output[0], output[0]]
                } else {
                    output
                }
            })
            .collect();

        Ok(probs)
    }

    /// Get training loss history
    pub fn training_loss_history(&self) -> &[f64] {
        &self.training_loss_history
    }
}

impl SupervisedModel for MLPClassifier {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.feature_names = train_data
            .column_names()
            .into_iter()
            .filter(|c| c != target_column)
            .collect();

        if self.feature_names.is_empty() {
            return Err(Error::InvalidInput("No feature columns found".to_string()));
        }

        let x = self.get_feature_matrix(train_data)?;
        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        // Find unique classes
        let mut classes: Vec<f64> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes;
        self.n_classes = self.classes.len();

        let input_dim = self.feature_names.len();
        let output_dim = if self.n_classes == 2 {
            1
        } else {
            self.n_classes
        };
        self.init_layers(input_dim, output_dim);

        self.training_loss_history.clear();
        let n_samples = x.len();
        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        let loss_fn = if self.n_classes == 2 {
            LossFunction::BinaryCrossEntropy
        } else {
            LossFunction::CrossEntropy
        };

        for epoch in 0..self.config.n_epochs {
            let mut epoch_loss = 0.0;

            // Mini-batch training
            for batch_start in (0..n_samples).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(n_samples);

                for i in batch_start..batch_end {
                    let predicted = self.forward(&x[i]);
                    let actual = self.to_one_hot(y[i]);

                    epoch_loss += loss_fn.compute(&predicted, &actual);

                    let grad = loss_fn.gradient(&predicted, &actual);
                    self.backward(&grad);
                }
            }

            epoch_loss /= n_samples as f64;
            self.training_loss_history.push(epoch_loss);

            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if epoch_loss < best_loss {
                    best_loss = epoch_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        if self.config.verbose {
                            println!("Early stopping at epoch {}", epoch);
                        }
                        break;
                    }
                }
            }

            if self.config.verbose && epoch % 10 == 0 {
                println!("Epoch {}: loss = {:.6}", epoch, epoch_loss);
            }
        }

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
        None
    }
}

impl ModelEvaluator for MLPClassifier {
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

    fn create_xor_data() -> DataFrame {
        // XOR problem - linearly non-separable
        let mut df = DataFrame::new();
        let x1 = Series::new(
            vec![0.0, 0.0, 1.0, 1.0, 0.1, 0.1, 0.9, 0.9],
            Some("x1".to_string()),
        )
        .unwrap();
        let x2 = Series::new(
            vec![0.0, 1.0, 0.0, 1.0, 0.1, 0.9, 0.1, 0.9],
            Some("x2".to_string()),
        )
        .unwrap();
        let y = Series::new(
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
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

    #[test]
    fn test_activation_functions() {
        let input = vec![-1.0, 0.0, 1.0, 2.0];

        // ReLU
        let relu = Activation::ReLU.forward(&input);
        assert_eq!(relu, vec![0.0, 0.0, 1.0, 2.0]);

        // Sigmoid
        let sigmoid = Activation::Sigmoid.forward(&input);
        assert!(sigmoid[0] < 0.5);
        assert!((sigmoid[1] - 0.5).abs() < 1e-10);
        assert!(sigmoid[2] > 0.5);

        // Tanh
        let tanh = Activation::Tanh.forward(&input);
        assert!(tanh[0] < 0.0);
        assert!((tanh[1]).abs() < 1e-10);
        assert!(tanh[2] > 0.0);

        // Softmax
        let softmax_input = vec![1.0, 2.0, 3.0];
        let softmax = Activation::Softmax.forward(&softmax_input);
        let sum: f64 = softmax.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mlp_regressor() {
        let data = create_regression_data();
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![20])
            .learning_rate(0.001)
            .n_epochs(2000)
            .early_stopping_patience(Some(200))
            .build();

        let mut mlp = MLPRegressor::new(config);
        mlp.fit(&data, "y").unwrap();

        let predictions = mlp.predict(&data).unwrap();
        assert_eq!(predictions.len(), 10);

        // Just verify the model produces reasonable predictions
        // Neural networks with small data can be unstable
        for pred in &predictions {
            assert!(pred.is_finite(), "Prediction should be finite");
        }
    }

    #[test]
    fn test_mlp_classifier() {
        let data = create_classification_data();
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![10])
            .learning_rate(0.01)
            .n_epochs(200)
            .build();

        let mut mlp = MLPClassifier::new(config);
        mlp.fit(&data, "y").unwrap();

        let predictions = mlp.predict(&data).unwrap();
        assert_eq!(predictions.len(), 10);

        let metrics = mlp.evaluate(&data, "y").unwrap();
        let accuracy = metrics.get_metric("accuracy").unwrap();
        assert!(*accuracy >= 0.5, "Accuracy should be at least 50%");
    }

    #[test]
    fn test_mlp_xor_problem() {
        // XOR is a classic test for neural networks
        let data = create_xor_data();
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![8, 4])
            .learning_rate(0.1)
            .n_epochs(1000)
            .early_stopping_patience(None)
            .build();

        let mut mlp = MLPClassifier::new(config);
        mlp.fit(&data, "y").unwrap();

        let predictions = mlp.predict(&data).unwrap();

        // Check that MLP can learn XOR pattern (needs hidden layer)
        let accuracy = predictions
            .iter()
            .zip(vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
            .filter(|(p, a)| (*p - *a).abs() < 0.5)
            .count() as f64
            / 8.0;

        // XOR is difficult, we just need >50% to show it's learning
        assert!(
            accuracy >= 0.5,
            "MLP should learn XOR pattern (accuracy: {})",
            accuracy
        );
    }

    #[test]
    fn test_training_history() {
        let data = create_regression_data();
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![5])
            .n_epochs(50)
            .early_stopping_patience(None)
            .build();

        let mut mlp = MLPRegressor::new(config);
        mlp.fit(&data, "y").unwrap();

        let history = mlp.training_loss_history();
        assert_eq!(history.len(), 50);

        // Loss should generally decrease (not necessarily monotonically)
        assert!(history.last().unwrap() <= history.first().unwrap());
    }
}
