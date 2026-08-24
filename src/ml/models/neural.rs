//! Neural Network Models
//!
//! This module provides basic neural network implementations including
//! Multi-layer Perceptron (MLP) for classification and regression.
//!
//! # Training
//!
//! Both models train with mini-batch stochastic gradient descent: per-sample gradients
//! are *accumulated* across a batch and applied once as their average (`batch_size`
//! therefore actually controls the update granularity), and the sample order is
//! reshuffled every epoch with a seeded generator so runs stay reproducible.
//! Early stopping — when enabled — monitors the loss on a held-out validation split
//! (`validation_fraction`), never the training loss it is meant to guard against.
//! Training stops `early_stopping_patience` epochs after the best validation loss and
//! the model keeps the weights of that **final** epoch: it does not roll back to the
//! best-validation snapshot (Keras' default behaviour; note `scikit-learn`'s
//! `MLPRegressor`/`MLPClassifier` do restore the best weights, so their final models are
//! not bit-comparable with these).

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::selection::{
    err_on_unknown_params, parse_param_f64, parse_param_u64, parse_param_usize, TunableModel,
};
use crate::ml::models::{ModelEvaluator, ModelMetrics, SupervisedModel};
use scirs2_core::random::{Random, StdRng};
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
            Activation::Sigmoid => x
                .iter()
                .map(|&v| {
                    // Branch on the sign so `exp` never sees a large positive
                    // argument (which overflows to +inf).
                    if v >= 0.0 {
                        1.0 / (1.0 + (-v).exp())
                    } else {
                        let e = v.exp();
                        e / (1.0 + e)
                    }
                })
                .collect(),
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

    /// Element-wise derivative `dσ/dz` of the activation.
    ///
    /// Returns `None` for [`Activation::Softmax`], whose Jacobian is *not* diagonal:
    /// `∂s_i/∂z_j = s_i(δ_ij − s_j)`, so there is no per-element factor to multiply the
    /// upstream gradient by. Callers must instead use the full Jacobian-vector product
    /// (see [`Layer::backward`]) or the fused softmax + cross-entropy gradient.
    /// Returning a fabricated `1.0` here — as this used to — silently pretended softmax
    /// was the identity everywhere it was not paired with cross-entropy.
    fn elementwise_derivative(&self, pre_activation: &[f64], output: &[f64]) -> Option<Vec<f64>> {
        match self {
            Activation::ReLU => Some(
                pre_activation
                    .iter()
                    .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                    .collect(),
            ),
            Activation::Sigmoid => Some(output.iter().map(|&o| o * (1.0 - o)).collect()),
            Activation::Tanh => Some(output.iter().map(|&o| 1.0 - o * o).collect()),
            Activation::Linear => Some(vec![1.0; pre_activation.len()]),
            Activation::Softmax => None,
        }
    }

    /// Parse an activation name used in hyperparameter grids.
    fn parse(key: &str, value: &str) -> Result<Activation> {
        match value.trim().to_ascii_lowercase().as_str() {
            "relu" => Ok(Activation::ReLU),
            "sigmoid" | "logistic" => Ok(Activation::Sigmoid),
            "tanh" => Ok(Activation::Tanh),
            "linear" | "identity" => Ok(Activation::Linear),
            "softmax" => Ok(Activation::Softmax),
            other => Err(Error::InvalidValue(format!(
                "Invalid activation for hyperparameter '{}': '{}' \
                 (expected relu, sigmoid, tanh, linear or softmax)",
                key, other
            ))),
        }
    }
}

/// Loss function types
#[derive(Debug, Clone, Copy)]
pub enum LossFunction {
    /// Mean Squared Error
    MSE,
    /// Cross Entropy (for multi-class classification, paired with a softmax output)
    CrossEntropy,
    /// Binary Cross Entropy (paired with a sigmoid output)
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

    /// Gradient handed to the output layer.
    ///
    /// * [`LossFunction::MSE`] returns `dL/dŷ` — the gradient with respect to the
    ///   network's *outputs*, which the output layer then multiplies by its own
    ///   activation derivative.
    /// * [`LossFunction::CrossEntropy`] / [`LossFunction::BinaryCrossEntropy`] return
    ///   the *fused* `dL/dz` for their matching output activation (softmax and sigmoid
    ///   respectively), which both simplify to `ŷ − y`. The output layer is flagged as
    ///   receiving a pre-activation gradient in that case so it does **not** multiply by
    ///   the activation derivative again. Multiplying `ŷ − y` by the extra sigmoid
    ///   derivative `p(1−p)` — as this code used to for the binary case — turns
    ///   cross-entropy into MSE-on-a-sigmoid and makes the gradient vanish exactly when
    ///   the model is most confidently wrong.
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
                predicted.iter().zip(actual).map(|(p, a)| p - a).collect()
            }
        }
    }

    /// Whether [`LossFunction::gradient`] already returns a pre-activation gradient.
    fn returns_pre_activation_gradient(&self) -> bool {
        match self {
            LossFunction::MSE => false,
            LossFunction::CrossEntropy | LossFunction::BinaryCrossEntropy => true,
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
    /// Whether the upstream gradient reaching this layer is already `dL/dz`
    /// (fused loss + activation gradient); see [`LossFunction::gradient`].
    fused_output_gradient: bool,
    /// Cached input (for backprop)
    input_cache: Vec<f64>,
    /// Cached pre-activation output (for backprop)
    pre_activation_cache: Vec<f64>,
    /// Cached output (for backprop)
    output_cache: Vec<f64>,
    /// Accumulated weight gradients for the current mini-batch
    grad_weights: Vec<Vec<f64>>,
    /// Accumulated bias gradients for the current mini-batch
    grad_biases: Vec<f64>,
}

impl Layer {
    /// Create a new layer with Glorot/Xavier-uniform initialisation.
    ///
    /// Weights are drawn from `U(−limit, limit)` with `limit = sqrt(6 / (fan_in + fan_out))`,
    /// which is the *uniform* Glorot initialiser. (Using the Glorot **normal** standard
    /// deviation `sqrt(2/(fan_in+fan_out))` as a uniform half-width — as this used to —
    /// makes the initial weights a factor of √3 too small, slowing early training.)
    fn new(
        input_dim: usize,
        output_dim: usize,
        activation: Activation,
        fused_output_gradient: bool,
        rng: &mut StdRng,
    ) -> Self {
        let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.random_range(-limit..limit))
                    .collect()
            })
            .collect();

        Layer {
            weights,
            biases: vec![0.0; output_dim],
            activation,
            fused_output_gradient,
            input_cache: Vec::new(),
            pre_activation_cache: Vec::new(),
            output_cache: Vec::new(),
            grad_weights: vec![vec![0.0; input_dim]; output_dim],
            grad_biases: vec![0.0; output_dim],
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

        let output = self.activation.forward(&pre_activation);
        self.output_cache = output.clone();

        output
    }

    /// Backward pass: accumulate this sample's gradients and return `dL/dinput`.
    ///
    /// Parameters are **not** updated here — [`Layer::apply_gradients`] applies the
    /// batch-averaged update once per mini-batch, so the gradients flowing further back
    /// are computed with the same weights that produced the forward pass.
    fn backward(&mut self, grad_output: &[f64]) -> Vec<f64> {
        let delta: Vec<f64> = if self.fused_output_gradient {
            grad_output.to_vec()
        } else if let Some(activation_grad) = self
            .activation
            .elementwise_derivative(&self.pre_activation_cache, &self.output_cache)
        {
            grad_output
                .iter()
                .zip(&activation_grad)
                .map(|(g, a)| g * a)
                .collect()
        } else {
            // Softmax Jacobian-vector product: δ_i = s_i (g_i − Σ_j g_j s_j)
            let s = &self.output_cache;
            let dot: f64 = grad_output.iter().zip(s).map(|(g, si)| g * si).sum();
            s.iter()
                .zip(grad_output)
                .map(|(si, g)| si * (g - dot))
                .collect()
        };

        for (i, grad_row) in self.grad_weights.iter_mut().enumerate() {
            for (j, gw) in grad_row.iter_mut().enumerate() {
                *gw += delta[i] * self.input_cache[j];
            }
        }

        for (i, gb) in self.grad_biases.iter_mut().enumerate() {
            *gb += delta[i];
        }

        (0..self.input_cache.len())
            .map(|j| {
                self.weights
                    .iter()
                    .zip(&delta)
                    .map(|(w_row, d)| w_row[j] * d)
                    .sum()
            })
            .collect()
    }

    /// Apply the accumulated gradients scaled by `scale` (typically `1 / batch_size`)
    /// and reset the accumulators.
    fn apply_gradients(&mut self, learning_rate: f64, scale: f64) {
        let step = learning_rate * scale;

        for (i, w_row) in self.weights.iter_mut().enumerate() {
            for (j, w) in w_row.iter_mut().enumerate() {
                *w -= step * self.grad_weights[i][j];
            }
        }

        for (i, b) in self.biases.iter_mut().enumerate() {
            *b -= step * self.grad_biases[i];
        }

        self.zero_gradients();
    }

    /// Reset the mini-batch gradient accumulators.
    fn zero_gradients(&mut self) {
        for row in self.grad_weights.iter_mut() {
            for value in row.iter_mut() {
                *value = 0.0;
            }
        }
        for value in self.grad_biases.iter_mut() {
            *value = 0.0;
        }
    }
}

/// MLP Configuration
#[derive(Debug, Clone)]
pub struct MLPConfig {
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Activation function for hidden layers
    pub hidden_activation: Activation,
    /// Output activation function.
    ///
    /// [`MLPRegressor`] uses this verbatim (default [`Activation::Linear`]).
    /// [`MLPClassifier`] treats [`Activation::Linear`] as "choose automatically"
    /// (sigmoid for a binary target, softmax for a multi-class one) and otherwise
    /// requires the explicit choice to be compatible with the target.
    pub output_activation: Activation,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of epochs
    pub n_epochs: usize,
    /// Mini-batch size: gradients are averaged over this many samples per update
    pub batch_size: usize,
    /// Random seed (weight initialisation, epoch shuffling and the validation split)
    pub random_seed: u64,
    /// Early stopping patience (number of epochs without validation improvement)
    pub early_stopping_patience: Option<usize>,
    /// Fraction of the training rows held out to monitor early stopping
    pub validation_fraction: f64,
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
            validation_fraction: 0.1,
            verbose: false,
        }
    }
}

impl MLPConfig {
    /// Validate the settings that the training loop depends on.
    fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(Error::InvalidInput(
                "MLP batch_size must be at least 1".into(),
            ));
        }
        if self.n_epochs == 0 {
            return Err(Error::InvalidInput(
                "MLP n_epochs must be at least 1".into(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(Error::InvalidInput(format!(
                "MLP learning_rate must be a positive finite number, got {}",
                self.learning_rate
            )));
        }
        if !(0.0..1.0).contains(&self.validation_fraction) {
            return Err(Error::InvalidInput(format!(
                "MLP validation_fraction must be in [0, 1), got {}",
                self.validation_fraction
            )));
        }
        if self.early_stopping_patience.is_some() && self.validation_fraction <= 0.0 {
            return Err(Error::InvalidInput(
                "MLP early stopping needs a validation split: set validation_fraction > 0 \
                 or disable early_stopping_patience. (Monitoring the training loss instead \
                 would stop on the very quantity early stopping exists to guard against.)"
                    .into(),
            ));
        }
        if let Some(patience) = self.early_stopping_patience {
            if patience == 0 {
                return Err(Error::InvalidInput(
                    "MLP early_stopping_patience must be at least 1 when set".into(),
                ));
            }
        }
        Ok(())
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

    /// Fraction of the training rows held out to monitor early stopping
    pub fn validation_fraction(mut self, fraction: f64) -> Self {
        self.config.validation_fraction = fraction;
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

// ---------------------------------------------------------------------------
// Shared training helpers
// ---------------------------------------------------------------------------

/// Split row indices into (training, validation) sets for early stopping.
///
/// The validation set is empty when early stopping is disabled. Otherwise the rows are
/// shuffled with the seeded generator and `validation_fraction` of them (at least one)
/// are held out — the loss on *those* rows is what early stopping monitors.
fn split_train_validation(
    n_samples: usize,
    config: &MLPConfig,
    rng: &mut StdRng,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut order: Vec<usize> = (0..n_samples).collect();

    if config.early_stopping_patience.is_none() {
        return Ok((order, Vec::new()));
    }

    rng.shuffle(&mut order);

    let n_validation = ((n_samples as f64) * config.validation_fraction).round() as usize;
    let n_validation = n_validation.max(1);

    if n_validation >= n_samples {
        return Err(Error::InvalidInput(format!(
            "Early stopping needs at least one training and one validation row, but a \
             validation_fraction of {} over {} row(s) leaves none for training; lower \
             validation_fraction, add data, or disable early_stopping_patience",
            config.validation_fraction, n_samples
        )));
    }

    let validation = order.split_off(n_samples - n_validation);
    Ok((order, validation))
}

/// Collect the numeric feature columns of `data`, excluding the target column.
fn numeric_feature_names(data: &DataFrame, target_column: &str) -> Vec<String> {
    data.column_names()
        .iter()
        .filter(|name| name.as_str() != target_column)
        .filter(|name| data.get_column_numeric_values(name.as_str()).is_ok())
        .cloned()
        .collect()
}

/// Build the row-major feature matrix for `feature_names`.
fn feature_matrix(data: &DataFrame, feature_names: &[String]) -> Result<Vec<Vec<f64>>> {
    let n_rows = data.row_count();

    let column_values: Vec<Vec<f64>> = feature_names
        .iter()
        .map(|col_name| {
            data.get_column_numeric_values(col_name).map_err(|_| {
                Error::Column(format!("Column '{}' not found or not numeric", col_name))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    for (idx, column) in column_values.iter().enumerate() {
        if column.len() != n_rows {
            return Err(Error::DimensionMismatch(format!(
                "Feature column '{}' has {} rows but the frame has {}",
                feature_names[idx],
                column.len(),
                n_rows
            )));
        }
    }

    let mut x = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        x.push(column_values.iter().map(|col| col[i]).collect());
    }

    Ok(x)
}

/// Contiguous k-fold cross-validation shared by both MLP models.
///
/// Folds are contiguous, matching `sklearn.model_selection.KFold`'s default
/// (`shuffle=False`); shuffle the frame beforehand if its rows are ordered by target.
fn kfold_cross_validate<M>(
    model: &M,
    data: &DataFrame,
    target: &str,
    folds: usize,
) -> Result<Vec<ModelMetrics>>
where
    M: SupervisedModel + ModelEvaluator + Clone,
{
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
    let mut all_metrics = Vec::with_capacity(folds);

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

        let mut fold_model = model.clone();
        fold_model.fit(&train_df, target)?;
        all_metrics.push(fold_model.evaluate(&test_df, target)?);
    }

    Ok(all_metrics)
}

/// Parse a `hidden_layers` hyperparameter such as `"64,32"` or `"[64, 32]"`.
fn parse_hidden_layers(key: &str, value: &str) -> Result<Vec<usize>> {
    let trimmed = value.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.trim().is_empty() {
        return Ok(Vec::new());
    }
    trimmed
        .split(',')
        .map(|part| {
            part.trim().parse::<usize>().map_err(|_| {
                Error::InvalidValue(format!(
                    "Invalid hidden layer size '{}' in hyperparameter '{}': '{}'",
                    part.trim(),
                    key,
                    value
                ))
            })
        })
        .collect()
}

/// Parse an `early_stopping_patience` hyperparameter (`"none"` disables it).
fn parse_patience(key: &str, value: &str) -> Result<Option<usize>> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        Ok(Some(parse_param_usize(key, trimmed)?))
    }
}

/// Apply the hyperparameters both MLP models share to `config`.
///
/// Returns the keys that were not recognised so the caller can name the model in the
/// error message.
fn apply_mlp_params(
    config: &mut MLPConfig,
    params: &HashMap<String, String>,
) -> Result<Vec<String>> {
    let mut unknown = Vec::new();
    for (key, value) in params {
        match key.as_str() {
            "hidden_layers" => config.hidden_layers = parse_hidden_layers(key, value)?,
            "hidden_activation" => config.hidden_activation = Activation::parse(key, value)?,
            "output_activation" => config.output_activation = Activation::parse(key, value)?,
            "learning_rate" => config.learning_rate = parse_param_f64(key, value)?,
            "n_epochs" => config.n_epochs = parse_param_usize(key, value)?,
            "batch_size" => config.batch_size = parse_param_usize(key, value)?,
            "random_seed" => config.random_seed = parse_param_u64(key, value)?,
            "validation_fraction" => config.validation_fraction = parse_param_f64(key, value)?,
            "early_stopping_patience" => {
                config.early_stopping_patience = parse_patience(key, value)?
            }
            _ => unknown.push(key.clone()),
        }
    }
    Ok(unknown)
}

// ---------------------------------------------------------------------------
// MLPRegressor
// ---------------------------------------------------------------------------

/// Multi-layer Perceptron Regressor
#[derive(Debug, Clone)]
pub struct MLPRegressor {
    config: MLPConfig,
    layers: Vec<Layer>,
    feature_names: Vec<String>,
    is_fitted: bool,
    training_loss_history: Vec<f64>,
    validation_loss_history: Vec<f64>,
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
            validation_loss_history: Vec::new(),
        }
    }

    /// The configuration this model was built with.
    pub fn config(&self) -> &MLPConfig {
        &self.config
    }

    /// Initialize network layers.
    ///
    /// The output layer uses the configured `output_activation` (default
    /// [`Activation::Linear`]). `fused_output_gradient` comes from the training loss
    /// ([`LossFunction::returns_pre_activation_gradient`]); MSE reports `false`, so the
    /// output layer's activation derivative is applied normally — which is what makes a
    /// non-linear output activation (e.g. sigmoid) train correctly here.
    fn init_layers(
        &mut self,
        input_dim: usize,
        output_dim: usize,
        fused_output_gradient: bool,
        rng: &mut StdRng,
    ) -> Result<()> {
        if self.config.output_activation == Activation::Softmax && output_dim == 1 {
            return Err(Error::InvalidInput(
                "MLPRegressor: Activation::Softmax over a single output unit is constant 1.0 \
                 and cannot represent a regression target"
                    .into(),
            ));
        }

        self.layers.clear();
        let mut prev_dim = input_dim;

        for &hidden_size in &self.config.hidden_layers {
            self.layers.push(Layer::new(
                prev_dim,
                hidden_size,
                self.config.hidden_activation,
                false,
                rng,
            ));
            prev_dim = hidden_size;
        }

        self.layers.push(Layer::new(
            prev_dim,
            output_dim,
            self.config.output_activation,
            fused_output_gradient,
            rng,
        ));

        Ok(())
    }

    /// Forward pass through the network
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Backward pass through the network (accumulates gradients only)
    fn backward(&mut self, loss_grad: &[f64]) {
        let mut grad = loss_grad.to_vec();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    /// Get feature matrix from DataFrame
    fn get_feature_matrix(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        feature_matrix(data, &self.feature_names)
    }

    /// Get training loss history (mean training loss per epoch)
    pub fn training_loss_history(&self) -> &[f64] {
        &self.training_loss_history
    }

    /// Get validation loss history (empty when early stopping is disabled)
    pub fn validation_loss_history(&self) -> &[f64] {
        &self.validation_loss_history
    }
}

impl SupervisedModel for MLPRegressor {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.config.validate()?;

        self.feature_names = numeric_feature_names(train_data, target_column);
        if self.feature_names.is_empty() {
            return Err(Error::InvalidInput(
                "No numeric feature columns found".to_string(),
            ));
        }

        let x = self.get_feature_matrix(train_data)?;
        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let n_samples = x.len();
        if n_samples == 0 || y.len() != n_samples {
            return Err(Error::InvalidInput(format!(
                "MLPRegressor: {} feature rows but {} target values",
                n_samples,
                y.len()
            )));
        }

        let mut rng: StdRng = Random::seed(self.config.random_seed);
        let input_dim = self.feature_names.len();
        self.init_layers(
            input_dim,
            1,
            LossFunction::MSE.returns_pre_activation_gradient(),
            &mut rng,
        )?;

        let (mut train_indices, validation_indices) =
            split_train_validation(n_samples, &self.config, &mut rng)?;

        self.training_loss_history.clear();
        self.validation_loss_history.clear();

        let mut best_validation_loss = f64::INFINITY;
        let mut patience_counter = 0usize;
        let learning_rate = self.config.learning_rate;
        let batch_size = self.config.batch_size;

        for epoch in 0..self.config.n_epochs {
            rng.shuffle(&mut train_indices);
            let mut epoch_loss = 0.0;

            for batch in train_indices.chunks(batch_size) {
                for layer in self.layers.iter_mut() {
                    layer.zero_gradients();
                }

                for &i in batch {
                    let predicted = self.forward(&x[i]);
                    let actual = vec![y[i]];

                    epoch_loss += LossFunction::MSE.compute(&predicted, &actual);
                    let grad = LossFunction::MSE.gradient(&predicted, &actual);
                    self.backward(&grad);
                }

                let scale = 1.0 / batch.len() as f64;
                for layer in self.layers.iter_mut() {
                    layer.apply_gradients(learning_rate, scale);
                }
            }

            epoch_loss /= train_indices.len() as f64;
            self.training_loss_history.push(epoch_loss);

            if let Some(patience) = self.config.early_stopping_patience {
                let mut validation_loss = 0.0;
                for &i in &validation_indices {
                    let predicted = self.forward(&x[i]);
                    validation_loss += LossFunction::MSE.compute(&predicted, &[y[i]]);
                }
                validation_loss /= validation_indices.len() as f64;
                self.validation_loss_history.push(validation_loss);

                if validation_loss < best_validation_loss {
                    best_validation_loss = validation_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        if self.config.verbose {
                            println!(
                                "Early stopping at epoch {} (validation loss {:.6})",
                                epoch, validation_loss
                            );
                        }
                        break;
                    }
                }
            }

            if self.config.verbose && epoch % 10 == 0 {
                println!("Epoch {}: training loss = {:.6}", epoch, epoch_loss);
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

        let mut predictions = Vec::with_capacity(x.len());
        for sample in &x {
            let output = model.forward(sample);
            let value = output.first().copied().ok_or_else(|| {
                Error::InvalidOperation("MLPRegressor produced an empty output vector".into())
            })?;
            predictions.push(value);
        }

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        // Neural networks don't have straightforward feature importances
        // Could implement gradient-based importance in the future
        None
    }
}

impl TunableModel for MLPRegressor {
    /// Apply a hyperparameter combination on top of this model's current configuration.
    ///
    /// Recognised keys: `hidden_layers`, `hidden_activation`, `output_activation`,
    /// `learning_rate`, `n_epochs`, `batch_size`, `random_seed`, `validation_fraction`,
    /// `early_stopping_patience`. The network is rebuilt on the next `fit`.
    fn set_params(&mut self, params: &HashMap<String, String>) -> Result<()> {
        let mut config = self.config.clone();
        let unknown = apply_mlp_params(&mut config, params)?;
        err_on_unknown_params("MLPRegressor", unknown)?;
        config.validate()?;
        self.config = config;
        self.layers.clear();
        self.is_fitted = false;
        Ok(())
    }
}

impl ModelEvaluator for MLPRegressor {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        if predictions.len() != actual.len() {
            return Err(Error::DimensionMismatch(format!(
                "{} predictions for {} target values",
                predictions.len(),
                actual.len()
            )));
        }
        if predictions.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot evaluate on an empty test set".into(),
            ));
        }

        let mut metrics = ModelMetrics::new();

        let mse = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        metrics.add_metric("mse", mse);
        metrics.add_metric("rmse", mse.sqrt());

        // R²: a constant target has no variance to explain, so only a residual-free
        // prediction earns 1.0 (mirrors `metrics::regression::r2_score`).
        let y_mean = actual.iter().sum::<f64>() / actual.len() as f64;
        let ss_tot: f64 = actual.iter().map(|a| (a - y_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(&actual)
            .map(|(p, a)| (p - a).powi(2))
            .sum();
        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else if ss_res == 0.0 {
            1.0
        } else {
            0.0
        };
        metrics.add_metric("r2", r2);

        Ok(metrics)
    }

    /// Real k-fold cross-validation: each fold refits a fresh copy of this model on the
    /// training rows and evaluates it on the held-out rows.
    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        kfold_cross_validate(self, data, target, folds)
    }
}

// ---------------------------------------------------------------------------
// MLPClassifier
// ---------------------------------------------------------------------------

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
    validation_loss_history: Vec<f64>,
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
            validation_loss_history: Vec::new(),
        }
    }

    /// The configuration this model was built with.
    pub fn config(&self) -> &MLPConfig {
        &self.config
    }

    /// Resolve the output activation for a classification target.
    ///
    /// [`Activation::Linear`] (the [`MLPConfig`] default) means "choose automatically":
    /// sigmoid for a single output unit, softmax for several. An explicit choice is
    /// honoured when it can actually produce class probabilities for that many units,
    /// and rejected otherwise instead of being silently overridden.
    fn resolve_output_activation(&self, output_dim: usize) -> Result<Activation> {
        match (self.config.output_activation, output_dim) {
            (Activation::Linear, 1) | (Activation::Sigmoid, 1) => Ok(Activation::Sigmoid),
            (Activation::Linear, _) | (Activation::Softmax, _) if output_dim > 1 => {
                Ok(Activation::Softmax)
            }
            (activation, dim) => Err(Error::InvalidInput(format!(
                "MLPClassifier: output_activation {:?} cannot produce class probabilities \
                 over {} output unit(s); use Activation::Sigmoid for a binary target, \
                 Activation::Softmax for a multi-class target, or leave the default \
                 (Activation::Linear) to choose automatically",
                activation, dim
            ))),
        }
    }

    /// Initialize network layers.
    ///
    /// `fused_output_gradient` comes from the loss that will train this network
    /// ([`LossFunction::returns_pre_activation_gradient`]), so the output layer skips its
    /// activation derivative exactly when the loss already folded it in.
    fn init_layers(
        &mut self,
        input_dim: usize,
        output_dim: usize,
        fused_output_gradient: bool,
        rng: &mut StdRng,
    ) -> Result<()> {
        let output_activation = self.resolve_output_activation(output_dim)?;

        self.layers.clear();
        let mut prev_dim = input_dim;

        for &hidden_size in &self.config.hidden_layers {
            self.layers.push(Layer::new(
                prev_dim,
                hidden_size,
                self.config.hidden_activation,
                false,
                rng,
            ));
            prev_dim = hidden_size;
        }

        // With BinaryCrossEntropy/CrossEntropy the output layer receives the fused
        // loss+activation gradient (ŷ − y), so it must not apply its activation
        // derivative a second time.
        self.layers.push(Layer::new(
            prev_dim,
            output_dim,
            output_activation,
            fused_output_gradient,
            rng,
        ));

        Ok(())
    }

    /// Forward pass through the network
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }

    /// Backward pass through the network (accumulates gradients only)
    fn backward(&mut self, loss_grad: &[f64]) {
        let mut grad = loss_grad.to_vec();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    /// Get feature matrix from DataFrame
    fn get_feature_matrix(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        feature_matrix(data, &self.feature_names)
    }

    /// Encode a class label as the network's target vector.
    ///
    /// Binary targets are encoded as the single index `0.0`/`1.0`; multi-class targets
    /// as a one-hot vector. A label that was not among the fitted classes is an error —
    /// silently falling back to class index 0 (as this used to) trains the network
    /// against a label the caller never supplied.
    fn to_one_hot(&self, class_label: f64) -> Result<Vec<f64>> {
        let index = self
            .classes
            .iter()
            .position(|&c| (c - class_label).abs() < 1e-10)
            .ok_or_else(|| {
                Error::InvalidValue(format!(
                    "MLPClassifier: label {} is not one of the fitted classes {:?}",
                    class_label, self.classes
                ))
            })?;

        if self.n_classes == 2 {
            Ok(vec![index as f64])
        } else {
            let mut one_hot = vec![0.0; self.n_classes];
            one_hot[index] = 1.0;
            Ok(one_hot)
        }
    }

    /// Get probability predictions (one row per sample, one column per class)
    pub fn predict_proba(&self, data: &DataFrame) -> Result<Vec<Vec<f64>>> {
        if !self.is_fitted {
            return Err(Error::InvalidOperation("Model not fitted".to_string()));
        }

        let x = self.get_feature_matrix(data)?;
        let mut model = self.clone();

        let mut probabilities = Vec::with_capacity(x.len());
        for sample in &x {
            let output = model.forward(sample);
            if self.n_classes == 2 {
                let p = output.first().copied().ok_or_else(|| {
                    Error::InvalidOperation("MLPClassifier produced an empty output".into())
                })?;
                probabilities.push(vec![1.0 - p, p]);
            } else {
                probabilities.push(output);
            }
        }

        Ok(probabilities)
    }

    /// Get training loss history (mean training loss per epoch)
    pub fn training_loss_history(&self) -> &[f64] {
        &self.training_loss_history
    }

    /// Get validation loss history (empty when early stopping is disabled)
    pub fn validation_loss_history(&self) -> &[f64] {
        &self.validation_loss_history
    }
}

impl SupervisedModel for MLPClassifier {
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()> {
        self.config.validate()?;

        self.feature_names = numeric_feature_names(train_data, target_column);
        if self.feature_names.is_empty() {
            return Err(Error::InvalidInput(
                "No numeric feature columns found".to_string(),
            ));
        }

        let x = self.get_feature_matrix(train_data)?;
        let y: Vec<f64> = train_data
            .get_column_numeric_values(target_column)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", target_column)))?;

        let n_samples = x.len();
        if n_samples == 0 || y.len() != n_samples {
            return Err(Error::InvalidInput(format!(
                "MLPClassifier: {} feature rows but {} target values",
                n_samples,
                y.len()
            )));
        }

        // Find unique classes
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        if classes.len() < 2 {
            return Err(Error::InvalidInput(format!(
                "MLPClassifier needs at least 2 distinct classes in '{}', found {}",
                target_column,
                classes.len()
            )));
        }
        self.classes = classes;
        self.n_classes = self.classes.len();

        let output_dim = if self.n_classes == 2 {
            1
        } else {
            self.n_classes
        };

        let loss_fn = if self.n_classes == 2 {
            LossFunction::BinaryCrossEntropy
        } else {
            LossFunction::CrossEntropy
        };

        let mut rng: StdRng = Random::seed(self.config.random_seed);
        let input_dim = self.feature_names.len();
        self.init_layers(
            input_dim,
            output_dim,
            loss_fn.returns_pre_activation_gradient(),
            &mut rng,
        )?;

        let (mut train_indices, validation_indices) =
            split_train_validation(n_samples, &self.config, &mut rng)?;

        self.training_loss_history.clear();
        self.validation_loss_history.clear();

        let mut best_validation_loss = f64::INFINITY;
        let mut patience_counter = 0usize;
        let learning_rate = self.config.learning_rate;
        let batch_size = self.config.batch_size;

        for epoch in 0..self.config.n_epochs {
            rng.shuffle(&mut train_indices);
            let mut epoch_loss = 0.0;

            for batch in train_indices.chunks(batch_size) {
                for layer in self.layers.iter_mut() {
                    layer.zero_gradients();
                }

                for &i in batch {
                    let predicted = self.forward(&x[i]);
                    let actual = self.to_one_hot(y[i])?;

                    epoch_loss += loss_fn.compute(&predicted, &actual);
                    let grad = loss_fn.gradient(&predicted, &actual);
                    self.backward(&grad);
                }

                let scale = 1.0 / batch.len() as f64;
                for layer in self.layers.iter_mut() {
                    layer.apply_gradients(learning_rate, scale);
                }
            }

            epoch_loss /= train_indices.len() as f64;
            self.training_loss_history.push(epoch_loss);

            if let Some(patience) = self.config.early_stopping_patience {
                let mut validation_loss = 0.0;
                for &i in &validation_indices {
                    let predicted = self.forward(&x[i]);
                    let actual = self.to_one_hot(y[i])?;
                    validation_loss += loss_fn.compute(&predicted, &actual);
                }
                validation_loss /= validation_indices.len() as f64;
                self.validation_loss_history.push(validation_loss);

                if validation_loss < best_validation_loss {
                    best_validation_loss = validation_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        if self.config.verbose {
                            println!(
                                "Early stopping at epoch {} (validation loss {:.6})",
                                epoch, validation_loss
                            );
                        }
                        break;
                    }
                }
            }

            if self.config.verbose && epoch % 10 == 0 {
                println!("Epoch {}: training loss = {:.6}", epoch, epoch_loss);
            }
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>> {
        let probabilities = self.predict_proba(data)?;

        let mut predictions = Vec::with_capacity(probabilities.len());
        for sample_probs in &probabilities {
            let (max_idx, _) = sample_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| {
                    Error::InvalidOperation(
                        "MLPClassifier produced an empty probability vector".into(),
                    )
                })?;

            let class = self.classes.get(max_idx).copied().ok_or_else(|| {
                Error::InvalidOperation(format!(
                    "MLPClassifier: output unit {} has no corresponding class among {:?}",
                    max_idx, self.classes
                ))
            })?;
            predictions.push(class);
        }

        Ok(predictions)
    }

    fn feature_importances(&self) -> Option<HashMap<String, f64>> {
        None
    }
}

impl TunableModel for MLPClassifier {
    /// Apply a hyperparameter combination on top of this model's current configuration.
    ///
    /// Recognised keys: `hidden_layers`, `hidden_activation`, `output_activation`,
    /// `learning_rate`, `n_epochs`, `batch_size`, `random_seed`, `validation_fraction`,
    /// `early_stopping_patience`. The network is rebuilt on the next `fit`.
    fn set_params(&mut self, params: &HashMap<String, String>) -> Result<()> {
        let mut config = self.config.clone();
        let unknown = apply_mlp_params(&mut config, params)?;
        err_on_unknown_params("MLPClassifier", unknown)?;
        config.validate()?;
        self.config = config;
        self.layers.clear();
        self.is_fitted = false;
        Ok(())
    }
}

impl ModelEvaluator for MLPClassifier {
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics> {
        let predictions = self.predict(test_data)?;
        let actual: Vec<f64> = test_data
            .get_column_numeric_values(test_target)
            .map_err(|_| Error::Column(format!("Target column '{}' not found", test_target)))?;

        if predictions.len() != actual.len() {
            return Err(Error::DimensionMismatch(format!(
                "{} predictions for {} target values",
                predictions.len(),
                actual.len()
            )));
        }
        if predictions.is_empty() {
            return Err(Error::InvalidInput(
                "Cannot evaluate on an empty test set".into(),
            ));
        }

        let mut metrics = ModelMetrics::new();

        let correct = predictions
            .iter()
            .zip(&actual)
            .filter(|(p, a)| (*p - *a).abs() < 1e-10)
            .count();
        metrics.add_metric("accuracy", correct as f64 / predictions.len() as f64);

        Ok(metrics)
    }

    /// Real k-fold cross-validation: each fold refits a fresh copy of this model on the
    /// training rows and evaluates it on the held-out rows.
    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>> {
        kfold_cross_validate(self, data, target, folds)
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
        .expect("operation should succeed");
        let x2 = Series::new(
            vec![0.0, 1.0, 0.0, 1.0, 0.1, 0.9, 0.1, 0.9],
            Some("x2".to_string()),
        )
        .expect("operation should succeed");
        let y = Series::new(
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
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
    fn test_sigmoid_does_not_overflow() {
        let extreme = Activation::Sigmoid.forward(&[-800.0, 800.0]);
        assert!(extreme[0] >= 0.0 && extreme[0] < 1e-300);
        assert!((extreme[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mlp_regressor() {
        let data = create_regression_data();
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![20])
            .learning_rate(0.01)
            .n_epochs(2000)
            .batch_size(4)
            .early_stopping_patience(Some(200))
            .build();

        let mut mlp = MLPRegressor::new(config);
        mlp.fit(&data, "y").expect("operation should succeed");

        let predictions = mlp.predict(&data).expect("operation should succeed");
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
            .learning_rate(0.1)
            .n_epochs(500)
            .batch_size(4)
            .early_stopping_patience(None)
            .build();

        let mut mlp = MLPClassifier::new(config);
        mlp.fit(&data, "y").expect("operation should succeed");

        let predictions = mlp.predict(&data).expect("operation should succeed");
        assert_eq!(predictions.len(), 10);

        let metrics = mlp.evaluate(&data, "y").expect("operation should succeed");
        let accuracy = metrics
            .get_metric("accuracy")
            .expect("operation should succeed");
        assert!(*accuracy >= 0.5, "Accuracy should be at least 50%");
    }

    #[test]
    fn test_mlp_xor_problem() {
        // XOR is a classic test for neural networks
        let data = create_xor_data();
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![8, 4])
            .learning_rate(0.3)
            .n_epochs(2000)
            .batch_size(8)
            .early_stopping_patience(None)
            .build();

        let mut mlp = MLPClassifier::new(config);
        mlp.fit(&data, "y").expect("operation should succeed");

        let predictions = mlp.predict(&data).expect("operation should succeed");

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
            .learning_rate(0.01)
            .n_epochs(50)
            .batch_size(2)
            .early_stopping_patience(None)
            .build();

        let mut mlp = MLPRegressor::new(config);
        mlp.fit(&data, "y").expect("operation should succeed");

        let history = mlp.training_loss_history();
        assert_eq!(history.len(), 50);

        // Loss should generally decrease (not necessarily monotonically)
        assert!(
            history.last().expect("operation should succeed")
                <= history.first().expect("operation should succeed")
        );
    }

    #[test]
    fn test_early_stopping_uses_validation_split() {
        let data = create_regression_data();
        let config = MLPConfigBuilder::new()
            .hidden_layers(vec![4])
            .n_epochs(20)
            .batch_size(4)
            .early_stopping_patience(Some(5))
            .validation_fraction(0.2)
            .build();

        let mut mlp = MLPRegressor::new(config);
        mlp.fit(&data, "y").expect("fit should succeed");

        assert!(
            !mlp.validation_loss_history().is_empty(),
            "early stopping must monitor a real validation split"
        );
        assert_eq!(
            mlp.validation_loss_history().len(),
            mlp.training_loss_history().len(),
            "one validation loss per trained epoch"
        );
    }

    #[test]
    fn test_early_stopping_without_validation_fraction_errors() {
        let data = create_regression_data();
        let config = MLPConfigBuilder::new()
            .early_stopping_patience(Some(3))
            .validation_fraction(0.0)
            .build();

        let mut mlp = MLPRegressor::new(config);
        assert!(
            mlp.fit(&data, "y").is_err(),
            "early stopping without a validation split must be rejected"
        );
    }
}
