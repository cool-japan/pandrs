//! Machine learning models
//!
//! This module provides interfaces and implementations for machine learning models,
//! including regression, classification, and utility functions for model evaluation
//! and cross-validation.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::optimized::OptimizedDataFrame;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::random::SliceRandom;
use std::collections::HashMap;

/// Metrics from model evaluation
#[derive(Debug, Clone)]
pub struct ModelMetrics {
    /// Specific metrics for the model (varies by model type)
    pub metrics: HashMap<String, f64>,
    /// Training time in seconds
    pub training_time: f64,
    /// Prediction time in seconds
    pub prediction_time: Option<f64>,
}

impl ModelMetrics {
    /// Create a new empty ModelMetrics instance
    pub fn new() -> Self {
        ModelMetrics {
            metrics: HashMap::new(),
            training_time: 0.0,
            prediction_time: None,
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }

    /// Get a metric by name
    pub fn get_metric(&self, name: &str) -> Option<&f64> {
        self.metrics.get(name)
    }

    /// Set training time
    pub fn set_training_time(&mut self, time: f64) {
        self.training_time = time;
    }

    /// Set prediction time
    pub fn set_prediction_time(&mut self, time: f64) {
        self.prediction_time = Some(time);
    }
}

/// Trait for evaluating models
pub trait ModelEvaluator {
    /// Evaluate a model using test data
    fn evaluate(&self, test_data: &DataFrame, test_target: &str) -> Result<ModelMetrics>;

    /// Cross-validate a model
    fn cross_validate(
        &self,
        data: &DataFrame,
        target: &str,
        folds: usize,
    ) -> Result<Vec<ModelMetrics>>;
}

/// Trait for supervised machine learning models
pub trait SupervisedModel: ModelEvaluator {
    /// Fit model to training data
    fn fit(&mut self, train_data: &DataFrame, target_column: &str) -> Result<()>;

    /// Predict using the fitted model
    fn predict(&self, data: &DataFrame) -> Result<Vec<f64>>;

    /// Get feature importances (if applicable)
    fn feature_importances(&self) -> Option<HashMap<String, f64>>;
}

/// R², with the same zero-`ss_tot` edge case handling as
/// `metrics::regression::r2_score` (`src/ml/metrics/regression.rs`): when
/// every value in `actual` is identical, `ss_tot` is `0` and the fraction
/// `1.0 - ss_res / ss_tot` is either `NaN` (`0.0/0.0`, exact predictions) or
/// `-inf` (nonzero `ss_res`) rather than a meaningful score. In that case
/// this reports `1.0` for an exact match and `0.0` otherwise — never `1.0`
/// unconditionally regardless of residuals, which silently hid prediction
/// error on constant-target folds/leaves.
pub(crate) fn r2_score_guarded(predictions: &[f64], actual: &[f64]) -> f64 {
    let y_mean = actual.iter().sum::<f64>() / actual.len() as f64;
    let ss_tot: f64 = actual.iter().map(|a| (a - y_mean).powi(2)).sum();
    let ss_res: f64 = predictions
        .iter()
        .zip(actual)
        .map(|(p, a)| (a - p).powi(2))
        .sum();
    if ss_tot == 0.0 {
        if ss_res == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Contiguous-fold k-fold cross-validation, shared by every [`SupervisedModel`]
/// in this module that previously stubbed `cross_validate` out as `Ok(vec![])`.
///
/// Splits `data` into `folds` contiguous blocks (the last block absorbs any
/// remainder). For each fold, clones `model`, fits the clone on the other
/// `folds - 1` blocks, evaluates it on the held-out block, and collects the
/// resulting [`ModelMetrics`]. This is exactly the pattern already used by
/// `LinearRegression::cross_validate` (`src/ml/models/linear.rs`); it is
/// factored out here so every model shares one audited implementation
/// instead of re-deriving (and potentially re-breaking) it per model.
///
/// Folds are contiguous rather than shuffled, matching the pre-existing
/// `LinearRegression` reference implementation this mirrors.
pub(crate) fn contiguous_kfold_cross_validate<M>(
    model: &M,
    data: &DataFrame,
    target: &str,
    folds: usize,
) -> Result<Vec<ModelMetrics>>
where
    M: SupervisedModel + Clone,
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

        let mut fold_model = model.clone();
        fold_model.fit(&train_df, target)?;
        let fold_metrics = fold_model.evaluate(&test_df, target)?;
        all_metrics.push(fold_metrics);
    }

    Ok(all_metrics)
}

/// Trait for unsupervised machine learning models
pub trait UnsupervisedModel: ModelEvaluator {
    /// Fit model to training data
    fn fit(&mut self, data: &DataFrame) -> Result<()>;

    /// Transform data using the fitted model
    fn transform(&self, data: &DataFrame) -> Result<DataFrame>;

    /// Fit and transform in one step
    fn fit_transform(&mut self, data: &DataFrame) -> Result<DataFrame> {
        self.fit(data)?;
        self.transform(data)
    }
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidation {
    /// Number of folds
    pub n_folds: usize,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for CrossValidation {
    fn default() -> Self {
        CrossValidation {
            n_folds: 5,
            shuffle: true,
            random_seed: None,
        }
    }
}

/// Split data into training and test sets
///
/// # Arguments
/// * `data` - DataFrame to split
/// * `test_size` - Fraction of data to use for testing (between 0 and 1)
/// * `shuffle` - Whether to shuffle data before splitting
/// * `random_seed` - Optional random seed for reproducibility
///
/// # Returns
/// * Tuple of (train_data, test_data)
pub fn train_test_split(
    data: &DataFrame,
    test_size: f64,
    shuffle: bool,
    random_seed: Option<u64>,
) -> Result<(DataFrame, DataFrame)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(Error::InvalidInput(
            "test_size must be between 0 and 1".into(),
        ));
    }

    let n_rows = data.nrows();
    let n_test = (n_rows as f64 * test_size).round() as usize;

    if n_test == 0 || n_test == n_rows {
        return Err(Error::InvalidInput(format!(
            "test_size {} would result in empty training or test set",
            test_size
        )));
    }

    // `shuffle=false` keeps the original sequential split (an explicit,
    // honest request for positional train/test blocks -- e.g. time series
    // callers rely on this). `shuffle=true` now really shuffles: previously
    // both `shuffle` and `random_seed` were accepted and silently ignored,
    // always taking the same sequential prefix/suffix split regardless of
    // what the caller asked for.
    let (train_indices, test_indices): (Vec<usize>, Vec<usize>) = if shuffle {
        let mut indices: Vec<usize> = (0..n_rows).collect();
        indices.shuffle(&mut seeded_rng(random_seed));
        let test_indices = indices[..n_test].to_vec();
        let train_indices = indices[n_test..].to_vec();
        (train_indices, test_indices)
    } else {
        (
            (0..(n_rows - n_test)).collect(),
            ((n_rows - n_test)..n_rows).collect(),
        )
    };

    let train_data = data.sample(&train_indices)?;
    let test_data = data.sample(&test_indices)?;

    Ok((train_data, test_data))
}

/// Split OptimizedDataFrame into training and test sets
///
/// # Arguments
/// * `data` - OptimizedDataFrame to split
/// * `test_size` - Fraction of data to use for testing (between 0 and 1)
/// * `random_seed` - Optional random seed for reproducibility
///
/// # Returns
/// * Tuple of (train_data, test_data)
pub fn train_test_split_opt(
    data: &OptimizedDataFrame,
    test_size: f64,
    random_seed: Option<u64>,
) -> Result<(OptimizedDataFrame, OptimizedDataFrame)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(Error::InvalidInput(
            "test_size must be between 0 and 1".into(),
        ));
    }

    let n_rows = data.row_count();
    let n_test = (n_rows as f64 * test_size).round() as usize;

    if n_test == 0 || n_test == n_rows {
        return Err(Error::InvalidInput(format!(
            "test_size {} would result in empty training or test set",
            test_size
        )));
    }

    // Unlike `train_test_split`, this entry point has no `shuffle` flag --
    // taking a `random_seed` parameter at all only makes sense if the split
    // is randomized, so this always shuffles (seeded when `random_seed` is
    // given, from system entropy otherwise). Previously `random_seed` was
    // accepted and silently ignored in favor of a fixed sequential split.
    let mut indices: Vec<usize> = (0..n_rows).collect();
    indices.shuffle(&mut seeded_rng(random_seed));
    let test_indices = indices[..n_test].to_vec();
    let train_indices = indices[n_test..].to_vec();

    let train_data = data.sample_rows(&train_indices)?;
    let test_data = data.sample_rows(&test_indices)?;

    Ok((train_data, test_data))
}

/// Build a seeded RNG when `seed` is given, otherwise one seeded from the
/// system entropy source (same pattern as `models::tree::seeded_rng`,
/// `stats::sampling::seeded_rng`, and
/// `optimized::split_dataframe::row_ops::sample_rows`).
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

pub mod ensemble;
pub mod evaluation;
pub mod linear;
pub mod neural;
pub mod selection;
pub mod tree;

// Re-export commonly used model types and functions
pub use ensemble::{
    GradientBoostingClassifier, GradientBoostingConfig, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestConfig, RandomForestRegressor,
};
pub use evaluation::{cross_val_score, learning_curve, validation_curve};
pub use linear::{LinearRegression, LogisticRegression};
pub use neural::{
    Activation, LossFunction, MLPClassifier, MLPConfig, MLPConfigBuilder, MLPRegressor,
};
pub use selection::{GridSearchCV, HyperparameterGrid, RandomizedSearchCV};
pub use tree::{DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeRegressor, SplitCriterion};
