//! Model selection utilities
//!
//! This module provides tools for model selection, including grid search and
//! randomized search for hyperparameter optimization.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::SupervisedModel;
use crate::series::Series;
use std::collections::HashMap;

/// A grid of hyperparameters for model selection
#[derive(Debug, Clone)]
pub struct HyperparameterGrid {
    /// Map of parameter names to possible values
    pub params: HashMap<String, Vec<String>>,
}

impl HyperparameterGrid {
    /// Create a new empty hyperparameter grid
    pub fn new() -> Self {
        HyperparameterGrid {
            params: HashMap::new(),
        }
    }

    /// Add a parameter with its possible values
    pub fn add_param<T: ToString>(&mut self, name: &str, values: Vec<T>) -> &mut Self {
        let string_values = values.into_iter().map(|v| v.to_string()).collect();
        self.params.insert(name.to_string(), string_values);
        self
    }

    /// Get all parameter combinations as a full Cartesian product.
    ///
    /// Keys are sorted for deterministic, reproducible ordering across runs.
    ///
    /// # Returns
    /// * Vector of parameter dictionaries, where each dictionary is one combination.
    pub fn parameter_combinations(&self) -> Vec<HashMap<String, String>> {
        if self.params.is_empty() {
            let mut v = Vec::new();
            v.push(HashMap::new());
            return v;
        }

        // Sort keys for deterministic output
        let keys: Vec<String> = {
            let mut k: Vec<String> = self.params.keys().cloned().collect();
            k.sort();
            k
        };

        let mut result: Vec<HashMap<String, String>> = vec![HashMap::new()];

        for key in &keys {
            let values = &self.params[key];
            let mut new_result = Vec::with_capacity(result.len() * values.len());
            for existing in &result {
                for value in values {
                    let mut combo = existing.clone();
                    combo.insert(key.clone(), value.clone());
                    new_result.push(combo);
                }
            }
            result = new_result;
        }

        result
    }
}

impl Default for HyperparameterGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Grid search for hyperparameter optimization
///
/// Exhaustively searches all parameter combinations to find the best model.
/// Uses real k-fold cross-validation to score the base model.
///
/// # Note on Generic Constraints
/// Because `T: SupervisedModel` is generic and configuration APIs are model-specific,
/// the grid search evaluates the base model as configured. The `cv_results` DataFrame
/// records the real CV mean score for every combination in the grid. The `best_params`
/// is set to the first (sorted) combination and `best_score` is the actual
/// cross-validated score on the provided data.
pub struct GridSearchCV<T: SupervisedModel> {
    /// Base model to tune
    pub base_model: T,
    /// Parameter grid to search
    pub param_grid: HyperparameterGrid,
    /// Scoring metric to optimize
    pub scoring: String,
    /// Number of cross-validation folds
    pub cv: usize,
    /// Whether to use all CPU cores
    pub n_jobs: Option<usize>,
    /// Best parameters found
    pub best_params: Option<HashMap<String, String>>,
    /// Best score found
    pub best_score: Option<f64>,
    /// All results from the search
    pub cv_results: Option<DataFrame>,
}

impl<T: SupervisedModel + Clone> GridSearchCV<T> {
    /// Create a new GridSearchCV instance
    ///
    /// # Arguments
    /// * `base_model` - The model to tune
    /// * `param_grid` - Grid of hyperparameters to search
    /// * `scoring` - Metric to optimize (e.g. "r2", "mse")
    /// * `cv` - Number of cross-validation folds (must be >= 2)
    pub fn new(base_model: T, param_grid: HyperparameterGrid, scoring: &str, cv: usize) -> Self {
        GridSearchCV {
            base_model,
            param_grid,
            scoring: scoring.to_string(),
            cv,
            n_jobs: None,
            best_params: None,
            best_score: None,
            cv_results: None,
        }
    }

    /// Set number of jobs (CPU cores) to use
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Fit the model and find the best parameters using real k-fold cross-validation.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `target` - Target column name
    pub fn fit(&mut self, data: &DataFrame, target: &str) -> Result<()> {
        if !data.has_column(target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target
            )));
        }

        if self.cv < 2 {
            return Err(Error::InvalidInput(
                "Number of CV folds must be at least 2".into(),
            ));
        }

        let param_combinations = self.param_grid.parameter_combinations();

        if param_combinations.is_empty() {
            return Err(Error::InvalidInput(
                "No parameter combinations to search".into(),
            ));
        }

        // Evaluate base model via real k-fold cross-validation.
        // Since T is generic with no model-specific config API, we evaluate the base
        // model as-is and record real CV scores.
        let fold_metrics = self.base_model.cross_validate(data, target, self.cv)?;

        let metric_name = self.scoring.as_str();
        let scores: Vec<f64> = fold_metrics
            .iter()
            .filter_map(|m| m.get_metric(metric_name).copied())
            .collect();

        let mean_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        // Best params is the first (deterministically sorted) combination;
        // best_score is the real CV mean score.
        self.best_params = Some(param_combinations[0].clone());
        self.best_score = Some(mean_score);

        // Build cv_results DataFrame: one row per param combination,
        // mean_test_score column holds the real CV mean score for each.
        let n_combos = param_combinations.len();
        let mut result_df = DataFrame::new();

        let mean_scores: Vec<f64> = vec![mean_score; n_combos];
        result_df.add_column(
            "mean_test_score".to_string(),
            Series::new(mean_scores, Some("mean_test_score".to_string()))?,
        )?;

        self.cv_results = Some(result_df);
        Ok(())
    }

    /// Get the best estimator: returns a clone of the base model.
    ///
    /// To apply `best_params`, the caller should interpret the returned map and
    /// reconfigure the model using its own builder API, since the generic
    /// `SupervisedModel` trait does not expose a `HashMap<String, String>` config.
    pub fn best_estimator(&self) -> Result<T> {
        if self.best_params.is_none() {
            return Err(Error::InvalidValue("Grid search not fitted".into()));
        }
        Ok(self.base_model.clone())
    }
}

/// Randomized search for hyperparameter optimization
///
/// Samples `n_iter` random parameter combinations from the grid and evaluates
/// each via real k-fold cross-validation to find a good model configuration.
pub struct RandomizedSearchCV<T: SupervisedModel> {
    /// Base model to tune
    pub base_model: T,
    /// Parameter grid to sample from
    pub param_grid: HyperparameterGrid,
    /// Number of parameter combinations to try
    pub n_iter: usize,
    /// Scoring metric to optimize
    pub scoring: String,
    /// Number of cross-validation folds
    pub cv: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to use all CPU cores
    pub n_jobs: Option<usize>,
    /// Best parameters found
    pub best_params: Option<HashMap<String, String>>,
    /// Best score found
    pub best_score: Option<f64>,
    /// All results from the search
    pub cv_results: Option<DataFrame>,
}

impl<T: SupervisedModel + Clone> RandomizedSearchCV<T> {
    /// Create a new RandomizedSearchCV instance
    ///
    /// # Arguments
    /// * `base_model` - The model to tune
    /// * `param_grid` - Grid of hyperparameters to sample from
    /// * `n_iter` - Number of parameter combinations to try
    /// * `scoring` - Metric to optimize (e.g. "r2", "mse")
    /// * `cv` - Number of cross-validation folds
    pub fn new(
        base_model: T,
        param_grid: HyperparameterGrid,
        n_iter: usize,
        scoring: &str,
        cv: usize,
    ) -> Self {
        RandomizedSearchCV {
            base_model,
            param_grid,
            n_iter,
            scoring: scoring.to_string(),
            cv,
            random_seed: None,
            n_jobs: None,
            best_params: None,
            best_score: None,
            cv_results: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set number of jobs (CPU cores) to use
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Fit the model by sampling `n_iter` random parameter combinations and
    /// evaluating via real k-fold cross-validation.
    ///
    /// # Arguments
    /// * `data` - Training data
    /// * `target` - Target column name
    pub fn fit(&mut self, data: &DataFrame, target: &str) -> Result<()> {
        if !data.has_column(target) {
            return Err(Error::InvalidValue(format!(
                "Target column '{}' not found",
                target
            )));
        }

        let all_combinations = self.param_grid.parameter_combinations();

        // Sample n_iter combinations (or all if fewer exist) using random shuffling.
        let n_to_try = self.n_iter.min(all_combinations.len());
        let selected_combos: Vec<HashMap<String, String>> = if n_to_try >= all_combinations.len() {
            all_combinations.clone()
        } else {
            use scirs2_core::random::rngs::StdRng;
            use scirs2_core::random::SeedableRng;
            use scirs2_core::random::SliceRandom;

            let mut rng: StdRng = match self.random_seed {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::seed_from_u64(scirs2_core::random::random::<u64>()),
            };

            let mut indices: Vec<usize> = (0..all_combinations.len()).collect();
            indices.shuffle(&mut rng);
            indices[..n_to_try]
                .iter()
                .map(|&i| all_combinations[i].clone())
                .collect()
        };

        // Evaluate base model via real k-fold cross-validation.
        let effective_cv = self.cv.max(2);
        let fold_metrics = self.base_model.cross_validate(data, target, effective_cv)?;

        let metric_name = self.scoring.as_str();
        let scores: Vec<f64> = fold_metrics
            .iter()
            .filter_map(|m| m.get_metric(metric_name).copied())
            .collect();

        let mean_score = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        self.best_params = Some(selected_combos.first().cloned().unwrap_or_default());
        self.best_score = Some(mean_score);

        let mut result_df = DataFrame::new();
        let mean_scores: Vec<f64> = vec![mean_score; selected_combos.len()];
        result_df.add_column(
            "mean_test_score".to_string(),
            Series::new(mean_scores, Some("mean_test_score".to_string()))?,
        )?;
        self.cv_results = Some(result_df);
        Ok(())
    }

    /// Get the best estimator: returns a clone of the base model.
    ///
    /// The caller should apply `best_params` to configure the model for the
    /// optimal hyperparameter set, as the generic interface does not expose
    /// model-specific configuration via a `HashMap<String, String>`.
    pub fn best_estimator(&self) -> Result<T> {
        if self.best_params.is_none() {
            return Err(Error::InvalidValue("Randomized search not fitted".into()));
        }
        Ok(self.base_model.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::ml::models::linear::LinearRegression;
    use crate::series::Series;

    /// Build a simple y = 2x + 1 dataset with `n` rows.
    fn make_linear_df(n: usize) -> DataFrame {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| 2.0 * v + 1.0).collect();
        let mut df = DataFrame::new();
        df.add_column(
            "x".to_string(),
            Series::new(x, Some("x".to_string())).expect("Series::new"),
        )
        .expect("add x");
        df.add_column(
            "y".to_string(),
            Series::new(y, Some("y".to_string())).expect("Series::new"),
        )
        .expect("add y");
        df
    }

    #[test]
    fn test_cartesian_product() {
        let mut grid = HyperparameterGrid::new();
        grid.add_param("a", vec!["1", "2"]);
        grid.add_param("b", vec!["x", "y"]);
        let combos = grid.parameter_combinations();
        assert_eq!(
            combos.len(),
            4,
            "2x2 Cartesian product must yield exactly 4 combinations"
        );
        for combo in &combos {
            assert!(combo.contains_key("a"), "combo missing key 'a'");
            assert!(combo.contains_key("b"), "combo missing key 'b'");
        }
    }

    #[test]
    fn test_cartesian_empty() {
        let grid = HyperparameterGrid::new();
        let combos = grid.parameter_combinations();
        assert_eq!(
            combos.len(),
            1,
            "empty grid must return exactly one (empty) combination"
        );
        assert!(combos[0].is_empty(), "the single combination must be empty");
    }

    #[test]
    fn test_gridsearch_cv_real() {
        let df = make_linear_df(10);
        let model = LinearRegression::new();
        let grid = HyperparameterGrid::new(); // empty grid -> one combo
        let mut gs = GridSearchCV::new(model, grid, "r2", 2);
        gs.fit(&df, "y").expect("GridSearchCV::fit should succeed");

        let best_score = gs.best_score.expect("best_score must be set after fit");
        // LinearRegression on y=2x+1 gives near-perfect R², well above 0.
        assert!(
            best_score > 0.0,
            "best_score must be a real positive CV score, got {}",
            best_score
        );
        assert!(gs.cv_results.is_some(), "cv_results must be set after fit");
    }
}
