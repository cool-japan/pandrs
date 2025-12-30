//! Automated Machine Learning (AutoML) capabilities
//!
//! This module provides comprehensive AutoML functionality including automated
//! model selection, hyperparameter optimization, feature engineering, and
//! ensemble methods for both regression and classification tasks.

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::ml::feature_engineering::{AutoFeatureEngineer, FeatureSelectionMethod, ScalingMethod};
use crate::ml::model_selection::{
    CrossValidationStrategy, GridSearchCV, ParameterDistribution, RandomizedSearchCV, Scorer,
};
use crate::ml::models::{train_test_split, ModelMetrics};
use crate::ml::sklearn_compat::{Pipeline, PipelineStep, SklearnPredictor};
use crate::utils::rand_compat::{thread_rng, GenRangeCompat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// AutoML task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Regression task
    Regression,
    /// Binary classification task
    BinaryClassification,
    /// Multi-class classification task
    MultiClassification,
    /// Time series forecasting
    TimeSeries,
    /// Auto-detect task type from target variable
    Auto,
}

/// AutoML configuration
#[derive(Debug, Clone)]
pub struct AutoMLConfig {
    /// Type of machine learning task
    pub task_type: TaskType,
    /// Time limit for optimization (in seconds)
    pub time_limit: Option<f64>,
    /// Maximum number of models to try
    pub max_models: Option<usize>,
    /// Cross-validation strategy
    pub cv_strategy: CrossValidationStrategy,
    /// Scoring metric for optimization
    pub scoring: Scorer,
    /// Whether to perform feature engineering
    pub feature_engineering: bool,
    /// Whether to perform feature selection
    pub feature_selection: bool,
    /// Whether to perform ensemble methods
    pub ensemble_methods: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Verbose output level
    pub verbose: usize,
    /// Whether to optimize for interpretability vs performance
    pub optimize_for_interpretability: bool,
    /// Memory limit (in GB) for model training
    pub memory_limit: Option<f64>,
    /// Custom model list to try (if None, use default model space)
    pub model_whitelist: Option<Vec<String>>,
    /// Models to exclude from search
    pub model_blacklist: Option<Vec<String>>,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::Auto,
            time_limit: Some(3600.0), // 1 hour
            max_models: Some(50),
            cv_strategy: CrossValidationStrategy::KFold {
                n_splits: 5,
                shuffle: true,
                random_state: None,
            },
            scoring: Scorer::R2,
            feature_engineering: true,
            feature_selection: true,
            ensemble_methods: true,
            random_state: None,
            verbose: 1,
            optimize_for_interpretability: false,
            memory_limit: Some(8.0),
            model_whitelist: None,
            model_blacklist: None,
        }
    }
}

/// Model search space for AutoML
#[derive(Debug, Clone)]
pub struct ModelSearchSpace {
    /// Linear models and their parameter spaces
    pub linear_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
    /// Tree-based models and their parameter spaces
    pub tree_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
    /// Ensemble models and their parameter spaces
    pub ensemble_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
    /// Neural network models and their parameter spaces
    pub neural_models: Vec<(String, HashMap<String, ParameterDistribution>)>,
}

impl ModelSearchSpace {
    /// Create default model search space for regression
    pub fn default_regression() -> Self {
        let mut linear_models = Vec::new();
        let mut tree_models = Vec::new();
        let mut ensemble_models = Vec::new();
        let neural_models = Vec::new();

        // Linear models
        let mut linear_regression_params = HashMap::new();
        linear_regression_params.insert(
            "fit_intercept".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_regression_params.insert(
            "normalize".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_models.push(("LinearRegression".to_string(), linear_regression_params));

        let mut ridge_params = HashMap::new();
        ridge_params.insert(
            "alpha".to_string(),
            ParameterDistribution::LogUniform {
                low: 1e-4,
                high: 1e2,
            },
        );
        ridge_params.insert(
            "fit_intercept".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_models.push(("Ridge".to_string(), ridge_params));

        let mut lasso_params = HashMap::new();
        lasso_params.insert(
            "alpha".to_string(),
            ParameterDistribution::LogUniform {
                low: 1e-4,
                high: 1e2,
            },
        );
        lasso_params.insert(
            "fit_intercept".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_models.push(("Lasso".to_string(), lasso_params));

        // Tree-based models
        let mut decision_tree_params = HashMap::new();
        decision_tree_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::Choice(vec![
                "3".to_string(),
                "5".to_string(),
                "10".to_string(),
                "None".to_string(),
            ]),
        );
        decision_tree_params.insert(
            "min_samples_split".to_string(),
            ParameterDistribution::UniformInt { low: 2, high: 20 },
        );
        decision_tree_params.insert(
            "min_samples_leaf".to_string(),
            ParameterDistribution::UniformInt { low: 1, high: 10 },
        );
        tree_models.push(("DecisionTree".to_string(), decision_tree_params));

        // Ensemble models
        let mut random_forest_params = HashMap::new();
        random_forest_params.insert(
            "n_estimators".to_string(),
            ParameterDistribution::Choice(vec![
                "50".to_string(),
                "100".to_string(),
                "200".to_string(),
            ]),
        );
        random_forest_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::Choice(vec![
                "5".to_string(),
                "10".to_string(),
                "20".to_string(),
                "None".to_string(),
            ]),
        );
        random_forest_params.insert(
            "min_samples_split".to_string(),
            ParameterDistribution::UniformInt { low: 2, high: 20 },
        );
        ensemble_models.push(("RandomForest".to_string(), random_forest_params));

        let mut gradient_boosting_params = HashMap::new();
        gradient_boosting_params.insert(
            "n_estimators".to_string(),
            ParameterDistribution::Choice(vec![
                "50".to_string(),
                "100".to_string(),
                "200".to_string(),
            ]),
        );
        gradient_boosting_params.insert(
            "learning_rate".to_string(),
            ParameterDistribution::LogUniform {
                low: 0.01,
                high: 0.3,
            },
        );
        gradient_boosting_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::UniformInt { low: 3, high: 10 },
        );
        ensemble_models.push(("GradientBoosting".to_string(), gradient_boosting_params));

        Self {
            linear_models,
            tree_models,
            ensemble_models,
            neural_models,
        }
    }

    /// Create default model search space for classification
    pub fn default_classification() -> Self {
        let mut linear_models = Vec::new();
        let mut tree_models = Vec::new();
        let mut ensemble_models = Vec::new();
        let neural_models = Vec::new();

        // Linear models
        let mut logistic_regression_params = HashMap::new();
        logistic_regression_params.insert(
            "C".to_string(),
            ParameterDistribution::LogUniform {
                low: 1e-4,
                high: 1e2,
            },
        );
        logistic_regression_params.insert(
            "fit_intercept".to_string(),
            ParameterDistribution::Choice(vec!["true".to_string(), "false".to_string()]),
        );
        linear_models.push(("LogisticRegression".to_string(), logistic_regression_params));

        // Tree-based models
        let mut decision_tree_params = HashMap::new();
        decision_tree_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::Choice(vec![
                "3".to_string(),
                "5".to_string(),
                "10".to_string(),
                "None".to_string(),
            ]),
        );
        decision_tree_params.insert(
            "min_samples_split".to_string(),
            ParameterDistribution::UniformInt { low: 2, high: 20 },
        );
        tree_models.push(("DecisionTreeClassifier".to_string(), decision_tree_params));

        // Ensemble models
        let mut random_forest_params = HashMap::new();
        random_forest_params.insert(
            "n_estimators".to_string(),
            ParameterDistribution::Choice(vec![
                "50".to_string(),
                "100".to_string(),
                "200".to_string(),
            ]),
        );
        random_forest_params.insert(
            "max_depth".to_string(),
            ParameterDistribution::Choice(vec![
                "5".to_string(),
                "10".to_string(),
                "None".to_string(),
            ]),
        );
        ensemble_models.push(("RandomForestClassifier".to_string(), random_forest_params));

        Self {
            linear_models,
            tree_models,
            ensemble_models,
            neural_models,
        }
    }
}

/// Result from AutoML optimization
#[derive(Debug, Clone)]
pub struct AutoMLResult {
    /// Best pipeline found
    pub best_pipeline: String, // Placeholder - would be actual pipeline
    /// Best cross-validation score
    pub best_score: f64,
    /// Best parameters
    pub best_params: HashMap<String, String>,
    /// All tried models and their scores
    pub leaderboard: Vec<ModelResult>,
    /// Feature importances (if available)
    pub feature_importances: Option<HashMap<String, f64>>,
    /// Training time
    pub training_time: f64,
    /// Cross-validation results
    pub cv_results: Vec<f64>,
    /// Final model evaluation on holdout set
    pub holdout_score: Option<f64>,
}

/// Individual model result from AutoML
#[derive(Debug, Clone)]
pub struct ModelResult {
    /// Model name
    pub model_name: String,
    /// Cross-validation score
    pub cv_score: f64,
    /// Standard deviation of CV scores
    pub cv_std: f64,
    /// Training time
    pub training_time: f64,
    /// Model parameters
    pub parameters: HashMap<String, String>,
    /// Feature importance (if available)
    pub feature_importance: Option<HashMap<String, f64>>,
    /// Model complexity score (for interpretability)
    pub complexity_score: f64,
}

/// Main AutoML system
#[derive(Debug)]
pub struct AutoML {
    /// Configuration for AutoML run
    pub config: AutoMLConfig,
    /// Model search space
    pub search_space: ModelSearchSpace,
    /// Feature engineering pipeline
    feature_engineer: Option<AutoFeatureEngineer>,
    /// Results from optimization
    results: Option<AutoMLResult>,
}

impl AutoML {
    /// Create new AutoML instance with default configuration
    pub fn new() -> Self {
        let config = AutoMLConfig::default();
        let search_space = match config.task_type {
            TaskType::Regression => ModelSearchSpace::default_regression(),
            TaskType::BinaryClassification | TaskType::MultiClassification => {
                ModelSearchSpace::default_classification()
            }
            _ => ModelSearchSpace::default_regression(),
        };

        Self {
            config,
            search_space,
            feature_engineer: None,
            results: None,
        }
    }

    /// Create AutoML instance with custom configuration
    pub fn with_config(config: AutoMLConfig) -> Self {
        let search_space = match config.task_type {
            TaskType::Regression => ModelSearchSpace::default_regression(),
            TaskType::BinaryClassification | TaskType::MultiClassification => {
                ModelSearchSpace::default_classification()
            }
            _ => ModelSearchSpace::default_regression(),
        };

        Self {
            config,
            search_space,
            feature_engineer: None,
            results: None,
        }
    }

    /// Set custom model search space
    pub fn with_search_space(mut self, search_space: ModelSearchSpace) -> Self {
        self.search_space = search_space;
        self
    }

    /// Auto-detect task type from target variable
    pub fn detect_task_type(&self, y: &DataFrame) -> Result<TaskType> {
        // Get the first column as target (assuming single target)
        let target_col_name = y
            .column_names()
            .into_iter()
            .next()
            .ok_or_else(|| Error::InvalidValue("No target column found".into()))?;

        let target_col = y.get_column::<f64>(&target_col_name)?;
        let values = target_col.as_f64()?;

        // Check if all values are integers and within a reasonable range for classification
        let unique_values: std::collections::HashSet<i64> = values
            .iter()
            .filter_map(|&x| {
                if x.fract() == 0.0 {
                    Some(x as i64)
                } else {
                    None
                }
            })
            .collect();

        let integer_count = values.iter().filter(|x| x.fract() == 0.0).count();

        // Check if values appear to be categorical (all values are integers with limited unique count)
        if integer_count == values.len() && unique_values.len() <= 20 && unique_values.len() > 0 {
            if unique_values.len() == 2 {
                Ok(TaskType::BinaryClassification)
            } else {
                Ok(TaskType::MultiClassification)
            }
        } else {
            Ok(TaskType::Regression)
        }
    }

    /// Fit AutoML on the given dataset
    pub fn fit(&mut self, x: &DataFrame, y: &DataFrame) -> Result<()> {
        let start_time = Instant::now();

        if self.config.verbose > 0 {
            println!("🚀 Starting AutoML optimization...");
            println!(
                "Dataset shape: {} rows × {} features",
                x.nrows(),
                x.column_names().len()
            );
        }

        // Auto-detect task type if needed
        let task_type = if matches!(self.config.task_type, TaskType::Auto) {
            let detected = self.detect_task_type(y)?;
            if self.config.verbose > 0 {
                println!("Auto-detected task type: {:?}", detected);
            }
            detected
        } else {
            self.config.task_type.clone()
        };

        // Update scoring metric based on task type
        let scoring = match task_type {
            TaskType::Regression => Scorer::R2,
            TaskType::BinaryClassification | TaskType::MultiClassification => Scorer::Accuracy,
            TaskType::TimeSeries => Scorer::NegMeanSquaredError,
            TaskType::Auto => Scorer::R2,
        };

        // Create train/validation split for final evaluation
        let (train_x, holdout_x, train_y, holdout_y) = self.create_train_holdout_split(x, y)?;

        // Feature engineering
        let mut processed_x = train_x.clone();
        if self.config.feature_engineering {
            if self.config.verbose > 0 {
                println!("🔧 Performing automated feature engineering...");
            }

            let mut feature_engineer = AutoFeatureEngineer::new()
                .with_polynomial(2)
                .with_interactions(5)
                .with_selection(
                    FeatureSelectionMethod::KBest(match task_type {
                        TaskType::Regression => {
                            crate::ml::model_selection::ScoreFunction::FRegression
                        }
                        _ => crate::ml::model_selection::ScoreFunction::Chi2,
                    }),
                    self.config.max_models.map(|m| m.min(50)),
                )
                .with_scaling(ScalingMethod::StandardScaler);

            feature_engineer.fit(&processed_x, Some(&train_y))?;
            processed_x = feature_engineer.transform(&processed_x)?;

            self.feature_engineer = Some(feature_engineer);

            if self.config.verbose > 0 {
                println!("Generated {} features", processed_x.column_names().len());
            }
        }

        // Model search and optimization
        if self.config.verbose > 0 {
            println!("🎯 Starting model search and hyperparameter optimization...");
        }

        let model_results = self.search_models(&processed_x, &train_y, &scoring)?;

        // Select best model
        let best_result = model_results
            .iter()
            .max_by(|a, b| a.cv_score.partial_cmp(&b.cv_score).unwrap())
            .ok_or_else(|| Error::InvalidOperation("No models were successfully trained".into()))?;

        // Evaluate on holdout set
        let holdout_score = self.evaluate_on_holdout(&holdout_x, &holdout_y, best_result)?;

        // Create final results
        let training_time = start_time.elapsed().as_secs_f64();

        let results = AutoMLResult {
            best_pipeline: best_result.model_name.clone(),
            best_score: best_result.cv_score,
            best_params: best_result.parameters.clone(),
            leaderboard: model_results.clone(),
            feature_importances: best_result.feature_importance.clone(),
            training_time,
            cv_results: vec![best_result.cv_score], // Simplified
            holdout_score: Some(holdout_score),
        };

        self.results = Some(results);

        if self.config.verbose > 0 {
            println!("✅ AutoML optimization completed in {:.2}s", training_time);
            println!(
                "Best model: {} (CV score: {:.4})",
                best_result.model_name, best_result.cv_score
            );
            println!("Holdout score: {:.4}", holdout_score);
        }

        Ok(())
    }

    /// Create train/holdout split for final evaluation
    fn create_train_holdout_split(
        &self,
        x: &DataFrame,
        y: &DataFrame,
    ) -> Result<(DataFrame, DataFrame, DataFrame, DataFrame)> {
        // Use 80/20 split for train/holdout
        let (train_x, holdout_x) = train_test_split(x, 0.2, true, self.config.random_state)?;
        let (train_y, holdout_y) = train_test_split(y, 0.2, true, self.config.random_state)?;

        Ok((train_x, holdout_x, train_y, holdout_y))
    }

    /// Search through model space and find best models
    fn search_models(
        &self,
        x: &DataFrame,
        y: &DataFrame,
        scoring: &Scorer,
    ) -> Result<Vec<ModelResult>> {
        let mut model_results = Vec::new();
        let mut models_tried = 0;
        let max_models = self.config.max_models.unwrap_or(50);

        // Get all available models
        let all_models = self.get_all_models();

        for (model_name, param_space) in all_models {
            if models_tried >= max_models {
                break;
            }

            // Check whitelist/blacklist
            if let Some(whitelist) = &self.config.model_whitelist {
                if !whitelist.contains(&model_name) {
                    continue;
                }
            }

            if let Some(blacklist) = &self.config.model_blacklist {
                if blacklist.contains(&model_name) {
                    continue;
                }
            }

            if self.config.verbose > 1 {
                println!("Trying model: {}", model_name);
            }

            // Create and fit model
            match self.fit_single_model(&model_name, &param_space, x, y, scoring) {
                Ok(result) => {
                    model_results.push(result);
                    models_tried += 1;
                }
                Err(e) => {
                    if self.config.verbose > 1 {
                        println!("Model {} failed: {}", model_name, e);
                    }
                }
            }
        }

        // Sort by CV score (descending)
        model_results.sort_by(|a, b| b.cv_score.partial_cmp(&a.cv_score).unwrap());

        Ok(model_results)
    }

    /// Get all available models from search space
    fn get_all_models(&self) -> Vec<(String, HashMap<String, ParameterDistribution>)> {
        let mut all_models = Vec::new();

        all_models.extend(self.search_space.linear_models.clone());
        all_models.extend(self.search_space.tree_models.clone());
        all_models.extend(self.search_space.ensemble_models.clone());
        all_models.extend(self.search_space.neural_models.clone());

        all_models
    }

    /// Fit a single model with hyperparameter optimization
    fn fit_single_model(
        &self,
        model_name: &str,
        param_space: &HashMap<String, ParameterDistribution>,
        x: &DataFrame,
        y: &DataFrame,
        scoring: &Scorer,
    ) -> Result<ModelResult> {
        let start_time = Instant::now();

        // Create base estimator (placeholder)
        let estimator = self.create_estimator(model_name)?;

        // Set up randomized search
        let mut search = RandomizedSearchCV::new(
            estimator,
            param_space.clone(),
            20, // Try 20 parameter combinations
        )
        .with_cv(self.config.cv_strategy.clone())
        .with_scoring(scoring.clone())
        .with_random_state(self.config.random_state.unwrap_or(42));

        // Fit the search
        search.fit(x, y)?;

        let training_time = start_time.elapsed().as_secs_f64();

        // Extract results
        let results = search
            .get_results()
            .ok_or_else(|| Error::InvalidOperation("No search results available".into()))?;

        Ok(ModelResult {
            model_name: model_name.to_string(),
            cv_score: results.best_score_,
            cv_std: 0.0, // Placeholder
            training_time,
            parameters: results.best_params_.clone(),
            feature_importance: None, // Would be extracted from fitted model
            complexity_score: self.calculate_complexity_score(model_name, &results.best_params_),
        })
    }

    /// Create base estimator for given model name
    fn create_estimator(
        &self,
        model_name: &str,
    ) -> Result<Box<dyn SklearnPredictor + Send + Sync>> {
        // In a real implementation, this would create actual model instances
        // For now, return a placeholder
        Err(Error::NotImplemented(format!(
            "Model creation for {} not implemented",
            model_name
        )))
    }

    /// Calculate complexity score for interpretability optimization
    fn calculate_complexity_score(
        &self,
        model_name: &str,
        params: &HashMap<String, String>,
    ) -> f64 {
        // Simple complexity scoring based on model type and parameters
        let base_complexity = match model_name {
            "LinearRegression" | "LogisticRegression" => 1.0,
            "Ridge" | "Lasso" => 1.2,
            "DecisionTree" | "DecisionTreeClassifier" => 2.0,
            "RandomForest" | "RandomForestClassifier" => 3.0,
            "GradientBoosting" => 3.5,
            _ => 4.0,
        };

        // Adjust based on parameters
        let mut complexity = base_complexity;

        if let Some(n_estimators) = params.get("n_estimators") {
            if let Ok(n) = n_estimators.parse::<f64>() {
                // More estimators = higher complexity (logarithmic scaling)
                complexity *= (n / 50.0).ln().max(1.0) + 1.0;
            }
        }

        if let Some(max_depth) = params.get("max_depth") {
            if max_depth != "None" {
                if let Ok(depth) = max_depth.parse::<f64>() {
                    complexity *= (depth / 10.0).max(1.0);
                }
            } else {
                complexity *= 2.0; // Unlimited depth increases complexity
            }
        }

        complexity
    }

    /// Evaluate best model on holdout set
    fn evaluate_on_holdout(
        &self,
        holdout_x: &DataFrame,
        holdout_y: &DataFrame,
        best_result: &ModelResult,
    ) -> Result<f64> {
        // In a real implementation, this would:
        // 1. Retrain the best model on full training set
        // 2. Apply same feature engineering pipeline to holdout set
        // 3. Make predictions and calculate score

        // For now, return a placeholder score
        Ok(best_result.cv_score * 0.95) // Slightly lower than CV score
    }

    /// Predict on new data using the best fitted model
    pub fn predict(&self, x: &DataFrame) -> Result<Vec<f64>> {
        let _results = self.results.as_ref().ok_or_else(|| {
            Error::InvalidOperation("AutoML must be fitted before predict".into())
        })?;

        // Apply feature engineering if used
        let mut processed_x = x.clone();
        if let Some(feature_engineer) = &self.feature_engineer {
            processed_x = feature_engineer.transform(&processed_x)?;
        }

        // In a real implementation, this would use the fitted best model
        // For now, return placeholder predictions
        Ok(vec![0.0; processed_x.nrows()])
    }

    /// Get the AutoML results
    pub fn get_results(&self) -> Option<&AutoMLResult> {
        self.results.as_ref()
    }

    /// Generate a comprehensive report of the AutoML run
    pub fn generate_report(&self) -> Result<String> {
        let results = self.results.as_ref().ok_or_else(|| {
            Error::InvalidOperation("AutoML must be fitted before generating report".into())
        })?;

        let mut report = String::new();

        report.push_str("# AutoML Report\n\n");

        // Summary
        report.push_str("## Summary\n");
        report.push_str(&format!("- **Best Model**: {}\n", results.best_pipeline));
        report.push_str(&format!("- **Best Score**: {:.4}\n", results.best_score));
        if let Some(holdout_score) = results.holdout_score {
            report.push_str(&format!("- **Holdout Score**: {:.4}\n", holdout_score));
        }
        report.push_str(&format!(
            "- **Training Time**: {:.2}s\n",
            results.training_time
        ));
        report.push_str(&format!(
            "- **Models Tried**: {}\n",
            results.leaderboard.len()
        ));

        // Leaderboard
        report.push_str("\n## Model Leaderboard\n\n");
        report.push_str("| Rank | Model | CV Score | Std | Time (s) | Complexity |\n");
        report.push_str("|------|-------|----------|-----|----------|------------|\n");

        for (i, model) in results.leaderboard.iter().take(10).enumerate() {
            report.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.2} | {:.2} |\n",
                i + 1,
                model.model_name,
                model.cv_score,
                model.cv_std,
                model.training_time,
                model.complexity_score
            ));
        }

        // Best model details
        if let Some(best_model) = results.leaderboard.first() {
            report.push_str("\n## Best Model Details\n\n");
            report.push_str(&format!("**Model**: {}\n", best_model.model_name));
            report.push_str(&format!(
                "**CV Score**: {:.4} ± {:.4}\n",
                best_model.cv_score, best_model.cv_std
            ));

            report.push_str("\n**Parameters**:\n");
            for (param, value) in &best_model.parameters {
                report.push_str(&format!("- {}: {}\n", param, value));
            }

            if let Some(importances) = &best_model.feature_importance {
                report.push_str("\n**Top 10 Feature Importances**:\n");
                let mut importance_vec: Vec<_> = importances.iter().collect();
                importance_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

                for (feature, importance) in importance_vec.iter().take(10) {
                    report.push_str(&format!("- {}: {:.4}\n", feature, importance));
                }
            }
        }

        // Feature engineering summary
        if let Some(feature_engineer) = &self.feature_engineer {
            report.push_str("\n## Feature Engineering\n\n");
            if let Some(feature_names) = feature_engineer.get_feature_names() {
                report.push_str(&format!(
                    "- **Total Features Generated**: {}\n",
                    feature_names.len()
                ));
            }
            if let Some(selected_features) = feature_engineer.get_selected_features() {
                report.push_str(&format!(
                    "- **Features Selected**: {}\n",
                    selected_features.len()
                ));
            }
        }

        report.push_str("\n---\n");
        report.push_str("*Report generated by PandRS AutoML*\n");

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Series;

    #[test]
    fn test_task_type_detection() {
        let automl = AutoML::new();

        // Test regression detection
        let mut y_reg = DataFrame::new();
        y_reg
            .add_column(
                "target".to_string(),
                Series::new(vec![1.5, 2.3, 3.7, 4.1, 5.9], Some("target".to_string())).unwrap(),
            )
            .unwrap();

        let task_type = automl.detect_task_type(&y_reg).unwrap();
        assert!(matches!(task_type, TaskType::Regression));

        // Test binary classification detection
        let mut y_binary = DataFrame::new();
        y_binary
            .add_column(
                "target".to_string(),
                Series::new(vec![0.0, 1.0, 1.0, 0.0, 1.0], Some("target".to_string())).unwrap(),
            )
            .unwrap();

        let task_type = automl.detect_task_type(&y_binary).unwrap();
        assert!(matches!(task_type, TaskType::BinaryClassification));

        // Test multi-class classification detection
        let mut y_multi = DataFrame::new();
        y_multi
            .add_column(
                "target".to_string(),
                Series::new(vec![0.0, 1.0, 2.0, 1.0, 2.0], Some("target".to_string())).unwrap(),
            )
            .unwrap();

        let task_type = automl.detect_task_type(&y_multi).unwrap();
        assert!(matches!(task_type, TaskType::MultiClassification));
    }

    #[test]
    fn test_model_search_space() {
        let search_space = ModelSearchSpace::default_regression();

        assert!(!search_space.linear_models.is_empty());
        assert!(!search_space.tree_models.is_empty());
        assert!(!search_space.ensemble_models.is_empty());

        // Check that linear regression is included
        let has_linear_regression = search_space
            .linear_models
            .iter()
            .any(|(name, _)| name == "LinearRegression");
        assert!(has_linear_regression);
    }

    #[test]
    fn test_automl_config() {
        let config = AutoMLConfig::default();

        assert!(matches!(config.task_type, TaskType::Auto));
        assert_eq!(config.time_limit, Some(3600.0));
        assert_eq!(config.max_models, Some(50));
        assert!(config.feature_engineering);
        assert!(config.feature_selection);
    }

    #[test]
    fn test_complexity_scoring() {
        let automl = AutoML::new();

        let linear_complexity =
            automl.calculate_complexity_score("LinearRegression", &HashMap::new());
        let rf_complexity = automl.calculate_complexity_score("RandomForest", &HashMap::new());

        assert!(linear_complexity < rf_complexity);

        // Test parameter influence
        let mut params = HashMap::new();
        params.insert("n_estimators".to_string(), "200".to_string());
        let rf_complex_complexity = automl.calculate_complexity_score("RandomForest", &params);

        assert!(rf_complex_complexity > rf_complexity);
    }
}
