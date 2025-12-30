//! Machine Learning Models Module
//!
//! Provides machine learning model implementations and utilities for model evaluation.
//! Implementation compatible with OptimizedDataFrame.

use crate::optimized::{OptimizedDataFrame, ColumnView};
use crate::error::{Result, Error};
use crate::column::{Float64Column, Column, ColumnTrait};
use crate::dataframe::DataValue;
use crate::stats;
use std::collections::HashMap;
use rand::Rng;
use crate::utils::rand_compat::GenRangeCompat;

/// Trait common to supervised learning models
pub trait SupervisedModel {
    /// Fit the model with training data
    fn fit(&mut self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<()>;
    
    /// Make predictions on new data
    fn predict(&self, df: &OptimizedDataFrame) -> Result<Vec<f64>>;
    
    /// Calculate model score (default is R^2)
    fn score(&self, df: &OptimizedDataFrame, target: &str) -> Result<f64> {
        // Get target variable
        let col = df.column(target)?;
        // Convert to Float64Column
        let y_true = self.extract_numeric_values(&col)?;
        
        // Get predictions
        let y_pred = self.predict(df)?;
        
        // Use R^2 score by default
        crate::ml::metrics::regression::r2_score(&y_true, &y_pred)
    }
    
    /// Helper method to extract numeric data from column
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

/// Linear Regression Model
pub struct LinearRegression {
    /// Regression coefficients
    coefficients: Vec<f64>,
    /// Intercept
    intercept: f64,
    /// Feature names
    feature_names: Vec<String>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl LinearRegression {
    /// Create a new linear regression model
    pub fn new() -> Self {
        LinearRegression {
            coefficients: Vec::new(),
            intercept: 0.0,
            feature_names: Vec::new(),
            fitted: false,
        }
    }
    
    /// Get coefficients
    pub fn coefficients(&self) -> HashMap<String, f64> {
        self.feature_names
            .iter()
            .zip(self.coefficients.iter())
            .map(|(name, coef)| (name.clone(), *coef))
            .collect()
    }
    
    /// Get intercept
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    
    /// Calculate R^2 score of the model
    pub fn r_squared(&self, df: &OptimizedDataFrame, target: &str) -> Result<f64> {
        self.score(df, target)
    }
    
    // Convert dataframe to linear regression data for stats module
    fn prepare_data_for_regression(&self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
        // Extract feature data
        let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(df.row_count());
        for _ in 0..df.row_count() {
            x_data.push(Vec::with_capacity(features.len()));
        }
        
        // Get feature values
        for &feature in features {
            let column = df.column(feature)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            // Add to matrix
            for (i, &value) in values.iter().enumerate() {
                if i < x_data.len() {
                    x_data[i].push(value);
                }
            }
        }
        
        // Get target variable values
        let target_column = df.column(target)?;
        
        let y_data = self.extract_numeric_values(&target_column)?;
        
        Ok((x_data, y_data))
    }
    
    // Implementation of linear regression using least squares method
    fn fit_linear_regression(&mut self, x: &[Vec<f64>], y: &[f64], feature_names: &[&str]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(Error::Empty("Empty data for linear regression".to_string()));
        }
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        if n_features != feature_names.len() {
            return Err(Error::Consistency(format!(
                "Number of features ({}) doesn't match feature names length ({})",
                n_features, feature_names.len()
            )));
        }
        
        // Calculate XTX (X^T * X) matrix for matrix computation
        let mut xtx = vec![vec![0.0; n_features + 1]; n_features + 1];
        let mut xty = vec![0.0; n_features + 1];
        
        // Set XTX part for bias term
        xtx[0][0] = n_samples as f64;
        
        // Set XTY part for bias term
        xty[0] = y.iter().sum();
        
        // Calculate XTX matrix for feature part
        for i in 0..n_features {
            // Interaction between bias term and feature
            let sum_xi = x.iter().map(|sample| sample[i]).sum::<f64>();
            xtx[0][i + 1] = sum_xi;
            xtx[i + 1][0] = sum_xi;
            
            // Interaction between features
            for j in 0..n_features {
                let sum_xixj = x.iter().map(|sample| sample[i] * sample[j]).sum::<f64>();
                xtx[i + 1][j + 1] = sum_xixj;
            }
            
            // Calculate XTY part
            let sum_xiy = x.iter().zip(y.iter()).map(|(sample, &yi)| sample[i] * yi).sum::<f64>();
            xty[i + 1] = sum_xiy;
        }
        
        // Solve linear system using Gaussian elimination
        let mut coeffs = self.solve_linear_system(&xtx, &xty)?;
        
        // Set intercept and coefficients
        self.intercept = coeffs.remove(0);
        self.coefficients = coeffs;
        self.feature_names = feature_names.iter().map(|&s| s.to_string()).collect();
        self.fitted = true;
        
        Ok(())
    }
    
    // Solve linear system using Gaussian elimination
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
        let n = a.len();
        
        // Create augmented coefficient matrix
        let mut augmented = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n + 1);
            row.extend_from_slice(&a[i]);
            row.push(b[i]);
            augmented.push(row);
        }
        
        // Forward elimination
        for i in 0..n {
            // Pivot selection (partial pivoting)
            let mut max_idx = i;
            let mut max_val = augmented[i][i].abs();
            
            for j in (i + 1)..n {
                let abs_val = augmented[j][i].abs();
                if abs_val > max_val {
                    max_idx = j;
                    max_val = abs_val;
                }
            }
            
            // Check for singular matrix
            if max_val < 1e-10 {
                return Err(Error::Computation("Singular matrix, cannot solve linear system".to_string()));
            }
            
            // Swap rows
            if max_idx != i {
                augmented.swap(i, max_idx);
            }
            
            // Normalize by diagonal element
            let pivot = augmented[i][i];
            for j in i..=n {
                augmented[i][j] /= pivot;
            }
            
            // Eliminate other rows
            for j in 0..n {
                if j != i {
                    let factor = augmented[j][i];
                    for k in i..=n {
                        augmented[j][k] -= factor * augmented[i][k];
                    }
                }
            }
        }
        
        // Get solution
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[i] = augmented[i][n];
        }
        
        Ok(x)
    }
}

impl SupervisedModel for LinearRegression {
    fn fit(&mut self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<()> {
        // Prepare data
        let (x_data, y_data) = self.prepare_data_for_regression(df, target, features)?;
        
        // Perform linear regression
        self.fit_linear_regression(&x_data, &y_data, features)?;
        
        Ok(())
    }
    
    fn predict(&self, df: &OptimizedDataFrame) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "Model has not been fitted yet".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        let mut predictions = Vec::with_capacity(n_rows);
        
        // Check for feature existence and map column indices
        let mut feature_columns = Vec::with_capacity(self.feature_names.len());
        for feature_name in &self.feature_names {
            let column = df.column(feature_name)?;
            feature_columns.push(column);
        }
        
        // Predict for each row
        for row_idx in 0..n_rows {
            let mut pred = self.intercept;
            
            // Add contribution of each feature
            for (i, column) in feature_columns.iter().enumerate() {
                let feature_value = match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        if let Some(float_col) = column.as_float64() {
                            float_col.get(row_idx)?.unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    },
                    crate::column::ColumnType::Int64 => {
                        if let Some(int_col) = column.as_int64() {
                            int_col.get(row_idx)?.map(|v| v as f64).unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    },
                    _ => 0.0 // Treat non-numeric types as 0 (or implement appropriate strategy)
                };
                
                pred += self.coefficients[i] * feature_value;
            }
            
            predictions.push(pred);
        }
        
        Ok(predictions)
    }
}

/// Logistic Regression Model
pub struct LogisticRegression {
    /// Regression coefficients
    coefficients: Vec<f64>,
    /// Intercept
    intercept: f64,
    /// Feature names
    feature_names: Vec<String>,
    /// Learning rate
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence threshold
    tol: f64,
    /// Whether the model has been fitted
    fitted: bool,
}

impl LogisticRegression {
    /// Create a new logistic regression model
    pub fn new(learning_rate: f64, max_iter: usize, tol: f64) -> Self {
        LogisticRegression {
            coefficients: Vec::new(),
            intercept: 0.0,
            feature_names: Vec::new(),
            learning_rate,
            max_iter,
            tol,
            fitted: false,
        }
    }
    
    /// Get coefficients
    pub fn coefficients(&self) -> HashMap<String, f64> {
        self.feature_names
            .iter()
            .zip(self.coefficients.iter())
            .map(|(name, coef)| (name.clone(), *coef))
            .collect()
    }
    
    /// Get intercept
    pub fn intercept(&self) -> f64 {
        self.intercept
    }
    
    /// Sigmoid function
    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }
    
    /// Calculate prediction probability for logistic regression
    fn predict_proba_row(&self, features: &[f64]) -> f64 {
        if features.len() != self.coefficients.len() {
            return 0.5; // Return default probability on error
        }
        
        let mut z = self.intercept;
        for i in 0..features.len() {
            z += features[i] * self.coefficients[i];
        }
        
        self.sigmoid(z)
    }
    
    /// Predict as a probability model
    pub fn predict_proba(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
        if !self.fitted {
            return Err(Error::InvalidOperation(
                "Model has not been fitted yet".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        let mut proba_class0 = Vec::with_capacity(n_rows);
        let mut proba_class1 = Vec::with_capacity(n_rows);
        
        // Get feature columns
        let mut feature_columns = Vec::with_capacity(self.feature_names.len());
        for feature_name in &self.feature_names {
            let column = df.column(feature_name)?;
            feature_columns.push(column);
        }
        
        // Calculate probability for each row
        for row_idx in 0..n_rows {
            let mut features = Vec::with_capacity(self.feature_names.len());
            
            // Get row features
            for column in &feature_columns {
                let value = match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        if let Some(float_col) = column.as_float64() {
                            float_col.get(row_idx)?.unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    },
                    crate::column::ColumnType::Int64 => {
                        if let Some(int_col) = column.as_int64() {
                            int_col.get(row_idx)?.map(|v| v as f64).unwrap_or(0.0)
                        } else {
                            0.0
                        }
                    },
                    _ => 0.0
                };
                
                features.push(value);
            }
            
            // Calculate probability
            let prob = self.predict_proba_row(&features);
            proba_class1.push(prob);
            proba_class0.push(1.0 - prob);
        }
        
        // Return result as a new dataframe
        let mut result_df = OptimizedDataFrame::new();
        
        // Add probability columns
        let class0_col = Float64Column::with_name(proba_class0, "probability_0");
        let class1_col = Float64Column::with_name(proba_class1, "probability_1");
        
        result_df.add_column("probability_0", Column::Float64(class0_col))?;
        result_df.add_column("probability_1", Column::Float64(class1_col))?;
        
        Ok(result_df)
    }
    
    /// Train logistic regression using stochastic gradient descent
    fn fit_logistic_regression(&mut self, x: &[Vec<f64>], y: &[f64], feature_names: &[&str]) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(Error::Empty("Empty data for logistic regression".to_string()));
        }
        
        let n_samples = x.len();
        let n_features = x[0].len();
        
        if n_features != feature_names.len() {
            return Err(Error::Consistency(format!(
                "Number of features ({}) doesn't match feature names length ({})",
                n_features, feature_names.len()
            )));
        }
        
        // Initialize parameters
        let mut weights = vec![0.0; n_features];
        let mut intercept = 0.0;
        
        // Stochastic gradient descent
        for _ in 0..self.max_iter {
            let mut weights_grad = vec![0.0; n_features];
            let mut intercept_grad = 0.0;
            let mut loss = 0.0;
            
            // Calculate gradient
            for i in 0..n_samples {
                let mut z = intercept;
                for j in 0..n_features {
                    z += weights[j] * x[i][j];
                }
                
                let y_pred = self.sigmoid(z);
                let error = y_pred - y[i];
                
                // Calculate loss function (cross-entropy)
                if y[i] > 0.5 {
                    loss -= (y_pred + 1e-15).ln();
                } else {
                    loss -= (1.0 - y_pred + 1e-15).ln();
                }
                
                // Update gradient
                intercept_grad += error;
                for j in 0..n_features {
                    weights_grad[j] += error * x[i][j];
                }
            }
            
            // Loss is monitored but not used here
            // Update parameters
            intercept -= self.learning_rate * intercept_grad / n_samples as f64;
            for j in 0..n_features {
                weights[j] -= self.learning_rate * weights_grad[j] / n_samples as f64;
            }
            
            // Convergence check
            if weights_grad.iter().all(|&g| g.abs() < self.tol) && intercept_grad.abs() < self.tol {
                break;
            }
        }
        
        // Set model parameters
        self.intercept = intercept;
        self.coefficients = weights;
        self.feature_names = feature_names.iter().map(|&s| s.to_string()).collect();
        self.fitted = true;
        
        Ok(())
    }
}

impl SupervisedModel for LogisticRegression {
    fn fit(&mut self, df: &OptimizedDataFrame, target: &str, features: &[&str]) -> Result<()> {
        // Extract feature data
        let mut x_data: Vec<Vec<f64>> = Vec::with_capacity(df.row_count());
        for _ in 0..df.row_count() {
            x_data.push(Vec::with_capacity(features.len()));
        }
        
        // Get feature values
        for &feature in features {
            let column = df.column(feature)?;
            
            let values = self.extract_numeric_values(&column)?;
            
            // Add to matrix
            for (i, &value) in values.iter().enumerate() {
                if i < x_data.len() {
                    x_data[i].push(value);
                }
            }
        }
        
        // Get target variable values (convert to 0/1 for binary classification)
        let target_column = df.column(target)?;
        
        let target_values = self.extract_target_values(&target_column)?;
        
        // Perform logistic regression
        self.fit_logistic_regression(&x_data, &target_values, features)?;
        
        Ok(())
    }
    
    fn predict(&self, df: &OptimizedDataFrame) -> Result<Vec<f64>> {
        // Calculate probabilities
        let proba_df = self.predict_proba(df)?;
        
        // If probability is 0.5 or higher, predict 1, otherwise 0
        let proba_col = proba_df.column("probability_1")?;
        
        let mut predictions = Vec::with_capacity(proba_col.len());
        
        if let Some(float_col) = proba_col.as_float64() {
            for i in 0..proba_col.len() {
                let prob = float_col.get(i)?.unwrap_or(0.0);
                let prediction = if prob >= 0.5 { 1.0 } else { 0.0 };
                predictions.push(prediction);
            }
        } else {
            return Err(Error::ColumnTypeMismatch {
                name: "probability_1".to_string(),
                expected: crate::column::ColumnType::Float64,
                found: proba_col.column_type(),
            });
        }
        
        Ok(predictions)
    }
}

impl LogisticRegression {
    // Convert target values (categorical) to numeric for classification
    fn extract_target_values(&self, col: &ColumnView) -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(col.len());
        
        match col.column_type() {
            // Use numeric values as is
            crate::column::ColumnType::Float64 => {
                let float_col = col.as_float64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Float64,
                        found: col.column_type(),
                    }
                )?;
                
                for i in 0..col.len() {
                    if let Ok(Some(value)) = float_col.get(i) {
                        values.push(if value > 0.5 { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // Treat missing values as 0
                    }
                }
            },
            crate::column::ColumnType::Int64 => {
                let int_col = col.as_int64().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Int64,
                        found: col.column_type(),
                    }
                )?;
                
                for i in 0..col.len() {
                    if let Ok(Some(value)) = int_col.get(i) {
                        values.push(if value != 0 { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // Treat missing values as 0
                    }
                }
            },
            // Treat "1", "true", "yes", etc. as positive class for string type
            crate::column::ColumnType::String => {
                let string_col = col.as_string().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::String,
                        found: col.column_type(),
                    }
                )?;
                
                for i in 0..col.len() {
                    if let Ok(Some(value)) = string_col.get(i) {
                        let lower_val = value.to_lowercase();
                        let is_positive = lower_val == "1" || 
                                        lower_val == "true" || 
                                        lower_val == "yes" || 
                                        lower_val == "t" || 
                                        lower_val == "y";
                        values.push(if is_positive { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // Treat missing values as 0
                    }
                }
            },
            // Convert boolean type to 1 for true and 0 for false
            crate::column::ColumnType::Boolean => {
                let bool_col = col.as_boolean().ok_or_else(|| 
                    Error::ColumnTypeMismatch {
                        name: col.column().name().unwrap_or("").to_string(),
                        expected: crate::column::ColumnType::Boolean,
                        found: col.column_type(),
                    }
                )?;
                
                for i in 0..col.len() {
                    if let Ok(Some(value)) = bool_col.get(i) {
                        values.push(if value { 1.0 } else { 0.0 });
                    } else {
                        values.push(0.0); // Treat missing values as 0
                    }
                }
            }
            // All cases are covered above, so this pattern is not needed
        }
        
        Ok(values)
    }
}

/// Model selection module - train/test split and cross-validation
pub mod model_selection {
    use std::marker::PhantomData;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use crate::optimized::OptimizedDataFrame;
    use crate::error::{Result, Error};
    use crate::column::{Column, ColumnTrait, Float64Column};
    use super::SupervisedModel;
    
    /// Split dataset into training set and test set
    pub fn train_test_split(
        df: &OptimizedDataFrame,
        test_size: f64,
        random_state: Option<u64>
    ) -> Result<(OptimizedDataFrame, OptimizedDataFrame)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(Error::InvalidValue(format!(
                "Invalid test_size: {}, must be between 0 and 1", test_size
            )));
        }
        
        let n_rows = df.row_count();
        let n_test = (n_rows as f64 * test_size).round() as usize;
        let n_train = n_rows - n_test;
        
        if n_train == 0 || n_test == 0 {
            return Err(Error::InvalidOperation(
                "Both train and test splits must have at least one sample".to_string()
            ));
        }
        
        // Create indices
        let mut indices: Vec<usize> = (0..n_rows).collect();
        
        // Shuffle randomly
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random()),
        };
        
        // Shuffle using Fisher-Yates algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        // Split into training set and test set
        let train_indices = &indices[0..n_train];
        let test_indices = &indices[n_train..];
        
        // Create new dataframes
        let mut train_df = OptimizedDataFrame::new();
        let mut test_df = OptimizedDataFrame::new();
        
        // Add columns
        for col_name in df.column_names() {
            let column = df.column(col_name).unwrap();
            
            // Create column for training data
            let train_column = match column.column_type() {
                crate::column::ColumnType::Float64 => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        if let Some(float_col) = column.as_float64() {
                            train_values.push(float_col.get(idx)?.unwrap_or(0.0));
                        } else {
                            train_values.push(0.0); // Default for incompatible column type
                        }
                    }
                    {
                        let col = Float64Column::with_name(train_values, col_name.clone());
                        Column::Float64(col)
                    }
                },
                crate::column::ColumnType::Int64 => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        if let Some(int_col) = column.as_int64() {
                            train_values.push(int_col.get(idx)?.unwrap_or(0));
                        } else {
                            train_values.push(0); // Default for incompatible column type
                        }
                    }
                    {
                        let col = crate::column::Int64Column::with_name(train_values, col_name.clone());
                        Column::Int64(col)
                    }
                },
                crate::column::ColumnType::String => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        if let Some(string_col) = column.as_string() {
                            if let Ok(Some(val)) = string_col.get(idx) {
                                train_values.push(val.to_string());
                            } else {
                                train_values.push(String::default());
                            }
                        } else {
                            train_values.push(String::default());
                        }
                    }
                    Column::String(crate::column::StringColumn::with_name(train_values, col_name.clone()))
                },
                crate::column::ColumnType::Boolean => {
                    let mut train_values = Vec::with_capacity(n_train);
                    for &idx in train_indices {
                        if let Some(bool_col) = column.as_boolean() {
                            train_values.push(bool_col.get(idx)?.unwrap_or(false));
                        } else {
                            train_values.push(false);
                        }
                    }
                    Column::Boolean(crate::column::BooleanColumn::with_name(train_values, col_name.clone()))
                },
            };
            
            // Create column for test data
            let test_column = match column.column_type() {
                crate::column::ColumnType::Float64 => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        if let Some(float_col) = column.as_float64() {
                            test_values.push(float_col.get(idx)?.unwrap_or(0.0));
                        } else {
                            test_values.push(0.0); // Default for incompatible column type
                        }
                    }
                    {
                        let col = Float64Column::with_name(test_values, col_name.clone());
                        Column::Float64(col)
                    }
                },
                crate::column::ColumnType::Int64 => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        if let Some(int_col) = column.as_int64() {
                            test_values.push(int_col.get(idx)?.unwrap_or(0));
                        } else {
                            test_values.push(0); // Default for incompatible column type
                        }
                    }
                    {
                        let col = crate::column::Int64Column::with_name(test_values, col_name.clone());
                        Column::Int64(col)
                    }
                },
                crate::column::ColumnType::String => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        if let Some(string_col) = column.as_string() {
                            if let Ok(Some(val)) = string_col.get(idx) {
                                test_values.push(val.to_string());
                            } else {
                                test_values.push(String::default());
                            }
                        } else {
                            test_values.push(String::default());
                        }
                    }
                    Column::String(crate::column::StringColumn::with_name(test_values, col_name.clone()))
                },
                crate::column::ColumnType::Boolean => {
                    let mut test_values = Vec::with_capacity(n_test);
                    for &idx in test_indices {
                        if let Some(bool_col) = column.as_boolean() {
                            test_values.push(bool_col.get(idx)?.unwrap_or(false));
                        } else {
                            test_values.push(false);
                        }
                    }
                    Column::Boolean(crate::column::BooleanColumn::with_name(test_values, col_name.clone()))
                },
            };
            
            // Add columns to dataframes
            train_df.add_column(col_name.clone(), train_column)?;
            test_df.add_column(col_name.clone(), test_column)?;
        }
        
        Ok((train_df, test_df))
    }
    
    /// Model evaluation using K-fold cross-validation
    pub fn cross_val_score<M>(
        model: &M,
        df: &OptimizedDataFrame,
        target: &str,
        features: &[&str],
        k_folds: usize
    ) -> Result<Vec<f64>>
    where
        M: SupervisedModel + Clone,
    {
        if k_folds < 2 {
            return Err(Error::InvalidValue(
                "Number of folds must be at least 2".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        if n_rows < k_folds {
            return Err(Error::InvalidOperation(format!(
                "Cannot perform {}-fold cross validation with only {} samples",
                k_folds, n_rows
            )));
        }
        
        // Prepare for fold splitting
        let fold_size = n_rows / k_folds;
        let remainder = n_rows % k_folds;
        
        // Store scores for each fold
        let mut scores = Vec::with_capacity(k_folds);
        
        // Train and evaluate model for each fold
        for fold_idx in 0..k_folds {
            // Determine test data range
            let test_start = fold_idx * fold_size + fold_idx.min(remainder);
            let test_end = test_start + fold_size + if fold_idx < remainder { 1 } else { 0 };
            
            // Create training and test dataframes
            let mut train_df = OptimizedDataFrame::new();
            let mut test_df = OptimizedDataFrame::new();
            
            // Split data for each column
            for col_name in df.column_names() {
                let column = df.column(col_name).unwrap();
                
                // Create columns for training and test data
                match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = if let Some(float_col) = column.as_float64() {
                                float_col.get(i)?.unwrap_or(0.0)
                            } else {
                                0.0 // Default for incompatible column type
                            };
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = Float64Column::with_name(train_values, col_name.clone());
                        let test_col = Float64Column::with_name(test_values, col_name.clone());
                        
                        train_df.add_column(col_name, Column::Float64(train_col))?;
                        test_df.add_column(col_name, Column::Float64(test_col))?;
                    },
                    crate::column::ColumnType::Int64 => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = if let Some(int_col) = column.as_int64() {
                                int_col.get(i)?.unwrap_or(0)
                            } else {
                                0 // Default for incompatible column type
                            };
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = crate::column::Int64Column::with_name(train_values, col_name.clone());
                        let test_col = crate::column::Int64Column::with_name(test_values, col_name.clone());
                        
                        train_df.add_column(col_name, Column::Int64(train_col))?;
                        test_df.add_column(col_name, Column::Int64(test_col))?;
                    },
                    crate::column::ColumnType::String => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = if let Some(string_col) = column.as_string() {
                                if let Ok(Some(val)) = string_col.get(i) {
                                    val.to_string()
                                } else {
                                    String::default()
                                }
                            } else {
                                String::default()
                            };
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = crate::column::StringColumn::with_name(train_values, col_name.clone());
                        let test_col = crate::column::StringColumn::with_name(test_values, col_name.clone());
                        
                        train_df.add_column(col_name, Column::String(train_col))?;
                        test_df.add_column(col_name, Column::String(test_col))?;
                    },
                    crate::column::ColumnType::Boolean => {
                        let mut train_values = Vec::with_capacity(n_rows - (test_end - test_start));
                        let mut test_values = Vec::with_capacity(test_end - test_start);
                        
                        for i in 0..n_rows {
                            let value = if let Some(bool_col) = column.as_boolean() {
                                bool_col.get(i)?.unwrap_or(false)
                            } else {
                                false
                            };
                            if i < test_start || i >= test_end {
                                train_values.push(value);
                            } else {
                                test_values.push(value);
                            }
                        }
                        
                        let train_col = crate::column::BooleanColumn::with_name(train_values, col_name.clone());
                        let test_col = crate::column::BooleanColumn::with_name(test_values, col_name.clone());
                        
                        train_df.add_column(col_name, Column::Boolean(train_col))?;
                        test_df.add_column(col_name, Column::Boolean(test_col))?;
                    },
                }
            }
            
            // Clone model and train
            let mut fold_model = model.clone();
            fold_model.fit(&train_df, target, features)?;
            
            // Calculate score on test data
            let score = fold_model.score(&test_df, target)?;
            scores.push(score);
        }
        
        Ok(scores)
    }
    
    /// Helper function to split dataset into K folds
    pub fn k_fold_split(
        df: &OptimizedDataFrame,
        k_folds: usize,
        random_state: Option<u64>
    ) -> Result<Vec<(OptimizedDataFrame, OptimizedDataFrame)>> {
        if k_folds < 2 {
            return Err(Error::InvalidValue(
                "Number of folds must be at least 2".to_string()
            ));
        }
        
        let n_rows = df.row_count();
        if n_rows < k_folds {
            return Err(Error::InvalidOperation(format!(
                "Cannot perform {}-fold cross validation with only {} samples",
                k_folds, n_rows
            )));
        }
        
        // Create indices
        let mut indices: Vec<usize> = (0..n_rows).collect();
        
        // Shuffle randomly
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(rand::random()),
        };
        
        // Shuffle using Fisher-Yates algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        
        // Calculate fold size
        let fold_size = n_rows / k_folds;
        let remainder = n_rows % k_folds;
        
        // Create splits for each fold
        let mut folds = Vec::with_capacity(k_folds);
        
        for fold_idx in 0..k_folds {
            // Determine test data range
            let test_start = fold_idx * fold_size + fold_idx.min(remainder);
            let test_end = test_start + fold_size + if fold_idx < remainder { 1 } else { 0 };
            
            // Extract training and test indices
            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices.iter()
                .enumerate()
                .filter(|&(i, _)| i < test_start || i >= test_end)
                .map(|(_, &idx)| idx)
                .collect();
            
            // Create training and test dataframes
            let mut train_df = OptimizedDataFrame::new();
            let mut test_df = OptimizedDataFrame::new();
            
            // Split data for each column
            for col_name in df.column_names() {
                let column = df.column(col_name).unwrap();
                
                // Extract values based on indices
                match column.column_type() {
                    crate::column::ColumnType::Float64 => {
                        let mut train_values = Vec::with_capacity(train_indices.len());
                        let mut test_values = Vec::with_capacity(test_indices.len());
                        
                        for &idx in &train_indices {
                            if let Some(float_col) = column.as_float64() {
                            train_values.push(float_col.get(idx)?.unwrap_or(0.0));
                        } else {
                            train_values.push(0.0); // Default for incompatible column type
                        }
                        }
                        
                        for &idx in &test_indices {
                            if let Some(float_col) = column.as_float64() {
                            test_values.push(float_col.get(idx)?.unwrap_or(0.0));
                        } else {
                            test_values.push(0.0); // Default for incompatible column type
                        }
                        }
                        
                        let train_col = Float64Column::with_name(train_values, col_name.clone());
                        let test_col = Float64Column::with_name(test_values, col_name.clone());
                        
                        train_df.add_column(col_name, Column::Float64(train_col))?;
                        test_df.add_column(col_name, Column::Float64(test_col))?;
                    },
                    // Handle other data types similarly
                    _ => {
                        // Omitted for brevity
                    }
                }
            }
            
            folds.push((train_df, test_df));
        }
        
        Ok(folds)
    }
}

/// Model persistence module - saving and loading models
pub mod model_persistence {
    use serde::{Serialize, Deserialize};
    use std::path::Path;
    use std::fs::{File, create_dir_all};
    use std::io::{BufReader, BufWriter, Write};
    use crate::error::{Result, Error};
    
    /// Model persistence trait
    pub trait ModelPersistence: Sized {
        /// Save model as a JSON file
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()>;
        
        /// Load model from a JSON file
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self>;
    }
    
    /// Data structure for linear regression model persistence
    #[derive(Serialize, Deserialize)]
    struct LinearRegressionData {
        coefficients: Vec<f64>,
        intercept: f64,
        feature_names: Vec<String>,
    }
    
    /// Data structure for logistic regression model persistence
    #[derive(Serialize, Deserialize)]
    struct LogisticRegressionData {
        coefficients: Vec<f64>,
        intercept: f64,
        feature_names: Vec<String>,
        learning_rate: f64,
        max_iter: usize,
        tol: f64,
    }
    
    impl ModelPersistence for super::LinearRegression {
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            if !self.fitted {
                return Err(Error::InvalidOperation(
                    "Cannot save unfitted model".to_string()
                ));
            }
            
            // Create model data
            let model_data = LinearRegressionData {
                coefficients: self.coefficients.clone(),
                intercept: self.intercept,
                feature_names: self.feature_names.clone(),
            };
            
            // Create directory if it doesn't exist
            if let Some(parent) = path.as_ref().parent() {
                if !parent.exists() {
                    create_dir_all(parent)?;
                }
            }
            
            // Save as JSON
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &model_data)?;
            
            Ok(())
        }
        
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self> {
            // Check if file exists
            if !path.as_ref().exists() {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Model file not found: {:?}", path.as_ref())
                )));
            }
            
            // Read data from JSON
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let model_data: LinearRegressionData = serde_json::from_reader(reader)?;
            
            // Reconstruct model
            let mut model = super::LinearRegression::new();
            model.coefficients = model_data.coefficients;
            model.intercept = model_data.intercept;
            model.feature_names = model_data.feature_names;
            model.fitted = true;
            
            Ok(model)
        }
    }
    
    impl ModelPersistence for super::LogisticRegression {
        fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()> {
            if !self.fitted {
                return Err(Error::InvalidOperation(
                    "Cannot save unfitted model".to_string()
                ));
            }
            
            // Create model data
            let model_data = LogisticRegressionData {
                coefficients: self.coefficients.clone(),
                intercept: self.intercept,
                feature_names: self.feature_names.clone(),
                learning_rate: self.learning_rate,
                max_iter: self.max_iter,
                tol: self.tol,
            };
            
            // Create directory if it doesn't exist
            if let Some(parent) = path.as_ref().parent() {
                if !parent.exists() {
                    create_dir_all(parent)?;
                }
            }
            
            // Save as JSON
            let file = File::create(path)?;
            let writer = BufWriter::new(file);
            serde_json::to_writer_pretty(writer, &model_data)?;
            
            Ok(())
        }
        
        fn load_model<P: AsRef<Path>>(path: P) -> Result<Self> {
            // Check if file exists
            if !path.as_ref().exists() {
                return Err(Error::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("Model file not found: {:?}", path.as_ref())
                )));
            }
            
            // Read data from JSON
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let model_data: LogisticRegressionData = serde_json::from_reader(reader)?;
            
            // Reconstruct model
            let mut model = super::LogisticRegression::new(
                model_data.learning_rate,
                model_data.max_iter,
                model_data.tol
            );
            model.coefficients = model_data.coefficients;
            model.intercept = model_data.intercept;
            model.feature_names = model_data.feature_names;
            model.fitted = true;
            
            Ok(model)
        }
    }
}