//! Data preprocessing for machine learning
//!
//! This module provides tools for preprocessing data before feeding it to
//! machine learning algorithms, including scaling, normalization, encoding
//! categorical variables, and handling missing values.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::series::Series;
use std::collections::{HashMap, HashSet};

/// Standard scaler for normalizing features to zero mean and unit variance
#[derive(Debug, Clone)]
pub struct StandardScaler {
    /// Mean values for each feature
    pub means: Option<HashMap<String, f64>>,
    /// Standard deviation values for each feature
    pub stds: Option<HashMap<String, f64>>,
    /// Columns to scale (if None, scale all numeric columns)
    pub columns: Option<Vec<String>>,
}

impl StandardScaler {
    /// Create a new StandardScaler
    pub fn new() -> Self {
        StandardScaler {
            means: None,
            stds: None,
            columns: None,
        }
    }

    /// Specify columns to scale
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, df: &DataFrame) -> Result<()> {
        let columns = match &self.columns {
            Some(cols) => cols.clone(),
            None => df.column_names().into_iter().collect(),
        };

        let mut means = HashMap::new();
        let mut stds = HashMap::new();

        for col_name in columns {
            if !df.has_column(&col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }

            let col = df.get_column::<f64>(&col_name)?;

            // Skip non-numeric columns
            if let Ok(numeric_data) = col.as_f64() {
                if numeric_data.is_empty() {
                    continue;
                }

                // Calculate mean
                let mean: f64 = numeric_data.iter().sum::<f64>() / numeric_data.len() as f64;
                means.insert(col_name.to_string(), mean);

                // Calculate standard deviation
                let variance: f64 = numeric_data
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>()
                    / numeric_data.len() as f64;

                let std_dev = variance.sqrt();
                stds.insert(col_name.to_string(), std_dev);
            }
        }

        self.means = Some(means);
        self.stds = Some(stds);

        Ok(())
    }

    /// Transform data using the fitted scaler
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if self.means.is_none() || self.stds.is_none() {
            return Err(Error::InvalidValue("StandardScaler not fitted".into()));
        }

        let means = self.means.as_ref().unwrap();
        let stds = self.stds.as_ref().unwrap();

        let mut result = DataFrame::new();

        // Add all columns from original DataFrame
        for col_name in df.column_names() {
            let col = df.get_column::<f64>(&col_name)?;

            // If this is a column we're scaling
            if means.contains_key(&col_name) && stds.contains_key(&col_name) {
                let mean = means[&col_name];
                let std_dev = stds[&col_name];

                if let Ok(numeric_data) = col.as_f64() {
                    // Avoid division by zero
                    if std_dev > 1e-10 {
                        let scaled_data: Vec<f64> =
                            numeric_data.iter().map(|&x| (x - mean) / std_dev).collect();

                        result.add_column(
                            col_name.to_string(),
                            Series::new(scaled_data, Some(col_name.to_string()))?,
                        )?;
                    } else {
                        // If standard deviation is zero, set all values to zero
                        let scaled_data = vec![0.0; numeric_data.len()];
                        result.add_column(
                            col_name.to_string(),
                            Series::new(scaled_data, Some(col_name.to_string()))?,
                        )?;
                    }
                } else {
                    // Non-numeric column, add as is
                    let col_clone = col.clone();
                    result.add_column(col_name.to_string(), col_clone)?;
                }
            } else {
                // Column not being scaled, add as is
                let col_clone = col.clone();
                result.add_column(col_name.to_string(), col_clone)?;
            }
        }

        Ok(result)
    }

    /// Fit the scaler to the data and transform it in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }
}

/// Min-Max scaler for scaling features to a specific range
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    /// Minimum values for each feature
    pub min_values: Option<HashMap<String, f64>>,
    /// Maximum values for each feature
    pub max_values: Option<HashMap<String, f64>>,
    /// Columns to scale (if None, scale all numeric columns)
    pub columns: Option<Vec<String>>,
    /// Feature range (min, max)
    pub feature_range: (f64, f64),
}

impl MinMaxScaler {
    /// Create a new MinMaxScaler with default range [0, 1]
    pub fn new() -> Self {
        MinMaxScaler {
            min_values: None,
            max_values: None,
            columns: None,
            feature_range: (0.0, 1.0),
        }
    }

    /// Create a new MinMaxScaler with custom range
    pub fn new_with_range(min: f64, max: f64) -> Self {
        MinMaxScaler {
            min_values: None,
            max_values: None,
            columns: None,
            feature_range: (min, max),
        }
    }

    /// Set the feature range (builder pattern)
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.feature_range = (min, max);
        self
    }

    /// Specify columns to scale
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    // Implementation details omitted for brevity
    // Similar to StandardScaler, but scales to [min, max] range
}

/// One-hot encoder for categorical variables
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    /// Categories for each feature
    pub categories: Option<HashMap<String, Vec<String>>>,
    /// Columns to encode (if None, encode all categorical columns)
    pub columns: Option<Vec<String>>,
    /// Whether to drop the first category
    pub drop_first: bool,
    /// Prefix for new column names
    pub prefix: Option<String>,
}

impl OneHotEncoder {
    /// Create a new OneHotEncoder
    pub fn new() -> Self {
        OneHotEncoder {
            categories: None,
            columns: None,
            drop_first: false,
            prefix: None,
        }
    }

    /// Specify columns to encode
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Set whether to drop the first category
    pub fn drop_first(mut self, drop_first: bool) -> Self {
        self.drop_first = drop_first;
        self
    }

    /// Set prefix for new column names
    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.prefix = Some(prefix);
        self
    }

    // Implementation details omitted for brevity
    // Creates binary columns for each category in categorical columns
}

/// Generate polynomial features
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    /// Degree of the polynomial
    pub degree: usize,
    /// Whether to include bias term (constant feature)
    pub include_bias: bool,
    /// Whether to include interaction features only
    pub interaction_only: bool,
    /// Columns to use (if None, use all numeric columns)
    pub columns: Option<Vec<String>>,
}

impl PolynomialFeatures {
    /// Create a new PolynomialFeatures instance
    pub fn new(degree: usize) -> Self {
        PolynomialFeatures {
            degree,
            include_bias: true,
            interaction_only: false,
            columns: None,
        }
    }

    /// Set whether to include bias term
    pub fn include_bias(mut self, include_bias: bool) -> Self {
        self.include_bias = include_bias;
        self
    }

    /// Set whether to include interaction features only
    pub fn interaction_only(mut self, interaction_only: bool) -> Self {
        self.interaction_only = interaction_only;
        self
    }

    /// Specify columns to use
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    // Implementation details omitted for brevity
    // Generates polynomial and interaction features
}

/// Bin continuous features into discrete bins
#[derive(Debug, Clone)]
pub struct Binner {
    /// Number of bins
    pub n_bins: usize,
    /// Bin edges for each feature
    pub bin_edges: Option<HashMap<String, Vec<f64>>>,
    /// Strategy for binning ('uniform', 'quantile', or 'kmeans')
    pub strategy: String,
    /// Columns to bin (if None, bin all numeric columns)
    pub columns: Option<Vec<String>>,
}

impl Binner {
    /// Create a new Binner with uniform strategy
    pub fn new(n_bins: usize) -> Self {
        Binner {
            n_bins,
            bin_edges: None,
            strategy: "uniform".to_string(),
            columns: None,
        }
    }

    /// Set binning strategy
    pub fn with_strategy(mut self, strategy: &str) -> Self {
        self.strategy = strategy.to_string();
        self
    }

    /// Specify columns to bin
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    // Implementation details omitted for brevity
    // Bins continuous features into discrete bins
}

/// Strategy for imputing missing values
#[derive(Debug, Clone, PartialEq)]
pub enum ImputeStrategy {
    /// Impute with mean
    Mean,
    /// Impute with median
    Median,
    /// Impute with most frequent value
    MostFrequent,
    /// Impute with constant value
    Constant(f64),
}

/// Imputer for handling missing values
#[derive(Debug, Clone)]
pub struct Imputer {
    /// Strategy for imputing missing values
    pub strategy: ImputeStrategy,
    /// Fill values for each feature
    pub fill_values: Option<HashMap<String, f64>>,
    /// Columns to impute (if None, impute all columns with missing values)
    pub columns: Option<Vec<String>>,
}

impl Imputer {
    /// Create a new Imputer with mean strategy
    pub fn new() -> Self {
        Imputer {
            strategy: ImputeStrategy::Mean,
            fill_values: None,
            columns: None,
        }
    }

    /// Set imputation strategy
    pub fn with_strategy(mut self, strategy: ImputeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Specify columns to impute
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    // Implementation details omitted for brevity
    // Imputes missing values in the data
}

/// Feature selector for selecting a subset of features
#[derive(Debug, Clone)]
pub struct FeatureSelector {
    /// Columns to select
    pub columns: Vec<String>,
}

impl FeatureSelector {
    /// Create a new FeatureSelector
    pub fn new(columns: Vec<String>) -> Self {
        FeatureSelector { columns }
    }

    /// Transform data by selecting features
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        for col_name in &self.columns {
            if !df.has_column(col_name) {
                return Err(Error::ColumnNotFound(col_name.to_string()));
            }

            // We need a specific type for the column, so let's use f64 as a placeholder
            let col = df.get_column::<f64>(col_name)?;
            // Clone the Series to own it before adding to the new DataFrame
            let new_series = col.clone();
            result.add_column(col_name.to_string(), new_series)?;
        }

        Ok(result)
    }
}

// No need for re-exports here as the types are already defined in this module
