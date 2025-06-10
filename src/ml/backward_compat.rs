//! Backward compatibility layer for the ML module
//!
//! This module provides backward compatibility for existing code that uses the old ML module structure.
//! It re-exports types and functions from the new module structure with appropriate deprecation notices.

#[allow(deprecated)]
pub mod models {
    //! Backward compatibility for ML models

    use crate::column::{Column, ColumnTrait, Float64Column};
    use crate::error::{Error, Result};
    use crate::optimized::{ColumnView, OptimizedDataFrame};
    use std::collections::HashMap;

    /// Trait common to supervised learning models (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::models::SupervisedModel` instead"
    )]
    pub use crate::ml::models::SupervisedModel;

    /// Linear Regression Model (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::models::linear::LinearRegression` instead"
    )]
    pub use crate::ml::models::linear::LinearRegression;

    /// Logistic Regression Model (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::models::linear::LogisticRegression` instead"
    )]
    pub use crate::ml::models::linear::LogisticRegression;

    /// Model selection module (backward compatibility)
    pub mod model_selection {
        use crate::error::Result;
        use crate::optimized::OptimizedDataFrame;

        /// Split dataset into training set and test set (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::models::train_test_split` instead"
        )]
        pub fn train_test_split(
            df: &OptimizedDataFrame,
            test_size: f64,
            random_state: Option<u64>,
        ) -> Result<(OptimizedDataFrame, OptimizedDataFrame)> {
            // Implementation without forwarding for now
            if test_size <= 0.0 || test_size >= 1.0 {
                return Err(crate::error::Error::InvalidInput(
                    "test_size must be between 0 and 1".into(),
                ));
            }

            let n_rows = df.row_count();
            let n_test = (n_rows as f64 * test_size).round() as usize;

            if n_test == 0 || n_test == n_rows {
                return Err(crate::error::Error::InvalidInput(format!(
                    "test_size {} would result in empty training or test set",
                    test_size
                )));
            }

            // Generate indices for training and test sets
            let train_indices: Vec<usize> = (0..(n_rows - n_test)).collect();
            let test_indices: Vec<usize> = ((n_rows - n_test)..n_rows).collect();

            // Use sample functionality in the optimized dataframe implementation
            // Create a fixed seed from the train/test indices to make sure we get consistent results
            let seed = 42;
            let train_data = df.sample(train_indices.len(), false, Some(seed))?;
            let test_data = df.sample(test_indices.len(), false, Some(seed + 1))?;

            Ok((train_data, test_data))
        }

        /// Model evaluation using K-fold cross-validation (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::models::evaluation::cross_val_score` instead"
        )]
        pub fn cross_val_score<M>(
            model: &M,
            df: &OptimizedDataFrame,
            target: &str,
            features: &[&str],
            k_folds: usize,
        ) -> Result<Vec<f64>>
        where
            M: crate::ml::models::SupervisedModel + Clone,
        {
            // This is a stub that returns an error, as the new API is different
            Err(crate::error::Error::InvalidOperation(
                "This function is deprecated. Please use `pandrs::ml::models::evaluation::cross_val_score` with the new API".into()
            ))
        }
    }

    /// Model persistence module (backward compatibility)
    pub mod model_persistence {
        use crate::error::Result;
        use std::path::Path;

        /// Model persistence trait (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::models::persistence::ModelPersistence` instead"
        )]
        pub trait ModelPersistence: Sized {
            /// Save model as a JSON file
            fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<()>;

            /// Load model from a JSON file
            fn load_model<P: AsRef<Path>>(path: P) -> Result<Self>;
        }
    }
}

#[allow(deprecated)]
pub mod anomaly_detection {
    //! Backward compatibility for anomaly detection

    use crate::error::Result;
    use crate::optimized::OptimizedDataFrame;

    /// Isolation Forest anomaly detection algorithm (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::anomaly::IsolationForest` instead"
    )]
    pub struct IsolationForest {
        // Internal implementation delegates to new version
        inner: crate::ml::anomaly::IsolationForest,
    }

    impl IsolationForest {
        /// Create a new IsolationForest instance (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::anomaly::IsolationForest::new` instead"
        )]
        pub fn new(
            n_estimators: usize,
            max_samples: Option<usize>,
            max_features: Option<f64>,
            contamination: f64,
            random_seed: Option<u64>,
        ) -> Self {
            let mut forest = crate::ml::anomaly::IsolationForest::new();
            forest.n_estimators = n_estimators;
            forest.max_samples = max_samples;
            forest.contamination = contamination;
            forest.random_seed = random_seed;

            IsolationForest { inner: forest }
        }

        /// Get anomaly scores (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::anomaly::IsolationForest::anomaly_scores` instead"
        )]
        pub fn anomaly_scores(&self) -> &[f64] {
            self.inner.anomaly_scores()
        }

        /// Get anomaly flags (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::anomaly::IsolationForest::labels` instead"
        )]
        pub fn labels(&self) -> &[i64] {
            self.inner.labels()
        }
    }

    /// Distance metric (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::clustering::DistanceMetric` instead"
    )]
    pub enum DistanceMetric {
        /// Euclidean distance
        Euclidean,
        /// Manhattan distance
        Manhattan,
        /// Cosine distance
        Cosine,
    }

    impl From<DistanceMetric> for crate::ml::clustering::DistanceMetric {
        fn from(metric: DistanceMetric) -> Self {
            match metric {
                DistanceMetric::Euclidean => crate::ml::clustering::DistanceMetric::Euclidean,
                DistanceMetric::Manhattan => crate::ml::clustering::DistanceMetric::Manhattan,
                DistanceMetric::Cosine => crate::ml::clustering::DistanceMetric::Cosine,
            }
        }
    }

    /// LOF (Local Outlier Factor) anomaly detection algorithm (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::anomaly::LocalOutlierFactor` instead"
    )]
    pub struct LocalOutlierFactor {
        // Internal implementation delegates to new version
        inner: crate::ml::anomaly::LocalOutlierFactor,
    }

    impl LocalOutlierFactor {
        /// Create a new LocalOutlierFactor instance (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::anomaly::LocalOutlierFactor::new` instead"
        )]
        pub fn new(n_neighbors: usize, contamination: f64, metric: DistanceMetric) -> Self {
            let lof = crate::ml::anomaly::LocalOutlierFactor::new(n_neighbors)
                .contamination(contamination);

            LocalOutlierFactor { inner: lof }
        }
    }

    /// One-Class SVM anomaly detection algorithm (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::anomaly::OneClassSVM` instead"
    )]
    pub struct OneClassSVM {
        // Internal implementation delegates to new version
        inner: crate::ml::anomaly::OneClassSVM,
    }

    impl OneClassSVM {
        /// Create a new OneClassSVM instance (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::anomaly::OneClassSVM::new` instead"
        )]
        pub fn new(nu: f64, gamma: f64, max_iter: usize, tol: f64) -> Self {
            let svm = crate::ml::anomaly::OneClassSVM::new().nu(nu).gamma(gamma);

            OneClassSVM { inner: svm }
        }
    }
}

/// Pipeline module (backward compatibility)
#[allow(deprecated)]
pub mod pipeline {
    use crate::error::Result;
    use crate::optimized::OptimizedDataFrame;

    /// Transformer trait (backward compatibility)
    #[deprecated(
        since = "0.1.0-alpha.2",
        note = "Use `pandrs::ml::pipeline::PipelineTransformer` instead"
    )]
    pub trait Transformer {
        /// Fit model to data
        fn fit(&mut self, df: &OptimizedDataFrame) -> Result<()>;

        /// Transform data
        fn transform(&self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame>;

        /// Fit and transform in one step
        fn fit_transform(&mut self, df: &OptimizedDataFrame) -> Result<OptimizedDataFrame> {
            self.fit(df)?;
            self.transform(df)
        }
    }
}

/// Metrics module (backward compatibility)
pub mod metrics {
    /// Regression metrics (backward compatibility)
    pub mod regression {
        use crate::error::Result;

        /// Mean squared error (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::regression::mean_squared_error` instead"
        )]
        pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::mean_squared_error(y_true, y_pred)
        }

        /// Mean absolute error (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::regression::mean_absolute_error` instead"
        )]
        pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::mean_absolute_error(y_true, y_pred)
        }

        /// Root mean squared error (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::regression::root_mean_squared_error` instead"
        )]
        pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::root_mean_squared_error(y_true, y_pred)
        }

        /// R² score (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::regression::r2_score` instead"
        )]
        pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::r2_score(y_true, y_pred)
        }

        /// Explained variance score (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::regression::explained_variance_score` instead"
        )]
        pub fn explained_variance_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::explained_variance_score(y_true, y_pred)
        }
    }

    /// Classification metrics (backward compatibility)
    pub mod classification {
        use crate::error::Result;

        /// Accuracy score (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::classification::accuracy_score` instead"
        )]
        pub fn accuracy_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::accuracy_score(y_true, y_pred)
        }

        /// Precision score (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::classification::precision_score` instead"
        )]
        pub fn precision_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::precision_score(y_true, y_pred)
        }

        /// Recall score (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::classification::recall_score` instead"
        )]
        pub fn recall_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::recall_score(y_true, y_pred)
        }

        /// F1 score (backward compatibility)
        #[deprecated(
            since = "0.1.0-alpha.2",
            note = "Use `pandrs::ml::metrics::classification::f1_score` instead"
        )]
        pub fn f1_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::f1_score(y_true, y_pred)
        }
    }
}
