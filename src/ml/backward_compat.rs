//! Backward compatibility layer for the ML module
//!
//! This module provides backward compatibility for existing code that uses the old ML module structure.
//! It re-exports types and functions from the new module structure with appropriate deprecation notices.

#[allow(deprecated)]
pub mod models {
    //! Backward compatibility for ML models

    /// Trait common to supervised learning models (backward compatibility)
    #[deprecated(
        since = "0.1.0",
        note = "Use `pandrs::ml::models::SupervisedModel` instead"
    )]
    pub use crate::ml::models::SupervisedModel;

    /// Linear Regression Model (backward compatibility)
    #[deprecated(
        since = "0.1.0",
        note = "Use `pandrs::ml::models::linear::LinearRegression` instead"
    )]
    pub use crate::ml::models::linear::LinearRegression;

    /// Logistic Regression Model (backward compatibility)
    #[deprecated(
        since = "0.1.0",
        note = "Use `pandrs::ml::models::linear::LogisticRegression` instead"
    )]
    pub use crate::ml::models::linear::LogisticRegression;

    /// Model selection module (backward compatibility)
    pub mod model_selection {
        use crate::error::Result;
        use crate::optimized::OptimizedDataFrame;
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::Rng;
        use scirs2_core::random::SeedableRng;
        use scirs2_core::random::SliceRandom;

        /// Split dataset into training set and test set (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::models::train_test_split` instead"
        )]
        pub fn train_test_split(
            df: &OptimizedDataFrame,
            test_size: f64,
            random_state: Option<u64>,
        ) -> Result<(OptimizedDataFrame, OptimizedDataFrame)> {
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

            // One real seeded permutation of all row indices, sliced into a
            // test prefix and a train suffix. Previously `train_indices` and
            // `test_indices` were disjoint *by construction* (a plain
            // sequential split), but were then thrown away: the actual rows
            // returned came from two independent `df.sample(count, ..)`
            // calls with different seeds, each an unconstrained random draw
            // over *all* `n_rows` — so the two returned sets could (and,
            // empirically, did) share rows. Slicing one permutation instead
            // guarantees the returned train/test sets are an actual
            // partition, and `random_state` is honored instead of a fixed
            // seed of `42`.
            let mut rng = match random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => {
                    let mut seed_bytes = [0u8; 32];
                    scirs2_core::random::rng().fill_bytes(&mut seed_bytes);
                    StdRng::from_seed(seed_bytes)
                }
            };
            let mut indices: Vec<usize> = (0..n_rows).collect();
            indices.shuffle(&mut rng);

            let test_indices: Vec<usize> = indices[..n_test].to_vec();
            let train_indices: Vec<usize> = indices[n_test..].to_vec();

            let train_data = df.sample_rows(&train_indices)?;
            let test_data = df.sample_rows(&test_indices)?;

            Ok((train_data, test_data))
        }

        /// Model evaluation using K-fold cross-validation (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::models::evaluation::cross_val_score` instead"
        )]
        pub fn cross_val_score<M>(
            _model: &M,
            _df: &OptimizedDataFrame,
            _target: &str,
            _features: &[&str],
            _k_folds: usize,
        ) -> Result<Vec<f64>>
        where
            M: crate::ml::models::SupervisedModel + Clone,
        {
            // Deliberately `Err`, not a fabricated score: `SupervisedModel`
            // (and the real cross-validation loop it plugs into, see
            // `models::contiguous_kfold_cross_validate`) operates on
            // `crate::dataframe::DataFrame`, while this legacy signature
            // takes an `OptimizedDataFrame`. There is no
            // `OptimizedDataFrame -> DataFrame` converter anywhere in the
            // crate to bridge the two (checked: no `From`/`to_dataframe`
            // exists), and writing a general column-type-dispatching one is
            // outside this module's scope. Erroring honestly is preferable
            // to silently dropping `_features`/discarding fold results.
            Err(crate::error::Error::InvalidOperation(
                "This function is deprecated and cannot be bridged to the current API: \
                 SupervisedModel operates on `DataFrame`, not `OptimizedDataFrame`, and no \
                 conversion between the two exists. Please use \
                 `pandrs::ml::models::evaluation::cross_val_score` with a `DataFrame` instead"
                    .into(),
            ))
        }
    }

    /// Model persistence module (backward compatibility)
    pub mod model_persistence {
        use crate::error::Result;
        use std::path::Path;

        /// Model persistence trait (backward compatibility)
        #[deprecated(
            since = "0.1.0",
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

    /// Isolation Forest anomaly detection algorithm (backward compatibility)
    #[deprecated(
        since = "0.1.0",
        note = "Use `pandrs::ml::anomaly::IsolationForest` instead"
    )]
    pub struct IsolationForest {
        // Internal implementation delegates to new version
        inner: crate::ml::anomaly::IsolationForest,
    }

    impl IsolationForest {
        /// Create a new IsolationForest instance (backward compatibility).
        ///
        /// Returns `Err` if `max_features` is `Some(_)`: the new
        /// `pandrs::ml::anomaly::IsolationForest` has no per-split feature
        /// subsampling to forward it to (unlike its Random Forest cousin, it
        /// always splits on a feature chosen uniformly at random from *all*
        /// features), so silently dropping a caller's explicit
        /// `max_features` request would misrepresent what the forest
        /// actually did. Pass `None` to opt into the (only) supported
        /// behavior.
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::IsolationForest::new` instead"
        )]
        pub fn new(
            n_estimators: usize,
            max_samples: Option<usize>,
            max_features: Option<f64>,
            contamination: f64,
            random_seed: Option<u64>,
        ) -> crate::error::Result<Self> {
            if max_features.is_some() {
                return Err(crate::error::Error::InvalidInput(
                    "IsolationForest (backward-compat) does not support max_features: the \
                     current implementation always draws split features uniformly from all \
                     columns; pass None instead of silently ignoring the requested value"
                        .into(),
                ));
            }

            let mut forest = crate::ml::anomaly::IsolationForest::new();
            forest.n_estimators = n_estimators;
            forest.max_samples = max_samples;
            forest.contamination = contamination;
            forest.random_seed = random_seed;

            Ok(IsolationForest { inner: forest })
        }

        /// Get anomaly scores (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::IsolationForest::anomaly_scores` instead"
        )]
        pub fn anomaly_scores(&self) -> &[f64] {
            self.inner.anomaly_scores()
        }

        /// Get anomaly flags (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::IsolationForest::labels` instead"
        )]
        pub fn labels(&self) -> &[i64] {
            self.inner.labels()
        }
    }

    /// Distance metric (backward compatibility)
    #[deprecated(
        since = "0.1.0",
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
        since = "0.1.0",
        note = "Use `pandrs::ml::anomaly::LocalOutlierFactor` instead"
    )]
    pub struct LocalOutlierFactor {
        // Internal implementation delegates to new version
        inner: crate::ml::anomaly::LocalOutlierFactor,
    }

    impl LocalOutlierFactor {
        /// Create a new LocalOutlierFactor instance (backward compatibility).
        ///
        /// Returns `Err` for `metric` other than `Euclidean`: the current
        /// `pandrs::ml::anomaly::LocalOutlierFactor` always computes
        /// Euclidean distances (its `algorithm` field only ever selects a
        /// neighbor-search strategy, never a distance function), so
        /// `Manhattan`/`Cosine` cannot actually be honored.
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::LocalOutlierFactor::new` instead"
        )]
        pub fn new(
            n_neighbors: usize,
            contamination: f64,
            metric: DistanceMetric,
        ) -> crate::error::Result<Self> {
            if !matches!(metric, DistanceMetric::Euclidean) {
                return Err(crate::error::Error::InvalidInput(
                    "LocalOutlierFactor (backward-compat) only supports DistanceMetric::Euclidean: \
                     the current implementation always computes Euclidean distances"
                        .into(),
                ));
            }

            let lof = crate::ml::anomaly::LocalOutlierFactor::new(n_neighbors)
                .contamination(contamination);

            Ok(LocalOutlierFactor { inner: lof })
        }

        /// Get LOF anomaly scores (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::LocalOutlierFactor::anomaly_scores` instead"
        )]
        pub fn anomaly_scores(&self) -> &[f64] {
            self.inner.anomaly_scores()
        }

        /// Get anomaly labels (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::LocalOutlierFactor::labels` instead"
        )]
        pub fn labels(&self) -> &[i64] {
            self.inner.labels()
        }
    }

    /// One-Class SVM anomaly detection algorithm (backward compatibility)
    #[deprecated(
        since = "0.1.0",
        note = "Use `pandrs::ml::anomaly::OneClassSVM` instead"
    )]
    pub struct OneClassSVM {
        // Internal implementation delegates to new version
        inner: crate::ml::anomaly::OneClassSVM,
    }

    impl OneClassSVM {
        /// Create a new OneClassSVM instance (backward compatibility).
        ///
        /// `max_iter` and `tol` control an iterative QP solver's stopping
        /// criteria; the current `pandrs::ml::anomaly::OneClassSVM` is a
        /// closed-form mean-RBF-similarity estimator with no iterative
        /// solver to bound, so there is nothing to forward either to. `0`
        /// and `0.0` are accepted as "no preference" sentinels (matching the
        /// natural "unset" value for a non-negative count and a tolerance);
        /// any other value is a request this implementation cannot honor and
        /// is rejected rather than silently ignored.
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::OneClassSVM::new` instead"
        )]
        pub fn new(nu: f64, gamma: f64, max_iter: usize, tol: f64) -> crate::error::Result<Self> {
            if max_iter != 0 || tol != 0.0 {
                return Err(crate::error::Error::InvalidInput(
                    "OneClassSVM (backward-compat) does not support max_iter/tol: the current \
                     implementation has no iterative solver to bound; pass 0 and 0.0 instead of \
                     silently ignoring the requested values"
                        .into(),
                ));
            }

            let svm = crate::ml::anomaly::OneClassSVM::new().nu(nu).gamma(gamma);

            Ok(OneClassSVM { inner: svm })
        }

        /// Get anomaly scores (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::OneClassSVM::anomaly_scores` instead"
        )]
        pub fn anomaly_scores(&self) -> &[f64] {
            self.inner.anomaly_scores()
        }

        /// Get anomaly labels (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::anomaly::OneClassSVM::labels` instead"
        )]
        pub fn labels(&self) -> &[i64] {
            self.inner.labels()
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
        since = "0.1.0",
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
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::regression::mean_squared_error` instead"
        )]
        pub fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::mean_squared_error(y_true, y_pred)
        }

        /// Mean absolute error (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::regression::mean_absolute_error` instead"
        )]
        pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::mean_absolute_error(y_true, y_pred)
        }

        /// Root mean squared error (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::regression::root_mean_squared_error` instead"
        )]
        pub fn root_mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::root_mean_squared_error(y_true, y_pred)
        }

        /// R² score (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::regression::r2_score` instead"
        )]
        pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            crate::ml::metrics::regression::r2_score(y_true, y_pred)
        }

        /// Explained variance score (backward compatibility)
        #[deprecated(
            since = "0.1.0",
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
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::classification::accuracy_score` instead"
        )]
        pub fn accuracy_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::accuracy_score(y_true, y_pred)
        }

        /// Precision score (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::classification::precision_score` instead"
        )]
        pub fn precision_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::precision_score(y_true, y_pred)
        }

        /// Recall score (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::classification::recall_score` instead"
        )]
        pub fn recall_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::recall_score(y_true, y_pred)
        }

        /// F1 score (backward compatibility)
        #[deprecated(
            since = "0.1.0",
            note = "Use `pandrs::ml::metrics::classification::f1_score` instead"
        )]
        pub fn f1_score(y_true: &[bool], y_pred: &[bool]) -> Result<f64> {
            crate::ml::metrics::classification::f1_score(y_true, y_pred)
        }
    }
}
