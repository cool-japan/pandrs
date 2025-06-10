//! Machine learning model evaluation metrics
//!
//! This module provides metrics for evaluating the performance of machine learning models,
//! including regression metrics and classification metrics.

pub mod classification;
pub mod regression;

// Add backward compatibility layer
#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use regression and classification modules directly instead"
)]
pub use regression::{
    explained_variance_score, mean_absolute_error, mean_squared_error, r2_score,
    root_mean_squared_error,
};

#[deprecated(
    since = "0.1.0-alpha.2",
    note = "Use regression and classification modules directly instead"
)]
pub use classification::{accuracy_score, f1_score, precision_score, recall_score};
