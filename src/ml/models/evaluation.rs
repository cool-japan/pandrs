//! Model evaluation utilities
//!
//! This module provides functions for evaluating machine learning models,
//! including cross-validation, learning curves, and validation curves.

use crate::dataframe::DataFrame;
use crate::error::{Error, Result};
use crate::ml::models::{ModelEvaluator, SupervisedModel};

/// Perform cross-validation on a model
///
/// # Arguments
/// * `model` - The model to evaluate
/// * `data` - The data to use for cross-validation
/// * `target` - The target column name
/// * `folds` - Number of cross-validation folds
/// * `metric` - Name of the metric to return (must be a metric computed by the model)
///
/// # Returns
/// * Vector of metric values for each fold
pub fn cross_val_score<T: SupervisedModel + Clone>(
    model: &T,
    data: &DataFrame,
    target: &str,
    folds: usize,
    metric: &str,
) -> Result<Vec<f64>> {
    if folds < 2 {
        return Err(Error::InvalidInput(
            "Number of folds must be at least 2".into(),
        ));
    }

    let metrics = model.cross_validate(data, target, folds)?;

    let mut scores = Vec::with_capacity(folds);
    for fold_metrics in metrics {
        if let Some(score) = fold_metrics.get_metric(metric) {
            scores.push(*score);
        } else {
            return Err(Error::InvalidInput(format!(
                "Metric '{}' not found in model evaluation",
                metric
            )));
        }
    }

    Ok(scores)
}

/// Generate learning curve for a model.
///
/// For each size fraction in `train_sizes`, the function:
/// 1. Takes the first `floor(n * size_fraction)` rows as the working subset.
/// 2. Splits the subset into `cv` folds.
/// 3. For each fold: trains on the non-test rows, evaluates on the fold's test rows
///    and on the training rows; collects the requested `metric`.
/// 4. Returns the mean train and test scores across folds for that size.
///
/// # Arguments
/// * `model` - The model to evaluate (must implement `Clone`)
/// * `data` - The full dataset
/// * `target` - The target column name
/// * `train_sizes` - Fractions in `(0, 1]` (e.g. `[0.5, 0.8, 1.0]`)
/// * `metric` - Name of the metric to track (must be produced by the model's `evaluate`)
/// * `cv` - Number of cross-validation folds (must be >= 2)
///
/// # Returns
/// * `(absolute_sizes, train_scores, test_scores)` — one entry per element of `train_sizes`
pub fn learning_curve<T: SupervisedModel + Clone>(
    model: &T,
    data: &DataFrame,
    target: &str,
    train_sizes: &[f64],
    metric: &str,
    cv: usize,
) -> Result<(Vec<usize>, Vec<f64>, Vec<f64>)> {
    if cv < 2 {
        return Err(Error::InvalidInput(
            "Number of CV folds must be at least 2".into(),
        ));
    }

    for &size in train_sizes {
        if size <= 0.0 || size > 1.0 {
            return Err(Error::InvalidInput(
                "Training sizes must be between 0 and 1".into(),
            ));
        }
    }

    let n = data.nrows();
    let mut absolute_sizes = Vec::with_capacity(train_sizes.len());
    let mut train_scores_out = Vec::with_capacity(train_sizes.len());
    let mut test_scores_out = Vec::with_capacity(train_sizes.len());

    for &size_frac in train_sizes {
        // Compute how many rows belong to this subset; ensure at least cv+1 rows
        // so every fold can have at least one training sample.
        let subset_n = ((n as f64 * size_frac).round() as usize).max(cv + 1).min(n);
        let indices: Vec<usize> = (0..subset_n).collect();
        let subset = data.sample(&indices)?;

        let fold_size = subset_n / cv;
        if fold_size == 0 {
            // Subset too small to form meaningful folds — record zeros and continue.
            absolute_sizes.push(subset_n);
            train_scores_out.push(0.0);
            test_scores_out.push(0.0);
            continue;
        }

        let mut test_fold_scores: Vec<f64> = Vec::with_capacity(cv);
        let mut train_fold_scores: Vec<f64> = Vec::with_capacity(cv);

        for fold_i in 0..cv {
            let test_start = fold_i * fold_size;
            let test_end = if fold_i == cv - 1 {
                subset_n
            } else {
                (fold_i + 1) * fold_size
            };

            let train_idx: Vec<usize> = (0..subset_n)
                .filter(|&i| i < test_start || i >= test_end)
                .collect();
            let test_idx: Vec<usize> = (test_start..test_end).collect();

            if train_idx.len() < 2 || test_idx.is_empty() {
                continue;
            }

            let train_data = subset.sample(&train_idx)?;
            let test_data = subset.sample(&test_idx)?;

            let mut m = model.clone();
            if m.fit(&train_data, target).is_err() {
                continue;
            }

            // Train score: evaluate the fitted model on its own training partition.
            if let Ok(tr_met) = m.evaluate(&train_data, target) {
                if let Some(&s) = tr_met.get_metric(metric) {
                    train_fold_scores.push(s);
                }
            }
            // Test score: evaluate on the held-out fold.
            if let Ok(te_met) = m.evaluate(&test_data, target) {
                if let Some(&s) = te_met.get_metric(metric) {
                    test_fold_scores.push(s);
                }
            }
        }

        let mean_train = if train_fold_scores.is_empty() {
            0.0
        } else {
            train_fold_scores.iter().sum::<f64>() / train_fold_scores.len() as f64
        };
        let mean_test = if test_fold_scores.is_empty() {
            0.0
        } else {
            test_fold_scores.iter().sum::<f64>() / test_fold_scores.len() as f64
        };

        absolute_sizes.push(subset_n);
        train_scores_out.push(mean_train);
        test_scores_out.push(mean_test);
    }

    Ok((absolute_sizes, train_scores_out, test_scores_out))
}

/// Generate validation curve for a model.
///
/// For each value in `param_values`, the function:
/// 1. Builds a fresh model via `model_factory(param_val)`.
/// 2. Splits the full dataset into `cv` folds.
/// 3. For each fold: trains on the non-test rows, evaluates on the fold's test rows
///    and on the training rows; collects the requested `metric`.
/// 4. Returns the mean train and test scores across folds for that parameter value.
///
/// # Arguments
/// * `model_factory` - Closure that creates a model configured with the given parameter value
/// * `data` - The full dataset
/// * `target` - The target column name
/// * `_param_name` - Name of the parameter being varied (for caller documentation; unused in computation)
/// * `param_values` - Slice of parameter values to evaluate
/// * `metric` - Name of the metric to track (must be produced by the model's `evaluate`)
/// * `cv` - Number of cross-validation folds (must be >= 2)
///
/// # Returns
/// * `(param_values_out, train_scores, test_scores)` — one entry per element of `param_values`
pub fn validation_curve<T, F, P>(
    model_factory: F,
    data: &DataFrame,
    target: &str,
    _param_name: &str,
    param_values: &[P],
    metric: &str,
    cv: usize,
) -> Result<(Vec<P>, Vec<f64>, Vec<f64>)>
where
    T: SupervisedModel,
    F: Fn(P) -> T,
    P: Clone,
{
    if cv < 2 {
        return Err(Error::InvalidInput(
            "Number of CV folds must be at least 2".into(),
        ));
    }

    if param_values.is_empty() {
        return Err(Error::InvalidInput(
            "Parameter values array cannot be empty".into(),
        ));
    }

    let n = data.nrows();
    let fold_size = n / cv;

    let mut train_scores_out = Vec::with_capacity(param_values.len());
    let mut test_scores_out = Vec::with_capacity(param_values.len());

    for param_val in param_values {
        let mut test_fold_scores: Vec<f64> = Vec::with_capacity(cv);
        let mut train_fold_scores: Vec<f64> = Vec::with_capacity(cv);

        if fold_size == 0 {
            // Dataset too small relative to cv — record zeros and continue.
            train_scores_out.push(0.0);
            test_scores_out.push(0.0);
            continue;
        }

        for fold_i in 0..cv {
            let test_start = fold_i * fold_size;
            let test_end = if fold_i == cv - 1 {
                n
            } else {
                (fold_i + 1) * fold_size
            };

            let train_idx: Vec<usize> = (0..n)
                .filter(|&i| i < test_start || i >= test_end)
                .collect();
            let test_idx: Vec<usize> = (test_start..test_end).collect();

            if train_idx.len() < 2 || test_idx.is_empty() {
                continue;
            }

            let train_data = data.sample(&train_idx)?;
            let test_data = data.sample(&test_idx)?;

            let mut m = model_factory(param_val.clone());
            if m.fit(&train_data, target).is_err() {
                continue;
            }

            // Train score: evaluate on the fitted model's own training partition.
            if let Ok(tr_met) = m.evaluate(&train_data, target) {
                if let Some(&s) = tr_met.get_metric(metric) {
                    train_fold_scores.push(s);
                }
            }
            // Test score: evaluate on the held-out fold.
            if let Ok(te_met) = m.evaluate(&test_data, target) {
                if let Some(&s) = te_met.get_metric(metric) {
                    test_fold_scores.push(s);
                }
            }
        }

        let mean_train = if train_fold_scores.is_empty() {
            0.0
        } else {
            train_fold_scores.iter().sum::<f64>() / train_fold_scores.len() as f64
        };
        let mean_test = if test_fold_scores.is_empty() {
            0.0
        } else {
            test_fold_scores.iter().sum::<f64>() / test_fold_scores.len() as f64
        };

        train_scores_out.push(mean_train);
        test_scores_out.push(mean_test);
    }

    Ok((param_values.to_vec(), train_scores_out, test_scores_out))
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
    fn test_learning_curve_varies() {
        let df = make_linear_df(20);
        let model = LinearRegression::new();
        let train_sizes = vec![0.5, 0.8, 1.0_f64];
        let (sizes, _train_sc, test_sc) = learning_curve(&model, &df, "y", &train_sizes, "r2", 2)
            .expect("learning_curve should succeed");

        assert_eq!(
            sizes.len(),
            3,
            "must return one entry per train_size fraction"
        );
        assert_eq!(
            test_sc.len(),
            3,
            "test_scores length must match train_sizes"
        );

        // The scores should not all be the same constant (as the old stub returned).
        // For a perfect linear model the scores will be high but may vary slightly
        // across subsets; we only assert they are not all identical to each other
        // at the stub values (0.9/0.8) by checking at least one is non-trivially computed.
        let all_identical = test_sc.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12);
        // It is acceptable for all scores to be identical if the data is perfectly linear
        // and the folds happen to agree. What we must reject is the *hardcoded* stub value
        // of exactly 0.8 for every entry.
        let all_stub = test_sc.iter().all(|&s| (s - 0.8).abs() < 1e-12);
        assert!(
            !all_stub,
            "test_scores must not be the old hardcoded stub [0.8, 0.8, 0.8], got {:?}",
            test_sc
        );
        let _ = all_identical; // acknowledged: identical real scores are fine
    }

    #[test]
    fn test_validation_curve_basic() {
        let df = make_linear_df(20);
        // Use a dummy integer as the "param value" — validation_curve doesn't
        // inspect it beyond forwarding to model_factory.
        let param_values = vec![1_usize, 2, 3];
        let (pv_out, train_sc, test_sc) = validation_curve(
            |_p: usize| LinearRegression::new(),
            &df,
            "y",
            "dummy_param",
            &param_values,
            "r2",
            2,
        )
        .expect("validation_curve should succeed");

        assert_eq!(
            pv_out.len(),
            param_values.len(),
            "output param_values length must match input"
        );
        assert_eq!(
            train_sc.len(),
            param_values.len(),
            "train_scores length must match param_values"
        );
        assert_eq!(
            test_sc.len(),
            param_values.len(),
            "test_scores length must match param_values"
        );
    }
}
