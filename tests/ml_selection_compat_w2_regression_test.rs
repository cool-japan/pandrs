//! Regression tests for the model_selection / sklearn_compat / automl fixes (Wave 2,
//! ml-selection-compat task).
//!
//! Covers, at the public-API level:
//!   - GridSearchCV really tunes hyperparameters (scores differ across combinations), which
//!     requires `SupervisedAdapter::set_params` to actually reach the wrapped model.
//!   - StratifiedKFold keeps per-fold class ratios balanced on class-sorted data, unlike plain
//!     KFold (demonstrated indirectly through `Scorer::RocAuc`, which now errors — rather than
//!     fabricating 0.5 — on a single-class fold).
//!   - TimeSeriesSplit-driven search runs end-to-end (the precise no-look-ahead index
//!     invariant is unit-tested directly against `compute_cv_fold` inside
//!     `src/ml/model_selection.rs`, since fold indices aren't part of the public API).
//!   - `SupervisedAdapter::set_params` round-trips into the fitted model's actual behavior, and
//!     rejects unknown parameter names instead of silently discarding them.

use pandrs::dataframe::DataFrame;
use pandrs::ml::models::ensemble::{RandomForestConfig, RandomForestRegressor};
use pandrs::ml::models::linear::LinearRegression;
use pandrs::ml::{
    CrossValidationStrategy, GridSearchCV, Scorer, SklearnPredictor, SupervisedAdapter,
};
use pandrs::series::Series;
use std::collections::HashMap;

fn df_single_column(name: &str, values: Vec<f64>) -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        name.to_string(),
        Series::new(values, Some(name.to_string())).expect("series creation should succeed"),
    )
    .expect("add_column should succeed");
    df
}

/// Grid search must actually tune hyperparameters — different combinations must produce
/// different cross-validation scores, and the combination that can represent the data better
/// must win.
///
/// `y = 10 + 2*x` has a clear nonzero intercept: `fit_intercept=true` should recover it
/// essentially perfectly (R² ≈ 1.0), while `fit_intercept=false` forces the fit through the
/// origin and cannot represent the true intercept, so its R² must be measurably worse. Before
/// the fix, `SupervisedAdapter::set_params` silently discarded every key but `target_col`, so
/// every combination fit the exact same untuned model and every combo's score was bit-identical
/// (`best_params_` was effectively the first combination in iteration order, never actually
/// evaluated against the alternatives).
#[test]
fn grid_search_cv_scores_differ_across_hyperparameter_combinations() {
    let n = 10i64;
    let x_vals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let y_vals: Vec<f64> = x_vals.iter().map(|&v| 10.0 + 2.0 * v).collect();

    let x = df_single_column("x", x_vals);
    let y = df_single_column("target", y_vals);

    let estimator: Box<dyn SklearnPredictor + Send + Sync> =
        Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));
    let mut param_grid = HashMap::new();
    param_grid.insert(
        "fit_intercept".to_string(),
        vec!["true".to_string(), "false".to_string()],
    );

    let mut search = GridSearchCV::new(estimator, param_grid)
        .with_cv(CrossValidationStrategy::KFold {
            n_splits: 2,
            shuffle: false,
            random_state: None,
        })
        // GridSearchCV defaults to R2 scoring, but set it explicitly for clarity.
        .with_scoring(Scorer::R2);
    search.fit(&x, &y).expect("grid search should succeed");

    let results = search.get_results().expect("results should be available");
    assert_eq!(
        results.cv_results_.len(),
        2,
        "both fit_intercept combinations should be evaluated"
    );

    let scores: Vec<f64> = results
        .cv_results_
        .iter()
        .map(|entry| entry.mean_test_score)
        .collect();
    assert!(
        (scores[0] - scores[1]).abs() > 1e-6,
        "fit_intercept=true and fit_intercept=false must score differently on data with a \
         genuine nonzero intercept — identical scores would mean set_params isn't reaching the \
         wrapped model at all, got scores {:?}",
        scores
    );

    assert_eq!(
        results
            .best_params_
            .get("fit_intercept")
            .map(|s| s.as_str()),
        Some("true"),
        "fit_intercept=true should be selected as the best hyperparameter: it can represent \
         the data's true nonzero intercept and the no-intercept fit cannot"
    );
    assert!(
        results.best_score_.is_some_and(|s| s > 0.9),
        "the winning combination should fit the (noiseless) linear data almost perfectly, got \
         {:?}",
        results.best_score_
    );
}

/// StratifiedKFold must keep per-fold class ratios balanced even when the input is sorted by
/// class — unlike plain contiguous `KFold`, which can hand an entire fold a single class.
///
/// Demonstrated indirectly through `Scorer::RocAuc`, which is undefined (and now correctly
/// returns `Err` rather than a fabricated `0.5`) whenever a fold's `y_true` contains only one
/// class. On 20 rows sorted class 0 then class 1 with 5 folds of size 4: plain `KFold` hands
/// folds {0-3, 4-7, 16-19} pure single-class blocks, so the whole grid search must error;
/// `StratifiedKFold` round-robins each class across folds (2 of each class per fold here), so
/// every fold is mixed and the search must succeed.
#[test]
fn stratified_kfold_avoids_single_class_folds_that_break_roc_auc() {
    let n = 20usize;
    let y_vals: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();
    let x_vals: Vec<f64> = y_vals.clone();

    let x = df_single_column("x", x_vals);
    let y = df_single_column("target", y_vals);

    let make_search = |cv: CrossValidationStrategy| {
        let estimator: Box<dyn SklearnPredictor + Send + Sync> =
            Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));
        GridSearchCV::new(estimator, HashMap::new())
            .with_cv(cv)
            .with_scoring(Scorer::RocAuc)
    };

    let mut plain_kfold_search = make_search(CrossValidationStrategy::KFold {
        n_splits: 5,
        shuffle: false,
        random_state: None,
    });
    assert!(
        plain_kfold_search.fit(&x, &y).is_err(),
        "plain KFold on class-sorted data should hand at least one fold a single class, and \
         RocAuc must error rather than fabricate a score for it"
    );

    let mut stratified_search = make_search(CrossValidationStrategy::StratifiedKFold {
        n_splits: 5,
        shuffle: false,
        random_state: None,
    });
    assert!(
        stratified_search.fit(&x, &y).is_ok(),
        "StratifiedKFold must keep every fold's class ratio balanced even on class-sorted \
         input, so RocAuc is always computable"
    );
}

/// End-to-end smoke test: a `TimeSeriesSplit`-driven search must complete without erroring and
/// produce a finite score. The precise no-look-ahead index invariant (every training index
/// strictly precedes its fold's test block, and `max_train_size` caps the window) is verified
/// directly against the private `compute_cv_fold` helper in
/// `src/ml/model_selection.rs`'s own test module, since raw fold indices aren't part of the
/// public API surface this integration test can reach.
#[test]
fn time_series_split_search_runs_end_to_end_without_lookahead_errors() {
    let n = 24usize;
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_vals = x_vals.clone();

    let x = df_single_column("x", x_vals);
    let y = df_single_column("target", y_vals);

    let estimator: Box<dyn SklearnPredictor + Send + Sync> =
        Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));
    let mut search = GridSearchCV::new(estimator, HashMap::new())
        .with_cv(CrossValidationStrategy::TimeSeriesSplit {
            n_splits: 4,
            max_train_size: None,
        })
        .with_scoring(Scorer::R2);

    search
        .fit(&x, &y)
        .expect("TimeSeriesSplit search should complete without look-ahead errors");
    let results = search.get_results().expect("results should be available");
    assert!(
        results.best_score_.is_some(),
        "a valid TimeSeriesSplit fold sequence over a monotonic series should produce a finite \
         score, not None"
    );
}

/// `SupervisedAdapter::set_params` must round-trip into the fitted model's actual behavior
/// (not be a no-op), and must reject unknown parameter names (matching scikit-learn's
/// `set_params`, which raises on invalid parameters) instead of silently discarding them.
#[test]
fn supervised_adapter_set_params_round_trip_changes_fitted_model() {
    let n = 30usize;
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .enumerate()
        .map(|(i, &v)| v + if i % 3 == 0 { 5.0 } else { -3.0 })
        .collect();

    let x = df_single_column("x", x_vals);
    let y = df_single_column("target", y_vals);

    let fit_and_predict = |n_estimators: &str| -> Vec<f64> {
        let mut adapter: Box<dyn SklearnPredictor + Send + Sync> =
            Box::new(SupervisedAdapter::new(
                RandomForestRegressor::new(RandomForestConfig::default()),
                "target",
            ));
        let mut params = HashMap::new();
        params.insert("n_estimators".to_string(), n_estimators.to_string());
        params.insert("random_seed".to_string(), "7".to_string());
        adapter
            .set_params(params)
            .expect("set_params with recognized keys should succeed");
        adapter.fit(&x, &y).expect("fit should succeed");
        adapter.predict(&x).expect("predict should succeed")
    };

    let preds_one_tree = fit_and_predict("1");
    let preds_many_trees = fit_and_predict("25");

    assert_eq!(preds_one_tree.len(), preds_many_trees.len());
    let total_abs_diff: f64 = preds_one_tree
        .iter()
        .zip(preds_many_trees.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        total_abs_diff > 1e-6,
        "n_estimators=1 vs n_estimators=25 must fit visibly different models — set_params must \
         really reach the wrapped RandomForestRegressor's config instead of being silently \
         discarded (previously ALL search trials fit the exact same untuned model)"
    );

    // Unknown keys must error rather than being silently ignored — this is the root-cause fix:
    // a search loop that passes a typo'd or model-inapplicable hyperparameter name must fail
    // loudly instead of quietly tuning nothing.
    let mut adapter: Box<dyn SklearnPredictor + Send + Sync> =
        Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));
    let mut bad_params = HashMap::new();
    bad_params.insert("this_is_not_a_real_param".to_string(), "123".to_string());
    assert!(
        adapter.set_params(bad_params).is_err(),
        "set_params with an unrecognized key must return Err, not silently succeed"
    );
}

/// `GridSearchCV` with `n_splits: 0` must return a clear `Err` instead of silently computing
/// `0.0 / 0.0 = NaN` as a fabricated mean score.
///
/// Before this fix, `GridSearchCV::cross_validate_params` computed `n_splits` and then looped
/// `for fold in 0..n_splits` — with `n_splits == 0` that loop body never runs, leaving
/// `fold_scores` empty, so `fold_scores.iter().sum() / fold_scores.len()` divided by zero and
/// silently stored a `NaN` "score" into `cv_results_` instead of erroring.
#[test]
fn grid_search_cv_errors_on_zero_splits_instead_of_producing_nan() {
    let x = df_single_column("x", vec![1.0, 2.0, 3.0, 4.0]);
    let y = df_single_column("target", vec![1.0, 2.0, 3.0, 4.0]);

    let estimator: Box<dyn SklearnPredictor + Send + Sync> =
        Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));
    let mut search =
        GridSearchCV::new(estimator, HashMap::new()).with_cv(CrossValidationStrategy::KFold {
            n_splits: 0,
            shuffle: false,
            random_state: None,
        });

    let result = search.fit(&x, &y);
    assert!(
        result.is_err(),
        "n_splits=0 must produce a clear Err, not a silently fabricated NaN score"
    );
}
