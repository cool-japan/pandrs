//! Regression tests for `src/ml/models/{ensemble,tree,mod,evaluation}.rs` and
//! `src/ml/backward_compat.rs` (Wave 2, ml-ensemble-trees task).
//!
//! Pins down behavior that was previously wrong:
//!
//! * `RandomForestClassifier`/`RandomForestRegressor` bootstrap sampling was
//!   an arithmetic progression (`(seed*1103515245 + i*12345) % n`), not a
//!   real i.i.d.-with-replacement draw — see
//!   `ml::models::ensemble::tests::test_classifier_bootstrap_indices_distinct_fraction_matches_theory`
//!   (and its regressor/cross-tree-variance siblings) inside
//!   `src/ml/models/ensemble.rs` for the precise statistical check against
//!   the private `bootstrap_indices` method; this file instead checks the
//!   *consequence* at the public API level: a bagged forest's predictions
//!   must actually depend on how many (differently bootstrapped) trees it
//!   averages over.
//! * `GradientBoosting{Regressor,Classifier}::fit` leaked the target column
//!   into the per-iteration residual trees' feature set (it never dropped
//!   `target_column` before fitting on `_residual`), so a fitted model could
//!   not `predict` on a `DataFrame` that lacks the target column at all —
//!   exactly the shape of data prediction is supposed to run on.
//! * `models::train_test_split` (and the deprecated
//!   `backward_compat::models::model_selection::train_test_split`) produced
//!   train/test sets that were not an actual partition of the input rows:
//!   `train_test_split` ignored `shuffle`/`random_seed` entirely (a fixed
//!   sequential split), and the backward-compat version drew train and test
//!   as two *independent* random samples that could (and did) overlap.
//! * `cross_validate` on `DecisionTreeRegressor` and all four
//!   `RandomForest{Classifier,Regressor}`/`GradientBoosting{Regressor,Classifier}`
//!   combinations unconditionally returned `Ok(vec![])` — zero fold scores,
//!   silently swallowed by any caller that only checked `is_ok()`.

use pandrs::dataframe::DataFrame;
use pandrs::ml::models::ensemble::{
    GradientBoostingClassifier, GradientBoostingConfig, GradientBoostingConfigBuilder,
    GradientBoostingRegressor, RandomForestClassifier, RandomForestConfig, RandomForestRegressor,
};
use pandrs::ml::models::tree::{DecisionTreeConfig, DecisionTreeRegressor};
use pandrs::ml::models::{train_test_split, ModelEvaluator, SupervisedModel};
use pandrs::series::Series;
use std::collections::HashSet;

fn df_from_columns(columns: &[(&str, Vec<f64>)]) -> DataFrame {
    let mut df = DataFrame::new();
    for (name, values) in columns {
        df.add_column(
            name.to_string(),
            Series::new(values.clone(), Some(name.to_string())).expect("Series::new"),
        )
        .expect("add_column");
    }
    df
}

/// A single-tree "forest" (`bootstrap=false`, `max_features` forced to use
/// every feature deterministically) must be exactly reproduced by fitting
/// more trees with the same config: with no bootstrap and no feature
/// subsampling, CART's split search is fully deterministic, so every tree in
/// the ensemble is bit-identical and averaging them changes nothing.
///
/// The contrasting case — `bootstrap=true` — must NOT have this property:
/// averaging many independently-bootstrapped trees has to differ measurably
/// from a single tree's fit. Before the fix, "bootstrap" was an arithmetic
/// progression with no real row-level variance between trees, so this
/// contrast would not have held (a 1-tree and a 40-tree "bootstrapped"
/// forest would have been suspiciously close to each other too).
#[test]
fn random_forest_bootstrap_changes_predictions_but_determinism_holds_without_it() {
    let n = 60usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x1
        .iter()
        .map(|&v| (v * 0.37).sin() * 5.0 + (v * 0.11).cos() * 3.0)
        .collect();
    let df = df_from_columns(&[("x1", x1), ("y", y)]);

    // -- Control: bootstrap=false, all features considered => deterministic.
    let deterministic_cfg = RandomForestConfig {
        max_features: Some(1), // the only feature there is
        bootstrap: false,
        random_seed: Some(1),
        ..RandomForestConfig::default()
    };
    let mut rf_one = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 1,
        ..deterministic_cfg.clone()
    });
    rf_one.fit(&df, "y").expect("fit should succeed");
    let mut rf_forty = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 40,
        ..deterministic_cfg
    });
    rf_forty.fit(&df, "y").expect("fit should succeed");

    let pred_one = rf_one.predict(&df).expect("predict should succeed");
    let pred_forty = rf_forty.predict(&df).expect("predict should succeed");
    for (a, b) in pred_one.iter().zip(&pred_forty) {
        assert!(
            (a - b).abs() < 1e-9,
            "bootstrap=false must make every tree identical, so a 1-tree and \
             40-tree forest must predict identically; got {} vs {}",
            a,
            b
        );
    }

    // -- Real bootstrap: predictions must now depend on n_estimators.
    let bootstrapped_cfg = RandomForestConfig {
        bootstrap: true,
        random_seed: Some(1),
        ..RandomForestConfig::default()
    };
    let mut rf_one_bs = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 1,
        ..bootstrapped_cfg.clone()
    });
    rf_one_bs.fit(&df, "y").expect("fit should succeed");
    let mut rf_forty_bs = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 40,
        ..bootstrapped_cfg
    });
    rf_forty_bs.fit(&df, "y").expect("fit should succeed");

    let pred_one_bs = rf_one_bs.predict(&df).expect("predict should succeed");
    let pred_forty_bs = rf_forty_bs.predict(&df).expect("predict should succeed");
    let max_diff = pred_one_bs
        .iter()
        .zip(&pred_forty_bs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff > 0.05,
        "averaging 40 differently-bootstrapped trees must differ measurably from a single \
         tree's fit (got max |diff| = {}); a fake bootstrap with no real row-level variance \
         would make every tree in the forest nearly the same tree",
        max_diff
    );
}

/// `GradientBoostingRegressor::predict` must work on a `DataFrame` that has
/// only the feature columns, never having seen the target column at all.
/// Before the fix, `residual_data` kept the original target column, which
/// `DecisionTreeRegressor::fit` then treated as an ordinary (extremely
/// predictive) feature; predicting on unlabeled data failed outright because
/// the fitted trees expected a "y" column that legitimately unlabeled data
/// will never have.
#[test]
fn gradient_boosting_regressor_predicts_on_frame_without_target_column() {
    let n = 40usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(&x2)
        .map(|(&a, &b)| 2.0 * a - 0.5 * b + 3.0)
        .collect();
    let train_df = df_from_columns(&[("x1", x1), ("x2", x2), ("y", y)]);

    let mut gb = GradientBoostingRegressor::new(
        GradientBoostingConfigBuilder::new()
            .n_estimators(15)
            .max_depth(3)
            .build(),
    );
    gb.fit(&train_df, "y").expect("fit should succeed");

    // Genuinely unlabeled data: no "y" column anywhere in scope.
    let unlabeled = df_from_columns(&[
        ("x1", vec![1.0, 5.0, 12.0, 30.0]),
        ("x2", vec![2.0, 8.0, 1.0, 4.0]),
    ]);
    let preds = gb
        .predict(&unlabeled)
        .expect("predict on unlabeled data must succeed, not fail with 'Column y not found'");
    assert_eq!(preds.len(), 4);
    assert!(
        preds.iter().all(|p| p.is_finite()),
        "predictions must be finite numbers: {:?}",
        preds
    );
}

/// Same target-leak fix, for the per-class residual trees inside
/// `GradientBoostingClassifier::fit`.
#[test]
fn gradient_boosting_classifier_predicts_on_frame_without_target_column() {
    let n = 40usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x1
        .iter()
        .map(|&v| if v < 20.0 { 0.0 } else { 1.0 })
        .collect();
    let train_df = df_from_columns(&[("x1", x1), ("y", y)]);

    let mut gb = GradientBoostingClassifier::new(
        GradientBoostingConfigBuilder::new()
            .n_estimators(10)
            .max_depth(2)
            .build(),
    );
    gb.fit(&train_df, "y").expect("fit should succeed");

    let unlabeled = df_from_columns(&[("x1", vec![0.0, 10.0, 25.0, 39.0])]);
    let preds = gb
        .predict(&unlabeled)
        .expect("predict on unlabeled data must succeed");
    assert_eq!(preds.len(), 4);
}

/// `models::train_test_split` with `shuffle=true` and a `random_seed` must
/// return a genuine partition: every original row appears in exactly one of
/// train/test, and the split is reproducible for a fixed seed.
#[test]
fn train_test_split_is_a_true_disjoint_partition_and_is_seed_reproducible() {
    let n = 97usize; // deliberately not evenly divisible by anything tidy
    let id: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let df = df_from_columns(&[("id", id)]);

    let (train_df, test_df) =
        train_test_split(&df, 0.3, true, Some(123)).expect("split should succeed");

    let train_ids = train_df
        .get_column_numeric_values("id")
        .expect("id column should be numeric");
    let test_ids = test_df
        .get_column_numeric_values("id")
        .expect("id column should be numeric");

    assert_eq!(
        train_ids.len() + test_ids.len(),
        n,
        "train + test row counts must cover every original row exactly once"
    );

    let train_set: HashSet<i64> = train_ids.iter().map(|&v| v.round() as i64).collect();
    let test_set: HashSet<i64> = test_ids.iter().map(|&v| v.round() as i64).collect();
    assert_eq!(train_set.len(), train_ids.len(), "train ids must be unique");
    assert_eq!(test_set.len(), test_ids.len(), "test ids must be unique");
    assert!(
        train_set.is_disjoint(&test_set),
        "train and test sets must not overlap -- previously two independent shuffles could \
         (and did) share rows: shared ids = {:?}",
        train_set.intersection(&test_set).collect::<Vec<_>>()
    );

    let union: HashSet<i64> = train_set.union(&test_set).cloned().collect();
    let expected: HashSet<i64> = (0..n as i64).collect();
    assert_eq!(
        union, expected,
        "train union test must equal the full original id set"
    );

    // The split must not just happen to be a disjoint partition by luck of a
    // sequential split -- it must actually be shuffled.
    let is_sequential_prefix_split = train_ids
        == (0..(n - test_ids.len()))
            .map(|i| i as f64)
            .collect::<Vec<_>>();
    assert!(
        !is_sequential_prefix_split,
        "shuffle=true must not degenerate into the old sequential prefix/suffix split"
    );

    // Same seed => same split (reproducibility).
    let (train_df_2, test_df_2) =
        train_test_split(&df, 0.3, true, Some(123)).expect("split should succeed");
    assert_eq!(
        train_df_2.get_column_numeric_values("id").unwrap(),
        train_ids,
        "the same random_seed must reproduce the same train split"
    );
    assert_eq!(
        test_df_2.get_column_numeric_values("id").unwrap(),
        test_ids,
        "the same random_seed must reproduce the same test split"
    );
}

/// The deprecated `backward_compat::models::model_selection::train_test_split`
/// must also return a disjoint partition (it used to draw train and test as
/// two independent random samples of the *whole* dataset, which could and
/// did overlap) and must honor `random_state` instead of a hardcoded seed.
#[test]
#[allow(deprecated)]
fn backward_compat_train_test_split_is_disjoint_and_honors_random_state() {
    use pandrs::ml::backward_compat::models::model_selection::train_test_split as compat_split;
    use pandrs::optimized::OptimizedDataFrame;

    let n = 50usize;
    let mut df = OptimizedDataFrame::new();
    df.add_float_column("id", (0..n).map(|i| i as f64).collect())
        .expect("add_float_column");

    let (train_df, test_df) = compat_split(&df, 0.4, Some(99)).expect("split should succeed");
    assert_eq!(train_df.row_count() + test_df.row_count(), n);

    let train_ids: Vec<f64> = (0..train_df.row_count())
        .map(|i| {
            train_df
                .column("id")
                .expect("id column")
                .as_float64()
                .expect("float column")
                .get(i)
                .expect("value")
                .expect("non-null")
        })
        .collect();
    let test_ids: Vec<f64> = (0..test_df.row_count())
        .map(|i| {
            test_df
                .column("id")
                .expect("id column")
                .as_float64()
                .expect("float column")
                .get(i)
                .expect("value")
                .expect("non-null")
        })
        .collect();

    let train_set: HashSet<i64> = train_ids.iter().map(|&v| v.round() as i64).collect();
    let test_set: HashSet<i64> = test_ids.iter().map(|&v| v.round() as i64).collect();
    assert!(
        train_set.is_disjoint(&test_set),
        "backward-compat train_test_split must not overlap train and test: shared = {:?}",
        train_set.intersection(&test_set).collect::<Vec<_>>()
    );

    // random_state must actually be honored: a different seed should (with
    // overwhelming probability, for n=50) produce a different split.
    let (train_df_b, _test_df_b) = compat_split(&df, 0.4, Some(7)).expect("split should succeed");
    let train_ids_b: Vec<f64> = (0..train_df_b.row_count())
        .map(|i| {
            train_df_b
                .column("id")
                .expect("id column")
                .as_float64()
                .expect("float column")
                .get(i)
                .expect("value")
                .expect("non-null")
        })
        .collect();
    assert_ne!(
        train_ids, train_ids_b,
        "different random_state values must (overwhelmingly likely) produce different splits"
    );
}

/// `cross_validate` must return exactly `folds` real, distinct
/// [`pandrs::ml::models::ModelMetrics`] entries for every model in this
/// module, not the previous unconditional `Ok(vec![])`.
#[test]
fn cross_validate_returns_k_real_scores_for_every_model() {
    let n = 60usize;
    let folds = 4usize;

    // Regression dataset.
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y_reg: Vec<f64> = x1.iter().map(|&v| 2.0 * v + 1.0).collect();
    let reg_df = df_from_columns(&[("x1", x1.clone()), ("y", y_reg)]);

    // Classification dataset.
    let y_clf: Vec<f64> = x1
        .iter()
        .map(|&v| if v < 30.0 { 0.0 } else { 1.0 })
        .collect();
    let clf_df = df_from_columns(&[("x1", x1), ("y", y_clf)]);

    let tree_reg = DecisionTreeRegressor::new(DecisionTreeConfig::default());
    let scores = tree_reg
        .cross_validate(&reg_df, "y", folds)
        .expect("DecisionTreeRegressor cross_validate should succeed");
    assert_eq!(scores.len(), folds, "DecisionTreeRegressor");
    for m in &scores {
        assert!(m.get_metric("r2").is_some());
    }

    let rf_clf = RandomForestClassifier::new(RandomForestConfig {
        n_estimators: 5,
        ..RandomForestConfig::default()
    });
    let scores = rf_clf
        .cross_validate(&clf_df, "y", folds)
        .expect("RandomForestClassifier cross_validate should succeed");
    assert_eq!(scores.len(), folds, "RandomForestClassifier");
    for m in &scores {
        assert!(m.get_metric("accuracy").is_some());
    }

    let rf_reg = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 5,
        ..RandomForestConfig::default()
    });
    let scores = rf_reg
        .cross_validate(&reg_df, "y", folds)
        .expect("RandomForestRegressor cross_validate should succeed");
    assert_eq!(scores.len(), folds, "RandomForestRegressor");

    let gb_reg = GradientBoostingRegressor::new(GradientBoostingConfig {
        n_estimators: 5,
        ..GradientBoostingConfig::default()
    });
    let scores = gb_reg
        .cross_validate(&reg_df, "y", folds)
        .expect("GradientBoostingRegressor cross_validate should succeed");
    assert_eq!(scores.len(), folds, "GradientBoostingRegressor");

    let gb_clf = GradientBoostingClassifier::new(GradientBoostingConfig {
        n_estimators: 5,
        ..GradientBoostingConfig::default()
    });
    let scores = gb_clf
        .cross_validate(&clf_df, "y", folds)
        .expect("GradientBoostingClassifier cross_validate should succeed");
    assert_eq!(scores.len(), folds, "GradientBoostingClassifier");
}

/// `GradientBoostingRegressor` with `GBLoss::AbsoluteError` (LAD) must
/// actually learn: training MAE should decrease over boosting iterations.
/// This exercises the median initial prediction and the per-leaf
/// terminal-region median overwrite together -- if either were broken (e.g.
/// leaves left at CART's mean-of-sign-residuals value), the model would not
/// reliably improve on its own training loss.
#[test]
fn gradient_boosting_lad_loss_reduces_training_error() {
    use pandrs::ml::models::ensemble::GBLoss;

    let n = 50usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    // A target with one gross outlier -- exactly the situation LAD/absolute
    // error is supposed to be robust to, unlike squared error.
    let mut y: Vec<f64> = x1.iter().map(|&v| 0.5 * v + 3.0).collect();
    y[0] = 500.0;
    let df = df_from_columns(&[("x1", x1), ("y", y)]);

    let mut gb = GradientBoostingRegressor::new(
        GradientBoostingConfigBuilder::new()
            .n_estimators(25)
            .learning_rate(0.1)
            .max_depth(3)
            .loss(GBLoss::AbsoluteError)
            .build(),
    );
    gb.fit(&df, "y").expect("fit should succeed");

    let scores = gb.train_scores();
    assert_eq!(scores.len(), 25);
    assert!(
        scores.last().unwrap() < scores.first().unwrap(),
        "LAD training MAE should decrease over boosting iterations: {:?}",
        scores
    );
}

/// `RandomForestClassifier::oob_score()` must be a real accuracy estimate
/// computed from out-of-bag predictions (rows each tree did not see in its
/// bootstrap sample), not `None` or a fabricated value. With `bootstrap`
/// now a real per-tree random draw and enough trees, virtually every row
/// has at least one out-of-bag tree.
#[test]
fn random_forest_classifier_oob_score_is_plausible_on_separable_data() {
    let n = 60usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x1
        .iter()
        .map(|&v| if v < 30.0 { 0.0 } else { 1.0 })
        .collect();
    let df = df_from_columns(&[("x1", x1), ("y", y)]);

    let mut rf = RandomForestClassifier::new(RandomForestConfig {
        n_estimators: 150,
        bootstrap: true,
        oob_score: true,
        random_seed: Some(5),
        ..RandomForestConfig::default()
    });
    rf.fit(&df, "y")
        .expect("fit with oob_score=true and enough trees should succeed");

    let oob = rf.oob_score().expect(
        "oob_score() must be Some after fitting with oob_score=true: with 150 trees the \
         chance every row is in-bag for all of them is negligible",
    );
    assert!(
        (0.0..=1.0).contains(&oob),
        "OOB accuracy must be a valid fraction, got {}",
        oob
    );
    assert!(
        oob > 0.7,
        "OOB accuracy on cleanly-separable data should be reasonably high, got {}",
        oob
    );
}

/// Same property for the regressor: `oob_score()` is a real out-of-bag R².
#[test]
fn random_forest_regressor_oob_score_is_plausible() {
    let n = 60usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x1.iter().map(|&v| 2.0 * v + 1.0).collect();
    let df = df_from_columns(&[("x1", x1), ("y", y)]);

    let mut rf = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 150,
        bootstrap: true,
        oob_score: true,
        random_seed: Some(5),
        max_features: Some(1),
        ..RandomForestConfig::default()
    });
    rf.fit(&df, "y")
        .expect("fit with oob_score=true and enough trees should succeed");

    let oob = rf
        .oob_score()
        .expect("oob_score() must be Some after fitting with oob_score=true");
    assert!(
        oob > 0.5,
        "OOB R^2 on an easy linear relationship should be reasonably high, got {}",
        oob
    );
}

/// `oob_score=true` with `bootstrap=false` has no out-of-bag rows at all
/// (every tree sees every row), so `fit` must reject the combination
/// outright rather than silently reporting `None` or a meaningless score.
#[test]
fn oob_score_without_bootstrap_is_rejected() {
    let df = df_from_columns(&[
        ("x1", vec![1.0, 2.0, 3.0, 4.0]),
        ("y", vec![0.0, 0.0, 1.0, 1.0]),
    ]);

    let mut rf_clf = RandomForestClassifier::new(RandomForestConfig {
        n_estimators: 3,
        bootstrap: false,
        oob_score: true,
        ..RandomForestConfig::default()
    });
    assert!(
        rf_clf.fit(&df, "y").is_err(),
        "oob_score=true with bootstrap=false must be rejected (classifier)"
    );

    let mut rf_reg = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 3,
        bootstrap: false,
        oob_score: true,
        ..RandomForestConfig::default()
    });
    assert!(
        rf_reg.fit(&df, "y").is_err(),
        "oob_score=true with bootstrap=false must be rejected (regressor)"
    );
}

/// `n_jobs` must only change wall-clock time, never which trees get built:
/// each tree's bootstrap draw and split search are seeded solely from its
/// `tree_idx`, so fitting the same config/seed with `n_jobs=1` (sequential)
/// and `n_jobs=4` (a scoped rayon thread pool) must produce bit-identical
/// forests.
#[test]
fn n_jobs_does_not_change_which_trees_are_built() {
    let n = 80usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 13) % 17) as f64).collect();
    let y_clf: Vec<f64> = x1
        .iter()
        .map(|&v| if v < 40.0 { 0.0 } else { 1.0 })
        .collect();
    let y_reg: Vec<f64> = x1
        .iter()
        .zip(&x2)
        .map(|(&a, &b)| a * 0.4 + b * 1.1)
        .collect();
    let clf_df = df_from_columns(&[("x1", x1.clone()), ("x2", x2.clone()), ("y", y_clf)]);
    let reg_df = df_from_columns(&[("x1", x1), ("x2", x2), ("y", y_reg)]);

    let base = RandomForestConfig {
        n_estimators: 30,
        bootstrap: true,
        random_seed: Some(9),
        ..RandomForestConfig::default()
    };

    let mut rf_seq = RandomForestClassifier::new(RandomForestConfig {
        n_jobs: 1,
        ..base.clone()
    });
    rf_seq.fit(&clf_df, "y").expect("fit should succeed");
    let mut rf_par = RandomForestClassifier::new(RandomForestConfig {
        n_jobs: 4,
        ..base.clone()
    });
    rf_par.fit(&clf_df, "y").expect("fit should succeed");
    assert_eq!(
        rf_seq.predict(&clf_df).expect("predict should succeed"),
        rf_par.predict(&clf_df).expect("predict should succeed"),
        "n_jobs=1 vs n_jobs=4 must build the same forest (classifier)"
    );

    let mut reg_seq = RandomForestRegressor::new(RandomForestConfig {
        n_jobs: 1,
        ..base.clone()
    });
    reg_seq.fit(&reg_df, "y").expect("fit should succeed");
    let mut reg_par = RandomForestRegressor::new(RandomForestConfig { n_jobs: 4, ..base });
    reg_par.fit(&reg_df, "y").expect("fit should succeed");
    let seq_preds = reg_seq.predict(&reg_df).expect("predict should succeed");
    let par_preds = reg_par.predict(&reg_df).expect("predict should succeed");
    for (a, b) in seq_preds.iter().zip(&par_preds) {
        assert!(
            (a - b).abs() < 1e-12,
            "n_jobs=1 vs n_jobs=4 must build the same forest (regressor): {} vs {}",
            a,
            b
        );
    }
}

/// `DecisionTreeRegressor::feature_importances()` (previously had no
/// implementation at all -- the classifier's counterpart was never ported)
/// and `RandomForestRegressor::feature_importances()` (previously always
/// `None`, even though the underlying trees existed to average) must both
/// report real, normalized importances after fitting.
#[test]
fn regressor_feature_importances_are_real_not_none() {
    let n = 50usize;
    let x1: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x2: Vec<f64> = (0..n).map(|i| ((i * 3) % 5) as f64).collect();
    let y: Vec<f64> = x1
        .iter()
        .zip(&x2)
        .map(|(&a, &b)| 3.0 * a + 0.01 * b)
        .collect();
    let df = df_from_columns(&[("x1", x1), ("x2", x2), ("y", y)]);

    let mut tree = DecisionTreeRegressor::new(DecisionTreeConfig::default());
    tree.fit(&df, "y").expect("fit should succeed");
    let tree_importances = tree
        .feature_importances()
        .expect("DecisionTreeRegressor must report real feature importances after fitting");
    assert!(!tree_importances.is_empty());
    let tree_sum: f64 = tree_importances.values().sum();
    assert!(
        (tree_sum - 1.0).abs() < 0.01,
        "importances should be normalized to sum to ~1, got {}",
        tree_sum
    );
    // x1 dominates the target by construction (coefficient 3.0 vs 0.01), so
    // it must carry more importance than the near-irrelevant x2.
    assert!(
        tree_importances["x1"] > tree_importances["x2"],
        "x1 should dominate feature importance: {:?}",
        tree_importances
    );

    let mut rf = RandomForestRegressor::new(RandomForestConfig {
        n_estimators: 20,
        random_seed: Some(3),
        ..RandomForestConfig::default()
    });
    rf.fit(&df, "y").expect("fit should succeed");
    let rf_importances = rf
        .feature_importances()
        .expect("RandomForestRegressor must report real feature importances after fitting");
    assert!(!rf_importances.is_empty());
    let rf_sum: f64 = rf_importances.values().sum();
    assert!(
        (rf_sum - 1.0).abs() < 0.05,
        "forest importances should be normalized to sum to ~1, got {}",
        rf_sum
    );
}
