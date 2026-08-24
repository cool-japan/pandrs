use super::*;
use crate::series::Series;

#[test]
fn test_parameter_distribution_sampling() {
    let mut rng = seeded_or_entropy_rng(Some(7));

    let uniform_int = ParameterDistribution::UniformInt { low: 1, high: 10 };
    let sample = uniform_int
        .sample(&mut rng)
        .expect("operation should succeed");
    let value: i64 = sample.parse().expect("operation should succeed");
    assert!(value >= 1 && value <= 10);

    let uniform_float = ParameterDistribution::UniformFloat {
        low: 0.0,
        high: 1.0,
    };
    let sample = uniform_float
        .sample(&mut rng)
        .expect("operation should succeed");
    let value: f64 = sample.parse().expect("operation should succeed");
    assert!(value >= 0.0 && value <= 1.0);

    let choice =
        ParameterDistribution::Choice(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    let sample = choice.sample(&mut rng).expect("operation should succeed");
    assert!(["a", "b", "c"].contains(&sample.as_str()));
}

#[test]
fn test_time_series_split_never_trains_on_future_rows() {
    let n = 20usize;
    let n_splits = 4usize;
    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(
            (0..n).map(|i| i as f64).collect::<Vec<f64>>(),
            Some("target".to_string()),
        )
        .expect("series creation should succeed"),
    )
    .expect("add_column should succeed");

    let cv = CrossValidationStrategy::TimeSeriesSplit {
        n_splits,
        max_train_size: None,
    };
    let mut previous_train_len = 0usize;
    for fold in 0..n_splits {
        let (train_indices, test_indices) =
            compute_cv_fold(&cv, n, &y, fold, n_splits).expect("fold should be computable");
        let min_test = *test_indices
            .iter()
            .min()
            .expect("test fold should be non-empty");
        assert!(
            train_indices.iter().all(|&i| i < min_test),
            "fold {fold}: train indices must all strictly precede the test block (no \
             look-ahead), got train={train_indices:?} test={test_indices:?}"
        );
        assert!(
            !train_indices.is_empty(),
            "fold {fold} should have a non-empty training window"
        );
        // Expanding window: each subsequent fold's training window is at least as large as
        // the previous one's (never shrinks).
        assert!(
            train_indices.len() >= previous_train_len,
            "fold {fold}: expanding window should not shrink train size run over run"
        );
        previous_train_len = train_indices.len();
    }

    // `max_train_size` caps the training window to the most recent rows instead of always
    // using the full history from index 0.
    let capped_cv = CrossValidationStrategy::TimeSeriesSplit {
        n_splits,
        max_train_size: Some(2),
    };
    let (train_indices, test_indices) = compute_cv_fold(&capped_cv, n, &y, n_splits - 1, n_splits)
        .expect("capped fold should be computable");
    assert_eq!(
        train_indices.len(),
        2,
        "max_train_size=2 should cap the training window to 2 rows, got {train_indices:?}"
    );
    let min_test = *test_indices.iter().min().unwrap();
    assert!(train_indices.iter().all(|&i| i < min_test));
}

#[test]
fn test_stratified_kfold_balances_class_ratio_on_sorted_data() {
    // 30 rows sorted by class: 20 of class 0 then 10 of class 1 (a 2:1 ratio) — the
    // scenario that breaks plain contiguous KFold (which would hand early folds only class
    // 0 and late folds only class 1).
    let n = 30usize;
    let n_splits = 5usize;
    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(
            (0..n)
                .map(|i| if i < 20 { 0.0 } else { 1.0 })
                .collect::<Vec<f64>>(),
            Some("target".to_string()),
        )
        .expect("series creation should succeed"),
    )
    .expect("add_column should succeed");

    let cv = CrossValidationStrategy::StratifiedKFold {
        n_splits,
        shuffle: false,
        random_state: None,
    };

    for fold in 0..n_splits {
        let (_train_indices, test_indices) = compute_cv_fold(&cv, n, &y, fold, n_splits)
            .expect("stratified fold should be computable");
        let labels = y
            .get_column::<f64>("target")
            .expect("target column")
            .as_f64()
            .expect("f64 values");
        let class0_in_fold = test_indices.iter().filter(|&&i| labels[i] == 0.0).count();
        let class1_in_fold = test_indices.iter().filter(|&&i| labels[i] == 1.0).count();
        // Overall ratio is 20:10 = 2:1; each fold of 6 rows should reflect that (4:2)
        // instead of being all-one-class.
        assert_eq!(
            class0_in_fold, 4,
            "fold {fold} should contain 4 class-0 rows (preserving the overall 2:1 ratio), \
             got {class0_in_fold}"
        );
        assert_eq!(
            class1_in_fold, 2,
            "fold {fold} should contain 2 class-1 rows (preserving the overall 2:1 ratio), \
             got {class1_in_fold}"
        );
    }
}

#[test]
fn test_log_uniform_rejects_non_positive_bounds() {
    let mut rng = seeded_or_entropy_rng(Some(1));
    let bad = ParameterDistribution::LogUniform {
        low: 0.0,
        high: 10.0,
    };
    assert!(
        bad.sample(&mut rng).is_err(),
        "LogUniform with low <= 0 must error instead of producing NaN"
    );

    let bad_negative = ParameterDistribution::LogUniform {
        low: -1.0,
        high: 10.0,
    };
    assert!(bad_negative.sample(&mut rng).is_err());
}

#[test]
fn test_scorer_r2() {
    let scorer = Scorer::R2;
    let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];

    let score = scorer
        .score(&y_true, &y_pred)
        .expect("operation should succeed");
    assert!(score > 0.9); // Should be high R²
}

#[test]
fn test_cross_validation_strategy() {
    let cv = CrossValidationStrategy::KFold {
        n_splits: 5,
        shuffle: true,
        random_state: Some(42),
    };

    match cv {
        CrossValidationStrategy::KFold { n_splits, .. } => assert_eq!(n_splits, 5),
        _ => panic!("Wrong CV strategy type"),
    }
}

#[test]
fn test_select_k_best() {
    let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);

    // Create test data
    let mut x = DataFrame::new();
    x.add_column(
        "feature1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("feature1".to_string()))
            .expect("operation should succeed"),
    )
    .expect("operation should succeed");
    x.add_column(
        "feature2".to_string(),
        Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("feature2".to_string()))
            .expect("operation should succeed"),
    )
    .expect("operation should succeed");
    x.add_column(
        "feature3".to_string(),
        Series::new(vec![0.1, 0.2, 0.3, 0.4, 0.5], Some("feature3".to_string()))
            .expect("operation should succeed"),
    )
    .expect("operation should succeed");

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(vec![3.0, 6.0, 9.0, 12.0, 15.0], Some("target".to_string()))
            .expect("operation should succeed"),
    )
    .expect("operation should succeed");

    // Fit and transform
    selector.fit(&x, &y).expect("operation should succeed");
    let selected = selector.transform(&x).expect("operation should succeed");

    // Should select 2 features
    assert_eq!(selected.column_names().len(), 2);
}

#[test]
fn test_select_k_best_preserves_original_column_order() {
    // Column order (by index): feature_a (0), feature_b (1), feature_c (2).
    // F-statistic order (descending): feature_c (r=1 exactly => F=+inf) > feature_a (strong
    // but imperfect correlation => large finite F) > feature_b (constant => F=0).
    // With k=2 the selected set is {feature_a, feature_c}. `transform` must emit them in their
    // ORIGINAL column order [feature_a, feature_c] — not score-rank order [feature_c,
    // feature_a], which was the previous (incorrect) behavior.
    let n = 10usize;
    let y_vals: Vec<f64> = (1..=n).map(|i| i as f64).collect();

    let feature_a_vals: Vec<f64> = y_vals
        .iter()
        .enumerate()
        .map(|(i, &v)| v + if i % 2 == 0 { 0.01 } else { -0.01 })
        .collect();
    let feature_b_vals: Vec<f64> = vec![7.0; n]; // constant => F = 0, always excluded
    let feature_c_vals: Vec<f64> = y_vals.clone(); // r = 1 exactly => F = +inf

    let mut x = DataFrame::new();
    x.add_column(
        "feature_a".to_string(),
        Series::new(feature_a_vals, Some("feature_a".to_string())).expect("series"),
    )
    .expect("add_column");
    x.add_column(
        "feature_b".to_string(),
        Series::new(feature_b_vals, Some("feature_b".to_string())).expect("series"),
    )
    .expect("add_column");
    x.add_column(
        "feature_c".to_string(),
        Series::new(feature_c_vals, Some("feature_c".to_string())).expect("series"),
    )
    .expect("add_column");

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(y_vals, Some("target".to_string())).expect("series"),
    )
    .expect("add_column");

    let mut selector = SelectKBest::new(ScoreFunction::FRegression, 2);
    selector.fit(&x, &y).expect("fit should succeed");
    let selected = selector.transform(&x).expect("transform should succeed");

    assert_eq!(
        selected.column_names().to_vec(),
        vec!["feature_a".to_string(), "feature_c".to_string()],
        "transform must emit selected columns in their ORIGINAL column order, not score-rank \
         order (feature_c has the higher F-statistic but the higher column index)"
    );
}

#[test]
fn test_select_k_best_transform_rejects_mismatched_columns() {
    let mut x = DataFrame::new();
    x.add_column(
        "feature1".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("feature1".to_string())).expect("series"),
    )
    .expect("add_column");
    x.add_column(
        "feature2".to_string(),
        Series::new(vec![2.0, 4.0, 6.0, 8.0], Some("feature2".to_string())).expect("series"),
    )
    .expect("add_column");

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("target".to_string())).expect("series"),
    )
    .expect("add_column");

    let mut selector = SelectKBest::new(ScoreFunction::FRegression, 1);
    selector.fit(&x, &y).expect("fit should succeed");

    // A DataFrame missing a column the selector was fitted on must be rejected rather than
    // silently indexing into the wrong columns by position.
    let mut x_wrong = DataFrame::new();
    x_wrong
        .add_column(
            "feature1".to_string(),
            Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("feature1".to_string())).expect("series"),
        )
        .expect("add_column");
    assert!(
        selector.transform(&x_wrong).is_err(),
        "transform on a DataFrame whose columns don't match fit-time columns must error \
         instead of silently transforming the wrong columns"
    );
}

// ── ROC AUC tests ────────────────────────────────────────────────────────

#[test]
fn test_roc_auc_perfect() {
    // Perfect ranking: all positives have strictly higher scores than all negatives
    let y_true = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let y_pred = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
    let scorer = Scorer::RocAuc;
    let auc = scorer.score(&y_true, &y_pred).expect("should compute AUC");
    assert!(
        (auc - 1.0).abs() < 1e-9,
        "Expected AUC = 1.0 for perfect ranking, got {auc}"
    );
}

#[test]
fn test_roc_auc_random() {
    // Tied scores between one positive and one negative → AUC = 0.5
    let y_true = vec![0.0, 1.0];
    let y_pred = vec![0.5, 0.5];
    let scorer = Scorer::RocAuc;
    let auc = scorer.score(&y_true, &y_pred).expect("should compute AUC");
    assert!(
        (auc - 0.5).abs() < 1e-9,
        "Expected AUC = 0.5 for tied scores, got {auc}"
    );
}

#[test]
fn test_roc_auc_inverted() {
    // Worst-case ranking: all positives have strictly lower scores than all negatives
    let y_true = vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let y_pred = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
    let scorer = Scorer::RocAuc;
    let auc = scorer.score(&y_true, &y_pred).expect("should compute AUC");
    assert!(
        auc.abs() < 1e-9,
        "Expected AUC = 0.0 for inverted ranking, got {auc}"
    );
}

// ── Chi-square tests ──────────────────────────────────────────────────────

#[test]
fn test_chi2_scores_correlated() {
    // feature1 is perfectly correlated to the target (target * 2);
    // feature2 is a constant — its chi2 score should be zero.
    let n = 20usize;
    let target_vals: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect(); // alternating 0/1
    let feat1_vals: Vec<f64> = target_vals.iter().map(|&v| v * 2.0).collect();
    let feat2_vals: Vec<f64> = vec![1.0; n]; // constant

    let mut x = DataFrame::new();
    x.add_column(
        "feature1".to_string(),
        Series::new(feat1_vals, Some("feature1".to_string()))
            .expect("series creation should succeed"),
    )
    .expect("add column should succeed");
    x.add_column(
        "feature2".to_string(),
        Series::new(feat2_vals, Some("feature2".to_string()))
            .expect("series creation should succeed"),
    )
    .expect("add column should succeed");

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(target_vals, Some("target".to_string()))
            .expect("series creation should succeed"),
    )
    .expect("add column should succeed");

    let selector = SelectKBest::new(ScoreFunction::Chi2, 1);
    let scores = selector.chi2_scores(&x, &y).expect("chi2 should succeed");
    assert_eq!(scores.len(), 2);
    // Correlated feature must score higher than constant feature
    assert!(
        scores[0] > scores[1],
        "Correlated feature chi2 ({}) should exceed constant feature chi2 ({})",
        scores[0],
        scores[1]
    );
}

// ── Mutual information tests ──────────────────────────────────────────────

#[test]
fn test_mutual_info_scores_vary() {
    // feature1 is perfectly correlated to target; feature2 is constant.
    // MI(feature1, target) should be strictly greater than MI(feature2, target).
    let n = 30usize;
    let target_vals: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect(); // classes 0/1/2
    let feat1_vals: Vec<f64> = target_vals.clone(); // perfect correlation
    let feat2_vals: Vec<f64> = vec![0.0; n]; // constant — zero MI

    let mut x = DataFrame::new();
    x.add_column(
        "correlated".to_string(),
        Series::new(feat1_vals, Some("correlated".to_string()))
            .expect("series creation should succeed"),
    )
    .expect("add column should succeed");
    x.add_column(
        "constant".to_string(),
        Series::new(feat2_vals, Some("constant".to_string()))
            .expect("series creation should succeed"),
    )
    .expect("add column should succeed");

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(target_vals, Some("target".to_string()))
            .expect("series creation should succeed"),
    )
    .expect("add column should succeed");

    let selector = SelectKBest::new(ScoreFunction::MutualInfoClassification, 1);
    let scores = selector
        .mutual_info_scores(&x, &y)
        .expect("mutual info should succeed");
    assert_eq!(scores.len(), 2);
    // Correlated feature must have strictly higher MI than constant feature
    assert!(
        scores[0] > scores[1],
        "Correlated feature MI ({}) should exceed constant feature MI ({})",
        scores[0],
        scores[1]
    );
}

#[test]
fn test_f_regression_is_real_f_statistic() {
    // Perfectly correlated feature -> r² = 1 -> F = +inf; uncorrelated-ish -> finite.
    let selector = SelectKBest::new(ScoreFunction::FRegression, 1);

    let mut x = DataFrame::new();
    x.add_column(
        "perfect".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("perfect".to_string())).unwrap(),
    )
    .unwrap();
    x.add_column(
        "weak".to_string(),
        Series::new(vec![1.0, 0.0, 1.0, 0.0, 1.0], Some("weak".to_string())).unwrap(),
    )
    .unwrap();

    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(vec![2.0, 4.0, 6.0, 8.0, 10.0], Some("target".to_string())).unwrap(),
    )
    .unwrap();

    let scores = selector.f_regression_scores(&x, &y).unwrap();
    assert_eq!(scores.len(), 2);
    // The exact-fit feature yields an F-statistic far larger than the weak one — and crucially
    // it is NOT just |corr| (which would be capped at 1.0).
    assert!(
        scores[0] > 1.0,
        "real F-statistic for a strong feature should exceed 1.0, got {}",
        scores[0]
    );
    assert!(scores[0] > scores[1]);
}

#[test]
fn test_grid_search_refits_best_estimator() {
    use crate::ml::models::linear::LinearRegression;
    use crate::ml::sklearn_compat::SupervisedAdapter;

    // y = 2*x + 1
    let mut x = DataFrame::new();
    x.add_column(
        "x".to_string(),
        Series::new(
            (1..=10).map(|i| i as f64).collect::<Vec<f64>>(),
            Some("x".to_string()),
        )
        .unwrap(),
    )
    .unwrap();
    let mut y = DataFrame::new();
    y.add_column(
        "target".to_string(),
        Series::new(
            (1..=10).map(|i| 2.0 * i as f64 + 1.0).collect::<Vec<f64>>(),
            Some("target".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let estimator: Box<dyn SklearnPredictor + Send + Sync> =
        Box::new(SupervisedAdapter::new(LinearRegression::new(), "target"));
    let mut grid = HashMap::new();
    grid.insert("fit_intercept".to_string(), vec!["true".to_string()]);

    let mut search = GridSearchCV::new(estimator, grid).with_cv(CrossValidationStrategy::KFold {
        n_splits: 2,
        shuffle: false,
        random_state: None,
    });
    search.fit(&x, &y).unwrap();

    // The best estimator is really refit on the full dataset (no longer a `None` placeholder).
    assert!(search.best_estimator().is_some());
    let results = search.get_results().unwrap();
    assert!(
        results.best_estimator_.is_some(),
        "SearchResults should describe the refit estimator"
    );

    // The refit estimator can produce predictions for all rows.
    let predictions = search.predict(&x).unwrap();
    assert_eq!(predictions.len(), 10);
}
