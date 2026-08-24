//! Regression tests for the ml/models + ml/preprocessing fabrication fixes
//! (Wave 2, ml-fabrications task).
//!
//! Covers, at the public-API level:
//!   - `models::selection::{GridSearchCV, RandomizedSearchCV}` really search: every
//!     parameter combination is applied to a clone of the base model and cross-validated
//!     on its own, so per-combination scores differ and the reported best is the genuine
//!     arg-best (they used to run ONE CV of the untuned base model, call the first sorted
//!     combination "best", and repeat that single score for every row of `cv_results`).
//!   - `preprocessing::{OneHotEncoder, PolynomialFeatures, Binner, Imputer}` have real
//!     `fit`/`transform` implementations (they used to be builder-only shells whose
//!     bodies were "omitted for brevity" while still being publicly re-exported).
//!   - `preprocessing`'s `columns = None` default selects the numeric columns of a mixed
//!     frame instead of hard-erroring on the first non-`f64` column.
//!   - `LogisticRegression`'s `C` is real L2 regularisation added to the IRLS normal
//!     equations (coefficients shrink monotonically as `C` decreases), and non-binary
//!     targets are rejected instead of silently clamped into `{0, 1}`.
//!   - `LinearRegression::normalize` no longer produces standardised coefficients that
//!     are then applied to raw features at predict time.
//!   - `MLPClassifier`'s binary cross-entropy gradient is no longer multiplied by an
//!     extra sigmoid derivative, so a linearly separable problem actually converges, and
//!     `MLPRegressor`/`MLPClassifier::cross_validate` run real folds instead of
//!     returning `Ok(vec![])`.

use pandrs::dataframe::DataFrame;
use pandrs::ml::models::linear::{LinearRegression, LogisticRegression};
use pandrs::ml::models::neural::{MLPClassifier, MLPConfigBuilder, MLPRegressor};
use pandrs::ml::models::selection::{GridSearchCV, HyperparameterGrid, RandomizedSearchCV};
use pandrs::ml::models::{ModelEvaluator, SupervisedModel};
use pandrs::ml::preprocessing::{
    Binner, HandleUnknown, ImputeStrategy, Imputer, OneHotEncoder, PolynomialFeatures,
    StandardScaler,
};
use pandrs::series::Series;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn df_f64(columns: &[(&str, Vec<f64>)]) -> DataFrame {
    let mut df = DataFrame::new();
    for (name, values) in columns {
        df.add_column(
            (*name).to_string(),
            Series::new(values.clone(), Some((*name).to_string())).expect("series"),
        )
        .expect("add column");
    }
    df
}

fn add_string_column(df: &mut DataFrame, name: &str, values: Vec<&str>) {
    let owned: Vec<String> = values.into_iter().map(|v| v.to_string()).collect();
    df.add_column(
        name.to_string(),
        Series::new(owned, Some(name.to_string())).expect("series"),
    )
    .expect("add column");
}

fn column_values(df: &DataFrame, name: &str) -> Vec<f64> {
    df.get_column::<f64>(name)
        .unwrap_or_else(|e| panic!("column '{}' missing: {:?}", name, e))
        .values()
        .to_vec()
}

/// y = 2x + 1, so a model WITH an intercept fits perfectly and one without does not.
fn linear_df(n: usize) -> DataFrame {
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    df_f64(&[("x", x), ("y", y)])
}

/// Two overlapping (non-separable) classes, so the unpenalised MLE is finite and the
/// effect of L2 shrinkage is measurable without the fit running off to infinity.
fn overlapping_binary_df() -> DataFrame {
    let x = vec![
        -2.0, -1.5, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0, -0.3, 0.3, -1.2, 1.2, 0.8, -0.8,
    ];
    let y = vec![
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
    ];
    df_f64(&[("x", x), ("y", y)])
}

// ---------------------------------------------------------------------------
// GridSearchCV / RandomizedSearchCV: the search actually searches
// ---------------------------------------------------------------------------

/// Every parameter combination must be cross-validated on its own: the per-combination
/// scores must differ, and the combination reported as best must be the genuinely best
/// one. Before the fix, `cv_results` held ONE score repeated for every row and
/// `best_params` was simply the first sorted combination.
#[test]
fn grid_search_scores_each_combination_independently() {
    let df = linear_df(12);
    let mut grid = HyperparameterGrid::new();
    grid.add_param("fit_intercept", vec!["true", "false"]);

    let mut search = GridSearchCV::new(LinearRegression::new(), grid, "r2", 3);
    search.fit(&df, "y").expect("grid search should fit");

    let results = search.cv_results.as_ref().expect("cv_results");
    assert_eq!(results.nrows(), 2, "one row per parameter combination");

    let scores = column_values(results, "mean_test_score");
    assert!(
        (scores[0] - scores[1]).abs() > 1e-6,
        "each combination must be scored on its own data; got repeated score {:?}",
        scores
    );

    // Per-fold scores are recorded too (3 folds -> 3 split columns).
    for fold in 0..3 {
        let split = column_values(results, &format!("split{}_test_score", fold));
        assert_eq!(
            split.len(),
            2,
            "split{} column must cover both combos",
            fold
        );
    }

    let ranks: Vec<f64> = results
        .get_column::<i64>("rank_test_score")
        .expect("rank_test_score")
        .values()
        .iter()
        .map(|&r| r as f64)
        .collect();
    assert!(
        ranks.contains(&1.0),
        "the best combination must be ranked 1"
    );

    let best = search.best_params.as_ref().expect("best_params");
    assert_eq!(
        best.get("fit_intercept").map(String::as_str),
        Some("true"),
        "y = 2x + 1 needs an intercept, so fit_intercept=true is the true best"
    );

    let best_score = search.best_score.expect("best_score");
    let best_of_scores = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (best_score - best_of_scores).abs() < 1e-12,
        "best_score {} must be the best of the per-combination scores {:?}",
        best_score,
        scores
    );
}

/// The best estimator must come back tuned AND refit, not as an untouched clone of the
/// base model.
#[test]
fn grid_search_best_estimator_is_tuned_and_refit() {
    let df = linear_df(12);
    let mut grid = HyperparameterGrid::new();
    grid.add_param("fit_intercept", vec!["true", "false"]);

    let base = LinearRegression::new().with_intercept(false);
    let mut search = GridSearchCV::new(base, grid, "r2", 3);
    search.fit(&df, "y").expect("grid search should fit");

    let best = search.best_estimator().expect("best_estimator");
    assert!(
        best.fit_intercept,
        "the winning parameter value must be applied to the returned estimator"
    );
    let predictions = best
        .predict(&df)
        .expect("the returned estimator must already be fitted");
    assert!((predictions[0] - 1.0).abs() < 1e-6, "refit on y = 2x + 1");
}

/// A parameter the model cannot accept must be an error, never silently skipped: a
/// silently dropped key means the reported score belongs to a configuration that was
/// never evaluated.
#[test]
fn grid_search_rejects_unknown_parameters() {
    let df = linear_df(10);
    let mut grid = HyperparameterGrid::new();
    grid.add_param("definitely_not_a_param", vec!["1", "2"]);

    let mut search = GridSearchCV::new(LinearRegression::new(), grid, "r2", 2);
    assert!(
        search.fit(&df, "y").is_err(),
        "unknown hyperparameters must fail loudly"
    );
}

/// A metric whose optimisation direction is unknown must be rejected — ranking an error
/// metric as "higher is better" would hand back the worst model in the grid.
#[test]
fn grid_search_rejects_unknown_scoring_metric() {
    let df = linear_df(10);
    let mut search = GridSearchCV::new(
        LinearRegression::new(),
        HyperparameterGrid::new(),
        "made_up_metric",
        2,
    );
    assert!(search.fit(&df, "y").is_err());
}

/// Error metrics are minimised, not maximised.
#[test]
fn grid_search_minimises_error_metrics() {
    let df = linear_df(12);
    let mut grid = HyperparameterGrid::new();
    grid.add_param("fit_intercept", vec!["true", "false"]);

    let mut search = GridSearchCV::new(LinearRegression::new(), grid, "mse", 3);
    search.fit(&df, "y").expect("grid search should fit");

    let results = search.cv_results.as_ref().expect("cv_results");
    let scores = column_values(results, "mean_test_score");
    let smallest = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        (search.best_score.expect("best_score") - smallest).abs() < 1e-12,
        "for mse the best score must be the SMALLEST, got {:?} from {:?}",
        search.best_score,
        scores
    );
    assert_eq!(
        search
            .best_params
            .as_ref()
            .and_then(|p| p.get("fit_intercept"))
            .map(String::as_str),
        Some("true")
    );
}

/// RandomizedSearchCV samples combinations and scores each of them for real.
#[test]
fn randomized_search_scores_every_sampled_combination() {
    let df = overlapping_binary_df();
    let mut grid = HyperparameterGrid::new();
    grid.add_param("C", vec!["0.01", "1.0", "100.0"]);

    let mut search = RandomizedSearchCV::new(LogisticRegression::new(), grid, 3, "accuracy", 2)
        .with_random_seed(7);
    search.fit(&df, "y").expect("randomized search should fit");

    let results = search.cv_results.as_ref().expect("cv_results");
    assert_eq!(results.nrows(), 3, "one row per sampled combination");

    let params = results
        .get_column::<String>("param_C")
        .expect("param_C column")
        .values()
        .to_vec();
    let mut sorted = params.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        3,
        "all three sampled C values must be recorded, got {:?}",
        params
    );
}

// ---------------------------------------------------------------------------
// OneHotEncoder
// ---------------------------------------------------------------------------

#[test]
fn one_hot_encoder_emits_hand_computed_indicator_columns() {
    let mut df = DataFrame::new();
    add_string_column(&mut df, "color", vec!["red", "green", "red", "blue"]);
    df.add_column(
        "value".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0], Some("value".to_string())).expect("series"),
    )
    .expect("add column");

    let mut encoder = OneHotEncoder::new();
    let out = encoder.fit_transform(&df).expect("fit_transform");

    // Categories are sorted: blue, green, red.
    assert_eq!(column_values(&out, "color_blue"), vec![0.0, 0.0, 0.0, 1.0]);
    assert_eq!(column_values(&out, "color_green"), vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(column_values(&out, "color_red"), vec![1.0, 0.0, 1.0, 0.0]);
    // Numeric columns are passed through untouched.
    assert_eq!(column_values(&out, "value"), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn one_hot_encoder_drop_first_and_prefix() {
    let mut df = DataFrame::new();
    add_string_column(&mut df, "color", vec!["red", "green", "blue"]);

    let mut encoder = OneHotEncoder::new()
        .drop_first(true)
        .with_prefix("c".to_string());
    let out = encoder.fit_transform(&df).expect("fit_transform");

    assert_eq!(
        out.column_names().len(),
        2,
        "3 categories minus the dropped first = 2 columns, got {:?}",
        out.column_names()
    );
    assert_eq!(column_values(&out, "c_green"), vec![0.0, 1.0, 0.0]);
    assert_eq!(column_values(&out, "c_red"), vec![1.0, 0.0, 0.0]);
}

#[test]
fn one_hot_encoder_handles_unknown_categories() {
    let mut fit_df = DataFrame::new();
    add_string_column(&mut fit_df, "color", vec!["red", "green"]);

    let mut strict = OneHotEncoder::new();
    strict.fit(&fit_df).expect("fit");

    let mut unseen = DataFrame::new();
    add_string_column(&mut unseen, "color", vec!["red", "purple"]);

    assert!(
        strict.transform(&unseen).is_err(),
        "the default (Error) policy must reject a category unseen during fit"
    );

    let lenient = strict.clone().with_handle_unknown(HandleUnknown::Ignore);
    let out = lenient.transform(&unseen).expect("ignore policy");
    assert_eq!(column_values(&out, "color_green"), vec![0.0, 0.0]);
    assert_eq!(
        column_values(&out, "color_red"),
        vec![1.0, 0.0],
        "an unknown category encodes as an all-zero row"
    );
}

// ---------------------------------------------------------------------------
// PolynomialFeatures
// ---------------------------------------------------------------------------

#[test]
fn polynomial_features_full_basis_with_hand_computed_values() {
    let df = df_f64(&[("a", vec![2.0, 3.0]), ("b", vec![5.0, 7.0])]);

    let mut poly = PolynomialFeatures::new(3);
    let out = poly.fit_transform(&df).expect("fit_transform");

    // sklearn: PolynomialFeatures(degree=3) on 2 features -> C(2+3, 3) = 10 columns
    // (bias, a, b, a^2, ab, b^2, a^3, a^2 b, a b^2, b^3).
    assert_eq!(
        out.column_names().len(),
        10,
        "full multiset basis expected, got {:?}",
        out.column_names()
    );

    assert_eq!(column_values(&out, "bias"), vec![1.0, 1.0]);
    assert_eq!(column_values(&out, "a"), vec![2.0, 3.0]);
    assert_eq!(column_values(&out, "b"), vec![5.0, 7.0]);
    assert_eq!(column_values(&out, "a^2"), vec![4.0, 9.0]);
    assert_eq!(column_values(&out, "a b"), vec![10.0, 21.0]);
    assert_eq!(column_values(&out, "b^2"), vec![25.0, 49.0]);
    assert_eq!(column_values(&out, "a^3"), vec![8.0, 27.0]);
    assert_eq!(column_values(&out, "a^2 b"), vec![20.0, 63.0]);
    assert_eq!(column_values(&out, "a b^2"), vec![50.0, 147.0]);
    assert_eq!(column_values(&out, "b^3"), vec![125.0, 343.0]);
}

#[test]
fn polynomial_features_interaction_only_and_no_bias() {
    let df = df_f64(&[
        ("a", vec![2.0]),
        ("b", vec![3.0]),
        ("c", vec![5.0]),
        ("y", vec![1.0]),
    ]);

    let mut poly = PolynomialFeatures::new(2)
        .include_bias(false)
        .interaction_only(true)
        .with_columns(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    let out = poly.fit_transform(&df).expect("fit_transform");

    // a, b, c, ab, ac, bc  (+ the passed-through target column y)
    assert_eq!(
        out.column_names().len(),
        7,
        "unexpected columns: {:?}",
        out.column_names()
    );
    assert_eq!(column_values(&out, "a b"), vec![6.0]);
    assert_eq!(column_values(&out, "a c"), vec![10.0]);
    assert_eq!(column_values(&out, "b c"), vec![15.0]);
    assert!(
        !out.column_names().iter().any(|n| n == "a^2"),
        "interaction_only must not emit squared terms"
    );
    assert_eq!(
        column_values(&out, "y"),
        vec![1.0],
        "columns outside the expansion must survive the transform"
    );
}

#[test]
fn polynomial_features_propagate_missing_values() {
    let df = df_f64(&[("a", vec![2.0, f64::NAN]), ("b", vec![3.0, 4.0])]);
    let mut poly = PolynomialFeatures::new(2).include_bias(false);
    let out = poly.fit_transform(&df).expect("fit_transform");

    let ab = column_values(&out, "a b");
    assert_eq!(ab[0], 6.0);
    assert!(
        ab[1].is_nan(),
        "a missing input must not be silently treated as 0"
    );
    assert_eq!(column_values(&out, "b^2"), vec![9.0, 16.0]);
}

// ---------------------------------------------------------------------------
// Binner
// ---------------------------------------------------------------------------

#[test]
fn binner_uniform_edges_and_assignments() {
    let df = df_f64(&[("x", vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])]);

    let mut binner = Binner::new(5);
    let out = binner.fit_transform(&df).expect("fit_transform");

    let edges = binner
        .bin_edges
        .as_ref()
        .expect("edges")
        .get("x")
        .expect("x edges")
        .clone();
    assert_eq!(edges, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // Bins are [e_i, e_{i+1}) with the top bin closed, so 5.0 lands in the last bin.
    assert_eq!(
        column_values(&out, "x"),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.0],
        "uniform bin assignment"
    );
}

#[test]
fn binner_quantile_strategy_balances_counts() {
    let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let df = df_f64(&[("x", values)]);

    let mut binner = Binner::new(2).with_strategy("quantile");
    let out = binner.fit_transform(&df).expect("fit_transform");

    let edges = binner
        .bin_edges
        .as_ref()
        .expect("edges")
        .get("x")
        .expect("x edges")
        .clone();
    // numpy's linear-interpolation quantiles of 0..9 at 0, 0.5, 1.
    assert_eq!(edges, vec![0.0, 4.5, 9.0]);

    let bins = column_values(&out, "x");
    assert_eq!(bins, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

/// Fixture check against scikit-learn 1.8.0:
/// `KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=...)` on a clustered column,
/// for all three strategies. Both the learned edges and the ordinal assignments must
/// match, which pins the bin convention (`[e_i, e_{i+1})` with a closed top bin) and the
/// quantile interpolation (numpy's linear method).
#[test]
fn binner_matches_sklearn_kbinsdiscretizer_fixture() {
    let values = vec![
        0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 30.0, 31.0, 32.0, 33.0,
    ];
    let df = df_f64(&[("x", values)]);

    let cases: [(&str, [f64; 4], [f64; 12]); 3] = [
        (
            "uniform",
            [0.0, 11.0, 22.0, 33.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        ),
        (
            "quantile",
            [0.0, 3.666666666666667, 18.0, 33.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        ),
        (
            "kmeans",
            [0.0, 6.5, 21.25, 33.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        ),
    ];

    for (strategy, expected_edges, expected_bins) in cases {
        let mut binner = Binner::new(3).with_strategy(strategy);
        let out = binner.fit_transform(&df).expect("fit_transform");

        let edges = binner
            .bin_edges
            .as_ref()
            .expect("edges")
            .get("x")
            .expect("x edges")
            .clone();
        assert_eq!(edges.len(), 4, "{}: expected 4 edges", strategy);
        for (actual, expected) in edges.iter().zip(expected_edges.iter()) {
            assert!(
                (actual - expected).abs() < 1e-9,
                "{}: edge {} differs from sklearn's {}",
                strategy,
                actual,
                expected
            );
        }

        assert_eq!(
            column_values(&out, "x"),
            expected_bins.to_vec(),
            "{}: ordinal assignment must match sklearn",
            strategy
        );
    }
}

#[test]
fn binner_rejects_unknown_strategy_and_constant_column() {
    let df = df_f64(&[("x", vec![1.0, 2.0, 3.0, 4.0])]);
    let mut bad = Binner::new(2).with_strategy("magic");
    assert!(
        bad.fit(&df).is_err(),
        "an unknown strategy must error rather than fall back to uniform"
    );

    let constant = df_f64(&[("x", vec![7.0, 7.0, 7.0])]);
    let mut binner = Binner::new(3);
    assert!(
        binner.fit(&constant).is_err(),
        "a constant column cannot be split into distinct bins"
    );
}

#[test]
fn binner_preserves_missing_values() {
    let df = df_f64(&[("x", vec![0.0, f64::NAN, 4.0])]);
    let mut binner = Binner::new(2);
    let out = binner.fit_transform(&df).expect("fit_transform");
    let bins = column_values(&out, "x");
    assert_eq!(bins[0], 0.0);
    assert!(
        bins[1].is_nan(),
        "missing must stay missing, not become bin 0"
    );
    assert_eq!(bins[2], 1.0);
}

// ---------------------------------------------------------------------------
// Imputer
// ---------------------------------------------------------------------------

#[test]
fn imputer_strategies_match_hand_computed_fills() {
    let raw = vec![1.0, f64::NAN, 2.0, 2.0, 10.0];

    let df = df_f64(&[("x", raw.clone())]);
    let mut mean = Imputer::new().with_strategy(ImputeStrategy::Mean);
    // observed = [1, 2, 2, 10] -> mean 3.75
    assert_eq!(
        column_values(&mean.fit_transform(&df).expect("mean"), "x"),
        vec![1.0, 3.75, 2.0, 2.0, 10.0]
    );

    let mut median = Imputer::new().with_strategy(ImputeStrategy::Median);
    // sorted observed = [1, 2, 2, 10] -> median (2 + 2)/2 = 2
    assert_eq!(
        column_values(&median.fit_transform(&df).expect("median"), "x"),
        vec![1.0, 2.0, 2.0, 2.0, 10.0]
    );

    let mut mode = Imputer::new().with_strategy(ImputeStrategy::MostFrequent);
    assert_eq!(
        column_values(&mode.fit_transform(&df).expect("mode"), "x"),
        vec![1.0, 2.0, 2.0, 2.0, 10.0]
    );

    let mut constant = Imputer::new().with_strategy(ImputeStrategy::Constant(-1.0));
    assert_eq!(
        column_values(&constant.fit_transform(&df).expect("constant"), "x"),
        vec![1.0, -1.0, 2.0, 2.0, 10.0]
    );
}

#[test]
fn imputer_errors_on_all_missing_column() {
    let df = df_f64(&[("x", vec![f64::NAN, f64::NAN])]);
    let mut imputer = Imputer::new().with_strategy(ImputeStrategy::Mean);
    assert!(
        imputer.fit(&df).is_err(),
        "a mean cannot be computed from a column with no observed values"
    );

    let mut constant = Imputer::new().with_strategy(ImputeStrategy::Constant(0.5));
    let out = constant
        .fit_transform(&df)
        .expect("constant strategy works");
    assert_eq!(column_values(&out, "x"), vec![0.5, 0.5]);
}

#[test]
fn imputer_leaves_unselected_columns_missing() {
    let df = df_f64(&[("x", vec![1.0, f64::NAN]), ("z", vec![f64::NAN, 2.0])]);
    let mut imputer = Imputer::new().with_columns(vec!["x".to_string()]);
    let out = imputer.fit_transform(&df).expect("fit_transform");

    assert_eq!(column_values(&out, "x"), vec![1.0, 1.0]);
    let z = column_values(&out, "z");
    assert!(
        z[0].is_nan(),
        "a column outside the imputer's scope keeps its missing values"
    );
}

// ---------------------------------------------------------------------------
// Default column selection on mixed frames
// ---------------------------------------------------------------------------

/// `columns = None` used to call `get_column::<f64>` on EVERY column and fail on the
/// first non-`f64` one, which made the documented "all numeric columns" default unusable
/// on any frame containing a string (or integer) column.
#[test]
fn default_column_selection_skips_non_numeric_columns() {
    let mut df = df_f64(&[("value", vec![1.0, 2.0, 3.0])]);
    add_string_column(&mut df, "label", vec!["a", "b", "c"]);
    df.add_column(
        "count".to_string(),
        Series::new(vec![10i64, 20, 30], Some("count".to_string())).expect("series"),
    )
    .expect("add column");

    let mut scaler = StandardScaler::new();
    let out = scaler
        .fit_transform(&df)
        .expect("mixed frames must not break the numeric default");

    let means = scaler.means.as_ref().expect("means");
    assert!(means.contains_key("value"));
    assert!(
        means.contains_key("count"),
        "integer columns are numeric and must be scaled"
    );
    assert!(
        !means.contains_key("label"),
        "string columns must be skipped, not fatal"
    );

    let scaled = column_values(&out, "value");
    assert!((scaled[1]).abs() < 1e-12, "centred middle value");

    let labels = out
        .get_column::<String>("label")
        .expect("the string column must survive with its type intact")
        .values()
        .to_vec();
    assert_eq!(labels, vec!["a", "b", "c"]);
}

// ---------------------------------------------------------------------------
// LogisticRegression: real L2, honest labels
// ---------------------------------------------------------------------------

/// `C` must behave like scikit-learn's inverse regularisation strength: shrinking `C`
/// shrinks the fitted coefficient monotonically. The old implementation multiplied the
/// unpenalised solution by `1 / (1 + 1/(C·n))` after the fact, which is not L2 at all and
/// whose effect vanishes as `n` grows.
#[test]
fn logistic_regression_l2_shrinks_coefficients_monotonically() {
    let df = overlapping_binary_df();
    let mut previous = f64::INFINITY;

    for &c in &[100.0_f64, 10.0, 1.0, 0.1, 0.01] {
        let mut model = LogisticRegression::new().with_regularization(c);
        model.fit(&df, "y").expect("fit");
        let coef = model
            .coefficients
            .as_ref()
            .expect("coefficients")
            .get("x")
            .copied()
            .expect("x coefficient")
            .abs();

        assert!(
            coef < previous,
            "|coef| must shrink as C decreases: C={} gave {} which is not below {}",
            c,
            coef,
            previous
        );
        previous = coef;
    }

    // Strong regularisation must pull the coefficient close to zero.
    assert!(
        previous < 0.2,
        "C = 0.01 should shrink the coefficient hard, got {}",
        previous
    );
}

/// Fixture check against scikit-learn 1.8.0: `sklearn.linear_model.LogisticRegression`
/// (`lbfgs`, `tol=1e-10`, `max_iter=10000`) fitted on the same overlapping data.
///
/// Both minimise `Σ −log p(y|x) + (1/(2C))·‖w‖²` with an unpenalised intercept, so the
/// coefficients must agree — the penalised IRLS normal equations `(X'WX + λD)β = X'Wz`
/// are the exact Newton step for that objective. The old post-hoc shrinkage produced
/// coefficients that were off by ~0.9 at `C = 0.01`.
#[test]
fn logistic_regression_matches_sklearn_l2_fixture() {
    let df = overlapping_binary_df();

    // (C, sklearn coef_, sklearn intercept_)
    let fixture = [
        (100.0_f64, 2.8712514467_f64, -0.0889283024_f64),
        (10.0, 2.4966380475, -0.0691107337),
        (1.0, 1.4619753439, -0.0278457390),
        (0.1, 0.4505480352, -0.0059658821),
        (0.01, 0.0629528828, -0.0007878453),
    ];

    for (c, expected_coef, expected_intercept) in fixture {
        let mut model = LogisticRegression::new()
            .with_regularization(c)
            .with_max_iter(500)
            .with_tolerance(1e-10);
        model.fit(&df, "y").expect("fit");

        let coef = model
            .coefficients
            .as_ref()
            .and_then(|m| m.get("x"))
            .copied()
            .expect("coefficient");
        let intercept = model.intercept.expect("intercept");

        assert!(
            (coef - expected_coef).abs() < 1e-6,
            "C={}: coefficient {} differs from sklearn's {}",
            c,
            coef,
            expected_coef
        );
        assert!(
            (intercept - expected_intercept).abs() < 1e-6,
            "C={}: intercept {} differs from sklearn's {}",
            c,
            intercept,
            expected_intercept
        );
    }
}

/// An unpenalised fit (`C = infinity`) must be at least as large in magnitude as any
/// penalised one.
#[test]
fn logistic_regression_unregularized_path() {
    let df = overlapping_binary_df();

    let mut unpenalised = LogisticRegression::new().without_regularization();
    unpenalised.fit(&df, "y").expect("fit");
    let free = unpenalised
        .coefficients
        .as_ref()
        .and_then(|c| c.get("x"))
        .copied()
        .expect("coefficient")
        .abs();

    let mut penalised = LogisticRegression::new().with_regularization(1.0);
    penalised.fit(&df, "y").expect("fit");
    let shrunk = penalised
        .coefficients
        .as_ref()
        .and_then(|c| c.get("x"))
        .copied()
        .expect("coefficient")
        .abs();

    assert!(
        free > shrunk,
        "unregularised |coef| {} must exceed the C=1 fit {}",
        free,
        shrunk
    );
}

/// Non-binary targets used to be silently clamped into {0, 1}: a three-class target
/// became a two-class one and classification metrics were reported for a problem the
/// caller never posed.
#[test]
fn logistic_regression_rejects_non_binary_targets() {
    let df = df_f64(&[
        ("x", vec![0.0, 1.0, 2.0, 3.0]),
        ("y", vec![0.0, 1.0, 2.0, 1.0]),
    ]);
    let mut model = LogisticRegression::new();
    let err = model.fit(&df, "y");
    assert!(err.is_err(), "a 3-class target must be rejected");

    let single_class = df_f64(&[("x", vec![0.0, 1.0]), ("y", vec![1.0, 1.0])]);
    assert!(
        LogisticRegression::new().fit(&single_class, "y").is_err(),
        "a single-class target must be rejected"
    );
}

#[test]
fn logistic_regression_rejects_non_positive_c() {
    let df = overlapping_binary_df();
    let mut model = LogisticRegression::new().with_regularization(0.0);
    assert!(
        model.fit(&df, "y").is_err(),
        "C = 0 is not a valid strength"
    );
}

// ---------------------------------------------------------------------------
// LinearRegression: normalize is applied consistently
// ---------------------------------------------------------------------------

/// With `normalize = true` the coefficients must be reported in the ORIGINAL feature
/// units, so predictions (and every consumer that reads `coefficients`/`intercept`
/// directly, such as the model-serving bridge) stay correct. Previously the scaling was
/// applied during `fit` only, so `predict` combined standardised coefficients with raw
/// features.
#[test]
fn linear_regression_normalize_predicts_in_original_units() {
    let x: Vec<f64> = (0..12).map(|i| 100.0 + 10.0 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 3.0 * v - 7.0).collect();
    let df = df_f64(&[("x", x.clone()), ("y", y.clone())]);

    let mut plain = LinearRegression::new();
    plain.fit(&df, "y").expect("fit");

    let mut normalised = LinearRegression::new().with_normalization(true);
    normalised.fit(&df, "y").expect("fit");

    let coef_plain = plain
        .coefficients
        .as_ref()
        .and_then(|c| c.get("x"))
        .copied()
        .expect("coefficient");
    let coef_norm = normalised
        .coefficients
        .as_ref()
        .and_then(|c| c.get("x"))
        .copied()
        .expect("coefficient");

    assert!(
        (coef_plain - 3.0).abs() < 1e-8 && (coef_norm - 3.0).abs() < 1e-8,
        "both fits must recover the slope 3 in original units: {} vs {}",
        coef_plain,
        coef_norm
    );

    let predictions = normalised.predict(&df).expect("predict");
    for (pred, actual) in predictions.iter().zip(y.iter()) {
        assert!(
            (pred - actual).abs() < 1e-6,
            "normalised model must predict from raw features: {} vs {}",
            pred,
            actual
        );
    }
}

/// Standardisation shifts the features by their mean, which only an intercept can
/// absorb; the combination must be rejected rather than quietly fitting a different
/// model.
#[test]
fn linear_regression_normalize_requires_intercept() {
    let df = linear_df(10);
    let mut model = LinearRegression::new()
        .with_normalization(true)
        .with_intercept(false);
    assert!(model.fit(&df, "y").is_err());
}

/// A constant target has no variance to explain: R² = 1.0 is only earned by a
/// residual-free prediction (mirrors `metrics::regression::r2_score`).
#[test]
fn linear_regression_r2_on_constant_target() {
    let df = df_f64(&[("x", vec![1.0, 2.0, 3.0]), ("y", vec![5.0, 5.0, 5.0])]);
    let mut model = LinearRegression::new();
    model.fit(&df, "y").expect("fit");
    let r2 = model.r_squared(&df, "y").expect("r2");
    assert!(
        (r2 - 1.0).abs() < 1e-9,
        "a perfect constant fit still scores 1.0, got {}",
        r2
    );

    let mut wrong = LinearRegression::new();
    wrong.fit(&df, "y").expect("fit");
    // Force a wrong constant prediction by evaluating on a shifted target.
    let shifted = df_f64(&[("x", vec![1.0, 2.0, 3.0]), ("y", vec![9.0, 9.0, 9.0])]);
    let metrics = wrong.evaluate(&shifted, "y").expect("evaluate");
    let r2_wrong = metrics.get_metric("r2").copied().expect("r2 metric");
    assert_eq!(
        r2_wrong, 0.0,
        "a constant target missed by the model must not score 1.0"
    );
}

// ---------------------------------------------------------------------------
// MLP: real gradients, real batches, real cross-validation
// ---------------------------------------------------------------------------

/// With the binary cross-entropy gradient multiplied by an extra sigmoid derivative the
/// output saturates and learning stalls. A linearly separable problem must converge.
#[test]
fn mlp_binary_classifier_converges_on_separable_data() {
    let x1: Vec<f64> = vec![
        -3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0, -2.2, 2.2,
    ];
    let y: Vec<f64> = x1
        .iter()
        .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
        .collect();
    let df = df_f64(&[("x1", x1), ("y", y.clone())]);

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![8])
        .learning_rate(0.2)
        .n_epochs(600)
        .batch_size(4)
        .random_seed(11)
        .early_stopping_patience(None)
        .build();

    let mut model = MLPClassifier::new(config);
    model.fit(&df, "y").expect("fit");

    let metrics = model.evaluate(&df, "y").expect("evaluate");
    let accuracy = metrics.get_metric("accuracy").copied().expect("accuracy");
    assert!(
        accuracy >= 0.95,
        "a linearly separable binary problem must converge, got accuracy {}",
        accuracy
    );

    let history = model.training_loss_history();
    assert!(
        history.last().expect("loss history") < history.first().expect("loss history"),
        "training loss must decrease: {:?} .. {:?}",
        history.first(),
        history.last()
    );
}

/// The multi-class (softmax + cross-entropy) path must learn a separable 3-class
/// problem. This pins the fused softmax gradient after `Activation::backward`'s
/// fabricated `1.0` derivative was replaced by an explicit fused-output-gradient flag
/// plus a real Jacobian-vector product for unfused uses of softmax.
#[test]
fn mlp_multiclass_classifier_learns_separable_classes() {
    let x: Vec<f64> = vec![
        -3.0, -2.6, -2.2, -1.8, -0.4, -0.2, 0.0, 0.2, 1.8, 2.2, 2.6, 3.0,
    ];
    let y: Vec<f64> = x
        .iter()
        .map(|&v| {
            if v < -1.0 {
                0.0
            } else if v < 1.0 {
                1.0
            } else {
                2.0
            }
        })
        .collect();
    let df = df_f64(&[("x", x), ("y", y)]);

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![12])
        .learning_rate(0.2)
        .n_epochs(800)
        .batch_size(4)
        .random_seed(5)
        .early_stopping_patience(None)
        .build();

    let mut model = MLPClassifier::new(config);
    model.fit(&df, "y").expect("fit");

    let accuracy = model
        .evaluate(&df, "y")
        .expect("evaluate")
        .get_metric("accuracy")
        .copied()
        .expect("accuracy");
    assert!(
        accuracy >= 0.9,
        "the softmax path must learn 3 separable classes, got accuracy {}",
        accuracy
    );

    // Probabilities must be a real distribution, not a constant vector.
    let probabilities = model.predict_proba(&df).expect("predict_proba");
    for row in &probabilities {
        let total: f64 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "softmax rows must sum to 1");
    }
    let first = &probabilities[0];
    let last = &probabilities[probabilities.len() - 1];
    assert!(
        first
            .iter()
            .zip(last.iter())
            .any(|(a, b)| (a - b).abs() > 0.1),
        "class probabilities must depend on the input"
    );
}

/// `cross_validate` used to return `Ok(vec![])` for both MLP models, which downstream
/// code averaged into a fabricated score of exactly 0.0.
#[test]
fn mlp_cross_validate_runs_real_folds() {
    let x: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
    let df = df_f64(&[("x", x), ("y", y)]);

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![4])
        .learning_rate(0.01)
        .n_epochs(50)
        .batch_size(4)
        .early_stopping_patience(None)
        .build();

    let model = MLPRegressor::new(config);
    let folds = model.cross_validate(&df, "y", 3).expect("cross_validate");

    assert_eq!(folds.len(), 3, "one ModelMetrics per fold");
    for (idx, fold) in folds.iter().enumerate() {
        let mse = fold.get_metric("mse").copied().expect("mse metric");
        assert!(mse.is_finite(), "fold {} must report a real mse", idx);
    }
}

/// `batch_size` must actually change the optimisation: full-batch and per-sample updates
/// on the same seed cannot produce identical weights (they used to, because every sample
/// triggered its own update regardless of `batch_size`).
#[test]
fn mlp_batch_size_changes_the_optimisation() {
    let x: Vec<f64> = (0..16).map(|i| i as f64 / 4.0).collect();
    let y: Vec<f64> = x.iter().map(|v| 0.5 * v + 0.25).collect();
    let df = df_f64(&[("x", x), ("y", y)]);

    let base = MLPConfigBuilder::new()
        .hidden_layers(vec![4])
        .learning_rate(0.05)
        .n_epochs(20)
        .random_seed(3)
        .early_stopping_patience(None);

    let mut per_sample = MLPRegressor::new(base.clone().batch_size(1).build());
    per_sample.fit(&df, "y").expect("fit");

    let mut full_batch = MLPRegressor::new(base.batch_size(16).build());
    full_batch.fit(&df, "y").expect("fit");

    let a = per_sample.predict(&df).expect("predict");
    let b = full_batch.predict(&df).expect("predict");
    let difference: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);

    assert!(
        difference > 1e-9,
        "batch_size must affect training; predictions were identical"
    );
}

/// An output activation that cannot produce class probabilities must be rejected rather
/// than silently overridden — `output_activation` used to be ignored entirely.
#[test]
fn mlp_classifier_validates_output_activation() {
    let df = df_f64(&[
        ("x", vec![0.0, 1.0, 2.0, 3.0]),
        ("y", vec![0.0, 0.0, 1.0, 1.0]),
    ]);

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![4])
        .n_epochs(5)
        .early_stopping_patience(None)
        .output_activation(pandrs::ml::models::neural::Activation::ReLU)
        .build();

    let mut model = MLPClassifier::new(config);
    assert!(
        model.fit(&df, "y").is_err(),
        "ReLU cannot produce class probabilities"
    );
}

/// The regressor honours the configured output activation: a sigmoid output cannot
/// exceed 1.0.
#[test]
fn mlp_regressor_honours_output_activation() {
    let df = df_f64(&[
        ("x", vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ("y", vec![0.1, 0.2, 0.4, 0.6, 0.8, 0.9]),
    ]);

    let config = MLPConfigBuilder::new()
        .hidden_layers(vec![4])
        .learning_rate(0.05)
        .n_epochs(100)
        .batch_size(3)
        .early_stopping_patience(None)
        .output_activation(pandrs::ml::models::neural::Activation::Sigmoid)
        .build();

    let mut model = MLPRegressor::new(config);
    model.fit(&df, "y").expect("fit");

    for prediction in model.predict(&df).expect("predict") {
        assert!(
            (0.0..=1.0).contains(&prediction),
            "a sigmoid output layer must bound predictions to [0, 1], got {}",
            prediction
        );
    }
}
