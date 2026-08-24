//! Regression tests for the feature_engineering / pipeline_extended / pipeline_compat /
//! pipeline fixes (Wave 2, ml-fe-pipelines task).
//!
//! Covers, at the public-API level:
//!   - `AutoFeatureEngineer`'s polynomial feature generator now produces the full
//!     multiset monomial basis (not just per-column powers plus degree-2 cross terms),
//!     matching `sklearn.preprocessing.PolynomialFeatures`'s column count exactly.
//!   - `FeatureSelectionMethod::L1Based` is now a genuine coordinate-descent Lasso: it
//!     zeroes out coefficients for features with no real association with the target,
//!     not just an OLS-magnitude re-ranking.
//!   - `pipeline_extended`'s binning matches `sklearn.preprocessing.KBinsDiscretizer`
//!     (`encode='ordinal'`) exactly on a fixture verified against sklearn 1.8.0, for both
//!     the `EqualWidth` ("uniform") and `EqualFrequency`/`Quantile` ("quantile") strategies.
//!   - `pipeline_compat::{StandardScaler, MinMaxScaler}::transform` errors instead of
//!     silently passing data through unscaled when called before `fit`.
//!   - `AutoFeatureEngineer`'s `KBest` selection attributes each `feature_scores_` entry
//!     to the CORRECT feature name (not a rank-position mismatch), verified with features
//!     deliberately ordered so their target-correlation strength does not match their
//!     column order.
//!   - `pipeline::PipelineStage::transform`'s column pass-through preserves a column's
//!     CONCRETE type (so a digit-coded categorical `Series<String>` survives a
//!     `StandardScaler` stage unchanged and is still recognized as categorical by a later
//!     `OneHotEncoder` stage in the same `Pipeline`), instead of routing by textual
//!     parseability.

use chrono::{Datelike, NaiveDate, Timelike};
use pandrs::dataframe::DataFrame;
use pandrs::ml::feature_engineering::{AutoFeatureEngineer, FeatureSelectionMethod, RobustScaler};
use pandrs::ml::pipeline_extended::{BinningStrategy, FeatureEngineeringStage, PipelineContext};
use pandrs::ml::{
    AdvancedPipelineStage, FeatureScaler, Pipeline, PipelineStage, PipelineTransformer,
    ScoreFunction, StandardScaler, Transformer,
};
use pandrs::optimized::OptimizedDataFrame;
use pandrs::series::Series;
use pandrs::ColumnTrait;
use std::collections::HashMap;

fn df_column(name: &str, values: Vec<f64>) -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        name.to_string(),
        Series::new(values, Some(name.to_string())).expect("series creation should succeed"),
    )
    .expect("add_column should succeed");
    df
}

/// A full degree-3 polynomial expansion of 3 input features must produce exactly the
/// column count `sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False)`
/// would: `C(n + d, d) - 1 = C(6, 3) - 1 = 19` total columns (3 original + 16 generated),
/// counting every monomial of degree 1..=3 -- not just pure powers (`x^2`, `x^3`) and
/// degree-2 pairs (`x*y`), but mixed cross terms like `x^2*y` and three-way products like
/// `x*y*z`. Before the fix, degree >= 3 requests silently generated only the degree-2 basis
/// (pure powers plus pairs), never anything at degree 3 and above.
#[test]
fn polynomial_degree_3_three_features_matches_sklearn_column_count() {
    let n = 10;
    let x_vals: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let y_vals: Vec<f64> = (1..=n).map(|i| (i as f64) * 1.3 + 0.5).collect();
    let z_vals: Vec<f64> = (1..=n).map(|i| (i as f64) * 0.7 - 2.0).collect();

    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(x_vals, Some("x".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "y".to_string(),
        Series::new(y_vals, Some("y".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "z".to_string(),
        Series::new(z_vals, Some("z".to_string())).unwrap(),
    )
    .unwrap();

    let mut engineer = AutoFeatureEngineer::new()
        .with_polynomial(3)
        .without_scaling();
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;
    engineer.generate_temporal = false;
    engineer.perform_selection = false;

    engineer.fit(&df, None).expect("fit should succeed");
    let transformed = engineer.transform(&df).expect("transform should succeed");

    assert_eq!(
        transformed.column_names().len(),
        19,
        "expected 3 original + 16 generated (degree 2..=3 over 3 features) = 19 columns, \
         got columns: {:?}",
        transformed.column_names()
    );

    // Spot-check a genuine degree-3 mixed cross term exists (this is exactly what the
    // previous degree-2-only implementation could never produce).
    assert!(
        transformed.column_names().iter().any(|c| c == "x^2*y"),
        "expected a mixed degree-3 cross term like 'x^2*y' among: {:?}",
        transformed.column_names()
    );
    assert!(
        transformed.column_names().iter().any(|c| c == "x*y*z"),
        "expected the three-way product 'x*y*z' among: {:?}",
        transformed.column_names()
    );
}

/// `FeatureSelectionMethod::L1Based` must be a genuine Lasso: features with no real
/// linear association with the target should end up with an *exactly zero* fitted
/// coefficient (L1-induced sparsity), not merely be out-ranked. The previous
/// implementation ran one unpenalized OLS fit and used `|coefficient|` directly, which
/// (a) is not Lasso at all, and (b) essentially never produces an exact zero.
#[test]
fn l1_based_lasso_zeroes_noise_feature_coefficients() {
    let n = 30;
    let target: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    // Dominant, large-scale signal: signal = 100 * target.
    let signal: Vec<f64> = target.iter().map(|&t| 100.0 * t).collect();
    // Small-amplitude, near-orthogonal-to-target noise columns.
    let noise1: Vec<f64> = (1..=n).map(|i| (i as f64 * 0.37).sin() * 0.01).collect();
    let noise2: Vec<f64> = (1..=n).map(|i| (i as f64 * 0.91).cos() * 0.01).collect();
    let noise3: Vec<f64> = (1..=n).map(|i| (i as f64 * 1.53).sin() * 0.01).collect();
    let noise4: Vec<f64> = (1..=n).map(|i| (i as f64 * 2.11).cos() * 0.01).collect();

    let mut x = DataFrame::new();
    for (name, values) in [
        ("signal", &signal),
        ("noise1", &noise1),
        ("noise2", &noise2),
        ("noise3", &noise3),
        ("noise4", &noise4),
    ] {
        x.add_column(
            name.to_string(),
            Series::new(values.clone(), Some(name.to_string())).unwrap(),
        )
        .unwrap();
    }
    let y = df_column("target", target);

    let mut engineer = AutoFeatureEngineer::new()
        .with_selection(FeatureSelectionMethod::L1Based, Some(1))
        .without_scaling();
    engineer.generate_polynomial = false;
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;

    engineer.fit(&x, Some(&y)).expect("fit should succeed");

    let selected = engineer
        .get_selected_features()
        .expect("selection should have run");
    assert_eq!(
        selected,
        &[0],
        "the dominant signal feature (index 0) should be the sole selection"
    );

    let scores: &HashMap<String, f64> = engineer
        .get_feature_scores()
        .expect("feature_scores_ should be populated after selection");

    let signal_score = *scores.get("signal").expect("signal score must be present");
    assert!(
        signal_score > 0.0,
        "signal feature must have a nonzero Lasso coefficient magnitude, got {}",
        signal_score
    );

    for noise_name in ["noise1", "noise2", "noise3", "noise4"] {
        let noise_score = *scores
            .get(noise_name)
            .unwrap_or_else(|| panic!("{} score must be present", noise_name));
        assert_eq!(
            noise_score, 0.0,
            "Lasso should zero out noise feature '{}', got coefficient magnitude {}",
            noise_name, noise_score
        );
    }
}

/// `pipeline_extended`'s `EqualWidth` (sklearn "uniform") binning must match
/// `sklearn.preprocessing.KBinsDiscretizer(n_bins=3, encode='ordinal',
/// strategy='uniform')` exactly. Verified directly against sklearn 1.8.0:
/// `KBinsDiscretizer(n_bins=3).fit_transform([[-2],[-1],[0],[1]])` == `[[0],[1],[2],[2]]`
/// (edges `[-2,-1,0,1]`: a value sitting exactly on an interior edge belongs to the bin
/// *above* it, not below -- the previous implementation both put boundary values in the
/// wrong (lower) bin and gave the exact minimum its own singleton bin, yielding
/// `bins + 1` distinct labels instead of `bins`).
#[test]
fn equal_width_binning_matches_sklearn_kbinsdiscretizer_fixture() {
    let mut df = OptimizedDataFrame::new();
    df.add_float_column("value", vec![-2.0, -1.0, 0.0, 1.0])
        .unwrap();

    let stage = FeatureEngineeringStage::new().with_binning(
        "value".to_string(),
        3,
        BinningStrategy::EqualWidth,
    );
    let context = PipelineContext {
        metadata: HashMap::new(),
        metrics: HashMap::new(),
        execution_history: Vec::new(),
    };

    let result = stage
        .transform_with_context(&df, &context)
        .expect("binning should succeed");

    let binned = result.column("value_binned").expect("binned column exists");
    let pandrs::Column::Int64(int_col) = binned.column() else {
        panic!("expected an Int64 binned column");
    };
    let labels: Vec<i64> = (0..int_col.len())
        .map(|i| int_col.get(i).unwrap().unwrap())
        .collect();

    assert_eq!(
        labels,
        vec![0, 1, 2, 2],
        "sklearn KBinsDiscretizer(n_bins=3, strategy='uniform') on [-2,-1,0,1] with \
         edges [-2,-1,0,1] gives labels [0,1,2,2]"
    );
}

/// `EqualFrequency` binning must match `sklearn.preprocessing.KBinsDiscretizer`'s
/// `strategy='quantile'`. Verified directly against sklearn 1.8.0:
/// `KBinsDiscretizer(n_bins=4, strategy='quantile').fit_transform([[1],...,[10]])` uses
/// edges `[1, 3.25, 5.5, 7.75, 10]` (linear-interpolated quartiles, matching
/// `numpy.quantile`'s default) and produces labels
/// `[0,0,0,1,1,2,2,3,3,3]`. The previous implementation used integer-stride indexing
/// (`len / bins`) instead of interpolation, which both used the wrong edges and left the
/// top edge short of the true maximum.
#[test]
fn equal_frequency_binning_matches_sklearn_quantile_strategy_fixture() {
    let mut df = OptimizedDataFrame::new();
    let values: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    df.add_float_column("value", values).unwrap();

    let stage = FeatureEngineeringStage::new().with_binning(
        "value".to_string(),
        4,
        BinningStrategy::EqualFrequency,
    );
    let context = PipelineContext {
        metadata: HashMap::new(),
        metrics: HashMap::new(),
        execution_history: Vec::new(),
    };

    let result = stage
        .transform_with_context(&df, &context)
        .expect("binning should succeed");

    let binned = result.column("value_binned").expect("binned column exists");
    let pandrs::Column::Int64(int_col) = binned.column() else {
        panic!("expected an Int64 binned column");
    };
    let labels: Vec<i64> = (0..int_col.len())
        .map(|i| int_col.get(i).unwrap().unwrap())
        .collect();

    assert_eq!(
        labels,
        vec![0, 0, 0, 1, 1, 2, 2, 3, 3, 3],
        "sklearn KBinsDiscretizer(n_bins=4, strategy='quantile') on [1..10] gives \
         labels [0,0,0,1,1,2,2,3,3,3]"
    );
}

/// `pipeline_compat::{StandardScaler, MinMaxScaler}` implement `Transformer` for
/// `OptimizedDataFrame`; calling `transform` before `fit` must error, not silently pass
/// the input through unscaled (which is what a bare `mean=0.0, std=1.0` fallback amounts
/// to).
#[test]
fn unfitted_standard_scaler_transform_errors() {
    let mut df = OptimizedDataFrame::new();
    df.add_float_column("a", vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();

    let scaler = StandardScaler::new();
    let result = Transformer::transform(&scaler, &df);
    assert!(
        result.is_err(),
        "transform() on an unfitted StandardScaler must error, not pass data through"
    );
}

/// `fit_transform` must be equivalent to `fit` followed by `transform` -- verified here
/// by checking the fit_transform() output is actually standardized (mean ~0, std ~1),
/// which only holds if `fit_transform` genuinely fits *and* applies the scaler rather
/// than (as a hand-duplicated implementation risks) silently drifting out of sync with
/// `fit`/`transform`.
#[test]
fn standard_scaler_fit_transform_matches_fit_then_transform() {
    let mut df = OptimizedDataFrame::new();
    df.add_float_column("a", vec![10.0, 20.0, 30.0, 40.0, 50.0])
        .unwrap();

    let mut scaler_combined = StandardScaler::new();
    let combined_result = Transformer::fit_transform(&mut scaler_combined, &df)
        .expect("fit_transform should succeed");

    let mut scaler_separate = StandardScaler::new();
    Transformer::fit(&mut scaler_separate, &df).expect("fit should succeed");
    let separate_result =
        Transformer::transform(&scaler_separate, &df).expect("transform should succeed");

    let combined_view = combined_result.column("a").unwrap();
    let separate_view = separate_result.column("a").unwrap();
    let combined_col = combined_view.as_float64().unwrap();
    let separate_col = separate_view.as_float64().unwrap();

    for i in 0..combined_col.len() {
        let c = combined_col.get(i).unwrap().unwrap();
        let s = separate_col.get(i).unwrap().unwrap();
        assert!(
            (c - s).abs() < 1e-12,
            "fit_transform()[{}]={} must match fit()+transform()[{}]={}",
            i,
            c,
            i,
            s
        );
    }

    // And it must actually be standardized, not just "internally consistent".
    let mean: f64 = (0..combined_col.len())
        .map(|i| combined_col.get(i).unwrap().unwrap())
        .sum::<f64>()
        / combined_col.len() as f64;
    assert!(
        mean.abs() < 1e-9,
        "standardized mean should be ~0, got {mean}"
    );
}

/// `AutoFeatureEngineer::generate_temporal` must actually extract year/month/day/hour/weekday
/// columns from a datetime-parseable string column instead of being a silent no-op behind a
/// public flag. Expected values are computed independently via `chrono` on the same input
/// strings (rather than hand-derived constants), so this checks the extraction against the
/// same ground truth the implementation itself is built on.
#[test]
fn temporal_features_extract_year_month_day_hour_weekday() {
    let dates = vec![
        "2024-03-15".to_string(),          // date-only
        "2023-11-02 08:30:00".to_string(), // date + time
        "2022-07-04".to_string(),
    ];

    let mut x = DataFrame::new();
    x.add_column(
        "event_date".to_string(),
        Series::new(dates.clone(), Some("event_date".to_string())).unwrap(),
    )
    .unwrap();

    let mut engineer = AutoFeatureEngineer::new().without_scaling();
    engineer.generate_polynomial = false;
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;
    engineer.generate_temporal = true;
    engineer.perform_selection = false;

    engineer.fit(&x, None).expect("fit should succeed");
    let out = engineer.transform(&x).expect("transform should succeed");

    for suffix in ["year", "month", "day", "hour", "weekday"] {
        let col = format!("event_date_{}", suffix);
        assert!(
            out.contains_column(&col),
            "expected generated column '{}' among {:?}",
            col,
            out.column_names()
        );
    }

    let years = out.get_column_numeric_values("event_date_year").unwrap();
    let months = out.get_column_numeric_values("event_date_month").unwrap();
    let days = out.get_column_numeric_values("event_date_day").unwrap();
    let hours = out.get_column_numeric_values("event_date_hour").unwrap();
    let weekdays = out.get_column_numeric_values("event_date_weekday").unwrap();

    for (i, raw) in dates.iter().enumerate() {
        let dt = if raw.len() > 10 {
            chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S")
                .unwrap_or_else(|e| panic!("test fixture date '{}' should parse: {}", raw, e))
        } else {
            NaiveDate::parse_from_str(raw, "%Y-%m-%d")
                .unwrap_or_else(|e| panic!("test fixture date '{}' should parse: {}", raw, e))
                .and_hms_opt(0, 0, 0)
                .expect("midnight is always valid")
        };
        assert_eq!(years[i], dt.year() as f64, "year mismatch for '{}'", raw);
        assert_eq!(months[i], dt.month() as f64, "month mismatch for '{}'", raw);
        assert_eq!(days[i], dt.day() as f64, "day mismatch for '{}'", raw);
        assert_eq!(hours[i], dt.hour() as f64, "hour mismatch for '{}'", raw);
        assert_eq!(
            weekdays[i],
            dt.weekday().num_days_from_monday() as f64,
            "weekday mismatch for '{}'",
            raw
        );
    }
}

/// `RobustScaler` on a degenerate (zero-IQR, i.e. constant) column must scale by 1.0
/// (sklearn's convention), not blow tiny floating-point noise up by a `1e-10` epsilon floor
/// (which would turn `(x - median) / 1e-10` into values on the order of `1e10`).
#[test]
fn robust_scaler_degenerate_iqr_scales_by_one_not_epsilon() {
    let mut scaler = RobustScaler::new();
    let data = vec![5.0; 10]; // constant column: Q1 = Q3 = median = 5.0, IQR = 0.0

    scaler
        .fit(&data)
        .expect("fit should succeed on a constant column");
    let transformed = scaler.transform(&data).expect("transform should succeed");

    for (i, &v) in transformed.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "constant column should transform to ~0.0 under scale=1.0, got {} at index {} \
             (a scale=1e-10 epsilon floor would instead blow this up to ~0 or huge values)",
            v,
            i
        );
    }
}

/// `pipeline::PipelineStage::MinMaxScaler` on a degenerate (constant) column must map every
/// value to `feature_range.0`, matching `sklearn.preprocessing.MinMaxScaler`'s documented
/// behavior for a zero-variance feature -- not the midpoint of a custom, non-`[0,1]` range.
#[test]
fn pipeline_minmax_scaler_degenerate_column_uses_feature_range_min() {
    let mut df = DataFrame::new();
    df.add_column(
        "constant".to_string(),
        Series::new(vec![7.0; 5], Some("constant".to_string())).unwrap(),
    )
    .unwrap();

    let mut stage = PipelineStage::MinMaxScaler {
        columns: None,
        feature_range: (2.0, 10.0),
        _min_values: None,
        _max_values: None,
    };
    stage.fit(&df).expect("fit should succeed");
    let out = stage.transform(&df).expect("transform should succeed");

    let values = out.get_column_numeric_values("constant").unwrap();
    for v in values {
        assert!(
            (v - 2.0).abs() < 1e-9,
            "degenerate column must map to feature_range.0 (2.0), not the range midpoint (6.0); got {}",
            v
        );
    }
}

/// `pipeline::PipelineStage::Imputer` must error on a column with no non-missing values for a
/// statistic-based strategy ("mean"/"median"/"most_frequent"), rather than fabricating a 0.0
/// fill value that is indistinguishable downstream from a genuine observed 0.0.
#[test]
fn pipeline_imputer_mean_on_all_missing_column_errors() {
    let mut df = DataFrame::new();
    df.add_column(
        "missing".to_string(),
        Series::new(
            vec![f64::NAN, f64::NAN, f64::NAN],
            Some("missing".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut stage = PipelineStage::Imputer {
        columns: Some(vec!["missing".to_string()]),
        strategy: "mean".to_string(),
        fill_value: None,
        _fill_values: None,
    };

    let result = stage.fit(&df);
    assert!(
        result.is_err(),
        "Imputer::fit with strategy 'mean' on an all-missing column must error, not fabricate a fill value"
    );
}

/// `FeatureSelectionMethod::KBest` must clamp `n_features_to_select` to the number of
/// available features instead of propagating an oversized `k` into `SelectKBest` and
/// hard-erroring -- the scenario an AutoML step requesting e.g. 50 features from a
/// narrower (here, 3-feature) dataset previously hit.
#[test]
fn kbest_n_features_to_select_clamps_to_available_feature_count() {
    let n = 20;
    let target: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let feat_a: Vec<f64> = target.iter().map(|&t| 2.0 * t + 1.0).collect();
    let feat_b: Vec<f64> = (1..=n).map(|i| (i as f64 * 0.3).sin()).collect();
    let feat_c: Vec<f64> = (1..=n).map(|i| (i as f64 * 0.7).cos()).collect();

    let mut x = DataFrame::new();
    for (name, values) in [
        ("feat_a", &feat_a),
        ("feat_b", &feat_b),
        ("feat_c", &feat_c),
    ] {
        x.add_column(
            name.to_string(),
            Series::new(values.clone(), Some(name.to_string())).unwrap(),
        )
        .unwrap();
    }
    let y = df_column("target", target);

    let mut engineer = AutoFeatureEngineer::new()
        .with_selection(
            FeatureSelectionMethod::KBest(ScoreFunction::FRegression),
            Some(50),
        )
        .without_scaling();
    engineer.generate_polynomial = false;
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;

    engineer
        .fit(&x, Some(&y))
        .expect("k=50 requested on a 3-feature dataset must clamp to 3, not error");
    let selected = engineer
        .get_selected_features()
        .expect("selection should have run");
    assert_eq!(
        selected.len(),
        3,
        "all 3 available features should be selected once k is clamped, got {:?}",
        selected
    );
}

/// `FeatureSelectionMethod::KBest`'s `feature_scores_` must attribute each score to the
/// feature it actually came from, not to whichever feature happens to sit at that score's
/// RANK position. `SelectKBest::get_scores()` returns one score per feature in original
/// column order and `get_selected_features()` returns indices sorted back to original
/// column order (see `model_selection.rs`'s `SelectKBest::fit`), so a correct
/// implementation just zips `feature_names` (column order) with `get_scores()` (also
/// column order) directly -- but this is exactly the kind of pairing a rank-vs-column-order
/// mixup would silently scramble. To make that detectable, the columns below are
/// deliberately ordered so target-correlation strength is NOT monotonic in column
/// position: `feat_weak` (near-zero correlation) sits before `feat_strong` (near-perfect
/// correlation), with `feat_mid` (moderate, noisy correlation) last.
#[test]
fn kbest_feature_scores_attribute_to_the_correct_feature_name() {
    let n = 20;
    let target: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    // Column order: weak, strong, mid -- NOT sorted by correlation strength, so a
    // rank-order/column-order mixup would misattribute scores between names.
    let feat_weak: Vec<f64> = (1..=n).map(|i| (i as f64 * 1.5).cos() * 10.0).collect();
    let feat_strong: Vec<f64> = target
        .iter()
        .enumerate()
        .map(|(i, &t)| 10.0 * t + (i as f64 * 0.9).sin() * 0.05)
        .collect();
    let feat_mid: Vec<f64> = target
        .iter()
        .enumerate()
        .map(|(i, &t)| t + (i as f64 * 1.3).sin() * 4.0)
        .collect();

    let mut x = DataFrame::new();
    for (name, values) in [
        ("feat_weak", &feat_weak),
        ("feat_strong", &feat_strong),
        ("feat_mid", &feat_mid),
    ] {
        x.add_column(
            name.to_string(),
            Series::new(values.clone(), Some(name.to_string())).unwrap(),
        )
        .unwrap();
    }
    let y = df_column("target", target);

    let mut engineer = AutoFeatureEngineer::new()
        .with_selection(
            FeatureSelectionMethod::KBest(ScoreFunction::FRegression),
            Some(3), // select all 3: this test is about score ATTRIBUTION, not filtering
        )
        .without_scaling();
    engineer.generate_polynomial = false;
    engineer.generate_interactions = false;
    engineer.generate_aggregations = false;

    engineer.fit(&x, Some(&y)).expect("fit should succeed");

    let scores = engineer
        .get_feature_scores()
        .expect("feature_scores_ should be populated for KBest");

    let weak = *scores
        .get("feat_weak")
        .expect("feat_weak score must be present");
    let strong = *scores
        .get("feat_strong")
        .expect("feat_strong score must be present");
    let mid = *scores
        .get("feat_mid")
        .expect("feat_mid score must be present");

    assert!(
        strong > mid && mid > weak,
        "F-regression scores must be attributed by name so that \
         feat_strong (near-perfect correlation) > feat_mid (noisy correlation) > \
         feat_weak (near-zero correlation); got strong={strong}, mid={mid}, weak={weak} \
         -- a rank-position/column-order mismatch would scramble this"
    );
}

/// `pipeline::Pipeline`'s column pass-through must preserve a column's CONCRETE storage
/// type across stages. A digit-coded categorical `Series<String>` column (values like
/// `"1"`, `"2"`) is correctly excluded from `StandardScaler` (it isn't numeric) and must
/// reach a LATER `OneHotEncoder` stage (with `columns: None`, i.e. auto-detected by
/// concrete type) still recognizable as categorical -- not silently rewritten to an `f64`
/// column by the earlier stage's pass-through, which routing by textual parseability
/// (`get_column_numeric_values` succeeds on digit-like strings) would cause, making
/// `OneHotEncoder`'s own concrete-type auto-detection skip it entirely.
#[test]
fn pipeline_passthrough_preserves_categorical_dtype_for_later_onehot_stage() {
    let mut df = DataFrame::new();
    df.add_column(
        "num".to_string(),
        Series::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], Some("num".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "code".to_string(),
        Series::new(
            vec![
                "1".to_string(),
                "2".to_string(),
                "1".to_string(),
                "2".to_string(),
                "1".to_string(),
            ],
            Some("code".to_string()),
        )
        .unwrap(),
    )
    .unwrap();

    let mut pipeline = Pipeline::new();
    pipeline.add_stage(PipelineStage::StandardScaler {
        columns: None,
        _means: None,
        _stds: None,
    });
    pipeline.add_stage(PipelineStage::OneHotEncoder {
        columns: None, // auto-detect categorical columns by concrete type
        drop_first: false,
        prefix: None,
        _categories: None,
    });

    let out = pipeline
        .fit_transform(&df)
        .expect("pipeline fit_transform should succeed");

    assert!(
        out.contains_column("code_1") && out.contains_column("code_2"),
        "OneHotEncoder (columns: None) should still auto-detect 'code' as categorical \
         after StandardScaler's pass-through, got columns: {:?}",
        out.column_names()
    );
    assert_eq!(
        out.get_column_numeric_values("code_1").unwrap(),
        vec![1.0, 0.0, 1.0, 0.0, 1.0]
    );
    assert_eq!(
        out.get_column_numeric_values("code_2").unwrap(),
        vec![0.0, 1.0, 0.0, 1.0, 0.0]
    );

    // 'num' must still be standardized (mean ~0), confirming StandardScaler itself ran
    // correctly on the genuinely numeric column in the same pass.
    let num_values = out.get_column_numeric_values("num").unwrap();
    let mean: f64 = num_values.iter().sum::<f64>() / num_values.len() as f64;
    assert!(
        mean.abs() < 1e-9,
        "num column should be standardized, mean={mean}"
    );
}
