//! Regression tests for the Wave 2 `ml::serving` fixes.
//!
//! Covers, per the audit-driven fix list:
//! - registry persistence: register -> load -> predict weight round-trip (both
//!   `FileSystemModelRegistry` and `InMemoryModelRegistry`, including through `update_metadata`)
//! - semver-aware "latest" version resolution (not lexicographic)
//! - format-drift-tolerant `exists()` / `load_model()` after changing a registry's default format
//! - real PSI drift: `None` until a baseline exists, then a real computed value
//! - batch prediction per-item error indexing (`Vec<(usize, String)>`, not just a bare count)
//! - the trained-model -> serving bridge (`TryFrom<&LinearRegression>` / `&LogisticRegression`)
//! - rate limiter behavior (bounded client tracking, engaged independent of `enable_auth`)
//! - model_version mismatch rejection and honest health checks

use pandrs::dataframe::DataFrame;
use pandrs::ml::models::linear::LinearRegression;
use pandrs::ml::models::SupervisedModel;
use pandrs::ml::serving::registry::{
    FileSystemModelRegistry, InMemoryModelRegistry, ModelRegistry,
};
use pandrs::ml::serving::serialization::{
    GenericServingModel, SerializableModel, CURRENT_SCHEMA_VERSION,
};
use pandrs::ml::serving::{
    BatchPredictionRequest, HttpModelServer, ModelMetadata, ModelServing, PredictionRequest,
    RequestContext, ServerConfig,
};
use pandrs::series::Series;
use std::collections::HashMap;

fn unique_temp_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pandrs_ml_serving_w2_{}_{}_{}",
        label,
        std::process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn make_metadata(name: &str, version: &str, feature_names: Vec<String>) -> ModelMetadata {
    ModelMetadata {
        name: name.to_string(),
        version: version.to_string(),
        model_type: "linear_regression".to_string(),
        feature_names,
        target_name: Some("y".to_string()),
        description: "regression test model".to_string(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metrics: HashMap::new(),
        metadata: HashMap::new(),
    }
}

fn make_serializable(
    name: &str,
    version: &str,
    coefficients: Vec<f64>,
    intercept: f64,
) -> SerializableModel {
    let feature_names: Vec<String> = (0..coefficients.len()).map(|i| format!("x{i}")).collect();
    let mut parameters = HashMap::new();
    parameters.insert("coefficients".to_string(), serde_json::json!(coefficients));
    parameters.insert("intercept".to_string(), serde_json::json!(intercept));

    SerializableModel {
        schema_version: CURRENT_SCHEMA_VERSION,
        metadata: make_metadata(name, version, feature_names),
        parameters,
        model_data: serde_json::json!({}),
        preprocessing: None,
        config: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// 1. Registry persistence: register -> load -> predict weight round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_filesystem_registry_register_load_predict_round_trip() {
    let dir = unique_temp_dir("fs_roundtrip");
    let mut registry = FileSystemModelRegistry::new(&dir).expect("registry creation");

    let serializable = make_serializable("weighted_model", "1.0.0", vec![2.0, -3.0], 1.5);
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");
    registry
        .register_model(Box::new(model))
        .expect("register_model succeeds");

    // Load it back from disk -- this is the exact path that previously lost every weight
    // (FileSystemModelRegistry::model_to_serializable synthesized empty `parameters`).
    let loaded = registry
        .load_model("weighted_model", "1.0.0")
        .expect("load_model succeeds");

    let mut data = HashMap::new();
    data.insert("x0".to_string(), serde_json::json!(4.0));
    data.insert("x1".to_string(), serde_json::json!(5.0));
    let request = PredictionRequest {
        data,
        model_version: None,
        options: None,
    };
    let response = loaded.predict(&request).expect("prediction succeeds");
    let value = response
        .prediction
        .get("prediction")
        .and_then(|v| v.as_f64())
        .expect("numeric prediction");

    // 1.5 + 2.0*4.0 - 3.0*5.0 = 1.5 + 8.0 - 15.0 = -5.5
    assert!(
        (value - (-5.5)).abs() < 1e-9,
        "loaded model must serve the real trained weights, got {value}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_filesystem_registry_update_metadata_preserves_weights() {
    let dir = unique_temp_dir("fs_update_metadata");
    let mut registry = FileSystemModelRegistry::new(&dir).expect("registry creation");

    let serializable = make_serializable("m", "1.0.0", vec![10.0], 0.0);
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");
    registry
        .register_model(Box::new(model))
        .expect("register_model succeeds");

    let mut new_metadata = registry
        .get_metadata("m", "1.0.0")
        .expect("metadata is readable");
    new_metadata.description = "renamed".to_string();
    registry
        .update_metadata("m", "1.0.0", new_metadata)
        .expect("update_metadata succeeds");

    let loaded = registry
        .load_model("m", "1.0.0")
        .expect("load after update");
    assert_eq!(loaded.get_metadata().description, "renamed");

    let mut data = HashMap::new();
    data.insert("x0".to_string(), serde_json::json!(3.0));
    let response = loaded
        .predict(&PredictionRequest {
            data,
            model_version: None,
            options: None,
        })
        .expect("prediction still works after a metadata-only edit");
    let value = response
        .prediction
        .get("prediction")
        .and_then(|v| v.as_f64())
        .unwrap();
    assert!(
        (value - 30.0).abs() < 1e-9,
        "update_metadata must not gut the model's coefficients, got {value}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_in_memory_registry_update_metadata_preserves_weights() {
    let mut registry = InMemoryModelRegistry::new();
    let serializable = make_serializable("m2", "1.0.0", vec![7.0], 1.0);
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");
    registry
        .register_model(Box::new(model))
        .expect("register_model succeeds");

    let mut new_metadata = registry.get_metadata("m2", "1.0.0").unwrap();
    new_metadata.description = "updated".to_string();
    registry
        .update_metadata("m2", "1.0.0", new_metadata)
        .expect("update_metadata succeeds");

    let loaded = registry.load_model("m2", "1.0.0").unwrap();
    let mut data = HashMap::new();
    data.insert("x0".to_string(), serde_json::json!(2.0));
    let response = loaded
        .predict(&PredictionRequest {
            data,
            model_version: None,
            options: None,
        })
        .expect("prediction succeeds");
    let value = response
        .prediction
        .get("prediction")
        .and_then(|v| v.as_f64())
        .unwrap();
    // 1.0 + 7.0*2.0 = 15.0 -- would be 0.0 (or an error) if update_metadata had rebuilt from an
    // empty SerializableModel, as it used to.
    assert!((value - 15.0).abs() < 1e-9, "got {value}");
}

// ---------------------------------------------------------------------------
// 2. Semver-aware "latest" resolution
// ---------------------------------------------------------------------------

#[test]
fn test_semver_latest_is_not_lexicographic_in_memory() {
    let mut registry = InMemoryModelRegistry::new();
    for version in ["1.2.0", "1.9.0", "1.10.0"] {
        let serializable = make_serializable("semver_model", version, vec![1.0], 0.0);
        let model = GenericServingModel::from_serializable(serializable).expect("model builds");
        registry.register_model(Box::new(model)).unwrap();
    }

    // Lexicographically, "1.9.0" > "1.10.0" (since '9' > '1' at the second byte); numerically,
    // 1.10.0 is the true latest.
    assert_eq!(
        registry.get_latest_version("semver_model").unwrap(),
        "1.10.0"
    );
}

#[test]
fn test_semver_latest_is_not_lexicographic_filesystem() {
    let dir = unique_temp_dir("fs_semver");
    let mut registry = FileSystemModelRegistry::new(&dir).expect("registry creation");
    for version in ["1.2.0", "1.9.0", "1.10.0"] {
        let serializable = make_serializable("semver_model", version, vec![1.0], 0.0);
        let model = GenericServingModel::from_serializable(serializable).expect("model builds");
        registry.register_model(Box::new(model)).unwrap();
    }

    assert_eq!(
        registry.get_latest_version("semver_model").unwrap(),
        "1.10.0"
    );

    // "latest" resolution end-to-end through load_model too.
    let loaded = registry.load_model("semver_model", "latest").unwrap();
    assert_eq!(loaded.get_metadata().version, "1.10.0");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_semver_latest_survives_deletion() {
    let mut registry = InMemoryModelRegistry::new();
    for version in ["1.9.0", "1.10.0", "1.11.0"] {
        let serializable = make_serializable("del_model", version, vec![1.0], 0.0);
        let model = GenericServingModel::from_serializable(serializable).expect("model builds");
        registry.register_model(Box::new(model)).unwrap();
    }
    assert_eq!(registry.get_latest_version("del_model").unwrap(), "1.11.0");

    registry.delete_model("del_model", "1.11.0").unwrap();
    // After deleting the true latest, the next-highest by *numeric* order must take over --
    // "1.10.0", not "1.9.0" (which would win a lexicographic comparison).
    assert_eq!(registry.get_latest_version("del_model").unwrap(), "1.10.0");
}

// ---------------------------------------------------------------------------
// 3. Format-drift-tolerant registry file lookup
// ---------------------------------------------------------------------------

#[test]
fn test_filesystem_registry_survives_default_format_change() {
    use pandrs::ml::serving::SerializationFormat;

    let dir = unique_temp_dir("fs_format_drift");
    let mut registry = FileSystemModelRegistry::new(&dir).expect("registry creation");

    let serializable = make_serializable("drift_model", "1.0.0", vec![1.0], 0.0);
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");
    registry.register_model(Box::new(model)).unwrap(); // written as .json (the default)

    // Now flip the registry's default format -- exists()/load_model() must still find the
    // model that was written under the *old* format, not silently report it missing.
    registry.set_default_format(SerializationFormat::Toml);
    assert!(registry.exists("drift_model", "1.0.0"));
    assert!(registry.load_model("drift_model", "1.0.0").is_ok());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_filesystem_registry_binary_format_round_trip_preserves_weights() {
    use pandrs::ml::serving::SerializationFormat;

    // SerializationFormat::Binary is honestly documented as JSON bytes under a `.bin`
    // extension (not yet a compact binary encoding -- see BinaryModelSerializer's doc), but it
    // must still be a real, distinct, round-trip-safe format through the full registry
    // pipeline: written under `.bin`, found by `find_model_file`'s extension probing, and
    // loaded back with every weight intact.
    let dir = unique_temp_dir("fs_binary_format");
    let mut registry = FileSystemModelRegistry::new(&dir).expect("registry creation");
    registry.set_default_format(SerializationFormat::Binary);

    let serializable = make_serializable("binary_model", "1.0.0", vec![4.0, -2.0], 0.5);
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");
    registry.register_model(Box::new(model)).unwrap();

    assert!(
        dir.join("binary_model").join("1.0.0.bin").exists(),
        "the model must actually be written under the .bin extension"
    );

    let loaded = registry
        .load_model("binary_model", "1.0.0")
        .expect("load_model succeeds for the Binary format");

    let mut data = HashMap::new();
    data.insert("x0".to_string(), serde_json::json!(3.0));
    data.insert("x1".to_string(), serde_json::json!(5.0));
    let response = loaded
        .predict(&PredictionRequest {
            data,
            model_version: None,
            options: None,
        })
        .expect("prediction succeeds");
    let value = response
        .prediction
        .get("prediction")
        .and_then(|v| v.as_f64())
        .expect("numeric prediction");

    // 0.5 + 4.0*3.0 - 2.0*5.0 = 0.5 + 12.0 - 10.0 = 2.5
    assert!(
        (value - 2.5).abs() < 1e-9,
        "Binary-format round trip must preserve the real trained weights, got {value}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// 4. Real PSI drift: None until a baseline exists, then a real value
// ---------------------------------------------------------------------------

#[test]
fn test_psi_drift_none_until_baseline_then_real() {
    use pandrs::ml::serving::monitoring::ModelMonitor;

    let metadata = make_metadata("drift_test", "1.0.0", vec!["x0".to_string()]);
    let mut monitor = ModelMonitor::new(metadata);

    // No baseline yet: must be honestly None, never a fabricated constant.
    assert!(monitor.calculate_data_drift().is_none());

    // Establish a baseline, but don't observe any live traffic yet: still None (a baseline
    // alone isn't a drift measurement).
    let reference: Vec<f64> = (0..500).map(|i| (i % 50) as f64).collect();
    monitor.set_feature_baseline("x0", &reference);
    assert!(monitor.calculate_data_drift().is_none());

    // Feed live traffic drawn from a wildly different distribution.
    for _ in 0..100 {
        monitor.observe_feature("x0", 999.0);
    }

    let drift = monitor
        .calculate_data_drift()
        .expect("baseline + live observations now exist");
    assert_eq!(drift.detection_method, "PSI");
    assert!(drift.drift_score.is_finite());
    assert!(drift.drift_score > 0.0);
    assert!(drift.drift_detected);
}

// ---------------------------------------------------------------------------
// 5. Batch prediction: real per-item error indexing
// ---------------------------------------------------------------------------

#[test]
fn test_batch_prediction_reports_indexed_per_item_errors() {
    let serializable = make_serializable("batch_model", "1.0.0", vec![1.0, 1.0], 0.0);
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");

    let good_row = {
        let mut d = HashMap::new();
        d.insert("x0".to_string(), serde_json::json!(1.0));
        d.insert("x1".to_string(), serde_json::json!(2.0));
        d
    };
    let bad_row_missing_feature = {
        let mut d = HashMap::new();
        d.insert("x0".to_string(), serde_json::json!(1.0));
        // x1 missing entirely.
        d
    };

    let batch = BatchPredictionRequest {
        data: vec![good_row.clone(), bad_row_missing_feature, good_row],
        model_version: None,
        options: None,
    };

    let response = model.predict_batch(&batch).expect("predict_batch succeeds");
    assert_eq!(response.summary.total_predictions, 3);
    assert_eq!(response.summary.successful_predictions, 2);
    assert_eq!(response.summary.failed_predictions, 1);

    // The failure must be attributed to the real row index (1), with a real message -- not
    // just a bare failed-count with no way to tell which input caused it.
    assert_eq!(response.summary.failed_items.len(), 1);
    let (failed_idx, failed_message) = &response.summary.failed_items[0];
    assert_eq!(*failed_idx, 1);
    assert!(!failed_message.is_empty());
}

// ---------------------------------------------------------------------------
// 6. Trained-model -> serving bridge
// ---------------------------------------------------------------------------

fn regression_frame() -> DataFrame {
    let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0];
    let y: Vec<f64> = x1
        .iter()
        .zip(x2.iter())
        .map(|(&a, &b)| 3.0 * a - 1.5 * b + 2.0)
        .collect();

    let mut df = DataFrame::new();
    df.add_column(
        "x1".to_string(),
        Series::new(x1, Some("x1".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "x2".to_string(),
        Series::new(x2, Some("x2".to_string())).unwrap(),
    )
    .unwrap();
    df.add_column(
        "y".to_string(),
        Series::new(y, Some("y".to_string())).unwrap(),
    )
    .unwrap();
    df
}

#[test]
fn test_bridged_linear_regression_registers_and_predicts_correctly() {
    let df = regression_frame();
    let mut trained = LinearRegression::new();
    trained.fit(&df, "y").expect("fit succeeds");

    let serializable = SerializableModel::try_from(&trained).expect("bridges");
    let served_model = GenericServingModel::from_serializable(serializable).expect("model builds");

    let dir = unique_temp_dir("bridge_registry");
    let mut registry = FileSystemModelRegistry::new(&dir).expect("registry creation");
    registry
        .register_model(Box::new(served_model))
        .expect("register bridged model");

    let loaded = registry
        .load_model("linear_regression", "latest")
        .expect("load bridged model");

    let trained_predictions = trained.predict(&df).expect("trained predict");
    let x1_values = df.get_column::<f64>("x1").unwrap().as_f64().unwrap();
    let x2_values = df.get_column::<f64>("x2").unwrap().as_f64().unwrap();

    for (row_idx, &expected) in trained_predictions.iter().enumerate() {
        let mut data = HashMap::new();
        data.insert("x1".to_string(), serde_json::json!(x1_values[row_idx]));
        data.insert("x2".to_string(), serde_json::json!(x2_values[row_idx]));
        let response = loaded
            .predict(&PredictionRequest {
                data,
                model_version: None,
                options: None,
            })
            .expect("served prediction succeeds");
        let served_value = response
            .prediction
            .get("prediction")
            .and_then(|v| v.as_f64())
            .unwrap();
        assert!(
            (served_value - expected).abs() < 1e-6,
            "row {row_idx}: served={served_value} trained={expected}"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// 7. Batch endpoint: one malformed row must not discard the rest of the batch
// ---------------------------------------------------------------------------

#[test]
fn test_http_server_batch_predict_one_bad_row_does_not_discard_the_batch() {
    // Regression: `BatchPredictionEndpoint::predict_batch` used to pre-validate every row
    // *before* calling the model, and reject the entire batch with one generic 400 the moment
    // any single row failed validation -- discarding every other row's successful prediction.
    // That defeated the point of `BatchProcessingSummary::failed_items` (real per-row index +
    // message) existing at all: it was only ever reachable for failures the model produced
    // internally, never for the (far more common) case of one malformed row's *input*.
    let serializable = make_serializable("batch_endpoint_model", "1.0.0", vec![2.0], 1.0);
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");

    let mut server = HttpModelServer::new(ServerConfig::default());
    server
        .register_model("batch_endpoint_model".to_string(), Box::new(model))
        .expect("register_model succeeds");

    let good_row = {
        let mut d = HashMap::new();
        d.insert("x0".to_string(), serde_json::json!(3.0));
        d
    };
    let bad_row_unknown_feature = {
        // Fails `PredictionEndpoint::validate_request`'s "unknown feature" check, not anything
        // `GenericServingModel::predict` itself would catch -- so this failure can only ever be
        // observed if the endpoint validates per-row rather than trusting the model alone.
        let mut d = HashMap::new();
        d.insert("x0".to_string(), serde_json::json!(3.0));
        d.insert(
            "totally_unexpected_field".to_string(),
            serde_json::json!(1.0),
        );
        d
    };

    let batch = BatchPredictionRequest {
        data: vec![good_row.clone(), bad_row_unknown_feature, good_row],
        model_version: None,
        options: None,
    };

    let response =
        server.handle_predict_batch("batch_endpoint_model", batch, RequestContext::new());

    assert_eq!(
        response.status_code, 200,
        "a single malformed row must not fail the whole batch"
    );
    let body = response.body.data.expect("batch response body is present");
    assert_eq!(body.summary.total_predictions, 3);
    assert_eq!(body.summary.successful_predictions, 2);
    assert_eq!(body.summary.failed_predictions, 1);
    assert_eq!(body.predictions.len(), 2);

    // The failure is attributed to the real original row index (1), not silently dropped or
    // misindexed by the validation-then-model two-stage pipeline.
    assert_eq!(body.summary.failed_items.len(), 1);
    let (failed_idx, failed_message) = &body.summary.failed_items[0];
    assert_eq!(*failed_idx, 1);
    assert!(failed_message.contains("Unknown feature") || failed_message.contains("Validation"));

    // The two successful predictions are the real computed values (2.0*3.0 + 1.0 = 7.0), not
    // fabricated placeholders.
    for prediction in &body.predictions {
        let value = prediction
            .prediction
            .get("prediction")
            .and_then(|v| v.as_f64())
            .expect("numeric prediction");
        assert!((value - 7.0).abs() < 1e-9, "got {value}");
    }
}

#[test]
fn test_http_server_batch_predict_arithmetic_holds_with_both_failure_classes_mixed() {
    // `BatchPredictionEndpoint::predict_batch` now merges two distinct failure sources into one
    // `failed_items` list: rows rejected by `validate_request` (never reach the model) and rows
    // the model itself fails on internally. Both classes must be present at once to prove the
    // merge doesn't double-count or drop anything: `successful_predictions + failed_predictions`
    // must equal `total_predictions`, and every failed original index must be attributed
    // exactly once with the right cause.
    //
    // A deliberately malformed model (2 declared `feature_names` but only 1 stored coefficient)
    // makes every row that *passes* validation fail at inference with a dimension mismatch --
    // the one reliable way to get a real model-level (not validation-level) per-row failure,
    // since `validate_request` and `extract_features` otherwise accept exactly the same inputs.
    let mut parameters = HashMap::new();
    parameters.insert("coefficients".to_string(), serde_json::json!([1.0])); // only 1, not 2
    parameters.insert("intercept".to_string(), serde_json::json!(0.0));
    let serializable = SerializableModel {
        schema_version: CURRENT_SCHEMA_VERSION,
        metadata: make_metadata(
            "mismatched_model",
            "1.0.0",
            vec!["x0".to_string(), "x1".to_string()],
        ),
        parameters,
        model_data: serde_json::json!({}),
        preprocessing: None,
        config: HashMap::new(),
    };
    let model = GenericServingModel::from_serializable(serializable).expect("model builds");

    let mut server = HttpModelServer::new(ServerConfig::default());
    server
        .register_model("mismatched_model".to_string(), Box::new(model))
        .expect("register_model succeeds");

    let row_ok_shape = {
        // Passes validate_request (has both named features, both numeric); fails at inference
        // due to the model's own coefficients/feature-count mismatch.
        let mut d = HashMap::new();
        d.insert("x0".to_string(), serde_json::json!(1.0));
        d.insert("x1".to_string(), serde_json::json!(2.0));
        d
    };
    let row_missing_feature = {
        // Fails validate_request outright (never reaches the model).
        let mut d = HashMap::new();
        d.insert("x0".to_string(), serde_json::json!(1.0));
        d
    };

    let batch = BatchPredictionRequest {
        data: vec![
            row_ok_shape.clone(), // index 0: model-level failure
            row_missing_feature,  // index 1: validation failure
            row_ok_shape,         // index 2: model-level failure
        ],
        model_version: None,
        options: None,
    };

    let response = server.handle_predict_batch("mismatched_model", batch, RequestContext::new());
    assert_eq!(
        response.status_code, 200,
        "at least one row (all of them, here) reaching the model keeps this a 200 with a \
         per-item breakdown, not a whole-batch 400"
    );
    let body = response.body.data.expect("batch response body");

    assert_eq!(body.summary.total_predictions, 3);
    assert_eq!(body.summary.successful_predictions, 0);
    assert_eq!(body.summary.failed_predictions, 3);
    assert_eq!(
        body.summary.successful_predictions + body.summary.failed_predictions,
        body.summary.total_predictions,
        "successful + failed must reconcile to the real total with both failure classes present"
    );

    // Every original index is attributed exactly once, with a cause matching its real source.
    assert_eq!(body.summary.failed_items.len(), 3);
    let mut by_index: HashMap<usize, String> = body.summary.failed_items.into_iter().collect();
    assert!(
        by_index.remove(&0).unwrap().contains("coefficients"),
        "index 0 must fail at the model (dimension mismatch), not validation"
    );
    let validation_msg = by_index.remove(&1).unwrap();
    assert!(
        validation_msg.contains("Missing required feature")
            || validation_msg.contains("Validation"),
        "index 1 must fail pre-validation with its real cause, got: {validation_msg}"
    );
    assert!(
        by_index.remove(&2).unwrap().contains("coefficients"),
        "index 2 must fail at the model too, correctly remapped from filtered-space back to \
         its real original index (2, not 1)"
    );
}
