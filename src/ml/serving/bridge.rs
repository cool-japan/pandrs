//! Bridges from trained `ml::models` estimators to the serving layer.
//!
//! Before this module existed, nothing in `ml::models` could be handed to `ml::serving` at
//! all: a caller who trained a [`LinearRegression`] or [`LogisticRegression`] had no supported
//! path to a [`SerializableModel`], and a hand-rolled one would have gotten the coefficient
//! layout wrong (the trained models store weights as a `HashMap<String, f64>` keyed by feature
//! name, while [`crate::ml::serving::serialization::GenericServingModel`] inference expects an
//! **ordered** `coefficients` array aligned index-for-index with `metadata.feature_names`).
//!
//! # Why `TryFrom`, not `From`
//!
//! An unfitted model (`coefficients: None`) cannot be converted into a servable model at all.
//! Modeling that as `From` would force either a panic/`unwrap()` (forbidden in production code)
//! or a fabricated all-zero model. `TryFrom` reports the real failure instead.
//!
//! # Why the feature order doesn't need to match training order
//!
//! `LinearRegression`/`LogisticRegression` don't publicly expose the original training column
//! order (only the fitted `coefficients: Option<HashMap<String, f64>>` and
//! `intercept: Option<f64>` are `pub`). This module instead sorts feature names
//! lexicographically to build a deterministic order, and uses that *same* order for both
//! `metadata.feature_names` and the `coefficients` array. Because `GenericServingModel`
//! inference always re-extracts input features **by name** against `metadata.feature_names`
//! (see `extract_features`), any consistent order is correct: there is no dependency on
//! matching the original training column order.

use crate::core::error::{Error, Result};
use crate::ml::models::linear::{LinearRegression, LogisticRegression};
use crate::ml::serving::serialization::{SerializableModel, CURRENT_SCHEMA_VERSION};
use crate::ml::serving::ModelMetadata;
use std::collections::HashMap;

/// Flatten a fitted coefficient map into a `(feature_names, coefficients)` pair, with both
/// ordered consistently by lexicographically-sorted feature name.
fn ordered_coefficients(coefficients: &HashMap<String, f64>) -> (Vec<String>, Vec<f64>) {
    let mut names: Vec<String> = coefficients.keys().cloned().collect();
    names.sort();
    let values: Vec<f64> = names.iter().map(|name| coefficients[name]).collect();
    (names, values)
}

/// Build the common `ModelMetadata` shell for a bridged model. Callers overlay any
/// caller-specific fields (name/version/target) they care about afterward.
fn bridged_metadata(model_type: &str, feature_names: Vec<String>) -> ModelMetadata {
    ModelMetadata {
        name: model_type.to_string(),
        version: "0.0.0".to_string(),
        model_type: model_type.to_string(),
        feature_names,
        target_name: None,
        description: format!(
            "Bridged from pandrs::ml::models::linear::{}",
            match model_type {
                "logistic_regression" => "LogisticRegression",
                _ => "LinearRegression",
            }
        ),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        metrics: HashMap::new(),
        metadata: HashMap::new(),
    }
}

impl TryFrom<&LinearRegression> for SerializableModel {
    type Error = Error;

    /// Convert a fitted [`LinearRegression`] into a servable model.
    ///
    /// # Errors
    /// Returns [`Error::InvalidOperation`] if `model` has not been fitted (`coefficients` is
    /// `None`) -- there is nothing real to serve yet.
    fn try_from(model: &LinearRegression) -> Result<Self> {
        let coefficients = model.coefficients.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "Cannot bridge an unfitted LinearRegression to a servable model; call fit() first"
                    .into(),
            )
        })?;
        let (feature_names, coeffs) = ordered_coefficients(coefficients);

        let mut parameters = HashMap::new();
        parameters.insert("coefficients".to_string(), serde_json::json!(coeffs));
        parameters.insert(
            "intercept".to_string(),
            serde_json::json!(model.intercept.unwrap_or(0.0)),
        );

        Ok(SerializableModel {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata: bridged_metadata("linear_regression", feature_names),
            parameters,
            model_data: serde_json::json!({"source": "ml::models::linear::LinearRegression"}),
            preprocessing: None,
            config: HashMap::new(),
        })
    }
}

impl TryFrom<&LogisticRegression> for SerializableModel {
    type Error = Error;

    /// Convert a fitted [`LogisticRegression`] into a servable binary-classification model.
    ///
    /// `LogisticRegression` fits on `{0, 1}`-clamped targets (see its `fit()`), so the two
    /// classes are labeled `"0"`/`"1"` -- matching what
    /// [`SupervisedModel::predict`](crate::ml::models::SupervisedModel::predict) itself returns (thresholded at
    /// 0.5), which is what the round-trip regression test compares against.
    ///
    /// # Errors
    /// Returns [`Error::InvalidOperation`] if `model` has not been fitted.
    fn try_from(model: &LogisticRegression) -> Result<Self> {
        let coefficients = model.coefficients.as_ref().ok_or_else(|| {
            Error::InvalidOperation(
                "Cannot bridge an unfitted LogisticRegression to a servable model; call fit() first"
                    .into(),
            )
        })?;
        let (feature_names, coeffs) = ordered_coefficients(coefficients);

        let mut parameters = HashMap::new();
        parameters.insert("coefficients".to_string(), serde_json::json!(coeffs));
        parameters.insert(
            "intercept".to_string(),
            serde_json::json!(model.intercept.unwrap_or(0.0)),
        );
        parameters.insert("classes".to_string(), serde_json::json!(["0", "1"]));

        Ok(SerializableModel {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata: bridged_metadata("logistic_regression", feature_names),
            parameters,
            model_data: serde_json::json!({"source": "ml::models::linear::LogisticRegression"}),
            preprocessing: None,
            config: HashMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::DataFrame;
    use crate::ml::models::linear::{LinearRegression, LogisticRegression};
    use crate::ml::models::SupervisedModel;
    use crate::ml::serving::serialization::GenericServingModel;
    use crate::ml::serving::{ModelServing, PredictionRequest};
    use crate::series::Series;

    /// Build a small synthetic regression frame: y = 2*x1 - 3*x2 + 1 (+ tiny noise-free offset).
    fn regression_frame() -> DataFrame {
        let x1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x2 = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let y: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .map(|(&a, &b)| 2.0 * a - 3.0 * b + 1.0)
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

    /// Build a small, linearly-separable synthetic binary classification frame.
    fn classification_frame() -> DataFrame {
        let x1 = vec![0.0, 0.5, 1.0, 4.0, 4.5, 5.0];
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut df = DataFrame::new();
        df.add_column(
            "x1".to_string(),
            Series::new(x1, Some("x1".to_string())).unwrap(),
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
    fn test_unfitted_linear_regression_bridge_fails_honestly() {
        let model = LinearRegression::new();
        let result = SerializableModel::try_from(&model);
        assert!(result.is_err(), "unfitted model must not bridge silently");
    }

    #[test]
    fn test_unfitted_logistic_regression_bridge_fails_honestly() {
        let model = LogisticRegression::new();
        let result = SerializableModel::try_from(&model);
        assert!(result.is_err(), "unfitted model must not bridge silently");
    }

    #[test]
    fn test_linear_regression_round_trip_predict_matches() {
        let df = regression_frame();
        let mut trained = LinearRegression::new();
        trained.fit(&df, "y").expect("fit succeeds");

        let serializable = SerializableModel::try_from(&trained).expect("bridges successfully");
        let served = GenericServingModel::from_serializable(serializable).expect("model builds");

        let trained_predictions = trained.predict(&df).expect("trained predict succeeds");

        for (row_idx, &expected) in trained_predictions.iter().enumerate() {
            let mut data = std::collections::HashMap::new();
            data.insert(
                "x1".to_string(),
                serde_json::json!(df.get_column::<f64>("x1").unwrap().as_f64().unwrap()[row_idx]),
            );
            data.insert(
                "x2".to_string(),
                serde_json::json!(df.get_column::<f64>("x2").unwrap().as_f64().unwrap()[row_idx]),
            );
            let request = PredictionRequest {
                data,
                model_version: None,
                options: None,
            };
            let response = served.predict(&request).expect("served predict succeeds");
            let served_value = response
                .prediction
                .get("prediction")
                .and_then(|v| v.as_f64())
                .expect("regression prediction is numeric");

            assert!(
                (served_value - expected).abs() < 1e-9,
                "row {row_idx}: served={served_value} trained={expected}"
            );
        }
    }

    #[test]
    fn test_logistic_regression_round_trip_predict_matches() {
        let df = classification_frame();
        let mut trained = LogisticRegression::new();
        trained.fit(&df, "y").expect("fit succeeds");

        let serializable = SerializableModel::try_from(&trained).expect("bridges successfully");
        let served = GenericServingModel::from_serializable(serializable).expect("model builds");

        let trained_predictions = trained.predict(&df).expect("trained predict succeeds");
        let trained_probabilities = trained.predict_proba(&df).expect("predict_proba succeeds");

        let x1_values = df.get_column::<f64>("x1").unwrap().as_f64().unwrap();
        for (row_idx, (&expected_label, &expected_proba)) in trained_predictions
            .iter()
            .zip(trained_probabilities.iter())
            .enumerate()
        {
            let mut data = std::collections::HashMap::new();
            data.insert("x1".to_string(), serde_json::json!(x1_values[row_idx]));
            let request = PredictionRequest {
                data,
                model_version: None,
                options: Some(crate::ml::serving::PredictionOptions {
                    include_probabilities: Some(true),
                    include_feature_importance: None,
                    include_confidence_intervals: None,
                    threshold: None,
                }),
            };
            let response = served.predict(&request).expect("served predict succeeds");

            let served_label_str = response
                .prediction
                .get("prediction")
                .and_then(|v| v.as_str())
                .expect("classification prediction is a string label");
            let served_label: f64 = served_label_str.parse().expect("label parses as 0/1");
            assert_eq!(
                served_label, expected_label,
                "row {row_idx}: served label {served_label} vs trained label {expected_label}"
            );

            let served_p1 = response
                .probabilities
                .as_ref()
                .and_then(|p| p.get("1"))
                .copied()
                .expect("probability for class '1' is present");
            assert!(
                (served_p1 - expected_proba).abs() < 1e-9,
                "row {row_idx}: served p(1)={served_p1} trained p={expected_proba}"
            );
        }
    }
}
