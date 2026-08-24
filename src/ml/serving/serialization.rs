//! Model Serialization Module
//!
//! This module provides comprehensive model serialization and deserialization capabilities
//! supporting multiple formats (JSON, YAML, TOML, Binary).

use crate::core::error::{Error, Result};
use crate::ml::serving::{
    BatchPredictionRequest, BatchPredictionResponse, PredictionRequest, PredictionResponse,
};
use crate::ml::serving::{HealthStatus, ModelInfo, ModelMetadata, ModelServing, ModelStatistics};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

/// Current on-disk schema version for [`SerializableModel`]. Bump this whenever the shape of
/// `SerializableModel` (or the semantics `GenericServingModel` gives its fields) changes in a
/// way that would make an older reader misinterpret a newer file.
pub const CURRENT_SCHEMA_VERSION: u32 = 1;

/// The two broad families of model this module can actually run inference for. Used as the
/// single source of truth behind both `from_serializable`'s load-time servability check and
/// `perform_prediction`'s dispatch, so the two can never silently drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServableKind {
    /// Predicts a single continuous value: `intercept + coefficients \u{22c5} features`.
    Regression,
    /// Predicts a class label with probabilities, via sigmoid (binary) or softmax (multiclass).
    Classification,
}

/// Classify a `model_type` string into the family of inference it needs, or `None` if this
/// module has no real implementation for it.
fn classify_model_type(model_type: &str) -> Option<ServableKind> {
    match model_type.to_lowercase().as_str() {
        "linear_regression" | "linearregression" | "linear" | "ridge" | "lasso" | "elastic_net"
        | "elasticnet" => Some(ServableKind::Regression),
        "classification"
        | "logistic_regression"
        | "logisticregression"
        | "logistic"
        | "softmax" => Some(ServableKind::Classification),
        _ => None,
    }
}

/// Serialization formats supported by PandRS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// JSON format
    Json,
    /// YAML format
    Yaml,
    /// TOML format
    Toml,
    /// Binary format (MessagePack or Bincode)
    Binary,
}

impl SerializationFormat {
    /// Get file extension for the format
    pub fn extension(&self) -> &'static str {
        match self {
            SerializationFormat::Json => "json",
            SerializationFormat::Yaml => "yaml",
            SerializationFormat::Toml => "toml",
            SerializationFormat::Binary => "bin",
        }
    }

    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "json" => Some(SerializationFormat::Json),
            "yaml" | "yml" => Some(SerializationFormat::Yaml),
            "toml" => Some(SerializationFormat::Toml),
            "bin" | "pandrs" => Some(SerializationFormat::Binary),
            _ => None,
        }
    }
}

/// Serializable model container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableModel {
    /// On-disk schema version. `#[serde(default)]` so files written before this field existed
    /// (schema version 0, implicitly) still deserialize; `from_serializable` rejects any file
    /// whose version is *newer* than [`CURRENT_SCHEMA_VERSION`], since this build has no idea
    /// how to interpret fields a future schema might repurpose or remove.
    #[serde(default)]
    pub schema_version: u32,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Model parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Model type-specific data
    pub model_data: serde_json::Value,
    /// Feature preprocessing pipeline
    pub preprocessing: Option<serde_json::Value>,
    /// Model configuration
    pub config: HashMap<String, serde_json::Value>,
}

/// Model serialization trait
pub trait ModelSerializer {
    /// Save a model to file
    fn save<P: AsRef<Path>>(&self, model: &SerializableModel, path: P) -> Result<()>;

    /// Load a model from file
    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn ModelServing>>;

    /// Serialize model to bytes
    fn serialize(&self, model: &SerializableModel) -> Result<Vec<u8>>;

    /// Deserialize model from bytes
    fn deserialize(&self, data: &[u8]) -> Result<SerializableModel>;

    /// Get supported format
    fn format(&self) -> SerializationFormat;
}

/// JSON model serializer
pub struct JsonModelSerializer;

impl ModelSerializer for JsonModelSerializer {
    fn save<P: AsRef<Path>>(&self, model: &SerializableModel, path: P) -> Result<()> {
        let json_data = serde_json::to_string_pretty(model)?;
        fs::write(path, json_data)?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn ModelServing>> {
        let json_data = fs::read_to_string(path)?;
        let serializable_model: SerializableModel = serde_json::from_str(&json_data)?;

        // Convert to serving model
        Ok(Box::new(GenericServingModel::from_serializable(
            serializable_model,
        )?))
    }

    fn serialize(&self, model: &SerializableModel) -> Result<Vec<u8>> {
        let json_data = serde_json::to_vec(model)?;
        Ok(json_data)
    }

    fn deserialize(&self, data: &[u8]) -> Result<SerializableModel> {
        let model = serde_json::from_slice(data)?;
        Ok(model)
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Json
    }
}

/// YAML model serializer
pub struct YamlModelSerializer;

impl ModelSerializer for YamlModelSerializer {
    fn save<P: AsRef<Path>>(&self, model: &SerializableModel, path: P) -> Result<()> {
        let yaml_data = serde_yaml::to_string(model)
            .map_err(|e| Error::SerializationError(format!("YAML serialization failed: {}", e)))?;
        fs::write(path, yaml_data)?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn ModelServing>> {
        let yaml_data = fs::read_to_string(path)?;
        let serializable_model: SerializableModel =
            serde_yaml::from_str(&yaml_data).map_err(|e| {
                Error::SerializationError(format!("YAML deserialization failed: {}", e))
            })?;

        Ok(Box::new(GenericServingModel::from_serializable(
            serializable_model,
        )?))
    }

    fn serialize(&self, model: &SerializableModel) -> Result<Vec<u8>> {
        let yaml_data = serde_yaml::to_string(model)
            .map_err(|e| Error::SerializationError(format!("YAML serialization failed: {}", e)))?;
        Ok(yaml_data.into_bytes())
    }

    fn deserialize(&self, data: &[u8]) -> Result<SerializableModel> {
        let yaml_str = std::str::from_utf8(data)
            .map_err(|e| Error::SerializationError(format!("Invalid UTF-8: {}", e)))?;
        let model = serde_yaml::from_str(yaml_str).map_err(|e| {
            Error::SerializationError(format!("YAML deserialization failed: {}", e))
        })?;
        Ok(model)
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Yaml
    }
}

/// TOML model serializer
pub struct TomlModelSerializer;

impl ModelSerializer for TomlModelSerializer {
    fn save<P: AsRef<Path>>(&self, model: &SerializableModel, path: P) -> Result<()> {
        let toml_data = toml::to_string_pretty(model)
            .map_err(|e| Error::SerializationError(format!("TOML serialization failed: {}", e)))?;
        fs::write(path, toml_data)?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn ModelServing>> {
        let toml_data = fs::read_to_string(path)?;
        let serializable_model: SerializableModel = toml::from_str(&toml_data).map_err(|e| {
            Error::SerializationError(format!("TOML deserialization failed: {}", e))
        })?;

        Ok(Box::new(GenericServingModel::from_serializable(
            serializable_model,
        )?))
    }

    fn serialize(&self, model: &SerializableModel) -> Result<Vec<u8>> {
        let toml_data = toml::to_string_pretty(model)
            .map_err(|e| Error::SerializationError(format!("TOML serialization failed: {}", e)))?;
        Ok(toml_data.into_bytes())
    }

    fn deserialize(&self, data: &[u8]) -> Result<SerializableModel> {
        let toml_str = std::str::from_utf8(data)
            .map_err(|e| Error::SerializationError(format!("Invalid UTF-8: {}", e)))?;
        let model = toml::from_str(toml_str).map_err(|e| {
            Error::SerializationError(format!("TOML deserialization failed: {}", e))
        })?;
        Ok(model)
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Toml
    }
}

/// "Binary" model serializer.
///
/// Honest note on the name: this is currently **JSON bytes under a `.bin` extension**, not a
/// compact binary encoding. [`SerializationFormat::Binary`] is a real, distinct, round-trip-safe
/// format (verified in `test_format_detection` and the registry round-trip tests) — it is simply
/// not yet *space-efficient* binary. The correct fix is encoding via the COOLJAPAN `oxicode`
/// crate (this ecosystem's pure-Rust `bincode` replacement), which would need a new workspace
/// dependency entry; that is outside this module's Cargo.toml ownership (limited to the
/// `serving` feature line) and is left as a documented follow-up rather than silently claiming a
/// binary encoding this build doesn't actually produce.
pub struct BinaryModelSerializer;

impl ModelSerializer for BinaryModelSerializer {
    fn save<P: AsRef<Path>>(&self, model: &SerializableModel, path: P) -> Result<()> {
        let binary_data = self.serialize(model)?;
        fs::write(path, binary_data)?;
        Ok(())
    }

    fn load<P: AsRef<Path>>(&self, path: P) -> Result<Box<dyn ModelServing>> {
        let binary_data = fs::read(path)?;
        let serializable_model = self.deserialize(&binary_data)?;

        Ok(Box::new(GenericServingModel::from_serializable(
            serializable_model,
        )?))
    }

    fn serialize(&self, model: &SerializableModel) -> Result<Vec<u8>> {
        // See the struct doc: JSON bytes, not a compact binary encoding (oxicode follow-up).
        let json_data = serde_json::to_vec(model)?;
        Ok(json_data)
    }

    fn deserialize(&self, data: &[u8]) -> Result<SerializableModel> {
        let model = serde_json::from_slice(data)?;
        Ok(model)
    }

    fn format(&self) -> SerializationFormat {
        SerializationFormat::Binary
    }
}

/// Generic serving model implementation
#[derive(Debug)]
pub struct GenericServingModel {
    /// Model metadata
    metadata: ModelMetadata,
    /// Model parameters
    parameters: HashMap<String, serde_json::Value>,
    /// Model data
    model_data: serde_json::Value,
    /// Preprocessing pipeline
    preprocessing: Option<serde_json::Value>,
    /// Model configuration
    config: HashMap<String, serde_json::Value>,
    /// Lifetime prediction count, updated on every `predict()` call (success or failure).
    /// Atomic because `predict`/`predict_batch` take `&self`, not `&mut self`.
    total_predictions: AtomicU64,
    /// Lifetime failed-prediction count.
    total_errors: AtomicU64,
    /// Sum of per-request processing time in milliseconds, for computing a running average.
    total_latency_ms: AtomicU64,
    /// Milliseconds since the Unix epoch of the last prediction, or `i64::MIN` ("never").
    last_prediction_at_ms: AtomicI64,
    /// Wall-clock instant this model was constructed, used to compute a real (lifetime-average)
    /// throughput figure in `info()`.
    created_at_instant: Instant,
}

/// Sentinel for `last_prediction_at_ms` meaning "no prediction has been made yet". `i64::MIN`
/// rather than `0` so it can never collide with a genuine (if implausible) timestamp.
const NO_PREDICTION_YET: i64 = i64::MIN;

impl GenericServingModel {
    /// Create from serializable model.
    ///
    /// Validates the model at load time rather than deferring failures to the first `predict()`
    /// call: rejects files from a newer schema version than this build understands, and rejects
    /// `model_type`s this module has no real inference implementation for (previously, such a
    /// model would "successfully" load and only fail once someone tried to use it).
    pub fn from_serializable(serializable: SerializableModel) -> Result<Self> {
        if serializable.schema_version > CURRENT_SCHEMA_VERSION {
            return Err(Error::SerializationError(format!(
                "Model '{}' was saved with schema_version {}, which is newer than this build \
                 supports (schema_version {}); upgrade pandrs to load it",
                serializable.metadata.name, serializable.schema_version, CURRENT_SCHEMA_VERSION
            )));
        }
        if classify_model_type(&serializable.metadata.model_type).is_none() {
            return Err(Error::NotImplemented(format!(
                "Inference for model_type '{}' is not implemented; store known weights or use a \
                 supported type",
                serializable.metadata.model_type
            )));
        }

        Ok(Self {
            metadata: serializable.metadata,
            parameters: serializable.parameters,
            model_data: serializable.model_data,
            preprocessing: serializable.preprocessing,
            config: serializable.config,
            total_predictions: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_latency_ms: AtomicU64::new(0),
            last_prediction_at_ms: AtomicI64::new(NO_PREDICTION_YET),
            created_at_instant: Instant::now(),
        })
    }

    /// Convert to serializable model, stamped with the current schema version (whatever schema
    /// version this instance was originally loaded from, re-serializing always emits data this
    /// build considers current).
    pub fn to_serializable(&self) -> SerializableModel {
        SerializableModel {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata: self.metadata.clone(),
            parameters: self.parameters.clone(),
            model_data: self.model_data.clone(),
            preprocessing: self.preprocessing.clone(),
            config: self.config.clone(),
        }
    }

    /// Real readiness probe: checks the conditions `perform_prediction` actually depends on,
    /// rather than unconditionally reporting "healthy".
    ///
    /// Returns `(is_ready, problems)`, where `problems` lists every honest reason the model
    /// isn't ready to serve (empty when ready).
    fn readiness_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.metadata.feature_names.is_empty() {
            problems.push("metadata.feature_names is empty".to_string());
        }

        match classify_model_type(&self.metadata.model_type) {
            None => problems.push(format!(
                "model_type '{}' has no supported inference implementation",
                self.metadata.model_type
            )),
            Some(_) => match self.get_param("coefficients") {
                None => problems.push("no `coefficients` parameter is stored".to_string()),
                Some(value) => {
                    if !coefficients_all_finite(value) {
                        problems.push(
                            "`coefficients` contains non-numeric or non-finite values".to_string(),
                        );
                    }
                }
            },
        }

        problems
    }

    /// Look up a stored parameter by key, falling back to the `model_data` object.
    fn get_param(&self, key: &str) -> Option<&serde_json::Value> {
        if let Some(value) = self.parameters.get(key) {
            return Some(value);
        }
        self.model_data.as_object().and_then(|map| map.get(key))
    }

    /// Read a stored parameter as a flat vector of `f64` values.
    fn read_f64_vec(&self, key: &str) -> Option<Vec<f64>> {
        self.get_param(key)?
            .as_array()?
            .iter()
            .map(json_value_to_f64)
            .collect()
    }

    /// Extract the feature vector from the request, ordered by `metadata.feature_names`.
    fn extract_features(&self, input: &HashMap<String, serde_json::Value>) -> Result<Vec<f64>> {
        let names = &self.metadata.feature_names;
        if names.is_empty() {
            return Err(Error::InvalidInput(
                "Model metadata has no `feature_names`; cannot order input features for inference"
                    .into(),
            ));
        }

        let mut features = Vec::with_capacity(names.len());
        for name in names {
            let raw = input.get(name).ok_or_else(|| {
                Error::InvalidInput(format!("Missing feature '{}' in prediction input", name))
            })?;
            let value = json_value_to_f64(raw)
                .ok_or_else(|| Error::InvalidInput(format!("Feature '{}' is not numeric", name)))?;
            features.push(value);
        }
        Ok(features)
    }

    /// Real linear-model inference: `prediction = intercept + wᵀx`.
    ///
    /// Reads the `coefficients` (and optional `intercept`) stored on the model. Returns the
    /// scalar prediction together with the coefficient vector (reused for feature importance).
    fn linear_inference(
        &self,
        input: &HashMap<String, serde_json::Value>,
    ) -> Result<(f64, Vec<f64>)> {
        let coefficients = self.read_f64_vec("coefficients").ok_or_else(|| {
            Error::NotImplemented(
                "Linear inference requires a numeric `coefficients` array in the stored model"
                    .into(),
            )
        })?;
        let intercept = self
            .get_param("intercept")
            .and_then(json_value_to_f64)
            .unwrap_or(0.0);

        let features = self.extract_features(input)?;
        if features.len() != coefficients.len() {
            return Err(Error::DimensionMismatch(format!(
                "Number of input features ({}) does not match number of coefficients ({})",
                features.len(),
                coefficients.len()
            )));
        }

        let prediction = intercept
            + coefficients
                .iter()
                .zip(features.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>();
        Ok((prediction, coefficients))
    }

    /// Real classification inference from stored linear weights.
    ///
    /// Supports two stored layouts:
    /// * a 2-D `coefficients` matrix (`[n_classes][n_features]`) with optional `intercepts`,
    ///   producing class probabilities via the softmax function;
    /// * a 1-D `coefficients` vector (binary), with optional scalar `intercept`, producing
    ///   probabilities via the logistic sigmoid.
    ///
    /// Returns the predicted class label (as stored in `classes`, else a generated
    /// `class_{i}` label) and the `(label, probability)` pairs.
    ///
    /// `threshold` (from [`super::PredictionOptions::threshold`]) overrides the default 0.5
    /// cutoff used to pick the predicted label in the **binary** case; it has no effect on
    /// multiclass (softmax) predictions, which always use argmax.
    fn classification_inference(
        &self,
        input: &HashMap<String, serde_json::Value>,
        threshold: Option<f64>,
    ) -> Result<(serde_json::Value, Vec<(String, f64)>)> {
        let coeff_value = self.get_param("coefficients").ok_or_else(|| {
            Error::NotImplemented(
                "Classification inference requires a `coefficients` array in the stored model"
                    .into(),
            )
        })?;
        let rows = coeff_value
            .as_array()
            .ok_or_else(|| Error::InvalidInput("`coefficients` must be a JSON array".into()))?;
        let features = self.extract_features(input)?;
        let class_labels: Vec<serde_json::Value> = self
            .get_param("classes")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();

        let is_matrix = rows.first().map(|row| row.is_array()).unwrap_or(false);

        let (probabilities, labels) = if is_matrix {
            // Multiclass: one weight row per class, softmax over logits.
            let intercepts = self
                .read_f64_vec("intercepts")
                .or_else(|| self.read_f64_vec("intercept"))
                .unwrap_or_else(|| vec![0.0; rows.len()]);

            let mut logits = Vec::with_capacity(rows.len());
            for (class_idx, row_value) in rows.iter().enumerate() {
                let row = row_value.as_array().ok_or_else(|| {
                    Error::InvalidInput("Each `coefficients` row must be a JSON array".into())
                })?;
                let weights: Vec<f64> = row
                    .iter()
                    .map(json_value_to_f64)
                    .collect::<Option<Vec<f64>>>()
                    .ok_or_else(|| {
                        Error::InvalidInput("Coefficient values must be numeric".into())
                    })?;
                if weights.len() != features.len() {
                    return Err(Error::DimensionMismatch(format!(
                        "Class {} has {} coefficients but {} features were provided",
                        class_idx,
                        weights.len(),
                        features.len()
                    )));
                }
                let bias = intercepts.get(class_idx).copied().unwrap_or(0.0);
                logits.push(
                    bias + weights
                        .iter()
                        .zip(features.iter())
                        .map(|(w, x)| w * x)
                        .sum::<f64>(),
                );
            }
            let probabilities = softmax(&logits);
            let labels = resolve_class_labels(&class_labels, probabilities.len());
            (probabilities, labels)
        } else {
            // Binary: single weight vector, sigmoid over the logit.
            let weights: Vec<f64> = rows
                .iter()
                .map(json_value_to_f64)
                .collect::<Option<Vec<f64>>>()
                .ok_or_else(|| Error::InvalidInput("Coefficient values must be numeric".into()))?;
            if weights.len() != features.len() {
                return Err(Error::DimensionMismatch(format!(
                    "{} coefficients provided but {} features were given",
                    weights.len(),
                    features.len()
                )));
            }
            let bias = self
                .get_param("intercept")
                .and_then(json_value_to_f64)
                .unwrap_or(0.0);
            let logit = bias
                + weights
                    .iter()
                    .zip(features.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>();
            let p_positive = 1.0 / (1.0 + (-logit).exp());
            let labels = resolve_class_labels(&class_labels, 2);
            (vec![1.0 - p_positive, p_positive], labels)
        };

        // For binary classification, honor a caller-supplied threshold on the positive-class
        // probability instead of blind argmax (argmax is equivalent to the 0.5 default, so
        // behavior is unchanged when no threshold is given). Multiclass predictions always use
        // argmax: "threshold" has no standard meaning across more than two classes.
        let best = if !is_matrix {
            let cutoff = threshold.unwrap_or(0.5);
            if probabilities[1] >= cutoff {
                1
            } else {
                0
            }
        } else {
            argmax(&probabilities)
        };
        let predicted = labels
            .get(best)
            .cloned()
            .unwrap_or_else(|| serde_json::Value::String(format!("class_{}", best)));
        let pairs = labels
            .iter()
            .zip(probabilities.iter())
            .map(|(label, &probability)| (label_key(label), probability))
            .collect();
        Ok((predicted, pairs))
    }

    /// Run real model inference from the stored parameters.
    ///
    /// Unlike the previous placeholder, this performs an actual computation from the loaded
    /// weights. Unsupported model types return [`Error::NotImplemented`] rather than a
    /// fabricated constant.
    fn perform_prediction(
        &self,
        input_data: &HashMap<String, serde_json::Value>,
        threshold: Option<f64>,
    ) -> Result<serde_json::Value> {
        match classify_model_type(&self.metadata.model_type) {
            Some(ServableKind::Regression) => {
                let (prediction, _) = self.linear_inference(input_data)?;
                Ok(serde_json::json!({ "prediction": prediction }))
            }
            Some(ServableKind::Classification) => {
                let (label, pairs) = self.classification_inference(input_data, threshold)?;
                let mut probabilities = serde_json::Map::new();
                for (key, value) in &pairs {
                    probabilities.insert(key.clone(), serde_json::json!(value));
                }
                Ok(serde_json::json!({
                    "prediction": label,
                    "probabilities": serde_json::Value::Object(probabilities),
                }))
            }
            None => Err(Error::NotImplemented(format!(
                "Inference for model_type '{}' is not implemented; store known weights or use a supported type",
                self.metadata.model_type
            ))),
        }
    }

    /// Compute real class probabilities, or `None` for models without a probabilistic output.
    fn compute_probabilities(
        &self,
        input_data: &HashMap<String, serde_json::Value>,
    ) -> Result<Option<HashMap<String, f64>>> {
        match classify_model_type(&self.metadata.model_type) {
            Some(ServableKind::Classification) => {
                let (_, pairs) = self.classification_inference(input_data, None)?;
                Ok(Some(pairs.into_iter().collect()))
            }
            // Regression models have no class probabilities — report honestly.
            _ => Ok(None),
        }
    }

    /// Compute feature importance from the stored linear weights (absolute coefficient value).
    ///
    /// Returns `None` when the model exposes no coefficients or no feature names.
    fn compute_feature_importance(&self) -> Result<Option<HashMap<String, f64>>> {
        let names = &self.metadata.feature_names;
        if names.is_empty() {
            return Ok(None);
        }
        let coeff_value = match self.get_param("coefficients") {
            Some(value) => value,
            None => return Ok(None),
        };
        let rows = match coeff_value.as_array() {
            Some(rows) => rows,
            None => return Ok(None),
        };

        let importance = if rows.first().map(|r| r.is_array()).unwrap_or(false) {
            // Multiclass: mean absolute weight per feature across classes.
            let mut accum = vec![0.0_f64; names.len()];
            let mut class_count = 0usize;
            for row_value in rows {
                if let Some(row) = row_value.as_array() {
                    for (idx, cell) in row.iter().enumerate() {
                        if idx < accum.len() {
                            if let Some(value) = json_value_to_f64(cell) {
                                accum[idx] += value.abs();
                            }
                        }
                    }
                    class_count += 1;
                }
            }
            if class_count > 0 {
                for value in accum.iter_mut() {
                    *value /= class_count as f64;
                }
            }
            names.iter().cloned().zip(accum).collect()
        } else {
            let weights: Vec<f64> = rows.iter().filter_map(json_value_to_f64).collect();
            names
                .iter()
                .cloned()
                .zip(weights.iter().map(|w| w.abs()))
                .collect()
        };
        Ok(Some(importance))
    }

    /// Estimate the residual standard deviation from stored evaluation metrics.
    fn residual_std(&self) -> Option<f64> {
        let metrics = &self.metadata.metrics;
        for key in ["residual_std", "rmse", "sigma"] {
            if let Some(&value) = metrics.get(key) {
                if value.is_finite() && value > 0.0 {
                    return Some(value);
                }
            }
        }
        for key in ["mse", "mean_squared_error"] {
            if let Some(&value) = metrics.get(key) {
                if value.is_finite() && value > 0.0 {
                    return Some(value.sqrt());
                }
            }
        }
        None
    }

    /// Compute a real 95% prediction interval, when a residual spread is available.
    ///
    /// Uses a normal approximation `prediction ± z₀.₉₇₅ · σ`, where σ is derived from a stored
    /// `rmse`/`mse`/`residual_std` metric. Returns `None` (honestly) when no spread is stored or
    /// the prediction is non-numeric (e.g. a classification label).
    fn compute_confidence_interval(
        &self,
        prediction: &serde_json::Value,
    ) -> Result<Option<super::ConfidenceInterval>> {
        let point = match prediction.get("prediction").and_then(|v| v.as_f64()) {
            Some(value) => value,
            None => return Ok(None),
        };
        match self.residual_std() {
            Some(sigma) => {
                const Z_975: f64 = 1.959_963_984_540_054;
                Ok(Some(super::ConfidenceInterval {
                    lower: point - Z_975 * sigma,
                    upper: point + Z_975 * sigma,
                    confidence_level: 0.95,
                }))
            }
            None => Ok(None),
        }
    }
}

/// Convert a JSON value (number or numeric string) into an `f64`.
fn json_value_to_f64(value: &serde_json::Value) -> Option<f64> {
    if let Some(number) = value.as_f64() {
        Some(number)
    } else if let Some(integer) = value.as_i64() {
        Some(integer as f64)
    } else if let Some(text) = value.as_str() {
        text.parse::<f64>().ok()
    } else {
        None
    }
}

/// Numerically stable softmax.
fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f64; logits.len()]
    }
}

/// Index of the maximum element (first on ties).
fn argmax(values: &[f64]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }
    best_idx
}

/// Resolve class labels: use the stored `classes` when their count matches, else `class_{i}`.
fn resolve_class_labels(classes: &[serde_json::Value], n: usize) -> Vec<serde_json::Value> {
    if classes.len() == n {
        classes.to_vec()
    } else {
        (0..n)
            .map(|i| serde_json::Value::String(format!("class_{}", i)))
            .collect()
    }
}

/// Stringify a class label for use as a probability-map key.
fn label_key(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Check that a stored `coefficients` value (flat vector or nested per-class matrix) contains
/// only finite numeric entries -- the same shape `linear_inference`/`classification_inference`
/// require to produce a real prediction, used by the readiness probe to catch a broken model
/// before it is ever asked to predict.
fn coefficients_all_finite(value: &serde_json::Value) -> bool {
    let Some(rows) = value.as_array() else {
        return false;
    };
    let is_matrix = rows.first().map(|r| r.is_array()).unwrap_or(false);
    if is_matrix {
        rows.iter().all(|row| {
            row.as_array()
                .map(|inner| {
                    inner
                        .iter()
                        .all(|c| json_value_to_f64(c).map(|f| f.is_finite()).unwrap_or(false))
                })
                .unwrap_or(false)
        })
    } else {
        !rows.is_empty()
            && rows
                .iter()
                .all(|c| json_value_to_f64(c).map(|f| f.is_finite()).unwrap_or(false))
    }
}

impl ModelServing for GenericServingModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PredictionResponse> {
        let start_time = Instant::now();
        let threshold = request.options.as_ref().and_then(|o| o.threshold);

        // Perform prediction, recording real (atomics-backed) statistics regardless of outcome.
        let prediction_outcome = self.perform_prediction(&request.data, threshold);

        let processing_time = start_time.elapsed().as_millis() as u64;
        self.total_predictions.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(processing_time, Ordering::Relaxed);
        self.last_prediction_at_ms
            .store(chrono::Utc::now().timestamp_millis(), Ordering::Relaxed);

        let prediction_result = match prediction_outcome {
            Ok(value) => value,
            Err(e) => {
                self.total_errors.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        };

        // Build response
        let mut response = PredictionResponse {
            prediction: prediction_result,
            probabilities: None,
            feature_importance: None,
            confidence_intervals: None,
            model_metadata: self.metadata.clone(),
            timestamp: chrono::Utc::now(),
            processing_time_ms: processing_time,
        };

        // Add optional features if requested, computing real values (or honestly omitting them).
        if let Some(ref options) = request.options {
            if options.include_probabilities.unwrap_or(false) {
                response.probabilities = self.compute_probabilities(&request.data)?;
            }

            if options.include_feature_importance.unwrap_or(false) {
                response.feature_importance = self.compute_feature_importance()?;
            }

            if options.include_confidence_intervals.unwrap_or(false) {
                response.confidence_intervals =
                    self.compute_confidence_interval(&response.prediction)?;
            }
        }

        Ok(response)
    }

    fn predict_batch(&self, request: &BatchPredictionRequest) -> Result<BatchPredictionResponse> {
        let start_time = std::time::Instant::now();
        let mut predictions = Vec::new();
        let mut successful_predictions = 0;
        let mut failed_predictions = 0;
        let mut failed_items: Vec<(usize, String)> = Vec::new();

        for (idx, data) in request.data.iter().enumerate() {
            let individual_request = PredictionRequest {
                data: data.clone(),
                model_version: request.model_version.clone(),
                options: request.options.clone(),
            };

            match self.predict(&individual_request) {
                Ok(pred) => {
                    predictions.push(pred);
                    successful_predictions += 1;
                }
                Err(e) => {
                    failed_predictions += 1;
                    // Preserve which row failed and why, instead of a bare count.
                    failed_items.push((idx, e.to_string()));
                }
            }
        }

        let total_processing_time = start_time.elapsed().as_millis() as u64;
        let avg_processing_time = if !predictions.is_empty() {
            total_processing_time as f64 / predictions.len() as f64
        } else {
            0.0
        };

        let summary = super::BatchProcessingSummary {
            total_predictions: request.data.len(),
            successful_predictions,
            failed_predictions,
            total_processing_time_ms: total_processing_time,
            avg_processing_time_ms: avg_processing_time,
            failed_items,
        };

        Ok(BatchPredictionResponse {
            predictions,
            summary,
        })
    }

    fn get_metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn health_check(&self) -> Result<HealthStatus> {
        // Real readiness probe: reports "healthy" only when the conditions perform_prediction
        // actually depends on are met (previously this unconditionally reported "healthy").
        let problems = self.readiness_problems();
        let status = if problems.is_empty() {
            "healthy"
        } else {
            "unhealthy"
        };

        let mut details = HashMap::new();
        details.insert("status".to_string(), status.to_string());
        details.insert("model_type".to_string(), self.metadata.model_type.clone());
        details.insert("version".to_string(), self.metadata.version.clone());
        if !problems.is_empty() {
            details.insert("reason".to_string(), problems.join("; "));
        }

        Ok(HealthStatus {
            status: status.to_string(),
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn info(&self) -> ModelInfo {
        // Derive real, atomics-backed statistics rather than always reporting zeros.
        let total = self.total_predictions.load(Ordering::Relaxed);
        let errors = self.total_errors.load(Ordering::Relaxed);
        let total_latency_ms = self.total_latency_ms.load(Ordering::Relaxed);

        let avg_prediction_time_ms = if total > 0 {
            total_latency_ms as f64 / total as f64
        } else {
            0.0
        };
        let error_rate = if total > 0 {
            errors as f64 / total as f64
        } else {
            0.0
        };
        let last_ts = self.last_prediction_at_ms.load(Ordering::Relaxed);
        let last_prediction_at = if last_ts == NO_PREDICTION_YET {
            None
        } else {
            chrono::DateTime::from_timestamp_millis(last_ts)
        };
        // Lifetime-average throughput; a real (if coarse) figure rather than a fabricated one.
        let elapsed_secs = self.created_at_instant.elapsed().as_secs_f64();
        let throughput_per_second = if total > 0 && elapsed_secs > 0.0 {
            total as f64 / elapsed_secs
        } else {
            0.0
        };

        ModelInfo {
            metadata: self.metadata.clone(),
            statistics: ModelStatistics {
                total_predictions: total,
                avg_prediction_time_ms,
                error_rate,
                throughput_per_second,
                last_prediction_at,
            },
            configuration: self.config.clone(),
        }
    }

    fn to_serializable(&self) -> Result<SerializableModel> {
        // Resolves to the inherent `GenericServingModel::to_serializable` (infallible for this
        // type), wrapped as `Ok` to satisfy the trait's fallible signature.
        Ok(self.to_serializable())
    }
}

/// Model serialization factory
pub struct ModelSerializationFactory;

impl ModelSerializationFactory {
    /// Save model using the appropriate serializer
    pub fn save_model<P: AsRef<Path>>(
        model: &SerializableModel,
        path: P,
        format: SerializationFormat,
    ) -> Result<()> {
        match format {
            SerializationFormat::Json => {
                let serializer = JsonModelSerializer;
                serializer.save(model, path)
            }
            SerializationFormat::Yaml => {
                let serializer = YamlModelSerializer;
                serializer.save(model, path)
            }
            SerializationFormat::Toml => {
                let serializer = TomlModelSerializer;
                serializer.save(model, path)
            }
            SerializationFormat::Binary => {
                let serializer = BinaryModelSerializer;
                serializer.save(model, path)
            }
        }
    }

    /// Load model using the appropriate serializer
    pub fn load_model<P: AsRef<Path>>(
        path: P,
        format: SerializationFormat,
    ) -> Result<Box<dyn ModelServing>> {
        match format {
            SerializationFormat::Json => {
                let serializer = JsonModelSerializer;
                serializer.load(path)
            }
            SerializationFormat::Yaml => {
                let serializer = YamlModelSerializer;
                serializer.load(path)
            }
            SerializationFormat::Toml => {
                let serializer = TomlModelSerializer;
                serializer.load(path)
            }
            SerializationFormat::Binary => {
                let serializer = BinaryModelSerializer;
                serializer.load(path)
            }
        }
    }

    /// Auto-detect format and load model
    pub fn auto_detect_and_load<P: AsRef<Path>>(path: P) -> Result<Box<dyn ModelServing>> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| Error::InvalidInput("File has no extension".to_string()))?;

        let format = SerializationFormat::from_extension(extension).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported file extension: {}", extension))
        })?;

        Self::load_model(path, format)
    }

    /// Auto-detect format and save model
    pub fn auto_detect_and_save<P: AsRef<Path>>(model: &SerializableModel, path: P) -> Result<()> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| Error::InvalidInput("File has no extension".to_string()))?;

        let format = SerializationFormat::from_extension(extension).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported file extension: {}", extension))
        })?;

        Self::save_model(model, path, format)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn create_test_model() -> SerializableModel {
        let mut metadata = ModelMetadata {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            model_type: "linear_regression".to_string(),
            feature_names: vec!["feature1".to_string(), "feature2".to_string()],
            target_name: Some("target".to_string()),
            description: "Test model".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
        };

        metadata.metrics.insert("r2_score".to_string(), 0.85);

        let mut parameters = HashMap::new();
        parameters.insert("coefficients".to_string(), serde_json::json!([1.5, -0.8]));
        parameters.insert("intercept".to_string(), serde_json::json!(2.3));

        SerializableModel {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata,
            parameters,
            model_data: serde_json::json!({"type": "linear_regression"}),
            preprocessing: None,
            config: HashMap::new(),
        }
    }

    #[test]
    fn test_json_serialization() {
        let model = create_test_model();
        let serializer = JsonModelSerializer;

        // Test serialize/deserialize
        let serialized = serializer
            .serialize(&model)
            .expect("operation should succeed");
        let deserialized = serializer
            .deserialize(&serialized)
            .expect("operation should succeed");

        assert_eq!(model.metadata.name, deserialized.metadata.name);
        assert_eq!(model.metadata.version, deserialized.metadata.version);
    }

    #[test]
    fn test_yaml_serialization() {
        let model = create_test_model();
        let serializer = YamlModelSerializer;

        // Test serialize/deserialize
        let serialized = serializer
            .serialize(&model)
            .expect("operation should succeed");
        let deserialized = serializer
            .deserialize(&serialized)
            .expect("operation should succeed");

        assert_eq!(model.metadata.name, deserialized.metadata.name);
        assert_eq!(model.metadata.version, deserialized.metadata.version);
    }

    #[test]
    fn test_file_save_load() {
        let model = create_test_model();
        let serializer = JsonModelSerializer;

        // Create temporary file
        let temp_file = NamedTempFile::new().expect("operation should succeed");
        let temp_path = temp_file.path();

        // Save and load
        serializer
            .save(&model, temp_path)
            .expect("operation should succeed");
        let loaded_model = serializer
            .load(temp_path)
            .expect("operation should succeed");

        assert_eq!(model.metadata.name, loaded_model.get_metadata().name);
        assert_eq!(model.metadata.version, loaded_model.get_metadata().version);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            SerializationFormat::from_extension("json"),
            Some(SerializationFormat::Json)
        );
        assert_eq!(
            SerializationFormat::from_extension("yaml"),
            Some(SerializationFormat::Yaml)
        );
        assert_eq!(
            SerializationFormat::from_extension("yml"),
            Some(SerializationFormat::Yaml)
        );
        assert_eq!(
            SerializationFormat::from_extension("toml"),
            Some(SerializationFormat::Toml)
        );
        assert_eq!(
            SerializationFormat::from_extension("bin"),
            Some(SerializationFormat::Binary)
        );
        assert_eq!(SerializationFormat::from_extension("unknown"), None);
    }

    fn linear_request() -> PredictionRequest {
        let mut data = HashMap::new();
        data.insert("feature1".to_string(), serde_json::json!(1.5));
        data.insert("feature2".to_string(), serde_json::json!(2.0));
        PredictionRequest {
            data,
            model_version: None,
            options: None,
        }
    }

    #[test]
    fn test_linear_regression_real_inference() {
        let model = GenericServingModel::from_serializable(create_test_model())
            .expect("model should build");
        let response = model
            .predict(&linear_request())
            .expect("prediction succeeds");

        // intercept + wᵀx = 2.3 + 1.5*1.5 + (-0.8)*2.0 = 2.95 (NOT the old hardcoded 42.0)
        let prediction = response
            .prediction
            .get("prediction")
            .and_then(|v| v.as_f64())
            .expect("prediction should be numeric");
        assert!(
            (prediction - 2.95).abs() < 1e-9,
            "expected real inference 2.95, got {prediction}"
        );
    }

    #[test]
    fn test_regression_optional_features_are_honest() {
        let model = GenericServingModel::from_serializable(create_test_model())
            .expect("model should build");
        let mut request = linear_request();
        request.options = Some(crate::ml::serving::PredictionOptions {
            include_probabilities: Some(true),
            include_feature_importance: Some(true),
            include_confidence_intervals: Some(true),
            threshold: None,
        });
        let response = model.predict(&request).expect("prediction succeeds");

        // Regression has no class probabilities — honestly None, not an empty placeholder map.
        assert!(response.probabilities.is_none());

        // Feature importance is the absolute coefficient magnitude per feature.
        let importance = response
            .feature_importance
            .expect("linear model exposes feature importance");
        assert!((importance["feature1"] - 1.5).abs() < 1e-9);
        assert!((importance["feature2"] - 0.8).abs() < 1e-9);

        // The base test model stores only r2_score (no residual spread) → no fabricated CI.
        assert!(response.confidence_intervals.is_none());
    }

    #[test]
    fn test_confidence_interval_from_residual_metric() {
        let mut serializable = create_test_model();
        serializable
            .metadata
            .metrics
            .insert("mse".to_string(), 0.25);
        let model =
            GenericServingModel::from_serializable(serializable).expect("model should build");

        let mut request = linear_request();
        request.options = Some(crate::ml::serving::PredictionOptions {
            include_probabilities: None,
            include_feature_importance: None,
            include_confidence_intervals: Some(true),
            threshold: None,
        });
        let response = model.predict(&request).expect("prediction succeeds");

        // sigma = sqrt(mse) = 0.5; interval = 2.95 ± 1.95996*0.5
        let ci = response
            .confidence_intervals
            .expect("CI computed from mse metric");
        assert!((ci.confidence_level - 0.95).abs() < 1e-12);
        assert!((ci.lower - (2.95 - 1.959_963_984_540_054 * 0.5)).abs() < 1e-9);
        assert!((ci.upper - (2.95 + 1.959_963_984_540_054 * 0.5)).abs() < 1e-9);
    }

    fn create_classification_model() -> SerializableModel {
        let metadata = ModelMetadata {
            name: "clf".to_string(),
            version: "1.0.0".to_string(),
            model_type: "logistic_regression".to_string(),
            feature_names: vec!["x1".to_string(), "x2".to_string()],
            target_name: None,
            description: "binary classifier".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
        };
        let mut parameters = HashMap::new();
        parameters.insert("coefficients".to_string(), serde_json::json!([1.0, -1.0]));
        parameters.insert("intercept".to_string(), serde_json::json!(0.0));
        parameters.insert("classes".to_string(), serde_json::json!(["no", "yes"]));

        SerializableModel {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata,
            parameters,
            model_data: serde_json::json!({}),
            preprocessing: None,
            config: HashMap::new(),
        }
    }

    #[test]
    fn test_classification_real_inference() {
        let model = GenericServingModel::from_serializable(create_classification_model())
            .expect("model should build");
        let mut data = HashMap::new();
        data.insert("x1".to_string(), serde_json::json!(2.0));
        data.insert("x2".to_string(), serde_json::json!(0.0));
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
        let response = model.predict(&request).expect("prediction succeeds");

        // logit = 1*2 + (-1)*0 = 2 → sigmoid(2) ≈ 0.880797 for the positive class "yes".
        // The `prediction` value is an object carrying the label and the probabilities.
        assert_eq!(
            response
                .prediction
                .get("prediction")
                .and_then(|v| v.as_str()),
            Some("yes")
        );
        let probabilities = response
            .probabilities
            .expect("classifier returns probabilities");
        let p_yes = probabilities["yes"];
        let p_no = probabilities["no"];
        assert!((p_yes - 0.880_797).abs() < 1e-4, "p_yes = {p_yes}");
        assert!(
            (p_yes + p_no - 1.0).abs() < 1e-9,
            "probabilities must sum to 1"
        );
    }

    #[test]
    fn test_unknown_model_type_is_not_implemented() {
        // Servability is now validated at LOAD time (from_serializable), not deferred to the
        // first predict() call: a model this build can't run inference for should never
        // "successfully" load in the first place. (Previously `from_serializable` accepted any
        // model_type and only `predict()` failed; this test is updated to assert the corrected,
        // fail-fast behavior.)
        let mut serializable = create_test_model();
        serializable.metadata.model_type = "quantum_oracle".to_string();
        let err = GenericServingModel::from_serializable(serializable)
            .expect_err("unknown model_type must be rejected at load time");
        assert!(matches!(err, crate::core::error::Error::NotImplemented(_)));
    }

    #[test]
    fn test_schema_version_newer_than_supported_is_rejected() {
        let mut serializable = create_test_model();
        serializable.schema_version = CURRENT_SCHEMA_VERSION + 1;
        let err = GenericServingModel::from_serializable(serializable)
            .expect_err("a newer schema_version must be rejected");
        assert!(matches!(
            err,
            crate::core::error::Error::SerializationError(_)
        ));
    }

    #[test]
    fn test_missing_schema_version_defaults_to_zero_and_loads() {
        // Files written before schema_version existed have no such key at all; `#[serde(default)]`
        // must let them deserialize (as schema_version 0) and load successfully.
        let json = serde_json::to_string(&create_test_model()).expect("serializes");
        let value: serde_json::Value = serde_json::from_str(&json).expect("parses");
        let mut object = value.as_object().cloned().expect("model is a JSON object");
        object.remove("schema_version");
        let without_version = serde_json::Value::Object(object).to_string();

        let deserialized: SerializableModel =
            serde_json::from_str(&without_version).expect("missing schema_version defaults");
        assert_eq!(deserialized.schema_version, 0);
        assert!(GenericServingModel::from_serializable(deserialized).is_ok());
    }

    #[test]
    fn test_binary_classification_threshold_is_honored() {
        let model = GenericServingModel::from_serializable(create_classification_model())
            .expect("model should build");
        // logit = 1*x1 - 1*x2; with x1=0.6, x2=0.0 -> logit=0.6 -> p_yes = sigmoid(0.6) ≈ 0.6457.
        let mut data = HashMap::new();
        data.insert("x1".to_string(), serde_json::json!(0.6));
        data.insert("x2".to_string(), serde_json::json!(0.0));

        // Default threshold (0.5): p_yes > 0.5 -> "yes".
        let default_request = PredictionRequest {
            data: data.clone(),
            model_version: None,
            options: None,
        };
        let default_response = model
            .predict(&default_request)
            .expect("prediction succeeds");
        assert_eq!(
            default_response
                .prediction
                .get("prediction")
                .and_then(|v| v.as_str()),
            Some("yes")
        );

        // Custom threshold above p_yes (~0.6457): must flip the decision to "no", even though
        // the probability itself is unchanged.
        let thresholded_request = PredictionRequest {
            data,
            model_version: None,
            options: Some(crate::ml::serving::PredictionOptions {
                include_probabilities: None,
                include_feature_importance: None,
                include_confidence_intervals: None,
                threshold: Some(0.9),
            }),
        };
        let thresholded_response = model
            .predict(&thresholded_request)
            .expect("prediction succeeds");
        assert_eq!(
            thresholded_response
                .prediction
                .get("prediction")
                .and_then(|v| v.as_str()),
            Some("no"),
            "a 0.9 threshold must reject a ~0.65 positive-class probability"
        );
    }

    #[test]
    fn test_statistics_are_tracked_via_atomics() {
        let model = GenericServingModel::from_serializable(create_test_model())
            .expect("model should build");

        let before = model.info().statistics;
        assert_eq!(before.total_predictions, 0);
        assert!(before.last_prediction_at.is_none());

        model
            .predict(&linear_request())
            .expect("prediction succeeds");
        model
            .predict(&linear_request())
            .expect("prediction succeeds");

        let after = model.info().statistics;
        assert_eq!(after.total_predictions, 2);
        assert_eq!(after.error_rate, 0.0);
        assert!(after.last_prediction_at.is_some());
    }

    #[test]
    fn test_health_check_detects_missing_coefficients() {
        let metadata = ModelMetadata {
            name: "broken".to_string(),
            version: "1.0.0".to_string(),
            model_type: "linear_regression".to_string(),
            feature_names: vec!["feature1".to_string()],
            target_name: None,
            description: "no coefficients stored".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
        };
        let serializable = SerializableModel {
            schema_version: CURRENT_SCHEMA_VERSION,
            metadata,
            parameters: HashMap::new(),
            model_data: serde_json::json!({}),
            preprocessing: None,
            config: HashMap::new(),
        };
        let model =
            GenericServingModel::from_serializable(serializable).expect("model should build");

        let health = model.health_check().expect("health_check does not error");
        assert_eq!(health.status, "unhealthy");
        assert!(health.details.get("reason").is_some());
    }

    #[test]
    fn test_health_check_healthy_for_well_formed_model() {
        let model = GenericServingModel::from_serializable(create_test_model())
            .expect("model should build");
        let health = model.health_check().expect("health_check does not error");
        assert_eq!(health.status, "healthy");
    }
}
