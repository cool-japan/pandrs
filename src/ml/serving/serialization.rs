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

/// Binary model serializer (using JSON for simplicity, could use MessagePack or Bincode)
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
        // Using JSON for binary format (could be replaced with MessagePack or Bincode)
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
    /// Model statistics
    statistics: ModelStatistics,
}

impl GenericServingModel {
    /// Create from serializable model
    pub fn from_serializable(serializable: SerializableModel) -> Result<Self> {
        Ok(Self {
            metadata: serializable.metadata,
            parameters: serializable.parameters,
            model_data: serializable.model_data,
            preprocessing: serializable.preprocessing,
            config: serializable.config,
            statistics: ModelStatistics {
                total_predictions: 0,
                avg_prediction_time_ms: 0.0,
                error_rate: 0.0,
                throughput_per_second: 0.0,
                last_prediction_at: None,
            },
        })
    }

    /// Convert to serializable model
    pub fn to_serializable(&self) -> SerializableModel {
        SerializableModel {
            metadata: self.metadata.clone(),
            parameters: self.parameters.clone(),
            model_data: self.model_data.clone(),
            preprocessing: self.preprocessing.clone(),
            config: self.config.clone(),
        }
    }

    /// Perform prediction logic (placeholder implementation)
    fn perform_prediction(
        &self,
        _input_data: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value> {
        // This is a placeholder implementation
        // In a real implementation, this would:
        // 1. Apply preprocessing if available
        // 2. Convert input data to the format expected by the model
        // 3. Run model inference
        // 4. Post-process results

        match self.metadata.model_type.as_str() {
            "linear_regression" => {
                // Simulate linear regression prediction
                Ok(serde_json::json!({"prediction": 42.0}))
            }
            "classification" => {
                // Simulate classification prediction
                Ok(serde_json::json!({
                    "prediction": "class_a",
                    "probabilities": {
                        "class_a": 0.7,
                        "class_b": 0.3
                    }
                }))
            }
            _ => {
                // Generic prediction
                Ok(serde_json::json!({"prediction": "unknown"}))
            }
        }
    }
}

impl ModelServing for GenericServingModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PredictionResponse> {
        let start_time = std::time::Instant::now();

        // Perform prediction
        let prediction_result = self.perform_prediction(&request.data)?;

        let processing_time = start_time.elapsed().as_millis() as u64;

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

        // Add optional features if requested
        if let Some(ref options) = request.options {
            if options.include_probabilities.unwrap_or(false) {
                response.probabilities = Some(HashMap::new()); // Placeholder
            }

            if options.include_feature_importance.unwrap_or(false) {
                response.feature_importance = Some(HashMap::new()); // Placeholder
            }

            if options.include_confidence_intervals.unwrap_or(false) {
                response.confidence_intervals = Some(super::ConfidenceInterval {
                    lower: 0.0,
                    upper: 100.0,
                    confidence_level: 0.95,
                }); // Placeholder
            }
        }

        Ok(response)
    }

    fn predict_batch(&self, request: &BatchPredictionRequest) -> Result<BatchPredictionResponse> {
        let start_time = std::time::Instant::now();
        let mut predictions = Vec::new();
        let mut successful_predictions = 0;
        let mut failed_predictions = 0;

        for data in &request.data {
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
                Err(_) => {
                    failed_predictions += 1;
                    // In a real implementation, we might want to include error information
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
        let mut details = HashMap::new();
        details.insert("status".to_string(), "healthy".to_string());
        details.insert("model_type".to_string(), self.metadata.model_type.clone());
        details.insert("version".to_string(), self.metadata.version.clone());

        Ok(HealthStatus {
            status: "healthy".to_string(),
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn info(&self) -> ModelInfo {
        ModelInfo {
            metadata: self.metadata.clone(),
            statistics: self.statistics.clone(),
            configuration: self.config.clone(),
        }
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
        let serialized = serializer.serialize(&model).unwrap();
        let deserialized = serializer.deserialize(&serialized).unwrap();

        assert_eq!(model.metadata.name, deserialized.metadata.name);
        assert_eq!(model.metadata.version, deserialized.metadata.version);
    }

    #[test]
    fn test_yaml_serialization() {
        let model = create_test_model();
        let serializer = YamlModelSerializer;

        // Test serialize/deserialize
        let serialized = serializer.serialize(&model).unwrap();
        let deserialized = serializer.deserialize(&serialized).unwrap();

        assert_eq!(model.metadata.name, deserialized.metadata.name);
        assert_eq!(model.metadata.version, deserialized.metadata.version);
    }

    #[test]
    fn test_file_save_load() {
        let model = create_test_model();
        let serializer = JsonModelSerializer;

        // Create temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path();

        // Save and load
        serializer.save(&model, temp_path).unwrap();
        let loaded_model = serializer.load(temp_path).unwrap();

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
}
