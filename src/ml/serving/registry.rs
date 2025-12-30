//! Model Registry Module
//!
//! This module provides model registry functionality for managing multiple models,
//! versions, and metadata in a centralized manner.

use crate::core::error::{Error, Result};
use crate::ml::serving::serialization::{
    BinaryModelSerializer, JsonModelSerializer, ModelSerializationFactory, SerializableModel,
    TomlModelSerializer, YamlModelSerializer,
};
use crate::ml::serving::{ModelMetadata, ModelSerializer, ModelServing, SerializationFormat};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Model registry trait for managing models
pub trait ModelRegistry {
    /// Register a new model
    fn register_model(&mut self, model: Box<dyn ModelServing>) -> Result<()>;

    /// Load a model by name and version
    fn load_model(&self, name: &str, version: &str) -> Result<Box<dyn ModelServing>>;

    /// List all available models
    fn list_models(&self) -> Result<Vec<ModelRegistryEntry>>;

    /// List all versions of a specific model
    fn list_versions(&self, name: &str) -> Result<Vec<String>>;

    /// Get model metadata
    fn get_metadata(&self, name: &str, version: &str) -> Result<ModelMetadata>;

    /// Delete a model version
    fn delete_model(&mut self, name: &str, version: &str) -> Result<()>;

    /// Update model metadata
    fn update_metadata(&mut self, name: &str, version: &str, metadata: ModelMetadata)
        -> Result<()>;

    /// Check if model exists
    fn exists(&self, name: &str, version: &str) -> bool;

    /// Get latest version of a model
    fn get_latest_version(&self, name: &str) -> Result<String>;

    /// Set model as default version
    fn set_default_version(&mut self, name: &str, version: &str) -> Result<()>;

    /// Get default version of a model
    fn get_default_version(&self, name: &str) -> Result<String>;
}

/// Model registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    /// Model name
    pub name: String,
    /// Available versions
    pub versions: Vec<String>,
    /// Default version
    pub default_version: Option<String>,
    /// Latest version
    pub latest_version: Option<String>,
    /// Model description
    pub description: String,
    /// Model tags
    pub tags: Vec<String>,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// In-memory model registry implementation
pub struct InMemoryModelRegistry {
    /// Stored models indexed by name and version
    models: HashMap<String, HashMap<String, Box<dyn ModelServing>>>,
    /// Model registry entries
    entries: HashMap<String, ModelRegistryEntry>,
    /// Default versions for each model
    default_versions: HashMap<String, String>,
}

impl InMemoryModelRegistry {
    /// Create a new in-memory registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            entries: HashMap::new(),
            default_versions: HashMap::new(),
        }
    }

    /// Get model key
    fn get_model_key(name: &str, version: &str) -> String {
        format!("{}:{}", name, version)
    }

    /// Update registry entry
    fn update_entry(&mut self, name: &str, version: &str, metadata: &ModelMetadata) {
        let entry = self
            .entries
            .entry(name.to_string())
            .or_insert_with(|| ModelRegistryEntry {
                name: name.to_string(),
                versions: Vec::new(),
                default_version: None,
                latest_version: None,
                description: metadata.description.clone(),
                tags: Vec::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            });

        if !entry.versions.contains(&version.to_string()) {
            entry.versions.push(version.to_string());
            entry.versions.sort();
        }

        // Update latest version (assuming semantic versioning)
        entry.latest_version = entry.versions.last().cloned();

        // Set as default if it's the first version
        if entry.default_version.is_none() {
            entry.default_version = Some(version.to_string());
            self.default_versions
                .insert(name.to_string(), version.to_string());
        }

        entry.updated_at = chrono::Utc::now();
    }
}

impl Default for InMemoryModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry for InMemoryModelRegistry {
    fn register_model(&mut self, model: Box<dyn ModelServing>) -> Result<()> {
        let metadata = model.get_metadata().clone(); // Clone metadata first
        let name = metadata.name.clone();
        let version = metadata.version.clone();

        // Check if model already exists
        if self.exists(&name, &version) {
            return Err(Error::InvalidOperation(format!(
                "Model '{}' version '{}' already exists",
                name, version
            )));
        }

        // Store model
        self.models
            .entry(name.clone())
            .or_insert_with(HashMap::new)
            .insert(version.clone(), model);

        // Update registry entry
        self.update_entry(&name, &version, &metadata);

        Ok(())
    }

    fn load_model(&self, name: &str, version: &str) -> Result<Box<dyn ModelServing>> {
        let resolved_version = if version == "latest" {
            self.get_latest_version(name)?
        } else if version == "default" {
            self.get_default_version(name)?
        } else {
            version.to_string()
        };

        self.models
            .get(name)
            .and_then(|versions| versions.get(&resolved_version))
            .ok_or_else(|| {
                Error::KeyNotFound(format!(
                    "Model '{}' version '{}' not found",
                    name, resolved_version
                ))
            })
            .map(|_| {
                // NOTE: This is a limitation of the in-memory registry
                // We cannot return a reference to the boxed trait object
                // In a real implementation, we would need to implement Clone for ModelServing
                // or use Arc<dyn ModelServing>
                return Err(Error::NotImplemented(
                    "Loading models from in-memory registry requires cloning support".to_string(),
                ));
            })?
    }

    fn list_models(&self) -> Result<Vec<ModelRegistryEntry>> {
        Ok(self.entries.values().cloned().collect())
    }

    fn list_versions(&self, name: &str) -> Result<Vec<String>> {
        self.entries
            .get(name)
            .map(|entry| entry.versions.clone())
            .ok_or_else(|| Error::KeyNotFound(format!("Model '{}' not found", name)))
    }

    fn get_metadata(&self, name: &str, version: &str) -> Result<ModelMetadata> {
        let resolved_version = if version == "latest" {
            self.get_latest_version(name)?
        } else if version == "default" {
            self.get_default_version(name)?
        } else {
            version.to_string()
        };

        self.models
            .get(name)
            .and_then(|versions| versions.get(&resolved_version))
            .map(|model| model.get_metadata().clone())
            .ok_or_else(|| {
                Error::KeyNotFound(format!(
                    "Model '{}' version '{}' not found",
                    name, resolved_version
                ))
            })
    }

    fn delete_model(&mut self, name: &str, version: &str) -> Result<()> {
        if let Some(versions) = self.models.get_mut(name) {
            if versions.remove(version).is_some() {
                // Update registry entry
                if let Some(entry) = self.entries.get_mut(name) {
                    entry.versions.retain(|v| v != version);

                    // Update latest version
                    entry.latest_version = entry.versions.last().cloned();

                    // Update default version if it was deleted
                    if entry.default_version.as_ref() == Some(&version.to_string()) {
                        entry.default_version = entry.versions.first().cloned();
                        if let Some(new_default) = &entry.default_version {
                            self.default_versions
                                .insert(name.to_string(), new_default.clone());
                        } else {
                            self.default_versions.remove(name);
                        }
                    }

                    // Remove entry if no versions left
                    if entry.versions.is_empty() {
                        self.entries.remove(name);
                        self.models.remove(name);
                        self.default_versions.remove(name);
                    }
                }

                Ok(())
            } else {
                Err(Error::KeyNotFound(format!(
                    "Model '{}' version '{}' not found",
                    name, version
                )))
            }
        } else {
            Err(Error::KeyNotFound(format!("Model '{}' not found", name)))
        }
    }

    fn update_metadata(
        &mut self,
        name: &str,
        version: &str,
        metadata: ModelMetadata,
    ) -> Result<()> {
        // For in-memory registry, we cannot update the metadata of existing models
        // since ModelServing trait doesn't provide a mutable metadata interface
        Err(Error::NotImplemented(
            "Updating metadata for in-memory models is not supported".to_string(),
        ))
    }

    fn exists(&self, name: &str, version: &str) -> bool {
        self.models
            .get(name)
            .map(|versions| versions.contains_key(version))
            .unwrap_or(false)
    }

    fn get_latest_version(&self, name: &str) -> Result<String> {
        self.entries
            .get(name)
            .and_then(|entry| entry.latest_version.clone())
            .ok_or_else(|| Error::KeyNotFound(format!("Model '{}' not found", name)))
    }

    fn set_default_version(&mut self, name: &str, version: &str) -> Result<()> {
        if !self.exists(name, version) {
            return Err(Error::KeyNotFound(format!(
                "Model '{}' version '{}' not found",
                name, version
            )));
        }

        self.default_versions
            .insert(name.to_string(), version.to_string());

        if let Some(entry) = self.entries.get_mut(name) {
            entry.default_version = Some(version.to_string());
            entry.updated_at = chrono::Utc::now();
        }

        Ok(())
    }

    fn get_default_version(&self, name: &str) -> Result<String> {
        self.default_versions
            .get(name)
            .cloned()
            .ok_or_else(|| Error::KeyNotFound(format!("Model '{}' not found", name)))
    }
}

/// File system model registry implementation
pub struct FileSystemModelRegistry {
    /// Base directory for storing models
    base_path: PathBuf,
    /// Registry metadata file
    registry_file: PathBuf,
    /// Registry entries cache
    entries: HashMap<String, ModelRegistryEntry>,
    /// Default serialization format
    default_format: SerializationFormat,
}

impl FileSystemModelRegistry {
    /// Create a new file system registry
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        let registry_file = base_path.join("registry.json");

        // Create base directory if it doesn't exist
        if !base_path.exists() {
            fs::create_dir_all(&base_path)?;
        }

        let mut registry = Self {
            base_path,
            registry_file,
            entries: HashMap::new(),
            default_format: SerializationFormat::Json,
        };

        // Load existing registry
        registry.load_registry()?;

        Ok(registry)
    }

    /// Set default serialization format
    pub fn set_default_format(&mut self, format: SerializationFormat) {
        self.default_format = format;
    }

    /// Get model directory path
    fn get_model_dir(&self, name: &str) -> PathBuf {
        self.base_path.join(name)
    }

    /// Get model file path
    fn get_model_file(&self, name: &str, version: &str) -> PathBuf {
        self.get_model_dir(name)
            .join(format!("{}.{}", version, self.default_format.extension()))
    }

    /// Load registry metadata from file
    fn load_registry(&mut self) -> Result<()> {
        if self.registry_file.exists() {
            let registry_data = fs::read_to_string(&self.registry_file)?;
            self.entries = serde_json::from_str(&registry_data)?;
        }
        Ok(())
    }

    /// Save registry metadata to file
    fn save_registry(&self) -> Result<()> {
        let registry_data = serde_json::to_string_pretty(&self.entries)?;
        fs::write(&self.registry_file, registry_data)?;
        Ok(())
    }

    /// Update registry entry
    fn update_entry(&mut self, name: &str, version: &str, metadata: &ModelMetadata) -> Result<()> {
        let entry = self
            .entries
            .entry(name.to_string())
            .or_insert_with(|| ModelRegistryEntry {
                name: name.to_string(),
                versions: Vec::new(),
                default_version: None,
                latest_version: None,
                description: metadata.description.clone(),
                tags: Vec::new(),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            });

        if !entry.versions.contains(&version.to_string()) {
            entry.versions.push(version.to_string());
            entry.versions.sort();
        }

        // Update latest version
        entry.latest_version = entry.versions.last().cloned();

        // Set as default if it's the first version
        if entry.default_version.is_none() {
            entry.default_version = Some(version.to_string());
        }

        entry.updated_at = chrono::Utc::now();

        self.save_registry()
    }

    /// Convert ModelServing to SerializableModel
    fn model_to_serializable(&self, model: &dyn ModelServing) -> Result<SerializableModel> {
        let metadata = model.get_metadata().clone();
        let info = model.info();

        Ok(SerializableModel {
            metadata,
            parameters: HashMap::new(), // Would need to extract from model
            model_data: serde_json::json!({}), // Would need to extract from model
            preprocessing: None,
            config: info.configuration,
        })
    }
}

impl ModelRegistry for FileSystemModelRegistry {
    fn register_model(&mut self, model: Box<dyn ModelServing>) -> Result<()> {
        let metadata = model.get_metadata();
        let name = &metadata.name;
        let version = &metadata.version;

        // Check if model already exists
        if self.exists(name, version) {
            return Err(Error::InvalidOperation(format!(
                "Model '{}' version '{}' already exists",
                name, version
            )));
        }

        // Create model directory
        let model_dir = self.get_model_dir(name);
        if !model_dir.exists() {
            fs::create_dir_all(&model_dir)?;
        }

        // Convert to serializable model
        let serializable_model = self.model_to_serializable(model.as_ref())?;

        // Save model to file
        let model_file = self.get_model_file(name, version);
        ModelSerializationFactory::save_model(
            &serializable_model,
            &model_file,
            self.default_format,
        )?;

        // Update registry entry
        self.update_entry(name, version, metadata)?;

        Ok(())
    }

    fn load_model(&self, name: &str, version: &str) -> Result<Box<dyn ModelServing>> {
        let resolved_version = if version == "latest" {
            self.get_latest_version(name)?
        } else if version == "default" {
            self.get_default_version(name)?
        } else {
            version.to_string()
        };

        let model_file = self.get_model_file(name, &resolved_version);

        if !model_file.exists() {
            return Err(Error::KeyNotFound(format!(
                "Model file not found: {:?}",
                model_file
            )));
        }

        ModelSerializationFactory::auto_detect_and_load(&model_file)
    }

    fn list_models(&self) -> Result<Vec<ModelRegistryEntry>> {
        Ok(self.entries.values().cloned().collect())
    }

    fn list_versions(&self, name: &str) -> Result<Vec<String>> {
        self.entries
            .get(name)
            .map(|entry| entry.versions.clone())
            .ok_or_else(|| Error::KeyNotFound(format!("Model '{}' not found", name)))
    }

    fn get_metadata(&self, name: &str, version: &str) -> Result<ModelMetadata> {
        let resolved_version = if version == "latest" {
            self.get_latest_version(name)?
        } else if version == "default" {
            self.get_default_version(name)?
        } else {
            version.to_string()
        };

        let model_file = self.get_model_file(name, &resolved_version);

        if !model_file.exists() {
            return Err(Error::KeyNotFound(format!(
                "Model file not found: {:?}",
                model_file
            )));
        }

        // For getting metadata, we need to read and deserialize the file
        let format = SerializationFormat::from_extension(
            model_file
                .extension()
                .and_then(|ext| ext.to_str())
                .ok_or_else(|| Error::InvalidInput("File has no extension".to_string()))?,
        )
        .ok_or_else(|| Error::InvalidInput("Unsupported file extension".to_string()))?;

        let serializable_model = match format {
            SerializationFormat::Json => {
                let serializer = JsonModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
            SerializationFormat::Yaml => {
                let serializer = YamlModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
            SerializationFormat::Toml => {
                let serializer = TomlModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
            SerializationFormat::Binary => {
                let serializer = BinaryModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
        };

        Ok(serializable_model.metadata)
    }

    fn delete_model(&mut self, name: &str, version: &str) -> Result<()> {
        let model_file = self.get_model_file(name, version);

        if !model_file.exists() {
            return Err(Error::KeyNotFound(format!(
                "Model '{}' version '{}' not found",
                name, version
            )));
        }

        // Delete model file
        fs::remove_file(&model_file)?;

        // Update registry entry
        if let Some(entry) = self.entries.get_mut(name) {
            entry.versions.retain(|v| v != version);

            // Update latest version
            entry.latest_version = entry.versions.last().cloned();

            // Update default version if it was deleted
            if entry.default_version.as_ref() == Some(&version.to_string()) {
                entry.default_version = entry.versions.first().cloned();
            }

            // Remove entry if no versions left
            if entry.versions.is_empty() {
                self.entries.remove(name);

                // Remove model directory if empty
                let model_dir = self.get_model_dir(name);
                if model_dir.exists() && model_dir.read_dir()?.next().is_none() {
                    fs::remove_dir(&model_dir)?;
                }
            }
        }

        self.save_registry()?;
        Ok(())
    }

    fn update_metadata(
        &mut self,
        name: &str,
        version: &str,
        new_metadata: ModelMetadata,
    ) -> Result<()> {
        let model_file = self.get_model_file(name, version);

        if !model_file.exists() {
            return Err(Error::KeyNotFound(format!(
                "Model '{}' version '{}' not found",
                name, version
            )));
        }

        // Load existing model
        let format = SerializationFormat::from_extension(
            model_file
                .extension()
                .and_then(|ext| ext.to_str())
                .ok_or_else(|| Error::InvalidInput("File has no extension".to_string()))?,
        )
        .ok_or_else(|| Error::InvalidInput("Unsupported file extension".to_string()))?;

        let mut serializable_model = match format {
            SerializationFormat::Json => {
                let serializer = JsonModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
            SerializationFormat::Yaml => {
                let serializer = YamlModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
            SerializationFormat::Toml => {
                let serializer = TomlModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
            SerializationFormat::Binary => {
                let serializer = BinaryModelSerializer;
                serializer.deserialize(&fs::read(&model_file)?)?
            }
        };

        // Update metadata
        serializable_model.metadata = new_metadata.clone();

        // Save updated model
        ModelSerializationFactory::save_model(&serializable_model, &model_file, format)?;

        // Update registry entry
        self.update_entry(name, version, &new_metadata)?;

        Ok(())
    }

    fn exists(&self, name: &str, version: &str) -> bool {
        self.get_model_file(name, version).exists()
    }

    fn get_latest_version(&self, name: &str) -> Result<String> {
        self.entries
            .get(name)
            .and_then(|entry| entry.latest_version.clone())
            .ok_or_else(|| Error::KeyNotFound(format!("Model '{}' not found", name)))
    }

    fn set_default_version(&mut self, name: &str, version: &str) -> Result<()> {
        if !self.exists(name, version) {
            return Err(Error::KeyNotFound(format!(
                "Model '{}' version '{}' not found",
                name, version
            )));
        }

        if let Some(entry) = self.entries.get_mut(name) {
            entry.default_version = Some(version.to_string());
            entry.updated_at = chrono::Utc::now();
        }

        self.save_registry()?;
        Ok(())
    }

    fn get_default_version(&self, name: &str) -> Result<String> {
        self.entries
            .get(name)
            .and_then(|entry| entry.default_version.clone())
            .ok_or_else(|| Error::KeyNotFound(format!("Model '{}' not found", name)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_in_memory_registry() {
        let registry = InMemoryModelRegistry::new();

        // Test that registry starts empty
        assert!(registry.list_models().unwrap().is_empty());

        // Test model existence
        assert!(!registry.exists("test_model", "1.0.0"));
    }

    #[test]
    fn test_filesystem_registry_creation() {
        let temp_dir = TempDir::new().unwrap();
        let registry = FileSystemModelRegistry::new(temp_dir.path()).unwrap();

        // Test that registry directory is created
        assert!(temp_dir.path().exists());
        assert!(registry.registry_file.exists() || registry.entries.is_empty());
    }

    #[test]
    fn test_model_registry_entry() {
        let entry = ModelRegistryEntry {
            name: "test_model".to_string(),
            versions: vec!["1.0.0".to_string(), "1.1.0".to_string()],
            default_version: Some("1.0.0".to_string()),
            latest_version: Some("1.1.0".to_string()),
            description: "Test model".to_string(),
            tags: vec!["test".to_string()],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        assert_eq!(entry.name, "test_model");
        assert_eq!(entry.versions.len(), 2);
        assert_eq!(entry.latest_version, Some("1.1.0".to_string()));
    }
}
