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
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// All file extensions a model version might be persisted under. Used to find a model file
/// regardless of which [`SerializationFormat`] was active when it was written, so that
/// changing a registry's `default_format` after some models are already registered doesn't
/// orphan them (see [`FileSystemModelRegistry::find_model_file`]).
const KNOWN_MODEL_EXTENSIONS: &[&str] = &["json", "yaml", "yml", "toml", "bin", "pandrs"];

/// Parse a version string's leading `major.minor.patch` numeric components.
///
/// Trailing pre-release/build metadata after the patch number (e.g. `-rc1`, `+build5`) is
/// ignored for comparison purposes; only the three leading numeric components are extracted.
/// Returns `None` when the string doesn't start with at least one numeric component.
fn parse_semver_prefix(version: &str) -> Option<(u64, u64, u64)> {
    let mut parts = version.splitn(3, '.');
    let major: u64 = parts.next()?.parse().ok()?;
    let minor: u64 = parts
        .next()
        .map(|s| {
            let numeric: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
            numeric.parse().unwrap_or(0)
        })
        .unwrap_or(0);
    let patch: u64 = parts
        .next()
        .map(|s| {
            let numeric: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
            numeric.parse().unwrap_or(0)
        })
        .unwrap_or(0);
    Some((major, minor, patch))
}

/// Compare two version strings for the purpose of finding a registry's "latest" version.
///
/// Versions that parse as `major.minor.patch` are compared numerically (so `"1.10.0"` sorts
/// after `"1.9.0"`, unlike a plain lexicographic string sort). When either version fails to
/// parse, falls back to a lexicographic comparison so registries using non-semver version
/// tags (e.g. content hashes) still get a deterministic, documented ordering.
pub(crate) fn compare_versions(a: &str, b: &str) -> Ordering {
    match (parse_semver_prefix(a), parse_semver_prefix(b)) {
        (Some(va), Some(vb)) => va.cmp(&vb),
        _ => a.cmp(b),
    }
}

/// Sort a list of version strings in-place using [`compare_versions`] (ascending; the last
/// element is the "latest").
fn sort_versions(versions: &mut [String]) {
    versions.sort_by(|a, b| compare_versions(a, b));
}

/// Write `contents` to `path` atomically: write to a sibling temp file, then rename over the
/// destination. On POSIX and Windows, rename within the same directory is atomic, so readers
/// never observe a partially-written registry file even if the process is interrupted
/// mid-write.
fn atomic_write(path: &Path, contents: &str) -> Result<()> {
    let dir = path.parent().ok_or_else(|| {
        Error::InvalidInput(format!(
            "Registry path '{}' has no parent directory",
            path.display()
        ))
    })?;
    let file_name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        Error::InvalidInput(format!(
            "Registry path '{}' has no file name",
            path.display()
        ))
    })?;
    let tmp_path = dir.join(format!(".{}.tmp", file_name));
    fs::write(&tmp_path, contents)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Model registry trait for managing models
///
/// Requires `Send + Sync` so a registry can be shared (typically via `Arc`) with a
/// [`crate::ml::serving::ModelServer`] that resolves unregistered model names through it.
pub trait ModelRegistry: Send + Sync {
    /// Register a new model
    fn register_model(&mut self, model: Box<dyn ModelServing>) -> Result<()>;

    /// Load a model by name and version
    fn load_model(&self, name: &str, version: &str) -> Result<Arc<dyn ModelServing>>;

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
///
/// Models are stored behind `Arc<dyn ModelServing>` so that `load_model` can hand
/// out clones of the shared pointer without requiring `ModelServing: Clone`.
pub struct InMemoryModelRegistry {
    /// Stored models indexed by name and version
    models: HashMap<String, HashMap<String, Arc<dyn ModelServing>>>,
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
            sort_versions(&mut entry.versions);
        }

        // Update latest version using numeric semver comparison, not lexicographic order.
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

        // Convert Box -> Arc so the stored value is cheaply shareable via load_model.
        let arc_model: Arc<dyn ModelServing> = Arc::from(model);

        self.models
            .entry(name.clone())
            .or_insert_with(HashMap::new)
            .insert(version.clone(), arc_model);

        // Update registry entry
        self.update_entry(&name, &version, &metadata);

        Ok(())
    }

    fn load_model(&self, name: &str, version: &str) -> Result<Arc<dyn ModelServing>> {
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
            .map(|arc_model| Arc::clone(arc_model))
            .ok_or_else(|| {
                Error::KeyNotFound(format!(
                    "Model '{}' version '{}' not found",
                    name, resolved_version
                ))
            })
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
        new_metadata: ModelMetadata,
    ) -> Result<()> {
        // Retrieve the existing Arc to read its current serializable representation.
        let existing_arc = self
            .models
            .get(name)
            .and_then(|versions| versions.get(version))
            .cloned()
            .ok_or_else(|| {
                Error::KeyNotFound(format!("Model '{}' version '{}' not found", name, version))
            })?;

        // Round-trip the existing model through its own `to_serializable()` so the real
        // parameters/model_data/preprocessing survive, then overlay only the metadata and
        // re-wrap as a fresh GenericServingModel. (Previously this rebuilt from a blank
        // SerializableModel with empty `parameters`, silently discarding the model's weights
        // on every metadata edit.)
        use crate::ml::serving::serialization::GenericServingModel;
        let mut serializable = existing_arc.to_serializable()?;
        serializable.metadata = new_metadata.clone();

        let rebuilt: Arc<dyn ModelServing> =
            Arc::new(GenericServingModel::from_serializable(serializable)?);

        // Replace the stored Arc in-place (same name + version key).
        if let Some(versions) = self.models.get_mut(name) {
            versions.insert(version.to_string(), rebuilt);
        }

        // Refresh the registry entry description / timestamps.
        self.update_entry(name, version, &new_metadata);

        Ok(())
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

    /// Get the model file path this registry would use to *write* a model version, using its
    /// currently-configured `default_format`.
    fn get_model_file(&self, name: &str, version: &str) -> PathBuf {
        self.get_model_dir(name)
            .join(format!("{}.{}", version, self.default_format.extension()))
    }

    /// Find the on-disk file for a model version, regardless of which [`SerializationFormat`]
    /// it was written under.
    ///
    /// Tries the current `default_format` first (the common case), then falls back to probing
    /// every known extension. This matters because `default_format` is a mutable,
    /// registry-wide setting ([`Self::set_default_format`]): without this fallback, changing it
    /// after some models were already persisted under the old format would make `exists`,
    /// `load_model`, etc. silently fail to find them (an "orphaned model" bug), even though the
    /// file is still sitting on disk.
    fn find_model_file(&self, name: &str, version: &str) -> Option<PathBuf> {
        let preferred = self.get_model_file(name, version);
        if preferred.exists() {
            return Some(preferred);
        }
        let dir = self.get_model_dir(name);
        for ext in KNOWN_MODEL_EXTENSIONS {
            let candidate = dir.join(format!("{}.{}", version, ext));
            if candidate.exists() {
                return Some(candidate);
            }
        }
        None
    }

    /// Load registry metadata from file
    fn load_registry(&mut self) -> Result<()> {
        if self.registry_file.exists() {
            let registry_data = fs::read_to_string(&self.registry_file)?;
            self.entries = serde_json::from_str(&registry_data)?;
        }
        Ok(())
    }

    /// Save registry metadata to file.
    ///
    /// Writes atomically (temp file + rename) so a crash or concurrent read mid-write can never
    /// observe a truncated/corrupt `registry.json`.
    fn save_registry(&self) -> Result<()> {
        let registry_data = serde_json::to_string_pretty(&self.entries)?;
        atomic_write(&self.registry_file, &registry_data)
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
            sort_versions(&mut entry.versions);
        }

        // Update latest version using numeric semver comparison, not lexicographic order.
        entry.latest_version = entry.versions.last().cloned();

        // Set as default if it's the first version
        if entry.default_version.is_none() {
            entry.default_version = Some(version.to_string());
        }

        entry.updated_at = chrono::Utc::now();

        self.save_registry()
    }

    /// Convert a `ModelServing` to its persistable `SerializableModel`, preserving the model's
    /// real parameters/weights via its own `to_serializable()` implementation. (Previously this
    /// synthesized a blank `SerializableModel` with empty `parameters`/`model_data`, so every
    /// model persisted through this registry silently lost its weights.)
    fn model_to_serializable(&self, model: &dyn ModelServing) -> Result<SerializableModel> {
        model.to_serializable()
    }

    /// Read and deserialize a `SerializableModel` from `model_file`, detecting the format from
    /// its extension (which may differ from `self.default_format` if the registry's default
    /// format was changed after this file was written).
    fn read_serializable_model(&self, model_file: &Path) -> Result<SerializableModel> {
        let format = SerializationFormat::from_extension(
            model_file
                .extension()
                .and_then(|ext| ext.to_str())
                .ok_or_else(|| Error::InvalidInput("File has no extension".to_string()))?,
        )
        .ok_or_else(|| Error::InvalidInput("Unsupported file extension".to_string()))?;

        Ok(match format {
            SerializationFormat::Json => JsonModelSerializer.deserialize(&fs::read(model_file)?)?,
            SerializationFormat::Yaml => YamlModelSerializer.deserialize(&fs::read(model_file)?)?,
            SerializationFormat::Toml => TomlModelSerializer.deserialize(&fs::read(model_file)?)?,
            SerializationFormat::Binary => {
                BinaryModelSerializer.deserialize(&fs::read(model_file)?)?
            }
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

    fn load_model(&self, name: &str, version: &str) -> Result<Arc<dyn ModelServing>> {
        let resolved_version = if version == "latest" {
            self.get_latest_version(name)?
        } else if version == "default" {
            self.get_default_version(name)?
        } else {
            version.to_string()
        };

        let model_file = self
            .find_model_file(name, &resolved_version)
            .ok_or_else(|| {
                Error::KeyNotFound(format!(
                    "Model file not found for '{}' version '{}' (searched extensions: {:?})",
                    name, resolved_version, KNOWN_MODEL_EXTENSIONS
                ))
            })?;

        // Deserialize from disk then wrap in Arc so the call site gets an owned,
        // cheaply-cloneable handle rather than an exclusive Box.
        let boxed = ModelSerializationFactory::auto_detect_and_load(&model_file)?;
        Ok(Arc::from(boxed))
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

        let model_file = self
            .find_model_file(name, &resolved_version)
            .ok_or_else(|| {
                Error::KeyNotFound(format!(
                    "Model file not found for '{}' version '{}'",
                    name, resolved_version
                ))
            })?;

        Ok(self.read_serializable_model(&model_file)?.metadata)
    }

    fn delete_model(&mut self, name: &str, version: &str) -> Result<()> {
        let model_file = self.find_model_file(name, version).ok_or_else(|| {
            Error::KeyNotFound(format!("Model '{}' version '{}' not found", name, version))
        })?;

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
        let model_file = self.find_model_file(name, version).ok_or_else(|| {
            Error::KeyNotFound(format!("Model '{}' version '{}' not found", name, version))
        })?;

        // Load the existing model in whatever format it was actually written (which may not
        // match `self.default_format` if that was changed after this file was saved), so the
        // real parameters/model_data survive the metadata edit.
        let format = SerializationFormat::from_extension(
            model_file
                .extension()
                .and_then(|ext| ext.to_str())
                .ok_or_else(|| Error::InvalidInput("File has no extension".to_string()))?,
        )
        .ok_or_else(|| Error::InvalidInput("Unsupported file extension".to_string()))?;

        let mut serializable_model = self.read_serializable_model(&model_file)?;

        // Overlay only the metadata; parameters/model_data/preprocessing/config are untouched.
        serializable_model.metadata = new_metadata.clone();

        // Save updated model, in its original format and at its original path.
        ModelSerializationFactory::save_model(&serializable_model, &model_file, format)?;

        // Update registry entry
        self.update_entry(name, version, &new_metadata)?;

        Ok(())
    }

    fn exists(&self, name: &str, version: &str) -> bool {
        self.find_model_file(name, version).is_some()
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

        // `exists` confirmed the file is present; the registry entry itself must also be
        // present for us to actually record the change. Previously, a missing entry here (e.g.
        // a corrupted/hand-edited registry.json with orphaned model files) made this function
        // silently no-op: it would fall through to `save_registry()` and return `Ok(())` without
        // ever setting a default version.
        let entry = self.entries.get_mut(name).ok_or_else(|| {
            Error::KeyNotFound(format!(
                "Model '{}' has files on disk but no registry entry; registry.json may be \
                 corrupt or out of sync with the model directory",
                name
            ))
        })?;
        entry.default_version = Some(version.to_string());
        entry.updated_at = chrono::Utc::now();

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
        assert!(registry
            .list_models()
            .expect("operation should succeed")
            .is_empty());

        // Test model existence
        assert!(!registry.exists("test_model", "1.0.0"));
    }

    #[test]
    fn test_filesystem_registry_creation() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let registry =
            FileSystemModelRegistry::new(temp_dir.path()).expect("operation should succeed");

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

    /// Verifies that a model registered with InMemoryModelRegistry can be subsequently
    /// loaded via load_model, and that the returned Arc<dyn ModelServing> carries the
    /// correct metadata fields.
    #[test]
    fn test_in_memory_registry_load_model() {
        use crate::ml::serving::serialization::{GenericServingModel, SerializableModel};
        use crate::ml::serving::{ModelMetadata, ModelServing};
        use std::collections::HashMap;

        // Build a minimal SerializableModel and wrap it in GenericServingModel.
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            model_type: "linear_regression".to_string(),
            feature_names: vec!["x1".to_string(), "x2".to_string()],
            target_name: Some("y".to_string()),
            description: "Unit-test model for load_model".to_string(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
        };

        let serializable = SerializableModel {
            schema_version: crate::ml::serving::serialization::CURRENT_SCHEMA_VERSION,
            metadata,
            parameters: HashMap::new(),
            model_data: serde_json::json!({}),
            preprocessing: None,
            config: HashMap::new(),
        };

        let generic_model = GenericServingModel::from_serializable(serializable)
            .expect("model creation must succeed");
        let boxed: Box<dyn ModelServing> = Box::new(generic_model);

        // Register and immediately load.
        let mut registry = InMemoryModelRegistry::new();
        registry
            .register_model(boxed)
            .expect("register_model must succeed");

        // load_model must succeed and return an Arc with the correct metadata.
        let loaded = registry
            .load_model("test_model", "1.0.0")
            .expect("load_model must succeed for a registered model");

        let returned_meta = loaded.get_metadata();
        assert_eq!(returned_meta.name, "test_model");
        assert_eq!(returned_meta.version, "1.0.0");
        assert_eq!(returned_meta.model_type, "linear_regression");
        assert_eq!(returned_meta.description, "Unit-test model for load_model");

        // Confirm the registry entry is consistent.
        assert!(registry.exists("test_model", "1.0.0"));
        let meta_from_registry = registry
            .get_metadata("test_model", "1.0.0")
            .expect("get_metadata must succeed");
        assert_eq!(meta_from_registry.name, "test_model");
        assert_eq!(meta_from_registry.version, "1.0.0");
    }
}
