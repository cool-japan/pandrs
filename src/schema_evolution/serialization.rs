//! Serialization and deserialization for schema evolution types
//!
//! Provides functions to save and load `DataFrameSchema` and `Migration`
//! to/from JSON and YAML formats.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::core::error::{Error, Result};

use super::evolution::Migration;
use super::schema::DataFrameSchema;

/// Supported serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaFormat {
    /// JSON format
    Json,
    /// YAML format
    Yaml,
}

impl SchemaFormat {
    /// Detect format from file extension
    pub fn from_path(path: &str) -> Option<SchemaFormat> {
        let lower = path.to_lowercase();
        if lower.ends_with(".json") {
            Some(SchemaFormat::Json)
        } else if lower.ends_with(".yaml") || lower.ends_with(".yml") {
            Some(SchemaFormat::Yaml)
        } else {
            None
        }
    }

    /// Return the conventional file extension (with leading dot)
    pub fn extension(&self) -> &str {
        match self {
            SchemaFormat::Json => ".json",
            SchemaFormat::Yaml => ".yaml",
        }
    }
}

/// Serialize a value to a string in the requested format
fn to_string_format<T: Serialize>(value: &T, format: SchemaFormat) -> Result<String> {
    match format {
        SchemaFormat::Json => serde_json::to_string_pretty(value)
            .map_err(|e| Error::SerializationError(e.to_string())),
        SchemaFormat::Yaml => {
            serde_yaml::to_string(value).map_err(|e| Error::SerializationError(e.to_string()))
        }
    }
}

/// Deserialize a value from a string, trying to detect the format automatically
fn from_str_format<T: for<'de> Deserialize<'de>>(content: &str, format: SchemaFormat) -> Result<T> {
    match format {
        SchemaFormat::Json => {
            serde_json::from_str(content).map_err(|e| Error::SerializationError(e.to_string()))
        }
        SchemaFormat::Yaml => {
            serde_yaml::from_str(content).map_err(|e| Error::SerializationError(e.to_string()))
        }
    }
}

/// Save a `DataFrameSchema` to a file
///
/// # Arguments
/// * `schema` - The schema to save
/// * `path`   - File path to write to
/// * `format` - The serialization format (Json or Yaml)
///
/// # Errors
/// Returns an error if serialization fails or the file cannot be written.
pub fn save_schema(schema: &DataFrameSchema, path: &str, format: SchemaFormat) -> Result<()> {
    let content = to_string_format(schema, format)?;
    fs::write(path, content).map_err(|e| Error::Io(e))?;
    Ok(())
}

/// Load a `DataFrameSchema` from a file.
///
/// The format is detected from the file extension (.json / .yaml / .yml).
/// If the extension is unknown, JSON is attempted first, then YAML.
///
/// # Arguments
/// * `path` - File path to read from
///
/// # Errors
/// Returns an error if the file cannot be read or deserialization fails.
pub fn load_schema(path: &str) -> Result<DataFrameSchema> {
    let content = fs::read_to_string(path).map_err(|e| Error::Io(e))?;
    let format = SchemaFormat::from_path(path).unwrap_or_else(|| {
        // Try to auto-detect from content
        if content.trim_start().starts_with('{') {
            SchemaFormat::Json
        } else {
            SchemaFormat::Yaml
        }
    });
    from_str_format(&content, format)
}

/// Save a `Migration` to a file
pub fn save_migration(migration: &Migration, path: &str, format: SchemaFormat) -> Result<()> {
    let content = to_string_format(migration, format)?;
    fs::write(path, content).map_err(|e| Error::Io(e))?;
    Ok(())
}

/// Load a `Migration` from a file
pub fn load_migration(path: &str) -> Result<Migration> {
    let content = fs::read_to_string(path).map_err(|e| Error::Io(e))?;
    let format = SchemaFormat::from_path(path).unwrap_or_else(|| {
        if content.trim_start().starts_with('{') {
            SchemaFormat::Json
        } else {
            SchemaFormat::Yaml
        }
    });
    from_str_format(&content, format)
}

/// Serialize a `DataFrameSchema` to a JSON string
pub fn schema_to_json(schema: &DataFrameSchema) -> Result<String> {
    to_string_format(schema, SchemaFormat::Json)
}

/// Deserialize a `DataFrameSchema` from a JSON string
pub fn schema_from_json(json: &str) -> Result<DataFrameSchema> {
    from_str_format(json, SchemaFormat::Json)
}

/// Serialize a `DataFrameSchema` to a YAML string
pub fn schema_to_yaml(schema: &DataFrameSchema) -> Result<String> {
    to_string_format(schema, SchemaFormat::Yaml)
}

/// Deserialize a `DataFrameSchema` from a YAML string
pub fn schema_from_yaml(yaml: &str) -> Result<DataFrameSchema> {
    from_str_format(yaml, SchemaFormat::Yaml)
}

/// Serialize a `Migration` to a JSON string
pub fn migration_to_json(migration: &Migration) -> Result<String> {
    to_string_format(migration, SchemaFormat::Json)
}

/// Deserialize a `Migration` from a JSON string
pub fn migration_from_json(json: &str) -> Result<Migration> {
    from_str_format(json, SchemaFormat::Json)
}

/// Serialize a `Migration` to a YAML string
pub fn migration_to_yaml(migration: &Migration) -> Result<String> {
    to_string_format(migration, SchemaFormat::Yaml)
}

/// Deserialize a `Migration` from a YAML string
pub fn migration_from_yaml(yaml: &str) -> Result<Migration> {
    from_str_format(yaml, SchemaFormat::Yaml)
}

/// A bundle of multiple schemas, useful for exporting a registry snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaBundle {
    /// All schemas in this bundle
    pub schemas: Vec<DataFrameSchema>,
    /// All migrations in this bundle
    pub migrations: Vec<Migration>,
    /// Bundle metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl SchemaBundle {
    /// Create a new empty bundle
    pub fn new() -> Self {
        SchemaBundle {
            schemas: Vec::new(),
            migrations: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a schema
    pub fn with_schema(mut self, schema: DataFrameSchema) -> Self {
        self.schemas.push(schema);
        self
    }

    /// Add a migration
    pub fn with_migration(mut self, migration: Migration) -> Self {
        self.migrations.push(migration);
        self
    }
}

impl Default for SchemaBundle {
    fn default() -> Self {
        Self::new()
    }
}

/// Save a schema bundle to a file
pub fn save_bundle(bundle: &SchemaBundle, path: &str, format: SchemaFormat) -> Result<()> {
    let content = to_string_format(bundle, format)?;
    fs::write(path, content).map_err(|e| Error::Io(e))?;
    Ok(())
}

/// Load a schema bundle from a file
pub fn load_bundle(path: &str) -> Result<SchemaBundle> {
    let content = fs::read_to_string(path).map_err(|e| Error::Io(e))?;
    let format = SchemaFormat::from_path(path).unwrap_or_else(|| {
        if content.trim_start().starts_with('{') {
            SchemaFormat::Json
        } else {
            SchemaFormat::Yaml
        }
    });
    from_str_format(&content, format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_evolution::schema::{
        ColumnSchema, DataFrameSchema, SchemaDataType, SchemaVersion,
    };

    fn make_schema() -> DataFrameSchema {
        DataFrameSchema::new("users", SchemaVersion::new(1, 0, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64).with_nullable(false))
            .with_column(ColumnSchema::new("name", SchemaDataType::String))
    }

    #[test]
    fn test_json_round_trip() {
        let schema = make_schema();
        let json = schema_to_json(&schema).expect("to json");
        let recovered: DataFrameSchema = schema_from_json(&json).expect("from json");
        assert_eq!(recovered.name, schema.name);
        assert_eq!(recovered.version, schema.version);
        assert_eq!(recovered.columns.len(), schema.columns.len());
    }

    #[test]
    fn test_yaml_round_trip() {
        let schema = make_schema();
        let yaml = schema_to_yaml(&schema).expect("to yaml");
        let recovered: DataFrameSchema = schema_from_yaml(&yaml).expect("from yaml");
        assert_eq!(recovered.name, schema.name);
        assert_eq!(recovered.version, schema.version);
        assert_eq!(recovered.columns.len(), schema.columns.len());
    }

    #[test]
    fn test_save_load_json_file() {
        let schema = make_schema();
        let dir = std::env::temp_dir();
        let path = dir.join("test_schema_serialization.json");
        let path_str = path.to_str().expect("path");

        save_schema(&schema, path_str, SchemaFormat::Json).expect("save");
        let recovered = load_schema(path_str).expect("load");
        assert_eq!(recovered.name, schema.name);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_save_load_yaml_file() {
        let schema = make_schema();
        let dir = std::env::temp_dir();
        let path = dir.join("test_schema_serialization.yaml");
        let path_str = path.to_str().expect("path");

        save_schema(&schema, path_str, SchemaFormat::Yaml).expect("save");
        let recovered = load_schema(path_str).expect("load");
        assert_eq!(recovered.name, schema.name);

        // Cleanup
        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_schema_format_detection() {
        assert_eq!(
            SchemaFormat::from_path("foo.json"),
            Some(SchemaFormat::Json)
        );
        assert_eq!(
            SchemaFormat::from_path("foo.yaml"),
            Some(SchemaFormat::Yaml)
        );
        assert_eq!(SchemaFormat::from_path("foo.yml"), Some(SchemaFormat::Yaml));
        assert_eq!(SchemaFormat::from_path("foo.csv"), None);
    }
}
