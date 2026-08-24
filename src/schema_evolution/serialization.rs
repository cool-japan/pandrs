//! Serialization and deserialization for schema evolution types
//!
//! Provides functions to save and load `DataFrameSchema` and `Migration`
//! to/from JSON and YAML formats.

use std::fs;

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

/// Deserialize a value from a string in a known format
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

/// Deserialize a value when the format is unknown: genuinely attempt JSON
/// first, then YAML, rather than guessing once from a content heuristic
/// (`{` at the start) and only trying that single guess. Returns an error
/// naming both parse failures if neither format accepts the content.
fn from_str_format_fallback<T: for<'de> Deserialize<'de>>(content: &str) -> Result<T> {
    match serde_json::from_str(content) {
        Ok(value) => Ok(value),
        Err(json_err) => match serde_yaml::from_str(content) {
            Ok(value) => Ok(value),
            Err(yaml_err) => Err(Error::SerializationError(format!(
                "content did not parse as JSON ({}) or YAML ({})",
                json_err, yaml_err
            ))),
        },
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
    let content = fs::read_to_string(path).map_err(Error::Io)?;
    match SchemaFormat::from_path(path) {
        Some(format) => from_str_format(&content, format),
        None => from_str_format_fallback(&content),
    }
}

/// Save a `Migration` to a file
pub fn save_migration(migration: &Migration, path: &str, format: SchemaFormat) -> Result<()> {
    let content = to_string_format(migration, format)?;
    fs::write(path, content).map_err(|e| Error::Io(e))?;
    Ok(())
}

/// Load a `Migration` from a file
pub fn load_migration(path: &str) -> Result<Migration> {
    let content = fs::read_to_string(path).map_err(Error::Io)?;
    match SchemaFormat::from_path(path) {
        Some(format) => from_str_format(&content, format),
        None => from_str_format_fallback(&content),
    }
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

impl SchemaBundle {
    /// Snapshot every schema and migration held by `registry` into a bundle.
    ///
    /// Each [`Migration`] already carries its own `schema_name` (see
    /// [`Migration::schema_name`]), so unlike the pre-`schema_name` design
    /// this captures the schema<->migration association, not just the raw
    /// lists -- [`Self::into_registry`] can rebuild a fully working registry
    /// from the result.
    pub fn from_registry(registry: &super::registry::SchemaRegistry) -> Self {
        let mut bundle = SchemaBundle::new();
        for name in registry.schema_names() {
            for version in registry.versions_of(name) {
                if let Some(schema) = registry.get_version(name, version) {
                    bundle = bundle.with_schema(schema.clone());
                }
            }
        }
        for migration in registry.all_migrations() {
            bundle = bundle.with_migration(migration.clone());
        }
        bundle
    }

    /// Rebuild a [`SchemaRegistry`](super::registry::SchemaRegistry) from
    /// this bundle: every schema is registered, and every migration is
    /// re-indexed via [`SchemaRegistry::add_migration`](super::registry::SchemaRegistry::add_migration)
    /// using the `schema_name` recorded on the migration itself. This is
    /// the operation that actually closes the save/load round trip -- a
    /// bundle with the association recorded but no way to restore a
    /// queryable registry from it would only be half a fix.
    ///
    /// Fails if any migration's `schema_name` is missing (see
    /// [`Migration::validate`]) or if two schemas/migrations collide,
    /// exactly as [`SchemaRegistry::register`](super::registry::SchemaRegistry::register)
    /// and `add_migration` would.
    pub fn into_registry(self) -> Result<super::registry::SchemaRegistry> {
        let mut registry = super::registry::SchemaRegistry::new();
        for schema in self.schemas {
            registry.register(schema)?;
        }
        for migration in self.migrations {
            registry.add_migration(migration)?;
        }
        Ok(registry)
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
    let content = fs::read_to_string(path).map_err(Error::Io)?;
    match SchemaFormat::from_path(path) {
        Some(format) => from_str_format(&content, format),
        None => from_str_format_fallback(&content),
    }
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
    fn test_load_schema_unknown_extension_falls_back_json_then_yaml() {
        let schema = make_schema();
        let dir = std::env::temp_dir();

        // A JSON body under an unrecognized extension must still load via
        // the genuine JSON-then-YAML fallback (not a single content guess).
        let json_path = dir.join("test_schema_fallback_json.dat");
        fs::write(&json_path, schema_to_json(&schema).expect("to json")).expect("write");
        let recovered = load_schema(json_path.to_str().expect("path")).expect("load json fallback");
        assert_eq!(recovered.name, schema.name);
        let _ = fs::remove_file(&json_path);

        // Likewise for a YAML body.
        let yaml_path = dir.join("test_schema_fallback_yaml.dat");
        fs::write(&yaml_path, schema_to_yaml(&schema).expect("to yaml")).expect("write");
        let recovered = load_schema(yaml_path.to_str().expect("path")).expect("load yaml fallback");
        assert_eq!(recovered.name, schema.name);
        let _ = fs::remove_file(&yaml_path);
    }

    #[test]
    fn test_bundle_round_trip_preserves_migration_schema_association() {
        use crate::schema_evolution::evolution::MigrationBuilder;
        use crate::schema_evolution::registry::SchemaRegistry;
        use crate::schema_evolution::schema::SchemaVersion;

        let v1 = DataFrameSchema::new("orders", SchemaVersion::new(1, 0, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64));
        let v2 = DataFrameSchema::new("orders", SchemaVersion::new(1, 1, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
            .with_column(ColumnSchema::new("total", SchemaDataType::Float64));

        let migration = MigrationBuilder::new(
            "orders_m1",
            "orders",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .add_column(ColumnSchema::new("total", SchemaDataType::Float64), None)
        .build();

        let mut registry = SchemaRegistry::new();
        registry.register(v1).expect("register v1");
        registry.register(v2).expect("register v2");
        registry.add_migration(migration).expect("add migration");

        let bundle = SchemaBundle::from_registry(&registry);

        let dir = std::env::temp_dir();
        let path = dir.join("test_bundle_round_trip_association.json");
        let path_str = path.to_str().expect("path");
        save_bundle(&bundle, path_str, SchemaFormat::Json).expect("save bundle");
        let loaded_bundle = load_bundle(path_str).expect("load bundle");
        let _ = fs::remove_file(&path);

        let rebuilt = loaded_bundle
            .into_registry()
            .expect("bundle round trip should preserve the schema<->migration association");

        let path_found = rebuilt.find_migration_path(
            "orders",
            &SchemaVersion::new(1, 0, 0),
            &SchemaVersion::new(1, 1, 0),
        );
        assert!(
            path_found.is_ok(),
            "migration path should survive a full bundle save/load round trip"
        );
        assert_eq!(path_found.expect("path").len(), 1);
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
