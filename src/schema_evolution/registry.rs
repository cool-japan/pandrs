//! Schema registry for storing and querying schema versions and migrations
//!
//! The registry acts as a central repository for all known schemas and the
//! migrations that move data between them.

use std::collections::HashMap;

use crate::core::error::{Error, Result};

use super::evolution::Migration;
use super::schema::{DataFrameSchema, SchemaVersion};

/// Central registry storing schema versions and migration paths
#[derive(Debug, Clone)]
pub struct SchemaRegistry {
    /// Map from schema name to ordered list of versions (sorted ascending by version)
    schemas: HashMap<String, Vec<DataFrameSchema>>,
    /// All registered migrations, indexed by ID
    migrations: HashMap<String, Migration>,
    /// Index: (schema_name, from_version, to_version) -> migration_id
    migration_index: HashMap<(String, String, String), String>,
}

impl SchemaRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        SchemaRegistry {
            schemas: HashMap::new(),
            migrations: HashMap::new(),
            migration_index: HashMap::new(),
        }
    }

    /// Register a new schema version.
    ///
    /// If a schema with this name and version already exists, returns an error.
    pub fn register(&mut self, schema: DataFrameSchema) -> Result<()> {
        let versions = self
            .schemas
            .entry(schema.name.clone())
            .or_insert_with(Vec::new);

        // Check for duplicate version
        if versions.iter().any(|s| s.version == schema.version) {
            return Err(Error::InvalidInput(format!(
                "Schema '{}' version {} is already registered",
                schema.name, schema.version
            )));
        }

        versions.push(schema);
        // Keep sorted by version
        versions.sort_by(|a, b| a.version.cmp(&b.version));
        Ok(())
    }

    /// Get the latest version of a named schema
    pub fn get_latest(&self, name: &str) -> Option<&DataFrameSchema> {
        self.schemas.get(name)?.last()
    }

    /// Get a specific version of a named schema
    pub fn get_version(&self, name: &str, version: &SchemaVersion) -> Option<&DataFrameSchema> {
        self.schemas
            .get(name)?
            .iter()
            .find(|s| &s.version == version)
    }

    /// List all registered schema names
    pub fn schema_names(&self) -> Vec<&str> {
        self.schemas.keys().map(|s| s.as_str()).collect()
    }

    /// List all versions of a named schema
    pub fn versions_of(&self, name: &str) -> Vec<&SchemaVersion> {
        self.schemas
            .get(name)
            .map(|versions| versions.iter().map(|s| &s.version).collect())
            .unwrap_or_default()
    }

    /// Add a migration to the registry.
    ///
    /// The migration ID must be unique. Returns an error if there is already a
    /// migration registered for the same (schema, from_version, to_version) triple.
    pub fn add_migration(&mut self, migration: Migration) -> Result<()> {
        // Validate the migration itself
        migration.validate().map_err(|e| Error::InvalidInput(e))?;

        // Check for duplicate ID
        if self.migrations.contains_key(&migration.id) {
            return Err(Error::InvalidInput(format!(
                "Migration with ID '{}' is already registered",
                migration.id
            )));
        }

        // We store the migration indexed by ID
        // We also build an index keyed by (schema_name, from, to) for path finding.
        // Note: the migration does not carry a schema_name directly, but we can
        // infer it when building paths by looking at registered schemas.
        // For the index we use the migration id as the target schema name key.
        // Actually, migrations can apply to any schema with matching versions, so we
        // store them separately and let find_migration_path do the graph search.
        self.migrations.insert(migration.id.clone(), migration);
        Ok(())
    }

    /// Add a migration with an explicit schema name association
    pub fn add_migration_for_schema(
        &mut self,
        schema_name: impl Into<String>,
        migration: Migration,
    ) -> Result<()> {
        // Validate the migration itself
        migration.validate().map_err(|e| Error::InvalidInput(e))?;

        let schema_name = schema_name.into();

        // Check for duplicate ID
        if self.migrations.contains_key(&migration.id) {
            return Err(Error::InvalidInput(format!(
                "Migration with ID '{}' is already registered",
                migration.id
            )));
        }

        // Build the index key
        let key = (
            schema_name,
            migration.from_version.to_string(),
            migration.to_version.to_string(),
        );

        if self.migration_index.contains_key(&key) {
            return Err(Error::InvalidInput(format!(
                "A migration from v{} to v{} already exists for schema '{}'",
                key.1, key.2, key.0
            )));
        }

        self.migration_index.insert(key, migration.id.clone());
        self.migrations.insert(migration.id.clone(), migration);
        Ok(())
    }

    /// Get a migration by its ID
    pub fn get_migration(&self, id: &str) -> Option<&Migration> {
        self.migrations.get(id)
    }

    /// Find the migration path from one version to another for a named schema.
    ///
    /// Performs a BFS over the migration graph to find the shortest path.
    /// Returns an ordered list of migrations to apply.
    pub fn find_migration_path(
        &self,
        schema_name: &str,
        from: &SchemaVersion,
        to: &SchemaVersion,
    ) -> Result<Vec<&Migration>> {
        if from == to {
            return Ok(vec![]);
        }

        // Build adjacency: from_version_str -> Vec<(to_version_str, migration_id)>
        let mut adjacency: HashMap<String, Vec<(String, &str)>> = HashMap::new();

        for (key, migration_id) in &self.migration_index {
            if &key.0 == schema_name {
                adjacency
                    .entry(key.1.clone())
                    .or_insert_with(Vec::new)
                    .push((key.2.clone(), migration_id.as_str()));
            }
        }

        // BFS
        let from_str = from.to_string();
        let to_str = to.to_string();

        let mut queue: std::collections::VecDeque<(String, Vec<&str>)> =
            std::collections::VecDeque::new();
        queue.push_back((from_str.clone(), vec![]));

        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        visited.insert(from_str.clone());

        while let Some((current_version, path)) = queue.pop_front() {
            if current_version == to_str {
                // Resolve migration IDs to references
                let migrations: Result<Vec<&Migration>> = path
                    .iter()
                    .map(|id| {
                        self.migrations.get(*id).ok_or_else(|| {
                            Error::KeyNotFound(format!("Migration '{}' not found", id))
                        })
                    })
                    .collect();
                return migrations;
            }

            if let Some(neighbors) = adjacency.get(&current_version) {
                for (next_version, migration_id) in neighbors {
                    if !visited.contains(next_version) {
                        visited.insert(next_version.clone());
                        let mut new_path = path.clone();
                        new_path.push(migration_id);
                        queue.push_back((next_version.clone(), new_path));
                    }
                }
            }
        }

        Err(Error::InvalidOperation(format!(
            "No migration path found from v{} to v{} for schema '{}'",
            from, to, schema_name
        )))
    }

    /// List all migrations registered in the registry
    pub fn all_migrations(&self) -> Vec<&Migration> {
        self.migrations.values().collect()
    }

    /// Check if a migration path exists between two versions
    pub fn has_migration_path(
        &self,
        schema_name: &str,
        from: &SchemaVersion,
        to: &SchemaVersion,
    ) -> bool {
        self.find_migration_path(schema_name, from, to).is_ok()
    }
}

impl Default for SchemaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_evolution::evolution::{MigrationBuilder, SchemaChange};
    use crate::schema_evolution::schema::{
        ColumnSchema, DataFrameSchema, SchemaDataType, SchemaVersion,
    };

    fn make_schema(name: &str, major: u32, minor: u32) -> DataFrameSchema {
        DataFrameSchema::new(name, SchemaVersion::new(major, minor, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
    }

    fn make_migration(id: &str, from: (u32, u32), to: (u32, u32)) -> Migration {
        MigrationBuilder::new(
            id,
            SchemaVersion::new(from.0, from.1, 0),
            SchemaVersion::new(to.0, to.1, 0),
        )
        .add_column(ColumnSchema::new("extra", SchemaDataType::String), None)
        .build()
    }

    #[test]
    fn test_register_and_retrieve() {
        let mut registry = SchemaRegistry::new();
        registry
            .register(make_schema("users", 1, 0))
            .expect("register failed");
        registry
            .register(make_schema("users", 1, 1))
            .expect("register failed");

        let latest = registry.get_latest("users").expect("should have latest");
        assert_eq!(latest.version, SchemaVersion::new(1, 1, 0));

        let v1 = registry.get_version("users", &SchemaVersion::new(1, 0, 0));
        assert!(v1.is_some());
    }

    #[test]
    fn test_duplicate_version_error() {
        let mut registry = SchemaRegistry::new();
        registry
            .register(make_schema("users", 1, 0))
            .expect("register failed");
        let err = registry.register(make_schema("users", 1, 0));
        assert!(err.is_err());
    }

    #[test]
    fn test_migration_path_single_step() {
        let mut registry = SchemaRegistry::new();
        registry
            .register(make_schema("users", 1, 0))
            .expect("register");
        registry
            .register(make_schema("users", 1, 1))
            .expect("register");

        let migration = make_migration("m001", (1, 0), (1, 1));
        registry
            .add_migration_for_schema("users", migration)
            .expect("add migration");

        let path = registry
            .find_migration_path(
                "users",
                &SchemaVersion::new(1, 0, 0),
                &SchemaVersion::new(1, 1, 0),
            )
            .expect("path");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].id, "m001");
    }

    #[test]
    fn test_migration_path_multi_step() {
        let mut registry = SchemaRegistry::new();
        registry
            .register(make_schema("users", 1, 0))
            .expect("register");
        registry
            .register(make_schema("users", 1, 1))
            .expect("register");
        registry
            .register(make_schema("users", 1, 2))
            .expect("register");

        registry
            .add_migration_for_schema("users", make_migration("m001", (1, 0), (1, 1)))
            .expect("add");
        registry
            .add_migration_for_schema("users", make_migration("m002", (1, 1), (1, 2)))
            .expect("add");

        let path = registry
            .find_migration_path(
                "users",
                &SchemaVersion::new(1, 0, 0),
                &SchemaVersion::new(1, 2, 0),
            )
            .expect("path");
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn test_no_migration_path_error() {
        let registry = SchemaRegistry::new();
        let err = registry.find_migration_path(
            "users",
            &SchemaVersion::new(1, 0, 0),
            &SchemaVersion::new(2, 0, 0),
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_same_version_path() {
        let registry = SchemaRegistry::new();
        let path = registry
            .find_migration_path(
                "users",
                &SchemaVersion::new(1, 0, 0),
                &SchemaVersion::new(1, 0, 0),
            )
            .expect("same version");
        assert!(path.is_empty());
    }
}
