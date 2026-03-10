//! Schema evolution and migration tools for DataFrames
//!
//! This module provides a comprehensive set of tools for defining, versioning,
//! and migrating DataFrame schemas over time. It is a core part of the PandRS
//! v0.3.0 feature set.
//!
//! # Overview
//!
//! Schema evolution allows you to:
//!
//! - **Define schemas** — specify the expected structure of a DataFrame including
//!   column names, types, nullability, constraints, and metadata.
//! - **Version schemas** — assign semantic versions to schemas so that you can
//!   track changes over time.
//! - **Create migrations** — define ordered sets of changes (add/remove/rename
//!   columns, change types, add constraints, etc.) that move data from one
//!   schema version to another.
//! - **Register schemas and migrations** — store them in a central registry and
//!   find migration paths automatically.
//! - **Apply migrations** — transform an actual `DataFrame` according to a
//!   migration, producing a new `DataFrame` that conforms to the target schema.
//! - **Validate data** — check that a `DataFrame` conforms to a schema, producing
//!   a detailed validation report.
//! - **Infer schemas** — automatically derive a schema from an existing `DataFrame`.
//! - **Check compatibility** — determine whether data can flow from one schema to
//!   another without data loss or type errors.
//! - **Serialize/deserialize** — save and load schemas and migrations as JSON or YAML.
//!
//! # Quick Start
//!
//! ```rust
//! use pandrs::schema_evolution::{
//!     DataFrameSchema, ColumnSchema, SchemaDataType, SchemaVersion,
//!     SchemaConstraint, Migration, SchemaChange,
//!     SchemaRegistry, SchemaMigrator,
//!     save_schema, load_schema, SchemaFormat,
//! };
//! use pandrs::{DataFrame, Series};
//!
//! // Define a schema
//! let v1 = DataFrameSchema::new("users", SchemaVersion::new(1, 0, 0))
//!     .with_column(ColumnSchema::new("id", SchemaDataType::Int64).with_nullable(false))
//!     .with_column(ColumnSchema::new("name", SchemaDataType::String))
//!     .with_constraint(SchemaConstraint::NotNull("id".to_string()));
//!
//! // Create a DataFrame
//! let mut df = DataFrame::new();
//! df.add_column("id".to_string(),
//!     Series::new(vec![1i64, 2, 3], Some("id".to_string())).expect("series"))
//!     .expect("add");
//! df.add_column("name".to_string(),
//!     Series::new(vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()], Some("name".to_string())).expect("series"))
//!     .expect("add");
//!
//! // Validate the DataFrame against the schema
//! let migrator = SchemaMigrator::empty();
//! let report = migrator.validate(&df, &v1).expect("validate");
//! assert!(report.is_valid);
//!
//! // Define a migration to v1.1
//! let migration = Migration::new(
//!     "m001",
//!     SchemaVersion::new(1, 0, 0),
//!     SchemaVersion::new(1, 1, 0),
//!     "Add email column",
//! )
//! .with_change(SchemaChange::AddColumn {
//!     schema: ColumnSchema::new("email", SchemaDataType::String),
//!     position: None,
//! });
//!
//! // Apply the migration
//! let migrated_df = migrator.apply_migration(&df, &migration).expect("migrate");
//! assert!(migrated_df.contains_column("email"));
//! ```

pub mod evolution;
pub mod migrator;
pub mod registry;
pub mod schema;
pub mod serialization;

// --- schema types ---
pub use schema::{
    ColumnSchema, DataFrameSchema, DefaultValue, SchemaConstraint, SchemaDataType, SchemaVersion,
};

// --- evolution types ---
pub use evolution::{Migration, MigrationBuilder, SchemaChange};

// --- registry ---
pub use registry::SchemaRegistry;

// --- migrator ---
pub use migrator::{
    BreakingChange, CompatibilityReport, SchemaMigrator, ValidationError, ValidationErrorType,
    ValidationReport,
};

// --- serialization helpers ---
pub use serialization::{
    load_bundle, load_migration, load_schema, migration_from_json, migration_from_yaml,
    migration_to_json, migration_to_yaml, save_bundle, save_migration, save_schema,
    schema_from_json, schema_from_yaml, schema_to_json, schema_to_yaml, SchemaBundle, SchemaFormat,
};
