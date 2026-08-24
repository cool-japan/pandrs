#![allow(clippy::result_large_err)]
//! Integration tests for the schema_evolution module

use pandrs::schema_evolution::{
    load_migration, load_schema, migration_to_json, migration_to_yaml, save_migration, save_schema,
    schema_from_json, schema_from_yaml, schema_to_json, schema_to_yaml,
};
use pandrs::schema_evolution::{
    ColumnSchema, DataFrameSchema, DefaultValue, Migration, MigrationBuilder, SchemaChange,
    SchemaConstraint, SchemaDataType, SchemaFormat, SchemaMigrator, SchemaRegistry, SchemaVersion,
    ValidationErrorType,
};
use pandrs::{DataFrame, Series};
use std::env;
use std::fs;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_df_users() -> DataFrame {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2, 3], Some("id".to_string())).expect("series"),
    )
    .expect("add id");
    df.add_column(
        "name".to_string(),
        Series::new(
            vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()],
            Some("name".to_string()),
        )
        .expect("series"),
    )
    .expect("add name");
    df
}

fn make_schema_v1() -> DataFrameSchema {
    DataFrameSchema::new("users", SchemaVersion::new(1, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64).with_nullable(false))
        .with_column(ColumnSchema::new("name", SchemaDataType::String))
        .with_constraint(SchemaConstraint::NotNull("id".to_string()))
}

fn make_schema_v2() -> DataFrameSchema {
    DataFrameSchema::new("users", SchemaVersion::new(1, 1, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64).with_nullable(false))
        .with_column(ColumnSchema::new("name", SchemaDataType::String))
        .with_column(
            ColumnSchema::new("email", SchemaDataType::String)
                .with_nullable(true)
                .with_default(DefaultValue::Str(String::new())),
        )
        .with_constraint(SchemaConstraint::NotNull("id".to_string()))
}

fn make_migration_v1_v2() -> Migration {
    MigrationBuilder::new(
        "m_users_001",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .description("Add email column")
    .author("test-suite")
    .add_column(
        ColumnSchema::new("email", SchemaDataType::String)
            .with_nullable(true)
            .with_default(DefaultValue::Str(String::new())),
        None,
    )
    .build()
}

// ── Schema definition tests ───────────────────────────────────────────────────

#[test]
fn test_schema_version_ordering() {
    let v1 = SchemaVersion::new(1, 0, 0);
    let v2 = SchemaVersion::new(1, 1, 0);
    let v3 = SchemaVersion::new(2, 0, 0);
    assert!(v1 < v2);
    assert!(v2 < v3);
    assert_eq!(v1.to_string(), "1.0.0");
    assert_eq!(v2.to_string(), "1.1.0");
}

#[test]
fn test_schema_version_next() {
    let v = SchemaVersion::new(1, 2, 3);
    assert_eq!(v.next_major(), SchemaVersion::new(2, 0, 0));
    assert_eq!(v.next_minor(), SchemaVersion::new(1, 3, 0));
    assert_eq!(v.next_patch(), SchemaVersion::new(1, 2, 4));
}

#[test]
fn test_schema_version_compatibility() {
    let v1 = SchemaVersion::new(1, 0, 0);
    let v2 = SchemaVersion::new(1, 5, 3);
    let v3 = SchemaVersion::new(2, 0, 0);
    assert!(v1.is_compatible_with(&v2));
    assert!(!v1.is_compatible_with(&v3));
}

#[test]
fn test_column_schema_builder() {
    let col = ColumnSchema::new("age", SchemaDataType::Int64)
        .with_nullable(false)
        .with_default(DefaultValue::Int(0))
        .with_description("User age in years")
        .with_tag("pii")
        .with_tag("numeric");

    assert_eq!(col.name, "age");
    assert!(!col.nullable);
    assert!(matches!(col.default_value, Some(DefaultValue::Int(0))));
    assert_eq!(col.description.as_deref(), Some("User age in years"));
    assert_eq!(col.tags, vec!["pii", "numeric"]);
}

#[test]
fn test_dataframe_schema_structure() {
    let schema = make_schema_v1();
    assert_eq!(schema.name, "users");
    assert_eq!(schema.version, SchemaVersion::new(1, 0, 0));
    assert_eq!(schema.columns.len(), 2);
    assert_eq!(schema.constraints.len(), 1);
    assert!(schema.has_column("id"));
    assert!(schema.has_column("name"));
    assert!(!schema.has_column("email"));

    let id_col = schema.get_column("id").expect("id column");
    assert_eq!(id_col.data_type, SchemaDataType::Int64);
    assert!(!id_col.nullable);
}

#[test]
fn test_schema_constraint_ids() {
    let c1 = SchemaConstraint::NotNull("col".to_string());
    let c2 = SchemaConstraint::Unique(vec!["a".to_string(), "b".to_string()]);
    let c3 = SchemaConstraint::Range {
        col: "age".to_string(),
        min: Some(0.0),
        max: Some(120.0),
    };
    assert_eq!(c1.generate_id(), "notnull_col");
    assert_eq!(c2.generate_id(), "unique_a_b");
    assert!(c3.generate_id().starts_with("range_age"));
}

// ── JSON serialization round-trip ─────────────────────────────────────────────

#[test]
fn test_schema_json_round_trip() {
    let schema = make_schema_v1();
    let json = schema_to_json(&schema).expect("to json");
    assert!(json.contains("users"));
    assert!(json.contains("Int64"));

    let recovered = schema_from_json(&json).expect("from json");
    assert_eq!(recovered.name, schema.name);
    assert_eq!(recovered.version, schema.version);
    assert_eq!(recovered.columns.len(), schema.columns.len());
    assert_eq!(recovered.constraints.len(), schema.constraints.len());
    // Column names preserved
    assert_eq!(recovered.column_names(), schema.column_names());
}

#[test]
fn test_schema_yaml_round_trip() {
    let schema = make_schema_v2();
    let yaml = schema_to_yaml(&schema).expect("to yaml");
    assert!(yaml.contains("users"));

    let recovered = schema_from_yaml(&yaml).expect("from yaml");
    assert_eq!(recovered.name, schema.name);
    assert_eq!(recovered.version, schema.version);
    assert_eq!(recovered.columns.len(), 3);
}

#[test]
fn test_migration_json_round_trip() {
    let migration = make_migration_v1_v2();
    let json = migration_to_json(&migration).expect("to json");
    assert!(json.contains("m_users_001"));

    let recovered: Migration =
        pandrs::schema_evolution::migration_from_json(&json).expect("from json");
    assert_eq!(recovered.id, migration.id);
    assert_eq!(recovered.from_version, migration.from_version);
    assert_eq!(recovered.to_version, migration.to_version);
    assert_eq!(recovered.changes.len(), migration.changes.len());
}

#[test]
fn test_migration_yaml_round_trip() {
    let migration = make_migration_v1_v2();
    let yaml = migration_to_yaml(&migration).expect("to yaml");

    let recovered: Migration =
        pandrs::schema_evolution::migration_from_yaml(&yaml).expect("from yaml");
    assert_eq!(recovered.id, migration.id);
    assert_eq!(recovered.changes.len(), migration.changes.len());
}

// ── File save/load ────────────────────────────────────────────────────────────

#[test]
fn test_save_load_schema_json() {
    let schema = make_schema_v1();
    let dir = env::temp_dir();
    let path = dir.join("pandrs_test_schema_json.json");
    let path_str = path.to_str().expect("path");

    save_schema(&schema, path_str, SchemaFormat::Json).expect("save");
    let loaded = load_schema(path_str).expect("load");
    assert_eq!(loaded.name, schema.name);
    assert_eq!(loaded.version, schema.version);
    assert_eq!(loaded.columns.len(), schema.columns.len());

    let _ = fs::remove_file(&path);
}

#[test]
fn test_save_load_schema_yaml() {
    let schema = make_schema_v2();
    let dir = env::temp_dir();
    let path = dir.join("pandrs_test_schema_yaml.yaml");
    let path_str = path.to_str().expect("path");

    save_schema(&schema, path_str, SchemaFormat::Yaml).expect("save");
    let loaded = load_schema(path_str).expect("load");
    assert_eq!(loaded.name, schema.name);
    assert_eq!(loaded.columns.len(), 3);

    let _ = fs::remove_file(&path);
}

#[test]
fn test_save_load_migration_json() {
    let migration = make_migration_v1_v2();
    let dir = env::temp_dir();
    let path = dir.join("pandrs_test_migration.json");
    let path_str = path.to_str().expect("path");

    save_migration(&migration, path_str, SchemaFormat::Json).expect("save");
    let loaded = load_migration(path_str).expect("load");
    assert_eq!(loaded.id, migration.id);
    assert_eq!(loaded.changes.len(), migration.changes.len());

    let _ = fs::remove_file(&path);
}

// ── Migration application ─────────────────────────────────────────────────────

#[test]
fn test_apply_add_column_migration() {
    let df = make_df_users();
    let migration = make_migration_v1_v2();
    let migrator = SchemaMigrator::empty();

    let result = migrator.apply_migration(&df, &migration).expect("migrate");
    assert!(result.contains_column("email"));
    assert!(result.contains_column("id"));
    assert!(result.contains_column("name"));
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_apply_rename_column_migration() {
    let df = make_df_users();
    let migration = MigrationBuilder::new(
        "m002",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .rename_column("name", "full_name")
    .build();

    let migrator = SchemaMigrator::empty();
    let result = migrator.apply_migration(&df, &migration).expect("migrate");
    assert!(result.contains_column("full_name"));
    assert!(!result.contains_column("name"));
    assert!(result.contains_column("id"));
}

#[test]
fn test_apply_remove_column_migration() {
    let df = make_df_users();
    let migration = MigrationBuilder::new(
        "m003",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .remove_column("name")
    .build();

    let migrator = SchemaMigrator::empty();
    let result = migrator.apply_migration(&df, &migration).expect("migrate");
    assert!(!result.contains_column("name"));
    assert!(result.contains_column("id"));
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_apply_change_type_migration() {
    let df = make_df_users();
    let migration = MigrationBuilder::new(
        "m004",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .change_type("id", SchemaDataType::String, None)
    .build();

    let migrator = SchemaMigrator::empty();
    let result = migrator.apply_migration(&df, &migration).expect("migrate");
    assert!(result.contains_column("id"));
    // After conversion to string, it should not be detected as numeric
    let id_values = result.get_column_string_values("id").expect("id values");
    assert_eq!(id_values.len(), 3);
}

#[test]
fn test_apply_multi_step_migration() {
    let df = make_df_users();
    let migration = MigrationBuilder::new(
        "m005",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(2, 0, 0),
    )
    .add_column(
        ColumnSchema::new("score", SchemaDataType::Float64).with_default(DefaultValue::Float(0.0)),
        None,
    )
    .rename_column("name", "full_name")
    .remove_column("id")
    .build();

    let migrator = SchemaMigrator::empty();
    let result = migrator.apply_migration(&df, &migration).expect("migrate");
    assert!(result.contains_column("score"));
    assert!(result.contains_column("full_name"));
    assert!(!result.contains_column("name"));
    assert!(!result.contains_column("id"));
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_metadata_only_changes_no_data_loss() {
    let df = make_df_users();
    let migration = MigrationBuilder::new(
        "m_meta",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 0, 1),
    )
    .build();
    // Manually add metadata-only changes
    let migration = Migration {
        changes: vec![
            SchemaChange::SetNullable {
                column: "name".to_string(),
                nullable: false,
            },
            SchemaChange::SetColumnDescription {
                column: "id".to_string(),
                description: "Primary key".to_string(),
            },
            SchemaChange::AddConstraint {
                constraint: SchemaConstraint::NotNull("name".to_string()),
            },
        ],
        ..migration
    };

    let migrator = SchemaMigrator::empty();
    let result = migrator.apply_migration(&df, &migration).expect("migrate");
    // Metadata-only changes don't touch the DataFrame data
    assert_eq!(result.column_names(), df.column_names());
    assert_eq!(result.row_count(), df.row_count());
}

// ── Validation ────────────────────────────────────────────────────────────────

#[test]
fn test_validation_valid_dataframe() {
    let df = make_df_users();
    let schema = make_schema_v1();
    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(report.is_valid, "errors: {:?}", report.errors);
    assert!(report.errors.is_empty());
}

#[test]
fn test_validation_missing_column() {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64], Some("id".to_string())).expect("series"),
    )
    .expect("add");
    // name is absent

    let schema = make_schema_v1();
    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(!report.is_valid);
    assert!(report
        .errors
        .iter()
        .any(|e| e.error_type == ValidationErrorType::MissingColumn && e.column == "name"));
}

#[test]
fn test_validation_extra_column_warning() {
    let mut df = make_df_users();
    df.add_column(
        "extra".to_string(),
        Series::new(
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
            Some("extra".to_string()),
        )
        .expect("series"),
    )
    .expect("add");

    let schema = make_schema_v1();
    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(report.is_valid); // extra column is a warning, not an error
    assert!(report.warnings.iter().any(|w| w.contains("extra")));
}

#[test]
fn test_validation_range_constraint_violation() {
    let mut df = DataFrame::new();
    df.add_column(
        "age".to_string(),
        Series::new(vec![25.0f64, 200.0, 30.0], Some("age".to_string())).expect("series"),
    )
    .expect("add");

    let schema = DataFrameSchema::new("test", SchemaVersion::initial())
        .with_column(ColumnSchema::new("age", SchemaDataType::Float64))
        .with_constraint(SchemaConstraint::Range {
            col: "age".to_string(),
            min: Some(0.0),
            max: Some(150.0),
        });

    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(!report.is_valid);
    assert!(report
        .errors
        .iter()
        .any(|e| e.error_type == ValidationErrorType::RangeViolation));
}

#[test]
fn test_validation_regex_constraint() {
    let mut df = DataFrame::new();
    df.add_column(
        "email".to_string(),
        Series::new(
            vec!["ok@test.com".to_string(), "not-an-email".to_string()],
            Some("email".to_string()),
        )
        .expect("series"),
    )
    .expect("add");

    let schema = DataFrameSchema::new("test", SchemaVersion::initial())
        .with_column(ColumnSchema::new("email", SchemaDataType::String))
        .with_constraint(SchemaConstraint::Regex {
            col: "email".to_string(),
            pattern: r"^[^@]+@[^@]+\.[^@]+$".to_string(),
        });

    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(!report.is_valid);
    assert!(report
        .errors
        .iter()
        .any(|e| e.error_type == ValidationErrorType::RegexViolation));
}

#[test]
fn test_validation_enum_constraint() {
    let mut df = DataFrame::new();
    df.add_column(
        "status".to_string(),
        Series::new(
            vec!["active".to_string(), "deleted".to_string()],
            Some("status".to_string()),
        )
        .expect("series"),
    )
    .expect("add");

    let schema = DataFrameSchema::new("test", SchemaVersion::initial())
        .with_column(ColumnSchema::new("status", SchemaDataType::String))
        .with_constraint(SchemaConstraint::Enum {
            col: "status".to_string(),
            values: vec!["active".to_string(), "pending".to_string()],
        });

    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(!report.is_valid);
    assert!(report
        .errors
        .iter()
        .any(|e| e.error_type == ValidationErrorType::EnumViolation));
}

#[test]
fn test_validation_unique_constraint() {
    let mut df = DataFrame::new();
    df.add_column(
        "code".to_string(),
        Series::new(
            vec!["A".to_string(), "B".to_string(), "A".to_string()],
            Some("code".to_string()),
        )
        .expect("series"),
    )
    .expect("add");

    let schema = DataFrameSchema::new("test", SchemaVersion::initial())
        .with_column(ColumnSchema::new("code", SchemaDataType::String))
        .with_constraint(SchemaConstraint::Unique(vec!["code".to_string()]));

    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(!report.is_valid);
    assert!(report
        .errors
        .iter()
        .any(|e| e.error_type == ValidationErrorType::UniqueViolation));
}

// ── Schema inference ──────────────────────────────────────────────────────────

#[test]
fn test_infer_schema_from_dataframe() {
    let df = make_df_users();
    let migrator = SchemaMigrator::empty();
    let schema = migrator.infer_schema(&df, "inferred_users");

    assert_eq!(schema.name, "inferred_users");
    assert_eq!(schema.version, SchemaVersion::initial());
    assert!(schema.has_column("id"));
    assert!(schema.has_column("name"));
    assert_eq!(schema.columns.len(), 2);

    // id is i64 values -> inferred as Int64
    let id_col = schema.get_column("id").expect("id");
    assert_eq!(id_col.data_type, SchemaDataType::Int64);

    // name is string -> inferred as String
    let name_col = schema.get_column("name").expect("name");
    assert_eq!(name_col.data_type, SchemaDataType::String);

    // Metadata should be populated
    assert!(schema.metadata.contains_key("row_count"));
}

#[test]
fn test_infer_schema_float_column() {
    let mut df = DataFrame::new();
    df.add_column(
        "score".to_string(),
        Series::new(vec![1.5f64, 2.7, 3.15], Some("score".to_string())).expect("series"),
    )
    .expect("add");

    let migrator = SchemaMigrator::empty();
    let schema = migrator.infer_schema(&df, "test");
    let col = schema.get_column("score").expect("score");
    assert_eq!(col.data_type, SchemaDataType::Float64);
}

// ── Compatibility checking ────────────────────────────────────────────────────

#[test]
fn test_compatibility_fully_compatible() {
    // Adding nullable columns is backward compatible
    let from = make_schema_v1();
    let to = make_schema_v2();

    let migrator = SchemaMigrator::empty();
    let report = migrator.check_compatibility(&from, &to);
    assert!(
        report.is_compatible,
        "breaking: {:?}",
        report.breaking_changes
    );
    assert!(report.breaking_changes.is_empty());
}

#[test]
fn test_compatibility_missing_required_column() {
    let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64));

    let to = DataFrameSchema::new("v2", SchemaVersion::new(2, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
        .with_column(ColumnSchema::new("mandatory", SchemaDataType::String).with_nullable(false));

    let migrator = SchemaMigrator::empty();
    let report = migrator.check_compatibility(&from, &to);
    assert!(!report.is_compatible);
    assert!(!report.breaking_changes.is_empty());
}

#[test]
fn test_compatibility_type_narrowing() {
    // Boolean -> Int64 is not castable (Boolean can only go to String or numeric 0/1)
    // Actually Boolean -> String is castable, but Boolean is not in the can_cast_to
    // for Int64 per our schema (only String->Int64, Float64->Int64, Int64->Int64).
    // Let's check Boolean (not castable to Categorical):
    let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
        .with_column(ColumnSchema::new("flag", SchemaDataType::Boolean));

    let to =
        DataFrameSchema::new("v2", SchemaVersion::new(2, 0, 0)).with_column(ColumnSchema::new(
            "flag",
            SchemaDataType::Categorical {
                categories: vec!["yes".to_string(), "no".to_string()],
            },
        ));

    let migrator = SchemaMigrator::empty();
    let report = migrator.check_compatibility(&from, &to);
    assert!(!report.is_compatible);
}

#[test]
fn test_compatibility_widening_type() {
    // Int64 -> Float64 is castable (non-breaking, just noted)
    let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
        .with_column(ColumnSchema::new("count", SchemaDataType::Int64));

    let to = DataFrameSchema::new("v2", SchemaVersion::new(1, 1, 0))
        .with_column(ColumnSchema::new("count", SchemaDataType::Float64));

    let migrator = SchemaMigrator::empty();
    let report = migrator.check_compatibility(&from, &to);
    assert!(report.is_compatible);
    assert!(!report.non_breaking_changes.is_empty());
}

#[test]
fn test_compatibility_dropped_column_noted() {
    let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
        .with_column(ColumnSchema::new("legacy", SchemaDataType::String));

    let to = DataFrameSchema::new("v2", SchemaVersion::new(2, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64));

    let migrator = SchemaMigrator::empty();
    let report = migrator.check_compatibility(&from, &to);
    // Dropping a column from the source doesn't prevent data flow to `to`
    assert!(report.is_compatible);
    assert!(report
        .non_breaking_changes
        .iter()
        .any(|c| c.contains("legacy")));
}

// ── Schema registry ───────────────────────────────────────────────────────────

#[test]
fn test_registry_register_and_retrieve() {
    let mut registry = SchemaRegistry::new();
    registry.register(make_schema_v1()).expect("register v1");
    registry.register(make_schema_v2()).expect("register v2");

    let latest = registry.get_latest("users").expect("latest");
    assert_eq!(latest.version, SchemaVersion::new(1, 1, 0));

    let v1 = registry.get_version("users", &SchemaVersion::new(1, 0, 0));
    assert!(v1.is_some());

    let versions = registry.versions_of("users");
    assert_eq!(versions.len(), 2);
}

#[test]
fn test_registry_duplicate_version_error() {
    let mut registry = SchemaRegistry::new();
    registry.register(make_schema_v1()).expect("first register");
    let err = registry.register(make_schema_v1());
    assert!(err.is_err());
}

#[test]
fn test_registry_migration_path_single_hop() {
    let mut registry = SchemaRegistry::new();
    registry.register(make_schema_v1()).expect("v1");
    registry.register(make_schema_v2()).expect("v2");
    registry
        .add_migration_for_schema("users", make_migration_v1_v2())
        .expect("add migration");

    let path = registry
        .find_migration_path(
            "users",
            &SchemaVersion::new(1, 0, 0),
            &SchemaVersion::new(1, 1, 0),
        )
        .expect("path");

    assert_eq!(path.len(), 1);
    assert_eq!(path[0].id, "m_users_001");
}

#[test]
fn test_registry_migration_path_multi_hop() {
    let mut registry = SchemaRegistry::new();

    let v1 = DataFrameSchema::new("products", SchemaVersion::new(1, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64));
    let v2 = DataFrameSchema::new("products", SchemaVersion::new(1, 1, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
        .with_column(ColumnSchema::new("name", SchemaDataType::String));
    let v3 = DataFrameSchema::new("products", SchemaVersion::new(1, 2, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
        .with_column(ColumnSchema::new("name", SchemaDataType::String))
        .with_column(ColumnSchema::new("price", SchemaDataType::Float64));

    registry.register(v1).expect("v1");
    registry.register(v2).expect("v2");
    registry.register(v3).expect("v3");

    let m1 = MigrationBuilder::new(
        "p_m001",
        "products",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .add_column(ColumnSchema::new("name", SchemaDataType::String), None)
    .build();

    let m2 = MigrationBuilder::new(
        "p_m002",
        "products",
        SchemaVersion::new(1, 1, 0),
        SchemaVersion::new(1, 2, 0),
    )
    .add_column(
        ColumnSchema::new("price", SchemaDataType::Float64).with_default(DefaultValue::Float(0.0)),
        None,
    )
    .build();

    registry
        .add_migration_for_schema("products", m1)
        .expect("m1");
    registry
        .add_migration_for_schema("products", m2)
        .expect("m2");

    let path = registry
        .find_migration_path(
            "products",
            &SchemaVersion::new(1, 0, 0),
            &SchemaVersion::new(1, 2, 0),
        )
        .expect("path");

    assert_eq!(path.len(), 2);
}

#[test]
fn test_registry_no_path_error() {
    let registry = SchemaRegistry::new();
    let err = registry.find_migration_path(
        "nonexistent",
        &SchemaVersion::new(1, 0, 0),
        &SchemaVersion::new(2, 0, 0),
    );
    assert!(err.is_err());
}

#[test]
fn test_registry_same_version_path_empty() {
    let registry = SchemaRegistry::new();
    let path = registry
        .find_migration_path(
            "any",
            &SchemaVersion::new(1, 0, 0),
            &SchemaVersion::new(1, 0, 0),
        )
        .expect("same version");
    assert!(path.is_empty());
}

// ── End-to-end: registry + migrator ─────────────────────────────────────────

#[test]
fn test_end_to_end_migrate_via_registry() {
    let mut registry = SchemaRegistry::new();
    registry.register(make_schema_v1()).expect("v1");
    registry.register(make_schema_v2()).expect("v2");
    registry
        .add_migration_for_schema("users", make_migration_v1_v2())
        .expect("add migration");

    let migrator = SchemaMigrator::new(registry);
    let df = make_df_users();
    let result = migrator
        .migrate(
            &df,
            "users",
            &SchemaVersion::new(1, 0, 0),
            &SchemaVersion::new(1, 1, 0),
        )
        .expect("migrate");

    assert!(result.contains_column("email"));
    assert!(result.contains_column("id"));
    assert!(result.contains_column("name"));
    assert_eq!(result.row_count(), 3);

    // Validate migrated result against target schema
    let _target_schema = make_schema_v2();
    let report = migrator.registry.get_latest("users").map(|s| {
        let m = SchemaMigrator::empty();
        m.validate(&result, s).expect("validate")
    });
    if let Some(report) = report {
        assert!(report.is_valid, "errors: {:?}", report.errors);
    }
}

#[test]
fn test_end_to_end_schema_bundle_save_load() {
    use pandrs::schema_evolution::{load_bundle, save_bundle, SchemaBundle};

    let bundle = SchemaBundle::new()
        .with_schema(make_schema_v1())
        .with_schema(make_schema_v2())
        .with_migration(make_migration_v1_v2());

    let dir = env::temp_dir();
    let path = dir.join("pandrs_test_bundle.json");
    let path_str = path.to_str().expect("path");

    save_bundle(&bundle, path_str, SchemaFormat::Json).expect("save bundle");
    let loaded = load_bundle(path_str).expect("load bundle");

    assert_eq!(loaded.schemas.len(), 2);
    assert_eq!(loaded.migrations.len(), 1);
    assert_eq!(loaded.schemas[0].name, "users");

    let _ = fs::remove_file(&path);
}

// ── SchemaChange breaking change detection ───────────────────────────────────

#[test]
fn test_schema_change_breaking_detection() {
    let rm = SchemaChange::RemoveColumn {
        name: "col".to_string(),
    };
    let add = SchemaChange::AddColumn {
        schema: ColumnSchema::new("new", SchemaDataType::String),
        position: None,
    };
    let rename = SchemaChange::RenameColumn {
        from: "a".to_string(),
        to: "b".to_string(),
    };
    let change_type = SchemaChange::ChangeType {
        column: "x".to_string(),
        new_type: SchemaDataType::Int64,
        converter: None,
    };
    let make_non_nullable = SchemaChange::SetNullable {
        column: "y".to_string(),
        nullable: false,
    };
    let make_nullable = SchemaChange::SetNullable {
        column: "y".to_string(),
        nullable: true,
    };

    assert!(rm.is_breaking());
    assert!(!add.is_breaking());
    // A rename breaks anything still addressing the column by its old name.
    assert!(rename.is_breaking());
    assert!(change_type.is_breaking());
    assert!(make_non_nullable.is_breaking());
    assert!(!make_nullable.is_breaking());

    let required_no_default = SchemaChange::AddColumn {
        schema: ColumnSchema::new("required", SchemaDataType::String).with_nullable(false),
        position: None,
    };
    assert!(required_no_default.is_breaking());
}

#[test]
fn test_migration_breaking_changes_summary() {
    let migration = MigrationBuilder::new(
        "breaking_m",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(2, 0, 0),
    )
    .remove_column("legacy_field")
    .add_column(ColumnSchema::new("new_field", SchemaDataType::String), None)
    .change_type("id", SchemaDataType::String, None)
    .build();

    assert!(migration.has_breaking_changes());
    let breaking = migration.breaking_changes();
    assert_eq!(breaking.len(), 2); // remove_column + change_type
}

// ── Default values ────────────────────────────────────────────────────────────

#[test]
fn test_add_column_with_bool_default() {
    let df = make_df_users();
    let migrator = SchemaMigrator::empty();
    // Apply via single-change migration
    let migration = MigrationBuilder::new(
        "bool_default",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .add_column(
        ColumnSchema::new("active", SchemaDataType::Boolean).with_default(DefaultValue::Bool(true)),
        None,
    )
    .build();
    let result = migrator.apply_migration(&df, &migration).expect("apply");
    assert!(result.contains_column("active"));
    let vals = result.get_column_numeric_values("active").expect("numeric");
    assert!(vals.iter().all(|&v| v == 1.0)); // true -> 1.0
}

#[test]
fn test_add_column_with_string_default() {
    let df = make_df_users();
    let migrator = SchemaMigrator::empty();
    let migration = MigrationBuilder::new(
        "str_default",
        "users",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .add_column(
        ColumnSchema::new("department", SchemaDataType::String)
            .with_default(DefaultValue::Str("engineering".to_string())),
        None,
    )
    .build();
    let result = migrator.apply_migration(&df, &migration).expect("apply");
    assert!(result.contains_column("department"));
    let vals = result
        .get_column_string_values("department")
        .expect("values");
    assert!(vals.iter().all(|v| v == "engineering"));
}
