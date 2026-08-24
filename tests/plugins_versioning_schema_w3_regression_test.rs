#![allow(clippy::result_large_err)]
//! Wave 3 regression tests for `pandrs::plugins`, `pandrs::versioning`, and
//! `pandrs::schema_evolution`.
//!
//! These exercise the public API end-to-end for the specific defects fixed
//! in this pass. Focused white-box tests for the same fixes also live
//! alongside the code they cover (`src/schema_evolution/migrator.rs`,
//! `src/schema_evolution/registry.rs`, `src/versioning/core.rs`,
//! `src/versioning/tracker.rs`, `src/versioning/mod.rs`,
//! `src/plugins/registry.rs`, `src/plugins/builtins/transform.rs`); this
//! file covers the same ground through the crate's public surface instead
//! of internal helpers.
//!
//! Covers:
//! - A migration preserves `i64` (including values above 2^53) and `bool`
//!   column types exactly, instead of re-materializing every column as
//!   `f64`/`String`.
//! - `NotNull` catches a real numeric `NaN`, both when the constrained
//!   column is declared in the schema and when it's only referenced by an
//!   otherwise-undeclared constraint.
//! - `SchemaRegistry::add_migration` (not just `add_migration_for_schema`)
//!   indexes the migration so `find_migration_path`/`migrate` finds it.
//! - `versioning::core::VersionId::new()` produces unique, well-formed ids.
//! - Unregistering a plugin from a local `PluginRegistry` actually frees
//!   its name for re-registration.

use pandrs::plugins::{CsvSourcePlugin, FilterTransformPlugin, PluginRegistry, TransformPlugin};
use pandrs::schema_evolution::{
    ColumnSchema, DataFrameSchema, MigrationBuilder, SchemaConstraint, SchemaDataType,
    SchemaMigrator, SchemaRegistry, SchemaVersion, ValidationErrorType,
};
use pandrs::versioning::{DataFrameVersioning, LineageTracker, VersionId};
use pandrs::{DataFrame, Series};
use std::collections::HashMap;
use std::collections::HashSet;

// ── 1. Migration preserves i64 and bool types ──────────────────────────────

#[test]
fn regression_migration_preserves_i64_and_bool_types() {
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        // Above 2^53: an f64 round trip would lose precision here.
        Series::new(vec![9_007_199_254_740_993i64, 42], Some("id".to_string())).expect("series"),
    )
    .expect("add id");
    df.add_column(
        "verified".to_string(),
        Series::new(vec![true, false], Some("verified".to_string())).expect("series"),
    )
    .expect("add verified");
    df.add_column(
        "name".to_string(),
        Series::new(
            vec!["Alice".to_string(), "Bob".to_string()],
            Some("name".to_string()),
        )
        .expect("series"),
    )
    .expect("add name");

    let migration = MigrationBuilder::new(
        "add_email",
        "accounts",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .add_column(
        ColumnSchema::new("email", SchemaDataType::String)
            .with_default(pandrs::schema_evolution::DefaultValue::Str(String::new())),
        None,
    )
    .build();

    let migrator = SchemaMigrator::empty();
    let result = migrator
        .apply_migration(&df, &migration)
        .expect("migration should apply");

    // The migration only added a column; id/verified must come through
    // with their real Rust types intact, not f64/String.
    let ids = result.get_column::<i64>("id").expect("id stays i64");
    assert_eq!(ids.values(), &[9_007_199_254_740_993i64, 42]);

    let verified = result
        .get_column::<bool>("verified")
        .expect("verified stays bool");
    assert_eq!(verified.values(), &[true, false]);
}

// ── 2. NotNull catches numeric NaN, declared and undeclared columns ────────

#[test]
fn regression_not_null_catches_numeric_nan_declared_column() {
    let mut df = DataFrame::new();
    df.add_column(
        "amount".to_string(),
        Series::new(vec![10.0f64, f64::NAN, 30.0], Some("amount".to_string())).expect("series"),
    )
    .expect("add");

    let schema = DataFrameSchema::new("orders", SchemaVersion::initial())
        .with_column(ColumnSchema::new("amount", SchemaDataType::Float64))
        .with_constraint(SchemaConstraint::NotNull("amount".to_string()));

    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(!report.is_valid, "a NaN entry must fail NOT NULL");
    assert!(report
        .errors
        .iter()
        .any(|e| e.error_type == ValidationErrorType::NullViolation));
}

#[test]
fn regression_not_null_catches_numeric_nan_undeclared_column() {
    let mut df = DataFrame::new();
    df.add_column(
        "amount".to_string(),
        Series::new(vec![10.0f64, f64::NAN, 30.0], Some("amount".to_string())).expect("series"),
    )
    .expect("add");

    // "amount" is not in `schema.columns` at all -- only reachable because
    // `validate()` iterates `schema.constraints` directly rather than
    // filtering constraints through the declared-columns loop.
    let schema = DataFrameSchema::new("orders", SchemaVersion::initial())
        .with_constraint(SchemaConstraint::NotNull("amount".to_string()));

    let migrator = SchemaMigrator::empty();
    let report = migrator.validate(&df, &schema).expect("validate");
    assert!(
        !report.is_valid,
        "a constraint on an undeclared column must still be evaluated"
    );
    assert!(report
        .errors
        .iter()
        .any(|e| e.error_type == ValidationErrorType::NullViolation));
}

// ── 3. add_migration (not just add_migration_for_schema) indexes ───────────

#[test]
fn regression_add_migration_indexes_for_find_migration_path() {
    let mut registry = SchemaRegistry::new();
    registry
        .register(
            DataFrameSchema::new("widgets", SchemaVersion::new(1, 0, 0))
                .with_column(ColumnSchema::new("id", SchemaDataType::Int64)),
        )
        .expect("register v1");
    registry
        .register(
            DataFrameSchema::new("widgets", SchemaVersion::new(1, 1, 0))
                .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
                .with_column(ColumnSchema::new("weight", SchemaDataType::Float64)),
        )
        .expect("register v2");

    let migration = MigrationBuilder::new(
        "widgets_m1",
        "widgets",
        SchemaVersion::new(1, 0, 0),
        SchemaVersion::new(1, 1, 0),
    )
    .add_column(
        ColumnSchema::new("weight", SchemaDataType::Float64)
            .with_default(pandrs::schema_evolution::DefaultValue::Float(0.0)),
        None,
    )
    .build();

    // The plain `add_migration` (not `_for_schema`) call is the one that
    // was previously a silent no-op for path-finding purposes.
    registry
        .add_migration(migration)
        .expect("add_migration should succeed and index the migration");

    let migrator = SchemaMigrator::new(registry);
    let mut df = DataFrame::new();
    df.add_column(
        "id".to_string(),
        Series::new(vec![1i64, 2], Some("id".to_string())).expect("series"),
    )
    .expect("add");

    let result = migrator
        .migrate(
            &df,
            "widgets",
            &SchemaVersion::new(1, 0, 0),
            &SchemaVersion::new(1, 1, 0),
        )
        .expect("migrate() should find the path add_migration indexed");
    assert!(result.contains_column("weight"));
}

// ── 4. VersionId::new() uniqueness ──────────────────────────────────────────

#[test]
fn regression_version_id_new_is_unique_across_many_calls() {
    let mut seen: HashSet<String> = HashSet::new();
    for _ in 0..500 {
        let id = VersionId::new();
        assert!(
            seen.insert(id.as_str().to_string()),
            "VersionId::new() produced a duplicate: {}",
            id
        );
    }
}

#[test]
fn regression_lineage_tracker_uses_unique_ids_end_to_end() {
    let mut tracker = LineageTracker::new();
    let mut df = DataFrame::new();
    df.add_column(
        "x".to_string(),
        Series::new(vec![1i64, 2, 3], Some("x".to_string())).expect("series"),
    )
    .expect("add");

    let v1 = df.create_version(&mut tracker);
    let v2 = df.create_version(&mut tracker);
    assert_ne!(v1, v2, "two independent registrations must not collide");
    assert!(tracker.get_version(&v1).is_some());
    assert!(tracker.get_version(&v2).is_some());
}

// ── 5. Plugin unregister (local registry) ───────────────────────────────────

#[test]
fn regression_plugin_unregister_frees_the_name_on_a_local_registry() {
    let mut registry = PluginRegistry::new();
    registry
        .register_source(CsvSourcePlugin::arc())
        .expect("register");
    registry
        .register_transform(FilterTransformPlugin::arc())
        .expect("register");
    assert_eq!(registry.plugin_count(), 2);

    let removed = registry.unregister_transform("filter");
    assert!(removed.is_some());
    assert!(!registry.has_transform("filter"));
    assert_eq!(registry.plugin_count(), 1);

    // The name is free again: re-registering must succeed where it would
    // previously have had no way to ever un-stick (no unregister existed).
    registry
        .register_transform(FilterTransformPlugin::arc())
        .expect("re-register after unregister must succeed");
    assert_eq!(registry.plugin_count(), 2);
}

// ── Bonus: end-to-end sanity for a few more fixes in this pass ─────────────

#[test]
fn regression_filter_plugin_rejects_unknown_operator() {
    let mut df = DataFrame::new();
    df.add_column(
        "v".to_string(),
        Series::new(vec![1i64, 2, 3], Some("v".to_string())).expect("series"),
    )
    .expect("add");

    let mut opts = HashMap::new();
    opts.insert("column".to_string(), "v".to_string());
    opts.insert("operator".to_string(), "does_not_exist".to_string());
    opts.insert("value".to_string(), "1".to_string());

    let result = FilterTransformPlugin::new().transform(df, &opts);
    assert!(
        result.is_err(),
        "an unknown filter operator must error rather than silently keep nothing"
    );
}

#[test]
fn regression_compatibility_report_separates_data_loss_from_non_breaking() {
    let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
        .with_column(ColumnSchema::new("legacy_notes", SchemaDataType::String));
    let to = DataFrameSchema::new("v2", SchemaVersion::new(2, 0, 0))
        .with_column(ColumnSchema::new("id", SchemaDataType::Int64));

    let migrator = SchemaMigrator::empty();
    let report = migrator.check_compatibility(&from, &to);
    assert!(
        report.is_compatible,
        "dropping a column doesn't block the flow"
    );
    assert!(report.data_loss.iter().any(|c| c.contains("legacy_notes")));
}
