//! Schema migrator: applies migrations to DataFrames and validates schemas
//!
//! The `SchemaMigrator` is the primary entry point for performing schema
//! evolution operations on actual data. It can:
//! - Apply individual migrations to a DataFrame
//! - Chain migrations to move data across multiple schema versions
//! - Validate a DataFrame against a schema (checking constraints and types)
//! - Infer a schema from an existing DataFrame
//! - Check compatibility between two schemas

use std::collections::HashMap;

use crate::core::error::{Error, Result};
use crate::dataframe::DataFrame;
use crate::series::Series;

use super::evolution::{Migration, SchemaChange};
use super::registry::SchemaRegistry;
use super::schema::{
    ColumnSchema, DataFrameSchema, DefaultValue, SchemaConstraint, SchemaDataType, SchemaVersion,
};

/// A single validation error found when checking a DataFrame against a schema
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// The column(s) involved in the error
    pub column: String,
    /// Human-readable error message
    pub message: String,
    /// The type of violation
    pub error_type: ValidationErrorType,
}

/// Classification of validation errors
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorType {
    /// Column exists in schema but not in DataFrame
    MissingColumn,
    /// Column exists in DataFrame but not in schema
    ExtraColumn,
    /// Column has the wrong data type
    TypeMismatch,
    /// Null value found in non-nullable column
    NullViolation,
    /// Value outside allowed range
    RangeViolation,
    /// Value does not match required regex pattern
    RegexViolation,
    /// Uniqueness constraint violated
    UniqueViolation,
    /// Enum constraint violated
    EnumViolation,
    /// Other constraint violation
    ConstraintViolation,
}

/// Report of schema validation results
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Whether the DataFrame is fully valid against the schema
    pub is_valid: bool,
    /// List of validation errors found
    pub errors: Vec<ValidationError>,
    /// Non-fatal warnings (e.g., extra columns not in schema)
    pub warnings: Vec<String>,
}

impl ValidationReport {
    fn new() -> Self {
        ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn add_error(
        &mut self,
        column: impl Into<String>,
        message: impl Into<String>,
        error_type: ValidationErrorType,
    ) {
        self.is_valid = false;
        self.errors.push(ValidationError {
            column: column.into(),
            message: message.into(),
            error_type,
        });
    }

    fn add_warning(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }
}

/// A change that breaks backward compatibility between two schemas
#[derive(Debug, Clone)]
pub struct BreakingChange {
    /// Description of the breaking change
    pub description: String,
    /// The column(s) affected
    pub affected_columns: Vec<String>,
}

/// Report comparing compatibility of two schemas
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    /// Whether `to`'s required data can be structurally satisfied by `from`
    /// (required columns present, types castable). This is a *flow*
    /// check, not a losslessness guarantee -- see `data_loss`.
    pub is_compatible: bool,
    /// List of breaking changes that prevent compatibility
    pub breaking_changes: Vec<BreakingChange>,
    /// List of non-breaking changes (informational; no data-loss connotation)
    pub non_breaking_changes: Vec<String>,
    /// Columns present in `from` but absent from `to`. A dropped column
    /// never makes `is_compatible` false (the target schema doesn't need
    /// that data), but the data in it is genuinely lost, which is a
    /// different claim than "non-breaking" -- keeping this separate from
    /// `non_breaking_changes` means "compatible" is never read as
    /// "lossless".
    pub data_loss: Vec<String>,
}

impl CompatibilityReport {
    fn new() -> Self {
        CompatibilityReport {
            is_compatible: true,
            breaking_changes: Vec::new(),
            non_breaking_changes: Vec::new(),
            data_loss: Vec::new(),
        }
    }

    fn add_breaking(&mut self, description: impl Into<String>, columns: Vec<String>) {
        self.is_compatible = false;
        self.breaking_changes.push(BreakingChange {
            description: description.into(),
            affected_columns: columns,
        });
    }

    fn add_non_breaking(&mut self, description: impl Into<String>) {
        self.non_breaking_changes.push(description.into());
    }
}

/// Copy a single column from `df` into `result` under `dest_name`, preserving
/// its concrete element type exactly.
///
/// `DataFrame::is_numeric_column` is a real, physical-type check (it is not
/// a stub), and `DataFrame::get_column::<T>` lets us downcast to each of the
/// concrete types this DataFrame model supports. Trying each in turn (rather
/// than routing everything through `get_column_numeric_values` /
/// `get_column_string_values`, which re-materialise the column as `f64` or
/// `String`) means:
/// - `i64` values above 2^53 keep their exact precision instead of being
///   rounded through `f64`.
/// - `bool` columns stay `bool` instead of becoming `"true"`/`"false"` strings.
/// - A `String` column whose values merely *look* numeric (e.g. a zip code
///   `"02134"`) is never misclassified as numeric and reinterpreted as a
///   number, which would silently destroy the leading zero.
///
/// A column whose concrete element type isn't one of `i64`/`f64`/`i32`/`f32`/
/// `bool`/`String` (e.g. a `chrono` date/time column) is genuinely
/// unsupported by this migration engine today: this returns
/// `Error::NotImplemented` naming the column rather than fabricating a
/// placeholder value that would silently corrupt every downstream consumer.
fn copy_one_column(
    df: &DataFrame,
    src_name: &str,
    dest_name: &str,
    result: &mut DataFrame,
) -> Result<()> {
    macro_rules! try_copy {
        ($ty:ty) => {
            if let Ok(s) = df.get_column::<$ty>(src_name) {
                let series = Series::new(s.to_vec(), Some(dest_name.to_string()))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                result.add_column(dest_name.to_string(), series)?;
                return Ok(());
            }
        };
    }
    try_copy!(i64);
    try_copy!(f64);
    try_copy!(i32);
    try_copy!(f32);
    try_copy!(bool);
    try_copy!(String);

    Err(Error::NotImplemented(format!(
        "schema migration cannot copy column '{}': its element type is not one of \
         i64, f64, i32, f32, bool, or String, which are the only concrete column \
         types this migration engine supports",
        src_name
    )))
}

/// Helper: copy all columns from `df` into a new DataFrame, preserving each
/// column's concrete element type exactly (see [`copy_one_column`]). The
/// order follows the provided `column_list`.
fn copy_columns(df: &DataFrame, column_list: &[String]) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in column_list {
        if !df.contains_column(col_name) {
            return Err(Error::ColumnNotFound(col_name.clone()));
        }
        copy_one_column(df, col_name, col_name, &mut result)?;
    }
    Ok(result)
}

/// Convert a single column of `df` to `new_type`, writing the result into
/// `result` under the same name.
///
/// Every branch produces the *real* Rust type it claims: `Int64` yields a
/// genuine `Series<i64>` (never `f64` standing in for it), `Boolean` yields
/// a genuine `Series<bool>`, etc. This matters because
/// [`SchemaMigrator::check_column_type`] validates a `Boolean`-declared
/// column by downcasting to `Series<bool>` -- a migration that produced
/// `f64` 0.0/1.0 values instead would make "migrate, then validate" fail
/// even on data the migration itself just wrote.
///
/// `DateTime`/`Categorical`/`List` targets have no dedicated physical
/// column representation in this DataFrame model (see `copy_one_column`),
/// so converting *to* one of them returns `Error::NotImplemented` rather
/// than silently storing the column as a plain string and claiming the
/// conversion happened.
fn convert_column_type(
    df: &DataFrame,
    column: &str,
    new_type: &SchemaDataType,
    result: &mut DataFrame,
) -> Result<()> {
    match new_type {
        SchemaDataType::Float64 => {
            let values: Vec<f64> = if df.is_numeric_column(column) {
                df.get_column_numeric_values(column)?
            } else if df.get_column::<bool>(column).is_ok() {
                df.get_column_numeric_values(column)? // bool -> 0.0/1.0
            } else {
                let string_values = df.get_column_string_values(column)?;
                string_values
                    .iter()
                    .map(|s| {
                        s.trim().parse::<f64>().map_err(|e| {
                            Error::Cast(format!("Cannot cast '{}' to Float64: {}", s, e))
                        })
                    })
                    .collect::<Result<Vec<f64>>>()?
            };
            let series = Series::new(values, Some(column.to_string()))
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            result.add_column(column.to_string(), series)?;
        }
        SchemaDataType::Int64 => {
            let values: Vec<i64> = if df.is_numeric_column(column) {
                df.get_column_numeric_values(column)?
                    .iter()
                    .map(|v| v.trunc() as i64)
                    .collect()
            } else if df.get_column::<bool>(column).is_ok() {
                df.get_column_numeric_values(column)?
                    .iter()
                    .map(|v| *v as i64)
                    .collect()
            } else {
                let string_values = df.get_column_string_values(column)?;
                string_values
                    .iter()
                    .map(|s| {
                        s.trim()
                            .parse::<i64>()
                            .or_else(|_| s.trim().parse::<f64>().map(|f| f.trunc() as i64))
                            .map_err(|e| {
                                Error::Cast(format!("Cannot cast '{}' to Int64: {}", s, e))
                            })
                    })
                    .collect::<Result<Vec<i64>>>()?
            };
            let series = Series::new(values, Some(column.to_string()))
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            result.add_column(column.to_string(), series)?;
        }
        SchemaDataType::Boolean => {
            let values: Vec<bool> = if let Ok(s) = df.get_column::<bool>(column) {
                s.to_vec()
            } else if df.is_numeric_column(column) {
                df.get_column_numeric_values(column)?
                    .iter()
                    .map(|v| *v != 0.0)
                    .collect()
            } else {
                let string_values = df.get_column_string_values(column)?;
                string_values
                    .iter()
                    .map(|s| match s.trim().to_lowercase().as_str() {
                        "true" | "1" | "yes" => Ok(true),
                        "false" | "0" | "no" | "" => Ok(false),
                        _ => Err(Error::Cast(format!("Cannot cast '{}' to Boolean", s))),
                    })
                    .collect::<Result<Vec<bool>>>()?
            };
            let series = Series::new(values, Some(column.to_string()))
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            result.add_column(column.to_string(), series)?;
        }
        SchemaDataType::String => {
            let values = df.get_column_string_values(column)?;
            let series = Series::new(values, Some(column.to_string()))
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            result.add_column(column.to_string(), series)?;
        }
        SchemaDataType::DateTime
        | SchemaDataType::Categorical { .. }
        | SchemaDataType::List { .. } => {
            return Err(Error::NotImplemented(format!(
                "ChangeType to {} is not implemented for column '{}': this DataFrame's \
                 schema-migration engine only supports converting to Int64, Float64, \
                 Boolean, or String columns (DateTime/Categorical/List have no dedicated \
                 physical column representation here, so silently storing the raw \
                 string and calling it converted would be dishonest)",
                new_type, column
            )));
        }
    }
    Ok(())
}

/// Primary entry point for schema migration and validation
pub struct SchemaMigrator {
    /// The registry of schemas and migrations
    pub registry: SchemaRegistry,
}

impl SchemaMigrator {
    /// Create a new migrator with the given registry
    pub fn new(registry: SchemaRegistry) -> Self {
        SchemaMigrator { registry }
    }

    /// Create a migrator with an empty registry
    pub fn empty() -> Self {
        SchemaMigrator {
            registry: SchemaRegistry::new(),
        }
    }

    /// Apply a single migration to a DataFrame, returning the transformed DataFrame.
    ///
    /// Applies each `SchemaChange` in the migration sequentially.
    pub fn apply_migration(&self, df: &DataFrame, migration: &Migration) -> Result<DataFrame> {
        let mut result = df.clone();
        for change in &migration.changes {
            result = self.apply_change(&result, change)?;
        }
        Ok(result)
    }

    /// Apply a single SchemaChange to a DataFrame
    fn apply_change(&self, df: &DataFrame, change: &SchemaChange) -> Result<DataFrame> {
        match change {
            SchemaChange::AddColumn { schema, position } => {
                self.apply_add_column(df, schema, *position)
            }
            SchemaChange::RemoveColumn { name } => self.apply_remove_column(df, name),
            SchemaChange::RenameColumn { from, to } => self.apply_rename_column(df, from, to),
            SchemaChange::ChangeType {
                column, new_type, ..
            } => self.apply_change_type(df, column, new_type),
            SchemaChange::ReorderColumns { order } => self.apply_reorder_columns(df, order),
            // Metadata-only changes don't modify the DataFrame data
            SchemaChange::AddConstraint { .. }
            | SchemaChange::RemoveConstraint { .. }
            | SchemaChange::SetDefault { .. }
            | SchemaChange::SetNullable { .. }
            | SchemaChange::SetColumnDescription { .. }
            | SchemaChange::AddColumnTag { .. }
            | SchemaChange::RemoveColumnTag { .. }
            | SchemaChange::SetMetadata { .. }
            | SchemaChange::RemoveMetadata { .. } => Ok(df.clone()),
        }
    }

    fn apply_add_column(
        &self,
        df: &DataFrame,
        schema: &ColumnSchema,
        position: Option<usize>,
    ) -> Result<DataFrame> {
        let row_count = df.row_count();

        // Gather existing columns first
        let existing_cols = df.column_names().to_vec();

        // If the column already exists, just return the df unchanged
        if existing_cols.iter().any(|c| c == &schema.name) {
            return Ok(df.clone());
        }

        // Determine the new column order
        let mut new_order = existing_cols.clone();
        let insert_pos = position.unwrap_or(new_order.len()).min(new_order.len());
        new_order.insert(insert_pos, schema.name.clone());

        // Copy existing columns
        let mut result = copy_columns(df, &existing_cols)?;

        // Add the new column with the appropriate default.
        match &schema.data_type {
            SchemaDataType::Int64 | SchemaDataType::Float64 => {
                // Resolve the fill value. When no default is given and the
                // column is declared nullable, the fill is `f64::NAN` --
                // `f64` is the only numeric column type in this DataFrame
                // model that can hold a real NA marker (there is no bit for
                // "missing" in `i64`), so a nullable numeric column with no
                // default is backed by `Series<f64>` and filled with NaN.
                // This mirrors the same int -> float NaN-upcast convention
                // `DataFrame::concat_rows` already uses for the equivalent
                // problem (a numeric column missing on one side of a
                // concat). A non-nullable column with no default has no
                // missing-value semantics to preserve, so it falls back to
                // a real zero of the declared type instead.
                let fill: f64 = match &schema.default_value {
                    Some(DefaultValue::Int(v)) => *v as f64,
                    Some(DefaultValue::Float(v)) => *v,
                    Some(DefaultValue::Null) => f64::NAN,
                    Some(_) => 0.0,
                    None if schema.nullable => f64::NAN,
                    None => 0.0,
                };

                if matches!(schema.data_type, SchemaDataType::Int64) && !fill.is_nan() {
                    let data: Vec<i64> = vec![fill as i64; row_count];
                    let series = Series::new(data, Some(schema.name.clone()))
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                    result.add_column(schema.name.clone(), series)?;
                } else {
                    let data: Vec<f64> = vec![fill; row_count];
                    let series = Series::new(data, Some(schema.name.clone()))
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                    result.add_column(schema.name.clone(), series)?;
                }
            }
            SchemaDataType::Boolean => {
                // Unlike Int64/Float64, `bool` has no NaN-equivalent in this
                // DataFrame model at all, so there is no honest way to
                // represent "missing" for a nullable Boolean column with no
                // default. Rather than silently upcasting to a numeric NA
                // representation (which would make the column fail the
                // `Series<bool>` round-trip check in `check_column_type`),
                // this falls back to a real `false` -- a genuine, typed
                // value -- and documents the limitation instead of
                // fabricating a null that can't exist here.
                let default = matches!(&schema.default_value, Some(DefaultValue::Bool(true)));
                let data: Vec<bool> = vec![default; row_count];
                let series = Series::new(data, Some(schema.name.clone()))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                result.add_column(schema.name.clone(), series)?;
            }
            SchemaDataType::String
            | SchemaDataType::DateTime
            | SchemaDataType::Categorical { .. }
            | SchemaDataType::List { .. } => {
                let default = match &schema.default_value {
                    Some(DefaultValue::Str(v)) => v.clone(),
                    Some(DefaultValue::Null) => String::new(),
                    _ => String::new(),
                };
                let data: Vec<String> = vec![default; row_count];
                let series = Series::new(data, Some(schema.name.clone()))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                result.add_column(schema.name.clone(), series)?;
            }
        }

        // Reorder to the desired position
        self.apply_reorder_columns(&result, &new_order)
    }

    fn apply_remove_column(&self, df: &DataFrame, name: &str) -> Result<DataFrame> {
        if !df.contains_column(name) {
            return Err(Error::ColumnNotFound(name.to_string()));
        }
        // Build list of columns to keep
        let keep: Vec<String> = df
            .column_names()
            .iter()
            .filter(|c| c.as_str() != name)
            .cloned()
            .collect();
        copy_columns(df, &keep)
    }

    fn apply_rename_column(&self, df: &DataFrame, from: &str, to: &str) -> Result<DataFrame> {
        if !df.contains_column(from) {
            return Err(Error::ColumnNotFound(from.to_string()));
        }
        // Build a new DataFrame with the column renamed, preserving every
        // column's concrete element type (see `copy_one_column`).
        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            let target_name = if col_name.as_str() == from {
                to
            } else {
                col_name
            };
            copy_one_column(df, col_name.as_str(), target_name, &mut result)?;
        }
        Ok(result)
    }

    fn apply_change_type(
        &self,
        df: &DataFrame,
        column: &str,
        new_type: &SchemaDataType,
    ) -> Result<DataFrame> {
        if !df.contains_column(column) {
            return Err(Error::ColumnNotFound(column.to_string()));
        }

        // Rebuild the DataFrame, replacing the target column with the converted version
        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            if col_name.as_str() != column {
                // Copy other columns as-is, preserving their concrete type.
                copy_one_column(df, col_name.as_str(), col_name.as_str(), &mut result)?;
            } else {
                convert_column_type(df, column, new_type, &mut result)?;
            }
        }
        Ok(result)
    }

    fn apply_reorder_columns(&self, df: &DataFrame, order: &[String]) -> Result<DataFrame> {
        // Validate all requested columns are present
        for col in order {
            if !df.contains_column(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }
        copy_columns(df, order)
    }

    /// Migrate a DataFrame from one schema version to another by finding and applying
    /// the migration path in the registry.
    pub fn migrate(
        &self,
        df: &DataFrame,
        schema_name: &str,
        from: &SchemaVersion,
        to: &SchemaVersion,
    ) -> Result<DataFrame> {
        let path = self.registry.find_migration_path(schema_name, from, to)?;

        let mut result = df.clone();
        for migration in path {
            result = self.apply_migration(&result, migration)?;
        }
        Ok(result)
    }

    /// Validate a DataFrame against a schema.
    ///
    /// Checks:
    /// - All schema columns exist in the DataFrame
    /// - Column types match the schema (heuristic)
    /// - Every constraint in `schema.constraints` is satisfied (NotNull,
    ///   Range, Enum, Regex, Unique, composite Unique) -- constraints are
    ///   evaluated by iterating `schema.constraints` directly, not by
    ///   looking them up per declared column, so a constraint referencing a
    ///   column that isn't (yet) listed in `schema.columns` is still
    ///   evaluated rather than silently skipped.
    pub fn validate(&self, df: &DataFrame, schema: &DataFrameSchema) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        let df_columns: std::collections::HashSet<String> =
            df.column_names().iter().cloned().collect();
        let schema_columns: std::collections::HashSet<String> =
            schema.columns.iter().map(|c| c.name.clone()).collect();

        // Check for missing columns (in schema but not in DataFrame)
        for col in &schema.columns {
            if !df_columns.contains(&col.name) {
                report.add_error(
                    &col.name,
                    format!(
                        "Column '{}' defined in schema but not present in DataFrame",
                        col.name
                    ),
                    ValidationErrorType::MissingColumn,
                );
            }
        }

        // Check for extra columns (in DataFrame but not in schema)
        for col_name in &df_columns {
            if !schema_columns.contains(col_name) {
                report.add_warning(format!(
                    "Column '{}' exists in DataFrame but is not defined in schema",
                    col_name
                ));
            }
        }

        // Check type compatibility for existing, declared columns.
        for col_schema in &schema.columns {
            if !df_columns.contains(&col_schema.name) {
                continue; // Already reported as missing
            }
            self.check_column_type(df, col_schema, &mut report);
        }

        // Evaluate every constraint, regardless of whether its column(s)
        // are declared in `schema.columns`. Warn (rather than silently
        // skip) when a referenced column doesn't exist in the DataFrame at
        // all, since in that case the constraint genuinely cannot be
        // evaluated one way or the other.
        for constraint in &schema.constraints {
            for col in constraint.affected_columns() {
                if !df_columns.contains(col) {
                    report.add_warning(format!(
                        "Constraint {} references column '{}', which is not present in \
                         the DataFrame; it cannot be evaluated",
                        constraint, col
                    ));
                }
            }
            self.check_constraint(df, constraint, &mut report);
        }

        Ok(report)
    }

    fn check_column_type(
        &self,
        df: &DataFrame,
        col_schema: &ColumnSchema,
        report: &mut ValidationReport,
    ) {
        // `is_numeric_column` only recognises i64/f64/i32/f32; a genuine
        // `Series<bool>` column is checked separately so a `Boolean`-typed
        // schema column can actually validate against real boolean data
        // (previously, routing Boolean through the numeric check meant a
        // `Series<bool>` column -- which `is_numeric_column` does not
        // recognise -- could never pass validation).
        let is_numeric = df.is_numeric_column(&col_schema.name);
        let is_bool = df.get_column::<bool>(&col_schema.name).is_ok();

        let type_ok = match &col_schema.data_type {
            SchemaDataType::Int64 | SchemaDataType::Float64 => is_numeric,
            SchemaDataType::Boolean => is_bool,
            SchemaDataType::String => !is_numeric && !is_bool,
            // DateTime, Categorical, List — stored as strings; can't distinguish further
            SchemaDataType::DateTime
            | SchemaDataType::Categorical { .. }
            | SchemaDataType::List { .. } => true,
        };

        if !type_ok {
            report.add_error(
                &col_schema.name,
                format!(
                    "Column '{}' expected type {} but actual type does not match",
                    col_schema.name, col_schema.data_type
                ),
                ValidationErrorType::TypeMismatch,
            );
        }
    }

    fn check_constraint(
        &self,
        df: &DataFrame,
        constraint: &SchemaConstraint,
        report: &mut ValidationReport,
    ) {
        match constraint {
            SchemaConstraint::NotNull(col) => {
                if !df.contains_column(col) {
                    return;
                }
                if df.is_numeric_column(col) {
                    // Numeric columns (i64/f64/i32/f32) represent a missing
                    // value as `f64::NAN` -- there is no null bit in this
                    // DataFrame's column model. `get_column_string_values`
                    // renders NaN as the literal (non-empty!) string "NaN",
                    // so an emptiness check on the stringified column can
                    // never catch a numeric NA; inspect the numeric values
                    // directly instead.
                    if let Ok(values) = df.get_column_numeric_values(col) {
                        let null_count = values.iter().filter(|v| v.is_nan()).count();
                        if null_count > 0 {
                            report.add_error(
                                col,
                                format!(
                                    "Column '{}' has {} NaN value(s), violating NOT NULL constraint",
                                    col, null_count
                                ),
                                ValidationErrorType::NullViolation,
                            );
                        }
                    }
                } else if let Ok(values) = df.get_column_string_values(col) {
                    // String (and other non-numeric) columns have no
                    // dedicated null marker either; empty string is the
                    // best-effort "no value" convention already used
                    // elsewhere in this DataFrame model.
                    let null_count = values.iter().filter(|v| v.is_empty()).count();
                    if null_count > 0 {
                        report.add_error(
                            col,
                            format!(
                                "Column '{}' has {} null/empty value(s), violating NOT NULL constraint",
                                col, null_count
                            ),
                            ValidationErrorType::NullViolation,
                        );
                    }
                }
            }
            SchemaConstraint::Range { col, min, max } => {
                if df.contains_column(col) && df.is_numeric_column(col) {
                    if let Ok(values) = df.get_column_numeric_values(col) {
                        for &v in &values {
                            if let Some(min_val) = min {
                                if v < *min_val {
                                    report.add_error(
                                        col,
                                        format!(
                                            "Column '{}' has value {} below minimum {}",
                                            col, v, min_val
                                        ),
                                        ValidationErrorType::RangeViolation,
                                    );
                                    break;
                                }
                            }
                            if let Some(max_val) = max {
                                if v > *max_val {
                                    report.add_error(
                                        col,
                                        format!(
                                            "Column '{}' has value {} above maximum {}",
                                            col, v, max_val
                                        ),
                                        ValidationErrorType::RangeViolation,
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            SchemaConstraint::Unique(cols) => {
                // Uniqueness over the combination of all listed columns
                // (single-column is just the cols.len() == 1 case of this).
                // Values are rendered to strings so columns of any
                // supported concrete type can be combined into one
                // composite key.
                if cols.is_empty() {
                    report.add_warning("Unique constraint has no columns specified".to_string());
                    return;
                }
                if !cols.iter().all(|c| df.contains_column(c)) {
                    return; // already reported by the top-level "unevaluatable" warning
                }

                let mut per_column_values: Vec<Vec<String>> = Vec::with_capacity(cols.len());
                for col in cols {
                    match df.get_column_string_values(col) {
                        Ok(values) => per_column_values.push(values),
                        Err(_) => return, // not evaluable for this column's type
                    }
                }
                let row_count = per_column_values[0].len();
                let mut seen: std::collections::HashSet<Vec<&str>> =
                    std::collections::HashSet::new();
                for row in 0..row_count {
                    let key: Vec<&str> = per_column_values
                        .iter()
                        .map(|col_values| col_values[row].as_str())
                        .collect();
                    if !seen.insert(key.clone()) {
                        report.add_error(
                            cols.join(", "),
                            format!(
                                "Columns ({}) have duplicate combination at row {}: {:?}",
                                cols.join(", "),
                                row,
                                key
                            ),
                            ValidationErrorType::UniqueViolation,
                        );
                        break;
                    }
                }
            }
            SchemaConstraint::Regex { col, pattern } => {
                if df.contains_column(col) {
                    match regex::Regex::new(pattern) {
                        Ok(re) => {
                            if let Ok(values) = df.get_column_string_values(col) {
                                for v in &values {
                                    if !v.is_empty() && !re.is_match(v) {
                                        report.add_error(
                                            col,
                                            format!(
                                                "Column '{}' value '{}' does not match pattern '{}'",
                                                col, v, pattern
                                            ),
                                            ValidationErrorType::RegexViolation,
                                        );
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            report.add_warning(format!(
                                "Invalid regex pattern '{}' for column '{}': {}",
                                pattern, col, e
                            ));
                        }
                    }
                }
            }
            SchemaConstraint::Enum {
                col,
                values: allowed,
            } => {
                if df.contains_column(col) {
                    if let Ok(values) = df.get_column_string_values(col) {
                        for v in &values {
                            if !v.is_empty() && !allowed.contains(v) {
                                report.add_error(
                                    col,
                                    format!(
                                        "Column '{}' value '{}' is not in allowed values {:?}",
                                        col, v, allowed
                                    ),
                                    ValidationErrorType::EnumViolation,
                                );
                                break;
                            }
                        }
                    }
                }
            }
            // ForeignKey constraints require cross-DataFrame data — emit warning
            SchemaConstraint::ForeignKey {
                col,
                ref_schema,
                ref_col,
            } => {
                report.add_warning(format!(
                    "ForeignKey constraint on '{}' referencing '{}.{}' cannot be validated without the referenced schema's data",
                    col, ref_schema, ref_col
                ));
            }
        }
    }

    /// Infer a schema from an existing DataFrame.
    ///
    /// Uses heuristics to determine column types based on the DataFrame's actual data.
    pub fn infer_schema(&self, df: &DataFrame, name: &str) -> DataFrameSchema {
        let mut schema = DataFrameSchema::new(name, SchemaVersion::initial());

        for col_name in df.column_names() {
            let data_type = if df.get_column::<bool>(&col_name).is_ok() {
                SchemaDataType::Boolean
            } else if df.is_numeric_column(&col_name) {
                // Try to determine if it's int or float
                if let Ok(values) = df.get_column_numeric_values(&col_name) {
                    let all_int = values.iter().all(|v| v.fract() == 0.0 && v.is_finite());
                    if all_int {
                        SchemaDataType::Int64
                    } else {
                        SchemaDataType::Float64
                    }
                } else {
                    SchemaDataType::Float64
                }
            } else {
                SchemaDataType::String
            };

            let col_schema = ColumnSchema::new(col_name.clone(), data_type);
            schema = schema.with_column(col_schema);
        }

        schema
            .with_metadata("inferred_from", "DataFrame")
            .with_metadata("row_count", df.row_count().to_string())
    }

    /// Check if two schemas are compatible (data can flow from `from` to `to`).
    ///
    /// A schema is compatible if:
    /// - All required (non-nullable without default) columns in `to` exist in `from`
    /// - The types of common columns are compatible (castable)
    pub fn check_compatibility(
        &self,
        from: &DataFrameSchema,
        to: &DataFrameSchema,
    ) -> CompatibilityReport {
        let mut report = CompatibilityReport::new();

        let from_columns: HashMap<&str, &ColumnSchema> =
            from.columns.iter().map(|c| (c.name.as_str(), c)).collect();

        for to_col in &to.columns {
            match from_columns.get(to_col.name.as_str()) {
                None => {
                    // Column in `to` but not in `from`
                    if !to_col.nullable && to_col.default_value.is_none() {
                        report.add_breaking(
                            format!(
                                "Column '{}' is required in target schema but does not exist in source",
                                to_col.name
                            ),
                            vec![to_col.name.clone()],
                        );
                    } else {
                        report.add_non_breaking(format!(
                            "Column '{}' will be added with default/null value",
                            to_col.name
                        ));
                    }
                }
                Some(from_col) => {
                    // Column exists in both - check type compatibility
                    if from_col.data_type != to_col.data_type {
                        if from_col.data_type.can_cast_to(&to_col.data_type) {
                            report.add_non_breaking(format!(
                                "Column '{}' will be cast from {} to {}",
                                to_col.name, from_col.data_type, to_col.data_type
                            ));
                        } else {
                            report.add_breaking(
                                format!(
                                    "Column '{}' cannot be cast from {} to {}",
                                    to_col.name, from_col.data_type, to_col.data_type
                                ),
                                vec![to_col.name.clone()],
                            );
                        }
                    }
                    // Nullability tightening is a breaking change
                    if from_col.nullable && !to_col.nullable && to_col.default_value.is_none() {
                        report.add_breaking(
                            format!(
                                "Column '{}' becomes non-nullable in target schema (possible null violation)",
                                to_col.name
                            ),
                            vec![to_col.name.clone()],
                        );
                    }
                }
            }
        }

        // Columns in `from` but not in `to`: this never blocks the flow (the
        // target doesn't need that data), so it's still noted among
        // `non_breaking_changes` for a full change list -- but it *is* real
        // data loss, so it's also recorded in `data_loss` rather than left
        // indistinguishable from a zero-impact change like a widening cast.
        let to_column_names: std::collections::HashSet<&str> =
            to.columns.iter().map(|c| c.name.as_str()).collect();
        for from_col in &from.columns {
            if !to_column_names.contains(from_col.name.as_str()) {
                let msg = format!(
                    "Column '{}' exists in source but not in target schema (data will be dropped)",
                    from_col.name
                );
                report.add_non_breaking(msg.clone());
                report.data_loss.push(msg);
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema_evolution::evolution::MigrationBuilder;
    use crate::schema_evolution::schema::{
        ColumnSchema, DataFrameSchema, SchemaConstraint, SchemaDataType, SchemaVersion,
    };
    use crate::Series;

    fn make_test_df() -> DataFrame {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1i64, 2, 3], Some("id".to_string())).expect("series"),
        )
        .expect("add");
        df.add_column(
            "name".to_string(),
            Series::new(
                vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()],
                Some("name".to_string()),
            )
            .expect("series"),
        )
        .expect("add");
        df
    }

    fn make_test_schema() -> DataFrameSchema {
        DataFrameSchema::new("test", SchemaVersion::initial())
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64).with_nullable(false))
            .with_column(ColumnSchema::new("name", SchemaDataType::String))
    }

    #[test]
    fn test_apply_add_column() {
        let df = make_test_df();
        let migrator = SchemaMigrator::empty();
        let change = SchemaChange::AddColumn {
            schema: ColumnSchema::new("score", SchemaDataType::Float64)
                .with_default(DefaultValue::Float(0.0)),
            position: None,
        };
        let result = migrator.apply_change(&df, &change).expect("apply");
        assert!(result.contains_column("score"));
        assert_eq!(result.row_count(), 3);
    }

    #[test]
    fn test_apply_remove_column() {
        let df = make_test_df();
        let migrator = SchemaMigrator::empty();
        let change = SchemaChange::RemoveColumn {
            name: "name".to_string(),
        };
        let result = migrator.apply_change(&df, &change).expect("apply");
        assert!(!result.contains_column("name"));
        assert!(result.contains_column("id"));
    }

    #[test]
    fn test_apply_rename_column() {
        let df = make_test_df();
        let migrator = SchemaMigrator::empty();
        let change = SchemaChange::RenameColumn {
            from: "name".to_string(),
            to: "full_name".to_string(),
        };
        let result = migrator.apply_change(&df, &change).expect("apply");
        assert!(result.contains_column("full_name"));
        assert!(!result.contains_column("name"));
    }

    #[test]
    fn test_apply_change_type_to_string() {
        let df = make_test_df();
        let migrator = SchemaMigrator::empty();
        let change = SchemaChange::ChangeType {
            column: "id".to_string(),
            new_type: SchemaDataType::String,
            converter: None,
        };
        let result = migrator.apply_change(&df, &change).expect("apply");
        assert!(result.contains_column("id"));
        // Should now be a string column
        assert!(!result.is_numeric_column("id"));
    }

    #[test]
    fn test_validate_valid() {
        let df = make_test_df();
        let schema = make_test_schema();
        let migrator = SchemaMigrator::empty();
        let report = migrator.validate(&df, &schema).expect("validate");
        assert!(report.is_valid);
    }

    #[test]
    fn test_validate_missing_column() {
        let mut df = DataFrame::new();
        df.add_column(
            "id".to_string(),
            Series::new(vec![1i64, 2, 3], Some("id".to_string())).expect("series"),
        )
        .expect("add");
        // name is missing
        let schema = make_test_schema();
        let migrator = SchemaMigrator::empty();
        let report = migrator.validate(&df, &schema).expect("validate");
        assert!(!report.is_valid);
        assert!(report
            .errors
            .iter()
            .any(|e| e.error_type == ValidationErrorType::MissingColumn));
    }

    #[test]
    fn test_validate_range_constraint() {
        let mut df = DataFrame::new();
        df.add_column(
            "age".to_string(),
            Series::new(vec![25.0f64, 150.0, 30.0], Some("age".to_string())).expect("series"),
        )
        .expect("add");

        let schema = DataFrameSchema::new("test", SchemaVersion::initial())
            .with_column(ColumnSchema::new("age", SchemaDataType::Float64))
            .with_constraint(SchemaConstraint::Range {
                col: "age".to_string(),
                min: Some(0.0),
                max: Some(120.0),
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
    fn test_infer_schema() {
        let df = make_test_df();
        let migrator = SchemaMigrator::empty();
        let schema = migrator.infer_schema(&df, "inferred");
        assert_eq!(schema.name, "inferred");
        assert!(schema.has_column("id"));
        assert!(schema.has_column("name"));
        let id_col = schema.get_column("id").expect("id col");
        assert_eq!(id_col.data_type, SchemaDataType::Int64);
    }

    #[test]
    fn test_check_compatibility_compatible() {
        let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
            .with_column(ColumnSchema::new("name", SchemaDataType::String));

        let to = DataFrameSchema::new("v2", SchemaVersion::new(1, 1, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
            .with_column(ColumnSchema::new("name", SchemaDataType::String))
            .with_column(ColumnSchema::new("email", SchemaDataType::String).with_nullable(true));

        let migrator = SchemaMigrator::empty();
        let report = migrator.check_compatibility(&from, &to);
        assert!(report.is_compatible);
    }

    #[test]
    fn test_check_compatibility_breaking() {
        let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64));

        let to = DataFrameSchema::new("v2", SchemaVersion::new(2, 0, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
            .with_column(
                ColumnSchema::new("required_field", SchemaDataType::String).with_nullable(false),
            );

        let migrator = SchemaMigrator::empty();
        let report = migrator.check_compatibility(&from, &to);
        assert!(!report.is_compatible);
    }

    #[test]
    fn test_apply_migration() {
        let df = make_test_df();
        let migration = MigrationBuilder::new(
            "m001",
            "test",
            SchemaVersion::new(1, 0, 0),
            SchemaVersion::new(1, 1, 0),
        )
        .add_column(
            ColumnSchema::new("active", SchemaDataType::Boolean)
                .with_default(DefaultValue::Bool(true)),
            None,
        )
        .rename_column("name", "full_name")
        .build();

        let migrator = SchemaMigrator::empty();
        let result = migrator.apply_migration(&df, &migration).expect("migrate");
        assert!(result.contains_column("active"));
        assert!(result.contains_column("full_name"));
        assert!(!result.contains_column("name"));
    }

    #[test]
    fn test_validate_regex_constraint() {
        let mut df = DataFrame::new();
        df.add_column(
            "email".to_string(),
            Series::new(
                vec!["valid@test.com".to_string(), "invalid-email".to_string()],
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
    fn test_validate_enum_constraint() {
        let mut df = DataFrame::new();
        df.add_column(
            "status".to_string(),
            Series::new(
                vec![
                    "active".to_string(),
                    "banned".to_string(),
                    "pending".to_string(),
                ],
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
    fn test_copy_columns_preserves_i64_and_bool_types() {
        let mut df = DataFrame::new();
        df.add_column(
            "big_id".to_string(),
            // Above 2^53: would lose precision if routed through f64.
            Series::new(
                vec![9_007_199_254_740_993i64, -1],
                Some("big_id".to_string()),
            )
            .expect("series"),
        )
        .expect("add");
        df.add_column(
            "flag".to_string(),
            Series::new(vec![true, false], Some("flag".to_string())).expect("series"),
        )
        .expect("add");

        let migrator = SchemaMigrator::empty();
        // Reordering exercises `copy_columns` directly.
        let change = SchemaChange::ReorderColumns {
            order: vec!["flag".to_string(), "big_id".to_string()],
        };
        let result = migrator.apply_change(&df, &change).expect("reorder");

        let ids = result.get_column::<i64>("big_id").expect("still i64");
        assert_eq!(ids.values(), &[9_007_199_254_740_993i64, -1]);

        let flags = result.get_column::<bool>("flag").expect("still bool");
        assert_eq!(flags.values(), &[true, false]);
    }

    #[test]
    fn test_change_type_to_boolean_and_back_round_trips_real_bool() {
        let mut df = DataFrame::new();
        df.add_column(
            "active".to_string(),
            Series::new(
                vec!["true".to_string(), "false".to_string(), "1".to_string()],
                Some("active".to_string()),
            )
            .expect("series"),
        )
        .expect("add");

        let migrator = SchemaMigrator::empty();
        let change = SchemaChange::ChangeType {
            column: "active".to_string(),
            new_type: SchemaDataType::Boolean,
            converter: None,
        };
        let result = migrator.apply_change(&df, &change).expect("apply");

        // Must be a genuine Series<bool>, not f64 0.0/1.0.
        let values = result
            .get_column::<bool>("active")
            .expect("real bool column");
        assert_eq!(values.values(), &[true, false, true]);

        // And it must validate as Boolean, closing the migrate-then-validate
        // round trip that a fake f64-backed "boolean" column would fail.
        let schema = DataFrameSchema::new("test", SchemaVersion::initial())
            .with_column(ColumnSchema::new("active", SchemaDataType::Boolean));
        let report = migrator.validate(&result, &schema).expect("validate");
        assert!(report.is_valid, "errors: {:?}", report.errors);
    }

    #[test]
    fn test_not_null_catches_numeric_nan() {
        let mut df = DataFrame::new();
        df.add_column(
            "score".to_string(),
            Series::new(vec![1.0f64, f64::NAN, 3.0], Some("score".to_string())).expect("series"),
        )
        .expect("add");

        // Case 1: the constraint's column IS declared in schema.columns.
        let schema = DataFrameSchema::new("test", SchemaVersion::initial())
            .with_column(ColumnSchema::new("score", SchemaDataType::Float64))
            .with_constraint(SchemaConstraint::NotNull("score".to_string()));

        let migrator = SchemaMigrator::empty();
        let report = migrator.validate(&df, &schema).expect("validate");
        assert!(!report.is_valid);
        assert!(report
            .errors
            .iter()
            .any(|e| e.error_type == ValidationErrorType::NullViolation));

        // Case 2: the constraint's column is present in the DataFrame but
        // NOT declared in schema.columns at all -- only reachable because
        // `validate()` now iterates `schema.constraints` directly instead
        // of filtering through the declared-columns loop.
        let undeclared_schema = DataFrameSchema::new("test", SchemaVersion::initial())
            .with_constraint(SchemaConstraint::NotNull("score".to_string()));
        let report2 = migrator
            .validate(&df, &undeclared_schema)
            .expect("validate");
        assert!(!report2.is_valid);
        assert!(report2
            .errors
            .iter()
            .any(|e| e.error_type == ValidationErrorType::NullViolation));
    }

    #[test]
    fn test_composite_unique_constraint() {
        let mut df = DataFrame::new();
        df.add_column(
            "region".to_string(),
            Series::new(
                vec!["us".to_string(), "us".to_string(), "eu".to_string()],
                Some("region".to_string()),
            )
            .expect("series"),
        )
        .expect("add");
        df.add_column(
            "code".to_string(),
            Series::new(
                vec!["A".to_string(), "A".to_string(), "A".to_string()],
                Some("code".to_string()),
            )
            .expect("series"),
        )
        .expect("add");

        // ("us","A") appears twice -> violation; ("eu","A") is unique.
        let schema = DataFrameSchema::new("test", SchemaVersion::initial())
            .with_column(ColumnSchema::new("region", SchemaDataType::String))
            .with_column(ColumnSchema::new("code", SchemaDataType::String))
            .with_constraint(SchemaConstraint::Unique(vec![
                "region".to_string(),
                "code".to_string(),
            ]));

        let migrator = SchemaMigrator::empty();
        let report = migrator.validate(&df, &schema).expect("validate");
        assert!(!report.is_valid);
        assert!(report
            .errors
            .iter()
            .any(|e| e.error_type == ValidationErrorType::UniqueViolation));
    }

    #[test]
    fn test_nullable_add_column_uses_real_na_not_zero() {
        let df = make_test_df();
        let migrator = SchemaMigrator::empty();

        // Float64, nullable, no default -> NaN (a real NA), not a fabricated 0.0.
        let change = SchemaChange::AddColumn {
            schema: ColumnSchema::new("score", SchemaDataType::Float64).with_nullable(true),
            position: None,
        };
        let result = migrator.apply_change(&df, &change).expect("apply");
        let values = result.get_column_numeric_values("score").expect("numeric");
        assert!(
            values.iter().all(|v| v.is_nan()),
            "expected all-NaN, got {:?}",
            values
        );

        // Non-nullable, no default -> a real (non-NA) zero of the declared type.
        let change2 = SchemaChange::AddColumn {
            schema: ColumnSchema::new("count", SchemaDataType::Int64).with_nullable(false),
            position: None,
        };
        let result2 = migrator.apply_change(&df, &change2).expect("apply");
        let ids = result2.get_column::<i64>("count").expect("real i64 column");
        assert_eq!(ids.values(), &[0i64, 0, 0]);
    }

    #[test]
    fn test_check_compatibility_reports_data_loss_separately() {
        let from = DataFrameSchema::new("v1", SchemaVersion::new(1, 0, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64))
            .with_column(ColumnSchema::new("legacy", SchemaDataType::String));

        let to = DataFrameSchema::new("v2", SchemaVersion::new(2, 0, 0))
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64));

        let migrator = SchemaMigrator::empty();
        let report = migrator.check_compatibility(&from, &to);
        // Still compatible: the flow doesn't need the dropped column.
        assert!(report.is_compatible);
        // But the drop is honestly recorded as data loss, not just folded
        // into the generic non-breaking-changes list.
        assert!(report.data_loss.iter().any(|c| c.contains("legacy")));
    }

    #[test]
    fn test_unsupported_column_type_errors_instead_of_fabricating() {
        let mut df = DataFrame::new();
        df.add_column(
            "when".to_string(),
            Series::new(
                vec![chrono::NaiveDate::from_ymd_opt(2024, 1, 1).expect("date")],
                Some("when".to_string()),
            )
            .expect("series"),
        )
        .expect("add");

        let migrator = SchemaMigrator::empty();
        let change = SchemaChange::ReorderColumns {
            order: vec!["when".to_string()],
        };
        let result = migrator.apply_change(&df, &change);
        assert!(
            result.is_err(),
            "an unsupported column type must error, not silently fabricate a placeholder"
        );
    }
}
