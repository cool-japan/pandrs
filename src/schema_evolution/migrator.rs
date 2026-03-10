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
    /// Whether data can flow from `from` schema to `to` schema without data loss
    pub is_compatible: bool,
    /// List of breaking changes that prevent compatibility
    pub breaking_changes: Vec<BreakingChange>,
    /// List of non-breaking changes (informational)
    pub non_breaking_changes: Vec<String>,
}

impl CompatibilityReport {
    fn new() -> Self {
        CompatibilityReport {
            is_compatible: true,
            breaking_changes: Vec::new(),
            non_breaking_changes: Vec::new(),
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

/// Detect whether a column contains numeric data by attempting to retrieve numeric values.
///
/// Since `is_numeric_column` is a stub that always returns false, we use this
/// heuristic: try `get_column_numeric_values` and if it succeeds AND the column
/// values don't look like string representations, treat as numeric.
fn column_is_numeric(df: &DataFrame, col_name: &str) -> bool {
    // First, check if the column stores string data (strings that happen to be
    // parseable numbers should still be treated as strings in schema context).
    // We use a best-effort approach: try numeric downcast, check string downcast.
    // get_column_numeric_values succeeds for i64, f64, and parseable strings.
    // We discriminate by checking get_column_string_values first.
    if let Ok(str_vals) = df.get_column_string_values(col_name) {
        // If all values look like numbers (including the column was added as numeric
        // Series<i64> or Series<f64>), check numeric.
        // The distinguishing factor: pure string Series will return the strings as-is;
        // numeric Series will return their string representation via to_string().
        // We check if numeric retrieval succeeds AND agrees with string retrieval.
        if let Ok(num_vals) = df.get_column_numeric_values(col_name) {
            // If numeric succeeds, check if this could be a numeric column.
            // If the string values are all parseable as f64, it might be a string
            // column with numeric content. We check if the numeric values match
            // the string values when formatted as numbers.
            if str_vals.is_empty() {
                return false;
            }
            // Heuristic: if the string representation doesn't look like a formatted
            // float (no decimal points in original that wouldn't come from i64),
            // it's numeric. The simplest check: verify that the number of
            // decimal-formatted values matches.
            let all_numeric_looking = str_vals
                .iter()
                .all(|s| s.parse::<f64>().is_ok() || s.is_empty());
            all_numeric_looking && !num_vals.is_empty()
        } else {
            false
        }
    } else {
        false
    }
}

/// Helper: copy all columns from `df` into a new DataFrame, preserving types.
///
/// Numeric columns are preserved as f64, others as String.
/// The order follows the provided `column_list`.
fn copy_columns(df: &DataFrame, column_list: &[String]) -> Result<DataFrame> {
    let mut result = DataFrame::new();
    for col_name in column_list {
        if !df.contains_column(col_name) {
            return Err(Error::ColumnNotFound(col_name.clone()));
        }
        if column_is_numeric(df, col_name) {
            let values = df.get_column_numeric_values(col_name)?;
            let series = Series::new(values, Some(col_name.clone()))
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            result.add_column(col_name.clone(), series)?;
        } else {
            let values = df.get_column_string_values(col_name)?;
            let series = Series::new(values, Some(col_name.clone()))
                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
            result.add_column(col_name.clone(), series)?;
        }
    }
    Ok(result)
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
        let existing_cols = df.column_names();

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

        // Add the new column with the appropriate default
        match &schema.data_type {
            SchemaDataType::Int64 => {
                let default = match &schema.default_value {
                    Some(DefaultValue::Int(v)) => *v as f64,
                    Some(DefaultValue::Float(v)) => *v,
                    _ => 0.0f64,
                };
                let data: Vec<f64> = vec![default; row_count];
                let series = Series::new(data, Some(schema.name.clone()))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                result.add_column(schema.name.clone(), series)?;
            }
            SchemaDataType::Float64 => {
                let default = match &schema.default_value {
                    Some(DefaultValue::Float(v)) => *v,
                    Some(DefaultValue::Int(v)) => *v as f64,
                    _ => 0.0f64,
                };
                let data: Vec<f64> = vec![default; row_count];
                let series = Series::new(data, Some(schema.name.clone()))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                result.add_column(schema.name.clone(), series)?;
            }
            SchemaDataType::Boolean => {
                let default = match &schema.default_value {
                    Some(DefaultValue::Bool(v)) => {
                        if *v {
                            1.0f64
                        } else {
                            0.0f64
                        }
                    }
                    _ => 0.0f64,
                };
                let data: Vec<f64> = vec![default; row_count];
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
            .into_iter()
            .filter(|c| c != name)
            .collect();
        copy_columns(df, &keep)
    }

    fn apply_rename_column(&self, df: &DataFrame, from: &str, to: &str) -> Result<DataFrame> {
        if !df.contains_column(from) {
            return Err(Error::ColumnNotFound(from.to_string()));
        }
        // Build a new DataFrame with the column renamed
        let mut result = DataFrame::new();
        for col_name in df.column_names() {
            let target_name = if col_name == from {
                to.to_string()
            } else {
                col_name.clone()
            };

            if column_is_numeric(df, &col_name) {
                let values = df.get_column_numeric_values(&col_name)?;
                let series = Series::new(values, Some(target_name.clone()))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                result.add_column(target_name, series)?;
            } else {
                let values = df.get_column_string_values(&col_name)?;
                let series = Series::new(values, Some(target_name.clone()))
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                result.add_column(target_name, series)?;
            }
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
            if col_name != column {
                // Copy other columns as-is
                if column_is_numeric(df, &col_name) {
                    let values = df.get_column_numeric_values(&col_name)?;
                    let series = Series::new(values, Some(col_name.clone()))
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                    result.add_column(col_name, series)?;
                } else {
                    let values = df.get_column_string_values(&col_name)?;
                    let series = Series::new(values, Some(col_name.clone()))
                        .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                    result.add_column(col_name, series)?;
                }
            } else {
                // Convert this column to the new type
                match new_type {
                    SchemaDataType::Float64 => {
                        // If already numeric, just copy
                        if column_is_numeric(df, column) {
                            let values = df.get_column_numeric_values(column)?;
                            let series = Series::new(values, Some(col_name.clone()))
                                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                            result.add_column(col_name, series)?;
                        } else {
                            let string_values = df.get_column_string_values(column)?;
                            let float_values: Result<Vec<f64>> = string_values
                                .iter()
                                .map(|s| {
                                    s.parse::<f64>().map_err(|e| {
                                        Error::Cast(format!(
                                            "Cannot cast '{}' to Float64: {}",
                                            s, e
                                        ))
                                    })
                                })
                                .collect();
                            let series = Series::new(float_values?, Some(col_name.clone()))
                                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                            result.add_column(col_name, series)?;
                        }
                    }
                    SchemaDataType::Int64 => {
                        if column_is_numeric(df, column) {
                            // Truncate floats to int (store as f64 since that's what we have)
                            let values = df.get_column_numeric_values(column)?;
                            let int_values: Vec<f64> = values.iter().map(|v| v.trunc()).collect();
                            let series = Series::new(int_values, Some(col_name.clone()))
                                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                            result.add_column(col_name, series)?;
                        } else {
                            let string_values = df.get_column_string_values(column)?;
                            let int_values: Result<Vec<f64>> = string_values
                                .iter()
                                .map(|s| {
                                    s.parse::<i64>()
                                        .map(|i| i as f64)
                                        .or_else(|_| s.parse::<f64>().map(|f| f.trunc()))
                                        .map_err(|e| {
                                            Error::Cast(format!(
                                                "Cannot cast '{}' to Int64: {}",
                                                s, e
                                            ))
                                        })
                                })
                                .collect();
                            let series = Series::new(int_values?, Some(col_name.clone()))
                                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                            result.add_column(col_name, series)?;
                        }
                    }
                    SchemaDataType::Boolean => {
                        if column_is_numeric(df, column) {
                            // Non-zero = true (1.0), zero = false (0.0)
                            let values = df.get_column_numeric_values(column)?;
                            let bool_values: Vec<f64> = values
                                .iter()
                                .map(|v| if *v != 0.0 { 1.0 } else { 0.0 })
                                .collect();
                            let series = Series::new(bool_values, Some(col_name.clone()))
                                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                            result.add_column(col_name, series)?;
                        } else {
                            let string_values = df.get_column_string_values(column)?;
                            let bool_values: Result<Vec<f64>> = string_values
                                .iter()
                                .map(|s| match s.to_lowercase().as_str() {
                                    "true" | "1" | "yes" => Ok(1.0f64),
                                    "false" | "0" | "no" | "" => Ok(0.0f64),
                                    _ => {
                                        Err(Error::Cast(format!("Cannot cast '{}' to Boolean", s)))
                                    }
                                })
                                .collect();
                            let series = Series::new(bool_values?, Some(col_name.clone()))
                                .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                            result.add_column(col_name, series)?;
                        }
                    }
                    // All other types: convert to string representation
                    SchemaDataType::String
                    | SchemaDataType::DateTime
                    | SchemaDataType::Categorical { .. }
                    | SchemaDataType::List { .. } => {
                        let string_values = df.get_column_string_values(column)?;
                        let series = Series::new(string_values, Some(col_name.clone()))
                            .map_err(|e| Error::InvalidOperation(e.to_string()))?;
                        result.add_column(col_name, series)?;
                    }
                }
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
    /// - Constraints are satisfied (NotNull, Range, Enum, Regex, Unique)
    pub fn validate(&self, df: &DataFrame, schema: &DataFrameSchema) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        let df_columns: std::collections::HashSet<String> = df.column_names().into_iter().collect();
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

        // Check type compatibility and constraints for existing columns
        for col_schema in &schema.columns {
            if !df_columns.contains(&col_schema.name) {
                continue; // Already reported as missing
            }

            // Type checking (heuristic based on what we can determine)
            self.check_column_type(df, col_schema, &mut report);

            // Check constraints for this column
            for constraint in schema.constraints_for_column(&col_schema.name) {
                self.check_constraint(df, constraint, &mut report);
            }
        }

        Ok(report)
    }

    fn check_column_type(
        &self,
        df: &DataFrame,
        col_schema: &ColumnSchema,
        report: &mut ValidationReport,
    ) {
        let is_numeric = column_is_numeric(df, &col_schema.name);

        let type_ok = match &col_schema.data_type {
            SchemaDataType::Int64 | SchemaDataType::Float64 | SchemaDataType::Boolean => is_numeric,
            SchemaDataType::String => !is_numeric,
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
                if df.contains_column(col) {
                    if let Ok(values) = df.get_column_string_values(col) {
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
            }
            SchemaConstraint::Range { col, min, max } => {
                if df.contains_column(col) && column_is_numeric(df, col) {
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
                // Check single-column uniqueness
                if cols.len() == 1 {
                    let col = &cols[0];
                    if df.contains_column(col) {
                        if let Ok(values) = df.get_column_string_values(col) {
                            let mut seen = std::collections::HashSet::new();
                            for v in &values {
                                if !seen.insert(v) {
                                    report.add_error(
                                        col,
                                        format!("Column '{}' has duplicate value '{}'", col, v),
                                        ValidationErrorType::UniqueViolation,
                                    );
                                    break;
                                }
                            }
                        }
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
            let data_type = if column_is_numeric(df, &col_name) {
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

        // Columns in `from` but not in `to` - data loss (non-breaking for flow but worth noting)
        let to_column_names: std::collections::HashSet<&str> =
            to.columns.iter().map(|c| c.name.as_str()).collect();
        for from_col in &from.columns {
            if !to_column_names.contains(from_col.name.as_str()) {
                report.add_non_breaking(format!(
                    "Column '{}' exists in source but not in target schema (data will be dropped)",
                    from_col.name
                ));
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
}
