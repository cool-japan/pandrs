//! Schema definition types for DataFrame schema evolution
//!
//! This module defines the core structures that describe the structure
//! of a DataFrame, including column types, constraints, and versioning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Semantic version for a schema
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaVersion {
    /// Major version - incremented for breaking changes
    pub major: u32,
    /// Minor version - incremented for backward-compatible additions
    pub minor: u32,
    /// Patch version - incremented for backward-compatible bug fixes
    pub patch: u32,
}

impl SchemaVersion {
    /// Create a new schema version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        SchemaVersion {
            major,
            minor,
            patch,
        }
    }

    /// Create initial version (1.0.0)
    pub fn initial() -> Self {
        SchemaVersion::new(1, 0, 0)
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &SchemaVersion) -> bool {
        self.major == other.major
    }

    /// Check if this version is greater than another
    pub fn is_greater_than(&self, other: &SchemaVersion) -> bool {
        if self.major != other.major {
            return self.major > other.major;
        }
        if self.minor != other.minor {
            return self.minor > other.minor;
        }
        self.patch > other.patch
    }

    /// Increment major version (resets minor and patch to 0)
    pub fn next_major(&self) -> Self {
        SchemaVersion::new(self.major + 1, 0, 0)
    }

    /// Increment minor version (resets patch to 0)
    pub fn next_minor(&self) -> Self {
        SchemaVersion::new(self.major, self.minor + 1, 0)
    }

    /// Increment patch version
    pub fn next_patch(&self) -> Self {
        SchemaVersion::new(self.major, self.minor, self.patch + 1)
    }
}

impl Default for SchemaVersion {
    fn default() -> Self {
        SchemaVersion::initial()
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for SchemaVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchemaVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.major.cmp(&other.major) {
            std::cmp::Ordering::Equal => match self.minor.cmp(&other.minor) {
                std::cmp::Ordering::Equal => self.patch.cmp(&other.patch),
                ord => ord,
            },
            ord => ord,
        }
    }
}

/// Data type for a schema column
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaDataType {
    /// 64-bit signed integer
    Int64,
    /// 64-bit floating point
    Float64,
    /// Boolean value
    Boolean,
    /// UTF-8 string
    String,
    /// Date and time with timezone
    DateTime,
    /// Categorical (enum-like) with a fixed set of string values
    Categorical {
        /// The allowed category values
        categories: Vec<std::string::String>,
    },
    /// Ordered list of elements
    List {
        /// The type of list elements
        element_type: Box<SchemaDataType>,
    },
}

impl SchemaDataType {
    /// Returns a human-readable name for the data type
    pub fn type_name(&self) -> &str {
        match self {
            SchemaDataType::Int64 => "Int64",
            SchemaDataType::Float64 => "Float64",
            SchemaDataType::Boolean => "Boolean",
            SchemaDataType::String => "String",
            SchemaDataType::DateTime => "DateTime",
            SchemaDataType::Categorical { .. } => "Categorical",
            SchemaDataType::List { .. } => "List",
        }
    }

    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, SchemaDataType::Int64 | SchemaDataType::Float64)
    }

    /// Check if this type can be cast to another type
    pub fn can_cast_to(&self, target: &SchemaDataType) -> bool {
        match (self, target) {
            // Same type always works
            (a, b) if a == b => true,
            // Numeric widening
            (SchemaDataType::Int64, SchemaDataType::Float64) => true,
            // Numeric to string
            (SchemaDataType::Int64, SchemaDataType::String) => true,
            (SchemaDataType::Float64, SchemaDataType::String) => true,
            (SchemaDataType::Boolean, SchemaDataType::String) => true,
            // String to numeric (may fail at runtime)
            (SchemaDataType::String, SchemaDataType::Int64) => true,
            (SchemaDataType::String, SchemaDataType::Float64) => true,
            (SchemaDataType::String, SchemaDataType::Boolean) => true,
            (SchemaDataType::String, SchemaDataType::DateTime) => true,
            // Categorical to string
            (SchemaDataType::Categorical { .. }, SchemaDataType::String) => true,
            // String to categorical
            (SchemaDataType::String, SchemaDataType::Categorical { .. }) => true,
            _ => false,
        }
    }
}

impl fmt::Display for SchemaDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaDataType::List { element_type } => write!(f, "List<{}>", element_type),
            SchemaDataType::Categorical { categories } => {
                write!(f, "Categorical({} categories)", categories.len())
            }
            other => write!(f, "{}", other.type_name()),
        }
    }
}

/// Default value for a column
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DefaultValue {
    /// Integer default
    Int(i64),
    /// Float default
    Float(f64),
    /// Boolean default
    Bool(bool),
    /// String default
    Str(std::string::String),
    /// NULL / missing value default
    Null,
    /// Current timestamp as default
    CurrentTimestamp,
}

impl fmt::Display for DefaultValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DefaultValue::Int(v) => write!(f, "{}", v),
            DefaultValue::Float(v) => write!(f, "{}", v),
            DefaultValue::Bool(v) => write!(f, "{}", v),
            DefaultValue::Str(v) => write!(f, "'{}'", v),
            DefaultValue::Null => write!(f, "NULL"),
            DefaultValue::CurrentTimestamp => write!(f, "CURRENT_TIMESTAMP"),
        }
    }
}

/// Schema constraint that must be satisfied by the data
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SchemaConstraint {
    /// Column must not contain null values
    NotNull(std::string::String),
    /// One or more columns must have unique values
    Unique(Vec<std::string::String>),
    /// Column values must be within a numeric range
    Range {
        /// Column name
        col: std::string::String,
        /// Inclusive minimum value (None = no lower bound)
        min: Option<f64>,
        /// Inclusive maximum value (None = no upper bound)
        max: Option<f64>,
    },
    /// Column values must match a regular expression
    Regex {
        /// Column name
        col: std::string::String,
        /// Regular expression pattern
        pattern: std::string::String,
    },
    /// Column values must reference values in another schema's column
    ForeignKey {
        /// Local column name
        col: std::string::String,
        /// Referenced schema name
        ref_schema: std::string::String,
        /// Referenced column name
        ref_col: std::string::String,
    },
    /// Column values must be one of the listed values
    Enum {
        /// Column name
        col: std::string::String,
        /// Allowed values
        values: Vec<std::string::String>,
    },
}

impl SchemaConstraint {
    /// Returns the constraint type name
    pub fn constraint_type(&self) -> &str {
        match self {
            SchemaConstraint::NotNull(_) => "NotNull",
            SchemaConstraint::Unique(_) => "Unique",
            SchemaConstraint::Range { .. } => "Range",
            SchemaConstraint::Regex { .. } => "Regex",
            SchemaConstraint::ForeignKey { .. } => "ForeignKey",
            SchemaConstraint::Enum { .. } => "Enum",
        }
    }

    /// Returns the columns affected by this constraint
    pub fn affected_columns(&self) -> Vec<&str> {
        match self {
            SchemaConstraint::NotNull(col) => vec![col.as_str()],
            SchemaConstraint::Unique(cols) => cols.iter().map(|s| s.as_str()).collect(),
            SchemaConstraint::Range { col, .. } => vec![col.as_str()],
            SchemaConstraint::Regex { col, .. } => vec![col.as_str()],
            SchemaConstraint::ForeignKey { col, .. } => vec![col.as_str()],
            SchemaConstraint::Enum { col, .. } => vec![col.as_str()],
        }
    }

    /// Generate a unique ID for this constraint
    pub fn generate_id(&self) -> std::string::String {
        match self {
            SchemaConstraint::NotNull(col) => format!("notnull_{}", col),
            SchemaConstraint::Unique(cols) => format!("unique_{}", cols.join("_")),
            SchemaConstraint::Range { col, min, max } => {
                format!("range_{}_{:?}_{:?}", col, min, max)
            }
            SchemaConstraint::Regex { col, pattern } => format!("regex_{}_{}", col, pattern),
            SchemaConstraint::ForeignKey {
                col,
                ref_schema,
                ref_col,
            } => {
                format!("fk_{}_{}_{}", col, ref_schema, ref_col)
            }
            SchemaConstraint::Enum { col, .. } => format!("enum_{}", col),
        }
    }
}

impl fmt::Display for SchemaConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaConstraint::NotNull(col) => write!(f, "NOT NULL({})", col),
            SchemaConstraint::Unique(cols) => write!(f, "UNIQUE({})", cols.join(", ")),
            SchemaConstraint::Range { col, min, max } => {
                write!(f, "RANGE({}: {:?}..{:?})", col, min, max)
            }
            SchemaConstraint::Regex { col, pattern } => {
                write!(f, "REGEX({}: {})", col, pattern)
            }
            SchemaConstraint::ForeignKey {
                col,
                ref_schema,
                ref_col,
            } => {
                write!(f, "FK({} -> {}.{})", col, ref_schema, ref_col)
            }
            SchemaConstraint::Enum { col, values } => {
                write!(f, "ENUM({}: {:?})", col, values)
            }
        }
    }
}

/// Definition of a single column in a schema
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// Column name
    pub name: std::string::String,
    /// Data type of the column
    pub data_type: SchemaDataType,
    /// Whether the column can contain null values
    pub nullable: bool,
    /// Default value when not specified
    pub default_value: Option<DefaultValue>,
    /// Human-readable description
    pub description: Option<std::string::String>,
    /// Tags for categorization
    pub tags: Vec<std::string::String>,
}

impl ColumnSchema {
    /// Create a new column schema with required fields
    pub fn new(name: impl Into<std::string::String>, data_type: SchemaDataType) -> Self {
        ColumnSchema {
            name: name.into(),
            data_type,
            nullable: true,
            default_value: None,
            description: None,
            tags: Vec::new(),
        }
    }

    /// Set nullability
    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Set default value
    pub fn with_default(mut self, default: DefaultValue) -> Self {
        self.default_value = Some(default);
        self
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<std::string::String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add a tag
    pub fn with_tag(mut self, tag: impl Into<std::string::String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

impl fmt::Display for ColumnSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}{}",
            self.name,
            self.data_type,
            if self.nullable {
                " (nullable)"
            } else {
                " (not null)"
            }
        )
    }
}

/// Complete schema definition for a DataFrame
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataFrameSchema {
    /// Schema name (identifies this schema in the registry)
    pub name: std::string::String,
    /// Semantic version of this schema
    pub version: SchemaVersion,
    /// Ordered list of column definitions
    pub columns: Vec<ColumnSchema>,
    /// Constraints that must be satisfied by data conforming to this schema
    pub constraints: Vec<SchemaConstraint>,
    /// Arbitrary key-value metadata
    pub metadata: HashMap<std::string::String, std::string::String>,
}

impl DataFrameSchema {
    /// Create a new empty schema
    pub fn new(name: impl Into<std::string::String>, version: SchemaVersion) -> Self {
        DataFrameSchema {
            name: name.into(),
            version,
            columns: Vec::new(),
            constraints: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a column to the schema
    pub fn with_column(mut self, column: ColumnSchema) -> Self {
        self.columns.push(column);
        self
    }

    /// Add a constraint to the schema
    pub fn with_constraint(mut self, constraint: SchemaConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add metadata
    pub fn with_metadata(
        mut self,
        key: impl Into<std::string::String>,
        value: impl Into<std::string::String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Find a column by name
    pub fn get_column(&self, name: &str) -> Option<&ColumnSchema> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Find a column's position by name
    pub fn column_position(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Check if a column exists
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.iter().any(|c| c.name == name)
    }

    /// Get all column names
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    /// Get constraints for a specific column
    pub fn constraints_for_column(&self, col_name: &str) -> Vec<&SchemaConstraint> {
        self.constraints
            .iter()
            .filter(|c| c.affected_columns().contains(&col_name))
            .collect()
    }
}

impl fmt::Display for DataFrameSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Schema: {} v{}", self.name, self.version)?;
        writeln!(f, "Columns ({}):", self.columns.len())?;
        for col in &self.columns {
            writeln!(f, "  - {}", col)?;
        }
        if !self.constraints.is_empty() {
            writeln!(f, "Constraints ({}):", self.constraints.len())?;
            for constraint in &self.constraints {
                writeln!(f, "  - {}", constraint)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version_ordering() {
        let v1 = SchemaVersion::new(1, 0, 0);
        let v2 = SchemaVersion::new(1, 1, 0);
        let v3 = SchemaVersion::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3);
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
    fn test_schema_version_display() {
        let v = SchemaVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_column_schema_builder() {
        let col = ColumnSchema::new("age", SchemaDataType::Int64)
            .with_nullable(false)
            .with_default(DefaultValue::Int(0))
            .with_description("User age in years")
            .with_tag("pii");
        assert_eq!(col.name, "age");
        assert!(!col.nullable);
        assert!(col.default_value.is_some());
        assert_eq!(col.tags, vec!["pii"]);
    }

    #[test]
    fn test_dataframe_schema_builder() {
        let schema = DataFrameSchema::new("users", SchemaVersion::initial())
            .with_column(ColumnSchema::new("id", SchemaDataType::Int64).with_nullable(false))
            .with_column(ColumnSchema::new("name", SchemaDataType::String))
            .with_constraint(SchemaConstraint::NotNull("id".to_string()));

        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.constraints.len(), 1);
        assert!(schema.has_column("id"));
        assert!(schema.has_column("name"));
        assert!(!schema.has_column("email"));
    }

    #[test]
    fn test_type_casting() {
        assert!(SchemaDataType::Int64.can_cast_to(&SchemaDataType::Float64));
        assert!(SchemaDataType::Int64.can_cast_to(&SchemaDataType::String));
        assert!(!SchemaDataType::Boolean.can_cast_to(&SchemaDataType::Int64));
    }
}
