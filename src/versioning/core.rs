//! Core data versioning and lineage tracking types
//!
//! This module provides the fundamental building blocks for tracking
//! DataFrame versions, operations, and data lineage.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};

/// Unique identifier for a data version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VersionId(pub String);

impl VersionId {
    /// Creates a new version ID
    pub fn new() -> Self {
        VersionId(uuid_v4())
    }

    /// Creates a version ID from a string
    pub fn from_str(s: &str) -> Self {
        VersionId(s.to_string())
    }

    /// Returns the inner string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for VersionId {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for VersionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Generate a simple UUID v4 with guaranteed uniqueness
fn uuid_v4() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Atomic counter ensures uniqueness even when called in the same nanosecond
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);

    // Combine timestamp, counter, and thread ID for uniqueness
    let thread_id = std::thread::current().id();
    let thread_hash = format!("{:?}", thread_id).len() as u64;

    let random_part: u64 = (timestamp as u64)
        .wrapping_add(counter)
        .wrapping_add(thread_hash.wrapping_mul(0x5851F42D4C957F2D));

    format!(
        "{:08x}-{:04x}-4{:03x}-{:04x}-{:012x}",
        (random_part >> 32) as u32,
        ((random_part >> 16) & 0xFFFF) as u16,
        (random_part & 0x0FFF) as u16,
        ((random_part >> 48) & 0x3FFF) as u16 | 0x8000,
        (random_part ^ counter) & 0xFFFFFFFFFFFF
    )
}

/// Types of operations that can be tracked
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    /// Data was created from scratch
    Create,
    /// Data was loaded from a source
    Load { source: String, format: String },
    /// Columns were selected
    Select { columns: Vec<String> },
    /// Rows were filtered
    Filter { condition: String },
    /// Data was sorted
    Sort {
        columns: Vec<String>,
        ascending: Vec<bool>,
    },
    /// Columns were added
    AddColumn { column_name: String },
    /// Columns were dropped
    DropColumn { columns: Vec<String> },
    /// Columns were renamed
    Rename { old_name: String, new_name: String },
    /// Data was aggregated
    Aggregate {
        group_by: Vec<String>,
        aggregations: Vec<String>,
    },
    /// DataFrames were joined
    Join {
        other_version: VersionId,
        join_type: String,
        on: Vec<String>,
    },
    /// DataFrames were concatenated
    Concat { other_versions: Vec<VersionId> },
    /// Values were filled or imputed
    FillNA { strategy: String },
    /// Data type was converted
    Cast { column: String, to_type: String },
    /// Custom transformation
    Transform { name: String, description: String },
    /// Data was saved
    Save { destination: String, format: String },
    /// Generic operation
    Custom {
        name: String,
        params: HashMap<String, String>,
    },
}

impl Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OperationType::Create => write!(f, "CREATE"),
            OperationType::Load { source, format } => write!(f, "LOAD({}, {})", source, format),
            OperationType::Select { columns } => write!(f, "SELECT({})", columns.join(", ")),
            OperationType::Filter { condition } => write!(f, "FILTER({})", condition),
            OperationType::Sort { columns, .. } => write!(f, "SORT({})", columns.join(", ")),
            OperationType::AddColumn { column_name } => write!(f, "ADD_COLUMN({})", column_name),
            OperationType::DropColumn { columns } => write!(f, "DROP({})", columns.join(", ")),
            OperationType::Rename { old_name, new_name } => {
                write!(f, "RENAME({} -> {})", old_name, new_name)
            }
            OperationType::Aggregate {
                group_by,
                aggregations,
            } => {
                write!(
                    f,
                    "AGGREGATE(BY: {}, AGG: {})",
                    group_by.join(", "),
                    aggregations.join(", ")
                )
            }
            OperationType::Join { join_type, on, .. } => {
                write!(f, "JOIN({}, ON: {})", join_type, on.join(", "))
            }
            OperationType::Concat { other_versions } => {
                write!(f, "CONCAT({} DataFrames)", other_versions.len())
            }
            OperationType::FillNA { strategy } => write!(f, "FILL_NA({})", strategy),
            OperationType::Cast { column, to_type } => {
                write!(f, "CAST({} -> {})", column, to_type)
            }
            OperationType::Transform { name, .. } => write!(f, "TRANSFORM({})", name),
            OperationType::Save {
                destination,
                format,
            } => {
                write!(f, "SAVE({}, {})", destination, format)
            }
            OperationType::Custom { name, .. } => write!(f, "CUSTOM({})", name),
        }
    }
}

/// Represents a single operation in the lineage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    /// Unique ID for this operation
    pub id: String,
    /// Type of operation
    pub operation_type: OperationType,
    /// When the operation occurred
    pub timestamp: DateTime<Utc>,
    /// Input version(s)
    pub inputs: Vec<VersionId>,
    /// Output version
    pub output: VersionId,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// User who performed the operation
    pub user: Option<String>,
    /// Duration of the operation in milliseconds
    pub duration_ms: Option<u64>,
}

impl Operation {
    /// Creates a new operation
    pub fn new(operation_type: OperationType, inputs: Vec<VersionId>, output: VersionId) -> Self {
        Operation {
            id: uuid_v4(),
            operation_type,
            timestamp: Utc::now(),
            inputs,
            output,
            metadata: HashMap::new(),
            user: None,
            duration_ms: None,
        }
    }

    /// Adds metadata to the operation
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Sets the user who performed the operation
    pub fn with_user(mut self, user: &str) -> Self {
        self.user = Some(user.to_string());
        self
    }

    /// Sets the duration of the operation
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }
}

/// Schema information for a DataFrame version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    /// Column names in order
    pub columns: Vec<String>,
    /// Column types
    pub types: HashMap<String, String>,
    /// Number of rows
    pub row_count: usize,
}

impl DataSchema {
    /// Creates a new schema
    pub fn new(columns: Vec<String>, types: HashMap<String, String>, row_count: usize) -> Self {
        DataSchema {
            columns,
            types,
            row_count,
        }
    }

    /// Checks if two schemas are compatible
    pub fn is_compatible(&self, other: &DataSchema) -> bool {
        // Same columns in the same order
        if self.columns != other.columns {
            return false;
        }

        // Same types
        for col in &self.columns {
            if self.types.get(col) != other.types.get(col) {
                return false;
            }
        }

        true
    }
}

/// Represents a version of data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataVersion {
    /// Unique identifier for this version
    pub id: VersionId,
    /// Human-readable name
    pub name: Option<String>,
    /// Description
    pub description: Option<String>,
    /// When this version was created
    pub created_at: DateTime<Utc>,
    /// Schema information
    pub schema: DataSchema,
    /// Parent version(s) this was derived from
    pub parents: Vec<VersionId>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Hash of the data for integrity checking
    pub data_hash: Option<String>,
    /// Size in bytes
    pub size_bytes: Option<usize>,
}

impl DataVersion {
    /// Creates a new data version
    pub fn new(schema: DataSchema) -> Self {
        DataVersion {
            id: VersionId::new(),
            name: None,
            description: None,
            created_at: Utc::now(),
            schema,
            parents: Vec::new(),
            tags: Vec::new(),
            metadata: HashMap::new(),
            data_hash: None,
            size_bytes: None,
        }
    }

    /// Sets the name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Sets the description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Sets the parents
    pub fn with_parents(mut self, parents: Vec<VersionId>) -> Self {
        self.parents = parents;
        self
    }

    /// Adds a tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Adds metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Represents a diff between two versions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Source version
    pub from_version: VersionId,
    /// Target version
    pub to_version: VersionId,
    /// Columns added
    pub columns_added: Vec<String>,
    /// Columns removed
    pub columns_removed: Vec<String>,
    /// Columns with changed types
    pub type_changes: HashMap<String, (String, String)>,
    /// Change in row count
    pub row_count_diff: i64,
    /// Operations between versions
    pub operations: Vec<Operation>,
}

impl VersionDiff {
    /// Creates a new diff
    pub fn new(from: VersionId, to: VersionId) -> Self {
        VersionDiff {
            from_version: from,
            to_version: to,
            columns_added: Vec::new(),
            columns_removed: Vec::new(),
            type_changes: HashMap::new(),
            row_count_diff: 0,
            operations: Vec::new(),
        }
    }

    /// Computes the diff between two schemas
    pub fn from_schemas(from: &DataVersion, to: &DataVersion) -> Self {
        let mut diff = VersionDiff::new(from.id.clone(), to.id.clone());

        // Find added columns
        for col in &to.schema.columns {
            if !from.schema.columns.contains(col) {
                diff.columns_added.push(col.clone());
            }
        }

        // Find removed columns
        for col in &from.schema.columns {
            if !to.schema.columns.contains(col) {
                diff.columns_removed.push(col.clone());
            }
        }

        // Find type changes
        for col in &from.schema.columns {
            if to.schema.columns.contains(col) {
                let from_type = from.schema.types.get(col);
                let to_type = to.schema.types.get(col);

                if from_type != to_type {
                    diff.type_changes.insert(
                        col.clone(),
                        (
                            from_type.cloned().unwrap_or_default(),
                            to_type.cloned().unwrap_or_default(),
                        ),
                    );
                }
            }
        }

        // Row count difference
        diff.row_count_diff = to.schema.row_count as i64 - from.schema.row_count as i64;

        diff
    }

    /// Checks if there are any changes
    pub fn has_changes(&self) -> bool {
        !self.columns_added.is_empty()
            || !self.columns_removed.is_empty()
            || !self.type_changes.is_empty()
            || self.row_count_diff != 0
    }
}

/// Errors that can occur during versioning operations
#[derive(Debug, Clone)]
pub enum VersioningError {
    /// Version not found
    VersionNotFound(VersionId),
    /// Operation not found
    OperationNotFound(String),
    /// Invalid operation
    InvalidOperation(String),
    /// Storage error
    StorageError(String),
    /// Serialization error
    SerializationError(String),
}

impl std::fmt::Display for VersioningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersioningError::VersionNotFound(id) => {
                write!(f, "Version not found: {}", id)
            }
            VersioningError::OperationNotFound(id) => {
                write!(f, "Operation not found: {}", id)
            }
            VersioningError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
            VersioningError::StorageError(msg) => {
                write!(f, "Storage error: {}", msg)
            }
            VersioningError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
        }
    }
}

impl std::error::Error for VersioningError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_id_creation() {
        let v1 = VersionId::new();
        let v2 = VersionId::new();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_operation_type_display() {
        let op = OperationType::Select {
            columns: vec!["a".to_string(), "b".to_string()],
        };
        assert!(op.to_string().contains("SELECT"));
        assert!(op.to_string().contains("a"));
        assert!(op.to_string().contains("b"));
    }

    #[test]
    fn test_data_schema_compatibility() {
        let schema1 = DataSchema::new(
            vec!["a".to_string(), "b".to_string()],
            [
                ("a".to_string(), "f64".to_string()),
                ("b".to_string(), "String".to_string()),
            ]
            .into_iter()
            .collect(),
            100,
        );

        let schema2 = DataSchema::new(
            vec!["a".to_string(), "b".to_string()],
            [
                ("a".to_string(), "f64".to_string()),
                ("b".to_string(), "String".to_string()),
            ]
            .into_iter()
            .collect(),
            200,
        );

        assert!(schema1.is_compatible(&schema2));
    }

    #[test]
    fn test_version_diff() {
        let schema1 = DataSchema::new(
            vec!["a".to_string(), "b".to_string()],
            [
                ("a".to_string(), "f64".to_string()),
                ("b".to_string(), "String".to_string()),
            ]
            .into_iter()
            .collect(),
            100,
        );

        let schema2 = DataSchema::new(
            vec!["a".to_string(), "c".to_string()],
            [
                ("a".to_string(), "f64".to_string()),
                ("c".to_string(), "i64".to_string()),
            ]
            .into_iter()
            .collect(),
            150,
        );

        let v1 = DataVersion::new(schema1);
        let v2 = DataVersion::new(schema2);

        let diff = VersionDiff::from_schemas(&v1, &v2);

        assert!(diff.columns_added.contains(&"c".to_string()));
        assert!(diff.columns_removed.contains(&"b".to_string()));
        assert_eq!(diff.row_count_diff, 50);
    }
}
