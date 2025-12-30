//! Data versioning and lineage tracking module
//!
//! This module provides comprehensive data versioning and lineage tracking
//! capabilities for DataFrames, allowing you to:
//!
//! - Track versions of data over time
//! - Record the history of operations performed on data
//! - Trace the lineage of data back to its sources
//! - Compare differences between versions
//! - Create snapshots and checkpoints
//!
//! # Quick Start
//!
//! ```rust
//! use pandrs::versioning::{LineageTracker, DataVersion, DataSchema, Operation, OperationType};
//!
//! // Create a lineage tracker
//! let mut tracker = LineageTracker::new();
//!
//! // Define a schema for your data
//! let schema = DataSchema::new(
//!     vec!["name".to_string(), "value".to_string()],
//!     [("name".to_string(), "String".to_string()),
//!      ("value".to_string(), "f64".to_string())]
//!     .into_iter().collect(),
//!     1000,
//! );
//!
//! // Register the initial version
//! let v1 = tracker.register_version(
//!     DataVersion::new(schema).with_name("raw_data")
//! );
//!
//! // Set a reference to the latest version
//! tracker.set_ref("latest", v1.clone()).unwrap();
//! ```
//!
//! # Recording Operations
//!
//! ```rust
//! use pandrs::versioning::{LineageTracker, DataVersion, DataSchema, Operation, OperationType};
//!
//! let mut tracker = LineageTracker::new();
//!
//! // Create initial version
//! let schema1 = DataSchema::new(
//!     vec!["a".to_string(), "b".to_string(), "c".to_string()],
//!     [("a".to_string(), "f64".to_string()),
//!      ("b".to_string(), "f64".to_string()),
//!      ("c".to_string(), "String".to_string())]
//!     .into_iter().collect(),
//!     1000,
//! );
//! let v1 = tracker.register_version(DataVersion::new(schema1));
//!
//! // Create derived version
//! let schema2 = DataSchema::new(
//!     vec!["a".to_string(), "b".to_string()],
//!     [("a".to_string(), "f64".to_string()),
//!      ("b".to_string(), "f64".to_string())]
//!     .into_iter().collect(),
//!     1000,
//! );
//! let v2 = tracker.register_version(
//!     DataVersion::new(schema2).with_parents(vec![v1.clone()])
//! );
//!
//! // Record the operation that created v2 from v1
//! let op = Operation::new(
//!     OperationType::Select { columns: vec!["a".to_string(), "b".to_string()] },
//!     vec![v1],
//!     v2.clone(),
//! );
//! tracker.record_operation(op);
//! ```
//!
//! # Querying Lineage
//!
//! ```rust
//! use pandrs::versioning::{LineageTracker, DataVersion, DataSchema};
//!
//! let mut tracker = LineageTracker::new();
//!
//! // ... register versions and operations ...
//!
//! // Get the lineage of a version
//! // let lineage = tracker.get_lineage(&some_version_id);
//!
//! // Get the operation history
//! // let history = tracker.get_operation_history(&some_version_id);
//!
//! // Compute diff between versions
//! // let diff = tracker.diff(&v1_id, &v2_id).unwrap();
//! ```
//!
//! # Thread-Safe Usage
//!
//! ```rust
//! use pandrs::versioning::{SharedLineageTracker, DataVersion, DataSchema};
//! use std::thread;
//!
//! let tracker = SharedLineageTracker::new();
//! let tracker_clone = tracker.clone();
//!
//! let handle = thread::spawn(move || {
//!     let schema = DataSchema::new(
//!         vec!["x".to_string()],
//!         [("x".to_string(), "f64".to_string())].into_iter().collect(),
//!         100,
//!     );
//!     tracker_clone.register_version(DataVersion::new(schema))
//! });
//!
//! let version_id = handle.join().unwrap();
//! assert!(tracker.get_version(&version_id).is_some());
//! ```

pub mod core;
pub mod tracker;

// Re-export main types
pub use core::{
    DataSchema, DataVersion, Operation, OperationType, VersionDiff, VersionId, VersioningError,
};

pub use tracker::{LineageConfig, LineageTracker, SharedLineageTracker, TrackerStats};

use crate::DataFrame;
use std::collections::HashMap;

/// Extension trait for DataFrame to integrate with versioning
pub trait DataFrameVersioning {
    /// Creates a DataSchema from this DataFrame
    fn to_schema(&self) -> DataSchema;

    /// Creates a versioned snapshot of this DataFrame
    fn create_version(&self, tracker: &mut LineageTracker) -> VersionId;

    /// Creates a versioned snapshot with a name
    fn create_named_version(&self, tracker: &mut LineageTracker, name: &str) -> VersionId;
}

impl DataFrameVersioning for DataFrame {
    fn to_schema(&self) -> DataSchema {
        let columns = self.column_names();
        let types: HashMap<String, String> = columns
            .iter()
            .map(|col| {
                let type_str = if self.is_numeric_column(col) {
                    "f64"
                } else if self.is_categorical(col) {
                    "Categorical"
                } else {
                    "String"
                };
                (col.clone(), type_str.to_string())
            })
            .collect();

        DataSchema::new(columns, types, self.row_count())
    }

    fn create_version(&self, tracker: &mut LineageTracker) -> VersionId {
        let schema = self.to_schema();
        let version = DataVersion::new(schema);
        tracker.register_version(version)
    }

    fn create_named_version(&self, tracker: &mut LineageTracker, name: &str) -> VersionId {
        let schema = self.to_schema();
        let version = DataVersion::new(schema).with_name(name);
        tracker.register_version(version)
    }
}

/// Builder for recording transformations on DataFrames
pub struct VersionedTransform<'a> {
    tracker: &'a mut LineageTracker,
    input_version: VersionId,
    operations: Vec<OperationType>,
}

impl<'a> VersionedTransform<'a> {
    /// Creates a new versioned transform
    pub fn new(tracker: &'a mut LineageTracker, input_version: VersionId) -> Self {
        VersionedTransform {
            tracker,
            input_version,
            operations: Vec::new(),
        }
    }

    /// Records a select operation
    pub fn select(mut self, columns: Vec<String>) -> Self {
        self.operations.push(OperationType::Select { columns });
        self
    }

    /// Records a filter operation
    pub fn filter(mut self, condition: &str) -> Self {
        self.operations.push(OperationType::Filter {
            condition: condition.to_string(),
        });
        self
    }

    /// Records a sort operation
    pub fn sort(mut self, columns: Vec<String>, ascending: Vec<bool>) -> Self {
        self.operations
            .push(OperationType::Sort { columns, ascending });
        self
    }

    /// Records a column addition
    pub fn add_column(mut self, column_name: &str) -> Self {
        self.operations.push(OperationType::AddColumn {
            column_name: column_name.to_string(),
        });
        self
    }

    /// Records a drop columns operation
    pub fn drop_columns(mut self, columns: Vec<String>) -> Self {
        self.operations.push(OperationType::DropColumn { columns });
        self
    }

    /// Records a custom transformation
    pub fn transform(mut self, name: &str, description: &str) -> Self {
        self.operations.push(OperationType::Transform {
            name: name.to_string(),
            description: description.to_string(),
        });
        self
    }

    /// Finalizes the transformation and creates a new version
    pub fn commit(self, output_schema: DataSchema) -> VersionId {
        let output_version =
            DataVersion::new(output_schema).with_parents(vec![self.input_version.clone()]);

        let output_id = self.tracker.register_version(output_version);

        // Record all operations
        for op_type in self.operations {
            let op = Operation::new(op_type, vec![self.input_version.clone()], output_id.clone());
            self.tracker.record_operation(op);
        }

        output_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Series;

    fn create_test_dataframe() -> DataFrame {
        let mut df = DataFrame::new();

        let names = Series::new(
            vec![
                "Alice".to_string(),
                "Bob".to_string(),
                "Charlie".to_string(),
            ],
            Some("name".to_string()),
        )
        .unwrap();

        let values = Series::new(vec![1.0, 2.0, 3.0], Some("value".to_string())).unwrap();

        df.add_column("name".to_string(), names).unwrap();
        df.add_column("value".to_string(), values).unwrap();

        df
    }

    #[test]
    fn test_dataframe_to_schema() {
        let df = create_test_dataframe();
        let schema = df.to_schema();

        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.row_count, 3);
    }

    #[test]
    fn test_dataframe_create_version() {
        let df = create_test_dataframe();
        let mut tracker = LineageTracker::new();

        let version_id = df.create_version(&mut tracker);
        let version = tracker.get_version(&version_id).unwrap();

        assert_eq!(version.schema.row_count, 3);
    }

    #[test]
    fn test_versioned_transform() {
        let df = create_test_dataframe();
        let mut tracker = LineageTracker::new();

        // Create initial version
        let v1 = df.create_named_version(&mut tracker, "original");

        // Apply transformation
        let schema2 = DataSchema::new(
            vec!["name".to_string()],
            [("name".to_string(), "String".to_string())]
                .into_iter()
                .collect(),
            3,
        );

        let v2 = VersionedTransform::new(&mut tracker, v1.clone())
            .select(vec!["name".to_string()])
            .commit(schema2);

        // Check lineage
        let lineage = tracker.get_lineage(&v2);
        assert_eq!(lineage.len(), 2);

        // Check operations
        let ops = tracker.get_operations_producing(&v2);
        assert_eq!(ops.len(), 1);
    }
}
