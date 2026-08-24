//! Data versioning and lineage tracking module
//!
//! This module provides comprehensive data versioning and lineage tracking
//! capabilities for DataFrames, allowing you to:
//!
//! - Track versions of data over time
//! - Record the history of operations performed on data
//! - Trace the lineage of data back to its sources
//! - Compare differences between versions
//! - Record schema-only versions cheaply ([`DataFrameVersioning::create_version`]),
//!   or take a genuine data snapshot you can restore later
//!   ([`DataFrameVersioning::create_snapshot`] + [`SnapshotStore`](crate::versioning::SnapshotStore))
//!
//! `create_version`/`create_named_version` capture schema metadata only --
//! they do not preserve the underlying data and cannot back a rollback.
//! `create_snapshot`/`create_named_snapshot` are the ones that do: they
//! clone the DataFrame's actual data into a [`SnapshotStore`](crate::versioning::SnapshotStore), which
//! `SnapshotStore::restore` later hands back out.
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
//! tracker.set_ref("latest", v1.clone()).expect("operation should succeed");
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
//! // let diff = tracker.diff(&v1_id, &v2_id).expect("operation should succeed");
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
//! let version_id = handle.join().expect("operation should succeed").expect("register should succeed");
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

/// An in-memory store of full DataFrame snapshots, keyed by [`VersionId`].
///
/// [`LineageTracker`] only ever sees a [`DataVersion`]'s **schema** (column
/// names/types/row count) -- it has no dependency on [`DataFrame`] and never
/// receives the underlying cell data, so it cannot honestly claim to
/// "snapshot" or "roll back" anything. This store is the actual
/// data-capturing counterpart: [`DataFrameVersioning::create_snapshot`]
/// clones the full `DataFrame` into it, and [`Self::restore`] hands that
/// clone back out -- a genuine rollback target, unlike
/// [`DataFrameVersioning::create_version`], which only ever records schema
/// metadata. Like [`LineageTracker`] itself, this is a plain in-memory
/// store with no persistence.
#[derive(Debug, Clone, Default)]
pub struct SnapshotStore {
    snapshots: HashMap<VersionId, DataFrame>,
}

impl SnapshotStore {
    /// Creates a new, empty snapshot store.
    pub fn new() -> Self {
        SnapshotStore {
            snapshots: HashMap::new(),
        }
    }

    /// Stores (or replaces) the snapshot for `id`.
    pub fn store(&mut self, id: VersionId, df: DataFrame) {
        self.snapshots.insert(id, df);
    }

    /// Retrieves the snapshot for `id`, if one was stored.
    pub fn restore(&self, id: &VersionId) -> Option<&DataFrame> {
        self.snapshots.get(id)
    }

    /// Removes and returns the snapshot for `id`, if one was stored.
    pub fn remove(&mut self, id: &VersionId) -> Option<DataFrame> {
        self.snapshots.remove(id)
    }

    /// Returns whether a snapshot is stored for `id`.
    pub fn contains(&self, id: &VersionId) -> bool {
        self.snapshots.contains_key(id)
    }

    /// Number of snapshots currently stored.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether the store holds no snapshots.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }
}

/// Compute a real SHA-256 hash over a DataFrame's rendered contents, for
/// [`DataFrameVersioning::create_snapshot`]'s `data_hash` when
/// [`crate::versioning::LineageConfig::compute_hashes`] is enabled.
///
/// Every column's name and every value's string representation are fed
/// through the hasher in a fixed (sorted) column order with separators, so
/// the result is deterministic and column-order-independent inputs don't
/// collide with each other. Columns whose element type can't be rendered to
/// a string (see `DataFrame::get_column_string_values`) are skipped rather
/// than failing the whole hash -- this is a change-detection fingerprint,
/// not a cryptographic commitment to every byte of the DataFrame.
fn compute_data_hash(df: &DataFrame) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    let mut columns = df.column_names().to_vec();
    columns.sort();
    for col_name in &columns {
        hasher.update(col_name.as_bytes());
        hasher.update([0u8]);
        if let Ok(values) = df.get_column_string_values(col_name) {
            for v in values {
                hasher.update(v.as_bytes());
                hasher.update([0u8]);
            }
        }
        hasher.update([0xFFu8]); // column separator
    }
    // `Sha256::finalize()`'s output type doesn't implement `LowerHex`
    // itself, so hex-encode byte by byte instead of `format!("{:x}", ..)`.
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

/// Extension trait for DataFrame to integrate with versioning
pub trait DataFrameVersioning {
    /// Creates a DataSchema from this DataFrame
    fn to_schema(&self) -> DataSchema;

    /// Records this DataFrame's **schema** (column names, types, row count)
    /// as a new version in `tracker`.
    ///
    /// This captures metadata only, not the underlying cell data -- despite
    /// the name, it is not a data snapshot and cannot back a rollback. Use
    /// [`Self::create_snapshot`] when you need the actual data preserved
    /// and retrievable later.
    fn create_version(&self, tracker: &mut LineageTracker) -> VersionId;

    /// Same as [`Self::create_version`], with a human-readable name attached.
    fn create_named_version(&self, tracker: &mut LineageTracker, name: &str) -> VersionId;

    /// Records this DataFrame's schema in `tracker` **and** stores a full
    /// clone of its data in `store`, so it can later be retrieved via
    /// [`SnapshotStore::restore`] -- a genuine rollback target, unlike
    /// [`Self::create_version`]. If `tracker`'s
    /// [`LineageConfig::compute_hashes`] is enabled, the registered
    /// version's `data_hash` is a real SHA-256 over the DataFrame's
    /// contents (see `compute_data_hash`); otherwise `data_hash` stays
    /// `None`, exactly as [`Self::create_version`] leaves it.
    fn create_snapshot(&self, tracker: &mut LineageTracker, store: &mut SnapshotStore)
        -> VersionId;

    /// Same as [`Self::create_snapshot`], with a human-readable name attached.
    fn create_named_snapshot(
        &self,
        tracker: &mut LineageTracker,
        store: &mut SnapshotStore,
        name: &str,
    ) -> VersionId;
}

impl DataFrameVersioning for DataFrame {
    fn to_schema(&self) -> DataSchema {
        let columns = self.column_names();
        let types: HashMap<String, String> = columns
            .iter()
            .map(|col| {
                // This DataFrame model has no distinct physical
                // representation for "categorical" data: a categorical
                // series is converted to a plain `Series<String>` the
                // moment it's added (see
                // `DataFrame::add_na_series_as_categorical`), so a
                // genuinely categorical column and an ordinary string
                // column are indistinguishable here after the fact.
                // Labelling every string column "Categorical" (the
                // previous behaviour) wasn't a real type distinction, just
                // a mislabeling of 100% of string columns; "String" is the
                // honest answer for both.
                let type_str = if self.is_numeric_column(col) {
                    "f64"
                } else if self.get_column::<bool>(col).is_ok() {
                    "bool"
                } else {
                    "String"
                };
                (col.clone(), type_str.to_string())
            })
            .collect();

        DataSchema::new(columns.to_vec(), types, self.row_count())
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

    fn create_snapshot(
        &self,
        tracker: &mut LineageTracker,
        store: &mut SnapshotStore,
    ) -> VersionId {
        let schema = self.to_schema();
        let mut version = DataVersion::new(schema);
        if tracker.config().compute_hashes {
            version.data_hash = Some(compute_data_hash(self));
        }
        let id = tracker.register_version(version);
        store.store(id.clone(), self.clone());
        id
    }

    fn create_named_snapshot(
        &self,
        tracker: &mut LineageTracker,
        store: &mut SnapshotStore,
        name: &str,
    ) -> VersionId {
        let schema = self.to_schema();
        let mut version = DataVersion::new(schema).with_name(name);
        if tracker.config().compute_hashes {
            version.data_hash = Some(compute_data_hash(self));
        }
        let id = tracker.register_version(version);
        store.store(id.clone(), self.clone());
        id
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
        .expect("operation should succeed");

        let values = Series::new(vec![1.0, 2.0, 3.0], Some("value".to_string()))
            .expect("operation should succeed");

        df.add_column("name".to_string(), names)
            .expect("operation should succeed");
        df.add_column("value".to_string(), values)
            .expect("operation should succeed");

        df
    }

    #[test]
    fn test_dataframe_to_schema() {
        let df = create_test_dataframe();
        let schema = df.to_schema();

        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.row_count, 3);
        // "name" is a plain string column: it must be reported as "String",
        // not fabricated as "Categorical" (this DataFrame model has no
        // physical way to tell the two apart, so claiming "Categorical" for
        // every string column was never a real type distinction).
        assert_eq!(schema.types.get("name").map(String::as_str), Some("String"));
        assert_eq!(schema.types.get("value").map(String::as_str), Some("f64"));
    }

    #[test]
    fn test_create_snapshot_stores_real_data_and_hash() {
        let df = create_test_dataframe();
        let mut tracker = LineageTracker::with_config(LineageConfig {
            compute_hashes: true,
            ..LineageConfig::default()
        });
        let mut store = SnapshotStore::new();

        let id = df.create_snapshot(&mut tracker, &mut store);

        // The tracker's own record is still schema-only...
        let version = tracker.get_version(&id).expect("version");
        assert_eq!(version.schema.row_count, 3);
        // ...but a real hash was computed (compute_hashes was on)...
        assert!(version.data_hash.is_some());
        // ...and the actual data is retrievable from the snapshot store,
        // which is what makes this a genuine rollback target.
        let restored = store.restore(&id).expect("snapshot present");
        assert_eq!(restored.row_count(), 3);
        assert_eq!(
            restored.get_column_string_values("name").expect("name"),
            df.get_column_string_values("name").expect("name")
        );
    }

    #[test]
    fn test_create_version_never_populates_data_hash() {
        // Even with compute_hashes on, the schema-only path has no data to
        // hash and must not fabricate one.
        let df = create_test_dataframe();
        let mut tracker = LineageTracker::with_config(LineageConfig {
            compute_hashes: true,
            ..LineageConfig::default()
        });
        let id = df.create_version(&mut tracker);
        let version = tracker.get_version(&id).expect("version");
        assert!(version.data_hash.is_none());
    }

    #[test]
    fn test_dataframe_create_version() {
        let df = create_test_dataframe();
        let mut tracker = LineageTracker::new();

        let version_id = df.create_version(&mut tracker);
        let version = tracker
            .get_version(&version_id)
            .expect("operation should succeed");

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
