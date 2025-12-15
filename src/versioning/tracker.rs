//! Lineage tracking and version management
//!
//! This module provides the main interface for tracking data versions
//! and their lineage.

use super::core::{
    DataSchema, DataVersion, Operation, OperationType, VersionDiff, VersionId, VersioningError,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

/// Configuration for the lineage tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageConfig {
    /// Maximum number of versions to keep in memory
    pub max_versions: usize,
    /// Whether to track detailed operation history
    pub track_operations: bool,
    /// Whether to compute and store data hashes
    pub compute_hashes: bool,
    /// Default user name for operations
    pub default_user: Option<String>,
}

impl Default for LineageConfig {
    fn default() -> Self {
        LineageConfig {
            max_versions: 1000,
            track_operations: true,
            compute_hashes: false,
            default_user: None,
        }
    }
}

/// Main lineage tracker for managing data versions
#[derive(Debug)]
pub struct LineageTracker {
    /// All tracked versions
    versions: HashMap<VersionId, DataVersion>,
    /// All tracked operations
    operations: Vec<Operation>,
    /// Index of operations by output version
    operations_by_output: HashMap<VersionId, Vec<usize>>,
    /// Index of operations by input versions
    operations_by_input: HashMap<VersionId, Vec<usize>>,
    /// Named references (like "latest", "production", etc.)
    refs: HashMap<String, VersionId>,
    /// Configuration
    config: LineageConfig,
    /// Order of version creation
    version_order: Vec<VersionId>,
}

impl LineageTracker {
    /// Creates a new lineage tracker with default config
    pub fn new() -> Self {
        Self::with_config(LineageConfig::default())
    }

    /// Creates a new lineage tracker with custom config
    pub fn with_config(config: LineageConfig) -> Self {
        LineageTracker {
            versions: HashMap::new(),
            operations: Vec::new(),
            operations_by_output: HashMap::new(),
            operations_by_input: HashMap::new(),
            refs: HashMap::new(),
            config,
            version_order: Vec::new(),
        }
    }

    /// Registers a new version
    pub fn register_version(&mut self, version: DataVersion) -> VersionId {
        let id = version.id.clone();

        // Enforce max versions limit
        if self.versions.len() >= self.config.max_versions {
            // Remove oldest version that's not referenced
            if let Some(oldest) = self.find_oldest_unreferenced_version() {
                self.remove_version(&oldest);
            }
        }

        self.version_order.push(id.clone());
        self.versions.insert(id.clone(), version);
        id
    }

    /// Finds the oldest version that's not referenced
    fn find_oldest_unreferenced_version(&self) -> Option<VersionId> {
        let referenced: HashSet<&VersionId> = self.refs.values().collect();

        for id in &self.version_order {
            if !referenced.contains(id) {
                return Some(id.clone());
            }
        }
        None
    }

    /// Removes a version and its associated operations
    fn remove_version(&mut self, version_id: &VersionId) {
        self.versions.remove(version_id);
        self.operations_by_output.remove(version_id);
        self.operations_by_input.remove(version_id);
        self.version_order.retain(|id| id != version_id);
    }

    /// Gets a version by ID
    pub fn get_version(&self, id: &VersionId) -> Option<&DataVersion> {
        self.versions.get(id)
    }

    /// Gets a version by reference name
    pub fn get_version_by_ref(&self, ref_name: &str) -> Option<&DataVersion> {
        self.refs.get(ref_name).and_then(|id| self.versions.get(id))
    }

    /// Sets a named reference to a version
    pub fn set_ref(&mut self, name: &str, version_id: VersionId) -> Result<(), VersioningError> {
        if !self.versions.contains_key(&version_id) {
            return Err(VersioningError::VersionNotFound(version_id));
        }
        self.refs.insert(name.to_string(), version_id);
        Ok(())
    }

    /// Gets a reference ID
    pub fn get_ref(&self, name: &str) -> Option<&VersionId> {
        self.refs.get(name)
    }

    /// Lists all references
    pub fn list_refs(&self) -> Vec<(&str, &VersionId)> {
        self.refs.iter().map(|(k, v)| (k.as_str(), v)).collect()
    }

    /// Records an operation
    pub fn record_operation(&mut self, operation: Operation) {
        if !self.config.track_operations {
            return;
        }

        let op_index = self.operations.len();

        // Index by output
        self.operations_by_output
            .entry(operation.output.clone())
            .or_insert_with(Vec::new)
            .push(op_index);

        // Index by inputs
        for input in &operation.inputs {
            self.operations_by_input
                .entry(input.clone())
                .or_insert_with(Vec::new)
                .push(op_index);
        }

        self.operations.push(operation);
    }

    /// Gets all operations that produced a version
    pub fn get_operations_producing(&self, version_id: &VersionId) -> Vec<&Operation> {
        self.operations_by_output
            .get(version_id)
            .map(|indices| indices.iter().map(|&i| &self.operations[i]).collect())
            .unwrap_or_default()
    }

    /// Gets all operations that used a version as input
    pub fn get_operations_using(&self, version_id: &VersionId) -> Vec<&Operation> {
        self.operations_by_input
            .get(version_id)
            .map(|indices| indices.iter().map(|&i| &self.operations[i]).collect())
            .unwrap_or_default()
    }

    /// Gets the full lineage of a version (all ancestor versions)
    pub fn get_lineage(&self, version_id: &VersionId) -> Vec<&DataVersion> {
        let mut lineage = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(version_id);

        while let Some(current_id) = queue.pop_front() {
            if visited.contains(current_id) {
                continue;
            }
            visited.insert(current_id.clone());

            if let Some(version) = self.versions.get(current_id) {
                lineage.push(version);

                for parent_id in &version.parents {
                    if !visited.contains(parent_id) {
                        queue.push_back(parent_id);
                    }
                }
            }
        }

        lineage
    }

    /// Gets all operations in the lineage of a version
    pub fn get_operation_history(&self, version_id: &VersionId) -> Vec<&Operation> {
        let mut history = Vec::new();
        let mut visited_versions = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(version_id.clone());

        while let Some(current_id) = queue.pop_front() {
            if visited_versions.contains(&current_id) {
                continue;
            }
            visited_versions.insert(current_id.clone());

            // Get operations that produced this version
            for op in self.get_operations_producing(&current_id) {
                history.push(op);

                // Add input versions to the queue
                for input_id in &op.inputs {
                    if !visited_versions.contains(input_id) {
                        queue.push_back(input_id.clone());
                    }
                }
            }
        }

        // Sort by timestamp
        history.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        history
    }

    /// Computes the diff between two versions
    pub fn diff(
        &self,
        from_id: &VersionId,
        to_id: &VersionId,
    ) -> Result<VersionDiff, VersioningError> {
        let from = self
            .versions
            .get(from_id)
            .ok_or_else(|| VersioningError::VersionNotFound(from_id.clone()))?;

        let to = self
            .versions
            .get(to_id)
            .ok_or_else(|| VersioningError::VersionNotFound(to_id.clone()))?;

        Ok(VersionDiff::from_schemas(from, to))
    }

    /// Lists all versions
    pub fn list_versions(&self) -> Vec<&DataVersion> {
        self.version_order
            .iter()
            .filter_map(|id| self.versions.get(id))
            .collect()
    }

    /// Lists versions by tag
    pub fn list_versions_by_tag(&self, tag: &str) -> Vec<&DataVersion> {
        self.versions
            .values()
            .filter(|v| v.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Searches versions by name pattern
    pub fn search_versions(&self, pattern: &str) -> Vec<&DataVersion> {
        let pattern_lower = pattern.to_lowercase();
        self.versions
            .values()
            .filter(|v| {
                v.name
                    .as_ref()
                    .map(|n| n.to_lowercase().contains(&pattern_lower))
                    .unwrap_or(false)
                    || v.description
                        .as_ref()
                        .map(|d| d.to_lowercase().contains(&pattern_lower))
                        .unwrap_or(false)
            })
            .collect()
    }

    /// Gets statistics about the tracker
    pub fn stats(&self) -> TrackerStats {
        let operation_counts: HashMap<String, usize> = self
            .operations
            .iter()
            .map(|op| op.operation_type.to_string())
            .fold(HashMap::new(), |mut acc, op_type| {
                *acc.entry(op_type).or_insert(0) += 1;
                acc
            });

        TrackerStats {
            version_count: self.versions.len(),
            operation_count: self.operations.len(),
            ref_count: self.refs.len(),
            operation_counts,
        }
    }

    /// Exports the lineage graph as a DOT format string
    pub fn export_dot(&self) -> String {
        let mut dot = String::from("digraph lineage {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add version nodes
        for (id, version) in &self.versions {
            let label = version.name.as_deref().unwrap_or(&id.0);
            let rows = version.schema.row_count;
            let cols = version.schema.columns.len();
            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\\n({} rows, {} cols)\"];\n",
                id, label, rows, cols
            ));
        }

        dot.push_str("\n");

        // Add edges for parent relationships
        for (id, version) in &self.versions {
            for parent_id in &version.parents {
                dot.push_str(&format!("  \"{}\" -> \"{}\";\n", parent_id, id));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Clears all data
    pub fn clear(&mut self) {
        self.versions.clear();
        self.operations.clear();
        self.operations_by_output.clear();
        self.operations_by_input.clear();
        self.refs.clear();
        self.version_order.clear();
    }
}

impl Default for LineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the tracker
#[derive(Debug, Clone)]
pub struct TrackerStats {
    /// Number of versions
    pub version_count: usize,
    /// Number of operations
    pub operation_count: usize,
    /// Number of references
    pub ref_count: usize,
    /// Count by operation type
    pub operation_counts: HashMap<String, usize>,
}

/// Thread-safe wrapper for LineageTracker
#[derive(Debug, Clone)]
pub struct SharedLineageTracker {
    inner: Arc<RwLock<LineageTracker>>,
}

impl SharedLineageTracker {
    /// Creates a new shared tracker
    pub fn new() -> Self {
        SharedLineageTracker {
            inner: Arc::new(RwLock::new(LineageTracker::new())),
        }
    }

    /// Creates a shared tracker with custom config
    pub fn with_config(config: LineageConfig) -> Self {
        SharedLineageTracker {
            inner: Arc::new(RwLock::new(LineageTracker::with_config(config))),
        }
    }

    /// Registers a version
    pub fn register_version(&self, version: DataVersion) -> VersionId {
        self.inner.write().unwrap().register_version(version)
    }

    /// Gets a version by ID
    pub fn get_version(&self, id: &VersionId) -> Option<DataVersion> {
        self.inner.read().unwrap().get_version(id).cloned()
    }

    /// Records an operation
    pub fn record_operation(&self, operation: Operation) {
        self.inner.write().unwrap().record_operation(operation)
    }

    /// Sets a reference
    pub fn set_ref(&self, name: &str, version_id: VersionId) -> Result<(), VersioningError> {
        self.inner.write().unwrap().set_ref(name, version_id)
    }

    /// Gets stats
    pub fn stats(&self) -> TrackerStats {
        self.inner.read().unwrap().stats()
    }
}

impl Default for SharedLineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_schema(cols: &[&str]) -> DataSchema {
        DataSchema::new(
            cols.iter().map(|s| s.to_string()).collect(),
            cols.iter()
                .map(|s| (s.to_string(), "String".to_string()))
                .collect(),
            100,
        )
    }

    #[test]
    fn test_register_version() {
        let mut tracker = LineageTracker::new();

        let version = DataVersion::new(create_test_schema(&["a", "b"])).with_name("test_v1");

        let id = tracker.register_version(version);

        assert!(tracker.get_version(&id).is_some());
    }

    #[test]
    fn test_set_and_get_ref() {
        let mut tracker = LineageTracker::new();

        let version = DataVersion::new(create_test_schema(&["a", "b"]));
        let id = tracker.register_version(version);

        tracker.set_ref("latest", id.clone()).unwrap();

        let ref_version = tracker.get_version_by_ref("latest");
        assert!(ref_version.is_some());
        assert_eq!(ref_version.unwrap().id, id);
    }

    #[test]
    fn test_record_operation() {
        let mut tracker = LineageTracker::new();

        let v1 = tracker.register_version(DataVersion::new(create_test_schema(&["a", "b"])));
        let v2 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a"])).with_parents(vec![v1.clone()]),
        );

        let op = Operation::new(
            OperationType::Select {
                columns: vec!["a".to_string()],
            },
            vec![v1.clone()],
            v2.clone(),
        );

        tracker.record_operation(op);

        let producing_ops = tracker.get_operations_producing(&v2);
        assert_eq!(producing_ops.len(), 1);

        let using_ops = tracker.get_operations_using(&v1);
        assert_eq!(using_ops.len(), 1);
    }

    #[test]
    fn test_lineage() {
        let mut tracker = LineageTracker::new();

        let v1 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "b"])).with_name("original"),
        );

        let v2 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a"]))
                .with_name("filtered")
                .with_parents(vec![v1.clone()]),
        );

        let v3 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "c"]))
                .with_name("transformed")
                .with_parents(vec![v2.clone()]),
        );

        let lineage = tracker.get_lineage(&v3);

        assert_eq!(lineage.len(), 3);
    }

    #[test]
    fn test_diff() {
        let mut tracker = LineageTracker::new();

        let v1 = tracker.register_version(DataVersion::new(create_test_schema(&["a", "b"])));
        let v2 = tracker.register_version(DataVersion::new(create_test_schema(&["a", "c"])));

        let diff = tracker.diff(&v1, &v2).unwrap();

        assert!(diff.columns_added.contains(&"c".to_string()));
        assert!(diff.columns_removed.contains(&"b".to_string()));
    }

    #[test]
    fn test_export_dot() {
        let mut tracker = LineageTracker::new();

        let v1 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "b"])).with_name("source"),
        );
        let v2 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a"]))
                .with_name("filtered")
                .with_parents(vec![v1]),
        );

        let dot = tracker.export_dot();

        assert!(dot.contains("digraph"));
        assert!(dot.contains("source"));
        assert!(dot.contains("filtered"));
    }

    #[test]
    fn test_shared_tracker() {
        let tracker = SharedLineageTracker::new();

        let version = DataVersion::new(create_test_schema(&["a", "b"]));
        let id = tracker.register_version(version);

        assert!(tracker.get_version(&id).is_some());
    }
}
