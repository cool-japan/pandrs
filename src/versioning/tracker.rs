//! Lineage tracking and version management
//!
//! This module provides the main interface for tracking data versions
//! and their lineage.

use super::core::{DataVersion, Operation, VersionDiff, VersionId, VersioningError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};

use crate::{read_lock_safe, write_lock_safe};

/// Escape a string for safe interpolation into a double-quoted DOT string
/// literal (used by [`LineageTracker::export_dot`]). Backslash and
/// double-quote are the two characters DOT treats specially inside `"..."`;
/// a literal newline is also escaped so a multi-line name can't break the
/// statement it's embedded in.
fn escape_dot(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out
}

/// Configuration for the lineage tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageConfig {
    /// Soft cap on the number of versions to keep in memory.
    ///
    /// This is best-effort, not a hard limit: a version that is still
    /// reachable as another live version's parent is never evicted, even
    /// past this cap, because evicting it would silently truncate that
    /// other version's lineage (`get_lineage` would just stop early with no
    /// indication data was discarded). If every tracked version is
    /// reachable this way, the tracker keeps growing past `max_versions`
    /// rather than corrupt lineage to honor the cap.
    pub max_versions: usize,
    /// Whether to track detailed operation history
    pub track_operations: bool,
    /// Whether to compute and store a real data hash on
    /// [`DataVersion::data_hash`].
    ///
    /// This tracker never sees a `DataFrame`'s actual cell data -- only the
    /// [`super::core::DataSchema`] passed to [`LineageTracker::register_version`]
    /// -- so it has nothing to hash. This flag is consulted by
    /// [`super::DataFrameVersioning::create_snapshot`], which *does* have
    /// the real `DataFrame` and computes a genuine SHA-256 over its
    /// contents when this is `true`. Plain `register_version` (and the
    /// `create_version`/`create_named_version` schema-only path) never
    /// populates `data_hash`, regardless of this setting -- there would be
    /// nothing honest to hash.
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

    /// Returns the tracker's configuration.
    pub fn config(&self) -> &LineageConfig {
        &self.config
    }

    /// Registers a new version.
    ///
    /// If `version.id` collides with an already-registered version (e.g. the
    /// same `DataVersion` was accidentally registered twice, or its `id`
    /// field -- which is public -- was set by hand to a colliding value), a
    /// fresh id is assigned instead of proceeding with the duplicate.
    /// Silently letting the collision through used to both overwrite the
    /// existing entry in `versions` *and* push a second, now-dangling
    /// reference to the same id onto `version_order`, so `list_versions()`
    /// returned one physical version twice while `versions.len()` no longer
    /// matched `version_order.len()`.
    pub fn register_version(&mut self, mut version: DataVersion) -> VersionId {
        if self.versions.contains_key(&version.id) {
            version.id = VersionId::new();
        }
        let id = version.id.clone();

        // Enforce max versions limit. `version.parents` is passed in
        // explicitly because this version -- and therefore its parent
        // links -- doesn't exist in `self.versions` yet at this point: the
        // eviction candidate search below only sees *already-registered*
        // parent links otherwise, so a version's own immediate parent
        // (the overwhelmingly common case: appending one new version to an
        // existing chain) would look "unreferenced" and be evicted right
        // out from under the version being registered this call.
        if self.versions.len() >= self.config.max_versions {
            // Remove oldest version that's not referenced
            if let Some(oldest) = self.find_oldest_unreferenced_version(&version.parents) {
                self.remove_version(&oldest);
            }
        }

        self.version_order.push(id.clone());
        self.versions.insert(id.clone(), version);
        id
    }

    /// Finds the oldest version that's neither a named reference, still a
    /// parent of some other live version, nor listed in `extra_protected`
    /// (the parents of a version currently being registered -- see the
    /// comment in [`Self::register_version`]).
    ///
    /// Excluding reachable parents is what keeps `get_lineage` honest: if a
    /// version were evicted while some other, still-tracked version listed
    /// it in `parents`, that other version's lineage would silently stop
    /// one hop early with no signal that anything was discarded.
    fn find_oldest_unreferenced_version(&self, extra_protected: &[VersionId]) -> Option<VersionId> {
        let named_refs: HashSet<&VersionId> = self.refs.values().collect();
        let referenced_as_parent: HashSet<&VersionId> = self
            .versions
            .values()
            .flat_map(|v| v.parents.iter())
            .collect();
        let extra_protected: HashSet<&VersionId> = extra_protected.iter().collect();

        for id in &self.version_order {
            if !named_refs.contains(id)
                && !referenced_as_parent.contains(id)
                && !extra_protected.contains(id)
            {
                return Some(id.clone());
            }
        }
        None
    }

    /// Removes a version and compacts operation storage so evicted
    /// versions don't leave their operations behind forever.
    fn remove_version(&mut self, version_id: &VersionId) {
        self.versions.remove(version_id);
        self.version_order.retain(|id| id != version_id);
        self.compact_operations();
    }

    /// Drops any operation whose input(s) *and* output are all no longer
    /// live, and rebuilds `operations_by_output`/`operations_by_input` with
    /// correct indices into the shrunk `operations` vec.
    ///
    /// `remove_version` previously only deleted the evicted id's own entry
    /// from the two index maps, leaving `operations` itself to grow
    /// forever regardless of `max_versions` -- eviction bounded the
    /// `versions` map but not the thing that usually dominates a lineage
    /// tracker's memory. An operation is kept as long as *either* endpoint
    /// (any input, or the output) is still queryable via `versions`, since
    /// [`Self::get_operations_using`]/[`Self::get_operations_producing`]
    /// can still be called on that live endpoint.
    ///
    /// Public as [`Self::compact`] for callers who want to reclaim memory
    /// on demand rather than only as a side effect of hitting `max_versions`.
    fn compact_operations(&mut self) {
        let live: HashSet<VersionId> = self.versions.keys().cloned().collect();
        let old_ops = std::mem::take(&mut self.operations);
        let mut new_by_output: HashMap<VersionId, Vec<usize>> = HashMap::new();
        let mut new_by_input: HashMap<VersionId, Vec<usize>> = HashMap::new();
        let mut new_ops = Vec::with_capacity(old_ops.len());

        for op in old_ops {
            let output_live = live.contains(&op.output);
            let any_input_live = op.inputs.iter().any(|i| live.contains(i));
            if !output_live && !any_input_live {
                continue;
            }
            let idx = new_ops.len();
            if output_live {
                new_by_output
                    .entry(op.output.clone())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
            for input in &op.inputs {
                if live.contains(input) {
                    new_by_input
                        .entry(input.clone())
                        .or_insert_with(Vec::new)
                        .push(idx);
                }
            }
            new_ops.push(op);
        }

        self.operations = new_ops;
        self.operations_by_output = new_by_output;
        self.operations_by_input = new_by_input;
    }

    /// Drop operations that are no longer reachable from any live version,
    /// on demand. See `Self::compact_operations` for what "reachable"
    /// means; `register_version` already calls this automatically when
    /// evicting a version to respect `max_versions`.
    pub fn compact(&mut self) {
        self.compact_operations();
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

    /// Exports the lineage graph as a DOT format string.
    ///
    /// Version ids and names are escaped before being interpolated into a
    /// quoted DOT string literal: an unescaped `"` or `\` in a version's
    /// `name` (an arbitrary caller-supplied string via `DataVersion::with_name`)
    /// would otherwise either produce invalid DOT or let a crafted name
    /// inject extra nodes/edges into the exported graph.
    pub fn export_dot(&self) -> String {
        let mut dot = String::from("digraph lineage {\n");
        dot.push_str("  rankdir=LR;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add version nodes
        for (id, version) in &self.versions {
            let label = version.name.as_deref().unwrap_or(id.as_str());
            let rows = version.schema.row_count;
            let cols = version.schema.columns.len();
            dot.push_str(&format!(
                "  \"{}\" [label=\"{}\\n({} rows, {} cols)\"];\n",
                escape_dot(id.as_str()),
                escape_dot(label),
                rows,
                cols
            ));
        }

        dot.push_str("\n");

        // Add edges for parent relationships
        for (id, version) in &self.versions {
            for parent_id in &version.parents {
                dot.push_str(&format!(
                    "  \"{}\" -> \"{}\";\n",
                    escape_dot(parent_id.as_str()),
                    escape_dot(id.as_str())
                ));
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
    pub fn register_version(&self, version: DataVersion) -> crate::error::Result<VersionId> {
        Ok(write_lock_safe!(self.inner, "version tracker inner write")?.register_version(version))
    }

    /// Gets a version by ID
    pub fn get_version(&self, id: &VersionId) -> Option<DataVersion> {
        read_lock_safe!(self.inner, "version tracker inner read")
            .ok()?
            .get_version(id)
            .cloned()
    }

    /// Records an operation
    pub fn record_operation(&self, operation: Operation) -> crate::error::Result<()> {
        write_lock_safe!(self.inner, "version tracker inner write")?.record_operation(operation);
        Ok(())
    }

    /// Sets a reference
    pub fn set_ref(&self, name: &str, version_id: VersionId) -> Result<(), VersioningError> {
        write_lock_safe!(self.inner, "version tracker inner write")
            .map_err(|_| VersioningError::StorageError("failed to acquire lock".to_string()))?
            .set_ref(name, version_id)
    }

    /// Gets stats
    pub fn stats(&self) -> crate::error::Result<TrackerStats> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?.stats())
    }

    // -- The following forward LineageTracker's remaining read (and the
    // -- `clear` write) methods. Earlier, only register_version/get_version/
    // -- record_operation/set_ref/stats were exposed, so a caller using only
    // -- the thread-safe wrapper had no way to reach `diff`, `get_lineage`,
    // -- or `export_dot` at all -- the "thread-safe" tracker's headline
    // -- lineage features were single-threaded-only in practice. Each
    // -- returns owned data (clones) since the lock guard doesn't outlive
    // -- the call.

    /// Gets a version by reference name
    pub fn get_version_by_ref(&self, ref_name: &str) -> crate::error::Result<Option<DataVersion>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .get_version_by_ref(ref_name)
            .cloned())
    }

    /// Gets a reference's version id
    pub fn get_ref(&self, name: &str) -> crate::error::Result<Option<VersionId>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .get_ref(name)
            .cloned())
    }

    /// Lists all references
    pub fn list_refs(&self) -> crate::error::Result<Vec<(String, VersionId)>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .list_refs()
            .into_iter()
            .map(|(name, id)| (name.to_string(), id.clone()))
            .collect())
    }

    /// Gets all operations that produced a version
    pub fn get_operations_producing(
        &self,
        version_id: &VersionId,
    ) -> crate::error::Result<Vec<Operation>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .get_operations_producing(version_id)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Gets all operations that used a version as input
    pub fn get_operations_using(
        &self,
        version_id: &VersionId,
    ) -> crate::error::Result<Vec<Operation>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .get_operations_using(version_id)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Gets the full lineage of a version (all ancestor versions)
    pub fn get_lineage(&self, version_id: &VersionId) -> crate::error::Result<Vec<DataVersion>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .get_lineage(version_id)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Gets all operations in the lineage of a version
    pub fn get_operation_history(
        &self,
        version_id: &VersionId,
    ) -> crate::error::Result<Vec<Operation>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .get_operation_history(version_id)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Computes the diff between two versions
    pub fn diff(
        &self,
        from_id: &VersionId,
        to_id: &VersionId,
    ) -> crate::error::Result<VersionDiff> {
        let guard = read_lock_safe!(self.inner, "version tracker inner read")?;
        guard
            .diff(from_id, to_id)
            .map_err(|e| crate::core::error::Error::InvalidOperation(e.to_string()))
    }

    /// Lists all versions
    pub fn list_versions(&self) -> crate::error::Result<Vec<DataVersion>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .list_versions()
            .into_iter()
            .cloned()
            .collect())
    }

    /// Lists versions by tag
    pub fn list_versions_by_tag(&self, tag: &str) -> crate::error::Result<Vec<DataVersion>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .list_versions_by_tag(tag)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Searches versions by name/description pattern
    pub fn search_versions(&self, pattern: &str) -> crate::error::Result<Vec<DataVersion>> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?
            .search_versions(pattern)
            .into_iter()
            .cloned()
            .collect())
    }

    /// Exports the lineage graph as a DOT format string
    pub fn export_dot(&self) -> crate::error::Result<String> {
        Ok(read_lock_safe!(self.inner, "version tracker inner read")?.export_dot())
    }

    /// Drops operations no longer reachable from any live version (see
    /// [`LineageTracker::compact`])
    pub fn compact(&self) -> crate::error::Result<()> {
        write_lock_safe!(self.inner, "version tracker inner write")?.compact();
        Ok(())
    }

    /// Clears all data
    pub fn clear(&self) -> crate::error::Result<()> {
        write_lock_safe!(self.inner, "version tracker inner write")?.clear();
        Ok(())
    }
}

impl Default for SharedLineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::core::{DataSchema, OperationType};
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

        tracker
            .set_ref("latest", id.clone())
            .expect("operation should succeed");

        let ref_version = tracker.get_version_by_ref("latest");
        assert!(ref_version.is_some());
        assert_eq!(ref_version.expect("operation should succeed").id, id);
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

        let diff = tracker.diff(&v1, &v2).expect("operation should succeed");

        assert!(diff.columns_added.contains(&"c".to_string()));
        assert!(diff.columns_removed.contains(&"b".to_string()));
    }

    #[test]
    fn test_export_dot() {
        let mut tracker = LineageTracker::new();

        let v1 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "b"])).with_name("source"),
        );
        let _v2 = tracker.register_version(
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
    fn test_eviction_never_drops_a_reachable_parent() {
        let mut tracker = LineageTracker::with_config(LineageConfig {
            max_versions: 2,
            ..LineageConfig::default()
        });

        let v1 = tracker
            .register_version(DataVersion::new(create_test_schema(&["a"])).with_name("root"));
        let v2 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "b"]))
                .with_name("mid")
                .with_parents(vec![v1.clone()]),
        );
        // Registering v3 pushes past max_versions=2; naive eviction would
        // pick v1 (oldest, unreferenced by name) even though v2.parents
        // still points at it.
        let v3 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "b", "c"]))
                .with_name("leaf")
                .with_parents(vec![v2.clone()]),
        );

        assert!(
            tracker.get_version(&v1).is_some(),
            "v1 is still v2's parent and must not be silently evicted"
        );
        let lineage = tracker.get_lineage(&v3);
        assert_eq!(lineage.len(), 3, "lineage must not be truncated");
    }

    #[test]
    fn test_eviction_still_reclaims_a_genuinely_disconnected_version() {
        // Sanity check that the parent-preserving fix doesn't disable
        // eviction outright: a version with no name, no children, and no
        // relation to the version currently being registered is still a
        // legitimate eviction candidate.
        let mut tracker = LineageTracker::with_config(LineageConfig {
            max_versions: 2,
            ..LineageConfig::default()
        });

        let orphan = tracker.register_version(DataVersion::new(create_test_schema(&["a"])));
        let _kept = tracker.register_version(DataVersion::new(create_test_schema(&["b"])));
        // A brand new, unrelated root (no parents) pushes past max_versions
        // with nothing connecting it to `orphan`.
        let _new_root = tracker.register_version(DataVersion::new(create_test_schema(&["c"])));

        assert!(
            tracker.get_version(&orphan).is_none(),
            "a version with no name/children/relation to the new one is still evictable"
        );
        assert_eq!(tracker.list_versions().len(), 2);
    }

    #[test]
    fn test_register_version_duplicate_id_gets_fresh_id() {
        let mut tracker = LineageTracker::new();
        let version = DataVersion::new(create_test_schema(&["a"]));

        let id1 = tracker.register_version(version.clone());
        // Registering the *same* DataVersion object again (same `.id`) must
        // not silently overwrite the first entry or duplicate `version_order`.
        let id2 = tracker.register_version(version);

        assert_ne!(id1, id2, "a colliding id must be reassigned, not reused");
        assert_eq!(tracker.list_versions().len(), 2);
        assert!(tracker.get_version(&id1).is_some());
        assert!(tracker.get_version(&id2).is_some());
    }

    #[test]
    fn test_remove_version_compacts_orphaned_operations_but_keeps_reachable_ones() {
        let mut tracker = LineageTracker::new();
        let v1 = tracker.register_version(DataVersion::new(create_test_schema(&["a"])));
        let v2 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "b"])).with_parents(vec![v1.clone()]),
        );
        let v3 = tracker.register_version(
            DataVersion::new(create_test_schema(&["a", "b", "c"])).with_parents(vec![v2.clone()]),
        );

        tracker.record_operation(Operation::new(
            OperationType::AddColumn {
                column_name: "b".to_string(),
            },
            vec![v1.clone()],
            v2.clone(),
        ));
        tracker.record_operation(Operation::new(
            OperationType::AddColumn {
                column_name: "c".to_string(),
            },
            vec![v2.clone()],
            v3.clone(),
        ));
        assert_eq!(tracker.operations.len(), 2);

        // `remove_version` is module-private, reachable directly from this
        // in-file test module. Removing v1: the v1->v2 operation's output
        // (v2) is still live, so it must survive compaction.
        tracker.remove_version(&v1);
        assert_eq!(
            tracker.operations.len(),
            2,
            "the v1->v2 op has a live output (v2) and must not be dropped"
        );
        assert_eq!(tracker.get_operations_producing(&v2).len(), 1);

        // Now remove v2 too: the v1->v2 operation has neither a live input
        // nor a live output any more, so it must finally be compacted away.
        // v2->v3 keeps its live output (v3) and survives.
        tracker.remove_version(&v2);
        assert_eq!(
            tracker.operations.len(),
            1,
            "v1->v2 has no live endpoint left and must be compacted away"
        );
        assert_eq!(tracker.get_operations_producing(&v3).len(), 1);
    }

    #[test]
    fn test_export_dot_escapes_special_characters() {
        let mut tracker = LineageTracker::new();
        tracker.register_version(
            DataVersion::new(create_test_schema(&["a"])).with_name("evil\" -> \"injected"),
        );

        let dot = tracker.export_dot();
        // The raw injection payload must not appear unescaped.
        assert!(!dot.contains("evil\" -> \"injected"));
        assert!(dot.contains("evil\\\" -> \\\"injected"));
    }

    #[test]
    fn test_shared_tracker_forwards_lineage_diff_and_export() {
        let tracker = SharedLineageTracker::new();

        let v1 = tracker
            .register_version(DataVersion::new(create_test_schema(&["a", "b"])).with_name("v1"))
            .expect("register v1");
        let v2 = tracker
            .register_version(
                DataVersion::new(create_test_schema(&["a"]))
                    .with_name("v2")
                    .with_parents(vec![v1.clone()]),
            )
            .expect("register v2");

        let lineage = tracker.get_lineage(&v2).expect("get_lineage");
        assert_eq!(lineage.len(), 2);

        let diff = tracker.diff(&v1, &v2).expect("diff");
        assert!(diff.columns_removed.contains(&"b".to_string()));

        let dot = tracker.export_dot().expect("export_dot");
        assert!(dot.contains("digraph"));

        let versions = tracker.list_versions().expect("list_versions");
        assert_eq!(versions.len(), 2);

        tracker.clear().expect("clear");
        assert!(tracker.list_versions().expect("list_versions").is_empty());
    }

    #[test]
    fn test_shared_tracker() {
        let tracker = SharedLineageTracker::new();

        let version = DataVersion::new(create_test_schema(&["a", "b"]));
        let id = tracker
            .register_version(version)
            .expect("operation should succeed");

        assert!(tracker.get_version(&id).is_some());
    }
}
