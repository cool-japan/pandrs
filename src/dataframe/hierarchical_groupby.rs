//! Hierarchical GroupBy functionality for DataFrames with multi-level grouping and nested operations
//!
//! This module provides advanced hierarchical grouping capabilities, allowing for:
//! - Multi-level group hierarchies with nested structure
//! - Hierarchical aggregation results with multi-index columns
//! - Group navigation and metadata management
//! - Nested group operations across different hierarchy levels
//! - Performance-optimized tree-based group storage

use std::collections::{BTreeMap, HashMap};
use std::fmt::Debug;
use std::sync::Arc;

use crate::core::error::{Error, Result};
use crate::dataframe::base::DataFrame;
use crate::dataframe::groupby::{AggFunc, CustomAggFn, NamedAgg};
use crate::dataframe::multi_index_results::{MultiIndexColumn, MultiIndexDataFrame, ToMultiIndex};
use crate::series::base::Series;

/// Represents a hierarchical key structure for multi-level grouping
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HierarchicalKey {
    /// Values at each level of the hierarchy
    pub levels: Vec<String>,
    /// Level names corresponding to grouping columns
    pub level_names: Vec<String>,
}

impl HierarchicalKey {
    /// Create a new hierarchical key
    pub fn new(levels: Vec<String>, level_names: Vec<String>) -> Self {
        Self {
            levels,
            level_names,
        }
    }

    /// Get the value at a specific level
    pub fn get_level(&self, level: usize) -> Option<&str> {
        self.levels.get(level).map(|s| s.as_str())
    }

    /// Get the key up to a specific level (partial key)
    pub fn partial_key(&self, up_to_level: usize) -> Self {
        let levels = self.levels.iter().take(up_to_level + 1).cloned().collect();
        let level_names = self
            .level_names
            .iter()
            .take(up_to_level + 1)
            .cloned()
            .collect();
        Self::new(levels, level_names)
    }

    /// Get the depth of this hierarchical key
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Create a display string for this key
    pub fn display_string(&self) -> String {
        self.levels.join(" | ")
    }

    /// Create a tuple representation for compatibility
    pub fn as_tuple(&self) -> Vec<&str> {
        self.levels.iter().map(|s| s.as_str()).collect()
    }
}

/// Tree node representing a group in the hierarchy
#[derive(Debug, Clone)]
pub struct GroupNode {
    /// Key for this level
    pub key: String,
    /// Full hierarchical key path to this node
    pub full_key: HierarchicalKey,
    /// Row indices belonging to this group
    pub indices: Vec<usize>,
    /// Child nodes (subgroups)
    pub children: BTreeMap<String, GroupNode>,
    /// Level in the hierarchy (0 = root level)
    pub level: usize,
    /// Whether this is a leaf node (has no children)
    pub is_leaf: bool,
    /// Aggregated values cached at this node
    pub cached_aggregations: HashMap<String, f64>,
}

impl GroupNode {
    /// Create a new group node
    pub fn new(key: String, full_key: HierarchicalKey, level: usize) -> Self {
        Self {
            key,
            full_key,
            indices: Vec::new(),
            children: BTreeMap::new(),
            level,
            is_leaf: true,
            cached_aggregations: HashMap::new(),
        }
    }

    /// Add a child node to this group
    pub fn add_child(&mut self, child_key: String, child_node: GroupNode) {
        self.children.insert(child_key, child_node);
        self.is_leaf = false;
    }

    /// Get the number of direct children
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Get total number of rows in this group and all subgroups
    pub fn total_size(&self) -> usize {
        let mut size = self.indices.len();
        for child in self.children.values() {
            size += child.total_size();
        }
        size
    }

    /// Get all leaf nodes under this node
    pub fn get_leaf_nodes(&self) -> Vec<&GroupNode> {
        if self.is_leaf {
            vec![self]
        } else {
            let mut leaves = Vec::new();
            for child in self.children.values() {
                leaves.extend(child.get_leaf_nodes());
            }
            leaves
        }
    }

    /// Navigate to a specific child by path
    pub fn navigate_path(&self, path: &[&str]) -> Option<&GroupNode> {
        if path.is_empty() {
            return Some(self);
        }

        if let Some(child) = self.children.get(path[0]) {
            child.navigate_path(&path[1..])
        } else {
            None
        }
    }

    /// Get all nodes at a specific level relative to this node
    pub fn get_nodes_at_level(&self, target_level: usize) -> Vec<&GroupNode> {
        if self.level == target_level {
            vec![self]
        } else if self.level < target_level {
            let mut nodes = Vec::new();
            for child in self.children.values() {
                nodes.extend(child.get_nodes_at_level(target_level));
            }
            nodes
        } else {
            Vec::new()
        }
    }
}

/// Hierarchical aggregation specification
#[derive(Clone)]
pub struct HierarchicalAgg {
    /// Column to aggregate
    pub column: String,
    /// Aggregation functions to apply at each level
    pub level_functions: Vec<(usize, AggFunc, String)>, // (level, function, alias)
    /// Custom aggregation function (if using AggFunc::Custom)
    pub custom_fn: Option<CustomAggFn>,
    /// Whether to propagate aggregations up the hierarchy
    pub propagate_up: bool,
    /// Whether to compute aggregations for intermediate levels
    pub include_intermediate: bool,
}

impl std::fmt::Debug for HierarchicalAgg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HierarchicalAgg")
            .field("column", &self.column)
            .field("level_functions", &self.level_functions)
            .field(
                "custom_fn",
                &self.custom_fn.as_ref().map(|_| "<custom_function>"),
            )
            .field("propagate_up", &self.propagate_up)
            .field("include_intermediate", &self.include_intermediate)
            .finish()
    }
}

impl HierarchicalAgg {
    /// Create a new hierarchical aggregation
    pub fn new(column: String) -> Self {
        Self {
            column,
            level_functions: Vec::new(),
            custom_fn: None,
            propagate_up: false,
            include_intermediate: true,
        }
    }

    /// Add an aggregation function for a specific level
    pub fn add_level_agg(mut self, level: usize, func: AggFunc, alias: String) -> Self {
        self.level_functions.push((level, func, alias));
        self
    }

    /// Enable propagation of aggregations up the hierarchy
    pub fn with_propagation(mut self) -> Self {
        self.propagate_up = true;
        self
    }

    /// Set custom aggregation function
    pub fn with_custom<F>(mut self, func: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.custom_fn = Some(Arc::new(func));
        self
    }
}

/// Group hierarchy tree for efficient navigation and operations
#[derive(Debug)]
pub struct GroupHierarchy {
    /// Root nodes of the hierarchy tree
    pub roots: BTreeMap<String, GroupNode>,
    /// Grouping column names at each level
    pub level_columns: Vec<String>,
    /// Maximum depth of the hierarchy
    pub max_depth: usize,
    /// Total number of groups across all levels
    pub total_groups: usize,
    /// Index mapping for fast lookups
    pub level_index: HashMap<usize, Vec<HierarchicalKey>>,
}

impl GroupHierarchy {
    /// Create a new group hierarchy
    pub fn new(level_columns: Vec<String>) -> Self {
        Self {
            roots: BTreeMap::new(),
            level_columns,
            max_depth: 0,
            total_groups: 0,
            level_index: HashMap::new(),
        }
    }

    /// Build the hierarchy from data
    pub fn build_from_data(&mut self, df: &DataFrame) -> Result<()> {
        // Verify all grouping columns exist
        for col in &self.level_columns {
            if !df.contains_column(col) {
                return Err(Error::ColumnNotFound(col.clone()));
            }
        }

        let num_rows = df.row_count();
        self.max_depth = self.level_columns.len();

        // Collect all data first to avoid borrowing issues
        let mut all_row_data = Vec::new();
        for row_idx in 0..num_rows {
            let mut row_path = Vec::new();
            for col_name in &self.level_columns {
                let col_values = df.get_column_string_values(col_name)?;
                let value = if row_idx < col_values.len() {
                    col_values[row_idx].clone()
                } else {
                    "NULL".to_string()
                };
                row_path.push(value);
            }
            all_row_data.push((row_idx, row_path));
        }

        // Now build the hierarchy
        for (row_idx, row_path) in all_row_data {
            self.add_row_to_hierarchy(row_idx, &row_path)?;
        }

        // Remove duplicates from level index and sort
        for level_keys in self.level_index.values_mut() {
            level_keys.sort();
            level_keys.dedup();
        }

        // Update leaf status for all nodes
        self.update_leaf_status();

        Ok(())
    }

    /// Add a row to the hierarchy
    fn add_row_to_hierarchy(&mut self, row_idx: usize, row_path: &[String]) -> Result<()> {
        for (level, value) in row_path.iter().enumerate() {
            let current_path = row_path[..=level].to_vec();
            let hierarchical_key =
                HierarchicalKey::new(current_path.clone(), self.level_columns[..=level].to_vec());

            // Update level index
            self.level_index
                .entry(level)
                .or_default()
                .push(hierarchical_key.clone());

            // Create or update the node
            self.create_or_update_node(&current_path, level, row_idx);
        }
        Ok(())
    }

    /// Create or update a node in the hierarchy
    fn create_or_update_node(&mut self, path: &[String], level: usize, row_idx: usize) {
        if path.is_empty() {
            return;
        }

        if path.len() == 1 {
            // Root level
            let key = &path[0];
            let hierarchical_key =
                HierarchicalKey::new(path.to_vec(), self.level_columns[..=level].to_vec());

            let node = self.roots.entry(key.clone()).or_insert_with(|| {
                self.total_groups += 1;
                GroupNode::new(key.clone(), hierarchical_key, level)
            });
            node.indices.push(row_idx);
        } else {
            // Handle multi-level path
            let parent_path = &path[..path.len() - 1];
            let child_key = &path[path.len() - 1];

            // Pre-calculate the hierarchical key to avoid borrowing issues
            let hierarchical_key =
                HierarchicalKey::new(path.to_vec(), self.level_columns[..=level].to_vec());

            // Check if parent exists, if not, return early
            if !self.path_exists(parent_path) {
                return;
            }

            // Navigate and update
            let mut current_map = &mut self.roots;
            for segment in parent_path {
                if let Some(node) = current_map.get_mut(segment) {
                    current_map = &mut node.children;
                } else {
                    return; // Parent doesn't exist
                }
            }

            // Now we're at the parent level, create/update child
            let child_node = current_map.entry(child_key.clone()).or_insert_with(|| {
                self.total_groups += 1;
                GroupNode::new(child_key.clone(), hierarchical_key, level)
            });
            child_node.indices.push(row_idx);
        }
    }

    /// Check if a path exists in the hierarchy
    fn path_exists(&self, path: &[String]) -> bool {
        if path.is_empty() {
            return true;
        }

        let mut current_map = &self.roots;
        for segment in path {
            if let Some(node) = current_map.get(segment) {
                current_map = &node.children;
            } else {
                return false;
            }
        }
        true
    }

    /// Find a mutable node by path
    fn find_node_mut(&mut self, path: &[String]) -> Option<&mut GroupNode> {
        if path.is_empty() {
            return None;
        }

        let mut current_node = self.roots.get_mut(&path[0])?;

        for key in &path[1..] {
            current_node = current_node.children.get_mut(key)?;
        }

        Some(current_node)
    }

    /// Update leaf status for all nodes in the hierarchy
    fn update_leaf_status(&mut self) {
        fn update_node_recursive(node: &mut GroupNode) {
            node.is_leaf = node.children.is_empty();
            for child in node.children.values_mut() {
                update_node_recursive(child);
            }
        }

        for root in self.roots.values_mut() {
            update_node_recursive(root);
        }
    }

    /// Get all nodes at a specific level
    pub fn get_level_nodes(&self, level: usize) -> Vec<&GroupNode> {
        if level == 0 {
            self.roots.values().collect()
        } else {
            let mut nodes = Vec::new();
            for root in self.roots.values() {
                nodes.extend(root.get_nodes_at_level(level));
            }
            nodes
        }
    }

    /// Get all leaf nodes in the hierarchy
    pub fn get_all_leaf_nodes(&self) -> Vec<&GroupNode> {
        let mut leaves = Vec::new();
        for root in self.roots.values() {
            leaves.extend(root.get_leaf_nodes());
        }
        leaves
    }

    /// Navigate to a specific group by hierarchical key
    pub fn navigate_to_group(&self, key: &HierarchicalKey) -> Option<&GroupNode> {
        if key.levels.is_empty() {
            return None;
        }

        if let Some(root) = self.roots.get(&key.levels[0]) {
            if key.levels.len() == 1 {
                Some(root)
            } else {
                let path: Vec<&str> = key.levels[1..].iter().map(|s| s.as_str()).collect();
                root.navigate_path(&path)
            }
        } else {
            None
        }
    }

    /// Get statistics about the hierarchy
    pub fn get_statistics(&self) -> HierarchyStatistics {
        let mut stats = HierarchyStatistics {
            total_levels: self.max_depth,
            total_groups: self.total_groups,
            groups_per_level: Vec::new(),
            leaf_groups: 0,
            max_group_size: 0,
            min_group_size: usize::MAX,
            avg_group_size: 0.0,
        };

        // Calculate statistics for each level
        for level in 0..self.max_depth {
            let level_nodes = self.get_level_nodes(level);
            stats.groups_per_level.push(level_nodes.len());

            for node in level_nodes {
                let group_size = node.indices.len();
                stats.max_group_size = stats.max_group_size.max(group_size);
                stats.min_group_size = stats.min_group_size.min(group_size);

                if node.is_leaf {
                    stats.leaf_groups += 1;
                }
            }
        }

        // Calculate average group size
        if self.total_groups > 0 {
            let total_size: usize = self
                .get_all_leaf_nodes()
                .iter()
                .map(|node| node.indices.len())
                .sum();
            stats.avg_group_size = total_size as f64 / stats.leaf_groups as f64;
        }

        if stats.min_group_size == usize::MAX {
            stats.min_group_size = 0;
        }

        stats
    }
}

/// Statistics about a group hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyStatistics {
    /// Total number of levels in the hierarchy
    pub total_levels: usize,
    /// Total number of groups across all levels
    pub total_groups: usize,
    /// Number of groups at each level
    pub groups_per_level: Vec<usize>,
    /// Number of leaf groups (groups with no children)
    pub leaf_groups: usize,
    /// Maximum group size (number of rows)
    pub max_group_size: usize,
    /// Minimum group size (number of rows)
    pub min_group_size: usize,
    /// Average group size
    pub avg_group_size: f64,
}

/// Hierarchical DataFrame GroupBy with multi-level operations
#[derive(Debug)]
pub struct HierarchicalDataFrameGroupBy {
    /// Original DataFrame
    df: DataFrame,
    /// Group hierarchy tree
    hierarchy: GroupHierarchy,
    /// Current navigation context (which level/group we're focused on)
    current_context: GroupNavigationContext,
}

/// Navigation context for hierarchical operations
#[derive(Debug, Clone)]
pub struct GroupNavigationContext {
    /// Current level being operated on (None = all levels)
    pub current_level: Option<usize>,
    /// Current group path (None = all groups at current level)
    pub current_path: Option<Vec<String>>,
    /// Whether to include child groups in operations
    pub include_children: bool,
    /// Whether to include parent context in results
    pub include_parents: bool,
}

impl GroupNavigationContext {
    /// Create a new navigation context
    pub fn new() -> Self {
        Self {
            current_level: None,
            current_path: None,
            include_children: true,
            include_parents: false,
        }
    }

    /// Focus on a specific level
    pub fn at_level(mut self, level: usize) -> Self {
        self.current_level = Some(level);
        self
    }

    /// Navigate to a specific group path
    pub fn at_path(mut self, path: Vec<String>) -> Self {
        self.current_path = Some(path);
        self
    }

    /// Include child groups in operations
    pub fn with_children(mut self) -> Self {
        self.include_children = true;
        self
    }

    /// Include parent context in results
    pub fn with_parents(mut self) -> Self {
        self.include_parents = true;
        self
    }
}

impl Default for GroupNavigationContext {
    fn default() -> Self {
        Self::new()
    }
}

impl HierarchicalDataFrameGroupBy {
    /// Create a new hierarchical DataFrame GroupBy
    pub fn new(df: DataFrame, group_columns: Vec<String>) -> Result<Self> {
        let mut hierarchy = GroupHierarchy::new(group_columns);
        hierarchy.build_from_data(&df)?;

        Ok(Self {
            df,
            hierarchy,
            current_context: GroupNavigationContext::new(),
        })
    }

    /// Get the hierarchy statistics
    pub fn hierarchy_stats(&self) -> HierarchyStatistics {
        self.hierarchy.get_statistics()
    }

    /// Navigate to a specific level in the hierarchy
    pub fn level(&mut self, level: usize) -> &mut Self {
        self.current_context = self.current_context.clone().at_level(level);
        self
    }

    /// Navigate to a specific group path
    pub fn group_path(&mut self, path: Vec<String>) -> &mut Self {
        self.current_context = self.current_context.clone().at_path(path);
        self
    }

    /// Include child groups in operations
    pub fn with_children(&mut self) -> &mut Self {
        self.current_context = self.current_context.clone().with_children();
        self
    }

    /// Include parent context in results
    pub fn with_parents(&mut self) -> &mut Self {
        self.current_context = self.current_context.clone().with_parents();
        self
    }

    /// Get all group keys at the current navigation context
    pub fn get_group_keys(&self) -> Vec<HierarchicalKey> {
        match (
            &self.current_context.current_level,
            &self.current_context.current_path,
        ) {
            (Some(level), None) => {
                // All groups at a specific level
                self.hierarchy
                    .get_level_nodes(*level)
                    .into_iter()
                    .map(|node| node.full_key.clone())
                    .collect()
            }
            (None, Some(path)) => {
                // Specific group and potentially its children
                if let Some(node) = self.navigate_to_group_by_path(path) {
                    if self.current_context.include_children {
                        let mut keys = vec![node.full_key.clone()];
                        for child in node.children.values() {
                            keys.extend(self.collect_all_child_keys(child));
                        }
                        keys
                    } else {
                        vec![node.full_key.clone()]
                    }
                } else {
                    Vec::new()
                }
            }
            (Some(level), Some(path)) => {
                // Specific group at a specific level
                if let Some(node) = self.navigate_to_group_by_path(path) {
                    if node.level == *level {
                        vec![node.full_key.clone()]
                    } else {
                        node.get_nodes_at_level(*level)
                            .into_iter()
                            .map(|n| n.full_key.clone())
                            .collect()
                    }
                } else {
                    Vec::new()
                }
            }
            (None, None) => {
                // All groups (leaf groups)
                self.hierarchy
                    .get_all_leaf_nodes()
                    .into_iter()
                    .map(|node| node.full_key.clone())
                    .collect()
            }
        }
    }

    /// Navigate to a group by path
    fn navigate_to_group_by_path(&self, path: &[String]) -> Option<&GroupNode> {
        let key = HierarchicalKey::new(
            path.to_vec(),
            self.hierarchy.level_columns[..path.len()].to_vec(),
        );
        self.hierarchy.navigate_to_group(&key)
    }

    /// Collect all child keys recursively
    fn collect_all_child_keys(&self, node: &GroupNode) -> Vec<HierarchicalKey> {
        let mut keys = vec![node.full_key.clone()];
        for child in node.children.values() {
            keys.extend(self.collect_all_child_keys(child));
        }
        keys
    }

    /// Apply hierarchical aggregations
    pub fn agg_hierarchical(&self, hierarchical_aggs: Vec<HierarchicalAgg>) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        // Add hierarchical key columns
        let group_keys = self.get_group_keys();
        if group_keys.is_empty() {
            return Ok(result);
        }

        let max_levels = group_keys.iter().map(|k| k.levels.len()).max().unwrap_or(0);

        // Create columns for each hierarchy level
        for level in 0..max_levels {
            let level_name = self
                .hierarchy
                .level_columns
                .get(level)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("level_{}", level));

            let level_values: Vec<String> = group_keys
                .iter()
                .map(|key| key.get_level(level).unwrap_or("").to_string())
                .collect();

            let level_series = Series::new(level_values, Some(level_name.clone()))?;
            result.add_column(level_name, level_series)?;
        }

        // Apply aggregations
        for agg in &hierarchical_aggs {
            for (level, func, alias) in &agg.level_functions {
                let mut agg_values = Vec::new();

                for key in &group_keys {
                    if let Some(node) = self.hierarchy.navigate_to_group(key) {
                        let agg_result = self.calculate_hierarchical_aggregation(
                            &agg.column,
                            *func,
                            node,
                            &agg.custom_fn,
                            agg.propagate_up,
                        )?;
                        agg_values.push(agg_result.to_string());
                    } else {
                        agg_values.push("NaN".to_string());
                    }
                }

                let agg_series = Series::new(agg_values, Some(alias.clone()))?;
                result.add_column(alias.clone(), agg_series)?;
            }
        }

        Ok(result)
    }

    /// Calculate aggregation for a specific node
    fn calculate_hierarchical_aggregation(
        &self,
        column: &str,
        func: AggFunc,
        node: &GroupNode,
        custom_fn: &Option<CustomAggFn>,
        propagate_up: bool,
    ) -> Result<f64> {
        // Get indices for aggregation
        let indices = if propagate_up {
            // Include all child indices
            self.collect_all_indices(node)
        } else {
            // Only direct indices
            node.indices.clone()
        };

        // Extract column values
        let column_values = self.df.get_column_string_values(column)?;
        let group_values: Vec<f64> = indices
            .iter()
            .filter_map(|&idx| {
                if idx < column_values.len() {
                    column_values[idx].parse::<f64>().ok()
                } else {
                    None
                }
            })
            .collect();

        if group_values.is_empty() {
            return Ok(f64::NAN);
        }

        // Apply aggregation function (reuse logic from basic GroupBy)
        match func {
            AggFunc::Sum => Ok(group_values.iter().sum()),
            AggFunc::Mean => Ok(group_values.iter().sum::<f64>() / group_values.len() as f64),
            AggFunc::Count => Ok(group_values.len() as f64),
            AggFunc::Min => Ok(group_values.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
            AggFunc::Max => Ok(group_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            AggFunc::Std => {
                if group_values.len() <= 1 {
                    Ok(0.0)
                } else {
                    let mean = group_values.iter().sum::<f64>() / group_values.len() as f64;
                    let variance = group_values
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (group_values.len() - 1) as f64;
                    Ok(variance.sqrt())
                }
            }
            AggFunc::Custom => {
                if let Some(custom_fn) = custom_fn {
                    Ok(custom_fn(&group_values))
                } else {
                    Err(Error::InvalidValue(
                        "Custom function not provided".to_string(),
                    ))
                }
            }
            _ => Err(Error::NotImplemented(format!(
                "Aggregation function {:?} not yet implemented for hierarchical groupby",
                func
            ))),
        }
    }

    /// Collect all indices from a node and its children
    fn collect_all_indices(&self, node: &GroupNode) -> Vec<usize> {
        let mut all_indices = node.indices.clone();
        for child in node.children.values() {
            all_indices.extend(self.collect_all_indices(child));
        }
        all_indices.sort();
        all_indices.dedup();
        all_indices
    }

    /// Get the size of each group in the hierarchy
    pub fn size(&self) -> Result<DataFrame> {
        let mut result = DataFrame::new();

        let group_keys = self.get_group_keys();
        if group_keys.is_empty() {
            return Ok(result);
        }

        let max_levels = group_keys.iter().map(|k| k.levels.len()).max().unwrap_or(0);

        // Create columns for each hierarchy level
        for level in 0..max_levels {
            let level_name = self
                .hierarchy
                .level_columns
                .get(level)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("level_{}", level));

            let level_values: Vec<String> = group_keys
                .iter()
                .map(|key| key.get_level(level).unwrap_or("").to_string())
                .collect();

            let level_series = Series::new(level_values, Some(level_name.clone()))?;
            result.add_column(level_name, level_series)?;
        }

        // Add size column
        let sizes: Vec<String> = group_keys
            .iter()
            .map(|key| {
                if let Some(node) = self.hierarchy.navigate_to_group(key) {
                    if self.current_context.include_children {
                        node.total_size().to_string()
                    } else {
                        node.indices.len().to_string()
                    }
                } else {
                    "0".to_string()
                }
            })
            .collect();

        let size_series = Series::new(sizes, Some("size".to_string()))?;
        result.add_column("size".to_string(), size_series)?;

        Ok(result)
    }

    /// Apply hierarchical aggregations and return multi-index result
    pub fn agg_hierarchical_multi_index(
        &self,
        hierarchical_aggs: Vec<HierarchicalAgg>,
    ) -> Result<MultiIndexDataFrame> {
        // First get the regular hierarchical result
        let regular_result = self.agg_hierarchical(hierarchical_aggs.clone())?;

        // Convert to multi-index format
        regular_result.to_multi_index(&hierarchical_aggs)
    }

    /// Cross-level aggregation: aggregate a column across different hierarchy levels
    pub fn cross_level_agg(
        &self,
        column: &str,
        func: AggFunc,
        source_level: usize,
        target_level: usize,
    ) -> Result<DataFrame> {
        if source_level >= self.hierarchy.max_depth || target_level >= self.hierarchy.max_depth {
            return Err(Error::InvalidValue(
                "Level exceeds hierarchy depth".to_string(),
            ));
        }

        let mut result = DataFrame::new();

        // Get all groups at the target level
        let target_groups = self.hierarchy.get_level_nodes(target_level);

        // Build result columns for target level
        for level in 0..=target_level {
            let level_name = self
                .hierarchy
                .level_columns
                .get(level)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("level_{}", level));

            let level_values: Vec<String> = target_groups
                .iter()
                .map(|node| node.full_key.get_level(level).unwrap_or("").to_string())
                .collect();

            let level_series = Series::new(level_values, Some(level_name.clone()))?;
            result.add_column(level_name, level_series)?;
        }

        // Calculate cross-level aggregations
        let mut agg_values = Vec::new();
        for target_node in &target_groups {
            // Get all source-level groups under this target node
            let source_nodes = if source_level > target_level {
                target_node.get_nodes_at_level(source_level)
            } else if source_level == target_level {
                vec![*target_node]
            } else {
                // Source level is higher (closer to root), use parent navigation
                let parent_key = target_node.full_key.partial_key(source_level);
                if let Some(parent_node) = self.hierarchy.navigate_to_group(&parent_key) {
                    vec![parent_node]
                } else {
                    Vec::new()
                }
            };

            // Collect all values from source nodes
            let mut all_values = Vec::new();
            for source_node in &source_nodes {
                let column_values = self.df.get_column_string_values(column)?;
                for &idx in &source_node.indices {
                    if idx < column_values.len() {
                        if let Ok(val) = column_values[idx].parse::<f64>() {
                            all_values.push(val);
                        }
                    }
                }
            }

            // Apply aggregation
            let agg_result = if all_values.is_empty() {
                f64::NAN
            } else {
                match func {
                    AggFunc::Sum => all_values.iter().sum(),
                    AggFunc::Mean => all_values.iter().sum::<f64>() / all_values.len() as f64,
                    AggFunc::Count => all_values.len() as f64,
                    AggFunc::Min => all_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    AggFunc::Max => all_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    AggFunc::Std => {
                        if all_values.len() <= 1 {
                            0.0
                        } else {
                            let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
                            let variance =
                                all_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                                    / (all_values.len() - 1) as f64;
                            variance.sqrt()
                        }
                    }
                    _ => f64::NAN,
                }
            };

            agg_values.push(agg_result.to_string());
        }

        let agg_alias = format!("{}_{}_from_level_{}", func.as_str(), column, source_level);
        let agg_series = Series::new(agg_values, Some(agg_alias.clone()))?;
        result.add_column(agg_alias, agg_series)?;

        Ok(result)
    }

    /// Nested transformation: apply a transformation within each group considering hierarchy
    pub fn nested_transform<F>(
        &self,
        column: &str,
        transform_fn: F,
        level: usize,
    ) -> Result<DataFrame>
    where
        F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
    {
        if level >= self.hierarchy.max_depth {
            return Err(Error::InvalidValue(
                "Level exceeds hierarchy depth".to_string(),
            ));
        }

        let mut result_values = vec![f64::NAN; self.df.row_count()];
        let column_values = self.df.get_column_string_values(column)?;

        // Apply transformation to each group at the specified level
        let level_nodes = self.hierarchy.get_level_nodes(level);
        for node in &level_nodes {
            // Get all indices for this group (including children if they exist)
            let group_indices = if node.is_leaf {
                node.indices.clone()
            } else {
                self.collect_all_indices(node)
            };

            // Extract values for this group
            let group_values: Vec<f64> = group_indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        column_values[idx].parse::<f64>().ok()
                    } else {
                        None
                    }
                })
                .collect();

            if !group_values.is_empty() {
                // Apply transformation
                let transformed = transform_fn(&group_values);

                // Map back to original indices
                for (i, &original_idx) in group_indices.iter().enumerate() {
                    if i < transformed.len() && original_idx < result_values.len() {
                        result_values[original_idx] = transformed[i];
                    }
                }
            }
        }

        // Create result DataFrame
        let mut result = self.df.clone();
        let transform_column_name = format!("{}_transformed", column);
        let transform_series = Series::new(
            result_values.iter().map(|&v| v.to_string()).collect(),
            Some(transform_column_name.clone()),
        )?;
        result.add_column(transform_column_name, transform_series)?;

        Ok(result)
    }

    /// Inter-level operations: compute relationships between parent and child groups
    pub fn inter_level_ratio(
        &self,
        column: &str,
        parent_level: usize,
        child_level: usize,
    ) -> Result<DataFrame> {
        if parent_level >= child_level || child_level >= self.hierarchy.max_depth {
            return Err(Error::InvalidValue(
                "Invalid level relationship".to_string(),
            ));
        }

        let mut result = DataFrame::new();
        let child_nodes = self.hierarchy.get_level_nodes(child_level);

        // Build result structure
        for level in 0..=child_level {
            let level_name = self
                .hierarchy
                .level_columns
                .get(level)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("level_{}", level));

            let level_values: Vec<String> = child_nodes
                .iter()
                .map(|node| node.full_key.get_level(level).unwrap_or("").to_string())
                .collect();

            let level_series = Series::new(level_values, Some(level_name.clone()))?;
            result.add_column(level_name, level_series)?;
        }

        // Calculate ratios
        let mut ratio_values = Vec::new();
        for child_node in &child_nodes {
            let parent_key = child_node.full_key.partial_key(parent_level);

            let child_sum = self.calculate_hierarchical_aggregation(
                column,
                AggFunc::Sum,
                child_node,
                &None,
                false,
            )?;

            let parent_sum =
                if let Some(parent_node) = self.hierarchy.navigate_to_group(&parent_key) {
                    self.calculate_hierarchical_aggregation(
                        column,
                        AggFunc::Sum,
                        parent_node,
                        &None,
                        true,
                    )?
                } else {
                    f64::NAN
                };

            let ratio = if parent_sum != 0.0 && !parent_sum.is_nan() {
                child_sum / parent_sum
            } else {
                f64::NAN
            };

            ratio_values.push(ratio.to_string());
        }

        let ratio_alias = format!("{}_ratio_to_level_{}", column, parent_level);
        let ratio_series = Series::new(ratio_values, Some(ratio_alias.clone()))?;
        result.add_column(ratio_alias, ratio_series)?;

        Ok(result)
    }

    /// Hierarchical filtering: filter groups based on hierarchical criteria
    pub fn hierarchical_filter<F>(
        &self,
        column: &str,
        level: usize,
        filter_fn: F,
    ) -> Result<HierarchicalDataFrameGroupBy>
    where
        F: Fn(f64) -> bool + Send + Sync,
    {
        if level >= self.hierarchy.max_depth {
            return Err(Error::InvalidValue(
                "Level exceeds hierarchy depth".to_string(),
            ));
        }

        // Get groups at the specified level and filter them
        let level_nodes = self.hierarchy.get_level_nodes(level);
        let mut selected_indices = std::collections::HashSet::new();

        for node in &level_nodes {
            let agg_value =
                self.calculate_hierarchical_aggregation(column, AggFunc::Sum, node, &None, true)?;

            if !agg_value.is_nan() && filter_fn(agg_value) {
                // Include all indices from this group and its children
                let all_indices = self.collect_all_indices(node);
                for idx in all_indices {
                    selected_indices.insert(idx);
                }
            }
        }

        // Create a filtered DataFrame
        let mut filtered_data = std::collections::HashMap::new();
        let column_names = self.df.column_names();

        for col_name in &column_names {
            let column_values = self.df.get_column_string_values(col_name)?;
            let filtered_values: Vec<String> = selected_indices
                .iter()
                .filter_map(|&idx| {
                    if idx < column_values.len() {
                        Some(column_values[idx].clone())
                    } else {
                        None
                    }
                })
                .collect();
            filtered_data.insert(col_name.clone(), filtered_values);
        }

        let filtered_df = DataFrame::from_map(filtered_data, None)?;

        // Create new hierarchical groupby with filtered data
        HierarchicalDataFrameGroupBy::new(filtered_df, self.hierarchy.level_columns.clone())
    }

    /// Cross-level comparison: compare values across different levels
    pub fn cross_level_comparison(
        &self,
        column: &str,
        base_level: usize,
        compare_levels: Vec<usize>,
    ) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let base_nodes = self.hierarchy.get_level_nodes(base_level);

        // Build result structure for base level
        for level in 0..=base_level {
            let level_name = self
                .hierarchy
                .level_columns
                .get(level)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("level_{}", level));

            let level_values: Vec<String> = base_nodes
                .iter()
                .map(|node| node.full_key.get_level(level).unwrap_or("").to_string())
                .collect();

            let level_series = Series::new(level_values, Some(level_name.clone()))?;
            result.add_column(level_name, level_series)?;
        }

        // Add base values
        let mut base_values = Vec::new();
        for base_node in &base_nodes {
            let base_value = self.calculate_hierarchical_aggregation(
                column,
                AggFunc::Sum,
                base_node,
                &None,
                true,
            )?;
            base_values.push(base_value.to_string());
        }

        let base_alias = format!("{}_base_level_{}", column, base_level);
        let base_series = Series::new(base_values, Some(base_alias.clone()))?;
        result.add_column(base_alias, base_series)?;

        // Add comparison values for each compare level
        for &compare_level in &compare_levels {
            let mut compare_values = Vec::new();

            for base_node in &base_nodes {
                let compare_value = if compare_level > base_level {
                    // Comparing to child level - aggregate children
                    let child_nodes = base_node.get_nodes_at_level(compare_level);
                    let child_sum: f64 = child_nodes
                        .iter()
                        .map(|child| {
                            self.calculate_hierarchical_aggregation(
                                column,
                                AggFunc::Sum,
                                child,
                                &None,
                                false,
                            )
                            .unwrap_or(0.0)
                        })
                        .sum();
                    child_sum
                } else if compare_level < base_level {
                    // Comparing to parent level
                    let parent_key = base_node.full_key.partial_key(compare_level);
                    if let Some(parent_node) = self.hierarchy.navigate_to_group(&parent_key) {
                        self.calculate_hierarchical_aggregation(
                            column,
                            AggFunc::Sum,
                            parent_node,
                            &None,
                            true,
                        )
                        .unwrap_or(f64::NAN)
                    } else {
                        f64::NAN
                    }
                } else {
                    // Same level
                    self.calculate_hierarchical_aggregation(
                        column,
                        AggFunc::Sum,
                        base_node,
                        &None,
                        true,
                    )
                    .unwrap_or(f64::NAN)
                };

                compare_values.push(compare_value.to_string());
            }

            let compare_alias = format!("{}_compare_level_{}", column, compare_level);
            let compare_series = Series::new(compare_values, Some(compare_alias.clone()))?;
            result.add_column(compare_alias, compare_series)?;
        }

        Ok(result)
    }

    /// Nested rollup: create rollup aggregations across hierarchy levels
    pub fn nested_rollup(&self, column: &str, func: AggFunc) -> Result<DataFrame> {
        let mut result = DataFrame::new();
        let mut all_groups = Vec::new();

        // Collect all groups from all levels
        for level in 0..self.hierarchy.max_depth {
            let level_nodes = self.hierarchy.get_level_nodes(level);
            for node in level_nodes {
                all_groups.push((level, node));
            }
        }

        // Sort by hierarchy depth and key
        all_groups.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.full_key.cmp(&b.1.full_key)));

        // Build result columns
        let max_levels = self.hierarchy.max_depth;
        for level in 0..max_levels {
            let level_name = self
                .hierarchy
                .level_columns
                .get(level)
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("level_{}", level));

            let level_values: Vec<String> = all_groups
                .iter()
                .map(|(_, node)| node.full_key.get_level(level).unwrap_or("").to_string())
                .collect();

            let level_series = Series::new(level_values, Some(level_name.clone()))?;
            result.add_column(level_name, level_series)?;
        }

        // Add rollup level indicator
        let rollup_levels: Vec<String> = all_groups
            .iter()
            .map(|(level, _)| level.to_string())
            .collect();
        let rollup_series = Series::new(rollup_levels, Some("rollup_level".to_string()))?;
        result.add_column("rollup_level".to_string(), rollup_series)?;

        // Add aggregated values
        let mut agg_values = Vec::new();
        for (_, node) in &all_groups {
            let agg_value =
                self.calculate_hierarchical_aggregation(column, func, node, &None, true)?;
            agg_values.push(agg_value.to_string());
        }

        let agg_alias = format!("{}_{}_rollup", func.as_str(), column);
        let agg_series = Series::new(agg_values, Some(agg_alias.clone()))?;
        result.add_column(agg_alias, agg_series)?;

        Ok(result)
    }
}

/// Extension trait to add hierarchical groupby functionality to DataFrame
pub trait HierarchicalGroupByExt {
    /// Create a hierarchical groupby with multiple levels
    fn hierarchical_groupby(&self, columns: Vec<String>) -> Result<HierarchicalDataFrameGroupBy>;
}

impl HierarchicalGroupByExt for DataFrame {
    fn hierarchical_groupby(&self, columns: Vec<String>) -> Result<HierarchicalDataFrameGroupBy> {
        HierarchicalDataFrameGroupBy::new(self.clone(), columns)
    }
}

/// Builder for creating hierarchical aggregation specifications
pub struct HierarchicalAggBuilder {
    column: String,
    level_functions: Vec<(usize, AggFunc, String)>,
    custom_fn: Option<CustomAggFn>,
    propagate_up: bool,
    include_intermediate: bool,
}

impl HierarchicalAggBuilder {
    /// Create a new hierarchical aggregation builder
    pub fn new(column: String) -> Self {
        Self {
            column,
            level_functions: Vec::new(),
            custom_fn: None,
            propagate_up: false,
            include_intermediate: true,
        }
    }

    /// Add aggregation for a specific level
    pub fn at_level(mut self, level: usize, func: AggFunc, alias: String) -> Self {
        self.level_functions.push((level, func, alias));
        self
    }

    /// Add aggregation for all levels
    pub fn at_all_levels(mut self, func: AggFunc, base_alias: String) -> Self {
        // This would be updated when we know the number of levels
        // For now, just add to level 0
        self.level_functions.push((0, func, base_alias));
        self
    }

    /// Enable propagation of aggregations up the hierarchy
    pub fn with_propagation(mut self) -> Self {
        self.propagate_up = true;
        self
    }

    /// Set custom aggregation function
    pub fn with_custom<F>(mut self, func: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.custom_fn = Some(Arc::new(func));
        self
    }

    /// Build the hierarchical aggregation
    pub fn build(self) -> HierarchicalAgg {
        HierarchicalAgg {
            column: self.column,
            level_functions: self.level_functions,
            custom_fn: self.custom_fn,
            propagate_up: self.propagate_up,
            include_intermediate: self.include_intermediate,
        }
    }
}

/// Utility functions for hierarchical groupby operations
pub mod utils {
    use super::*;

    /// Create a simple hierarchical aggregation
    pub fn simple_hierarchical_agg(column: &str, func: AggFunc, level: usize) -> HierarchicalAgg {
        HierarchicalAggBuilder::new(column.to_string())
            .at_level(level, func, format!("{}_{}", func.as_str(), column))
            .build()
    }

    /// Create aggregations for all levels
    pub fn all_levels_agg(column: &str, func: AggFunc, max_levels: usize) -> Vec<HierarchicalAgg> {
        (0..max_levels)
            .map(|level| simple_hierarchical_agg(column, func, level))
            .collect()
    }

    /// Create comprehensive aggregations (sum, mean, count) for a column
    pub fn comprehensive_agg(column: &str, level: usize) -> Vec<HierarchicalAgg> {
        vec![
            simple_hierarchical_agg(column, AggFunc::Sum, level),
            simple_hierarchical_agg(column, AggFunc::Mean, level),
            simple_hierarchical_agg(column, AggFunc::Count, level),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_key() {
        let key = HierarchicalKey::new(
            vec!["A".to_string(), "X".to_string(), "1".to_string()],
            vec![
                "category".to_string(),
                "region".to_string(),
                "product".to_string(),
            ],
        );

        assert_eq!(key.depth(), 3);
        assert_eq!(key.get_level(0), Some("A"));
        assert_eq!(key.get_level(1), Some("X"));
        assert_eq!(key.get_level(2), Some("1"));
        assert_eq!(key.display_string(), "A | X | 1");

        let partial = key.partial_key(1);
        assert_eq!(partial.depth(), 2);
        assert_eq!(partial.levels, vec!["A", "X"]);
    }

    #[test]
    fn test_group_node() {
        let key = HierarchicalKey::new(vec!["A".to_string()], vec!["category".to_string()]);
        let mut node = GroupNode::new("A".to_string(), key, 0);

        assert!(node.is_leaf);
        assert_eq!(node.child_count(), 0);

        node.indices = vec![0, 1, 2];
        assert_eq!(node.total_size(), 3);

        let child_key = HierarchicalKey::new(
            vec!["A".to_string(), "X".to_string()],
            vec!["category".to_string(), "region".to_string()],
        );
        let child = GroupNode::new("X".to_string(), child_key, 1);
        node.add_child("X".to_string(), child);

        assert!(!node.is_leaf);
        assert_eq!(node.child_count(), 1);
    }

    #[test]
    fn test_hierarchical_agg_builder() {
        let agg = HierarchicalAggBuilder::new("sales".to_string())
            .at_level(0, AggFunc::Sum, "total_sales".to_string())
            .at_level(1, AggFunc::Mean, "avg_sales".to_string())
            .with_propagation()
            .build();

        assert_eq!(agg.column, "sales");
        assert_eq!(agg.level_functions.len(), 2);
        assert!(agg.propagate_up);
        assert_eq!(
            agg.level_functions[0],
            (0, AggFunc::Sum, "total_sales".to_string())
        );
        assert_eq!(
            agg.level_functions[1],
            (1, AggFunc::Mean, "avg_sales".to_string())
        );
    }

    #[test]
    fn test_group_hierarchy_creation() {
        let hierarchy = GroupHierarchy::new(vec!["region".to_string(), "department".to_string()]);
        assert_eq!(hierarchy.level_columns, vec!["region", "department"]);
        assert_eq!(hierarchy.max_depth, 0);
        assert_eq!(hierarchy.total_groups, 0);
    }

    #[test]
    fn test_hierarchical_groupby_creation() {
        let mut df = DataFrame::new();

        // Create test data
        let regions = vec!["North", "North", "South", "South"];
        let departments = vec!["Sales", "Marketing", "Sales", "Marketing"];
        let values = vec!["100", "200", "150", "250"];

        df.add_column(
            "region".to_string(),
            Series::new(
                regions.iter().map(|s| s.to_string()).collect(),
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "department".to_string(),
            Series::new(
                departments.iter().map(|s| s.to_string()).collect(),
                Some("department".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "sales".to_string(),
            Series::new(
                values.iter().map(|s| s.to_string()).collect(),
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df
            .hierarchical_groupby(vec!["region".to_string(), "department".to_string()])
            .unwrap();

        let stats = hierarchical_gb.hierarchy_stats();
        assert_eq!(stats.total_levels, 2);
        assert_eq!(stats.leaf_groups, 4);
        assert!(stats.total_groups >= 6); // 2 regions + 4 departments = 6 groups minimum
    }

    #[test]
    fn test_hierarchical_size_calculation() {
        let mut df = DataFrame::new();

        // Create test data with repeated groups
        let regions = vec!["North", "North", "North", "South", "South"];
        let departments = vec!["Sales", "Sales", "Marketing", "Sales", "Marketing"];
        let values = vec!["100", "150", "200", "120", "180"];

        df.add_column(
            "region".to_string(),
            Series::new(
                regions.iter().map(|s| s.to_string()).collect(),
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "department".to_string(),
            Series::new(
                departments.iter().map(|s| s.to_string()).collect(),
                Some("department".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "sales".to_string(),
            Series::new(
                values.iter().map(|s| s.to_string()).collect(),
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df
            .hierarchical_groupby(vec!["region".to_string(), "department".to_string()])
            .unwrap();

        let sizes = hierarchical_gb.size().unwrap();
        assert_eq!(sizes.row_count(), 4); // 4 unique region-department combinations
    }

    #[test]
    fn test_hierarchical_aggregation() {
        let mut df = DataFrame::new();

        // Create test data
        let regions = vec!["North", "North", "South", "South"];
        let departments = vec!["Sales", "Sales", "Sales", "Marketing"];
        let values = vec!["100", "200", "150", "250"];

        df.add_column(
            "region".to_string(),
            Series::new(
                regions.iter().map(|s| s.to_string()).collect(),
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "department".to_string(),
            Series::new(
                departments.iter().map(|s| s.to_string()).collect(),
                Some("department".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "sales".to_string(),
            Series::new(
                values.iter().map(|s| s.to_string()).collect(),
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df
            .hierarchical_groupby(vec!["region".to_string(), "department".to_string()])
            .unwrap();

        let agg = HierarchicalAggBuilder::new("sales".to_string())
            .at_level(0, AggFunc::Sum, "region_total".to_string())
            .at_level(1, AggFunc::Sum, "dept_total".to_string())
            .build();

        let result = hierarchical_gb.agg_hierarchical(vec![agg]).unwrap();
        assert!(result.row_count() > 0);
        assert!(result.contains_column("region_total"));
        assert!(result.contains_column("dept_total"));
    }

    #[test]
    fn test_cross_level_aggregation() {
        let mut df = DataFrame::new();

        // Create test data with 3 levels
        let regions = vec!["North", "North", "North", "North"];
        let departments = vec!["Sales", "Sales", "Marketing", "Marketing"];
        let products = vec!["A", "B", "A", "B"];
        let values = vec!["100", "200", "150", "250"];

        df.add_column(
            "region".to_string(),
            Series::new(
                regions.iter().map(|s| s.to_string()).collect(),
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "department".to_string(),
            Series::new(
                departments.iter().map(|s| s.to_string()).collect(),
                Some("department".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "product".to_string(),
            Series::new(
                products.iter().map(|s| s.to_string()).collect(),
                Some("product".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "sales".to_string(),
            Series::new(
                values.iter().map(|s| s.to_string()).collect(),
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df
            .hierarchical_groupby(vec![
                "region".to_string(),
                "department".to_string(),
                "product".to_string(),
            ])
            .unwrap();

        let result = hierarchical_gb
            .cross_level_agg("sales", AggFunc::Sum, 2, 1)
            .unwrap();
        assert!(result.row_count() > 0);
        assert!(result.contains_column("sum_sales_from_level_2"));
    }

    #[test]
    fn test_inter_level_ratio() {
        let mut df = DataFrame::new();

        // Create test data with clear hierarchy
        let regions = vec!["North", "North", "South", "South"];
        let departments = vec!["Sales", "Marketing", "Sales", "Marketing"];
        let values = vec!["400", "200", "300", "100"];

        df.add_column(
            "region".to_string(),
            Series::new(
                regions.iter().map(|s| s.to_string()).collect(),
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "department".to_string(),
            Series::new(
                departments.iter().map(|s| s.to_string()).collect(),
                Some("department".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "sales".to_string(),
            Series::new(
                values.iter().map(|s| s.to_string()).collect(),
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df
            .hierarchical_groupby(vec!["region".to_string(), "department".to_string()])
            .unwrap();

        let result = hierarchical_gb.inter_level_ratio("sales", 0, 1).unwrap();
        assert!(result.row_count() > 0);
        assert!(result.contains_column("sales_ratio_to_level_0"));
    }

    #[test]
    fn test_nested_rollup() {
        let mut df = DataFrame::new();

        // Create simple test data
        let regions = vec!["North", "North", "South"];
        let departments = vec!["Sales", "Marketing", "Sales"];
        let values = vec!["100", "200", "150"];

        df.add_column(
            "region".to_string(),
            Series::new(
                regions.iter().map(|s| s.to_string()).collect(),
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "department".to_string(),
            Series::new(
                departments.iter().map(|s| s.to_string()).collect(),
                Some("department".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "sales".to_string(),
            Series::new(
                values.iter().map(|s| s.to_string()).collect(),
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df
            .hierarchical_groupby(vec!["region".to_string(), "department".to_string()])
            .unwrap();

        let result = hierarchical_gb
            .nested_rollup("sales", AggFunc::Sum)
            .unwrap();
        assert!(result.row_count() > 0);
        assert!(result.contains_column("rollup_level"));
        assert!(result.contains_column("sum_sales_rollup"));
    }

    #[test]
    fn test_hierarchical_filtering() {
        let mut df = DataFrame::new();

        // Create test data with varying sales amounts
        let regions = vec!["North", "North", "South", "South"];
        let departments = vec!["Sales", "Marketing", "Sales", "Marketing"];
        let values = vec!["1000", "100", "2000", "50"]; // Sales and Marketing departments have different totals

        df.add_column(
            "region".to_string(),
            Series::new(
                regions.iter().map(|s| s.to_string()).collect(),
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "department".to_string(),
            Series::new(
                departments.iter().map(|s| s.to_string()).collect(),
                Some("department".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        df.add_column(
            "sales".to_string(),
            Series::new(
                values.iter().map(|s| s.to_string()).collect(),
                Some("sales".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df
            .hierarchical_groupby(vec!["region".to_string(), "department".to_string()])
            .unwrap();

        // Filter departments with sales > 500
        let filtered_gb = hierarchical_gb
            .hierarchical_filter(
                "sales",
                1, // Department level
                |total_sales| total_sales > 500.0,
            )
            .unwrap();

        let filtered_stats = filtered_gb.hierarchy_stats();
        assert!(filtered_stats.total_groups > 0);
        // Should exclude Marketing departments (100 + 50 = 150 < 500)
        // Should include Sales departments (1000 and 2000 > 500)
    }

    #[test]
    fn test_utility_functions() {
        // Test simple aggregation utility
        let simple_agg = utils::simple_hierarchical_agg("sales", AggFunc::Sum, 0);
        assert_eq!(simple_agg.column, "sales");
        assert_eq!(simple_agg.level_functions.len(), 1);
        assert_eq!(simple_agg.level_functions[0].0, 0);
        assert_eq!(simple_agg.level_functions[0].1, AggFunc::Sum);

        // Test comprehensive aggregation utility
        let comprehensive_aggs = utils::comprehensive_agg("sales", 1);
        assert_eq!(comprehensive_aggs.len(), 3);
        assert!(comprehensive_aggs
            .iter()
            .any(|agg| agg.level_functions[0].1 == AggFunc::Sum));
        assert!(comprehensive_aggs
            .iter()
            .any(|agg| agg.level_functions[0].1 == AggFunc::Mean));
        assert!(comprehensive_aggs
            .iter()
            .any(|agg| agg.level_functions[0].1 == AggFunc::Count));

        // Test all levels aggregation utility
        let all_levels_aggs = utils::all_levels_agg("sales", AggFunc::Mean, 3);
        assert_eq!(all_levels_aggs.len(), 3);
        for (i, agg) in all_levels_aggs.iter().enumerate() {
            assert_eq!(agg.level_functions[0].0, i);
            assert_eq!(agg.level_functions[0].1, AggFunc::Mean);
        }
    }

    #[test]
    fn test_error_handling() {
        let mut df = DataFrame::new();

        // Test with non-existent column
        df.add_column(
            "valid_column".to_string(),
            Series::new(
                vec!["A".to_string(), "B".to_string()],
                Some("valid_column".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let result = df.hierarchical_groupby(vec!["non_existent_column".to_string()]);
        assert!(result.is_err());

        // Test with valid setup for other error conditions
        df.add_column(
            "region".to_string(),
            Series::new(
                vec!["North".to_string(), "South".to_string()],
                Some("region".to_string()),
            )
            .unwrap(),
        )
        .unwrap();

        let hierarchical_gb = df.hierarchical_groupby(vec!["region".to_string()]).unwrap();

        // Test cross-level aggregation with invalid levels
        let result = hierarchical_gb.cross_level_agg("valid_column", AggFunc::Sum, 10, 0);
        assert!(result.is_err());

        // Test inter-level ratio with invalid level relationship
        let result = hierarchical_gb.inter_level_ratio("valid_column", 1, 0); // parent >= child
        assert!(result.is_err());
    }
}
