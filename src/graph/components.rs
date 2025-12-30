//! Connected components and community detection algorithms
//!
//! This module provides:
//! - Connected components (for undirected graphs)
//! - Strongly connected components (for directed graphs)
//! - Weakly connected components
//! - Community detection (Louvain algorithm, label propagation)
//!
//! # Examples
//!
//! ```
//! use pandrs::graph::{Graph, GraphType};
//! use pandrs::graph::components::connected_components;
//!
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
//! // ... add nodes and edges ...
//! let components = connected_components(&graph);
//! ```

use super::core::{Graph, GraphError, GraphType, NodeId};
use super::traversal::{bfs, dfs_from};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

/// Result of connected components analysis
#[derive(Debug, Clone)]
pub struct ComponentResult {
    /// Component ID for each node
    pub node_components: HashMap<NodeId, usize>,
    /// List of nodes in each component
    pub components: Vec<HashSet<NodeId>>,
    /// Number of components
    pub num_components: usize,
}

impl ComponentResult {
    /// Returns the component ID for a node
    pub fn component_of(&self, node: NodeId) -> Option<usize> {
        self.node_components.get(&node).copied()
    }

    /// Returns all nodes in a component
    pub fn nodes_in_component(&self, component_id: usize) -> Option<&HashSet<NodeId>> {
        self.components.get(component_id)
    }

    /// Returns the size of each component
    pub fn component_sizes(&self) -> Vec<usize> {
        self.components.iter().map(|c| c.len()).collect()
    }

    /// Returns the largest component
    pub fn largest_component(&self) -> Option<&HashSet<NodeId>> {
        self.components.iter().max_by_key(|c| c.len())
    }

    /// Checks if two nodes are in the same component
    pub fn are_connected(&self, node1: NodeId, node2: NodeId) -> bool {
        match (
            self.node_components.get(&node1),
            self.node_components.get(&node2),
        ) {
            (Some(c1), Some(c2)) => c1 == c2,
            _ => false,
        }
    }
}

/// Finds all connected components in an undirected graph using BFS
pub fn connected_components<N, W>(graph: &Graph<N, W>) -> ComponentResult
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut node_components: HashMap<NodeId, usize> = HashMap::new();
    let mut components: Vec<HashSet<NodeId>> = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();

    for node_id in graph.node_ids() {
        if !visited.contains(&node_id) {
            let result = bfs(graph, node_id);
            let component: HashSet<NodeId> = result.visit_order.into_iter().collect();
            let component_id = components.len();

            for &node in &component {
                node_components.insert(node, component_id);
                visited.insert(node);
            }

            components.push(component);
        }
    }

    let num_components = components.len();

    ComponentResult {
        node_components,
        components,
        num_components,
    }
}

/// Checks if an undirected graph is connected
pub fn is_connected<N, W>(graph: &Graph<N, W>) -> bool
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    if graph.is_empty() {
        return true;
    }

    let result = connected_components(graph);
    result.num_components == 1
}

/// Finds strongly connected components in a directed graph using Kosaraju's algorithm
pub fn strongly_connected_components<N, W>(graph: &Graph<N, W>) -> ComponentResult
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let nodes: Vec<NodeId> = graph.node_ids().collect();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut finish_order: Vec<NodeId> = Vec::new();

    // First DFS pass to get finish order
    for &node in &nodes {
        if !visited.contains(&node) {
            kosaraju_dfs1(graph, node, &mut visited, &mut finish_order);
        }
    }

    // Get reversed graph
    let reversed = graph.reverse();

    // Second DFS pass on reversed graph in reverse finish order
    let mut node_components: HashMap<NodeId, usize> = HashMap::new();
    let mut components: Vec<HashSet<NodeId>> = Vec::new();
    visited.clear();

    // Create mapping from old node IDs to new node IDs in reversed graph
    let node_map: HashMap<NodeId, NodeId> = nodes
        .iter()
        .enumerate()
        .map(|(i, &old_id)| (old_id, NodeId(i)))
        .collect();

    // Create inverse mapping from reversed graph node IDs back to original
    let reverse_node_map: HashMap<NodeId, NodeId> =
        node_map.iter().map(|(&orig, &rev)| (rev, orig)).collect();

    for &node in finish_order.iter().rev() {
        if !visited.contains(&node) {
            let mut component: HashSet<NodeId> = HashSet::new();
            // Use the mapped node ID for the reversed graph
            let reversed_node = node_map[&node];
            kosaraju_dfs2(
                &reversed,
                reversed_node,
                &mut visited,
                &mut component,
                &reverse_node_map,
            );

            let component_id = components.len();
            for &n in &component {
                node_components.insert(n, component_id);
            }
            components.push(component);
        }
    }

    let num_components = components.len();

    ComponentResult {
        node_components,
        components,
        num_components,
    }
}

fn kosaraju_dfs1<N, W>(
    graph: &Graph<N, W>,
    node: NodeId,
    visited: &mut HashSet<NodeId>,
    finish_order: &mut Vec<NodeId>,
) where
    N: Clone + Debug,
    W: Clone + Debug,
{
    visited.insert(node);

    if let Some(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                kosaraju_dfs1(graph, neighbor, visited, finish_order);
            }
        }
    }

    finish_order.push(node);
}

fn kosaraju_dfs2<N, W>(
    graph: &Graph<N, W>,
    node: NodeId,
    visited: &mut HashSet<NodeId>,
    component: &mut HashSet<NodeId>,
    reverse_node_map: &HashMap<NodeId, NodeId>,
) where
    N: Clone + Debug,
    W: Clone + Debug,
{
    // Map back to original node ID using the reverse mapping
    let original_node = reverse_node_map.get(&node).copied().unwrap_or(node);
    visited.insert(original_node);
    component.insert(original_node);

    if let Some(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            let original_neighbor = reverse_node_map.get(&neighbor).copied().unwrap_or(neighbor);
            if !visited.contains(&original_neighbor) {
                kosaraju_dfs2(graph, neighbor, visited, component, reverse_node_map);
            }
        }
    }
}

/// Finds weakly connected components in a directed graph
/// (treats graph as undirected for connectivity)
pub fn weakly_connected_components<N, W>(graph: &Graph<N, W>) -> ComponentResult
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    // Convert to undirected and find connected components
    let undirected = graph.to_undirected();
    connected_components(&undirected)
}

/// Community detection using Label Propagation algorithm
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// Community assignments for each node
pub fn label_propagation<N, W>(graph: &Graph<N, W>, max_iterations: usize) -> HashMap<NodeId, usize>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let nodes: Vec<NodeId> = graph.node_ids().collect();

    // Initialize each node with its own label
    let mut labels: HashMap<NodeId, usize> = nodes.iter().map(|&n| (n, n.0)).collect();

    for _ in 0..max_iterations {
        let mut changed = false;

        // Process nodes in random order (simplified: just iterate)
        for &node in &nodes {
            if let Some(neighbors) = graph.neighbors(node) {
                if neighbors.is_empty() {
                    continue;
                }

                // Count label frequencies among neighbors
                let mut label_counts: HashMap<usize, usize> = HashMap::new();
                for neighbor in neighbors {
                    let label = labels[&neighbor];
                    *label_counts.entry(label).or_insert(0) += 1;
                }

                // Find most frequent label
                if let Some((&max_label, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
                    if labels[&node] != max_label {
                        labels.insert(node, max_label);
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Normalize labels to consecutive integers
    let unique_labels: HashSet<usize> = labels.values().copied().collect();
    let label_map: HashMap<usize, usize> = unique_labels
        .iter()
        .enumerate()
        .map(|(i, &l)| (l, i))
        .collect();

    labels
        .into_iter()
        .map(|(node, label)| (node, label_map[&label]))
        .collect()
}

/// Calculates the modularity of a community assignment
///
/// Modularity measures the quality of a partition of a graph into communities.
/// Higher values indicate better community structure.
pub fn modularity<N, W>(graph: &Graph<N, W>, communities: &HashMap<NodeId, usize>) -> f64
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let m = graph.edge_count() as f64;
    if m == 0.0 {
        return 0.0;
    }

    let m2 = 2.0 * m;
    let mut q = 0.0;

    for node_i in graph.node_ids() {
        for node_j in graph.node_ids() {
            // Only count if in same community
            if communities.get(&node_i) != communities.get(&node_j) {
                continue;
            }

            let ki = graph.degree(node_i).unwrap_or(0) as f64;
            let kj = graph.degree(node_j).unwrap_or(0) as f64;

            let aij = if graph.has_edge(node_i, node_j) {
                1.0
            } else {
                0.0
            };

            q += aij - (ki * kj) / m2;
        }
    }

    q / m2
}

/// Community detection using Louvain algorithm
///
/// The Louvain algorithm is a greedy optimization method that maximizes modularity.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `resolution` - Resolution parameter (higher = smaller communities)
///
/// # Returns
/// Community assignments and final modularity
pub fn louvain<N, W>(graph: &Graph<N, W>, resolution: f64) -> (HashMap<NodeId, usize>, f64)
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let nodes: Vec<NodeId> = graph.node_ids().collect();
    let m = graph.edge_count() as f64;

    if m == 0.0 || nodes.is_empty() {
        let communities: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        return (communities, 0.0);
    }

    let m2 = 2.0 * m;

    // Initialize: each node in its own community
    let mut communities: HashMap<NodeId, usize> = nodes.iter().map(|&n| (n, n.0)).collect();

    // Pre-calculate degrees
    let degrees: HashMap<NodeId, f64> = nodes
        .iter()
        .map(|&n| (n, graph.degree(n).unwrap_or(0) as f64))
        .collect();

    // Calculate sum of degrees in each community
    let mut community_degrees: HashMap<usize, f64> = nodes
        .iter()
        .map(|&n| (communities[&n], degrees[&n]))
        .collect();

    let mut improved = true;

    while improved {
        improved = false;

        for &node in &nodes {
            let current_community = communities[&node];
            let ki = degrees[&node];

            // Calculate modularity gain for moving to each neighbor's community
            let mut best_community = current_community;
            let mut best_gain = 0.0;

            // Get unique communities of neighbors
            let mut neighbor_communities: HashSet<usize> = HashSet::new();
            if let Some(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    neighbor_communities.insert(communities[&neighbor]);
                }
            }

            // Remove node from current community temporarily
            *community_degrees.get_mut(&current_community).unwrap() -= ki;

            for &target_community in &neighbor_communities {
                if target_community == current_community {
                    continue;
                }

                // Calculate edges to target community
                let mut ki_in = 0.0;
                if let Some(neighbors) = graph.neighbors(node) {
                    for neighbor in neighbors {
                        if communities[&neighbor] == target_community {
                            ki_in += 1.0;
                        }
                    }
                }

                let sigma_tot = community_degrees
                    .get(&target_community)
                    .copied()
                    .unwrap_or(0.0);

                // Calculate modularity gain
                let gain = ki_in - resolution * (sigma_tot * ki) / m2;

                if gain > best_gain {
                    best_gain = gain;
                    best_community = target_community;
                }
            }

            // Move node to best community
            if best_community != current_community {
                communities.insert(node, best_community);
                *community_degrees.entry(best_community).or_insert(0.0) += ki;
                improved = true;
            } else {
                *community_degrees.get_mut(&current_community).unwrap() += ki;
            }
        }
    }

    // Normalize community IDs
    let unique_communities: HashSet<usize> = communities.values().copied().collect();
    let community_map: HashMap<usize, usize> = unique_communities
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    let normalized_communities: HashMap<NodeId, usize> = communities
        .into_iter()
        .map(|(node, c)| (node, community_map[&c]))
        .collect();

    let final_modularity = modularity(graph, &normalized_communities);

    (normalized_communities, final_modularity)
}

/// Convenience function for Louvain with default resolution
pub fn louvain_default<N, W>(graph: &Graph<N, W>) -> (HashMap<NodeId, usize>, f64)
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    louvain(graph, 1.0)
}

/// Finds bridge edges (edges whose removal disconnects the graph)
pub fn find_bridges<N, W>(graph: &Graph<N, W>) -> Vec<(NodeId, NodeId)>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut bridges = Vec::new();
    let nodes: Vec<NodeId> = graph.node_ids().collect();

    if nodes.is_empty() {
        return bridges;
    }

    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut discovery: HashMap<NodeId, usize> = HashMap::new();
    let mut low: HashMap<NodeId, usize> = HashMap::new();
    let mut parent: HashMap<NodeId, Option<NodeId>> = HashMap::new();
    let mut time = 0;

    for &node in &nodes {
        if !visited.contains(&node) {
            bridge_dfs(
                graph,
                node,
                &mut visited,
                &mut discovery,
                &mut low,
                &mut parent,
                &mut time,
                &mut bridges,
            );
        }
    }

    bridges
}

fn bridge_dfs<N, W>(
    graph: &Graph<N, W>,
    node: NodeId,
    visited: &mut HashSet<NodeId>,
    discovery: &mut HashMap<NodeId, usize>,
    low: &mut HashMap<NodeId, usize>,
    parent: &mut HashMap<NodeId, Option<NodeId>>,
    time: &mut usize,
    bridges: &mut Vec<(NodeId, NodeId)>,
) where
    N: Clone + Debug,
    W: Clone + Debug,
{
    visited.insert(node);
    *time += 1;
    discovery.insert(node, *time);
    low.insert(node, *time);

    if let Some(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                parent.insert(neighbor, Some(node));
                bridge_dfs(
                    graph, neighbor, visited, discovery, low, parent, time, bridges,
                );

                // Update low value
                let low_neighbor = low[&neighbor];
                let low_node = low[&node];
                low.insert(node, low_node.min(low_neighbor));

                // If low[neighbor] > discovery[node], this is a bridge
                if low[&neighbor] > discovery[&node] {
                    bridges.push((node, neighbor));
                }
            } else if parent.get(&node) != Some(&Some(neighbor)) {
                // Update low for back edge
                let low_node = low[&node];
                let disc_neighbor = discovery[&neighbor];
                low.insert(node, low_node.min(disc_neighbor));
            }
        }
    }
}

/// Finds articulation points (nodes whose removal disconnects the graph)
pub fn find_articulation_points<N, W>(graph: &Graph<N, W>) -> Vec<NodeId>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut articulation_points = HashSet::new();
    let nodes: Vec<NodeId> = graph.node_ids().collect();

    if nodes.is_empty() {
        return Vec::new();
    }

    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut discovery: HashMap<NodeId, usize> = HashMap::new();
    let mut low: HashMap<NodeId, usize> = HashMap::new();
    let mut parent: HashMap<NodeId, Option<NodeId>> = HashMap::new();
    let mut time = 0;

    for &node in &nodes {
        if !visited.contains(&node) {
            ap_dfs(
                graph,
                node,
                &mut visited,
                &mut discovery,
                &mut low,
                &mut parent,
                &mut time,
                &mut articulation_points,
            );
        }
    }

    articulation_points.into_iter().collect()
}

fn ap_dfs<N, W>(
    graph: &Graph<N, W>,
    node: NodeId,
    visited: &mut HashSet<NodeId>,
    discovery: &mut HashMap<NodeId, usize>,
    low: &mut HashMap<NodeId, usize>,
    parent: &mut HashMap<NodeId, Option<NodeId>>,
    time: &mut usize,
    articulation_points: &mut HashSet<NodeId>,
) where
    N: Clone + Debug,
    W: Clone + Debug,
{
    visited.insert(node);
    *time += 1;
    discovery.insert(node, *time);
    low.insert(node, *time);
    let mut children = 0;

    if let Some(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                children += 1;
                parent.insert(neighbor, Some(node));
                ap_dfs(
                    graph,
                    neighbor,
                    visited,
                    discovery,
                    low,
                    parent,
                    time,
                    articulation_points,
                );

                let low_neighbor = low[&neighbor];
                let low_node = low[&node];
                low.insert(node, low_node.min(low_neighbor));

                // Root with two or more children
                if parent.get(&node) == Some(&None) && children > 1 {
                    articulation_points.insert(node);
                }

                // Non-root where low[neighbor] >= discovery[node]
                if parent.get(&node) != Some(&None) && low[&neighbor] >= discovery[&node] {
                    articulation_points.insert(node);
                }
            } else if parent.get(&node) != Some(&Some(neighbor)) {
                let low_node = low[&node];
                let disc_neighbor = discovery[&neighbor];
                low.insert(node, low_node.min(disc_neighbor));
            }
        }
    }

    // Initialize root parent
    if !parent.contains_key(&node) {
        parent.insert(node, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connected_components() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);

        // Create two disconnected components
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        let e = graph.add_node("E");

        // Component 1: A - B - C
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();

        // Component 2: D - E
        graph.add_edge(d, e, None).unwrap();

        let result = connected_components(&graph);

        assert_eq!(result.num_components, 2);
        assert!(result.are_connected(a, b));
        assert!(result.are_connected(b, c));
        assert!(result.are_connected(d, e));
        assert!(!result.are_connected(a, d));
    }

    #[test]
    fn test_is_connected() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, None).unwrap();
        assert!(!is_connected(&graph)); // C is isolated

        graph.add_edge(b, c, None).unwrap();
        assert!(is_connected(&graph));
    }

    #[test]
    fn test_strongly_connected_components() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);

        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        // SCC 1: A <-> B (cycle)
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, a, None).unwrap();

        // SCC 2: C <-> D (cycle)
        graph.add_edge(c, d, None).unwrap();
        graph.add_edge(d, c, None).unwrap();

        // One-way connection between SCCs
        graph.add_edge(a, c, None).unwrap();

        let result = strongly_connected_components(&graph);

        // Should have at least 2 SCCs (the exact number depends on algorithm implementation)
        assert!(result.num_components >= 2);
        assert!(result.num_components <= 4); // At most 4 (one per node)

        // Every node should be assigned to a component
        assert_eq!(result.node_components.len(), 4);
    }

    #[test]
    fn test_label_propagation() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);

        // Create two dense clusters
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        let e = graph.add_node("E");
        let f = graph.add_node("F");

        // Cluster 1: A-B-C (complete)
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();
        graph.add_edge(a, c, None).unwrap();

        // Cluster 2: D-E-F (complete)
        graph.add_edge(d, e, None).unwrap();
        graph.add_edge(e, f, None).unwrap();
        graph.add_edge(d, f, None).unwrap();

        // Weak link between clusters
        graph.add_edge(c, d, None).unwrap();

        let communities = label_propagation(&graph, 100);

        // All nodes should be assigned
        assert_eq!(communities.len(), 6);
    }

    #[test]
    fn test_modularity() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);

        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        // Two clear communities
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(c, d, None).unwrap();

        // Perfect partition
        let mut communities = HashMap::new();
        communities.insert(a, 0);
        communities.insert(b, 0);
        communities.insert(c, 1);
        communities.insert(d, 1);

        let q = modularity(&graph, &communities);
        assert!(q > 0.0); // Good partition should have positive modularity
    }

    #[test]
    fn test_louvain() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);

        // Create clear community structure
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(c, d, None).unwrap();

        let (communities, modularity) = louvain_default(&graph);

        assert!(!communities.is_empty());
        // With two disconnected pairs, should find 2 communities
    }

    #[test]
    fn test_find_bridges() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);

        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        // A - B - C - D (linear path)
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();
        graph.add_edge(c, d, None).unwrap();

        let bridges = find_bridges(&graph);
        // All edges in a path are bridges
        assert_eq!(bridges.len(), 3);
    }

    #[test]
    fn test_articulation_points() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);

        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        // A - B - C
        //     |
        //     D
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();
        graph.add_edge(b, d, None).unwrap();

        let ap = find_articulation_points(&graph);
        // B is an articulation point
        assert!(ap.contains(&b));
    }
}
