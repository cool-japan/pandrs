//! Graph traversal algorithms
//!
//! This module provides various graph traversal algorithms including:
//! - Breadth-First Search (BFS)
//! - Depth-First Search (DFS)
//! - Topological Sort (for DAGs)
//!
//! # Examples
//!
//! ```
//! use pandrs::graph::{Graph, GraphType, bfs, dfs};
//!
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! let c = graph.add_node("C");
//! graph.add_edge(a, b, None).unwrap();
//! graph.add_edge(b, c, None).unwrap();
//!
//! // BFS traversal starting from node A
//! let order = bfs(&graph, a);
//! ```

use super::core::{Graph, GraphError, GraphType, NodeId};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

/// Result of a BFS traversal
#[derive(Debug, Clone)]
pub struct BfsResult {
    /// Order in which nodes were visited
    pub visit_order: Vec<NodeId>,
    /// Parent of each node in the BFS tree (None for the source)
    pub parent: HashMap<NodeId, Option<NodeId>>,
    /// Distance from source to each node
    pub distance: HashMap<NodeId, usize>,
}

/// Result of a DFS traversal
#[derive(Debug, Clone)]
pub struct DfsResult {
    /// Order in which nodes were first discovered
    pub discovery_order: Vec<NodeId>,
    /// Order in which nodes were finished (all descendants visited)
    pub finish_order: Vec<NodeId>,
    /// Parent of each node in the DFS tree (None for roots)
    pub parent: HashMap<NodeId, Option<NodeId>>,
    /// Discovery time for each node
    pub discovery_time: HashMap<NodeId, usize>,
    /// Finish time for each node
    pub finish_time: HashMap<NodeId, usize>,
}

/// Performs Breadth-First Search starting from a source node
///
/// # Arguments
/// * `graph` - The graph to traverse
/// * `source` - The starting node
///
/// # Returns
/// A `BfsResult` containing the visit order, parent map, and distances
pub fn bfs<N, W>(graph: &Graph<N, W>, source: NodeId) -> BfsResult
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut visit_order = Vec::new();
    let mut parent: HashMap<NodeId, Option<NodeId>> = HashMap::new();
    let mut distance: HashMap<NodeId, usize> = HashMap::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();

    // Initialize source
    visited.insert(source);
    parent.insert(source, None);
    distance.insert(source, 0);
    queue.push_back(source);

    while let Some(current) = queue.pop_front() {
        visit_order.push(current);

        if let Some(neighbors) = graph.neighbors(current) {
            let current_dist = distance[&current];

            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, Some(current));
                    distance.insert(neighbor, current_dist + 1);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    BfsResult {
        visit_order,
        parent,
        distance,
    }
}

/// Performs BFS and returns only the nodes reachable from the source
pub fn bfs_reachable<N, W>(graph: &Graph<N, W>, source: NodeId) -> HashSet<NodeId>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    bfs(graph, source).visit_order.into_iter().collect()
}

/// Finds the shortest path (in terms of edge count) between two nodes using BFS
///
/// # Returns
/// The path as a vector of node IDs, or None if no path exists
pub fn shortest_path_bfs<N, W>(
    graph: &Graph<N, W>,
    source: NodeId,
    target: NodeId,
) -> Option<Vec<NodeId>>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let result = bfs(graph, source);

    if !result.parent.contains_key(&target) {
        return None; // Target not reachable
    }

    // Reconstruct path by following parents
    let mut path = Vec::new();
    let mut current = target;

    while let Some(parent) = result.parent.get(&current) {
        path.push(current);
        match parent {
            Some(p) => current = *p,
            None => break, // Reached source
        }
    }

    path.reverse();
    Some(path)
}

/// Performs Depth-First Search on the entire graph
///
/// # Arguments
/// * `graph` - The graph to traverse
///
/// # Returns
/// A `DfsResult` containing discovery/finish times and orders
pub fn dfs<N, W>(graph: &Graph<N, W>) -> DfsResult
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut discovery_order = Vec::new();
    let mut finish_order = Vec::new();
    let mut parent: HashMap<NodeId, Option<NodeId>> = HashMap::new();
    let mut discovery_time: HashMap<NodeId, usize> = HashMap::new();
    let mut finish_time: HashMap<NodeId, usize> = HashMap::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut time = 0;

    // Visit all nodes (handles disconnected graphs)
    for node_id in graph.node_ids() {
        if !visited.contains(&node_id) {
            dfs_visit(
                graph,
                node_id,
                None,
                &mut visited,
                &mut discovery_order,
                &mut finish_order,
                &mut parent,
                &mut discovery_time,
                &mut finish_time,
                &mut time,
            );
        }
    }

    DfsResult {
        discovery_order,
        finish_order,
        parent,
        discovery_time,
        finish_time,
    }
}

/// DFS starting from a specific source node
pub fn dfs_from<N, W>(graph: &Graph<N, W>, source: NodeId) -> DfsResult
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut discovery_order = Vec::new();
    let mut finish_order = Vec::new();
    let mut parent: HashMap<NodeId, Option<NodeId>> = HashMap::new();
    let mut discovery_time: HashMap<NodeId, usize> = HashMap::new();
    let mut finish_time: HashMap<NodeId, usize> = HashMap::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut time = 0;

    dfs_visit(
        graph,
        source,
        None,
        &mut visited,
        &mut discovery_order,
        &mut finish_order,
        &mut parent,
        &mut discovery_time,
        &mut finish_time,
        &mut time,
    );

    DfsResult {
        discovery_order,
        finish_order,
        parent,
        discovery_time,
        finish_time,
    }
}

/// Helper function for recursive DFS
fn dfs_visit<N, W>(
    graph: &Graph<N, W>,
    node: NodeId,
    node_parent: Option<NodeId>,
    visited: &mut HashSet<NodeId>,
    discovery_order: &mut Vec<NodeId>,
    finish_order: &mut Vec<NodeId>,
    parent: &mut HashMap<NodeId, Option<NodeId>>,
    discovery_time: &mut HashMap<NodeId, usize>,
    finish_time: &mut HashMap<NodeId, usize>,
    time: &mut usize,
) where
    N: Clone + Debug,
    W: Clone + Debug,
{
    visited.insert(node);
    parent.insert(node, node_parent);
    *time += 1;
    discovery_time.insert(node, *time);
    discovery_order.push(node);

    if let Some(neighbors) = graph.neighbors(node) {
        for neighbor in neighbors {
            if !visited.contains(&neighbor) {
                dfs_visit(
                    graph,
                    neighbor,
                    Some(node),
                    visited,
                    discovery_order,
                    finish_order,
                    parent,
                    discovery_time,
                    finish_time,
                    time,
                );
            }
        }
    }

    *time += 1;
    finish_time.insert(node, *time);
    finish_order.push(node);
}

/// Iterative DFS that doesn't risk stack overflow on deep graphs
pub fn dfs_iterative<N, W>(graph: &Graph<N, W>, source: NodeId) -> Vec<NodeId>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut visit_order = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut stack: Vec<NodeId> = vec![source];

    while let Some(current) = stack.pop() {
        if visited.contains(&current) {
            continue;
        }

        visited.insert(current);
        visit_order.push(current);

        if let Some(neighbors) = graph.neighbors(current) {
            // Push neighbors in reverse order to maintain natural DFS order
            for neighbor in neighbors.into_iter().rev() {
                if !visited.contains(&neighbor) {
                    stack.push(neighbor);
                }
            }
        }
    }

    visit_order
}

/// Performs topological sort on a directed acyclic graph (DAG)
///
/// # Returns
/// A vector of nodes in topological order, or an error if the graph has a cycle
pub fn topological_sort<N, W>(graph: &Graph<N, W>) -> Result<Vec<NodeId>, GraphError>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    if graph.graph_type() != GraphType::Directed {
        return Err(GraphError::InvalidOperation(
            "Topological sort requires a directed graph".to_string(),
        ));
    }

    let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
    let mut result = Vec::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();

    // Calculate in-degrees
    for node_id in graph.node_ids() {
        in_degree.insert(node_id, graph.in_degree(node_id).unwrap_or(0));
    }

    // Add all nodes with in-degree 0 to the queue
    for (&node_id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node_id);
        }
    }

    while let Some(current) = queue.pop_front() {
        result.push(current);

        if let Some(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if let Some(degree) = in_degree.get_mut(&neighbor) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    // If we didn't process all nodes, there's a cycle
    if result.len() != graph.node_count() {
        return Err(GraphError::CycleDetected);
    }

    Ok(result)
}

/// Checks if a directed graph has a cycle
pub fn has_cycle<N, W>(graph: &Graph<N, W>) -> bool
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    if graph.graph_type() != GraphType::Directed {
        // Undirected graphs always have cycles if they have any edges
        return graph.edge_count() > 0;
    }

    topological_sort(graph).is_err()
}

/// Finds all nodes at a given distance from the source
pub fn nodes_at_distance<N, W>(graph: &Graph<N, W>, source: NodeId, distance: usize) -> Vec<NodeId>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let result = bfs(graph, source);
    result
        .distance
        .iter()
        .filter_map(|(&node, &dist)| if dist == distance { Some(node) } else { None })
        .collect()
}

/// Returns the eccentricity of a node (maximum distance to any other node)
pub fn eccentricity<N, W>(graph: &Graph<N, W>, node: NodeId) -> Option<usize>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let result = bfs(graph, node);

    if result.distance.len() != graph.node_count() {
        // Graph is not connected from this node
        return None;
    }

    result.distance.values().copied().max()
}

/// Returns the diameter of the graph (maximum eccentricity)
pub fn diameter<N, W>(graph: &Graph<N, W>) -> Option<usize>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut max_eccentricity = 0;

    for node_id in graph.node_ids() {
        match eccentricity(graph, node_id) {
            Some(e) => max_eccentricity = max_eccentricity.max(e),
            None => return None, // Graph is not connected
        }
    }

    Some(max_eccentricity)
}

/// Returns the radius of the graph (minimum eccentricity)
pub fn radius<N, W>(graph: &Graph<N, W>) -> Option<usize>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let mut min_eccentricity = usize::MAX;

    for node_id in graph.node_ids() {
        match eccentricity(graph, node_id) {
            Some(e) => min_eccentricity = min_eccentricity.min(e),
            None => return None, // Graph is not connected
        }
    }

    if min_eccentricity == usize::MAX {
        None
    } else {
        Some(min_eccentricity)
    }
}

/// Returns the center of the graph (nodes with eccentricity equal to the radius)
pub fn center<N, W>(graph: &Graph<N, W>) -> Option<Vec<NodeId>>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let r = radius(graph)?;

    let center_nodes: Vec<NodeId> = graph
        .node_ids()
        .filter(|&node_id| eccentricity(graph, node_id) == Some(r))
        .collect();

    Some(center_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph<&'static str, f64> {
        let mut graph = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        let e = graph.add_node("E");

        // Create a simple path: A - B - C - D - E
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();
        graph.add_edge(c, d, None).unwrap();
        graph.add_edge(d, e, None).unwrap();

        graph
    }

    #[test]
    fn test_bfs() {
        let graph = create_test_graph();
        let a = NodeId(0);

        let result = bfs(&graph, a);

        assert_eq!(result.visit_order.len(), 5);
        assert_eq!(result.visit_order[0], a);
        assert_eq!(result.distance[&a], 0);
        assert_eq!(result.distance[&NodeId(4)], 4); // E is 4 edges away from A
    }

    #[test]
    fn test_shortest_path_bfs() {
        let graph = create_test_graph();
        let a = NodeId(0);
        let e = NodeId(4);

        let path = shortest_path_bfs(&graph, a, e).unwrap();

        assert_eq!(path.len(), 5);
        assert_eq!(path[0], a);
        assert_eq!(path[4], e);
    }

    #[test]
    fn test_dfs() {
        let graph = create_test_graph();

        let result = dfs(&graph);

        assert_eq!(result.discovery_order.len(), 5);
        assert_eq!(result.finish_order.len(), 5);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        // A -> B -> D
        // A -> C -> D
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(a, c, None).unwrap();
        graph.add_edge(b, d, None).unwrap();
        graph.add_edge(c, d, None).unwrap();

        let sorted = topological_sort(&graph).unwrap();

        // A must come before B and C
        // B and C must come before D
        let pos_a = sorted.iter().position(|&n| n == a).unwrap();
        let pos_b = sorted.iter().position(|&n| n == b).unwrap();
        let pos_c = sorted.iter().position(|&n| n == c).unwrap();
        let pos_d = sorted.iter().position(|&n| n == d).unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_has_cycle() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        // A -> B -> C (no cycle)
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();

        assert!(!has_cycle(&graph));

        // Add C -> A to create a cycle
        graph.add_edge(c, a, None).unwrap();
        assert!(has_cycle(&graph));
    }

    #[test]
    fn test_eccentricity_diameter_radius() {
        let graph = create_test_graph();
        let a = NodeId(0);
        let c = NodeId(2); // Center node in a path

        // In a path A-B-C-D-E:
        // - Eccentricity of A is 4 (distance to E)
        // - Eccentricity of C is 2 (distance to A or E)
        assert_eq!(eccentricity(&graph, a), Some(4));
        assert_eq!(eccentricity(&graph, c), Some(2));

        // Diameter is the maximum eccentricity (4)
        assert_eq!(diameter(&graph), Some(4));

        // Radius is the minimum eccentricity (2)
        assert_eq!(radius(&graph), Some(2));
    }
}
