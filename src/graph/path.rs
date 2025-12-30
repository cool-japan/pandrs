//! Shortest path algorithms for weighted graphs
//!
//! This module provides:
//! - Dijkstra's algorithm (non-negative weights)
//! - Bellman-Ford algorithm (handles negative weights)
//! - Floyd-Warshall algorithm (all-pairs shortest paths)
//! - A* search (with heuristic)
//!
//! # Examples
//!
//! ```
//! use pandrs::graph::{Graph, GraphType};
//! use pandrs::graph::path::dijkstra;
//!
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
//! // ... add weighted edges ...
//! ```

use super::core::{Edge, Graph, GraphError, NodeId};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::Debug;

/// Result of a shortest path computation
#[derive(Debug, Clone)]
pub struct ShortestPathResult {
    /// Distance from source to each reachable node
    pub distances: HashMap<NodeId, f64>,
    /// Parent node in the shortest path tree
    pub predecessors: HashMap<NodeId, Option<NodeId>>,
}

impl ShortestPathResult {
    /// Reconstructs the path from source to target
    pub fn path_to(&self, target: NodeId) -> Option<Vec<NodeId>> {
        if !self.predecessors.contains_key(&target) {
            return None;
        }

        let mut path = Vec::new();
        let mut current = target;

        loop {
            path.push(current);
            match self.predecessors.get(&current)? {
                Some(pred) => current = *pred,
                None => break, // Reached source
            }
        }

        path.reverse();
        Some(path)
    }

    /// Returns the distance to a specific node
    pub fn distance_to(&self, target: NodeId) -> Option<f64> {
        self.distances.get(&target).copied()
    }
}

/// State for Dijkstra's priority queue
#[derive(Debug, Clone)]
struct DijkstraState {
    node: NodeId,
    distance: f64,
}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for DijkstraState {}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Performs Dijkstra's shortest path algorithm
///
/// # Arguments
/// * `graph` - The graph (edge weights must be non-negative)
/// * `source` - The starting node
/// * `weight_fn` - Function to extract weight from edge (returns f64)
///
/// # Returns
/// Shortest path result or error if negative weights encountered
pub fn dijkstra<N, W, F>(
    graph: &Graph<N, W>,
    source: NodeId,
    weight_fn: F,
) -> Result<ShortestPathResult, GraphError>
where
    N: Clone + Debug,
    W: Clone + Debug,
    F: Fn(&Edge<W>) -> f64,
{
    let mut distances: HashMap<NodeId, f64> = HashMap::new();
    let mut predecessors: HashMap<NodeId, Option<NodeId>> = HashMap::new();
    let mut heap: BinaryHeap<DijkstraState> = BinaryHeap::new();
    let mut visited: HashSet<NodeId> = HashSet::new();

    // Initialize source
    distances.insert(source, 0.0);
    predecessors.insert(source, None);
    heap.push(DijkstraState {
        node: source,
        distance: 0.0,
    });

    while let Some(DijkstraState { node, distance }) = heap.pop() {
        if visited.contains(&node) {
            continue;
        }
        visited.insert(node);

        // Skip if we've found a better path already
        if let Some(&best) = distances.get(&node) {
            if distance > best {
                continue;
            }
        }

        // Check all neighbors
        if let Some(neighbors) = graph.neighbors(node) {
            for neighbor in neighbors {
                if let Some(edge) = graph.get_edge_between(node, neighbor) {
                    let weight = weight_fn(edge);

                    if weight < 0.0 {
                        return Err(GraphError::InvalidOperation(
                            "Dijkstra's algorithm requires non-negative weights".to_string(),
                        ));
                    }

                    let new_distance = distance + weight;

                    let is_better = distances
                        .get(&neighbor)
                        .map(|&d| new_distance < d)
                        .unwrap_or(true);

                    if is_better {
                        distances.insert(neighbor, new_distance);
                        predecessors.insert(neighbor, Some(node));
                        heap.push(DijkstraState {
                            node: neighbor,
                            distance: new_distance,
                        });
                    }
                }
            }
        }
    }

    Ok(ShortestPathResult {
        distances,
        predecessors,
    })
}

/// Dijkstra's algorithm with default weight extraction (assumes `Option<f64>` weights)
pub fn dijkstra_default<N>(
    graph: &Graph<N, f64>,
    source: NodeId,
) -> Result<ShortestPathResult, GraphError>
where
    N: Clone + Debug,
{
    dijkstra(graph, source, |edge| edge.weight.unwrap_or(1.0))
}

/// Performs Bellman-Ford shortest path algorithm
///
/// Can handle negative edge weights but will detect negative cycles.
///
/// # Arguments
/// * `graph` - The graph
/// * `source` - The starting node
/// * `weight_fn` - Function to extract weight from edge
///
/// # Returns
/// Shortest path result or error if negative cycle detected
pub fn bellman_ford<N, W, F>(
    graph: &Graph<N, W>,
    source: NodeId,
    weight_fn: F,
) -> Result<ShortestPathResult, GraphError>
where
    N: Clone + Debug,
    W: Clone + Debug,
    F: Fn(&Edge<W>) -> f64,
{
    let n = graph.node_count();
    let mut distances: HashMap<NodeId, f64> = HashMap::new();
    let mut predecessors: HashMap<NodeId, Option<NodeId>> = HashMap::new();

    // Initialize
    for node_id in graph.node_ids() {
        distances.insert(node_id, f64::INFINITY);
        predecessors.insert(node_id, None);
    }
    distances.insert(source, 0.0);

    // Relax edges n-1 times
    for _ in 0..n - 1 {
        let mut changed = false;

        for (_, edge) in graph.edges() {
            let weight = weight_fn(edge);
            let src_dist = distances[&edge.source];

            if src_dist < f64::INFINITY {
                let new_dist = src_dist + weight;
                if new_dist < distances[&edge.target] {
                    distances.insert(edge.target, new_dist);
                    predecessors.insert(edge.target, Some(edge.source));
                    changed = true;
                }
            }
        }

        // Early termination if no changes
        if !changed {
            break;
        }
    }

    // Check for negative cycles
    for (_, edge) in graph.edges() {
        let weight = weight_fn(edge);
        let src_dist = distances[&edge.source];

        if src_dist < f64::INFINITY && src_dist + weight < distances[&edge.target] {
            return Err(GraphError::NegativeWeightCycle);
        }
    }

    Ok(ShortestPathResult {
        distances,
        predecessors,
    })
}

/// Bellman-Ford with default weight extraction
pub fn bellman_ford_default<N>(
    graph: &Graph<N, f64>,
    source: NodeId,
) -> Result<ShortestPathResult, GraphError>
where
    N: Clone + Debug,
{
    bellman_ford(graph, source, |edge| edge.weight.unwrap_or(1.0))
}

/// Result of Floyd-Warshall all-pairs shortest paths
#[derive(Debug, Clone)]
pub struct AllPairsShortestPaths {
    /// Distance matrix: `distances[i][j]` = shortest path from i to j
    pub distances: HashMap<NodeId, HashMap<NodeId, f64>>,
    /// Next hop matrix for path reconstruction
    pub next: HashMap<NodeId, HashMap<NodeId, Option<NodeId>>>,
}

impl AllPairsShortestPaths {
    /// Reconstructs the path between two nodes
    pub fn path(&self, source: NodeId, target: NodeId) -> Option<Vec<NodeId>> {
        if self.distances.get(&source)?.get(&target)? == &f64::INFINITY {
            return None;
        }

        let mut path = vec![source];
        let mut current = source;

        while current != target {
            current = self.next.get(&current)?.get(&target)?.as_ref().copied()?;
            path.push(current);
        }

        Some(path)
    }

    /// Gets the distance between two nodes
    pub fn distance(&self, source: NodeId, target: NodeId) -> Option<f64> {
        self.distances.get(&source)?.get(&target).copied()
    }
}

/// Performs Floyd-Warshall all-pairs shortest paths algorithm
///
/// # Arguments
/// * `graph` - The graph
/// * `weight_fn` - Function to extract weight from edge
pub fn floyd_warshall<N, W, F>(
    graph: &Graph<N, W>,
    weight_fn: F,
) -> Result<AllPairsShortestPaths, GraphError>
where
    N: Clone + Debug,
    W: Clone + Debug,
    F: Fn(&Edge<W>) -> f64,
{
    let nodes: Vec<NodeId> = graph.node_ids().collect();
    let n = nodes.len();

    // Initialize distance and next matrices
    let mut dist: HashMap<NodeId, HashMap<NodeId, f64>> = HashMap::new();
    let mut next: HashMap<NodeId, HashMap<NodeId, Option<NodeId>>> = HashMap::new();

    for &i in &nodes {
        let mut dist_row: HashMap<NodeId, f64> = HashMap::new();
        let mut next_row: HashMap<NodeId, Option<NodeId>> = HashMap::new();

        for &j in &nodes {
            if i == j {
                dist_row.insert(j, 0.0);
                next_row.insert(j, None);
            } else {
                dist_row.insert(j, f64::INFINITY);
                next_row.insert(j, None);
            }
        }

        dist.insert(i, dist_row);
        next.insert(i, next_row);
    }

    // Initialize direct edges
    for (_, edge) in graph.edges() {
        let weight = weight_fn(edge);
        dist.get_mut(&edge.source)
            .unwrap()
            .insert(edge.target, weight);
        next.get_mut(&edge.source)
            .unwrap()
            .insert(edge.target, Some(edge.target));
    }

    // Main Floyd-Warshall loop
    for &k in &nodes {
        for &i in &nodes {
            for &j in &nodes {
                let ik = dist[&i][&k];
                let kj = dist[&k][&j];

                if ik < f64::INFINITY && kj < f64::INFINITY {
                    let through_k = ik + kj;
                    if through_k < dist[&i][&j] {
                        dist.get_mut(&i).unwrap().insert(j, through_k);
                        let k_next = next[&i][&k];
                        next.get_mut(&i).unwrap().insert(j, k_next);
                    }
                }
            }
        }
    }

    // Check for negative cycles (diagonal should be 0)
    for &i in &nodes {
        if dist[&i][&i] < 0.0 {
            return Err(GraphError::NegativeWeightCycle);
        }
    }

    Ok(AllPairsShortestPaths {
        distances: dist,
        next,
    })
}

/// Floyd-Warshall with default weight extraction
pub fn floyd_warshall_default<N>(graph: &Graph<N, f64>) -> Result<AllPairsShortestPaths, GraphError>
where
    N: Clone + Debug,
{
    floyd_warshall(graph, |edge| edge.weight.unwrap_or(1.0))
}

/// A* search algorithm state
#[derive(Debug, Clone)]
struct AStarState {
    node: NodeId,
    f_score: f64, // f = g + h
    g_score: f64, // Actual cost from start
}

impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}

impl Eq for AStarState {}

impl PartialOrd for AStarState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AStarState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Performs A* search algorithm
///
/// # Arguments
/// * `graph` - The graph
/// * `source` - The starting node
/// * `target` - The target node
/// * `weight_fn` - Function to extract weight from edge
/// * `heuristic` - Function to estimate remaining distance to target
///
/// # Returns
/// The path and its total cost, or None if no path exists
pub fn astar<N, W, F, H>(
    graph: &Graph<N, W>,
    source: NodeId,
    target: NodeId,
    weight_fn: F,
    heuristic: H,
) -> Option<(Vec<NodeId>, f64)>
where
    N: Clone + Debug,
    W: Clone + Debug,
    F: Fn(&Edge<W>) -> f64,
    H: Fn(NodeId) -> f64,
{
    let mut open_set: BinaryHeap<AStarState> = BinaryHeap::new();
    let mut g_score: HashMap<NodeId, f64> = HashMap::new();
    let mut came_from: HashMap<NodeId, NodeId> = HashMap::new();
    let mut closed_set: HashSet<NodeId> = HashSet::new();

    g_score.insert(source, 0.0);
    open_set.push(AStarState {
        node: source,
        f_score: heuristic(source),
        g_score: 0.0,
    });

    while let Some(AStarState {
        node: current,
        g_score: current_g,
        ..
    }) = open_set.pop()
    {
        if current == target {
            // Reconstruct path
            let mut path = vec![current];
            let mut node = current;
            while let Some(&prev) = came_from.get(&node) {
                path.push(prev);
                node = prev;
            }
            path.reverse();
            return Some((path, current_g));
        }

        if closed_set.contains(&current) {
            continue;
        }
        closed_set.insert(current);

        if let Some(neighbors) = graph.neighbors(current) {
            for neighbor in neighbors {
                if closed_set.contains(&neighbor) {
                    continue;
                }

                if let Some(edge) = graph.get_edge_between(current, neighbor) {
                    let weight = weight_fn(edge);
                    let tentative_g = current_g + weight;

                    let is_better = g_score
                        .get(&neighbor)
                        .map(|&g| tentative_g < g)
                        .unwrap_or(true);

                    if is_better {
                        came_from.insert(neighbor, current);
                        g_score.insert(neighbor, tentative_g);

                        let f = tentative_g + heuristic(neighbor);
                        open_set.push(AStarState {
                            node: neighbor,
                            f_score: f,
                            g_score: tentative_g,
                        });
                    }
                }
            }
        }
    }

    None // No path found
}

/// Finds all shortest paths between source and target (for unweighted graphs)
pub fn all_shortest_paths<N, W>(
    graph: &Graph<N, W>,
    source: NodeId,
    target: NodeId,
) -> Vec<Vec<NodeId>>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    use super::traversal::bfs;

    let result = bfs(graph, source);

    // Check if target is reachable
    if !result.distance.contains_key(&target) {
        return vec![];
    }

    let target_dist = result.distance[&target];

    // Use BFS to find all paths of minimum length
    let mut all_paths: Vec<Vec<NodeId>> = Vec::new();
    let mut current_paths: Vec<Vec<NodeId>> = vec![vec![source]];

    for _ in 0..target_dist {
        let mut next_paths: Vec<Vec<NodeId>> = Vec::new();

        for path in current_paths {
            let last = *path.last().unwrap();

            if let Some(neighbors) = graph.neighbors(last) {
                for neighbor in neighbors {
                    // Only extend if it's on a shortest path
                    if result.distance.get(&neighbor) == Some(&(result.distance[&last] + 1)) {
                        let mut new_path = path.clone();
                        new_path.push(neighbor);

                        if neighbor == target {
                            all_paths.push(new_path);
                        } else {
                            next_paths.push(new_path);
                        }
                    }
                }
            }
        }

        current_paths = next_paths;
    }

    all_paths
}

/// Computes the k-shortest paths between source and target
/// Uses Yen's algorithm
pub fn k_shortest_paths<N, W, F>(
    graph: &Graph<N, W>,
    source: NodeId,
    target: NodeId,
    k: usize,
    weight_fn: F,
) -> Vec<(Vec<NodeId>, f64)>
where
    N: Clone + Debug,
    W: Clone + Debug,
    F: Fn(&Edge<W>) -> f64 + Clone,
{
    let mut result: Vec<(Vec<NodeId>, f64)> = Vec::new();
    let mut candidates: Vec<(Vec<NodeId>, f64)> = Vec::new();

    // Find first shortest path
    if let Ok(sp_result) = dijkstra(graph, source, &weight_fn) {
        if let Some(path) = sp_result.path_to(target) {
            let cost = sp_result.distance_to(target).unwrap_or(f64::INFINITY);
            result.push((path, cost));
        }
    }

    if result.is_empty() {
        return result;
    }

    while result.len() < k {
        let (last_path, _) = &result[result.len() - 1];

        for i in 0..last_path.len() - 1 {
            let spur_node = last_path[i];
            let root_path: Vec<NodeId> = last_path[..=i].to_vec();

            // Create a modified graph by removing edges
            let mut removed_edges: Vec<(NodeId, NodeId)> = Vec::new();

            for (path, _) in &result {
                if path.len() > i && path[..=i] == root_path {
                    removed_edges.push((path[i], path[i + 1]));
                }
            }

            // Calculate spur path using modified graph (without actually modifying)
            // This is a simplified version - full implementation would modify graph
            if let Ok(sp_result) = dijkstra(graph, spur_node, &weight_fn) {
                if let Some(spur_path) = sp_result.path_to(target) {
                    // Check if we need to skip removed edges
                    let mut valid = true;
                    if !spur_path.is_empty() {
                        let first_edge = (spur_node, spur_path[1]);
                        if removed_edges.contains(&first_edge) {
                            valid = false;
                        }
                    }

                    if valid && spur_path.len() > 1 {
                        let mut total_path = root_path.clone();
                        total_path.pop(); // Remove spur node (it's already in spur_path)
                        total_path.extend(spur_path);

                        let total_cost = total_path
                            .windows(2)
                            .map(|w| {
                                graph
                                    .get_edge_between(w[0], w[1])
                                    .map(|e| weight_fn(e))
                                    .unwrap_or(f64::INFINITY)
                            })
                            .sum();

                        // Check if this path is already in candidates or result
                        let path_exists = candidates.iter().any(|(p, _)| *p == total_path)
                            || result.iter().any(|(p, _)| *p == total_path);

                        if !path_exists {
                            candidates.push((total_path, total_cost));
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Find the candidate with minimum cost
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let best = candidates.remove(0);
        result.push(best);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::core::GraphType;

    fn create_weighted_graph() -> Graph<&'static str, f64> {
        let mut graph = Graph::new(GraphType::Directed);

        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        let e = graph.add_node("E");

        // Create weighted edges
        //     A --1--> B --2--> C
        //     |        |        |
        //     4        1        1
        //     v        v        v
        //     D --1--> E <------+
        graph.add_edge(a, b, Some(1.0)).unwrap();
        graph.add_edge(b, c, Some(2.0)).unwrap();
        graph.add_edge(a, d, Some(4.0)).unwrap();
        graph.add_edge(b, e, Some(1.0)).unwrap();
        graph.add_edge(c, e, Some(1.0)).unwrap();
        graph.add_edge(d, e, Some(1.0)).unwrap();

        graph
    }

    #[test]
    fn test_dijkstra() {
        let graph = create_weighted_graph();
        let a = NodeId(0);
        let e = NodeId(4);

        let result = dijkstra_default(&graph, a).unwrap();

        // Shortest path A -> B -> E = 1 + 1 = 2
        assert!((result.distance_to(e).unwrap() - 2.0).abs() < 1e-6);

        let path = result.path_to(e).unwrap();
        assert_eq!(path.len(), 3); // A, B, E
    }

    #[test]
    fn test_bellman_ford() {
        let graph = create_weighted_graph();
        let a = NodeId(0);
        let e = NodeId(4);

        let result = bellman_ford_default(&graph, a).unwrap();

        // Should give same result as Dijkstra for this graph
        assert!((result.distance_to(e).unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_floyd_warshall() {
        let graph = create_weighted_graph();

        let apsp = floyd_warshall_default(&graph).unwrap();

        let a = NodeId(0);
        let e = NodeId(4);

        assert!((apsp.distance(a, e).unwrap() - 2.0).abs() < 1e-6);

        let path = apsp.path(a, e).unwrap();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_astar() {
        let graph = create_weighted_graph();
        let a = NodeId(0);
        let e = NodeId(4);

        // Simple heuristic (always 0, degenerates to Dijkstra)
        let result = astar(&graph, a, e, |edge| edge.weight.unwrap_or(1.0), |_| 0.0);

        assert!(result.is_some());
        let (path, cost) = result.unwrap();
        assert!((cost - 2.0).abs() < 1e-6);
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn test_negative_weights_dijkstra() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");

        graph.add_edge(a, b, Some(-1.0)).unwrap();

        let result = dijkstra_default(&graph, a);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_weights_bellman_ford() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, Some(1.0)).unwrap();
        graph.add_edge(b, c, Some(-1.0)).unwrap();

        // Should work fine - negative weight but no negative cycle
        let result = bellman_ford_default(&graph, a);
        assert!(result.is_ok());
    }

    #[test]
    fn test_negative_cycle_detection() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, Some(1.0)).unwrap();
        graph.add_edge(b, c, Some(-2.0)).unwrap();
        graph.add_edge(c, b, Some(0.5)).unwrap(); // Creates negative cycle B -> C -> B

        let result = bellman_ford_default(&graph, a);
        assert!(matches!(result, Err(GraphError::NegativeWeightCycle)));
    }
}
