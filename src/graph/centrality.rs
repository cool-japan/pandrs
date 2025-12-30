//! Centrality metrics for graph analysis
//!
//! This module provides various centrality measures:
//! - Degree centrality
//! - Closeness centrality
//! - Betweenness centrality
//! - Eigenvector centrality
//! - PageRank
//! - Katz centrality
//!
//! # Examples
//!
//! ```
//! use pandrs::graph::{Graph, GraphType};
//! use pandrs::graph::centrality::{degree_centrality, pagerank};
//!
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
//! // ... add nodes and edges ...
//! ```

use super::core::{Graph, GraphType, NodeId};
use super::traversal::bfs;
use std::collections::HashMap;
use std::fmt::Debug;

/// Calculates the degree centrality for all nodes
///
/// Degree centrality is the fraction of nodes that a node is connected to.
/// For directed graphs, this calculates the total degree (in + out).
///
/// # Returns
/// A map from node ID to its degree centrality (0.0 to 1.0)
pub fn degree_centrality<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n <= 1 {
        return graph
            .node_ids()
            .map(|id| (id, if n == 1 { 1.0 } else { 0.0 }))
            .collect();
    }

    let max_degree = (n - 1) as f64;

    graph
        .node_ids()
        .filter_map(|id| graph.degree(id).map(|deg| (id, deg as f64 / max_degree)))
        .collect()
}

/// Calculates the in-degree centrality for all nodes (directed graphs)
pub fn in_degree_centrality<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n <= 1 {
        return graph
            .node_ids()
            .map(|id| (id, if n == 1 { 1.0 } else { 0.0 }))
            .collect();
    }

    let max_degree = (n - 1) as f64;

    graph
        .node_ids()
        .filter_map(|id| graph.in_degree(id).map(|deg| (id, deg as f64 / max_degree)))
        .collect()
}

/// Calculates the out-degree centrality for all nodes (directed graphs)
pub fn out_degree_centrality<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n <= 1 {
        return graph
            .node_ids()
            .map(|id| (id, if n == 1 { 1.0 } else { 0.0 }))
            .collect();
    }

    let max_degree = (n - 1) as f64;

    graph
        .node_ids()
        .filter_map(|id| {
            graph
                .out_degree(id)
                .map(|deg| (id, deg as f64 / max_degree))
        })
        .collect()
}

/// Calculates the closeness centrality for all nodes
///
/// Closeness centrality is the reciprocal of the average shortest path distance
/// from a node to all other reachable nodes.
///
/// # Returns
/// A map from node ID to its closeness centrality
pub fn closeness_centrality<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n <= 1 {
        return graph.node_ids().map(|id| (id, 0.0)).collect();
    }

    let mut centrality = HashMap::new();

    for node_id in graph.node_ids() {
        let result = bfs(graph, node_id);
        let reachable: Vec<&usize> = result.distance.values().filter(|&&d| d > 0).collect();

        if reachable.is_empty() {
            centrality.insert(node_id, 0.0);
        } else {
            let total_distance: usize = reachable.iter().map(|&&d| d).sum();
            let num_reachable = reachable.len() as f64;

            // Wasserman and Faust formula for disconnected graphs
            let closeness = if total_distance > 0 {
                (num_reachable / (n as f64 - 1.0)) * (num_reachable / total_distance as f64)
            } else {
                0.0
            };

            centrality.insert(node_id, closeness);
        }
    }

    centrality
}

/// Calculates the betweenness centrality for all nodes
///
/// Betweenness centrality measures how often a node lies on the shortest path
/// between other pairs of nodes.
///
/// # Returns
/// A map from node ID to its betweenness centrality
pub fn betweenness_centrality<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    let mut centrality: HashMap<NodeId, f64> = graph.node_ids().map(|id| (id, 0.0)).collect();

    if n <= 2 {
        return centrality;
    }

    let nodes: Vec<NodeId> = graph.node_ids().collect();

    // Use Brandes' algorithm for unweighted graphs
    for &source in &nodes {
        let mut stack: Vec<NodeId> = Vec::new();
        let mut predecessors: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut sigma: HashMap<NodeId, f64> = HashMap::new();
        let mut distance: HashMap<NodeId, i64> = HashMap::new();

        for &node in &nodes {
            predecessors.insert(node, Vec::new());
            sigma.insert(node, 0.0);
            distance.insert(node, -1);
        }

        sigma.insert(source, 1.0);
        distance.insert(source, 0);

        // BFS
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let dist_v = distance[&v];

            if let Some(neighbors) = graph.neighbors(v) {
                for w in neighbors {
                    // First time visiting w?
                    if distance[&w] < 0 {
                        queue.push_back(w);
                        distance.insert(w, dist_v + 1);
                    }

                    // Is this a shortest path to w via v?
                    if distance[&w] == dist_v + 1 {
                        let sigma_v = sigma[&v];
                        *sigma.get_mut(&w).unwrap() += sigma_v;
                        predecessors.get_mut(&w).unwrap().push(v);
                    }
                }
            }
        }

        // Accumulation phase
        let mut delta: HashMap<NodeId, f64> = nodes.iter().map(|&id| (id, 0.0)).collect();

        while let Some(w) = stack.pop() {
            for &v in &predecessors[&w] {
                let coeff = (sigma[&v] / sigma[&w]) * (1.0 + delta[&w]);
                *delta.get_mut(&v).unwrap() += coeff;
            }

            if w != source {
                *centrality.get_mut(&w).unwrap() += delta[&w];
            }
        }
    }

    // Normalize
    let scale = if graph.graph_type() == GraphType::Directed {
        1.0 / ((n as f64 - 1.0) * (n as f64 - 2.0))
    } else {
        2.0 / ((n as f64 - 1.0) * (n as f64 - 2.0))
    };

    for value in centrality.values_mut() {
        *value *= scale;
    }

    centrality
}

/// Calculates PageRank for all nodes
///
/// PageRank is an iterative algorithm that measures the importance of nodes
/// based on the link structure.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `damping_factor` - Probability of following a link (typically 0.85)
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// A map from node ID to its PageRank score
pub fn pagerank<N, W>(
    graph: &Graph<N, W>,
    damping_factor: f64,
    max_iterations: usize,
    tolerance: f64,
) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let nodes: Vec<NodeId> = graph.node_ids().collect();

    // Initialize PageRank uniformly
    let initial_rank = 1.0 / n as f64;
    let mut rank: HashMap<NodeId, f64> = nodes.iter().map(|&id| (id, initial_rank)).collect();

    // Pre-compute out-degrees
    let out_degrees: HashMap<NodeId, usize> = nodes
        .iter()
        .map(|&id| (id, graph.out_degree(id).unwrap_or(0)))
        .collect();

    // Find dangling nodes (nodes with no outgoing edges)
    let dangling_nodes: Vec<NodeId> = nodes
        .iter()
        .filter(|&id| out_degrees[id] == 0)
        .copied()
        .collect();

    let base_rank = (1.0 - damping_factor) / n as f64;

    for _ in 0..max_iterations {
        let mut new_rank: HashMap<NodeId, f64> = HashMap::new();

        // Calculate dangling sum
        let dangling_sum: f64 = dangling_nodes.iter().map(|&id| rank[&id]).sum();
        let dangling_contrib = damping_factor * dangling_sum / n as f64;

        for &node in &nodes {
            let mut incoming_sum = 0.0;

            // Get predecessors (nodes that link to this node)
            if let Some(predecessors) = graph.predecessors(node) {
                for pred in predecessors {
                    let pred_out_degree = out_degrees[&pred];
                    if pred_out_degree > 0 {
                        incoming_sum += rank[&pred] / pred_out_degree as f64;
                    }
                }
            }

            let new_pr = base_rank + damping_factor * incoming_sum + dangling_contrib;
            new_rank.insert(node, new_pr);
        }

        // Check convergence
        let diff: f64 = nodes
            .iter()
            .map(|&id| (rank[&id] - new_rank[&id]).abs())
            .sum();

        rank = new_rank;

        if diff < tolerance {
            break;
        }
    }

    rank
}

/// Convenience function for PageRank with default parameters
pub fn pagerank_default<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    pagerank(graph, 0.85, 100, 1e-6)
}

/// Calculates eigenvector centrality using power iteration
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// A map from node ID to its eigenvector centrality
pub fn eigenvector_centrality<N, W>(
    graph: &Graph<N, W>,
    max_iterations: usize,
    tolerance: f64,
) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let nodes: Vec<NodeId> = graph.node_ids().collect();

    // Initialize with uniform values
    let initial = 1.0 / (n as f64).sqrt();
    let mut centrality: HashMap<NodeId, f64> = nodes.iter().map(|&id| (id, initial)).collect();

    for _ in 0..max_iterations {
        let mut new_centrality: HashMap<NodeId, f64> = HashMap::new();

        for &node in &nodes {
            let mut sum = 0.0;

            // Sum centralities of neighbors
            if let Some(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    sum += centrality[&neighbor];
                }
            }

            new_centrality.insert(node, sum);
        }

        // Normalize
        let norm: f64 = new_centrality.values().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for value in new_centrality.values_mut() {
                *value /= norm;
            }
        }

        // Check convergence
        let diff: f64 = nodes
            .iter()
            .map(|&id| (centrality[&id] - new_centrality[&id]).abs())
            .sum();

        centrality = new_centrality;

        if diff < tolerance {
            break;
        }
    }

    centrality
}

/// Convenience function for eigenvector centrality with default parameters
pub fn eigenvector_centrality_default<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    eigenvector_centrality(graph, 100, 1e-6)
}

/// Calculates Katz centrality
///
/// Katz centrality computes centrality based on the total number of walks
/// between nodes, weighted by the walk length.
///
/// # Arguments
/// * `graph` - The graph to analyze
/// * `alpha` - Attenuation factor (must be less than 1/lambda_max)
/// * `beta` - Weight attributed to immediate neighbors
/// * `max_iterations` - Maximum number of iterations
/// * `tolerance` - Convergence tolerance
pub fn katz_centrality<N, W>(
    graph: &Graph<N, W>,
    alpha: f64,
    beta: f64,
    max_iterations: usize,
    tolerance: f64,
) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n == 0 {
        return HashMap::new();
    }

    let nodes: Vec<NodeId> = graph.node_ids().collect();

    // Initialize with beta
    let mut centrality: HashMap<NodeId, f64> = nodes.iter().map(|&id| (id, 0.0)).collect();

    for _ in 0..max_iterations {
        let mut new_centrality: HashMap<NodeId, f64> = HashMap::new();

        for &node in &nodes {
            let mut sum = beta;

            // Sum alpha * centrality of neighbors
            if let Some(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    sum += alpha * centrality[&neighbor];
                }
            }

            new_centrality.insert(node, sum);
        }

        // Check convergence
        let diff: f64 = nodes
            .iter()
            .map(|&id| (centrality[&id] - new_centrality[&id]).abs())
            .sum();

        centrality = new_centrality;

        if diff < tolerance {
            break;
        }
    }

    // Normalize
    let max_val = centrality
        .values()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_val > 0.0 {
        for value in centrality.values_mut() {
            *value /= max_val;
        }
    }

    centrality
}

/// Convenience function for Katz centrality with default parameters
pub fn katz_centrality_default<N, W>(graph: &Graph<N, W>) -> HashMap<NodeId, f64>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    katz_centrality(graph, 0.1, 1.0, 100, 1e-6)
}

/// Calculates hub and authority scores (HITS algorithm)
///
/// # Returns
/// A tuple of (hub_scores, authority_scores)
pub fn hits<N, W>(
    graph: &Graph<N, W>,
    max_iterations: usize,
    tolerance: f64,
) -> (HashMap<NodeId, f64>, HashMap<NodeId, f64>)
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let n = graph.node_count();
    if n == 0 {
        return (HashMap::new(), HashMap::new());
    }

    let nodes: Vec<NodeId> = graph.node_ids().collect();

    // Initialize uniformly
    let initial = 1.0 / n as f64;
    let mut hubs: HashMap<NodeId, f64> = nodes.iter().map(|&id| (id, initial)).collect();
    let mut authorities: HashMap<NodeId, f64> = nodes.iter().map(|&id| (id, initial)).collect();

    for _ in 0..max_iterations {
        let mut new_authorities: HashMap<NodeId, f64> = HashMap::new();
        let mut new_hubs: HashMap<NodeId, f64> = HashMap::new();

        // Update authorities (based on hub scores of predecessors)
        for &node in &nodes {
            let mut auth_sum = 0.0;
            if let Some(predecessors) = graph.predecessors(node) {
                for pred in predecessors {
                    auth_sum += hubs[&pred];
                }
            }
            new_authorities.insert(node, auth_sum);
        }

        // Update hubs (based on authority scores of successors)
        for &node in &nodes {
            let mut hub_sum = 0.0;
            if let Some(neighbors) = graph.neighbors(node) {
                for neighbor in neighbors {
                    hub_sum += new_authorities[&neighbor];
                }
            }
            new_hubs.insert(node, hub_sum);
        }

        // Normalize authorities
        let auth_norm: f64 = new_authorities.values().map(|&v| v * v).sum::<f64>().sqrt();
        if auth_norm > 0.0 {
            for value in new_authorities.values_mut() {
                *value /= auth_norm;
            }
        }

        // Normalize hubs
        let hub_norm: f64 = new_hubs.values().map(|&v| v * v).sum::<f64>().sqrt();
        if hub_norm > 0.0 {
            for value in new_hubs.values_mut() {
                *value /= hub_norm;
            }
        }

        // Check convergence
        let auth_diff: f64 = nodes
            .iter()
            .map(|&id| (authorities[&id] - new_authorities[&id]).abs())
            .sum();
        let hub_diff: f64 = nodes
            .iter()
            .map(|&id| (hubs[&id] - new_hubs[&id]).abs())
            .sum();

        authorities = new_authorities;
        hubs = new_hubs;

        if auth_diff < tolerance && hub_diff < tolerance {
            break;
        }
    }

    (hubs, authorities)
}

/// Convenience function for HITS with default parameters
pub fn hits_default<N, W>(graph: &Graph<N, W>) -> (HashMap<NodeId, f64>, HashMap<NodeId, f64>)
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    hits(graph, 100, 1e-6)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_star_graph() -> Graph<&'static str, f64> {
        // Star graph: center connected to 4 peripheral nodes
        let mut graph = Graph::new(GraphType::Undirected);
        let center = graph.add_node("center");
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        graph.add_edge(center, a, None).unwrap();
        graph.add_edge(center, b, None).unwrap();
        graph.add_edge(center, c, None).unwrap();
        graph.add_edge(center, d, None).unwrap();

        graph
    }

    #[test]
    fn test_degree_centrality() {
        let graph = create_star_graph();
        let centrality = degree_centrality(&graph);

        // Center node should have highest degree centrality (4/4 = 1.0)
        let center = NodeId(0);
        assert!((centrality[&center] - 1.0).abs() < 1e-6);

        // Peripheral nodes should have degree centrality of 1/4 = 0.25
        let a = NodeId(1);
        assert!((centrality[&a] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_closeness_centrality() {
        let graph = create_star_graph();
        let centrality = closeness_centrality(&graph);

        // Center node should have highest closeness centrality
        let center = NodeId(0);
        let peripheral = NodeId(1);

        assert!(centrality[&center] > centrality[&peripheral]);
    }

    #[test]
    fn test_betweenness_centrality() {
        let graph = create_star_graph();
        let centrality = betweenness_centrality(&graph);

        // Center node should have highest betweenness
        // All shortest paths between peripheral nodes go through center
        let center = NodeId(0);
        let peripheral = NodeId(1);

        assert!(centrality[&center] > centrality[&peripheral]);
        assert!((centrality[&peripheral] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_pagerank() {
        let graph = create_star_graph();
        let pr = pagerank_default(&graph);

        // All PageRank values should sum to approximately 1
        let sum: f64 = pr.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_eigenvector_centrality() {
        let graph = create_star_graph();
        let centrality = eigenvector_centrality_default(&graph);

        // All nodes should have computed centrality
        assert!(!centrality.is_empty());
        assert_eq!(centrality.len(), 5);

        // All centrality values should be non-negative
        for &val in centrality.values() {
            assert!(val >= 0.0);
        }

        // Check that values are normalized (sum of squares ≈ 1)
        let sum_sq: f64 = centrality.values().map(|&v| v * v).sum();
        assert!((sum_sq - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_hits() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        // A -> B -> C
        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();

        let (hubs, authorities) = hits_default(&graph);

        // A is a pure hub (only outgoing edges from perspective of where links go)
        // C is a pure authority (only incoming edges)
        assert!(hubs[&a] > authorities[&a]);
        assert!(authorities[&c] > hubs[&c]);
    }
}
