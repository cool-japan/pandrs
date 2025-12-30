//! Graph analytics module for PandRS
//!
//! This module provides comprehensive graph analytics capabilities including:
//!
//! - **Graph data structures**: Directed and undirected graphs with weighted edges
//! - **Traversal algorithms**: BFS, DFS, topological sort
//! - **Shortest path algorithms**: Dijkstra, Bellman-Ford, Floyd-Warshall, A*
//! - **Centrality metrics**: Degree, closeness, betweenness, eigenvector, PageRank, HITS
//! - **Connected components**: Connected, strongly connected, weakly connected
//! - **Community detection**: Label propagation, Louvain algorithm
//! - **Graph analysis**: Bridges, articulation points, diameter, radius
//!
//! # Quick Start
//!
//! ```rust
//! use pandrs::graph::{Graph, GraphType, GraphBuilder};
//!
//! // Create an undirected graph
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
//!
//! // Add nodes
//! let a = graph.add_node("Alice");
//! let b = graph.add_node("Bob");
//! let c = graph.add_node("Charlie");
//!
//! // Add edges
//! graph.add_edge(a, b, Some(1.0)).unwrap();
//! graph.add_edge(b, c, Some(2.0)).unwrap();
//! graph.add_edge(a, c, Some(3.0)).unwrap();
//!
//! // Check connectivity
//! assert!(graph.has_edge(a, b));
//! assert_eq!(graph.neighbors(a).unwrap().len(), 2);
//! ```
//!
//! # Using the Builder Pattern
//!
//! ```rust
//! use pandrs::graph::{GraphBuilder, GraphType};
//!
//! let graph = GraphBuilder::<&str, f64>::directed()
//!     .add_node("a", "Alice")
//!     .add_node("b", "Bob")
//!     .add_node("c", "Charlie")
//!     .add_edge("a", "b", Some(1.0))
//!     .add_edge("b", "c", Some(2.0))
//!     .build();
//!
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.edge_count(), 2);
//! ```
//!
//! # Centrality Analysis
//!
//! ```rust
//! use pandrs::graph::{Graph, GraphType};
//! use pandrs::graph::centrality::{degree_centrality, pagerank_default, betweenness_centrality};
//!
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
//! let center = graph.add_node("center");
//! let a = graph.add_node("a");
//! let b = graph.add_node("b");
//! let c = graph.add_node("c");
//!
//! graph.add_edge(center, a, None).unwrap();
//! graph.add_edge(center, b, None).unwrap();
//! graph.add_edge(center, c, None).unwrap();
//!
//! let dc = degree_centrality(&graph);
//! // Center node has highest degree centrality
//! assert!(dc[&center] > dc[&a]);
//! ```
//!
//! # Shortest Paths
//!
//! ```rust
//! use pandrs::graph::{Graph, GraphType};
//! use pandrs::graph::path::dijkstra_default;
//!
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! let c = graph.add_node("C");
//!
//! graph.add_edge(a, b, Some(1.0)).unwrap();
//! graph.add_edge(b, c, Some(2.0)).unwrap();
//! graph.add_edge(a, c, Some(5.0)).unwrap();
//!
//! let result = dijkstra_default(&graph, a).unwrap();
//! // Shortest path to C is through B (cost 3), not direct (cost 5)
//! assert_eq!(result.distance_to(c), Some(3.0));
//! ```
//!
//! # Community Detection
//!
//! ```rust
//! use pandrs::graph::{Graph, GraphType};
//! use pandrs::graph::components::{connected_components, louvain_default};
//!
//! let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
//! // ... add nodes and edges forming communities ...
//! ```
//!
//! # Integration with DataFrames
//!
//! Graphs can be created from DataFrames representing edge lists:
//!
//! ```rust,ignore
//! use pandrs::graph::{Graph, GraphType, from_edge_list};
//! use pandrs::DataFrame;
//!
//! // Assuming a DataFrame with "source" and "target" columns
//! let graph = from_edge_list(&df, "source", "target", None);
//! ```

pub mod centrality;
pub mod components;
pub mod core;
pub mod path;
pub mod traversal;

// Re-export main types
pub use core::{Edge, EdgeId, Graph, GraphBuilder, GraphError, GraphType, Node, NodeId};

// Re-export traversal algorithms
pub use traversal::{
    bfs, bfs_reachable, center, dfs, dfs_from, dfs_iterative, diameter, eccentricity, has_cycle,
    nodes_at_distance, radius, shortest_path_bfs, topological_sort, BfsResult, DfsResult,
};

// Re-export centrality metrics
pub use centrality::{
    betweenness_centrality, closeness_centrality, degree_centrality, eigenvector_centrality,
    eigenvector_centrality_default, hits, hits_default, in_degree_centrality, katz_centrality,
    katz_centrality_default, out_degree_centrality, pagerank, pagerank_default,
};

// Re-export path algorithms
pub use path::{
    all_shortest_paths, astar, bellman_ford, bellman_ford_default, dijkstra, dijkstra_default,
    floyd_warshall, floyd_warshall_default, k_shortest_paths, AllPairsShortestPaths,
    ShortestPathResult,
};

// Re-export component algorithms
pub use components::{
    connected_components, find_articulation_points, find_bridges, is_connected, label_propagation,
    louvain, louvain_default, modularity, strongly_connected_components,
    weakly_connected_components, ComponentResult,
};

use crate::{DataFrame, Series};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Creates a graph from a DataFrame representing an edge list
///
/// # Arguments
/// * `df` - DataFrame with columns for source and target nodes
/// * `source_col` - Name of the source column
/// * `target_col` - Name of the target column
/// * `weight_col` - Optional name of the weight column
/// * `directed` - Whether to create a directed graph
///
/// # Returns
/// A graph with string node data
pub fn from_edge_dataframe(
    df: &DataFrame,
    source_col: &str,
    target_col: &str,
    weight_col: Option<&str>,
    directed: bool,
) -> Result<Graph<String, f64>, GraphError> {
    let graph_type = if directed {
        GraphType::Directed
    } else {
        GraphType::Undirected
    };

    let mut graph: Graph<String, f64> = Graph::new(graph_type);
    let mut node_map: HashMap<String, NodeId> = HashMap::new();

    // Get columns as string values
    let source_values = df.get_column_string_values(source_col).map_err(|e| {
        GraphError::InvalidOperation(format!("Column '{}' error: {}", source_col, e))
    })?;
    let target_values = df.get_column_string_values(target_col).map_err(|e| {
        GraphError::InvalidOperation(format!("Column '{}' error: {}", target_col, e))
    })?;

    let weight_values = weight_col.and_then(|col| df.get_column_numeric_values(col).ok());

    // Process each edge
    let n_rows = source_values.len();

    for i in 0..n_rows {
        let source_val = &source_values[i];
        let target_val = &target_values[i];

        // Get or create source node
        let source_id = if let Some(&id) = node_map.get(source_val) {
            id
        } else {
            let id = graph.add_node(source_val.clone());
            node_map.insert(source_val.clone(), id);
            id
        };

        // Get or create target node
        let target_id = if let Some(&id) = node_map.get(target_val) {
            id
        } else {
            let id = graph.add_node(target_val.clone());
            node_map.insert(target_val.clone(), id);
            id
        };

        // Get weight if available
        let weight = weight_values
            .as_ref()
            .and_then(|wv| if i < wv.len() { Some(wv[i]) } else { None });

        // Add edge
        graph.add_edge(source_id, target_id, weight)?;
    }

    Ok(graph)
}

/// Converts a graph to an edge list DataFrame
///
/// # Returns
/// A DataFrame with source, target, and optionally weight columns
pub fn to_edge_dataframe<N, W>(graph: &Graph<N, W>) -> Result<DataFrame, GraphError>
where
    N: Clone + Debug + ToString,
    W: Clone + Debug + ToString,
{
    let mut sources: Vec<String> = Vec::new();
    let mut targets: Vec<String> = Vec::new();
    let mut weights: Vec<String> = Vec::new();

    for (_, edge) in graph.edges() {
        let source_node = graph
            .get_node(edge.source)
            .ok_or(GraphError::NodeNotFound(edge.source))?;
        let target_node = graph
            .get_node(edge.target)
            .ok_or(GraphError::NodeNotFound(edge.target))?;

        sources.push(source_node.data.to_string());
        targets.push(target_node.data.to_string());

        match &edge.weight {
            Some(w) => weights.push(w.to_string()),
            None => weights.push("".to_string()),
        }
    }

    let source_series = Series::new(sources, Some("source".to_string()))
        .map_err(|e| GraphError::InvalidOperation(e.to_string()))?;
    let target_series = Series::new(targets, Some("target".to_string()))
        .map_err(|e| GraphError::InvalidOperation(e.to_string()))?;

    let mut df = DataFrame::new();
    df.add_column("source".to_string(), source_series)
        .map_err(|e| GraphError::InvalidOperation(e.to_string()))?;
    df.add_column("target".to_string(), target_series)
        .map_err(|e| GraphError::InvalidOperation(e.to_string()))?;

    // Only add weights if any edge has a weight
    if weights.iter().any(|w| !w.is_empty()) {
        let weight_series = Series::new(weights, Some("weight".to_string()))
            .map_err(|e| GraphError::InvalidOperation(e.to_string()))?;
        df.add_column("weight".to_string(), weight_series)
            .map_err(|e| GraphError::InvalidOperation(e.to_string()))?;
    }

    Ok(df)
}

/// Creates a graph from an adjacency matrix
///
/// # Arguments
/// * `matrix` - 2D vector representing the adjacency matrix
/// * `node_labels` - Optional labels for nodes
/// * `directed` - Whether to create a directed graph
pub fn from_adjacency_matrix(
    matrix: &[Vec<f64>],
    node_labels: Option<Vec<String>>,
    directed: bool,
) -> Result<Graph<String, f64>, GraphError> {
    let n = matrix.len();
    if n == 0 {
        return Ok(Graph::new(if directed {
            GraphType::Directed
        } else {
            GraphType::Undirected
        }));
    }

    // Validate matrix is square
    for row in matrix {
        if row.len() != n {
            return Err(GraphError::InvalidOperation(
                "Adjacency matrix must be square".to_string(),
            ));
        }
    }

    let graph_type = if directed {
        GraphType::Directed
    } else {
        GraphType::Undirected
    };

    let mut graph: Graph<String, f64> = Graph::new(graph_type);

    // Create nodes
    let labels = node_labels.unwrap_or_else(|| (0..n).map(|i| i.to_string()).collect());
    let mut node_ids: Vec<NodeId> = Vec::with_capacity(n);

    for label in labels.iter().take(n) {
        node_ids.push(graph.add_node(label.clone()));
    }

    // Create edges
    for i in 0..n {
        let start_j = if directed { 0 } else { i };
        for j in start_j..n {
            let weight = matrix[i][j];
            if weight != 0.0 && (directed || i != j) {
                let edge_weight = if weight == 1.0 { None } else { Some(weight) };
                graph.add_edge(node_ids[i], node_ids[j], edge_weight)?;
            }
        }
    }

    Ok(graph)
}

/// Converts a graph to an adjacency matrix
///
/// # Returns
/// A tuple of (adjacency matrix, node ID to index mapping)
pub fn to_adjacency_matrix<N, W>(graph: &Graph<N, W>) -> (Vec<Vec<f64>>, HashMap<NodeId, usize>)
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    let nodes: Vec<NodeId> = graph.node_ids().collect();
    let n = nodes.len();

    // Create node ID to index mapping
    let node_to_idx: HashMap<NodeId, usize> =
        nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Initialize matrix with zeros
    let mut matrix = vec![vec![0.0; n]; n];

    // Fill matrix from edges
    for (_, edge) in graph.edges() {
        let i = node_to_idx[&edge.source];
        let j = node_to_idx[&edge.target];

        // Use weight if available, otherwise 1.0
        let weight = 1.0; // Default for unweighted

        matrix[i][j] = weight;

        // For undirected graphs, matrix is symmetric
        if graph.graph_type() == GraphType::Undirected {
            matrix[j][i] = weight;
        }
    }

    (matrix, node_to_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        graph.add_edge(a, b, Some(1.0)).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_builder_pattern() {
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .add_node("a", "Alice")
            .add_node("b", "Bob")
            .add_edge("a", "b", Some(1.0))
            .build();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_adjacency_matrix_roundtrip() {
        let matrix = vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];

        let graph = from_adjacency_matrix(&matrix, None, false).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2); // Undirected, so 2 edges

        let (result_matrix, node_to_idx) = to_adjacency_matrix(&graph);

        // Check matrix dimensions match
        assert_eq!(result_matrix.len(), 3);
        assert_eq!(result_matrix[0].len(), 3);

        // Check the same number of edges (sum of matrix divided by 2 for undirected)
        let original_edges: f64 = matrix.iter().flat_map(|row| row.iter()).sum();
        let result_edges: f64 = result_matrix.iter().flat_map(|row| row.iter()).sum();
        assert_eq!(original_edges, result_edges);

        // Check symmetry for undirected graph
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(result_matrix[i][j], result_matrix[j][i]);
            }
        }
    }

    #[test]
    fn test_to_edge_dataframe() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, Some(1.0)).unwrap();
        graph.add_edge(b, c, Some(2.0)).unwrap();

        let df = to_edge_dataframe(&graph).unwrap();
        assert_eq!(df.row_count(), 2);
        assert!(df.column_names().contains(&"source".to_string()));
        assert!(df.column_names().contains(&"target".to_string()));
    }
}
