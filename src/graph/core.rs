//! Core graph data structures for graph analytics
//!
//! This module provides the fundamental building blocks for graph representation:
//! - `Graph`: The main graph container supporting directed/undirected and weighted/unweighted graphs
//! - `Node`: Vertex representation with optional data
//! - `Edge`: Edge representation with optional weights
//!
//! # Examples
//!
//! ```
//! use pandrs::graph::{Graph, GraphType};
//!
//! // Create an undirected graph
//! let mut graph: Graph<String, f64> = Graph::new(GraphType::Undirected);
//! let a = graph.add_node("A".to_string());
//! let b = graph.add_node("B".to_string());
//! graph.add_edge(a, b, Some(1.0));
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Display};
use std::hash::Hash;

/// Represents the type of graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphType {
    /// Edges have no direction
    Undirected,
    /// Edges have a direction from source to target
    Directed,
}

/// Unique identifier for a node in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

impl Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Unique identifier for an edge in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId(pub(crate) usize);

impl Display for EdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Edge({})", self.0)
    }
}

/// A node (vertex) in the graph
#[derive(Debug, Clone)]
pub struct Node<N> {
    /// Unique identifier
    pub id: NodeId,
    /// Optional data associated with the node
    pub data: N,
    /// Outgoing edge IDs (or all edges for undirected graphs)
    pub(crate) outgoing: Vec<EdgeId>,
    /// Incoming edge IDs (only for directed graphs)
    pub(crate) incoming: Vec<EdgeId>,
}

impl<N> Node<N> {
    /// Creates a new node with the given ID and data
    pub fn new(id: NodeId, data: N) -> Self {
        Node {
            id,
            data,
            outgoing: Vec::new(),
            incoming: Vec::new(),
        }
    }

    /// Returns the degree of the node (number of edges)
    pub fn degree(&self) -> usize {
        self.outgoing.len() + self.incoming.len()
    }

    /// Returns the out-degree of the node
    pub fn out_degree(&self) -> usize {
        self.outgoing.len()
    }

    /// Returns the in-degree of the node
    pub fn in_degree(&self) -> usize {
        self.incoming.len()
    }
}

/// An edge in the graph
#[derive(Debug, Clone)]
pub struct Edge<W> {
    /// Unique identifier
    pub id: EdgeId,
    /// Source node ID
    pub source: NodeId,
    /// Target node ID
    pub target: NodeId,
    /// Optional weight
    pub weight: Option<W>,
}

impl<W> Edge<W> {
    /// Creates a new edge
    pub fn new(id: EdgeId, source: NodeId, target: NodeId, weight: Option<W>) -> Self {
        Edge {
            id,
            source,
            target,
            weight,
        }
    }
}

/// Error types for graph operations
#[derive(Debug, Clone)]
pub enum GraphError {
    /// Node not found
    NodeNotFound(NodeId),
    /// Edge not found
    EdgeNotFound(EdgeId),
    /// Invalid operation
    InvalidOperation(String),
    /// Cycle detected (for operations that require acyclic graphs)
    CycleDetected,
    /// Graph is not connected
    NotConnected,
    /// Negative weight cycle detected
    NegativeWeightCycle,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeNotFound(id) => write!(f, "Node not found: {}", id),
            GraphError::EdgeNotFound(id) => write!(f, "Edge not found: {}", id),
            GraphError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            GraphError::CycleDetected => write!(f, "Cycle detected in graph"),
            GraphError::NotConnected => write!(f, "Graph is not connected"),
            GraphError::NegativeWeightCycle => write!(f, "Negative weight cycle detected"),
        }
    }
}

impl std::error::Error for GraphError {}

/// A graph data structure supporting both directed and undirected graphs
///
/// Type parameters:
/// - `N`: Node data type
/// - `W`: Edge weight type
#[derive(Debug, Clone)]
pub struct Graph<N, W> {
    /// Type of graph (directed or undirected)
    graph_type: GraphType,
    /// All nodes in the graph
    nodes: HashMap<NodeId, Node<N>>,
    /// All edges in the graph
    edges: HashMap<EdgeId, Edge<W>>,
    /// Counter for generating unique node IDs
    next_node_id: usize,
    /// Counter for generating unique edge IDs
    next_edge_id: usize,
    /// Adjacency list for fast neighbor lookup (source -> [(target, edge_id)])
    adjacency: HashMap<NodeId, Vec<(NodeId, EdgeId)>>,
    /// Reverse adjacency list for directed graphs (target -> [(source, edge_id)])
    reverse_adjacency: HashMap<NodeId, Vec<(NodeId, EdgeId)>>,
}

impl<N, W> Graph<N, W>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    /// Creates a new empty graph
    pub fn new(graph_type: GraphType) -> Self {
        Graph {
            graph_type,
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    /// Returns the type of the graph
    pub fn graph_type(&self) -> GraphType {
        self.graph_type
    }

    /// Returns the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Checks if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Adds a node to the graph and returns its ID
    pub fn add_node(&mut self, data: N) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;

        let node = Node::new(id, data);
        self.nodes.insert(id, node);
        self.adjacency.insert(id, Vec::new());
        self.reverse_adjacency.insert(id, Vec::new());

        id
    }

    /// Removes a node and all its connected edges from the graph
    pub fn remove_node(&mut self, node_id: NodeId) -> Result<N, GraphError> {
        // First, collect all edges connected to this node
        let edges_to_remove: Vec<EdgeId> = self
            .edges
            .iter()
            .filter(|(_, edge)| edge.source == node_id || edge.target == node_id)
            .map(|(id, _)| *id)
            .collect();

        // Remove all connected edges
        for edge_id in edges_to_remove {
            self.remove_edge(edge_id)?;
        }

        // Remove the node itself
        let node = self
            .nodes
            .remove(&node_id)
            .ok_or(GraphError::NodeNotFound(node_id))?;

        self.adjacency.remove(&node_id);
        self.reverse_adjacency.remove(&node_id);

        Ok(node.data)
    }

    /// Gets a reference to a node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&Node<N>> {
        self.nodes.get(&node_id)
    }

    /// Gets a mutable reference to a node by ID
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut Node<N>> {
        self.nodes.get_mut(&node_id)
    }

    /// Adds an edge between two nodes
    pub fn add_edge(
        &mut self,
        source: NodeId,
        target: NodeId,
        weight: Option<W>,
    ) -> Result<EdgeId, GraphError> {
        // Check that both nodes exist
        if !self.nodes.contains_key(&source) {
            return Err(GraphError::NodeNotFound(source));
        }
        if !self.nodes.contains_key(&target) {
            return Err(GraphError::NodeNotFound(target));
        }

        let edge_id = EdgeId(self.next_edge_id);
        self.next_edge_id += 1;

        let edge = Edge::new(edge_id, source, target, weight);
        self.edges.insert(edge_id, edge);

        // Update adjacency lists
        self.adjacency
            .get_mut(&source)
            .unwrap()
            .push((target, edge_id));

        if self.graph_type == GraphType::Directed {
            self.reverse_adjacency
                .get_mut(&target)
                .unwrap()
                .push((source, edge_id));

            // Update node edge lists
            self.nodes.get_mut(&source).unwrap().outgoing.push(edge_id);
            self.nodes.get_mut(&target).unwrap().incoming.push(edge_id);
        } else {
            // For undirected graphs, add the reverse edge in adjacency
            self.adjacency
                .get_mut(&target)
                .unwrap()
                .push((source, edge_id));

            // Update node edge lists (both directions)
            self.nodes.get_mut(&source).unwrap().outgoing.push(edge_id);
            self.nodes.get_mut(&target).unwrap().outgoing.push(edge_id);
        }

        Ok(edge_id)
    }

    /// Removes an edge from the graph
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> Result<Edge<W>, GraphError> {
        let edge = self
            .edges
            .remove(&edge_id)
            .ok_or(GraphError::EdgeNotFound(edge_id))?;

        // Update adjacency lists
        if let Some(neighbors) = self.adjacency.get_mut(&edge.source) {
            neighbors.retain(|(_, eid)| *eid != edge_id);
        }

        if self.graph_type == GraphType::Directed {
            if let Some(neighbors) = self.reverse_adjacency.get_mut(&edge.target) {
                neighbors.retain(|(_, eid)| *eid != edge_id);
            }

            // Update node edge lists
            if let Some(node) = self.nodes.get_mut(&edge.source) {
                node.outgoing.retain(|eid| *eid != edge_id);
            }
            if let Some(node) = self.nodes.get_mut(&edge.target) {
                node.incoming.retain(|eid| *eid != edge_id);
            }
        } else {
            // For undirected graphs
            if let Some(neighbors) = self.adjacency.get_mut(&edge.target) {
                neighbors.retain(|(_, eid)| *eid != edge_id);
            }

            // Update node edge lists
            if let Some(node) = self.nodes.get_mut(&edge.source) {
                node.outgoing.retain(|eid| *eid != edge_id);
            }
            if let Some(node) = self.nodes.get_mut(&edge.target) {
                node.outgoing.retain(|eid| *eid != edge_id);
            }
        }

        Ok(edge)
    }

    /// Gets a reference to an edge by ID
    pub fn get_edge(&self, edge_id: EdgeId) -> Option<&Edge<W>> {
        self.edges.get(&edge_id)
    }

    /// Gets a mutable reference to an edge by ID
    pub fn get_edge_mut(&mut self, edge_id: EdgeId) -> Option<&mut Edge<W>> {
        self.edges.get_mut(&edge_id)
    }

    /// Returns an iterator over all nodes
    pub fn nodes(&self) -> impl Iterator<Item = (&NodeId, &Node<N>)> {
        self.nodes.iter()
    }

    /// Returns an iterator over all edges
    pub fn edges(&self) -> impl Iterator<Item = (&EdgeId, &Edge<W>)> {
        self.edges.iter()
    }

    /// Returns an iterator over all node IDs
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    /// Returns an iterator over all edge IDs
    pub fn edge_ids(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.edges.keys().copied()
    }

    /// Returns the neighbors of a node (outgoing edges for directed graphs)
    pub fn neighbors(&self, node_id: NodeId) -> Option<Vec<NodeId>> {
        self.adjacency
            .get(&node_id)
            .map(|neighbors| neighbors.iter().map(|(n, _)| *n).collect())
    }

    /// Returns the predecessors of a node (incoming edges for directed graphs)
    pub fn predecessors(&self, node_id: NodeId) -> Option<Vec<NodeId>> {
        if self.graph_type == GraphType::Directed {
            self.reverse_adjacency
                .get(&node_id)
                .map(|neighbors| neighbors.iter().map(|(n, _)| *n).collect())
        } else {
            // For undirected graphs, predecessors are the same as neighbors
            self.neighbors(node_id)
        }
    }

    /// Returns the edges connected to a node
    pub fn edges_of(&self, node_id: NodeId) -> Option<Vec<EdgeId>> {
        let node = self.nodes.get(&node_id)?;

        if self.graph_type == GraphType::Directed {
            let mut edges: Vec<EdgeId> = node.outgoing.clone();
            edges.extend(node.incoming.iter().cloned());
            Some(edges)
        } else {
            Some(node.outgoing.clone())
        }
    }

    /// Checks if an edge exists between two nodes
    pub fn has_edge(&self, source: NodeId, target: NodeId) -> bool {
        if let Some(neighbors) = self.adjacency.get(&source) {
            neighbors.iter().any(|(n, _)| *n == target)
        } else {
            false
        }
    }

    /// Gets the edge between two nodes if it exists
    pub fn get_edge_between(&self, source: NodeId, target: NodeId) -> Option<&Edge<W>> {
        let neighbors = self.adjacency.get(&source)?;
        let (_, edge_id) = neighbors.iter().find(|(n, _)| *n == target)?;
        self.edges.get(edge_id)
    }

    /// Returns the degree of a node
    pub fn degree(&self, node_id: NodeId) -> Option<usize> {
        self.nodes.get(&node_id).map(|n| {
            if self.graph_type == GraphType::Directed {
                n.outgoing.len() + n.incoming.len()
            } else {
                n.outgoing.len()
            }
        })
    }

    /// Returns the in-degree of a node (for directed graphs)
    pub fn in_degree(&self, node_id: NodeId) -> Option<usize> {
        if self.graph_type == GraphType::Directed {
            self.nodes.get(&node_id).map(|n| n.incoming.len())
        } else {
            self.degree(node_id)
        }
    }

    /// Returns the out-degree of a node (for directed graphs)
    pub fn out_degree(&self, node_id: NodeId) -> Option<usize> {
        self.nodes.get(&node_id).map(|n| n.outgoing.len())
    }

    /// Creates a subgraph containing only the specified nodes and their connecting edges
    pub fn subgraph(&self, node_ids: &[NodeId]) -> Graph<N, W> {
        let node_set: HashSet<NodeId> = node_ids.iter().copied().collect();
        let mut subgraph = Graph::new(self.graph_type);

        // Map old node IDs to new node IDs
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        // Add nodes
        for &node_id in node_ids {
            if let Some(node) = self.nodes.get(&node_id) {
                let new_id = subgraph.add_node(node.data.clone());
                id_map.insert(node_id, new_id);
            }
        }

        // Add edges between nodes in the subgraph
        for edge in self.edges.values() {
            if node_set.contains(&edge.source) && node_set.contains(&edge.target) {
                let new_source = id_map[&edge.source];
                let new_target = id_map[&edge.target];
                let _ = subgraph.add_edge(new_source, new_target, edge.weight.clone());
            }
        }

        subgraph
    }

    /// Returns a reversed version of the graph (only meaningful for directed graphs)
    pub fn reverse(&self) -> Graph<N, W> {
        let mut reversed = Graph::new(self.graph_type);

        // Map old node IDs to new node IDs
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        // Add all nodes
        for (node_id, node) in &self.nodes {
            let new_id = reversed.add_node(node.data.clone());
            id_map.insert(*node_id, new_id);
        }

        // Add reversed edges
        for edge in self.edges.values() {
            let new_source = id_map[&edge.target];
            let new_target = id_map[&edge.source];
            let _ = reversed.add_edge(new_source, new_target, edge.weight.clone());
        }

        reversed
    }

    /// Converts a directed graph to an undirected graph
    pub fn to_undirected(&self) -> Graph<N, W> {
        let mut undirected = Graph::new(GraphType::Undirected);

        // Map old node IDs to new node IDs
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        // Add all nodes
        for (node_id, node) in &self.nodes {
            let new_id = undirected.add_node(node.data.clone());
            id_map.insert(*node_id, new_id);
        }

        // Add edges (avoiding duplicates for undirected)
        let mut added_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
        for edge in self.edges.values() {
            let new_source = id_map[&edge.source];
            let new_target = id_map[&edge.target];

            // Normalize edge direction to avoid duplicates
            let normalized = if new_source.0 < new_target.0 {
                (new_source, new_target)
            } else {
                (new_target, new_source)
            };

            if !added_edges.contains(&normalized) {
                let _ = undirected.add_edge(new_source, new_target, edge.weight.clone());
                added_edges.insert(normalized);
            }
        }

        undirected
    }

    /// Clears all nodes and edges from the graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.adjacency.clear();
        self.reverse_adjacency.clear();
        self.next_node_id = 0;
        self.next_edge_id = 0;
    }
}

impl<N, W> Default for Graph<N, W>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    fn default() -> Self {
        Graph::new(GraphType::Undirected)
    }
}

/// Builder for creating graphs more conveniently
pub struct GraphBuilder<N, W> {
    graph: Graph<N, W>,
    node_map: HashMap<String, NodeId>,
}

impl<N, W> GraphBuilder<N, W>
where
    N: Clone + Debug,
    W: Clone + Debug,
{
    /// Creates a new graph builder
    pub fn new(graph_type: GraphType) -> Self {
        GraphBuilder {
            graph: Graph::new(graph_type),
            node_map: HashMap::new(),
        }
    }

    /// Creates a new undirected graph builder
    pub fn undirected() -> Self {
        Self::new(GraphType::Undirected)
    }

    /// Creates a new directed graph builder
    pub fn directed() -> Self {
        Self::new(GraphType::Directed)
    }

    /// Adds a named node
    pub fn add_node(mut self, name: &str, data: N) -> Self {
        let id = self.graph.add_node(data);
        self.node_map.insert(name.to_string(), id);
        self
    }

    /// Adds an edge between named nodes
    pub fn add_edge(mut self, source: &str, target: &str, weight: Option<W>) -> Self {
        if let (Some(&src), Some(&tgt)) = (self.node_map.get(source), self.node_map.get(target)) {
            let _ = self.graph.add_edge(src, tgt, weight);
        }
        self
    }

    /// Builds and returns the graph
    pub fn build(self) -> Graph<N, W> {
        self.graph
    }

    /// Returns the node ID for a given name
    pub fn get_node_id(&self, name: &str) -> Option<NodeId> {
        self.node_map.get(name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_graph() {
        let graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_add_nodes() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.get_node(a).unwrap().data, "A");
        assert_eq!(graph.get_node(b).unwrap().data, "B");
    }

    #[test]
    fn test_add_edges() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, Some(1.0)).unwrap();
        graph.add_edge(b, c, Some(2.0)).unwrap();

        assert_eq!(graph.edge_count(), 2);
        assert!(graph.has_edge(a, b));
        assert!(graph.has_edge(b, a)); // Undirected, so reverse exists
        assert!(graph.has_edge(b, c));
        assert!(!graph.has_edge(a, c));
    }

    #[test]
    fn test_directed_graph() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
        let a = graph.add_node("A");
        let b = graph.add_node("B");

        graph.add_edge(a, b, Some(1.0)).unwrap();

        assert!(graph.has_edge(a, b));
        assert!(!graph.has_edge(b, a)); // Directed, so reverse doesn't exist
    }

    #[test]
    fn test_neighbors() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(a, c, None).unwrap();

        let neighbors = graph.neighbors(a).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&b));
        assert!(neighbors.contains(&c));
    }

    #[test]
    fn test_graph_builder() {
        let graph: Graph<&str, f64> = GraphBuilder::undirected()
            .add_node("a", "A")
            .add_node("b", "B")
            .add_node("c", "C")
            .add_edge("a", "b", Some(1.0))
            .add_edge("b", "c", Some(2.0))
            .build();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_remove_node() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();

        assert_eq!(graph.edge_count(), 2);

        graph.remove_node(b).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0); // All edges connected to B are removed
    }

    #[test]
    fn test_subgraph() {
        let mut graph: Graph<&str, f64> = Graph::new(GraphType::Undirected);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        graph.add_edge(a, b, None).unwrap();
        graph.add_edge(b, c, None).unwrap();
        graph.add_edge(c, d, None).unwrap();

        let subgraph = graph.subgraph(&[a, b, c]);
        assert_eq!(subgraph.node_count(), 3);
        assert_eq!(subgraph.edge_count(), 2); // Only edges within the subgraph
    }
}
