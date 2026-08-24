//! Wave 3 regression tests for `pandrs::graph`.
//!
//! Covers the Yen's k-shortest-paths fix: the previous implementation ran
//! Dijkstra on the *unmodified* graph for every spur and only rejected a
//! candidate whose very first edge collided with a previously used one, so
//! paths 2..k were frequently missing or duplicates of path 1. The fix
//! builds a genuinely modified copy of the graph per spur (edges used by
//! same-root paths removed, earlier root-path nodes removed) before running
//! Dijkstra again.

use pandrs::graph::core::{Edge, Graph, GraphType, NodeId};
use pandrs::graph::path::k_shortest_paths;

/// Small graph, hand-verified by exhaustive enumeration:
///
/// ```text
/// S --1--> A --1--> B --1--> T
/// S --5--------------------> B
/// A --10-------------------> T
/// ```
///
/// All source->target paths and their costs:
///   S->A->B->T = 1+1+1 = 3   (cheapest)
///   S->B->T    = 5+1   = 6
///   S->A->T    = 1+10  = 11  (most expensive)
///
/// so `k_shortest_paths(.., 3, ..)` must return exactly these three paths,
/// in ascending-cost order [3, 6, 11].
struct TestGraph {
    graph: Graph<&'static str, f64>,
    s: NodeId,
    a: NodeId,
    b: NodeId,
    t: NodeId,
}

fn build_graph() -> TestGraph {
    let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
    let s = graph.add_node("S");
    let a = graph.add_node("A");
    let b = graph.add_node("B");
    let t = graph.add_node("T");

    graph.add_edge(s, a, Some(1.0)).expect("add S->A");
    graph.add_edge(a, b, Some(1.0)).expect("add A->B");
    graph.add_edge(b, t, Some(1.0)).expect("add B->T");
    graph.add_edge(s, b, Some(5.0)).expect("add S->B");
    graph.add_edge(a, t, Some(10.0)).expect("add A->T");

    TestGraph { graph, s, a, b, t }
}

fn weight(edge: &Edge<f64>) -> f64 {
    edge.weight.unwrap_or(1.0)
}

#[test]
fn test_yen_k_shortest_paths_finds_all_three() {
    let TestGraph {
        graph,
        s: source,
        a,
        b,
        t: target,
    } = build_graph();

    let paths = k_shortest_paths(&graph, source, target, 3, weight);

    // The previous implementation's shallow "first edge only" collision
    // check happens to reject every candidate for this graph (each
    // candidate's first edge collides with an already-used one), so
    // `candidates` ends up empty after the very first iteration and the
    // old code returns just 1 path here, not 3. This is the crux of the
    // regression: the fix must actually try the alternatives.
    assert_eq!(
        paths.len(),
        3,
        "expected all 3 source->target paths, got {:?}",
        paths.iter().map(|(_, c)| *c).collect::<Vec<_>>()
    );

    let costs: Vec<f64> = paths.iter().map(|(_, c)| *c).collect();
    assert!(
        (costs[0] - 3.0).abs() < 1e-9,
        "1st path cost should be 3.0 (S-A-B-T), got {}",
        costs[0]
    );
    assert!(
        (costs[1] - 6.0).abs() < 1e-9,
        "2nd path cost should be 6.0 (S-B-T), got {}",
        costs[1]
    );
    assert!(
        (costs[2] - 11.0).abs() < 1e-9,
        "3rd path cost should be 11.0 (S-A-T), got {}",
        costs[2]
    );

    // Costs must be non-decreasing (Yen's algorithm always emits paths in
    // ascending cost order).
    assert!(costs[0] <= costs[1] && costs[1] <= costs[2]);

    // Every returned path must actually start at source and end at target.
    for (path, _) in &paths {
        assert_eq!(path.first(), Some(&source));
        assert_eq!(path.last(), Some(&target));
    }

    // The three paths must be distinct (no duplicates from an
    // insufficiently-modified search graph).
    let node_paths: Vec<&Vec<NodeId>> = paths.iter().map(|(p, _)| p).collect();
    assert_ne!(node_paths[0], node_paths[1]);
    assert_ne!(node_paths[1], node_paths[2]);
    assert_ne!(node_paths[0], node_paths[2]);

    // And they must match the exact hand-verified node sequences.
    assert_eq!(*node_paths[0], vec![source, a, b, target]); // S-A-B-T, cost 3
    assert_eq!(*node_paths[1], vec![source, b, target]); // S-B-T, cost 6
    assert_eq!(*node_paths[2], vec![source, a, target]); // S-A-T, cost 11
}

#[test]
fn test_yen_k_shortest_paths_k_larger_than_available_stops_early() {
    let TestGraph {
        graph,
        s: source,
        t: target,
        ..
    } = build_graph();

    // Only 3 distinct source->target paths exist in this graph; asking for
    // more must return exactly those 3, not loop forever or fabricate
    // extras.
    let paths = k_shortest_paths(&graph, source, target, 10, weight);
    assert_eq!(paths.len(), 3);
}

#[test]
fn test_yen_k_shortest_paths_k_zero_returns_empty() {
    let TestGraph {
        graph,
        s: source,
        t: target,
        ..
    } = build_graph();
    let paths = k_shortest_paths(&graph, source, target, 0, weight);
    assert!(paths.is_empty());
}

#[test]
fn test_yen_k_shortest_paths_first_path_matches_dijkstra() {
    use pandrs::graph::path::dijkstra;

    let TestGraph {
        graph,
        s: source,
        t: target,
        ..
    } = build_graph();
    let dijkstra_result = dijkstra(&graph, source, weight).expect("dijkstra");
    let dijkstra_cost = dijkstra_result.distance_to(target).expect("reachable");

    let paths = k_shortest_paths(&graph, source, target, 1, weight);
    assert_eq!(paths.len(), 1);
    assert!((paths[0].1 - dijkstra_cost).abs() < 1e-9);
}

#[test]
fn test_yen_k_shortest_paths_no_path_returns_empty() {
    let mut graph: Graph<&str, f64> = Graph::new(GraphType::Directed);
    let a = graph.add_node("A");
    let b = graph.add_node("B");
    // No edge between A and B: unreachable.
    let paths = k_shortest_paths(&graph, a, b, 3, weight);
    assert!(paths.is_empty());
}
