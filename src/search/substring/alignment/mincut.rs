// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimum-weight cuts of an alignment graph.
//!
//! Removing the source and sink leaves a forest: each internal node has
//! at most one outgoing internal edge. Source and terminal edges become
//! costs of placing a node on either side of the cut.
//!
//! A forward pass combines subtree costs. A backward pass assigns sides,
//! preferring the sink side on ties. Edges leaving the source side form
//! the cut, returned in their original order.

use super::graph::{AlignmentGraph, Edge};

/// A cut solver reusable with different weights on the same alignment graph.
/// Each solve takes linear time and reuses the topology and scratch buffers.
pub(in crate::search::substring) struct MinCut<'g> {
    edges: &'g [Edge],
    /// Internal nodes incident to edges, in increasing needle-offset order.
    active: Vec<usize>,
    /// Next internal node, or the sink for a forest root. Zero means unused.
    parent: Vec<usize>,
    /// Weight of the internal edge to the parent; zero for forest roots.
    parent_weight: Vec<i64>,
    /// Cost of choosing the source side minus the cost of choosing the sink
    /// side. Forced assignments are represented separately.
    delta: Vec<i64>,
    /// An unenumerated source edge prevents placing this node on the sink side.
    forced_source: Vec<bool>,
    /// Node assignments reconstructed by the latest solve.
    source_side: Vec<bool>,
    /// Selected edge indices, in original edge order.
    cut: Vec<u32>,
}

impl<'g> MinCut<'g> {
    /// Record the topology once and allocate reusable buffers.
    ///
    /// Alignment construction guarantees forward edges, at most one internal
    /// successor per node, and unenumerated edges only from the source to an
    /// internal node. Nodes are needle offsets, with source 0 and sink last.
    pub(in crate::search::substring) fn new(graph: &'g AlignmentGraph) -> Self {
        let node_count = graph.node_count();
        let sink = graph.sink() as usize;
        let mut parent = vec![0; node_count];
        let mut forced_source = vec![false; node_count];

        for edge in &graph.edges {
            let from = edge.from as usize;
            let to = edge.to as usize;

            // Mark incident internal nodes. Until a continuation is found,
            // treat each as a root with a zero-cost virtual edge to the sink.
            if from != 0 && parent[from] == 0 {
                parent[from] = sink;
            }
            if to != sink && parent[to] == 0 {
                parent[to] = sink;
            }

            if !edge.cuttable() {
                forced_source[to] = true;
            } else if from != 0 && to != sink {
                parent[from] = to;
            }
        }

        let active = parent
            .iter()
            .enumerate()
            .filter_map(|(node, &next)| (next != 0).then_some(node))
            .collect();

        Self {
            edges: &graph.edges,
            active,
            parent,
            parent_weight: vec![0; node_count],
            delta: vec![0; node_count],
            forced_source,
            source_side: vec![false; node_count],
            cut: Vec::new(),
        }
    }

    /// Return the cheapest cut as ascending indices into the graph's edges.
    /// Among equal-cost cuts, choose the inclusion-minimal source side.
    ///
    /// Weights must be nonnegative, with their sum fitting in `i64`.
    /// The planner's existing bounds put that sum below 2^51.
    /// Only cuttable edges are passed to `weight`.
    pub(in crate::search::substring) fn solve(&mut self, weight: impl Fn(&Edge) -> u64) -> &[u32] {
        let sink = self.parent.len() - 1;

        // Source edges charge the sink side; terminal edges charge the source
        // side. Internal edge weights are used when combining subtrees.
        // Direct source-to-sink costs accumulate at the sink and do not affect
        // assignments: those edges must be cut regardless.
        self.delta.fill(0);
        for edge in self.edges.iter().filter(|edge| edge.cuttable()) {
            let from = edge.from as usize;
            let to = edge.to as usize;
            let cost = weight(edge) as i64;

            if from == 0 {
                self.delta[to] -= cost;
            } else if to == sink {
                self.delta[from] += cost;
            } else {
                self.parent_weight[from] = cost;
            }
        }

        // Children have smaller offsets than their parents. Moving a parent
        // to the source side can save at most the connecting edge's weight;
        // a child forced to the source side always saves that whole weight.
        for &node in &self.active {
            let edge_cost = self.parent_weight[node];
            let contribution = if self.forced_source[node] {
                -edge_cost
            } else {
                self.delta[node].min(0).max(-edge_cost)
            };
            self.delta[self.parent[node]] += contribution;
        }

        // Parents are assigned first. Choosing the source side also pays
        // for the outgoing edge when the parent belongs to the sink side.
        self.source_side[0] = true;
        self.source_side[sink] = false;
        for &node in self.active.iter().rev() {
            let threshold = if self.source_side[self.parent[node]] {
                0
            } else {
                -self.parent_weight[node]
            };
            self.source_side[node] = self.forced_source[node] || self.delta[node] < threshold;
        }

        // Direct source-to-sink edges are included automatically.
        self.cut.clear();
        self.cut
            .extend(self.edges.iter().enumerate().filter_map(|(index, edge)| {
                (edge.cuttable()
                    && self.source_side[edge.from as usize]
                    && !self.source_side[edge.to as usize])
                    .then_some(index as u32)
            }));
        &self.cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::substring::alignment::graph::tests::{synthetic_edge, synthetic_graph};

    /// Return cut endpoints in edge order for readable test expectations.
    fn cut_steps(graph: &AlignmentGraph) -> Vec<(u32, u32)> {
        MinCut::new(graph)
            .solve(|edge| u64::from(edge.frequency()))
            .iter()
            .map(|&at| (graph.edges[at as usize].from, graph.edges[at as usize].to))
            .collect()
    }

    /// Enumerate all partitions and intersect the optimal source sides.
    /// This checks both optimality and tie-breaking independently of the DP.
    fn exhaustive_cut(graph: &AlignmentGraph, weight: impl Fn(&Edge) -> u64) -> Vec<u32> {
        let mut best = u64::MAX;
        let mut common = u64::MAX;
        'partitions: for interior in 0..1u64 << (graph.node_count() - 2) {
            let side = 1 | (interior << 1);
            let mut cost = 0;
            for edge in &graph.edges {
                if side & (1 << edge.from) != 0 && side & (1 << edge.to) == 0 {
                    if !edge.cuttable() {
                        continue 'partitions;
                    }
                    cost += weight(edge);
                }
            }
            if cost < best {
                best = cost;
                common = side;
            } else if cost == best {
                common &= side;
            }
        }
        graph
            .edges
            .iter()
            .enumerate()
            .filter_map(|(at, edge)| {
                (common & (1 << edge.from) != 0 && common & (1 << edge.to) == 0)
                    .then_some(at as u32)
            })
            .collect()
    }

    /// Four-node alignment forests with absent, forbidden, zero and weighted
    /// edges. Reuse each solver with changing weights, including above u32::MAX.
    #[test]
    fn cuts_agree_with_exhaustive_partitions() {
        const ARCS: [(u32, u32); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        'graphs: for configuration in 0..5usize.pow(ARCS.len() as u32) {
            let mut choices = configuration;
            let mut edges = Vec::new();
            for (from, to) in ARCS {
                let choice = choices % 5;
                choices /= 5;
                if choice == 1 && (from != 0 || to == 3) {
                    continue 'graphs;
                }
                if choice != 0 {
                    let frequency = [None, Some(0), Some(1), Some(3)][choice - 1];
                    edges.push(synthetic_edge(from, to, frequency));
                }
            }
            let graph = synthetic_graph(4, edges);
            let mut solver = MinCut::new(&graph);
            for large in [false, true, false] {
                let weight = |edge: &Edge| {
                    if large {
                        (1u64 << 32) + u64::from(u32::MAX - edge.frequency())
                    } else {
                        u64::from(edge.frequency())
                    }
                };
                assert_eq!(
                    solver.solve(weight),
                    exhaustive_cut(&graph, weight),
                    "graph={configuration}, large={large}"
                );
            }
        }
    }

    /// Cut a shared suffix once for 6 instead of two separate branches for 4 + 4.
    #[test]
    fn shared_suffix_beats_two_local_choices() {
        let graph = synthetic_graph(
            5,
            vec![
                synthetic_edge(0, 1, None),
                synthetic_edge(0, 2, None),
                synthetic_edge(1, 3, Some(4)),
                synthetic_edge(2, 3, Some(4)),
                synthetic_edge(3, 4, Some(6)),
            ],
        );
        assert_eq!(cut_steps(&graph), vec![(3, 4)]);
    }

    /// Separate paths both need a cut, including a path with a zero-weight edge.
    #[test]
    fn disjoint_paths_are_cut_separately() {
        let graph = synthetic_graph(
            5,
            vec![
                synthetic_edge(0, 1, None),
                synthetic_edge(1, 4, Some(5)),
                synthetic_edge(0, 2, None),
                synthetic_edge(2, 3, Some(9)),
                synthetic_edge(3, 4, Some(0)),
            ],
        );
        assert_eq!(cut_steps(&graph), vec![(1, 4), (3, 4)]);
    }

    /// Cutting an internal continuation does not cover a parallel terminal path.
    /// Direct source-to-sink edges are mandatory; unused offsets are harmless.
    #[test]
    fn terminal_edges_are_charged_alongside_internal_edges() {
        let graph = synthetic_graph(
            5,
            vec![
                synthetic_edge(0, 1, None),
                synthetic_edge(1, 2, Some(2)),
                synthetic_edge(1, 4, Some(5)),
                synthetic_edge(2, 4, Some(3)),
                synthetic_edge(0, 4, Some(7)),
            ],
        );
        assert_eq!(cut_steps(&graph), vec![(1, 2), (1, 4), (0, 4)]);
    }

    /// A long path exercises the iterative solver without growing the call stack.
    #[test]
    fn deep_path_does_not_exhaust_the_stack() {
        const NODES: u32 = u16::MAX as u32 + 1;
        let mut edges: Vec<Edge> = (0..NODES - 1)
            .map(|v| synthetic_edge(v, v + 1, Some(7)))
            .collect();
        let cheapest = NODES / 2;
        edges[cheapest as usize] = synthetic_edge(cheapest, cheapest + 1, Some(3));
        let graph = synthetic_graph(NODES as usize, edges);
        assert_eq!(cut_steps(&graph), vec![(cheapest, cheapest + 1)]);
    }
}
