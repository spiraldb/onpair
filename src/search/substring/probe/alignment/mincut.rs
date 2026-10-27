// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimum-weight cut of an alignment graph, in linear time.
//!
//! A cut puts every node on the source side or the sink side and selects the edges
//! that cross from the first to the second. Each internal node has at most one
//! internal successor, so without the source and sink the nodes form a forest whose
//! roots lead to the sink. That makes the minimum cut a dynamic program over trees.
//! Alignment and range edges become costs for placing their internal endpoint on
//! one side.
//!
//! A forward pass, from children to parents in offset order, accumulates subtree
//! costs. A backward pass assigns sides from the roots down, preferring the sink
//! side on ties. Unenumerated alignment edges, whose tokens cannot be in a filter,
//! force their target onto the source side, so they are never cut.

use super::graph::{AlignmentGraph, Edge};

/// A cut solver for one graph, reusable with different edge weights.
pub(in crate::search::substring::probe) struct MinCut<'g> {
    edges: &'g [Edge],
    /// Internal nodes incident to edges, in increasing needle-offset order.
    active: Vec<u16>,
    /// Next internal node, or the sink for a forest root. Zero means unused.
    parent: Vec<u16>,
    /// Weight of the internal edge to the parent; zero for forest roots.
    parent_weight: Vec<i64>,
    /// Cost of the source side minus cost of the sink side, for the subtree.
    delta: Vec<i64>,
    /// Whether an unenumerated alignment edge enters this node, forcing it onto the
    /// source side.
    forced_source: Vec<bool>,
    /// Side assignments from the latest solve.
    source_side: Vec<bool>,
    /// Cut edge indices from the latest solve, in edge order.
    cut: Vec<u32>,
}

impl<'g> MinCut<'g> {
    /// Records the forest of `graph` and allocates the solve buffers.
    ///
    /// Relies on the alignment graph's shape: edges go forward, each internal node
    /// has at most one internal successor, and unenumerated edges leave the source.
    pub(in crate::search::substring::probe) fn new(graph: &'g AlignmentGraph) -> Self {
        let node_count = graph.node_count();
        let sink = graph.sink();
        let mut parent = vec![0; node_count];
        let mut forced_source = vec![false; node_count];

        for edge in &graph.edges {
            let (from, to) = (edge.from, edge.to);

            // Mark incident internal nodes. Until a continuation is found,
            // treat each as a root with a zero-cost virtual edge to the sink.
            if from != 0 && parent[usize::from(from)] == 0 {
                parent[usize::from(from)] = sink;
            }
            if to != sink && parent[usize::from(to)] == 0 {
                parent[usize::from(to)] = sink;
            }

            if !edge.cuttable() {
                forced_source[usize::from(to)] = true;
            } else if from != 0 && to != sink {
                parent[usize::from(from)] = to;
            }
        }

        let active = (0..=sink)
            .filter(|&node| parent[usize::from(node)] != 0)
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

    /// Returns the cheapest cut as ascending indices into the graph's edges. Among
    /// equal-cost cuts, it picks the one with the inclusion-minimal source side.
    ///
    /// `weight` is only called for cuttable edges. Weights must be nonnegative and
    /// their sum must fit in `i64`, which the planner's weights do.
    pub(in crate::search::substring::probe) fn solve(
        &mut self,
        weight: impl Fn(&Edge) -> u64,
    ) -> &[u32] {
        let sink = self.parent.len() - 1;

        // Alignment edges charge the sink side; range edges charge the source
        // side. Greedy edge weights are used when combining subtrees.
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
            let node = usize::from(node);
            let edge_cost = self.parent_weight[node];
            let contribution = if self.forced_source[node] {
                -edge_cost
            } else {
                self.delta[node].min(0).max(-edge_cost)
            };
            self.delta[usize::from(self.parent[node])] += contribution;
        }

        // Parents are assigned first. Choosing the source side also pays
        // for the outgoing edge when the parent belongs to the sink side.
        self.source_side[0] = true;
        self.source_side[sink] = false;
        for &node in self.active.iter().rev() {
            let node = usize::from(node);
            let threshold = if self.source_side[usize::from(self.parent[node])] {
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
    use crate::search::substring::probe::alignment::graph::tests::{
        synthetic_edge, synthetic_graph,
    };

    /// Return cut endpoints in edge order for readable test expectations.
    fn cut_steps(graph: &AlignmentGraph) -> Vec<(u16, u16)> {
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
        const ARCS: [(u16, u16); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
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

    /// Cutting the greedy edge out of a node leaves its range edge to cut as well.
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
        // The longest pattern: its sink is the largest `u16` offset.
        const NODES: usize = u16::MAX as usize + 1;
        let mut edges: Vec<Edge> = (0..u16::MAX)
            .map(|v| synthetic_edge(v, v + 1, Some(7)))
            .collect();
        let cheapest = u16::MAX / 2;
        edges[usize::from(cheapest)] = synthetic_edge(cheapest, cheapest + 1, Some(3));
        let graph = synthetic_graph(NODES, edges);
        assert_eq!(cut_steps(&graph), vec![(cheapest, cheapest + 1)]);
    }
}
