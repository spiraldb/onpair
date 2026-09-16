// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimum-weight edge cuts of an alignment graph.
//!
//! A cut intersects every source-to-sink path, so its tokens form a sound
//! probe cover. The planner supplies edge weights and compares the resulting
//! covers using the scan cost model.
//!
//! Dinic's algorithm finds the cut through maximum flow. Each graph edge has
//! a forward arc and a reverse arc that can undo earlier flow. Unenumerated
//! edges receive a capacity larger than any finite cut, so they stay uncut.
//! The solver reuses its topology and buffers when the planner changes weights.

use std::collections::VecDeque;

use super::graph::Edge;

/// Residual graph and scratch buffers for Dinic's maximum-flow algorithm.
///
/// Outgoing arcs occupy contiguous slices of shared arrays. `arc_offsets` locates
/// each node's slice; `reverse` links each arc to its reverse.
struct Dinic {
    /// Row starts, `num_nodes + 1` entries; node `v` owns `arc_offsets[v]..arc_offsets[v + 1]`.
    arc_offsets: Vec<u32>,
    /// Endpoint of each arc.
    to: Vec<u32>,
    /// Index of each reverse arc, whose capacity increases when flow is sent.
    reverse: Vec<u32>,
    /// Residual capacity of each arc.
    capacity: Vec<u64>,
    /// BFS distance from the source, or `-1` for nodes outside the level graph.
    level: Vec<i32>,
    /// First outgoing arc still worth trying in the current level graph.
    next_arc: Vec<u32>,
    /// Forward arc for each original edge, used to reload its weight.
    forward: Vec<u32>,
    /// Reusable queue for breadth-first traversal.
    queue: VecDeque<usize>,
    /// Arcs on the current source-to-node path, replacing recursive DFS.
    path: Vec<u32>,
}

impl Dinic {
    /// Allocate the residual topology from `(from, to)` edges.
    /// Each edge gets a reverse arc. `refill` supplies capacities before solving.
    fn new(num_nodes: usize, arcs: impl ExactSizeIterator<Item = (u32, u32)> + Clone) -> Self {
        let mut arc_offsets = vec![0u32; num_nodes + 1];
        for (from, to) in arcs.clone() {
            arc_offsets[from as usize + 1] += 1;
            arc_offsets[to as usize + 1] += 1;
        }
        for v in 0..num_nodes {
            arc_offsets[v + 1] += arc_offsets[v];
        }

        let slots = arcs.len() * 2;
        let mut cursor = arc_offsets.clone();
        let mut edge_to = vec![0u32; slots];
        let mut reverse = vec![0u32; slots];
        let mut forward = vec![0u32; arcs.len()];
        for ((from, to), forward_arc) in arcs.zip(&mut forward) {
            let fwd = cursor[from as usize];
            cursor[from as usize] += 1;
            let rev = cursor[to as usize];
            cursor[to as usize] += 1;
            edge_to[fwd as usize] = to;
            reverse[fwd as usize] = rev;
            edge_to[rev as usize] = from;
            reverse[rev as usize] = fwd;
            *forward_arc = fwd;
        }

        Self {
            arc_offsets,
            to: edge_to,
            reverse,
            capacity: vec![0u64; slots],
            level: vec![-1; num_nodes],
            next_arc: vec![0; num_nodes],
            forward,
            queue: VecDeque::new(),
            path: Vec::new(),
        }
    }

    /// Reset forward capacities from edge weights and clear all reverse capacities.
    fn refill(&mut self, capacities: impl Iterator<Item = u64>) {
        self.capacity.fill(0);
        for (at, capacity) in capacities.enumerate() {
            self.capacity[self.forward[at] as usize] = capacity;
        }
    }

    /// Find each node's distance from the source through positive-capacity arcs.
    /// Returns whether the sink is reachable. On the final failed search,
    /// nonnegative levels identify the source side of the minimum cut.
    fn build_levels(&mut self, source: usize, sink: usize) -> bool {
        self.level.fill(-1);
        self.level[source] = 0;
        self.queue.clear();
        self.queue.push_back(source);
        while let Some(v) = self.queue.pop_front() {
            for arc in self.arc_offsets[v] as usize..self.arc_offsets[v + 1] as usize {
                let to = self.to[arc] as usize;
                if self.capacity[arc] > 0 && self.level[to] < 0 {
                    self.level[to] = self.level[v] + 1;
                    self.queue.push_back(to);
                }
            }
        }
        self.level[sink] >= 0
    }

    /// Send flow until no source-to-sink path remains in the current level graph.
    /// An explicit path buffer keeps call-stack usage independent of graph depth.
    fn blocking_flow(&mut self, source: usize, sink: usize) {
        self.next_arc
            .copy_from_slice(&self.arc_offsets[..self.level.len()]);

        let mut path = std::mem::take(&mut self.path);
        path.clear();
        let mut node = source;
        loop {
            if node == sink {
                // Source and sink differ, so the path is non-empty. Record the
                // first minimum: that arc will be saturated by this augment.
                let mut bottleneck = u64::MAX;
                let mut saturated = 0;
                for (at, &arc) in path.iter().enumerate() {
                    let capacity = self.capacity[arc as usize];
                    if capacity < bottleneck {
                        bottleneck = capacity;
                        saturated = at;
                    }
                }
                for &arc in &path {
                    self.capacity[arc as usize] -= bottleneck;
                    self.capacity[self.reverse[arc as usize] as usize] += bottleneck;
                }

                // Resume before the first saturated arc. The earlier path
                // still has capacity and can be reused.
                node = self.to[self.reverse[path[saturated] as usize] as usize] as usize;
                path.truncate(saturated);
                continue;
            }

            // Follow arcs with capacity that advance exactly one BFS level.
            let end = self.arc_offsets[node + 1];
            while self.next_arc[node] < end {
                let arc = self.next_arc[node] as usize;
                if self.capacity[arc] > 0
                    && self.level[self.to[arc] as usize] == self.level[node] + 1
                {
                    break;
                }
                self.next_arc[node] += 1;
            }

            if self.next_arc[node] < end {
                let arc = self.next_arc[node];
                path.push(arc);
                node = self.to[arc as usize] as usize;
            } else if let Some(arc) = path.pop() {
                // This node cannot reach the sink in the current level graph.
                self.level[node] = -1;
                node = self.to[self.reverse[arc as usize] as usize] as usize;
            } else {
                self.path = path;
                return;
            }
        }
    }

    /// Repeat level searches and blocking flows until the sink is unreachable.
    /// The final levels are retained for cut extraction.
    fn max_flow(&mut self, source: usize, sink: usize) {
        while self.build_levels(source, sink) {
            self.blocking_flow(source, sink);
        }
    }
}

/// A cut solver reusable with different weights on the same graph.
/// The edge slice is borrowed for the lifetime of the solver.
/// Each solve resets capacities and reuses the residual topology and buffers.
pub(in crate::search::substring) struct MinCut<'g> {
    flow: Dinic,
    edges: &'g [Edge],
    sink: usize,
    /// The cut of the last solve, as indices into the borrowed edge slice.
    cut: Vec<u32>,
}

impl<'g> MinCut<'g> {
    /// Nodes are numbered `0..node_count`, with source 0 and sink `node_count - 1`.
    /// Requires at least two nodes, valid endpoints, and residual arc IDs that
    /// fit in `u32`. Alignment graph construction establishes these invariants.
    pub(in crate::search::substring) fn new(edges: &'g [Edge], node_count: usize) -> Self {
        Self {
            flow: Dinic::new(node_count, edges.iter().map(|edge| (edge.from, edge.to))),
            edges,
            sink: node_count - 1,
            cut: Vec::new(),
        }
    }

    /// The cheapest set of cuttable edges whose removal disconnects the source
    /// from the sink, as ascending indices into the edges supplied at construction.
    /// `weight` prices cutting one edge and is consulted for cuttable edges only.
    ///
    /// Every source-to-sink path must include a cuttable edge, and the sum of
    /// finite weights plus one must fit in `u64`. The alignment builder and
    /// planner guarantee these bounds.
    /// `weight` must return the same value for an edge throughout this solve.
    pub(in crate::search::substring) fn solve(&mut self, weight: impl Fn(&Edge) -> u64) -> &[u32] {
        let edges = self.edges;
        // More expensive than all cuttable edges together, so no minimum
        // cut needs an edge whose token IDs are unenumerated.
        let finite_sum: u64 = edges
            .iter()
            .filter(|edge| edge.cuttable())
            .map(&weight)
            .sum();
        let infinite = finite_sum + 1;
        self.flow.refill(edges.iter().map(|edge| {
            if edge.cuttable() {
                weight(edge)
            } else {
                infinite
            }
        }));

        self.flow.max_flow(0, self.sink);

        // The final BFS already marks nodes reachable from the source.
        // Edges leaving that set form the cut; no extra traversal is needed.
        self.cut.clear();
        self.cut
            .extend(edges.iter().enumerate().filter_map(|(at, edge)| {
                (edge.cuttable()
                    && self.flow.level[edge.from as usize] >= 0
                    && self.flow.level[edge.to as usize] < 0)
                    .then_some(at as u32)
            }));
        &self.cut
    }
}

/// Build a solver for one test cut and return references to the selected edges.
#[cfg(test)]
pub(in crate::search::substring) fn min_cut(
    edges: &[Edge],
    node_count: usize,
    weight: impl Fn(&Edge) -> u64,
) -> Vec<&Edge> {
    let mut solver = MinCut::new(edges, node_count);
    solver
        .solve(weight)
        .iter()
        .map(|&at| &edges[at as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::substring::alignment::graph::tests::synthetic_edge;

    fn by_frequency(edge: &Edge) -> u64 {
        u64::from(edge.frequency())
    }

    /// Return cut endpoints in edge order for readable test expectations.
    fn steps(cut: &[&Edge]) -> Vec<(u32, u32)> {
        cut.iter().map(|edge| (edge.from, edge.to)).collect()
    }

    /// All finite cuts of a four-node graph, with source 0 and sink 3.
    fn partition_cuts(edges: &[Edge]) -> Vec<Vec<u32>> {
        (1..8u32)
            .step_by(2)
            .map(|side| {
                edges
                    .iter()
                    .enumerate()
                    .filter_map(|(at, edge)| {
                        (side & (1 << edge.from) != 0 && side & (1 << edge.to) == 0)
                            .then_some(at as u32)
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|cut| cut.iter().all(|&at| edges[at as usize].cuttable()))
            .collect()
    }

    /// Check all four-node DAGs over absent, uncuttable, and three weighted edges.
    /// Reuse each solver with weights above `u32::MAX` to check capacity handling.
    #[test]
    fn cuts_agree_with_exhaustive_partitions() {
        const ARCS: [(u32, u32); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        for configuration in 0..5usize.pow(ARCS.len() as u32) {
            let mut choices = configuration;
            let mut edges = Vec::new();
            for (from, to) in ARCS {
                let choice = choices % 5;
                choices /= 5;
                if choice != 0 {
                    let frequency = [None, Some(0), Some(1), Some(3)][choice - 1];
                    edges.push(synthetic_edge(from, to, frequency));
                }
            }
            let cuts = partition_cuts(&edges);
            // An all-uncuttable path violates the alignment builder's contract.
            if cuts.is_empty() {
                continue;
            }
            let mut solver = MinCut::new(&edges, 4);
            for large in [false, true] {
                let weight = |edge: &Edge| {
                    if large {
                        (1u64 << 32) + u64::from(u32::MAX - edge.frequency())
                    } else {
                        u64::from(edge.frequency())
                    }
                };
                let cost = |cut: &[u32]| -> u64 {
                    cut.iter().map(|&at| weight(&edges[at as usize])).sum()
                };
                let cut = solver.solve(weight);
                // Membership proves both disconnection and finite probe coverage.
                assert!(
                    cuts.iter().any(|c| c == cut),
                    "graph={configuration}, large={large}"
                );
                assert_eq!(cost(cut), cuts.iter().map(|c| cost(c)).min().unwrap());
            }
        }
    }

    /// Cut a shared suffix once for 6 instead of two separate branches for 4 + 4.
    #[test]
    fn shared_suffix_beats_two_local_choices() {
        // 0 -> 1 -(4)-> 3 -(6)-> 4  and  0 -> 2 -(4)-> 3 -(6)-> 4
        let edges = [
            synthetic_edge(0, 1, None),
            synthetic_edge(0, 2, None),
            synthetic_edge(1, 3, Some(4)),
            synthetic_edge(2, 3, Some(4)),
            synthetic_edge(3, 4, Some(6)),
        ];
        assert_eq!(steps(&min_cut(&edges, 5, by_frequency)), vec![(3, 4)]);
    }

    /// Separate paths both need a cut, including a path with a zero-weight edge.
    #[test]
    fn disjoint_paths_are_cut_separately() {
        // 0 -> 1 -(5)-> 4  and  0 -> 2 -(9)-> 3 -(0)-> 4
        let edges = [
            synthetic_edge(0, 1, None),
            synthetic_edge(1, 4, Some(5)),
            synthetic_edge(0, 2, None),
            synthetic_edge(2, 3, Some(9)),
            synthetic_edge(3, 4, Some(0)),
        ];
        assert_eq!(
            steps(&min_cut(&edges, 5, by_frequency)),
            vec![(1, 4), (3, 4)]
        );
    }

    /// A long path exercises the iterative solver without growing the call stack.
    #[test]
    fn deep_path_does_not_exhaust_the_stack() {
        const LEN: u32 = 100_000;
        let mut edges: Vec<Edge> = (0..LEN - 1)
            .map(|v| synthetic_edge(v, v + 1, Some(7)))
            .collect();
        let cheapest = LEN / 2;
        edges[cheapest as usize] = synthetic_edge(cheapest, cheapest + 1, Some(3));

        assert_eq!(
            steps(&min_cut(&edges, LEN as usize, by_frequency)),
            vec![(cheapest, cheapest + 1)]
        );
    }
}
