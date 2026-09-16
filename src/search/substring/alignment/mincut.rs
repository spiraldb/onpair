// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimum-weight cut of the alignment DAG: the cheapest sound cover.
//!
//! Every source-to-sink path in [`AlignmentGraph`](super::graph::AlignmentGraph)
//! is one layout of the pattern across token boundaries, and a set of probes
//! meeting every path is a cover the scan can trust. Probes are edges, so
//! picking the cheapest such set is a minimum cut of the edge weights, which is
//! max-flow directly. The weights are the caller's; the steps no probe stands
//! for get a capacity no finite cut can reach, which keeps the minimum cut on
//! the edges where it means something.
//!
//! The point of cutting the *merged* DAG rather than choosing per alignment is
//! that alignments converging on a shared suffix are paid for once, at the join,
//! instead of once each.
//!
//! # Cost
//! Dinic is `O(V^2 E)` in the abstract, but the DAG it runs on here is small and
//! narrow. Greedy parsing crosses the pattern a whole token at a time, so the
//! state path is `n / token_len` rather than `n` — the node array is indexed by
//! needle offset, so the offsets between are isolated — and every path leaves
//! the source through one of at most `MAX_TOKEN_SIZE` alignments, which caps how
//! many augmenting paths there are to find, independently of `n`. A 66-byte
//! pattern over a 12k-token dictionary cuts in ~3us, two orders of magnitude
//! under the dictionary pass in
//! [`AlignmentGraph::new`](super::graph::AlignmentGraph::new) that produced
//! it; even a synthetic worst case of one state per byte over 1024 bytes stays
//! near 150us. The solve is not what a plan waits on, and needs no length guard.

use std::collections::VecDeque;

use super::graph::Edge;

/// The residual graph, in CSR: one allocation per array rather than one per
/// node, since the whole arc set is known before the first push.
struct Dinic {
    /// Row starts, `num_nodes + 1` entries; node `v` owns `head[v]..head[v + 1]`.
    head: Vec<u32>,
    /// Endpoint of each arc.
    to: Vec<u32>,
    /// Each arc's twin, so pushing flow can credit the residual back.
    twin: Vec<u32>,
    /// Residual capacity of each arc.
    cap: Vec<u64>,
    /// BFS distance from the source, or `-1` for nodes outside the level graph.
    level: Vec<i32>,
    /// Current-arc cursor: the first arc of each node not yet ruled out this
    /// phase. Never rewinds, which is what bounds the blocking flow.
    next: Vec<u32>,
    /// Forward arc of each edge, in edge order, so a re-solve refills
    /// capacities without rebuilding the CSR.
    forward: Vec<u32>,
    /// The BFS frontier, held across solves rather than allocated per phase.
    queue: VecDeque<usize>,
    /// The advancing DFS path, as the arcs taken to reach the current node.
    path: Vec<u32>,
}

impl Dinic {
    /// Build the residual graph over `num_nodes` nodes from `arcs`, each
    /// `(from, to, capacity)`. Every arc gets a zero-capacity twin.
    fn new(num_nodes: usize, arcs: &[(u32, u32)]) -> Self {
        let mut head = vec![0u32; num_nodes + 1];
        for &(from, to) in arcs {
            head[from as usize + 1] += 1;
            head[to as usize + 1] += 1;
        }
        for v in 0..num_nodes {
            head[v + 1] += head[v];
        }

        let slots = arcs.len() * 2;
        let mut cursor = head.clone();
        let mut edge_to = vec![0u32; slots];
        let mut twin = vec![0u32; slots];
        let mut forward = vec![0u32; arcs.len()];
        for (at, &(from, to)) in arcs.iter().enumerate() {
            let fwd = cursor[from as usize];
            cursor[from as usize] += 1;
            let rev = cursor[to as usize];
            cursor[to as usize] += 1;
            edge_to[fwd as usize] = to;
            twin[fwd as usize] = rev;
            edge_to[rev as usize] = from;
            twin[rev as usize] = fwd;
            forward[at] = fwd;
        }

        Self {
            head,
            to: edge_to,
            twin,
            cap: vec![0u64; slots],
            level: vec![-1; num_nodes],
            next: vec![0; num_nodes],
            forward,
            queue: VecDeque::new(),
            path: Vec::new(),
        }
    }

    /// Reset the residual to `capacities`, one per edge in edge order. The
    /// topology is a property of the graph, not of the weights, so it survives.
    fn refill(&mut self, capacities: impl Iterator<Item = u64>) {
        self.cap.fill(0);
        for (at, capacity) in capacities.enumerate() {
            self.cap[self.forward[at] as usize] = capacity;
        }
    }

    /// Layer the residual graph by distance from `source`, and report whether
    /// `sink` is still reachable. Leaves `level` describing the source side of
    /// the residual graph, which is the minimum cut once it returns `false`.
    fn build_levels(&mut self, source: usize, sink: usize) -> bool {
        self.level.fill(-1);
        self.level[source] = 0;
        self.queue.clear();
        self.queue.push_back(source);
        while let Some(v) = self.queue.pop_front() {
            for arc in self.head[v] as usize..self.head[v + 1] as usize {
                let to = self.to[arc] as usize;
                if self.cap[arc] > 0 && self.level[to] < 0 {
                    self.level[to] = self.level[v] + 1;
                    self.queue.push_back(to);
                }
            }
        }
        self.level[sink] >= 0
    }

    /// Saturate the current level graph, one augmenting path at a time.
    ///
    /// Iterative rather than the textbook recursion, which would put the length
    /// of the longest source-to-sink path on the call stack. Nothing in this
    /// solver bounds that length, and it costs nothing to not depend on it.
    fn blocking_flow(&mut self, source: usize, sink: usize) {
        self.next.copy_from_slice(&self.head[..self.level.len()]);

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
                    let capacity = self.cap[arc as usize];
                    if capacity < bottleneck {
                        bottleneck = capacity;
                        saturated = at;
                    }
                }
                for &arc in &path {
                    self.cap[arc as usize] -= bottleneck;
                    self.cap[self.twin[arc as usize] as usize] += bottleneck;
                }

                // Retreat only as far as the first arc the augment saturated:
                // the prefix before it still admits flow, so re-walking it from
                // the source would be wasted work.
                node = self.to[self.twin[path[saturated] as usize] as usize] as usize;
                path.truncate(saturated);
                continue;
            }

            // Advance along the current arc, skipping whatever the level graph
            // or an earlier augment has already ruled out.
            let end = self.head[node + 1];
            while self.next[node] < end {
                let arc = self.next[node] as usize;
                if self.cap[arc] > 0 && self.level[self.to[arc] as usize] == self.level[node] + 1 {
                    break;
                }
                self.next[node] += 1;
            }

            if self.next[node] < end {
                let arc = self.next[node];
                path.push(arc);
                node = self.to[arc as usize] as usize;
            } else if let Some(arc) = path.pop() {
                // A dead end in the level graph stays one for the rest of the
                // phase, so drop the node out of it rather than revisiting.
                self.level[node] = -1;
                node = self.to[self.twin[arc as usize] as usize] as usize;
            } else {
                self.path = path;
                return;
            }
        }
    }

    fn max_flow(&mut self, source: usize, sink: usize) {
        while self.build_levels(source, sink) {
            self.blocking_flow(source, sink);
        }
    }
}

/// A minimum-cut solver bound to one graph, re-solvable at new weights.
///
/// The residual topology is a property of the graph and the weights are not, so
/// a sweep that re-prices the same DAG rebuilds nothing: capacities are refilled
/// in place and the cut lands in a buffer the solver owns. That matters because
/// the graph is tiny — a handful of live nodes — so building it cost about what
/// solving it does.
pub(in crate::search::substring) struct MinCut {
    flow: Dinic,
    sink: usize,
    /// The cut of the last solve, as indices into the caller's edge slice.
    cut: Vec<u32>,
}

impl MinCut {
    /// Nodes are numbered `0..node_count`, with source 0 and sink `node_count - 1`.
    /// Requires at least two nodes, valid endpoints, and residual arc IDs that
    /// fit in `u32`. Alignment graph construction establishes these invariants.
    pub(in crate::search::substring) fn new(edges: &[Edge], node_count: usize) -> Self {
        let arcs: Vec<(u32, u32)> = edges.iter().map(|edge| (edge.from, edge.to)).collect();
        Self {
            flow: Dinic::new(node_count, &arcs),
            sink: node_count - 1,
            cut: Vec::new(),
        }
    }

    /// The cheapest set of cuttable edges whose removal disconnects the source
    /// from the sink, as ascending indices into `edges`. `weight` prices cutting
    /// one edge and is consulted for cuttable edges only.
    ///
    /// `edges` must be the graph used at construction. Every source-to-sink path
    /// must include a cuttable edge, and the sum of finite weights plus one must
    /// fit in `u64`. The alignment builder and planner guarantee these bounds.
    /// `weight` must return the same value for an edge throughout this solve.
    pub(in crate::search::substring) fn solve(
        &mut self,
        edges: &[Edge],
        weight: impl Fn(&Edge) -> u64,
    ) -> &[u32] {
        // One more than every finite cut, so a minimum cut never prefers an
        // uncuttable step over the edges that stand for real probes.
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

        // `max_flow` stops on the level pass that failed to reach the sink, and
        // that pass is exactly a BFS of the residual graph from the source — so
        // `level` already marks the source side, and no second traversal is
        // needed. An edge is cut when it straddles the two sides, which also
        // means it is saturated: an unsaturated edge would have carried the BFS
        // across.
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

/// One cut of `edges` at `weight`, building a solver to do it. The planner
/// prices the same graph many times and uses [`MinCut::solve`] directly; this
/// is for callers that cut once.
#[cfg(test)]
pub(in crate::search::substring) fn min_cut(
    edges: &[Edge],
    node_count: usize,
    weight: impl Fn(&Edge) -> u64,
) -> Vec<&Edge> {
    let mut solver = MinCut::new(edges, node_count);
    solver
        .solve(edges, weight)
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

    /// Compare endpoints without depending on edge construction order.
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

    /// Every four-node DAG, including zero weights, ties and uncuttable edges.
    /// Reuse the solver with reversed weights above u32::MAX as well.
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
                let cut = solver.solve(&edges, weight);
                // Membership proves both disconnection and finite probe coverage.
                assert!(
                    cuts.iter().any(|c| c == cut),
                    "graph={configuration}, large={large}"
                );
                assert_eq!(cost(cut), cuts.iter().map(|c| cost(c)).min().unwrap());
            }
        }
    }

    /// The property the merged DAG exists for: two alignments converging on one
    /// shared suffix are cut once at the join for 6, not once each for 4 + 4.
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

    /// Two disjoint paths have to be cut on both, and a zero-weight probe is
    /// always worth taking.
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

    /// Depth is a property of the graph, not of any pattern length this solver
    /// gets to assume, so the DFS has to stay off the call stack. A recursive
    /// `send` overflows well before this path does.
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
