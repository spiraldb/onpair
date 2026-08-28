// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimum-weight cut of the alignment DAG: the cheapest sound cover.
//!
//! Every source-to-sink path in [`AlignmentGraph`](super::graph::AlignmentGraph)
//! is one layout of the pattern across token boundaries, and a set of probes
//! meeting every path is a cover the scan can trust. Probes are edges, so
//! picking the cheapest such set is a minimum cut of the edge weights, which is
//! max-flow directly. The steps no
//! probe stands for get a capacity no finite cut can reach, which keeps the
//! minimum cut on the edges where it means something.
//!
//! The point of cutting the *merged* DAG rather than choosing per alignment is
//! that alignments converging on a shared suffix are paid for once, at the join,
//! instead of once each.
//!
//! # Cost
//! Dinic is `O(V^2 E)` in the abstract, but the DAG it runs on here is small and
//! narrow. Greedy parsing crosses the pattern a whole token at a time, so the
//! state chain is `n / token_len` rather than `n` — the node array is indexed by
//! needle offset, so the offsets between are isolated — and every path leaves
//! the source through one of at most `MAX_TOKEN_SIZE` alignments, which caps how
//! many augmenting paths there are to find, independently of `n`. A 66-byte
//! pattern over a 12k-token dictionary cuts in ~3us, two orders of magnitude
//! under the dictionary pass in
//! [`build_alignment_graph`](super::graph::build_alignment_graph) that produced
//! it; even a synthetic worst case of one state per byte over 1024 bytes stays
//! near 150us. The solve is not what a plan waits on, and needs no length guard.

use std::collections::VecDeque;

use super::graph::{Edge, Nodes};

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
}

impl Dinic {
    /// Build the residual graph over `num_nodes` nodes from `arcs`, each
    /// `(from, to, capacity)`. Every arc gets a zero-capacity twin.
    fn new(num_nodes: usize, arcs: &[(u32, u32, u64)]) -> Self {
        let mut head = vec![0u32; num_nodes + 1];
        for &(from, to, _) in arcs {
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
        let mut cap = vec![0u64; slots];
        for &(from, to, capacity) in arcs {
            let fwd = cursor[from as usize];
            cursor[from as usize] += 1;
            let rev = cursor[to as usize];
            cursor[to as usize] += 1;
            edge_to[fwd as usize] = to;
            twin[fwd as usize] = rev;
            cap[fwd as usize] = capacity;
            edge_to[rev as usize] = from;
            twin[rev as usize] = fwd;
        }

        Self {
            head,
            to: edge_to,
            twin,
            cap,
            level: vec![-1; num_nodes],
            next: vec![0; num_nodes],
        }
    }

    /// Layer the residual graph by distance from `source`, and report whether
    /// `sink` is still reachable. Leaves `level` describing the source side of
    /// the residual graph, which is the minimum cut once it returns `false`.
    fn build_levels(&mut self, source: usize, sink: usize) -> bool {
        self.level.fill(-1);
        self.level[source] = 0;
        let mut queue = VecDeque::from([source]);
        while let Some(v) = queue.pop_front() {
            for arc in self.head[v] as usize..self.head[v + 1] as usize {
                let to = self.to[arc] as usize;
                if self.cap[arc] > 0 && self.level[to] < 0 {
                    self.level[to] = self.level[v] + 1;
                    queue.push_back(to);
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
    fn blocking_flow(&mut self, source: usize, sink: usize) -> u64 {
        debug_assert_ne!(source, sink, "the source and sink are distinct nodes");
        self.next.copy_from_slice(&self.head[..self.level.len()]);

        let mut total = 0u64;
        // The advancing DFS path, as the arcs taken to reach `node`.
        let mut path: Vec<u32> = Vec::new();
        let mut node = source;
        loop {
            if node == sink {
                let bottleneck = path
                    .iter()
                    .map(|&arc| self.cap[arc as usize])
                    .min()
                    .expect("source and sink differ, so a path to the sink has arcs");
                for &arc in &path {
                    self.cap[arc as usize] -= bottleneck;
                    self.cap[self.twin[arc as usize] as usize] += bottleneck;
                }
                total += bottleneck;

                // Retreat only as far as the first arc the augment saturated:
                // the prefix before it still admits flow, so re-walking it from
                // the source would be wasted work.
                let saturated = path
                    .iter()
                    .position(|&arc| self.cap[arc as usize] == 0)
                    .expect("the bottleneck arc is saturated by its own definition");
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
                return total;
            }
        }
    }

    fn max_flow(&mut self, source: usize, sink: usize) -> u64 {
        let mut total = 0u64;
        while self.build_levels(source, sink) {
            let sent = self.blocking_flow(source, sink);
            total = total.checked_add(sent).expect("max-flow capacity overflow");
        }
        total
    }
}

/// The cheapest set of cuttable edges whose removal disconnects `source` from
/// `sink`, in ascending edge order.
///
/// `edges[i]` is `(from, to, cost)`, where `cost` is what cutting that edge costs
/// or `None` for an edge no cut may select.
///
/// # Panics
/// Panics if some source-to-sink path runs entirely through uncuttable edges,
/// which no cut can block. Returning a set that fails to disconnect them would
/// hand back an unsound cover instead.
pub(super) fn min_cut(edges: &[Edge], nodes: Nodes) -> Vec<&Edge> {
    debug_assert!(
        edges.len() * 2 <= u32::MAX as usize,
        "the residual graph outgrew u32 arc ids"
    );

    // One more than every finite cut, so a minimum cut never prefers an
    // uncuttable step over the edges that stand for real probes.
    let finite_sum = edges
        .iter()
        .filter_map(Edge::cost)
        .try_fold(0u64, |acc, cost| acc.checked_add(u64::from(cost)))
        .expect("sum of probe weights overflowed u64");
    let infinite = finite_sum + 1;

    let residual: Vec<(u32, u32, u64)> = edges
        .iter()
        .map(|edge| (edge.from, edge.to, edge.cost().map_or(infinite, u64::from)))
        .collect();

    let mut flow = Dinic::new(nodes.count(), &residual);
    let value = flow.max_flow(nodes.source() as usize, nodes.sink() as usize);
    assert!(
        value < infinite,
        "the alignment DAG has a source-to-sink path with no probe on it"
    );

    // `max_flow` stops on the level pass that failed to reach the sink, and that
    // pass is exactly a BFS of the residual graph from the source — so `level`
    // already marks the source side, and no second traversal is needed. An edge
    // is cut when it straddles the two sides, which also means it is saturated:
    // an unsaturated edge would have carried the BFS across.
    edges
        .iter()
        .filter(|edge| {
            edge.cost().is_some()
                && flow.level[edge.from as usize] >= 0
                && flow.level[edge.to as usize] < 0
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The endpoints of a cut, which is what a caller reads off it. Comparing
    /// those rather than positions says which steps were chosen without
    /// depending on the order they were built in.
    fn steps(cut: &[&Edge]) -> Vec<(u32, u32)> {
        cut.iter().map(|edge| (edge.from, edge.to)).collect()
    }

    /// The property the merged DAG exists for: two alignments converging on one
    /// shared suffix are cut once at the join for 6, not once each for 4 + 4.
    #[test]
    fn shared_suffix_beats_two_local_choices() {
        // 0 -> 1 -(4)-> 3 -(6)-> 4  and  0 -> 2 -(4)-> 3 -(6)-> 4
        let edges = [
            Edge::synthetic(0, 1, None),
            Edge::synthetic(0, 2, None),
            Edge::synthetic(1, 3, Some(4)),
            Edge::synthetic(2, 3, Some(4)),
            Edge::synthetic(3, 4, Some(6)),
        ];
        assert_eq!(steps(&min_cut(&edges, Nodes::new(4))), vec![(3, 4)]);
    }

    /// Two disjoint paths have to be cut on both, and a zero-weight probe is
    /// always worth taking.
    #[test]
    fn disjoint_paths_are_cut_separately() {
        // 0 -> 1 -(5)-> 4  and  0 -> 2 -(9)-> 3 -(0)-> 4
        let edges = [
            Edge::synthetic(0, 1, None),
            Edge::synthetic(1, 4, Some(5)),
            Edge::synthetic(0, 2, None),
            Edge::synthetic(2, 3, Some(9)),
            Edge::synthetic(3, 4, Some(0)),
        ];
        assert_eq!(steps(&min_cut(&edges, Nodes::new(4))), vec![(1, 4), (3, 4)]);
    }

    /// Depth is a property of the graph, not of any pattern length this solver
    /// gets to assume, so the DFS has to stay off the call stack. A recursive
    /// `send` overflows well before this chain does.
    #[test]
    fn deep_chain_does_not_exhaust_the_stack() {
        const LEN: u32 = 100_000;
        let mut edges: Vec<Edge> = (0..LEN - 1)
            .map(|v| Edge::synthetic(v, v + 1, Some(7)))
            .collect();
        let cheapest = LEN / 2;
        edges[cheapest as usize] = Edge::synthetic(cheapest, cheapest + 1, Some(3));

        assert_eq!(
            steps(&min_cut(&edges, Nodes::new(LEN as usize - 1))),
            vec![(cheapest, cheapest + 1)]
        );
    }
}
