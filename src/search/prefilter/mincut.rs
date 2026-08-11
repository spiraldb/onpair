// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Minimum-weight vertex cut of the alignment DAG: the cheapest sound cover.
//!
//! Every source-to-sink path in [`AlignmentGraph`] is one layout of the pattern
//! across token boundaries, and a set of probes meeting every path is a cover
//! the scan can trust. Picking the cheapest such set is a minimum-weight *vertex*
//! cut, which max-flow solves once every node is split into an `in -> out` edge
//! carrying that node's weight: cutting the edge is cutting the node, and giving
//! the DAG's own arcs a capacity no finite cut can reach keeps the minimum cut on
//! split edges where it means something.
//!
//! The point of cutting the *merged* DAG rather than choosing per alignment is
//! that alignments converging on a shared suffix are paid for once, at the join,
//! instead of once each.
//!
//! # Cost
//! Dinic is `O(V^2 E)` in the abstract, but the DAG it runs on here is small and
//! narrow. Greedy parsing crosses the pattern a whole token at a time, so the
//! state chain is `n / token_len` rather than `n`, and every path leaves the
//! source through one of at most `MAX_TOKEN_SIZE` alignments — which caps how
//! many augmenting paths there are to find, independently of `n`. A 66-byte
//! pattern over a 12k-token dictionary builds a 33-node DAG and cuts it in ~3us,
//! two orders of magnitude under the dictionary pass in
//! [`build_alignment_graph`](super::graph::build_alignment_graph) that produced
//! it; even a synthetic worst case of one state per byte over 1024 bytes stays
//! near 150us. The solve is not what a plan waits on, and needs no length guard.

use std::collections::VecDeque;

use super::graph::AlignmentGraph;

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
        debug_assert_ne!(source, sink, "the split graph separates source from sink");
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

/// The cheapest set of cuttable nodes whose removal disconnects `source` from
/// `sink`, in ascending node order.
///
/// `capacity[v]` is what cutting `v` costs, or `None` for a node no cut may
/// select. `edges` are the arcs as `(from, to)`.
///
/// # Panics
/// Panics if some source-to-sink path runs entirely through uncuttable nodes,
/// which no cut can block. Returning a set that fails to disconnect them would
/// hand back an unsound cover instead.
fn min_cut(capacity: &[Option<u32>], edges: &[(u32, u32)], source: u32, sink: u32) -> Vec<u32> {
    let num_nodes = capacity.len();
    debug_assert!(
        num_nodes * 2 <= u32::MAX as usize,
        "the split graph outgrew u32 node ids"
    );
    let entry = |node: u32| node * 2;
    let exit = |node: u32| node * 2 + 1;

    // One more than every finite cut, so a minimum cut never prefers an arc or
    // an uncuttable node over the split edges that stand for real probes.
    let finite_sum = capacity
        .iter()
        .flatten()
        .try_fold(0u64, |acc, &weight| acc.checked_add(u64::from(weight)))
        .expect("sum of probe weights overflowed u64");
    let infinite = finite_sum + 1;

    let mut arcs = Vec::with_capacity(num_nodes + edges.len());
    for (node, &weight) in capacity.iter().enumerate() {
        let node = node as u32;
        arcs.push((entry(node), exit(node), weight.map_or(infinite, u64::from)));
    }
    for &(from, to) in edges {
        arcs.push((exit(from), entry(to), infinite));
    }

    let mut flow = Dinic::new(num_nodes * 2, &arcs);
    let value = flow.max_flow(exit(source) as usize, entry(sink) as usize);
    assert!(
        value < infinite,
        "the alignment DAG has a source-to-sink path with no probe on it"
    );

    // `max_flow` stops on the level pass that failed to reach the sink, and that
    // pass is exactly a BFS of the residual graph from the source — so `level`
    // already marks the source side, and no second traversal is needed. A node
    // is cut when its split edge straddles the two sides.
    (0..num_nodes as u32)
        .filter(|&node| {
            capacity[node as usize].is_some()
                && flow.level[entry(node) as usize] >= 0
                && flow.level[exit(node) as usize] < 0
        })
        .collect()
}

/// The cheapest set of probe nodes covering every layout of the pattern.
///
/// See [`min_cut`] for what "cheapest" is solved against; the
/// [`contained`](AlignmentGraph::contained) tokens are not part of it, being
/// mandatory rather than chosen.
pub(super) fn minimum_vertex_cut(graph: &AlignmentGraph) -> Vec<u32> {
    let capacity: Vec<Option<u32>> = (0..graph.num_nodes() as u32)
        .map(|node| graph.is_probe(node).then_some(graph.weight(node)))
        .collect();
    min_cut(&capacity, &graph.edges, graph.source, graph.sink)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the merged DAG exists for: two alignments converging on one
    /// shared suffix are cut once at the join for 6, not once each for 4 + 4.
    #[test]
    fn shared_suffix_beats_two_local_choices() {
        // 0 -> 1(4) -> 3(6) -> 4  and  0 -> 2(4) -> 3(6) -> 4
        let capacity = [None, Some(4), Some(4), Some(6), None];
        let edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)];
        assert_eq!(min_cut(&capacity, &edges, 0, 4), vec![3]);
    }

    /// Two disjoint paths have to be cut on both, and a zero-weight probe is
    /// always worth taking.
    #[test]
    fn disjoint_paths_are_cut_separately() {
        // 0 -> 1(5) -> 4  and  0 -> 2(9) -> 3(0) -> 4
        let capacity = [None, Some(5), Some(9), Some(0), None];
        let edges = [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)];
        assert_eq!(min_cut(&capacity, &edges, 0, 4), vec![1, 3]);
    }

    /// Depth is a property of the graph, not of any pattern length this solver
    /// gets to assume, so the DFS has to stay off the call stack. A recursive
    /// `send` overflows well before this chain does.
    #[test]
    fn deep_chain_does_not_exhaust_the_stack() {
        const LEN: u32 = 100_000;
        let mut capacity = vec![Some(7u32); LEN as usize];
        capacity[0] = None;
        capacity[LEN as usize - 1] = None;
        capacity[LEN as usize / 2] = Some(3);
        let edges: Vec<(u32, u32)> = (0..LEN - 1).map(|v| (v, v + 1)).collect();

        assert_eq!(min_cut(&capacity, &edges, 0, LEN - 1), vec![LEN / 2]);
    }
}
