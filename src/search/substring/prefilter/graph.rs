// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The merged alignment DAG: every way the pattern can lie across token
//! boundaries, as one graph.
//!
//! A source-to-sink path is one such layout, and the probes on its edges are
//! token sets a scan could look for to catch it. A set of probes meeting every
//! path is therefore a sound cover, and the cheapest one is a minimum-weight
//! cut — which is what this graph exists to be handed to.
//!
//! # Nodes and edges
//! A node is a parse position, an edge is a parse step, and probing is a
//! property of a step — so probes ride on the edges and a cut selects edges.
//!
//! A node id *is* a needle offset: node `o` is the position at `needle[o..]`,
//! node `0` the source and node `n` the sink with every byte consumed, and an
//! edge `o -> o'` means one token covered `needle[o..o']`. Two alignments
//! reaching the same offset therefore land on the same node facing an identical
//! remaining parse: state merging is what the numbering means rather than
//! something the builder arranges, and it is why the greedy parse can be
//! memoized. Offsets the parse never reaches are isolated nodes.
//!
//! Out of the source, one edge per feasible alignment `k >= 1`, whose first
//! token ends with `needle[..k]` and so covered those bytes as its tail, probed
//! by [`ProbeSet::Set`] — the tokens that token can be. Alignment `0` needs no
//! edge: nothing precedes its first token, so its layouts begin at the source.
//! Between states, [`ProbeSet::Point`] for a token of the greedy parse; into the
//! sink, [`ProbeSet::Range`] for the tokens a needle suffix is a prefix of, the
//! occurrence ending inside a longer token. Each probe carries its term
//! frequency; what cutting it costs is the cut's caller's to say.
//! [`ProbeSet::SetTooBig`] is the one step a cut may not select: its
//! probe was never materialized, so it is entered free and the cut pays further
//! along the chain. Alignment `1` is that step for nearly every first byte:
//! a byte ends more than [`PROBE_SET_SIZE_LIMIT_K1`] tokens unless it is rare.
//!
//! An offset out of which no token is a prefix of the remaining needle has no
//! parse step of its own. The dictionary has to supply a transition for every
//! byte itself, so a needle that finds none is a malformed column rather than a
//! shorter graph.
//!
//! A token whose bytes contain the *whole* needle is represented as an edge going
//! straight from source to sink. It will always be a part of the min cut. Two
//! such edges exist: the needle at a token's start is the terminal range out of
//! node 0, the needle further in is the contained set.
//!
//! # Size
//! Exactly `n + 1` nodes and at most `2n + 16` edges for a needle of `n` bytes,
//! all in flat arrays with no per-node allocation. Building it is dominated by the one
//! dictionary pass it shares with every other approach, not by the graph.

use memchr::memmem::Finder;

use super::cover::ProbeCover;
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::core::validate::{InvalidColumn, panic_malformed};
use crate::search::index::TokenFrequencyIndexView;
use crate::search::lookup::narrow;
use crate::search::prefix_range;

/// Largest first-token set still worth enumerating as an explicit probe.
pub(super) const PROBE_SET_SIZE_LIMIT: usize = 512;

/// Alignment 1's own, much lower limit: for needle `"XYZ"` every token ending
/// in `X` qualifies, while at `k = 2` only the far fewer ending in `XY` do.
pub(super) const PROBE_SET_SIZE_LIMIT_K1: usize = 16;

/// The token set an edge probes for, or [`ProbeSet::SetTooBig`] for the one
/// step a cut may not select.
#[derive(Clone, Debug)]
pub(super) enum ProbeSet {
    /// A single token: an interior token of the greedy parse.
    Point(Token),
    /// Every token a needle suffix is a prefix of.
    Range(TokenRange),
    /// An explicit token set.
    Set(Box<[Token]>),
    /// Would have been a [`Set`](ProbeSet::Set), but more than
    /// [`PROBE_SET_SIZE_LIMIT`] tokens qualified.
    SetTooBig,
}

/// One parse step, and the probe that catches every layout crossing it.
#[derive(Clone, Debug)]
pub(super) struct Edge {
    pub(super) from: u32,
    pub(super) to: u32,
    /// The token set a scan looks for to catch this step.
    probe: ProbeSet,
    /// Term frequency of `probe`; zero when it carries none.
    frequency: u32,
}

impl Edge {
    /// Whether a cut may select this edge: it carries a probe.
    pub(super) fn cuttable(&self) -> bool {
        !matches!(self.probe, ProbeSet::SetTooBig)
    }

    /// The token set this step probes for.
    pub(super) fn probe(&self) -> &ProbeSet {
        &self.probe
    }

    /// Codes the probe matches in the indexed stream.
    pub(super) fn frequency(&self) -> u32 {
        self.frequency
    }

    /// What the probe puts in a cover, as `(points, ranges)` before adjacent
    /// ones merge.
    pub(super) fn shape(&self) -> (u32, u32) {
        match &self.probe {
            ProbeSet::SetTooBig => (0, 0),
            ProbeSet::Point(_) => (1, 0),
            ProbeSet::Range(_) => (0, 1),
            ProbeSet::Set(ids) => (ids.len() as u32, 0),
        }
    }

    /// Used for testing
    #[cfg(test)]
    pub(super) fn synthetic(from: u32, to: u32, frequency: Option<u32>) -> Self {
        Self {
            from,
            to,
            probe: frequency.map_or(ProbeSet::SetTooBig, |_| ProbeSet::Point(Token::MAX)),
            frequency: frequency.unwrap_or(0),
        }
    }
}

/// The node numbering, which the needle's length fixes entirely: ids are needle
/// offsets, so there is nothing else to know about the node set.
#[derive(Clone, Copy, Debug)]
pub(super) struct Nodes {
    needle_len: usize,
}

impl Nodes {
    pub(super) fn new(needle_len: usize) -> Self {
        Self { needle_len }
    }

    pub(super) fn count(self) -> usize {
        self.needle_len + 1
    }

    /// Where every layout begins: no needle byte consumed yet.
    pub(super) fn source(self) -> u32 {
        0
    }

    /// Where every layout ends: every needle byte consumed.
    pub(super) fn sink(self) -> u32 {
        self.needle_len as u32
    }
}

/// The alignment DAG for one needle over one dictionary.
pub(super) struct AlignmentGraph {
    /// The parse steps, each carrying its probe and that probe's frequency.
    pub(super) edges: Vec<Edge>,
    pub(super) nodes: Nodes,
}

impl ProbeCover {
    /// The cover `cut`'s probes form: every id a run of one, every range
    /// itself. Merging is [`from_runs`](Self::from_runs)'s work.
    pub(super) fn from_edge_cut(cut: &[&Edge]) -> Self {
        let point = |id: Token| TokenRange {
            begin: id,
            last: id,
        };
        let mut runs = Vec::with_capacity(cut.len());
        for edge in cut {
            match &edge.probe {
                ProbeSet::SetTooBig => debug_assert!(false, "cut selected an unprobed step"),
                ProbeSet::Point(id) => runs.push(point(*id)),
                ProbeSet::Range(range) => runs.push(*range),
                ProbeSet::Set(ids) => runs.extend(ids.iter().map(|&id| point(id))),
            }
        }
        Self::from_runs(runs)
    }
}

/// Greedy longest in-needle token at `suffix`, capped at [`MAX_TOKEN_SIZE`].
/// Replicates the encoder's longest-prefix match restricted to the needle.
///
/// `None` if no token is a prefix of `suffix`: the walk covers `len == 1`, so a
/// miss means the dictionary has no single-byte token for `suffix[0]`. Reading an
/// id off the empty range instead would probe for an unrelated token and quietly
/// cost selectivity, so the caller panics on the malformed column instead.
///
/// One narrowing walk rather than a [`prefix_range`] per candidate length: the
/// range for `suffix[..k + 1]` is the range for `suffix[..k]` narrowed by one
/// more byte, so restarting at each length would re-search the whole dictionary
/// every time. Once the range empties no longer prefix can match, and a longer
/// prefix always narrows further, so the walk stops there.
fn greedy_in_needle(dict: CompactDictionaryView<'_>, suffix: &[u8]) -> Option<(Token, usize)> {
    debug_assert!(
        !suffix.is_empty(),
        "greedy_in_needle needs a non-empty suffix"
    );
    let num_tokens = dict.num_tokens();
    if num_tokens == 0 {
        return None;
    }
    let mut range = TokenRange {
        begin: 0,
        last: (num_tokens - 1) as Token,
    };
    let mut longest = None;
    for (k, &byte) in suffix.iter().take(MAX_TOKEN_SIZE).enumerate() {
        range = narrow(dict, range, k, byte);
        if range.is_empty() {
            break;
        }
        // Every token in `range` starts with `suffix[..=k]`; the shortest sorts
        // first, and it *is* that prefix exactly when its length matches.
        if dict.token_len(range.begin) == k + 1 {
            longest = Some((range.begin, k + 1));
        }
    }
    longest
}

/// One vectorized sweep of the whole token payload, attributing each match to the
/// token holding it, in place of a windowed comparison per token.
///
/// Attribution is a cursor rather than a search: matches arrive in ascending
/// order, so the cursor crosses each token boundary at most once across the whole
/// sweep.
///
/// # Where to resume
/// This is the subtle part, and plain non-overlapping iteration gets it wrong.
struct Builder<'d, 'n, 'f> {
    dict: CompactDictionaryView<'d>,
    needle: &'n [u8],
    frequencies: TokenFrequencyIndexView<'f>,
    edges: Vec<Edge>,
    nodes: Nodes,
    /// `greedy[o]`: the greedy parse at needle offset `o`, computed once when
    /// there is one. Every alignment reaching `o` reuses it — the memo *is* the
    /// state merging.
    greedy: Vec<Option<(Token, usize)>>,
    /// `built[o]`: whether the steps out of offset `o` have been emitted. Node
    /// ids are offsets, so this is all the builder has left to track.
    built: Vec<bool>,
}

impl Builder<'_, '_, '_> {
    /// Term frequency of `set`. Summing an explicit set cannot overflow: its
    /// ids are distinct, so the sum is at most the code stream's length.
    fn frequency_of(&self, set: &ProbeSet) -> u32 {
        match set {
            ProbeSet::SetTooBig => 0,
            ProbeSet::Point(id) => self.frequencies.frequency(*id),
            ProbeSet::Range(range) => self.frequencies.range_frequency(*range),
            ProbeSet::Set(ids) => ids.iter().map(|&id| self.frequencies.frequency(id)).sum(),
        }
    }

    /// Add the step `from -> to`, which can be stepped by `probe`.
    fn add_edge(&mut self, from: u32, to: u32, probe: ProbeSet) {
        let frequency = self.frequency_of(&probe);
        self.edges.push(Edge {
            from,
            to,
            probe,
            frequency,
        });
    }

    /// The greedy step out of `offset`, or `None` when the dictionary holds no
    /// token that is a prefix of what remains.
    fn greedy_at(&mut self, offset: usize) -> Option<(Token, usize)> {
        if self.greedy[offset].is_none() {
            self.greedy[offset] = greedy_in_needle(self.dict, &self.needle[offset..]);
        }
        self.greedy[offset]
    }

    /// The tokens the needle suffix at `offset` is a prefix of, or the empty
    /// range when there are none. A suffix longer than a token can be no
    /// token's prefix, so that case skips the dictionary search outright.
    fn terminal_range(&self, offset: usize) -> TokenRange {
        let suffix = &self.needle[offset..];
        if suffix.len() > MAX_TOKEN_SIZE {
            return TokenRange::EMPTY;
        }
        prefix_range(self.dict, suffix)
    }

    /// Emit the steps out of `start` and out of every offset its greedy chain
    /// reaches, stopping at the first offset already emitted. Iterative on
    /// purpose: `memmem` accepts needles of any length, so a recursive walk
    /// would put needle length on the stack.
    fn ensure_chain(&mut self, start: usize) {
        let n = self.needle.len();
        let mut offset = start;
        while !self.built[offset] {
            self.built[offset] = true;
            let next = self.build_state(offset);
            if next >= n {
                break;
            }
            offset = next;
        }
    }

    /// Emit the steps out of the state at `offset` and return where its greedy
    /// step lands. A successor is named by its offset, so it needs no node to
    /// exist yet, which is what lets the chain run forwards.
    ///
    /// # Dead ends
    /// No token is a prefix of `needle[offset..]` only when the dictionary lacks
    /// a single-byte token for that byte, which a conformant one cannot, so the
    /// dead end is a malformed column rather than a shorter graph.
    fn build_state(&mut self, offset: usize) -> usize {
        let state = offset as u32;

        // The occurrence may end inside a longer token starting here. At offset
        // 0 that token is one the needle is a prefix of and the step runs
        // source to sink, which is why `contained_tokens` leaves those out: one
        // interval probes them, where the contained set would spend an id each.
        let terminal_token_range = self.terminal_range(offset);
        if !terminal_token_range.is_empty() {
            self.add_edge(
                state,
                self.nodes.sink(),
                ProbeSet::Range(terminal_token_range),
            );
        }

        let next_token = self.greedy_at(offset);

        if let Some((token, token_length)) = next_token {
            let next_offset = offset + token_length;
            if next_offset < self.needle.len() {
                self.add_edge(state, next_offset as u32, ProbeSet::Point(token));
            } else {
                // A greedy step that reaches the needle's end consumed
                // `needle[offset..]` exactly, so that token should appear in the
                // terminal range.
                debug_assert!(
                    terminal_token_range.contains(token),
                    "the exact final token belongs to its own prefix range"
                );
            }
            next_offset
        } else {
            panic_malformed(InvalidColumn::IncompleteAlphabet)
        }
    }
}

/// For each alignment `k >= 1`, how many tokens end with `needle[..k]` and,
/// while the set stays small enough to be worth probing for, which ones. At
/// most [`PROBE_SET_SIZE_LIMIT`] ids for each of at most [`MAX_TOKEN_SIZE`] alignments, so
/// fixed arrays hold them all.
///
/// Counts saturate one past [`PROBE_SET_SIZE_LIMIT`]: the planner asks only whether a set is
/// empty or too big to name, so counting further would keep the pass running
/// for an answer nothing reads.
pub(super) struct FirstTokenSets {
    pub(super) count: [usize; MAX_TOKEN_SIZE],
    pub(super) ids: [Token; MAX_TOKEN_SIZE * PROBE_SET_SIZE_LIMIT],
}

impl FirstTokenSets {
    fn new() -> Self {
        Self {
            count: [0; MAX_TOKEN_SIZE],
            ids: [0; MAX_TOKEN_SIZE * PROBE_SET_SIZE_LIMIT],
        }
    }

    fn record(&mut self, k: usize, id: usize) {
        if self.count[k] < PROBE_SET_SIZE_LIMIT {
            self.ids[k * PROBE_SET_SIZE_LIMIT + self.count[k]] = id as Token;
        }
        self.count[k] = (self.count[k] + 1).min(PROBE_SET_SIZE_LIMIT + 1);
    }
}

/// Everything one needle needs from the dictionary.
pub(super) struct Candidates {
    pub(super) first: FirstTokenSets,
    /// Tokens holding the whole needle at a non-zero offset, ascending and
    /// without duplicates. A token that starts with the needle is in the
    /// terminal range instead, even when it holds the needle again further in.
    pub(super) contained: Vec<Token>,
}

/// Both sets from the dictionary.
pub(super) fn alignment_candidates(dict: CompactDictionaryView<'_>, needle: &[u8]) -> Candidates {
    let (payload, offsets) = dict.token_payload();
    let mut candidates = Candidates {
        first: FirstTokenSets::new(),
        contained: Vec::new(),
    };
    if needle.len() > 1 {
        alignment_k1_candidates(payload, offsets, needle[0], &mut candidates.first);
    }
    sweep_for_candidates(payload, offsets, needle, &mut candidates);
    candidates
}

/// Alignment 1: the tokens ending in `byte`, with `needle[1]` starting the
/// next token. A token that is only `byte` is alignment 0's and skipped. Most bytes end far more than [`PROBE_SET_SIZE_LIMIT_K1`]
/// tokens, so the pass stops there and saturates the count, which marks the
/// set as never enumerated rather than as one that fit.
fn alignment_k1_candidates(payload: &[u8], offsets: &[u32], byte: u8, first: &mut FirstTokenSets) {
    for (token_id, bounds) in offsets.windows(2).enumerate() {
        let (token_begin, token_end) = (bounds[0] as usize, bounds[1] as usize);
        if token_end - token_begin > 1 && payload[token_end - 1] == byte {
            first.record(1, token_id);
            if first.count[1] > PROBE_SET_SIZE_LIMIT_K1 {
                first.count[1] = PROBE_SET_SIZE_LIMIT + 1;
                return;
            }
        }
    }
}

/// One `memmem` sweep of the payload for `needle[..2]`, or for the whole of a
/// one-byte needle. Every alignment `k >= 2` and every contained occurrence
/// begins with it, and two adjacent bytes are far rarer than one.
///
/// A hit's tail, from the hit to its token's end, says what the hit is: an
/// alignment when it is shorter than the needle and a prefix of it, a
/// contained occurrence when it starts with the needle. A hit at a token's
/// start is alignment 0's or the terminal range's and is skipped.
fn sweep_for_candidates(
    payload: &[u8],
    offsets: &[u32],
    needle: &[u8],
    candidates: &mut Candidates,
) {
    let needle_len = needle.len();
    let finder = Finder::new(&needle[..needle_len.min(2)]);
    let (mut search_from, mut token_id) = (0usize, 0usize);
    while let Some(found_offset) = finder.find(&payload[search_from..]) {
        let hit_offset = search_from + found_offset;
        search_from = hit_offset + 1;
        while offsets[token_id + 1] as usize <= hit_offset {
            token_id += 1;
        }
        let (token_begin, token_end) = (offsets[token_id] as usize, offsets[token_id + 1] as usize);
        let token_tail_from_hit = &payload[hit_offset..token_end];
        if token_tail_from_hit.len() < needle_len.min(2) {
            continue;
        }
        let at_token_start = hit_offset == token_begin;
        if token_tail_from_hit.starts_with(needle) {
            if !at_token_start {
                candidates.contained.push(token_id as Token);
            }
            search_from = token_end + 1 - needle_len;
        } else if !at_token_start
            && token_tail_from_hit.len() < needle_len
            && needle.starts_with(token_tail_from_hit)
        {
            // The tail is a prefix of the needle, so the token ends with exactly
            // that many needle bytes: alignment k is its length.
            candidates.first.record(token_tail_from_hit.len(), token_id);
        }
    }
}

/// Build the alignment DAG for `needle` over `dict`, each probe carrying its
/// term frequency in the indexed code stream.
pub(super) fn build_alignment_graph(
    dict: CompactDictionaryView<'_>,
    needle: &[u8],
    frequencies: TokenFrequencyIndexView<'_>,
) -> AlignmentGraph {
    debug_assert!(!needle.is_empty());
    debug_assert!(frequencies.num_tokens() == dict.num_tokens());

    let n = needle.len();
    debug_assert!(n < u32::MAX as usize, "needle outgrew u32 node ids");

    let mut b = Builder {
        dict,
        needle,
        frequencies,
        edges: Vec::new(),
        nodes: Nodes::new(n),
        greedy: vec![None; n],
        built: vec![false; n],
    };

    let kmax = n.min(MAX_TOKEN_SIZE);
    let candidates = alignment_candidates(dict, needle);
    let (first, contained) = (&candidates.first, candidates.contained);

    for k in 0..kmax {
        // Alignment 0 is always feasible: the first token can start at the needle.
        if k != 0 && first.count[k] == 0 {
            continue;
        }
        // Emits the chain of steps from this alignment's entry to the sink.
        b.ensure_chain(k);

        // Alignment 0 begins at the source node itself, with nothing consumed
        // before its first token, so it needs no entry step at all. At `k > 0`
        // the occurrence starts inside its first token, which covered
        // `needle[..k]` as its tail; the set of tokens it could be is what
        // probes that step — when the pass kept them.
        if k != 0 {
            let probe = if first.count[k] <= PROBE_SET_SIZE_LIMIT {
                ProbeSet::Set(
                    first.ids[k * PROBE_SET_SIZE_LIMIT..k * PROBE_SET_SIZE_LIMIT + first.count[k]]
                        .into(),
                )
            } else {
                ProbeSet::SetTooBig
            };
            b.add_edge(b.nodes.source(), k as u32, probe);
        }
    }

    // A token holding the whole needle needs no boundary, going from source to sink
    if !contained.is_empty() {
        b.add_edge(
            b.nodes.source(),
            b.nodes.sink(),
            ProbeSet::Set(contained.into()),
        );
    }

    // The bound the module doc advertises: two edges per needle offset, plus
    // one per alignment and one for the contained set. The node count is not a
    // bound but an identity.
    debug_assert!(
        b.edges.len() <= 2 * n + 16,
        "graph outgrew its documented size bound"
    );

    AlignmentGraph {
        edges: b.edges,
        nodes: b.nodes,
    }
}
