// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The merged alignment DAG: every way the pattern can lie across token
//! boundaries, as one graph.
//!
//! A source-to-sink path is one such layout, and the probe nodes along it are
//! token sets a scan could look for to catch it. A set of probes meeting every
//! path is therefore a sound cover, and the cheapest one is a minimum-weight
//! vertex cut — which is what this graph exists to be handed to.
//!
//! # Nodes
//! * **Source and sink** — no probe.
//! * **Alignment** — one per feasible `k`, meaning the occurrence's first token
//!   ends with `needle[..k]`. No probe.
//! * **State** — one per reachable needle byte offset. No probe. Two alignments
//!   that reach the same offset face an identical remaining parse, so they share
//!   one state; that merge is what lets a single cut reason about every
//!   alignment at once, and it is also why the greedy parse can be memoized.
//! * **Probe** — [`ProbeSet::Point`] for an interior token of the greedy parse,
//!   [`ProbeSet::Range`] for the tokens a suffix of the needle is a prefix of
//!   (the occurrence ends inside a longer token), [`ProbeSet::Set`] for an
//!   enumerated first-token set. Weighted by term frequency; the only nodes a
//!   cut may select.
//!
//! Tokens whose bytes contain the *whole* needle are not in the graph at all.
//! They are mandatory — such an occurrence crosses no boundary, so no path
//! represents it — and go into the cover unconditionally.
//!
//! # Size
//! At most `3n + 32` nodes and `4n + 48` edges for a needle of `n` bytes, all in
//! flat arrays with no per-node allocation. Building it is dominated by the one
//! dictionary pass it shares with every other approach, not by the graph.

use memchr::memmem::Finder;

use super::ProbeWindow;
use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::search::index::TokenFrequencyIndexView;
use crate::search::prefix_range;

/// Largest first-token set still cheap enough to enumerate as an explicit
/// probe. A wider set is not built, so the alignment is entered unconditionally
/// and the cut has to pay for it further along the chain.
pub(super) const SET_CAP: usize = 16;

/// The token set a node probes for, or [`ProbeSet::None`] for the structural
/// nodes a cut may not select.
#[derive(Clone, Copy, Debug)]
pub(super) enum ProbeSet {
    /// Source, sink, alignment and state nodes.
    None,
    /// A single token: an interior token of the greedy parse.
    Point(Token),
    /// Every token a needle suffix is a prefix of.
    Range(TokenRange),
    /// A first-token set, as `first_set_ids[start..start + len]`.
    Set { start: u32, len: u32 },
}

/// The alignment DAG for one needle over one dictionary.
pub(super) struct AlignmentGraph {
    /// Probe per node, parallel with `weight`.
    probe: Vec<ProbeSet>,
    /// Maximum code radius around a hit of this probe node.
    window: Vec<Option<(usize, usize)>>,
    /// Term frequency of each node's probe; zero for structural nodes.
    weight: Vec<u32>,
    /// Every first-token set's ids, back to back.
    first_set_ids: Vec<Token>,
    /// Arcs as `(from, to)` node ids.
    pub(super) edges: Vec<(u32, u32)>,
    /// Tokens whose bytes contain the whole needle: mandatory, outside the cut.
    pub(super) contained: Vec<Token>,
    pub(super) source: u32,
    pub(super) sink: u32,
    pub(super) num_tokens: usize,
}

impl AlignmentGraph {
    pub(super) fn num_nodes(&self) -> usize {
        self.probe.len()
    }

    /// Term frequency of `node`'s probe, or zero if it carries none.
    pub(super) fn weight(&self, node: u32) -> u32 {
        self.weight[node as usize]
    }

    /// Whether a cut may select `node`.
    pub(super) fn is_probe(&self, node: u32) -> bool {
        !matches!(self.probe[node as usize], ProbeSet::None)
    }

    /// The token ids `cut`'s probes cover, as a membership table over the
    /// dictionary.
    ///
    /// The [`contained`](Self::contained) tokens are *not* included: they are
    /// mandatory rather than chosen, so unioning them in is the caller's job
    /// when it assembles the cover.
    pub(super) fn membership(&self, cut: &[u32]) -> Vec<bool> {
        let mut members = vec![false; self.num_tokens];
        for &node in cut {
            match self.probe[node as usize] {
                ProbeSet::None => debug_assert!(false, "cut selected a structural node"),
                ProbeSet::Point(id) => members[id as usize] = true,
                ProbeSet::Range(range) => {
                    for id in range.begin..=range.last {
                        members[id as usize] = true;
                    }
                }
                ProbeSet::Set { start, len } => {
                    for &id in &self.first_set_ids[start as usize..(start + len) as usize] {
                        members[id as usize] = true;
                    }
                }
            }
        }
        members
    }

    pub(super) fn localization(&self, cut: &[u32]) -> Vec<ProbeWindow> {
        let mut bounds = vec![None::<(usize, usize)>; self.num_tokens];
        let mut merge = |id: Token, before: usize, after: usize| {
            let slot = &mut bounds[id as usize];
            *slot = Some(slot.map_or((before, after), |(old_before, old_after)| {
                (old_before.max(before), old_after.max(after))
            }));
        };
        for &id in &self.contained {
            merge(id, 0, 0);
        }
        for &node in cut {
            let (before, after) =
                self.window[node as usize].expect("a selected probe carries localization bounds");
            match self.probe[node as usize] {
                ProbeSet::None => unreachable!("a cut cannot select a structural node"),
                ProbeSet::Point(id) => merge(id, before, after),
                ProbeSet::Range(range) => {
                    for id in range.begin..=range.last {
                        merge(id, before, after);
                    }
                }
                ProbeSet::Set { start, len } => {
                    for &id in &self.first_set_ids[start as usize..(start + len) as usize] {
                        merge(id, before, after);
                    }
                }
            }
        }
        bounds
            .into_iter()
            .enumerate()
            .filter_map(|(token, bounds)| {
                bounds.map(|(before_codes, after_codes)| ProbeWindow {
                    token: token as Token,
                    before_codes,
                    after_codes,
                })
            })
            .collect()
    }
}

/// Greedy longest in-needle token at `suffix`, capped at [`MAX_TOKEN_SIZE`].
/// Replicates the encoder's longest-prefix match restricted to the needle.
fn greedy_in_needle(dict: CompactDictionaryView<'_>, suffix: &[u8]) -> (Token, usize) {
    debug_assert!(
        !suffix.is_empty(),
        "greedy_in_needle needs a non-empty suffix"
    );
    for len in (1..=suffix.len().min(MAX_TOKEN_SIZE)).rev() {
        let range = prefix_range(dict, &suffix[..len]);
        if !range.is_empty() && dict.token_len(range.begin) == len {
            return (range.begin, len);
        }
    }
    let range = prefix_range(dict, &suffix[..1]);
    (range.begin, 1) // Complete dictionaries make this fallback reachable only by a bug.
}

/// Every token whose bytes contain the whole needle, and so matches it on its
/// own. Provably empty once the needle outgrows a token, which skips the pass.
///
/// Ascending, without duplicates.
pub(super) fn contained_tokens(dict: CompactDictionaryView<'_>, needle: &[u8]) -> Vec<Token> {
    let mut out = Vec::new();
    if needle.len() > MAX_TOKEN_SIZE {
        return out;
    }
    let (payload, offsets) = dict.token_payload();
    contained_by_search(payload, offsets, needle, &mut out);
    out
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
/// After an *accepted* match at `hit`, nothing before `hit + n` remains to be
/// found — this token is already recorded, and the next token begins at or after
/// `hit + n`. After a match *rejected* for spanning two tokens, resuming at
/// `hit + n` would step over the start of the token it ran into: with `a` and `aa`
/// adjacent in the payload, the rejected match of `aa` at offset 0 would hide the
/// real one at offset 1. A rejected match therefore resumes at the boundary it
/// crossed, the earliest offset at which a match can still lie inside one token.
fn contained_by_search(payload: &[u8], offsets: &[u32], needle: &[u8], out: &mut Vec<Token>) {
    let n = needle.len();
    let finder = Finder::new(needle);
    let mut from = 0usize;
    let mut id = 0usize;
    while let Some(found) = finder.find(&payload[from..]) {
        let hit = from + found;
        while offsets[id + 1] as usize <= hit {
            id += 1;
        }
        let end = offsets[id + 1] as usize;
        if hit + n <= end {
            if out.last() != Some(&(id as Token)) {
                out.push(id as Token);
            }
            from = hit + n;
        } else {
            from = end;
        }
        // Both arms advance: `hit < end` because `hit` lies inside token `id`.
    }
}

struct Builder<'d, 'n, 'f> {
    dict: CompactDictionaryView<'d>,
    needle: &'n [u8],
    frequencies: TokenFrequencyIndexView<'f>,
    probe: Vec<ProbeSet>,
    window: Vec<Option<(usize, usize)>>,
    weight: Vec<u32>,
    first_set_ids: Vec<Token>,
    edges: Vec<(u32, u32)>,
    sink: u32,
    /// `greedy[o]`: the greedy parse at needle offset `o`, computed at most
    /// once. Every alignment reaching `o` reuses it — the memo *is* the state
    /// merging.
    greedy: Vec<Option<(Token, usize)>>,
    state_node: Vec<Option<u32>>,
}

impl Builder<'_, '_, '_> {
    /// Term frequency of `set`. Summing an explicit set cannot overflow: its
    /// ids are distinct, so the sum is at most the code stream's length.
    fn weight_of(&self, set: ProbeSet) -> u32 {
        match set {
            ProbeSet::None => 0,
            ProbeSet::Point(id) => self.frequencies.frequency(id),
            ProbeSet::Range(range) => self.frequencies.range_frequency(range),
            ProbeSet::Set { start, len } => self.first_set_ids
                [start as usize..(start + len) as usize]
                .iter()
                .map(|&id| self.frequencies.frequency(id))
                .sum(),
        }
    }

    fn add_node(&mut self, set: ProbeSet, window: Option<(usize, usize)>) -> u32 {
        let weight = self.weight_of(set);
        self.probe.push(set);
        self.window.push(window);
        self.weight.push(weight);
        debug_assert!(self.probe.len() <= u32::MAX as usize, "node id overflow");
        (self.probe.len() - 1) as u32
    }

    fn add_edge(&mut self, from: u32, to: u32) {
        self.edges.push((from, to));
    }

    fn greedy_at(&mut self, offset: usize) -> (Token, usize) {
        if let Some(hit) = self.greedy[offset] {
            return hit;
        }
        let hit = greedy_in_needle(self.dict, &self.needle[offset..]);
        self.greedy[offset] = Some(hit);
        hit
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

    /// Materialize the state at `start` and every state its greedy chain
    /// reaches, stopping at the first offset already built. Iterative on
    /// purpose: `memmem` accepts needles of any length, so a recursive walk
    /// would put needle length on the stack.
    fn ensure_chain(&mut self, start: usize) -> u32 {
        let n = self.needle.len();
        let mut chain = Vec::new();
        let mut offset = start;
        while self.state_node[offset].is_none() {
            chain.push(offset);
            let (_, len) = self.greedy_at(offset);
            let next = offset + len;
            if next >= n {
                break;
            }
            offset = next;
        }
        // Reverse order, so each point probe's successor state already exists.
        for &offset in chain.iter().rev() {
            self.build_state(offset);
        }
        self.state_node[start].expect("the chain built the requested state")
    }

    fn build_state(&mut self, offset: usize) {
        let state = self.add_node(ProbeSet::None, None);
        self.state_node[offset] = Some(state);

        // The occurrence may end inside a longer token starting here.
        let terminal = self.terminal_range(offset);
        if !terminal.is_empty() {
            let node = self.add_node(ProbeSet::Range(terminal), Some((offset, 0)));
            self.add_edge(state, node);
            self.add_edge(node, self.sink);
        }

        let (token, len) = self.greedy_at(offset);
        let next = offset + len;
        if next < self.needle.len() {
            let node = self.add_node(
                ProbeSet::Point(token),
                Some((offset, self.needle.len() - next)),
            );
            let next_state =
                self.state_node[next].expect("states are built in reverse chain order");
            self.add_edge(state, node);
            self.add_edge(node, next_state);
        } else {
            debug_assert!(
                !terminal.is_empty(),
                "the exact final token belongs to its own prefix range"
            );
        }
    }
}

/// Build the alignment DAG for `needle` over `dict`, weighting each probe by
/// its term frequency in the indexed code stream.
pub(super) fn build_alignment_graph(
    dict: CompactDictionaryView<'_>,
    needle: &[u8],
    frequencies: TokenFrequencyIndexView<'_>,
) -> AlignmentGraph {
    debug_assert!(!needle.is_empty());
    debug_assert_eq!(frequencies.num_tokens(), dict.num_tokens());

    let n = needle.len();
    let ntok = dict.num_tokens();
    let contained = contained_tokens(dict, needle);

    let mut b = Builder {
        dict,
        needle,
        frequencies,
        probe: Vec::new(),
        window: Vec::new(),
        weight: Vec::new(),
        first_set_ids: Vec::new(),
        edges: Vec::new(),
        sink: 0,
        greedy: vec![None; n],
        state_node: vec![None; n],
    };
    let source = b.add_node(ProbeSet::None, None);
    let sink = b.add_node(ProbeSet::None, None);
    b.sink = sink;

    // First-token sets in one dictionary pass: for each alignment k >= 1, how
    // many tokens end with needle[..k] and — while the set stays small enough to
    // be worth probing for — which ones. At most `SET_CAP` ids for each of at
    // most `MAX_TOKEN_SIZE` alignments, so a fixed scratch array holds them all.
    //
    // The pass is driven by a final-byte table rather than by trying every k
    // against every token. A token can only end with `needle[..k]` if its own
    // last byte is `needle[k - 1]`, so one lookup yields the handful of k worth
    // comparing — for most tokens, none at all. That turns the inner loop from
    // `kmax` suffix comparisons into a load and a branch.
    let kmax = n.min(MAX_TOKEN_SIZE);
    let mut ks_ending_in = [0u16; 256];
    for k in 1..kmax {
        ks_ending_in[needle[k - 1] as usize] |= 1 << (k - 1);
    }
    let mut first_count = [0usize; MAX_TOKEN_SIZE];
    let mut first_ids = [0 as Token; MAX_TOKEN_SIZE * SET_CAP];
    for id in 0..ntok {
        let token = dict.token(id as Token);
        let len = token.len();
        // A suffix of length k needs a token at least that long.
        let fits = if len >= MAX_TOKEN_SIZE {
            u16::MAX
        } else {
            (1u16 << len) - 1
        };
        let mut ks = ks_ending_in[token[len - 1] as usize] & fits;
        while ks != 0 {
            let k = ks.trailing_zeros() as usize + 1;
            ks &= ks - 1;
            if token[len - k..] == needle[..k] {
                if first_count[k] < SET_CAP {
                    first_ids[k * SET_CAP + first_count[k]] = id as Token;
                }
                first_count[k] += 1;
            }
        }
    }

    for k in 0..kmax {
        // Alignment 0 is always feasible: the first token can start at the needle.
        if k != 0 && first_count[k] == 0 {
            continue;
        }
        let alignment = b.add_node(ProbeSet::None, None);
        b.add_edge(source, alignment);
        let state = b.ensure_chain(k);

        if k != 0 && first_count[k] <= SET_CAP {
            let start = b.first_set_ids.len() as u32;
            let len = first_count[k];
            b.first_set_ids
                .extend_from_slice(&first_ids[k * SET_CAP..k * SET_CAP + len]);
            let node = b.add_node(
                ProbeSet::Set {
                    start,
                    len: len as u32,
                },
                Some((0, n - k)),
            );
            b.add_edge(alignment, node);
            b.add_edge(node, state);
        } else {
            b.add_edge(alignment, state);
        }
    }

    AlignmentGraph {
        probe: b.probe,
        window: b.window,
        weight: b.weight,
        first_set_ids: b.first_set_ids,
        edges: b.edges,
        contained,
        source,
        sink,
        num_tokens: ntok,
    }
}
