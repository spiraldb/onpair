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
//! occurrence ending inside a longer token. Probes are weighted by term
//! frequency. [`ProbeSet::TooBigSet`] is the one step a cut may not select: its
//! probe was never materialized, so it is entered free and the cut pays further
//! along the chain.
//!
//! An offset out of which no token is a prefix of the remaining needle has no
//! parse step of its own. A dictionary with an escape code answers that with the
//! escape; one without has to supply a transition for every byte itself, so a
//! needle that finds none is a malformed column rather than a shorter graph.
//!
//! Tokens whose bytes contain the *whole* needle are mandatory and join the
//! cover unconditionally, since such an occurrence crosses no boundary. The cut
//! never chooses them: the one edge that probes a subset of them, `0 -> n`, is
//! a source-to-sink path on its own, so every cut pays it alike.
//!
//! # Size
//! Exactly `n + 1` nodes and at most `2n + 16` edges for a needle of `n` bytes,
//! all in flat arrays with no per-node allocation. Building it is dominated by the one
//! dictionary pass it shares with every other approach, not by the graph.

use memchr::memmem::Finder;

use crate::core::dictionary::{CompactDictionaryView, DictionaryView};
use crate::core::types::{MAX_TOKEN_SIZE, Token, TokenRange};
use crate::core::validate::{InvalidColumn, panic_malformed};
use crate::search::index::TokenFrequencyIndexView;
use crate::search::prefix_range;

/// Largest first-token set still cheap enough to enumerate as an explicit
/// probe. A wider set is not built, so the alignment is entered unconditionally
/// and the cut has to pay for it further along the chain.
pub(super) const SET_CAP: usize = 16;

/// The token set an edge probes for, or [`ProbeSet::TooBigSet`] for the one
/// step a cut may not select.
#[derive(Clone, Copy, Debug)]
pub(super) enum ProbeSet {
    /// Would have been a [`Set`](ProbeSet::Set), but more than [`SET_CAP`] tokens qualified.
    TooBigSet,
    /// A single token: an interior token of the greedy parse.
    Point(Token),
    /// Every token a needle suffix is a prefix of.
    Range(TokenRange),
    /// A first-token set, as `first_set_ids[start..start + len]`.
    Set { start: u32, len: u32 },
}

/// One parse step, and the probe that catches every layout crossing it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Edge {
    pub(super) from: u32,
    pub(super) to: u32,
    /// The token set a scan looks for to catch this step.
    probe: ProbeSet,
    /// Term frequency of `probe`; zero when it carries none.
    weight: u32,
}

impl Edge {
    /// What cutting this edge costs, or `None` if no cut may select it.
    pub(super) fn cost(&self) -> Option<u32> {
        match self.probe {
            ProbeSet::TooBigSet => None,
            _ => Some(self.weight),
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
    /// Every first-token set's ids, back to back.
    first_set_ids: Vec<Token>,
    /// The parse steps, each carrying its probe and that probe's weight.
    pub(super) edges: Vec<Edge>,
    /// Tokens whose bytes contain the whole needle: mandatory, outside the cut.
    pub(super) contained: Vec<Token>,
    pub(super) nodes: Nodes,
    pub(super) num_tokens: usize,
}

impl AlignmentGraph {
    /// The token ids `cut`'s probes cover, as a membership table over the
    /// dictionary.
    ///
    /// The [`contained`](Self::contained) tokens are *not* included: they are
    /// mandatory rather than chosen, so unioning them in is the caller's job
    /// when it assembles the cover.
    pub(super) fn membership(&self, cut: &[u32]) -> Vec<bool> {
        let mut members = vec![false; self.num_tokens.max(256)];
        for &edge in cut {
            match self.edges[edge as usize].probe {
                ProbeSet::TooBigSet => debug_assert!(false, "cut selected an unprobed step"),
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
}

/// Greedy longest in-needle token at `suffix`, capped at [`MAX_TOKEN_SIZE`].
/// Replicates the encoder's longest-prefix match restricted to the needle.
///
/// `None` if no token is a prefix of `suffix`: the loop covers `len == 1`, so a
/// miss means the dictionary has no single-byte token for `suffix[0]`. Reading an
/// id off the empty range instead would probe for an unrelated token and quietly
/// cost selectivity, so the caller decides — an escape, or a malformed panic.
fn greedy_in_needle(dict: CompactDictionaryView<'_>, suffix: &[u8]) -> Option<(Token, usize)> {
    debug_assert!(
        !suffix.is_empty(),
        "greedy_in_needle needs a non-empty suffix"
    );
    for len in (1..=suffix.len().min(MAX_TOKEN_SIZE)).rev() {
        let range = prefix_range(dict, &suffix[..len]);
        if !range.is_empty() && dict.token_len(range.begin) == len {
            return Some((range.begin, len));
        }
    }

    None
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
    first_set_ids: Vec<Token>,
    edges: Vec<Edge>,
    nodes: Nodes,
    /// `greedy[o]`: the greedy parse at needle offset `o`, computed at most
    /// once. Every alignment reaching `o` reuses it — the memo *is* the state
    /// merging.
    greedy: Vec<Option<(Token, usize)>>,
    /// `built[o]`: whether the steps out of offset `o` have been emitted. Node
    /// ids are offsets, so this is all the builder has left to track.
    built: Vec<bool>,
    /// The code standing in for a byte no token covers, or `None` when the
    /// dictionary has to supply a transition for every byte itself.
    escape_token: Option<Token>,
}

impl Builder<'_, '_, '_> {
    /// Term frequency of `set`. Summing an explicit set cannot overflow: its
    /// ids are distinct, so the sum is at most the code stream's length.
    fn weight_of(&self, set: ProbeSet) -> u32 {
        match set {
            ProbeSet::TooBigSet => 0,
            ProbeSet::Point(id) => self.frequencies.frequency(id),
            ProbeSet::Range(range) => self.frequencies.range_frequency(range),
            ProbeSet::Set { start, len } => self.first_set_ids
                [start as usize..(start + len) as usize]
                .iter()
                .map(|&id| self.frequencies.frequency(id))
                .sum(),
        }
    }

    /// Add the step `from -> to`, probed by `probe`. Any [`ProbeSet::Set`] must
    /// already have its ids in `first_set_ids`, since the weight is summed here.
    fn add_edge(&mut self, from: u32, to: u32, probe: ProbeSet) {
        let weight = self.weight_of(probe);
        self.edges.push(Edge {
            from,
            to,
            probe,
            weight,
        });
    }

    /// The greedy step out of `offset`, or `None` when the dictionary holds no
    /// token that is a prefix of what remains — see [`build_state`](Self::build_state).
    fn greedy_at(&mut self, offset: usize) -> (Token, usize) {
        if let Some(hit) = self.greedy[offset] {
            return hit;
        }
        if let Some(hit) = greedy_in_needle(self.dict, &self.needle[offset..]) {
            self.greedy[offset] = Some(hit);
            return hit;
        }

        if let Some(escape_token) = self.escape_token {
            return (escape_token, 1);
        }
        // if no escape tokens are allowed, the dictionary must guarantee a transition e.g.,
        // by containing all the individual bytes.
        panic_malformed(InvalidColumn::IncompleteAlphabet)
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
    /// step lands, or `None` when there is no step to take. A successor is named
    /// by its offset, so it needs no node to exist yet — which is what lets the
    /// chain run forwards.
    ///
    /// # Dead ends
    /// No token being a prefix of `needle[offset..]` is not malformed input: it
    /// says no tokenization of any row can have a boundary here with the needle
    /// running past it. A token starting at `offset` either ends inside the needle
    /// — then its bytes *are* such a prefix — or covers the rest of it, and that
    /// case is the terminal range emitted above. So the lane simply ends, and
    /// since the state can no longer reach the accept node, no layout runs through
    /// it and the cut owes nothing for it.
    ///
    /// A complete alphabet makes this unreachable, since every byte is then a
    /// token of its own. An FSST table is the case that reaches it.
    fn build_state(&mut self, offset: usize) -> usize {
        let state = offset as u32;

        // The occurrence may end inside a longer token starting here.
        let terminal = self.terminal_range(offset);
        if !terminal.is_empty() {
            self.add_edge(state, self.nodes.sink(), ProbeSet::Range(terminal));
        }

        let (token, len) = self.greedy_at(offset);
        let next = offset + len;
        if next < self.needle.len() {
            self.add_edge(state, next as u32, ProbeSet::Point(token));
        } else {
            // A greedy step that reaches the needle's end consumed
            // `needle[offset..]` exactly, so that token is in its own prefix
            // range: this step is `offset -> sink`, already probed above by a
            // range containing it. A parallel point probe would only make the
            // cut pay twice for one step.
            debug_assert!(
                terminal.contains(token),
                "the exact final token belongs to its own prefix range"
            );
        }
        next
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
    // for FSST we also have the frequency of escape bytes
    debug_assert!(frequencies.num_tokens() >= dict.num_tokens());

    let n = needle.len();
    debug_assert!(n < u32::MAX as usize, "needle outgrew u32 node ids");
    let ntok = dict.num_tokens();
    let contained = contained_tokens(dict, needle);

    let mut b = Builder {
        dict,
        needle,
        frequencies,
        first_set_ids: Vec::new(),
        edges: Vec::new(),
        nodes: Nodes { needle_len: n },
        greedy: vec![None; n],
        built: vec![false; n],
        escape_token: Some(0xFF),
    };

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
        // Emits the chain of steps from this alignment's entry to the sink.
        b.ensure_chain(k);

        // Alignment 0 begins at the source node itself, with nothing consumed
        // before its first token, so it needs no entry step at all. At `k > 0`
        // the occurrence starts inside its first token, which covered
        // `needle[..k]` as its tail; the set of tokens it could be is what
        // probes that step — when the pass kept them.
        if k != 0 {
            let probe = if first_count[k] <= SET_CAP {
                let start = b.first_set_ids.len() as u32;
                let len = first_count[k];
                b.first_set_ids
                    .extend_from_slice(&first_ids[k * SET_CAP..k * SET_CAP + len]);
                ProbeSet::Set {
                    start,
                    len: len as u32,
                }
            } else {
                ProbeSet::TooBigSet
            };
            b.add_edge(b.nodes.source(), k as u32, probe);
        }
    }

    // The bound the module doc advertises: two edges per needle offset, plus
    // one per alignment. The node count is not a bound but an identity.
    debug_assert!(
        b.edges.len() <= 2 * n + 16,
        "graph outgrew its documented size bound"
    );

    AlignmentGraph {
        first_set_ids: b.first_set_ids,
        edges: b.edges,
        contained,
        nodes: b.nodes,
        num_tokens: ntok,
    }
}
