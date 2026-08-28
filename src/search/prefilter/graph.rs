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
//! frequency. [`ProbeSet::SetTooBig`] is the one step a cut may not select: its
//! probe was never materialized, so it is entered free and the cut pays further
//! along the chain.
//!
//! An offset out of which no token is a prefix of the remaining needle has no
//! parse step of its own. A dictionary with an escape code answers that with the
//! escape; one without has to supply a transition for every byte itself, so a
//! needle that finds none is a malformed column rather than a shorter graph.
//!
//! A token whose bytes contain the *whole* needle is represented as an edge going
//! straight from source to sink. It will always be a part of the min cut.
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

/// The token set an edge probes for, or [`ProbeSet::SetTooBig`] for the one
/// step a cut may not select.
#[derive(Clone, Copy, Debug)]
pub(super) enum ProbeSet {
    /// A single token: an interior token of the greedy parse.
    Point(Token),
    /// Every token a needle suffix is a prefix of.
    Range(TokenRange),
    /// An explicit token set, as `set_ids[start..start + len]`.
    Set { start: u32, len: u32 },
    /// Would have been a [`Set`](ProbeSet::Set), but more than [`SET_CAP`] tokens qualified.
    SetTooBig,
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
            ProbeSet::SetTooBig => None,
            _ => Some(self.weight),
        }
    }

    /// Used for testing
    #[cfg(test)]
    pub(super) fn synthetic(from: u32, to: u32, cost: Option<u32>) -> Self {
        Self {
            from,
            to,
            probe: cost.map_or(ProbeSet::SetTooBig, |_| ProbeSet::Point(Token::MAX)),
            weight: cost.unwrap_or(0),
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
    /// Every explicit probe set's ids, back to back.
    set_ids: Vec<Token>,
    /// The parse steps, each carrying its probe and that probe's weight.
    pub(super) edges: Vec<Edge>,
    pub(super) nodes: Nodes,
    pub(super) num_tokens: usize,
}

impl AlignmentGraph {
    /// The token ids `cut`'s probes cover, as a membership table over the
    /// dictionary.
    pub(super) fn membership(&self, cut: &[&Edge], escape_token: Option<Token>) -> Vec<bool> {
        let escape_byte_range = escape_token.unwrap_or(0) as usize + 1;
        let members_size = self.num_tokens.max(escape_byte_range);
        let mut members = vec![false; members_size];
        for edge in cut {
            match edge.probe {
                ProbeSet::SetTooBig => debug_assert!(false, "cut selected an unprobed step"),
                ProbeSet::Point(id) => members[id as usize] = true,
                ProbeSet::Range(range) => {
                    for id in range.begin..=range.last {
                        members[id as usize] = true;
                    }
                }
                ProbeSet::Set { start, len } => {
                    for &id in &self.set_ids[start as usize..(start + len) as usize] {
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
    set_ids: Vec<Token>,
    edges: Vec<Edge>,
    nodes: Nodes,
    /// `greedy[o]`: the greedy parse at needle offset `o`, computed once when
    /// there is one. Every alignment reaching `o` reuses it — the memo *is* the
    /// state merging.
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
            ProbeSet::SetTooBig => 0,
            ProbeSet::Point(id) => self.frequencies.frequency(id),
            ProbeSet::Range(range) => self.frequencies.range_frequency(range),
            ProbeSet::Set { start, len } => self.set_ids[start as usize..(start + len) as usize]
                .iter()
                .map(|&id| self.frequencies.frequency(id))
                .sum(),
        }
    }

    /// Intern `ids` as an explicit probe set, which is what makes it weighable.
    fn create_probe_set(&mut self, ids: &[Token]) -> ProbeSet {
        let start = self.set_ids.len() as u32;
        self.set_ids.extend_from_slice(ids);
        ProbeSet::Set {
            start,
            len: ids.len() as u32,
        }
    }

    /// Add the step `from -> to`, which can be stepped by `probe`.
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
    /// token that is a prefix of what remains.
    fn greedy_at(&mut self, offset: usize) -> Option<(Token, usize)> {
        // A `None` result is not memoized: no token is a prefix of the rest of
        // the needle, so the lane dead-ends here and the offset is not
        // revisited.
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
    /// step lands, or `None` when there is no step to take. A successor is named
    /// by its offset, so it needs no node to exist yet — which is what lets the
    /// chain run forwards.
    ///
    /// # Dead ends
    /// No token being a prefix of `needle[offset..]` can happen if the dictionary does not
    /// contain all 0x00 to 0xFF bytes as tokens, like in the case of FSST. In this case, there
    /// needs to be an escape byte to allow the parse to continue. For simplicity, if we need
    /// an escape byte, we directly go to the sink instead of tracing the remaining chain.
    /// If we don't have an escape byte, we will panic.
    fn build_state(&mut self, offset: usize) -> usize {
        let state = offset as u32;

        // The occurrence may end inside a longer token starting here.
        //
        // Not at offset 0, though: every token the whole needle is a prefix of
        // also contains it, so the contained step already probes for them. A
        // parallel edge would make the cut pay twice for one path — and the two
        // are drawn as two cards naming the same token.
        let terminal_token_range = self.terminal_range(offset);
        if !terminal_token_range.is_empty() && offset != 0 {
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
        } else if let Some(escape_token) = self.escape_token {
            // if we find an escape token, we add an edge to the sink with the escape token
            self.add_edge(state, self.nodes.sink(), ProbeSet::Point(escape_token));
            self.needle.len()
        } else {
            // if no escape tokens are allowed, the dictionary must guarantee a transition e.g.,
            // by containing all the individual bytes.
            panic_malformed(InvalidColumn::IncompleteAlphabet)
        }
    }
}

/// Build the alignment DAG for `needle` over `dict`, weighting each probe by
/// its term frequency in the indexed code stream.
pub(super) fn build_alignment_graph(
    dict: CompactDictionaryView<'_>,
    needle: &[u8],
    frequencies: TokenFrequencyIndexView<'_>,
    escape_token: Option<Token>,
) -> AlignmentGraph {
    debug_assert!(!needle.is_empty());
    // for FSST we also have the frequency of escape bytes
    debug_assert!(frequencies.num_tokens() >= dict.num_tokens());

    let n = needle.len();
    debug_assert!(n < u32::MAX as usize, "needle outgrew u32 node ids");
    let ntok = dict.num_tokens();

    let mut b = Builder {
        dict,
        needle,
        frequencies,
        set_ids: Vec::new(),
        edges: Vec::new(),
        nodes: Nodes::new(n),
        greedy: vec![None; n],
        built: vec![false; n],
        escape_token,
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
                b.create_probe_set(&first_ids[k * SET_CAP..k * SET_CAP + first_count[k]])
            } else {
                ProbeSet::SetTooBig
            };
            b.add_edge(b.nodes.source(), k as u32, probe);
        }
    }

    // A token holding the whole needle needs no boundary, going from source to sink
    let contained = contained_tokens(dict, needle);
    if !contained.is_empty() {
        let probe = b.create_probe_set(&contained);
        b.add_edge(b.nodes.source(), b.nodes.sink(), probe);
    }

    // The bound the module doc advertises: two edges per needle offset, plus
    // one per alignment and one for the contained set. The node count is not a
    // bound but an identity.
    debug_assert!(
        b.edges.len() <= 2 * n + 16,
        "graph outgrew its documented size bound"
    );

    AlignmentGraph {
        set_ids: b.set_ids,
        edges: b.edges,
        nodes: b.nodes,
        num_tokens: ntok,
    }
}
