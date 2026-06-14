// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Generalized OnPair over an *integer element alphabet* instead of bytes.
//
// The byte OnPair compressor (this crate's parent) treats a string column as
// `(concatenated bytes, per-row offsets)` and builds an incremental-BPE
// dictionary of byte sequences. A dictionary-encoded list column is exactly the
// same shape — `(concatenated element-ids, per-row offsets)` — only the
// alphabet is the set of distinct elements (e.g. distinct stack frames) rather
// than the 256 byte values. This module ports the technique to that alphabet:
//
//   * base tokens  : one per distinct element id (vs. 256 single bytes)
//   * merge step   : frequent adjacent token *pairs* merged into longer tokens,
//                    capped at MAX_TOKEN_ELEMS *elements* (vs. 16 bytes)
//   * encode       : greedy longest-prefix match -> code stream
//
// The dynamic-threshold controller is ported verbatim from the byte trainer
// (counting elements instead of bytes). The LPM is a simple length-probing hash
// map keyed by element slices: this is an offline ratio experiment, so clarity
// beats the byte version's u64-packing micro-optimizations.

use hashbrown::HashMap;

/// Max elements per merged token (the element-count analogue of the byte
/// compressor's `MAX_TOKEN_SIZE = 16`).
pub const MAX_TOKEN_ELEMS: usize = 16;

/// Decode-side dictionary: token `i` is `elems[offsets[i]..offsets[i + 1]]`,
/// each entry a base element id.
#[derive(Default, Clone)]
pub struct Dictionary {
    pub elems: Vec<u32>,
    pub offsets: Vec<u32>,
}

impl Dictionary {
    pub fn num_tokens(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn token(&self, id: u32) -> &[u32] {
        let b = self.offsets[id as usize] as usize;
        let e = self.offsets[id as usize + 1] as usize;
        &self.elems[b..e]
    }
}

/// Longest-prefix matcher over element sequences. `get` works on a borrowed
/// slice because `Box<[u32]>: Borrow<[u32]>`, so probing needs no allocation.
struct Lpm {
    map: HashMap<Box<[u32]>, u32>,
    next_id: u32,
}

impl Lpm {
    fn with_base(num_distinct: u32) -> Self {
        let mut map = HashMap::with_capacity(num_distinct as usize);
        for e in 0..num_distinct {
            map.insert(vec![e].into_boxed_slice(), e);
        }
        Self {
            map,
            next_id: num_distinct,
        }
    }

    fn insert(&mut self, seq: &[u32]) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(seq.to_vec().into_boxed_slice(), id);
        id
    }

    fn size(&self) -> usize {
        self.next_id as usize
    }

    /// Longest token that is a prefix of `data`, with its element length.
    /// Total whenever every base element is present (always, after `with_base`).
    #[inline]
    fn find_longest_match(&self, data: &[u32]) -> (u32, usize) {
        let max = data.len().min(MAX_TOKEN_ELEMS);
        for len in (1..=max).rev() {
            if let Some(&id) = self.map.get(&data[..len]) {
                return (id, len);
            }
        }
        unreachable!("base element token must always match")
    }
}

// ── Dynamic-threshold controller (ported from src/trainer.rs) ────────────────

struct DynamicThreshold {
    capacity: usize,
    scan_budget: usize,
    check_interval: usize,
    threshold: u8,
    entries_created: usize,
    elems_scanned: usize,
    entries_at_check: usize,
    elems_at_check: usize,
    next_checkpoint: usize,
}

impl DynamicThreshold {
    fn new(capacity: usize, total_elems: usize, scan_fraction: f64) -> Self {
        let scan_budget = (total_elems as f64 * scan_fraction) as usize;
        let check_interval = (capacity / 128).max(64);
        Self {
            capacity,
            scan_budget,
            check_interval,
            threshold: 2,
            entries_created: 0,
            elems_scanned: 0,
            entries_at_check: 0,
            elems_at_check: 0,
            next_checkpoint: check_interval,
        }
    }

    fn budget_exhausted(&self) -> bool {
        self.elems_scanned > self.scan_budget
    }

    fn on_elems_scanned(&mut self, n: usize) {
        self.elems_scanned += n;
    }

    fn on_entry_created(&mut self) {
        self.entries_created += 1;
        if self.entries_created >= self.next_checkpoint {
            self.rebalance();
        }
    }

    fn rebalance(&mut self) {
        let delta_e = self.entries_created - self.entries_at_check;
        let delta_b = self.elems_scanned - self.elems_at_check;
        let recent_rate = if delta_b > 0 {
            delta_e as f64 / delta_b as f64
        } else {
            1e9
        };
        let e_rem = self.capacity.saturating_sub(self.entries_created).max(1);
        let b_rem = self.scan_budget.saturating_sub(self.elems_scanned).max(1);
        let target_rate = e_rem as f64 / b_rem as f64;
        let ratio = if target_rate > 0.0 {
            recent_rate / target_rate
        } else {
            1e9
        };
        if ratio > 2.0 && self.threshold < 255 {
            self.threshold += 1;
        } else if ratio < 0.5 {
            self.threshold = self.threshold.saturating_sub(1).max(2);
        }
        self.entries_at_check = self.entries_created;
        self.elems_at_check = self.elems_scanned;
        self.next_checkpoint = self.entries_created + self.check_interval;
    }
}

/// A trained encoder.
pub struct Parser {
    pub dict: Dictionary,
    lpm: Lpm,
}

/// Train an integer-OnPair dictionary on a flattened element stream.
///
/// * `data`/`offsets`: Arrow-style list layout; row `r` is
///   `data[offsets[r]..offsets[r + 1]]`, each entry a base element id in
///   `0..num_distinct`.
/// * `capacity`: max total tokens (base + merged).
/// * `sample_fraction`: dynamic-threshold scan budget, as in the byte trainer.
pub fn train(
    data: &[u32],
    offsets: &[u32],
    num_distinct: u32,
    capacity: usize,
    sample_fraction: f64,
) -> Parser {
    assert!(capacity >= num_distinct as usize);
    let n = offsets.len() - 1;

    let mut dict = Dictionary {
        elems: (0..num_distinct).collect(),
        offsets: (0..=num_distinct).collect(),
    };
    let mut lpm = Lpm::with_base(num_distinct);

    let merge_capacity = capacity - num_distinct as usize;
    let total_elems = data.len();
    let mut ctrl = DynamicThreshold::new(merge_capacity.max(1), total_elems, sample_fraction);
    let mut threshold = ctrl.threshold;

    let mut freq: HashMap<u64, u8> = HashMap::new();
    let mut full = false;
    let mut exhausted = false;

    for r in 0..n {
        if full || exhausted {
            break;
        }
        let s = offsets[r] as usize;
        let e = offsets[r + 1] as usize;
        if e == s {
            continue;
        }
        let row = &data[s..e];
        let len = row.len();

        let (mut prev_id, mut prev_len) = lpm.find_longest_match(row);
        let mut pos = prev_len;
        ctrl.on_elems_scanned(prev_len);
        if ctrl.budget_exhausted() {
            break;
        }

        while pos < len {
            let (curr_id, curr_len) = lpm.find_longest_match(&row[pos..]);
            ctrl.on_elems_scanned(curr_len);
            if ctrl.budget_exhausted() {
                exhausted = true;
                break;
            }
            let pair_len = prev_len + curr_len;
            if pair_len <= MAX_TOKEN_ELEMS {
                let key = ((prev_id as u64) << 32) | curr_id as u64;
                let slot = freq.entry(key).or_insert(0);
                *slot = slot.saturating_add(1);
                if *slot >= threshold {
                    let pair = &row[pos - prev_len..pos + curr_len];
                    let new_id = lpm.insert(pair);
                    dict.elems.extend_from_slice(pair);
                    dict.offsets.push(dict.elems.len() as u32);
                    if lpm.size() >= capacity {
                        full = true;
                        break;
                    }
                    ctrl.on_entry_created();
                    threshold = ctrl.threshold;
                    freq.remove(&key);
                    prev_id = new_id;
                    prev_len = pair_len;
                    pos += curr_len;
                    continue;
                }
            }
            prev_id = curr_id;
            prev_len = curr_len;
            pos += curr_len;
        }
    }

    Parser { dict, lpm }
}

impl Parser {
    /// Encode `data`/`offsets` into a flat code stream plus per-row code
    /// offsets (a code/token may span a row boundary, so rows can't be
    /// recovered from the code stream alone).
    pub fn encode(&self, data: &[u32], offsets: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let n = offsets.len() - 1;
        let mut codes = Vec::with_capacity(data.len());
        let mut code_offsets = Vec::with_capacity(n + 1);
        code_offsets.push(0);
        for r in 0..n {
            let s = offsets[r] as usize;
            let e = offsets[r + 1] as usize;
            let mut pos = s;
            while pos < e {
                let (id, mlen) = self.lpm.find_longest_match(&data[pos..e]);
                codes.push(id);
                pos += mlen;
            }
            code_offsets.push(codes.len() as u32);
        }
        (codes, code_offsets)
    }
}
