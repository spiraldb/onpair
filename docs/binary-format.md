# OnPair binary format

The decode-side state of a compressed column is three byte buffers — the
**dictionary bytes**, the **dictionary offsets**, and the **codes** — plus the
scalar `bits` (code width, `9..=16`). This document specifies their binary
layout and the invariants a decoder relies on.

Decoding is a gather-copy: the code stream is a sequence of dictionary indices;
each index names a token (a short byte string), and the decoded output is the
tokens concatenated in code order.

## Conventions

- All multi-byte integers are **little-endian**.
- `MAX_TOKEN_SIZE = 16` — the maximum token length in bytes and the decoder's
  fixed read width.
- `N` — the number of dictionary tokens. `N <= 2^bits`.
- `M` — the number of codes (decoded values = number of tokens emitted).

## 1. Dictionary offsets — `dict_offsets`

`N + 1` little-endian **`u32`** values delimiting the tokens within
`dict_bytes`:

```
dict_offsets: [ o0=0, o1, o2, …, oN ]          (N+1 × u32 LE)
              token i  =  dict_bytes[ o_i .. o_{i+1} )
```

Invariants:

- `o0 == 0`.
- Strictly increasing: `o_i < o_{i+1}` for all `i` (every token is non-empty).
- Each token length `o_{i+1} - o_i` is in `1 ..= MAX_TOKEN_SIZE`.
- `oN` is the **logical length** of the dictionary (sum of all token lengths).

`N = dict_offsets.len() - 1`. An empty dictionary is `[0]` (`N == 0`).

## 2. Dictionary bytes — `dict_bytes`

The `N` token byte strings concatenated in index order, with no separators,
followed by **trailing decoder padding**:

```
dict_bytes: [ token0 ‖ token1 ‖ … ‖ token_{N-1} ‖ <padding> ]
            └─────────── oN logical bytes ───────────┘
```

The decoder reads a fixed `MAX_TOKEN_SIZE` bytes starting at every token offset
(then slices to the token's true length — the branchless "fat read, advance by
`len`" pattern), so the buffer must stay readable `MAX_TOKEN_SIZE` past the
**highest** offset, which is the last token's:

```
dict_bytes.len()  >=  o_{N-1} + MAX_TOKEN_SIZE
```

Equivalently, the slack past the logical end `oN` is **variable**:

```
padding = MAX_TOKEN_SIZE - len(last token)        // 0 .. MAX_TOKEN_SIZE-1
```

— `0` when the last token is a full `MAX_TOKEN_SIZE` bytes, up to
`MAX_TOKEN_SIZE - 1` (15) when it is a single byte. (For a width-`W` token cap
the range is `0 ..= W-1`.) Padding byte values are unspecified; the decoder
never uses them as token data. A writer may emit exactly this minimum or
over-allocate to the `MAX_TOKEN_SIZE - 1` worst case (what `compress` does);
both are valid.

## 3. Codes

`M` values, each a dictionary index in `[0, N)`. Every code fits in `bits` bits
because `N <= 2^bits`. 

**Packed form**: codes are bit-packed **LSB-first at `bits` bits each** into a
stream of little-endian `u64` words. A code whose bits straddle a 64-bit word
boundary is split — low bits in the current word, remaining high bits open the
next.

```
word k holds codes packed LSB-first:  bit position of code j = j * bits
packed size = ceil(M * bits / 8) bytes
```

A word-at-a-time unpacker over-reads up to 7 bytes when it extracts the last
code (its `u64` load starts near the end of the buffer). This is a reader
concern, not a format one: the unpacker reads the bulk fast with full-width word
loads and switches to an exact, masked tail for the final few codes whose load
would cross `ceil(M * bits / 8)` — the same fast-region-plus-exact-tail split
the decode loop uses (§Decode, `fat::decode_loop`). No padding word is emitted.

Reference onpair appends one zero `u64` after the packed codes (its unpacker
over-reads unconditionally). OnPair neither writes nor requires it; a reader
ingesting such a buffer ignores any bytes past `ceil(M * bits / 8)`.

**In-memory form** (this crate): `codes` is materialized as a `[u16]`, one
element per code, unpacked — the `bits`-wide packing is applied only when
serializing. The two are equivalent under `bits`.

Invariant: every code `< N` (indexes a real token).

## Related: row offsets

A column also carries `code_offsets`, `R + 1` offsets into the code stream that
delimit the `R` input rows (row `r`'s codes are `codes[ o_r .. o_{r+1} )`).
Their integer width matches the input offset type (`u32`/`u64`). The compressor
emits them because a token may span a row boundary, so the row structure cannot
otherwise be recovered. They index the codes and are not part of the code/dict
encoding.

## Decode

```
out = []
for c in codes:                       # in order
    out.extend( dict_bytes[ dict_offsets[c] .. dict_offsets[c+1] ) )
```

The fast path reads a fixed `MAX_TOKEN_SIZE` bytes from `dict_offsets[c]` and
advances `out` by the true token length; the dictionary padding (§2) makes that
read in bounds.

## Validation

A decoder fed an untrusted/deserialized column should check, in `O(N)` (plus
`O(M)` for the codes), before decoding:

1. `dict_offsets` strictly increasing, every token length `<= MAX_TOKEN_SIZE`;
2. `dict_bytes.len() >= o_{N-1} + MAX_TOKEN_SIZE` (the padding above);
3. every code `< N`.

In this crate these are `Parts::validate_dictionary` (1–2, dictionary only) and
`Parts::validate` (1–3). The safe decoders (`decompress`, `decompress_into`)
enforce 1–2 once up front and check 3 per code in the decode loop.
