# OnPair in-memory format

This document specifies the **common in-memory representation** of an
OnPair-compressed column: the layout of its constituent buffers, the invariants
over them, and the language-neutral structures used to address them. It is the
contract that independent implementations agree on so that a column produced by
one is intelligible to another.

**Scope.** This is the *plain* interchange form — the representation
implementations share to exchange data. It fixes the buffers, their invariants,
and the in-memory addressing of a column whose codes are plain `u16`. Denser
*compressed* encodings of the codes (bit-packing or anything else)
and the on-disk serialization of a column are **out of scope**: each
implementation **MAY** use them internally, but they are not part of this shared
contract. What crosses between implementations is always the plain form defined
here.

## Overview

OnPair compresses a sequence of byte strings by replacing recurring substrings
("tokens") with fixed-width integer **codes** into a **dictionary**. Decoding is
a gather-copy: each code names a dictionary token, and the output is those
tokens concatenated in code order.

The representation is defined in two layers:

1. **Decodable data** (§3) — a dictionary plus a code stream. Decoding it
   reconstructs the original bytes as a single **concatenated payload**: the
   full content, with no boundaries between the records that produced it. This
   layer is self-contained and carries no notion of rows.

2. **Row partitioning** (§4) — an optional layer that records where each row
   begins and ends within the code stream, recovering the **delimited records**
   from the concatenated payload. It indexes into the code stream; the code
   stream does not depend on it.

The layering is one-directional: the decodable data is meaningful on its own,
and the row layer is meaningful only in terms of the data it partitions. A
consumer that needs only the concatenated payload uses the first layer; a consumer that
needs the original records back uses both.

## 1. Conventions

- Multi-byte **integer** fields (offsets and `u16` codes) are
  **little-endian**. Token bytes and dictionary bytes are raw byte strings, to
  which byte order does not apply. Supported hosts are little-endian, so this
  fixed byte order *is* the host's native representation — fixed-width buffers
  can be indexed directly as arrays of their element type with no per-element
  byte-swapping (§5).
- `MAX_TOKEN_SIZE = 16` — the maximum token length in bytes, and the fixed read
  width a decoder uses per token (§3.1).
- `N` — the number of dictionary tokens, `256 <= N <= 2^16` (§3).
- `M` — the number of codes (= number of tokens emitted on decode).
- `R` — the number of rows, when the row layer is present (§4).
- **Element count vs. byte length.** Sequence sizes are stated as *logical
  element counts*, independent of how the elements are encoded; raw byte buffers
  are stated as *byte lengths*. The distinction is maintained deliberately
  throughout.
- Normative requirements use **MUST**, **MUST NOT**, and **MAY**.

## 2. Offsets

Token and row boundaries are sequences of monotonically non-decreasing unsigned
integers, stored as **contiguous fixed-width little-endian integers** and indexed
directly. The width is fixed per field — `u32` for dictionary offsets (§3.2),
`u64` for row offsets (§4) — chosen generously rather than minimally, since this
is an interchange form (Scope, above) where a few bytes per offset do not matter
and a single fixed width keeps every reader trivial.

Implementations that want a more compact or randomly-addressable boundary
representation use it *outside* this format, alongside the decodable data (§3),
not as an alternative encoding within it.

## 3. Decodable data

The self-contained unit that reconstructs the decompressed output. It comprises
a dictionary (§3.1, §3.2) and a code stream (§3.3).

**Completeness.** A dictionary **MUST** contain all 256 single-byte tokens — one
length-1 token for every byte value `0x00`–`0xFF` — at arbitrary indices. This
guarantees that *any* byte string is encodable (a byte with no longer matching
token falls back to its single-byte token), and it underpins compressed-domain
search: a query string can always be encoded into codes and matched against the
stream (e.g. equality by comparing code sequences) because encoding can never
fail. Consequently `N >= 256` for every dictionary; there is no empty
dictionary.

**Dictionary ordering.** A dictionary **MAY** have its tokens arranged in
ascending bytewise-lexicographic order (token `i` is `<=` token `i+1` under
unsigned byte comparison). Sortedness affects only the order of the tokens, not
the structure of the buffers or how they are decoded; the `is_sorted` flag (§5)
asserts that order rather than encoding it. It is advertised so that a consumer
may use order-dependent operations — binary search over tokens, prefix-range
queries, compressed pattern matching — without itself having to verify the
ordering. When the flag is set, the ordering invariant **MUST** hold; a producer
**MUST NOT** set it otherwise. When it is clear, no ordering is implied and
order-dependent operations are unavailable.

### 3.1 Dictionary bytes

The `N` token byte strings, concatenated in index order with no separators,
followed by trailing **read-padding**:

```
dict_bytes:  token_0 ‖ token_1 ‖ … ‖ token_{N-1} ‖ <padding>
             └────────── L logical bytes ─────────┘
                                                   ↑
                                          o_N (first padding byte)
```

`L` is the sum of all token lengths — the extent of the real token content,
ending exactly where padding begins. It equals the final dictionary offset `o_N`
(§3.2), which therefore points at the first padding byte (or one past the buffer
if there is no padding).

**Read-padding (normative).** A decoder reads a fixed `MAX_TOKEN_SIZE` bytes
beginning at each token's offset and consumes only the token's true length — a
branchless "wide read, advance by length" pattern that avoids a per-token
branch. The buffer therefore **MUST** remain readable for `MAX_TOKEN_SIZE` bytes
past the *highest* token offset (that of the last token):

```
readable length of dict_bytes  >=  offset(last token) + MAX_TOKEN_SIZE
```

The slack past the logical end `L` is `MAX_TOKEN_SIZE - length(last token)`:
zero when the last token is a full `MAX_TOKEN_SIZE` bytes, up to
`MAX_TOKEN_SIZE - 1` when it is a single byte. A producer **MAY** emit exactly
this minimum or over-allocate to the worst case; both conform. Padding byte
**values are unspecified**; a decoder never consumes them as token data.

This requirement is a property of token decoding and is independent of how the
code stream (§3.3) is encoded.

### 3.2 Dictionary offsets

`N + 1` little-endian **`u32`** offsets (§2) delimiting the tokens within
`dict_bytes`. Each adjacent pair `(o_i, o_{i+1})` brackets token `i`:

```
token_i  =  dict_bytes[ o_i .. o_{i+1} )

example:  dict_bytes:    c  a  t  d  o  g  f  i  s  h
                         │        │        │           │
          offset:        0        3        6           10

          dict_offsets = [ 0, 3, 6, 10 ]      (N = 3, o_N = 10 = L)
              token_0 = dict_bytes[0 .. 3)  = "cat"
              token_1 = dict_bytes[3 .. 6)  = "dog"
              token_2 = dict_bytes[6 .. 10) = "fish"
```

(The example is schematic; a conformant dictionary has `N >= 256` by
completeness, §3.)

Invariants:

- `o_0 == 0`.
- **Strictly increasing**: `o_i < o_{i+1}` for all `i` — every token is
  non-empty.
- Each token length `o_{i+1} - o_i` lies in `1 ..= MAX_TOKEN_SIZE`.
- `o_N == L`, the logical byte length of the dictionary.
- Element count is `N + 1`, with `N >= 256` (completeness, §3).

**Width.** Dictionary offsets are `u32`. The largest offset `o_N <= N ·
MAX_TOKEN_SIZE` and `N <= 2^16`, so `o_N` always fits in 32 bits; `u32` covers
every conformant dictionary.

### 3.3 Code stream

`M` codes, each a dictionary index in `[0, N)`, stored as `M` contiguous
little-endian `u16` — one per code. The code width is fixed: every code is a
`u16`, regardless of dictionary size. No tag, no alternative encoding, and no
trailing slack (`u16` elements are independently addressable and incur no
over-read).

This is the plain interchange form (Scope, above): the representation all
implementations share for exchange. An implementation **MAY** use a denser
internal encoding of the codes — bit-packing or anything else —
within its own code, but such encodings are outside this specification and never
cross between implementations; the shared form is always plain `u16`.

Invariants:

- Every code is `< N`.
- Element count is `M`.

### 3.4 Decode

```
output = []
for c in code stream (in order):
    output ‖= dict_bytes[ dict_offsets[c] .. dict_offsets[c+1] )
```

The fast path reads a fixed `MAX_TOKEN_SIZE` bytes from `dict_offsets[c]` and
advances the output cursor by the token's true length; the read-padding (§3.1)
keeps that read in bounds.

Decode consults only the dictionary and the code stream. It does not reference
the row layer (§4).

## 4. Row partitioning (optional)

An OnPair column **MAY** add a row layer over the decodable data: `R + 1`
little-endian **`u64`** offsets (§2) **into the code stream** (code positions, not
byte positions) that delimit `R` rows. Each adjacent pair brackets one row:

```
index:          0     1     2     3   …    R
row_offsets:  [ r_0   r_1   r_2   r_3 …  r_R ]

row k's codes  =  code_stream[ r_k .. r_{k+1} )

e.g. M = 9 codes, offsets = [0, 4, 4, 9]:
        row 0 = codes[0..4)   (4 codes)
        row 1 = codes[4..4)   (empty row)
        row 2 = codes[4..9)   (5 codes)      (R = 3, r_R = 9 = M)
```

The row layer enables random access to individual rows and per-row operations.
Because the code stream is a flat concatenation with no in-band record
delimiters, row structure cannot be recovered from the codes alone; the row
layer carries it explicitly. The decodable data is fully usable without it.

Invariants when present:

- `r_0 == 0` and `r_R == M` (the final offset is the total code count).
- **Non-decreasing**: `r_k <= r_{k+1}`. Unlike dictionary offsets these need not
  strictly increase — an empty row is a legal `r_k == r_{k+1}`.
- Element count is `R + 1`.

**Width.** Row offsets are `u64`, which indexes any code stream (`r_R == M`,
`M <= 2^64`).

## 5. In-memory layout

The structures below are an in-memory addressing convenience, built locally to
point at buffers the consumer already holds; what crosses between implementations
is the buffer *contents* (§2–§4), not these structures.

They are presented in C for definiteness. They are **non-owning views**: each
pointer borrows memory owned elsewhere and valid for that owner's lifetime; the
structures allocate nothing and free nothing, and pointers are read-only.

```c
#include <stdint.h>

// The code stream (§3.3): M plain little-endian u16 codes.
typedef struct OnPairCodes {
    const uint16_t* data;        // M codes, each a dictionary index < N
    uint64_t        count;       // M
} OnPairCodes;

// The dictionary (§3.1, §3.2): token bytes plus their offsets.
typedef struct OnPairDictionary {
    const uint8_t*  dict_bytes;      // §3.1; read-padded
    uint64_t        dict_bytes_len;  // readable byte extent (bounds-check target)
    const uint32_t* dict_offsets;    // §3.2; N+1 u32 offsets into dict_bytes
    uint64_t        dict_offsets_len;// N + 1
    uint8_t         is_sorted;       // §3; 1 if tokens are bytewise-lexicographic, else 0
    uint8_t         _reserved[7];    // MUST be zero
} OnPairDictionary;

// Decodable data (§3): everything needed to reconstruct the output.
typedef struct OnPairData {
    OnPairDictionary dict;         // §3.1, §3.2
    OnPairCodes      codes;        // §3.3
} OnPairData;

// A row-partitioned column (§4): decodable data plus a row layer.
typedef struct OnPairColumn {
    OnPairData      data;          // §3
    const uint64_t* row_offsets;   // §4; R+1 u64 offsets into the code stream
    uint64_t        row_offsets_len; // R + 1 (0 if no row layer)
} OnPairColumn;
```

`OnPairData` yields the decompressed bytes as a single concatenated payload: it
recovers the token content but not the boundaries between the original strings,
so it supports bulk decompression but not reconstruction of individual records.
`OnPairColumn` adds the row layer that delimits those records, enabling
per-string reconstruction and random access. A column with no row layer sets
`row_offsets == NULL` and `row_offsets_len == 0`.

### Layout rules (normative)

- **Reserved bytes** (in `OnPairDictionary`) pad the `is_sorted` byte so the
  surrounding pointer and 64-bit fields fall on their natural 8-byte boundaries
  (on a 64-bit host), making the in-memory layout explicit rather than left to
  implicit compiler padding. They **MUST** be written zero and **MAY** be
  validated as zero; a future revision may assign them meaning.
- **Pointee alignment.** Each pointer **MUST** be aligned to its element width:
  2 bytes for the `u16` code stream, 4 for the `u32` dictionary offsets, 8 for
  the `u64` row offsets. `dict_bytes` has no alignment requirement.
- **Endianness.** Integers within the addressed buffers are little-endian (§1).
- **`dict_bytes_len`** is the readable length including read-padding, and **MUST**
  satisfy the §3.1 bound. (The logical length `L` is recoverable as the last
  dictionary offset; `dict_bytes_len` records the padded extent the §3.1
  guarantee refers to.)
- **`is_sorted`** is `0` or `1` (§3). When `1`, the dictionary-ordering invariant
  **MUST** hold. It is a property assertion only; it does not affect any layout.
- **Lifetime.** All pointers borrow memory owned elsewhere. A consumer **MUST
  NOT** retain a view beyond the owner's lifetime, nor free through it.

### Accessing a buffer

Every buffer has a fixed, known element type (`u16` codes, `u32` dictionary
offsets, `u64` row offsets), so a consumer reads each through a typed pointer
with no per-buffer or per-element dispatch. Because supported hosts are
little-endian (§1), the buffers' fixed little-endian contents are already in
native form, so all of them are directly indexable as arrays of their element
type with no byte-swapping. The inner loop is monomorphic.

## 6. Validation

Before decoding untrusted or deserialized input, a consumer should verify, in
`O(N)` over the dictionary plus `O(M)` over the codes:

1. All reserved bytes are zero (§5).
2. Dictionary offsets: `o_0 == 0`, strictly increasing, each token length
   `<= MAX_TOKEN_SIZE` (§3.2).
3. Completeness: all 256 single-byte tokens are present, so `N >= 256` (§3).
   This can be folded into the dictionary scan of check 2 by recording which
   byte values appear as length-1 tokens.
4. `dict_bytes` is readable for `MAX_TOKEN_SIZE` past the last token offset
   (§3.1).
5. Every code is `< N` (§3.3).
6. If a row layer is present: `r_0 == 0`, non-decreasing, and `r_R == M` (§4).
7. If `is_sorted` is set: tokens are non-decreasing in bytewise-lexicographic
   order (§3). This check is optional — a consumer that does not use
   order-dependent operations may skip it.

All checks except 5 are up-front and run once (checks 1–4 and 6, plus the
optional 7); check 5 is the per-code `O(M)` bound, typically folded into the
decode loop.
