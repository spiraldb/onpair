#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Pack the boolean decode masks (maskbool.lst) into little-endian u8 bytes.

Each mask is a list(bool); we group 8 bools into one byte (LSB-first, matching
the `_le` suffix on closure_local_decode_mask_le) and emit one row of decimal
byte elements. This gives OnPair a ~256-symbol alphabet so a 16-element token
spans 128 bits instead of 16 -- see the README "list(bool)" section.
"""
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    src, dst = DATA / "maskbool.lst", DATA / "maskbyte.lst"
    distinct, rows, nbytes = set(), 0, 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            bits = [1 if x == "1" else 0 for x in line.split("\t")]
            out = []
            for b in range(0, len(bits), 8):
                val = sum(bit << i for i, bit in enumerate(bits[b : b + 8]))
                out.append(val)
                distinct.add(val)
            fout.write("\t".join(map(str, out)))
            fout.write("\n")
            rows += 1
            nbytes += len(out)
    print(f"rows={rows} bytes={nbytes} avg={nbytes/rows:.1f} distinct={len(distinct)}: {sorted(distinct)}")


if __name__ == "__main__":
    main()
