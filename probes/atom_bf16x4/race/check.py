"""check.py RUN.bin...: per run, the reader observations torn within a 4-byte half and across
the two halves of a slot and the reads taken mid-sequence (out = {torn half, torn slot, mid, 256} per slot)."""

import struct
import sys

for f in sys.argv[1:]:
    d = open(f, "rb").read()
    w = struct.unpack("<%dI" % (len(d) // 4), d)
    half, slot, mid = sum(w[0::4]), sum(w[1::4]), sum(w[2::4])
    print(
        f"{f}: {len(w) // 4} slots x 256 reads; mid-sequence {mid}; torn half {half}; torn slot {slot}"
    )
