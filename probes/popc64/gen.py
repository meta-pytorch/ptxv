# probes/popc64 input: 256 u64 words (0, all ones, single bits, alternating patterns, random).
import random
import struct
import sys

r = random.Random(29)
xs = [
    0,
    2**64 - 1,
    1,
    2**63,
    2**32,
    2**32 - 1,
    0x5555555555555555,
    0xAAAAAAAAAAAAAAAA,
    0xFFFFFFFF00000000,
    0x00000000FFFFFFFF,
]
xs += [1 << r.randrange(64) for _ in range(54)]
xs += [r.getrandbits(64) for _ in range(256 - len(xs))]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<256Q", *xs)
)
