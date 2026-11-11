# probes/bf16_sink input: 256 words of bf16 pairs (x), then 128 words of bf16 pairs (A),
# random finite bf16 values in [-4, 4] (deterministic).
import random
import struct
import sys

r = random.Random(23)


def bf():
    f = struct.unpack("<I", struct.pack("<f", r.uniform(-4, 4)))[0]
    return f >> 16


ws = [bf() | bf() << 16 for _ in range(256 + 128)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<384I", *ws)
)
