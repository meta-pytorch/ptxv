# probes/redux_mask input: 64 u32 with mixed high bits (deterministic).
import struct
import sys

xs = [(0x9E3779B9 * (i + 1)) & 0xFFFFFFFF for i in range(64)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<64I", *xs)
)
