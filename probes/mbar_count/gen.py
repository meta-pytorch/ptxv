# probes/mbar_count input: 32 u32 (deterministic).
import struct
import sys

xs = [(0x9E3779B9 * (i + 7)) & 0xFFFFFFFF for i in range(32)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<32I", *xs)
)
