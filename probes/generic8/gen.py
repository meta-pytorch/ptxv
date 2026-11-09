# probes/generic8 input: 256 bytes (deterministic, all byte values present).
import random
import sys

r = random.Random(8)
b = list(range(256))
r.shuffle(b)
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(bytes(b))
