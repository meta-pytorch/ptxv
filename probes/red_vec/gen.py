# probes/red_vec input: 512 f32, small integers and halves (+-), so sums are exact in any order.
import random
import struct
import sys

r = random.Random(17)
xs = [r.randint(-64, 64) / 2.0 for _ in range(512)]
open(sys.argv[1] if len(sys.argv) > 1 else "in.bin", "wb").write(
    struct.pack("<512f", *xs)
)
