# in.bin: 1024 threads x 8 f32: random normals scaled by 2^k (k in -20..20), with edge values
# (0, -0, inf, nan, subnormals, max) mixed in. Seed 5.
import random
import struct

random.seed(5)
edge = [
    0.0,
    -0.0,
    float("inf"),
    -float("inf"),
    float("nan"),
    1e-40,
    -3e-39,
    3.4e38,
    -3.4e38,
    1.0,
]
v = []
for i in range(1024 * 8):
    v.append(
        random.choice(edge)
        if random.random() < 0.08
        else random.gauss(0, 1) * 2.0 ** random.randint(-20, 20)
    )
open("in.bin", "wb").write(struct.pack("<%df" % len(v), *v))
