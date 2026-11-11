# probes/fmad_f16 input: 256 threads x 5 groups x (a, b, c, d) f16, deterministic. c and d cancel
# a b to +- 1 ulp in 3 of 4 groups; random halves in the 4th.
import random
import struct

r = random.Random(37)


def h(v):
    return struct.unpack("<H", struct.pack("<e", v))[0]


out = []
for t in range(256 * 5):
    if t % 4 == 3:
        out += [r.getrandbits(16) for _ in range(4)]
        continue
    a = struct.unpack("<e", struct.pack("<e", r.choice([1, -1]) * (1 + r.random())))[0]
    b = struct.unpack("<e", struct.pack("<e", r.choice([1, -1]) * (1 + r.random())))[0]
    out += [h(a), h(b), h(-a * b) ^ r.choice([0, 1]), h(-a * b) ^ r.choice([0, 1, 2])]
open("in.bin", "wb").write(struct.pack("<%dH" % len(out), *out))
