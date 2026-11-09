# probes/clc3d: check.py OUT.bin (a GB300 output, `PTXV_HW_DUMP`): every cancel names a CTA in
# the grid that never ran, at (raw0, raw1 & 0xffff, raw1 >> 16), and get_first_ctaid::y / ::z
# agree with that decoding.
import struct
import sys

w = struct.unpack("<65536I", open(sys.argv[1], "rb").read())
ran = set()
cancels = []
for c in range(8192):
    r = w[8 * c : 8 * c + 8]
    if r[0]:
        ran.add(c)
    if r[1]:
        x, y, z = r[2], r[3] & 0xFFFF, r[3] >> 16
        assert (r[6], r[7]) == (y, z), (c, r)
        assert r[4] == 1 and r[5] == 0, (c, r)
        cancels.append(x + 16 * (y + 4 * z))
assert len(cancels) == len(set(cancels))
assert not set(cancels) & ran
assert len(ran) + len(cancels) == 8192, (len(ran), len(cancels))
zs = sorted({v // 64 for v in cancels})
print(
    f"ran {len(ran)}, cancelled {len(cancels)}, victims with z > 0: {sum(1 for v in cancels if v >= 64)}, z range {zs[0] if zs else None}..{zs[-1] if zs else None}"
)
