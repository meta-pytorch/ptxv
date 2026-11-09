# in.bin: 512 words. Each bf16 half is random bits (1/4), a special value (1/8: +-0, +-inf,
# NaN, smallest subnormal, largest finite) or a random normal near 1 (rest), seed 6.
import random
import struct

random.seed(6)
SPEC = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x7F7F, 0x8001]


def h():
    r = random.random()
    if r < 0.25:
        return random.getrandbits(16)
    if r < 0.375:
        return random.choice(SPEC)
    return (random.getrandbits(1) << 15) | random.randint(0x3E00, 0x40FF)


open("in.bin", "wb").write(struct.pack("<512I", *[h() | h() << 16 for _ in range(512)]))
