# in.bin: 64 threads x 4096 bytes; byte i of row r = (i * 13 + r * 7 + 1) & 0xff.
open("in.bin", "wb").write(
    bytes((i * 13 + r * 7 + 1) & 0xFF for r in range(64) for i in range(4096))
)
