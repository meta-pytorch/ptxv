# in.bin: 128 x 16 bytes, byte i = (i * 37 + 11) & 0xff.
open("in.bin", "wb").write(bytes((i * 37 + 11) & 0xFF for i in range(128 * 16)))
