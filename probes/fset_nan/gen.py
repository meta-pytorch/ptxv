# probes/fset_nan: writes kernel.ptx and in.bin. `gen.py --check out.bin` compares a recorded
# output with a bit-level model written here, independent of ptxv.
#
# Thread t (block 1024): a, b (f32 bits), c (u32) at in + 16 t; 12 f32 words at out + 48 t,
# word i = 1.0f when shape i holds, else +0 (word 11: -1.0f when c == 0, the guard is off).
import random
import struct
import sys

N = 1024
U = 0x3FC00001  # the .f32 parameter: `FSET.BF.NAN.AND Rd, Ra, URb, PT`
IMM = 0x41A00000  # 20.0
INF = 0x7F800000
SHAPES = [
    ("a", "a", None),  # isnan(a): `FSET.BF.NAN.AND Rd, Ra, Ra, PT`
    ("b", "b", "selp"),  # the same from setp.nan.f32 + selp.f32 1.0, 0.0
    ("a", "b", None),
    ("a", "u", None),
    ("a", "imm", None),
    ("a", "inf", None),
    ("a", "zero", None),
    ("|a|", "-b", None),
    ("a", "b", "c"),  # set.nan.and.f32.f32 d, a, b, c: `..., P`
    ("b", "a", "!c"),  # `..., !P`
    ("a", "a", "c"),
    ("b", "a", "guard"),  # executed only when c != 0: `@P FSET.BF.NAN.AND`
]


def ptx():
    o = [
        "// probes/fset_nan (written by gen.py, which documents the records): the unordered float",
        "// compare as a 1.0f / 0 value, SASS `FSET.BF.NAN.AND Rd, a, b, Pc | !Pc | PT` with b a",
        "// register (also a itself: isnan), a uniform register, an immediate or RZ, |R| / -R",
        "// operands and a guard, from set.nan[.and].f32.f32 and from setp.nan.f32 + selp.f32.",
        ".version 8.8",
        ".target sm_103a",
        ".address_size 64",
        ".visible .entry fset_nan(.param .u64 pin, .param .u64 pout, .param .f32 pu)",
        "{",
        " .reg .b32 %r<4>; .reg .b64 %rd<8>; .reg .f32 %f<12>; .reg .pred %p<4>;",
        " ld.param.u64 %rd1,[pin]; ld.param.u64 %rd2,[pout]; ld.param.f32 %f3,[pu];",
        " cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;",
        " mov.u32 %r1,%tid.x; mul.wide.u32 %rd3,%r1,16; add.s64 %rd4,%rd1,%rd3;",
        " mul.wide.u32 %rd5,%r1,48; add.s64 %rd6,%rd2,%rd5;",
        " ld.global.f32 %f1,[%rd4]; ld.global.f32 %f2,[%rd4+4]; ld.global.u32 %r2,[%rd4+8];",
        " setp.ne.u32 %p1,%r2,0;",
        " abs.f32 %f4,%f1; neg.f32 %f5,%f2;",
    ]
    reg = {"a": "%f1", "b": "%f2", "u": "%f3", "|a|": "%f4", "-b": "%f5"}
    reg.update({"imm": "0f%08X" % IMM, "inf": "0f%08X" % INF, "zero": "0f00000000"})
    for i, (x, y, how) in enumerate(SHAPES):
        if how == "selp":
            o.append(" setp.nan.f32 %%p2,%s,%s;" % (reg[x], reg[y]))
            o.append(" selp.f32 %f6,0f3F800000,0f00000000,%p2;")
        elif how == "guard":
            o.append(" mov.f32 %f6,0fBF800000;")
            o.append(" @!%p1 bra G;")
            o.append(" set.nan.f32.f32 %%f6,%s,%s;" % (reg[x], reg[y]))
            o.append("G:")
        elif how:
            o.append(
                " set.nan.and.f32.f32 %%f6,%s,%s,%s;"
                % (reg[x], reg[y], how.replace("c", "%p1"))
            )
        else:
            o.append(" set.nan.f32.f32 %%f6,%s,%s;" % (reg[x], reg[y]))
        o.append(" st.global.f32 [%%rd6+%d],%%f6;" % (4 * i))
    return "\n".join(o + [" ret;", "}"]) + "\n"


def edges():
    e = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000]
    e += [IMM, IMM + 1, IMM - 1, U, 0x7F7FFFFF, 0xFF7FFFFF, INF, 0xFF800000]
    # Quiet and signalling NaNs of either sign: smallest and largest payloads.
    e += [0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFA5A5A5]
    return e


def records():
    r = random.Random(2026)
    e = edges()
    rec = [(a, b, c) for a in e for b in e for c in (0, 1)]
    while len(rec) < N:
        a, b = r.getrandbits(32), r.getrandbits(32)
        if r.getrandbits(1):
            a |= INF  # an infinity or a NaN
        rec.append((a, r.choice([b, b | INF, a]), r.getrandbits(1) * r.getrandbits(32)))
    return rec


def isnan(x):
    return x & 0x7FFFFFFF > INF


def model(a, b, c):
    val = {"a": a, "b": b, "u": U, "|a|": a & 0x7FFFFFFF, "-b": b ^ 0x80000000}
    val.update({"imm": IMM, "inf": INF, "zero": 0})
    w = []
    for x, y, how in SHAPES:
        t = isnan(val[x]) or isnan(val[y])
        if how == "guard" and c == 0:
            w.append(0xBF800000)
            continue
        if how in ("c", "!c"):
            t = t and (c != 0) == (how == "c")
        w.append(0x3F800000 if t else 0)
    return w


if __name__ == "__main__":
    rec = records()
    n = len(SHAPES)
    assert len(rec) == N
    if sys.argv[1:2] == ["--check"]:
        got = struct.unpack("<%dI" % (n * N), open(sys.argv[2], "rb").read())
        bad = sum(list(got[n * t : n * t + n]) != model(*rec[t]) for t in range(N))
        nan = sum(isnan(a) or isnan(b) for a, b, _ in rec)
        print(
            "%d records (%d with a NaN operand), %d shapes, words %s: %d mismatching records"
            % (N, nan, n, sorted("%#x" % v for v in set(got)), bad)
        )
        sys.exit(bad != 0)
    open("kernel.ptx", "w").write(ptx())
    open("in.bin", "wb").write(
        b"".join(struct.pack("<IIII", a, b, c, 0) for a, b, c in rec)
    )
