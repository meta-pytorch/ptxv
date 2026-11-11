# probes/dsetp_ltu_le: writes kernel.ptx and in.bin. `gen.py --check out.bin` compares a
# recorded output with an IEEE model written here on Python floats, independent of ptxv.
#
# Record g = 1024 * ctaid.x + tid.x (grid 2 x block 1024): a, b (f64), c, k (u32) at in + 24 g.
# Output words at out + 24 g: w0 | w1 << 32 has bit i set when shape i holds (shapes()); w2,
# w3, w4 are written by one of the two arms of a branch on, in turn, (a <u LO) || (a >u HI),
# (a <= 1.0) && (k != 7) and (a <u U) && (k != 7); w5 is not written.
import math
import random
import struct
import sys

N = 2048
M32 = 0xFFFFFFFF
U = 0x3FB999999999999A  # 0.1: the .f64 parameter and a constant ptxas keeps in a uniform pair
LO, HI = 0x3F50624DD2F1A9FC, 0x3FEFF7CED916872B  # 0.001, 0.999: uniform pairs as well
ONE = 0x3FF0000000000000  # immediates have a zero low word: 1.0 and IMMS
# +Inf, 0, -2, 2^-1023
IMMS = [0x7FF0000000000000, 0, 0xC000000000000000, 0x0008000000000000]


def shapes():
    """(x, y, cmp, bop, pred) of each output bit; pred "c" is c != 0, "!c" its negation."""
    s = []
    for cmp in ("le", "ltu"):
        for y in ("b", "u", "one"):
            for bop in ("and", "or", "xor"):
                for pred in ("c", "!c"):
                    s.append(("a", y, cmp, bop, pred))
    for i in range(len(IMMS)):
        s.append(("a", "imm%d" % i, "le", "and", "c"))
        s.append(("a", "imm%d" % i, "ltu", "or", "c"))
    for x, y, cmp in (
        ("|a|", "b", "le"),
        ("-a", "b", "ltu"),
        ("a", "|b|", "ltu"),
        ("a", "-b", "le"),
        ("-|a|", "u", "le"),
        ("|a|", "imm0", "ltu"),
    ):
        s.append((x, y, cmp, "and", "c"))
        s.append((x, y, cmp, "xor", "!c"))
    s.append(("a", "a", "le", "and", "c"))
    s.append(("a", "a", "ltu", "or", "c"))
    return s


def arm(init, base, c, k):
    """The value one arm of a branch stores: 12 integer operations, so that ptxas keeps a
    real branch (`@Pk BRA P`) instead of a predicated store."""
    r = init
    for i in range(4):
        r ^= (c + base + i) * k & M32
    return r


def ptx_arm(init, base, off):
    o = [" mov.u32 %%r6,%d;" % init]
    for i in range(4):
        o.append(
            " add.u32 %%r7,%%r3,%d; mul.lo.u32 %%r7,%%r7,%%r4; xor.b32 %%r6,%%r6,%%r7;"
            % (base + i)
        )
    return o + [" st.global.u32 [%%rd6+%d],%%r6;" % off]


def ptx_branch(n, cond, off):
    o = cond + [" @%%p5 bra T%d;" % n] + ptx_arm(2 * n + 2, 40 + 16 * n, off)
    return (
        o
        + [" bra.uni J%d;" % n, "T%d:" % n]
        + ptx_arm(2 * n + 1, 32 + 16 * n, off)
        + ["J%d:" % n]
    )


def ptx():
    o = [
        "// probes/dsetp_ltu_le (written by gen.py, which documents the records): setp.le.f64 and",
        "// setp.ltu.f64 folded by ptxas with and.pred / or.pred / xor.pred into",
        "// DSETP.{LE,LTU}.{AND,OR,XOR} Pd, PT, a, b, Pc | !Pc | PT, with b a register pair, a",
        "// uniform pair (the .f64 parameter, constants with low bits), an immediate or RZ, and",
        "// R, |R|, -R, -|R| operands. The three branches at the end are the shapes of captured",
        "// kernels: a range check `DSETP.GTU.AND P, PT, R, UR, PT; DSETP.LTU.OR P, PT, R, UR, P`",
        "// and `DSETP.LE.AND P, PT, R, 1, PT` (`.LTU ... UR, PT`) feeding `@Pk BRA P`.",
        ".version 8.8",
        ".target sm_103a",
        ".address_size 64",
        ".visible .entry dsetp_ltu_le(.param .u64 pin, .param .u64 pout, .param .f64 pu)",
        "{",
        " .reg .b32 %r<12>; .reg .b64 %rd<8>; .reg .f64 %fd<12>; .reg .pred %p<8>;",
        " ld.param.u64 %rd1,[pin]; ld.param.u64 %rd2,[pout]; ld.param.f64 %fd3,[pu];",
        " cvta.to.global.u64 %rd1,%rd1; cvta.to.global.u64 %rd2,%rd2;",
        " mov.u32 %r1,%ctaid.x; shl.b32 %r1,%r1,10; mov.u32 %r2,%tid.x; add.u32 %r1,%r1,%r2;",
        " mul.wide.u32 %rd3,%r1,24; add.s64 %rd4,%rd1,%rd3; add.s64 %rd6,%rd2,%rd3;",
        " ld.global.f64 %fd1,[%rd4]; ld.global.f64 %fd2,[%rd4+8];",
        " ld.global.u32 %r3,[%rd4+16]; ld.global.u32 %r4,[%rd4+20];",
        " setp.ne.u32 %p1,%r3,0; not.pred %p2,%p1;",
        " abs.f64 %fd4,%fd1; neg.f64 %fd5,%fd1; neg.f64 %fd6,%fd4;",
        " abs.f64 %fd7,%fd2; neg.f64 %fd8,%fd2;",
        " mov.u32 %r8,0; mov.u32 %r9,0;",
    ]
    reg = {
        "a": "%fd1",
        "b": "%fd2",
        "u": "%fd3",
        "|a|": "%fd4",
        "-a": "%fd5",
        "-|a|": "%fd6",
    }
    reg.update({"|b|": "%fd7", "-b": "%fd8", "one": "0d%016X" % ONE})
    reg.update({"imm%d" % i: "0d%016X" % v for i, v in enumerate(IMMS)})
    for i, (x, y, cmp, bop, pred) in enumerate(shapes()):
        w = 8 + i // 32
        o.append(" setp.%s.f64 %%p3,%s,%s;" % (cmp, reg[x], reg[y]))
        o.append(" %s.pred %%p5,%%p3,%s;" % (bop, "%p1" if pred == "c" else "%p2"))
        o.append(
            " selp.u32 %%r5,0x%x,0,%%p5; or.b32 %%r%d,%%r%d,%%r5;" % (1 << i % 32, w, w)
        )
    o.append(" st.global.u32 [%rd6],%r8; st.global.u32 [%rd6+4],%r9;")
    k7 = [" setp.ne.u32 %p4,%r4,7;", " and.pred %p5,%p3,%p4;"]
    o += ptx_branch(
        0,
        [
            " setp.ltu.f64 %%p3,%%fd1,0d%016X;" % LO,
            " setp.gtu.f64 %%p4,%%fd1,0d%016X;" % HI,
            " or.pred %p5,%p3,%p4;",
        ],
        8,
    )
    o += ptx_branch(1, [" setp.le.f64 %%p3,%%fd1,0d%016X;" % ONE] + k7, 12)
    o += ptx_branch(2, [" setp.ltu.f64 %%p3,%%fd1,0d%016X;" % U] + k7, 16)
    return "\n".join(o + [" ret;", "}"]) + "\n"


def edges():
    e = [0, 1 << 63, 1, 1 << 63 | 1, 0x000FFFFFFFFFFFFF, 0x0010000000000000]
    e += [0x0008000000000000, 0x0007FFFFFFFFFFFF, 0x0008000000000001]
    e += [ONE - 1, ONE, ONE + 1, ONE | 1 << 63, 0x4000000000000000, 0xC000000000000000]
    e += [U - 1, U, U + 1, LO - 1, LO, HI, HI + 1]
    e += [
        0x7FEFFFFFFFFFFFFF,
        0xFFEFFFFFFFFFFFFF,
        0x7FF0000000000000,
        0xFFF0000000000000,
    ]
    # Quiet and signalling NaNs of either sign, with payloads.
    e += [
        0x7FF8000000000000,
        0xFFF8000000000001,
        0x7FF0000000000001,
        0xFFF4DEADBEEF0000,
    ]
    return e


def records():
    r = random.Random(2026)
    e = edges()
    rec = [(a, b, c, r.choice([7, 7, 0, 1, 8])) for a in e for b in e for c in (0, 1)]
    while len(rec) < N:
        m = r.randrange(6)
        a = r.getrandbits(64)
        if m >= 3:
            a = struct.unpack("<Q", struct.pack("<d", r.uniform(-2, 2)))[0]
        b = (a, a + 1, a ^ 1 << 63, r.choice(e), r.getrandbits(64), r.getrandbits(64))[
            m
        ]
        c = r.getrandbits(1) * r.getrandbits(32)
        rec.append((a, b & (1 << 64) - 1, c, r.choice([7, 0, 3, 0x9E3779B9])))
    return rec


def fcmp(cmp, x, y):
    """IEEE comparison of two f64 bit patterns: le is false, ltu / gtu true on a NaN."""
    fx, fy = (struct.unpack("<d", struct.pack("<Q", v))[0] for v in (x, y))
    un = math.isnan(fx) or math.isnan(fy)
    return {"le": not un and fx <= fy, "ltu": un or fx < fy, "gtu": un or fx > fy}[cmp]


def model(g, a, b, c, k):
    sign = 1 << 63
    val = {"a": a, "b": b, "u": U, "|a|": a & ~sign, "-a": a ^ sign, "-|a|": a | sign}
    val.update({"|b|": b & ~sign, "-b": b ^ sign, "one": ONE})
    val.update({"imm%d" % i: v for i, v in enumerate(IMMS)})
    w = 0
    for i, (x, y, cmp, bop, pred) in enumerate(shapes()):
        q = (c != 0) == (pred == "c")
        t = fcmp(cmp, val[x], val[y])
        w |= {"and": t and q, "or": t or q, "xor": t != q}[bop] << i
    taken = [
        fcmp("ltu", a, LO) or fcmp("gtu", a, HI),
        fcmp("le", a, ONE) and k != 7,
        fcmp("ltu", a, U) and k != 7,
    ]
    br = [
        arm(2 * n + 1, 32 + 16 * n, c, k) if t else arm(2 * n + 2, 40 + 16 * n, c, k)
        for n, t in enumerate(taken)
    ]
    return [w & M32, w >> 32] + br + [0]


if __name__ == "__main__":
    rec = records()
    assert len(rec) == N and len(shapes()) <= 64
    if sys.argv[1:2] == ["--check"]:
        got = struct.unpack("<%dI" % (6 * N), open(sys.argv[2], "rb").read())
        bad = sum(list(got[6 * g : 6 * g + 6]) != model(g, *rec[g]) for g in range(N))
        nan = sum(any(v & ~(1 << 63) > 0x7FF << 52 for v in r[:2]) for r in rec)
        print(
            "%d records (%d with a NaN operand), %d shapes and 3 branches: %d mismatching records"
            % (N, nan, len(shapes()), bad)
        )
        sys.exit(bad != 0)
    open("kernel.ptx", "w").write(ptx())
    open("in.bin", "wb").write(b"".join(struct.pack("<QQII", *r) for r in rec))
