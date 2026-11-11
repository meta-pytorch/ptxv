# probes/tma_im2col generator: tensor maps (cuTensorMapEncodeIm2col through
# probes/tmap_encode/tmap_encode), tensors, coordinate tables, kernel.ptx and case.toml.
#   python3 gen.py <tmap_encode binary>        (needs a GPU context for the encoder)
import os
import random
import struct
import subprocess
import sys

ENC = sys.argv[1]
r = random.Random(77)
# (name, dtype code, element bytes, dims (C, spatial..., N; innermost first), lower corner,
#  upper corner, channels per pixel, pixels per column, swizzle 0 none / 1 32B / 2 64B / 3 128B)
LMAPS = [
    ("m0", 2, 4, [8, 6, 5, 4, 2], [-1, -1, -1], [1, 0, -1], 8, 64, 0),
    ("m1", 2, 4, [16, 7, 6, 3], [-2, -1], [0, 1], 4, 48, 0),
    ("m2", 2, 4, [32, 40, 3], [-3], [2], 32, 32, 3),
    ("m3", 0, 1, [64, 16, 16, 16, 1], [0, 0, 0], [0, 0, 0], 128, 128, 3),
    ("m4", 0, 1, [64, 8, 8, 8, 1], [-1, -1, -1], [-1, -1, -1], 64, 64, 2),
    ("m5", 1, 2, [64, 9, 5, 2], [-1, -1], [-1, 1], 64, 16, 3),
]
# Stores (cp.async.bulk.tensor...im2col_no_offs): the smem box (from `src`) to the tensor. A
# store whose pixel box reaches outside the tensor (lower corner < 0 or upper corner > 0) is an
# illegal instruction on GB300.
SMAPS = [
    ("s0", 1, 2, [128, 8, 8, 4, 1], [0, 0, 0], [0, 0, 0], 64, 128, 3),
    ("s1", 2, 4, [8, 7, 5, 2], [1, 0], [0, -1], 8, 24, 0),
    ("s2", 2, 4, [16, 20, 2], [2], [-3], 16, 16, 2),
]
K, KS = 8, 4
OBITS = {
    5: 5,
    4: 8,
    3: 16,
}  # im2col offset width per rank (ptxas packs them in 16 bits)


def nbytes(m):
    s = m[2]
    for d in m[3]:
        s *= d
    return s


def boxbytes(m):
    return m[6] * m[7] * m[2]


def encode(m):
    name, dt, es, dims, lo, up, ch, px, sw = m
    strides, s = [], dims[0] * es
    for d in dims[1:]:
        strides.append(s)
        s *= d
    line = "im2col %d %d 0 %d 0 0 0x10000 %s %s %s %s %d %d %s" % (
        dt,
        len(dims),
        sw,
        " ".join(map(str, dims)),
        " ".join(map(str, strides)),
        " ".join(map(str, lo)),
        " ".join(map(str, up)),
        ch,
        px,
        " ".join(["1"] * len(dims)),
    )
    h = subprocess.run([ENC], input=line + "\n", capture_output=True, text=True).stdout
    h = h.split(" => ")[1].strip()
    assert not h.startswith("ERR"), (line, h)
    return h


def table(maps, k, offsets):
    rows = []
    for m in maps:
        name, dt, es, dims, lo, up, ch, px, sw = m
        sp = dims[1:-1]
        for i in range(k):
            # channel start: a multiple of 16 bytes (anything else is an illegal instruction)
            c0 = r.choice([0, 0, 1, 2]) * (16 // es) if ch < dims[0] else 0
            pix = [r.randint(lo[j], sp[j] - 1 + up[j]) for j in range(len(sp))]
            if i == 0:
                pix = list(lo)
            if i == 1:
                pix = [sp[j] - 1 + up[j] for j in range(len(sp))]
            n = r.randint(0, dims[-1] - 1)
            nb = OBITS[len(dims)]
            offs = [r.randint(0, 3) for _ in sp] if offsets else [0] * len(sp)
            if offsets and i == 2:
                offs = [(1 << nb) - 1 for _ in sp]
            crd = [c0] + pix + [n]
            rows.append(crd + [0] * (5 - len(crd)) + offs + [0] * (3 - len(offs)))
    return rows


os.makedirs("data", exist_ok=True)
rows = table(LMAPS, K, True) + table(SMAPS, KS, False)
open("data/coords.bin", "wb").write(b"".join(struct.pack("<8i", *c) for c in rows))
for m in LMAPS + SMAPS:
    n = nbytes(m) // m[2]
    if m[0].startswith("s"):
        data = struct.pack(f"<{n}I", *[0xEEEEEEEE] * n)[: nbytes(m)]
    elif m[2] == 4:
        data = struct.pack(f"<{n}I", *[0x10000000 | i for i in range(n)])
    elif m[2] == 2:
        data = struct.pack(f"<{n}H", *[0x8000 | (i & 0x7FFF) for i in range(n)])
    else:
        data = bytes([1 + (i * 7) % 251 for i in range(n)])
    open(f"data/{m[0]}.bin", "wb").write(data)
SRC = 16384
open("data/src.bin", "wb").write(bytes([(i * 13 + 5) & 0xFF for i in range(SRC)]))
OUT = sum(boxbytes(m) * K for m in LMAPS)

# kernel.ptx
P = [
    """// probes/tma_im2col (gen.py): im2col TMA through cuTensorMapEncodeIm2col maps.
// Loads: cp.async.bulk.tensor.{3,4,5}d.shared::cluster.global.im2col (SASS UTMALDG.<n>D.IM2COL),
// K per map with coordinates and offsets from `coords`, each box copied out to `out` in turn.
// Stores: cp.async.bulk.tensor.{3,4,5}d.global.shared::cta.im2col_no_offs.bulk_group (SASS
// UTMASTG.<n>D.IM2COL) of the box `src` holds, KS per map, into tensors s0..s2.
.version 9.0
.target sm_103a
.address_size 64
"""
]
params = ",\n".join(
    [f"    .param .align 64 .b8 {m[0]}[128]" for m in LMAPS + SMAPS]
    + ["    .param .u64 out", "    .param .u64 crd", "    .param .u64 src"]
)
P.append(f".visible .entry tma_im2col(\n{params}\n)\n.maxntid 128, 1, 1\n{{")
P.append("""    .reg .pred %p<8>;
    .reg .b16 %h<8>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<16>;
    .shared .align 1024 .b8 sm[16384];
    .shared .align 8 .b64 bar;
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, sm;
    mov.u32 %r2, bar;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra INIT_DONE;
    mbarrier.init.shared::cta.b64 [%r2], 1;
    fence.mbarrier_init.release.cluster;
INIT_DONE:
    bar.sync 0;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd1, %rd1;
    ld.param.u64 %rd2, [crd];
    cvta.to.global.u64 %rd2, %rd2;
    mov.u32 %r3, 0;                 // mbarrier phase""")


def coords_load(row):
    return f"""    mad.lo.u32 %r5, %r4, 32, {row * 32};
    cvt.u64.u32 %rd3, %r5;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.v4.u32 {{%r10, %r11, %r12, %r13}}, [%rd4];
    ld.global.v4.u32 {{%r14, %r15, %r16, %r17}}, [%rd4+16];"""


def copy(name, size):
    """Unrolled: thread t copies bytes 16t + 2048j of the box (j < size / 2048, guarded)."""
    out = []
    for j, o in enumerate(range(0, size, 2048)):
        out.append(f"""    add.u32 %r8, %r6, {o};
    setp.lt.u32 %p2, %r8, {size};
    add.u32 %r7, %r1, %r8;
    @%p2 ld.shared.v4.u32 {{%r20, %r21, %r22, %r23}}, [%r7];
    cvt.u64.u32 %rd7, %r8;
    add.u64 %rd8, %rd1, %rd7;
    @%p2 st.global.v4.u32 [%rd8], {{%r20, %r21, %r22, %r23}};""")
    return "\n".join(out)


row = 0
for m in LMAPS:
    name, rank, size = m[0], len(m[3]), boxbytes(m)
    cs = ", ".join(f"%r{10 + k}" for k in range(rank))
    os_ = ", ".join(f"%h{1 + k}" for k in range(rank - 2))
    P.append(f"""    // load map {name}: rank {rank}, {size} bytes per box
    mov.u32 %r4, 0;
L_{name}:
    @%p0 bra W_{name};
{coords_load(row)}
    cvt.u16.u32 %h1, %r15;
    cvt.u16.u32 %h2, %r16;
    cvt.u16.u32 %h3, %r17;
    mbarrier.arrive.expect_tx.shared::cta.b64 %rd5, [%r2], {size};
    mov.b64 %rd9, {name};
    cvta.param.u64 %rd6, %rd9;
    cp.async.bulk.tensor.{rank}d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%r1], [%rd6, {{{cs}}}], [%r2], {{{os_}}};
W_{name}:
    mbarrier.try_wait.parity.shared::cta.b64 %p1, [%r2], %r3;
    @!%p1 bra W_{name};
    xor.b32 %r3, %r3, 1;
    shl.b32 %r6, %r0, 4;
{copy(name, size)}
    add.u64 %rd1, %rd1, {size};
    fence.proxy.async.shared::cta;
    bar.sync 0;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p3, %r4, {K};
    @%p3 bra L_{name};""")
    row += K
# stores: fill smem from src once (all threads), then thread 0 stores each box, waiting each
P.append(f"""    // stores: smem = src (16384 bytes), then the boxes
    ld.param.u64 %rd10, [src];
    cvta.to.global.u64 %rd10, %rd10;
    shl.b32 %r6, %r0, 4;
F_SRC:
    cvt.u64.u32 %rd7, %r6;
    add.u64 %rd8, %rd10, %rd7;
    ld.global.v4.u32 {{%r20, %r21, %r22, %r23}}, [%rd8];
    add.u32 %r7, %r1, %r6;
    st.shared.v4.u32 [%r7], {{%r20, %r21, %r22, %r23}};
    add.u32 %r6, %r6, 2048;
    setp.lt.u32 %p2, %r6, {SRC};
    @%p2 bra F_SRC;
    fence.proxy.async.shared::cta;
    bar.sync 0;
    @%p0 bra DONE;""")
for m in SMAPS:
    name, rank = m[0], len(m[3])
    cs = ", ".join(f"%r{10 + k}" for k in range(rank))
    P.append(f"""    // store map {name}: rank {rank}
    mov.u32 %r4, 0;
S_{name}:
{coords_load(row)}
    mov.b64 %rd9, {name};
    cvta.param.u64 %rd6, %rd9;
    cp.async.bulk.tensor.{rank}d.global.shared::cta.im2col_no_offs.bulk_group [%rd6, {{{cs}}}], [%r1];
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
    add.u32 %r4, %r4, 1;
    setp.lt.u32 %p3, %r4, {KS};
    @%p3 bra S_{name};""")
    row += KS
P.append("DONE:\n    ret;\n}\n")
open("kernel.ptx", "w").write("\n".join(P))

# case.toml
L = [
    'name = "probe_tma_im2col"',
    'description = """',
    open("desc.txt").read().strip(),
    '"""',
    'ptx = "kernel.ptx"',
    'entry = "tma_im2col"',
    "grid = [1, 1, 1]",
    "block = [128, 1, 1]",
    'outputs = ["out", ' + ", ".join(f'"{m[0]}"' for m in SMAPS) + "]",
    "",
]
for m in LMAPS + SMAPS:
    L += [
        "[[buffers]]",
        f'name = "{m[0]}"',
        'dtype = "u8"',
        f"shape = [{nbytes(m)}]",
        'init = "file"',
        f'file = "data/{m[0]}.bin"',
    ]
    if m[0].startswith("s"):
        L += [f'expect = "expected/{m[0]}.bin"']
    L += [""]
L += [
    "[[buffers]]",
    'name = "coords"',
    'dtype = "i32"',
    f"shape = [{len(rows) * 8}]",
    'init = "file"',
    'file = "data/coords.bin"',
    "",
    "[[buffers]]",
    'name = "src"',
    'dtype = "u8"',
    f"shape = [{SRC}]",
    'init = "file"',
    'file = "data/src.bin"',
    "",
    "[[buffers]]",
    'name = "out"',
    'dtype = "u8"',
    f"shape = [{OUT}]",
    'init = "zeros"',
    'expect = "expected/out.bin"',
    "",
]
for m in LMAPS + SMAPS:
    L += [
        "[[params]]",
        'kind = "tensormap"',
        f'buffer = "{m[0]}"',
        f"dims = {m[3]}",
        f"box = [{m[6]}, {m[7]}]",
        f'swizzle = "{["none", "32B", "64B", "128B"][m[8]]}"',
        f'raw = "{encode(m)}"',
        "",
    ]
L += [
    "[[params]]",
    'kind = "ptr"',
    'buffer = "out"',
    "",
    "[[params]]",
    'kind = "ptr"',
    'buffer = "coords"',
    "",
    "[[params]]",
    'kind = "ptr"',
    'buffer = "src"',
    "",
    "[[variants]]",
    'id = "good"',
    'expect = ["clean"]',
    "",
]
open("case.toml", "w").write("\n".join(L))
