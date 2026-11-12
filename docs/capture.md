# Capturing any torch op or generated kernel as a ptxv case

**Status (2026-11-03):** `tools/capture.py` turns a torch expression, a Python function, or a
Triton / CuTeDSL kernel into a ptxv case: every kernel launched, in order, with its cubin(s),
its parameters mapped to named buffers, the exact input bytes, and the GPU's output bytes. 33
ops captured in the smoke test; all 33 captures worked. 4 of them lift on today's lifter:
cumsum, cat, a Triton cat, and a CuTeDSL add. The CuTeDSL add lifts on both front-ends. For
those 4, `check` is clean and the simulator's output equals the GPU's bit for bit. The
others are blocked on missing SASS instructions, listed below.

One kernel out of a sequence (rung 3b): with `--kernels`, a kept kernel starts from exactly
what earlier, filtered-out kernels wrote, and its outputs are what it changed (see "Snapshots"
below; fixtures/r3b_trtllm_bmm is gemm1 out of flashinfer's 5-launch MoE). The launch recorder
snapshots the memory of every kept launch, with or without `--kernels` (2026-11-12, MISS 173).

Multi-GPU (rung 4a): `tools/capture_dist.py` records a `torchrun --nproc-per-node N` run and
merges every rank into one multi-GPU case; see [rung4.md](rung4.md).

## Usage

```sh
source env.sh
# A torch expression; inputs are seeded CPU tensors moved to the GPU.
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py \
    --op 'torch.softmax(x, -1)' --inputs 'x=randn(4,128,bf16)' --out fixtures/c_softmax
# Every kernel a function launches: the file defines run(**inputs) -> output(s), and
# optionally inputs() -> {name: tensor}.
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py --file tools/capture_examples/mixed.py --out DIR
# One Triton (or CuTeDSL) kernel, keeping its PTX: a PTX case, so both front-ends work.
CUDA_VISIBLE_DEVICES=3 $PTXV_PYTHON tools/capture.py \
    --triton tools/capture_examples/triton_softmax.py:softmax_kernel --out DIR
# Then:
ptxv check DIR            # verdict, plus "expect (output 0): out: bitwise equal to the captured GPU output"
PTXV_GPUS=3 ptxv hw DIR   # simulator vs GB300, many runs
```

- `--inputs NAME=EXPR` can be repeated or comma-separated. Helpers: `randn`, `rand`,
  `zeros`, `ones`, `arange`, `randint(lo, hi, *shape)`, `full(*shape, value)`, `eye(n)`, each
  with an optional trailing dtype (`bf16 f16 f32 f64 i32 i64 i16 i8 u8 bool`). `torch` and
  `F` are also available. `--seed` (default 0) seeds the generator.
- Outputs:
  - the tensors the op returns (a tensor, a tuple, a namedtuple or a dict);
  - any input it changed in place (`index_add_`, `out=`).
- Other options:
  - `--kernels REGEX` keeps only some launches;
  - `--binary` drops the PTX (writes it to `jit.ptx` for reference only);
  - `--no-warmup`;
  - `--no-prune` (see cubins below);
  - `--work DIR` keeps the raw recordings.
- `tools/capture_smoke.py` runs the batch below. It writes `results.json` and prints the table.

## What a captured case contains

| file | content |
|---|---|
| `case.toml` | every launch in stream order (`[[launches]]`: grid, block, cluster, dynamic smem, PDL), its parameters, the buffers, `outputs`, one `good` variant expecting `clean` |
| `k0.cubin`, `k1.cubin`, ... | one per module the kernels came from. A launch whose kernel is not in the first launch's module names its own (`[[launches]] cubin = "k1.cubin"`) |
| `kernel.ptx` (+ `jit.cubin`) | Triton / CuTeDSL: when every launch is one JIT kernel with dumped PTX, the case is a PTX case |
| `data/NAME.bin` | exact bytes of every buffer that was live before the call (`init = "file"`) |
| `expected/NAME.bin` | the GPU's bytes of each output after the call (`expect`, `expect_offset`) |
| `capture.json` | the launches, buffers, pointer-like parameter words that matched no allocation (`unresolved`), and warnings |

How it works. This generalizes `fixtures/r3_grouped_mm/capture.py` and
`fixtures/r2_cublas_gemm/capture.py`.

- **Recording.**
  - The script re-executes itself under `probes/launchcap`. It records every launch (now
    including the PDL attribute) and, with `LAUNCHCAP_MEMOPS=1`, every
    `cudaMemcpy*` / `cudaMemset*`, in order. It also dumps library images.
  - It also runs under `probes/launchcap/cubincap`, which uses CUPTI to dump every module
    the driver loads.
  - Triton and CuTeDSL dump their PTX (`TRITON_KERNEL_DUMP`, `CUTE_DSL_KEEP`). Triton is
    pointed at CUDA 13's ptxas, cuobjdump and nvdisasm, because its bundled tools do not know
    sm_103a.
- **Warm-up, then capture.** The op runs once to warm up (lazy module loading, autotuning,
  library workspaces). It is captured on the second call, between two snapshots of the
  caching allocator: the blocks live before the call, and the allocations made during it.
  Inputs the warm-up changed in place (`index_add_`, an Inductor `in_out_ptr` kernel) are
  restored first, so the capture reads the inputs as given (before this, an in-place kernel
  was captured on its own warm-up result: docs/compile-vs-eager.md).
- **Launches through a runtime's cuGetProcAddress.** launchcap now hands its hooked
  `cuGetProcAddress` to every runtime that asks for it, not only under `LAUNCHCAP_VMM`.
  cuBLAS's statically linked cudart resolves `cuLaunchKernel` that way: its small-shape
  kernels (`gemmSN_NN_kernel` for an 8x128 @ 128x384 f32 mm) were not recorded at all
  ("no kernel launches recorded").
- **Pointers.**
  - Every 8-byte parameter word that points into one of those blocks becomes a `ptr`. A
    pointer into the middle of a block gets a `byte_offset`; one past the end also counts.
  - A 128-byte `CUtensorMap` whose address does is decoded into a `tensormap`
    (docs/tensormap.md): dims, strides, box, swizzle, and the raw bytes for hardware runs. It
    can sit at the top level, or 64-byte aligned inside a packed argument block (cuBLAS; its
    bit-1 maps are packed 64 bytes apart).
  - The descriptor check is strict: rank and type valid, unused slots zero, the row fits its
    stride, and the derived box-bytes and swizzle words are consistent for driver-built maps.
    Without it, a TensorInfo struct (a pointer followed by sizes) passes for a map.
  - Other parameters are kept as exact bytes: 4-byte ones as `u32`, the rest as `bytes`.
    Pointers inside struct parameters become `fields`.
- **Buffers.**
  - Overlapping blocks merge: an allocation reused during the call is one buffer.
  - If every reference into a buffer lies within one named tensor, the buffer is that tensor,
    with its dtype and shape. Otherwise it is the whole block range as `u8`, with the tensor
    at an offset (and `expect_offset`).
  - A buffer that a tensor map reads takes the map's element type, because the simulator
    takes the element size from the buffer.
- **Initial contents (2026-11-12, MISS 173).**
  - A buffer starts with the bytes the first kept launch that references it saw: that
    launch's pre-snapshot ("Snapshots" below). Whatever wrote them inside the call is
    covered: a filtered-out launch, a `cudaMemcpy` (an op's `clone()`), a `cudaMemset`, an
    input uploaded from the host.
  - A buffer that a copy, a fill or a filtered-out launch changes between two kept launches
    that reference it refuses the case (`capture: refused (MISS 173)`), as does an output
    changed after the last kept launch that references it: a case cannot express either.
  - Without a covering snapshot (an allocation over 256 MiB, a launch into a capturing
    stream), device-to-device copies and fills before the first kept launch are folded into
    the initial contents, as before. A buffer that something else writes before its first kept
    reader then refuses the case, and an allocation made during the call is undefined until
    written (`init = "undef"`; a read of it is reported).
  - Before 2026-11-12, only `--kernels` captures had snapshots, and runtime `<<<>>>` launches
    (PyTorch's own kernels) never did. Allocations made during the call started as zeros;
    copies and fills between kernels were dropped with a warning, host uploads between kernels
    without one: a kernel reading an input built inside the call read zeros.
  - `capture.json` `initial_bytes` names each buffer's source ("snapshot before kept launch
    N", "before the call", "undefined"); `memops_between_kernels` lists the copies and fills
    between kept launches.
- **Cubins.**
  - Each kernel's cubin is the smallest captured ELF image defining it. CUPTI's images are
    preferred, because they are exact. The size launchcap's dump computed for an ELF ignored
    data after the section headers (fixed: it now takes the furthest table or section). This
    truncated cuDNN's image.
  - Module cubins from libtorch_cuda hold hundreds of kernels (9 MB for SoftMax.cu). The
    code of every *entry* kernel not launched is overwritten with the module's NOP padding.
    The layout, symbols and relocations are unchanged, so the cubin loads and disassembles
    as before, but it compresses about 5x smaller (the committed c_cat cubin is 2.3 MB raw,
    about 60 KB in git). `--no-prune` keeps the original.

- **Compiled-in cluster shapes (2026-11-04).** A kernel built with `__cluster_dims__` is
  launched with no cluster attribute (launchcap records 1 x 1 x 1). The capture reads
  `EIATTR_CTA_PER_CLUSTER` (0x3d) from the cubin's `.nv.info.<kernel>` and writes `cluster`
  (flashinfer's routing kernel: 8 x 1 x 1).
- **Preferred cluster dimension (2026-11-06).** cuBLAS launches its `*_2x2f_*` kernels with
  `cudaLaunchAttributePreferredClusterDimension` (mm_4k_8k: preferred 4 x 1 x 1, cluster
  2 x 1 x 1). launchcap and the Python API's `ptxvcap.c` record it as `preferred_cluster`, and
  the case gets `preferred_cluster = [...]` beside `cluster` (S-CGA-SHAPE). Earlier captures
  dropped it.
- **Architecture (2026-11-07).** `case.toml` gets `arch = "sm_XYa"`, the device the capture
  ran on (`PTXV_ARCH` overrides; tools/capture_dist.py writes it too). ptxv compiles a PTX
  case's kernel for that architecture, so a capture made on a GB200 is checked as sm_100a
  code; without the line it is compiled for the default, sm_103a.
- **Snapshots (`--kernels`, 2026-11-04).** Filtered-out launches may write the kept ones'
  inputs (a MoE's routing kernel writes the permutation the GEMM reads). Before this change,
  those buffers started as zeros.
  - `probes/launchcap` takes `LAUNCHCAP_SNAP=regex` (and `LAUNCHCAP_SNAP_DIR`). For a
    matching launch it synchronizes, then writes every device allocation that an 8-byte
    parameter word points into (`cuMemGetAddressRange`; tensor-map addresses included), once
    before the kernel and once after. The record carries `"snap": N`.
  - Only launches recorded before they are issued are covered: driver launches,
    `_cudaLaunchKernelEx` and, with `LAUNCHCAP_SNAP` set, runtime `<<<>>>` launches and
    `cudaLaunchKernelExC` (2026-11-12: their function is looked up before the launch; before,
    they were recorded after it and had no snapshot).
  - `capture.py` passes `--kernels` as the regex, or `.` (every launch) without it. Each
    buffer starts from the pre-snapshot of the first kept launch that references it.
  - The op's own outputs are replaced: every buffer whose bytes the kept launches changed is an
    output, with the post-snapshot of the last kept launch that references it as `expect`, and
    so is the op's own result where a kept launch references it, even unchanged (2026-11-12:
    the warm-up can leave the same bytes there, and the capture had no output).
  - Allocations outside the caching allocator that a kept launch points into become buffers
    too (from the snapshot).
  - Kept launches that are not contiguous are warned about.

Schema additions (crates/case, minimal):

- `init = "file"` with `file = "data/x.bin"`;
- `expect` / `expect_offset` on a buffer: `ptxv check` prints the comparison for every
  terminal output;
- `cubin` on a `[[launches]]` entry (binary-only cases): `load_with`, the hw runner and the
  test harness load it;
- `byte_offset` on a `tensormap`;
- dtypes `i64`, `u16`, `i16`, `i8`.

Two other changes:

- `PTXV_LIFT_REPORT=1` now reports every launch's kernel (`lift report for K: N kinds of
  instruction not lowered`). A fully lowered case goes on to the check.
- `PTXV_GPUS=3` (or `0,3`) restricts which GPUs `hw` / `equiv` may pick. Without it,
  `CUDA_VISIBLE_DEVICES` restricts the choice, read as the CUDA runtime reads it (indices or
  `GPU-` UUID prefixes, up to the first entry that names no device). Before 2026-11-06 it was
  ignored: the worker's own `CUDA_VISIBLE_DEVICES` replaced it, so `CUDA_VISIBLE_DEVICES=3
  ptxv hw` could run on any free GPU.

## Smoke test (2026-11-03, torch 2.12, Triton 3.5, GB300)

Shapes: 4x128 rows, 64x64 matmul and attention (1x2x64x64). Inputs are seeded randn. The
check is `ptxv check --certify --max-states 200000` under a 32 GB memory cap. The SASS front
runs on the captured cubins, or on ptxas's output for a PTX case. "missing" is
`PTXV_LIFT_REPORT=1`, with register numbers folded.

| op | kernels (cubins) | lifts (SASS) | missing SASS instructions | verdict | sim = GPU bitwise |
|---|---|---|---|---|---|
| softmax | softmax_warp_forward (1) | no (0/1 kernels) | `FADD modifier not modeled (FTZ/SAT)`, `bad f16 imm`, `bad f32 operand -R`, `sub-word LDG not modeled`, `FCHK`, `FFMA.RM`, `FFMA.RP`, `FFMA.RZ`, `FFMA.SAT`, `FSEL`, `FSETP.GEU.AND`, `FSETP.GT.AND`, `FSETP.GTU.FTZ.AND`, `FSETP.NEU.AND`, `FSETP.NEU.FTZ.AND`, `MUFU.EX2`, `MUFU.RSQ` | - | - |
| log_softmax | softmax_warp_forward (1) | no (0/1 kernels) | `bad f32 operand -R`, `sub-word LDG not modeled`, `FFMA.RM`, `FFMA.SAT`, `FSEL`, `FSETP.GEU.AND`, `FSETP.GT.AND`, `FSETP.NEU.AND`, `MUFU.EX2` | - | - |
| layer_norm | at::native::vectorized_layer_norm_kernel (1) | no (0/1 kernels) | `FADD modifier not modeled (FTZ/SAT)`, `bad f32 operand -R`, `sub-word LDG not modeled`, `FCHK`, `FFMA.RM`, `FFMA.RP`, `FFMA.RZ`, `FSETP.GEU.AND`, `FSETP.GT.AND`, `FSETP.GTU.FTZ.AND`, `FSETP.NEU.FTZ.AND`, `MUFU.RSQ`, `VIMNMX.U32`, `VIMNMX` | - | - |
| rms_norm | at::native::vectorized_layer_norm_kernel (1) | no (0/1 kernels) | `FADD modifier not modeled (FTZ/SAT)`, `bad f32 operand -R`, `sub-word LDG not modeled`, `FCHK`, `FFMA.RM`, `FFMA.RP`, `FFMA.RZ`, `FSETP.GEU.AND`, `FSETP.GTU.FTZ.AND`, `FSETP.NEU.FTZ.AND`, `MUFU.RSQ`, `VIMNMX.U32`, `VIMNMX` | - | - |
| sum_dim | at::native::reduce_kernel (1) | no (0/1 kernels) | `special register SR_LTMASK not modeled`, `POPC`, `VIMNMX.U32` | - | - |
| max_dim | at::native::reduce_kernel (1) | no (0/1 kernels) | `special register SR_LTMASK not modeled`, `FSEL`, `FSETP.GEU.AND`, `FSETP.GT.AND`, `FSETP.NAN.AND`, `FSETP.NAN.OR`, `POPC`, `VIMNMX.U32` | - | - |
| mean_dim | at::native::reduce_kernel (1) | no (0/1 kernels) | `special register SR_LTMASK not modeled`, `POPC`, `VIMNMX.U32` | - | - |
| cumsum | at::native::tensor_kernel_scan_innermost_dim (1) | yes |  | sass: [clean] | sass: yes |
| gelu | at::native::vectorized_elementwise_kernel (1) | no (0/1 kernels) | `bad f32 operand -R`, `sub-word LDG not modeled`, `BREAK.RELIABLE`, `FSEL`, `FSETP.GE.AND`, `MUFU.EX2` | - | - |
| silu | at::native::vectorized_elementwise_kernel (1) | no (0/1 kernels) | `FADD modifier not modeled (FTZ/SAT)`, `sub-word LDG not modeled`, `BREAK.RELIABLE`, `FCHK`, `FFMA.RM`, `FFMA.RP`, `FFMA.RZ`, `FFMA.SAT`, `FSETP.GTU.FTZ.AND`, `FSETP.NEU.FTZ.AND`, `MUFU.EX2`, `MUFU.RSQ` | - | - |
| sdpa | cudnn_generated_fort_native_sdpa_sm80_flash_fprop_wmma_f16_k (1) | no (0/1 kernels) | `FADD modifier not modeled (FTZ/SAT)`, `FFMA.FTZ`, `FMNMX.FTZ`, `FMNMX3.FTZ`, `FMUL.FTZ`, `FSEL`, `FSETP.NEU.FTZ.AND`, `HMMA.16816.F32.BF16`, `IDP.2A.LO.U16.U8`, `LDGDEPBAR`, `LDGSTS.E.BYPASS.128`, `LDSM.16.M88.4`, `LDSM.16.MT88.4`, `MUFU.EX2` | - | - |
| embedding | at::native::indexSelectSmallIndex (1) | no (0/1 kernels) | `ST.E` | - | - |
| index_add | at::native::indexFuncSmallIndex (1) | no (0/1 kernels) | `ATOMS variant not modeled`, `operand modifier not modeled: -R`, `ATOM.E.ADD.F32.FTZ.RN.STRONG.GPU`, `ATOM.E.CAST.SPIN`, `QSPC.E.D`, `QSPC.E.S`, `ST.E` | - | - |
| matmul_bf16 | nvjet_sm103_tst_64x8_64x16_1x4_h_bz_NNT (1) | no (0/1 kernels) | `LDTM shape not modeled yet (only 32x32b, 16x64b .xN)`, `STSM.16.MT88.2` | - | - |
| matmul_relu | nvjet_sm103_tst_64x8_64x16_1x4_h_bz_NNT<br>at::native::vectorized_elementwise_kernel (2) | no (0/2 kernels) | `LDTM shape not modeled yet (only 32x32b, 16x64b .xN)`, `STSM.16.MT88.2`, `sub-word LDG not modeled`, `BREAK.RELIABLE`, `FMNMX`, `FSETP.NAN.AND` | - | - |
| sort | elementwise_kernel_with_index<br>at::native::elementwise_kernel<br>at::native::radixSortKVInPlace (3) | no (1/3 kernels) | `VIMNMX.U32`, `SHFL predicate output not modeled`, `operand modifier not modeled: -R`, `FSEL`, `ST.E.64`, `ST.E`, `UBMSK` | - | - |
| triton_softmax | softmax_kernel (1) | no (0/1 kernels) | `bad f32 operand -R`, `sub-word LDG not modeled`, `FMNMX`, `FSETP.GEU.AND`, `FSETP.GT.AND`, `MUFU.EX2` | ptx: unsupported `mad.wide.s32` | - |
| triton_layernorm | layernorm_kernel (1) | no (0/1 kernels) | `sub-word LDG not modeled`, `FSEL`, `FSETP.GEU.AND`, `FSETP.GT.AND`, `MUFU.SQRT` | ptx: panic: index out of bounds at crates/sim/src/exec.rs:695 (`self.k.code` is empty) | - |
| triton_cumsum | cumsum_kernel (1) | no (0/1 kernels) | `FSEL` | ptx: unsupported `ld.global.b32 { %r1 }, [...]` (one-element vector braces) | - |
| triton_cat | cat_kernel (1) | yes |  | sass: [clean]; ptx: unsupported `mad.wide.s32` | sass: yes |
| cute_add | kernel_cutlass_add_kernel_tensorptrf32gmemo41281281_tensorptr (1) | yes |  | sass: [clean]; ptx: [clean] | sass: yes; ptx: yes |
| mixed_3_modules | nvjet_sm103_tst_64x8_64x16_1x4_h_bz_NNT<br>at::native::unrolled_elementwise_kernel<br>scale_kernel (3) | no (1/3 kernels) | `LDTM shape not modeled yet (only 32x32b, 16x64b .xN)`, `STSM.16.MT88.2`, `bad f32 operand |R|`, `operand modifier not modeled: -R`, `sub-word LDG not modeled`, `BREAK.RELIABLE`, `F2F.F32.F64`, `F2F.F64.F32`, `F2FP.F16.F32.PACK_AB`, `F2I.S64.TRUNC`, `FSEL`, `FSETP.NEU.AND`, `HADD2.F32`, `I2F.S64` | - | - |
| add_f32 | at::native::vectorized_elementwise_kernel (1) | no (0/1 kernels) | `BREAK.RELIABLE` | - | - |
| transpose_copy | at::native::elementwise_kernel (1) | no (0/1 kernels) | `VIMNMX.U32` | - | - |
| index_select | at::native::indexSelectSmallIndex (1) | no (0/1 kernels) | `ST.E` | - | - |
| cat | at::native::CatArrayBatchedCopy_vectorized (1) | yes |  | sass: [clean] | sass: yes |
| mul_scalar_f32 | at::native::vectorized_elementwise_kernel (1) | no (0/1 kernels) | `BREAK.RELIABLE` | - | - |
| where | at::native::vectorized_elementwise_kernel<br>at::native::vectorized_elementwise_kernel (2) | no (0/2 kernels) | `bad address R`, `BREAK.RELIABLE`, `FSETP.GE.AND`, `FSETP.GEU.AND`, `FSETP.GT.AND`, `FSETP.GTU.AND`, `FSETP.LE.AND`, `FSETP.LT.AND`, `sub-word LDG not modeled`, `FSEL` | - | - |
| to_bf16 | at::native::vectorized_elementwise_kernel (1) | no (0/1 kernels) | `bad address R`, `BREAK.RELIABLE` | - | - |
| flip | at::native::index_elementwise_kernel (1) | no (0/1 kernels) | `VIMNMX.U32` | - | - |
| gather | at::native::_scatter_gather_elementwise_kernel (1) | no (0/1 kernels) | `VIMNMX.U32` | - | - |
| fill | at::native::vectorized_elementwise_kernel (1) | no (0/1 kernels) | `BREAK.RELIABLE` | - | - |
| add_i32 | at::native::vectorized_elementwise_kernel (1) | no (0/1 kernels) | `BREAK.RELIABLE` | - | - |

Instructions blocking the most ops (number of ops in the table above):

| ops | missing |
|---|---|
| 11 | sub-word `LDG` (`LDG.E.U16` / `.U8`: every bf16 kernel) |
| 10 | `FSEL` |
| 10 | `BREAK.RELIABLE` (every `vectorized_elementwise_kernel`: add, mul, fill, gelu, silu, where, ...) |
| 9 | `VIMNMX.U32` (`elementwise_kernel`, the reductions, gather, flip, layer norm) |
| 8 / 7 / 5 / 4 / 3 | `FSETP.{GEU,GT,NEU.FTZ,GTU.FTZ,NEU}.AND` and friends (`GE`, `LT`, `LE`, `NAN`) |
| 6 | f32 operand negation `-R` (`FADD R, R, -R`) |
| 6 / 4 / 1 | `MUFU.EX2` / `MUFU.RSQ` / `MUFU.SQRT` (agent B's) |
| 5 | `FADD.FTZ` / `.SAT` |
| 5 / 4 / 4 / 3 | `FFMA.RM` / `.RP` / `.RZ` / `.SAT` (the division / rsqrt slow paths), `FCHK` (4) |
| 4 | `ST.E` (generic stores: embedding, index_select, index_add, sort) |
| 3 | `SR_LTMASK`, `POPC` (all reductions) |
| 3 | `LDTM` shapes other than 32x32b / 16x64b, `STSM.16.MT88.2` (the 64x64 nvjet GEMM) |
| 1 each | `HMMA.16816.F32.BF16`, `LDSM.16.M88.4`, `LDSM.16.MT88.4`, `LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, `FMUL.FTZ`, `FMNMX(3).FTZ`, `IDP.2A` (cuDNN's sm80 flash attention); `ATOM.E.ADD.F32.FTZ.RN`, `ATOM.E.CAST.SPIN`, `ATOMS`, `QSPC.E.{S,D}` (index_add); `F2F.F32.F64`, `F2F.F64.F32`, `F2I.S64.TRUNC`, `I2F.S64`, `HADD2.F32`, `F2FP.F16.F32.PACK_AB` (a float copy kernel); `SHFL` predicate output, `UBMSK`, `ST.E.64` (sort) |

Six ops are blocked by a single instruction, so they would lift next:

- `BREAK.RELIABLE`: add_f32, mul_scalar_f32, fill, add_i32;
- `VIMNMX.U32`: transpose_copy, flip, gather;
- `ST.E`: embedding, index_select;
- `FSEL`: triton_cumsum.

PTX front, on the Triton kernels:

- `mad.wide.s32` is not parsed (Triton emits it for every address computation);
- `ld.global.b32 { %r1 }, [...]` (one-element vector braces) is not parsed;
- the layer-norm PTX panics with index out of bounds at `crates/sim/src/exec.rs:695`, because
  the parsed kernel has no code (a ptxv bug, not investigated).

Robustness, as exercised by the batch:

- **Several kernels per op:** sort (3 kernels, 3 modules), where (2), matmul_relu (2).
- **Kernels from different modules:** cuBLAS nvjet + libtorch (matmul_relu); cuBLAS +
  libtorch + Triton (mixed_3_modules); a cuDNN-generated kernel (sdpa picks cuDNN's sm80
  flash attention on this build).
- **Pointers into the middle of an allocation:** mixed's `c[16:48].float()` becomes
  `buffer = "W0", byte_offset = 2048`.
- **Scalars:** u32 and byte parameters.
- **Tensor maps:** the six cuBLAS maps (A, B and C, twice each) inside its packed block, matching the
  hand-written r2 capture.
- **`unresolved` (capture.json):** on this aarch64 box, host libraries are mapped near the
  device VA, so uninitialized host padding in struct parameters (cuDNN's tensor descriptors,
  which carry `"tensor_0"`) shows up as pointer-like words. None was a device pointer in the
  ops above.

## Committed fixtures (good variants, lift today)

| fixture | op | kernel | check | sim vs GB300 |
|---|---|---|---|---|
| `c_cat` | `torch.cat([x, y], 0)`, 8x32 f32 | `CatArrayBatchedCopy_vectorized` (libtorch, binary) | clean, 17 states | bitwise = captured output; `hw` 100/100 |
| `c_cumsum` | `torch.cumsum(x, -1)`, 4x128 f32 | `tensor_kernel_scan_innermost_dim` (libtorch, binary) | clean, 503 states | bitwise = captured output; `hw` 100/100 |
| `c_cute_add` | CuTeDSL add (tools/capture_examples/cute_add.py), 4x128 f32 | `kernel_cutlass_add_kernel_...` (PTX case: CuTeDSL's PTX 9.3, stamped 9.0 for CUDA 13.0's ptxas, and ptxas's SASS) | clean on both front-ends (E-WELLSYNC), 33 states | bitwise = captured output on both; `hw` 100/100 |
| `c_triton_cat` | Triton cat (tools/capture_examples/triton_cat.py) | `cat_kernel` (Triton cubin, `--binary`: the PTX front does not parse `mad.wide.s32` yet; PTX kept as jit.ptx) | clean, 193 states | bitwise = captured output; `hw` 100/100 |

`crates/sim/tests/fixtures.rs::captured_outputs_match_the_gpu` checks every fixture with
`expect` bytes. Each must be clean on every schedule, and its output must equal the GPU's
bit for bit.

## Equivalence of two captured implementations

`fixtures/c_equiv_cat/equiv.toml`: torch's cat (reference) vs. the Triton cat, both reading
the same captured `x`, `y`.

```
candidate c_triton_cat good sass
    L0 bitwise               FAILS at 512 of 512 outputs (first: 0)
    L1 same function         holds (preconditions not all met: for the captured inputs only)
    L2 no less accurate      holds (preconditions not all met: for the captured inputs only)
    untracked: output 0 has no dataflow history (value -1.1258398294448853)
```

- **What works:** the recipe runs end to end. Both kernels are clean and taint-clean, and both
  outputs are equal.
- **What does not, and why:** the numerics shadow gives a value an origin only when an FP
  operation touches it. A pure copy has no rounding node, so its outputs are "untracked
  constants", and L0 compares a leaf on one side with a constant on the other.
- **What equiv still needs:** for data-movement ops, leaves identified through plain loads and
  stores (ROADMAP rung 2e precondition 3).
- **The arithmetic pair:** torch.cumsum vs. Triton `tl.cumsum`
  (tools/capture_examples/triton_cumsum.py, same seeded `x`). It is blocked only on `FSEL`
  in the Triton kernel. Once that lifts, capture it to `fixtures/c_triton_cumsum` and point
  an equiv.toml at `../c_cumsum` and it. L1 should hold, and L0 should fail, because the
  scan orders differ.

## Any process: CUDA_INJECTION64_PATH (2026-11-08)

`python/ptxv/_native/ptxvcap.c`, the library behind `ptxv.capture()`, is also a CUDA
injection library: with `CUDA_INJECTION64_PATH` pointing at it, the driver calls its
`InitializeInjection` at `cuInit`, and it records launches of any process with no Python in
it: C++ gtests and benchmarks, bundled Python executables, Python without `import ptxv`.
Same CUPTI callbacks, same `launches.jsonl` + `snap/` + module images, same case writer.

```sh
# build the library (what ptxv.capture builds on first use), then run any program under it
gcc -shared -fPIC -O2 -o ptxvcap.so python/ptxv/_native/ptxvcap.c -I$CUDA_HOME/include \
    -I$CUDA_HOME/targets/*/include -ldl -lpthread
PTXV_CAPTURE_DIR=$W/cap PTXV_CAPTURE_FILTER=rms_norm CUDA_INJECTION64_PATH=$PWD/ptxvcap.so \
    ./rms_norm_test --gtest_filter=RmsNorm.BFloat16Numerics
# one case per snapshotted launch (the internal wrappers ptxvcap-run / ptxvcap_cases.py do both)
```

| variable | default | |
|---|---|---|
| `PTXV_CAPTURE_DIR` | required | output dir; `%r` becomes `$RANK` (else `p<pid>`), so ranks and child processes write separate dirs |
| `PTXV_CAPTURE_FILTER` | all | POSIX ERE on the (mangled) kernel name |
| `PTXV_CAPTURE_SKIP` | none | names to leave out |
| `PTXV_CAPTURE_INDEX` | `0` | invocation indices to snapshot, counted per kernel name over matching launches: `3`, `0,5-7`, `all` |
| `PTXV_CAPTURE_MAX` | 16 | snapshotted launches in total |
| `PTXV_CAPTURE_RECORD` | `selected` | `matching`: also write unselected matching launches (no snapshot) |
| `PTXV_CAPTURE_MAX_ALLOC`, `PTXV_CAPTURE_BUDGET` | 2G, 32G | largest allocation copied; total snapshot bytes |
| `PTXV_CAPTURE_FATBIN` | 1 | write the fatbin image of a selected launch's library / module (`modules/<fnv>.fatbin`, the launch's `"fatbin"`) |
| `PTXV_CUPTI` | `libcupti.so.13`, then the CUDA 13 toolkit's | |

A capture directory holds `launches.jsonl` (ptxv.capture's format plus `inv`, the invocation
index), `snap/s<N>_{pre,post}_<base>.bin` (every driver allocation that a parameter word, a
struct field or one level of pointer table points into, before and after the launch, with the
context synchronized around it), `modules/<fnv>.cubin` (every image the driver loaded,
exact), `modules/<fnv>.json` (provenance, below), `modules/<fnv>.fatbin`, `process.json`
(argv, pid, rank env) and `summary.json` (written at exit: recorded, snapshotted, matched,
seen, graph launches, skipped snapshots).

A capture directory converts to cases with the same case writer as ptxv.capture
(`ptxv._casewriter.write_case`), given a capture object whose launches come from
`launches.jsonl`, whose `dir` is the capture directory and whose module directory is
`modules/`; no tensor names and no allocator blocks.

- Buffer extents. Without torch's allocator trace, only driver allocations are known
  (`cuMemGetAddressRange`). `--size alloc` takes the whole allocation (exact for one
  `cudaMalloc` per tensor, the usual C++ test). `span` (default) runs from the lowest pointer
  the launch passes into an allocation to its end, which drops the unrelated head of a
  caching-allocator segment. `split` makes one buffer per distinct pointer, up to the next
  one. It approximates the tensors of a segment, but a kernel reading across that boundary
  then faults in the simulator, so rule that out with `alloc` before calling a fault a bug.
- TMA descriptors (`CUtensorMap`, at the top level or inside a parameter struct) are decoded
  by the case writer, as for ptxv.capture. Descriptors stored in global memory stay raw bytes.
- Provenance. When the driver loads a module, ptxvcap writes `<hash>.json` with the shared
  objects on the load's stack and the first one outside the driver, runtime and CUPTI:
  `libcublasLt.so.13.2.2.2`, `libcudnn_engines_runtime_compiled.so.9.17.1`, `libtorch_cuda.so`,
  or the program itself. That names the library a kernel ships in and its version (written
  by ptxv.capture too).
- PTX. A selected launch's fatbin is kept. `ptxvcap_cases.py` extracts its PTX and pairs it
  with two cubins: the one the driver's JIT makes from it (`cuLink*`, on a GPU host) and
  ptxas's. `_ptxsrc` can then match the PTX by machine code. On GB300 that match fails for a
  CUDA 13 build with sm_80, sm_90a, sm_100a and sm_100 SASS + sm_100f PTX and no sm_103: the
  driver loads the **sm_100 cubin** and runs an sm_103 image it derives from it (ELF flags
  0x5006704). Loading the sm_100 cubin alone reproduces the captured image bit for bit.
  Per kernel, in a 380-kernel module: 285 byte-identical to the sm_100 SASS, 71 with the same
  instructions and different scheduling control words, 24 with NOPs inserted (up to 1.48x the
  code; no other instruction changes). JIT-compiling the sm_100f PTX gives different code, and
  sm_100a does not load on sm_103 (CUDA_ERROR_NO_BINARY_FOR_GPU). Such kernels are
  binary-only cases: the captured image is what runs.
- Launches of lazily loaded modules carry no `"module"`, because the image arrives inside
  the launch call. Their module is then the smallest captured image that defines the name.
- One CUPTI subscriber per process: do not combine this with `import ptxv` (ptxv.capture) in
  the same process. If the process already subscribes (a profiler, Kineto tracing), init fails
  with 39; the library then prints a line and the process runs uncaptured.
- Graph launches are counted, not captured (as with ptxv.capture).
- A program whose own dynamic loader does not search the system library path needs the
  driver's libcuda on `LD_LIBRARY_PATH`; a statically linked cudart reports "CUDA driver
  version is insufficient" without it.
- Without names, buffers are `u8` (dtype unknown to a C++ process). For Python processes,
  in-process `ptxv.capture` gives tensor names, dtypes and exact extents from the caching
  allocator; prefer it when the process can import ptxv.

Validation (2026-11-08, GB300): a C++ gtest of hand-written CUDA RMSNorm kernels (8
snapshotted launches of 260 matching: forward, backward, weight-gradient kernels; 7 cases
`clean` with outputs bitwise equal to the GPU, the eighth changed no buffer), a torch script
without `import ptxv` (cuDNN SDPA and a cuBLASLt nvjet GEMM with cluster 8x1x1 and PDL: clean,
bitwise equal), and an embedding-bag workload in a bundled Python executable (CUDA and Triton
table-batched-embedding forward and backward kernels).

## Launch order (2026-11-11)

A case's `[[launches]]` list is one ordered list per GPU, and launch-order exclusions (S-NC's
stream order, E-L1-CORES's kernel boundary) trust it only when the case says
`launch_order = "recorded"` (ASSUMPTIONS.md E-LAUNCH-ORDER). The three capture paths record
what that needs and write only the order CUDA enforces (`python/ptxv/_order.py`):

- Every launch record carries its stream handle, `stream_id` (cuStreamGetId: unique for the
  process; stream 0 resolved through the entry point's flavour, `_ptsz` or legacy), whether
  it was issued into a graph capture (`capturing`, `capture_id`), its ProgrammaticEvent /
  LaunchCompletionEvent events and its memory synchronization domain (the launch's or
  stream's domain through its domain map). On a capturing stream cuStreamGetId and
  cuStreamGetDevice fail (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED) and invalidate the capture
  (GB300, CUDA 13 driver), so they are left out there; the stream is then named by its
  handle within the capture.
- Stream order calls are recorded in order with the launches: cuEventRecord(WithFlags),
  cuStreamWaitEvent, cuGraphLaunch and cuStreamUpdateCaptureDependencies (both versions), both
  flavours, and cuEventDestroy (both versions), cuIpcOpenEventHandle and cuIpcGetEventHandle. probes/launchcap logs them with
  `LAUNCHCAP_ORDER=1` (tools/capture.py sets it; `LAUNCHCAP_VMM=1`, which tools/capture_dist.py
  sets, implies it); ptxvcap (ptxv.capture) writes them to `events.jsonl` with an `oseq` shared
  with `launches.jsonl` (injection mode: `PTXV_CAPTURE_ORDER=1`). A launch or stream order call
  is written once it returned, and only when it succeeded (a failed wait orders nothing), with
  its host thread (`tid`) and the host times just before the call and after it returned (`t0`,
  `t1`, CLOCK_MONOTONIC ns). probes/launchcap keeps the launch being recorded per thread.
- Edges: the previous launch on the same stream (programmatic when the launch carries
  programmatic stream serialization); an event recorded on one stream and waited for on
  another (the waiting stream's next launch after every launch the recording stream had been
  given); a ProgrammaticEvent (a programmatic edge; none with triggerAtBlockStart); none for
  a LaunchCompletionEvent or an event with the external flag; inside a graph capture the same
  rules among the capture's launches (a stream that waits for an event recorded in a capture
  joins it; after cuStreamUpdateCaptureDependencies the stream's next launch has no edge from
  what the stream was given before; a per-thread default stream is each thread's own). A launch issued into a capture has no edge to launches outside it: the graph
  runs when launched, maybe several times, so its kernels are unordered with the program's
  other kernels in the case. An event's record ends at cuEventDestroy (the driver hands the
  handle to the next new event, and a wait on an event never recorded orders nothing) and at
  any graph launch (a graph's event record nodes record events when it runs); an event shared
  with another process (IPC open or export) orders nothing until destroyed, since the other
  process can record it. Launches filtered out (`--kernels`, `select`) still order the kept
  ones through them (full edges, or full edges then a final programmatic one).
- Encoding: `stream` (per GPU, numbered by first use), `after` (the predecessor; omitted when
  it is the previous launch on the GPU on the same stream; -1: none), `pdl` (that edge is
  programmatic), `waits` (further predecessors it starts after they completed),
  `mem_sync_domain`; capture.json `launch_order` lists every launch's predecessors and notes
  each stream switch ("ordered by an event" or "they may overlap"). A single-launch case gets
  none of these lines.
- `launch_order = "unknown"` (with the reason in a comment) when a launch record has no stream
  id or no thread (an older capture library), when launches between the kept ones were not
  written (injection mode with a filter), when the case's launches are not in issue order,
  when the kept launches of one process are on several GPUs (the case has no per-launch
  device), or when calls of several host threads overlap in time: a record whose `t0` is not
  after the `t1` of every earlier record of another thread (the driver may have taken the two
  calls in the other order). Threads that take turns (a backward thread launching while the
  calling thread waits) keep a recorded order. ptxv checks such a case under the strict
  reading: no grid is ordered after another, so any two may run at the same time
  (ASSUMPTIONS.md E-LAUNCH-ORDER; `PTXV_UNRECORDED_ORDER=ordered` keeps the list's order and
  only drops the launch-order exclusions). An older capture whose order other evidence
  establishes, such as a re-capture with the order recorded, can carry
  `launch_order = "declared"` instead, with the evidence in a comment.
- Not recorded, so no edge: host synchronization, the legacy default stream's implicit
  synchronization with blocking streams, stream memory operations (cuStreamWaitValue,
  memcpy), host functions, events recorded before the window, cooperative launches in
  probes/launchcap (cuLaunchCooperativeKernel and cudaLaunchCooperativeKernel are not hooked
  there). A missing edge only adds behaviours, with one exception at the strict level: after a
  wait for a ProgrammaticEvent, an unrecorded operation on the waiting stream (a memset, a
  copy, a host function) does not stop the programmatic edge from reaching the stream's next
  launch, which CUDA orders only after that operation (itself started at the primary's
  trigger), not after the primary's later writes. On GB300 the waiting stream's next kernel
  never started before the primary ended (0 of 100 runs directly after the wait, 0 of 100
  with a memset between; 100 of 100 overlapped without the wait), so there the programmatic
  edge is weaker than the hardware.

The hardware runner (`ptxv hw`, crates/hw `stream_plan`) runs each launch on a non-blocking
stream of its GPU, a programmatic edge to the previous launch on the launch's stream as
programmatic stream serialization (otherwise as a full edge), and `mem_sync_domain` through the
launch's domain map; `ptxv hw` applies the variant's launch-order edits. A launch takes its case
stream's stream only where the stream order that adds is in the case: the stream's latest
launch is its predecessor, or completes before it starts through other edges. Otherwise, as
with `after = -1`, it gets a stream of its own. Every predecessor other than the stream's latest
launch is waited for through an event, an earlier launch of the same stream too, so a GPU run
does not rely on the stream order being transitive through programmatic launches (GB300's is:
a plain launch behind a programmatic one started before that one's primary ended in 0 of 100
runs, although the programmatic launch started early in 99 of 100; CUDA does not state it). A
GPU run then has every order of the case, and more only through programmatic edges run as full
ones, through an event's coverage, and through that transitivity. An event covers all the work
its stream was given before it, so a launch that waits for it also waits for an earlier launch of
that stream which the case does not complete before it: with L0, L1 programmatic after L0 on
stream 0, and L2 on stream 1 after L1, the GPU starts L2 after L0 completed and the case does not
(plain CUDA event semantics; captures write such a primary into `waits`, so this arises in
hand-written lists). Until 2026-11-12, a runner limit of recorded orders, a launch whose
`after` names an earlier launch of its stream, or a launch on another stream, also waited for
its stream's latest launch (captures write that shape too: python/ptxv/_order.py takes the
latest full predecessor as `after` when the stream's previous kept launch is not a
predecessor), and an earlier predecessor on its stream was left to the stream order. Before
2026-11-11 the runner ran every kernel in list order on the NULL stream, so no GPU run could
show an overlap the case allows. GB300, a 2 ms delay before the writer's store: e_launch_order
UNORD (the reader after nothing) differs from the ordered output in 200 of 200 runs, good (an
event edge) in 0 of 200; e_launch_waits NOWAIT 200 of 200, good 0 of 200; e_runner_after good
(the reader after the first writer only) 200 of 200, CHAIN (after the second) 0 of 200. The
runner before 2026-11-11 gave 0 of 200 for the first four; the one before 2026-11-12 gave 0 of
200 for e_runner_after good. Without a delay, the B37 variants of r4c_chunk2, r4c_dwrap and
r4c_nowrap differ from the capture in 199, 200 and 200 of 200 runs (0 of 200 before
2026-11-12); r4b_moe B37 in 0 of 200 (the race does not show at this timing).

Older captures (before 2026-11-11) carry no `launch_order`, so the exclusions do not apply to
their multi-launch cases, and E-PDL still orders their grids as listed: a plain race between
launches that such a capture serialized is not reported (re-capture, or declare the order where
the capture.json below shows it: fixtures/r4b_moe). An older probes/launchcap capture of a
process that launched from several threads at once can also give a launch another thread's
PDL attribute, stream, packed parameter buffer or preferred cluster (the library kept the
launch being recorded in process-wide state): with two threads launching through
cudaLaunchKernelEx at once (GB300, 2000 launches each, 3 runs per runtime linkage), 1091 to
1944 of the 2000 PDL launches were written without PDL, and 877 to 1828 of the 2000 others with
it; with the threads one after the other, none. What an older capture.json can tell:
tools/capture.py's (keys `argv`, `filtered_out`) records nothing about streams; ptxv.capture's
warns "launches on several streams" when the region used more than one stream handle (absent:
one handle, which does not show one stream: a destroyed stream's handle is reused, and 0x0 is a
different stream per thread and entry point); tools/capture_dist.py's has each launch's stream,
each rank's streams and the event records / waits, and its work directory the rank's records
(ASSUMPTIONS.md E-LAUNCH-ORDER states when those suffice to declare the order).

Validation (2026-11-11, GB300, torch 2.12): two streams with an event (k0 on s1 and k1 on s2
unordered, k2 on s2 waiting for k0, k3 on the current stream waiting for both) gives
`after = -1` for k1, `waits = [0]` for k2 and `after = 2` for k3 through tools/capture.py and
ptxv.capture alike (check: clean). The same without the event, where k1 reads what k0 writes,
gives k1 `after = -1` and `check` reports the race; captured with the previous tools, it lists
the launches in API order and is clean with either binary. A graph captured with a fork and a join gives
k1 after k0 and k2 after k1 inside the capture; the eager kernel after the replay is unordered
with the graph's kernels (a race reported between them: the conservative reading above).
tools/capture_dist.py with two ranks of that program gives each rank the same relations,
mapped into the case's interleaving. An event recorded after k0 on s1 and destroyed, then a
new event (given the destroyed one's handle) waited for on s2 before k1 without being recorded:
k1 `after = -1` through both paths and `check` reports the race (before cuEventDestroy was
recorded the wait ordered k1 after k0, clean). One process with a kernel on each of two GPUs:
`launch_order = "unknown"`. Two threads launching through cudaLaunchKernelEx (one with PDL,
one without) at once: every launch keeps its own PDL attribute and stream, and the order is
`unknown` (the calls overlap); the threads one after the other: `recorded`.

## Limits

- Launch order: see "Launch order"; a graph's kernels are unordered with the kernels outside
  their capture.
- Memory operations between kernels are not modeled. Their bytes reach a case through the
  snapshot of the next kept launch; one that changes a buffer between two kept launches that
  reference it refuses the case (MISS 173).
- Memory outside the caching allocator is not known. That includes `cudaMalloc` by a library
  itself, module globals, and host-pinned memory. Pointers to it stay raw bytes, listed as
  `unresolved` when they look like device addresses.
- Live blocks over 4 MB that are not inputs (a cuBLAS workspace from the warm-up) start as
  zeros.
- Tensor maps stored in global memory are captured as raw bytes. Their embedded addresses
  are the capture's, so they are only right if a kernel rewrites them (as r3's does).
- An op whose output aliases a workspace that was reused during the call gets the whole
  block as one `u8` buffer. Only the output tensor's bytes are compared (`expect_offset`).

## Equivalence candidates (2026-11-03)

Captured on GPU 3 for docs/equiv-reassoc.md, each with the inputs of the matching torch fixture
(seeded randn 4x128 f32). All lift; `check` is clean and the simulator equals the GPU bitwise,
except the atomic row sum, whose bits depend on the order of its atomics:

| fixture | source (tools/capture_examples/) | kernels | check |
|---|---|---|---|
| `c_triton_cumsum` | triton_cumsum.py (tl.cumsum) | `cumsum_kernel` | clean, 61 states |
| `c_triton_rowsum_splitk` | triton_rowsum_splitk.py (`--file`: two launches) | `partial_kernel`, `combine_kernel` | clean, 85 states |
| `c_triton_rowsum_atomic` | triton_rowsum_atomic.py (tl.atomic_add: `REDG.E.ADD.F32.FTZ.RN`, now modeled) | `rowsum_atomic_kernel` | `--slices`: nondeterministic (N-ATOMIC), 96 states |
| `c_triton_softmax_online` | triton_softmax_online.py (running max and sum) | `softmax_online_kernel` | clean, 21 states |
