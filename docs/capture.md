# Capturing any torch op or generated kernel as a ptxv case

**Status (2026-11-03):** `tools/capture.py` turns a torch expression, a Python function, or a
Triton / CuTeDSL kernel into a ptxv case: every kernel launched, in order, with its cubin(s),
its parameters mapped to named buffers, the exact input bytes, and the GPU's output bytes. 33
ops captured in the smoke test; all 33 captures worked. 4 of them lift on today's lifter:
cumsum, cat, a Triton cat, and a CuTeDSL add. The CuTeDSL add lifts on both front-ends. For
those 4, `check` is clean and the simulator's output equals the GPU's bit for bit. The
others are blocked on missing SASS instructions, listed below.

One kernel out of a sequence (rung 3b): with `--kernels`, the launch recorder snapshots the
kept launches' memory. A kept kernel then starts from exactly what earlier, filtered-out kernels
wrote, and its outputs are what it changed (see "Snapshots" below; fixtures/r3b_trtllm_bmm is
gemm1 out of flashinfer's 5-launch MoE).

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
- **Initial contents.**
  - Allocations made during the call start as zeros, which a correct kernel does not read.
  - Device-to-device `cudaMemcpy` / `cudaMemset` before the first kernel are folded into the
    initial contents. That covers an op's `clone()` or zero fill: `x.index_add(...)` copies
    `x` into the output before its kernel. Such operations *between* kernels cannot be
    expressed in a case and are warned about (the sort op has one).
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
  - Only launches recorded before they are issued are covered: driver launches and
    `_cudaLaunchKernelEx`. Runtime `<<<>>>` launches are recorded after the fact and get no
    snapshot.
  - `capture.py` passes `--kernels` as the regex. Buffers start from the first kept launch's
    pre-snapshot.
  - The op's own outputs are replaced: every buffer whose bytes the kept launches changed is an
    output, with the last kept launch's post-snapshot as `expect`.
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

## Limits

- One stream. Work on other streams, and events, are not recorded.
- Memory operations between kernels are not modeled (warned).
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
