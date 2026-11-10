# NVLS multicast memory (multimem)

NVLink SHARP multicast objects give an address range whose accesses go to one bound buffer
per GPU; the NVLink switch reduces `multimem.ld_reduce` loads and fans out `multimem.st` /
`multimem.red`. ptxv models them for both front-ends. Assumptions: S-MULTIMEM, S-MC-ALIAS,
E-MC-ALIAS-HW, L-ALIAS-FENCE, N-MULTIMEM, N-MULTIMEM-ROUND16.

## Case format

```toml
[[multicast]]
name = "mc"
size = 128                # bytes of multicast address space
binds = [                 # cuMulticastBindMem, cut at case buffers
    { device = 0, offset = 0, buffer = "buf0", byte_offset = 0, size = 128 },
    { device = 1, offset = 0, buffer = "buf1", byte_offset = 0, size = 128 },
]
```

Parameters and pointer-table words name a multicast object like a buffer (`kind = "ptr",
buffer = "mc"`). The simulator gives it an address slot after the buffers; it has no storage.
`mc_alias = "hw"` (or `PTXV_MC_ALIAS=hw`) turns the alias-proxy check off (E-MC-ALIAS-HW).
`ptxv hw` refuses cases with multicast objects.

## Capture

`tools/capture_dist.py` with `probes/launchcap` (`LAUNCHCAP_VMM=1`) records
`cuMulticastCreate`, `cuMulticastAddDevice`, `cuMulticastBindMem` (multicast and memory
offsets), `cuMulticastBindAddr`, `cuMulticastUnbind` and `cuMemMap` of the multicast handle.
A kernel parameter or table word inside a multicast mapping becomes a reference to a
`[[multicast]]` object `mcN`; its binds are the bound allocation ranges intersected with the
owner rank's case buffers (a torch symmetric allocation: the data buffer and the signal pad).
Binds are taken as of the end of the capture; an unbind is warned about.

- Libraries that link libcuda directly (mscclpp: deep_cp's multicast buffer) bypass
  `cuGetProcAddress`; launchcap also interposes the exported `cuMulticast*` and `cuMem*`
  names. Recordings made before that have `mc_create` / `mc_add_device` but no bind, and the
  object comes out with no binds (every access `unmodeled`).
- POSIX-fd exports (`cuMemExportToShareableHandle` type 1) on GB300 all report one anonymous
  inode (`fd:6:910`), so the inode does not name the allocation. Such imports are matched to
  exports by time: in time order, each import takes the earliest unmatched earlier export of
  another rank whose kind (memory / multicast) and size fit the importer's later use. In the
  deep_cp / dist_gemm recordings every match had one candidate with 2 ranks; with 4 ranks
  149-168 of 259 had several, so 4-rank peer pointers through such imports are a guess (the
  capture prints the counts). Fabric handles are exact.
- Each rank's launch takes the cubin from its own rank's captured images: CuTeDSL compiles
  the rank id into same-named kernels (dist_gemm's two ranks differ), and one shared image
  made every rank reduce rank 1's rows.
- 128-byte CUtensorMaps in packed parameter blocks (CuTeDSL's grid-constant maps) become
  `tensormap` parameters / fields, decoded as in `tools/capture.py`.

## Lowering (ptxas 13.0, sm_103a; `probes/multimem/forms_sass.txt`)

| PTX | SASS |
|---|---|
| `multimem.ld_reduce[.weak/.relaxed/.acquire].<scope>.add.f32` | `LDGMC.E.ADD.F32.RN.STRONG.SYS` (always `.STRONG.SYS`; acquire adds `CCTL.IVALL`) |
| `... .add.v4.f32` / `.v4.bf16x2` / `.acc::f32.v4.bf16x2` | `LDGMC.E.ADD.F32x4.RN` / `ADD.BF16x8.RN` / `HPADD.BF16x8.RN` |
| `... .min/.max.bf16x2`, `.add.u32`, `.min.s32`, `.xor.b32`, `.add.u64` | `LDGMC.E.MIN.BF16x2.RN`, `ADD.32`, `MIN.S32`, `XOR.32`, `ADD.64` |
| `multimem.st.relaxed.sys.v4.f32` | `STG.E.128.STRONG.SYS` (weak: `STG.E.128`; release: `MEMBAR.ALL.SYS` first) |
| `multimem.red.release.sys.add.s32` | warp-aggregated `REDG.E.ADD.S32.STRONG.SYS` after `MEMBAR.ALL.SYS` |
| `multimem.red.relaxed.sys.add.f32` / `.v4.bf16x2` | `REDG.E.ADD.F32.FTZ.RN.STRONG.SYS` / `REDG.E.ADD.BF16x8.RN.STRONG.SYS` |
| `fence.proxy.alias` | `LDCU URx, c[0x0][0x2d8]; UP2UR.B3`; bit 0 of URx set: `MEMBAR.SC.GPU`, clear: `MEMBAR.SC.SYS; CCTL.IVALL`; `UR2UP UPR, URx.B3` (the word is 0 on GB300, `results/c2d8.txt`) |

Rejected by ptxas: `min`/`max` on `f32`, `multimem.red` `min`/`max` on floats, `.inc`.
In SASS a multimem store or reduction is an ordinary `STG` / `REDG` whose address is in a
multicast range; the simulator decides by address.

## Semantics

- A multimem access is one access per bound GPU, each with the instruction's semantics and
  scope, at one step (S-MULTIMEM). Race detection and release / acquire act per location.
- PTX front: a non-multimem access to a multicast address, or a multimem operation on another
  address, is undefined behavior: `violation`. SASS front: an access to a multicast address
  other than `LDGMC` / `STG` / `REDG` is `unmodeled`.
- Alias proxy (S-MC-ALIAS): an access through one view (multicast / unicast) of a byte last
  written through the other is a `race:alias` unless a `fence.proxy.alias` or a kernel
  boundary lies on the happens-before path from the write. Strong accesses (flags) are not
  exempt (PTX ISA 8.7: moral strength needs the same proxy): a strong access through one view
  after the same thread's strong access through the other view needs a `fence.proxy.alias`
  between them, and a strong access observing another thread's strong write through the
  other view needs the observer's own earlier `fence.proxy.alias` or one on the
  happens-before path from the write. GB300 hardware level (`mc_alias_hw`): `undecided`.

## GB300 measurements (`probes/multimem`, 4 GPUs in one NVLink domain, 2026-11-06)

`MULTICAST_SUPPORTED = 1` on the one GPU probed on a second host and on all 4 GPUs of a
4-GPU node; granularity 2 MiB; `cuMulticastCreate` with one device fails
(`CUDA_ERROR_INVALID_VALUE`). Multi-GPU runs: the 4-GPU node (samples in `results/`).

- **Determinism.** `multimem.ld_reduce` results are bitwise identical across the issuing GPU,
  4 repetitions, two processes and a permuted member-to-GPU mapping: a symmetric,
  schedule-independent function of the bound values.
- **Integers**: exact. **f16 / bf16 min / max**: minNum / maxNum, -0 < +0 (0 mismatches in
  524288 lanes each).
- **f32 add**: no fixed summation order fits. The result is the window-truncated exact sum
  rounded to nearest even: 63 bits below the largest operand's last bit reproduce 262094 of
  262144 scalar lanes; every miss has operand bits below the window (mostly ties), and the
  model flags such lanes.
- **f16 / bf16 add** (`.acc::f32` or not): always one of the two neighbours of the sum, not
  round-to-nearest. The direction depends on the other three lanes of the 8-byte group: up
  when frac + r >= 1, r the interleaved truncated mantissas of lanes l+1..l+3. Exact on the
  controlled sweeps, 99.7% (bf16) / 96.7% (f16) of random lanes. Outputs are compared
  within one ulp when this rounding is involved (N-MULTIMEM-ROUND16).
- **NaN**: one NaN operand propagates quieted; inf + -inf gives the canonical NaN.
- **multimem.red**: concurrent `add.s32` from 4 GPUs: exact counts in every member; concurrent
  `add.f32` (8 adds per GPU): each of 1200 sampled member values equals the sequential sum in
  some GPU-major arrival order.
- **multimem.st**: every member receives the bytes.
- **Message passing** (`results/litmus.txt`): `multimem.st` data + `multimem.red.release.sys`
  flag, readers `ld.acquire.sys` on their unicast flag and data: 0 stale reads in 200 x 32768 x
  3 without `fence.proxy.alias`; with relaxed red / poll: stale reads in 200/200 runs.
- **Reverse message passing** (`results/litmus2.txt`, the dist_gemm RS shape): unicast
  `st.global` or TMA-store data, `multimem.red.release.sys` flag, `ld.acquire.sys` poll, then
  `multimem.ld_reduce`: 0 wrong lanes (200 runs x 131072 lanes, 4 and 2 GPUs) without the
  alias fence; the controls (relaxed; `wait_group.read` only) were 0 as well, so the probe
  does not demonstrate sensitivity.

## Verdicts

- Strict (default): an alias-proxy race is a `race`; the GB300 level classifies it
  `undecided` (the hardware model has no alias proxy) and points at `mc_alias = "hw"`.
- `mc_alias = "hw"` (E-MC-ALIAS-HW, argued): the views are one location; validated only for
  the multicast-write / unicast-read shape.

## Fixture

`fixtures/r1h_nvls`: 2 GPUs, stage through the unicast buffer, `fence.proxy.alias`, barrier by
`multimem.red.release.sys` on the multicast flag with `ld.acquire.sys` polls, then
`multimem.ld_reduce` of both ranks. Mutants: B23 (no alias fence), B63 (red at `.gpu`), B64
(relaxed red), B65 (plain store to the multicast address).
