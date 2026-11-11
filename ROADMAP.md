# ROADMAP: from one mbarrier to a 4-GPU MoE layer

Decisions taken 2026-10-30 (they amend docs/design.md §2, §4, §9, §11):

1. **Binary first, one IR.** ptxv reads the compiled cubin (SASS) and lifts it into the same
   IR the PTX front-end produces. Verification runs on the SASS lift whenever a cubin exists,
   because that is what executes. When PTX also exists (JIT sources), we verify both and
   require identical verdicts. A disagreement is a lifter bug or a `ptxas` bug, and both are
   worth knowing. Reports show a readable **pseudo-PTX** rendering of the lift: this is the
   "decompiled to PTX" view. It is for humans and cannot be re-assembled in general.
2. **No source inspection, ever.** Inputs are binaries plus captured launches (`.ptxcap`).
   Library source (CUTLASS headers, DeepEP .cu) may be used to *build mutants* for the bug
   suite, never by the verifier.
3. **Exhaustive at the captured grid** is the default verdict, made tractable by decomposition
   (below). A shrunk grid is only a fallback, and it must pass a coverage-equivalence check
   against the full grid.
4. **Blackwell only**, on the GB300s here. **GPU policy:** never touch a GPU that has a compute
   process ptxv did not start. `hw/` checks `nvidia-smi --query-compute-apps` before every run
   and picks from the free set; a busy GPU is never shared. As of 2026-10-30, one of the four
   GPUs runs another team's server, so ptxv uses the other three. Multi-GPU rungs
   run at 3 ranks, plus 4 ranks whenever the fourth GPU is free. Verification itself is CPU-only and
   can use any rank count.
5. **Shared memory across GPUs = NVLink (single node) for now**, designed to extend to
   multi-node (section below).

## Why binary-first is feasible

Evidence from this box: every real shipped kernel is SASS-only. That covers torch 2.12
`libtorch_cuda.so` (445 sm_100a + 445 sm_103a ELFs, 0 PTX), DeepEP (`deep_ep_cpp`, 0 PTX) and
flashinfer's 7564 prebuilt trtllm-gen cubins. A PTX-only tool could not look at any of them.

`nvdisasm` (CUDA 13.0, plus pip `nvidia-cuda-nvdisasm` 13.3) disassembles sm_100/103 cleanly,
and the Blackwell async machinery is visible as named opcodes. Example: the trtllm-gen MoE
GEMM `Bmm_E4m3_E4m3E4m3_Fp32_t128x32x256u2_s5_…_dynBatch_sm100f.cubin`, 1.4k SASS lines:

| SASS | PTX meaning |
|---|---|
| `UTCQMMA gdesc[..], gdesc[..], tmem[..], tmem[..], idesc[..]` | `tcgen05.mma.kind::f8f6f4` |
| `UTCBAR` | `tcgen05.commit` → mbarrier |
| `UTCATOMSWS.FIND_AND_SET.ALIGN` | `tcgen05.alloc` |
| `LDTM` / `STTM` | `tcgen05.ld` / `st` |
| `UTMALDG.3D`, `UTMALDG.2D.GATHER4`, `UTMASTG.4D`, `UTMACMDFLUSH` | TMA load / gather4 / store / bulk commit |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [mbar], parity` | `mbarrier.try_wait.parity` |
| `SYNCS.ARRIVE*`, `NANOSLEEP.SYNCS` | mbarrier arrive / suspend hint |
| `FENCE.VIEW.ASYNC.S` | `fence.proxy.async.shared::cta` |
| `ELECT`, `USETMAXREG`, `WARPSYNC`, `BSSY/BSYNC`, `ENDCOLLECTIVE` | `elect.sync`, `setmaxnreg`, convergence |
| `MEMBAR.{CTA,GPU,SYS}`, `ACQBULK`, `DEPBAR` | fences / async-group waits |

**The lifter is built from a "Rosetta" corpus, not from guesses.** We generate thousands of
small PTX probes covering every opcode × modifier we care about, compile them with `ptxas
-arch=sm_100a/sm_103a`, align the PTX with the resulting SASS, and derive the SASS → IR table.
Every lifted semantics rule is then validated by differential execution on GB300. An unknown
SASS opcode or modifier is a **hard error** naming the instruction and is never skipped.

SASS-specific modeling: uniform registers and predicates; convergence barriers (`BSSY/BSYNC`,
`WARPSYNC`) as reconvergence hints under independent thread scheduling; `ENDCOLLECTIVE`;
`DEPBAR` mapped to async-group waits. Register scoreboards (the control bits that make a
variable-latency load complete before its use) are **verified, not trusted** (done
2026-10-30, [docs/sass-ctrl.md](docs/sass-ctrl.md)). Each instruction's control word is
decoded, and a CFG dataflow check proves that no register is read or overwritten while a
variable-latency operation on it may be outstanding. The decoding is validated by a one-bit
mutation sweep on GB300 with 0 misses. This replaces "trust ptxas" in docs/design.md §9 with the
narrower "trust nvdisasm + the scoreboard model + stall counts". `ptxas` reorderings and fma
contraction become visible and verified rather than modeled as nondeterminism.

## Exhaustive at the captured grid: how

A persistent grouped GEMM on 4 GPUs has ~148 CTAs × 4 GPUs × ~12 warps. Flat interleaving is
hopeless, but the structure is very regular. The verdict is assembled from three exact pieces:

* **A. Intra-CTA / intra-cluster skeleton**, explored exhaustively per *behavior class*. Two
  CTAs are in the same class when their skeleton traces agree. For grouped GEMM the class is
  the sequence of (k-iterations, empty-group, edge-tile) events in its work list, not the
  group or tile ids. Classes are computed for all CTAs at full grid and deduplicated, typically
  to a few dozen.
* **B. Inter-CTA / inter-GPU objects:** tile-scheduler atomics, CLC responses, cross-GPU
  flags and counters, peer buffers used as mailboxes. These are explored exhaustively with
  each CTA abstracted to its interface trace from A, using symmetry and partial-order
  reduction.
* **C. Footprints for the full grid, all GPUs:** every pair of epochs that A+B leave unordered
  gets a write-overlap sweep over physical addresses, including peer memory and multicast
  aliases.

This is complete for the captured grid when every inter-CTA channel is in B. That holds by
construction: anything not in B is ghost memory, and C reports any unordered access to it as a
race. Data-dependent inputs (group sizes, routing) are fixed to the capture, plus a
**group-size sweep** mode. It enumerates edge-case vectors (0, 1, tile−1, tile, tile+1, a
mixture, all tokens on one expert), because grouped-GEMM bugs live at group boundaries.

## The ladder

Each rung has an exit criterion, and the bug suite (`BUGS.md`) grows with every rung. Nothing
advances until its mutants are caught.

### Rung 0: one TMA, one mbarrier (the "something simple") ✅ done 2026-10-30, see [docs/rung0.md](docs/rung0.md)
A fixture kernel we write directly in PTX: 1 CTA, a producer warp `elect`s one lane that does
`mbarrier.arrive.expect_tx` + `cp.async.bulk.tensor.2d` of a 64×64 bf16 tile; consumer warps
`try_wait.parity 0`, sum the tile in fp32 and store it.
* Builds the complete vertical slice: PTX parser, SASS lifter (on the `ptxas` output of the
  same kernel), IR, `sim`, mbarrier + TMA + async proxy, HB, deadlock and race detection,
  `ptxv-capture`, `hw/` runner, witness replay.
* Exit: identical verdicts from the PTX and SASS front-ends; `sim` output bitwise equal to the
  GPU; bugs B01–B03 caught; B01's replay hangs on hardware (watchdog-killed) and B02's replay
  produces a wrong sum.

### Rung 1: feature ladder (fixtures, one feature per kernel)
*1a ✅ done 2026-10-30 ([docs/rung1a.md](docs/rung1a.md)): tcgen05 tile bitwise = GB300; tensor-core
arithmetic calibrated (unique model out of 820).*
*1b ✅ done 2026-10-30 ([docs/rung1b.md](docs/rung1b.md)): warp-specialized TMA↔tcgen05 pipeline with
TMA-store epilogue, exhaustive on both fronts via DPOR (481 states), bitwise = GB300; B04/B09/B10/B11/B50
caught, every GPU failure predicted.*
*1c(i) ✅ done 2026-10-30 ([docs/rung1c.md](docs/rung1c.md)): 2-CTA cluster with multicast TMA, remote
mbarrier arrive, `barrier.cluster`; both fronts exhaustive, bitwise = GB300; B13/B14 caught.*
*1c(ii) ✅ done 2026-10-30: `cta_group::2` pair MMA, multicast commit, 2-SM TMA; the SASS front models
ptxas's hidden paired-alloc handshake (driver-initialized reserved smem, probed); bitwise = GB300; B12
caught (GB300 hangs).*
*1d ✅ done 2026-10-30 ([docs/rung1d.md](docs/rung1d.md)): CLC persistent scheduler with lazy CTA launch
and cancellation modeled from a GB300 probe; both fronts; B17/B53/B54 caught and reproduced on GB300 at
8192 CTAs.*
*1e ✅ done 2026-10-30 ([docs/rung1e.md](docs/rung1e.md)): multi-kernel launches (one machine, grids
concatenated); PDL trigger/wait modeled from a GB300 probe; B18/B55 caught on both fronts; B18 reproduced
on GB300 by witness delay; the non-PDL control is clean.*
*1f ✅ done 2026-10-30 ([docs/rung1f.md](docs/rung1f.md)): two GPUs, flag ping-pong over NVLink peer
memory; PTX scope model (moral strength, release/acquire, scoped fences) and flag-poll loops; B20 wrong
1000/1000 on GB300 (stale L1), B19 latent (20,000 runs), B56 SASS-safe.*
*1g ✅ done 2026-10-30 ([docs/rung1g.md](docs/rung1g.md)) on **3 GPUs** (the fourth is busy): one-shot all-reduce
over symmetric memory, two calls; replica-equality check (N4); both fronts exhaustive (SASS good: 3.3M states), bitwise = GB300; B21/B22/B24
caught; B24 diverges on GB300 every run. Multicast/NVLS: 1h below.*
*1h ✅ 2026-11-06 ([docs/nvls.md](docs/nvls.md)): NVLS multicast objects (case `[[multicast]]`, capture), `multimem.ld_reduce` / `.st` / `.red` and SASS `LDGMC` / `STG` / `REDG`, alias proxy (S-MC-ALIAS), GB300 numerics (N-MULTIMEM); `fixtures/r1h_nvls` with B23/B63/B64/B65.*
1a tcgen05 single-CTA tile (alloc → mma → commit → ld → dealloc) · 1b S-stage TMA↔MMA pipeline,
persistent over T tiles · 1c 2-CTA cluster (`cta_group::2`, multicast TMA, multicast commit,
final cluster barrier) · 1d CLC persistent scheduler · 1e PDL pair of kernels · 1f **2-GPU**
flag ping-pong over NVLink peer memory · 1g **4-GPU** one-shot all-reduce over symmetric
memory (+ multimem variant if the box exposes multicast; probe in rung 0).
* Exit: every fixture has bitwise sim/GPU conformance in both front-ends, and all of its
  mutants (B04–B24) are caught; the MMA calibration for kind::f16 and kind::f8f6f4 has
  passed (docs/design.md §5.2).

### Rung 2: real dense GEMM, single GPU (PTX + SASS available)
CuTeDSL sm100 persistent dense GEMM (the CuTeDSL bundled with flashinfer / `nvidia-cutlass-dsl`
4.6.1) and the Triton 3.5 persistent tcgen05 matmul. Both are JIT, so capture gets PTX *and*
cubin. This is the first real-world cross-check of the lifter.
* Exit: clean verdict at the captured grid; PTX verdict == SASS verdict; bitwise conformance;
  PTX-level mutants of these kernels caught.
* **Triton: done (2026-10-31, docs/rung2.md).** Exhaustive on both front-ends at the
  captured grid, bitwise = GB300, and 4/4 mutants caught. As compiled, Triton's PTX breaks the
  contract (T1: no tcgen05 thread-sync fences; T2: descriptor version bits), and this ptxas
  compiles it safely, so the verdicts differ: PTX [race, violation], SASS [violation]. With
  both fixed (`fenced`), PTX and SASS are both clean.
* **CuTeDSL: done (2026-10-31, docs/rung2.md).** Warp-specialized (TMA, MMA, 4 epilogue
  warps). PTX = SASS = GB300 bitwise (= Triton's output). SASS clean, exhaustive at 1 and 2
  tiles per CTA. PTX: as compiled, it breaks the contract three ways (C1–C3), all compiled
  safely. `fenced` is clean, exhaustive at 1 tile and bounded at 2. 4/4 mutants caught; 3 fail
  on GB300. It needed a synchronization reduction and a parallel explorer.
  - Open: the 2-tile PTX is past 20M states per leader policy.
  - Wrapping capture (`r2_cute_wrap`, 2-stage ring, 1 accumulator buffer): SASS clean,
    exhaustive at 6.5M states. Parity mutants CM5 and CM6 are caught, and fail on GB300. Its
    PTX is open.

### Rung 3: real grouped GEMM, single GPU (binary only)
* **3a** `torch._grouped_mm` bf16 (CUTLASS sm100 array GEMM inside `libtorch_cuda.so`, SASS
  only; device-side tensormap updates per group).
* **3b** flashinfer trtllm-gen `batched_gemm` cubins: fp8 block-scaled, `GATHER4` TMA, routed
  "dynBatch". Prebuilt and source-free, the purest binary target.
  - Done so far (docs/lowp.md): narrow-format cvt, `kind::f8f6f4` and the block-scaled kinds
    calibrated on GB300; `fixtures/r3_cute_mxf8`, a CuTeDSL MXF8 block-scaled GEMM, matches
    GB300 bitwise through both front-ends. B30 is caught by the exact spec; B31 needs a
    kernel with software promotion.
  - **trtllm-gen gemm1: done (2026-11-04, docs/rung3b.md).** flashinfer 0.5.3's fp8
    block-scale MoE gemm1 (`..._dsFp8_..._bN_tma_..._dynBatch`, 13 warps, gather4 of routed
    tokens, per-128-K software promotion, e4m3 output with per-token amax scales), captured
    alone out of the 5-launch MoE with launchcap snapshots. Whole grid certified (E-WELLSYNC),
    contracts only (E-BULK-EXIT, E-TC-EXIT); = GB300 bitwise (captured bytes, 1000 runs);
    numerics claim (exact reference, e4m3 half-ulp) and dependency claim tested; B25, B27,
    B28, B30 caught (B26, B29, B31 do not apply, BUGS.md). Partial tiles read unwritten
    smem (T3b, benign).
  - **The whole MoE call: done (2026-11-04, docs/rung3b.md).** Routing (8-CTA cluster,
    generic DSMEM through `SR_SWINHI`), gemm1, activation, gemm2 and finalize as one 5-launch
    case, PDL included (`fixtures/r3b_trtllm_moe`). Lifted and validated on GB300
    (`fixtures/s_routing_forms`): `F2FP.SATFINITE.BF16.F32.PACK_AB`, `HSETP2.BF16_V2`,
    `ATOMS.POPC.INC`, the generic shared window (S-SWIN).
    - Finding T3b-2: gemm2 frees TMEM after half its epilogue (a race, benign on GB300).
      With that barrier fixed, all five launches are certified by one run, contracts only.
    - = GB300 bitwise, 1000/1000.
    - Stage-wise numerics claims hold, as do the end-to-end claim (exact f64 MoE) and the
      routing dependency claim. The output does not depend on the atomic lane order.
    - B70-B72 are caught.
    - gemm1 at K = 2048 (stages wrap) is certified, with B73.
* **3c** CuTeDSL grouped GEMM (JIT, PTX available) as the cross-check for 3a/3b.
* Exit: exhaustive verdict at the captured grid + group-size sweep for all three; mutants
  B25–B31 caught; numerics N0/N1 and the **dependency claim** (output (g,i,j) depends exactly
  on A_g[i,:] and B_g[:,j]) proven for 3a/3b.

### Rung 2e / 3e: equivalence, a reference kernel vs. a candidate kernel
*Requested 2026-10-30; built after rung 2, and extended to grouped GEMM in rung 3.* Goal:
given a reference kernel R (e.g. cuBLAS/CUTLASS SASS) and a candidate C (e.g. a generated
kernel, as PTX and/or SASS), **prove** that C computes "the same thing" as R under a stated
numerical definition. The question is "is this generated kernel a drop-in replacement?", and
the answer should be a proof plus a certificate, not a test-pass rate. Input:
`equiv.toml` naming both cases, the output correspondence (identity, or a layout map), the
input class (shapes / group sizes / dtypes, plus value ranges if needed), and the level
required.

**Preconditions (checked, not assumed):**
- each kernel is race-free and deadlock-free at its own launch, so it is deterministic: one
  output per input on every schedule (what `check` already proves);
- **control flow and addresses do not depend on input data** (a new taint check). Then one
  symbolic execution covers *all* inputs of that shape. If they do depend on data (grouped
  GEMM group sizes, MoE routing), the proof is per configuration and sweeps the class, as in
  rung 3;
- leaves are identified by origin (buffer, element index). Today `num` records only the
  value; this is also needed for the rung 3 dependency claim.

**Levels, strongest first.** Each level is a theorem for all inputs in the class:

| level | claim | how it is proven | typical use |
|---|---|---|---|
| **L0 bitwise** | C(x) = R(x) bit for bit, for every x | the canonical rounding DAGs are identical: hash-consing, with operands sorted at commutative nodes, and each MMA node as (accumulator, multiset of products), which the calibrated model makes order-invariant within a K-group | refactors, tiling or schedule changes that keep reduction order |
| **L1 same function** | exact(C) = exact(R): ignoring rounding, the same real function (e.g. the same Σₖ aᵢₖbₖⱼ, any order) | canonical polynomials over input leaves, compared exactly (coefficient multisets); Schwartz–Zippel with exact arithmetic as a fast path | catches real logic bugs: wrong index, missing term (B28), double count |
| **L2 no less accurate** | L1, and C's worst-case error bound is ≤ R's for every input: per product term, C's error multiplier (rounding nodes on its path and their ε) is ≤ R's | per-leaf comparison of the a-priori bound (Cor 1a structure), which is monotone in \|terms\|, so it holds for every x | **generated kernels**: "as accurate as cuBLAS", with a certificate |
| **L3 tolerance τ** | L1, and \|C(x) − R(x)\| ≤ τ(x) for every x, with τ user-chosen (k·u·Σ\|aᵢₖbₖⱼ\|, or N ulps of the exact value) | the bounds of C and R summed and compared against τ symbolically; plus N1/N2 bias claims (RZ accumulators) where needed | different precision or algorithm (split-K, fp8 with promotion) |
| L4 empirical | within τ on sampled distributions and on hardware | runs on the simulator and GB300 (H-RUNS) | evidence only, never reported as a proof |

**Failure output:** the first level that fails, the output elements and terms where the
canonical forms differ, and a **numerical witness**: inputs built to make the difference
large (e.g. cancellation inputs for a changed summation order, or a single non-zero term for
a missing or extra term). The witness is replayed on GB300, so "not equivalent" comes with a
concrete input on which the two kernels really differ.

**Assumptions this adds** (to be registered in ASSUMPTIONS.md when built): the arithmetic
models of both kernels (N-MMA for every MMA kind involved, N-FADD); the taint check's
coverage; and the input class (claims hold only for the shapes/configurations proven).

* **Status (2026-11-02): done** ([docs/rung2e.md](docs/rung2e.md)): Triton = CuTeDSL at L0;
  cuBLAS L2+L3 (τ = 576u) on the captured inputs, all proven (cuBLAS cluster 0 explored
  exhaustively with the phase-aware reduction; E-WELLSYNC certifies all 32 clusters).
* Exit (rung 2e):
  - CuTeDSL GEMM vs. Triton GEMM vs. the cuBLAS SASS for the same shape. L1 proven between
    all of them; L0 where tiling matches; L2/L3 results reported;
  - a generated-kernel loop demo: three mutated candidates (one reordered reduction, one
    with a dropped K-slice, one with an RZ epilogue) come back as L0-fail/L1-pass,
    L1-fail with a hardware witness, and L2-fail respectively.
* Exit (rung 3e): grouped GEMM candidates checked against `torch._grouped_mm` across the
  group-size sweep.

### Rung 4: real multi-GPU grouped GEMM (4 GPUs, the target)
* **4a** torch symmetric-memory one-shot / two-shot all-reduce (SASS in libtorch): `.sys`
  scope, peer flags, buffer reuse across calls, cross-rank bitwise consistency (N4).
  *Status 2026-11-03 ([docs/rung4.md](docs/rung4.md)): captured from `torchrun
  --nproc-per-node 3` (GPUs 0,1,3) by `tools/capture_dist.py` (fabric-handle peer mappings,
  device pointer tables, multicast recorded); one call exhaustive and clean for both kernels,
  one-shot bitwise = GB300 on every rank; T4a: the one-shot's closing barrier is release-only
  (strict-PTX race across calls; clean under the opt-in E-EXIT-FENCE); B57–B62 caught, B61/B62
  hang on GB300.*
* *4b partly done 2026-11-03 ([docs/rung4.md](docs/rung4.md)): DeepEP intranode dispatch →
  torch gather + CUTLASS grouped GEMM + index_add → DeepEP combine on 2 GPUs (DeepEP has no
  3-rank kernels, 4 needs the busy fourth GPU), 28 launches as one launch group; one deterministic run
  bitwise = GB300 on both ranks (and GB300 1000/1000), dependency claim tested, B32–B40 caught
  (B34/B35/B36 hang on GB300); T4b: DeepEP's relaxed barrier polls and shared tail relay race
  under strict PTX (opt-ins E-POLL-ACQ, E-SMEM-CUMUL; GB300 litmus 2026-11-04: E-POLL-ACQ
  refuted, E-SMEM-CUMUL validated, docs/poll-acq-litmus.md). Exhaustive verdict open (multi-node).*
* **4b** **Expert-parallel MoE layer**, all real binaries: DeepEP intranode `dispatch` (NVLink,
  3–4 ranks) → grouped GEMM (rung 3 kernel) → DeepEP `combine`. It is verified as one *launch
  group* spanning 4 GPUs × 3 kernels, with captured stream/event/PDL dependencies. This
  includes the cross-kernel races (the GEMM reading token counts from dispatch), cross-GPU
  buffer reuse between layers, and end-to-end numerics: output token t depends exactly on
  its top-k experts, and the combine order is consistent across ranks.
* **4c** (stretch) flashinfer/trtllm MoE all-to-all or another fused comm+GEMM kernel, if one
  can be captured on this box.
* Exit: exhaustive verdict at the captured grid on all 4 GPUs, under `torchrun
  --nproc-per-node 4` captures; multi-GPU mutants B32–B40 caught; at least one B3x replayed
  on hardware.

## Multi-node extension (designed in, not built yet)

Nothing in the core assumes one node. The verifier sees a set of grids, a physical memory map
and async engines, and multi-node adds entries to each:

* **Multi-node NVLink (MNNVL, e.g. GB300 NVL72 racks): almost free.** Peer memory is still
  plain `ld/st/atom` on fabric-mapped addresses, under the same PTX model at `.sys` scope.
  The work is in capture: record fabric-handle exports/imports (`CU_MEM_HANDLE_TYPE_FABRIC`,
  multicast objects spanning nodes) and merge per-node `.ptxcap`s by launch group. Checks,
  HB rules and the bug suite carry over unchanged.
* **Network transports (NVSHMEM IBGDA, DeepEP internode/low-latency): a new async engine.**
  The GPU writes work-queue entries and rings a NIC doorbell (MMIO), data lands in remote
  memory asynchronously, and completion is observed by polling a CQ or a signal word.
  Ordering comes from **verbs/RC-QP semantics**, not from the PTX model. We model this as a
  **NIC actor** per QP, with a semantics-table row for each operation: put, put-with-signal,
  atomic, fence, quiet. It is the same mechanism as the TMA/tcgen05 engines: a pending-op
  queue, nondeterministic completion, HB edges only where verbs guarantee them. The SASS lift
  sees the WQE writes and doorbell stores, and the capture records QP/MR registrations.
  NVSHMEM's host-proxy transport becomes a CPU-thread actor.
* **Built (2026-11-06, docs/nic.md):** the NIC actor for IBGDA queue pairs (mlx5 WQEs as
  DOCA GPUNetIO posts them: RDMA write, fetch-add / compare-swap, CQEs), NVLink domains,
  capture hooks in `tools/capture_dist.py`, fixture `r5a_ibgda_putsig`, GB300 + ConnectX-8
  probes (probes/ibgda). NVSHMEM's host-proxy transport and BlueFlame doorbells are not
  modeled.
* **New bug classes** to add then: missing `quiet` before signal across nodes, reusing a send
  buffer before local completion, assuming put ordering on different QPs, CQ overrun.
* **Validation** needs a real multi-node setup (a Slurm cluster) for the litmus campaign
  (docs/design.md §7.2). That is the one part this box cannot do.

Numerics are unaffected: the DAG just spans more ranks.

## First concrete steps (rung 0)
1. Toolchain: stable Rust and Cargo with crates.io dependencies. Workspace skeleton from docs/design.md §8.
2. `hw/`: minimal driver-API FFI. Load a cubin or PTX, encode a tensor map, launch with a
   cluster attribute, run a watchdog for hangs.
3. Rung 0 fixture PTX + its three mutants; `ptxas` → cubin → `nvdisasm`.
4. Parser and lifter for exactly the opcodes rung 0 uses. `sim` with mbarrier/TMA. HB +
   deadlock/race checks. Run, compare with the GPU, replay.

## Open: undecided DeepEP questions need reductions, not hardware (2026-11-05)

Multi-node Slurm runs (8 ranks x 48 cpus x 4G) showed both are limited by the
state space, not memory:
- **Combine (r4b group 6):** 1.6M states, no terminal state; about 85 enabled actions per state that
  the sync reduction does not cut (BAR.SYNC, a global strong poll, strong-model STS/LDS polls);
  one run is about 2100 steps deep. Needs the shared-memory relay rule (docs/rung4.md) or a sync
  reduction covering strong-model STS/LDS polls.
- **notify_dispatch minimum residency (r4b group 2):** R=1 passed 2.4e8 states without saturating,
  because `sync_por_state_ok` disables the sync reduction while a CTA is unlaunched
  (E-RESIDENCY). Needs a launch-aware sync reduction (slot / exit dependencies), with a
  soundness argument.
- **Bug (unfixed):** `--residency N` with `--groups` also bounds group 1; at N=1 the layout group
  reports a residency deadlock. Per-group bounds work only through `--residency auto`.
