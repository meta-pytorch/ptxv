# ptxv design document

The original design document. Parts describe plans rather than the current implementation; the status of each part is in ROADMAP.md and the per-rung docs.

**Status:** rungs 0 and 1a implemented and validated on GB300 (2026-10-30): [docs/rung0.md](rung0.md), [docs/rung1a.md](rung1a.md); SASS scoreboards verified: [docs/sass-ctrl.md](sass-ctrl.md). Every verdict rests on the assumptions in [ASSUMPTIONS.md](../ASSUMPTIONS.md). Equivalence of a candidate kernel to a reference (ROADMAP rung 2e/3e): [docs/rung2e.md](rung2e.md); for non-identical implementations (reassociated sums, split-K, fp32 atomics, online vs two-pass softmax) with a proven bound per output: [docs/equiv-reassoc.md](equiv-reassoc.md). Build: `source env.sh && cargo build --release`. Python API (capture in process, sessions, bounds as functions): [docs/python.md](python.md); the CLI's JSON output: [docs/json.md](json.md). **Read `ROADMAP.md` next.** It records
the decisions taken after this draft (binary-first SASS lifting into one IR, exhaustive at the
captured grid, the rung 0 → 4-GPU MoE ladder), and `BUGS.md` is the subtle-bug suite.
**Target:** Blackwell only (`sm_100a` / `sm_103a`). The dev box has 4× GB300 (sm_103a, 284 GB,
NV18 all-to-all, CUDA 13.0), and every claim the tool makes is checked against those GPUs.
**Language:** Rust. Exhaustive schedule exploration, bit-exact emulation and Monte Carlo over
inputs all need to be fast and parallel.

ptxv takes a real PTX kernel, together with the launch it was captured from (grid, cluster,
params, tensor maps, the multi-GPU memory map), and answers two questions:

1. **Execution.** Is it free of deadlocks, data races, protocol misuse (mbarrier, TMEM, TMA,
   cluster, symmetric memory) and undefined behavior, *under every schedule the hardware may
   pick*? This is the question TLC answers for a TLA+ spec: explore the reachable state graph
   exhaustively, check invariants and liveness under fairness, and print a counterexample trace
   when one fails.
2. **Numerics.** For given inputs, or a set or distribution of inputs, how far can each output
   be from the exact real-number result? Each answer is labeled with its strength: a rigorous
   bound, a probabilistic bound under a stated rounding model, or a statistical estimate with
   confidence. The tool also reports where the error comes from.

This is **not a Lean-grade verifier**. The trusted base includes our encoding of PTX semantics,
our arithmetic models and `ptxas`. Section 9 lists exactly what is trusted, and every trusted
piece is cross-checked against hardware. The design emits certificates so that a small checker
(or Lean) could later re-verify results without trusting the search.

This project turns the fp_noise rounding-error theory (an internal note, not yet published)
into a tool. That theory gives the exact error identity (Thm 1), the deterministic bound
(Cor 1a), the discrete residual model (Cor 1d), schedule energies (Thm 2) and the flip rate
(Cor 4c). ptxv applies them automatically to the reduction trees it extracts from real PTX.

---

## 0. The core idea in one paragraph

A kernel is three things woven together: a **synchronization skeleton** (who waits for whom:
barriers, mbarrier phases, async-op completions, flags across CTAs and GPUs), **memory
footprints** (which bytes each piece of code reads and writes between sync points), and
**dataflow** (the floating-point DAG that produces each output). ptxv separates them:

* the **skeleton** is small after slicing, so it is model-checked exhaustively, TLA+-style;
* **footprints** are *computed*, not explored. A race is two footprints that overlap while
  the skeleton cannot order them;
* the **dataflow DAG** is extracted once. It is valid for every schedule because part 1 proved
  the kernel deterministic, or because part 1 enumerated the few classes of DAG that can occur.
  Numerics are then analyzed on that DAG.

The two halves meet at one artifact, the **determinism certificate**. Part 1 proves that every
schedule yields the same per-output DAG, or lists the possible DAGs (fp atomics, TMA reduce,
cross-rank reduction order). Part 2 may analyze a single execution only because part 1 proved
that. This link is what makes "verify numerics on one run" sound.

---

## 1. Scope: what "modern PTX" means here

Everything below is in scope for v1, and each item must have semantics, a checker rule and a
hardware test. The list is ordered by how much real kernels (CUTLASS sm100, FlashAttention-4 /
CuTeDSL, DeepGEMM sm100, Triton Blackwell, symmetric-memory collectives) depend on it.

### 1.1 Tensor memory and 5th-gen tensor cores (`tcgen05`)
* `tcgen05.alloc / dealloc / relinquish_alloc_permit`: column-count rules, exactly one alloc
  per dealloc, everything freed before exit, and allocation **blocking** when co-resident CTAs
  hold too many columns. That last one is a real deadlock source, so it is modeled with SM
  residency (§3.6).
* `tcgen05.mma` in every kind: `f16`, `tf32`, `f8f6f4`, `i8`, `mxf8f6f4`, `mxf4`, `mxf4nvf4`,
  `.sp`, `.ws`, block-scaled variants with scale factors in TMEM, `.cta_group::1/2` (2-CTA
  pairs), and collector-buffer hints (`collector::a::fill/use/lastuse/discard`). Reusing a
  collector buffer after `discard`/`lastuse` is a protocol check. The issuing thread is single,
  and its MMAs complete in order.
* `tcgen05.commit` → mbarrier arrive, including `.multicast::cluster` with a CTA mask.
* `tcgen05.ld / st / wait::ld / wait::st`: register results are undefined until the wait (a
  *register race* with an async unit). Lane-quadrant restriction: warp `w` may access only TMEM
  lanes `32·(w%4) … +31`.
* `tcgen05.cp` (smem → TMEM, e.g. scale factors), `tcgen05.shift`.
* `tcgen05.fence::before_thread_sync / after_thread_sync`: tcgen05 ops are ordered across
  threads only when one of these fences surrounds a thread sync.
* Descriptor decoding: shared-memory matrix descriptors (start, LBO, SBO, swizzle mode,
  base offset) and instruction descriptors are runtime 64-/32-bit values built by integer
  arithmetic. The skeleton tracks them concretely, and footprints come from decoding them.

### 1.2 TMA / bulk async copies
* `cp.async.bulk.tensor` in tile, im2col and `tile::gather4/scatter4` modes, `.multicast::cluster`,
  `.cta_group::2`, cache hints, `prefetch.tensor`. The footprint is the decoded box of the
  `CUtensorMap`: dims, strides, box, element strides, swizzle, interleave, OOB fill.
* `cp.async.bulk` (contiguous), `cp.reduce.async.bulk(.tensor)`. A TMA *reduce-add* into
  global memory is an order-nondeterministic fp reduction, so it feeds the numerics as a DAG
  class.
* `cp.async.bulk.commit_group / wait_group(.read)`: the smem source may be reused only after
  `.read` completes.
* Device-side tensor maps: `tensormap.replace` + `fence.proxy.tensormap::generic.release/acquire`
  (grouped GEMM / MoE).

### 1.3 mbarriers
`init`, `arrive(.release.{cta,cluster})`, `arrive.expect_tx`, `complete_tx` (from TMA,
`st.async`, `tcgen05.commit`), `arrive_drop`, remote arrive through `mapa`/`shared::cluster`,
`test_wait / try_wait(.parity)(.acquire.{cta,cluster})`, `invalidate`,
`fence.mbarrier_init.release.cluster`. The state is (phase, pending arrivals, tx-count). The
checks are listed in §3.4.

### 1.4 Clusters and distributed shared memory
`barrier.cluster.arrive(.release|.relaxed)/wait(.acquire)`, `mapa`, `getctarank`,
`ld/st/atom/red.shared::cluster`, `st.async`/`red.async` with mbarrier `complete_tx`, and
non-portable cluster sizes up to 16. **A CTA must not exit while a peer can still access its
smem.** Checking this requires a final cluster barrier or an equivalent HB chain.

### 1.5 Scheduling and launch features
* Cluster launch control: `clusterlaunchcontrol.try_cancel` / `query_cancel`. Whether a
  cancel succeeds is an *environment choice* that the model checker explores.
* Programmatic dependent launch: `griddepcontrol.launch_dependents / wait`. This requires a
  multi-kernel model: kernel N+1's prologue overlaps kernel N.
* `setmaxnreg` (warpgroup-uniform), `elect.sync`, `redux.sync`, `match.sync`, `vote.sync`,
  `shfl.sync` (membermask conformance under independent thread scheduling), `nanosleep`.

### 1.6 Shared memory (on-chip)
`ld/st/atom/red.shared::cta`, `ldmatrix/stmatrix`, generic ↔ async proxy handoffs
(`fence.proxy.async.shared::cta` before a TMA store reads smem written by `st.shared`), dynamic
smem bounds and alignment.

### 1.7 Multi-GPU (first-class, not an add-on)
* **Peer memory over NVLink:** `ld/st/atom/red.global` on addresses that map to another GPU's
  physical memory (cudaIpc, `cuMemMap` of imported handles, NVSHMEM symmetric heap,
  PyTorch symmetric memory).
* **System scope:** `.sys` for `ld.acquire / st.release / atom / red / fence.{sc,acq_rel}.sys`.
  The most common real bug is a cross-GPU flag published with `.gpu` scope, or a `volatile`
  (relaxed.sys) flag with no release fence.
* **Multicast / NVLS:** `multimem.ld_reduce / st / red` on multicast addresses (if the box
  exposes multicast objects; checked in M0). Reductions performed in the switch are a separate
  arithmetic model that needs calibration (§5.3).
* **Virtual aliasing:** symmetric memory is often mapped twice, at a unicast VA and a multicast
  VA backed by the same physical pages. Accesses through different VAs are ordered only through
  `fence.proxy.alias`. Race detection therefore works on **physical** addresses, with an
  alias-proxy tag.
* **NVSHMEM:** device-side NVSHMEM calls (P2P/NVLink transport) are linked bitcode that lowers
  to ordinary peer ld/st/atom plus fences. We verify the *linked* PTX. IBGDA / host-proxy
  transports (network) are out of scope for v1, which is single-node.
* **Cross-GPU progress:** kernels on different GPUs are launched independently, with arbitrary
  skew, and nothing guarantees they are co-scheduled. Persistent comm+compute kernels that spin
  on peers, or on CTAs that are not resident, are classic deadlocks. The model includes launch
  skew and per-GPU SM residency (§3.6).

### Out of scope for v1
Pre-Blackwell-only features (`wgmma`; Hopper can come later, since the IR is arch-parameterized),
textures/surfaces, graphics interop, network transports. (SASS-level verification *is* in
scope: the default front-end lifts the binary and checks its scoreboards, §9.)

---

## 2. Inputs: capturing a real launch

A PTX file alone is not enough. Footprints depend on tensor maps, params and memory maps
created on the host. Running a workload under **`ptxv-capture`** records everything:

* an `LD_PRELOAD` shim over the CUDA driver API (`cuModuleLoadData*`, `cuLibraryLoad*`,
  `cuLaunchKernel(Ex)` + launch attributes (cluster dims, PDL), `cuTensorMapEncode*`,
  `cuMemCreate/Map/Import*`, `cuMulticast*`, `cudaIpc*`);
* the output is one `.ptxcap` bundle per launch group: PTX text (from the fatbin, or the JIT
  input), kernel name, grid/block/cluster dims, dynamic smem, raw param bytes, decoded tensor
  maps, the **physical memory map across all GPUs** (allocation → owner GPU, every VA mapped in
  every process, unicast/multicast), the stream / PDL dependencies between launches, and
  optionally snapshots of the input tensors;
* for multi-GPU work, one bundle covers *all ranks*: N processes (or one process driving N
  GPUs), merged by launch-group id;
* Python helpers for Triton, CuTeDSL and torch: `ptxv.capture(fn, *args)`.

The bundle stores the **cubin** (SASS) as the primary artifact, plus PTX when it exists (JIT
sources). Shipped libraries on this box (torch, DeepEP, flashinfer cubins) contain no PTX at
all; see ROADMAP "Why binary-first".

Every later stage (model checker, numerics, hardware replay) consumes `.ptxcap`. Bundles are
small (KBs, unless tensors are snapshotted) and reproducible, so they are what bug reports carry.

---

## 3. Part 1: execution verification

### 3.1 Semantics: what we model and why it is enough

We do **not** enumerate weak-memory reorderings. We rely on the **DRF-SC** property of the PTX
memory model: a program with no data races among morally-strong accesses has only sequentially
consistent behaviors (Lustig et al., ASPLOS'19 formalize the scoped model). So we:

1. explore schedules under an **interleaving (SC) semantics** of the synchronization skeleton;
2. in every explored execution, compute the **happens-before** relation the PTX model induces,
   using scopes, proxies and async completions;
3. report any pair of conflicting accesses that is not ordered by HB and not morally strong.

If (3) never fires on any reachable execution, DRF-SC lifts the SC result to the real weak
model. **Caveat to check early (M3):** the published formalization predates the async proxy,
tcgen05 and tensormap proxies. We encode the PTX ISA's prose rules for these, tag each rule
with the ISA section it comes from, and validate the encoding with litmus tests on GB300 (§7).

HB edges come from a data-driven **semantics table**. Each opcode+modifier combination has one
row giving its proxy, scope, ordering (relaxed/acquire/release/acq_rel/sc/weak), footprint
function, async class, completion mechanism, ordering guarantees among same-thread async ops,
and uniformity requirements (`.sync`, `.aligned`, warpgroup-uniform). Examples:

| construct | HB effect |
|---|---|
| `barrier.cta.sync` / `bar.sync id, n` | all prior ops of the participating threads →hb→ their subsequent ops (CTA scope) |
| mbarrier phase completion | every `arrive.release.S` in phase φ, and every `complete_tx` source (TMA async-proxy writes, `tcgen05.commit` covering its MMAs) →hb→ every `try_wait.acquire.S'` that observes φ complete, if the scopes cover both sides |
| `fence.proxy.async` | generic ops before it →hb→ async-proxy ops after it, in the same thread |
| `tcgen05.fence::before_thread_sync` · sync · `after_thread_sync` | prior tcgen05 ops of thread A →hb→ subsequent tcgen05 ops of thread B |
| `st.release.sys` →rf→ `ld.acquire.sys` | synchronizes-with across GPUs; with `.gpu` scope only if both threads are on the same GPU |
| `fence.proxy.alias` | orders accesses to one physical location through different VAs |
| `barrier.cluster.arrive.release` / `wait.acquire` | cluster-wide HB, including DSMEM |
| `griddepcontrol.launch_dependents` / `wait` | primary-grid ops before trigger →hb→ dependent ops after wait (memory visibility rules per ISA) |

HB is tracked with **vector clocks over epochs**, not over individual instructions (§3.3).

### 3.2 Actors

* **Thread groups.** A warp starts as one actor. Because of independent thread scheduling, a
  warp can split at divergent branches into *warp-splits* (lanes that share a PC). Splits run
  as separate actors and merge opportunistically. Uniform code (most of a warp-specialized
  kernel) stays a single actor, which is the main source of tractability.
* **Async engines** as actors with queues of pending ops: a TMA unit per CTA, an MMA pipe per
  CTA or CTA pair (in-order per issuing thread), a TMEM ld/st unit, a bulk-copy unit, a CLC
  responder. *When* a pending op completes is a scheduler choice. Its effect (footprint, then
  `complete_tx` / commit-arrive) is atomic at completion, preceded by a "reads done" event
  where `.read` semantics require one.
* **Environment:** launch skew per GPU, CTA residency slots per SM, CLC cancel outcomes, the
  values atomics return (when not determined by the skeleton), and data-dependent branch
  outcomes the user marks as free.

### 3.3 Skeleton slicing (what makes exhaustive search feasible)

Backward-slice the IR from (a) every sync-visible instruction, (b) every branch condition that
controls one, and (c) every **address** and descriptor operand of a memory or async op. What
remains is the skeleton: loop counters, stage indices, phase bits, tile-scheduler state, CLC
responses, descriptors, flags. FP data and most registers drop out. Memory locations whose
values flow into the skeleton (flags, counters, tile queues, signal words) stay concrete. All
other memory is *ghost*: it tracks only who last wrote and read each location, which is what
race detection needs.

An **epoch** is the run of instructions between two sync-visible ops of one actor. For each
epoch instance the interpreter computes its footprint: a set of byte ranges per memory space
and physical location, plus register-file entries touched by async units. It is stored as
compressed interval sets or strided-box unions. Footprints depend only on skeleton state, so
they are computed **once per epoch instance, not once per interleaving**.

Race check: two epochs whose footprints conflict (overlap, at least one write, not both
morally strong) and that are unordered by HB → race, reported with both PTX source spans and
the HB-free path.

### 3.4 Properties checked (built in)

Safety invariants on every reachable state:

* **Data races:** smem, DSMEM, global, peer-GPU global (physical), TMEM, and *registers vs
  async units* (tcgen05.ld results read before `wait::ld`).
* **mbarrier protocol:** arrivals beyond the expected count; tx-count underflow or overflow
  (expect_tx bytes ≠ bytes actually delivered); **parity aliasing** (a waiter falls ≥ 2 phases
  behind, so `try_wait.parity` passes on a stale phase, a silent wrong-answer bug); use before
  init, or cross-CTA use without `fence.mbarrier_init`; use after invalidate; re-init while
  phases are pending.
* **TMEM:** alloc/dealloc pairing, leak at exit, out-of-range columns, lane-quadrant
  violations, use after dealloc, missing `relinquish_alloc_permit` when the protocol requires
  it.
* **Proxies:** generic write → async read with no `fence.proxy.async`; tensormap modified
  without the tensormap fences; unicast/multicast aliasing with no `fence.proxy.alias`.
* **Uniformity / convergence:** `.aligned` instructions executed by a partial warp;
  `.sync` membermask ≠ the set of lanes that actually arrive; `setmaxnreg` not
  warpgroup-uniform; `elect.sync` leader assumptions.
* **Cluster lifetime:** a CTA exits while a peer may still access its smem or mbarriers.
* **Bounds:** global/peer accesses inside captured allocations; smem inside the dynamic size;
  TMA boxes inside tensor-map bounds (or relying on OOB fill knowingly); descriptor alignment.
* **Scope sufficiency:** every synchronizes-with edge the kernel *relies on* has adequate scope
  (`.cta` vs `.cluster` vs `.gpu` vs `.sys`). Reported even when no race manifests in the
  explored configuration, because a scope that is too narrow is the main multi-GPU bug class.

Liveness, under weak fairness for each actor (TLA+ `WF`):

* **Termination / no deadlock:** every fair execution reaches "all actors exited, all async
  queues drained". A state with no enabled actor before that point is a deadlock. A fair cycle
  with no progress (spin loops that never get satisfied) is a livelock. Both come out of SCC
  detection on the state graph.

**Determinism:** for every reachable terminal state, the per-output producing DAG (§5.4) is
identical. Otherwise the tool emits the set of distinct DAG classes, typically caused by fp
atomics, TMA reduce, CLC-dependent tile ownership, or per-rank reduction order.

**Two-level verdicts (implemented, `crates/sim/src/twolevel.rs`).** Races and stale reads
are found under the PTX memory model. When a verdict has any (or relies on a refuted
assumption), `ptxv check` re-checks the case the same way (exploration, certificate, launch
groups, slices) under the GB300 hardware model (E-HW-L1: non-coherent per-SM L1 over
sequentially consistent memory) and classifies each finding as **observable on GB300**
(reachable there; the witness is the hardware-model finding) or **latent** (a PTX-model
violation the current hardware does not expose, with the reason, e.g. "no fill of line L on
SM s between the L1 invalidation at pc and the read"). The hardware model is closed-world
(E-L1-CLOSED: only the case's loads fill L1), so level 2 also runs the havoc check
(E-L1-HAVOC) as a probe: a finding latent only because nothing outside the case fills the
line is **latent-closed-world**, with the havoc witness (`stale-l1-havoc:<pc>`); a legal
co-resident kernel makes it observable (DeepEP notify_dispatch, docs/deepep-coreader.md).
Latent findings stay bugs: the
verdict's classes do not change. `--no-gb300` turns the second level off. A refuted
assumption (E-POLL-ACQ) can back a verdict only with an explicit opt-in (case
`assume_refuted`, `--assume-refuted`), which the verdict names.

**NaN / Inf outputs (opt-in, `crates/sim/src/nonfinite.rs`).** `ptxv check --nonfinite`
(Python `check(..., nonfinite=True)`) scans the float output buffers of every distinct
terminal state by dtype (f64, f32, f16, bf16, e4m3 (NaN only), e5m2; fp4 / fp6 have no
NaN / Inf) and adds a `nonfinite-output` finding per buffer with NaN / Inf when every float
input element is finite. Inputs that already hold NaN / Inf make it a note
(`--nonfinite-strict`: a finding). Provenance: during the check, every arithmetic
instruction (FADD / FFMA / FFMA2 / MUFU / div, the f16 / bf16 / fp8 / fp4 conversions,
HFMA2, DFMA) whose result is non-finite while its operands were finite (overflow, 0 / 0,
log / rsqrt of 0 or a negative), or NaN while no operand was NaN (inf - inf, 0 * inf), is a
*birth*, reported with its operands. Moves, selects, min / max and loads are never births and
an infinite constant is an operand like any other, so a mask's -inf is not one; the NaN it
later produces is. A finding names the births of its kind (Inf or NaN). Birth counts are
executions seen during the check: an explored check repeats them per schedule. The
simulator's NaN / Inf behavior is GB300's on the probed instructions (fixtures
c_nonfinite_*: satfinite e4m3 / e5m2 conversions saturate, MUFU.EX2 overflows to +inf,
`max.f32(NaN, 0)` = 0, f16 conversion overflow; outputs bitwise equal to the captures).
errdist reports P(output NaN / Inf) from its Tier 3 count (`p_nonfinite`, docs/errdist.md).

User invariants: a small predicate language over skeleton state, e.g. "stage `s` is written
by TMA at most once per phase", "every tile is claimed exactly once across CLC + atomics", or
"a rank never reuses buffer `b` before all peers signal done".

### 3.5 Search engine (TLC-style, specialized)

* Stateful exploration (parallel BFS by default, DFS for deep traces) over **128-bit
  fingerprints** of canonicalized skeleton states, kept in a lock-free sharded set. As in TLC,
  we report the fingerprint-collision probability bound.
* **Partial-order reduction:** persistent and sleep sets, using static independence from the
  semantics table (ops on disjoint sync objects commute; local epochs are invisible).
  *Implemented (2026-10-30)* as dynamic POR with sleep sets over recorded footprints
  (`crates/sim/src/dpor.rs`, `PTXV_EXPLORER=dpor`), cross-checked against the stateful
  explorer on every fixture; it is what makes rung 1b exhaustive ([docs/rung1b.md](rung1b.md)).
* **Symmetry reduction:** warps, warpgroups, CTAs and ranks that run identical skeleton code
  are canonicalized by sorting their skeleton states. Footprints are *not* symmetric, but they
  are checked separately (§3.3), so this is sound.
* **Loop cutoffs (toward parameterized proofs):** a pipelined mainloop's skeleton depends on
  `k mod S`, the phase bits and predicates such as `k == 0` / `k == K-1`. We abstract the
  counter into those predicates and check that the abstraction is exact for the skeleton.
  Exploration then covers *every* K ≥ the cutoff, not just the captured K. When the
  abstraction cannot be justified, we fall back to the captured K and say so.
* **Shape shrinking:** exhaustive interleaving on a reduced grid (e.g. 2–4 clusters per GPU,
  2–4 GPUs, small M/N/K that keep every code path, edge tiles included), plus a
  **full-scale footprint pass**. At the captured full shape, a cheap sweep over all epoch pairs
  that may be concurrent (anything not ordered by grid-level sync) checks write-footprint
  disjointness across all CTAs and all GPUs. This catches tile-index arithmetic that is only
  wrong at large shapes.
* **Stochastic mode** for configurations too large to explore exhaustively: PCT (probabilistic
  concurrency testing) gives a stated lower bound on the probability of hitting any bug of
  depth d per run. We report that bound and coverage (skeleton states, epoch pairs, phase
  combinations), never "verified".
* **Counterexamples:** a TLC-style trace (actor timeline, skeleton state diff per step, PTX
  source line per step, the two racing footprints), plus **witness replay on GPU**. We rewrite
  the PTX to insert `nanosleep` and spin delays that steer real hardware toward the trace,
  then run it on the GB300s to show the corrupted output or hang. A bug the tool reports and
  that hardware then exhibits is the best evidence for both.
* Optional export of the sliced skeleton to TLA+, for inspection with TLC or Apalache. This is
  a sanity check on our engine, not the main path.

### 3.6 Residency and progress

The PTX model guarantees forward progress only among *co-resident* threads, and only within
limits. We model per-SM slots (from captured smem, register and TMEM usage, and occupancy),
CTA dispatch order as an environment choice, and slots freed on exit. A CTA that waits on a
CTA that has not launched yet, a comm CTA that spins on a GEMM CTA that cannot be resident, or
a TMEM alloc that blocks on another resident CTA's columns is a deadlock that this model finds.
Across GPUs, kernel start skew is unbounded but fair: each kernel eventually starts unless a
stream or PDL dependency (from the capture) forbids it.

*Implemented (2026-11-04)* as two opt-in modes. `ptxv check --residency N|auto` (E-RESIDENCY,
`crates/sim/src/residency.rs`): at most N CTAs per GPU resident, clusters becoming resident in
any order as slots free up; `auto` reports per GPU the bounds that deadlock and the residency
the kernel needs ("deadlock-free iff >= N co-resident CTAs"), and whether the launch
guarantees it. `ptxv check --tmem-cotenant` (E-TMEM-COTENANT, E-TMEM-PERMIT,
`crates/sim/src/tenant.rs`): allocations at any free columns, CTA pairs sharing an SM's
tensor memory, a happens-before analysis for CTAs that hold columns or the allocation permit
while waiting on a CTA that cannot allocate, and missing `relinquish_alloc_permit`.

---

## 4. Shared substrate: IR and interpreter

* **`sass-lift`:** `nvdisasm` output → the same IR, using a SASS→IR table derived from a
  Rosetta corpus (PTX probes compiled by `ptxas`, aligned with their SASS, validated on
  hardware). This is the primary front-end (ROADMAP decision 1). Unknown opcodes are hard
  errors. It also decodes each instruction's control bits and statically verifies the
  register scoreboards on every CFG path ([docs/sass-ctrl.md](sass-ctrl.md)).
* **`ptx-parse`:** full PTX ISA 9.x grammar for the Blackwell subset first, with source spans.
  Prior art to consult: ZLUDA's Rust PTX parser. Hand-written with `logos` + recursive descent
  (or `winnow`/`chumsky`; all three are on crates.io).
* **`ptx-ir`:** typed virtual registers, CFG, divergence/uniformity analysis, loop structure,
  and a per-instruction pointer into the semantics table.
* **`sim`:** one interpreter, generic over a **value domain** and a **scheduler**:
  * value domains: `Concrete` (bits), `Skeleton` (concrete only on sliced registers, ghost
    elsewhere), `Shadow` (bits + exact/interval real + optional linear error form, §5.2);
  * schedulers: model-checker-controlled, PCT random, "deterministic fastest" (for numerics);
  * SIMT execution as SoA lanes (`[u32; 32]`) with active masks, so per-warp ops vectorize;
  * memory spaces: registers, smem, DSMEM (via a cluster map), TMEM (lanes × columns),
    global as a physical memory map spanning GPUs, plus multicast objects;
  * async engines as in §3.2.

The interpreter is validated by **differential conformance**: run the same `.ptxcap` with the
same inputs on GB300 and in `sim/Concrete`, and require bitwise-equal outputs. This runs on
every kernel in the corpus and on every CI change.

---

## 5. Part 2: numerical verification

### 5.1 What a claim is

A claim is a statement about every output element (or a chosen subset), compared to a
**reference semantics**:

* `ref = exact` is the kernel's *own algorithm* in exact real arithmetic. It isolates rounding
  error (A2/A3/A6 in the fp_noise taxonomy);
* `ref = spec` is a user-provided math spec (e.g. `softmax(QKᵀ/√d)·V` over the *same quantized
  inputs*), so the error also includes approximations (`ex2.approx`, online softmax, FTZ,
  scale handling, i.e. A4/A5/A7);
* `ref = kernel'` is a second kernel, for *relative* claims ("this FP8 kernel with promotion
  every 128 is never worse than 2× the error of the fp32-accumulating baseline", "bitwise equal
  across ranks", "batch-invariant").

Every claim carries a **strength level**:

| level | statement | how |
|---|---|---|
| **N0 measured** | for these inputs, the error of each output is exactly e (up to reference precision) | GPU run + exact reference (§5.5) |
| **N1 per-input rigorous** | for these inputs, \|ŷ−y\| ≤ B for every output | shadow execution or running-error analysis on the extracted DAG |
| **N2 input-set rigorous** | for *all* inputs in a box / format / norm ball, \|ŷ−y\| ≤ B(x) | symbolic error templates + transfer-function composition (§5.6) |
| **N3 probabilistic** | P(\|ŷ−y\| > t) ≤ ε, over rounding (under M1, or exactly for SR) and/or over an input distribution D | Azuma/Bernstein on the linear error form; Monte Carlo with Clopper–Pearson / DKW confidence |
| **N4 structural** | the per-output DAG is identical across configs/shapes/ranks: batch invariance, cross-rank consistency, tiling invariance (fp_noise Thm 3) | template equality, no arithmetic at all |

N4 is unusually strong and cheap. "Row i of the output does not depend on batch size M" or
"all 4 ranks get bitwise-identical all-reduce results" are proofs about DAG shape, and they
hold for *every* input.

### 5.2 Arithmetic models (`fpcore`)

Every fp-producing instruction has a model with two faces:

1. **bit-exact emulation**, when we know the hardware algorithm, and
2. an **error contract** that bounds analyses use: correctly rounded with unit u, faithful,
   relative ε, absolute ε, or "unknown → interval of the format's range" (which forces a
   calibration task).

| class | examples | model |
|---|---|---|
| IEEE with explicit rounding | `add/sub/mul/fma.{rn,rz,rm,rp}` for f32/f64/f16/bf16 (+x2), `.ftz`, `.sat` | generic softfloat over `(ebits, mbits, bias, ftz, sat, nan-style)` in pure integer Rust, tested against `rustc_apfloat` and hardware |
| **contractible pairs** | `mul.f32` followed by `add.f32` *without* `.rn` | **nondeterministic**: PTX lets `ptxas` fuse these into fma. Analyses cover both outcomes, or `--pin-sass` reads the cubin to resolve them |
| conversions | `cvt` to/from e4m3/e5m2/e2m3/e3m2/e2m1(x2), ue8m0, tf32 (`.rna`), `.satfinite`, **`cvt.rs`** (stochastic rounding with explicit random bits) | bit-exact. `cvt.rs` random bits are an input; SR's M1 holds exactly |
| MUFU approximations | `ex2/lg2/rcp/rsqrt/sqrt/sin/cos/tanh.approx`, f16/bf16 `ex2`, `div.approx/full` | contract from the ISA tables by default. **Unary f32 functions are exhaustively measured on GB300** (2³² inputs takes seconds), which yields the true max error and a bit-exact table where useful. 16-bit variants are tabulated outright |
| **tcgen05.mma** | each `kind` × input type × accumulate type × block-scale variant | parameterized MMA micro-model: product exactness, group size before rounding, alignment to max exponent, extra bits kept, truncate vs RN, normalization point, subnormal/FTZ handling, where block scales are applied (ue8m0 scales are exact powers of two; ue4m3 scales may round). **Calibrated on sm_103a** with Fasi/FTTN-style probes, then accepted only after ≥10⁹ random MMAs match bitwise. Until then, the contract form is used |
| **NVLS switch reductions** | `multimem.ld_reduce/red` add for f32/f16/bf16 | calibrated the same way (in-switch accumulation width and order). If ranks can disagree, that shows up as a DAG class |
| atomics / reduce | `red/atom.add.f32/f16x2/bf16x2`, `cp.reduce.async.bulk .add` | each op is RN-exact, but the *order* is nondeterministic, so it becomes a set of DAGs (fp_noise Cor 4b) |

Calibration data lives under `calib/sm_103a/` with probe kernels and logs, so a model can be
re-derived and diffed on new silicon or drivers.

### 5.3 Multi-GPU numerics

* Reduction order across ranks (ring vs tree vs one-shot vs NVLS) is part of the DAG, so
  fp_noise Thm 2 energies apply with S = #ranks.
* **Cross-rank bitwise consistency (N4):** one-shot symmetric-memory all-reduce kernels often
  sum peers in rank-rotated order, so ranks hold *different bits* for the "same" tensor and
  tensor-parallel replicas silently diverge. ptxv proves consistency or shows the two DAGs.
* Fused GEMM + reduce-scatter/all-reduce: the end-to-end DAG per output spans GPUs. The
  extraction follows peer stores and multimem ops through the physical memory map.

### 5.4 Dataflow extraction and templates

With determinism proved, run once with the `Shadow` domain and record the producing DAG of
every output. Leaves are input elements, nodes are fp ops with their arithmetic models. A full
GEMM has ~MNK nodes, so we **hash-cons by shape**. Outputs whose DAGs agree modulo leaf indices
form a **template class**. A dense GEMM has a handful (interior tiles, edge tiles, stream-K
seams), and attention has a few per mask pattern. All analysis below is per template, then
evaluated on data. This is also where N4 comes from: template classes are compared across
shapes, configs and ranks.

### 5.5 Measured mode (N0) at full scale on GPU

Run the real kernel on the GB300s, and compute the reference *exactly* on GPU, at full size
and at hardware speed:
* linear parts (GEMM / reductions): an Ozaki-scheme exact dot product (error-free splitting
  onto int8/fp64 tensor cores), so every output's true error is known;
* nonlinear parts: fp64 plus a double-double correction, or intervals where needed.

This mode is cheap, has no modeling assumptions, and serves as ground truth that every other
level must agree with. **A rigorous claim contradicted by a measurement is a bug in ptxv**,
and CI treats it that way.

### 5.6 Bounds (N1, N2) and attribution

* **Exact error decomposition (generalizes fp_noise Thm 1):** for a fixed input,
  `ŷ − y = Σ_v λ_v e_v + R`, where `e_v` is the local rounding error of node v (recorded
  exactly by the shadow), `λ_v = ∂y/∂(node v)` comes from reverse-mode AD over the DAG, and
  `R` is a second-order remainder with a rigorous interval bound. For pure sums and products
  (GEMM K-loops) `R = 0`, which recovers Thm 1 exactly.
* **N1 bound:** `|ŷ − y| ≤ Σ_v |λ_v| u_v |a_v| + |R|`. For template classes this collapses
  into closed forms. For GEMM it becomes `γ_eff(template) ⊙ (|A|·|B|)`, and that product is
  itself a GEMM, computed on GPU in fp64 with directed rounding. Bounds for every output
  therefore cost about as much as running the kernel.
* **N2 (input sets):** compose per-block **error transfer functions**. A block (a reduction,
  an elementwise map, online-softmax rescaling, a normalization) is summarized as
  `|out_err| ≤ Σ_i κ_i |in_err_i| + local(x)`, with κ and local bounded by interval arithmetic
  over the input set. The composition is sound by the mean-value theorem and the triangle
  inequality. It scales the way Satire scales FPTaylor-style analysis to large DAGs: abstract
  repeated subexpressions, and do not expand them.
* **Range events (A7):** in the same pass, prove no overflow, saturation, NaN or unexpected
  subnormal/FTZ in any intermediate for inputs in the declared ranges. This covers fp16/fp8
  intermediates, `.satfinite` clamps and block-scale overflow.
* **Attribution:** sum `|λ_v| u_v |a_v|` (bounds) or `λ_v e_v` (actual error) by PTX
  instruction, source line and arithmetic model. Example output: "71% of the error budget is
  the e4m3 accumulation inside `tcgen05.mma.kind::f8f6f4` at line 812; 18% is the bf16 output
  `cvt`; 6% is `ex2.approx`."

### 5.7 Probabilistic and distributional claims (N3), stochastic simulation

* **Over rounding:** the linear form `Σ λ_v δ_v a_v` with |δ_v| ≤ u_v gives Azuma/Bernstein
  tail bounds under M1, which hold exactly for SR. We also check M1's preconditions: fp_noise
  showed RZ violates M1 and that discrete residuals (Cor 1d) break the continuous model.
  Templates know which nodes are exact (N_v ≤ 1), so this matters in practice.
* **Over inputs:** Monte Carlo over a declared distribution D (e.g. bf16-quantized N(0,1),
  real captured activations, heavy-tailed ones), rayon-parallel on CPU templates or on the
  GPUs in measured mode. Results are quantiles and tail probabilities with Clopper–Pearson /
  DKW confidence, e.g. "P(rel err > 2⁻⁹) ≤ 10⁻⁶ with 99% confidence, n = 4.6·10⁶".
* **Monte Carlo arithmetic** (Verificarlo/CADNA-style random rounding of every node) as a
  cheap sensitivity probe: how many significant bits survive in each output.
* **Adversarial input search,** the numerics counterpart of a TLC counterexample: maximize the
  error, or the violation of a claim, over the input set with gradient-based search on the
  linear form plus evolutionary search on the bits. Tight witnesses show a bound is not loose,
  and violating inputs are counterexamples replayable on hardware.
* **Theory cross-check:** every template also gets the zero-parameter fp_noise predictions
  (Thm 2 energy, Cor 1d discrete moments, Cor 4c flip rates under schedule change). Measured
  vs predicted is reported, so the theory and the tool keep each other honest.

---

## 6. User-facing shape

```bash
# capture (any workload; multi-GPU via torchrun works)
ptxv capture -o run.ptxcap -- torchrun --nproc-per-node 4 my_kernel_test.py

# execution: exhaustive on a shrunk config, plus full-scale footprints
ptxv check run.ptxcap --shrink auto --liveness --report check.json
# -> OK (exhaustive, 1.9M states, cutoff K>=6 proven) | VIOLATION + trace + replay script

ptxv replay run.ptxcap --trace check.json#3 --gpus 0,1,2,3     # steer hardware into the bug

# numerics
ptxv num run.ptxcap --claims claims.toml --report num.json
```

```toml
# claims.toml
[[claim]]
name   = "fp8 gemm componentwise"
ref    = "spec:matmul"            # exact A@B on the quantized inputs
level  = "N2"
inputs = { A = "e4m3[M,K] blockscale=ue8m0/32", B = "e4m3[K,N] blockscale=ue8m0/32" }
bound  = "abs <= 2^-10 * (|A|@|B|) + 0.5 ulp_bf16(C)"

[[claim]]
name  = "cross-rank allreduce consistency"
level = "N4"
property = "bitwise_equal_across_ranks(out)"

[[claim]]
name   = "attention tail"
ref    = "spec:sdpa"
level  = "N3"
inputs = { dist = "captured:activations.pt" }
bound  = "P(rel_err > 2^-8) <= 1e-6 @ 99%"
```

Rust API first. pyo3 bindings (`import ptxv`) for Triton/CuTeDSL users in M6.

---

## 7. How we know the tool is right (the 4× GB300 loop)

1. **Differential conformance:** for every corpus kernel and random inputs, `sim` output is
   bitwise equal to hardware output. Any diff is a semantics bug.
2. **Litmus campaigns:** litmus tests for the PTX model (from the herd/Alloy formalization)
   plus new ones for mbarrier/TMA/tcgen05/proxy/cluster/`.sys`/alias/multimem, run billions of
   times across 1–4 GPUs. Our model must *allow* every behavior the hardware shows. An observed
   behavior the model forbids is a **soundness bug**. Forbidden-but-never-observed behaviors
   are fine, since the model may be weaker than the hardware.
3. **Mutation suite** (the main metric of usefulness): take correct kernels and inject known
   bug classes, such as deleting a `fence.proxy.async`, off-by-one parity, wrong `expect_tx`
   bytes, `.sys`→`.gpu` scope, dropping the final cluster barrier, a TMEM leak, a missing
   `tcgen05.wait::ld`, rank-rotated summation, or a fused mul+add that changes a bound. Report
   (a) the detection rate, (b) how often witness replay reproduces the bug on hardware, and
   (c) the time to verdict.
4. **Numerics ground truth:** each N1/N2 claim is checked against N0 measurements on large
   random and adversarial input sets. Each N3 claim is checked by its predicted vs observed
   tail frequency.
5. **Arithmetic calibration** is accepted only after bitwise agreement on ≥10⁹ random cases
   per (instruction, kind, type) on sm_103a, and re-run on driver/toolkit changes.

Corpus to start from: CUTLASS sm100 dense, block-scaled and grouped GEMMs; FlashAttention-4
(CuTeDSL, Blackwell); DeepGEMM sm100; Triton tutorials compiled for sm_100a; PyTorch
symmetric-memory one-shot/two-shot all-reduce; NVSHMEM put/signal examples; fused
GEMM+reduce-scatter; our own minimal kernels, one per feature in §1.

---

## 8. Code layout (Cargo workspace)

```
ptxv/
  crates/
    sass-lift/   nvdisasm text -> IR (primary front-end), Rosetta probe corpus + derived tables
    ptx-parse/   lexer+parser, AST, spans
    ptx-ir/      typed IR, CFG, uniformity, loops
    ptx-sem/     semantics table (proxy/scope/order/footprint/async/uniformity per opcode),
                 descriptor decoders (tensormap, smem desc, instr desc, TMEM addr), ISA cites
    fpcore/      generic softfloat, conversions, MUFU models, MMA micro-models, NVLS model
    sim/         SIMT interpreter generic over value domain + scheduler; memory + async engines
    mc/          slicing, epochs, footprints, HB vector clocks, BFS/DFS, POR, symmetry,
                 cutoffs, liveness (SCC), PCT, traces, TLA+ export
    num/         shadow exec, DAG/templates, error decomposition, N1–N4 engines,
                 transfer functions, Monte Carlo, adversarial search, attribution
    claims/      claim language, reports (JSON + text)
    capture/     LD_PRELOAD driver-API shim -> .ptxcap
    hw/          thin CUDA driver FFI (no cudarc in vendor): load/launch PTX on 1–4 GPUs,
                 litmus runner, calibration probes, witness replay, Ozaki exact reference
    ptxv-cli/
    ptxv-py/     pyo3 (later)
  litmus/  calib/sm_103a/  corpus/  mutants/
```

Candidate crates (crates.io): `logos`, `winnow`, `chumsky`,
`rayon`, `dashmap`, `ahash`, `smallvec`, `bitvec`, `petgraph`, `rand_xoshiro`, `statrs`,
`num-bigint`, `num-rational`, `rug` (+`gmp-mpfr-sys`, for MPFR intervals), `rustc_apfloat`,
`half`, `proptest`, `clap`, `pyo3`. `cudarc` is **not** vendored, so `hw/` uses its own minimal
driver-API FFI. No Rust toolchain is on PATH on this box yet. That is the first setup task
(rustup into `/data`, Cargo pointed at the vendored sources; Buck targets can come later if this
graduates out of `users/`).

---

## 9. Trusted base and honest limits

The authoritative, itemized list is **[ASSUMPTIONS.md](../ASSUMPTIONS.md)**. Every assumption
has an ID, a status and a failure direction, and every verdict prints the IDs it depends on.
This section is the summary.

Trusted, and cross-checked where possible:
* **`nvdisasm` + the scoreboard model** (when verifying the SASS lift, the default). This
  narrower trust base replaces trusting `ptxas` (ROADMAP). Variable-latency register
  hazards are *checked* from the decoded control bits ([docs/sass-ctrl.md](sass-ctrl.md)).
  What stays trusted: fixed-latency hazards covered by stall counts, the in-order operand-read
  rule for memory-pipe instructions, and scoreboards that guard memory ordering rather than
  registers (fence completion). All three are validated by the mutation sweep, not proven.
* **`ptxas`** (PTX-only inputs): we verify PTX, not SASS. PTX-permitted freedoms (fma contraction) are modeled
  as nondeterminism. Compiler bugs are out of scope. Hardware differential runs catch some.
* **The semantics table**: prose-derived for async/tcgen05/tensormap/alias proxies. Each rule
  cites the ISA section, and litmus tests guard soundness (§7.2).
* **DRF-SC for the extended model**: assumed. It is formally established for the core scoped
  model only.
* **Arithmetic models**: bit-exact only where calibrated; contracts otherwise.
* **The capture**: correct only for the launch that was captured. Claims are per `.ptxcap`
  (plus the cutoff-generalized K, and the N2 input sets).

Limits to be explicit about in every report:
* Exhaustive results hold for the *explored configuration*: the shrunk grid plus a cutoff
  proof when one applies, plus the full-scale footprint pass. They are not a proof for all
  grids unless the symmetry/cutoff arguments are certified.
* Data-dependent control flow (MoE routing, paged-attention block tables, early exit) is
  either fixed to captured data or explicitly marked free, and the free version can blow up.
* A stochastic (PCT) verdict is a probability statement and is printed as one.

**Path to Lean-grade:** the model checker can emit the reachable skeleton graph plus epoch
footprints and the HB derivation, and the numerics can emit per-template linear forms and
interval certificates. A small independent checker, or Lean definitions of the semantics
table, can re-check those without trusting the search or the floating-point code that
produced them.

---

## 10. Milestones

| | deliverable | exit criterion |
|---|---|---|
| **M0** | toolchain; parser + IR; `ptxv-capture` (single + multi-GPU); multicast availability probe | 100% of corpus PTX parses; `.ptxcap` round-trips launches on 4 GPUs |
| **M1** | `sim/Concrete` for scalar/SIMT/smem/global/peer memory; `hw/` runner | bitwise conformance on non-async kernels incl. peer-memory all-reduce |
| **M2** | async engines: mbarrier, TMA (all modes), tcgen05 (alloc/mma/ld/st/cp/commit), clusters/DSMEM, CLC, PDL; MMA calibration v1 | bitwise conformance on CUTLASS sm100 GEMM + FA4 at small shapes |
| **M3** | model checker: slicing, epochs, HB, races, protocol checks, liveness, POR/symmetry, traces, witness replay; litmus campaign v1 | mutation suite ≥ 90% detection on single-GPU mutants; ≥ 1 replayed on hardware per class |
| **M4** | multi-GPU: `.sys` HB, alias proxy, multimem, residency + launch skew, cross-GPU footprint pass | all multi-GPU mutants detected (scope downgrade, missing release, rank-rotated reuse, comm/compute residency deadlock) |
| **M5** | numerics N0/N1/N4: Ozaki reference, shadow DAG, templates, decomposition, attribution, batch-invariance + cross-rank consistency | N1 never contradicted by N0 on corpus; N4 verdicts match bitwise experiments |
| **M6** | N2 transfer functions, N3 probabilistic + Monte Carlo + adversarial search; claims TOML; pyo3 | published claims for the corpus with measured tightness (bound / max observed) |

## 11. Open questions for you

*Answered 2026-10-30, see ROADMAP:* Q1 → grouped GEMM from real binaries, via rung 0 fixtures;
Q3 → exhaustive at the captured grid (shrinking only with a coverage-equivalence check);
Q4 → read the binary (SASS lift); Q2 → NVLink for now, multi-node designed in (ROADMAP).


1. **Primary target kernels.** Which 2–3 kernels should drive M2–M5 first (e.g. a CuTeDSL
   block-scaled FP8/NVFP4 GEMM, FA4, a symmetric-memory all-reduce, or a fused GEMM+RS)?
2. **"shmem".** I read this as NVSHMEM / symmetric memory (multi-GPU), *and* covered on-chip
   shared memory/DSMEM. Should device-initiated NVSHMEM over the network (IBGDA) be in scope
   eventually, or is single-node NVLink the target?
3. **Cutoff ambition.** Is "exhaustive at a shrunk shape + full-scale footprints" an acceptable
   default, or do you want parameterized proofs (all K, all grid sizes) as a v1 goal?
4. **SASS pinning.** Resolve fma contraction by reading the cubin (more precise) or keep
   PTX-only nondeterminism (more portable)?

## References

* L. Lamport, *Specifying Systems* (TLA+); Y. Yu, P. Manolios, L. Lamport, "Model Checking TLA+ Specifications" (TLC), 1999.
* D. Lustig, S. Sahasrabuddhe, O. Giroux, "A Formal Analysis of the NVIDIA PTX Memory Consistency Model", ASPLOS 2019.
* NVIDIA, *PTX ISA* (9.x): memory consistency model, async proxy, mbarrier, tcgen05, TMA, multimem, cluster launch control.
* A. Betts et al., "GPUVerify: A Verifier for GPU Kernels", OOPSLA 2012. G. Li et al., GKLEE, PPoPP 2012.
* T. Cogumbreiro et al., "Memory Access Protocols: Certified Data-Race Freedom for GPU Kernels" (Faial), 2021–23.
* A. K. Kamath, A. Basu, "iGUARD: In-GPU Advanced Race Detection", SOSP 2021; ScoRD, ISCA 2020.
* M. Musuvathi et al. / S. Burckhardt et al., "A Randomized Scheduler with Probabilistic Guarantees of Finding Bugs" (PCT), ASPLOS 2010.
* C. Flanagan, P. Godefroid, "Dynamic Partial-Order Reduction", POPL 2005.
* N. J. Higham, *Accuracy and Stability of Numerical Algorithms*; Higham & Mary, probabilistic rounding error analysis, SISC 2019/2020.
* A. Solovyev et al., FPTaylor, TOPLAS 2018. A. Das et al., Satire, SC 2020. E. Darulova et al., Daisy, TACAS 2018.
* M. Fasi, N. J. Higham, M. Mikaitis, S. Pranesh, "Numerical Behavior of NVIDIA Tensor Cores", PeerJ CS 2021; X. Li et al., FTTN (matrix-accelerator numerics testing), 2024.
* K. Ozaki et al., error-free transformation for matrix multiplication (Ozaki scheme).
* C. Denis et al., Verificarlo (Monte Carlo arithmetic), 2016.
* fp_noise: the rounding-error theory ptxv applies (Thm 1, Cor 1a/1d/4c, Thm 2); internal note, not yet published.
