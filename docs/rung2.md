# Rung 2: real dense GEMM (Triton 3.5 and CuTeDSL 4.6.1 persistent tcgen05 matmuls)

**Status: done (2026-10-31).** Triton below; CuTeDSL in [its own section](#cutedsl-461-gemm);
the exploration work it needed (a synchronization reduction and a parallel explorer) in
[Exploration at this size](#exploration-at-this-size).

## Triton 3.5 GEMM

- **First real kernel:** Triton's persistent TMA + tcgen05 matmul lifts on both front-ends,
  and both are explored **exhaustively** at the captured grid (2 persistent CTAs, 4 tiles,
  2 k-steps per tile, 3-stage pipeline).
- **Agreement:** PTX model, SASS model and GB300 agree bitwise (1000/1000 runs).
- **Numerics:** Thm 1 is exact and the spec matches on both front-ends.
- **Findings:** Triton's PTX breaks the PTX contract in two ways (T1, T2 below), which this
  ptxas happens to compile safely. With both fixed (variant `fenced`), the PTX and SASS
  verdicts are both clean.
- **Mutants:** the four PTX-level mutants of the fixed kernel are all caught on both
  front-ends. Two (M3, M4) are wrong on GB300 in every run; M1 and M2 never fail on hardware.

### The kernel

`fixtures/r2_triton_gemm/capture.py` JIT-compiles this kernel and writes `kernel.ptx` and
`case.toml`. It is Triton tutorial 09's persistent matmul with host `TensorDescriptor`s:

```
for tile in range(pid, num_tiles, NUM_SMS) (flattened with the k loop):
    for k: a = TMA load, b = TMA load            # 3-stage mbarrier ring
           acc = tcgen05.mma(a, b^T, acc)         # TMEM accumulator, commit -> mbarrier
    epilogue: tcgen05.ld -> st.shared -> fence.proxy.async -> TMA store
```

The capture is C[512x128] f32 = A[512x128] bf16 · B[128x128]ᵀ, with BM/BN/BK = 128/128/64,
NUM_SMS = 2 and 4 warps. Each CTA computes 2 tiles × 2 k-steps. `variants.py` tags the PTX
with `// @TAG` markers and writes the variants.

### Verdicts

Every check below is exhaustive.

| variant | PTX model | SASS model | GB300 (1000 runs) |
|---|---|---|---|
| good (as compiled) | race + violation (T1, T2), 49,806 states | violation (T2), 14,787 states | bitwise = models |
| fenced (T1 + T2 fixed) | **clean**, 49,818 states | **clean**, 15,129 states | bitwise = models |
| M1: no TMA-ready wait before the loop's MMAs | race + nondeterministic + violation (bounded) | same (bounded) | correct 1000/1000 |
| M2: no accumulator-ready wait before the epilogue `tcgen05.ld` | race + nondeterministic + violation (bounded) | same (bounded) | correct 1000/1000 |
| M3: no `cp.async.bulk.wait_group.read` before the last tile's epilogue reuses the staging smem | race (bounded) | race, exhaustive | **wrong 1000/1000** (~21,600 words) |
| M4: no `fence.proxy.async` before the TMA store | race (bounded) | race, exhaustive | **wrong 1000/1000** (132–136 words) |

#### T1: Triton emits no tcgen05 thread-sync fences

The PTX ISA (9.7.18.6) orders tcgen05 operations of different threads only through
`tcgen05.fence::before_thread_sync`, a synchronization, and `tcgen05.fence::after_thread_sync`.
Triton 3.5 emits neither fence, so three hand-offs are unordered by the PTX contract:
1. `tcgen05.alloc` (warp 0) → `bar.sync` → every warp's `tcgen05.st` that zeroes its TMEM
   quadrant;
2. that `tcgen05.st` → `bar.sync` → the elected thread's first `tcgen05.mma` into the same
   columns;
3. MMA → commit → mbarrier wait → the epilogue's `tcgen05.ld` (the B06 pattern).

The SASS is clean: ptxas compiles both fences to nothing, `LDTM`/`STTM` are scoreboarded, and
the mbarrier wait orders the rest. GB300 computes the right answer. It is the same category as
B05/B06/B56: a contract the binary does not need on this chip and this ptxas. `fenced`
inserts the fences at all three hand-offs and at the dealloc.

#### T2: the shared-memory descriptor's version field is 0

Bits 46–48 of a tcgen05 shared-memory matrix descriptor must be 0b001. Triton's constant
`0x4000004004000000` leaves them 0. GB300 ignores the field. The model reports a `violation`
on both front-ends: it reads the descriptor in the SASS too. `fenced` uses
`0x4000404004000000`.

**M3 needed a second try.** The first M3 deleted the drain in the loop's epilogue, and the
model called it clean, correctly. At this grid each CTA runs the loop epilogue once, with no
store in flight; the second tile goes through the peeled epilogue after the loop, which has
its own drain. M3 now deletes that drain instead, and GB300 confirms the race.

**Open question (M1 SASS).** The scoreboard check also reports a WAR on `UR10` (a
`UTCHMMA` operand) in ptxas's output for M1. It may be a false alarm about when `UTCHMMA`
reads its uniform operands. The good and fixed binaries do not trigger it.

### Model pieces (ASSUMPTIONS.md)

- **E-ELECT (PTX), changed.** The PTX ISA says `elect.sync` picks its leader
  *deterministically*: the same leader for the same membermask. The explorer used to try every
  active lane at every elect, so a dozen elects per warp made the kernel unexplorable. Now the
  leader is a fixed function of the active set, and the kernel is explored twice: once with
  the lowest active lane as every leader, once with the highest (`per_leader_policy`). B46
  (code that assumes lane 0 leads) is still caught.
- **E-CTA-INDEP, new.** CTAs that share no synchronization object (no global atomics or strong
  accesses, clusters, CLC, or PDL) are explored one at a time. Their only contact is plain
  global memory, where races are found by vector clocks in either order.
- **Local ample sets (under E-DRF), new.** Actions that commute with everything are expanded
  alone:
  - `bar.sync` arrivals;
  - whole-warp steps that start after a wait;
  - `wait_group`;
  - exits with nothing pending;
  - guarded-off ops;
  - TMA-store and MMA completions.
- **Effect of the reductions.** The good kernel's PTX went from unfinished at 400k states
  (all three reductions off; 20 min) to exhaustive at 49,806 states (3 min). The existing suite gives the same verdicts with the
  reductions on.
- **E-SPIN, extended.** ptxas's out-of-line retry stubs may recompute the parity register
  (`R4 = UR6 << 31; TRYWAIT [a], R4`). These are accepted when:
  - the registers they write are dead at the continuation (a predicate-aware liveness walk);
  - the stub polls the same phase, which is checked every time the original poll succeeds, by
    running the stub on a copy of the thread.
- **E-WARPCOLL, new.** `stmatrix` / `shfl` exchange data within a converged warp in one step.
- **N-MUFU, new.** `MUFU.RCP` is modeled as the correctly rounded reciprocal. ptxas uses it
  only in its integer-division idiom, whose fixups make the quotient exact.
- **S-TCGEN05, extended.**
  - alloc / dealloc / relinquish are synchronous;
  - a thread's asynchronous tcgen05 ops are ordered after its own earlier instructions;
  - `tcgen05.st` / `wait::st` are modeled.

### Fixes made on the way

- **Uniform registers (a real model bug).** sm_100 has UR0–UR254, and ptxas uses UR63 and UR64
  as ordinary registers. The model took UR63 to be `URZ`: writes to UR63/UR64 were dropped and
  reads returned 0. This kernel keeps its load-loop bound in UR64, so the SASS model stopped
  issuing TMA loads and deadlocked. `URZ` is now index 255.
- **PTX front-end:**
  - module-scope and `.extern .shared` symbols (dynamic smem);
  - inline-asm `{ }` scopes (labels and registers renamed per scope);
  - `.loc` lines;
  - `stmatrix`, `shfl.sync.idx`, `bfe`, `div`/`rem`/`min`/`max`, `selp`, predicate logic;
  - `tcgen05.st` / `wait::st`, `mbarrier.inval`, vector `ld`/`st`.
- **SASS front-end:**
  - `LEA.HI`, `IABS`, `SGXT`, `I2F`, `F2I`, `MUFU.RCP`;
  - `STTM`, `STSM`, `UIMAD`;
  - negated `IMAD`/`LEA` operands, `ISETP.OR`, `SYNCS.CCTL.IV` (mbarrier inval).
- **Numerics:** PTX `st.shared.v4` now carries the provenance of all four words. Before, every
  output looked exact and `spec_match` failed.
- **`Op::src_regs` is exhaustive** (it used to fall through to "reads nothing"), and the new
  `Op::dst_regs` is exhaustive too.

## CuTeDSL 4.6.1 GEMM

`fixtures/r2_cute_gemm/capture.py` runs CUTLASS's `dense_gemm_persistent.py` example (the
copy bundled with flashinfer, CuTeDSL 4.6.1) on the same problem as Triton:
C[512x128] f32 = A[512x128] bf16 · B[128x128]ᵀ, 128x128 tiles, 2 persistent CTAs. The
kernel is warp-specialized, which Triton's is not: 6 warps per CTA.
- a TMA warp fills a 6-stage A/B ring (full / empty mbarriers);
- an MMA warp issues `tcgen05.mma` into a double-buffered TMEM accumulator and commits to
  accumulator-full;
- 4 epilogue warps wait for accumulator-full, `tcgen05.ld`, stage through shared memory,
  TMA-store, and arrive on accumulator-empty.

`fixtures/r2_cute_small` is the same kernel at M = 256: one tile per CTA. The 2-tile capture
overlaps tile 2's mainloop with tile 1's epilogue; the 1-tile capture does not. Neither
wraps the 6-stage ring or reuses an accumulator buffer (that needs ≥ 7 k-blocks and ≥ 3
tiles per CTA). `fixtures/r2_cute_wrap` does: see [Wrapping the rings](#wrapping-the-rings).

### Capturing it

CuTeDSL is a JIT with its own runtime, so capture needed more than Triton did:
- **The launch.** CuTeDSL calls the driver through `cuGetProcAddress` and its own
  `_cudaLaunchKernelEx`. `probes/launchcap` (an LD_PRELOAD recorder) hooks those, `dlsym`,
  and `cuLaunchKernel(Ex)`. It records the grid, block, dynamic smem and parameter bytes,
  using `cuKernelGetParamInfo` for the parameter layout. Parameters are opaque byte structs
  (case parameter kind `bytes`).
- **Tensor maps.** The DSL encodes its own descriptors and sets two bits of word 2
  (0x40002) that `cuTensorMapEncodeTiled` does not. The capture finds the geometry by
  matching re-encoded candidates with those bits masked. The case keeps the raw 128 bytes,
  which hardware runs use (with the address patched in).
- **PTX version.** The bundled compiler emits PTX 9.3; CUDA 13.0's ptxas takes 9.0. The
  capture lowers the version line. The kernel uses nothing newer.
- **API drift.** The example targets a slightly newer CuTeDSL than the bundled one; the
  capture shims `cute.make_fragment` and the `cute.arch.ProxyKind` / `SharedSpace`
  namespaces. The kernel code is the example's.

### Verdicts

| variant | PTX model | SASS model | GB300 (1000 runs) |
|---|---|---|---|
| good (as compiled), 1 tile | race + violation (C1–C3), exhaustive, 243,112 states | **clean**, exhaustive, 6,701 states | bitwise = models |
| good, 2 tiles | race + violation (bounded, 40M states) | **clean**, exhaustive, 728,553 states | bitwise = models (= Triton's output) |
| fenced (C1–C3 fixed), 1 tile | **clean**, exhaustive, 233,032 states | **clean**, exhaustive, 6,701 states | bitwise = models |
| fenced, 2 tiles | clean so far (bounded, 40M states) | **clean**, exhaustive, 728,553 states | bitwise = models |
| CM1: no accumulator-full wait before the epilogue `tcgen05.ld` | deadlock + race + violation (bounded) | deadlock + race + violation, exhaustive (1 tile) | **wrong 1000/1000** |
| CM2: no accumulator-empty arrive | deadlock, exhaustive (1 tile) | deadlock, exhaustive (1 and 2 tiles) | **hang** |
| CM3: no `fence.proxy.async` before the TMA store | race, exhaustive (1 tile) | race, exhaustive (1 and 2 tiles) | **wrong** 1000/1000 (1 tile), 999/1000 (2 tiles) |
| CM4: expect_tx of half the bytes the TMA loads deliver | nondeterministic + race + violation (bounded) | nondeterministic + race + violation, exhaustive (1 tile) | 1 tile: wrong 1/1000; 2 tiles: hang |

**PTX = SASS = GB300 bitwise** for good and fenced, and the output equals Triton's bit for
bit (same problem, same MMA order). Numerics: Thm 1 exact and spec match on both front-ends.

The good kernel's PTX breaks the PTX contract three ways. This ptxas compiles all three safely,
so the SASS is clean, and GB300 is correct:
- **C1: no `tcgen05.wait::ld`.** The epilogue stores the `tcgen05.ld` registers to shared
  memory without waiting (the B05 pattern). In the SASS, `LDTM` is scoreboarded.
- **C2: no tcgen05 thread-sync fences.** MMA → commit → wait → `tcgen05.ld`, and the last
  `tcgen05.ld` → barrier → `tcgen05.dealloc`, are unordered by the PTX contract (T1 again).
- **C3: the TMA store is issued by all 32 lanes** of epilogue warp 0, not one elected lane.
  Each lane stores the same tile, so the 32 bulk stores race with each other (write-write on
  global memory). They write identical bytes, so GB300 is correct.

`fenced` adds `tcgen05.wait::ld` after the load, the fences at each hand-off, and an
`elect.sync` around the store. That makes PTX and SASS both clean. The 2-tile PTX has
not finished: each leader policy passes 20M states. The 1-tile PTX is exhaustive.

The mutants are made from `fenced` in the PTX, and every one is caught on both front-ends.
CM1, CM2 and CM3 are wrong or hang on GB300 in every run, except one CM3 run at 2 tiles. CM4 at 1 tile fails 1 run in 1000; at 2 tiles
it hangs.

### Wrapping the rings

Phase-parity and overrun bugs only show once a ring wraps. `fixtures/r2_cute_wrap` captures
the same kernel with shallow pipelines (`CUTE_STAGES=2,1,2`: a 2-stage A/B ring, one
accumulator buffer, 2 C stages; capture.py patches the example's `_compute_stages`) and
K = 256. Each CTA runs 2 tiles × 4 k-blocks, so the A/B ring wraps 3 times per CTA and the
accumulator is reused once.

| variant | SASS model | GB300 (1000 runs) |
|---|---|---|
| good | **clean**, exhaustive, 6,472,038 states (5 min on all cores) | bitwise = models |
| fenced | **clean**, exhaustive, 6,472,038 states | bitwise = models |
| CM1–CM4 (as above) | caught (bounded, 2M states). CM4 is a deadlock here | CM1, CM3 wrong 1000/1000; CM2, CM4 hang |
| CM5: the TMA warp never flips its ab-empty wait phase | deadlock + fault + race + violation (bounded, 50M). The fault: an arrival past the expected count | **launch failure** |
| CM6: the MMA warp never flips its accumulator-empty wait phase | deadlock + nondeterministic + race (bounded, 10M) | **wrong 1000/1000** |

PTX and SASS numerics agree bitwise with GB300 (Thm 1 exact, spec match). CM5 and CM6 are new
mutants that exist only because the rings wrap; the earlier captures cannot express them.
The PTX of this capture has not been explored to completion.

**A scoreboard rule refined (L-OPREAD).** At first fenced SASS reported one finding, a static
WAR: `UISETP UP1` overwrote the `UP1` operand of an earlier `UTCHMMA` (which has `rd=SB2`)
without waiting on SB2. The model assumed uniform-unit instructions read every operand late.
ptxas shows otherwise. It places the SB2 wait right before the overwrite of that `UTCHMMA`'s
`UR48`, two instructions later, and leaves the `UP1` write unguarded. So uniform-unit
instructions read their URs late and their predicate operands at issue. The check now models
that. (Triton's M1 UR10 open question is a UR, so it is unaffected.)

### Model fixes on the way

- **elect.sync is a convergence point (E-ALIGNED).** A diverged warp reached `elect.sync` in
  two splits, and each elected a leader, so two threads initialized the same mbarriers.
- **TMA, commit, and TMA-mbarrier address offsets** (`[%r+1024]`) were dropped by the
  lowering; `shared::cta` TMA loads; `mov.b64 {lo, hi}` pack / unpack; MMA's
  disable-output-lane vector; `mbarrier.arrive` with a count.
- **E-SPIN:**
  - *Peeks.* CUTLASS tests a barrier early and waits later only if the test failed. A
    try_wait whose every failure path reaches a blocking retry is run as a non-blocking test.
    The peek's failure path may pass ptxas's tcgen05 guardrail trap calls.
  - *Retry stubs.* ptxas's out-of-line retry stubs may poll through a copied address register
    (`R0 = UR9; try_wait [R0+0x30]`); a run-time check confirms it is the same mbarrier.
- **TMEM alloc / dealloc race-check the shadow they clear** (S-TCGEN05). Before, an MMA
  completing after the dealloc was reported, and one completing before was not.

## Exploration at this size

The warp-specialized kernel is where the explorer's state count ran away. The 2-tile SASS
hit 5M states after 2.5 hours with new end states still appearing. Two pieces fixed it.

**A synchronization reduction** (ASSUMPTIONS.md, E-CTA-INDEP, "Synchronization reduction").
A whole-warp step runs alone when its synchronization footprint conflicts with nothing
another warp, or any warp's async units, can still do. Plain memory and TMEM are left out
under E-DRF.
- *Footprints.* Mbarrier keys are per phase, so a consumer's poll is independent of a
  producer that will not complete that phase.
- *What "can still do" means.* What each entity was seen to touch at pcs it can still reach
  in the CFG, plus what blocked steps wait on.
- *The bug it had.* A step that the chosen action disables never runs after it, so it was
  never learned. CuTeDSL CM1 lost its race that way: the arrive that strands the MMA warp
  was taken first, every explored schedule deadlocked, and the MMA never ran. Seeding the
  sets with random executions happened to hide it. A test now runs the reduction unseeded.
- *Check.* Every fixture and variant gives the same verdict with it on and off.
- *Effect:*
  - CuTeDSL 1 tile SASS: 38,816 → 6,701 states;
  - Triton PTX: 49,806 → 17,454;
  - r1b B09 / B10 now finish under 400k states, which they did not before.

**A parallel explorer.**
- *Design.* Work-stealing DFS over a sharded visited set (`PTXV_THREADS`; the default is
  every core; either way at most one worker per 50k states of the cap, so that a capped
  search runs the same number of depth-first lines on any machine with enough cores). A thief unshares the state it steals, because
  contention on shared reference counts was 40% of cycles at 64 threads. The CLI uses
  mimalloc.
- *Check.* Every complete run gives the same classes, states, terminal states and outputs
  as the serial search.
- *Effect.* 20–78x on large runs. The 2-tile CuTeDSL SASS (728k states) takes 48 s instead
  of 56 min.
- *Caveat.* A capped run depends on the thread schedule. Under the per-worker bound
  (`STATES_PER_WORKER`) it no longer depends on the number of cores beyond cap / 50k.

Before both pieces, single-threaded exploration itself got 2.7x faster (profile-driven):
- MMA results are memoized;
- page tables are copy-on-write at two levels;
- TMEM and shared memory are accessed in page runs;
- up to two readers per byte are kept inline;
- the shadow hash is cheaper;
- a warp's lockstep clocks are joined with one allocation.

A progress line (states, rate, frontier, findings) prints every 5 s; `PTXV_QUIET=1`
turns it off.

## Reproduce

```
cd fixtures/r2_triton_gemm && TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
    CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py          # optional: re-capture
ptxv check fixtures/r2_triton_gemm
ptxv hw fixtures/r2_triton_gemm --runs 1000
cargo test --release --test fixtures r2
cd fixtures/r2_cute_gemm && CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py && $PTXV_PYTHON variants.py
cd fixtures/r2_cute_small && CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py 256 128 128 2 && $PTXV_PYTHON variants.py
cd fixtures/r2_cute_wrap && CUTE_STAGES=2,1,2 CUDA_VISIBLE_DEVICES=0 $PTXV_PYTHON capture.py 512 128 256 2 && $PTXV_PYTHON variants.py
cargo test --release --test fixtures r2_cute_wrap -- --ignored   # ~20 min
ptxv check fixtures/r2_cute_small && ptxv hw fixtures/r2_cute_small --runs 1000
```
