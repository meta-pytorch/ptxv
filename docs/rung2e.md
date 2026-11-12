# Rung 2e: kernel equivalence (Triton vs. CuTeDSL vs. cuBLAS)

**Status: done (2026-11-02; the cuBLAS leg proven the same day).**
- **Triton vs. CuTeDSL:** proven at L0 (bit for bit) on exhaustive preconditions.
- **cuBLAS:** L1 and L2 hold on the captured inputs, and L3 holds at τ = 576u, **proven**:
  cluster 0 is now explored exhaustively (27,301 states with the phase-aware reduction, was
  >2.3B states without finishing; see [The bounded leg](#the-bounded-leg-cublas) for the
  history), and the E-WELLSYNC certificate independently covers all 32 clusters.
- **Generated-kernel demo:** each mutant comes back at its expected level.

`ptxv equiv` and the levels L0–L3 are defined in ROADMAP.md (rung 2e / 3e); the code is
`crates/sim/src/equiv.rs` and `crates/cli/src/equiv.rs`.

## The problem and the kernels

C[512x128] f32 = A[512x128] bf16 · B[128x128]ᵀ, for three kernels:

| kernel | source | launch | fixture |
|---|---|---|---|
| Triton 3.5 persistent TMA + tcgen05 matmul | PTX and SASS | 2 CTAs, 128x128x64 tiles | `r2_triton_gemm` (variant `fenced`) |
| CuTeDSL 4.6.1 `dense_gemm_persistent` | SASS | persistent, 1 tile per CTA | `r2_cute_gemm` (`good`, `fenced`) |
| cuBLAS `nvjet_sm103_tss_64x8_64x16_1x4_h_bz_TNT` (torch.mm's pick) | cubin only | 128 CTAs in 32 clusters of 4, 64-row tiles, multicast TMA and commits | `r2_cublas_gemm` |

The Triton SASS kernel is the reference: `ptxv equiv fixtures/r2e_gemm/equiv.toml`.

## Results

| candidate vs. Triton SASS | L0 bitwise | L1 same function | L2 no less accurate | L3 (τ = 576u) | verdict |
|---|---|---|---|---|---|
| Triton PTX (`fenced`) | proven | proven | proven | proven | **L0** |
| CuTeDSL SASS (`fenced`) | proven | proven | proven | proven | **L0** |
| CuTeDSL SASS (`good`) | proven | proven | proven | proven | **L0** |
| cuBLAS SASS | fails (all outputs) | holds | holds | holds | **L2+L3** |

- **L0 for Triton and CuTeDSL.** Both kernels split K = 128 into the same two 64-wide
  k-steps, and the MMA model is order-invariant within a K-group. So their rounding DAGs
  are identical, bit for bit, for every input.
- **Why cuBLAS fails L0.**
  - It uses a different tiling: 64-row tiles, with 16-wide MMA k-slices grouped
    differently.
  - Its epilogue computes `alpha·acc + beta·C` with `FFMA2`, at alpha = 1 and beta = 0.
    That is a rounding node `x·1 + 0`, which Triton and CuTe don't have.
  - On values, `x·1 + 0` is the identity except at −0, which becomes +0.
  - During the trial, a temporary diagnostic confirmed that this epilogue node is the only
    remaining L0 difference.
  - **Update (MISS 109):** the level is now decided with that final +0 peeled
    (`Canon::l0_z`): the pair is L0~0, L0 up to the sign of zero (equal bits except that a -0
    accumulator becomes +0), for the captured inputs (cluster 0 is not explored
    exhaustively). From the peel until MISS 109 it was reported as L0.
- **L1:** all four kernels compute exactly Σₖ aᵢₖ·bⱼₖ for every output. The canonical
  polynomials over input leaves are equal.
- **L2:** no candidate's per-term error multiplier exceeds the reference's.
- **L3:** each kernel's a-priori bound is 288u per product term (u = 2⁻²⁴). So |C − R| ≤
  576u · Σₖ |aᵢₖ bⱼₖ| for every input, and 576u is the least τ the bounds prove.
  `tau_k = 576` in the equiv.toml.
  **Update (2026-11-03):** the MMA error contract is now derived from the model
  (`MmaModel::eps_terms`, ≈ 10u per full K=16 step instead of ≈ 36u; docs/equiv-mma.md §1).
  Each kernel's bound is now 79.5u per product term, so |C − R| ≤ 159u · Σₖ |aᵢₖ bⱼₖ|. τ = 576u
  is still proven, but it is no longer the least τ the bounds give.
- **Relies on (cuBLAS): E-BULK-EXIT.** cuBLAS's CTAs exit without
  `cp.async.bulk.wait_group.read` after their TMA store. GB300 keeps the shared memory
  alive until the store has read it (probes/tma_store_exit). `check` reports this as a
  `contract` finding, and equiv lists it as what the verdict relies on.

## Preconditions

`equiv` proves a level only if each kernel is race-free and deadlock-free on every
schedule (one output per input), and its control flow and addresses don't depend on input
data (the taint check).

- **Triton, CuTeDSL:** exhaustive `check` (docs/rung2.md), taint-clean.
- **cuBLAS:**
  - **Taint:** clean.
  - **Hardware agreement:** the simulator matches GB300 bit for bit (1000 runs).
  - **Race and deadlock freedom:** checked in three steps.
    1. **Cluster independence (E-CTA-INDEP):** clusters share nothing but global memory.
    2. **Symmetry:** one deterministic run of all 128 CTAs, with `%ctaid` as a taint
       source. Every guard or control value that depends on position is the same in all 32
       clusters, at the same (CTA rank, thread). No global byte is written by one thread
       and touched by another. So cluster 0's exploration stands for all 32. This run
       takes about 2.5 minutes. Fixture `e_sym` covers it: a bug in only one CTA is
       refused and found.
    3. **Cluster 0 explored:** not exhaustive. The equiv run explores 5M states. A
       separate run reached 143.6M states in 8.6 hours (still running). It found no race,
       deadlock or violation, only the E-BULK-EXIT contract finding.

## The bounded leg: cuBLAS

A single cluster is 4 CTAs × 256 threads, with:
- a 16-stage TMA ring,
- multicast TMA loads into all 4 CTAs,
- `UTCBAR.MULTICAST` commits that free each stage in all 4 CTAs,
- a per-CTA TMEM allocator.

Work it took to get this far, all committed with fixtures or tests:

| problem | fix | effect |
|---|---|---|
| 128 CTAs × 256 threads per state | explore one cluster at a time (slices), then symmetry | 32 explorations → 1 |
| 1.3 MB per queued state | uniform shadow pages (`Pages::for_run`); TMA, TMEM-mark and MMA-tile shadow updated per run | 29 GB → 4.8 GB at 200k states |
| shared reference counts on every step | per-worker kernel/launch copies; two-level `CowVec` for threads, warps, smem, TMEM; scans limited to the explored cluster | 4.9k → 14.7k states/s (96 threads) |
| producer's per-lane stage drain before exit | `exit_bound`: part of a warp on an invisible path to exit runs in one order (fixture `e_exitbound`) | drain loop gone from the branching profile |
| commits blocked by the epilogue's accumulator poll | parked warps that only the candidate can release are left out of the conflict check; the dependence is recorded with the choice | commit rejections nearly gone (2,200 → 30 per 60k states); same verdicts elsewhere (r2_cute_small −4% states) |

The remaining blowup, from the reduction's rejection profile (`PTXV_POR_DEBUG`), comes from
two over-approximations in the learned per-pc footprints:

1. **TMEM alloc vs. TMEM free in the same CTA.**
   - The free (`UTCATOMSWS.AND`, epilogue warp 0) is reachable only after the accumulator
     wait. That wait needs the MMA, and the MMA needs the alloc. The reduction sees only
     control-flow reachability.
   - So each CTA's alloc interleaves with everything its producer does first.
2. **The MMA warp's stage-full poll vs. another CTA's multicast TMA.**
   - The footprints merge mbarrier phases. So a later-phase TMA into the same stage looks
     like it can complete the next phase before the poll.
   - In fact the next phase needs a stage-empty arrival, and that arrival needs this CTA's
     own commit.

Both need phase-aware reasoning: how many arrivals each barrier phase still needs, and who
can supply them. That is the next piece of work, and it would turn the cuBLAS leg into a
proof.

### Phase-aware reduction: cluster 0 completes (later, 2026-11-02)

With the reductions below (ASSUMPTIONS.md, E-DRF "Phase-aware footprints", "Gated
reachability", "TMA load completions as candidates"), cluster 0 is explored exhaustively:
**27,301 states, 16 terminal states, one output, only the E-BULK-EXIT contract finding, ~11 s
on 32 threads** (before: not finished after 1.6 billion states). 3,000 random complete
executions of the cluster without any reduction (`PTXV_CHECK_WALKS=3000`) all end in one of
the 16 terminal states, with no finding the search missed. The equiv run's cuBLAS leg can
now be rerun as a proof (not done here).

| switch off | states (cap 2M) |
|---|---|
| nothing | 27,301 (complete) |
| `PTXV_NO_GATE` (gated reachability) | 278,217 (complete) |
| `PTXV_NO_PHASE_CLIP` (phase clipping, pending-op mbarriers) | > 2M |
| `PTXV_NO_TMA_POR` (TMA completions as candidates) | > 2M |
| every new switch | > 2M (≈ before) |

What each does here:
1. **TMEM alloc vs. free.** The gated reach of epilogue warp 0 stops at its accumulator-full
   poll (parity and address still in their registers, phase 0 not complete, and nothing
   but the MMA warp's commit, which comes after the alloc, can complete it), so the free
   behind it is not "still possible" before the alloc.
2. **Stage-full poll vs. multicast TMA.** Learned footprints are clipped to the mbarriers'
   current phases, and an async unit whose warp cannot issue again before the candidate
   only completes its pending operations (their own mbarriers). TMA completions themselves
   can now be expanded alone, which removes the "only TMA completions enabled" states.
   What remains are the MMA warp's *peek* (non-blocking try_wait at 0x1d80) racing a
   pending TMA into that stage: those orders really differ (the peek's result does).

## Generated-kernel demo (`fixtures/r2e_tile`)

The rung 1a tcgen05 tile GEMM is the reference, with "generated" candidates
(`crates/cli/tests/equiv.rs` checks all five):

| candidate | expected and obtained |
|---|---|
| same kernel via PTX | L0 |
| E1: reordered reduction | L1 (not L0) |
| E2: dropped K slice | not L1; witness: one input pair set to 1 gives 1 vs. 0, in the simulator and on GB300 |
| E3: split accumulators, `add.rz` epilogue | L1, not L2 |
| E4: an address depends on input data | L0 (unproven): the taint check flags it |

## Beyond identical orders

docs/equiv-reassoc.md extends `ptxv equiv` to implementations that compute the same operation
in different orders or with different algorithms. Each output gets the bound it achieves (L1 plus
a rigorous Cor 1a / Thm 4 bound) and a JSON summary:
- input leaves through plain moves: torch.cat vs Triton cat is L0;
- reassociated sums: butterfly, split-K, fp32 atomics (bounded over every order), cumsum;
- softmax: torch vs Triton two-pass and online, in a normal form with exhaustively swept exp and
  1/x contracts;
- layer norm and RMS norm (f32): torch (Welford) vs Triton (two-pass), a polynomial in the
  inputs and one swept rsqrt-like scalar;
- the attention core (rescaled P·V accumulation, scores given); the bf16 tensor-core pair is
  blocked (HMMA numerics, bf16 provenance; listed there).

## Reproduce

```
ptxv equiv fixtures/r2e_gemm/equiv.toml          # ~50 min (cuBLAS: symmetry + 5M states)
ptxv equiv fixtures/r2e_tile/equiv.toml
ptxv check fixtures/r2_cublas_gemm --slices --max-states N   # cluster 0 alone after symmetry
```
