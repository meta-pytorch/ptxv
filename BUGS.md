# BUGS: the subtle-bug suite ptxv must catch

Every entry is a **(correct kernel, mutant)** pair, with the property that must fire and an
honest note on how the bug behaves on real hardware. The suite exists to show what ptxv finds
that testing misses. The usefulness metric (ROADMAP) is the detection rate over this suite,
together with how often hardware reproduces the bug on its own (`hw-rate`, measured over 10⁴
runs) versus under witness replay.

How mutants are built (the verifier never sees source):
* **fixtures:** we write the PTX, so we write the mutant PTX directly;
* **JIT kernels** (CuTeDSL, Triton): mutate the captured PTX, then recompile with `ptxas`;
* **binary-only kernels** (torch, trtllm-gen, DeepEP): **SASS patching**. Mutation operators
  are learned from Rosetta probe diffs: NOP out an instruction (keeping its control bits),
  change an immediate (`expect_tx` bytes, parity, barrier count), or downgrade a scope
  (`MEMBAR.SYS`→`.GPU`, `.release.sys`→`.gpu`). Rebuilding from library source is allowed
  *for making mutants only*.

Columns: **P** = property (docs/design.md §3.4/§5), **where** = rung that introduces it.

## Single CTA / cluster (async machinery)

| id | bug | P | hw behaviour | where |
|---|---|---|---|---|
| B01 | `expect_tx` bytes ≠ bytes delivered (one tile too few) | deadlock (tx-count never hits 0) | ✅ caught; hangs every time on GB300 | R0 |
| B02 | consumer waits on the wrong parity in the first phase | race + deadlock + async-after-exit (all three found) | ✅ caught; **1/10k wrong** on GB300 (stale smem from the previous launch hides it), 10k/10k with smem scrubbed; hang reproduced by witness replay | R0 |
| B03 | `mbarrier.init` by one thread, no `bar.sync` / `fence.mbarrier_init` before other warps use it | race on the mbarrier object | ✅ caught; **never** visible on GB300 (0/10k, also scrubbed) | R0 |
| B04 | epilogue `st.shared` → TMA store with no `fence.proxy.async.shared::cta` | proxy race (generic write → async read) | ✅ both fronts; **GB300 wrong 1000/1000** (61 distinct outputs, smem scrubbed) | R1b |
| B05 | registers from `tcgen05.ld` used before `tcgen05.wait::ld` | register read before ready (PTX); SASS clean | ✅ PTX catches; GB300 correct 1k/1k: this ptxas scoreboards LDTM anyway | R1a |
| B06 | missing `tcgen05.fence::after_thread_sync` after the mbarrier wait that hands the TMEM accumulator to the epilogue | tcgen05 ordering race (PTX); SASS clean | ✅ PTX catches; GB300 correct 1k/1k (fence is a NOP in this ptxas) | R1a |
| B07 | `tcgen05.dealloc` while other warps still `tcgen05.ld` (no barrier) | TMEM race + read of deallocated TMEM | ✅ both fronts; never visible on GB300 (freed TMEM keeps data) | R1a |
| B08 | missing `relinquish_alloc_permit` with 2 CTAs/SM, each allocating > half of TMEM | deadlock under residency 2 | hangs only at certain occupancies | R1a |
| B09 | persistent kernel: MMA warp starts tile t+1 into the same TMEM accumulator before the epilogue's "tmem empty" arrive | TMEM race | ✅ both fronts (bounded search); GB300 correct 1000/1000: the epilogue drains TMEM first | R1b |
| B10 | pipeline with S stages: producer's "empty" wait uses the wrong phase after wraparound (k ≥ S) | race (TMA over a stage the MMA reads) + mbarrier over-arrival fault | ✅ both fronts (bounded search); **GB300: unspecified launch failure** | R1b |
| B11 | consumer skips its `arrive` on the "empty" barrier for the last k-iteration of a tile | deadlock (phase drift) | ✅ both fronts, exhaustive; **GB300 hangs** (watchdog) | R1b |
| B12 | 2-CTA MMA: `tcgen05.commit` multicast mask omits the peer CTA | peer deadlock | ✅ both fronts, exhaustive; **GB300 hangs** (watchdog) | R1c |
| B13 | CTA exits with no final `barrier.cluster` while the peer still arrives on its mbarrier (remote consumer release) | cluster-lifetime violation | ✅ both fronts, exhaustive; GB300 correct 1000/1000 (UB, invisible) | R1c |
| B14 | cluster mbarriers initialized without `fence.mbarrier_init.release.cluster` before a relaxed cluster arrive | race on a remote mbarrier (init vs the peer's multicast complete_tx) | ✅ PTX; SASS clean by construction: ptxas emits the same fence anyway (binaries differ by one NOP); GB300 correct | R1c |
| B15 | named barrier count 256 where 384 threads participate in one code path | barrier misuse / deadlock | path-dependent hang | R1b |
| B16 | `.aligned` op (`tcgen05.alloc`) reached by a partial warp | uniformity violation (PTX); SASS: race on ptxas's relinquish flag + guardrail trap + deadlock | ✅ both; **GB300 hangs** | R1a |
| B17 | CLC: the `try_cancel` response is used without waiting for its mbarrier | race + read of unwritten smem + lost tiles | ✅ both fronts, exhaustive; GB300 at 8192 CTAs: **launch failure** (unspecified launch failure or illegal address, every run) | R1d |
| B18 | PDL: dependent kernel reads the primary's output without `griddepcontrol.wait` | cross-kernel race | ✅ both fronts, exhaustive; GB300 correct 1000/1000 (and at 512 CTAs), **wrong 200/200** with a 20 µs delay before the primary's epilogue store; the same code launched without PDL is clean and stays correct under the delay | R1e |

| B41 | TMEM never deallocated | TMEM leak at exit | ✅ both; driver error "tensor memory not completely freed" | R1a |
| B42 | first MMA with enable_input_d = 1 (accumulates into uninitialized TMEM) | accumulate into never-written TMEM | ✅ both; GB300 999/1000 wrong (run 0 correct: TMEM starts zeroed) | R1a |
| B43 | tcgen05.commit to the wrong mbarrier | deadlock | ✅ both; GB300 hangs | R1a |
| B44 | SASS patch: the scoreboard wait between LDTM and the first use of its registers is cleared | scoreboard RAW (SASS only; the PTX is correct) | ✅ SASS; GB300 correct 1k/1k (the STG read queues behind LDTM) | R1a |
| B45 | SASS patch: LDTM gets no write scoreboard | variable-latency result without scoreboard | ✅ SASS; GB300 correct 1k/1k | R1a |
| B46 | leader chosen with `elect.sync`, code assumes it is lane 0 | race (PTX contract); SASS clean | ✅ PTX; GB300 correct (it elects the lowest lane, probes/elect) | assumption fixture (E-ELECT) |
| B47 | accumulator initialization deleted | unwritten register read (PTX); SASS: numerics spec mismatch | ✅ PTX; SASS via numerics (ptxas reuses `%tid.x`'s register); GB300 outputs bitwise unchanged | assumption fixture (S-INIT) |
| B48 | partial producer: consumers read shared memory nobody wrote (no race) | unwritten smem read | ✅ both; GB300 wrong 200/200 with smem scrubbed | assumption fixture (S-INIT) |
| B49 | wait loop counts its spins (failure path does work) | `unmodeled` on PTX (E-SPIN shape check); SASS clean (ptxas deletes the dead counter) | ✅ correct refusal | assumption fixture (E-SPIN) |
| B50 | TMA store issuer reuses the staging buffer without `cp.async.bulk.wait_group.read` | race (generic write over the store's async read) + nondeterministic | ✅ both fronts, exhaustive; GB300 correct 1000/1000 (the store wins) | R1b |
| B53 | CLC: the wait parity is not flipped for the next request; the stale response keeps saying "cancelled" | mbarrier over-arrival fault (re-requests while a phase is pending) | ✅ both fronts (bounded); GB300 at 8192 CTAs: **unspecified launch failure** | R1d |
| B54 | CLC: a CTA exits after a successful cancel without doing the stolen tile | lost work (nondeterministic output) | ✅ both fronts, exhaustive; GB300 at 8192 CTAs: **≈1985 of 8192 tiles lost** every run | R1d |
| B55 | PDL: dependent overwrites the primary's input before `griddepcontrol.wait` (write-after-read across kernels) | cross-kernel race | ✅ both fronts, exhaustive; GB300 correct (ptxas consumes the load before `PREEXIT`, so the read is done before the dependent can launch): contract-level, like B14 | R1e |
| B56 | relaxed poll followed by `fence.acq_rel.gpu` as the acquire for a peer GPU's release | race (PTX: the fence's scope does not include the writer) | ✅ PTX; SASS clean (only `MEMBAR` is scoped; the acquire half `CCTL.IVALL` is the same as for `.sys`); GB300 correct 20000/20000: contract-level | R1f |
| SBx | every one-bit scoreboard mutant of the r0 and r1a binaries (`ptxv sb-sweep`) | scoreboard RAW/WAW/WAR | 0 missed; see docs/sass-ctrl.md | R1a |

## Real kernels (rung 2: Triton 3.5 and CuTeDSL 4.6.1 persistent tcgen05 GEMMs)

Findings in the kernel as compiled (T) and PTX-level mutants of the fixed kernel (`fenced`).
All are in `fixtures/r2_triton_gemm` (docs/rung2.md).

| id | bug | P | hw behaviour | where |
|---|---|---|---|---|
| T1 | **Triton emits no `tcgen05.fence::{before,after}_thread_sync`**: alloc → zeroing `tcgen05.st`, that st → MMA, and MMA → commit → wait → `tcgen05.ld` are all unordered by the PTX contract | tcgen05 ordering race (PTX); SASS clean | ✅ PTX, exhaustive; GB300 correct 1000/1000 (the fences are NOPs in this ptxas): contract-level, like B06 | R2 |
| T2 | **Triton's smem descriptor has version bits 46–48 = 0** (PTX ISA requires 0b001) | violation, both fronts | ✅ both, exhaustive; GB300 ignores the field | R2 |
| R2-M1 | no TMA-ready mbarrier wait before the loop's MMAs | race + nondeterministic + violation (loads never observed before exit) | ✅ both fronts (bounded); GB300 correct 1000/1000 | R2 |
| R2-M2 | no accumulator-ready wait before the epilogue `tcgen05.ld` | race + nondeterministic + violation | ✅ both fronts (bounded); GB300 correct 1000/1000 | R2 |
| R2-M3 | no `cp.async.bulk.wait_group.read` before the last tile's epilogue reuses the staging smem | race | ✅ both (SASS exhaustive); **GB300 wrong 1000/1000** (~21,600 words, also with smem scrubbed) | R2 |
| R2-M4 | no `fence.proxy.async.shared::cta` before the TMA store | proxy race | ✅ both (SASS exhaustive); **GB300 wrong 1000/1000** (132–136 words) | R2 |
| C1 | **CuTeDSL's epilogue uses `tcgen05.ld` registers without `tcgen05.wait::ld`** | register read before ready (PTX); SASS clean | ✅ PTX, exhaustive (1 tile); GB300 correct 1000/1000 (LDTM is scoreboarded): contract-level, like B05 | R2 |
| C2 | **CuTeDSL emits no tcgen05 thread-sync fences** (MMA → commit → wait → `tcgen05.ld`; last ld → barrier → dealloc) | TMEM ordering race (PTX); SASS clean | ✅ PTX, exhaustive (1 tile); GB300 correct: contract-level, like T1 | R2 |
| C3 | **CuTeDSL's TMA store is issued by all 32 lanes** of epilogue warp 0 (no elect) | write-write race between the 32 bulk stores (PTX); SASS clean | ✅ PTX, exhaustive (1 tile); GB300 correct (identical bytes) | R2 |
| R2-CM1 | CuTeDSL: no accumulator-full wait before the epilogue `tcgen05.ld` | deadlock + race + violation (SASS also `contract`: its last commit is a trailing arrive, E-TC-EXIT) | ✅ both (SASS exhaustive, 1 tile); **GB300 wrong 1000/1000** | R2 |
| R2-CM2 | CuTeDSL: no accumulator-empty arrive | deadlock | ✅ both, exhaustive; **GB300 hangs** | R2 |
| R2-CM3 | CuTeDSL: no `fence.proxy.async` before the TMA store | proxy race | ✅ both, exhaustive (1 tile); **GB300 wrong** 1000/1000 (1 tile), 999/1000 (2 tiles) | R2 |
| R2-CM4 | CuTeDSL: `expect_tx` of half the bytes the two TMA loads deliver | nondeterministic + race + violation | ✅ both (SASS exhaustive, 1 tile); GB300: 1 tile wrong 1/1000, 2 tiles hang | R2 |
| R2-CM5 | CuTeDSL (wrapping capture): the TMA warp never flips its ab-empty wait phase | deadlock + fault (arrival past the expected count) + race | ✅ SASS (bounded); **GB300 launch failure** | R2 |
| R2-CM6 | CuTeDSL (wrapping capture): the MMA warp never flips its accumulator-empty wait phase | race + nondeterministic + deadlock | ✅ SASS (bounded); **GB300 wrong 1000/1000** | R2 |

## Multi-GPU

| id | bug | P | hw behaviour | where |
|---|---|---|---|---|
| B19 | flag published to a peer with `st.release.gpu` (should be `.sys`) | scope insufficiency → race on the flag (not morally strong) and on the payload | ✅ both fronts, exhaustive; GB300 correct 20000/20000: **essentially always passes on hardware**, the archetypal bug ptxv exists to catch | R1f |
| B20 | data via `st.global` to peer, flag with release, reader polls with `ld.relaxed.sys` and no acquire fence | race on the payload | ✅ both fronts, exhaustive; **GB300 wrong 1000/1000**: the payload load hits a stale L1 line from the previous round (no `CCTL.IVALL`) | R1f |
| B21 | cross-GPU barrier without an epoch: signals a constant 1 and resets each flag after seeing it; a fast peer's next signal is wiped by the reset | deadlock (lost signal) | ✅ both fronts, exhaustive; GB300 (3 GPUs) correct 1000/1000: appears only under launch skew | R1g |
| B22 | rank restages its symmetric buffer for call i+1 before all peers have read call i (no closing barrier) | cross-GPU race across calls (+ divergent, nondeterministic consequences) | ✅ both fronts, exhaustive; GB300 correct 1000/1000: appears only with skew between ranks | R1g/R4a |
| B23 | writes through the multicast VA, reads through the unicast VA of the same pages, no `fence.proxy.alias` | alias-proxy race | rare | R1g |
| B24 | one-shot all-reduce sums peers in rank-rotated order | **N4**: ranks hold different bits (`divergent`, case `replicas`) | ✅ both fronts; **GB300 (3 GPUs): ranks differ 1000/1000** (45 of 192 words) while every rank is "numerically fine" | R1g/R4a |
| B57 | torch one-shot all-reduce (libtorch SASS, 3 ranks): the start barrier's release is `MEMBAR.ALL.GPU` instead of `.SYS` (SASS patch) | race: a peer reads the staged input without synchronizing with it | ✅ SASS (bounded, two calls); GB300 correct 1000/1000 (latent, like B19) | R4a |
| B58 | same kernel: relaxed start signal (the put's `MEMBAR.ALL.SYS` NOPed) | race on the staged input | ✅ SASS (bounded); GB300 correct 1000/1000 | R4a |
| B59 | same kernel: the start barrier's wait loses its acquire (`CCTL.IVALL` NOPed) | race on peers' buffers | ✅ SASS (bounded); GB300 correct 1000/1000 | R4a |
| B60 | same kernel: no closing barrier (`@P0 EXIT` made unconditional): call 2 restages the symmetric buffer while peers still read call 1 (torch's B22) | race + nondeterministic + divergent | ✅ SASS (bounded); GB300 correct 1000/1000: needs rank skew | R4a |
| B61 | same kernel: lost signal, the closing put swaps in 0 instead of 1 | deadlock | ✅ SASS; **GB300 hangs** (watchdog) | R4a |
| B62 | torch two-shot all-reduce (in place): lost signal, the start put swaps in 0 instead of 1 | deadlock | ✅ SASS; **GB300 hangs** (watchdog) | R4a |
| T4a | **torch one-shot all-reduce as shipped**: the closing barrier is `sync_remote_blocks<memory_order_release>`, whose wait has no acquire, so under the PTX model a peer's call-1 reads of a rank's buffer are not ordered before that rank's call-2 restaging | race (strict PTX model); clean with E-EXIT-FENCE | ✅ SASS (bounded, `PTXV_EXIT_FENCE=none`); GB300 correct 1000/1000: contract-level (load buffering through a control dependency and a kernel boundary) | R4a |

## Grouped GEMM (single GPU)

| id | bug | P | hw behaviour | where |
|---|---|---|---|---|
| B25 | `expect_tx` shrunk for the partial last M-tile of a group (TMA still delivers the full box, with OOB fill) | deadlock | ✅ R3b (trtllm-gen gemm1, SASS patch: the token tile armed with 512 of the 1024 gathered bytes): race + violation, not deadlock: the phase completes after one gather4 load, the MMA reads bytes the other has not written, which may land after exit; **GB300 hangs** every run (docs/rung3b.md) | R3 |
| B26 | empty group (M_g = 0): CTA skips the mainloop but not the barrier phase bookkeeping | parity drift → aliasing in the next group | n/a for R3b: trtllm-gen gemm1 schedules one tile per CTA (static) and experts without tokens get no CTA, so no phase carries across tiles | R3 |
| B27 | tile scheduler prefix sum off by one at a group boundary | write-write race across CTAs (two CTAs own one tile) | ✅ R3b (SASS patch: the epilogue's token offset is `ctaid.y * 4`): race inside the CTA and across CTAs (`--slices`); GB300 wrong 100/100, 6 distinct outputs | R3 |
| B28 | accumulator not reset (`enable_input_d`/accumulate flag) when a CTA's next tile starts a new group | **dependency claim** fails: output (g,i,j) has leaves from group g−1 | ✅ R3b as the per-K-block reset (the second K-block's first MMA accumulates): violation, accumulate into never-written TMEM (K = 2 blocks, so its stage was not used before); GB300 wrong 99/100 | R3 |
| B29 | per-group `tensormap.replace` without `fence.proxy.tensormap::generic` release/acquire | tensormap proxy race | TMA occasionally uses a stale descriptor; n/a for R3b (the maps are kernel parameters) | R3a |
| B30 | block-scale factor index off by one K-block (fp8/fp4) | numerics: spec claim fails, N1 bound violated | ✅ R3b (SASS patch: the weight-scale pointer is not advanced after K-block 0): execution unchanged, the exact-reference bound fails for > 1000 of 8192 outputs (tests/r3b.rs); GB300 = simulator's mutant output bitwise 1000/1000. Also r3_cute_mxf8 (exact spec) | R3b |
| B31 | fp8 accumulation with no promotion (or promotion interval too long) | numerics: N2 bound for large K violated | n/a for R3b: trtllm-gen dsFp8 promotes once per 128-K block because each block has its own scale (a longer interval mis-scales, which is B30), and Blackwell's kind::f8f6f4 accumulates in f32 TMEM, 25 bits below the largest exponent (N-MMA-F8): no narrow accumulator to promote from, unlike Hopper's | R3b |
| B70 | MoE routing: the expert-count atomic adds 0 (flashinfer `routingIndicesClusterKernel`, `IMAD.MOV.U32 R10, RZ, RZ, 0x0` before `ATOMS.ADD` at 0x2af0), so an expert's tokens all get slot 0 | race (several threads store the permutation's slot-0 entries) + routing claim | ✅ R3b whole MoE (fixtures/r3b_trtllm_moe): race; GB300 wrong 1000/1000, = the simulator's B70 output bitwise 1000/1000 | R3b |
| B71 | MoE finalize applies the first expert's routing weight to the second (`LDG.E.U16 [R6.64+0x2]` -> `[R6.64]`) | numerics: the finalize claim fails (> half the outputs); execution unchanged (certified) | ✅ R3b; GB300 wrong 1000/1000, = the simulator's output bitwise 1000/1000 | R3b |
| B72 | MoE finalize reads the second expert's row through the first's permutation index (`LDG.E [R8.64+0x4]` -> `[R8.64]`) | numerics: the finalize claim fails; execution unchanged (certified) | ✅ R3b; GB300 wrong 1000/1000, = the simulator's output bitwise 1000/1000 | R3b |
| B73 | stage ring wrap: the producer does not flip its empty-barrier parity when the stage index wraps (trtllm-gen gemm1 at K = 2048, `LOP3.LUT ... 0x3c` -> `0xf0` at 0x14c0) | fault (re-arms a full barrier whose phase is pending) + deadlock; invisible while no stage is reused (K = 256) | ✅ R3b (fixtures/r3b_trtllm_bmm_k2048); **GB300 faults** on the first run (`cuCtxSynchronize: unspecified launch failure`) | R3b |
| T3b | **trtllm-gen gemm1 with partial token tiles reads the padded tokens' scale slots, which its scale loader skipped** (fixtures/r3b_trtllm_bmm_partial) | violation (unwritten shared memory reaches `ATOMS.MAX`, S-INIT) | ✅ as compiled; benign for the outputs: the TMA store clips padded rows, scale stores are guarded; GB300 = simulator bitwise 100/100, also with smem scrubbed | R3b |
| T3b-2 | **trtllm-gen gemm2 (bf16 output) frees TMEM after a barrier of half its epilogue** (fixtures/r3b_trtllm_moe): warp 0's `UTCATOMSWS.AND` follows `BAR.SYNC 0x9, 0x80` (warps 0-3), while the second M half's MMAs (committed to the barrier warps 4-7 wait on) and warps 4-7's `LDTM` are unordered with it | race (TMEM write vs free) | ✅ as compiled; benign on GB300 (one CTA per SM: nothing reallocates the columns; freed TMEM keeps data); out = simulator bitwise 1000/1000. Variant `fixed` (both halves at `BAR.SYNC 0x9, 0x100`) is certified clean | R3b |

## MoE layer, 4 GPUs (dispatch → grouped GEMM → combine)

| id | bug | P | hw behaviour | where |
|---|---|---|---|---|
| B32 | the compute stream does not wait for dispatch's event: the expert gather (first consumer of the received tokens) runs unordered with DeepEP dispatch | cross-kernel race (launch-group HB) | ✅ race (`Variant::after`); not replayable by the one-stream-per-GPU runner | R4b |
| B33 | dispatch sender: no release fence between the token payload and the channel-tail store | cross-GPU race | ✅ race; GB300 correct 1000/1000 (latent) | R4b |
| B34 | buffer reuse without a barrier: `cached_notify_combine` skips both `barrier_block`s, so combine writes the NVLink buffer the peer's dispatch may still read | cross-GPU race across kernels + deadlock | ✅ race + deadlock; **GB300 hangs** every run | R4b |
| B35 | dispatch sender never publishes its channel tail (planned: CTA-residency deadlock, not modelled) | deadlock | ✅ deadlock; **GB300 hangs** every run | R4b |
| B36 | `notify_dispatch`'s opening barrier loses a signal (planned: rank-order combine reduction, open) | deadlock | ✅ deadlock; **GB300 hangs** every run | R4b |
| B37 | the comm stream does not wait for the compute stream before combine (planned: int32/int64 footprint mismatch, open) | cross-kernel race | ✅ race (`Variant::after`); not replayable | R4b |
| B38 | wrong expert offset: `prepare_grouped_gemm_data` does not load `offs[g-1]` | out-of-bounds TMA (fault) + deadlock | ✅ fault; GB300 output unchanged 1000/1000 (stray rows outside the observed tensors) | R4b |
| B39 | `notify_dispatch`: release fence omitted between the token counts written to the peer and the barrier signal | race | ✅ race; GB300 correct 1000/1000 | R4b |
| B40 | `.gpu`-scoped barrier atomics on the peer GPU's slots (planned B38; planned B40 undersized buffer: open) | scope insufficiency (race) | ✅ race; GB300 correct 1000/1000 (contract-level) | R4b |

## Numerics-only mutants (use the claims engine)

| id | bug | P |
|---|---|---|
| N01 | output `cvt` to bf16 with `.rz` instead of `.rn` | bias claim / N1 bound vs RN baseline |
| N02 | softmax uses `ex2.approx.ftz` on inputs that reach the subnormal range | range-event claim (A7) |
| N03 | split-K partials combined with fp16 atomics | determinism (DAG classes) + N2 bound |
| N04 | fused mul+add contraction changes a bound the PTX author relied on (visible only in SASS) | PTX verdict ≠ SASS verdict |

Every entry gets: a fixture or mutant generator under `mutants/`, the expected ptxv verdict
(property + trace shape), and recorded `hw-rate` and replay-rate. CI fails if a known mutant
stops being caught.
