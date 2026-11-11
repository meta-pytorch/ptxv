# Rung 4a: torch symmetric-memory all-reduce on 3 GPUs (captured, binary only)

(Rung 4b, the DeepEP MoE layer, is the second half of this document.)

**Status (2026-11-03):** torch 2.12's `torch.distributed._symmetric_memory` one-shot and
two-shot all-reduce (the SASS kernels inside `libtorch_cuda.so`) captured from a
`torchrun --nproc-per-node 3` run on GPUs 0, 1, 3, turned into multi-GPU ptxv cases, and
verified on the SASS front-end.

- One call (3 grids, one per GPU): **exhaustive and clean** for both kernels. The simulator's
  one-shot outputs equal every rank's GPU bytes bitwise (1.19M states, 6 s).
- Two back-to-back calls (6 grids, the buffer reused): clean but **bounded** at 400M states
  for both kernels, with every reached terminal state equal to the GPU bytes. Exhaustive
  two-call verdicts need the multi-node explorer (see "Open").
- **One finding in torch as shipped (T4a):** under the strict PTX memory model, the one-shot's
  closing barrier does not order a peer's reads of call 1 before this rank's restaging in
  call 2. Its wait side has no acquire. The case is clean only with the opt-in assumption
  E-EXIT-FENCE (kernel completion acquires).
- 6 SASS mutants (B57–B62), each caught. The two lost-signal mutants **hang on GB300**.
- GB300 runs of both good kernels: bitwise equal to the simulator on all three ranks,
  1000/1000.

## Capture: `tools/capture_dist.py`

```
CUDA_VISIBLE_DEVICES=0,1,3 $PTXV_PYTHON tools/capture_dist.py --nproc 3 \
    --file tools/capture_examples/symm_oneshot.py --out fixtures/r4a_oneshot --exit-fence sys
```

The target file defines `setup(rank, world, group_name)` and `run(**inputs)`:

- `symm_oneshot.py`: two `one_shot_all_reduce_copy(buf, x_i)` calls. The kernel copies the
  rank's input into its symmetric buffer itself, so the second call's copy overwrites the
  buffer peers read in the first.
- `symm_twoshot.py`: `buf.copy_(x)` then two in-place `two_shot_all_reduce_(buf)` calls.

The tool runs torchrun with itself as the worker, under `probes/launchcap` and `cubincap`.
`%r` in their output paths is the rank, so each rank writes its own directory. Each rank:

1. runs `setup()`, warms up, synchronizes with a barrier;
2. snapshots its memory: caching-allocator blocks, its own VMM allocations (cut to the
   symmetric layout), and the named tensors;
3. runs `run()` once more under allocator history;
4. writes its launches, VMM log, outputs and byte dumps.

The parent then merges all ranks into one case.

### What launchcap now records

| record | content |
|---|---|
| launches | added: `stream`, `dev` (cuCtxGetDevice), `t` (CLOCK_MONOTONIC ns, comparable across the ranks' processes) |
| `vmm` (with `LAUNCHCAP_VMM=1`) | `cuMemCreate`, `cuMemExportToShareableHandle` (FABRIC handle bytes; for POSIX fds the fd's dev/inode), `cuMemImportFromShareableHandle`, `cuMemAddressReserve`, `cuMemMap`/`Unmap`, `cuMemRelease`, `cuMulticastCreate`/`AddDevice`/`BindMem`, `cuIpcGetMemHandle`/`OpenMemHandle` |
| `ev` | `cuEventRecord(WithFlags)`, `cuStreamWaitEvent`, `cuStreamWriteValue32` |
| `op` | added: `cuMemsetD32Async` (torch's signal-pad reset) |

torch's `DriverAPI` resolves its driver calls through `cudaGetDriverEntryPointByVersion`.
cudart looks those up with the `cuGetProcAddress` it got from `cuGetProcAddress` itself, so
launchcap now hands out its hook for that lookup too. A runtime launch's driver call
underneath is not recorded twice (`in_rt`).

### How peer memory appears, and how it is resolved

What the recordings show on this box (GB300, driver 13.0):

- **Allocation.** Each rank's symmetric buffer is one `cuMemCreate` of 2 MB (one granule)
  with `requestedHandleTypes = FABRIC` (8). It is exported as a **64-byte fabric handle**.
- **Peer mappings.** Every peer imports that handle (`cuMemImportFromShareableHandle`, type
  8, the *same 64 bytes*), then reserves and maps it at an address of its own. So a physical
  allocation has a global name, the fabric handle, and its owner is the rank that created it.
  For POSIX-fd exports the tool matches the fd's (st_dev, st_ino); for legacy IPC, the
  64-byte `cudaIpcMemHandle`.
- **Layout.** Data buffer at offset 0 (`buffer_size` = 256 bytes here), signal pad right
  after it (`signal_pad_size` = 9216 bytes, 3 ranks × 768 u32 slots). The tool reads this
  layout from the `_SymmetricMemory` handle that `setup()` returns.
- **Kernel arguments.** The kernels take **device-side pointer tables**: `T** buffer_ptrs_dev`
  and `uint32_t** signal_pads_dev`, two 24-byte arrays in caching-allocator memory, written by
  a `cudaMemcpy` at rendezvous. They hold *this rank's* virtual addresses of every rank's
  buffer.
- **Resolution.** A pointer v seen by rank r is resolved through r's mappings, as
  `(owner, owner's address + offset)`. Then the owner's buffer at that address becomes the
  case buffer (`buf_r1`, `sig_r2`, ...). Every buffer a parameter references is scanned for
  8-byte words that resolve this way. Those words become the buffer's `ptrs` (new in
  `crates/case`, `Buffer::ptrs`), filled in with the right buffer's address by the simulator
  and the hw runner. In both cases every pointer resolved; `unresolved` is empty.
- **Multicast (NVLS).** torch also creates a 2 MB multicast object (`cuMulticastCreate`,
  3 devices). Each rank binds its own allocation to it and maps it (`multicast_ptr`). The
  capture records and reports it (`capture.json` `multicast`, `memory_map`). Neither the
  one-shot nor the two-shot kernel references it: no parameter or table word points into
  it. A pointer into a multicast mapping would be reported as `unresolved`
  (`why: multicast`). Modelling `multimem.*` (B23) is only needed for `multimem_all_reduce_`.
- **Order and streams.** All launches are on each rank's stream 0, with no events. The case
  lists every rank's launches in its stream order, with `device = rank`. Cross-rank
  submission times are kept in `capture.json` (`t_ns`), but they do not constrain the model:
  grids on different GPUs are unordered (E-PDL).
- **Replicas.** Outputs with the same name on every rank (`o1`, `o2`; two-shot's `buf`) form
  replica groups, which must end bitwise equal (N4).

The case files are `fixtures/r4a_oneshot` and `fixtures/r4a_twoshot`:

- `case.toml`;
- `k0.cubin`: libtorch's `CUDASymmetricMemoryOps.cu` module, with the kernels that are not
  launched NOPed out;
- `data/`, `expected/`: per-rank bytes;
- `capture.json`.

## The kernels (SASS, read off the capture)

`one_shot_all_reduce_kernel<float, 16, -1>` (world size 3 takes the generic `-1` variant),
1 CTA × 32 threads per rank, 64 floats:

```
copy      input -> buffer_ptrs[rank]             (LDG/STG.E.128, 16 lanes)
barrier   put: MEMBAR.ALL.SYS; CAS.SYS(pad[peer][rank], 0 -> 1) loop     (lanes 0..2)
          wait: CAS.SYS(pad[rank][peer], 1 -> 0) loop; CCTL.IVALL         acq_rel
BAR.SYNC
reduce    out[i] = ((0 + buf_0[i]) + buf_1[i]) + buf_2[i]   (pointer table, 8/4/2/1 unrolled)
BAR.SYNC
barrier   put: MEMBAR.ALL.SYS; CAS loop
          wait: CAS loop, no CCTL.IVALL                                   release only
```

`two_shot_all_reduce_kernel_inplace<float, 16, -1>`:

- start barrier: a relaxed put (no fence) and an acquiring wait;
- reduce-scatter: rank r reduces its third of every buffer;
- all-gather: rank r writes its sum into every peer's buffer (with `__cuda_sm20_rem_u64` /
  `div_u64` calls);
- closing barrier: acq_rel.

## Lift gaps closed (each validated on GB300)

| gap | fix | validation |
|---|---|---|
| `ATOM.E.CAS.STRONG.SYS` / `ATOMG.E.CAS.*` (compare-and-swap returning the old value) | `AtomOp::Cas` (ir, sass lift, exec). A failed compare only reads: no write, nothing published | `fixtures/s_cas` (new): generic and global CAS, succeeding and failing, through a device pointer table; GB300 bitwise 200/200 |
| CAS signal loops (`while (cas(a,0,1) != 0);`) were not E-SPIN polls, so the explorer stepped every failed attempt | `spin::poll_loops` accepts a CAS as the poll op, release fences in its body, and loop-invariant prefix writes; the enabledness dry-run, blocked-poll race read and DPOR/explorer footprints cover `Op::Atom` | s_cas handoff (64 put/wait pairs) clean; its lost-signal mutant is a deadlock and **hangs on GB300**; its relaxed-put mutant is a race |
| scoreboard false positive: `LDG.E.64 R8, [R6.64]` ... `LDG.E.64 R6, [R6.64+0x18]` reported as WAR | the queued-load rule (L-OPREAD) now also applies when the later load reads the register as its address | one-shot bitwise = GB300 on 3 ranks |
| pointer tables in device memory | case `Buffer::ptrs`, `init_bytes_with`; sim `build_launch` and hw runner fill them | s_cas, r4a (hw 1000/1000) |
| binary mutants beyond control bits | `CtrlPatch` gains `nop` (the kernel's padding NOP, control bits kept), `bits = [[start, width, value]]` over the 128-bit word, and `expect` (the patched disassembly, checked) | every r4a mutant disassembles as intended |

Each such run was single-GPU on GPU 0 (s_cas), except the 3-rank r4a runs on GPUs 0, 1, 3.

## Verdicts (SASS front-end; good kernels with `exit_fence = "sys"`)

| case | scope | verdict | states | sim = GB300 |
|---|---|---|---|---|
| r4a_oneshot | call 1 (3 grids) | **clean, exhaustive**; strict model also clean | 1,188,823 | bitwise on all 3 ranks (o1) |
| r4a_twoshot | call 1 (3 grids) | **clean, exhaustive** | 1,005,247 | (expected bytes are after 2 calls) |
| r4a_oneshot | 2 calls (6 grids) | clean, **bounded** (no finding; 1 distinct output over 5.4M terminal states) | 400,000,119 (cap, 50 min, ~25 GB) | o1, o2 bitwise on all 3 ranks |
| r4a_twoshot | 2 calls (6 grids) | clean, **bounded** (no finding; 1 distinct output over 5.0M terminal states) | 400,000,114 (cap, 46 min) | buf bitwise on all 3 ranks |
| r4a_oneshot | 2 calls, **strict PTX** (`PTXV_EXIT_FENCE=none`) | **race** (T4a), found within 1.3M states | | |

`--certify` does not apply: C-ORDER refuses the unordered strong accesses of a CAS put and
its wait (E-WELLSYNC covers load polls, not CAS signals), so these cases are explored.

### T4a: the closing barrier of torch's one-shot is release-only

```
RACE: read-write race: k2.t0 sass:0x1220 `LDG.E.128 R4, [R4.64]`   (peer, call 1 reduce)
      and k3.t0 sass:0x02b0 `STG.E.128 [R10.64], R4`                (this rank, call 2 copy)
```

The chain is:

1. Peer p reads rank r's buffer.
2. p runs `MEMBAR.ALL.SYS` and puts a CAS signal on r's pad.
3. r's closing wait CAS reads that signal, but **relaxed, with no `CCTL.IVALL`**
   (`sync_remote_blocks<std::memory_order_release>`).
4. r's kernel exits, and its next kernel stores into the buffer.

Under the PTX model, p's read and r's later store are unordered. This is load buffering
through a control dependency and a kernel boundary. Hardware does not perform a store before
the branch it depends on resolves, and GB300 never shows it (1000/1000). The two-shot's
closing barrier is acq_rel and has no such gap.

ptxv keeps the strict model by default. A case can state the extra guarantee with
`exit_fence = "sys"` (E-EXIT-FENCE), which makes kernel completion a `fence.acq_rel.sys`.
The verdicts above use it, and each verdict lists it under `assumes:`. The fix in torch
would be an acquire on the closing wait (`memory_order_acq_rel`), i.e. one `CCTL.IVALL`.

### Mutants

All are SASS patches of the captured cubin (`patch`), checked on the full two-call case.

| id | patch | expected | ptxv | GB300 (3 GPUs) |
|---|---|---|---|---|
| B57 | one-shot start put: `MEMBAR.ALL.SYS` → `.GPU` (bit 76) | race | race | correct 1000/1000 (latent, like B19) |
| B58 | one-shot start put: the fence NOPed (relaxed signal) | race | race | correct 1000/1000 |
| B59 | one-shot start wait: `CCTL.IVALL` NOPed (no acquire) | race | race | correct 1000/1000 |
| B60 | one-shot: `@P0 EXIT` → `EXIT` before the closing barrier (buffer reuse without a barrier) | race, nondeterministic, divergent | all three: replicas differ in 64 words | correct 1000/1000 (needs rank skew) |
| B61 | one-shot closing put swaps in 0 (`IMAD.MOV.U32 R9, RZ, RZ, 0x0`): lost signal | deadlock | deadlock (568,760 states) | **hangs** (watchdog), every run |
| B62 | two-shot start put swaps in 0: lost signal | deadlock | deadlock | **hangs** (watchdog), every run |

The two-shot's closing barrier protects only the *consumer* of the result. In the two-shot
each rank reads only its own slice of every buffer, and it wrote that slice itself in the
previous call. So dropping or weakening the closing barrier changes nothing *within* these
two calls: the mutant would be "clean". Catching it needs the consumer kernel in the case,
which is rung 4b work (multi-kernel launch groups).

## Tests

All run under `cargo test --release`.

- `crates/sim/tests/wellsync.rs` `r4a_certified_first_call_matches_exploration`: both kernels,
  one call, exhaustive and clean, and equal to the certified verdict. The one-shot runs under
  both the strict model and E-EXIT-FENCE and must equal the GPU bytes.
- `crates/sim/tests/fixtures.rs`:
  - `r4a_oneshot_strict_model_reports_the_reuse_race`: T4a.
  - `r4a_mutants_b57` ... `r4a_mutants_b62`: B57–B62, bounded at 3M states, one test each.
  - `s_cas_variants`.
- `captured_outputs_match_the_gpu` skips multi-GPU cases, which need millions of states.

## Open / next

- **Two-call exhaustiveness at scale.** The two-call good cases are in the hundreds of
  millions of states on one box (32-lane warps whose 3 signalling lanes split off, × 6 grids
  with arbitrary skew). **Done by certificate:** the certificate rule for CAS semaphores
  (E-WELLSYNC C-CAS) works because a slot's puts and waits alternate 0/1, so each successful
  CAS's reads-from is determined. With it, `ptxv check --certify` decides both good kernels,
  one call and two, in one run of about 165 steps (0.01 s): clean, and bitwise equal to the
  GPU bytes. One-call certified verdicts equal the exhaustive ones. The strict two-call
  one-shot is refused (its run reports T4a), and so is every two-call mutant
  (`crates/sim/tests/wellsync.rs`). The multi-node explorer (docs/multinode.md) remains the
  way to an exhaustive check that does not rest on E-WELLSYNC.
- **NVLS / `multimem`** (B23): the capture already records the multicast object and its
  bindings. The lifter and memory model need `multimem.ld_reduce` / `st` and the alias
  proxy.
- **Rank skew on hardware.** B60 needs one rank delayed between its start barrier and its
  reduction. A SASS delay-injection mutant (NANOSLEEP in place of a no-op) would be the
  replay.
- **4 ranks** when all four GPUs are free: `--nproc 4` with `CUDA_VISIBLE_DEVICES=0,1,2,3`. Nothing
  in the tool assumes 3. torch then picks the `k_world_size = 4` template instances.

# Rung 4b: an expert-parallel MoE layer on 2 GPUs (DeepEP + torch, captured, binary only)

**Status (2026-11-03):** one MoE layer captured as a single launch group from a
`torchrun --nproc-per-node 2` run on GPUs 0 and 1: DeepEP intranode dispatch → torch gather +
CUTLASS grouped GEMM + routing-weight scaling + `index_add_` → DeepEP combine. 28 launches
(14 per rank, 13 distinct kernels from 8 modules), 17,060 threads, SASS front-end.

- **What it runs:** T=8 tokens of width H=128 (bf16) per rank, E=4 experts (2 per rank), top-2
  routing (`tools/capture_examples/moe_deepep.py`).
- **Why this stack:** DeepEP was already installed in `$PTXV_PYTHON`'s env (the real
  production EP library: NVLink buffers through CUDA IPC, host-mapped counters, a comm stream
  with events, TMA bulk copies, warp-vote barriers). Its intranode kernels exist only for 2, 4
  and 8 ranks, and 4 would need a GPU that was in use by another job, so this is 2 ranks. The GEMM is torch's own
  `_grouped_mm` (the rung 3a CUTLASS kernel), so the expert compute is the real thing too.
- **Run:** one deterministic schedule of the whole layer takes 25 s. Its findings are E-BULK-EXIT
  contracts only, and **both ranks' outputs equal the GPU bytes bitwise**.
- **Assumptions:** this needs two new opt-in assumptions, E-POLL-ACQ and E-SMEM-CUMUL. Without
  them DeepEP is racy under the strict PTX model (T4b below).
- **Dependency claim** (a test): output token t of rank s depends exactly on its top-k
  experts.
- **Mutants:** B32–B40, all 9 caught. Hardware replays on 2 GB300s:
  - three hang: B34, B35, B36;
  - five are latent (correct 1000/1000);
  - two are stream-order mutants, which the runner cannot replay.
- **Hardware:** the good layer is bitwise equal to the simulator on both ranks, 1000/1000
  runs.
- **Not exhaustive.** Bounded exploration (300k states) finds nothing, but that only covers
  interleavings near the start. See "Verdict and scale".

## Capture

```
CUDA_VISIBLE_DEVICES=0,1 $PTXV_PYTHON tools/capture_dist.py --nproc 2 \
    --file tools/capture_examples/moe_deepep.py --out fixtures/r4b_moe
```

The setup pins the shape of the run:

- `deep_ep.Buffer.set_num_sms(2)`, `Config(2, 6, 256, 6, 128)`: one channel, i.e. 2 CTAs of
  768 threads for dispatch and for combine.
- `torch._C._set_sm_carveout_experimental(sm_count - 2)`. The CUTLASS grouped GEMM is
  persistent: torch launches one CTA per SM, 152 × 384 threads, almost all idle at this size.
  The carveout is torch's own knob for overlapping communication with GEMMs, and it leaves
  the GEMM one 2-CTA cluster. This cut the case from 132k threads to 17k.

What `capture_dist.py` / launchcap gained:

| what | how |
|---|---|
| legacy CUDA IPC (DeepEP's NVLink buffers: `cudaIpcGetMemHandle` / `OpenMemHandle`) | `ipc:<handle>` physical ids, sized from the exporter's `cuMemAlloc`; each importer's pointer maps onto the owner's buffer |
| plain `cuMemAlloc` / `cuMemHostAlloc` regions (DeepEP's own buffers and host-mapped counters) outside torch's caching allocator | launchcap logs `alloc` / `free` / `host_alloc` / `host_devptr`. Live regions not in `torch.cuda.memory_snapshot()` are read back (≤ 16 MB) and become `dev*_r*` / `host*_r*` buffers |
| H2D copies between kernels (the expert-major row order the host builds after DeepEP's CPU sync) | launchcap records the bytes of uploads ≤ 1 MB; merge folds an upload between kernels into the destination's initial contents (a note) |
| comm stream ↔ compute stream | every stream switch between consecutive launches must be ordered by an event recorded on the old stream and waited on the new one; otherwise a warning. Here all 4 switches per rank are (notes in `capture.json`) |
| `REPLICAS` declared by the target module | replica groups come from it only (they were inferred before) |
| cubins | each one is slimmed (`tools/cubin_slim.py`) to the launched kernels. nvdisasm rejects the full DeepEP module (internode NVSHMEM relocations) |

Six "unresolved pointer" warnings remain. All are words in torch's by-value kernel
arguments that only look like pointers:

- `0x400000004` and `0x400000200` are pairs of 32-bit sizes;
- `0x411111112` is an offset-calculator constant;
- one host-stack-like value (`0xfff9...`) is unused padding past `dims` in a `TensorInfo`.

None is dereferenced, which the run confirms: every access resolves.

The launch group is serialized per GPU in capture order. That is valid because every
cross-stream dependency is an event that the order respects. The stream-order mutants below
edit it (`Variant::after`).

## The kernels

| rank | kernels (in order) |
|---|---|
| both | `get_dispatch_layout` (comm stream), `notify_dispatch<2>`, `dispatch<2,768,8192>` |
| 0 | `vectorized_gather` (index_select of 9 rows), `prepare_grouped_gemm_data`, CUTLASS sm100 grouped GEMM, `index_elementwise` (weights gather), copy f32, `mul`, `fill`, `indexFuncLargeIndex` (index_add_), bf16 copy |
| 1 | `indexSelectSmallIndex` (index_select of 5 rows), then the same, with `indexFuncSmallIndex` |
| both | `cached_notify_combine<2>`, `combine<bf16,2,768,4096>` (comm stream) |

The two ranks run different torch kernels because their row counts differ.

## Lift and model gaps closed

| gap | fix |
|---|---|
| DeepEP's `barrier_block` and channel waits: warp-level poll loops, with a forward branch around the poll load, `BSYNC`, `BRA.DIV`, `VOTE.ALL`, a clock timeout and a back branch. E-SPIN only knew straight-line loops, so the explorer spun forever | `spin::wpoll_loops` + `spin::wpoll_dry` (E-SPIN, *warp-level poll loops*): a split at the loop head runs one iteration on a copy, and blocks if that is a stutter step (same registers, predicates and memory). Covers combine's coordinator loop too (`while (true) { if (done) return; ... }`: unconditional back branch, `EXIT` inside, same-value republishing stores) and SASS `LDS` polls of the shared tail relay. Blocked heads read their flags for the race detector and the DPOR/explorer footprints |
| copy loops reading peer data with `ld.volatile` were reported as polls that "may not terminate" | `poll_loops` ignores a loop whose exit does not depend on the loaded value (forward taint, killed by overwrites) |
| `CS2R Rd, SR_CLOCKLO` (clock64) | reads 0 (E-CLOCK, new): timeouts never fire |
| `CALL.ABS.NOINC `(vprintf)``, `32@lo/hi($str)` and `FN+.L@srel` operands | the call is a no-op (E-CLOCK), and the relocations become immediates (label address or 0). The `BPT.TRAP` after the call is modelled |
| `c[0x0][0x2f8]` (generic base of the local window, for vprintf argument pointers) | an unmapped constant (S-CBANK) |
| `BRA.DIV`, `WARPSYNC.COLLECTIVE`, `ENDCOLLECTIVE` | `Op::DivCheck` runs the fast path and reports `unmodeled` if a live lane of the mask is elsewhere (E-LOCKSTEP) |
| `UBLKCP.S.G` / `UBLKCP.G.S` (1-D bulk copies, size in 16-byte units) | `TmaLoad` / `TmaStore` with `dims = 0`: an ad-hoc 1-D map over the global range (`bulk_map`) |
| generic `ATOM.E.{ADD,OR,AND,XOR,EXCH}`, `ISETP ... XOR`, `VIADDMNMX` with `-R`, `LEA.HI` with a carry-out | ir + sass lift + exec |
| CUTLASS 4.x grouped GEMM with several groups per CTA: async tensormap-updater warps (8–11) update and `fence.proxy.tensormap`-acquire each map, then hand it to the load warps through an mbarrier pipeline. The model required the TMA's own thread to have fenced | S-TMAP: a fence of the same bytes by another thread of the CTA that happens-before the TMA also counts (`tools/capture_examples/grouped_mm_2groups.py` reproduces this on one GPU) |
| TMA loads/stores inside the tensor map's bounds but past the end of its allocation panicked the shadow memory | reported as a `fault` (B38) |
| per-step O(threads) scans (`grid_complete`, the warp-aligned and `BRA.DIV` lane counts, wellsync's taint sweep, grid gates) | per-grid / per-warp ranges. A 17k-thread run went from not finishing in 30 minutes to 25 s |
| mutants of a launch's own cubin or of another kernel of the module | `CtrlPatch` gains `kernel` and `cubin`. The hw runner loads patched own cubins (`PTXV_PATCHED_PREFIX`) |
| stream-order mutants | `Variant::after = [[launch, after]]` (`Case::for_variant`, `Grid::after`, `Launch::prev_grid`) |

## T4b: DeepEP's synchronization is racy under the strict PTX model

The deterministic run without the opt-ins reports two families of races
(`r4b_moe_strict_model_reports_deepep_races`):

1. **`barrier_block`.** The signal side is a release: `MEMBAR.ALL.SYS`, then
   `ATOM.ADD.STRONG.SYS` on the peer's slot. The wait side is not an acquire: it polls the
   slot with relaxed `ld.volatile`, votes `__all_sync`, then `__syncthreads`, with no fence.
   So every handoff through it is unordered by PTX:
   - per-rank token counts written into the peer's buffer in `notify_dispatch`, read after
     its second barrier;
   - the NVLink buffer reused from dispatch to combine across `cached_notify_combine`;
   - channel state zeroed in notify, then used by dispatch.
2. **Combine's tail relay.** Coordinator lanes acquire each channel tail
   (`LDG.E.STRONG.SYS` + `CCTL.IVALL`) and pass it to the receiving warps through
   `st.volatile.shared`, with no fence. The warps poll it, then read the token data with
   `LDG.E.128.STRONG.SYS`. The data reads are not ordered after the senders' writes.

GB300 never shows either: the good layer is bitwise equal to the simulator on 1000/1000
runs. Two opt-ins state the hardware facts the verdict relies on. `fixtures/r4b_moe` sets
both, and its verdicts list them under `assumes:`.

- **E-POLL-ACQ.** Leaving a *warp-level* poll loop acquires what the polls observed.
  Straight-line polls (dispatch's and combine's channel-tail waits, which have their own
  `CCTL.IVALL`) keep the strict model. The writer side is untouched, so a missing release
  fence or a too-narrow signal scope is still caught (B33, B39, B40).
  **Refuted on GB300** (2026-11-04, docs/poll-acq-litmus.md). After this loop shape, a weak
  read of a line valid in the reader's L1 returns the pre-release value (near 100% of reads
  across SMs and over NVLink). DeepEP escapes for reasons outside the rule: the barrier's
  entry `fence.acq_rel.sys` invalidates L1 before the poll, and combine reads volatile. So
  the T4b races are real PTX races that are latent on GB300; they are not false alarms.
- **E-SMEM-CUMUL.** A SASS shared store releases, at CTA scope, what its thread acquired.
  GB300 shared memory is SC (docs/smem-litmus.md). The thread's own unfenced global writes
  are not covered, which is where the litmus tests did see stale data. **Validated on GB300**
  for combine's shape (coordinator `CCTL.IVALL` acquire, `STS` relay; no stale read,
  docs/poll-acq-litmus.md). `CCTL.IVALL` invalidates the whole SM's L1.

Dispatch's ring-slot reuse is a third such race: the receiver's `UBLKCP.S.G` slot read is
not ordered before its relaxed head store (no MEMBAR). It is latent on GB300 too, because
the mbarrier wait completes before the head store issues: 0 overwritten copies in 2.13e10,
while a control that publishes the head before the wait sees overwrites
(docs/poll-acq-litmus.md, "Dispatch ring-slot reuse").

Each fix in DeepEP would be one instruction: an acquire fence after the barrier's vote, and
`st.release.cta` (`MEMBAR.ALL.CTA` + `STS`) for the relay.

## Verdict and scale

| what | result |
|---|---|
| `PTXV_RUN_ONCE=1 ptxv check fixtures/r4b_moe --variant good` (one deterministic schedule) | terminal; only E-BULK-EXIT contracts (CUTLASS epilogue TMA store and DeepEP's `UBLKCP.G.S` at CTA exit, as in r3a); `y_r0`, `y_r1` bitwise equal to the GPU (25 s) |
| `ptxv check fixtures/r4b_moe --variant good --max-states 300000` | no finding, incomplete. No terminal state reached: the frontier is still in the first kernels' interleavings |
| `--certify` (before 2026-11-04) | refused: C-WAIT (CUTLASS's peeks on the second tile), C-ORDER (`index_add_`'s f32 `ATOM.ADD` from different CTAs), and LDS/STS relay accesses that the certificate counts as races (the explorer's `smem` pass would drop those) |
| `--certify` (2026-11-04) | the three are resolved (see "Certificate and launch groups" below). Still refused, now for DeepEP only: C-ORDER on the warp-level poll loads of `barrier_block` against the peer's signal atomics (notify, cached notify), on dispatch's channel-tail store against the receiver's poll, and on combine's shared-memory progress protocol |

Races are happens-before facts, so a single run that reaches both accesses reports them,
whatever the order. That is how every mutant below is caught. A full verdict needs
exploration well beyond one box. The multi-node explorer (docs/multinode.md) is the
path. Suggested command, **not submitted**:

```
sbatch --ntasks=16 --cpus-per-task=48 --mem-per-cpu=4G $PTXV_SLURM_SBATCH_ARGS tools/ptxv.sbatch \
    check fixtures/r4b_moe --variant good --max-states 3000000000
```

A phase decomposition would make it tractable. Without PDL, each GPU's grid k+1 starts only
after grid k completes, so only concurrently running DeepEP kernel pairs interact across GPUs.
The torch kernels in between are single-GPU and independent per CTA. That is future work.

**Numerics.** `r4b_moe_token_depends_exactly_on_its_experts` replaces each expert's weights in
turn and runs the layer. Exactly the output rows of the tokens routed to that expert change,
on both ranks. The layer is also deterministic:

- `index_add_` adds at most 2 values per element onto 0, and f32 addition is commutative;
- combine sums each token's per-rank contributions in a fixed rank order (`j = 0..R-1` in
  DeepEP's combine loop).

Both ranks' bytes equal the GPU's. A rank-order mutant of combine (BUGS.md's original B36)
needs a SASS edit of the reduction loop and is open.

## Mutants

All are checked on the full 28-launch case, using `PTXV_RUN_ONCE` and the test
`r4b_moe_mutants`. The hardware runs use `PTXV_HW_EXPECT=run PTXV_GPUS=0,1 ptxv hw
fixtures/r4b_moe --variant V --runs 1000`.

| id | bug | ptxv | GB300 (2 GPUs) |
|---|---|---|---|
| B32 | the compute stream does not wait for dispatch's event: rank 0's expert gather is ordered after nothing (`after = [[6, -1]]`) | race (gather vs dispatch's bulk store), outputs differ | not replayed (one stream per GPU) |
| B33 | dispatch sender: the release fence before the channel-tail store is NOPed | race | correct 1000/1000 (latent) |
| B34 | buffer reuse without a barrier: `cached_notify_combine` skips both `barrier_block`s | race + deadlock | **hangs**, every run |
| B35 | dispatch sender never publishes its channel tail (the store is NOPed) | deadlock | **hangs**, every run |
| B36 | `notify_dispatch`'s opening barrier loses a signal (`ATOM.ADD` NOPed) | deadlock (+ races) | **hangs**, every run |
| B37 | the comm stream does not wait for the compute stream before combine (`after = [[19, 5]]`) | race (combine reads the bf16 expert outputs during the cast) | not replayed |
| B38 | wrong expert offset: `prepare_grouped_gemm_data` does not load `offs[g-1]` | fault (TMA past the gathered rows' allocation) + deadlock | `y` unchanged 1000/1000: the stray rows land outside the observed tensors |
| B39 | `notify_dispatch`: the release fence before its closing barrier is NOPed, so the token counts are not published by the signal | race | correct 1000/1000 |
| B40 | scope too narrow: the closing barrier's signals are `ATOM.ADD.STRONG.GPU` (bits 77..80 = 7) instead of `.SYS` | race (not morally strong across GPUs) | correct 1000/1000 (contract-level, like B56) |

Compared with BUGS.md's plan:

- B32, B33, B34 and B39 are the planned bugs.
- B40 is the planned B38 (`.gpu` scope).
- The planned B35 (CTA residency), B36 (rank-order reduction), B37 (int32/int64 footprint)
  and B40 (undersized buffer) are replaced:
  - residency is not modelled;
  - the other three need routing sweeps or reduction-loop edits.

## Tests

- `r4b_moe_run_matches_the_gpu`: the good layer's run has contracts only, and both outputs are
  bitwise equal to the GPU.
- `r4b_moe_mutants`: B32–B40, the expected classes.
- `r4b_moe_token_depends_exactly_on_its_experts`: the dependency claim.
- `r4b_moe_strict_model_reports_deepep_races`: T4b.

## Certificate and launch groups (2026-11-04)

### The three certificate refusals

| refusal | what it was | resolution |
|---|---|---|
| C-WAIT on CUTLASS's second-tile producer peek (`TRYWAIT P0` at 0x4140) | **a model artifact**: the peek's taint survived the next tile's blocking `TRYWAIT P0` (0x43b0), which overwrites `P0`. Its retry branch was then taken for a read of the peek | a try_wait now ends the taint like any other write (E-WELLSYNC "Peek taint ends at a try_wait") |
| C-ORDER on `index_add_`'s `ATOM.E.ADD.F32.FTZ.RN` from different CTAs | a real order dependence in general, but not here: at most two adds per element onto +0, which commute exactly | **C-FADD**: enumerate every linear extension of happens-before over each word's unordered fp32 adds, fold with the executor's add; equal bits are accepted. If only the output words differ and nothing reads them later, the verdict is `nondeterministic` with every reachable output (c_triton_rowsum_atomic: 9 outputs, the GPU's among them). Otherwise refused, with the values |
| race on smem byte 0x400 (combine `STS [R6]` 0x28c0 vs `LDS [R8]` 0x35d0) | **an artifact of the weak pass**: the certificate ran only on the kernel with LDS/STS plain (`smem::two_pass`). In the strong model (S-SMEM-STRONG) these are relaxed CTA-scope strong accesses, so the relay is a strong-access pair, not a race. It is the DeepEP tail relay that E-SMEM-CUMUL is about, not a new pattern | `check_certified` now tries the strong model when the weak one refuses. The strong model then refuses combine on C-ORDER: its coordinator / receiver progress protocol reads schedule-dependent values (next item) |

Two more changes were needed on the way:

- *BSYNC ends a strong store's clock window*: CUTLASS's tensormap build (lane 0 `STS`, BSYNC,
  all lanes `LDS`) was unordered in the strong model.
- *Uncapped reasons*: `PTXV_WS_ALL=1` prints every reason. The cap of 8 per condition had
  hidden the DeepEP C-ORDER reasons behind the CUTLASS and `index_add_` ones.

With these, the torch expert computation (gather, `prepare_grouped_gemm_data`, the CUTLASS
grouped GEMM on both ranks, weight scaling, `index_add_`, casts: 18 launches) **certifies as
one group** (test `r4b_torch_group_certifies`).

### What remains for the certificate: DeepEP's synchronization

Each DeepEP kernel synchronizes across GPUs through polls of flags that peers write
concurrently:

- `barrier_block`: warp-level poll loops (`wpoll`) over counters that both ranks
  `ATOM.ADD`;
- dispatch: the receiver's lane-0 poll of the channel tail; the poll loop has a clock
  timeout and is not an E-SPIN `poll_loops` shape, so C-POLL does not apply;
- combine: the coordinator's progress loop republishes the minimum of the warps' heads.

C-POLL covers only straight-line polls on non-atomic writes. More fundamentally:

- dispatch's receivers consume chunks between the tail values they happen to observe;
- combine's coordinator writes whatever minimum head it observes.

In both, the *sequence of events* of a thread depends on the schedule. A certificate is one
run whose events every schedule repeats, so it cannot decide these kernels. Exploration must.

**Update (C-RING, below):** dispatch now certifies. Combine still does not.

### Launch groups (E-LAUNCH-GROUPS, `ptxv check --groups`)

`--groups "2,3;4,5;6-18,21-25;19,26;20,27"` checks:

| group | launches |
|---|---|
| 1 | layout |
| 2 | notify |
| 3 | dispatch |
| 4 | torch |
| 5 | cached notify |
| 6 | combine |

One group after another:

- each group by the certificate, else by exploration, from the state the previous group
  left;
- a cross-group check: the earlier groups' race shadow is demoted to plain accesses, plus
  obligations for strong reads and unordered strong writes;
- an explored group continues from the meet of its terminal states.

`--groups auto` certifies the whole case and splits off the refused launches. The argument
is in ASSUMPTIONS.md.

Two explorer changes make the DeepEP groups explorable at all:

- **Multi-grid partial-order reduction.** The synchronization reduction used to switch off
  whenever two grids were live, which is always the case for a 2-GPU collective. It now runs
  once nothing is left to launch. Notify on 2 GPUs went from > 8M states (incomplete, frontier
  flat at 90k) to **24,329 states, complete**: 99 terminal states, one final memory.
  - `PTXV_CHECK_WALKS=300` on that group: 300 random executions of the unreduced model all
    end in end states the reduced search reached, and 0 findings were missed.
  - The multi-GPU fixture tests (r1f, r1g, r4a, s_cas) pass with it.
- **E-BAR-OVER walk with barrier ids in registers.** DeepEP's `bar.sync (tid >= 384), 384`
  recomputes its id from `%tid`. The walk now evaluates MOV / SEL from thread constants and
  reads registers flow-sensitively. Before, every warp of the 768-thread CTA counted for
  both halves' barriers, which oversubscribed both and serialized their arrivals.

| group | verdict |
|---|---|
| 1 layout (2 launches) | certified, clean (192 steps) |
| 2 notify_dispatch (2) | **exhaustive**: clean, 24,329 states, 99 terminal states, one final memory (13 s) |
| 3 dispatch (2) | **not decided**: about 30k states in 13 min, 2 terminal states, frontier growing. Each state costs about 400 KB and about 10 ms (look-ahead over 24 warps). The remaining branching: named-barrier arrivals still counted as oversubscribed at `0x7f40` / `0x3780`, the receivers' `UBLKCP.S.G` completions, the tail and head polls |
| 4 torch (18) | certified, contracts only (from a dispatch end state) |
| 5 cached_notify_combine (2) | **exhaustive** from a dispatch end state: clean, 3,378 states, 33 terminal states, one final memory (33 s) |
| 6 combine (2) | **not decided**. 25k states: clean, 345 terminal states, one output, bitwise equal to the GPU on both ranks. A 200k-state run (4 passes: LDS/STS plain and strong, each also without the peek changes) ran out of its 90 GB cap with the frontier at 61k: about 450 KB per state |

"From a dispatch end state" means the debugging shortcut `PTXV_GROUPS_RUN=3`: groups 1-3 run
once, deterministically. A full chain needs dispatch's meet, which needs dispatch decided.

### Mutants in group mode

```
PTXV_GROUPS_BOUNDED=1 PTXV_POR_SEED_WALKS=0 [PTXV_GROUPS_RUN=k] ptxv check fixtures/r4b_moe \
    --variant V --groups "2,3;4,5;6-18,21-25;19,26;20,27" --max-states 25000
```

`PTXV_GROUPS_BOUNDED` continues past an incomplete group from the meet of the terminal states
it found, and marks the verdict incomplete.

| id | where the bug is | caught in group mode |
|---|---|---|
| B32 | torch gather ordered after nothing (stream) | race in the torch group: the demoted shadow at the boundary (dispatch's bulk stores vs the gather), `PTXV_GROUPS_RUN=3`; test `r4b_groups_catch_b32_at_the_boundary` |
| B33 | dispatch sender fence | race in the dispatch group (bounded) |
| B34 | cached notify skips its barriers | race in the combine group (bounded, `RUN=4`). The cnc group alone is clean: both sides of the race are elsewhere. The deadlock half is not reached within 25k states |
| B35 | dispatch never publishes its tail | deadlock in the dispatch group |
| B36 | notify loses a signal | race in the notify group (25k bound; the deadlock half is not reached) |
| B37 | combine not ordered after the cast (stream) | race in the combine group at the boundary (`RUN=5`) |
| B38 | wrong expert offset | fault in the torch group (`RUN=3`) |
| B39 | notify closing fence | race in the notify group (complete exploration, 16k states) |
| B40 | notify signal scope `.GPU` | race in the notify group (complete exploration, 14k states) |

Every mutant is caught (all 9 still are in the deterministic run, `r4b_moe_mutants`).

So the layer's exhaustive verdict reduces to two DeepEP kernels, dispatch and combine on 2
GPUs. Everything else is decided: certified, or exhaustively explored with the multi-grid
reduction.

### C-RING: certificates for ring-buffer polls (2026-11-04)

C-RING (ASSUMPTIONS.md, E-WELLSYNC) is a certificate rule for polls of monotone counters
whose observed value steers the program: DeepEP's channel tails and heads, and its start and
end offsets.

- *Counters, checked.* A word every strong write of which is a store of one thread, with
  values that never decrease.
- *Classes.* For each poll-loop entry, every candidate write (from the coherence bound on) is
  replayed from the iteration's start. Exiting candidates are grouped by the state they
  leave. Exits must be upward closed (no lost wakeup), and failed iterations must be
  invisible.
- *One run per sequence of choices.* A poll with one class is decided by the run, with the
  least synchronization of its class. A poll with several classes is a choice. The
  certificate runs once per sequence of classes, forcing each choice's read to its class's
  earliest write. Every run must certify, and all must leave the same outputs.

A ring consumer chunks its slots by the tails it sees. Each chunking is one sequence of
choices, so the search runs each and compares (chunking-obliviousness, checked
semantically). The ownership facts are checked by the race detector, pairwise on every
payload byte, in every run:

- each slot is written before the tail release that covers it;
- it is read after an acquire that observed that release;
- it is reused only after the head release that covers it.

A one-run taint check cannot replace the search in DeepEP's receiver
(`intranode::dispatch` sass:0x3cf0 to 0x8010). With implicit flows counted, the observed
tail reaches:

- the slot-to-warp assignment (`R13`, relative to the observed head);
- each warp's TMA mbarrier parity (`R37`, flipped once per slot the warp copies);
- the output offset (`R27`, the sum of the observed batch sizes).

Without implicit flows the check is unsound: a batch counter (fixture e_ring R7) never
depends on the observed value as data, yet changes the result.

| group | verdict with C-RING |
|---|---|
| 1 layout | certified |
| 2 notify | not certified (`barrier_block`'s counters are atomic adds, not C-RING counters); explored exhaustively as before: 24,329 states, clean |
| 3 dispatch | **certified from notify's meet: contracts only (DeepEP's `UBLKCP.G.S` at CTA exit, E-BULK-EXIT), 4 runs.** Two of the four rings have two tail writes (6, then 8), so each receiver's first observation has two classes. The other polls are single class: the senders' head waits on the initial head, start and end offsets, and the second tail. 18 counter polls |
| 4 torch | certified from dispatch's meet (clean: the CUTLASS epilogue contract no longer appears, as at base: `r4b_torch_group_certifies` fails there too) |
| 5 cached notify | not certified (`barrier_block`); explored exhaustively: 3,350 states, clean |
| 6 combine | **not certified** (below) |

Groups 1 to 5 run in 48 s without any `PTXV_GROUPS_RUN` shortcut:

```
PTXV_GROUPS_CERT_ONLY=6 ptxv check fixtures/r4b_moe --variant good \
    --groups "2,3;4,5;6-18,21-25;19,26;20,27"
```

`PTXV_GROUPS_CERT_ONLY=all|k` (debugging) stops at a refused group, any or group k,
instead of exploring it. A certified group whose certificate took several runs continues
from their meet (E-LAUNCH-GROUPS), with every run's obligations.

**Why combine is still refused.** In the strong LDS/STS model (the weak one has the known
relay race), C-RING refuses two loops.

- *The receivers' wait.* Lanes 0 and 1 of each receiver warp poll their own rank's tail
  relay (`LDS R3, [R8]` sass:0x35d0, two words per iteration). C-RING handles one counter
  per exiting iteration.
- *The coordinator's progress loop* (sass:0x23a0 to 0x2f30). Each iteration does three
  things:
  - it reads 23 `warp_retired` flags and up to 23 warp heads;
  - it copies the acquired channel tail into the shared relay (`STS` 0x28c0);
  - it republishes the minimum head (relaxed `STG` 0x2f00, carried `R7 = last_head`).

  Its iterations are visible progress steps, not invisible failed ones, and the values they
  publish depend on every warp's progress. Treating them as choices would mean one run per
  combination of 23 warps' states.

What would decide it is a relay rule: a loop that only republishes monotone functions of
monotone inputs to counters whose every reader is a single-class wait. The rule would give
such a wait the release of the source counter's first write that satisfies it as its
synchronization, a bound valid in every schedule. Termination would rest on the relay
eventually republishing the source's final value. That is open.

**W's small instances.** Agent W's `r4c_nowrap` (3 tokens per ring, chunks of 1, no wrap)
dispatch group certifies in 256 runs, 9 minutes, contracts only. In `r4c_dwrap` the
dispatch ring wraps. There the first C-RING run reports a race: a receiver's
`UBLKCP.S.G` read of a slot against the sender's reuse write `STG.E.128` (sass:0x2780). The
receiver publishes its head with a *relaxed* store (`st_relaxed_sys_global`, no
`MEMBAR`), so under the strict model nothing orders its slot reads before the sender's
overwrite. This is a third T4b-style race, which E-POLL-ACQ does not cover (it acquires on
the reader side). It is latent in r4b: the ring never wraps at 8 tokens per 256 slots. The
fix in DeepEP would be `st.release` for the head.

## Open

- **Exhaustive verdict.** Down to combine (dispatch is certified by C-RING). Options:
  - make the E-BAR-OVER count exact for the remaining two barriers;
  - teach the TMA-completion reduction about `UBLKCP.S.G` (1-D bulk loads completing on
    per-warp mbarriers);
  - run the two groups on a Slurm cluster. That needs `groups.rs`'s meet to gather terminal states
    from every rank: the multi-node explorer keeps them per rank.
- **Certificates for notify and cnc.** Both are explored exhaustively, but neither certifies.
  C-RING refuses `barrier_block`'s warp-poll reads: its counters are written by atomic adds
  from both ranks, and an exiting iteration reads two of them. Extending C-RING to
  commutative-add counters would make both certify without exploration. Their intermediate
  values depend on the order of the adds, so the candidates would be subsets of the
  concurrent adds.
- **Combine** (above): a relay rule for the coordinator's progress loop, and several counter
  words per exiting iteration for the receivers' wait.
- **C-RING cost.** One run per sequence of choices multiplies over independent rings
  (r4c_nowrap: 4^4 = 256). Rings whose choices touch disjoint state could be searched one
  at a time.
- **C-FADD tests.** `fadd_orders_outside_the_outputs_are_refused` checks that order-dependent
  adds outside the outputs are refused. No fixture yet has an order-dependent word that a
  later instruction of the same launch reads; that case takes the same refusal path.
- **4 ranks** (DeepEP's next template) once a fourth GPU is free: `--nproc 4`.
- **Rank skew on hardware.** The latent races (B33, B39, B40) need a delay injected on one rank
  to show on GB300.

# Rung 4c: DeepEP dispatch and combine at the smallest ring (2 GPUs, exhaustive)

Rung 4b left the MoE layer's exhaustive verdict at two DeepEP kernels, dispatch and combine.
Rung 4c shrinks them to the smallest instance that still exercises the channel ring and
explores it exhaustively. These verdicts are the ground truth for the C-RING certificate.

## Instances

`tools/capture_examples/deepep_ring.py`: layout, notify, dispatch, cached notify, combine,
with no torch kernels between them. Sizes:

- 3 tokens of 16 bf16 per rank;
- 2 experts, top-2, so every token goes to both ranks;
- 2 SMs, so 1 channel: sender CTA 0, receiver CTA 1;
- dispatch ring of `DEEPEP_RECV` slots, chunks of `DEEPEP_SEND` tokens;
- combine ring of `DEEPEP_CRECV` slots.

Captured on GPUs 0 and 1 (`tools/capture_dist.py --nproc 2`):

| fixture | ring (dispatch / combine) | chunk | what it exercises |
|---|---|---|---|
| `r4c_nowrap` | 4 / 4 | 1 | no slot reuse |
| `r4c_dwrap` | 2 / 4 | 1 | the dispatch ring wraps: 3 tokens through 2 slots |
| `r4c_chunk2` | 4 / 4 | 2 | 2-token chunks, so two sender warps copy per chunk |

**Combine liveness (a DeepEP constraint, not a bug in the instance).** Combine's ring
cannot wrap at this size. Each of its 23 receiver warps holds the ring head at its previous
token while it waits for its next one. So a combine ring smaller than min(23, tokens per
channel) deadlocks. GB300 agrees: with `DEEPEP_RECV=2` for both kernels, combine's senders and
receivers time out and trap. Hence the separate `DEEPEP_CRECV`.

The launch order differs per capture, so each case's group spec differs (`--groups`):

| fixture | `--groups` (notify; dispatch; cached notify; combine) |
|---|---|
| `r4c_nowrap` | `"2,3;4,5;6,7;8,9"` |
| `r4c_dwrap` | `"2,3;4,5;6,8;7,9"` |
| `r4c_chunk2` | `"1,3;4,5;6,7;8,9"` |

## T4c: ring slot reuse is load buffering under the strict PTX model

The receiver publishes its head with a relaxed store after a `bar.sync` of its warps. The
sender polls the head and then overwrites the slot. In PTX, nothing orders the receiver's
slot reads before the sender's overwrite. A read-write race needs a happens-before edge from
the read, and a relaxed store carries none. That is load buffering. Under the strict model
the wrapping ring races (`r4c_ring_reuse_is_load_buffering_under_the_strict_model`).

Hardware does not reorder a load after a later store that depends on the loaded value's
consumption. The opt-in **E-READ-DONE** (case `read_done`) states this: a read whose value
was consumed is done, and strong stores and barrier arrivals after it carry it. It is used
only for write-after-read checks. It is **not validated across NVLink**; a litmus test is open.

All r4c cases set `poll_acquire`, `smem_cumulative` and `read_done` (rung 4b's opt-ins plus
E-READ-DONE). Note that agent X's litmus tests **refuted E-POLL-ACQ on GB300** (main
`553c5f08` era). The r4c verdicts that rely on it inherit that problem, as r4b's do.

## Reductions added

- **E-SLICE** (`slices.rs`, `--groups ... --slices` or `PTXV_SLICES=1`).
  - What it does:
    - a deterministic run gives each warp's footprint, and a union-find joins warps that
      share a location or a CTA-wide barrier;
    - each slice is explored with the other threads exited;
    - a cross-slice check confirms that no location one slice writes is touched by another;
    - `Meet::compose` combines the slices' meets for the next group.
  - Dispatch splits into its 4 (source, destination) flows: one rank's 12 sender warps for
    a destination, with the other rank's 12 receiver warps for that source.
  - E-BAR-OVER disappears: within a slice each CTA holds only one flow's 384 threads.
  - Argument and status: ASSUMPTIONS.md E-SLICE.
- **E-POLL-ANY** (`exec.rs` `vip_ok`).
  - A strong load whose step ends the same for every value its location can hold commutes
    with that location's writes, up to dead registers.
  - The candidate values are learned over rounds.
  - Acquire loads are included in the first search, with their acquire dropped. A race
    found that way is searched again with relaxed loads only, then without the reduction.
  - Cutting a monotone location's candidates to those at least the current value is
    opt-in (`PTXV_VIP_MONO=1`). It makes nowrap dispatch's flow incomplete at 300k states
    instead of complete at 4,983, even when used only as a fallback: a load that is a poll in
    some states and not in others leaves fewer states equal.
  - Negative tests: `fixtures/e_poll_any`.
- **E-READ-DONE**: see above.
- `Machine::syncs` is now copy-on-write (`machine.rs` `Syncs`). Successor states shared
  deep copies of every sync cell's clocks, which dominated the profile with E-READ-DONE.

Not done: the `UBLKCP.S.G` TMA-completion reduction (dispatch flows complete without it) and
payload abstraction (memory never dominated).

## Verdicts

```
PTXV_SLICES=1 PTXV_GROUPS_UPTO=4 ptxv check fixtures/r4c_nowrap --variant good \
    --groups "2,3;4,5;6,7;8,9" --max-states 300000
```

All exhaustive (complete) unless marked. 144-core box under heavy shared load; times are wall
clock.

| group | `r4c_nowrap` | `r4c_chunk2` | `r4c_dwrap` |
|---|---|---|---|
| 1 layout | certified (192 steps) | certified | certified |
| 2 notify (5 slices) | clean, 28,235 states, 99 terminal, 4 s | clean, 28,235 states, 99 terminal | clean (run with the chain) |
| 3 dispatch (4 flows) | contract (E-BULK-EXIT only), 4 × 4,983 states, 256 terminal combinations, 253 s | contract (E-BULK-EXIT only), 4 × 2,309 states, 16 combinations, 228 s | **open**: see below |
| 4 cached notify (9 slices) | clean, 3,528 states, 33 terminal, 4 s | clean, 3,528 states, 33 terminal | not reached |
| 5 combine (2 slices) | **open** | **open** | **open** |

The dispatch contract is DeepEP's `UBLKCP.G.S` bulk store still reading shared memory at CTA
exit (E-BULK-EXIT, as in rung 4b; GB300 drains it).

**dwrap dispatch.** One flow is about 2M states or more in the plain model.
- With E-POLL-ANY on acquire loads, the receiver's tail wait (`LDG.STRONG` + `CCTL.IVALL`)
  loses its acquire, and the token reads race with the sender's writes. That is a write-read
  race the reduction itself causes.
- With E-POLL-ANY on relaxed loads only, the sender's head poll skips the observation of the
  head store. So it misses the E-READ-DONE edge, and the slot overwrite races with the
  receiver's reads (T4c again).
- Both fall back to the plain model, which does not finish locally.

The next step is a VIP read that joins the location's read entries (`SyncCell::rr`). That is
sound when one thread writes the location: its read entries only grow, so any later value
carries at least the current one. A VIP read with that join must then depend on the
location's writes for the POR, or both orders must join the same entries.

**Combine.** Combine splits into one slice per destination rank (coordinator warps of both
ranks plus that rank's receivers). It is not exhaustive: slice 0 was clean but incomplete at
5M states (2,384 s). The coordinator's scan of the retired flags in shared memory races with
up to 20 idle receiver warps, which is about 2^20 interleavings in the strong shared-memory
pass. Slurm command, **not submitted**. Earlier groups run on every rank (locally); only
the last group is distributed:

```
PTXV_SLICES=1 sbatch --ntasks=16 --cpus-per-task=48 --mem-per-cpu=4G $PTXV_SLURM_SBATCH_ARGS tools/ptxv.sbatch \
    check fixtures/r4c_nowrap --variant good --groups "2,3;4,5;6,7;8,9" --slices \
    --max-states 3000000000
```

## Mutants

Deterministic run (`PTXV_RUN_ONCE=1`); chunk2 was run for B44 only:

| id | what | nowrap | dwrap | chunk2 |
|---|---|---|---|---|
| B33 | dispatch sender release fence NOPed | race | race | |
| B34 | cached notify skips its barriers | race | race | |
| B35 | dispatch tail never published | deadlock | deadlock | |
| B36 | notify loses a barrier signal | race, fault, deadlock | race | |
| B37 | combine not ordered after cached notify (stream) | race | race | |
| B41 | dispatch receiver's `bar.sync` before the head store NOPed | latent (no slot reuse) | race | |
| B42 | dispatch sender's `bar.sync` before the tail store NOPed | race | race | |
| B43 | combine sender release fence NOPed | race | race | |
| B44 | combine sender's `bar.sync` before the tail store NOPed | latent (1-token chunks: only warp 0 copies) | latent | race |

Each fixture lists them with the expected verdicts. Every mutant is caught by at least one
instance.

In group mode, exhaustively within the group (`PTXV_GROUPS_RUN=2 PTXV_GROUPS_UPTO=3`, nowrap
dispatch): B35 deadlocks (2,864 states, complete). B33 races in all 4 flows in the plain
model (`PTXV_NO_VIP=1`).

## Tests

- `r4c_dispatch_splits_into_flows`;
- `r4c_notify_group_by_slices`;
- `r4c_ring_reuse_is_load_buffering_under_the_strict_model`;
- `r4c_ring_mutants`;
- `e_poll_any_keeps_value_sensitive_polls`.

## Open

- dwrap dispatch and every combine (above).
- E-READ-DONE across NVLink: a litmus test.
- E-POLL-ACQ is refuted on GB300 (agent X), and the r4c cases opt into it, as r4b does. The
  warp-level poll of `barrier_block` needs its own fence, or the verdicts must be read as
  "under E-POLL-ACQ".
- `fixtures/e_poll_any` variant `dl` exposed an E-WELLSYNC bug: `--certify` said clean, but
  it deadlocks. Fixed by agent V's C-POLL liveness check (`wellsync.rs`): `--certify` now
  reports the deadlock.
