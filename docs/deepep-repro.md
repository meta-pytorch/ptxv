# Does the GB300 L1 mechanism corrupt existing DeepEP intranode code?

**Status (2026-11-04):** static audit of the installed DeepEP's intranode kernels, a kernel
boundary probe, an opt-in L1 model in ptxv validated against the litmus results, and a
hardware stress campaign on GPUs 0 and 1. Follow-up to docs/poll-acq-litmus.md.

## Bottom line

> **Update (later the same day; see the addendum at the end).** A contrived but legal
> co-resident reader of DeepEP's own buffer, together with NVLink traffic toward the victim
> GPU, does make unmodified DeepEP read stale notify counts: 6 in 171,398 layer calls, one of
> which led to an illegal memory access, and 0 in the controls. The conclusion below holds for
> DeepEP without such a reader.

**(b) Not reproducible in existing code.** We found no way to get stale data out of
unmodified DeepEP intranode code used the way an MoE layer uses it, and we have an
argument for why.

- **The argument.** Only one handoff in the six intranode kernels reads guarded data with
  weak loads after a fenceless poll: `notify_dispatch`'s token counts, read after its second
  `barrier_block`. That barrier starts with `MEMBAR.ALL.SYS; CCTL.IVALL`, which empties the
  reading SM's L1. Between that point and the reads, nothing loads those lines on that SM:
  - not the same CTA (SASS checked);
  - not the kernel's other CTAs;
  - not the other DeepEP kernels, which are serialized on the comm stream;
  - not concurrent compute kernels, which never touch DeepEP's NVLink buffer.
- **Everything else is safe for one of two reasons.**
  - Peer-written channel data is read with volatile loads (`LDG.E.*.STRONG.SYS`, this build
    has `DISABLE_AGGRESSIVE_PTX_INSTRS`) or with bulk copies, and neither goes through L1.
  - The remaining weak reads read data written by an earlier kernel. A new probe shows that
    GB300 invalidates L1 at kernel boundaries: 0 stale in 1.56e10 reads.
- **ptxv's new L1 model agrees.** It reproduces every litmus row, positive and negative. It
  reproduces r1f B20's wrong GB300 output exactly (`{2, 1}`). On the captured 28-launch DeepEP
  layer it finds no stale read:
  - in a deterministic run;
  - in an exhaustive exploration of `notify_dispatch` (25,828 states);
  - in the certified dispatch group.
- **Hardware stress found 0 mismatches.** This was unmodified dispatch → combine with
  every output checked bitwise against an all_gather reference:
  - repro.py: 2,861 iterations, 1.6e7 received rows;
  - bank.py: 245,920 layer repetitions, 1.63e6 dispatch / combine calls, 1.67e9 received
    rows (18 minutes).

  The runs varied:
  - hidden sizes from 128 to 7168, including 2064 and 2880;
  - `num_sms` from 2 to 48;
  - ring sizes from 8 to 256 tokens;
  - fp8;
  - cached and `num_worst_tokens` paths;
  - async / `previous_event`;
  - concurrent GEMM streams with 4096² and 8192² matrices.
- **Ring-slot reuse (second race) is latent by construction.** Every read of a slot has
  completed before the head store can issue. The bytes are in registers that a later
  instruction already consumed, or the mbarrier has reported them in shared memory. No
  config or alternative path changes that in this build.

The PTX-model races stand: DeepEP is correct on GB300 only because of hardware behavior
outside the PTX memory model. The one-instruction fixes in the paste still apply. But we
cannot show stale data from existing code, so by the agreed criterion this is not a report
upstream.

**Side observation, not a memory-model issue:** combine deadlocks if
`num_max_nvl_chunked_recv_tokens` < 23 (below).

## What was audited

- **Version.** DeepEP `1.2.1.2026.6.4.43822+567632d` (wheel built from upstream
  `567632dd`), sm_103a cubin extracted from `deep_ep_cpp.cpython-312-aarch64-linux-gnu.so`
  with `cuobjdump`, all intranode kernels for 2 ranks.
- **Source caveat.** The wheel ships no CUDA sources and GitHub is blocked from this host.
  - The closest source on hand is a vendored DeepEP checkout (upstream `b7fef49`, older).
  - It differs in at least one place we checked. In the installed `notify_dispatch`, each
    thread writes one count (`per_rank_buffer[rank * R + thread]`; one `ST.E` at
    sass:0x0730), so only column `rank` of `rank_prefix_matrix` is meaningful.
  - Every claim below is therefore checked against the installed SASS. The source served
    only to name things.
- **What could not be seen.** Host-side C++ of the installed version (stream handling,
  asserts). We relied on the behavior of the Python API and the old source.

## What a stale read needs

From the litmus work: a weak load (`LD.E`, `LDG.E`, `LDG.E.CONSTANT`) by SM s returns old
bytes iff SM s holds a valid L1 copy of the line that was filled before the writer's bytes
landed, and no `CCTL.IVALL` on SM s came between that fill and the read. Volatile / strong
loads and bulk copies don't use L1. A fill after an acquire that observed the writer's
release is fresh. So per handoff the question is: is there a weak read of peer-written (or
other-SM-written) bytes, and if so, can any load on the reading SM fill that line between
the last invalidation and the writer's write?

## Handoff audit

Weak loads in each kernel (sm_103a, 2 ranks; `cuobjdump -sass` of the installed wheel's kernels, `k_*.sass`). Kernels:

- `notify_dispatch` (ND);
- `cached_notify_dispatch` (CND);
- `dispatch` (D);
- `cached_notify_combine` (CNC);
- `combine` (C);
- `get_dispatch_layout` (L).

| # | handoff | how the reader is ordered after the writer | reader's loads | reachable stale? | why |
|---|---|---|---|---|---|
| 1 | ND: per-rank counts written by every rank into each rank's buffer header, read by CTA 0 after its 2nd barrier (in-place prefix sum, `moe_recv_counter`, copy to `rank_prefix_matrix`) | `barrier_block<false>`: entry `MEMBAR.ALL.SYS; CCTL.IVALL` (0x1000/0x1030), peer `ATOM.ADD.STRONG.SYS` signals, relaxed `LDG.E.STRONG.SYS` poll + `VOTE.ALL` (0x1340..0x1430), `BAR.SYNC` (0x1590) | `LD.E` 0x15f0, 0x1600, 0x1670, 0x16d0, 0x16f0, 0x1900.. | **no** | see "Handoff 1 in detail" below |
| 2 | ND: per-expert counts (same lines, same barrier) | same | `LD.E` 0x16d0/0x16f0 | **no** | as 1 |
| 3 | ND: channel metadata zeroed (memset after the counts) before peers' dispatch writes it | 3rd barrier (WAW) | none (dispatch polls these words with `LDG.E.STRONG.SYS`) | n/a | no weak read |
| 4 | ND → D: `rank_prefix_matrix` in the buffer header (row 0 on rank 1 is peer-written), read by D's receivers (`LD.E` 0x3940) and the `num_worst_tokens` cleanup (`LD.E` 0x80c0) | ND's 2nd barrier, then the kernel boundary | `LD.E` | **no** | GB300 invalidates L1 at kernel boundaries (probe below); within D nothing changes those words. A stale copy from the *previous* call's D on the same SM (same address) does not survive the boundary |
| 5 | CND: `rank_prefix_matrix` copied into the buffer + memset, then D | `barrier_block<true>`, local stores, `barrier_block<false>`, kernel boundary | none in CND; D as in 4 | **no** | as 4 |
| 6 | D: start / end offsets, channel tail → receivers | sender `MEMBAR.ALL.SYS; STG.E.STRONG.SYS`; receiver lane polls `LDG.E.STRONG.SYS` (start / end, no fence) or `LDG.E.STRONG.SYS; CCTL.IVALL` (tail) | x: `UBLKCP.S.G` (bulk, not L1); src_idx / topk_idx / weights / scales: `LDG.E[.64].STRONG.SYS` (0x48f0..0x7e90) | **no** | no weak read of channel data in this build |
| 7 | CNC: buffer zeroing between D and C | `barrier_block<true>` / `<false>` | no reads after the barriers; CTAs ≥ 1 read `send_head` (`LDG.E.CONSTANT`, written by D on this GPU, earlier kernel) | **no** | kernel boundary |
| 8 | C: channel data, tail relay, head | sender release; coordinator `LDG.E.STRONG.SYS; CCTL.IVALL` → `STS` relay → receivers `LDS` poll | x: `LDG.E.128.STRONG.SYS` (66); weights and `send_head`: `LDG.E.STRONG.SYS` (0x4a60, 0x3540); weak loads only of inputs from earlier kernels (`rank_prefix_matrix`, `channel_prefix_matrix`, bias `LDG.E.128.CONSTANT`) | **no** | no weak read of peer data; inputs cross a kernel boundary |
| 9 | L: `get_dispatch_layout` | single kernel, no cross-SM handoff | | n/a | |
| 10 | `intranode::barrier` | barrier only | none | n/a | |

Fenceless polls followed by any read, besides `barrier_block`:

- D's start / end offset polls (`ld.volatile`, no fence). Everything read after them is
  volatile or bulk.
- D's sender head poll. Only stores follow it.

### Handoff 1 in detail

The window is from CTA 0's entry fence (0x1030) to its reads (0x15f0). The guarded lines are
the first lines of the local NVLink buffer: counts at offset 0, then per-expert counts and the
channel metadata that the memset clears. Candidate loads on that SM, and what each one does:

- **Same CTA, other warps.** After the entry fence and `BAR.SYNC`, threads ≥ R skip the
  atomics and leave the poll on their first vote (value 0) to wait at 0x1590.
  - The only loads in the window are `LDG.E.64` of the pointer arrays (0x10a0, 0x1110,
    0x1340: `barrier_signal_ptrs[...]`). Those live at `buffer + num_nvl_bytes + 32`, which
    is a different line.
    - That line also holds the barrier signal words, and 0x1340 reloads it weakly on every
      poll iteration. The signals are read only with volatile loads, so this is harmless.
- **Same CTA, earlier phase.** Between barrier 1 (`<true>`, no fence) and barrier 2, CTA 0
  only *stores* to the header (0x0730 and the per-expert loop). It reads inputs
  (`num_tokens_per_rank`, `num_tokens_per_expert`), not the buffer. Any earlier copy is
  dropped by the entry fence.
- **Same thread re-reading.** The prefix sum reads `local[t]`, then `local[R + t]`. Both reads
  come after the poll, so a line filled by the first is filled after the peers' bytes have
  landed.
- **Retries.** The poll loop contains no weak payload load.
- **Other CTAs of `notify_dispatch`.** CTAs 1..R are 128 threads and may share the SM. They
  read `is_token_in_rank` and `channel_prefix_matrix` (torch tensors), never the buffer.
- **Other DeepEP kernels.**
  - All intranode kernels go to the Buffer's comm stream and are serialized. The previous
    kernel's lines don't survive the boundary.
  - Low-latency kernels use the RDMA buffer, not this one.
  - A second `Buffer` has its own allocation.
- **Concurrent compute kernels** (torch overlap, any `num_sms`). They run on other SMs or
  co-reside, but they only touch torch tensors. The NVLink buffer is a private `cudaMalloc`,
  outside the caching allocator. The only way to expose it is
  `Buffer.get_local_buffer_tensor`, which no intranode path uses. A user kernel reading that
  view *during* `notify_dispatch` would recreate the `*-pr0-payld` litmus, but that is a
  modification, not existing usage.
- **Consecutive calls, `previous_event`, async, configs.** These change stream ordering and
  grid sizes only. `notify_dispatch` always launches 1 + R CTAs of 128 threads. None of these
  adds a load of the header lines on CTA 0's SM inside the window.

So the protecting condition from the paste ("nothing loads the guarded lines between the
entry fence and the reads") holds for every path through the existing code we could find.
It would fail only if code were added: a prefetch, a user kernel over the buffer view, or a
DeepEP change that reads the header weakly in another CTA during `notify_dispatch`.

## Kernel boundaries (probes/l1_kbound)

`kbound.cu` runs per iteration, on GPUs 0 and 1 over NVLink, in both directions:

1. Kernel K1 has every CTA weak-load the same 64 lines. It then signals with a relaxed atomic
   that depends on the loaded values.
2. A persistent writer kernel on the peer GPU waits for every CTA, writes the lines, does
   `fence.acq_rel.sys` and sets a flag.
3. Kernel K2 polls the flag with `ld.volatile` and no fence, then weak-reads the lines.

| mode | between K1's fills and K2's reads | runs | reads | stale |
|---|---|---:|---:|---:|
| 1 | no boundary (one kernel does both; positive control) | 4 | 6,225,920,000 | 6,225,920,000 |
| 0 | K1, K2 consecutive on one stream | 4 | 6,225,920,000 | 0 |
| 3 | K2 on a second stream after an event on K1 (DeepEP's comm-stream pattern) | 3 | 4,669,440,000 | 0 |
| 4 | an empty kernel between K1 and K2 | 3 | 4,669,440,000 | 0 |
| 2 | mode 0 + `fence.acq_rel.sys` after the poll (negative control) | 3 | 4,669,440,000 | 0 |

- **The write landed while K2 was already running.** In 99.6–99.8% of K2's CTAs the first
  poll still saw the old flag, so the boundary itself is what invalidated the lines, not a
  late fill.
- **Load forms.** `LD.E`, `LDG.E` and `LDG.E.CONSTANT` all behave the same.
- **Not covered.** An intra-GPU writer, where the persistent writer and K1 / K2 share one GPU,
  hung on GPU 3 and was not pursued. The same-GPU case is CUDA's ordinary stream-order
  guarantee.
- **Conclusion: L1 does not survive a kernel boundary on GB300.**

Raw lines: `probes/l1_kbound/results.txt`.

## The L1 model in ptxv (E-HW-L1, opt-in)

`crates/sim/src/l1.rs`; case `hw_l1 = true` or `PTXV_HW_L1=1`; off by default. ASSUMPTIONS.md
E-HW-L1 has the full rule.

**State and rules.**

- Each SM (= CTA, unless `PTXV_L1_COLOCATE` groups CTAs) has 128-byte lines.
- A weak global load hits a line or fills it.
- Lines are dropped by:
  - `CCTL.IVALL`;
  - PTX `fence.acq_rel` / `fence.sc` at gpu / sys scope;
  - acquire loads at gpu / sys scope;
  - kernel boundaries (a line from another grid never hits).
- Strong loads neither hit nor fill.
- The SM's own stores update its copy; its atomics drop the line.

**Two checks, both kind `stale`.**

- **Operational** (`stale-l1:<pc>`): in the explored schedule a load hit a line whose bytes
  differ from memory. The load returns the stale bytes, as the hardware would.
- **Predictive** (`stale-l1-pred:<R>:<F>`). The last write W of the bytes R reads is by
  another SM. Some weak load F of that line on R's SM is not ordered after W. No
  invalidation on R's SM is ordered after F or after W and before R.

  Like a happens-before race check, this covers every schedule with the run's happens-before
  order. It matters because ptxv's stateful explorer does not schedule weak loads (E-DRF).
  DPOR does, and it finds the operational witnesses as well (table).

**Hardware-faithful mode** = `hw_l1` + E-POLL-ACQ.

- E-POLL-ACQ is refuted as an *acquire*. What the model still takes from it is only the
  *ordering*: after the poll exits, an L1 miss returns the new data. The litmus results
  support that: a fill after the poll is never stale, and neither is a read whose line was
  invalidated by the entry fence.
- E-HW-L1 then adds the non-coherence.
- Without E-POLL-ACQ (strict model), the predictive check also flags `notify_dispatch`'s
  second read of a line (0x1600 after 0x15f0). Nothing in the PTX model orders the peer's
  write before the first read, which is the T4b race again.

### Validation against the litmus results

| fixture / variant | shape | GB300 (docs/poll-acq-litmus.md, docs/rung1f.md) | ptxv `hw_l1`, both front-ends | DPOR operational hit |
|---|---|---|---|---|
| l1_litmus pr0 | barrier_block: entry fence, no load in the window | 0 stale in 1.4e12 (`bb-f0`) | race (T4b), **no stale** | no |
| l1_litmus pr0-payld | entry fence + another warp loads the payload during the barrier | 0.14% stale | race, **stale** | yes |
| l1_litmus pr1 | no fence, no earlier load (line not in L1) | not stale when the line is not valid | race, no stale | no |
| l1_litmus pr2 | weak load just before the poll | 76% stale | race, **stale** | yes |
| l1_litmus pr3 | weak load in every poll iteration | 80% stale | race, **stale** | yes |
| l1_litmus neg | load before the poll + `fence.acq_rel.sys` after it | 0 stale (`ctl-neg-poll-rd3`) | race (the early load), no stale | no |
| l1_litmus neg-payld | pr0-payld + acquire fence after the poll (the proposed DeepEP fix) | — | race, no stale | no |
| r1f_pingpong B20 | relaxed poll, no acquire | wrong 1000/1000, out = {2, 1} | **stale**, operational hit; run ends with out = {2, 1} | yes |
| r1f_pingpong good / fenced / B19 / B56 | acquire, or fence after the poll | correct / latent | no stale | no |

The relay shapes (E-SMEM-CUMUL) are not encoded as fixtures. DeepEP's relays (dispatch tail,
combine tail) are followed only by volatile reads, so L1 does not enter.

Tests: `l1_litmus`, `l1_model_r1f_pingpong`, `r4b_moe_l1_model` (crates/sim/tests/fixtures.rs).

### DeepEP in the model (fixtures/r4b_moe, 2 ranks, 28 launches)

| run | result |
|---|---|
| `PTXV_RUN_ONCE=1 PTXV_HW_L1=1 ptxv check fixtures/r4b_moe --variant good` (E-POLL-ACQ + E-SMEM-CUMUL + E-HW-L1) | terminal; findings: only the 2 E-BULK-EXIT contracts; **no stale**; `y_r0`, `y_r1` bitwise equal to the GPU (33 s) |
| same with `PTXV_POLL_ACQUIRE=0 PTXV_SMEM_CUMULATIVE=0` (strict) | the T4b races, plus predictive stale on `notify_dispatch` 0x1600 / 0x16f0. These have no hb from the peer's write to the first read, which is the PTX-model artifact explained above |
| `PTXV_HW_L1=1 PTXV_GROUPS_CERT_ONLY=6 ptxv check fixtures/r4b_moe --variant good --groups "2,3;4,5;6-18,21-25;19,26;20,27"` | layout certified clean; **notify exhaustive: 25,828 states, 108 terminal, clean**; dispatch certified (contracts only, 4 C-RING runs); torch certified clean; cached_notify_combine exhaustive: 3,542 states, clean; combine not decided (as before) |
| combine group, bounded: `PTXV_HW_L1=1 PTXV_GROUPS_BOUNDED=1 PTXV_POR_SEED_WALKS=0 PTXV_GROUPS_RUN=5 ... --max-states 5000` | incomplete as before. 5,001 states, 34 terminal, one output; **no stale**. The verdict is contracts only: the weak LDS/STS pass's relay race is cleared by the strong pass, as in rung4.md. Took 33 min: the L1 state's copy-on-write (whole map per weak load) makes exploration of load-heavy kernels slow, so a 25k-state run was stopped |

No witness trace exists: the model found no schedule of the existing kernels in which a
DeepEP weak read returns stale data. Its positive results on the litmus shapes show it would
have flagged the `payld` situation had DeepEP produced it.

## Two-level verdicts and the retirement of E-POLL-ACQ (2026-11-04)

`ptxv check` now classifies every PTX-model race / stale read against the GB300 hardware
model (crates/sim/src/twolevel.rs, ASSUMPTIONS.md E-HW-L1 "Two-level verdicts"): the L1 above
over sequentially consistent memory (`hw_order`, the ordering half of E-POLL-ACQ, without its
acquire half; this replaces "hw_l1 + E-POLL-ACQ" above). A refuted assumption needs an explicit
opt-in (`assume_refuted`, `--assume-refuted`), and fixtures/r4b_moe no longer sets
`poll_acquire`, so its verdicts are the strict PTX model's.

**Closed world, and why "latent" needed a qualifier.** The hardware model is closed-world
(E-L1-CLOSED): only the case's own loads fill L1 lines. Under that assumption the
notify_dispatch reads below are not reachable, which is what this section first reported
("all 7 latent"). But agent Y reproduced exactly those stale reads on GB300 with unmodified
DeepEP plus a legal co-resident reader kernel (docs/deepep-coreader.md), and agent Z's havoc
mode (E-L1-HAVOC) flags exactly those reads. Level 2 therefore also runs the havoc check, as a
probe inside the same hardware-model run (predictive, no extra exploration), and splits latent
into **latent** (no fill could expose it, even from outside the case) and
**latent-closed-world** (latent only under E-L1-CLOSED; observable when a co-resident kernel
fills the line; the witness is `stale-l1-havoc:<read pc>` and the reason names E-L1-CLOSED).

**notify_dispatch group, exhaustive** (`ptxv check fixtures/r4b_moe --variant good --groups
"2,3;4,5;6-18,21-25;19,26;20,27"`; the group check stops at the first group with findings):
strict model 31,103 states, complete, 7 races; hardware model 20,800 states, complete, clean
in a closed world, one final memory. 4 are latent-closed-world, 3 latent:

| race (both ranks) | GB300 classification |
|---|---|
| `ST.E` 0x0730 (peer's count) → `LD.E` 0x15f0 / 0x1600 | **latent-closed-world**: ordered by the barrier's signal under GB300's ordering, and in a closed world every fill of the header line on the reading SM between `CCTL.IVALL` 0x1030 and the read comes after the write landed (276 weak loads, none stale); but a fill by anything else on that SM in that window leaves a copy the read hits (`stale-l1-havoc:<0x15f0 / 0x1600>`), which is what the co-reader repro does |
| `ST.E` 0x0f80 (per-expert count) → `LD.E` 0x16d0 / 0x16f0 | **latent-closed-world**: same line, same reason |
| `ST.E` 0x0730 / 0x0f80 vs `ST.E` 0x1630 / 0x1d50 (write-write) | latent: ordered under GB300's ordering; L1 is write-through |

This is the "Handoff 1 in detail" argument, now produced by the tool, with its limit: it
holds only while no other kernel touches the header line on the reading SM.

**Whole layer, one deterministic run of each model** (`PTXV_RUN_ONCE=1`): 17 races, 15
latent and 2 latent-closed-world. The latter are notify's 0x0730 → 0x1600 and 0x0f80 → 0x16f0
(this run orders rank 0's reads after its peer's writes; rank 1's direction is the exhaustive
group's). Latent: notify's write-write races above; notify's memset (0x1d50) vs dispatch's channel metadata (strong stores
and `LDG.E.STRONG.SYS` polls: no L1); dispatch's ring (strong loads, bulk copies) vs combine's
writes into the reused buffer (`STG.E.128` 0x1b50), and cached_notify_combine's zeroing vs
combine (strong loads). No stale read in the hardware model; outputs equal the GPU's. The
dispatch group could not be explored under the strict model within 40 GB (killed after 23
min; with E-POLL-ACQ it certified); cached_notify_combine is clean in both models (3,430 /
3,298 states); combine stays undecided (bounded, 501 states).

Mutants (one run each): B33, B39 and B40 (GB300: correct, latent) have only latent races plus
the same 2 latent-closed-world notify reads as the good variant; B32, B34 and B37 have
observable ones (ordering removed: either order reachable), B32 and B37 also the 2
latent-closed-world reads, B32 also 4 stale reads only the hardware model reports; the
deadlocking mutants' hardware-model runs do not terminate, so their races are undecided.

## Hardware repro attempts (GPUs 0 and 1)

Both scripts run unmodified DeepEP the way an MoE layer does:

1. `get_dispatch_layout`;
2. dispatch, with top-k ids and weights;
3. combine of `recv_x * (rank + 1)`, with the received weights;
4. optionally the cached-handle dispatch / combine and the `num_worst_tokens` dispatch.

Every output is compared bitwise with a reference built from all_gather'ed inputs:

- the notify counts (`num_recv_tokens_per_expert_list`, the received count, column `rank` of
  `rank_prefix_matrix`);
- every received row, its source index, top-k ids and weights;
- every combined token and the combined weights.

Token values are small integers, so all bf16 sums are exact.

Routing changes on every call, so consecutive calls change counts and buffer contents.
Concurrent work is realistic only: bf16 GEMMs on a side stream that overlap DeepEP's comm
stream, plus DeepEP's async / `previous_event` mode. Nothing loads DeepEP's buffers
artificially.

- **`tools/deepep_repro/repro.py`.** Per-iteration CPU reference. 7 configurations ×
  90 s: **2,861 iterations, 16,103,000 received rows, 0 mismatches**
  (`tools/deepep_repro/results1.txt`).
- **`tools/deepep_repro/bank.py`.** A bank of 24–40 precomputed cases replayed in a shared
  random order, compared on the GPU.

| options (`bank.py --secs 120 ...`) | layer reps (per rank) | dispatch + combine calls (both ranks) | received rows (both ranks) | mismatches |
|---|---:|---:|---:|---:|
| `--bank 24 --hidden 7168 --tokens 1-4096 --num-sms 20 --cached --worst` | 36,016 | 360,160 | 299,528,256 | 0 |
| `--bank 40 --hidden 2048,2880,4096,5120,7168 --tokens 1-2048 --num-sms 2,4,8,24,48 --cached` | 41,024 | 328,192 | 156,913,701 | 0 |
| `--bank 24 --hidden 7168,4096 --tokens 1-4096 --num-sms 20 --overlap 4 --async --cached` | 22,176 | 177,408 | 184,865,282 | 0 |
| `--bank 24 --hidden 7168 --tokens 256-8192 --num-sms 20,24 --overlap 8 --gemm 8192 --async` | 6,672 | 26,688 | 113,925,580 | 0 |
| `--bank 24 --hidden 2064,2880 --tokens 1-2048 --num-sms 2,8,20 --dcfg 6,30 --ccfg 4,30 --worst` | 22,256 | 133,536 | 92,857,372 | 0 |
| `--bank 24 --hidden 128,256,2064 --tokens 1-512 --num-sms 2,4 --dcfg 1,8 --ccfg 1,24 --experts 8 --topk 2 --cached --worst` | 18,720 | 187,200 | 14,864,450 | 0 |
| `--bank 24 --hidden 7168 --tokens 1-4096 --num-sms 20 --no-sync --async --overlap 4` | 62,272 | 124,544 | 517,375,491 | 0 |
| `--bank 32 --hidden 7168,4096 --tokens 1-64 --num-sms 20 --cached --overlap 4` | 19,664 | 157,312 | 2,502,242 | 0 |
| `--bank 24 --hidden 7168 --tokens 1-8192 --num-sms 24 --experts 256 --topk 8 --cached --overlap 2` | 17,120 | 136,960 | 284,411,568 | 0 |
| **total (9 runs, 18 min)** | **245,920** | **1,632,000** | **1,667,243,942** | **0** |

How to read the table:

- Each layer rep runs `notify_dispatch` at least once: twice with `--worst`; `--cached` adds
  `cached_notify_dispatch` and `cached_notify_combine`.
- `--overlap N` launches N bf16 GEMMs of `--gemm`² on a side stream before every DeepEP call.
  They run concurrently with DeepEP's kernels on the remaining SMs.
- `--dcfg` / `--ccfg` are (send chunk, ring tokens) for dispatch / combine.
- `--no-sync` is the CUDA-graph-style `num_worst_tokens` path, with no CPU synchronization.

Commands: `tools/deepep_repro/sweep.sh <plan> <out> <secs>` (`SCRIPT=bank.py` for
bank.py), plans `plan1.txt` / `plan2.txt`; single runs are
`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 tools/deepep_repro/bank.py --secs 120 <options>`.

## Ring-slot reuse (second race)

- **Shape.** The dispatch receiver copies a slot to shared memory with `UBLKCP.S.G` and waits
  on the mbarrier (`SYNCS.PHASECHK.TRANS64.TRYWAIT`, 0x4270 / 0x44d0). It reads src_idx,
  top-k and scales with `LDG.*.STRONG.SYS` whose values feed `STG`s to the outputs. Then it
  does `BAR.SYNC` (0x7f40), and one lane publishes the head with a relaxed
  `STG.E.STRONG.SYS` (0x7fc0).
- **Why it is latent.** Before any warp reaches that `BAR.SYNC`:
  - each of its slot loads has returned, because its value has been consumed by a store
    (scoreboard wait);
  - each bulk copy has landed, because the mbarrier completes only after the bytes are in
    shared memory (ringwar control in docs/poll-acq-litmus.md).

  So when the head store issues, no read of the slot is outstanding. A later overwrite cannot
  change data already in registers or shared memory. This does not depend on slot size,
  chunk size, hidden size, `num_sms` or rank count.
- **Combine's ring is latent for the same reason.** Receivers load with
  `LDG.E.128.STRONG.SYS` and consume every value (reduction → `STS` to the TMA staging
  buffer; weights → `STG`) before they update their head in shared memory. The coordinator
  publishes the minimum of those heads.
- **The non-bulk path** (`UNROLLED_WARP_COPY` with `ld_nc_global`) exists only in
  `DISABLE_SM90_FEATURES` builds. It is not in this sm_100a/103a wheel, and it would also
  consume every loaded value before the barrier.
- **What would break it.** A code change that publishes the head before a slot load's value
  is consumed. Or hardware that completes an mbarrier transaction before the copy has read
  its source, which the ringwar harness would detect (0.15–0.83% overwrites in its control).
  Under the PTX model it remains a race. The fix is still `st.release.sys` for the head.
- **Repro coverage.** Runs with 8- to 30-token rings, where the ring wraps constantly, showed
  no mismatch either.

## Side observation: combine deadlocks with fewer than 23 ring slots

Not a memory-model issue, but it is existing code with a config that the host asserts
accept (`0 < send < recv`):

- **Symptom.** With `num_max_nvl_chunked_recv_tokens` ≤ 22 in the *combine* config, combine
  times out ("DeepEP timeout for combine receivers / senders") and traps. The process dies
  with `unspecified launch failure`.
- **Threshold.**
  - 8, 16 and 22 hang;
  - 23, 24 and 32 work (`--ccfg 1,N`, hidden 128, 8 experts, top-2, `num_sms 2`;
    `tools/deepep_repro/combine_small_buf.txt`).
- **Reading.** 23 is the number of combine receiver warps (768 / 32 − 1). Each warp waits for
  its own token's slot, and the coordinator publishes the minimum of the warps' heads. When
  a warp's token lies beyond the ring, the minimum cannot advance.
- **Reproduce:**

  ```
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 tools/deepep_repro/repro.py --secs 15 \
      --hidden 128 --tokens 1-512 --num-sms 2 --ccfg 1,16 --experts 8 --topk 2
  ```

- **Practical impact.** DeepEP's default configs use 256 slots, so normal use is not
  affected. Dispatch with an 8-slot ring works.

## What would change the conclusion

| claim | basis | how it could fail |
|---|---|---|
| nothing loads ND's header lines on CTA 0's SM in the barrier window | SASS of the installed module; Python / stream structure | a different DeepEP build (source differs from what we read), user code over `get_local_buffer_tensor`, a future prefetch |
| kernel boundaries invalidate L1 | probe, 1.56e10 reads, two GPUs, three launch patterns | another driver / chip; MPS or green contexts were not tested |
| channel data never goes through L1 | `DISABLE_AGGRESSIVE_PTX_INSTRS` build: `ld_nc_global` = volatile | a build without that flag makes D's src_idx / top-k / scale reads `LDG.E.CONSTANT`. Those are then E-SMEM-CUMUL-shaped (acquire by lane 0, then `bar.sync`, then the read), which the litmus validated (`relay-f0-rd4`: 0 stale), not the barrier shape |
| L1 model | litmus rows + B20 exact output; predictive check is pairwise with capped logs (E-HW-L1) | rare under-reports; the hardware campaign is the independent check |

**Not examined:**

- internode / low-latency (NVSHMEM) kernels;
- 4 or 8 ranks (only 3 GPUs were free);
- other GPUs;
- CUDA graphs. Graph replay changes launch mechanics but not the kernel boundaries.

## Files

- `probes/l1_kbound/` (kbound.cu, run.sh, results.txt);
- `crates/sim/src/l1.rs`, hooks in `crates/sim/src/exec.rs`, the `hw_l1` case flag, E-HW-L1
  in ASSUMPTIONS.md / `assumptions.rs`;
- `fixtures/l1_litmus/`; tests in `crates/sim/tests/fixtures.rs`;
- `tools/deepep_repro/` (repro.py, bank.py, sweep.sh, plans, results).

# Addendum (2026-11-04, later): a legal co-resident reader does corrupt unmodified DeepEP

The user asked for a contrived repro in which every added action is allowed by the PTX memory
model and the CUDA programming model. The new parts are a co-resident reader kernel that
leaves an L1 line which DeepEP's own non-acquire poll-then-read then hits, plus ordinary
co-running traffic. This changes the bottom line above for that setting. **Unmodified DeepEP
returned wrong token counts (6 times in 171,398 layer calls), and in one case dispatch then
made an illegal memory access.** The same runs with the reader pointed at a private tensor, or
with no reader, never failed (0 in 157,840 and 0 in 94,896 calls). Without the co-reader, the
conclusion (b) above still stands.

## The reader action and why it is legal

- **What it reads.** One 32-bit word of DeepEP's local NVLink buffer, read through the public
  `Buffer.get_local_buffer_tensor(torch.int32, ...)`:
  - int index `1 - rank`, i.e. byte 4 on rank 0 and byte 0 on rank 1;
  - this is `rank_prefix_matrix[0][1 - rank]`, a word the installed build never uses.
- **How.** A separate kernel on its own stream: 4 × SMs CTAs, lane 0 of each loops on a weak
  `ld.global.u32` (`LDG.E`) for ~1 ms. The load reaches L1; see the ptxas note below for how
  the loop keeps it.
- **No kernel writes that word while the reader runs.** Checked on the installed sm_103a SASS:
  - `get_dispatch_layout` does not touch the buffer.
  - In `notify_dispatch`, rank r's buffer receives `per_rank_buffer[p * 2 + r]` from each rank p
    (`ST.E` 0x0730), i.e. words r and 2 + r. The in-place prefix sum writes words 2 and 3
    (0x1630). The per-expert rows and the memset start at byte 16. So word 1 − r is never
    written; it is only read (0x15f0, 0x1900).
  - `dispatch` addresses all its buffer state from `buffer + 16` (channel offsets and rings).
  - The only kernels that write bytes 0..15 are `cached_notify_combine` / `combine` (memset,
    queue heads) and `cached_notify_dispatch`.
- **Ordering.** The reader stream waits on the compute stream at the start of each layer
  call, so it runs after the previous combine. The compute stream waits on the reader stream
  before the next combine. The reader therefore never overlaps a writer of its word: no data
  race in the PTX model (conflicts are per byte) and plain stream-ordered CUDA. It shares the
  128-byte line, and the 32-byte sector, with words the peer does write: `per_rank_buffer`
  row p on rank r, and the start of rank 0's per-expert row on rank 1.
- **Co-running traffic.** Rank 1 runs a kernel that stores into a scratch buffer of GPU 0,
  exported with `cudaIpcGetMemHandle`. This is NVLink traffic 1 → 0, and the buffer is not
  DeepEP's memory.
- **Launch timing.** The reader is launched from a host thread 0–100 µs (`--reader-delay-us`)
  after the main thread enters `buf.dispatch()`, so it typically starts after
  `notify_dispatch` has started.
- **Nothing in DeepEP is modified, delayed or instrumented.**

## Why this specific setup (mimic and model)

`probes/notify_coreader/mimic.cu` re-implements `notify_dispatch`'s count handoff with its
exact synchronization on GPUs 0/1 and adds co-readers whose placement is verified with
`%smid`. Stale counts per 20,000 iterations on the vulnerable rank:

| co-reader on the victim CTA's SM | peer store natural | peer store 2 µs late (control) | NVLink push 1 → 0 (legal) |
|---|---:|---:|---:|
| none | 0 | 0 | — |
| warps of the same CTA | 0 | 20,000 | 20,000 |
| other CTAs of the same kernel | 0 | 20,000 | 20,000 |
| another kernel, launched before the victim | 0 | 0 (of 200,000) | 15 (of 200,000) |
| another kernel, launched after the victim | 0 | 0 (of 200,000) | 41,114 (of 200,000) |

- **Timing.** At DeepEP's natural timing, the peer's store lands before the victim's entry
  fence (`MEMBAR.SYS; CCTL.IVALL`) completes. The window opens only if the store is at least
  ~0.75 µs later (`wdelay` sweep: 0 stale at 500 ns, 227 at 750 ns, 18,716 at 1 µs).
  Congestion on the link toward the victim provides that delay legally.
- **L1 sharing between kernels.** Concurrent kernels on one SM do share L1 lines, but how
  often it shows depends on launch order and traffic. We do not know the mechanism.
- **Sector granularity.** Fills are per 32-byte sector: in the rank-1 failure only the words
  in the reader's sector were stale.
- **A compiler gotcha.** ptxas hoists a weak load with a loop-invariant address out of a spin
  loop, even when it comes from `asm volatile`. Our first reader (and the mimic's) loaded once
  and saw nothing. The readers now index with `n & zmask`, where `zmask = 0` is passed at run
  time. Any probe that spins on a weak load needs the same care.

**The model.**

- `fixtures/l1_litmus` variant `pr0-lineshare` is DeepEP's barrier shape with another warp of
  the SM reading a *different, never-written* word of the payload line. It reports the T4b
  race only on the victim's read, nothing on the co-reader, and a predictive `stale`, on both
  front-ends (test `l1_litmus`).
- My E-HW-L1 rule "lines from another grid never hit" is right for consecutive kernels but
  under-reports concurrent ones (ASSUMPTIONS.md, "Known gap"). Agent Z's `l1_coresident` /
  `l1_havoc` model, with fixtures `l1_coreader` and `r4b_coreader`, covers that and finds the
  same reads vulnerable (`docs/deepep-coreader.md`).

## Hardware runs (unmodified DeepEP, GPUs 0 and 1)

Command for one run. It stops both ranks at the first notify / dispatch mismatch, because a
wrong count makes the following combine hang:

```
cd "$PTXV_ROOT"   # the ptxv checkout
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 tools/deepep_repro/coreader.py --secs 120 \
    --bank 16 --hidden 7168,2048 --tokens 1-4096 --num-sms 20 --push-from 1 --reader-delay-us 100
```

- Build the reader once with
  `nvcc -O3 -arch=sm_103a -shared -Xcompiler -fPIC -o libreader.so reader.cu` in
  `tools/deepep_repro/coreader/`.
- Plans and results: `tools/deepep_repro/plan_coreader{,2,3}.txt` →
  `results_coreader{,2,3}.txt`.

| setting (all: unmodified DeepEP, 2 ranks, `num_sms` 20, hidden 7168/2048, push 1 → 0) | layer calls | mismatches |
|---|---:|---:|
| co-reader on the DeepEP word, launched 0–30 / 100 / 300 / 400 µs into `dispatch()` | 171,398 | **6** |
| same, reader on a private tensor (`--reader-target other`): same load, timing and traffic | 157,840 | 0 |
| push only, no reader | 94,896 | 0 |
| co-reader launched *before* `dispatch()` (with push / skew / hog / p2p variants) | 372,096 | 0 |

All six mismatches:

| run | rank | what was wrong | then |
|---|---|---|---|
| exploratory (delay 100) | 0 | received count 2029, expected 2875 | combine hang (timeout trap) |
| plan2 #1 (delay 100) | 0 | received count 2079, expected 2225 | stopped |
| plan2 #5 (delay 100) | 1 | `num_recv_tokens_per_expert_list` entries 1..3 = 347 / 342 / 334, expected 486 / 505 / 495. Every wrong word lies in the reader's 32-byte sector (bytes 0..31); entry 0 was stored before the fill; entries ≥ 4, in other sectors, were right | stopped |
| plan3 (delay 300) | 0 | received count 4329, expected 5513 | **illegal memory access in DeepEP** (dispatch with the wrong offsets) |
| plan3 (delay 30) | 0 | received count 3734, expected 6847 | stopped |
| plan3 (delay 300) | 0 | received count 2141, expected 2691 | stopped |

- **Rate.** 6 events in 171,398 calls is 3.5e-5 per layer call (Poisson 95% interval
  1.3e-5 to 7.6e-5).
- **Comparison with the controls.** If the private-tensor controls failed at the same rate,
  the chance of seeing 0 failures in them is about 0.4%. Against all controls (reader off
  included), it is about 1.5e-4.
- **What the rank-0 values mean.** They are the old content of word 2 (a combine queue head
  from the previous call) plus rank 0's own count. That is the stale `per_rank_buffer[1][0]`
  read at sass 0x1600.

## What this means

- **Minimal ingredients.** Unmodified DeepEP intranode, plus:
  1. a legal reader of a never-written word in line 0 of the local NVLink buffer, running
     concurrently with `notify_dispatch` on the same SMs;
  2. NVLink traffic toward the victim GPU, which delays the peer's count store past the entry
     fence.

  With these, `notify_dispatch` reads a stale count. Wrong counts propagate: wrong received
  sizes, wrong per-expert counts, and, once, an out-of-bounds write.
- **What makes it contrived.** Ingredient 1 is a reader of DeepEP's *internal* buffer. The API
  exposes that buffer, but no real workload reads it during dispatch.
- **Legal but untested alternatives.** Without ingredient 1 we found no realistic action that
  fills those lines. A reader of a different allocation that shares the line is impossible,
  because allocations are line-aligned.
- **The fix is the same one instruction** (`fence.acq_rel.sys`, or acquire polls, after
  `barrier_block`'s vote). With that fix the mimic, the litmus and the model are all clean.
  It has not been tested on DeepEP itself, which we did not modify.
