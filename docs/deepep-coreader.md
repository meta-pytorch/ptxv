# A legal co-resident reader and DeepEP's notify_dispatch, in the L1 model

**Status (2026-11-04):** model only (agent Z). The hardware attempt belongs to agent Y.
Follow-up to docs/deepep-repro.md. That document found no stale read in unmodified DeepEP and
gave the reason: nothing on the reading SM loads the guarded lines between
`barrier_block`'s entry fence and the reads. This one asks whether a co-resident kernel can do
that load and still be legal.

## Question

DeepEP is racy in the PTX model (docs/rung4.md, T4b). On GB300 a stale read needs a stale copy
of the line in the reader SM's L1 (E-HW-L1). The question is whether a co-resident kernel whose
every action is allowed can leave such a copy. Allowed means: no data race in the PTX memory
model, and ordinary CUDA (another stream, reading DeepEP's buffer through the public
`Buffer.get_local_buffer_tensor`). The idea is to read *other bytes of the same 128-byte line*.
Conflicts in the PTX model are per byte, so reading a byte that nobody writes races with
nothing. The fill, however, brings the whole line.

## What was added to ptxv

| | case option | env | assumption |
|---|---|---|---|
| closed world (the old implicit rule, now named and reported) | default | | E-L1-CLOSED |
| one CTA per SM (the old implicit rule, now named and reported) | default | `PTXV_L1_COLOCATE` | E-L1-PLACE |
| co-resident launches share SM L1s; placement explored | `l1_coresident = true` | `PTXV_L1_CORES=1`, `PTXV_L1_PLACE=pairs\|all\|none` | E-L1-CORES |
| any outside agent may fill any line on any SM at any time | `l1_havoc = true` | `PTXV_L1_HAVOC=1` | E-L1-HAVOC |

- Both options imply `hw_l1`, and both are off by default, so existing verdicts don't change.
  `hw_l1` verdicts now list E-L1-CLOSED and E-L1-PLACE (or the options that replace them).
- Code: `crates/sim/src/l1_cores.rs`, plus small hooks in `l1.rs` (hit test, predictive check,
  footprints), `exec.rs` `launch_cta`, and `ptxv check` (`explore_placements` wraps the
  explorer, DPOR and `--groups`).
- The PTX front-end now parses `vote.sync.all.pred`.

ASSUMPTIONS.md has the rules and their soundness arguments. In short:

- **Co-residency.**
  - The model:
    - Launches that are not stream-ordered may share SMs, within 32 CTAs, 2048 threads and
      228 KB of shared memory per SM.
    - A line filled by one grid hits for a co-resident grid.
    - A CTA's launch drops the lines other grids left on its SM.
  - `pairs` placements decide whether some placement reaches a stale read.
  - DPOR gets L1 footprints (spaces 20/21), so it orders a co-resident fill between an
    invalidation and a remote write and produces operational witnesses.
- **Havoc.** It is predictive and does not model the scheduler. A weak load R is reported when
  all of these hold:
  - the last write W of the bytes R reads comes from another SM;
  - W's grid is not stream-ordered before R's grid, and W does not happen-before R's launch;
  - no L1 invalidation on R's SM is ordered W → I → R.

## Litmus: fixtures/l1_coreader

The victim is DeepEP's barrier handoff (l1_litmus pr0):

- the producer does a weak store of `pay[0]`, then `fence.acq_rel.sys` and a relaxed atomic
  flag;
- the consumer CTA runs the entry fence, `bar.sync`, then a warp-level `ld.volatile` +
  `vote.all` poll with no fence, `bar.sync`, and a weak read of `pay[0]`.

The co-reader is a second kernel with `after = -1` (another stream). Thread 0 weak-loads
`pay[24]`, which is in the same line and is never written. The case sets `poll_acquire` and
`hw_l1`, as r4b_moe does.

| variant | closed world | `l1_coresident` (stateful) | `l1_coresident` + DPOR | `l1_havoc` |
|---|---|---|---|---|
| good | clean | stale (placement consumer CTA + reader CTA) | stale, with an operational hit: the consumer returns 0 instead of 1 | stale (no reader needed) |
| racy-reader (the reader loads `pay[0]`) | race | race, stale | race, stale | race, stale |
| fixed (`fence.acq_rel.sys` after the poll) | clean | clean | clean | clean |

All rows hold for both front-ends (`crates/sim/tests/fixtures.rs` `l1_coreader`). The reader is
legal: the race detector finds nothing in `good` in any mode, and it does flag `racy-reader`.

## DeepEP

### Havoc on the captured layer (fixtures/r4b_moe)

Command:

```
PTXV_L1_HAVOC=1 PTXV_GROUPS_UPTO=2 ptxv check fixtures/r4b_moe --variant good --front sass \
    --groups "2,3;4,5;6-18,21-25;19,26;20,27"
```

Results:

- Layout is certified clean.
- The notify group (launches 2, 3) is exhaustive: 25,828 states, 108 terminal, 24 s, verdict
  `stale`.
- The vulnerable reads are exactly the post-barrier reads by notify_dispatch CTA 0 of the
  counts the peer wrote. All of them are in line 0 of the reader's local NVLink buffer
  (`buffer_ptrs[rank]`):

| rank | read (sass) | bytes | written by the peer at |
|---|---|---|---|
| 0 | `LD.E` 0x1600 (prefix sum, `local[t+2]`) | 0x8 (per-rank row 1, col 0) | rank 1 `ST.E` 0x0730 |
| 0 | `LD.E` 0x16f0 (per-expert sum) | 0x18 (per-expert row 1) | rank 1 `ST.E` 0x0f80 |
| 1 | `LD.E` 0x15f0 (prefix sum, `local[t]`) | 0x4 (per-rank row 0, col 1) | rank 0 `ST.E` 0x0730 |
| 1 | `LD.E` 0x16d0 (per-expert sum) | 0x10 (per-expert row 0) | rank 0 `ST.E` 0x0f80 |
| 1 | `LD.E` 0x1900 (copy to `rank_prefix_matrix`) | 0x4 | rank 0 `ST.E` 0x0730 |

- Nothing in dispatch is reported. Dispatch's weak reads of `rank_prefix_matrix` (0x3940) read
  bytes whose writes happen before dispatch's launch, so the kernel boundary drops older
  copies.
- A deterministic run (`PTXV_RUN_ONCE=1 PTXV_L1_HAVOC=1`, all 28 launches) reports a schedule
  dependent subset (rank 0's two reads) and nothing else. Both outputs stay bitwise equal to
  the GPU's, because havoc only predicts.

### A concrete legal co-reader: fixtures/r4b_coreader

The case is r4b_moe's launches 0-3 (layout and notify, same cubin and data), plus one
co-reader per GPU (`coreader.cubin`, 32 threads, `after = -1`). Thread 0 runs
`LDG.E [buf + 0x60]` on its rank's local buffer: bytes 96..99 of line 0, which nothing in the
case writes.

Command:

```
[PTXV_L1_CORES=1] ptxv check fixtures/r4b_coreader --variant good --front sass --groups "2,3,4,5"
```

| mode | result |
|---|---|
| default (closed world, one CTA per SM) | clean, complete (26,030 states). The readers race with nothing. |
| `PTXV_L1_CORES=1` (19 pair placements) | stale, complete (494,570 states over all placements, 45 s). The same five reads, each only in the placement "notify CTA 0 + that rank's reader" (`kernel 2 CTA 0 + kernel 4 CTA 0` on rank 0, `kernel 3 CTA 0 + kernel 5 CTA 0` on rank 1). |
| `PTXV_L1_HAVOC=1` | stale, the same five reads |

The co-reader action is one weak `LDG.E` of bytes 0x60..0x63. Each stale read needs three
things:

- that load must run on the SM of notify CTA 0 of the rank it reads;
- it must run after that CTA's entry fence (`MEMBAR.ALL.SYS; CCTL.IVALL` at 0x1000/0x1030 in
  the second `barrier_block`);
- it must run before the peer's `ST.E` of the counts lands.

There is no other condition. Nothing on that SM invalidates between the fill and the read: the
barrier's poll has no fence, and E-POLL-ACQ orders but does not invalidate. A reader that spins
on the load refills the line after every invalidation, so on hardware the window is "CTA 0
passes its entry fence before the peer's counts land". `crates/sim/tests/fixtures.rs`
`r4b_coreader_notify` checks all three rows.

**Operational witness (DPOR).** Command:

```
PTXV_L1_CORES=1 PTXV_L1_PLACE=none PTXV_L1_COLOCATE="7,11" PTXV_EXPLORER=dpor ptxv check fixtures/r4b_coreader ...
```

This pins rank 1's notify CTA 0 (global CTA 7) and its reader (CTA 11) to one SM. It did not
finish: DPOR has no launch groups, so it explores layout and notify together. After about
14 minutes it had reached more than 172,000 states and 1,212 terminal states without an
operational hit, and was stopped. So the DeepEP result is the predictive one (stateful explorer,
complete). The operational witness exists only for the litmus (fixtures/l1_coreader under
DPOR: the consumer returns 0 instead of 1).

### Is the reader legal in a whole layer?

In the reduced case nothing writes bytes 0x60..0x63, so the reader is legal there. In the
captured layer it is not. A deterministic run of all 28 launches with the reader added on
rank 1 reports a race on byte 0x60 with:

- dispatch (rank 0's sender, `STG.E.128` at 0x2780: the x ring of channel-rank 0 starts at
  byte 48 when `num_channels` = 1);
- combine (`STG.E.128` at 0x1b50: its x ring starts at byte 16).

Line 0 of the buffer is reused by every kernel of the layer. Reading it legally needs bytes
that no DeepEP kernel writes while the reader runs. From the upstream layout, the installed
notify SASS, and these two races:

- notify writes [0, 16) (counts) and [16, 16 + 8·E/R) (per-expert counts). Its memset clears
  [16, 16 + 32·nc), where nc = `num_channels` = `num_sms`/2.
- dispatch writes its channel metadata at [16, 16 + 32·nc). Its x ring of channel-rank 0
  (channel 0, sender rank 0) starts at 16 + 32·nc, and slot k is written only when that
  channel-rank carries more than k tokens in the call (the memset resets head and tail every
  call).
- combine writes head / tail at [0, 16·nc). Its x ring of channel-rank 0 starts at 16·nc.
- cached_notify_combine clears [0, 16·nc). cached_notify_dispatch writes [0, 16) and the same
  memset as notify.

So with 2 ranks, `num_sms` = 4 (nc = 2) and at most 8 experts, bytes [80, 128) of rank 1's
line 0 are never written, provided two routing conditions hold:

- no token of rank 0's channel 0 is dispatched to rank 1;
- no token of rank 1's channel 0 is dispatched to rank 0. This keeps combine's channel-rank 0
  on rank 1 empty.

Channel c covers a contiguous range of each rank's tokens. So the configuration is: route the
first half of each rank's tokens only to that rank's own experts, and the second half freely.
The peer's count then still changes every call, so a stale read is observable as wrong
`num_recv_tokens_per_expert`, wrong `moe_recv_counter` and a wrong `rank_prefix_matrix`.

The user initializes [80, 128) once through `get_local_buffer_tensor` and synchronizes. DeepEP
does not memset its NVLink buffer at init. After that, a reader of those bytes on another
stream races with nothing for the whole program. That is contrived, but every action is
allowed.

This layout was derived from upstream source and checked against the installed binary only for
notify (SASS) and for the two writes the race detector found. It has not been captured with
nc = 2.

## What this does and does not show

- In the model (E-HW-L1 + E-L1-CORES), a legal co-resident reader makes DeepEP's existing
  notify_dispatch reads stale. Nothing in DeepEP is modified, and the reader races with
  nothing.
- The DeepEP side of the window is exactly what docs/deepep-repro.md identified: it is
  protected only because nothing on the SM loads line 0 between 0x1030 and the reads.
- Whether GB300 shows it depends on two unprobed points:
  - the timing window (rank r's CTA 0 must pass its entry fence before the peer's counts land);
  - E-L1-CORES itself: co-resident CTAs of concurrent kernels sharing L1 lines is "not probed"
    in E-HW-L1.
- `fence.acq_rel.sys` after the barrier poll (the one-instruction fix) makes all of it clean in
  the model (fixtures/l1_coreader `fixed`).
