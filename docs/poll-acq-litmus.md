# Poll-exit acquire and shared-memory cumulativity on GB300 (E-POLL-ACQ, E-SMEM-CUMUL)

**Status: hardware evidence, 2026-11-04.** GB300 (sm_103a), CUDA 13.0.88, driver 580.126.20.
Cross-GPU runs on GPUs 0 and 1 (NVLink, NV18), intra-GPU runs on GPU 3. This note tests the
two hardware behaviours that DeepEP relies on beyond the PTX model (docs/rung4.md, T4b). ptxv
takes both as opt-in assumptions.

**Answers.**

1. **E-POLL-ACQ is refuted.** Leaving a warp-level poll loop does not acquire. The loop is
   DeepEP's `barrier_block` shape: relaxed `ld.volatile` polls, `__all_sync`, a clock timeout,
   optionally a `bar.sync` with a second warp, and no fence. After it, a weak read
   (`LD.E`, `LDG.E` or `LDG.E.CONSTANT`) returns the value from before the producer's
   release whenever the payload line is valid in the consumer SM's L1. Producer and consumer
   only have to be on different SMs. This happened in every cross-SM placement: on another SM
   of the same GPU, and across NVLink in both directions, with the payload in the consumer's
   or in the producer's memory. Without stress almost every such read was stale
   (99.99 to 100%): 476,705,632,772 stale of 701,986,992,896 cross-SM reads in 220 runs.
   The cause is L1 non-coherence, not reordering: an acquire fence after the loop
   (`fence.acq_rel.sys` = `MEMBAR.ALL.SYS; CCTL.IVALL`) removed every stale read. So did a
   volatile read (`LDG.E.STRONG.SYS`, which does not hit in L1), and so did an L1 holding no
   copy of the line.
2. **DeepEP's actual shapes never returned stale data.** These were 3,538,947,554,048 payload
   reads in 90 runs, 2.27e12 of them cross-SM (table "DeepEP shapes"). They are safe on this hardware for reasons outside the
   assumption:
   - every `barrier_block` that guards data starts with `fence.acq_rel.sys`
     (`MEMBAR.ALL.SYS; CCTL.IVALL`). That invalidates L1 *before* the poll, and nothing loads
     the payload lines between that fence and the reads;
   - combine reads its channel data with `ld.volatile` (`LDG.E.128.STRONG.SYS`).

   The first condition depends on cache state. In the same `barrier_block` shape, when other
   warps of the SM kept loading the payload lines during the barrier, 0.07 to 0.8% of reads
   were stale (rows `*-pr0-payld`: 160,397,332 of 4.07e10 reads).
3. **E-SMEM-CUMUL is supported** (no stale read). The rule tested is that a shared store
   passes on, at CTA scope, what its thread acquired. In the shape tested, the coordinator
   lane acquires with `ld.acquire.sys` (`LDG.E.STRONG.SYS; CCTL.IVALL`) and relays the value
   with `st.volatile.shared` (`STS`). The receiver polls with `LDS`, then reads. The receiver's
   reads were never stale, whether weak, volatile or `.nc`, and including when the receiver
   kept the payload lines in L1, or reloaded them, while it polled: 0 stale in 8,274,458,067,200
   reads in 180 runs, 5,282,878,210,816 of them cross-SM. The mechanism is that
   `CCTL.IVALL` invalidates the whole SM's L1, not only the issuing warp's lines. When the
   coordinator only relaxed-polls (no `CCTL.IVALL`), the same relay gives stale reads at 6 to
   100% (positive control). So the rule holds only for real acquires. It does **not** extend
   to what E-POLL-ACQ calls an acquire.

In total, 642 runs made 13,472,170,625,792 payload reads.

Litmus evidence is not proof. Zero stale reads in N instances bounds the rate under these
conditions; it does not rule out a rarer interleaving, another chip, or another driver.

## Harness

Source: `probes/poll_acq/pollacq.cu`. Build it with
`nvcc -O3 -arch=sm_103a -std=c++17 -o pollacq pollacq.cu`, then run
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 ./pollacq <poll|relay> <form> <rd> cdev=0 pdev=1 [key=val..]`.

- `sweep.py <plan> <out> --secs S` runs a plan. It calibrates each configuration, runs it for
  about S seconds, and waits while a process it did not start is on one of its GPUs.
- `summarize.py [--by placement|stress|both]` aggregates the results.
- `roles.sass.txt` lists the memory and synchronization SASS of every kernel.
- Raw lines are in `results/{refute,controls,deepep,cumul}.txt`.
- `crates/sim/tests/poll_acq_litmus.rs` replays them.

**Structure.**

- A *pair* is a producer CTA and a consumer CTA. Each runs `W` (2 or 4) independent instances,
  one warp each, or two warps for the relay and `bar.sync` forms.
- Lane *l* of an instance owns a 16-byte payload (4 words). The lane stride `ps` is 4 words
  (lanes share lines) or 32 words (one 128-byte line per lane).
- Lanes `< nf` (`nf` = 1, 2, 8 or 32) own a flag.
- Iteration *k* runs as follows:
  1. The producer waits for the consumer's ack of k-1.
  2. It sleeps a random 0..`dmask` cycles.
  3. It writes payload = k.
  4. It releases, then signals.
  5. The consumer waits, reads the four words and classifies them: all k is ok; any word < k
     is **stale**; anything else is "other" (never seen).
  6. The consumer acks with `st.volatile`.
- Iteration 1 starts from zeroed memory. Each lane read is one instance ("reads" below).
- "waited" counts iterations in which the consumer's first poll still saw the old flag, so the
  read follows the flag's arrival closely. It is the witness that the race window was
  exercised: 70 to 100% per family.

**Producer** (`wf`), in the SASS forms DeepEP compiles to:

| wf | PTX | SASS | DeepEP |
|---|---|---|---|
| 0 | generic `st.u32` x4; `fence.acq_rel.sys`; `__syncwarp`; `atom.add.sys` on the flag | `ST.E` x4; `MEMBAR.ALL.SYS`; `CCTL.IVALL`; `ATOM.E.ADD.STRONG.SYS PT, RZ` | notify_dispatch stores + `barrier_block` signal (k0.cubin 0x0730.., 0x1000, 0x1100) |
| 1 | `st.global.v4`; `__syncwarp`; `st.release.sys.global` flag | `STG.E.128`; `MEMBAR.ALL.SYS`; `STG.E.STRONG.SYS` | dispatch / combine channel tail |

**Consumer**: `poll` test (E-POLL-ACQ), `form`:

| form | shape | SASS (roles.sass.txt) |
|---|---|---|
| 0 | warp poll: lanes `< nf` `ld.volatile` their flag, others vote true; `if (__all_sync(...)) break;` clock timeout | `LDG.E.STRONG.SYS`; `VOTE.ALL`; `CS2R SR_CLOCKLO`; back branch; then the reads. DeepEP: the same plus `BSSY/BSYNC`, `BRA.DIV` around the vote (k0.cubin 0x1340..0x1430) |
| 1 | per-lane spin, no vote (control shape) | `LDG.E.STRONG.SYS` loop |
| 2 | form 0, then `bar.sync 1+w, 64` with a second warp, which does the reads | as form 0 + `BAR.SYNC`; DeepEP `BAR.SYNC 0x0` at 0x1590 |

**Consumer**: `relay` test (E-SMEM-CUMUL), `form`:

| form | coordinator warp | receiver warp |
|---|---|---|
| 0 | `LDG.E.STRONG.SYS; CCTL.IVALL; STS` (`ld.acquire.sys` + `st.volatile.shared`), re-published every poll, as in DeepEP combine (k0.cubin combine 0x2870..0x28c0) | `LDS` poll + `VOTE.ALL`, then the reads (combine 0x35d0..0x4190) |
| 1 | `LDG.E.STRONG.SYS; STS`, no `CCTL.IVALL` (positive control) | same |

**Payload read** (`rd`):

| rd | PTX | SASS |
|---|---|---|
| 0 | generic `ld.u32` x4 | `LD.E`: notify_dispatch's reads after `barrier_block` (0x15f0..) |
| 1 | `ld.global.u32` x4 | `LDG.E` |
| 2 | `ld.volatile.global.v4.u32` | `LDG.E.128.STRONG.SYS`: combine's channel reads (0x4190; DeepEP built with `DISABLE_AGGRESSIVE_PTX_INSTRS`) |
| 3 | `fence.acq_rel.sys`, then `ld.global` (negative control) | `MEMBAR.ALL.SYS; CCTL.IVALL; LDG.E` |
| 4 | `ld.global.nc.v4.u32` | `LDG.E.128.CONSTANT` (DeepEP's default `ld_nc_global`) |

**Payload state in the reader's L1 at poll exit** (`pr`):

| pr | state |
|---|---|
| 0 | `fence.acq_rel.sys` before the poll, no payload load after it (DeepEP's `barrier_block` entry fence) |
| 1 | the reader's own read from the previous iteration (natural) |
| 2 | weak payload load just before the poll |
| 3 | weak payload load before the poll and in every poll iteration (the reload can be in flight when the flag arrives) |

**Placements.**

| name | producer / consumer |
|---|---|
| `x01` | GPU 1 writes over NVLink into GPU 0's memory; GPU 0 polls and reads locally (DeepEP intranode) |
| `x10` | the reverse direction |
| `x01-loc1` | payload in the producer's memory (GPU 1); the consumer reads it over NVLink; flag local to the consumer |
| `i-far` | one GPU, consumer CTA i, producer CTA i+76; one CTA per SM (120 KB dynamic smem) |
| `i-near` | one GPU, CTAs 2i / 2i+1 |
| `i-same` | producer and consumer warps in the same CTA (same SM) |

Cross-GPU runs have 148 pairs (148 CTAs per GPU), intra runs 76 pairs (152 CTAs) or 148 CTAs
(i-same).

**Stress.** Four extra warps per CTA (`stress=4`), in one of these modes:

| mode | what the stress warps do |
|---|---|
| `l2` | random weak loads / stores over 512 MB |
| `atom` | atomics on 8 hot lines |
| `nvl` | random loads / stores to the peer GPU's scratch (cross runs) or local scratch (intra) |
| `l1` | loads over 1 MB (L1 thrash) |
| `payld` | weak loads of the CTA's own payload lines |

Each configuration also draws `dmask` from {0, 63, 255, 4095} cycles, `nf` from
{1, 2, 8, 32}, `ps` from {4, 32}, `W` from {2, 4} and a seed.

## Results

Each row sums its runs over placements and stress modes. Instances are lane payload reads.

Exact totals (all placements, and cross-SM = without `i-same`):

| group | runs | reads | stale | cross-SM runs | cross-SM reads | cross-SM stale |
|---|---:|---:|---:|---:|---:|---:|
| DeepEP shapes | 90 | 3538947554048 | 0 | 75 | 2269165409024 | 0 |
| E-SMEM-CUMUL relay | 180 | 8274458067200 | 0 | 150 | 5282878210816 | 0 |
| E-POLL-ACQ refutation | 264 | 1223923395072 | 476705632772 | 220 | 701986992896 | 476705632772 |
| positive controls | 72 | 312806190080 | 128308936902 | 60 | 162659512832 | 128308936902 |
| negative controls | 36 | 122035419392 | 0 | 30 | 82094533888 | 0 |
| all | 642 | 13472170625792 | 605014569674 | | | |

### DeepEP shapes (plan `deepep`, ~30 s per run, stress none / l2 / atom / nvl / l1)

| family | shape | runs | reads (cross-SM) | stale (cross-SM) | stale % | reads (i-same) | stale (i-same) | waited % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bb-f0` | barrier_block (wf0), consumer warp reads, `LD.E`, entry fence (pr0) | 30 | 8.44e+11 | 0 | 0.000 | 5.52e+11 | 0 | 84.6 |
| `bb-f2` | barrier_block (wf0) + `bar.sync`, second warp reads, `LD.E`, entry fence (pr0) | 30 | 6.53e+11 | 0 | 0.000 | 2.55e+11 | 0 | 100.0 |
| `poll-f0-rd2-pr3` | warp poll, `LDG.E.128.STRONG.SYS`, reloaded in every poll iteration (pr3) | 30 | 7.72e+11 | 0 | 0.000 | 4.62e+11 | 0 | 98.6 |

### E-SMEM-CUMUL (plan `cumul`, ~30 s per run, no stress and five stress modes)

| family | shape | runs | reads (cross-SM) | stale (cross-SM) | stale % | reads (i-same) | stale (i-same) | waited % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `relay-f0-rd1-pr3` | relay: coordinator `LDG.E.STRONG.SYS; CCTL.IVALL; STS`, `LDG.E`, reloaded in every poll iteration (pr3) | 36 | 1.02e+12 | 0 | 0.000 | 6.56e+11 | 0 | 99.9 |
| `relay-f0-rd1-pr1` | relay: coordinator `LDG.E.STRONG.SYS; CCTL.IVALL; STS`, `LDG.E`, line from previous read (pr1) | 36 | 1.05e+12 | 0 | 0.000 | 7.03e+11 | 0 | 99.8 |
| `relay-f0-rd2-pr3` | relay: coordinator `LDG.E.STRONG.SYS; CCTL.IVALL; STS`, `LDG.E.128.STRONG.SYS`, reloaded in every poll iteration (pr3) | 36 | 1.06e+12 | 0 | 0.000 | 3.95e+11 | 0 | 99.5 |
| `relay-f0-rd4-pr3` | relay: coordinator `LDG.E.STRONG.SYS; CCTL.IVALL; STS`, `LDG.E.128.CONSTANT`, reloaded in every poll iteration (pr3) | 36 | 1.08e+12 | 0 | 0.000 | 6.12e+11 | 0 | 100.0 |
| `relay-f0-rd0-pr2` | relay: coordinator `LDG.E.STRONG.SYS; CCTL.IVALL; STS`, `LD.E`, reloaded before the poll (pr2) | 36 | 1.08e+12 | 0 | 0.000 | 6.25e+11 | 0 | 99.8 |

### E-POLL-ACQ refutation (plan `refute`, ~3 s per run, stress none / l1)

| family | shape | runs | reads (cross-SM) | stale (cross-SM) | stale % | reads (i-same) | stale (i-same) | waited % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `poll-f0-rd0-pr1` | warp poll, `LD.E`, line from previous read (pr1) | 12 | 3.28e+10 | 23362685402 | 71.299 | 3.17e+10 | 0 | 99.2 |
| `poll-f0-rd0-pr2` | warp poll, `LD.E`, reloaded before the poll (pr2) | 12 | 3.38e+10 | 24592901594 | 72.730 | 1.66e+10 | 0 | 98.5 |
| `poll-f0-rd0-pr3` | warp poll, `LD.E`, reloaded in every poll iteration (pr3) | 12 | 2.89e+10 | 23288890364 | 80.535 | 2.78e+10 | 0 | 98.2 |
| `poll-f0-rd1-pr1` | warp poll, `LDG.E`, line from previous read (pr1) | 12 | 3.2e+10 | 22441417763 | 70.048 | 3.26e+10 | 0 | 99.9 |
| `poll-f0-rd1-pr2` | warp poll, `LDG.E`, reloaded before the poll (pr2) | 12 | 3.31e+10 | 25117126465 | 75.789 | 2.46e+10 | 0 | 98.5 |
| `poll-f0-rd1-pr3` | warp poll, `LDG.E`, reloaded in every poll iteration (pr3) | 12 | 3.13e+10 | 24917786452 | 79.731 | 2.94e+10 | 0 | 99.3 |
| `poll-f0-rd4-pr1` | warp poll, `LDG.E.128.CONSTANT`, line from previous read (pr1) | 12 | 3.59e+10 | 17149492750 | 47.833 | 2.85e+10 | 0 | 99.3 |
| `poll-f0-rd4-pr2` | warp poll, `LDG.E.128.CONSTANT`, reloaded before the poll (pr2) | 12 | 2.92e+10 | 21581844792 | 73.856 | 2.57e+10 | 0 | 99.8 |
| `poll-f0-rd4-pr3` | warp poll, `LDG.E.128.CONSTANT`, reloaded in every poll iteration (pr3) | 12 | 3.13e+10 | 26773233946 | 85.486 | 3.09e+10 | 0 | 98.3 |
| `poll-f1-rd0-pr1` | lane spin, `LD.E`, line from previous read (pr1) | 12 | 3.23e+10 | 21596724969 | 66.844 | 2.89e+10 | 0 | 99.4 |
| `poll-f1-rd0-pr2` | lane spin, `LD.E`, reloaded before the poll (pr2) | 12 | 3.46e+10 | 24598458750 | 71.013 | 2.46e+10 | 0 | 98.2 |
| `poll-f1-rd0-pr3` | lane spin, `LD.E`, reloaded in every poll iteration (pr3) | 12 | 3.46e+10 | 27559935397 | 79.689 | 2.78e+10 | 0 | 98.7 |
| `poll-f1-rd1-pr1` | lane spin, `LDG.E`, line from previous read (pr1) | 12 | 3.41e+10 | 23589601707 | 69.214 | 3.5e+10 | 0 | 99.6 |
| `poll-f1-rd1-pr2` | lane spin, `LDG.E`, reloaded before the poll (pr2) | 12 | 3.37e+10 | 25686828460 | 76.269 | 1.95e+10 | 0 | 98.8 |
| `poll-f1-rd1-pr3` | lane spin, `LDG.E`, reloaded in every poll iteration (pr3) | 12 | 3.23e+10 | 26805923053 | 82.890 | 2.66e+10 | 0 | 99.6 |
| `poll-f2-rd0-pr1` | warp poll + `bar.sync`, second warp reads, `LD.E`, line from previous read (pr1) | 12 | 2.99e+10 | 20649900240 | 69.061 | 1.7e+10 | 0 | 100.0 |
| `poll-f2-rd0-pr2` | warp poll + `bar.sync`, second warp reads, `LD.E`, reloaded before the poll (pr2) | 12 | 2.78e+10 | 18974169631 | 68.323 | 1.51e+10 | 0 | 100.0 |
| `poll-f2-rd0-pr3` | warp poll + `bar.sync`, second warp reads, `LD.E`, reloaded in every poll iteration (pr3) | 12 | 2.87e+10 | 18908218906 | 65.795 | 1.86e+10 | 0 | 100.0 |
| `poll-f2-rd1-pr1` | warp poll + `bar.sync`, second warp reads, `LDG.E`, line from previous read (pr1) | 12 | 2.86e+10 | 19798510195 | 69.196 | 1.51e+10 | 0 | 100.0 |
| `poll-f2-rd1-pr2` | warp poll + `bar.sync`, second warp reads, `LDG.E`, reloaded before the poll (pr2) | 12 | 2.83e+10 | 20213719546 | 71.496 | 1.86e+10 | 0 | 100.0 |
| `poll-f2-rd1-pr3` | warp poll + `bar.sync`, second warp reads, `LDG.E`, reloaded in every poll iteration (pr3) | 12 | 2.81e+10 | 18937865058 | 67.405 | 1.33e+10 | 0 | 100.0 |
| `poll-f0-rd0-pr0-payld` | warp poll, `LD.E`, entry fence (pr0), stress warps load the payload lines | 6 | 2.56e+10 | 34934890 | 0.137 | 6.69e+09 | 0 | 92.2 |
| `poll-f2-rd0-pr0-payld` | warp poll + `bar.sync`, second warp reads, `LD.E`, entry fence (pr0), stress warps load the payload lines | 6 | 1.51e+10 | 125462442 | 0.833 | 7.58e+09 | 0 | 100.0 |

### Controls (plan `controls`, ~3 s per run, every placement x {none + five stress modes})

| family | shape | runs | reads (cross-SM) | stale (cross-SM) | stale % | reads (i-same) | stale (i-same) | waited % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ctl-pos-relay-f1` | relay, coordinator relaxed poll without `CCTL.IVALL`, `LDG.E`, line from previous read (pr1) | 36 | 8.08e+10 | 62814229029 | 77.717 | 7.39e+10 | 0 | 99.9 |
| `ctl-pos-relay-f1-pr2` | relay, coordinator relaxed poll without `CCTL.IVALL`, `LDG.E`, reloaded before the poll (pr2) | 36 | 8.18e+10 | 65494707873 | 80.033 | 7.63e+10 | 0 | 99.6 |
| `ctl-neg-poll-rd3` | warp poll, `fence.acq_rel.sys`; `LDG.E`, reloaded in every poll iteration (pr3) | 36 | 8.21e+10 | 0 | 0.000 | 3.99e+10 | 0 | 99.8 |

### E-POLL-ACQ refutation by placement

| placement | rows | runs | reads | stale | stale % |
|---|---|---:|---:|---:|---:|
| i-far | pr0 + payld | 2 | 1.07e+10 | 6931753 | 0.065 |
| i-far | pr1-3, l1 thrash | 21 | 9.22e+10 | 61776721248 | 66.997 |
| i-far | pr1-3, no stress | 21 | 8.2e+10 | 82043442023 | 100.000 |
| i-near | pr0 + payld | 2 | 1.3e+10 | 86611156 | 0.666 |
| i-near | pr1-3, l1 thrash | 21 | 8.84e+10 | 62029464719 | 70.154 |
| i-near | pr1-3, no stress | 21 | 8.05e+10 | 80465428049 | 99.994 |
| i-same | pr0 + payld | 2 | 1.43e+10 | 0 | 0.000 |
| i-same | pr1-3, l1 thrash | 21 | 2.49e+11 | 0 | 0.000 |
| i-same | pr1-3, no stress | 21 | 2.59e+11 | 0 | 0.000 |
| x01 | pr0 + payld | 2 | 6.54e+09 | 20036933 | 0.306 |
| x01 | pr1-3, l1 thrash | 21 | 5.24e+10 | 9306001322 | 17.758 |
| x01 | pr1-3, no stress | 21 | 5.63e+10 | 56340810496 | 99.999 |
| x01-loc1 | pr0 + payld | 2 | 4.02e+09 | 28222524 | 0.702 |
| x01-loc1 | pr1-3, l1 thrash | 21 | 4.62e+10 | 7051119720 | 15.261 |
| x01-loc1 | pr1-3, no stress | 21 | 5.43e+10 | 54265239552 | 99.999 |
| x10 | pr0 + payld | 2 | 6.37e+09 | 18594966 | 0.292 |
| x10 | pr1-3, l1 thrash | 21 | 5.67e+10 | 10985148727 | 19.372 |
| x10 | pr1-3, no stress | 21 | 5.23e+10 | 52281859584 | 99.999 |

### Positive control (relay without `CCTL.IVALL`), stale % by placement and stress

| placement | none | l2 | atom | nvl | l1 | payld |
|---|---:|---:|---:|---:|---:|---:|
| x01 | 100.0 | 61.4 | 100.0 | 71.8 | 6.8 | 100.0 |
| x10 | 100.0 | 51.3 | 100.0 | 71.7 | 12.2 | 100.0 |
| x01-loc1 | 100.0 | 57.1 | 100.0 | 74.0 | 11.3 | 100.0 |
| i-far | 100.0 | 86.5 | 100.0 | 82.1 | 71.8 | 100.0 |
| i-near | 100.0 | 83.5 | 100.0 | 82.3 | 70.5 | 100.0 |
| i-same | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Dispatch ring-slot reuse (V's finding)

DeepEP intranode dispatch reuses ring slots in this order (k0.cubin, dispatch):

1. The receiver bulk-reads a slot into shared memory: `UBLKCP.S.G` at 0x4250, then the
   mbarrier wait (`SYNCS.PHASECHK.TRANS64.TRYWAIT`) at 0x4270.
2. After `BAR.SYNC`, the elected lane publishes the ring head with a relaxed store and no
   MEMBAR (`STG.E.STRONG.SYS` at 0x7fc0).
3. The remote sender polls the head (`LDG.E.STRONG.SYS`), then overwrites the slot
   (`STG.E.128`).

Under the PTX model the slot read is not ordered before the head store, so the overwrite
races with the read. `probes/poll_acq/ringwar.cu` runs exactly this shape (sender: poll
head, `STG.E.128` x N, `MEMBAR.ALL.SYS; STG.E.STRONG.SYS` tail; receiver:
`LDG.E.STRONG.SYS; CCTL.IVALL` tail, `UBLKCP.S.G`, `TRYWAIT`, `STG.E.STRONG.SYS` head). It
counts copies that contain the sender's *next* value ("early"). The runs used 148 pairs x 4
warps, GPUs 0 <-> 1 in both directions, slots of 512 / 4096 / 8192 bytes, the slot in the
receiver's memory (DeepEP) or in the sender's (the copy crosses NVLink), and delays of
0 / 255 / 4095 cycles. Raw lines are in `ringwar.txt`.

| order | runs | copies | early |
|---|---:|---:|---:|
| DeepEP: wait on the mbarrier, then the relaxed head store | 36 | 21312000000 | 0 |
| control: head stored right after issuing the copy, then wait (8 KB slot, remote) | 4 | 710400000 | 3772856 |

So the race V found is real in the PTX model and latent on GB300. The mbarrier wait returns
only after the bytes have landed in shared memory, so by the time the head store issues, the
read is finished in hardware. The control shows that the harness detects an overwrite when
the head is published while the copy is still in flight (0.15 to 0.83% of copies). The PTX
fix is a release on the head store (`st.release.sys`, i.e. `MEMBAR.ALL.SYS` first).

## What this means for ptxv and DeepEP

- **E-POLL-ACQ is unsound as a model rule.** Every verdict that relies on it can miss a stale
  read: any kernel whose poll loop exits with a payload line valid in L1. For example, it
  would accept `barrier_block<kSyncOnly = true>` followed by reads of peer-written data. The
  strict model's T4b race reports on DeepEP (docs/rung4.md) are real PTX-model races.
- **DeepEP's handoffs through `barrier_block` are latent on GB300**, not wrong. Each guarded
  read follows the barrier's own entry `fence.acq_rel.sys`, and in the captured layer nothing
  loads those lines in the window. They would break if another warp or CTA on the SM read the
  payload during the barrier (rows `*-pr0-payld`). The PTX fix is one instruction: an acquire
  fence after the vote (`fence.acq_rel.sys`, or `ld.acquire` polls).
- **Rules consistent with these results** (not implemented):
  - a poll exit acquires for the thread's later *strong* reads (`.STRONG.SYS/GPU`, volatile)
    only;
  - a `CCTL.IVALL` acquires for every thread of the CTA (E-SMEM-CUMUL as validated);
  - the "entry fence + no intervening load" argument needs per-SM L1 state, which the model
    does not have.
- **E-SMEM-CUMUL** is validated for the coordinator shape DeepEP uses, with an acquire that is
  a real `CCTL.IVALL`. With E-POLL-ACQ's pseudo-acquire as the source of the relayed
  knowledge, the relay is stale (positive control).
- **Same-SM caveat.** With producer and consumer on one SM (`i-same`), nothing was ever stale,
  not even the positive controls. The producer's stores pass through the shared L1. Those
  runs exercise the code but are not sensitive, and the summary counts them separately.
- **Not covered:** kernel boundaries (DeepEP's notify -> dispatch and dispatch ->
  cached_notify_combine handoffs cross a kernel end, see E-EXIT-FENCE / E-PDL); more than two
  GPUs; clusters.
