# Shared-memory litmus tests on GB300 (LDS/STS ordering in hardware)

**Status: hardware evidence, 2026-11-03.** GB300 (sm_103a), CUDA 13.0.88, driver 580.126.20, GPUs
0/1/3. This is the hardware half of the question "may ptxv treat plain SASS `LDS`/`STS` as
relaxed, CTA-scope strong accesses?" The compiler half is `docs/smem-order-mapping.md`. It shows
that ptxas emits the same `LDS`/`STS` bits for weak, volatile and relaxed shared accesses.

**Answer.** 3.1 x 10^12 litmus instances ran on shared memory in one CTA. They produced **no
non-SC outcome**:

- MP, MP with a spinning reader, SB, LB, IRIW, CoRR and CoWR, each with and without fences.
- MP through a merged `STS.64 {data,flag}`.
- 32/64/128-bit tearing.

Each shape ran 2.6 x 10^10 to 1.6 x 10^11 times, under six timing and stress regimes. The
interleaving outcomes that SC allows all show up in large numbers, so the races were really
exercised. The same harness does catch weak behaviour in global memory: up to 68% stale reads
when L1 is not invalidated, and real IRIW violations. So the zeros are not a blind spot.

In practice GB300 shared memory behaves as **sequentially consistent, coherent and
single-copy-atomic up to 128 bits**. This holds for accesses from different warps of a CTA, and
for DSMEM within a cluster. **Modelling LDS/STS as relaxed strong accesses at CTA scope is
supported, and conservative.** Hardware is stronger: even SB and IRIW without fences never
failed. For pure shared-memory traffic the fences changed nothing observable. The only failures
in this study were compiler reorderings and L1 staleness for **global** data:

- the writer's `MEMBAR` matters when a shared flag guards **global** data;
- the reader's acquire matters only because it stops ptxas from hoisting the payload load.

## Harness

Source: `probes/smem_litmus/litmus.cu`. Build it with
`nvcc -O3 -arch=sm_103a -std=c++17 -o litmus litmus.cu`, then run
`CUDA_VISIBLE_DEVICES=<0|1|3> ./litmus <TEST> <mem> <cta|clu> <wf> <rf> [key=val..]`.
`sweep.py` runs every config under six settings, `asym.sh` runs the asymmetric-bank layouts,
and `summarize.py` aggregates results. Raw lines are in `results/*.txt`.

- **Grid.** Each launch is 304 CTAs (2 per SM) or 152 two-CTA clusters. Each group loops over
  `iters` iterations: group barrier; the evaluator warp classifies the previous iteration;
  roles go to warps picked at random (or, with `fixed=1`, warps 0..3); each role sleeps a
  random 0..`dmask` cycles (spin on `SR_CLOCKLO`, with `NANOSLEEP` 1 time in 16); then the
  litmus code runs. Warps without a role hammer a separate 8 KB scratch region with random
  `LDS`/`STS` (0..`smask` iterations).
- **Instances.** Lane *l* of every role warp is instance *l*, with its own locations. So each
  iteration runs 32 independent instances per group (16 or 8 in same-warp mode).
- **Values.** Iteration *k* writes *k+1*, so a location holds exactly *k* ("old") before that.
  Any other value is counted as `?`. Iteration 0 is not counted.
- **Settings** (`sweep.py`):

  | setting | warps | dmask | smask | notes |
  |---|---|---|---|---|
  | `rand` | 16 | 63 | 63 | |
  | `tight` | 16 | 0 | 0 | |
  | `wide` | 4 | 511 | 0 | |
  | `fixed4` | 4 | 31 | 0 | fixed warps |
  | `bank` | 32 | 63 | 255 | lane stride 32 words: every location of every lane is in one bank, so a 32-way conflict replay on every access |
  | `samewarp` | 16 | 31 | 63 | roles are lane halves (or quarters) of one warp |

  `asym.sh` adds layouts where only *one* of x/y is 32-way bank-conflicted
  (`xis=32 is=1`, and the reverse).
- **Memory flavours.**

  | name | PTX | SASS |
  |---|---|---|
  | `shv` | `ld/st.volatile.shared` | `LDS`/`STS` |
  | `shw` | weak `ld/st.shared` | the same `LDS`/`STS` |
  | `glw` | weak global | `LDG.E`/`STG.E` |
  | `gls` | `relaxed.gpu` global | `.STRONG.GPU` |
  | `dsm` | weak `shared::cluster` after `mapa` | `LD.E`/`ST.E` |
  | `dsr` | `relaxed.cluster` | `LD.E.STRONG.GPU`/`ST.E.STRONG.GPU` |

  In cluster mode, roles alternate between rank 0 and rank 1 and location *j* lives in rank
  *j&1*.
- **Fences.**
  - `wf`: 0 none, 1 `fence.acq_rel.cta` (`MEMBAR.ALL.CTA`), 2 `st.release` (`MEMBAR.ALL.CTA; STS`),
    3 `fence.sc.cta` (`MEMBAR.SC.CTA`).
  - `rf`: 0 none, 1 `fence.acq_rel`, 2 `ld.acquire.cta` (`LDS; NOP wait{SB}`), 3 `fence.sc`.
  - MPG and MPGN use different encodings; see their section.

### SASS shapes (verified)

`sass_roles.py` produced `probes/smem_litmus/roles.sass.txt`. It lists, for every kernel, the
memory and fence SASS minus the harness's own accesses. `ctrl.py` decodes the scoreboard bits.

```
MP shv wf=0 rf=0   writer: STS [x]; STS [y]              (two STS back to back, no fence)
                   reader: LDS R23,[y] wbar=SB1; LDS R21,[x] wbar=SB2 wait={}   <- no scoreboard wait between
MP shv wf=0 rf=2   reader: LDS [y] wbar=SB1; NOP wait{SB1}; LDS [x]   (ld.acquire.cta = scoreboard wait only)
MP shv wf=2        writer: STS [x]; MEMBAR.ALL.CTA rbar=SB1; STS [y] wait{SB1}
MP shv wf=1/rf=1   ...; MEMBAR.ALL.CTA; NOP wait{SB0..SB5}; ...
SB shv wf=0        STS [mine]; LDS [other]  (stall 4, no wait between)
LB shv             LDS [mine]; STS [other]
CORR shv           STS x4 to one address; reader LDS x4 to one address
MPS / MPV shv      reader: 64x-unrolled poll LDS [flag] (x4 loop), then LDS [data]
MPV shv            writer: STS.64 [slot] = {data,flag}  (wf=0 flag at +4, wf=1 flag at +0)
TEAR64/128 shv     STS.64 x8 / LDS.64 x8,  STS.128 x8 / LDS.128 x8
```

`shw` (weak) gives SASS identical to `shv` for MP/SB/LB/IRIW. For **CORR shw** and **MPS shw**,
ptxas legally merged the four weak stores and loads into one each, and hoisted the weak poll out
of the loop. Those two rows therefore test nothing and are listed only for completeness.
Likewise ptxas merges repeated weak or relaxed stores to one address for global and DSMEM:
CORR `glw`/`gls`/`dsm`/`dsr` have a single store and four loads (four loads for `gls`/`dsr`,
one for weak), and COWR `dsm` was folded entirely.

## Results

Per row: total instances over all settings, and forbidden (non-SC) outcomes. Outcomes are
reader values, 1 = new, 0 = old. All files: `results/sweep1..4.txt`, `soak.txt`, `asym.txt`.

### Shared memory, one CTA, different warps (`shv`, SASS = plain LDS/STS)

| test | fences (SASS) | instances | forbidden | SC outcomes seen (witnesses that the race happened) |
|---|---|---:|---:|---|
| MP | none | 1.63e11 | **0** | 00 5.2e10, 01 7.7e9, 11 4.7e10 |
| MP | writer `MEMBAR.ALL.CTA` (fence.acq_rel) | 1.34e11 | 0 | 00, 01, 11 |
| MP | writer st.release | 1.63e11 | 0 | 00, 01, 11 |
| MP | writer `MEMBAR.SC.CTA` | 2.7e10 | 0 | 00, 01, 11 |
| MP | reader fence.acq_rel | 2.7e10 | 0 | 00, 01, 11 |
| MP | reader ld.acquire (`NOP wait`) | 1.63e11 | 0 | 00, 01 2e10, 11 |
| MP | both acq_rel / release+acquire | 2.7e10 / 1.34e11 | 0 / 0 | |
| MP, reader polls flag <=256x then reads data | none / rel+acq | 1.63e11 / 1.34e11 | 0 / 0 | 11 (00 only in same-warp mode) |
| MPV: one `STS.64 {data,flag}`, reader polls flag `LDS`, separate data `LDS` | none / acquire; flag at +0 or +4 | 1.34e11 x3, 2.7e10 | 0 (no stale, no `?`) | 11 |
| SB | none | 1.63e11 | **0** | 01 3.7e10, 10 5.7e10, 11 1.3e10 |
| SB | `MEMBAR.ALL.CTA` / `MEMBAR.SC.CTA` | 1.34e11 / 2.7e10 | 0 / 0 | |
| LB | none | 1.63e11 | 0 | 00, 01, 10 |
| CoRR (4 STS vs 4 LDS, same address) | none | 1.34e11 | 0 decreasing, 0 out of range | mixed (reads in the middle of the store sequence) 2.0e10 |
| CoWR (2 writers, each `STS; LDS; STS; LDS`) | none | 1.34e11 | 0 | saw the other writer 2.2e10 |
| IRIW (2 writers, 2 readers, 4 warps) | none | 1.59e11 | **0** | all 15 allowed outcomes, incl. 0101 1.5e7 |
| IRIW | readers `MEMBAR.SC.CTA` | 1.30e11 | 0 | |
| tearing `STS`/`LDS` 32-bit (byte-replicated pattern) | | 1.34e11 | 0 torn | reader saw a mid-sequence value in 5.2e10 |
| tearing `STS.64`/`LDS.64` | | 1.34e11 | 0 torn | overlap 5.8e10 |
| tearing `STS.128`/`LDS.128` | | 1.34e11 | 0 torn | overlap 6.8e10 |
| weak `ld/st.shared`: MP, SB, LB, IRIW (same SASS) | none | 2.7e10 each | 0 | |

The `asym.sh` layouts are included in the totals above: only x, or only y, 32-way
bank-conflicted, for MP/MPS/SB/LB/IRIW and MP with acquire or release, 2.9e10 each. They found
no violation either. A writer's conflicted data `STS` that replays 32 times never lets a later
conflict-free flag `STS` become visible first.

**Same-warp variants** put the roles in lane halves of one warp. They are SC trivially. ptxas
predicates both roles into one instruction stream (`@!P STS; @P LDS`) and Volta-style
divergence runs the paths one after another. Almost every same-warp instance therefore sees
"all new" (or, for polling readers, "all old", because the reader path runs to completion
before the writer path). There were no violations.

### Contrast: DSMEM, and global memory

| test | memory (SASS) | instances | forbidden |
|---|---|---:|---:|
| MP / MPS / SB / LB / IRIW / CORR | DSMEM across 2 CTAs, weak (`ST.E`/`LD.E`) | 6.3e9 each | 0 |
| same set, plus COWR | DSMEM `relaxed.cluster` (`.STRONG.GPU`) | 6.3e9 each | 0 |
| MP rel/acq, SB `fence.sc`, IRIW `fence.sc` | DSMEM | 6.3e9 each | 0 |
| TEAR64 / TEAR128 | DSMEM `ST.E.64/128` vs `LD.E.64/128` | 6.3e9 each | 0 torn |
| MP / MPS / SB / LB / CORR / IRIW | global, one CTA (one SM), weak and `.STRONG.GPU` | 2.6e10 each | 0 |
| MP / MPS / SB / LB / CORR | global, 2 CTAs on 2 SMs, weak and `.STRONG.GPU` | 6.3e9 each | 0 |
| **IRIW** | **global weak `LDG.E`, 2 SMs** | 6.3e9 | **4** (1010) |
| IRIW | global `.STRONG.GPU`, 2 SMs, with or without `fence.sc` | 6.3e9 each | 0 |
| **MPL1** (positive control) | global, 2 SMs. Writer: `STG x; MEMBAR.ALL.GPU; STG.STRONG.GPU y` (correct release). Reader: `LDG x` (primes L1); poll `LDG.STRONG.GPU y`; `LDG x`, no acquire | 6.3e9 | **1.26e9 (20%; 68% in `fixed4`)** |
| MPL1 with reader `fence.acq_rel.gpu` (`MEMBAR; CCTL.IVALL`) | | 6.3e9 | 0 |
| MPL1 within one SM | | 2.7e10 | 0 |

The weak behaviour of global memory is L1 staleness and non-multi-copy-atomicity across SMs.
It appears quickly when it is allowed, so the harness's sensitivity is adequate. DSMEM showed
nothing weak in these shapes.

### Coordinator's shapes: shared flag guarding a global payload (one CTA)

These use separate encodings.

- **MPG**: the reader polls up to 256 times. `wf` is 0 (weak `STG` payload, plain `STS` flag),
  2 (`st.release.cta` flag) or 4 (`st.volatile.global` payload, so ptxas keeps `STG` first).
  `rf` bits: 1 primes L1 with a payload load (4 means prime through the neighbour word), 2 polls
  with `ld.acquire.cta`.
- **MPGN**: the same with no polling. Its `rf` bit 4 means the payload load is
  `ld.relaxed.cta.global`.

| SASS writer | SASS reader | instances | stale payload after flag seen |
|---|---|---:|---:|
| **`STS flag; STG data`**: ptxas put the flag first (weak `st.global` before volatile `st.shared`, wf=0) | prime `LDG`; poll `LDS`; `LDG` (with or without the acquire `NOP wait`) | 2.7e10 x3 | **5e7 to 2.2e8 (0.2 to 0.8%; 3.5% in `bank`)** |
| same | no prime (data not in L1) | 2.7e10 x2 | 0 |
| `STG.STRONG.SYS data; STS flag`, no MEMBAR (wf=4) | prime; poll `LDS` (with or without acquire); `LDG` | 2.7e10 x3 | **0** |
| `STG data; MEMBAR.ALL.CTA; STS flag` (st.release) | prime; poll; `LDG`, with or without acquire | 2.7e10 x4 | 0 |
| `STG data; STS flag`, no MEMBAR (MPGN wf=0) | prime (neighbour word); `LDS flag; NOP wait{SB}; LDG data` (acquire = scoreboard wait only) | 2.7e10 x2 | **0** |
| `STG; MEMBAR.ALL.CTA; STS` | the same | 2.7e10 x2 | 0 |
| either | `rf=1`, no acquire: **ptxas hoisted the payload `LDG` above the flag `LDS`** | 2.7e10 x2 | 1.9e7 / 2.1e8 (expected: the SASS is reordered) |

What these rows show:

- **Hardware.** When the SASS order is `STG data` then `STS flag` (from one thread), and the
  reader issues `LDS flag`, then `LDG data` after the flag value arrived, no stale payload was
  seen. This held with or without `MEMBAR.ALL.CTA`.
- **Every stale result comes from SASS whose order is itself wrong.** In each case ptxas had
  moved a weak global access across the shared flag access, which is legal for the PTX as
  written (no release / no acquire). Combined with a reader whose L1 holds the payload line,
  this gives stale data.
- **For ptxv:** `st.release.cta` (the MEMBAR) and `ld.acquire.cta` (the scoreboard wait) are
  what stop ptxas from reordering. Verified SASS needs no extra hardware ordering beyond its
  instruction order for the same-SM case.
- **The racy prime gets merged across an acquire** (`probes/smem_litmus/cse/*.ptx`). For
  `ld.global x; ld.acquire.{cta.shared|gpu.global} f; ld.global x`, ptxas emits **one** `LDG`
  before the acquire and reuses its value. It keeps two only with `fence.acq_rel` or a strong
  data load. This is legal only because the first load races with the writer. SASS-level
  checking in ptxv sees the reuse as a real hoist, which is correct.

## Conclusion for ptxv

1. **Model shared `LDS`/`STS` (and `STS.64`/`.128`) as relaxed, CTA-scope strong, single-copy-
   atomic accesses.** Confidence is high that hardware is at least this strong. No coherence
   violation (CoRR, CoWR), tearing or MP/SB/LB/IRIW violation appeared in 3.1e12 instances,
   with 1e10 or more witnesses of real interleaving per shape. The test is not conclusive in
   the formal sense, but the harness detects global-memory weakness at rates from 1e-9
   (IRIW) up to 0.68 (MPL1).
2. **The fences needed are the PTX ones, enforced through the compiler, not stronger.**
   - For pure shared traffic, MEMBAR.CTA/SC and the acquire wait had no observable hardware
     effect: SB and IRIW were SC even without fences.
   - ptxv should still require release/acquire (or fences) for synchronisation. Hardware
     acting SC is not an architectural promise. The `MEMBAR` and the scoreboard wait are what
     keep ptxas from moving weak global accesses across the flag, which was the only source of
     real failures here.
   - Treating `LDS` without a following wait as unordered against later independent loads is
     not needed for shared-only data. It matters only for global payloads guarded by shared
     flags, and there only because of the compiler.
3. **DSMEM** (`LD.E`/`ST.E` to `shared::cluster`) also behaved SC and coherent across two CTAs.
   Its SASS already encodes strength (`.STRONG.SM/GPU/SYS`), so it needs no special rule.
4. **Global memory across SMs is genuinely weak.** It shows L1 staleness and IRIW violations.
   Keep modelling plain `LDG`/`STG` as weak.
