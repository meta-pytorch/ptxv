# Rung 1d: a persistent kernel driven by cluster launch control (CLC)

**Status: done (2026-10-30).**
- The good kernel is explored **exhaustively** on both front-ends and is clean. The stateful
  explorer and DPOR agree on it.
- All 3 mutants get their expected verdicts on both front-ends.
- On GB300 at 8192 CTAs, every mutant fails the way the model predicts, and the good kernel
  does every tile exactly once.

CLC is the first feature where *which CTAs exist* depends on the schedule. A CTA can be
cancelled before it launches, and its work is then done by the CTA that cancelled it. The
model therefore had to stop assuming that every CTA of the grid runs (E-GRID).

## The kernel

`fixtures/r1d_clc/kernel.ptx`: a grid of 4 one-warp CTAs covers 4 tiles, with
`out[32 * tile + lane] = 1000 * tile + lane`.

```
all       : tile = ctaid.x
loop      : write the tile ; bar.sync
  t0      : arrive.expect_tx(bar, 16) ; clusterlaunchcontrol.try_cancel [resp], [bar]
            wait bar (parity flips each request) ; ld.shared.b128 resp
            next = is_canceled(resp) ? first_ctaid.x(resp) : -1 ; st.shared next_tile
  all     : bar.sync ; tile = next_tile ; repeat while tile != -1
```

Each tile is done exactly once: either by its own CTA, or by the CTA that cancelled it before
it launched.

## What the probe established (E-CLC)

The model should not guess what `try_cancel` may return. `probes/clc` launches 8192 one-warp
CTAs, has each issue one `try_cancel`, and records whether it ran, the decoded fields and the
raw 16-byte response:
- **Cancelled CTAs never run.** The set of victims is exactly the set of CTAs that wrote
  nothing: about 1970 per run.
- **Victims are not taken in launch order.** CTA 0 got 5404, the next callers 5492, 5260, and
  so on. So the model allows a cancel of *any* unlaunched CTA.
- **Response encoding.** Success is `{ctaid.x, ctaid.y, 1, 0}`; failure is `{2, 0, 0, 0}`.
  `is_canceled` tests bit 0 of word 2. In SASS: `LOP3.LUT P0, RZ, R6, 0x1, RZ, 0xc0, !PT` on
  the third word of an `LDS.128`, then `SEL` picks word 0 or -1.

The model (E-CLC, `exec.rs` `launch_cta` / `complete_clc`):
- every CTA starts **unlaunched**, and a `Launch` action launches any one of them;
- a response either fails (allowed at any time) or cancels any CTA that has not launched;
- the response is 16 bytes written to shared memory plus `complete_tx(16)` on the mbarrier.

The failure-is-always-allowed rule is conservative: a correct kernel must not depend on a
cancel succeeding.

## DPOR: launch keys

A launch and a cancel of the same CTA write that CTA's *launch key* (footprint space 4), so
DPOR explores both orders. The first version of the response completion had no footprint for
launches. DPOR then saw it as independent of every `Launch` and explored a single trace: a
soundness bug, caught because the stateful explorer found 73 terminal states and DPOR found
1. Reading *all* launch keys fixed coverage but exploded the search. The rule that is both
sound and small: **a response reads the keys of the CTAs that have already launched.** Each
launch removed a possible outcome; an unlaunched CTA is still a candidate whatever the order.
With that rule DPOR needs 3,452 states and finds the same 73 terminal states as the stateful
explorer, which needs 430k.

## Verdicts

| variant | bug | PTX model | SASS model | GB300 at 8192 CTAs (`tools/r1d_scale.sh`, 3 runs) |
|---|---|---|---|---|
| good | | clean, exhaustive (3,452 states, 73 terminal) | clean, exhaustive (32,983 states) | 0 of 8192 tiles wrong |
| B17 | the response is read without waiting for its mbarrier | race (async write vs `ld.shared`) + read of unwritten smem + nondeterministic (lost tiles), exhaustive | same | **unspecified launch failure**, every run (an earlier build: illegal memory access) |
| B53 | the wait parity is not flipped for the next request | fault: `mbarrier.arrive.expect_tx` over-arrives (a request is issued while the previous phase is pending), bounded | same | **unspecified launch failure**, every run |
| B54 | a CTA exits after a successful cancel without doing the stolen tile | nondeterministic (tiles lost), exhaustive | same | **1980–1993 of 8192 tiles lost**, every run |

**The fixture's grid shows nothing on hardware.** At 4 CTAs every CTA is resident at once, so
every `try_cancel` fails and all three mutants behave correctly. The hardware evidence needs a
grid large enough that most CTAs are still unlaunched: `tools/r1d_scale.sh` reruns the same
kernel and variants at 8192 CTAs, where the model cannot run but its predictions can be
checked. This is E-GRID made concrete: the model covers the success paths at 4 CTAs because it
allows any launch order, not because 4 CTAs exercise them on GB300.

**B17: the verdict is right, and the crash is outside its scope.** The model reports the race
and the read of unwritten memory, then follows *one* outcome of the racy read: the load runs
before the response lands (E-DRF: plain loads are not scheduling points), sees zeros,
"fails", and the CTA exits, losing tiles. On hardware the load can also see the *previous*
response, a stale success. The CTA then redoes that tile, issues another request while the
mbarrier phase is still pending, and over-arrives, which is B53's fault. E-DRF documents
exactly this: a racy kernel's later consequences may be missed, but the race is always
reported. A clean verdict is unaffected.

**B53** is the same over-arrival that rung 1b's B10 produced (also a launch failure on GB300):
the hardware does not tolerate an arrive on an mbarrier with no pending arrivals. Its search
is bounded (20k states) because the stale "cancelled" response repeats forever.

## New model and lift pieces

- IR: `ClcTryCancel { resp, mbar }`, `ClcQuery` (is_canceled / first_ctaid x, y, z), `Sel`.
- PTX: `clusterlaunchcontrol.try_cancel.async…b128` and `query_cancel.*`; `.b128` registers
  are two 64-bit halves (the first version defaulted them to 32 bits, so tiles "disappeared"
  in the model; the three-way diff caught it).
- SASS: `UGETNEXTWORKID.SELFCAST [URresp], [URmbar]` = `try_cancel` without multicast;
  `SEL` / `USEL`; `LOP3.LUT` with a predicate destination.
- Explorer: a `Launch(cta)` action and per-alternative `Clc` completions; DPOR processes
  `Pid::Launch(cta)`.
- Not modeled: multicast responses (`.multicast::cluster::all`), CLC in a cluster launch,
  2-D/3-D grids (the response carries y, but only 1-D grids were probed).

## Reproduce

```
PTXV_EXPLORER=dpor ptxv check fixtures/r1d_clc --max-states 20000   # B53 is unbounded
ptxv probe probes/clc
tools/r1d_scale.sh 3                                                 # GB300, 8192 CTAs
cargo test --release --test fixtures r1d                             # ~2 min
```
