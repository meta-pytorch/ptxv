# Rung 1c: thread-block clusters

## Part (i): two CTAs sharing a multicast operand

**Status: done (2026-10-30).**
- The good kernel is explored **exhaustively** on both front-ends and is clean.
- PTX model, SASS model and GB300 agree **bitwise** (1000/1000 runs, shared memory scrubbed).
- B13 and B14 get their expected verdicts.

Part (ii) (below) adds `cta_group::2`: paired TMEM allocation, the 2-SM MMA, multicast commit
and B12.

### The kernel

`fixtures/r1c_cluster/kernel.ptx`: a cluster of 2 CTAs (`.reqnctapercluster 2, 1, 1`)
computes D[256×16] = A[256×128] · B[16×128]ᵀ. CTA rank r owns rows 128r…128r+127 and runs its
own `cta_group::1` MMAs. K is 2 k-blocks of 64 through **one** shared-memory stage.

```
both CTAs : init full / empty / peer_empty / done ; fence.mbarrier_init.release.cluster ;
            barrier.cluster.arrive.relaxed ; barrier.cluster.wait            (CUTLASS cluster start)
t0, per k : expect_tx(16 KB A + 2 KB B) ; TMA own A rows
            rank 0 only: TMA B with .multicast::cluster, ctaMask 0b11 -> same offsets in both CTAs
            wait full ; 4 x tcgen05.mma ; commit -> empty
            before reusing the stage: wait empty ; rank 1: REMOTE mbarrier.arrive.release.cluster on
            rank 0's peer_empty ; rank 0: wait peer_empty before multicasting the next B
all       : wait done ; tcgen05.ld ; store row ; dealloc ; barrier.cluster arrive + wait (final)
```

Rank 0 may overwrite the shared B stage only after rank 1's MMA has read it. That is the
CUTLASS multicast consumer-release pattern, done here with a remote arrive. Rank 1 also
releases the *last* stage, which nobody waits for. The final cluster barrier is what keeps
that last arrive from landing in an exited CTA.

### Verdicts

| variant | bug | PTX model | SASS model | GB300 (1000 runs, scrubbed) |
|---|---|---|---|---|
| good | | clean, exhaustive (157 states) | clean, exhaustive (2,378 states) | bitwise = both models |
| B13 | no final `barrier.cluster` | violation: rank 1's last remote arrive reaches rank 0 after it exited | same | correct (UB, invisible) |
| B14 | no `fence.mbarrier_init.release.cluster` | race: rank 1's `mbarrier.init` vs rank 0's multicast `complete_tx` | **clean**: the binary is the same | correct |

B14 is PTX-level UB that the binary cannot express. ptxas emits `FENCE.VIEW.ASYNC.S` ahead of
the `SYNCS.EXCH` inits *with or without* the PTX fence; the two cubins differ by one NOP. It
is the same pattern as B05/B06 in rung 1a: the PTX front-end finds a contract violation that
this ptxas happens to make safe.

### What the SASS revealed

| PTX | SASS (sm_103a) | model |
|---|---|---|
| any `.shared` address | `LEA R, SR_CgaCtaId, off, 0x18`: **rank in bits 24+** | S-DSMEM: `(rank << 24) \| offset`, `shared::cluster` selects the CTA |
| `mapa.shared::cluster.u32 d, a, 0` | `PRMT R, RZ, 0x654, R` (clear the top byte) | `Mapa`; `PRMT` implemented generally |
| `barrier.cluster.arrive.relaxed` / `.arrive` | `UCGABAR_ARV`; the release form adds `MEMBAR.ALL.CTA; MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` | relaxed arrive releases only what fences released (E-CLUSTER-BAR) |
| `barrier.cluster.wait` | `UCGABAR_WAIT; CCTL.IVALL`, **guarded by `c[0x0][0x36c] == 1`** | `0x36c` = `%is_explicit_cluster`, measured by `probes/cluster_flag` |
| `fence.mbarrier_init.release.cluster` | `FENCE.VIEW.ASYNC.S` **hoisted above** the inits, also emitted without the PTX fence | SASS: a relaxed arrive releases the thread's earlier inits |
| `mbarrier.arrive.release.cluster.shared::cluster` | `MEMBAR…; SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R], RZ` | remote `MbarArrive` |
| `try_wait.parity.acquire.cluster` | `SYNCS.PHASECHK…TRYWAIT P0; @P0 CCTL.IVALL; @!P0 BRA` | the E-SPIN shape check now allows model no-ops whatever their guard |
| `cp.async.bulk.tensor…multicast::cluster`, mask `0b11` | `UTMALDG.2D.MULTICAST [URa], [URb], URc` with **URc = mask << 16** | one async delivery per destination CTA |
| `%cluster_nctarank` | `CS2R.32 R, SR_CgaSize` | |

### New model pieces (ASSUMPTIONS.md)

* **E-CLUSTER-BAR.** `barrier.cluster` is a split arrive/wait over the cluster's live threads.
  A relaxed arrive publishes only what the thread's last fences released. mbarrier-init
  visibility travels on its own clock, so missing-fence bugs (B14) are caught without hiding
  data races.
* **S-DSMEM**, **S-SCOPE** (scopes treated as sufficient).
* **S-CLUSTER-LIFE.** A remote access after the whole target CTA exited is reported and
  dropped; dropping avoids consequential deadlocks in the issuer's CTA. Independently of the
  explored schedule, every remote access must happen-before *some* exit of the target, checked
  at terminal states.
* TMA `complete_tx` and `tcgen05.commit` now check mbarrier-init ordering. Before this rung
  they only checked that the barrier was initialized.

**Lift fix found along the way.** The scoreboard rule "vector instructions read URs at issue"
now also applies to instructions with a read scoreboard. ptxas overwrites the `desc[UR6]` of
an `STG` that has `rd=SB0` right away. Re-scoring the GB300 mutation sweep moved one mutant
per kernel from "flagged, GPU won" to "not flagged", and still left **0 missed**
(sass-ctrl.md).

### Reproduce

```
PTXV_EXPLORER=dpor ptxv check fixtures/r1c_cluster
ptxv hw fixtures/r1c_cluster --scrub
ptxv probe probes/cluster_flag
cargo test --release --test fixtures r1c
```

The stateful explorer does not finish this kernel in 15 minutes (two CTAs double the
interleavings). DPOR has been cross-checked against it on every fixture it can finish.

## Part (ii): one 2-SM MMA (`cta_group::2`)

**Status: done (2026-10-30).**
- PTX model, SASS model and GB300 agree **bitwise** (1000/1000 runs, scrubbed).
- B12 deadlocks in both models and **hangs on GB300**.

### The kernel

`fixtures/r1c_2sm/kernel.ptx` runs one `tcgen05.mma.cta_group::2` over a CTA pair:
D[256×16] = A[256×64] · B[16×64]ᵀ with M = 256.

```
both CTAs : warp 0: tcgen05.alloc.cta_group::2 (same warp, same slot); init; cluster barrier
t0        : TMA its 128 A rows and its 8 B rows with .cta_group::2, signaling rank 0's barrier
rank 0 t0 : wait; 4 x tcgen05.mma.cta_group::2 (idesc M = 256); commit .multicast::cluster 0b11
all       : wait done; tcgen05.ld; store; cluster barrier; dealloc.cta_group::2
```

### How the operand split was established

The model should not guess how a 2-SM MMA splits its operands. Order of work:
1. Write the kernel under hypothesis H1 (A and D split by rows; B split by N halves; same
   addresses in both CTAs).
2. Run it on GB300 with known inputs and compare against an fp64 reference: all 4,096 outputs
   matched, deterministic over 100 runs.
3. Only then model it. The PTX model then matched GB300 **bitwise**, numerics included (E-PAIR).

### What the SASS revealed: a hidden handshake

`tcgen05.alloc.cta_group::2` is not one instruction. ptxas emits a **cross-CTA protocol** that
runs over reserved shared memory the kernel never initializes:

```
leader (even rank)                               peer (odd rank)
UTCATOMSWS.2CTA.FIND_AND_SET  (both TMEMs)
TRYWAIT [0x48], parity [0x50]=1   (passes)       TRYWAIT [0x48], parity [0x50]=0   (blocks)
SYNCS.ARRIVE.TRANS64.RED [peer 0x48], tx=4
STAS [peer slot, peer 0x48], column   -------->  (phase 0 completes) LDS slot
ATOMS.XOR [0x50] (flip phase)                    SYNCS.ARRIVE.TRANS64.RED.A1T0 [leader 0x48]
                                                 ATOMS.XOR [0x50]
```

For this to work, the barrier at 0x48 must already be an initialized mbarrier, and the
leader's phase bit must already be 1. Three probes settled it (S-RESERVED):
- `probes/reserved_smem`: reserved smem is zero at kernel entry, **except** in
  `cta_group::2` cubins. There the driver sets `0x50 = 1` on rank 0 (plus two words at 0x0).
  There is no prologue code, so the cubin attribute `EIATTR_TCGEN05_2CTA_USED` is what
  triggers it.
- `probes/mbar_reserved`: the word at 0x48 *reads* as 0 but behaves as a **count-1 mbarrier in
  phase 0** under exactly the handshake's operations.
- `probes/mbar_zero`: a zero word you store yourself is *not* such a barrier (the arrive
  faults). So the state lives in the sync unit, not in the memory word.

The same probes also found: in a cluster launch, a `shared::cta` access whose rank byte is not
the CTA's own **traps** (illegal instruction). The model now reports it as a fault, and the PTX
front-end rank-tags every shared address the way ptxas does.

New SASS forms:
- `UTCATOMSWS.2CTA`, `UTCHMMA.2CTA`, `UTCBAR.2CTA.MULTICAST` (mask unshifted) and
  `UTMALDG.2D.2CTA` (the mbarrier may be in the peer);
- `STAS [Ra.64], Rv` (= `st.async…complete_tx`, Ra+1 holds the mbarrier);
- `SYNCS.ARRIVE.TRANS64.RED` (remote `arrive.expect_tx`; `.A1T0` plain arrive, `.A0TR`
  expect_tx only);
- `UPRMT`, `ATOMS.XOR`;
- `SYNCS.PHASECHK.TRANS64` (non-suspending re-poll). The E-SPIN shape check now accepts the
  `@!P0 NANOSLEEP.SYNCS; @!P0 SYNCS.PHASECHK.TRANS64 P0` retry form.

### Verdicts

| variant | bug | PTX model | SASS model | GB300 (1000 runs, scrubbed) |
|---|---|---|---|---|
| good | | clean, exhaustive (119 states) | clean, exhaustive (431 states, handshake included) | bitwise = both models |
| B12 | multicast commit mask omits the peer (`0b01`) | deadlock: the peer waits for `done` forever | deadlock | **hang** (watchdog) |
