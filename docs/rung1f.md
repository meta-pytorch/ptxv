# Rung 1f: two GPUs, flag ping-pong over NVLink peer memory

**Status: done (2026-10-30).**
- The model now knows **scopes**: which strong accesses are morally strong, and which
  release/acquire pairs synchronize, at `.cta` / `.cluster` / `.gpu` / `.sys` (S-SCOPE).
- Kernels can run on **several GPUs**, launched independently with arbitrary skew.
- The good kernel and a fence-based variant are explored exhaustively on both front-ends and
  are clean. PTX model, SASS model and two GB300s agree bitwise (1000/1000 runs).
- B19, B20 and B56 are races in the PTX model. B20 is **wrong in every GB300 run**. B19 never
  fails on GB300 (20,000 runs). B56 is clean in SASS, and the doc explains why that is correct.

## The kernel

`fixtures/r1f_pingpong/kernel.ptx`. The same kernel runs on GPU 0 (rank 0) and GPU 1
(rank 1). Each rank owns a data word and a flag word in its own memory and writes the peer's
words through peer pointers (`device = 1` in the case; the hardware runner enables peer
access).

```
round k = 0, 1, 2:
  sender   (rank k % 2): st.global [peer data] = v + 1 ; st.release.sys [peer flag] = k + 1
  receiver           : POLL: ld.acquire.sys r, [my flag] ; setp.lt p, r, k + 1 ; @p bra POLL
                       v = ld.global [my data]
out[rank] = v                                                        -> out = {2, 3}
```

## Verdicts

| variant | change | PTX model | SASS model | 2 × GB300 |
|---|---|---|---|---|
| good | | clean, exhaustive | clean, exhaustive | bitwise = models, 1000/1000 |
| fenced | relaxed poll, then `fence.acq_rel.sys` (the volatile + `__threadfence_system()` pattern) | clean | clean | bitwise = models |
| B19 | flag released with `st.release.gpu` | race: the flag store and the peer's poll are not morally strong, so the payload is unordered too | same (`STG.E.STRONG.GPU` after `MEMBAR.ALL.GPU`) | **correct 20,000/20,000** |
| B20 | relaxed poll, no acquire | race on the payload | same | **wrong 1000/1000**: out = {2, **1**} |
| B56 | relaxed poll, then `fence.acq_rel.gpu` | race: the fence's scope does not include the writer on the other GPU | **clean** | correct 20,000/20,000 |

**B20 fails on hardware, and the reason is visible in the SASS.** Rank 1 reads its data word
in round 0 (value 1), so the line is in its L1. In round 2 it polls the flag with
`LDG.E.STRONG.SYS`, which bypasses L1 and sees the new flag. Then it reads the payload with a
plain `LDG.E` that hits the stale L1 line and returns 1 instead of 3. The acquire in the good
kernel is exactly the `CCTL.IVALL` (L1 invalidate) that ptxas puts after the acquire load. The
model reports the race, not the stale value: under E-DRF a plain read sees the latest write in
the explored interleaving. The race is the contract violation, and the L1 is how this GPU
punishes it.

**B19 is the archetypal latent bug.** The release only orders the payload store at GPU scope,
but NVLink delivers the payload before the flag in practice, so 20,000 runs are correct. The
PTX memory model gives no such guarantee: the flag store and the peer's poll are not morally
strong. The verifier is the only way to see this bug.

**B56 is safe in SASS, and the model shows why.** `fence.acq_rel.gpu` compiles to
`MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; CCTL.IVALL`. Only the release half (`MEMBAR`) carries a
scope. The acquire half is the same L1 invalidate as for `.sys`, and the poll itself is a
`.SYS` strong load. So the binary acquires the peer's release; only the PTX does not promise
it. Same category as B05/B06/B14: a PTX contract violation this ptxas compiles safely
(`expect_sass = ["clean"]`).

## Model pieces (ASSUMPTIONS.md)

**S-SCOPE (rewritten; was "scopes treated as sufficient").**
- Strong accesses: `ld/st.relaxed|acquire|release.<scope>`, `.volatile` = relaxed.sys, and
  `atom`/`red` (default relaxed.gpu; new in the PTX front-end). They carry their scope into the
  shadow memory (`Epoch::strong`). A conflicting pair that is not ordered by happens-before is
  a race unless the two accesses are morally strong.
- Each location's last strong write is a synchronization source (`SyncCell`). A release
  write publishes its clock. A relaxed write publishes the clock of the writer's last release
  fence, at the narrower scope. RMWs continue the release sequence.
- A morally strong read acquires that release when the release's scope includes the reader:
  at once for an acquire read, at the next acquire fence for a relaxed read.
- SASS: `.STRONG.<scope>` loads and stores are relaxed; `MEMBAR.<scope>` is a release fence;
  `CCTL.IVALL` is an acquire fence. The E-SPIN shape checker treats `CCTL.IVALL` as a no-op
  for control purposes, as before.

**E-SPIN extended to flag polls.** A strong load is now a scheduling point, so a poll loop
would otherwise run one iteration per step without end. `spin::poll_loops` recognizes
`H: ...; r = ld.strong [a]; ...; @p bra H` loops: register-only body, no register carried
across iterations. The model runs a thread there only when the iteration exits
(`split_enabled` tries the step on a copy), and DPOR treats a blocked poll as reading its
location. Other loops around strong loads get an `unmodeled` finding.

**Multi-GPU launches (E-PDL extended).** `device = N` on a launch puts its grid on another
GPU. Stream order, and PDL, apply only to the previous grid on the same GPU. A grid on a new GPU
starts unlaunched with an always-open gate, so the explorer chooses when it launches: launch
skew between GPUs is explored, not assumed away.

## Infrastructure

- Case format: buffers have a `device` (the GPU whose memory holds them); launches have a
  `device`. Variants can carry several edits (`edits = [{marker, find, replace}]`), used here
  to switch the poll to relaxed *and* insert a fence.
- `ptxv hw` / `probe`: a case on N GPUs gets N free GPUs (`gpu_env`). The runner creates one
  context per GPU, enables peer access between all pairs, allocates each buffer on its GPU,
  launches each kernel on its GPU, and synchronizes all of them.
- New lowerings: PTX `atom`/`red` (add/or/and/xor/exch, 32/64-bit, sem + scope), scoped
  `fence.acq_rel`/`fence.sc`; SASS `ATOMG`/`REDG`, `LDG/STG.STRONG.*`, scoped `MEMBAR`,
  `CCTL.IVALL` as an acquire fence.

## Reproduce

```
ptxv check fixtures/r1f_pingpong
ptxv hw fixtures/r1f_pingpong --runs 1000        # needs two free GPUs
ptxv probe fixtures/r1f_pingpong --variant B20   # out = [2, 1]
cargo test --release --test fixtures r1f
```
