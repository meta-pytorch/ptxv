# Rung 1g: one-shot all-reduce over symmetric memory (3 GPUs, two calls)

**Status: done (2026-10-30), on 3 GPUs instead of 4.** The fourth GPU had a foreign process,
and the GPU policy is to avoid it. The kernel has 3 ranks; going to 4 needs only the case generator.
- The good kernel is explored **exhaustively on the PTX front-end** and is clean (stateful,
  89,085 states).
- The SASS good kernel is also explored **exhaustively** and is clean (stateful, 3,294,901
  states, 7 minutes). Every mutant is exhaustive on both front-ends.
- PTX model, SASS model and three GB300s agree bitwise (1000/1000 runs).
- B21, B22 and B24 get their expected verdicts on both front-ends. **B24 diverges on GB300 in
  every run.**

Multicast / NVLS (`multimem.*`, B23) is not part of this rung.

## The kernel

`fixtures/r1g_allreduce/kernel.ptx`, launched as 6 grids: 3 ranks × 2 back-to-back calls.
`gen_case.py` writes `case.toml`. Call 2 of rank r is stream-ordered after call 1 of rank r
(same GPU); nothing orders the ranks.

```
per call, rank r (one thread in the model; the kernel strides over the 32 elements):
  1. stage    buf_r[i] = in[i]                           (symmetric buffer, own GPU)
  2. barrier  sig_p[r] = s (st.release.sys) for every peer p; wait sig_r[p] >= s
              (ld.acquire.sys); s = 2*epoch, monotonic across calls: no flag is ever reset
  3. reduce   out[i] = ((0 + buf_0[i]) + buf_1[i]) + buf_2[i]   (same order on every rank)
  4. barrier  s = 2*epoch + 1: nobody restages before every peer has finished reading
```

## A new verdict: replicas must agree (N4)

The case lists `replicas = [[out_0_c, out_1_c, out_2_c], ...]`. At every terminal state the
model compares those buffers bytewise; a difference is a `divergent` finding. This is the N4
claim from the README: tensor-parallel replicas must hold the *same bits*, not merely numbers
that are each within tolerance.

## Verdicts

| variant | bug | PTX model | SASS model | 3 × GB300 (1000 runs) |
|---|---|---|---|---|
| good | | clean, exhaustive (89,085 states) | clean, exhaustive (3,294,901 states) | bitwise = models |
| B21 | barrier signals a constant 1 and resets each flag after seeing it | deadlock, exhaustive | deadlock, exhaustive | correct 1000/1000 |
| B22 | no closing barrier | race + nondeterministic + divergent, exhaustive | same | correct 1000/1000 |
| B24 | each rank sums starting from its own buffer | divergent (10–13 of 32 words per pair) | divergent | **ranks differ in every run** (45 of 192 words) |

- **B21's deadlock trace** is the classic lost signal. A fast rank finishes barrier A and
  sends its barrier-B signal (1) into a slow peer's slot. The slow peer has not yet *seen* the
  barrier-A signal in that slot, so when it does, it resets the slot to 0 and wipes the
  barrier-B signal. It then waits at barrier B forever. On hardware the ranks are launched
  within microseconds of each other, so the window never opens.
- **B22** needs a rank to start call 2 while a peer is still reducing call 1. The model
  explores that skew (a new GPU's grid launches at any time, E-PDL); the hardware does not
  produce it unprompted.
- **B24** is the bug no test catches: every rank's result is a correctly rounded sum of the
  same three numbers, only in a different order. The replica check catches it in the model,
  and the GB300 outputs differ on every run.

## Fixes made on the way

- **Constant bank:** `%ntid` is `c[0x0][0x360..0x368]` and `%nctaid` is `c[0x0][0x370..0x378]`
  on sm_103a, read off ptxas output. The old 0x0 / 0xc entries were guesses that no fixture
  had exercised; this kernel's stride loop was the first to read `%ntid` in SASS.
- **Poll loops:** the enabling dry-run now runs exactly one iteration (load to back branch)
  and asks whether the branch went back. Re-reaching the same poll pc is not a failed poll:
  the barrier's peer loop polls one site for each peer.
- **Numerics:** a 64-bit move into two 32-bit registers (`LDC.64 R10`) now clears the rounding
  provenance of R11 too. Before, it raised a false warning.

## State space

Six grids with independent launch times give many traces. Three things helped:
- one thread per rank (the kernel strides, so any block size runs it);
- the stateful explorer, which merges states that DPOR revisits: the PTX good kernel takes
  89k stateful states, while DPOR was unfinished at 2M;
- monotonic signal values.

A one-call variant needs 847 DPOR states on PTX and 1,248 on SASS. The second call multiplies
the cross-rank interleavings. The SASS good kernel was explored further in the background:

The SASS good kernel completed exhaustively at 3,294,901 states (10,648 terminal states, 425 s,
`--max-states 80000000`). The test suite does not run it; it runs the SASS mutants.

## Reproduce

```
ptxv check fixtures/r1g_allreduce                     # stateful; SASS good is bounded
ptxv hw fixtures/r1g_allreduce --runs 1000            # needs three free GPUs
cargo test --release --test fixtures r1g
```
