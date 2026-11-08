# Grid cost: checking large launches without simulating every thread

Status: design (2026-11-07); reduction 3 implemented, 1 and 2 not yet. Owner: SC.

## Problem

A pilot over 200 corpus kernels at their native launch shapes: 66 timeouts or OOMs, and 65 of them happen inside the certificate's single
deterministic run, before any exploration. The run interprets every thread of the grid:

- Memory is about 20 KB per simulated thread (registers, race shadow, sync log), so 16M-thread
  launches stop at load (`oom` after 8-20 s).
- Time follows the dynamic thread-instruction count. Small grids time out too: one corpus kernel (dash/5042 k0) is
  one CTA of 1024 threads that reads 77 KB per thread, and dash/3153 k1 (64 CTAs, 16 KB per
  thread) does not certify one 128-thread slice in 300 s. These are grid-stride or
  persistent loops whose trip count grows with the data, not with the grid.
- Profile of the certificate run on dash/5042 (perf, one thread): about 20% atomic reference
  counts (copy-on-write `Arc` checks that a single-path run never needs), about 15% race shadow
  (`Readers::insert_read`, `access_x`), about 10% sync-log events per lane (`pre_inst_log`,
  vector clocks), and the remainder the per-lane interpreter. No single hot spot: the cost is
  per thread-instruction.

So the levers are (1) not interpreting CTAs whose behavior is implied by others, (2) not
iterating loops whose iterations are implied by earlier ones, and (3) a cheaper per-step cost.
Only (1) and (2) change the asymptotics.

## Reduction 1: representative CTAs (E-CTA-REP)

**Claim.** A CTA's execution is a function of its position (`%ctaid`, cluster rank), the launch
parameters, `%tid`, and the bytes it loads. Run one CTA c0 with a taint shadow whose sources are
(a) the position and (b) every byte loaded from global memory or written by another CTA. If on
c0's path:

1. no tainted value reaches a branch condition, a guard, an `EXIT` predicate, a barrier or
   mbarrier operand (address, count, phase), a shared-memory or TMEM address, an elect / vote /
   shuffle control operand, or a loop bound; and
2. every global address and TMA coordinate is affine in the position with untainted
   coefficients, `addr = a + b·ctaid.x + c·ctaid.y + d·ctaid.z`, where `a..d` are untainted
   (an affine-in-position extension of the position taint: add, subtract, multiply or shift
   by an untainted constant, `IMAD` with one untainted factor; anything else on a
   position-tainted value makes it non-affine, and a non-affine global address refuses),

then every CTA c follows c0's path. By induction along the path, every value that reaches
control, synchronization or a shared/TMEM address is untainted, so it is equal in c and c0.
Every global access of c is c0's access shifted by `Δ(c) = (b, c, d)·(c - c0)`. The
certificate's per-CTA conditions (C-PHASE, C-WAIT, C-ORDER on shared memory, C-WARP) depend only
on the path and the shared/TMEM addresses, so they hold for c when they hold for c0.

**Cross-CTA conditions.** The footprint of CTA c is F(c) = F(c0) + Δ(c) per access, separately for
reads and writes. These must hold for every pair of CTAs:

- no write range of F(c) overlaps a read or write range of F(c') for c ≠ c'. This is checked
  exactly as a 1-D (or 3-D) lattice problem. Writes of c0 occupy ranges `[lo_i, hi_i)` with
  stride vector s. The pairs (i, j, k) with `lo_i + k·s < hi_j` and `lo_j < hi_i + k·s` for some
  nonzero lattice offset k inside the grid are found by interval arithmetic on k, bounded by
  the span of F(c0) over |s|;
- all of F(c) lies inside its allocation for every c. Affine addresses are monotone in each
  coordinate, so checking the grid's corner CTAs suffices;
- no global strong access, atomic or poll across CTAs (those are cross-CTA synchronization: the
  reduction refuses and the whole grid runs as today).

**Outputs.** Only c0's outputs are simulated, so the bitwise comparison with the GPU covers
c0's footprint. The verdict covers every CTA. A `--rep-outputs k` option simulates k evenly
spaced CTAs to compare more output.

**Regimes for position-dependent control (E-CTA-REP regimes).** Tail CTAs and grid-stride
loops make control depend on the position: Triton's `mask = offs < n` or
`for i in range(pid, n, nprog)`. Condition 1 is relaxed to "every tainted control value is a
comparison of an affine-in-position value with an untainted bound". Each such decision holds
for an interval of CTAs (per coordinate), and the CTAs whose decision sequence equals c0's form
the intersection R(c0) of those intervals. Repeat with the first CTA outside the covered set
until every CTA is covered. A tail mask gives 2 regimes; a grid-stride loop gives one regime per
distinct trip count, usually 2. The cross-CTA checks above run over the union of the regimes'
footprints, each with its own Δ.

**Refusals** (the launch runs whole, as today, and the verdict says why): a non-affine global
address; a tainted value in control that is not an affine comparison; more than R regimes
(default 16); global atomics or strong accesses; DSMEM or clusters with more than one CTA (a
first version handles cluster size 1; clusters extend by treating the cluster as the unit);
CLC (cluster launch control: the CTA-to-work mapping is dynamic); launch groups whose later
kernels read what the earlier ones wrote (left to E-LAUNCH-GROUPS, unchanged).

**Cost.** One CTA run per regime with taint, plus the footprint arithmetic. A 16M-thread grid of
65536 identical CTAs becomes one 256-thread run.

**Relation to existing code.** `explore_symmetric` already runs the whole launch once with
position taint and compares per-thread position traces across clusters, then explores one
cluster. E-CTA-REP removes the whole-launch run. It needs (a) data taint in addition to position
taint (num.rs taint shadow, extended with an affine coefficient vector per value), (b) the
lattice disjointness check (new, `rep.rs`), and (c) a certificate entry point that runs a
representative slice machine (`wellsync::slice_machine`) per regime.

**Soundness tests.** A fixture with three mutants, each touching only CTAs other than c0:
- a tail CTA whose mask is off by one, so it writes past the end: the regime split must run it;
- a CTA whose address uses `ctaid*ctaid`: refused as non-affine;
- two CTAs whose footprints overlap through a stride smaller than the tile: the lattice check
  must find it.
The good kernel certifies with the whole-grid verdict and outputs on c0's footprint.

## Reduction 2: loop summaries (E-LOOP-AFFINE), after 1

For the per-thread long loops (dash/5042: 1 CTA, 805 iterations per thread), the same affine
domain with an iteration symbol. A loop whose body:
- has no synchronization other than CTA barriers, with the same count every iteration;
- has untainted control except the loop's own `i < n` exit, which is affine in i;
- has global addresses affine in i with untainted coefficients, and shared/TMEM addresses
  untainted or affine in i,

is run for iterations 0 and 1. The induction step: every quantity that reaches control or an
address in iteration k+1 is the same affine function of k+1 as in k, because the body's
register-to-register transfer on those quantities is the same affine map each iteration. That
map is checked symbolically on the body once, not by observation. The loop's footprint is the
union over k < n of F(0) + k·s, which goes to the same lattice check, per thread and across
threads. The data values are not computed for the skipped iterations, so output comparison
covers only the iterations that ran. This changes the verdict's coverage statement (a note),
not its soundness.

Refuses: any data or carried value that reaches control (a while-loop on data, a reduction
whose result decides a branch), a non-affine address, synchronization whose count varies.

## Reduction 3: per-step cost (engineering, independent)

Implemented first (semantics-preserving: no new assumption, nothing refused). Profiles of the
certificate run (perf, one thread) on finish_block_lanes, quack rmsnorm, varlen philox and
dash/5042 named these per-lane costs, each removed without changing a result:

- `Readers::insert_read` scanned every reader of a byte for the reader's own entries; the list
  is sorted by actor, so a bisection finds them (read-shared bytes of 1024-thread CTAs hold
  about 1000 readers).
- The race-shadow loop of `access_x` looked up the same vector-clock entry once per byte;
  `VcMemo` keeps the answers for the access (the clock does not change during it).
- A barrier release joined each arrival's clock into the accumulated one by a full merge;
  `vc_join_many` merges each distinct base once (arrivals after an earlier barrier share one)
  and inserts the own entries together. Whether the barrier is complete is counted up to the
  first live thread past the arrivals instead of over the whole CTA.
- `Op::src_regs` allocated per lane; `src_regs_at` keeps the last instruction's list (the
  lanes of a warp run one instruction in turn). `wr` and the LOP3 undefined-bit rule no longer
  allocate per call; `pre_inst_log` no longer clones a clock per lane.

Not done: a copy-on-write-free single-path mode (the `Arc` reference counts are about 15% of
the remaining time; avoiding them needs `unsafe` or a separate machine type) and a warp-vector
interpreter. `PTXV_CERT_PROGRESS=N` prints the certificate run's step count every N steps,
which measures runs that do not finish.

## Measurement plan

The 52 native-shape cases from the pilot's list, at their native grids, on the cluster
(32 CPUs, 128 GB, 600 s as in the pilot):
- before: the pilot's numbers (all timeout or OOM);
- after reduction 1: how many certify, time and RSS. Then classify the refusals (non-affine,
  data-dependent control, atomics), which decides how much reduction 2 is worth.
Expected from the shapes: the 16M-thread elementwise or norm kernels (affine, tail masks) fall
under reduction 1. The data-loop kernels (dash/5042, 3153, 2056, 1093) need reduction 2.
