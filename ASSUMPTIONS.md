# ASSUMPTIONS: everything a ptxv verdict rests on

A ptxv verdict ("clean", "race", ...) is a theorem *relative to* the assumptions below. This
file is the single registry of them. Rules:

* Every assumption has an ID. The code that relies on it carries an `ASSUME(ID)` comment, and
  every verdict lists the IDs it depends on (`assumes:` in `ptxv check` output and in
  `summary.md`). `cargo test` fails if this file, the registry in
  `crates/sim/src/assumptions.rs`, and the code tags disagree.
* **Status** is one of:
  - **enforced**: checked at runtime; violating inputs give `unmodeled`, not a silent answer;
  - **validated**: tested against GB300 or a cross-check, with the evidence named;
  - **argued**: justified from documentation or reasoning, but not tested;
  - **refuted**: tested against GB300 and false; kept (opt-in) only so that old verdicts can
    be reproduced, and listed in their `assumes:`;
  - **unvalidated**: chosen for convenience; may be wrong.
* **If wrong** says which way the verdict fails:
  - a *miss* means we could call a buggy kernel clean (unsound);
  - a *false alarm* means we could report a bug that cannot happen (incomplete).
* Features the model does not cover are not assumptions. They fail loudly: an unknown SASS
  opcode is a hard error, and an unmodeled operation or value gives a finding of kind
  `unmodeled`, which makes the verdict inconclusive.

Kinds: `code` assumptions are tagged in the source; `process` assumptions are about how we test
and are not tagged.

---

## E: execution model (how schedules are explored)

### E-SPIN · code
**A failed `mbarrier.try_wait` blocks the thread until the mbarrier changes** (instead of
returning false and letting the thread do something else).
- Where: `exec.rs` `try_wait_result`, `split_enabled`; checked by `spin.rs`.
- Status: **enforced** (2026-10-30). `spin.rs` proves, for every try_wait on both front-ends,
  that blocking is exact:
  - the instructions before the retry branch are register-only, do not read the result, and
    keep the poll operands;
  - the failure path only re-polls the same mbarrier with the same operands (plus no-ops and
    branches);
  - every successful re-poll rejoins the continuation.

  Any other shape gives an `unmodeled` finding (fixture r0 B49). All fixture binaries pass.

  **Compare-and-swap update loops** (2026-11-06, ptxas's loop for `atom.add.noftz.bf16`:
  `H: new = f(old); got = ATOMG.E.CAS [a], old, new; P = got != old; old = got; @P BRA H`) are
  not polls and are not blocked: the model runs every iteration as a step. That terminates in
  every schedule, since a failed attempt reads a value another thread wrote after this
  thread's previous read of the location, and the writes are finite (`spin::cas_update_loop`:
  register-only body, the compare operand the only register carried, refreshed from the CAS
  result before the branch). probes/bf16x2 (two such loops) = GB300 bitwise.

  **Loops whose exits do not test a strong load** (2026-11-06): a strong load inside a loop
  makes it a poll only if the loop's exit depends on the loaded value. With a conditional back
  branch the branch's predicate is tested (DeepEP's `ld.volatile` copy loops); with an
  unconditional back branch, every exit of the loop (`EXIT`, a branch out of it) is, by a
  forward register-taint dataflow over the loop's control flow and back edge to a fixpoint
  (`spin::loop_exit_tainted`; an unguarded write from untainted operands clears a register,
  a guarded one changes nothing). A loop none of
  whose exits depends on the value terminates without other threads' writes, so the model
  runs its iterations as steps (Triton's pipelined main loop in
  `_kernel_dispatch_and_calc_grads` reads a `volatile` word each iteration and stores what it
  read; it was reported `unmodeled`). Fixture e_vol_loop: `good` explores clean, `SPIN` (exit
  on the loaded value) and `CARRY` (exit on a counter that adds the loaded value each lap)
  keep the `unmodeled` finding (tests/reductions.rs). Memory flow is
  not followed (a loaded value stored and re-read), as in the conditional case.

  **Warp-level poll loops** (rung 4b, DeepEP `barrier_block` and channel waits): a loop whose
  body branches forward, reconverges (`BSYNC`), checks convergence (`BRA.DIV`), votes and
  loads, but writes no memory, with a strong load inside and a back branch to its head
  (`spin::wpoll_loops`). The split at the head runs one iteration on a copy
  (`spin::wpoll_dry`): if every member comes back to the head with unchanged registers and
  predicates, the iteration is a stutter step and the split blocks; its loads are recorded
  for the race detector and as its DPOR footprint (`blocked_poll_read` analogue). The first
  iteration, which changes registers, runs for real. Clock timeouts inside need E-CLOCK.

  **Recomputing retry stubs** (rung 2): ptxas may recompute the parity into a free register
  at the head of the out-of-line retry stub (`R4 = UR6 << 31; w2: TRYWAIT [a], R4`). This is
  accepted when the stub prefix is straight-line, unguarded and register-only, `w2` polls the
  same mbarrier with the same result predicate, and every register the prefix writes is dead
  at the continuation (`spin::live_after`, predicate-aware). Whether `w2` polls the same phase
  is checked whenever the original poll succeeds: the stub runs on a copy of the thread
  (`check_stub_parity`); a different phase gives `unmodeled`.
  The stub may also poll through a register computed between the wait and its retry branch
  (FA4: `TRYWAIT [UR10+0xc0]; ...; UR6 = UR10+0xc0; @!P0 BRA stub`, stub `R2 = UR6; TRYWAIT
  [R2+URZ]`): that segment is pure (`spin::retry_branch`), so `check_stub_parity` runs it
  first, with the poll's predicate false as on the retry path, then the stub, and compares
  the mbarrier address and parity as before.

  **Peeks** (rung 2, CuTeDSL): CUTLASS tests an mbarrier early (`try_wait` into a predicate),
  does other work, and later skips or runs a blocking wait loop depending on the result. Such
  a try_wait does not have the retry shape, but it is now run as what it is: a non-blocking
  test that returns false while the phase is pending (`Machine::peeks`). It qualifies when,
  assuming it and every other peek fail, every path from it back to itself passes a
  retry-shaped (blocking) try_wait (`spin::is_peek`, which tracks the result predicate). A
  success is progress, so a peek cannot spin without the mbarrier moving. A path back to the
  test that passes no blocking wait also qualifies when none of its branches and side effects
  depends on a failed result (forward dataflow from every assumed-failed result, guards
  included; a store or other side effect reading one counts as a dependence): the same path is
  taken when the test succeeds, so that loop is the kernel's own, not a wait (2026-11-07;
  FA bwd Sm100, BlackwellFusedTopkSelection: a test before a tile loop whose empty case skips
  the result; fixtures/e_peek_loop good, and SPIN, whose loop repeats while the test fails,
  stays `unmodeled`). Other try_waits without the retry shape are still blocked on and
  reported `unmodeled` (fixture r0 B49).
  (rung 3b) The walk also follows the result through a token register: trtllm-gen keeps it
  as `SEL R4, RZ, 0x1, !P1` and later decides with `ISETP.NE.AND P1, PT, R4, 0x1` whether to
  wait (known predicates and registers from SEL with constant operands and integer EQ/NE
  compares). A try_wait whose result goes to `PT` (trtllm-gen polls the next stage ahead of
  time) is lowered as a no-op: it does not block, writes nothing, and dropping its acquire only
  removes happens-before edges (it may add a race report, never hide one).

  **Poll loops on strong loads** (rung 1f, `spin::poll_loops`): a loop
  `H: ...; r = ld.strong [a]; ...; @p bra H` whose body is otherwise register-only (plus
  acquire fences) and carries no register from one iteration to the next is run only when the
  iteration exits: `split_enabled` tries the step on a copy. A failed iteration changes nothing
  a later one does not overwrite, so this is exact. A strong load inside a backward branch's
  range with any other shape gets an `unmodeled` finding.

  **Poll loops on plain loads** (2026-11-02, fixture `e_spin_smem`): ptxas compiles
  `ld.acquire.cta` / `ld.relaxed.cta` / `ld.volatile` of shared memory to a plain `LDS`, so a
  shared-flag spin is a plain-load loop in SASS (plain `ld` spins in PTX likewise). Plain
  loads are not scheduling points (E-DRF), so such a loop used to run inside one transition
  until the 10M-instruction local-step limit, never seeing the other warp's store (a `fault`
  verdict after minutes; a racy poll also pushed one race finding per iteration, which grew
  memory without bound). A plain load with exactly the poll-loop shape above is now blocked
  the same way: the step that reaches it stops there while the iteration would fail. Plain
  loads in loops of any other shape (reductions, carried counters) are not polls and are not
  listed or reported. A failed iteration only re-reads the flag, so skipping it is exact
  for registers and control; for the race detector, a split that stops at a poll that would
  fail (strong or plain) reads the flag once, without effect on registers and without
  acquiring (`spin::blocked_poll_read`). The polling thread synchronizes with nobody while it
  spins, so every later write of the flag by another thread is checked against that read as
  it would be against the skipped iterations: a race unless both are morally strong. A poll
  that never succeeds is a reported deadlock. (Not acquiring from a failed read's writer can
  only add races.) On SASS, LDS/STS are relaxed strong at CTA scope (S-SMEM-STRONG), so a
  shared-flag handoff that ptxas kept in order is clean, as on GB300.

  **Compare-and-swap polls** (2026-11-03, rung 4a, torch symmetric memory): the signal
  `while (cas(&pad[i], 0, 1) != 0);` / `while (cas(&pad[i], 1, 0) != 1);` compiles to
  `H: R6 = 0; YIELD; [MEMBAR.ALL.SYS]; CAS R0, [R2], R6, R7; [CCTL.IVALL]; ISETP; @P0 BRA H`.
  It is a poll loop of the shape above with the CAS as the poll operation (`AtomOp::Cas`; a
  failed compare is a read only, so a failed iteration changes nothing), a release fence
  allowed in the body (repeating it after attempts that wrote nothing publishes nothing new),
  and loop-invariant prefix writes (`R6 = 0`, recomputed identically) not counted as carried.
  The exiting iteration's CAS swaps and writes. Blocked CAS polls read the flag for the race
  detector and the explorers' footprints like a blocked load poll. Validated on GB300 by
  `fixtures/s_cas` (bitwise; its lost-signal mutant hangs, as the model's deadlock says).
  The certificate (E-WELLSYNC) covers CAS signals with C-CAS (below): a put and its wait
  are not HB-ordered, but the values force their order (2026-11-03).

  **Retries that loop back to the poll's own prefix** (2026-11-02, rung 3a CUTLASS grouped
  GEMM): `H: R3 = R2 << 31; R0 = UR5; w: try_wait [R0], R3; @!P0 nanosleep; @!P0 re-poll;
  @P0 EXIT; BRA H` (an `@d EXIT` decides: the exit is the continuation) and `H: R2 = UR7; w:
  try_wait [R2+0x40], R73; ...; @!P0 BRA H`. The retry path (after unconditional branches)
  runs register-only straight-line code back into `w`, rewriting poll operands. Accepted
  with a run-time check: on every successful poll the executor reruns `H..w` on a copy and
  requires the same mbarrier, the same parity, and every register unchanged (the prefix is
  what the thread ran before its first poll), so skipping the retry is exact; otherwise
  `unmodeled` (`spin-parity`). Tests: `crates/sim/tests/spin_shapes.rs`.
  **Peeks in the explorer (`explore.rs` `peekwait_set`, `peekskip_set`; `PTXV_NO_PEEKWAIT`
  turns both off, `PTXV_NO_PEEKSKIP` the second).** A peek races with the arrival that
  completes the phase it reads, so the stateful explorer branched on every such order,
  though CUTLASS's peeks only decide whether a later blocking wait runs. Two model changes
  make a peek's outcome invisible where that is provably all it decides. They apply only in
  the explorer (`explore_one`); the certificate (E-WELLSYNC) keeps the plain model.
  - *Peek as one blocking wait (`peekwait`).* The shape is `d = try_wait [A], P`, then
    register-only code, then `@d bra J`. The failure side has register-only code (not
    touching the poll's operands), then a try_wait with the same operands and the retry
    shape, then `@!d bra` to its retry, falling through to J. Every register the failure side
    wrote (other than `d`) is dead at J. The peek is then run as that blocking wait: it is
    disabled while the phase is pending, as a failed try_wait is under the spin model. A
    failed peek followed, with no effect, by a wait that blocks until the phase completes
    and joins J with the same live registers is exactly that. r3: 7 of the epilogue's 17
    peeks (`d040`/`d060` and its siblings).
  - *Peek as a failed test (`peekskip`).* For CUTLASS's "test now, work, then wait": the
    peek's result decides, maybe through derived predicates (PLOP3, ISETP's combining
    predicate, VOTE) and across loop iterations, whether a later blocking wait on the same
    mbarrier runs. Two worlds are walked in lockstep from the peek, `d` true in one and
    false in the other:
    - instructions reading nothing that differs run alike in both, whatever their effect;
    - a branch on a differing value must agree, or the failure world goes alone through
      register-only code, blocking try_waits and other peeks to where the success world
      went;
    - registers still differing there must be dead.

    Then every behaviour of the success world is one of the failure world (it waits where
    the other skipped, and the wait passes at once), so the peek is run as a test that
    fails. Two things are checked at run time, else the verdict gets `unmodeled`:
    - *Same wait:* the skipped wait is on the peek's mbarrier and parity
      (`peekskip-target`).
    - *No ABA:* the mbarrier completes no phase while the window from the peek to the
      worlds' join is open and the test would pass (`peekskip-aba`: then the success world
      went on while the failure world waits a phase longer). A failed test reads no phase,
      so it no longer depends on the completion it races with. Its footprint reads only
      the key of that ABA completion, so the explorer runs both orders of exactly that
      completion, and the check sees it.

    A peek whose lanes disagree on its guard runs as before, and so does one whose result
    skips no wait (its success also acquires, which a failed test would not). r3: 10 more peeks (producer
    `3df0`/`4060`/`4140`/`42f0`, MMA `82e0`/`85c0`, scheduler `93b0`, epilogue
    `b740`/`b7a0`/`b7c0`).
  - Racy kernels: a search that finds a race is run again without the peek changes (a note
    says so). Under E-DRF the reductions do not keep the outcomes of racy kernels, and the
    peek changes removed branching that happened to expose some of them (r2_cute_gemm CM1:
    `nondeterministic` disappeared).
  - Soundness check: `PTXV_CHECK_WALKS` runs its random executions in the changed model
    (every end state must be one the explorer reached), then again in the plain model,
    comparing terminal outputs and findings (end states differ in dead registers and in
    where a deadlocked thread waits).

### E-ALLOC · code
**A `tcgen05.alloc` that cannot succeed blocks until columns are freed.** ptxas actually emits
a `UTCATOMSWS` + `NANOSLEEP` retry loop. Who else holds columns: E-TMEM-COTENANT (opt-in).
- Where: `exec.rs` `split_enabled` (TMEM alloc blocking).
- Status: **argued** (same structure as E-SPIN; the retry loop is visible in the SASS).
- If wrong: *miss*, as for E-SPIN.
- Retire: the same static retry-loop check, applied to the alloc loop.

### E-DRF · code
**DRF-SC reduction:** plain loads/stores and TMEM accesses are not scheduling points. Only
synchronization operations interleave; races among plain accesses are found with vector clocks.
- Where: `ir` `Op::is_visible`, `explore.rs`.
- Status: **argued.** This is the standard theorem for the core scoped PTX model. Its extension
  to the async, tcgen05 and tensormap proxies is assumed (docs/design.md §9).
- If wrong: *miss*, for a racy kernel whose bad outcome needs an interleaving of plain
  accesses. The race itself is still reported, so a clean verdict stays correct as long as the
  theorem holds for race-free programs.
- Retire: a litmus campaign for the proxies (docs/design.md §7.2).

### E-CTA-REP · code
**A CTA whose position-dependent control values equal a representative's behaves as the
representative does, up to its global addresses** (`ptxv check --certify --rep`;
docs/grid-cost.md reduction 1; 2026-11-07). A CTA's execution is a function of its position
(`%ctaid`), of values equal in every CTA (parameters, `%tid`, constant banks, E-CLOCK's clock)
and of the bytes it loads. For a representative CTA r three runs of r's slice decide:
- a data run (num.rs taint with every global byte a source, `Taint::all_global`): no guard,
  control operand or address depends on loaded data; a comparison excused by N-FMAX or
  N-FINITE refuses;
- a position run (num.rs taint, `%ctaid` the source, as E-CTA-INDEP's symmetry check): every
  position-dependent guard, control operand (`Taint::trace`'s points) and global address
  operand (`pos_free_regs`, and a cp.async source) is recorded with the dataflow graph of the
  per-thread register instructions that computed it from `%ctaid`, every other operand a
  constant;
- the certificate (E-WELLSYNC) on r's slice; a representative that stands for no other CTA is
  explored instead when it does not certify.
For every other CTA c the graph is replayed on c's corresponding threads by the instructions
themselves (`Machine::exec_pure`, so `%ctaid` reads c's position). If every recorded control
value equals r's, c takes r's path: by induction over the path, every other control value is
position- and data-independent, hence equal. So c has r's synchronization, shared-memory
addresses and global accesses, at its replayed addresses. c's verdict is r's when its accesses
also overlap each other exactly as r's do (the same pairs: the same races and orders under the
same happens-before) and lie in their allocations. An address outside every allocation is a
`fault` finding of c. Across CTAs, a byte one CTA writes and another accesses is a `race`. This
is more conservative than the explorer, which keeps S-SAME-VALUE writes as notes. A CTA whose
control values differ joins a later regime with its own representative (at most `PTXV_REP_MAX`,
default 16).
- Refusals: the launch is then checked as before, with the cause in a note
  `E-CTA-REP refused: ...`. Causes: a multi-kernel launch; clusters of more than one CTA;
  multicast, NIC or multi-GPU memory; data-dependent control or address; a global atomic or
  strong access; a position-dependent value through memory or a cross-lane operation (vote,
  match, redux, R2UR); TMA coordinates that depend on the position; a global access completing
  after its instruction; S-SAME-VALUE in a representative; a non-certified representative that
  stands for other CTAs; more than `PTXV_REP_MAX` regimes.
- Outputs: only representatives run, so the launch's outputs are not assembled or compared.
- Where: `rep.rs` (hooks in exec.rs `exec_traced` and `access_x`); CLI `--rep` (PTXV_REP=1).
- Status: **argued**. It relies on the replay reproducing c's values: every instruction of the
  graph is a per-thread register instruction (num.rs `pure_op`, cross-lane ones excluded) whose
  sources and old destinations are set before it runs, and a recorded node whose register
  changed by a write the log did not see is not replayable.
- Evidence: fixtures/e_cta_rep. good: 2 representatives for 8 CTAs (CTAs 0-6, the tail CTA 7),
  clean. TAIL1 (the tail stores past the end): fault. AND and STRIDE (CTAs write each other's
  elements): race. N1024 (CTA 7 is never simulated and stores past the end: its replayed
  address): fault. DATACTL (a branch on loaded data): refused, decided by the fallback. Each
  mutant has the full certificate's classes on both front-ends (tests/reductions.rs).
  Measured: triton `_finish_block_lanes` (5000 CTAs) and quack rmsnorm 4096x8192 (4096 CTAs)
  each certify with one representative.

### E-RACE-DEAD · code
**A race whose racing loads' values reach nothing has no effect on the outcome** (class
`race-dead`: kept as a finding with its GB300 label, no verdict class; 2026-11-07). A
read-write race is a plain or relaxed load by a thread that happens-before does not order with
a write; the load may return either value. The check is run again with every racing load
(thread, pc) of the race leaving its destination registers undefined (`Machine::dead_probe`),
and S-INIT reports any undefined bit that reaches a store, an address, a predicate or branch,
an atomic or synchronization operand (so also every output). If the rerun reports no finding
the first run did not, and ends as it did, the loaded value reaches nothing on any explored
path, so a load of the other value leaves the same events; otherwise the race stays `race`.
Applied only when the races are the check's only failing findings (no violation, fault,
deadlock or unmodeled finding; one output for the explorer): a run that already reports
undefined values cannot show that the racing loads add none (a use of a never-written
location has the same signature as a use of a probed load; fixtures e_elect B46,
e_stasync SA2), and the rerun must report no failing finding at all.
- Which loads: `Machine::race_log` records every racing read of every explored transition
  (exec.rs, before the per-instruction deduplication of findings), shared by every copy of the
  machine; the decision needs a complete exploration (or the certificate's run, which covers
  every schedule). A race with another kind of read side (write-write, an atomic, an
  acquiring load, which synchronizes with what it reads, an asynchronous unit) stays `race`.
- The racing reads come from the search whose verdict is returned. When a race makes
  the explorer search again (without the value-insensitive polls or the peek changes), the
  race log is cleared first (2026-11-07). The first search's value-insensitive acquire polls
  acquire nothing, so they logged reads that the returned search orders. That made
  fixtures/l1_litmus neg-payld a race in exploration but race-dead in the certificate.
- Where: `race_dead.rs`; explorer `explore.rs` `explore_one` (a second exploration of the same
  model with the loads undefined); certificate `wellsync.rs` `run_logged` / `probe_run`
  (with C-ORDER's dead reads). DPOR (`--dpor`) does not apply it.
- Status: **argued** from the S-INIT semantics: a value that never reaches a sink cannot change
  memory, control flow or synchronization, so both values of the racing load give the same
  execution. Relies on S-INIT's sink coverage (its listed limits apply).
- Evidence: fixtures/e_race_dead: good (threads other than 0 load a word thread 0 then
  overwrites; warp 1 never uses it) is `race-dead` and clean on both front-ends, explored and
  certified; RD1 (warp 1 stores the value) stays `race` (tests/reductions.rs). Triton kernels
  that `tl.load` a scalar in every thread and store it from one lane:
  `_kernel_map_routing_global`, `_local_stats_partial_reduction`.
- If wrong: *miss*, for a use of the loaded value S-INIT does not report.

### E-ELECT · code
**Which lane `elect.sync` / `ELECT` picks.** The PTX ISA says *"elects one predicated active
leader … deterministically, i.e. the same leader … for the same membermask"*, without saying
which lane.
- Where: `explore.rs` `actions` (PTX), `exec.rs` `Op::Elect`.
- Status:
  - **PTX front-end (rung 2): the leader is a fixed function of the active set**, as the
    "deterministically" clause says, and the explorer runs the kernel twice: once with the
    lowest active lane as every leader and once with the highest (`per_leader_policy`), then
    merges the verdicts. Until rung 2 every active lane was explored at every elect. That is
    exact under no reading of the ISA (it lets one warp pick different leaders for the same
    mask) and costs 32^(elects) schedules, so Triton's GEMM (a dozen elects per warp) could
    not finish. `PTXV_ELECT=all` restores it for small kernels.
  - Covered: code that assumes the leader is lane 0 (B46: the highest-lane run reports it).
    Not covered: code correct for the two end lanes and wrong for some middle lane. We know
    of no idiom like that.
  - **SASS front-end: validated.** It uses the lowest active lane, which GB300 picked for all
    8 active sets in `probes/elect` (2026-10-30).
  - A PTX membermask held in a register (DeepEP: `mov.b32 %r, -1`) is read when executed:
    every executing lane must hold 0xffffffff, else the record is refused (unmodeled). An
    immediate other than -1 is refused at lift. ptxas 13.0 drops the register (`ELECT P, UR,
    PT` even for 0xffff), so the SASS front sees a full-warp elect. probes/elect_regmask:
    GB300 20/20, both fronts bitwise (2026-11-08).
- Evidence: fixture e_elect B46 (code that assumes the leader is lane 0) is a race on PTX,
  clean on SASS, and correct on GB300.
- If wrong (SASS): *miss*, if a future chip elects differently. Re-run `ptxv probe probes/elect`
  on new hardware.

### E-CTA-INDEP · code
**Clusters (CTAs, without clusters) that share no synchronization object can be explored one
at a time.** When the kernel has no global atomics or strong global accesses, no executable
CLC (E-CLC eager launch) and a single grid, the stateful explorer only expands the lowest
cluster that can move (`explore.rs` `ctas_independent`, `cta_ample`). Cluster-scope
operations (DSMEM, remote arrivals, multicast TMA / commits, `cta_group::2`) only name CTAs of
the issuer's cluster; `griddepcontrol` is a no-op in a single grid.
- **Slices** (`ptxv check --slices`, equiv `slices = true`; `explore_sliced`): each cluster is
  explored in a machine where every other CTA counts as exited, and afterwards no global byte
  one slice wrote may be read or written by another (union over every explored transition,
  `GFoot`); otherwise a cross-cluster `race` is reported. With that, the slices' schedules
  interleave freely, so each terminal state, deadlock and race of the launch is one of a
  slice or a cross-slice conflict. Cost: per slice, not per launch (r2_cublas_gemm: 32
  clusters of 4 CTAs). Slices of warps across CTAs and GPUs, with a check over every
  footprint space: E-SLICE.
- **Symmetry** (`explore_symmetric`, what `--slices` tries first): one deterministic run of the
  whole launch with the taint shadow in *position mode* (source `%ctaid`; global addresses and
  TMA coordinates may depend on it). Every guard outcome and control operand value (shared /
  TMEM / mbarrier addresses, masks, counts) that depends on the position is recorded per
  thread; corresponding threads (CTA rank, thread index) of every cluster must have recorded
  the same sequence, and no global byte one actor writes may be accessed by another. Then
  cluster 0 is explored alone and stands for all. Argument: by induction over any schedule,
  cluster c and cluster 0 take the same steps with the same control values and the same
  shared/TMEM addresses (position-dependent values enter control only as recorded, and are
  determined by the thread's own path, or by shared memory written race-free exactly as in
  cluster 0); their global accesses never conflict with anyone's, so global reads return
  host data; so cluster 0's races, deadlocks and determinism are every cluster's. Not
  symmetric (any of the above fails): falls back to all slices, and says why.
  - Evidence: `fixtures/e_sym` (each bug only in CTA 3: refused and found; good: accepted);
    r2_cublas_gemm symmetric (32 clusters); r2_triton_gemm not (persistent tile loop by
    pid); r2_cute_gemm symmetric.
  - Lost relative to all slices: kernels whose control depends on position (edge tiles,
    persistent schedulers, CLC that runs) get no speedup; overlapping global writes between
    threads are refused even when ordered.
- Status: **argued.** Actions of different CTAs then touch disjoint mbarriers, barriers,
  shared memory and TMEM, so they commute. Their only contact is plain global memory, and a
  race there is found by vector clocks in either order (E-DRF). The state graph is acyclic
  (every action ticks a clock or retires an async op), so no cycle proviso is needed.
  `PTXV_NO_CTA_POR=1` turns it off; the test suite still passes with it on (rung 2).
- If wrong: *miss*, for a cross-CTA interaction through a channel the qualification test
  overlooks.

- **Parked warps released by the candidate** (`sync_ample`, `released_by`): a warp all of
  whose live threads are parked at an mbarrier poll (spin model: blocked until the phase
  completes) takes no step before that phase completes. If no entity other than the
  candidate's own later work (its warp's future async operations; for a pipe completion, the
  later completions of its pipe) can still arrive at that mbarrier, by the learned sets, the
  parked warp's future comes after the candidate and is left out of the conflict check. The
  choice is recorded with those mbarriers' arrival keys as writes, so anyone later learned to
  arrive there forces another round. Debug switch `PTXV_NO_RELEASED`. r2_cute_small: 4% fewer
  states, same verdicts; r2_cublas_gemm: the commit completions that the epilogue's
  accumulator poll blocked become ample.

**Local ample sets (same argument, under E-DRF).** One action alone is expanded if it
commutes with everything:
- a `bar.sync` arrival, unless more threads than the barrier's count may still arrive at it
  (E-BAR-OVER);
- a whole-warp step that starts at an invisible instruction (register, plain-memory,
  async-issue work after a wait);
- a whole-warp `wait_group`, an exit with no pending async ops, or a visible op guarded off
  in every lane;
- a whole-warp plain `mbarrier.arrive` that cannot complete the phase (it only decrements
  the pending count, so it commutes with the other arrivals and no try_wait can tell), in a
  kernel without counted arrivals: a counted arrival does not commute with it (with three
  pending, a count-3 arrival completes the phase alone, after a count-1 one it overshoots;
  `fixtures/mbar_count_order`, whose race the rule hid);
- the completion of a TMA store or an MMA (data movement; wait_group / commit observe it
  later);
- barrier.cluster arrive / wait (like `bar.sync` arrivals);
- an unguarded exit of *part* of a warp with no pending async ops of its threads (exited
  threads never merge again; an exit commutes with barrier arrivals).
- a split of *part* of a warp on its way out (`exit_bound`): every path its threads can take
  reaches an unguarded exit through invisible instructions only (guards uniform over the
  split and evaluated on current predicates; no uniform-register writes, no warp-collective
  operation), none of its async ops pending. A split it would merge with on the way runs the
  same invisible instructions from there, so merging only regroups invisible steps of threads
  that exit. r2_cublas_gemm: the producer warp drains its 16 stages lane by lane and exits.
  Evidence: `fixtures/e_exitbound` (races on the leaving lanes' path still found; as with
  every reduction under E-DRF, a racy variant may show fewer of its outcomes). Debug switch
  `PTXV_NO_EXIT_BOUND`.

A split also runs through a visible instruction guarded off in every member instead of
stopping there (it is a no-op).

Each action's conflicts are races found in either order. "Whole-warp" matters because a
split that moves can change which splits of its warp merge. `PTXV_NO_LOCAL_POR=1` turns
this off.

**Synchronization reduction (persistent sets, same argument).** A whole-warp step is also
expanded alone when its *synchronization footprint* conflicts with nothing another entity
can still do (`explore.rs` `sync_ample`). Entities are warps, plus each warp's async units.
- **What a footprint holds:** TMEM allocator, CTA exit, bulk groups, grid gates, strong
  accesses, and mbarrier keys. Plain memory and TMEM data are left out, because their
  conflicts are races (E-DRF).
  - An arrival into phase k reads key (M, k); the arrival that completes phase k also writes
    (M, k+1). So arrivals into the same phase commute. Counted ones need more (space 16,
    `rec_count`): every arrival into phase k reads the phase's pending-count key, and an
    arrival of several at once (a count, or a remote operation carrying several lanes) that
    takes the last pending arrivals writes it, so it conflicts with every arrival into its
    phase (DPOR found the count-first order of `fixtures/mbar_count_order` on the PTX front
    only by chance without it).
  - A poll while the current phase is k reads (M, k), and the completion of phase k writes it.
    Init and inval write every key of M. So a consumer's poll is independent of everything
    except the one completion that would change its result.
  - TMEM alloc / dealloc stay in (space 3) although they also clear the TMEM race shadow.
    They check that shadow first (S-TCGEN05), so a TMEM race with them is found in either
    order too.
- **What "can still do" means:** every footprint seen during the search, keyed by entity and
  start pc. For a warp, that is the pcs reachable in the CFG from where its live threads are.
  For async units, it is the issue pcs reachable from where the issuing warp's threads are,
  plus the issue pcs of its pending operations. A call may return, and a return may go
  anywhere. Two things are added to what was executed:
  - **Blocked steps.** In every state, a blocked split adds what may have disabled it or will
    enable it (`blocked_sync`): a blocked try_wait reads every key of its M, a blocked poll
    loop reads its flag (space 10/11), and a blocked alloc writes the allocator. This is
    DPOR's "next transition of a disabled process".
  - **Polls, for every phase.** A poll learned at a pc reads every key of its M, not only the
    phase it saw. The phase it reads depends on the phase, which the chosen step may change.
- **Learned sets, then a final check.** The sets are seeded by 16 random complete executions
  and then learned while the search runs. A set can grow by something that conflicts with the
  footprint of an action already chosen (for another entity). Then that choice may have been
  wrong, and the search runs again with the larger sets. Findings and outputs of every round
  are kept, since every transition is a real step. Growth that conflicts with no choice cannot
  change any choice, because conflict distributes over union.
- **Soundness argument.** Take the final round, and a wrong choice `a` with no wrong choice
  after it (the graph is acyclic). Below `a` the reduction is right, so every maximal
  execution that starts with `a` is explored up to reordering. Some execution has another
  entity `e` reach, without `a`, a step `c` that depends on `a`. Run `a` first: the steps
  before `c` commute with `a`, so `e` still reaches `c`'s pc, with the same registers.
  - If `c` is still enabled, `e` later runs a step from that pc, or stays blocked there. A
    later poll may see another phase, which is why polls are learned for every phase.
  - If `a` disabled `c`, `e` is blocked at that pc: `blocked_sync` records it.

  Either way the round learns a footprint at (`e`, pc) that conflicts with `a`. If it was
  learned before the choice, `a` was not chosen. If after, the round is rerun. This is the
  persistent-set theorem; the state graph is acyclic, so there is no cycle proviso. It relies
  on the same footprint relation as the DPOR explorer.
- **Bug fixed (rung 2):** the first version learned only executed steps, so the argument
  missed `c` when `a` disables it. CuTeDSL CM1 (no accumulator-full wait) has an epilogue
  warp complete the accumulator-empty phase before the MMA warp polls it. That poll then
  never runs, so the sets never held it, and every explored schedule deadlocked: 5,266
  states, no race. The full search finds the race and 64 terminal states. The 16 seed walks
  hid this (with `PTXV_POR_SEED_WALKS=0` it came back). It is now independent of the seeds.
- **Clusters (rung 2e):** remote arrivals, multicast TMA and commits touch the *target* CTA's
  mbarrier keys, so they are named like local ones; barrier.cluster arrive / wait write the
  cluster's key (space 13) and are also local-ample (they commute like `bar.sync` arrivals).
  Exits need no key: they commute with cluster arrivals, and a remote access racing the
  target's exit is reported order-independently at terminal states (S-CLUSTER-LIFE). The new
  keys are recorded only in sync-only footprints (`rec_sync`), so DPOR is unchanged.
- **Pipe completions as candidates:** the completion of a tcgen05 pipe op (MMA, commit) is
  expanded alone when it is its entity's only enabled action and conflicts with nothing other
  entities (or the entity's non-pipe ops) can still do: the entity's later pipe completions
  come after it (in-order pipe; SASS issues tcgen05 from one thread per warp).
- **Barriers cut "can still do":** a warp's steps past a barrier that the candidate warp has
  not arrived at cannot come before the candidate's step, so they are not counted: an
  all-thread `bar.sync` of the same CTA (a running candidate has not arrived at any), a
  barrier.cluster wait of the same cluster when the candidate's threads have not arrived at
  the current generation (`reach_sets` with cuts, `frozen_by`). Without this, four CTAs'
  mbarrier inits before the cluster barrier interleaved freely (they conflict with polls that
  can only run after it).
- **Phase-aware footprints (rung 3a, `refine` in `sync_ample`; `PTXV_NO_PHASE_CLIP`).** The
  learned sets merge every phase of an mbarrier and every operation issued at a pc. A
  conflict found with them is re-checked against what the entity can still do from this
  state, and dismissed if that no longer conflicts:
  - *Phases only grow.* Until an init / inval of mbarrier M, its future arrivals (space 7
    reads), polls and completions (space 12) touch keys >= its current phase k, and
    completions write arrival keys >= k + 1. So learned items below that are dropped.
    Not for an mbarrier some live learned footprint may still (re)initialize: init and
    inval now also write space 15 (sync-only), and an mbarrier with a live space-15 writer,
    or not materialized yet, is left as is. A choice that relied on clipping M records a
    write of M's space-15 key, so an init of M learned later forces another round.
  - *Pending operations name their mbarrier.* An async unit whose warp can no longer issue
    at that pc (or is the candidate's warp: it issues nothing before the candidate) only
    completes the operations already pending there, and those arrive at the mbarriers they
    name (one per TMA box / commit target).
  - *Own async units.* The candidate warp's async units can only complete what is already
    pending before the candidate runs (`PTXV_NO_OWN_ASYNC`).
- **Gated reachability (`gate_live` in `sync_ample`; `PTXV_NO_GATE`).** "Can still do" for
  a warp is its control-flow reach, cut at mbarrier polls it cannot pass before the
  candidate. A blocking poll (spin model, E-SPIN; not a peek) stops the warp when, for every
  live thread of the warp, its guard, mbarrier address and parity are fixed (computed from
  registers that no instruction the warp may run before then writes, so they still hold
  their current values), its awaited phase has not completed, and no live entity can arrive
  at, initialize or invalidate that mbarrier before the candidate: no learned footprint of a
  live (entity, pc) names it in space 7 or 15. A warp is live at the pcs of its gated region
  (the candidate's warp nowhere: its next step is the candidate); an async unit at the issue
  pcs in its warp's region (not the candidate's) and at its pending operations' mbarriers
  (an async candidate's own operation excluded). Regions and completable mbarriers grow
  together to a least fixpoint. A step of a stopped warp at the stopping poll cannot come
  before the candidate either (the poll stays disabled, and the candidate cannot disable it:
  it awaits the incomplete phase). The choice records, for every mbarrier that stopped a
  warp, writes of its arrival keys (space 7) and init key (space 15): an arrival there
  learned later forces another round.
  - Argument: take, in an execution from here without the candidate, the first step that
    leaves a gated region. It passes a stopping poll: its registers are the current ones
    (nothing before it in the region wrote them), so the awaited phase of that mbarrier
    completed, or it was re-initialized; someone did that earlier, from a pc inside its own
    region (or a pending operation did): so the mbarrier was completable, and the poll does
    not stop. Contradiction. The learned footprints it relies on are checked by the round
    mechanism, as for every other learned set.
  - Effect (r2_cublas_gemm): the epilogue's TMEM free (`UTCATOMSWS.AND`) sits behind the
    accumulator-full poll, whose phase only the MMA warp's commit completes: each CTA's TMEM
    alloc no longer interleaves with everything else (issue 1 of docs/rung2e.md); the MMA
    warp's stage-full poll no longer conflicts with another CTA's later-round multicast TMA,
    which sits behind the producer's stage-empty poll (issue 2).
- **Value-tracking look-ahead (`lookahead` in `explore.rs`; `PTXV_NO_VALTRACK`).** The
  gate gives up on a poll as soon as its address or parity register is written on the way,
  which is the case for every stage index and phase parity carried round a pipeline loop.
  The look-ahead runs every live warp but the candidate's abstractly from the current state,
  as far as it can get without the candidate, and is intersected with the gated regions
  (both are sound).
  - *Values.* A register is known (the same in every lane, or known per lane) or unknown (the
    same in every lane, or not). Register-only instructions are evaluated by the executor
    itself on a scratch copy of the machine (`Machine::exec_pure`), lane by lane when the
    operands differ between lanes. Loads and the like give unknown values.
  - *Mbarriers.* Each one gets an upper bound on the phases it can complete before the
    candidate. The bound counts every arrival of every configuration of every warp, per lane
    and per execution, plus the pending commits; transaction bytes are ignored. A blocking
    poll whose mbarrier and parity are known passes only if a phase of the other parity can
    have completed, between the lower bound this configuration already observed and the
    upper bound; otherwise it waits. Waiting configurations are run again as the bounds grow
    (a least fixpoint). A peek that cannot pass returns false, and its configuration also
    waits, to be run again with the true outcome if the bound grows. Unknown mbarriers or
    parities, named barriers and cluster barriers pass (but those the candidate cuts, as
    above).
  - *Control.* A branch on a known value goes one way (or the way E-CONST-BRA decided). On
    a uniform unknown value it goes both ways, as alternative configurations. On a
    lane-dependent value only the elected-body shape is followed (the lanes going one way
    wait at a BSYNC there, E-BSYNC). A warp that starts split, with one split running and the
    rest at a WARPSYNC / BSYNC whose mask covers every live lane, is followed the same way.
    Anything else gives up on the warp.
  - *Merging.* Configurations that reach a branch target are joined when no register is
    known in both with different values. So different loop iterations stay apart, and their
    stage indices and parities stay known. When a configuration joins (or is covered by) one
    it descends from, it has closed a cycle that is no longer unrolled. Every arrival in that
    control-flow strongly connected component then counts as unbounded.
  - *Giving up.* A warp is given up on when it exceeds its step budget (`PTXV_LA_BUDGET`,
    4000), diverges, (re)initializes an mbarrier, or starts with blocked, unlaunched or
    diverged lanes. It then gets its gated region, and every mbarrier its learned footprints
    there name gets unbounded arrivals. An arrival at an unknown mbarrier counts as unbounded
    at every mbarrier the learned footprints of its pc name.
  - *Use.* First, a warp, or its async unit, takes no step before the candidate at a pc the
    look-ahead did not execute. Second, the bounds clip footprints from above (`upper`):
    before a warp candidate, an mbarrier with phase k and bound C sees arrivals and polls
    with keys <= k + C, phase-key completions <= k + C - 1, and arrival-key completions
    <= k + C. Both record, with the choice, writes of the arrival and init keys of the
    mbarriers that held a configuration back or were clipped. So an arrival there learned
    later forces another round, as for the gate.
  - Argument: by induction over any execution from here without the candidate, each warp's
    steps are those of one of its configurations: the same pc, every known value equal, and
    observed completions at least the configuration's lower bound. Arrivals made so far are
    counted in the bounds: summed over configurations, unbounded on closed cycles, and the
    learned footprints are complete for the mbarriers recorded with the choice. So a poll that
    waits in the look-ahead has not succeeded in the execution, and a peek it fails fails.
  - Evidence: `PTXV_CHECK_WALKS` now also checks the look-ahead directly. At random states of
    its random executions it takes the next warp as the candidate and runs a random
    continuation without that warp. It checks that every step of a tracked warp starts at a
    pc the look-ahead executed, and that no mbarrier completes more phases than its bound.
    Unknown targets use the final learned footprints. r2_cublas_gemm cluster 0: 27,301 ->
    10,787 states, same verdict; 638 checks, 221 of them with a warp held back, 0 violations.
  - On its own, not enough for r3_grouped_mm. See "What remains".
- **TMA load completions as candidates (`PTXV_NO_TMA_POR`).** The completion of a TMA load
  (data into shared memory, complete_tx on its mbarrier) is expanded alone when it conflicts
  with nothing others can still do, as refined above (the entity's other pending operations
  complete in any order, so they are checked like other entities, by their own mbarriers).
  Parked warps only this completion can release are left out as for pipe completions.
- **Launches (`local_ample`; `PTXV_NO_LAUNCH_POR`).** Launching a CTA whose gate is open is
  expanded alone: it commutes with every other action and stays enabled (the gate only
  opens). It sets the CTA's threads running at their entry pc and joins the previous grid's
  completion into their clocks, which no other thread's step reads: an unlaunched thread
  counts as live and not arrived at every barrier, like a launched one that has not stepped,
  and its shared memory is alive either way. Only a try_cancel response reads which CTAs
  launched, so not with CLC.
- **Multi-grid launches (`sync_por_state_ok`; `PTXV_NO_GRID_POR`).** The reduction runs in
  states where nothing is left to launch and every live thread and pending operation belongs
  to one grid: the earlier grids completed, their launch gates (space 9) can no longer change,
  and the state behaves as a single-grid launch whose initial memory the earlier grids left.
  Before this, r3 (a prepare kernel, then the GEMM) ran with the reduction off.
- **BSYNC reconvergence (`local_ample`, `whole_or_parked`; `PTXV_NO_BSYNC_POR`).** Under
  E-BSYNC: an arrival at `BSYNC Bb` by lanes all of which execute it is expanded alone when
  Bb's recorded mask covers every live lane of the warp and every lane already parked waits
  at this same BSYNC. The release then comes exactly with the last live lane's arrival and
  frees every live lane, whatever the order, and the arrival touches nothing outside its
  warp. While the rest of a warp is parked like that, its running split counts as the whole
  warp (an invisible run cannot merge with parked lanes; the parked lanes move only after it
  arrives). CUTLASS elects one lane for its mbarrier inits and tensor-map copies while 31
  lanes wait at the BSYNC.
- **Dead synchronization bookkeeping (`forget_dead_sync`; `PTXV_NO_DEAD_SYNC`).** Not a
  reduction of interleavings but of states: bookkeeping no future step can read is cleared
  before a state is hashed, so states that differ only in it are one state. An mbarrier's
  arrival epochs (`Mbar::atomics`) are read only by a later `mbarrier.init` of it (the
  init-vs-arrival race check): cleared once no live thread can reach any `mbarrier.init`. A
  named barrier's generation count and last release clock (`bar_gen`, `bar_prev`) and its
  idle generation record are read only by later arrivals there (or an exit completing a
  generation in progress): dropped once no live thread of its CTA may still arrive at it
  (`BarCfg::reach`) and no generation is in progress. Reachability is the control-flow
  graph's (E-RET, E-CONST-BRA enforced). Exit checks, outputs and deadlock reports read none
  of these. r3: the terminal states of one slice went from thousands (differing only in
  arrival epochs) to 16.
- **Where it is off:** kernels with executable CLC, `st.async` or plain `shared::cluster`
  loads/stores; multi-grid states with more than one grid live. `PTXV_NO_SYNC_POR=1` turns
  it off.
- **Evidence:** CuTeDSL GEMM at 1 tile per CTA, fenced SASS went from 38,816 to 6,701 states
  with the same verdict. CM1 SASS takes 49,479 states with it and 384,737 without, with the
  same findings, even with no seed walks. Every fixture gives the same verdict with it on and
  off (rung 2).
- **What remains:** branching where a peek or poll races with the producer that completes the
  phase it reads. Those orders really differ. In an r3_grouped_mm slice, value tracking alone
  did not change the frontier. Two of the three conflicts behind that are now gone:
  - the epilogue's fast-path peek (`TRYWAIT` then `@P0 BRA` past the blocking retry) of a
    stage-full mbarrier, against the load warp's arrival that completes that phase: handled
    by the peek changes (E-SPIN, "Peeks in the explorer");
  - `BAR.SYNC 6` of the epilogue warps against the MMA warp's `BAR.ARV 6`: handled by
    `Machine::arv_current` (E-BAR-ARRIVE);
  - still there: the load warp's re-poll of a stage-empty mbarrier against the epilogue's
    arrivals into the next phase, which the look-ahead cannot rule out on paths through
    loaded tile flags. It costs states but does not stop a slice from completing.
  - With the first two, an r3 slice completes by exploration alone (look-ahead budget 4000):
    slice 1048576 takes 738,823 states (633 s), slice 1048577 738,919 states (864 s), both
    with the certificate's verdict (contract).

### E-SLICE · code
**Warps of a launch (group) that share nothing can be explored one slice at a time**, also
across CTAs and GPUs: the generalization of cluster slices (E-CTA-INDEP) to sets of warps
(`slices.rs`; `PTXV_SLICES=1` with `ptxv check --groups`, for every explored group). DeepEP
dispatch on 2 GPUs splits into its 4 (source, destination) flows per channel: the sender
warps of one rank for one destination with the receiver warps of the other rank for that
source. Combine splits per destination rank.
- Partition (`slices::partition`, a proposal only): one deterministic run, with the full
  footprint (`Footprint`, every space) of each step attributed to the warp that took it (an
  async operation's to its issuer's warp). Two warps are joined when they touch a common item,
  one writing it:
  - a byte of shared or global memory;
  - an mbarrier, a TMEM word or the TMEM allocator, a cluster barrier;
  - a named barrier: any two users of one `(CTA, id)` are joined, writes or not;
  - an all-thread `bar.sync` joins every warp of its CTA.

  The slices are the connected components.
- Exploration (`slices::restrict`): each slice on the group's machine with every other live
  thread exited. Every footprint item of every explored transition is recorded (`SFoot` for
  shared memory and synchronization spaces, `GFoot` for global memory), dry-run look-ahead
  steps and both shared-memory passes included. Every live grid's stream predecessor must be
  finished (a grid boundary inside the group would join slices).
- Cross-slice check (`slices::conflict`): no item that one slice wrote is read or written by
  another. Named-barrier generations and mbarrier phase keys conflict on any use. An
  all-thread `bar.sync` or `barrier.cluster` of one slice conflicts with every other slice
  that has threads in that CTA or cluster: their completion counts exits. A conflict makes the
  verdict incomplete. It is not reported as a race: slices may synchronize, and then they
  must be explored together.
- Argument. Let E be an execution of the whole group, and e its first step (of slice A) whose
  behaviour depends on another slice B: e reads or writes an item that a step w of B wrote
  earlier in E, or writes an item that w read, or completes an all-thread barrier whose count
  depends on B's exits. Before e, every step of A and of B depended only on its own slice. So
  A's steps up to e, and B's steps up to w, are executions of A alone and of B alone, which
  the per-slice explorations cover (under E-DRF the reduced search reaches every step of every
  execution, as E-CTA-INDEP's slices rely on). Then e and w are both recorded, and the check
  fails. If the check passes, no step of E depends on another slice. E is an interleaving of
  one execution per slice, each reachable alone, and every such interleaving is an
  execution. The group's races, faults, deadlocks (a state where every slice is terminal or
  deadlocked, at least one deadlocked) and terminal states are the slices', combined.
- Continuation (E-LAUNCH-GROUPS): the next group starts from one deterministic run's end
  state of the whole group, with every thread's clock the join of the slices' meets (a thread
  of a CTA several slices share also holds the async completions another slice folded into
  it), the union of the slices' final clocks and obligations, and each synchronization source
  from the slice that changed it (`Meet::compose`). A join of lower bounds on disjoint
  components is a lower bound. Final memory is unique when every slice's is.
- Status: **argued**. Evidence: the r4c fixtures. Notify splits into the 2-GPU barrier CTA
  pair and 4 single CTAs. Dispatch splits into 4 flows of 24 warps, cnc into 9 slices,
  combine into 2. Mutants that break a flow are still caught inside their slice (docs/rung4.md,
  rung 4c).
- If wrong: *miss* of an interaction through a channel the footprint does not name.
- Where: `slices.rs`; `machine.rs` `SFoot`, `Machine::rec`; `exec.rs` `Op::BarSync`
  (`SPACE_CTA_ALL`); `groups.rs` `Meet::compose`, `Opts::slices`.

### E-POLL-ANY · code
**A strong load whose step ends the same for every value its location can hold commutes with
the writes of that location** (a value-insensitive poll; `exec.rs` `vip_ok`, in the stateful
explorer; `PTXV_NO_VIP=1` turns it off). DeepEP's senders poll the ring head before each
chunk (`while (free slots < chunk) head = ld.volatile(...)`). While the ring cannot be
full, every value passes, but each of the 12 sender warps' polls raced with each head store,
and the search branched on every order (dispatch: no flow completed in 300k states; now
4,983 each).
- The check, at the step of a split starting at a strong global or shared load whose active
  lanes all read one location of at most 8 bytes (in the first search also an acquire load,
  `VipLearn::acquire`; DeepEP's channel waits are `LDG.STRONG.SYS` + `CCTL.IVALL`, which
  the lifter makes acquires):
  - the candidate values are every value any explored write left at that location
    (`VipLearn`, over all rounds) and the current one. With `PTXV_VIP_MONO=1`, if those fail
    and the location is *monotone* (no explored write lowered it, as an unsigned integer), the
    check is retried with only those at least the current one: no later write can bring back
    a smaller value. It is off by default. A load that is a poll in some states and not in
    others leaves fewer states equal: nowrap dispatch takes 4,983 states per flow without the
    cut, and is incomplete at 300k with it, even as a fallback. DeepEP's ring heads only grow, so a
    sender's poll for a free slot, once the head allows it, passes for every later head;
  - the step runs on a copy once per candidate. Every member must end at the same pc and
    state, with the same predicates and uniform registers, and with registers that differ only
    where they are dead (`spin::live_after`), and the memory it wrote (ranges and bytes) must
    agree: a step that stores the loaded value (`LDS R5; ...; STG [..], R5` in one step) is
    not value-insensitive (fixtures/e_lds_atoms LA1 lost an outcome before). A poll iteration that fails for some value ends
    elsewhere, so it is refused.
- The load then runs:
  - without a footprint on its location (it commutes with the writes there);
  - without entering the race shadow's read set, but `Machine::vipr`, which no write clears
    and every later write is checked against (a race unless ordered after it, or both
    morally strong). A strong poll's flag read at arrival (`spin::after_step`) goes there too,
    whether the poll would fail or not, so that the step that reaches a poll commutes with
    the flag's writes as well;
  - without observing the location's release (no pending acquire, no E-READ-DONE edge).
- Learning: watching a new location, a write leaving a new value at a watched one, or a
  write lowering a watched one marks the round dirty, and `explore_one_model` runs another round. The last round decides with
  every value of every explored execution.
- Argument. Let L be such a step of warp w at state s, and W a step of another entity that
  writes the location. L is enabled whatever the location holds (a failing iteration would
  end elsewhere). W reads nothing L writes (w's registers, `vipr`), and checks `vipr` where
  it would have checked the shadow, so both orders report the same conflicts. After W, L
  reads a value W left, which is a candidate (for a monotone location, W's value is at least
  the current one), so it ends with the same pcs, predicates and live registers. The two orders reach states that differ at most in registers dead in w,
  which no later step, finding or output reads. Dropping the acquire removes only
  happens-before edges, so it can add a race report and never hides one. A search that
  reports a race is run again with E-POLL-ANY on relaxed loads only. If that search reports a
  race too, the search runs once more without E-POLL-ANY and without the peek changes, and
  that verdict stands. A clean verdict with it is therefore the plain model's.
- `vipr` moves to the launch-group obligations at a group boundary and in every terminal
  state of a meet (E-LAUNCH-GROUPS), like other strong reads.
- Status: **argued** (an exact reduction up to dead registers, plus a conservative weakening).
  Negative tests: `fixtures/e_poll_any` (`crates/sim/tests/fixtures.rs`
  `e_poll_any_keeps_value_sensitive_polls`): a poll that passes on the initial flag but
  would block on a later, lower value is not reduced (the value is learned in a later round,
  and the store makes the location non-monotone), so its deadlock is still found; a race
  after a reduced poll is still found.
- If wrong: *miss*, for a value the learning did not see.
- Where: `exec.rs` `vip_ok`, `access_x`, `Op::Ld`; `spin.rs` `after_step`; `machine.rs`
  `VipLearn`, `Machine::vipr`; `explore.rs` `explore_one`, `explore_one_model`.

### E-CLOCK · code
**SASS clock reads return 0, and `vprintf` does nothing.** DeepEP bounds every spin with a
`clock64()` timeout (`CS2R Rd, SR_CLOCKLO`; ~200 G cycles) that prints and traps; torch's
symmetric-memory `barrier_kernel` bounds its CAS spins with `%globaltimer` (`CS2R Rd,
SR_GLOBALTIMERLO`), read as 0 alike. The model has
no time, so a timeout is a schedule property it cannot decide; it reads the clock as 0 so the
difference never exceeds the bound, and a poll loop around it stays a stutter loop (E-SPIN
warp-level polls). The `vprintf` before a trap is skipped; the `BPT.TRAP` after it is modeled.
- PTX front (2026-11-07, front parity): `%clock` / `%clock64` / `%globaltimer[_lo|_hi]` read
  0; `call.uni (retval0), vprintf, (param0, param1)` does nothing (its `st.param`s are dropped,
  `ld.param [retval0]` gives an undefined register: S-INIT); `cvta.local.u64` adds the SASS
  front's unmapped generic local base (S-CBANK). Each vprintf call site (either front) is a
  note "vprintf reachable at ...", so a timeout / print path is visible per record.
  probes/printf_clock: DeepEP's timeout shape (clock64 + globaltimer deltas, a stack-built
  argument buffer, vprintf, trap), GB300 50/50, both front-ends bitwise.
  `call.uni __assertfail, (param0, .., param4)` (Triton / CUDA device asserts; 2026-11-08)
  aborts the grid as the SASS front's `CALL.ABS` to it (a trap; its `st.param`s are dropped).
  probes/assertfail: not taken, GB300 20/20 and both fronts bitwise; taken, both fronts report
  the trap and GB300 fails the launch with "device-side assert triggered", printing the
  message from the module `.global` initializers.
- Where: `crates/sass/src/lib.rs` (`CS2R ... SR_CLOCKLO`), `exec.rs` `CallAbs`; `ptx`
  `vprintf_calls`, `Lower::val`, the `cvta` arm; `explore.rs` `static_findings` (notes).
- Status: **argued.** The timeouts guard against hangs of seconds; a hang the model sees is a
  deadlock finding instead of a trap.
- If wrong: *miss* on code whose results depend on elapsed time (none in the captured stacks:
  the clock only feeds timeout compares).

### E-LOCKSTEP · code
**SASS only: lanes of a converged warp are ordered instruction by instruction** (a memory
access by lane i happens before the next instruction of every lane). PTX lanes stay
independent threads.
- **STSM / LDSM** (2026-11-06) are joined too: full-warp collectives, executed by every lane
  together. Before, they were not: Triton kernels with `.reqntid 32` (fla chunk_kda /
  chunk_gated_delta_rule) have `bar.warp.sync` between `stmatrix` and `ld.shared` in PTX,
  which ptxas drops, and the SASS front reported "observable" races between `STSM` by lane 1
  and `LDS.128` by lane 0 (`fixtures/e_stsm_warp`). The ordering itself is E-WARP-ORDER.
- `BRA.DIV` and its out-of-line `WARPSYNC.COLLECTIVE` path: see E-COLLECTIVE.
- Lanes of one atomic instruction on one address are performed in lane order. PTX leaves
  that order open; `PTXV_ATOM_LANES=rev` reverses it to show what depends on it (rung 3b:
  flashinfer's routing assigns each token its slot within an expert with `ATOMS.ADD`, so the
  permutation depends on it, the MoE output does not). The certificate does not rely on it
  (C-ORDER refuses such atomics).
- Memory instructions joined: `LD*` / `ST*` / atomics / `LDTM` and (2026-11-05) `STTM`: FA4's
  softmax warps store P with `STTM`, wait (`FENCE.VIEW.ASYNC.T`) and one elected lane per warp
  arrives on the MMA warp's mbarrier; without the join every other lane's `STTM` raced with
  the PV `UTCHMMA` (a false race in every FA4 forward / decode kernel).
- **Known gap:** every split is joined, also one the model merged at a pc after a divergent
  branch without a convergence point, where hardware need not run the two sides together.
  Joining only while the warp is converged (`Machine::wdiv`, as the PTX labels below use it)
  was tried on 2026-11-06 and made `fixtures/e_mma_pair_m128`'s SASS exploration exceed 2M
  states (245k before), so it is not applied.
- Where: `exec.rs` `step_split_inner` (lockstep VC join), `Machine::atom_lanes_rev`.
- Status: **argued.** ptxas's own warp-synchronous code (TMEM dealloc bookkeeping) depends on
  it, and without it that code races. The in-order issue of a converged warp is validated
  (E-WARP-ORDER); the gap above is not.
- If wrong: *miss* on warp-synchronous SASS that is racy under independent thread scheduling.
- Retire: a statement of the reconvergence rules in NVIDIA documentation, and the join gated
  on convergence.

### E-WARP-ORDER · code
**GB300 issues the shared-memory instructions of a converged warp in program order:** a store
by lane i is visible to a load by lane j on a later instruction, and a load by lane j does not
see a store by lane i on a later instruction, with no `bar.warp.sync` between them.
- SASS: the basis of the E-LOCKSTEP join (no race finding for such pairs).
- **Converged** (both uses): no divergence since a `BSYNC` / `WARPSYNC` released every live
  lane of the warp together (`Machine::wdiv`: set when the lanes of a split leave an
  instruction at different pcs, or a live lane of the warp is not in the split; false at
  launch).
- PTX: such a pair is still a data race (the PTX memory model needs `bar.warp.sync`, and
  ptxas may reorder unsynchronized accesses of different lanes). It is reported, with the
  label `gb300: latent-needs-divergence` instead of the hardware-model classification, when
  both accesses are plain memory instructions (`ld` / `st` / atomics / `ldmatrix` /
  `stmatrix`) by lanes of one warp that is converged at the second
  access and has not diverged since the first (`Machine::div_clk`: the first access's own
  clock is at least the clock at the last divergence). The label depends on the path that
  first reports the race.
- Where: `exec.rs` `step_split_inner` (join, `note_divergence`), `warpsync_release`
  (`note_convergence`), `access_x` (label); `twolevel.rs` `LATENT_DIV` (labelled findings are
  not reclassified).
- Status: **validated** (`probes/warp_order`, GB300, CUDA 13.0, 2026-11-06): 8 shapes
  (`STS` -> `LDS`, `STS.128` -> `LDS.128`, a 32-bit store into a 64-bit load, `STSM.16.M88.4`
  -> `LDS.128` of one row, `LDS` -> `STS`, `LDS.128` -> `STS.128`, `LDSM.16.M88.4` ->
  `STS.128`), lane pairs (1, 0), (0, 1), (31, 0), (5, 17), byte offsets in banks 0, 1, 7, 31
  and two 16-byte rows; 32 warps x 296 CTAs and 1 warp x 2368 CTAs, unrolled and not (the
  pairs adjacent in SASS or up to 6 instructions apart), 50 runs, 4096 iterations each:
  0 wrong of 4.5e11 checks. Controls: the same store / load on the two sides of a
  lane-dependent branch (a diverged warp) are wrong in 99.98% of checks; unsynchronized
  warps in 99.6%.
- If wrong: *miss* on SASS (a reordered pair no finding covers); on PTX only the label is
  wrong.
- Retire: NVIDIA documentation of intra-warp memory ordering.

### E-STSM-DUP · code
**Lanes of one `stmatrix` (SASS `STSM`) that write the same shared word with the same value
write it once.** Triton passes one row address for replicated rows, so lanes holding equal
data store to the same bytes in one instruction (fla `chunk_kda_bwd_kernel_wy_dqkg_fused_forloop`:
lanes 0 and 16 of `STSM.16.M88` on byte 0x4c0, reported as a write-write race until
2026-11-06). Equal values leave the same bytes whatever the order. Different values are a
`violation` (`stsm-dup:<pc>`): which lane wins is neither specified by PTX nor probed; the
model keeps the first lane's value.
Not covered by E-WARP-ORDER, which orders instructions, not the lanes of one.
- Applies to the non-transposed form; `.trans` 16-bit writes are race-checked as before.
- Where: `exec.rs` `Op::StMatrix`. Fixture: `e_stsm_warp` variants `dup` (clean), `dupdiff`
  (violation).
- Status: **argued** (same bytes in either order).
- If wrong: *miss* only if hardware tore a same-value write, which leaves the same bytes.
- Retire: never (a modelling simplification for equal values).

### E-ALIGNED · code
**`.sync.aligned` instructions are convergence points:** the whole warp executes them
together, and a lane guarded off still counts as a partial-warp execution (violation).
`elect.sync` with the full membermask is one too (PTX ISA: it "causes the executing thread to
wait until all threads in the membermask execute" it). Rung 2's CuTeDSL kernel needs this: a
leader that branched into an if-block and the rest of the warp reach the next `elect.sync` at
different times. Without the rule each part elected its own leader and both initialized the
same mbarriers.
- SASS `SHFL` is gated the same way (all live lanes of the warp arrive first): ptxas only
  emits it where the warp is converged, reconverging with `BSYNC` after divergence, and the
  model treats `BSSY`/`BSYNC` as no-ops.
- **Aligned named barriers** (`bar.sync` / `bar.arrive` = `barrier.*.aligned`, and every
  SASS `BAR` outside a `WARPSYNC.COLLECTIVE` region): every live lane of a warp executes the
  same barrier instruction together. A split waits at one until the rest of the warp is there;
  if the other live lanes are at a different aligned barrier (or a `WARPSYNC`), the warp can
  never converge: a `violation` (divergent aligned barrier), as is an aligned barrier guarded
  off in part of the warp (PTX). ptxas assumes it too: it compiles `bar.sync` in both arms of
  a branch to `@P0 WARPSYNC.ALL; @P0 BAR; @!P0 WARPSYNC.ALL; @!P0 BAR`, and the non-aligned
  `barrier.sync` to `BRA.DIV` around the `BAR` with a collective slow path. Only the per-warp
  rule is modeled: different warps reaching one barrier from different pcs (what
  compute-sanitizer synccheck also flags, the FA4 warp-role bug's pre-fix) is clean (GB200 completes).
  `IR Op::BarSync::aligned`, `exec.rs` `aligned_bar_others`, `sass` `aligned_bars`.
- Where: `exec.rs` `split_enabled`, `exec`.
- Status: **validated.** The PTX ISA requires it, and B16 (partial-warp alloc) hangs GB300.
  Named barriers: two halves of a warp at two `bar.sync 1, 64` hang GB300, at two
  `barrier.sync 1, 64` complete (fixtures/e_aligned_bar AB1 / AB2, `ptxv probe`
  2026-11-05); the FA4 warp-role case's `probe_intrawarp` hangs GB200 (fixtures/e_fa4_warp_roles).
- If wrong: *false alarm*.

### E-WARPCOLL · code
**`stmatrix` / `STSM` and `shfl.sync` / `SHFL` move data among the 32 lanes of a converged
warp in one step**: every lane's source registers are read, then every destination is written.
- Where: `exec.rs` `exec` (warp-level arms).
- Status: **argued.** Both are `.sync.aligned` in PTX (E-ALIGNED), so the whole warp executes
  them together. A partial warp gives an error, not a verdict. Rung 2's Triton epilogue stages
  its tile through `stmatrix`, and its output matches GB300 bitwise.
- SASS `SHFL` (no member mask) waits for the warp's other live lanes, except lanes parked at
  a `BSYNC` (2026-11-06): those rendezvous there with the lanes their `BSSY` recorded, so they
  do not reach the SHFL first. ptxas's warp-aggregated atomic (`VOTEU.ANY / FLO / POPC /
  @leader ATOMG / SHFL.IDX` in a branch a subset of lanes takes; DeepEP hybrid dispatch,
  Triton MoE `_verify_kernel`) then shuffles among the lanes in the region, as the hardware
  does, instead of deadlocking.
- PTX `redux.sync` with a membermask naming a subset of the warp (an immediate other than
  0xffffffff, or a register; 2026-11-07): the lanes the mask names (live ones) execute it
  together and it reduces over them. A mismatch between the executing lanes and the mask, or
  lanes that disagree on the mask, is a violation (PTX: undefined). Validated by
  probes/redux_mask (GB300 bitwise on both front-ends): even / odd halves on 0x55555555 /
  0xaaaaaaaa, a register 0x0000ffff beside an immediate 0xffff0000, one lane on 0x8, and a
  register full mask. Its `mask_mismatch` variant (32 lanes on 0xaaaaaaaa) completes on GB300
  with every lane holding the full-warp sum; ptxas emits a plain `REDUX`, so the SASS verdict is
  clean while the PTX verdict reports the mismatch.
- PTX `shfl.sync` / `vote.sync` (any / all / uni / ballot) with a partial or register
  membermask, and lanes naming different masks at one instruction (2026-11-07): the executing
  lanes are partitioned by the mask each names; each group waits for the live lanes of its
  mask and runs once, together with lanes of the same mask standing at another instruction of
  the same kind (`redux.sync`: same operation), each lane reading its own operands and writing
  its own destination. A group whose lanes (with those partners) differ from its mask is a
  violation. Validated by probes/coll_split (pair masks, a pair split across two `shfl.sync`
  or two `vote.sync.ballot`) and probes/shfl_mask (half-warp masks on the two sides of a
  branch, quad-mask `shfl.up` / `vote.any` / `.all` / `.uni`, a 0xffff `shfl.down` the other
  half skips, `redux.sync` groups split across two instructions): PTX front bitwise = GB300
  (50/50 scrubbed). shfl_mask's `skip_half` (32 lanes on 0xffff) is a violation. ptxas's
  `WARPSYNC.EXCLUSIVE Rm` before a quad-mask `VOTE` is refused on the SASS front.
- If wrong: *miss* only if the hardware could expose a half-done exchange, which a
  `.sync.aligned` instruction cannot.

### E-COLLECTIVE · code
**ptxas's collective lowering of a run-time member mask** (`vote.sync` / `redux.sync` /
`__ballot_sync(m, ...)` with `m` not a constant; torch.topk's gatherTopK):
`REDUX.OR URa, Rm; R2UR P, URZ, Rm; @!P BRA.DIV URa, slow`, fast path = the collective op,
slow path = `WARPSYNC.COLLECTIVE Rm, end; op...; ENDCOLLECTIVE; end:`.
- `BRA.DIV URm, L` is warp-collective: every lane of the executing split branches unless the
  guard holds on all of them and their lane mask equals URm; otherwise all fall through (a
  guard false everywhere: no branch, unobserved).
- `WARPSYNC.COLLECTIVE Rm` parks the arriving lanes; the group of lanes naming the lowest
  lane's mask enters once all its live lanes are parked at this instruction and no lane of the
  warp is inside the region; `ENDCOLLECTIVE` lets the next parked group in. A lane whose mask
  does not contain itself is an error. Lanes of one mask parked at a different collective
  instruction are not combined (they wait: a deadlock, not a wrong value).
- `BRA.DIV` falling through (2026-11-07): the lanes of the member mask are converged, and the
  fall-through orders them (their clocks are joined) as the slow path's `WARPSYNC.COLLECTIVE`
  does. ptxas lowers `__syncwarp()` (`bar.warp.sync -1`, often merged with the next
  `elect.sync`) to `BRA.DIV 0xffffffff, slow` with no WARPSYNC on the converged path, so
  without the join a lane's DEPBAR-completed bulk store had no happens-before to the elected
  lane's bulk load into the same shared bytes (DeepEP hybrid dispatch, a false race).
  probes/syncwarp_fast (that shape): GB300 50/50, both front-ends bitwise and clean.
- `REDUX` (AND) / `.OR` / `.XOR` / `.SUM`, `CREDUX.MIN` / `.MAX` (`.S32`): reduction of the
  active lanes' 32-bit values into the uniform register. `SR_LEMASK` / `GEMASK` / `GTMASK` /
  `EQMASK` as `%lanemask_*`.
- Where: `exec.rs` (`Op::Bra` with `div`, `Op::WarpSync { collective }`, `Op::EndCollective`,
  `collective_release`, `Op::Redux`), `explore.rs` lookahead (BRA.DIV forks unless decided).
- Status: **validated** by fixtures/s_warpcoll (GB300 bitwise, 2026-11-03): every form above
  over data-dependent active sets, the shrinking-mask loop of gatherTopK, and legal non-uniform
  member masks (two and three groups in one warp: BRA.DIV taken, the slow path runs each group
  separately; the alternative "only guard-true lanes branch" would give the 24-lane ballot
  instead of GB300's per-group ones). torch.topk's gatherTopK (fixtures pair in
  docs/compare.md) is certified clean and bitwise = GB300.
- **Lanes of one mask at different `WARPSYNC.COLLECTIVE` instructions** (2026-11-07; ptxas
  13.0 emits one slow path per call site, so a pair split across two branches parks at two):
  GB300 combines them. probes/coll_split (pair masks 3 << (lane & 30)): with the pair's lanes
  at two different `shfl.sync.bfly` (source = destination register) or `vote.sync.ballot`
  call sites, every lane gets its partner's value and the pair's ballot, 50/50 scrubbed, as if
  both ran one instruction (serialized regions would read an overwritten register). Model:
  the group enters once all its live lanes are parked at collective instructions whose regions
  are the same code (operations and guards); they run one of the regions together and each
  lane continues after its own region's `ENDCOLLECTIVE` (`Thread::coll_home`); a group
  entering after another left a region is ordered after it. Regions that differ keep waiting
  (unmodeled). SASS `SHFL` also does not wait for lanes parked at a `WARPSYNC` or standing at
  another `SHFL` (the two sides of a branch shuffling within their own lanes). Before, both
  cases deadlocked: probes/warpsync_pairs (ptxas's lowering of a split `bar.warp.sync`) and
  probes/coll_split now equal GB300 on the SASS front-end.
- If wrong: *miss/false alarm* only for collective regions entered by lanes at different
  instructions whose regions differ (unmodeled: they wait).
- `BRA.CONV URm, L` (DeepEP's `k_hybrid_combine` exit, `UMOV UR4, -1; BRA.CONV UR4, fast`,
  slow path = `WARPSYNC.COLLECTIVE ... ENDCOLLECTIVE` inline) is taken exactly where `BRA.DIV
  URm` would fall through. **Argued** (the inverse test; not run on GB300): no ptxas input
  that emits it has been reduced to a fixture. If wrong, the other path runs; in the kernels
  seen both paths do the same work.

### E-COLLECTIVE-ALL · code
**ptxas's divergent path of a non-aligned `barrier.sync`** (CuTeDSL `barrier(aligned=False)`;
FA4's fix for its warp-role barrier bug): `UMOV UR, -1; BRA.DIV UR, slow; BAR.SYNC id, n` with the slow path
`WARPSYNC.COLLECTIVE.ALL end; BAR.SYNC R, R; ENDCOLLECTIVE; end:` (R = n << 16 | id).
- `WARPSYNC.COLLECTIVE.ALL` parks the arriving lanes until every live lane of the warp is parked
  at such an instruction, at any pc (two branches each with their own barrier.sync have two);
  then all of them go on, their histories joined, each into its own region. An exiting lane
  re-checks. Inside, the barrier counts threads as everywhere in the model (E-BAR-EXIT); since
  all live lanes of the warp arrive, that equals GB300's per-warp count.
- Where: `exec.rs` `Op::WarpSync { all: true }`, `collective_all_release`, `Op::Exit`.
- Status: **validated** (2026-11-05, GB300 GPU 3, fixtures/s_bardiv*.cu): both warps split into
  two barrier branches (count 64): completes, neighbours' values visible (s_bardiv, hw 20/20
  bitwise); warp 0 split, warp 1 converged (64): completes (s_bardiv_mixed, 20/20); warp 0
  split, warp 1 converged, count 96: hangs (e_bardiv_short: model deadlock); half of each warp
  at the barrier (two branches, count 32), the other half spinning on a flag set after it:
  hangs (e_bardiv_hang: model deadlock; per-thread counting would complete). Same kernel with
  one barrier branch: ptxas emits `WARPSYNC.ALL; BAR.SYNC` instead (hangs; modeled already).
- If wrong: *miss / false alarm* for lanes that reach the gate and never the barrier, or a
  gate that does not wait for lanes at other collective instructions (not observed).
- Retire: a per-warp barrier arrival model (AJ's aligned-barrier work).

### E-BAR-EXIT · code
**Barriers and exited threads.**
- `bar.sync` without a thread count waits only for threads that have not exited.
- A barrier with an explicit count counts *warps*: a warp's arrival counts the warp size (32),
  its exited or nonexistent lanes arriving with its live ones (BAR is executed per warp). A
  warp none of whose lanes arrives (all exited) adds nothing. The explorer's barrier
  reductions (`arv_current`, `bar_oversub`) bound each warp that may arrive by 32.
- Where: `exec.rs` `bar_try_release`, `Op::Exit`.
- Status: **validated** (2026-10-30, `probes/bar_exit` on GB300):
  - whole warp exited: completes;
  - half warp exited: completes;
  - `bar.sync 0, 64` with a warp exited: **hangs**;
  - nobody exits (control): completes.

  The simulator reproduces all four on both front-ends (regression test
  `probe_bar_exit_matches_hardware`).
- Partly exited warps (2026-11-05, `probes/bar_exit_warp`, `ptxv probe` on GB300): 16 + 32,
  1 + 32 and 16 + 16 survivors of two warps complete `bar.sync 1, 64`; 16 survivors of one
  warp complete `bar.sync 1, 32` alone; 16 survivors that `bar.arrive 1, 64` release a full
  warp's `bar.sync 1, 64`; a 48-thread block (a 16-lane last warp) completes `bar.sync 1, 64`;
  16 + 32 survivors of `bar.sync 1, 96` (the third warp exited) **hang**, and with the third
  warp present complete. Both front-ends reproduce all nine
  (`probe_bar_exit_warp_matches_hardware`). Where: `Machine::bar_arrivals`.
- History: until 2026-10-30 the first two cases were reported as deadlocks, which was a false
  alarm. The ISA says *"barriers exclusively waiting on arrivals from exited threads are always
  released"*. Until 2026-11-05 counted barriers counted threads: a half-exited warp at a
  counted BAR (fixtures/e_aligned_bar AB3, SASS) was a false deadlock.

### E-BAR-ARRIVE · code
**`bar.arrive` / `BAR.ARV` on a named barrier.**
- The arriving thread counts one arrival toward the barrier's current generation and continues
  without waiting. Its clock at the arrival is released: when the generation completes, every
  `bar.sync` waiter of that generation acquires it (the PTX memory model makes
  `bar{.cta}.arrive` a release, `bar{.cta}.sync` an acquire+release). Its later accesses are
  not released.
- A thread arriving again (`bar.arrive` or `bar.sync`) before the generation it arrived in with
  `bar.arrive` completed: the PTX ISA gives it no meaning. Reported as a violation
  (`bararrive2`), then modeled as the hardware does: the arrival counts again.
- Mixing thread counts on one barrier (`bar.arrive 1, 64` with `bar.sync 1, 96`) is the
  existing `barcount` violation; the model then keeps the first count.
- Explorer: an arrival that does not wait can land in the next generation, so arrivals do not
  commute with each other at a generation boundary. `bar.arrive` writes the barrier's arrival
  state (footprint space 14), `bar.sync` reads it (writes it too at an over-subscribed
  barrier, E-BAR-OVER); the stateful explorer's local reduction does not commute `bar.sync`
  arrivals in a kernel that uses `bar.arrive`.
- Explorer refinement (`Machine::arv_current`; `PTXV_NO_ARVGEN`): an arrival lands in the
  current generation in every order when no thread can arrive twice in one generation
  (this one has not arrived in it, and no `bar.arrive` on the id can reach another arrival
  there, `BarCfg::arv_repeat`), every site on the id names the same immediate thread count
  (`BarCfg::arv_mixed`: the first arrival fixes the generation's count, so with mixed counts
  the order decides whether it completes; e_bararv AB3 lost its deadlock without this check),
  and the generation cannot complete without it (all-thread
  barrier; or the arrivals so far plus every other thread that may still arrive, by
  `thread_may_arrive` as for E-BAR-OVER, stay below the count). Such an arrival reads the
  generation's key, like a `bar.sync` arrival, instead of writing every generation's
  (including the E-BAR-OVER write of g-1 when it is not ordered after the previous
  release). r3: the MMA warp's `BAR.ARV 6` no longer conflicts with the epilogue's
  `BAR.SYNC 6` (count 160: 128 epilogue threads + its 32).
- Where: `exec.rs` `Op::BarSync`, `bar_try_release`; `explore.rs` `local_ample`.
- Status: **validated** (2026-11-02, `probes/bar_arrive` on GB300; regression test
  `probe_bar_arrive_matches_hardware`):
  - warp 0 `bar.arrive 1` then `bar.sync 2`, warp 1 `bar.sync 2` then `bar.sync 1`:
    completes (the arrive neither blocks nor fails to count); with `bar.sync 1` in place of
    the arrive (control): hangs;
  - warp 0 arrives twice on `bar.arrive 1, 64` before warp 1 can arrive: hangs (warp 0's
    64 arrivals completed the generation alone, warp 1 then waits in the next one); the same
    with warp 1 never using barrier 1 completes (the second arrival does not block);
  - `bar.sync 1, 96` against `bar.arrive 1, 64`: **traps** (illegal instruction), reported
    as a violation.

  `fixtures/e_bararv` good matches GB300 bitwise (200 runs).
- If wrong: *miss* if hardware released less than the arriver's earlier accesses (the probe
  shows completion and counting, not memory ordering; the ordering is the ISA's guarantee).

### E-BAR-OVER · code
**A named barrier used by more threads than its count.**
- `bar.sync id, count` completes a generation every `count` arrivals (PTX ISA: once the
  count is reached the waiting threads are released and the barrier resets). More threads
  may use the barrier than `count`: the surplus arrivals start the next generation. This is
  defined behavior, not a violation, and GB300 does not trap on it (unlike a mismatched
  count, E-BAR-ARRIVE). Which threads share a generation, and so who synchronizes with whom
  (happens-before) and who is left waiting alone, then depends on arrival order.
- Explorer (only barriers with a count; all-thread `bar.sync` is unaffected):
  - *Who may still arrive*: the live threads of the CTA from whose pc a `bar.sync` /
    `bar.arrive` on that id (or a non-immediate id) is reachable, those parked in it included
    (`Machine::bar_oversub`). The walk decides guarded branches by predicates whose values it
    can compute (comparisons on the path over registers the walked region never writes, or
    the current values of such predicates), so warp-role dispatch on the warp index is
    followed. With at most `count` such threads, each arriving once per generation (it
    waits), the next generation completes exactly when all of them arrived, in any order.
  - *Barrier ids in registers* (2026-11-04, DeepEP dispatch/combine:
    `bar.sync (tid >= 384), 384`, the id recomputed from `%tid` in each loop body). The walk
    tracks register values computed on the walked path from thread constants (special
    registers, immediates, launch constants) through MOV / SEL with known predicates. A
    register outside that path environment is read at its current value only where no walked
    path from the thread's pc may write it (forward may-write dataflow over the walked
    region), instead of only where nothing in the region writes it. A `bar.sync` on a
    register id counts for barrier `id` only if the id is unknown or equal. Both are exact on
    the paths they describe and conservative elsewhere, so "who may still arrive" is still
    an over-approximation. Thread constants read are recorded with the walk's memo key, so a
    walk is reused only by threads that agree on them. Kernels whose barrier ids are all
    immediates keep the earlier region-wide walk (`BarCfg::regid`): the same
    over-approximation, without the per-lane memo cost.
  - Local reduction: a counted `bar.sync` arrival is local-ample only when at most `count`
    threads may still arrive (`local_ample`); otherwise the synchronization reduction and
    DPOR decide from footprints.
  - Footprint (space 14, key = barrier id and generation): an arrival into generation g reads
    key g. An arrival into g not ordered (vector clocks) after everything generation g-1
    released could have landed in g-1: it writes key g-1. The arrival (or exit) completing a
    generation writes its key when more than `count` threads may still arrive, so an arrival
    that a DPOR sleep set holds back wakes up when the generation it would land in changes.
- Where: `explore.rs` `bar_cfg`, `Machine::bar_oversub`, `thread_may_arrive`, `local_ample`;
  `exec.rs` `Op::BarSync`, `bar_try_release`; `machine.rs` `bar_prev`.
- Status: **validated** (2026-11-02, `fixtures/e_barover` on GB300, `ptxv hw --scrub`, with
  `--delay` nanosleeps forcing arrival orders):
  - BO1 (4 warps on `bar.sync 1, 64`): natural order correct 1000/1000; producer delayed and
    the other warps delayed less: the consumer is released with another warp and reads
    unwritten shared memory, 200/200 wrong;
  - BO2: natural order correct 1000/1000; producer delayed: 200/200 wrong;
  - BO3: natural order **hangs** (the consumer is left alone in the second generation);
    producer delayed: completes, 200/200 wrong.

  Before this rule the stateful explorer called BO1 and BO3 clean and found only the
  deadlock of BO2, and DPOR missed BO3's deadlock (it commuted all `bar.sync` arrivals).
- If wrong: *cost* only (more interleavings of arrivals; the reachability count is
  conservative).


### E-BAR-RED · code
**Barrier reductions (`bar.red`, SASS `BAR.RED` + `B2R.RESULT`).**
- `bar.red.popc.u32 d, a{, b}, {!}c` / `bar.red.{and,or}.pred p, a{, b}, {!}c` is a
  `bar.sync a{, b}` whose waiters each contribute predicate `c`, read at arrival. When the
  generation completes, each waiter's barrier result is set from the generation's waiters:
  the number of true contributions (`.popc`), all true (`.and`), any true (`.or`).
- ptxas emits `BAR.RED.{POPC,AND,OR}.DEFER_BLOCKING id[, count], {!}P` followed by
  `B2R.RESULT Rd` (`.popc`) or `B2R.RESULT RZ, Pd` (`.and` / `.or`); the model keeps the
  result per thread until the next `bar.red` and reads it at `B2R.RESULT`. A count read after
  `.and` / `.or`, a predicate after `.popc`, or a `B2R.RESULT` before any `bar.red` is refused
  at run time; `BAR.RED` with register id or count operands is refused at lift.
- Waiters of one generation using different reductions, or `bar.red` together with plain
  `bar.sync`, are reported (violation) and get no result. `bar.arrive` arrivals in a
  `bar.red` generation do not contribute.
- probes/bar_red (2026-11-07, SCALE pilot: ptxas's `bar.red.or.pred` exit vote): CTA-wide
  popc / and / or on barriers 0-2 with plain and negated sources, and counted barriers 3-4
  (count 64) reached by warps 0-1 only. GB300 200/200 scrubbed, equal to the reduction
  computed from the inputs; both front-ends bitwise.
### E-GRID · code
**Exhaustive only for the captured launch:** the grid, block and cluster shape from the case,
with no shrinking. Whether its CTAs are all resident at once is E-RESIDENCY (until
2026-11-04 it was part of this assumption). Exception (E-CLC): in a kernel that uses cluster
launch control, CTAs launch lazily, in any order, and a CTA may be cancelled before it
launches.
- Where: `machine.rs` `Machine::new`, `lib.rs` `build_launch`.
- Status: **argued**. Rung 1d showed one limit concretely: at the fixture's 4 CTAs every CTA
  is resident at once on GB300, so `try_cancel` never succeeds there; the model (which allows
  any launch order) still covers the success paths that appear on hardware only at thousands
  of CTAs.
- If wrong: *miss* of bugs that need a larger grid.
- Retire: certified cutoff / symmetry arguments over grid sizes (docs/design.md §3.5).
- "No other kernels" matters to the L1 model too: a co-resident kernel's loads fill lines on
  shared SMs (E-L1-CLOSED, E-L1-PLACE; opt-outs E-L1-CORES and E-L1-HAVOC, 2026-11-04).

### E-RESIDENCY · code
**Co-residency of a launch's CTAs.** Default model: every CTA of a launch is resident at
once (no SM limits, no other kernels). Opt-in limited-residency model (`ptxv check
--residency N`, case `residency = N`, `PTXV_RESIDENCY=N`): at most N CTAs per GPU hold a
residency slot. Every CTA starts unlaunched; a cluster becomes resident as a unit (the
hardware co-schedules a cluster's CTAs) in **any order**, whenever the GPU's resident CTAs
leave room for it; a CTA frees its slot once all its threads exited and none of its async
operations is pending. A state where every resident CTA is blocked and a CTA is left without
a slot is a *residency deadlock*: "CTA k never becomes resident (E-RESIDENCY): N of at most
N CTAs resident on GPU g, all blocked" (finding `deadlock`, signature `deadlock:residency`).
- Why any order: CUDA does not specify dispatch order; `probes/clc` saw CTAs start out of
  order. A kernel that is only safe under in-order dispatch (CTA i waits for CTA i-1 by
  `ctaid`) is reported; CUB's single-pass scan takes dynamic tile ids for this reason
  (fixture e_resid `chain` vs `ticket`).
- **The minimum residency** (`ptxv check --residency auto`, `residency::analyze`): the
  residency bounds R (CTAs per GPU, from one launch unit up to every CTA) that deadlock, and
  the smallest R from which every bound is deadlock-free. Printed as "deadlock-free iff >= N
  co-resident CTAs per GPU" (or "for", when some smaller bound is deadlock-free too), with
  whether the launch guarantees N:
  - one launch unit (a single cluster): guaranteed;
  - a cooperative launch (case `cooperative = true`): guaranteed, since the driver refuses a
    grid that cannot be co-resident;
  - otherwise **not guaranteed**: other kernels (streams, MPS, green contexts, a PDL
    neighbour) can hold SMs. The report says whether N fits an idle GPU: SMs (152 on the dev
    box) x CTAs per SM, from threads, shared memory (+1 KB reserved per CTA), registers (the
    highest SASS register + 2, rounded to 8; unknown on the PTX front-end) and 32 CTAs/SM.
  - Method, kernels with a certificate (E-WELLSYNC): the *launch-set search*. Every schedule
    performs the certified run's events with its happens-before; an event is enabled by
    earlier events only and nothing disables it. So the events a set L of launched clusters
    performs before nothing moves do not depend on order, and the state after launching L is
    a function of L. The search runs every reachable L once (adding one cluster to a
    quiescent state, then running `run_once`'s policy without launching until nothing
    moves), with its bottleneck b(L), the fewest slots with which L can be reached (min over
    the ways to reach L of the max over steps of resident CTAs + the cluster's size). L
    deadlocks at R when b(L) <= R, L is not terminal, and no unlaunched cluster fits. With
    the search complete (at most 2^14 sets), this is a proof for every R.
  - Method, other kernels: exhaustive exploration with each bound. If an exploration is
    incomplete, the launch-set search over deterministic runs adds the deadlocks it finds;
    those are real (every visited state is reached by an execution), but deadlock-freedom is
    then not proven, and the report says so.
  - Launch groups (`--groups`): per group, from the end state of one deterministic run of
    the earlier groups (as `PTXV_GROUPS_RUN`).
- **Reliance in default verdicts.** A kernel with a poll loop on global or generic memory
  (`residency::cross_cta_waits`: a wait on another CTA or GPU) lists E-RESIDENCY, and when
  some GPU runs more than one launch unit its verdict (explored or certified) carries the
  note "relies on E-RESIDENCY: every CTA is assumed co-resident ...; poll loop(s) at ... wait
  on other CTAs, and a non-cooperative launch does not guarantee co-residency". The
  certificate refuses under limited residency (its progress argument has every CTA
  resident); C-RING's note names E-RESIDENCY.
- Where: `residency.rs` (`with_residency`, `residency_admits`, `launch_unit`,
  `launch_search`, `analyze`, `reliance_note`), `explore.rs` (`actions`: gated launches;
  `local_ample` / `sync_por_state_ok`: no reduction across launches; `clc_eager` off),
  `machine.rs` `Machine::new`, `wellsync.rs` (refusal, notes). DPOR (`PTXV_EXPLORER=dpor`)
  defers to the stateful explorer in this mode.
- Status: **argued**. Evidence: fixture e_resid (grid barrier and ctaid chain need all 4
  CTAs; ticket chain and last-block reduction need 1), exploration per bound and the
  launch-set search agree on every variant and both front-ends
  (`crates/sim/tests/residency.rs`).
- If wrong: default model *miss* (a residency deadlock is assumed away; the note says so);
  limited model *false alarm* if the hardware dispatched in order or kept a cluster's CTAs
  apart (neither is promised), *miss* if a CTA's slot were freed before its async operations
  drain (it is not: they belong to the CTA).
- Retire (default model): `--residency auto` on every case whose verdict carries the note.

### E-TMEM-COTENANT · code
**Opt-in: an adversarial co-tenant shares each SM's tensor memory** (`ptxv check
--tmem-cotenant`, case `tmem_cotenant = true`, `PTXV_TMEM_COTENANT=1`; `tenant.rs`). The
default model gives every CTA a whole SM's 512 columns. The co-tenant (another CTA of this
kernel, a PDL dependent, a kernel on another stream) holds any columns for an arbitrary but
finite time:
- *Placement*: an allocation may land at any free, naturally aligned column run (the
  co-tenant held the runs below while it allocated): explored as a choice of the allocating
  split, between the lowest and the highest free run (`PTXV_TMEM_PLACE=all`: every run;
  placement changes only addresses, and the two ends catch code that assumes column 0). A kernel that assumes column 0 accesses columns it does not own (`violation`, fixture
  e_tmem_cotenant `fixed0`).
- *Waiting*: an allocation that does not fit blocks (E-ALLOC) until the co-tenant releases,
  which it always does, so waiting alone cannot deadlock. A deadlock needs a co-tenant that
  cannot release before this CTA progresses: CTA A holds columns (or its permit,
  E-TMEM-PERMIT) while waiting on CTA B, and B, on A's SM, cannot allocate.
  - Explored: each pair of TMEM-using CTAs of different clusters on one GPU is placed on one
    SM (`Env::tmem_share`, at most 8 pairs): their allocations share 512 columns.
  - Happens-before analysis (`hb_findings`, any launch size): one recorded run gives each
    CTA's allocations, frees, relinquish and exit with vector clocks. For CTAs A, B of
    different clusters on one GPU, a deadlock is reported when B's allocation can come after
    A's (it does not happen-before A's allocation) and A cannot release before it (it
    happens-before the free of A's columns, when the two allocations exceed 512 columns; or
    before A's relinquish / exit, under E-TMEM-PERMIT). With a certificate (E-WELLSYNC) the
    run's happens-before is every schedule's; without one it describes that run.
- *Missing relinquish*: a CTA that allocated and exits without
  `tcgen05.relinquish_alloc_permit` is reported (`contract`, signature `tmem-relinquish`):
  under E-TMEM-PERMIT no other CTA on its SM can allocate until it exits.
- Not modeled: a co-tenant that never releases (then no kernel that allocates is live), and
  placements of more than two CTAs on one SM.
- Where: `tenant.rs`; `exec.rs` `tmem_find`, `tmem_find_group` (partner columns, permit),
  the allocation's placement (`tenant_place`); `explore.rs` `actions` (placement choices).
- Status: **argued**. Evidence: fixture e_tmem_cotenant (good clean; `norelinq` permit
  deadlock + contract; `big` columns deadlock; `indep` contract only; `fixed0` violation),
  pair exploration and the happens-before analysis agree, both front-ends
  (`crates/sim/tests/residency.rs`).
- If wrong: *miss* if three or more co-tenants are needed for a deadlock; *false alarm*
  only through E-TMEM-PERMIT.

### E-TMEM-PERMIT · code
**Co-tenant mode: a CTA's allocation permit blocks the allocations of the CTAs sharing its
SM** from its first `tcgen05.alloc` until it executes `tcgen05.relinquish_alloc_permit`
(SASS `UVIRTCOUNT.DEALLOC.SMPOOL`) or exits. The PTX ISA says only that a CTA may not
allocate after relinquishing; CUTLASS relinquishes right after allocating so that "the next
CTA" on the SM can allocate, and the SASS name points to a per-SM pool count. Case
`tmem_permit = false` checks columns only.
- Where: `tenant.rs` `tmem_permit_blocked`, `hb_findings`; `exec.rs` sets `Tmem::permit`.
- Status: **unvalidated** (no GB300 probe yet: two CTAs pinned to one SM, the first
  allocating without relinquish and spinning on a flag the second sets after allocating).
- If wrong: *false alarm* (the permit deadlocks and `tmem-relinquish` contracts) if the
  hardware lets co-tenants allocate free columns regardless; *miss* if the permit also keeps
  a TMEM-using CTA from becoming resident (then it is a residency effect, E-RESIDENCY).
- Retire: the probe above.

### E-CLUSTER-BAR · code
**`barrier.cluster` is a split arrive/wait barrier over every live thread of the cluster**
(exited threads are not waited for, as for `bar.sync`). A `.release` arrive publishes the
thread's whole history. A `.relaxed` arrive publishes only what the thread's last fences
released: everything before its last full fence (`MEMBAR` / `fence.acq_rel`), and its mbarrier
inits before its last `fence.mbarrier_init` (the CUTLASS cluster-start pattern). Init
visibility travels on a separate clock (`Thread::vis`) that only mbarrier-init checks read,
so B14 is caught without hiding data races.
- SASS: the arrive is always `UCGABAR_ARV` (relaxed; `.release` adds `MEMBAR.ALL.CTA;
  MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` before it), and ptxas emits the `FENCE.VIEW.ASYNC.S` of
  `mbarrier.init` *before* the `SYNCS.EXCH` inits, with or without the PTX fence (rung 1c B14:
  the binaries differ by one NOP). So in a binary the sync unit must order a thread's inits
  before its later relaxed arrive; the SASS model releases every earlier init there.
- Where: `exec.rs` `ClusterArrive` / `ClusterWait` / `cluster_try_complete`.
- Status: **argued** (PTX ISA; ptxas codegen). Rung 1c good kernel bitwise = GB300.
- If wrong: *miss* if exited threads should block the barrier (hang) or if SYNCS inits need
  the fence after them in hardware; *false alarm* if relaxed arrives release more.
- Retire: a probe with a thread exiting before the cluster barrier (like E-BAR-EXIT), and a
  litmus that moves the inits after the fence in SASS (binary patch).

### E-CLC · code
**Cluster launch control.** In a kernel that issues `clusterlaunchcontrol.try_cancel`, every
CTA starts unlaunched and the explorer launches CTAs in any order. A try_cancel response either
fails (`{2, 0, 0, 0}`, allowed at any time) or cancels *any* CTA that has not launched
(`{ctaid.x, ctaid.y | ctaid.z << 16, 1, 0}`); a cancelled CTA never runs. ctaid.z's place is
ptxas 13.0's decoding (get_first_ctaid::y = word 1 & 0xffff, ::z = word 1 >> 16, .v4 = {x, y,
z, 0}; fixtures/r1d_clc3d, both fronts, 2026-11-08). probes/clc3d (16 x 4 x 128 CTAs, three
GB300 runs): every successful response is {x, y | z << 16, 1, 0} of a CTA that never ran,
1966-1968 cancels per run, z up to 127. The response is 16 bytes written to
shared memory plus `complete_tx(16)` on the mbarrier. DPOR: a launch and a cancel of the same
CTA write its launch key, and a response completion reads the keys of CTAs that already
launched (each removed a possible outcome).
- Evidence: `probes/clc` (8192 CTAs, one try_cancel each, two runs): the cancelled CTAs are
  exactly the ones that never ran; victims are not in launch order (caller 0 got 5404, then
  5492, 5260, ...); the response encoding above.
- Where: `machine.rs` `Machine::new` (unlaunched start), `explore.rs` (`Launch`, `Clc` actions),
  `exec.rs` `launch_cta`, `complete_clc`.
- Status: **validated** for 1-D grids without clusters. `fixtures/r1d_clc`: both explorers agree
  (73 terminal states), and at 8192 CTAs GB300 shows the predicted failures (B17 and B53
  launch failures, B54 about 1985 of 8192 tiles lost; `tools/r1d_scale.sh`).
- If wrong: *false alarm* if the hardware never fails a cancel while CTAs remain (the
  model allows it; a correct kernel cannot depend on it either way).
- Multicast (`multicast::cluster::all`, `UGETNEXTWORKID.BROADCAST`, probes/clc_mcast): the
  whole cluster of the victim is cancelled, and every CTA of the issuer's cluster gets the
  same response. A failure is {2, 0, 0, 0} or all zero; the model allows both (2026-11-08,
  4096 CTAs: 376 failed clusters got {2, 0, 0, 0}, 6 got zeros; probes/clc_one_cluster: a grid
  of one cluster always fails, {2, 0, 0, 0} in 50 of 50 runs).
- **Eager launch** (`explore.rs` `clc_eager`): if a deterministic run with every CTA launched
  up front executes no try_cancel (cuBLAS gates its try_cancel on a launch parameter that is
  0), exploration starts with every CTA launched, and any try_cancel that does execute is
  reported "unmodeled". When none executes, launch times do not matter: a late launch is a
  schedule in which the CTA's threads have not stepped yet. Eager is chosen only when that run
  ends with every thread exited (a deadlocked or stuck run gives lazy launch; before
  2026-11-08 it gave eager). A try_cancel in a grid of one cluster executes under eager launch:
  nothing is left to cancel in any schedule, so the response is a failure (the Triton 2-CTA
  matmul, grid = one cluster of 2). The certificate (`wellsync.rs`
  `certify_inner`, since 2026-11-06) runs on the eagerly launched machine directly instead of a
  separate trial run: a run that executes a try_cancel stops, and the certificate starts over
  with every CTA unlaunched. The decision is the trial's (same deterministic run).

### E-CPASYNC · code
**`cp.async` (SASS `LDGSTS`) reads its global source at issue and writes shared memory when
`cp.async.wait_group n` retires its group** (at most n of the thread's committed groups stay
incomplete; `LDGDEPBAR` commits, `DEPBAR.LE SBk, n` waits). The write is race-checked with the
thread's clock at issue: nothing the thread did after issuing orders it. A source predicate
that is false (src-size 0) writes zeros and reads nothing.
- Where: `exec.rs` `Op::CpAsync` / `CpAsyncCommit` / `CpAsyncWait`, `Thread::cp_pending`;
  `sass` lowering of `LDGSTS`, `LDGDEPBAR` and the `DEPBAR.LE` post-pass; `ptx` lowering of
  `cp.async.{ca,cg}` (src-size 0 or an ignore-src predicate: zero fill), `commit_group`,
  `wait_group`, and `wait_all` (= commit_group + wait_group 0).
- Scoreboard scope (2026-11-07): on the SASS front `DEPBAR.LE SBk, n` waits on the warp's
  scoreboard, which counts the warp's LDGDEPBAR executions (divergent paths count once each):
  every lane's copies older than the warp's newest n commits complete, and the waiting lanes
  are ordered after them (`Warp::cp_seq`, `CpPending::wgroup`). The per-thread rule (PTX
  semantics) still applies on top. Before, only the waiting thread's own copies counted:
  a false race in CUDA C++ `cp.async.wait_group; __syncwarp()` (ptxas: `DEPBAR.LE SB0, 0x5;
  NOP; LDS`, scale dash/6393; GB300 1000/1000 scrubbed runs = simulator). probes/cpasync_warp:
  odd / even lanes copy on two divergent paths, one converged DEPBAR, each lane reads its
  neighbour's slot: GB300 1000/1000 scrubbed runs correct for wait_group 0 and 1 (the second
  group also landed in time with 1, so the probe does not show the counting itself).
- src-size (2026-11-07): an immediate or register src-size s copies the first s bytes and
  zero-fills the rest (s larger than the cp-size: a violation). ptxas encodes a partial s as
  `LDGSTS...ZFILL` with source address + size - s (the address's low bits mod the size are the
  zero count) and a predicate false for s = 0; Triton's masked loads use s = 16 / 0.
  probes/cpasync_srcsize (16-, 8-, 4-byte copies, s from registers and an immediate): GB300
  200/200 scrubbed runs, both front-ends bitwise. Before, `.ZFILL` was admitted and ignored
  (16 bytes read from the shifted address) and a register src-size was not lowered.
- Status: **validated**: `fixtures/s_sm80` (two groups, wait_group 1 then 0, zero fill),
  torch's sm80 flash attention (3-stage pipeline) and Triton's flash attention (masked
  prefetch past the end) match GB300 bitwise. A read of the destination before the wait sees
  the old bytes in the model; on hardware it may see either: reported as a race only when
  another thread's access is involved.
- If wrong: *miss* for a thread that reads its own cp.async destination before waiting.
- Scoreboard reuse (2026-11-05, mslk's FA4 decode): ptxas may count cp.async groups
  (LDGDEPBAR's write scoreboard) and bulk groups (UTMACMDFLUSH's read scoreboard) on the same
  SBk. Each `DEPBAR.LE SBk, n` is classified by the kinds that may be outstanding on SBk where
  it is reached (`sass` `depbar_kinds`, a path-insensitive dataflow; `DEPBAR.LE SBk, 0`
  clears SBk): one kind, that wait; both, both waits with the same n. The scoreboard holds
  at most n groups of both kinds together, which implies at most n of each, so completing
  only that much is weaker than the hardware's wait (sound; exact for n = 0). Until
  2026-11-06 both with n > 0 was not lowered (CUTLASS grouped GEMM, Triton grouped dgrad).
  Before 2026-11-05 any DEPBAR on such an SBk was a bulk wait, the cp.async never landed,
  and the kernel silently read zeros.
- A thread that exits with a cp.async it never waited for (no wait_group / mbarrier arrive
  retired it): E-CPASYNC-EXIT.

### E-CPASYNC-MBAR · code
**`cp.async.mbarrier.arrive.shared.b64 [m]` makes the mbarrier track every `cp.async` the
thread issued so far: their shared-memory writes land, as one asynchronous operation, before
it acts on m.** ptxas compiles the plain form to `SYNCS.ARRIVE.TRANS64.RED.A0T1` (expect one
transaction unit now) plus `ARRIVES.LDGSTSBAR.64.TRANSCNT` (complete that unit once the copies
landed), and `.noinc` to `ARRIVES.LDGSTSBAR.64.ARVCNT` (one arrival once they landed); both are
modeled that way (`Op::CpAsyncMbarArrive`, `AsyncKind::CpArrive`, a scheduler choice like a
TMA completion). The writes are race-checked as the operation's actor, whose clock a waiter
on m acquires. The copies leave the thread's cp.async groups: a later `wait_group` does not
wait for them again (no kernel seen does both).
- Where: `sass` `ARRIVES` / `SYNCS.ARRIVE...A0T1` lowering; `ptx` `cp.async.mbarrier.arrive`
  (the same two operations for the plain form); `exec.rs` `Op::CpAsyncMbarArrive`,
  `complete_cp_arrive`; `explore.rs` (pending arrivals, look-ahead: `.noinc` gives up).
- Status: **validated**: fixtures/s_cpasync_mbar (both forms, one CTA, the consumer reads
  both buffers after waiting) = GB300 1000/1000 and its recorded bytes; fixtures/s_ptx_cute_ops
  (both forms, 64 threads, through the PTX front-end too) = GB300 bitwise; trtllm-gen's scale
  loader (fixtures/r3b_trtllm_bmm, the TRANSCNT form) = GB300 bitwise.
- If wrong: *miss* for a consumer that reads before the copies land (the model would order it).

### E-PDL · code
**Programmatic dependent launch.** A case may launch several kernels on one stream
(`[[launches]]`); the model runs them as one machine with the grids' code concatenated. A later
grid's CTAs start unlaunched. Without PDL they may launch once the previous grid has
*completed* (every thread exited, every async operation done), and then everything that grid
did happens-before them. With PDL (`pdl = true`) they may launch once every CTA of the previous
grid has executed `griddepcontrol.launch_dependents` (one thread of the CTA suffices) or has
exited entirely, and they get **no** ordering from it. `griddepcontrol.wait` blocks until the
previous grid has completed and then acquires everything it did; in the first grid it is a
no-op. The trigger orders no memory. Only the immediately preceding grid *on the same GPU* is a
prerequisite. A grid on a GPU no earlier grid ran on (`device = N`, rung 1f) is launched
independently: it may launch at any point, with no ordering against the other GPUs' kernels
(arbitrary launch skew).
- Evidence: `probes/pdl` (GB300, 20 runs per experiment). With the attribute, a dependent
  without `wait` runs while the primary is still running after every CTA triggered (19/20; one
  launch came late) or when one thread per CTA triggered (20/20). It never runs early if any
  CTA neither triggered nor exited, or without the attribute (0/20). After `wait` the
  primary's last write is always visible (all experiments).
  `launch_dependents` = `PREEXIT`, `wait` = `ACQBULK`.
- Where: `machine.rs` `Grid`, `Machine::new`; `exec.rs` `grid_gate_open`, `grid_complete`,
  `grid_done_vc`, `launch_cta`; `explore.rs` (gated `Launch` actions, expanded alone in
  `local_ample`; the synchronization reduction once one grid is left, `sync_por_state_ok`);
  `dpor.rs` (footprint space 9).
- Status: **validated** for a two-kernel chain on one GPU (rung 1e: `fixtures/r1e_pdl` is
  bitwise = GB300, B18 is reproduced with a witness delay, and the same code launched without
  PDL (`fixtures/r1e_serial`) is clean and stays correct under the delay).
- If wrong: *miss* if the hardware gave a later grid less ordering than modeled (e.g. a
  dependent launched before a CTA that neither triggered nor exited). Writes before the
  trigger were visible early in every probe run, but the model treats them as unordered, as
  the PTX ISA does: a dependent that relies on them is reported, even if GB300 never shows it.

### E-LAUNCH-GROUPS · code
**A multi-launch case may be checked one group of launches at a time** (`ptxv check
--groups SPEC`, `groups.rs`). The launches are partitioned into groups G1 < ... < Gn.
`SPEC` lists groups 2.. (`"2,3;4,5;6-18,21-25"`, group 1 is the rest), or is `auto`
(certify the whole case; on refusal, move the launches the refusal names to a later group).
- Conditions (checked, `validate`):
  - the groups are closed under stream order (a launch's predecessor on its GPU is in its
    group or an earlier one);
  - no PDL edge crosses groups.
- Procedure:
  1. Group i runs on the machine group i-1 left, with the launches of later groups held back:
     their threads count as exited, as in a slice.
  2. It is decided by the certificate (E-WELLSYNC, weak then strong LDS/STS model), else by
     exploration.
  3. Before group i+1 starts, its launches are put back in their initial state. Every
     access in the global race shadow is *demoted* to plain, so S-SCOPE's morally-strong
     exemption no longer covers it. Group i's *obligations* are added to `Machine::xg`
     (accumulated over groups). An obligation is an earlier access (epoch, write or read)
     that every later access, for a write, or every later write, for a read, must happen
     after. A violation is a `race` finding of the later group.
     - A certified group's obligations are all of its strong reads of global memory, and
       every strong write that another strong write of the same word does not order
       (commutative atomics, C-FADD, CAS links).
     - An explored group continues from the *meet* of its terminal states (`Meet`). That is
       one state whose thread clocks, completed async operations and synchronization sources
       are pointwise minima over the terminal states (a source is kept only where all agree).
       Its race shadow is the first terminal state's. Every byte whose shadow differs between
       terminal states contributes every terminal state's last write and reads as
       obligations.
       - *Final clocks.* An actor of the group (thread, or async unit whose completions are
         folded into its CTA's clocks) is known to the next grid on its GPU at its *largest*
         final clock over the terminal states (`Meet::finish`). Knowing an actor's final
         clock means knowing all of its events, whatever their numbering in one execution or
         another: dispatch's receivers issue a schedule-dependent number of bulk copies, so a
         minimum would name an event that is not the last one. Partial knowledge (other
         threads' clocks, synchronization sources) stays the minimum.
       - *Exit checks.* Completed async operations of finished CTAs leave `completed`, folded
         into their CTA's clocks: their exit checks ran at the group's end.
     - A group whose terminal states differ in any global byte, that is incomplete, or that
       has findings stops the procedure (later groups would need one continuation per final
       memory: not implemented).
- Argument. Let E be any execution of the whole case. We show that E is equivalent (same
  per-thread steps, findings and final memory) to E1 · E2 · ... · En, where Ei is an
  execution of group i from the state the earlier groups leave. Equivalently: for any
  conflicting accesses x of group i and y of group j > i, x precedes y in E. Then
  non-conflicting steps of different groups commute, and the only other contacts are
  excluded: synchronization objects are per CTA, and a later group's launch gate depends
  only on its own GPU's earlier launches, which are in its group or earlier (stream
  closure); without PDL those cannot wait on later groups.
  - Why x precedes y. The check runs on y in the continuation of group j and compares it
    against the shadow and the obligations.
    - The race shadow of a byte (`Sh`) holds its last write, the reads since, and every
      earlier access that write does not *cover* (`Sh::hw`, and reads kept in `Sh::r`; see
      E-WELLSYNC "Detector precision"). Every earlier access of the byte is in the shadow,
      HB-before a write in it that covers it, or is an obligation:
      - plain accesses are ordered with every write, else the group reported a race. Until
        2026-11-05 a strong write replaced the shadow outright, so a plain write followed by
        its own thread's atomic was dropped while another thread's atomic was still
        unordered with it (soundness-harness miss 1); now such an access stays in the
        shadow, is demoted with it and checked against the later groups;
      - strong reads need not be ordered before a later write (a poll that read an earlier
        value), so they are obligations;
      - unordered strong writes are obligations.
    - For an explored group, the shadow and obligations together cover the frontier (the
      shadow's writes and reads) of *every* terminal state. Under E-DRF the reduced search
      reaches every terminal state of the full one: equivalent executions end in the same
      state, the shadow included.
    - Any access x of group i in E either is in the frontier of E's projection on group i, or
      precedes that frontier's last write in time. y happens after the frontier, so y
      follows x.
    - Clocks: the continuation's clocks are lower bounds of every terminal state's (a
      certified run's clocks are lower bounds of every schedule's, C-WAIT), and propagation
      is monotone. An epoch (actor, clock) in an obligation comes from one terminal state.
      If the continuation proves y knows it, then in executions whose projection is that
      terminal state, y really knows it. In the others the obligation is vacuous or covered
      by their own frontier.
  - Values. Group j reads, for every byte an earlier group wrote, the earlier groups' final
    value: every conflicting earlier write precedes the read. That value is unique: a
    certified group's final memory is schedule-independent, and an explored group must
    agree in all terminal states. So exploring group j from the meet is exploring all of
    group j's behaviours in E.
- Multi-grid partial-order reduction. Several grids live at once (one per GPU in a
  collective's group) used to switch the synchronization reduction off (`sync_por_state_ok`,
  E-PDL). It now runs once nothing is left to launch and no live PDL grid waits on a running
  predecessor. The state is then a closed set of CTAs whose only contact is global memory:
  named in the footprints (strong accesses, spaces 10/11) or a race (E-DRF), as between the
  CTAs of one grid. `PTXV_NO_MULTIGRID_POR` restores the old rule. Effect: DeepEP
  `notify_dispatch` on 2 GPUs went from > 8M states (incomplete) to 24,329 (complete).
- Status: **argued**. Evidence below (docs/rung4.md, rung 4b "Launch groups").
- If wrong: *miss* of a cross-group interaction that the obligations do not cover.
- Where: `groups.rs`; `exec.rs` `access_x` (`xg`); `explore.rs` (`terminal_sink`,
  `sync_por_state_ok`); `wellsync.rs` (`certify_keep`, `carry`).

### E-POLL-ACQ · code
**Opt-in: a thread that leaves a warp-level poll loop acquires what its polls observed**
(case `poll_acquire = true`; `PTXV_POLL_ACQUIRE=0|1` overrides; since it is refuted, either
one is refused unless the case also opts in explicitly with `assume_refuted = ["E-POLL-ACQ"]`,
`ptxv check --assume-refuted E-POLL-ACQ` or `PTXV_ASSUME_REFUTED=E-POLL-ACQ`, and the verdict
names the opt-in: `refuted_opt_in` in `--json`, a note and a warning in the text), as if the loop ended with
an acquire fence scoped by the loads (`acquire_pending`, like `CCTL.IVALL`). Only the
warp-level polls of E-SPIN (`spin::wpoll_loops`) qualify; a straight-line poll keeps the
strict model, so a missing `CCTL.IVALL` after DeepEP's channel-tail polls is still seen.
- Why it exists (rung 4b): DeepEP's `barrier_block` signals with `MEMBAR.ALL.SYS` +
  `ATOM.ADD.STRONG.SYS` (a release) but waits with relaxed `ld.volatile` polls, a vote and
  `__syncthreads`, and no fence. Under the strict PTX model every cross-rank handoff through
  it (per-rank token counts in `notify_dispatch`, NVLink buffer reuse between dispatch and
  combine) is a race (docs/rung4.md, T4b).
- Status: **refuted** (GB300 litmus, 2026-11-04, `docs/poll-acq-litmus.md`,
  `probes/poll_acq`). After DeepEP's exact loop shape (`LDG.E.STRONG.SYS` polls, `VOTE.ALL`,
  clock timeout, optional `BAR.SYNC` with a reader warp, no fence), a weak read (`LD.E`,
  `LDG.E`, `LDG.E.CONSTANT`) returns the pre-release value whenever the payload line is valid
  in the reader SM's L1. This held in every cross-SM placement: another SM of the GPU, and
  NVLink in both directions with the payload local or remote: 476,705,632,772 stale of
  701,986,992,896 cross-SM weak reads in 220 runs (99.99-100% without stress, 15-70% under L1
  thrash). A `fence.acq_rel.sys` after the loop (negative control), volatile reads, or an L1
  without the line removed all of it. The cause is L1
  non-coherence, not reordering: the old argument (in-order issue, control dependence) does
  not address it. With producer and consumer on one SM nothing was stale.
  - DeepEP itself never read stale data in its compiled shapes (0 stale in 3,538,947,554,048
    reads, 2.27e12 cross-SM, 90 runs). Two things protect it: the barrier's entry `fence.acq_rel.sys` (`CCTL.IVALL` *before* the poll) with no payload load before the
    reads, and combine's volatile `LDG.E.128.STRONG.SYS` reads. The entry fence only works
    while nothing on the SM reloads the lines: with other warps loading them during the
    barrier, 160,397,332 of 4.07e10 reads (0.07-0.8% per placement) were stale.
  - Consequence: the strict model's T4b reports (docs/rung4.md) are real PTX-model races,
    latent on GB300 for the reason above. Verdicts that list E-POLL-ACQ can miss stale reads
    in other code. fixtures/r4b_moe no longer sets it (2026-11-04): its verdicts are the
    strict model's, and the two-level check (E-HW-L1) classifies the T4b races as latent
    (docs/deepep-repro.md, "Two-level verdicts").
- If relied on: *miss* of a stale read after such a loop whenever the payload line is in L1.
- Where: `exec.rs` `poll_exit_acquire`.

### E-SMEM-CUMUL · code
**Opt-in: a SASS shared-memory store releases, at CTA scope, what its thread acquired**
from other threads (not its own earlier writes; case `smem_cumulative = true`,
`PTXV_SMEM_CUMULATIVE=0|1`). The store's synchronization clock is its release clock joined
with `Thread::acq_view`, the join of everything the thread acquired.
- Why it exists (rung 4b): DeepEP combine's coordinator lanes acquire each channel tail
  (`LDG.E.STRONG.SYS` + `CCTL.IVALL`) and relay it to the receiving warps through
  `st.volatile.shared` with no fence; the warps poll it and then read the token data. Under
  the strict model the receivers' reads are not ordered after the senders' writes.
- Status: **validated** (GB300 litmus, 2026-11-04, `docs/poll-acq-litmus.md`,
  `probes/poll_acq`) for the shape DeepEP uses. The coordinator lane runs
  `LDG.E.STRONG.SYS; CCTL.IVALL; STS` (`ld.acquire.sys` + `st.volatile.shared`); the receiver
  polls with `LDS` + `VOTE.ALL`, then reads the remote data. 0 stale in 5,282,878,210,816
  cross-SM reads (150 runs of about 30 s): cross-GPU over NVLink both ways with the payload
  local or remote, and on other SMs of one GPU. The receiver read weakly
  (`LDG.E`, `LD.E`), volatile (`LDG.E.128.STRONG.SYS`, combine's form) or `.nc`. Its L1 held
  the payload lines from earlier reads or reloaded them during the poll, without stress and under five stress
  modes, with zero stale reads. The positive control is the same relay with a relaxed poll
  and no `CCTL.IVALL`: stale in every cross-SM placement and stress cell (6-100%). So the
  hardware mechanism is that `CCTL.IVALL` invalidates the whole SM's L1. The rule holds only
  when the thread's acquisition was a real acquire. It does not hold for knowledge
  "acquired" through E-POLL-ACQ (refuted). Same-SM runs (producer in the same CTA) were
  never stale, not even the controls, and are not counted. Litmus evidence, not proof.
- Where: `exec.rs` `sync_publish`, `sync_observe`, `acquire_pending`.

### E-HW-L1 · code
**Opt-in: GB300's per-SM L1 is not coherent** (case `hw_l1 = true`; `PTXV_HW_L1=0|1`
overrides; `crates/sim/src/l1.rs`). A weak global load (`LD.E`, `LDG.E`, `LDG.E.CONSTANT`;
PTX `ld`, `ld.global`, `ld.global.nc`) goes through its SM's L1: a hit returns the bytes the
line was filled with, a miss fills the 128-byte line from memory. `CCTL.IVALL` (SASS; PTX
`fence.acq_rel` / `fence.sc` and acquire loads at gpu / sys scope) drops every line of the
SM; lines never survive into another grid (kernel boundary). Strong loads (`.STRONG.*`,
volatile, acquire) neither hit nor fill. The SM's own stores update its copy; its atomics drop
the line; bulk / TMA loads bypass L1. Each CTA is its own SM unless `PTXV_L1_COLOCATE`
groups CTAs (E-L1-PLACE; nothing outside the case fills lines, E-L1-CLOSED; both opt-out:
E-L1-CORES, E-L1-HAVOC). Findings have kind `stale`: `stale-l1:<pc>` (the explored schedule read a stale
line; the load returns the stale bytes) and `stale-l1-pred:<R>:<F>` (a predictive check: the
last write W of the bytes R reads is by another SM and happens-before R, a weak load F of the
line on R's SM is not ordered after W, and no invalidation on that SM is ordered after F or W
and before R). The predictive check reasons pairwise and keeps, per SM and line, the first and
last two loads of each thread and the last four invalidations of each thread, so it can both
over- and under-report in principle; a witness should be confirmed by a schedule or on the
GPU.
- Why it exists: E-POLL-ACQ is refuted on GB300 by stale L1 hits (docs/poll-acq-litmus.md);
  DeepEP's `barrier_block` is safe there only because of its entry `CCTL.IVALL`
  (docs/deepep-repro.md). The model decides which of the strict model's races can return
  stale data on this hardware.
- Status: **probed**. Litmus results: a weak read after a fenceless poll is stale when the
  line is valid in the reader's L1, never when an acquire (`CCTL.IVALL`) intervenes, and the
  `CCTL.IVALL` of any thread of the SM covers the whole SM (E-SMEM-CUMUL rows); same-SM
  producers are never stale (docs/poll-acq-litmus.md). Kernel boundaries invalidate:
  probes/l1_kbound (0 stale in 1.56e10 reads after a boundary, 100% of 6.2e9 without one). Not probed:
  whether strong loads fill L1, whether atomics update or drop the line.
- **Known gap (probed 2026-11-04, probes/notify_coreader): concurrent kernels do share L1
  lines.** The rule "a line from another grid never hits" is right for consecutive kernels
  but not for grids running at the same time on one SM. A co-reader kernel's fill was hit by
  the victim kernel in 20.6% of trials with NVLink traffic toward the victim GPU (reader
  launched after the victim). Same-kernel co-readers: 100%. The model therefore under-reports
  co-resident-kernel scenarios; agent Z's `l1_coresident` / `l1_havoc` options cover them.
- **Two-level verdicts** (`twolevel.rs`, on by default in `ptxv check`, `--no-gb300` /
  `PTXV_GB300=0` off; `PTXV_HW_MODEL=1` checks under the hardware model directly). When a
  verdict has a race or stale finding, or relies on a refuted assumption, the check runs again
  under the *hardware model*: this L1 on every SM, over sequentially consistent memory
  (`Launch::hw_order`: every strong write publishes its writer's clock and every strong read
  that observes it acquires that clock, at any scope; strong accesses never race with each
  other). That is the ordering half of E-POLL-ACQ (after a poll exits, an L1 miss returns the
  new data; litmus: a fill after the poll was never stale), not its acquire half: stale lines
  stay until an invalidation. Store reordering by a missing release fence is not modeled
  (r4b B33 / B39 were latent on GB300). The hardware model also hashes the predictive check's
  logs ((thread, pc) per entry) into the state, so that deduplication cannot merge away the
  path on which a load filled a line early (l1_litmus pr3). Each level-1 race / stale finding
  gets `gb300`: *observable* if the hardware model reports the same race (the accesses are
  unordered even with sequentially consistent memory) or a stale read at one of its reads;
  *latent* otherwise, with the reason from per-pc L1 statistics of the level-2 run (no fill of
  the line on that SM since the last invalidation; only fills after the write; the read is a
  strong / bulk load); *undecided* when no match and level 2 was not exhaustive. The hardware
  model is closed-world (E-L1-CLOSED), so level 2 also runs the havoc check (E-L1-HAVOC) as a
  probe in the same run (`L1::havoc_probe`: reports kind `probe`, which no explorer, launch
  group or verdict class acts on; `hw_launch` turns a level-1 `l1_havoc` off so placements stay
  explored): *latent* splits into **latent** (no fill, not even an outside one, exposes it)
  and **latent-closed-world** (the probe flags a read involved: observable once a co-resident
  kernel fills the line; DeepEP notify_dispatch's 0x15f0 / 0x1600 / 0x16d0 / 0x16f0, reproduced
  on GB300 with a co-reader, docs/deepep-coreader.md). Probe witnesses at reads no level-1
  finding covers (a case relying on a refuted assumption) are listed as extra
  latent-closed-world reads. Litmus agreement (crates/sim/tests/fixtures.rs
  `twolevel_l1_litmus`): a hardware-model stale read exactly in pr0-payld, pr2, pr3
  (GB300 stale) and pr0-lineshare (the co-reader shape), none in pr0, pr1, neg, neg-payld (GB300 0 stale; pr0 / pr1 are
  latent-closed-world: pr0-lineshare is pr0 plus a line-sharing reader); r1f B20 observable.
- If wrong: *miss* (a stale hit the model drops) or *false alarm*. For the two-level
  classification: a finding called latent that GB300 can expose through a mechanism outside
  the model (store reordering, L2 effects), or observable through an over-approximation of
  the predictive check.
- Where: `l1.rs`, `l1_cores.rs`; `exec.rs` (`Op::Ld`, `Ld256`, `St`, `Atom`, `Fence`; `sync_publish`,
  `sync_observe_cell`, `morally_strong` for `hw_order`); `twolevel.rs`.

### E-HW-WEAK-FLAG · code
**GB300 hardware model (two-level check only): an aligned weak global store of at most 8
bytes, after a sys-scope fence of its thread, is a relaxed sys store** (`Launch::hw_order`;
`exec.rs` `Op::St`, `sync_publish`). It is single-copy atomic, morally strong with strong
accesses (no race with a strong poll), and publishes the thread's clock at that fence
(`membar_vc`), not its whole history. Level-1 (PTX) verdicts are unchanged: there the store
is weak and races.
- Why it exists: msl DeepEP's `nvlink_barrier_wo_local_sync` (index L5) signals peers with
  a plain `ST.E` after `MEMBAR.SC.SYS`; the peer polls with `LD.E.STRONG.SYS` and ends with
  `MEMBAR.SC.SYS; CCTL.IVALL`. The hardware model classified the flag races, and the
  downstream races in DeepEP combine (bulk writes waited by `wait_group 0`, then published
  through this barrier), as observable, because only strong writes published.
- Status: **validated** (`probes/weak_flag`, GB300, 2 GPUs over NVLink, 2026-11-07). The
  writer stores 256 payload words per slot (weak) into the peer GPU, then `fence.sc.sys` and
  a weak 32-bit flag store. The reader polls `ld.volatile`, then `fence.sc.sys`, then reads
  the payload weakly with its L1 warmed with the old values. 4096 slots per run.
  - Weak flag: 0 stale words in 500 runs.
  - Weak flag with a concurrent NVLink co-tenant under MPS: 0 in 300 runs.
  - `st.relaxed.sys` flag: 0 in 500 runs.
  - Controls: without the writer fence, 26,688 stale words in 343/500 runs (40,512 in 42/300
    under the MPS co-tenant); without the reader fence, every word is stale.
- Scope: only after a sys-scope fence (`fence.sc.sys` / `fence.acq_rel.sys`, SASS
  `MEMBAR.SC.SYS` / `MEMBAR.ALL.SYS`); without one the store stays weak, as the control
  requires. Not covered: wider or misaligned stores, shared memory, multicast.
- If wrong: *miss* of a stale read behind a weak flag (a race called latent).
- Where: `exec.rs` `Op::St` (`weak_flag`), `sync_publish` (`weak_pub`); `twolevel.rs` MODEL.

### E-HW-CORW · code
**GB300 hardware model (two-level check only): a write by a thread to bytes whose last read
by that thread was a strong load returning the last write's bytes is coherence-after every
earlier write of those bytes** (`Launch::hw_order`; `exec.rs` `access_x`, `Epoch::sv` on
reads). Level-1 (PTX) verdicts are unchanged: the write-write race stays a PTX race.
- Why it exists: msl DeepEP hybrid dispatch polls a NIC-written slot with `ld.acquire.sys`
  and clears it with a plain `STG` (docs/nic.md finding L). The hardware model called the
  NIC write / clear pair observable, although the clear cannot be ordered before a write
  its own thread already read.
- Argument: E-HW-L1 memory behind the L1s is sequentially consistent and a strong load
  bypasses L1, so the load returns the memory value, the last write in the explored order;
  every write of the byte's shadow precedes it in coherence order. Read-to-write coherence
  of one location (CoRW, single-copy atomicity per byte) puts the thread's later write after
  them. Only write-write pairs are dropped; the read's own races and other threads' reads are
  still checked.
- Status: **argued** from E-HW-L1 (coherence per location); no separate probe.
- Scope: the read must be strong (a weak read may return stale L1 bytes) and the thread's
  latest read of the byte (a later plain read of it replaces the mark); any write to the byte
  in between clears the mark.
- If wrong: *miss* of a final-value race (a write-write race called latent).
- Where: `exec.rs` `access_x` (`corw`), `machine.rs` `Readers::forget_rf`; `twolevel.rs`
  MODEL.

### E-L1-CLOSED · code
**Closed world for L1: no agent outside the case loads case buffers on an SM the case uses.**
Implicit in E-HW-L1 until 2026-11-04; listed in every `hw_l1` verdict without `l1_havoc`.
- Why it matters: on GB300 a weak load returns stale bytes only from a stale copy of its line
  in its SM's L1, and some load on that SM must have made the copy. In the case, only the
  case's own loads fill lines. A kernel outside the case that runs at the same time (another
  stream, another process under MPS) and reads *other bytes* of the same 128-byte line is not
  a data race in the PTX model and is legal CUDA, yet its fill can leave a copy that a racy
  read of the case hits (docs/deepep-coreader.md).
- Status: **argued** for a case that is the whole program on its GPUs; false whenever user
  code reads case buffers concurrently (DeepEP's `get_local_buffer_tensor`).
- If wrong: *miss* (a stale read through an outside fill).
- Retire: `l1_havoc` (E-L1-HAVOC), or model the outside reader as a launch of the case with
  `l1_coresident` (E-L1-CORES).
- Two-level verdicts (E-HW-L1): the level-2 hardware model is closed-world too, but runs the
  havoc check as a probe (`L1::havoc_probe`, findings of kind `probe`), so no finding is called
  latent on E-L1-CLOSED's strength alone: one whose read the probe flags is
  **latent-closed-world** (reason names E-L1-CLOSED, witness `stale-l1-havoc:<pc>`).
- Where: `l1_cores.rs` `l1_havoc_check` (the check it switches off); `assumptions.rs`
  `for_machine`; `twolevel.rs` (the probe).

### E-L1-PLACE · code
**L1 placement: each CTA is alone on its SM** (or grouped by `PTXV_L1_COLOCATE`). CTAs that
share an SM share its L1 lines, so a fill by one can serve a stale copy to another; a CTA alone
only meets its own fills. Listed in every `hw_l1` verdict without `l1_coresident` or `l1_havoc`.
- Status: **unvalidated**: the hardware places CTAs of one or several kernels on one SM
  whenever the limits allow (DeepEP notify_dispatch's CTAs are 128 threads).
- If wrong: *miss*.
- Retire: `l1_coresident` (placements explored, E-L1-CORES) or `l1_havoc`.
- Where: `l1.rs` `l1_sm_of_cta`, `L1::new`.

### E-L1-CORES · code
**Opt-in: co-resident launches share SM L1s, and the placement is a nondeterministic choice**
(case `l1_coresident = true`, `PTXV_L1_CORES=0|1`; implies `hw_l1`; `crates/sim/src/l1_cores.rs`).
- Model:
  - launches that are not stream-ordered (a launch with `after = -1` is on another stream)
    may be resident together; a PDL edge counts as unordered;
  - a line filled by one grid hits for another grid's load on the same SM when the two may be
    resident together (`l1_shared`); a CTA's launch drops the lines other grids left on its SM
    (the kernel boundary of probes/l1_kbound, taken as happening at launch on each SM);
  - the placement (which CTAs share an SM) is enumerated by `explore_placements`:
    `PTXV_L1_PLACE=pairs` (default: every CTA alone, plus each pair of CTAs that may share an
    SM), `all` (every partition within the limits, capped at 4096) or `none`;
  - limits per SM: same GPU, grids not stream-ordered, at most 32 CTAs, 2048 threads and
    228 KB of shared memory (1 KB reserved per CTA). Registers are not known and not checked,
    which only adds placements;
  - DPOR records two footprint spaces so that it orders fills against changes of the line and
    of the SM's copy: 20 (line contents: loads read, stores / atomics write) and 21 (an SM's
    L1: loads read their line's key, invalidations, atomics and CTA launches write all). The
    stateful explorer does not schedule weak loads (E-DRF); the predictive check of E-HW-L1
    covers it, now also across co-resident grids.
- Pairs suffice to decide whether some placement reaches a stale read: take the first stale
  read R of a schedule under any placement, served by a line that CTA F filled. Under the
  placement with only F and R's CTA together, each earlier load on those SMs sees a subset of
  the invalidations, stores and fills it saw before, so it returns the same bytes or is itself
  a stale read (an earlier one), and otherwise F's fill survives to R. The pairs do not list
  every stale read of every placement.
- Status: **argued**. Not probed: whether a kernel launch drops lines on every SM or only on
  the SMs it runs on, and what happens at a co-resident kernel's completion (the model drops
  nothing there, which only keeps more lines: a *false alarm* direction).
- If wrong: *miss* if the hardware shares lines in a way the model does not (e.g. lines that
  survive a launch), *false alarm* if a launch or completion drops more than modeled.
- Where: `l1_cores.rs`; hooks in `l1.rs` (`l1_load` hit test, predictive check),
  `exec.rs` `launch_cta`; `ptxv check` wraps the explorer (`explore_placements`).

### E-L1-HAVOC · code
**Opt-in: any agent outside the case may fill any line of a case buffer on any SM at any
time** (case `l1_havoc = true`, `PTXV_L1_HAVOC=0|1`; implies `hw_l1`). A weak global load R is
reported (`stale-l1-havoc:<pc>`, kind `stale`) when the last write W of bytes it reads is by
another SM, W's grid is not stream-ordered before R's grid and W does not happen-before the
completion of the grid R's grid waits for (either way R's launch drops every older copy), and no L1 invalidation I on R's SM satisfies W -> I -> R in happens-before. A fill
placed after the last invalidation that precedes R and before W lands then leaves a copy
without W's bytes, and R hits it. This answers "is this read vulnerable to anything
co-resident?" without modelling the scheduler or the other agent; with it, E-L1-CLOSED and
E-L1-PLACE are not listed.
- Soundness of the condition: a read ordered after W by an acquire at gpu / sys scope on its
  own SM (or after any CCTL.IVALL on its SM that W happens-before) is never reported. Under
  the strict PTX model every read in a race-free program is such a read, so a reported read
  is racy in the PTX model (possibly hidden by an opt-in such as E-POLL-ACQ).
- Approximations: pairwise reasoning on the run's happens-before order (covers every schedule
  with that order, like a race check); invalidations are taken from E-HW-L1's capped log (the
  last four per thread and SM), which can only over-report; it does not check that W changed
  the bytes (a write of the old value is reported too).
- Status: **argued**.
- If wrong: *false alarm* (over-reporting); a *miss* only through E-HW-L1's own rules.
- Where: `l1_cores.rs` `l1_havoc_check`, called from `l1.rs` `l1_load`.

### E-READ-DONE · code
**Opt-in: a read that its thread has used is ordered before the writes that depend on a
later strong store of that thread** (no load buffering of completed reads; case
`read_done = true`, `PTXV_READ_DONE=0|1`). A strong store, and an arrival at a `bar.sync`,
carry the thread's *completed reads*. A strong read that observes the store orders them
before itself, for write-after-read checks only.
- Completed reads (`Machine::read_release`):
  - the thread's own reads up to its oldest load whose destination registers no
    instruction has read or overwritten yet (`Thread::ld_pend`). An instruction that uses a
    load's register waits for it on the hardware (scoreboard);
  - every read of an async unit (TMA) whose completion the thread knows of (an async unit's
    clock is known only through its completion);
  - the completed reads the thread already holds.

  They travel as clock entries of read actors (`RR_BIT | actor`) with the thread's other
  happens-before knowledge. A write-after-read check accepts a read `(a, c)` when the writer
  knows `a`'s reads up to `c` (`Machine::read_before`). Read-after-write and write-after-write
  checks are unchanged.
- Why it exists (rung 4c): DeepEP reuses a ring slot once the receiver's head says it was
  consumed. The receiver copies the slot (the `LDG`s feed `STG`s, the bulk loads complete on
  an mbarrier), meets at `bar.sync`, then publishes the head with `st.relaxed.sys`, with no
  fence. The sender polls the head with `ld.volatile` and overwrites the slot. Under the
  strict PTX model the receiver's reads are not ordered before the sender's overwrite, a
  load-buffering shape. `fixtures/r4c_dwrap` (a 2-slot ring that wraps) reports these races
  without the option (test `r4c_ring_reuse_is_load_buffering_under_the_strict_model`).
- Status: **argued**. The reads have returned their values before the store issues (their
  registers were consumed, or the bulk copy's completion was awaited), and GPUs do not make
  a store visible before it issues. The litmus tests saw no load buffering on GB300, weak or
  `.STRONG.GPU`, within an SM and across SMs (docs/smem-litmus.md, 2.6e10 and 6.3e9
  instances). Across GPUs (NVLink) it is not validated.
- If wrong: *miss* of a slot overwritten while a peer still reads it (only in cases that opt
  in).
- Where: `exec.rs` `read_release`, `read_before`, `sync_publish`, `sync_observe`,
  `Op::BarSync`, `bar_try_release`, `exec_inner` (`ld_pend`); `machine.rs` `RR_BIT`.

### E-MC-ALIAS-HW · code
**Opt-in: the multicast and unicast views of NVLS memory are one location** (case
`mc_alias = "hw"`, `PTXV_MC_ALIAS=hw`). The S-MC-ALIAS check is off: an access through one
view after a write through the other needs only the usual happens-before.
- Why it exists: GB300 does not show the alias-proxy reordering that PTX allows, and the
  CuTe multimem GEMM + reduce-scatter kernels in the kernel DB use no `fence.proxy.alias`.
- Status: **validated** for two shapes (multicast write then unicast read; unicast write then
  `multimem.ld_reduce`, below). `probes/multimem` `mm litmus` (4 GB300, one NVLink
  domain, 2026-11-06, `results/litmus.txt`): a writer does `multimem.st` of 16 bytes then
  `multimem.red.release.sys` +1 on a flag slot; readers on the 3 other GPUs poll their
  unicast flag with `ld.acquire.sys` and read the data through the unicast address. 200 runs x
  32768 slots x 3 readers: 0 stale reads without `fence.proxy.alias`, 0 with it on the reader,
  0 with it on both sides. Control (`multimem.red.relaxed` / `ld.relaxed`): stale reads in
  200/200 runs (415792 total). Reverse shape (unicast write, then `multimem.ld_reduce`;
  `mm litmus2`, `results/litmus2.txt`): every GPU writes a 4 KiB block with `st.global` or a
  TMA store (`cp.async.bulk` + `wait_group 0`), `multimem.red.release.sys` on the block's flag,
  polls its unicast flag (`ld.acquire.sys`) until all members arrived, then reduces the block
  through the multicast address: 0 wrong lanes in 200 runs x 131072 lanes x 4 (and x 2) GPUs
  without `fence.proxy.alias`. Both controls (relaxed red / poll; `wait_group.read` only, so
  the writes may land after the flag) were also 0: this probe did not show that it can detect
  late data (`LDGMC` bypasses L1, and the all-member poll leaves time), so the reverse shape
  is **not validated**, only not contradicted. Write-after-read is not probed.
  Sensitive reverse probe (`probes/multimem/l2alias.cu`, 2 and 3 GB300, 2026-11-07,
  `results/l2alias.txt`). One writer GPU, 64 CTAs, each bulk-storing 128 KiB to its own
  unicast buffer and then adding to a per-CTA multicast flag. 64 reader CTAs, on the
  writer's GPU or a peer, each poll one flag and immediately `multimem.ld_reduce` that CTA's
  bytes, last bulk copy first. Detector control: with the flag issued before the data,
  stale words appear at 0 ns spin (peer reader, 25 of 3,200 CTAs), 300 ns (777 and 2,815 of
  3,200) and 1 us (2,872 and 2,962 of 3,200). L2's shape (`wait_group 0`, `red.release` /
  `ld.acquire`, no `fence.proxy.alias`), with and without HBM co-tenant noise, readers on
  the writer GPU and on peers, 2 and 3 members: 0 stale words in 76,800 CTA-instances
  (2.1e6 words per run, 200 runs per configuration). Also 0 with `wait_group.read` only
  (relaxed or release flag), with `st.global` data, and with the fence. So a write that
  completed before the flag is never stale through the multicast view. The fence-free and
  fenced shapes cannot be told apart at this resolution.
- Where: `lib.rs` `build_launch` (`mc_alias_hw`); `exec.rs` `alias_check`.
- If wrong: *miss* of an alias-proxy race in cases that opt in.
- Retire: a PTX ISA statement that multicast views are not aliases, or litmus coverage of the
  remaining shapes.

### E-EXIT-FENCE · code
**Opt-in: kernel completion is a `fence.acq_rel` at a scope** (case `exit_fence = "sys"` or
`"gpu"`; `PTXV_EXIT_FENCE=none|gpu|sys` overrides). At its EXIT a thread acquires the
releases it observed with relaxed strong reads (`acquire_pending` at that scope). The exit
already releases its whole history to the next grid on its GPU (E-PDL), so with the option a
relaxed observation of a peer's release, followed by the end of the kernel, orders the peer's
earlier accesses before everything the next kernel on this GPU does. Default: off (the
strict PTX model, where only an acquire fence or acquire read does that).
- Why it exists (rung 4a): torch's one-shot all-reduce ends with
  `sync_remote_blocks<memory_order_release>`: the put has `MEMBAR.ALL.SYS`, the wait is a
  relaxed CAS with no `CCTL.IVALL`. Under the strict model a peer's call-1 reads of this
  rank's symmetric buffer are not ordered before this rank's call-2 restaging store, a
  load-buffering shape (the store is control-dependent on the CAS read, and the kernel ends
  in between). `fixtures/r4a_oneshot` is clean with `exit_fence = "sys"` and races without it
  (test `r4a_oneshot_strict_model_reports_the_reuse_race`).
- Status: **argued**. The CUDA programming model orders a kernel after the previous one on
  its stream and makes the earlier kernel's writes visible, and GPUs do not perform a store
  before the branch it depends on resolves; neither is a PTX memory-model rule. GB300 never
  shows the reordering (r4a one-shot, 1000/1000 bitwise).
- Where: `exec.rs` `Op::Exit`; `lib.rs` `build_launch` (scope); `assumptions::for_machine`
  lists it on every verdict of a case that sets it.
- If wrong: *miss* of a cross-kernel load-buffering race (only in cases that opt in).

### E-BULK-EXIT · code
**A CTA's shared memory outlives its exit until its bulk (TMA) stores have read it.** A bulk
store that no thread of the CTA waited for (`cp.async.bulk.wait_group.read`) before exiting is
reported as a `contract` finding (the PTX ISA requires the wait), not as an async-after-exit
violation; equivalence accepts `contract` findings as preconditions and names them.
- Where: `exec.rs` `exit_checks`; `cli/src/equiv.rs` `prepare`.
- Status: **validated** (`probes/tma_store_exit`, GB300, CUDA 13.0, 2026-11-01): 4864 CTAs per run,
  32 waves, each refilling the shared memory the previous CTA stored from: exiting right after
  the commit gave 0 wrong words in 200 runs; the positive control (the CTA overwrites its own
  buffer after the commit) 200/200 runs wrong. cuBLAS's nvjet GEMM relies on it.
  Peer destination (`tse_peer.cu`, 2026-11-07, DeepEP's shape: bulk stores to the other GPU
  over NVLink, with co-tenant kernels on both GPUs writing across NVLink): exit right after
  the commit gave 0 wrong words in 200 runs x 4864 CTAs; the overwrite control was wrong in
  200/200 runs. Overwriting 200 ns-3 us after the commit was still wrong in every run (the
  bulk engine reads the source microseconds late); 10 us and later was not. The SM keeps the
  CTA's shared memory until those reads finish.
- If wrong: *miss* of a store reading freed shared memory after exit.
- Retire: a statement in NVIDIA documentation either way.

### E-CPASYNC-EXIT · code
**A cp.async no thread waited for lands before the CTA's shared memory is reused.** A thread
that exits with a cp.async no `cp.async.wait_group` / `cp.async.mbarrier.arrive` retired is
reported as a `contract` finding (`cp-async-exit:<pc>`; the PTX ISA defines no completion
for it), not as a violation. Its writes land at the exit as an actor no thread is ordered
after: a read or write of those bytes not ordered before the issue is a race, including
the issuing thread's own read after the issue (unlike E-CPASYNC's waited copies) and any
access after the exit. A pipelined prologue that loads more stages than the main loop
consumes and exits without `wait_group 0` (a decode-combine kernel: trailing splits with no
valid rows; a paged decode kernel's page-index prefetch) has this shape.
- Where: `exec.rs` `Op::Exit`; `hwprofile.rs` GB300 discharges `cp-async-exit:`.
- Status: **validated** (`probes/cpasync_exit`, GB300, CUDA 13.0, 2026-11-08): each CTA fills
  its shared memory with its own pattern, checks it after a spin, then issues 16-byte
  `cp.async.cg` copies of a marker from a cold 1 GiB buffer and exits without a wait
  (`LDGSTS.E.BYPASS.128`, then `EXIT`). 32 waves per launch. 0 marker words in a later CTA's
  shared memory in 500 runs (1, 2 and 6 CTAs per SM; 48, 24 and 8 copies per thread; spins
  of 2e5, 2e4 and 2e3 cycles). Positive control (the CTA overwrites the slots it copies into
  without waiting): 113,206,148 marker words over 20 of 20 runs. The decode-combine kernel
  with trailing empty splits (counts 2, 7, 33): 200 of 200 scrubbed runs each equal to the
  simulator bitwise.
- If wrong: *miss* of a cp.async landing in shared memory the next CTA on the SM owns.
- Retire: a statement in NVIDIA documentation either way.

### E-TC-EXIT · code
**A CTA's shared memory outlives its exit until its tcgen05 pipe has drained.** A
`tcgen05.commit` that no thread of the CTA observed before exiting is reported as a `contract`
finding (its arrival may land after the exit under the PTX model), not as an async-after-exit
violation, when every earlier operation of its tcgen05 pipe was observed (a trailing arrive).
Otherwise it stays a violation, like the unobserved MMAs before it (CuTeDSL's CM1 mutant). trtllm-gen's MMA warp ends with commits to load-stage "empty" barriers that the
already finished loader warps never wait for.
- Where: `exec.rs` `exit_checks`.
- Status: **validated** (`probes/tc_commit_exit`, GB300, CUDA 13.0, 2026-11-04): one CTA per
  SM, 4864 CTAs x 200 runs; each CTA initializes an mbarrier (count 1) at the shared offset
  the previous CTA committed to and tests it after ~20 us. trtllm-gen's shape (commit F,
  commit V, wait F, exit) and a wider window (64 more MMAs and a commit after the wait, then
  dealloc and exit) gave 0 early phases; the positive control (the CTA arrives itself) 200/200
  runs, every CTA.
- If wrong: *miss* of a commit arriving on the next CTA's mbarrier.
- Retire: a statement in NVIDIA documentation either way.

### E-TC-FENCE-NOP · code
**A race of the PTX front that `tcgen05.fence::before_thread_sync` / `::after_thread_sync`
around the synchronization would order is a `contract` finding, not a race.** The PTX ISA
requires the fences (9.7.18.6.3; 9.7.18.6.4.4 example 2: `tcgen05.mma; tcgen05.commit` in one
thread, `mbarrier.try_wait; tcgen05.fence::after_thread_sync; tcgen05.ld` in another), and
S-TCGEN05 models them as the only bridge between a thread's synchronization history and its
tcgen05 view. None of the 219 kernel_db PTX cases that use tcgen05 (Triton, CuTeDSL;
2026-11-08) contains either fence; the PTX front reported TMEM races in 37 records of the
parity report, classified observable by the hardware-model level, which kept the PTX fence
rules. When a PTX check has races and the
kernel uses tcgen05, `ptxv check` reruns it with both fences implied around every instruction
(`Launch::tc_fence_implied`); a race that is gone there becomes a contract finding naming the
missing fences. The hardware-model level (twolevel.rs) also implies them.
- Where: `twolevel.rs` `tc_fence_contract`, `exec.rs` `exec_inner`, `main.rs` check.
- Evidence: ptxas 13.0 (sm_103a) lowers both fences to `NOP`: fixtures/e_fa4_warp_roles with
  its 8 fences removed gives the same SASS apart from 8 fewer NOPs, the placement of a few
  independent instructions and call return addresses; r1a_tcgen05_tile B06 (no
  `after_thread_sync` after the wait) is SASS-clean and bitwise correct on GB300 in 1000 of
  1000 launches (docs/rung1a.md). The SASS front checks the lowered order itself (LDTM / UTCMMA
  are scoreboarded accesses of the issuing thread), so `sass_gb300` needs no profile item.
- Status: **argued.** No litmus separates the fence from the schedule ptxas emits; the claim
  is that the fence has no SASS of its own, not that every schedule without it is ordered.
- If wrong: a race reported as contract (*miss* at the PTX level only; the SASS front still
  reports a race in the binary).
- Retire: a ptxas that lowers the fences to an instruction, or a GB300 litmus.

### E-REMOTE-FIFO · code
**Opt-in: remote mbarrier arrivals from one CTA to one CTA land in the order of their
issues, when the issues are ordered by happens-before** (case `remote_fifo = true`,
`PTXV_REMOTE_FIFO=0|1`). Before a remote arrival B from CTA S lands on CTA T, every arrival
from S to T still in flight whose issue happens-before B's issue lands first (oldest first);
one that already landed as an async step while T never waited on its phase is covered when
T waits on B's phase (or, if B was drained by its issuer's cluster release, by that
release, as in S-REMOTE-INFLIGHT). Remote stores and `st.async` words are not covered and
stay in flight as in S-REMOTE-INFLIGHT. This is weaker than "a thread's arrivals land by its
next CTA barrier": it orders arrivals only against later arrivals to the same CTA.
- Why it exists: FA4's 2-CTA forward (`FlashAttentionForwardSm100`, index L1). CTA 1's last
  `.release.cta` consumer releases (sass 0xa4b0 / 0xb2f0) land on leader EMPTY barriers that
  no thread waits for. The issuing warps then reach named barrier 2 (`BAR.ARV 0x2, 0x1a0`),
  after which CTA 1's MMA warp sends the dealloc arrive (0x4520) that the leader waits for
  before exiting. Under S-REMOTE-INFLIGHT the first arrivals may still be in flight at the
  exit (violation). With this option the case certifies (E-WELLSYNC, 1372 steps, outputs =
  GPU), and so does ptxas's `tcgen05.alloc.cta_group::2` ack at 0x2330 (the same warp's
  0x4520 arrive follows it). The FusedRMSNormBwd pre-fix epilogue
  (fixtures/e_rmsnorm_cluster_drain `bug`, `bug_k3`: one thread's earlier-stage arrive, then
  its last-stage arrive that the target waits for) is also clean under it. That matches
  GB300 (1000/1000 correct), but the source is still outside the PTX contract.
- Not a PTX rule: under the PTX model an arrive is not cumulative through a later release of
  another thread, so neither a `.release.cluster` A nor a `.release.cluster` B orders A
  before the exit (both remain violations in the strict model). The fix inside the contract
  is a wait on every stage's final phase (producer_tail) or a cluster barrier before exit.
- Status: **validated** for the probed shapes (`probes/remote_arrive_exit`, GB300, CUDA 13.0,
  2026-11-07). On GB300 a remote arrive or `st.shared::cluster` that reaches an exited CTA
  fails the launch (`cudaErrorLaunchFailure`). A late access is a crash, not a silent write:
  in `rae.cu` the successor CTAs on the exited CTA's SM never saw a changed word.
  `fa4pair.cu` / `fa4quad.cu` (one binary, targets chosen at runtime, 76 or 38 test clusters
  per launch): warps 0-7 of rank 1 arrive on rank 0 (`fence.proxy.async` +
  `.release.cta`, FA4's SASS sequence), then `bar.sync` or `bar.arrive`; warp 8 arrives on the
  mbarrier rank 0 waits for, and rank 0 exits. Also tested: thread 0 issues B itself right
  after its arrivals (`same`). Result: 0 failed launches in about 3.6e5 test clusters,
  including up to 256 arrivals per thread ahead of B, 64-1024 remote stores per thread to a
  third CTA, and DSMEM noise clusters.
  Positive controls fail in every process at or before the 4th launch: the same arrive
  issued after the barrier, and 64 remote stores per thread to rank 0 ahead of the barrier.
  So remote stores do stay in flight past a barrier and another warp's arrive, which is why
  they are excluded. Order, measured directly (`order.cu`): the source issues arrivals A on
  X, then (after `bar.sync`, or in the same thread) B on Y. The target waits on Y and at
  once test_waits X; X incomplete = B landed before A. Configurations: clusters of 2-16,
  source in the same TPC as the target or not, 1-256 A arrivals per thread, 0-500 ns between A
  and B, DSMEM noise CTAs in the cluster, a co-tenant process under MPS. Result: 0 of 401,200
  clusters saw B before A. Positive control, the same shape with remote stores in place of A:
  24,914 of 64,400 clusters saw B before a store. That control was blind in some
  configurations (clusters of 8 / 16 without noise, 500 ns): 0 there. Stores from the same
  thread as B were never overtaken (0 of 15,200). A cluster stays inside one GPC, so
  arrivals across GPCs cannot occur and were not tested.
- If wrong: *miss* of an arrival landing in an exited CTA (on GB300: a failed launch).
- Retire: an NVIDIA statement on remote-arrival ordering.
- Where: `exec.rs` `complete_remote`, `deliver_fifo_before`, `fifo_transfer`.

### E-DESC-VERSION · code
**tcgen05.mma ignores the fixed bits 46-48 (descriptor version) of its shared-memory matrix
descriptors on GB300:** a descriptor with any value there behaves like one with the
documented 0b001. Triton 3.5's tcgen05 kernels leave the bits at 0. `check` reports such an
MMA as a `contract` finding (sig `desc-version:<pc>`; a `violation` until 2026-11-06), which
the verdict lists as relied on (`ptxv equiv`: "relies on: ..."), like E-BULK-EXIT. A
`tcgen05.cp` descriptor with other bits stays a `violation` (not probed). The finding does
not break the E-WELLSYNC certificate. It checks a descriptor value, and the certificate
shows that values are schedule-independent, so every schedule reports the same finding. It
therefore passes through the certificate like a contract finding: it stays reported, and
the certificate still decides the launch. Before, Triton's tcgen05 kernels always fell back
to exploration, at 35 GB per kernel on a 4096 x 4096 x 8192 tile
(`tests/wellsync.rs` `desc_version_finding_does_not_break_the_certificate`).
- Where: `exec.rs` (the descriptor checks in the MMA paths and `complete_tc_cp`),
  `cli/src/equiv.rs` `prepare`, `wellsync.rs` (the certificate's findings).
- Status: **validated** (`probes/desc_version`, GB300, CUDA 13.0, 2026-11-06): one
  `tcgen05.mma.cta_group::1`, M = 128, N = 64, from fixed random shared-memory images, in 80
  configs (kind::f16 bf16 and kind::f8f6f4 e4m3; layouts none, 128B-base-32B, 128B, 64B,
  32B; A and B K- or MN-major; two (LBO, SBO) each; 69 distinct outputs). With bits 46-48 =
  0..7 on both descriptors and the mixed (0, 1) / (1, 0), every output of every config is
  bitwise equal to the (1, 1) output, two runs each. Earlier evidence: the Triton bf16 matmul
  (`fixtures/c_triton_mm_bf16`) on 2000 random inputs is bitwise equal to cuBLAS and the
  simulator; the Triton block-scaled fixtures (`c_triton_mxf4`, `c_triton_mxf8`,
  `c_triton_nvf4_dq`, bits 0) match GB300 bitwise. Block-scaled kinds are covered only by
  those fixtures, not by the probe.
- If wrong: *miss* (GB300 decoding the descriptor differently from the model).
- Retire: a statement in NVIDIA documentation, or Triton setting the bits.

### E-ASYNC-STEP · code
**A TMA box, an MMA tile, or a commit completes in one scheduler step.**
- Where: `exec.rs` `complete_tma`, `complete_mma`.
- Status: **argued.** Races with readers are still found by vector clocks (they do not depend
  on partial completion). What is lost is value nondeterminism of a racy read that sees half a
  box.
- If wrong: *miss* only of output values in kernels already reported as racy.
- `wait_group.read n` is modeled as `wait_group n` (the store's global writes are done when its
  reads are). Waiting longer can only hide a race on the *source* buffer, never on the
  destination, which stays invisible to the kernel until it exits (S-GLOBAL). SASS agrees
  that there is one wait: ptxas lowers both forms to the same `DEPBAR.LE SBk, n` on the read
  scoreboard of `UTMACMDFLUSH` (rung 1b).
- A wait bit on that scoreboard in any instruction's control word is the same wait with n = 0
  (`Kernel::bulk_read_waits`; cuBLAS's MXF8 nvjet waits so on the `BAR.SYNC` before it reuses
  a TMA-store staging buffer, and has no `DEPBAR`). A scoreboard counts every operation on it,
  so the wait may be longer than modeled, never shorter.
- **Per scoreboard** (2026-11-05). A group counts on the read scoreboard of the
  `UTMACMDFLUSH` that committed it (`Kernel::bulk_commit_sb`, `Thread::bulk_sb`), and a wait
  on SBk (control-word bit or `DEPBAR.LE SBk, n`, whose n counts SBk's groups) covers only
  SBk's groups. Before, a wait on *any* scoreboard some flush of the kernel used drained every
  group. ptxas reuses scoreboards: CUTLASS's grouped GEMM (r3) flushes its epilogue stores on
  SB0, other flushes of the kernel use SB1–SB4, and the epilogue's `BSYNC` waits on SB3 for
  its own `MEMBAR` / `LDTM` / `UTMASTG` operand reads. The model drained the first store there,
  which made the later `DEPBAR.LE SB0, 0x1` and `BAR.SYNC 0x1, 0x80` redundant: NOPing either
  (soundness-harness miss 3) certified clean while GB300 was wrong 200/200, since the staging
  buffer was rewritten before the store had read it. `fixtures/e_gmm_epilogue_reuse`
  (`tests/wellsync.rs` `tma_store_wait_counts_its_own_scoreboard`). Verdict change:
  `r3_grouped_mm` good is now `contract` (was clean): its last store is still reading at exit,
  since the epilogue's last waits are `DEPBAR.LE SB0, 0x1` and a `BSYNC` on SB2, which the old
  rule counted as a full drain (E-BULK-EXIT).

### E-NANOSLEEP · code
**`nanosleep` is a no-op.**
- Where: `ptx` and `sass` lowering (`Op::Nop`).
- Status: **validated by semantics.** Sleeping is a timing hint, and the model explores all
  timings anyway.

### E-SETMAXNREG · code
**`setmaxnreg` is a no-op, and an increase always completes.** PTX
`setmaxnreg.{inc,dec}.sync.aligned.u32 N` and SASS `USETMAXREG.{TRY_ALLOC,ALLOC,DEALLOC}`
change only the warpgroup's register budget; `TRY_ALLOC` always succeeds.
- Where: `ptx` lowering (`Op::Nop`; the form and an immediate N in 24..=256, a multiple of 8,
  are checked), `sass` lowering (`USETMAXREG`).
- Status: **argued**; GB300 agrees only while the requested counts fit the CTA's register
  pool. Then the outputs are unaffected: fixtures/s_ptx_packed (.maxnreg 64,
  256 threads, warpgroup 0 `dec 40`, warpgroup 1 `inc 80`; ptxas emits
  `USETMAXREG.DEALLOC.CTAPOOL 0x28` and a `USETMAXREG.TRY_ALLOC.CTAPOOL` loop) = GB300 bitwise.
  probes/setmaxnreg_pool (same shape): `inc 80` and `inc 88` (3,072 registers requested,
  exactly the 3,072 released) complete; `inc 96` (4,096 requested) hangs until the watchdog
  kills it (2026-11-06, `results.txt`). The model completes all three.
- If wrong: miss (a deadlock): an increase that the CTA's pool cannot satisfy spins forever on
  GB300. Without `.maxnreg` (or a known register count at entry) ptxas drops setmaxnreg with a
  warning.
- Retire: model the pool (registers at entry x threads; dec releases, inc waits for enough).

### E-RET · code
**A SASS `RET.REL.NODEC` lands right after a `CALL.REL.NOINC`.** ptxas loads the return
address with `MOV Rx, <pc after the CALL>` right before the call; the register is data, so
in principle a return could go anywhere.
- Where: `explore.rs` `cfg_succ` (a return's control-flow successors are the pcs after the
  calls, not every pc), used by the synchronization reduction's reachability (`reach_sets`)
  and by E-BAR-OVER's walk (`bar_cfg`); `exec.rs` `Op::Ret` checks every executed return.
- Status: **enforced.** A return anywhere else is reported as `unmodeled` (`ret-target`), so
  a verdict never silently relies on a wrong graph: take the first such return in any
  execution; up to it the graph was right, so the reduced search reaches an execution
  equivalent to that prefix (same per-thread steps), which executes the return and reports
  it. `PTXV_NO_RET_SITES=1` restores "a return may go anywhere".
- Why it matters: the CUTLASS grouped GEMM (r3) calls a division helper before its cluster
  barrier; with "anywhere", every warp could reach every pc, and the reduction was useless.
- If wrong: an `unmodeled` finding, never a miss.

### E-CONST-BRA · code
**A branch whose guard is a launch constant goes the way constant propagation says.** The
explorer's control-flow graph (used by the synchronization reduction's reachability and by
E-BAR-OVER) drops the other edge.
- Where: `explore.rs` `const_guards` (forward constant propagation from each grid's entry:
  constant-bank reads, i.e. the grid's parameters and launch shape, and immediates, through
  `Mov` and `Setp` only; every other register write, and everything after a call or return,
  is unknown; registers start unknown; a pc reached by several grids is folded only where
  they agree), `cfg_succ`; `exec.rs` `Op::Bra` checks the guard of every folded branch on
  every execution.
- Status: **enforced.** A folded branch that goes the other way is reported as `unmodeled`
  (`const-bra`), with the same argument as E-RET (the first such branch in any execution is
  reached, up to reordering, by the reduced search). The analysis is path-insensitive over
  a warp's uniform registers; a divergent split writing one that another split reads is the
  case the check exists for. `PTXV_NO_CONST_BRA=1` turns it off.
- Why it matters: CUTLASS (r3) chooses between the cluster barrier and `bar.sync 0` by
  `c[0x0][0x36c]` (%is_explicit_cluster) after its mbarrier inits; without folding, the
  other CTA's warps could reach their TMA and commits without the cluster barrier, so no
  init could be expanded alone.
- If wrong: an `unmodeled` finding, never a miss.

---

### E-WELLSYNC · code
**A well-synchronized run stands for every schedule.** `ptxv check --certify` (with or
without `--slices`) runs the kernel once (`run_once`'s schedule, `explore.rs` `DetSched`: the
current warp runs until it blocks, then the first async operation that can complete, then the
next warp in cyclic order; any valid schedule serves, and this one looks at one warp per step
instead of every thread of the launch) with an event log (`wellsync.rs`) and checks that every synchronization point is *determined*; then that run's
verdict (findings, outputs, termination) is the verdict of the launch, and no exploration
happens. Otherwise it prints the condition that failed (instruction, mbarrier, phase) and
explores. The idea is SentryWard's (static verification of warp-specialized Blackwell
kernels): if barrier phases/generations are completed by the same arrivals in every schedule
and every wait observes a determined phase, happens-before (HB) is schedule-invariant.
- Conditions (each checked on the run's vector clocks; "x HB y" is the FastTrack epoch test,
  strict for an event that follows a strong store in the same clock window, since a strong
  store does not tick its thread):
  - **C-PHASE.** Every mbarrier operation (arrive, expect_tx, complete_tx of TMA /
    `tcgen05.commit` / `st.async` / CLC) that landed in phase k is HB-after every operation of
    phase k-1 (else it could land in k-1); an expect_tx that is not an arrival is HB-before
    some arrival of its phase (else the phase could complete with tx-count 0 before the bytes
    are expected; with that, at the last arrival tx = expected - completed ≥ 0, zero exactly
    when all complete_tx are done); a final phase with all arrivals has tx-count ≥ 0;
    `mbarrier.inval` is HB-after every use. Phases are numbered from the initial phase
    (S-MBAR-MEM bit 63: phase 0 counts as completed at init); re-initialization starts a new
    incarnation (keyed by the init epoch).
  - **C-WAIT.** A successful try_wait/test_wait observed phase k (the current phase minus
    one). Lower bound: every operation of phase k-1 is HB-before it (it cannot arrive while
    phase k-1 is current and see k-2 of the same parity). Upper bound: let h be the first
    phase ≥ k+1 that has an operation after the wait or does not complete in the run, where
    "after" means HB-after the waiter, or, on SASS, after any lane that ran the wait in the
    same step (C-WARP guarantees those lanes always run it together, so a later event of any
    of them is later in time; the model's uniform ops carry only the first lane's clock). In
    any schedule the wait arrives while the current phase is in [k, h], returns at the first
    phase of the other parity, observing k + 2j (only *more* HB than in the run: every check
    and finding is monotone in the clocks, and DRF values do not change), and must not arrive
    at h with h of its parity: so h - k must be odd (usually h = k+1: the wait is HB-before
    the next phase, the pipeline's empty/full handshake). Failed blocking polls are invisible
    (E-SPIN).
  - **Peeks** (try_wait without the retry shape, spin.rs) return the same value in every
    schedule (h = k+1 for a success, h = k for a failure), unless its predicate is used
    only by one branch that skips a blocking wait on the same mbarrier and parity (CUTLASS /
    cuBLAS "peek, work, then wait"), possibly through predicates derived from it by
    `PLOP3` / `ISETP` / `VOTE`, or kept as a GPR token (`SEL R4, RZ, 0x1, !P1`, read by the
    next iteration's `ISETP.NE.AND P1, PT, R4, 0x1`: CUTLASS's producer try-acquire) (each
    recorded with its value had the peek returned the other result: the instruction rerun on
    a copy; lanes combined by a vote must come from one peek instruction instance, which all
    see the same mbarrier state). Then both sides of the branch are run on copies of the
    machine (`skip_equiv`): they must meet with no memory or synchronization effect except
    that wait (passed as such; an instruction predicated off for every lane has no effect,
    as the `@!P0 NANOSLEEP` / `@!P0 SYNCS.PHASECHK` of a retry stub after its try_wait
    passed: cute.ResidualKernelTMA), and every register in which they then differ must be dead at
    the meeting point (`walk_dead`). The lanes are walked on from there on a copy whose
    register values are the run's: ALU ops and branches run; memory / sync effects without a
    register result are stepped over (they cannot steer the lanes); a result from memory,
    the TMEM or an unknown input makes its register unknown, and a branch on an unknown
    predicate forks the walk. A differing register read on any path fails, one written
    (guard known true) is dead on that path, except that a move, a byte permute with a
    constant selector (`PRMT`) or a funnel shift by a constant (`SHF`) carries the differing
    bits on bit by bit, and a result with none is the same on both sides (cute.ResidualKernelTMA:
    the retry stub's `MOV R0, UR32` leaves R0 different; `PRMT R0, R15, 0x7632, R0` keeps its
    upper half, which `SHF.L.U32 R5, R0, 0x10, RZ` shifts out; certified, output equal to
    GB300's), and where the walk cannot go on (budget,
    divergent branch, calls) static liveness over the control-flow graph decides (the values
    other threads write could steer the path anywhere). Rung 3b's gemm1 at K = 2048 needs
    the walk: its loaders' peek results survive to the next iteration's peek, past loads.
    Such a peek is the wait placed at the branch:
    a successful one gets the blocking-wait upper bound measured at the branch; a failed one
    needs nothing more (in a schedule where it succeeds, the skip is the run's wait done
    early, with more HB). A failed peek that is never read changes nothing; a successful
    one acquires, so it must be determined.
  - **C-GEN.** Every `bar.sync`/`bar.arrive` arrival in generation g is HB-after every
    arrival of generation g-1 (subsumes E-BAR-OVER ambiguity); the lanes of one instruction
    land in one generation. `barrier.cluster`: an arrival in g by a thread that waited on
    g-1 itself is fine, any other must be HB-after every arrival of g-1.
  - **C-ORDER.** Conflicting strong accesses (scoped ld/st, atomics) are HB-ordered, except
    two atomics of the same commutative operation (add/or/and/xor/min/max) with unused
    results that do not acquire (final value order-independent), and likewise any two
    "key max" atomics: `ATOMS.MAX.S32` of a word with the sign bit clear and `ATOMS.MIN`
    (unsigned) of one with it set are both the maximum under one total order of 32-bit words
    (sign clear: the word as a signed integer, all above sign set; sign set: smaller words
    above larger), so they commute with each other (CUTLASS's float atomic max, as in
    fmha_cutlassF_f32; the operand's sign is read at the access); the lanes of one SASS
    atomic step (one instruction instance of one warp) are ordered by lane order
    (E-LOCKSTEP; C-WARP keeps them together in every schedule: flashinfer's routing assigns
    token slots with `ATOMS.ADD`); TMEM allocator operations of a CTA are
    totally ordered; no try_cancel / CLC response.
    **Equal-value writes** (2026-11-06, wellsync/order.rs `pair`). Two plain (non-atomic)
    writes that store the same bytes where they overlap need no order: either order leaves
    those bytes. The values are the run's in every schedule, by induction along
    happens-before: a write's value comes from reads before it, each ordered with every write
    it conflicts with (or a poll's exit value under C-POLL, a CAS chain's under C-CAS), so
    each reads the run's value; a read unordered with one of the writes is its own pair, and
    a read after both already has both in its clock whichever it reads. Not on words a poll
    reads (the poll may exit on either write and synchronize with a different writer), and
    not in runs with C-RING choices or forced reads (the run's values are one choice among
    several, fixtures/e_ring_warp W1). This is S-SAME-VALUE for the certificate with LDS/STS
    strong, where the plain one refuses runs with strong accesses or polls: Triton's
    all-reduce result stored by every warp with `STS.128 [UR4]` in
    `_kernel_dispatch_and_calc_grads` (1200 s timeout → certified in 5.5 s, outputs bitwise
    equal to the GPU). fixtures/e_same_strong: good certifies; SV1 (warp 1 stores another
    value) is refused and found nondeterministic.
    **Dead reads** (2026-11-06, wellsync.rs `dead_reads`). When every conflict of the run is a
    write against a plain load by a thread, the run is repeated with those loads (thread, pc)
    leaving their destination registers undefined (S-INIT propagation). If that run reports no
    undefined-register use the first run did not, and ends as it did, no loaded value reaches
    memory, an address, a predicate, control flow or a synchronization operand along the run's
    path: a load that reads another value in another schedule leaves the same events, and the
    conflicts are dropped (a note records them). CuTe's TMEM-address broadcast in
    BlackwellScoreSummary / BlackwellBlockScoreTopk: the warps that skip the named barrier load
    the address before the allocating warp stores it, and never use it (192 loads; 2M states
    incomplete → certified in 0.1 s). fixtures/e_dead_read: good certifies; DR1 (the value is
    stored) and DR2 (a branch tests it) are refused (crates/sim/tests/reductions.rs). Relies
    on S-INIT reporting every use of an undefined bit (its listed limits apply: an operand
    kind S-INIT does not check could carry a used value unreported).
  - **C-FADD** (2026-11-04, rung 4b `index_add_`). fp32 atomic adds of one word
    (`ATOM/RED.ADD.F32.FTZ.RN`, unused results, not acquiring) that happens-before does not
    order are not refused outright. Every other access of the word is ordered with each of
    them (C-ORDER for strong ones, the race detector for plain ones), so it lies before or
    after a whole connected component of the "unordered" relation among the adds (if x is
    after a1 and before a2, then a1 HB a2; induct along unordered chains). So the value a
    component starts from is the run's (the word's value before the component's first add in
    run order), and no access observes an intermediate value (results are unused). Every
    schedule applies the component in some linear extension of happens-before restricted to
    it. `fadd_word` enumerates them all (at most 2^16, else refused) and folds the operands
    (recorded per access) with the executor's own add (`f32_fma(acc, 1, x, RN, ftz)`,
    N-ATOMIC); every order must leave the run's bits. Then the word's final value is
    schedule-independent and the value argument below holds (r4b `index_add_`, at most two
    adds per element onto +0, which commute exactly: accepted). Orders that give different
    bits are refused with the reachable values listed, with one exception:
    *nondeterministic in the output only*. That applies when
    - the component holds the word's last writes (no later strong write; the final race
      shadow's last write is one of its adds);
    - nothing reads the word afterwards (no reader in the final shadow since that write);
    - the word belongs to an output buffer.
    Then the order changes only those final bytes: every schedule's value is among the
    enumerated ones (a superset: an extension need not be schedulable), nothing else differs
    (results unused, nothing reads the word), and the verdict is the run's, with every
    combination of the words' values as the possible outputs (`nondeterministic`; enumerated up to 256,
    counted beyond). `c_triton_rowsum_atomic`: 9 reachable outputs. The GPU's captured
    output is one of them, and every output of a bounded exploration is among them (test
    `fadd_orders_are_reported_as_output_nondeterminism`).
    Release semantics need no check: the release sequence joins along the RMW chain, which is
    order-independent.
  - **Peek taint ends at a try_wait** (2026-11-04). A peek's predicate is tracked until it
    is overwritten. A blocking try_wait that writes the same predicate overwrites it too (with
    its own result, true whenever it returns; a peek there starts a new taint). This used to be
    skipped for every try_wait, so CUTLASS's second-tile producer peek (`TRYWAIT P0` at
    0x4140, `P0` dead on the tile-end path, rewritten by the next tile's blocking `TRYWAIT P0`
    at 0x43b0) was reported as read by that wait's retry branch, whose direction does not
    change when the stale taint is flipped. Guarded-off lanes do not overwrite (`active`).
  - **Strong-model pass** (2026-11-04). `check_certified` tries the certificate on the weak
    kernel (LDS/STS plain, as `smem::two_pass` explores), then on the kernel itself (LDS/STS
    relaxed strong at CTA scope, S-SMEM-STRONG), and explores only if both refuse. Each
    certificate alone is a verdict for every schedule of its model. The strong one is the
    hardware-faithful model, and the weak one only stands for it when it shows no race.
    A race between plain LDS/STS (a shared-memory relay) is a strong-access pair in the
    strong model, which C-ORDER / C-POLL decide.
  - **Convergence barriers end a strong store's clock window** (2026-11-04). A strong store
    does not tick its thread, so a later event in the same window is ambiguous ("amb") and
    needs strictly more than its clock. At a BSYNC / WARPSYNC arrival, as at EXIT, the
    certificate run ticks a thread whose window is open. The release joins the arriving
    lane's clock into every released lane, after the store. Without this, CUTLASS's
    tensormap build (lane 0 `STS`, `BSYNC`, all lanes `LDS`) was refused in the strong model.
  - **C-POLL.** Flag polls (E-SPIN poll loops). A poll load can read only the flag's
    HB-latest write before it (unique: two unordered writes are a race or fail C-ORDER) or
    a strong write concurrent with it; atomic or partially overlapping flag writes are not
    checked. For each candidate the loop body is replayed on a copy of the machine taken
    before the load, with the flag's bytes and synchronization source replaced:
    - an iteration that failed in the run fails on every candidate, so the thread cannot
      leave earlier;
    - an exiting one fails or exits at the same pc, with the same live registers (static
      liveness) and at least the run's clock and pending releases (a pending release from
      another writer counts if every acquire-fence scope that admits the run's writer admits
      it);
    - liveness (2026-11-04): for an exiting one, every candidate whose iteration fails is
      HB-before a concurrent candidate whose iteration exits. For the value before the
      concurrent writes (the poll's own past), the writes that count are the race shadow's
      last writes of the flag's bytes before the first concurrent write the run did before
      the poll, or before the poll if there is none. With none (the initial contents) it
      holds trivially.

    Why liveness is needed. A failed iteration is invisible only if the thread later leaves
    the loop. The run's exit shows that one candidate lets the poll pass. It does not show
    that the poll can still read that candidate in every execution: a concurrent write it
    fails on may come later in coherence order and never be overwritten. Example: e_poll_any
    `dl`. Thread 0 stores the flag 1, then bar.sync, then polls it. Thread 32 stores 0 after
    the barrier, so its store is coherence-after the 1. The run's poll read 1 and exited.
    With thread 32's store first, the poll reads 0 forever, and exploration finds the
    deadlock. Before this check the certificate said clean on both fronts.

    Argument. The candidates are the writes of the flag that one execution of the poll can
    read. In any execution E, coherence order on them extends happens-before (write-write
    coherence). So a candidate with an HB-later exiting candidate is not coherence-latest,
    and the coherence-latest candidate exits. The candidates' writers do not depend on the
    poll thread getting past the loop: none is HB-after the poll, and failed iterations
    release nothing. So by the progress induction every candidate is eventually performed
    in E. After that, the poll reads the coherence-latest one (E-SPIN: a spinning load
    eventually observes the latest write) and exits. Writes HB-after the poll are not
    candidates of this execution, since a load cannot read a write HB-after it. C-ORDER and
    the race detector already order the candidates totally by HB. The check is stated for
    a partial order anyway, so it does not rely on that.

    C-RING needs no extra check. A counter is written by one thread's strong stores with
    nondecreasing values, and the value before them (the plain or initial write) is
    coherence-before the first, so coherence order is program order. C-RING's exits are
    upward closed, and the run's read exits, so the coherence-latest write (the counter's
    last) exits too.

    Failed iterations are invisible (E-SPIN), so the rest of the program sees the run's
    outcome with at least the run's happens-before. Reading a later release in the run can
    give more HB than another schedule's earlier read (r1g: a peer's next-barrier signal).
    So the run continues from the load with the least clock of the exiting candidates known
    then, the concurrent writes before it in the run. The check above compares every
    candidate against that clock. An exit releases the thread's whole history (the next
    grid on its GPU joins its final clock), so a strong store's open clock window
    (non-ticking) is closed at EXIT in a certificate run.
  - **C-POLL-ATOM** (2026-11-07, index L2's `Sm100GemmRsMultimemKernel` flags, DeepEP's
    DeviceSyncer). A poll whose concurrent writes are atomics (C-POLL refuses those) is
    accepted when:
    - every strong write of the polled bytes not HB-ordered with the poll is an integer
      `add` of the polled width (u32 / u64) whose result is unused and that does not
      acquire (`red`, SASS `REDG` / `RED`);
    - each adds a positive amount (as a signed value of the width), and the base plus all
      of them does not wrap. So a proper subset of the adds sums to less than all of them.
      The base is the value before the first concurrent add the run made, or the value the
      run read if none came before it;
    - an exiting iteration in the run read after every concurrent add (it saw the full
      sum), and replayed on the base plus every other subset sum it fails;
    - an iteration that failed in the run fails on every subset sum, the full one included
      (as in C-POLL);
    - at most 16 concurrent adds and 256 distinct partial sums; otherwise it is refused.

    Argument. Adds commute, so after any set S of the concurrent adds (and every add
    HB-before the poll, which the base includes) the word holds base + sum(S). Subsets
    over-approximate the sets an execution can show, which only adds failing replays. The
    poll exits only on the full sum, and only S = all concurrent adds produces it. The value
    it then reads is written by the coherence-last of those RMWs, and each RMW is in the
    release sequence of every coherence-earlier release on the word. So an acquire read of
    the full sum synchronizes with every add's release: the same set as in the run, with
    the clocks of the certificate's induction. The exit, its registers and its
    synchronization are therefore the run's.
    Liveness: none of the adds is HB-after the poll, so none waits for it, and by the
    progress induction all of them are performed. The poll then reads the full sum (E-SPIN)
    and exits.
    Reset by the poller: a strong store to the polled bytes by the polling thread,
    program-order-after an exiting read that C-POLL-ATOM accepted, is coherence-after every
    add that exit needed: the read saw the full sum, written by the coherence-last of them,
    and a write program-order-after a read of a location is coherence-after the write it read
    (CoRW, sequential consistency per location for morally strong accesses; a pair that is
    not morally strong is a race for the race detector). C-ORDER accepts those pairs (the
    value left is the store's in every execution); adds the exit did not need (a later round)
    still need happens-before. Index L2's multimem kernel resets its cross-GPU flag word this
    way after polling it.
    Fixture `e_poll_atom` (good and reset: certified; ge1, an exit on a partial count: refused
    and races).
  - **C-TICKET** (2026-11-07, index L2's tile scheduler and arrival counters,
    `wellsync/ticket.rs`). C-ORDER refuses unordered atomics whose results are used. A
    *ticket word* is one where some atomic's result is used and every strong access in the
    run is a write of the same bytes by one lane of a thread (one lane per instruction step):
    a relaxed read-modify-write (not acquiring; not a compare-and-swap, which is C-CAS's) or a
    strong store. No load reads it (a poll is C-POLL's); plain accesses are races for the
    race detector. Given the order of its writes (its coherence order), every value its
    atomics return is determined, so that order is the only freedom; the certificate
    enumerates it:
    - the first run that finds ticket words restarts the search in ticket mode, where C-ORDER
      leaves the writes of those words unordered and each run forces a prefix of every word's
      writer sequence: a lane at a gated write waits until the writes before its forced
      position are done. Waiting only delays a write, so a forced run that ends is an
      execution; a run stuck with a lane waiting for a forced writer that never writes is
      no execution (that writer cannot write first) and is dropped;
    - branches, as dynamic partial-order reduction: everything else that conflicts is ordered
      by happens-before (the certificate's conditions, checked on every run), so a ticket
      word's writes are the only dependent events left unordered. For each write c_i past
      the forced prefix and each other thread u, with c_j u's first later write of the word:
      if c_j happens-after none of c_i .. c_{j-1}, a run forces the prefix up to c_i with u in
      place of c_i's writer. These are a superset of Flanagan-Godefroid's backtracking points
      (which reverse only the last such pair); a write that happens-after one of them is
      reached by reversing an earlier pair. The writes do not synchronize with each other
      (relaxed, not acquiring), so the run's happens-before has no edge that only this order
      created. Each forced prefix runs once;
    - every run must certify on its own and all must leave the same outputs. Results that
      depend on which thread processed which ticket are refused (`C-TICKET: the claim order
      reaches the outputs`), as are more than 256 runs (`PTXV_RING_MAX_RUNS`, shared with
      C-RING) and a word that stops qualifying in a later run.
    Not a permutation argument: a persistent kernel's CTA-local state (pipeline stage and
    phase) depends on how many tickets its CTA processed, so "a claimant's work is a function
    of the ticket value" does not hold in general; the enumeration needs no such property.
    Fixture `e_ticket` (good: 95 runs, certified; ctaid: refused, nondeterministic; race:
    refused, race).
  - **C-CAS.** Compare-and-swap semaphores (E-SPIN poll loops whose poll operation is a
    CAS: torch symmetric memory's put `while (cas(&w, 0, 1) != 0);` and wait `while
    (cas(&w, 1, 0) != 1);`, `MEMBAR.SYS` / `ATOM.E.CAS.STRONG.SYS` / `CCTL.IVALL`). A put and
    the wait that consumes it are not HB-ordered (the CAS is relaxed; the wait acquires the
    put's release fence, not the CAS itself), so C-ORDER alone refuses them; but the values
    force the order. A successful CAS of a poll (a *link*) reads exactly its compare value and
    writes its swap value. For every word whose links are not all HB-ordered:
    1. every link changes the value (compare ≠ swap);
    2. one *kind* (compare → swap) per compare value (else two kinds compete for it);
    3. the links of one kind are totally HB-ordered (else either may consume the value);
    4. in the run only the links write the word between the first and the last link: each
       link read the bytes and the synchronization source the previous one left, and no
       other strong write lies in between. The word's last write before the first link
       (race shadow) is HB-before every link;
    5. each link's iteration, replayed on a copy of the machine taken before its instruction,
       leaves the loop on its compare value and fails on every other value of the chain (the
       value before the first link and every swap value).

    Every CAS-poll attempt that failed in the run must fail its iteration (replayed on the
    value it returned): its read is not a link and is not checked otherwise. It must also
    observe no release beyond what the thread knew (its clock and pending releases): in
    another schedule the attempt is not made, so a gain would make the run's clocks larger
    than that schedule's. Other accesses to the word are not links: strong ones stay under
    C-ORDER, plain ones under the race detector, flag polls under C-POLL (which refuses
    atomic writes).

    Argument. Let s_1..s_n be the links in run order. All other writes to the word are
    HB-before every link or HB-after the last one: by (4) none lies between them in the run,
    and each is ordered with the links. A strong one is ordered by C-ORDER; a plain one is
    ordered because R is race-free. By induction over E: once E has done s_1..s_(i-1) and no
    other link, the word holds s_i's compare value. The next link to succeed must be of
    s_i's kind (2), and of that kind it must be the earliest not yet done (3: the later ones
    are HB-after it, and E's clocks contain R's). That is s_i. Failed attempts change nothing
    (E-SPIN) and cannot leave the loop on a value the chain holds (5). So each link reads
    from the same write as in R and returns the same old value. The synchronization source
    it reads is the same function of clocks that contain R's: the release clock (the
    `MEMBAR` before the CAS) joined along the RMW chain (`sync_publish`). The acquire
    (`CCTL.IVALL` after the wait) therefore gives at least the run's clock. Progress: when E
    has done s_1..s_(i-1), s_i is enabled. The blocked-poll reads the race detector records
    in E come from link threads before their success, at a clock containing the first
    write's; each link's write clears them. A plain read of the word between two links is
    checked against the next link in R, so it is pinned between the same links in E.
    Without E-EXIT-FENCE, the one-shot's call-2 wait fails against call 1's last wait and
    acquires its release (another grid that the strict model does not order), so it is
    refused; its run also reports T4a.

    Evidence. `crates/sim/tests/wellsync.rs`:
    - r4a one-shot and two-shot, one call and two calls, certify. With the cases'
      E-EXIT-FENCE, the outputs are bitwise equal to the GPU bytes.
    - For one call, the certified verdict and output equal exhaustive exploration's, under
      both the fence and the strict model.
    - The strict two-call one-shot is refused, and its run reports the race.
    - The mutants B57–B62 are refused, or certified with the explorer's verdict (B60 for one
      call: its bug needs the second call).
    - `s_cas` and `fixtures/e_cas_sem` certify, and their outputs equal the explorer's.
    - `e_cas_sem`'s other entry points must be refused, and the explorer shows the bug in
      each: a spin lock (two unordered 0 → 1 links), two kinds with one compare value (a
      deadlock), a put loop that exits on a failed compare, and a strong store to the word.
  - **C-RING** (2026-11-04, rung 4b DeepEP channels). Polls of monotone counters whose
    observed value steers the program: a ring buffer's consumer copies the slots between its
    head and the tail it saw, so how it chunks the slots, and with that its sequence of
    events, depends on the schedule. Human reasoning about such a ring is ownership transfer:
    slot i belongs to the producer until a tail release covers i, then to the consumer until
    a head release covers i. C-RING checks the counters, decides every poll of them, and runs
    the certificate once per distinguishable sequence of observed values (`wellsync/ring.rs`).
    - *Counters, checked.* A poll load is a load inside an E-SPIN poll loop (straight,
      `polls`, or warp-level, `wpolls`). It reads a *counter* when every write of the word in
      the run is a strong store (no atomic) of one thread, of the same width and offset, the
      values never decrease (unsigned), and no plain write ends the sequence. The values a
      thread writes are the same in every schedule the certificate covers (the induction
      below), so the check on the run's values is a check on all of them. A warp-level poll of
      a word that strong writes are concurrent with and that is not a counter is refused
      (barrier counters built from atomic adds, such as DeepEP's `barrier_block`, are not
      counters). A straight poll of a non-counter flag stays C-POLL's.
    - *Candidates.* For one entry into a poll loop: every write of the counter from the
      coherence bound on, where the bound is the latest write that happens-before the loop's
      first read, or that a read happening-before it read (the thread's own earlier polls
      included: a relaxed counter store after a release fence is not itself happens-before
      its reader, so the first bound alone would let a poll read back an older value). Writes
      that happen-after the read are kept, an over-approximation (see *dropped runs*).
    - *Classes.* The exiting iteration is replayed on a copy of the machine from its start
      (the snapshot at the loop head, or before the load of a straight loop) with the
      counter's bytes and synchronization source replaced by each candidate's. A candidate
      fails (back at the head) or exits. Exiting candidates are grouped by what the rest of
      the program can see: each member's exit pc and every register the loop writes that is
      live there (liveness over the model: `CALL.ABS` vprintf reads only its target, and a
      write under a guard kills the register for reads under the same guard).
    - *Invisible failed iterations.* A failed iteration (one the run made, or a failing
      candidate's) must not change a register that the next iteration reads before writing
      it (loop-local liveness), and from the state it leaves, every class's exit must leave
      the same state as from the snapshot. Then an execution may leave the loop at another
      iteration than the run did, on the same value, with the same continuation.
    - *No lost wakeup.* In the counter's write order, once a candidate exits every later
      one exits: a wait on a monotone counter, once satisfied, stays satisfied.
    - *Synchronization.* Candidates of one class leave the same state except for what they
      acquire, which grows along the counter's writes (one writer). The earliest member
      acquires the least: the run continues with its clocks (lowered at the exit from the
      replay, as C-POLL does), and every member must acquire at least that.
    - *Choices and the search.* A poll with one class is decided by the run. A poll with
      several classes is a *choice*: the certificate is run once per sequence of classes, a
      depth-first search in which every run spawns, at each choice it did not inherit, one
      run per other class, with the earlier choices of several classes pinned to the run's
      class. A pinned or forced choice (`Machine::ring`) makes the loop wait, at its head,
      until the class's earliest write exists, and its read return that write's bytes and
      synchronize with that write. At most `PTXV_RING_MAX_RUNS` (256) runs, else refused.
    - *Dropped runs.* A forced run that gets stuck because a thread waits for a forced write
      that never comes, while its poll would exit on the value already written, is dropped:
      no execution makes those choices (the thread reads a written value eventually and
      leaves with another class, which a sibling run covers). Its log must hold every other
      condition.
    - *Chunking-obliviousness, checked semantically.* Every run of the search must certify
      (no finding but exit-time contracts), and all must leave the same outputs, else the
      certificate is refused as not chunking-oblivious. A consumer whose stored values,
      addresses or control depend on how it chunked (a threshold on the batch size, a batch
      counter) gives different outputs for different choices.
    - *Launch groups.* A certified group whose certificate took several runs continues from
      their meet, as an explored group from its terminal states (E-LAUNCH-GROUPS), with every
      run's obligations; if the runs leave different global memory there is no continuation.
    - *Reliance.* The verdict notes E-RESIDENCY (the producer's and the consumer's CTAs are
      resident at the same time: a consumer CTA spinning while its producer CTA cannot launch
      would hang, which the default model does not represent; `--residency auto` does) and
      E-POLL-ACQ when the case opts in (DeepEP's warp-level polls acquire only through it).

    Argument. Let E be any execution. Its *choice sequence* is, at every loop entry where the
    search found a choice, the class of the write the exiting read read. Claim: the search ran
    a certificate run R for that sequence, and E repeats R's events with R's outcomes and
    clocks containing R's. Induction over E's events in E's order, as in E-WELLSYNC's
    argument, with R the run whose choices agree with E's so far:
    - At a choice point c, E's exiting read reads a write w of the counter. Every event of E
      so far is an event of R, and so is w. E's clocks contain R's, so w is at or after R's
      coherence bound: w is a candidate at c in R, and its class is one of R's classes. If it
      is R's class, continue with R. If not, continue with the run that forces c to w's class:
      it is identical to R up to c in run order, and every event E made before c is not
      happens-after c, so it has the same outcome there.
    - E's read of w leaves the state R's read left (same class), and synchronizes with at
      least what R's synchronized with (the earliest member acquires the least). Failed
      iterations of E are invisible, whatever values they read and wherever E leaves the loop.
    - Every other event is decided by the existing conditions (C-PHASE ... C-CAS, C-WARP),
      which R satisfies, as in E-WELLSYNC.

    So every execution has the outcome of one run of the search; the runs agree, so the
    verdict is theirs. Termination: in E every blocking point is R's; a counter wait exits on
    every value at or after the first it exits on, and the counter's last write is an event of
    R, hence of E, so under E-SPIN's fairness (a blocked poll eventually reads the latest
    write) the wait exits; every other wait is decided as in E-WELLSYNC. One terminal run per
    choice sequence gives termination of every fair schedule.

    Why not one run with a taint check. A taint rule "the observed value may reach only loop
    bounds over slots and the next poll of the counter" is unsound with explicit-flow taint:
    a batch counter (`n++` per batch, stored at the end) is never data-dependent on the
    observed value, yet differs between chunkings (fixture e_ring R7). With implicit flows
    taken into account it refuses DeepEP: in dispatch's receiver
    (`intranode::dispatch`, sass:0x3cf0..0x8010) the slot-to-warp assignment is relative to
    the observed head (`R13`), each warp's TMA mbarrier parity flips once per slot the warp
    copies (`R37`), and the output offset is the sum of the observed batch sizes (`R27`). The
    per-slot effects are the same under every chunking, but showing it takes relational
    reasoning across threads. C-RING therefore runs each chunking and compares. The ownership
    facts (each slot written before the tail release that covers it, read after an acquire
    that observed it, reused only after the head release that covers it) are what the race
    detector checks pairwise, on every payload byte, in every run.

    Cost: one run per sequence of choices, at most the product over rings of 2^(w-1) for w
    tail writes. r4b dispatch: 4 runs (two of its four rings have two tail writes); DeepEP at
    3 tokens per ring and chunks of 1 (agent W's `r4c_nowrap`): 256 runs, 9 min.

    Evidence. `crates/sim/tests/wellsync.rs`:
    - `fixtures/e_ring` (one GPU), `e_ring2` (two GPUs, the ring in the consumer's memory, as
      DeepEP lays out its channels) and `e_ring_warp` (DeepEP's consumer shape: a warp-level
      poll by lane 0, a shared-memory relay, the slots of each chunk shared by two warps),
      PTX and SASS: the good variants certify (4 runs, one dropped), and their outputs equal
      exhaustive exploration's (`certified_outputs_match_exploration`).
    - Mutants (`ring_mutants_are_refused_with_their_reason`), each refused: R1 publish before
      the payload write, R2 a read past the observed tail, R3 reuse without the head wait, R5
      the tail released at too narrow a scope, R6 a relaxed tail store, W1 two warps copying
      one slot (all races in some run of the search); R4 a threshold branch on the batch size
      and R7 a batch counter (not chunking-oblivious). The explorer gives each its expected
      class.
    - `e_spin_smem` SP6 (a flag set to 1 then 2, the value stored) is now refused by C-RING
      as not chunking-oblivious (it was C-POLL's refusal).
    - r4b in group mode: dispatch certifies (docs/rung4.md).
  - **C-WARP.** Lane membership: an instruction whose effect depends on which lanes run it
    together runs with lanes that became ready at that pc in one event (an instruction or a
    barrier release: then they meet there in every schedule). Where more lanes joining
    would change it too (SASS lane collectives, `BRA.U.ANY`, TMEM collectives and uniform
    operations, which run once per lane group; PTX elect / vote / redux / R2UR / BSSY /
    tcgen05.alloc), runs of the same pc by other lane groups of the warp must be HB-ordered
    with it (they cannot meet there in another schedule). SASS per-lane memory and
    synchronization instructions need only the first part (more lanes only add E-LOCKSTEP
    happens-before); SASS lane-local arithmetic, branches and EXIT need neither. PTX
    `.aligned` ops skip the first part (the explorer waits for all live lanes). BSYNC parks
    only lanes BSSY recorded. A WARPSYNC/BSYNC release need not wait for the order of an
    exited lane's exit: an exit after the arrivals re-checks the release
    (`exit_warpsync_release`, E-BSYNC) and frees the same lanes with the same joined clock
    (an exit adds none). Before 2026-11-06 the certificate required the exit HB-before
    every arrival and refused ptxas's `@P0 EXIT` inside a BSSY region
    (`fixtures/e_bsync_exit`, GB300 output as expected; certified outputs equal the
    explorer's).
    SASS `BREAK` needs neither part (2026-11-06): it clears each executing lane's own bit of the
    warp's convergence mask and releases the lanes parked at that barrier's `BSYNC` once every
    lane still in the mask is parked, joining only the parked lanes' clocks (E-BSYNC). Bit l
    changes only through lane l's own `BSSY` / `BREAK` or a release lane l is parked in, so the
    mask, the release point and its clocks do not depend on how the breaking lanes are
    grouped. Triton's predicated-store chains reach `BREAK.RELIABLE` from two paths
    (fixtures/e_brk; tests/reductions.rs keeps its verdicts and those of its two mutants).
  - **Residency.** The run has every CTA resident (E-RESIDENCY). A cross-CTA wait is decided
    only under that assumption: the certified verdict of a kernel with a poll loop on global
    memory, on a GPU running more than one launch unit, carries the E-RESIDENCY note, and the
    certificate refuses under limited residency or co-tenant mode. `residency::analyze` uses
    the certificate the other way round: its order-independence makes the launch-set search
    exact.
  - The run reports **no finding** (race, deadlock, violation, fault, unmodeled, trap)
    except exit-time `contract` findings (E-BULK-EXIT: an HB check at the end, and every
    schedule's clocks contain the run's, so the run reports the union): a race breaks the
    value argument, and the rest are left to the explorer. Schedule-free
    findings (SASS scoreboard, E-SPIN shape) are added as in exploration. PTX `elect`:
    one run per leader policy (E-ELECT).
- Argument: let R be the run, E any execution. Induction over E: each actor's next event
  in E is its next event in R, with the same outcome and a vector clock ⊇ R's. Values: plain
  reads read the HB-latest write (R race-free; the detector is precise, below, and E has
  at least R's HB edges), strong reads are ordered with every write (C-ORDER) or are CAS
  links that read from R's predecessor (C-CAS), allocator results and warp collectives are
  determined (C-ORDER, C-WARP). Synchronization: C-PHASE
  puts every operation in its run phase (it cannot run before phase k-1 completes, and
  phase k cannot complete without it), so completions join the same clocks; C-WAIT and
  C-GEN fix what waits observe. Progress: each blocking event's enabling condition depends
  only on events before it in R, and every event that could disable it is HB-after it
  (C-WAIT's h), so the R-first event E has not executed is enabled: E continues until it has
  executed exactly R's events, reaching R's final state. So every schedule terminates with
  R's outputs and no finding. With C-RING polls whose observed value steers the program,
  R is one run per sequence of choices, and E follows the run of its own choices (C-RING's
  argument).
- Detector precision (the value step above needs it): the run's race detector reports every
  pair of conflicting accesses that R's HB does not order, unless both are strong and morally
  strong (S-SCOPE). Vector clocks give that only if the per-byte shadow keeps every earlier
  access a later one could race with. `Machine::access_x` drops an earlier access e at a
  write w only when w *covers* e: e is HB-before w, and every access that races with e races
  with w. That holds when w is plain, or when both are strong, in one CTA, and w's scope is no
  wider than e's (moral strength with w then implies it with e). A strong write covers no
  plain access, and an unordered strong pair is kept (`Sh::hw` for writes, `Sh::r` for
  reads); an unordered pair with a plain side is reported. A read replaces only the same
  actor's reads of at least its strength. FastTrack's single last-write epoch broke this:
  warp 0's plain `STS` init followed by its own `ATOMS` became the strong `ATOMS` alone, and
  the other warps' unordered atomics were morally strong with it. The certificate called
  that clean while GB300 was wrong 200/200 (soundness-harness miss 1, `fixtures/
  e_cert_fasttrack`, `tests/wellsync.rs` `plain_write_hidden_by_own_atomic_is_a_race`).
- Where: `crates/sim/src/wellsync.rs`, `wellsync/ring.rs` (C-RING); hooks marked `wellsync.rs` in `exec.rs` (`exec`,
  `mbar_atomic`, `mbar_maybe_complete`, `bar_try_release`, `cluster_try_complete`,
  `warpsync_release`, `access_x`) and the `Machine::wlog` slot (empty, unhashed, dropped by
  clones). `--slices`: per cluster, then the cross-cluster check of E-CTA-INDEP (a certified
  slice's run touches exactly the bytes every schedule touches).
- A failed certificate's unforced runs are executions of the model: a race one of them
  reports (unless the run had LDS/STS plain and `smem::strong_race` may dismiss it) is
  printed (`certificate run race (...)`) and added to the verdict of the exploration that
  replaces the certificate when that exploration does not complete (`run_races`,
  `merge_run_races`; `tests/wellsync.rs`
  `certificate_run_race_survives_an_incomplete_exploration`). C-RING runs with forced reads
  do not count.
- Evidence: `crates/sim/tests/wellsync.rs` runs the certificate on every fixture variant
  and both front-ends: a certified verdict must equal the expected (exhaustive) one, and on
  a representative subset the certified output equals the exhaustive explorer's unique
  output.
- Status: **argued** (checked against the fixture suite).
- If wrong: *miss* (one run would hide a schedule-dependent bug). The conditions refuse
  rather than approximate: any doubt fails the certificate and the explorer decides.

## S: memory and synchronization semantics

### S-MBAR · code
**mbarrier semantics:** phase counter, pending-arrival count, signed tx-count; a phase
completes when pending = 0 and tx = 0; `try_wait.parity p` succeeds when the phase with
parity p has completed (so parity 1 succeeds immediately after init).
- Where: `exec.rs` mbarrier ops, `try_wait_result`, `complete_tma`.
- Status: **validated.** r0 B01 hangs on GB300 as predicted (tx never reaches 0), and the r0
  B02 hang mode reproduced under witness replay. Bitwise agreement holds on all good kernels.
- Arrival counts (2026-11-07): `mbarrier.arrive [a], count` (SASS
  `SYNCS.ARRIVE.TRANS64.ART0` / `.RED.ART0`, the count in a register) subtracts `count` from
  the pending count at once; a count above the pending count is a fault. probes/mbar_count
  (register and immediate counts, shared::cta and shared::cluster addresses, phases
  completed by one or two counted arrivals; test_wait after each step): GB300 50/50
  scrubbed, both front-ends bitwise; its `over` variant (3 arrivals with 2 pending) is an
  unspecified launch failure on GB300. A remote counted arrival in flight (S-REMOTE-INFLIGHT)
  is delivered as `count` single arrivals.
- `mbarrier.expect_tx` / `complete_tx` (2026-11-08; SASS `SYNCS.ARRIVE.TRANS64.RED.A0TR` /
  `.A0TX`, `.A0T1`): no arrival, the tx-count raised / lowered (it may go negative), the phase
  completing when pending and tx are both 0. They are relaxed (PTX ISA): on the thread's own
  CTA's mbarrier they publish only what the thread's last fence released (`membar_vc`),
  unlike an arrival. Before, the model joined the issuer's whole clock, so a store published
  by a bare complete_tx was not a race. On another CTA's mbarrier (SASS `.RED.A0TX` in
  CUTLASS / cuBLAS kernels) the issuer's whole clock is still joined, stronger than relaxed:
  the target's waiter must be ordered after the op landed for the cluster-lifetime check
  (S-CLUSTER-LIFE), and the clock is what carries that. The PTX front refuses that case at
  run time (unmodeled) instead of over-synchronizing. probes/mbar_expect_tx (register / immediate tx,
  shared::cta and shared::cluster addresses, tx below 0, test_wait per step; a fence then
  complete_tx publishing a shared store): GB300 50/50 scrubbed, both front-ends bitwise; its
  `relaxed` variant (no fence) is a race on the PTX front (clean on SASS: S-SMEM-STRONG).

### S-MBAR-MEM · code
**An mbarrier can be initialized by a plain `STS.64` of the encoded state word:**
low = (2^20 − n) << 1, high = (2^20 − n) << 11. This is ptxas's lowering of `mbarrier.init` on
a divergent path.
- **Bit 63 is the initial phase parity** (`SYNCS.EXCH.64` or plain store): cuBLAS starts its
  "empty" barriers with it set, so the first parity-0 wait passes (`probes/mbar_phase1`: with
  bit 63 patched into ptxas's init words, every try_wait result flips; counts unchanged).
  Phase 0 then counts as completed at init.
- Where: `exec.rs` `mbar_decode_mem`, `MbarInit`.
- Status: **validated** (r1a bitwise = GB300; r2_cublas_gemm bitwise = GB300). Count and
  phase bit decoded.

### S-TCGEN05 · code
**tcgen05 ordering (PTX front-end):**
- `tcgen05.ld` executes as a per-thread actor whose view advances only at
  `fence::after_thread_sync`;
- `fence::before_thread_sync` releases that view;
- MMA and commit share one in-order pipe per issuing thread;
- a commit releases every earlier op of that pipe;
- (rung 2) `tcgen05.alloc` / `dealloc` / `relinquish_alloc_permit` are synchronous: the
  thread itself performs them (PTX ISA 9.7.18.6: only mma/cp/shift/ld/st are asynchronous);
- an alloc or dealloc writes every word of its columns: an access of those words that is not
  ordered with it is a race, whichever comes first (`tmem_mark`). Allocs and deallocs do not
  race with each other;
- a warp-wide `tcgen05.alloc` reached by a warp whose lanes are all guarded off is not
  executed by it and does not wait on the allocator (Triton's `@p tcgen05.alloc` with
  p = tid < 32, reached by four warps: before 2026-11-08 the other three warps blocked there);
- (rung 2) a thread's asynchronous tcgen05 ops are ordered after its own earlier
  instructions (the actor's clock includes the thread's clock at issue);
- (rung 2) `tcgen05.st` is asynchronous like `ld`; `wait::st` completes the stores only for
  the same thread's later tcgen05 ops. Another thread sees them through
  `fence::before_thread_sync`, a synchronization, and `fence::after_thread_sync`.
- (2026-11-08) "The same thread's later tcgen05 ops" include its MMA pipe (`tcgen05.mma` /
  `cp` / `shift`): `wait::st` / `wait::ld` order the thread's completed `tcgen05.st` /
  `tcgen05.ld` before the asynchronous tcgen05 operations it issues afterwards, with no
  `tcgen05.fence` (`exec.rs` `tc_wait_orders_pipe`; before, only its later `tcgen05.ld` /
  `st` were ordered, and the PTX front reported a race between a thread's own `tcgen05.st`
  of the A operand and its later `tcgen05.mma` in the mslk mx8mx4 grouped GEMM).
  PTX ISA 9.7.18.6.1: `.wait::*` are synchronous tcgen05 instructions, "inherently ordered
  with respect to other tcgen05 operations" of the thread, and `tcgen05.st -> tcgen05.wait::st`
  is implicitly pipelined (9.7.18.6.2.1); 9.7.18.6.4.2 ("Non-pipelined instructions, same
  thread") orders `st; wait::st; ld` with no fence, and the fence in its example 2 follows an
  mbarrier wait, not a tcgen05.wait. GB300 (`probes/tc_st_mma`, 296 CTAs x 2000 iterations
  per mode, 9.47M words each): an MMA issued by the storing thread right after its own
  `STTM` read the new words in every case, as C and as A in TMEM, with and without the fences
  and also with no `wait::st` (so the probe shows the hardware order but cannot separate
  wait::st from issue order). The SASS front already orders them (STTM is the thread's own
  scoreboarded access), so `sass_gb300` needs no profile item. Stores by other threads of
  the warp or CTA still need the fences (the remaining mx8mx4 PTX races).
  Fixture `s_tc_st_mma` (clean; the MMA issued before `wait::st`: race).
- SASS `STTM` / `LDTM` are scoreboarded (L-MEMSB) and modeled as the thread's own accesses.
- Where: `exec.rs` `tc_actor`, `pipe_actor`, `push_async`.
- Status: **argued, medium confidence.** It follows the PTX ISA's examples. On GB300 the
  fences are NOPs (B06 is invisible), so hardware cannot confirm the rule.
- Races the fences would order are reported as contract findings (E-TC-FENCE-NOP).
- If wrong: either direction.
- Retire: a tcgen05 litmus campaign.

### S-TMA · code
**TMA semantics:** N-D tiled box, innermost dimension contiguous in smem, then the
shared-memory swizzle (none / 32B / 64B / 128B: XOR of address bits [4,4+B) with [7,7+B));
SWIZZLE_* MMA descriptors address the same layout (row r at (r/8)·SBO + (r%8)·span).
Loads zero-fill out-of-bounds elements and complete_tx = box bytes; stores skip
out-of-bounds elements.
4-bit maps (`16U4_ALIGN8B`, case `elem4`) are modeled as byte maps (dims and box in bytes) whose
coordinate 0 counts elements, two per byte (an odd one is reported as not modeled). The global
and shared layouts are packed, so the copy is the byte copy. Evidence: r3_cute_mxf4 (CuTeDSL
MXF4, A and B through such maps) equals GB300 bitwise on both front-ends.
With a swizzle, a box whose inner dimension is narrower than the span has each smem row
padded to the span: element (row q, column c) sits at q·max(span, inner bytes) + c·esize before
the address swizzle (`exec.rs` `tma_box_off`); complete_tx still counts the unpadded box bytes.
**im2col mode** (`cp.async.bulk.tensor.<n>d ... .im2col` loads, SASS `UTMALDG.<n>D.IM2COL`;
`.im2col_no_offs` stores, `UTMASTG.<n>D.IM2COL`; `sim/im2col.rs`): the box is `pixels` rows of
`channels` elements, laid out and swizzled as a 2-D tiled box. Pixel q is the start pixel
(the spatial coordinates and n of the instruction) advanced q steps in row-major order through
the pixel bounding box [lower_i, dim_i - 1 + upper_i] of each spatial dimension (innermost
first; past the last one n advances); element (q, c) is the tensor element at channel c0 + c
and spatial coordinates pixel + offset_i. Offsets are unsigned, packed into 16 bits as ptxas
passes them in the UR (rank 5: three 5-bit fields, rank 4: two 8-bit, rank 3: 16 bits).
Out-of-tensor elements read as 0 (loads) or are skipped (stores). An illegal instruction on
GB300 (reported as a fault): a channel start not on a 16-byte boundary, and a store through a
map whose pixel box reaches outside the tensor (lower corner < 0 or upper corner > 0). Not
modeled: a start pixel outside the bounding box or a start n outside the tensor, element
strides, im2col reductions, Im2colWide maps. Evidence: `probes/tma_im2col` (2026-11-08): 48
loads through six maps (rank 3 / 4 / 5; u8 / u16 / u32; corners from -3 to 2; channels past
C; no swizzle / 64B / 128B; the largest offset per rank) and 12 stores through three maps =
an independent Python model and the simulator, bitwise; the fault conditions were found by
single-access variants (each faulting case gave an illegal instruction; aligned starts and
contained boxes did not).
The swizzle phase comes from the shared-memory address, not from the box: a swizzled box may
start at any 128-byte-aligned address (not only at a multiple of 8 rows x span), and element
(q, c) lands at swz(start + q·span + c·esize). A swizzled start that is not 128-byte aligned
faults (reported as `fault`). This is what multicast slices rely on: cuBLAS
`nvjet_sm103_tst_64x8_64x16_2x4_h_bz_TNT` (cluster of 8) has rank 1 multicast rows 4..7 of a
128B-swizzled tile to stage base 0x2400 + 0x200.
Shared-to-shared bulk copies (`cp.async.bulk.shared::cluster.shared::cta.mbarrier::
complete_tx::bytes`, SASS `UBLKCP.S.S [URd], [URs], URn`) are the non-tensor bulk load with the
issuing CTA's shared memory as the source (read by the bulk actor, proxy-checked like a TMA
store's source), landing in the CTA named by the destination's cluster address and completing
its bytes on the mbarrier at URd+1 (`probes/bulk_s2s`: a 2-CTA cluster, each CTA copying 4 KB
to its peer and 4 KB to itself, 20/20 GB300 runs identical, both front-ends bitwise,
`probe_bulk_s2s_matches_hardware`, 2026-11-06).
- Where: `exec.rs` `complete_tma`, `complete_tma_store`, `tma_swizzle_ok`, `swz`,
  `tma_box_off`; `case` `tmap_geometry`.
- Evidence (start alignment): probes/tma_mcast_swizzle (GB300, 2026-11-03). Spans 32/64/128 B
  and starts 0..0x780 in 128-byte steps (48 configs): a cluster-of-4 multicast load
  (rank r loads 4 rows to start + 4r·span on every CTA) and a TMA store both match
  swz(address) in 48/48. The box-relative model (start + swz(offset)) fails every load and
  every store whose start is not a multiple of 8 rows x span. Starts 0x10, 0x20, 0x40 fault with
  "misaligned address" for every span. fixtures/c_mm_bf16_mcast (that nvjet kernel):
  certified (E-WELLSYNC), `hw` 1000/1000 bitwise = simulator, also with smem scrubbed. Before,
  the simulator required (8 x span)-alignment and reported a false `violation`, which
  defeated the certificate; the exploration then ran past 200k states, and `compare` printed a
  false MISMATCH (it compared one explored cluster's output with the whole launch's).
- Status: **validated** for in-bounds 2-D/3-D boxes, no swizzle (r0, r1a) and 128B swizzle
  with TMA store (r1b: bitwise 1000/1000 with smem scrubbed). Narrow inner boxes:
  probes/tma_narrow_box (GB300, 2026-11-03) loads and stores a 16-row u16 box for every
  swizzle (none/32B/64B/128B) and inner width 16..128 bytes: the padded layout matches all 13
  configurations, the dense one fails all 6 with inner < span; fixtures/c_mm_bf16_ttn (cuBLAS
  8 x 16 bf16 box, SWIZZLE_64B) is bitwise equal to the GPU. Out-of-bounds fill/skip is
  **unvalidated** (no fixture exercises it yet; rung 3, B25). Element addresses wrap modulo
  2^64: trtllm-gen's dynBatch output map (fixtures/r3b_trtllm_bmm) has dims 2^31 and a stride
  of 2^35 - 256 that its coordinate 2^30 cancels, and it clips padded token rows with a
  coordinate in an 8-row dimension; the store matches GB300 bitwise.

### S-TMA-2CTA-MCAST · code
**`UTMALDG.<N>D.MULTICAST.2CTA`** (cp.async.bulk.tensor ... .multicast::cluster.cta_group::2;
cuBLAS's 2-CTA nvjet kernels, picked for bf16 M = N >= 512): the box lands at the same offset
in every CTA rank of the ctaMask (UR bits 16..31, as `.MULTICAST`), and each destination's
transaction completes on the mbarrier at the operand's offset in the CTA of the destination's
pair whose peer bit is the mbarrier address's own (CUTLASS's 2SM multicast loads clear the peer
bit: the pair leader's mbarrier).
- Where: `exec.rs` `Op::TmaLoad` (`mask` with `pair`).
- Status: **validated** (2026-11-05) by `probes/tma_2cta_mcast`: cluster of 4, one load of a
  512-byte box by rank 0 with ctaMask 0xf / 1 / 2 / 4 / 8 / 3 / 6, the mbarrier of rank 0 or 1,
  the destination address of rank 0 or 1, with and without `.cta_group::2`; each CTA's
  mbarrier expects 512, 1024, or the model's bytes, and is polled for ~10 ms (168 configs).
  On GB300 the box lands in every masked rank at the destination's offset (the destination
  address's rank does not matter); without `.cta_group::2` each destination completes its
  own mbarrier (the mbarrier rank does not matter); with it, the mbarrier at the offset in
  rank (r & ~1) | (mbarrier rank & 1), so a pair leader's mbarrier gets 1024 bytes for mask
  0xf. Every flag agrees with the rule (an mbarrier that gets fewer bytes than it expects
  never completes; more: either, the transactions can step over zero, not used), and the 56
  configs where every mbarrier expects exactly the model's bytes equal both front-ends bitwise
  (`crates/sim/tests/tma_2cta_mcast.rs`). End to end: the cuBLAS 2-CTA nvjet kernels above
  (E-PAIR) use `UTMALDG.3D.MULTICAST.2CTA` and equal GPU bitwise.
- If wrong: *false alarm* (deadlock / tx mismatch on the wrong mbarrier) or *miss* if a
  transaction completes an mbarrier that should still wait.

### S-TMA-GATHER4 · code
**`cp.async.bulk.tensor.2d...tile::gather4 [d], [map, {c, r0, r1, r2, r3}], [m]`** (SASS
`UTMALDG.2D.GATHER4`) loads `box[0]` elements at column c of each of the four rows into shared
memory exactly as a `box[0] x 4` tiled box at d would land (rows in operand order, padded to
the swizzle span, the address swizzle, S-TMA), zero-filling out-of-bounds elements (rows
outside the map or negative, columns past either edge); complete_tx = 4 x box[0] x element
bytes. The map is an ordinary tiled 2-D map with box height 1 (trtllm-gen's token map:
box 128 x 1, u8, 128B swizzle); anything else is reported. A column whose byte offset is not a
multiple of 16 is an illegal instruction on GB300 (reported as `fault`).
- Where: `sass` `UTMALDG` lowering (5 coordinates); `ptx` `.tile::gather4`; `exec.rs`
  `complete_tma`.
- Status: **validated**: fixtures/s_gather4 (six gathers through 128B / 64B / 32B / no-swizzle
  maps, narrow 128B box, rows 64, 1000, -1 and repeated rows, columns -8 and past the right
  edge, destinations at 128-byte offsets not aligned to 8 rows x span) = GB300 1000/1000 on
  both front-ends, and its recorded bytes. Columns 3 and 100 (u16) and 4 (u16, 128B) are
  illegal instructions on GB300 (variant `misaligned`); fixtures/r3b_trtllm_bmm = GB300 bitwise.
- If wrong: wrong shared-memory images of gathered rows (bitwise mismatches).

### S-PROXY · code
**Async-proxy visibility of generic writes.** A generic-proxy store to shared memory
(`st.shared`, cp.async) is visible to a later async-proxy read (TMA store source,
`tcgen05.mma` operand) only if some thread executed `fence.proxy.async` after the store
happened-before it (the writer itself, or a thread that synchronized with the writer), and
that fence happens-before the async operation's issue. Otherwise the result is a proxy race.
- Where: `exec.rs` `proxy_check` (each fence's vector clock recorded in `Op::Fence`).
- Rung 5: `probes/proxy_fence` answers the open question below. Rung 1b's pipeline with the
  fence moved after the epilogue's barrier, executed by every thread (Triton's order) or only
  by the issuing thread, is correct in 1000/1000 scrubbed runs on GB300; without any fence,
  1000/1000 runs are wrong (994 distinct outputs). Before, a fence by another thread was not
  credited, and Triton's flash attention (`cp.async.wait_group; bar.sync;
  fence.proxy.async; tcgen05.mma`) was a false alarm.
- Status: **validated** (the necessity direction). Rung 1b B04 deletes the epilogue's
  `fence.proxy.async` (in SASS: `MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC.S`); with shared memory
  scrubbed, GB300 writes a wrong output in **1000/1000 runs (61 distinct outputs)**, and both
  front-ends report the proxy race. The sufficiency side is the writer-side formulation
  CUTLASS uses (every epilogue thread fences, then the named barrier, then one thread issues
  the store); the good kernel matches GB300 bitwise in 1000/1000 scrubbed runs. A fence by a
  different thread that happens-after the write is credited too (rung 5 probe above).
- If wrong: *false alarm* if hardware/PTX also credits other threads' fences; *miss* if
  fences need more (e.g. a specific scope).
- Open: closed by `probes/proxy_fence` (see above).
- State spaces (2026-11-06). ptxas on sm_103a: `fence.proxy.async` and
  `fence.proxy.async.shared::cluster` are `MEMBAR.ALL.GPU` + `FENCE.VIEW.ASYNC.S`,
  `.shared::cta` is `MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC.S`, `.global` is `FENCE.VIEW.ASYNC.G`
  alone, and `tcgen05.wait::st` is `FENCE.VIEW.ASYNC.T` (one per wait in probes/tmem_ldred).
  Only the `.S` forms count for the shared-memory check. `FENCE.VIEW.ASYNC.G` /
  `fence.proxy.async.global` orders global memory only and covers no check
  (`FenceKind::ProxyAsyncGlobal`); `FENCE.VIEW.ASYNC.T` lifts as `tcgen05.wait::st`. Before,
  all three lifted as the shared-memory fence, so a `.global` or `.T` fence hid a missing
  `.shared::cta` one. probes/proxy_fence variant `global` puts `fence.proxy.async.global` in
  place of the epilogue's fence (SASS `FENCE.VIEW.ASYNC.G`, no `MEMBAR`): with shared memory
  scrubbed, GB300 writes a wrong output in 1000/1000 runs (528 distinct outputs), and both
  front-ends report the proxy race.
- Not modeled: generic-proxy writes to global memory read by an async-proxy op (TMA load from
  global) are not checked for a `.global` fence, so a missing one is a *miss*.

### S-MULTIMEM · code
**NVLS multicast memory.** A case's `[[multicast]]` objects (capture: `cuMulticastCreate`,
`cuMulticastAddDevice`, `cuMulticastBindMem`, `cuMemMap` of the multicast handle) get an
address range of their own; each `binds` entry maps a range of it to a buffer on one GPU.
- `multimem.ld_reduce` (SASS `LDGMC`) loads the bytes from the bound location of every GPU
  and reduces them (N-MULTIMEM). `multimem.st` / `multimem.red` (SASS: plain `STG` / `REDG`
  with the address in the multicast range; `probes/multimem/forms_sass.txt`) store / reduce
  at every bound location. Each per-location access carries the instruction's semantics and
  scope (race detection, release / acquire) and they happen at one step, in GPU order.
- ptxas emits `.STRONG.SYS` for every `multimem.ld_reduce`, `.weak` included; the SASS front
  lifts `LDGMC` as relaxed `.sys`, the PTX front keeps `.weak` as a plain load.
- PTX front: a non-multimem access to a multicast address, or a multimem operation on another
  address, is undefined behavior (PTX ISA 8.2.3) and reported as a `violation`. SASS front:
  `STG` / `REDG` on a multicast address are multimem stores / reductions; any other access
  to one (LDG, ATOMG with a result, TMA) is `unmodeled`. A multicast range a GPU has no
  memory bound at is `unmodeled`.
- Status: **argued**. GB300 (`probes/multimem`, 4 GPUs): `multimem.st` lands in every member
  (`results/st.txt`); concurrent `multimem.red.release.sys.add.s32` from 4 GPUs gives the exact
  count in every member; `multimem.ld_reduce` reads every member (N-MULTIMEM). "At one step" is
  not a hardware claim: the PTX model treats the per-location accesses as independent, in an
  unspecified order.
- Where: `crates/case` `Multicast`; `lib.rs` `build_launch`; `exec.rs` `mc_resolve`,
  `mem_locs`, `Op::LdReduce`, `Op::St`, `Op::Atom`; PTX front `"multimem"`; SASS `LDGMC`.
- If wrong: *miss* of a protocol that observes two members of one multimem store or
  reduction in different orders with relaxed reads (the model lands them together).
- Retire: model each per-location access as its own scheduled event.

### S-MC-ALIAS · code
**Alias proxy between multicast and unicast views.** The multicast address and a unicast
address of the same bytes are virtual aliases (PTX ISA 8.6, 8.9.5: "proxy-preserved base
causality order"). An access through one view is ordered after a write through the other only
if a `fence.proxy.alias`, executed by the writer after the write or by a thread that observed
the write first, happens before the access; a kernel boundary on the same GPU (stream order)
counts as such a fence (PTX: CUDA API synchronization participates in proxy-preserved order).
Otherwise: `race:alias`.
- Pairs of strong accesses are not checked, and release / acquire through aliased flags
  synchronizes as through one address: NVLS barriers write a flag with `multimem.red` and poll
  it through the unicast address. PTX defines moral strength per memory location and does not
  say how it treats aliases; GB300 orders the probed shape (E-MC-ALIAS-HW).
- The shadow of every global byte records the view of its last write (`Sh::mcv`).
- Not modeled: unicast mappings of one buffer at different virtual addresses on different
  GPUs (peer memory) are also aliases under the PTX definition; they are treated as one
  address, as before (torch symmetric memory and DeepEP never fence them). Write-after-read
  across views is not checked.
- SASS: L-ALIAS-FENCE identifies the fence.
- Status: **argued** (PTX ISA text). GB300 does not need the fence in the probed shape
  (E-MC-ALIAS-HW).
- Where: `exec.rs` `alias_check`, `alias_mark`, `launch_cta`, `Op::Fence { ProxyAlias }`.
- If wrong: *false alarm* where hardware orders the views (see E-MC-ALIAS-HW).

### E-CPASYNC-PROXY · code
**cp.async writes and async-proxy reads.** The PTX ISA performs cp.async in the generic
proxy, so an async-proxy read of the bytes (TMA store source, `tcgen05.mma` operand) needs
a `fence.proxy.async` after the copy completed (cp.async.wait_group / wait_all) that
happens-before the read. ptxv reports a missing fence as a `contract` finding (the verdict
stays clean, "relies on E-CPASYNC-PROXY"), not a race.
- Where: `exec.rs` `proxy_check`: a shared-memory byte whose last write is a `cp.async`
  (`LDGSTS`; the copy completes in the issuing thread's name at the wait). Before
  2026-11-06 it was a proxy race like an `st.shared` write.
- Evidence: Triton kernels that fill shared memory with cp.async (LDGSTS), then
  `cp.async.wait_group; bar.sync`, then `tcgen05.mma` with no fence anywhere: mslk fp8
  groupwise `.pinned` (two captures) and msl_ops `triton._wgrad.bf16.72x128`, 1000 scrubbed
  runs each on GB300, every run bitwise equal to the simulator (F2, KA, 2026-11-06).
  probes/proxy_cpasync (cp.async, wait_all, bar.sync, TMA store; variants with and without
  the fence) is correct in 1000/1000 scrubbed runs either way, but its `st_none` variant
  (st.shared writers, no fence) is also correct, so the probe does not discriminate; rung 1b
  B04 (st.shared writers) fails 1000/1000.
- Status: **unvalidated**: no failure seen, but no probe discriminates and no mechanism is argued.
- If wrong: *miss* reported only as a contract finding.

### S-INIT · code
**Initial state:**
- registers and shared memory hold garbage until written, and reading them first is reported
  (`violation`);
- the reserved first 1 KiB of shared memory (ptxas/driver bookkeeping) is initialized at
  launch, as zeros;
- global buffers are initialized by the host.
- Where: `machine.rs` `Machine::new`; `exec.rs` `exec_thread` (registers) and `access_x`
  (smem).
- Status: **enforced** (2026-10-30), with these limits:
  - register reads are checked for operands that `Op::src_regs` lists (ALU, loads, stores,
    atomics). Address operands of mbarrier/TMA/tcgen05 ops are not checked yet;
  - the simulated value of an unwritten location is 0, so numerics on such a run are
    meaningless (the verdict already says so);
  - the zero contents of the reserved prefix are validated only by bitwise agreement.
- Evidence:
  - s_init B48 (partial producer): flagged on both fronts; GB300 is wrong in 200/200 scrubbed
    runs.
  - r0 B47 (deleted accumulator init): flagged on PTX. On SASS, ptxas reuses `%tid.x`'s
    register, so no unwritten register is read, but the numerics spec check fails. Outputs are
    bitwise those of the good kernel.
- **Undefined values propagate through register-only operations** (rung 4; predicates too
  since 2026-11-06, below): a written R / UR register keeps a mask of undefined bits. An ALU operation that reads undefined bits computes
  the undefined bits of its result, bit-precisely for moves, selects, byte permutes (`PRMT`),
  constant shifts, bitwise LUTs, low bits of adds and multiplies, and `F2FP` packing, and
  wholly otherwise; SHFL carries the source lane's mask. The read is reported only when an
  undefined bit reaches anything else: memory (stores), addresses, predicates and control
  flow, synchronization operands. Triton's cross-warp reductions shuffle and combine registers
  that only some lanes loaded, and ptxas's bf16 unpacking (`PRMT R9, R8, 0x7632, R9`) reads
  a register whose undefined half the next shift discards; both are correct code.
  (rung 5) Spills keep the mask: `STL` stores it with the data and `LDL` restores it
  (never-stored local memory is undefined). A compare whose combining predicate decides the
  result alone (`ISETP.NE.OR P6, PT, R18, RZ, P2` with P2 set) does not use its operands:
  torch's `where` loads R18 only under !P2.
- **Poison propagation to sinks** (2026-11-06). Undefined values are reported only where they
  reach a sink; register-only operations (moves, integer / float arithmetic, compares,
  selects, `PLOP3`, `P2R` / `R2P`, conversions) make their outputs undefined instead:
  - a predicate computed from undefined bits is undefined (`wp`): compares (`ISETP` /
    `FSETP` / `DSETP` / `HSETP2` / `FCHK`), carry-outs (`IADD3 R11, P0, PT, R2, R11, RZ` in
    Triton's `_philox_checksum_kernel`, where lanes that skipped a predicated `LDS` add
    undefined registers and the predicated store discards the result), `LOP3` predicate
    outputs, `PLOP3` outputs that depend on an undefined input, and an `ISETP` whose
    combining predicate is undefined. Before, each of these was reported at the operation;
  - an undefined guard on a register-only operation makes its destinations undefined (new
    value or old one);
  - `R2UR` and `HMMA` with an undefined input make their destinations undefined (`HMMA`
    mixes lanes: all of them);
  - packed f32x2 `FFMA2` / `FADD2` / `FMUL2`: each lane of the result is undefined only if
    the same lane of an operand is (2026-11-06; before, either lane tainted both: a false
    alarm on CuTeDSL FlashAttentionBackwardPreprocess, which packs two rows in one FADD2,
    `fixtures/e_uninit_f32x2`); PTX front (2026-11-08): a virtual register keeps undefined
    bits per bit (`Thread::vund`), `mov.b64 %rd, {lo, hi}` / `mov.b64 {lo, hi}, %rd` keep each
    half's, and `.f32x2` operations read each lane's. Before, a PTX virtual register was
    defined or undefined as a whole: FlashAttentionBackwardPreprocess 1x1x16384 repacks
    products of two rows into one 64-bit register, the out-of-range row poisoned the in-range
    one, and the PTX front reported undef-insensitive at ptx:647 where SASS is clean (a lowering
    imprecision, not an uninitialized read; `fixtures/e_uninit_f32x2_lanes`). Also on the
    PTX front (2026-11-08): a store of a virtual register marks undefined only the bytes
    whose bits are, and integer conversions of virtual registers (`cvt.u16.u32`,
    `cvt.s32.s8`, ...) keep the low bits' undefined bits (a sign extension takes the sign
    bit's). Before, Triton mxfp8 / nvfp4 quantize kernels stored the defined low byte of an
    `e4m3x2` / `e2m1x2` pack whose other byte came from an unwritten register, and the PTX
    front reported undef-insensitive where SASS is clean (`fixtures/e_uninit_cvt_lanes`).
    Conversions of a 16-bit or narrower source read only its bits (2026-11-08):
    `F2F.F64.BF16` / `.F16` (and FRND / F2F of 16-bit formats) the selected half, `I2F.S16` /
    `.U8` the low bits or `Ra.H1`. Before, `F2F.F64.BF16 R2, R16` after ptxas's
    `PRMT R16, RZ, 0x7610, R16` (zero into the low half, the high half unwritten) was
    undefined (`fixtures/e_uninit_half_cvt`);
  - lane by lane too (2026-11-06): `F2FP` narrow packs (`PACK_AB[_MERGE_C]`, `.RS` random
    bits taint both lanes) and unpacks (`UNPACK_B`), `HADD2.F32 Rd, -RZ, Ra.Hx_Hx` (one
    half), `HSETP2.BF16_V2` (the selected half of each operand); `FSETP` / `DSETP` / `HSETP2`
    whose defined combining predicate decides the result (`AND` false, `OR` true) are
    defined. msl_ops triton `_nvfp4x2_k_quant_kernel` (25 FCHK sites) and `_metrics_2d`
    were reported before (`fixtures/e_uninit_lanes`);
  - through memory (2026-11-06; generic and shared::cluster addresses, DSMEM included, since
    2026-11-07: BlackwellHistogramSelect's `ST.E.128` into a peer CTA): a shared or global
    store (or `atom.exch`) of undefined data
    marks the bytes undefined (`Sh::undef`) instead of being reported; a load or an
    `atom.exch` of them gives undefined registers (as for never-written shared memory); any
    other reader (TMA, MMA, other atomics) reports; global bytes still undefined at a terminal state are reported
    (`uninit-gmem-exit`), except the tail of a used tensor map (S-TMAP-TAIL). Before, the
    store was the report (cute FusedRMSNormCombineFwdParallelTOPKStream's reduction
    buffer: clean now, `fixtures/e_uninit_smem`). Per byte (2026-11-07): a vector store
    marks only the bytes of its undefined registers (SASS: per byte of each 32-bit register;
    PTX `st.vN`: per element), and a store of the bytes a location already holds makes them
    defined unless an unordered store of undefined data may land after it. Before, one
    undefined register of an `STS.128` / `ST.E.128` marked all 16 bytes: msl
    BlackwellHistogramSelect rank 0 reads entry 56 of a vector store whose entry 59 is
    undefined (GB300 with 21 adversarial shared-memory fills: outputs bitwise equal to the
    clean run; `fixtures/e_uninit_vec`).
  - `REDUX` (2026-11-07): an undefined bit in any active lane's operand makes the result
    undefined in every lane (a sum carries into all bits), reported where it is used. Before,
    the operand was a sink: Triton's cross-warp reduction (corpus pilot dash/5819, `_col_phase`)
    loads the warp partials only in threads tid < 32 (`@!P1 LDS R12`), every warp runs
    `REDUX.SUM.S32` on R12, and only thread 0 stores the result (`fixtures/e_uninit_redux`).
  - `STSM` data marks the shared bytes undefined like `STS`, and an `LDSM` of undefined
    bytes gives undefined registers (2026-11-07; their addresses stay sinks).
  - TMEM (2026-11-07): a `tcgen05.st` of undefined data leaves the words as never written;
    a `tcgen05.ld` of never-written words gives undefined registers (a `tcgen05.ld.red`
    result too), reported where used; the load itself is a `note` (`tmem-uninit-ld:<pc>`, was
    a violation). An MMA that reads never-written TMEM (accumulator with `enable_input_d`, A
    in TMEM, scale factors) leaves all of its D undefined (never written) and records a
    `note` (`mma-undef:<pc>`); before, an accumulator or scale-factor read was a violation
    (`mma-uninit-acc` / `-sf`) and an A read was not checked. Triton's pipelined attention prologue
    (corpus pilot dash/5713, `_t_causal_attention_kernel`) skips the first QK^T MMA when the
    loop runs zero times but its `LDTM` stays unconditional (GB300 with TMEM poisoned by a
    NaN tcgen05 matmul on every SM before each call: 0/20 wrong outputs, finite-fill control
    0/20; corpus pilot findings, 2026-11-07). `fixtures/e_uninit_tmem` is that shape with
    the scores unused: clean. dash/5713 itself stays reported: the scores feed the softmax
    (row max, `MUFU.EX2`), the O rescale factor (`FMUL R4, R111, R4` at 0x2eb0 on O loaded
    from TMEM) and P (`STTM`, then the PV MMA), and reach the output stores; that the result
    does not depend on them (O is zero there, m is reset by `FSEL R152, R152, -INF, P0`) is
    a property of the values, which this register-level propagation does not decide; the
    multi-fill rerun reports it `undef-insensitive` (S-INIT-FILLS).
  - Debug knob: `PTXV_UNDEF_TRACE=<thread>` prints each instruction the thread runs on
    undefined inputs (where a report's undefined bits came from).
  - sinks (reported): store data and addresses, every load / atomic / async-copy / TMA /
    tcgen05 / mbarrier / barrier operand that `Op::src_regs` lists, a guard of anything
    that is not register-only (branches, `EXIT`, memory, synchronization), branch conditions
    and `BRA.DIV` masks, `STSM` / `LDSM` operands, `SHFL` lane and clamp (its data keeps the
    source lane's mask), votes, `WARPSYNC` masks, `ELECT`, and every operation not
    listed as register-only. Anything not known to be register-only is a sink.
  - Where: `exec.rs` `undef_result`, `wp`, `reg_only`, `sinit_pre`, `poison_regs`,
    `exec_thread_inner`.
  - Evidence: `fixtures/e_uninit_discard` (`tests/fixtures.rs`
    `undefined_values_discarded_by_predicated_store_are_clean`): a predicated `LDS`, a 64-bit
    add with carry on the loaded registers, a store under the same predicate: clean on both
    fronts; variants that store unconditionally or branch on the carry are reported. The
    msl_ops `_philox_checksum_kernel` case is clean and matches its captured GPU output
    bitwise.
- **Loads of never-written shared memory** do not report by themselves: the loaded registers
  become undefined and a later read of them is the finding (`Machine::stale`). cuBLAS loads a
  try_cancel response's ctaid before testing its success bit, and never uses it when the
  request did not run. A multi-register `LDS` (`.64` / `.128`) makes undefined only the
  registers whose bytes nothing wrote: cuBLAS `gemmSN_TN_kernel`'s split-K reduction reads rows
  of 9-word slots with `LDS.128`, the 9th word of each slot padding nothing writes and nothing
  uses (fixtures/c_mm_f32_gemmsn_tn: clean, `hw --scrub` 1000/1000 = simulator with smem
  NaN-poisoned; before, all four registers were undefined and the output store was reported).
- If wrong: *miss* for unchecked operand kinds.

- `discard.global.L2 [a], 128` (PTX front, 2026-11-07, 3 SCALE corpus kernels): the line's
  contents become indeterminate (PTX: a weak write of an unstable indeterminate value),
  modeled as weak stores of a register nothing writes, so a later load gives undefined data
  and the write is race-checked. probes/discard_l2: write, discard, rewrite, read = GB300 50/50
  bitwise; `read_after` (no rewrite) is a violation, while GB300 returned the pre-discard
  values in all runs (the PTX contract allows either). SASS `CCTL.E.RML2` stays refused.
### S-INIT-FILLS · code
**Multi-fill rerun (an evidence class, not a certificate).** When every S-INIT report of a
verdict is undefined data left in global memory at the end of the launch
(`uninit-gmem-exit`), no undefined bit reached a branch, address, predicate, guard,
synchronization operand or non-load reader on the explored paths (those are sinks, reported
as `uninit-reg` / `uninit-smem` / `uninit-gmem`, and any of them refuses the rerun). The
launch is rerun on one deterministic schedule per fill, with every unwritten register,
uniform register, shared byte outside the reserved prefix, TMEM word and never-stored local
word holding the fill: 0, NaN (0x7fffffff), +Inf, -Inf, and two pseudo-random patterns. If
the output buffers are bitwise equal in all six runs (each run reaching every reported
undefined output, and no run reporting a sink), the
reports become kind `undef-insensitive`, with the note `undef-fills: 6 fills ... outputs
bitwise equal`; otherwise they stay violations and the note says which fill differed.
  Not run when the verdict has a race or several outcomes: undefined data from an
  ordering bug (a read before its write in some schedule) depends on the schedule.
- Where: `fills.rs` (`check`, called by `explore`, `explore_dpor`, `check_certified`);
  `PTXV_NO_FILLS` turns it off.
- Status: **argued** (2026-11-07). The gap: six fills are six points of a function of the
  undefined values; an output that differs only for other values (a compare against a
  specific constant, an exact cancellation that NaN and Inf break but random words do not)
  is missed. One schedule per fill: outputs that also depend on the schedule are compared
  on that schedule only. Undefined predicates are not filled (they would be sinks).
- Evidence: `fixtures/e_uninit_fills` (min(x, -inf) of an undefined x: undef-insensitive;
  x + 1: violation). Corpus pilot dash/5713 (Triton `_t_causal_attention_kernel`): Triton
  reads never-written TMEM (the scores of a QK^T MMA its peeled prologue skipped), and
  correctness depends on later value masking (O is zero there, m is reset to -inf, the PV
  result is overwritten): undef-insensitive in 6 fills, consistent with GB300 0/20 wrong
  under a NaN TMEM poison. The PTX good variants of `e_uninit_f32x2` / `e_uninit_lanes`
  were undef-insensitive too until PTX registers kept undefined bits per lane (2026-11-08;
  clean since, as on SASS).
- If wrong: *miss* of an output that depends on undefined values only at untried values.

### S-INIT-PRED · code
**Predicates start undefined too, and reading one before any write is reported** like an
unwritten register (`violation`, `uninit-reg:<pc>`). Until 2026-11-05 `Thread::preds` was
zero-filled with no undefined state, so a never-written predicate read as false in every
lane. The soundness harness (docs/soundness-harness.md, miss 2) NOPed the cuBLAS tile
scheduler's fast-path `SYNCS.PHASECHK.TRANS64.TRYWAIT P4` in five nvjet kernels
(`c_mm_bf16_{mcast,nnt,ttn}`, `c_smm_nvf4`, `r2_cublas_gemm`): the `BRA !P4` after it read the
unwritten P4, every lane took the retry stub (the good kernel), and ptxv certified clean while
GB300 was wrong or hung in 198/200 runs. On GB300 the unwritten P4 is not false in every lane
(`tools/mutate/repro/uninit_pred` controls, 200/200 correct each).
- Defined: SASS `PT` / `UPT` (constants), and every predicate (per thread; uniform ones per
  warp) once an executed instruction writes it (`Machine::wp`: `ISETP`, `PLOP3`, `R2P`,
  votes, `SYNCS` results, ...).
- Reads checked: the guard of every instruction (the SASS lifter makes a branch condition
  the guard), in every member lane (`exec_inner`); predicate operands through `Op::src_regs`
  (`reg_umask` of `File::P` / `File::UP`), with S-INIT's rule that a register-only operation
  computes on undefined inputs and reports only where the value is used:
  - `SEL` / `FSEL` on an undefined predicate: an undefined result;
  - `PLOP3`: an output that depends on an undefined input is itself undefined, reported where
    it is used (ptxas's `PLOP3.LUT P0, PT, PT, PT, PT, 0x8, 0x0` constants do not depend;
    the uniformizing loop's first `PLOP3.LUT P0, PT, P0, P1, ...` reads the unwritten P0 and
    its result is overwritten by `R2UR P0` unread: probes/memhints). Until 2026-11-06 the
    read itself was reported. An operation computed from undefined bits leaves its predicate outputs
    undefined (`Machine::wp`);
  - `P2R` keeps the undefined bits of the predicates it packs and `R2P` restores them (a
    spilled `PR`);
  - an `ISETP` whose combining predicate is undefined gives an undefined result (S-INIT's
    "decided" rule needs a defined combining predicate); reported until 2026-11-06;
  - since 2026-11-06 undefined predicates also arise from undefined register inputs, and
    are reported where they reach a sink (S-INIT, "Poison propagation"). The NOPed try-wait
    below is still reported: the `BRA !P4` is a sink.
- Where: `machine.rs` `Thread::pdef`, `Warp::up_def`; `exec.rs` `reg_umask`, `pred_undef`,
  `wp`, `exec_inner`, `undef_result`, `Op::R2p`; E-SPIN / C-PEEK state comparisons
  (`exec.rs`, `spin.rs`, `wellsync.rs` `reg_diff`) include the defined bits.
- Status: **enforced** (2026-11-05). Evidence: `fixtures/e_uninit_pred` (`tests/wellsync.rs`
  `branch_on_unwritten_predicate_is_reported`): the NOPed try-wait is reported at the
  `BRA !P4` (0x1b80); the good kernel and the three GB300-correct controls still certify.
  Every fixture keeps its verdict on both front-ends (full suite).
- If wrong: *false alarm* where hardware defines a predicate the model does not write (an
  unmodelled instruction's predicate output); *miss* only for predicates the lifter writes
  that hardware does not.

### S-TMAP-TAIL · code
**GB300's TMA ignores bytes 0x40..0x7f of a 128-byte tensor map in global memory.** A store
that leaves undefined data there is a `contract` finding (`tmap-tail-undef:<pc>`), not an
S-INIT violation, when a TMA of the launch uses that map; a TMA reading undefined bytes in
0x00..0x3f is a violation (`uninit-gmem`). msl_ops dist_moe (`dm_combine_w2` / `w4`) copies
only the 64-byte template and publishes the rest of the map with lanes 16..31's undefined
registers (`ATOMG.EXCH` at 0x2790).
  The tensormap proxy fence (`fence.proxy.tensormap`, `UTMACCTL.IV`) reads the 128 bytes with
  the same exemption.
- Where: `exec.rs` `tmap_at` and `Op::TmapFence` (`undef_ok_from`), `undef_gmem_check` (from
  `exit_checks`).
- Status: **validated** (`probes/tmap_tail`, GB300, CUDA 13.0, 2026-11-06): 24 driver-encoded
  rank-2 maps (u32 / f32 with NaN OOB fill; swizzle none / 32B / 64B / 128B; boxes inside
  and across both tensor edges), each loaded through global memory with bytes 0x40..0x7f as
  encoded (the derived box-bytes / swizzle-repeat words are nonzero), zero, all-ones and
  random (3 runs each): every shared-memory tile bitwise equal to the first encoded run's (0
  of 216 loads with the tail altered differ, 0 of 48 repeat encoded loads). Library-built maps (cuBLAS packs maps 64 bytes apart) were already
  evidence that bytes 64..127 are not read (docs/tensormap.md). Not probed: rank > 2,
  im2col, sub-byte types.
- If wrong: *miss* of a TMA that depends on the undefined tail.
- Retire: NVIDIA documentation of the descriptor layout.

### S-DSMEM · code
**A CTA's shared addresses are `(cluster rank << 24) | offset`**, its window inside the
cluster's shared window. A `shared::cluster` access goes to the CTA named by the rank byte. In a
cluster launch, a `shared::cta` access whose rank byte is not the CTA's own is a fault (GB300
traps with *illegal instruction*: `probes/reserved_smem`, variant `cluster_untagged`). `mapa`
replaces the rank byte.
- Evidence: ptxas forms every shared address as `LEA R, SR_CgaCtaId, off, 0x18`, and lowers
  `mapa ..., 0` to `PRMT R, RZ, 0x654, R` (clear the top byte).
- `mapa` with a rank outside the cluster (2026-11-08): ptxas lowers mapa to `PRMT R, rank,
  0x654, a` with no check, and CuTe kernels compute a peer address in every lane (rank =
  %laneid) and use it only in lanes below the cluster size. The result is an undefined
  address (S-INIT), not a fault; an access through it is reported. Before, the mapa itself
  faulted (PTX front runs of CuTe 2-CTA GEMMs). probes/mapa_lanes: GB300 50/50 scrubbed, both
  fronts bitwise; variant oob (a lane stores through rank 2): reported on both fronts, GB300
  fails the launch.
- Where: `exec.rs` `sreg` (`SharedAddr`), `resolve`, `rank_cta`, `Mapa`.
- Status: **validated** by rung 1c bitwise agreement (remote arrive + multicast TMA).
- If wrong: *miss* / *false alarm* on DSMEM addressing; would show up as lift disagreements.

### S-SWIN · code
**The generic shared window.** A generic address whose high word is `SR_SWINHI` is the
shared::cluster address in its low word: `(SWINHI << 32) | (rank << 24) | offset`. ptxas forms
`cvta.shared` as `{SR_SWINHI, LEA(SR_CgaCtaId, off, 0x18)}` and a generic `mapa.u64` as
`PRMT R, Rrank, 0x654, Rlo` (replace the rank byte); `LD.E` / `ST.E` through such an address
reach that CTA's shared memory. The model's window is `0x7c00_0000_0000` (GB300's high word
was 0xfffa, near its global allocations); only its being one 4 GB window matters. QSPC of a
window address is reported as not modeled.
- Evidence: flashinfer's routing cluster kernel reads peer CTAs' shared memory this way.
- Where: `machine.rs` `SHARED_GENERIC_BASE`, `exec.rs` `sreg` (`SharedWindowHi`), `resolve`,
  `Cvta`, `Mapa`.
- Status: **validated**: `fixtures/s_routing_forms` (cvta.shared, generic mapa, `LD.E` /
  `ST.E` / `LD.E.U16` into this and the peer CTA) = GB300 bitwise, 1000/1000.
- If wrong: *miss* / *false alarm* on generic DSMEM accesses; shows as a bitwise mismatch.

### E-PAIR · code
**`cta_group::2`:** the PTX paired `tcgen05.alloc/dealloc/relinquish` act on each CTA's own TMEM
(both CTAs of the pair issue them; same allocation in each). The pair MMA is issued by the even
CTA; CTA i supplies A rows and receives D rows 128i.., and supplies B rows (N/2)i.., all at the
same descriptor / TMEM addresses. A multicast commit becomes one commit per destination CTA in
the issuing pipe.
- Where: `ptx` tcgen05 lowering, `exec.rs` `Op::Mma` (pair), `complete_mma`, `Op::TcCommit`.
- Status: **validated** by `fixtures/r1c_2sm`: an fp64 reference first confirmed the operand
  split on GB300, then the PTX model matched GB300 bitwise (1000/1000, scrubbed). B12 (commit
  mask without the peer) hangs on GB300, as the model's deadlock predicts.
- Block-scaled pair MMAs (N-MMA-BS): each CTA's D rows use the scale factors in its own TMEM.
  The pair `tcgen05.cp.cta_group::2` (SASS `UTCCP.T.S.2CTA.4x32dp128bit`, S-TCCP) is issued by
  the even CTA and copies each CTA's shared memory to its own TMEM, at the same addresses; it
  also writes the peer's TMEM, so the peer must observe it before exiting, as for the pair MMA.
  An odd CTA issuing either is reported as unmodeled.
- Status (block-scaled pairs, 2026-11-04): **validated** by `probes/tc_pair_bs` (MMA) and
  `probes/tc_pair_bs/cp` (copy). Every operand and every smem word differs between the two
  CTAs, and both front-ends equal GB300 bitwise. In the copy, CTA r's TMEM lane 32w + i,
  column j holds its own word 4i + j (rank << 16 | 4i + j); no word comes from the peer.
  End to end: cuBLASLt's NVFP4 kernel `fixtures/c_smm_nvf4` (M = N = 128 as an M = 256 pair
  tile, B split 64 / 64) equals GB300 bitwise (docs/equiv-mma.md §5).
- **M = 128 pairs** (PTX ISA "Layout B", `cta_group::2`, dense): CTA i supplies A rows 0..63
  of its own descriptor (D rows 64i..) and B rows (N/2)i.. ; its D row r, column c is at TMEM
  lane r + 64 (c / (N/2)), column c % (N/2): lanes 0..63 hold the first half of N, lanes
  64..127 the second, N/2 columns wide (`exec.rs` `mma_d_layout`, `MmaD`). Shapes: N = 16..256
  in steps of 16 for M = 128 and 256 (PTX ISA table 48); block scaling and sparsity with
  M = 128 pairs are rejected (not measured). **A in TMEM** (2026-11-07, KA: FlashAttention
  Bf16x6 forward / backward): every D lane reads the A row in its own TMEM lane, so lanes
  64..127 (the second N half, B rows of CTA 1) use lanes 64..127's A rows, not lanes 0..63's
  (`complete_mma` `arow`). probes/tc_pair_atmem (N = 64, pattern operands naming A row, B row
  and CTA in each exact result; both CTAs' whole TMEM dumped): GB300 20/20, both front-ends
  bitwise.
- Status (M = 128 pairs, 2026-11-05): **validated** by `probes/tc_m128` (kind::f16 bf16, N = 16,
  32, 48, 64, 128, 256; kind::f8f6f4 e4m3, N = 16, 32, 64; M = 256 N = 32 / 64 as controls).
  Each CTA's whole TMEM is sentinel-filled before the MMA and dumped after it; with operands
  that encode (A row, B row, CTA) in the exact result, exactly the cells above change, with
  the predicted rows. Both front-ends equal GB300 bitwise on every config
  (`crates/sim/tests/tc_m128.rs`; fixture `e_mma_pair_m128`). End to end: the cuBLAS 2-CTA
  nvjet kernels with M = 64 per CTA (`..._64x32_64x16_2x2_2cta_h_bz_NNT` for 512 x 128 x 512
  and 512 x 256 x 512, `..._128x64_64x10_2x2_2cta_...` for 1024 x 128 x 1024) certify
  (E-WELLSYNC) and equal their captured GPU outputs bitwise; before, their epilogue's reads of
  lanes 64..127 were reported as reads of never-written TMEM.
- If wrong: sparse M = 128 pairs ("Layout C") and other forms are rejected as unmodeled.

### S-RESERVED · code
**Reserved shared memory (0x0..0x3ff) starts at 0, except in `cta_group::2` cubins**
(`EIATTR_TCGEN05_2CTA_USED`), where the driver sets words `0x0 = 0x1ffe00`, `0x4 = 0x7ffff800`,
`0x50 = 1` in even ranks (the leader's handshake phase bit), and the word at `0x48` is a
**count-1 mbarrier in phase 0** (its sync-unit state is not visible through loads, which read
0). ptxas's paired-allocation handshake runs on these: the leader's `UTCATOMSWS.2CTA` allocates
in both CTAs, then `arrive.expect_tx(4)` + `STAS` hand the column to the peer's slot through
the peer's 0x48 barrier, and the peer arrives back on the leader's.
- Evidence: `probes/reserved_smem` (dump at kernel entry; zero without the 2CTA attribute),
  `probes/mbar_reserved` (try_wait / arrive.expect_tx / complete_tx on 0x48 behave as count 1,
  phase 0), `probes/mbar_zero` (a plain-stored zero word is *not* such a barrier: the arrive
  faults). There is no prologue code in the SASS.
- Where: `machine.rs` `Machine::new` (SASS front, kernels with a paired alloc or MMA).
- Status: **validated** by `fixtures/r1c_2sm`: SASS model = PTX model = GB300 bitwise.
- If wrong: the handshake would deadlock or race in the model; other reserved words are
  unknown and read as written-by-launch values.

### S-SCOPE · code
**Strong memory operations follow the PTX memory model's scope rules; mbarrier scopes are
treated as sufficient.** Since rung 1f:
- `ld/st.relaxed|acquire|release.<scope>`, `.volatile` (= relaxed.sys) and `atom` / `red`
  (default relaxed.gpu) are strong accesses. Two strong accesses are **morally strong** when
  each one's scope includes the other's thread (`.cta`: same CTA, `.cluster`, `.gpu`: same
  GPU, `.sys`: all). Morally strong accesses do not race; any other conflicting pair not
  ordered by happens-before is a race.
- A strong write is the location's synchronization source. It releases the writer's clock
  (release write), or the clock at its last release fence (relaxed write after
  `fence.acq_rel`/`MEMBAR`, at the narrower of the two scopes). A read-modify-write continues
  the release sequence; a plain write ends it. A release store ticks its thread's clock after
  publishing it (2026-11-04), as fences and atomics already did: before, a plain write after
  `st.release` in the same clock window looked ordered before every reader that acquired
  the release, so a payload written after its flag was published was reported clean (PTX
  front-end; SASS `st.release` is `MEMBAR` + a relaxed store; fixture e_ring R1). Across the
  65 other PTX fixtures, no verdict changes; explored states grow by up to 2.7x
  (r1g_allreduce good: 119,899 to 326,045 states, clean either way).
- A morally strong read whose thread the release's scope includes acquires it at once
  (acquire read) or at its next acquire fence (relaxed read; `fence.acq_rel.<scope>` acquires
  observations whose writer its scope includes, and SASS `CCTL.IVALL` acquires all of them).
- `membar.cta` / `.gl` / `.sys` are `fence.sc.cta` / `.gpu` / `.sys` (PTX ISA; ptxas emits the
  same `MEMBAR.SC.<scope>` [+ `ERRBAR` + `CCTL.IVALL`]) and are modeled as `fence.sc` is (as
  `fence.acq_rel` at that scope: no total SC order beyond the interleaving, which is weaker
  than PTX, not stronger). `fence.acquire.cta` / `.cluster` / `.gpu` acquire only
  observations whose writer the scope includes (`.sys`: all); ptxas emits nothing for `.cta`
  and `CCTL.IVALL` for `.cluster` / `.gpu`, so the SASS front is exact for the code (an
  all-scope acquire after `.cluster` / `.gpu`, none after `.cta`: LDG.STRONG.SM is read as
  relaxed, so a CTA-scope acquire pattern is reported there, weaker than the hardware).
  probes/fence_scopes (2026-11-07): cross-CTA and in-CTA message passing; acquire narrowed to
  `.cta` / `.cluster` or release to `membar.cta` across CTAs is reported on the PTX front;
  GB300 50/50 on the good variant.
- `red.global.add.v2` / `.v4.f32` (SASS `REDG.E.ADD.F32x2` / `F32x4.FTZ.RN`) is one FADD.FTZ
  atomic per element, each on its own (the PTX ISA gives no atomicity across elements;
  2026-11-07, probes/red_vec = GB300 on both front-ends). `atom.global.add.v2` / `.v4.f32`
  (SASS `ATOMG.E.ADD.F32x2` / `F32x4.FTZ.RN`; Triton's `atom.global.gpu.acq_rel.add.v2.f32`)
  is the same per element and returns each element's old value; a destination overlapping a
  later source element is refused (2026-11-07, probes/atom_vec: specials and subnormals in
  addends and starting values, GB300 20/20, both front-ends bitwise).
- SASS: `ATOMG` / `REDG .E.<op>.STRONG.SM` (PTX `atom` / `red.global.cta`) are CTA-scope
  atomics (2026-11-07; probes/generic8 = GB300 on both front-ends, with generic `LD.E.U8` /
  `.S8` and `ST.E.U8`).
- SASS: `LDG/STG.E.STRONG.<SYS|GPU|SM>` are relaxed strong accesses; `MEMBAR.*.<scope>` is a
  release fence; `CCTL.IVALL` is an acquire fence (ptxas: `ld.acquire` = `LDG.STRONG` +
  `CCTL.IVALL`, `st.release` = `MEMBAR` + `STG.STRONG`, `fence.acq_rel` = `MEMBAR` +
  `CCTL.IVALL`).
- `fence.sc` is treated as `fence.acq_rel`. mbarrier `.release.cta/.cluster` and
  `try_wait.acquire.cluster` scopes are still treated as sufficient.
- Scope defaults never strengthen the model (2026-11-06 audit). A PTX `fence.sc` /
  `fence.acq_rel` without a scope, and a SASS `MEMBAR` without one, are refused (before: `.sys`,
  the strongest). A SASS atomic or `.STRONG` access without a scope token takes `.cta`, the
  weakest (before: `.gpu`); ptxas prints a token on every global and generic form in the
  corpus, and the only token-less forms are `ATOMS.*` (shared memory, CTA scope anyway) and
  `ATOM.E.CAST.SPIN` (the shared-memory leg of a generic atomic). Weaker readings that remain:
  `MEMBAR.SC` and `fence.sc` act as `MEMBAR.ALL` / `fence.acq_rel` (no SC total order), and
  `atom` / `red` without a scope are `.gpu`, the PTX ISA default.
- Where: `exec.rs` `morally_strong`, `scope_includes`, `sync_publish`, `sync_observe`,
  `acquire_pending`; `ptx` `mem_sync`; `sass` `sass_strong`, `sass_scope`.
- Status: **validated** for `.gpu` vs `.sys` across two GPUs (`fixtures/r1f_pingpong`): B20
  (relaxed poll, no acquire) is a race in both models and **wrong 1000/1000 on GB300** (a stale
  L1 hit on the payload); B19 (release at `.gpu`) is a race in both models and latent on GB300
  (20,000 runs). B56 (`fence.acq_rel.gpu` as the acquire) is a race in PTX and clean in SASS:
  only the release half of the fence is scoped.
- If wrong: *miss* of mbarrier scope bugs (a remote arrive with `.release.cta`).
- Retire: scopes on mbarrier operations.

### S-RED-ASYNC · code
**`red.async.release.{gpu,sys}.global.add` is a red of that semantics and scope
that has completed before the issuing thread's next instruction. SASS
`UREDGR.E.ADD.NA.STRONG.GPU|SYS` releases the issuing thread's earlier accesses at its
scope.**
- ptxas 13.0's lowering is a per-lane ELECT loop: `MEMBAR.ALL.CTA; UREDGR [UR], UR; DEPBAR.LE
  SB, 0`. DEPBAR waits for the reduction, hence "completed". On its own, MEMBAR.ALL.CTA does
  not order global stores before a gpu-scope flag (control below), so the release rests on
  UREDGR.
- Where: `ptx` atom / red arm (`async`), `sass` `UREDGR` (executed once per warp, by its first
  active thread, which releases only that thread's clock; ptxas's loop runs one elected lane
  per iteration).
- Refused: other operations (`.min` / `.max` / `.or` / `.and` / `.xor` / `.inc` / `.dec`).
  ptxas 13.0 and 13.1 compile them to `UREDGR.E.ADD`, and GB300 adds: `100 min.s32 6` = 106.
  Also refused: other types, other spaces, `atom.async`, and UREDGR with other modifiers.
- Refused as well (2026-11-09): `.relaxed` on `.global`. The PTX ISA (9.5, red.async) defines
  the `.global` form as `red.async{.mmio}.release.{gpu,sys}{.global}.add.{u32,s32,u64,s64}`
  only; `.relaxed` needs `.cluster` scope, `.shared::cluster` and an mbarrier, and `.mmio`
  needs `.sys` (not modeled). ptxas 13.1 / 13.2 accept `.relaxed` and the other operations on
  `.global` without a diagnostic and emit the release's `MEMBAR.ALL.CTA` + `UREDGR.E.ADD`, so
  the SASS front runs them as the release add; probes/red_async_ops (GB200, sm_100a, ptxas
  13.1): `.min.s32` 10 with 3 gives 13, `.inc.u32` 7 with 5 gives 12, `.or.b32` 0x11 with 0x10
  gives 0x21, i.e. every op added. A PTX kernel with one of these forms is refused, so a
  wrong-result bug behind them is reported as a lift gap, not missed.
- Status: **probed** (hardware evidence, probes/redasync_litmus, 2026-11-08). The test is message
  passing between CTAs on different SMs: weak payload stores, the form on a flag,
  `ld.acquire.gpu` polls, then weak reads. In the three configurations in which controls were
  stale, `red.async.release.gpu` gave 0 stale reads in 1,689,600,000. MEMBAR.ALL.CTA + REDG
  gave 21,730,362 stale reads; `red.relaxed.gpu` gave 461,790 in 153,600,000. Totals:
  2.72e9 reads under the async form, 0 stale. probes/red_async: GB300 50/50 scrubbed, both
  front-ends bitwise; its `relaxed` variant is refused on the PTX front (not an ISA form) and
  clean on the SASS front (ptxas emits the release).

### S-SMEM-STRONG · code
**SASS `LDS`/`STS` of the CTA's own shared memory are relaxed strong accesses at CTA scope;
an `LDS` whose scoreboard is waited on before the warp issues any later non-ALU
instruction is `ld.acquire.cta`.** This is the hardware-faithful model of the binary. The PTX
front-end keeps the PTX model, where a weak shared access can race.
- **Root pairs** (2026-11-06): under this model an unsynchronized `STS` / `LDS` pair of two
  warps is not a race; its consequence (a nondeterministic output, a global race
  downstream) is what the verdict reports. When `check --certify` falls back to exploration
  and the verdict has a race or nondeterministic outputs, the certificate run's unordered
  shared-memory pairs (C-ORDER) are listed first as findings of kind `unordered` (sig
  `unordered-smem:<pc>:<pc>`, not a verdict class), so the place where a barrier is missing is
  named (mslk moe index_shuffling: `STS` 0x08f0 / `LDS` 0x0940 behind a global write-write
  race at `STG` 0x1280; GB300 delay replay wrong 1000/1000). They come from one run and are
  candidates: the verdict does not show that each one reaches the consequence.
- **Evidence:**
  - ptxas emits bit-identical `LDS`/`STS`/`ATOMS` for weak, `.volatile` and `.relaxed`
    shared accesses at every scope, and flipping encoding bits shows no strength or scope
    field (`docs/smem-order-mapping.md`). Strength only steers ptxas's optimizer (weak spins
    are deleted, weak loads merged).
  - `ld.acquire.cta.shared` is `LDS` plus a wait on its scoreboard; `st.release.cta` is
    `MEMBAR.ALL.CTA` + `STS`.
  - About 3.1e12 one-CTA litmus instances on GB300 (MP, SB, LB, CoRR, CoWR, IRIW, tearing
    at 32/64/128 bits, the merged `STS.64 {flag,data}`) showed no non-SC outcome, while the
    allowed outcomes appeared billions of times and a global-memory control did show stale
    reads (`docs/smem-litmus.md`, `probes/smem_litmus`).
  - An aligned `.64`/`.128` access is one access.
- **Cost:** strong accesses are scheduling points. A verdict is therefore computed in two
  passes (`smem.rs`):
  - first with `LDS`/`STS` plain. Without a race that is also the strong model's verdict:
    strong accesses only add happens-before, and race-free plain accesses commute (E-DRF);
  - with a race, the strong kernel decides.

  So a missing `bar.sync` between warps shows up on SASS as `nondeterministic` output, not
  as `race` (or not at all if the values agree). The same holds for an `LDS`/`STS` against an
  `ATOMS` (shared atomics are strong; fixtures/e_lds_atoms). Strong reads are recorded as reads for POR
  (they commute).
- Where: `sass` `SMEM_RELAXED`, `smem_acquires`; `sim` `smem.rs`; `exec.rs` `access_x`.
- Status: **validated** (litmus, above); zero observations is not a proof.
- If wrong: *miss* of reorderings of shared accesses within a CTA that the hardware does
  perform.

### S-CLUSTER-LIFE · code
**A CTA's shared memory exists only while one of its threads runs.** An access to another
CTA's shared memory (remote mbarrier arrive, multicast TMA delivery, `tcgen05.commit` to a
peer, `shared::cluster` load/store) after the whole target exited is a violation, and the
access is dropped. Independently of the schedule explored, every remote arrive / TMA issue /
commit into CTA X must happen-before the exit of *some* thread of X (checked at terminal
states, like async-after-exit), since otherwise a schedule exists in which X exits first.
- Where: `exec.rs` `check_alive`, `dead_target`, `note_remote`, `exit_checks`.
- Status: **argued** (PTX ISA: a CTA's shared memory must not be accessed after it exits;
  CUTLASS ends cluster kernels with a cluster barrier for this reason). A multicast delivery
  is checked by its issue, not its landing: if the target syncs with the issuer after the
  issue but never observes the landing, a late landing is missed. Remote `ld/st.shared::cluster`
  get only the immediate check, not the terminal one.
- If wrong: *false alarm* if hardware keeps the memory alive until the whole cluster exits.
- GB300 (`probes/remote_arrive_exit` rae.cu / window.cu, 2026-11-07): it does not. A remote
  mbarrier arrive or `st.shared::cluster` that reaches an exited peer fails the launch
  (`cudaErrorLaunchFailure`). Successor CTAs already resident on the exited CTA's SM were
  never written. A violation of this rule is therefore a crash on GB300, not a silent
  corruption.

### S-REMOTE-INFLIGHT · code
**A remote mbarrier arrival (`mbarrier.arrive[.expect_tx] .shared::cluster` to another CTA,
SASS `SYNCS.ARRIVE.TRANS64.RED`) and an `st.async` into another CTA (with its complete_tx) are
in flight from issue until delivered.** Deliveries are separate steps the explorer schedules
at any time, in any order among the issuer's remote operations (also to different
mbarriers), except that a release of the issuer at cluster scope or wider delivers all of
them first: `barrier.cluster.arrive` (not `.relaxed`), `fence.acq_rel` / `fence.sc` at cluster
/ gpu / sys scope, `mbarrier.arrive.release.cluster`, SASS `MEMBAR.ALL.GPU` / `.SYS`.
`.release.cta` and `.relaxed` arrivals deliver nothing. The issuer's exit delivers nothing
either. Arrivals and st.async to the issuer's own CTA land at issue, as before.
- Violations (S-CLUSTER-LIFE): the target CTA's last thread exits while one is in flight to
  it; a delivery after the target exited; at the end, a delivery the target never waited for
  (no successful wait on the phase it counted toward or a later one) unless a later cluster
  release of the issuer (which would have drained it) happens-before a target thread's exit.
  The last one decides it order-independently, so the certificate (E-WELLSYNC) refuses such
  a run.
- Why: a FusedRMSNormBwd bug: the pre-fix epilogue waited only on the last stage's
  empty barrier; the peer's `.release.cta` arrive on the other stage's barrier, issued earlier,
  could still land after the exit. With arrivals landing at issue, program order made it
  look ordered (the reporters' case was clean on both fronts). Now: fixtures/e_rmsnorm_cluster_drain
  `bug`/`bug_k3` are violations, `good`/`good_k3` clean, on both fronts.
- SASS evidence for the drain: ptxas emits `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` before a
  `.release.cluster` arrive and before `barrier.cluster.arrive`, and nothing before a
  `.release.cta` arrive (sm_103a, ptxas 13.0); `CGAERRBAR` presumably waits for the CTA's
  outstanding cluster (CGA) operations.
- One pending operation per warp instruction and target mbarrier: the lanes of one
  instruction that arrive on (or st.async into) the same mbarrier land together, in lane
  order (`push_remote`). Deliveries read the target's exit key (space 4, index 1), which the
  exit of its last thread writes, so deliveries to one CTA commute with each other; the
  TMA-completion reduction (`sync_ample`) takes them as candidates too.
- State space (stateful explorer): the reporters' 2-CTA case 6k -> 20k (PTX), 0.2k -> 0.6k
  (SASS); r1c_cluster PTX 3.7k -> 11k, SASS 1.1k -> 2.3k; r1c_2sm SASS 99k -> 358k (ptxas's
  peer `SYNCS.ARRIVE.TRANS64.RED` for cta_group::2 TMA); c_smm_nvf4 SASS 991 -> 832.
- Where: `exec.rs` `Op::MbarArrive`, `Op::StAsync`, `complete_remote`, `drain_remote`,
  `Op::Exit`, `exit_checks`; `machine.rs` `AsyncKind::Remote`, `remote_obs`, `mbar_seen`.
  `PTXV_REMOTE_ATOMIC=1` lands them at issue (the old model).
- Status: **argued.** Not observed on hardware: a late arrival into an exited CTA leaves no
  visible trace on GB300 or GB200 (the reporters saw nothing even with no drain). The FIFO-free order
  among one issuer's remote arrivals is the conservative reading of the PTX memory model
  (`.release.cta` does not order them for another CTA).
- Order (2026-11-07): one warp's remote operations on one mbarrier land in issue order
  (`AsyncOp::fifo` = `remote_pipe(warp, cta, mbar)`). PTX: two arrivals by one thread on one
  mbarrier are program-ordered morally strong read-modify-writes of one location, so
  coherence order follows program order (PTX ISA memory model, Coherence axiom over base
  causality order). Operations on different mbarriers stay unordered (`.release.cta` orders
  nothing for another CTA). Ordered per warp, not per thread: two instructions of one warp
  with disjoint lane sets on one mbarrier are also kept in order, which the PTX model does
  not require (a possible *miss*; remote arrivals are normally issued by one elected lane).
  Remote operations in such an order are not TMA-completion reduction candidates.
- If wrong: *false alarm* if hardware delivers one thread's remote arrivals in order across
  mbarriers, or keeps a CTA's shared memory until its inbound arrivals land; *miss* if a
  cluster-scope release does not wait for earlier remote arrivals.
- GB300 (`probes/remote_arrive_exit`, 2026-11-07): remote *stores* issued before a barrier
  and another warp's arrive can still be in flight when the target exits (the launch fails,
  every process). Remote *arrivals* were never seen out of happens-before issue order (0 of
  401,200 clusters, `order.cu`), which is the opt-in E-REMOTE-FIFO.
- Retire: an NVIDIA statement on remote-operation ordering.

### S-REMOTE-RESERVED · code
**A remote arrival on an mbarrier in the reserved shared-memory prefix (offset < 0x400) is
ptxas's or the driver's protocol, not the kernel's.** ptxas implements
`tcgen05.alloc.cta_group::2` with a handshake on the reserved mbarrier at 0x48: the peer
waits on its own 0x48 and arrives on the leader's (`SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ,
[R13+URZ], RZ` after `ATOMS.OR` into the allocator words). The last such arrival is never
waited for, so S-REMOTE-INFLIGHT's lifetime checks (`remote-unobserved`,
`remote-inflight-exit`, a delivery after the target exited) fire on every cta_group::2
kernel. For reserved-prefix mbarriers they are `contract` findings instead of violations.
- Where: `exec.rs` `remote_reserved` (exit_checks, `Op::Exit`, `complete_remote`).
- Status: **argued** (2026-11-07). Evidence: msl_ops FlashAttentionForwardSm100 2-CTA
  (cases-MN `cute.FlashAttentionForwardSm100.bf16.1x384x1x128`): the third of MN's three
  S-REMOTE-INFLIGHT violations, sass 0x2330, is this handshake; GB300 runs of cta_group::2
  kernels show no effect. Whether a late handshake arrival can disturb the next CTA on the SM
  (which reuses the reserved prefix) is not probed.
- If wrong: *miss* of a ptxas / driver protocol bug.
- Retire: a GB300 probe that observes a late remote arrival (e.g. into a CTA that a
  co-resident successor reuses).

### S-NIC-DB · code
**The doorbell of an IBGDA queue pair is a strong system-scope store (`st.mmio.relaxed.sys`,
SASS `STG.E.64.MMIO.SYS`, or `st.relaxed.sys`) to the QP's doorbell register; the WQE and
payload stores the NIC reads for the WQEs it announces must happen-before it through a release
of GPU scope or wider** (`fence.release.gpu` / `.sys`, `fence.acq_rel` / `fence.sc` at those
scopes, SASS `MEMBAR.ALL.GPU` / `.SYS`, or a release store).
- Model (nic.rs `nic_store`): a store to a declared doorbell announces WQEs up to the 16-bit
  producer index it carries; their reads use the ringing thread's release clock (its last
  release fence, as `sync_publish`), or an empty clock when that fence is narrower than GPU
  scope or absent. A WQE or payload byte whose last write is not ordered before it is a race
  (`nic-wqe-unordered`, `nic-src-unordered`); the WQE then executes with the bytes as they are.
  A weak store to a doorbell, or a store to the doorbell page that is not an 8-byte ring
  (BlueFlame), is `unmodeled`. The doorbell record (DBR) store is plain memory; the NIC acts
  on doorbell rings only.
- Source: DOCA GPUNetIO `doca_gpu_dev_verbs_submit_db` / `ring_db` / `mark_wqes_ready`
  (fence.release.gpu, then `st.mmio.relaxed.sys`), mlx5 PRM doorbell format (ctrl segment
  dword 0: WQE index << 8; dword 1: QPN << 8). Fixture r5a_ibgda_putsig N1 (no fence), N2
  (`.cta` fence), N5 (a WQE segment written after the doorbell).
- Status: **argued.** Hardware (probes/ibgda P3, 20 configurations x 1e5 iterations): with no
  fence, or a `.cta` fence, between the WQE / payload stores and the doorbell, no stale WQE,
  stale payload, error CQE or hang was observed. The requirement follows DOCA and the PTX
  model (the NIC is an observer outside the GPU); on this hardware a fence-less doorbell
  verdict may be a false alarm.
- If wrong: *miss* if a GPU-scope release does not order stores in GPU memory before the
  NIC's DMA reads; *false alarm* if a narrower fence also suffices.
- Retire: probes/ibgda P3 with and without each fence.

### S-NIC-ORDER · code
**The NIC executes a queue pair's WQEs one at a time, in posting order; nothing orders WQEs
of different queue pairs. Only IBTA ordering is modeled.**
- Model: one pending `AsyncKind::Nic` op per QP, completing the oldest announced WQE; a QP
  declared with another `ordering` (out-of-order data placement) gives `unmodeled`.
- Source: IBTA RC transport (requests of one QP are processed in order); DOCA
  `0001-gpunetio-dp-ordering-ooo.patch` (dp_ordering off by default).
- Status: **validated** for the cross-QP half: probes/ibgda P2 observed stale data after a
  signal on another QP in 1e5 / 1e5 iterations (1 MiB writes; not observed at 8 B / 4 KiB).
  In-order execution within a QP: argued (P1, P4 consistent).
- If wrong: *miss* if WQEs of one QP can take effect out of order under IBTA ordering.
- Retire: probes/ibgda P2 / P4.

### S-NIC-SIGNAL · code
**An RDMA atomic (fetch-add / compare-swap) takes effect after the same queue pair's earlier
RDMA writes are placed, and an observer that acquires the atomic's value sees them**
(put-with-signal without a WQE fence bit).
- Model: each WQE's clock includes the QP's previous WQE's clock; the atomic is a release.
- Source: DOCA `doca_gpu_dev_verbs_put_signal_thread` and mscclpp `putWithSignal` post the
  write and the atomic with no fence bit; NVSHMEM `put_signal` semantics. Fixture
  r5a_ibgda_putsig N3 (the signal posted first).
- Status: **validated** (not refuted): probes/ibgda P1, put of 8 B / 4 KiB / 1 MiB then a
  fetch-add signal on the same QP, stale data not observed in 1e5 iterations per size and
  receiver variant (15 configurations).
- If wrong: *miss*.
- Retire: probes/ibgda P1.

### S-NIC-WRITE · code
**RDMA write data synchronizes with nothing directly and has no placement order**, within or
across writes (aligned words are single-copy atomic, S-NIC-WORD; a lone aligned 4- or 8-byte
write is a release, S-NIC-FLAG). A reader that polls a word of write data and then reads
other words of it races with them.
- Status: **argued** (conservative: no ordering or single-copy atomicity is assumed).
  probes/ibgda P4: a flag write after a large write on one QP was never seen before the data
  in 1e5 iterations (4 KiB, 1 MiB), so polling a write's last word may be a false alarm here.
- If wrong: *false alarm* for kernels that poll the last word of a write (not observed in the
  target kernels).
- Retire: probes/ibgda P4 would only support a weaker model, not a stronger one.

### S-NIC-WORD · code
**Each naturally aligned 8-byte word of RDMA write data (4-byte at the edges of a write) lands
single-copy atomic: a relaxed system-scope write of the NIC.** Unaligned edge bytes are plain
writes. No order between words and no release (S-NIC-WRITE). A strong load of such a word
does not race with it; DeepEP's hybrid notify relies on this (`recv_and_reduce`: each count
of a multi-int put is polled with `ld.acquire.sys` and decoded on its own).
- Source: PCIe writes do not tear naturally aligned qwords; nothing in IBTA.
- Where: `nic.rs` `complete_nic` (`OP_RDMA_WRITE`).
- Status: **validated** by probes/ibgda P2 (1 MiB) and P6a (4 KiB, 1 MiB): the receiver read
  8-byte words with `ld.relaxed.sys` while a write was landing; 6.03e9 (P2) and 1.3e10 (P6a)
  words held the old or the new tag, none a mix of both (`other` = 0).
- If wrong: *miss* of a torn read of an 8-byte word.

### S-NIC-FLAG · code
**A WQE whose only data segment is a naturally aligned 4- or 8-byte RDMA write lands as one
strong system-scope write that releases the QP's earlier WQEs** (the same release a fetch-add
signal gets, S-NIC-SIGNAL). mscclpp's `PortChannelIbgda` signals this way: a 4-byte write of
the outbound semaphore token into the peer's inbound token, polled with `ld.acquire.sys` /
`LD.E.STRONG.SYS`.
- Source: IBTA RC executes a QP's writes in order; a PCIe write of an aligned dword / qword is
  not torn. Neither says the flag is placed after the earlier write's data (PCIe relaxed
  ordering may reorder them), so this is the probe's claim.
- Where: `nic.rs` `complete_nic` (`OP_RDMA_WRITE`).
- Status: **validated** by probes/ibgda P4 (8-byte flag) and P4b (4-byte flag): a 4 KiB or
  1 MiB write then the flag on the same QP, one or two doorbells, receiver `ld.acquire.sys` on
  the flag then the data: stale data not observed, 1e5 iterations each (8 configurations).
- If wrong: *miss* of a stale read after a flag (the probes saw none).

### S-NIC-ATOMIC · code
**A NIC atomic is a strong system-scope read-modify-write of an aligned 8-byte word whose
value is little-endian in memory; NIC atomics are atomic with each other only.** A GPU atomic on
a word a NIC atomic also updates, unordered with it, is `unmodeled` (`nic-atomic-mixed`).
- Source: IB atomicity guarantee (IBV_ATOMIC_HCA: between QPs of one HCA).
- Status: **validated**: probes/ibgda P0 (fetch-adds of 1 leave 1, 2, 3, 4 as little-endian
  words; the fetched values come back little-endian); P7 (1e5 NIC fetch-adds and 16.8M GPU
  `red.add.sys` on one word: every NIC add landed, 4.9M GPU adds were lost).
- If wrong: values (not ordering) differ for kernels that compare the word's value.
- Retire: probes/ibgda P1 / P7.

### S-NIC-CQE · code
**With CQ update requested, the NIC writes the WQE's CQE (owner bit = pass parity over the
ring, opcode 0, WQE counter) after the WQE completed; for an RDMA write, completion means the
data was placed at the peer. The CQE write is a strong system-scope release of everything the
WQE and the QP's earlier WQEs did.** A poller that acquires the CQE may reuse the send buffer
and is ordered after the remote placement.
- Source: IBTA RC (a write completes once acknowledged); DOCA `poll_cq_at` (owner bit, then
  `fence.acquire.sys`). Fixture r5a_ibgda_putsig N4 (send buffer reused without polling).
- Status: **validated** (not refuted): probes/ibgda P5 (poll QP A's CQE, then signal on QP B:
  stale not observed in 1e5 per size, 8 B / 4 KiB / 1 MiB) and P6 (overwriting the source
  after the CQE poll: never received, 1e5; without the poll: the overwritten bytes arrived in
  1e5 / 1e5 iterations).
- If wrong: *miss* if a completion can precede remote placement or the NIC's last read of the
  source.
- Retire: probes/ibgda P5 / P6.

### S-NIC-DOMAIN · code
**A GPU loads, stores and atomics only memory of its own NVLink domain; another domain's
memory (another node without multi-node NVLink) is reachable only through the NIC.** A direct
access is a fault (case `domains`).
- Where: `exec.rs` `resolve` -> nic.rs `nic_domain_check`. Fixture r5a_ibgda_putsig N8.
- Status: **argued** (a peer mapping across nodes does not exist without MNNVL).
- If wrong: *false alarm* on MNNVL systems if `domains` is set wrongly.
- Retire: n/a (configuration).

### S-ALIGN · code
**A load, store or atomic of `n` bytes (`n` a power of two; a vector counts its whole size,
`ld.v4.u32` = 16, SASS `LDS.128` = 16) must be `n`-aligned; a misaligned one is a `fault`.**
Global, shared, shared::cluster and generic addresses; parameter space is not checked.
- Where: `exec.rs` `check_aligned` (`Op::Ld`, `Op::St`, `Op::Atom`).
- Status: **validated.** GB300 traps with "misaligned address" on `st.global.u64` at +4,
  `ld.shared.v4.u32` at +8, `atom.global.add.u64` (ptxas: `REDG.E.ADD.64`) at +4 and
  `st.global.u32` at +2 (fixtures/e_misaligned, `ptxv probe` 2026-11-05). Found by AK.
- If wrong: *false alarm* only if some form tolerated a misaligned address.

### S-GLOBAL · code
**Global memory is touched only by this kernel:** buffers come from the case, and nothing else
reads or writes them concurrently. Addresses are synthetic (GLOBAL_BASE, PARAM_BASE).
- Where: `lib.rs` `build_launch`, `machine.rs` constants.
- Status: **argued** for single-kernel cases. Cross-kernel and cross-GPU sharing is rung
  1e/1f/4 work.
- Multi-process captures (rung 4a, `tools/capture_dist.py`): a rank's initial bytes include
  device copies it made before its first kernel (two-shot's `buf.copy_(x)`), and initial
  contents happen-before every kernel of every GPU. So whether a peer's start barrier orders
  that copy (two-shot's put is relaxed) is outside the verdict.

### S-GLOBAL-INIT · code
**Module data globals (`__device__` variables) start from the case's buffer, else from the
ELF.** A kernel reaches a global through a relocated constant-bank word (`LDC.64 R4,
c[0x4][0x0]`, `R_CUDA_64` against a `.nv.global` / `.nv.global.init` symbol). The launch backs
each (module, symbol, GPU) with one buffer: the case buffer declaring `symbol` (and `cubin`,
`device`), whose `init` is the captured contents, or else an automatic buffer holding the
`.nv.global.init` bytes (zeros for `.nv.global`). Launches of one module on one GPU share it, so
its state carries from launch to launch within the case. On hardware the buffer is the module's
own global (`cuModuleGetGlobal`), rewritten with the same bytes before every run. An
automatic buffer with neither captured bytes nor an initializer is listed in one note per
launch ("N module global(s) without captured bytes or an initializer, zero at module load
assumed: X, ..."; 2026-11-07), so the assumption shows per record. The PTX front reads the
module's `.global` declarations (`.visible` / `.extern` / `.weak`, scalars and arrays, integer
initializers as element bytes) and resolves a symbol operand (`mov.b64 %rd, sym`,
`[sym+off]`) to the same buffer, through a synthetic relocated word in bank 0x1f; a symbol with
any other initializer (`generic(...)`, floats) is refused where used (2026-11-08).
- Where: `lib.rs` `build_launch` (`global_buffers`), `exec.rs` `Op::LdConst`, `hw` `run_with`,
  `sass/src/elf.rs` `Elf::data_symbol`, `ptx/src/lib.rs` `parse_kernel` (module `.global`
  lines), `exec.rs` `SReg::GlobalAddr`.
- Status: **argued.** A captured case declares the globals its kernels' banks name, with their
  device contents at the snapshot (`tools/capture_dist.py`, `ptxv.capture`). An automatic buffer
  is right for a fresh module and wrong for a long-running process whose earlier kernels
  changed the global (DeepEP's `epSyncer` counts barrier rounds). fixtures/s_globals is
  bitwise equal to GB300; probes/ptx_globals (initialized, zero and scratch globals, PTX front)
  is bitwise equal to GB300 on both fronts (20/20, 2026-11-08).
- Not modeled: a global reached through a relocated instruction immediate (`32@lo(sym)`,
  relocatable code): address 0, as before. `__constant__` globals.
- If wrong: *miss* or *false alarm*: the kernels run from other initial state than the
  process had.
- Retire: capture every referenced global (done for capture_dist captures).

### S-CBANK · code
**Driver constants:** `c[0x0][0x0..0x14]` = block/grid dims, `0x348` = generic param base,
`0x110` = generic address of constant bank 0 itself (cuBLAS reads a tensor map at
`c[0x0][0x110] + 0x6c0`, i.e. cbank 0x6c0 = param offset 0x340; consistent with 0x348),
`0x358` (global descriptor for `desc[URx]`) ignored, `0x36c` = `%is_explicit_cluster` (1 iff
the launch names a cluster shape; `probes/cluster_flag` on GB300; ptxas skips
`barrier.cluster` unless it is 1), `0x37c` = stack pointer 0xfffc (no local memory), `0x2f8` = generic base of the local window
(an unmapped 0x7d00_0000_0000: DeepEP only builds vprintf argument pointers from it), params at
`constant0 size − param size`. Any other driver constant is an error.
- PTX front: `cvta.local.u64` adds the same base and `cvta.to.local.u64` subtracts it
  (2026-11-08); a local access whose address is not below 2^32 (a converted generic address
  outside the local window, undefined in PTX) is a fault. probes/local_cvta: GB300 50/50
  scrubbed, both fronts bitwise.
- Where: `exec.rs` `Machine::cbank`, `Machine::local_addr`; `ptx` the `cvta` arms.
- Status: **validated** by bitwise agreement. Only param offset 0 has been exercised.

### S-CGA-SHAPE · code
**`SR_CgaSize` is the index of the CTA's cluster shape: 0 for the launch's regular cluster shape,
1 for its preferred one** (`cudaLaunchAttributePreferredClusterDimension`). ptxas reads the
cluster special registers from the driver table at `c[0x0][x - 0xa0 * SR_CgaSize]`, one table
per shape (S-CBANK). A launch with a preferred dimension runs each group of CTAs of the
preferred shape either as one preferred cluster or as regular clusters; one launch mixes both,
and every CTA's registers describe its own cluster.
- Evidence: `probes/cga_size` (2026-11-06, GPU 1): `SR_CgaSize` read raw (patched cubin) for
  clusters of 1, 2, 4, 8, 16 without a preferred dimension: 0 in every CTA. With a preferred
  dimension (2 -> 4, 1 -> 2, 1 -> 4, 2 -> 8, 4 -> 8, 4 -> 16; 512 CTAs): 1 in preferred clusters,
  0 in regular ones; with 200 KB of dynamic shared memory 0-22% of the CTAs ran in regular
  clusters, without it none. Every group of the preferred size was wholly one shape.
  `%cluster_nctarank`, `%cluster_ctarank`, `%cluster_nctaid`, `%clusterid`, `%nclusterid` and
  `%cluster_ctaid` describe the CTA's own cluster in both. A preferred dimension without a
  cluster dimension fails to launch. `crates/sim/tests/cga_size.rs` replays all 14 recorded
  configurations on the SASS front-end (each CTA from the shape it ran in). Before, ptxv set
  `SR_CgaSize` to the CTA count (consistent with the table only through the matching offset);
  cuBLAS tests `SR_CgaSize != 0` to choose its preferred-cluster path (mm_4k_8k:
  `128x256_64x6_2x2f_2cta`, captured with preferred [4, 1, 1] and cluster [2, 1, 1]), which then
  multicast to rank 2 of a 2-CTA cluster. With the model, mm_4k_8k with its cluster launch
  control parameter (`c[0x0][0xd10]`) set to 0 certifies in all 384 slices of both shapes, and
  the assembled output equals the GPU's capture bitwise (2026-11-06). The captured launch
  itself (parameter 2) executes `UGETNEXTWORKID.BROADCAST`, which is unmodeled (E-CLC).
- Capture: `probes/launchcap` and the Python API's `ptxvcap.c` record the attribute
  (`preferred_cluster`), and `tools/capture.py` / `_casewriter.py` write it to case.toml; `ptxv
  hw` passes it to the launch.
- Model: `Grid::preferred`, `Grid::cga_shape`, `Launch::preferred_shape` (every cluster
  preferred). The preferred shape must be a multiple of the regular one and divide the grid.
  `ptxv check --slices --certify` slices both shapes; the cross-cluster check pairs clusters of
  the two shapes whose CTAs are disjoint (those can coexist), each preferred cluster must write
  the same global bytes as the regular clusters of its CTAs, and the two shapes' assembled
  outputs must be equal, so that every mix ends with the same memory. Other checks of such a
  launch report `unmodeled` (only the regular shape was checked).
- Where: `exec.rs` `sreg` (`SReg::CgaShape`), `cbank`; `wellsync.rs` `check_sliced_one`;
  `cli` `cmd_check`.
- Status: **validated** for 1-D clusters (above). Clusters with y or z > 1 were not probed.
- If wrong: a *miss* or a *false alarm* in kernels that branch on `SR_CgaSize`, if the
  hardware's choice of shape per group is not free of other constraints the kernel can observe.

### S-SAME-VALUE · code
**A plain write of the byte value the last (plain) write left there conflicts with nothing.**
Every access unordered with such a pair sees that value in either order, so the pair is not
reported as a race: it is a note (kind `note`, printed `NOTE:`, no verdict class, does not stop
the certificate). The PTX memory model still calls it a data race. Triton layout conversions
store a replicated value from several warps to one shared address (`STS [R10+UR4], R17` from
t0 and t64), which made the certificate refuse and the explorer run out of states.
- Rule (`exec.rs` `access_x`, per byte): the write is plain, the last write is plain, and the
  written byte equals the byte in memory. The unordered earlier writes stay in the shadow
  (`Sh::hw`) and the unordered reads stay in its readers, so later accesses are still checked
  against them. The same-value write is marked (`Epoch::sv`): later reads are not checked
  against it (they see the value in either order) but against the write before it, which
  stays in `Sh::hw` while ordered. A plain last write with uncovered earlier writes therefore
  implies they wrote the same value; launch groups' `demote` (strong writes made plain) moves the last write into
  them, so that the rule cannot apply there. Strong writes keep their own rules (C-ORDER).
- Own store (2026-11-07, msl DeepEP `DeviceSyncer`: every CTA stores the next
  `currentCountIdx`, then reads it back): a plain read of a byte whose shadow holds the
  reader's own plain store is not checked against the shadow's writes; a note
  (`same-value:...:own`). By the rule above every write in the shadow stored the current value
  (a store of another value drops the others), so the own store did, and coherence lets the
  read return only the own store or a later one. Writes made later are checked against the
  read when they come: another value races with it, the same value takes the same-value path.
  A store of another value that happens before the read but is unordered with the own store
  is a write-write race with it, reported at whichever store comes second. Only plain stores
  of threads on the reader's GPU qualify: an async-proxy, NVLS-view, NIC, peer-GPU or DSMEM
  store in the shadow disqualifies the byte, as do strong writes demoted at a launch-group
  boundary (`Sh::mixed`). Only thread reads (not an async proxy's read). Per byte, so the own
  store must cover every byte read; other threads (same warp or SM) do not count. The read is
  still recorded in the shadow's readers, so later stores and landing async writes are checked
  against it. The shadow is keyed by allocation (`Loc::Global { alloc, off }`): two mappings
  of one physical page that a case lists as separate allocations are outside the claim (as
  for the rest of the shadow); the NVLS view is the one alias modeled (S-MC-ALIAS). Fixture
  `s_sv_own` (good clean; `diff`, `noown`, `xhb` race).
- Not covered: read-modify-write sequences that happen to agree (FA4's idempotent row-max
  revert): those are reported as races.
- Certificate (E-WELLSYNC): one run shows equal values only for that schedule. The certificate
  accepts a same-value note only when the run has no strong accesses and no polls, so every
  value written is fixed by the launch inputs. Otherwise it refuses and the explorer decides.
  Without this restriction `fixtures/e_ring_warp` W1 (a warp ring whose writes carry values
  read through polls) certified clean while the explorer finds the race.
- Evidence: `probes/same_value` (2026-11-06): lane 0 of every warp stores the same word to one
  shared and one global address without synchronization, 1000 / 1000 runs with 2 CTAs x 4
  warps and 200 / 200 with 16 CTAs x 8 warps read back v. `probe_same_value_matches_hardware`
  replays it; its `differ` variant (warp w stores v + w) stays a race. Triton's
  `_kernel_fp32_to_lowp_sr` now certifies in 0.25 s with GB300's output (was: incomplete at
  2M states after 173 s).
- Status: **validated** (above; byte stores of equal value cannot produce another value).
- If wrong: a *miss* only if hardware could produce a value other than the one both writes
  store.

### S-SMEM-ADDR32 · code
**Shared-memory addresses are 32-bit: the parts of an address operand (`[R+UR+imm]`) add
modulo 2^32.** ptxas folds a uniform base and a per-thread offset into one operand; a negative
offset is a large unsigned register value. cuBLAS's nvjet mainloops index their mbarrier ring
as `[R8+UR30]` with R8 = -0x28 when the stage index wraps (mm_1k_4k, mm_4k_8k).
- Evidence: `probes/smem_addr_wrap` (2026-11-06): `LDS R7, [R4+UR4+0x28]` with R4 = 0xffffffd8,
  clusters of 2, 20 runs: every thread reads its own word. A 64-bit sum names CTA rank 256,
  which the model reported as an illegal-instruction fault (and a deterministic run retried the
  faulting step forever: the mm_1k_4k certificate run that did not finish).
  `probe_smem_addr_wrap_matches_hardware` (tests/fixtures.rs) replays it.
- Where: `exec.rs` `resolve`.
- Status: **validated** (above).
- If wrong: a *false alarm* (a fault) or a *miss* (an access resolved to the wrong offset) in
  kernels whose shared address parts overflow 32 bits.

### E-BSYNC · exec
**`BSSY Bb` records the lanes executing it; `BSYNC Bb` waits until all of them that have not
exited arrive, then orders them like `bar.warp.sync`** (their clocks are joined). ptxas compiles
`__syncwarp()` after a divergent region to the reconverging `BSYNC` alone (rung 3a: lane 0
copies a tensor-map template to shared memory, then after `BSYNC` all 32 lanes read it).
- Where: `exec.rs` `Op::Bssy` / `Op::Bsync`, `Warp::bmask`; `explore.rs` `bsync_parked`,
  `whole_or_parked`, `local_ample` (BSYNC arrivals and runs beside parked lanes expanded
  alone, see "BSYNC reconvergence" under E-DRF).
- `BMOV.32[.CLEAR] Rd, Bb` copies barrier b's lane mask into Rd (`.CLEAR`: and empties
  it); `BMOV.32 Bb, Ra` sets it from Ra (Triton's parallel_attn_bwd_dkv: `BSSY B11`,
  `BMOV.32.CLEAR R2, B11`, a region, `BMOV.32 B11, R2`, `BSYNC B11`). The mask encoding in
  the register is the model's: a kernel whose saved register is read by anything but a
  restore is refused (`l2::check_bmov`). A `.CLEAR` by part of a warp while the barrier
  records lanes outside that part, and a restore whose lanes disagree, are unmodeled (the
  case clears its spare B10, empty, from two parts of a warp). Not probed in isolation; the case's output is the evidence.
- Rung 4 (nested regions, ptxas's division slow path inside a loop body): a lane parked at
  another barrier register (or at a `WARPSYNC`) is not waiting at this `BSYNC`; parts of a
  warp that execute one `BSSY` separately add their lanes to its mask, and a release removes
  the released lanes; lanes already parked at the `BSYNC` take part in it. (Before, a release
  counted lanes parked at any convergence barrier, and a second part's `BSSY` overwrote the
  mask: s_alu2 deadlocked and faulted in some schedules.) `BREAK Bb` (`BREAK.RELIABLE`, torch's
  elementwise loops leaving a region early) removes the executing lanes from Bb's mask, and
  lanes already parked at Bb's `BSYNC` are released if the rest of the mask is now there.
  `BREAK Pc, Bb` (`@!P0 BREAK.RELIABLE !P2, B2`, ptxas's div.rn.ftz slow path, paired with
  `@!P0 BRA !P2, ...`) removes only the lanes where the guard and Pc hold, as S-BRA-COND's
  branch takes them (`probes/div_ftz`: div.rn.ftz.f32 on every pair of 32 edge patterns =
  GB300 on the SASS front-end, `probe_div_ftz_matches_hardware`, 2026-11-06).
- A lane's `EXIT` re-checks the lanes of its warp parked at a plain `WARPSYNC` or a `BSYNC`
  (`exit_warpsync_release`; each barrier lanes are parked at, since nested regions park at
  different ones): exited lanes do not take part. Before 2026-11-05 only
  `WARPSYNC.COLLECTIVE.ALL` was re-checked, so lanes that parked at `WARPSYNC.ALL` before
  their warp-mates' `EXIT` stayed parked (a DPOR-only false deadlock on e_aligned_bar AB3,
  SASS; GB300 completes, probes/bar_exit_warp `half`).
- A plain `WARPSYNC Rm` whose lanes name different masks (2026-11-07; CuTeDSL RouterEpilogue:
  `WARPSYNC R8`, R8 = 3 << (lane & 0x1e), before pair shuffles; PTX `bar.warp.sync` on a
  per-lane mask) releases each mask group once its live lanes are parked, at this or another
  plain `WARPSYNC`. Before, only the first arriving lane's group was released and the rest
  stayed parked (a false deadlock in both RouterEpilogue msl_ops cases; now clean and bitwise
  = their GB300 captures). probes/warpsync_pairs (pair-mask exchanges through shared memory:
  converged, after a branch, the two lanes of each pair on different paths) = GB300 on the
  PTX front-end (50/50 scrubbed). ptxas 13.0 lowers those `bar.warp.sync` to
  `WARPSYNC.COLLECTIVE`; pair lanes at different collective instructions enter together
  (E-COLLECTIVE), so the SASS front-end equals GB300 too.
- Status: **argued** from the ISA's reconvergence semantics and the CUTLASS source
  (`__syncwarp` there); not probed. Before this, BSSY/BSYNC were no-ops, which reported a race
  on that pattern. s_alu2's divergent division paths match GB300 bitwise.

### S-TMAP · code
**A TMA operand that is not a declared tensor-map parameter is a 128-byte `CUtensorMap` in
global memory**, decoded with the layout in docs/tensormap.md (reverse-engineered from
`cuTensorMapEncodeTiled`, 3097/3097 random encodings, driver 580). im2col maps (`cuTensorMapEncodeIm2col`) add
the pixel-box corners (word 0x38) and pixels per column (word 0x3c), from single-field
encoder sweeps (docs/tensormap.md). The thread issuing the TMA
must have executed `fence.proxy.tensormap::generic.acquire` (SASS `CCTL.E.C.LDCU.IV.DEEP` +
`UTMACCTL.IV`) on that address since its contents last changed, or another thread of its CTA
must have fenced the same bytes and that fence must happen-before the TMA (rung 4b: the
CUTLASS 4.x grouped GEMM's tensormap-updater warps update and fence each map, then hand it
to the load and epilogue warps through an mbarrier pipeline, `IsTensorMapUpdateAsync`);
otherwise a violation is reported (the TMA unit may use a stale cached copy). The descriptor bytes are read as an
ordinary load by that thread at the fence and at the TMA (race-checked).
- Maps built on the device (`tensormap.replace` into zeroed memory, published with
  `tensormap.cp_fenceproxy`: Triton's device-side descriptors, CuTeDSL grouped GEMMs) leave
  the driver's derived words 0x40 (box bytes) and 0x48 (swizzle repeat) 0. The decoder takes
  0 there as "not provided"; any other value must still equal the derived one.
  `probes/tmap_device` (2026-11-06): GB300 TMA-loads the right box through such a map (5/5
  runs) and the published bytes match the simulator's bitwise (SASS front-end: ptxas lowers
  `tensormap.replace` to shared loads / LOP3 / stores, and the PTX front-end does not parse it).
- **A tensor-map parameter with `raw` bytes is passed as those bytes (word 0 = the buffer's
  address)**, as `hw::run` passes it, so kernels that copy the map out of the parameters and
  patch it (CuTe `GroupedGemmKernel`, `_Sm100GroupedTf32x3GemmKernel`, dist_moe combine:
  `LDC` the first 64 bytes into shared memory, clear bit 21 of word 8, patch address / dims /
  strides, publish with `ATOMG.EXCH`) see the template. Before, the parameter bytes were 0 and
  the published map decoded as rank 1 (TMA dimensionality fault). `probes/tmap_param_copy`
  (2026-11-06): a driver-encoded f32 map copied (64 bytes) out of the parameters, patched with
  `tensormap.replace` (address, dim 1 = 24) and published; GB300 loads the box with rows past
  24 zero-filled (5/5 runs) and the box and published bytes = simulator (SASS). Without `raw`
  the parameter bytes stay 0 (the declared geometry is used only for TMAs on the parameter).
- Where: `exec.rs` `tmap_at`, `Op::TmapFence`; `tmap.rs`; `lib.rs` `build_launch`.
- Status: **validated** for the layout (encoder sweep); the fence rule follows the PTX memory
  model (proxy fences are per thread). The meaning of descriptor bits 1 and 21 of word 8 is
  unknown (reported, not used).
- **Tensor-map parameters** (`kind = "tensormap"` with the captured `raw` bytes) take element
  size, extents, strides and box from the decoded descriptor, not from the dtype the case gives
  the buffer (2026-11-06). The capture records a buffer's dtype from the tensor it matched, and
  CuTe's FusedDoubleRMSNormBwd maps f32 over two buffers captured as f64: with 8-byte elements
  each TMA copy counted 0x2000 bytes against an expect_tx of 0x3000 for three 0x1000-byte
  copies, so phase 0 completed after two copies. Descriptors with element strides or a swizzle
  atom the TMA model does not cover keep the case geometry (`lib.rs` `param_tmap_geometry`; test
  `tensormap_param_geometry_comes_from_the_descriptor`).
- PTX front (2026-11-07, 20 SCALE corpus records): `tensormap.replace.tile.<field>
  .shared::cta.b1024` lowers to ptxas 13.0's read-modify-write of the field's bits in shared
  memory (layout learned from ptxas's code for every field, ordinal and enum value: 0x08 rank /
  elemtype / interleave / swizzle / fill bits, 0x0c + 4 i and 0x1c strides, 0x20 + 4 i dims - 1,
  0x34 element strides - 1, byte 0x37 + i box dims - 1; elemtype 11 / 12 also set bit 16, which
  0..10 leave alone); `tensormap.cp_fenceproxy...release.gpu.sync.aligned [g], [s], 128` is lane
  l copying word l as a release store, the warp leaving it together; `fence.proxy.tensormap
  ::generic.acquire.<scope> [a], 128` is the SASS front's TmapFence. probes/tmap_replace
  (random map bytes, every field / ordinal / value): GB300 50/50, both front-ends bitwise;
  probes/tmap_device and probes/tmap_param_copy now bitwise on the PTX front too.

### S-BRA-COND · code
**`@P0 BRA P1, target`** (a guard and a condition operand; the CUTLASS grouped GEMM, rung 3a)
**is taken by the threads where both P0 and P1 hold**; the others fall through.
- **`@P0 EXIT P1`** (torch's `reduce_kernel`: `if (mult != 0 && threadIdx.x != 0) return`)
  likewise exits the threads where both hold; it is lowered as that branch to a plain `EXIT`
  of the function.
- Where: `sass` `BRA` / `EXIT` lowering (`Op::Bra::cond`), `exec.rs` `Op::Bra`.
- Status: **argued** from the surrounding code (P1 is a loop-continue flag, P0 a bound check);
  the EXIT form is **validated** by `fixtures/r4_sum_torch` (GB300 bitwise; the previous
  lowering ignored the operand and exited every thread, writing no output).

### S-TMEM · code
**TMEM is per CTA (one SM), 128 lanes × 512 columns, allocated in 32-column units through a
16-unit bitmap;** words are unallocated / allocated-unwritten / written.
- Where: `exec.rs` `tmem_*`, `machine.rs` `Tmem`.
- **M=64 MMA** (`cta_group::1`): D row i is lane lane0 + 32(i/16) + i%16, lane0 = 0 or 16.
  **M=128** (`cta_group::1`): row i is lane i, for every N = 8..256 in steps of 8 (PTX ISA
  table 48; `probes/tc_m128`: N = 8, 16, 24, 40, 136, 256 for kind::f16 and 8, 24 for
  kind::f8f6f4 write exactly lanes 0..127 x columns 0..N-1 of a sentinel-filled TMEM, both
  front-ends bitwise). Block-scaled M = 128 N = 8 (FA4 decode's `UTCQMMA`; `probes/mma_bs`
  `expected/n8`, 2026-11-06: mxf8f6f4 1X, mxf4 2X, mxf4nvf4 4X with ue8m0 / ue4m3, three N = 8
  MMAs accumulating onto an N = 64 one) writes columns 0..7 only, equal bit for bit to the
  N = 64 result's. A in TMEM with N % 16 == 8 (the FP8-KV decode's P.V MMA, N = 8), outside
  the PTX ISA's table (N % 16 == 0 with A in TMEM): probes/tc_mma_ts_n8 (2026-11-08, GB200,
  N = 8 and 24) writes exactly columns col0..col0+N of all 128 lanes, equal to GB300's N = 16
  output on the shared columns, both front-ends bitwise; modeled as Layout D and reported as a
  `contract` finding (`mma-ts-shape`). Block-scaled N % 16 == 8 above 8 is rejected (not
  measured).
  M = 128 with `cta_group::2`: E-PAIR.
  **`tcgen05.ld.16x64b.xN`** (SASS `LDTM.16dp64bit`): lane l of the warp reads TMEM lane
  l/4 + 8(l%2), register j column 2j + (l/2)%2. **SASS loads take the lane quadrant from the
  warp**, and the address's lane field only mod 32 (cuBLAS passes lane 0 / 16 from every
  warp); the PTX front-end still reports a lane outside the warp's quadrant.
- **`.16x32bx2`** (SASS `LDTM` / `STTM.16dp32bit_t0_t15` and `_t16_t31`, one per lane
  half): lane l of the half reads / writes TMEM lane (address lane) + l%16, N consecutive
  columns. **`.16x256b`** (`LDTM.16dp256bit`): register 4j + 2h + c of lane l is TMEM lane
  l/4 + 8h, column 8j + 2(l%4) + c (an mma accumulator fragment). **`.16x128b.xN`**
  (`LDTM.16dp128bit.xN`, cuBLAS's NVFP4 2-CTA epilogue): register 2j + h of lane l is TMEM
  lane l/4 + 8h, column 4j + l%4 (`probes/tmem_16x128b`: .x1, .x2, .x4 and lane field + 16,
  both front-ends = GB300). **`tcgen05.st.16x256b` / `.16x128b`** (SASS `STTM.16dp256bit` /
  `STTM.16dp128bit`) write the same cells the loads of those shapes read
  (`probes/tmem_st16`, 2026-11-06: .x1 / .x2 / .x4 of 16x256b, .x2 / .x4 / .x16 of 16x128b,
  lane field 0 and 16, every cell of the 32 x 128 block read back = GB300). PTX
  **`.16x32bx2 [a], split`** is the two SASS halves: lanes 0..15 at a, lanes 16..31 at a +
  split columns (`probes/tmem_x2`: st .x2 / .x4 / .x16 and ld .x8 = GB300).
  **`LDTM.STAT.16dp32bit_t0_t15` / `_t16_t31`** (`tcgen05.ld.red.16x32bx2`, FA4 decode's row
  max) reduce the N values the half loads, as the 32x32b form (`probes/tmem_ldred16`: .x64
  max, .x16 min.NaN, .x8 max.abs, .x4 min.abs.NaN on special-heavy data = GB300). The PTX
  front-end lowers `tcgen05.ld.red.{32x32b,16x32bx2}...{min,max}[.abs][.NaN].f32 {d..}, red,
  [a][, split]` to the same load with the reduction into `red`; other forms are refused
  (probes/tmem_ldred and tmem_ldred16 kernel.ptx = GB300 bitwise, 2026-11-07).
  **A in TMEM**
  (`tcgen05.mma [d], [a], ...`): A row i at the lane D's row i uses, relative to A's lane
  field, two bf16 per 32-bit column, the even K index in the low half. Narrow kinds
  (f8f6f4, block-scaled mxf8f6f4 / mxf4 / mxf4nvf4; the NVFP4x2-KV attention's P.V
  `UTCQMMA.2CTA tmem[..]`): the row's 32-byte K slice in 8 columns, byte k at column k / 4,
  byte k % 4 (probes/mma_bs/ts, 2026-11-08: mxf8f6f4 scale_vec::1X, M = 128, four K slices;
  GB200's output equals the A-from-shared-memory run's bit for bit, both front-ends bitwise).
  cta_group::2 with a narrow A in TMEM follows E-PAIR (each CTA's rows from its own TMEM):
  probes/mma_bs/mn `pair_ts_*` (mxf8f6f4, M = 256, N = 128) = GB200 bitwise. **MN-major** shared
  operands (swizzled): CUTLASS's canonical ((T,s/16,m),(8,k)) : ((1,T,LBO),(s/2,SBO)).
  Without swizzle (kind::f16; the mslk decode kernels' A): ((T,1,m),(8,k)) : ((1,T,SBO),(1T,LBO)),
  i.e. element (r, k) at (r/8) SBO + (r%8) 2 + (k/8) LBO + (k%8) 16 bytes: the roles of LBO and
  SBO swap relative to the swizzled forms. `probes/tc_m128` `mma_img` (raw shared-memory images):
  A, B and both MN-major, two LBO / SBO arrangements each, M = 128 / 64, N = 8..32, decode
  correctly and equal both front-ends bitwise. The narrow kinds with 8-bit elements (e4m3 /
  e5m2, kind::f8f6f4 and mxf8f6f4) use the same layouts with T = 16 one-byte elements: swizzled,
  element (r, k) at (r%s) + (r/s) LBO + (k%8) s + (k/8) SBO bytes, then the address swizzle;
  without swizzle, (r%16) + (r/16) SBO + (k%8) 16 + (k/8) LBO. The swizzle uses the absolute
  address: a start 512 B into a 1024-byte SWIZZLE_128B atom (the mxfp8 decode's MN-major A) reads
  the absolute phase. probes/mma_bs/mn (GB200, 2026-11-08): host-built shared-memory images, one
  CTA (M = 128, N = 64; A MN-major SWIZZLE_128B, the same at +512 B, SWIZZLE_64B with LBO 512,
  SWIZZLE_32B with LBO 256, no swizzle; B SWIZZLE_64B / 32B; A and B) and cta_group::2 (M = 256,
  N = 128, B rows 64r.. per CTA; A in TMEM with B MN-major SWIZZLE_64B / 32B / none, A in shared
  memory MN-major, both): every D equals the K-major layout's bit for bit (8,192 / 32,768
  outputs), and both front-ends equal GB200. Controls: the MN image read with bit 15 clear, or
  written with a start-relative swizzle phase, changes all 8,192 outputs. MN-major FP6 / FP4
  operands, MN-major A in TMEM and swizzled MN-major starts off a swizzle-row boundary are
  refused (not probed).
  Validated end to end: Triton's flash attention (S = QK^T read back with `.16x32bx2`, P
  stored to TMEM and used as A, V as an MN-major B) and cuBLAS nvjet (`.16x256b`) match GB300
  bitwise.
- Status: **validated** for one CTA per SM (B41's leak is reported by the driver too).
  Several CTAs sharing one SM's TMEM is not modeled (rung 1b/1c, B08). The M=64 layout and
  16x64b mapping: `probes/tmem_16x64b` and r2_cublas_gemm bitwise = GB300 (1000 runs).

### S-TCCP · code
**`tcgen05.cp.cta_group::1.32x128b.warpx4 [taddr], sdesc` (SASS `UTCCP.T.S.4x32dp128bit`)
copies 32 rows x 16 bytes from shared memory to TMEM columns taddr..taddr+3.** With
`cta_group::2` (`UTCCP.T.S.2CTA.4x32dp128bit`) each CTA of the pair does the same from its own
shared memory into its own TMEM (E-PAIR). The PTX `[taddr+off]` offset adds to the address
word, as SASS `tmem[URa+off]`. Row r goes to
lane 32w + r of every sub-partition w (the four copies are identical), and word j of the lane
is bytes 4j..4j+3 of the row. Source rows follow the K-major matrix descriptor: without
swizzle, 8-row core matrices SBO apart with rows 16 B apart; swizzled, rows `span` apart and
then the address swizzle.
- The copy is an asynchronous op in the issuing thread's tcgen05 pipe: in order with that
  thread's MMAs and commits. It reads shared memory as the async proxy and writes TMEM as the
  tcgen05 pipe actor, both race-checked. The numerics shadow forgets the written words (scale
  codes, not values).
- Where: `exec.rs` `complete_tc_cp`. Front-ends: PTX `tcgen05.cp`, SASS `UTCCP`.
- Status: **validated** end to end by `fixtures/r3_cute_mxf8`. There CUTLASS copies its
  512-byte scale-factor atoms with this form: row i holds the 4 K-block scales of rows i,
  i + 32, i + 64 and i + 96. Both front-ends equal GB300 bitwise on all 16,384 outputs, with a
  distinct random scale per (row, K-block); the same holds for the B30 mutant. The pair form:
  `probes/tc_pair_bs/cp` and `fixtures/c_smm_nvf4` (E-PAIR).
- Not modeled, reported as errors: other shapes (128x256b, 4x256b, 128x128b, 64x128b
  warpx2), and decompression (`.b8x16.b6x16_p32` / `.b4x16_p64`).

---

## L: SASS lift (binary front-end)

### L-ALIAS-FENCE · code
**SASS alias fence.** ptxas 13.0 lowers `fence.proxy.alias` (sm_103a, `probes/multimem`) to
`LDCU URx, c[0x0][0x2d8]; UP2UR.B3 URx, UPR, URx, 0xff; ULOP3.LUT UP0, URZ, URx, 0x1, ...;`
then `MEMBAR.SC.GPU` if bit 0 of the driver word is set, else `MEMBAR.SC.SYS; CCTL.IVALL`, then
`UR2UP UPR, URx.B3`. The SASS front credits a `MEMBAR.SC.*` inside such a `UP2UR.Bk` / `UR2UP`
bracket as an alias fence (S-MC-ALIAS), in addition to its release / acquire effect, and reads
c[0x0][0x2d8] as 0. `membar.sys` / `fence.sc.sys` give `MEMBAR.SC.SYS; ERRBAR; CGAERRBAR;
CCTL.IVALL` with no bracket.
- Status: **validated**: the lowering by disassembly; c[0x0][0x2d8] = 0 on every GB300
  read (`probes/multimem/c2d8`, a patched `LDC`; 4 GPUs of a GB300 node with and
  without a multicast object, 1 and 2 visible GPUs, and one GPU of a second host:
  `results/c2d8.txt`). What bit 0 means to the driver is not known.
- Where: `exec.rs` `Op::Fence { Release }`, `alias_bracket`, `cbank0` (0x2d8).
- If wrong: *miss* (a pair that is not an alias fence credited) or *false alarm* (another
  lowering of the alias fence).

### L-NVDISASM · code
**`nvdisasm` (CUDA 13.0) prints what the bits encode.**
- Where: `sass` `load_cubin`, `parse_disasm`.
- Status: **trusted** (vendor tool). Cross-checked indirectly by every bitwise hardware match.
- Encodings nvdisasm 13.0 prints nothing for (a cubin from a newer ptxas: CuTe DSL's bundled
  one) leave a hole in the listing, and the kernel is refused ("instruction addresses are not
  contiguous 16-byte slots"). `undis.rs` decodes one such encoding from the kernel's `.text`
  bytes, printed as nvdisasm 13.4 prints it: `F2FP.SATFINITE.BF16.S2_6.UNPACK_B Rd, Ra[.H1],
  Rb.H0 | imm.H0` (`cvt...bf16x2.s2f6x2`). Every fixed bit must match the observed pattern
  (any other value stays a hole). Fields checked against nvdisasm 13.4 on all 1,088 instances
  in the kernel DB (FA4 int4 / fp8 KV attention, token mean pooling) and the 56 of
  probes/s2f6_cvt (2026-11-08), where the decoded kernel = GB300 bitwise.

### L-ROSETTA · code
**Each SASS opcode/modifier lowering means what the Rosetta probe derived,** only for the
forms seen. Any other form is a hard error.
- Where: `sass` `lower_line`.
- `.E` on `LDG` / `STG` / `LD` / `ST` / `REDG` / `ATOMG` means a 64-bit address register pair
  even when nvdisasm prints `[R10]` without `.64` (cuBLAS's cutlass3x SIMT sgemm computes the
  pair with `LEA` / `LEA.HI.X`; the case matches GB300 bitwise, docs/compile-vs-eager.md).
- `@g EXIT Pc` is `@g BRA Pc` to a plain EXIT, or, when the function has none, to an EXIT
  guarded by the same `@g` (Triton's `_remap_contiguous_ids`: `@!P0 EXIT P1` ... `@!P0
  EXIT`): the branching lanes reach it directly, so the guard still holds there.
- `ATOMG.E.INC` (`atom.inc.u32`): old >= v ? 0 : old + 1 (`probes/atom_inc`, 2026-11-06:
  49 edge pairs and random ones, two increments per word, both front-ends = GB300).
- `REDG.E.{OR,AND,XOR}[.64].STRONG.{SM,GPU,SYS}` (`red.global.{or,and,xor}.b32 / .b64`): one
  atomic bit operation per thread (`probes/red_bitops`, 2026-11-08: four threads per word at
  each scope, both front-ends = GB300).
- `[R.U32+UR]` addresses (`ST.E [RZ.U32+UR6], R2`, CuTe DSL dist kernels): the register
  zero-extended plus the 64-bit uniform pair. Read from the operand syntax, not probed (ptxas
  does not emit it for plain PTX).
- `LDSM.8.MT1616[.2]` (`ldmatrix.m16n16.x1 / .x2.trans.b8`, FA4 fp8 decode): byte b of
  register 2m + h of lane l is matrix m's row 4(l%4) + b (the row address of lane 16m + that
  row), column l/4 + 8h (`probes/ldsm_b8`, 2026-11-06: every byte traced to its source, x1
  and x2 = GB300, SASS front-end; the PTX front-end does not lower `ldmatrix`).
- Status: **validated** per form by differential runs (r0, r1a and r1b bitwise: PTX model =
  SASS model = GB300). Rung 1b added `UTMASTG`, `UTMACMDFLUSH` (bulk commit), `DEPBAR.LE` on
  the flush's read scoreboard (bulk wait), `SYNCS.ARRIVE.TRANS64.A1T0` (plain arrive),
  `R2UR P, URd, Ra` (uniformization loop; P = "lane value differs", the only polarity under
  which the single-lane loop exits), `CS2R Rd, SRZ`, `MEMBAR.ALL.CTA` and `CCTL.IVALL`, plus
  128-bit `STS` (a model bug caught by the differential run: the SASS model stored the low
  8 bytes twice).
- MXF8 pairs (fixtures/c_smm_mxf8, cuBLASLt nvjet, bitwise = GB300 on its captured input and
  in 1000 random-input GB300 runs against an L0 proof): the immediate of `[U]IADD3.64` is the
  encoding's 32-bit field (bits 32..63), sign-extended (`UIADD3.64 UR52, ..., -0x800` is
  `0xfffff800_...`; zero-extended, the tensor-map pointer it steps back left the parameter
  block and the run faulted).
- PTX forms lowered to the same IR as the SASS ptxas emits for them (2026-11-05, the mslk
  decode kernels): `redux.sync.{add,min,max}.{u32,s32}` / `.{and,or,xor}.b32` /
  `.{min,max}[.abs][.NaN].f32` (full member mask only; every lane gets the result) as
  `REDUX` / `CREDUX`; `neg.f32` as `FADD d, -a, -RZ` (a NaN becomes the canonical NaN, not a
  sign flip); `setp.<cmp>[.ftz].f32 p, a, b` as `FSETP.<cmp>.AND p, PT, a, b, PT`.
  fixtures/e_ptx_redux (+-0, +-inf, a NaN) = GB300 bitwise on both front-ends.
- PTX forms of the mslk decode kernels (2026-11-06; fixtures/s_ptx_packed = GB300 bitwise on
  both front-ends, 50/50 runs, with subnormal, infinite and NaN lanes; ptxas output read with
  nvdisasm): `fma.rn.f32x2` / `mul.rn.f32x2` / `add.rn.f32x2` / `sub.rn.f32x2` on .b64
  registers as `FFMA2` / `FMUL2` / `FADD2` (sub: `FADD2 d, a, -c`), operands `.F32x2.HI_LO`
  (lane 0 = the low 32 bits): one rounding to nearest per lane, subnormals kept, NaN results
  the canonical 0x7fffffff. Only `.rn` is accepted: without it ptxas may contract a mul and an
  add into one FFMA2, and `.rz/.rm/.rp`, `.ftz`, `.sat` and immediate operands are refused.
  `cvt.rn.{bf16x2,f16x2}.f32 d, a, b` as `F2FP.{BF16,F16}.F32.PACK_AB d, a, b` (a in the high
  half; `.relu` / `.satfinite` / other roundings refused unless the narrow-format path, N-CVT-LP,
  covers them). `ld` / `st` `.v2.b64`, one register per element. Vector operands whose
  registers are not consecutive (`st.shared.v4.b32 [a], {%r262, %r261, %r260, %r259}`) go
  through hidden scratch registers `%__vec0..7`: moves into them before a store, out of them
  after a load (ptxas allocates the registers instead; the values are the same). `div.s32`
  (a zero divisor is a fault: the PTX ISA leaves the result unspecified) and `cp.async.ca`
  4 / 8 bytes, `.cg` 16 bytes (`LDGSTS.E` / `.E.64` / `.E.BYPASS.128`) are covered by the same
  fixture.
- Numerics of those forms: the two halves of a 64-bit PTX register are f32x2 lanes, each with
  its own node (`num::lane_reg`; the high half under the pseudo index `idx | V_HI`, which the
  PTX front-end keeps free). `mov.b64` packing and unpacking, 64-bit moves, loads and stores
  keep each lane's node (crates/sim/tests/ptx_lanes.rs); `fma.rn.f32x2` is tracked as FFMA2 is
  (x*1 + 0 outside segment mode; per lane in segment mode). Segment mode builds no slice for half
  a register: a tainted lane without a node is a recorded slice failure, not a guess.
- Lift gaps of the kernel DB (agent L, 2026-11-06; each form refused outside what its probe
  covers, modifier sets checked against the sm_103a encoding table, third-party, lookup only):
  `REDUX.SUM.S32` (redux.sync.add.s32) is the wrapping sum over the active lanes, as
  `REDUX.SUM` (probes/redux_s32: signed extremes, wrapping sums, 8 partial active masks behind
  `BRA.DIV`, both front-ends where PTX allows); `ATOMG` / `REDG .E.ADD.S32` add as `.ADD`
  (same probe); the `ATOMG` / `REDG` modifier set must be one of the listed forms
  (`atomg_op`: `.F16x2`, `.F32x2`, `.F64`, `.INC` / `.DEC`, ... are refused; before, an
  unlisted float form such as `.ADD.F32x2` lowered as an integer add); a 32-bit
  `ATOMG.E.CAS PT, Rd, [Ra], Rb, Rc` with unrelated compare / new registers is compare Rb,
  new Rc (probes/bf16x2's CAS loops); `IMAD.WIDE[.U32] d, a, b, -c` is a*b - c mod 2^64
  (probes/int_misc; with a carry or `~c` refused); `UP2UR URd, UPR, URa, mask` is P2R on the
  uniform predicates (no probe: ptxas does not emit it for PTX written to provoke it; the
  CuTe grouped TF32x3 GEMM uses it to materialize flags); PTX `.local` arrays
  (`__local_depot0`, through `%SPL`) are the thread's local memory with offsets from 0, and
  scalar `ld` / `st .local` of up to 32 bits are SASS `LDL` / `STL` (the `.loc` filter used to
  drop `.local` declarations); `popc.b32` is `POPC`, `popc.b64` a `POPC` per half and an add (probes/popc64, GB300 20/20,
  both front-ends bitwise, 2026-11-07); `abs` / `neg .f16[x2]` are `HADD2 d, |R|, -RZ.H0_H0` /
  `HADD2 d, -R, -RZ.H0_H0` (per lane |x| or -x plus -0: a NaN becomes 0x7fff; probes/f16_absneg
  on every f16 pattern, GB300 20/20, both front-ends bitwise, 2026-11-07); `not.b16/32/64` is
  XOR with all ones. A narrowing integer `cvt` to a signed type (`cvt.s8.s32 %rs, %r`, Triton's
  int8 dequantize) sign-extends the kept bits to the destination register (PTX ISA; ptxas
  `I2F.S8` for a following `cvt.rn.f32.s16`); the PTX front zero-extended them before
  2026-11-07 (a sim != GPU kernel-DB record, mslk fa4 dequantize_kv): probes/cvt_s8_narrow,
  GB300 20/20, both front-ends bitwise. A signed load narrower than 64 bits into a register
  declared 64 bits wide (`ld.param.s32 %rd11`, Triton's ignore index compared with an int64
  label) sign-extends to 64 bits (PTX ISA; the PTX front zero-extended above bit 31 before
  2026-11-07: the sim != GPU record msl_ops triton._finish.f32.72): probes/ld_sext64, GB300
  20/20, both front-ends bitwise.
- Rung 2e (cuBLAS nvjet, r2_cublas_gemm bitwise = GB300): `PMTRIG` (no-op);
  `SYNCS.ARRIVE.TRANS64.RED.A0TR` / `.A0TX` (no arrival; expect / complete tx, checked against
  ptxas: `mbarrier.expect_tx` / `complete_tx`); `UTCBAR.MULTICAST` (cta_group::1, mask operand
  as for `.2CTA`, checked against ptxas); `BRA.U P, UPT` (both must hold; only UPT modeled);
  `BRA.U.NOT_TID0` (warp-uniform: only the warp of CTA thread 0 falls through,
  `probes/not_tid0` with the branch patched into a ptxas cubin); `FFMA2` (`fma.rn.f32x2`,
  operands `.F32` broadcast / `.F32x2.HI_LO` / `.LO_HI`, checked against ptxas);
  `R2UR d, a` of a lane-varying value when the first use of d is a logical right shift into d
  that drops every varying bit (`R2UR UR6, R2` with R2 = %tid.x then `>> 5`: the warp index);
  a TMA store's shared::cta source carrying the CTA's own rank byte.
- torch.topk (fixtures/s_warpcoll bitwise = GB300): `REDUX.OR/.XOR/.SUM`, `CREDUX.MIN/.MAX
  [.S32]`, `SR_LEMASK/GEMASK/GTMASK/EQMASK`, `BRA.DIV`, `WARPSYNC.COLLECTIVE`,
  `ENDCOLLECTIVE` (E-COLLECTIVE).
- Triton bf16 kernels (fixtures/c_triton_gelu_bf16 bitwise = GB300, and 65,536 outputs in the
  elementwise GB300 check): `FMUL2` / `FADD2` with an f32 immediate operand (`0.5`, `1`),
  broadcast to both lanes, its bits taken from the encoding's word 32..63.
- Rung 4 (softmax, layer/rms norm, reductions; fixtures/s_alu2 bitwise = GB300): `FADD`
  with `.RZ/.RM/.RP/.FTZ` and `-x` / `|x|` operands, `FFMA` / `FMUL` with `.RZ/.RM/.RP/.FTZ/
  .SAT`, `FMNMX[.FTZ|.NAN]` / `FMNMX3`, `FSETP` (all 16 comparisons, `.FTZ`, `.AND/.OR/.XOR`),
  `FSEL` (= SEL), `FCHK`, `FMUL2`, `VIMNMX[.U32]` (selector PT = min, !PT = max, like
  VIADDMNMX), `POPC`, `SR_LTMASK`, `LDG.E.U8/.U16` (zero-extended), NaN immediates (`+QNAN`,
  printed alike for every payload: bits taken from the encoding), `@P EXIT Q` (S-BRA-COND).
  `R2P PR, R, mask` (inverse of P2R) is lowered but not exercised by a hardware test.
  Equivalence fixtures (docs/equiv-reassoc.md; c_attn_core_ref / c_attn_core_flash bitwise =
  GB300): `FADD2` (f32x2 sum, as FFMA2 with b = 1). `LDGSTS.E[.LTC128B]` 4- and 8-byte forms
  (cp.async.ca 4 / 8 bytes, zero-fill predicate as for .128) are lowered; the plain forms are
  exercised by fixtures/s_ptx_packed (= GB300), the zero-fill forms by no hardware test.
  From D's torch-op sweep (docs/capture.md; the kernels' outputs match GB300 bitwise where they
  run to the end): `FADD.SAT`, `FSEL` with `-x` / `|x|` operands (the modifier applies to the
  selected value), `VIADD Rd, Ra, -Rb` / `-URb`, `BREAK.RELIABLE` (E-BSYNC), `ST.E` (generic =
  global here: only global memory reaches it in these kernels) and `STG.E/ST.E.U8/.S8/.U16/.S16`
  (low byte / half of the register; the sign forms store the same bits).
- Rung 5 (flash attention, torch sm80 `flash_fwd_kernel`, cuDNN SDPA, Triton's tcgen05 flash
  attention; each matches GB300 bitwise):
  - `HMMA.16816.F32.BF16` (warp-collective, PTX ISA fragments; N-HMMA), `LDSM.16.M88.N` /
    `.MT88.N` (ldmatrix, N = 1/2/4; `fixtures/s_sm80`), `LDGSTS.E.BYPASS[.LTC128B].128` with
    an optional source predicate (`!PT` / `P`: src-size 0, zero fill), `LDGDEPBAR`, `DEPBAR.LE
    SBk, n` on LDGDEPBAR's scoreboard (E-CPASYNC; s_sm80), `STL[.128]` / `LDL[.LU]` (local
    memory, private per thread; s_sm80), `LEA Rd, P, -Ra, Rb, s` (b + ~(a << s) + 1 with its
    carry; s_sm80), `IDP.2A.{LO,HI}.{U16,S16}.{U8,S8}` (`fixtures/s_dp2a`, all 8 forms).
  - tcgen05 (S-TMEM): `LDTM` / `STTM.16dp32bit_t0_t15` / `_t16_t31` (`.16x32bx2` split by
    lane halves), `LDTM.16dp256bit[.xN]`, `UTCHMMA tmem[URa], gdesc[..]` (A in TMEM), MN-major
    B descriptors (instruction descriptor bit 16), `STSM.16.MT88.N`. The bracket offset of
    `tmem[UR+off]` is part of the address (`+0x100000` = lane 16).
- Compile-vs-eager gaps (fixtures/s_lift_k: every form on all pairs of the s_alu2 operands,
  `hw` 1000/1000 bitwise; and the captured kernels, each certified clean and = GB300 bitwise:
  fixtures/c_cve_rope_bf16_inductor, c_mm_f32_gemmsn, c_cve_sdpa_f32):
  - `FFMA2` with a negated operand (`-R16.F32`, `-R6.F32x2.HI_LO`, also `-a`): the sign bit
    of each lane flipped before the fma (`Op::Ffma2` `neg`; Triton's bf16 RoPE);
  - `VIMNMX3[.U32] d, a, b, c, PT / !PT`: minimum / maximum of three (`Op::IntMnmx3`;
    cuBLAS gemmSN_NN_kernel), and `STS.128 [Ra], RZ` (16 zero bytes);
  - `FSET.BF.cmp.AND d, a, b, p`: 1.0f / 0 (`Op::Fset`), `FCHK P, a, <f32 immediate>`,
    `ATOMS.MIN` / `.MAX` / `.MIN.S32` / `.MAX.S32 RZ` (32-bit integer min / max),
    `LDL.U8` / `.U16` (zero-extended) and `STL.U8` / `.U16` (low byte / half into its word),
    `HMMA.1688.F32.TF32` (N-HMMA): CUTLASS fmha_cutlassF_f32;
  - a warp-collective matrix instruction (`LDSM`, `STSM`, `HMMA`) whose guard is false in
    every lane is a no-op (`@P0 LDSM.16.M88.4` in fmha_cutlassF_f32).
- Rung 5, D's torch-op sweep (each capture's output matches GB300 bitwise):
  - `ATOM.E.ADD.F32.FTZ.RN.STRONG.GPU P, RZ, [Ra.64], Rb` (generic f32 atomic add; P = the
    address was global), `ATOM.E.CAST.SPIN` / `ATOMS.CAST.SPIN` (CAS loops: d / P = swapped),
    `QSPC.E.{S,D}` (false: no generic shared window is modeled): torch's `index_add`.
  - `SHFL.{UP,DOWN,IDX,BFLY} P, ...` (P = the source lane was in range), `BMSK[.W]` /
    `UBMSK` (`fixtures/s_bmsk`).
  - `LDG.E.ENL2.256 Rhi, Rlo, [..]` (bytes 0..16 to Rlo, 16..32 to Rhi, checked against
    ptxas's `ld.global.v8.b32`); with a fourth operand `mask` (ptxas's form of a
    `ld.global.v8.b32` / `.v4.b64` only some of whose elements are used; bit i = word i), each
    half's selected words go, in order, to consecutive registers from that half's base (RZ
    for a half with none) and no other register is written (2026-11-07; before, all eight
    were written, clobbering live registers; probes/ldg256_mask, 18 element subsets, GB300
    20/20, both front-ends bitwise; PTX `.global.v4.b64` lifted with it), `LDG.E.S8/.S16` / `LDS.S16` (sign-extended), `F2F.F32.F64` /
    `.F64.F32` / `.F16.F32`, `F2FP.F16.F32.PACK_AB`, `HADD2.F32 Rd, -RZ, Ra.H0_H0` (f16 ->
    f32), `F2I.S64/.U64.TRUNC`, `I2F.S64/.U64/.S16/.U16 (Ra.H1)/.S8`, `LEA.HI` with `-Rb`
    (`fixtures/s_cvt`).
  - `DFMA` / `DMUL` / `DADD` `[.RZ|.RM|.RP]` with `-x` / `|x|` and f64 immediates, `DSETP`
    (`fixtures/s_f64`; N-FALU), and `DSETP.{MIN,MAX}.AND Pd, Pq, a, b` (ptxas's fmin / fmax:
    Pd = take a (-0 below +0, a NaN b loses), Pq = both NaN; s_f64).
- Rung 3b (flashinfer trtllm-gen gemm1, fixtures/r3b_trtllm_bmm certified and = GB300
  bitwise; docs/rung3b.md):
  - `UTMALDG.2D.GATHER4 [URa], [URb]`: URa+2 the column, URa+3..URa+6 four row coordinates
    (S-TMA-GATHER4; fixtures/s_gather4, PTX and SASS = GB300 1000/1000);
  - `SYNCS.ARRIVE.TRANS64.RED.A0T1 RZ, [a], RZ` (no arrival, expect one transaction unit) and
    `ARRIVES.LDGSTSBAR.64.TRANSCNT [a]` / `.ARVCNT [a]`: ptxas's `cp.async.mbarrier.arrive` /
    `.noinc` (E-CPASYNC-MBAR; fixtures/s_cpasync_mbar = GB300 1000/1000);
  - `SYNCS.PHASECHK.TRANS64.TRYWAIT PT, [a], R` (result discarded): lowered as a no-op (E-SPIN).
- Real CuTeDSL kernels (FA4 forward / decode, quack softmax, mslk decode, FA4 block sparsity;
  2026-11-05):
  - `CREDUX.MIN|.MAX|.MINABS|.MAXABS.F32[.NAN]` (redux.sync f32): FMNMX's order (-0 < +0) over
    the active lanes, `ABS` of |x|, NaN inputs ignored unless `.NAN` (then the canonical NaN);
    a lone NaN gives the canonical NaN too. `PLOP3.LUT Pd, Pd2, a, b, c, lut, lut2`: Pd2 =
    lut2(a, b, c) with the same index convention (only `0xc8`, the block-sparsity kernel's
    all / any, exercised). `R2P PR, Ra.Bk, mask`: byte k of Ra. `LEA.HI d, a, -b, RZ, s`
    (torch's fp8 conversion). All in fixtures/s_lift_l (262,144 words, GB300 bitwise, 20/20
    runs; 32-, 16- and 1-lane reductions over special f32 patterns).
  - `LDTM.STAT.xN.MIN|MAX[ABS].F32[.NAN] Rred, Rd, tmem[..]` (tcgen05.ld.red 32x32b; FA4's row
    max): Rd.. as `LDTM.xN`, Rred the min / max of the N values with the CREDUX semantics;
    `Rred = RZ` is a plain load. GB300 traps (misaligned address) when the start column is
    odd; the model reports a fault there. `LDTM tmem[UR+-0x10]`: the negative offset added to
    the 32-bit TMEM address (mslk decode). probes/tmem_ldred (16,384 words, all-NaN and +-0
    rows, GB300 bitwise 100/100). `STTM tmem[UR+-0x60]` likewise (2026-11-07, 6 SCALE corpus
    records refused it as "bad offset"): probes/tmem_stneg (`STTM.x8 tmem[UR4+-0x10]`, read
    back) = GB300 on both front-ends, 50/50 scrubbed.
  - `BAR.SYNC Ra, Ra` / `BAR.ARV Ra, Ra` (ptxas's register form): id = Ra bits 0..3, thread
    count = Ra >> 16 (ptxas builds `count << 16 | id & 0xf`; validated through
    fixtures/s_bardiv, E-COLLECTIVE-ALL). Two different registers are not lowered.
  - `F2I[.FTZ][.U32|.S64|.U64][.TRUNC|.FLOOR|.CEIL][.NTZ]` (cvt.r{z,n,m,p}i: libdevice
    sinf / cosf use `F2I.FTZ.NTZ`): `.FTZ` flushes a subnormal source first, the f32 is
    rounded to an integer, then converted with saturation (NaN: 0, or 2^63 for 64-bit);
    `.NTZ` has no effect on the integer. `I2F.F64[.S64|.U64|.S32][.RZ|.RM|.RP]`: correctly
    rounded int -> f64. Other F2I / I2F modifiers are not lowered (before, any F2I with
    `.TRUNC` and any I2F without F64 / F16 / FTZ were accepted whatever else they carried).
    fixtures/s_f2i (393,216 words, GB300 bitwise 20/20); the Triton 1 - cos(x) /
    2 sin^2(x/2) pair (agent AN) then certifies, bitwise = GPU.
- PTX front additions (2026-11-07, SCALE pilot): `abs[.ftz].f32` as ptxas's FADD |a|, -RZ and
  `abs.s32` as IABS (probes/abs_f32); `cvt.rn.f32.f64` / `cvt.f64.f32` (F2F), `cvt.sat.f32.f32`
  (FADD.SAT RZ, a) and `cvt.r{n,m,p,z}i[.ftz].{s,u}{32,64}.f32` (F2I) (probes/cvt_misc); a
  register or partial immediate `redux.sync` / `shfl.sync` / `vote.sync` membermask (per
  mask group: E-WARPCOLL, probes/redux_mask, probes/coll_split, probes/shfl_mask); immediate elements in vector stores; `prmt.b32` (default mode,
  probes/prmt_ptx); `createpolicy.fractional` when its result is only the policy operand of
  `.L2::cache_hint` instructions (a zero placeholder: L-PTX-HINT; other uses refused);
  `sqrt.rn.f32` as the IEEE square root, NaN 0x7fffffff (the f64 root rounded to f32, then
  + -0; GB300's ptxas code gives exactly that on 4096 patterns, probes/sqrt_rn; 10 SCALE
  corpus records); `brx.idx[.uni] i, $L` with `$L: .branchtargets ...` (per thread, as `bra`, to entry i; an
  index outside the list is a fault, as SASS `BRX` to an address that is not a branch target;
  probes/brx_idx, 30 SCALE corpus records). GB300 bitwise on both front-ends where probed.
- `copysign.f32 d, a, b` (2026-11-07, 36 SCALE corpus records): b with the sign bit of a, a
  bit operation (ptxas: `LOP3.LUT d, b, 0x80000000, a, 0xb8`). probes/copysign (NaN payloads,
  signed zeros, Inf, subnormals, an immediate operand): GB300 50/50, both front-ends bitwise.
- `cvt.rna.tf32.f32` (2026-11-07, 12 SCALE corpus records): ptxas adds 0x1000 to a finite
  operand and masks with 0xffffe000 (ties away from zero); Inf / NaN are only masked, so a NaN
  whose payload is all below bit 13 becomes Inf. probes/cvt_tf32 (ties, near overflow, NaN
  payloads, Inf, subnormals): GB300 50/50, both front-ends bitwise.
- `bfi.b32 d, a, b, pos, len` (2026-11-07, PX: 17 kernel-DB records): b with bits pos ..
  pos + len - 1 replaced by the low bits of a, pos and len taken mod 256, bits above 31
  dropped, b itself for len 0 or pos > 31 (ptxas: PRMT of the low bytes, BMSK, SHF.L, LOP3
  0xe2). `clz.b32` (6 records): 31 - bfind.u32, 32 for 0 (FLO.U32 + IADD3). probes/bfi_clz
  (pos / len around 0, 31, 32, 255, 256; immediates; zero and all-ones a): GB300 50/50, both
  front-ends bitwise. `.b64` forms are refused.
- `cvt.ftz.sat.f32.f32` / `cvt.ftz.f32.f32` (2026-11-07, PX: 49 kernel-DB records): ptxas
  emits `FADD.FTZ.SAT d, RZ, a` (clamp to [0, 1], NaN and subnormals +0) and
  `FADD.FTZ d, a, -RZ` (subnormals to a zero of their sign, -0 kept, NaN canonical).
  probes/cvt_ftz (signed zeros, subnormals, NaN payloads, infinities, 1 and neighbors): GB300
  50/50, both front-ends bitwise.
- Float literals (2026-11-07, PX: 38 kernel-DB records): a decimal literal (`1.0`, `20.`,
  `.5e1`, `1.4426950408889634`) is an f64 value (PTX ISA) rounded to nearest even in an `.f32`
  instruction (`1.00000017881393432617`, a tie, gives 0x3f800002; `1e-46` +0; `1e39` +Inf);
  a `0d` literal in an `.f32` instruction is converted the same way; a `0f` literal keeps its
  32 raw bits, zero-extended in an `.f64` instruction (`mov.f64 d, 0f3DCCCCCD` gives
  0x000000003dcccccd on GB300, not 0.1f widened). The type is the instruction's last type
  modifier; a decimal literal in an instruction whose last type is not `.f32` / `.f64` is
  refused. probes/float_lit: GB300 50/50, both front-ends bitwise.
- PTX syntax (2026-11-07, SCALE pilot): `%lanemask_{lt,le,ge,gt,eq}` as SR_LTMASK etc.;
  registers declared without `%` inside an inline-asm scope (`{.reg .s64 a; ...}`, CuTeDSL);
  `[%rd+-N]` addresses; a `{...}` operand list starting a continuation line (after a comma, or
  after a bare mnemonic line as in `mma.sync...f32` then `{%r1, ..},`) is an operand, not a
  scope; a directive line without `;` (`.loc`) still ends a statement. Before, such a list
  unbalanced the scope pass, which then renamed a later label and reported it unknown.
  probes/ptx_syntax: GB300 50/50 scrubbed, both front-ends bitwise.
- Packed f32x2 with no rounding modifier (2026-11-07): `mul.f32x2` is `.rn` when every use is
  a multiplicand, a store, a conversion, or an unpack (`mov.b64 {lo, hi}`) whose halves are only
  stored or converted (Triton's RMSNorm before `cvt.rn.bf16x2.f32`), and `add` / `sub.f32x2`
  when neither operand is a float mul without `.rn`; ptxas cannot contract those into FFMA2.
  probes/f32x2_norn (FMUL2 / FADD2 in ptxas's output): GB300 100/100 scrubbed, both front-ends
  bitwise. Also `.rn` (2026-11-07, 301 SCALE corpus records refused): a `mul.f32x2` read as an
  `fma.rn.f32x2` addend, or unpacked into halves read by any scalar op (add / sub / neg / fma
  addend), a half repacked into the other lane of a vector read by anything but a repack
  (Triton's `mov.b64 {_, hi}, p; mov.b64 v, {hi, x}; add.f32x2`), or into its own lane of a
  vector read only as above. ptxas 13.0 emits FMUL2 and the readers apart (no FFMA / FFMA2);
  probes/f32x2_unpack (cancelling operands: a contraction would differ in 43 of 255 lanes):
  GB300 50/50 scrubbed, both front-ends bitwise. 238 of the 301 records then lift. A product
  read directly by a plain `add` / `sub.f32x2` (FFMA2) stays refused.
  A half repacked into its own lane next to a value not from its product (2026-11-08, CuTe /
  FA4 backward preprocess: `{q.hi, p.hi}`, `{q.lo, p.lo}`) is treated as the other-lane case,
  and integer reads of the product bits (`shr` / `shl` / `and` / `or` / `xor`, Triton's
  stochastic rounding) as stores. The identity repack `{p.lo, p.hi}` (or with a copy, select
  or second unpack of p in the other lane) stays refused: ptxas 13.0 contracts it into FFMA2.
  probes/f32x2_repack (cancelling addends): FMUL2 + FADD2 in ptxas's output, GB300 50/50
  scrubbed, both front-ends bitwise; 8 PX records then lift.
- A plain `mul.f32x2` whose register no `add` / `sub.f32x2` can read (2026-11-08,
  `x2_no_add_readers`; by register name, so loop-carried and redefined accumulators are
  covered): every statement naming it reads it as a multiplicand or fma operand, stores it,
  selects it (`selp`), copies it (`mov`, followed) or unpacks it into halves no statement
  repacks. ptxas has nothing to contract it into (FMUL2). probes/f32x2_sinks (a selp of the
  product then an add, the product as an fma addend, a copy as a multiplicand, a loop-carried
  product unpacked into scalar adds; cancelling addends): GB300 50/50 scrubbed, both fronts
  bitwise; 6 PX records then lift. Halves that are repacked (directly or through copies)
  are followed into the repacked vector, which must pass the same test (2026-11-08: a Triton
  fp8 GEMM's accumulator scaled each iteration, unpacked into mma accumulators and repacked);
  a repacked vector an add reads stays refused.
- `add` / `sub` / `mul` `.bf16x2` / `.bf16` with no rounding modifier (2026-11-07, SCALE
  pilot): the same rule as f32x2 (ptxas emits HADD2.BF16_V2 / HFMA2.BF16_V2 ..., -RZ).
  probes/bf16_norn: GB300 50/50 scrubbed, both front-ends bitwise. Plain `fma.bf16x2` stays
  refused. A pack `mov.b32 %r, {%h1, %h2}` of plain muls that is only stored is accepted
  (ptxas emits one HMUL2.BF16_V2 for the two muls; probes/bf16_norn word 4, SCALE pilot
  dash/1703).
  Through packs and unpacks (`.b32`, `.b64`), `prmt` and `shfl.sync` (2026-11-07, 10 SCALE
  corpus records), a plain `mul.bf16` / `.bf16x2` whose products reach only stores,
  conversions, `mma` operands or multiplicands is `.rn` too (Triton's dequant into an
  `mma.sync` B operand; a fused-attention recompute through prmt / shfl / prmt into cvt).
  probes/bf16_sink: GB300 50/50, both front-ends bitwise (ptxas: HMUL2.BF16_V2, PRMT, SHFL,
  HMMA, no HFMA2). Also through a `selp` data operand (Triton's MX GEMM NaN override
  `selp.b16 %h, 0x7FC0, %prod, %p`) and into `stmatrix` / `tcgen05.st` (2026-11-08, 9 PX
  records): probes/bf16_selp, GB300 20/20, both front-ends bitwise (HMUL2.BF16_V2, SEL, STSM /
  STTM, no HFMA2).
- `rcp.rn.f32` (2026-11-07, SCALE pilot) as `div.rn.f32 d, 1.0, a`: the IEEE reciprocal,
  subnormals kept, NaN 0x7fffffff. probes/rcp_rn (4096 inputs: specials, subnormals,
  magnitudes near 2^+-126; GB300 50/50 scrubbed, also equal to the f64 reciprocal rounded to
  f32): both front-ends bitwise. `.ftz` stays refused.
- `cvt.{rz,rm,rp}.f32.f64` / SASS `F2F.F32.F64.{RZ,RM,RP}` (2026-11-07, SCALE pilot): directed
  rounding of the f64 (overflow to the largest finite value where the direction says so).
  probes/cvt_f64_rnd: GB300 50/50 scrubbed, both front-ends bitwise.
- More PTX forms (2026-11-07, SCALE pilot): `cvt.{rn,rz,rm,rp}.{f32,f64}.{s,u}{16,32,64}` as
  I2F (probes/cvt_i2f: edge and random 64-bit inputs); `abs.s64`, `brev.b32`,
  `rcp.{rn,rz,rm,rp}.f64` as `div.<rnd>.f64 d, 1.0, a` (probes/ptx_misc2). GB300 50/50
  scrubbed each, both front-ends bitwise.
- PTX `tcgen05.mma [taddr + off], [a + off], ...` (2026-11-07, SCALE pilot): the offset adds
  to the TMEM address word, as SASS `tmem[URa+off]` and `tcgen05.cp`; the PTX front computes it
  into a vector scratch register under the statement's guard. probes/tc_mma_off (M = 128,
  N = 16: D at column 64 with A in shared memory, D at column 128 with A in TMEM at
  [taddr + 184 + 8]): GB300 50/50 scrubbed, both front-ends bitwise. GB300 also ran N = 8 with
  A in TMEM (the PTX ISA requires N % 16 == 0 there) and gave the A-in-shared-memory result;
  modeled since 2026-11-08 as a `contract` finding (probes/tc_mma_ts_n8, S-TMEM above). An immediate enable-input-d (`..., idesc, 0` / `1`,
  CuTeDSL; PX: 22 kernel-DB records) is a constant predicate: 0 overwrites D, 1 accumulates
  (ptxas: UTCHMMA ... !UPT / UPT). probes/tc_mma_imm (`0` then `1` on the same D, and `0` with
  A in TMEM): GB300 20/20 scrubbed, both front-ends bitwise. Operand order after idesc
  (PTX ISA): an optional braced disable-output-lane vector, the enable-input-d predicate,
  then an optional scale-input-d immediate (0..15; D = A*B + D*2^-s, not with
  `.block_scale`). A brace-less second trailing operand is scale-input-d, never a mask:
  `..., 0, 1` overwrites D (enable 0, the scale of an unread accumulator is moot). The scale
  is `Op::Mma.scale_d` (scale-input-d bullet of N-MMA). Kernel DB (2026-11-08):
  381 records with tcgen05.mma, 164 with a mask (vector + enable, read as such before),
  none with a scale-input-d operand.
- `min` / `max.f64`, `neg` / `abs.f64` (2026-11-07, PX: 2 + 2 kernel-DB records), as ptxas
  emits them: `DSETP.MIN/MAX.AND p, q, a, b` (p: a is the min / max, -0 below +0, a NaN
  loses; q: both NaN), a 64-bit select, and when both are NaN b's NaN quieted (high word |
  0x80000, payload kept: not canonical as for f32); `DADD d, -RZ, -a` / `DADD d, -RZ, |a|`.
  The PTX front uses two hidden predicates for p and q. probes/f64_misc (all pairs of 15
  specials incl. signaling / negative NaNs with payloads, random): GB300 50/50, both
  front-ends bitwise. `rsqrt.approx.f64` without .ftz (PX: 4 records) follows ptxas's
  expansion exactly (`fp::rsqrt_approx_f64`): MUFU.RSQ64H of the high word, one DFMA
  refinement (e = 1 - a y0^2, y = y0 + (0.375 e + 0.5) y0 e), and outside [2^-1022, 2^1023)
  its slow path (+-0 -> +-Inf, NaN -> a + a, negative -> 0xfff8000000000000, +Inf -> +0,
  subnormal -> the refinement of a 2^54 times 2^27). probes/rsqrt_f64 (4096 inputs: specials,
  boundaries, 1024 subnormals, every exponent, random words): GB300 20/20, both front-ends
  bitwise.
- `div.rn.ftz.f32` / `sqrt.rn.ftz.f32` (2026-11-07, PX: 6 + 5 kernel-DB records): the IEEE
  quotient / root of the operands with subnormals flushed to a zero of their sign, the result
  flushed too, NaN canonical. probes/ftz_div_sqrt (subnormal operands, quotients around
  2^-126, specials, random; 85 quotients differ from the unflushed IEEE ones): GB300 50/50,
  both front-ends bitwise (SASS: certified; the slow-path divergence exceeds the explicit
  exploration budget).
- 64-bit and vector `ld` / `st` `.local` (2026-11-07, PX: 9 kernel-DB records, most also
  needing device calls): one access per element at its offset (`.v2` / `.v4`), a 64-bit
  element one virtual register holding both words (`LdLocal` / `StLocal` on a V register of 8
  bytes). probes/local_vec (v4.b32, v2.b64, b64, f64 stores read back crosswise; ptxas:
  STL.128 / LDL.128 / LDL.64): GB300 20/20, both front-ends bitwise. `cvta.local` and generic
  addresses into local memory stay refused.
- Cluster special registers (2026-11-07, PX: 11 kernel-DB records): `%cluster_ctaid.d` =
  ctaid.d mod cluster.d, `%cluster_nctaid.d` = cluster.d, `%clusterid.d` = ctaid.d / cluster.d,
  `%nclusterid.d` = grid.d / cluster.d (ptxas reads the shapes from the driver table at
  `c[0x0][0x2a0 + ... - 0xa0 * SR_CgaSize]`, S-CGA-SHAPE). probes/cluster_sreg (grid 4 x 4 x 2,
  clusters 2 x 2 x 1, all 14 cluster registers): GB300 20/20, both front-ends bitwise.
- Byte moves (2026-11-07, PX: 27 + 3 kernel-DB records): `mov.b32 r, {b0, b1, b2, b3}` packs
  the low byte of each (b0 lowest), `mov.b16 h, {lo, hi}` the low bytes of two (upper half
  clear), `mov.b16 {lo, hi}, h` unpacks two bytes zero-extended (as the existing
  `mov.b32 {b0, b1, b2, b3}, r`). probes/mov_pack (CUTLASS-style `.b8` scoped registers: byte
  reverse, byte swap, mixed words): GB300 50/50, both front-ends bitwise. `mov.b128` is
  refused (no 128-bit registers).
- Inline-asm scopes (2026-11-07, PX): a scoped range declaration (`.reg .u32 %tmp32_<1>;`,
  `.reg .pred %p<1>;`, Triton's spin loops) renames its expanded names (`%tmp32_0`) as well,
  so their uses resolve (they were "unknown register": 33 kernel-DB records) and a scoped
  `%p0` no longer refers to the body's `%p0` (an alias that clobbered it); a word right after
  `.` is never renamed (`mad.lo.u32` next to a scoped register `lo` became `mad.lo__sN.u32`).
- SASS `FFMA2` / `FMUL2` / `FADD2` with a negated uniform register (`-UR6.F32`,
  `-UR6.F32x2.HI_LO`: ptxas folds `neg.f32` of kernel parameters) and `|R|` operands (abs.f32
  of both halves folded; applied before a negation) (2026-11-07, SCALE pilot).
  probes/ffma2_ur: GB300 50/50 scrubbed, both front-ends bitwise.
- `cp.async.bulk.prefetch.tensor.<n>d.L2.global[.tile][.L2::cache_hint]` / SASS
  `UTMAPF.L2.<n>D` (2026-11-07, SCALE pilot) as a cache hint, no memory effect (like
  `UBLKPF.L2`); `.im2col` forms refused. probes/tma_prefetch: GB300 50/50 scrubbed, both
  front-ends bitwise.
- `atom` / `red` `.add.f64` / SASS `ATOMG` / `REDG` `.E.ADD.F64.RN` (2026-11-07, SCALE pilot):
  one f64 add rounded to nearest, subnormals kept, as DADD (inf + -inf gives 0xfff8000000000000
  on GB300, as the DADD model). Not commutative for the explorer (the order changes the bits).
  probes/atom_f64 (each thread's own word: specials, ties, subnormal sums, overflow; plus exact
  red adds to 4 shared words): GB300 100/100 scrubbed, both front-ends bitwise.
- `mad.wide.{s,u}16 d, a, imm, c` (2026-11-07, SCALE pilot): d = ext16(a) * ext16(imm) + c in
  32 bits (a extended into the scratch register; a register multiplier is refused).
  probes/mad_wide16: GB300 50/50 scrubbed, both front-ends bitwise.
- `set.<cmp>.{u32,bf16x2,f16x2}.{bf16x2,f16x2}` / SASS `HSET2[.BF16_V2][.BF].<cmp>.<bop> d, a,
  b, Pc` (2026-11-07, SCALE pilot): HSETP2's per-lane comparison combined with Pc, written per
  half as 0xffff / 0, or with `.BF` 1.0 in the operand format (bf16 0x3f80, f16 0x3c00) / 0.
  probes/hset2 (14 comparisons over bf16 and f16 specials, register and immediate-pair
  operands): GB300 50/50 scrubbed, both front-ends bitwise.
- `movmatrix.sync.aligned.m8n8.trans.b16` / SASS `MOVM.16.MT88` (2026-11-07, SCALE pilot,
  binary-only kernel dash/4901): warp-collective transpose of the 8x8 matrix of 16-bit
  elements (row lane / 4, columns 2 (lane % 4) + {0, 1}); undefined halves move with their
  data; the f32 numerics see the result as a fresh value. probes/movm (lane-coded and random
  halves, two transposes in a row): GB300 50/50 scrubbed, both front-ends bitwise.
- DeepEP hybrid / DeepCP tokens (2026-11-06; probes/memhints, GB300 500/500 scrubbed runs,
  both front-ends bitwise): `LDG.E.NA[.U8|.64][.STRONG.SYS]` and `LDG.E.NA.128.CONSTANT` are
  `.L1::no_allocate` (a cache hint; `.CONSTANT` = `.nc` as before), `ST.E.EF.128` is
  `.L1::evict_first` (a hint); `.SM` is the CTA scope on `LD` / `ST` / `ATOM` (ptxas prints
  `ld.relaxed.cta` as `LD.E.64.STRONG.SM`; a scope token without `.STRONG` stays weak, the
  weaker reading); `STG.E.64.MMIO.SYS` is `st.mmio.relaxed.sys`, lowered as a strong relaxed
  `.sys` store (other `.MMIO` forms refused); `UBLKPF.L2` (`cp.async.bulk.prefetch.L2`) is a
  no-op. ptxas wraps the prefetch in a uniformizing loop: `R2UR P0, UR8, R12` sets P0 per lane
  when its value differs from the chosen one, `R2UR.OR P0, UR9, R13` ORs the high half's
  difference into P0 (before, `.OR` was accepted and ignored), and a `PLOP3` before the first
  R2UR computes from the undefined P0 a result nothing reads: an undefined predicate result
  now stays undefined until used (S-INIT-PRED). ptxas also merges adjacent strong accesses of
  different scopes into one at the wider scope (`ld.relaxed.gpu` + `ld.relaxed.sys` of
  adjacent words as `LD.E.128.STRONG.SYS`; the stronger access is a valid implementation).

### L-RPCMOV · code
**`RPCMOV.32 Rpc.LO, Ra` / `RPCMOV.32 Rd, Rpc.LO` move a register into and out of a
per-thread 32-bit slot (the low half of the return-PC register) that only RPCMOV touches**.
ptxas parks a register there under register pressure (CuTe DSL / CUTLASS kernels). Modeled
as a per-thread register past the kernel's register count (`cuobjdump -res-usage`), which no
instruction of the kernel names. A kernel that also contains a `CALL` other than into a
`BPT.TRAP` function (which could clobber Rpc) is refused, as is a register count of 0 or 255.
- Where: `sass/src/l2.rs` (`RPCMOV`, `check_kernel`).
- Status: **validated** on GB300 by `probes/rpcmov` (a save after the loads and a restore
  into another register, inserted by patch.py: each of 256 threads gets its own value back,
  `probe_rpcmov_matches_hardware`), 2026-11-06. That a non-trapping CALL clobbers Rpc is not
  tested (refused instead).
- If wrong: *miss* or *false alarm* (a wrong restored value).

### L-PTX-HINT · code
**PTX cache hints do not change semantics:** `cp.async`'s and non-tensor `cp.async.bulk`'s
`.L2::cache_hint` (with its cache-policy operand) and `.L2::64B` / `.L2::128B` / `.L2::256B`
prefetch sizes are accepted and dropped by the PTX front-end. Every other unknown modifier of the instructions it lowers
is rejected (fail closed).
- Where: `ptx` `lower_inst`, the `cp.async` and `cp.async.bulk` arms (`ASSUME(L-PTX-HINT)`).
- Status: **argued**: the PTX ISA defines them as cache-eviction priorities and prefetch
  sizes, with no effect on the values copied or on ordering. fixtures/s_ptx_cute_ops (one
  `.L2::128B` copy) = GB300 bitwise. probes/bulk_hint (cp.async.bulk with `.L2::cache_hint`
  both ways; ptxas puts the policy in the UBLKCP memory descriptor) = GB300 20/20 on both
  front-ends (2026-11-08).
- SASS: the L2 eviction policy of a 256-bit global access (`LDG` / `STG .E.ENL2.256`,
  `.ELL2` = PTX `.L2::evict_last`, `.EFL2` = `.L2::evict_first`) is likewise dropped
  (2026-11-07; 4 SCALE corpus records refused ELL2): probes/ldst256_l2 = GB300 on both
  front-ends, bitwise equal to probes/ldst256's output.
- If wrong: *miss* only if a hint changed the data or the completion order.
- Retire: never needed for verdicts; listed so that dropped modifiers are visible.

### PTX-FMAD · code
**A PTX float `mul` without a rounding modifier and the `add` / `sub` without one that reads
it are contracted as ptxas 13.0's default (`--fmad=true`) contracts them.** The PTX ISA leaves
this to the code generator: "mul/add sequences with no rounding modifiers may be optimized to
use fused-multiply-add instructions on the target device", while an explicit rounding modifier
is "treated conservatively". The rounding of such a pair is therefore implementation-defined
at the PTX contract level (either value is allowed); the model fixes the one ptxas 13.0
produces. The SASS front-end is exact (it executes the FFMA / FMUL + FADD ptxas emitted).
- Where: `ptx` `fmad_pairs` (analysis), `lower_inst` add / sub arm (`ASSUME(PTX-FMAD)`).
- Contracted (one FFMA / FFMA.FTZ / FFMA.SAT / DFMA, a sub negating the product or the addend):
  an `.f32` / `.f64` mul whose every live reader is an add / sub of the same type and `.ftz`
  setting with the product in one operand, in the same basic block after it (a label no
  branch names does not end the block), unguarded, with the mul's operands not redefined in
  between: ptxas contracts it into each reader and drops the mul. Readers through a plain
  copy (`mov`) or a `neg` (FFMA -a, b, c) count as the readers they feed. `.sat` on the add
  is kept (FFMA.SAT). All or nothing per mul (2026-11-08, probes/fmad_multi): any other live
  reader keeps the FMUL and every reader then rounds the product (FMUL + FADD): a store, a mul
  or fma operand, a compare, `max`, a conversion, a shuffle, a reduction, `add.rn`, an add of
  the other `.ftz` setting, `add p, p`, a `neg` or copy read by such, a reader in another
  block or reached through a join (a guarded mul read by an unguarded add: FMUL + FADD). Dead
  readers do not count (ptxas removes them first; liveness by register name, so a statement
  counted dead is dead). An add / sub of two such products folds the one with no other
  reader, the first operand's when both have none (probes/fmad_multi: also when the second
  is defined first); the other mul keeps its FMUL for all its readers. Readers are the
  statements the mul's definition reaches: every reader of the register when it has one
  definition, else those before the next unguarded (or same-guard) redefinition in the block
  (inline-asm scopes that reuse one register for a chain).
- Scalar `.f16` (2026-11-08, probes/fmad_f16): the same rule, a plain `mul.f16` into plain
  `add.f16` / `sub.f16` readers (one HFMA2 each, the result in the low half); a converted,
  stored or f16x2-packed product keeps HMUL2 + HADD2. Refused for `.f16`: a `mov` copy or `neg`
  of the product read by an add (not probed), an add of two products.
- Not contracted, as ptxas: a rounding modifier on either instruction; `mul.ftz` with an add
  without `.ftz`, or the reverse (FMUL.FTZ + FADD; FMUL + FADD.FTZ.SAT).
- Refused (fail closed), when no other reader keeps the FMUL: a contracting reader in another
  block or before the mul (ptxas contracts across an if-converted branch in some kernels and
  not in others), under a guard, or after the mul's operands change; a guarded mul; a
  definition among several that reaches the end of its block (unless a reader in the block
  keeps the FMUL); an add of two products both read elsewhere too. From the multi-reader rule
  until the f16 extension (both 2026-11-08) such a definition with no reader in its block
  was lowered as not contracted; 15 kernel_db records (loop-carried products) lifted that way
  and are refused again.
- `div.full` / `div.approx [.ftz] .f32 d, a, C` with a constant divisor (an immediate,
  decimal or `0f`, or a register whose only definition is an unguarded `mov` of one that
  dominates the division) is ptxas's `FMUL [.FTZ] d, a, RN(1/C)` for 2^-126 <= |C| <= 2^126,
  C != +-1, and contracts as a plain mul (FFMA a, RN(1/C), c with a following add); C = 1 is
  a (`.ftz`: FADD.FTZ a, -RZ), C = -1 is FADD -a, -RZ. `div_by_const` rewrites such statements
  to that `mul` / `mov` / add before the analysis; other constants (the scaled range) stay
  divisions. Before 2026-11-07 they ran as the general expansion (MUFU.RCP of C): the sim !=
  GPU record mslk fmha.merge_attentions_varargs (lse / log2e, then + max, as one FFMA).
  probes/div_const: 12 forms (immediates, a mov register, decimal, +-1, .ftz, div.approx,
  2^+-126, contracted add / sub) on specials and random values, GB300 20/20, both front-ends
  bitwise; `recip_constants` checks RN(1/C) on 200,000 constants.
- Status: **validated** by probes/fmad_multi (2026-11-08): 28 multi-reader shapes (two adds,
  add + sub, `.sat`, `.ftz`, through `neg` and `mov`, dead readers; add + store / mul / setp /
  `add.rn` / fma addend / `add p, p` / other `.ftz` / `neg` / `max` / `cvt` / copy; adds of two
  products with either read elsewhere), addends cancelling the product to +- 1 ulp: PTX and
  SASS fronts bitwise (GB300 below). 55 kernel_db records then lift, each PTX front = SASS
  front of ptxas's cubin bitwise (`ptxv num`).
  probes/fmad (2026-11-07): every contracted and kept-apart form on
  random, cancelling, subnormal, signed-zero and Inf / NaN operands; GB300 50/50 scrubbed,
  PTX and SASS fronts bitwise (the unfused value differs from GB300's in 122 of 232 finite
  lanes of output 0). Before, the PTX front rounded the product separately: `a = b =
  0x3f800800, c = -1`: GB300 0x3a000400, PTX front 0x3a000000.
- Records: no probe or fixture in the repo has a contractible pair (Triton and CUTLASS emit
  `fma.rn` themselves). KA records with one either check the SASS front (a `cubin`) or do not
  lift for other reasons (mslk fa4 decode / fwd). SCALE corpus PTX files with a plain mul read
  by a plain add / sub: 554 (list kept outside the repo).
- `.f32x2` (2026-11-07, PX: 4 kernel-DB records whose PTX-front output was 1 ulp off GB300,
  e.g. a Triton gated product sum; 127 records refused on plain `mul` / `add.f32x2`): ptxas
  13.0 contracts a `mul.f32x2` into every `add` / `sub.f32x2` reading it when all its readers
  are such adds with the same rounding mode and `.ftz` setting, **explicit `.rn` / `.rz` /
  `.rm` / `.rp` included** (FFMA2, FFMA2.RZ, FFMA2.RM; none = `.rn`, so `mul.f32x2` +
  `add.rn.f32x2` contracts too), the mul then disappearing; with any other reader (a store, an
  fma or mul operand, an unpack, an add of another mode, `add p, p`) it contracts none (FMUL2
  + FADD2). Scalar `.rn` pairs and `.rn` f64 / f16 / bf16 (x2) pairs stay apart, as the ISA's
  "treated conservatively" says; the f32x2 behavior departs from it. Contraction happens
  before ptxas merges equal expressions (probes/fmad2: four equal products in separate
  registers keep their own decisions). An add / sub of two such muls folds its first
  operand's product (FMUL2 of the second, FFMA2), whichever is defined first, when the second
  has no other reader (probes/fmad2: 64 of 64 lanes match that fold, 0 the other) (scalar FFMA does the same; still refused there). A label no branch
  names (Triton's `$L__tmp` debug labels) does not end the block. `fmad2_pairs`; refused: a
  contracting reader in another block, under a guard, before the mul or after its operands
  change, an add of two such muls whose second is read elsewhere too, and a qualifying add
  next to a reader of a kind not probed (e.g. `selp`). probes/fmad2:
  GB300 50/50 scrubbed, both fronts bitwise; the unfused value differs on all 64 lanes of the
  `.rn` pair.
- If wrong: *false alarm* on bitwise / equivalence / error-distribution results of the PTX
  front only; other toolchains or `--fmad=false` may contract differently.

### L-LOP3-P · code
**`LOP3.LUT Pd, Rd, a, b, c, lut, Pq` sets Pd = (result != 0) OR Pq; `LOP3.LUT.PAND` sets
Pd = (result != 0) AND Pq.** Pq may be negated (`!Pq`) and may be Pd itself (read before the
write). ptxas emits only the default form with `!PT` (Pd = result != 0); cuBLAS's NVFP4 2-CTA
nvjet uses `.PAND` for its warp-role dispatch (`LOP3.LUT.PAND P0, RZ, R3, UR4, RZ, 0x30, P0`:
leader CTA and warp range).
- Encoding: bit 80 = `.PAND`, bits 87-89 = Pq, bit 90 = its negation (nvdisasm prints the
  patched forms as expected).
- Where: `sass` `lower_line` (`Op::Lop3::pq`), `exec.rs`, `seg.rs`; taint: a tainted Pq taints
  Pd (`num.rs`).
- Status: **validated** by `probes/lop3_pmode` (not a plain `ptxv probe`: `patch.py` rewrites
  ptxas's `LOP3.LUT P0, RZ, a, b, RZ, 0x30, !PT` into `.PAND` / default with P1 / !P1). On
  GB300, 1,024 triples (half with a & ~b = 0, c of either sign) give Pd = (a & ~b != 0) AND /
  OR (!)P1 in all four forms, and the SASS front-end reproduces all of them (tests/fixtures.rs,
  2026-11-04). Other `LOP3` modifiers are errors.
- If wrong: wrong control flow wherever the predicate output is used.

### L-SHF · code
**A shift by a run-time u32 amount `n` clamps to the operand width, unsigned: `SHF` .U32 / .S32
use min(n, 32), .U64 / .S64 min(n, 64); `.W` takes n mod 32.** The 64-bit `hi:lo` is shifted
(arithmetic for .S32 / .S64 right shifts) and `.HI` selects the upper word. n >= 2^31 is a large
amount, not a negative one. PTX: `shl` / `shr` saturate at the width, `shf.*.clamp` at 32,
`shf.*.wrap` mod 32 (ptxas lowers them to these SHF forms and relies on the clamp, e.g. the
low word of `shl.b64` is `SHF.L.U32 d, x, n, RZ`).
- Where: `ir` `Op::Shf` (`wide`, `wrap`); `sass` `lower_line` (`SHF` / `USHF`; .W with a
  64-bit type and unknown modifiers are errors); `ptx` (`shf`); `exec.rs` `shf_amount` /
  `shf_value` (also the undefined-bit mask).
- Status: **validated** by `fixtures/s_shf` (authored PTX, both front-ends): 64 x 128 lanes x 64
  amounts (0..33, 47 / 48 / 63 / 64 / 65 / 95..97 / 127..129 / 191 / 192 / 255..257 / 511 /
  1023 / 2^16, 2^31 - 1, 2^31, 2^31 + 1, ..., 2^32 - 1) through every form, plus uniform-operand
  rows; GB300 output equal to both front-ends bitwise (20/20 runs, 2026-11-05,
  `s_shf_matches_hardware`). USHF (uniform datapath) shares the model; ptxas emitted no USHF
  for the probe, so it is covered by argument only.
- History: SHF used to take n mod 64. FA4's causal / local masks
  (`r2p_bitmask_below`: `0xffffffff >> max(32 (k + 1) - limit, 0)`, n up to 128) then kept
  columns they mask: c_fa4_causal / c_fa4_local certified clean with 36,288 / 50,751 output
  bytes different from the GPU's. Both are now bitwise equal.
- If wrong: wrong values (silent), wherever a shift amount can reach the width.

### L-CTRL · code
**Control word layout `hi >> 41`:** stall[0:4), yield[4], wbar[5:8), rbar[8:11), wait[11:17),
reuse[17:21); 7 = no scoreboard; waits take effect before issue.
- Where: `scoreboard.rs` `Ctrl::decode`.
- Status: **validated.** 208 one-bit mutants across r0 and r1a: every hardware failure was
  flagged (docs/sass-ctrl.md).

### L-OPREAD · code
**Operand-read timing:**
- memory-pipe instructions read vector registers late, in issue order, so a read scoreboard
  on a later one covers earlier ones;
- vector instructions read uniform registers at issue, with or without a read scoreboard
  (rung 1c: ptxas overwrites the `desc[UR6]` of an `STG` with `rd=SB0` right away);
- uniform-unit instructions read URs late, but predicate operands at issue (rung 2:
  ptxas overwrites the `UP1` operand of a `UTCHMMA` that has `rd=SB2` with no wait, and
  places the SB2 wait right before it overwrites the instruction's `UR48`);
- TMEM loads (`LDTM`) complete in issue order, so a write scoreboard on a later one covers an
  earlier untracked one (rung 2e: ptxas gives only the last of two back-to-back `LDTM`s a
  write scoreboard and uses both results after waiting on it); likewise shuffles (rung 3a:
  an unscoreboarded `SHFL.BFLY` result read after a wait on a later `SHFL.DOWN`'s), and a
  `SHFL` cannot overwrite a register an earlier load/store has not read yet (same in-order
  queue: `STG ..., R21` with no read scoreboard, then `SHFL.BFLY PT, R21`), also when the
  earlier read has a read scoreboard the shuffle does not wait on (Inductor's persistent
  RMSNorm: `@!P0 STG.E.128 [R32.64], R40` with rd=SB0, then `SHFL.BFLY PT, R41, ...` with no
  wait; fixtures/c_cve_addnorm_inductor, GB300 bitwise 1000/1000). Behaviour on GB300
  matches (fixtures/s_alu, bitwise);
- MUFU results complete in issue order, so a write scoreboard on a later `MUFU` covers
  earlier untracked ones (ptxas gives most `MUFU`s of fixtures/s_mufu no write scoreboard and
  reads their results after waiting on a later `MUFU`'s; GB300 bitwise, 100 runs);
- guards are read at issue;
- (rung 4) a memory instruction's operand reads are complete once its write scoreboard is
  (torch's layer_norm: `LDG.E.64 R18, [R18.64]` with wr=SB2 only, a wait on SB2, then R18
  overwritten); an `LDS` with no write scoreboard lands before a later `SHFL` of the same
  warp reads its operands, and is complete once that shuffle's write scoreboard is (Triton's
  row reduction: `@!P1 LDS R3` with no scoreboard, `SHFL.BFLY R4, R3` with wr=SB0, then R3
  read after a wait on SB0). Both are ptxas's own assumptions; the kernels match GB300.
- (rung 5) More in-order units, each forced by a false positive on a kernel that matches
  GB300:
  - an earlier untracked `LDSM`, `S2UR`, or conversion-unit result (`MUFU` / `F2I` / `I2F` /
    `F2F` share one queue) is covered by a later one's write scoreboard;
  - TMEM stores (`STTM`) read their registers before a later `LDTM`'s result lands, and a
    later `LDTM` writes its registers only after the earlier `STTM` / `LDTM`s read theirs
    (`probes/tmem_ldred16`: `STTM.x32 tmem[..], R4` with no read scoreboard, then
    `LDTM.STAT... R3, R4, ..` with no wait; the stored values read back intact on GB300);
  - `LDGMC` (multimem.ld_reduce) is in the same in-order queue as `LDG` (symm_mem
    _ReduceScatterKernel: `LDGMC ... R16, [R8.64+URZ]` then `LDGMC ... R8, [R8.64+UR20]` with
    no read scoreboard; both ranks' outputs = GB300; ptxas's own assumption);
  - a global / local load (`LDG`, `LD`, `LDL`) also writes its destination only after an
    earlier global atomic / reduction (`ATOMG`, `REDG`) read its operands (CUTLASS grouped
    GEMMs: `ATOMG.E.EXCH.STRONG.GPU PT, RZ, [R28], R10` with no read scoreboard, then
    `@P0 LDG.E.64 R28` with no wait; ptxas's own assumption, not yet probed);
  - a load (`LDG`, `LDS`, `LD`, `LDL`, `LDSM`) writes its destination only after the earlier
    instructions of the in-order memory queue (now including `LDGSTS`, `LDL`, `STL`) read
    their operands, tracked by a read scoreboard or not;
  - a generic `ATOM`'s predicate result is known once its read scoreboard completes, also
    when it returns a register on its write scoreboard (`ATOM.E.CAS.64 P0, R8, [R22], R8,
    R10` with wr=SB5 rd=SB0, then `@P0 BRA` waiting on SB0 only; fixtures/s_atom_ext);
  - value-returning shared atomics share the shared-memory queue with `LDS` / `SHFL` /
    `LDSM`: an untracked `ATOMS.MIN R12` result is read after a wait on a later
    `ATOMS.MAX.S32 R13`'s write scoreboard (fixtures/s_atom_ext, GB300 bitwise);
  - `LDTM.16dp32bit_t0_t15` and `_t16_t31` into the same registers write different lanes
    (no WAW).
  The scoreboard parser also stopped reading `idesc[URx]` as 64-bit and an address operand
  first (`SYNCS.CCTL.IV [UR7+..]`) as a destination.
  A load (`LDG`/`LDS`/`LD`) with no scoreboard of its own writes its destination only after
  the earlier instructions of the same in-order memory queue have read their operands (torch's
  reduce: `LDG.E R23, [R12.64]` with no read scoreboard, then `LDG.E R12, [R14.64]`). (rung
  4a) That holds when the load also reads the register as its own address: torch's one-shot
  all-reduce loads the peer pointer table with `LDG.E.64 R8, [R6.64]` ... `LDG.E.64 R6,
  [R6.64+0x18]` (bitwise = GB300 on 3 GPUs).
- Where: `scoreboard.rs` (`ASYNC_READ`, the untracked-read rule).
- Status: **validated** by the sweep (each rule was forced by a hardware counterexample or a
  false positive).
- If wrong: *miss* (WAR on registers).

- Read attribution by unit (2026-11-06): an untracked read moves onto a later instruction's
  read scoreboard only when both are in the same unit (memory queue, global atomics, or the
  same opcode). Before, `SYNCS.ARRIVE.TRANS64 ..., R2` with no read scoreboard was put on the
  next `UTMALDG`'s (which reads only uniform registers) instead of the later `SYNCS.ARRIVE`'s
  that ptxas waits on (Triton grouped fp8 GEMM).
- A wait on a write scoreboard completes the instruction's operand reads even when it writes
  no register (`ATOMG.E.EXCH PT, RZ, [R32], R6` with wr=SB4: the SB4 wait frees R32; CUTLASS
  grouped wgrad). Before, the empty write entry was dropped and the wait did not count.
- A load writes its result after earlier global atomics / reductions read their operands
  (ptxas: no-return `ATOMG.E.EXCH ..., R8` with no read scoreboard, then `LDG.E R8` with no
  wait; CUTLASS mx8 grouped GEMM). From ptxas only, not probed.
- A TMEM load writes its registers after the warp's earlier TMEM stores read theirs:
  probes/sttm_ldtm (`.maxnreg 40` makes ptxas emit `STTM.x16 tmem[UR6], R20` with no read
  scoreboard, then `LDTM.x16 R20, tmem[UR7+0x10]`): GB300 1000/1000 runs store the old R20.
### L-SB-ORDER · code
**A read scoreboard on a memory-pipe instruction covers the operand reads of every earlier
memory-pipe instruction** (`MIO_INORDER`: LDS / STS / LDG / STG / LD / ST / ATOMS / SHFL /
LDGSTS / LDL / STL / LDSM / STSM), tracked by an earlier read scoreboard or not: once it is
released (a wait on it), they have read their registers. Applied only where the covering
instruction executes whenever the earlier one did: it is unguarded, or it has the same guard,
not redefined in between (a guarded-off instruction may release its scoreboard at once; not
probed, kept conservative).
- Where: `scoreboard.rs` `check` (the `rbar` of a memory-queue instruction adds `RCOV`
  covers to the pending reads before it; a wait on that read scoreboard retires them).
- Status: **validated** by `probes/sb_mio_order` (GB300, 2026-11-05, `results.txt`): an
  `STS [Rt]` with no read scoreboard, an `STS [Rt+0x400]` with rd=SB4 (optionally an
  `LDG` with rd=SB2 in between: quack rmsnorm_bwd N=256's shape), then `LDC Rt` waiting on SB4
  only: 0 of 1000 runs wrong (1184 CTAs x 256 threads each); the `LDG`'s read scoreboard as
  the only cover: 0 of 1000. Controls: the read scoreboard on the first STS instead (the
  second uncovered): 200 of 200 runs wrong (~38k words per run); no wait: 200 of 200 (~600k
  words). The static check agrees on all five windows (`probes/sb_mio_order/fixture`,
  `probe_sb_mio_order_matches_hardware`).
- History: until 2026-11-05 an untracked read was attached to the *next* read scoreboard only,
  so quack rmsnorm_bwd N=256 (`@P5 STS.64 [R2], RZ` .. `@!P4 LDG` rd=SB2 .. `@P5 STS.64
  [R2+0x1100], RZ` rd=SB4, `@!P4 LDC.64 R2` waiting on SB4) reported a WAR: a false alarm.
- If wrong: *miss* (WAR on registers).

### L-STALL · code
**Fixed-latency hazards are covered by the stall counts.** These are not checked.
- Where: `scoreboard.rs` (not modeled).
- Status: **trusted.**
- If wrong: *miss*.
- Retire: a per-opcode latency table plus a sweep over stall fields.

### L-MEMSB · code
**Scoreboards that order memory rather than registers** (FENCE completion, waits before
BAR/EXIT on SYNCS) are trusted. The memory model treats the fence as complete at issue.
- Where: `scoreboard.rs` (no registers to track).
- Status: **trusted.** Their removal was benign in all sweep mutants, which is weak evidence.
- If wrong: *miss*.
- `DEPBAR.LE SBk, n, {j,...}` (DeepEP hybrid kernels, DeepCP a2a) is read as `DEPBAR.LE SBk, n`
  plus `DEPBAR {j,...}` (each listed scoreboard drained), the meaning the brace list has in
  the plain `DEPBAR {5,4,3}` form. **Argued**, not probed. For cp.async / bulk groups only the
  SBk part is a group wait (the listed ones stay outstanding: conservative). If wrong: *miss*
  for a register written under a listed scoreboard and read after the DEPBAR.

### L-OPCLASS · code
**The variable-latency (`VAR_LAT`) and asynchronous-reader (`ASYNC_READ`) opcode sets are
complete.**
- Where: `scoreboard.rs`.
- Status: **validated on the corpus** (every opcode ptxas gave a scoreboard is in the sets).
  `POPC` stays variable-latency, but a `POPC` without a write scoreboard is covered by a wait
  on a later `POPC`'s, as for the other in-order units (ptxas for sm_103a schedules it so;
  probes/int_misc and the CuTe DietGpu decode kernels; GB300 500/500 runs correct). The
  third-party sm_103a table's `is_variable_latency` cannot cross-check the sets: it marks
  `LDG`, `LDS`, `MUFU`, `SHFL` and `POPC` alike as fixed latency (baseline defaults, as its
  README says), so it is not evidence either way.
- The f64 pipe (`DADD` / `DMUL` / `DFMA`) completes in issue order (2026-11-06): ptxas for
  sm_103a leaves some of them without a write scoreboard and reads the result after a wait
  on a later f64 op's scoreboard, and lets a later f64 op overwrite a register an earlier
  one reads (rd set) with no wait. probes/f64_queue reproduces both; GB300 output = model,
  both front-ends bitwise (before: a false SCOREBOARD violation and Triton split-k's f64 LSE).
  `DSETP` is in the f64 queue too (2026-11-07): ptxas leaves `DADD R6` with no write
  scoreboard and reads R6 right after a wait on a later `DSETP`'s scoreboard, under 128 stall
  cycles later (CuTe PtgcRecordKernel, a false SCOREBOARD violation; probes/f64_dsetp = GB300
  50/50, both front-ends bitwise).
  Conversions with an f64 operand (`I2F.F64.*`, `F2F.F32.F64`, `F2F.F64.*`, `F2I.*.F64`) are
  in the f64 queue, not the conversion queue (2026-11-07): ptxas leaves `I2F.F64.U64.RZ R8`
  with no write scoreboard and reads R8 after a wait on a later `I2F.F64.RM`'s scoreboard
  only, not on the scoreboard of the `I2F.U16` issued between them (probes/cvt_i2f, GB300
  50/50 scrubbed = model, both front-ends bitwise). `MUFU.RCP64H` stays in the conversion
  queue: no evidence either way.
- The conversion / transcendental queue and the f64 pipe read their operands in issue order
  (2026-11-07): a later op of the same queue (`inorder_queue`: `xu`, or `f64` with the
  conversions that have an f64 operand) does not overwrite a register an earlier one has not
  read yet, with no wait on the earlier one's read scoreboard. ptxas schedules this for sm_103a
  and sm_100a (CuTe top-k: `F2F.F64.F32 R98, R6` with rd=SB1, then `F2F.F64.F32 R6, R19` with no
  wait on SB1; Triton attention backward: `MUFU.EX2 R151, R147` with rd=SB1, then `MUFU.EX2
  R147, R33`; and without the read scoreboard in its own schedules). probes/xu_war: variant
  `rd_sb` puts a never-waited read scoreboard on the first of such a pair; GB300 = model, 200
  runs per variant (before: a false SCOREBOARD WAR violation).
- The f64 pipe is fixed-latency on sm_100a (2026-11-07): ptxas 13.0 for sm_100a gives `DADD` /
  `DMUL` / `DFMA` no write scoreboard and reads their results after stall counts alone
  (`DFMA R4, -R6, R2, 1` with st=8, then `DFMA R4, R4, R4, R4`), where for sm_103a it pads them
  with NOPs or scoreboards them. The check keys on the cubin's SM version (ELF `e_flags`): for
  sm_100 these three are not in `VAR_LAT` (a write scoreboard, if present, is still honoured);
  every other SM keeps the sm_103a sets. probes/f64_queue/cubins/sm100a.cubin (the probe's PTX
  built for sm_100a) certifies with GB300's IEEE-exact outputs; before, a false SCOREBOARD
  violation (also reported on fp64 Triton RMSNorm cubins from a GB200 sweep). `POPC` is not
  changed: ptxas 13.0 still scoreboards it for sm_100a in the int_misc / flo_popc probes.
- `FLO` shares `POPC`'s queue (2026-11-06): ptxas gives `FLO.U32` no write scoreboard and
  reads its result after a wait on a later `POPC`'s (DeepEP hybrid; probes/flo_popc, GB300
  500/500 scrubbed runs = model, both front-ends bitwise). `BREV` is not merged: no evidence.
- `FRND` shares the MUFU / conversion queue (2026-11-06): Triton's mxint8 KV quantizer
  (msl_ops triton/quant/mxint8_kv.py) reads a no-scoreboard `MUFU.RCP` after a wait on a later
  `FRND`'s scoreboard; the capture's output = GB300. ptxas only.
- `REDUX` is one in-order queue (2026-11-06): Triton's MoE `_select_kernel` (msl_ops
  sparse_router) has 20 `REDUX.SUM.S32 URd` with no write scoreboard, each read after waits on
  later REDUXes' scoreboards; the capture's output = GB300. ptxas only, no dedicated probe.
- f64-pipe results with no write scoreboard and no later f64 op covering them are complete
  128 stall cycles after issue (`F64_LAT`, 2026-11-06). ptxas for sm_103a schedules them as
  fixed latency: each is followed by NOP padding (`st=15` x 3, `st=4`), and the first
  uncovered read in the msl_ops corpus comes 128 counted cycles later at the soonest (f64
  readers; `F2F.BF16.F64` readers 448 or more). Stall counts are minimums, so the counted
  cycles bound the real ones from below. Trusted from ptxas (as L-STALL); the Triton
  layernorm / rmsnorm captures that rely on it equal GB300.
- If wrong: *miss* for a new opcode that needs a scoreboard.

---

## N: numerics

### N-FADD · code
**`FADD` / `add.f32` is IEEE binary32 round-to-nearest-even, no FTZ, canonical NaN
0x7fffffff.**
- Where: `fp.rs` `f32_add_rn`.
- Status: **validated** (r0 bitwise on 10,000 launches). FTZ/directed modes give `unmodeled`.

### N-EQUIV · code
**Equivalence proofs (rung 2e, `ptxv equiv`).** A level is reported PROVEN for every input of
the captured shape when both kernels pass the preconditions:
- an exhaustive `check` with no finding (one output per input, whatever the schedule);
- the taint check (`num.rs` `Taint`) finds no guard, branch condition, address,
  coordinate, descriptor, count or lane index derived from an input buffer, in one
  execution.

  Then that execution's dataflow is the dataflow for every input. Taint flows through
  registers (any source to every destination), memory bytes and TMEM words; an op that reads a
  tainted byte taints what it writes. The canonical forms (`equiv.rs`) then decide the level.
- Where: `num.rs` (`Taint`, origins), `equiv.rs`, `cli/src/equiv.rs`.
- Status: **argued.**
  - L0 holds when the canonical rounding DAGs are identical. An MMA step is canonicalized as
    (accumulator, multiset of products), which is exact only because the MMA model (N-MMA) is
    order-invariant within its K group. An accumulator of constant +0 is identified with "no
    accumulator", which the model also computes the same way.
  - L1 compares exact polynomials, which is exact.
  - L2 and L3 compare first-order error multipliers (sums of unit roundoffs along each
    product's paths, u for a round-to-nearest add, 2u for round-toward-zero, the MMA
    contract's eps per step). This is the Cor 1a bound structure, not the exact error.
  - Origins are tracked for f32 input elements read by loads (N-LEAF), for bf16 / f16
    elements and bf16 / f16 roundings, byte by byte, and for FP8 / ue8m0 bytes that narrow
    tensor-core steps read (N-HALF-LEAF, N-ROUND).
  - Taint also flows through local memory (STL / LDL) and from a cp.async source to its
    destination. HMMA operands and STL values are values, not control. A half-warp LDTM taints
    only the lanes that execute it. A value of any other origin makes the comparison
    "untracked" (inconclusive, not proven).
  - Several outputs over the explored schedules are accepted only when they come from fp32
    atomic adds (N-ATOMIC).
  - L0 up to the sign of zero: an output that is x + (+0) (an FFMA by an exact 1 with a +0
    addend, `num.rs`) has its L0 form compared with x's peeled (`Canon::l0_z`). Round to
    nearest gives x + 0 = x except -0 + 0 = +0, so equal forms give equal bits but for the sign
    of a zero, and |C - R| = 0. The report says so (cuBLAS's alpha / beta epilogue,
    fixtures/c_equiv_mxf8_cublas).
  - FP4 MMA steps (model `u16-f35-live-sfields`) sum 16-element units exactly and truncate
    each: their L0 form is a multiset of units, each a multiset of terms (`Mma::units`), since
    the model is invariant under permuting a unit's terms and whole units, but not terms
    across units.
  - Input buffers are paired by name, with the same dtype and shape; u8 and u16 byte buffers
    (FP8 codes, scale factors) need only the same size, because their leaves are byte offsets
    (`halfin` `half_mem`).
  - Comparisons decided by finiteness (N-FINITE) are not input data; a proof that relies on
    them holds for inputs with finite elements on which no operation overflows. The branch of
    a float atomic max diamond is not a violation (N-FMAX).
  - `LOP3` taint is bitwise (`num.rs` `lop3_taint`): a result bit is input data only where it
    can take both values as the operands' tainted bits vary (and, under a data-dependent guard
    in segment mode, the destination's old bit); the predicate output (result != 0) is not when
    an input-independent result bit is 1. Each result bit is the LUT of the operands' same bits,
    so this is exact. Other instructions read a partly tainted register as tainted. (CUTLASS
    keeps index masks and data-dependent flags in one word, fixtures/c_cve_sdpa_f32 R108;
    `flag_bits_keep_their_own_taint`.)
  - Per-thread taint (2026-11-04), each exact: a `SEL` / `FSEL` whose predicate is
    input-independent carries only the chosen operand's taint (a causal mask's -inf); a
    shuffle's destination the taint of the lane it read; a load's register the taint of its own
    word (a whole-word register) or of its access; a store of 32-bit registers gives each word
    its own register's taint; a float-max atomic's word is input data only if this thread's
    read found tainted bytes. In segment mode a register operation under a guard tainted for
    some members taints (and drops the nodes of) only the members whose own guard is tainted;
    a member whose guard is input-independent and false keeps its register, bitwise taint
    included (warp-collective and uniform operations: every member, as before). Negative tests:
    fixtures/c_equiv_online (`online_bookkeeping_taint_is_exact`: a mask chosen by the data, a
    shuffle from data lanes, a float2's data word keep their guards).
  - Evidence:
    - the demo (fixtures/r2e_tile): a reordered reduction is L1 not L0, a dropped K slice
      fails L1 with a witness that differs on GB300, an `add.rz` epilogue is L1 not L2, and a
      data-dependent branch is flagged;
    - Triton vs CuTeDSL: L0 (fixtures/r2e_gemm). Their outputs were already bitwise equal on
      GB300 for the captured input.
    - MXF8 GEMMs (fixtures/c_equiv_mxf8_cublas, c_equiv_mxf8_triton): CuTeDSL vs cuBLASLt (L0
      up to the sign of zero) and vs Triton tl.dot_scaled (L0) were bitwise equal on 1000
      random GB300 inputs each (16.4M outputs, tools/mx_gpu_check.py); the K-reversed Triton
      kernel (L1) differed in 84% of them.
- If wrong: *false proof*, for a data dependence the taint rules miss (for example through a
  path the model does not lower) or an MMA unit that is not order-invariant within a group.
- Retire: run proven-L0 pairs on random inputs on GB300 (bitwise), as a standing check.

- A captured S-SAME-VALUE exception involving input-derived bytes is not sufficient for
  equivalence over all inputs. The numerical run retains an explicit symbolic-equality
  precondition for such unordered writes. Constant same-value writes remain supported.

### N-LEAF · code
**Input values keep their identity through pure data movement.** A 4-byte load of a word of
an f32 input buffer that no store, copy or atomic of the run has written and that still holds
its initial bits is an `Input` leaf (buffer, element index). Registers, shared and global
memory, `SHFL` (any mode), `MOV`, `IMAD.MOV` (`IMAD d, RZ, RZ, c`), and `SEL` / `FSEL`
without operand modifiers and with a predicate that does not depend on input data (taint)
carry the leaf unchanged. Vector loads give each 32-bit register its own word; every other
register a load writes loses its node (it used to keep a stale one).
- cp.async (`LDGSTS`, 4 / 8 / 16 bytes) gives each shared-memory word the node of its global
  word (or the `Input` leaf, as a load would) when the copy completes; `ldmatrix` without
  `.trans` (`LDSM.16.M88`) moves whole aligned 32-bit words, each register keeping its word's
  node (Triton's layout conversions of f32 values); so does `stmatrix` without `.trans`
  (`STSM.16.M88`, Inductor's persistent RMSNorm). `LDG.E.ENL2.256` (two register quads) is
  two 16-byte vector loads (torch's bfloat16 copy kernel).
- Where: `num.rs` `num_ld`, `num_cp_async`, `num_sel`, `num_put`; `exec.rs` `Op::Shfl`,
  `Op::Sel`, `Op::Fsel`, `Op::Imad`, `Op::LdMatrix`; `equiv.rs` (`NodeKind::Input` -> monomial
  `buf[idx]`).
- Status: **argued.** These operations copy bits (a shuffle's lane choice, a select's
  predicate and an address do not depend on input data: N-EQUIV's taint check), so the value
  is the input's for every input. Evidence: torch.cat vs a Triton cat (`fixtures/c_equiv_cat`)
  is L0, and 4 random-input trials give bitwise equal outputs; cumsum, the row sums and the
  split-K / atomic sums are L1 with leaves read through LDG, LDS, SHFL.UP/DOWN/BFLY and
  IMAD.MOV.
- If wrong: *false proof* (a value moved by an operation this list misreads keeps the wrong
  identity). A write path without a numerics hook (a TMA store into an input buffer) is
  caught only by the initial-bits comparison.

### N-REASSOC · code
**The per-output bound of `ptxv equiv` is fp_noise Cor 1a, rigorous, for each kernel,
combined by Thm 4.** For a monomial (product of input leaves) m with occurrences o along
paths of rounding nodes v, the computed value is sum_o c_o x_m prod_v (1 + delta_v) with
|delta_v| <= eps_v (u = 2^-24 for a nearest add, 2u toward zero, the MMA contract's eps), so
|K(x) - E(x)| <= sum_m |x_m| g_m with g_m = sum_o |c_o| (prod_v (1 + eps_v) - 1). Then
|C - R| <= sum_m |x_m| (g_R,m + g_C,m) <= k u sum_m a_m |x_m|, a_m = sum_o |c_o|, k the largest
(g_R,m + g_C,m) / (a_m u) (rounded up by 2^-40 for its f64 evaluation).
- Requirements: no overflow (an infinite or NaN partial sum is outside the bound); subnormal
  sums of nearest adds are exact (no absolute term); FTZ operations (fp32 atomics) add an
  absolute 3 * 2^-126 per add (`abs_extra`). L1 must hold (same exact polynomial).
- A monomial whose occurrences cancel (coefficient 0, as x in (x + y) - x) keeps its g_m: it
  is still sum_o |c_o| (prod (1 + eps_v) - 1), on either side alone if need be. (Until
  2026-11-04 such monomials were dropped from the bound; `cancelled_monomial_keeps_its_error`:
  x = 1e8, y = 1 gave 2u |y| for an error of 1.)
- Where: `equiv.rs` `Term::grown`, `add_poly_n`, `rigorous`, `OutBound`.
- Status: **argued** (fp_noise THEORY Cor 1a, Thm 4), checked empirically by `--trials`: for
  every pair in `fixtures/c_equiv_sum` and `c_equiv_cumsum` (20 to 200 random inputs, normal
  and uniform [1, 2), random schedules) no |C - R| exceeds its bound (docs/equiv-reassoc.md
  has the tightness).
- If wrong: *false proof* of a tolerance (a too small bound).

Coefficient arithmetic used for equality must be exact in f64. An inexact sum or product
prevents an equality proof from that canonical form. The norm bound
uses a separate polynomial calculation that charges each coefficient residual to its
error budget and marks the result approximate; it cannot establish L1 from rounded
coefficients (`equiv.rs`, `norm.rs`).

### N-EXACT · code
**An f32 product of two narrow-format input leaves (FMUL, no addend), or a bf16 / f16 rounding
of such a product or leaf, is exact for every input when the format holds every value the
operands can take** (`equiv.rs` `ExactSet`, `Side::exact_set`). The finite codes of a narrow
format are N 2^L with |N| < 2^s (s = mantissa bits + 1), L >= 1 - bias - mantissa bits (its
subnormal spacing), |value| < 2^top (top = 1 + the exponent of its largest value). A product
of two such sets is one with s1 + s2, L1 + L2, top1 + top2. A format with p significand bits,
smallest spacing 2^lmin and every p-bit value below 2^tmax finite holds every member if
s <= p, L >= lmin and top <= tmax: a member's highest bit is at most s - 1 above its lowest,
which is at or above both L and its exponent - p + 1. f32 (24, -149, 128), bf16 (8, -133,
128), f16 (11, -24, 16). Such an FMUL gets no rounding unit (N-REASSOC's u) and such a rounding
neither its relative unit nor its subnormal absolute term (N-ROUND).
- e2m1 is (2, -1, 3), e4m3 / ue4m3 (4, -9, 9), ue8m0 (1, -127, 128). e2m1 x ue4m3 = (6, -10, 12)
  fits f32 and bf16: dequantizing NVFP4 to bf16 is exact. e4m3 x ue8m0 = (5, -136, 137) fits
  neither (2^-9 x 2^-127 is below bf16's spacing; 448 x 2^127 overflows): MXF8 dequantization
  is not claimed exact.
- Where: `equiv.rs` (`ExactSet`, the `Fma` and `Round` arms of `node_inner`).
- Status: **argued** (the derivation above), and checked by brute force over every code pair
  of six format pairs (`exact_sets_are_exact`: each product a set claims exact is exact in
  f32, bf16 and f16; e4m3 x ue8m0 is neither claimed nor exact in bf16).
- If wrong: *false proof* of a tolerance (a rounding taken as exact).

### N-ATOMIC · code
**`atom/red.add.f32` (SASS `ATOMG/REDG.E.ADD.F32.FTZ.RN`) is one indivisible
round-to-nearest add with FTZ of operands and result; fp32 atomic adds without a returned
value commute up to the order of each location's adds.**
- Semantics: exec computes old + v with `f32_fma(old, 1, v, RN, ftz)` (N-FALU's FTZ rule).
- Orders: every value a location can end with is the fl-sequential sum of its start value
  and its S added values in some order (fp_noise Cor 4b). In any order a value sits under at
  most S roundings, S - 1 when the start is +0 (the first add to +0 is exact); `equiv` uses
  that worst case for every value, so its bound covers every order, not only the explored
  ones.
- Exploration: `--slices` files such bytes as reductions (`GFoot::red`): two clusters'
  reductions into one byte are not a cross-cluster conflict (any other access of the byte by
  another cluster still is), and the verdict notes the byte and reports the launch as
  nondeterministic. `ptxv equiv` accepts "nondeterministic" when every atomic of the run is
  such an add (no other atomic, no returned value).
- Where: `exec.rs` `Op::Atom`, `num.rs` `num_atom`, `equiv.rs` `NodeKind::AtomAdd`,
  `explore.rs` `explore_sliced`, `machine.rs` `GFoot`, `cli/src/equiv.rs` `prepare`.
- Status: **argued** (PTX ISA: `atom.add.f32` rounds to nearest even and flushes subnormal
  inputs and results to sign-preserving zero). Evidence: `fixtures/c_triton_rowsum_atomic`
  (Triton tl.atomic_add, 4 programs per row): the simulator's outputs over random schedules
  stay within the bound against torch.sum; GB300's captured output is one of the orders
  (docs/equiv-reassoc.md).
- If wrong: *false proof* (an add that is not one rounding) or a missed conflict (an atomic
  add whose value is read by another cluster during the launch is still a read: caught).

- The nondeterminism waiver is limited to straight-line reduction kernels whose global
  writes are only non-returning fp32 atomic adds. The structural check rejects conditional
  branches, other writes, strong loads, shared-memory communication and lane-to-uniform
  transfers. An unrelated fp32 atomic no longer excuses a nondeterministic output. Other
  atomic programs retain their execution precondition (`NumState::atomic_order_only`).

### N-SEG · code
**Segment mode (`equiv.toml` `segments = true`): instruction slices stand for the values the
numerics DAG has no node for.** Products (`FFMA`/`FMUL` round-to-nearest, `-x` modifiers),
negated adds and exact maxima (`FMNMX`, or an `FSETP` + `SEL` choosing between the compare's own
operands) get nodes when they combine values that depend on different tracked nodes; every
other input-derived register is materialized, when a tracked operation consumes it or it is
stored, as a backward slice along the thread's executed instruction trace (`seg.rs`).
- A guard that depends on input data makes the guarded instruction a `Guard` in the slice
  (both outcomes) and removes the node of its destination for every lane; its taint violation
  is excused when the instruction is inside a materialized slice (`ptxv equiv` lists them as
  "relies on: N-SEG"). Branches that depend on input data are still violations, except
  ptxas's `div.rn.f32` sequence (`MUFU.RCP`, `FCHK`, five `FFMA`, `@!P BRA`, a call of
  `*div_rn*slowpath*`, `BSYNC`; for a divisor known at compile time, `FCHK P, a, b` with an
  immediate b and the quotient tail `q0 = a y`, `r = a - b q0`, `q = y r + q0`, the reciprocal
  y refined from constants and possibly hoisted out of the sequence), which is one `DivRn`:
  the correctly rounded quotient on both paths, as PTX defines `div.rn.f32`. Registers the slow path may write (other than the
  quotient) must be dead after it on every thread's trace (`seg::divrn_dead`).
  Fast-path operands, modifiers and reaching definitions must match the division idiom.
  The complete slow helper must match a catalogued compiler sequence in `seg/*_rn_helper.json`
  (each records its fixture source), up to injective register and barrier renaming. All
  constants, guards and internal edges are checked, together with the caller's argument,
  result and return-address wiring; outside entries into its body are refused. The same
  complete-helper requirement applies to reciprocal recognition. Unknown helpers remain
  untracked.
- Also nodes: a product of two tracked nodes, the same one twice included (x * x in sums of
  squares); FFMA2 / FMUL2 / FADD2 lane by lane (two FFMA / FADD on the lanes' registers). A
  vector store (`STS.64`) materializes each word's value as a 4-byte store does.
- A slice's value at other arguments is `SProg::eval`, which applies exec.rs's semantics
  (N-FALU, N-MUFU, N-FCHK) to the slice's instructions; a slice with an instruction outside
  that set is not materialized (the output stays untracked). Also in that set (elementwise
  kernels, docs/equiv-elementwise.md): `F2FP.BF16/F16.F32.PACK_AB`, `PRMT`, FFMA2 / FMUL2 /
  FADD2 per lane (only the lane's registers are its sources), and the bf16 -> f32 widening;
  `PLOP3` (torch's rsqrtf combines its subnormal test with the bounds test through predicated
  `PLOP3`s; exec.rs's truth-table semantics; fixtures/c_cve_addnorm_eager, GB300 bitwise).
- Guard decisions: inside the `new` side of a `Guard`, another instruction under a guard with
  the same predicate value (the same write of the same register) takes its `new` side too, and
  the `old` side its `old` (erf's `@P MUFU.EX2 R; @P FADD R, -R, 1; @P LOP3 S, R, ...`: R is
  defined only where P holds). Exact: the `Guard` selects `new` exactly when the predicate
  holds.
- ptxas's `rcp.rn.f32` sequence is also one `DivRn(1, b)`: an exponent filter (IADD3 / LOP3 /
  ISETP), `@P BRA` to a CALL of the slow path for the filtered inputs, and otherwise MUFU.RCP
  with one Newton step (FFMA, FADD.FTZ, FFMA) (`seg.rs` `rcprn_at`). `rcp_fast_ok` checks, over
  all 2^32 bit patterns and once per process, that the fast path equals the correctly rounded
  1/b for every b the filter sends to it. The
  slow path is PTX's definition, as for div.rn. Only the fast path's and the exact sequence's
  pcs belong to the idiom, so FADDs interleaved with it are still anchored. torch's FA2 kernel
  computes 1/l this way.
- The div.rn sequence is also recognized when the slow-path function writes the quotient
  register itself (torch's vectorized kernels: no `MOV q, res` after the call); the callee's
  complete body and the caller's bindings are checked as above.
- Flag words (2026-11-04, CUTLASS's f32 fmha keeps ex2 range predicates as bits of R108):
  a register is sliced in the bits its reader uses (`Slicer::value_m`). `LOP3` result bit i is
  the LUT of the operands' bits i, so a `LOP3 R = lut(R, imm, imm)` that leaves the demanded
  bits unchanged for both of their values is skipped, executed or not (`lop3_keeps`); a LOP3's
  source is demanded in the bits where some value of the other operands lets it change a
  demanded result bit (every bit for the predicate output, `lop3_demand`); a definition whose
  demanded bits are untainted (bitwise taint) is its value. Bits outside the demand are left
  unspecified and nothing reads them. A value sliced in every bit serves any demand.
  `P2R` (only the predicates of its mask are sources), 32-bit `ISETP` and `VIADD` are slice
  operations (exec.rs's semantics). Tests `flag_word_writes_keep_other_bits`,
  `flag_word_reads_demand_their_bits`, `predicate_save_ops_evaluate`.
- A float atomic max diamond's branch (N-FMAX) is crossed by slices: both sides write no
  register (RZ destinations) and join right after, so every register is the same on either.
- rcp.rn: the exponent filter may be `VIADD y, b, 0x1800000`, and the quotient may be b's or
  the reciprocal estimate's register (`FFMA R0, R2, R0, R2`; b is read at the filter). Every
  register either path may write after the branch, the slow-path callee's included
  (`rcprn_clobbers`; RZ excepted), must be dead after the sequence; until 2026-11-04 only the
  fast path's were checked.
- Where: `seg.rs`, `num.rs` (`num_anchor`, `num_node_mat`, `num_ffma`, `num_fmnmx`,
  `num_fsetp`, `num_sel2`, `taint_post`), `cli/src/equiv.rs` (`prepare`).
- Status: **argued.** The trace is the same for every input because control flow does not
  depend on input data (N-EQUIV); the slice re-executes the trace's instructions on other
  argument values. The div.rn slow path is not simulated (its result is PTX's definition); its
  fast path is (s_alu2 checks `div.rn` against GB300). Evidence: the torch / Triton softmax
  pairs (`fixtures/c_equiv_softmax`): 100 random inputs, every output within its bound.
- If wrong: *false proof* (a slice that does not compute what the kernel does on other
  inputs, or a slow path that is not correctly rounded).

### N-ELEMENTWISE · code
**Elementwise mode (`equiv.toml` `elementwise`, tried in segment mode; docs/equiv-elementwise.md):
an output whose numerics node is a DAG over a single input leaf (slices, `Fma`, `Add`, constant
leaves; for 16-bit outputs one half of a packed word, `F2FP` operand by operand) is that
input element's function for every input,** and evaluating it (`ETree::eval`) on every bit
pattern of the input type gives every value the kernel can produce at that output.
- Preconditions: as for any proof (no race / deadlock, control flow and addresses independent
  of input data except guards inside slices, N-SEG); output i reads the same (buffer, element)
  in both kernels.
- Each kernel against the true function: `elementwise::true_fn` in f64 (`exp`, `tanh` from
  Rust's libm; erf / erfc by a Taylor series below 3 and a continued fraction above, relative
  error < 1e-13 checked against reference values in the unit test; sigmoid / silu / gelu forms
  free of cancellation). Errors are reported as |K - T|, |K - T| / |T| and |K - T| / ulp(T) in
  the output format; inputs whose true value overflows the format are counted apart. The f64
  error of T (< 1e-13 relative) changes reported ulp errors by < 1e-5 ulp.
- Identification (no `function` given): the library function closest to both kernels on ~900
  sample inputs, if within 2^-5 (deviation relative to max(|T|, 2^-10)) and 8x closer than
  any other; else no comparison with a true function.
- Product outputs (docs/compile-vs-eager.md): an output whose node is `Fma{a, b, c: None}` (one
  f32 multiply, round to nearest, no FTZ) of an input leaf y and a DAG over one other leaf x is
  rn(f(x) * y); f is swept as above, and the pair needs the same (buffer, element) y on both
  sides. Equal f bitwise gives equal outputs (the same correctly rounded product); otherwise
  |C - R| <= |y| max|f_C - f_R| + 2^-24 (|C| + |R|) + 2^-149 (each rounding: relative 2^-24
  of its result, or absolute 2^-150 below the normal range). Not checked on the GPU
  (`elementwise_gpu` refuses product outputs).
- 16-bit product outputs bf16(rn(f(x) * y)) (`Round` of such an `Fma`) of 16-bit x and y: both
  kernels are evaluated at every one of the 2^32 (x, y) pairs (`product_sweep`: f from each
  kernel's element function, then `f32_fma` and the 16-bit rounding), so the reported maxima
  (ulps overall and where f and both outputs are normal, |C - R|, bitwise-equal pairs) are
  exact over all inputs.
- Where: `elementwise.rs`, `cli/src/equiv.rs` (`elementwise_pair`, `elementwise_gpu_check`),
  `cli/src/compare.rs`.
- Status: **validated** on GB300 (`elementwise_gpu`, 2026-11-03): for each kernel of the four
  committed pairs (fixtures/c_equiv_elementwise: torch vs Triton tanh-GELU, erf-GELU, SiLU in
  f32, erf-GELU in bf16) every output of 72 launches (73,728 outputs per f32 kernel: every
  bf16 bit pattern widened to f32, then random patterns) and of 64 launches (all 65,536 bf16
  patterns) equals its element function bitwise. The exhaustive sweep itself is only as good
  as the slice semantics (N-SEG, N-FALU, N-MUFU), which these runs exercise.
- If wrong: a wrong difference bound (miss or false alarm) if a slice's semantics differ from
  the hardware's for inputs the GB300 runs did not cover.

### N-HALF-LEAF · code
**A bf16 or f16 input element widened to f32 by byte moves is an exact `Input` leaf**
(`halfin.rs`). Registers loaded from an unwritten bf16 / f16 input (any width, `LDG.E.U16`
zero-extended) carry the provenance of each byte; `MOV`, `PRMT` (default mode), `SHF.L/R.U32`
by whole bytes with a zero high operand, `IMAD` by 2^(8k) plus 0, and `LOP3` with one register
and constant operands (per byte identity or zero, checked bit by bit) propagate it, every other
write drops it. A register holding [0, 0, lo, hi] of one bf16 element, the bf16 -> f32
conversion op of one, or `HADD2.F32 -RZ, h` of one f16 element gets the leaf (allocation,
element index). Each step is a bit-exact function of the bytes, so the value is the element's
for every input.
- Where: `halfin.rs`, `num.rs` (`num_ld`, `taint_post` hooks), `cli/src/equiv.rs` (`prepare`;
  trials and captured bounds draw bf16 inputs).
- Status: **argued** (bit-level identity of the listed instructions under exec.rs's semantics,
  which the GPU-matching fixtures validate); shown on torch `x.sum(-1, dtype=float32)` vs
  `x.float().sum(-1)` for bf16 x (L1, 19u Σ|x|) and the bf16 GELU pair (exhaustive, GB300
  exhaustive).
- If wrong: *untracked*, never a wrong leaf: an unrecognized widening leaves the value without
  a node.
- f16: an f16 element or f16 rounding node widened by `HADD2.F32` (PTX `cvt.f32.f16`) is that
  leaf or node. Their `N` bytes are marked as f16 (`f16_nodes`) and are never taken as a bf16
  operand or a bf16 widening. bf16 and f16 outputs are read element by element; an output
  element holding an input element's two bytes is that element's leaf (`half_out_leaves`).
- A `PRMT` whose other source has no provenance still moves the bytes of rounding nodes
  (`N`), as it moved input bytes.
- **Converted codes** (2026-11-04, `half_unpack`). `cvt.rn.f16x2.<fmt>x2` / `bf16x2` (SASS
  `F2FP.<F16|BF16>.<fmt>.UNPACK_B`, `CvtUnpack`, no `.relu`) of input code bytes: each result
  half is the 16-bit encoding of that code's byte leaf (in the format the cvt reads; e2m1:
  nibble leaves). The conversion is exact (N-CVT-LP, every code measured) and every finite
  narrow value is representable in f16 and bf16 (ue8m0: bf16 only; its f16 form is not taken).
  f16 results mark the leaf `f16_nodes`, so `HADD2.F32` widens them to the leaf itself. A zero
  byte converts to +0 (ue8m0: unknown). `LOP3 x & 0x7f` of an input byte gives `E7` provenance,
  and its e4m3 conversion is the byte's **ue4m3** leaf: e4m3(c & 0x7f) = ue4m3(c) for all 256
  bytes, NaN included (`halfin::tests::ue4m3_is_masked_e4m3`). Shown end to end by
  `fixtures/c_triton_nvf4_dq` (a dequantize-then-bf16 NVFP4 GEMM, = GB300 bitwise), whose
  products are the same monomials a b sa sb as the block-scaled kernels' (docs/equiv-mma.md
  §5).
- **Byte leaves (narrow tensor cores, docs/equiv-mma.md §5).** Bytes of u8 / u16 input buffers
  are elements (`Half::Byte`). An FP8 operand byte that a `kind::f8f6f4` / `mxf8f6f4` step reads
  from shared memory, and a ue8m0 scale byte it reads from TMEM, is an `Input` leaf of that
  byte, decoded in the format the step reads (`byte_leaf`; the format is in the leaf index, so
  one byte read as two formats gives two leaves). A zero byte is +0 for e4m3 / e5m2 and the
  constant 2^-127 for ue8m0. `tcgen05.cp` moves byte provenance into TMEM.
- **FP4 leaves (kinds mxf4 / mxf4nvf4, agent N).** Element j of a row's K = 64 slice is nibble
  j % 2 of byte 16 (j / 32) + (j % 32) / 2 (the layout `complete_mma_lp` reads, which matched
  GB300 in probes/mma_bs and now in r3_cute_mxf4 and the Triton MXF4 kernels, bitwise). Its leaf
  is input nibble 2 byte + nib, decoded as e2m1 (`code_op`, `equiv::leaf_code`). The step's
  contract counts the units that can hold a nonzero product (a structurally zero factor drops
  its product); before, it took ceil(products / 16), which undercounts units with dropped
  products (never reached: FP4 had no provenance, so no product was dropped).
- **Extension (tensor-core operands, docs/equiv-mma.md §1).** Byte provenance is also tracked
  through shared, global, local memory and TMEM (4 bytes per column word). It moves with
  stores and loads of every width, cp.async, TMA load and store, LDSM / LDSM.T, STSM, SHFL,
  IMAD.MOV, tcgen05.cp, LDTM and STTM. A byte is an input element's (`E`), a bf16 rounding
  node's (`N`, N-ROUND) or zero. An MMA operand register whose halves are whole elements gives
  the product's factors: one memoized `Input` node per bf16 element, or the rounding node. An
  output half whose two bytes are one rounding node's is that node. As before, each step is a
  bit-exact move, and anything unrecognized drops the provenance (the product becomes
  untracked).

### N-SOFTMAX · code
**The softmax bound.** An output in the normal form exp(L_i) / sum_j exp(L_j) (`softmax.rs`)
is bounded by the theorem in `softmax.rs` / docs/equiv-reassoc.md, from:
- each exp slice's contract, found by evaluating it on every f32 argument d <= 0 (2^31 - 2^23
  + 2 values) against `f64::exp`: |g(d) - e^d| <= (r0 + r1 |d|) e^d for d >= -87 (e^d >= 2^-126),
  <= a below; each 1/x slice's on every f32 in the divisor's proven range;
- the argument of each exp, an exact linear form provably <= 0 (x - max(X) with x in X,
  max(A) - max(B) with A in B, or -inf) rounded at most once;
- the rounding nodes on each term's path (sums, online softmax's l * exp(m_old - m_new) +
  s), the division (div.rn: u; div.full: the 1/x slice's contract and one rounding);
- the denominator having exactly one term exp(x_j - max(X)) per member of X (so S >= 1).
- Inputs are finite (no NaN or infinity). Products that underflow add 2^-150 absolute each.
- Where: `softmax.rs`, `equiv.rs` (`compare`), sweep cache `fixtures/c_equiv_softmax/sweeps.json`
  (recomputed by `ptxv equiv --no-sweep-cache`).
- Status: **argued** (the proof is in docs/equiv-reassoc.md); the sweeps are exhaustive.
  Evidence: no trial output above its bound (100 random inputs per pair).
- If wrong: *false proof* of a tolerance.

### N-NORM · code
**The norm bound (`ptxv equiv`, layer norm and RMS norm).** An output that is, in segment mode,
a polynomial in the input leaves and one nonlinear scalar rho per row, P(x, rho) =
sum_m c_m x_m rho^e_m (e_m in {0, 1}), with rho^ computed as g(A^) (`norm.rs`):
- A^ is a rounded polynomial of the inputs: a slice's prefix of round-to-nearest FFMA / FADD /
  FMUL and div.rn by a constant (exact for a power of two, else one rounding) is folded into
  it; each kernel's A and output polynomial carry Cor 1a multipliers (N-REASSOC), with rho^ an
  exact leaf in the output's.
- g, the rest of the slice (one argument), approximates a^p, p in {-1/2, -1, 1/2}: found from
  g(4) / g(1) and bounded by evaluating it on every positive normal f32 (2^31 - 2^24 values):
  |g(a) - a^p| <= r0 a^p (`softmax::sweep`, cached in the fixture's sweeps.json).
- The reference's exact polynomials are the pivot. Coefficient differences (Welford's f32
  constant 1/3 against a two-pass mean's 1/128) are charged as errors, |dc_m| |x_m|; outputs
  whose coefficients differ are "L1~" (the same function up to rounded constants), and more
  than 2^-10 relative is refused.
- Theorem (docs/equiv-reassoc.md): with rho* = A_R(x)^p and e_K = (sum_m |x_m| (g^A_K,m +
  |dc^A_m|)) / A_R(x) <= 1/2, |C - R| <= |P1_R(x)| rho* (eta_R + eta_C) + sum_m |x_m| rho*^e_m
  H^e_m (g_R,m + g_C,m + |dc_m|), eta_K the extreme corner of |(1 +- r0)(1 +- e_K)^p - 1|,
  H = 1 + max eta. Compact form for kappa = S_A / A_R <= kappa_max (e <= 2^-10): (rel + kappa_u
  kappa) u |Y1| + add u T(x); kappa <= 1 when A's monomials are squares with positive
  coefficients (RMS norm).
- Requirements: inputs finite; no product underflows (as N-REASSOC); A has a positive constant
  term (eps) and A_R(x) >= 2^-125, so A^ stays a positive normal f32 (the swept domain).
- Where: `norm.rs`, `equiv.rs` (`compare`, `OutBound::norm`), `cli/src/equiv.rs` (trials,
  `PTXV_EQUIV_HW_TRIALS`).
- Status: **argued** (the proof is in docs/equiv-reassoc.md); the sweeps are exhaustive.
  Evidence: fixtures/c_equiv_rmsnorm and c_equiv_layernorm, 200 simulator trials and 2000 GB300
  inputs per pair, no output over its bound (largest fractions 0.33 and 0.40).
- If wrong: *false proof* of a tolerance.

### N-ATTN · code
**The attention bound (`ptxv equiv`): outputs in the weighted normal form
o = sum_j exp(L_j) v_j / sum_j exp(L_j)** (`softmax.rs`, `wquot`), L_j = x_j - max(X) as in
N-SOFTMAX (x_j an input or a computed score) and v_j input leaves:
- Flash form: rcp(S^) * N^ (div.full), N^ a weighted sum of exponentials whose rescalings
  exp(m_old - m_new) telescope (as online softmax). Normalize-first form: a sum of v_j times
  normalized exponentials exp(L_j) / S^ that all divide by one S^ (identical rounding DAGs: a
  structural hash), each term carrying its own division error.
- Each term's computed value is exp(L_j) v_j phi_j, phi_j in [lo e^(-s|L|), up e^(s|L|)]: a
  rounding of a signed sum is a common factor (1 + delta) of its terms; two occurrences of one
  term with opposite signs are refused. Absolute errors are tracked as a0 + av max|v|.
- Theorem (docs/equiv-reassoc.md): |o^ - o| <= K_rel V1 + up r_hi b_up Vmax + K_abs, V1 =
  sum_j p_j |v_j|, Vmax = max_j |v_j|, K_rel = max(up r_hi - 1, 1 - lo r_lo), r in
  [rho_lo / S_hi, rho_up / S_lo], b_up = s ln n e^(1024 s) + n e^-1000; Thm 4 adds the kernels.
- Computed scores (q . k_j) as exp arguments: a node that is a rounded polynomial of the
  inputs is an atom keyed by the hash of its exact polynomial; every node of one kernel with
  that key must have an identical rounding DAG (structural hash), so the atom is one value per
  kernel. Each kernel is bounded against softmax of its own computed scores; with Delta_K =
  max_j |s^_K,j - s_j| <= max_j sum_m |x_m| g_K,m (Cor 1a), p_j(s^) / p_j(s) is in
  [e^(-2 Delta), e^(2 Delta)], so |o^_K - o(s)| <= K_rel e^(2 Delta_K) V1 + ... +
  (e^(2 Delta_K) - 1) V1, V1 at the exact scores.
- Tensor cores (docs/equiv-mma.md §3): an MMA node (N-MMA) adds its accumulator and products,
  each term record widened by (1 +- eps). A Round node (N-ROUND) widens by (1 +- 2^-8). Scores
  from MMA nodes are score atoms, with Delta_K from the MMA multipliers.
- Base 2: exp2 slices have exhaustive contracts (`exp2_contract`, every f32 argument in the
  domain). The atoms are then in log2 units, and a score error Delta counts as ln 2 Delta in
  the natural exponent. A pair with different bases is refused.
- Shifted form: an FFMA s c' - m c' with m = fl(c' max s) = max fl(c' s_k) (rounding to
  nearest is monotone) has the exact argument c' s_j - max_k fl(c' s_k). It differs from
  c'(s_j - max s) by theta <= 2^-8 while every |c' s| <= SHIFT_X = 2^16. Records widen by
  e^(32 theta s), absolute parts by 2^theta. The per-input bound checks the condition
  (`AttnEval::covered`), and the compact form limits Q to it.
- Flash form with fma(slice of 1/S^, N^) (FA2's rcp.rn, N-SEG): `rcp_slice` reduces the slice
  to the row sum and its exhaustive 1/x contract.
- Requirements: as N-SOFTMAX (finite inputs, exp and 1/x contracts from exhaustive sweeps), an
  exact zero times a finite value only where the other factor is provably finite.
- Where: `softmax.rs` (`WSum`, `wquot`, `wbound`, `wsame`, `AttnEval`, `rcp_slice`,
  `exp2_contract`, `shift_factor`), `equiv.rs` (`compare`).
- Status: **argued**. Evidence: fixtures/c_equiv_attn (equiv.toml: scores given; full.toml: q,
  k, v), 200 simulator trials and 2000 GB300 inputs per pair, no output over its bound (largest
  fractions 0.063 and 0.026). fixtures/c_equiv_sdpa (torch FA2 vs Triton tcgen05, bf16): all
  8192 outputs L1, no simulator or GB300 trial over its bound (docs/equiv-mma.md).
- If wrong: *false proof* of a tolerance.

### N-FALU · code
**The SASS f32 ALU forms are IEEE binary32 operations with these measured NVIDIA rules:**
- `FADD`, `FMUL`, `FFMA`, `FFMA2`, `FMUL2`, `FADD2` (lowered as `FFMA2` with b = 1.0, a*1 exact):
  one rounding of the exact result in the
  instruction's mode (`.RZ`, `.RM`, `.RP`, default nearest-even); every NaN result is the
  canonical NaN 0x7fffffff (also for NaN inputs: payloads are not propagated); an exact zero
  sum is -0 only for (-0) + (-0), or for opposite signs under `.RM`. Operand `|x|` clears and
  `-x` flips the sign bit (before the operation; payload bits untouched).
- `.FTZ`: subnormal inputs become zero of their sign; the result is rounded to 24 significant
  bits *as if the exponent were unbounded*, and then flushed to zero of its sign if it is below
  2^-126 in magnitude. (So under `FMUL.FTZ`, (1 - 2^-24)·2^-126 gives 0 while (1 - 2^-46)·2^-126
  rounds up to 2^-126; flushing the exact value or the IEEE-rounded value both disagree with
  GB300.)
- `.SAT` (`FADD`, `FMUL`, `FFMA`): the result is clamped to [+0, 1]; NaN and every negative
  result (-0 included) give +0.
- `FSEL d, -a, |b|, P`: the selected operand's sign bit is flipped / cleared, no other change
  (NaN payloads kept).
- `FMNMX` (`min.f32` / `max.f32`, predicate operand PT = min, !PT = max) and `FMNMX3`: -0 is
  below +0; a NaN operand is ignored, two NaNs give the canonical NaN; `.NAN` (`min.NaN`): any
  NaN operand gives the canonical NaN; `.FTZ` flushes subnormal operands.
  `|x|` operands (PTX `min` / `max` `.abs.f32` with three sources: FMNMX3 |a|, |b|, |c|)
  compare the cleared-sign values under the same rules (probes/fmnmx3_abs, every triple of 16
  edge patterns, GB300 50/50 scrubbed, 2026-11-08).
- `FSETP.<cmp>[.FTZ].<AND|OR|XOR>`: the 16 PTX comparisons (ordered ones false on NaN,
  `U` forms true), combined with the predicate operand.
- Conversions (rung 5, `fixtures/s_cvt`): `F2F.F64.F32` / `.F32.F64` keep a NaN's sign and
  payload (shifted, quiet bit set); `F2F.F32.F64` rounds to nearest; `F2I.S64/.U64.TRUNC`
  saturate and give 2^63 for NaN; `HADD2.F32` (f16 -> f32) gives the canonical NaN;
  f32 -> f16 (`F2F.F16.F32`, `F2FP.F16.F32.PACK_AB`) rounds to nearest even, overflows to
  infinity and gives 0x7fff for NaN. `ATOM.ADD.F32.FTZ.RN` adds as `FADD.FTZ`. PTX
  `div.full.f32` is ptxas's expansion (scale by 1/4 or 2^24, `MUFU.RCP`, `FMUL`;
  `fixtures/s_ptxdiv`, both front-ends). f32 -> bf16 (`F2FP.BF16.F32.PACK_AB`, and
  `F2F.BF16.F32` for `cvt.rn.bf16.f32` into a 16-bit register) rounds to nearest even and gives
  the canonical 0x7fff for every NaN (the model used to keep the payload, quieted:
  `fp::f32_to_bf16_rn_gpu`; host-side torch conversions keep `f32_to_bf16_rn`). PTX
  `div.rn.f32` (ptxas: `FCHK`, `MUFU.RCP` + `FFMA` refinement, slow-path call) is the
  correctly rounded IEEE quotient with the canonical NaN; `cvt.rzi.{u32,s32}.f32` is
  `F2I[.U32].TRUNC.NTZ` (`fixtures/s_ptx_cute_ops`: the 64 special values x random bits,
  both front-ends, GB300 bitwise).
- f64 (rung 5, `fixtures/s_f64`, 147,456 results): `DFMA` / `DMUL` / `DADD` round once in
  their mode (RN / RZ / RM / RP), subnormals kept (no FTZ). Unlike f32, a NaN operand
  propagates: quieted, sign and payload kept, operand modifiers not applied, in the priority
  b, c, a (DADD a, b: b first); an invalid operation gives 0xfff8000000000000. `DSETP` is the
  IEEE comparison. PTX `div.{rn,rz,rm,rp}.f64` / `sqrt.*.f64` are correctly rounded, with the
  same NaN rules (a before b) (`probes/f64_ptx`, PTX front).
- `MUFU.<any>` of a NaN is the canonical NaN (ptxas's `MUFU.RSQ R, -QNAN` NaN idiom); other
  MUFU inputs are N-MUFU's.
- Where: `fp.rs` `f32_fma`, `round_f32_ftz`, `f32_minmax`, `f32_cmp`, `f32_sat`; `exec.rs`
  `Op::Fadd`, `Op::Ffma`, `Op::Ffma2`, `Op::Fmnmx`, `Op::Fsetp`, `Op::Mufu`.
- Status: **validated** bitwise on GB300 by `fixtures/s_alu2`: 128 operands (64 special bit
  patterns: zeros, subnormals, normal edges, infinities, NaN payloads, ties; 64 random) x 128,
  every pair through 44 output words (720,896 words; `FADD.SAT`, `FSEL` with modifiers and
  sub-word stores included), plus three more random seeds (the 40-word version: four)
  (2026-11-03). `cargo test` (`s_alu2_matches_hardware`) checks the model against the hash of
  GB300's output. The FTZ rule was forced by 12 mismatches of the two simpler rules.
- If wrong: *miss* or *false alarm* in numerics and in any value-dependent control flow.

### N-F2I-F64 · code
**f64 -> integer (`F2I[.S64|.U64|.U32].F64[.TRUNC|.FLOOR|.CEIL]`, PTX `cvt.r{z,n,m,p}i.{s,u}{32,64}
.f64`): the f64 rounded to an integer in the mode (none = nearest-even), saturated to the
destination range (infinities to the bounds); NaN gives 0x80000000 for the 32-bit forms (signed
and unsigned, unlike the f32 source's 0) and 2^63 for the 64-bit forms.** Subnormals round like
any value (no flush). Other modifiers on an f64 source (`.FTZ`, `.NTZ`, ...) are lifting
errors.
- Where: `sass` `lower_line` (`F2I` with `.F64`), `ptx` (`cvt`), `exec.rs` (`Op::F2I`,
  `f64: true`).
- Status: **validated** by `fixtures/s_f2i_f64` (authored PTX, both front-ends): 256 operands
  (`gen.py`: signed zeros, subnormals, infinities, NaNs with payloads, ties, 2^31 / 2^32 /
  2^63 / 2^64 and their bit neighbours, every magnitude class, random patterns) through all 16
  forms equal GB300 (100/100 runs, 2026-11-05, `s_f2i_f64_matches_hardware`). The first
  hypothesis (NaN 0 for 32-bit, as for f32) differed in exactly those 48 words.
- Unblocks torch's cross entropy with label smoothing (`reduce_kernel<.., sum_functor<long>>`:
  `F2I.S64.F64.TRUNC`).
- If wrong: wrong values (silent).

### N-FCHK · code
**SASS `FCHK P, a, b` (ptxas's `div.rn.f32` range check) sets P unless 25 <= ea <= 254,
2 <= eb <= 251 and -124 <= ea - eb <= 126** (biased exponent fields; signs and significands do
not matter; zero, subnormal, infinite and NaN operands set it). P selects ptxas's slow
division path.
- Where: `fp.rs` `fchk`; `exec.rs` `Op::Fchk`.
- Status: **validated** on GB300 by `probes/fchk` (the probe's `FSETP` rewritten into
  `FCHK` with a write scoreboard): every pair of the s_alu2 operands (16,384) and of 128
  boundary operands (exponents 1-3, 23-27, 126-129, 150-153, 246-254, ... x four
  significands) matches the rule; no cell depends on the significands. End to end: `div.rn`
  in s_alu2 matches GB300 bitwise (it did not with a guessed range). `cargo test`
  (`probe_fchk_matches_hardware`) runs the patched cubin against the hash of GB300's output.
  `FCHK P, -Ra, Rb` (div.rn of a negated numerator): `probes/fchk_neg` (the same cubin with
  the dividend's negate bit set) gives the same output hash on GB300
  (`probe_fchk_neg_matches_hardware`), 2026-11-06. `FCHK P, |Ra|, |Rb|` (Triton's lerp):
  `probes/fchk_abs` (both abs bits set, instruction bits 73 and 62) gives the same output on
  GB300 (`probe_fchk_abs_matches_hardware`), 2026-11-06. A negated divisor is refused.
- If wrong: a quotient computed on the wrong path; *miss* or *false alarm* in numerics.

### N-BF16X2 · code
**Packed and mixed bf16 arithmetic rounds once, IEEE-like, with GB300's NaN rule.**
- `HFMA2.BF16_V2` (and `HMUL2` = a*b + -0, `HADD2` = a*1 + b): per lane a*b + c of the bf16
  values, one rounding to bf16 nearest-even, subnormal inputs and results kept (gradual
  underflow, quantum 2^-133), overflow to infinity, every NaN result 0x7fff (also `neg.bf16x2`
  of a NaN, ptxas's -0*0 - x); an exact zero sum is -0 only for (-0) + (-0); `.RELU` turns a
  negative result (-0 included) into +0 and keeps 0x7fff. Lane selectors `.H0_H0` / `.H1_H1`,
  `-` operand flips, `-RZ`, immediate pairs (encoding word 32..63, high half printed first).
- `ATOMG` / `REDG .E.ADD.BF16x2.RN` and `atom` / `red .add.noftz .bf16x2 / .v2.bf16 / .bf16`:
  per lane as `HADD2.BF16_V2` (old + v).
- `FHFMA.BF16[.RZ|.RM|.RP][.SAT]` / `FHADD.BF16` (`fma` / `add` / `sub .f32.bf16`): a*b + c
  with a, b the bf16 halves (`.H1` selects the high one) and c an f32, one rounding to f32 in
  the mode, as FFMA (canonical NaN, `.SAT` clamps to [+0, 1]).
- `HMNMX2.BF16_V2[.NAN][.XORSIGN]` (`min` / `max` [.NaN] [.xorsign.abs] `.bf16x2` / `.bf16`):
  FMNMX's rules on the f32 widenings, `.XORSIGN` as N-XORSIGN, every NaN result 0x7fff.
- Numerics: a lane of HMUL2 / HADD2 (or a bf16 atomic add) is an f32 `Fma` / `Add` node and
  a `Round` to bf16: the sum or product of two bf16 values rounded to f32 and then to bf16 is
  the single bf16 rounding (f32 has at least 2 x 8 + 2 bits; a bf16 product is exact in f32
  wherever its bf16 rounding can change), and the f32 intermediate is checked against the
  hardware lane. A general HFMA2 (nonzero addend: the f32 rounding can move a bf16 tie),
  `.RELU` and HMNMX2 lose track (as `num_clobber`); FHFMA / FHADD with round to nearest and
  no `.SAT` are f32 `Fma` / `Add` nodes. The halves move through `halfin` byte provenance.
- Where: `fp.rs` `bf16_fma`, `hfma2_bf16`, `ffma_bf16`, `hmnmx2_bf16`; `exec.rs`
  `Op::Hfma2Bf16`, `Op::FfmaBf16`, `Op::Hmnmx2Bf16`, `AtomOp::AddBf16x2`; `num.rs`
  `bf16_lane_node`, `num_hfma2_bf16`, `num_ffma_bf16`, `num_atom_bf16`; `seg.rs` `eval_inst`.
- Status: **validated** on GB300 (2026-11-06, CUDA 13.0): `probes/bf16x2` (8192 lane pairs:
  every pair of 42 edge patterns, 1600 exact sum / product ties, random; 16 ops including the
  atomics, 327,680 output bytes) and `probes/bf16_ext` (FHFMA / FHADD in all modes and
  `.SAT`, HMNMX2 in every form, 31 x 4096 words) match bitwise on both front-ends
  (`probe_bf16x2_matches_hardware`, `probe_bf16_ext_matches_hardware`); every case was first
  checked against an exact rational model of the rule above.
- If wrong: *miss* or *false alarm* in numerics and in value-dependent control flow.
- Retire: nothing to retire; extend the probe for new modifiers (`.FTZ`, `.SAT`, `.FMZ`,
  `.OOB` on the packed forms are refused).

### N-XORSIGN · code
**`FMNMX[.NAN][.FTZ].XORSIGN d, |a|, |b|, p` (`min` / `max` [.NaN] [.ftz] `.xorsign.abs.f32`)
is FMNMX of |a| and |b| with the sign bit sign(a) XOR sign(b) of the operands as stored**: the
`|x|` nvdisasm prints does not clear the sign the XOR reads (`|UR6|` holding -448 flips the
result's sign), a NaN operand's sign takes part when the other operand is chosen, and a NaN
result is the canonical 0x7fffffff (no sign).
- Where: `fp.rs` `f32_minmax_xorsign`; `exec.rs` `Op::Fmnmx`; `sass` FMNMX lowering (only
  |register| / |uniform register| operands or a non-negative immediate).
- Status: **validated** on GB300 by `probes/fmnmx_xorsign` (every pair of 32 edge patterns,
  8 forms, both front-ends bitwise, `probe_fmnmx_xorsign_matches_hardware`), 2026-11-06.
- If wrong: *miss* or *false alarm* in numerics (Triton's symmetric clamp before an FP8
  conversion: mslk silu_mul_quant, gather_scale_quant).

### N-FCVT · code
**`FRND` and `F2F` between IEEE formats round exactly as IEEE 754 in the instruction's
mode**: `FRND[.F64|.F16|.BF16][.FTZ][.TRUNC|.FLOOR|.CEIL]` (`cvt.r{n,z,m,p}i[.ftz]`) rounds
to an integral value (a zero result keeps the input's sign; `.FTZ`, f32 only, flushes a
subnormal input first); `F2F.{BF16,F16}.F64[.RZ|.RM|.RP]`, `F2F.BF16.F16[.R*]` and
`F2F.F16.BF16[.R*]` round once, with gradual underflow and the mode's overflow (infinity or
the largest finite); `F2F.F64.{BF16,F16}` is exact. NaN results: an f64 source or result keeps
the sign and payload, quieted (the payload's top bits on narrowing, shifted up on widening);
FRND of f32 / f16 / bf16 and the f16 <-> bf16 conversions give the canonical NaN
(0x7fffffff / 0x7fff). `Ra.H1` selects the high half of a 16-bit source; a 16-bit result is
zero-extended in its register (as `F2F.BF16.F32`; not observable from PTX).
- Where: `fp.rs` `fcvt`, `round_ffmt`; `exec.rs` `Op::FCvt`; `sass/src/l2.rs` (FRND, F2F);
  `ptx/src/l2.rs` (`cvt`).
- Status: **validated** on GB300 by `probes/fcvt`: 8192 inputs per format (f32 and f64
  edges, halfway points between adjacent bf16 / f16 values and their f64 neighbours, the
  overflow and subnormal thresholds, NaN payloads, random values; every 8th 16-bit pattern
  plus 64 edges), 34 conversion forms including the `.H1` ones, both front-ends bitwise
  (`probe_fcvt_matches_hardware`), 2026-11-06. The zero-extension of 16-bit results is
  argued. `cvt.bf16.f16` / `cvt.f16.bf16` without a rounding modifier (PX: 5 kernel-DB
  records) are ptxas's F2F.BF16.F16 / F2F.F16.BF16, nearest: probes/cvt_bf16_f16 on every
  16-bit pattern, GB300 20/20, both front-ends bitwise (2026-11-07).
  `cvt.rn` / `.rz [.relu] [.satfinite] .tf32.f32` (SASS `F2FP [.SATFINITE] [.RELU]
  .TF32.F32.PACK_B [.RZ] Rd, Ra`; UMA Nitro) rounds off the low 13 mantissa bits (nearest
  even or toward zero), keeps subnormals, gives 0x7fffe000 for every NaN (`.relu` included),
  +0 for any other negative input under `.relu`, and Inf for Inf or a nearest rounding that
  overflows (`.satfinite`: the largest finite tf32, 0x7f7fe000, with the sign): probes/cvt_tf32_rn,
  all eight forms on 65,536 words (every sign and exponent, mantissas at the rounding edges),
  GB300 20/20, both front-ends bitwise (2026-11-07). Numerics provenance is opaque for it.
- If wrong: *miss* or *false alarm* in numerics (Triton fake-quant rounding, f64
  layer-norm statistics, mslk fp8 quantizers).

### N-F2I16 · code
**`F2I[.FTZ].{S16,U16}[.TRUNC|.FLOOR|.CEIL].NTZ Rd, Ra` (`cvt.r{n,z,m,p}i[.ftz].{s16,u16}.f32`)
rounds as F2I, saturates to the 16-bit range, maps NaN to 0, and leaves the result sign-
(S16) or zero-extended (U16) in the 32-bit register** (ptxas stores that register with a
32-bit STG for the widened value).
- Where: `exec.rs` `Op::F2I` (`bits: 16`); `sass/src/l2.rs`; `ptx/src/l2.rs`.
- Status: **validated** on GB300 by `probes/f2i16` (2048 edge / boundary / random inputs, 16
  forms, the 16-bit and the 32-bit values, both front-ends bitwise,
  `probe_f2i16_matches_hardware`), 2026-11-06. The extension of the `.FTZ` forms is read
  through ptxas's PRMT there, so it is argued from the non-ftz forms.
- If wrong: *miss* or *false alarm* in integer results (Triton's int16 / uint16 stores).

### N-INT16X2 · code
**`VIMNMX.{U16x2,S16x2} d, PT, PT, a, b, sel` and `VIMNMX3.{U16x2,S16x2} d, a, b, c, sel`
(`min` / `max` `.u16x2` / `.s16x2`) take the minimum (sel PT) or maximum (!PT) per 16-bit
lane**, unsigned or signed, lanes packed back in place. `.RELU` (signed forms:
`VIMNMX.RELU`, `VIMNMX.S16x2.RELU`, `min / max.relu.s32 / .s16x2`; also `VIADDMNMX.RELU`, the
min / max of (a + b) mod 2^32 and c) clamps each result at 0. Predicate outputs are refused.
- Where: `exec.rs` `Op::IntMnmx3` (`x16`); `sass/src/l2.rs`; `ptx/src/l2.rs`.
- Status: **validated** on GB300 by `probes/int16x2` (1024 operand triples of edge and random
  halves, register and immediate forms, both front-ends bitwise,
  `probe_int16x2_matches_hardware`) and `probes/imnmx_relu` (the `.RELU` forms on the same
  operands, `probe_imnmx_relu_matches_hardware`), 2026-11-06.
- If wrong: *miss* or *false alarm* in integer results (Triton's bf16 abs-max through u16
  max, nvfp4 quantizers).

### N-BULK-RED · code
**`cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32` (SASS
`UBLKRED.G.S.ADD.F32.RN [URg], [URs], URn`) is the bulk store's asynchronous operation
(issuer's bulk group, URn 16-byte units, shared memory read by the bulk actor) with each f32
added to its global word as an atomic read-modify-write at GPU scope**: round to nearest,
subnormals kept (no flush, unlike `RED.ADD.F32.FTZ`), the canonical NaN 0x7fffffff, an
exact zero sum +0 unless both are -0. Concurrent bulk reductions of one word do not race;
their order is a schedule choice, and errdist records each word as an atomic add (the
summation order of FA4's dQ accumulation).
- Where: `exec.rs` `bulk_red_f32` (from `complete_tma_store`); `num.rs` `num_bulk_add`;
  `sass/src/l2.rs` (UBLKRED); `ptx/src/l2.rs` (`cp.reduce.async.bulk`).
- Status: **validated** on GB300 by `probes/bulk_red` (4096 edge / tie / random pairs; four
  CTAs reducing the same 1024 words, 20/20 runs identical; both front-ends bitwise,
  `probe_bulk_red_matches_hardware`), 2026-11-06. Atomicity per word under contention is
  observed (no lost update in the four-CTA region), not proven; other reduction ops and
  types are refused.
- Tensor reductions, `cp.reduce.async.bulk.tensor.<N>d...add.tile.bulk_group` (SASS
  `UTMAREDG.<N>D.ADD`), add each element in the tensor map's type the same way: f32 as above;
  f16 / bf16 round to nearest even, subnormals kept, NaN 0x7fff (`fp.rs` `add16`); other
  element types are unmodeled at run time. **Validated** by `probes/tma_red` (bf16, f16 and
  f32 tensors, every pair of 64 edge patterns each, both front-ends bitwise,
  `probe_tma_red_matches_hardware`), 2026-11-06. 16-bit sums leave the errdist graph. Two
  16-bit bulk reductions of one word by different asynchronous operations, neither
  happening before the other, are reported `unmodeled` (`bulk-red16-order`): the sum depends
  on their order and the explorer does not branch on it; a single-order "clean" verdict
  once disagreed with the GPU (moe.scatter_add_along_first_dim, duplicate rows; GB300 gives
  92 distinct outputs in 200 runs). They are also not commuting reductions for the
  per-cluster decomposition (only f32 bulk sums are, as N-ATOMIC's).
- If wrong: *miss* (a real race between bulk reductions and other writes is still
  reported; a lost-update hazard between two bulk reductions would not be) or a wrong
  errdist ordering.

### N-MUFU · code
**`MUFU.EX2/RCP/RSQ/LG2/SIN/COS/SQRT/TANH/RCP64H/RSQ64H` (f32, `.F16`, `.BF16`, `.F16x2`, `.BF16x2`) and
the PTX `ex2/rcp/rsqrt/lg2/sin/cos/sqrt/tanh.approx[.ftz]` (and `rcp/rsqrt.approx.ftz.f64`) ops are
modeled bit-exactly.** Each
unit is a 64- or 128-segment quadratic interpolator (truncated squarer, 26/16/10-14-bit
coefficient tables) reconstructed from the hardware's outputs; TANH is the same form per
input binade 2^-8..2^2 (51 segments on an input truncated to 17..21 mantissa bits); each PTX op is modeled as ptxas's expansion around the unit (probes/mufu/README.md).
- Where: `mufu.rs`, `mufu_tables.rs` (generated by probes/mufu/gen_tables.py from
  probes/mufu/fits), `sass` lowering of `MUFU.*`, `ptx` lowering of `*.approx`, `exec.rs`
  `Op::Mufu`; numerics: `num.rs` `NodeKind::Mufu` (error bound per op), `equiv.rs`
  (MUFU pseudo-leaves).
- Status: **validated** (2026-11-03, GB300 GPU 1, CUDA 13.0, driver 580). Every op was run on
  all 2^32 f32 inputs (16-bit forms: all 2^16; packed forms: all 2^32 lane pairs) and
  `ptxv mufu-sweep` found 0 differences from the model: `MUFU.EX2/RCP/RSQ/LG2/SQRT`
  (= the `.ftz` PTX ops), `MUFU.TANH` (= `tanh.approx.f32`), bare `MUFU.SIN/COS` (a patched cubin), `ex2/rcp/rsqrt/lg2/sqrt.approx.f32`
  (denormal expansions), `sin/cos.approx.f32`, `ex2.approx.f16/.ftz.bf16` and their x2
  forms, `tanh.approx.f16/bf16` lanes. Replay without a GPU: fixtures/mufu/samples.json
  (specials, 2^20 random inputs per op, every 16-bit input; the full 2^32 replay is an ignored
  test), fixtures/s_mufu (SASS) and probes/mufu/approx_ptx (PTX front-end) against recorded
  GB300 outputs, and the rung-4 softmax / layer_norm / rms_norm kernels (expected/).
- Error against the true function (f64 reference), maxima over the sweep, used as each MUFU
  node's error bound (|hw - f(x)| <= r ulp(f(x)) + a):
  - EX2 2.056 ulp; RCP 0.975 ulp; RSQ 1.515 ulp; SQRT 0.944 ulp (plus, for the `.ftz` forms,
    a = 2^-126 where a result flushes to 0, and 2^-63 for SQRT of a flushed denormal);
  - LG2 1 ulp + 2.62 * 2^-24 (a fixed-point unit; relative error is unbounded near x = 1);
  - bare SIN/COS (revolutions) 1 ulp + 5.46 * 2^-24; PTX `sin/cos.approx` (radians)
    (8 + 2.7 |x|) * 2^-24 for |x| <= 1024, no useful bound beyond;
  - TANH 134 ulp.
- **f64:** `MUFU.RCP64H/RSQ64H` (= `rcp/rsqrt.approx.ftz.f64`) are modeled bit-exactly: they read
  only the f64 high word, so all 2^32 high words were swept (0 differences); the mantissa is the
  f32 RCP/RSQ unit's, truncated to 20 bits. Bounds (f32 ulps of f(x), any low word): RCP64H
  16.6, RSQ64H 9.1. Replay: fixtures/mufu/samples.json, probes/mufu/f64_ptx (both fronts).
  The f64 sequences built on them (`div.rn.f64`, `sqrt.rn.f64`, `rsqrt.approx.f64`) still fail to
  lift (`DFMA` / `DMUL` / `DADD` / `DSETP` are not lowered; PTX `div/sqrt.rn.f64` neither).
- Other formats: not lowered.
- If wrong: *miss* or *false alarm* in outputs and numerics of kernels using these ops (a
  different GPU, driver or ptxas version may differ: rerun probes/mufu).

### N-MULTIMEM · code
**`multimem.ld_reduce` arithmetic.** Measured on GB300 (`probes/multimem/mm.cu` `ldr`, 4
GPUs, inputs: random bit patterns plus narrow-exponent, subnormal, zero, inf and NaN classes;
65536 16-byte vectors per type; `analyze.py`):
- results are bitwise identical whichever GPU issues the load, across 4 repetitions, across
  processes, and when the member-to-GPU mapping is permuted: the reduction is a symmetric
  function of the members, not schedule-dependent;
- integers (`add` 32 / 64 bit, `min.s32`, `max.u32`, `xor`): exact (262144 lanes each);
- `f32` add (scalar, `.v4`, `.weak`): each operand truncated toward zero to a window of 63 bits
  below the largest operand's last mantissa bit, the truncated values summed exactly, then
  rounded to nearest even, subnormals kept. This reproduces 262094 / 262081 / 262084 of the
  262144 lanes (scalar, `.v4`, `.weak`); every miss has an operand with bits below the window
  (mostly ties of the truncated sum), and the model reports such lanes as `inexact`. No tested window
  (0 to 136 bits) and no window with sticky bits is exact; the plain exact sum misses
  3 to 16 of 4000 sampled lanes; no fixed summation order (24 folds, 3 trees, with or without
  FTZ) fits;
- `f16` / `bf16` `min` / `max`: IEEE minNum / maxNum (NaN operands ignored, -0 < +0), 524288
  lanes each, 0 mismatches;
- NaN in an add: the result is one of the NaN operands, quieted (with two different NaNs not
  always the first; not modeled); inf + -inf: the canonical NaN (`0x7fc00000`, `0x7fc0`);
- f16 / bf16 adds: N-MULTIMEM-ROUND16. `f64`, fp8 (`e4m3`, `e5m2`) and f16 overflow: not
  modeled (`unmodeled`).
- Status: **validated** for the items listed (f32: bitwise where no operand bit is truncated);
  `crates/sim/tests/multimem.rs` replays a sample
  of the recorded lanes (`probes/multimem/results/ldr`).
- Where: `crates/sim/src/multimem.rs`.
- If wrong: wrong reduced values (numerics), not schedules.

### N-MULTIMEM-ROUND16 · code
**f16 / bf16 `multimem.ld_reduce` add rounding.** The result is always one of the two
neighbours of the sum (all random lanes), with or without `.acc::f32`, but not the
nearest-even one in 3% (bf16) / 20% (f16) of random lanes. The direction depends on the other
lanes of the 8-byte group: controlled sweeps (`mm sr3`, `results/sr3`) give "up" exactly when
frac + r >= 1, with frac the sum's position between the neighbours and r the truncated
mantissas of lanes l+1, l+2, l+3 (mod 4 in the group) interleaved most significant bit first;
with every other lane zero the sum is truncated. The rule reproduces 99.7% of non-representable
random bf16 lanes and 96.7% of f16 lanes; misses include groups with a NaN lane.
- Model: `add16` applies the rule and marks the run `inexact`; an output compared with the
  captured GPU bytes then also passes within one unit in the last place per element (reported).
- Status: **unvalidated** (partial fit).
- Where: `crates/sim/src/multimem.rs` `add16`; output comparison in the CLI.
- If wrong: simulator outputs differ from GPU outputs by one ulp in some elements.
- Retire: the exact rule (more sweeps: other bits, groups with special lanes).

### N-MMA · code
**One `tcgen05.mma.kind::f16` (bf16 × bf16 → f32) step is
`g16-f25-rz-accin-tzero-einputs`.**
- Where: `fp.rs` `DEFAULT_MMA`, `exec.rs` `complete_mma`.
- Status: **validated, and enforced at its edges.** It is the unique match of 820 candidates
  on 49,152 GB300 outputs (docs/rung1a.md). That calibration covers finite normal inputs only;
  subnormal/Inf/NaN inputs or accumulators now give `unmodeled`. f16 and tf32 operands: below;
  other formats (f16 D, mixed A / B) are rejected by the idesc check; the narrow kinds are N-MMA-F8 / N-MMA-BS. Shapes (PTX ISA table 48, the
  layouts in S-TMEM / E-PAIR): `cta_group::1` M = 64 or 128 with N = 8..256 in steps of 8
  (M = 128 N % 16 != 0 only with A in shared memory, and with block scaling only N = 8:
  measured forms), `cta_group::2` M = 128 or 256 with N = 16..256 in steps of 16; anything else is
  `unmodeled`. The arithmetic does not depend on N or the layout: `probes/tc_m128` random-operand
  configs (M = 128 N = 8 / 24, pairs N = 32 / 64 / 256, e4m3 for N-MMA-F8) equal the model
  bitwise.
- **scale-input-d** (2026-11-08; `tcgen05.mma ..., enable_input_d, s`, SASS `UTCHMMA ...,
  UPd, s`, s = 1..15, kind::f16 / kind::tf32 only): D = A B + D * 2^-s is the same step with
  the accumulator D * 2^-s, exact. `probes/mma_scale_d` (bf16 and tf32, s = 1, 2, 3, 4, 7, 8,
  9, 12, 15, twice in a row, and with enable-input-d = 0) = the model bitwise on 32,768 GB300
  outputs per input set, 9 random input sets (one committed). Accumulators whose scaled value
  is not a normal f32 or zero are `unmodeled`. GB300 truncates a value that falls below 2^-126
  (toward zero, e.g. 0x807fffff * 2^-1 = 0x803fffff), -0 gives +0, NaN gives 0x7fffffff and Inf
  stays Inf; this was measured on data/C_special inputs (gen.py) but is not modeled. The
  numerics shadow restarts a scaled accumulator cell as a leaf, with a warning when its old
  node carried rounding error (`num_tmem_rescale`). Other kinds with a scale are refused. Both
  front-ends lower it (the PTX front since 2026-11-08); `probes/mma_scale_d` is bitwise on both.
- **f16 and tf32 operands** (2026-11-07): kind::f16 with f16 A / B (instruction descriptor
  format 0) is the same model over 16 products (unique match of 820 candidates, `ptxv
  calib-mma probes/tc_f16`, 6 input sets / 49,152 GB300 outputs; f16 subnormal inputs are
  `unmodeled`); kind::tf32 (format 2; PTX `kind::tf32` = SASS `UTCHMMA` with that descriptor)
  is `g8-f25-rz-accin-tzero-einputs` over 8 products with the low 13 mantissa bits of each f32
  word dropped, as `mma.sync` tf32 (N-HMMA; unique with g16 among 2,460 model x conversion
  candidates, `ptxv calib-mma probes/tc_tf32`, 49,152 outputs). A tf32 K slice is 8 elements of
  4 bytes (16-byte core-matrix rows of 4), A in TMEM one element per column. MN-major tf32
  (idesc bit 15 / 16; KA: MHCRoutingBackwardStream, idesc 0x8088910) is read only through
  smem descriptor layout 1, SWIZZLE_128B_BASE32B: element (r, k) at swz32(start + (r % 32) 4 +
  (k % 4) 128 + (r / 32) LBO + (k / 4) SBO), swz32(a) = a ^ (((a >> 7) & 3) << 5) (4-row
  atoms of 128 bytes, 32-byte granules); GB300 returned all zeros for MN-major tf32 through
  layouts 0 / 2 / 4 / 6, which stay refused, as does layout 1 anywhere else (K-major,
  16-bit, tcgen05.cp, narrow kinds). probes/tc_tf32_mn (A MN-major, then B MN-major; layout
  recovered by solving each D row for the elements read): GB300 20/20, both front-ends
  bitwise (2026-11-07). probes/tc_f16 and probes/tc_tf32 (D[128x64], 4 / 8 MMAs) are
  bitwise = GB300 on both front-ends (50/50 scrubbed), and probes/mma_zero's kind::f16 f16 and
  kind::tf32 configs (signed zeros, subnormal results) now replay bitwise too.
- **Finite overflow** (2026-11-07): a step whose truncated sum is 2^128 or more gives Inf of
  its sign, not FLT_MAX as a literal round toward zero would (`sum_round_terms`, all models).
  probes/tc_mma_ovf (kind::f16 bf16, inputs of magnitude 2^[58, 65)): 1,647 Inf words on
  GB300; the earlier model gave FLT_MAX there under a `clean` verdict (4,938 bytes differed).
  Both front-ends now bitwise, 20/20 scrubbed; HMMA likewise (N-HMMA, probes/hmma_special).
- **Error contract** used by numerics and equivalence (`MmaModel::eps_terms`, docs/equiv-mma.md
  §1). A step adding n terms (products and the accumulator) satisfies |computed - sum| <=
  ((n - 1) 2^-25 + 2^-23 (1 + (n - 1) 2^-25)) sum |terms|, which is about 10u for n = 17. It
  follows from the model: every alignment exponent e has 2^e <= |term|; the term at emax is
  exact (at most 24 significant bits <= 26); each other term loses < 2^(emax - 25); the final
  rounding is toward zero. MMA steps are `NodeKind::Mma` nodes (`num_mma`), whose operands are
  any tracked values (input elements, or bf16 roundings of computed nodes: N-ROUND, N-HALF-LEAF).
  An f32 result below the normal range is rounded toward zero, less than 2^-149 off. The
  canonical forms add that to the absolute part of every MMA step (the relative contract
  alone does not cover it).
- **Signed zeros: every zero result is +0** (validated, `probes/mma_zero`, GB300, CUDA 13.0,
  2026-11-05). It holds whether the sum is exactly zero (all products ±0, also all -0 with
  C = -0; exact cancellation) or a nonzero sum below 2^-149 that rounds to zero (one product
  or a cancellation residue, either sign), with C disabled, C = +0 or C = -0. Subnormal
  results keep their sign. Measured on tcgen05 kind::f16 (bf16 M128 / M64, f16 M128),
  kind::tf32, kind::f8f6f4 (e4m3, e2m1; f32 and f16 D), mma.sync m16n8k16 bf16 / f16 and
  m16n8k8 tf32 (C from registers and RZ). Before 2026-11-05 the models followed IEEE
  (-0 for an underflowing negative sum, or when every term was -0), which differed from
  GB300 on up to 12% of a probe config's outputs (all -0 vs +0) and in the reported case
  (witness: K = 16, one product bf16 0x16b2 x 0x96e5 = -40762 * 2^-178, GB300 +0). Where: `fp.rs`
  `round_int_to_f32` / `sum_round_*`, `mma_lp.rs` `round_int` / `sum_round` / `round_exact` /
  `Fp4Model::step`, `errdist/chain.rs` `step`; `crates/sim/tests/mma_zero.rs` replays the
  probe bitwise (both front-ends for tcgen05, SASS for mma.sync).
- **Overflow is signed infinity, including with RZ mantissa rounding** (validated,
  `probes/mma_overflow`, GB300 SM103, driver 580.126.20, ptxas 13.1.115, 2026-11-06).
  One BF16 product `0x7f00 * 0x4000` is exactly 2^128 and gives `0x7f800000`; the
  negative case gives `0xff800000`. Products are not individually clamped to FP32:
  2^128 - 2^127 gives 2^127, and 2^254 - 2^254 gives +0. Alignment can itself cross the
  overflow boundary: 2^128 - 2^102 gives infinity, while 2^128 - 2^103 gives the largest
  finite float. All inputs and input accumulators in this probe are normal or zero.
  Six configurations cover BF16 tcgen05 M128 N16 K16, BF16 HMMA, and TF32 HMMA, with
  accumulation enabled and disabled. Before this fix, `fp::round_int_to_f32` saturated
  RZ overflow to the largest finite float; the full simulator differed on 95,744 of
  243,712 replayed words. The corrected model matches all eight PTX/SASS replays.
  This does not extend the numerical error bounds to overflowing computations or
  establish the overflow behavior of the separate narrow-format models.

### N-HMMA · code
**One `HMMA.16816.F32.BF16` (`mma.sync.m16n8k16` bf16 × bf16 + f32) is the N-MMA model:
`g16-f25-rz-accin-tzero-einputs`** (all 16 products and the accumulator aligned to the
largest eₐ + e_b, truncated toward zero 25 bits below it, summed exactly, rounded toward zero).
- Where: `fp.rs` `HMMA_MODEL`, `exec.rs` `Op::Hmma` (fragments as in the PTX ISA). The PTX
  front lowers `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` and
  `m16n8k8.row.col.f32.tf32.tf32.f32` to the same op (2026-11-07, SCALE pilot); probes/hmma and
  probes/tmma through it are bitwise equal to GB300 (their kernel.ptx assembled by ptxas: the
  same output, 20/20 scrubbed), and probes/mma_zero now runs every config on both fronts.
  `mma.sync.m16n8k8` bf16 (SASS `HMMA.1688.F32.BF16`, 2026-11-07, SCALE pilot): the same
  model over 8 products; `ptxv calib-mma probes/bmma8 --step 8` (8 accumulating steps,
  6 input sets, 98,304 GB300 outputs) matches exactly `g16-f25-rz-accin-tzero-einputs` (and
  its g8 twin); probes/bmma8 expected/ in cargo test on both fronts.
  `mma.sync.m16n8k16.row.col.f32.f16.f16.f32` (SASS `HMMA.16816.F32`, 2026-11-07, PX: 8
  kernel-DB records): the same fragments and model with f16 elements; `ptxv calib-mma
  probes/hmma_f16` (6 input sets incl. magnitudes 2^[-14, 14] and cancellation, 98,304 GB300
  outputs) matches exactly `g16-f25-rz-accin-tzero-einputs`; probes/hmma_f16 expected/ in
  cargo test on both fronts. f16 subnormal / Inf / NaN elements are refused (not calibrated),
  as for bf16; the 16-bit numerics provenance treats f16 operands as opaque.
- Status: **validated**: `ptxv calib-mma probes/hmma` (one warp per 16x8 tile, K = 64 in 4
  accumulating steps) scores 820 candidates on 98,304 GB300 outputs (6 input sets: normal,
  log-uniform 2^±12 / 2^±30, cancellation, alignment and accumulator probes): exactly this
  model matches all, as for tcgen05 (2026-11-03). `probes/hmma/expected` is checked bitwise in
  `cargo test`. Zero results are +0, as N-MMA (probes/mma_zero, bf16 and tf32, C in registers
  or RZ).
- Special values, bf16 (2026-11-07, `fp.rs` `hmma_bf16_step`): subnormal inputs and
  accumulators are exact and align at exponent -126; subnormal results are kept (rounded
  toward zero); a NaN operand, Inf * 0 or Inf - Inf gives 0x7fffffff, otherwise an Inf gives
  that Inf; a finite sum of 2^128 or more gives Inf of its sign (round-toward-zero to FLT_MAX,
  the earlier model, was wrong there). probes/hmma_special (160 tests, 20,480 outputs: each
  case kind above, subnormal x Inf, alignment against products 2^-26 below a subnormal one):
  GB300 50/50 scrubbed, both front-ends bitwise; of 64 model variants (flush inputs /
  accumulators / results, overflow to FLT_MAX or Inf, true or -126 subnormal exponent) only
  this one matches every output. The numerics drop the node of a result with special operands.
  f16 (HMMA.16816.F32) and tf32 special values stay refused, and so does an f16 result of
  FLT_MAX magnitude or more.
- Numerics: each HMMA is one `NodeKind::Mma` node per output element (`num.rs` `num_hmma`), with
  N-MMA's error contract. Operand halves come from 16-bit provenance (N-HALF-LEAF).
- **TF32: `HMMA.1688.F32.TF32`** (`mma.sync.m16n8k8` tf32, f32 words in the registers) is
  `g8-f25-rz-accin-tzero-einputs` with the low 13 mantissa bits of each operand word dropped
  (truncated, not rounded): all 8 products and the accumulator aligned to the largest
  e_a + e_b, truncated 25 bits below it, summed exactly, rounded toward zero
  (`fp.rs` `TMMA_MODEL`, `tf32_in`; `exec.rs` `hmma_tf32`). Validated: `ptxv calib-mma
  probes/tmma` (one warp per 16x8 tile, K = 64 in 8 accumulating steps, operands with random
  low mantissa bits) scores 2,460 candidates (820 models x truncate / round-to-nearest-even /
  round-half-away input conversion) on 98,304 GB300 outputs (the 6 bf16 calibration sets
  regenerated as f32): only g8 / g16 (the same with K = 8) with truncation match all
  (2026-11-03). `probes/tmma/expected` (GB300, `hw` 1000/1000) is checked in `cargo test`;
  fixtures/c_cve_sdpa_f32 (192 HMMA.1688 per warp) is bitwise = GB300. Subnormal / Inf / NaN
  operands give `unmodeled`. Numerics: each HMMA.1688 is an `Mma` node per output element with
  N-MMA's contract for 9 terms; its factors are TF32 parts of tracked values (N-TF32).
- If wrong: *miss* or *false alarm* in numerics.


### N-IMMA · code
**`IMMA.16832.S8.S8.SAT` (`mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32`) is
D = sat_s32(C + Σ_{k<32} A[m][k] B[k][n]) in exact integers.** Fragments as in the PTX ISA
(g = lane / 4, t = lane % 4): a_j byte i = A[g + 8 (j & 1)][4 t + i + 16 (j >> 1)], b_j byte
i = B[4 t + i + 16 j][g], c_j / d_j = [g + 8 (j >> 1)][2 t + (j & 1)].
- Where: `exec.rs` `imma_s8` (`Op::Hmma { s8: true }`); the PTX front lowers only this
  `mma.sync` integer shape, the SASS front only `IMMA.16832.S8.S8.SAT` (other IMMA modifiers
  are outside modset.rs and refused).
- Status: **validated**: probes/imma_s8 (one warp, 4 chained steps, 512 outputs: random
  bytes, all -128 x -128, -128 x 127, C within 2^19 of both s32 limits; 9 outputs saturate at
  INT32_MAX and 24 at INT32_MIN, later steps start from saturated C) is bitwise equal to
  GB300 on both front-ends (50/50 scrubbed runs, 2026-11-07). On the same inputs a wrapping
  sum differs from GB300 on 87 outputs, a clamp after each product (k order) on 36.

### N-TF32 · code
**TF32 operands and the 3xTF32 split (`num.rs` `NodeKind::Tf32`, `equiv.rs` `tf32_chain`;
docs/equiv-mma.md §6).** For an f32 value x (a node's hardware value):
- hi(x) = x with the low 13 mantissa bits cleared. Nodes: a TF32 MMA operand read from a
  tracked register (the MMA truncates it, N-HMMA), and a `LOP3` whose LUT computes x & 0xffffe000
  bit for bit from one tracked register and input-independent operands. For x zero or normal,
  |hi(x) - x| < 2^-10 |x|: in the canonical forms one rounding node of unit 2^-10.
- lo(x) = x - hi(x): `FADD d, x, -h` (round to nearest, no other modifier) with h = hi of the
  same node. Exact: x and hi(x) share sign and exponent, so the difference is x's dropped bits
  (checked against the result). |lo(x)| < 2^-10 |x| for x normal.
- lo'(x): CUTLASS's round_half_ulp_truncate, the bits of lo(x) plus an input-independent
  0x1000 (`IMAD s, 1, k`, `LEA s, k, 0`), truncated by the MMA. That is lo(x) rounded half
  away from zero to 11 significant bits (a carry into the exponent is the correct rounding):
  |lo' - lo| <= 2^-11 |lo| for lo normal. lo(x) read by the MMA unrounded: <= 2^-10 |lo|.
- A step's products and accumulator follow N-MMA's contract with the TF32 model (`TMMA_MODEL`,
  n = 9 terms, about 6u): its derivation needs significand products of at most 26 bits, and a
  TF32 product has 22.
- 3xTF32: along one MMA accumulator chain (each step's value a term of the next, coefficient
  1), a product hi(a) hi(b) with one lo'(a) hi(b) and one hi(a) lo'(b), a and b equal canonical
  forms, is one term a b (1 + theta):
  hi_a hi_b + lo'_a hi_b + hi_a lo'_b - a b = (lo'_a - lo_a) hi_b + hi_a (lo'_b - lo_b) - lo_a lo_b,
  so |theta| <= 2^-20 + 2^-10 (d_a + d_b) (2^-19 for CUTLASS's split). Below the last of the
  three steps the products are rounded separately: |t1| + |t2| + |t3| <= (1 + w_a + w_b) |a b|,
  w = 2^-10 (1 + d), under those steps' growth h; the term enters at the last step with
  multiplier theta + (1 + w_a + w_b)(h - 1). An unmatched hi * hi product is hi(a) hi(b); an
  unmatched correction is untracked (no proof).
- Requirement: every value read as a TF32 operand or split is 0 or at least 2^-103 in
  magnitude. Then hi and lo are normal or zero (lo is a multiple of x's f32 ulp >= 2^-126), the
  relative bounds hold, and the MMA's operands are in N-HMMA's calibrated domain. Reported as a
  `condition:` of the bound.
- Where: `num.rs` (`num_tf32_hi`, `num_tf32_lo`, `num_tf32_lor`, `tf32_factor`, `num_hmma`),
  `exec.rs` (`Op::Lop3`, `Op::Fadd`, `Op::Imad`, `Op::Lea`, `hmma_tf32`), `equiv.rs`
  (`NodeKind::Tf32`, `tf32_chain`).
- Status: **argued.** theta is checked on 2,000,000 random pairs (exponents 2^-103 and up,
  mantissas at the rounding boundaries; `tf32_split_theta_holds`: within the bound, the worst
  above a quarter of it). Evidence: fixtures/c_equiv_tf32 (3xTF32 and 1xTF32 GEMMs against an
  f32 SIMT GEMM): L1, 239.31u / 183.42u (3xTF32) and 32895.59u (1xTF32) times sum |a_ik b_kj|;
  20 simulator trials each, at most 0.20 / 0.40 of the bound used.
- If wrong: *false proof* of a tolerance.

### N-FINITE · code
**A comparison of a value that finite inputs keep finite with an input-independent infinity or
NaN has the same outcome for every input with finite elements on which no operation
overflows (a taint precondition relaxation, `ptxv equiv`).** `num_fin_setp`: an `FSETP` (any
comparison, modifiers, combining predicate) whose one tainted operand is a node `fin` proves
finite, whose other operand is input-independent and not finite (+-inf, NaN), and whose
combining predicate and guard are input-independent, writes an input-independent predicate (its
taint is cleared). Typical: `FSETP.NE.AND P, PT, |x|, +INF, PT` = isfinite(x) (CUTLASS's TF32
small-part rounding), a comparison with a running maximum's -inf start.
- `fin`: input leaves; constants (leaves of input-independent values) that are finite; sums,
  FMAs, MMA steps (every product tracked), 16-bit roundings, `MUFU.EX2/TANH/SIN/COS` and TF32
  parts of `fin` nodes; a max (min) of `fin` nodes, or of one and a constant other than +inf
  (-inf). Not: untracked values, MUFU.RCP/RSQ/LG2/SQRT (finite arguments can give an
  infinity or NaN).
- Slices (`seg_val`, 2026-11-04): each slice value is abstracted as finite, one constant (an
  infinity allowed), or unknown, for admissible inputs: arguments that are `fin` are finite;
  an operation whose sources are all constants is evaluated (exec.rs's semantics); FADD / FFMA
  (FMUL) / FMNMX / FSEL / SEL / MOV / IMAD.MOV / MUFU.EX2, TANH, SIN, COS of finite values and
  finite constants are finite (FMNMX: or its identity infinity); an FADD of an infinity and a
  finite value is that infinity (exact in every rounding mode; .SAT clamps it to a constant);
  a selection or `Guard` by a constant predicate is the chosen value, otherwise the join; all
  else is unknown, which only predicates may consume. A slice is `fin` when its value is
  finite or a finite constant: online softmax's masked exponential exp2(-inf - mi) is the
  constant 0. A slice without arguments is one value for every input. Test
  `infinity_minus_finite_is_fixed` (negatives: an unknown argument, -inf times a finite value,
  a selection by an unknown predicate).
- An HMMA accumulator operand RZ is the input-independent constant 0 (it was an unmarked
  leaf, so MMA chains starting from zero were not `fin`).
- Argument: take an admissible input (finite elements, no operation of the dataflow DAG
  rounding a finite exact result to an infinity) and the first instruction where its execution
  would leave the captured trace. Up to there every node's value is the DAG's at that input, so
  by induction over the DAG every `fin` node is finite (finite operands give finite results
  unless an operation overflows). Comparing a finite value with an infinity or NaN does not
  depend on the finite value, so a relaxed comparison decides as captured, and every other
  guard, branch and address is untainted: no instruction leaves the trace. Hence the captured
  dataflow is every admissible input's.
- The verdict lists "relies on: N-FINITE (...)": the proof is for admissible inputs only. The
  numerics' bounds already assume them (N-REASSOC: no overflow; N-SOFTMAX: finite inputs).
- Negative tests: fixtures/c_equiv_tf32/neg.toml (a sign test `small > 0` of the split, and
  `isfinite` of the float whose bits are x's shifted left) still report data-dependent guards.
- Where: `num.rs` (`fin`, `num_fin_setp`, `taint_post`), `exec.rs` (`Op::Fsetp`),
  `cli/src/equiv.rs` (`prepare`).
- Status: **argued.**
- If wrong: *false proof* (a relaxed guard that decides otherwise for some admissible input).

### N-FMAX · code
**CUTLASS's float atomic max (`atomicMaxFloat`) is one float maximum whichever side its
branch takes (a taint precondition relaxation, `ptxv equiv`).** `num.rs` `fmax_diamond`
recognizes: `ISETP.GE.AND P, PT, v, RZ, PT` (or `.GT ... v, -0x1`), straight-line (no write
of P or v, no branch target) to `@[!]P BRA T`; the fall-through `ATOMS.MAX.S32 RZ, [a], v` or
`ATOMS.MIN RZ, [a], v` (unsigned), `BRA J`, the other at T, J right after it; the signed max
where v's sign bit is clear, the unsigned min where it is set. `fmax_seq` recognizes the
sequential form (CuTeDSL's `smem_fmax`, FA4 decode): `@P BRA A1; ATOMS.MIN RZ, [a], v; A1:`,
then only `NOP` / `BSYNC` / `BSSY`, then `@!P BRA A2; ATOMS.MAX.S32 RZ, [a], v; A2:`, with the
same sign test on P; exactly one atomic executes. Its data-dependent branches are not
violations; each of its atomics makes the shared word's node `Max(old, v)`.
- Argument: for non-NaN float bit patterns L (the word) and v: if v's sign bit is clear, every
  L with the sign bit set is a negative signed integer below v (and a float below or equal:
  -0 < +0), and non-negative floats order as signed integers; if v's sign bit is set, every L
  with the sign bit clear is an unsigned integer below v and a float above it, and negative
  floats order inversely to their unsigned bits. So either side writes max(L, v) in N-FALU's
  `FMNMX` order (-inf < ... < -0 < +0 < ... < +inf), exactly. Both sides access the same 4
  bytes atomically, with the same footprint (race verdicts unchanged), and the maximum is
  commutative and associative, so any order of the atomics gives the maximum of all values.
  The other input's path changes no register (RZ destinations) and joins immediately.
- Checked per atomic (else a violation): the word's previous value and v are non-NaN for every
  admissible input (an input-independent non-NaN constant, or `fin`, N-FINITE), and the result
  is the float maximum. The proof therefore holds for inputs with finite elements and no
  overflow ("relies on: N-FMAX").
- Negative test: fixtures/c_equiv_fmax `c_fmax_atomic_bad` (the choice made by `v > 0.5`)
  still reports its data-dependent branch.
- The first atomic of a word takes its old value as a constant when this thread's read found
  no input-derived bytes (it was the whole instruction's read: a later lane on another word saw
  an earlier lane's write). Slices cross the diamond (N-SEG): its sides write no register.
- Where: `num.rs` (`fmax_diamond`, `taint_pre`, `num_fmax_atom`), `exec.rs` (`Op::Atom`),
  `cli/src/equiv.rs` (`prepare`).
- Status: **argued.** Evidence: fixtures/c_equiv_fmax (row maxima through atomicMaxFloat into
  shared memory against a sequential fmaxf): L0.
- If wrong: *false proof*.

### N-ROUND · code
**A bf16 (f16) rounding of a computed f32 value is round-to-nearest-even: |rnd(x) - x| <=
2^-8 |x| (2^-11 |x|), plus 2^-134 (2^-25) absolute where the result is subnormal; f16 overflow
is outside the bound.** This covers `F2FP.BF16.F32.PACK_AB`, `F2FP.F16.F32.PACK_AB` and
`F2F.F16.F32` of a computed value (P before P.V in flash attention; PTX `cvt.rn.bf16.f32` /
`cvt.rn.f16.f32`) and the 16-bit rounding of an output before it is stored. These are
`NodeKind::Round { f16 }` nodes. Their bytes carry `N{node, b}` provenance, so the 16-bit
halves can be MMA operands (bf16 only: f16 nodes are marked) or outputs.
- In the canonical forms, the argument's multipliers grow by u16. In the weighted normal form
  (N-ATTN), every term record widens by (1 +- 2^-8). Every rounding node carries its operands'
  absolute parts through, times (1 + eps), and an FMA scales them by |k| when the other factor
  is an exact constant k.
- Outputs that are roundings to one 16-bit format of L1-equal values compare in the 16-bit
  output form (docs/equiv-mma.md §2). With each kernel's Cor 1a bound G_K(x) before the
  rounding, rounding is monotone, so |C - R| <= max(rnd(y + G_C) - rnd(y - G_R), rnd(y + G_R) -
  rnd(y - G_C)) (`round16_gap`: per input, y and G in f64 widened by 2^-49). For every input,
  |C - R| <= 2 u16 |y| + (1 + u16) sum_m |x_m| (g_R,m + g_C,m) + (1 + u16)(A_R + A_C) + 2 a16.
  C and R are equal or adjacent 16-bit values where (G_R + G_C)(1 + h) < h |y|, h = u16 / 2 (the
  spacing at v exceeds u16 |v|). The norm form peels the output rounding the same way. For
  attention, each kernel's bound becomes (1 + 2^-8) bound + 2^-8 V1 + 2^-134.
- Eager's 16-bit roundings between kernels are rounding nodes inside the value before the
  output: the pair is reported as rounding before the output (not the same function), and the
  bound above holds unchanged.
- Where: `num.rs` `num_f2fp16`, `halfin.rs` `half_out_node`, `equiv.rs` (Round canon,
  `compare`, `rnd16`, `round16_gap`), `norm.rs` (`NormBound::rounded`), `softmax.rs` (`wsum`).
- Status: **argued**. The rounding itself is exec.rs's `f32_to_bf16_rn` / `f32_to_f16_rn`.
  The bf16 GELU sweep validates the first bit for bit on all inputs against GB300, and s_alu2
  validates the f16 conversion. `rnd16` (f64) matches both on 2^20 values, and the interval and
  one-ulp claims hold on 400,000 random intervals (`round16_tests`). The bounds are the
  standard ones for round to nearest with 8 (11) significand bits. On the compile-vs-eager bf16
  pairs, GB300's differences reach the exact per-input bound (1.000 of it: one bf16 ulp) and
  never exceed it.
- If wrong: *false proof* of a tolerance.

### N-MMA-F8 · code
**One `tcgen05.mma.kind::f8f6f4` step is `g32-f25-rz-accin-tzero-efields`; with an f16 D it
rounds to nearest instead.** The step works as follows (GB300, sm_103a, 2026-11-03):
1. Take all 32 products of the instruction (K = 32) **and** the accumulator, and find the
   largest alignment exponent. For a product it is e_a + e_b, read from the exponent fields;
   a subnormal operand counts at its format's minimum exponent 1 - bias.
2. Truncate every term toward zero to 25 bits below that exponent.
3. Add the terms exactly and round once: toward zero to f32, or to nearest even to f16.

This is the kind::f16 structure (N-MMA) with one group of 32 instead of 16.
- Operand layout, K-major, in shared memory: each 16-byte core-matrix row holds 16 K elements.
  - FP8: one byte each.
  - FP6: 12 bytes, LSB-first bit stream; 4 bytes unused.
  - FP4: 8 bytes, low nibble first; 8 bytes unused.

  A K = 32 slice is two such rows, LBO apart (no swizzle), or contiguous (swizzled). An f16 D
  takes one TMEM column per element, in its low half, with the high half 0.
- Where: `mma_lp.rs` `DEFAULT_LP_MMA` / `DEFAULT_LP_MMA_F16`; `exec.rs` `complete_mma_lp`.
  Front-ends: PTX `kind::f8f6f4`; SASS `UTCQMMA` (with a sixth `tmem[URs]` operand it is
  block-scaled).
- Status: **validated.** `ptxv calib-mma probes/mma_f8` scores 3,624 candidates. Each
  configuration has 10 input sets of 8,192 outputs, 4 accumulating steps each:
  - sets: normal, uniformly random codes (subnormals included), log-uniform over the full
    range, cancellation, alignment and accumulator probes, a subnormal-alignment probe, f16
    ties below f32 precision, and wide-range first steps (e5m2 / e4m3 via idesc0) that leave
    large accumulators;
  - configurations: e4m3², e5m2², e4m3×e5m2, e2m3², e3m2², e2m1², e4m3×e2m1 (f32 D), e4m3²
    and e2m1² (f16 D).

  Results:
  - every f32 configuration matches exactly this model, or this model and the `einputs`
    variant where no subnormal decides the alignment. e5m2 and e4m3×e5m2 rule out `einputs`
    (4,547 and 2,215 of 8,192 outputs on the subnormal probe) and `eprod`;
  - e4m3² → f16 matches only the RN form (and its `einputs` twin). e2m1² → f16 is consistent
    with it: FP4 products cannot separate the candidates further.

  `probes/mma_f8/expected/` holds 9 of these runs, and `cargo test` checks both front-ends
  bitwise (tests/mma_lp.rs). Zero results are +0, as N-MMA (probes/mma_zero: e4m3 / e2m1,
  f32 and f16 D, C disabled / ±0).
- Not modeled, reported as `unmodeled`:
  - NaN / Inf operands or accumulators, subnormal f32 accumulators, f16 overflow;
  - MN-major FP6 / FP4 operands (8-bit MN-major and A in TMEM: S-TMEM), sparsity, negate.
- Numerics (this kind and N-MMA-BS): each step is a `NodeKind::Mma` node with the exact
  dataflow (exact specs such as `gemm_tn_mx` are checked on it). FP8 operands and ue8m0 scales
  are byte leaves (N-HALF-LEAF), and a block-scaled product is a b sa sb. The error contract
  (f32 D) is derived from the model (`mma_lp::lp_eps`, docs/equiv-mma.md §5):
  |computed - sum t| <= eps sum |t| + 2^-149, eps = tau + 2^-23 (1 + tau). A term that is a
  multiple of Q = 2^(emax - 25) loses nothing, any other less than Q. The setter is such a
  multiple. A normal setter has 2^emax <= |t_k|. A setter with a subnormal operand has
  2^emax <= 2^s |t_k| (s = the subnormal operands' mantissa bits), and `lp_trunc` counts the
  terms that can lose next to it by enumerating the formats' exponents. eps is 17.5u to 130u
  for FP8 (u = 2^-24), 17.5u to 34u for FP6 / FP4 operand pairs. Checked against the model on
  60,000 adversarial steps (error <= the data-dependent bound <= the contract) and reached
  within 0.9 (`mma_lp::contract_tests`). An f16 D is not tracked; FP6 / FP4 operands of this
  kind have no byte provenance (their products are untracked).
- If wrong: *miss* or *false alarm* in numerics on fp8 / fp6 / fp4 GEMMs.

### N-MMA-BS · code
**Block-scaled `tcgen05.mma` (`.block_scale`, kinds mxf8f6f4, mxf4, mxf4nvf4) computes what
`complete_mma_lp` computes** (GB300, sm_103a, CUDA 13.0, 2026-11-03).

Scale-factor placement in TMEM (all measured with `probes/mma_bs`, which writes SFA / SFB with
`tcgen05.st.32x32b` and varies them per lane and column):
- SFA for D row m (sub-partition w = m / 32) is the 32-bit word at lane m, column [sfa] + w.
- SFB for D column n is at lane 32w + n % 32, column [sfb] + n / 32; every sub-partition w reads
  its own copy, so all four must hold the same values (this is what
  `tcgen05.cp.32x128b.warpx4` writes).
- Within the word, the scale for K block b is byte sf_id (1X: one block per instruction),
  byte sf_id + b (2X, sf_id 0 or 2), byte b (4X, sf_id 0). The idesc sf_id fields select the
  bytes.
- The scale TMEM addresses must be 4-column aligned: unaligned ones fault with "misaligned
  address".
- Bits 30-31 of a scale-factor address (the top of the lane field) are ignored. CUTLASS puts
  the sf_id there as well as in the idesc. On GB300, setting them to 1 with idesc sf_id 0
  leaves D bitwise unchanged; idesc sf_id 1 changes it, again regardless of these bits. The
  simulator masks them.
- Illegal instruction on GB300: invalid sf_id for the vector size, mxf4nvf4 2X with ue4m3
  scales, mxf4 with A/B format code 5. The simulator reports these as faults.

Operand placement beyond K-major shared memory (2026-11-08; probes/mma_mxf8_tmem on GB300,
one CTA; probes/mma_bs/ts and mma_bs/mn on GB200, one CTA with M = 128 and cta_group::2 with
M = 256; 8-bit formats):
- A in TMEM: row m at lane m. Word c of the 8 columns at [a] holds elements 4c..4c+3, low byte
  first.
- MN-major SWIZZLE_128B, A or B: K-row k (128 M/N elements of a byte each) at
  (k / 8) SBO + (k % 8) 128, M/N atoms of 128 LBO apart, then the address swizzle.
- With A in TMEM / B MN-major (fa4 fwd P*V, idesc 0x08a10000) and with A MN-major
  (0x08a08000), D on GB300 equals the K-major shared-memory path on the same e4m3 operands
  and ue8m0 scales bitwise (four K = 32 MMAs). Both front-ends reproduce all three tiles
  bitwise.
- MN-major with the other layouts (S-TMEM, probes/mma_bs/mn): swizzled with span s (128 /
  64 / 32 B), element (r, k) at (r % s) + (r / s) LBO + (k % 8) s + (k / 8) SBO, then the
  address swizzle on the absolute address (a start 512 B into a 1024-byte SWIZZLE_128B atom
  reads the absolute phase); without swizzle, (r % 16) + (r / 16) SBO + (k % 8) 16 + (k / 8)
  LBO. cta_group::2: each CTA's A rows from its own TMEM or shared memory and the B rows
  (N/2) i.. (E-PAIR). Every D equals the K-major layout's bit for bit on GB200.
- Refused: these placements with 6- or 4-bit formats, MN-major A in TMEM, swizzled MN-major
  starts off a swizzle-row boundary, M = 64, and M = 128 with cta_group::2 (not probed).

Arithmetic:
- **mxf8f6f4** (scale_vec 1X, ue8m0, K = 32): the N-MMA-F8 model
  `g32-f25-rz-accin-tzero-efields` with terms a·b·sa·sb. The alignment exponent of a product is
  e_a + e_b + e(sa) + e(sb), from the exponent fields.
- **mxf4 / mxf4nvf4** (E2M1, packed 32 per 16-byte row, K = 64; 2X ue8m0 or 4X ue8m0 / ue4m3):
  `u16-f35-live-sfields` (`mma_lp.rs` `Fp4Model`). The steps:
  1. Sum the products of each unit of 16 K elements exactly. FP4 products are short, so the
     sum fits.
  2. Multiply each unit sum by its sa·sb.
  3. Compute emax: the maximum of the accumulator exponent and, over the *live* units only,
     e(sa) + e(sb) from the scale exponent fields. A unit is live if any of its products is
     nonzero.
  4. Truncate each unit and the accumulator toward zero at 2^(emax - 35), add them exactly, and
     round toward zero to f32.

  The alignment depends only on the scales, not on the product magnitudes. A targeted scan
  (two units, one large scale and one small, the small unit's product varied) showed that the
  point where the small unit is dropped depends only on the scale gap and the small product.
- Where: `mma_lp.rs` `DEFAULT_LP_MMA` / `DEFAULT_FP4_MMA`; `exec.rs` `complete_mma_lp`.
  Front-ends: PTX `tcgen05.mma.cta_group::1.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale
  [.scale_vec::NX]`; SASS `UTCQMMA` with a sixth `tmem[URs]` operand (mxf8f6f4), `UTCOMMA`
  (mxf4 and nvf4 2X), and `UTCOMMA.BLOCK16` (nvf4 4X).
- Status: **validated.** `ptxv calib-mma probes/mma_bs`, 4 chained steps × 8,192 outputs per
  set. Sets: normal, random codes, log-uniform, cancellation, alignment / accumulator /
  subnormal / mid probes, a scale gap set (per-block scales spread widely), and for ue4m3 an
  edge set (subnormal and maximum ue4m3 scales).
  - mxf8f6f4 (e4m3², e5m2², e2m1², e4m3×e2m3; 2,416 candidates): the only match is
    `g32-f25-rz-accin-tzero-efields`. Its g64 twin also matches, but with K = 32 it is the same
    function.
  - mxf4 2X ue8m0 (360 candidates): unit 16, frac 35. The live and fields choices cannot be
    told apart here.
  - mxf4nvf4 4X ue8m0: only the live variants match. With all scales instead, probe_sub
    matches 7,354 of 8,192 outputs.
  - mxf4nvf4 4X ue4m3: only the sfields variants match. The sprod variant matches 8,023 of
    8,192 outputs on ue4m3_edge, and unit 8 matches 7,180.

  `probes/mma_bs/expected/` holds 10 of these runs, and `cargo test` checks both front-ends
  bitwise (tests/mma_lp.rs). End to end, `fixtures/r3_cute_mxf8` (the CuTeDSL MXF8 GEMM,
  scales copied in with `tcgen05.cp`) equals GB300 bitwise through both front-ends, as does
  its B30 mutant.
- **cta_group::2** (`UTCOMMA.2CTA.BLOCK16`, 2026-11-04): the arithmetic is unchanged, and each
  CTA's D rows use the scale factors in that CTA's own TMEM, with the one-CTA layout: SFA for
  its rows and SFB for all N columns, so each CTA must hold all of SFB. A / B split as E-PAIR.
  Measured by `probes/tc_pair_bs` (mxf4nvf4 block16 ue4m3, M = 256, N = 64): every operand and
  scale word differs between the two CTAs. "Own SFA and SFB" matches GB300 on all 16,384
  outputs. SFB from the leader, SFB from the CTA that supplies B row n, and SFA from the leader
  each match only the half of the outputs where they coincide with it. Both front-ends equal
  GB300 bitwise (tests/mma_lp.rs). End to end: cuBLASLt's NVFP4 2-CTA kernel
  (`fixtures/c_smm_nvf4`) equals GB300 bitwise on the captured input and on 1,000 random ones
  (through `fixtures/c_equiv_nvf4_cublas`, docs/equiv-mma.md §5).
- Not modeled, reported as `unmodeled` / outside the model: NaN scales; NaN / Inf operands;
  subnormal f32 accumulators (D is f32 only for these kinds); MN-major FP6 / FP4 operands
  (8-bit MN-major and A in TMEM are modeled: S-TMEM, probes/mma_bs/mn and /ts); M = 64.
- Zero results (also scaled sums that underflow) are +0, N-MMA's rule. Unmeasured for the
  block-scaled kinds themselves: probes/mma_zero covers kind::f8f6f4 and the other kinds,
  which share the rule; the code path is the same.
- Error contracts (docs/equiv-mma.md §5): mxf8f6f4 is N-MMA-F8's (one scale pair per step
  scales every product alike). mxf4 / mxf4nvf4 (`mma_lp::fp4_eps`): unit sums are multiples of
  2^(se - 2) (ue8m0) or 2^(se - 8) (ue4m3), so the setter is exact. A live unit has a product
  >= 1/4, so 2^se <= 4 (ue8m0) or 2^8 (ue4m3) times its terms. Up to 3 units and the
  accumulator lose less than Q = 2^(emax - 35): eps = 2.008u / 2.5u. **Requires** every ue4m3
  scale of a live unit to be nonzero: a zero scale sets emax while adding nothing to sum |t|, and
  no relative bound exists (a test truncates an accumulator to 0 that way). Steps with one get
  the loose contract; for inputs with one the contract is not claimed.
- If wrong: *miss* or *false alarm* in numerics on MX / NVFP4 GEMMs; a *false proof* of a
  tolerance if a contract is too small.

### N-CVT-LP · code
**Narrow-format conversions (`cvt` to / from e4m3, e5m2, e2m3, e3m2, e2m1, ue8m0, and the
`.rs` packs) compute what `lowp.rs` computes** (GB300, CUDA 13.0, 2026-11-03):
- Encodes (`F2FP...PACK_AB_MERGE_C`, `...UNPACK_B_MERGE_C` from f16x2 / bf16x2): round to nearest
  even from the exact value (f32 subnormals included); `.satfinite`: overflow and ±Inf give
  ±max. NaN gives the positive NaN code (e4m3 / e5m2 0x7f) or, for FP6 / FP4, which have no
  NaN, the positive maximum (0x1f / 0x7). `.relu` maps negative inputs and -0 to +0. ptxas
  requires `.satfinite` for these targets.
- ue8m0 (`.rz` / `.rp`, f32 or bf16x2): the exponent of |x| (sign ignored), rounded down /
  up; below 2^-127 gives 0. Without `.satfinite`, overflow and Inf give 0xff (NaN); with it,
  0xfe. NaN always gives 0xff.
- Decodes (`F2FP.<F16|BF16>.<fmt>.UNPACK_B`) are exact. NaN gives 0x7fff, and ue8m0 0 gives
  bf16 0x0040 (2^-127). FP6 codes are read from the low 6 bits of each byte (the top 2 bits
  are ignored). `.relu` maps negatives to +0. An e2m1x2 source byte other than byte 0
  (`UNPACK_B Rd, Ra.B1 / .B2 / .B3`, ptxas's lowering of `mov.b32 {c0, c1, c2, c3}` + `cvt` of
  `c1..c3`) reads byte k of Ra; `probes/e2m1_bytes` (2026-11-06) covers every byte value in
  every position, plain and `.relu` (`.relu` also maps -0 to +0).
- `.rs` (stochastic rounding): the random addend sits just below the result's last bit, and
  the result rounds up iff dropped bits + addend carry into it. The addend is 13 bits for
  f16 and 16 bits otherwise. Which bits of the random word feed which lane is in
  `lowp::rs_bits`:
  - f16x2 / bf16x2: the high lane takes r[16..], the low lane r[0..];
  - 8-bit lanes: per `F2FP...RS Rd, Ra, Rb, Rc`, b takes c[16..32) and a takes its 16-bit
    bit reversal;
  - e2m1: b takes c[24..32) · c[8..16), and a takes its bit reversal.

  For x4, ptxas chains two F2FPs, and the second takes the first's merged result as c. So
  the PTX lanes {a, b, e, f} get rev(r.hi), r.hi, rev(r.lo), r.lo for 8-bit formats.
- `MERGE_C`: d = c << (2 lanes) | lanes (8-bit lanes: c << 16; e2m1: c << 8).
- s2f6 (`cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2`, SASS `F2FP.SATFINITE.BF16.S2_6
  .UNPACK_B`; `lowp::cvt_s2f6x2_scaled`, `Op::CvtS2f6x2`): lane i takes byte i of the 16-bit
  code pair and byte i of the 16-bit scale pair; the code is a two's-complement 8-bit fixed
  point with 6 fraction bits (int8 / 64), times 2^(scale - 127); one round to nearest even to
  bf16, subnormals kept, overflow to +-0x7f7f (only -128/64 * 2^127 overflows); scale 0xff
  gives 0x7fff for every code. `probes/s2f6_cvt` (GB300, 2026-11-08): every (code, scale)
  pair of both lanes through register and immediate scales and either data half = the model
  bitwise (`cargo test`), and `run.sh` checks all 2^32 (data, scale) pairs against a C model
  on the GPU: 0 mismatches. PTX ISA 9.1: ptxas 13.1 assembles it, ptxas 13.0 does not. The
  PTX front lowers it to the same op (`a`, `scale` .b16, `hi` = false); `probes/s2f6_cvt`'s
  kernel.ptx is bitwise on the PTX front as well.
- Where: `lowp.rs`; `exec.rs` `Op::CvtPack` / `CvtRsX4` / `CvtUnpack` / `CvtS2f6x2`; PTX
  `cvt_lp`; SASS `f2fp_lp`.
- Status: **validated.** `ptxv cvt-sweep probes/cvt_lp` covers 51 PTX forms, each over every
  input, and finds 0 mismatches:
  - every 16-bit source value (2^16) in both lanes;
  - every 8-bit / 4-bit code pair for the decodes;
  - 2^20 f32 operand sets: every representable value and midpoint of each FP8 / FP6 / FP4
    format ±2 ulps, powers of two ±1 ulp, zeros, subnormals, Inf and NaN in both signs, then
    shaped random values, each with random bits.

  `probes/cvt_lp/sweep.json` holds the per-op hashes of the hardware outputs, and `cargo
  test` recomputes them from the model. `probes/cvt_lp/gb300_out.bin` checks both front-ends
  bitwise on 1,024 sampled threads.
- If wrong: a *miss* or *false alarm* in numerics on narrow-format data. Forms outside the
  probe stay unmodeled: f16 / bf16 `.rn` with `.relu` / `.satfinite`, other SASS F2FP forms,
  and `.H1` sources of `UNPACK_B_MERGE_C`.

### N-BOUND · code
**The Cor 1a a-priori bound uses each rounding node's ε** (u for FADD, the model ε for MMA)
and is first-order.
- Where: `num.rs` `out_rec`.
- Status: **argued** (fp_noise THEORY). Thm 1 itself is checked exactly, not assumed. The
  bound has never been violated, but it is not a proof for inputs outside the run.

### N-ERRDIST-IID · code
**`ptxv errdist` models the inputs exactly as the spec describes them.** Elements are
independent given their hierarchical scales; a block's quantization scale is a function of
its own elements (MX: ue8m0 from the block amax, NVFP4: ue4m3 from amax / 6 times the global
scale). Element laws are pruned below 1e-16 per atom (the pruned mass is reported).
- Where: `errdist/spec.rs` (sampling), `errdist/laws.rs` (element / block laws), `errdist/schema.rs`.
- Status: **argued.** Real activations and weights are not i.i.d.; every estimate is
  conditional on the spec. The block laws match sampling (`laws.rs` tests).
- If wrong: the estimates describe another input population; Tier 3 on captured data (an
  `empirical` / `codes` spec) is the check.

### N-ERRDIST-TEMPLATE · code
**Tiers 0 and 1 analyze output 0's accumulation chain** and report it for every output.
The CLI notes how many distinct chain shapes the program has.
- Where: `cli/src/errdist.rs` `analytic`.
- Status: **argued.** With i.i.d. elements, outputs of the same chain shape have the same law.
  Tier 3 averages over all outputs.
- If wrong: outputs with another shape (edge tiles, split-K) have another law.

### N-ERRDIST-RESIDUE · code
**Tier 1 carries the accumulator's residue explicitly.** A step's local error is τ_acc +
Στ_terms + ρ (truncation at Q = 2^(emax - F), then the round toward zero to ulp(S)). Tier 1
computes the joint law of (the terms' truncated sum mod m = r·4, the dropped part τ) by a
2D FFT over Z_m × τ bins of Q / tsc(r), exactly for the terms (`tier1::steptrans`). The
accumulator enters as zero, as a lattice (its own bits known from the previous step's
grid), or as a fine residue law. Its 2 lowest bits at the result's ulp are tracked in the
state (K_BITS = 2), and its upper bits are taken from the previous step's residue law
conditioned on those low bits (`upper_given_low`). The accumulator's bits above the
tracked ones are independent of the new terms given the cell.
- Where: `errdist/tier1.rs` (`steptrans`, `acc_fine`, `prev_res`, `upper_given_low`);
  `errdist/fine.rs`.
- Status: **argued; validated.** The per-step P(δ != 0) and the low-bit law match
  simulation within 5% on synthetic e4m3 / e5m2 / bf16 chains (normal, Laplace, Student-t;
  K from one step up), and P(differ) matches Tier 3 within 5% (8%: bf16 Student-t /
  Laplace, MX normal recipe 1.06x) on the docs/errdist.md §7 matrix.
- If wrong: biased local error laws. `--uniform` gives the uniform-residue heuristic for
  comparison.

### N-ERRDIST-MARKOV · code
**Tier 1 carries the error as a law conditional on the partial sum's (sign, binade) cell
and its 2 tracked low bits**, and treats the next local error as independent of the past
error given that state. The partial sums' law is a merged atom cloud (moment-preserving
bins); the exact sum's final cell masses are replaced by the characteristic-function
values (N-ERRDIST-CF). Contributions below 1e-13 per transition go to the law's tail
(reported as `model_mass`, part of Tier 1's applicability check).
- Where: `errdist/tier1.rs` `run`; `errdist/laws.rs` `Cloud`.
- Status: **argued; validated** against Tier 3 (docs/errdist.md §7). The remaining bias
  (MX normal recipe, 1.06x) is clustering of the accumulator's bits above the tracked ones.
- If wrong: wrong per-cell error laws. The cell masses are exact (N-ERRDIST-CF).

### N-ERRDIST-CF · code
**The exact sum's (sign, binade) masses come from inverting the characteristic function**,
which is exact: the product of the term laws' φ. The inversion uses Simpson's rule on
[0, 12/σ] with 4096 points, after merging the term atoms into bins of σ/4096 at their means.
The chain's conditional error law within each cell is kept. Skipped when φ does not decay
(lattice-like laws).
- Where: `errdist/cf.rs`, `errdist/markov.rs` `run`.
- Status: **argued.** The test reproduces f(0) of a uniform sum to 2e-4, including its
  Edgeworth term. On bf16 the cancellation cell masses move from 1.5x to within 10% of Tier 3.
- If wrong: biased weights for the cancellation cells, which dominate the 16-bit flip rates.

### N-ERRDIST-CLT · code
**Tier 2 (the DAG CLT) takes every node value as normal (independent leaves, exact
moments), and the node rounding errors and TF32 split errors as independent given the
output y.** Each node's value has a lattice law (P(a multiple of 2^L), from the leaves'
lowest bits; raised to the ulp after its rounding), so its rounding error is a residue
uniform on the lattice grid below the ulp (exact zero, ties to even). A node's value given
y is the regression of a sub-sum on y; the 16 largest node errors given y are convolved
as discrete laws, the rest summed as a zero-inflated normal. MMA steps use the model's
truncation at Q = 2^(emax - F) with the alignment exponent's law from the terms' exponent
fields and the accumulator. TF32 splits: e = h_a d_b + d_a h_b - l_a l_b exactly in the
bilinear moments of (x, lo, d), including its covariance with y. P(differ) integrates over
the exact y's position within its ulp (its own lattice).
- Where: `errdist/tier2.rs` (`dag`, `DagClt::law`, `dag_single`, `dag_pair`).
- Status: **argued; validated on add / FMA / TF32 DAGs** (reductions, split-K, atomics,
  scans, 1x / 3xTF32 GEMMs: within 5% of Tier 3, docs/errdist.md §7) and on long bf16 MMA
  pairs with the correlated round-toward-zero means (K = 960-16384: 1.03-1.14x of Tier 3).
  Normal partial sums fail under heavy-tailed or scale-mixture terms (bf16 outliers 2.6x):
  the tier refuses when y's excess kurtosis exceeds 1 (docs/errdist.md §3.2). The
  block-scaled chain CLT (`tier2::single` / `pair`) is always refused; block-scaled
  chains use `tier2bs` (N-ERRDIST-BSCLT).
- If wrong: Tier 2 is the default headline only where it also agrees with the run's Tier 3
  cross-check (300 inputs); otherwise the headline is Tier 3.

### N-ERRDIST-PAIR · code
**A pair's difference D = C - R comes from both kernels' pre-final deviations, taken as
independent given the exact y's position U within its ulp** (Tier 1: per U bin, each
kernel's post-rounding error is (m - U)·ulp; Tier 2: given y, identical sub-DAGs (equal
canonical hashes) carry the same error in both kernels and drop out of D). Identical
programs are reported as D ≡ 0 by Tier 0 instead.
- Where: `errdist/combine.rs` `pair`; `errdist/tier2.rs` `dag_pair`; `errdist/tier0.rs`.
- Status: **argued; validated on bf16, TF32 and f32-sum pairs** (within 5-8% of Tier 3).
  **Known to be biased for block-scaled pairs that share the K partition in another order**
  (MXF8 / MXF4 Triton-rev: 1.2-1.3x): the kernels truncate the same terms in every step,
  so their deviations are positively correlated. Tier 1 reports itself not applicable
  there.
- If wrong: biased pair estimates. Tier 3 is exact per sample.

### N-ERRDIST-FLIP · code
**With 16-bit outputs, the f32 values a and b round to different bf16 / f16 values with
probability min(1, |a - b| / ulp16)**, where y's position on the 16-bit grid is taken as
uniform. This applies to the single-kernel misrounding and to the pair flip rate in Tier 1.
Tier 0 enumeration computes the misrounding exactly.
- Where: `errdist/combine.rs`.
- Status: **argued.** |D| ≪ ulp16 here, and y's low bits are smooth at that scale.
- If wrong: biased flip rates for discrete inputs (y on a coarse lattice).

### N-ERRDIST-DAG · code
**Tier 3's exact value of a non-chain output is the real-arithmetic evaluation of its DAG**
(`nonlin::real::eval`, double-double): adds, FMAs and MMA steps over leaves, 16-bit roundings
as errors of their argument. A TF32 split's exact value is that of the f32 input: hi(x) stands for x and
the lo parts are errors with exact value 0 (so 1xTF32's operand truncation is part of the
error). Atomic adds are evaluated as round-to-nearest adds in the simulator's order.
- Where: `errdist/nonlin/real.rs` (`Op::Tf32`), `errdist/mc.rs` (`Setup::new`).
- Status: **argued.** The simulator's own atomic order is one admissible order; GB300 runs
  of the same kernels agree on the sampled inputs (docs/errdist.md §7).
- If wrong: another reference value for split / atomic kernels.

### N-ERRDIST-MC · code
**Tier 3 evaluates the numerics DAG on sampled inputs, which is bit-exact per sample**
given N-EQUIV, N-MMA and the simulator / GPU validation. Its confidence intervals take
inputs as the independent unit. For P(differ) this is the wider of Clopper-Pearson over
outputs and a batch-means (Student t) interval over the per-input fractions; for the mean,
batch means per input. Outputs of one input are correlated: a GEMM row, an MX block's
scale. The output-level binomial interval alone was about 3x too narrow on MXF8.
- Where: `errdist/mc.rs` `p_interval`, `estimate`; `errdist/validate.rs`.
- Status: **argued.** Simulator and GPU runs reproduce the DAG's outputs bit for bit (0
  mismatches, docs/errdist.md §7). On MXF8 the batch-means interval matches the spread
  across seeds.
- If wrong: intervals too narrow when inputs are few (the t interval assumes the per-input
  fractions' mean is near normal); the point estimates stay unbiased.

### N-ERRDIST-BSCLT · code
**Tier 2 on a block-scaled MMA chain (MX, NVFP4) takes the partial sums as normal given y,
and the steps' errors as independent given y and y's lattice class** (the lowest set bit
over all terms, against y's ulp). Each step is one alignment truncation at
Q = 2^(emax - F) and one round toward zero (N-MMA-BS); the class fixes every step's lattice
law and y's position within its ulp. The largest step residues stay discrete; the rest and
the truncations are a zero-inflated normal.
- Where: `errdist/tier2bs.rs` (`ChainPre`, `estimate`, `y_kurtosis`); gate in
  `crates/cli/src/errdist.rs` `analytic`.
- Status: **argued; single kernels only.** The tier is refused when y's excess kurtosis
  (from the group-mixture cumulants of the terms) exceeds `tier2::KURT_MAX` = 1, as for the
  DAG CLT: on MXF8, outliers (2.54x of Tier 3, kurtosis 84), ints_scaled (2.35x, 34),
  lognormal, Student t and code specs are refused; normal (0.97x) and Laplace (1.01x)
  pass. Pairs are refused structurally, not run: the pair combination came out 0.77x of
  Tier 3 on an MXF8 reverse-K pair (docs/errdist.md §5.3.1).
- If wrong: an applicable estimate still has to agree with the run's Tier 3 cross-check to
  become the headline (docs/errdist.md §3.2).

### N-ERRDIST-IDENT · code
**An instruction slice computes the real function it was fitted to** (non-chain programs).
Each slice (the instructions between two numerics nodes: an exp range reduction plus
MUFU.EX2, a Newton rsqrt, a polynomial, a division) is fitted on the captured input and 24
perturbed inputs to f(c0 + sum c_i a_i) with f in {exp2, 1/x, 1/sqrt, sqrt, tanh, identity},
a divisor or multiplier form, a / b, a polynomial of degree <= 3, or a named function
(erf, gelu, silu, ...) with an outer affine map. Constants snap to named values (log2 e,
1/128, eps, ...) when that keeps the fit within the slice's error. The slice's own error
against f is then tabulated bit-exactly (N-ERRDIST-NL-INDEP), so a slightly wrong f shows up
as an error law, not as a silent bias.
- Where: `errdist/nonlin/ident.rs` `identify`.
- Status: **argued.** Every slice of the softmax / norm / attention / elementwise fixtures is
  identified with max relative deviation <= 1e-7 (tanh.approx: 2^-9) on its probes. An
  unidentified slice blocks the analytic tiers, and Tier 3 still runs.
- If wrong: the "exact" value of the outputs is another function, and Tier 3 measures the
  error against it.

### N-ERRDIST-REAL · code
**The exact value of a non-chain output is its algorithm in real arithmetic**: every
rounding removed, every identified slice replaced by its real function, and maxima taken on
the real values. The value is evaluated in double-double arithmetic.
- Where: `errdist/nonlin/real.rs`; used by `errdist/mc.rs` `observe` and the analytic tiers.
- Status: **argued.** This defines "C - exact" for softmax, norms and attention. For a
  softmax it is softmax(x) computed exactly, given that the slices are identified.
- If wrong: single-kernel errors are measured against another reference. Pairs (C - R) do
  not depend on it.

### N-ERRDIST-NL-LIN · code
**Non-chain Tiers 1 and 2 propagate local errors to first order** through the DAG's
derivatives (reverse-mode adjoints per output cone) at 48 nominal inputs sampled from the
spec. They then average over the inputs. Tier 1 convolves up to 6 dominant discrete sites
exactly on a 1/64-ulp grid, with the rest normal. Tier 2 is all normal. The output's
position on its grid follows the final op: uniform for continuous values, the lattice
residue for exact sums, the table offsets for final slices.
- Where: `errdist/nonlin/lin.rs`, `errdist/nonlin/agg.rs` `estimate`.
- Status: **argued.** Single kernels: within 1-2% of Tier 3 on softmax, RMSNorm, layer norm
  and attention under the normal spec (docs/errdist.md §9). Elementwise functions are
  poorly linearized; there the elementwise law is used (N-ERRDIST-NL-ELEM).
- If wrong: biased estimates. `best` uses an analytic estimate only when it agrees with Tier
  3 (within max(5%, CI)).

### N-ERRDIST-NL-INDEP · code
**Distinct rounding sites have independent local errors given their cells.** A slice's or
MUFU unit's local error law in a cell (sign, binade, 1/8 binade of its index variable) is
taken as the law for an index uniform over the cell's f32 values. The slice's other
arguments are drawn from its identification probes. Round-to-nearest sites use their exact
lattice residue law, with the parity of the low bit for ties.
- Where: `errdist/nonlin/table.rs`, `errdist/nonlin/lin.rs` `KModel::state`.
- Status: **argued.** Tables are bit-exact evaluations, 96 per cell. Per-class diagnostics
  against bit-exact local errors (`PTXV_NL_DIAG`) agree in mean and variance.
- If wrong: correlated sites (one value rounded twice, a shared exp argument) are mis-weighted.

### N-ERRDIST-NL-JOINT · code
**In a pair, ops of the two kernels holding the same real value share their position on the
rounding grid.** Such pairs are found by equal real values at two inputs (within 2^-20, and
re-checked per input). Their difference is round(w + Z) - round(w), where Z is the
upstream deviation, computed recursively down to depth 16. For two slices, both own errors
cancel when they compute the same function of the index (equal tables, or equal
evaluations), or when the candidate's slice equals, bit for bit, the reference's sub-DAG
above its arguments (fused mean / eps / rsqrt). Otherwise a joint table of the two slices at
equal arguments is used, else they are independent.
- Where: `errdist/nonlin/agg.rs` `Ctx::joint`, `composite_ident`; `table.rs` `pair_table`.
- Status: **argued; known biased.** Pairs come out 3-12% high against Tier 3 on softmax,
  online softmax, RMSNorm and layer norm (docs/errdist.md §9). Reductions in different
  orders are the main gap.
- If wrong: biased pair estimates. `best` falls back to Tier 3.

### N-ERRDIST-NL-ELEM · code
**Elementwise outputs (one to three input elements per output) get their exact law** by
pushing the elements' code law through the bit-exact output function. This is exhaustive
for 16-bit inputs whose product of code counts is at most 4e6 (Tier 0, exact up to
N-ERRDIST-IID). For f32 there are 2^20 codes stratified over the law (Tier 1, CI from the
effective sample size). Larger products use a Latin hypercube with every leaf stratified
(Tier 1, binomial CI). Every output must have the same structure (checked on 64 outputs).
Leaves of a packed 16-bit word that the output does not depend on are checked at 64 points.
- Where: `errdist/nonlin/enumerate.rs` `elementwise`.
- Status: **argued.** Agrees with Tier 3 within its CI on gelu / silu / softcap, silu*mul and
  add (f32 and bf16), single and pairs (docs/errdist.md §9).
- If wrong: the law of another population of elements.

### N-ERRDIST-INTFIX · code
**errdist holds integer input buffers (i8 / i16 / i32 / i64 / u32 / u64) at their captured
values.** They are not taint sources in errdist's numerics run, so control flow on them (a
sequence length, a block count) is fixed rather than a data-dependent violation.
- Where: `cli/src/equiv.rs` `prepare_with` (`int_buffer`), called by `cli/src/errdist.rs`
  `kernel`.
- Status: **argued.** Integer buffers are never DAG leaves (the leaf formats are f32, bf16,
  f16 and the narrow float formats), so Tier 3 never resamples them: it already evaluates
  every input with the captured lengths and indices. The report notes the buffers.
- If wrong: none for the reported distribution, which is conditional on the captured integer
  inputs; another sequence length is another errdist run.

### N-ERRDIST-FINCAP · code
**errdist's extraction run replaces every non-finite element (infinity, NaN) of an f32 / bf16
/ f16 input buffer with +0.** The captured data can hold them (FA4 combine's `lse_partial` is
-inf for a split with no keys); errdist's input distributions are finite, so a path taken on
an infinite element is a path no sample takes. The report notes the count.
- Where: `cli/src/equiv.rs` `finite_inputs`, called from `prepare_with` in errdist mode.
- Status: **argued.** The DAG is extracted on one input and re-evaluated on samples; control
  that depends on input data is still a taint violation unless excused (N-FINITE, N-SEG,
  N-ERRDIST-SEL), now judged on finite values. The race / deadlock exploration still runs on
  the captured data, and so does the comparison with the captured GPU outputs (a plain
  simulator run at the unmodified input stands in for the DAG's values there).
- If wrong: none for finite distributions. A distribution with infinite support points would
  need its own extraction path.

### N-ERRDIST-SEL · code
**A store under a guard computed from input data (`@P STS [x], v`) is the selection
`P ? v : old` in every word it covers.** Segment mode builds `NodeKind::Sel` with the guard's
predicate as an instruction slice of its argument nodes, whichever way the captured input
went, and the guard is then not a taint violation (`relies on: N-ERRDIST-SEL`). Tier 3
evaluates the predicate per input, bit-exactly. The exact value takes the branch the
predicate gives at its arguments' exact values rounded to f32. The linearized and CLT tiers
follow one branch at their nominal inputs, so they are refused on any program with a
selection; the elementwise law re-evaluates it per code and still applies.
- Where: `num.rs` `num_gst_pre` / `num_gst_post`, `errdist/nonlin/ops.rs` `NlOp::Sel`,
  `errdist/nonlin/real.rs`, `cli/src/errdist.rs` (the refusal), `cli/src/equiv.rs`.
- Status: **argued.** A store's address does not depend on input data (the taint check), so
  each covered word holds either the stored value or the previous one; the previous value
  must have a node or be an input-independent constant, else the guard stays a violation.
  On FA4 decode (`fa4.decode_sm100`, `sum > 0 ? 1 / sum : 0`) the DAG agrees with the
  simulator bit for bit on the probe and validation inputs.
- If wrong: the exact reference's branch differs from the kernel's only where rounding moves
  the predicate's arguments across its threshold; Tier 3's C - exact then counts the other
  branch's value as error.

### N-ERRDIST-INERR · code
**An input's error is independent across elements, and the linearized tier pushes it
through to first order.** A spec's per-tensor `error` (relative, absolute, a 16-bit
rounding, a moment form, or joint (value, error) pairs from an upstream errstate) is drawn
independently per element. The linearized tier takes it as the leaf's local law (moments
given the value; for pairs, per value bin) and propagates J e_in + e_own. An errstate
carries the per-element joint law only (`model: "iid"`): the error-error, value-value and
value-error correlations across outputs (same row, other rows) are measured and written as
diagnostics (`corr`), not carried. Richer states (a factor model, low rank) would go in a new
`model`, read by the same `error` field.
- Where: `errdist/spec.rs` `InputError`, `sample`; `errdist/nonlin/lin.rs` `KModel::state`;
  `errdist/errstate.rs`; `errdist/nonlin/agg.rs` `curvature`.
- Status: **argued.** Cross-output correlations measured on matmul, matmul -> RMSNorm,
  matmul -> softmax, attention and MLP outputs (docs/errdist.md §10.3). Error-error and
  value-error |rho| are at most 0.016 (softmax's same-row -0.016 is its sum-to-one
  constraint, -1/(n-1)). The error magnitudes are correlated within a row: squared errors
  0.18 on attention, 0.07 on the MLP, 0.008 on matmul. Composed stage by stage under
  independence, against end-to-end Tier 3, the abs error std is within 5% on matmul ->
  RMSNorm, matmul -> softmax, attention, MLP and a compile-vs-eager silu*mul pair. Hand-written
  upstream errors are within 4%. GB300 agrees with the end-to-end Tier 3 on every region. The tier is refused when the curvature probe's
  second-order effect exceeds 0.1 of the first-order one.
- If wrong: correlated upstream errors (a shared scale, a row-wise statistic feeding every
  element) are mis-weighted downstream. The composed Tier 3 shares the assumption, and only
  an end-to-end run (Tier 3 or GPU) checks it.

---

## H: hardware testing (process)

### H-PROFILE · process
**Two-level verdicts: `ptx_contract` and `sass_gb300`.** When the strict check (the case's
model, PTX or SASS front) reports anything undefined or deviating (a race, a stale read, a
violation, a contract deviation, undefined values, a deadlock, several outcomes; anything
but clean / unmodeled), `ptxv check` checks the variant's SASS front again under a hardware
profile (`--hw-profile gb300`, the default; `none` / `PTXV_HW_PROFILE=none` turns it off)
and reports both levels: `ptx_contract: <classes>` (strict) and `sass_gb300: certified |
clean | incomplete | <classes>`, with the profile assumptions the second level used. The
second level never replaces the first: a contract finding stays a finding of the row.
Profile `gb300` (`crates/sim/src/hwprofile.rs`), every entry probe-backed:
- model switches: E-HW-L1 (non-coherent per-SM L1; GB300 litmus, docs/poll-acq-litmus.md),
  E-MC-ALIAS-HW (multicast / unicast views one location; `probes/multimem` `mm litmus` and
  `probes/l2alias`, validated for multicast write then unicast read and unicast write then
  `multimem.ld_reduce`; write-after-read is not probed and the profile applies it anyway, a
  possible *miss*), E-REMOTE-FIFO (one CTA's remote mbarrier arrivals to another land in
  happens-before issue order; `probes/remote_arrive_exit` `order.cu`);
- discharged contract deviations: E-BULK-EXIT (`probes/tma_store_exit`, `bulk-store-at-exit`),
  E-TC-EXIT (`probes/tc_commit_exit`, `tc-commit-at-exit`), E-DESC-VERSION
  (`probes/desc_version`, `desc-version`), S-TMAP-TAIL (`probes/tmap_tail`,
  `tmap-tail-undef`).
Not in the profile: argued or unvalidated assumptions (E-CPASYNC-PROXY, S-REMOTE-RESERVED,
E-POLL-ACQ (refuted), the twolevel.rs `hw_order` model).
- Evidence: `fixtures/e_twolevel_bulk` (`good`: contract at level 1, certified under the
  profile; `nofence`: race at both levels; `waited`: clean).
- If wrong: a `sass_gb300: certified` is as good as the probes behind the assumptions it
  lists, on the hardware and toolkit they ran on.

### H-RUNS · process
**N clean hardware runs are evidence, not proof.** "GB300 correct 1,000/1,000" bounds a
per-launch failure rate (~3/N at 95%) under the conditions tested: one GPU, idle machine,
default clocks.

### H-SCRUB · process
**`--scrub` poisons the shared memory of every SM before each launch.** It assumes the
scrubber grid lands on every SM (one CTA per SM, max smem).

### H-DELAY · process
**Witness replay inserts `nanosleep` into the PTX,** so ptxas recompiles and may schedule
differently. A replay shows that the predicted behavior is *possible*, not that the original
binary takes that path.

### H-GPU · process
**Experiments run only on GPUs with no foreign compute processes** (0, 1, 3 on the test machine;
the fourth GPU was in use by another job). Results come from GB300 with CUDA 13.0 and driver 580.
