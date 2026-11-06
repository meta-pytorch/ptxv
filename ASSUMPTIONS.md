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
  success is progress, so a peek cannot spin without the mbarrier moving. Other try_waits
  without the retry shape are still blocked on and reported `unmodeled` (fixture r0 B49).
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
  the pending count, so it commutes with the other arrivals and no try_wait can tell);
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
    (M, k+1). So arrivals into the same phase commute.
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
- Where: `crates/sass/src/lib.rs` (`CS2R ... SR_CLOCKLO`), `exec.rs` `CallAbs`.
- Status: **argued.** The timeouts guard against hangs of seconds; a hang the model sees is a
  deadlock finding instead of a trap.
- If wrong: *miss* on code whose results depend on elapsed time (none in the captured stacks:
  the clock only feeds timeout compares).

### E-LOCKSTEP · code
**SASS only: lanes executing together in a converged split are ordered instruction by
instruction** (a memory access by lane i happens before the next instruction of every lane).
PTX lanes stay independent threads.
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
- Where: `exec.rs` `step_split` (lockstep VC join), `Machine::atom_lanes_rev`.
- Status: **argued.** ptxas's own warp-synchronous code (TMEM dealloc bookkeeping) depends on
  it, and without it that code races.
- If wrong: *miss* on warp-synchronous SASS that is racy under independent thread scheduling.
- Retire: a litmus test of intra-warp ordering on GB300.

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
- If wrong: *miss/false alarm* only for collective regions entered by lanes at different
  instructions (unmodeled: they wait).

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
(`{ctaid.x, ctaid.y, 1, 0}`); a cancelled CTA never runs. The response is 16 bytes written to
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
  model allows it; a correct kernel cannot depend on it either way). Multicast responses
  (`UGETNEXTWORKID.BROADCAST`) are parsed but executing one is "unmodeled".
- **Eager launch** (`explore.rs` `clc_eager`): if a deterministic run with every CTA launched
  up front executes no try_cancel (cuBLAS gates its try_cancel on a launch parameter that is
  0), exploration starts with every CTA launched, and any try_cancel that does execute is
  reported "unmodeled". When none executes, launch times do not matter: a late launch is a
  schedule in which the CTA's threads have not stepped yet.

### E-CPASYNC · code
**`cp.async` (SASS `LDGSTS`) reads its global source at issue and writes shared memory when
`cp.async.wait_group n` retires its group** (at most n of the thread's committed groups stay
incomplete; `LDGDEPBAR` commits, `DEPBAR.LE SBk, n` waits). The write is race-checked with the
thread's clock at issue: nothing the thread did after issuing orders it. A source predicate
that is false (src-size 0) writes zeros and reads nothing.
- Where: `exec.rs` `Op::CpAsync` / `CpAsyncCommit` / `CpAsyncWait`, `Thread::cp_pending`;
  `sass` lowering of `LDGSTS`, `LDGDEPBAR` and the `DEPBAR.LE` post-pass; `ptx` lowering of
  `cp.async.{ca,cg}` (src-size 0 or an ignore-src predicate: zero fill; a partial or register
  src-size is rejected), `commit_group`, `wait_group`, and `wait_all` (= commit_group +
  wait_group 0).
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
  clears SBk): one kind, that wait; both with n = 0, both waits (the scoreboard drains); both
  with n > 0, not lowered. Before, any DEPBAR on such an SBk was a bulk wait, the cp.async
  never landed, and the kernel silently read zeros.
- A thread that exits with a cp.async it never waited for (no wait_group / mbarrier arrive
  retired it) is reported (`violation`, `cp-async-exit:<pc>`): its write has no defined
  completion.

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
- If wrong: *miss* of a store reading freed shared memory after exit.
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

### E-DESC-VERSION · code
**A tcgen05 shared-memory matrix descriptor with the fixed bits 46-48 (descriptor version) at 0
behaves on GB300 like one with the documented value 0b001.** `check` reports such descriptors
as a violation (sig `desc-version:<pc>`). `ptxv equiv` accepts them as a precondition and lists
"relies on: E-DESC-VERSION". Triton 3.5's tcgen05 kernels leave the bits at 0. The finding
does not break the E-WELLSYNC certificate. It checks a descriptor value, and the certificate
shows that values are schedule-independent, so every schedule reports the same finding. It
therefore passes through the certificate like a contract finding: it stays reported, and
the certificate still decides the launch. Before, Triton's tcgen05 kernels always fell back
to exploration, at 35 GB per kernel on a 4096 x 4096 x 8192 tile
(`tests/wellsync.rs` `desc_version_finding_does_not_break_the_certificate`).
- Where: `exec.rs` (the descriptor check), `cli/src/equiv.rs` `prepare`, `wellsync.rs` (the
  certificate's findings).
- Status: **validated** for the kernels that use it: the Triton bf16 matmul
  (`fixtures/c_triton_mm_bf16`), run on GB300 on 2000 random inputs, is bitwise equal to
  cuBLAS's and to the simulator's outputs. The attention kernel `r4_attn_triton` matches its
  captured GPU outputs.
- If wrong: *miss* in equivalence (GB300 decoding the descriptor differently from the model).
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
without `--slices`) runs the kernel once (`run_once`'s schedule) with an event log
(`wellsync.rs`) and checks that every synchronization point is *determined*; then that run's
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
    that wait (passed as such), and every register in which they then differ must be dead at
    the meeting point (`walk_dead`). The lanes are walked on from there on a copy whose
    register values are the run's: ALU ops and branches run; memory / sync effects without a
    register result are stepped over (they cannot steer the lanes); a result from memory,
    the TMEM or an unknown input makes its register unknown, and a branch on an unknown
    predicate forks the walk. A differing register read on any path fails, one written
    (guard known true) is dead on that path, and where the walk cannot go on (budget,
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
    only lanes BSSY recorded; a WARPSYNC/BSYNC release waits only for lanes whose exit is
    HB-before all arrivals (an exit does not re-check a warpsync release).
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
- (rung 2) a thread's asynchronous tcgen05 ops are ordered after its own earlier
  instructions (the actor's clock includes the thread's clock at issue);
- (rung 2) `tcgen05.st` is asynchronous like `ld`; `wait::st` completes the stores only for
  the same thread's later tcgen05 ops. Another thread sees them through
  `fence::before_thread_sync`, a synchronization, and `fence::after_thread_sync`.
- SASS `STTM` / `LDTM` are scoreboarded (L-MEMSB) and modeled as the thread's own accesses.
- Where: `exec.rs` `tc_actor`, `pipe_actor`, `push_async`.
- Status: **argued, medium confidence.** It follows the PTX ISA's examples. On GB300 the
  fences are NOPs (B06 is invisible), so hardware cannot confirm the rule.
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
The swizzle phase comes from the shared-memory address, not from the box: a swizzled box may
start at any 128-byte-aligned address (not only at a multiple of 8 rows x span), and element
(q, c) lands at swz(start + q·span + c·esize). A swizzled start that is not 128-byte aligned
faults (reported as `fault`). This is what multicast slices rely on: cuBLAS
`nvjet_sm103_tst_64x8_64x16_2x4_h_bz_TNT` (cluster of 8) has rank 1 multicast rows 4..7 of a
128B-swizzled tile to stage base 0x2400 + 0x200.
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
- **Undefined values propagate through register-only operations** (rung 4): a written R / UR
  register keeps a mask of undefined bits. An ALU operation that reads undefined bits computes
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
- **Loads of never-written shared memory** do not report by themselves: the loaded registers
  become undefined and a later read of them is the finding (`Machine::stale`). cuBLAS loads a
  try_cancel response's ctaid before testing its success bit, and never uses it when the
  request did not run. A multi-register `LDS` (`.64` / `.128`) makes undefined only the
  registers whose bytes nothing wrote: cuBLAS `gemmSN_TN_kernel`'s split-K reduction reads rows
  of 9-word slots with `LDS.128`, the 9th word of each slot padding nothing writes and nothing
  uses (fixtures/c_mm_f32_gemmsn_tn: clean, `hw --scrub` 1000/1000 = simulator with smem
  NaN-poisoned; before, all four registers were undefined and the output store was reported).
- If wrong: *miss* for unchecked operand kinds.

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
  - `PLOP3`: reported only if an output depends on an undefined input (ptxas's
    `PLOP3.LUT P0, PT, PT, PT, PT, 0x8, 0x0` constants do not);
  - `P2R` keeps the undefined bits of the predicates it packs and `R2P` restores them (a
    spilled `PR`);
  - an `ISETP` whose combining predicate is undefined is reported (S-INIT's "decided" rule
    needs a defined combining predicate).
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

### S-DSMEM · code
**A CTA's shared addresses are `(cluster rank << 24) | offset`**, its window inside the
cluster's shared window. A `shared::cluster` access goes to the CTA named by the rank byte. In a
cluster launch, a `shared::cta` access whose rank byte is not the CTA's own is a fault (GB300
traps with *illegal instruction*: `probes/reserved_smem`, variant `cluster_untagged`). `mapa`
replaces the rank byte.
- Evidence: ptxas forms every shared address as `LEA R, SR_CgaCtaId, off, 0x18`, and lowers
  `mapa ..., 0` to `PRMT R, RZ, 0x654, R` (clear the top byte).
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
  in steps of 16 for M = 128 and 256 (PTX ISA table 48); A in TMEM, block scaling and sparsity
  with M = 128 pairs are rejected (not measured).
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
- SASS: `LDG/STG.E.STRONG.<SYS|GPU|SM>` are relaxed strong accesses; `MEMBAR.*.<scope>` is a
  release fence; `CCTL.IVALL` is an acquire fence (ptxas: `ld.acquire` = `LDG.STRONG` +
  `CCTL.IVALL`, `st.release` = `MEMBAR` + `STG.STRONG`, `fence.acq_rel` = `MEMBAR` +
  `CCTL.IVALL`).
- `fence.sc` is treated as `fence.acq_rel`. mbarrier `.release.cta/.cluster` and
  `try_wait.acquire.cluster` scopes are still treated as sufficient.
- Where: `exec.rs` `morally_strong`, `scope_includes`, `sync_publish`, `sync_observe`,
  `acquire_pending`; `ptx` `mem_sync`; `sass` `sass_strong`, `sass_scope`.
- Status: **validated** for `.gpu` vs `.sys` across two GPUs (`fixtures/r1f_pingpong`): B20
  (relaxed poll, no acquire) is a race in both models and **wrong 1000/1000 on GB300** (a stale
  L1 hit on the payload); B19 (release at `.gpu`) is a race in both models and latent on GB300
  (20,000 runs). B56 (`fence.acq_rel.gpu` as the acquire) is a race in PTX and clean in SASS:
  only the release half of the fence is scoped.
- If wrong: *miss* of mbarrier scope bugs (a remote arrive with `.release.cta`).
- Retire: scopes on mbarrier operations.

### S-SMEM-STRONG · code
**SASS `LDS`/`STS` of the CTA's own shared memory are relaxed strong accesses at CTA scope;
an `LDS` whose scoreboard is waited on before the warp issues any later non-ALU
instruction is `ld.acquire.cta`.** This is the hardware-faithful model of the binary. The PTX
front-end keeps the PTX model, where a weak shared access can race.
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
- If wrong: *false alarm* if hardware delivers one thread's remote arrivals in order, or
  keeps a CTA's shared memory until its inbound arrivals land; *miss* if a cluster-scope
  release does not wait for earlier remote arrivals.
- Retire: a GB300 probe that observes a late remote arrival (e.g. into a CTA that a
  co-resident successor reuses).

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

### S-CBANK · code
**Driver constants:** `c[0x0][0x0..0x14]` = block/grid dims, `0x348` = generic param base,
`0x110` = generic address of constant bank 0 itself (cuBLAS reads a tensor map at
`c[0x0][0x110] + 0x6c0`, i.e. cbank 0x6c0 = param offset 0x340; consistent with 0x348),
`0x358` (global descriptor for `desc[URx]`) ignored, `0x36c` = `%is_explicit_cluster` (1 iff
the launch names a cluster shape; `probes/cluster_flag` on GB300; ptxas skips
`barrier.cluster` unless it is 1), `0x37c` = stack pointer 0xfffc (no local memory), `0x2f8` = generic base of the local window
(an unmapped 0x7d00_0000_0000: DeepEP only builds vprintf argument pointers from it), params at
`constant0 size − param size`. Any other driver constant is an error.
- Where: `exec.rs` `Machine::cbank`.
- Status: **validated** by bitwise agreement. Only param offset 0 has been exercised.

### E-BSYNC · exec
**`BSSY Bb` records the lanes executing it; `BSYNC Bb` waits until all of them that have not
exited arrive, then orders them like `bar.warp.sync`** (their clocks are joined). ptxas compiles
`__syncwarp()` after a divergent region to the reconverging `BSYNC` alone (rung 3a: lane 0
copies a tensor-map template to shared memory, then after `BSYNC` all 32 lanes read it).
- Where: `exec.rs` `Op::Bssy` / `Op::Bsync`, `Warp::bmask`; `explore.rs` `bsync_parked`,
  `whole_or_parked`, `local_ample` (BSYNC arrivals and runs beside parked lanes expanded
  alone, see "BSYNC reconvergence" under E-DRF).
- Rung 4 (nested regions, ptxas's division slow path inside a loop body): a lane parked at
  another barrier register (or at a `WARPSYNC`) is not waiting at this `BSYNC`; parts of a
  warp that execute one `BSSY` separately add their lanes to its mask, and a release removes
  the released lanes; lanes already parked at the `BSYNC` take part in it. (Before, a release
  counted lanes parked at any convergence barrier, and a second part's `BSSY` overwrote the
  mask: s_alu2 deadlocked and faulted in some schedules.) `BREAK Bb` (`BREAK.RELIABLE`, torch's
  elementwise loops leaving a region early) removes the executing lanes from Bb's mask, and
  lanes already parked at Bb's `BSYNC` are released if the rest of the mask is now there.
- A lane's `EXIT` re-checks the lanes of its warp parked at a plain `WARPSYNC` or a `BSYNC`
  (`exit_warpsync_release`): exited lanes do not take part. Before 2026-11-05 only
  `WARPSYNC.COLLECTIVE.ALL` was re-checked, so lanes that parked at `WARPSYNC.ALL` before
  their warp-mates' `EXIT` stayed parked (a DPOR-only false deadlock on e_aligned_bar AB3,
  SASS; GB300 completes, probes/bar_exit_warp `half`).
- Status: **argued** from the ISA's reconvergence semantics and the CUTLASS source
  (`__syncwarp` there); not probed. Before this, BSSY/BSYNC were no-ops, which reported a race
  on that pattern. s_alu2's divergent division paths match GB300 bitwise.

### S-TMAP · code
**A TMA operand that is not a declared tensor-map parameter is a 128-byte `CUtensorMap` in
global memory**, decoded with the layout in docs/tensormap.md (reverse-engineered from
`cuTensorMapEncodeTiled`, 3097/3097 random encodings, driver 580). The thread issuing the TMA
must have executed `fence.proxy.tensormap::generic.acquire` (SASS `CCTL.E.C.LDCU.IV.DEEP` +
`UTMACCTL.IV`) on that address since its contents last changed, or another thread of its CTA
must have fenced the same bytes and that fence must happen-before the TMA (rung 4b: the
CUTLASS 4.x grouped GEMM's tensormap-updater warps update and fence each map, then hand it
to the load and epilogue warps through an mbarrier pipeline, `IsTensorMapUpdateAsync`);
otherwise a violation is reported (the TMA unit may use a stale cached copy). The descriptor bytes are read as an
ordinary load by that thread at the fence and at the TMA (race-checked).
- Where: `exec.rs` `tmap_at`, `Op::TmapFence`; `tmap.rs`.
- Status: **validated** for the layout (encoder sweep); the fence rule follows the PTX memory
  model (proxy fences are per thread). The meaning of descriptor bits 1 and 21 of word 8 is
  unknown (reported, not used).

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
  front-ends bitwise). N % 16 != 0 with A in TMEM or block scaling is rejected (not measured).
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
  both front-ends = GB300). **A in TMEM**
  (`tcgen05.mma [d], [a], ...`): A row i at the lane D's row i uses, relative to A's lane
  field, two bf16 per 32-bit column, the even K index in the low half. **MN-major** shared
  operands (swizzled): CUTLASS's canonical ((T,s/16,m),(8,k)) : ((1,T,LBO),(s/2,SBO)).
  Without swizzle (kind::f16; the mslk decode kernels' A): ((T,1,m),(8,k)) : ((1,T,SBO),(1T,LBO)),
  i.e. element (r, k) at (r/8) SBO + (r%8) 2 + (k/8) LBO + (k%8) 16 bytes: the roles of LBO and
  SBO swap relative to the swizzled forms. `probes/tc_m128` `mma_img` (raw shared-memory images):
  A, B and both MN-major, two LBO / SBO arrangements each, M = 128 / 64, N = 8..32, decode
  correctly and equal both front-ends bitwise.
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

### L-NVDISASM · code
**`nvdisasm` (CUDA 13.0) prints what the bits encode.**
- Where: `sass` `load_cubin`, `parse_disasm`.
- Status: **trusted** (vendor tool). Cross-checked indirectly by every bitwise hardware match.

### L-ROSETTA · code
**Each SASS opcode/modifier lowering means what the Rosetta probe derived,** only for the
forms seen. Any other form is a hard error.
- Where: `sass` `lower_line`.
- `.E` on `LDG` / `STG` / `LD` / `ST` / `REDG` / `ATOMG` means a 64-bit address register pair
  even when nvdisasm prints `[R10]` without `.64` (cuBLAS's cutlass3x SIMT sgemm computes the
  pair with `LEA` / `LEA.HI.X`; the case matches GB300 bitwise, docs/compile-vs-eager.md).
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
    ptxas's `ld.global.v8.b32`), `LDG.E.S8/.S16` / `LDS.S16` (sign-extended), `F2F.F32.F64` /
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
    rows, GB300 bitwise 100/100).
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

### L-PTX-HINT · code
**PTX cache hints do not change semantics:** `cp.async`'s `.L2::cache_hint` (with its
cache-policy operand) and `.L2::64B` / `.L2::128B` / `.L2::256B` prefetch sizes are accepted
and dropped by the PTX front-end. Every other unknown modifier of the instructions it lowers
is rejected (fail closed).
- Where: `ptx` `lower_inst`, the `cp.async` arm (`ASSUME(L-PTX-HINT)`).
- Status: **argued**: the PTX ISA defines them as cache-eviction priorities and prefetch
  sizes, with no effect on the values copied or on ordering. fixtures/s_ptx_cute_ops (one
  `.L2::128B` copy) = GB300 bitwise.
- If wrong: *miss* only if a hint changed the data or the completion order.
- Retire: never needed for verdicts; listed so that dropped modifiers are visible.

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
  - TMEM stores (`STTM`) read their registers before a later `LDTM`'s result lands;
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

### L-OPCLASS · code
**The variable-latency (`VAR_LAT`) and asynchronous-reader (`ASYNC_READ`) opcode sets are
complete.**
- Where: `scoreboard.rs`.
- Status: **validated on the corpus** (every opcode ptxas gave a scoreboard is in the sets).
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
  `*div_rn*slowpath*`, `BSYNC`), which is one `DivRn`: the correctly rounded quotient on both
  paths, as PTX defines `div.rn.f32`. Registers the slow path may write (other than the
  quotient) must be dead after it on every thread's trace (`seg::divrn_dead`).
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
  writes up to its `RET` are checked as before.
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
- If wrong: a quotient computed on the wrong path; *miss* or *false alarm* in numerics.

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

### N-MMA · code
**One `tcgen05.mma.kind::f16` (bf16 × bf16 → f32) step is
`g16-f25-rz-accin-tzero-einputs`.**
- Where: `fp.rs` `DEFAULT_MMA`, `exec.rs` `complete_mma`.
- Status: **validated, and enforced at its edges.** It is the unique match of 820 candidates
  on 49,152 GB300 outputs (docs/rung1a.md). That calibration covers finite normal inputs only;
  subnormal/Inf/NaN inputs or accumulators now give `unmodeled`. Other formats are rejected by
  the idesc check; the narrow kinds are N-MMA-F8 / N-MMA-BS. Shapes (PTX ISA table 48, the
  layouts in S-TMEM / E-PAIR): `cta_group::1` M = 64 or 128 with N = 8..256 in steps of 8
  (M = 128 N % 16 != 0 only with A in shared memory and without block scaling: measured
  forms), `cta_group::2` M = 128 or 256 with N = 16..256 in steps of 16; anything else is
  `unmodeled`. The arithmetic does not depend on N or the layout: `probes/tc_m128` random-operand
  configs (M = 128 N = 8 / 24, pairs N = 32 / 64 / 256, e4m3 for N-MMA-F8) equal the model
  bitwise.
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
- Where: `fp.rs` `HMMA_MODEL`, `exec.rs` `Op::Hmma` (fragments as in the PTX ISA).
- Status: **validated**: `ptxv calib-mma probes/hmma` (one warp per 16x8 tile, K = 64 in 4
  accumulating steps) scores 820 candidates on 98,304 GB300 outputs (6 input sets: normal,
  log-uniform 2^±12 / 2^±30, cancellation, alignment and accumulator probes): exactly this
  model matches all, as for tcgen05 (2026-11-03). `probes/hmma/expected` is checked bitwise in
  `cargo test`. Subnormal / Inf / NaN inputs or accumulators give `unmodeled`. Zero results
  are +0, as N-MMA (probes/mma_zero, bf16 and tf32, C in registers or RZ).
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
where v's sign bit is clear, the unsigned min where it is set. Its data-dependent branch is
not a violation; each of its atomics makes the shared word's node `Max(old, v)`.
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
  - MN-major or TMEM A operands, sparsity, negate.
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
  subnormal f32 accumulators (D is f32 only for these kinds); MN-major or TMEM A operands;
  M = 64.
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
  are ignored). `.relu` maps negatives to +0.
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
- Where: `lowp.rs`; `exec.rs` `Op::CvtPack` / `CvtRsX4` / `CvtUnpack`; PTX `cvt_lp`;
  SASS `f2fp_lp`.
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
