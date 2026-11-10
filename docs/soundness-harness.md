# Soundness harness: mutants, ptxv check, GB300 triage

**Status (2026-11-05).** `tools/mutate/` finds *misses*: kernels that ptxv calls clean but
that misbehave on GB300. It deletes or weakens one synchronization point at a time, checks
every mutant, and runs the mutants ptxv calls clean on the GPU under random delays. Results of
the first run are in "Results" below.

The motivation is the three misses an outside study found (remote mbarrier
arrives delivered instantly; `.aligned` dropped). Each came from a semantics shortcut with no
ASSUMPTIONS entry. If ptxv calls a mutant clean but the mutant is buggy, something in the
model is too generous.

## Pieces

| file | what |
|---|---|
| `tools/mutate/mutate.py` | mutant generator. One case in, one sibling case directory per mutant out |
| `tools/mutate/harness.py` | `baseline`, `gen`, `check`, `probe`, `hw`, `stress`, `report`. State goes in `--runs DIR` as JSONL, and every step resumes |
| `tools/mutate/test_mutate.py` | unit tests (no ptxv, no GPU): `$PTXV_PYTHON -m pytest -q tools/mutate` |

### Mutants

A mutant directory holds `case.toml`, the case's variants replaced by two:
- `good`: the base kernel;
- `mut`: the base kernel with a marker edit.

It also holds `kernel.ptx` (the base PTX, with the edited line tagged `// @MUT0`) and
`mutant.json` (operator, line, before/after). Everything else in the case is symlinked.
`ptxv check DIR --variant mut` checks the mutant on both front-ends; the SASS front-end
compiles the mutated PTX with ptxas. `ptxv hw DIR --variant mut` compares the mutant's GPU
output with the good kernel's expected output. An edit replaces exactly one line, so PTX
line numbers, and with them `--delay LINE:NS`, stay the same as in the base kernel.

PTX operators (`ptx_mutants`):

| op | edit |
|---|---|
| `del-bar` | delete `bar.sync` / `bar.arrive` / `barrier.*` / `bar.warp.sync` / `barrier.cluster.arrive|wait` / `griddepcontrol.wait` |
| `del-wait` | `mbarrier.try_wait`/`test_wait` → `mov.pred P, 1` (the wait loop exits at once); delete `tcgen05.wait::*`, `cp.async.bulk.wait_group*`, `cp.async.wait_group/all` |
| `weaken-wait` | `wait_group N` → `N+1`; `cp.async.bulk.wait_group` → `.read` |
| `parity` | flip a try_wait parity: an immediate, or a register through a scoped `xor` temp |
| `del-arrive` | delete `mbarrier.arrive*` / `cp.async.mbarrier.arrive*` |
| `drop-expect-tx` | `mbarrier.arrive.expect_tx` → plain arrive; delete `mbarrier.expect_tx` |
| `del-fence` | delete `fence.*` (proxy, acq_rel, sc, mbarrier_init), `membar.*`, `tcgen05.fence::*` |
| `del-commit` | delete `cp.async.bulk.commit_group`, `cp.async.commit_group`, `tcgen05.commit*` |
| `relax` | `.release` / `.acquire` / `.acq_rel` → `.relaxed` on ld/st/atom/red, mbarrier arrive/try_wait, `barrier.cluster.arrive` |
| `scope` | `.cluster`→`.cta`, `.gpu`→`.cta`, `.sys`→`.gpu` on the same ops and on `fence.acq_rel/sc` (not `fence.proxy`, not `mbarrier_init`); `membar.sys→gl→cta` |
| `move-fence` | swap an unguarded `fence.*` (not proxy / mbarrier_init) or `membar.*` with the memory access next to it (after the next one, before the previous one; no label in between) |
| `acq-fence` | insert `fence.acq_rel.gpu` after a relaxed (or semantics-less) global `ld` / `atom` / `red`: the access becomes fence-acquired (`Machine::fence_acq`); a stronger kernel, so a clean verdict is expected unless the access was exempted as non-acquiring |
| `atom-plain` | `atom.add.u32/s32 d, [a], v` with its result used → `ld` / `add` / `st` (relaxed): the read-modify-write is no longer atomic |

Scope mutants on mbarriers (`.release.cluster` → `.release.cta` on a remote arrive,
`.acquire.cluster` → `.acquire.cta` on a wait) and the SASS `ERRBAR` / `CGAERRBAR` deletions
survive while the mbarrier model ignores arrive / wait scope and delivers weak remote
`shared::cluster` stores at once (soundness miss of 2026-11-09, a 2-CTA hand-off whose stale
read was measured on hardware); a survivor of this kind is a miss, not noise.

Binary-only cases (`cubin = ...`) get SASS mutants instead (`sass_mutants`). Each one NOPs
one `BAR.*`, `WARPSYNC`, `UCGABAR_*`, `SYNCS.ARRIVE*`, `SYNCS.PHASECHK*`, `MEMBAR`,
`FENCE*`, `DEPBAR`, `LDGDEPBAR`, `UTC*BAR`, `ACQBULK`, `CCTL`, `ERRBAR` or `CGAERRBAR` of the entry kernel. It does
this with a `patch = [{at, nop = true}]` variant (crates/sass/src/scoreboard.rs). These
mutants are SASS-only, and `ptxv hw` cannot add delays to them (no PTX).

**Reach probe** (`reach_probe`). The mutated site is replaced by `trap;` (PTX) or by
`BPT.TRAP 0x1` (SASS, bit patch; guard and control bits kept). If ptxv finds no trap, the
site never executes at this launch. The mutant then equals the base kernel and is labelled
`dead`. The probes of dead sites also run 20 times on GB300 (`probe_hw.jsonl`). A trap there
would mean the model's reachability is wrong.

## Running it

```
source env.sh
cp $CARGO_TARGET_DIR/release/ptxv <scratch>/ptxv-bin/ptxv     # private copy
H="$PTXV_PYTHON -I tools/mutate/harness.py --runs RUNS --ptxv <scratch>/ptxv-bin/ptxv"
$H baseline --modes cert --jobs 12 --timeout 300 --max-states 500000   # every fixture's good
$H gen                                                                 # mutants of the clean ones
$H check --modes cert --jobs 16 --timeout 150 --max-states 300000      # --certify on every mutant
$H check --modes explore --jobs 16 --timeout 150 --max-states 300000   # plain exploration where cert said clean
$H probe --jobs 8 --gpus 1,3                                           # reach probes (+ GB300 trap runs)
$H hw --gpus 1,3 --plans 12 --hw-runs 200                              # triage the candidate misses
$H stress --gpus 1,3 --plans 12 --hw-runs 200                          # unmutated clean kernels under delays
$H report --out RUNS/report.md
```

- **Other cases.** `--cases FILE` takes cases other than `fixtures/*`, one per line:
  `path [base-variant] [arch]`. For example, an outside study's fix variants, with
  `sm_103a` replacing `sm_100a`: the base variant is materialized as the mutants' `good`
  (in `RUNS/bases/`), and the `.target` and `arch` are rewritten. Key names:
  `<case>[@<variant>]`, where `bugs/D123.../x` becomes `D123_x`.
- **Limits.** Every ptxv run sits in its own `systemd-run --user --scope -p MemoryMax=16G
  -p MemorySwapMax=0` (`--mem`), under `timeout`. With `--jobs 20` that is at most 320G.
  Shrink first.
- **GPUs.** `--gpus` sets `PTXV_GPUS`. Use idle GPUs only (never one in use by others). A
  run that finds every allowed GPU busy waits and retries. Multi-GPU cases need two GPUs in
  the list.
- `--only REGEX` restricts every step to some case keys; `--mutants REGEX` (check) to some
  mutants, and `--redo` reruns their check records.
- `--slices` (check) passes `--slices` to ptxv: one cluster at a time (E-CTA-INDEP), for
  cases whose whole launch is too large to certify or explore within the limits (e.g.
  `r3_grouped_mm`, 152 CTAs): `--only r3_grouped_mm --slices`.
- **Batch runner.** `check --emit-jobs FILE [--job-prefix P]` writes the pending runs as tasks
  (`id case_dir VAR=value... ptxv check {case} ...`, one per line) instead of running them;
  a batch runner executes them and leaves `<id>.json` (stdout), `<id>.log` (stderr),
  `<id>.exit` and optionally `<id>.run.json` (`seconds`, `timeout`) in a results directory;
  `check --ingest DIR` records the finished ones (`runner: true`). Mutants link the case's
  files with relative symlinks, so the runs directory can be mirrored next to the case tree.
  The explore pass depends on the cert results: emit, run and ingest cert first.
- **Synchronization fuzzer** (`tools/mutate/fuzz_sync.py`): random small kernels (1-3 CTAs,
  one actor per warp: plain / relaxed / release stores, plain / relaxed / acquire loads,
  atomic adds with used or unused results, polls of a counter, fences, counter resets,
  ticket-guarded accesses; half of them shaped as arrival counters or ticket dispensers),
  each checked with `--certify` and by plain exploration. A certified verdict whose classes
  or output count differ from a complete exploration's is a certificate soundness bug; such
  cases are kept under `OUT/disagree/`. It found the C-POLL-ATOM base miss (2026-11-08).
- A race found by a failed certificate's own run (ptxv prints `certificate run race (...)`
  before it explores; a feasible execution, LDS/STS-strong model) makes a run `caught` even
  when the exploration after it runs out of states or time (`cert_run` in check.jsonl).

### Source mutants

`tools/mutate/srcmut.py --out DIR` mutates ptxv itself: each mutant removes one entry of a
rule table a soundness argument depends on (num.rs `value_fault_operands`: an integer
divisor, `cvta.to.shared`, a constant-bank base), builds ptxv from a detached copy of the
tree (with the working tree's changes), and runs the fixture variants the entry exists for
(`fixtures/e_rep_divzero` DATA / POS / CVTA / LDC under `--certify --rep`). A mutant is caught
when some run's classes differ from the expectation, and survives otherwise. An entry no
fixture can reach is listed as a documented exclusion with its reason (`EXCLUDED`: IADD3 with
one carry-out). `tok-merge` gives every member of an instruction one C-WARP token again
(wellsync.rs `post_inst`): `fixtures/e_lockstep_nobsync` NOBSYNC must then certify (MISS 30).
`shfl-wait` makes a SASS SHFL wait for the rest of the warp again (exec.rs `split_enabled`):
`fixtures/e_warpcoll_split` NOBSYNC is then clean instead of a fault (MISS 39).
A mutant may also name cargo tests (`tests`: package, test binary, name filter), run in the
mutant's tree; it is caught when a test that ran fails (claims checked against recorded GPU
outputs, such as num's error bounds; adversarial tests). A build error, a missing test target or
a filter that matches no test is an `error`, not a catch. `PTXV_JOB_WRAPPER` prefixes the builds
and test runs (a memory-capped job runner). The mutant's tree holds the working
tree's tracked changes and its untracked files (new fixtures and tests). `--only` takes a
comma-separated list.

### Adversarial tests of proof rules

`crates/sim/tests/adversarial.rs`, one test per rule: the rule's premise and conclusion, a
kernel that violates the premise while looking like the good case (`fixtures/e_*`, PTX and
SASS where both fronts apply; binary-only with a `patch` where only SASS has the shape), and
the source mutant in `srcmut.py` that weakens the rule's implementation and fails the test.
The helper `adv` checks each variant's expected classes on both fronts and that a
certificate, when one holds, reports exactly the exploration's classes. A kernel that comes
out clean or certified although it is bad is a soundness miss; its test pins the current
verdict with the miss named in the doc comment, so a fix shows up as a test change.

| rule | fixture | mutant |
|---|---|---|
| E-CONST-BRA | e_const_bra, e_const_bra2 (+ an injected wrong fold) | cbra-nocheck |
| E-RET | e_ret (binary-only; + an injected wrong return table) | ret-nocheck |
| E-ELECT | e_elect_mid (PTX gap: a middle-lane leader), e_elect_lane | elect-lowonly, elect-high-sass |
| E-CLOCK | e_clock_timeout (miss: a timeout path, GB300 wrong 20/20) | (none: the rule is the shortcut) |
| E-NANOSLEEP | e_nanosleep | (none: a no-op has nothing to weaken) |
| S-BRA-COND | (r4_sum_torch, probes/div_ftz) | bracond-ignore |
| E-BSYNC / E-LOCKSTEP | e_lockstep_nobsync (binary-only; GB300 wrong without the BSYNC; certificate miss fixed, MISS 30) | bsync-own, tok-merge |
| E-WARP-ORDER | e_warp_order_div (PTX-contract miss on CONV) | (E-LOCKSTEP join: none yet) |
| E-DRF (TMEM) | e_drf_tmem | tmem-norace |
| E-BAR-EXIT | (probes/bar_exit, bar_exit_warp) | barexit-phantom |
| E-ALIGNED | (e_aligned_bar AB1 / AB3) | aligned-noviol |
| E-ALIGNED, part of the warp exited | e_aligned_exit, e_aligned_exit2 (PTX contract violation; SASS clean, as GB300) | aligned-sass-guarded, aligned-ptx-exit |
| S-ALIGN | (e_misaligned) | align-4 |
| S-DSMEM | e_dsmem_rank (a foreign rank through shared::cta faults; GB300 traps) | dsmem-rank |
| S-SMEM-ADDR32 | (probes/smem_addr_wrap) | addr32-nowrap |
| S-GLOBAL-INIT | (s_globals, probes/ptx_globals) | ginit-zero |
| E-STSM-DUP | (e_stsm_warp dupdiff) | stsm-dup-nocheck |
| S-CLUSTER-LIFE | e_cluster_life (certificate miss on NOFINAL, fixed) | (none yet) |
| S-SCOPE | e_scope_cta | scope-cta-all |
| E-CLUSTER-BAR | e_cluster_exit (exited threads; GB300 completes) | clbar-relaxed-inits |
| E-BAR-RED | (probes/bar_red) | barred-and-or |
| E-BAR-ARRIVE | (e_bararv) | bararv-notick, bararv-gen |
| E-BAR-OVER | (e_barover) | barover-2x |
| S-MBAR | (r0_tma_sum B01) | mbar-tx-ignore |
| S-MBAR-MEM | e_mbar_sts64 (ptxas's STS.64 init; COUNT3 hangs GB300) | mbarmem-count |
| E-SETMAXNREG | (probes/setmaxnreg_pool inc96, a recorded miss; pinned) | (none: the rule is the shortcut) |
| E-POLL-ANY | e_poll_any_live | vip-nolive |
| E-ASYNC-STEP | (e_gmm_epilogue_reuse) | bulk-anysb |
| E-ALLOC | e_tmem_alloc_block (HOLD hangs GB300) | alloc-noblock |
| L-OPCLASS | e_sb_mufu (binary-only; NOWAIT wrong on GB300 16/50) | opclass-nomufu |
| N-FADD | e_fadd_nan (GB300 bits) | fma-nan-payload |
| N-FCHK, N-XORSIGN, N-F2I16, N-F2FP-RZ, N-QMUL4, N-IMMA, L-SHF, L-LOP3-P | (their GB300 replay probes) | fchk-swap, xorsign-minmax, f2i16-zext, f2fprz-inf, qmul4-nan, imma-wrap, shf-wrap, lop3p-and |

`tools/mutate/certsweep.py` checks every fixture variant on each front-end with and without
`--certify`; a certified verdict whose classes differ from a complete exploration's is a
certificate miss. On 2026-11-09 it found the S-CLUSTER-LIFE and E-BSYNC certificate misses above
among 1,126 runs (the other differences were explorations over budget, and e_ticket_addr's
clean-up-to-permutation).

## Reading the results

Per (mutant, front-end, mode), `check.jsonl` holds a status:

| status | meaning |
|---|---|
| `caught` | a class other than clean/contract (race, deadlock, violation, fault, nondeterministic, trap, ...) |
| `contract` | only contract deviations (GB300 tolerates them); counted as clean for triage |
| `clean` | complete and clean: a **candidate miss** |
| `budget` | state budget or timeout without a finding |
| `rejected` | the mutant does not load, lift or compile |

- **Per front-end.** A front-end counts as caught if either mode caught the mutant.
- **Modes.** `cert` is `--certify`; when the certificate fails, it falls back to exploration.
  `explore` runs only where cert said clean.
- **Disagreement.** `cert_disagrees` means `--certify` was clean but plain exploration found
  something, which points at an unsound certificate.

Final labels (`report`):

| label | meaning |
|---|---|
| `caught` | every runnable front-end caught it |
| `dead` | candidate, but the reach probe says the site never executes at this launch (equivalent mutant) |
| `candidate` | clean on some front-end, not triaged yet |
| `confirmed` | **confirmed miss**: wrong output, hang or trap on GB300 while ptxv says clean, and the good kernel passes under the same delays |
| `unobserved` | clean on GB300 in every trial (1 unsteered + `--plans` random delay plans, `--hw-runs` runs each). Either redundant (another sync covers it) or a real bug that GB300 does not show: see the per-mutant notes |
| `redundant` | (manual, `RUNS/manual.jsonl`) argued redundant given other synchronization |
| `budget` / `rejected` | as above |

- **Delay plans** (`delay_plans`) put 1–3 `nanosleep`s of 0.5–100 µs before PTX lines next to
  synchronization or memory instructions. Half the plans stay within 8 lines of the mutated
  line. The seed is the CRC of the mutant name, so a plan can be reproduced.
- **Witness.** For a `confirmed` mutant, `hw.jsonl` holds the failing plan (`delays`) and
  the GPU outcome. Rerun it with
  `ptxv hw RUNS/mutants/<case>/<mutant> --variant mut --delay L:NS ...`.
- **Stress.** `stress.jsonl` holds the unmutated kernels under the same delay plans. Any
  failure there is a miss in ptxv, or a harness bug (check the good kernel's expected output
  first: `PTXV_HW_EXPECT=run` replaces the numerics run when that fails).

## Results (2026-11-05)

**Setup.** ptxv was main 93477555ca66 plus the working-tree changes of that day, in the
`base` commit of the agent copy. `fixtures/`: 142 cases have a `good` variant. Of those,
124 had a `good` that `--certify` reports clean or contract-only on at least one front-end.
The other 18 were left out: 10 by design (violation / race / nondeterministic) and 8 over
budget at 500k states / 300 s.

There were also 19 fix variants from the outside study (bugs/D110 to D129, copied, run with
`sm_103a`). Budgets were 300k states / 150 s / 16 GB per mutant and mode. GB300 used 200 runs
per trial and up to 12 delay plans.

Full tables: `docs/soundness-results.md` (fixtures × labels, operators × labels, every
non-caught mutant).

| | mutants | caught | cert-miss | dead | redundant (same SASS) | unobserved | confirmed | budget | rejected | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixtures (124 cases) | 1830 | 467 | 7 | 242 | 23 | 671 | 7 | 333 | 2 | 78 |
| study fixes (19) | 306 | 170 | 0 | 0 | 4 | 68 | 0 | 55 | 0 | 9 |

`cert-miss` (`--certify` clean, exploration caught it) are miss 1 below. 4 of its 7 also
failed on GB300. `confirmed` are misses 2 and 3.

- **other.** GB300 runs that could not be made:
  - `c_smm_mxf8`: its unmutated `good` traps (illegal instruction) under `ptxv hw`, a
    pre-existing replay problem;
  - `r1g_allreduce` and `r4a_*` need 3 GPUs (only 3 were free);
  - `D123_cas_*`, `D127_min_vote`, `D127_real`: `ptxv hw` loads PTX cases through the PTX
    front-end, which rejects `atom.cas`, `vote.any` and `mad.lo.s64`.
- **dead.** The reach probe never executes the site at this launch, so the mutant is the
  base kernel there. For example, CUTLASS and cuBLAS paths for other shapes, and the unused
  modes of `e_resid`.

### Confirmed misses

Each one: ptxv calls the kernel clean (certified), and GB300 is wrong while the good kernel
passes the same runs.

**1. `--certify` misses a race between a plain write and other threads' atomics, when the
writer's own atomic to the same word follows its write.**
- **Mutants.** 7, one root cause: `c_fmax_atomic`, `c_fmax_atomic_bad`, `c_online_ok`,
  `c_online_bad_{mp,sel,shfl,word}`. Each NOPs the `BAR.SYNC` between warp 0's `STS` init of
  the shared maxima and every warp's `ATOMS.MAX`/`ATOMS.MIN`.
- **ptxv.** `--certify` reports clean ("well-synchronized"). Plain exploration of the same
  mutant reports the race; the harness lists these under `cert_disagrees`.
- **GB300.** `c_online_{ok,bad_mp,bad_shfl,bad_word}`: 51–64 of 200 runs wrong, unsteered (cubin, no delays possible), with or without the NOP's scoreboards cleared. `c_online_bad_sel` and `c_fmax_*`: unobserved.
- **Reproducer.** `tools/mutate/repro/cert_fasttrack` (PTX, both front-ends): `nosync` is
  clean under `--certify` and a race under exploration. On GB300 it is wrong in 200/200 runs,
  and `good` passes 200/200.
- **Root cause.**
  - `crates/sim/src/exec.rs` `Machine::access_x`: the FastTrack shadow keeps one last-write
    epoch (and the reads since), with its strength. A strong write (the atomic) replaces a
    plain write (`s.w = Some(me)`) and clears plain reads (`s.r.clear()`). Later accesses
    are compared only against the strong epoch, and `morally_strong` drops strong-vs-strong
    pairs. So another thread's atomic that is unordered with the earlier plain `STS`, but
    comes after the writer's own atomic, is no race.
  - `crates/sim/src/wellsync.rs` `check_certified` tries the weak (LDS/STS plain)
    certificate first. Its soundness argument, in the module doc and in ASSUMPTIONS
    E-WELLSYNC ("the detector is FastTrack-precise"), assumes the run's race detector sees
    every race. That fails for this mixed plain/strong history: one run in the run-once
    order reports nothing.
  - Exploration still finds the race, because it also runs the other warps' atomic before
    warp 0's.
  - The E-LAUNCH-GROUPS argument (ASSUMPTIONS, "FastTrack's shadow of a byte holds its last
    write ...: plain accesses are ordered with every write, else the group reported a race")
    makes the same assumption. Not tested.

**2. A branch on a predicate register nothing has written.**
- **Mutants.** 5, one pattern, all cuBLAS nvjet kernels: `c_mm_bf16_mcast` 0x1b70,
  `c_mm_bf16_nnt` 0x1b00, `c_mm_bf16_ttn` 0x1b10, `c_smm_nvf4` 0x2040, `r2_cublas_gemm`
  0x1b00. Each NOPs the tile scheduler's fast-path
  `SYNCS.PHASECHK.TRANS64.TRYWAIT P4, [URx], R4`. The next instruction, `BRA !P4` to the
  retry stub, then reads a P4 that nothing wrote on that path (checked with
  `PTXV_TRACE_CTA`).
- **ptxv.** Clean (certified). P4 reads as 0 in every lane, so every lane takes the retry
  stub, which waits properly: the good kernel in all but name.
- **GB300.** 198/200 runs wrong, or a hang.
- **Controls** in `tools/mutate/repro/uninit_pred`, each 200/200 correct on GB300:
  - `nowait_stub`: NOP plus an unconditional branch to the stub. That is exactly what ptxv
    executes for `nowait`.
  - `nowait_nobra`: NOP plus no branch (P4 = 1 in every lane).
  - `stub`: the try-wait kept, the branch made unconditional.

  So on GB300 the unwritten P4 is not 0 in every lane.
- **Root cause.** The predicate file has no undefined state. `crates/sim/src/machine.rs`
  zero-initializes `Thread::preds` (`Machine::new`), and `exec.rs` reads it through `prd`.
  The uninitialized-register check (`exec_thread_inner` → `report_uninit`, through
  `src_regs` / `reg_umask`) covers general registers only. No ASSUMPTIONS entry covers it.
  Compiler output should not read unwritten predicates. A binary or lifter bug that does is
  accepted silently.
- **Note.** These are binary mutants (an instruction NOPed). The original SASS NOP patch
  kept the control bits; the reproducer clears the scoreboards and fails the same way.

**3. CUTLASS grouped-GEMM epilogue: rewriting a TMA-store staging buffer before the store has
read it.**
- **Mutants.** `r3_grouped_mm` `sass-del-bar-0xc420` (NOP the `BAR.SYNC 0x1, 0x80` after
  `DEPBAR.LE SB0, 0x1`) and `sass-del-wait-0xc400` (NOP that `DEPBAR.LE`, the bulk-store
  read wait).
- **ptxv.** `--certify` clean (certified, 25k steps). A one-run trace (`PTXV_TRACE_CTA`)
  shows warps 6–8 storing subtile 2 into the staging buffer that store 0 reads, with no
  happens-before path to warp 5's `DEPBAR.LE`, and no race reported. Plain exploration with
  `--slices`: no finding within its budget.
- **GB300.** Both 200/200 runs wrong; good 200/200 correct. The scoreboards were cleared in
  the reproducer.
- **Reproducer.** `tools/mutate/repro/gmm_epilogue_reuse` (`PTXV_HW_EXPECT=run`).
- **Root cause: not isolated.** The hand-written PTX analogue (`tools/mutate/repro/
  tma_store_reuse`: two staging buffers, `wait_group.read 1`, no second barrier) *is*
  caught on both front-ends. So the gap is specific to this kernel. Candidates are the SASS
  front-end's attribution of the `UTMASTG` source read (`complete_tma_store`) or its address
  range for this tensor map, against the `STS.128` addresses. This needs a ptxv-side trace
  of the store's shadow accesses.
- **Fixed** (fixtures/e_gmm_epilogue_reuse, E-ASYNC-STEP): a wait covers only the bulk groups
  flushed on its own scoreboard. The certificate's run now reports the race (`race:proxy`),
  so the certificate fails. Plain `--certify` then explores the whole launch (152 CTAs) and
  ends as `budget`; the certificate runs alone take longer than the 150 s limit. With
  `harness.py check --slices` (ptxv `--certify --slices`, E-CTA-INDEP) the first GEMM
  cluster's certificate run reports the race after ~5 s, ptxv prints it as `certificate run
  race (cluster 1048576): race:proxy:3737:3581: ...` before exploring, and the harness
  classifies both mutants as `caught` (`cert_run`), 2026-11-05. The good kernel certifies
  under `--slices` (77 clusters, contract-only). ptxv's verdict also keeps such a race when
  the exploration that replaces the certificate does not complete (`merge_run_races`).

### Known misses, re-found

The harness's own check.
- **Stress.** `D117@probe_intrawarp` (one warp split across two aligned `bar.sync`):
  ptxv clean, GB300 hang, unsteered.
- **Unobserved on GB300**, as the study also found: `D113@bug` (remote arrive in flight at
  exit) and `D117@prefix` (divergent aligned named barriers). Both are ptxv clean, and both
  stayed correct in every trial (1 + 12 delay plans).

### Unobserved and redundant mutants

**Same SASS** (`redundant`, 27 mutants): ptxas emits the base kernel's code for the mutant,
so GB300 runs the base kernel.
- `.acquire` → `.relaxed` on `mbarrier.try_wait` in the CuTe kernels (`r2_cute_*`, `r3_cute_*`).
- `.release` → `.relaxed` on the mbarrier arrives of `e_sym` and `e_exitbound`.
- The dead-mode lines of `e_resid`.
- `fence.mbarrier_init` in D117 / D127.

**Unobserved** (fixtures 671, study 68): clean in ptxv, correct on GB300 in every trial.

By operator:

| set | SASS NOP | del-bar | tcgen05 fence | other fence | del-wait | relax / scope | weaken-wait | del-arrive / commit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixtures | 513 | 42 | 31 | 27 | 24 | 17 | 14 | 3 |
| study | – | 6 | 15 | 7 | 15 | 22 | – | 3 |

By front-end, among the fixtures:
- 606 were clean on the SASS front-end only, in binary-only cases. Of those, 308 were
  `contract`-only in cases whose `good` is already `contract` (the cuBLAS / trtllm-gen GEMMs).
- 50 were clean on both front-ends.
- 33 were caught by the PTX front-end and clean on the SASS front-end: PTX-level weakenings
  that this ptxas output does not exhibit, e.g. the tcgen05 fences.

Examples argued redundant from the code. These are not checked mechanically.
- `r1g_allreduce` runs one thread per CTA, so `bar.sync` is a no-op.
- The D123 signal-pad kernels move only flags: relaxing or rescoping a poll protects no
  payload.
- `r1b_pipeline`'s final `cp.async.bulk.wait_group 0` comes after a `.read` wait that already
  covers shared memory.
- `r1c_cluster`'s last remote arrive is to a phase nobody waits for.

The rest, the SASS NOPs above all, are weakenings that GB300 timing did not expose. Cubin
cases get no delays (no PTX), so they had only unsteered runs: 1 trial of 200 runs.

**Limits.**
- 12 random delay plans do not prove a mutant harmless. `unobserved` means "no wrong output
  seen".
- The dead sites' trap probes were not run on GB300 (`probe --gpus` was skipped).
- Labels differ in how strongly they are established:
  - `dead` rests on ptxv's reachability (exhaustive under the certificate, or the run).
  - `redundant` rests on ptxas determinism.
- 388 mutants (333 + 55) stayed `budget`: large cuBLAS / trtllm-gen / D127 kernels, where
  exploration after a failed certificate does not finish in 150 s.
- The SASS NOP patches of this run kept the instruction's control bits. `mutate.py` now
  clears the scoreboards, as a deletion would. Every confirmed SASS mutant was re-run with
  them cleared (the reproducers) and failed the same way.
