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
It also makes ptxas's warp-aggregated atomic deadlock (`fixtures/e_shfl_warpagg` good, forum
post 74). `shfl-wait-bsync` restores the 2026-11-07 rule (wait for live lanes not parked at a
BSYNC, another SHFL or a WARPSYNC): `e_shfl_warpagg` NOBSYNC deadlocks. `shfl-undef-src` and
`shfl-undef-src-ptx` give a source lane that is not executing the shuffle the reader's own
value instead of a fault (exec.rs SASS arm, `exec_combined`): `e_shfl_warpagg` XLANE (and
`e_warpcoll_mask` XSRC on PTX) lose the fault. `shfl-wait-parked` is forum post 74's proposed
rule (wait unless every other live lane is parked at a BSYNC / WARPSYNC or an aligned BAR):
`e_warpcoll_split` NOBSYNC is clean again (MISS 39). `sc-any-write` lets writes that are not
morally strong force a Fence-SC order (S-FENCE-SC): `fixtures/e_fence_sc_publish` CTASCOPE loses
its race on the partials (test `fence_sc_publication`). `sc-drop-forcing` orders SC fences as
executed, without the forcing pair of atomics: `fixtures/e_fence_sc_unforced` (two lanes' fences
in one step, which the explorer runs in lane order only) is clean. `sc-lane-step` takes the
coherence pair of two lanes of one PTX atom step (`e_fence_sc_lanes` good clean), `sc-ticket-addr`
logs C-TICKET-ADDR claims in the certificate's runs (`e_ticket_addr_fence` SC certified on PTX),
`sc-membar-acq` counts a SASS `MEMBAR.SC` without its `CCTL.IVALL` (`e_fence_sc_publish` NOCCTL
clean), `sc-width` drops the equal-width condition (`e_fence_sc_mixed` good clean on SASS).
`sc-lane-pc` keys the lane-step exclusion on one instruction again (`e_fence_sc_lanes` INDIRECT
and PCS clean, PCS certified). `obs-lane-step` lets a lane synchronize with what a lane of its
warp wrote at the same PTX instruction in lane order, `obs-lane-cont` drops the patterns that
write continued from writers outside the warp too (`e_atom_lane_cont` a false race),
`obs-lane-samewarp` keeps the patterns continued from other lanes of the warp (`e_atom_lane_order`
THREE clean), `ticket-warp` lets C-TICKET accept claims by lanes of one warp that happen-before
does not order (MISS 140: `e_atom_lane_order` good clean, NOREAD certified), and `ticket-samepc`
limits that check to claims at one instruction (MISS 187: `e_fence_sc_lanes` PCS_READER0
certified).
`bsync-coll` keeps the GB300 profile's BSYNC waiting beside lanes parked at a
`WARPSYNC.COLLECTIVE` (since MISS 81 and gap G1 no fixture catches it: outside the
cross-region shape of E-BSYNC's residual G2 it only drops the released lanes' clock join, which
is conservative, not equivalent; see ASSUMPTIONS.md), and `exit-coll` drops an
EXIT's re-check of collective groups: the profile verdict of `fixtures/e_bsync_parked_collective`
then deadlocks (E-HW-BSYNC-COLLECTIVE).
`cpasync-write-issue-clock` lands a cp.async's write as its issuing thread at its issue clock
again (MISS 34: `fixtures/e_cpasync_early` LATEWAIT clean), `cparrive-arrive-clock` lands
mbarrier-tracked copies at the arrive's clock (`e_cpasync_mbar_early` EARLY / OWN clean),
`cp-overlap-off` drops the check between copies retired together (`e_cpasync_overlap` SAME /
GROUPS / MBAR clean), `cp-relanded-off` the one between copies two waits landed and
`cp-wait-epoch-fixed` the per-wait epochs of a thread's wait actor (LATER clean),
`cp-warp-owed-off` the join of copies another path's `DEPBAR` retired (`e_cpasync_div_wait`
good a race on the SASS front) (E-CPASYNC, E-CPASYNC-MBAR). `async-ids-check-off` drops the
check of a thread's async actor ids, `async-ids-wrap` reports and wraps an id past them as
before MISS 68 (`async_actor_ids_fail_closed`, with the ids cut to 4096: a crash, a run past
the limit), and `async-ids-stride-4096` restores the fixed 4096 ids per thread
(`e_async_ids_nc`, `e_async_ids_stream`, `e_async_ids_last` unmodeled). `bulk-read-is-full-wait` makes
`cp.async.bulk.wait_group.read` acquire the bulk store's writes again and `bulk-write-read-epoch`
runs those writes at the epoch of its reads (MISS 32: `fixtures/e_tma_store_readback` READ clean
on the PTX front); `bulk-depbar-acquires-writes` lets a SASS bulk `DEPBAR` acquire the writes
without the `CCTL.IVALL` again (MISS 102: `fixtures/e_tma_read_l1` READ4 clean and certified on
the SASS front) (E-ASYNC-STEP). `over-arrival-late-off` stops recording mbarrier updates that
land after a completion their actor does not know of (`fixtures/mbar_over_arrival` clean on both
explorers), `bra-u-per-lane` lowers a guarded SASS `BRA.U` per lane again (`probes/bra_u` differs
from GB300), `s2f6-relu-dropped` drops the s2f6 decode's `.relu` (`probes/s2f6_cvt/forms` differs
from GB300).
A mutant may also name cargo tests (`tests`: package, test binary, name filter), run in the
mutant's tree (test binary `lib`: the package's unit tests; `bin:NAME`: binary NAME's, the
filter their full path); it is caught when a test that ran fails (claims checked against recorded GPU
outputs, such as num's error bounds; adversarial tests). A build error, a missing test target or
a filter that matches no test is an `error`, not a catch. `PTXV_JOB_WRAPPER` prefixes the builds
and test runs (a memory-capped job runner). The mutant's tree holds the working
tree's tracked changes and its untracked files (new fixtures and tests). `--only` takes a
comma-separated list; one that names nothing, or a name that is not a mutant that runs, exits 2
before anything runs. `--check-anchors` builds and runs nothing: it prints every mutant whose
find text does not occur exactly once in its file (a run reports such a mutant as an `error`)
and exits 1 if there is one; with `--root DIR` it checks the files under DIR instead (the
anchors test, `crates/cli/tests/srcmut_anchors.rs`, plants a broken anchor in a scratch tree
there as its positive control; the path without `--root` is covered by `pytest tools/mutate`,
which the same test binary runs). The mutants of files under `fb/` (the `kdb-*` ones, which
mutate the kernel_db tooling) cannot run or be checked in the open-source export, which drops
`fb/`: in a tree that has neither the file nor `fb/OSS_EXCLUDE.txt` a run and `--check-anchors`
skip such a mutant and print the count of skipped mutants on stderr. In the internal tree
(`fb/OSS_EXCLUDE.txt` present) a missing `fb/` file is a broken anchor like any other.

Open source-mutant gaps (no mutant in `srcmut.py` until a test catches it):
- E-ELECT candidates (`elect_lanes::candidates`): the key of an active set's candidate lanes
  ignoring the active sets whose leaders are already fixed (`others`) survives. No test builds
  clones of an electing warp that sit in different active sets (a role branch that sends the
  leader and its clones to one elect at different times), where the key decides.
- Equivalent, not a gap: a shuffle from unknown source lanes taken as one value for every lane
  (`elect_lanes::warp_classes`); the shuffle's source operand, an effect operand, already
  tells those lanes apart.
- Equivalent, not a gap: num's clobber taint (`num_clobber`, `Taint::clobber_taint`) beside
  N-NUM-TRACK's rule for operands holding a rounding (`rounded_pre`): an untainted such operand
  puts its lane in `cur_rounded`, which taints every destination left without a node; a
  tainted one taints the destination through the instruction's taint rule (whole, except
  LOP3's bitwise rule, which leaves untainted only bits fixed by the operands' untainted
  bits).

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
| E-ELECT | e_elect_mid (MISS 29: a middle-lane leader, PTX clean and certified with the two end-lane policies; race now, explored and under `--certify`), e_elect_lane; e_elect_lane_number, e_elect_tid_redef, e_elect_tccp, e_elect_ballot, e_elect_tid_ctl, e_elect_shfl, e_elect_tid_brx (taint misses 49-51, 71, 72, 79: clean then, race now) | elect-fn-off, elect-fn-cert-off, elect-fn-lowest, elect-cand-ends, elect-lowonly, elect-high-sass, elect-lane-ls, elect-tidx-flow, elect-tccp-noread, elect-mma-noread, elect-mem-ls, elect-mem-perlane, elect-mma-tile, elect-coll-pattern, elect-coll-ctl, elect-tid-ctl, elect-brx-idx |
| E-ELECT lane classes | (e_elect_lane_classes, e_elect_lane_classes_match_every_lane); e_elect_mid (the candidates) | elect-cls-lanereg, elect-cls-coll, elect-cls-uncontrolled, elect-cls-effects, elect-cand-ends |
| E-CLOCK | e_clock_timeout (was a miss: a timeout path, GB300 wrong 20/20, 200/200 in the MISS 28 audit; now explored both ways, the race); e_clock_timeout32, e_clock_mem, e_clock_data, e_clock_addr (a clock-derived address: GB300 stores either word), clock_pace_store / _live (MISS 28; tests clock_taint_guarded_kill, clock_taint_no_third_category, probe_clock32); e_clock_callee (miss 70: the clock read in a callee, GB300 wrong 200/200; a RET returns to the call sites now); e_clock_lanes, e_clock_brx, e_clock_hang (misses 76-78: a lane-varying clock predicate, a clock index, a 2000-cycle hang guard; GB300 wrong 400/400, 208/400, 20/20); e_clock_hi (the pair's high half); e_clock_hang_wrap, e_clock_hang_callee, e_clock_lanes_brx, e_clock_lanes_call, e_clock_brx_mem (the review of 76-78); e_clock_hang_or (a guard's AND / OR combine: the operand the trap side forces; no exemption on the OR-true / AND-false trap side), e_clock_hang_undef (a start read or a deadline undefined on some path) | clock-ret-sites, clock-lanes-off, clock-brx-index, clock-hang-any, clock-hang-threshold, clock-pair-hi, clock-hang-starts, cfg-brx, clock-brx-mem (clock-hang-callee, cfg-call: excluded, see srcmut.py), clock-hang-peel, clock-hang-undef, clock-choice-off, clock32-ptx, clock32-sass, clock-delay-impure, clock-delay-live, clock-mem-off, clock-out-complete, clock-cert-choice, clock-cert-store, clock-effect-off, clock-local-slot, clock-guard-kill, clock-data-guard |
| E-NANOSLEEP | e_nanosleep | (none: a no-op has nothing to weaken) |
| S-BRA-COND | (r4_sum_torch, probes/div_ftz) | bracond-ignore |
| E-BSYNC / E-LOCKSTEP | e_lockstep_nobsync (binary-only; GB300 wrong without the BSYNC; certificate miss fixed, MISS 30); e_bsync_plain (a BSYNC without .RECONVERGENT holds no lane, MISS 81); e_bsync_plain SLEEP_RECONV, e_bsync_parked_collective RECONV (nor does a .RECONVERGENT one around a region that can yield or wait in a collective: gap G1; SASS clean / deadlock before) | bsync-own, tok-merge, bsync-plain-holds, bsync-form-plain, bsync-g1-off, bsync-g1-nocall, bsync-g1-nocoll, bsync-g1-break |
| E-LOCKSTEP (the lanes of one instruction, PTX) | e_lane_order good, RELAXED, EXCH, V100, PAIR and e_lane_order_ticket THREE, CAS3, FADD3, VEC2, good, EXCHLIVE (MISS 74: PTX clean with lane order only; nondeterministic now); VECX (float adds exact in every order: lane order, complete); (test adds_exact_only_when_every_partial_sum_is); e_mm_lane good, RED, TRAP0 (gap A: a multicast address kept lane order, clean and complete; now the orders for every copy at once, incomplete with the note; test e_mm_lane_multicast_lanes_take_any_order); e_lo_sync good, FENCED (gap B: a dead-result atomic whose lanes publish a clock commuted, clean; race now; RELAXED_NOREAD still commutes; test e_lo_sync_synchronizing_lanes_take_any_order) | lane-order-off, lane-order-last, lane-order-live, lane-order-perm, lane-order-mark, lane-order-cert, lane-order-vec-first, lane-order-exact-off, lane-order-exact-loose, lane-order-multi-off, lane-order-sync-commute, lane-order-fence |
| E-LOCKSTEP (the lanes of one shared store, strict SASS) | e_sts_keep good, HIGH, OUT, e_sts_keep_wide good, V4, e_sts_shapes (every variant) and e_lane_order good, RELAXED, PLAIN, V100, PLAIN100, PAIR (MISS 75 at the strict level: SASS kept lane 31's value only, so e_sts_keep good, OUT and the e_lane_order variants were clean and HIGH trapped; GB300 traps on good; now one order per distinct value; test e_sts_keep_strict_sass_takes_every_lane) | sts-lane-order |
| E-HW-STS-LOWEST | e_lane_order good, V100 and e_sts_keep under the GB300 profile (MISS 75: lane 31's value before, lane 0's now, as GB300; test gb300_profile_keeps_the_lowest_lane_of_one_store); e_sts_shapes, e_sts_keep_wide (the width rule: the lowest active lane of the last 32 / (bytes / 4)-lane phase, GB300's output for 19 shapes; lane 0 at every width before, e_sts_keep_wide clean where GB300 traps; test gb300_profile_keeps_the_last_phase_of_one_store) | sts-lowest-off, sts-lowest-explored, sts-width-off |
| E-LOCKSTEP (the lanes of one global store, strict SASS) | e_stg_keep good, HIGH, OUT, HALF, ODD, PLAIN, PLAIN_HALF, W64, U8, SYS, SYS_HALF, CTA (MISS 168: SASS kept lane 31's value at both levels and under the certificate, so good was clean where GB300 traps in every launch; now one order per distinct value; test e_stg_keep_strict_sass_takes_every_lane) | stg-lane-order |
| E-HW-STG-LOWEST | e_stg_keep under the GB300 profile (MISS 168: the lowest active lane's value for one 32-bit STG in every measured form but MMIO, GB300's output for 16 variants; good traps; W64, U8 and the MMIO variants stay explored, MMIO_TRAP31 traps as on GB300 (was clean); test e_stg_keep_strict_sass_takes_every_lane); probes/stg_mmio_lane (`run.sh`); e_mm_lane good SASS under the profile (a multicast STG stays explored; test e_mm_lane_multicast_lanes_take_any_order) | stg-lowest-off, stg-lowest-wide, stg-lowest-mcast, stg-lowest-mmio |
| S-PROXY-WAR / E-HW-PROXY-WAR | e_mbar_lww WAR (strict: contract on both fronts; GB300 profile: discharged), WARF (clean) | proxy-war-off, proxy-war-nodischarge |
| E-HW-SPLIT-UR | e_split_ur (strict: nondeterministic through a switch inside a `VOTEU.ANY` window that holds an `ATOMS`; GB300 profile clean, GB300's output bitwise; the certificate and DPOR clean: MISS 150, pinned); (probes/warpsync_pairs SASS under the GB300 profile: the windows after each `VOTEU.ANY UR5`) | split-ur-off, split-ur-sts, split-ur-ungated |
| E-HW-BSYNC-YIELD | e_bsync_plain good under the GB300 profile (MISS 81: race; the certificate refuses at the plain BSYNC), probes/syncwarp_fast (profile clean, bitwise; strict: the uniform-register clobber stays); e_bsync_block good, WARPSYNC (a lane blocked at another barrier's BSYNC / at a WARPSYNC opens a plain BSYNC: race under the profile), NOBLOCK (clean there) | bsync-yield-off, bsync-cert-yield, bsync-yield-strict, bsync-yield-noblock |
| E-WARP-ORDER | e_warp_order_div CONV, s_atom_ext (MISS 31: PTX clean, s_atom_ext certified; nondeterministic now, SASS clean); e_warp_order_reconv good, BARV, SHFLV, ELECTV, SPAN and e_warp_order_partial good, NOHALF (MISS 73: PTX clean after a divergence and a bar.sync / shfl.sync / elect.sync, or for lanes that stay together while the rest of the warp branches away; nondeterministic now, SASS clean); (test warp_order_model_divergence_is_hashed) | warp-order-off, warp-order-commute, warp-order-group-split, warp-order-gather-bar, warp-order-gather-warpsync, warp-order-gather-coll, warp-order-hash (the SASS join: none yet); warp-order-complete (e_warp_order_other good, TRAP: the reported pair's other order unexplored, complete before, incomplete now) |
| E-DRF (TMEM) | e_drf_tmem | tmem-norace |
| E-BAR-EXIT | (probes/bar_exit, bar_exit_warp) | barexit-phantom |
| E-ALIGNED | (e_aligned_bar AB1 / AB3) | aligned-noviol |
| S-NC | e_ld_nc, e_ld_nc_forms (per byte, .v4 / .v8, red, bulk copy, another CTA), e_ld_nc_pdl (stream order, PDL), e_ld_nc_xgpu (a peer GPU's write before the reader starts; signal and acquire scopes, fence patterns, an unordered reader; WAITS: a join), e_ld_nc_xgpu_unord (the launch order unknown or recorded), e_ld_nc_xgpu_frel (a .gpu fence heading the hand-off's release sequence, MISS 112), e_ld_nc_xgpu_bulk (bulk writes in flight after wait_group.read; MISS 32, MISS 102); probes/nc_hoist and nc_pdl (GB300 stale words 200/200 each) | nc-coherent, nc-after-skip, nc-before-skip, nc-sass-ldg, nc-sass-256, nc-ptx-drop, nc-ptx-vec-drop, nc-pdl-ordered, nc-start-readclock, nc-start-gate, nc-start-waits, nc-start-sass, nc-start-profile (MISS 21; nc-done-off, nc-start-off: precision) |
| S-SAME-VALUE, per byte over a uniform shadow page | s_sv_page (a 512-byte bulk reload that keeps only the page's first byte: clean until 2026-11-11 with the read after it, PAGE, or concurrent with it, CONC; race now) | sv-page-onebyte (MISS 69) |
| S-SCOPE, complete overlap (PTX front) | e_scope_mixed (a u8 / u64 acquire of a u32 release flag: clean until 2026-11-11, MIXED / WIDE; a race now; VEC clean), e_warp_order_mixed (a u8 load of a u32 store in a converged warp: nondeterministic until 2026-11-11, MIXED; a race now), e_scope_mm_lane (a packed f16x2 multimem.st against an f16x2 multimem.ld_reduce: a race until 2026-11-11, good; nondeterministic now) | scope-overlap, scope-overlap-race, scope-overlap-sync, scope-elem-vector, scope-elem-mmst (precision) (MISS 80) |
| S-SCOPE, release sequences through read-modify-writes | e_rseq_scope (a .cta release continued by an atomic of another CTA: clean until 2026-11-11, CTAREL; a race now) | rseq-scope (MISS 100) |
| S-SCOPE, the first operation of a continued release pattern | e_rseq_head (a .cta fence heading the pattern, FREL), e_rseq_facq (a .cta acquire fence or acquire read after the atomic, CTAFENCE, CTAACQ): synchronized until 2026-11-11, races now; e_rseq_head TWOHEAD, OWN (one pattern fails, another passes; the patterns' join stands for them only when all pass), e_rseq_twopat CTAFENCE (a .cta acquire fence takes the pattern of its CTA only; the SASS front's one pending entry is SASS-only) | rseq-head-rel, rseq-head-acq, rseq-head-fence, rseq-head-join, rseq-head-cache, rseq-head-own, rseq-head-shortcut (MISS 112) |
| S-NIC-ATOMIC, release sequences through NIC atomics | e_nic_rseq (a .gpu release on the other GPU, good, and a .cta fence heading the pattern, FREL: the receiver's read of d2 races beside S-NIC-ATOMIC's unmodeled since 2026-11-12; FREL was unmodeled only; SYSREL, FREL_SYS: no race; NICHEAD, a NIC atomic heading the sequence and a .gpu acquire fence that leaves out its doorbell thread: clean on the PTX front until 2026-11-12, a race on dst now; SASS clean) | nic-rseq-moral, nic-rseq-join (MISS 166) |
| S-SCOPE, the NIC's scopes | e_nic_cta (a .cta or .cluster acquire in the doorbell thread's CTA, CTA and CLUSTER: clean on the PTX front until 2026-11-12, a race now; SASS: CTA a race, CLUSTER clean; GPU, SYS clean; OTHERCTA a race) | nic-scope-cta, nic-scope-cluster (MISS 193) |
| S-SCOPE, complete overlap with the NIC (PTX front) | e_nic_cta (a 4-byte poll of the fetch-add's 8-byte word, PART and PART_OTHERCTA, and of an RDMA write's 8-byte chunk, WORDPART: clean until 2026-11-12, races now; PART_SRC and FLAGHALF (a 4-byte poll of a lone 8-byte flag write), the partial poll acquires nothing: the read after it races; a 4-byte read of CQE bytes 60-63, CQEPART: clean until 2026-11-12, unmodeled now; WORDFULL, WORD4, FLAGFULL clean; CQECTA a race, not unmodeled), e_nic_rseq SYSREL_PART (a 4-byte head write the NIC atomic no longer continues: d2 races since 2026-11-12); e_nic_conv2 (states that differ only in the element a NIC access covers: CONV2_XY races, clean when the state hash leaves NicState::acc out), e_nic_sharedcq (two QPs' CQE writes on one CQE: one location, clean; unmodeled without the rule) | nic-overlap, nic-overlap-word4, nic-overlap-cqe, nic-overlap-scope, nic-overlap-rseq, scope-overlap-sync, acc-unhashed, cqe-rule-off (MISS 194) |
| E-RACE-DEAD, mixed-size data races (PTX front) | e_race_dead RDMIX (a byte load of a word another thread overwrites, the value dead: race-dead and clean on the PTX front until 2026-11-12, a race now; SASS race-dead, clean), e_race_dead good (uniform size: race-dead, clean, both fronts); probes/race_dead_mixed (good, PLAIN a race now; UNIFORM, UNIFORM_PLAIN race-dead, clean); e_nic_cta WORDDEAD (a masked 4-byte read of an RDMA write's 8-byte chunk) and CQEDEAD (a masked .cta read of CQE bytes whose overlap is not modeled): race-dead and clean until 2026-11-12, races now | race-dead-mixed, race-dead-uniform-off (precision), race-dead-mixed-hw (placement), race-dead-mixed-nic, race-dead-unmodeled (MISS 226) |
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
| (ptxas tcgen05 guardrail race) | e_tmem_alloc_block good, SASS trap: a true positive (probes/tmem_alloc_race null on unmodified code; probes/tmem_alloc_race_wide traps) | (none: no exemption) |
| L-OPCLASS | e_sb_mufu (binary-only; NOWAIT wrong on GB300 16/50); probes/frnd_queue (miss 82: FRND does not cover MUFU, GB300 wrong 1000/1000); probes/xu_queue, xu_queue_rev (miss 96: MUFU / I2F / F2F do not cover F2I, F2I does not cover MUFU; GB300 wrong 1000/1000 each) | opclass-nomufu, frnd-xu, xu-mufu |
| E-HW-SB-QUEUE | probes/redux_queue (REDUX in order: a strict `sb-hwq:` contract, discharged at gb300) | sb-hwq-strict |
| L-OPREAD | probes/utmastg_queue (miss 101: a TMA store's coordinate registers are reads; COVER a strict contract, NOWAIT a violation) | tma-ur-group |
| N-FADD | e_fadd_nan (GB300 bits) | fma-nan-payload |
| N-FCHK, N-XORSIGN, N-F2I16, N-F2FP-RZ, N-QMUL4, N-IMMA, L-SHF, L-LOP3-P | (their GB300 replay probes) | fchk-swap, xorsign-minmax, f2i16-zext, f2fprz-inf, qmul4-nan, imma-wrap, shf-wrap, lop3p-and |
| N-CVT-F16 (MISS 22) | probes/cvt_f16 (GB300 bits per form; `cvt_f16_unmodeled_forms_refused` rebuilds each refused form) | cvt-f32f16-types, cvt-f32bf16-types, cvt-rn-f16-mods, cvt-rn-bf16-mods, cvt-f16-sat-clamp, cvt-lp-mods |
| L-ROSETTA add / sub.sat.s32, SASS VIADD.16x2 d, a, -b; L-PTX-HINT integer modifiers (MISS 23) | probes/int_sat, probes/sub16x2 (GB300 bits; the PTX front refuses sub.{u,s}16x2), `int_arith_unmodeled_forms_refused`, `sass_sign_predicate_not_a_destination` | int-mods-gate, sub16x2-ptx-lowered, sub16x2-borrow, plop3-sign-bit, sign-pdst, addsat-wrap, subsat-wrap |
| PTX-FMAD mul.sat (MISS 24) | probes/fmad_sat (GB300 bits) | fmad-mul-sat |
| N-MULTIMEM `.acc` (MISS 25) | `multimem_acc_unmodeled_forms_refused` (fixtures/r1h_nvls rewritten) | mm-acc-f16, mm-acc-placement |
| N-CVT-F16 cvt integer path (MISS 59) | probes/cvt_bf16_same (GB300 bits, SASS; PTX refuses), `cvt_non_integer_types_refused` | cvt-int-prefix |
| N-CVT-LP `.satfinite` decodes (MISS 60) | probes/cvt_lp_satfinite (GB300 bits of the plain decode, PTX; both fronts refuse `.satfinite`) | cvt-lp-decode-sat |
| N-CVT-LP SASS `UNPACK_B` scale (MISS 61) | probes/cvt_lp_scaled (SASS refuses register scales; PTX refuses `.scaled`), probes/cvt_lp_satfinite `kernel_ptx.cubin` (the identity scale, GB300 bits) | unpack-b-scale |
| PTX-FMAD `mul.ftz` through `neg` (MISS 62) | probes/fmad_ftz_neg, probes/fmad_multi (GB300 bits, both fronts) | fmad-ftz-neg |
| L-ROSETTA / N-SEG `.SIGN` predicates read as bit 31 by every reader, PLOP3 inputs only (MISS 97) | fixtures/c_equiv_rsign (EW, not L0; GB300 bits), unit tests `sign_predicate_reads_bit_31` (seg), `constant_sign_predicate_reads_bit_31` (num), `const_guard_reads_sign_bit`, `lookahead_pred_reads_sign_bit` (explore), `sign_register_reads_bit_31` (elect_lanes), `plop3_sign_input_with_undefined_bit_31_is_undefined`, `plop3_sign_input_definedness_is_bit_31_of_its_mask`, `sass_sign_predicate_plop3_inputs_only` | sign-seg, sign-num, sign-constguard, sign-lacfg, sign-elect, sign-predundef, sign-predundef-mask, sign-parser |
| N-EQUIV / N-ELEMENTWISE captured GPU outputs refuse a contradicted level (MISS 98) | fixtures/c_equiv_captured (both modes), fixtures/c_equiv_atomic (the review's probes: float atomic adds exempt only the outputs that depend on their order; GB300 captures, one word changed), unproven levels (c_equiv_atomic eq_unproven.toml, `compare_unproven_level_contradicted_by_the_captures`), partial captures (c_equiv_atomic eq_offset.toml), `captured_outputs_against_the_level`, `captured_pairs_against_the_level`, `compare::tests::a_contradicted_run_is_kept`, the float-atomic exemption in `row_sums_are_bounded` | captured-refusal, captured-simgpu, captured-ew, captured-l0, captured-bound, captured-atomic-exempt, captured-atomic-whole, captured-atomic-chain, captured-nonfinite, captured-l0-nan, captured-unproven, compare-rerun-contradicted, captured-offset, captured-ignore |
| N-EQUIV L0 up to the sign of zero is its own level, L0~0, never L0 (MISS 109) | fixtures/c_equiv_zero_sign (x + (+0) against x: L0 fails, L0~0 holds; GB300 captures, +0 against -0 at 16 outputs, not a contradiction; b + a stays L0; captures one word changed refused), `compare_zero_sign_is_not_bitwise`, `captured_zero_signs_against_the_level`, `compare::tests::the_zero_sign_level_is_not_bitwise`, fixtures/c_equiv_mxf8_cublas, c_equiv_nvf4_cublas (`*_is_bitwise_up_to_zero_signs`) | l0z-relabel, l0z-strongest, l0z-prefix, l0z-headline, l0z-captured-strict, l0z-captured-any, l0z-captured-l0 |
| N-CVT-LP forms ptxas 13.0 and 13.3 reject, refused by the PTX front (T2) | `cvt_lp_forms_ptxas_rejects_refused` (lines of the cvt enumeration), `cvt_lp_operand_types_ptxas_rejects_refused` (lines with other operand register types) | lp-encode-rn-sat, lp-ue8m0-encode, lp-ue8m0-f16x2, lp-e2m1-byte, lp-x4-sat, lp-ue8m0-decode, lp-reg-types, lp-x4-rbits, lp-half-src, lp-decode-src |
| N-ATOMIC an fp32 atomic add's flush flag is part of the L0 form; a chain of both flush modes is unordered (MISS 134) | fixtures/c_equiv_atomic_ftz (red.add.f32 against a bulk add of the same value into +0, and both modes in opposite orders: not L0, the bound holds against GB300 captures that differ in 64 and 79 words on subnormal inputs; one mode in both orders stays L0), `atomic_flush_modes_are_not_bitwise`, `atomic_flush_modes_are_part_of_the_canonical_form`, `mixed_flush_atomic_adds_are_order_dependent` | atom-l0-noflag, atom-l0-mixed-sorted, atom-order-mixed-off |
| N-EQUIV L0~0 only where x + (+0) keeps x's NaNs (MISS 135) | fixtures/c_equiv_zero_sign_nan (a copy against the copy + 0: L2 with bound 0, not L0~0; GB300 captures with NaN payloads, which the + 0 makes 0x7fffffff), fixtures/c_equiv_zero_sign_choice (a guarded store of the input b or of the payload constant 0x7fc12345 over a + b, against it + 0: level none, no peel; GB300 captures), `zero_sign_needs_canonical_nans`, `zero_sign_choice_is_not_l0z`, `zero_peel_needs_canonical_nans` | l0z-nan-any (fixture); l0z-nan-choice, l0z-nan-leaf (unit test: l0z-nan-choice's L0~0 on max.f32(a, b) + 0 against max.f32(a, b) is true on GB300; l0z-nan-leaf changes no level) |
| N-ATOMIC / S-SWIN an f32 atomic add flushes subnormals in global memory only; the SASS generic f32 add is not performed at a shared-window address; QSPC of window addresses (MISS 142) | probes/atom_f32_space (shared, generic-to-shared, shared::cluster, into the other CTA of a cluster, global; GB300 bits, both fronts), probes/qspc_window (own, peer and global addresses; GB300 bits, SASS), fixtures/c_equiv_atomic_space (red.shared.add.f32 and a generic red.add.f32 into shared memory against red.global.add.f32 into +0: not L0 on the PTX front, the bound holds against GB300 captures that differ in 128 of 256 words; unproven on the SASS front), `probe_atom_f32_space_matches_hardware`, `probe_qspc_window_matches_hardware`, `shared_atomic_adds_do_not_flush`, `fadd_orders_in_shared_memory_keep_subnormals` | atom-shared-flush, atom-shared-numftz, atom-generic-shm-performed, qspc-cluster-own, qspc-cta-any, cfadd-shared-flush |
| N-EQUIV L2 compares the absolute term; only flushing fp32 atomic adds carry the flush term (MISS 143) | fixtures/c_equiv_atomic_ftz reversed pairs (the flushing kernel as the candidate against the bulk one: L1, not L2; as the reference: L2; GB300 captures within the bound), fixtures/c_equiv_atomic_space PTX pairs (L1), `l2_needs_the_flush_term`, `l2_compares_the_flush_term` | abs-ftz-all-adds, l2-abs-off |
| N-MMA-BS an f32 tensor-core step that leaves the f32 range is not modeled (MISS 144) | probes/mma_bs/ovf (mxf8, mxf4: GB300 +-Inf where a step overflows; unmodeled on both fronts), `probe_mma_bs_overflow_is_unmodeled` | mma-ovf-saturate, mma-ovf-unreported |
| ptxv compare's headline and per_output_bound carry the bound's absolute term (MISS 145) | fixtures/c_equiv_atomic_ftz ftz1_ref_s against blk1_cand_s through `ptxv compare` ("0 * sum\|terms\| + 3.5e-38", report.json `max_abs_extra`, report.md), `compare_headline_carries_the_absolute_term`, `the_headline_carries_the_absolute_term` | stats-abs-off, headline-abs-off |
| N-EQUIV a max keeps its kind in its L0 form (FMNMX, FMNMX.NAN, a compare and select's mask; a dropped FMNMX infinity); maxima share a value form only where both operands are non-NaN, relying on N-FINITE (MISS 197, 202) | probes/equiv_claims select_max, max_identity, fmnmx_nan (`run.sh`: GB300's recorded outputs against the claims; 18, 8 and 6 contradicted claims before, 0 after), `max_kinds_are_not_l0`, `maxima_of_maybe_nan_operands_share_no_value_form`, `max_kinds_are_part_of_the_l0_form`, `a_dropped_identity_is_part_of_the_l0_form`, `select_masks_mirror_and_pick`, `kinds_differ_only_at_signed_zeros_and_nans`, `maxima_evaluate_as_their_kind`, `row_maxima_read_other_kinds_only_where_finite`, `real_projection_rejects_polynomial_operands_with_non_finite_constants`, `models_of_earlier_versions_are_refused` | max-select-as-fmnmx, max-nan-flag-dropped, max-select-mask-off, max-identity-flag-off, max-value-share-any, max-value-reliance-off, max-identity-reliance-off, errdist-model-version-17, errdist-max-eval-fmnmx, softmax-max-any-kind, softmax-seg-max-any |
| N-REASSOC an f32 product or fma rounded to a subnormal carries 2^-150 absolute (none where the exact result is on the 2^-149 grid or the product is exact); an absolute term that a later product or an MMA operand scales leaves no bound, and neither L2 nor L3 except at an L0 or L0~0 output (MISS 201) | probes/equiv_claims product_underflow (`run.sh`: GB300's recorded outputs against the claims; 2 contradicted claims before, 0 after), `products_rounded_to_subnormals_have_no_bound`, `a_subnormal_product_rounding_carries_an_absolute_term`, `an_absolute_term_in_an_mma_operand_has_no_bound` | product-sub-term-off, product-sub-grid-off, abs-in-product-levels-off, abs-in-product-at-l0, mma-operand-abs-round-only, tf32-operand-abs-round-only |
| ptxv equiv's L3 line says "+ A" and gives A where L3 holds and the bound has an absolute term (MISS 143 residual); printed bounds round up and printed condition limits down, in ptxv equiv and ptxv compare (MISS 145 residual) | fixtures/c_equiv_atomic_ftz eq_tau_i_s (L3 at 4u: GB300's captured outputs differ by up to 1.165e-38 where 4u sum\|terms\| is 0, within A = 3.526e-38, printed 3.6e-38), `l3_line_carries_the_absolute_term`, `printed_bounds_round_outward`, `the_headline_carries_the_absolute_term`, `compare_headline_carries_the_absolute_term`, `f16_outputs_are_bounded`, `tf32_gemms_are_bounded` | l3-line-abs-off, printed-bound-nearest, printed-abs-nearest, printed-limit-nearest |
| N-EQUIV a hardware FMNMX / FMNMX3's max (the SASS front) is peeled outright for L0~0, a PTX max.f32 only where its operands are, since ptxas folds max.f32 of one value into a copy (MISS 135 follow-up) | fixtures/c_equiv_zero_sign_fmnmx (max(a, b) and max(max(a, b), c) against them + 0: L0~0 on SASS, L2 on PTX; max.f32(a, a) against it + 0: L2 on both; GB300 captures with NaN payloads: 16, 8 and 56 words differ, 0, 0 and 24 at NaNs), `zero_peel_takes_hardware_maxima`, `zero_peel_needs_canonical_nans` | l0z-fmnmx-hw-off, l0z-fmnmx-hw-any-front; l0z-nan-choice (now caught end to end: its L0~0 for the folded PTX pair is false on GB300) |

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
