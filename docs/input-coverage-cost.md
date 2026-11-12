# Input coverage: cost and budgets

Status: implemented (2026-11-09). Code: `crates/sim/src/coverage.rs` (provenance shadow, A),
`crates/cli/src/main.rs` (ordering, budgets, B), `crates/sim/src/membudget.rs` (time budget).
Assumption: E-INPUT-INDEP.

## 1. Cost of input coverage (A)

Input coverage is one deterministic run with a provenance shadow (`coverage::analyze`). It ran
after the check, as a second run of the launch, and on a corpus sample it cost more than the
check itself: median +207%, p90 +7157% wall over the same check without it (section 3).

Profiles named three causes:

- **Copies of the machine copied the shadow.** The simulator copies its machine for
  look-aheads: the certificate's peek walks (`wellsync::skip_equiv`, `walk_dead`), the
  poll-loop trials (`spin::wpoll_dry`, `Machine::split_enabled`), the parity check of retry
  stubs, the value-insensitive poll trials, the C-RING / C-POLL / CAS replays and the poll
  records. Each copy cloned the whole shadow: a per-byte hash map of every memory byte written.
  On `cute.FusedRMSNormBwd` 92% of A's time was that memcpy (A 1268 s, the check 4.8 s).
  Copies now get an inert placeholder (`ProvState::clone`): a look-ahead's instructions are
  not part of the recorded run. The recorded run's machine must keep the live shadow: the run
  loops check it after every step in debug builds (`coverage::live_shadow`), and a run that
  ends on a copy reports nothing (`stopped = "copy"`; tests panic, `coverage::set_strict`).
  Every copy site found by tracing A on the 219 fixtures and 41 corpus records takes the
  machine by shared reference and returns derived values only (a bool, clocks, a result), so
  no copy can become the run's state.
- **Per-byte hash maps with SipHash.** Registers are now dense per-thread vectors and memory
  is paged; each value is one packed word (`Packed`: a single input word, a mask of
  allocations 0..61, or an index into a table of the rest; exact, every value unpacks to the
  one packed). The union of every written byte's provenance, which an MMA reads, is a count per
  allocation bit instead of a scan of all memory. Per-instruction facts (`Info`) are computed
  once per pc; per-thread scratch (bytes read, store data) are vectors indexed by thread,
  reused across instructions; the warp-collective union is computed once per instruction, not
  once per lane. Register arithmetic of a thread whose registers are all clean is skipped (it
  leaves them clean). Exited threads drop their register shadow.
- **A second run.** On the plain `--certify` path the shadow now rides on the certificate's
  first run (`coverage::tap_request` / `tap_attach` / `tap_finish`), the deterministic run the
  certificate makes anyway, instead of a run of its own. Elsewhere (`--slices`, `--groups`,
  `--rep`, DPOR, residency / co-tenant / NIC launches, `PTXV_COVERAGE_SEPARATE=1`) A is a run of
  its own before the check.

**Why riding on the certificate's run reports the same thing.** The certificate's first run
uses the same deterministic scheduler (`DetSched`) and the same fault-loop stop as
`explore::run_once`, on the same launch. Its machine differs in what it logs (the sync log,
the C-RING control without forced reads, the E-RACE-DEAD read log) and, on the SASS front, in
LDS/STS being plain (`smem::weak`), which changes no value or control flow. A run that stops
for a reason `run_once` would not stop for (a cluster launch control choice, the eager-launch
failure of E-CLC) or a launch with cluster launch control is not used: A then runs on its own
after the check. Differential: identical reports on every fixture (section 3).

**Order and caps.** A comes before any exploration: riding on the certificate's run, or as a
run of its own before the check. It has its own caps: `--coverage-seconds` /
`PTXV_COVERAGE_SECONDS` (default 1800 s; none with `--coverage-only`; with `--time-budget`, at
most what is left of the whole check's budget) and
`PTXV_COVERAGE_MAX_RSS` (default 75% of the memory budget, `membudget.rs`). Past either, a run
of its own stops and a riding shadow detaches (the run goes on without it); the report says
`stopped` ("time" / "memory"), keeps only the input-dependent operands and comparisons seen so
far, claims nothing untaken or unexecuted, and is not input-independent. The verdict is never
affected: the memory the shadow used is returned to the OS before the exploration
(`mi_collect`), so the exploration's memory budget starts where it would have.

## 2. Checks on mutated inputs (B)

- **Budget.** `--mutate-budget-seconds` (default 1800) and `--mutate-budget-states` (default
  2 x `--max-states`) are totals over all mutated checks. Each check gets the states left (at
  most `--max-states`); an exploration stops at the time budget, incomplete, with the note
  "exploration stopped at the time budget" (the synchronization reduction's seed walks before
  it stop there too, with a note of their own); no check starts once either is spent, and the
  row says how many were not checked (`mutation_budget`). Nested in `--time-budget` (the
  whole check's): the mutated checks stop at the earlier end, and the note names the flag
  whose end it was; the certificates of mutated checks stop at `--time-budget` only.
- **Reuse.** A mutated machine is the base machine with the mutated bytes (`coverage::apply`):
  the decoded and lifted program and the case setup are the base run's. The synchronization
  reduction's learned sets are not carried over: a mutated check that certifies (the common
  case in the corpus) makes no exploration, and seeding an exploration with another input's
  sets only adds footprints (sound, but no saving where the certificate decides).
- **Skip.** Before a mutated check, the deterministic run on the mutated input runs with the
  provenance shadow and stops as soon as the mutated comparison takes the side the mutation was
  to make taken (usually early). If the run finishes without taking it and reaches exactly the
  instructions the captured input's run reached, the mutation opens no code on that run, and
  the check is skipped (`skipped` in the row; `PTXV_MUTATE_NO_SKIP=1` checks it anyway). B is
  path evidence, apart from the verdict; a skipped mutation is one whose only difference on
  that run is values on code already checked.

## 3. Measurements

Corpus sample: 41 records spanning small, large and timeout-prone launches (`ptxv check
--certify` at each record's flags), one node per batch, `PTXV_THREADS=8`, 10 tasks in
parallel. Wall from process start to exit, peak RSS from `wait4`. Overhead: (with - without) /
without, `without` floored at 1 s. Before: the parent revision; after: this change.

| default check (A on) vs `--no-input-coverage` | wall median | p90 | max | RSS median | p90 | max |
|---|---|---|---|---|---|---|
| before (40 of 41; one did not finish within 4 h) | +207% | +7157% | +26402% | +0.0% | +75% | +197% |
| after (41 of 41) | +8.0% | +19.2% | +34.5% | +0.0% | +7.1% | +38.6% |

- After, `--coverage-only` vs `--no-input-coverage`: wall median -1.5%, p90 +4.4%, max +23%.
- After, the 21 records whose check without A takes 5 s or more: wall median +10.9%, p90
  +25.4% (A's share grows with the run; the shorter ones are dominated by start-up).
- `--no-input-coverage`, after vs before: wall median -0.5%, max +3.0%; RSS max +1.2%.
- Verdicts (classes, states, finding signatures) identical with and without A: 41 of 41.
- RSS maximum: a PTX checksum launch of 640,000 threads (31.3 GB vs 22.5 GB). Every thread is
  live at once and each keeps a register shadow up to its highest input-derived virtual
  register. The next largest is +22%.
- Largest wall gain: `cute.FusedRMSNormBwd` 257x5120, default check 1231 s before, 6.7 s
  after (`--no-input-coverage` 5.6 s).

Identity of the report. `fixtures.rs`
`coverage_same_without_shadow_in_look_ahead_copies`: on every fixture, A's report (sites,
untaken sides, input-dependent operands, unexecuted ranges, everything but `seconds`) is the
same with copies carrying the full shadow, with inert copies, and riding on the certificate's
run. Corpus, CLI: `--coverage-only` before vs after, 219 of 219 fixture launches identical;
riding vs a run of its own, 210 identical, 9 not compared (the full check timed out).
