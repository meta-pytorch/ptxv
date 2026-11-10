# JSON output of the CLI

**Status (2026-11-04).** `check`, `equiv` and `compare` take `--json`; `equiv` and `compare` take
`--emit-bound FILE`. The Python package (docs/python.md) reads only these.

## `--json`

One JSON document, on one line, on stdout. Everything the command would print otherwise goes
to stderr, including the output of child processes (`compare` runs `equiv`). Exit codes do not
change: `check` exits 1 when a verdict misses its expectation, and the document is still
written. A command that fails (a case that does not load, a kernel that does not lift) writes
`{"error": "<message chain>"}` instead, and exits 1.

Every document has:

| field | meaning |
|---|---|
| `command` | `"check"`, `"equiv"` or `"compare"` |
| `schema` | 1. Bumped on an incompatible change; added fields keep it |
| `ptxv` | the crate version |

### `ptxv check CASE --json`

| field | meaning |
|---|---|
| `case`, `name` | the case directory and its `name` |
| `pass` | every row met its expectation |
| `rows[]` | one per variant and front-end |
| `rows[].variant`, `front` | e.g. `good`, `sass` |
| `rows[].expect`, `got`, `pass` | expected and obtained classes (`clean`, `race`, `deadlock`, `violation`, `fault`, `unmodeled`, `contract`, `nondeterministic`, `undef-insensitive`, `inconclusive`; `nonfinite-output` with `--nonfinite`). `clean` only for a complete verdict (exhaustive exploration or certificate); an incomplete search that found nothing is `inconclusive` (since 2026-11-07; older binaries said `clean` with `verdict.complete` false) |
| `rows[].verdict` | `states`, `transitions`, `complete`, `terminal_states`, `distinct_outputs`, `findings[]` (`kind`, `sig`, `msg`, `trace[]`), `assumptions[]` (ASSUMPTIONS.md IDs; `E-WELLSYNC` = certified), `notes[]`, `seconds` |
| `rows[].expect_report[]` | captured cases: each output against the GPU's bytes (`output 0: out: bitwise equal to the captured GPU output (2048 bytes)`) |
| `rows[].refuted_opt_in[]` | refuted assumptions (ASSUMPTIONS.md) the verdict relies on by explicit opt-in (`assume_refuted` / `--assume-refuted`), e.g. `["E-POLL-ACQ"]`; empty otherwise |
| `rows[].verdict.findings[].gb300` | races and stale reads, when the second level ran: `{class, reason, witness?}`, `class` = `observable` (reachable under the GB300 hardware model, E-HW-L1; `witness` = the hardware-model finding's `sig`), `latent` (a PTX-model bug not reachable on current GB300, even with outside fills; `reason` says why), `latent-closed-world` (not reachable in the closed-world hardware model, E-L1-CLOSED, but a weak read involved is stale if an agent outside the case fills its line, e.g. a legal co-resident kernel: observable on GB300 with such a neighbour; `witness` = the havoc check's `stale-l1-havoc:<pc>`, `reason` names E-L1-CLOSED) or `undecided` (the hardware-model check was not exhaustive). The classes in `got` do not change |
| `rows[].verdict.findings[].pcs[]` | instruction indices involved (races: earlier and later access; stale: read, fill, write), when known |
| `rows[].nonfinite` | with `--nonfinite` / `--nonfinite-strict` (docs/design.md 3.4): `strict`; `outputs[]` (per float output buffer, the terminal state with the most NaN / Inf: `buffer`, `dtype`, `elements`, `nan`, `inf`, `first` = [index, value]); `terminal_outputs`, `terminal_outputs_nonfinite`; `inputs[]` (float input buffers holding NaN / Inf, same fields); `births[]` (instructions that produced a NaN / Inf from operands without one, in program order: `at`, `pc`, `kind` = `inf from finite` / `nan from finite` / `nan from inf`, `thread`, `operands[]`, `result`, `count`) |
| `rows[].gb300` | present when the second level ran (`ptxv check` runs it whenever the verdict has a race / stale finding or relies on a refuted assumption; `--no-gb300` / `PTXV_GB300=0` turn it off): `model`, `why`, `states`, `complete`, `seconds`, `observable`, `latent`, `latent_closed_world`, `undecided`, `extra[]` (stale reads only the hardware model reports: `gb300.class` `observable`, or `latent-closed-world` for a read only the havoc probe flags, e.g. when level 1 relied on a refuted assumption; findings as above) |
| `rows[].two_level` | present when the strict check reported anything but clean / unmodeled and a hardware profile is on (`--hw-profile gb300`, the default; `none` / `PTXV_HW_PROFILE=none` turn it off; ASSUMPTIONS.md H-PROFILE): `ptx_contract` (the row's classes, strict model), `contract_front`, `sass_gb300` (`certified` / `clean` / `incomplete` / the SASS front's remaining classes under the profile), `hw_profile`, `hw_assumptions[]` (`{id, probe, discharged[]}`: the profile assumptions the second level used, with the contract finding signatures each discharged), `sass_classes`, `sass_complete`, `sass_certified`, `sass_states`, `sass_findings[]`, `sass_notes[]`, `seconds`. The row's `got` / `pass` are the strict check's |
| `rows[].input_coverage` | input coverage (`crates/sim/src/coverage.rs`; off with `--no-input-coverage` / `PTXV_NO_INPUT_COVERAGE`): one deterministic run of the captured input with a provenance shadow. `sites`: comparisons (`setp` / ISETP / FSETP / DSETP) whose operands derive from input bytes; `untaken[]` (`{pc, at, never, sources, inputs, note, refuse?}`; `inputs[]`: `{buffer, bytes}`, single-word inputs first with `bytes` = `[first, last]` byte offsets read, then buffers the operands derive from otherwise with `bytes` null): those where no thread produced the predicate value `never`, so that side was not exercised on this input (one schedule). `comparisons[]`: `{pc, at, n_true, n_false, inputs}` per input-dependent comparison (thread observations per outcome; any instruction that writes a predicate from input data counts, FCHK included; predicate logic of other sites does not). `input_dependent[]`: `{pc, at, what, inputs}`, instructions other than register arithmetic whose guard (`what` "guard") or address, control or synchronization operand ("operand") derived from input data. `input_independent`: the run finished and `input_dependent` is empty: control path, accesses and synchronization are the same for every input of this shape with these parameters (one schedule's provenance; value-dependent parts of the verdict, such as same-value writes, determinism and the comparison with the GPU output, hold for the captured input only). `unexecuted[]`: `{first, last, first_at, last_at, first_loc, last_loc, kernel, instructions, gate_pc, gate_at, gate}` (`*_loc`: PTX line or SASS address) runs of instructions no thread reached (a predicated instruction whose guard was never true included), with the reached branch or predicated instruction whose untaken side leads there through unreached code; `gate` "input" (its condition derives from input data), "parameter" (it does not), "divergence" (BRA.DIV / BRA.CONV, ptxas's fallback for a warp that is not converged) or "none"; totals `unexecuted_instructions`, `behind_input`, `behind_parameter`. `stopped` (absent when the run finished): "time" or "memory" when the run passed its own cap (`--coverage-seconds` / `PTXV_COVERAGE_SECONDS`, default 1800 s, none with `--coverage-only`; `PTXV_COVERAGE_MAX_RSS`, default 75% of the memory budget): then only `input_dependent` and the comparisons seen so far are reported, nothing is claimed untaken or unexecuted, and `input_independent` is false. On the plain `--certify` path the shadow rides on the certificate's first run (`PTXV_COVERAGE_SEPARATE=1`: a run of its own before the check); elsewhere it is a run of its own before the exploration (docs/input-coverage-cost.md). A note: not part of the verdict, `got` or `pass`; `ptxv compare` does not run it. `--coverage-only`: this run alone, no exploration (`got` = `["coverage-only"]`) |
| `rows[].mutated_inputs[]` | `--mutate-inputs N` (`--mutate-elements K`, default 2): for up to N untaken sides whose comparison reads one input word against a value that does not depend on the input, the check on the input with K elements (threads of different warps first) set to a flipping value: `{mutation ("ok[0]=0, ok[32]=0"), pc, within_invariants, at, makes, classes, complete, certified, states, findings[], notes[], skipped?, seconds}`. `skipped`: not checked, and why: the deterministic run on the mutated input (stopped as soon as the comparison takes the side it was to make taken) finished without taking it and reached exactly the instructions of the captured input's run (`PTXV_MUTATE_NO_SKIP=1` checks it anyway). Budget: `--mutate-budget-seconds` (default 1800) and `--mutate-budget-states` (default 2 x `--max-states`) for all mutated checks together; each gets the states left (at most `--max-states`), an exploration stops at the time budget (note "exploration stopped at the time budget"), and no check starts after either is spent (`mutation_budget`). Evidence class `on-mutated-input`: a mutated input may be one the producer cannot generate, so these are candidates for triage, never part of the verdict on the captured input. One input word: a load of the word, a byte or halfword of a wider load (PRMT / BFE / AND / shift), or the old value of an atomic add of a non-negative constant (compared monotonically; the flipping value is the counter's initial value). `--mutate-input 'name[i]=v,...'` (repeatable) checks a mutation given by hand the same way; `pc` is null for it. Declared input invariants (case `buffers[].invariant = { values = [...], min = .., max = .. }`, integer buffers) are checked on the captured bytes (a printed line per invariant; one that does not hold is not used); B then writes only values they admit, and `within_invariants` (printed `on-mutated-input (within declared invariants)`) says every element written lies in a buffer whose invariant admits it: the mutated input is one the declared domain allows |
| `rows[].mutation_budget` | absent unless the `--mutate-inputs` budget stopped the mutated checks: how many were not checked and which budget was spent |
| `rows[].mutation_refusals[]` | `{pc, reason}`: untaken sides `--mutate-inputs` did not mutate (an operand derived from input data other than as one word: arithmetic, an atomic result other than the old value of an add of a constant, an input-dependent address, a shuffle, TMEM; both operands input words; a 64-bit ISETP.EX pair; an input-derived combining predicate; a float comparison) |

### `ptxv equiv FILE --json`

| field | meaning |
|---|---|
| `file`, `ok` | the equiv.toml; every expectation in it held |
| `bound_file` | the `--emit-bound` path, or null |
| `candidates[]` | exactly the entries of `<out>/equiv.json` (`strongest`, `proven`, `report`, `captured`, `witness`, ...; docs/compare.md) |

### `ptxv compare REF CAND --json`

`<out>/report.json` (docs/compare.md, "The report"), plus `report_dir`.

## `--emit-bound FILE`: the bound as data

`equiv` writes one document per candidate (a bare object for one candidate, else an array);
`compare` writes the one of the equiv run it kept. Format `"ptxv-bound"`, version 1. It holds
what `OutBound::bound_at` (crates/sim/src/equiv.rs) needs to evaluate the proven bound τ(x)
on any input, for the forms that have an exact closed form.

| field | meaning |
|---|---|
| `format`, `version` | `"ptxv-bound"`, 1 |
| `u` | 2^-24 |
| `reference`, `candidate`, `level`, `proven` | as in equiv.json |
| `output` | `{name, count, buffer: {dtype, shape}}`: the case's first output, whose elements are the outputs below |
| `inputs[]` | `{name, buffer: {dtype, shape}}`; a leaf's `buf` indexes this list |
| `untracked` | non-null: outputs without dataflow history; no output has a bound |
| `per_output` | arrays of length `count`: `level`, `form`, `bound_u`, `abs_extra`, `round_rel`, `no_bound` |
| `terms` | `sum` and `round` forms: arrays `out`, `g`, `coef`, `deg`, and the leaves `leaf_buf`, `leaf_idx` (term t owns the next `deg[t]` leaves). Terms are grouped by output |
| `softmax` | arrays `out`, `row`, `leaf_buf`, `leaf_idx` (the output's own leaf), `rel_u`, `add_u`, and `rows[]` (`leaf_buf`, `leaf_idx`: the softmax row) |

Leaves are elements of input buffers: element index for f32 / bf16 / f16 buffers. Leaves of
byte buffers (FP8 / FP4 codes) carry a format tag in the index (`equiv::leaf_tagged`) and are
not evaluable from this file.

Per output, by `form`:

| form | τ(x) |
|---|---|
| `sum` | (Σ_t \|Π leaves of t\| g_t + abs_extra) (1 + 1e-12) |
| `round` | (round_rel \|y\| + Σ_t \|Π\| g_t + abs_extra) (1 + 1e-12), y = Σ_t coef_t Π leaves of t. The compact 16-bit form, valid for every input; `bound_at` uses the tighter gap between the two roundings |
| `softmax` | ((rel_u y + add_u) u + abs_extra)(1 + 1e-9) + 1e-300, y = exp(x_i − max) / Σ_j exp(x_j − max) over the row |
| `none` | no bound (`no_bound` says why) |
| `opaque` | a bound exists (norm, attention forms) but its evaluation data is not exported; `bound_u` is its input-independent coefficient |

An L0 output has `form = "sum"` and evaluates to its `abs_extra` (0). Evaluated in f64, as
`bound_at` does; the slack factors cover the f64 rounding of the evaluation.
`python/ptxv/bound.py` is the reference evaluator, and its tests check it against `equiv`'s
own evaluation at the captured inputs (`captured.bound_abs`): equal for the sum form, within
1e-14 relative for the softmax form.
