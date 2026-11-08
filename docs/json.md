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
