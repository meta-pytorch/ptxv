# Targets: one directory per hardware target

ptxv verifies kernels for NVIDIA GB300 (`sm_100` / `sm_103`). This directory holds the
specifications and assumptions of every target, including targets that ptxv cannot check
yet. A target starts as a written model (what the hardware guarantees, what ptxv would have
to assume) and gains front-ends, probes and verdicts later.

## Layout

```
targets/
  README.md          this file: rules shared by every target
  _template/         copy this to start a target
  gb300/             the current target; its assumptions are the top-level ASSUMPTIONS.md
  h100/              NVIDIA Hopper (sm_90, sm_90a)
  a100/              NVIDIA Ampere (sm_80)
fb/targets/          internal-only targets (not exported), same layout
```

Each target directory contains:

| File | Content |
|---|---|
| `target.toml` | identity, ISA, compile targets, status, ID prefix (schema below) |
| `ASSUMPTIONS.md` | the target's assumption registry, in the format of the top-level ASSUMPTIONS.md |
| `SPEC.md` | the target's model: execution, memory, synchronization, numerics, ISA, toolchain |
| `probes/` | hardware probes that validate the target's assumptions, one subdirectory each |

## Working without conflicts

* Edit only your target's directory. A change to a shared file (this README, the template,
  `crates/targets`, the top-level ASSUMPTIONS.md) is its own diff.
* Assumption IDs carry the target's prefix: `<PREFIX>-<CLASS>-<NAME>`, for example
  `H100-S-WGMMA-ASYNC`. Classes are those of the top-level ASSUMPTIONS.md: `E` execution,
  `S` memory and synchronization, `L` lift, `N` numerics, `H` hardware testing. IDs are
  unique across all targets; `cargo test -p ptxv-targets` checks this.
* A target that differs from GB300 only in some respects says so: `based_on = "gb300"` in
  `target.toml`, and SPEC.md lists the differences instead of restating shared behavior.
  An assumption that holds unchanged on the base target is referenced by its base ID, not
  copied.
* Material that cannot be published (internal hardware, documents under NDA) goes in
  `fb/targets/<id>/`, never in `targets/`. `fb/` is not exported.

## target.toml

```toml
id = "h100"                       # = the directory name
name = "NVIDIA H100 (Hopper)"
vendor = "nvidia"
isa = ["ptx", "sass"]             # instruction sets a front-end reads
arch = ["sm_90", "sm_90a"]        # compile targets or chip variants
id_prefix = "H100"                # assumption ID prefix, upper case
status = "scaffold"               # see below
based_on = "gb300"                # optional: the target whose model this one extends

[[front]]                         # one per verdict level, in display order (at least one)
id = "ptx"                        # [a-z0-9_], unique in the target; records and findings use it
label = "PTX contract"            # the label tools and the dashboard show
kind = "contract"                 # contract | hardware | ir
description = "..."
```

## Fronts

A front is a level at which ptxv gives a verdict on a kernel: an ISA's contract (`contract`),
the binary as the hardware runs it (`hardware`) or a compiler IR (`ir`). Records and forum
findings name a target and give one verdict per front id; tools and the dashboard take the
labels from `target.toml`, never from names of their own. `cargo test -p ptxv-targets`
checks the fronts and that `targets/fronts.json` (public targets) and
`fb/targets/fronts.json` (all targets) match the target.toml files; regenerate both with
`PTXV_BLESS=1 cargo test -p ptxv-targets`. A new target needs only its directory.

Status, in order:

| Status | Meaning |
|---|---|
| `scaffold` | directory exists; no model yet |
| `spec` | SPEC.md and ASSUMPTIONS.md being written; no front-end |
| `lifting` | a front-end reads the target's binaries or IR into ptxv's IR |
| `checking` | ptxv produces verdicts on the target's fixtures |
| `verified` | assumptions validated on hardware; verdicts on real kernels |

## ASSUMPTIONS.md entries

Same rules as the top-level ASSUMPTIONS.md (status vocabulary, "If wrong"), with one more
kind: `spec`, an assumption written down before any code relies on it. When code starts
relying on it, the kind becomes `code` and the code carries `ASSUME(ID)`.

```
### H100-S-EXAMPLE · spec
**One-sentence statement of what is assumed.**
- Source: the document section, probe or reasoning it rests on.
- Status: **unvalidated** (YYYY-MM-DD)
- If wrong: miss | false alarm, and how.
```

Status is one of `enforced`, `validated`, `argued`, `refuted`, `unvalidated`. A `validated`
entry names its evidence (probe and hardware). Evidence from one target does not validate
another target's assumption.

## SPEC.md

Sections, in this order (copy from `_template/SPEC.md`):

1. Execution model: threads, groups (warps, cores, tiles), scheduling and forward progress.
2. Memory hierarchy and address spaces.
3. Memory model: ordering, scopes, coherence, proxies.
4. Synchronization primitives: barriers, async completion mechanisms.
5. Asynchronous engines: copy engines, tensor or matrix units, DMA.
6. Numerics: formats, rounding, accumulation order, special values.
7. ISA and binaries: what a front-end reads, disassembler and toolchain.
8. Differences from the base target (when `based_on` is set).
9. Open questions.

Each statement cites a source (public documentation section, probe, or "unknown").
