# Strict memory-safety verification

The memory-safety result enabled by `--strict` or `--strict-memory-safety` starts
from raw PTX bytes, an entry name, and a typed manifest. A `safe` result implies memory safety of every
execution of the admitted program in both the independent mathematical memory
model and the shared native `ExecutionMachine` owner, for all initial memory payloads
and legal nonaliasing allocation placements with the same shapes and definedness.
Unsupported semantics and an exhausted search are inconclusive.

## Exported theorem

The strongest endpoint is
`MemorySafetyExecution.Source.source_check_reports_safe_every_source_program`.
Actual source checking followed by the actual reported status `"safe"` implies
the guarantees below for every program admitted by the independent grammar and
manifest construction, and every payload/placement instance of each program.
Whole-module uniqueness connects all grammar derivations; no preferred parse or
supplied parser correspondence is assumed. The conclusion also supplies an
admitted program, so the universal claim is nonvacuous.

`MemorySafetyExecution.Source.source_check_is_safe_full_memory_safety` connects
actual extracted Rust acceptance to the independent `SourceMemorySafe` property.
Its only acceptance premises are that the actual source checker returned the
report and the actual `Report::is_safe` accessor returned true. The candidate
producer is arbitrary: the checker recomputes initial state, every actor's
successors, structural equality, and certificate closure. The equivalent
`source_check_reports_safe_full_memory_safety` endpoint starts from the actual
`Report::status_name` returning `"safe"`.

`source_check_is_safe_instruction_spans` additionally exposes the bounds,
alignment, lifetime and initialization obligations for the actual instruction
fetched at every reachable running actor, without a supplied instruction or
evaluation premise. The
`source_check_reports_safe_every_source_execution` corollary covers every pair
of accesses in any unbounded execution of any such instance, with no termination
premise.

The conclusion includes:

- Independent whole-source grammar and instruction correspondence, including
  every input byte, global initializer, register declaration, guard, label and
  instruction position.
- Actual manifest admission and complete three-dimensional actor inventory.
- Successful construction of the native strict `Machine`, with all enabled
  steps and all schedules covered, including callbacks passed to its API.
- Full state correspondence to an independent interpreter, including jump
  targets, predicate/register writes, signed little-endian loads and every
  stored byte. Every independent trace is matched by a checked native trace.
- Bounds, alignment, nonwrapping addresses, allocation lifetime, and initialized
  reads for each concrete native byte access; exact store frames and explicit
  load/store access occurrence, including dead loads and same-value stores.
- No conflicting pair of accesses by distinct actors in any execution history.
- Live, initialized declared output allocations on every terminal execution.
- The same guarantees for every `MemoryInstance`: identical code, outputs,
  actor state, allocation shapes, lifetime and definedness, with arbitrary
  payload bytes and legal nonaliasing bases.

The original typed-IR theorem
`MemorySafetyIR.reported_safe_launch_memory_safe` is retained. It is a narrower
route with its own independent launch and instruction correspondence.

## Admission scope

The raw source route accepts the scalar integer subset described by
`FrontendProgram.Grammar`: global loads and stores of 1, 2, 4 or 8 bytes,
register moves, selected integer arithmetic, comparisons, predicates, branches,
thread/launch identifiers and exits. Register-based addresses retain
object provenance. Source headers, instruction modifiers and launch features
must match the supported forms exactly. Every source byte, including comments,
must be printable ASCII (32–126) or whitespace (9–13). Raw `nop`, indexed bracket expressions,
unsupported operand forms, reserved symbols and ambiguous builtin declarations
are refused. Declared predicate names must use `%p` followed by zero or more
ASCII decimal digits. Counted declaration stems in both register banks must end
with a nondigit; ordinary numbered register names remain supported.

Input-dependent payload arithmetic is allowed. Input-dependent guards or
addresses are refused, including guards captured as false. Pointer
scalarization, pointer comparisons, provenance-destroying bit operations and
pointer stores are refused. Every memory byte has separate definedness and
dependency state. These restrictions make the execution simulation valid for
all initial payloads and placements.

Atomics, barriers, shared memory, asynchronous operations, calls, parameters,
alias mappings, external actors, alternate hardware memory models and other
unsupported features are inconclusive. `SupportedManifest` specifies the exact
manifest exclusions. Global literals and supported manifest buffers determine
the reference initialization; the universal theorem additionally permits
arbitrary bytes with the same initialization flags.

The existing `clean` and `certified` results do not imply this separate strict
memory-safety result. Legacy execution remains available through
`Machine`; its verdicts are not covered by the raw strict theorem.

## Proof boundaries

- `FrontendProgram`: independent raw syntax and manifest construction, refined
  by the actual extracted parser and constructor.
- `ValueFlow` and `ValueFlowDependencies`: actual instruction evaluation,
  register/storage semantics and dependency transfer.
- `MemorySafetyExecution.Native`: actual prepare/commit and owned `ExecutionMachine`
  semantics, exact byte effects, access completeness, all-input and placement
  simulation, trace safety and terminal observations. Its independent interpreter
  uses mathematical lists and bit vectors, without calling Rust execution helpers.
- `MemorySafetyExecution.SourceSoundness` and `SourceUniversal`: reported raw
  acceptance composed with independent grammar uniqueness, construction and
  universal native and mathematical execution safety properties.
- `MemorySafety`: independent memory/event specification and checker refinement.
- `MemorySafetyIR`: retained typed-IR launch theorem.
- `MemorySafety/Code`: combined extraction of the actual Rust paths, with a
  deterministic license comment and unchanged generated bodies (see
  [EXTRACTION.md](EXTRACTION.md)).

## Trust boundary and reproduction

The theorem starts from raw PTX bytes and the typed manifest supplied to the
checker. Manifest deserialization, file loading, compiler transformations, CUDA
and hardware execution, and concurrent host/other-kernel accesses outside the
admitted launch remain outside the theorem. Memory safety does not require
termination. A cap can make a safe program inconclusive.

The qualified Cargo release build enables overflow checks, matching the checked
arithmetic of the extraction. PTX operations that wrap use explicit wrapping
operations. Changing compiler flags, Cargo profiles or enabled features requires
qualifying the resulting executable again.

Rust-to-Lean extraction, its recorded primitive models, and Lean's kernel remain
trusted. No project axioms, `sorry`, evaluator oracles, supplied parser
correspondence or trusted exploration-completeness flags are used. The audit
permits only `propext`, `Classical.choice`, and `Quot.sound`.

[EXTRACTION.md](EXTRACTION.md) documents the offline freshness check, staged build,
axiom audit, direct kernel replay and genuine extraction recipe. The formal gate
reruns extraction, checks the reviewed endpoint types, and verifies the cached
backend sources and compiled environments. A separate reviewed specification
policy prevents ordinary resealing from silently weakening the safety definitions.
Relative manifests bind the Rust
closure, acceptance route, primitive models, generated code and authored proofs
to exact hashes.
