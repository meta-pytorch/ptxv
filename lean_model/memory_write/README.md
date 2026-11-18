# Memory-write effects and atomic execution

A Lean implementation of centralized byte/history writes and atomic instruction
transitions, with a dependency-free Rust workspace crate extracted by Aeneas and
proved to refine the model. This Lean
4.35/Aeneas package sits beside the existing Lean 4.34 numerical models.

## Interface

| Module | Responsibility |
| --- | --- |
| `Module` | Checked writes and captured/deferred async completion update storage and both written sets together; CAS returns old bytes and models numeric-only effects on failure |
| `Production` | Address normalization, permissions, allocation bounds, local growth, TMEM allocation/free/poison, typed origin interpretation |
| `Adapter` | Raw production planes, packed local values/undefined masks, TMEM word keys, initial half/f32 fallback and normalized state representation |
| `Storage` | Separate concrete bytes, origins, taint, definedness, written markers and numeric histories; uniform/dense Pages import; checked reads and sparse storage |
| `Effects` | Central write, complete/partial numeric history updates, batches, immutable pending payloads and completion tickets |
| `Transfers` | Overlapping copies, scatter, zero fill, bulk/TMA transfers, captured cp.async including zero tails, completion-time reads, exchange and CAS |
| `Model`, `Representation` | Byte/history semantics and representation across all admissible inputs |
| `Source`, `Theorem`, `Snapshot` | Refinement and termination proofs for the freshly extracted Rust write/snapshot kernel |
| `RustSpan`, `RustCapture`, `RustWords`, `RustRefinement`, `RustEngine`, `RustNumeric`, `RustCopies` | Actual Rust span checks, ordered capture, word overrides, checked transitions, numeric-only forget and independent markers refine the Lean implementation |
| `RustAtomic` | Extracted atomic numeric footprints invalidate exactly the intersecting four-byte histories and mark exactly the touched numeric words; failed CAS selects only the exact numeric key |
| `AtomicInteger` | Integer atomic results refine mathematical wrapping arithmetic, signed ordering, bounded increment/decrement, and width-normalized CAS |
| `AtomicExecution`, `AtomicRefinement` | Complete scalar operation dispatch, typed width rejection, floating callback routing, packed lanes, CAS outcomes, returns, predicates, epoch decisions, and derived memory footprints |
| `AtomicBytes` | Actual byte decoding and encoding use the little-endian word model |
| `AtomicComponentOrder`, `AtomicComponentPlan`, `AtomicComponentTrace` | Independent component choices, exact retirement, frozen operands and arbitrary prefixes preserving returned observations and future enabledness |
| `RustAtomicComponentPlan`, `RustAtomicComponentTrace` | Source-used vector planner, choice enumeration, immutable component requests and ticket eligibility refine the component model |
| `AtomicHistorySafety`, `AsyncHistorySafety`, `CheckedControlSafety`, `WriteConsumerSafety`, `MarkerConsumerSafety` | Source-to-consumer consequences for historical stale-history, snapshot, guard, captured-zero, and fallback bugs |
| `Producers`, `Consumers` | Constructive literal/register payload soundness and safe origin/history/untainted/initial-memory consumers |
| `SymbolicState`, `SymbolicStateRefinement` | All-input register metadata transfers, frozen operands, stable node meanings, namespace/ID safety, and derived memory payload/control contracts |
| `ExecutionReduction`, `ExecutionReductionRefinement` | Extracted concrete poll replay audits preserve supplied register/store observations and read sequences; universal instruction/schedule preservation remains outside this certificate |
| `StoreWord`, `StoreWordRefinement` | Composes frozen symbolic operands, the actual Rust request constructor, and the actual memory engine into one four-byte store transition |
| `WellFormed`, `AsyncSafety` | Inventory and progress invariants, fresh tickets, at-most-once completion, and generation leases preventing free/reuse ABA |
| `Footprint`, `Commutation` | Exact byte/history footprints and observational commutation for independent prepared effects |
| `MachineRegisters`, `MachineRegistersWords`, `MachineRegistersUniform`, `MachineRegistersVirtual`, `MachineRegistersCoupling` | Owned register updates, packed undefinedness, sparse virtual masks, and uniform-register effect coupling |
| `MachineRegistersPoison`, `MachineRegistersFill` | Value-preserving poisoning and initialization that preserves definedness and already-written values |
| `MachineRegistersVirtualMap`, `MachineRegistersDefinedness` | Canonical sparse masks, exact key lookups, selected-bit semantics, and the composed virtual-register contract |
| `MachineControl`, `MachineWarp` | Thread control transitions and owned warp convergence state |
| `MachineRecorderModel`, `MachineRecorderRefinement`, `MachineRecorderFinish`, `MachineRecorderObservation` | Action lifecycle, typed delivery and direct/buffered observations |
| `MachineFootprint`, `MachineFootprintMerge`, `MachineFootprintContexts` | Actual footprint storage, range updates, summary merging and context reads |
| `MachineActionModel`, `MachineActionBasic`, `MachineActionFinish`, `MachineActionMutations`, `MachineActionContracts` | Owned facade refinement, snapshot/frame properties and mutation/recording composition |

The normalized address includes space, allocation and byte offset. TMEM numeric
and taint word keys are distinguished from its half-origin byte keys. Origins
retain Rust's `Zero`, `E`, `N` and `E7` constructors. Numeric histories distinguish
known rounded values from opaque nodes, whose presence makes no unjustified
claim about their value.

The [coverage inventory](../../probes/memory_write/coverage.json) maps 21 Rust
write/metadata families to their centralized effects and records source hashes.
Store, arithmetic, matrix, remote and NIC producers supply result bytes and
metadata through the same checked write interface. Their arithmetic is outside
this module. Copies select their own source spans using checked reads; callers
do not supply a pre-proved destination transition or an arbitrary source slice.

## Theorem

`rust_checked_write_refines` proves that the extracted Rust checked operation
terminates and exactly matches `applyEffect`, including errors and unchanged
state on rejection. `rust_checked_write_preserves_representation` transports the
existing semantic theorem through that refinement. Source capture derives its
payload certificate from represented actual source bytes. The engine proof adds
both independent written sets; finite-vector growth bounds are explicit.

The production crate is `crates/memory-write`. The probe manifest compiles that
same source through its `[lib]` path, so extraction cannot drift to a separate
reference copy. The generic `Machine` atomic path uses its execution and footprint
helpers. Eligible ordinary four-byte stores use a bounded production adapter
to commit bytes and metadata together; other writer dispatch still needs
migration into the checked engine. [STORE_INTERFACE.md](STORE_INTERFACE.md)
describes the shared contracts, production ownership, and remaining adapters.
Rust validates live inventory and writer class;
`MemoryLayout` capacities, allocation permissions and local growth remain the
allocator/caller boundary.

## Machine state ownership

Native `Thread` and `Warp` own the extracted core state directly. Read-only
dereferencing exposes it to existing readers; mutation goes through the core
operations. There is no copied state to reconcile after execution.

`source_write_virtual_contract` proves that the actual Rust write stores the
requested value, sets exactly its written bit, preserves canonical sparse masks,
and gives the requested partial mask at that key. Every other register's value,
written bit, and sparse-mask lookup is unchanged. Bounds and available vector
capacity are explicit preconditions, alongside canonical input masks.

`LaneRegisters` holds values, predicate values and definedness, written bits,
and sparse partial-undefined masks. `WarpRegisters` holds uniform values and
their definedness. Scalar writes preserve the packed 32-bit value/mask format,
register-pair boundaries, and constant-register behavior. Uniform mutations
return their space-17 typed resource access in the same operation that changes
storage. The fused action facade routes that access through the owned recorder
and returns the slice observation to the native sink exactly once.
Sparse-mask removal uses an indexed copy into a fresh vector because the cached
extraction backend does not support `Vec::remove`. Removing from a table with
multiple entries therefore adds an allocation; removing its sole entry does not.

`ThreadCore<St>` owns the program counter, status, register storage, and pending
barrier-acquire key. The native executor calls `apply_thread_flow`, whose actual
`St::Run` comparison is included in selected-Sim extraction. Barrier release and
acquire consumption use the same owned state. Existing program-counter recording
sites remain responsible for scheduling dependencies.

`WarpCore` owns uniform registers, convergence masks, and the async commit
sequence. Mask writes return the matching space-19 resource access. BMOV also
retains its existing access before operand evaluation, preserving fault-prefix
behavior. Async sequence advancement does not introduce a new scheduling
resource. Numeric-history notifications, race clocks, and instruction dispatch
remain native adapters.

## Machine action recording

`ActionState` owns the actual boxed `Footprint` and recorder trace. Machine reads
that state through read-only dereferencing; begin, recording, finish and footprint
replacement go through the extracted facade. Context comes from the owned
footprint: each record reads the mode in constant time, and finish scans raw
membership ranges once. Snapshots copy the footprint and start with no unfinished
trace. The native `SFoot` atomic sink consumes the facade's returned access.

`MachineRecorderRefinement` and `MachineRecorderFinish` prove the actual lifecycle
against an independent Lean recorder model. Nested begin invalidates batch
independence; fault recording keeps the observed prefix; finish consumes the trace
even when the footprint has disappeared. `MachineRecorderObservation` connects
actual Rust effect folding to direct observations for traces without arrivals.
Arrival batching retains its separate reduction obligations.

`MachineFootprint`, `MachineFootprintMerge` and `MachineFootprintContexts` prove
actual range appending, summary merging and context computation. Result equations
include capacity failures; successful-update and coverage corollaries state finite
bounds explicitly. Coalescing preserves the previous streaming range policy.
Normalization and union are retained as separate, unselected operations.

The `MachineAction` modules compose these proofs through actual owned storage,
without a callback premise for context or footprint delivery. Fused uniform and
mask writes prove the exact storage update and the single routed access together.
Slice observations are retained with recording disabled and with synchronization
filtering. Findings are recorded before diagnostic deduplication.

## Atomic execution

The production atomic path uses `atomic_exec::execute` for its replacement value,
old-value return, result width, predicate, write decision, and epoch decision.
Unsupported scalar width/operation combinations are rejected before address or
memory access. The integer core normalizes all operands to the instruction width
before arithmetic and CAS comparison. The same extracted module selects f32 FTZ,
f64, scalar bf16, and independent packed bf16 lane callbacks. Byte decoding and
encoding also use extracted code.

`rust_atomic_execute_spec` proves total exact refinement to `atomicExecutionModel`.
Integer execution requires no floating arithmetic premise. Floating execution
uses `AtomicFloatContract`, which names each callback's word semantics and
termination; this is a contract of the existing arithmetic subsystem. It does
not establish the internal `fp.rs` algorithms or their agreement with a GPU or
the standalone FloatLib models.

`rust_atomic_execute_footprint` derives the memory footprint from the actual
execution result. Its byte write and numeric update share the same width and
write decision. For a successful write, the helper includes every history
whose four-byte window intersects the write, including starts before the first
written byte. Numeric written markers cover all touched aligned words. A failed
CAS instead selects exactly the destination history and numeric marker. Empty
successful spans select nothing; terminal addresses never wrap.

The footprint proof covers every `u64` offset, `u8` width, and write outcome and
connects its history selection to `Model.Overlaps`. Real instruction regressions
exercise integer and floating operation families, CAS success/failure and masked
operands, SASS register pairs, CAST.SPIN predicates, vectors, shared memory, and
the failed-CAS byte/provenance/taint/definedness frame. PTX `atom.dec` also has a
parser regression.

Return-bearing f32 v2/v4 atomics now use hashed issued-split continuations and
expose every pending lane/component choice through both explorers. The actual
mixed-return trap was missed by the old vector scheduler (11 stateful / 12 DPOR
states, complete and clean), and is found with the continuation (112 / 82 states,
complete). `source_any_interleaved_prefix` and
`source_admitted_prefix_preserves_returns_and_enabledness` connect the extracted
planner to the observation model, including frozen source metadata and addresses.

Native regressions cover same-warp alternatives, guarded lanes, delayed arrivals,
aliased operands, v4 retirement, issue faults, per-access bounds, coverage and
certificate adapters. Whole-instruction C-TICKET forcing excludes every word
touched by a vector f32 atomic; disjoint vector certificates remain supported.
The native scheduler, clocks and certificate analysis remain explicit adapters.
Returnless RED and packed bf16x2 component scheduling remain uncovered; the RED
prototype exceeded a 10,000-state sample without reaching a terminal state.
[ATOMIC_BOUNDARIES.md](ATOMIC_BOUNDARIES.md) records the remaining scope and the
fault-prefix distinction. CAS semantic preservation across admissible inputs
still requires comparison stability or a conditional memory model.

## Memory representation

[BUG_THEOREMS.md](BUG_THEOREMS.md) maps the historical memory failures to closed
theorems and compiling negative controls. The proofs connect actual Rust capture,
writes, and independent markers to origin, numeric-history, and constant-value
consumers; the production adapter boundaries are explicit.

`rust_copy_progress_and_preserves_representation` composes actual Rust capture
with actual checked engine completion and derives payload soundness from the
source. Issue and completion share the same `Semantics`: origin/node IDs, initial
memory, admissible inputs, and capture retain their meaning across both states.

Generation checks live in the Lean `AsyncSafety` wrapper. Production `Machine`
and the synchronous Rust engine do not yet enforce them. A caller retaining a
Rust payload for later completion must keep its destination allocation identity
valid until completion, or supply equivalent generation checks.

`MemoryWrite.write_preserves_representation` establishes that a successful
checked write preserves representation of its specified memory update and
records exactly the independent numeric and half written sets. The input
conditions are representation of the old state and soundness of the producer's
payload and explicit replacement histories. The checked implementation establishes bounds and permission checks,
all byte-plane updates, partial-word invalidation, and replacement histories.

Representation requires captured bytes to agree at an admissible capture,
known origins/histories to denote the actual rounded bits on every admissible
input, and untainted bytes to have input-independent values. Definedness and
initial-memory fallback are tracked separately. No facts are inferred merely
from the presence of an opaque numeric node.

`completeCaptured_preserves_representation` derives payload soundness from
actual source reads at issue and applies it to a separately represented state
at completion. Source bytes may change in between. Scatter captures all sources
before any destination write. `completeDeferred_preserves_representation`
instead reads the completion state. The CAS semantic theorem includes stability
of its comparison across admissible inputs; dependent address/guard flags are
rejected. Failed CAS preserves byte memory while forgetting the numeric history
and marking the numeric word, matching Rust's separate shadow effects.

`apply_write_effect_preserves_representation` additionally proves the extracted
Rust kernel returns successfully and establishes `WriteSpec`, destination
coverage, and key preservation. The kernel's generated files in `Code/` must
never be edited by hand.

These theorems cover the centralized module. They do not assert that every
existing Rust instruction dispatch or arithmetic producer has been extracted
and proved to call it correctly. Race clocks, alias tracking and GPU memory
ordering remain with their existing handlers.

## Reproduce offline

Use installed Charon/Aeneas tools, cached Lean dependencies and Lean 4.35. The
tool JSON names absolute paths `charon`, `aeneas`, `backend`, and `lake`; nothing
is downloaded.

```sh
source env.sh
python3 probes/memory_write/check.py \
  --tools /path/to/cached-tools.json \
  --out /tmp/ptxv-memory-write-check --negative-controls
```

This freshly extracts the Rust crate, checks generated files byte for byte,
builds the whole module, runs Rust and Lean regressions, and audits every project
theorem's transitive axioms. Only Lean's `propext`, `Classical.choice`, and
`Quot.sound` are permitted. Every registered Rust mutation must compile after fresh
extraction and then fail both regression tests and the source-refinement proof.
A receipt records source/tool hashes only after the requested checks pass.
`--regenerate` updates extraction output following intentional Rust edits.
The shared `lean_model/generated_lean.py` publisher prepends the standard license
comment without changing any Aeneas output bytes. Each extraction retains the
raw files in `raw-generated/` and records the policy plus raw and published
hashes in `export.json` and its final receipt. Byte equality compares the
published files, including that deterministic comment.

The production replay driver runs real Rust `Machine` paths in an isolated
source archive. Its instrumentation is confined to test modules and a read-only
provenance getter; hashes and exact patches are recorded. It compares bytes,
origins, taint, definedness, known/opaque numeric histories, and each written set
with executable Lean:

```sh
python3 probes/memory_write/production_check.py \
  --source /path/to/fixed-ptxv-archive \
  --revision SOURCE_REVISION --work /tmp/ptxv-production-replay \
  --target-dir /tmp/ptxv-production-target \
  --lean-project lean_model/memory_write
```

Production replay is differential evidence for the recorded cases. Arithmetic
node IDs come from the arithmetic producers, not destination histories. It also
checks the production invariant that numeric history starts are aligned. The
separate unaligned boundary corpus documents a rejected malformed prestate.
The wide-atomic regression requires the production span-invalidation fix;
`--allow-known-wide-atomic-bug` is only for reproducing rejection of old code.

## Barrier reductions

`BarrierReduction.finite_prefix_budget` preserves complete, blocked and failed observations,
including finding lists and parity-indexed release/acquire clocks, when a concrete bound
keeps every preceding plain arrival in the selected action's generation. A local pending
check alone lacks this bound: `plain_arrival_generation_membership_changes_acquire` gives
the three-arrival counterexample in `fixtures/mbar_plain_generation_order`.

Production removes that local shortcut and records every positive arrival's pending-count
write, including remote batches crossing phases. `BarrierReductionRefinement` proves the
extracted count update, completion predicate, pending-write decision and disabled-allocation
progress/frame contract.

## Definedness precision

`DefinednessRefinement` proves the extracted bit-window selection, per-byte store masks,
placement of each source register's bytes and ordered/unordered write updates. The model
derives conversion agreement from defined consumed bits and characterizes a vector store's
undefined bytes exactly as the union of its source windows. Bits outside a consumed window
cannot poison its result. This precision obligation rejects the historical whole-register
conversion and whole-vector store behavior; a soundness-only theorem permits both.

The simulator uses these kernels for narrow conversions and store masks. Instruction-to-window
decoding and the register-mask capture adapters remain separate producer obligations.

## Analysis cache transparency

`AnalysisCacheMayWrite` proves that accepted cache keys refer to the same immutable program
and the actual extracted may-write result/query equals fresh computation. Issuance proofs
exclude reused generations; replay proofs require the full key and exact optional read
observations, retaining every earlier mismatch. The runtime program owner privately binds
its identity to its kernel, CFG and register index. Tuning-only clones share this owner.

The full walk-to-read-tree producer and worklist correctness against graph paths retain
separate obligations. Atomic issuance linearization and the immutable publication wrapper
use Rust's standard-library guarantees; no extra Lean axioms model them.

`BarrierTransition.rust_transition` refines the complete production counter kernel:
signed transaction updates, zero/count arrivals, errors, completion, expected-count reset,
and wrapping phase advance. Local, remote, cp.async, tcgen, CLC, st.async and TMA producers
call this kernel. `rust_publication_parity` proves the actual old-parity publication plan;
`source_completion_wait_acquires_release` composes it with the explicit clock adapter.
`source_nonwriting_transition_frames_counters` requires every arrival, nonzero transaction
change or completion to write phase membership. The transaction-only counterexample in
`fixtures/mbar_transaction_generation_order` previously lost a reachable deadlock in PTX DPOR.
SASS already reported this fixture's deadlock.

Transaction overflow is a typed framed fault, and over-arrival retains the preceding
transaction update without publishing a clock. Phase wrap is explicitly modulo 2^32.
Full scheduler completeness, init validity, issue-time remote aggregation, native clock
operations and general DPOR/ABA behavior retain separate obligations.

`BarrierStaging.source_live_counter_stages` covers the actual mutable pending/transaction
updates before publication and the reset/phase writes afterward. The generic extracted
`publish_release` moves the current clock into exactly the old-parity done slot and clears
current without changing the other slot. `BarrierClockMove.source_counter_release_sequence_refines_publication`
composes the actual begin/publish/finish source calls. Native pointwise clock join/empty
projection, actor bookkeeping and full explorer preservation remain separate obligations.

Barrier phase dependency keys intentionally identify phases modulo 2^20. `BarrierPhaseKey.phase_key_identifies_object_and_phase` states that exact alias rule; it does not claim full phase injectivity. The extracted packing arithmetic stays within the u32 offset's object range without u64 overflow. Production completion now calls extracted `next_arrival_key` on the transition's actual after-phase; `source_completion_dependency_matches_next_arrival` proves the completion write matches that next arrival read. The old packed-key-plus-one rule crossed into another offset at low20-bit phase wrap, reproduced by a native Commit footprint control. All seven arrival/byte producers exercise the corrected wrapper at u32 wrap. This is a producer/key consistency fix; whole-explorer phase clipping and DPOR/ABA completeness remain separate obligations.

`BarrierPolling.source_progress_refines_complete_blocked_failed` covers actual Rust wait enabledness and `source_poll_acquires_observed_release` covers its generic release selection. An error is enabled once per actor/pc so its diagnostic can execute. The extracted monitor then blocks duplicate errors, while a valid ready query remains enabled and clears the monitor on acquisition. Native Thread hashing includes this monitor. `diagnosed_error_is_observational_stutter`, `duplicate_error_erasure_preserves_future_trace`, `poll_trace_failure_sticky`, and `poll_trace_findings_prefix` justify omitting diagnosed repeats without erasing the initial finding or absorbing the actor; native init/completion re-enabling is tested. The uninitialized fixture was deadlock-only before and now reports the existing `violation`/`mbar-uninit` diagnostic with both explorers. `source_two_completions_close_poll_window` gives a source-refined phase/ABA enabledness obligation, including wrap; it is not a whole-explorer completeness result. Native initialization decoding, VC clone/join representation, diagnostic payloads and actor adapters remain explicit boundaries.

### Sparse release-clock source composition

`SparseClock.source_join_counter_release_sequence_refines_publication` composes
actual counter staging, sorted sparse-clock lookup/join, empty construction and
release publication. Native VC lookup, entries and join consume this kernel while
retaining Arc sharing and cached hashes. Sorted input bases, allocation room and
shared-Arc content equality remain explicit; other native VC producers are not
proved by this slice.

### Blocking waits and nonblocking peeks

The actual `split_enabled` and forced-peek dispatch consume extracted
`barrier::wait_mode` and `wait_policy`. Pending peeks advance without acquiring;
an invalid query reports its first diagnostic and then blocks at that PC until a
valid query. Forced failure is allowed only for valid nonblocking queries.
`RustBarrierWaitPolicy.source_policy_preserves_enabledness` and
`source_error_reports_once_in_both_modes` prove this policy against independent
mode and observation semantics. Native diagnostic messages, register writes,
peek classification and the general forced-failure reduction remain adapters.
The model records fault events per actor; native findings deduplicate signatures
across actors, so event histories are not an exact model of the native collector.

### Actual cached may-write paths

`MayWrite.may_write_sets_query_catches_every_path_write` proves that actual page
allocation, row merges, queue scheduling, compaction and queries retain every
write on an admissible graph path. `AnalysisCache` composes this result with
acceptance for the current immutable program identity. This slice assumes
successful analysis and query, bounded graph fields and a valid start; totality,
the actual register interner, CFG/reach producers and native publication remain
separate obligations. Missing PC rows stay conservatively unknown. Native
oracle comparisons and six compiling mutations exercise each major data path.

### Actual register index producer

`RegisterIndex.built_register_query_catches_every_path_write` connects the actual
nested graph-write builder, full-field register keys, sparse interning and query
lookup to the may-write path theorem. Production program publication and queries
use the same extracted index. Existing mappings are preserved, new IDs are
bounded and injective, and file/index bits survive page encoding. Successful
construction remains a premise here. The native iterator that maps actual IR
registers to keys, the instruction producer's index bounds, CFG/reach producers
and legacy destination fallback retain separate contracts. Cross-subsystem
composition belongs to the sibling `subsystem_interop` package.

### Successful may-write analysis and queries

`MayWrite.may_write_sets_returns_and_queries_catch_every_path_write` proves the
actual source analysis returns successfully and every path-write query returns
true. It requires a bounded graph, valid start and the explicit finite capacity
condition `reach.len * (wordCount(register_count) * 64 + 1) + 1 < Usize.max`.
Successful analysis, successful queries and a precomputed closure are derived,
rather than supplied. Finite known-row/bit facts bound scheduling and queue
capacity. Query binary search terminates even for unordered rows; shifts stay
within the word width. Missing rows remain conservatively unknown.

`MayWriteMainProgress` isolates successful return from the older conditional
correctness proof. Its non-decreasing-pop control fails in this new leaf.
The query controls separately exercise missing-PC progress and register64.
Two intentional nontermination controls have eight-second native budgets;
receipts distinguish these timeouts from assertion failures. Physical allocation
failure and the remaining native CFG/publication adapters are outside this proof.

### Atomic certificate inputs match the executed request

`AtomicRequestLog.source_fadd_log_matches_actual_compute` proves that the actual
resolved request supplies the same operand to arithmetic and the C-FADD log.
`source_component_log_is_frozen` composes this projection with captured component
operands. Production passes one request to both access hooks and arithmetic; the
hooks no longer reread the original vector instruction. In a three-CTA v2/v4 RED
regression, both explorers find two outputs while the old log certified clean;
the repaired certificate reports nondeterminism and the same outputs. Native
controls cover scalar/component operands, result/acquire flags, packed and
unsupported widths. The native log collector and whole certificate algorithm
retain their existing adapter contracts; this does not close RED scheduling.

### Shared warp state and execution dependencies

A scheduled SASS counterexample lets lanes 1–31 change UP0 while lane 0 branches on it.
The old DPOR explorer completed clean in five states, although an enabled peer-first
execution traps. Recording each split's shared-warp read and every actual UR/UP value
or definedness write restores the trap (11 DPOR states; nine stateful states). UR-derived
and negated-UP branches, blocked SHFL, and private R/P controls cover the same boundary.

`ExecutionFootprint.rust_overlaps` proves the actual Rust two-pointer overlap scan
sound and complete for sorted ranges. `rust_conflicts` proves the production conflict
policy, and `source_shared_mutation_is_a_dependency` connects shared-warp writes and
reads to that decision. Native normalization, mutation-hook coverage, and the complete
DPOR algorithm remain adapter obligations; this is not a whole-explorer completeness
proof. The conservative warp resource can increase exploration when uniform writes
interleave with divergent peers; read/read actions and different warps remain independent.

### Floating atomic order certificates

`AtomicConfluence.source_all_permutations_match` proves that the actual bounded
checker accepts only when every permutation of the logged operand multiset has
the observed final bits. `source_every_admissible_prefix_is_covered` connects each
relevant prefix to a checked layer and local commutation. This includes duplicate
operands and uses the actual arithmetic callback contract, without associativity.
The native C-FADD adapter builds untrusted layers, validates original-operand and
layer-domain membership, depth-indexed closure, all pair commutations and the
reference result. Capacity or work rejection falls through to the existing
bounded order enumeration. The existing C-ORDER/no-observer and faithful-log
premises remain explicit; this theorem does not justify an explorer reduction.
The 128-thread RED probe now certifies clean with bitwise expected output. The
separate same-warp RED component scheduling gap remains open.

### Packed bf16 atomic returns

Returning `bf16x2` adds expose both half choices through the component continuation.
The native mixed-return trap was missed by whole-instruction scheduling; the new
stateful and DPOR runs find it completely. `RustAtomicPacked` proves the extracted
half selector, frozen component transition, return assembly and per-half definedness,
and admits every enabled component prefix with exact returned observations.
C-TICKET excludes these packed instructions until forcing has component identities.
The scalar access, clock, diagnostic, coverage and numeric adapters have native
controls; their whole-executor refinement remains explicit. Returnless RED component
scheduling remains open. Issue preflight may fault before any component writes;
later access faults preserve completed memory writes without partial register returns.

### Scalar integer word lane scheduling

Three same-warp lanes expose a middle-return trap in an order such as 1,0,2.
Bundled forward/reverse execution completed clean in four states; the continuation
finds the trap in 29 states in both explorers. C-TICKET also falsely certified the
old Add32 kernel clean by dropping forced intra-warp orders as infeasible. Each
continued lane now has frozen-word readiness and a distinct claim; unsupported
bundled same-warp/same-PC claims are excluded from tickets. Ordinary scalar
cross-warp tickets keep their existing policy.

The integer source policy covers PTX returning OR/AND/XOR/Add/Exch/MaxU/Inc/Dec, width
four, without predicate or new-value operands. `rust_plan` and
`rust_supported_operation` prove that inventory. `rust_frozen_request` refines the
production request producer: the actual typed operation and captured source feed
arithmetic and both audit hooks, with width four and compare zero.
`source_frozen_request_executes_integer_model` composes extracted request,
old-value resolution and execution with the integer reference model.
`source_every_enabled_scalar_prefix_preserves_observations` and
`source_middle_lane_claim_order_is_admitted` prove the extracted plan admits
arbitrary enabled lane prefixes with their returns, findings, frozen operands
and future pending actions. `AtomicComponentPlan.observe_trace_failure_sticky`
preserves failure across those traces. `source_matching_forced_actor_can_advance`
and `source_guarded_off_ticket_advances` prove the local progress decisions;
`source_unsupported_bundled_claims_are_excluded` rejects unsupported ticket pairs.
These cover the extracted planner and its trace model. Native action construction,
issue snapshots and scheduler dispatch remain tested adapters. `rust_replacement_defined` proves Exch uses
source definedness independently of old memory; other operations conservatively
require both words defined. The returned old word retains its own undefinedness.

MinU/MaxS need an additional N-FMAX source/site adapter. MinS continuation exceeded
the existing s_atom_ext fixture's 2M-state budget (the bundled baseline passes in
30.81 seconds), so it remains excluded pending a justified reduction. Their active
overlapping word claims, or shared return registers, now report `unmodeled`; disjoint private
claims and a single active lane retain coverage. `rust_bundled_order_unmodeled`
and `source_disjoint_private_word_claims_keep_their_coverage` refine this actual
policy. The native wrapper resolves aligned words before checking pairs and
preserves the existing arithmetic and numeric hook. Both explorers report the
coverage gap for the actual counterexamples; singleton witnesses still trap.

Native controls cover all six lane orders for all eight continued operations,
frozen UR source/address aliases, actual audit flags, undefined old/new data,
empty/partial/late guards, and issue/access faults. Issue captures every fallible
operand before writes; later access faults preserve completed prefixes. Numeric
replacement history for the new integer paths is cleared conservatively. Actual
Op projection, word resolution/pair scan, diagnostic delivery, per-access liveness,
clock publication and coherent PC retirement remain tested native adapters.
MinU/MinS/MaxS interleavings, integer64, scalar F64/bf16, CAS/CAS.SPIN, predicate
results, SASS scalar execution and returnless RED remain outside the continuation
coverage. There is no whole-explorer or whole-certificate soundness claim.

### Returning scalar F32 lane scheduling

The same middle-return counterexample applies to three scalar F32 adds of 1.0.
Both explorers with forward/reverse bundled orders completed clean in four states,
while the actual order 1,0,2 traps on lane zero's returned 1.0. The production-used
eligibility now admits PTX returning F32 Add, width four, without predicate or
new-value operands. Its existing captured-word continuation finds the trap in
29 states; C-TICKET executes the forced claim and reports the trap.

`ScalarFloat.every_distinct_lane_order_with_context_is_admitted` proves every
nonrepeating issued-lane order is admitted, including arbitrary context actions.
`every_contextual_capture_prefix_preserves_observations_and_enabledness` preserves
the exact return/finding trace, latched operands and remaining plans. The separate
word model proves the returned value is the preceding memory word and that the
update affects only the selected address and return owner. It requires no
associativity or absence of observers. `equal_final_words_do_not_preserve_return_observations`
exhibits the three-lane witness under its three explicit floating-add facts.

`source_frozen_float_request_executes_model` refines actual request production,
old-word resolution and arithmetic. `source_float_returns_the_old_word_and_writes_the_captured_sum`
proves its old return, replacement, width and write/epoch controls, using the existing
`AtomicFloatContract` for the floating callbacks. `source_contextual_float_plans_preserve_the_word_effect_trace`
connects the extracted planner to the word model. Contexts frame the issue plans
and frozen operands in that model; actual issue snapshots, memory resolution,
valid-access/writeback/clock adapters and scheduler dispatch remain native boundaries.

Native controls cover all six orders with aliased uniform source/return storage
and retained numeric history, separate old/source definedness, private words,
a single active lane, actual audit flags, and the earlier issue/access-fault paths.
Scalar F64/bf16, integer64, CAS/predicate results, SASS scalar lane orders and
returnless RED retain their explicitly unsupported continuation scope.

## Bounded arrival traversal

Both production WALKS variants use the extracted `arrival_walk::may_arrive`
frontier/retry driver. `ArrivalWalk.rust_may_arrive` proves successful return for
any finite budget and proves that false requires a settled final callback world
with no reachable hit in its independent graph. The refinement theorem
`may_arrive_catches_every_locally_simulated_path` lifts this to an independent
concrete reachability observation via local entry/step/observation simulation.
Exhaustion and frontier overflow return true (possibly reachable).

`actual_dependency_scan_certifies_settlement` proves the actual used/written
scan covers every used register. Native HashSet/tuple-key and slice-index
correspondence, local Op/Env/REnv expansion coverage, alias/definedness and
future-read stability establishing the final local simulation remain explicit.
These are local semantic and native-container boundaries; whole traversal output
correctness is not assumed. Callback termination and finite child production are
required. The budget bounds driver iterations, not wall-clock time or RAM.

### Clock constructor histories

Actual single/tick, shared-base bulk join, meet, tail, unseen-clock filter,
batched duplicate-own merge and complete read-release selection now use extracted
kernels. `producer_histories_have_sorted_bases` derives base ordering from successful
covered constructor histories; `source_release_from_producer_histories` discharges
the prior ordering premise for publication. Tick overflow retains its existing
headroom premise. The native Arc representation, hashes, pending-minimum and input
sort/iterator adapters remain explicit; arbitrary external sorted imports retain
their sortedness precondition. The producer inventory records each boundary.

The merge preserves its bulk-copy suffix after coalescing its first boundary
entry. Independent exact-vector oracles, native barriers and the actual read-done
ring regression cover the constructor refactor. This adds producer coverage and
does not claim a new explorer bug or whole-explorer invariant.

### Unsupported local returning-atomic pairs

Returning F64/Add64/CAS32/BF16 have the same three-lane middle-return trap. Both explorers
with either bundled lane order completed clean in four states, while actual order
1,0,2 trapped. The source-used admission decision now checks every returning PTX
atomic without a native continuation: an active same-warp pair on the same resolved
word, or sharing a return owner, reports `unmodeled`. The counterexamples now report
that gap in four states. Their existing C-ORDER rejection is retained.

`AtomicLaneCoverage.an_undiagnosed_returning_interfering_pair_has_a_continuation`
is the independent necessary local condition. `rust_bundled_order_unmodeled` refines
the actual Rust decision; `source_uncontinued_returning_interfering_pair_is_diagnosed`
and `source_an_undiagnosed_returning_interfering_pair_has_a_continuation` connect
its outcome to that condition. The actual layout query, guard/membership scan,
resolved-word/return-owner projection and diagnostic delivery remain native adapters.
Disjoint and single-claim controls are compatibility evidence.

This is not global coverage for disjoint words. A separate actual witness has
CTA0's two lanes add to different u64 words and CTA1 read them with add-zero atomics.
The order writer0,observer0,observer1,writer1 traps on (1,0); bundled execution still
completes clean (stateful 18 states, DPOR 20). The local pair monitor leaves this
external-observer gap unresolved. Scalar64/CAS continuation, SASS scalar orders and
returnless RED still require their own scheduling coverage.

### Complete local atomic claim enumeration

The production admission adapter now calls the extracted `unmodeled_claims` scan
on its frozen active claims. `rust_unmodeled_claims` proves that it returns true
exactly when an ordered index pair requires the local diagnostic. The inner and
outer loop invariants preserve the remaining suffix/pair observations; no sorting
or distinctness premise is required. `source_an_undiagnosed_claim_list_covers_every_interfering_pair`
therefore discharges the complete-pair enumeration premise of the previous policy.
A native control puts the only collision in lanes 1/2, after a disjoint first lane.

Guard evaluation, resolved location/warp/owner projection, the layout query and
report delivery remain native adapters. Region zero represents an unresolved word;
shared-return claims still interfere independently of resolved-word equality.
This closes local enumeration and does not repair the documented external observer
miss for disjoint unsupported wide atomics.

### Returned-word coverage across physical frontends

An actual sm_103a SASS F32 atomic retained three active lanes. Both explorers
completed clean in four states and certification accepted clean, while the native
lane order 1,0,2 trapped. The PTX-only admission exemption caused this gap. The
architecture-free source-used `returned_word_claims_unmodeled` now reports the
missing local scheduling coverage and certification refuses; the explorers report
`unmodeled` in four states. Compiled integer Add32 aggregation, private words and
a single active lane retain clean certificates. These are CPU simulation tests of
ptxas-generated cubins, with no GPU execution.

`AtomicReturningCoverage.an_undiagnosed_interfering_returned_word_pair_has_a_continuation`
states the independent obligation without a frontend parameter.
`rust_returned_word_claims_unmodeled` refines the actual complete Rust pair scan;
`source_every_undiagnosed_interfering_returned_word_pair_has_a_continuation`
discharges that obligation for every enumerated pair. Native claim production,
continuation lookup and report/certificate delivery remain explicit adapters.
Predicate-only CAS.SPIN returns, returnless RED, and external observers of
disjoint bundled words remain outside this local returned-word contract.

### Local atomic admission, including predicate and memory-only CAS

Actual ptxas-generated predicate-only CAS.SPIN and discarded-result CAS also
certified clean while a native lane order trapped. The typed IR producer now
selects normalized observable outputs and both compare-exchange operations,
including discarded outputs. PT/UPT remain constants. The source-used local
admission dispatcher retains their pair obligations during certificate collection.
Compiled private-word and single-active max loops remain clean; all six concrete
winner orders retain their findings. These are CPU tests of actual cubins, with
no GPU execution.

`AtomicResult.source_local_order_monitored_iff` proves the typed output/operation
producer. `rust_local_claims_unmodeled` refines the actual admission decision and
complete pair scan; `source_monitored_interfering_pairs_have_a_continuation_or_diagnostic`
connects every selected pair to the independent continuation obligation. Actual
claim/continuation construction, native location/owner projection and finding
transport remain adapter boundaries.

For scalar F32 RED, ordinary exploration now reports unsupported local overlapping
lane orders. The cancellation case previously completed clean with zero although
the middle permutation produced one; an observer could trap. Certificate collection
defers only this RED check to existing C-FADD/C-ORDER conditions: the pure example
still certifies both outputs, the observer still forces refusal, and exact/confluent
reductions remain certifiable. `certificate_collection_preserves_monitored_atomic_obligations`
prevents this deferment from suppressing returned-value or CAS obligations.
This is no whole-certificate theorem. Other unselected returnless operations and
the external-observer miss for disjoint bundled words remain outside this local
coverage contract.

### Discarded exchange memory effects

An actual compiled global EXCH with RZ output also certified clean while a middle
lane writing last made a later load trap. The typed local-order producer now
selects EXCH even without outputs. Its independent exchangeMemory policy and
source_unused_EXCH_is_monitored theorem extend AtomicResult to fifteen source
and observation theorems; compareExchangeMemory remains exactly CAS/CAS.SPIN.
The fifth classifier control restores this omission and fails Rust and Lean after
the generated code compiles. The Interop actual_exchange_requires_a_continuation_or_diagnostic
theorem composes that producer with the actual complete-pair scan.

The real cubin now reports unmodeled in all four explorer modes and certification
refuses. Private-word and single-active controls retain clean certificates; all
six singleton Machine orders retain their findings. The old classifier/negative
nine-theorem witness remains pinned separately. This adds one operation family,
with no global independence claim for disjoint-word external observers or safety
claim for other unselected returnless families. No GPU execution was used.
