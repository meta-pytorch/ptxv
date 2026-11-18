# Instruction effect contracts

This leaf refines actual Rust instruction producers for selected memory-family
register reads, writes, control/data sources, access windows and register ownership.
The independent Lean specifications use actual `Op` operands. The production
simulator consumes the same Rust helpers. Legacy operation families, execution
context and native collection adapters retain explicit boundaries.

Regenerate source through `probes/instruction_effects/check.py`; generated Code
files are not edited by hand. The sibling `subsystem_interop` package imports both
this leaf and `memory_write` to compose keys, symbolic state and memory/cache
contracts. Neither leaf depends on the sibling composition package.

### Actual control-flow production

`ControlFlow` proves the actual selected instruction and guard produce each
successor row and the actual source loop produces the complete row list. A guarded
BRX retains its fallthrough alongside every listed target. Production `cfg_succ`
and SASS dependency-barrier classification use this extracted producer. The sibling
Interop package connects these actual edges to selected memory-Op destinations,
full register identities, the actual register interner and may-write queries.

The native parsed-PTX regression previously advanced through a false BRX guard to
a register write that the static graph and production cache missed. It now checks
the actual parsed destination, dynamic PC, graph edge and cache answer. This is a
static-analysis false negative; it does not establish a missed explorer verdict.
Return-table completeness, target validity, sound folds, parser spans and dynamic
indirect-call resolution remain distinct source-input obligations. This query
composition still assumes successful analysis/query and conservative reach bits.

### Constant predicate reads and comparison writes

The source normalizer preserves every predicate read and negation while removing
architecturally constant predicate register evidence. PTX P7 remains mutable;
SASS P7 and either frontend's UP7 are constant. The actual comparison plan derives
a full creator key and only remembers eligible writes to mutable predicates.
Ignored destinations create no record for any eligibility flag, preserve all
other cache keys and require no evidence-validity premise. Cache soundness is
preserved when evidence created for eligible destinations is valid.

The SASS parser and numeric selection consume normalized predicates; numeric
FSETP recording uses the extracted plan. Parsed SASS FSETP/SEL operations previously
made a selected value's numeric history disagree with its actual bits when SEL
used P7 rather than PT. The regression checks both spellings using initialized
memory inputs. It is not a whole SASS-kernel or GPU test. Native HashMap updates,
comparison evidence construction and the numeric callback implementation remain
adapters; the proved creation plan makes their policy explicit.

### Predicate knowledge used by arrival walkers

Both arrival walkers call the extracted normalized predicate reader. Constants
skip the register callback and predicate negation is applied exactly once. The
source theorem preserves reader state and derives sound known truth from sound
partial register knowledge. Parsed false Exit, Trap and Bra controls exercise
fresh, memoized and repeat queries. Native callback projection, temporal knowledge
stability, the full walker and cache-versus-fresh behavior remain separate
obligations; this leaf does not prove a whole explorer verdict.

### Guarded value-tracking lookahead

The actual BRX/call/return producer reads the guard before decoding a target.
The source theorem for a false guard returns Next and frames reader state after
the predicate callback, without target validity, target-read or capacity premises.
For an unknown uniform guard, the actual source lifting retains both the taken
outcome and fallthrough, assuming successful actual target decoding and one
append slot. Divergent/invalid control remains conservative unknown.

The native regression compares parsed false-guard BRX with actual Machine
execution, retains all required unknown-guard alternatives and checks false calls
and returns even with invalid targets. This establishes local lookahead coverage.
Target decoding, callback soundness, numeric abstraction and whole worklist or
explorer completeness remain separate obligations.

### Temporal knowledge in arrival analysis

Mutable warp-shared UP/UR values are unknown to arrival walkers, including local
environments, captured operands and facts derived from them. Architectural
constants are normalized before that policy. Extracted ownership and knowledge
proofs connect private register separation to sound partial predicates under
explicit peer interference and native callback contracts. Parsed SASS regressions
show that a peer can change UP while the queried lane remains at the same PC;
fresh and memoized queries now both retain the reachable barrier. Cache replay
was already detecting changed observations: the repaired premise is future value
stability. Native transition projection and full traversal correctness remain
separate obligations. Whole-warp LaWarp simulation has a different coverage
boundary and is not changed by this thread-local policy.


The eager exit-drain consumer uses the same temporal predicate policy. A scheduled
parsed SASS regression with no static findings previously completed clean in five
stateful-explorer states despite an enabled peer-first trap. Keeping both actions
finds the trap in nine states. Private/constant controls retain precision; the
drain's own predicate writes prevent stale known values. Two native integration
mutations check routing and this own-write restriction. They do not add a proof of
the native drain loop or whole-explorer reduction; DPOR's shared-register resource
coverage is a separate contract.


Static guard propagation also applies the temporal predicate/value policy before
consulting local facts. Five parsed peer-write cases previously published a true
fold after its real guard became false, including private facts derived from UR.
Actual execution followed the real guard and emitted E-CONST-BRA; the correction
retains both static paths and avoids that spurious mismatch. The native candidate
worklist and callback semantics remain unproved by this policy leaf. Two native
routing controls exercise that adapter; they are not source-proof rejections.

### Exact comparison-derived selections

The actual FSETP/SEL pair retains an exact two-instruction segment instead of an
associative Max node. Actual comparison capture and pair matching derive predicate
correspondence, modes, operand node IDs, negation and order. The selected operand
bits refine the source pair for every node valuation and a universal comparison
callback, preserving signed zeros, equality ties and NaN payloads. FMNMX keeps its
own Max semantics. The parsed signed-zero regression requires the retained segment;
14,400 native class combinations cover all sixteen IR comparison modes.

The float primitive, native maps, trace/comparison freshness and history soundness
remain explicit adapters. Raw creator predicate keys index the creator's trace;
owner-aware invalidation removes aliases across shared writes. The first policy
leaf leaves the actual segment constructor and bindings to its companion proof.


The companion `SelectionProgram.actual_selection_program_refines_source_operands`
connects actual pair matching to the source-used simulator constructor and its real
argument/SOp/output bindings for all node valuations. Explicit inverse FCmp maps
relate the two generated IR representations. Both IR and Sim are freshly extracted
by the instruction checker; portable subsystem composition includes all six
constructor modules and pins the complete Sim archive inventory. No external type
templates or aliases are introduced. The general SProg evaluator and native maps
retain separate contracts; the constructor interpreter is independent of that loop.


Actual Fsetp/Sel evaluation delegates to the extracted comparison_instruction
helper. Its source theorem preserves result and mutable callback order, including
failed/diverging callbacks, modifiers, FTZ and selected-only reads. Composition
connects the real IR pair, simulator constructor and evaluator to exact selected
bits for all node valuations. Six compiling mutations exercise both wrong values
and output-preserving extra observations. Native register/constant bindings,
the generic segment loop and host floating comparison remain explicit boundaries.


The binding companion extracts actual SVal reads, first-match register lookup,
predicate lookup/negation and missing-entry defaults. Its composed execution uses
the produced argument/binding rows, runs the actual first evaluator, inserts its
actual returned value as prior[0], executes the selector, then reads the real
output binding. Constructor bounds and presence eliminate defaults for this pair;
no manually supplied predicate answer remains. The generic segment driver and
host floating primitive retain their separate contracts. Six compiling mutations
check argument/prior indices, key identity, first-match order and predicate policy.


The actual segment evaluation loop delegates to the extracted program::execute.
Its source theorem equals an independent list interpreter for arbitrary callback
results and initial scratch, including errors, failure and divergence. Composition
runs the generated pair through that real driver and the proved binding/evaluator
leaves. Ordered callback prefixes and output lookup are explicit; six compiling
mutations reject lost clearing, wrong prefixes, suppressed errors and skipped work.
The selected per-SOp native dispatcher is closed by SelectionBound below; the host floating primitive remains an adapter.


The actual eval_with now instantiates the extracted bound::Evaluator. Its source
ProgramEvaluator dispatch uses the actual operation fields, full register/predicate
bindings and comparison_instruction. actual_op_bound_evaluator_execution composes
actual IR producer, Rust program constructor, generic driver and real dispatcher.
Constructor-derived eligibility replaces caller-supplied per-SOp callback mappings;
the returned bits satisfy independent selector semantics for every node valuation.
The concrete FP comparison primitive contract is discharged by IEEEComparison below. Unsupported
operations and constant-bank inputs retain exact original fallback delegation.
Six compiling source controls and the 193-test release simulator suite pass.


Constant-fold candidates are checked by the extracted certificate producer before
static pruning. Empty entry, actual instruction transfers and successor checks
derive reachable fact and guard truth. Extracted in-place row merges and final
code-row publication close their per-PC iteration equations. Partial-state path soundness
additionally derives defined guards and correct values for known published folds;
unknown guards forget written facts even when old and assigned bits agree. Native Machine/local
instruction value/definedness/physical-alias agreement, stable launch-constant callbacks, dynamic target/return
coverage and native grid-to-certified-row membership remain explicit boundaries.


IEEEComparison independently decodes binary32 words as exact signed dyadic
values, infinities or NaN. source_f32_cmp proves the actual extracted production
comparator for all words, all sixteen modes and FTZ, including signed zeros and
checked integer operations. Actual eval_with uses the proved IeeeComparison
instance. actual_op_source_ieee_execution composes real IR producer, constructor,
driver, bindings, dispatch and concrete comparator without a floating callback
premise. NodeReaderContract, captured trace/node observations and the plain
producer condition remain explicit; unrelated arithmetic and hardware-model
boundaries retain their existing contracts. Forty-four public theorems and eight
compiling semantic controls accompany the 196-test release simulator suite.
The focused release-library microbenchmark measured median +0.4245 ns/call
(8.6%); this does not estimate whole-simulator speed.


SelectionHardware closes the observed numeric value producer. Actual Num node
construction calls seg::hardware::evaluate, whose extracted source uses actual
IR selector policy and the proved IEEE comparator. source_hardware_matches_actual_program
proves produced observed bits equal actual constructed-program execution, and
actual_op_hardware_and_program_source_selection derives original Fsetp/Sel
semantics for both outputs. Output equality is proved. Capture/map/trace inputs,
the final native Node push and existing NodeReaderContract remain explicit.
Eight public theorems, four compiling semantic controls and 20,736 native
helper/actual-program interoperability cases accompany 197 passing release tests.

### Actual memory metadata delivery

The source-used driver derives destinations from the actual memory operation,
clears all applicable histories, prepares captured sources exactly once, then
clears destination bits, assigns active taint and forces guard-dependent taint.
Unsupported operations perform no callbacks; supported empty writes still prepare
sources once. `actual_memory_metadata_delivery_spec` proves that ordered fold,
and `actual_memory_metadata_frame` proves projected-cell preservation under the
explicit alias relation, including uniform owners and virtual high cells.

Active and guard-dependent participants, finite register/list bounds, callback
semantics and projected native map/half/comparison leaves remain explicit. Source
queries observe captured instruction state without rereading overwritten operands.
Native old/new map differentials cover aliases, overlapping and duplicate sets,
guard-only threads, empty writes and callback-failure prefixes. Vector atomic
continuations retain their separate conservative finish adapter; no intermediate
whole-state soundness or callback panic atomicity is claimed.

### Actual store byte payloads and reader failures

Actual supported St/Exch and St256 operations derive byte windows and read them
through the extracted register-reader loop used by Machine. The independent
model looks up the first containing source window for each byte and fills holes
with zero. Actual-Op theorems derive layout, support and coverage, rather than
assuming a caller-supplied operand list. Virtual and physical source widths retain
explicit bounds; unsupported descriptors retain their original native path.

The successful callback contract yields exact little-endian output bytes. The
failure contract derives actual source windows and propagates the first returned
reader error after a successful prefix. Callback interior mutation and panic
prefixes are additional native-tested observations, not formal exception traces.
The native adapter calls Machine.rd directly; parsed stores/exchanges and St256
with disjoint, identical and overlapping source bases inspect actual output bytes.

### Actual masked load delivery

`actual_masked_load_execute_spec` derives both halves, compact destination ranks,
source-word offsets and exact little-endian values from the actual Ld256 operation
and 32-byte input. The extracted writer loop performs the actual register and
numeric callbacks. `actual_masked_load_preserves_frame` specializes its independent
writer transition contract to projected register keys. Caller-provided operand or
action correctness lists are not premises. Native callback projections and writer
validity remain explicit; callback panic prefixes are tested separately.

The historical version before the masked-load source-word fix fails the same parsed SASS test
at mask0x3c (R8 receives source word0 instead of word2); the corrected application
passes all255 nonzero masks. Native numeric provenance tests also check Input
allocation/source-word identity and untouched destinations. This does not prove
the SASS parser-to-mask translation or GPU behavior.

### Typed atomic local-order classification

AtomicResult refines the actual typed operation/output producer: observable word or
normalized writable predicate outputs and CAS/CAS.SPIN memory effects require monitoring.
Thirteen universal source and observation theorems reuse the proved predicate normalizer;
four compiling source controls reject dropped outputs/classifiers. Other unselected
operation families are not proved safe by this policy.

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

## Future shared-data reads in eager exit

`exit_bound` now rejects mutable uniform data reads before admitting invisible
operations, after skipping a known-false guard. The extracted read-slice scan
returns successfully and accepts exactly thread-private register files.
`AnalysisReadComposition` closes actual Mov and LdConst operand classification
without a read-list premise, and composes selected memory-family read enumeration
with the scan under its existing register-span bound. Other Op::src_regs native
iterator correspondence, native exit-bound routing and the full reduction
observation contract remain explicit boundaries. Parsed enabled peer-first
regressions cover both constant-bank faults and undefined addresses previously
lost by the actual stateful explorer; private/constants retain eager precision.
