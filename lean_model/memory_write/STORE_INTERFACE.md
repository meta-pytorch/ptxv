# Composing symbolic operands with memory effects

`store::WordStoreInput` freezes a source word, predicate, and both halves of a
normalized byte offset. `prepare_word` derives a four-byte payload and the
checked control flags from those same operands. `commit_word` prepares that
request and calls `engine::apply`; callers do not supply replacement histories
or independently construct dependency flags.

The source history is installed only at the complete word's first byte.
Missing history remains missing, including for a captured zero. Every byte
retains the source's input dependence and definedness. This constructor makes
no byte-origin claim.

## Semantic composition

The symbolic and memory modules share `Semantics`: admissible inputs, the
captured input, initial memory, and origin/word interpretations. A symbolic
word meaning agrees with the memory interpretation through its four rounded
bytes. Extending the node environment preserves existing meanings.

The store proof starts with sound source, predicate, and low/high address
metadata, and represented old memory. It derives the payload certificate and,
on acceptance, stable guard/address execution. It then applies memory
preservation to the input-dependent store semantics. Neither `PayloadSound`
nor `StableStoreExecution` is an unexplained caller premise.

Structural premises remain explicit: the captured storage view represents
memory, marker vectors have capacity, and successful enabled writes have a
live nonwrapping destination span. The checks reject dependent control,
including a predicate captured as false. Rejection leaves the entire engine
unchanged.

## Production integration

`crates/sim/src/memory_effect.rs` captures at most seven cells: the four
destination bytes and up to three earlier history starts. It freezes the
source before access bookkeeping, preserves the existing access/race shadow
transition against the old bytes, then invokes `commit_word` once. The result
is projected into the paged bytes and symbolic maps before observers such as
VIP tracking and synchronization publication inspect memory.

This path suppresses the old byte copy, taint update, and subsequent `num_st`
repair. Existing marker entries are retained, new numeric and half markers are
inserted independently, and shared input-origin fallback is invalidated.
The scoped view does not copy the complete allocation.

The initial migration covers single-destination ordinary four-byte stores with
numerical tracking, fully defined scalar register/immediate sources, defined
independent guard/address operands, and no known source half origins. Position
taint, multicast, conditional controls, and existing undefined destination
shadow bytes retain their existing dispatch. Destination eligibility ensures
the race shadow cannot add undefinedness that the prepared payload omits.

The Rust constructor and engine transition are extracted and proved. Actual
`Machine` register capture, address normalization, paged-storage projection,
race clocks, and synchronization hooks are integration boundaries checked by
Machine regressions. The current result is not universal extraction of the
complete instruction dispatcher or scheduler.

## Next migration boundaries

Byte origins and per-byte definedness need corresponding sound constructors
before packed-half and partially undefined stores can use this path. Symbolic
conditional writes need a conditional effect rather than clearing the checked
dependency flags. Other widths and asynchronous completion can then reuse the
same ownership boundary with their own capture and lifetime contracts.

Execution reduction must consume all relevant effects or prove its observation
projection sufficient. Equality of captured bytes alone does not justify
discarding metadata changes, access events, or findings.
