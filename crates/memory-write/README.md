# Memory-write effects

Dependency-free Rust implementation of normalized byte/history writes, checked
source capture, and independent numeric/half fallback markers.

`checked::capture_span` reads an ordered live span into an owned payload before
any destination changes. `checked::checked_write` rejects dependent addresses or
guards, unsupported address spaces, wrapping spans, and missing live bytes before
mutation. `engine::apply` couples accepted writes with both marker sets. Ordered `WordPatch` updates override histories after byte
invalidation; `numeric::forget_numeric` provides the numeric-only shadow effect
used by failed CAS while preserving byte and half planes.

Keys contain address space, allocation identity, and byte offset. The caller
supplies the live inventory, including local growth or TMEM allocation already
performed by its allocator. Every intersected four-byte history is invalidated;
a replacement history is installed only when all four bytes are written.

The extraction harness in `probes/memory_write` compiles these exact source files
through its `[lib]` path. `lean_model/memory_write/MemoryWrite/RustRefinement.lean`
connects checked Rust writes to the existing Lean `applyEffect` implementation;
`RustCapture` derives payload soundness from represented source memory, and
`RustEngine` connects byte and marker updates. Generated Lean files are checked
for freshness and must not be edited manually.

Payload origins, known numeric nodes, and input-independence flags require sound
producer certificates. Their semantic meaning is shared across the source and
destination and remains stable when captured source storage is freed. Allocation
permissions and capacities are established by the caller's live inventory. Allocation identity must remain valid until a captured destination
write completes; the Lean `AsyncSafety` wrapper specifies generation checks for
free/reuse. The crate currently provides synchronous effects and owned payloads.

Run the crate tests offline with:

```sh
cargo test --offline --locked --manifest-path probes/memory_write/Cargo.toml
```

The complete extraction, proof, regression, and mutation gate is documented in
`lean_model/memory_write/README.md`.
