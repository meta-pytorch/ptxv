# Memory-write effects

Dependency-free Rust cores for byte/history writes, owned payloads, atomic
execution, register/control mutations, effect footprints, and barrier clocks.
The simulator adapters supply allocations and execution state, then apply the
prepared effects to concrete bytes and symbolic metadata together.

Source capture precedes destination mutation, including overlapping copies.
Every intersected four-byte history is invalidated; replacement history requires
all four bytes to be written. Address-space and allocation identity are retained
through each effect. Allocation lifetime and capacity remain the caller's
responsibility.

Run the core tests with:

```sh
cargo test --offline --locked -p ptxv-memory-write
```
