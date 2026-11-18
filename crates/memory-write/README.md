# Memory effects core

Dependency-free data types and functions for owned memory payloads, byte and
history writes, atomic components, barriers, and machine state effects.

This change introduces the reusable core without connecting it to native
execution. The simulator continues to use its existing transition logic.

Run the core tests with `cargo test --release -p ptxv-memory-write --lib`.
