# Selected-memory primitive bindings

The private Aeneas backend adds three primitive mappings and a generated import of `MemorySafety.Primitives`.

* `alloc::sync::Arc<T>` maps to Lean `Id T`. No Arc operation is selected or modeled. The typed-IR admission proof establishes `launch.nic = none`; raw-source admission establishes empty `manifest.nic_qps` and `manifest.nic_mrs`. Both routes reject NIC configuration. Reference counts and NIC contents therefore do not affect any accepted execution. This is a representation abstraction for the excluded field, not a proof of Arc.
* `alloc::string::String::new` maps to the closed definition `MemorySafety.Primitives.stringNew : String := ""`. This extends the existing UTF-8 `String` abstraction with its empty constructor. The raw frontend consumes the exact UTF-8 instruction bytes.

* `alloc::string::String::as_bytes` maps to `MemorySafety.Primitives.stringBytes`, which maps `String.toUTF8` bytes exactly to Aeneas bytes and rejects lengths above `usize::MAX`. No actual Rust string can have a longer allocation.

The concrete Bool `PartialEq::ne` body is included and trait methods filtered against the backend model; it is not a new primitive.

Apply `aeneas-memory-primitives.patch` to the pinned original backend source, build `src/main.exe` using the cached opam switch, and use the resulting binary in the recorded extraction driver. Generated Lean remains unedited. All primitive definitions are computable and introduce no axioms.

The compiler printer uses margin 1000 and max indentation 990. The prior default margin 80/max indentation 68 produced invalid Lean when constructor patterns wrapped under the actual launch guard chain. This formatting-only patch preserves all translated terms and is included in the source patch and binary fingerprint.
