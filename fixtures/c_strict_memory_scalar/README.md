# Strict scalar memory safety

Two threads write the global `out` buffer, initially `[0, 0]`. The good kernel
uses `%tid.x * 4` to address distinct words and finishes with `[7, 7]`.

Run from the repository root:

```sh
ptxv check fixtures/c_strict_memory_scalar --front ptx --variant good --strict --max-states 100000 --json
```

Read `rows[0].strict_memory_safety.status`. Change `--variant` to try the
negative examples:

| Variant | Address stride | Strict status | Ordinary PTX/SASS classes |
|---|---:|---|---|
| `good` | 4 bytes | `safe` | `clean` |
| `same_value_conflict` | 0 bytes | `unsafe` | `clean` |
| `out_of_bounds` | 8 bytes | `unsafe` | `fault` |

The conflicting stores both write 7 to `out[0]`. Equal values still conflict
under strict memory safety. Thread 1 in `out_of_bounds` writes beyond the
8-byte allocation. Both negative strict checks exit unsuccessfully.

With `--variant good --max-states 1`, the strict status is `inconclusive`
(`state_cap`), also with an unsuccessful exit. Increase the cap to obtain a
checked result.

An ordinary `clean` result does not establish strict memory safety. See the
[proof scope](../../lean_model/memory_safety/README.md). These examples run the
verifier on the CPU; they make no GPU execution claim.
