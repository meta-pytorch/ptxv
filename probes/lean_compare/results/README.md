# Validation receipts

`validation.json` summarizes the checks against the unchanged Rust base and
records the source hashes used. `scalar.json`, `mma.json`, and `smoke.json`
contain the comparison results. `mma-overflow.json` is an extended-domain
comparison with two reported disagreements; its runner exits nonzero.

The exhaustive FP32 MUFU sweep was not rerun for this integration. Optional
FloatLib and generated-certificate sources outside the default Lake targets are
not all covered by the core build.
