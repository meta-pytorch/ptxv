# Rust–Lean comparison

Compare ptxv's Rust arithmetic with the [Lean models](../../lean_model/README.md)
and retained GB300 outputs. This probe needs no GPU. It links the simulator crates
directly; it does not copy their arithmetic.

| Location | Contents |
| --- | --- |
| `lean_model/{Blackwell,Shared}/` | Models, theorems, and Lean replay executables |
| `lean_model/tests/` | Lean executable smoke tests |
| `probes/lean_compare/` | Rust bridge, comparison runners, result receipts |
| Other `probes/` directories and `fixtures/` | Hardware capture code and retained input/output bytes |

This addition is independent of simulator fixes. It leaves the Rust arithmetic
unchanged. [RESULTS.md](RESULTS.md) records agreement and remaining differences
against the base revision.

## Run

Install the Lean version in `lean_model/lean-toolchain`. From the repository root:

```sh
source env.sh
(cd lean_model && lake build && lake env lean Audit.lean && \
  lake build check_mufu_sm103 check_ptxv)
python3 lean_model/tests/smoke.py

cargo build --locked --release --manifest-path probes/lean_compare/Cargo.toml \
  --target-dir probes/lean_compare/target
python3 probes/lean_compare/cross_check.py --output /tmp/ptxv-lean-scalar
python3 probes/lean_compare/mma_cross_check.py --output /tmp/ptxv-lean-mma

cargo test --locked --release --manifest-path probes/lean_compare/Cargo.toml \
  --target-dir probes/lean_compare/target
```

Use fresh output directories. The explicit Cargo target directory makes these
commands independent of `CARGO_TARGET_DIR`; each Python runner also accepts
`--rust /path/to/ptxv-lean-bridge`. Add `--offline` to Cargo when dependencies are
already cached. The separate lockfile avoids resolving the CLI's dependencies.
It is not a substitute for running the complete root-workspace suite.

The Rust tests replay the original MUFU and MMA signed-zero test sources.
They need CUDA 13's `ptxas` and `nvdisasm`. The exhaustive 2^32-input MUFU replay
remains ignored by default.

## Optional FloatLib checks

The native scalar run checks 14 forms. The seven PTX wrappers additionally need
an already-built FloatLib checkout matching `lean_model/floatlib-lock.json`:

```sh
python3 probes/lean_compare/cross_check.py --output /tmp/ptxv-lean-wrappers \
  --floatlib /path/to/FloatLib
```

The runner checks all pinned source hashes and the Lean toolchain before using
that checkout. The source pin does not certify previously compiled `.olean`
files; build the dependency from those sources. FloatLib and its dependencies
are not vendored here.

`mma_cross_check.py --capture DIR` can also replay the earlier M128/N8/K16 BF16
holdout. That optional capture is external; its original hashes are in the
validation receipts. The default MMA comparison generates its inputs and needs
no capture from another branch.

The default domain excludes overflow. To compare both overflow signs as well,
use `mma_cross_check.py --output NEW_DIR --check-overflow`. It writes the
disagreements and exits nonzero when Rust and Lean differ; it does not apply a
simulator fix.

These runs establish agreement on the listed inputs. Lean theorems establish
properties of the Lean models. Neither is a proof of the full Rust simulator,
compiler, GPU, or synchronization semantics.
