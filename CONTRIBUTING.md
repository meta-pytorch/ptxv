# Contributing to ptxv

We want to make contributing to this project as easy and transparent as possible.

## Pull requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs or behaviour, update the documentation (`README.md`, `docs/`).
4. If you've added a modelling shortcut, add an `ASSUMPTIONS.md` entry and tag the code with
   `ASSUME(ID)`; `cargo test --test assumptions` checks that the two agree.
5. Ensure the test suite passes and `cargo clippy` is clean.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this
once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Building

Requirements: a stable Rust toolchain (Cargo) and the CUDA 13 toolkit binaries `nvdisasm`
and `cuobjdump` on `PATH` (the SASS front-end runs them). A GPU is not needed to build or to
verify kernels.

```
source env.sh              # PTXV_ROOT, CUDA on PATH; respects an existing CARGO_HOME / CARGO_TARGET_DIR
cargo build --release      # -> target/release/ptxv
```

`env.sh` reads optional overrides from `$PTXV_CUDA_HOME` (default `/usr/local/cuda-13.0`) and
`$PTXV_PYTHON` (the Python used by helper scripts, default `python3`).

## Testing

```
cargo test --release --no-fail-fast
cargo clippy --release --all-targets
PYTHONPATH=python python3 -m pytest -q python/tests
```

- The full Rust suite takes about 20-25 minutes; `tests/fixtures.rs` explores real kernels.
  Run a subset with `cargo test --release -p ptxv-sim --test fixtures <name_substring>`.
- Exhaustive explorations can use a lot of memory. On a shared machine, run them under a cap,
  e.g. `systemd-run --user --scope -p MemoryMax=48G -p MemorySwapMax=0 <cmd>`.
- **GPU tests are optional.** Hardware-dependent Rust tests skip when no idle Blackwell GPU
  is available. Python GPU tests run only with `PTXV_TEST_GPU=<index>`. ptxv never uses a GPU
  that has compute processes it did not start; `PTXV_GPUS` / `CUDA_VISIBLE_DEVICES` restrict
  the set further, and `PTXV_EXCLUDE_GPUS=<i,j>` makes the test harness and helper scripts
  refuse specific GPUs.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is clear and has
sufficient instructions to be able to reproduce the issue: the ptxv command line, the case
directory (or how it was captured), and the output.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe disclosure of security
bugs. In those cases, please go through the process outlined on that page and do not file a
public issue.

## License

By contributing to ptxv, you agree that your contributions will be licensed under the LICENSE
file in the root directory of this source tree.
