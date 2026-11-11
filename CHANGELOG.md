# Changelog

All notable changes to ptxv are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow the `version` in
`Cargo.toml` and `python/pyproject.toml`.

## [Unreleased]

### Added

- `CHANGELOG.md`, GitHub Actions CI (`.github/workflows/ci.yml`), issue and pull request
  templates.
- Copyright and license headers on every source file.

## [0.1.0]

First public release.

- `ptxv` command line: execution checking of captured launches of NVIDIA Blackwell
  (sm_100a / sm_103a) kernels from their PTX or SASS. Data races, deadlocks, hangs and
  protocol misuse (mbarrier, TMEM, TMA, clusters, multi-GPU memory) over the schedules the
  hardware may pick, with certificates under the assumptions in `ASSUMPTIONS.md`.
- Numerics: equivalence of two kernels with a per-output bound (`compare`) and error
  distributions under an input distribution (`errdist`).
- Python front end (`python/`): capture of kernels launched from PyTorch, Triton and CuTeDSL,
  and an API over the verifier.
- Lean models of GB300 instructions (`lean_model/`).
- Hardware target specifications and assumptions (`targets/`).
