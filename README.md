<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/img/ptxv-icon-dark-mode.png">
    <img src="docs/img/ptxv-icon.png" alt="ptxv" width="128">
  </picture>
</p>

# ptxv

A model checker for NVIDIA Blackwell (sm_100a / sm_103a) GPU kernels. ptxv is the explicit-state kind of verifier: it takes the PTX or SASS of one captured launch, builds a model of the machine that runs it (warps, async units, barriers, the PTX memory model), and explores the schedules the hardware may pick, checking properties on every state it reaches. It checks a kernel in two steps:

1. **Execution.** Data races, deadlocks, hangs and protocol misuse (mbarrier, TMEM, TMA, clusters, multi-GPU memory) over the schedules the hardware may pick. Each finding comes with the schedule that reaches it. The result also says whether every schedule computes each output with the same operations, or which variants can occur (for example with floating-point atomics).
2. **Numerics.** Because step 1 established which operations produce each output on every schedule, they are extracted from one execution and analyzed: equivalence to a reference kernel with a proven per-output bound, and the distribution of the error under a distribution of inputs.

> [!WARNING]
> **ptxv is a work in progress.** Use it to catch bugs opportunistically, not to certify that a kernel is correct. A finding comes with the schedule that reaches it, which you can replay to confirm it. A `clean` verdict has no such check: ptxv still has false cleans, kernels it calls clean that misbehave on hardware, because a model shortcut is too generous or a hardware behavior is not yet modeled. The [soundness harness](docs/soundness-harness.md) looks for them ([latest results](docs/soundness-results.md)). Every verdict also holds only for the captured inputs and launch shape, under the assumptions it lists ([ASSUMPTIONS.md](ASSUMPTIONS.md)).

## Example

```python
import ptxv  # import before the first CUDA work
import torch
from ptxv import dist as D

a = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)

with ptxv.capture() as ref:
    y_ref = a @ b

with ptxv.capture() as cand:
    y = my_matmul(a, b)

# 1. Execution: races, deadlocks and hangs over all schedules
print(cand.kernels[0].check())
# Verdict(classes=['clean'], certified, ...)

# 2. Numerics: equivalence to the reference, with a proven bound per output
eq = ptxv.compare(ref.kernels[0], cand.kernels[0])
print(eq.headline)
# PROVEN L1 (same function) for every input, |C - R| <= ... * sum|terms|

# 2. Numerics: distribution of C - R under an input distribution
#    (analyzed once, then queried per distribution)
model = ptxv.errdist_model(cand.kernels[0], ref=ref.kernels[0])
e = model(D.Normal(0, 1))
print(e.tier, e.meta["p_differ"], e.worst_case)
print(model(D.Normal(0, 1).quantize("mxf8")))
```

The same steps from the command line, on captured case directories: `ptxv check <case>`, `ptxv compare <ref> <cand>`, `ptxv errdist <equiv.toml> --inputs <spec.json>`. The end of an `errdist` report for a bf16 GEMM pair (cuBLAS against a Triton kernel that sums K in reverse), inputs ~ N(0, 1):

```
summary: tier 2 (CLT on both DAGs), agrees with the tier 3 cross-check
    P(bits differ) = 3.829e-5 (tier 3: 3.499e-5 [2.533e-5, 4.714e-5])
    error in output ulps (|k|): median 0, 99% within 0, 99.9% within 0, largest seen in tier 3: 4
    worst case (proven): |C - R| <= 2 bf16 ulps of the exact result + 3.56e-6 x sum|a_k*b_k| + 9.2e-41
```

errdist answers with an analytic central-limit model (Tier 2) when its applicability checks pass and a small Monte Carlo run (Tier 3) agrees; otherwise with Tier 3. Inputs that make every rounding exact are proven to give zero error (Tier 0). See [docs/errdist.md](docs/errdist.md).

![errdist: the distribution of output error, in ulps, for two softmax kernels](docs/img/errdist_softmax.png)

*errdist on two Triton softmax kernels over 64 f32 logits: `exp(x) / sum` and the max-subtracted `exp(x − max) / sum`. Each stem is the probability of an output error of k ulps (256 outputs per input). Near 0 both stay within a few ulps. Near 50 the naive kernel loses tens of ulps: Triton computes exp(x) as 2^(x·log2 e), and the f32 rounding of an exp2 argument near 72 is magnified in the result.*

## Build

```
source env.sh
cargo build --release
pip install -e python/
```

Requires a Rust toolchain and the CUDA 13 toolkit (`nvdisasm`, `cuobjdump`). Checking a kernel needs no GPU; capturing kernels and validating against hardware need a Blackwell GPU.

## Testing

```
cargo test --release                        # quick tier
cargo test --release -- --include-ignored   # full tier
```

The quick tier is the default: tests that take more than 30 s are marked `#[ignore = "heavy: <reason>"]` and skipped. It ran in about 6 minutes (342 s, after the build) on a 144-core machine. The full tier also runs the heavy tests (about 6 hours of test time, so run it on many cores), and the few tests ignored for other reasons. See [CONTRIBUTING.md](CONTRIBUTING.md#testing).

## Trust

A clean verdict is not a proof (see the warning at the top). Every verdict lists the modeling assumptions it relies on ([ASSUMPTIONS.md](ASSUMPTIONS.md)). The semantics of instructions, the memory model and the arithmetic units are encoded by hand and validated against GB300 hardware; they are not a formal specification. Unsupported instructions are rejected; an audit so that unsupported modifiers are also rejected, rather than ignored, is in progress.

## Related verifiers

![GPU kernel verifiers against the ideal](docs/img/verifiers.png)

*What each tool's own sources claim, per property. Y: claimed within the tool's stated scope, not complete coverage; P: partial or restricted; N: the source says unsupported or out of scope; ?: not stated in the sources read. Tools published before 2024 are rated from the related-work sections of later papers. The ptxv row applies the same standard to ptxv's documentation; its P cells are the open gaps: equivalence is checked against a reference kernel rather than a specification, verdicts hold for the captured inputs and launch shape, and not every proof rule has an adversarial test.*

## Design

**Execution.** The SASS (or PTX) of the captured launch is lifted to one IR, including the SASS scoreboard and dependency-barrier bits. A simulator holds every warp, async unit (TMA, tcgen05, remote stores) and barrier as explicit state, and an explorer with partial-order reductions visits the schedules the hardware may pick. Along each schedule, vector clocks track the happens-before relation of the PTX memory model (barriers, mbarrier phases, release/acquire with scopes, proxy fences, async completions). Two conflicting accesses left unordered by it, and not both morally strong, are a race; a state from which nothing can move before all threads exit is a deadlock. By the model's DRF-SC property, a race-free result under interleavings holds for the weak model (for the async, tcgen05 and tensormap proxies this is assumed; see ASSUMPTIONS.md). When every barrier phase, wait and value is determined, a certificate (E-WELLSYNC) shows that one run decides all schedules, and replaces the search. Each finding carries the schedule that reaches it.

![A witness schedule on SASS](docs/img/sim.gif)

*fixtures/e_stasync, variant SA2: CTA 0 reads a word that CTA 1 stores into its shared memory (STAS) without waiting on the mbarrier; ptxv's witness replayed step by step, ending on the verdict and on the certified `good` variant.*

**Numerics.** Step 1 also establishes that every schedule computes each output with the same operations in the same order, or lists the variants (e.g. floating-point atomics). The numerics DAG of one execution is then the kernel's dataflow on every schedule: leaves are input elements; nodes are tensor-core steps, adds, FMAs and roundings, each with a model of its arithmetic (the tensor-core models are calibrated on GB300). errdist evaluates the DAG under an input distribution: Tier 0 proves the error is zero when every rounding is exact on the inputs' lattice; Tier 2 propagates per-node rounding laws with a central-limit model, and is reported when its applicability checks pass and Tier 3 agrees; Tier 3 is Monte Carlo through the DAG, bit-exact per sample. For a pair such as the one below, the two DAGs share their terms and differ in the order of the roundings, which is where the outputs differ.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/errdist_dag-dark.svg">
  <img src="docs/img/errdist_dag.svg" alt="The numerics DAG of one output for two kernels">
</picture>

*The DAG of out[0, 0] for a 64×64×64 bf16 GEMM (fixtures/c_equiv_mm_bf16): cuBLAS, and a Triton kernel that walks its two K blocks in reverse. The same 64 products enter the f32 accumulator in a different order. Drawn from ptxv's own model by [tools/figures/dag.py](tools/figures/dag.py).*

Details: [docs/design.md](docs/design.md) (§0, §3) and [docs/errdist.md](docs/errdist.md) (§4, §5).

## Documentation

Start with [docs/capture.md](docs/capture.md) and [docs/python.md](docs/python.md) to check your own kernels, then [docs/design.md](docs/design.md) for how the checks work.

| topic | docs |
|---|---|
| **Using ptxv** | |
| capturing kernels from PyTorch, Triton, CuTeDSL | [capture.md](docs/capture.md) |
| Python API | [python.md](docs/python.md) |
| JSON output of the CLI | [json.md](docs/json.md) |
| torch.compile vs eager, kernel by kernel | [compile-vs-eager.md](docs/compile-vs-eager.md) |
| **Execution checking** | |
| semantics, search, certificates | [design.md](docs/design.md) |
| SASS control bits (scoreboards, dependency barriers) | [sass-ctrl.md](docs/sass-ctrl.md) |
| SASS instruction coverage | [lift-coverage.md](docs/lift-coverage.md) |
| large grids, input coverage and their cost | [grid-cost.md](docs/grid-cost.md), [input-coverage-cost.md](docs/input-coverage-cost.md) |
| multi-GPU and multi-node: NVLink, NVLS multicast, GPU-initiated RDMA | [multinode.md](docs/multinode.md), [nvls.md](docs/nvls.md), [nic.md](docs/nic.md) |
| worked examples, from one TMA to MoE and all-reduce | [rung0](docs/rung0.md), [1a](docs/rung1a.md), [1b](docs/rung1b.md), [1c](docs/rung1c.md), [1d](docs/rung1d.md), [1e](docs/rung1e.md), [1f](docs/rung1f.md), [1g](docs/rung1g.md), [2](docs/rung2.md), [2e](docs/rung2e.md), [3b](docs/rung3b.md), [4](docs/rung4.md) |
| **Numerics** | |
| equivalence of two kernels | [compare.md](docs/compare.md), [equiv-reassoc.md](docs/equiv-reassoc.md), [equiv-mma.md](docs/equiv-mma.md), [equiv-elementwise.md](docs/equiv-elementwise.md) |
| error distributions (`errdist`) | [errdist.md](docs/errdist.md) |
| low-precision formats (fp8, fp4, MX, NVFP4) | [lowp.md](docs/lowp.md) |
| **Trust and validation** | |
| modeling assumptions behind each verdict | [ASSUMPTIONS.md](ASSUMPTIONS.md) |
| the bug suite ptxv must catch | [BUGS.md](BUGS.md) |
| finding false cleans: mutants checked against GB300 | [soundness-harness.md](docs/soundness-harness.md), [soundness-results.md](docs/soundness-results.md) |
| hardware litmus tests behind the memory model | [smem-litmus.md](docs/smem-litmus.md), [poll-acq-litmus.md](docs/poll-acq-litmus.md), [smem-order-mapping.md](docs/smem-order-mapping.md), [tensormap.md](docs/tensormap.md) |
| **Project** | |
| roadmap and known gaps | [ROADMAP.md](ROADMAP.md) |
| fixtures that need regeneration (third-party kernels) | [fixtures.md](docs/fixtures.md) |
| contributing and testing | [CONTRIBUTING.md](CONTRIBUTING.md) |

## License

ptxv is BSD licensed, as found in the [LICENSE](LICENSE) file.

Contributing: [CONTRIBUTING.md](CONTRIBUTING.md). Changes: [CHANGELOG.md](CHANGELOG.md).
