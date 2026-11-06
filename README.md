# ptxv

A verifier for NVIDIA Blackwell (sm_100a / sm_103a) GPU kernels. It runs on the PTX or SASS of a captured launch and checks a kernel in two steps:

1. **Execution.** Data races, deadlocks, hangs and protocol misuse (mbarrier, TMEM, TMA, clusters, multi-GPU memory) over the schedules the hardware may pick. The result also says whether every schedule computes each output with the same operations, or which variants can occur (for example with floating-point atomics).
2. **Numerics.** Because step 1 established which operations produce each output on every schedule, they are extracted from one execution and analyzed: equivalence to a reference kernel with a proven per-output bound, and the distribution of the error under a distribution of inputs.

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

## Build

```
source env.sh
cargo build --release
pip install -e python/
```

Requires a Rust toolchain and the CUDA 13 toolkit (`nvdisasm`, `cuobjdump`). Checking a kernel needs no GPU; capturing kernels and validating against hardware need a Blackwell GPU.

## Trust

Every verdict lists the modeling assumptions it relies on ([ASSUMPTIONS.md](ASSUMPTIONS.md)). The semantics of instructions, the memory model and the arithmetic units are encoded by hand and validated against GB300 hardware; they are not a formal specification. Unsupported instructions are rejected; an audit so that unsupported modifiers are also rejected, rather than ignored, is in progress.

## Documentation

| topic | doc |
|---|---|
| capturing kernels from PyTorch, Triton, CuTeDSL | [docs/capture.md](docs/capture.md) |
| Python API | [docs/python.md](docs/python.md) |
| execution checking: semantics, search, certificates | [docs/design.md](docs/design.md), [docs/sass-ctrl.md](docs/sass-ctrl.md), [docs/rung0.md](docs/rung0.md) – [docs/rung4.md](docs/rung4.md) |
| equivalence of two kernels | [docs/compare.md](docs/compare.md), [docs/equiv-reassoc.md](docs/equiv-reassoc.md), [docs/equiv-mma.md](docs/equiv-mma.md), [docs/equiv-elementwise.md](docs/equiv-elementwise.md) |
| error distributions (`errdist`) | [docs/errdist.md](docs/errdist.md) |
| low-precision formats (fp8, fp4, MX, NVFP4) | [docs/lowp.md](docs/lowp.md) |
| torch.compile vs eager | [docs/compile-vs-eager.md](docs/compile-vs-eager.md) |
| SASS instruction coverage | [docs/lift-coverage.md](docs/lift-coverage.md) |
| multi-node exploration | [docs/multinode.md](docs/multinode.md) |
| JSON output | [docs/json.md](docs/json.md) |
| fixtures that need regeneration (third-party kernels) | [docs/fixtures.md](docs/fixtures.md) |
| roadmap and known gaps | [ROADMAP.md](ROADMAP.md), [BUGS.md](BUGS.md) |

## License

BSD-3-Clause (proposed, subject to Meta open source review); see [LICENSE](LICENSE). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).
