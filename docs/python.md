# Python API (`python/ptxv`)

**Status (2026-11-04).** `import ptxv` captures the kernels a region of a PyTorch, Triton or
CuTeDSL program launches, in process, as ptxv cases. It runs `check` and `compare` on them
through a session: local subprocesses, a `ptxvd` daemon, or Slurm. Results come back as
objects. The proven equivalence bound τ(x) is a vectorized function of the inputs. Error
distributions go through an isolated adapter to `ptxv errdist` (agent T). Until that
subcommand is merged, the distribution of τ(x) works.

Install: `pip install -e python/` (deps: numpy; torch and pandas optional). The `ptxv`
binary is found through `$PTXV_BIN`, `bin = ".."` in `~/.config/ptxv/config.toml`,
`$CARGO_TARGET_DIR/release/ptxv` or the checkout's `target/release/ptxv`. When none of these
exists, `cargo build --release` runs in the checkout. Caches (cases, results, module images,
the capture library) go to `$PTXV_CACHE` (default `$TMPDIR/ptxv-<uid>`).

## Examples

A torch op:

```python
import ptxv                      # before the first CUDA work (see Capture)
import torch, torch.nn.functional as F

x = torch.randn(4, 4096, device="cuda"); w = torch.randn(4096, device="cuda")
with ptxv.capture() as cap:
    y = F.rms_norm(x, (4096,), w)
cap.kernels       # [Kernel('vectorized_layer_norm_kernel', grid=(4, 1, 1), block=(...))]
k = cap.kernels[0]
k.case()          # Case('vectorized_layer_norm_kernel', outputs=['y', 'W0'], ...): x, w, y named
k.check()         # Verdict(classes=['clean'], certified, states=312, sim=GPU)
```

A Triton kernel against torch (both captured on the same tensors):

```python
from triton_cumsum import run    # tools/capture_examples/triton_cumsum.py
eq = ptxv.compare("torch.cumsum(x, -1)", run, inputs={"x": x})
eq                # Equivalence(level='L1', proven=True, bound=<tau(x): 512 outputs, max 15u>)
eq.headline       # 'PROVEN L1 (same function) for every input, |C - R| <= 15.00u * sum|terms|; GPU: ...'
tau = eq.bound(x=x)                       # per-output bound, a tensor shaped like the output
eq.bound.check(torch.cumsum(x, -1), run(x), {"x": x})   # {'violations': 0, 'max_fraction': 0.59, ...}
eq.bound(x=torch.randn(16, 4, 128, device="cuda"))      # batched: (16, 4, 128)
```

A pair of existing cases (no GPU):

```python
eq = ptxv.compare("fixtures/c_cumsum", "fixtures/c_triton_cumsum")
```

Error distributions:

```python
from ptxv import dist as D
xd = D.Normal(0, 1).per_channel(D.LogNormal(0, 1)).quantize("mxf8", scales="SFA")
eq.bound.distribution({"x": D.Normal(0, 1).quantize("bf16")}, n=1000)
#   ErrDist(metric='bound', n=1000, outputs=512, mean=4.04e-05, p50=3.9e-05, p99=9.08e-05, ...)
ptxv.errdist(k_cand, {"A": xd, "B": D.Laplace(0, 0.5)}, ref=k_ref)   # needs `ptxv errdist`
```

Long jobs: `wait=False` returns a `Job`, a `concurrent.futures.Future` with `.progress`,
`.log`, `add_progress_callback` and a `cancel()` that also stops a running job.

```python
job = k.check(wait=False, max_states=10**8)
job.progress      # the explorer's last progress line
v = job.result()
```

## Capture

`ptxv.capture(names=None, *, snapshots="all", max_alloc="1G", budget="16G", ops=False,
write_only=())` is a context manager. The calling thread runs the region normally.

- **Mechanism: CUPTI callbacks** (`python/ptxv/_native/ptxvcap.c`, built with gcc on first
  use and loaded with ctypes; it links neither libcupti nor libcuda).
  - Driver-API callbacks on `cuLaunchKernel`, `cuLaunchKernelEx`, `cuLaunchCooperativeKernel`
    (and `_ptsz`) record the function, grid, block, cluster, PDL attribute, dynamic smem and
    parameter bytes (`cuFuncGetParamInfo`; packed `extra` buffers as one blob). These
    callbacks are enabled only inside a region. Runtime launches (`<<<>>>`,
    `cudaLaunchKernel`) arrive here too.
  - Snapshots: at API_ENTER, the context is synchronized and every allocation a pointer word
    of the parameters reaches (`cuMemGetAddressRange`) is copied out. At API_EXIT, after
    another synchronize, the same allocations are copied again. Each launch is therefore a
    self-contained case, whatever ran before it.
  - Pointer tables (2026-11-06): the first 1 MB from each parameter pointer (within its
    allocation; before 2026-11-07, only whole allocations up to 1 MB, which missed tables in
    torch's 2 MiB small-tensor segments) is read and scanned for 8-byte words inside other
    allocations; those allocations are snapshotted
    too (one level). The case writer turns such words in non-float buffers into `ptrs`
    entries (filled with the target buffer's address in the simulator and on hardware), so a
    multi-tensor kernel that loads tensor addresses from a table reads the captured tensors.
    A data word that happens to equal a device address is taken for a pointer (warned as
    "pointer words inside buffers").
  - Module images: `CUPTI_CBID_RESOURCE_MODULE_LOADED` from `import ptxv` on, written once
    per content hash. When several captured modules define the launched name (CuTeDSL
    truncates specialized kernel names), modules compiled for another block shape
    (`EIATTR_REQNTID`) are dropped first; then the module the launch recorded (handle map)
    is used, else one whose parameter layout (`EIATTR_KPARAM_INFO`) matches the launch's,
    with a warning if several distinct images match. The handle map needs the image to load
    inside the load call: under `CUDA_MODULE_LOADING=LAZY` (the default) a library's module
    loads at its first launch and later launches of it have no recorded image. Capture with
    `CUDA_MODULE_LOADING=EAGER` when several specializations share a kernel name.
- **Why CUPTI.** launchcap (`probes/launchcap`) interposes symbols with LD_PRELOAD. Loaded
  into a running process, it misses every entry point the runtime already resolved through
  `cuGetProcAddress`. CUPTI sees every launch, needs no re-exec, and can be switched on and
  off around a region. The cost: module images exist only for modules loaded after
  `import ptxv`. For programs that import CUDA libraries first, `python -m ptxv run
  script.py` imports ptxv before the script runs.
- **Not perturbing.** The synchronizations and copies happen outside the stream's work, so
  outputs are unchanged. The GPU test compares them bitwise, with and without capture. Timing
  changes.
- **Buffers.**
  - Boundaries come from torch's caching allocator: blocks live at entry, plus allocations
    recorded during the region by `torch.cuda.memory._record_memory_history`. A block
    allocated in the region counts only if it is live at a launch of the group (trace
    `time_us` against the launch record's `t_us`, both wall clock): a freed block whose memory
    two later tensors share does not join them into one buffer.
  - A pointer equal to a named tensor's end belongs to that tensor only if nothing follows it
    in the buffer's address range.
  - Names, dtypes and shapes come from tensors bound to local variables of the calling frame
    (or `names=`). Other buffers are `W0`, `W1`, ... as u8.
  - Without torch, buffers are whole driver allocations.
- **Stale names.** A named tensor smaller than the live allocator block at its address
  (beyond the allocator's 512-byte rounding) does not bound the buffer: the buffer takes the
  block's extent, with a warning (a workspace reusing a freed tensor's block). Where a tensor
  map reads a buffer as an element type of another size than its name says, the buffer takes
  the tensor map's type.
- **Vacuous inputs.** capture.json `warnings` name every input buffer whose bytes are all
  one value (all zeros: a buffer a collective or another stream fills after the snapshot, so
  V = 0 gives O = 0 whatever the kernel computes), and every launch outside the case that
  points into an input after that input's snapshot and before the case's last launch (its
  writes are not in the initial contents). `fb/kernel_db/scan_inputs.py` finds uniform
  inputs in existing case directories (single- and multi-device).
- **Outputs.**
  - An output is a buffer the group changed, or one allocated inside the region (it may get
    back the bytes it held).
  - Every output starts from its pre-launch snapshot (2026-11-06; before, outputs allocated
    in the region started as zeros): bytes no kernel writes, e.g. rows of a `torch.empty`
    output a kernel leaves alone, compare equal with the captured output. Older outputs keep
    their bytes, as for an in-place update. Outputs listed in `write_only` start as zeros.
  - Override with `case(outputs=[...])`.
- **Selection.**
  - `cap.kernels` deduplicates by kernel and launch shape (`Kernel.instances` holds every
    launch). `cap.kernels["regex"]`, `cap.kernels.filter(name=, op=)`.
  - `cap.select(name=, op=, index=)` and `cap.all()` give multi-launch groups, written as one
    case with `[[launches]]`.
  - `ops=True` tags each launch with the aten op that issued it (a TorchDispatchMode: off by
    default, since it changes how torch.compile traces).
- **Case format.** The same as tools/capture.py writes (docs/capture.md): `case.toml`,
  `data/`, `expected/`, cubins pruned to the launched kernels, and `capture.json`.
  `Launches.write(dir)` exports one. `case(ptx=True)` makes a PTX case when Triton's cache
  holds a cubin identical to the captured one (`jit.ptx` is kept either way).
- **Warnings** (`cap.warnings`, `capture.json`):
  - CUDA graph launches (not captured);
  - launches under stream capture (no snapshots);
  - several streams;
  - allocations over `max_alloc`, or snapshots over `budget`;
  - an allocation that changed between two launches of a group, outside the group.
- `ptxv.capture_call(fn, inputs)` (what `compare` uses for callables) runs `fn` once to warm up
  (lazy loading, autotuning), restores inputs it changed, then captures. `fn` may be a torch
  expression string over the inputs.
- **Triton on this box:** Triton's bundled ptxas does not know sm_103a. Set
  `TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas` (and `_CUOBJDUMP_`, `_NVDISASM_`).

## Results

| object | from | fields |
|---|---|---|
| `Verdict` | `check` | `classes`, `certified`, `complete`, `states`, `findings`, `assumptions`, `notes`, `sim_vs_gpu`, `sim_equals_gpu`, `clean`, `json` |
| `Equivalence` | `compare` | `level`, `proven`, `headline`, `blockers`, `relies_on`, `witness`, `gpu`, `per_output_bound`, `bound`, `reference`, `candidate`, `markdown`, `json` |
| `Bound` | `--emit-bound` | `__call__(**inputs)` (numpy or torch, batched), `check(ref, cand, inputs)`, `distribution(dists, n)`, `evaluable`, `used_inputs`, `max_u()`, `to_pandas()`, `json` |
| `ErrDist` | `errdist`, `Bound.distribution` | `quantile(q)`, `quantiles`, `mean`, `std`, `max`, `cdf(t)`, `tail(t)` (with a Clopper-Pearson bound), `per_output()`, `sample(n)`, `to_numpy()`, `to_pandas()`, `json` |

`Bound` evaluates the `sum`, `round` and `softmax` forms (docs/json.md). For `norm` and
attention outputs it returns NaN: their evaluation data is not exported yet.

## Sessions and backends

`ptxv.Session(backend="local", *, cache=True, workers=2, mem_cap="64G", threads=None)`.
`with Session(...)` makes it the default for `Kernel.check()` and the other front ends.

- **local** (`LocalBackend`).
  - Each job is one `ptxv <cmd> --json` subprocess under `systemd-run --user --scope -p
    MemoryMax=<mem_cap> -p MemorySwapMax=0`, with `CUDA_VISIBLE_DEVICES=""` and
    `PTXV_THREADS = cores / workers`.
  - The child runs in its own session. Cancelling kills that process group, by PID.
- **ptxvd** (`Session("ptxvd://127.0.0.1:8765")`, server `python -m ptxv.daemon`).
  - stdlib HTTP + JSON. Binds 127.0.0.1 unless `--bind` says otherwise.
  - Case bundles (tar.gz, verified against their content hash) are uploaded once. Jobs queue
    on `--workers` local slots, through the same runner and memory cap.
  - Results are cached by job key. A job identical to a running one shares it. Progress
    streams as NDJSON. The API is in the module docstring of `ptxv/daemon.py`.
- **Slurm** (`SlurmBackend(host="login", root="/shared/dir", ntasks=8, dry_run=False)`,
  `Session("slurm://login?ntasks=16")`, or behind ptxvd with `--slurm-host` and job argument
  `backend="slurm"`).
  - Configuration: constructor arguments, else `PTXV_SLURM_HOST` (ssh login node; unset runs
    sbatch locally), `PTXV_SLURM_ROOT` (absolute directory shared by the nodes; required),
    `PTXV_SLURM_SBATCH_ARGS` (site options such as `--partition=... --account=...`) and
    `PTXV_SLURM_SCRIPT` (default `ROOT/ptxv/tools/ptxv.sbatch`).
  1. Cases and job files are rsynced to `ROOT/ptxv-runs/{cases/<sha>, py/<job>}`.
  2. `sbatch tools/ptxv.sbatch <ptxv args> --json` runs on the login node. `check` gets
     `ntasks` ranks of the distributed explorer (docs/multinode.md); other kinds get 1.
  3. The backend then polls `sacct` and reads rank 0's JSON from `ROOT/ptxv-runs/<id>.out`.
  - `dry_run=True` returns the commands and runs nothing.
  - Enabling it for real needs ssh access to the login node, and the source and build under
    ROOT (docs/multinode.md). Only the dry run is tested.
- **Cache key:** sha256 of the job kind and arguments, the content hash of every case
  directory (cubins, case.toml, input bytes) and the ptxv binary's hash. Slurm results are
  kept in their own namespace (the cluster's build).

## Error distributions (adapter)

- `ptxv.dist`:
  - element distributions: `Normal`, `Uniform`, `Laplace`, `StudentT`, `LogNormal`,
    `Discrete`, `Integers` ([lo, hi)), `Const`, `Empirical(tensor)`, `Mixture`;
  - affine maps: `d * a + b`;
  - wrappers: `.per_channel(scale_dist, axis)` (hierarchical scales), `.outliers(prob,
    factor, axis)`, `.clip(lo, hi)`;
  - `.quantize(recipe, block, axis, scales=, scale_rule=, global_scale=)` with recipes bf16,
    f16, e4m3 / e5m2 / e2m1 casts, fp8 (per-tensor scale), mxf8, mxf8_e5m2, mxf6, mxfp4,
    nvfp4.
  - Every distribution samples in numpy (`sample(shape, rng)`; the dequantized values) and
    serializes (`spec()`, `from_spec`).
- `python/ptxv/_errdist_adapter.py` is the only code that knows `ptxv errdist`:
  - `ptxv.dist` specs go to its input spec v1 (docs/errdist.md §2 in agent T's tree);
  - an `equiv.toml` is written for pairs;
  - report v1 (§3) comes back as a summary-mode `ErrDist` (the `best` estimate's quantiles
    and moments, `out_of_domain`, `validation` in `meta`).
  - Not expressible in spec v1 (`AdapterError`): `clip`, per-tensor-scaled fp8, mxf6.
- `ptxv.errdist(target, inputs, ref=None, metric="abs" | "u" | "ulp", seed=,
  mc_inputs=, sim_trials=, gpu_trials=, tier1=False)` raises NotImplementedError until the
  binary has the subcommand. The headline is Tier 0, else Tier 2 where its applicability
  check and the default 300-input Tier 3 cross-check pass, else Tier 3 (docs/errdist.md
  §3.2); Tier 1 only with `tier1=True`. `ErrDist.tier` is the headline's tier,
  `ErrDist.crosscheck` the cross-check entries.
- Inputs that carry an error (docs/errdist.md §2.1, §10): `d.with_error(e)` with
  `RelativeError(std)`, `AbsoluteError(std)`, `RoundError("bf16")` (an f32 value that went
  through a 16-bit tensor) or `LinearError(alpha, beta, s0, s1)`. The kernel reads the value
  plus the error; the reference is exact on the error-free value.
- Composition: `m = ptxv.errdist_model(case)`; `d = m(inputs, errstate=True)` also writes the
  output's errstate (joint value / error pairs), and `d.as_input()` is that output as the
  next kernel's input (`dist.Upstream(errstate_file, tensor)`). `m1.then(m2, "h", w=Normal())`
  chains models: `m1`'s output feeds input `h` of `m2`, and `(m1.then(m2, "h"))(inputs)` is
  the `ErrDist` of `m2`'s output against the exact chain.

```python
mm = ptxv.errdist_model("chain/mm")      # h = x @ w
rms = ptxv.errdist_model("chain/rms")    # F.rms_norm(h, (64,))
N = dist.Normal()
h = mm({"x": N, "w": N}, errstate=True).as_input()
d = rms({"h": h})                        # tier 2: abs std 1.52e-7 (end to end, tier 3: 1.50e-7)
d = rms({"h": N.with_error(dist.RoundError("bf16"))})   # h went through a bf16 tensor
```

## Tests

`python/tests` (`$PTXV_PYTHON -m pytest python/tests`):
- unit tests, no GPU: distributions and quantizers, the adapter, the bound evaluator against
  `equiv`'s own evaluation, sessions, caching, cancellation by PID, the daemon, the Slurm dry
  run;
- `PTXV_TEST_GPU=3` adds the GPU tests: capture of tiny ops (unperturbed, certified clean,
  simulator = GPU), multi-launch regions and selection, and torch vs Triton cumsum (L1, no
  bound violation).
