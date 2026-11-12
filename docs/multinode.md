# Multi-node exploration

`ptxv` can split one stateful search across many processes on many machines. Set
`PTXV_DIST_DIR` to a directory every process can see, a new one for every launch (the
`addr.<rank>` and `refused.<rank>` files of an earlier launch would be read as this one's).
Each process runs the same command as one rank of the job (`crates/sim/src/dist.rs`,
`DistRound` in `explore.rs`).

## How it works

- **The same program everywhere.** Every rank parses the case, runs the deterministic parts
  (the symmetry run, `clc_eager`, seed walks) itself, and joins each stateful search as a
  collective. Searches are numbered in program order, so the n-th search is the same on
  every rank. Elect leader policies run one after the other in this mode.
- **Visited states are partitioned.** The owner of a state is `fingerprint >> 64 mod ranks`.
  - Only the owner's set decides whether a state is new.
  - A rank that reaches a state owned elsewhere sends the owner 16 bytes and keeps the
    state until the answer arrives. States are not shipped.
  - So each state is expanded by exactly one rank, the one whose question the owner answers
    first, just as threads share one visited set within a process.
- **Work moves by stealing.**
  - An idle rank asks a random rank for work.
  - The victim sends up to 16 of its shallowest queued states as schedules (action
    sequences from the initial state).
  - The thief replays them. This is the only time a state is rebuilt.
- **The synchronization reduction's learned sets** are broadcast as they grow. A choice that
  a set grown elsewhere contradicts marks the round dirty, exactly as with threads.
- **Rank 0 coordinates.** Every rank reports its status every 20 ms: idle or not, work
  messages sent and received, states owned, terminal states.
  - **Termination:** all ranks idle, with sent equal to received, in two consecutive
    complete waves.
  - **State cap:** the sum over ranks.
  - **End of a round:**
    1. Each rank sends every peer a marker, so all learned-set messages sent before it have
       arrived.
    2. Each rank sends rank 0 its findings, outputs, counts and dirty flag.
    3. Rank 0 merges them and sends the result back, so every rank continues with the same
       verdict.
- **Output.** Only rank 0 prints. `out/` dirs are per rank (`--out DIR/rank<r>`; without
  `--out`, each rank's own run directory `out/<case>/run-<host>-<pid>-<start>-<ns>/rank<r>`,
  `<host>` = host name, boot and pid namespace). A killed rank's run directory is removed by a
  later run on the same host, boot and pid namespace only; a batch system that gives each job a
  pid namespace of its own (Slurm with a pid namespace per job) leaves earlier jobs' directories.

**Not supported yet:** exploring every slice when cluster symmetry fails, because the
cross-slice footprint check (`GFoot`) is per process.

**Refused in this mode:** whatever would let each rank decide by its own clock, since ranks
that decide differently run different searches and wait for each other's messages forever:
`check --time-budget` / `PTXV_TIME_BUDGET` (each rank would cut its own phases), and input
mutation and the concolic modes (`--mutate-inputs`, `--mutate-input`, `--concolic`,
`--exhaustive-concolic`: which mutated inputs are checked depends on the coverage run's cap,
`--mutate-budget-seconds`, the capped pass that skips a mutation and the solvers' budgets).
Input coverage alone is not refused: it only adds notes, and rank 0's are reported.

**A rank that stops stops the job.** A rank that ends with an error (an argument it refuses,
a case it cannot load, a failure later) writes `refused.<rank>` with the reason into
`PTXV_DIST_DIR`. Ranks still waiting for it at the rendezvous, and connected ranks once its
connection closes, stop with exit status 1 and that reason instead of waiting for it. A rank
killed by a signal or a panic leaves no such file: the others still wait for it.

## Running

Locally, N ranks on one machine (to test the code path):

```
tools/dist_local.sh 3 check fixtures/r2_cute_small --front sass
```

On a Slurm cluster, with a directory `$PTXV_SLURM_ROOT` shared by the nodes (absolute path)
and site options in `$PTXV_SLURM_SBATCH_ARGS` (partition, account, QoS, ...):

- **Source:** rsync the tree to `$PTXV_SLURM_ROOT/ptxv`.
- **Build:** on a compute node (same architecture as the nodes that run ranks):
  ```
  CARGO_TARGET_DIR=$PTXV_SLURM_ROOT/ptxv-target-x86 cargo build --release --offline
  ```
  If the compute nodes have no network access, run `cargo fetch` on the login node first.
  `PTXV_SLURM_BIN` points the batch script at a binary elsewhere.
- **Run:**
  ```
  sbatch --ntasks=16 --cpus-per-task=48 --mem-per-cpu=4G \
      --output=$PTXV_SLURM_ROOT/ptxv-runs/%j.out $PTXV_SLURM_SBATCH_ARGS tools/ptxv.sbatch \
      check fixtures/r2_cublas_gemm --slices --max-states 3000000000
  ```
  `PTXV_SLURM_ROOT` must be in the job's environment (sbatch exports the caller's by default).
  Whole nodes are usually taken, so ask for ranks of a few dozen cores.
- **Logs:** output in `$PTXV_SLURM_ROOT/ptxv-runs/<job>.out`; each rank's stderr
  (rank 0's progress line) in `<job>.dist/err.<rank>`.
- **From Python:** `SlurmBackend` (docs/python.md) does the copy, sbatch and polling with the
  same variables (`PTXV_SLURM_HOST` for the login node over ssh).

The CPU nodes have the CUDA toolkit (`nvdisasm`, `ptxas`) but no driver. `ptxv-hw` loads
`libcuda` with `dlopen` only when a hardware run asks for it.

## Evidence

Verdicts match single-process runs, apart from capped explorations, whose order differs:

| setup | fixtures |
|---|---|
| 3 local ranks | r0_tma_sum, r1b_pipeline, r1d_clc, r1f_pingpong, e_sym, r2_cute_small (PTX and SASS, every variant) |
| 2 Slurm nodes | r2_cute_small, every variant on both front-ends |
