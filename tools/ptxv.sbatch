#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-node ptxv on Slurm CPU nodes (docs/multinode.md). Whole nodes are often taken: ask for
# ranks of a few dozen cores each, which Slurm packs onto partly used nodes, e.g.
#   PTXV_SLURM_ROOT=/shared/me sbatch --ntasks=16 --cpus-per-task=48 --mem-per-cpu=4G \
#       --output=/shared/me/ptxv-runs/%j.out $PTXV_SLURM_SBATCH_ARGS tools/ptxv.sbatch \
#       check fixtures/r2_cublas_gemm --slices --max-states 2000000000
# Cluster-specific options (partition, account, QoS, ...) go on the sbatch command line, e.g.
# from $PTXV_SLURM_SBATCH_ARGS. Layout under PTXV_SLURM_ROOT (required, absolute):
#   ptxv/                       the source tree (the job runs there; case paths are relative to it)
#   ptxv-target-x86/release/ptxv  the build for the compute nodes (override: PTXV_SLURM_BIN)
#   ptxv-runs/<job>.dist/       per-rank stderr and the explorer's exchange files
# Set PTXV_SLURM_CUDA_HOME if the CUDA toolkit is not in /usr/local/cuda-13.0.
#SBATCH --job-name=ptxv
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
set -e
ROOT=${PTXV_SLURM_ROOT:?set PTXV_SLURM_ROOT to the shared directory}
BIN=${PTXV_SLURM_BIN:-$ROOT/ptxv-target-x86/release/ptxv}
cd "$ROOT/ptxv"
export PTXV_DIST_DIR=$ROOT/ptxv-runs/$SLURM_JOB_ID.dist
export PTXV_THREADS=${PTXV_THREADS:-$SLURM_CPUS_PER_TASK}
export PATH=${PTXV_SLURM_CUDA_HOME:-/usr/local/cuda-13.0}/bin:$PATH
mkdir -p "$PTXV_DIST_DIR"
echo "job $SLURM_JOB_ID: $SLURM_NTASKS ranks on $SLURM_NNODES nodes ($SLURM_JOB_NODELIST), $PTXV_THREADS threads each: ptxv $*"
srun --kill-on-bad-exit=1 bash -c \
  'exec "$0" "$@" 2>"$PTXV_DIST_DIR/err.$SLURM_PROCID"' "$BIN" "$@"
