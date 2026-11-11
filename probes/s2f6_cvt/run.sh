#!/bin/bash
# probes/s2f6_cvt. PTX ISA 9.1 needs ptxas >= 13.1 (PTXAS13_1=<path>); nvdisasm 13.0 prints
# nothing for the F2FP encodings, so kernel.cubin is checked in and the case is binary-only.
#   1. kernel.cubin:    $PTXAS13_1 -arch=sm_103a kernel.ptx -o kernel.cubin
#   2. expected/:       PTXV_GPUS=<idle> ptxv hw-worker probes/s2f6_cvt kernel.cubin --runs 3 \
#                         --dump expected/out.bin
#   3. exhaustive check (2^32 pairs vs. the model, independent of ptxv):
set -e
cd "$(dirname "$0")"
T=$(mktemp -d)
# nvcc 13.0 writes the PTX, then its own ptxas check of the asm fails (expected).
nvcc -arch=compute_103a -ptx exhaustive.cu -o "$T/x.ptx" 2>/dev/null || test -s "$T/x.ptx"
sed -i 's/^\.version 9\.0/.version 9.1/' "$T/x.ptx"
"$PTXAS13_1" -arch=sm_103a "$T/x.ptx" -o "$T/exhaustive.cubin"
cc -O2 exhaustive_host.c -I"$PTXV_CUDA_HOME/include" -L"$PTXV_CUDA_HOME/lib64/stubs" -lcuda -lm -o "$T/host"
"$T/host" "$T/exhaustive.cubin"
