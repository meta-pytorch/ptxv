/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Host for the MUFU sweep: loads mufu.cubin (and the patched mufu_raw.cubin) with the driver
// API and streams one op's outputs, in input order, as raw little-endian u32 to stdout.
//   probe <dir> <kernel> [start count]     (default: all 2^32 inputs)
//   kernel "raw:k_sin" takes k_sin from mufu_raw.cubin.
// g++ -O2 -o probe probe.cpp -I/usr/local/cuda/include -lcuda
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#define CK(x)                                                                            \
    do {                                                                                 \
        CUresult r = (x);                                                                \
        if (r != CUDA_SUCCESS) {                                                         \
            const char* s;                                                               \
            cuGetErrorString(r, &s);                                                     \
            fprintf(stderr, "%s: %s\n", #x, s);                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: probe <dir> <kernel> [start count]\n");
        return 2;
    }
    std::string dir = argv[1], kname = argv[2], file = dir + "/mufu.cubin";
    if (kname.rfind("raw:", 0) == 0) {
        kname = kname.substr(4);
        file = dir + "/mufu_raw.cubin";
    }
    uint64_t start = argc > 3 ? strtoull(argv[3], 0, 0) : 0;
    uint64_t count = argc > 4 ? strtoull(argv[4], 0, 0) : (1ull << 32);
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CUmodule mod;
    CK(cuModuleLoad(&mod, file.c_str()));
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, kname.c_str()));
    const uint64_t CH = 1ull << 26;
    CUdeviceptr d;
    CK(cuMemAlloc(&d, CH * 4));
    uint32_t* h;
    CK(cuMemAllocHost((void**)&h, CH * 4));
    for (uint64_t off = 0; off < count; off += CH) {
        uint32_t base = (uint32_t)(start + off);
        uint32_t n = (uint32_t)((count - off) < CH ? (count - off) : CH);
        void* args[] = {&base, &n, &d};
        CK(cuLaunchKernel(fn, 4096, 1, 1, 256, 1, 1, 0, 0, args, 0));
        CK(cuMemcpyDtoH(h, d, (size_t)n * 4));
        if (fwrite(h, 4, n, stdout) != n) {
            fprintf(stderr, "short write\n");
            return 1;
        }
    }
    fflush(stdout);
    return 0;
}
