/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// cuTensorMapEncodeTiled driver: reverse-engineering aid for the 128-byte CUtensorMap format
// (docs/tensormap.md). Reads one encode request per stdin line and prints the result.
//
//   gcc -O2 -o tmap_encode tmap_encode.c -I/usr/local/cuda-13.0/include -ldl
//   CUDA_VISIBLE_DEVICES=0 ./tmap_encode < sweep.txt
//
// Input line (whitespace separated, numbers in C notation, '#' starts a comment):
//   dtype rank interleave swizzle l2 oob addr  dim[rank]  stride[rank-1]  box[rank]  estride[rank]
// A line starting with "im2col" instead calls cuTensorMapEncodeIm2col:
//   im2col dtype rank interleave swizzle l2 oob addr  dim[rank]  stride[rank-1]
//          lower[rank-2]  upper[rank-2]  channelsPerPixel pixelsPerColumn  estride[rank]
// A line starting with "im2colw" calls cuTensorMapEncodeIm2colWide:
//   im2colw dtype rank interleave swizzle l2 oob addr  dim[rank]  stride[rank-1]
//           lowerW upperW  channelsPerPixel pixelsPerColumn mode  estride[rank]
// Output line: the input echoed, then " => " and either "ERR <code>" or the 128 bytes as hex
// (byte 0 first).
//
// If $TMAP_ALLOC is set to a byte count, that much device memory is cuMemAlloc'ed once and
// its base is added to every addr (so addr becomes an offset into a real allocation).
//
// Only a context is created (encoding needs no launch); libcuda is dlopened.
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef CUresult (*init_t)(unsigned);
typedef CUresult (*devget_t)(CUdevice *, int);
typedef CUresult (*ctx_t)(CUcontext *, CUdevice);
typedef CUresult (*setctx_t)(CUcontext);
typedef CUresult (*alloc_t)(CUdeviceptr *, size_t);
typedef CUresult (*enc_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
                          const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
                          const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

typedef CUresult (*im2col_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
                             const cuuint64_t *, const cuuint64_t *, const int *, const int *,
                             cuuint32_t, cuuint32_t, const cuuint32_t *, CUtensorMapInterleave,
                             CUtensorMapSwizzle, CUtensorMapL2promotion,
                             CUtensorMapFloatOOBfill);

typedef CUresult (*im2colw_t)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
                              const cuuint64_t *, const cuuint64_t *, int, int, cuuint32_t,
                              cuuint32_t, const cuuint32_t *, CUtensorMapInterleave,
                              CUtensorMapIm2ColWideMode, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

int main(void) {
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW);
    if (!cuda) { fprintf(stderr, "dlopen libcuda: %s\n", dlerror()); return 1; }
    init_t init = (init_t)dlsym(cuda, "cuInit");
    devget_t devget = (devget_t)dlsym(cuda, "cuDeviceGet");
    ctx_t retain = (ctx_t)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    setctx_t setctx = (setctx_t)dlsym(cuda, "cuCtxSetCurrent");
    enc_t enc = (enc_t)dlsym(cuda, "cuTensorMapEncodeTiled");
    im2col_t im2col = (im2col_t)dlsym(cuda, "cuTensorMapEncodeIm2col");
    im2colw_t im2colw = (im2colw_t)dlsym(cuda, "cuTensorMapEncodeIm2colWide");
    if (!init || !devget || !retain || !setctx || !enc) { fprintf(stderr, "dlsym\n"); return 1; }
    CUdevice dev;
    CUcontext ctx;
    if (init(0) || devget(&dev, 0) || retain(&ctx, dev) || setctx(ctx)) {
        fprintf(stderr, "context setup failed\n");
        return 1;
    }
    unsigned long long base = 0;
    if (getenv("TMAP_ALLOC")) {
        alloc_t alloc = (alloc_t)dlsym(cuda, "cuMemAlloc_v2");
        CUdeviceptr d;
        if (!alloc || alloc(&d, strtoull(getenv("TMAP_ALLOC"), 0, 0))) {
            fprintf(stderr, "cuMemAlloc failed\n");
            return 1;
        }
        base = (unsigned long long)d;
        fprintf(stderr, "alloc base 0x%llx\n", base);
    }
    char line[4096];
    while (fgets(line, sizeof line, stdin)) {
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        line[strcspn(line, "\n")] = 0;
        unsigned long long v[64];
        int n = 0;
        char *p = line, *end;
        while (*p == ' ' || *p == '\t') p++;
        int is_im2colw = strncmp(p, "im2colw", 7) == 0;
        int is_im2col = !is_im2colw && strncmp(p, "im2col", 6) == 0;
        if (is_im2colw) p += 7;
        if (is_im2col) p += 6;
        while (n < 64) {
            unsigned long long x = strtoull(p, &end, 0);
            if (end == p) break;
            v[n++] = x;
            p = end;
        }
        if (n == 0) continue;
        if (n < 7) { printf("%s => BADLINE\n", line); continue; }
        v[6] += base;
        unsigned rank = (unsigned)v[1];
        if (is_im2colw) {
            if (!im2colw || rank < 3 || rank > 5 || n != 7 + rank + (rank - 1) + 5 + rank) {
                printf("%s => BADLINE\n", line);
                continue;
            }
            cuuint64_t dim[5], stride[5];
            cuuint32_t es[5];
            int k = 7;
            for (unsigned i = 0; i < rank; i++) dim[i] = v[k++];
            for (unsigned i = 0; i + 1 < rank; i++) stride[i] = v[k++];
            int lo = (int)(long long)v[k++], hi = (int)(long long)v[k++];
            cuuint32_t cpp = (cuuint32_t)v[k++], ppc = (cuuint32_t)v[k++];
            CUtensorMapIm2ColWideMode mode = (CUtensorMapIm2ColWideMode)v[k++];
            for (unsigned i = 0; i < rank; i++) es[i] = (cuuint32_t)v[k++];
            CUtensorMap m;
            memset(&m, 0xa5, sizeof m);
            CUresult r = im2colw(&m, (CUtensorMapDataType)v[0], rank, (void *)(uintptr_t)v[6],
                                 dim, stride, lo, hi, cpp, ppc, es, (CUtensorMapInterleave)v[2],
                                 mode, (CUtensorMapSwizzle)v[3], (CUtensorMapL2promotion)v[4],
                                 (CUtensorMapFloatOOBfill)v[5]);
            printf("%s => ", line);
            if (r != CUDA_SUCCESS) { printf("ERR %d\n", (int)r); continue; }
            const unsigned char *b = (const unsigned char *)&m;
            for (int i = 0; i < 128; i++) printf("%02x", b[i]);
            printf("\n");
            continue;
        }
        if (is_im2col) {
            if (rank < 3 || rank > 5 || n != 7 + rank + (rank - 1) + 2 * (rank - 2) + 2 + rank) {
                printf("%s => BADLINE\n", line);
                continue;
            }
            cuuint64_t dim[5], stride[5];
            int lo[3], hi[3];
            cuuint32_t es[5];
            int k = 7;
            for (unsigned i = 0; i < rank; i++) dim[i] = v[k++];
            for (unsigned i = 0; i + 1 < rank; i++) stride[i] = v[k++];
            for (unsigned i = 0; i + 2 < rank; i++) lo[i] = (int)(long long)v[k++];
            for (unsigned i = 0; i + 2 < rank; i++) hi[i] = (int)(long long)v[k++];
            cuuint32_t cpp = (cuuint32_t)v[k++], ppc = (cuuint32_t)v[k++];
            for (unsigned i = 0; i < rank; i++) es[i] = (cuuint32_t)v[k++];
            CUtensorMap m;
            memset(&m, 0xa5, sizeof m);
            CUresult r = im2col(&m, (CUtensorMapDataType)v[0], rank, (void *)(uintptr_t)v[6],
                                dim, stride, lo, hi, cpp, ppc, es, (CUtensorMapInterleave)v[2],
                                (CUtensorMapSwizzle)v[3], (CUtensorMapL2promotion)v[4],
                                (CUtensorMapFloatOOBfill)v[5]);
            printf("%s => ", line);
            if (r != CUDA_SUCCESS) { printf("ERR %d\n", (int)r); continue; }
            const unsigned char *b = (const unsigned char *)&m;
            for (int i = 0; i < 128; i++) printf("%02x", b[i]);
            printf("\n");
            continue;
        }
        if (rank < 1 || rank > 5 || n != 7 + rank + (rank - 1) + rank + rank) {
            printf("%s => BADLINE\n", line);
            continue;
        }
        cuuint64_t dim[5], stride[5];
        cuuint32_t box[5], es[5];
        int k = 7;
        for (unsigned i = 0; i < rank; i++) dim[i] = v[k++];
        for (unsigned i = 0; i + 1 < rank; i++) stride[i] = v[k++];
        for (unsigned i = 0; i < rank; i++) box[i] = (cuuint32_t)v[k++];
        for (unsigned i = 0; i < rank; i++) es[i] = (cuuint32_t)v[k++];
        CUtensorMap m;
        memset(&m, 0xa5, sizeof m);  // so untouched bytes would show up
        CUresult r = enc(&m, (CUtensorMapDataType)v[0], rank, (void *)(uintptr_t)v[6], dim,
                         stride, box, es, (CUtensorMapInterleave)v[2],
                         (CUtensorMapSwizzle)v[3], (CUtensorMapL2promotion)v[4],
                         (CUtensorMapFloatOOBfill)v[5]);
        printf("%s => ", line);
        if (r != CUDA_SUCCESS) { printf("ERR %d\n", (int)r); continue; }
        const unsigned char *b = (const unsigned char *)&m;
        for (int i = 0; i < 128; i++) printf("%02x", b[i]);
        printf("\n");
    }
    return 0;
}
