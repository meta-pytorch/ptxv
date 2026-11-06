/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SR_CgaSize probe host: launches patched.cubin (patch.py) with several cluster / preferred
// cluster configurations and prints, per configuration, a histogram of each CTA's record
// (kernel.ptx: raw SR_CgaSize, %cluster_nctarank, %cluster_nctaid.x, ...). With an output
// directory, writes each configuration's raw records to <dir>/<name>.bin.
//   gcc -O2 -o run run.c -I/usr/local/cuda-13.0/include -L/usr/local/cuda-13.0/lib64/stubs -lcuda
//   CUDA_VISIBLE_DEVICES=<gpu> ./run patched.cubin expected > results.txt
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x)                                                                    \
    do {                                                                         \
        CUresult r_ = (x);                                                       \
        if (r_ != CUDA_SUCCESS) {                                                \
            const char *s_;                                                      \
            cuGetErrorString(r_, &s_);                                           \
            printf("  error %d (%s) at %s\n", (int)r_, s_, #x);                  \
            return 1;                                                            \
        }                                                                        \
    } while (0)

enum { W = 16 };

static const char *outdir;

static int run(CUfunction f, unsigned grid, unsigned cl, unsigned pref, unsigned smem) {
    char name[64];
    snprintf(name, sizeof name, "g%u_c%u_p%u_s%u", grid, cl, pref, smem ? 1 : 0);
    printf("config %s: grid %u cluster %u preferred %u smem %u\n", name, grid, cl, pref, smem);
    CUdeviceptr d;
    size_t bytes = (size_t)grid * W * 4;
    CK(cuMemAlloc(&d, bytes));
    CK(cuMemsetD8(d, 0, bytes));
    CUlaunchAttribute at[2];
    unsigned na = 0;
    if (cl) {
        at[na].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        at[na].value.clusterDim.x = cl;
        at[na].value.clusterDim.y = 1;
        at[na].value.clusterDim.z = 1;
        na++;
    }
    if (pref) {
        at[na].id = CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION;
        at[na].value.preferredClusterDim.x = pref;
        at[na].value.preferredClusterDim.y = 1;
        at[na].value.preferredClusterDim.z = 1;
        na++;
    }
    CUlaunchConfig c;
    memset(&c, 0, sizeof c);
    c.gridDimX = grid;
    c.gridDimY = 1;
    c.gridDimZ = 1;
    c.blockDimX = 32;
    c.blockDimY = 1;
    c.blockDimZ = 1;
    c.sharedMemBytes = smem;
    c.attrs = at;
    c.numAttrs = na;
    void *args[] = {&d};
    CK(cuLaunchKernelEx(&c, f, args, NULL));
    CK(cuCtxSynchronize());
    unsigned *h = malloc(bytes);
    CK(cuMemcpyDtoH(h, d, bytes));
    CK(cuMemFree(d));
    if (outdir && grid <= 512) {
        char path[512];
        snprintf(path, sizeof path, "%s/%s.bin", outdir, name);
        FILE *fp = fopen(path, "wb");
        if (fp) {
            fwrite(h, 1, bytes, fp);
            fclose(fp);
        }
    }
    // Histogram of (raw, nctarank, ctarank % nctarank == ctaid % nctarank, nctaid.x,
    // clusterid == ctaid / nctaid.x, nclusterid.x, cluster_ctaid.x == ctarank).
    unsigned seen[64][8];
    unsigned cnt[64];
    unsigned ns = 0, missing = 0;
    for (unsigned i = 0; i < grid; i++) {
        unsigned *r = h + i * W;
        if (r[15] != 1 || r[13] != i) {
            missing++;
            continue;
        }
        unsigned raw = r[0];
        int k = (int)raw / -0xa0;
        unsigned key[8] = {raw, (unsigned)k, r[4], r[6], r[10], r[5] == r[11], r[9] == i / (r[6] ? r[6] : 1),
                           r[5] == i % (r[4] ? r[4] : 1)};
        unsigned j;
        for (j = 0; j < ns; j++)
            if (!memcmp(seen[j], key, sizeof key)) break;
        if (j == ns && ns < 64) {
            memcpy(seen[ns], key, sizeof key);
            cnt[ns++] = 0;
        }
        if (j < 64) cnt[j]++;
    }
    for (unsigned j = 0; j < ns; j++)
        printf("  %5u CTAs: raw*-0xa0 %#010x SR_CgaSize %u | nctarank %u nctaid.x %u nclusterid.x %u "
               "cluster_ctaid.x==ctarank %u clusterid==ctaid/nctaid.x %u ctarank==ctaid%%nctarank %u\n",
               cnt[j], seen[j][0], seen[j][1], seen[j][2], seen[j][3], seen[j][4], seen[j][5], seen[j][6],
               seen[j][7]);
    if (missing) printf("  %u CTAs without a record\n", missing);
    free(h);
    return 0;
}

int main(int argc, char **argv) {
    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CK(cuCtxSetCurrent(ctx));
    CUmodule m;
    CK(cuModuleLoad(&m, argc > 1 ? argv[1] : "patched.cubin"));
    outdir = argc > 2 ? argv[2] : 0;
    CUfunction f;
    CK(cuModuleGetFunction(&f, m, "k"));
    unsigned big = 200 * 1024;
    CK(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, big));
    CK(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1));
    struct {
        unsigned grid, cl, pref, smem;
    } cfg[] = {
        {64, 0, 0, 0},     {64, 1, 0, 0},     {64, 2, 0, 0},     {64, 4, 0, 0},
        {64, 8, 0, 0},     {64, 16, 0, 0},    {512, 2, 0, big},  {512, 2, 4, 0},
        {512, 2, 4, big},  {512, 1, 2, big},  {512, 1, 4, big},  {512, 2, 8, big},
        {512, 4, 8, big},  {512, 4, 16, big}, {2048, 2, 4, big}, {2048, 0, 4, big},
    };
    for (unsigned i = 0; i < sizeof cfg / sizeof cfg[0]; i++)
        run(f, cfg[i].grid, cfg[i].cl, cfg[i].pref, cfg[i].smem);
    return 0;
}
