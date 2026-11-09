// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Reads driver constant-bank words c[0x0][0x2d0/0x2d8/0x2dc] (patched LDC) on every GPU,
// before and after creating a multicast object over all of them.
#include <cuda.h>
#include <stdio.h>
#include <vector>
#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s; cuGetErrorString(r_, &s); printf("%s -> %s\n", #x, s); return 1; } } while (0)
static int readall(int n, std::vector<CUcontext> &ctx, const char *tag) {
    const char *names[] = {"k2d0.cubin", "k2d8.cubin", "k2dc.cubin"};
    for (int d = 0; d < n; d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        printf("%s dev %d:", tag, d);
        for (auto nm : names) {
            CUmodule m; CUfunction f; CK(cuModuleLoad(&m, nm)); CK(cuModuleGetFunction(&f, m, "k"));
            CUdeviceptr o; CK(cuMemAlloc(&o, 4)); CK(cuMemsetD32(o, 0xdeadbeef, 1));
            unsigned v = 0; void *a[] = {&o, &v};
            CK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, 0, a, 0)); CK(cuCtxSynchronize());
            unsigned h; CK(cuMemcpyDtoH(&h, o, 4)); printf(" %s=%#x", nm, h);
            cuMemFree(o); cuModuleUnload(m);
        }
        printf("\n");
    }
    return 0;
}
int main() {
    CK(cuInit(0)); int n; CK(cuDeviceGetCount(&n));
    std::vector<CUcontext> ctx(n);
    for (int d = 0; d < n; d++) { CUdevice dev; CK(cuDeviceGet(&dev, d)); CK(cuDevicePrimaryCtxRetain(&ctx[d], dev)); }
    if (readall(n, ctx, "plain")) return 1;
    if (n > 1) {
        CUmulticastObjectProp p = {}; p.numDevices = n; p.size = 2 << 20; p.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CUmemGenericAllocationHandle mc; CK(cuMulticastCreate(&mc, &p));
        for (int d = 0; d < n; d++) { CUdevice dev; CK(cuDeviceGet(&dev, d)); CK(cuMulticastAddDevice(mc, dev)); }
        if (readall(n, ctx, "after-mc")) return 1;
    }
    return 0;
}
