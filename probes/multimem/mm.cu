// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// NVLS multicast (multimem.*) probe for GB300. One process drives every visible GPU.
//
//   nvcc -O2 -arch=sm_103a -o mm mm.cu -lcuda
//   ./mm info                        MULTICAST_SUPPORTED per device, granularity
//   ./mm ldr OUTDIR [order]          multimem.ld_reduce numerics: per-member inputs + results
//                                    issued from every device, REPS times (dumped as .bin)
//   ./mm red OUTDIR                  concurrent multimem.red from every device: member contents
//   ./mm litmus OUTDIR               message passing: multimem.st data + multimem.red flag on one
//                                    device, poll + unicast read on the others (variants)
//   ./mm litmus2 OUTDIR              reverse: unicast (st / TMA) writes + multimem.red flag on
//                                    every device, poll + multimem.ld_reduce read (variants)
//
// [order]: device order for cuMulticastAddDevice / BindMem, e.g. "3,2,1,0" (default ascending).
// The analysis is analyze.py.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>

#define CK(x)                                                                       \
    do {                                                                            \
        CUresult r_ = (x);                                                          \
        if (r_ != CUDA_SUCCESS) {                                                   \
            const char *s_ = 0;                                                     \
            cuGetErrorString(r_, &s_);                                              \
            fprintf(stderr, "%s:%d %s -> %d %s\n", __FILE__, __LINE__, #x, r_, s_); \
            exit(1);                                                                \
        }                                                                           \
    } while (0)
#define RK(x)                                                                                \
    do {                                                                                     \
        cudaError_t r_ = (x);                                                                \
        if (r_ != cudaSuccess) {                                                             \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(r_)); \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

// ------------------------------------------------------------------ kernels
// One 16-byte vector per thread, or one 32-bit word (W = 1) / 64-bit word (W = 2).
#define LDR4(NAME, INS)                                                                  \
    __global__ void NAME(const uint4 *mc, uint4 *out, int n) {                            \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                    \
        if (i >= n) return;                                                               \
        uint32_t a, b, c, d;                                                              \
        asm volatile(INS " {%0,%1,%2,%3}, [%4];"                                          \
                     : "=r"(a), "=r"(b), "=r"(c), "=r"(d)                                 \
                     : "l"(mc + i)                                                        \
                     : "memory");                                                         \
        out[i] = make_uint4(a, b, c, d);                                                  \
    }
#define LDR1(NAME, INS)                                                                  \
    __global__ void NAME(const uint32_t *mc, uint32_t *out, int n) {                      \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                    \
        if (i >= n) return;                                                               \
        uint32_t a;                                                                       \
        asm volatile(INS " %0, [%1];" : "=r"(a) : "l"(mc + i) : "memory");                \
        out[i] = a;                                                                       \
    }
#define LDR2(NAME, INS)                                                                  \
    __global__ void NAME(const uint64_t *mc, uint64_t *out, int n) {                      \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                                    \
        if (i >= n) return;                                                               \
        uint64_t a;                                                                       \
        asm volatile(INS " %0, [%1];" : "=l"(a) : "l"(mc + i) : "memory");                \
        out[i] = a;                                                                       \
    }

LDR4(k_f32x4, "multimem.ld_reduce.relaxed.sys.global.add.v4.f32")
LDR1(k_f32, "multimem.ld_reduce.relaxed.sys.global.add.f32")
LDR1(k_f32w, "multimem.ld_reduce.weak.global.add.f32")
LDR4(k_bf16x8, "multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2")
LDR4(k_bf16x8a, "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2")
LDR4(k_f16x8, "multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2")
LDR4(k_f16x8a, "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.f16x2")
LDR4(k_e4m3x16a, "multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.e4m3x4")
LDR4(k_e4m3x16, "multimem.ld_reduce.relaxed.sys.global.add.v4.e4m3x4")
LDR4(k_e5m2x16a, "multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.e5m2x4")
LDR1(k_bf16x2min, "multimem.ld_reduce.relaxed.sys.global.min.bf16x2")
LDR1(k_bf16x2max, "multimem.ld_reduce.relaxed.sys.global.max.bf16x2")
LDR1(k_f16x2max, "multimem.ld_reduce.relaxed.sys.global.max.f16x2")
LDR1(k_u32add, "multimem.ld_reduce.relaxed.sys.global.add.u32")
LDR1(k_s32min, "multimem.ld_reduce.relaxed.sys.global.min.s32")
LDR1(k_u32max, "multimem.ld_reduce.relaxed.sys.global.max.u32")
LDR1(k_xor, "multimem.ld_reduce.relaxed.sys.global.xor.b32")
LDR2(k_u64add, "multimem.ld_reduce.relaxed.sys.global.add.u64")
LDR2(k_f64, "multimem.ld_reduce.relaxed.sys.global.add.f64")

struct LdrTest {
    const char *name;
    void *fn;
    int width;  // bytes per thread: 16, 4 or 8
};
static LdrTest LDR[] = {
    {"f32x4", (void *)k_f32x4, 16},       {"f32", (void *)k_f32, 4},
    {"f32w", (void *)k_f32w, 4},          {"bf16x8", (void *)k_bf16x8, 16},
    {"bf16x8a", (void *)k_bf16x8a, 16},   {"f16x8", (void *)k_f16x8, 16},
    {"f16x8a", (void *)k_f16x8a, 16},     {"e4m3x16a", (void *)k_e4m3x16a, 16},
    {"e4m3x16", (void *)k_e4m3x16, 16},   {"e5m2x16a", (void *)k_e5m2x16a, 16},
    {"bf16x2min", (void *)k_bf16x2min, 4}, {"bf16x2max", (void *)k_bf16x2max, 4},
    {"f16x2max", (void *)k_f16x2max, 4},  {"u32add", (void *)k_u32add, 4},
    {"s32min", (void *)k_s32min, 4},      {"u32max", (void *)k_u32max, 4},
    {"xor", (void *)k_xor, 4},            {"u64add", (void *)k_u64add, 8},
    {"f64", (void *)k_f64, 8},
};

// multimem.red from every device at once on the same elements.
__global__ void k_red_f32(float *mc, const float *src, int n, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int r = 0; r < rounds; r++) {
        float v = src[r * n + i];
        asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], %1;" ::"l"(mc + i), "f"(v)
                     : "memory");
    }
}
__global__ void k_red_bf16x8(uint4 *mc, const uint4 *src, int n, int rounds) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int r = 0; r < rounds; r++) {
        uint4 v = src[r * n + i];
        asm volatile("multimem.red.relaxed.sys.global.add.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(
                         mc + i),
                     "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                     : "memory");
    }
}
__global__ void k_red_s32(int *mc, int n, int rounds, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int r = 0; r < rounds; r++)
        asm volatile("multimem.red.release.sys.global.add.s32 [%0], %1;" ::"l"(mc + i), "r"(rank + 1)
                     : "memory");
}
__global__ void k_st_f32x4(uint4 *mc, const uint4 *src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint4 v = src[i];
    asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(mc + i), "r"(v.x),
                 "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

// Message passing. Writer (one device): for slot k: multimem.st data[k] (16 bytes, weak),
// variant fence, multimem.red flag[k] += 1. Readers (other devices, one warp per slot range):
// poll their own unicast flag[k] until nonzero, variant fence, read data[k] through the unicast
// address, count mismatches.
//   variant 0: red.release / ld.acquire                         (no alias fence)
//   variant 1: red.release / ld.acquire + fence.proxy.alias      (strict PTX)
//   variant 2: red.relaxed / ld.relaxed                          (no ordering: control)
//   variant 3: fence.proxy.alias + red.release / ld.acquire + fence.proxy.alias
__global__ void k_mp_writer(uint4 *data_mc, int *flag_mc, int slots, uint32_t tag, int var) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= slots) return;
    uint32_t v = tag ^ (uint32_t)k;
    asm volatile("multimem.st.weak.global.v4.f32 [%0], {%1,%1,%1,%1};" ::"l"(data_mc + k), "r"(v)
                 : "memory");
    if (var == 3) asm volatile("fence.proxy.alias;" ::: "memory");
    if (var == 2)
        asm volatile("multimem.red.relaxed.sys.global.add.s32 [%0], 1;" ::"l"(flag_mc + k) : "memory");
    else
        asm volatile("multimem.red.release.sys.global.add.s32 [%0], 1;" ::"l"(flag_mc + k) : "memory");
}
__global__ void k_mp_reader(const uint4 *data_uc, const int *flag_uc, int slots, uint32_t tag,
                            int var, unsigned *bad, unsigned *first_bad, int zero) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= slots) return;
    int f = 0;
    long spins = 0;
    while (f == 0 && spins < (1L << 26)) {
        if (var == 2)
            asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(f) : "l"(flag_uc + (k | zero)) : "memory");
        else
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(f) : "l"(flag_uc + (k | zero)) : "memory");
        spins++;
    }
    if (f == 0) {
        atomicAdd(bad + 1, 1u);  // timeout
        return;
    }
    if (var == 1 || var == 3) asm volatile("fence.proxy.alias;" ::: "memory");
    uint32_t a, b, c, d;
    asm volatile("ld.weak.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                 : "l"(data_uc + (k | zero))
                 : "memory");
    uint32_t v = tag ^ (uint32_t)k;
    if (a != v || b != v || c != v || d != v) {
        atomicAdd(bad, 1u);
        atomicCAS(first_bad, 0xffffffffu, (unsigned)k);
    }
}

// ------------------------------------------------------------------ host
// Reverse message passing (unicast write -> multicast read, the dist_gemm RS pattern): every
// member writes its own block of data through its unicast VA, publishes with one multimem.red on
// the block's multicast flag, polls its own unicast flag until every member arrived, then sums
// the block with multimem.ld_reduce and counts lanes != the expected sum.
//   variant 0: st.global data; red.release / ld.acquire           (no alias fence)
//   variant 1: cp.async.bulk data (TMA store, wait_group 0); red.release / ld.acquire (no alias fence)
//   variant 2: st.global; red.relaxed / ld.relaxed                  (no ordering: control)
//   variant 3: variant 1 + fence.proxy.alias before the red and after the poll (strict PTX)
//   variant 4: variant 1 with cp.async.bulk.wait_group.read 0 (source read only: the writes
//              may land after the flag; sensitivity control)
__device__ __forceinline__ uint32_t mp2_val(uint32_t tag, int r, int k, int lane) {
    return (tag ^ (uint32_t)(k * 16 + lane)) + (uint32_t)r * 0x01000193u;
}
__global__ void k_mp2(uint4 *data_uc, const uint32_t *data_mc, int *flag_mc, const int *flag_uc,
                      int rank, int nm, uint32_t tag, int var, unsigned *bad) {
    __shared__ __align__(128) uint4 sbuf[256];
    int t = threadIdx.x, b = blockIdx.x, k = b * 256 + t;
    uint4 v = make_uint4(mp2_val(tag, rank, k, 0), mp2_val(tag, rank, k, 1), mp2_val(tag, rank, k, 2),
                         mp2_val(tag, rank, k, 3));
    if (var == 1 || var == 3 || var == 4) {
        sbuf[t] = v;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (t == 0) {
            unsigned sa = (unsigned)__cvta_generic_to_shared(sbuf);
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 4096;" ::"l"(data_uc + b * 256),
                         "r"(sa)
                         : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            if (var == 4)
                asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
            else
                asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
        }
    } else {
        asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(data_uc + k), "r"(v.x), "r"(v.y), "r"(v.z),
                     "r"(v.w)
                     : "memory");
        __syncthreads();
    }
    __shared__ int ok;
    if (t == 0) {
        if (var == 3) asm volatile("fence.proxy.alias;" ::: "memory");
        if (var == 2)
            asm volatile("multimem.red.relaxed.sys.global.add.s32 [%0], 1;" ::"l"(flag_mc + b) : "memory");
        else
            asm volatile("multimem.red.release.sys.global.add.s32 [%0], 1;" ::"l"(flag_mc + b) : "memory");
        int f = 0;
        long spins = 0;
        while (f < nm && spins < (1L << 26)) {
            if (var == 2)
                asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(f) : "l"(flag_uc + b) : "memory");
            else
                asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(f) : "l"(flag_uc + b) : "memory");
            spins++;
        }
        if (var == 3) asm volatile("fence.proxy.alias;" ::: "memory");
        ok = f >= nm;
        if (!ok) atomicAdd(bad + 1, 1u);
    }
    __syncthreads();
    if (!ok) return;
    unsigned nbad = 0;
    for (int lane = 0; lane < 4; lane++) {
        uint32_t got, want = 0;
        asm volatile("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];"
                     : "=r"(got)
                     : "l"(data_mc + 4 * k + lane)
                     : "memory");
        for (int r = 0; r < nm; r++) want += mp2_val(tag, r, k, lane);
        nbad += got != want;
    }
    if (nbad) atomicAdd(bad, nbad);
}

struct Mc {
    int ndev;
    std::vector<int> devs;  // CUdevice ordinals in add order
    std::vector<CUcontext> ctx;
    size_t size;
    CUmemGenericAllocationHandle mc;
    std::vector<CUmemGenericAllocationHandle> mem;
    std::vector<CUdeviceptr> uc, mcva;
};

static void set_access(CUdeviceptr va, size_t size, int dev) {
    CUmemAccessDesc ad = {};
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad.location.id = dev;
    ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CK(cuMemSetAccess(va, size, &ad, 1));
}

static Mc make_mc(const std::vector<int> &devs, size_t want) {
    Mc m;
    m.ndev = devs.size();
    m.devs = devs;
    for (int d : devs) {
        CUcontext c;
        CK(cuDevicePrimaryCtxRetain(&c, d));
        m.ctx.push_back(c);
    }
    CUmulticastObjectProp p = {};
    p.numDevices = m.ndev;
    p.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    p.size = want;
    size_t gran = 0;
    CK(cuMulticastGetGranularity(&gran, &p, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    m.size = (want + gran - 1) / gran * gran;
    p.size = m.size;
    CK(cuCtxSetCurrent(m.ctx[0]));
    CK(cuMulticastCreate(&m.mc, &p));
    for (int d : devs) CK(cuMulticastAddDevice(m.mc, d));
    for (int i = 0; i < m.ndev; i++) {
        int d = devs[i];
        CK(cuCtxSetCurrent(m.ctx[i]));
        CUmemAllocationProp ap = {};
        ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ap.location.id = d;
        ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        CUmemGenericAllocationHandle h;
        CK(cuMemCreate(&h, m.size, &ap, 0));
        m.mem.push_back(h);
        CK(cuMulticastBindMem(m.mc, 0, h, 0, m.size, 0));
    }
    for (int i = 0; i < m.ndev; i++) {
        int d = devs[i];
        CK(cuCtxSetCurrent(m.ctx[i]));
        CUdeviceptr u, v;
        CK(cuMemAddressReserve(&u, m.size, 0, 0, 0));
        CK(cuMemMap(u, m.size, 0, m.mem[i], 0));
        set_access(u, m.size, d);
        CK(cuMemAddressReserve(&v, m.size, 0, 0, 0));
        CK(cuMemMap(v, m.size, 0, m.mc, 0));
        set_access(v, m.size, d);
        m.uc.push_back(u);
        m.mcva.push_back(v);
    }
    return m;
}

static void dump(const std::string &path, const void *p, size_t n) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) {
        perror(path.c_str());
        exit(1);
    }
    fwrite(p, 1, n, f);
    fclose(f);
}

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)rng;
}

// Inputs that separate summation orders: per element, a random mix of wide-exponent values,
// near-cancelling pairs, subnormals, signed zeros, inf/nan (a small fraction) and plain random bits.
static void fill(uint8_t *buf, size_t n, int width_kind, int member) {
    for (size_t i = 0; i < n; i += 4) {
        uint32_t r = rnd(), w;
        uint32_t cls = rnd() % 64;
        if (cls < 40) {
            w = r;  // raw bits (any format)
        } else if (cls < 52) {
            // f32-ish narrow exponent band (lots of rounding interplay)
            w = (r & 0x807fffffu) | ((120u + rnd() % 16) << 23);
        } else if (cls < 58) {
            // 16-bit lanes with narrow exponents (bf16 / f16)
            uint32_t lo = (r & 0x807f) | ((120u + rnd() % 12) << 7);
            uint32_t hi = ((r >> 16) & 0x83ff) | ((12u + rnd() % 6) << 10);
            w = (rnd() & 1) ? (lo | (lo << 16)) : (hi | (lo << 16));
        } else if (cls < 61) {
            w = r & 0x80ffffffu;  // subnormal-ish
        } else if (cls < 62) {
            w = (r & 1) ? 0x80000000u : 0u;
        } else {
            w = (r & 0x80000000u) | 0x7f800000u | ((r & 3) == 0 ? (r & 0x7fffff) : 0);
        }
        (void)width_kind;
        (void)member;
        memcpy(buf + i, &w, 4);
    }
}

static std::vector<int> parse_order(const char *s, int ndev) {
    std::vector<int> v;
    if (!s) {
        for (int i = 0; i < ndev; i++) v.push_back(i);
        return v;
    }
    const char *p = s;
    while (*p) {
        v.push_back(atoi(p));
        while (*p && *p != ',') p++;
        if (*p) p++;
    }
    return v;
}

int main(int argc, char **argv) {
    CK(cuInit(0));
    int ndev = 0;
    CK(cuDeviceGetCount(&ndev));
    std::string mode = argc > 1 ? argv[1] : "info";
    if (mode == "info") {
        for (int d = 0; d < ndev; d++) {
            CUdevice dev;
            CK(cuDeviceGet(&dev, d));
            int mc = -1, fab = -1, vmm = -1;
            cuDeviceGetAttribute(&mc, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev);
            cuDeviceGetAttribute(&fab, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
            cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev);
            char name[128];
            cuDeviceGetName(name, sizeof name, dev);
            printf("device %d %s MULTICAST_SUPPORTED=%d FABRIC=%d VMM=%d\n", d, name, mc, fab, vmm);
        }
        if (ndev >= 1) {
            CUmulticastObjectProp p = {};
            p.numDevices = ndev;
            p.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
            p.size = 1;
            size_t g1 = 0, g2 = 0;
            CUcontext c;
            CK(cuDevicePrimaryCtxRetain(&c, 0));
            CK(cuCtxSetCurrent(c));
            CUresult r1 = cuMulticastGetGranularity(&g1, &p, CU_MULTICAST_GRANULARITY_MINIMUM);
            CUresult r2 = cuMulticastGetGranularity(&g2, &p, CU_MULTICAST_GRANULARITY_RECOMMENDED);
            printf("granularity min %zu (%d) recommended %zu (%d)\n", g1, r1, g2, r2);
            p.size = g2 ? g2 : (2 << 20);
            CUmemGenericAllocationHandle h;
            CUresult r3 = cuMulticastCreate(&h, &p);
            const char *s = 0;
            cuGetErrorString(r3, &s);
            printf("cuMulticastCreate(numDevices=%d) -> %d %s\n", ndev, r3, s ? s : "");
        }
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: mm info | ldr OUT [order] | red OUT [order] | litmus OUT [order] | litmus2 OUT [order]\n");
        return 2;
    }
    std::string out = argv[2];
    std::vector<int> order = parse_order(argc > 3 ? argv[3] : nullptr, ndev);
    int nm = order.size();
    const int N = 1 << 16;  // 16-byte vectors per test (1 MiB)
    const int REPS = 4;
    Mc m = make_mc(order, 4u << 20);
    {
        FILE *f = fopen((out + "/meta.txt").c_str(), "w");
        fprintf(f, "mode %s\nmembers", mode.c_str());
        for (int d : order) fprintf(f, " %d", d);
        fprintf(f, "\nsize %zu\nN %d\nREPS %d\n", m.size, N, REPS);
        fclose(f);
    }
    if (mode == "ldr") {
        std::vector<uint8_t> in((size_t)N * 16), res((size_t)N * 16);
        for (auto &t : LDR) {
            int n = N * 16 / t.width;
            for (int i = 0; i < nm; i++) {
                fill(in.data(), in.size(), 0, i);
                CK(cuCtxSetCurrent(m.ctx[i]));
                CK(cuMemcpyHtoD(m.uc[i], in.data(), in.size()));
                dump(out + "/" + t.name + ".in" + std::to_string(i) + ".bin", in.data(), in.size());
            }
            for (int i = 0; i < nm; i++) {
                CK(cuCtxSetCurrent(m.ctx[i]));
                CK(cuCtxSynchronize());
            }
            for (int i = 0; i < nm; i++) {
                CK(cuCtxSetCurrent(m.ctx[i]));
                void *o;
                RK(cudaSetDevice(order[i]));
                RK(cudaMalloc(&o, in.size()));
                for (int r = 0; r < REPS; r++) {
                    RK(cudaMemset(o, 0, in.size()));
                    void *mcp = (void *)m.mcva[i];
                    void *args[] = {&mcp, &o, &n};
                    RK(cudaLaunchKernel(t.fn, dim3((n + 255) / 256), dim3(256), args, 0, 0));
                    RK(cudaDeviceSynchronize());
                    RK(cudaMemcpy(res.data(), o, in.size(), cudaMemcpyDeviceToHost));
                    dump(out + "/" + t.name + ".out" + std::to_string(i) + "." + std::to_string(r) +
                             ".bin",
                         res.data(), res.size());
                }
                RK(cudaFree(o));
            }
            printf("ldr %s done\n", t.name);
        }
        return 0;
    }
    if (mode == "red") {
        // f32: N/4 words, each member starts from its own random value, every device adds
        // ROUNDS values at once. bf16x8: same with vectors. s32: counts.
        const int ROUNDS = 8;
        int nf = N;  // floats
        std::vector<float> init(nf), src((size_t)nf * ROUNDS);
        for (int rep = 0; rep < REPS; rep++) {
            for (int kind = 0; kind < 3; kind++) {
                const char *kn = kind == 0 ? "f32" : kind == 1 ? "bf16x8" : "s32";
                size_t bytes = kind == 1 ? (size_t)N * 16 : (size_t)nf * 4;
                std::vector<uint8_t> ib(bytes);
                for (int i = 0; i < nm; i++) {
                    if (kind == 2)
                        memset(ib.data(), 0, bytes);
                    else
                        fill(ib.data(), bytes, 0, i);
                    // keep magnitudes finite-ish for adds: clear exponent top bit sometimes
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    CK(cuMemcpyHtoD(m.uc[i], ib.data(), bytes));
                    dump(out + "/red_" + kn + ".init" + std::to_string(i) + "." + std::to_string(rep) +
                             ".bin",
                         ib.data(), bytes);
                }
                std::vector<void *> srcd(nm);
                for (int i = 0; i < nm; i++) {
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    RK(cudaSetDevice(order[i]));
                    size_t sb = bytes * ROUNDS;
                    std::vector<uint8_t> sh(sb);
                    fill(sh.data(), sb, 0, 100 + i);
                    RK(cudaMalloc(&srcd[i], sb));
                    RK(cudaMemcpy(srcd[i], sh.data(), sb, cudaMemcpyHostToDevice));
                    dump(out + "/red_" + kn + ".src" + std::to_string(i) + "." + std::to_string(rep) +
                             ".bin",
                         sh.data(), sb);
                    RK(cudaDeviceSynchronize());
                }
                for (int i = 0; i < nm; i++) {
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    RK(cudaSetDevice(order[i]));
                    void *mcp = (void *)m.mcva[i];
                    int rounds = ROUNDS, rank = i;
                    if (kind == 0) {
                        void *args[] = {&mcp, &srcd[i], &nf, &rounds};
                        RK(cudaLaunchKernel((void *)k_red_f32, dim3(nf / 256), dim3(256), args, 0, 0));
                    } else if (kind == 1) {
                        int nv = N;
                        void *args[] = {&mcp, &srcd[i], &nv, &rounds};
                        RK(cudaLaunchKernel((void *)k_red_bf16x8, dim3(nv / 256), dim3(256), args, 0, 0));
                    } else {
                        void *args[] = {&mcp, &nf, &rounds, &rank};
                        RK(cudaLaunchKernel((void *)k_red_s32, dim3(nf / 256), dim3(256), args, 0, 0));
                    }
                }
                for (int i = 0; i < nm; i++) {
                    RK(cudaSetDevice(order[i]));
                    RK(cudaDeviceSynchronize());
                    RK(cudaFree(srcd[i]));
                }
                for (int i = 0; i < nm; i++) {
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    CK(cuMemcpyDtoH(ib.data(), m.uc[i], bytes));
                    dump(out + "/red_" + kn + ".fin" + std::to_string(i) + "." + std::to_string(rep) +
                             ".bin",
                         ib.data(), bytes);
                }
            }
            printf("red rep %d done\n", rep);
        }
        // multimem.st: every member gets the bytes.
        std::vector<uint8_t> sb((size_t)N * 16), got((size_t)N * 16);
        fill(sb.data(), sb.size(), 0, 7);
        CK(cuCtxSetCurrent(m.ctx[0]));
        RK(cudaSetDevice(order[0]));
        void *s;
        RK(cudaMalloc(&s, sb.size()));
        RK(cudaMemcpy(s, sb.data(), sb.size(), cudaMemcpyHostToDevice));
        void *mcp = (void *)m.mcva[0];
        int nv = N;
        void *args[] = {&mcp, &s, &nv};
        RK(cudaLaunchKernel((void *)k_st_f32x4, dim3(nv / 256), dim3(256), args, 0, 0));
        RK(cudaDeviceSynchronize());
        int okall = 1;
        for (int i = 0; i < nm; i++) {
            CK(cuCtxSetCurrent(m.ctx[i]));
            CK(cuMemcpyDtoH(got.data(), m.uc[i], got.size()));
            int ok = memcmp(got.data(), sb.data(), sb.size()) == 0;
            okall &= ok;
            printf("st member %d (device %d): %s\n", i, order[i], ok ? "equal" : "DIFFERENT");
        }
        FILE *f = fopen((out + "/st.txt").c_str(), "w");
        fprintf(f, "multimem.st.relaxed.sys.v4.f32 from member 0: all members equal = %d\n", okall);
        fclose(f);
        return 0;
    }
    if (mode == "litmus2") {
        // data: 128 blocks x 4 KiB at offset 0; flags: 128 x 4 B at offset 2 MiB.
        const int BLOCKS = 128;
        const int RUNS = 200;
        FILE *f = fopen((out + "/litmus2.txt").c_str(), "w");
        for (int var = 0; var < 5; var++) {
            unsigned long tot_bad = 0, tot_to = 0, runs_bad = 0;
            for (int run = 0; run < RUNS; run++) {
                uint32_t tag = 0x3c000000u ^ ((uint32_t)run << 8) ^ (uint32_t)var;
                for (int i = 0; i < nm; i++) {
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    CK(cuMemsetD8(m.uc[i], 0, m.size));
                    CK(cuCtxSynchronize());
                }
                std::vector<void *> badd(nm);
                for (int i = 0; i < nm; i++) {
                    RK(cudaSetDevice(order[i]));
                    RK(cudaMalloc(&badd[i], 8));
                    RK(cudaMemset(badd[i], 0, 8));
                    RK(cudaDeviceSynchronize());
                }
                for (int i = 0; i < nm; i++) {
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    RK(cudaSetDevice(order[i]));
                    void *d = (void *)m.uc[i];
                    void *dm = (void *)m.mcva[i];
                    void *flm = (void *)(m.mcva[i] + (2u << 20));
                    void *flu = (void *)(m.uc[i] + (2u << 20));
                    int rank = i, n = nm, v = var;
                    void *b = badd[i];
                    void *args[] = {&d, &dm, &flm, &flu, &rank, &n, &tag, &v, &b};
                    RK(cudaLaunchKernel((void *)k_mp2, dim3(BLOCKS), dim3(256), args, 0, 0));
                }
                unsigned long rb = 0;
                for (int i = 0; i < nm; i++) {
                    RK(cudaSetDevice(order[i]));
                    RK(cudaDeviceSynchronize());
                    unsigned h[2];
                    RK(cudaMemcpy(h, badd[i], 8, cudaMemcpyDeviceToHost));
                    rb += h[0];
                    tot_to += h[1];
                    RK(cudaFree(badd[i]));
                }
                tot_bad += rb;
                runs_bad += rb > 0;
            }
            fprintf(f, "variant %d: runs %d lanes/run %d readers %d: stale lanes %lu in %lu runs, timeouts %lu\n",
                    var, RUNS, BLOCKS * 256 * 4, nm, tot_bad, runs_bad, tot_to);
            printf("variant %d: stale %lu (runs %lu) timeouts %lu\n", var, tot_bad, runs_bad, tot_to);
            fflush(f);
        }
        fclose(f);
        return 0;
    }
    if (mode == "litmus") {
        // data: SLOTS x 16 B at offset 0; flags: SLOTS x 4 B at offset 2 MiB.
        const int SLOTS = 1 << 15;
        const int RUNS = 200;
        FILE *f = fopen((out + "/litmus.txt").c_str(), "w");
        for (int var = 0; var < 4; var++) {
            unsigned long tot_bad = 0, tot_to = 0, runs_bad = 0;
            for (int run = 0; run < RUNS; run++) {
                int w = run % nm;  // writer member rotates
                uint32_t tag = 0x5a000000u ^ ((uint32_t)run << 8) ^ (uint32_t)var;
                for (int i = 0; i < nm; i++) {
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    CK(cuMemsetD8(m.uc[i], 0, m.size));
                    CK(cuCtxSynchronize());
                }
                std::vector<void *> badd(nm);
                for (int i = 0; i < nm; i++) {
                    if (i == w) continue;
                    RK(cudaSetDevice(order[i]));
                    RK(cudaMalloc(&badd[i], 16));
                    unsigned init[4] = {0, 0, 0xffffffffu, 0};
                    RK(cudaMemcpy(badd[i], init, 16, cudaMemcpyHostToDevice));
                }
                // readers first (they poll), then the writer
                for (int i = 0; i < nm; i++) {
                    if (i == w) continue;
                    CK(cuCtxSetCurrent(m.ctx[i]));
                    RK(cudaSetDevice(order[i]));
                    void *d = (void *)m.uc[i];
                    void *fl = (void *)(m.uc[i] + (2u << 20));
                    int slots = SLOTS, v = var, zero = 0;
                    void *b = badd[i];
                    void *fb = (char *)badd[i] + 8;
                    void *args[] = {&d, &fl, &slots, &tag, &v, &b, &fb, &zero};
                    RK(cudaLaunchKernel((void *)k_mp_reader, dim3(SLOTS / 128), dim3(128), args, 0, 0));
                }
                {
                    CK(cuCtxSetCurrent(m.ctx[w]));
                    RK(cudaSetDevice(order[w]));
                    void *d = (void *)m.mcva[w];
                    void *fl = (void *)(m.mcva[w] + (2u << 20));
                    int slots = SLOTS, v = var;
                    void *args[] = {&d, &fl, &slots, &tag, &v};
                    RK(cudaLaunchKernel((void *)k_mp_writer, dim3(SLOTS / 128), dim3(128), args, 0, 0));
                }
                unsigned long rb = 0;
                for (int i = 0; i < nm; i++) {
                    RK(cudaSetDevice(order[i]));
                    RK(cudaDeviceSynchronize());
                    if (i == w) continue;
                    unsigned h[4];
                    RK(cudaMemcpy(h, badd[i], 16, cudaMemcpyDeviceToHost));
                    rb += h[0];
                    tot_to += h[1];
                    RK(cudaFree(badd[i]));
                }
                tot_bad += rb;
                runs_bad += rb > 0;
            }
            fprintf(f, "variant %d: runs %d slots/run %d readers %d: stale reads %lu in %lu runs, timeouts %lu\n",
                    var, RUNS, SLOTS, nm - 1, tot_bad, runs_bad, tot_to);
            printf("variant %d: stale %lu (runs %lu) timeouts %lu\n", var, tot_bad, runs_bad, tot_to);
            fflush(f);
        }
        fclose(f);
        return 0;
    }
    if (mode == "sr") {
        // Rounding of 16-bit ld_reduce results (bf16x8a / bf16x8 / f16x8a): which of the two
        // neighbours of the sum is returned, as a function of address and data.
        //   pattern 0: every vector identical: lane l of member 0 = 1.0, member 1 = l/8 ulp
        //              (l = 1..7; lane 0: 0), members 2.. = 0
        //   pattern 1: as 0 but member 0's value varies with the vector index (1 + (i % 64) ulp)
        //   pattern 2: as 0 but member 1's addend is (i % 8 + 1)/16 ulp + 2^-20 jitter in lane l
        //   pattern 3: pattern 0 shifted: same data, read through a second multicast object
        // Each pattern is run twice (rep 0 / 1) from member 0 and once from member 1.
        Mc m2 = make_mc(order, 4u << 20);
        const char *tests[] = {"bf16x8a", "bf16x8", "f16x8a"};
        for (const char *tn : tests) {
            LdrTest *t = nullptr;
            for (auto &x : LDR)
                if (!strcmp(x.name, tn)) t = &x;
            bool half16 = tn[0] == 'f';
            uint16_t one = half16 ? 0x3c00 : 0x3f80;
            int mbits = half16 ? 10 : 7;
            int bias = half16 ? 15 : 127;
            int ebits = half16 ? 5 : 8;
            auto enc = [&](double v) -> uint16_t {
                // positive normal values only
                int e = 0;
                double f = frexp(v, &e);  // v = f * 2^e, f in [0.5, 1)
                int ex = e - 1 + bias;
                double man = (f * 2 - 1) * (1 << mbits);
                uint16_t mi = (uint16_t)man;
                (void)ebits;
                return (uint16_t)((ex << mbits) | mi);
            };
            double ulp1 = ldexp(1.0, -mbits);
            for (int pat = 0; pat < 4; pat++) {
                std::vector<uint16_t> buf((size_t)N * 8);
                for (int mi = 0; mi < nm; mi++) {
                    for (int i = 0; i < N; i++)
                        for (int l = 0; l < 8; l++) {
                            uint16_t v = 0;
                            if (mi == 0) {
                                v = one;
                                if (pat == 1) v = (uint16_t)(one + (i % 64));
                            } else if (mi == 1) {
                                double a = (pat == 2) ? ((i % 8) + 1) / 16.0 * ulp1 + ldexp(1.0, -20) * l
                                                      : l / 8.0 * ulp1;
                                v = a > 0 ? enc(a) : 0;
                            }
                            buf[(size_t)i * 8 + l] = v;
                        }
                    Mc &mm = pat == 3 ? m2 : m;
                    CK(cuCtxSetCurrent(mm.ctx[mi]));
                    CK(cuMemcpyHtoD(mm.uc[mi], buf.data(), buf.size() * 2));
                    CK(cuCtxSynchronize());
                    dump(out + "/sr_" + tn + ".p" + std::to_string(pat) + ".in" + std::to_string(mi) + ".bin",
                         buf.data(), buf.size() * 2);
                }
                for (int run = 0; run < 3; run++) {
                    int mi = run < 2 ? 0 : 1;
                    Mc &mm = pat == 3 ? m2 : m;
                    CK(cuCtxSetCurrent(mm.ctx[mi]));
                    RK(cudaSetDevice(order[mi]));
                    void *o;
                    RK(cudaMalloc(&o, (size_t)N * 16));
                    void *mcp = (void *)mm.mcva[mi];
                    int n = N;
                    void *args[] = {&mcp, &o, &n};
                    RK(cudaLaunchKernel(t->fn, dim3((n + 255) / 256), dim3(256), args, 0, 0));
                    RK(cudaDeviceSynchronize());
                    std::vector<uint8_t> res((size_t)N * 16);
                    RK(cudaMemcpy(res.data(), o, res.size(), cudaMemcpyDeviceToHost));
                    RK(cudaFree(o));
                    dump(out + "/sr_" + tn + ".p" + std::to_string(pat) + ".r" + std::to_string(run) + ".bin",
                         res.data(), res.size());
                }
            }
            printf("sr %s done\n", tn);
        }
        return 0;
    }
    if (mode == "sr2") {
        // Which bits decide the rounding direction of a 16-bit ld_reduce result. Lane 0 of every
        // vector is the test lane: member 0 = 1.0, member 1 = 1/2 ulp (frac 0.5), members 2/3
        // as below; lanes 1..7 as below.
        //   q0: member 3 lane 0 = random tiny value (|v| < 2^-40), others 0; lanes 1..7 zero
        //   q1: lane 0 members 2/3 = 0; lanes 1..7 random in every member
        //   q2: member 0 lane 0 = 1.0 + random mantissa; lanes 1..7 zero
        //   q3: member 2 lane 0 = +t, member 3 lane 0 = -t (t random, 2^-4..2^-1); lanes 1..7 zero
        //   q4: as q0 but member 3 lane 0 = random value in [2^-20, 2^-12)
        const char *tests[] = {"bf16x8a", "bf16x8", "f16x8a"};
        for (const char *tn : tests) {
            LdrTest *t = nullptr;
            for (auto &x : LDR)
                if (!strcmp(x.name, tn)) t = &x;
            bool h = tn[0] == 'f';
            int mb = h ? 10 : 7, bias = h ? 15 : 127;
            auto mk = [&](int s, int e, uint32_t man) -> uint16_t {  // value 2^e * (1 + man/2^mb)
                return (uint16_t)((s << 15) | ((e + bias) << mb) | (man & ((1u << mb) - 1)));
            };
            uint16_t one = mk(0, 0, 0), half = mk(0, -mb - 1, 0);
            for (int q = 0; q < 5; q++) {
                std::vector<std::vector<uint16_t>> b(nm, std::vector<uint16_t>((size_t)N * 8, 0));
                for (int i = 0; i < N; i++) {
                    size_t o = (size_t)i * 8;
                    b[0][o] = one;
                    b[1][o] = half;
                    if (q == 0 && nm > 3) b[3][o] = mk(rnd() & 1, -40 - (int)(rnd() % (h ? 0 + 1 : 40)), rnd());
                    if (q == 0 && h && nm > 3) b[3][o] = (uint16_t)((rnd() & 0x8000) | (rnd() & 0x3ff));  // f16 subnormal
                    if (q == 1)
                        for (int mi = 0; mi < nm; mi++)
                            for (int l = 1; l < 8; l++) b[mi][o + l] = (uint16_t)rnd();
                    if (q == 2) b[0][o] = mk(0, 0, rnd());
                    if (q == 3 && nm > 3) {
                        uint16_t tv = mk(0, -1 - (int)(rnd() % 4), rnd());
                        b[2][o] = tv;
                        b[3][o] = tv | 0x8000;
                    }
                    if (q == 4 && nm > 3) b[3][o] = mk(rnd() & 1, -20 + (int)(rnd() % 8), rnd());
                }
                for (int mi = 0; mi < nm; mi++) {
                    CK(cuCtxSetCurrent(m.ctx[mi]));
                    CK(cuMemcpyHtoD(m.uc[mi], b[mi].data(), b[mi].size() * 2));
                    CK(cuCtxSynchronize());
                    dump(out + "/sr2_" + tn + ".q" + std::to_string(q) + ".in" + std::to_string(mi) + ".bin",
                         b[mi].data(), b[mi].size() * 2);
                }
                for (int run = 0; run < 2; run++) {
                    int mi = run;
                    CK(cuCtxSetCurrent(m.ctx[mi]));
                    RK(cudaSetDevice(order[mi]));
                    void *o;
                    RK(cudaMalloc(&o, (size_t)N * 16));
                    void *mcp = (void *)m.mcva[mi];
                    int n = N;
                    void *args[] = {&mcp, &o, &n};
                    RK(cudaLaunchKernel(t->fn, dim3((n + 255) / 256), dim3(256), args, 0, 0));
                    RK(cudaDeviceSynchronize());
                    std::vector<uint8_t> res((size_t)N * 16);
                    RK(cudaMemcpy(res.data(), o, res.size(), cudaMemcpyDeviceToHost));
                    RK(cudaFree(o));
                    dump(out + "/sr2_" + tn + ".q" + std::to_string(q) + ".r" + std::to_string(run) + ".bin",
                         res.data(), res.size());
                }
            }
            printf("sr2 %s done\n", tn);
        }
        return 0;
    }
    if (mode == "sr3") {
        // Rounding direction of one lane vs the data of another lane. Vector i: k = i % 128,
        // f = (i / 128) % 8, section s = i / 16384. The test lane holds member 0 = 1.0 and
        // member 1 = f/8 ulp (members 2.. 0); the sweep lane(s) hold member 0 = 1 + k/128
        // (k in the top 7 mantissa bits), others 0. Other lanes 0.
        //   s0: test lane 0, sweep lane 1      s1: test lane 1, sweep lane 0
        //   s2: test lane 1, sweep lane 2      s3: test lane 0, sweep lanes 2..7
        const char *tests[] = {"bf16x8a", "bf16x8", "f16x8a"};
        for (const char *tn : tests) {
            LdrTest *t = nullptr;
            for (auto &x : LDR)
                if (!strcmp(x.name, tn)) t = &x;
            bool h = tn[0] == 'f';
            int mb = h ? 10 : 7, bias = h ? 15 : 127;
            auto mk = [&](int e, uint32_t man) -> uint16_t {
                return (uint16_t)(((e + bias) << mb) | (man & ((1u << mb) - 1)));
            };
            std::vector<std::vector<uint16_t>> b(nm, std::vector<uint16_t>((size_t)N * 8, 0));
            for (int i = 0; i < N; i++) {
                int k = i % 128, f = (i / 128) % 8, sec = (i / 16384) % 4;
                size_t o = (size_t)i * 8;
                int tl = (sec == 1 || sec == 2) ? 1 : 0;
                b[0][o + tl] = mk(0, 0);
                // f/8 ulp = 2^(-mb-3) * f
                if (f) {
                    int e = -mb - 3, mm = f;
                    while (mm >= 2) { mm >>= 1; e++; }
                    uint32_t man = (uint32_t)(((f << mb) >> (e + mb + 3)) - (1u << mb)) & ((1u << mb) - 1);
                    b[1][o + tl] = mk(e, man);
                }
                uint16_t sw = mk(0, (uint32_t)k << (mb - 7));
                if (sec == 0) b[0][o + 1] = sw;
                if (sec == 1) b[0][o + 0] = sw;
                if (sec == 2) b[0][o + 2] = sw;
                if (sec == 3)
                    for (int l = 2; l < 8; l++) b[0][o + l] = sw;
            }
            for (int mi = 0; mi < nm; mi++) {
                CK(cuCtxSetCurrent(m.ctx[mi]));
                CK(cuMemcpyHtoD(m.uc[mi], b[mi].data(), b[mi].size() * 2));
                CK(cuCtxSynchronize());
                dump(out + "/sr3_" + tn + ".in" + std::to_string(mi) + ".bin", b[mi].data(), b[mi].size() * 2);
            }
            CK(cuCtxSetCurrent(m.ctx[0]));
            RK(cudaSetDevice(order[0]));
            void *o;
            RK(cudaMalloc(&o, (size_t)N * 16));
            void *mcp = (void *)m.mcva[0];
            int n = N;
            void *args[] = {&mcp, &o, &n};
            RK(cudaLaunchKernel(t->fn, dim3((n + 255) / 256), dim3(256), args, 0, 0));
            RK(cudaDeviceSynchronize());
            std::vector<uint8_t> res((size_t)N * 16);
            RK(cudaMemcpy(res.data(), o, res.size(), cudaMemcpyDeviceToHost));
            RK(cudaFree(o));
            dump(out + "/sr3_" + tn + ".r0.bin", res.data(), res.size());
            printf("sr3 %s done\n", tn);
        }
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
}
