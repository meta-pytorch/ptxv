// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Cross-process multicast vs unicast on GB300 (S-VA-ALIAS / E-MC-ALIAS-HW, option d of forum
// post 55). Variant of xp.cu: rank 0 creates an NVLS multicast object (fabric handle), both ranks
// add their GPU, bind a local allocation and map the multicast and their unicast range. The
// writer (rank 1) stores the payload with multimem.st and the flag with multimem.st.relaxed /
// multimem.st.release / multimem.red.release; the reader (rank 0) polls and reads its own
// unicast copy. The reader acks with multimem.st.release (the writer polls its unicast copy).
// XP_SINGLE=1: one process drives both GPUs (the in-process shape of E-MC-ALIAS-HW).
//
//   nvcc -O2 -arch=sm_103a -std=c++17 -o xpmc xpmc.cu -lcuda
//   torchrun --nproc-per-node 2 --no-python ./xpmc OUT.jsonl [iters] [slots]
//
// Two processes on one node (torchrun --no-python, ranks 0 and 1, GPU = LOCAL_RANK). Rank 0
// allocates the message buffer on its GPU with cuMemCreate and exports a fabric handle; rank 1
// imports and maps it. So writer and reader reach the bytes through two mappings in two
// address spaces. Message passing per lane, like probes/poll_acq: the writer stores a 16-byte
// payload (weak st.global) in its own 128-byte line, optionally fence.proxy.alias, then the
// lane's flag; the reader optionally warms the payload line in L1, polls its flag, optionally
// fence.proxy.alias, reads the payload (weak ld.global) and classifies it: ok (== k), stale
// (< k), other. It acks through the buffer so the writer never runs ahead.
//
// Matrix per run: placement (peer writes into the owner's memory / owner writes, peer reads
// remote), flag store (volatile, relaxed, release) x flag load (volatile, relaxed, acquire),
// scope (gpu, sys), fence.proxy.alias (none, writer, reader, both), stress (none; co-reader
// warps refilling the payload lines while the poll runs plus an NVLink copy stream).
// Control: XP_SINGLE=1 runs one process driving both GPUs through ONE mapping of the buffer
// (no alias): the same matrix, the reference for "one location".
//
//   nvcc -O2 -arch=sm_103a -std=c++17 -o xp xp.cu -lcuda
//   torchrun --nproc-per-node 2 --no-python ./xp OUT.jsonl [iters] [slots]
//   XP_SINGLE=1 CUDA_VISIBLE_DEVICES=0,1 ./xp OUT.jsonl [iters] [slots]
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
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
#define RK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t r_ = (x);                                                                    \
        if (r_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(r_)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

typedef uint32_t u32;
typedef unsigned long long u64;

// Layout (u32 words) of the shared buffer, per slot s and lane l:
//   payload: PAY + (s * 32 + l) * 32 (one 128-byte line per lane)
//   flag:    FLAG + (s * 32 + l) * 32 (own line)
//   ack:     ACK + (s * 32 + l) * 32 (own line)
#define MAXSLOTS 128
#define LINEW 32
#define PAY 0
#define FLAG (MAXSLOTS * 32 * LINEW)
#define ACK (2 * MAXSLOTS * 32 * LINEW)
#define WORDS (3 * MAXSLOTS * 32 * LINEW)
#define TIMEOUT 200000000ull

struct Cfg {
    int ws, rs;     // flag store: 0 volatile 1 relaxed 2 release; flag load: 0 volatile 1 relaxed 2 acquire
    int sys;        // scope: 0 gpu 1 sys
    int wf, rf;     // fence.proxy.alias on writer / reader
    int stress;     // co-reader warps + NVLink copy
    int dmask;      // writer delay mask (clock cycles)
    int priv;       // control: plain st/ld on a private buffer (not bound to the multicast object)
};

__device__ __forceinline__ u32 smid() {
    u32 r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void delay(u32 &seed, u32 mask) {
    if (!mask) return;
    seed = seed * 1664525u + 1013904223u;
    u64 t0 = clock64(), d = (seed >> 8) & mask;
    while (clock64() - t0 < d) {
    }
}
__device__ __forceinline__ void st_flag(u32 *p, u32 v, int sem, int sys) {
    if (sem == 0) asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    else if (sem == 1 && sys) asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    else if (sem == 1) asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    else if (sys) asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
    else asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ u32 ld_flag(const u32 *p, int sem, int sys) {
    u32 v;
    if (sem == 0) asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else if (sem == 1 && sys) asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else if (sem == 1) asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else if (sys) asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    else asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ u32 ld_weak(const u32 *p) {
    u32 v;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ u32 ld_ack(const u32 *p) {
    u32 v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void alias_fence() { asm volatile("fence.proxy.alias;" ::: "memory"); }


__device__ __forceinline__ void mc_flag(u32 *p, u32 k, int sem, int sys) {
    if (sem == 0 && sys) asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(k) : "memory");
    else if (sem == 0) asm volatile("multimem.st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(k) : "memory");
    else if (sem == 1 && sys) asm volatile("multimem.st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(k) : "memory");
    else if (sem == 1) asm volatile("multimem.st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(k) : "memory");
    else if (sys) asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
    else asm volatile("multimem.red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

__global__ void k_writer(u32 *MC, u32 *U, int iters, Cfg c, u32 *abort_, u32 seed0) {
    const u32 s = blockIdx.x, l = threadIdx.x & 31, idx = s * 32 + l;
    if (threadIdx.x >= 32) return;
    u32 *pay = MC + PAY + idx * LINEW, *flag = MC + FLAG + idx * LINEW, *ack = U + ACK + idx * LINEW;
    u32 seed = seed0 ^ (idx * 2654435761u);
    for (u32 k = 1; k <= (u32)iters; k++) {
        u64 t0 = clock64();
        while (ld_ack(ack) < k - 1) {
            if (clock64() - t0 > TIMEOUT || *(volatile u32 *)abort_) return;
        }
        alias_fence();
        delay(seed, c.dmask);
        if (c.priv) asm volatile("st.weak.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(pay), "r"(k) : "memory");
        else asm volatile("multimem.st.weak.global.v4.f32 [%0], {%1,%1,%1,%1};" ::"l"(pay), "r"(k) : "memory");
        if (c.wf) alias_fence();
        if (c.priv) st_flag(flag, k, c.ws == 0 ? 1 : 2, c.sys);
        else mc_flag(flag, k, c.ws, c.sys);
    }
}

__global__ void k_reader(u32 *MC, u32 *U, int iters, Cfg c, u64 *cnt, u32 *abort_, u32 *done) {
    const u32 s = blockIdx.x, w = threadIdx.x >> 5, l = threadIdx.x & 31, idx = s * 32 + l;
    u32 *pay = U + PAY + idx * LINEW, *flag = U + FLAG + idx * LINEW, *ack = MC + ACK + idx * LINEW;
    if (w > 0) {
        u32 sink = 0;
        while (!*(volatile u32 *)(done + s)) {
            sink += ld_weak(pay);
            if (*(volatile u32 *)abort_) break;
        }
        if (sink == 0xffffffffu) atomicAdd(cnt + 4, 1ull);
        return;
    }
    u64 ok = 0, stale = 0, other = 0, tmo = 0, sink = 0;
    for (u32 k = 1; k <= (u32)iters; k++) {
        sink += ld_weak(pay);
        u64 t0 = clock64();
        bool bad = false;
        while (ld_flag(flag, c.rs, c.sys) < k) {
            if (clock64() - t0 > TIMEOUT) {
                bad = true;
                break;
            }
        }
        if (bad) {
            tmo++;
            atomicExch(abort_, 1u);
            break;
        }
        if (c.rf) alias_fence();
        u32 v[4];
        asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3])
                     : "l"(pay)
                     : "memory");
        bool good = v[0] == k && v[1] == k && v[2] == k && v[3] == k;
        bool st = v[0] < k || v[1] < k || v[2] < k || v[3] < k;
        ok += good;
        stale += !good && st;
        other += !good && !st;
        alias_fence();
        if (c.priv) asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ack), "r"(k) : "memory");
        else asm volatile("multimem.st.release.sys.global.u32 [%0], %1;" ::"l"(ack), "r"(k) : "memory");
    }
    if (sink == 0xffffffffffffull) atomicAdd(cnt + 4, 1ull);
    atomicAdd(cnt + 0, ok);
    atomicAdd(cnt + 1, stale);
    atomicAdd(cnt + 2, other);
    atomicAdd(cnt + 3, tmo);
    __syncwarp();
    if (l == 0) *(volatile u32 *)(done + s) = 1;
}

__global__ void k_traffic(const uint4 *src, uint4 *dst, size_t n, u32 *stop, int mode) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
    uint4 acc = {0, 0, 0, 0};
    while (!*(volatile u32 *)stop) {
        for (size_t j = i; j < n; j += st) {
            if (mode == 1) {  // read only
                uint4 v = src[j];
                acc.x ^= v.x;
            } else {
                dst[j] = src[j];
            }
        }
        if (acc.x == 0x12345678u) dst[i] = acc;
    }
}

// ---------------------------------------------------------------- host
static std::string dir;
static int rank_ = 0, world = 1;
static void barrier(const char *tag, int n) {
    char p[512];
    snprintf(p, sizeof p, "%s/b_%s_%d_%d", dir.c_str(), tag, n, rank_);
    FILE *f = fopen(p, "w");
    if (f) fclose(f);
    for (int r = 0; r < world; r++) {
        snprintf(p, sizeof p, "%s/b_%s_%d_%d", dir.c_str(), tag, n, r);
        struct stat sb;
        long tries = 0;
        while (stat(p, &sb) != 0) {
            usleep(200);
            if (++tries > 50000000) {
                fprintf(stderr, "barrier %s timeout\n", tag);
                exit(1);
            }
        }
    }
}
static void wait_file(const char *p) {
    struct stat sb;
    while (stat(p, &sb) != 0) usleep(1000);
}

int main(int argc, char **argv) {
    // Lazy module loading deadlocks the stress rows: the first launch of k_writer happens while the
    // spinning k_traffic kernel occupies the device, and the lazy load of k_writer waits for it
    // (k_traffic only stops after k_writer finishes). Load all kernels up front.
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    const char *out = argc > 1 ? argv[1] : "xpmc.jsonl";
    int iters = argc > 2 ? atoi(argv[2]) : 2000;
    int slots = argc > 3 ? atoi(argv[3]) : 64;
    if (slots > MAXSLOTS) slots = MAXSLOTS;
    bool single = getenv("XP_SINGLE") && atoi(getenv("XP_SINGLE"));
    const char *only = getenv("XP_ONLY");
    if (!single) {
        rank_ = atoi(getenv("RANK") ? getenv("RANK") : "0");
        world = atoi(getenv("WORLD_SIZE") ? getenv("WORLD_SIZE") : "1");
        if (world != 2) {
            fprintf(stderr, "needs 2 ranks (or XP_SINGLE=1)\n");
            return 1;
        }
    }
    dir = getenv("XP_DIR") ? getenv("XP_DIR") : "/dev/shm/xpmc";
    mkdir(dir.c_str(), 0700);
    CK(cuInit(0));
    std::vector<int> devs = single ? std::vector<int>{0, 1}
                                   : std::vector<int>{atoi(getenv("LOCAL_RANK") ? getenv("LOCAL_RANK") : "0")};
    std::vector<CUcontext> ctx;
    for (int d : devs) {
        CUcontext c;
        CK(cuDevicePrimaryCtxRetain(&c, d));
        ctx.push_back(c);
    }
    CK(cuCtxSetCurrent(ctx[0]));
    size_t bytes = (size_t)WORDS * 4, tbytes = 64ull << 20;
    CUmulticastObjectProp p = {};
    p.numDevices = 2;
    p.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    p.size = bytes + tbytes;
    size_t gran = 0;
    CK(cuMulticastGetGranularity(&gran, &p, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    size_t size = (p.size + gran - 1) / gran * gran;
    p.size = size;
    CUmemGenericAllocationHandle mc;
    char hp[512];
    snprintf(hp, sizeof hp, "%s/mchandle", dir.c_str());
    if (single || rank_ == 0) {
        CK(cuMulticastCreate(&mc, &p));
        if (!single) {
            CUmemFabricHandle fh;
            CK(cuMemExportToShareableHandle(&fh, mc, CU_MEM_HANDLE_TYPE_FABRIC, 0));
            FILE *f = fopen((std::string(hp) + ".tmp").c_str(), "wb");
            fwrite(&fh, sizeof fh, 1, f);
            fclose(f);
            rename((std::string(hp) + ".tmp").c_str(), hp);
        }
    } else {
        wait_file(hp);
        CUmemFabricHandle fh;
        FILE *f = fopen(hp, "rb");
        if (fread(&fh, sizeof fh, 1, f) != 1) return 1;
        fclose(f);
        CK(cuMemImportFromShareableHandle(&mc, &fh, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    for (int d : devs) CK(cuMulticastAddDevice(mc, d));
    barrier("added", 0);
    std::vector<u32 *> U, MC;
    for (size_t i = 0; i < devs.size(); i++) {
        CK(cuCtxSetCurrent(ctx[i]));
        CUmemAllocationProp ap = {};
        ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ap.location.id = devs[i];
        ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        CUmemGenericAllocationHandle h;
        CK(cuMemCreate(&h, size, &ap, 0));
        CK(cuMulticastBindMem(mc, 0, h, 0, size, 0));
        CUdeviceptr u, v;
        CUmemAccessDesc ad = {};
        ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ad.location.id = devs[i];
        ad.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CK(cuMemAddressReserve(&u, size, 0, 0, 0));
        CK(cuMemMap(u, size, 0, h, 0));
        CK(cuMemSetAccess(u, size, &ad, 1));
        CK(cuMemAddressReserve(&v, size, 0, 0, 0));
        CK(cuMemMap(v, size, 0, mc, 0));
        CK(cuMemSetAccess(v, size, &ad, 1));
        U.push_back((u32 *)u);
        MC.push_back((u32 *)v);
    }
    // Private control (XP_PRIV=1, one process): one allocation on GPU 0, not bound to the multicast
    // object, one mapping used by both GPUs (peer access). No virtual alias anywhere.
    u32 *P = nullptr;
    bool priv = getenv("XP_PRIV") && atoi(getenv("XP_PRIV"));
    if (priv) {
        if (!single) {
            fprintf(stderr, "XP_PRIV needs XP_SINGLE=1\n");
            return 1;
        }
        CK(cuCtxSetCurrent(ctx[0]));
        CUmemAllocationProp ap = {};
        ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        ap.location.id = devs[0];
        CUmemGenericAllocationHandle h;
        CK(cuMemCreate(&h, size, &ap, 0));
        CUdeviceptr u;
        CK(cuMemAddressReserve(&u, size, 0, 0, 0));
        CK(cuMemMap(u, size, 0, h, 0));
        CUmemAccessDesc ad[2] = {};
        for (int i = 0; i < 2; i++) {
            ad[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            ad[i].location.id = devs[i];
            ad[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        CK(cuMemSetAccess(u, size, ad, 2));
        P = (u32 *)u;
    }
    barrier("bound", 0);
    u64 *cnt[2];
    u32 *abort_[2], *done[2], *stop[2];
    uint4 *tdst[2];
    cudaStream_t sm[2], st[2];
    for (size_t i = 0; i < devs.size(); i++) {
        RK(cudaSetDevice(devs[i]));
        RK(cudaMalloc(&cnt[i], 8 * sizeof(u64)));
        RK(cudaMalloc(&abort_[i], 4));
        RK(cudaMalloc(&done[i], MAXSLOTS * 4));
        RK(cudaMalloc(&stop[i], 4));
        RK(cudaMalloc(&tdst[i], tbytes));
        RK(cudaStreamCreateWithFlags(&sm[i], cudaStreamNonBlocking));
        RK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
    }
    FILE *fo = (single || rank_ == 0) ? fopen(out, "a") : nullptr;
    const char *wsn[] = {"mm.st.relaxed", "mm.st.release", "mm.red.release"}, *rsn[] = {"volatile", "relaxed", "acquire"};
    int run = 0;
    // Writer: rank 1 / GPU 1; reader: rank 0 / GPU 0 (local index ri, wi in single mode).
    for (int stress = 0; stress < 2; stress++)
        for (int sys = 1; sys >= 0; sys--)
            for (int fen = 0; fen < 4; fen++)
                for (int ws = 0; ws < 3; ws++)
                    for (int rs = 0; rs < 3; rs++) {
                        Cfg c{ws, rs, sys, fen & 1, (fen >> 1) & 1, stress, 0, priv};
                        char name[256];
                        snprintf(name, sizeof name, "%s %s/%s.%s wf%d rf%d stress%d", priv ? "private" : "multicast", wsn[ws], rsn[rs],
                                 sys ? "sys" : "gpu", c.wf, c.rf, stress);
                        if (only && !strstr(name, only)) continue;
                        for (int dm = 0; dm < 2; dm++) {
                            c.dmask = dm ? 0x3ff : 0;
                            run++;
                            int ri = 0, wi = single ? 1 : 0;
                            bool iread = single || rank_ == 0, iwrite = single || rank_ == 1;
                            for (size_t i = 0; i < devs.size(); i++) {
                                RK(cudaSetDevice(devs[i]));
                                if (!priv) RK(cudaMemset(U[i], 0, bytes));
                                if (priv && i == 0) RK(cudaMemset(P, 0, bytes));
                                RK(cudaMemset(cnt[i], 0, 8 * sizeof(u64)));
                                RK(cudaMemset(abort_[i], 0, 4));
                                RK(cudaMemset(done[i], 0, MAXSLOTS * 4));
                                RK(cudaMemset(stop[i], 0, 4));
                                RK(cudaDeviceSynchronize());
                            }
                            if (getenv("XP_HB")) { fprintf(stderr, "[r%d] run %d start\n", rank_, run); }
                            barrier("start", run);
                            if (iread) {
                                RK(cudaSetDevice(devs[ri]));
                                k_reader<<<slots, (stress && !getenv("XP_NOCOREAD")) ? 64 : 32, 0, sm[ri]>>>(priv ? P : MC[ri], priv ? P : U[ri], iters, c, cnt[ri], abort_[ri], done[ri]);
                                RK(cudaGetLastError());
                            }
                            if (iwrite) {
                                RK(cudaSetDevice(devs[wi]));
                                if (stress && !getenv("XP_NOTRAFFIC"))
                                    k_traffic<<<getenv("XP_TBLOCKS") ? atoi(getenv("XP_TBLOCKS")) : 16, 256, 0, st[wi]>>>(
                                        getenv("XP_TSRC") ? (const uint4 *)tdst[wi] : (const uint4 *)(U[wi] + WORDS), tdst[wi],
                                        tbytes / 16, stop[wi], getenv("XP_TMODE") ? atoi(getenv("XP_TMODE")) : 0);  // plain loads: the unicast range
                                k_writer<<<slots, 32, 0, sm[wi]>>>(priv ? P : MC[wi], priv ? P : U[wi], iters, c, abort_[wi], 777u + run);
                                RK(cudaGetLastError());
                            }
                            if (getenv("XP_HB")) { fprintf(stderr, "[r%d] run %d launched\n", rank_, run); }
                            for (size_t i = 0; i < devs.size(); i++) {
                                RK(cudaSetDevice(devs[i]));
                                RK(cudaStreamSynchronize(sm[i]));
                                if (getenv("XP_HB")) { fprintf(stderr, "[r%d] run %d stream %zu done\n", rank_, run, i); }
                                u32 one = 1;
                                RK(cudaMemcpy(stop[i], &one, 4, cudaMemcpyHostToDevice));
                                RK(cudaDeviceSynchronize());
                            }
                            u64 h_[8] = {};
                            if (iread) RK(cudaMemcpy(h_, cnt[ri], sizeof h_, cudaMemcpyDeviceToHost));
                            barrier("end", run);
                            if (fo) {
                                fprintf(fo,
                                        "{\"mode\":\"%s\",\"place\":\"%s\",\"ws\":\"%s\",\"rs\":\"%s\",\"scope\":\"%s\","
                                        "\"wf\":%d,\"rf\":%d,\"stress\":%d,\"dmask\":%d,\"iters\":%d,\"slots\":%d,"
                                        "\"ok\":%llu,\"stale\":%llu,\"other\":%llu,\"timeout\":%llu}\n",
                                        single ? "one-process" : "two-process", priv ? "private" : "multicast", wsn[ws], rsn[rs], sys ? "sys" : "gpu", c.wf,
                                        c.rf, stress, c.dmask, iters, slots, h_[0], h_[1], h_[2], h_[3]);
                                fflush(fo);
                                printf("%-60s d%d ok %llu stale %llu other %llu tmo %llu\n", name, dm, h_[0], h_[1], h_[2], h_[3]);
                                fflush(stdout);
                            }
                            barrier("rep", run);
                        }
                    }
    if (fo) fclose(fo);
    return 0;
}
