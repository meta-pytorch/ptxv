// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Cross-process virtual aliases on GB300 (S-VA-ALIAS, option d of forum post 55).
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

// Counters per run: [0] ok [1] stale [2] other [3] timeouts [4] sink.
__global__ void k_writer(u32 *B, int iters, Cfg c, u32 *abort_, u32 seed0) {
    const u32 s = blockIdx.x, l = threadIdx.x & 31, idx = s * 32 + l;
    if (threadIdx.x >= 32) return;
    u32 *pay = B + PAY + idx * LINEW, *flag = B + FLAG + idx * LINEW, *ack = B + ACK + idx * LINEW;
    u32 seed = seed0 ^ (idx * 2654435761u);
    for (u32 k = 1; k <= (u32)iters; k++) {
        u64 t0 = clock64();
        while (ld_ack(ack) < k - 1) {
            if (clock64() - t0 > TIMEOUT || *(volatile u32 *)abort_) return;
        }
        alias_fence();  // the handshake itself is kept sound
        delay(seed, c.dmask);
        asm volatile("st.global.v4.u32 [%0], {%1,%1,%1,%1};" ::"l"(pay), "r"(k) : "memory");
        if (c.wf) alias_fence();
        st_flag(flag, k, c.ws, c.sys);
    }
}

__global__ void k_reader(u32 *B, int iters, Cfg c, u64 *cnt, u32 *abort_, u32 *done) {
    const u32 s = blockIdx.x, w = threadIdx.x >> 5, l = threadIdx.x & 31, idx = s * 32 + l;
    u32 *pay = B + PAY + idx * LINEW, *flag = B + FLAG + idx * LINEW, *ack = B + ACK + idx * LINEW;
    if (w > 0) {
        // Co-reader warps: keep the CTA's payload lines in L1 while warp 0 polls.
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
        sink += ld_weak(pay);  // warm the line in L1
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
        asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ack), "r"(k) : "memory");
    }
    if (sink == 0xffffffffffffull) atomicAdd(cnt + 4, 1ull);
    atomicAdd(cnt + 0, ok);
    atomicAdd(cnt + 1, stale);
    atomicAdd(cnt + 2, other);
    atomicAdd(cnt + 3, tmo);
    __syncwarp();
    if (l == 0) *(volatile u32 *)(done + s) = 1;
}

// NVLink background traffic: copy from the shared (remote for one side) buffer region into a
// local scratch until stopped.
__global__ void k_traffic(const uint4 *src, uint4 *dst, size_t n, u32 *stop) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, st = (size_t)gridDim.x * blockDim.x;
    while (!*(volatile u32 *)stop) {
        for (size_t j = i; j < n; j += st) dst[j] = src[j];
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
        int tries = 0;
        while (stat(p, &sb) != 0) {
            usleep(200);
            if (++tries > 50000000) {
                fprintf(stderr, "barrier %s timeout\n", tag);
                exit(1);
            }
        }
    }
}

int main(int argc, char **argv) {
    const char *out = argc > 1 ? argv[1] : "xp.jsonl";
    int iters = argc > 2 ? atoi(argv[2]) : 2000;
    int slots = argc > 3 ? atoi(argv[3]) : 64;
    if (slots > MAXSLOTS) slots = MAXSLOTS;
    bool single = getenv("XP_SINGLE") && atoi(getenv("XP_SINGLE"));
    const char *only = getenv("XP_ONLY");  // substring filter on config names
    if (!single) {
        rank_ = atoi(getenv("RANK") ? getenv("RANK") : "0");
        world = atoi(getenv("WORLD_SIZE") ? getenv("WORLD_SIZE") : "1");
        if (world != 2) {
            fprintf(stderr, "needs 2 ranks (or XP_SINGLE=1)\n");
            return 1;
        }
    }
    dir = getenv("XP_DIR") ? getenv("XP_DIR") : "/dev/shm/xp";
    mkdir(dir.c_str(), 0700);
    CK(cuInit(0));
    int ldev = single ? 0 : atoi(getenv("LOCAL_RANK") ? getenv("LOCAL_RANK") : "0");
    CUdevice dev0, dev1;
    CK(cuDeviceGet(&dev0, ldev));
    CK(cuDeviceGet(&dev1, single ? 1 : ldev));
    RK(cudaSetDevice(ldev));
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = single ? 0 : 0;  // the owner is GPU 0 / rank 0
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t gran = 0;
    CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size_t bytes = (size_t)WORDS * 4;
    size_t tbytes = 256ull << 20;  // traffic source region after the message words
    size_t size = ((bytes + tbytes + gran - 1) / gran) * gran;
    CUmemGenericAllocationHandle h;
    char hp[512];
    snprintf(hp, sizeof hp, "%s/handle", dir.c_str());
    if (single || rank_ == 0) {
        CK(cuMemCreate(&h, size, &prop, 0));
        if (!single) {
            CUmemFabricHandle fh;
            CK(cuMemExportToShareableHandle(&fh, h, CU_MEM_HANDLE_TYPE_FABRIC, 0));
            FILE *f = fopen((std::string(hp) + ".tmp").c_str(), "wb");
            fwrite(&fh, sizeof fh, 1, f);
            fclose(f);
            rename((std::string(hp) + ".tmp").c_str(), hp);
        }
    } else {
        struct stat sb;
        while (stat(hp, &sb) != 0) usleep(1000);
        CUmemFabricHandle fh;
        FILE *f = fopen(hp, "rb");
        if (fread(&fh, sizeof fh, 1, f) != 1) return 1;
        fclose(f);
        CK(cuMemImportFromShareableHandle(&h, &fh, CU_MEM_HANDLE_TYPE_FABRIC));
    }
    CUdeviceptr va;
    CK(cuMemAddressReserve(&va, size, 0, 0, 0));
    CK(cuMemMap(va, size, 0, h, 0));
    CUmemAccessDesc acc[2] = {};
    acc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc[0].location.id = ldev;
    acc[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    acc[1] = acc[0];
    acc[1].location.id = 1;
    CK(cuMemSetAccess(va, size, acc, single ? 2 : 1));
    u32 *B = (u32 *)va;
    // Per-process scratch: counters, abort, done, traffic destination, stop flag.
    std::vector<int> devs = single ? std::vector<int>{0, 1} : std::vector<int>{ldev};
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
    const char *wsn[] = {"volatile", "relaxed", "release"}, *rsn[] = {"volatile", "relaxed", "acquire"};
    int run = 0;
    // placement 0: rank 1 (peer, through its imported mapping) writes, rank 0 (owner) reads.
    // placement 1: rank 0 writes its own memory, rank 1 reads it remotely through its mapping.
    for (int place = 0; place < 2; place++)
        for (int stress = 0; stress < 2; stress++)
            for (int sys = 1; sys >= 0; sys--)
                for (int fen = 0; fen < 4; fen++)
                    for (int ws = 0; ws < 3; ws++)
                        for (int rs = 0; rs < 3; rs++) {
                            Cfg c{ws, rs, sys, fen & 1, (fen >> 1) & 1, stress, 0};
                            char name[256];
                            snprintf(name, sizeof name, "%s %s/%s.%s wf%d rf%d stress%d",
                                     place ? "owner-writes" : "peer-writes", wsn[ws], rsn[rs],
                                     sys ? "sys" : "gpu", c.wf, c.rf, stress);
                            if (only && !strstr(name, only)) continue;
                            for (int dm = 0; dm < 2; dm++) {
                                c.dmask = dm ? 0x3ff : 0;
                                run++;
                                int wdev = single ? (place ? 0 : 1) : -1, rdev = single ? (place ? 1 : 0) : -1;
                                bool iwrite = single || rank_ == (place ? 0 : 1);
                                bool iread = single || rank_ == (place ? 1 : 0);
                                // Reset the message words (owner side) and the counters.
                                if (single || rank_ == 0) {
                                    RK(cudaSetDevice(single ? 0 : ldev));
                                    RK(cudaMemset(B, 0, bytes));
                                    RK(cudaDeviceSynchronize());
                                }
                                for (size_t i = 0; i < devs.size(); i++) {
                                    RK(cudaSetDevice(devs[i]));
                                    RK(cudaMemset(cnt[i], 0, 8 * sizeof(u64)));
                                    RK(cudaMemset(abort_[i], 0, 4));
                                    RK(cudaMemset(done[i], 0, MAXSLOTS * 4));
                                    RK(cudaMemset(stop[i], 0, 4));
                                    RK(cudaDeviceSynchronize());
                                }
                                barrier("start", run);
                                int ri = single ? (rdev == 0 ? 0 : 1) : 0, wi = single ? (wdev == 0 ? 0 : 1) : 0;
                                if (iread) {
                                    RK(cudaSetDevice(single ? rdev : ldev));
                                    k_reader<<<slots, stress ? 64 : 32, 0, sm[ri]>>>(B, iters, c, cnt[ri], abort_[ri], done[ri]);
                                    RK(cudaGetLastError());
                                }
                                if (iwrite) {
                                    RK(cudaSetDevice(single ? wdev : ldev));
                                    if (stress)
                                        k_traffic<<<16, 256, 0, st[wi]>>>((const uint4 *)(B + WORDS), tdst[wi],
                                                                         tbytes / 16, stop[wi]);
                                    k_writer<<<slots, 32, 0, sm[wi]>>>(B, iters, c, abort_[wi], 12345u + run);
                                    RK(cudaGetLastError());
                                }
                                for (size_t i = 0; i < devs.size(); i++) {
                                    RK(cudaSetDevice(devs[i]));
                                    RK(cudaStreamSynchronize(sm[i]));
                                    u32 one = 1;
                                    RK(cudaMemcpy(stop[i], &one, 4, cudaMemcpyHostToDevice));
                                    RK(cudaDeviceSynchronize());
                                }
                                u64 h_[8] = {};
                                if (iread) RK(cudaMemcpy(h_, cnt[ri], sizeof h_, cudaMemcpyDeviceToHost));
                                barrier("end", run);
                                // The reader's process reports (rank 0 in placement 0; rank 1 writes a
                                // side file merged by rank 0).
                                char sp[512];
                                snprintf(sp, sizeof sp, "%s/r_%d", dir.c_str(), run);
                                if (!single && iread && rank_ == 1) {
                                    FILE *f = fopen((std::string(sp) + ".tmp").c_str(), "wb");
                                    fwrite(h_, sizeof h_, 1, f);
                                    fclose(f);
                                    rename((std::string(sp) + ".tmp").c_str(), sp);
                                }
                                if (fo) {
                                    if (!single && !iread) {
                                        struct stat sb;
                                        while (stat(sp, &sb) != 0) usleep(500);
                                        FILE *f = fopen(sp, "rb");
                                        if (fread(h_, sizeof h_, 1, f) != 1) return 1;
                                        fclose(f);
                                    }
                                    fprintf(fo,
                                            "{\"mode\":\"%s\",\"place\":\"%s\",\"ws\":\"%s\",\"rs\":\"%s\",\"scope\":\"%s\","
                                            "\"wf\":%d,\"rf\":%d,\"stress\":%d,\"dmask\":%d,\"iters\":%d,\"slots\":%d,"
                                            "\"ok\":%llu,\"stale\":%llu,\"other\":%llu,\"timeout\":%llu}\n",
                                            single ? "single-mapping" : "two-process", place ? "owner-writes" : "peer-writes",
                                            wsn[ws], rsn[rs], sys ? "sys" : "gpu", c.wf, c.rf, stress, c.dmask, iters, slots,
                                            h_[0], h_[1], h_[2], h_[3]);
                                    fflush(fo);
                                    printf("%-55s d%d ok %llu stale %llu other %llu tmo %llu\n", name, dm, h_[0], h_[1], h_[2], h_[3]);
                                    fflush(stdout);
                                }
                                barrier("rep", run);
                            }
                        }
    if (fo) fclose(fo);
    return 0;
}
