/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Reproducibility study for the latent remote-arrive hand-off (S-MBAR-SCOPE): the tile-id
// broadcast of probes/mbar_scope (producer: weak remote store of round k into the consumer's
// slot, the variant's fence, a remote arrive on the consumer's `full`; consumer: wait, read,
// give back through `empty`), under legal levers that could delay the remote store:
//   - cluster size and placement: clusters of C CTAs (non-portable sizes allowed), producer
//     rank 0, consumer rank 1 or rank C-1 (`far`);
//   - co-resident kernel on another stream (`cot`): 1 = DSMEM traffic between the CTAs of its
//     own clusters, 2 = HBM / L2 streaming stores and loads, 3 = both;
//   - noise: the other warps of the producer and consumer CTAs run traffic: 1 = weak remote
//     stores into the peer's scratch, 2 = local 16-byte shared stores and loads, 3 = remote
//     loads from the peer's scratch, 4 / 5 = as 1 from the consumer only / producer only.
// Variants (v):
//   0 PFC  st.shared::cluster; fence.proxy.async.shared::cluster; arrive.release.cta; wait.acquire.cta
//          (ptxas: ST.E; MEMBAR.ALL.CTA; MEMBAR.ALL.GPU; FENCE.VIEW.ASYNC.S; bare SYNCS.ARRIVE.RED:
//          the tile-id broadcast of 2-CTA grouped-GEMM tile schedulers)
//   1 NOF  st.shared::cluster; arrive.release.cta (no fence)
//   2 PCTA st.shared::cluster; fence.acq_rel.cta; arrive.release.cta (MEMBAR.ALL.CTA only)
//   3 CLU  st.shared::cluster; arrive.release.cluster; wait.acquire.cluster (fenced control)
//   7 LSTS producer stores into its *own* slot (st.shared::cta), bare remote arrive; the
//          consumer reads the producer's slot with ld.shared::cluster (local store, remote reader)
//   8 LCLU as 7, with arrive.release.cluster / wait.acquire.cluster (fenced control)
//   9 LCTA as 7, with fence.acq_rel.cta before the arrive
//  10 LWW  STS.128 own slot, bare remote arrive; the peer then bulk-copies (cp.async.bulk
//          shared::cluster) its value over the same 16 bytes: write-write, the CuTe attention
//          backward shape (STS, bare arrive, peer UBLKCP); checked by the producer
//  11 LWWC as 10, with fence.acq_rel.cta before the arrive (control)
//  12 LWWP as 10, with fence.proxy.async.shared::cta before the arrive
//  13 LWWK as 10, with arrive.release.cluster / wait.acquire.cluster
//  14 LWWB as 10, with fence.proxy.async.shared::cta and arrive.release.cluster (full fix)
//  15 LWAR generic 16-byte load of the own slot, bare remote arrive, then the peer's bulk copy
//          over the slot; a load returning the copy's value counts as a failure (S-PROXY-WAR)
//  16 LWARK as 15, with arrive.release.cluster / wait.acquire.cluster (the usual stage release)
//  17 LWARC as 15, with fence.acq_rel.cta (MEMBAR.ALL.CTA) before the .cta arrive
//   6 LATE arrive first, then spin `d` ns, then the store (delay-window positive control)
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o window window.cu
// Run:   ./window v rounds launches C far noise cot d
//        env MAIN_SMEM_KB / COT_SMEM_KB: dynamic shared memory padding for the hand-off kernel /
//        the co-tenant (forces them onto different SMs when the sum exceeds the SM's capacity).
//        ./window 99 ms secs 0 0 0 cot 0: standalone co-tenant process (for MPS runs).
//        env COT_MULT: co-tenant CTAs per SM (grid = SMs x COT_MULT); COT_THREADS: its block
//        size (32-256, default 256); COT_SMEM_KB: its shared memory (>= 1, default 16).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define SCRATCH 1024

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned mapa(unsigned a, unsigned r) {
    unsigned d;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(r));
    return d;
}
__device__ __forceinline__ void st_remote(unsigned a, unsigned v) {
    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void arrive_remote(unsigned bar, int v) {
    if (v == 3 || v == 8 || v == 13 || v == 14 || v == 16) asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(bar) : "memory");
    else asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(bar) : "memory");
}
__device__ __forceinline__ bool try_wait(unsigned bar, unsigned parity, int v) {
    unsigned ok;
    if (v == 3 || v == 8 || v == 13 || v == 14 || v == 16)
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    else
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}

struct Log { unsigned stale, future, rounds, first_round, prod_smid; };
__device__ unsigned g_seen[8];  // SMs that ran a co-tenant CTA (bit per %smid)

// One instantiation per variant, so every variant's producer path is straight-line code (the
// STS / ST.E and the arrive back to back, as in the kernels) with no variant branch between.
template <int V>
__global__ void __launch_bounds__(256)
k(int, int rounds, int far, int noise, unsigned d, unsigned tag, Log* L, volatile int* stop) {
    constexpr int v = V;
    __shared__ __align__(8) unsigned long long full, empty;
    __shared__ unsigned slot;
    __shared__ __align__(16) unsigned scratch[SCRATCH];
    __shared__ __align__(16) unsigned slot4[4], src4[4];
    __shared__ __align__(8) unsigned long long bulkbar;
    unsigned rank, csize;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(csize));
    unsigned cl = blockIdx.x / csize;
    unsigned cons = far ? csize - 1 : 1;
    unsigned fullA = (unsigned)__cvta_generic_to_shared(&full);
    unsigned emptyA = (unsigned)__cvta_generic_to_shared(&empty);
    unsigned slotA = (unsigned)__cvta_generic_to_shared(&slot);
    unsigned scrA = (unsigned)__cvta_generic_to_shared(scratch);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(fullA));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(emptyA));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"((unsigned)__cvta_generic_to_shared(&bulkbar)));
        slot = 0xffffffffu;
        for (int i = 0; i < 4; i++) slot4[i] = 0xffffffffu;
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    bool prod = rank == 0, consr = rank == cons;
    if (prod && threadIdx.x == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); L[cl].prod_smid = sm; }
    unsigned peer = prod ? cons : 0;
    unsigned rslot = mapa(slotA, peer), rfull = mapa(fullA, peer), rempty = mapa(emptyA, peer);
    unsigned rscr = mapa(scrA, peer);
    if (threadIdx.x >= 32) {
        if (noise && (prod || consr)) {
            unsigned i = threadIdx.x, acc = 0;
            unsigned lscr = scrA + 16 * ((threadIdx.x - 32) % (SCRATCH / 4));
            if (noise == 1 || (noise == 4 && consr) || (noise == 5 && prod))
                // remote weak stores into the peer's scratch (4: consumer -> producer only,
                // 5: producer -> consumer only)
                while (!*stop) { st_remote(rscr + 4 * (i % SCRATCH), i); i += 224; }
            else if (noise == 2)  // local 16-byte shared stores and loads (MIO pressure, no fabric)
                while (!*stop) {
                    unsigned a, b, c, e;
                    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" :: "r"(lscr), "r"(i) : "memory");
                    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(e) : "r"(lscr) : "memory");
                    acc += a ^ e; i++;
                }
            else if (noise == 3)  // remote loads from the peer's scratch (fabric reads)
                while (!*stop) {
                    unsigned a;
                    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(a) : "r"(rscr + 4 * (i % SCRATCH)) : "memory");
                    acc += a; i += 224;
                }
            if (acc == 0x12345678u) scratch[0] = acc;
        }
    } else if (v >= 15 && threadIdx.x == 0 && prod) {
        // WAR (S-PROXY-WAR): read the own slot with a generic 16-byte load (round r sees the
        // peer's copy of round r-1), arrive on the peer's `full`, and only then consume the
        // loaded value; the peer then bulk-copies round r's value over the slot. A load that
        // returns round r's value read the later async write.
        unsigned bb = (unsigned)__cvta_generic_to_shared(&bulkbar);
        unsigned s4 = (unsigned)__cvta_generic_to_shared(slot4);
        unsigned stale = 0, future = 0, first = 0xffffffffu;
        for (int r = 0; r < rounds; r++) {
            unsigned want = r == 0 ? 0xffffffffu : (tag + r - 1) ^ 0x80000000u;
            unsigned a, b, c, e;
            asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%0], 16;" :: "r"(bb) : "memory");
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(a), "=r"(b), "=r"(c), "=r"(e) : "r"(s4) : "memory");
            if (v == 17) asm volatile("fence.acq_rel.cta;" ::: "memory");
            arrive_remote(rfull, v);
            while (!try_wait(bb, r & 1, v)) {}
            if (a != want) {
                if (first == 0xffffffffu) first = r;
                if (a == ((tag + r) ^ 0x80000000u)) future++; else stale++;
            }
            if (b + c + e == 0x12345u) scratch[1] = a;
            arrive_remote(rempty, v);
        }
        L[cl].stale = future; L[cl].future = stale; L[cl].rounds = rounds; L[cl].first_round = first;
        *stop = 1;
    } else if (v >= 10 && threadIdx.x == 0 && prod) {
        // WW: STS.128 into the own slot, bare remote arrive; the peer then bulk-copies its
        // value over the same 16 bytes (complete_tx on our bulkbar). After the copy completes
        // the slot must hold the peer's value; our value means the STS landed after the copy.
        unsigned bb = (unsigned)__cvta_generic_to_shared(&bulkbar);
        unsigned s4 = (unsigned)__cvta_generic_to_shared(slot4);
        unsigned stale = 0, future = 0, first = 0xffffffffu;
        for (int r = 0; r < rounds; r++) {
            unsigned val = tag + r, want = val ^ 0x80000000u;
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;" :: "r"(bb) : "memory");
            asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" :: "r"(s4), "r"(val) : "memory");
            if (v == 11) asm volatile("fence.acq_rel.cta;" ::: "memory");
            if (v == 12 || v == 14) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            arrive_remote(rfull, v);
            while (!try_wait(bb, r & 1, v)) {}
            unsigned got = *(volatile unsigned*)&slot4[0];
            if (got != want) {
                if (first == 0xffffffffu) first = r;
                if (got == val) stale++; else future++;
            }
            arrive_remote(rempty, v);  // the peer may reuse its source buffer
        }
        L[cl].stale = stale; L[cl].future = future; L[cl].rounds = rounds; L[cl].first_round = first;
        *stop = 1;
    } else if (v >= 10 && threadIdx.x == 0 && consr) {
        unsigned src = (unsigned)__cvta_generic_to_shared(src4);
        unsigned rs4 = mapa((unsigned)__cvta_generic_to_shared(slot4), 0);
        unsigned rbb = mapa((unsigned)__cvta_generic_to_shared(&bulkbar), 0);
        for (int r = 0; r < rounds; r++) {
            unsigned want = (tag + r) ^ 0x80000000u;
            if (r > 0) while (!try_wait(emptyA, (r - 1) & 1, v)) {}
            asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" :: "r"(src), "r"(want) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            while (!try_wait(fullA, r & 1, v)) {}
            asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], 16, [%2];"
                         :: "r"(rs4), "r"(src), "r"(rbb) : "memory");
        }
        while (!try_wait(emptyA, (rounds - 1) & 1, v)) {}
    } else if (threadIdx.x == 0 && prod) {
        for (int r = 0; r < rounds; r++) {
            unsigned val = tag + r;
            if (v == 6) {
                arrive_remote(rfull, v);
                unsigned long long t0 = gtime();
                while (gtime() - t0 < d) {}
                st_remote(rslot, val);
            } else if (v >= 7) {
                *(volatile unsigned*)&slot = val;
                if (v == 9) asm volatile("fence.acq_rel.cta;" ::: "memory");
                arrive_remote(rfull, v);
            } else {
                st_remote(rslot, val);
                if (v == 2) asm volatile("fence.acq_rel.cta;" ::: "memory");
                if (v == 0) asm volatile("fence.proxy.async.shared::cluster;" ::: "memory");
                arrive_remote(rfull, v);
            }
            while (!try_wait(emptyA, r & 1, v)) {}
        }
        *stop = 1;
    } else if (threadIdx.x == 0 && consr) {
        unsigned stale = 0, future = 0, first = 0xffffffffu;
        for (int r = 0; r < rounds; r++) {
            while (!try_wait(fullA, r & 1, v)) {}
            unsigned got;
            if (v >= 7) asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(got) : "r"(mapa(slotA, 0)) : "memory");
            else got = *(volatile unsigned*)&slot;
            unsigned want = tag + r;
            if (got != want) {
                if (first == 0xffffffffu) first = r;
                if ((int)(got - want) > 0 && got != 0xffffffffu) future++; else stale++;
            }
            if (v == 6) {
                // the late store must land before the give-back lets the producer go on
                unsigned long long t0 = gtime();
                while (*(volatile unsigned*)&slot != want && gtime() - t0 < 100000) {}
            }
            arrive_remote(rempty, v);
        }
        L[cl].stale = stale; L[cl].future = future; L[cl].rounds = rounds; L[cl].first_round = first;
    }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

// Co-resident traffic: 2-CTA clusters; mode bit 0: DSMEM stores into the peer CTA; bit 1:
// streaming global stores and loads over `buf` (n words). Runs until *stop or t_end.
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(256)
cot(int mode, unsigned* buf, size_t n, volatile int* stop, unsigned long long dur, unsigned words) {
    extern __shared__ __align__(16) unsigned s[];  // `words` words (COT_SMEM_KB)
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned rs = mapa((unsigned)__cvta_generic_to_shared(s), rank ^ 1);
    if (threadIdx.x == 0) { unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); atomicOr(&g_seen[(sm / 32) & 7], 1u << (sm % 32)); }
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned long long t0 = gtime();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    int it = 0;
    while (!*stop && gtime() - t0 < dur) {
        if (mode & 1) for (int j = 0; j < 16; j++) st_remote(rs + 4 * ((threadIdx.x + blockDim.x * j) % words), acc + j);
        if (mode & 2) {
            for (int j = 0; j < 8; j++) {
                size_t a = (i + (size_t)(it * 8 + j) * stride) % n;
                acc += buf[a];
                buf[(a + n / 2) % n] = acc;
            }
        }
        it++;
    }
    if (acc == 0xdeadbeef) buf[0] = acc;
    asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

int main(int argc, char** argv) {
    if (argc < 9) { fprintf(stderr, "usage: %s v rounds launches C far noise cot d\n", argv[0]); return 2; }
    int v = atoi(argv[1]), rounds = atoi(argv[2]), launches = atoi(argv[3]), C = atoi(argv[4]);
    int far = atoi(argv[5]), noise = atoi(argv[6]), cotm = atoi(argv[7]);
    unsigned d = atoi(argv[8]);
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    void (*kf)(int, int, int, int, unsigned, unsigned, Log*, volatile int*) = nullptr;
    switch (v) {
        case 0: kf = k<0>; break; case 1: kf = k<1>; break; case 2: kf = k<2>; break;
        case 3: kf = k<3>; break; case 6: kf = k<6>; break; case 7: kf = k<7>; break;
        case 8: kf = k<8>; break; case 9: kf = k<9>; break;
        case 10: kf = k<10>; break; case 11: kf = k<11>; break;
        case 12: kf = k<12>; break; case 13: kf = k<13>; break; case 14: kf = k<14>; break;
        case 15: kf = k<15>; break; case 16: kf = k<16>; break; case 17: kf = k<17>; break; default: kf = k<0>; break;
    }
    if (C > 8) cudaFuncSetAttribute(kf, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    int msm = getenv("MAIN_SMEM_KB") ? atoi(getenv("MAIN_SMEM_KB")) * 1024 : 0;
    int csm = (getenv("COT_SMEM_KB") ? atoi(getenv("COT_SMEM_KB")) : 16) * 1024;
    if (csm < 1024) csm = 1024;
    int cthr = getenv("COT_THREADS") ? atoi(getenv("COT_THREADS")) : 256;
    if (cthr < 32 || cthr > 256) cthr = 256;
    cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, msm);
    cudaFuncSetAttribute(cot, cudaFuncAttributeMaxDynamicSharedMemorySize, csm);
    if (v == 99) {  // standalone co-tenant: back-to-back launches of `rounds` ms for `launches` s
        unsigned* b; size_t nn = (size_t)1 << 28; int* st;
        cudaMalloc(&st, 4); cudaMemset(st, 0, 4);
        if (cotm & 2) { cudaMalloc(&b, nn * 4); cudaMemset(b, 1, nn * 4); } else { cudaMalloc(&b, 4096); nn = 1024; }
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
        long L = 0; float ms = 0;
        while (ms < launches * 1000.f) {
            int cm = getenv("COT_MULT") ? atoi(getenv("COT_MULT")) : 1;
            cot<<<sms * (cm < 1 ? 1 : cm), cthr, csm>>>(cotm, b, nn, st, (unsigned long long)rounds * 1000000ull, csm / 4);
            cudaError_t e = cudaDeviceSynchronize(); L++;
            if (e != cudaSuccess) { printf("cot: %s\n", cudaGetErrorString(e)); return 1; }
            cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
        }
        printf("co-tenant mode %d: %ld launches of %d ms\n", cotm, L, rounds);
        return 0;
    }
    int clusters = sms / C;
    // fewer clusters when the co-tenant needs room on the SMs
    if (cotm) clusters = clusters / 2 > 0 ? clusters / 2 : 1;
    Log* L; int* stop; int* cstop; unsigned* buf;
    size_t n = (size_t)1 << 28;  // 1 GiB of words for the HBM stream
    cudaMalloc(&L, clusters * sizeof(Log));
    cudaMalloc(&stop, sizeof(int));
    cudaMalloc(&cstop, sizeof(int));
    if (cotm & 2) { cudaMalloc(&buf, n * 4); cudaMemset(buf, 1, n * 4); } else { cudaMalloc(&buf, 4096); n = 1024; }
    cudaStream_t s0, s1;
    cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    Log* h = (Log*)malloc(clusters * sizeof(Log));
    unsigned long long stale = 0, future = 0, total = 0, clbad = 0, round0 = 0, coloc = 0, nprod = 0;
    int cmult = getenv("COT_MULT") ? atoi(getenv("COT_MULT")) : 1;
    if (cmult < 1) cmult = 1;
    for (int it = 0; it < launches; it++) {
        cudaMemsetAsync(L, 0, clusters * sizeof(Log), s0);
        cudaMemsetAsync(stop, 0, sizeof(int), s0);
        cudaMemsetAsync(cstop, 0, sizeof(int), s1);
        cudaStreamSynchronize(s0); cudaStreamSynchronize(s1);
        unsigned zero8[8] = {0};
        cudaMemcpyToSymbol(g_seen, zero8, sizeof(zero8));
        if (cotm) cot<<<sms * cmult, cthr, csm, s1>>>(cotm, buf, n, cstop, 200000000ull, csm / 4);
        unsigned tag = 1 + (unsigned)it * (unsigned)rounds;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(clusters * C); cfg.blockDim = dim3(256); cfg.stream = s0; cfg.dynamicSmemBytes = msm;
        cudaLaunchAttribute at[1];
        at[0].id = cudaLaunchAttributeClusterDimension;
        at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
        cfg.attrs = at; cfg.numAttrs = 1;
        cudaError_t e = cudaLaunchKernelEx(&cfg, kf, v, rounds, far, noise, d, tag, L, stop);
        if (e != cudaSuccess) { printf("launch: %s\n", cudaGetErrorString(e)); return 1; }
        e = cudaStreamSynchronize(s0);
        int one = 1;
        cudaMemcpyAsync(cstop, &one, sizeof(int), cudaMemcpyHostToDevice, s0);
        cudaStreamSynchronize(s0);
        cudaStreamSynchronize(s1);
        if (e != cudaSuccess) { printf("run: %s\n", cudaGetErrorString(e)); return 1; }
        cudaMemcpy(h, L, clusters * sizeof(Log), cudaMemcpyDeviceToHost);
        unsigned seen[8];
        cudaMemcpyFromSymbol(seen, g_seen, sizeof(seen));
        for (int c = 0; c < clusters; c++) { nprod++; unsigned sm = h[c].prod_smid; if (seen[(sm / 32) & 7] >> (sm % 32) & 1) coloc++; }
        for (int c = 0; c < clusters; c++) {
            stale += h[c].stale; future += h[c].future; total += h[c].rounds;
            if (h[c].stale + h[c].future) { clbad++; if (h[c].first_round == 0) round0++; }
        }
    }
    printf("v=%d C=%d far=%d noise=%d cot=%d d=%u rounds=%d launches=%d clusters=%d: hand-offs %llu stale %llu future %llu clusters_bad %llu (first at round 0: %llu) producer SM also ran a co-tenant CTA: %llu/%llu\n",
           v, C, far, noise, cotm, d, rounds, launches, clusters, total, stale, future, clbad, round0, coloc, nprod);
    return 0;
}
