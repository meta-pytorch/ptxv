/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Probe for S-REMOTE-INFLIGHT / S-CLUSTER-LIFE: what happens on GB300 to a remote mbarrier
// arrival (or a remote shared::cluster store) issued into a cluster peer that has already
// exited? FA4 2-CTA forward (L1) can exit the leader with CTA 1's final remote
// `.release.cta` arrivals in flight.
//
// Victim kernel `vic`: cluster (2,1,1), one CTA per SM (~200 KiB dynamic smem). After a
// cluster barrier, rank 0 records its SM and exits; rank 1 spins D us, then arrives on rank
// 0's mbarrier offset (mode 0) or stores a magic pattern into 1 KiB of rank 0's smem
// (mode 1), and records the issue time.
// Successor kernel `suc` (another stream, launched right after `vic`): one CTA per SM. Each
// CTA fills its first 4 KiB of smem with a pattern, initializes an mbarrier (count 1) at the
// victim's offset, records SM and start time, sleeps W us, and records any changed word and
// whether the mbarrier phase completed.
//   mode 0: victim rank 1 remote-arrives late (FA4 shape, widened window)
//   mode 1: victim rank 1 remote-stores 256 words late
//   mode 2: positive control of the detector: rank 0 does not exit; it fills, inits and
//           checks exactly like a successor, so the late arrival lands in a live CTA
//   mode 3: positive control for stores (as 2, with the mode 1 store)
//   mode 4: sanity: rank 1 spins but issues no remote operation
//   mode 5: control: rank 0 stays alive (zeroed smem, no mbarrier init) and rank 1 arrives late
//           on the uninitialized word (separates "target exited" from "not an mbarrier")
//   mode 6: as 0, but rank 0 initializes the mbarrier before exiting (FA4: the EMPTY barriers
//           are initialized)
// The host joins the logs: a successor CTA is *exposed* when it started on the SM of an
// exited rank 0 before that cluster's rank 1 issued its late operation, and was still
// checking after it.
// Build: nvcc -gencode arch=compute_103a,code=sm_103a -O2 -o rae rae.cu
// Run:   CUDA_VISIBLE_DEVICES=<free gpu> ./rae [runs] [delay_us] [first_mode] [last_mode] [no_successor]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int NW = 1024;       // pattern words (4 KiB)
constexpr int OFF = 512;       // mbarrier / store word offset (2 KiB)
constexpr unsigned MAGIC = 0xA5A50000u;

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ unsigned smid() {
    unsigned r;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void spin_us(unsigned us) {
    unsigned long long t0 = gtime();
    while (gtime() - t0 < us * 1000ull) __nanosleep(200);
}
__device__ __forceinline__ bool phase0_done(unsigned a) {
    unsigned v;
    asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared::cta.b64 p, [%1], 0;\n"
                 " selp.u32 %0, 1, 0, p;\n}" : "=r"(v) : "r"(a) : "memory");
    return v;
}

struct VicLog { unsigned sm0, sm1; unsigned long long t_exit0, t_issue, t_end1; };
struct SucLog { unsigned sm, changed, phase, first_bad; unsigned long long t_start, t_end; };

// Fill + init + wait + check; used by successors and by rank 0 in the positive controls.
__device__ void fill_check(unsigned* s, unsigned seed, unsigned wait_us, SucLog* L) {
    unsigned base = (unsigned)__cvta_generic_to_shared(s);
    for (int i = threadIdx.x; i < NW; i += blockDim.x) s[i] = seed ^ (i * 2654435761u);
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(base + OFF * 4));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        L->sm = smid();
        L->t_start = gtime();
    }
    __syncthreads();
    spin_us(wait_us);
    __syncthreads();
    if (threadIdx.x == 0) {
        L->phase = phase0_done(base + OFF * 4);
        unsigned ch = 0, fb = 0xffffffffu;
        for (int i = 0; i < NW; i++) {
            if (i == OFF || i == OFF + 1) continue;  // the mbarrier itself
            if (s[i] != (seed ^ (i * 2654435761u))) { if (!ch) fb = i; ch++; }
        }
        L->changed = ch;
        L->first_bad = fb;
        L->t_end = gtime();
    }
}

__global__ void __cluster_dims__(2, 1, 1) vic(int mode, unsigned delay_us, unsigned wait_us,
                                              VicLog* V, SucLog* C) {
    extern __shared__ __align__(1024) unsigned s[];
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    unsigned cl = blockIdx.x / 2;
    unsigned base = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("barrier.cluster.arrive.release;\n barrier.cluster.wait.acquire;" ::: "memory");
    if (rank == 0) {
        if (threadIdx.x == 0) V[cl].sm0 = smid();
        if (mode == 2 || mode == 3) { fill_check(s, 0x1234567u ^ cl, wait_us, &C[cl]); }
        if (mode == 5) { for (int i = threadIdx.x; i < NW; i += blockDim.x) s[i] = 0; __syncthreads(); spin_us(wait_us); }
        if (mode == 6 && threadIdx.x == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(base + OFF * 4));
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        if (threadIdx.x == 0) V[cl].t_exit0 = gtime();
        return;
    }
    if (threadIdx.x == 0) {
        V[cl].sm1 = smid();
        unsigned remote;
        // mode 2/3: give rank 0 time to fill and init (it starts its wait at once)
        spin_us(delay_us);
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(remote) : "r"(base + OFF * 4));
        V[cl].t_issue = gtime();
        if (mode == 0 || mode == 2 || mode == 5 || mode == 6) {
            asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
        } else if (mode == 4) {
            // sanity: no remote operation at all
        } else {
            for (int i = 0; i < 256; i++)
                asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(remote + 8 + 4 * i), "r"(MAGIC | i) : "memory");
        }
        V[cl].t_end1 = gtime();
    }
}

__global__ void suc(unsigned wait_us, SucLog* C) {
    extern __shared__ __align__(1024) unsigned s[];
    fill_check(s, 0x9e3779b9u ^ blockIdx.x, wait_us, &C[blockIdx.x]);
}

int main(int argc, char** argv) {
    int runs = argc > 1 ? atoi(argv[1]) : 50;
    unsigned delay = argc > 2 ? atoi(argv[2]) : 50;
    unsigned wait = delay * 3 + 50;
    int m0 = argc > 3 ? atoi(argv[3]) : 0, m1 = argc > 4 ? atoi(argv[4]) : 3;
    int nosuc = argc > 5 ? atoi(argv[5]) : 0;
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    int smem = 200 * 1024;
    CK(cudaFuncSetAttribute(vic, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CK(cudaFuncSetAttribute(suc, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CK(cudaFuncSetAttribute(vic, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    int ncl = sms / 2, nv = ncl * 2, ns = sms * 2;
    VicLog* V; SucLog* C; SucLog* P;
    CK(cudaMalloc(&V, ncl * sizeof(VicLog)));
    CK(cudaMalloc(&C, ns * sizeof(SucLog)));
    CK(cudaMalloc(&P, ncl * sizeof(SucLog)));
    cudaStream_t a, b;
    CK(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking));
    std::vector<VicLog> hv(ncl);
    std::vector<SucLog> hc(ns), hp(ncl);
    printf("sms %d, clusters %d, successor CTAs %d, delay %u us, wait %u us\n", sms, ncl, ns, delay, wait);
    for (int mode = m0; mode <= m1; mode++) {
        long exposed = 0, exp_bad = 0, exp_phase = 0, any_bad = 0, any_phase = 0, ctrl_hit = 0, ctrl_n = 0;
        long on_sm0 = 0, before_end1 = 0;
        double lag_sum = 0; long lag_n = 0;
        for (int r = 0; r < runs; r++) {
            CK(cudaMemset(V, 0, ncl * sizeof(VicLog)));
            CK(cudaMemset(C, 0, ns * sizeof(SucLog)));
            CK(cudaMemset(P, 0, ncl * sizeof(SucLog)));
            CK(cudaDeviceSynchronize());
            vic<<<nv, 128, smem, a>>>(mode, delay, wait, V, P);
            if (mode != 2 && mode != 3 && mode != 5 && !nosuc) suc<<<ns, 128, smem, b>>>(wait, C);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hv.data(), V, ncl * sizeof(VicLog), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hc.data(), C, ns * sizeof(SucLog), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hp.data(), P, ncl * sizeof(SucLog), cudaMemcpyDeviceToHost));
            if (mode == 2 || mode == 3) {
                for (auto& p : hp) { ctrl_n++; ctrl_hit += (mode == 2 ? p.phase : p.changed > 0); }
                continue;
            }
            for (auto& c : hc) { any_bad += c.changed > 0; any_phase += c.phase; }
            for (auto& v : hv)
                for (auto& c : hc) {
                    if (c.sm != v.sm0 || c.t_start < v.t_exit0) continue;
                    on_sm0++;
                    lag_sum += (double)(c.t_start - v.t_exit0); lag_n++;
                    if (c.t_start < v.t_end1) before_end1++;
                    if (c.t_start < v.t_issue && c.t_end > v.t_end1) {
                        exposed++;
                        exp_bad += c.changed > 0;
                        exp_phase += c.phase;
                    }
                }
        }
        if (mode != 2 && mode != 3)
            printf("mode %d: %d runs: successor CTAs on an exited rank-0 SM %ld (started before rank 1 finished %ld, "
                   "mean lag after rank-0 exit %.1f us); exposed %ld: changed smem %ld, completed phase %ld; "
                   "all successors: changed %ld, phase %ld\n",
                   mode, runs, on_sm0, before_end1, lag_n ? lag_sum / lag_n / 1000 : 0.0, exposed, exp_bad,
                   exp_phase, any_bad, any_phase);
        else
            printf("mode %d (control): %ld / %ld live rank-0 CTAs detected the late %s\n", mode, ctrl_hit, ctrl_n,
                   mode == 2 ? "arrival" : "store");
    }
    return 0;
}
