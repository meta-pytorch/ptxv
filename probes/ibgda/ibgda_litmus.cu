// IBGDA ordering litmus probes (GB300 + ConnectX, DOCA GPUNetIO device verbs via mscclpp's
// PortChannelIbgdaDeviceHandle). One CTA per side: rank 0 = sender, rank 1 = receiver.
// Every iteration uses fresh tags: word w of iteration it in config cfg is
//   tag(cfg, it, w) = cfg << 56 | it << 24 | w            (w < 2^24, it < 2^32)
// so a stale word (older iteration / zero) is distinguishable from a newer one.
//
// Buffer layout (same on both ranks; the whole GpuBuffer is registered):
//   SRC_OFF  .. +3 MiB : sender source slots (slot it % 3)
//   DST_OFF  .. +1 MiB : receiver destination
//   CTRL_OFF          : TOKEN, ACK, FLAG, P7 word, DONE, SMALLSRC (one 128 B line each)
#include <mscclpp/port_channel_ibgda_device.hpp>

#include <cstdint>

using namespace mscclpp;

#define SRC_OFF (0ull)
#define SLOT_BYTES (1ull << 20)
#define DST_OFF (4ull << 20)
#define CTRL_OFF (8ull << 20)
#define TOKEN_OFF (CTRL_OFF + 0)
#define ACK_OFF (CTRL_OFF + 128)
#define FLAG_OFF (CTRL_OFF + 256)
#define P7_OFF (CTRL_OFF + 384)
#define DONE_OFF (CTRL_OFF + 512)
#define SMALLSRC_OFF (CTRL_OFF + 640)

#define NEWBIT (1ull << 55)

enum Mode {
  M_P1 = 1, // DOCA put + DOCA signal (FA) on the same QP
  M_P2 = 2, // DOCA put on QP A, DOCA signal on QP B
  M_P3 = 3, // custom post: write + FA WQEs, one doorbell, chosen fence
  M_P4 = 4, // DOCA put (large) then DOCA put (8 B flag) on the same QP
  M_P5 = 5, // DOCA put on QP A, flush A (poll CQE), DOCA signal on QP B
  M_P6A = 6, // put, overwrite source without CQ poll, signal (same QP)
  M_P6B = 7, // put, flush, overwrite source, signal (same QP)
};

// variant bits
#define V_DBFENCE(v) ((v) & 7) // P3: 0 none, 1 fence.release.cta, 2 .gpu, 3 .sys, 4 fence.sc.sys
#define V_PF_NONE 0x10 // no payload fence (bar.sync only) before posting
#define V_BATCH 0x20 // put with SKIP_DB_RINGING: the following op's doorbell covers both
#define V_FLAG4 0x40 // P4: the flag put is 4 bytes (mscclpp's IBGDA signal token size)
#define V_RECV(v) (((v) >> 8) & 3) // 0 ld.acquire.sys token + ld.relaxed.sys data
                                   // 1 ld.relaxed.sys token (no fence) + weak data
                                   // 2 ld.acquire.sys token + weak data
                                   // 3 ld.relaxed.sys token + fence.acquire.sys + ld.relaxed.sys data

// result slots
enum {
  R_STATUS = 0, // 0 ok, 1 token timeout, 2 ack timeout, 3 cq error, 4 cq timeout, 5 unsupported
  R_ITERS,
  R_BAD_ITERS,
  R_STALE,
  R_FUTURE,
  R_NEW,
  R_OTHER,
  R_FIRST_IT,
  R_FIRST_W,
  R_FIRST_V,
  R_CQ_STATUS,
  R_NS,
  R_ZERO,
  R_P7_WORD,
  R_P7_GPU,
  R_P7_NIC,
  R_SYND,
  R_INFO0 = 32,
};

__device__ __forceinline__ uint64_t gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ uint64_t ld_acq_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t ld_rlx_sys(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint64_t ld_weak(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_weak(uint64_t* p, uint64_t v) {
  asm volatile("st.global.b64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void fence_acq_sys() {
  asm volatile("fence.acquire.sys;" ::: "memory");
}

template <int K>
__device__ __forceinline__ void db_fence() {
  if (K == 1) asm volatile("fence.release.cta;" ::: "memory");
  if (K == 2) asm volatile("fence.release.gpu;" ::: "memory");
  if (K == 3) asm volatile("fence.release.sys;" ::: "memory");
  if (K == 4) asm volatile("fence.sc.sys;" ::: "memory");
}

__device__ __forceinline__ uint64_t tag(int cfg, uint64_t it, uint64_t w) {
  return ((uint64_t)cfg << 56) | (it << 24) | w;
}

// Non-trapping flush: poll the send CQE of the last reserved WQE, with a timeout.
// Returns 0, a negative DOCA status (error CQE), or 1 on timeout.
__device__ int flush_nb(PortChannelIbgdaDeviceHandle& ch, uint64_t timeout, uint64_t* res) {
  doca_gpu_dev_verbs_qp* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(ch.qp_);
  uint64_t ticket = *(volatile uint64_t*)&qp->sq_rsvd_index;
  if (ticket == 0) return 0;
  --ticket;
  doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  uint64_t t0 = gtime();
  while (true) {
    int st = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, ticket);
    if (st == 0) return 0;
    if (st < 0) {
      const doca_gpunetio_ib_mlx5_cqe64* cqe =
          reinterpret_cast<const doca_gpunetio_ib_mlx5_cqe64*>(cq->cqe_daddr);
      const doca_gpunetio_ib_mlx5_err_cqe_ex* err =
          reinterpret_cast<const doca_gpunetio_ib_mlx5_err_cqe_ex*>(&cqe[(ticket + cq->cqe_rsvd) & (cq->cqe_num - 1)]);
      res[R_SYND] = ((uint64_t)err->syndrome) | ((uint64_t)err->vendor_err_synd << 8) |
          ((uint64_t)err->hw_err_synd << 16) |
          ((uint64_t)doca_gpu_dev_verbs_bswap16(err->wqe_counter) << 32);
      return st;
    }
    if (gtime() - t0 > timeout) return 1;
  }
}

// Fence-configurable post of [RDMA write, 8 B fetch-add] as two WQEs with one doorbell.
// WQE contents come from DOCA's own prepare helpers (identical format, CQ_UPDATE on both,
// no fence bit). The QP is owned by this single thread (no reservation atomics, no lock).
// Doorbell sequence mirrors doca_gpu_dev_verbs_submit_db: DB, DBR, DB; the only fence is K,
// placed between the WQE stores and the first doorbell store.
template <int K>
__device__ int post_write_fa(PortChannelIbgdaDeviceHandle& ch, uint64_t dstOff, uint64_t srcOff,
                             uint32_t size, uint64_t sigOff) {
  doca_gpu_dev_verbs_qp* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(ch.qp_);
  const int handler = (int)qp->nic_handler;
  if (handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB &&
      handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_NO_DBR)
    return -1;
  volatile uint64_t* rsvd = (volatile uint64_t*)&qp->sq_rsvd_index;
  const uint64_t base = *rsvd;
  *rsvd = base + 2;
  doca_gpu_dev_verbs_wqe* w0 = doca_gpu_dev_verbs_get_wqe_ptr(qp, (uint16_t)base);
  doca_gpu_dev_verbs_wqe_prepare_write(
      qp, w0, (uint16_t)base, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, ch.remoteBase_ + dstOff, ch.rkeyBe_,
      ch.localBase_ + srcOff, ch.lkeyBe_, size);
  doca_gpu_dev_verbs_wqe* w1 = doca_gpu_dev_verbs_get_wqe_ptr(qp, (uint16_t)(base + 1));
  doca_gpu_dev_verbs_wqe_prepare_atomic(
      qp, w1, (uint16_t)(base + 1), DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
      DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, ch.remoteBase_ + sigOff, ch.rkeyBe_, ch.sinkAddr_,
      ch.sinkLkeyBe_, sizeof(uint64_t), 1ull, 0);
  *(volatile uint64_t*)&qp->sq_ready_index = base + 2;
  *(volatile uint64_t*)&qp->sq_wqe_pi = base + 2;
  db_fence<K>();
  uint64_t dbv = doca_gpu_dev_verbs_prepare_db(qp, base + 2);
  uint64_t* db = (uint64_t*)qp->sq_db;
  asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;" ::"l"(db), "l"(dbv) : "memory");
  if (handler == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
    doca_priv_gpu_dev_verbs_update_dbr(qp, (uint32_t)(base + 2));
    asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;" ::"l"(db), "l"(dbv) : "memory");
  }
  return 0;
}

__device__ int post_p3(int k, PortChannelIbgdaDeviceHandle& ch, uint64_t dstOff, uint64_t srcOff,
                       uint32_t size, uint64_t sigOff) {
  switch (k) {
    case 0: return post_write_fa<0>(ch, dstOff, srcOff, size, sigOff);
    case 1: return post_write_fa<1>(ch, dstOff, srcOff, size, sigOff);
    case 2: return post_write_fa<2>(ch, dstOff, srcOff, size, sigOff);
    case 3: return post_write_fa<3>(ch, dstOff, srcOff, size, sigOff);
    default: return post_write_fa<4>(ch, dstOff, srcOff, size, sigOff);
  }
}

__device__ void fill(uint8_t* buf, uint64_t off, uint64_t nwords, uint64_t v0, uint64_t extra) {
  uint64_t* p = (uint64_t*)(buf + off);
  for (uint64_t w = threadIdx.x; w < nwords; w += blockDim.x) st_weak(p + w, (v0 | w) | extra);
}

extern "C" __global__ void __launch_bounds__(1024) probe_sender(
    PortChannelIbgdaDeviceHandle* chs, uint8_t* buf, uint64_t* res, int mode, int variant,
    int cfg, int iters, uint64_t size, uint64_t timeout) {
  __shared__ int s_abort;
  PortChannelIbgdaDeviceHandle cha = chs[0];
  PortChannelIbgdaDeviceHandle chb = chs[1];
  const uint64_t nwords = size / 8;
  const uint64_t* ack = (const uint64_t*)(buf + ACK_OFF);
  uint64_t prev_ack = 0;
  if (threadIdx.x == 0) s_abort = 0;
  uint64_t t0 = gtime();
  __syncthreads();
  int it;
  for (it = 1; it <= iters; ++it) {
    const uint64_t slot = SRC_OFF + (uint64_t)(it % 3) * SLOT_BYTES;
    const uint64_t tb = tag(cfg, it, 0);
    fill(buf, slot, nwords, tb, 0);
    if (threadIdx.x == 0) st_weak((uint64_t*)(buf + SMALLSRC_OFF), tb);
    __syncthreads();
    if (threadIdx.x == 0) {
      if (!(variant & V_PF_NONE)) __threadfence_system();
      const bool batch = variant & V_BATCH;
      switch (mode) {
        case M_P1:
          cha.put(DST_OFF, slot, size, batch);
          cha.atomicAdd(TOKEN_OFF, 1);
          break;
        case M_P2:
          cha.put(DST_OFF, slot, size);
          chb.atomicAdd(TOKEN_OFF, 1);
          break;
        case M_P3:
          if (post_p3(V_DBFENCE(variant), cha, DST_OFF, slot, (uint32_t)size, TOKEN_OFF) != 0) {
            res[R_STATUS] = 5;
            s_abort = 1;
          }
          break;
        case M_P4:
          cha.put(DST_OFF, slot, size, batch);
          cha.put(FLAG_OFF, SMALLSRC_OFF, (variant & V_FLAG4) ? 4 : 8);
          break;
        case M_P5: {
          cha.put(DST_OFF, slot, size);
          int st = flush_nb(cha, timeout, res);
          if (st) {
            res[R_STATUS] = st < 0 ? 3 : 4;
            res[R_CQ_STATUS] = (uint64_t)(int64_t)st;
            s_abort = 1;
          } else {
            chb.atomicAdd(TOKEN_OFF, 1);
          }
          break;
        }
        case M_P6A:
          cha.put(DST_OFF, slot, size);
          break;
        case M_P6B: {
          cha.put(DST_OFF, slot, size);
          int st = flush_nb(cha, timeout, res);
          if (st) {
            res[R_STATUS] = st < 0 ? 3 : 4;
            res[R_CQ_STATUS] = (uint64_t)(int64_t)st;
            s_abort = 1;
          }
          break;
        }
      }
    }
    if (mode == M_P6A || mode == M_P6B) {
      __syncthreads();
      fill(buf, slot, nwords, tb, NEWBIT); // overwrite the source with the "new" pattern
      __syncthreads();
      if (threadIdx.x == 0 && !s_abort) cha.atomicAdd(TOKEN_OFF, 1);
    }
    if (threadIdx.x == 0 && !s_abort) {
      // The ACK word is advanced by one NIC fetch-add per iteration; wait for its raw bytes to
      // change (independent of the byte order the NIC uses for the add).
      uint64_t tw = gtime();
      while (true) {
        uint64_t a = ld_acq_sys(ack);
        if (a != prev_ack) {
          prev_ack = a;
          break;
        }
        if (gtime() - tw > timeout) {
          res[R_STATUS] = 2;
          s_abort = 1;
          break;
        }
      }
      if (!s_abort) {
        int st = flush_nb(cha, timeout, res);
        if (!st) st = flush_nb(chb, timeout, res);
        if (st) {
          res[R_STATUS] = st < 0 ? 3 : 4;
          res[R_CQ_STATUS] = (uint64_t)(int64_t)st;
          s_abort = 1;
        }
        // raw fetched (old) value the NIC wrote back to the local sink word
        if (it <= 4) res[R_INFO0 + it] = ld_rlx_sys((const uint64_t*)cha.sinkAddr_);
        if (it <= 4) res[R_INFO0 + 8 + it] = prev_ack; // raw ACK word after `it` fetch-adds
      }
    }
    __syncthreads();
    if (s_abort) break;
  }
  if (threadIdx.x == 0) {
    res[R_ITERS] = (uint64_t)(it - 1);
    res[R_NS] = gtime() - t0;
  }
}

extern "C" __global__ void __launch_bounds__(1024) probe_receiver(
    PortChannelIbgdaDeviceHandle* chs, uint8_t* buf, uint64_t* res, int mode, int variant,
    int cfg, int iters, uint64_t size, uint64_t timeout) {
  __shared__ int s_abort;
  __shared__ unsigned long long s_cnt[6]; // stale future new other zero bad
  __shared__ unsigned long long s_first; // (w + 1) of the first bad word in this iteration
  PortChannelIbgdaDeviceHandle ch = chs[0];
  const uint64_t nwords = size / 8;
  const int rv = V_RECV(variant);
  const uint64_t* token = (const uint64_t*)(buf + TOKEN_OFF);
  const uint64_t* flag = (const uint64_t*)(buf + FLAG_OFF);
  const uint64_t* dst = (const uint64_t*)(buf + DST_OFF);
  const bool p6 = (mode == M_P6A || mode == M_P6B);
  uint64_t prev_tok = 0;
  if (threadIdx.x == 0) s_abort = 0;
  uint64_t t0 = gtime();
  __syncthreads();
  int it;
  for (it = 1; it <= iters; ++it) {
    if (threadIdx.x < 6) s_cnt[threadIdx.x] = 0;
    if (threadIdx.x == 0) {
      s_first = 0;
      uint64_t tw = gtime();
      const uint64_t tb = tag(cfg, it, 0);
      while (true) {
        bool ok;
        if (mode == M_P4) {
          uint64_t f = (rv == 1 || rv == 3) ? ld_rlx_sys(flag) : ld_acq_sys(flag);
          // 4-byte flag: its low word (little-endian) carries the iteration's low 8 bits.
          ok = (variant & V_FLAG4) ? ((uint32_t)f == (uint32_t)tb) : (f == tb);
        } else {
          // one NIC fetch-add per iteration: wait for the raw token bytes to change
          uint64_t t = (rv == 1 || rv == 3) ? ld_rlx_sys(token) : ld_acq_sys(token);
          ok = (t != prev_tok);
          if (ok) {
            prev_tok = t;
            if (it <= 4) res[R_INFO0 + it] = t; // raw token after `it` fetch-adds
          }
        }
        if (ok) break;
        if (gtime() - tw > timeout) {
          res[R_STATUS] = 1;
          s_abort = 1;
          break;
        }
      }
      if (rv == 3) fence_acq_sys();
    }
    __syncthreads();
    if (s_abort) break;
    unsigned long long c_st = 0, c_fu = 0, c_nw = 0, c_ot = 0, c_ze = 0;
    unsigned long long first = 0;
    for (uint64_t w = threadIdx.x; w < nwords; w += blockDim.x) {
      const uint64_t v = (rv == 1 || rv == 2) ? ld_weak(dst + w) : ld_rlx_sys(dst + w);
      const uint64_t e = tag(cfg, it, w);
      if (v == e) continue;
      if (p6 && v == (e | NEWBIT)) {
        c_nw++;
      } else {
        if (v == 0) {
          c_ze++;
          c_st++;
        } else if ((v >> 56) != (uint64_t)cfg || (v & 0xffffff) != w || (v & NEWBIT)) {
          c_ot++;
        } else {
          uint64_t vit = (v >> 24) & 0xffffffffull;
          if (vit < (uint64_t)it) c_st++;
          else c_fu++;
        }
      }
      if (!first) first = w + 1;
    }
    if (c_st | c_fu | c_nw | c_ot) {
      atomicAdd(&s_cnt[0], c_st);
      atomicAdd(&s_cnt[1], c_fu);
      atomicAdd(&s_cnt[2], c_nw);
      atomicAdd(&s_cnt[3], c_ot);
      atomicAdd(&s_cnt[4], c_ze);
      atomicAdd(&s_cnt[5], 1ull);
      atomicCAS(&s_first, 0ull, first);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      // P6: "new" words are the measured outcome, not a failure; bad = stale/future/other.
      const bool bad = (s_cnt[0] | s_cnt[1] | s_cnt[3]) != 0;
      if (s_cnt[5]) {
        res[R_STALE] += s_cnt[0];
        res[R_FUTURE] += s_cnt[1];
        res[R_NEW] += s_cnt[2];
        res[R_OTHER] += s_cnt[3];
        res[R_ZERO] += s_cnt[4];
        if (s_cnt[2]) res[R_INFO0 + 0] += 1; // iterations with any new word
        if (bad) {
          res[R_BAD_ITERS] += 1;
          if (!res[R_FIRST_IT]) {
            res[R_FIRST_IT] = it;
            res[R_FIRST_W] = s_first - 1;
            res[R_FIRST_V] = ld_rlx_sys(dst + (s_first - 1));
          }
        }
      }
      ch.atomicAdd(ACK_OFF, 1);
      int st = flush_nb(ch, timeout, res);
      if (st) {
        res[R_STATUS] = st < 0 ? 3 : 4;
        res[R_CQ_STATUS] = (uint64_t)(int64_t)st;
        s_abort = 1;
      }
    }
    __syncthreads();
    if (s_abort) break;
  }
  if (threadIdx.x == 0) {
    res[R_ITERS] = (uint64_t)(it - 1);
    res[R_NS] = gtime() - t0;
  }
}

// P7: NIC fetch-add (+1, from the sender) and GPU red.add.sys (+addend, receiver threads) on the
// same receiver word. addend is chosen by the driver from the measured NIC byte order so the two
// counts occupy disjoint bytes. The sender flushes its last FA before setting DONE (FA completions mean
// the responder executed them).
extern "C" __global__ void __launch_bounds__(1024) p7_sender(
    PortChannelIbgdaDeviceHandle* chs, uint8_t* buf, uint64_t* res, int nfa, int batch,
    uint64_t timeout) {
  if (threadIdx.x != 0) return;
  PortChannelIbgdaDeviceHandle ch = chs[0];
  uint64_t t0 = gtime();
  int i;
  for (i = 0; i < nfa; ++i) {
    ch.atomicAdd(P7_OFF, 1);
    if ((i % batch) == batch - 1) {
      int st = flush_nb(ch, timeout, res);
      if (st) {
        res[R_STATUS] = st < 0 ? 3 : 4;
        res[R_CQ_STATUS] = (uint64_t)(int64_t)st;
        break;
      }
    }
  }
  int st = flush_nb(ch, timeout, res);
  if (st && !res[R_STATUS]) res[R_STATUS] = st < 0 ? 3 : 4;
  ch.atomicAdd(DONE_OFF, 1);
  flush_nb(ch, timeout, res);
  res[R_P7_NIC] = i;
  res[R_NS] = gtime() - t0;
}

extern "C" __global__ void __launch_bounds__(1024) p7_receiver(
    uint8_t* buf, uint64_t* res, int gpu_on, uint64_t max_gpu_per_thread, uint64_t addend,
    uint64_t timeout) {
  __shared__ unsigned long long s_n;
  uint64_t* word = (uint64_t*)(buf + P7_OFF);
  const uint64_t* done = (const uint64_t*)(buf + DONE_OFF);
  if (threadIdx.x == 0) s_n = 0;
  __syncthreads();
  uint64_t t0 = gtime();
  unsigned long long n = 0;
  while (ld_rlx_sys(done) == 0) {
    if (gpu_on && n < max_gpu_per_thread) {
      asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(word), "l"(addend) : "memory");
      n++;
    }
    if (gtime() - t0 > timeout) {
      res[R_STATUS] = 1;
      break;
    }
  }
  atomicAdd(&s_n, n);
  __syncthreads();
  if (threadIdx.x == 0) {
    fence_acq_sys();
    res[R_P7_GPU] = s_n;
    res[R_P7_WORD] = ld_acq_sys(word);
    res[R_NS] = gtime() - t0;
  }
}

// Dump the device QP fields the model cares about.
extern "C" __global__ void qp_info(PortChannelIbgdaDeviceHandle* chs, int n, uint64_t* out) {
  if (threadIdx.x != 0) return;
  for (int c = 0; c < n; ++c) {
    doca_gpu_dev_verbs_qp* qp = reinterpret_cast<doca_gpu_dev_verbs_qp*>(chs[c].qp_);
    uint64_t* o = out + c * 16;
    o[0] = (uint64_t)qp->nic_handler;
    o[1] = (uint64_t)qp->mem_type;
    o[2] = qp->sq_wqe_num;
    o[3] = (uint64_t)qp->sq_wqe_daddr;
    o[4] = (uint64_t)qp->sq_dbrec;
    o[5] = (uint64_t)qp->sq_db;
    o[6] = (uint64_t)qp->cq_sq.cqe_daddr;
    o[7] = qp->cq_sq.cqe_num;
    o[8] = (uint64_t)qp->cq_sq.mem_type;
    o[9] = (uint64_t)qp->cq_sq.dbrec;
    o[10] = qp->sq_num;
    o[11] = qp->sq_rsvd_index;
    o[12] = (uint64_t)chs[c].remoteBase_;
    o[13] = (uint64_t)chs[c].localBase_;
    o[14] = (uint64_t)chs[c].sinkAddr_;
  }
}
