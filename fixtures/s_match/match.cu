// match.any.sync (SASS MATCH.ANY / MATCH.ANY.U64) and cp.async.bulk.prefetch.L2 (UBLKPF.L2).
// out[tid*4 + 0]: 32-bit match of key % 5; +1: 64-bit match of (key % 3) << 32 | (key & 1);
// +2: match.any among the lanes of a divergent branch (tid % 4 != 0); +3: 0xdead (not run).
#include <cstdint>
extern "C" __global__ void matchk(const uint32_t *rnd, uint32_t *out) {
  unsigned t = threadIdx.x;
  if (t == 0)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], 256;" ::"l"(rnd) : "memory");
  uint32_t k = rnd[t];
  out[t * 4 + 0] = __match_any_sync(0xffffffffu, k % 5);
  uint64_t w = ((uint64_t)(k % 3) << 32) | (k & 1);
  out[t * 4 + 1] = __match_any_sync(0xffffffffu, w);
  if (t % 4 != 0) {
    unsigned m = __activemask();
    out[t * 4 + 2] = __match_any_sync(m, k % 2);
  } else {
    out[t * 4 + 2] = 0xdead;
  }
  out[t * 4 + 3] = 0;
}
