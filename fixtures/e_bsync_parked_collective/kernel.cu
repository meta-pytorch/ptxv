// E-HW-BSYNC-COLLECTIVE fixture (probe B2 with values instead of timestamps; lanes 1-15 exit
// before the store): lanes 0-15 do
// MUFU work while lanes 16-31 block at a full-warp __syncwarp in the else branch; lanes 0-15
// reach a __syncwarp only after the join, so they wait at the join's BSYNC while lanes 16-31
// wait in BRA.DIV's WARPSYNC.COLLECTIVE. GB300 passes the BSYNC, then the two collectives
// combine; lanes 16-31's second __syncwarp completes once lanes 0-15 have exited.
extern "C" __global__ void b2h(const float* c, float* out, int work) {
  int lane = threadIdx.x & 31;
  float v = c[lane];
  if (lane < 16) {
    for (int k = 0; k < work; ++k) v = __sinf(v) + 1.0f;
  } else {
    __syncwarp(0xffffffffu);
    v = v * 2.0f;
  }
  __syncwarp(0xffffffffu);
  if (lane == 0 || lane >= 16) out[lane] = v;
}
