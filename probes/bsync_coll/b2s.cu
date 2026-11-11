// Probe B2, per warp (stress): o[0] lane 0 done with its work, o[1] lane 0 past the join's
// BSYNC, o[2] lane 16 past its __syncwarp, o[3] lane 16 arriving at its __syncwarp.
extern "C" __global__ void b2s(const int* c, unsigned long long* t, int work) {
  int lane = threadIdx.x & 31;
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  unsigned long long* o = t + 4 * w;
  float v = c[lane];
  int n = work + (w % 7) * (work / 3 + 1);
  if (lane < 16) {
    for (int k = 0; k < n; ++k) v = __sinf(v) + 1.0f;
    unsigned long long a; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(a));
    if (lane == 0) o[0] = a;
  } else {
    unsigned long long p; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p));
    if (lane == 16) o[3] = p;
    __syncwarp(0xffffffffu);
    unsigned long long a; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(a));
    if (lane == 16) o[2] = a;
  }
  unsigned long long b; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(b));
  if (lane == 0) o[1] = b;
  __syncwarp(0xffffffffu);
  if (v == 12345.f) t[0] = 1;
}
