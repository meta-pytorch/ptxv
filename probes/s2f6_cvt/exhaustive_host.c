// Host side of probes/s2f6_cvt/exhaustive (CUDA driver API): builds the model table
// M[s << 8 | c] = bf16(RNE, subnormals kept, satfinite)(int8(c) / 64 * 2^(s - 127)), s = 0xff ->
// 0x7fff, runs the kernel over all 2^32 pairs and prints the mismatch count.
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do { CUresult e = (x); if (e) { const char* m; cuGetErrorString(e, &m); \
  fprintf(stderr, "%s: %s\n", #x, m); exit(2); } } while (0)

static uint16_t bf16_rne_satfinite(double v) {
  if (v == 0) return 0;
  uint16_t sign = v < 0 ? 0x8000 : 0;
  double a = fabs(v);
  int e;
  frexp(a, &e);  // a = f * 2^e, f in [0.5, 1)
  int E = e - 1 < -126 ? -126 : e - 1;
  double q = ldexp(a, 7 - E);  // significand units of 2^(E - 7), exact
  double n = floor(q), r = q - n;
  if (r > 0.5 || (r == 0.5 && fmod(n, 2) == 1)) n += 1;
  if (n == 0) return sign;
  if (n >= 256) { E += 1; n /= 2; }
  if (E > 127) return sign | 0x7f7f;
  if (n < 128) return sign | (uint16_t)n;
  return sign | (uint16_t)(((E + 127) << 7) | ((int)n - 128));
}

int main(int argc, char** argv) {
  static uint16_t M[65536];
  for (int s = 0; s < 256; s++)
    for (int c = 0; c < 256; c++)
      M[s << 8 | c] = s == 255 ? 0x7fff : bf16_rne_satfinite(ldexp((double)(int8_t)c, s - 133));
  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CK(cuCtxCreate(&ctx, NULL, 0, dev));
  CUmodule mod; CK(cuModuleLoad(&mod, argc > 1 ? argv[1] : "exhaustive.cubin"));
  CUfunction f; CK(cuModuleGetFunction(&f, mod, "full"));
  CUdeviceptr dm, db, df;
  CK(cuMemAlloc(&dm, sizeof M)); CK(cuMemAlloc(&db, 8)); CK(cuMemAlloc(&df, 20));
  CK(cuMemcpyHtoD(dm, M, sizeof M)); CK(cuMemsetD32(db, 0, 2)); CK(cuMemsetD32(df, 0xffffffff, 5));
  void* args[] = {&dm, &db, &df};
  CK(cuLaunchKernel(f, 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
  CK(cuCtxSynchronize());
  unsigned long long nb; uint32_t fi[5];
  CK(cuMemcpyDtoH(&nb, db, 8)); CK(cuMemcpyDtoH(fi, df, 20));
  printf("pairs %llu mismatches %llu", 1ull << 32, nb);
  if (nb) printf(" first data=%#x scale=%#x gpu=%#x model=%#x", fi[1], fi[2], fi[3], fi[4]);
  printf("\n");
  return nb != 0;
}
