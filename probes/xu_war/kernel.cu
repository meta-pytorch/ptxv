/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

extern "C" __global__ void xuwar(const float4* f, double* o){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  float4 a=f[4*i],b=f[4*i+1],c=f[4*i+2],d=f[4*i+3];
  float x[16]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w,c.x,c.y,c.z,c.w,d.x,d.y,d.z,d.w};
  double v[16];
  #pragma unroll
  for(int j=0;j<16;j++) v[j]=(double)x[j];
  double s=0;
  #pragma unroll
  for(int j=0;j<16;j++){ s = s*v[j] + v[15-j]; }
  #pragma unroll
  for(int j=0;j<16;j++) o[17*i+j]=v[j];
  o[17*i+16]=s;
}
