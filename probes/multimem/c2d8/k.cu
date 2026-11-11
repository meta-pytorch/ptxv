// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

extern "C" __global__ void k(unsigned *out, unsigned v) { out[threadIdx.x] = v; }
