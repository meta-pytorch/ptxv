# Row sums of squares in f32, one thread per row, sequential over K (an FFMA chain per row).
# `ptxv num` once reported max_err = 0 on this kernel although every FFMA rounds: the FFMA
# results were untracked and the output was compared with its own simulated value.
import torch
import triton
import triton.language as tl


@triton.jit
def num_sumsq_kernel(x_ptr, y_ptr, K: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)
    for k in range(K):
        v = tl.load(x_ptr + rows * K + k)
        acc += v * v
    tl.store(y_ptr + rows, acc)


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(256, 64, generator=g).cuda()}


def run(x):
    y = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    num_sumsq_kernel[(x.shape[0] // 64,)](x, y, K=x.shape[1], R=64)
    return y
