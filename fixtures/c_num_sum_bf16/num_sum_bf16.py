# Row sums in f32, one thread per row, sequential over K (an FADD chain per row), stored as
# bf16: `ptxv num` reads the output per 16-bit element and tracks the final rounding.
import torch
import triton
import triton.language as tl


@triton.jit
def num_sum_bf16_kernel(x_ptr, y_ptr, K: tl.constexpr, R: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    acc = tl.zeros([R], dtype=tl.float32)
    for k in range(K):
        v = tl.load(x_ptr + rows * K + k)
        acc += v
    tl.store(y_ptr + rows, acc.to(tl.bfloat16))


def inputs():
    g = torch.Generator().manual_seed(0)
    return {"x": torch.randn(256, 64, generator=g).cuda()}


def run(x):
    y = torch.empty(x.shape[0], device=x.device, dtype=torch.bfloat16)
    num_sum_bf16_kernel[(x.shape[0] // 64,)](x, y, K=x.shape[1], R=64)
    return y
