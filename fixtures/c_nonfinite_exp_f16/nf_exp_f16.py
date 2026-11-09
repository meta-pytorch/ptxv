# exp of f16 inputs computed in f32 and stored as f16: e^x > 65504 overflows the conversion to
# +inf (ptxv check --nonfinite: Inf outputs from finite inputs, produced by the cvt).
import torch
import triton
import triton.language as tl


@triton.jit
def nf_exp_f16_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i).to(tl.float32)
    tl.store(y_ptr + i, tl.exp(x).to(tl.float16))


def inputs():
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1024, generator=g) * 4 + 6).to(torch.float16)
    return {"x": x.cuda()}


def run(x):
    y = torch.empty_like(x)
    nf_exp_f16_kernel[(x.numel() // 256,)](x, y, BLOCK=256)
    return y
