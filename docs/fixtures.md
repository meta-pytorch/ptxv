# Fixtures that need regeneration

Some fixtures run third-party kernels (cuBLAS / cuBLASLt, CUTLASS, TRT-LLM-gen via flashinfer,
DeepEP, FlashAttention). Their kernel binaries are not distributed with ptxv; everything else
in those fixture directories is (case.toml with variants and expected verdicts, inputs, GPU
outputs). `fixtures/VENDOR_FIXTURES.toml` lists each one: the vendor, the missing files, the
library version it was captured with and the capture command. Tests that need a missing file
print `SKIP: vendor fixture files not present` and pass.

To regenerate, on a Blackwell (sm_100a / sm_103a) machine with the library installed and an
idle GPU:

```
CUDA_VISIBLE_DEVICES=<gpu> PTXV_PYTHON=<python with torch> python3 tools/regen_fixture.py --missing
python3 tools/regen_fixture.py --list        # what is present / missing
```

| library | version the cases were captured with | fixtures |
|---|---|---|
| torch (cuBLAS / cuBLASLt, libtorch's CUTLASS grouped GEMM, mem-efficient and flash attention) | 2.12.0a0+git808dd93, CUDA 13.0 | c_mm_bf16_*, c_smm_*, c_mm_f32_gemmsn*, c_mm2_*, r2_cublas_gemm, r3_grouped_mm, c_cve_sdpa_f32, r4_sdpa_flash_torch |
| CUTLASS CuTeDSL (the copy bundled with flashinfer) | 4.6.1 (flashinfer 0.5.3) | r2_cute_gemm, r2_cute_small, r2_cute_wrap, r3_cute_mxf4, r3_cute_mxf8, r3_cute_nvf4 |
| flashinfer + flashinfer-cubin (trtllm-gen) | 0.5.3 | r3b_trtllm_bmm, r3b_trtllm_bmm_k2048, r3b_trtllm_bmm_partial, r3b_trtllm_moe |
| DeepEP (2 GPUs) | 1.2.1 (upstream 567632d) | r4b_moe, r4c_chunk2, r4c_dwrap, r4c_nowrap (and r4b_coreader, which reads r4b_moe) |

Other versions may select different kernels, in which case a fixture's expected verdicts can
change; `ptxv check` on the regenerated case shows what the new kernel does.
