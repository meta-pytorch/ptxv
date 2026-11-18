# equiv claims against GB300

Repros of `ptxv equiv` claims that GB300's outputs contradict. Each probe directory holds
elementwise kernels (out[i] from element i of the inputs; grid 4 x 256), their exact inputs
(`data/`), GB300's output of each case (`<case>/expected/out.bin`, ptxas 13.0, sm_103a: one run,
then 5 plain and 5 scrubbed runs bitwise identical; the set-y cases of select_max one run, then
20 scrubbed, and mz_selsw_y also 20 plain), one equiv.toml per pair and front, and `pairs.toml`.
On 2026-11-12 every case reproduced its expected/out.bin in 5 more runs (`run.sh` on a GB300).

```bash
bash probes/equiv_claims/select_max/run.sh          # MISS 197
bash probes/equiv_claims/max_identity/run.sh        # MISS 197
bash probes/equiv_claims/fmnmx_nan/run.sh           # MISS 202
bash probes/equiv_claims/product_underflow/run.sh   # MISS 201
```

`PTXV` names the binary (default `ptxv`). On a GPU node every case is first rerun 5 times
through `ptxv hw --no-num` against its expected/out.bin (`HW_RUNS=0` skips the rerun).

`check.py` prints, per pair, how GB300's two outputs differ on each input set (zero sign, NaN or
value) and, per front, ptxv's claim: `ptxv equiv` on a copy of the probe without the captured
outputs, so the claim is the model's proof for every input. A claim is checked per output at its
level: L0 needs equal words; L0~0 equal words or +0 against -0; L1 / L2 / L3 equal values (+0 = -0,
two NaNs agree) where the kernels compute the same exact selection or rounding of their inputs
(select_max, max_identity, fmnmx_nan), else the printed per-output bound |C - R| <= k u S + A, each
kernel's own bound against the exact value, and L3's tolerance tau u S + A (product_underflow, S =
|a b c|). A claim that relies on N-FINITE covers only the outputs whose input elements are finite.
Exit status: 1 while some claim is contradicted, 2 if a run fails, 0 otherwise.

Input sets: f finite with zeros of both signs and ties; n with infinities and NaN payloads; y
(x = a + b, y = a - b) with +0 against -0 at 32 elements and inf against NaN at 32; p a > 0 (s =
sqrt.approx.ftz(a) finite); q as p at 0..511 and finite a < 0 at 512..1023 (s NaN there, an
admissible input even under N-FINITE); u a ~ 2^-80, b ~ 2^-60, c ~ 2^100, a * b subnormal at
512 elements; g (select_max) finite without zeros or ties, with no GPU output: select_max's
claims are taken there. Before the fix a select's node had FMNMX's value, so at a tie of zeros in
sets f and y, where the select returns its second operand's sign, the node does not hold the
simulated bits; since MISS 177 such an output is untracked and the pair level none, while the
model still proves L0 at every other output. max_identity: f finite with zeros of both signs
and ties; n NaN payloads at a[8j] and b[8j] (both NaN there), infinities at a[8j + 1].

| probe | GB300 | claim before the fix (both fronts) |
|---|---|---|
| `select_max` | max.f32 / min.f32 against a > b ? a : b / a < b ? a : b, the two select orders, gt against ge, gt against gtu, a clamp by FMNMX against one by selects, and the same over x = a + b, y = a - b (FSEL and SEL): 0 to 32 words differ at zero signs on finite inputs, 48 to 112 with NaN inputs | L0 PROVEN for all 9 pairs, taken on set g (18 claims contradicted); the 4 pairs over s = sqrt.approx(a) are level none |
| `max_identity` | max.f32(a, -inf) / min.f32(a, +inf) against the same op of a and a (folded by ptxas into a copy; and with a loaded twice, FMNMX), and max(max(a, -inf), b) against max(a, b) (FMNMX3): 128 of 1024 words differ where a (and b) is NaN, -inf / +inf against NaN | L0 PROVEN on both fronts for the unfolded pairs and on the PTX front for the folded ones (8 claims contradicted) |
| `fmnmx_nan` | max.f32 against max.NaN.f32: 32 of 1024 words differ with NaN inputs; over s: 512 of 512 at finite a < 0 | L0 PROVEN over the inputs (also in segment mode, with no N-FINITE reliance printed) and over s outside segment mode (6 claims contradicted); over s in segment mode, none |
| `product_underflow` | (a * b) * c against a * (b * c): 688 of 1024 words differ | L2 + L3 PROVEN, \|C - R\| <= 4.00u sum\|terms\|: 512 outputs beyond it, up to 3317x; (a * b) * c beyond its own 2u bound at 512, up to 6634x; L3 (8u) beyond at 512 (2 claims contradicted) |
