# red.async.release.gpu.global ordering on GB300 (S-RED-ASYNC)

Hardware evidence, 2026-11-08. GB300 (sm_103a), CUDA 13.0, driver 580.126.20, ptxas 13.0. Intra-GPU.

ptxas 13.0 lowers `red.async.{relaxed,release}.{gpu,sys}.global.add` to a per-lane loop:
`ELECT; R2UR.BROADCAST; MEMBAR.ALL.CTA; UREDGR.E.ADD.NA[.S32|.64].STRONG.GPU|SYS [UR], UR;
DEPBAR.LE SB, 0`. The non-async `red.release.gpu` is `MEMBAR.ALL.GPU; REDG.E.ADD.STRONG.GPU`.
A CTA-scope fence does not order global stores before a gpu-scope flag on its own, so the
question was whether UREDGR provides the gpu-scope release.

Answer: yes, in every configuration tried. In the three configurations in which the controls
gave stale reads, `red.async.release.gpu` gave 0 stale reads in 1,689,600,000. In the same
configurations, `fence.acq_rel.cta; red.relaxed.gpu` (MEMBAR.ALL.CTA + REDG, the same fence
as the async form) gave 21,730,362 stale reads, and `red.relaxed.gpu` gave 461,790 in
153,600,000. Over all configurations: 2,720,000,000 reads under the async form, 0 stale.

Litmus evidence is not proof: zero stale reads bounds the rate under these conditions only.

## Harness

`redlit.cu` (build: `nvcc -O3 -arch=sm_103a -std=c++17 -o redlit redlit.cu`). Message
passing: a producer warp writes its lanes' payload words (weak `st.global`), then each lane
signals its own flag with the form under test. The consumer warp, in another CTA on another SM,
polls with `ld.acquire.gpu` and reads the payload (weak `ld.global`). Any word older than the
iteration counts as stale. The consumer acks with `st.release.gpu`. Forms: 0 red.async.release.gpu,
1 red.relaxed.gpu, 2 red.release.gpu, 3 fence.acq_rel.cta + red.relaxed.gpu.

- Mode 1 is the deterministic-accumulation shape of FA backward: a bulk copy or bulk add
  reduction, then `wait_group 0`, then `bar.sync`, then another thread signals. Even the
  controls were never stale there.
- `sweep.sh` and `sweep_stress.sh` run the configurations, and `summarize.py` aggregates them.
- Raw lines are in `results/sweep1.txt` (all forms, 8 configurations, 10 rounds) and
  `results/stress1.txt` (L2-thrash configurations, 60 rounds).

## Also found: ptxas compiles other red.async operations to an add

ptxas 13.0 and 13.1 accept `red.async.release.gpu.global.{min,max,or,and,xor,inc,dec}` and
emit `UREDGR.E.ADD`. On GB300, `100 min.s32 6` gives 106 (a CUDA test, one thread, `red.async.release.gpu.global.min.s32`). The
PTX front refuses these operations.
