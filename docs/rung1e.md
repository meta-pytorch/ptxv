# Rung 1e: a pair of kernels with programmatic dependent launch (PDL)

**Status: done (2026-10-30).**
- The model now runs **several kernels of one stream as one machine**.
- The good pair is explored exhaustively on both front-ends and is clean. The PTX model, the
  SASS model and GB300 agree bitwise (1000/1000 runs).
- B18 and B55 are races on both front-ends.
- On GB300, B18 is wrong in every run once a 20 µs delay is put before the primary's epilogue
  store. The same code launched without PDL is clean in the model and stays correct under the
  same delay.

## What the probe established (E-PDL)

`probes/pdl/pdl.cu` is CUDA C++, because it needs `cudaLaunchKernelEx` with the
programmatic-stream-serialization attribute. Each experiment runs a primary that writes a flag
at the very end, after a 2 ms spin, and a dependent that reads the flag before and after
`griddepcontrol.wait`. 20 runs each:

| experiment | dependent ran early | flag visible after `wait` |
|---|---|---|
| E1 every thread triggers | 19/20 | 20/20 |
| E2 no trigger | 0/20 | 20/20 |
| E3 only thread 0 of the CTA triggers | 20/20 | 20/20 |
| E4 4 CTAs, CTA 0 never triggers | 0/20 | 20/20 |
| E5 data written before the trigger | 20/20 (and saw the data 20/20) | 20/20 |
| E6 trigger, launched without the PDL attribute | 0/20 | 20/20 |

The model (E-PDL):
- A later grid's CTAs start **unlaunched**.
- **Without PDL** they may launch once the previous grid has *completed*: every thread exited
  and every async operation it issued is done. Everything that grid did then happens-before
  them.
- **With PDL** they may launch once every CTA of the previous grid has triggered
  (`griddepcontrol.launch_dependents`; one thread suffices, E3) or has exited entirely (E2,
  E4). They get **no** ordering from the trigger.
- `griddepcontrol.wait` blocks until the previous grid has completed, then acquires everything
  it did. In the first grid it is a no-op.

E5 shows that writes made before the trigger were visible early on GB300. The model still
treats them as unordered, as the PTX ISA does: `wait` is the only visibility guarantee. SASS:
`launch_dependents` = `PREEXIT`, `wait` = `ACQBULK`.

## Multi-kernel launches in the model

A case lists further kernels under `[[launches]]` (entry, grid, block, params, `pdl`). They
come from the same PTX file and use the same buffers. Both front-ends parse every entry, and
`Kernel::concat` joins them into one code stream: branch targets, labels and static findings
are relocated, and each kernel keeps its own parameters, shared-memory size and register
names (`Kernel::parts`). Execution needed only per-CTA lookups:
- `Launch` is now a list of `Grid`s (dims, cluster shape, parameter block, entry pc, first
  global CTA id, `pdl`);
- special registers, constant bank 0, parameter space, tensor maps, cluster ids, the
  shared-memory window and SASS `RET` targets resolve through the thread's grid;
- thread names read `k1.cta0.t5`;
- a gated CTA that never launches is reported as such in deadlock traces.

DPOR has a new footprint space 9. A trigger or exit by a thread writes its CTA's gate key,
and an async completion writes its issuer CTA's key. A gated launch and `griddepcontrol.wait`
read the keys of every CTA of the previous grid. They are recorded only when the CTA's grid has
a successor, so single-kernel exploration is unchanged (the full suite gives identical
results).

## The kernels

`fixtures/r1e_pdl/kernel.ptx` (2 CTAs × 32 threads per kernel):

```
produce : x = in[i] ("mainloop") ; griddepcontrol.launch_dependents ; buf[i] = 3x + 1 ("epilogue")
consume : (prologue: addresses) ; griddepcontrol.wait ; in[i] = 0 (reset for the next step) ;
          out[i] = 2 * buf[i] + 100                                   [launched with PDL]
```

The trigger sits where CUTLASS calls `launch_dependent_grids()`: after the mainloop, before
the epilogue. The dependent's prologue therefore overlaps the primary's epilogue.

## Verdicts

| variant | bug | PTX / SASS model (PDL) | GB300 (1000 runs) | without PDL (`r1e_serial`) |
|---|---|---|---|---|
| good | | clean, exhaustive (64 states) | bitwise = models | clean |
| B18 | consume has no `griddepcontrol.wait` | race: produce's epilogue store vs consume's load (and in[] read vs reset) | correct 1000/1000 and at 512 CTAs (`tools/r1e_scale.sh`); **wrong 200/200** with `--delay 38:20000` (20 µs before the epilogue store: out = 100, buf read as 0) | clean; correct under the same delay |
| B55 | consume resets in[] before the wait | race: produce's mainloop load vs consume's store (write-after-read) | correct, also with a delay before the load | clean |

**B18 passes on hardware without help.** The dependent's launch latency after the trigger is
longer than this epilogue (one store per thread), even at 512 CTAs. A slower epilogue is
enough to break it, which the delay demonstrates. That is the pattern listed in BUGS.md:
"passes when the primary is fast".

**B55 is contract-level, like B14.** In the produce cubin, the `IMAD` that consumes the
mainloop load comes before `PREEXIT`. The load is complete before any dependent CTA can launch,
so this binary cannot show the hazard. A primary that triggers with a load still in flight
could. The PTX contract gives no ordering before `wait` either way.

**The control case is the point of E6.** `fixtures/r1e_serial` launches the same two kernels
without the attribute. The model gives the dependent the primary's whole history at launch, so
both mutants are clean, and on GB300 B18 stays correct under the delay that breaks it with PDL.
The same code is correct or racy depending only on a launch attribute. The verifier has to see
the launch, not just the kernel, which is why the capture format records launch attributes
(design.md §2).

## Infrastructure changes

- `ptxv hw` launches every kernel of a case with `cuLaunchKernelEx` (attribute
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` when `pdl = true`). It now resets
  *every* buffer before each run, not just the outputs, because consume writes its input.
- SASS: `LDG` (32/64/128-bit); `ACQBULK`, `PREEXIT`.
- The model's single-schedule reference run is too slow for 512-CTA grids (it took more than
  10 minutes), so the scaled hardware check uses the closed form instead
  (`tools/r1e_scale.sh`).

## Reproduce

```
nvcc -arch=sm_103a -O2 -o /tmp/pdl probes/pdl/pdl.cu && CUDA_VISIBLE_DEVICES=<free> /tmp/pdl
ptxv check fixtures/r1e_pdl ; ptxv check fixtures/r1e_serial
ptxv hw fixtures/r1e_pdl --runs 1000
ptxv hw fixtures/r1e_pdl --variant B18 --runs 200 --delay 38:20000     # witness
ptxv hw fixtures/r1e_serial --variant B18 --runs 200 --delay 38:20000  # control: correct
tools/r1e_scale.sh 5 512
cargo test --release --test fixtures r1e
```
