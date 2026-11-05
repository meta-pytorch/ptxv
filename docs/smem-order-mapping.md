# How ptxas maps PTX shared-memory ordering to SASS (sm_103a)

**Status: static evidence only, 2026-11-03.** This note asks what SASS `LDS` / `STS` / `ATOMS`
mean in the memory model. It looks only at the compiler (ptxas 13.0.88, `-arch=sm_103a`,
`nvdisasm -c -hex`). Hardware litmus tests are in `probes/smem_litmus/`, run by a separate
agent.

Reproduce with `probes/smem_order/`:

- `gen.py`: one PTX kernel per instruction, all with the same surrounding code. Output is in
  `map_O3.txt` and `map_O0.txt`.
- `gen_opt.py`: ordering and optimization probes. Output is in `opt_O3.txt` and `opt_O0.txt`.
- `bitflip.py`: flips each bit of one instruction and runs nvdisasm on the result.
- `spin/`: the `fixtures/e_spin_smem` variants, compiled.

Each output line shows the instruction, its 128-bit hex word (`hi_lo`) and its decoded control
word (stall / yield / wbar / rbar / wait; layout from `docs/sass-ctrl.md`).

## 1. Mapping table (`-O3`; `-O0` gives the same opcodes and modifiers)

### `.shared::cta`, through a shared-window address

| PTX | SASS |
|---|---|
| `ld.shared` (weak), `ld.weak.shared`, `ld.volatile.shared`, `ld.relaxed.{cta,cluster,gpu,sys}.shared` | `LDS R, [R]`, **the same 128 bits in every case**: `000ea80000000800_0000000000057984` |
| `ld.acquire.cta.shared` | `LDS` (same instruction bits) followed by **`NOP` with wait=SB(LDS)**. No CCTL, no MEMBAR |
| `ld.acquire.{cluster,gpu,sys}.shared` | `LDS`, then `CCTL.IVALL` waiting on the LDS scoreboard |
| `st.shared`, `st.volatile.shared`, `st.relaxed.{cta..sys}.shared` | `STS [R], R`, **the same bits in every case**: `000fe20000000800_0000000300007388` |
| `st.release.cta.shared` | `MEMBAR.ALL.CTA` (rbar=SB0), then `STS` waiting on SB0 |
| `st.release.{cluster,gpu}.shared` | `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR; STS` |
| `st.release.sys.shared` | `MEMBAR.ALL.SYS; ERRBAR; CGAERRBAR; STS` |
| `atom.{relaxed.*,default}.shared.add` | `ATOMS.ADD`, the same bits at every scope (ATOMS has no scope field) |
| `atom.acq_rel.cta.shared` | `MEMBAR.ALL.CTA; ATOMS` (the acquire half is the scoreboard wait on the result) |
| `atom.acq_rel.{cluster..sys}.shared` | `MEMBAR.ALL.{GPU,SYS}; ERRBAR; CGAERRBAR; ATOMS; CCTL.IVALL` |
| `red.relaxed.*.shared` / `red.release.*` | `ATOMS.ADD RZ`, with the same MEMBAR prefixes as `st.release` |
| `ld/st.mmio.*.shared` | ptxas error: `.mmio` cannot be applied to `.shared::cta` or `.shared::cluster` |

### `.shared::cluster` and global: the strength is encoded in the instruction

| PTX | `.shared::cluster` (generic `LD`/`ST`/`ATOM`) | global |
|---|---|---|
| weak | `LD.E` (hi `..100900`) | `LDG.E` |
| volatile, relaxed.sys | `LD.E.STRONG.SYS` (`..114900`) | `LDG.E.STRONG.SYS` |
| relaxed.cta | `LD.E.STRONG.SM` (`..10a900`) | `LDG.E.STRONG.SM` |
| relaxed.cluster / relaxed.gpu | `LD.E.STRONG.GPU` (`..10e900`) | `LDG.E.STRONG.GPU` |
| acquire.cta | `.STRONG.SM` + **`NOP` waiting on its scoreboard** (no CCTL) | same |
| acquire.{cluster,gpu,sys} | `.STRONG.<GPU/SYS>` + `CCTL.IVALL` | same |
| release.cta | `MEMBAR.ALL.CTA` + `.STRONG.SM` | same |
| atom default | `ATOM.E.ADD.STRONG.GPU` | `ATOMG.E.ADD.STRONG.GPU` |
| mmio | error | `LDG.E.MMIO.SYS` / `STG.E.MMIO.SYS` |

### Fences (placed between `STS` and `LDS`)

| PTX | SASS |
|---|---|
| `fence.sc.cta`, `membar.cta` | `MEMBAR.SC.CTA` (rbar=SB2); the next LDS waits on SB2 |
| `fence.acq_rel.cta` | `MEMBAR.ALL.CTA`, then the next op waits on all scoreboards (`NOP wait=111111` in `spin/relfence`) |
| `fence.{sc,acq_rel}.{cluster,gpu}`, `membar.gl` | `MEMBAR.{SC,ALL}.GPU; ERRBAR; CGAERRBAR; CCTL.IVALL` |
| `fence.{sc,acq_rel}.sys` | the same, with `.SYS` |
| `fence.acquire.cluster` | `CCTL.IVALL` (waits on all scoreboards) |
| `fence.release.cluster` | `MEMBAR.ALL.GPU; ERRBAR; CGAERRBAR` |
| `fence.proxy.async` | `MEMBAR.ALL.GPU; FENCE.VIEW.ASYNC.S` |
| `fence.proxy.async.shared::cta` | `MEMBAR.ALL.CTA; FENCE.VIEW.ASYNC.S` |
| `fence.mbarrier_init.release.cluster`, `bar.warp.sync` | nothing |
| `mbarrier.arrive` vs `.relaxed.cta` | the same `SYNCS.ARRIVE.TRANS64.A1T0` bits |
| `mbarrier.try_wait` vs `.relaxed.cta` | the same `SYNCS.PHASECHK.TRANS64.TRYWAIT` bits |

## 2. Encoding diffs: no hidden strong bit on LDS / STS / ATOMS

- **The 128 bits are identical.** For each of these groups, every listed variant compiles to
  exactly the same instruction word, including the control bits:
  - `ld.shared`, `ld.weak`, `ld.volatile` and `ld.relaxed.{cta,cluster,gpu,sys}` (all LDS);
  - the same six `st` variants (all STS);
  - `atom.relaxed` at every scope (all ATOMS).

  There is nothing that nvdisasm fails to print, because nothing differs.
- `ld.acquire.cta.shared` differs only in the control word: wbar SB0 instead of SB2, stall 2,
  plus the trailing `NOP` with wait=SB0. The bits for opcode, modifiers and operands are the
  same.
- `bitflip.py` flipped bits 72–104 of each instruction:
  - **Generic `LD`:** bits 77–80 form a sem/scope field (`CONSTANT.PRIVATE`, `CONSTANT.CTA`,
    `CONSTANT`, `MMIO.GPU`; ptxas sets `STRONG.SM/GPU/SYS` here). Bits 84–86 are cache hints
    (`EF`, `LU`, `NA`).
  - **`LDS`, `STS`, `ATOMS`:** flipping bits 76, 77, 80–90 or 92–104 changes nothing in the
    nvdisasm output. These opcodes only have fields for size, `.X4`/`.X8` scaling,
    `+UR`, and the ATOMS operation. **The shared opcodes have no field for strength or
    scope.** (An undocumented field is still possible, but ptxas never sets one: the bits are
    zero in every variant.)
- PTX ISA §8.3 explains why scope changes nothing: "the synchronizing effect of
  `ld.relaxed.shared.sys` is identical to that of `ld.relaxed.shared.cluster`, since no thread
  outside the same cluster can execute an operation that accesses the same memory location."
  ptxas goes one step further and uses one LDS for every scope. CTA-local shared memory is
  only reachable from the SM, and the SM is the CTA's home.

## 3. ptxas treats weak and strong differently when optimizing, then emits the same instruction

Results from `opt_O3.txt`, count of LDS/STS emitted:

| probe (`.shared::cta`) | weak | relaxed.cta / relaxed.gpu | acquire.cta / release.cta | volatile |
|---|---|---|---|---|
| P1: two loads from the same address | **merged into 1** | 2 | 2 | 2 |
| P5: store, then load from the same address | **forwarded, no LDS** | STS + LDS | STS + LDS | STS + LDS |
| P4: spin `while (ld [f] == 0)` | **loop deleted** (one LDS, no branch) | real loop, LDS every iteration | real loop + NOP wait | real loop |
| `spin/sp3` (e_spin_smem with a weak poll) | **poll deleted entirely**; consumer reads data at once | – | – | – |
| P2: two stores to the same address | merged into 1 | **merged into 1** (allowed for relaxed) | 2 (each has its own MEMBAR) | 2 |
| P3: adjacent aligned 32-bit pair | `LDS.64` / `STS.64` | **`LDS.64` / `STS.64`** | **`LDS.64`; `MEMBAR` + `STS.64`** | 2 scalar ops |

At `-O0` nothing is merged or forwarded. Every variant then emits its accesses one for one,
with the same opcodes.

**Conclusion of §3: "weak vs strong" for shared memory exists only in the compiler.** Weak
accesses may be deleted, merged, forwarded or hoisted. Strong ones keep their count, except
for merges that are legal for relaxed accesses. The hardware op that remains is the same
`LDS`/`STS` either way. So at the SASS level, every LDS/STS that exists is an access that
really executes. What PTX means by "weak" (the compiler may assume no concurrent writer) has
already been applied, and the binary records nothing of it.

### Ordering idioms visible in the binary (these matter for the model)

1. **CTA-scope acquire is a scoreboard wait.** `ld.acquire.cta` (shared, and global
   `LDG.STRONG.SM`) is the load followed by an instruction that waits for the load's write
   scoreboard before anything later issues. There is no CCTL. In-order issue then keeps every
   later access after the load completes.
   - In the poll loops (`spin/good` against `spin/sp1`/`sp2`), the acquire NOP is the *only*
     difference. In the relaxed version, the `ISETP` that consumes the flag waits on the same
     scoreboard one instruction later.
   - So in any LDS poll that branches on its value, later accesses already issue after the
     flag load completes.
2. **CTA-scope release is `MEMBAR.ALL.CTA` + scoreboard.** The MEMBAR sets a read
   scoreboard, and the release store waits on it. ptxv already models this as a release fence
   at CTA scope.
3. **`fence.acq_rel.cta` = `MEMBAR.ALL.CTA` + wait on all scoreboards.** Again there is no
   CCTL: at CTA scope the acquire half is just "wait for outstanding loads".
4. **ptxas relies on LDS completing in order.** In `O1` and `P1`, the first LDS gets *no*
   write scoreboard (wbar=7). Its result is read after a wait on the *second* LDS's
   scoreboard. ptxv's scoreboard check already models this as the in-order memory queue
   (`scoreboard.rs`: "the queue is in order").
5. **ptxas merges a weak data store and a release flag store into one `STS.64` after the
   MEMBAR.** In `fixtures/e_spin_smem`, the PTX is:

   ```
   st.shared [data], 42; st.release.cta.shared [flag], 1
   ```

   and the producer compiles to:

   ```
   MEMBAR.ALL.CTA; STS.64 [flag], {1, 42}
   ```

   The data write is now *after* the release fence. PTX §8.2.4 models a vector access as
   scalar accesses "executed in an unspecified order". If that rule applied to `STS.64`, this
   transformation would be wrong. ptxas therefore relies on the hardware making the two
   halves of an aligned `STS.64` visible together, to an observer in the same CTA.

## 4. What the documentation says (PTX ISA, CUDA 13.0, docs.nvidia.com/cuda/parallel-thread-execution)

- §8.4, Operation types:
  - "strong operation: A memory fence operation, or a memory operation with a .relaxed,
    .acquire, .release, .acq_rel, .volatile, or .mmio qualifier."
  - "weak operation: An ld or st instruction with a .weak qualifier."
- §9.7.10.8 `ld`: "The .weak qualifier indicates a memory instruction with no synchronization.
  The effects of this instruction become visible to other threads only when synchronization
  is established by other means. ... When none of these is specified, the .weak qualifier is
  assumed by default." `.relaxed` and `.acquire` "May be used with .global, .shared spaces".
- §8.4.2: volatile is "equivalent to a relaxed memory operation with system-scope" with extra
  constraints: the instruction count is preserved, and there is no reordering among volatile
  instructions. This matches P1–P3: volatile is never merged.
- §8.7: morally strong = "related in program order ... or each operation is strong and
  specifies a scope that includes the thread executing the other operation", same proxy, and
  complete overlap.
- §8.7.1: "Two conflicting memory operations are said to be in a data-race if they are not
  related in causality order and they are not morally strong." Uniform-size races are not UB
  in PTX (§8.7.2: the axioms still describe them). Single-copy atomicity (§8.10.3) and
  coherence-based axioms are guaranteed only for morally strong pairs.
- §8.3: scope on `.shared` is capped by who can reach the location (quoted in §2).
- §8.2.4: vector accesses are "a set of equivalent memory operations with a scalar data type,
  executed in an unspecified order". ptxas output relies on more than this (§3 item 5).
- The PTX model is a contract between the source and ptxas. The ISA says nothing about SASS,
  and NVIDIA publishes no SASS-level memory model.
  - Lustig, Sahasrabuddhe, Giroux, "A Formal Analysis of the NVIDIA PTX Memory Consistency
    Model", ASPLOS 2019 (github.com/NVlabs/ptxmemorymodel), formalizes the PTX level only. It
    proves the mapping from C++ scoped atomics *to PTX* sound.
  - The PTX-to-SASS mapping is not public. The tables above are the de facto mapping.
- Earlier hardware: Alglave et al., "GPU Concurrency: Weak Behaviours and Programming
  Assumptions", ASPLOS 2015, observed weak behaviours on Fermi/Kepler-era NVIDIA GPUs,
  including intra-CTA message passing through shared memory without fences, as I recall. I
  could not fetch the paper from this host to check the exact chips and tests, so treat this
  as unverified. Even if true, it concerns shared memory *without* a MEMBAR, and does not
  contradict the idioms above.

## 5. Conclusion

**Recommended: option (a), with one refinement.** Treat SASS `LDS`/`STS`/`ATOMS` (and
`LDSM`/`STSM`) on CTA-local shared memory as **relaxed strong accesses at `.cta` scope**. Two of
them are morally strong when both threads are in the same CTA.

Why:

- The binary cannot tell weak from strong: weak, relaxed and volatile compile to the same 128
  bits at every scope.
- The opcodes have no field that could carry strength.
- ptxas uses this same instruction to implement every *strong* shared access the PTX model
  defines, including `ld.relaxed.sys.shared`. Its correctness for correct PTX therefore
  requires LDS/STS to behave at least as relaxed.cta strong (for shared memory, scope above
  CTA adds nothing).

Keeping them weak (option c) is not wrong for the PTX it came from. But it reports a race for
every correct PTX handoff through a shared flag, because the evidence of strength is gone
after compilation. Option (b) (only some encodings are strong) has no support: there is no
distinguishing encoding.

Refinements the SASS front-end needs so that acquire/release pairs come out clean:

1. **CTA-scope acquire from the control word.** An `LDS` (or `LDG.STRONG.SM`) acts as an
   acquire at `.cta` scope if its write scoreboard is waited on before the next memory
   instruction issues. This is the `ld.acquire.cta` idiom, and it holds automatically when a
   poll branches on the value. Likewise, `MEMBAR.ALL.CTA` followed by a full scoreboard wait
   is `fence.acq_rel.cta`.
   - Today ptxv treats `CCTL.IVALL` as the only acquire. That is correct only for scope ≥
     cluster.
   - Consequence for global: S-SCOPE already treats `LDG.STRONG.SM` as relaxed strong. With
     no CCTL, a SASS `ld.acquire.cta.global` still has no acquire, so a CTA-scope global
     handoff would also be reported as a race.
2. **Vector shared accesses are single events.** Model an aligned `STS.64`/`STS.128` (and
   `LDS.64`/`.128`) as one access to all its bytes, not as per-element accesses in arbitrary
   order. Without this, `e_spin_smem/good`'s merged `MEMBAR; STS.64 {flag,data}` still shows a
   data race, even after (a).
3. **Generic `LD`/`ST`/`ATOM`** that resolve to shared memory keep their explicit
   `.STRONG.<scope>` field. Two options:
   - follow the encoding, as S-SCOPE does now; or
   - for consistency with (a), treat any access that lands in CTA-local shared memory as
     relaxed.cta.

   Remote DSMEM (`shared::cluster`) accesses compile to generic `LD.E.STRONG.GPU`, so their
   strength is explicit and S-SCOPE handles them as it does today.
4. **Global `LDG`/`STG` stay as they are.** For global memory the compiler *does* emit
   different instructions for weak and strong (`LDG.E` vs `LDG.E.STRONG.*`), and B20 showed a
   weak or under-scoped global poll really going stale through L1 on GB300. Keep S-SCOPE
   unchanged there, apart from item 1.

**Confidence:**

- High that there is no hidden encoding difference and that the strength is compiler-only.
- Medium-high that (a) is the right model: it is the minimum hardware behaviour that ptxas's
  mapping of correct PTX depends on.
- Medium on refinements 1 and 2 until litmus tests confirm them.

## 6. Suggested litmus shapes for `probes/smem_litmus/` (intra-CTA, two warps)

1. **MP-LDS-nodep.** Writer: `STS data; MEMBAR.ALL.CTA; STS flag`. Reader: `LDS flag; LDS
   data` with no wait in between (write the SASS by hand, or use relaxed loads with no
   dependency). Can the reader see flag=1, data=0? This tests in-order LDS completion across
   warps; the expected answer is never.
2. **MP-noMEMBAR.** As 1, but the writer does `STS data; STS flag` with no MEMBAR (PTX: two
   `st.relaxed.cta`). This shows whether the MEMBAR is needed at all for shared memory, and
   so whether `STS` can be treated as release-ordered in hardware. The model should still
   require the MEMBAR.
3. **MP-STS64.** The writer's only store is `STS.64 {flag,data}` (the e_spin_smem shape). The
   reader polls flag with LDS, then reads data with a separate LDS. Is data ever stale? This
   tests refinement 2.
4. **MP-LDS-then-LDG.** Shared flag, global payload, at CTA scope: writer `STG; MEMBAR.ALL.CTA;
   STS flag`, reader `LDS flag (+NOP wait); LDG data`. Does a CTA-scope acquire built from a
   scoreboard wait order a later global load?
5. **SB / CoRR on shared** (two warps): checks coherence and single-copy atomicity for
   relaxed.cta. The expected answer is never.
