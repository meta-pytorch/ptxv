# IBGDA ordering probes: results

Run 2026-11-06 on two GB300 nodes of a Slurm cluster (one GPU per node, GPU 0: rank 0
sender, rank 1 receiver), `driver.py` driving `ibgda_litmus.cu`, one srun task per node.
Site-specific run scripts and the raw records are kept outside the public tree.

## Environment

- NIC: ConnectX-8 (vendor_part_id 4131), firmware 40.48.1000, link layer Ethernet (RoCE v2).
  HCA chosen by mscclpp for GPU 0: `CppTransport.IB0`. GIDs 0-3 are zero; index 7 is the
  global RoCE v2 GID, so every run sets `MSCCLPP_IB_GID_INDEX=7` (the default 3 fails the
  INIT->RTR transition with DOCA error 21).
- The two nodes were in one NVL72 domain; the probes use only the NIC path between them.
- QPs: DOCA GPUNetIO via mscclpp `PortChannelIbgda`, `nic_handler` 2 (GPU_SM_DB: GPU rings the
  doorbell and writes the doorbell record), SQ and CQ in device memory (1024 entries each),
  doorbell register in host-mapped MMIO, IBTA ordering (`MSCCLPP_IB_OOO_DP` unset).
- Kernel: `ibgda_litmus.cu`, nvcc (CUDA 13.0) for sm_103a against the conda env's DOCA
  headers (`comms.mscclpp.utils.doca_include_dir()`).
- Every iteration writes fresh tags (config, iteration, word); "stale" counts words holding an
  older iteration's tag or zero. N = iterations per configuration.

## Results

| probe | configuration | N | outcome |
|---|---|---|---|
| P0 | 8 fetch-adds of 1 on a zeroed word | 8 | token reads 1, 2, 3, 4 as little-endian u64; fetched values 0, 1, 2, 3 little-endian: the NIC fetch-add treats memory as little-endian |
| P1 | DOCA put (8 B / 4 KiB / 1 MiB) + fetch-add signal, same QP, no fence bit; receiver `ld.acquire.sys` token, relaxed.sys data | 1e5 each | stale not observed |
| P1 | same, receiver relaxed token + `fence.acquire.sys` | 1e5 each | stale not observed |
| P1 | same, receiver acquire token + weak data loads | 1e5 each | stale not observed |
| P1 | same, receiver relaxed token, no fence, weak data loads | 1e5 each | stale not observed |
| P1 | put with skipped doorbell + signal (one doorbell for both) | 1e5 each | stale not observed |
| P3 | custom write + fetch-add WQEs, one doorbell; fence between WQE stores and doorbell: `fence.sc.sys`, `fence.release.sys`, `.gpu`, `.cta`, none; payload stores followed by `fence.sc.sys`; 8 B and 1 MiB | 1e5 each (10 configs) | no stale data, no error CQE, no timeout |
| P3 | same with the payload ordered only by `bar.sync` (no fence before posting) | 1e5 each (10 configs) | no stale data, no error CQE, no timeout |
| P5 | put on QP A, poll A's CQE, signal on QP B; 8 B / 4 KiB / 1 MiB | 1e5 each | stale not observed |
| P6a | put, overwrite the source without polling the CQ, signal (same QP); 4 KiB / 1 MiB | 1e5 each | receiver got overwritten bytes in 1e5 / 1e5 iterations (4 KiB: 51,200,000 words; 1 MiB: 13,107,199,392 words) |
| P6b | put, poll the CQE, overwrite the source, signal | 1e5 each | overwritten bytes not observed |
| P2 | put on QP A, signal on QP B, no flush of A; 8 B / 4 KiB | 1e5 each | stale not observed |
| P2 | same, 1 MiB | 1e5 | stale data after the signal in 1e5 / 1e5 iterations (6,033,808,296 stale words) |
| P4 | large write then 8 B flag write, same QP (two doorbells / one doorbell); receiver polls the flag (`ld.acquire.sys`); 4 KiB / 1 MiB | 1e5 each | stale not observed |
| P4b | same with a 4 B flag write (mscclpp's signal token size), run 2026-11-06 on a second node pair | 1e5 each | stale not observed |
| P7 | 1e5 NIC fetch-adds (+1, low half) while 1024 receiver threads issue `red.relaxed.sys.add.u64` (+2^32, high half) on the same word | 1 run | low half 100000 (every NIC add landed); high half 11,833,344 of 16,777,216 GPU adds (4,943,872 GPU atomics lost) |
| P7 | control: NIC fetch-adds only | 1 run | word = 100000 |

## What the model takes from this (docs/nic.md)

- S-NIC-SIGNAL (P1), S-NIC-CQE (P5, P6b): consistent with the model; no violation observed.
- S-NIC-ORDER (P2): QPs are unordered; observed for 1 MiB writes.
- S-NIC-CQE send-buffer reuse (P6a): reusing the source before the completion is observed
  changes what the peer receives; the model reports it (fixture N4).
- S-NIC-WORD (P2, P6a): no torn 8-byte word among 1.9e10 words read while a write landed.
- S-NIC-ATOMIC (P0, P7): little-endian values; NIC atomics are not atomic with GPU atomics on
  the same word.
- S-NIC-DB (P3): no failure observed without a fence, or with a CTA-scope fence, in 1e5
  iterations per configuration. The model still requires a release of GPU scope or wider
  (DOCA's choice, and what the PTX memory model needs for an outside observer); verdicts on
  fence-less doorbells are possible false alarms on this hardware.
- S-NIC-WRITE / S-NIC-FLAG (P4, P4b): placement order between writes of one QP was not violated
  in 1e5 iterations per configuration. The model orders only a lone aligned 4- or 8-byte write
  (a signal token) after the QP's earlier WQEs; other write data stays unordered (possible
  false alarms for polling inside bulk data).
