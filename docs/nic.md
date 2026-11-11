# GPU-initiated RDMA (IBGDA)

A kernel that talks to another node posts mlx5 work-queue entries (WQEs) into a send-queue
ring in GPU memory, rings the queue pair's doorbell with an MMIO store, and the NIC of its
node executes them: RDMA writes and atomics into the peer GPU's memory, then a completion
entry (CQE) in GPU memory that the kernel polls. ptxv models this with a NIC actor per queue
pair (`crates/sim/src/nic.rs`), so internode kernels (DOCA GPUNetIO verbs, the mscclpp
`PortChannelIbgda` handles that DeepEP ports use) get verdicts.

## Case description

```toml
domains = [0, 1]          # NVLink domain of each GPU (index = device)

[[nic_qps]]
name = "qp0"
device = 0                # GPU that posts; its node's NIC executes
peer = 1                  # GPU whose memory the remote addresses name
qpn = 0x42                # QP number in the WQE control segment and the doorbell
sq = "sq"                 # buffer holding the send ring (64-byte WQE blocks)
sq_offset = 0
sq_wqes = 2               # ring size (power of two)
cq = "cq"                 # buffer holding the send CQ (64-byte CQEs)
cq_cqes = 2
db = "uar"                # buffer standing for the doorbell register
db_offset = 0
sq_ci = 0                 # WQE counter the NIC starts at
cq_pi = 0                 # CQE counter the NIC starts at
# ordering = "ibta"       # default; anything else is unmodeled

[[nic_mrs]]               # memory registrations the NIC checks keys against
key = 0x200
buffer = "dst"            # on the buffer's GPU; optional offset / len
```

- **Domains.** A load, store or atomic from one domain into another domain's buffer is a
  fault (S-NIC-DOMAIN). Without `domains` every GPU is in one domain.
- **Capture.** `tools/capture_dist.py` writes these entries when the captured file defines
  `nic(inputs)` (one dict per `PortChannelIbgdaDeviceHandle`: peer rank, device address,
  size) and `DOMAINS`. It reads each handle's DOCA QP struct (`doca_gpu_dev_verbs_qp`) for
  the rings, the doorbell record and the counters, makes the doorbell register a synthetic
  8-byte buffer, points the handle's remote base / token at the peer's buffers, and emits
  one `nic_mrs` entry per key and buffer an address of the handle falls in (the whole
  buffer: registration bounds are not recorded). The QP must be quiescent at the snapshot
  (posted = reserved = polled); otherwise a warning. `fb/nic/deep_ep.py` is the DeepEP
  hybrid driver with these hooks.

## Execution

- **Doorbell.** A strong `.sys` 8-byte store to a QP's doorbell (`st.mmio.relaxed.sys`, SASS
  `STG.E.64.MMIO.SYS`) carries the producer index (ctrl dword 0 = index << 8, big-endian)
  and the QP number. Each new WQE gets an actor and the ringing thread's release clock: its
  last release fence's clock if that fence is GPU scope or wider, else nothing. Other stores
  to the doorbell page (BlueFlame) and weak doorbell stores are `unmodeled`. The doorbell
  record is plain memory.
- **WQE execution.** One pending op per QP executes the oldest announced WQE, at any point
  after the doorbell (a scheduler choice). It reads the 64-byte WQE, checks index and QP
  number, then:
  - RDMA write (opcode 0x08): reads the local data segments, writes the remote bytes: each
    aligned 8-byte word (4-byte at the edges) as a relaxed `.sys` write of the NIC actor,
    unaligned edge bytes as plain writes (S-NIC-WORD); a lone naturally aligned 4- or 8-byte
    segment (a signal token) is one strong `.sys` write that releases the QP's earlier WQEs
    (S-NIC-FLAG);
  - atomic fetch-add / compare-swap (0x11, 0x12): 8-byte strong `.sys` read-modify-write of
    the remote word (little-endian value), a release of the QP's history; the old value goes
    to the local sink;
  - NOP; everything else (write with immediate, inline data, multi-block WQEs, masked
    atomics, reads) is `unmodeled`.
  - Keys and addresses are checked against `nic_mrs` and the QP's devices; a mismatch is a
    `fault` and the QP stops (no CQE is modeled for it).
- **Completion.** With CQ update requested (fm_ce_se CE = 2), the NIC writes the CQE (WQE
  counter, opcode 0, owner bit = pass parity) as a strong `.sys` release of the WQE's effects
  and the QP's earlier WQEs.
- **Findings.** `nic-wqe-unordered` / `nic-src-unordered` (WQE or payload byte not ordered
  before its doorbell, or a send buffer rewritten before the completion was observed),
  ordinary races between NIC writes and GPU accesses, `nic-atomic-mixed` (GPU and NIC atomics
  on one word, unordered), `nic-rkey` / `nic-lkey` faults.
- **Search.** Certificates (E-WELLSYNC), the synchronization and per-CTA reductions, DPOR and
  launch groups are off for cases with queue pairs: the stateful explorer decides them.

## Semantics table

| ID | rule | source | GB300 + ConnectX-8 evidence (probes/ibgda/results.md) |
|---|---|---|---|
| S-NIC-DB | WQE and payload stores must be released at GPU scope or wider before the doorbell store; the doorbell is a strong `.sys` store | DOCA `submit_db` / `ring_db` / `mark_wqes_ready`; mlx5 doorbell format | P3: no failure observed without a fence or with a `.cta` fence (20 configs x 1e5); argued, possible false alarms |
| S-NIC-ORDER | a QP's WQEs execute one at a time in posting order; QPs are unordered; IBTA ordering only | IBTA RC; DOCA dp_ordering patch (off by default) | P2: signal on QP B seen before QP A's 1 MiB write in 1e5 / 1e5 |
| S-NIC-SIGNAL | an atomic takes effect after the same QP's earlier writes are placed and releases them | DOCA `put_signal`, mscclpp `putWithSignal` (no fence bit) | P1: stale not observed, 15 configs x 1e5 |
| S-NIC-WRITE | write data synchronizes with nothing; no placement order | conservative | P4: write order kept in 1e5 (model does not rely on it beyond S-NIC-FLAG) |
| S-NIC-WORD | each aligned 8-byte word of write data is single-copy atomic (relaxed `.sys`) | PCIe qword writes | P2 / P6a: 1.9e10 words read mid-write, none torn |
| S-NIC-FLAG | a lone aligned 4- or 8-byte write is one strong `.sys` write releasing the QP's earlier WQEs | mscclpp `PortChannelIbgda` signal (token write); IBTA RC write order | P4 (8 B) and P4b (4 B): flag after a 4 KiB / 1 MiB write, stale not observed, 8 configs x 1e5 |
| S-NIC-ATOMIC | 8-byte strong RMW, little-endian value, atomic only with NIC atomics | IBV_ATOMIC_HCA | P0: little-endian; P7: 4.9M of 16.8M concurrent GPU adds lost |
| S-NIC-CQE | a CQE follows the WQE's completion (write: placed at the peer) and releases it | IBTA RC acks; DOCA `poll_cq_at` | P5: stale not observed (3 x 1e5); P6: reuse before the poll delivered new bytes 1e5 / 1e5, after the poll never |
| S-NIC-DOMAIN | another NVLink domain's memory is reachable only through the NIC | configuration | n/a |

## Fixture

`fixtures/r5a_ibgda_putsig`: rank 0 posts an RDMA write and a fetch-add signal on one QP as
DOCA builds them, rings, polls the CQE, reuses its send buffer; rank 1 polls the signal and
reads the data. Mutants: N1 no fence before the doorbell, N2 a `.cta` fence, N3 the signal
posted before the data, N4 the CQ not polled before the send buffer is reused, N5 a WQE
segment written after the doorbell, N6 a wrong rkey, N7 a relaxed signal poll with no
acquire, N8 a direct load from the other domain.
Variant T signals with an 8-byte RDMA write of a token (mscclpp's form, S-NIC-FLAG) and is
clean; N9 posts that token before the data write (race).

## Captured kernels (2026-11-06)

Captured on one 4-GPU GB300 node with `fb/nic/capall.sh` (DOCA `nic_handler` 2,
IBTA ordering, rail NIC pinned to each GPU's first port). Fake NVLink domains split the GPUs:
`ep4` = deep_ep at 4 ranks, 2 domains of 2; `ep2` and `cp2l1` = 2 ranks on GPUs 0 and 2, one
per domain; `cp4l2` = deep_cp at 4 ranks, 2 domains of 2. `ptxv check --max-states 400000`
(the deep_cp a2a cp2l1 `qkv_fwd` / `out_bwd` cases also at 2000000).

| case (kernels) | verdict | findings |
|---|---|---|
| ep4/ep2 dispatch, meta (k_hybrid_dispatch, k_hybrid_copy[_meta]) | race, violation (ep2: also deadlock); incomplete | A, B, C, D, E; ep2 deadlock F |
| ep4/ep2 cached (k_hybrid_dispatch_cached, k_hybrid_copy_cached) | race, violation, contract; incomplete | A, B, C, D, E, G |
| ep4/ep2 combine (k_hybrid_combine, k_hybrid_reduce) | race, violation, contract; incomplete | A, B, C, E, G, H |
| cp2l1 / cp4l2 qkv_fwd, qkv_bwd, out_fwd, out_bwd (A2aHybrid.exchange_sequence_windows_kernel) | no finding in 400000 states (cp2l1 qkv_fwd / out_bwd: 2000000), incomplete; cp4l2 qkv_fwd: violation I | I |
| cp2l1 flush (_FlushPeerChannels) | clean, exhaustive (15 states) | - |
| cp2l1 ag, rs (AllGather, ReduceScatter) | violation J; rs also unmodeled K | J, K |

Findings (the NIC races of the first runs went away with S-NIC-FLAG / S-NIC-WORD, which
probes P4b / P2 / P6a support; L below, listed on 2026-11-07, does involve a NIC write):
- A. mscclpp `DeviceSyncer::sync` (epSyncer, `concurrency_device.hpp`): every CTA's thread 0
  writes `count_[next] = 0` and `currentCountIdx_` with plain stores, and reads
  `currentCountIdx_` plainly: write-write / write-read races on the same words (same values).
- B. `nvlink_barrier_wo_local_sync` (comm_shim.cuh) signals peers with plain stores
  (`MemoryChannel::write`) that the peers poll with volatile loads: write-read race.
- C. A 128-bit WQE store reads a register no instruction wrote: the ctrl segment's signature
  and reserved bytes keep an uninitialized value (`(x & 0xffffff) | 0x08000000`); the
  `UBLKCP.G.S` TMA stores read staging shared memory that nothing wrote in the launch.
- D. Counters cleaned with plain stores while other threads update them atomically
  (`REDG.ADD` at window offset 0x10 vs `STG.E.64 RZ`; `ATOMG.ADD` at +0x19010 vs a
  `STG.E.64 RZ` in the same CTA): write-write races. Not checked against the source yet.
- E. A `STG.E.STRONG.SYS` / `STG.E.64.STRONG.SYS` flag and a plain `STG` of the same word in
  another warp (dispatch +0xc01.., combine -0x7e6ff0): write-write race. Not triaged.
- F. ep2 dispatch / meta: a deadlock with lanes parked at a `SHFL.IDX` (0xc4e0) whose other
  lanes are elsewhere. Not triaged (a warp-collective modeling limit is possible).
- G. `k_hybrid_reduce` exits with a bulk store (`UBLKCP.G.S`) possibly still reading shared
  memory: no `cp.async.bulk.wait_group.read` (E-BULK-EXIT; GB300 drains it).
- H. combine: a TMA store to global memory and a later `LDG.E.NA.128.CONSTANT` (ld.global.nc)
  of the same bytes in the same launch are unordered.
- I. ptxas reuses `ATOMG.E.ADD.64.STRONG.SYS [R2]`'s address register in the next `LDG R2`
  with no wait: the scoreboard model does not put ATOMG in the in-order memory queue
  (L-OPREAD). A lifter gap, not a kernel finding.
- J. `FLO.U32` with no write scoreboard, its result read after a wait on a later `POPC`'s
  scoreboard: the scoreboard model does not treat FLO / POPC as one in-order unit. A lifter
  gap.
- K. ReduceScatter's CQ waits keep the `E-SPIN` shape check failing (another branch inside
  the loop that the warp-level poll rule does not take).

## Two hosts (2026-11-07)

`tools/capture_dist.py --nnodes 2` (one torchrun per host, shared `--work`), from
the ops tree at revision b3029644a563, on two GB300 hosts (`fb/nic/capall2n.sh`; the
scheduler placed both in one NVL72 fabric clique, nvidia-smi `CliqueId` 32766 on all
GPUs). NVLink domains = hosts; every IBGDA QP connects GPUs on different hosts and goes over
the hosts' rail NICs. Each handle's peer was matched by rkey + remote base (DOCA's DevX QPs
are not listed by `rdma res show qp`, so no QPN pairing); all agree with the drivers'
labels. Sets, all from the same msl copy:
`ep8n2` / `cp8n2` = 8 ranks, 4 per host (deep_ep nvl_size 4, deep_cp `DEEPCP_LOCAL=4`);
`ep4n2` / `cp4n2` = 4 ranks, 2 per host; `ep4s` / `cp4s` = 4 ranks on one host, 2 fake
domains of 2. `ptxv check --certify --max-states 400000`, 8 threads (the certificate does
not cover NIC QPs: every case is explored). More threads widen the explored frontier: on
ep4s dispatch 8 / 12 threads give contract, race, violation (391 / 20 terminal states), 16
gives race, violation (0 terminal), 24 and 32 spend the 400000 states at depth ~1000 and
end `clean` (incomplete, 0 terminal states); runs above 8 threads are not used.

| case | verdict | findings |
|---|---|---|
| ep4n2 dispatch, cached, combine | as ep4s: contract, race, violation; incomplete | identical finding sets (instruction-normalized): A B C D E G L M N; combine A B C E G H |
| ep4n2 meta | as ep4s: race, violation; incomplete | A B C D E L M N |
| ep8n2 combine | contract, race, violation; incomplete | the 7 findings of ep4s combine |
| ep8n2 meta | race, violation; incomplete | 6 of ep4s meta's 16 |
| ep8n2 dispatch, cached | clean, no terminal state in 400000 states; incomplete | none reached |
| cp8n2 / cp4n2 / cp4s qkv_fwd, qkv_bwd, out_fwd, out_bwd, ag | clean; incomplete | - |
| cp8n2 / cp4n2 / cp4s flush | clean, exhaustive (1311 / 63 / 63 states) | - |
| cp8n2 / cp4n2 / cp4s rs | unmodeled (poll); incomplete | K |

With the same binary the 2026-11-06 single-host cases (older msl) give the same finding
sets as ep4s at the instruction level; I and J are gone there too (lifter fixes on the
integration base: ATOMG in the in-order queue, FLO / POPC), and ep2's deadlock F no longer
appears. New on the list (present in the single-host runs as well, not specific to two
hosts):
- L. dispatch / cached: a receive word the peer's NIC writes (RDMA write placed after the
  doorbell `STG.E.64.MMIO.SYS`) and a local `STG.E [+0x16020 / +0x16030], RZ` that clears
  it are unordered: write-write race. Not triaged against the source.
- M. dispatch / cached: `REDG.E.ADD.64.STRONG.GPU` to the same word from two GPUs (and a
  `.sys` load of it): `.gpu` scope does not include the other GPU (S-SCOPE). Not triaged.
- N. dispatch / cached / meta: TMA store (`UBLKCP.G.S`) and TMA load (`UBLKCP.S.G`) of the
  same staging shared memory, and k_hybrid_dispatch's TMA store vs k_hybrid_copy's TMA
  load of the same global bytes, unordered. Not triaged.

Two hosts add no finding: every finding of the 2-host cases is one of the single-host
fake-domain cases', and at 4 ranks the sets are identical. What the second host changes in
the capture: POSIX-fd / IPC keys and host clocks are per host, the QP peer comes from the
recorded keys instead of the driver, and domains come from the hosts.

