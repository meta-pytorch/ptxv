/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! NIC actor for GPU-initiated RDMA (IBGDA, docs/nic.md).
//!
//! A kernel posts mlx5 work-queue entries into a send-queue ring in GPU memory and rings the
//! QP's doorbell with an MMIO store; the NIC of its node fetches the WQEs, executes them in
//! order (RDMA write, atomic fetch-add / compare-swap), writes into the peer GPU's memory and
//! reports completions as CQEs in GPU memory, which the kernel polls. Each QP is one in-order
//! actor: at most one pending `AsyncKind::Nic` op per QP, completing the oldest announced WQE.
//! The rules (doorbell fence, per-QP order, signal after data, completion) are the rows of
//! the semantics table in docs/nic.md, each tagged `S-NIC-*`.

use std::collections::VecDeque;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_ir::MemSync;
use ptxv_ir::Scope;

use crate::exec::Loc;
use crate::machine::*;

/// Resolved configuration (from the case's `nic_qps` / `nic_mrs` / `domains`).
#[derive(Clone, Debug)]
pub struct NicCfg {
    pub qps: Vec<QpCfg>,
    pub mrs: Vec<MrCfg>,
    /// GPU of each allocation.
    pub alloc_dev: Vec<u32>,
    /// NVLink domain of each GPU (empty: one domain).
    pub domains: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct QpCfg {
    pub name: String,
    pub device: u32,
    pub peer: u32,
    pub qpn: u32,
    /// (allocation, byte offset) of the send-queue ring, the CQ and the doorbell register.
    pub sq: (usize, u64),
    pub sq_wqes: u32,
    pub cq: (usize, u64),
    pub cq_cqes: u32,
    pub db: (usize, u64),
    pub sq_ci: u64,
    pub cq_pi: u64,
    /// Out-of-order data placement (anything but IBTA ordering): not modeled.
    pub ordering: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MrCfg {
    pub key: u32,
    pub device: u32,
    pub alloc: usize,
    pub lo: u64,
    pub hi: u64,
}

impl NicCfg {
    pub fn from_case(m: &ptxv_case::Manifest, allocs: &[Alloc]) -> Result<Option<NicCfg>> {
        if m.nic_qps.is_empty() && m.domains.is_empty() {
            return Ok(None);
        }
        let idx = |name: &str| -> Result<usize> {
            m.buffers
                .iter()
                .position(|b| b.name == name)
                .ok_or_else(|| anyhow!("nic: no buffer {name}"))
        };
        let alloc_dev: Vec<u32> = m.buffers.iter().map(|b| b.device).collect();
        let ndev = 1 + alloc_dev
            .iter()
            .copied()
            .chain(m.all_launches().iter().map(|l| l.device))
            .max()
            .unwrap_or(0);
        if !m.domains.is_empty() && (m.domains.len() as u32) < ndev {
            bail!(
                "domains lists {} GPUs but the case uses {ndev}",
                m.domains.len()
            );
        }
        let mut qps = Vec::new();
        for q in &m.nic_qps {
            let fits = |name: &str, off: u64, n: u64| -> Result<(usize, u64)> {
                let a = idx(name)?;
                if off + n > allocs[a].init.len() as u64 {
                    bail!(
                        "nic qp {}: {n} bytes at {off:#x} do not fit in buffer {name}",
                        q.name
                    );
                }
                Ok((a, off))
            };
            if !q.sq_wqes.is_power_of_two() || !q.cq_cqes.is_power_of_two() {
                bail!(
                    "nic qp {}: sq_wqes and cq_cqes must be powers of two",
                    q.name
                );
            }
            let sq = fits(&q.sq, q.sq_offset, 64 * q.sq_wqes as u64)?;
            let cq = fits(&q.cq, q.cq_offset, 64 * q.cq_cqes as u64)?;
            let db = fits(&q.db, q.db_offset, 8)?;
            if alloc_dev[sq.0] != q.device || alloc_dev[cq.0] != q.device {
                bail!(
                    "nic qp {}: the send and completion queues must be in GPU {}'s memory",
                    q.name,
                    q.device
                );
            }
            qps.push(QpCfg {
                name: q.name.clone(),
                device: q.device,
                peer: q.peer,
                qpn: q.qpn,
                sq,
                sq_wqes: q.sq_wqes,
                cq,
                cq_cqes: q.cq_cqes,
                db,
                sq_ci: q.sq_ci,
                cq_pi: q.cq_pi,
                ordering: q.ordering.clone().filter(|o| o != "ibta"),
            });
        }
        let mut mrs = Vec::new();
        for r in &m.nic_mrs {
            let a = idx(&r.buffer)?;
            let size = allocs[a].init.len() as u64;
            let hi = r.len.map_or(size, |l| r.offset + l);
            if r.offset > hi || hi > size {
                bail!("nic mr {:#x}: outside buffer {}", r.key, r.buffer);
            }
            mrs.push(MrCfg {
                key: r.key,
                device: alloc_dev[a],
                alloc: a,
                lo: r.offset,
                hi,
            });
        }
        Ok(Some(NicCfg {
            qps,
            mrs,
            alloc_dev,
            domains: m.domains.clone(),
        }))
    }

    /// NVLink domain of GPU `d`.
    pub fn domain(&self, d: u32) -> u32 {
        self.domains.get(d as usize).copied().unwrap_or(0)
    }
}

/// A WQE announced by a doorbell and not executed yet.
#[derive(Clone, Debug, Hash)]
pub struct NicWqe {
    /// Full WQE counter.
    pub idx: u64,
    /// Actor of its execution (`Machine::tma_actor_id` of the ringing thread).
    pub actor: u32,
    /// What the doorbell released to the NIC (S-NIC-DB): empty unless the doorbell store was
    /// preceded by a release of at least GPU scope.
    pub vc: VC,
    /// Scope of that release (None: no release fence before the doorbell).
    pub scope: Option<Scope>,
    pub issuer: u32,
    /// The doorbell store, and the issuer's own clock at it.
    pub pc: u32,
    pub db_clock: u32,
}

#[derive(Clone, Debug, Hash)]
pub struct QpState {
    /// WQEs announced by doorbells (full counter).
    pub pi: u64,
    /// WQEs executed.
    pub ci: u64,
    /// CQEs written.
    pub cq_pi: u64,
    pub queue: VecDeque<NicWqe>,
    /// Clock of the QP's last executed WQE (S-NIC-ORDER: the next one follows it).
    pub last: VC,
    /// The QP stopped (an error completion; what the NIC does then is not modeled).
    pub dead: bool,
}

/// NIC state of a machine (hashed); `names` (actor -> description) is not.
#[derive(Clone, Debug, Default)]
pub struct NicState {
    pub qps: Vec<QpState>,
    pub names: Vec<(u32, String)>,
    /// Remote locations NIC atomics wrote: (allocation, offset).
    pub atomics: Vec<(usize, u64)>,
}

impl Hash for NicState {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.qps.hash(h);
        self.atomics.hash(h);
    }
}

impl NicState {
    pub fn new(cfg: &NicCfg) -> NicState {
        NicState {
            qps: cfg
                .qps
                .iter()
                .map(|q| QpState {
                    pi: q.sq_ci,
                    ci: q.sq_ci,
                    cq_pi: q.cq_pi,
                    queue: VecDeque::new(),
                    last: VC::new(),
                    dead: false,
                })
                .collect(),
            names: vec![],
            atomics: vec![],
        }
    }
}

/// mlx5 opcodes (PRM; DOCA `doca_gpunetio_verbs_def.h`).
const OP_NOP: u8 = 0x00;
const OP_RDMA_WRITE: u8 = 0x08;
const OP_RDMA_WRITE_IMM: u8 = 0x09;
const OP_ATOMIC_CS: u8 = 0x11;
const OP_ATOMIC_FA: u8 = 0x12;

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes(b[..4].try_into().unwrap())
}
fn be64(b: &[u8]) -> u64 {
    u64::from_be_bytes(b[..8].try_into().unwrap())
}

impl Machine {
    pub fn nic_cfg(&self) -> Option<&Arc<NicCfg>> {
        self.launch.nic.as_ref()
    }

    /// Name of a NIC actor, if `a` is one.
    pub fn nic_actor_name(&self, a: u32) -> Option<String> {
        let n = self.nic.as_ref()?;
        n.names
            .iter()
            .find(|(x, _)| *x == a)
            .map(|(_, s)| s.clone())
    }

    pub fn is_nic_actor(&self, a: u32) -> bool {
        self.nic
            .as_ref()
            .is_some_and(|n| n.names.iter().any(|(x, _)| *x == a))
    }

    /// ASSUME(S-NIC-DOMAIN): a GPU reaches memory of another NVLink domain only through the
    /// NIC. Thread `t`'s access to allocation `alloc`.
    pub fn nic_domain_check(&self, t: usize, alloc: usize) -> Result<(), String> {
        let Some(cfg) = self.nic_cfg() else {
            return Ok(());
        };
        if cfg.domains.is_empty() {
            return Ok(());
        }
        let dev = self.launch.g(self.threads[t].cta as u32).device;
        let adev = cfg.alloc_dev[alloc];
        if cfg.domain(dev) != cfg.domain(adev) {
            return Err(format!(
                "GPU {dev} accesses buffer {} of GPU {adev} directly, but they are in different NVLink domains (only the NIC reaches it; S-NIC-DOMAIN)",
                self.launch.allocs[alloc].name
            ));
        }
        Ok(())
    }

    /// A store by thread `t` that landed in global memory: a doorbell when it is a QP's
    /// doorbell register. Called after the store's own memory effects.
    pub fn nic_store(&mut self, t: usize, pc: u32, loc: Loc, data: &[u8], sync: Option<MemSync>) {
        let Some(cfg) = self.launch.nic.clone() else {
            return;
        };
        let Loc::Global { alloc, off } = loc else {
            return;
        };
        let hits: Vec<usize> = (0..cfg.qps.len())
            .filter(|&q| cfg.qps[q].db.0 == alloc)
            .collect();
        if hits.is_empty() {
            return;
        }
        let Some(q) = hits
            .iter()
            .copied()
            .find(|&q| cfg.qps[q].db.1 == off && data.len() == 8)
        else {
            let msg = format!(
                "store of {} bytes at offset {off:#x} of the doorbell page {} at {}: only 8-byte doorbell rings are modeled (BlueFlame WQE writes are not)",
                data.len(),
                self.launch.allocs[alloc].name,
                self.k.describe(pc as usize)
            );
            self.report("unmodeled", format!("nic-uar:{pc}"), msg);
            return;
        };
        let qc = &cfg.qps[q];
        // ASSUME(S-NIC-DB): the doorbell is a strong system-scope store (st.mmio.relaxed.sys,
        // SASS STG.E.64.MMIO.SYS, or st.relaxed.sys); a weak store may be buffered.
        if !sync.is_some_and(|m| m.scope == Scope::Sys) {
            let msg = format!(
                "doorbell of QP {} rung by a store that is not strong at system scope at {} (st.mmio.relaxed.sys expected; S-NIC-DB)",
                qc.name,
                self.k.describe(pc as usize)
            );
            self.report("unmodeled", format!("nic-db-weak:{pc}"), msg);
            return;
        }
        let ctrl0 = be32(&data[0..4]);
        let qpn = be32(&data[4..8]) >> 8;
        if qpn != qc.qpn {
            let msg = format!(
                "doorbell of QP {} at {} carries QP number {qpn:#x}, not {:#x}",
                qc.name,
                self.k.describe(pc as usize),
                qc.qpn
            );
            self.report("violation", format!("nic-db-qpn:{pc}"), msg);
            return;
        }
        let pi16 = (ctrl0 >> 8) & 0xffff;
        // What the doorbell releases (as `sync_publish`): a release store its thread's clock,
        // a relaxed one what the thread's last release fence released, at that fence's scope.
        let m = sync.unwrap();
        let th = &self.threads[t];
        let (rel, scope) = if m.releases() {
            (th.vc.clone(), Some(m.scope))
        } else if th.membar_vc == VC::new() {
            (VC::new(), None)
        } else {
            (th.membar_vc.clone(), Some(th.membar_scope))
        };
        // ASSUME(S-NIC-DB): a release of at least GPU scope orders the WQE and payload stores
        // (in this GPU's memory) before the NIC's reads that the doorbell triggers.
        let vc = if scope.is_some_and(|s| s >= Scope::Gpu) {
            rel
        } else {
            VC::new()
        };
        self.rec(16, q as u32, 0, 1, true);
        let st = &self.nic.as_ref().unwrap().qps[q];
        if st.dead {
            return;
        }
        let delta = (pi16 as u64).wrapping_sub(st.pi) & 0xffff;
        if delta == 0 || delta >= 0x8000 {
            return; // nothing new (a repeated or older ring)
        }
        let first = st.queue.is_empty();
        let from = st.pi;
        let mut new = Vec::new();
        for idx in from..from + delta {
            let serial = self.threads[t].tma_count;
            self.threads[t].make_mut().tma_count += 1;
            let actor = self.tma_actor_id(t, serial);
            new.push(NicWqe {
                idx,
                actor,
                vc: vc.clone(),
                scope,
                issuer: t as u32,
                pc,
                db_clock: vc_get(&self.threads[t].vc, t as u32),
            });
        }
        let name_of = |w: &NicWqe| format!("NIC (QP {}, WQE {})", qc.name, w.idx);
        let ns = self.nic.as_mut().unwrap();
        for w in &new {
            ns.names.push((w.actor, name_of(w)));
        }
        let st = &mut ns.qps[q];
        st.pi = from + delta;
        st.queue.extend(new);
        if first {
            self.nic_push_op(q);
        }
    }

    /// The pending op that executes QP `q`'s oldest announced WQE.
    fn nic_push_op(&mut self, q: usize) {
        let w = self.nic.as_ref().unwrap().qps[q].queue[0].clone();
        let mut vc = w.vc.clone();
        // ASSUME(S-NIC-SIGNAL): a WQE executes after the QP's previous one completed, so its
        // effects (an atomic's release) follow the earlier WQEs' writes.
        vc_join(&mut vc, &self.nic.as_ref().unwrap().qps[q].last);
        vc_tick(&mut vc, w.actor);
        self.pending.push(AsyncOp {
            sh: vec![],
            fifo: None,
            actor: w.actor,
            vc,
            issuer: w.issuer,
            pc: w.pc,
            kind: AsyncKind::Nic { qp: q as u16 },
            vis: VC::new(),
        });
        self.pending
            .sort_by_key(|o| (o.actor, vc_get(&o.vc, o.actor)));
    }

    /// Bytes `[off, off + n)` of allocation `alloc` not ordered before `vc`: the first such
    /// byte and its last writer.
    fn nic_unordered(&self, alloc: usize, off: u64, n: u64, vc: &VC) -> Option<(u64, Epoch)> {
        let mem = &self.gmem[alloc];
        for i in off..off + n {
            let s = mem.sh.get(i as usize);
            if let Some(w) = s.writes().find(|w| !w.hb(vc)) {
                return Some((i, *w));
            }
        }
        None
    }

    /// Allocation and offset of `[addr, addr + n)` on GPU `dev`, checked against the memory
    /// registration `key`.
    fn nic_mr(&self, key: u32, dev: u32, addr: u64, n: u64) -> Result<(usize, u64), String> {
        let cfg = self.launch.nic.as_ref().unwrap();
        let a = self
            .launch
            .allocs
            .iter()
            .position(|al| addr >= al.base && addr + n <= al.base + al.init.len() as u64)
            .ok_or_else(|| format!("address {addr:#x} (+{n}) is outside every buffer"))?;
        let off = addr - self.launch.allocs[a].base;
        if cfg.alloc_dev[a] != dev {
            return Err(format!(
                "address {addr:#x} is in buffer {} of GPU {}, not GPU {dev}",
                self.launch.allocs[a].name, cfg.alloc_dev[a]
            ));
        }
        if !cfg
            .mrs
            .iter()
            .any(|r| r.key == key && r.alloc == a && off >= r.lo && off + n <= r.hi)
        {
            return Err(format!(
                "no memory registration with key {key:#x} covers {n} bytes at {off:#x} of buffer {}",
                self.launch.allocs[a].name
            ));
        }
        Ok((a, off))
    }

    fn nic_fail(&mut self, q: usize, kind: &str, sig: String, msg: String) {
        self.report(kind, sig, msg);
        let st = &mut self.nic.as_mut().unwrap().qps[q];
        st.dead = true;
        st.queue.clear();
    }

    /// Execute the oldest announced WQE of QP `q` (a scheduler choice; `op` is its pending op).
    pub fn complete_nic(&mut self, op: &AsyncOp, q: usize) {
        let cfg = self.launch.nic.clone().unwrap();
        let qc = &cfg.qps[q];
        let Some(w) = self.nic.as_mut().unwrap().qps[q].queue.pop_front() else {
            return;
        };
        self.rec(16, q as u32, 0, 1, true);
        let (actor, vc, pc) = (op.actor, op.vc.clone(), op.pc);
        let who = format!("NIC (QP {}, WQE {})", qc.name, w.idx);
        let db = self.k.describe(pc as usize);
        if let Some(o) = &qc.ordering {
            let msg = format!(
                "QP {} uses data-placement ordering \"{o}\": only IBTA ordering is modeled (S-NIC-ORDER)",
                qc.name
            );
            self.nic_fail(q, "unmodeled", format!("nic-ordering:{q}"), msg);
            return;
        }
        // ASSUME(S-NIC-DB): the NIC reads the WQE after the doorbell that announced it.
        let slot = (w.idx & (qc.sq_wqes as u64 - 1)) * 64;
        let (sa, so) = (qc.sq.0, qc.sq.1 + slot);
        if let Some((b, e)) = self.nic_unordered(sa, so, 64, &w.vc) {
            let why = match w.scope {
                None => "no release fence precedes the doorbell store".to_string(),
                Some(s) if s < Scope::Gpu => format!(
                    "the release before the doorbell is only {} scope",
                    format!("{s:?}").to_lowercase()
                ),
                Some(_) => "the write is not ordered before the doorbell's release".to_string(),
            };
            let msg = format!(
                "{who} reads the WQE (byte {:#x} of {}) written by {} at {}, but {why}: the NIC can fetch a stale WQE after the doorbell at {db} (fence.release.gpu or stronger before the doorbell; S-NIC-DB)",
                b - qc.sq.1,
                self.launch.allocs[sa].name,
                self.actor_name(e.actor),
                self.k.describe(e.pc as usize)
            );
            // Executed with the bytes as they are (one of the outcomes).
            self.report_pcs(
                "race",
                format!("nic-wqe-unordered:{}", e.pc),
                msg,
                vec![e.pc, pc],
            );
        }
        let wqe = self.access(actor, &vc, pc, Loc::Global { alloc: sa, off: so }, 64, None);
        let ctrl0 = be32(&wqe[0..4]);
        let opcode = (ctrl0 & 0xff) as u8;
        let widx = (ctrl0 >> 8) & 0xffff;
        let qpn_ds = be32(&wqe[4..8]);
        let ds = qpn_ds & 0x3f;
        let fm_ce_se = wqe[11];
        let ce = (fm_ce_se >> 2) & 3;
        if widx as u64 != w.idx & 0xffff || qpn_ds >> 8 != qc.qpn {
            let msg = format!(
                "{who}: the WQE in slot {} carries index {widx:#x} and QP number {:#x} (expected {:#x} and {:#x}): the NIC would execute a stale or foreign WQE",
                w.idx & (qc.sq_wqes as u64 - 1),
                qpn_ds >> 8,
                w.idx & 0xffff,
                qc.qpn
            );
            self.nic_fail(q, "violation", format!("nic-bad-wqe:{q}:{}", w.idx), msg);
            return;
        }
        if ds > 4 {
            let msg = format!(
                "{who}: WQE of {ds} segments spans more than one 64-byte block (not modeled)"
            );
            self.nic_fail(q, "unmodeled", format!("nic-wqe-size:{q}"), msg);
            return;
        }
        let mut vc = vc;
        match opcode {
            OP_NOP => {}
            OP_RDMA_WRITE => {
                let raddr = be64(&wqe[16..24]);
                let rkey = be32(&wqe[24..28]);
                let mut roff = 0u64;
                for s in 2..ds as usize {
                    let d = &wqe[16 * s..16 * s + 16];
                    let bc = be32(&d[0..4]);
                    let lkey = be32(&d[4..8]);
                    let laddr = be64(&d[8..16]);
                    if bc & 0x8000_0000 != 0 {
                        let msg = format!("{who}: inline data segment (not modeled)");
                        self.nic_fail(q, "unmodeled", format!("nic-inline:{q}"), msg);
                        return;
                    }
                    let n = bc as u64;
                    if n == 0 {
                        continue;
                    }
                    let src = match self.nic_mr(lkey, qc.device, laddr, n) {
                        Ok(x) => x,
                        Err(e) => {
                            let msg = format!(
                                "{who}: local access error ({e}): an error completion, nothing delivered"
                            );
                            self.nic_fail(q, "fault", format!("nic-lkey:{pc}"), msg);
                            return;
                        }
                    };
                    let dst = match self.nic_mr(rkey, qc.peer, raddr + roff, n) {
                        Ok(x) => x,
                        Err(e) => {
                            let msg = format!(
                                "{who}: remote access error ({e}): an error completion, nothing delivered"
                            );
                            self.nic_fail(q, "fault", format!("nic-rkey:{pc}"), msg);
                            return;
                        }
                    };
                    // ASSUME(S-NIC-DB): the payload must be ordered before the doorbell too.
                    if let Some((b, e)) = self.nic_unordered(src.0, src.1, n, &w.vc) {
                        let what = if e.actor == w.issuer && e.clock > w.db_clock {
                            "after the doorbell: the send buffer is reused before the WQE's completion was observed (poll its CQE first; S-NIC-CQE)"
                        } else {
                            "not ordered before the doorbell: the NIC can send stale data (S-NIC-DB)"
                        };
                        let msg = format!(
                            "{who} (doorbell at {db}) reads its source (byte {:#x} of {}) written by {} at {}, {what}",
                            b,
                            self.launch.allocs[src.0].name,
                            self.actor_name(e.actor),
                            self.k.describe(e.pc as usize)
                        );
                        self.report_pcs(
                            "race",
                            format!("nic-src-unordered:{}", e.pc),
                            msg,
                            vec![e.pc, pc],
                        );
                    }
                    let data = self.access(
                        actor,
                        &vc,
                        pc,
                        Loc::Global {
                            alloc: src.0,
                            off: src.1,
                        },
                        n as usize,
                        None,
                    );
                    let loc = Loc::Global {
                        alloc: dst.0,
                        off: dst.1,
                    };
                    if ds == 3 && (n == 4 || n == 8) && (raddr + roff).is_multiple_of(n) {
                        // ASSUME(S-NIC-FLAG): a lone naturally aligned 4- or 8-byte RDMA write
                        // (mscclpp's IBGDA signal token) lands as one strong .sys write that
                        // releases the QP's earlier WQEs, like a fetch-add signal.
                        self.access_x(actor, &vc, pc, loc, n as usize, Some(&data), STRONG_SYS);
                        self.nic_publish(dst.0, dst.1, actor, &vc, false);
                    } else {
                        // ASSUME(S-NIC-WRITE): other RDMA write data synchronizes with nothing
                        // and has no placement order. ASSUME(S-NIC-WORD): each naturally
                        // aligned 8-byte word (4-byte at the edges) lands single-copy atomic,
                        // a relaxed .sys write; unaligned edge bytes are plain writes.
                        let base = raddr + roff;
                        let mut k = 0u64;
                        while k < n {
                            let a = base + k;
                            let w = if a.is_multiple_of(8) && n - k >= 8 {
                                8
                            } else if a.is_multiple_of(4) && n - k >= 4 {
                                4
                            } else {
                                1
                            };
                            let l = Loc::Global {
                                alloc: dst.0,
                                off: dst.1 + k,
                            };
                            let bytes = &data[k as usize..(k + w) as usize];
                            if w == 1 {
                                self.access(actor, &vc, pc, l, 1, Some(bytes));
                            } else {
                                self.access_x(
                                    actor,
                                    &vc,
                                    pc,
                                    l,
                                    w as usize,
                                    Some(bytes),
                                    STRONG_SYS,
                                );
                            }
                            k += w;
                        }
                        self.nic_sync_clear(dst.0, dst.1, n);
                    }
                    roff += n;
                }
            }
            OP_ATOMIC_FA | OP_ATOMIC_CS => {
                let raddr = be64(&wqe[16..24]);
                let rkey = be32(&wqe[24..28]);
                let swap_add = be64(&wqe[32..40]);
                let compare = be64(&wqe[40..48]);
                let d = &wqe[48..64];
                let bc = be32(&d[0..4]);
                let lkey = be32(&d[4..8]);
                let laddr = be64(&d[8..16]);
                if bc != 8 || !raddr.is_multiple_of(8) {
                    let msg = format!(
                        "{who}: atomic of {bc} bytes at {raddr:#x} (only aligned 8-byte atomics are modeled)"
                    );
                    self.nic_fail(q, "unmodeled", format!("nic-atomic-size:{q}"), msg);
                    return;
                }
                let dst = match self.nic_mr(rkey, qc.peer, raddr, 8) {
                    Ok(x) => x,
                    Err(e) => {
                        let msg = format!("{who}: remote access error ({e}): an error completion");
                        self.nic_fail(q, "fault", format!("nic-rkey:{pc}"), msg);
                        return;
                    }
                };
                let sink = match self.nic_mr(lkey, qc.device, laddr, 8) {
                    Ok(x) => x,
                    Err(e) => {
                        let msg = format!("{who}: local access error ({e}): an error completion");
                        self.nic_fail(q, "fault", format!("nic-lkey:{pc}"), msg);
                        return;
                    }
                };
                // ASSUME(S-NIC-ATOMIC): the atomic is a strong system-scope read-modify-write
                // of the NIC, ordered after the QP's earlier WQEs (S-NIC-SIGNAL), releasing
                // what they wrote; the 64-bit value is little-endian in memory (probes/ibgda
                // P0: fetch-adds of 1 leave 1, 2, 3 as little-endian words).
                self.nic_atomic_mixed(dst.0, dst.1, &vc, pc, &who);
                let loc = Loc::Global {
                    alloc: dst.0,
                    off: dst.1,
                };
                let old = self.access_x(actor, &vc, pc, loc, 8, None, STRONG_SYS);
                let o = u64::from_le_bytes(old[..8].try_into().unwrap());
                let n = if opcode == OP_ATOMIC_FA {
                    o.wrapping_add(swap_add)
                } else if o == compare {
                    swap_add
                } else {
                    o
                };
                self.access_x(actor, &vc, pc, loc, 8, Some(&n.to_le_bytes()), STRONG_SYS);
                self.nic_publish(dst.0, dst.1, actor, &vc, true);
                let ns = self.nic.as_mut().unwrap();
                if !ns.atomics.contains(&(dst.0, dst.1)) {
                    ns.atomics.push((dst.0, dst.1));
                    ns.atomics.sort_unstable();
                }
                // The fetched value returns to the local buffer.
                self.access(
                    actor,
                    &vc,
                    pc,
                    Loc::Global {
                        alloc: sink.0,
                        off: sink.1,
                    },
                    8,
                    Some(&old),
                );
                self.nic_sync_clear(sink.0, sink.1, 8);
            }
            OP_RDMA_WRITE_IMM => {
                let msg = format!(
                    "{who}: RDMA write with immediate consumes a receive WQE at the peer (not modeled)"
                );
                self.nic_fail(q, "unmodeled", format!("nic-op:{q}:{opcode}"), msg);
                return;
            }
            _ => {
                let msg = format!("{who}: mlx5 opcode {opcode:#x} is not modeled");
                self.nic_fail(q, "unmodeled", format!("nic-op:{q}:{opcode}"), msg);
                return;
            }
        }
        // ASSUME(S-NIC-CQE): with CQ update requested the NIC writes a CQE once the WQE
        // completed, which for an RDMA write means the data was placed at the peer; the CQE
        // write is a strong system-scope release of everything the WQE (and the QP's earlier
        // WQEs) did. Owner bit = pass parity over the CQ ring; opcode 0 (requester).
        if ce == 2 {
            let st = &self.nic.as_ref().unwrap().qps[q];
            let n = st.cq_pi;
            let ncq = qc.cq_cqes as u64;
            let off = qc.cq.1 + (n & (ncq - 1)) * 64;
            let mut cqe = [0u8; 64];
            cqe[60..62].copy_from_slice(&((w.idx & 0xffff) as u16).to_be_bytes());
            cqe[63] = ((n / ncq) & 1) as u8;
            let loc = Loc::Global {
                alloc: qc.cq.0,
                off,
            };
            self.access_x(actor, &vc, pc, loc, 64, Some(&cqe), STRONG_SYS);
            for k in [56u64, 60, 62, 63] {
                self.nic_publish(qc.cq.0, off + k, actor, &vc, false);
            }
            self.nic.as_mut().unwrap().qps[q].cq_pi = n + 1;
        } else if ce != 0 {
            let msg = format!("{who}: completion mode {ce} (only 'always' and 'on error')");
            self.nic_fail(q, "unmodeled", format!("nic-ce:{q}"), msg);
            return;
        }
        vc_tick(&mut vc, actor);
        let st = &mut self.nic.as_mut().unwrap().qps[q];
        st.ci = w.idx + 1;
        st.last = vc;
        if !st.queue.is_empty() {
            self.nic_push_op(q);
        }
    }

    /// The NIC's strong write at (alloc, off) as a synchronization source releasing `vc`.
    fn nic_publish(&mut self, alloc: usize, off: u64, actor: u32, vc: &VC, rmw: bool) {
        let key = (1u8, alloc as u32, off);
        let mut rel = vc.clone();
        if rmw {
            if let Some(old) = self.syncs.get(&key) {
                vc_join(&mut rel, &old.rel);
            }
        }
        self.syncs.insert(
            key,
            SyncCell {
                writer: actor,
                scope: Scope::Sys,
                rel,
                rel_scope: Scope::Sys,
                rr: VC::new(),
            },
        );
    }

    /// Plain NIC writes end the release sequences of the bytes they cover.
    fn nic_sync_clear(&mut self, alloc: usize, off: u64, n: u64) {
        let keys: Vec<(u8, u32, u64)> = self
            .syncs
            .range((1u8, alloc as u32, off)..(1u8, alloc as u32, off + n))
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            self.syncs.remove(&k);
        }
    }

    /// ASSUME(S-NIC-ATOMIC): NIC atomics are atomic with respect to each other only, not to
    /// GPU atomics on the same word. A NIC atomic with a GPU strong write it is not ordered
    /// after, or (`nic_gpu_atomic`) a GPU atomic not ordered after a NIC atomic, is not
    /// modeled.
    fn nic_atomic_mixed(&mut self, alloc: usize, off: u64, vc: &VC, pc: u32, who: &str) {
        let mem = &self.gmem[alloc];
        let mut bad = None;
        for i in off..off + 8 {
            for w in mem.sh.get(i as usize).writes() {
                if w.strong != 0 && !w.hb(vc) && !self.is_nic_actor(w.actor) {
                    bad = Some(*w);
                }
            }
        }
        if let Some(e) = bad {
            let msg = format!(
                "{who} (doorbell at {}) performs an atomic on a word that {} at {} also writes atomically, unordered with it: NIC atomics are not atomic with respect to GPU atomics (S-NIC-ATOMIC)",
                self.k.describe(pc as usize),
                self.actor_name(e.actor),
                self.k.describe(e.pc as usize)
            );
            self.report_pcs(
                "unmodeled",
                format!("nic-atomic-mixed:{}", e.pc),
                msg,
                vec![e.pc, pc],
            );
        }
    }

    /// A GPU atomic by thread `t` (clock `vc`) at `loc`: see `nic_atomic_mixed`.
    pub fn nic_gpu_atomic(&mut self, t: usize, loc: Loc, n: usize, pc: u32) {
        let Some(ns) = self.nic.as_ref() else {
            return;
        };
        let Loc::Global { alloc, off } = loc else {
            return;
        };
        if !ns
            .atomics
            .iter()
            .any(|&(a, o)| a == alloc && o < off + n as u64 && off < o + 8)
        {
            return;
        }
        let vc = self.threads[t].vc.clone();
        let mem = &self.gmem[alloc];
        let mut bad = None;
        for i in off..off + n as u64 {
            for w in mem.sh.get(i as usize).writes() {
                if self.is_nic_actor(w.actor) && !w.hb(&vc) {
                    bad = Some(*w);
                }
            }
        }
        if let Some(e) = bad {
            let msg = format!(
                "atomic at {} on a word that {} updates atomically, unordered with it: NIC atomics are not atomic with respect to GPU atomics (S-NIC-ATOMIC)",
                self.k.describe(pc as usize),
                self.actor_name(e.actor)
            );
            self.report_pcs(
                "unmodeled",
                format!("nic-atomic-mixed:{pc}"),
                msg,
                vec![e.pc, pc],
            );
        }
    }
}
