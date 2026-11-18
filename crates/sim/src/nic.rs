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
use crate::num::NicAtomicNum;

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
    /// The strong accesses of the NIC actors, sorted by actor (MISS 194: what decides
    /// complete overlap with them, `NicState::elem`).
    pub acc: Vec<NicAcc>,
}

/// The kind of a strong access of the NIC (`NicAcc`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NicAccKind {
    /// An atomic's aligned 8-byte word (S-NIC-ATOMIC).
    Atomic,
    /// A lone naturally aligned 4- or 8-byte RDMA write (S-NIC-FLAG).
    Flag,
    /// Other RDMA write data (S-NIC-WORD): naturally aligned 8-byte chunks, 4-byte ones at the
    /// edges, plain bytes at unaligned edges.
    Words,
    /// A 64-byte CQE (S-NIC-CQE), whose fields the model publishes at bytes 56-59, 60-61, 62
    /// and 63.
    Cqe,
}

/// One strong access of a NIC actor: its allocation range and the address of its first byte
/// (S-NIC-WORD's chunks are naturally aligned in the address space).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NicAcc {
    pub actor: u32,
    pub alloc: u32,
    pub off: u64,
    pub n: u64,
    pub addr: u64,
    pub kind: NicAccKind,
}

/// The element of a NIC strong access that covers a byte (`NicState::elem`): what complete
/// overlap with a thread's access is decided against (S-SCOPE, MISS 194).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NicElem {
    /// One location of this many bytes at this allocation offset: an atomic's word, a flag
    /// write, an RDMA write's chunk.
    Exact(u64, u8),
    /// A CQE field (offset, size): an access of another extent is not modeled.
    Field(u64, u8),
    /// No element the model decides: a CQE byte outside its fields, or no recorded access.
    Unknown,
}

impl Hash for NicState {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.qps.hash(h);
        self.atomics.hash(h);
        self.acc.hash(h);
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
            acc: vec![],
        }
    }

    /// Records a strong access of a NIC actor (`acc` stays sorted by actor).
    pub fn note_acc(&mut self, a: NicAcc) {
        let i = self.acc.partition_point(|x| x.actor <= a.actor);
        self.acc.insert(i, a);
    }

    /// Is `actor` a NIC actor with a recorded strong access?
    pub fn has_acc(&self, actor: u32) -> bool {
        self.acc
            .get(self.acc.partition_point(|x| x.actor < actor))
            .is_some_and(|x| x.actor == actor)
    }

    /// MISS 194: the element of NIC actor `actor`'s strong access that covers byte `off` of
    /// allocation `alloc`. S-NIC-WORD's chunks follow `complete_nic`: the naturally aligned
    /// 8-byte word holding the byte when the write covers all of it, else the 4-byte one, else
    /// the byte (a plain write).
    pub fn elem(&self, actor: u32, alloc: usize, off: u64) -> NicElem {
        let lo = self.acc.partition_point(|x| x.actor < actor);
        for x in self.acc[lo..].iter().take_while(|x| x.actor == actor) {
            if x.alloc as usize != alloc || off < x.off || off >= x.off + x.n {
                continue;
            }
            return match x.kind {
                NicAccKind::Atomic | NicAccKind::Flag => NicElem::Exact(x.off, x.n as u8),
                NicAccKind::Words => {
                    let (a, end) = (x.addr + (off - x.off), x.addr + x.n);
                    let (w8, w4) = (a & !7, a & !3);
                    if w8 >= x.addr && w8 + 8 <= end {
                        NicElem::Exact(off - (a - w8), 8)
                    } else if w4 >= x.addr && w4 + 4 <= end {
                        NicElem::Exact(off - (a - w4), 4)
                    } else {
                        NicElem::Exact(off, 1)
                    }
                }
                NicAccKind::Cqe => match off - x.off {
                    56..=59 => NicElem::Field(x.off + 56, 4),
                    60 | 61 => NicElem::Field(x.off + 60, 2),
                    62 => NicElem::Field(x.off + 62, 1),
                    63 => NicElem::Field(x.off + 63, 1),
                    _ => NicElem::Unknown,
                },
            };
        }
        NicElem::Unknown
    }

    /// The first byte of the CQE when NIC actor `actor`'s strong access that covers byte `off` of
    /// allocation `alloc` is a CQE write.
    pub fn cqe_at(&self, actor: u32, alloc: usize, off: u64) -> Option<u64> {
        let lo = self.acc.partition_point(|x| x.actor < actor);
        self.acc[lo..]
            .iter()
            .take_while(|x| x.actor == actor)
            .find(|x| x.alloc as usize == alloc && off >= x.off && off < x.off + x.n)
            .filter(|x| x.kind == NicAccKind::Cqe)
            .map(|x| x.off)
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
    pub fn nic_store(
        &mut self,
        t: usize,
        pc: u32,
        loc: Loc,
        data: &[u8],
        sync: Option<MemSync>,
    ) -> Result<(), String> {
        let Some(cfg) = self.launch.nic.clone() else {
            return Ok(());
        };
        let Loc::Global { alloc, off } = loc else {
            return Ok(());
        };
        let hits: Vec<usize> = (0..cfg.qps.len())
            .filter(|&q| cfg.qps[q].db.0 == alloc)
            .collect();
        if hits.is_empty() {
            return Ok(());
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
            return Ok(());
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
            return Ok(());
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
            return Ok(());
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
        self.rec(
            Resource {
                space: ResourceSpace::NicQueue,
                owner: q as u32,
                start: 0,
                end: 1,
            },
            true,
        );
        let st = &self.nic.as_ref().unwrap().qps[q];
        if st.dead {
            return Ok(());
        }
        let delta = (pi16 as u64).wrapping_sub(st.pi) & 0xffff;
        if delta == 0 || delta >= 0x8000 {
            return Ok(()); // nothing new (a repeated or older ring)
        }
        let first = st.queue.is_empty();
        let from = st.pi;
        let mut new = Vec::new();
        for idx in from..from + delta {
            // A failed allocation (MISS 68) fails the store's step before the ring advances.
            let actor = self.async_actor(t)?;
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
        Ok(())
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
            seq: 0,
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
        self.rec(
            Resource {
                space: ResourceSpace::NicQueue,
                owner: q as u32,
                start: 0,
                end: 1,
            },
            true,
        );
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
                    let from = Loc::Global {
                        alloc: src.0,
                        off: src.1,
                    };
                    let data = self.access(actor, &vc, pc, from, n as usize, None);
                    // N-NUM-TRACK: the write moves the source words' numerics nodes and taint
                    // (MISS 224: the destination kept the nodes of what it held before).
                    let snapshot = self.num_read_snapshot(Some(from), &data, pc);
                    let loc = Loc::Global {
                        alloc: dst.0,
                        off: dst.1,
                    };
                    let flag = ds == 3 && (n == 4 || n == 8) && (raddr + roff).is_multiple_of(n);
                    self.nic.as_mut().unwrap().note_acc(NicAcc {
                        actor,
                        alloc: dst.0 as u32,
                        off: dst.1,
                        n,
                        addr: raddr + roff,
                        kind: if flag {
                            NicAccKind::Flag
                        } else {
                            NicAccKind::Words
                        },
                    });
                    if flag {
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
                    self.num_copy_snapshot(loc, &data, snapshot.as_ref(), pc);
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
                self.nic.as_mut().unwrap().note_acc(NicAcc {
                    actor,
                    alloc: dst.0 as u32,
                    off: dst.1,
                    n: 8,
                    addr: raddr,
                    kind: NicAccKind::Atomic,
                });
                let loc = Loc::Global {
                    alloc: dst.0,
                    off: dst.1,
                };
                let old = self.access_x(actor, &vc, pc, loc, 8, None, STRONG_SYS);
                // N-NUM-TRACK: the fetched value carries the old word's numerics nodes to the
                // local buffer (MISS 224); what may differ in the exact computation is read
                // before the write, for the word's own nodes (MISS 257, `num_nic_atomic`).
                let snapshot = self.num_read_snapshot(Some(loc), &old, pc);
                let word = |k: u64| Loc::Global {
                    alloc: dst.0,
                    off: dst.1 + k,
                };
                let field = |k: u64| Loc::Global {
                    alloc: sa,
                    off: so + k,
                };
                let old_inexact = [0, 4].map(|k| self.num_mem_inexact(word(k), 4));
                let operand_inexact = self.num_mem_inexact(field(32), 8);
                let compare_inexact = opcode == OP_ATOMIC_CS && self.num_mem_inexact(field(40), 8);
                let o = u64::from_le_bytes(old[..8].try_into().unwrap());
                let n = if opcode == OP_ATOMIC_FA {
                    o.wrapping_add(swap_add)
                } else if o == compare {
                    swap_add
                } else {
                    o
                };
                self.access_x(actor, &vc, pc, loc, 8, Some(&n.to_le_bytes()), STRONG_SYS);
                let a = NicAtomicNum {
                    old: o,
                    new: n,
                    add: (opcode == OP_ATOMIC_FA).then_some(swap_add),
                    swapped: opcode == OP_ATOMIC_CS && o == compare,
                    old_inexact,
                    operand_inexact,
                    compare_inexact,
                };
                self.num_nic_atomic(loc, &a, snapshot.as_ref());
                self.nic_publish(dst.0, dst.1, actor, &vc, true);
                let ns = self.nic.as_mut().unwrap();
                if !ns.atomics.contains(&(dst.0, dst.1)) {
                    ns.atomics.push((dst.0, dst.1));
                    ns.atomics.sort_unstable();
                }
                // The fetched value returns to the local buffer.
                let to = Loc::Global {
                    alloc: sink.0,
                    off: sink.1,
                };
                self.access(actor, &vc, pc, to, 8, Some(&old));
                self.num_copy_snapshot(to, &old, snapshot.as_ref(), pc);
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
            self.nic.as_mut().unwrap().note_acc(NicAcc {
                actor,
                alloc: qc.cq.0 as u32,
                off,
                n: 64,
                addr: off,
                kind: NicAccKind::Cqe,
            });
            self.access_x(actor, &vc, pc, loc, 64, Some(&cqe), STRONG_SYS);
            // N-NUM-TRACK: the CQE words hold no tracked value (MISS 224).
            for k in (0..64).step_by(4) {
                self.num_forget(Loc::Global {
                    alloc: qc.cq.0,
                    off: off + k,
                });
            }
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

    /// The NIC's strong write at (alloc, off) as a synchronization source releasing `vc`. A NIC
    /// atomic (`rmw`) continues the release sequence of the write it reads as a GPU
    /// read-modify-write does (`sync_publish`).
    fn nic_publish(&mut self, alloc: usize, off: u64, actor: u32, vc: &VC, rmw: bool) {
        let key = (1u8, alloc as u32, off);
        let mut rel = vc.clone();
        let mut cont = Vec::new();
        let mut cont_join = VC::new();
        if rmw {
            if let Some(old) = self.syncs.get(&key) {
                // ASSUME(S-NIC-ATOMIC), ASSUME(S-SCOPE): the NIC atomic, a strong system-scope
                // read-modify-write by its WQE's actor (on the GPU of the thread that rang the
                // doorbell, in no CTA or cluster: `nic_apart`, MISS 193), continues the release
                // sequence of the write it reads only when the two are morally strong (PTX ISA 9.5
                // section 8.9.2): the write's scope includes the NIC (`.sys`, or `.gpu` on the
                // doorbell thread's GPU), and at the strict PTX level the write is one location
                // with the atomic's 8-byte word (`overlap_with_nic`, MISS 194). The write's own
                // release pattern and those it continued keep their first operations' threads and
                // scopes (section 8.9.4, `SyncCell::cont`), as `sync_publish` keeps them. Under the
                // hardware ordering every strong pair is morally strong and every release is at
                // system scope (E-HW-L1). Until 2026-11-12 the NIC atomic continued every write's
                // sequence and joined the write's own release into its own, at system scope.
                if self.launch.hw_order
                    || self.scopes_mutual(old.writer, strong_code(old.scope), actor, STRONG_SYS)
                        && self.overlap_at(old.writer, old.pc, actor, u32::MAX, Some((alloc, off)))
                            != crate::exec::Overlap::Partial
                {
                    if self.launch.hw_order {
                        vc_join(&mut rel, &old.rel);
                        for c in &old.cont {
                            vc_join(&mut rel, &c.rel);
                        }
                    } else {
                        cont = old.cont.clone();
                        cont_join = old.cont_join.clone();
                        SyncCell::add_cont(
                            &mut cont,
                            &mut cont_join,
                            old.writer,
                            &old.rel,
                            old.rel_scope,
                        );
                    }
                }
            }
        }
        self.syncs.insert(
            key,
            SyncCell {
                writer: actor,
                scope: Scope::Sys,
                pc: u32::MAX,
                rel,
                rel_scope: Scope::Sys,
                cont,
                cont_join,
                rr: VC::new(),
                own: 0,
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

#[cfg(test)]
mod elem_tests {
    use super::*;

    fn acc(actor: u32, off: u64, n: u64, addr: u64, kind: NicAccKind) -> NicAcc {
        NicAcc {
            actor,
            alloc: 0,
            off,
            n,
            addr,
            kind,
        }
    }

    /// MISS 194: `NicState::elem` of RDMA write data names the chunk `complete_nic` writes the
    /// byte with (naturally aligned 8 bytes, else 4, else a plain byte), for every start
    /// address and length up to 40 bytes.
    #[test]
    fn words_elem_matches_the_chunks() {
        for base in 0..16u64 {
            for n in 1..40u64 {
                let mut ns = NicState::default();
                // The allocation offset differs from the address: chunks follow the address.
                ns.note_acc(acc(7, 100 + base, n, 0x1000 + base, NicAccKind::Words));
                let mut k = 0u64;
                while k < n {
                    let a = 0x1000 + base + k;
                    let w = if a.is_multiple_of(8) && n - k >= 8 {
                        8
                    } else if a.is_multiple_of(4) && n - k >= 4 {
                        4
                    } else {
                        1
                    };
                    for i in 0..w {
                        assert_eq!(
                            ns.elem(7, 0, 100 + base + k + i),
                            NicElem::Exact(100 + base + k, w as u8),
                            "base {base} n {n} byte {}",
                            k + i
                        );
                    }
                    k += w;
                }
                assert_eq!(ns.elem(7, 0, 100 + base + n), NicElem::Unknown);
                assert_eq!(ns.elem(7, 1, 100 + base), NicElem::Unknown);
                assert_eq!(ns.elem(8, 0, 100 + base), NicElem::Unknown);
            }
        }
    }

    /// A CQE's elements are its fields at bytes 56-59, 60-61, 62 and 63; other bytes are not
    /// decided. Records stay sorted by actor; an atomic and a flag are one element.
    #[test]
    fn cqe_fields_and_records() {
        let mut ns = NicState::default();
        ns.note_acc(acc(9, 64, 64, 64, NicAccKind::Cqe));
        ns.note_acc(acc(3, 8, 8, 0x2008, NicAccKind::Atomic));
        ns.note_acc(acc(9, 256, 4, 0x3000, NicAccKind::Flag));
        assert_eq!(
            ns.acc.iter().map(|x| x.actor).collect::<Vec<_>>(),
            [3, 9, 9]
        );
        assert!(ns.has_acc(3) && ns.has_acc(9) && !ns.has_acc(4));
        for (b, e) in [
            (56, NicElem::Field(120, 4)),
            (59, NicElem::Field(120, 4)),
            (60, NicElem::Field(124, 2)),
            (61, NicElem::Field(124, 2)),
            (62, NicElem::Field(126, 1)),
            (63, NicElem::Field(127, 1)),
            (0, NicElem::Unknown),
            (55, NicElem::Unknown),
        ] {
            assert_eq!(ns.elem(9, 0, 64 + b), e, "CQE byte {b}");
        }
        assert_eq!(ns.cqe_at(9, 0, 64), Some(64));
        assert_eq!(ns.cqe_at(9, 0, 127), Some(64));
        assert_eq!(ns.cqe_at(9, 0, 256), None);
        assert_eq!(ns.cqe_at(3, 0, 8), None);
        assert_eq!(ns.elem(3, 0, 12), NicElem::Exact(8, 8));
        assert_eq!(ns.elem(9, 0, 258), NicElem::Exact(256, 4));
    }
}
