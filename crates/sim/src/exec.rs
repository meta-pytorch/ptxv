/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Instruction semantics and transitions.
//!
//! A *transition* runs one warp-split (threads of one warp at the same pc) from a visible
//! instruction up to, but not including, its next visible instruction, or until it
//! diverges or blocks. Visible instructions are the scheduling points; plain memory
//! accesses are checked for races with vector clocks instead (DRF-SC, docs/design.md §3.1).

use std::collections::BTreeMap;
use std::sync::Arc;

use ptxv_ir::*;

use crate::fp;
use crate::machine::*;

/// A simulator error is either a fault of the kernel (bad address, ...) or a limit of the
/// model ("... not modeled ..."), which makes the verdict inconclusive, not the kernel wrong.
fn fault_kind(e: &str) -> &'static str {
    if e.contains("not modeled") || e.contains("unmodeled") {
        "unmodeled"
    } else if e.contains("after all its threads exited") {
        // S-CLUSTER-LIFE: the same rule `exit_checks` reports order-independently.
        "violation"
    } else if e.contains("used before mbarrier.init") {
        // S-MBAR: any operation on an uninitialized mbarrier is protocol misuse, as a
        // try_wait on one is (`mbar-uninit`).
        "violation"
    } else {
        "fault"
    }
}

/// Shared-memory swizzle (CUTLASS Swizzle<B,4,3>): XOR the 16-byte chunk index (address
/// bits [4, 4+B)) with the row-of-128-bytes index (bits [7, 7+B)); B = 3/2/1 for a 128/64/32-byte
/// span. Applied to the CTA shared-memory address, not the offset within the box: a TMA
/// start that is 128-byte but not (8 rows x span)-aligned (a multicast slice) takes the phase
/// of its address (GB300, probes/tma_mcast_swizzle). ASSUME(S-TMA)
pub fn swz(addr: u32, span: u32) -> u32 {
    let b = match span {
        128 => 7,
        64 => 3,
        32 => 1,
        _ => 0,
    };
    addr ^ (((addr >> 7) & b) << 4)
}

/// Shared-memory byte offset (before the swizzle) of box element `e` (row-major over the box,
/// innermost dimension first) of a TMA box. Rows of the innermost dimension are contiguous;
/// with a swizzle, each row is padded to the swizzle span when the inner box is narrower
/// (GB300, probes/tma_narrow_box: an 8 x 16 bf16 box with SWIZZLE_64B puts row r at 64r, not
/// 16r; loads and stores alike, every swizzle and inner width). ASSUME(S-TMA)
pub fn tma_box_off(e: u64, box0: u32, esize: u32, span: u32) -> u32 {
    let inner = box0 * esize;
    let pitch = if span > inner { span } else { inner };
    ((e / box0 as u64) * pitch as u64 + (e % box0 as u64) * esize as u64) as u32
}

/// Upper bound on instructions in one transition (catches local infinite loops).
/// Generic address of the (unmodeled) local-memory window, c[0x0][0x2f8].
const LOCAL_GENERIC_BASE: u64 = 0x7d00_0000_0000;

const MAX_LOCAL_STEPS: usize = 10_000_000;

/// `MAX_LOCAL_STEPS`, or `PTXV_MAX_LOCAL_STEPS` (debugging: find a runaway loop quickly).
fn max_local_steps() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PTXV_MAX_LOCAL_STEPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MAX_LOCAL_STEPS)
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    Shared {
        cta: u16,
        off: u32,
    },
    Global {
        alloc: usize,
        off: u64,
    },
    /// Kernel parameters of grid `grid`.
    Param {
        grid: u16,
        off: u32,
    },
}

enum Flow {
    Next,
    Stay,
    Jump(u32),
}

/// A 4-bit tensor map's coordinate 0 counts elements, two per byte; the map is in bytes.
// ASSUME(S-TMA): see ASSUMPTIONS.md
fn elem4_coord(tm: &TmapDesc, cs: &mut [i64]) -> Result<(), String> {
    if tm.elem4 {
        if cs[0] % 2 != 0 {
            return Err(format!(
                "TMA coordinate {} of a 4-bit tensor map is odd: not modeled",
                cs[0]
            ));
        }
        cs[0] /= 2;
    }
    Ok(())
}

impl Machine {
    // ---------------------------------------------------------------- operand access

    // ASSUME(S-CBANK): see ASSUMPTIONS.md
    pub fn cbank(&self, t: usize, off: u32, bytes: u8) -> Result<u64, String> {
        let l = self.launch.g(self.threads[t].cta as u32);
        if off >= l.param_cbank {
            let o = (off - l.param_cbank) as usize;
            let mut v = 0u64;
            for i in 0..bytes as usize {
                v |= (*l.params.get(o + i).ok_or("cbank read past params")? as u64) << (8 * i);
            }
            return Ok(v);
        }
        // The per-cluster-size table: ptxas reads cluster special registers at
        // `c[0x0][x - 0xa0 * %cluster_nctarank]` (SR_CgaSize). probes/cluster_sregs (GB300):
        // x = 0x2a0 + 4d: %cluster_nctaid.d; 0x2b0 + 4d: f32 1 / %cluster_nctaid.d (%clusterid
        // = trunc(%ctaid * it)); 0x2c0 + 4d: %nclusterid.d; 0x2cc: %cluster_nctarank.
        let cl = l.cluster;
        let size = cl.iter().product::<u32>();
        let x = off.wrapping_add(0xa0 * size);
        if bytes == 4 && (0x2a0..0x2d0).contains(&x) && x.is_multiple_of(4) {
            let d = ((x & 0xf) / 4) as usize;
            let v = match x & !0xf {
                0x2a0 if d < 3 => Some(cl[d] as u64),
                // Rounded up, so that trunc(k * cl * it) = k for every k in range.
                0x2b0 if d < 3 => Some(
                    crate::fp::f64_to_f32(1.0 / cl[d] as f64, ptxv_ir::Rnd::Rp).to_bits() as u64,
                ),
                0x2c0 if d < 3 => Some((l.grid[d] / cl[d]) as u64),
                0x2c0 => Some(size as u64),
                _ => None,
            };
            if let Some(v) = v {
                return Ok(v);
            }
        }
        Ok(match off {
            // %ntid / %nctaid (sm_103a, read off ptxas output; the old 0x0 / 0xc guesses
            // were never exercised).
            0x360 | 0x364 | 0x368 => l.block[((off - 0x360) / 4) as usize] as u64,
            0x370 | 0x374 | 0x378 => l.grid[((off - 0x370) / 4) as usize] as u64,
            cbank::PARAM_GENERIC_BASE => PARAM_BASE,
            // Generic address of constant bank 0 itself (cuBLAS: `c[0x0][0x110] + 0x6c0` = the
            // tensor map at cbank 0x6c0, i.e. param offset 0x340). Consistent with 0x348.
            0x110 => PARAM_BASE - l.param_cbank as u64,
            0x358 => 0, // global memory descriptor for `desc[URx]`: not modeled, ignored
            // %is_explicit_cluster (probes/cluster_flag: 1 with .reqnctapercluster, else 0);
            // ptxas skips barrier.cluster when it is not 1.
            0x36c => l.explicit_cluster as u64,
            0x37c => 0xfffc, // initial stack pointer (no local memory modeled)
            // Generic address of the local-memory window (`cvta.local`; DeepEP turns its
            // stack into a vprintf argument pointer). Unmapped: a generic access through it
            // is reported. ASSUME(S-CBANK)
            0x2f8 => LOCAL_GENERIC_BASE & 0xffff_ffff,
            0x2fc => LOCAL_GENERIC_BASE >> 32,
            _ => {
                return Err(format!(
                    "read of unmodeled driver constant c[0x0][{off:#x}]"
                ));
            }
        })
    }

    fn sreg(&self, t: usize, s: SReg) -> u64 {
        let th = &self.threads[t];
        let cta = th.cta as u32;
        let g = self.launch.g(cta);
        (match s {
            SReg::Tid(d) => th.tid[d as usize],
            SReg::NTid(d) => g.block[d as usize],
            SReg::CtaId(d) => self.launch.ctaid(cta)[d as usize],
            SReg::NCtaId(d) => g.grid[d as usize],
            SReg::LaneId => th.lane,
            SReg::WarpId => (th.tid[0] + th.tid[1] * g.block[0]) / 32,
            SReg::ClusterCtaRank => self.launch.cluster_of(cta).1,
            SReg::ClusterNCtaRank => g.cluster_size(),
            SReg::LaneMaskLt => (1u32 << th.lane) - 1,
            SReg::LaneMaskLe => ((2u64 << th.lane) - 1) as u32,
            SReg::LaneMaskGe => !((1u32 << th.lane) - 1),
            SReg::LaneMaskGt => !(((2u64 << th.lane) - 1) as u32),
            SReg::LaneMaskEq => 1u32 << th.lane,
            // ASSUME(S-DSMEM): see ASSUMPTIONS.md
            SReg::SharedAddr(off) => (self.launch.cluster_of(cta).1 << 24) | off,
            SReg::SharedWindowHi => (SHARED_GENERIC_BASE >> 32) as u32,
        }) as u64
    }

    pub fn rd(&self, t: usize, v: &Val) -> Result<u64, String> {
        Ok(match v {
            Val::Imm(x) => *x,
            Val::Const(off, b) => self.cbank(t, *off, *b)?,
            Val::SReg(s) => self.sreg(t, *s),
            Val::Reg(r, n) => {
                let th = &self.threads[t];
                match r.file {
                    File::V => th.regs[r.idx as usize],
                    File::R => {
                        let g = |i: u16| {
                            if i >= 255 {
                                0
                            } else {
                                th.regs[i as usize] & 0xffff_ffff
                            }
                        };
                        let lo = g(r.idx);
                        if *n >= 2 {
                            lo | (g(r.idx + 1) << 32)
                        } else {
                            lo
                        }
                    }
                    File::UR => {
                        let w = &self.warps[th.warp as usize];
                        let g = |i: u16| {
                            if i >= 255 {
                                0
                            } else {
                                w.ur[i as usize] & 0xffff_ffff
                            }
                        };
                        let lo = g(r.idx);
                        if *n >= 2 {
                            lo | (g(r.idx + 1) << 32)
                        } else {
                            lo
                        }
                    }
                    _ => return Err("predicate used as a value".into()),
                }
            }
        })
    }

    /// Undefined bits of slot `s` of the value being written (S-INIT): all of them while a
    /// load of never-written shared memory runs, those of `vmask` while an operation on
    /// partly undefined registers runs.
    fn slot_umask(&self, s: u16) -> u32 {
        match self.vmask {
            _ if self.stale.is_some() => !0,
            Some(m) if s < 2 => (m >> (32 * s)) as u32,
            Some(m) if m != 0 => !0,
            _ => 0,
        }
    }

    /// Undefined bits of a 32-bit register (S-INIT): all of them until it is written; a
    /// written R / UR register keeps the undefined bits of its value in bits 32..63.
    pub fn reg_umask(&self, t: usize, r: Reg) -> u32 {
        let th = &self.threads[t];
        if r.idx >= 255 && matches!(r.file, File::R | File::UR) {
            return 0;
        }
        match r.file {
            File::V if th.written[r.idx as usize / 64] >> (r.idx % 64) & 1 == 1 => 0,
            File::R if th.written[r.idx as usize / 64] >> (r.idx % 64) & 1 == 1 => {
                (th.regs[r.idx as usize] >> 32) as u32
            }
            File::UR => {
                let w = &self.warps[th.warp as usize];
                if w.ur_written[r.idx as usize / 64] >> (r.idx % 64) & 1 == 1 {
                    (w.ur[r.idx as usize] >> 32) as u32
                } else {
                    !0
                }
            }
            File::V | File::R => !0,
            _ => 0,
        }
    }

    /// Write `v` to `r`; SASS files take `slots` 32-bit registers.
    pub fn wr(&mut self, t: usize, r: Reg, v: u64, slots: u8) {
        match r.file {
            File::V => {
                let stale = self.slot_umask(0) != 0 || self.slot_umask(1) != 0;
                let th = self.threads[t].make_mut();
                th.regs[r.idx as usize] = v;
                if stale {
                    th.written[r.idx as usize / 64] &= !(1 << (r.idx % 64));
                } else {
                    th.written[r.idx as usize / 64] |= 1 << (r.idx % 64);
                }
            }
            File::R => {
                let um: Vec<u32> = (0..slots.max(1) as u16)
                    .map(|s| self.slot_umask(s))
                    .collect();
                let th = self.threads[t].make_mut();
                for s in 0..slots.max(1) as u16 {
                    let i = r.idx + s;
                    let u = um[s as usize];
                    if i < 255 {
                        th.regs[i as usize] = (v >> (32 * s)) & 0xffff_ffff
                            | if u != !0 { (u as u64) << 32 } else { 0 };
                        if u == !0 {
                            th.written[i as usize / 64] &= !(1 << (i % 64));
                        } else {
                            th.written[i as usize / 64] |= 1 << (i % 64);
                        }
                    }
                }
            }
            File::UR => {
                let um: Vec<u32> = (0..slots.max(1) as u16)
                    .map(|s| self.slot_umask(s))
                    .collect();
                let w = self.threads[t].warp as usize;
                let wp = self.warps[w].make_mut();
                for s in 0..slots.max(1) as u16 {
                    let i = r.idx + s;
                    let u = um[s as usize];
                    if i < 255 {
                        wp.ur[i as usize] = (v >> (32 * s)) & 0xffff_ffff
                            | if u != !0 { (u as u64) << 32 } else { 0 };
                        if u == !0 {
                            wp.ur_written[i as usize / 64] &= !(1 << (i % 64));
                        } else {
                            wp.ur_written[i as usize / 64] |= 1 << (i % 64);
                        }
                    }
                }
            }
            File::P | File::UP => self.wp(t, r, v != 0),
        }
    }

    pub fn prd(&self, t: usize, p: Pred) -> bool {
        let v = match p.reg {
            None => true,
            Some(r) => match r.file {
                File::P => {
                    if r.idx == 7 && self.k.front == Front::Sass {
                        true
                    } else {
                        let th = &self.threads[t];
                        (th.preds[(r.idx / 64) as usize] >> (r.idx % 64)) & 1 == 1
                    }
                }
                File::UP => {
                    if r.idx == 7 {
                        true
                    } else {
                        (self.warps[self.threads[t].warp as usize].up >> r.idx) & 1 == 1
                    }
                }
                _ => false,
            },
        };
        v != p.neg
    }

    pub fn wp(&mut self, t: usize, r: Reg, b: bool) {
        match r.file {
            File::P => {
                if r.idx == 7 && self.k.front == Front::Sass {
                    return;
                }
                let th = self.threads[t].make_mut();
                let (w, bit) = ((r.idx / 64) as usize, r.idx % 64);
                th.preds[w] = (th.preds[w] & !(1 << bit)) | ((b as u64) << bit);
            }
            File::UP => {
                if r.idx == 7 {
                    return;
                }
                let w = self.threads[t].warp as usize;
                let wp = self.warps[w].make_mut();
                wp.up = (wp.up & !(1 << r.idx)) | ((b as u8) << r.idx);
            }
            _ => {}
        }
    }

    pub(crate) fn addr(&self, t: usize, a: &Addr) -> Result<u64, String> {
        let mut v = a.off as u64;
        if let Some(b) = &a.base {
            v = v.wrapping_add(self.rd(t, b)?);
        }
        if let Some(b) = &a.base2 {
            v = v.wrapping_add(self.rd(t, b)?);
        }
        Ok(v)
    }

    // ---------------------------------------------------------------- memory

    pub fn resolve(&self, t: usize, space: Space, a: u64, n: usize) -> Result<Loc, String> {
        let cta = self.threads[t].cta;
        // ASSUME(S-SWIN): a generic address in the shared window is its shared::cluster address.
        let (space, a) = match space {
            Space::Generic if a >> 32 == SHARED_GENERIC_BASE >> 32 => {
                (Space::SharedCluster, a & 0xffff_ffff)
            }
            _ => (space, a),
        };
        let loc = match space {
            Space::Shared => {
                // GB300 (probes/reserved_smem): in a cluster launch a shared::cta access whose
                // rank byte is not the CTA's own traps (illegal instruction).
                let own = self.launch.cluster_of(cta as u32).1 as u64;
                if self.launch.g(cta as u32).explicit_cluster && a >> 24 != own {
                    return Err(format!(
                        "shared::cta address {a:#x} names CTA rank {} but the CTA is rank {own} (illegal instruction on GB300)",
                        a >> 24
                    ));
                }
                Loc::Shared {
                    cta,
                    off: (a & 0xff_ffff) as u32,
                }
            }
            Space::SharedCluster => {
                let target = self.rank_cta(cta, a)?;
                if target != cta {
                    self.check_alive(target, t as u32)?;
                }
                Loc::Shared {
                    cta: target,
                    off: (a & 0xff_ffff) as u32,
                }
            }
            Space::Param => Loc::Param {
                grid: self.launch.grid_index(cta as u32) as u16,
                off: a as u32,
            },
            Space::Global | Space::Generic => {
                if space == Space::Generic
                    && a >= PARAM_BASE
                    && a < PARAM_BASE + self.launch.g(cta as u32).params.len() as u64
                {
                    Loc::Param {
                        grid: self.launch.grid_index(cta as u32) as u16,
                        off: (a - PARAM_BASE) as u32,
                    }
                } else {
                    let i = self
                        .launch
                        .allocs
                        .iter()
                        .position(|al| {
                            a >= al.base && a + n as u64 <= al.base + al.init.len() as u64
                        })
                        .ok_or_else(|| {
                            format!("global address {a:#x} (+{n}) is outside every allocation")
                        })?;
                    Loc::Global {
                        alloc: i,
                        off: a - self.launch.allocs[i].base,
                    }
                }
            }
        };
        if let Loc::Shared { cta, off } = loc {
            let smem = self.launch.g(cta as u32).smem_bytes;
            if off as usize + n > smem as usize {
                return Err(format!(
                    "shared access at {off:#x} (+{n}) past the {smem} B window"
                ));
            }
        }
        Ok(loc)
    }

    fn space_name(loc: &Loc) -> &'static str {
        match loc {
            Loc::Shared { .. } => "smem",
            Loc::Global { .. } => "gmem",
            Loc::Param { .. } => "param",
        }
    }

    /// Read or write `n` bytes as `actor` (clock `vc`) with race checking.
    pub fn access(
        &mut self,
        actor: u32,
        vc: &VC,
        pc: u32,
        loc: Loc,
        n: usize,
        write: Option<&[u8]>,
    ) -> Vec<u8> {
        self.access_x(actor, vc, pc, loc, n, write, 0)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn access_x(
        &mut self,
        actor: u32,
        vc: &VC,
        pc: u32,
        loc: Loc,
        n: usize,
        write: Option<&[u8]>,
        strong: u8,
    ) -> Vec<u8> {
        let atomic = strong != 0;
        // E-POLL-ANY: the read of a value-insensitive poll (`vip_ok`).
        let vip = self.vip_read == 1 && write.is_none() && atomic && vip_where(loc).is_some();
        if atomic && self.wlog.is_on() {
            crate::wellsync::strong_access(self, actor, vc, pc, loc, n, write); // wellsync.rs
        }
        if let (Loc::Global { alloc, off }, Some(g)) = (loc, self.gfoot.as_ref()) {
            let mine = g.cluster.is_none_or(|c| {
                let cta = self.threads[self.actor_thread(actor)].cta as u32;
                self.launch.cluster_of(cta).0 == c
            });
            if mine {
                if self.red_access {
                    g.add_red(alloc, off, n);
                } else {
                    g.add(alloc, off, n, write.is_some());
                }
            }
        }
        if let (Loc::Global { alloc, off }, Some(g)) = (loc, self.gown.as_mut()) {
            g.add(alloc, off, n, actor, write.is_some());
        }
        let mut out = Vec::with_capacity(n);
        if let Loc::Param { grid, off } = loc {
            let params = &self.launch.grids[grid as usize].params;
            for i in 0..n {
                out.push(*params.get(off as usize + i).unwrap_or(&0));
            }
            if write.is_some() {
                self.report(
                    "violation",
                    format!("param-write:{pc}"),
                    format!(
                        "store to kernel parameter space at {}",
                        self.k.describe(pc as usize)
                    ),
                );
            }
            return out;
        }
        // Strong accesses also as synchronization (spaces 10/11): plain-vs-strong conflicts are
        // races, found by vector clocks in either order (E-DRF), strong-vs-strong are not. Two
        // strong reads commute (memory, the synchronization source and the per-actor reader
        // shadow are unchanged by either), so a strong read is recorded as a read.
        if atomic && !vip {
            let w = write.is_some();
            match loc {
                Loc::Shared { cta, off } => {
                    self.rec(10, cta as u32, off as u64, off as u64 + n as u64, w)
                }
                Loc::Global { alloc, off } => self.rec(11, alloc as u32, off, off + n as u64, w),
                Loc::Param { .. } => {}
            }
        }
        match loc {
            _ if vip => {}
            Loc::Shared { cta, off } => self.rec(
                0,
                cta as u32,
                off as u64,
                off as u64 + n as u64,
                write.is_some(),
            ),
            Loc::Global { alloc, off } => {
                self.rec(1, alloc as u32, off, off + n as u64, write.is_some())
            }
            Loc::Param { .. } => {}
        }
        let me = Epoch {
            actor,
            clock: vc_get(vc, actor),
            pc,
            strong,
        };
        let mut races: Vec<(Epoch, &'static str, u64)> = Vec::new();
        let mut uninit: Option<usize> = None;
        let (mem, base): (&mut Mem, u64) = match loc {
            Loc::Shared { cta, off } => (&mut self.smem[cta as usize], off as u64),
            Loc::Global { alloc, off } => (&mut self.gmem[alloc], off),
            Loc::Param { .. } => unreachable!(),
        };
        let shared = matches!(loc, Loc::Shared { .. });
        // Numerics: a write ends any origin the bytes had (a TMA load re-sets it after); the
        // taint shadow follows the bytes.
        if let Some(ns) = self.num.as_mut() {
            if let (Loc::Shared { cta, off }, Some(_)) = (loc, write) {
                ns.forget_orig(cta, off, n);
            }
            let (key, off) = match loc {
                Loc::Shared { cta, off } => ((0u8, cta as u64), off as u64),
                Loc::Global { alloc, off } => ((1u8, alloc as u64), off),
                Loc::Param { .. } => ((2u8, 0), 0),
            };
            if key.0 != 2 {
                match write {
                    Some(_) => ns.taint.write(key, off, n as u64),
                    None => ns.taint.read(key, off, n as u64),
                }
            }
        }
        let rd_done = self.launch.read_done;
        // E-POLL-ANY: the values of watched locations this write overwrites (monotonicity).
        let vip_old: Vec<((u32, u64, u8), u64)> =
            match (write.is_some(), vip_where(loc), self.vip.as_ref()) {
                (true, Some((wh, off)), Some(vl))
                    if vl.allocs.load(std::sync::atomic::Ordering::Relaxed) >> (wh % 64) & 1
                        == 1 =>
                {
                    let g = vl.values.lock().unwrap();
                    g.keys()
                        .filter(|&&(a, o, nb)| a == wh && o < off + n as u64 && off < o + nb as u64)
                        .map(|&(a, o, nb)| {
                            let v = (0..nb as usize).fold(0u64, |v, i| {
                                v | (*mem.bytes.get(o as usize + i) as u64) << (8 * i)
                            });
                            ((a, o, nb), v)
                        })
                        .collect()
                }
                _ => vec![],
            };
        // Page by page: one copy-on-write check per run of bytes.
        let mut i = 0;
        while i < n {
            let a0 = base as usize + i;
            let k = mem.sh.for_run(a0, n - i, |a, s| {
                if let Some(w) = s.w {
                    if w.actor != actor && !w.hb(vc) {
                        let kind = if write.is_some() {
                            "write-write"
                        } else {
                            "write-read"
                        };
                        races.push((w, kind, a as u64));
                    }
                }
                if write.is_some() {
                    for r in &s.r {
                        if r.actor != actor
                            && !(r.hb(vc) || rd_done && r.clock <= vc_get(vc, r.actor | RR_BIT))
                        {
                            races.push((*r, "read-write", a as u64));
                        }
                    }
                    s.w = Some(me);
                    s.r.clear();
                } else if vip {
                } else {
                    // ASSUME(S-INIT): shared memory outside the driver-initialized reserved
                    // prefix holds stale data until written (B02: the previous launch's tile).
                    if s.w.is_none() && shared && a >= SMEM_RESERVED && uninit.is_none() {
                        uninit = Some(a);
                        if let Some(st) = self.stale.as_mut() {
                            st.get_or_insert(a as u32);
                        }
                    }
                    s.r.insert(me);
                }
            });
            if let Some(wr) = write {
                mem.bytes.run_mut(a0)[..k].copy_from_slice(&wr[i..i + k]);
            }
            out.extend_from_slice(&mem.bytes.run(a0)[..k]);
            i += k;
        }
        if let Some((wh, off)) = vip_where(loc) {
            if vip {
                let mut map = (*self.vipr).clone();
                for i in 0..n as u64 {
                    epoch_insert(map.entry((wh, off + i)).or_default(), me);
                }
                self.vipr = std::sync::Arc::new(map);
            } else if write.is_some() {
                // Every write after a value-insensitive poll read must be ordered after it.
                if !self.vipr.is_empty() {
                    for i in 0..n as u64 {
                        if let Some(v) = self.vipr.get(&(wh, off + i)) {
                            for e in v {
                                if e.actor != actor && !self.read_before(e, vc) {
                                    races.push((*e, "read-write", off + i));
                                }
                            }
                        }
                    }
                }
                if let Some(vl) = self.vip.as_ref() {
                    if vl.allocs.load(std::sync::atomic::Ordering::Relaxed) >> (wh % 64) & 1 == 1 {
                        let mem = match loc {
                            Loc::Shared { cta, .. } => &self.smem[cta as usize],
                            Loc::Global { alloc, .. } => &self.gmem[alloc],
                            Loc::Param { .. } => unreachable!(),
                        };
                        let mut g = vl.values.lock().unwrap();
                        for (&(a, o, nb), set) in g.iter_mut() {
                            if a == wh && o < off + n as u64 && off < o + nb as u64 {
                                let v = (0..nb as usize).fold(0u64, |v, i| {
                                    v | (*mem.bytes.get(o as usize + i) as u64) << (8 * i)
                                });
                                if set.insert(v) {
                                    vl.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                // A write that lowers the value: the location is not monotone.
                                let lowers =
                                    vip_old.iter().any(|&(k, old)| k == (a, o, nb) && v < old);
                                if lowers {
                                    let mut nm = vl.nonmono.lock().unwrap();
                                    if nm.insert((a, o, nb)) {
                                        vl.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(a) = uninit.filter(|_| self.stale.is_none()) {
            let msg = format!(
                "{} at {} reads shared memory byte {a:#x}, which nothing wrote in this launch (on hardware: stale data from an earlier kernel)",
                self.actor_name(actor),
                self.k.describe(pc as usize)
            );
            self.report("violation", format!("uninit-smem:{pc}"), msg);
        }
        // ASSUME(E-LAUNCH-GROUPS): accesses of earlier launch groups this one must follow.
        if let (Loc::Global { alloc, off }, Some(xg)) = (loc, self.xg.as_ref()) {
            for i in 0..n as u64 {
                if let Some(v) = xg.get(&(alloc as u32, off + i)) {
                    for &(e, w) in v {
                        let after = if w {
                            e.hb(vc)
                        } else {
                            self.read_before(&e, vc)
                        };
                        if (w || write.is_some()) && e.actor != actor && !after {
                            let kind = match (w, write.is_some()) {
                                (true, true) => "write-write",
                                (true, false) => "write-read",
                                _ => "read-write",
                            };
                            races.push((e, kind, off + i));
                        }
                    }
                }
            }
        }
        // Strong accesses that are morally strong with each other do not race (S-SCOPE).
        races
            .retain(|(other, _, _)| !self.morally_strong(other.actor, other.strong, actor, strong));
        // One finding per (instruction pair, kind); report the first byte.
        let mut seen = std::collections::BTreeSet::new();
        for (other, kind, a) in races {
            if !seen.insert((other.pc, kind)) {
                continue;
            }
            // Keyed by the earlier access: one finding per conflicting write/read site, even
            // when ptxas unrolled the other side into many instructions.
            // `race-ldsts`: both sides are SASS LDS/STS of a thread, which the strong model
            // does not count as a race (S-SMEM-STRONG, smem.rs re-checks).
            let ldsts = |x: u32, at: u32| {
                self.k.front == Front::Sass
                    && (x as usize) < self.threads.len()
                    && matches!(
                        self.k.code[at as usize].op,
                        Op::Ld {
                            space: Space::Shared,
                            ..
                        } | Op::St {
                            space: Space::Shared,
                            ..
                        }
                    )
            };
            let tag = if ldsts(other.actor, other.pc) && ldsts(actor, pc) {
                "race-ldsts"
            } else {
                "race"
            };
            let sig = format!("{tag}:{}:{}:{}", Self::space_name(&loc), other.pc, kind);
            let msg = format!(
                "{kind} race on {} byte {a:#x}: {} at {} and {} at {} are not ordered by happens-before",
                Self::space_name(&loc),
                self.actor_name(other.actor),
                self.k.describe(other.pc as usize),
                self.actor_name(actor),
                self.k.describe(pc as usize)
            );
            self.report_pcs("race", sig, msg, vec![other.pc, pc]);
        }
        out
    }

    // ---------------------------------------------------------------- scheduling queries

    /// Group runnable threads into warp-splits: (warp, pc) -> members.
    pub fn splits(&self) -> Vec<(u32, u32, Vec<usize>)> {
        let mut m: std::collections::BTreeMap<(u32, u32), Vec<usize>> = Default::default();
        for (i, t) in self.live_threads() {
            if t.st == St::Run {
                m.entry((t.warp, t.pc)).or_default().push(i);
            }
        }
        m.into_iter().map(|((w, pc), v)| (w, pc, v)).collect()
    }

    /// An mbarrier that was initialized with a plain 8-byte store of the encoded state word
    /// (ptxas's divergent-path lowering of mbarrier.init: STS.64 + SYNCS.CCTL.IVALL).
    /// Word layout observed from ptxas: low = (2^20 - n) << 1, high = (2^20 - n) << 11.
    // ASSUME(S-MBAR-MEM): see ASSUMPTIONS.md
    pub fn mbar_decode_mem(&self, cta: u16, off: u32) -> Option<Mbar> {
        let mem = &self.smem[cta as usize];
        if off as usize + 8 > mem.bytes.len() {
            return None;
        }
        let mut w = 0u64;
        for i in 0..8 {
            w |= (*mem.bytes.get(off as usize + i) as u64) << (8 * i);
        }
        let (lo, hi) = (w as u32, (w >> 32) as u32);
        let x = lo >> 1;
        if lo & 1 != 0 || x == 0 || x > 0xf_ffff || hi & 0x7fff_ffff != x << 11 {
            return None;
        }
        let phase = hi >> 31; // initial phase parity (probes/mbar_phase1)
        let init = mem.sh.get(off as usize).w?;
        let n = 0x10_0000 - x;
        let mut v = VC::new();
        vc_tick(&mut v, init.actor);
        for _ in 1..init.clock {
            vc_tick(&mut v, init.actor);
        }
        Some(Mbar {
            expected: n,
            pending: n,
            tx: 0,
            phase,
            cur: VC::new(),
            done: if phase == 1 {
                [v, VC::new()]
            } else {
                [VC::new(), v]
            },
            init,
            atomics: vec![],
        })
    }

    fn mbar_materialize(&mut self, cta: u16, off: u32) {
        if !self.mbars.contains_key(&(cta, off)) {
            if let Some(m) = self.mbar_decode_mem(cta, off) {
                self.mbars.insert((cta, off), m);
            }
        }
    }

    /// Would this thread's try_wait at its pc succeed? Ok(None) = would block.
    // ASSUME(E-SPIN, S-MBAR): see ASSUMPTIONS.md (E-SPIN is checked statically in spin.rs)
    pub fn try_wait_result(&self, t: usize) -> Result<Option<VC>, String> {
        let k = self.k.clone();
        let inst = &k.code[self.threads[t].pc as usize];
        let Op::MbarTryWait {
            addr,
            parity,
            bit31,
            ..
        } = &inst.op
        else {
            return Ok(Some(VC::new()));
        };
        if !self.prd(t, inst.guard) {
            return Ok(Some(VC::new()));
        }
        let a = self.addr(t, addr)?;
        let Loc::Shared { cta, off } = self.resolve(t, Space::Shared, a, 8)? else {
            unreachable!()
        };
        let p = self.rd(t, parity)?;
        let p = if *bit31 { (p >> 31) & 1 } else { p & 1 } as u32;
        let decoded;
        let mb = match self.mbars.get(&(cta, off)) {
            Some(m) => Some(m),
            None => {
                decoded = self.mbar_decode_mem(cta, off);
                decoded.as_ref()
            }
        };
        match mb {
            None => Err(format!(
                "mbarrier at smem {off:#x} used before mbarrier.init"
            )),
            Some(mb) => {
                if mb.phase % 2 != p {
                    Ok(Some(mb.done[p as usize].clone()))
                } else {
                    Ok(None)
                }
            }
        }
    }

    // ASSUME(E-ALIGNED, E-ALLOC): see ASSUMPTIONS.md
    /// Every member's bulk groups older than its newest `n` have completed.
    fn bulk_wait_ready(&self, members: &[usize], n: u32) -> bool {
        members.iter().all(|&t| {
            let upto = self.threads[t].bulk_group.saturating_sub(n);
            !self.pending.iter().any(|o| {
                o.issuer == t as u32
                    && matches!(o.kind, AsyncKind::TmaStore { group, .. } if group < upto)
            })
        })
    }

    /// `cp.async.bulk.wait_group[.read] n` by thread `t`: acquires its completed groups older
    /// than the newest n.
    fn bulk_wait(&mut self, t: usize, n: u32) {
        let upto = self.threads[t].bulk_group.saturating_sub(n);
        for g in 0..upto {
            self.rec_acq(sync_key(3, t as u64, g as u64, 0));
        }
        let th = self.threads[t].make_mut();
        let mut j = VC::new();
        th.bulk_done.retain(|(g, vc)| {
            if *g < upto {
                vc_join(&mut j, vc);
                false
            } else {
                true
            }
        });
        vc_join(&mut th.vc, &j);
    }

    pub fn split_enabled(&self, members: &[usize], pc: u32) -> bool {
        let inst = &self.k.code[pc as usize];
        // PTX `.sync.aligned` instructions are convergence points: they execute once every
        // live lane of the warp has arrived (ptxas emits WARPSYNC before them). SASS `SHFL`
        // too: it has no member mask, and ptxas places it after a `BSYNC` (reconvergence,
        // which the model does not track) when lanes diverged before (rung 3a: lane 0 copies
        // tensor-map templates, then all 32 lanes shuffle).
        // PTX `vote.sync` with the full member mask (the only one the front-end accepts)
        // likewise waits for every live lane of the warp.
        if inst.op.is_warp_aligned()
            && (self.k.front == Front::Ptx || matches!(inst.op, Op::Shfl { .. }))
            || self.k.front == Front::Ptx
                && matches!(inst.op, Op::VoteAll { .. } | Op::VoteAny { .. })
        {
            let w = self.threads[members[0]].warp;
            let live = self
                .warp_threads(w)
                .filter(|&u| self.threads[u].st != St::Exited)
                .count();
            if members.len() < live {
                return false;
            }
        }
        // ASSUME(E-SPIN): a warp-level poll loop's split at its head runs only when the
        // iteration would not be a stutter step (spin::wpoll_dry).
        if let Some(&b) = self.wpolls.get(&pc) {
            // C-RING: a forced read waits for its write (wellsync/ring.rs).
            if !self.ring_ready(members, pc) {
                return false;
            }
            return crate::spin::wpoll_dry(self, members, pc, b).is_none();
        }
        if self.k.bulk_read_waits.binary_search(&(pc as usize)).is_ok()
            && !self.bulk_wait_ready(members, 0)
        {
            return false;
        }
        match &inst.op {
            // A peek (no retry shape) never blocks: it returns false while the phase is pending.
            Op::MbarTryWait { .. }
                if self.peeks.contains(&pc) && !self.bar_cfg.peekwait.contains(&pc) =>
            {
                true
            }
            Op::MbarTryWait { .. } => members
                .iter()
                .any(|&t| matches!(self.try_wait_result(t), Ok(Some(_)))),
            // ASSUME(E-SPIN): a poll loop's failed iteration changes nothing a later one does
            // not overwrite (spin::poll_loops), so the split runs only when some member's
            // iteration exits the loop: try it on a copy. Plain-load polls too (ptxas's LDS
            // for a CTA-scope acquire): the step that reaches one stops there while it fails.
            Op::Ld { .. } | Op::Atom { .. } if self.polls.contains_key(&pc) => {
                let (h, b) = self.polls[&pc];
                if !self.ring_ready(members, h) {
                    return false;
                }
                let mut m = self.clone();
                m.fp = None;
                m.num = None;
                // The body from the load to the back branch is straight-line (spin.rs).
                let mut p = pc;
                loop {
                    if m.exec(members, p).is_err() {
                        return true; // the real step reports it
                    }
                    if p == b {
                        break members.iter().any(|&t| m.threads[t].pc != h);
                    }
                    p += 1;
                }
            }
            // ASSUME(E-PDL): blocks until the previous grid completed (a no-op in the first).
            Op::GridDepWait => {
                let gi = self.launch.grid_index(self.threads[members[0]].cta as u32);
                self.launch
                    .prev_grid(gi)
                    .is_none_or(|p| self.grid_complete(p))
            }
            // ASSUME(E-ASYNC-STEP): `.read` waits for the whole store (read and write are one step).
            // Each thread waits for its own bulk groups: the split runs once all of them can.
            Op::BulkWait { n, .. } => self.bulk_wait_ready(members, *n),
            // TMEM allocation blocks while it cannot succeed (ptxas's SASS spins on failure).
            Op::TmemAlloc { ncols, .. } => {
                let t = members[0];
                match self.rd(t, ncols) {
                    Ok(n) if n >= 32 && n.is_power_of_two() && n <= 512 => self
                        .tmem_find(self.threads[t].cta, (n / 32) as u32)
                        .is_some(),
                    _ => true, // invalid size: let exec report it
                }
            }
            Op::TmemAllocRaw { units, .. } => {
                let t = members[0];
                if !members.iter().any(|&m| self.prd(m, inst.guard)) {
                    return true;
                }
                let pair = matches!(inst.op, Op::TmemAllocRaw { pair: true, .. });
                match (
                    self.rd(t, units),
                    self.alloc_group(self.threads[t].cta, pair),
                ) {
                    (Ok(n), Ok(ctas)) if n >= 1 && n.is_power_of_two() && n <= 16 => {
                        self.tmem_find_group(&ctas, n as u32).is_some()
                    }
                    _ => true,
                }
            }
            _ => true,
        }
    }

    /// Launch CTA `cta` (CLC kernels start with every CTA unlaunched, E-CLC). DPOR: writes
    /// the CTA's launch key (space 4), which a cancel of the same CTA also writes.
    pub fn launch_cta(&mut self, cta: u16) {
        self.rec(4, cta as u32, 0, 1, true);
        // ASSUME(E-PDL): a later grid's CTA launches when the previous grid allows it
        // (`grid_gate_open`): without PDL after that grid completed, and then everything it
        // did happens-before the CTA's threads.
        let gi = self.launch.grid_index(cta as u32);
        let done = match self.launch.prev_grid(gi) {
            Some(p) => {
                self.rec_grid_reads(p);
                (!self.launch.grids[gi].pdl).then(|| self.grid_done_vc_cached(p))
            }
            None => None,
        };
        self.l1_cta_launch(cta); // ASSUME(E-L1-CORES): see l1_cores.rs
        for t in 0..self.threads.len() {
            if self.threads[t].cta == cta && self.threads[t].st == St::Unlaunched {
                let th = self.threads[t].make_mut();
                th.st = St::Run;
                if let Some(d) = &done {
                    vc_join(&mut th.vc, d);
                }
            }
        }
    }

    // ---------------------------------------------------------------- multi-grid (E-PDL)

    /// Has grid `gi` completed: every thread exited and every async operation it issued done?
    pub fn grid_complete(&self, gi: usize) -> bool {
        // Completion is final; `done_vcs` is filled only once the grid completed.
        if self.done_vcs[gi].is_some() {
            return true;
        }
        let in_grid = |cta: u16| self.launch.grid_index(cta as u32) == gi;
        self.grid_threads(gi)
            .all(|t| self.threads[t].st == St::Exited)
            && !self
                .pending
                .iter()
                .any(|o| in_grid(self.threads[o.issuer as usize].cta))
    }

    /// The threads of grid `gi` (threads are ordered by CTA, and a grid's CTAs are
    /// contiguous).
    pub fn grid_threads(&self, gi: usize) -> std::ops::Range<usize> {
        let g = &self.launch.grids[gi];
        let (lo, hi) = (g.cta0, g.cta0 + g.ctas());
        let first = |c: u32| {
            let (mut a, mut b) = (0, self.threads.len());
            while a < b {
                let mid = (a + b) / 2;
                if (self.threads[mid].cta as u32) < c {
                    a = mid + 1;
                } else {
                    b = mid;
                }
            }
            a
        };
        first(lo)..first(hi)
    }

    /// `grid_done_vc` of a completed grid, computed once.
    pub fn grid_done_vc_cached(&mut self, gi: usize) -> VC {
        if let Some(v) = &self.done_vcs[gi] {
            return v.clone();
        }
        let v = self.grid_done_vc(gi);
        self.done_vcs[gi] = Some(v.clone());
        v
    }

    /// Everything grid `gi` did (its threads' final clocks and its completed async
    /// operations): what its completion makes visible.
    pub fn grid_done_vc(&self, gi: usize) -> VC {
        let in_grid = |cta: u16| self.launch.grid_index(cta as u32) == gi;
        let mut v = VC::new();
        for t in self.grid_threads(gi) {
            vc_join(&mut v, &self.threads[t].vc);
        }
        for &(cta, actor, clock, _) in &self.completed {
            if in_grid(cta) && vc_get(&v, actor) < clock {
                vc_join(&mut v, &vc_single(actor, clock));
            }
        }
        v
    }

    /// May the CTAs of grid `gi` (> 0) launch? With PDL once every CTA of the previous grid
    /// triggered (`griddepcontrol.launch_dependents`, any one thread suffices) or exited
    /// entirely (probes/pdl); without PDL once the previous grid completed.
    pub fn grid_gate_open(&self, gi: usize) -> bool {
        // ASSUME(E-PDL): a grid on a GPU no earlier grid ran on is launched independently,
        // at any time (launch skew between GPUs).
        let Some(p) = self.launch.prev_grid(gi) else {
            return true;
        };
        if !self.launch.grids[gi].pdl {
            return self.grid_complete(p);
        }
        let g = &self.launch.grids[p];
        (g.cta0..g.cta0 + g.ctas()).all(|c| {
            self.triggered[c as usize / 64] >> (c % 64) & 1 == 1
                || self
                    .threads
                    .iter()
                    .all(|t| t.cta as u32 != c || t.st == St::Exited)
        })
    }

    /// Does CTA `cta`'s grid have a successor that waits on it (gate footprints needed)?
    fn has_dependent(&self, cta: u16) -> bool {
        let gi = self.launch.grid_index(cta as u32);
        (gi + 1..self.launch.grids.len()).any(|j| self.launch.prev_grid(j) == Some(gi))
    }

    /// DPOR (space 9, id = CTA): thread `t`'s trigger (`what` = 0) or exit (1), which the next
    /// grid's launch gate and `griddepcontrol.wait` read.
    fn rec_gate(&mut self, t: usize, what: u64) {
        let cta = self.threads[t].cta;
        if self.fp.is_none() || !self.has_dependent(cta) {
            return;
        }
        let i = 2 * t as u64 + what;
        self.rec(9, cta as u32, i, i + 1, true);
    }

    /// DPOR: an async completion of a grid with a successor changes that grid's completion.
    fn rec_async_gate(&mut self, issuer: u32, actor: u32) {
        let cta = self.threads[issuer as usize].cta;
        if self.fp.is_none() || !self.has_dependent(cta) {
            return;
        }
        let i = (1u64 << 40) + actor as u64;
        self.rec(9, cta as u32, i, i + 1, true);
    }

    // ---------------------------------------------------------------- scopes (S-SCOPE)

    /// The thread behind an actor (threads, their tcgen05 actors, their async operations).
    pub fn actor_thread(&self, a: u32) -> usize {
        let n = self.nthreads();
        (if a < 3 * n { a % n } else { (a - 3 * n) / 4096 }) as usize
    }

    /// Does scope `s` (as seen by actor `x`) include actor `y`?
    pub fn scope_includes(&self, s: Scope, x: u32, y: u32) -> bool {
        let (cx, cy) = (
            self.threads[self.actor_thread(x)].cta as u32,
            self.threads[self.actor_thread(y)].cta as u32,
        );
        match s {
            Scope::Sys => true,
            Scope::Gpu => self.launch.g(cx).device == self.launch.g(cy).device,
            Scope::Cluster => self.launch.cluster_of(cx).0 == self.launch.cluster_of(cy).0,
            Scope::Cta => cx == cy,
        }
    }

    /// PTX memory model: two strong accesses are morally strong when each one's scope
    /// includes the other's thread. `0` = a plain access.
    pub fn morally_strong(&self, a: u32, sa: u8, b: u32, sb: u8) -> bool {
        if sa == 0 || sb == 0 {
            return false;
        }
        // E-HW-L1 hardware ordering: strong accesses are single-copy atomic at any scope.
        if (sa == STRONG_SYS && sb == STRONG_SYS) || self.launch.hw_order {
            return true;
        }
        let scope = |c: u8| [Scope::Cta, Scope::Cluster, Scope::Gpu, Scope::Sys][c as usize - 1];
        self.scope_includes(scope(sa), a, b) && self.scope_includes(scope(sb), b, a)
    }

    fn sync_key(loc: Loc) -> Option<(u8, u32, u64)> {
        match loc {
            Loc::Shared { cta, off } => Some((0, cta as u32, off as u64)),
            Loc::Global { alloc, off } => Some((1, alloc as u32, off)),
            Loc::Param { .. } => None,
        }
    }

    /// A strong write by thread `t`: it becomes the location's synchronization source. A
    /// read-modify-write continues the release sequence of the write it read.
    fn sync_publish(&mut self, t: usize, loc: Loc, m: MemSync, rmw: bool) {
        let Some(key) = Self::sync_key(loc) else {
            return;
        };
        let th = &self.threads[t];
        // E-HW-L1 hardware ordering (Launch::hw_order, two-level check only): memory behind
        // the L1s is sequentially consistent, so a strong write passes on its writer's whole
        // history, at system scope.
        let (mut rel, rel_scope) = if self.launch.hw_order {
            (th.vc.clone(), Scope::Sys)
        } else if m.releases() {
            (th.vc.clone(), m.scope)
        } else {
            (th.membar_vc.clone(), m.scope.min(th.membar_scope))
        };
        // ASSUME(E-SMEM-CUMUL): a relaxed SASS shared store also passes on, at CTA scope,
        // what its thread acquired (GB300 shared memory is SC, docs/smem-litmus.md, and the
        // acquired writes were visible at their scope before the store issued). Validated by
        // litmus for acquires that are a real CCTL.IVALL (docs/poll-acq-litmus.md); what
        // E-POLL-ACQ (refuted) adds to `acq_view` is not covered.
        let mut rel_scope = rel_scope;
        if self.launch.smem_cumulative
            && !m.releases()
            && self.k.front == Front::Sass
            && matches!(loc, Loc::Shared { .. })
        {
            vc_join(&mut rel, &th.acq_view);
            rel_scope = rel_scope.max(Scope::Cta);
        }
        let mut rr = if self.launch.read_done {
            self.read_release(t)
        } else {
            VC::new()
        };
        if rmw {
            if let Some(old) = self.syncs.get(&key) {
                vc_join(&mut rel, &old.rel);
                vc_join(&mut rr, &old.rr);
            }
        }
        self.syncs.insert(
            key,
            SyncCell {
                writer: t as u32,
                scope: m.scope,
                rel,
                rel_scope,
                rr,
            },
        );
        if self.ring.is_some() {
            self.ring_wrote(key);
        }
    }

    /// ASSUME(E-READ-DONE): what thread `t`'s strong store or barrier arrival orders before
    /// itself among reads, as clock entries of the readers' read actors (`RR_BIT`): `t`'s own
    /// reads up to its last load not used yet (`Thread::ld_pend`), every read of an async unit
    /// whose completion `t` knows of (an async unit's clock is known only through its
    /// completion), and the read entries `t` already holds.
    pub fn read_release(&self, t: usize) -> VC {
        let th = &self.threads[t];
        let me = t as u32;
        let own = vc_get(&th.vc, me);
        let done = th
            .ld_pend
            .iter()
            .map(|p| p.3)
            .min()
            .map_or(own, |c| c.saturating_sub(1));
        let n3 = 3 * self.nthreads();
        let mut e: std::collections::BTreeMap<u32, u32> = Default::default();
        for (a, c) in vc_tail(&th.vc, n3) {
            let k = if a & RR_BIT != 0 {
                a
            } else if a >= n3 {
                a | RR_BIT
            } else {
                continue;
            };
            let x = e.entry(k).or_insert(0);
            *x = (*x).max(c);
        }
        if done > 0 {
            let x = e.entry(me | RR_BIT).or_insert(0);
            *x = (*x).max(done);
        }
        vc_from_sorted(e.into_iter().collect())
    }

    /// E-READ-DONE: is the read `r` ordered before an actor whose clock is `vc`, for a
    /// write-after-read check?
    #[inline]
    pub fn read_before(&self, r: &Epoch, vc: &VC) -> bool {
        r.hb(vc) || self.launch.read_done && r.clock <= vc_get(vc, r.actor | RR_BIT)
    }

    /// A plain write ends the location's release sequence.
    fn sync_clear(&mut self, loc: Loc) {
        if let Some(key) = Self::sync_key(loc) {
            self.syncs.remove(&key);
        }
    }

    /// A strong read by thread `t` observes the location's last strong write: if the two are
    /// morally strong and the release's scope includes `t`, an acquire read synchronizes now;
    /// a relaxed one leaves it for the next acquire fence.
    fn sync_observe(&mut self, t: usize, loc: Loc, m: MemSync) {
        let Some(key) = Self::sync_key(loc) else {
            return;
        };
        let c = self.syncs.get(&key).cloned();
        self.sync_observe_cell(t, c, m);
    }

    /// `sync_observe` of a given synchronization source (a forced C-RING read).
    fn sync_observe_cell(&mut self, t: usize, c: Option<SyncCell>, m: MemSync) {
        let Some(c) = c else {
            return;
        };
        let me = t as u32;
        let hw = self.launch.hw_order;
        if c.writer == me
            || (!hw
                && (!self.morally_strong(c.writer, strong_code(c.scope), me, strong_code(m.scope))
                    || !self.scope_includes(c.rel_scope, c.writer, me)))
        {
            return;
        }
        // ASSUME(E-READ-DONE): the writer's completed reads come before this read, and so
        // before every later step of this thread (write-after-read ordering only).
        if let Some(rr) = crate::machine::rr_new(&self.threads[t].vc, &c.rr) {
            vc_join(&mut self.threads[t].make_mut().vc, &rr);
        }
        // Hardware ordering: nothing new when the reader already has the writer's clock at
        // the write (every published clock is a whole snapshot of its writer's knowledge).
        // (`acq_view` only feeds E-SMEM-CUMUL's relaxed shared stores, which under the
        // hardware ordering publish the whole clock anyway.)
        if hw {
            if vc_get(&self.threads[t].vc, c.writer) >= vc_get(&c.rel, c.writer) {
                return;
            }
            vc_join(&mut self.threads[t].make_mut().vc, &c.rel);
            return;
        }
        let th = self.threads[t].make_mut();
        if m.acquires() {
            vc_join(&mut th.vc, &c.rel);
            vc_join(&mut th.acq_view, &c.rel);
        } else {
            match th.acq_pending.iter_mut().find(|p| p.0 == c.writer) {
                Some(p) => {
                    vc_join(&mut p.1, &c.rel);
                    p.2 = p.2.min(c.rel_scope);
                }
                None => th.acq_pending.push((c.writer, c.rel, c.rel_scope)),
            }
        }
    }

    /// Acquire fence of scope `s` (None: SASS CCTL.IVALL, scoped by the loads): earlier
    /// relaxed observations of releases whose writer `s` includes synchronize.
    fn acquire_pending(&mut self, t: usize, s: Option<Scope>) {
        if self.threads[t].acq_pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.threads[t].make_mut().acq_pending);
        let mut j = VC::new();
        for (w, rel, _) in &pending {
            if s.is_none_or(|s| self.scope_includes(s, t as u32, *w)) {
                vc_join(&mut j, rel);
            }
        }
        let th = self.threads[t].make_mut();
        vc_join(&mut th.vc, &j);
        vc_join(&mut th.acq_view, &j);
    }

    /// DPOR: read the trigger / exit / completion state of every CTA of grid `gi`.
    pub fn rec_grid_reads(&mut self, gi: usize) {
        if self.fp.is_none() {
            return;
        }
        let g = &self.launch.grids[gi];
        for c in g.cta0..g.cta0 + g.ctas() {
            self.rec(9, c, 0, u64::MAX, false);
        }
    }

    /// Complete the try_cancel response `i`: `victim` = the CTA it cancels (one that has not
    /// launched), or None (the request fails; allowed at any time, E-CLC).
    pub fn complete_clc(&mut self, i: usize, victim: Option<u16>) {
        let op = self.pending.remove(i);
        let AsyncKind::Clc { cta, resp, mbar } = op.kind else {
            unreachable!()
        };
        // DPOR: a CTA that already launched removed a "cancel it" outcome, so the completion
        // (even a failure) reads the launch keys of launched CTAs. Unlaunched CTAs are still
        // alternatives of this very transition and need no dependency.
        for c in 0..self.launch.ctas() {
            if self
                .threads
                .iter()
                .any(|t| t.cta == c as u16 && t.st != St::Unlaunched)
                && victim != Some(c as u16)
            {
                self.rec(4, c, 0, 1, false);
            }
        }
        let words: [u32; 4] = match victim {
            Some(v) => {
                self.rec(4, v as u32, 0, 1, true);
                for t in 0..self.threads.len() {
                    if self.threads[t].cta == v {
                        self.threads[t].make_mut().st = St::Exited;
                    }
                }
                // GB300 (probes/clc): {first ctaid.x, .y, 1, 0}; failure {2, 0, 0, 0}.
                let [x, y, _] = self.launch.ctaid(v as u32);
                [x, y, 1, 0]
            }
            None => [2, 0, 0, 0],
        };
        let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.access(
            op.actor,
            &op.vc,
            op.pc,
            Loc::Shared { cta, off: resp },
            16,
            Some(&data),
        );
        let rec = (cta, op.actor, vc_get(&op.vc, op.actor), op.pc);
        if let Err(k) = self.completed.binary_search(&rec) {
            self.completed.insert(k, rec);
        }
        self.rec(0, cta as u32, mbar as u64, mbar as u64 + 8, false);
        self.mbar_materialize(cta, mbar);
        if self.mbars.contains_key(&(cta, mbar)) {
            let _ = self.mbar_atomic(op.actor, &op.vc, &op.vis, op.pc, cta, mbar);
            let mb = self.mbars.get_mut(&(cta, mbar)).unwrap();
            mb.tx -= 16;
            vc_join(&mut mb.cur, &op.vc);
            self.mbar_maybe_complete(cta, mbar);
        } else {
            let msg = format!(
                "try_cancel issued at {} signals an uninitialized mbarrier at {mbar:#x}",
                self.k.describe(op.pc as usize)
            );
            self.report("violation", format!("clc-uninit:{}", op.pc), msg);
        }
    }

    /// CTAs whose TMEM a (paired) allocation updates: this CTA, or its 2-CTA pair (issued by
    /// the even rank).
    pub fn alloc_group(&self, cta: u16, pair: bool) -> Result<Vec<u16>, String> {
        if !pair {
            return Ok(vec![cta]);
        }
        let (_, rank) = self.launch.cluster_of(cta as u32);
        if rank % 2 != 0 {
            return Err(format!("UTCATOMSWS.2CTA issued by odd CTA rank {rank}"));
        }
        Ok(vec![cta, self.rank_cta(cta, ((rank + 1) as u64) << 24)?])
    }

    /// First run free in every CTA of the group.
    /// ASSUME(E-TMEM-COTENANT, E-TMEM-PERMIT): the columns of an SM partner count as taken,
    /// and its unrelinquished permit blocks (tenant.rs; no partners by default).
    pub fn tmem_find_group(&self, ctas: &[u16], units: u32) -> Option<u32> {
        if ctas.iter().any(|&c| self.tmem_permit_blocked(c)) {
            return None;
        }
        let mask = ((1u32 << units) - 1) as u16;
        let bits = ctas.iter().fold(0u16, |b, &c| b | self.tmem_busy(c));
        (0..16u32)
            .step_by(units as usize)
            .find(|&i| bits & (mask << i) == 0)
    }

    /// First free, naturally aligned run of `units` 32-column units.
    pub fn tmem_find(&self, cta: u16, units: u32) -> Option<u32> {
        if self.tmem_permit_blocked(cta) {
            return None;
        }
        let bits = self.tmem_busy(cta);
        let mask = ((1u32 << units) - 1) as u16;
        (0..16u32)
            .step_by(units as usize)
            .find(|&i| bits & (mask << i) == 0)
    }

    /// Mark columns [col, col+ncols) as freshly allocated or freed: their contents become
    /// undefined, and every later access must happen-after this event.
    fn tmem_reset(&mut self, cta: u16, col: u32, ncols: u32, actor: u32, vc: &VC, pc: u32) {
        self.tmem_mark(cta, col, ncols, actor, vc, pc, 0)
    }

    fn tmem_free(&mut self, cta: u16, col: u32, ncols: u32, actor: u32, vc: &VC, pc: u32) {
        self.tmem_mark(cta, col, ncols, actor, vc, pc, 2)
    }

    #[allow(clippy::too_many_arguments)]
    fn tmem_mark(
        &mut self,
        cta: u16,
        col: u32,
        ncols: u32,
        actor: u32,
        vc: &VC,
        pc: u32,
        state: u8,
    ) {
        let e = Epoch {
            actor,
            clock: vc_get(vc, actor),
            pc,
            strong: TMEM_MARK,
        };
        // The mark is a write of every word: an access it does not happen-after is a race
        // (use before free / after alloc). Checking it here, and not only at a later access,
        // keeps race detection independent of the order of the two (E-DRF): the mark clears
        // the shadow. Marks do not race with each other (the allocator orders them).
        let mut races: Vec<(Epoch, &'static str, usize, u32)> = Vec::new();
        let tm = &mut self.tmem[cta as usize];
        // Columns are contiguous in TMEM storage (`tmem_idx`): the marked words are one run.
        let (lo, hi) = (
            tmem_idx(0, col as usize),
            tmem_idx(0, (col + ncols) as usize),
        );
        let mut i = lo;
        while i < hi {
            i += tm.sh.for_run(i, hi - i, |j, sh| {
                let (lane, c) = (j % TMEM_LANES, (j / TMEM_LANES) as u32);
                let w =
                    sh.w.filter(|w| w.strong != TMEM_MARK && w.actor != actor && !w.hb(vc))
                        .map(|w| (w, "write-write"));
                let r =
                    sh.r.iter()
                        .find(|r| r.actor != actor && !r.hb(vc))
                        .map(|r| (*r, "read-write"));
                for (o, kind) in w.into_iter().chain(r) {
                    if !races.iter().any(|x| x.0.pc == o.pc && x.1 == kind) {
                        races.push((o, kind, lane, c));
                    }
                }
                sh.w = Some(e);
                sh.r.clear();
            });
        }
        for j in lo..hi {
            *tm.init.get_mut(j) = state;
        }
        for (other, kind, lane, col) in races {
            let sig = format!("race:tmem:{}:{}", other.pc, kind);
            let msg = format!(
                "{kind} race on TMEM lane {lane} column {col}: {} at {} and the {} by {} at {} are not ordered by happens-before",
                self.actor_name(other.actor),
                self.k.describe(other.pc as usize),
                if state == 2 {
                    "deallocation"
                } else {
                    "allocation"
                },
                self.actor_name(actor),
                self.k.describe(pc as usize)
            );
            self.report("race", sig, msg);
        }
    }

    /// Race-checked TMEM word access (lane, column) as `actor`.
    #[allow(clippy::too_many_arguments)]
    pub fn tmem_access(
        &mut self,
        cta: u16,
        lane: usize,
        col: usize,
        actor: u32,
        vc: &VC,
        pc: u32,
        write: Option<u32>,
    ) -> (u32, u8) {
        let me = Epoch {
            actor,
            clock: vc_get(vc, actor),
            pc,
            strong: 0,
        };
        let i = tmem_idx(lane, col);
        self.rec(2, cta as u32, i as u64, i as u64 + 1, write.is_some());
        if let Some(ns) = self.num.as_mut() {
            match write {
                Some(_) => ns.taint.write((3, cta as u64), i as u64, 1),
                None => ns.taint.read((3, cta as u64), i as u64, 1),
            }
        }
        let mut race: Option<(Epoch, &'static str)> = None;
        {
            let tm = &mut self.tmem[cta as usize];
            let sh = tm.sh.get(i);
            if let Some(w) = sh.w {
                if w.actor != actor && !w.hb(vc) {
                    race = Some((
                        w,
                        if write.is_some() {
                            "write-write"
                        } else {
                            "write-read"
                        },
                    ));
                }
            }
            if let Some(wv) = write {
                if race.is_none() {
                    race =
                        sh.r.iter()
                            .find(|r| r.actor != actor && !r.hb(vc))
                            .map(|r| (*r, "read-write"));
                }
                let s = tm.sh.get_mut(i);
                s.w = Some(me);
                s.r.clear();
                *tm.words.get_mut(i) = wv;
                if *tm.init.get(i) != 2 {
                    *tm.init.get_mut(i) = 1;
                }
            } else {
                tm.sh.get_mut(i).r.insert(me);
            }
        }
        if let Some((other, kind)) = race {
            let sig = format!("race:tmem:{}:{}", other.pc, kind);
            let msg = format!(
                "{kind} race on TMEM lane {lane} column {col}: {} at {} and {} at {} are not ordered by happens-before",
                self.actor_name(other.actor),
                self.k.describe(other.pc as usize),
                self.actor_name(actor),
                self.k.describe(pc as usize)
            );
            self.report("race", sig, msg);
        }
        let tm = &self.tmem[cta as usize];
        (*tm.words.get(i), *tm.init.get(i))
    }

    /// `tmem_access` over a block of `nl` lanes x `nc` columns (data lane-major: `lane * nc +
    /// col`), page by page. Returns the words (before any write) and their `init` states ORed
    /// as `tmem_state_report` bits. A read with `record_read == false` checks for races but
    /// leaves no reader epoch: only for a read the same transition overwrites (MMA's
    /// accumulator read), where the write clears the readers anyway.
    #[allow(clippy::too_many_arguments)]
    pub fn tmem_block(
        &mut self,
        cta: u16,
        (lane0, nl): (usize, usize),
        (col0, nc): (usize, usize),
        actor: u32,
        vc: &VC,
        pc: u32,
        write: Option<&[u32]>,
        record_read: bool,
    ) -> (Vec<u32>, u8) {
        let me = Epoch {
            actor,
            clock: vc_get(vc, actor),
            pc,
            strong: 0,
        };
        let mut out = vec![0u32; nl * nc];
        let mut states = 0u8;
        // First race per (other pc, kind), with its lane and column.
        let mut races: Vec<(Epoch, &'static str, usize, usize)> = Vec::new();
        let mut race = |o: Epoch, kind: &'static str, lane: usize, col: usize| {
            if !races.iter().any(|r| r.0.pc == o.pc && r.1 == kind) {
                races.push((o, kind, lane, col));
            }
        };
        for c in 0..nc {
            let col = col0 + c;
            let lo = tmem_idx(lane0, col);
            self.rec(2, cta as u32, lo as u64, (lo + nl) as u64, write.is_some());
            if let Some(ns) = self.num.as_mut() {
                match write {
                    Some(_) => ns.taint.write((3, cta as u64), lo as u64, nl as u64),
                    None => ns.taint.read((3, cta as u64), lo as u64, nl as u64),
                }
            }
            let tm = &mut self.tmem[cta as usize];
            // The column's lanes are contiguous within one page (PAGE % TMEM_LANES == 0).
            let words = &tm.words.run(lo)[..nl];
            let init = &tm.init.run(lo)[..nl];
            for l in 0..nl {
                out[l * nc + c] = words[l];
                states |= match init[l] {
                    0 => 1,
                    2 => 2,
                    _ => 0,
                };
            }
            if let Some(ws) = write {
                let words = &mut tm.words.run_mut(lo)[..nl];
                for (l, w) in words.iter_mut().enumerate() {
                    *w = ws[l * nc + c];
                }
                for st in &mut tm.init.run_mut(lo)[..nl] {
                    if *st != 2 {
                        *st = 1;
                    }
                }
            }
        }
        // The shadow: whole columns are contiguous in storage (`tmem_idx`), so a block of all
        // lanes is one run, and an MMA's accumulator tile keeps uniform shadow pages.
        let check = |sh: &Sh, j: usize, race: &mut dyn FnMut(Epoch, &'static str, usize, usize)| {
            let (lane, col) = (j % TMEM_LANES, j / TMEM_LANES);
            if let Some(w) = sh.w {
                if w.actor != actor && !w.hb(vc) {
                    let kind = if write.is_some() {
                        "write-write"
                    } else {
                        "write-read"
                    };
                    race(w, kind, lane, col);
                }
            }
            if write.is_some() {
                if let Some(r) = sh.r.iter().find(|r| r.actor != actor && !r.hb(vc)) {
                    race(*r, "read-write", lane, col);
                }
            }
        };
        let spans: Vec<(usize, usize)> = if nl == TMEM_LANES {
            vec![(tmem_idx(0, col0), nc * TMEM_LANES)]
        } else {
            (0..nc).map(|c| (tmem_idx(lane0, col0 + c), nl)).collect()
        };
        let tm = &mut self.tmem[cta as usize];
        for (lo, n) in spans {
            if write.is_none() && !record_read {
                for j in lo..lo + n {
                    check(tm.sh.get(j), j, &mut race);
                }
                continue;
            }
            let mut i = lo;
            while i < lo + n {
                i += tm.sh.for_run(i, lo + n - i, |j, s| {
                    check(s, j, &mut race);
                    if write.is_some() {
                        s.w = Some(me);
                        s.r.clear();
                    } else {
                        s.r.insert(me);
                    }
                });
            }
        }
        for (other, kind, lane, col) in races {
            let sig = format!("race:tmem:{}:{}", other.pc, kind);
            let msg = format!(
                "{kind} race on TMEM lane {lane} column {col}: {} at {} and {} at {} are not ordered by happens-before",
                self.actor_name(other.actor),
                self.k.describe(other.pc as usize),
                self.actor_name(actor),
                self.k.describe(pc as usize)
            );
            self.report("race", sig, msg);
        }
        (out, states)
    }

    fn tmem_state_report(&mut self, states: u8, what: &str, pc: u32) {
        // states: bit 0 = some word never written since alloc, bit 1 = some word unallocated
        if states & 2 != 0 {
            self.report("violation", format!("tmem-unalloc:{pc}"), format!("{what} at {} accesses TMEM columns that are not allocated (never allocated, or already deallocated)", self.k.describe(pc as usize)));
        } else if states & 1 != 0 {
            self.report(
                "violation",
                format!("tmem-uninit-ld:{pc}"),
                format!(
                    "{what} at {} reads TMEM that was never written since allocation",
                    self.k.describe(pc as usize)
                ),
            );
        }
    }

    // ASSUME(S-TCGEN05): see ASSUMPTIONS.md
    fn tc_actor(&mut self, t: usize) -> u32 {
        let a = self.tc_actor_id(t);
        if vc_get(&self.threads[t].tc_vc, a) == 0 {
            vc_tick(&mut self.threads[t].make_mut().tc_vc, a);
        }
        // ASSUME(S-TCGEN05): an async tcgen05 op is issued after the thread's own earlier
        // operations (program order, e.g. its tcgen05.alloc), but sees what the thread
        // acquired from others only after fence::after_thread_sync.
        let own = vc_get(&self.threads[t].vc, t as u32);
        if vc_get(&self.threads[t].tc_vc, t as u32) < own {
            vc_join(
                &mut self.threads[t].make_mut().tc_vc,
                &vc_single(t as u32, own),
            );
        }
        a
    }

    fn pipe_actor(&mut self, t: usize) -> u32 {
        self.pipe_actor_id(t)
    }

    /// Keep pending async ops in a canonical order (by actor, then issue clock), so that
    /// commuting issues produce identical states. Per-pipe FIFO order is preserved.
    fn push_async(&mut self, op: AsyncOp) {
        self.pending.push(op);
        self.pending
            .sort_by_key(|o| (o.actor, vc_get(&o.vc, o.actor)));
    }

    /// Clock for a new op in thread t's in-order tcgen05 pipe (includes all earlier ops).
    fn pipe_issue(&mut self, t: usize) -> (u32, VC) {
        let a = self.pipe_actor(t);
        let th = self.threads[t].make_mut();
        let mut vc = th.vc.clone();
        vc_join(&mut vc, &th.pipe_vc);
        vc_tick(&mut vc, a);
        th.pipe_vc = vc.clone();
        (a, vc)
    }

    /// Checks to run when the kernel finishes (TMEM leaks).
    pub fn exit_checks(&mut self) {
        // An async operation must complete before its CTA exits (it writes or reads the CTA's
        // shared memory / TMEM). That is guaranteed only if some thread of the CTA observed the
        // completion (through an mbarrier, a commit, or wait_group) before exiting: otherwise the
        // CTA can exit first in some schedule. Checked like a race, so independent of the order
        // in which this schedule happened to complete the op.
        let completed = std::mem::take(&mut self.completed);
        for &(cta, actor, clock, pc) in &completed {
            let observed = self
                .threads
                .iter()
                .any(|t| t.cta == cta && vc_get(&t.vc, actor) >= clock);
            if !observed {
                // ASSUME(E-BULK-EXIT): a bulk (TMA) store only reads the CTA's shared memory,
                // and GB300 keeps that memory until the read is done (probes/tma_store_exit).
                // The PTX ISA still requires wait_group.read first: a contract finding.
                if matches!(self.k.code[pc as usize].op, Op::TmaStore { .. }) {
                    let msg = format!(
                        "{} (issued at {}) may still be reading shared memory when its CTA exits: the PTX ISA requires cp.async.bulk.wait_group.read first; GB300 drains it (E-BULK-EXIT)",
                        self.actor_name(actor),
                        self.k.describe(pc as usize)
                    );
                    self.report("contract", format!("bulk-store-at-exit:{pc}"), msg);
                    continue;
                }
                // ASSUME(E-TC-EXIT): a tcgen05.commit moves no data; GB300 keeps the CTA's
                // shared memory until its tcgen05 pipe drained (probes/tc_commit_exit). Only
                // when every earlier op of its pipe was observed (a trailing arrive).
                let unobserved = |&(c, a, k, q): &(u16, u32, u32, u32)| {
                    c == cta
                        && a == actor
                        && k < clock
                        && !matches!(self.k.code[q as usize].op, Op::TcCommit { .. })
                        && !self
                            .threads
                            .iter()
                            .any(|t| t.cta == c && vc_get(&t.vc, a) >= k)
                };
                if matches!(self.k.code[pc as usize].op, Op::TcCommit { .. })
                    && !completed.iter().any(unobserved)
                {
                    let msg = format!(
                        "{} (issued at {}) may arrive on its mbarrier after the CTA exited: no thread waited for it; GB300 drains the tcgen05 pipe first (E-TC-EXIT)",
                        self.actor_name(actor),
                        self.k.describe(pc as usize)
                    );
                    self.report("contract", format!("tc-commit-at-exit:{pc}"), msg);
                    continue;
                }
                let msg = format!(
                    "{} (issued at {}) is not ordered before the exit of any thread of its CTA: it can complete after the CTA exited (into released shared memory)",
                    self.actor_name(actor),
                    self.k.describe(pc as usize)
                );
                self.report("violation", format!("async-after-exit:{pc}"), msg);
            }
        }
        self.completed = completed;
        // Replicas (case `replicas`, e.g. all-reduce ranks) must end bitwise equal (N4).
        for g in self.launch.replicas.clone() {
            let all: Vec<Vec<u8>> = g
                .iter()
                .map(|&i| {
                    let m = &self.gmem[i];
                    (0..m.bytes.len()).map(|b| *m.bytes.get(b)).collect()
                })
                .collect();
            let first = all[0].clone();
            for (k, &i) in g.iter().enumerate().skip(1) {
                let other = &all[k];
                if *other != first {
                    let words = first
                        .chunks(4)
                        .zip(other.chunks(4))
                        .filter(|(a, b)| a != b)
                        .count();
                    let msg = format!(
                        "replicas differ: {} and {} differ in {words} of {} words (ranks hold different bits)",
                        self.launch.allocs[g[0]].name,
                        self.launch.allocs[i].name,
                        first.len() / 4
                    );
                    self.report("divergent", format!("replicas:{}:{i}", g[0]), msg);
                }
            }
        }
        // ASSUME(S-CLUSTER-LIFE): an access to another CTA's shared memory (remote arrive,
        // multicast TMA, commit) must happen-before the exit of some thread of that CTA;
        // otherwise a schedule exists in which the whole target CTA exited first.
        let remote = std::mem::take(&mut self.remote);
        for &(target, actor, clock, pc) in &remote {
            let alive = self
                .threads
                .iter()
                .any(|t| t.cta == target && vc_get(&t.vc, actor) >= clock);
            if !alive {
                let msg = format!(
                    "{} at {} accesses the shared memory of CTA {target}, but is not ordered before the exit of any of its threads: CTA {target} can exit first (cluster lifetime; a final barrier.cluster is missing)",
                    self.actor_name(actor),
                    self.k.describe(pc as usize)
                );
                self.report("violation", format!("cluster-life:{pc}"), msg);
            }
        }
        self.remote = remote;
        for cta in 0..self.tmem.len() {
            let bits = self.tmem[cta].alloc;
            if bits != 0 {
                let cols: Vec<u32> = (0..16)
                    .filter(|i| bits >> i & 1 == 1)
                    .map(|i| i * 32)
                    .collect();
                self.report("violation", format!("tmem-leak:{cta}"), format!("TMEM leak: CTA {cta} exits with columns {cols:?} (32-column units) still allocated; tcgen05.dealloc is required before exit"));
            }
        }
    }

    // ---------------------------------------------------------------- transitions

    /// Does it matter whether splits of a warp merge before executing `pc`? PTX lanes are
    /// independent threads except at warp-collective instructions; converged SASS lanes are
    /// also ordered by E-LOCKSTEP, so every SASS pc counts.
    pub fn merge_sensitive(&self, pc: u32) -> bool {
        self.k.front == Front::Sass
            || self.k.code.get(pc as usize).is_some_and(|i| {
                i.op.is_warp_aligned()
                    || matches!(
                        i.op,
                        Op::Elect { .. }
                            | Op::VoteAny { .. }
                            | Op::VoteAll { .. }
                            | Op::Redux { .. }
                    )
            })
    }

    fn rec_pc(&mut self, warp: u32, pc: u32, write: bool) {
        if self.fp.is_some() && self.merge_sensitive(pc) {
            self.rec(8, warp, pc as u64, pc as u64 + 1, write);
        }
    }

    /// One action of the split of `warp` at `pc0`. For DPOR, the warp's merge-sensitive
    /// program counters are a resource (space 8): the split writes the pc it leaves and the
    /// pcs it stops at, and reads the pcs it runs through, because another split of the same
    /// warp waiting at one of them would merge with it and change what executes.
    pub fn step_split(&mut self, warp: u32, pc0: u32) {
        let tracked = self.fp.is_some();
        let polls = !self.polls.is_empty() || !self.wpolls.is_empty();
        let ids: Vec<usize> = if tracked || polls {
            self.threads
                .iter()
                .enumerate()
                .filter(|(_, t)| t.warp == warp && t.pc == pc0 && t.st == St::Run)
                .map(|(i, _)| i)
                .collect()
        } else {
            vec![]
        };
        self.rec_pc(warp, pc0, true);
        // E-POLL-ANY: a strong load whose step ends the same for every value its location
        // can hold runs as a value-insensitive poll.
        if self.vip.is_some() && self.vip_read == 0 && self.vip_ok(warp, pc0) {
            self.vip_read = 1;
        }
        self.step_split_inner(warp, pc0);
        self.vip_read = 0;
        // ASSUME(E-SPIN): threads stopped at a poll that would fail block there, but read the
        // flag once for the race detector (spin::blocked_poll_read).
        if polls {
            crate::spin::after_step(self, warp, &ids);
        }
        for t in ids {
            let pc = self.threads[t].pc;
            self.rec_pc(warp, pc, true);
        }
    }

    /// ASSUME(E-POLL-ANY): is the step of `warp`'s split at `pc0` a value-insensitive
    /// poll? It must start at a relaxed strong global or shared load whose active lanes read
    /// one location (at most 8 bytes), and running the step with each value the search has seen
    /// there (`VipLearn`, every write of every explored execution, and the current value; for a
    /// monotone location only those at least the current one) must end
    /// every member at the same pc and state with the same predicates and uniform registers,
    /// and with registers that differ only where they are dead (`spin::live_after`). Then the
    /// load runs without a footprint on its location, without entering the race shadow's reads
    /// (it goes to `vipr` instead, checked by every later write) and without observing the
    /// location's release (`vip_read`): it commutes with every write of the location, up to
    /// dead registers.
    fn vip_ok(&mut self, warp: u32, pc0: u32) -> bool {
        let k = self.k.clone();
        let inst = &k.code[pc0 as usize];
        let Op::Ld {
            space,
            bytes,
            addr,
            sync: Some(ms),
            ..
        } = &inst.op
        else {
            return false;
        };
        // An acquire load is one only in the first search (`VipLearn::acquire`): dropping its
        // edge only removes happens-before, so a clean verdict stands, but it can turn a flag
        // handoff into a race report and a re-run.
        let acq_ok = self.vip.as_ref().is_some_and(|v| v.acquire);
        if !matches!(space, Space::Global | Space::Shared) || ms.acquires() && !acq_ok {
            return false;
        }
        if *bytes > 8 {
            return false;
        }
        let members: Vec<usize> = self
            .warp_threads(warp)
            .filter(|&t| self.threads[t].pc == pc0 && self.threads[t].st == St::Run)
            .collect();
        let mut loc = None;
        for &t in &members {
            if !self.prd(t, inst.guard) {
                continue;
            }
            let Ok(a) = self.addr(t, addr) else {
                return false;
            };
            let Ok(l) = self.resolve(t, *space, a, *bytes as usize) else {
                return false;
            };
            if loc.is_some_and(|x| x != l) {
                return false;
            }
            loc = Some(l);
        }
        let Some(l) = loc else { return false };
        let Some((wh, off)) = vip_where(l) else {
            return false;
        };
        let key = (wh, off, *bytes);
        let vip = self.vip.clone().unwrap();
        let mut vals = vip.watch(key);
        let n = *bytes as usize;
        let mem = |m: &Machine| -> Mem {
            match l {
                Loc::Shared { cta, .. } => m.smem[cta as usize].clone(),
                Loc::Global { alloc, .. } => m.gmem[alloc].clone(),
                Loc::Param { .. } => unreachable!(),
            }
        };
        let m0 = mem(self);
        let cur = (0..n).fold(0u64, |v, i| {
            v | (*m0.bytes.get(off as usize + i) as u64) << (8 * i)
        });
        drop(m0);
        vals.insert(cur);
        // A monotone location (no explored write lowered it) only takes values >= the
        // current one from here. Opt-in (`PTXV_VIP_MONO=1`), and tried only when every value
        // fails: a load that is a poll in some states and not in others makes fewer states
        // equal (`vipr`).
        let mono = (std::env::var("PTXV_VIP_MONO").is_ok_and(|x| x != "0")
            && !vip.nonmono.lock().unwrap().contains(&key))
        .then(|| {
            vals.iter()
                .copied()
                .filter(|&v| v >= cur)
                .collect::<std::collections::BTreeSet<u64>>()
        })
        .filter(|m| m.len() < vals.len());
        if vals.len() <= 64 && self.vip_try(warp, pc0, l, off, n, &members, vals) {
            return true;
        }
        match mono {
            Some(m) if m.len() <= 64 => self.vip_try(warp, pc0, l, off, n, &members, m),
            _ => false,
        }
    }

    /// `vip_ok` for one candidate set: the step from every value ends the same.
    #[allow(clippy::too_many_arguments)]
    fn vip_try(
        &mut self,
        warp: u32,
        pc0: u32,
        l: Loc,
        off: u64,
        n: usize,
        members: &[usize],
        vals: std::collections::BTreeSet<u64>,
    ) -> bool {
        let k = self.k.clone();
        type Out = Vec<(u32, St, Vec<u64>, Vec<u64>)>;
        let mut first: Option<(Out, Vec<u64>)> = None;
        let mut dead: std::collections::BTreeSet<Reg> = Default::default();
        let file = if k.front == Front::Sass {
            File::R
        } else {
            File::V
        };
        for v in vals {
            let mut c = self.clone();
            c.fp = None;
            c.num = None;
            c.gfoot = None;
            c.sfoot = None;
            c.vip = None;
            c.wlog = Default::default();
            c.findings.clear();
            let cm = match l {
                Loc::Shared { cta, .. } => &mut c.smem[cta as usize],
                Loc::Global { alloc, .. } => &mut c.gmem[alloc],
                Loc::Param { .. } => unreachable!(),
            };
            for i in 0..n {
                *cm.bytes.run_mut(off as usize + i).first_mut().unwrap() = (v >> (8 * i)) as u8;
            }
            c.step_split_inner(warp, pc0);
            if !c.findings.is_empty() {
                return false;
            }
            let out: Out = members
                .iter()
                .map(|&t| {
                    let th = &c.threads[t];
                    (th.pc, th.st, th.preds.clone(), th.regs.clone())
                })
                .collect();
            let ur = c.warps[warp as usize].ur.clone();
            match &first {
                None => first = Some((out, ur)),
                Some((o0, ur0)) => {
                    if *ur0 != ur {
                        return false;
                    }
                    for (a, b) in o0.iter().zip(&out) {
                        if a.0 != b.0 || a.1 != b.1 || a.2 != b.2 {
                            return false;
                        }
                        for (i, (x, y)) in a.3.iter().zip(&b.3).enumerate() {
                            if x != y {
                                dead.insert(Reg {
                                    file,
                                    idx: i as u16,
                                });
                            }
                        }
                    }
                }
            }
        }
        let Some((o0, _)) = first else { return false };
        if dead.is_empty() {
            return true;
        }
        o0.iter().all(|(pc, st, _, _)| {
            *st == St::Exited || crate::spin::live_after(&k.code, *pc as usize, &dead).is_none()
        })
    }

    /// Indices of `warp`'s threads (threads are laid out warp by warp).
    pub fn warp_threads(&self, warp: u32) -> std::ops::Range<usize> {
        let n = self.threads.len();
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.threads[mid].warp < warp {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut end = lo;
        while end < n && self.threads[end].warp == warp {
            end += 1;
        }
        lo..end
    }

    fn step_split_inner(&mut self, warp: u32, pc0: u32) {
        let wt = self.warp_threads(warp);
        let mut members: Vec<usize> = wt
            .clone()
            .filter(|&i| {
                let t = &self.threads[i];
                t.pc == pc0 && t.st == St::Run
            })
            .collect();
        for &t in &members {
            if let Some(key) = self.threads[t].acq_bar {
                self.threads[t].make_mut().acq_bar = None;
                self.rec_acq(key);
            }
        }
        let k = self.k.clone();
        let mut pc = pc0;
        let mut first = true;
        for _ in 0..max_local_steps() {
            let inst = &k.code[pc as usize];
            // PTX elect.sync starts its own action, so the explorer can choose the leader.
            let elect = k.front == Front::Ptx && matches!(inst.op, Op::Elect { .. });
            // A visible instruction guarded off in every member is a no-op: run through it
            // (its own step would commute with everything, E-DRF).
            let visible = inst.op.is_visible()
                && (inst.guard.is_always() || members.iter().any(|&t| self.prd(t, inst.guard)));
            if !first && (visible || elect || !self.split_enabled(&members, pc)) {
                return;
            }
            first = false;
            if let Err(e) = self.exec(&members, pc) {
                let sig = format!("fault:{pc}");
                self.report(
                    fault_kind(&e),
                    sig,
                    format!("{e} at {}", k.describe(pc as usize)),
                );
                // A faulting step may stop before recording its footprint; for DPOR it
                // conflicts with everything, so every order that avoids the fault is explored.
                if let Some(f) = self.fp.as_mut() {
                    f.all = true;
                }
                // Stop the faulting threads so exploration can continue elsewhere.
                for &t in &members {
                    self.threads[t].make_mut().st = St::Exited;
                }
                return;
            }
            // ASSUME(E-LOCKSTEP): see ASSUMPTIONS.md
            // SASS: a converged split executes in lockstep, so its lanes' memory effects are
            // ordered by instruction (ptxas relies on this for its own warp-synchronous code).
            // PTX lanes are independent threads and get no such ordering.
            if k.front == Front::Sass
                && members.len() > 1
                && matches!(
                    inst.op,
                    Op::Ld { .. } | Op::St { .. } | Op::Atom { .. } | Op::TmemLd { .. }
                )
            {
                let j = vc_join_all(members.iter().map(|&t| &self.threads[t].vc));
                for &t in &members {
                    let th = self.threads[t].make_mut();
                    th.vc = j.clone();
                    vc_tick(&mut th.vc, t as u32);
                }
            }
            // Continue only while the split stays together, runnable, and not blocked, and no
            // other lane of the warp joined it (it must re-merge at a scheduling point).
            let t0 = &self.threads[members[0]];
            let npc = t0.pc;
            if members
                .iter()
                .any(|&t| self.threads[t].st != St::Run || self.threads[t].pc != npc)
                || npc == pc && matches!(inst.op, Op::MbarTryWait { .. })
            {
                return;
            }
            if wt.clone().any(|i| {
                let t = &self.threads[i];
                t.pc == npc && t.st == St::Run && !members.contains(&i)
            }) {
                return;
            }
            pc = npc;
            self.rec_pc(warp, pc, false);
            members.retain(|&t| self.threads[t].st == St::Run);
            if pc as usize >= k.code.len() {
                return;
            }
        }
        self.report(
            "fault",
            format!("runaway:{pc0}"),
            format!(
                "more than {} local instructions without a visible operation from {}",
                max_local_steps(),
                k.describe(pc0 as usize)
            ),
        );
    }

    pub(crate) fn exec(&mut self, members: &[usize], pc: u32) -> Result<(), String> {
        // wellsync.rs: the certificate run logs every instruction (no-op otherwise).
        let r = if self.wlog.is_on() {
            crate::wellsync::pre_inst(self, members, pc);
            let r = self.exec_traced(members, pc);
            crate::wellsync::post_inst(self, members, pc, r.is_ok());
            r
        } else {
            self.exec_traced(members, pc)
        };
        if self.launch.poll_acquire && r.is_ok() {
            self.poll_exit_acquire(members, pc);
        }
        r
    }

    /// ASSUME(E-POLL-ACQ): threads that just left a warp-level poll loop (`wpolls`) acquire
    /// their pending relaxed observations, as at an acquire fence scoped by the loads (SASS
    /// CCTL.IVALL). Refuted on GB300 (docs/poll-acq-litmus.md): weak reads of lines valid in L1
    /// stay stale after such a loop. Kept opt-in so that old verdicts reproduce.
    fn poll_exit_acquire(&mut self, members: &[usize], pc: u32) {
        // Warp-level polls only (DeepEP's barrier_block: vote over relaxed `ld.volatile`
        // polls). A straight-line poll keeps the strict model: its own acquire (CCTL.IVALL in
        // DeepEP's channel waits) is what is checked.
        let inside = |p: u32, m: &Self| m.wpolls.iter().any(|(&h, &b)| h <= p && p <= b);
        if !inside(pc, self) {
            return;
        }
        for &t in members {
            let np = self.threads[t].pc;
            if self.threads[t].st != St::Exited && !inside(np, self) {
                self.acquire_pending(t, None);
            }
        }
    }

    /// One instruction for `members` (wellsync.rs: lookahead on copies of the machine).
    pub(crate) fn exec_one(&mut self, members: &[usize], pc: u32) -> Result<(), String> {
        self.exec(members, pc)
    }

    fn exec_traced(&mut self, members: &[usize], pc: u32) -> Result<(), String> {
        // Debugging: `PTXV_TRACE_CTA=n` prints every instruction CTA n executes (warp, lanes).
        static TRACE: std::sync::OnceLock<Option<u16>> = std::sync::OnceLock::new();
        if let Some(c) = *TRACE.get_or_init(|| {
            std::env::var("PTXV_TRACE_CTA")
                .ok()
                .and_then(|v| v.parse().ok())
        }) {
            if self.threads[members[0]].cta == c {
                eprintln!(
                    "trace w{} x{} {}",
                    self.threads[members[0]].warp,
                    members.len(),
                    self.k.describe(pc as usize)
                );
            }
        }
        if self.num.is_none() {
            return self.exec_inner(members, pc);
        }
        let guard = self.k.code[pc as usize].guard;
        let active: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&t| self.prd(t, guard))
            .collect();
        self.taint_pre(members, &active, pc);
        let r = self.exec_inner(members, pc);
        self.taint_post(members, &active, pc);
        r
    }

    fn exec_inner(&mut self, members: &[usize], pc: u32) -> Result<(), String> {
        let k = self.k.clone();
        let inst = &k.code[pc as usize];
        // A control-word wait on the bulk read scoreboard holds for every member, guard or not.
        if k.bulk_read_waits.binary_search(&(pc as usize)).is_ok() {
            for &t in members {
                self.bulk_wait(t, 0);
            }
        }
        let active: Vec<usize> = members
            .iter()
            .copied()
            .filter(|&t| self.prd(t, inst.guard))
            .collect();
        // ASSUME(E-READ-DONE): an instruction that reads (or overwrites) a pending load's
        // registers waits for the load: the read is complete.
        if self.launch.read_done {
            let mut regs: Option<Vec<Reg>> = None;
            for &t in &active {
                if self.threads[t].ld_pend.is_empty() {
                    continue;
                }
                let rs = regs.get_or_insert_with(|| {
                    let mut v = inst.op.src_regs();
                    v.extend(inst.op.dst_regs());
                    v
                });
                let hit = |p: &(u8, u16, u8, u32)| {
                    rs.iter()
                        .any(|r| r.file as u8 == p.0 && r.idx >= p.1 && r.idx < p.1 + p.2 as u16)
                };
                if self.threads[t].ld_pend.iter().any(hit) {
                    self.threads[t].make_mut().ld_pend.retain(|p| !hit(p));
                }
            }
        }
        if k.front == Front::Ptx && inst.op.is_warp_aligned() && !active.is_empty() {
            let w = self.threads[members[0]].warp;
            let live = self
                .threads
                .iter()
                .filter(|t| t.warp == w && t.st != St::Exited)
                .count();
            if active.len() < live {
                let msg = format!(
                    ".sync.aligned instruction {} executed by {} of the warp's {live} live lanes ({}); every lane must execute it together",
                    k.describe(pc as usize),
                    active.len(),
                    self.actor_name(active[0] as u32)
                );
                self.report("violation", format!("aligned:{pc}"), msg);
            }
        }
        // A warp-collective matrix op whose guard is false in every lane does nothing
        // (`@P0 LDSM.16.M88.4` in CUTLASS's fmha, P0 warp-uniform).
        if active.is_empty()
            && matches!(
                inst.op,
                Op::LdMatrix { .. } | Op::StMatrix { .. } | Op::Hmma { .. }
            )
        {
            for &t in members {
                self.threads[t].make_mut().pc = pc + 1;
            }
            return Ok(());
        }
        match &inst.op {
            Op::R2Ur { d, a, p } => {
                let vals: Vec<u64> = active
                    .iter()
                    .map(|&t| self.rd(t, a))
                    .collect::<Result<_, _>>()?;
                if let Some(&v0) = vals.first() {
                    if let Some(p) = p {
                        for (&t, &v) in active.iter().zip(&vals) {
                            self.wr(t, *p, (v != v0) as u64, 1);
                        }
                    } else if vals.iter().any(|&v| v != v0)
                        && !shifted_out(
                            &k.code,
                            pc as usize,
                            *d,
                            vals.iter().fold(0, |m, &v| m | (v ^ v0)),
                        )
                    {
                        self.report(
                            "violation",
                            format!("r2ur:{pc}"),
                            format!("R2UR of a non-uniform value at {}", k.describe(pc as usize)),
                        );
                    }
                    self.wr(active[0], *d, v0, 1);
                }
            }
            Op::VoteAny { d, a } => {
                let mut mask = 0u64;
                for &t in &active {
                    if self.prd(t, *a) {
                        mask |= 1 << self.threads[t].lane;
                    }
                }
                for &t in &active {
                    self.wr(t, *d, mask, 1);
                }
            }
            Op::Redux { d, a, op } => {
                use ptxv_ir::ReduxOp as R;
                let mut acc: Option<u32> = None;
                for &t in &active {
                    let x = self.rd(t, a)? as u32;
                    acc = Some(match acc {
                        None => x,
                        Some(v) => match op {
                            R::And => v & x,
                            R::Or => v | x,
                            R::Xor => v ^ x,
                            R::Sum => v.wrapping_add(x),
                            R::MinU => v.min(x),
                            R::MaxU => v.max(x),
                            R::MinS => (v as i32).min(x as i32) as u32,
                            R::MaxS => (v as i32).max(x as i32) as u32,
                        },
                    });
                }
                if let (Some(&t), Some(v)) = (active.first(), acc) {
                    self.wr(t, *d, v as u64, 1);
                }
            }
            // ASSUME(E-BSYNC): see ASSUMPTIONS.md
            Op::Bssy { b } => {
                let m = active
                    .iter()
                    .fold(0u32, |m, &t| m | 1 << self.threads[t].lane);
                if let Some(&t0) = members.first() {
                    let w = self.threads[t0].warp as usize;
                    // Parts of a warp that reach the BSSY separately each add their lanes (the
                    // BSYNC release removes them again). ASSUME(E-BSYNC)
                    self.warps[w].make_mut().bmask[*b as usize] |= m;
                }
                for &t in members {
                    self.threads[t].make_mut().pc = pc + 1;
                }
                return Ok(());
            }
            // ASSUME(E-BSYNC): see ASSUMPTIONS.md
            Op::Break { b } => {
                let gone = active
                    .iter()
                    .fold(0u32, |m, &t| m | 1 << self.threads[t].lane);
                for &t in members {
                    self.threads[t].make_mut().pc = pc + 1;
                }
                if let Some(&t0) = members.first() {
                    let wi = self.threads[t0].warp;
                    self.warps[wi as usize].make_mut().bmask[*b as usize] &= !gone;
                    // Lanes parked at that BSYNC may be complete now.
                    let parked: Vec<(u32, u32)> = self
                        .threads
                        .iter()
                        .filter(|th| {
                            th.warp == wi
                                && th.st == St::WarpSync
                                && matches!(self.k.code[th.pc as usize].op, Op::Bsync { b: c } if c == *b)
                        })
                        .map(|th| (th.lane, th.pc))
                        .collect();
                    if let Some(&(_, at)) = parked.first() {
                        let m = parked.iter().fold(0u32, |m, (l, _)| m | 1 << l)
                            | self.warps[wi as usize].bmask[*b as usize];
                        self.warpsync_release(wi, m, at);
                    }
                }
                return Ok(());
            }
            // ASSUME(E-COLLECTIVE): see ASSUMPTIONS.md
            Op::WarpSync {
                mask,
                collective: true,
            } => {
                let mut groups: Vec<u32> = Vec::new();
                for &t in &active {
                    let m = self.rd(t, mask)? as u32;
                    if m >> self.threads[t].lane & 1 == 0 {
                        return Err(format!(
                            "{} at {}: the member mask {m:#x} does not contain the lane",
                            self.actor_name(t as u32),
                            k.describe(pc as usize)
                        ));
                    }
                    if !groups.contains(&m) {
                        groups.push(m);
                    }
                }
                for &t in members {
                    let th = self.threads[t].make_mut();
                    if active.contains(&t) {
                        th.st = St::WarpSync;
                    } else {
                        th.pc = pc + 1;
                    }
                }
                // One mask group at a time: the lowest lane's group enters first.
                if let Some(&t0) = active.first() {
                    let m = self.rd(t0, mask)? as u32;
                    self.collective_release(self.threads[t0].warp, m, pc);
                }
                return Ok(());
            }
            // ASSUME(E-COLLECTIVE): see ASSUMPTIONS.md
            Op::EndCollective { start } => {
                for &t in members {
                    self.threads[t].make_mut().pc = pc + 1;
                }
                if let Some(&t0) = members.first() {
                    let w = self.threads[t0].warp;
                    let next = self
                        .threads
                        .iter()
                        .enumerate()
                        .find(|(_, th)| {
                            th.warp == w && th.st == St::WarpSync && th.pc == *start as u32
                        })
                        .map(|(i, _)| i);
                    if let Some(t1) = next {
                        if let Op::WarpSync { mask, .. } = &k.code[*start].op {
                            let m = self.rd(t1, mask)? as u32;
                            self.collective_release(w, m, *start as u32);
                        }
                    }
                }
                return Ok(());
            }
            Op::WarpSync { .. } | Op::Bsync { .. } => {
                let m = match (&inst.op, active.first()) {
                    (Op::WarpSync { mask, .. }, Some(&t)) => self.rd(t, mask)? as u32,
                    (Op::Bsync { b }, Some(&t)) => {
                        let wi = self.threads[t].warp;
                        let w = &self.warps[wi as usize];
                        let own = active
                            .iter()
                            .fold(0u32, |m, &t| m | 1 << self.threads[t].lane);
                        // Lanes reaching BSYNC that BSSY did not record also take part, and so
                        // do lanes already parked at this barrier: two parts of a warp that
                        // reach one BSSY separately overwrite its mask (the later part's
                        // BSYNC would otherwise leave the earlier one parked for good).
                        // ASSUME(E-BSYNC)
                        let parked = self
                            .threads
                            .iter()
                            .filter(|th| {
                                th.warp == wi
                                    && th.st == St::WarpSync
                                    && matches!(self.k.code[th.pc as usize].op, Op::Bsync { b: c } if c == *b)
                            })
                            .fold(0u32, |m, th| m | 1 << th.lane);
                        w.bmask[*b as usize] | own | parked
                    }
                    _ => 0,
                };
                for &t in members {
                    let th = self.threads[t].make_mut();
                    if active.contains(&t) {
                        th.st = St::WarpSync;
                    } else {
                        th.pc = pc + 1;
                    }
                }
                if let Some(&t0) = active.first() {
                    self.warpsync_release(self.threads[t0].warp, m, pc);
                }
                return Ok(());
            }
            Op::TmemAlloc { .. } | Op::TmemDealloc { .. } | Op::TmemRelinquish => {
                // Warp-wide: one allocation/deallocation per warp, not per lane.
                if let Some(&rep) = active.first() {
                    self.exec_thread(rep, inst, pc)?;
                }
            }
            // ASSUME(E-COLLECTIVE): BRA.DIV: the whole split goes one way.
            Op::Bra {
                target,
                div: Some(dm),
                ..
            } => {
                let lanes = members
                    .iter()
                    .fold(0u32, |m, &t| m | 1 << self.threads[t].lane);
                let taken = match members.first() {
                    Some(&t0) if !active.is_empty() => {
                        active.len() != members.len() || self.rd(t0, dm)? as u32 != lanes
                    }
                    _ => false,
                };
                for &t in members {
                    self.threads[t].make_mut().pc = if taken { *target as u32 } else { pc + 1 };
                }
                return Ok(());
            }
            Op::Bra {
                target,
                any,
                tid0_falls,
                cond,
                div: None,
            } => {
                // ASSUME(E-CONST-BRA): a branch the explorer's control-flow graph decided from
                // launch constants (`explore::const_guards`) must go that way.
                if let Some(v) = self.bar_cfg.fold.get(pc as usize).copied().flatten() {
                    if members.iter().any(|t| active.contains(t) != v) {
                        let msg = format!(
                            "branch at {} goes against its launch-constant guard (expected {}): the explorer's control-flow graph does not cover it (E-CONST-BRA)",
                            self.k.describe(pc as usize),
                            if v { "taken" } else { "not taken" }
                        );
                        self.report("unmodeled", format!("const-bra:{pc}"), msg);
                    }
                }
                let go_all = *any && !active.is_empty();
                // NOT_TID0: the warp of CTA thread 0 falls through (linear thread id < 32).
                let tid0_warp = |m: &Self, t: usize| {
                    let th = &m.threads[t];
                    let b = m.launch.g(th.cta as u32).block;
                    (th.tid[0] + b[0] * (th.tid[1] + b[1] * th.tid[2])) < 32
                };
                for &t in members {
                    let falls = *tid0_falls && tid0_warp(self, t);
                    let taken = go_all
                        || (!*any
                            && active.contains(&t)
                            && !falls
                            && cond.is_none_or(|c| self.prd(t, c)));
                    self.threads[t].make_mut().pc = if taken { *target as u32 } else { pc + 1 };
                }
                return Ok(());
            }
            // ASSUME(E-WARPCOLL): stmatrix / shfl move data between the lanes of one
            // converged warp in a single step.
            Op::StMatrix { addr, v, trans } => {
                if active.len() != 32 {
                    return Err(format!(
                        "stmatrix executed by {} lanes (needs the full warp)",
                        active.len()
                    ));
                }
                let mut by_lane = [0usize; 32];
                for &t in &active {
                    by_lane[self.threads[t].lane as usize] = t;
                }
                for (i, vi) in v.iter().enumerate() {
                    for j in 0..8usize {
                        let row_lane = by_lane[8 * i + j];
                        let row = self.addr(row_lane, addr)?;
                        for q in 0..4usize {
                            let src = by_lane[4 * j + q];
                            let w = self.rd(src, vi)? as u32;
                            let vc = self.threads[src].vc.clone();
                            if !*trans {
                                let loc =
                                    self.resolve(row_lane, Space::Shared, row + 4 * q as u64, 4)?;
                                self.access(src as u32, &vc, pc, loc, 4, Some(&w.to_le_bytes()));
                                // A whole 32-bit word: it keeps its node (N-LEAF).
                                self.num_st(src, vi, loc, 4);
                            } else {
                                // element (j, 2q + h) of the fragment goes to row 2q + h, column j.
                                for h in 0..2usize {
                                    let tl = by_lane[8 * i + 2 * q + h];
                                    let base = self.addr(tl, addr)?;
                                    let loc =
                                        self.resolve(tl, Space::Shared, base + 2 * j as u64, 2)?;
                                    let half = ((w >> (16 * h)) as u16).to_le_bytes();
                                    self.access(src as u32, &vc, pc, loc, 2, Some(&half));
                                    self.half_stsm_half(src, vi, h, loc);
                                }
                            }
                        }
                    }
                }
                for &t in &active {
                    self.tick(t);
                }
            }
            Op::LdMatrix { d, n, addr, trans } => {
                if active.len() != 32 {
                    return Err(format!(
                        "ldmatrix executed by {} lanes (needs the full warp)",
                        active.len()
                    ));
                }
                let mut by_lane = [0usize; 32];
                for &t in &active {
                    by_lane[self.threads[t].lane as usize] = t;
                }
                let mut out = Vec::new();
                for i in 0..*n as usize {
                    for &t in &active {
                        let l = self.threads[t].lane as usize;
                        let vc = self.threads[t].vc.clone();
                        // (row, byte offset in the row) of the two b16 elements lane l gets.
                        let parts: [(usize, u64); 2] = if !*trans {
                            [(l / 4, 4 * (l % 4) as u64), (l / 4, 4 * (l % 4) as u64 + 2)]
                        } else {
                            [
                                (2 * (l % 4), 2 * (l / 4) as u64),
                                (2 * (l % 4) + 1, 2 * (l / 4) as u64),
                            ]
                        };
                        let mut w = 0u32;
                        // Without .trans both halves are one aligned 32-bit word of the row.
                        let mut word = None;
                        let mut hlocs = [None; 2];
                        for (h, (row, off)) in parts.into_iter().enumerate() {
                            let rl = by_lane[8 * i + row];
                            let base = self.addr(rl, addr)?;
                            let loc = self.resolve(rl, Space::Shared, base + off, 2)?;
                            hlocs[h] = Some(loc);
                            if !*trans && h == 0 {
                                word = match loc {
                                    Loc::Shared { off, .. } if off % 4 == 0 => Some(loc),
                                    _ => None,
                                };
                            }
                            let b = self.access(t as u32, &vc, pc, loc, 2, None);
                            w |= (u16::from_le_bytes([b[0], b[1]]) as u32) << (16 * h);
                        }
                        out.push((t, i, w, word, hlocs));
                    }
                }
                for (t, i, w, word, hlocs) in out {
                    let r = Reg {
                        file: d.file,
                        idx: d.idx + i as u16,
                    };
                    self.wr(t, r, w as u64, 1);
                    // 16-bit provenance: each half from its own row (`.trans` reshuffles them).
                    self.half_ldsm(t, r, hlocs);
                    // ASSUME(N-LEAF): the word moves intact, with its node.
                    match word.and_then(|loc| self.num.as_ref().and_then(|ns| ns.mem_node(loc))) {
                        Some(n) => self.num_put(t, r, Some(n)),
                        None => self.num_clobber(t, r, &[], pc),
                    }
                }
                for &t in &active {
                    self.tick(t);
                }
            }
            // ASSUME(N-HMMA): see ASSUMPTIONS.md
            Op::Hmma { d, a, b, c, tf32 } => {
                if active.len() != 32 {
                    return Err(format!(
                        "HMMA executed by {} lanes (needs the full warp)",
                        active.len()
                    ));
                }
                if *tf32 {
                    self.hmma_tf32(&active, *d, a, b, c, pc)?;
                    for &t in &active {
                        self.tick(t);
                    }
                    for &t in members {
                        self.threads[t].make_mut().pc = pc + 1;
                    }
                    return Ok(());
                }
                let bf = |x: u32| f32::from_bits(x << 16);
                let mut am = [[0f32; 16]; 16];
                let mut bm = [[0f32; 8]; 16];
                let mut cm = [[0f32; 8]; 16];
                let mut lanes = [0usize; 32];
                for &t in &active {
                    let l = self.threads[t].lane as usize;
                    lanes[l] = t;
                    let (g, q) = (l / 4, l % 4);
                    let rd = |m: &Self, v: &Val| m.rd(t, v).map(|x| x as u32);
                    for (j, v) in a.iter().enumerate() {
                        let x = rd(self, v)?;
                        let (row, col) = (g + 8 * (j & 1), 2 * q + 8 * (j >> 1));
                        am[row][col] = bf(x & 0xffff);
                        am[row][col + 1] = bf(x >> 16);
                    }
                    for (j, v) in b.iter().enumerate() {
                        let x = rd(self, v)?;
                        let k = 2 * q + 8 * j;
                        bm[k][g] = bf(x & 0xffff);
                        bm[k + 1][g] = bf(x >> 16);
                    }
                    for (j, v) in c.iter().enumerate() {
                        cm[g + 8 * (j >> 1)][2 * q + (j & 1)] = f32::from_bits(rd(self, v)?);
                    }
                }
                let odd = |x: f32| !x.is_finite() || (x != 0.0 && !x.is_normal());
                if am
                    .iter()
                    .flatten()
                    .chain(bm.iter().flatten())
                    .any(|&x| odd(x))
                {
                    return Err("HMMA input (subnormal/Inf/NaN) is outside the calibrated model: not modeled".into());
                }
                if cm.iter().flatten().any(|&x| odd(x)) {
                    return Err("HMMA accumulator (subnormal/Inf/NaN) is outside the calibrated model: not modeled".into());
                }
                let model = fp::HMMA_MODEL;
                // Numerics: the 16-bit provenance of every operand element and the
                // accumulators' nodes, read before any result is written (d may be c).
                use crate::halfin::HalfOp;
                let mut ap = [[HalfOp::Unknown; 16]; 16];
                let mut bp = [[HalfOp::Unknown; 8]; 16];
                let mut cn: [[Option<u32>; 8]; 16] = [[None; 8]; 16];
                if self.num.is_some() {
                    for &t in &active {
                        let l = self.threads[t].lane as usize;
                        let (g, q) = (l / 4, l % 4);
                        for (j, v) in a.iter().enumerate() {
                            let hb = self.half_val(t, v).unwrap_or([None; 4]);
                            let (row, col) = (g + 8 * (j & 1), 2 * q + 8 * (j >> 1));
                            ap[row][col] = self.half_op(hb[0], hb[1]);
                            ap[row][col + 1] = self.half_op(hb[2], hb[3]);
                        }
                        for (j, v) in b.iter().enumerate() {
                            let hb = self.half_val(t, v).unwrap_or([None; 4]);
                            let k = 2 * q + 8 * j;
                            bp[k][g] = self.half_op(hb[0], hb[1]);
                            bp[k + 1][g] = self.half_op(hb[2], hb[3]);
                        }
                        for (j, v) in c.iter().enumerate() {
                            let (row, col) = (g + 8 * (j >> 1), 2 * q + (j & 1));
                            cn[row][col] = self.num_acc_node(t, v, cm[row][col].to_bits(), pc);
                        }
                    }
                }
                for (l, &t) in lanes.iter().enumerate() {
                    let (g, q) = (l / 4, l % 4);
                    for j in 0..4usize {
                        let (row, col) = (g + 8 * (j >> 1), 2 * q + (j & 1));
                        let terms: Vec<fp::Term> = (0..16)
                            .map(|k| {
                                fp::Term::prod(
                                    am[row][k] as f64,
                                    bm[k][col] as f64,
                                    model.exp_from_inputs,
                                )
                            })
                            .collect();
                        let r = model.apply_terms(cm[row][col], &terms);
                        let dr = Reg {
                            file: d.file,
                            idx: d.idx + j as u16,
                        };
                        self.wr(t, dr, r.to_bits() as u64, 1);
                        if self.num.is_some() {
                            // ASSUME(N-HMMA): one step of the calibrated model per instruction.
                            let prods: Vec<f64> = terms.iter().map(|x| x.v).collect();
                            let ops: Vec<HalfOp> =
                                (0..16).flat_map(|k| [ap[row][k], bp[k][col]]).collect();
                            self.num_hmma(
                                t,
                                dr,
                                (cn[row][col], cm[row][col]),
                                (&prods, &ops),
                                r,
                                pc,
                                false,
                            );
                        }
                    }
                }
                for &t in &active {
                    self.tick(t);
                }
            }
            Op::Shfl {
                d,
                a,
                lane,
                clamp,
                mode,
                p,
            } => {
                let mut by_lane: [Option<usize>; 32] = [None; 32];
                for &t in &active {
                    by_lane[self.threads[t].lane as usize] = Some(t);
                }
                let mut out = Vec::new();
                for &t in &active {
                    // PTX shfl: c = segmask << 8 | clamp.
                    let c = self.rd(t, clamp)? as u32;
                    let me = self.threads[t].lane;
                    let seg = (c >> 8) & 31;
                    let min_lane = me & seg;
                    let max_lane = min_lane | (c & 31 & !seg);
                    let b = self.rd(t, lane)? as u32 & 31;
                    let (j, ok) = match mode {
                        ShflMode::Idx => {
                            let j = min_lane | (b & !seg);
                            (j, j <= max_lane)
                        }
                        ShflMode::Up => (me.wrapping_sub(b), me >= max_lane + b),
                        ShflMode::Down => (me + b, me + b <= max_lane),
                        ShflMode::Bfly => ((me ^ b) & 31, (me ^ b) <= max_lane),
                    };
                    let src = if ok { j as usize } else { me as usize };
                    let Some(st) = by_lane[src] else {
                        return Err(format!(
                            "shfl reads lane {src}, which is not executing it (result undefined)"
                        ));
                    };
                    out.push((
                        t,
                        self.rd(st, a)?,
                        self.val_umask(st, a),
                        ok,
                        self.num_node_mat(st, a),
                        self.half_val(st, a),
                    ));
                    // The destination carries the source lane's taint (N-EQUIV).
                    if let (Some(ns), Val::Reg(r, _)) = (self.num.as_mut(), a) {
                        let x = ns.taint.reg_tainted(st, *r);
                        ns.taint.shfl_src.push((t, x));
                    }
                }
                // ASSUME(S-INIT): the source lane's undefined bits stay undefined.
                for (t, v, um, ok, n, hb) in out {
                    self.half_set_reg(t, *d, hb);
                    if let Some(p) = p {
                        self.wp(t, *p, ok);
                    }
                    self.vmask = (um != 0).then_some(um);
                    self.wr(t, *d, v, 1);
                    self.vmask = None;
                    // The moved value keeps its node (lane choice is not data-dependent: taint).
                    self.num_put(t, *d, n);
                }
            }
            Op::VoteAll { d, a, uni } => {
                let v = active.iter().all(|&t| self.prd(t, *a))
                    || *uni && active.iter().all(|&t| !self.prd(t, *a));
                for &t in &active {
                    self.wp(t, *d, v);
                }
            }
            // ASSUME(E-ELECT): see ASSUMPTIONS.md
            Op::Elect { d, lane } => {
                let lowest = active.iter().copied().min_by_key(|&t| self.threads[t].lane);
                let leader = match self.elect_choice.take() {
                    Some(c) if active.contains(&c) => Some(c),
                    _ => lowest,
                };
                for &t in &active {
                    self.wp(t, *d, Some(t) == leader);
                }
                if let (Some(l), Some(r)) = (leader, lane) {
                    let ln = self.threads[l].lane as u64;
                    self.wr(l, *r, ln, 1);
                }
            }
            _ if inst.uniform => {
                if let Some(&rep) = active.first() {
                    match self.exec_thread(rep, inst, pc)? {
                        Flow::Next => {}
                        _ => return Err("uniform instruction with non-trivial control flow".into()),
                    }
                }
            }
            _ => {
                let rev = self.atom_lanes_rev && matches!(inst.op, Op::Atom { .. });
                let n = members.len();
                for i in 0..n {
                    let t = members[if rev { n - 1 - i } else { i }];
                    let flow = if active.contains(&t) {
                        self.exec_thread(t, inst, pc)?
                    } else {
                        Flow::Next
                    };
                    let th = self.threads[t].make_mut();
                    match flow {
                        Flow::Next if th.st == St::Run => th.pc = pc + 1,
                        Flow::Jump(x) => th.pc = x,
                        _ => {}
                    }
                }
                return Ok(());
            }
        }
        for &t in members {
            self.threads[t].make_mut().pc = pc + 1;
        }
        Ok(())
    }

    fn slots_of(v: &Val) -> u8 {
        match v {
            Val::Reg(_, n) => *n,
            Val::Const(_, b) => (*b / 4).max(1),
            _ => 1,
        }
    }

    /// Release the mask group `mask` parked at the `WARPSYNC.COLLECTIVE` at `pc` once all of
    /// its live lanes are parked there and no other lane of the warp is inside the region
    /// (between `pc` and its `ENDCOLLECTIVE`): groups run the region one at a time.
    /// ASSUME(E-COLLECTIVE)
    fn collective_release(&mut self, warp: u32, mask: u32, pc: u32) {
        let end = (pc as usize + 1..self.k.code.len())
            .find(|&q| matches!(self.k.code[q].op, Op::EndCollective { start } if start == pc as usize))
            .unwrap_or(pc as usize) as u32;
        let busy = self
            .threads
            .iter()
            .any(|th| th.warp == warp && th.st != St::Exited && th.pc > pc && th.pc <= end);
        let ready = self.threads.iter().all(|th| {
            th.warp != warp
                || (mask >> th.lane) & 1 == 0
                || th.st == St::Exited
                || (th.st == St::WarpSync && th.pc == pc)
        });
        if !busy && ready {
            self.warpsync_release(warp, mask, pc);
        }
    }

    fn warpsync_release(&mut self, warp: u32, mask: u32, pc: u32) {
        let lanes: Vec<usize> = (0..self.threads.len())
            .filter(|&i| {
                self.threads[i].warp == warp
                    && (mask >> self.threads[i].lane) & 1 == 1
                    && self.threads[i].st != St::Exited
            })
            .collect();
        // ASSUME(E-BSYNC): a lane parked at another convergence barrier (a nested BSSY/BSYNC
        // region: ptxas's division slow path inside a loop body's B0 region uses B1) is not
        // waiting here.
        let same = |i: usize| match (
            &self.k.code[pc as usize].op,
            &self.k.code[self.threads[i].pc as usize].op,
        ) {
            (Op::Bsync { b }, Op::Bsync { b: c }) => b == c,
            (Op::WarpSync { .. }, Op::WarpSync { .. }) => true,
            _ => false,
        };
        if lanes
            .iter()
            .all(|&i| self.threads[i].st == St::WarpSync && same(i))
        {
            if self.wlog.is_on() {
                crate::wellsync::warpsync_released(self, warp, mask, &lanes); // wellsync.rs
            }
            let mut j = VC::new();
            for &i in &lanes {
                vc_join(&mut j, &self.threads[i].vc);
            }
            for &i in &lanes {
                let th = self.threads[i].make_mut();
                th.vc = j.clone();
                vc_tick(&mut th.vc, i as u32);
                th.st = St::Run;
                th.pc += 1;
            }
            // The released lanes have passed this convergence barrier.
            if let Op::Bsync { b } = self.k.code[pc as usize].op {
                let done = lanes
                    .iter()
                    .fold(0u32, |m, &i| m | 1 << self.threads[i].lane);
                self.warps[warp as usize].make_mut().bmask[b as usize] &= !done;
            }
        }
    }

    /// A peek in `peekskip`: run it as a failed test here only if every lane of the split
    /// at it agrees on its guard (the static proof treats the lanes alike).
    fn peekskip_here(&self, t: usize, pc: u32) -> bool {
        let g = self.k.code[pc as usize].guard;
        let w = self.threads[t].warp;
        let v = self.prd(t, g);
        self.warp_threads(w).all(|u| {
            let th = &self.threads[u];
            th.warp != w || th.st != St::Run || th.pc != pc || self.prd(u, g) == v
        })
    }

    /// Thread `t`, with open peek windows (`Thread::peekwin`), is about to run `inst` at
    /// `pc`: windows end where their worlds joined; at a wait one skips, the wait must be on
    /// the peek's mbarrier and parity (else `unmodeled`), and the window ends once it passes.
    fn peekwin_step(&mut self, t: usize, inst: &Inst, pc: u32) -> Result<(), String> {
        let cfg = self.bar_cfg.clone();
        let wins = self.threads[t].peekwin.clone();
        let mut keep = Vec::new();
        for w in wins {
            let Some((joins, ws)) = cfg.peekskip.get(&w.3) else {
                continue;
            };
            if joins.contains(&pc) {
                continue;
            }
            if ws.contains(&pc) {
                if let Op::MbarTryWait {
                    addr,
                    parity,
                    bit31,
                    ..
                } = &inst.op
                {
                    let a = self.addr(t, addr)?;
                    let tgt = match self.resolve(t, Space::Shared, a, 8)? {
                        Loc::Shared { cta, off } => Some((cta, off)),
                        _ => None,
                    };
                    let p = self.rd(t, parity)?;
                    let p = if *bit31 { (p >> 31) & 1 } else { p & 1 } as u32;
                    if tgt != Some((w.0, w.1)) || p != w.2 {
                        let msg = format!(
                            "{} waits at {} where the peek at {} it follows was run as a failed test, but on another mbarrier or parity (E-SPIN peek skip)",
                            self.actor_name(t as u32),
                            self.k.describe(pc as usize),
                            self.k.describe(w.3 as usize)
                        );
                        self.report("unmodeled", format!("peekskip-target:{}", w.3), msg);
                        continue;
                    }
                    let passes = matches!(self.try_wait_result(t), Ok(Some(_)));
                    if passes && !cfg.peekskip.contains_key(&pc) {
                        continue;
                    }
                }
            }
            keep.push(w);
        }
        if keep != self.threads[t].peekwin {
            self.threads[t].make_mut().peekwin = keep;
        }
        Ok(())
    }

    /// Mbarrier (cta, off) is completing its phase `ph`: a peek window open on it whose test
    /// would pass now sees an ABA (the success world went on, the failure world will wait).
    fn peekwin_aba(&mut self, cta: u16, off: u32, ph: u32) {
        let hits: Vec<(usize, u32)> = self
            .live_threads()
            .flat_map(|(u, th)| {
                th.peekwin
                    .iter()
                    .filter(|w| w.0 == cta && w.1 == off && ph % 2 != w.2)
                    .map(move |w| (u, w.3))
                    .collect::<Vec<_>>()
            })
            .collect();
        for (u, peek) in hits {
            let msg = format!(
                "{}: the mbarrier its peek at {} tests completed another phase before the thread reached the wait it skips; the peek was run as a failed test (E-SPIN peek skip), which does not cover this order",
                self.actor_name(u as u32),
                self.k.describe(peek as usize)
            );
            self.report("unmodeled", format!("peekskip-aba:{peek}"), msg);
        }
    }

    /// The explorer's look-ahead (`explore::lookahead`): run the register-only instruction at
    /// `pc` for thread `t` of this scratch machine. False if it did not simply fall through.
    pub(crate) fn exec_pure(&mut self, t: usize, pc: u32) -> bool {
        let k = self.k.clone();
        self.threads[t].make_mut().pending_ld.clear();
        matches!(
            self.exec_thread(t, &k.code[pc as usize], pc),
            Ok(Flow::Next)
        )
    }

    /// Undefined bits of operand `v` (slot i in bits 32i..; a V register: all or none).
    fn val_umask(&self, t: usize, v: &Val) -> u64 {
        match v {
            Val::Reg(r, n) if r.file != File::V => (0..(*n).clamp(1, 2) as u16)
                .map(|i| {
                    (self.reg_umask(
                        t,
                        Reg {
                            file: r.file,
                            idx: r.idx + i,
                        },
                    ) as u64)
                        << (32 * i)
                })
                .fold(0, |a, b| a | b),
            Val::Reg(r, _) if self.reg_umask(t, *r) != 0 => !0,
            _ => 0,
        }
    }

    fn report_uninit(&mut self, t: usize, r: Reg, pc: u32) {
        let name = if r.file == File::V {
            self.k
                .vreg_names_at(pc as usize)
                .get(r.idx as usize)
                .cloned()
                .unwrap_or_default()
        } else {
            format!("{:?}{}", r.file, r.idx)
        };
        let msg = format!(
            "{} reads register {name}, which nothing has written (or was loaded from shared memory nothing wrote), at {}",
            self.actor_name(t as u32),
            self.k.describe(pc as usize)
        );
        self.report("violation", format!("uninit-reg:{pc}"), msg);
    }

    fn exec_thread(&mut self, t: usize, inst: &Inst, pc: u32) -> Result<Flow, String> {
        self.stale = None;
        self.vmask = None;
        let r = self.exec_thread_inner(t, inst, pc);
        self.stale = None;
        self.vmask = None;
        r
    }

    /// S-INIT for a register-only operation that read undefined bits: the undefined bits of
    /// its result (bit-precise where a byte permute, shift, mask, select, pack or low-bits
    /// arithmetic discards undefined input bits; otherwise the whole result), or None when
    /// the read itself is reported (any other operation, or a predicate result that depends
    /// on an undefined bit).
    fn undef_result(&self, t: usize, op: &Op) -> Option<u64> {
        let u = |v: &Val| self.val_umask(t, v);
        let u32_ = |v: &Val| self.val_umask(t, v) as u32;
        let val = |v: &Val| self.rd(t, v).unwrap_or(0);
        // Bits at and above the lowest undefined one (carries propagate upward).
        let up = |m: u32| {
            if m == 0 {
                0
            } else {
                !0u32 << m.trailing_zeros()
            }
        };
        let all = |m: u64| if m != 0 { !0u64 } else { 0 };
        let m: u64 = match op {
            Op::Mov { a, .. } => u(a),
            Op::Sel { a, b, p, .. } | Op::Fsel { a, b, p, .. } => {
                if self.prd(t, *p) {
                    u(a)
                } else {
                    u(b)
                }
            }
            Op::Prmt { a, sel, b, .. } => {
                if u(sel) & 0xffff != 0 {
                    return Some(!0);
                }
                let (ua, ub) = (u32_(a), u32_(b));
                let src = ua as u64 | (ub as u64) << 32;
                let sl = val(sel);
                let mut m = 0u32;
                for i in 0..4 {
                    let n = (sl >> (4 * i)) & 0xf;
                    let byte = (src >> (8 * (n & 7))) & 0xff;
                    let um = if n & 8 != 0 {
                        // sign replication: the selected byte's top bit
                        if byte & 0x80 != 0 { 0xff } else { 0 }
                    } else {
                        byte
                    };
                    m |= (um as u32) << (8 * i);
                }
                m as u64
            }
            Op::Shf {
                lo,
                sh,
                hi,
                right,
                signed,
                high,
                ..
            } => {
                if u(sh) != 0 {
                    return Some(!0);
                }
                let v = (u32_(hi) as u64) << 32 | u32_(lo) as u64;
                let s = (val(sh) & 63) as u32;
                let r = if *right {
                    if *signed && v >> 63 == 1 {
                        (v >> s) | !(u64::MAX >> s)
                    } else {
                        v >> s
                    }
                } else {
                    v << s
                };
                (if *high { r >> 32 } else { r & 0xffff_ffff }) as u32 as u64
            }
            Op::Int {
                op, bits: 32, a, b, ..
            } => {
                let (ua, ub) = (u32_(a), u32_(b));
                let (x, y) = (val(a) as u32, val(b) as u32);
                (match op {
                    IntOp::Shl | IntOp::ShrU | IntOp::ShrS if ub != 0 => !0,
                    IntOp::Shl => ua.checked_shl(y).unwrap_or(0),
                    IntOp::ShrU => ua.checked_shr(y).unwrap_or(0),
                    IntOp::ShrS => ((ua as i32) >> y.min(31)) as u32,
                    IntOp::And => (ua | ub) & !((!x & !ua) | (!y & !ub)),
                    IntOp::Or => (ua | ub) & !((x & !ua) | (y & !ub)),
                    IntOp::Xor => ua | ub,
                    IntOp::Add | IntOp::Sub => up(ua | ub),
                    _ => {
                        if ua | ub != 0 {
                            !0
                        } else {
                            0
                        }
                    }
                }) as u64
            }
            Op::Lop3 {
                a, b, c, lut, p, ..
            } => {
                let us = [u32_(a), u32_(b), u32_(c)];
                let xs = [val(a) as u32, val(b) as u32, val(c) as u32];
                let mut m = 0u32;
                for bit in 0..32 {
                    let free: Vec<usize> = (0..3).filter(|&k| us[k] >> bit & 1 == 1).collect();
                    if free.is_empty() {
                        continue;
                    }
                    let mut outs = 0u8;
                    for combo in 0..1u32 << free.len() {
                        let mut v = xs.map(|x| x >> bit & 1);
                        for (j, &k) in free.iter().enumerate() {
                            v[k] = combo >> j & 1;
                        }
                        let i = v[0] << 2 | v[1] << 1 | v[2];
                        outs |= 1 << (lut >> i & 1);
                    }
                    if outs == 0b11 {
                        m |= 1 << bit;
                    }
                }
                if p.is_some() && m != 0 {
                    return None; // the predicate depends on an undefined bit
                }
                m as u64
            }
            Op::Imad {
                a,
                b,
                c,
                wide: false,
                hi: false,
                cout: None,
                cin: None,
                ..
            } => {
                let (ua, ub, uc) = (u32_(a), u32_(b), u32_(c));
                let (x, y) = (val(a) as u32, val(b) as u32);
                // a*b: bits below lowest-undefined(a) + trailing zeros of a defined b are set.
                let pa = if ua == 0 {
                    0
                } else if ub == 0 {
                    up(ua.checked_shl(y.trailing_zeros()).unwrap_or(0))
                } else {
                    !0
                };
                let pb = if ub == 0 {
                    0
                } else if ua == 0 {
                    up(ub.checked_shl(x.trailing_zeros()).unwrap_or(0))
                } else {
                    !0
                };
                up(pa | pb | uc) as u64
            }
            Op::Iadd3 {
                a,
                b,
                c,
                wide: false,
                cout,
                cin,
                ..
            } if cout.is_empty() && cin.is_empty() => up(u32_(a) | u32_(b) | u32_(c)) as u64,
            Op::Lea {
                a,
                b,
                shift,
                cout: None,
                neg_a: false,
                ..
            } => up((u32_(a).checked_shl(*shift as u32).unwrap_or(0)) | u32_(b)) as u64,
            Op::F2fpBf16 { a, b, .. } => {
                (if u(a) != 0 { 0xffff_0000u64 } else { 0 }) | if u(b) != 0 { 0xffff } else { 0 }
            }
            op if propagates_undef(op)
                && op
                    .dst_regs()
                    .iter()
                    .all(|d| matches!(d.file, File::R | File::UR | File::V)) =>
            {
                all(op.src_vals().iter().map(u).fold(0, |x, y| x | y))
            }
            _ => return None,
        };
        // A result that is still wholly defined needs no mask.
        Some(m)
    }

    fn exec_thread_inner(&mut self, t: usize, inst: &Inst, pc: u32) -> Result<Flow, String> {
        if !self.threads[t].peekwin.is_empty() {
            self.peekwin_step(t, inst, pc)?;
        }
        let mask = |bits: u8, v: u64| {
            if bits >= 64 {
                v
            } else {
                v & ((1u64 << bits) - 1)
            }
        };
        if !self.threads[t].pending_ld.is_empty() {
            for r in inst.op.src_regs() {
                let key = (r.file as u8, r.idx);
                if self.threads[t].pending_ld.contains(&key) {
                    let name = if r.file == File::V {
                        self.k
                            .vreg_names_at(pc as usize)
                            .get(r.idx as usize)
                            .cloned()
                            .unwrap_or_default()
                    } else {
                        format!("{:?}{}", r.file, r.idx)
                    };
                    let msg = format!(
                        "{} reads {name}, written by tcgen05.ld, before tcgen05.wait::ld (value undefined) at {}",
                        self.actor_name(t as u32),
                        self.k.describe(pc as usize)
                    );
                    self.report("violation", format!("ld-unwaited:{pc}"), msg);
                    // Report once per load: forget the rest of its pending registers.
                    self.threads[t].make_mut().pending_ld.clear();
                    break;
                }
            }
        }
        // ASSUME(S-INIT): registers start undefined; reading one before any write is a bug
        // (hardware leaves whatever the previous kernel had). A register-only operation that
        // writes only data registers makes its result undefined instead (compilers compute on
        // lanes whose values they discard: Triton's cross-warp reductions shuffle registers
        // that only some lanes loaded); the read is reported where an undefined value reaches
        // memory, an address, a predicate, control flow or a synchronization operand.
        // A sub-word store reads only the low bytes of its data register (`STG.E.U16` of a
        // packed bf16 pair's low half).
        let low_only = match &inst.op {
            Op::St {
                bytes: b @ (1 | 2),
                v: Val::Reg(r, _),
                ..
            } if r.file != File::V => Some((*r, (1u32 << (8 * *b as u32)) - 1)),
            _ => None,
        };
        // A spill (STL) keeps its data's undefined bits in local memory (LDL restores them).
        let spill = match &inst.op {
            Op::StLocal {
                v: Val::Reg(r, _), ..
            } => Some(r.file),
            _ => None,
        };
        let undef = inst.op.src_regs().into_iter().find(|r| {
            if let (
                Some(f),
                Op::StLocal {
                    v: Val::Reg(vr, _),
                    bytes,
                    ..
                },
            ) = (spill, &inst.op)
            {
                if r.file == f && r.idx >= vr.idx && r.idx < vr.idx + (*bytes as u16 / 4).max(1) {
                    return false;
                }
            }
            let u = self.reg_umask(t, *r);
            match low_only {
                Some((v, m)) if v == *r => u & m != 0,
                _ => u != 0,
            }
        });
        // A compare whose combining predicate decides the result alone (`ISETP.NE.OR P6, PT,
        // R18, RZ, P2` with P2 set: torch's `where` loads R18 only under !P2) does not use
        // the undefined operand.
        let decided = match &inst.op {
            Op::Setp {
                and,
                or,
                ex: None,
                xor: false,
                ..
            } => self.prd(t, *and) == *or,
            _ => false,
        };
        if let (Some(r), false) = (undef, decided) {
            match self.undef_result(t, &inst.op) {
                Some(m) => self.vmask = Some(m),
                None => self.report_uninit(t, r, pc),
            }
        }
        match &inst.op {
            Op::EndCollective { .. } => {}
            Op::CvtInt {
                d,
                a,
                from_bits,
                to_bits,
                signed,
            } => {
                let v = mask(*from_bits, self.rd(t, a)?);
                let w = *from_bits as u32;
                let v = if *signed && w < 64 {
                    (((v << (64 - w)) as i64) >> (64 - w)) as u64
                } else {
                    v
                };
                self.wr(t, *d, mask(*to_bits, v), (*to_bits / 32).max(1));
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::Flo { d, a, sh } => {
                let x = self.rd(t, a)? as u32;
                self.wr(
                    t,
                    *d,
                    if x == 0 {
                        0xffff_ffff
                    } else if *sh {
                        x.leading_zeros() as u64
                    } else {
                        31 - x.leading_zeros() as u64
                    },
                    1,
                );
            }
            Op::Atom {
                op,
                space,
                bytes,
                addr,
                v,
                d,
                sync,
                p,
                new,
            } => {
                let a = self.addr(t, addr)?;
                let loc = self.resolve(t, *space, a, *bytes as usize)?;
                let wide = *bytes == 8;
                let bmask = if wide { u64::MAX } else { 0xffff_ffff };
                let x = self.rd(t, v)? & bmask;
                // CAS: `v` is the compare value; the new value is `new`, or the register (pair)
                // right after `v` (SASS {compare, new} pairs).
                let (x, cmp) = match (op, v, new) {
                    (AtomOp::CasSpin | AtomOp::Cas, _, Some(nv)) => (self.rd(t, nv)? & bmask, x),
                    (AtomOp::CasSpin | AtomOp::Cas, Val::Reg(r, _), None) => {
                        let step = if wide && r.file != File::V { 2 } else { 1 };
                        let hi = Reg {
                            file: r.file,
                            idx: r.idx + step,
                        };
                        let n = if wide { 2 } else { 1 };
                        (self.rd(t, &Val::Reg(hi, n))? & bmask, x)
                    }
                    (AtomOp::CasSpin | AtomOp::Cas, _, None) => {
                        return Err("CAS without a new-value operand".into());
                    }
                    _ => (x, 0),
                };
                let vc = self.threads[t].vc.clone();
                let sc = strong_code(sync.scope);
                if self.l1.is_some() {
                    self.l1_atom(t, loc, *bytes as usize); // ASSUME(E-HW-L1)
                }
                self.red_access = *op == AtomOp::FAddFtz && d.is_none();
                let old = self.access_x(t as u32, &vc, pc, loc, *bytes as usize, None, sc);
                self.sync_observe(t, loc, *sync);
                let mut o = 0u64;
                for (i, b) in old.iter().enumerate() {
                    o |= (*b as u64) << (8 * i);
                }
                let n = match op {
                    AtomOp::Or => o | x,
                    AtomOp::And => o & x,
                    AtomOp::Add => o.wrapping_add(x),
                    AtomOp::Exch => x,
                    AtomOp::Xor => o ^ x,
                    AtomOp::MinU if wide => o.min(x),
                    AtomOp::MaxU if wide => o.max(x),
                    AtomOp::MinS if wide => (o as i64).min(x as i64) as u64,
                    AtomOp::MaxS if wide => (o as i64).max(x as i64) as u64,
                    AtomOp::MinU => (o as u32).min(x as u32) as u64,
                    AtomOp::MaxU => (o as u32).max(x as u32) as u64,
                    AtomOp::MinS => (o as i32).min(x as i32) as u32 as u64,
                    AtomOp::MaxS => (o as i32).max(x as i32) as u32 as u64,
                    // ASSUME(N-FALU): as FADD.FTZ (round to nearest, flush).
                    AtomOp::FAddFtz => fp::f32_fma(
                        o as u32,
                        0x3f80_0000,
                        Some(x as u32),
                        ptxv_ir::Rnd::Rn,
                        true,
                    ) as u64,
                    AtomOp::CasSpin | AtomOp::Cas => {
                        if o == cmp {
                            x
                        } else {
                            o
                        }
                    }
                };
                let swapped = matches!(op, AtomOp::CasSpin | AtomOp::Cas) && o == cmp;
                // A failed `Cas` is a read only (no write, nothing published): a poll loop
                // of failing compare-and-swaps leaves memory untouched.
                if *op != AtomOp::Cas || swapped {
                    let data: Vec<u8> = (0..*bytes).map(|i| (n >> (8 * i)) as u8).collect();
                    self.access_x(t as u32, &vc, pc, loc, *bytes as usize, Some(&data), sc);
                    self.sync_publish(t, loc, *sync, true);
                }
                self.red_access = false;
                if let Some(d) = d {
                    let (r, w) = if *op == AtomOp::CasSpin {
                        (swapped as u64, 1)
                    } else {
                        (o, if wide { 2 } else { 1 })
                    };
                    self.wr(t, *d, r, w);
                }
                if let Some(p) = p {
                    // Generic: the address resolved to global memory (performed).
                    let v = if *op == AtomOp::CasSpin {
                        swapped
                    } else {
                        true
                    };
                    self.wp(t, *p, v);
                }
                let fadd = (*op == AtomOp::FAddFtz).then_some((o as u32, x as u32, n as u32));
                let fmax = matches!(op, AtomOp::MaxS | AtomOp::MinU)
                    && self.num_fmax_atom(t, v, loc, (o as u32, x as u32, n as u32), pc);
                if !fmax {
                    self.num_atom(t, v, loc, fadd, *d, pc);
                }
                // A failed compare-and-swap only read, like a load: no new epoch (otherwise
                // every failed attempt of a lane that polls alongside a succeeding one would
                // be a distinct state).
                if *op != AtomOp::Cas || swapped {
                    self.tick(t);
                }
            }
            Op::Call { target } => return Ok(Flow::Jump(*target as u32)),
            Op::Brx { base, off, targets } => {
                let a = (self.rd(t, base)? as i64).wrapping_add(*off);
                let to = (a >> 4) as usize + self.k.entry_of(pc as usize);
                if a & 15 != 0 || !targets.contains(&to) {
                    return Err(format!(
                        "BRX to code address {a:#x}, not one of its branch targets"
                    ));
                }
                return Ok(Flow::Jump(to as u32));
            }
            Op::Ret { addr } => {
                // A code address within this kernel (16 bytes per SASS instruction).
                let a = self.rd(t, addr)?;
                let to = (a >> 4) as u32 + self.k.entry_of(pc as usize) as u32;
                // ASSUME(E-RET): a return lands right after a call of its function (the
                // explorer's control-flow graph, `explore::ret_table`); checked on every return.
                let site = self.bar_cfg.rets.is_empty()
                    || self
                        .bar_cfg
                        .rets
                        .get(pc as usize)
                        .is_some_and(|r| r.binary_search(&(to as usize)).is_ok());
                if !site {
                    let msg = format!(
                        "RET at {} to code address {a:#x}, which is not right after a CALL of its function: the explorer's control-flow graph does not cover it (E-RET)",
                        self.k.describe(pc as usize)
                    );
                    self.report("unmodeled", format!("ret-target:{pc}"), msg);
                }
                return Ok(Flow::Jump(to));
            }
            Op::CallAbs { target, syms } => {
                let a = self.rd(t, target)?;
                let name = syms.iter().find(|(x, _)| *x == a).map(|(_, n)| n.as_str());
                // DeepEP's device asserts print before `BPT.TRAP`: the print is not modeled
                // (it has no effect on the kernel), the trap that follows is. ASSUME(E-CLOCK)
                if name == Some("vprintf") {
                    return Ok(Flow::Next);
                }
                if name != Some("__assertfail") {
                    return Err(format!(
                        "CALL.ABS to {} not modeled",
                        name.map(String::from).unwrap_or(format!("{a:#x}"))
                    ));
                }
                if let Some(f) = self.fp.as_mut() {
                    f.all = true;
                }
                let msg = format!(
                    "device-side assertion failed (__assertfail) in {}: the grid is aborted",
                    self.actor_name(t as u32)
                );
                self.report("trap", format!("assert:{pc}"), msg);
                for th in self.threads.iter_mut() {
                    th.make_mut().st = St::Exited;
                }
                self.pending.clear();
                self.completed.clear();
                return Ok(Flow::Stay);
            }
            Op::Trap { why } => {
                if let Some(f) = self.fp.as_mut() {
                    f.all = true;
                }
                let msg = format!(
                    "kernel trap (BPT.TRAP) by {} in `{}`: the grid is aborted",
                    self.actor_name(t as u32),
                    if why.is_empty() { "?" } else { why }
                );
                self.report("trap", format!("trap:{why}"), msg);
                for th in self.threads.iter_mut() {
                    th.make_mut().st = St::Exited;
                }
                self.pending.clear();
                self.completed.clear();
                return Ok(Flow::Stay);
            }
            Op::TmemAlloc { dst, ncols } => {
                {
                    let c = self.threads[t].cta as u32;
                    self.rec(3, c, 0, 1, true);
                }
                let n = self.rd(t, ncols)? as u32;
                if !(32..=512).contains(&n) || !n.is_power_of_two() {
                    return Err(format!(
                        "tcgen05.alloc of {n} columns (must be a power of two in [32, 512])"
                    ));
                }
                let cta = self.threads[t].cta;
                if self.tmem[cta as usize].relinquished {
                    self.report(
                        "violation",
                        format!("alloc-after-relinquish:{pc}"),
                        format!(
                            "tcgen05.alloc after tcgen05.relinquish_alloc_permit at {}",
                            self.k.describe(pc as usize)
                        ),
                    );
                }
                let units = n / 32;
                let idx = match self.tenant_place(&[cta], units) {
                    Some(p) if !self.tmem_permit_blocked(cta) => p,
                    _ => self
                        .tmem_find(cta, units)
                        .ok_or("tcgen05.alloc cannot succeed")?,
                };
                self.tmem[cta as usize].alloc |= (((1u32 << units) - 1) << idx) as u16;
                self.tmem[cta as usize].permit = true;
                let col = idx * 32;
                // ASSUME(S-TCGEN05): alloc is synchronous (PTX ISA 9.7.18.6.1): the thread itself
                // writes the address, ordered by ordinary synchronization.
                let a = self.addr(t, dst)?;
                let loc = self.resolve(t, Space::Shared, a, 4)?;
                let vc = self.threads[t].vc.clone();
                self.access(t as u32, &vc, pc, loc, 4, Some(&col.to_le_bytes()));
                self.tmem_reset(cta, col, n, t as u32, &vc, pc);
            }
            Op::TmemAllocRaw { d, ok, units, pair } => {
                let n = self.rd(t, units)? as u32;
                let cta = self.threads[t].cta;
                let ctas = self.alloc_group(cta, *pair)?;
                for &c in &ctas {
                    self.rec(3, c as u32, 0, 1, true);
                }
                let idx = match self.tenant_place(&ctas, n) {
                    Some(p) if !ctas.iter().any(|&c| self.tmem_permit_blocked(c)) => p,
                    _ => self
                        .tmem_find_group(&ctas, n)
                        .ok_or("UTCATOMSWS.FIND_AND_SET cannot succeed")?,
                };
                let vc = self.threads[t].vc.clone();
                for &c in &ctas {
                    self.tmem[c as usize].alloc |= (((1u32 << n) - 1) << idx) as u16;
                    self.tmem[c as usize].permit = true;
                    self.tmem_reset(c, idx * 32, n * 32, t as u32, &vc, pc);
                }
                self.wr(t, *d, idx as u64, 1);
                self.wp(t, *ok, true);
                self.tick(t);
            }
            // ASSUME(E-CLC): see ASSUMPTIONS.md
            Op::ClcTryCancel {
                resp,
                mbar,
                multicast,
            } => {
                if *multicast {
                    return Err("multicast clusterlaunchcontrol.try_cancel is not modeled".into());
                }
                if self.clc_eager {
                    return Err("try_cancel executed although every CTA was launched up front (E-CLC eager launch): not modeled on this path".into());
                }
                let r = self.rd(t, resp)?;
                let m = self.rd(t, mbar)?;
                let Loc::Shared { cta, off: roff } = self.resolve(t, Space::Shared, r, 16)? else {
                    unreachable!()
                };
                let Loc::Shared { off: moff, .. } = self.resolve(t, Space::Shared, m, 8)? else {
                    unreachable!()
                };
                let serial = self.threads[t].tma_count;
                self.threads[t].make_mut().tma_count += 1;
                let actor = self.tma_actor_id(t, serial);
                let mut vc = self.threads[t].vc.clone();
                vc_tick(&mut vc, actor);
                self.push_async(AsyncOp {
                    fifo: None,
                    actor,
                    vc,
                    issuer: t as u32,
                    pc,
                    kind: AsyncKind::Clc {
                        cta,
                        resp: roff,
                        mbar: moff,
                    },
                    vis: self.threads[t].vis.clone(),
                });
                self.tick(t);
            }
            Op::ClcQuery { d, q, ctaid } => {
                let lo = self.rd(t, &Val::Reg(*q, 1))?;
                let hi = self.rd(
                    t,
                    &Val::Reg(
                        Reg {
                            file: q.file,
                            idx: q.idx + 1,
                        },
                        1,
                    ),
                )?;
                match ctaid {
                    None => self.wp(t, *d, hi & 1 == 1),
                    Some(0) => self.wr(t, *d, lo & 0xffff_ffff, 1),
                    Some(_) => self.wr(t, *d, (lo >> 32) & 0xffff_ffff, 1),
                }
            }
            Op::StAsync { addr, mbar, v } => {
                let a = self.rd(t, addr)?;
                let m = self.rd(t, mbar)?;
                if self.dead_target(t, a, pc)? {
                    self.tick(t);
                    return Ok(Flow::Next);
                }
                let loc = self.resolve(t, Space::SharedCluster, a, 4)?;
                let Loc::Shared { cta, .. } = loc else {
                    unreachable!()
                };
                if cta != self.threads[t].cta {
                    self.note_remote(cta, t as u32, pc);
                }
                let x = self.rd(t, v)? as u32;
                let vc = self.threads[t].vc.clone();
                let vis = self.threads[t].vis.clone();
                // ASSUME(E-ASYNC-STEP): the store and its complete_tx happen in one step.
                self.access(t as u32, &vc, pc, loc, 4, Some(&x.to_le_bytes()));
                let Loc::Shared { cta: mc, off } = self.resolve(t, Space::SharedCluster, m, 8)?
                else {
                    unreachable!()
                };
                self.rec(0, mc as u32, off as u64, off as u64 + 8, false);
                self.mbar_atomic(t as u32, &vc, &vis, pc, mc, off)?;
                let mb = self.mbars.get_mut(&(mc, off)).unwrap();
                mb.tx -= 4;
                vc_join(&mut mb.cur, &vc);
                self.mbar_maybe_complete(mc, off);
                self.tick(t);
            }
            Op::TmemFreeRaw { mask } => {
                {
                    let c = self.threads[t].cta as u32;
                    self.rec(3, c, 0, 1, true);
                }
                let m = self.rd(t, mask)? as u16;
                let cta = self.threads[t].cta;
                let freed = self.tmem[cta as usize].alloc & !m;
                self.tmem[cta as usize].alloc &= m;
                let vc = self.threads[t].vc.clone();
                for u in 0..16u32 {
                    if freed >> u & 1 == 1 {
                        self.tmem_free(cta, u * 32, 32, t as u32, &vc, pc);
                    }
                }
                self.tick(t);
            }
            Op::TmemRelinquish => {
                {
                    let c = self.threads[t].cta as u32;
                    self.rec(3, c, 0, 1, true);
                }
                let cta = self.threads[t].cta;
                self.tmem[cta as usize].relinquished = true;
            }
            Op::TmemDealloc { taddr, ncols } => {
                {
                    let c = self.threads[t].cta as u32;
                    self.rec(3, c, 0, 1, true);
                }
                let v = self.rd(t, taddr)? as u32;
                let n = self.rd(t, ncols)? as u32;
                let (lane, col) = (v >> 16, v & 0xffff);
                if lane != 0 || col % 32 != 0 || !(32..=512).contains(&n) || !n.is_power_of_two() {
                    return Err(format!(
                        "tcgen05.dealloc of an invalid range (lane {lane}, column {col}, {n} columns)"
                    ));
                }
                let cta = self.threads[t].cta;
                let units = n / 32;
                let m = (((1u32 << units) - 1) << (col / 32)) as u16;
                if self.tmem[cta as usize].alloc & m != m {
                    self.report(
                        "violation",
                        format!("dealloc-unalloc:{pc}"),
                        format!(
                            "tcgen05.dealloc of columns {col}..{} that are not allocated",
                            col + n
                        ),
                    );
                }
                self.tmem[cta as usize].alloc &= !m;
                let ta = self.tc_actor(t);
                let mut vc = self.threads[t].tc_vc.clone();
                vc_tick(&mut vc, ta);
                self.threads[t].make_mut().tc_vc = vc.clone();
                self.tmem_free(cta, col, n, ta, &vc, pc);
            }
            Op::Mma {
                d,
                adesc,
                bdesc,
                idesc,
                enable_d,
                pair,
                disable,
                a_tmem,
                kind,
                scale,
            } => {
                for m in disable {
                    if self.rd(t, m)? != 0 {
                        return Err(
                            "tcgen05.mma with output lanes disabled is not modeled yet".into()
                        );
                    }
                }
                let d = self.rd(t, d)? as u32;
                let ad = self.rd(t, adesc)?;
                let bd = self.rd(t, bdesc)?;
                let id = self.rd(t, idesc)? as u32;
                let en = self.prd(t, *enable_d);
                let sc = match scale {
                    Some((sa, sb)) => Some((self.rd(t, sa)? as u32, self.rd(t, sb)? as u32)),
                    None => None,
                };
                let cta = self.threads[t].cta;
                // ASSUME(E-PAIR): see ASSUMPTIONS.md
                let peer = if *pair {
                    let (_, rank) = self.launch.cluster_of(cta as u32);
                    if rank % 2 != 0 {
                        return Err(format!(
                            "tcgen05.mma.cta_group::2 issued by odd CTA rank {rank} (the even CTA of the pair issues)"
                        ));
                    }
                    let p = self.rank_cta(cta, ((rank + 1) as u64) << 24)?;
                    if self.dead_target(t, ((rank + 1) as u64) << 24, pc)? {
                        self.tick(t);
                        return Ok(Flow::Next);
                    }
                    self.note_remote(p, t as u32, pc);
                    Some(p)
                } else {
                    None
                };
                let (actor, vc) = self.pipe_issue(t);
                self.push_async(AsyncOp {
                    fifo: Some(t as u32),
                    actor,
                    vc,
                    issuer: t as u32,
                    pc,
                    kind: AsyncKind::Mma {
                        cta,
                        peer,
                        d,
                        adesc: ad,
                        bdesc: bd,
                        idesc: id,
                        enable: en,
                        a_tmem: *a_tmem,
                        kind: *kind,
                        scale: sc,
                    },
                    vis: self.threads[t].vis.clone(),
                });
                self.tick(t);
            }
            Op::TcCp {
                taddr,
                off,
                sdesc,
                pair,
            } => {
                let ta = (self.rd(t, taddr)? as u32).wrapping_add(*off);
                let sd = self.rd(t, sdesc)?;
                let cta = self.threads[t].cta;
                // ASSUME(E-PAIR, S-TCCP): see ASSUMPTIONS.md
                let peer = if *pair {
                    let (_, rank) = self.launch.cluster_of(cta as u32);
                    if rank % 2 != 0 {
                        return Err(format!(
                            "tcgen05.cp.cta_group::2 issued by odd CTA rank {rank}: not modeled (the even CTA of the pair issues)"
                        ));
                    }
                    let p = self.rank_cta(cta, ((rank + 1) as u64) << 24)?;
                    if self.dead_target(t, ((rank + 1) as u64) << 24, pc)? {
                        self.tick(t);
                        return Ok(Flow::Next);
                    }
                    self.note_remote(p, t as u32, pc);
                    Some(p)
                } else {
                    None
                };
                let (actor, vc) = self.pipe_issue(t);
                self.push_async(AsyncOp {
                    fifo: Some(t as u32),
                    actor,
                    vc,
                    issuer: t as u32,
                    pc,
                    kind: AsyncKind::Cp {
                        cta,
                        peer,
                        taddr: ta,
                        sdesc: sd,
                    },
                    vis: self.threads[t].vis.clone(),
                });
                self.tick(t);
            }
            Op::TcCommit { mbar, mask, off } => {
                let m = self.rd(t, mbar)?.wrapping_add(*off as u64);
                let me = self.threads[t].cta;
                // Targets: the CTA the address names, or (multicast) every rank in the mask at
                // the same offset. Each becomes its own commit in this thread's pipe.
                let addrs: Vec<u64> = match mask {
                    None => vec![m],
                    Some(mv) => {
                        let mv = self.rd(t, mv)? & 0xffff;
                        (0..16u64)
                            .filter(|r| mv >> r & 1 == 1)
                            .map(|r| (m & 0xff_ffff) | (r << 24))
                            .collect()
                    }
                };
                for a in addrs {
                    let cta = self.rank_cta(me, a)?;
                    if cta != me {
                        if self.dead_target(t, a, pc)? {
                            continue;
                        }
                        self.note_remote(cta, t as u32, pc);
                    }
                    let (actor, vc) = self.pipe_issue(t);
                    self.push_async(AsyncOp {
                        fifo: Some(t as u32),
                        actor,
                        vc,
                        issuer: t as u32,
                        pc,
                        kind: AsyncKind::Commit {
                            cta,
                            mbar: (m & 0xff_ffff) as u32,
                        },
                        vis: self.threads[t].vis.clone(),
                    });
                }
                self.tick(t);
            }
            Op::TmemLd {
                d,
                n,
                taddr,
                off,
                tracked,
                s16x64b,
                half,
                s16x256b,
                s16x128b,
            } => {
                if half.is_some_and(|h| self.threads[t].lane / 16 != h as u32) {
                    return Ok(Flow::Next); // the other half of the warp's lanes
                }
                // The offset is part of the address (`tmem[UR9+0x100000]`: lane 16).
                let v = (self.rd(t, taddr)? as u32).wrapping_add(*off);
                let mut lane0 = (v >> 16) as usize;
                let col0 = (v & 0xffff) as usize;
                let (wq, l, cta) = {
                    let th = &self.threads[t];
                    // warp's TMEM lane quadrant
                    ((th.tid[0] / 32 % 4) as usize, th.lane as usize, th.cta)
                };
                // ASSUME(S-TMEM): GB300 takes the quadrant from the warp and only the lane
                // field mod 32 from the address (probes/tmem_16x64b; cuBLAS's epilogue passes
                // lane 0 / 16 from every warp). The PTX contract still requires the warp's own
                // lanes, so only the PTX front-end reports it.
                if self.k.front == Front::Sass {
                    lane0 = 32 * wq + lane0 % 32;
                }
                // (lane, column offset of register j)
                if *s16x256b || *s16x128b {
                    // ASSUME(S-TMEM): the mma-accumulator-like layout (r2 matmul bitwise).
                    if lane0 != 32 * wq && lane0 != 32 * wq + 16 {
                        let msg = format!(
                            "{} (warp quadrant {wq}) addresses TMEM lanes {lane0}.. at {}",
                            self.actor_name(t as u32),
                            self.k.describe(pc as usize)
                        );
                        self.report("violation", format!("tmem-quadrant:{pc}"), msg);
                    }
                    let (actor, vc) = if *tracked {
                        let a = self.tc_actor(t);
                        let th = self.threads[t].make_mut();
                        vc_tick(&mut th.tc_vc, a);
                        (a, th.tc_vc.clone())
                    } else {
                        (t as u32, self.threads[t].vc.clone())
                    };
                    let mut states = 0u8;
                    for j in 0..*n as usize {
                        // ASSUME(S-TMEM): 16x128b measured (probes/tmem_16x128b).
                        let (lane, col) = if *s16x128b {
                            (lane0 + l / 4 + 8 * (j % 2), col0 + 4 * (j / 2) + l % 4)
                        } else {
                            (
                                lane0 + l / 4 + 8 * (j / 2 % 2),
                                col0 + 8 * (j / 4) + 2 * (l % 4) + j % 2,
                            )
                        };
                        if lane >= TMEM_LANES || col >= TMEM_COLS {
                            return Err(format!(
                                "tcgen05.ld outside TMEM (lane {lane}, column {col})"
                            ));
                        }
                        let (w, st) = self.tmem_access(cta, lane, col, actor, &vc, pc, None);
                        states |= match st {
                            0 => 1,
                            2 => 2,
                            _ => 0,
                        };
                        let r = Reg {
                            file: d.file,
                            idx: d.idx + j as u16,
                        };
                        self.wr(t, r, w as u64, 1);
                        self.num_tmem_ld(t, r, cta, lane, col);
                        if *tracked {
                            self.threads[t]
                                .make_mut()
                                .pending_ld
                                .push((r.file as u8, r.idx));
                        }
                    }
                    self.tmem_state_report(states, "tcgen05.ld", pc);
                    return Ok(Flow::Next);
                }
                let (lane, cstep, cadd) = if *s16x64b {
                    (lane0 + l / 4 + 8 * (l % 2), 2, l / 2 % 2)
                } else if half.is_some() {
                    (lane0 + l % 16, 1, 0)
                } else {
                    (lane0 + l, 1, 0)
                };
                let span = cstep * *n as usize;
                let sixteen = *s16x64b || half.is_some();
                if lane0 != 32 * wq && !(sixteen && lane0 == 32 * wq + 16) {
                    let msg = format!(
                        "{} (warp quadrant {wq}) addresses TMEM lanes {lane0}.. at {}: a warp may only access lanes {}..{}",
                        self.actor_name(t as u32),
                        self.k.describe(pc as usize),
                        32 * wq,
                        32 * wq + 31
                    );
                    self.report("violation", format!("tmem-quadrant:{pc}"), msg);
                }
                if lane >= TMEM_LANES || col0 + span > TMEM_COLS {
                    return Err(format!(
                        "tcgen05.ld outside TMEM (lane {lane}, columns {col0}+{span})"
                    ));
                }
                let (actor, vc) = if *tracked {
                    let a = self.tc_actor(t);
                    let th = self.threads[t].make_mut();
                    vc_tick(&mut th.tc_vc, a);
                    (a, th.tc_vc.clone())
                } else {
                    (t as u32, self.threads[t].vc.clone())
                };
                let mut states = 0u8;
                for j in 0..*n as usize {
                    let col = col0 + cstep * j + cadd;
                    let (w, st) = self.tmem_access(cta, lane, col, actor, &vc, pc, None);
                    states |= match st {
                        0 => 1,
                        2 => 2,
                        _ => 0,
                    };
                    let r = Reg {
                        file: d.file,
                        idx: d.idx + j as u16,
                    };
                    self.wr(t, r, w as u64, 1);
                    self.num_tmem_ld(t, r, cta, lane, col);
                    if *tracked {
                        self.threads[t]
                            .make_mut()
                            .pending_ld
                            .push((r.file as u8, r.idx));
                    }
                }
                self.tmem_state_report(states, "tcgen05.ld", pc);
            }
            Op::TmemWaitLd => self.threads[t].make_mut().pending_ld.clear(),
            Op::Fence {
                kind: FenceKind::TcBefore,
            } => {
                let th = self.threads[t].make_mut();
                let tc = th.tc_vc.clone();
                vc_join(&mut th.vc, &tc);
            }
            Op::Fence {
                kind: FenceKind::TcAfter,
            } => {
                let th = self.threads[t].make_mut();
                let v = th.vc.clone();
                vc_join(&mut th.tc_vc, &v);
            }
            Op::R2Ur { .. }
            | Op::VoteAny { .. }
            | Op::Redux { .. }
            | Op::WarpSync { .. }
            | Op::Bssy { .. }
            | Op::Break { .. }
            | Op::Bsync { .. } => {
                return Err("warp-collective op reached the per-thread executor".into());
            }
            // ASSUME(S-PROXY): record where in this thread's history the fence sits (and what
            // it observed); the async proxy sees the generic writes that happen-before the
            // fence once the fence HB its op.
            Op::Fence {
                kind: FenceKind::ProxyAsync,
            } => {
                let v = self.threads[t].vc.clone();
                self.threads[t].make_mut().fences.push(v);

                self.tick(t);
            }
            Op::BulkCommit => self.threads[t].make_mut().bulk_group += 1,
            // Enabled (split_enabled) once every group older than the newest n completed.
            Op::BulkWait { n, .. } => self.bulk_wait(t, *n),
            Op::TmaStore {
                dims,
                src,
                off,
                tmap,
                coords,
            } => {
                let srca = self.rd(t, src)?.wrapping_add(*off as u64);
                let tma = self.rd(t, tmap)?;
                // A shared::cta source; in a cluster launch it carries the CTA's own rank byte
                // (cuBLAS), as for every shared::cta access (`resolve`).
                let Loc::Shared { off: srca, .. } = self.resolve(t, Space::Shared, srca, 1)? else {
                    unreachable!()
                };
                let srca = srca as u64;
                let (tm, cs) = if *dims == 0 {
                    self.bulk_map(t, tma, &coords[0])?
                } else {
                    let tm = self.tmap_at(t, tma, *dims, pc)?;
                    let mut cs = coords
                        .iter()
                        .map(|c| self.rd(t, c).map(|v| v as u32 as i32 as i64))
                        .collect::<Result<Vec<_>, _>>()?;
                    elem4_coord(&tm, &mut cs)?;
                    (tm, cs)
                };
                let serial = self.threads[t].tma_count;
                self.threads[t].make_mut().tma_count += 1;
                let actor = self.tma_actor_id(t, serial);
                let mut vc = self.threads[t].vc.clone();
                vc_tick(&mut vc, actor);
                let cta = self.threads[t].cta;
                let group = self.threads[t].bulk_group;
                self.push_async(AsyncOp {
                    fifo: None,
                    actor,
                    vc,
                    issuer: t as u32,
                    pc,
                    kind: AsyncKind::TmaStore {
                        cta,
                        src: (srca & 0xff_ffff) as u32,
                        tmap: tm,
                        coords: cs,
                        group,
                    },
                    vis: self.threads[t].vis.clone(),
                });
                self.tick(t);
            }
            Op::Nop => {}
            Op::Fence {
                kind: FenceKind::MbarInit,
            } => {
                let th = self.threads[t].make_mut();
                th.minit_vc = th.vc.clone();
                self.tick(t);
            }
            // ASSUME(S-SCOPE): fences release / acquire at their scope.
            Op::Fence {
                kind: k @ (FenceKind::Membar(sc) | FenceKind::Release(sc)),
            } => {
                if let FenceKind::Membar(_) = k {
                    self.acquire_pending(t, Some(*sc));
                    // ASSUME(E-HW-L1): PTX fence.acq_rel / fence.sc at gpu / sys scope ends
                    // with CCTL.IVALL.
                    if self.l1.is_some() && matches!(sc, Scope::Gpu | Scope::Sys) {
                        self.l1_inval(t, pc);
                    }
                }
                let th = self.threads[t].make_mut();
                th.membar_vc = th.vc.clone();
                th.membar_scope = *sc;
                self.tick(t);
            }
            Op::Fence {
                kind: FenceKind::Acquire,
            } => {
                self.acquire_pending(t, None);
                if self.l1.is_some() {
                    self.l1_inval(t, pc); // ASSUME(E-HW-L1): CCTL.IVALL
                }
            }
            Op::Mapa { d, a, rank } => {
                let a = self.rd(t, a)?;
                let r = self.rd(t, rank)?;
                let cs = self.launch.cluster_size(self.threads[t].cta as u32);
                if r >= cs as u64 {
                    return Err(format!("mapa to CTA rank {r} (cluster size {cs})"));
                }
                // A generic address (mapa.u64 on the shared window) keeps its high word.
                if a >> 32 != 0 && a >> 32 != SHARED_GENERIC_BASE >> 32 {
                    return Err(format!("mapa of {a:#x}: not a shared address"));
                }
                self.wr(
                    t,
                    *d,
                    (a & !0xff00_0000) | (r << 24),
                    1 + (a >> 32 != 0) as u8,
                );
            }
            // ASSUME(E-PDL): the trigger orders no memory; it only lets the next grid launch.
            Op::GridDepLaunch => {
                self.rec_gate(t, 0);
                let cta = self.threads[t].cta as usize;
                self.triggered[cta / 64] |= 1 << (cta % 64);
            }
            // Enabled (split_enabled) once the previous grid completed; then everything it did
            // happens-before this thread's next step.
            Op::GridDepWait => {
                let gi = self.launch.grid_index(self.threads[t].cta as u32);
                if let Some(p) = self.launch.prev_grid(gi) {
                    self.rec_grid_reads(p);
                    let done = self.grid_done_vc_cached(p);
                    vc_join(&mut self.threads[t].make_mut().vc, &done);
                }
                self.tick(t);
            }
            Op::Sel { d, a, b, p } => {
                let first = self.prd(t, *p);
                let (src, other) = if first { (a, b) } else { (b, a) };
                let v = self.rd(t, src)?;
                // PTX `selp.b64` / `.u64` selects a 64-bit virtual register.
                let v = if d.file == File::V {
                    v
                } else {
                    v & 0xffff_ffff
                };
                self.wr(t, *d, v, 1);
                self.num_sel2(t, *d, src, Some((other, first)), *p, pc);
            }
            Op::Prmt { d, a, sel, b } => {
                let x = (self.rd(t, a)? & 0xffff_ffff) | ((self.rd(t, b)? & 0xffff_ffff) << 32);
                let s = self.rd(t, sel)?;
                let mut out = 0u64;
                for i in 0..4 {
                    let n = (s >> (4 * i)) & 0xf;
                    let byte = (x >> (8 * (n & 7))) & 0xff;
                    let v = if n & 8 != 0 {
                        if byte & 0x80 != 0 { 0xff } else { 0 }
                    } else {
                        byte
                    };
                    out |= v << (8 * i);
                }
                self.wr(t, *d, out, 1);
            }
            // ASSUME(E-CLUSTER-BAR): see ASSUMPTIONS.md
            Op::ClusterArrive { release } => {
                let cta = self.threads[t].cta;
                let (cid, _) = self.launch.cluster_of(cta as u32);
                self.rec_sync(13, cid, 0, 1, true);
                if self.threads[t].cl_arrived.is_some() {
                    let msg = format!(
                        "{} arrives at barrier.cluster twice without waiting in between ({})",
                        self.actor_name(t as u32),
                        self.k.describe(pc as usize)
                    );
                    self.report("violation", format!("cluster-arrive2:{pc}"), msg);
                    return Ok(Flow::Stay);
                }
                let th = &self.threads[t];
                // A release arrive publishes everything; a relaxed one only what the thread's
                // last fences released (MEMBAR: all prior accesses; fence.mbarrier_init: inits).
                // SASS: ptxas hoists fence.mbarrier_init's FENCE.VIEW.ASYNC *above* the
                // SYNCS.EXCH inits, so in a binary the sync unit itself must order a thread's
                // inits before its later (relaxed) UCGABAR_ARV: there, every earlier init is
                // released (`vis` only feeds mbarrier-init checks).
                let (rel, vis) = if *release {
                    (th.vc.clone(), th.vc.clone())
                } else if self.k.front == Front::Sass {
                    (th.membar_vc.clone(), th.vc.clone())
                } else {
                    let mut v = th.minit_vc.clone();
                    vc_join(&mut v, &th.membar_vc);
                    (th.membar_vc.clone(), v)
                };
                let cb = self.clusters.entry(cid).or_default();
                let g = cb.gen;
                vc_join(&mut cb.rel, &rel);
                vc_join(&mut cb.vis, &vis);
                if let Err(k) = cb.arrived.binary_search(&(t as u32)) {
                    cb.arrived.insert(k, t as u32);
                }
                self.threads[t].make_mut().cl_arrived = Some(g);
                self.rec_rel(sync_key(4, cid as u64, g as u64, 0));
                self.tick(t);
                self.cluster_try_complete(cta);
            }
            Op::ClusterWait => {
                let cta = self.threads[t].cta;
                let (cid, _) = self.launch.cluster_of(cta as u32);
                self.rec_sync(13, cid, 0, 1, true);
                let Some(g) = self.threads[t].cl_arrived else {
                    let msg = format!(
                        "{} waits at barrier.cluster without having arrived ({})",
                        self.actor_name(t as u32),
                        self.k.describe(pc as usize)
                    );
                    self.report("violation", format!("cluster-wait0:{pc}"), msg);
                    return Ok(Flow::Stay);
                };
                let done = self
                    .clusters
                    .get(&cid)
                    .and_then(|cb| cb.done.iter().find(|d| d.0 == g).cloned());
                match done {
                    Some((_, rel, vis)) => {
                        self.rec_acq(sync_key(4, cid as u64, g as u64, 0));
                        self.cluster_acquire(t, cid, &rel, &vis, true);
                    }
                    None => {
                        self.threads[t].make_mut().st = St::ClusterWait;
                        return Ok(Flow::Stay);
                    }
                }
            }
            // LDCU.128: four registers from the constant bank.
            Op::Mov {
                d,
                a: Val::Const(off, 16),
            } => {
                for i in 0..4u16 {
                    let v = self.cbank(t, off + 4 * i as u32, 4)?;
                    let r = Reg {
                        file: d.file,
                        idx: d.idx + i,
                    };
                    self.wr(t, r, v, 1);
                    self.num_mov(t, r, &Val::Imm(v));
                }
            }
            Op::Mov { d, a } => {
                let v = self.rd(t, a)?;
                self.wr(t, *d, v, Self::slots_of(a));
                self.num_mov(t, *d, a);
            }
            Op::Int { op, bits, d, a, b } => {
                let (x, y) = (mask(*bits, self.rd(t, a)?), mask(*bits, self.rd(t, b)?));
                let w = *bits as u32;
                let sx = |v: u64| ((v << (64 - w)) as i64) >> (64 - w);
                let r = match op {
                    IntOp::Add => x.wrapping_add(y),
                    IntOp::Sub => x.wrapping_sub(y),
                    IntOp::Mul => x.wrapping_mul(y),
                    IntOp::And => x & y,
                    IntOp::Or => x | y,
                    IntOp::Xor => x ^ y,
                    IntOp::Shl => {
                        if y >= w as u64 {
                            0
                        } else {
                            x << y
                        }
                    }
                    IntOp::ShrU => {
                        if y >= w as u64 {
                            0
                        } else {
                            x >> y
                        }
                    }
                    IntOp::ShrS => (sx(x) >> y.min(w as u64 - 1)) as u64,
                    IntOp::DivU | IntOp::DivS | IntOp::RemU | IntOp::RemS if y == 0 => {
                        return Err("integer division by zero (result undefined)".into());
                    }
                    IntOp::DivU => x / y,
                    IntOp::RemU => x % y,
                    IntOp::DivS => sx(x).wrapping_div(sx(y)) as u64,
                    IntOp::RemS => sx(x).wrapping_rem(sx(y)) as u64,
                    IntOp::MinU => x.min(y),
                    IntOp::MaxU => x.max(y),
                    IntOp::MinS => sx(x).min(sx(y)) as u64,
                    IntOp::MaxS => sx(x).max(sx(y)) as u64,
                    IntOp::Abs => sx(x).unsigned_abs(),
                    IntOp::MulHiU => ((x as u128 * y as u128) >> w) as u64,
                    IntOp::MulHiS => ((sx(x) as i128 * sx(y) as i128) >> w) as u64,
                    IntOp::Popc => mask(*bits, x).count_ones() as u64,
                    IntOp::Bmsk | IntOp::BmskWrap => {
                        let (s, n) = if *op == IntOp::BmskWrap {
                            (x & 31, y & 31)
                        } else {
                            ((x & 0xffff_ffff).min(32), (y & 0xffff_ffff).min(32))
                        };
                        let m = (1u64 << n) - 1;
                        (m << s) & 0xffff_ffff
                    }
                };
                self.wr(t, *d, mask(*bits, r), (*bits / 32).max(1));
                self.num_clobber(t, *d, &[a, b], pc);
            }
            Op::Bfe {
                d,
                a,
                pos,
                len,
                signed,
            } => {
                // PTX bfe.b32: pos and len are taken mod 256; the field is clamped to the word.
                let x = self.rd(t, a)? as u32;
                let p = (self.rd(t, pos)? & 0xff) as u32;
                let l = (self.rd(t, len)? & 0xff) as u32;
                let r = if l == 0 {
                    0
                } else {
                    let field = if p >= 32 { 0 } else { x >> p };
                    let width = l.min(32u32.saturating_sub(p.min(32)));
                    let m = if width >= 32 {
                        u32::MAX
                    } else {
                        (1u32 << width) - 1
                    };
                    let mut v = field & m;
                    if *signed {
                        let sbit = if p + l >= 32 {
                            x >> 31
                        } else {
                            (x >> (p + l - 1)) & 1
                        };
                        if sbit == 1 && l < 32 {
                            v |= !m;
                        }
                    }
                    v
                };
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[a], pc);
            }
            // tcgen05.st: written through the thread's tcgen05 actor (asynchronous); wait::st
            // makes the stores part of the thread's history.
            Op::TmemSt {
                taddr,
                off,
                v,
                tracked,
                half,
            } => {
                if half.is_some_and(|h| self.threads[t].lane / 16 != h as u32) {
                    return Ok(Flow::Next); // the other half of the warp's lanes
                }
                let a = (self.rd(t, taddr)? as u32).wrapping_add(*off);
                let mut lane0 = (a >> 16) as usize;
                let col0 = (a & 0xffff) as usize;
                let (wq, l, cta) = {
                    let th = &self.threads[t];
                    ((th.tid[0] / 32 % 4) as usize, th.lane as usize, th.cta)
                };
                // ASSUME(S-TMEM): as tcgen05.ld (the lane field counts mod 32 in SASS).
                if self.k.front == Front::Sass {
                    lane0 = 32 * wq + lane0 % 32;
                }
                let lane = lane0 + if half.is_some() { l % 16 } else { l };
                if lane0 != 32 * wq && !(half.is_some() && lane0 == 32 * wq + 16) {
                    let msg = format!(
                        "{} (warp quadrant {wq}) addresses TMEM lanes {lane0}.. at {}: a warp may only access lanes {}..{}",
                        self.actor_name(t as u32),
                        self.k.describe(pc as usize),
                        32 * wq,
                        32 * wq + 31
                    );
                    self.report("violation", format!("tmem-quadrant:{pc}"), msg);
                }
                if lane >= TMEM_LANES || col0 + v.len() > TMEM_COLS {
                    return Err(format!(
                        "tcgen05.st outside TMEM (lane {lane}, columns {col0}+{})",
                        v.len()
                    ));
                }
                let vals: Vec<u32> = v
                    .iter()
                    .map(|x| self.rd(t, x).map(|y| y as u32))
                    .collect::<Result<_, _>>()?;
                // SASS STTM is scoreboarded like LDTM: the thread itself is the writer.
                let (actor, vc) = if *tracked {
                    let a = self.tc_actor(t);
                    let th = self.threads[t].make_mut();
                    vc_tick(&mut th.tc_vc, a);
                    (a, th.tc_vc.clone())
                } else {
                    (t as u32, self.threads[t].vc.clone())
                };
                let mut states = 0u8;
                for (j, w) in vals.iter().enumerate() {
                    let (_, st) = self.tmem_access(cta, lane, col0 + j, actor, &vc, pc, Some(*w));
                    states |= if st == 2 { 2 } else { 0 };
                    self.num_tmem_st(t, &v[j], cta, lane, col0 + j);
                }
                self.tmem_state_report(states, "tcgen05.st", pc);
            }
            // ASSUME(S-TCGEN05): wait::st completes the stores for this thread's later tcgen05
            // operations (same actor); other threads see them only through
            // fence::before_thread_sync + a sync + fence::after_thread_sync (PTX ISA 9.7.18.6.4.4).
            Op::TmemWaitSt => {}
            // The mbarrier object ends: later plain accesses to its bytes are ordinary memory.
            // ASSUME(S-TMAP): see ASSUMPTIONS.md
            Op::TmapFence { addr } => {
                let a = self.addr(t, addr)?;
                let loc = self.resolve(t, Space::Global, a, 128)?;
                let vc = self.threads[t].vc.clone();
                let bytes = self.access(t as u32, &vc, pc, loc, 128, None);
                let clock = vc_get(&self.threads[t].vc, t as u32);
                let th = self.threads[t].make_mut();
                th.tmap_fence.retain(|(x, _, _)| *x != a);
                th.tmap_fence.push((a, bytes, clock));
                self.tick(t);
            }
            Op::MbarInval { addr } => {
                let a = self.addr(t, addr)?;
                let Loc::Shared { cta, off } = self.resolve(t, Space::Shared, a, 8)? else {
                    unreachable!()
                };
                self.mbar_materialize(cta, off);
                let vc = self.threads[t].vc.clone();
                let vis = self.threads[t].vis.clone();
                self.mbar_atomic(t as u32, &vc, &vis, pc, cta, off)?;
                self.rec(0, cta as u32, off as u64, off as u64 + 8, true);
                self.rec(6, cta as u32, off as u64, off as u64 + 8, true);
                self.rec_phase(cta, off, None, true);
                self.rec_sync(15, cta as u32, off as u64, off as u64 + 1, true);
                self.mbars.remove(&(cta, off));
                self.tick(t);
            }
            Op::StMatrix { .. } | Op::Shfl { .. } | Op::LdMatrix { .. } | Op::Hmma { .. } => {
                return Err("warp-collective op reached the per-thread executor".into());
            }
            Op::MulWide { d, a, b, signed } => {
                let (x, y) = (self.rd(t, a)? as u32, self.rd(t, b)? as u32);
                let r = if *signed {
                    (x as i32 as i64 * y as i32 as i64) as u64
                } else {
                    x as u64 * y as u64
                };
                self.wr(t, *d, r, 2);
                self.num_clobber(t, *d, &[a, b], pc);
            }
            Op::Imad {
                d,
                a,
                b,
                c,
                neg_c,
                not_c,
                wide,
                hi,
                signed,
                cin,
                cout,
            } => {
                let (x, y) = (self.rd(t, a)? & 0xffff_ffff, self.rd(t, b)? & 0xffff_ffff);
                let mut c0 = self.rd(t, c)?;
                if *not_c {
                    c0 = !c0 & 0xffff_ffff;
                }
                if *neg_c {
                    c0 = c0.wrapping_neg();
                }
                let ci = cin.map(|p| self.prd(t, p) as u64).unwrap_or(0);
                if *wide || *hi {
                    let p = if *signed {
                        (x as u32 as i32 as i64 * y as u32 as i32 as i64) as u64
                    } else {
                        x * y
                    };
                    let s = p as u128 + c0 as u128 + ci as u128;
                    if let Some(co) = cout {
                        self.wp(t, *co, s >> 64 != 0);
                    }
                    if *hi {
                        self.wr(t, *d, (s as u64) >> 32, 1);
                    } else {
                        self.wr(t, *d, s as u64, 2);
                    }
                } else {
                    let r = x.wrapping_mul(y).wrapping_add(c0).wrapping_add(ci);
                    self.wr(t, *d, r & 0xffff_ffff, 1);
                }
                // `IMAD.MOV.U32 d, RZ, RZ, c` (ptxas's move idiom): the value moves.
                let mov = !(*wide || *hi || *neg_c || *not_c) && cin.is_none();
                if mov && (*a == Val::Imm(0) || *b == Val::Imm(0)) {
                    self.num_mov(t, *d, c);
                } else {
                    // `IMAD.IADD.U32 d, s, 0x1, k`: s + k (3xTF32 low-part rounding).
                    let r = self.rd(t, &Val::Reg(*d, 1))? as u32;
                    let one = |m: &Self, y: &Val| m.tf32_const(t, y) == Some(1);
                    let tf = mov
                        && ((one(self, b) && self.num_tf32_lor(t, *d, &[a, c], r, pc))
                            || (one(self, a) && self.num_tf32_lor(t, *d, &[b, c], r, pc)));
                    if !tf {
                        self.num_clobber(t, *d, &[a, b, c], pc);
                    }
                }
            }
            Op::Iadd3 {
                d,
                a,
                b,
                c,
                neg,
                not,
                wide,
                cout,
                cin,
            } => {
                let m = if *wide { u64::MAX } else { 0xffff_ffff };
                // A negated input adds its complement plus one (two's complement), which is
                // what the carry-out sees: `IADD3 R, P0, PT, RZ, -R12, RZ` carries iff R12 == 0.
                let mut s: u128 = 0;
                for ((v, n), c) in [a, b, c].iter().zip(neg).zip(not) {
                    let mut x = self.rd(t, v)? & m;
                    if *c {
                        x = !x & m;
                    }
                    s += if *n { (!x & m) as u128 + 1 } else { x as u128 };
                }
                for p in cin {
                    s += self.prd(t, *p) as u128;
                }
                let carry = s >> if *wide { 64 } else { 32 };
                if carry as usize > cout.len() && !cout.is_empty() {
                    return Err(format!(
                        "IADD3 carry of {carry} with {} carry-out predicate(s) not modeled",
                        cout.len()
                    ));
                }
                for (k, co) in cout.iter().enumerate() {
                    self.wp(t, *co, carry > k as u128);
                }
                self.wr(t, *d, (s as u64) & m, if *wide { 2 } else { 1 });
                self.num_clobber(t, *d, &[a, b, c], pc);
            }
            Op::Pack64 { d, lo, hi } => {
                let v = (self.rd(t, lo)? & 0xffff_ffff) | (self.rd(t, hi)? << 32);
                self.wr(t, *d, v, 2);
                self.num_clobber(t, *d, &[lo, hi], pc);
            }
            Op::Unpack64 { lo, hi, a } => {
                let v = self.rd(t, a)?;
                if let Some(r) = lo {
                    self.wr(t, *r, v & 0xffff_ffff, 1);
                    self.num_clobber(t, *r, &[a], pc);
                }
                if let Some(r) = hi {
                    self.wr(t, *r, v >> 32, 1);
                    self.num_clobber(t, *r, &[a], pc);
                }
            }
            Op::LeaHi {
                d,
                a,
                b,
                c,
                shift,
                sx32,
                cin,
                neg_b,
                cout,
            } => {
                let lo = self.rd(t, a)? & 0xffff_ffff;
                let hi = if *sx32 {
                    if lo >> 31 == 1 { 0xffff_ffff } else { 0 }
                } else {
                    self.rd(t, c)?
                };
                let wide = (hi << 32) | lo;
                let hi = ((wide << shift) >> 32) & 0xffff_ffff;
                let ci = cin.map(|p| self.prd(t, p) as u64).unwrap_or(0);
                let bv = self.rd(t, b)? & 0xffff_ffff;
                let bv = if *neg_b { (!bv & 0xffff_ffff) + 1 } else { bv };
                let sum = hi.wrapping_add(bv).wrapping_add(ci);
                let r = sum & 0xffff_ffff;
                if let Some(p) = cout {
                    self.wp(t, *p, sum >> 32 & 1 == 1);
                }
                self.wr(t, *d, r, 1);
                self.num_clobber(t, *d, &[a, b, c], pc);
            }
            Op::I2F {
                d,
                a,
                signed,
                rnd,
                bits,
                hi,
            } => {
                let x = self.rd(t, a)? >> if *hi { 16 } else { 0 };
                let x = match (*bits, *signed) {
                    (64, _) => x,
                    (16, true) => x as u16 as i16 as i64 as u64,
                    (16, false) => x & 0xffff,
                    (8, true) => x as u8 as i8 as i64 as u64,
                    (8, false) => x & 0xff,
                    (_, true) => x as u32 as i32 as i64 as u64,
                    (_, false) => x & 0xffff_ffff,
                };
                self.wr(t, *d, fp::int_to_f32(x, *signed, *rnd).to_bits() as u64, 1);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::F2I { d, a, signed, bits } => {
                // Rust's float->int `as` truncates, saturates, and maps NaN to 0, as F2I does.
                let x = f32::from_bits(self.rd(t, a)? as u32);
                // 64-bit results: NaN gives 2^63 (fixtures/s_cvt), saturation as Rust's.
                let r = match (*bits, *signed) {
                    (64, _) if x.is_nan() => 1 << 63,
                    (64, true) => x as i64 as u64,
                    (64, false) => x as u64,
                    (_, true) => x as i32 as u32 as u64,
                    (_, false) => x as u32 as u64,
                };
                self.wr(t, *d, r, *bits / 32);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::Ffma {
                d,
                a,
                b,
                c,
                neg,
                abs,
                rnd,
                ftz,
                sat,
            } => {
                let mut v = [
                    self.rd(t, a)? as u32,
                    self.rd(t, b)? as u32,
                    match c {
                        Some(c) => self.rd(t, c)? as u32,
                        None => 0,
                    },
                ];
                for i in 0..3 {
                    v[i] = fmod(v[i], neg[i], abs[i]);
                }
                let [x, y, z] = v;
                let r = fp::f32_fma(x, y, c.as_ref().map(|_| z), *rnd, *ftz);
                let r = if *sat { fp::f32_sat(r) } else { r };
                self.wr(t, *d, r as u64, 1);
                let plain = *rnd == Rnd::Rn && !*ftz && !*sat && !abs.iter().any(|&m| m);
                match (c, plain) {
                    (_, true) => self.num_ffma(t, *d, (a, b, c.as_ref()), *neg, [x, y, z], r, pc),
                    (Some(c), false) => self.num_clobber(t, *d, &[a, b, c], pc),
                    (None, false) => self.num_clobber(t, *d, &[a, b], pc),
                }
            }
            Op::Fmnmx {
                d,
                a,
                b,
                c,
                neg,
                abs,
                min,
                ftz,
                nan,
            } => {
                let x = fmod(self.rd(t, a)? as u32, neg[0], abs[0]);
                let y = fmod(self.rd(t, b)? as u32, neg[1], abs[1]);
                let lo = self.prd(t, *min);
                let mut r = fp::f32_minmax(x, y, lo, *nan, *ftz);
                if let Some(c) = c {
                    let z = fmod(self.rd(t, c)? as u32, neg[2], abs[2]);
                    r = fp::f32_minmax(r, z, lo, *nan, *ftz);
                }
                self.wr(t, *d, r as u64, 1);
                let plain = !*ftz && !*nan && !neg.iter().chain(abs).any(|&m| m);
                // The min / max choice must not depend on input data.
                let lo_const = min
                    .reg
                    .is_none_or(|p| !self.num.as_ref().is_some_and(|n| n.taint.reg_tainted(t, p)));
                let srcs: Vec<&Val> = match c {
                    Some(c) => vec![a, b, c],
                    None => vec![a, b],
                };
                self.num_fmnmx(t, *d, &srcs, (plain && lo_const).then_some(lo), pc);
            }
            Op::Fsetp {
                d,
                cmp,
                a,
                b,
                neg,
                abs,
                c,
                bop,
                ftz,
            } => {
                let x = fmod(self.rd(t, a)? as u32, neg[0], abs[0]);
                let y = fmod(self.rd(t, b)? as u32, neg[1], abs[1]);
                let r = fp::f32_cmp(x, y, *cmp, *ftz);
                let q = self.prd(t, *c);
                let v = match bop {
                    BoolOp::And => r && q,
                    BoolOp::Or => r || q,
                    BoolOp::Xor => r != q,
                };
                self.wp(t, *d, v);
                let plain = c.reg.is_none()
                    && !c.neg
                    && *bop == BoolOp::And
                    && !*ftz
                    && !neg.iter().chain(abs).any(|&m| m);
                self.num_fsetp(t, *d, *cmp, a, b, plain);
                self.num_fin_setp(t, *d, *cmp, (a, b), (*neg, *abs), (*c, *ftz), pc);
            }
            // ASSUME(L-ROSETTA): fixtures/s_routing_forms = GB300.
            Op::Hsetp2Bf16 {
                d,
                cmp,
                a,
                b,
                hi,
                c,
                bop,
            } => {
                let lane = |v: u32, h: bool| (if h { v >> 16 } else { v & 0xffff }) << 16;
                let x = lane(self.rd(t, a)? as u32, hi[0]);
                let y = lane(self.rd(t, b)? as u32, hi[1]);
                let r = fp::f32_cmp(x, y, *cmp, false);
                let q = self.prd(t, *c);
                let v = match bop {
                    BoolOp::And => r && q,
                    BoolOp::Or => r || q,
                    BoolOp::Xor => r != q,
                };
                self.wp(t, *d, v);
                self.num_fsetp(t, *d, *cmp, a, b, false);
            }
            Op::Fset {
                d,
                cmp,
                a,
                b,
                neg,
                abs,
                c,
                bop,
                ftz,
            } => {
                let x = fmod(self.rd(t, a)? as u32, neg[0], abs[0]);
                let y = fmod(self.rd(t, b)? as u32, neg[1], abs[1]);
                let r = fp::f32_cmp(x, y, *cmp, *ftz);
                let q = self.prd(t, *c);
                let v = match bop {
                    BoolOp::And => r && q,
                    BoolOp::Or => r || q,
                    BoolOp::Xor => r != q,
                };
                self.wr(t, *d, if v { 0x3f80_0000 } else { 0 }, 1);
                self.num_clobber(t, *d, &[a, b], pc);
            }
            // ASSUME(N-FALU): f64 as the f32 forms (exact, one rounding; fixtures/s_f64).
            Op::Dfma {
                d,
                a,
                b,
                c,
                neg,
                abs,
                rnd,
            } => {
                // Operand modifiers do not touch a NaN (it propagates as it is: s_f64).
                let m = |v: u64, n: bool, a: bool| {
                    if v & !(1 << 63) > 0x7ff0_0000_0000_0000 {
                        return v;
                    }
                    let v = if a { v & !(1 << 63) } else { v };
                    if n { v ^ 1 << 63 } else { v }
                };
                let x = m(self.rd(t, a)?, neg[0], abs[0]);
                let y = m(self.rd(t, b)?, neg[1], abs[1]);
                let z = match c {
                    Some(c) => Some(m(self.rd(t, c)?, neg[2], abs[2])),
                    None => None,
                };
                self.wr(t, *d, fp::f64_fma(x, y, z, *rnd), 2);
                self.num_clobber(t, *d, &[a, b], pc);
            }
            Op::DsetpMinMax { d, q, a, b, max } => {
                let (x, y) = (self.rd(t, a)?, self.rd(t, b)?);
                let nan = |u: u64| u & !(1 << 63) > 0x7ff0_0000_0000_0000;
                let (fx, fy) = (f64::from_bits(x), f64::from_bits(y));
                let below = fx < fy || fx == fy && x >> 63 == 1 && y >> 63 == 0;
                let above = fx > fy || fx == fy && x >> 63 == 0 && y >> 63 == 1;
                let sel = !nan(x) && (nan(y) || if *max { above } else { below });
                self.wp(t, *d, sel);
                if let Some(q) = q {
                    self.wp(t, *q, nan(x) && nan(y));
                }
            }
            Op::DivSqrt64 { d, a, b, rnd } => {
                let x = self.rd(t, a)?;
                let y = match b {
                    Some(b) => Some(self.rd(t, b)?),
                    None => None,
                };
                self.wr(t, *d, fp::f64_div_sqrt(x, y, *rnd), 2);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::Dsetp {
                d,
                cmp,
                a,
                b,
                neg,
                abs,
                c,
                bop,
            } => {
                let m = |v: u64, n: bool, a: bool| {
                    let v = if a { v & !(1 << 63) } else { v };
                    if n { v ^ 1 << 63 } else { v }
                };
                let x = m(self.rd(t, a)?, neg[0], abs[0]);
                let y = m(self.rd(t, b)?, neg[1], abs[1]);
                let r = fp::f64_cmp(x, y, *cmp);
                let q = self.prd(t, *c);
                let v = match bop {
                    BoolOp::And => r && q,
                    BoolOp::Or => r || q,
                    BoolOp::Xor => r != q,
                };
                self.wp(t, *d, v);
            }
            // ASSUME(N-FCHK): see ASSUMPTIONS.md
            Op::Fchk { d, a, b } => {
                let (x, y) = (self.rd(t, a)? as u32, self.rd(t, b)? as u32);
                self.wp(t, *d, fp::fchk(x, y));
            }
            Op::Fsel {
                d,
                a,
                b,
                p,
                neg,
                abs,
            } => {
                let (v, k) = if self.prd(t, *p) { (a, 0) } else { (b, 1) };
                let x = fmod(self.rd(t, v)? as u32, neg[k], abs[k]);
                self.wr(t, *d, x as u64, 1);
                if neg[k] || abs[k] {
                    self.num_clobber(t, *d, &[v], pc);
                } else {
                    self.num_sel(t, *d, v, *p, pc);
                }
            }
            Op::R2p { a, mask } => {
                let x = self.rd(t, a)?;
                for i in 0..7u16 {
                    if mask >> i & 1 == 1 {
                        let p = Reg {
                            file: File::P,
                            idx: i,
                        };
                        self.wp(t, p, x >> i & 1 == 1);
                    }
                }
            }
            // ASSUME(N-FALU): IEEE conversions. F2F keeps a NaN's sign and payload (shifted;
            // quiet bit set), as Rust's float casts do (fixtures/s_cvt); f16 -> f32 gives the
            // canonical NaN, f32 -> f16 0x7fff.
            Op::F2F { d, a, to64, rnd } => {
                let x = self.rd(t, a)?;
                if *to64 {
                    let f = f32::from_bits(x as u32);
                    self.wr(t, *d, (f as f64).to_bits(), 2);
                } else {
                    let f = f64::from_bits(x);
                    self.wr(t, *d, fp::f64_to_f32(f, *rnd).to_bits() as u64, 1);
                }
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::CvtF32F16 { d, a, hi } => {
                let h = (self.rd(t, a)? >> if *hi { 16 } else { 0 }) as u16;
                let v = fp::f16_to_f64(h);
                let r = if v.is_nan() {
                    0x7fff_ffff
                } else {
                    (v as f32).to_bits()
                };
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::F2fF16 { d, a, bf16 } => {
                let x = self.rd(t, a)? as u32;
                // A rounding node in the low half, zero above (read before d, maybe a).
                self.num_f2fp16(t, *d, (&Val::Imm(0), a), (0, x), pc, !*bf16);
                let r = if *bf16 {
                    fp::f32_to_bf16_rn_gpu(x)
                } else {
                    fp::f32_to_f16_rn(x)
                };
                self.wr(t, *d, r as u64, 1);
            }
            Op::F2fpF16 { d, a, b } => {
                let (x, y) = (self.rd(t, a)? as u32, self.rd(t, b)? as u32);
                let r = (fp::f32_to_f16_rn(x) as u64) << 16 | fp::f32_to_f16_rn(y) as u64;
                // Rounding nodes for both halves (read before d, which may be a or b).
                self.num_f2fp16(t, *d, (a, b), (x, y), pc, true);
                self.wr(t, *d, r, 1);
            }
            Op::F2fpBf16 { d, a, b } => {
                let (x, y) = (self.rd(t, a)? as u32, self.rd(t, b)? as u32);
                let r = (fp::f32_to_bf16_rn_gpu(x) as u64) << 16 | fp::f32_to_bf16_rn_gpu(y) as u64;
                // Rounding nodes for both halves (read before d, which may be a or b).
                self.num_f2fp_bf16(t, *d, (a, b), (x, y), pc);
                self.wr(t, *d, r, 1);
            }
            // ASSUME(N-CVT-LP): see ASSUMPTIONS.md
            Op::CvtPack {
                d,
                a,
                b,
                c,
                half,
                to,
                rnd,
                sat,
                relu,
            } => {
                let (x, y, z) = (
                    self.rd(t, a)? as u32,
                    self.rd(t, b)? as u32,
                    self.rd(t, c)? as u32,
                );
                let r = crate::lowp::pack(x, y, z, *half, *to, *rnd, *sat, *relu);
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[a, b, c], pc);
            }
            Op::CvtRsX4 {
                d,
                src,
                rbits,
                to,
                relu,
            } => {
                let mut v = [0u32; 4];
                for (i, s) in src.iter().enumerate() {
                    v[i] = self.rd(t, s)? as u32;
                }
                let r = crate::lowp::pack_rs_x4(v, self.rd(t, rbits)? as u32, *to, *relu);
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[&src[0], &src[1], &src[2], &src[3], rbits], pc);
            }
            Op::CvtUnpack {
                d,
                a,
                hi,
                from,
                to,
                relu,
            } => {
                let x = (self.rd(t, a)? >> if *hi { 16 } else { 0 }) as u32;
                let r = crate::lowp::unpack(x, *from, *to, *relu);
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::Brev { d, a } => {
                let x = self.rd(t, a)? as u32;
                self.wr(t, *d, x.reverse_bits() as u64, 1);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::AddMinMax {
                d,
                a,
                b,
                c,
                signed,
                max,
                neg_b,
            } => {
                let bv = self.rd(t, b)? as u32;
                let bv = if *neg_b { bv.wrapping_neg() } else { bv };
                let s = (self.rd(t, a)? as u32).wrapping_add(bv);
                let z = self.rd(t, c)? as u32;
                let r = match (*signed, *max) {
                    (false, false) => s.min(z),
                    (false, true) => s.max(z),
                    (true, false) => (s as i32).min(z as i32) as u32,
                    (true, true) => (s as i32).max(z as i32) as u32,
                };
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[a, b, c], pc);
            }
            Op::IntMnmx3 {
                d,
                a,
                b,
                c,
                signed,
                max,
            } => {
                let v = [self.rd(t, a)?, self.rd(t, b)?, self.rd(t, c)?].map(|x| x as u32);
                let r = match (*signed, *max) {
                    (false, false) => v.into_iter().min(),
                    (false, true) => v.into_iter().max(),
                    (true, false) => v.into_iter().map(|x| x as i32).min().map(|x| x as u32),
                    (true, true) => v.into_iter().map(|x| x as i32).max().map(|x| x as u32),
                };
                self.wr(t, *d, r.unwrap() as u64, 1);
                self.num_clobber(t, *d, &[a, b, c], pc);
            }
            Op::P2r { d, a, mask } => {
                let mut pr = 0u64;
                for i in 0..7u16 {
                    let p = Pred {
                        reg: Some(Reg {
                            file: File::P,
                            idx: i,
                        }),
                        neg: false,
                    };
                    pr |= (self.prd(t, p) as u64) << i;
                }
                let m = *mask as u64;
                let r = (self.rd(t, a)? & !m & 0xffff_ffff) | (pr & m);
                self.wr(t, *d, r, 1);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::LdConst {
                d,
                bank,
                base,
                off,
                bytes,
                data,
            } => {
                let o = (self.rd(t, base)? as u32).wrapping_add(*off);
                let v = if *bank == 0 {
                    self.cbank(t, o, (*bytes).min(8))?
                } else {
                    let mut v = 0u64;
                    for i in 0..(*bytes).min(8) as usize {
                        let b = data.get(o as usize + i).ok_or_else(|| {
                            format!("read past constant bank {bank} (c[{bank:#x}][{o:#x}])")
                        })?;
                        v |= (*b as u64) << (8 * i);
                    }
                    v
                };
                self.wr(t, *d, v, (*bytes / 4).max(1));
                self.num_clobber(t, *d, &[base], pc);
            }
            // ASSUME(N-MUFU): see ASSUMPTIONS.md
            Op::Mufu {
                d,
                a,
                func,
                fmt,
                hi,
                neg,
                abs,
                ptx,
            } => {
                // PTX f64 forms: the unit reads the high word and writes it, low word 0.
                let f64 = *fmt == MufuFmt::F64;
                let raw = (self.rd(t, a)? >> if f64 { 32 } else { 0 }) as u32;
                let mut x = raw;
                if *abs {
                    x &= 0x7fff_ffff;
                }
                if *neg {
                    x ^= 0x8000_0000;
                }
                let r = match ptx {
                    None => crate::mufu::mufu(*func, *fmt, *hi, x),
                    Some(ftz) => crate::mufu::ptx_approx(*func, *fmt, *ftz, x),
                }
                .ok_or_else(|| format!("{func:?} ({fmt:?}) is not modeled (N-MUFU)"))?;
                if f64 {
                    self.wr(t, *d, (r as u64) << 32, 2);
                } else {
                    self.wr(t, *d, r as u64, 1);
                }
                self.num_mufu(
                    t,
                    *d,
                    a,
                    (raw, r),
                    (*func, *fmt, ptx.is_some()),
                    (*neg, *abs),
                    pc,
                );
            }
            Op::Lea {
                d,
                a,
                b,
                shift,
                cout,
                neg_a,
            } => {
                let sa = (self.rd(t, a)? << shift) & 0xffff_ffff;
                // A negated operand adds its complement plus one (as IADD3), which the carry sees.
                let sa = if *neg_a { (!sa & 0xffff_ffff) + 1 } else { sa };
                let r = sa + (self.rd(t, b)? & 0xffff_ffff);
                if let Some(co) = cout {
                    self.wp(t, *co, r >> 32 != 0);
                }
                self.wr(t, *d, r & 0xffff_ffff, 1);
                // `LEA d, s, k, 0x0`: s + k (3xTF32 low-part rounding).
                let tf = *shift == 0
                    && !*neg_a
                    && cout.is_none()
                    && self.num_tf32_lor(t, *d, &[a, b], r as u32, pc);
                if !tf {
                    self.num_clobber(t, *d, &[a, b], pc);
                }
            }
            Op::Shf {
                d,
                lo,
                sh,
                hi,
                right,
                signed,
                high,
            } => {
                let v = (self.rd(t, hi)? << 32) | (self.rd(t, lo)? & 0xffff_ffff);
                let s = (self.rd(t, sh)? & 63) as u32;
                let r = if *right {
                    if *signed {
                        ((v as i64) >> s) as u64
                    } else {
                        v >> s
                    }
                } else {
                    v << s
                };
                self.wr(t, *d, if *high { r >> 32 } else { r & 0xffff_ffff }, 1);
                self.num_clobber(t, *d, &[lo, sh, hi], pc);
            }
            Op::Lop3 {
                d,
                a,
                b,
                c,
                lut,
                p,
                pq,
            } => {
                let (x, y, z) = (self.rd(t, a)?, self.rd(t, b)?, self.rd(t, c)?);
                let mut r = 0u64;
                for i in 0..8 {
                    if (lut >> i) & 1 == 1 {
                        let fa = if i & 4 != 0 { x } else { !x };
                        let fb = if i & 2 != 0 { y } else { !y };
                        let fc = if i & 1 != 0 { z } else { !z };
                        r |= fa & fb & fc;
                    }
                }
                // Pq is read before d / P are written (P may be Pq).
                let q = pq.map(|(q, and)| (self.prd(t, q), and));
                self.wr(t, *d, r & 0xffff_ffff, 1);
                if let Some(p) = p {
                    let nz = r & 0xffff_ffff != 0;
                    self.wp(
                        t,
                        *p,
                        match q {
                            None => nz,
                            Some((q, true)) => nz && q,
                            Some((q, false)) => nz || q,
                        },
                    );
                }
                // `LOP3.LUT d, x, 0xffffe000, RZ, 0xc0`: the TF32 high part of x.
                if !self.num_tf32_hi(t, *d, [a, b, c], *lut, pc) {
                    self.num_clobber(t, *d, &[a, b, c], pc);
                }
            }
            Op::Plop3 { d, a, b, c, lut } => {
                let i = ((self.prd(t, *a) as u8) << 2)
                    | ((self.prd(t, *b) as u8) << 1)
                    | self.prd(t, *c) as u8;
                self.wp(t, *d, (lut >> i) & 1 == 1);
            }
            Op::Setp {
                d,
                cmp,
                bits,
                a,
                b,
                and,
                or,
                xor,
                ex,
            } => {
                let (x, y) = (mask(*bits, self.rd(t, a)?), mask(*bits, self.rd(t, b)?));
                let w = *bits as u32;
                let (sx, sy) = (
                    ((x << (64 - w)) as i64) >> (64 - w),
                    ((y << (64 - w)) as i64) >> (64 - w),
                );
                let r = match cmp {
                    Cmp::Eq => x == y,
                    Cmp::Ne => x != y,
                    Cmp::LtU => x < y,
                    Cmp::LeU => x <= y,
                    Cmp::GtU => x > y,
                    Cmp::GeU => x >= y,
                    Cmp::LtS => sx < sy,
                    Cmp::LeS => sx <= sy,
                    Cmp::GtS => sx > sy,
                    Cmp::GeS => sx >= sy,
                };
                // ISETP.EX: equal high words defer to the low words' result.
                let r = match ex {
                    Some(p) if x == y => self.prd(t, *p),
                    _ => r,
                };
                let v = if *xor {
                    r ^ self.prd(t, *and)
                } else if *or {
                    r || self.prd(t, *and)
                } else {
                    r && self.prd(t, *and)
                };
                self.wp(t, *d, v);
            }
            Op::Fadd {
                d,
                a,
                b,
                rnd,
                neg,
                abs,
                ftz,
                sat,
            } => {
                let x = fmod(self.rd(t, a)? as u32, neg[0], abs[0]);
                let y = fmod(self.rd(t, b)? as u32, neg[1], abs[1]);
                // a*1 + b with one rounding is the correctly rounded sum.
                let r = fp::f32_fma(x, 0x3f80_0000, Some(y), *rnd, *ftz);
                let r = if *sat { fp::f32_sat(r) } else { r };
                self.wr(t, *d, r as u64, 1);
                // The numerics DAG models plain sums (Thm 1 nodes); other forms are values
                // without a tracked derivation.
                let seg = self.num.as_ref().is_some_and(|n| n.segments);
                let plain = !(abs.iter().any(|&m| m) || *ftz || *sat || *rnd != Rnd::Rn);
                if self.num_tf32_lo(t, *d, (a, b), *neg, plain, r, pc) {
                    // x - hi(x): the TF32 low part of x (3xTF32 split).
                } else if seg
                    && !(abs.iter().any(|&m| m) || *ftz || *sat || matches!(rnd, Rnd::Rm | Rnd::Rp))
                {
                    self.num_fadd_seg(t, *d, (a, b), *neg, (x, y, r), *rnd == Rnd::Rz, pc);
                } else if neg.iter().chain(abs).any(|&m| m)
                    || *ftz
                    || *sat
                    || matches!(rnd, Rnd::Rm | Rnd::Rp)
                {
                    self.num_clobber(t, *d, &[a, b], pc);
                } else {
                    self.num_fadd(t, *d, a, b, (x, y, r), *rnd == Rnd::Rz, pc);
                }
            }
            Op::Ffma2 { d, a, b, c, neg } => {
                let lanes = |m: &mut Self, (v, l): &(Val, F2Lanes)| -> Result<[u32; 2], String> {
                    let x = m.rd(t, v)?;
                    let (lo, hi) = (x as u32, (x >> 32) as u32);
                    Ok(match l {
                        F2Lanes::Bcast => [lo, lo],
                        F2Lanes::Pair => [lo, hi],
                        F2Lanes::Swap => [hi, lo],
                    })
                };
                let sign = |v: [u32; 2], n: bool| if n { v.map(|w| w ^ 0x8000_0000) } else { v };
                let (x, y, z) = (
                    sign(lanes(self, a)?, neg[0]),
                    sign(lanes(self, b)?, neg[1]),
                    sign(lanes(self, c)?, neg[2]),
                );
                // One rounding to nearest per lane; NaN results are canonical.
                let r = [0, 1].map(|i| fp::f32_fma(x[i], y[i], Some(z[i]), Rnd::Rn, false));
                self.wr(t, *d, r[0] as u64 | (r[1] as u64) << 32, 2);
                self.num_ffma2(t, *d, [a, b, c], *neg, [x, y, z], pc);
            }
            Op::Hfma2 { d, a, b, c, neg_a } => {
                let (x, y, z) = (
                    self.rd(t, a)? as u32,
                    self.rd(t, b)? as u32,
                    self.rd(t, c)? as u32,
                );
                let x = if *neg_a { x ^ 0x8000_8000 } else { x };
                let lo = fp::f16_fma_rn(x as u16, y as u16, z as u16) as u32;
                let hi =
                    fp::f16_fma_rn((x >> 16) as u16, (y >> 16) as u16, (z >> 16) as u16) as u32;
                self.wr(t, *d, ((hi << 16) | lo) as u64, 1);
                self.num_clobber(t, *d, &[a, b, c], pc);
            }
            Op::CvtF32Bf16 { d, a } => {
                let v = (self.rd(t, a)? & 0xffff) << 16;
                self.wr(t, *d, v, 1);
                self.num_clobber(t, *d, &[a], pc);
            }
            Op::Cvta {
                d,
                a,
                space,
                to_generic,
            } => {
                let v = self.rd(t, a)?;
                let r = match (space, to_generic) {
                    (Space::Param, true) => v + PARAM_BASE,
                    (Space::Param, false) => v - PARAM_BASE,
                    (Space::Global, _) => v,
                    // ASSUME(S-SWIN): the shared window.
                    (Space::Shared | Space::SharedCluster, true) => {
                        SHARED_GENERIC_BASE | (v & 0xffff_ffff)
                    }
                    (Space::Shared | Space::SharedCluster, false) => {
                        if v >> 32 != SHARED_GENERIC_BASE >> 32 {
                            return Err(format!("cvta.to.shared of {v:#x}: not in the window"));
                        }
                        v & 0xffff_ffff
                    }
                    _ => return Err("cvta for this space not modeled yet".into()),
                };
                self.wr(t, *d, r, 2);
            }
            Op::Ld {
                d,
                n,
                space,
                bytes,
                addr,
                sync,
                sext,
            } => {
                let a = self.addr(t, addr)?;
                let loc = self.resolve(t, *space, a, *bytes as usize)?;
                let vc = self.threads[t].vc.clone();
                let sc = sync.map_or(0, |m| strong_code(m.scope));
                // Sign extension of a 1- / 2-byte element to 32 bits.
                let ext = |v: u64, es: u32| -> u64 {
                    if *sext && es < 4 {
                        let sh = 64 - 8 * es;
                        (((v << sh) as i64 >> sh) as u64) & 0xffff_ffff
                    } else {
                        v
                    }
                };
                // ASSUME(S-INIT): a load of stale shared memory yields undefined registers
                // (reported when used, not here: cuBLAS loads the try_cancel response's ctaid
                // before testing its success bit).
                let defer = matches!(d.file, File::R | File::V);
                if defer {
                    self.stale = Some(None);
                }
                let mut data = self.access_x(t as u32, &vc, pc, loc, *bytes as usize, None, sc);
                // Some(Some(_)) now: the `wr`s below leave the registers unwritten.
                if self.stale == Some(None) {
                    self.stale = None;
                }
                // E-READ-DONE: the read completes when an instruction uses its registers.
                if self.launch.read_done && matches!(d.file, File::R | File::V) {
                    let k = if d.file == File::R {
                        (*bytes).div_ceil(4).max(1)
                    } else {
                        (*n).max(1)
                    };
                    let c = vc_get(&vc, t as u32);
                    self.threads[t]
                        .make_mut()
                        .ld_pend
                        .push((d.file as u8, d.idx, k, c));
                }
                // ASSUME(E-WELLSYNC) C-RING: a forced counter read (wellsync/ring.rs) returns
                // the forced write's bytes and synchronizes with that write.
                let forced = if self.ring.is_some() {
                    self.ring_value(t, pc, loc)
                } else {
                    None
                };
                let was_forced = forced.is_some();
                match (sync, forced) {
                    (Some(m), Some((b, cell))) => {
                        data = b;
                        self.sync_observe_cell(t, cell, *m);
                    }
                    // E-POLL-ANY: a value-insensitive poll acquires nothing (it may only add
                    // races).
                    (Some(_), None) if self.vip_read == 1 && vip_where(loc).is_some() => {}
                    (Some(m), None) => self.sync_observe(t, loc, *m),
                    (None, Some((b, _))) => data = b,
                    (None, None) => {}
                }
                // ASSUME(E-HW-L1): weak global loads go through the SM's L1; strong loads bypass
                // it; a PTX acquire at gpu / sys scope invalidates it (SASS: CCTL.IVALL).
                if self.l1.is_some() {
                    match sync {
                        None if !was_forced && matches!(loc, Loc::Global { .. }) => {
                            self.l1_load(t, pc, loc, &mut data)
                        }
                        Some(m) if m.acquires() && matches!(m.scope, Scope::Gpu | Scope::Sys) => {
                            self.l1_inval(t, pc)
                        }
                        _ => {}
                    }
                }
                if *n > 1 && *bytes / *n < 4 {
                    // PTX ld.vN of 8/16-bit elements: one element per register.
                    let es = (*bytes / *n) as usize;
                    for i in 0..*n as usize {
                        let mut v = 0u64;
                        for (j, b) in data[i * es..(i + 1) * es].iter().enumerate() {
                            v |= (*b as u64) << (8 * j);
                        }
                        let v = ext(v, es as u32);
                        let r = Reg {
                            file: d.file,
                            idx: d.idx + i as u16,
                        };
                        self.wr(t, r, v, 1);
                    }
                    self.num_ld(t, *d, loc, (*bytes, *n, *sext), &data);
                    return Ok(Flow::Next);
                }
                if *bytes > 8 || *n > 1 {
                    // ld.b128 (two 64-bit PTX slots) / LDS.128 (four R registers) / ld.vN.b32.
                    let words: Vec<u64> = data
                        .chunks(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64)
                        .collect();
                    match d.file {
                        File::V if *n == 1 => {
                            for h in 0..(words.len() / 2) {
                                let r = Reg {
                                    file: File::V,
                                    idx: d.idx + h as u16,
                                };
                                self.wr(t, r, words[2 * h] | (words[2 * h + 1] << 32), 1);
                            }
                        }
                        _ => {
                            // A partly stale vector load (S-INIT): only the registers whose
                            // bytes nothing wrote become undefined (cuBLAS gemmSN's split-K
                            // reduction reads a row of 9-word slots whose 9th word is padding).
                            let stale = self.stale.take();
                            let unwritten = |m: &Self, i: usize| match loc {
                                Loc::Shared { cta, off } => (0..4).any(|b| {
                                    let a = off as usize + 4 * i + b;
                                    a >= SMEM_RESERVED && m.smem[cta as usize].sh.get(a).w.is_none()
                                }),
                                _ => true,
                            };
                            for (i, w) in words.iter().enumerate() {
                                let r = Reg {
                                    file: d.file,
                                    idx: d.idx + i as u16,
                                };
                                if stale.is_some() {
                                    let u = unwritten(self, i);
                                    self.vmask = Some(if u { !0 } else { 0 });
                                }
                                self.wr(t, r, *w, 1);
                            }
                            if stale.is_some() {
                                self.vmask = None;
                            }
                            self.stale = stale;
                        }
                    }
                    self.num_ld(t, *d, loc, (*bytes, *n, *sext), &data);
                    return Ok(Flow::Next);
                }
                let mut v = 0u64;
                for (i, b) in data.iter().enumerate() {
                    v |= (*b as u64) << (8 * i);
                }
                let v = ext(v, *bytes as u32);
                self.wr(t, *d, v, (*n).max((*bytes / 4).max(1)));
                self.num_ld(t, *d, loc, (*bytes, *n, *sext), &data);
            }
            Op::St {
                space,
                bytes,
                addr,
                v,
                sync,
            } => {
                let a = self.addr(t, addr)?;
                let loc = self.resolve(t, *space, a, *bytes as usize)?;
                let data: Vec<u8> = if let Val::Reg(r @ Reg { file: File::V, .. }, n @ 2..) = v {
                    // PTX `st.vN`: one virtual register per element.
                    let w = (*bytes / *n) as usize;
                    let mut d = Vec::with_capacity(*bytes as usize);
                    for i in 0..*n as u16 {
                        let x = self.rd(
                            t,
                            &Val::Reg(
                                Reg {
                                    file: r.file,
                                    idx: r.idx + i,
                                },
                                1,
                            ),
                        )?;
                        d.extend_from_slice(&x.to_le_bytes()[..w]);
                    }
                    d
                } else if *bytes > 8 && matches!(v, Val::Imm(0)) {
                    // `STS.128 [R8], RZ`: every word is RZ (cuBLAS gemmSN_NN_kernel).
                    vec![0; *bytes as usize]
                } else if *bytes > 8 {
                    // STS.128 / st.v4: one 32-bit register per word.
                    let Val::Reg(r, _) = v else {
                        return Err("store wider than 64 bits of a non-register".into());
                    };
                    let mut d = Vec::with_capacity(*bytes as usize);
                    for i in 0..(*bytes / 4) as u16 {
                        let w = self.rd(
                            t,
                            &Val::Reg(
                                Reg {
                                    file: r.file,
                                    idx: r.idx + i,
                                },
                                1,
                            ),
                        )?;
                        d.extend_from_slice(&(w as u32).to_le_bytes());
                    }
                    d
                } else {
                    let x = self.rd(t, v)?;
                    (0..*bytes).map(|i| (x >> (8 * i)) as u8).collect()
                };
                let vc = self.threads[t].vc.clone();
                let sc = sync.map_or(0, |m| strong_code(m.scope));
                self.access_x(t as u32, &vc, pc, loc, *bytes as usize, Some(&data), sc);
                if self.l1.is_some() {
                    self.l1_store(t, loc, &data); // ASSUME(E-HW-L1)
                }
                match sync {
                    Some(m) => self.sync_publish(t, loc, *m, false),
                    None => self.sync_clear(loc),
                }
                // A release store publishes the thread's current clock: later events of the
                // thread must not look covered by it (FastTrack ticks after a release). Without
                // this, a plain write after `st.release` in the same clock window looked
                // ordered before every reader that acquired the release (fixture e_ring R1:
                // a payload written after its tail was published was reported clean).
                if sync.is_some_and(|m| m.releases()) {
                    self.tick(t);
                }
                self.num_st(t, v, loc, *bytes);
            }
            Op::Dp2a {
                d,
                a,
                b,
                c,
                hi,
                a_signed,
                b_signed,
            } => {
                let (x, y, z) = (
                    self.rd(t, a)? as u32,
                    self.rd(t, b)? as u32,
                    self.rd(t, c)? as u32,
                );
                let mut r = z;
                for i in 0..2u32 {
                    let h = x >> (16 * i) & 0xffff;
                    let h = if *a_signed {
                        h as u16 as i16 as i32 as u32
                    } else {
                        h
                    };
                    let bb = y >> (8 * (i + 2 * *hi as u32)) & 0xff;
                    let bb = if *b_signed {
                        bb as u8 as i8 as i32 as u32
                    } else {
                        bb
                    };
                    r = r.wrapping_add(h.wrapping_mul(bb));
                }
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[a, b, c], pc);
            }
            // ASSUME(N-MUFU): MUFU.RCP as the unit model; the FMULs round to nearest.
            // ASSUME(N-FALU): div.rn.f32 is the correctly rounded IEEE quotient.
            Op::DivFull { d, a, b, rn: true } => {
                let (x, y) = (self.rd(t, a)? as u32, self.rd(t, b)? as u32);
                self.wr(t, *d, crate::seg::div_rn(x, y) as u64, 1);
                self.num_clobber(t, *d, &[a, b], pc);
            }
            Op::DivFull { d, a, b, rn: false } => {
                let (x, y) = (self.rd(t, a)? as u32, self.rd(t, b)? as u32);
                let mul = |p: u32, q: u32| fp::f32_fma(p, q, None, ptxv_ir::Rnd::Rn, false);
                let yb = f32::from_bits(y).abs();
                let (x, y) = if yb > 2f32.powi(126) {
                    (mul(x, 0x3e80_0000), mul(y, 0x3e80_0000))
                } else if yb < f32::MIN_POSITIVE {
                    (mul(x, 0x4b80_0000), mul(y, 0x4b80_0000))
                } else {
                    (x, y)
                };
                let r = mul(crate::mufu::rcp(y), x);
                self.wr(t, *d, r as u64, 1);
                self.num_clobber(t, *d, &[a, b], pc);
            }
            // Generic addresses resolve to global memory or the parameter window here; the
            // shared window (S-SWIN) is not modeled for QSPC.
            Op::Qspc { p, addr } => {
                let a = self.addr(t, addr)?;
                if a >> 32 == SHARED_GENERIC_BASE >> 32 {
                    return Err("QSPC of a shared-window address not modeled".into());
                }
                self.resolve(t, Space::Generic, a, 1)?;
                self.wp(t, *p, false);
            }
            Op::Ld256 { hi, lo, addr, sync } => {
                let a = self.addr(t, addr)?;
                let loc = self.resolve(t, Space::Global, a, 32)?;
                let vc = self.threads[t].vc.clone();
                let sc = sync.map_or(0, |m| strong_code(m.scope));
                let mut data = self.access_x(t as u32, &vc, pc, loc, 32, None, sc);
                if let Some(m) = sync {
                    self.sync_observe(t, loc, *m);
                } else if self.l1.is_some() {
                    self.l1_load(t, pc, loc, &mut data); // ASSUME(E-HW-L1)
                }
                for (i, c) in data.chunks(4).enumerate() {
                    let base = if i < 4 { *lo } else { *hi };
                    let r = Reg {
                        file: base.file,
                        idx: base.idx + (i % 4) as u16,
                    };
                    self.wr(t, r, u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64, 1);
                }
                // Numerics: two 16-byte vector loads (each register keeps its word's node).
                for (h, base) in [(0usize, *lo), (1, *hi)] {
                    let l = match loc {
                        Loc::Global { alloc, off } => Loc::Global {
                            alloc,
                            off: off + 16 * h as u64,
                        },
                        other => other,
                    };
                    self.num_ld(t, base, l, (16, 1, false), &data[16 * h..16 * h + 16]);
                }
            }
            // Local memory is private to the thread: no races, no other observers.
            Op::LdLocal { d, bytes, addr } if *bytes < 4 => {
                // LDL.U8 / .U16: zero-extended byte / half of the word holding it.
                let a = self.addr(t, addr)? as u32;
                if !a.is_multiple_of(*bytes) {
                    return Err(format!("unaligned local load at {a:#x}"));
                }
                let (w, sh, m) = (a & !3, 8 * (a & 3), (1u64 << (8 * *bytes)) - 1);
                let loc = &self.threads[t].local;
                let (v, um) = match loc.binary_search_by_key(&w, |e| e.0) {
                    Ok(j) => (loc[j].1 & 0xffff_ffff, loc[j].1 >> 32),
                    Err(_) => (0, 0xffff_ffff),
                };
                let um = (um >> sh) & m;
                self.vmask = (um != 0).then_some(um);
                self.wr(t, *d, (v >> sh) & m, 1);
                self.vmask = None;
                self.num_clobber(t, *d, &[], pc);
            }
            Op::StLocal { bytes, addr, v } if *bytes < 4 => {
                // STL.U8 / .U16: the low byte / half into the word holding it.
                let a = self.addr(t, addr)? as u32;
                if !a.is_multiple_of(*bytes) {
                    return Err(format!("unaligned local store at {a:#x}"));
                }
                let (w, sh, m) = (a & !3, 8 * (a & 3), (1u64 << (8 * *bytes)) - 1);
                let x = self.rd(t, v)? & m;
                let xm = match v {
                    Val::Reg(r, _) => self.reg_umask(t, *r) as u64 & m,
                    _ => 0,
                };
                let loc = &self.threads[t].local;
                let (old, oum) = match loc.binary_search_by_key(&w, |e| e.0) {
                    Ok(j) => (loc[j].1 & 0xffff_ffff, loc[j].1 >> 32),
                    Err(_) => (0, 0xffff_ffff),
                };
                let nv = (old & !(m << sh)) | (x << sh);
                let num = (oum & !(m << sh)) | (xm << sh);
                if self.num.is_some() {
                    self.num_local_st(t, &Val::Imm(nv), w);
                }
                let e = nv | (num << 32);
                let loc = &mut self.threads[t].make_mut().local;
                match loc.binary_search_by_key(&w, |e| e.0) {
                    Ok(j) => loc[j].1 = e,
                    Err(j) => loc.insert(j, (w, e)),
                }
            }
            Op::LdLocal { d, bytes, addr } => {
                let a = self.addr(t, addr)? as u32;
                if !a.is_multiple_of(4) {
                    return Err(format!("unaligned local load at {a:#x}"));
                }
                for i in 0..(*bytes / 4).max(1) {
                    let w = a + 4 * i;
                    let loc = &self.threads[t].local;
                    // ASSUME(S-INIT): never-stored local memory is undefined.
                    let (v, um) = match loc.binary_search_by_key(&w, |e| e.0) {
                        Ok(j) => (loc[j].1 & 0xffff_ffff, (loc[j].1 >> 32) as u32),
                        Err(_) => (0, u32::MAX),
                    };
                    let r = Reg {
                        file: d.file,
                        idx: d.idx + i as u16,
                    };
                    self.vmask = (um != 0).then_some(um as u64);
                    self.wr(t, r, v, 1);
                    self.vmask = None;
                    self.num_local_ld(t, r, w);
                }
            }
            Op::StLocal { bytes, addr, v } => {
                let a = self.addr(t, addr)? as u32;
                if !a.is_multiple_of(4) {
                    return Err(format!("unaligned local store at {a:#x}"));
                }
                for i in 0..(*bytes / 4).max(1) {
                    let (x, um) = match v {
                        Val::Reg(r, _) => {
                            let r = Reg {
                                file: r.file,
                                idx: r.idx + i as u16,
                            };
                            let rv = Val::Reg(r, 1);
                            (self.rd(t, &rv)? & 0xffff_ffff, self.reg_umask(t, r))
                        }
                        _ => ((self.rd(t, v)? >> (32 * i)) & 0xffff_ffff, 0),
                    };
                    let w = a + 4 * i;
                    if self.num.is_some() {
                        let vi = match v {
                            Val::Reg(r, _) => Val::Reg(
                                Reg {
                                    file: r.file,
                                    idx: r.idx + i as u16,
                                },
                                1,
                            ),
                            _ => Val::Imm(x),
                        };
                        self.num_local_st(t, &vi, w);
                    }
                    let e = x | ((um as u64) << 32);
                    let loc = &mut self.threads[t].make_mut().local;
                    match loc.binary_search_by_key(&w, |e| e.0) {
                        Ok(j) => loc[j].1 = e,
                        Err(j) => loc.insert(j, (w, e)),
                    }
                }
            }
            // ASSUME(E-CPASYNC): the global source is read at issue; the shared destination is
            // written when a wait_group retires its group, race-checked with the clock at
            // issue (nothing the thread does after issuing orders the write).
            Op::CpAsync {
                dst,
                src,
                bytes,
                src_ok,
            } => {
                let vc = self.threads[t].vc.clone();
                let mut from = None;
                let data = if !self.prd(t, *src_ok) {
                    vec![0; *bytes as usize]
                } else {
                    let sa = self.addr(t, src)?;
                    let sloc = self.resolve(t, Space::Global, sa, *bytes as usize)?;
                    if let Loc::Global { alloc, off } = sloc {
                        from = Some((alloc, off));
                    }
                    self.access(t as u32, &vc, pc, sloc, *bytes as usize, None)
                };
                let da = self.addr(t, dst)?;
                let Loc::Shared { off, .. } =
                    self.resolve(t, Space::Shared, da, *bytes as usize)?
                else {
                    unreachable!()
                };
                let th = self.threads[t].make_mut();
                let group = th.cp_group;
                th.cp_pending.push(CpPending {
                    group,
                    off,
                    src: from,
                    data,
                    vc,
                    pc,
                });
            }
            // ASSUME(E-CPASYNC-MBAR): every cp.async the thread issued so far (committed or
            // not) lands with this op; a later wait_group does not wait for them again.
            Op::CpAsyncMbarArrive { addr, tx } => {
                let a = self.addr(t, addr)?;
                let Loc::Shared { cta, off } = self.resolve(t, Space::Shared, a, 8)? else {
                    unreachable!()
                };
                let writes = std::mem::take(&mut self.threads[t].make_mut().cp_pending);
                let serial = self.threads[t].tma_count;
                self.threads[t].make_mut().tma_count += 1;
                let actor = self.tma_actor_id(t, serial);
                let mut vc = self.threads[t].vc.clone();
                vc_tick(&mut vc, actor);
                let vis = self.threads[t].vis.clone();
                self.push_async(AsyncOp {
                    fifo: None,
                    actor,
                    vc,
                    issuer: t as u32,
                    pc,
                    kind: AsyncKind::CpArrive {
                        cta,
                        mbar: off,
                        writes,
                        tx: *tx,
                    },
                    vis,
                });
                self.tick(t);
            }
            Op::CpAsyncCommit => self.threads[t].make_mut().cp_group += 1,
            Op::CpAsyncWait { n } => {
                let th = &self.threads[t];
                let upto = th.cp_group.saturating_sub(*n);
                let cta = th.cta;
                let (done, rest): (Vec<_>, Vec<_>) =
                    th.cp_pending.iter().cloned().partition(|c| c.group < upto);
                self.threads[t].make_mut().cp_pending = rest;
                for c in done {
                    let loc = Loc::Shared { cta, off: c.off };
                    self.access(t as u32, &c.vc, c.pc, loc, c.data.len(), Some(&c.data));
                    self.sync_clear(loc);
                    self.num_cp_async(c.src, cta, c.off, &c.data);
                }
            }
            Op::Exit => {
                self.rec_gate(t, 1);
                // ASSUME(E-EXIT-FENCE): kernel completion acquires the thread's relaxed
                // observations (case `exit_fence`); the exit releases its history anyway.
                if let Some(sc) = self.launch.exit_fence {
                    self.acquire_pending(t, Some(sc));
                }
                let th = self.threads[t].make_mut();
                th.st = St::Exited;
                let cta = th.cta;
                // An all-threads barrier may now be complete without this thread.
                let ids: Vec<u32> = self
                    .bars
                    .keys()
                    .filter(|k| k.0 == cta)
                    .map(|k| k.1)
                    .collect();
                for id in ids {
                    self.bar_try_release(cta, id);
                }
                self.cluster_try_complete(cta);
                return Ok(Flow::Stay);
            }
            // ASSUME(E-BAR-EXIT, E-BAR-ARRIVE): see ASSUMPTIONS.md
            Op::BarSync { id, count, arrive } => {
                let arrive = *arrive;
                let id = self.rd(t, id)? as u32;
                let count = match count {
                    Some(c) => Some(self.rd(t, c)? as u32),
                    None => None,
                };
                let cta = self.threads[t].cta;
                if count.is_none() {
                    if let Some(s) = self.sfoot.as_ref() {
                        s.add(SPACE_CTA_ALL, cta as u32, 0, 1, true);
                    }
                }
                let bar = self.bars.entry((cta, id)).or_insert(Bar {
                    arrived: vec![],
                    arrivers: vec![],
                    rel: VC::new(),
                    count,
                });
                if bar.count != count {
                    let f = |c: Option<u32>| c.map(|c| c.to_string()).unwrap_or("all".into());
                    let msg = format!(
                        "bar.sync {id} used with thread counts {} and {}",
                        f(bar.count),
                        f(count)
                    );
                    self.report("violation", format!("barcount:{id}"), msg);
                }
                // A thread arriving twice in one generation (bar.arrive, then bar.arrive or
                // bar.sync again before the barrier completed): the PTX ISA gives no meaning
                // to it. Reported; modeled as the hardware counter would, counting twice.
                let bar = self.bars.get(&(cta, id)).unwrap();
                if bar.arrivers.binary_search(&(t as u32)).is_ok() {
                    let msg = format!(
                        "{} arrives at named barrier {id} again ({}) before the generation it \
                         arrived in with bar.arrive completed",
                        self.actor_name(t as u32),
                        self.k.describe(pc as usize)
                    );
                    self.report("violation", format!("bararrive2:{id}:{pc}"), msg);
                }
                // DPOR / sync reduction footprint (space 14, index id << 32 | generation):
                // an arrival reads its generation's key. bar.arrive writes every generation
                // of the barrier: an arrival that does not wait can land in the next one
                // (which generation each lands in decides who is released and with what).
                // ASSUME(E-BAR-OVER): a bar.sync arrival at a counted barrier that is not
                // ordered after the previous generation's release could have landed in that
                // generation: it writes that generation's key.
                let g = self.bar_gen.get(&(cta, id)).copied().unwrap_or(0);
                let key = ((id as u64) << 32) | g as u64;
                // ASSUME(E-BAR-ARRIVE): unless it provably lands in this generation
                // (`Machine::arv_current`, PTXV_NO_ARVGEN).
                if arrive && !(self.fp.is_some() && self.arv_current(t, cta, id, count)) {
                    let b = (id as u64) << 32;
                    self.rec(14, cta as u32, b, b + (1 << 32), true);
                } else if g > 0
                    && self.fp.is_some()
                    && self.bar_prev.get(&(cta, id)).is_some_and(|p| {
                        let mut v = self.threads[t].vc.clone();
                        vc_join(&mut v, p);
                        v != self.threads[t].vc
                    })
                {
                    self.rec(14, cta as u32, key - 1, key, true);
                }
                self.rec(14, cta as u32, key, key + 1, false);
                self.rec_rel(sync_key(1, cta as u64, id as u64, g as u64));
                if arrive {
                    let mut vc = self.threads[t].vc.clone();
                    if self.launch.read_done {
                        let rr = self.read_release(t);
                        if let Some(rr) = crate::machine::rr_new(&vc, &rr) {
                            vc_join(&mut vc, &rr);
                        }
                    }
                    let bar = self.bars.get_mut(&(cta, id)).unwrap();
                    let k = match bar.arrivers.binary_search(&(t as u32)) {
                        Ok(k) | Err(k) => k,
                    };
                    bar.arrivers.insert(k, t as u32);
                    vc_join(&mut bar.rel, &vc);
                    // Later accesses of the arriving thread are not released.
                    self.tick(t);
                    self.bar_try_release(cta, id);
                } else {
                    let bar = self.bars.get_mut(&(cta, id)).unwrap();
                    if let Err(k) = bar.arrived.binary_search(&(t as u32)) {
                        bar.arrived.insert(k, t as u32);
                    }
                    self.threads[t].make_mut().st = St::Bar;
                    self.bar_try_release(cta, id);
                    return Ok(Flow::Stay);
                }
            }
            Op::MbarInit {
                addr,
                count,
                encoded,
            } => {
                let a = self.addr(t, addr)?;
                let Loc::Shared { cta, off } = self.resolve(t, Space::Shared, a, 8)? else {
                    unreachable!()
                };
                if off % 8 != 0 {
                    return Err(format!("mbarrier at {off:#x} is not 8-byte aligned"));
                }
                let raw = self.rd(t, count)?;
                let n = if *encoded {
                    // SASS writes the initial 64-bit state word; the low word holds
                    // (2^20 - count) << 1 (observed from ptxas output, see sass crate docs).
                    0x10_0000 - (((raw & 0xffff_ffff) >> 1) & 0xf_ffff) as u32
                } else {
                    raw as u32
                };
                // Bit 63 of an encoded word: the initial phase parity (GB300,
                // probes/mbar_phase1; cuBLAS starts its "empty" barriers in phase 1, so the
                // first parity-0 wait passes). Phase 0 then counts as completed at init.
                let phase = if *encoded { (raw >> 63) as u32 } else { 0 };
                let vc = self.threads[t].vc.clone();
                let me = Epoch {
                    actor: t as u32,
                    clock: vc_get(&vc, t as u32),
                    pc,
                    strong: 0,
                };
                if let Some(old) = self.mbars.get(&(cta, off)).cloned() {
                    for e in std::iter::once(old.init).chain(old.atomics.iter().copied()) {
                        if e.actor != t as u32 && !e.hb(&vc) {
                            self.mbar_race(e, me, off);
                        }
                    }
                }
                self.rec(0, cta as u32, off as u64, off as u64 + 8, true);
                self.rec(6, cta as u32, off as u64, off as u64 + 8, true);
                self.rec_phase(cta, off, None, true);
                self.rec_sync(15, cta as u32, off as u64, off as u64 + 1, true);
                self.rec(
                    7,
                    cta as u32,
                    (off as u64) << 20,
                    ((off as u64) + 1) << 20,
                    true,
                );
                self.rec_rel(sync_key(2, cta as u64, off as u64, u64::MAX));
                if phase == 1 {
                    self.rec_rel(sync_key(2, cta as u64, off as u64, 0));
                }
                let done = if phase == 1 {
                    [vc.clone(), VC::new()]
                } else {
                    [VC::new(), vc.clone()]
                };
                self.mbars.insert(
                    (cta, off),
                    Mbar {
                        expected: n,
                        pending: n,
                        tx: 0,
                        phase,
                        cur: VC::new(),
                        done,
                        init: me,
                        atomics: vec![],
                    },
                );
                self.tick(t);
            }
            Op::MbarArrive {
                addr,
                tx,
                remote,
                count,
                complete,
            } => {
                let arrivals = match count {
                    Some(c) => self.rd(t, c)?,
                    None => 1,
                };
                if arrivals > 1 {
                    return Err(
                        "mbarrier.arrive with a count other than 1 is not modeled yet".into(),
                    );
                }
                let a = self.addr(t, addr)?;
                if *remote && self.dead_target(t, a, pc)? {
                    self.tick(t);
                    return Ok(Flow::Next);
                }
                let space = if *remote {
                    Space::SharedCluster
                } else {
                    Space::Shared
                };
                let Loc::Shared { cta, off } = self.resolve(t, space, a, 8)? else {
                    unreachable!()
                };
                if cta != self.threads[t].cta {
                    self.note_remote(cta, t as u32, pc);
                }
                let txv = match tx {
                    Some(v) => self.rd(t, v)? as i64,
                    None => 0,
                };
                let vc = self.threads[t].vc.clone();
                let vis = self.threads[t].vis.clone();
                self.rec(0, cta as u32, off as u64, off as u64 + 8, false);
                self.mbar_atomic(t as u32, &vc, &vis, pc, cta, off)?;
                let mb = self.mbars.get_mut(&(cta, off)).unwrap();
                mb.tx += if *complete { -txv } else { txv };
                if arrivals == 1 {
                    if mb.pending == 0 {
                        return Err(format!(
                            "mbarrier at {off:#x}: more arrivals than the expected count {}",
                            mb.expected
                        ));
                    }
                    mb.pending -= 1;
                }
                vc_join(&mut mb.cur, &vc);
                self.mbar_maybe_complete(cta, off);
                self.tick(t);
            }
            // ASSUME(E-SPIN): a peek run as a failed test (`explore::peekskip_set`).
            Op::MbarTryWait {
                d,
                addr,
                parity,
                bit31,
            } if self.bar_cfg.peekskip.contains_key(&pc) && self.peekskip_here(t, pc) => {
                let a = self.addr(t, addr)?;
                let Loc::Shared { cta, off } = self.resolve(t, Space::Shared, a, 8)? else {
                    unreachable!()
                };
                let p = self.rd(t, parity)?;
                let p = if *bit31 { (p >> 31) & 1 } else { p & 1 } as u32;
                self.rec(0, cta as u32, off as u64, off as u64 + 8, false);
                // Only the completion that would be an ABA in the window (see
                // `peekskip_set`): of the current phase if the test would pass now, else of
                // the next one.
                if let Some(ph) = self.mbars.get(&(cta, off)).map(|mb| mb.phase) {
                    let pass = ph % 2 != p;
                    self.rec_phase(cta, off, Some(if pass { ph } else { ph + 1 }), false);
                }
                self.wp(t, *d, false);
                let th = self.threads[t].make_mut();
                th.peekwin.retain(|w| w.3 != pc);
                th.peekwin.push((cta, off, p, pc));
            }
            Op::MbarTryWait { d, .. } => match self.try_wait_result(t) {
                Ok(Some(done)) => {
                    let a = self.addr(
                        t,
                        match &inst.op {
                            Op::MbarTryWait { addr, .. } => addr,
                            _ => unreachable!(),
                        },
                    )?;
                    let Loc::Shared { cta, off } = self.resolve(t, Space::Shared, a, 8)? else {
                        unreachable!()
                    };
                    if let Some(&(s, w2)) = self.stubs.get(&pc) {
                        self.check_stub_parity(t, pc, s, w2, Loc::Shared { cta, off });
                    }
                    let vc = self.threads[t].vc.clone();
                    self.rec(0, cta as u32, off as u64, off as u64 + 8, false);
                    self.rec(6, cta as u32, off as u64, off as u64 + 8, false);
                    let ph = self.mbars.get(&(cta, off)).map(|mb| mb.phase);
                    self.rec_phase(cta, off, ph, false);
                    // Acquire the phase whose completion this poll observed (u64::MAX = init).
                    if let Some(mb) = self.mbars.get(&(cta, off)) {
                        let ph = mb.phase as i64;
                        let p = self.poll_parity(t);
                        let want = if ph >= 1 && (ph - 1) % 2 == p {
                            ph - 1
                        } else {
                            ph - 2
                        };
                        let k = if want < 0 { u64::MAX } else { want as u64 };
                        self.rec_acq(sync_key(2, cta as u64, off as u64, k));
                    }
                    let vis = self.threads[t].vis.clone();
                    self.mbar_atomic(t as u32, &vc, &vis, pc, cta, off)?;
                    let th = self.threads[t].make_mut();
                    vc_join(&mut th.vc, &done);
                    self.wp(t, *d, true);
                    self.tick(t);
                }
                // (A peek the explorer runs as a blocking wait, `explore::peekwait_set`,
                // blocks like one.)
                Ok(None) if self.peeks.contains(&pc) && !self.bar_cfg.peekwait.contains(&pc) => {
                    // ASSUME(E-SPIN): a failed peek reads the phase state and returns false.
                    let a = self.addr(
                        t,
                        match &inst.op {
                            Op::MbarTryWait { addr, .. } => addr,
                            _ => unreachable!(),
                        },
                    )?;
                    if let Loc::Shared { cta, off } = self.resolve(t, Space::Shared, a, 8)? {
                        self.rec(0, cta as u32, off as u64, off as u64 + 8, false);
                        self.rec(6, cta as u32, off as u64, off as u64 + 8, false);
                        let ph = self.mbars.get(&(cta, off)).map(|mb| mb.phase);
                        self.rec_phase(cta, off, ph, false);
                    }
                    self.wp(t, *d, false);
                }
                Ok(None) => return Ok(Flow::Stay),
                Err(e) => {
                    self.report(
                        "violation",
                        format!("mbar-uninit:{pc}"),
                        format!(
                            "{e} ({} at {})",
                            self.actor_name(t as u32),
                            self.k.describe(pc as usize)
                        ),
                    );
                    return Ok(Flow::Stay);
                }
            },
            Op::TmaLoad {
                dims,
                dst,
                mbar,
                tmap,
                coords,
                mask,
                pair,
                off,
                mbar_off,
            } => {
                let dsta = self.rd(t, dst)?.wrapping_add(*off as u64);
                let mba = self.rd(t, mbar)?.wrapping_add(*mbar_off as u64);
                let tma = self.rd(t, tmap)?;
                let me = self.threads[t].cta;
                // Destination CTAs: the rank in the address, or every rank in the ctaMask
                // (same offsets in each).
                let mut mbar_cta = None;
                let targets: Vec<u16> = match mask {
                    None => {
                        let d = self.rank_cta(me, dsta)?;
                        let mc = self.rank_cta(me, mba)?;
                        if mc != d && !*pair {
                            return Err("TMA destination and mbarrier are in different CTAs (only .cta_group::2 may signal the peer's mbarrier)".into());
                        }
                        if mc != me {
                            if self.dead_target(t, mba, pc)? {
                                self.tick(t);
                                return Ok(Flow::Next);
                            }
                            self.note_remote(mc, t as u32, pc);
                        }
                        mbar_cta = Some(mc);
                        vec![d]
                    }
                    Some(mv) => {
                        // SASS `UTMALDG.MULTICAST`: the ctaMask is bits 16..31 of the UR
                        // (ptxas writes `UMOV UR, mask << 16`).
                        let mv = self.rd(t, mv)?;
                        let mv = if self.k.front == Front::Sass {
                            mv >> 16
                        } else {
                            mv
                        } & 0xffff;
                        let mut v = Vec::new();
                        for r in 0..16u64 {
                            if mv >> r & 1 == 1 {
                                v.push(self.rank_cta(me, r << 24)?);
                            }
                        }
                        v
                    }
                };
                let (tm, cs) = if *dims == 0 {
                    self.bulk_map(t, tma, &coords[0])?
                } else {
                    let tm = self.tmap_at(t, tma, *dims, pc)?;
                    let mut cs = coords
                        .iter()
                        .map(|c| self.rd(t, c).map(|v| v as u32 as i32 as i64))
                        .collect::<Result<Vec<_>, _>>()?;
                    elem4_coord(&tm, &mut cs)?;
                    (tm, cs)
                };
                // One async op per destination: multicast copies land independently.
                for cta in targets {
                    if cta != me {
                        let rank = self.launch.cluster_of(cta as u32).1 as u64;
                        if self.dead_target(t, rank << 24, pc)? {
                            continue;
                        }
                        self.note_remote(cta, t as u32, pc);
                    }
                    let serial = self.threads[t].tma_count;
                    self.threads[t].make_mut().tma_count += 1;
                    let actor = self.tma_actor_id(t, serial);
                    let mut vc = self.threads[t].vc.clone();
                    vc_tick(&mut vc, actor);
                    let vis = self.threads[t].vis.clone();
                    self.push_async(AsyncOp {
                        fifo: None,
                        actor,
                        vc,
                        issuer: t as u32,
                        pc,
                        kind: AsyncKind::Tma {
                            cta,
                            dst: (dsta & 0xff_ffff) as u32,
                            mbar_cta: mbar_cta.unwrap_or(cta),
                            mbar: (mba & 0xff_ffff) as u32,
                            tmap: tm.clone(),
                            coords: cs.clone(),
                        },
                        vis,
                    });
                }
                self.tick(t);
            }
            Op::VoteAll { .. } | Op::Elect { .. } | Op::Bra { .. } => {
                return Err("warp-collective op reached the per-thread executor".into());
            }
        }
        Ok(Flow::Next)
    }

    /// The tensor map a TMA op of thread `t` names: a launch parameter the case declares, or
    /// one built in global memory (rung 3a: CUTLASS patches templates per group), decoded
    /// from its bytes (`crate::tmap`). A device-built map must have been fenced into the
    /// tensormap proxy by `t` since it last changed (`Thread::tmap_fence`).
    // ASSUME(S-TMAP): see ASSUMPTIONS.md
    fn tmap_at(&mut self, t: usize, tma: u64, dims: u8, pc: u32) -> Result<Arc<TmapDesc>, String> {
        let gi = self.launch.grid_index(self.threads[t].cta as u32);
        let m = match self
            .launch
            .tensormaps
            .iter()
            .find(|m| m.grid == gi && PARAM_BASE + m.param_off as u64 == tma)
        {
            Some(m) => Arc::new(m.clone()),
            None => {
                let loc = self.resolve(t, Space::Global, tma, 128).map_err(|e| {
                    format!("TMA tensor-map operand {tma:#x} is neither a declared tensor map nor in global memory ({e})")
                })?;
                let vc = self.threads[t].vc.clone();
                let bytes = self.access(t as u32, &vc, pc, loc, 128, None);
                let mut fenced = self.threads[t]
                    .tmap_fence
                    .iter()
                    .find(|(a, _, _)| *a == tma)
                    .map(|(_, b, _)| *b == bytes);
                // A fence of these bytes by another thread of the CTA that happens-before
                // this use also counts (CUTLASS 4.x grouped GEMM: tensormap-updater warps
                // update and fence.proxy.tensormap-acquire each map, then hand it to the load
                // warps through an mbarrier pipeline).
                if fenced != Some(true) {
                    let cta = self.threads[t].cta;
                    let vc = &self.threads[t].vc;
                    let other = (0..self.threads.len()).any(|u| {
                        u != t
                            && self.threads[u].cta == cta
                            && self.threads[u].tmap_fence.iter().any(|(a, b, c)| {
                                *a == tma && *b == bytes && *c <= vc_get(vc, u as u32)
                            })
                    });
                    if other {
                        fenced = Some(true);
                    }
                }
                if fenced != Some(true) {
                    let why = if fenced.is_none() {
                        "without a tensormap proxy fence (fence.proxy.tensormap / UTMACCTL.IV) by this thread"
                    } else {
                        "after it changed since this thread's last tensormap proxy fence (the TMA unit may use a stale copy)"
                    };
                    let msg = format!(
                        "{} uses the tensor map at {tma:#x} {why}",
                        self.actor_name(t as u32)
                    );
                    self.report("violation", format!("tmap-fence:{pc}"), msg);
                }
                let d = crate::tmap::decode(&bytes)
                    .map_err(|e| format!("tensor map at {tma:#x}: {e}"))?;
                if d.estrides.iter().any(|&e| e != 1)
                    || d.swizzle_atom != crate::tmap::SwizzleAtom::B16
                {
                    return Err(format!(
                        "tensor map at {tma:#x}: element strides / swizzle atom not modeled"
                    ));
                }
                let i = (d.base.wrapping_sub(GLOBAL_BASE) >> 32) as usize;
                let a = self
                    .launch
                    .allocs
                    .get(i)
                    .filter(|a| d.base >= a.base)
                    .ok_or_else(|| {
                        format!(
                            "tensor map at {tma:#x}: base {:#x} is not in any buffer",
                            d.base
                        )
                    })?;
                Arc::new(TmapDesc {
                    grid: gi,
                    param_off: u32::MAX,
                    buffer: a.name.clone(),
                    base: d.base,
                    esize: d.esize,
                    dims: d.dims,
                    strides: d.strides,
                    swizzle: d.swizzle,
                    boxd: d.boxd,
                    elem4: false,
                })
            }
        };
        if m.dims.len() != dims as usize {
            return Err("TMA dimensionality does not match the tensor map".into());
        }
        Ok(m)
    }

    /// Allocation holding a tensor map's base, and the base's offset in it.
    /// A non-tensor bulk copy (`UBLKCP`, lowered as a TMA op with `dims = 0`): `units` 16-byte
    /// units at the 64-bit global address in `gaddr`, as a 1-D tensor map of 16-byte elements
    /// whose box is the whole copy, at coordinate 0. ASSUME(S-TMA)
    fn bulk_map(
        &mut self,
        t: usize,
        units: u64,
        gaddr: &Val,
    ) -> Result<(Arc<TmapDesc>, Vec<i64>), String> {
        let g = self.rd(t, gaddr)?;
        // SASS UBLKCP counts 16-byte units; PTX cp.async.bulk gives bytes (a multiple of 16).
        let n = if self.k.front == ptxv_ir::Front::Ptx {
            let b = units & 0xffff_ffff;
            if !b.is_multiple_of(16) {
                return Err(format!("cp.async.bulk of {b} bytes: not a multiple of 16"));
            }
            b / 16
        } else {
            units & 0xffff
        };
        if n == 0 {
            return Err("bulk copy of 0 bytes".into());
        }
        // The global range must lie in one allocation.
        self.resolve(t, Space::Global, g, (n * 16) as usize)?;
        Ok((
            Arc::new(TmapDesc {
                grid: 0,
                param_off: 0,
                buffer: String::new(),
                base: g,
                esize: 16,
                dims: vec![n],
                strides: vec![],
                swizzle: 0,
                boxd: vec![n as u32],
                elem4: false,
            }),
            vec![0],
        ))
    }

    fn tmap_alloc(&self, m: &TmapDesc) -> (usize, u64) {
        let i = self
            .launch
            .allocs
            .iter()
            .rposition(|a| a.base <= m.base)
            .expect("tensor map base");
        (i, m.base - self.launch.allocs[i].base)
    }

    /// Release barrier `id` of `cta` if enough threads arrived: its explicit count, or every
    /// thread of the CTA that has not exited (counts arrivals, not exits: GB300 hangs on
    /// `bar.sync 0, 64` with a warp exited, but passes `bar.sync 0`).
    /// Parity requested by the try_wait at thread t's pc.
    /// ASSUME(E-SPIN): a try_wait whose retry stub `s..w2` recomputes the parity blocks
    /// exactly only if the stub would poll the same phase: run the stub on a copy of this
    /// thread and compare.
    fn check_stub_parity(&mut self, t: usize, pc: u32, s: u32, w2: u32, loc: Loc) {
        let mut m = self.clone();
        m.fp = None;
        m.num = None;
        m.threads[t].make_mut().pc = s;
        for p in s..w2 {
            if m.exec(&[t], p).is_err() {
                break;
            }
        }
        m.threads[t].make_mut().pc = w2;
        let same = m.poll_parity(t) == self.poll_parity(t)
            && matches!(&m.k.code[w2 as usize].op, Op::MbarTryWait { addr, .. }
                if m.addr(t, addr).ok().and_then(|a| m.resolve(t, Space::Shared, a, 8).ok())
                    == Some(loc));
        // A retry that loops back to the poll's own prefix (w2 == pc, spin::retry_shape)
        // must also recompute every register to the value it has: the model skips it.
        let (a, b) = (&m.threads[t], &self.threads[t]);
        let (wa, wb) = (&m.warps[a.warp as usize], &self.warps[b.warp as usize]);
        let same = same
            && (w2 != pc
                || a.regs == b.regs
                    && a.preds == b.preds
                    && a.written == b.written
                    && wa.ur == wb.ur
                    && wa.up == wb.up
                    && wa.ur_written == wb.ur_written);
        if !same {
            let msg = format!(
                "E-SPIN does not hold for the try_wait at {}: its retry path at {} polls a different phase (or recomputes a register to a different value), so a failed poll is not the same as blocking",
                self.k.describe(pc as usize),
                self.k.describe(w2 as usize)
            );
            self.report("unmodeled", format!("spin-parity:{pc}"), msg);
        }
    }

    fn poll_parity(&self, t: usize) -> i64 {
        match &self.k.code[self.threads[t].pc as usize].op {
            Op::MbarTryWait { parity, bit31, .. } => {
                let p = self.rd(t, parity).unwrap_or(0);
                (if *bit31 { (p >> 31) & 1 } else { p & 1 }) as i64
            }
            _ => 0,
        }
    }

    /// CTA named by the rank byte of a shared::cluster address, relative to `cta`'s cluster.
    pub(crate) fn rank_cta(&self, cta: u16, a: u64) -> Result<u16, String> {
        let r = (a >> 24) as u32;
        self.launch
            .cta_of_rank(cta as u32, r)
            .map(|c| c as u16)
            .ok_or_else(|| {
                format!(
                    "shared::cluster address {a:#x} names CTA rank {r} (cluster size {})",
                    self.launch.cluster_size(cta as u32)
                )
            })
    }

    /// ASSUME(S-CLUSTER-LIFE): a CTA's shared memory exists only while one of its threads
    /// runs. An access after the whole target CTA exited is reported here; an access that
    /// merely *could* come after (no happens-before to any of the target's exits) is
    /// reported at terminal states (`exit_checks`).
    fn check_alive(&self, target: u16, actor: u32) -> Result<(), String> {
        if self
            .threads
            .iter()
            .filter(|u| u.cta == target)
            .all(|u| u.st == St::Exited)
        {
            return Err(format!(
                "{} accesses the shared memory of CTA {target} after all its threads exited",
                self.actor_name(actor)
            ));
        }
        Ok(())
    }

    /// Is the CTA named by shared::cluster address `a` gone (all its threads exited)? Then the
    /// access is reported and dropped: it cannot affect the live CTAs, and stopping the
    /// issuer would only add consequential deadlocks.
    fn dead_target(&mut self, t: usize, a: u64, pc: u32) -> Result<bool, String> {
        let me = self.threads[t].cta;
        let target = self.rank_cta(me, a)?;
        if target == me {
            return Ok(false);
        }
        match self.check_alive(target, t as u32) {
            Ok(()) => Ok(false),
            Err(e) => {
                let msg = format!("{e} at {}", self.k.describe(pc as usize));
                self.report("violation", format!("cluster-life:{pc}"), msg);
                Ok(true)
            }
        }
    }

    fn note_remote(&mut self, target: u16, t: u32, pc: u32) {
        let clock = vc_get(&self.threads[t as usize].vc, t);
        let e = (target, t, clock, pc);
        if let Err(k) = self.remote.binary_search(&e) {
            self.remote.insert(k, e);
        }
    }

    /// Complete the cluster barrier generation of `cta`'s cluster once every live thread of
    /// the cluster arrived (exited threads are not waited for, as for bar.sync).
    fn cluster_try_complete(&mut self, cta: u16) {
        let (cid, _) = self.launch.cluster_of(cta as u32);
        let Some(cb) = self.clusters.get(&cid) else {
            return;
        };
        if cb.arrived.is_empty() {
            return;
        }
        let ids = self.cluster_threads(cta);
        let members: Vec<u32> = ids
            .iter()
            .filter(|&&u| self.threads[u].st != St::Exited)
            .map(|&u| u as u32)
            .collect();
        if !members.iter().all(|u| cb.arrived.binary_search(u).is_ok()) {
            return;
        }
        let cb = self.clusters.get_mut(&cid).unwrap();
        let g = cb.gen;
        let rel = std::mem::take(&mut cb.rel);
        let vis = std::mem::take(&mut cb.vis);
        cb.arrived.clear();
        cb.gen += 1;
        cb.done.push((g, rel.clone(), vis.clone()));
        // Wake this cluster's waiters of generation g (other clusters count generations too).
        let waiters: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&u| {
                let th = &self.threads[u];
                th.st == St::ClusterWait && th.cl_arrived == Some(g)
            })
            .collect();
        if self.wlog.is_on() {
            crate::wellsync::released(self, &waiters); // wellsync.rs
        }
        for &u in &waiters {
            self.cluster_acquire(u, cid, &rel, &vis, false);
            let th = self.threads[u].make_mut();
            th.st = St::Run;
            th.pc += 1;
            th.acq_bar = Some(sync_key(4, cid as u64, g as u64, 0));
        }
        if !waiters.is_empty() {
            self.cluster_forget(cta, cid);
        }
    }

    /// Threads of `cta`'s cluster (threads are laid out CTA by CTA).
    fn cluster_threads(&self, cta: u16) -> Vec<usize> {
        let first = |c: u32| {
            let (mut a, mut b) = (0, self.threads.len());
            while a < b {
                let mid = (a + b) / 2;
                if (self.threads[mid].cta as u32) < c {
                    a = mid + 1;
                } else {
                    b = mid;
                }
            }
            a
        };
        let n = self.launch.cluster_size(cta as u32);
        let mut v = Vec::new();
        for r in 0..n {
            if let Some(c) = self.launch.cta_of_rank(cta as u32, r) {
                v.extend(first(c)..first(c + 1));
            }
        }
        v.sort_unstable();
        v
    }

    /// Forget barrier.cluster generations nobody of this cluster waits for any more.
    fn cluster_forget(&mut self, cta: u16, cid: u32) {
        let pending: Vec<u32> = self
            .cluster_threads(cta)
            .into_iter()
            .filter(|&u| self.threads[u].st != St::Exited)
            .filter_map(|u| self.threads[u].cl_arrived)
            .collect();
        if let Some(cb) = self.clusters.get_mut(&cid) {
            cb.done.retain(|d| pending.contains(&d.0));
        }
    }

    /// Thread u passes barrier.cluster generation g.
    fn cluster_acquire(&mut self, u: usize, cid: u32, rel: &VC, vis: &VC, forget: bool) {
        let th = self.threads[u].make_mut();
        vc_join(&mut th.vc, rel);
        vc_join(&mut th.vis, vis);
        th.cl_arrived = None;
        vc_tick(&mut th.vc, u as u32);
        if forget {
            let cta = self.threads[u].cta;
            self.cluster_forget(cta, cid);
        }
    }

    fn bar_try_release(&mut self, cta: u16, id: u32) {
        let Some(bar) = self.bars.get(&(cta, id)) else {
            return;
        };
        let need = match bar.count {
            Some(c) => c as usize,
            None => self
                .threads
                .iter()
                .filter(|t| t.cta == cta && t.st != St::Exited)
                .count(),
        };
        let n = bar.arrived.len() + bar.arrivers.len();
        if n < need || n == 0 {
            return;
        }
        let Bar {
            arrived,
            rel,
            count,
            ..
        } = self.bars.remove(&(cta, id)).unwrap();
        let g = self.bar_gen.entry((cta, id)).or_insert(0);
        let key = sync_key(1, cta as u64, id as u64, *g as u64);
        *g += 1;
        // Waiters acquire every arrival: their own and the bar.arrive releases.
        let mut j = rel;
        for &a in &arrived {
            vc_join(&mut j, &self.threads[a as usize].vc);
            // ASSUME(E-READ-DONE): an arrival carries its completed reads.
            if self.launch.read_done {
                let rr = self.read_release(a as usize);
                if let Some(rr) = crate::machine::rr_new(&j, &rr) {
                    vc_join(&mut j, &rr);
                }
            }
        }
        if count.is_some() {
            // E-BAR-OVER: completing generation g changes where a later arrival lands (an
            // arrival into g reads its key): the completion writes it when more threads than
            // the count may still arrive (a sleeping DPOR arrival must wake up).
            if self.fp.is_some() && self.bar_oversub(cta, id, count) {
                let k = ((id as u64) << 32) | (*self.bar_gen.get(&(cta, id)).unwrap() - 1) as u64;
                self.rec(14, cta as u32, k, k + 1, true);
            }
            self.bar_prev.insert((cta, id), j.clone());
        }
        if self.wlog.is_on() {
            let ts: Vec<usize> = arrived.iter().map(|&a| a as usize).collect();
            crate::wellsync::released(self, &ts); // wellsync.rs
        }
        for &a in &arrived {
            let th = self.threads[a as usize].make_mut();
            th.vc = j.clone();
            vc_tick(&mut th.vc, a);
            th.st = St::Run;
            th.pc += 1;
            th.acq_bar = Some(key);
            let (w, pc) = (th.warp, th.pc);
            self.rec_pc(w, pc, true);
        }
    }

    fn tick(&mut self, t: usize) {
        let th = self.threads[t].make_mut();
        vc_tick(&mut th.vc, t as u32);
    }

    fn mbar_race(&mut self, a: Epoch, b: Epoch, off: u32) {
        let sig = format!("race:mbar:{}:{}", a.pc, b.pc);
        let msg = format!(
            "race on the mbarrier object at smem {off:#x}: {} at {} and {} at {} are not ordered (mbarrier.init must happen-before every other use)",
            self.actor_name(a.actor),
            self.k.describe(a.pc as usize),
            self.actor_name(b.actor),
            self.k.describe(b.pc as usize)
        );
        self.report("race", sig, msg);
    }

    /// Atomic mbarrier access: must be ordered after the init (a plain write).
    fn mbar_atomic(
        &mut self,
        actor: u32,
        vc: &VC,
        vis: &VC,
        pc: u32,
        cta: u16,
        off: u32,
    ) -> Result<(), String> {
        self.mbar_materialize(cta, off);
        let mb = self
            .mbars
            .get(&(cta, off))
            .ok_or_else(|| format!("mbarrier at smem {off:#x} used before mbarrier.init"))?;
        let init = mb.init;
        let me = Epoch {
            actor,
            clock: vc_get(vc, actor),
            pc,
            strong: STRONG_SYS,
        };
        if init.actor != actor && !init.hb(vc) && !init.hb(vis) {
            self.mbar_race(init, me, off);
        }
        let mb = self.mbars.get_mut(&(cta, off)).unwrap();
        epoch_insert(&mut mb.atomics, me);
        if self.wlog.is_on() {
            crate::wellsync::mbar_op(self, actor, vc, pc, cta, off); // wellsync.rs
        }
        Ok(())
    }

    /// Footprint space 12 (the stateful explorer's sync reduction): the completion of phase
    /// `ph` of the mbarrier at `off` writes key (off, ph); a poll while the current phase is
    /// `ph` reads it (only that completion can change its result or what it acquires). `None`:
    /// every phase (init / inval).
    fn rec_phase(&mut self, cta: u16, off: u32, ph: Option<u32>, write: bool) {
        let base = (off as u64) << 20;
        match ph {
            Some(p) => {
                let k = base | (p as u64 & 0xf_ffff);
                self.rec(12, cta as u32, k, k + 1, write)
            }
            None => self.rec(12, cta as u32, base, base + (1 << 20), write),
        }
    }

    /// Called after every arrival / complete_tx. DPOR footprint (dpor.rs): an arrival into
    /// phase k reads key (off, k); the one that completes phase k also writes key (off, k+1)
    /// and the phase state that polls read. So arrivals into the same phase commute, arrivals
    /// straddling a phase boundary conflict, and polls conflict only with completions.
    fn mbar_maybe_complete(&mut self, cta: u16, off: u32) {
        if self.wlog.is_on() {
            crate::wellsync::mbar_update(self, cta, off); // wellsync.rs
        }
        let (phase, completes) = {
            let mb = &self.mbars[&(cta, off)];
            (mb.phase, mb.pending == 0 && mb.tx == 0)
        };
        let key = ((off as u64) << 20) | (phase as u64 & 0xf_ffff);
        self.rec(7, cta as u32, key, key + 1, false);
        self.rec_rel(sync_key(2, cta as u64, off as u64, phase as u64));
        if completes {
            self.rec(7, cta as u32, key + 1, key + 2, true);
            self.rec(6, cta as u32, off as u64, off as u64 + 8, true);
            self.rec_phase(cta, off, Some(phase), true);
        }
        if completes && !self.bar_cfg.peekskip.is_empty() {
            self.peekwin_aba(cta, off, phase);
        }
        let mb = self.mbars.get_mut(&(cta, off)).unwrap();
        if mb.pending == 0 && mb.tx == 0 {
            let p = (mb.phase % 2) as usize;
            mb.done[p] = std::mem::take(&mut mb.cur);
            mb.phase += 1;
            mb.pending = mb.expected;
        }
    }

    /// Complete pending async op `i` (a scheduler choice).
    pub fn complete_async(&mut self, i: usize) {
        if let Some(ns) = self.num.as_mut() {
            ns.taint.begin_async();
        }
        let op = self.pending.remove(i);
        let cta = match &op.kind {
            AsyncKind::Tma { cta, .. }
            | AsyncKind::Mma { cta, .. }
            | AsyncKind::Cp { cta, .. }
            | AsyncKind::Commit { cta, .. }
            | AsyncKind::Clc { cta, .. }
            | AsyncKind::TmaStore { cta, .. }
            | AsyncKind::CpArrive { cta, .. } => *cta,
        };
        self.rec_async_gate(op.issuer, op.actor);
        // Async-after-exit is checked at the end, order-independently (exit_checks).
        let rec = (cta, op.actor, vc_get(&op.vc, op.actor), op.pc);
        if let Err(k) = self.completed.binary_search(&rec) {
            self.completed.insert(k, rec);
        }
        // A pair MMA / cp also writes the peer's TMEM: the peer must observe it before exiting.
        if let AsyncKind::Mma { peer: Some(p), .. } | AsyncKind::Cp { peer: Some(p), .. } = &op.kind
        {
            let rec = (*p, op.actor, vc_get(&op.vc, op.actor), op.pc);
            if let Err(k) = self.completed.binary_search(&rec) {
                self.completed.insert(k, rec);
            }
        }
        match op.kind.clone() {
            // Completed through `complete_clc` (a scheduler choice with outcomes).
            AsyncKind::Clc { .. } => unreachable!(),
            AsyncKind::Tma {
                cta,
                dst,
                mbar_cta,
                mbar,
                tmap,
                coords,
            } => self.complete_tma(&op, cta, dst, mbar_cta, mbar, tmap, coords),
            AsyncKind::TmaStore {
                cta,
                src,
                tmap,
                coords,
                group,
            } => self.complete_tma_store(&op, cta, src, tmap, coords, group),
            AsyncKind::Mma {
                cta,
                peer,
                d,
                adesc,
                bdesc,
                idesc,
                enable,
                a_tmem,
                kind,
                scale,
            } => {
                if let Err(e) = self.complete_mma(
                    &op,
                    cta,
                    peer,
                    d,
                    (adesc, bdesc, idesc),
                    enable,
                    a_tmem,
                    (kind, scale),
                ) {
                    self.report(
                        fault_kind(&e),
                        format!("mma:{}", op.pc),
                        format!("{e} (tcgen05.mma at {})", self.k.describe(op.pc as usize)),
                    );
                }
            }
            AsyncKind::Cp {
                cta,
                peer,
                taddr,
                sdesc,
            } => {
                // Pair (GB300, probes/tc_pair_bs/cp): each CTA's smem to its own TMEM.
                for c in std::iter::once(cta).chain(peer) {
                    if let Err(e) = self.complete_tc_cp(&op, c, taddr, sdesc) {
                        self.report(
                            fault_kind(&e),
                            format!("tccp:{}", op.pc),
                            format!("{e} (tcgen05.cp at {})", self.k.describe(op.pc as usize)),
                        );
                        break;
                    }
                }
            }
            AsyncKind::CpArrive {
                cta,
                mbar,
                writes,
                tx,
            } => self.complete_cp_arrive(&op, cta, mbar, writes, tx),
            AsyncKind::Commit { cta, mbar } => {
                self.rec(0, cta as u32, mbar as u64, mbar as u64 + 8, false);
                self.mbar_materialize(cta, mbar);
                if cta != self.threads[op.issuer as usize].cta {
                    if let Err(e) = self.check_alive(cta, op.actor) {
                        self.report("violation", format!("cluster-life:{}", op.pc), e);
                        return;
                    }
                }
                if self.mbars.contains_key(&(cta, mbar)) {
                    let _ = self.mbar_atomic(op.actor, &op.vc, &op.vis, op.pc, cta, mbar);
                }
                match self.mbars.get_mut(&(cta, mbar)) {
                    None => {
                        let msg = format!(
                            "tcgen05.commit at {} arrives on an uninitialized mbarrier at {mbar:#x}",
                            self.k.describe(op.pc as usize)
                        );
                        self.report("violation", format!("commit-uninit:{}", op.pc), msg);
                    }
                    Some(mb) => {
                        if mb.pending == 0 {
                            let msg = format!(
                                "tcgen05.commit at {}: more arrivals than the expected count {}",
                                self.k.describe(op.pc as usize),
                                mb.expected
                            );
                            // Over-arrival is a launch failure on GB300 (B10, a thread arrive);
                            // a commit's arrive is the same mbarrier operation (S-MBAR).
                            self.report("fault", format!("commit-overflow:{}", op.pc), msg);
                            return;
                        }
                        mb.pending -= 1;
                        vc_join(&mut mb.cur, &op.vc);
                        self.mbar_maybe_complete(cta, mbar);
                    }
                }
            }
        }
    }

    /// Decode a K-major shared-memory matrix descriptor -> (start, LBO, SBO, swizzle span).
    /// Layouts (bits 61-63): 0 = no swizzle (8x16 B core matrices), 2 = SWIZZLE_128B,
    /// 4 = SWIZZLE_64B, 6 = SWIZZLE_32B (rows of `span` bytes, 8-row groups SBO apart).
    fn smem_desc(d: u64) -> Result<(u32, u32, u32, u32), String> {
        let start = ((d & 0x3fff) << 4) as u32;
        let lbo = (((d >> 16) & 0x3fff) << 4) as u32;
        let sbo = (((d >> 32) & 0x3fff) << 4) as u32;
        let version = (d >> 46) & 3;
        let base_off = (d >> 49) & 7;
        let lbo_mode = (d >> 52) & 1;
        let layout = (d >> 61) & 7;
        let _ = version; // checked by the caller (`desc_version_check`)
        let span = match layout {
            0 => 0,
            2 => 128,
            4 => 64,
            6 => 32,
            _ => return Err(format!("smem descriptor layout {layout} not modeled")),
        };
        if base_off != 0 || lbo_mode != 0 {
            return Err(format!(
                "smem descriptor with base offset {base_off}, lbo mode {lbo_mode} not modeled"
            ));
        }
        Ok((start, lbo, sbo, span))
    }

    /// tcgen05.cp.32x128b.warpx4: 32 rows x 16 bytes from shared memory (K-major descriptor:
    /// no swizzle = 8-row core matrices SBO apart, rows 16 B apart; swizzled = rows `span`
    /// apart, then the address swizzle) to TMEM columns col..col+3, row r at lane 32w + r of
    /// each sub-partition w; word j = bytes 4j..4j+3 of the row.
    // ASSUME(S-TCCP): see ASSUMPTIONS.md
    fn complete_tc_cp(
        &mut self,
        op: &AsyncOp,
        cta: u16,
        taddr: u32,
        sdesc: u64,
    ) -> Result<(), String> {
        let (lane, col) = ((taddr >> 16) as usize, (taddr & 0xffff) as usize);
        if lane != 0 || col + 4 > TMEM_COLS {
            return Err(format!(
                "TMEM address lane {lane} column {col}: not modeled (lane 0, 4 columns in range)"
            ));
        }
        let v = (sdesc >> 46) & 7;
        if v != 1 {
            let msg = format!(
                "tcgen05.cp at {}: shared-memory descriptor bits 46-48 are {v:#05b}; the PTX ISA requires the fixed value 0b001",
                self.k.describe(op.pc as usize)
            );
            self.report("violation", format!("desc-version:{}", op.pc), msg);
        }
        let (start, _lbo, sbo, sw) = Self::smem_desc(sdesc)?;
        let row = |r: u32| {
            if sw == 0 {
                start + (r / 8) * sbo + (r % 8) * 16
            } else {
                swz(start + (r / 8) * sbo + (r % 8) * sw, sw)
            }
        };
        let last = (0..32).map(row).max().unwrap() + 16;
        if last as usize > self.launch.g(cta as u32).smem_bytes as usize {
            return Err(format!(
                "tcgen05.cp source at smem {start:#x} reads past shared memory (to {last:#x})"
            ));
        }
        let mut words = vec![0u32; 32 * 4];
        for r in 0..32u32 {
            let off = row(r);
            self.proxy_check(cta, off, 16, &op.vc, op.pc, "tcgen05.cp");
            let b = self.access(op.actor, &op.vc, op.pc, Loc::Shared { cta, off }, 16, None);
            for j in 0..4 {
                words[r as usize * 4 + j] =
                    u32::from_le_bytes(b[4 * j..4 * j + 4].try_into().unwrap());
            }
        }
        let mut all = Vec::with_capacity(TMEM_LANES * 4);
        for _ in 0..4 {
            all.extend_from_slice(&words);
        }
        let (_, st) = self.tmem_block(
            cta,
            (0, TMEM_LANES),
            (col, 4),
            op.actor,
            &op.vc,
            op.pc,
            Some(&all),
            true,
        );
        if st & 2 != 0 {
            self.tmem_state_report(2, "tcgen05.cp", op.pc);
        }
        if self.num.is_some() {
            // Byte provenance moves along (scale-factor bytes for the block-scaled MMA kinds).
            // ASSUME(N-HALF-LEAF): see ASSUMPTIONS.md
            let srcs: Vec<Vec<Option<crate::halfin::Src>>> = (0..32u32)
                .map(|r| self.half_mem(Loc::Shared { cta, off: row(r) }, 16, None))
                .collect();
            for l in 0..TMEM_LANES {
                for c in 0..4 {
                    self.num_tmem_forget(cta, l, col + c);
                    let b = &srcs[l % 32][4 * c..4 * c + 4];
                    self.half_mem_set(crate::halfin::tkey(cta, l, col + c, 0), b);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    // ASSUME(E-ASYNC-STEP): see ASSUMPTIONS.md
    #[allow(clippy::too_many_arguments)]
    fn complete_mma(
        &mut self,
        op: &AsyncOp,
        cta0: u16,
        peer: Option<u16>,
        d: u32,
        (adesc, bdesc, idesc): (u64, u64, u32),
        enable: bool,
        a_tmem: bool,
        (kind, _scale): (ptxv_ir::MmaKind, Option<(u32, u32)>),
    ) -> Result<(), String> {
        if kind != ptxv_ir::MmaKind::F16 {
            return self.complete_mma_lp(
                op,
                cta0,
                peer,
                d,
                (adesc, bdesc, idesc),
                enable,
                a_tmem,
                (kind, _scale),
            );
        }
        let f = |lo: u32, n: u32| (idesc >> lo) & ((1 << n) - 1);
        let (sparse, sat, cfmt, afmt, bfmt) = (f(0, 3), f(3, 1), f(4, 2), f(7, 3), f(10, 3));
        let (nega, negb, amaj, bmaj) = (f(13, 1), f(14, 1), f(15, 1), f(16, 1));
        let n = f(17, 6) << 3;
        let m = f(24, 5) << 4;
        // MN-major operands (bits 15/16) are modeled for swizzled layouts (Triton's flash
        // attention: V as the B of P x V).
        if sparse != 0
            || sat != 0
            || nega != 0
            || negb != 0
            || (amaj != 0 && a_tmem)
            || f(30, 2) != 0
        {
            return Err(format!(
                "instruction descriptor {idesc:#x}: sparsity/saturate/negate/MN-major/shift not modeled yet"
            ));
        }
        if cfmt != 1 || afmt != 1 || bfmt != 1 {
            return Err(format!(
                "instruction descriptor {idesc:#x}: only bf16 x bf16 -> f32 is modeled so far (D fmt {cfmt}, A {afmt}, B {bfmt})"
            ));
        }
        // cta_group::2 (E-PAIR): CTA i of the pair holds A rows / D rows 128i.., and B rows
        // (N/2)i.. at the same descriptor addresses (GB300: r1c_2sm bitwise).
        let ctas: Vec<u16> = std::iter::once(cta0).chain(peer).collect();
        // M=64 (cta_group::1, cuBLAS nvjet): D row i is TMEM lane lane0 + 32(i/16) + i%16,
        // lane0 = 0 or 16 (half of each warp's sub-partition; read back with
        // tcgen05.ld.16x64b). ASSUME(S-TMEM): validated bitwise against GB300 (r2_cublas_gemm).
        let m64 = m == 64 && ctas.len() == 1;
        let ok_n = if m64 { n % 8 == 0 } else { n % 16 == 0 };
        if !(m64 || m == 128 * ctas.len() as u32) || n == 0 || !ok_n || n > 256 {
            return Err(format!(
                "MMA shape M={m} N={n} not modeled (M=128 per CTA of the group, N%16==0; or M=64 N%8==0 in one CTA)"
            ));
        }
        let rows = if m64 { 64usize } else { 128 };
        let nh = n / ctas.len() as u32;
        let (lane0, col0) = ((d >> 16) as usize, (d & 0xffff) as usize);
        if !(lane0 == 0 || m64 && lane0 == 16) || col0 + n as usize > TMEM_COLS {
            return Err(format!(
                "MMA D at TMEM lane {lane0} column {col0} (+{n}) out of range"
            ));
        }
        // D's TMEM lane runs (first lane, lanes), in row order.
        let runs: Vec<(usize, usize)> = if m64 {
            (0..4).map(|g| (lane0 + 32 * g, 16)).collect()
        } else {
            vec![(lane0, 128)]
        };
        let row_lane = |mi: usize| {
            if m64 {
                lane0 + 32 * (mi / 16) + mi % 16
            } else {
                lane0 + mi
            }
        };
        // PTX ISA: descriptor bits 46-48 are the fixed constant 0b001. Triton 3.5 leaves them 0
        // and GB300 computes correctly anyway: a contract violation, reported, not a fault.
        for (which, dsc) in [("A", adesc), ("B", bdesc)] {
            if which == "A" && a_tmem {
                continue;
            }
            let v = (dsc >> 46) & 7;
            if v != 1 {
                let msg = format!(
                    "tcgen05.mma at {}: {which} shared-memory descriptor bits 46-48 are {v:#05b}; the PTX ISA requires the fixed value 0b001 (GB300 ignores them)",
                    self.k.describe(op.pc as usize)
                );
                self.report("violation", format!("desc-version:{}", op.pc), msg);
            }
        }
        let (sa, lboa, sboa, swa) = if a_tmem {
            (0, 0, 0, 0)
        } else {
            Self::smem_desc(adesc)?
        };
        let (sb, lbob, sbob, swb) = Self::smem_desc(bdesc)?;
        // MN-major is carried as a flag bit on the swizzle span.
        const MN: u32 = 1 << 31;
        for (maj, sw) in [(amaj, swa), (bmaj, swb)] {
            if maj != 0 && sw == 0 {
                return Err(format!(
                    "instruction descriptor {idesc:#x}: MN-major operand without swizzle not modeled"
                ));
            }
        }
        let swa = if amaj != 0 { swa | MN } else { swa };
        let swb = if bmaj != 0 { swb | MN } else { swb };
        const K: usize = 16;
        // ASSUME(S-TMA): swizzled K-major: row r at (r/8)*SBO + (r%8)*span, K contiguous within
        // the row (the K=16 slice must not cross it), then the address swizzle.
        // MN-major, swizzled (CUTLASS's canonical layout ((T,s/16,m),(8,k)) :
        // ((1,T,LBO),(s/2,SBO)) for span s): 8 K-rows of `s` bytes along M/N per atom, atoms
        // LBO apart along M/N and SBO apart along K, then the address swizzle.
        let elem = |start: u32, lbo: u32, sbo: u32, sw: u32, r: usize, k: usize| {
            if sw & MN != 0 {
                let sw = sw & !MN;
                let per = sw as usize / 2;
                swz(
                    start
                        + (r % per) as u32 * 2
                        + (k % 8) as u32 * sw
                        + (r / per) as u32 * lbo
                        + (k / 8) as u32 * sbo,
                    sw,
                )
            } else if sw == 0 {
                start
                    + (r as u32 / 8) * sbo
                    + (r as u32 % 8) * 16
                    + (k as u32 / 8) * lbo
                    + (k as u32 % 8) * 2
            } else {
                swz(
                    start + (r as u32 / 8) * sbo + (r as u32 % 8) * sw + k as u32 * 2,
                    sw,
                )
            }
        };
        for (start, sw) in [(sa, swa), (sb, swb)] {
            if a_tmem && start == sa && sw == swa && start == 0 && sw == 0 || sw & MN != 0 {
                continue;
            }
            if sw != 0 && (start % sw) as usize + K * 2 > sw as usize {
                return Err(format!(
                    "MMA K slice at smem {start:#x} crosses a {sw}-byte swizzle row: not modeled"
                ));
            }
        }
        let rd_mat = |me: &mut Machine, cta: u16, start, lbo, sbo, sw, rows: usize| -> Vec<f64> {
            let mut v = Vec::with_capacity(rows * K);
            // One access per run of elements that are contiguous in shared memory: 8 (one
            // 16-byte core-matrix row / swizzle chunk) when the start is 16-byte aligned.
            let run = if (0..rows).all(|r| {
                (0..K).all(|k| {
                    elem(start, lbo, sbo, sw, r, k)
                        == elem(start, lbo, sbo, sw, r, k & !7) + 2 * (k as u32 & 7)
                })
            }) {
                8
            } else {
                1
            };
            for r in 0..rows {
                for k0 in (0..K).step_by(run) {
                    let off = elem(start, lbo, sbo, sw, r, k0);
                    let len = 2 * run;
                    me.proxy_check(cta, off, len, &op.vc, op.pc, "tcgen05.mma");
                    let b = me.access(op.actor, &op.vc, op.pc, Loc::Shared { cta, off }, len, None);
                    for e in b.chunks_exact(2) {
                        v.push(
                            f32::from_bits((u16::from_le_bytes([e[0], e[1]]) as u32) << 16) as f64,
                        );
                    }
                }
            }
            v
        };
        let a_mats: &[(u32, u32, u32, u32, u32)] = if a_tmem {
            &[]
        } else {
            &[(sa, lboa, sboa, swa, rows as u32)]
        };
        for &(start, lbo, sbo, sw, rows) in a_mats.iter().chain(&[(sb, lbob, sbob, swb, nh)]) {
            let last = (0..rows as usize)
                .flat_map(|r| {
                    [
                        elem(start, lbo, sbo, sw, r, 0),
                        elem(start, lbo, sbo, sw, r, K - 1),
                    ]
                })
                .max()
                .unwrap()
                + 2;
            if last as usize > self.launch.g(cta0 as u32).smem_bytes as usize {
                return Err(format!(
                    "MMA operand at smem {start:#x} reads past shared memory (to {last:#x})"
                ));
            }
        }
        let mut b = Vec::new();
        for &c in &ctas {
            b.extend(rd_mat(self, c, sb, lbob, sbob, swb, nh as usize));
        }
        // A in TMEM: row i at the lane D's row i uses (relative to A's address), K/2 columns
        // of two bf16 each, the even element in the low half.
        // ASSUME(S-TMEM): see ASSUMPTIONS.md
        let a_all: Vec<Vec<f64>> = if a_tmem {
            let (al, ac) = ((adesc >> 16) as usize & 0xffff, (adesc & 0xffff) as usize);
            if !(al == 0 || m64 && al == 16) || ac + K / 2 > TMEM_COLS {
                return Err(format!("MMA A at TMEM lane {al} column {ac} out of range"));
            }
            let mut out = Vec::new();
            for &c in &ctas {
                let mut v = Vec::with_capacity(rows * K);
                for mi in 0..rows {
                    let lane = row_lane(mi) - lane0 + al;
                    for j in 0..K / 2 {
                        let (w, _) =
                            self.tmem_access(c, lane, ac + j, op.actor, &op.vc, op.pc, None);
                        for h in 0..2 {
                            v.push(f32::from_bits((w >> (16 * h)) << 16) as f64);
                        }
                    }
                }
                out.push(v);
            }
            out
        } else {
            ctas.iter()
                .map(|&c| rd_mat(self, c, sa, lboa, sboa, swa, rows))
                .collect()
        };
        // Numerics: the 16-bit provenance of each operand element (an input element, a rounded
        // value, zero, or unknown).
        use crate::halfin::HalfOp;
        let prov_mat = |me: &mut Machine, cta: u16, start, lbo, sbo, sw, rows: usize| {
            (0..rows)
                .flat_map(|r| (0..K).map(move |k| (r, k)))
                .map(|(r, k)| {
                    let off = elem(start, lbo, sbo, sw, r, k);
                    let b = me.half_mem(Loc::Shared { cta, off }, 2, None);
                    me.half_op(b[0], b[1])
                })
                .collect::<Vec<_>>()
        };
        let (a_prov, b_prov): (Vec<Vec<HalfOp>>, Vec<HalfOp>) = if self.num.is_some() {
            let mut ap = Vec::new();
            for &c in &ctas {
                if a_tmem {
                    let (al, ac) = ((adesc >> 16) as usize & 0xffff, (adesc & 0xffff) as usize);
                    let mut v = Vec::with_capacity(rows * K);
                    for mi in 0..rows {
                        let lane = row_lane(mi) - lane0 + al;
                        for j in 0..K / 2 {
                            let w = self.half_tmem(c, lane, ac + j);
                            v.push(self.half_op(w[0], w[1]));
                            v.push(self.half_op(w[2], w[3]));
                        }
                    }
                    ap.push(v);
                } else {
                    ap.push(prov_mat(self, c, sa, lboa, sboa, swa, rows));
                }
            }
            let mut bp = Vec::new();
            for &c in &ctas {
                bp.extend(prov_mat(self, c, sb, lbob, sbob, swb, nh as usize));
            }
            (ap, bp)
        } else {
            (vec![], vec![])
        };
        // ASSUME(N-MMA): the model is calibrated on finite normal (or zero) inputs only.
        let normal = |x: f64| x == 0.0 || (x.is_finite() && x.abs() >= f32::MIN_POSITIVE as f64);
        let mut outside = a_all
            .iter()
            .flatten()
            .chain(b.iter())
            .find(|&&x| !normal(x))
            .map(|x| format!("MMA input {x:e}"));
        let model = self.launch.mma_model;
        let mut uninit = false;
        let mut unalloc = false;
        // Pass 1: read the accumulators (race-checked TMEM reads; pass 3 overwrites them, so
        // no reader epoch is left).
        let mut accs: Vec<Vec<f32>> = Vec::with_capacity(ctas.len());
        for &cta in &ctas {
            if !enable {
                let tm = &self.tmem[cta as usize];
                unalloc |= (0..rows).any(|mi| {
                    (0..n as usize).any(|ni| *tm.init.get(tmem_idx(row_lane(mi), col0 + ni)) == 2)
                });
                accs.push(vec![0.0; rows * n as usize]);
                continue;
            }
            let mut words = Vec::with_capacity(rows * n as usize);
            for &run in &runs {
                let (w, st) = self.tmem_block(
                    cta,
                    run,
                    (col0, n as usize),
                    op.actor,
                    &op.vc,
                    op.pc,
                    None,
                    false,
                );
                words.extend(w);
                unalloc |= st & 2 != 0;
                uninit |= st & 1 != 0;
            }
            let v: Vec<f32> = words.into_iter().map(f32::from_bits).collect();
            if outside.is_none() {
                if let Some(acc) = v.iter().find(|a| !normal(**a as f64)) {
                    outside = Some(format!("MMA accumulator {acc:e}"));
                }
            }
            accs.push(v);
        }
        // Pass 2: the arithmetic, memoized on its exact inputs (a pure function of them).
        let results: Vec<u32> = if outside.is_some() {
            // Outside the model: D is unknown. Write NaN so later reads are not also reported
            // as reads of unwritten TMEM; the verdict says "unmodeled".
            vec![f32::NAN.to_bits(); ctas.len() * rows * n as usize]
        } else {
            use std::hash::Hash;
            use std::hash::Hasher;
            let key = {
                let mut h1 = std::collections::hash_map::DefaultHasher::new();
                let mut h2 = FastHasher {
                    s: 0x9e37_79b9_7f4a_7c15,
                };
                for h in [&mut h1 as &mut dyn Hasher, &mut h2 as &mut dyn Hasher] {
                    model.name().hash(&mut HashAdapter(h));
                    enable.hash(&mut HashAdapter(h));
                    n.hash(&mut HashAdapter(h));
                    for x in a_all.iter().flatten().chain(b.iter()) {
                        h.write_u64(x.to_bits());
                    }
                    for x in accs.iter().flatten() {
                        h.write_u32(x.to_bits());
                    }
                }
                (h1.finish(), h2.finish())
            };
            let hit = self.mma_cache.lock().unwrap().get(&key).cloned();
            match hit {
                Some(r) => r.as_ref().clone(),
                None => {
                    let mut out = Vec::with_capacity(ctas.len() * rows * n as usize);
                    for (ci, _) in ctas.iter().enumerate() {
                        let a = &a_all[ci];
                        for mi in 0..rows {
                            for ni in 0..n as usize {
                                let terms: Vec<fp::Term> = (0..K)
                                    .map(|k| {
                                        fp::Term::prod(
                                            a[mi * K + k],
                                            b[ni * K + k],
                                            model.exp_from_inputs,
                                        )
                                    })
                                    .collect();
                                let acc = accs[ci][mi * n as usize + ni];
                                out.push(model.apply_terms(acc, &terms).to_bits());
                            }
                        }
                    }
                    let mut c = self.mma_cache.lock().unwrap();
                    if c.len() > 4096 {
                        c.clear();
                    }
                    c.insert(key, std::sync::Arc::new(out.clone()));
                    out
                }
            }
        };
        // Pass 3: write D (race-checked) and the numerics shadow.
        let per = rows * n as usize;
        for (ci, &cta) in ctas.iter().enumerate() {
            let mut at = ci * per;
            for &run in &runs {
                let len = run.1 * n as usize;
                self.tmem_block(
                    cta,
                    run,
                    (col0, n as usize),
                    op.actor,
                    &op.vc,
                    op.pc,
                    Some(&results[at..at + len]),
                    true,
                );
                at += len;
            }
            if outside.is_none() && self.num.is_some() {
                let a = &a_all[ci];
                for mi in 0..rows {
                    for ni in 0..n as usize {
                        let r = f32::from_bits(results[ci * per + mi * n as usize + ni]);
                        let prods: Vec<f64> =
                            (0..K).map(|k| a[mi * K + k] * b[ni * K + k]).collect();
                        let ops: Vec<HalfOp> = (0..K)
                            .flat_map(|k| [a_prov[ci][mi * K + k], b_prov[ni * K + k]])
                            .collect();
                        let acc = accs[ci][mi * n as usize + ni];
                        // ASSUME(N-MMA): one step of the calibrated model per K = 16 slice.
                        self.num_mma(
                            cta,
                            row_lane(mi),
                            col0 + ni,
                            enable,
                            acc,
                            (&prods, &ops),
                            r,
                            op.pc,
                            &|n, a| model.eps_terms(n + a as usize),
                            None,
                            0,
                        );
                    }
                }
            }
        }
        if let Some(what) = outside {
            return Err(format!(
                "{what} (subnormal/Inf/NaN) is outside the calibrated tensor-core model: not modeled"
            ));
        }
        if unalloc {
            self.tmem_state_report(2, "tcgen05.mma", op.pc);
        }
        if uninit {
            let msg = format!(
                "tcgen05.mma at {} accumulates (enable_input_d = 1) into TMEM that was never written since allocation: the result includes garbage",
                self.k.describe(op.pc as usize)
            );
            self.report("violation", format!("mma-uninit-acc:{}", op.pc), msg);
        }
        Ok(())
    }

    /// `tcgen05.mma` of the narrow kinds (kind::f8f6f4 and the block-scaled kinds): K-major
    /// operands in shared memory, M = 128 per CTA of the group (or M = 64), f32 or f16 D.
    /// Operand bytes per row and instruction: 32 (two 16-byte core-matrix rows); kind::f8f6f4
    /// reads 16 elements per 16-byte row (FP8: a byte each; FP6: 12 bytes, LSB-first bit
    /// stream; FP4: 8 bytes, low nibble first; GB300: probes/mma_f8).
    // ASSUME(N-MMA-F8, S-TMA): see ASSUMPTIONS.md
    #[allow(clippy::too_many_arguments)]
    fn complete_mma_lp(
        &mut self,
        op: &AsyncOp,
        cta0: u16,
        peer: Option<u16>,
        d: u32,
        (adesc, bdesc, idesc): (u64, u64, u32),
        enable: bool,
        a_tmem: bool,
        (kind, scale): (ptxv_ir::MmaKind, Option<(u32, u32)>),
    ) -> Result<(), String> {
        use ptxv_ir::Lp;
        use ptxv_ir::MmaKind;
        let f = |lo: u32, n: u32| (idesc >> lo) & ((1 << n) - 1);
        let scaled = scale.is_some();
        let (afmt, bfmt) = (f(7, 3), f(10, 3));
        let (sparse, nega, negb, amaj, bmaj) = (f(2, 1), f(13, 1), f(14, 1), f(15, 1), f(16, 1));
        let n = f(17, 6) << 3;
        let m = f(24, 5) << 4;
        let f8 = |c: u32| match c {
            0 => Some(Lp::E4m3),
            1 => Some(Lp::E5m2),
            3 => Some(Lp::E2m3),
            4 => Some(Lp::E3m2),
            5 => Some(Lp::E2m1),
            _ => None,
        };
        let bad = |what: &str| {
            Err(format!(
                "instruction descriptor {idesc:#x} (kind::{kind:?}): {what} not modeled yet"
            ))
        };
        if sparse != 0 || nega != 0 || negb != 0 || amaj != 0 || bmaj != 0 || a_tmem {
            return bad("sparsity / negate / MN-major / A in TMEM");
        }
        // (A format, B format, f16 D, elements per instruction).
        let (fa, fb, f16_d, kn) = match kind {
            MmaKind::F8f6f4 => {
                if f(0, 2) != 0 || f(3, 1) != 0 || f(30, 2) != 0 || f(4, 2) > 1 {
                    return bad("saturate / shift / D format");
                }
                let (Some(a), Some(b)) = (f8(afmt), f8(bfmt)) else {
                    return bad("operand format");
                };
                (a, b, f(4, 2) == 0, 32usize)
            }
            MmaKind::Mxf8f6f4 => {
                if f(0, 2) != 0 || f(31, 1) != 0 || f(23, 1) != 1 {
                    return bad("sparse id / K size / non-ue8m0 scales");
                }
                let (Some(a), Some(b)) = (f8(afmt), f8(bfmt)) else {
                    return bad("operand format");
                };
                (a, b, false, 32)
            }
            MmaKind::Mxf4 | MmaKind::Mxf4nvf4 { .. } => {
                if f(0, 2) != 0 || f(31, 1) != 0 || afmt != 1 || bfmt != 1 {
                    return bad("sparse id / K = 96 / operand format (E2M1 = 1)");
                }
                (Lp::E2m1, Lp::E2m1, false, 64)
            }
            MmaKind::F16 => unreachable!(),
        };
        if scaled != !matches!(kind, MmaKind::F8f6f4) {
            return bad("scale-factor operands");
        }
        // Block scaling: (scale format, elements per scale, the byte of the TMEM word that holds
        // block b's scale). GB300 (probes/mma_bs): 1X reads byte sf_id, 2X bytes sf_id + b
        // (sf_id 0 or 2), 4X bytes b (sf_id 0); ue4m3 only with 4X; other ids are an illegal
        // instruction.
        let (asf, bsf) = (f(29, 2), f(4, 2));
        let ue8 = f(23, 1) == 1;
        let bsc: Option<(Lp, usize, [u32; 2])> = match kind {
            MmaKind::Mxf8f6f4 => Some((Lp::Ue8m0, 32, [asf, bsf])),
            MmaKind::Mxf4 | MmaKind::Mxf4nvf4 { x4: false } => {
                if asf % 2 != 0 || bsf % 2 != 0 || !ue8 {
                    return Err(format!(
                        "instruction descriptor {idesc:#x}: scale_vec::2X with scale-factor id {asf}/{bsf} or ue4m3 scales: illegal instruction (GB300)"
                    ));
                }
                Some((Lp::Ue8m0, 32, [asf, bsf]))
            }
            MmaKind::Mxf4nvf4 { x4: true } => {
                if asf != 0 || bsf != 0 {
                    return Err(format!(
                        "instruction descriptor {idesc:#x}: scale_vec::4X with scale-factor id {asf}/{bsf}: illegal instruction (GB300)"
                    ));
                }
                Some((if ue8 { Lp::Ue8m0 } else { Lp::Ue4m3 }, 16, [0, 0]))
            }
            _ => None,
        };
        let ctas: Vec<u16> = std::iter::once(cta0).chain(peer).collect();
        let m64 = m == 64 && ctas.len() == 1;
        let ok_n = if m64 { n % 8 == 0 } else { n % 16 == 0 };
        if !(m64 || m == 128 * ctas.len() as u32) || n == 0 || !ok_n || n > 256 {
            return Err(format!(
                "MMA shape M={m} N={n} not modeled (M=128 per CTA of the group, N%16==0; or M=64 N%8==0 in one CTA)"
            ));
        }
        if scaled && m64 {
            return bad("block scaling with M = 64");
        }
        let rows = if m64 { 64usize } else { 128 };
        let nh = n / ctas.len() as u32;
        let (lane0, col0) = ((d >> 16) as usize, (d & 0xffff) as usize);
        if !(lane0 == 0 || m64 && lane0 == 16) || col0 + n as usize > TMEM_COLS {
            return Err(format!(
                "MMA D at TMEM lane {lane0} column {col0} (+{n}) out of range"
            ));
        }
        let runs: Vec<(usize, usize)> = if m64 {
            (0..4).map(|g| (lane0 + 32 * g, 16)).collect()
        } else {
            vec![(lane0, 128)]
        };
        let row_lane = |mi: usize| {
            if m64 {
                lane0 + 32 * (mi / 16) + mi % 16
            } else {
                lane0 + mi
            }
        };
        for (which, dsc) in [("A", adesc), ("B", bdesc)] {
            let v = (dsc >> 46) & 7;
            if v != 1 {
                let msg = format!(
                    "tcgen05.mma at {}: {which} shared-memory descriptor bits 46-48 are {v:#05b}; the PTX ISA requires the fixed value 0b001 (GB300 ignores them)",
                    self.k.describe(op.pc as usize)
                );
                self.report("violation", format!("desc-version:{}", op.pc), msg);
            }
        }
        let (sa, lboa, sboa, swa) = Self::smem_desc(adesc)?;
        let (sb, lbob, sbob, swb) = Self::smem_desc(bdesc)?;
        // Byte `kb` (0..32) of row r's K slice: K-major, no swizzle: 16-byte core-matrix rows,
        // 8-row groups SBO apart, the two 16-byte K chunks LBO apart; swizzled: row r at
        // (r/8)*SBO + (r%8)*span, the slice contiguous, then the address swizzle.
        // ASSUME(S-TMA): see ASSUMPTIONS.md
        let byte_at = |start: u32, lbo: u32, sbo: u32, sw: u32, r: usize, kb: usize| {
            if sw == 0 {
                start
                    + (r as u32 / 8) * sbo
                    + (r as u32 % 8) * 16
                    + (kb as u32 / 16) * lbo
                    + kb as u32 % 16
            } else {
                swz(
                    start + (r as u32 / 8) * sbo + (r as u32 % 8) * sw + kb as u32,
                    sw,
                )
            }
        };
        for (start, sw) in [(sa, swa), (sb, swb)] {
            if sw != 0 && (start % sw) as usize + 32 > sw as usize {
                return Err(format!(
                    "MMA K slice at smem {start:#x} crosses a {sw}-byte swizzle row: not modeled"
                ));
            }
        }
        for &(start, lbo, sbo, sw, nr) in &[
            (sa, lboa, sboa, swa, rows as u32),
            (sb, lbob, sbob, swb, nh),
        ] {
            let last = (0..nr as usize)
                .flat_map(|r| {
                    [
                        byte_at(start, lbo, sbo, sw, r, 0),
                        byte_at(start, lbo, sbo, sw, r, 31),
                    ]
                })
                .max()
                .unwrap()
                + 1;
            if last as usize > self.launch.g(cta0 as u32).smem_bytes as usize {
                return Err(format!(
                    "MMA operand at smem {start:#x} reads past shared memory (to {last:#x})"
                ));
            }
        }
        // Raw bytes: rows x 32, two 16-byte accesses per row.
        let rd = |me: &mut Machine, cta: u16, start, lbo, sbo, sw, nr: usize| -> Vec<u8> {
            let mut v = Vec::with_capacity(nr * 32);
            for r in 0..nr {
                for c in 0..2 {
                    let off = byte_at(start, lbo, sbo, sw, r, 16 * c);
                    me.proxy_check(cta, off, 16, &op.vc, op.pc, "tcgen05.mma");
                    v.extend(me.access(
                        op.actor,
                        &op.vc,
                        op.pc,
                        Loc::Shared { cta, off },
                        16,
                        None,
                    ));
                }
            }
            v
        };
        // Codes of one row's slice: 16 elements per 16-byte chunk (32 packed FP4 for the K = 64
        // kinds).
        let epc = kn / 2;
        let codes = |bytes: &[u8], t: Lp| -> Vec<u32> {
            let w = crate::lowp::width(t) as usize;
            let mut out = Vec::with_capacity(kn);
            for ch in bytes.chunks(16) {
                let bits = u128::from_le_bytes(ch.try_into().unwrap());
                for j in 0..epc {
                    out.push(((bits >> (w * j)) & ((1 << w) - 1)) as u32);
                }
            }
            out
        };
        let mut a_bytes = Vec::new();
        let mut b_bytes = Vec::new();
        for &c in &ctas {
            a_bytes.push(rd(self, c, sa, lboa, sboa, swa, rows));
            b_bytes.extend(rd(self, c, sb, lbob, sbob, swb, nh as usize));
        }
        let to_ops = |bytes: &[u8], t: Lp| -> Vec<crate::mma_lp::Operand> {
            bytes
                .chunks(32)
                .flat_map(|row| codes(row, t))
                .map(|c| crate::mma_lp::Operand::new(c, t))
                .collect()
        };
        let a_all: Vec<Vec<crate::mma_lp::Operand>> =
            a_bytes.iter().map(|b| to_ops(b, fa)).collect();
        let b_ops = to_ops(&b_bytes, fb);
        // ASSUME(N-MMA-F8): calibrated on finite operands (subnormals included) and finite
        // accumulators (f32: normal or zero).
        let mut outside = a_all
            .iter()
            .flatten()
            .chain(b_ops.iter())
            .find(|x| !x.v.is_finite())
            .map(|x| format!("MMA input {}", x.v));
        let model = if f16_d {
            crate::mma_lp::DEFAULT_LP_MMA_F16
        } else {
            crate::mma_lp::DEFAULT_LP_MMA
        };
        let mut uninit = false;
        let mut unalloc = false;
        let mut accs: Vec<Vec<u32>> = Vec::with_capacity(ctas.len());
        for &cta in &ctas {
            if !enable {
                let tm = &self.tmem[cta as usize];
                unalloc |= (0..rows).any(|mi| {
                    (0..n as usize).any(|ni| *tm.init.get(tmem_idx(row_lane(mi), col0 + ni)) == 2)
                });
                accs.push(vec![0; rows * n as usize]);
                continue;
            }
            let mut words = Vec::with_capacity(rows * n as usize);
            for &run in &runs {
                let (w, st) = self.tmem_block(
                    cta,
                    run,
                    (col0, n as usize),
                    op.actor,
                    &op.vc,
                    op.pc,
                    None,
                    false,
                );
                words.extend(w);
                unalloc |= st & 2 != 0;
                uninit |= st & 1 != 0;
            }
            if outside.is_none() {
                let ok = |w: &u32| {
                    if f16_d {
                        crate::fp::f16_to_f64(*w as u16).is_finite()
                    } else {
                        let x = f32::from_bits(*w);
                        x == 0.0 || x.is_normal()
                    }
                };
                if let Some(w) = words.iter().find(|w| !ok(w)) {
                    outside = Some(format!("MMA accumulator {w:#x}"));
                }
            }
            accs.push(words);
        }
        // Scale factors (block-scaled kinds), per D row m in sub-partition w = m / 32 (GB300,
        // probes/mma_bs): A's for row m at TMEM lane m, column [scale_a] + w; B's for row n at
        // lane 32 w + n % 32, column [scale_b] + n / 32 (every sub-partition holds its own copy).
        // Codes per (row, block), row-major. cta_group::2 (probes/tc_pair_bs): each CTA's D rows
        // use the scale factors in that CTA's TMEM, so each CTA holds SFB for all N columns.
        // ASSUME(N-MMA-BS, S-TMEM, E-PAIR): see ASSUMPTIONS.md
        let nb = bsc.map_or(1, |(_, blk, _)| kn / blk);
        let mut sf_uninit = false;
        let mut sfa: Vec<Vec<u32>> = vec![Vec::new(); ctas.len()];
        let mut sfb: Vec<Vec<u32>> = vec![Vec::new(); ctas.len()]; // per CTA: (w, n, block)
        if let (Some((sft, _, ids)), Some((sa_addr, sb_addr))) = (bsc, scale) {
            // Address bits 30-31 are ignored (GB300: CUTLASS puts the sf_id there too; varying
            // them alone leaves D unchanged, the idesc sf_id selects the byte).
            let (sa_addr, sb_addr) = (sa_addr & 0x3fff_ffff, sb_addr & 0x3fff_ffff);
            let (la, ca) = ((sa_addr >> 16) as usize, (sa_addr & 0xffff) as usize);
            let (lb, cb) = ((sb_addr >> 16) as usize, (sb_addr & 0xffff) as usize);
            if la != 0
                || lb != 0
                || ca + 4 > TMEM_COLS
                || cb + (n as usize).div_ceil(32) > TMEM_COLS
            {
                return Err(format!(
                    "scale factors at TMEM lane {la}/{lb}, column {ca}/{cb}: not modeled (lane 0, in range)"
                ));
            }
            let byte = |w: u32, b: usize, id: u32| (w >> (8 * (id as usize + b))) & 0xff;
            let mut rd_word = |me: &mut Machine, cta: u16, lane: usize, col: usize| -> u32 {
                let (w, st) = me.tmem_access(cta, lane, col, op.actor, &op.vc, op.pc, None);
                unalloc |= st == 2;
                sf_uninit |= st == 0;
                w
            };
            for (ci, &cta) in ctas.iter().enumerate() {
                for mi in 0..rows {
                    let w = rd_word(self, cta, mi, ca + mi / 32);
                    for b in 0..nb {
                        sfa[ci].push(byte(w, b, ids[0]));
                    }
                }
                for sp in 0..4 {
                    for ni in 0..n as usize {
                        let w = rd_word(self, cta, 32 * sp + ni % 32, cb + ni / 32);
                        for b in 0..nb {
                            sfb[ci].push(byte(w, b, ids[1]));
                        }
                    }
                }
            }
            let nan = |c: &u32| crate::mma_lp::Operand::scale(*c, sft).v.is_nan();
            if outside.is_none() && sfa.iter().chain(&sfb).flatten().any(nan) {
                outside = Some("MMA scale factor NaN".into());
            }
        }
        let sft = bsc.map(|x| x.0);
        let per = rows * n as usize;
        let results: Vec<u32> = if outside.is_some() {
            vec![f32::NAN.to_bits(); ctas.len() * per]
        } else {
            use std::hash::Hash;
            use std::hash::Hasher;
            let key = {
                let mut h1 = std::collections::hash_map::DefaultHasher::new();
                let mut h2 = FastHasher {
                    s: 0x51ed_270b_27a4_c3d1,
                };
                for h in [&mut h1 as &mut dyn Hasher, &mut h2 as &mut dyn Hasher] {
                    model.name().hash(&mut HashAdapter(h));
                    (kind, idesc, enable, n).hash(&mut HashAdapter(h));
                    for x in a_bytes.iter().flatten().chain(b_bytes.iter()) {
                        h.write_u8(*x);
                    }
                    for x in accs
                        .iter()
                        .flatten()
                        .chain(sfa.iter().flatten())
                        .chain(sfb.iter().flatten())
                    {
                        h.write_u32(*x);
                    }
                }
                (h1.finish(), h2.finish())
            };
            let hit = self.mma_cache.lock().unwrap().get(&key).cloned();
            match hit {
                Some(r) => r.as_ref().clone(),
                None => {
                    use crate::mma_lp::Operand;
                    let blk = kn / nb;
                    let mut out = Vec::with_capacity(ctas.len() * per);
                    for (ci, a) in a_all.iter().enumerate() {
                        for mi in 0..rows {
                            for ni in 0..n as usize {
                                let acc = enable.then_some(accs[ci][mi * n as usize + ni]);
                                let (ar, br) =
                                    (&a[mi * kn..mi * kn + kn], &b_ops[ni * kn..ni * kn + kn]);
                                let Some(st) = sft else {
                                    out.push(model.step(acc, ar, br, f16_d));
                                    continue;
                                };
                                let sa: Vec<Operand> = (0..nb)
                                    .map(|b| Operand::scale(sfa[ci][mi * nb + b], st))
                                    .collect();
                                let sp = mi / 32;
                                let sb: Vec<Operand> = (0..nb)
                                    .map(|b| {
                                        Operand::scale(sfb[ci][(sp * n as usize + ni) * nb + b], st)
                                    })
                                    .collect();
                                out.push(if kn == 64 {
                                    let p: Vec<f64> =
                                        ar.iter().zip(br).map(|(x, y)| x.v * y.v).collect();
                                    crate::mma_lp::DEFAULT_FP4_MMA.step(acc, &p, &sa, &sb, blk)
                                } else {
                                    let t: Vec<Operand> = (0..kn)
                                        .map(|j| {
                                            ar[j].mul(&br[j]).mul(&sa[j / blk]).mul(&sb[j / blk])
                                        })
                                        .collect();
                                    model.step_terms(acc, &t, f16_d)
                                });
                            }
                        }
                    }
                    let mut c = self.mma_cache.lock().unwrap();
                    if c.len() > 4096 {
                        c.clear();
                    }
                    c.insert(key, std::sync::Arc::new(out.clone()));
                    out
                }
            }
        };
        if outside.is_none()
            && f16_d
            && results
                .iter()
                .any(|&w| crate::fp::f16_to_f64(w as u16).is_infinite())
        {
            outside = Some("MMA f16 result overflow".into());
        }
        for (ci, &cta) in ctas.iter().enumerate() {
            let mut at = ci * per;
            for &run in &runs {
                let len = run.1 * n as usize;
                self.tmem_block(
                    cta,
                    run,
                    (col0, n as usize),
                    op.actor,
                    &op.vc,
                    op.pc,
                    Some(&results[at..at + len]),
                    true,
                );
                at += len;
            }
            // 8-bit operand and scale-factor provenance (N-HALF-LEAF): each product's factors
            // as input-byte leaves, read where the step reads them. FP4 kinds (K = 64): a code
            // per nibble, element j of a row at byte 16 (j / 32) + (j % 32) / 2 of its slice,
            // nibble j % 2.
            // ASSUME(N-HALF-LEAF): see ASSUMPTIONS.md
            let fp4 = kn == 64 && fa == Lp::E2m1 && fb == Lp::E2m1;
            let track = self.num.as_ref().is_some_and(|ns| ns.half.is_on())
                && (kn == 32 && crate::lowp::width(fa) == 8 && crate::lowp::width(fb) == 8 || fp4);
            let slot = |j: usize| {
                if fp4 {
                    (16 * (j / 32) + (j % 32) / 2, (j % 2) as u64)
                } else {
                    (j, 0)
                }
            };
            let (mut a_prov, mut b_prov, mut sa_prov, mut sb_prov) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            if track {
                let op_of = |me: &mut Machine, cta: u16, off: u32, t: Lp, nib: u64| {
                    let s = me.half_mem(Loc::Shared { cta, off }, 1, None)[0];
                    me.code_op(s, t, nib)
                };
                for r in 0..rows {
                    for j in 0..kn {
                        let (kb, nib) = slot(j);
                        let off =
                            byte_at(sa, lboa, sboa, swa, r, 16 * (kb / 16)) + (kb % 16) as u32;
                        a_prov.push(op_of(self, cta, off, fa, nib));
                    }
                }
                for &bc in &ctas {
                    for r in 0..nh as usize {
                        for j in 0..kn {
                            let (kb, nib) = slot(j);
                            let off =
                                byte_at(sb, lbob, sbob, swb, r, 16 * (kb / 16)) + (kb % 16) as u32;
                            b_prov.push(op_of(self, bc, off, fb, nib));
                        }
                    }
                }
                if let (Some((st, _, ids)), Some((sa_addr, sb_addr))) = (bsc, scale) {
                    let (sa_addr, sb_addr) = (sa_addr & 0x3fff_ffff, sb_addr & 0x3fff_ffff);
                    let (ca, cb) = ((sa_addr & 0xffff) as usize, (sb_addr & 0xffff) as usize);
                    for mi in 0..rows {
                        let w = self.half_tmem(cta, mi, ca + mi / 32);
                        for bk in 0..nb {
                            sa_prov.push(self.byte_op(w[ids[0] as usize + bk], st));
                        }
                    }
                    for sp in 0..4 {
                        for ni in 0..n as usize {
                            let w = self.half_tmem(cta, 32 * sp + ni % 32, cb + ni / 32);
                            for bk in 0..nb {
                                sb_prov.push(self.byte_op(w[ids[1] as usize + bk], st));
                            }
                        }
                    }
                }
            }
            if self.num.is_some() {
                for mi in 0..rows {
                    for ni in 0..n as usize {
                        let r = results[ci * per + mi * n as usize + ni];
                        if outside.is_some() || f16_d {
                            self.num_tmem_forget(cta, row_lane(mi), col0 + ni);
                            continue;
                        }
                        let a = &a_all[ci][mi * kn..mi * kn + kn];
                        let b = &b_ops[ni * kn..ni * kn + kn];
                        let blk = kn / nb;
                        let sc = |j: usize| match sft {
                            Some(st) => {
                                crate::mma_lp::Operand::scale(sfa[ci][mi * nb + j / blk], st).v
                                    * crate::mma_lp::Operand::scale(
                                        sfb[ci][((mi / 32) * n as usize + ni) * nb + j / blk],
                                        st,
                                    )
                                    .v
                            }
                            None => 1.0,
                        };
                        let prods: Vec<f64> = a
                            .iter()
                            .zip(b)
                            .enumerate()
                            .map(|(j, (x, y))| x.v * y.v * sc(j))
                            .collect();
                        let (ops, scl) = if track {
                            let ops: Vec<crate::halfin::HalfOp> = (0..kn)
                                .flat_map(|j| [a_prov[mi * kn + j], b_prov[ni * kn + j]])
                                .collect();
                            let scl: Option<Vec<crate::halfin::HalfOp>> = sft.map(|_| {
                                (0..kn)
                                    .flat_map(|j| {
                                        [
                                            sa_prov[mi * nb + j / blk],
                                            sb_prov[((mi / 32) * n as usize + ni) * nb + j / blk],
                                        ]
                                    })
                                    .collect()
                            });
                            (ops, scl)
                        } else {
                            (vec![crate::halfin::HalfOp::Unknown; 2 * kn], None)
                        };
                        let acc = f32::from_bits(accs[ci][mi * n as usize + ni]);
                        // FP4: the units that can hold a nonzero product (the contract counts
                        // them; a structurally zero factor drops its product).
                        let zero_op = |j: usize| {
                            [ops[2 * j], ops[2 * j + 1]]
                                .into_iter()
                                .chain(scl.iter().flat_map(|s| [s[2 * j], s[2 * j + 1]]))
                                .any(|o| o == crate::halfin::HalfOp::Zero)
                        };
                        let live_units = (0..kn)
                            .step_by(16)
                            .filter(|&u| (u..u + 16).any(|j| !zero_op(j)))
                            .count();
                        // ue4m3 scales: the FP4 contract needs every live unit's scales nonzero.
                        let zero_scale = kn == 64
                            && (0..kn).step_by(16).any(|u| {
                                sc(u) == 0.0 && (u..u + 16).any(|j| a[j].v * b[j].v != 0.0)
                            });
                        // ASSUME(N-MMA-F8, N-MMA-BS): the contracts derived from the models
                        // (`mma_lp::lp_eps`, `fp4_eps`); the loose one outside them.
                        let eps_of = |np: usize, ac: bool| -> f64 {
                            let loose = crate::fp::NARROW_MMA_CONTRACT.eps_terms(np + ac as usize);
                            if kn == 64 {
                                let _ = np;
                                match sft.filter(|_| !zero_scale) {
                                    Some(st) => {
                                        crate::mma_lp::fp4_eps(st, live_units, ac).unwrap_or(loose)
                                    }
                                    None => loose,
                                }
                            } else {
                                crate::mma_lp::lp_eps(fa, fb, np, ac)
                            }
                        };
                        self.num_mma(
                            cta,
                            row_lane(mi),
                            col0 + ni,
                            enable,
                            acc,
                            (&prods, &ops),
                            f32::from_bits(r),
                            op.pc,
                            &eps_of,
                            scl.as_deref(),
                            if kn == 64 { 16 } else { 0 },
                        );
                    }
                }
            }
        }
        if let Some(what) = outside {
            return Err(format!(
                "{what} (NaN/Inf, or a subnormal f32 accumulator) is outside the calibrated tensor-core model: not modeled"
            ));
        }
        if unalloc {
            self.tmem_state_report(2, "tcgen05.mma", op.pc);
        }
        if uninit {
            let msg = format!(
                "tcgen05.mma at {} accumulates (enable_input_d = 1) into TMEM that was never written since allocation: the result includes garbage",
                self.k.describe(op.pc as usize)
            );
            self.report("violation", format!("mma-uninit-acc:{}", op.pc), msg);
        }
        if sf_uninit {
            let msg = format!(
                "tcgen05.mma at {} reads scale factors from TMEM that was never written since allocation",
                self.k.describe(op.pc as usize)
            );
            self.report("violation", format!("mma-uninit-sf:{}", op.pc), msg);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    // ASSUME(E-ASYNC-STEP, S-TMA): see ASSUMPTIONS.md
    /// S-PROXY: an async-proxy read (TMA store, tcgen05.mma operand) of shared memory last
    /// written by a thread's generic store is ordered only if that thread executed
    /// fence.proxy.async after the store and the fence happens-before the async operation.
    // ASSUME(S-PROXY): see ASSUMPTIONS.md
    pub fn proxy_check(&mut self, cta: u16, off: u32, n: usize, vc: &VC, pc: u32, what: &str) {
        for i in 0..n {
            let Some(w) = self.smem[cta as usize].sh.get(off as usize + i).w else {
                continue;
            };
            let a = w.actor as usize;
            if a >= self.threads.len() {
                continue; // written by an async unit (TMA): same proxy
            }
            // ASSUME(S-PROXY): a fence by any thread that observed the write (write HB fence)
            // and happens-before the async op covers it (probes/proxy_fence: the issuer's own
            // fence after the barrier suffices on GB300).
            let own = self.threads[a].fences.iter().any(|fv| {
                w.clock <= vc_get(fv, w.actor) && vc_get(fv, w.actor) <= vc_get(vc, w.actor)
            });
            let covered = own
                || self.threads.iter().enumerate().any(|(f, th)| {
                    let seen = vc_get(vc, f as u32);
                    th.fences
                        .iter()
                        .any(|fv| w.clock <= vc_get(fv, w.actor) && vc_get(fv, f as u32) <= seen)
                });
            if covered {
                continue;
            }
            let sig = format!("race:proxy:{}:{pc}", w.pc);
            let msg = format!(
                "proxy race on smem byte {:#x}: generic write by {} at {} is not ordered before the async-proxy read by {what} at {} (no fence.proxy.async after the write that happens-before the async op)",
                off as usize + i,
                self.actor_name(w.actor),
                self.k.describe(w.pc as usize),
                self.k.describe(pc as usize)
            );
            self.report("race", sig, msg);
            return;
        }
    }

    // ASSUME(E-ASYNC-STEP, S-TMA): see ASSUMPTIONS.md
    /// Swizzled TMA: the shared-memory start must be 128-byte aligned (GB300 faults with
    /// "misaligned address" otherwise, any span) and the inner box at most the span. The start
    /// need not be aligned to the swizzle repeat (8 rows x span): the swizzle phase comes from
    /// the address bits (`swz`), as multicast slices rely on. ASSUME(S-TMA)
    fn tma_swizzle_ok(&mut self, op: &AsyncOp, start: u32, m: &TmapDesc) -> bool {
        if m.swizzle == 0 || (start.is_multiple_of(128) && m.boxd[0] * m.esize <= m.swizzle) {
            return true;
        }
        let msg = format!(
            "{} with {}B swizzle: shared-memory address {start:#x} must be 128-byte aligned and the inner box at most {} bytes",
            self.actor_name(op.actor),
            m.swizzle,
            m.swizzle
        );
        self.report("fault", format!("tma-swizzle:{}", op.pc), msg);
        false
    }

    fn complete_tma_store(
        &mut self,
        op: &AsyncOp,
        cta: u16,
        src: u32,
        tmap: Arc<TmapDesc>,
        coords: Vec<i64>,
        group: u32,
    ) {
        let op = op.clone();
        if cta != self.threads[op.issuer as usize].cta {
            if let Err(e) = self.check_alive(cta, op.actor) {
                self.report("violation", format!("cluster-life:{}", op.pc), e);
                return;
            }
        }
        let m = tmap;
        if !self.tma_swizzle_ok(&op, src, &m) {
            return;
        }
        let total: u64 = m.boxd.iter().map(|&b| b as u64).product();
        let es = m.esize as usize;
        let (alloc, boff) = self.tmap_alloc(&m);
        let mut idx = vec![0u64; m.dims.len()];
        for e in 0..total {
            let mut rem = e;
            for (d, b) in m.boxd.iter().enumerate() {
                idx[d] = rem % *b as u64;
                rem /= *b as u64;
            }
            let mut inb = true;
            let mut goff = boff;
            for d in 0..m.dims.len() {
                let g = coords[d] + idx[d] as i64;
                if g < 0 || g as u64 >= m.dims[d] {
                    inb = false;
                }
                // Wraps mod 2^64: trtllm-gen's dynBatch output map has dims 2^31 and a
                // stride 2^35 - 256 that coordinate 2^30 cancels (docs/rung3b.md). ASSUME(S-TMA)
                goff = goff.wrapping_add(if d == 0 {
                    (g.max(0) as u64).wrapping_mul(es as u64)
                } else {
                    (g.max(0) as u64).wrapping_mul(m.strides[d - 1])
                });
            }
            let soff = swz(
                src + tma_box_off(e, m.boxd[0], m.esize, m.swizzle),
                m.swizzle,
            );
            if soff as usize + es > self.launch.g(cta as u32).smem_bytes as usize {
                self.report(
                    "fault",
                    format!("tma-oob:{}", op.pc),
                    format!("{} reads past shared memory", self.actor_name(op.actor)),
                );
                return;
            }
            self.proxy_check(cta, soff, es, &op.vc, op.pc, "the TMA store");
            let data = self.access(
                op.actor,
                &op.vc,
                op.pc,
                Loc::Shared { cta, off: soff },
                es,
                None,
            );
            if inb && goff as usize + es > self.launch.allocs[alloc].init.len() {
                self.report(
                    "fault",
                    format!("tma-gmem-oob:{}", op.pc),
                    format!(
                        "{} writes past the end of its tensor map's allocation (byte {goff:#x})",
                        self.actor_name(op.actor)
                    ),
                );
                return;
            }
            if inb {
                // Out-of-bounds box elements are not written.
                self.access(
                    op.actor,
                    &op.vc,
                    op.pc,
                    Loc::Global { alloc, off: goff },
                    es,
                    Some(&data),
                );
                if es == 4 {
                    self.num_copy(
                        Loc::Shared { cta, off: soff },
                        Loc::Global { alloc, off: goff },
                    );
                }
                self.half_copy(
                    Some(Loc::Shared { cta, off: soff }),
                    Loc::Global { alloc, off: goff },
                    es,
                    &data,
                );
            }
        }
        self.rec_rel(sync_key(3, op.issuer as u64, group as u64, 0));
        let iss = op.issuer as usize;
        self.threads[iss]
            .make_mut()
            .bulk_done
            .push((group, op.vc.clone()));
    }

    #[allow(clippy::too_many_arguments)]
    /// `cp.async.mbarrier.arrive`'s asynchronous half: the tracked cp.async writes land
    /// (race-checked as the op's actor), then one transaction unit completes or one arrival
    /// happens on the mbarrier.
    fn complete_cp_arrive(
        &mut self,
        op: &AsyncOp,
        cta: u16,
        mbar: u32,
        writes: Vec<CpPending>,
        tx: bool,
    ) {
        for c in writes {
            let loc = Loc::Shared { cta, off: c.off };
            self.access(op.actor, &op.vc, c.pc, loc, c.data.len(), Some(&c.data));
            self.sync_clear(loc);
            self.num_cp_async(c.src, cta, c.off, &c.data);
        }
        self.rec(0, cta as u32, mbar as u64, mbar as u64 + 8, false);
        self.mbar_materialize(cta, mbar);
        if self.mbars.contains_key(&(cta, mbar)) {
            let _ = self.mbar_atomic(op.actor, &op.vc, &op.vis, op.pc, cta, mbar);
        }
        let Some(mb) = self.mbars.get_mut(&(cta, mbar)) else {
            let msg = format!(
                "cp.async.mbarrier.arrive at {} on an uninitialized mbarrier at {mbar:#x}",
                self.k.describe(op.pc as usize)
            );
            self.report("violation", format!("cparrive-uninit:{}", op.pc), msg);
            return;
        };
        if tx {
            mb.tx -= 1;
        } else if mb.pending == 0 {
            let msg = format!(
                "cp.async.mbarrier.arrive at {}: more arrivals than the expected count {}",
                self.k.describe(op.pc as usize),
                mb.expected
            );
            self.report("fault", format!("cparrive-overflow:{}", op.pc), msg);
            return;
        } else {
            mb.pending -= 1;
        }
        vc_join(&mut mb.cur, &op.vc);
        self.mbar_maybe_complete(cta, mbar);
    }

    #[allow(clippy::too_many_arguments)]
    fn complete_tma(
        &mut self,
        op: &AsyncOp,
        cta: u16,
        dst: u32,
        mbar_cta: u16,
        mbar: u32,
        tmap: Arc<TmapDesc>,
        coords: Vec<i64>,
    ) {
        let op = op.clone();
        if cta != self.threads[op.issuer as usize].cta {
            if let Err(e) = self.check_alive(cta, op.actor) {
                self.report("violation", format!("cluster-life:{}", op.pc), e);
                return;
            }
        }
        let m = tmap;
        if !self.tma_swizzle_ok(&op, dst, &m) {
            return;
        }
        // `tile::gather4` (2-D map, 5 coordinates): rows coords[1..5] at column coords[0], each
        // box[0] wide, land as one 4-row box would. ASSUME(S-TMA-GATHER4)
        let gather4 = coords.len() == m.dims.len() + 3;
        if gather4 && (m.dims.len() != 2 || m.boxd[1] != 1) {
            self.report(
                "violation",
                format!("tma-gather4:{}", op.pc),
                format!(
                    "{}: tile::gather4 needs a 2-D tensor map with box height 1 (box {:?})",
                    self.actor_name(op.actor),
                    m.boxd
                ),
            );
            return;
        }
        // GB300 raises an illegal instruction for a gather4 column whose byte offset is not
        // a multiple of 16 (fixtures/s_gather4 `misaligned`). ASSUME(S-TMA-GATHER4)
        if gather4 && (coords[0] * m.esize as i64).rem_euclid(16) != 0 {
            self.report(
                "fault",
                format!("tma-gather4-align:{}", op.pc),
                format!(
                    "{}: tile::gather4 column {} is not 16-byte aligned ({} B elements): illegal instruction on GB300",
                    self.actor_name(op.actor),
                    coords[0],
                    m.esize
                ),
            );
            return;
        }
        // Box walk: innermost dimension contiguous in smem, no swizzle.
        let total: u64 = if gather4 {
            4 * m.boxd[0] as u64
        } else {
            m.boxd.iter().map(|&b| b as u64).product()
        };
        let es = m.esize as usize;
        let (alloc, boff) = self.tmap_alloc(&m);
        // Element addresses first; then the accesses, merged into contiguous runs: one
        // shadow update per run keeps whole shadow pages uniform (`Pages::for_run`). All bytes
        // are accessed by the same actor at the same clock, so grouping changes nothing else.
        let mut idx = vec![0u64; m.dims.len()];
        let mut elems: Vec<(u32, Option<u64>)> = Vec::with_capacity(total as usize);
        for e in 0..total {
            let mut rem = e;
            for (d, b) in m.boxd.iter().enumerate() {
                idx[d] = rem % *b as u64;
                rem /= *b as u64;
            }
            let mut inb = true;
            let mut goff = boff;
            for d in 0..m.dims.len() {
                let g = if gather4 && d == 1 {
                    coords[1 + (e / m.boxd[0] as u64) as usize]
                } else {
                    coords[d] + idx[d] as i64
                };
                if g < 0 || g as u64 >= m.dims[d] {
                    inb = false;
                }
                // Wraps mod 2^64: trtllm-gen's dynBatch output map has dims 2^31 and a
                // stride 2^35 - 256 that coordinate 2^30 cancels (docs/rung3b.md). ASSUME(S-TMA)
                goff = goff.wrapping_add(if d == 0 {
                    (g.max(0) as u64).wrapping_mul(es as u64)
                } else {
                    (g.max(0) as u64).wrapping_mul(m.strides[d - 1])
                });
            }
            let soff = swz(
                dst + tma_box_off(e, m.boxd[0], m.esize, m.swizzle),
                m.swizzle,
            );
            if soff as usize + es > self.launch.g(cta as u32).smem_bytes as usize {
                self.report(
                    "fault",
                    format!("tma-oob:{}", op.pc),
                    format!("{} writes past shared memory", self.actor_name(op.actor)),
                );
                return;
            }
            elems.push((soff, inb.then_some(goff)));
        }
        let runs = |mut offs: Vec<u64>| -> Vec<(u64, usize)> {
            offs.sort_unstable();
            offs.dedup();
            let mut r: Vec<(u64, usize)> = Vec::new();
            for o in offs {
                match r.last_mut() {
                    Some((s, n)) if *s + *n as u64 == o => *n += es,
                    _ => r.push((o, es)),
                }
            }
            r
        };
        let size = self.launch.allocs[alloc].init.len() as u64;
        if elems
            .iter()
            .any(|e| e.1.is_some_and(|g| g + es as u64 > size))
        {
            self.report(
                "fault",
                format!("tma-gmem-oob:{}", op.pc),
                format!(
                    "{} reads past the end of its tensor map's allocation",
                    self.actor_name(op.actor)
                ),
            );
            return;
        }
        let mut src: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        for (off, n) in runs(elems.iter().filter_map(|e| e.1).collect()) {
            let d = self.access(op.actor, &op.vc, op.pc, Loc::Global { alloc, off }, n, None);
            src.insert(off, d);
        }
        let fetch = |goff: u64| -> &[u8] {
            let (s, d) = src.range(..=goff).next_back().expect("read run");
            let o = (goff - s) as usize;
            &d[o..o + es]
        };
        let mut img: BTreeMap<u32, &[u8]> = BTreeMap::new();
        let zero = vec![0u8; es];
        for (soff, g) in &elems {
            img.insert(*soff, g.map_or(&zero[..], fetch));
        }
        for (off, n) in runs(elems.iter().map(|e| e.0 as u64).collect()) {
            let mut data = Vec::with_capacity(n);
            for o in (off..off + n as u64).step_by(es) {
                data.extend_from_slice(img[&(o as u32)]);
            }
            self.access(
                op.actor,
                &op.vc,
                op.pc,
                Loc::Shared {
                    cta,
                    off: off as u32,
                },
                n,
                Some(&data),
            );
        }
        // Origins are tracked for 2-byte (bf16) inputs, the MMA operands modeled so far.
        for (soff, g) in elems {
            if self.num.is_some() {
                let data = img[&soff].to_vec();
                self.half_copy(
                    g.map(|off| Loc::Global { alloc, off }),
                    Loc::Shared { cta, off: soff },
                    es,
                    &data,
                );
            }
            self.num_tma_elem(
                cta,
                soff,
                g.filter(|_| es == 2).map(|g| (alloc, g / es as u64)),
            );
        }
        let bytes = (total as usize * es) as i64;
        let cta = mbar_cta;
        if cta != self.threads[op.issuer as usize].cta {
            if let Err(e) = self.check_alive(cta, op.actor) {
                self.report("violation", format!("cluster-life:{}", op.pc), e);
                return;
            }
        }
        self.rec(0, cta as u32, mbar as u64, mbar as u64 + 8, false);
        self.mbar_materialize(cta, mbar);
        // complete_tx is an mbarrier operation: its init must happen-before the issue.
        if self.mbars.contains_key(&(cta, mbar)) {
            let _ = self.mbar_atomic(op.actor, &op.vc, &op.vis, op.pc, cta, mbar);
        }
        match self.mbars.get_mut(&(cta, mbar)) {
            None => {
                let msg = format!(
                    "{} signals complete_tx on an uninitialized mbarrier at {mbar:#x}",
                    self.actor_name(op.actor)
                );
                self.report("violation", format!("tma-uninit:{}", op.pc), msg);
            }
            Some(mb) => {
                mb.tx -= bytes;
                vc_join(&mut mb.cur, &op.vc);
                self.mbar_maybe_complete(cta, mbar);
            }
        }
    }

    /// `HMMA.1688.F32.TF32` (`mma.sync.m16n8k8` tf32, fragments in `Op::Hmma`'s doc): each
    /// operand word is read as tf32 (`fp::TF32_IN`) and every output is one step of
    /// `fp::TMMA_MODEL` over its 8 products and accumulator. Results are not tracked by the
    /// numerics (f32 operand provenance through MMA is not modeled). ASSUME(N-HMMA)
    fn hmma_tf32(
        &mut self,
        active: &[usize],
        d: Reg,
        a: &[Val],
        b: &[Val],
        c: &[Val],
        pc: u32,
    ) -> Result<(), String> {
        let mut am = [[0f32; 8]; 16];
        let mut bm = [[0f32; 8]; 8];
        let mut cm = [[0f32; 8]; 16];
        let mut lanes = [0usize; 32];
        let tf = |x: u32| f32::from_bits(fp::tf32_in(x));
        for &t in active {
            let l = self.threads[t].lane as usize;
            lanes[l] = t;
            let (g, q) = (l / 4, l % 4);
            for (j, v) in a.iter().enumerate() {
                am[g + 8 * (j & 1)][q + 4 * (j >> 1)] = tf(self.rd(t, v)? as u32);
            }
            for (j, v) in b.iter().enumerate() {
                bm[q + 4 * j][g] = tf(self.rd(t, v)? as u32);
            }
            for (j, v) in c.iter().enumerate() {
                cm[g + 8 * (j >> 1)][2 * q + (j & 1)] = f32::from_bits(self.rd(t, v)? as u32);
            }
        }
        let odd = |x: f32| !x.is_finite() || (x != 0.0 && !x.is_normal());
        if am
            .iter()
            .flatten()
            .chain(bm.iter().flatten())
            .any(|&x| odd(x))
        {
            return Err(
                "HMMA input (subnormal/Inf/NaN) is outside the calibrated model: not modeled"
                    .into(),
            );
        }
        if cm.iter().flatten().any(|&x| odd(x)) {
            return Err(
                "HMMA accumulator (subnormal/Inf/NaN) is outside the calibrated model: not modeled"
                    .into(),
            );
        }
        let model = fp::TMMA_MODEL;
        // Numerics: each operand element's factor (N-TF32) and the accumulators' nodes, read
        // before any result is written (d may be a, b or c).
        use crate::halfin::HalfOp;
        let mut ap = [[HalfOp::Unknown; 8]; 16];
        let mut bp = [[HalfOp::Unknown; 8]; 8];
        let mut cn: [[Option<u32>; 8]; 16] = [[None; 8]; 16];
        if self.num.is_some() {
            for &t in active {
                let l = self.threads[t].lane as usize;
                let (g, q) = (l / 4, l % 4);
                for (j, v) in a.iter().enumerate() {
                    let w = self.rd(t, v)? as u32;
                    ap[g + 8 * (j & 1)][q + 4 * (j >> 1)] = self.tf32_factor(t, v, w, pc);
                }
                for (j, v) in b.iter().enumerate() {
                    let w = self.rd(t, v)? as u32;
                    bp[q + 4 * j][g] = self.tf32_factor(t, v, w, pc);
                }
                for (j, v) in c.iter().enumerate() {
                    let (row, col) = (g + 8 * (j >> 1), 2 * q + (j & 1));
                    cn[row][col] = self.num_acc_node(t, v, cm[row][col].to_bits(), pc);
                }
            }
        }
        let mut out = Vec::with_capacity(128);
        for (l, &t) in lanes.iter().enumerate() {
            let (g, q) = (l / 4, l % 4);
            for j in 0..4usize {
                let (row, col) = (g + 8 * (j >> 1), 2 * q + (j & 1));
                let terms: Vec<fp::Term> = (0..8)
                    .map(|k| {
                        fp::Term::prod(am[row][k] as f64, bm[k][col] as f64, model.exp_from_inputs)
                    })
                    .collect();
                let r = model.apply_terms(cm[row][col], &terms);
                out.push((t, j, r, row, col, terms));
            }
        }
        for (t, j, r, row, col, terms) in out {
            let dr = Reg {
                file: d.file,
                idx: d.idx + j as u16,
            };
            self.wr(t, dr, r.to_bits() as u64, 1);
            if self.num.is_some() {
                // ASSUME(N-HMMA): one step of the calibrated TF32 model per instruction.
                let prods: Vec<f64> = terms.iter().map(|x| x.v).collect();
                let ops: Vec<HalfOp> = (0..8).flat_map(|k| [ap[row][k], bp[k][col]]).collect();
                self.num_hmma(
                    t,
                    dr,
                    (cn[row][col], cm[row][col]),
                    (&prods, &ops),
                    r,
                    pc,
                    true,
                );
            }
        }
        Ok(())
    }
}

/// Lets `Hash` values feed a `&mut dyn Hasher`.
struct HashAdapter<'a>(&'a mut dyn std::hash::Hasher);
impl std::hash::Hasher for HashAdapter<'_> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    fn write(&mut self, b: &[u8]) {
        self.0.write(b)
    }
}

/// ASSUME(L-ROSETTA): `R2UR d, a` of a value that differs across lanes in the bits `diff` is
/// harmless when, in straight-line code, the first instruction reading `d` is a logical right
/// shift of `d` into `d` by an immediate that drops all of `diff` (cuBLAS: `R2UR UR6, R2`
/// with R2 = %tid.x, then `USHF.R.U32 UR6, UR6, 0x5, URZ` = the warp index): whichever lane
/// R2UR takes, the result is the same.
fn shifted_out(code: &[Inst], pc: usize, d: Reg, diff: u64) -> bool {
    for i in &code[pc + 1..] {
        let reads = i.op.src_regs().contains(&d);
        if !reads {
            if matches!(
                i.op,
                Op::Bra { .. }
                    | Op::Exit
                    | Op::Call { .. }
                    | Op::Ret { .. }
                    | Op::Brx { .. }
                    | Op::Trap { .. }
            ) {
                return false;
            }
            if i.op.dst_regs().contains(&d) {
                return true; // overwritten unread
            }
            continue;
        }
        return matches!(
            &i.op,
            Op::Shf {
                d: dd,
                lo: Val::Reg(l, 1),
                sh: Val::Imm(s),
                hi: Val::Imm(0),
                right: true,
                signed: false,
                high: false,
            } if *dd == d && *l == d && *s < 32 && diff >> *s == 0
        ) && i.guard.is_always();
    }
    false
}

/// A float operand's `|x|` then `-x` modifiers (sign-bit operations, NaN payloads kept).
fn fmod(u: u32, neg: bool, abs: bool) -> u32 {
    let u = if abs { u & 0x7fff_ffff } else { u };
    if neg { u ^ 0x8000_0000 } else { u }
}

/// Register-only operations whose result is undefined when a source is (S-INIT): computed,
/// not reported, unless a destination is a predicate (control flow) or not a data register.
fn propagates_undef(op: &Op) -> bool {
    matches!(
        op,
        Op::Mov { .. }
            | Op::Int { .. }
            | Op::MulWide { .. }
            | Op::Imad { .. }
            | Op::Iadd3 { .. }
            | Op::CvtInt { .. }
            | Op::Flo { .. }
            | Op::Brev { .. }
            | Op::Lea { .. }
            | Op::LeaHi { .. }
            | Op::Shf { .. }
            | Op::Lop3 { .. }
            | Op::Prmt { .. }
            | Op::Bfe { .. }
            | Op::I2F { .. }
            | Op::F2I { .. }
            | Op::Fadd { .. }
            | Op::Ffma { .. }
            | Op::Ffma2 { .. }
            | Op::Fmnmx { .. }
            | Op::Hfma2 { .. }
            | Op::CvtF32Bf16 { .. }
            | Op::F2fpBf16 { .. }
            | Op::Pack64 { .. }
            | Op::Unpack64 { .. }
            | Op::AddMinMax { .. }
            | Op::IntMnmx3 { .. }
            | Op::Mufu { .. }
            | Op::DivFull { .. }
            | Op::Dp2a { .. }
            | Op::F2F { .. }
            | Op::CvtF32F16 { .. }
            | Op::F2fpF16 { .. }
            | Op::F2fF16 { .. }
            | Op::CvtPack { .. }
            | Op::CvtRsX4 { .. }
            | Op::CvtUnpack { .. }
            | Op::Dfma { .. }
            | Op::DivSqrt64 { .. }
    )
}

/// E-POLL-ANY: the key of a location in `VipLearn` and `Machine::vipr`: (allocation, offset)
/// for global memory, (bit 31 | CTA, offset) for shared memory.
pub fn vip_where(loc: Loc) -> Option<(u32, u64)> {
    match loc {
        Loc::Global { alloc, off } => Some((alloc as u32, off)),
        Loc::Shared { cta, off } => Some((0x8000_0000 | cta as u32, off as u64)),
        Loc::Param { .. } => None,
    }
}
