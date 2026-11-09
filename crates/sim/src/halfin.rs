/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! bf16 / f16 input leaves for the numerics shadow (docs/equiv-elementwise.md, "bf16 and f16
//! inputs").
//!
//! The numerics track f32 input words as `Input` leaves (N-LEAF). A bf16 or f16 element is
//! exactly representable in f32, but kernels load it packed (two per 32-bit word, or one
//! zero-extended by `LDG.E.U16`) and widen it with integer instructions: `SHF.L.U32 d, r, 0x10`,
//! `IMAD.U32 d, r, 0x10000, RZ`, `LOP3.LUT d, r, 0xffff0000, RZ, 0xc0`, `PRMT` byte moves, or
//! `HADD2.F32 d, -RZ, r.H0_H0` for f16. This module follows the provenance of each byte of the
//! registers such a load writes through those instructions (byte moves and shifts by whole
//! bytes, zero bytes from constants), and gives a register an `Input` leaf (allocation, element
//! index) when it holds exactly one element widened to f32: bf16 bytes [0, 0, lo, hi], or the
//! f16 -> f32 conversion of a half holding one element. Every step is a bit-exact function of
//! the bytes (no rounding), so the leaf's value is the element's for every input.
//! ASSUME(N-HALF-LEAF): see ASSUMPTIONS.md

use std::collections::BTreeMap;
use std::collections::HashMap;

use ptxv_ir::*;

use crate::exec::Loc;
use crate::machine::Machine;
use crate::num::NodeKind;

/// Where a register or memory byte comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Src {
    Zero,
    /// Byte `b` (0 = low) of element `elem` of input allocation `alloc`.
    E {
        alloc: usize,
        elem: u64,
        b: u8,
    },
    /// Byte `b` (0 = low) of the bf16 encoding of numerics node `node`, whose hardware value is
    /// exactly that bf16 value widened to f32 (a `Round` node, or an `Input` leaf).
    N {
        node: u32,
        b: u8,
    },
    /// Input byte `elem` of `alloc` (a byte-code input) with bit 7 cleared (`LOP3 x & 0x7f`):
    /// read as e4m3 it is the byte's ue4m3 value for every byte (`half_unpack`).
    E7 {
        alloc: usize,
        elem: u64,
    },
}

pub(crate) type Bytes = [Option<Src>; 4];

/// Half-precision input elements kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    Bf16,
    F16,
    /// Byte-granular codes (FP8 operands, ue8m0 scale factors of the narrow MMA kinds): `E`
    /// elements are bytes; their value is decoded per use (`byte_leaf`).
    Byte,
}

/// What a 16-bit operand half holds, for a tensor-core product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HalfOp {
    /// Provably +0 or -0 for every input (zero bytes): the product is exactly zero.
    Zero,
    /// A numerics node whose value is the bf16 element.
    Node(u32),
    /// Unknown provenance.
    Unknown,
}

#[derive(Default, Debug, Clone)]
pub struct HalfState {
    /// Input allocations of bf16 / f16 elements (equiv sets them, like `f32_inputs`).
    pub inputs: BTreeMap<usize, Half>,
    /// Track 16-bit provenance even without half-precision inputs (bf16 outputs, rounded
    /// operands of tensor-core steps).
    pub track: bool,
    /// Per (thread, register) of the R file: provenance of its 4 bytes.
    regs: HashMap<(u32, u16), Bytes>,
    /// Registers given provenance by the current instruction (loads, moves, conversions).
    set_now: Vec<(u32, u16)>,
    /// Memory bytes: (space, allocation / CTA / thread, byte offset) -> provenance. Spaces:
    /// 0 shared (CTA), 1 global (allocation), 3 TMEM (CTA; 4 * word index + byte), 4 local
    /// (thread). A byte without an entry has no provenance, except unwritten bytes of a
    /// half-precision input (`gwritten`).
    mem: HashMap<(u8, u64, u64), Src>,
    /// Global bytes written during the run.
    gwritten: std::collections::HashSet<(u64, u64)>,
    /// `Input` leaves of half-precision input elements, made on first use.
    input_nodes: HashMap<(usize, u64), u32>,
    /// Output allocations of bf16 elements (equiv reads one value per 16-bit element).
    pub bf16_outputs: std::collections::BTreeSet<usize>,
    /// ... and of f16 elements.
    pub f16_outputs: std::collections::BTreeSet<usize>,
    /// Nodes whose `N` bytes are an f16 encoding (f16 roundings and constants, f16 input
    /// leaves); every other `N` node's bytes are bf16.
    pub f16_nodes: std::collections::HashSet<u32>,
}

impl HalfState {
    pub(crate) fn forget_register(&mut self, thread: usize, reg: Reg) {
        if reg.file == File::R {
            self.regs.remove(&(thread as u32, reg.idx));
        }
    }

    pub fn is_on(&self) -> bool {
        self.track || !self.inputs.is_empty()
    }
}

pub(crate) fn hkey(loc: Loc) -> (u8, u64, u64) {
    match loc {
        Loc::Shared { cta, off } => (0, cta as u64, off as u64),
        Loc::Global { alloc, off } => (1, alloc as u64, off),
        Loc::Param { grid, off } => (2, grid as u64, off as u64),
    }
}

/// The TMEM key of byte `b` of word (lane, col) of `cta`.
pub(crate) fn tkey(cta: u16, lane: usize, col: usize, b: u64) -> (u8, u64, u64) {
    (
        3,
        cta as u64,
        4 * crate::machine::tmem_idx(lane, col) as u64 + b,
    )
}

impl Machine {
    /// Provenance of `n` memory bytes at `loc` (`data`: the bytes read, for the unwritten-input
    /// fallback).
    pub(crate) fn half_mem(&self, loc: Loc, n: usize, data: Option<&[u8]>) -> Vec<Option<Src>> {
        let Some(ns) = self.num.as_ref() else {
            return vec![None; n];
        };
        let (sp, a, off) = hkey(loc);
        (0..n)
            .map(|i| {
                let o = off + i as u64;
                if let Some(&s) = ns.half.mem.get(&(sp, a, o)) {
                    return Some(s);
                }
                if sp != 1 || ns.half.gwritten.contains(&(a, o)) {
                    return None;
                }
                let alloc = a as usize;
                let kind = *ns.half.inputs.get(&alloc)?;
                let init = *self.launch.allocs.get(alloc)?.init.get(o as usize)?;
                if data.is_some_and(|d| d.get(i) != Some(&init)) {
                    return None;
                }
                Some(if kind == Half::Byte {
                    Src::E {
                        alloc,
                        elem: o,
                        b: 0,
                    }
                } else {
                    Src::E {
                        alloc,
                        elem: o / 2,
                        b: (o % 2) as u8,
                    }
                })
            })
            .collect()
    }

    /// Set the provenance of `n` memory bytes at key `k` (None: none).
    pub(crate) fn half_mem_set(&mut self, k: (u8, u64, u64), srcs: &[Option<Src>]) {
        let Some(ns) = self.num.as_mut() else { return };
        if !ns.half.is_on() {
            return;
        }
        for (i, s) in srcs.iter().enumerate() {
            let key = (k.0, k.1, k.2 + i as u64);
            if k.0 == 1 {
                ns.half.gwritten.insert((k.1, key.2));
            }
            match s {
                Some(s) => {
                    ns.half.mem.insert(key, *s);
                }
                None => {
                    ns.half.mem.remove(&key);
                }
            }
        }
    }

    /// Byte provenance of an operand (R register or immediate).
    pub(crate) fn half_val(&self, t: usize, v: &Val) -> Option<Bytes> {
        let ns = self.num.as_ref()?;
        match v {
            Val::Imm(c) => Some(imm_bytes(*c as u32)),
            Val::Reg(r, _) if r.file == File::R => ns.half.regs.get(&(t as u32, r.idx)).copied(),
            _ => None,
        }
    }

    /// Register `r` of thread `t` now holds bytes `b` (marked as set by this instruction).
    pub(crate) fn half_set_reg(&mut self, t: usize, r: Reg, b: Option<Bytes>) {
        let Some(ns) = self.num.as_mut() else { return };
        if !ns.half.is_on() || r.file != File::R {
            return;
        }
        match b {
            Some(b) if b.iter().any(|x| x.is_some()) => {
                ns.half.regs.insert((t as u32, r.idx), b);
            }
            _ => {
                ns.half.regs.remove(&(t as u32, r.idx));
            }
        }
        ns.half.set_now.push((t as u32, r.idx));
    }

    /// A load into `d` (bytes, vector count) from `loc` read `data`: each register gets the
    /// provenance of the bytes it holds (16-bit and 8-bit loads: zero-extended unless `sext`).
    pub(crate) fn half_on_ld(
        &mut self,
        t: usize,
        d: Reg,
        loc: Loc,
        (bytes, nv, sext): (u8, u8, bool),
        data: &[u8],
    ) {
        let Some(ns) = self.num.as_ref() else { return };
        if !ns.half.is_on() || d.file != File::R {
            return;
        }
        let n = bytes as usize;
        let src = self.half_mem(loc, n, Some(data));
        let reg = |i: usize| Reg {
            file: d.file,
            idx: d.idx + i as u16,
        };
        let regs: Vec<(Reg, Bytes)> = if nv > 1 && n / nv as usize != 4 {
            // One element per register (PTX ld.vN of 8/16-bit elements).
            let es = n / nv as usize;
            (0..nv as usize)
                .map(|i| {
                    let mut b: Bytes = [None; 4];
                    for j in 0..4 {
                        b[j] = if j < es {
                            src[i * es + j]
                        } else if sext {
                            None
                        } else {
                            Some(Src::Zero)
                        };
                    }
                    (reg(i), b)
                })
                .collect()
        } else if n >= 4 {
            (0..n / 4)
                .map(|w| {
                    (
                        reg(w),
                        [src[4 * w], src[4 * w + 1], src[4 * w + 2], src[4 * w + 3]],
                    )
                })
                .collect()
        } else {
            let mut b: Bytes = [None; 4];
            for (j, x) in b.iter_mut().enumerate() {
                *x = if j < n {
                    src[j]
                } else if sext {
                    None
                } else {
                    Some(Src::Zero)
                };
            }
            vec![(d, b)]
        };
        for (r, b) in regs {
            self.half_set_reg(t, r, Some(b));
        }
    }

    /// A store of `v` (`bytes`) to `loc`: the memory bytes take the register bytes' provenance.
    pub(crate) fn half_on_st(&mut self, t: usize, v: &Val, loc: Loc, bytes: u8) {
        let Some(ns) = self.num.as_ref() else { return };
        if !ns.half.is_on() {
            return;
        }
        let n = bytes as usize;
        let mut srcs: Vec<Option<Src>> = Vec::with_capacity(n);
        match v {
            Val::Reg(r, _) if r.file == File::R => {
                for w in 0..n.div_ceil(4) {
                    let b = self
                        .half_val(
                            t,
                            &Val::Reg(
                                Reg {
                                    file: r.file,
                                    idx: r.idx + w as u16,
                                },
                                1,
                            ),
                        )
                        .unwrap_or([None; 4]);
                    srcs.extend_from_slice(&b[..(n - 4 * w).min(4)]);
                }
            }
            Val::Imm(c) => {
                for i in 0..n {
                    srcs.push(((*c >> (8 * (i % 8))) & 0xff == 0).then_some(Src::Zero));
                }
            }
            _ => srcs = vec![None; n],
        }
        self.half_mem_set(hkey(loc), &srcs);
    }

    /// Memory-to-memory copy of `n` bytes (cp.async, TMA store): provenance moves along.
    pub(crate) fn half_copy(&mut self, from: Option<Loc>, to: Loc, n: usize, data: &[u8]) {
        let Some(ns) = self.num.as_ref() else { return };
        if !ns.half.is_on() {
            return;
        }
        let srcs = match from {
            Some(f) => self.half_mem(f, n, Some(data)),
            // Zero fill.
            None => vec![Some(Src::Zero); n],
        };
        self.half_mem_set(hkey(to), &srcs);
    }

    /// LDSM: register `r` of thread `t` gets its two halves from shared memory at `locs`.
    pub(crate) fn half_ldsm(&mut self, t: usize, r: Reg, locs: [Option<Loc>; 2]) {
        if !self.num.as_ref().is_some_and(|n| n.half.is_on()) {
            return;
        }
        let mut b: Bytes = [None; 4];
        for (h, l) in locs.iter().enumerate() {
            if let Some(l) = l {
                let s = self.half_mem(*l, 2, None);
                b[2 * h] = s[0];
                b[2 * h + 1] = s[1];
            }
        }
        self.half_set_reg(t, r, Some(b));
    }

    /// LDSM.8.MT1616 (`ldmatrix.m16n16.trans.b8`): byte b of register `r` of thread `t` comes
    /// from shared memory at `locs[b]`.
    pub(crate) fn half_ldsm_bytes(&mut self, t: usize, r: Reg, locs: [Option<Loc>; 4]) {
        if !self.num.as_ref().is_some_and(|n| n.half.is_on()) {
            return;
        }
        let mut b: Bytes = [None; 4];
        for (i, l) in locs.iter().enumerate() {
            if let Some(l) = l {
                b[i] = self.half_mem(*l, 1, None)[0];
            }
        }
        self.half_set_reg(t, r, Some(b));
    }

    /// STSM .trans: half `h` of register `v` of thread `t` goes to shared memory at `loc`.
    pub(crate) fn half_stsm_half(&mut self, t: usize, v: &Val, h: usize, loc: Loc) {
        if !self.num.as_ref().is_some_and(|n| n.half.is_on()) {
            return;
        }
        let b = self.half_val(t, v).unwrap_or([None; 4]);
        self.half_mem_set(hkey(loc), &b[2 * h..2 * h + 2]);
    }

    /// Local memory word `addr` of thread `t`: (store) bytes of `v`.
    pub(crate) fn half_local_st(&mut self, t: usize, v: &Val, addr: u32) {
        if !self.num.as_ref().is_some_and(|n| n.half.is_on()) {
            return;
        }
        let b = self.half_val(t, v).unwrap_or([None; 4]);
        self.half_mem_set((4, t as u64, addr as u64), &b);
    }

    /// (load) register `d` of thread `t` gets local word `addr`'s bytes.
    pub(crate) fn half_local_ld(&mut self, t: usize, d: Reg, addr: u32) {
        let Some(ns) = self.num.as_ref() else { return };
        if !ns.half.is_on() {
            return;
        }
        let mut b: Bytes = [None; 4];
        for (i, x) in b.iter_mut().enumerate() {
            *x = ns
                .half
                .mem
                .get(&(4, t as u64, addr as u64 + i as u64))
                .copied();
        }
        self.half_set_reg(t, d, Some(b));
    }

    /// The node a bf16 (`f16`: f16) output element (allocation, element index) holds: a
    /// rounded value (its `Round` node) or an input element's leaf of that format; None: no
    /// provenance (or a constant).
    pub fn half_out_node(&self, alloc: usize, elem: u64, f16: bool) -> Option<u32> {
        let ns = self.num.as_ref()?;
        let b = self.half_mem(
            Loc::Global {
                alloc,
                off: 2 * elem,
            },
            2,
            None,
        );
        let want = if f16 { Half::F16 } else { Half::Bf16 };
        match (b[0], b[1]) {
            (Some(Src::N { node, b: 0 }), Some(Src::N { node: n2, b: 1 }))
                if node == n2 && ns.half.f16_nodes.contains(&node) == f16 =>
            {
                Some(node)
            }
            (
                Some(Src::E {
                    alloc: a,
                    elem: e,
                    b: 0,
                }),
                Some(Src::E {
                    alloc: a2,
                    elem: e2,
                    b: 1,
                }),
            ) if a == a2 && e == e2 && ns.half.inputs.get(&a) == Some(&want) => {
                ns.half.input_nodes.get(&(a, e)).copied()
            }
            _ => None,
        }
    }

    /// After the run: an `Input` leaf for every bf16 output element that holds an input
    /// element moved byte for byte (a copy kernel), so `half_out_node` finds it.
    pub fn half_out_leaves(&mut self) {
        let Some(ns) = self.num.as_ref() else { return };
        let outs: Vec<usize> = ns
            .half
            .bf16_outputs
            .iter()
            .chain(&ns.half.f16_outputs)
            .copied()
            .collect();
        for a in outs {
            let n = self.gmem[a].bytes.len() / 2;
            for e in 0..n as u64 {
                let b = self.half_mem(
                    Loc::Global {
                        alloc: a,
                        off: 2 * e,
                    },
                    2,
                    None,
                );
                if let (Some(Src::E { alloc, elem, b: 0 }), Some(Src::E { b: 1, .. })) =
                    (b[0], b[1])
                {
                    if b[1] == Some(Src::E { alloc, elem, b: 1 }) {
                        self.half_input_node(alloc, elem);
                    }
                }
            }
        }
    }

    /// Bytes of TMEM word (lane, col) of `cta`.
    pub(crate) fn half_tmem(&self, cta: u16, lane: usize, col: usize) -> Bytes {
        let mut b: Bytes = [None; 4];
        if let Some(ns) = self.num.as_ref() {
            for (i, x) in b.iter_mut().enumerate() {
                *x = ns.half.mem.get(&tkey(cta, lane, col, i as u64)).copied();
            }
        }
        b
    }

    /// The tensor-core operand a 16-bit half with bytes (lo, hi) holds: zero, a node (made for
    /// an input element on first use), or unknown. bf16 only.
    pub(crate) fn half_op(&mut self, lo: Option<Src>, hi: Option<Src>) -> HalfOp {
        let Some(ns) = self.num.as_mut() else {
            return HalfOp::Unknown;
        };
        match (lo, hi) {
            (Some(Src::Zero), Some(Src::Zero)) => HalfOp::Zero,
            // bf16 operands only: an f16 encoding is not the bf16 value.
            (Some(Src::N { node, b: 0 }), Some(Src::N { node: n2, b: 1 }))
                if node == n2 && !ns.half.f16_nodes.contains(&node) =>
            {
                HalfOp::Node(node)
            }
            (
                Some(Src::E { alloc, elem, b: 0 }),
                Some(Src::E {
                    alloc: a2,
                    elem: e2,
                    b: 1,
                }),
            ) if alloc == a2 && elem == e2 && ns.half.inputs.get(&alloc) == Some(&Half::Bf16) => {
                match self.half_input_node(alloc, elem) {
                    Some(n) => HalfOp::Node(n),
                    None => HalfOp::Unknown,
                }
            }
            _ => HalfOp::Unknown,
        }
    }

    /// The factor an 8-bit code byte with provenance `s` is for a narrow tensor-core step that
    /// reads it as format `t`: a leaf of an input byte, zero (a zero byte of a format whose code
    /// 0 is +0), the constant of code 0 (ue8m0: 2^-127), or unknown.
    pub(crate) fn byte_op(&mut self, s: Option<Src>, t: ptxv_ir::Lp) -> HalfOp {
        self.code_op(s, t, 0)
    }

    /// `byte_op` for nibble `nib` (0 = low) of a byte holding two packed FP4 codes (`t` =
    /// E2m1): the leaf of input nibble 2 byte + nib.
    pub(crate) fn code_op(&mut self, s: Option<Src>, t: ptxv_ir::Lp, nib: u64) -> HalfOp {
        match s {
            Some(Src::E { alloc, elem, b: 0 })
                if self.num.as_ref().and_then(|n| n.half.inputs.get(&alloc))
                    == Some(&Half::Byte) =>
            {
                let elem = if t == ptxv_ir::Lp::E2m1 {
                    2 * elem + nib
                } else {
                    elem
                };
                match self.byte_leaf(alloc, elem, t) {
                    Some(n) => HalfOp::Node(n),
                    None => HalfOp::Unknown,
                }
            }
            Some(Src::Zero) => {
                let v = crate::lowp::value(0, t);
                if v == 0.0 {
                    return HalfOp::Zero;
                }
                match self.num.as_mut() {
                    Some(ns) => HalfOp::Node(ns.leaf((v as f32).to_bits(), 0)),
                    None => HalfOp::Unknown,
                }
            }
            _ => HalfOp::Unknown,
        }
    }

    /// The `Input` leaf of input byte `elem` of `alloc` read as a code of format `t` (made on
    /// first use): its value decoded; the leaf's element index carries the format
    /// (`equiv::leaf_tagged`), so one byte read as two formats gives two leaves. FP4 (E2m1):
    /// `elem` is a nibble index (`equiv::leaf_code`).
    pub(crate) fn byte_leaf(&mut self, alloc: usize, elem: u64, t: ptxv_ir::Lp) -> Option<u32> {
        let key = crate::equiv::leaf_tagged(elem, t);
        let ns = self.num.as_mut()?;
        if let Some(&n) = ns.half.input_nodes.get(&(alloc, key)) {
            return Some(n);
        }
        let init = &self.launch.allocs.get(alloc)?.init;
        let c = crate::equiv::leaf_code(t, elem, |i| init.get(i as usize).map(|&b| b as u32))?;
        let v = crate::lowp::value(c, t);
        if !v.is_finite() {
            return None;
        }
        let bits = (v as f32).to_bits();
        ns.nodes.push(crate::num::Node {
            kind: NodeKind::Input((alloc, key)),
            hw: bits,
            exact: crate::exact::from(v),
            err: vec![],
            pc: 0,
        });
        let id = (ns.nodes.len() - 1) as u32;
        ns.half.input_nodes.insert((alloc, key), id);
        Some(id)
    }

    /// The `Input` leaf of a 16-bit input element (made on first use): its value widened.
    pub(crate) fn half_input_node(&mut self, alloc: usize, elem: u64) -> Option<u32> {
        let ns = self.num.as_mut()?;
        if let Some(&n) = ns.half.input_nodes.get(&(alloc, elem)) {
            return Some(n);
        }
        let kind = *ns.half.inputs.get(&alloc)?;
        let init = &self.launch.allocs.get(alloc)?.init;
        let o = 2 * elem as usize;
        let h = u16::from_le_bytes(init.get(o..o + 2)?.try_into().ok()?);
        let bits = match kind {
            Half::Bf16 => (h as u32) << 16,
            Half::F16 => crate::elementwise::f16_to_f32_bits(h),
            Half::Byte => return None,
        };
        ns.nodes.push(crate::num::Node {
            kind: NodeKind::Input((alloc, elem)),
            hw: bits,
            exact: crate::exact::from(f32::from_bits(bits) as f64),
            err: vec![],
            pc: 0,
        });
        let id = (ns.nodes.len() - 1) as u32;
        ns.half.input_nodes.insert((alloc, elem), id);
        if kind == Half::F16 {
            ns.half.f16_nodes.insert(id);
        }
        Some(id)
    }

    /// `cvt.rn.f16x2.<from>x2` / `cvt.rn.bf16x2.<from>x2` (SASS `F2FP.<F16|BF16>.<from>.UNPACK_B`)
    /// of two input codes: each result half is the 16-bit encoding of the code's leaf
    /// (`byte_leaf`, the format the cvt reads), exact for every finite code (N-CVT-LP: the
    /// conversion is exact; narrow values are representable in f16 and bf16, ue8m0 in bf16
    /// only). An f16 result marks the leaf `f16_nodes`, so it is widened by HADD2.F32
    /// (`CvtF32F16`) to the leaf itself; a leaf already used as f16 is not taken as bf16. A zero
    /// source byte is code 0 (+0, except ue8m0: unknown); `.relu`, other sources: unknown.
    /// ASSUME(N-HALF-LEAF): see ASSUMPTIONS.md
    fn half_unpack(&mut self, t: usize, op: &Op) -> Option<(Reg, Bytes)> {
        let Op::CvtUnpack {
            d,
            a: Val::Reg(r, _),
            hi,
            from,
            to,
            relu: false,
            byte,
        } = op
        else {
            return None;
        };
        if r.file != File::R || !matches!(to, Lp::F16 | Lp::Bf16) {
            return None;
        }
        let f16 = *to == Lp::F16;
        if f16 && *from == Lp::Ue8m0 {
            return None;
        }
        let x = *self.num.as_ref()?.half.regs.get(&(t as u32, r.idx))?;
        let base = if *hi { 2 } else { 0 } + *byte as usize;
        if *from != Lp::E2m1 && base > 2 {
            return None;
        }
        // (source byte, nibble) of the low and high codes.
        let codes = if *from == Lp::E2m1 {
            [(x[base], 0), (x[base], 1)]
        } else {
            [(x[base], 0), (x[base + 1], 0)]
        };
        let mut o: Bytes = [None; 4];
        for (h, (src, nib)) in codes.into_iter().enumerate() {
            // e4m3 of (byte & 0x7f) = ue4m3 of the byte, every byte (`ue4m3_is_masked_e4m3`).
            let (src, fmt) = match src {
                Some(Src::E7 { alloc, elem }) if *from == Lp::E4m3 => {
                    (Some(Src::E { alloc, elem, b: 0 }), Lp::Ue4m3)
                }
                Some(Src::E7 { .. }) => continue,
                s => (s, *from),
            };
            let pair = match src {
                Some(Src::Zero) if *from != Lp::Ue8m0 => [Some(Src::Zero); 2],
                Some(Src::E { alloc, elem, b: 0 })
                    if self.num.as_ref()?.half.inputs.get(&alloc) == Some(&Half::Byte) =>
                {
                    let elem = if *from == Lp::E2m1 {
                        2 * elem + nib
                    } else {
                        elem
                    };
                    let Some(node) = self.byte_leaf(alloc, elem, fmt) else {
                        continue;
                    };
                    let ns = self.num.as_mut()?;
                    if f16 {
                        ns.half.f16_nodes.insert(node);
                    } else if ns.half.f16_nodes.contains(&node) {
                        continue;
                    }
                    [Some(Src::N { node, b: 0 }), Some(Src::N { node, b: 1 })]
                }
                _ => continue,
            };
            o[2 * h] = pair[0];
            o[2 * h + 1] = pair[1];
        }
        o.iter().any(|b| b.is_some()).then_some((*d, o))
    }

    /// After an instruction (before `taint_post` drops untouched nodes): byte provenance of
    /// its destinations, and an `Input` leaf for a register holding one widened element.
    pub(crate) fn half_post(&mut self, active: &[usize], pc: u32) {
        let Some(ns) = self.num.as_ref() else { return };
        if !ns.half.is_on() {
            return;
        }
        let inst = self.k.code[pc as usize].clone();
        for &t in active {
            // Read before the destination (possibly the source) loses its provenance.
            let unpacked = self.half_unpack(t, &inst.op);
            let ns = self.num.as_ref().unwrap();
            let get = |v: &Val| -> Option<Bytes> {
                match v {
                    Val::Imm(c) => Some(imm_bytes(*c as u32)),
                    Val::Reg(r, 1) if r.file == File::R => {
                        ns.half.regs.get(&(t as u32, r.idx)).copied()
                    }
                    _ => None,
                }
            };
            let out: Option<(Reg, Bytes)> = match &inst.op {
                Op::Mov { d, a } => get(a).map(|b| (*d, b)),
                Op::Prmt { d, a, sel, b } => match (get(a), get(b), sel) {
                    (Some(x), Some(y), Val::Imm(s)) => {
                        let src = [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]];
                        let mut o: Bytes = [None; 4];
                        for (i, ob) in o.iter_mut().enumerate() {
                            let nib = (*s >> (4 * i)) & 0xf;
                            let sb = src[(nib & 7) as usize];
                            *ob = if nib & 8 != 0 {
                                // Sign replication: only a zero byte stays known.
                                (sb == Some(Src::Zero)).then_some(Src::Zero)
                            } else {
                                sb
                            };
                        }
                        Some((*d, o))
                    }
                    // A register operand without provenance contributes unknown bytes.
                    (x, y, Val::Imm(s)) => {
                        let src: [Option<Src>; 8] = {
                            let x = x.unwrap_or([None; 4]);
                            let y = y.unwrap_or([None; 4]);
                            [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
                        };
                        let mut o: Bytes = [None; 4];
                        for (i, ob) in o.iter_mut().enumerate() {
                            let nib = (*s >> (4 * i)) & 0xf;
                            *ob = if nib & 8 != 0 {
                                None
                            } else {
                                src[(nib & 7) as usize]
                            };
                        }
                        o.iter()
                            .any(|b| matches!(b, Some(Src::E { .. } | Src::N { .. })))
                            .then_some((*d, o))
                    }
                    _ => None,
                },
                Op::Shf {
                    d,
                    lo,
                    sh: Val::Imm(k),
                    hi,
                    right,
                    signed: false,
                    high: false,
                    ..
                } if k % 8 == 0 && *k < 32 => match (get(lo), get(hi)) {
                    (Some(x), Some(h)) if h == [Some(Src::Zero); 4] => {
                        let s = (*k / 8) as usize;
                        let mut o: Bytes = [Some(Src::Zero); 4];
                        for (i, ob) in o.iter_mut().enumerate() {
                            *ob = if *right {
                                if i + s < 4 { x[i + s] } else { Some(Src::Zero) }
                            } else if i >= s {
                                x[i - s]
                            } else {
                                Some(Src::Zero)
                            };
                        }
                        Some((*d, o))
                    }
                    _ => None,
                },
                Op::Imad {
                    d,
                    a,
                    b,
                    c,
                    neg_c: false,
                    not_c: false,
                    wide: false,
                    hi: false,
                    cin: None,
                    cout: None,
                    ..
                } if matches!(a, Val::Imm(0)) || matches!(b, Val::Imm(0)) => {
                    // `IMAD.MOV.U32 d, RZ, RZ, c`: a move.
                    get(c).map(|x| (*d, x))
                }
                Op::Imad {
                    d,
                    a,
                    b,
                    c,
                    neg_c: false,
                    not_c: false,
                    wide: false,
                    hi: false,
                    cin: None,
                    cout: None,
                    ..
                } => {
                    // x * 2^(8s) + 0: a shift left by whole bytes.
                    let (x, m) = match (a, b) {
                        (Val::Imm(m), r) | (r, Val::Imm(m)) => (get(r), *m as u32),
                        _ => (None, 0),
                    };
                    match (x, c) {
                        (Some(x), Val::Imm(0))
                            if m.is_power_of_two() && m.trailing_zeros() % 8 == 0 =>
                        {
                            let s = (m.trailing_zeros() / 8) as usize;
                            let mut o: Bytes = [Some(Src::Zero); 4];
                            for (i, ob) in o.iter_mut().enumerate().skip(s) {
                                *ob = x[i - s];
                            }
                            Some((*d, o))
                        }
                        _ => None,
                    }
                }
                Op::Lop3 {
                    d,
                    a,
                    b,
                    c,
                    lut,
                    p: None,
                    ..
                } => lop3_bytes(&[get(a), get(b), get(c)], [a, b, c], *lut).map(|o| (*d, o)),
                _ => None,
            };
            let ns = self.num.as_mut().unwrap();
            // Conversions read their source's provenance before the destination (possibly the
            // same register) loses it.
            let mut leaf: Option<(Reg, usize, u64)> = None;
            let mut widened: Option<(Reg, u32)> = None;
            // bf16 widened by a conversion op ((a & 0xffff) << 16).
            if let Op::CvtF32Bf16 {
                d,
                a: Val::Reg(r, _),
            } = &inst.op
            {
                if let Some(x) = ns.half.regs.get(&(t as u32, r.idx)) {
                    if let (Some(Src::N { node, b: 0 }), Some(Src::N { node: n2, b: 1 })) =
                        (x[0], x[1])
                    {
                        if node == n2 && !ns.half.f16_nodes.contains(&node) {
                            widened = Some((*d, node));
                        }
                    }
                    if let (
                        Some(Src::E { alloc, elem, b: 0 }),
                        Some(Src::E {
                            alloc: a2,
                            elem: e2,
                            b: 1,
                        }),
                    ) = (x[0], x[1])
                    {
                        if alloc == a2
                            && elem == e2
                            && ns.half.inputs.get(&alloc) == Some(&Half::Bf16)
                        {
                            leaf = Some((*d, alloc, elem));
                        }
                    }
                }
            }
            // f16 widened by HADD2.F32 (exact).
            if let Op::CvtF32F16 {
                d,
                a: Val::Reg(r, _),
                hi,
            } = &inst.op
            {
                if let Some(x) = ns.half.regs.get(&(t as u32, r.idx)) {
                    let (l, h) = if *hi { (x[2], x[3]) } else { (x[0], x[1]) };
                    // A rounded f16 value widened back: the rounding node itself.
                    if let (Some(Src::N { node, b: 0 }), Some(Src::N { node: n2, b: 1 })) = (l, h) {
                        if node == n2 && ns.half.f16_nodes.contains(&node) {
                            widened = Some((*d, node));
                        }
                    }
                    if let (
                        Some(Src::E { alloc, elem, b: 0 }),
                        Some(Src::E {
                            alloc: a2,
                            elem: e2,
                            b: 1,
                        }),
                    ) = (l, h)
                    {
                        if alloc == a2
                            && elem == e2
                            && ns.half.inputs.get(&alloc) == Some(&Half::F16)
                        {
                            leaf = Some((*d, alloc, elem));
                        }
                    }
                }
            }
            // Every R destination loses its provenance unless computed above (or set by a load
            // in this instruction).
            for r in inst.op.dst_regs() {
                if r.file == File::R && !ns.half.set_now.contains(&(t as u32, r.idx)) {
                    ns.half.regs.remove(&(t as u32, r.idx));
                }
            }
            if let Some((d, o)) = out {
                if d.file == File::R {
                    if o.iter().all(|b| b.is_none()) {
                        continue;
                    }
                    ns.half.regs.insert((t as u32, d.idx), o);
                    // bf16 widened: [0, 0, lo, hi] of one element.
                    if let [
                        Some(Src::Zero),
                        Some(Src::Zero),
                        Some(Src::E { alloc, elem, b: 0 }),
                        Some(Src::E {
                            alloc: a2,
                            elem: e2,
                            b: 1,
                        }),
                    ] = o
                    {
                        if alloc == a2
                            && elem == e2
                            && ns.half.inputs.get(&alloc) == Some(&Half::Bf16)
                        {
                            leaf = Some((d, alloc, elem));
                        }
                    }
                    // A rounded bf16 value widened back: the rounding node itself.
                    if let [
                        Some(Src::Zero),
                        Some(Src::Zero),
                        Some(Src::N { node, b: 0 }),
                        Some(Src::N { node: n2, b: 1 }),
                    ] = o
                    {
                        if node == n2 && !ns.half.f16_nodes.contains(&node) {
                            widened = Some((d, node));
                        }
                    }
                }
            }
            if let Some((d, o)) = unpacked {
                ns.half.regs.insert((t as u32, d.idx), o);
            }
            if let Some((d, n)) = widened {
                ns.set_reg_node(t, d, Some(n));
            }
            if let Some((d, alloc, elem)) = leaf {
                let bits = self.rd(t, &Val::Reg(d, 1)).unwrap_or(0) as u32;
                let ns = self.num.as_mut().unwrap();
                ns.push_input_leaf(t, d, (alloc, elem), bits, pc);
            }
        }
        if let Some(ns) = self.num.as_mut() {
            ns.half.set_now.clear();
        }
    }
}

fn imm_bytes(c: u32) -> Bytes {
    let mut o: Bytes = [None; 4];
    for (i, ob) in o.iter_mut().enumerate() {
        if (c >> (8 * i)) & 0xff == 0 {
            *ob = Some(Src::Zero);
        }
    }
    o
}

/// LOP3 with one operand a tracked register and the others constants: per byte, the identity
/// of that byte, or zero; anything else unknown.
fn lop3_bytes(ops: &[Option<Bytes>; 3], vals: [&Val; 3], lut: u8) -> Option<Bytes> {
    let regs: Vec<usize> = (0..3)
        .filter(|&i| matches!(vals[i], Val::Reg(r, _) if r.file == File::R))
        .collect();
    let [ri] = regs.as_slice() else {
        return None;
    };
    let x = ops[*ri]?;
    let consts: Vec<u32> = (0..3)
        .map(|i| match vals[i] {
            Val::Imm(c) => *c as u32,
            _ => 0,
        })
        .collect();
    let f = |bits: [u32; 3]| -> u32 {
        let idx = (bits[0] << 2) | (bits[1] << 1) | bits[2];
        (lut as u32 >> idx) & 1
    };
    let mut o: Bytes = [None; 4];
    for (byte, ob) in o.iter_mut().enumerate() {
        let (mut ident, mut zero) = (true, true);
        // Identity on bits 0-6 and zero on bit 7 (`x & 0x7f`).
        let mut ident7 = true;
        for j in 0..8 {
            let bit = 8 * byte + j;
            let mut b = [0u32; 3];
            for i in 0..3 {
                b[i] = (consts[i] >> bit) & 1;
            }
            b[*ri] = 0;
            let f0 = f(b);
            b[*ri] = 1;
            let f1 = f(b);
            ident &= f0 == 0 && f1 == 1;
            zero &= f0 == 0 && f1 == 0;
            ident7 &= if j == 7 {
                f0 == 0 && f1 == 0
            } else {
                f0 == 0 && f1 == 1
            };
        }
        *ob = if zero {
            Some(Src::Zero)
        } else if ident {
            x[byte]
        } else if let (true, Some(Src::E { alloc, elem, b: 0 })) = (ident7, x[byte]) {
            Some(Src::E7 { alloc, elem })
        } else {
            None
        };
    }
    Some(o)
}

impl crate::num::NumState {
    /// An `Input` leaf for register `d` of thread `t` (a widened half-precision element).
    pub(crate) fn push_input_leaf(
        &mut self,
        t: usize,
        d: Reg,
        orig: (usize, u64),
        bits: u32,
        pc: u32,
    ) {
        self.nodes.push(crate::num::Node {
            kind: NodeKind::Input(orig),
            hw: bits,
            exact: crate::exact::from(f32::from_bits(bits) as f64),
            err: vec![],
            pc,
        });
        let id = (self.nodes.len() - 1) as u32;
        self.set_reg_node(t, d, Some(id));
    }
}

#[cfg(test)]
mod tests {
    use ptxv_ir::Lp;

    /// `Src::E7`: e4m3 of (byte & 0x7f) is the byte's ue4m3 value for all 256 bytes (NaN for
    /// 0x7f / 0xff alike).
    #[test]
    fn ue4m3_is_masked_e4m3() {
        for c in 0..256u32 {
            let (e, u) = (
                crate::lowp::value(c & 0x7f, Lp::E4m3),
                crate::lowp::value(c, Lp::Ue4m3),
            );
            assert!(e == u || e.is_nan() && u.is_nan(), "{c:#x}: {e} vs {u}");
            assert_eq!(
                crate::lowp::to_16(c & 0x7f, Lp::E4m3, Lp::F16, false),
                crate::lowp::to_16(c, Lp::Ue4m3, Lp::F16, false),
                "{c:#x}"
            );
        }
    }
}
