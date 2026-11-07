/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PTX front-end: text -> `ptxv_ir::Kernel`.
//!
//! The parser is generic over PTX statement syntax (guard, dotted opcode, operand list), so
//! it accepts the whole ISA grammatically. Lowering is per opcode and fails loudly on
//! anything it does not model yet; we never skip an instruction.

use std::collections::HashMap;
use std::collections::HashSet;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_ir::*;

mod l2;

/// ptxas puts user shared memory after a 1 KiB system-reserved prefix on Blackwell; we
/// mirror that so PTX and SASS reports show the same smem offsets.
pub const SMEM_RESERVED: u32 = 0x400;

#[derive(Debug, Clone)]
struct Stmt {
    line: u32,
    labels: Vec<String>,
    text: String,
}

/// (opcode, operands) of a statement, its guard dropped.
fn stmt_parts(text: &str) -> (&str, Vec<&str>) {
    let t = text.trim().trim_end_matches(';').trim();
    let t = match t.strip_prefix('@') {
        Some(r) => r.split_once(char::is_whitespace).map_or("", |x| x.1).trim(),
        None => t,
    };
    let (op, rest) = t.split_once(char::is_whitespace).unwrap_or((t, ""));
    let ops = rest
        .split(',')
        .map(|x| x.trim().trim_matches(|c| c == '{' || c == '}').trim())
        .filter(|x| !x.is_empty())
        .collect();
    (op, ops)
}

/// Destinations of `mul.f16` / `mul.f16x2` without `.rn` that ptxas cannot contract into an
/// FMA: every use is a multiplicand of `mul` / `fma`, a store or a conversion (Triton's
/// int4 / fp8 KV dequant: `mul.f16 %rs230, %rs6, %rs229` feeding `fma.rn.f16` as a). Any
/// other use (an add / sub, an fma addend, a move ptxas could propagate) leaves the mul
/// refused.
fn uncontractible_f16_muls(insts: &[&Stmt]) -> HashSet<String> {
    let mut cand: HashSet<String> = HashSet::new();
    for st in insts {
        let (op, ops) = stmt_parts(&st.text);
        if op.starts_with("mul.")
            && (op.ends_with(".f16") || op.ends_with(".f16x2"))
            && !op.split('.').any(|m| m == "rn")
        {
            if let Some(d) = ops.first() {
                cand.insert(d.to_string());
            }
        }
    }
    // Registers every use of which is a store or a conversion (the destinations of `mov`
    // packs that only get stored).
    let stored_only = |r: &str| {
        insts.iter().all(|st| {
            let (op, ops) = stmt_parts(&st.text);
            let base = op.split('.').next().unwrap_or("");
            let is_def = |i: usize| i == 0 && !matches!(base, "st" | "red" | "atom");
            ops.iter()
                .enumerate()
                .all(|(i, o)| is_def(i) || *o != r || matches!(base, "st" | "cvt"))
        })
    };
    for st in insts {
        let (op, ops) = stmt_parts(&st.text);
        let base = op.split('.').next().unwrap_or("");
        let is_def = |i: usize| i == 0 && !matches!(base, "st" | "red" | "atom");
        for (i, o) in ops.iter().enumerate() {
            if is_def(i) || !cand.contains(*o) {
                continue;
            }
            let safe = match base {
                "mul" => true,
                "fma" => i == 1 || i == 2,
                "st" | "cvt" => true,
                // `mov.b32 %r13, {%h10, %h10}`: a pack whose result is only stored.
                "mov" => st.text.contains('{') && stored_only(ops[0]),
                _ => false,
            };
            if !safe {
                cand.remove(*o);
            }
        }
    }
    cand
}

/// Remove comments, keeping newlines so line numbers survive.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                if b[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 2;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn line_of(src: &str, byte: usize) -> u32 {
    src[..byte].bytes().filter(|&c| c == b'\n').count() as u32 + 1
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == '$' || c == '%'
}

/// Split the body into statements terminated by ';', peeling off leading labels.
fn split_statements(src: &str, start: usize, end: usize) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_start = start;
    let mut labels = Vec::new();
    let body = &src[start..end];
    let mut idx = 0;
    let bytes = body.as_bytes();
    while idx < bytes.len() {
        let c = bytes[idx] as char;
        if c == ';' {
            let t = cur.trim().to_string();
            if !t.is_empty() || !labels.is_empty() {
                let lead = cur.len() - cur.trim_start().len();
                out.push(Stmt {
                    line: line_of(src, cur_start + lead),
                    labels: std::mem::take(&mut labels),
                    text: t,
                });
            }
            cur.clear();
            cur_start = start + idx + 1;
        } else if c == ':' && !cur.trim().is_empty() {
            // A label is `IDENT:` at statement start, not part of `::`.
            let t = cur.trim();
            let next = bytes.get(idx + 1).map(|&b| b as char);
            let prev_colon = idx > 0 && bytes[idx - 1] == b':';
            if next != Some(':')
                && !prev_colon
                && t.chars().next().is_some_and(is_ident_start)
                && t.chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
            {
                labels.push(t.to_string());
                cur.clear();
                cur_start = start + idx + 1;
            } else {
                cur.push(c);
            }
        } else {
            if cur.is_empty() {
                cur_start = start + idx;
            }
            cur.push(c);
        }
        idx += 1;
    }
    out
}

/// Split operands at top-level commas.
fn split_operands(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '[' | '{' => {
                depth += 1;
                cur.push(c)
            }
            ']' | '}' => {
                depth -= 1;
                cur.push(c)
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

fn parse_int(s: &str) -> Option<u64> {
    let s = s.trim();
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let s = s.trim_end_matches('U');
    let v = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).ok()?
    } else if let Some(h) = s.strip_prefix("0f").or_else(|| s.strip_prefix("0F")) {
        u64::from_str_radix(h, 16).ok()? // f32 bit pattern
    } else if let Some(h) = s.strip_prefix("0d").or_else(|| s.strip_prefix("0D")) {
        u64::from_str_radix(h, 16).ok()?
    } else {
        s.parse::<u64>().ok()?
    };
    Some(if neg { v.wrapping_neg() } else { v })
}

struct Lower<'a> {
    regs: &'a HashMap<String, Reg>,
    labels: &'a HashMap<String, usize>,
    /// Shared symbols -> shared-window address; param symbols -> param-space offset.
    shared_syms: &'a HashMap<String, u32>,
    /// `.local` arrays of the kernel (`__local_depot0`): offsets in the thread's local frame.
    local_syms: &'a HashMap<String, u32>,
    param_syms: &'a HashMap<String, u32>,
    line: u32,
    text: &'a str,
    /// Ops to run after the lowered one, under the same guard: a PTX instruction that has no
    /// single IR op (`cp.async.mbarrier.arrive`, `vote.sync.any`, `mad.lo.s64`, ...).
    extra: Vec<Op>,
    /// Hidden scratch register for those expansions (allocated only if a statement needs it:
    /// `%__x64`).
    tmp_v: Option<Reg>,
    /// Ops to run before the lowered one, under the same guard (moves into `vec_v`).
    pre: Vec<Op>,
    /// `mul.f16[x2]` destinations ptxas cannot contract (`uncontractible_f16_muls`).
    f16_mul_ok: &'a HashSet<String>,
    /// Hidden consecutive scratch registers for vector operands whose registers are not
    /// consecutive (`st.shared.v4.b32 [a], {%r262, %r261, %r260, %r259}`): `%__vec0..7`.
    vec_v: Option<Reg>,
}

/// Most elements of a vector operand (`.v8`).
const VEC_SCRATCH: u16 = 8;

/// Does this statement have a `{...}` vector operand of an `ld` / `st` whose registers are not
/// consecutive (it needs the vector scratch registers)?
fn needs_vec_scratch(text: &str, regs: &HashMap<String, Reg>) -> bool {
    let t = text.trim();
    let t = match t.strip_prefix('@') {
        Some(r) => r.split_once(char::is_whitespace).map_or("", |x| x.1).trim(),
        None => t,
    };
    if !(t.starts_with("ld.") || t.starts_with("st.") || t.starts_with("ldmatrix.")) {
        return false;
    }
    let Some((_, rest)) = t.split_once('{') else {
        return false;
    };
    let Some((inner, _)) = rest.split_once('}') else {
        return false;
    };
    let rs: Vec<Option<&Reg>> = inner.split(',').map(|x| regs.get(x.trim())).collect();
    rs.len() > 1
        && rs.iter().enumerate().any(|(i, r)| match (r, rs[0]) {
            (Some(r), Some(r0)) => r.file != r0.file || r.idx != r0.idx + i as u16,
            _ => true,
        })
}

/// Does this statement's expansion need the hidden scratch register?
fn needs_scratch(text: &str) -> bool {
    let t = text.trim();
    let t = match t.strip_prefix('@') {
        Some(r) => r.split_once(char::is_whitespace).map_or("", |x| x.1).trim(),
        None => t,
    };
    t.starts_with("mad.lo.s64")
        || t.starts_with("mad.lo.u64")
        || (t.starts_with("atom.") || t.starts_with("red.")) && t.contains(".v2.bf16")
}

impl<'a> Lower<'a> {
    fn scratch_v(&self) -> Result<Reg> {
        self.tmp_v
            .ok_or_else(|| self.err("internal: no scratch register"))
    }

    fn err(&self, msg: impl std::fmt::Display) -> anyhow::Error {
        anyhow!("ptx line {}: {msg}\n    `{}`", self.line, self.text)
    }

    fn reg(&self, s: &str) -> Result<Reg> {
        // A one-element vector `{ %r1 }` (Triton: `ld.global.b32 { %r1 }, [..]`).
        let s = match s.trim().strip_prefix('{').and_then(|x| x.strip_suffix('}')) {
            Some(x) if !x.contains(',') => x.trim(),
            _ => s,
        };
        self.regs
            .get(s)
            .copied()
            .ok_or_else(|| self.err(format!("unknown register {s}")))
    }

    fn sreg(s: &str) -> Option<SReg> {
        let dim = |c: &str| match c {
            "x" => Some(0u8),
            "y" => Some(1),
            "z" => Some(2),
            _ => None,
        };
        let (base, d) = s
            .split_once('.')
            .map(|(a, b)| (a, Some(b)))
            .unwrap_or((s, None));
        Some(match (base, d) {
            ("%tid", Some(c)) => SReg::Tid(dim(c)?),
            ("%ntid", Some(c)) => SReg::NTid(dim(c)?),
            ("%ctaid", Some(c)) => SReg::CtaId(dim(c)?),
            ("%nctaid", Some(c)) => SReg::NCtaId(dim(c)?),
            ("%laneid", None) => SReg::LaneId,
            ("%warpid", None) => SReg::WarpId,
            ("%cluster_ctarank", None) => SReg::ClusterCtaRank,
            ("%cluster_nctarank", None) => SReg::ClusterNCtaRank,
            _ => return None,
        })
    }

    fn val(&self, s: &str) -> Result<Val> {
        let s = s.trim();
        let s = match s.strip_prefix('{').and_then(|x| x.strip_suffix('}')) {
            Some(x) if !x.contains(',') => x.trim(),
            _ => s,
        };
        if let Some(sr) = Self::sreg(s) {
            return Ok(Val::SReg(sr));
        }
        if s.starts_with('%') {
            return Ok(Val::Reg(self.reg(s)?, 1));
        }
        if let Some(v) = parse_int(s) {
            return Ok(Val::Imm(v));
        }
        if let Some(&a) = self.shared_syms.get(s) {
            return Ok(Val::SReg(SReg::SharedAddr(a)));
        }
        if let Some(&a) = self.local_syms.get(s) {
            return Ok(Val::Imm(a as u64));
        }
        if let Some(&a) = self.param_syms.get(s) {
            return Ok(Val::Imm(a as u64));
        }
        Err(self.err(format!("bad operand `{s}`")))
    }

    /// Predicate operand, optionally negated (`%p3`, `!%p3`).
    fn pred_src(&self, s: &str) -> Result<Pred> {
        let s = s.trim();
        let (neg, s) = match s.strip_prefix('!') {
            Some(r) => (true, r),
            None => (false, s),
        };
        Ok(Pred {
            reg: Some(self.pred_dst(s)?),
            neg,
        })
    }

    /// An `ld` / `st` vector operand `{%r16, %r17, ...}` -> (first register, count, moves).
    /// Registers that are not consecutive go through the vector scratch registers: the moves
    /// into them run before a store (`pre`), out of them after a load (`extra`).
    fn reg_vec_mem(&self, s: &str, load: bool) -> Result<(Reg, u8, Vec<Op>)> {
        if let Ok((r, n)) = self.reg_vec(s) {
            return Ok((r, n, vec![]));
        }
        let mut moves = Vec::new();
        let inner = s.trim().trim_start_matches('{').trim_end_matches('}');
        let regs: Vec<Reg> = inner
            .split(',')
            .map(|r| self.reg(r.trim()))
            .collect::<Result<_>>()?;
        let base = self
            .vec_v
            .ok_or_else(|| self.err("internal: no vector scratch registers"))?;
        if regs.len() > VEC_SCRATCH as usize || regs.iter().any(|r| r.file != File::V) {
            return Err(self.err("vector operand registers must be consecutive"));
        }
        for (i, r) in regs.iter().enumerate() {
            let tmp = Reg {
                file: File::V,
                idx: base.idx + i as u16,
            };
            moves.push(if load {
                Op::Mov {
                    d: *r,
                    a: Val::Reg(tmp, 1),
                }
            } else {
                Op::Mov {
                    d: tmp,
                    a: Val::Reg(*r, 1),
                }
            });
        }
        Ok((base, regs.len() as u8, moves))
    }

    /// `{%r16, %r17, ...}` -> (first register, count); registers must be consecutive.
    fn reg_vec(&self, s: &str) -> Result<(Reg, u8)> {
        let inner = s.trim().trim_start_matches('{').trim_end_matches('}');
        let regs: Vec<Reg> = inner
            .split(',')
            .map(|r| self.reg(r.trim()))
            .collect::<Result<_>>()?;
        for (i, r) in regs.iter().enumerate() {
            if r.idx != regs[0].idx + i as u16 || r.file != regs[0].file {
                return Err(self.err("vector operand registers must be consecutive"));
            }
        }
        Ok((regs[0], regs.len() as u8))
    }

    fn pred_dst(&self, s: &str) -> Result<Reg> {
        let r = self.reg(s.trim())?;
        if r.file != File::P {
            return Err(self.err(format!("{s} is not a predicate")));
        }
        Ok(r)
    }

    fn dst(&self, s: &str) -> Result<Reg> {
        self.reg(s.trim())
    }

    /// `[base]`, `[base+off]`, `[sym]`, `[sym+off]`, `[imm]`.
    fn addr(&self, s: &str) -> Result<Addr> {
        let inner = s
            .trim()
            .strip_prefix('[')
            .and_then(|x| x.strip_suffix(']'))
            .ok_or_else(|| self.err(format!("expected [address], got `{s}`")))?
            .trim();
        let (b, off) = if let Some(p) = inner.rfind(['+', '-']).filter(|&p| p > 0) {
            let o =
                parse_int(&inner[p..].replace('+', "")).ok_or_else(|| self.err("bad offset"))?;
            (inner[..p].trim(), o as i64)
        } else {
            (inner, 0)
        };
        if b.starts_with('%') {
            return Ok(Addr {
                base: Some(Val::Reg(self.reg(b)?, 1)),
                base2: None,
                off,
            });
        }
        if let Some(&a) = self.shared_syms.get(b) {
            return Ok(Addr {
                base: Some(Val::SReg(SReg::SharedAddr(a))),
                base2: None,
                off,
            });
        }
        if let Some(&a) = self.param_syms.get(b) {
            return Ok(Addr {
                base: None,
                base2: None,
                off: a as i64 + off,
            });
        }
        if let Some(v) = parse_int(b) {
            return Ok(Addr {
                base: None,
                base2: None,
                off: v as i64 + off,
            });
        }
        Err(self.err(format!("bad address `{s}`")))
    }
}

/// Narrow-format `cvt` forms (N-CVT-LP): `.<f8|f6|f4|ue8m0>x2` from f32 pairs or f16x2 /
/// bf16x2, the `.rs` x4 / f16x2 / bf16x2 packs, and the f16x2 / bf16x2 decodes. None: not a
/// narrow form.
fn cvt_lp(lw: &Lower, mods: &[&str], ops: &[String]) -> Result<Option<Op>> {
    let n = mods.len();
    let (to, from) = (mods[n - 2], mods[n - 1]);
    let has = |m: &str| mods.contains(&m);
    let lp = |t: &str| -> Option<(Lp, u8)> {
        let (base, x) = t.split_at(t.len().checked_sub(2)?);
        let x = x.strip_prefix('x')?.parse::<u8>().ok()?;
        Some((
            match base {
                "e4m3" => Lp::E4m3,
                "e5m2" => Lp::E5m2,
                "e2m3" => Lp::E2m3,
                "e3m2" => Lp::E3m2,
                "e2m1" => Lp::E2m1,
                "ue8m0" => Lp::Ue8m0,
                "f16" => Lp::F16,
                "bf16" => Lp::Bf16,
                _ => return None,
            },
            x,
        ))
    };
    let rnd = if has("rn") {
        LpRnd::Rn
    } else if has("rz") {
        LpRnd::Rz
    } else if has("rp") {
        LpRnd::Rp
    } else if has("rs") {
        LpRnd::Rs
    } else {
        return Ok(None);
    };
    let (sat, relu) = (has("satfinite"), has("relu"));
    let op = |i: usize| -> Result<&str> {
        ops.get(i)
            .map(|s| s.as_str())
            .ok_or_else(|| lw.err(format!("cvt: missing operand {i}")))
    };
    Ok(Some(match (lp(to), from) {
        // Encodes from an f32 pair: d = {a (high lane), b}.
        (Some((t, 2)), "f32")
            if !matches!(t, Lp::F16 | Lp::Bf16) || rnd == LpRnd::Rs || rnd == LpRnd::Rn && sat =>
        {
            let rs = rnd == LpRnd::Rs;
            Op::CvtPack {
                d: lw.dst(op(0)?)?,
                a: lw.val(op(1)?)?,
                b: lw.val(op(2)?)?,
                c: if rs { lw.val(op(3)?)? } else { Val::Imm(0) },
                half: None,
                to: t,
                rnd,
                sat,
                relu,
            }
        }
        (Some((t, 4)), "f32") if rnd == LpRnd::Rs => {
            let v: Vec<Val> = op(1)?
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| lw.val(x.trim()))
                .collect::<Result<_>>()?;
            let src: [Val; 4] = v
                .try_into()
                .map_err(|_| lw.err("cvt .x4 wants four sources"))?;
            Op::CvtRsX4 {
                d: lw.dst(op(0)?)?,
                src,
                rbits: lw.val(op(2)?)?,
                to: t,
                relu,
            }
        }
        (Some((t, 2)), "f16x2" | "bf16x2") if !matches!(t, Lp::F16 | Lp::Bf16) => Op::CvtPack {
            d: lw.dst(op(0)?)?,
            a: lw.val(op(1)?)?,
            b: Val::Imm(0),
            c: Val::Imm(0),
            half: Some(if from == "f16x2" { Lp::F16 } else { Lp::Bf16 }),
            to: t,
            rnd,
            sat,
            relu,
        },
        // Decodes: d = {high code, low code} as f16x2 / bf16x2.
        (Some((t @ (Lp::F16 | Lp::Bf16), 2)), _) => match lp(from) {
            Some((f, 2)) if !matches!(f, Lp::F16 | Lp::Bf16) && rnd == LpRnd::Rn => Op::CvtUnpack {
                d: lw.dst(op(0)?)?,
                a: lw.val(op(1)?)?,
                hi: false,
                from: f,
                to: t,
                relu,
                byte: 0,
            },
            _ => return Ok(None),
        },
        _ => return Ok(None),
    }))
}

fn type_bits(t: &str) -> Option<u8> {
    Some(match t {
        "b8" | "u8" | "s8" => 8,
        "b16" | "u16" | "s16" | "f16" | "bf16" => 16,
        "b32" | "u32" | "s32" | "f32" => 32,
        "b64" | "u64" | "s64" | "f64" => 64,
        "b128" => 128,
        "pred" => 1,
        _ => return None,
    })
}

fn space_of(m: &str) -> Option<Space> {
    Some(match m {
        "shared" | "shared::cta" => Space::Shared,
        "shared::cluster" => Space::SharedCluster,
        "global" => Space::Global,
        "param" => Space::Param,
        _ => return None,
    })
}

fn scope_of(mods: &[&str]) -> Option<Scope> {
    mods.iter().find_map(|m| match *m {
        "cta" => Some(Scope::Cta),
        "cluster" => Some(Scope::Cluster),
        "gpu" => Some(Scope::Gpu),
        "sys" => Some(Scope::Sys),
        _ => None,
    })
}

/// Memory semantics of an ld / st / atom / red. `.volatile` is relaxed at system scope (PTX
/// ISA: volatile = relaxed.sys for memory-model purposes); `atom` / `red` default to
/// relaxed.gpu. ASSUME(S-SCOPE)
fn mem_sync(mods: &[&str], rmw: bool) -> Result<Option<MemSync>> {
    let sem = mods.iter().find_map(|m| match *m {
        "relaxed" => Some(Sem::Relaxed),
        "acquire" => Some(Sem::Acquire),
        "release" => Some(Sem::Release),
        "acq_rel" => Some(Sem::AcqRel),
        _ => None,
    });
    if mods.contains(&"volatile") {
        return Ok(Some(MemSync {
            sem: Sem::Relaxed,
            scope: Scope::Sys,
        }));
    }
    // ld/st.mmio.relaxed.sys: a strong relaxed system-scope access (SASS STG.E.64.MMIO.SYS,
    // probes/memhints).
    if mods.contains(&"mmio") {
        if sem != Some(Sem::Relaxed) || scope_of(mods) != Some(Scope::Sys) || rmw {
            bail!("only ld/st.mmio.relaxed.sys is modeled");
        }
        return Ok(Some(MemSync {
            sem: Sem::Relaxed,
            scope: Scope::Sys,
        }));
    }
    Ok(match (sem, rmw) {
        (None, false) => None,
        (sem, _) => Some(MemSync {
            sem: sem.unwrap_or(Sem::Relaxed),
            scope: scope_of(mods).unwrap_or(Scope::Gpu),
        }),
    })
}

/// Inline-asm scopes (`{ .reg .pred p; L: ... }` inside an entry body, as Triton emits for
/// every try_wait) are flattened: labels and registers declared in a scope are renamed
/// `name__sN` within it, and the braces become spaces, so lines keep their positions. A scope
/// brace is one at statement position (first on its line, or after `;`, `{`, `}` or a label);
/// vector operand braces are not.
fn flatten_scopes(src: &str) -> String {
    let b = src.as_bytes();
    // Pass 1: find scope braces (depth >= 2: inside an entry body).
    let mut depth = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    let mut scopes: Vec<(usize, usize)> = Vec::new();
    let mut prev = b';';
    let mut line_start = true; // only whitespace since the last newline
    // An array initializer (`.global .b8 _$_str[11] = {95, 95, ...};`, Triton's assert
    // strings) is not a scope.
    let mut in_init = false;
    for (i, &c) in b.iter().enumerate() {
        let stmt_pos = line_start || matches!(prev, b';' | b'{' | b'}' | b':');
        if in_init {
            in_init = c != b'}';
            if !c.is_ascii_whitespace() {
                prev = c;
            }
            continue;
        }
        match c {
            b'{' if prev == b'=' => in_init = true,
            b'{' if depth == 0 || stmt_pos => {
                depth += 1;
                stack.push(i);
            }
            b'}' if stmt_pos && !stack.is_empty() => {
                let open = stack.pop().unwrap();
                if depth >= 2 {
                    scopes.push((open, i));
                }
                depth -= 1;
            }
            _ => {}
        }
        if !c.is_ascii_whitespace() {
            prev = c;
            line_start = false;
        } else if c == b'\n' {
            line_start = true;
        }
    }
    if scopes.is_empty() {
        return src.to_string();
    }
    let mut out: Vec<u8> = b.to_vec();
    /// A scope's brace offsets and its (old, new) name pairs.
    type ScopeRenames = (usize, usize, Vec<(String, String)>);
    let mut renames: Vec<ScopeRenames> = Vec::new();
    for (n, &(o, c)) in scopes.iter().enumerate() {
        let body = &src[o + 1..c];
        let mut names = Vec::new();
        for st in body.split(';') {
            let t = st.trim();
            // labels: `IDENT:` at statement start (possibly several)
            let mut rest = t;
            while let Some((l, r)) = rest.split_once(':') {
                let l = l.trim();
                if !l.is_empty()
                    && l.chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
                    && !r.starts_with(':')
                {
                    names.push(l.to_string());
                    rest = r;
                } else {
                    break;
                }
            }
            if let Some(r) = rest.trim().strip_prefix(".reg") {
                for tok in r.split([' ', '\t', '\n', ',']) {
                    let tok = tok.trim();
                    if !tok.is_empty() && !tok.starts_with('.') {
                        names.push(tok.split('<').next().unwrap().to_string());
                    }
                }
            }
        }
        let map = names
            .into_iter()
            .map(|x| (x.clone(), format!("{x}__s{n}")))
            .collect();
        renames.push((o, c, map));
        out[o] = b' ';
        out[c] = b' ';
    }
    // Pass 2: rename whole identifiers inside each scope (innermost scopes come first in
    // `scopes`, so outer renames do not touch already-renamed names).
    let mut text = String::from_utf8(out).unwrap();
    for (o, c, map) in renames.into_iter().rev() {
        if map.is_empty() {
            continue;
        }
        let seg = &text[o..=c];
        let mut res = String::with_capacity(seg.len() + 64);
        let sb = seg.as_bytes();
        let mut i = 0;
        while i < sb.len() {
            let ch = sb[i] as char;
            if ch.is_ascii_alphabetic() || ch == '_' || ch == '$' || ch == '%' {
                let st = i;
                while i < sb.len()
                    && ((sb[i] as char).is_ascii_alphanumeric()
                        || sb[i] == b'_'
                        || sb[i] == b'$'
                        || sb[i] == b'%')
                {
                    i += 1;
                }
                let w = &seg[st..i];
                match map.iter().find(|(a, _)| a == w) {
                    Some((_, r)) => res.push_str(r),
                    None => res.push_str(w),
                }
            } else {
                res.push(ch);
                i += 1;
            }
        }
        text.replace_range(o..=c, &res);
    }
    text
}

/// `[.align A] .type name[N]` of a shared declaration: (name, alignment, bytes). `name[]`
/// (extern) has size 0.
fn shared_decl(rest: &str) -> Result<(String, u32, u32)> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut align = 1u32;
    let mut elem = 1u32;
    let mut size = 0u32;
    let mut name = String::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == ".align" {
            align = toks[i + 1].parse()?;
            i += 1;
        } else if let Some(b) = toks[i].strip_prefix('.').and_then(type_bits) {
            elem = (b as u32 / 8).max(1);
        } else if let Some((n, c)) = toks[i].split_once('[') {
            name = n.to_string();
            let c = c.trim_end_matches(']');
            size = if c.is_empty() {
                0
            } else {
                elem * c.parse::<u32>()?
            };
        } else {
            name = toks[i].to_string();
            size = elem;
        }
        i += 1;
    }
    Ok((name, align, size))
}

/// Parse a PTX module and lower the named entry (or the only entry).
pub fn parse_kernel(src: &str, entry: Option<&str>) -> Result<Kernel> {
    // `.loc` / `.file` debug directives end at the newline, not at a `;`: drop them (keeping
    // the newline) so they do not swallow the next statement.
    let src: String = strip_comments(src)
        .lines()
        .map(|l| {
            let t = l.trim_start();
            // (`.local` declarations stay: only `.loc` itself.)
            let dir = |d: &str| {
                t.strip_prefix(d)
                    .is_some_and(|r| r.is_empty() || r.starts_with(char::is_whitespace))
            };
            if dir(".loc") || dir(".file") { "" } else { l }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let src = flatten_scopes(&src);
    let mut pos = 0;
    let (name, params_src, body_start, body_end, entry_at) = loop {
        let at = src[pos..]
            .find(".entry")
            .ok_or_else(|| anyhow!("no .entry found"))?
            + pos;
        let after = &src[at + 6..];
        let name_end = after.find(['(', ' ', '\n', '\t']).unwrap_or(after.len());
        let mut name = after[..name_end].trim().to_string();
        if name.is_empty() {
            name = after
                .trim_start()
                .split(['(', ' ', '\n'])
                .next()
                .unwrap_or("")
                .to_string();
        }
        let lp = at
            + 6
            + after
                .find('(')
                .ok_or_else(|| anyhow!("entry without params"))?;
        let rp = lp
            + src[lp..]
                .find(')')
                .ok_or_else(|| anyhow!("unterminated params"))?;
        let lb = rp
            + src[rp..]
                .find('{')
                .ok_or_else(|| anyhow!("entry without body"))?;
        let mut depth = 0;
        let mut rb = lb;
        for (i, c) in src[lb..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        rb = lb + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if entry.is_none_or(|e| e == name) {
            break (name, src[lp + 1..rp].to_string(), lb + 1, rb, at);
        }
        pos = rb;
    };

    // Parameters: `.param .align A .b8 name[N]` / `.param .u64 name`.
    let mut params = Vec::new();
    let mut param_syms = HashMap::new();
    let mut off = 0u32;
    for p in params_src
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        let toks: Vec<&str> = p.split_whitespace().collect();
        let mut align = 0u32;
        let mut elem = 0u32;
        let mut name = String::new();
        let mut count = 1u32;
        let mut i = 0;
        // `.param .u64 .ptr .global .align 1 p`: after `.ptr`, the state space and alignment
        // describe the pointee, not the parameter.
        let mut ptr = false;
        while i < toks.len() {
            let t = toks[i];
            if t == ".ptr" {
                ptr = true;
            } else if ptr && matches!(t, ".global" | ".shared" | ".const" | ".local") {
            } else if t == ".align" {
                if !ptr {
                    align = toks[i + 1].parse()?;
                }
                i += 1;
            } else if let Some(b) = t.strip_prefix('.').and_then(type_bits) {
                elem = (b as u32 / 8).max(1);
            } else if t != ".param" {
                if let Some((n, c)) = t.split_once('[') {
                    name = n.to_string();
                    count = c.trim_end_matches(']').parse()?;
                } else {
                    name = t.to_string();
                }
            }
            i += 1;
        }
        let size = elem * count;
        let align = if align == 0 { elem } else { align };
        off = off.div_ceil(align) * align;
        param_syms.insert(name.clone(), off);
        params.push(Param {
            name,
            offset: off,
            size,
        });
        off += size;
    }

    let stmts = split_statements(&src, body_start, body_end);

    // Pass 1: declarations, labels.
    let mut regs = HashMap::new();
    let mut vreg_names = Vec::new();
    let mut npred = 0u16;
    let mut shared_syms = HashMap::new();
    let mut local_syms = HashMap::new();
    let mut local_bytes = 0u32;
    let mut shared_syms_pre: Vec<(String, u32)> = Vec::new();
    let mut smem = SMEM_RESERVED;
    // Module-scope shared variables (before the entry). `.extern .shared x[]` is the dynamic
    // shared memory window: it starts after all static shared memory.
    let mut externs = Vec::new();
    for line in src[..entry_at].split([';', '\n']) {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(".extern") {
            if let Some(rest) = rest.trim().strip_prefix(".shared") {
                let (name, align, _) = shared_decl(rest)?;
                externs.push((name, align));
            }
        } else if let Some(rest) = t.strip_prefix(".shared") {
            let (name, align, size) = shared_decl(rest)?;
            smem = smem.div_ceil(align) * align;
            shared_syms_pre.push((name, smem));
            smem += size;
        }
    }
    let mut labels = HashMap::new();
    let mut insts_src: Vec<&Stmt> = Vec::new();
    for st in &stmts {
        for l in &st.labels {
            labels.insert(l.clone(), insts_src.len());
        }
        let t = st.text.as_str();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix(".reg") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            let is_pred = toks.first() == Some(&".pred");
            // A .b128 register takes two consecutive 64-bit slots (low, high).
            let is_b128 = toks.first() == Some(&".b128");
            for decl in toks[1..].join(" ").split(',') {
                let decl = decl.trim();
                let names: Vec<String> = if let Some((base, n)) = decl.split_once('<') {
                    let n: u32 = n.trim_end_matches('>').parse()?;
                    (0..n).map(|i| format!("{base}{i}")).collect()
                } else {
                    vec![decl.to_string()]
                };
                for nm in names {
                    let r = if is_pred {
                        npred += 1;
                        Reg {
                            file: File::P,
                            idx: npred - 1,
                        }
                    } else {
                        vreg_names.push(nm.clone());
                        let r = Reg {
                            file: File::V,
                            idx: (vreg_names.len() - 1) as u16,
                        };
                        if is_b128 {
                            vreg_names.push(format!("{nm}.hi"));
                        }
                        r
                    };
                    regs.insert(nm, r);
                }
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix(".shared") {
            let (name, align, size) = shared_decl(rest)?;
            smem = smem.div_ceil(align) * align;
            shared_syms.insert(name, smem);
            smem += size;
            continue;
        }
        // `.local .align A .b8 __local_depot0[N]` (the compiler's stack frame, addressed through
        // `%SPL`): per-thread local memory, as SASS LDL / STL (offsets from 0).
        if let Some(rest) = t.strip_prefix(".local") {
            let (name, align, size) = shared_decl(rest)?;
            local_bytes = local_bytes.div_ceil(align) * align;
            local_syms.insert(name, local_bytes);
            local_bytes += size;
            continue;
        }
        if t.starts_with('.') {
            continue; // other directives (.maxntid, .pragma, ...) do not affect semantics here
        }
        insts_src.push(st);
    }

    for (n, a) in shared_syms_pre {
        shared_syms.insert(n, a);
    }
    for (n, align) in externs {
        smem = smem.div_ceil(align.max(16)) * align.max(16);
        shared_syms.insert(n, smem);
    }

    // Scratch registers for multi-op expansions.
    let mut tmp_v = None;
    if insts_src.iter().any(|st| needs_scratch(&st.text)) {
        vreg_names.push("%__x64".to_string());
        tmp_v = Some(Reg {
            file: File::V,
            idx: (vreg_names.len() - 1) as u16,
        });
    }
    let mut vec_v = None;
    if insts_src
        .iter()
        .any(|st| needs_vec_scratch(&st.text, &regs))
    {
        vec_v = Some(Reg {
            file: File::V,
            idx: vreg_names.len() as u16,
        });
        for i in 0..VEC_SCRATCH {
            vreg_names.push(format!("%__vec{i}"));
        }
    }
    // The numerics track the high half of a 64-bit register under `idx | 0x8000`
    // (sim::num::V_HI): indices stay below it.
    if vreg_names.len() > 0x8000 {
        bail!("kernel {name}: more than 32768 virtual registers");
    }

    let f16_mul_ok = uncontractible_f16_muls(&insts_src);
    // Pass 2: instructions. A statement may lower to several ops (`Lower::extra`); branch
    // targets and labels are statement indices until remapped below.
    let mut code = Vec::new();
    let mut start = Vec::with_capacity(insts_src.len() + 1);
    for st in &insts_src {
        let mut lw = Lower {
            regs: &regs,
            labels: &labels,
            shared_syms: &shared_syms,
            local_syms: &local_syms,
            param_syms: &param_syms,
            line: st.line,
            text: &st.text,
            extra: Vec::new(),
            tmp_v,
            pre: Vec::new(),
            vec_v,
            f16_mul_ok: &f16_mul_ok,
        };
        let inst = lower_inst(&mut lw).with_context(|| format!("in kernel {name}"))?;
        start.push(code.len());
        let (guard, loc, text) = (inst.guard, inst.loc, inst.text.clone());
        let mut ops = lw.pre.clone();
        ops.push(inst.op.clone());
        ops.extend(lw.extra.iter().cloned());
        // Every op of an expansion runs under the statement's guard, so none but the last
        // may write it.
        if let Some(g) = guard.reg {
            if ops[..ops.len() - 1]
                .iter()
                .any(|o| o.dst_regs().contains(&g))
            {
                return Err(lw.err("expansion writes its own guard predicate"))
                    .with_context(|| format!("in kernel {name}"));
            }
        }
        for op in lw.pre {
            code.push(Inst {
                guard,
                op,
                uniform: false,
                loc,
                text: text.clone(),
            });
        }
        code.push(inst);
        for op in lw.extra {
            code.push(Inst {
                guard,
                op,
                uniform: false,
                loc,
                text: text.clone(),
            });
        }
    }
    start.push(code.len());
    if code.len() != insts_src.len() {
        for i in code.iter_mut() {
            if let Op::Bra { target, .. } = &mut i.op {
                *target = start[*target];
            }
        }
        for v in labels.values_mut() {
            *v = start[*v];
        }
    }

    // `.reqntid x[, y[, z]]` / `.maxntid ...` between the parameter list and the body.
    let ntid = {
        let head = &src[entry_at..body_start];
        let head = head.rsplit_once(')').map_or(head, |x| x.1);
        let dir = |d: &str| -> Option<[u32; 3]> {
            let i = head.find(d)? + d.len();
            let rest = &head[i..];
            let end = rest.find(['.', '{', '\n']).unwrap_or(rest.len());
            let v: Vec<u32> = rest[..end]
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            (!v.is_empty() && v.len() <= 3).then(|| {
                [
                    v[0],
                    v.get(1).copied().unwrap_or(1),
                    v.get(2).copied().unwrap_or(1),
                ]
            })
        };
        dir(".reqntid")
            .map(|dims| Ntid { req: true, dims })
            .or_else(|| dir(".maxntid").map(|dims| Ntid { req: false, dims }))
    };
    Ok(Kernel {
        name,
        front: Front::Ptx,
        code,
        nregs: [vreg_names.len() as u16, 0, 0, npred, 0],
        params,
        param_cbank: 0x380,
        smem_bytes: smem,
        vreg_names,
        labels: {
            let mut l: Vec<(usize, String)> = labels.iter().map(|(n, i)| (*i, n.clone())).collect();
            l.sort();
            l
        },
        static_findings: vec![],
        bulk_read_waits: vec![],
        bulk_commit_sb: vec![],
        parts: vec![],
        ntid,
        globals: vec![],
        global_relocs: vec![],
    })
}

fn lower_inst(lw: &mut Lower) -> Result<Inst> {
    let text = lw.text.trim();
    let (guard, rest) = if let Some(r) = text.strip_prefix('@') {
        let (g, rest) = r
            .split_once(char::is_whitespace)
            .ok_or_else(|| lw.err("bad guard"))?;
        let (neg, g) = match g.strip_prefix('!') {
            Some(x) => (true, x),
            None => (false, g),
        };
        (
            Pred {
                reg: Some(lw.pred_dst(g)?),
                neg,
            },
            rest.trim(),
        )
    } else {
        (Pred::TRUE, text)
    };
    let (opcode, ops_src) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let ops = split_operands(ops_src);
    let parts: Vec<&str> = opcode.split('.').collect();
    let base = parts[0];
    let mods = &parts[1..];
    let has = |m: &str| mods.contains(&m);
    let ty = mods
        .iter()
        .rev()
        .find_map(|m| type_bits(m).map(|b| (b, *m)));
    let bits = ty.map(|t| t.0).unwrap_or(32);
    let tname = ty.map(|t| t.1).unwrap_or("b32");
    let mut extra: Vec<Op> = Vec::new();
    let op_n = |i: usize| -> Result<&str> {
        ops.get(i)
            .map(|s| s.as_str())
            .ok_or_else(|| lw.err(format!("missing operand {i}")))
    };

    // The 16-bit float type of an arithmetic op (`bf16x2` / `f16x2` are not in `type_bits`).
    let ht = mods.last().copied().unwrap_or("");
    let half_t = matches!(ht, "bf16" | "bf16x2" | "f16" | "f16x2");
    let op = match base {
        // Agent L2's forms (l2.rs): cvt.r*i between equal float types, cvt f64 <-> f16 /
        // bf16 and f16 <-> bf16, min / max .u16x2 / .s16x2, cp.reduce.async.bulk .add.f32.
        _ if l2::claims(base, mods) => l2::lower(lw, base, mods, &ops)?,
        // Packed / scalar bf16 arithmetic as the SASS ptxas emits (`HFMA2.BF16_V2`, `HMUL2`,
        // `HADD2`; probes/bf16x2, both front-ends): `fma.rn[.relu]`, `mul.rn`, `add.rn`,
        // `sub.rn` .bf16x2 / .bf16 and `neg.bf16x2` / `.bf16` (-0*0 - x: a NaN becomes 0x7fff).
        // Without `.rn` ptxas may contract a mul and an add into one HFMA2: refused, as are
        // immediates, `.ftz` / `.sat` / `.oob` and f16 arithmetic. ASSUME(N-BF16X2)
        // Mixed precision `fma` / `add` / `sub` `.{rn,rz,rm,rp}[.sat].f32.bf16`: SASS FHFMA.BF16 /
        // FHADD.BF16 (probes/bf16_ext, both front-ends). ASSUME(N-BF16X2)
        "add" | "sub" | "fma"
            if ht == "bf16" && mods.len() >= 2 && mods[mods.len() - 2] == "f32" =>
        {
            let rest = &mods[..mods.len() - 2];
            let rnd = match rest.iter().find(|m| **m != "sat") {
                Some(&"rn") => Rnd::Rn,
                Some(&"rz") => Rnd::Rz,
                Some(&"rm") => Rnd::Rm,
                Some(&"rp") => Rnd::Rp,
                _ => bail!(lw.err(format!("{base}.f32.bf16 needs a rounding mode"))),
            };
            if rest.len() > 2 || rest.len() == 2 && !rest.contains(&"sat") {
                bail!(lw.err(format!(
                    "{base}.f32.bf16 with modifiers {rest:?} not modeled"
                )));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.f32.bf16 immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            let (a, b, c) = if base == "fma" {
                (reg(1)?, reg(2)?, reg(3)?)
            } else {
                (reg(1)?, Val::Imm(0x3f80), reg(2)?)
            };
            Op::FfmaBf16 {
                d,
                a,
                b,
                c,
                hi: [false; 2],
                neg: [false, false, base == "sub"],
                rnd,
                sat: has("sat"),
            }
        }
        // `min` / `max` [.NaN] [.xorsign.abs] `.bf16x2` / `.bf16`: SASS HMNMX2.BF16_V2
        // (probes/bf16_ext). ASSUME(N-BF16X2)
        "min" | "max" if matches!(ht, "bf16" | "bf16x2") => {
            let rest = &mods[..mods.len() - 1];
            if rest
                .iter()
                .any(|m| !matches!(*m, "NaN" | "xorsign" | "abs"))
                || has("xorsign") != has("abs")
            {
                bail!(lw.err(format!("{base}.{ht} with modifiers {rest:?} not modeled")));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.{ht} immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            let op = Op::Hmnmx2Bf16 {
                d,
                a: reg(1)?,
                b: reg(2)?,
                sel: [HSel::Pair; 2],
                min: Pred {
                    reg: None,
                    neg: base == "max",
                },
                nan: has("NaN"),
                xorsign: has("xorsign"),
            };
            if ht == "bf16" {
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
            }
            op
        }
        // f16 arithmetic as ptxas emits it (SASS `HFMA2` / `HMUL2` / `HADD2`, probes/f16_arith):
        // `fma.rn`, `mul.rn`, `add.rn`, `sub.rn` `.f16` / `.f16x2`, and `mul.f16[x2]` without
        // `.rn` where no use lets ptxas contract it (`uncontractible_f16_muls`). Register
        // operands only; `.ftz` / `.sat` / `.relu` are refused. ASSUME(L-ROSETTA)
        "add" | "sub" | "mul" | "fma" if matches!(ht, "f16" | "f16x2") => {
            let rest = &mods[..mods.len() - 1];
            let d_name = op_n(0)?.trim();
            let ok = rest == ["rn"]
                || base == "mul" && rest.is_empty() && lw.f16_mul_ok.contains(d_name);
            if !ok {
                bail!(lw.err(format!(
                    "{base}.{ht} with modifiers {rest:?} not modeled (only .rn, or a mul ptxas cannot contract)"
                )));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.{ht} immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            let (a, b, c, neg_c) = match base {
                "mul" => (reg(1)?, reg(2)?, Val::Imm(0x8000_8000), false),
                "add" | "sub" => (reg(1)?, Val::Imm(0x3c00_3c00), reg(2)?, base == "sub"),
                _ => (reg(1)?, reg(2)?, reg(3)?, false),
            };
            let op = Op::Hfma2 {
                d,
                a,
                b,
                c,
                neg_a: false,
                neg_c,
                sel: [HSel::Pair; 3],
            };
            if ht == "f16" {
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
            }
            op
        }
        // `abs.bf16x2` / `.bf16`: -0*0 + |x| (SASS `HFMA2.BF16_V2 d, -RZ.H0_H0, RZ.H0_H0, |R|`;
        // a NaN becomes 0x7fff).
        "add" | "sub" | "mul" | "fma" | "neg" | "abs" if matches!(ht, "bf16" | "bf16x2") => {
            let rest = &mods[..mods.len() - 1];
            let ok = match base {
                "neg" | "abs" => rest.is_empty(),
                "fma" => rest == ["rn"] || rest == ["rn", "relu"],
                _ => rest == ["rn"],
            };
            if !ok {
                bail!(lw.err(format!(
                    "{base}.{ht} with modifiers {rest:?} not modeled (only .rn, fma .rn.relu)"
                )));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.{ht} immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            let (a, b, c, neg) = match base {
                "mul" => (reg(1)?, reg(2)?, Val::Imm(0x8000_8000), [false; 3]),
                "add" | "sub" => (
                    reg(1)?,
                    Val::Imm(0x3f80_3f80),
                    reg(2)?,
                    [false, false, base == "sub"],
                ),
                "neg" => (
                    Val::Imm(0x8000_8000),
                    Val::Imm(0),
                    reg(1)?,
                    [false, false, true],
                ),
                "abs" => (Val::Imm(0x8000_8000), Val::Imm(0), reg(1)?, [false; 3]),
                _ => (reg(1)?, reg(2)?, reg(3)?, [false; 3]),
            };
            let op = Op::Hfma2Bf16 {
                d,
                a,
                b,
                c,
                sel: [HSel::Pair; 3],
                neg,
                abs: [false, false, base == "abs"],
                relu: has("relu"),
            };
            if ht == "bf16" {
                // A 16-bit register: the high lane (computed from the operands' high bits)
                // cleared.
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
            }
            op
        }
        "add" | "sub" | "mul" | "fma" | "neg" | "abs" | "min" | "max" if half_t => {
            bail!(lw.err(format!("{base}.{ht} not modeled")))
        }
        "mov" if op_n(0)?.trim().starts_with('{') => {
            let halves: Vec<&str> = op_n(0)?
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| x.trim())
                .collect();
            if !(halves.len() == 2 && (bits == 64 || bits == 32) || halves.len() == 4 && bits == 32)
            {
                bail!(lw.err(
                    "only mov.b64 {lo, hi} / mov.b32 {lo, hi} / mov.b32 {b0, b1, b2, b3} unpacking is modeled"
                ));
            }
            let r = |x: &str| -> Result<Option<Reg>> {
                if x == "_" {
                    Ok(None)
                } else {
                    lw.dst(x).map(Some)
                }
            };
            if bits == 32 {
                // `mov.b32 {h0, h1}, r`: two 16-bit halves, as PRMT with zero upper bytes.
                let a = lw.val(op_n(1)?)?;
                let half = |d: Reg, sel: u64| Op::Prmt {
                    d,
                    a: a.clone(),
                    sel: Val::Imm(sel),
                    b: Val::Imm(0),
                };
                // `mov.b32 {b0, b1, b2, b3}, r`: four bytes, each zero-extended.
                let sels: &[u64] = if halves.len() == 4 {
                    &[0x4440, 0x4441, 0x4442, 0x4443]
                } else {
                    &[0x4410, 0x4432]
                };
                let mut ops: Vec<Op> = halves
                    .iter()
                    .zip(sels)
                    .map(|(h, &sel)| Ok((r(h)?, sel)))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter_map(|(d, sel)| d.map(|d| half(d, sel)))
                    .collect();
                if ops.is_empty() {
                    bail!(lw.err("mov.b32 {_, _} has no destination"));
                }
                let first = ops.remove(0);
                extra.extend(ops);
                first
            } else {
                Op::Unpack64 {
                    lo: r(halves[0])?,
                    hi: r(halves[1])?,
                    a: lw.val(op_n(1)?)?,
                }
            }
        }
        "mov" if op_n(1)?.trim().starts_with('{') => {
            let halves: Vec<&str> = op_n(1)?
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| x.trim())
                .collect();
            if halves.len() != 2 || (bits != 64 && bits != 32) {
                bail!(lw.err("only mov.b64 / mov.b32 %d, {lo, hi} packing is modeled"));
            }
            if bits == 32 {
                // `mov.b32 r, {h0, h1}`: r = h1 << 16 | h0 (PRMT bytes a0 a1 b0 b1).
                Op::Prmt {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(halves[0])?,
                    sel: Val::Imm(0x5410),
                    b: lw.val(halves[1])?,
                }
            } else {
                Op::Pack64 {
                    d: lw.dst(op_n(0)?)?,
                    lo: lw.val(halves[0])?,
                    hi: lw.val(halves[1])?,
                }
            }
        }
        "mov" if !has("pred") => Op::Mov {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
        },
        // Packed f32x2 (sm_100+): `fma.rn.f32x2`, `mul.rn.f32x2`, `add.rn.f32x2`, `sub.rn
        // .f32x2` on .b64 registers, lane 0 in the low 32 bits. ptxas emits FFMA2 / FMUL2 /
        // FADD2: one rounding per lane, subnormals kept, canonical NaN (fixtures/s_ptx_packed
        // = GB300; `.rz` / `.rm` / `.rp`: FFMA2.RZ etc., probes/bf16_ext). Without a rounding
        // mode ptxas may contract mul + add into one FFMA2; that, .ftz and .sat are refused.
        // ASSUME(L-ROSETTA)
        "add" | "sub" | "mul" | "fma" if mods.last() == Some(&"f32x2") => {
            let rest = &mods[..mods.len() - 1];
            let rnd = match rest {
                ["rn"] => Rnd::Rn,
                ["rz"] => Rnd::Rz,
                ["rm"] => Rnd::Rm,
                ["rp"] => Rnd::Rp,
                _ => bail!(lw.err(format!(
                    "{base}.f32x2 with modifiers {rest:?} not modeled (one of .rn / .rz / .rm / .rp)"
                ))),
            };
            let reg = |s: &str| -> Result<(Val, F2Lanes)> {
                match lw.val(s)? {
                    v @ Val::Reg(..) => Ok((v, F2Lanes::Pair)),
                    _ => bail!(lw.err(format!("{base}.f32x2 operand `{s}` must be a register"))),
                }
            };
            let d = lw.dst(op_n(0)?)?;
            let (a, b) = (reg(op_n(1)?)?, reg(op_n(2)?)?);
            let one = (Val::Imm(0x3f80_0000), F2Lanes::Bcast);
            match base {
                // a*1 + b: the rounded sum (as FADD2); sub negates b.
                "add" | "sub" => Op::Ffma2 {
                    d,
                    a,
                    b: one,
                    c: b,
                    neg: [false, false, base == "sub"],
                    rnd,
                },
                // a*b + (-0): the rounded product (as FMUL2; +0 under .rm).
                "mul" => Op::Ffma2 {
                    d,
                    a,
                    b,
                    c: (
                        Val::Imm(if rnd == Rnd::Rm { 0 } else { 0x8000_0000 }),
                        F2Lanes::Bcast,
                    ),
                    neg: [false; 3],
                    rnd,
                },
                _ => Op::Ffma2 {
                    d,
                    a,
                    b,
                    c: reg(op_n(3)?)?,
                    neg: [false; 3],
                    rnd,
                },
            }
        }
        // f64 arithmetic: as ptxas lowers it (DADD a, -b for sub; DMUL; DFMA).
        "add" | "sub" | "mul" | "fma" if tname == "f64" => {
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            let (a, b) = (lw.val(op_n(1)?)?, lw.val(op_n(2)?)?);
            let d = lw.dst(op_n(0)?)?;
            match base {
                "add" | "sub" => Op::Dfma {
                    d,
                    a,
                    b: Val::Imm(1f64.to_bits()),
                    c: Some(b),
                    neg: [false, false, base == "sub"],
                    abs: [false; 3],
                    rnd,
                },
                "mul" => Op::Dfma {
                    d,
                    a,
                    b,
                    c: None,
                    neg: [false; 3],
                    abs: [false; 3],
                    rnd,
                },
                _ => Op::Dfma {
                    d,
                    a,
                    b,
                    c: Some(lw.val(op_n(3)?)?),
                    neg: [false; 3],
                    abs: [false; 3],
                    rnd,
                },
            }
        }
        "div" | "sqrt" if tname == "f64" && ["rn", "rz", "rm", "rp"].iter().any(|m| has(m)) => {
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            Op::DivSqrt64 {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: if base == "div" {
                    Some(lw.val(op_n(2)?)?)
                } else {
                    None
                },
                rnd,
            }
        }
        "add" | "sub" if tname.starts_with('f') => {
            if bits != 32 {
                bail!(lw.err("only add.f32 / sub.f32 are modeled so far"));
            }
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            Op::Fadd {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                rnd,
                neg: [false, base == "sub"],
                abs: [false; 2],
                ftz: has("ftz"),
                sat: has("sat"),
            }
        }
        // mul.f32 / fma.rn.f32 (one rounding; mul is the rounded product).
        "mul" | "fma" if tname == "f32" && !has("wide") && !has("hi") => {
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            Op::Ffma {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                c: if base == "fma" {
                    Some(lw.val(op_n(3)?)?)
                } else {
                    None
                },
                neg: [false; 3],
                abs: [false; 3],
                rnd,
                ftz: has("ftz"),
                sat: has("sat"),
            }
        }
        // ASSUME(N-MUFU): see ASSUMPTIONS.md
        "ex2" | "rcp" | "rsqrt" | "lg2" | "sin" | "cos" | "tanh" | "sqrt" if has("approx") => {
            let func = match base {
                "ex2" => MufuFunc::Ex2,
                "rcp" => MufuFunc::Rcp,
                "rsqrt" => MufuFunc::Rsq,
                "lg2" => MufuFunc::Lg2,
                "sin" => MufuFunc::Sin,
                "cos" => MufuFunc::Cos,
                "tanh" => MufuFunc::Tanh,
                _ => MufuFunc::Sqrt,
            };
            let fmt = match mods.last().copied() {
                Some("f32") => MufuFmt::F32,
                Some("f16") => MufuFmt::F16,
                Some("bf16") => MufuFmt::Bf16,
                Some("f16x2") => MufuFmt::F16x2,
                Some("bf16x2") => MufuFmt::Bf16x2,
                // `rsqrt.approx.f64` (no .ftz) adds a Newton step and a slow path: not modeled.
                Some("f64") if has("ftz") && matches!(base, "rcp" | "rsqrt") => MufuFmt::F64,
                _ => bail!(lw.err(format!("{base}.approx of this type is not modeled"))),
            };
            let func = match (func, fmt) {
                (MufuFunc::Rcp, MufuFmt::F64) => MufuFunc::Rcp64h,
                (MufuFunc::Rsq, MufuFmt::F64) => MufuFunc::Rsq64h,
                (f, MufuFmt::F32 | MufuFmt::F64) => f,
                (f @ (MufuFunc::Ex2 | MufuFunc::Tanh), _) => f,
                _ => bail!(lw.err(format!("{base}.approx of this type is not modeled"))),
            };
            Op::Mufu {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                func,
                fmt,
                hi: false,
                neg: false,
                abs: false,
                ptx: Some(has("ftz")),
            }
        }
        // Predicate logic: and/or/xor/not/mov on .pred (a = 0xF0, b = 0xCC LUT convention).
        "and" | "or" | "xor" | "not" | "mov" if has("pred") => {
            let d = lw.pred_dst(op_n(0)?)?;
            let src = |i: usize| -> Result<Pred> {
                let o = op_n(i)?;
                match parse_int(o) {
                    Some(0) => Ok(Pred {
                        reg: None,
                        neg: true,
                    }),
                    Some(_) => Ok(Pred::TRUE),
                    None => lw.pred_src(o),
                }
            };
            let a = src(1)?;
            let (b, lut) = match base {
                "and" => (src(2)?, 0xc0),
                "or" => (src(2)?, 0xfc),
                "xor" => (src(2)?, 0x3c),
                "not" => (Pred::TRUE, 0x0f),
                _ => (Pred::TRUE, 0xf0),
            };
            Op::Plop3 {
                d,
                a,
                b,
                c: Pred::TRUE,
                lut,
                d2: None,
            }
        }
        "selp" => {
            // 32-bit selp of an immediate: the register gets 32 bits (exec keeps all 64 bits
            // of a virtual register, for selp.b64).
            let w = |v: Val| match v {
                Val::Imm(x) if bits <= 32 => Val::Imm(x & 0xffff_ffff),
                v => v,
            };
            Op::Sel {
                d: lw.dst(op_n(0)?)?,
                a: w(lw.val(op_n(1)?)?),
                b: w(lw.val(op_n(2)?)?),
                p: lw.pred_src(op_n(3)?)?,
            }
        }
        "bfe" => Op::Bfe {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            pos: lw.val(op_n(2)?)?,
            len: lw.val(op_n(3)?)?,
            signed: tname.starts_with('s'),
        },
        // min.f32 / max.f32 [.ftz] [.NaN] (2 or 3 sources), as SASS FMNMX / FMNMX3;
        // `.xorsign.abs` (two sources): FMNMX.XORSIGN (probes/fmnmx_xorsign). ASSUME(N-XORSIGN)
        "min" | "max" if tname == "f32" => {
            let n = (1..4).take_while(|&i| op_n(i).is_ok()).count();
            if let Some(m) = mods
                .iter()
                .find(|m| !matches!(**m, "ftz" | "NaN" | "xorsign" | "abs" | "f32"))
            {
                bail!(lw.err(format!("{base}.f32: modifier .{m} not modeled")));
            }
            let xorsign = has("xorsign");
            if xorsign != has("abs") || (xorsign && n != 2) {
                bail!(lw.err(format!("{base}.f32: .xorsign.abs form not modeled")));
            }
            Op::Fmnmx {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                c: if n == 3 {
                    Some(lw.val(op_n(3)?)?)
                } else {
                    None
                },
                neg: [false; 3],
                abs: [false; 3],
                min: Pred {
                    reg: None,
                    neg: base == "max",
                },
                ftz: has("ftz"),
                nan: has("NaN"),
                xorsign,
            }
        }
        // `not.b16` / `.b32` / `.b64`: x XOR all ones of the width.
        "not" if matches!(tname, "b16" | "b32" | "b64") => Op::Int {
            op: IntOp::Xor,
            bits,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            }),
        },
        // `add.u16x2` / `add.s16x2` (SASS VIADD.16x2; probes/viadd16x2): two 16-bit lanes,
        // each modulo 2^16. The .sat form (s16x2 only) is not modeled.
        "add" if matches!(mods, ["u16x2" | "s16x2"]) => Op::Int {
            op: IntOp::Add16x2,
            bits: 32,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        // `popc.b32` (SASS POPC; probes/int_misc). `.b64` is not modeled.
        "popc" if mods == ["b32"] => Op::Int {
            op: IntOp::Popc,
            bits: 32,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(0),
        },
        // `bfind[.shiftamt].u32` (SASS FLO.U32[.SH]; probes/flo_popc): 0xffffffff for 0.
        // `.s32` / `.u64` are not modeled.
        "bfind" if mods == ["u32"] || mods == ["shiftamt", "u32"] => Op::Flo {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            sh: mods.len() == 2,
        },
        "bmsk" => Op::Int {
            op: if has("wrap") {
                IntOp::BmskWrap
            } else {
                IntOp::Bmsk
            },
            bits: 32,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        "dp2a" => Op::Dp2a {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            c: lw.val(op_n(3)?)?,
            hi: has("hi"),
            a_signed: mods.get(mods.len().wrapping_sub(2)) == Some(&"s32"),
            b_signed: mods.last() == Some(&"s32"),
        },
        // div.full.f32 (Triton's softmax / layer norm divide): ptxas's RCP expansion.
        "div" if matches!(mods, ["full" | "rn", "f32"]) => Op::DivFull {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            rn: has("rn"),
        },
        "div" | "rem" | "min" | "max" if !tname.starts_with('f') => {
            let s = tname.starts_with('s');
            let op = match (base, s) {
                ("div", false) => IntOp::DivU,
                ("div", true) => IntOp::DivS,
                ("rem", false) => IntOp::RemU,
                ("rem", true) => IntOp::RemS,
                ("min", false) => IntOp::MinU,
                ("min", true) => IntOp::MinS,
                ("max", false) => IntOp::MaxU,
                _ => IntOp::MaxS,
            };
            Op::Int {
                op,
                bits,
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
            }
        }
        "shfl" => {
            let mode = if has("idx") {
                ShflMode::Idx
            } else if has("up") {
                ShflMode::Up
            } else if has("down") {
                ShflMode::Down
            } else if has("bfly") {
                ShflMode::Bfly
            } else {
                bail!(lw.err("shfl mode"));
            };
            if !has("sync") || parse_int(op_n(4)?) != Some(u32::MAX as u64) && op_n(4)? != "-1" {
                bail!(lw.err("only shfl.sync with a full member mask is modeled"));
            }
            // `d|p`: the in-range predicate.
            let (dd, pp) = match op_n(0)?.split_once('|') {
                Some((d, p)) => (d.trim(), Some(lw.pred_dst(p.trim())?)),
                None => (op_n(0)?, None),
            };
            Op::Shfl {
                p: pp,
                d: lw.dst(dd)?,
                a: lw.val(op_n(1)?)?,
                lane: lw.val(op_n(2)?)?,
                clamp: lw.val(op_n(3)?)?,
                mode,
            }
        }
        // `ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16 {d..}, [a]` (SASS LDSM).
        "ldmatrix" => {
            if !(has("m8n8") && has("b16"))
                || mods.iter().any(|m| {
                    !matches!(
                        *m,
                        "sync"
                            | "aligned"
                            | "m8n8"
                            | "x1"
                            | "x2"
                            | "x4"
                            | "trans"
                            | "shared"
                            | "b16"
                    )
                })
            {
                bail!(
                    lw.err(
                        "only ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16 is modeled"
                    )
                );
            }
            let num: u8 = mods
                .iter()
                .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse().ok()))
                .ok_or_else(|| lw.err("missing .xN"))?;
            let (d, n, moves) = lw.reg_vec_mem(op_n(0)?, true)?;
            if n != num {
                bail!(lw.err("register count does not match .xN"));
            }
            extra.extend(moves);
            Op::LdMatrix {
                d,
                n,
                addr: lw.addr(op_n(1)?)?,
                trans: has("trans"),
                b8: false,
            }
        }
        "stmatrix" => {
            if !(has("m8n8") && has("b16")) {
                bail!(lw.err("only stmatrix.m8n8.b16 is modeled"));
            }
            let num: usize = mods
                .iter()
                .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse().ok()))
                .ok_or_else(|| lw.err("missing .xN"))?;
            let v: Vec<Val> = op_n(1)?
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .split(',')
                .map(|r| lw.val(r))
                .collect::<Result<_>>()?;
            if v.len() != num {
                bail!(lw.err("register count does not match .xN"));
            }
            Op::StMatrix {
                addr: lw.addr(op_n(0)?)?,
                v,
                trans: has("trans"),
            }
        }
        "add" | "sub" | "and" | "or" | "xor" | "shl" | "shr" => {
            let op = match base {
                "add" => IntOp::Add,
                "sub" => IntOp::Sub,
                "and" => IntOp::And,
                "or" => IntOp::Or,
                "xor" => IntOp::Xor,
                "shl" => IntOp::Shl,
                _ if tname.starts_with('s') => IntOp::ShrS,
                _ => IntOp::ShrU,
            };
            Op::Int {
                op,
                bits,
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
            }
        }
        // `neg.f32 d, a` (no .ftz): what ptxas emits, `FADD d, -a, -RZ` (exact for every non-NaN
        // value, -0 + -0 = -0 included; a NaN gives the canonical NaN; fixtures/e_ptx_redux).
        "neg" if tname == "f32" && !has("ftz") => Op::Fadd {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(0x8000_0000),
            rnd: Rnd::Rn,
            neg: [true, false],
            abs: [false; 2],
            ftz: false,
            sat: false,
        },
        // `shf.{l,r}.{clamp,wrap}.b32 d, a, b, c`: funnel shift of b:a by c, clamped to 32 or
        // mod 32 (GB300: fixtures/s_shf). ASSUME(L-SHF)
        "shf" => {
            let wrap = match (has("clamp"), has("wrap")) {
                (true, false) => false,
                (false, true) => true,
                _ => bail!(lw.err("shf needs exactly one of .clamp / .wrap")),
            };
            if has("l") == has("r") || bits != 32 {
                bail!(lw.err("shf form not modeled"));
            }
            Op::Shf {
                d: lw.dst(op_n(0)?)?,
                lo: lw.val(op_n(1)?)?,
                hi: lw.val(op_n(2)?)?,
                sh: lw.val(op_n(3)?)?,
                right: has("r"),
                signed: false,
                high: has("l"),
                wide: false,
                wrap,
            }
        }
        // `neg.s32 d, a` = 0 - a.
        "neg" if tname.starts_with('s') => Op::Int {
            op: IntOp::Sub,
            bits,
            d: lw.dst(op_n(0)?)?,
            a: Val::Imm(0),
            b: lw.val(op_n(1)?)?,
        },
        "mul" if has("wide") => Op::MulWide {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            signed: tname.starts_with('s'),
        },
        "mul" if has("hi") => Op::Int {
            op: if tname.starts_with('s') {
                IntOp::MulHiS
            } else {
                IntOp::MulHiU
            },
            bits,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        "mad" if has("lo") && bits == 32 => Op::Imad {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            c: lw.val(op_n(3)?)?,
            neg_c: false,
            not_c: false,
            wide: false,
            hi: false,
            signed: false,
            cin: None,
            cout: None,
        },
        // `mad.lo.{s,u}64 d, a, b, c` (CuTeDSL's 64-bit address arithmetic): d = a*b + c mod
        // 2^64, through a scratch register (d may alias c).
        "mad" if has("lo") && bits == 64 && !tname.starts_with('f') => {
            if !matches!(mods, ["lo", "s64" | "u64"]) {
                bail!(lw.err("mad: only mad.lo.{s64,u64} is modeled in 64 bits"));
            }
            let tmp = lw.scratch_v()?;
            extra.push(Op::Int {
                op: IntOp::Add,
                bits: 64,
                d: lw.dst(op_n(0)?)?,
                a: Val::Reg(tmp, 1),
                b: lw.val(op_n(3)?)?,
            });
            Op::Int {
                op: IntOp::Mul,
                bits: 64,
                d: tmp,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
            }
        }
        // `mad.wide.s32 d, a, b, c` (Triton's address arithmetic): 32x32 -> 64 plus a 64-bit c.
        "mad" if has("wide") && bits == 32 => Op::Imad {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            c: lw.val(op_n(3)?)?,
            neg_c: false,
            not_c: false,
            wide: true,
            hi: false,
            signed: tname.starts_with('s'),
            cin: None,
            cout: None,
        },
        "prefetch" => Op::Nop, // a cache hint
        "mul" if has("lo") || !tname.starts_with('f') => Op::Int {
            op: IntOp::Mul,
            bits,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        // setp.<cmp>[.ftz].f32 p, a, b (one predicate, no combining operand) as SASS
        // `FSETP.<cmp>[.FTZ].AND p, PT, a, b, PT`: the ordered compares are false and the `u`
        // ones true when an operand is NaN.
        // `setp.<cmp>.f16 / .bf16 p, a, b` and `.f16x2 / .bf16x2 p|q, a, b` (SASS HSETP2
        // [.BF16_V2], probes/hsetp2): per lane the exact f32 widening compared; no `.ftz`, no
        // combining predicate.
        "setp" if matches!(ht, "f16" | "f16x2" | "bf16" | "bf16x2") => {
            let cmp = match mods[0] {
                "eq" => FCmp::Eq,
                "ne" => FCmp::Ne,
                "lt" => FCmp::Lt,
                "le" => FCmp::Le,
                "gt" => FCmp::Gt,
                "ge" => FCmp::Ge,
                "equ" => FCmp::Equ,
                "neu" => FCmp::Neu,
                "ltu" => FCmp::Ltu,
                "leu" => FCmp::Leu,
                "gtu" => FCmp::Gtu,
                "geu" => FCmp::Geu,
                "num" => FCmp::Num,
                "nan" => FCmp::Nan,
                c => bail!(lw.err(format!("unsupported setp compare {c}"))),
            };
            if mods.len() != 2 || ops.len() != 3 {
                bail!(lw.err(format!(
                    "setp.{} form not modeled (no .ftz, no combining predicate)",
                    mods.join(".")
                )));
            }
            let x2 = ht.ends_with("x2");
            let (d, q) = match op_n(0)?.split_once('|') {
                Some((p, q)) if x2 => (lw.pred_dst(p.trim())?, Some(lw.pred_dst(q.trim())?)),
                None if !x2 => (lw.pred_dst(op_n(0)?)?, None),
                _ => bail!(lw.err("setp half destination form")),
            };
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err("setp half immediate operand not modeled"));
                }
                Ok(v)
            };
            let sel = if x2 { HSel::Pair } else { HSel::H0 };
            Op::Hsetp2Bf16 {
                d,
                q,
                cmp,
                a: reg(1)?,
                b: reg(2)?,
                sel: [sel; 2],
                neg: [false; 2],
                abs: [false; 2],
                c: Pred::TRUE,
                bop: BoolOp::And,
                f16: ht.starts_with("f16"),
            }
        }
        "setp" if tname == "f32" => {
            let cmp = match mods[0] {
                "eq" => FCmp::Eq,
                "ne" => FCmp::Ne,
                "lt" => FCmp::Lt,
                "le" => FCmp::Le,
                "gt" => FCmp::Gt,
                "ge" => FCmp::Ge,
                "equ" => FCmp::Equ,
                "neu" => FCmp::Neu,
                "ltu" => FCmp::Ltu,
                "leu" => FCmp::Leu,
                "gtu" => FCmp::Gtu,
                "geu" => FCmp::Geu,
                "num" => FCmp::Num,
                "nan" => FCmp::Nan,
                c => bail!(lw.err(format!("unsupported setp compare {c}"))),
            };
            if !matches!(mods.len(), 2 | 3) || mods.len() == 3 && mods[1] != "ftz" {
                bail!(lw.err("setp.f32 modifiers not modeled"));
            }
            if op_n(0)?.contains('|') || op_n(3).is_ok() {
                bail!(lw.err("setp.f32 with p|q or a combining predicate not modeled"));
            }
            Op::Fsetp {
                d: lw.pred_dst(op_n(0)?)?,
                cmp,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                neg: [false; 2],
                abs: [false; 2],
                c: Pred::TRUE,
                bop: BoolOp::And,
                ftz: has("ftz"),
            }
        }
        "setp" => {
            let s = tname.starts_with('s');
            let cmp = match (mods[0], s) {
                ("eq", _) => Cmp::Eq,
                ("ne", _) => Cmp::Ne,
                ("lt", false) | ("lo", _) => Cmp::LtU,
                ("le", false) | ("ls", _) => Cmp::LeU,
                ("gt", false) | ("hi", _) => Cmp::GtU,
                ("ge", false) | ("hs", _) => Cmp::GeU,
                ("lt", true) => Cmp::LtS,
                ("le", true) => Cmp::LeS,
                ("gt", true) => Cmp::GtS,
                ("ge", true) => Cmp::GeS,
                (c, _) => bail!(lw.err(format!("unsupported setp compare {c}"))),
            };
            if tname.starts_with('f') {
                bail!(lw.err("float setp not modeled yet"));
            }
            Op::Setp {
                d: lw.pred_dst(op_n(0)?)?,
                cmp,
                bits,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                and: Pred::TRUE,
                or: false,
                xor: false,
                ex: None,
            }
        }
        "bra" => {
            let l = op_n(0)?;
            let target = *lw
                .labels
                .get(l)
                .ok_or_else(|| lw.err(format!("unknown label {l}")))?;
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: None,
            }
        }
        "redux" => {
            // redux.sync.{add,min,max}.{u32,s32} / .{and,or,xor}.b32 /
            // .{min,max}[.abs][.NaN].f32 d, a, membermask (full warp only): the SASS
            // REDUX / CREDUX forms ptxas emits for them (mslk decode: redux.sync.max.NaN.f32 is
            // CREDUX.MAX.F32.NAN), with every member lane receiving the result.
            // ASSUME(L-ROSETTA): fixtures/e_ptx_redux = GB300.
            if parse_int(op_n(2)?) != Some(u32::MAX as u64) && op_n(2)? != "-1" {
                bail!(lw.err("redux.sync with a partial membermask not modeled"));
            }
            let op = match mods {
                ["sync", "add", "u32" | "s32"] => ReduxOp::Sum,
                ["sync", "and", "b32"] => ReduxOp::And,
                ["sync", "or", "b32"] => ReduxOp::Or,
                ["sync", "xor", "b32"] => ReduxOp::Xor,
                ["sync", "min", "u32"] => ReduxOp::MinU,
                ["sync", "max", "u32"] => ReduxOp::MaxU,
                ["sync", "min", "s32"] => ReduxOp::MinS,
                ["sync", "max", "s32"] => ReduxOp::MaxS,
                ["sync", m @ ("min" | "max"), rest @ .., "f32"]
                    if matches!(rest, [] | ["abs"] | ["NaN"] | ["abs", "NaN"]) =>
                {
                    let (abs, nan) = (rest.contains(&"abs"), rest.contains(&"NaN"));
                    if *m == "min" {
                        ReduxOp::MinF { abs, nan }
                    } else {
                        ReduxOp::MaxF { abs, nan }
                    }
                }
                _ => bail!(lw.err("redux.sync form not modeled")),
            };
            Op::Redux {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                op,
            }
        }
        // `match.any.sync.b32` / `.b64 d, a, membermask` (SASS MATCH.ANY[.U64]): full warp only,
        // as vote (probes/match_any).
        "match" if matches!(mods, ["any", "sync", "b32" | "b64"]) => {
            if parse_int(op_n(2)?) != Some(u32::MAX as u64) && parse_int(op_n(2)?) != Some(u64::MAX)
            {
                bail!(lw.err("match.any.sync with a partial membermask not modeled"));
            }
            Op::MatchAny {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                wide: has("b64"),
            }
        }
        "vote" => {
            // vote.sync.{all,any,uni}.pred d, a, membermask / vote.sync.ballot.b32 d, a,
            // membermask (full warp only; fixtures/l1_coreader's warp-level poll, DeepEP
            // barrier_block's `VOTE.ALL`, CuTeDSL's `vote.sync.any.pred`). any = !all(!a).
            if !matches!(
                mods,
                ["sync", "all" | "any" | "uni", "pred"] | ["sync", "ballot", "b32"]
            ) {
                bail!(
                    lw.err(
                        "only vote.sync.{all,any,uni}.pred and vote.sync.ballot.b32 are modeled"
                    )
                );
            }
            if parse_int(op_n(2)?) != Some(u32::MAX as u64) && parse_int(op_n(2)?) != Some(u64::MAX)
            {
                bail!(lw.err("vote.sync with a partial membermask not modeled"));
            }
            let a = lw.pred_src(op_n(1)?)?;
            let not_a = Pred {
                reg: a.reg,
                neg: !a.neg,
            };
            if has("ballot") {
                Op::VoteAny {
                    d: lw.dst(op_n(0)?)?,
                    a,
                }
            } else {
                let d = lw.pred_dst(op_n(0)?)?;
                let pd = Pred {
                    reg: Some(d),
                    neg: false,
                };
                if has("all") || has("uni") {
                    Op::VoteAll {
                        d,
                        a,
                        uni: has("uni"),
                    }
                } else if has("any") {
                    extra.push(Op::Plop3 {
                        d,
                        a: pd,
                        b: Pred::TRUE,
                        c: Pred::TRUE,
                        lut: 0x0f,
                        d2: None,
                    });
                    Op::VoteAll {
                        d,
                        a: not_a,
                        uni: false,
                    }
                } else {
                    bail!(lw.err("vote.sync mode not modeled"));
                }
            }
        }
        "elect" => {
            // elect.sync d|p, membermask. Only the full-warp mask is modeled (the leader is
            // chosen among the active threads executing it).
            if op_n(1)? != "0xffffffff" && op_n(1)? != "-1" {
                bail!(lw.err("elect.sync with a partial membermask not modeled"));
            }
            let (d, p) = op_n(0)?
                .split_once('|')
                .ok_or_else(|| lw.err("elect.sync wants d|p"))?;
            let lane = if d.trim() == "_" {
                None
            } else {
                Some(lw.dst(d.trim())?)
            };
            Op::Elect {
                d: lw.pred_dst(p.trim())?,
                lane,
            }
        }
        "ret" | "exit" => Op::Exit,
        // `trap` aborts the grid (as SASS BPT.TRAP; `__assertfail` lowers to a call PTX
        // keeps symbolic, so only explicit traps are seen here).
        "trap" => Op::Trap { why: String::new() },
        // A timing hint only: it cannot change which schedules are possible.
        // ASSUME(E-NANOSLEEP): see ASSUMPTIONS.md
        "nanosleep" => Op::Nop,
        // setmaxnreg.{inc,dec}.sync.aligned.u32 N (ptxas: USETMAXREG): only the register budget
        // changes. N must be an immediate in 24..=256, a multiple of 8. ASSUME(E-SETMAXNREG)
        "setmaxnreg" => {
            if !matches!(mods, ["inc" | "dec", "sync", "aligned", "u32"]) {
                bail!(lw.err(format!("setmaxnreg form .{} not modeled", mods.join("."))));
            }
            match parse_int(op_n(0)?) {
                Some(n) if (24..=256).contains(&n) && n % 8 == 0 && ops.len() == 1 => Op::Nop,
                _ => bail!(lw.err("setmaxnreg wants one immediate in 24..=256, a multiple of 8")),
            }
        }
        "cvta" => {
            let to = has("to");
            let space = mods
                .iter()
                .find_map(|m| space_of(m))
                .ok_or_else(|| lw.err("cvta without a space"))?;
            Op::Cvta {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                space,
                to_generic: !to,
            }
        }
        "cvt" => {
            let n = mods.len();
            let (to, from) = (mods[n - 2], mods[n - 1]);
            let int = |t: &str| t.starts_with('u') || t.starts_with('s') || t.starts_with('b');
            if to == "f32" && from == "bf16" {
                Op::CvtF32Bf16 {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                }
            } else if to == "f32" && from == "f16" {
                // cvt.f32.f16 (SASS HADD2.F32 d, -RZ, a.H0_H0): exact.
                Op::CvtF32F16 {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    hi: false,
                }
            } else if to == "f16" && from == "f32" && has("rn") {
                // cvt.rn.f16.f32 (SASS F2FP.F16.F32.PACK_AB Rd, RZ, Ra).
                Op::F2fpF16 {
                    d: lw.dst(op_n(0)?)?,
                    a: Val::Imm(0),
                    b: lw.val(op_n(1)?)?,
                }
            } else if mods == ["rn", "bf16x2", "f32"] || mods == ["rn", "f16x2", "f32"] {
                // cvt.rn.{bf16x2,f16x2}.f32 d, a, b: d = {a (high half), b}, each rounded to
                // nearest even; ptxas emits F2FP.{BF16,F16}.F32.PACK_AB d, a, b
                // (fixtures/s_ptx_packed = GB300). .relu / .satfinite / other roundings are
                // the narrow-format path (cvt_lp) or refused. ASSUME(L-ROSETTA)
                let (d, a, b) = (lw.dst(op_n(0)?)?, lw.val(op_n(1)?)?, lw.val(op_n(2)?)?);
                if to == "bf16x2" {
                    Op::F2fpBf16 { d, a, b }
                } else {
                    Op::F2fpF16 { d, a, b }
                }
            } else if to == "bf16" && from == "f32" && has("rn") {
                // cvt.rn.bf16.f32 (SASS F2FP.BF16.F32.PACK_AB Rd, RZ, Ra).
                Op::F2fpBf16 {
                    d: lw.dst(op_n(0)?)?,
                    a: Val::Imm(0),
                    b: lw.val(op_n(1)?)?,
                }
            } else if to == "f32" && matches!(from, "s32" | "u32") {
                // cvt.{rn,rz,rm,rp}.f32.{s32,u32} (SASS I2FP.F32.S32 / I2F).
                let rnd = if has("rz") {
                    Rnd::Rz
                } else if has("rm") {
                    Rnd::Rm
                } else if has("rp") {
                    Rnd::Rp
                } else if has("rn") {
                    Rnd::Rn
                } else {
                    bail!(lw.err("cvt.f32 from an integer needs a rounding modifier"))
                };
                Op::I2F {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    signed: from == "s32",
                    rnd,
                    bits: 32,
                    hi: false,
                    f64: false,
                }
            } else if from == "f32"
                && matches!(to, "s32" | "u32" | "s64" | "u64")
                && (mods == ["rzi", to, from] || mods == ["rzi", "ftz", to, from])
            {
                // cvt.rzi[.ftz].{s,u}{32,64}.f32 (SASS F2I[.U32|.S64|.U64].TRUNC): as F2I
                // (a subnormal input truncates to 0 with or without .ftz).
                Op::F2I {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    signed: to.starts_with('s'),
                    bits: type_bits(to).unwrap(),
                    rnd: Rnd::Rz,
                    ftz: has("ftz"),
                    f64: false,
                }
            } else if from == "f64"
                && matches!(to, "s32" | "u32" | "s64" | "u64")
                && n == 3
                && matches!(mods[0], "rzi" | "rni" | "rmi" | "rpi")
            {
                // cvt.r{z,n,m,p}i.{s,u}{32,64}.f64 (SASS F2I[.S64|.U64|.U32].F64[.TRUNC|.FLOOR|
                // .CEIL]). ASSUME(N-F2I-F64)
                Op::F2I {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    signed: to.starts_with('s'),
                    bits: type_bits(to).unwrap(),
                    rnd: match mods[0] {
                        "rzi" => Rnd::Rz,
                        "rni" => Rnd::Rn,
                        "rmi" => Rnd::Rm,
                        _ => Rnd::Rp,
                    },
                    ftz: false,
                    f64: true,
                }
            } else if let Some(op) = cvt_lp(lw, mods, &ops)? {
                op
            } else if int(to) && int(from) && n == 2 {
                Op::CvtInt {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    from_bits: type_bits(from).unwrap(),
                    to_bits: type_bits(to).unwrap(),
                    signed: from.starts_with('s'),
                }
            } else {
                bail!(lw.err("this cvt is not modeled yet"))
            }
        }
        "tcgen05" => {
            let sub = mods.first().copied().unwrap_or("");
            // ASSUME(E-PAIR): cta_group::2 alloc / dealloc / relinquish act on each CTA's own
            // TMEM (both CTAs of the pair issue them, the same allocation in each).
            let pair = has("cta_group::2");
            let cg1 = || -> Result<()> { Ok(()) };
            match sub {
                "alloc" => {
                    cg1()?;
                    Op::TmemAlloc {
                        dst: lw.addr(op_n(0)?)?,
                        ncols: lw.val(op_n(1)?)?,
                    }
                }
                "relinquish_alloc_permit" => {
                    cg1()?;
                    Op::TmemRelinquish
                }
                "dealloc" => {
                    cg1()?;
                    Op::TmemDealloc {
                        taddr: lw.val(op_n(0)?)?,
                        ncols: lw.val(op_n(1)?)?,
                    }
                }
                "mma" => {
                    cg1()?;
                    let vec = |x: &str| mods.contains(&x);
                    let kind = if has("kind::f16") {
                        MmaKind::F16
                    } else if has("kind::f8f6f4") {
                        MmaKind::F8f6f4
                    } else if has("kind::mxf8f6f4")
                        && !vec("scale_vec::2X")
                        && !vec("scale_vec::4X")
                    {
                        MmaKind::Mxf8f6f4
                    } else if has("kind::mxf4") && !vec("scale_vec::1X") && !vec("scale_vec::4X") {
                        MmaKind::Mxf4
                    } else if has("kind::mxf4nvf4") && !vec("scale_vec::1X") {
                        MmaKind::Mxf4nvf4 {
                            x4: vec("scale_vec::4X") || vec("block16"),
                        }
                    } else {
                        bail!(lw.err(
                            "tcgen05.mma kind not modeled (f16, f8f6f4, mxf8f6f4, mxf4, mxf4nvf4)"
                        ))
                    };
                    let scaled = !matches!(kind, MmaKind::F16 | MmaKind::F8f6f4);
                    if scaled != has("block_scale") || has("ws") || has("sp") {
                        bail!(lw.err("tcgen05.mma variant not modeled"));
                    }
                    // Operands: [d], a, bdesc, idesc, ([scale_a], [scale_b],) (mask,) p.
                    let base = if scaled { 6 } else { 4 };
                    if !(ops.len() == base + 1 || ops.len() == base + 2) {
                        bail!(lw.err("tcgen05.mma: unexpected operand count"));
                    }
                    let scale = if scaled {
                        let t = |i: usize| -> Result<Val> {
                            let a = lw.addr(op_n(i)?)?;
                            if a.off != 0 {
                                bail!(lw.err("tcgen05.mma scale address offset not modeled"));
                            }
                            a.base
                                .ok_or_else(|| lw.err("tmem address must be a register"))
                        };
                        Some((t(4)?, t(5)?))
                    } else {
                        None
                    };
                    let disable: Vec<Val> = if ops.len() == base + 2 {
                        op_n(base)?
                            .trim()
                            .trim_matches(|c| c == '{' || c == '}')
                            .split(',')
                            .map(|x| lw.val(x.trim()))
                            .collect::<Result<_>>()?
                    } else {
                        vec![]
                    };
                    let da = lw.addr(op_n(0)?)?;
                    if da.off != 0 {
                        bail!(lw.err("tcgen05.mma with a TMEM address offset not modeled"));
                    }
                    let d = da
                        .base
                        .ok_or_else(|| lw.err("tmem address must be a register"))?;
                    // `[a]`: A in TMEM.
                    let a_tmem = op_n(1)?.trim().starts_with('[');
                    let adesc = if a_tmem {
                        let aa = lw.addr(op_n(1)?)?;
                        if aa.off != 0 {
                            bail!(lw.err("tcgen05.mma with an A TMEM offset not modeled"));
                        }
                        aa.base
                            .ok_or_else(|| lw.err("tmem address must be a register"))?
                    } else {
                        lw.val(op_n(1)?)?
                    };
                    Op::Mma {
                        d,
                        adesc,
                        bdesc: lw.val(op_n(2)?)?,
                        idesc: lw.val(op_n(3)?)?,
                        enable_d: lw.pred_src(op_n(ops.len() - 1)?)?,
                        pair,
                        disable,
                        a_tmem,
                        kind,
                        scale,
                    }
                }
                "cp" => {
                    // ASSUME(S-TCCP): see ASSUMPTIONS.md
                    if !has("32x128b")
                        || !has("warpx4")
                        || mods.iter().any(|m| m.starts_with("b8x16"))
                    {
                        bail!(lw.err(
                            "tcgen05.cp variant not modeled (32x128b.warpx4, no decompression)"
                        ));
                    }
                    let ta = lw.addr(op_n(0)?)?;
                    // `[taddr+off]`: the offset adds to the TMEM address word, as SASS
                    // `tmem[URa+off]`.
                    let off = u32::try_from(ta.off)
                        .map_err(|_| lw.err("tcgen05.cp: negative TMEM address offset"))?;
                    Op::TcCp {
                        taddr: ta
                            .base
                            .ok_or_else(|| lw.err("tmem address must be a register"))?,
                        off,
                        sdesc: lw.val(op_n(1)?)?,
                        pair,
                    }
                }
                "commit" => {
                    cg1()?;
                    let ma = lw.addr(op_n(0)?)?;
                    let a = ma
                        .base
                        .ok_or_else(|| lw.err("mbarrier address must be a register"))?;
                    let mask = if has("multicast::cluster") {
                        Some(lw.val(op_n(1)?)?)
                    } else {
                        None
                    };
                    Op::TcCommit {
                        mbar: a,
                        off: ma.off as u32,
                        mask,
                    }
                }
                "ld" => {
                    let x2 = has("16x32bx2");
                    if !has("32x32b") && !has("16x64b") && !has("16x128b") && !has("16x256b") && !x2
                    {
                        bail!(lw.err(
                            "only the 32x32b, 16x64b, 16x128b, 16x256b and 16x32bx2 tcgen05.ld shapes are modeled"
                        ));
                    }
                    if x2 != (ops.len() == 3) || ops.len() > 3 {
                        bail!(lw.err("tcgen05.ld: unexpected operands"));
                    }
                    let (d, n) = lw.reg_vec(op_n(0)?)?;
                    let num: u8 = mods
                        .iter()
                        .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse().ok()))
                        .ok_or_else(|| lw.err("missing .xN"))?;
                    let per = if has("16x256b") {
                        4
                    } else if has("16x128b") {
                        2
                    } else {
                        1
                    };
                    if num as u16 * per != n as u16 {
                        bail!(lw.err("register count does not match .xN"));
                    }
                    let a = lw.addr(op_n(1)?)?;
                    let taddr = a
                        .base
                        .ok_or_else(|| lw.err("tmem address must be a register"))?;
                    let ld = |half: Option<u8>, off: u32| Op::TmemLd {
                        d,
                        n,
                        taddr: taddr.clone(),
                        off,
                        tracked: true,
                        s16x64b: has("16x64b"),
                        half,
                        s16x256b: has("16x256b"),
                        s16x128b: has("16x128b"),
                        red: None,
                    };
                    if x2 {
                        // `.16x32bx2 d, [a], split`: lanes 0..15 at a, lanes 16..31 at
                        // a + split columns (SASS LDTM.16dp32bit_t0_t15 + _t16_t31;
                        // probes/tmem_x2). ASSUME(S-TMEM)
                        let split = parse_int(op_n(2)?.trim())
                            .ok_or_else(|| lw.err("bad split offset"))?
                            as u32;
                        extra.push(ld(Some(1), (a.off as u32).wrapping_add(split)));
                        ld(Some(0), a.off as u32)
                    } else {
                        ld(None, a.off as u32)
                    }
                }
                "wait::ld" => Op::TmemWaitLd,
                "wait::st" => Op::TmemWaitSt,
                "st" => {
                    let x2 = has("16x32bx2");
                    if !has("32x32b") && !has("16x256b") && !has("16x128b") && !x2 {
                        bail!(lw.err(
                            "only the 32x32b / 16x256b / 16x128b / 16x32bx2 tcgen05.st shapes are modeled so far"
                        ));
                    }
                    if ops.len() != 2 + x2 as usize {
                        bail!(lw.err("tcgen05.st: unexpected operands"));
                    }
                    let num: usize = mods
                        .iter()
                        .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse::<usize>().ok()))
                        .ok_or_else(|| lw.err("missing .xN"))?
                        * if has("16x256b") {
                            4
                        } else if has("16x128b") {
                            2
                        } else {
                            1
                        };
                    let v: Vec<Val> = op_n(1 + x2 as usize)?
                        .trim()
                        .trim_start_matches('{')
                        .trim_end_matches('}')
                        .split(',')
                        .map(|r| lw.val(r))
                        .collect::<Result<_>>()?;
                    if v.len() != num {
                        bail!(lw.err("register count does not match .xN"));
                    }
                    let a = lw.addr(op_n(0)?)?;
                    let taddr = a
                        .base
                        .ok_or_else(|| lw.err("tmem address must be a register"))?;
                    let st = |half: Option<u8>, off: u32| Op::TmemSt {
                        taddr: taddr.clone(),
                        off,
                        v: v.clone(),
                        tracked: true,
                        half,
                        s16x256b: has("16x256b"),
                        s16x128b: has("16x128b"),
                    };
                    if x2 {
                        // `.16x32bx2 [a], split, {v}`: as tcgen05.ld.16x32bx2.
                        // ASSUME(S-TMEM)
                        let split = parse_int(op_n(1)?.trim())
                            .ok_or_else(|| lw.err("bad split offset"))?
                            as u32;
                        extra.push(st(Some(1), (a.off as u32).wrapping_add(split)));
                        st(Some(0), a.off as u32)
                    } else {
                        st(None, a.off as u32)
                    }
                }
                "fence::before_thread_sync" => Op::Fence {
                    kind: FenceKind::TcBefore,
                },
                "fence::after_thread_sync" => Op::Fence {
                    kind: FenceKind::TcAfter,
                },
                _ => bail!(lw.err(format!("tcgen05.{sub} is not modeled yet"))),
            }
        }
        // st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [addr], v, [mbar]: a store
        // into (another CTA's) shared memory that completes 4 transaction bytes on the
        // mbarrier there (SASS STAS). Lowering it as a plain st dropped the complete_tx.
        "st" if has("async") => {
            let vn = mods
                .iter()
                .find_map(|m| m.strip_prefix('v').and_then(|x| x.parse::<u32>().ok()))
                .unwrap_or(1);
            let bytes = bits as u32 / 8 * vn;
            if !(has("mbarrier::complete_tx::bytes") && has("shared::cluster"))
                || !(bits == 32 || bits == 64)
                || !matches!(bytes, 4 | 8 | 16)
                || has("release")
                || has("mmio")
            {
                bail!(lw.err(
                    "only st.async.shared::cluster.mbarrier::complete_tx::bytes of 4, 8 or 16 bytes is modeled"
                ));
            }
            let reg = |s: &str| -> Result<Val> {
                let a = lw.addr(s)?;
                if a.off != 0 || a.base2.is_some() {
                    bail!(lw.err("st.async wants register addresses"));
                }
                a.base
                    .ok_or_else(|| lw.err("st.async wants register addresses"))
            };
            let v = if vn > 1 {
                let (r, k) = lw.reg_vec(op_n(1)?)?;
                if k as u32 != vn {
                    bail!(lw.err("vector length does not match .vN"));
                }
                Val::Reg(r, k)
            } else {
                lw.val(op_n(1)?)?
            };
            Op::StAsync {
                addr: reg(op_n(0)?)?,
                v,
                mbar: reg(op_n(2)?)?,
                bytes: bytes as u8,
            }
        }
        // `ld` / `st` `.local` (scalar, up to 32 bits) through a `.local` array's address: the
        // thread's local memory, as SASS LDL / STL. ASSUME(L-ROSETTA)
        "ld" | "st" if has("local") => {
            if mods.iter().any(|m| m.starts_with('v')) || bits > 32 {
                bail!(lw.err("only scalar ld / st .local of up to 32 bits are modeled"));
            }
            if base == "ld" && tname.starts_with('s') && bits < 32 {
                bail!(lw.err("sign-extending ld.local not modeled"));
            }
            if let Some(m) = mods
                .iter()
                .find(|m| type_bits(m).is_none() && **m != "local")
            {
                bail!(lw.err(format!("{base}.local: modifier .{m} not modeled")));
            }
            let bytes = (bits / 8).max(1) as u32;
            if base == "ld" {
                Op::LdLocal {
                    d: lw.dst(op_n(0)?)?,
                    bytes,
                    addr: lw.addr(op_n(1)?)?,
                }
            } else {
                Op::StLocal {
                    bytes,
                    addr: lw.addr(op_n(0)?)?,
                    v: lw.val(op_n(1)?)?,
                }
            }
        }
        // ASSUME(S-MULTIMEM): multimem.st / .red lower to a global St / Atom (as ptxas lowers
        // them to STG / REDG); the simulator sends them to every bound location when the
        // address is a multicast one. multimem.ld_reduce is its own op.
        "multimem" => {
            let kind = mods.first().copied().unwrap_or("");
            if !matches!(kind, "ld_reduce" | "st" | "red") {
                bail!(lw.err(format!("multimem.{kind} not modeled")));
            }
            let rest = &mods[1..];
            if let Some(m) = rest.iter().find(|m| {
                !matches!(
                    **m,
                    "weak"
                        | "relaxed"
                        | "acquire"
                        | "release"
                        | "cta"
                        | "cluster"
                        | "gpu"
                        | "sys"
                        | "global"
                        | "add"
                        | "min"
                        | "max"
                        | "and"
                        | "or"
                        | "xor"
                        | "acc::f32"
                        | "acc::f16"
                        | "v2"
                        | "v4"
                        | "v8"
                        | "b32"
                        | "b64"
                        | "u32"
                        | "u64"
                        | "s32"
                        | "s64"
                        | "f32"
                        | "f64"
                        | "f16"
                        | "f16x2"
                        | "bf16"
                        | "bf16x2"
                        | "e4m3"
                        | "e4m3x2"
                        | "e4m3x4"
                        | "e5m2"
                        | "e5m2x2"
                        | "e5m2x4"
                )
            }) {
                bail!(lw.err(format!("multimem.{kind}: modifier .{m} not modeled")));
            }
            let tyname = *rest.last().unwrap_or(&"");
            let (ety, ebytes): (McTy, u8) = match tyname {
                "u32" | "b32" => (McTy::U32, 4),
                "s32" => (McTy::S32, 4),
                "u64" | "b64" => (McTy::U64, 8),
                "s64" => (McTy::S64, 8),
                "f32" => (McTy::F32, 4),
                "f64" => (McTy::F64, 8),
                "f16" => (McTy::F16, 2),
                "f16x2" => (McTy::F16, 4),
                "bf16" => (McTy::Bf16, 2),
                "bf16x2" => (McTy::Bf16, 4),
                "e4m3" => (McTy::E4m3, 1),
                "e4m3x2" => (McTy::E4m3, 2),
                "e4m3x4" => (McTy::E4m3, 4),
                "e5m2" => (McTy::E5m2, 1),
                "e5m2x2" => (McTy::E5m2, 2),
                "e5m2x4" => (McTy::E5m2, 4),
                _ => bail!(lw.err("multimem: missing type")),
            };
            let vec_n = rest
                .iter()
                .find_map(|m| m.strip_prefix('v').and_then(|x| x.parse::<u8>().ok()))
                .unwrap_or(1);
            let bytes = ebytes * vec_n;
            if !matches!(bytes, 4 | 8 | 16) {
                bail!(lw.err("multimem: only 32 / 64 / 128-bit accesses are modeled"));
            }
            let has_r = |m: &str| rest.contains(&m);
            let sem = if has_r("acquire") {
                Some(Sem::Acquire)
            } else if has_r("release") {
                Some(Sem::Release)
            } else if has_r("relaxed") {
                Some(Sem::Relaxed)
            } else {
                None
            };
            let scope = scope_of(rest);
            // Defaults (PTX ISA): ld_reduce / st weak, red relaxed at .sys.
            let sync = match (kind, sem) {
                ("red", s) => Some(MemSync {
                    sem: s.unwrap_or(Sem::Relaxed),
                    scope: scope.unwrap_or(Scope::Sys),
                }),
                (_, None) => None,
                (_, Some(s)) => Some(MemSync {
                    sem: s,
                    scope: scope.unwrap_or(Scope::Gpu),
                }),
            };
            if let Some(m) = sync {
                if (kind == "ld_reduce" && m.releases()) || (kind != "ld_reduce" && m.acquires()) {
                    bail!(lw.err("invalid memory semantics for this multimem operation"));
                }
            }
            let op = if has_r("add") {
                McOp::Add
            } else if has_r("min") {
                McOp::Min
            } else if has_r("max") {
                McOp::Max
            } else if has_r("and") {
                McOp::And
            } else if has_r("or") {
                McOp::Or
            } else if has_r("xor") {
                McOp::Xor
            } else {
                McOp::Add
            };
            let regs = (bytes / 4).max(1);
            match kind {
                "ld_reduce" => {
                    let wide = ety.lane_bytes() == 8;
                    let (d, k, moves) = if wide {
                        (lw.dst(op_n(0)?)?, 1, vec![])
                    } else if regs > 1 {
                        lw.reg_vec_mem(op_n(0)?, true)?
                    } else {
                        (
                            lw.dst(op_n(0)?.trim_matches(|c| c == '{' || c == '}'))?,
                            1,
                            vec![],
                        )
                    };
                    if !wide && k != regs {
                        bail!(lw.err(
                            "multimem.ld_reduce: destination count does not match the access"
                        ));
                    }
                    let addr = lw.addr(op_n(1)?)?;
                    lw.extra = moves;
                    Op::LdReduce {
                        d,
                        n: k,
                        bytes,
                        addr,
                        op,
                        ty: ety,
                        acc: has_r("acc::f32") || has_r("acc::f16"),
                        sync,
                    }
                }
                "st" => {
                    let wide = ety.lane_bytes() == 8;
                    let addr = lw.addr(op_n(0)?)?;
                    let v = if !wide && regs > 1 {
                        let (r, k, moves) = lw.reg_vec_mem(op_n(1)?, false)?;
                        if k != regs {
                            bail!(lw.err("multimem.st: source count does not match the access"));
                        }
                        lw.pre = moves;
                        Val::Reg(r, k)
                    } else {
                        lw.val(op_n(1)?.trim_matches(|c| c == '{' || c == '}'))?
                    };
                    Op::St {
                        space: Space::Global,
                        bytes,
                        addr,
                        v,
                        sync,
                    }
                }
                _ => {
                    // multimem.red: the integer reductions and the scalar f32 add (as SASS
                    // REDG .F32.FTZ.RN) are modeled; vector / 16-bit float adds are not.
                    let aop = match (op, ety, vec_n) {
                        (McOp::Add, McTy::F32, 1) => AtomOp::FAddFtz,
                        (
                            _,
                            McTy::F32
                            | McTy::F64
                            | McTy::F16
                            | McTy::Bf16
                            | McTy::E4m3
                            | McTy::E5m2,
                            _,
                        ) => {
                            bail!(lw.err("multimem.red on this float type / vector not modeled"))
                        }
                        (McOp::Add, _, 1) => AtomOp::Add,
                        (McOp::And, _, 1) => AtomOp::And,
                        (McOp::Or, _, 1) => AtomOp::Or,
                        (McOp::Xor, _, 1) => AtomOp::Xor,
                        (McOp::Min, McTy::S32 | McTy::S64, 1) => AtomOp::MinS,
                        (McOp::Max, McTy::S32 | McTy::S64, 1) => AtomOp::MaxS,
                        (McOp::Min, _, 1) => AtomOp::MinU,
                        (McOp::Max, _, 1) => AtomOp::MaxU,
                        _ => bail!(lw.err("multimem.red with a vector not modeled")),
                    };
                    Op::Atom {
                        p: None,
                        op: aop,
                        space: Space::Global,
                        bytes,
                        addr: lw.addr(op_n(0)?)?,
                        v: lw.val(op_n(1)?)?,
                        d: None,
                        sync: sync.unwrap(),
                        new: None,
                    }
                }
            }
        }
        "ld" | "st" => {
            let space = mods
                .iter()
                .find_map(|m| space_of(m))
                .unwrap_or(Space::Generic);
            let bytes = (bits / 8).max(1);
            if bytes > 8 && base == "st" {
                bail!(lw.err("st.b128 not modeled yet"));
            }
            // Vectors: consecutive registers only (`{%r5, %r6, %r7, %r8}`), one element per
            // register.
            let vec_n = mods
                .iter()
                .find_map(|m| m.strip_prefix('v').and_then(|x| x.parse::<u8>().ok()));
            if let Some(n) = vec_n {
                // 64-bit elements: `.v2.b64` (one PTX register each).
                if bits > 64 || bits == 64 && n != 2 {
                    bail!(lw.err("only vector elements up to 32 bits, and .v2.b64, are modeled"));
                }
                let sync = mem_sync(mods, false).map_err(|e| lw.err(e.to_string()))?;
                let opnd = if base == "ld" { op_n(0)? } else { op_n(1)? };
                let (r, k, moves) = lw.reg_vec_mem(opnd, base == "ld")?;
                if k != n {
                    bail!(lw.err("vector length does not match .vN"));
                }
                let bytes = (bits / 8).max(1) * n;
                let inst = Inst {
                    guard,
                    op: if base == "ld" {
                        Op::Ld {
                            d: r,
                            n,
                            space,
                            bytes,
                            addr: lw.addr(op_n(1)?)?,
                            sync,
                            sext: tname.starts_with('s') && bits < 32,
                        }
                    } else {
                        Op::St {
                            space,
                            bytes,
                            addr: lw.addr(op_n(0)?)?,
                            v: Val::Reg(r, n),
                            sync,
                        }
                    },
                    uniform: false,
                    loc: lw.line,
                    text: text.to_string(),
                };
                if base == "ld" {
                    lw.extra = moves;
                } else {
                    lw.pre = moves;
                }
                return Ok(inst);
            }
            let sync = mem_sync(mods, false).map_err(|e| lw.err(e.to_string()))?;
            if let Some(m) = sync {
                if (base == "ld" && m.releases()) || (base == "st" && m.acquires()) {
                    bail!(lw.err("invalid memory semantics for this access"));
                }
            }
            if base == "ld" {
                Op::Ld {
                    d: lw.dst(op_n(0)?)?,
                    n: 1,
                    space,
                    bytes,
                    addr: lw.addr(op_n(1)?)?,
                    sync,
                    sext: tname.starts_with('s') && bits < 32,
                }
            } else {
                Op::St {
                    space,
                    bytes,
                    addr: lw.addr(op_n(0)?)?,
                    v: lw.val(op_n(1)?)?,
                    sync,
                }
            }
        }
        // `atom` / `red` `.add.noftz` `.bf16x2` / `.v2.bf16` / `.bf16` on global memory: per-lane
        // bf16 adds (SASS `ATOMG` / `REDG .E.ADD.BF16x2.RN`; the scalar form is a CAS loop with
        // `HADD2.BF16_V2`). `.v2.bf16` packs its two halves through the scratch register.
        // probes/bf16x2, both front-ends. ASSUME(N-BF16X2)
        "atom" | "red" if has("noftz") => {
            let bf = match ht {
                "bf16x2" => true,
                "bf16" => false,
                _ => bail!(lw.err(format!("{base}.noftz.{ht} not modeled"))),
            };
            if let Some(m) = mods.iter().find(|m| {
                !matches!(
                    **m,
                    "add"
                        | "noftz"
                        | "bf16"
                        | "bf16x2"
                        | "v2"
                        | "relaxed"
                        | "acquire"
                        | "release"
                        | "acq_rel"
                        | "cta"
                        | "cluster"
                        | "gpu"
                        | "sys"
                        | "global"
                )
            }) {
                bail!(lw.err(format!("{base}: modifier .{m} not modeled")));
            }
            if !has("add") || !has("global") || (has("v2") && bf) {
                bail!(lw.err(format!("{base}.noftz form not modeled")));
            }
            let pair = bf || has("v2");
            let sync = mem_sync(mods, true)
                .map_err(|e| lw.err(e.to_string()))?
                .unwrap();
            let (d, a, v) = if base == "atom" {
                (Some(op_n(0)?), op_n(1)?, op_n(2)?)
            } else {
                (None, op_n(0)?, op_n(1)?)
            };
            let halves = |x: &str| -> Option<Vec<String>> {
                let inner = x.trim().strip_prefix('{')?.strip_suffix('}')?;
                Some(inner.split(',').map(|h| h.trim().to_string()).collect())
            };
            let (v, d) = if has("v2") {
                let tmp = lw.scratch_v()?;
                let (Some(vh), dh) = (halves(v), d.map(halves)) else {
                    bail!(lw.err("v2.bf16 operands must be {a, b} vectors"));
                };
                if vh.len() != 2
                    || dh
                        .as_ref()
                        .is_some_and(|x| x.as_ref().is_none_or(|x| x.len() != 2))
                {
                    bail!(lw.err("v2.bf16 operands must be {a, b} vectors"));
                }
                lw.pre.push(Op::Prmt {
                    d: tmp,
                    a: lw.val(&vh[0])?,
                    sel: Val::Imm(0x5410),
                    b: lw.val(&vh[1])?,
                });
                if let Some(Some(dh)) = dh {
                    for (h, sel) in [(&dh[0], 0x4410), (&dh[1], 0x4432)] {
                        if h != "_" {
                            extra.push(Op::Prmt {
                                d: lw.dst(h)?,
                                a: Val::Reg(tmp, 1),
                                sel: Val::Imm(sel),
                                b: Val::Imm(0),
                            });
                        }
                    }
                    (Val::Reg(tmp, 1), Some(tmp))
                } else {
                    (Val::Reg(tmp, 1), None)
                }
            } else {
                (lw.val(v)?, d.map(|x| lw.dst(x)).transpose()?)
            };
            Op::Atom {
                p: None,
                op: if pair {
                    AtomOp::AddBf16x2
                } else {
                    AtomOp::AddBf16
                },
                space: Space::Global,
                bytes: if pair { 4 } else { 2 },
                addr: lw.addr(a)?,
                v,
                d,
                sync,
                new: None,
            }
        }
        "atom" | "red" => {
            // Fail closed: every modifier must be one modeled here (no `.noftz`, cache hints,
            // vectors or 16-bit types).
            if let Some(m) = mods.iter().find(|m| {
                !matches!(
                    **m,
                    "add"
                        | "or"
                        | "and"
                        | "xor"
                        | "exch"
                        | "cas"
                        | "min"
                        | "max"
                        | "inc"
                        | "b32"
                        | "b64"
                        | "u32"
                        | "u64"
                        | "s32"
                        | "s64"
                        | "f32"
                        | "relaxed"
                        | "acquire"
                        | "release"
                        | "acq_rel"
                        | "cta"
                        | "cluster"
                        | "gpu"
                        | "sys"
                        | "global"
                        | "shared"
                        | "shared::cta"
                        | "shared::cluster"
                )
            }) {
                bail!(lw.err(format!("{base}: modifier .{m} not modeled")));
            }
            let space = mods
                .iter()
                .find_map(|m| space_of(m))
                .unwrap_or(Space::Generic);
            let op = if has("add") && tname == "f32" {
                AtomOp::FAddFtz
            } else if has("add") {
                AtomOp::Add
            } else if has("or") {
                AtomOp::Or
            } else if has("and") {
                AtomOp::And
            } else if has("xor") {
                AtomOp::Xor
            } else if has("exch") && base == "atom" {
                AtomOp::Exch
            } else if has("cas") && base == "atom" {
                AtomOp::Cas
            } else if (has("inc") || has("dec")) && tname == "u32" {
                if has("inc") { AtomOp::Inc } else { AtomOp::Dec }
            } else if (has("min") || has("max")) && matches!(tname, "u32" | "s32" | "u64" | "s64") {
                // Integer min / max (CUTLASS / CuTeDSL `smem_fmax`: a float max through
                // `red.max.s32` for non-negative and `red.min.u32` for negative values).
                match (has("min"), tname.starts_with('s')) {
                    (true, false) => AtomOp::MinU,
                    (true, true) => AtomOp::MinS,
                    (false, false) => AtomOp::MaxU,
                    (false, true) => AtomOp::MaxS,
                }
            } else {
                bail!(lw.err("this atom/red operation is not modeled yet"));
            };
            if (tname.starts_with('f') && op != AtomOp::FAddFtz) || (bits != 32 && bits != 64) {
                bail!(lw.err("only 32/64-bit integer atom/red are modeled"));
            }
            let sync = mem_sync(mods, true)
                .map_err(|e| lw.err(e.to_string()))?
                .unwrap();
            let (d, a, v) = if base == "atom" {
                (Some(lw.dst(op_n(0)?)?), op_n(1)?, op_n(2)?)
            } else {
                (None, op_n(0)?, op_n(1)?)
            };
            // `atom.cas d, [a], b, c`: compare b, new c.
            let new = if op == AtomOp::Cas {
                Some(lw.val(op_n(3)?)?)
            } else {
                None
            };
            Op::Atom {
                p: None,
                op,
                space,
                bytes: (bits / 8),
                addr: lw.addr(a)?,
                v: lw.val(v)?,
                d,
                sync,
                new,
            }
        }
        "barrier" if has("cluster") => {
            if has("arrive") {
                if has("relaxed") && has("release") {
                    bail!(lw.err("barrier.cluster.arrive cannot be both relaxed and release"));
                }
                Op::ClusterArrive {
                    release: !has("relaxed"),
                }
            } else if has("wait") {
                if has("relaxed") {
                    bail!(lw.err("barrier.cluster.wait.relaxed not modeled"));
                }
                Op::ClusterWait
            } else {
                bail!(lw.err("only barrier.cluster.arrive / wait are modeled"));
            }
        }
        "clusterlaunchcontrol" if has("try_cancel") => {
            // `.multicast::cluster::all`: every CTA of the cluster gets the response and the
            // completion at the same offsets (SASS UGETNEXTWORKID.BROADCAST, probes/clc_mcast).
            let multicast = mods.contains(&"multicast::cluster::all");
            if !(has("async") && has("shared::cta") && has("mbarrier::complete_tx::bytes"))
                || mods
                    .iter()
                    .any(|m| m.starts_with("multicast") && *m != "multicast::cluster::all")
            {
                bail!(
                    lw.err("only try_cancel to shared::cta [.multicast::cluster::all] is modeled")
                );
            }
            Op::ClcTryCancel {
                resp: lw
                    .addr(op_n(0)?)?
                    .base
                    .ok_or_else(|| lw.err("register address"))?,
                mbar: lw
                    .addr(op_n(1)?)?
                    .base
                    .ok_or_else(|| lw.err("register address"))?,
                multicast,
            }
        }
        "clusterlaunchcontrol" if has("query_cancel") => {
            let ctaid = if has("is_canceled") {
                None
            } else if has("get_first_ctaid::x") {
                Some(0)
            } else if has("get_first_ctaid::y") {
                Some(1)
            } else {
                bail!(lw.err("this query_cancel form is not modeled"));
            };
            Op::ClcQuery {
                d: if ctaid.is_none() {
                    lw.pred_dst(op_n(0)?)?
                } else {
                    lw.dst(op_n(0)?)?
                },
                q: lw.reg(op_n(1)?.trim())?,
                ctaid,
            }
        }
        "mapa" => {
            // mapa.u64 without a space: a generic address in the shared window (S-SWIN).
            if !has("shared::cluster") && !(mods.len() == 1 && has("u64")) {
                bail!(lw.err("only mapa.shared::cluster / generic mapa.u64 are modeled"));
            }
            Op::Mapa {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                rank: lw.val(op_n(2)?)?,
            }
        }
        // bar{.cta}.sync / barrier{.cta}.sync{.aligned} (also bare `bar` / `barrier.cta`), and
        // bar{.cta}.arrive / barrier{.cta}.arrive{.aligned}, which always takes a count.
        // `bar.warp.sync membermask` (WARPSYNC).
        "bar" if mods == ["warp", "sync"] => Op::WarpSync {
            mask: lw.val(op_n(0)?)?,
            collective: false,
            all: false,
        },
        "bar" | "barrier" => {
            let arrive = has("arrive");
            let known = |m: &&str| matches!(*m, "sync" | "arrive" | "cta" | "aligned");
            if !mods.iter().all(known) || (arrive && has("sync")) {
                bail!(lw.err("only bar.sync / bar.arrive are modeled so far"));
            }
            let id = lw.val(op_n(0)?)?;
            let count = match ops.get(1) {
                Some(c) => Some(lw.val(c)?),
                None if arrive => bail!(lw.err("bar.arrive requires a thread count")),
                None => None,
            };
            // `bar.*` is `barrier.*.aligned`.
            let aligned = base == "bar" || has("aligned");
            Op::BarSync {
                id,
                count,
                arrive,
                packed: false,
                aligned,
            }
        }
        // ASSUME(E-PDL): see ASSUMPTIONS.md
        "griddepcontrol" => match mods.first().copied() {
            Some("wait") => Op::GridDepWait,
            Some("launch_dependents") => Op::GridDepLaunch,
            _ => bail!(lw.err("unknown griddepcontrol form")),
        },
        "fence" => {
            if has("mbarrier_init") {
                Op::Fence {
                    kind: FenceKind::MbarInit,
                }
            } else if has("acq_rel") || has("sc") {
                // ASSUME(S-SCOPE): fence.sc is treated as fence.acq_rel (no SC order is
                // modeled beyond what the interleaving gives).
                // The scope is required (PTX ISA `fence{.sem}.scope`): no default that could be
                // stronger than what the kernel asked for.
                Op::Fence {
                    kind: FenceKind::Membar(
                        scope_of(mods).ok_or_else(|| lw.err("fence without a scope"))?,
                    ),
                }
            } else if mods.first() == Some(&"proxy") && has("async") {
                // `fence.proxy.async` with no state space orders all of them (ptxas: MEMBAR.ALL.GPU
                // + FENCE.VIEW.ASYNC.S); `.global` orders global memory only, which no proxy
                // check models, so it never covers a shared-memory check. ASSUME(S-PROXY)
                match &mods[1..] {
                    ["async"] | ["async", "shared::cta" | "shared::cluster"] => Op::Fence {
                        kind: FenceKind::ProxyAsync,
                    },
                    ["async", "global"] => Op::Fence {
                        kind: FenceKind::ProxyAsyncGlobal,
                    },
                    _ => bail!(lw.err("this fence.proxy form is not modeled")),
                }
            } else if mods == ["proxy", "alias"] {
                // ASSUME(S-MC-ALIAS): a bidirectional proxy fence (no release / acquire of
                // its own).
                Op::Fence {
                    kind: FenceKind::ProxyAlias,
                }
            } else {
                bail!(lw.err("this fence is not modeled yet"))
            }
        }
        "mbarrier" => match mods.first().copied() {
            Some("inval") => Op::MbarInval {
                addr: lw.addr(op_n(0)?)?,
            },
            Some("init") => Op::MbarInit {
                addr: lw.addr(op_n(0)?)?,
                count: lw.val(op_n(1)?)?,
                encoded: false,
            },
            Some("arrive") => {
                // ASSUME(S-SCOPE): .release.cta / .release.cluster are not distinguished.
                let remote = has("shared::cluster");
                let tx = if has("expect_tx") {
                    Some(lw.val(op_n(2)?)?)
                } else {
                    None
                };
                let count = if !has("expect_tx") && ops.len() > 2 {
                    Some(lw.val(op_n(2)?)?)
                } else {
                    None
                };
                Op::MbarArrive {
                    addr: lw.addr(op_n(1)?)?,
                    tx,
                    remote,
                    count,
                    complete: false,
                    cluster: has("release") && has("cluster"),
                }
            }
            Some("try_wait") if has("parity") => Op::MbarTryWait {
                d: lw.pred_dst(op_n(0)?)?,
                addr: lw.addr(op_n(1)?)?,
                parity: lw.val(op_n(2)?)?,
                bit31: false,
            },
            _ => bail!(lw.err("this mbarrier op is not modeled yet")),
        },
        // Non-bulk cp.async (SASS LDGSTS / LDGDEPBAR / DEPBAR / ARRIVES.LDGSTSBAR):
        // `cp.async.{ca,cg}.shared{::cta}.global[.L2::...] [dst], [src], cp-size{, src-size |
        // ignore-src}{, cache-policy}`. src-size must be 0 (zero fill) or cp-size; ignore-src
        // is a predicate (true: zero fill).
        "cp" if mods.len() >= 4
            && mods[0] == "async"
            && matches!(mods[1], "ca" | "cg")
            && matches!(mods[2], "shared" | "shared::cta")
            && mods[3] == "global" =>
        {
            // ASSUME(L-PTX-HINT): `.L2::cache_hint` (with its cache-policy operand) and the
            // `.L2::{64,128,256}B` prefetch sizes are performance hints only.
            if let Some(m) = mods[4..]
                .iter()
                .find(|m| !matches!(**m, "L2::cache_hint" | "L2::64B" | "L2::128B" | "L2::256B"))
            {
                bail!(lw.err(format!("cp.async: modifier .{m} not modeled")));
            }
            let size =
                parse_int(op_n(2)?).ok_or_else(|| lw.err("cp.async size must be an immediate"))?;
            if !matches!(size, 4 | 8 | 16) || (mods[1] == "cg" && size != 16) {
                bail!(lw.err("cp.async size must be 4, 8 or 16 (16 for .cg)"));
            }
            let hint = has("L2::cache_hint");
            let n_extra = ops.len() - 3 - hint as usize;
            let src_ok = match n_extra {
                0 => Pred::TRUE,
                1 => {
                    let o = op_n(3)?.trim();
                    match parse_int(o) {
                        Some(0) => Pred {
                            reg: None,
                            neg: true,
                        },
                        Some(n) if n == size => Pred::TRUE,
                        Some(_) => bail!(lw.err("cp.async with a partial src-size not modeled")),
                        None => match lw.reg(o.trim_start_matches('!')) {
                            Ok(r) if r.file == File::P => {
                                let p = lw.pred_src(o)?;
                                Pred {
                                    reg: p.reg,
                                    neg: !p.neg,
                                }
                            }
                            _ => bail!(lw.err("cp.async with a register src-size not modeled")),
                        },
                    }
                }
                _ => bail!(lw.err("cp.async: unexpected operands")),
            };
            Op::CpAsync {
                dst: lw.addr(op_n(0)?)?,
                src: lw.addr(op_n(1)?)?,
                bytes: size as u32,
                src_ok,
            }
        }
        "cp" if mods == ["async", "commit_group"] => Op::CpAsyncCommit,
        "cp" if mods == ["async", "wait_group"] => {
            let n = parse_int(op_n(0)?)
                .ok_or_else(|| lw.err("wait_group count must be an immediate"))?;
            Op::CpAsyncWait { n: n as u32 }
        }
        // `cp.async.wait_all` = commit_group + wait_group 0.
        "cp" if mods == ["async", "wait_all"] => {
            extra.push(Op::CpAsyncWait { n: 0 });
            Op::CpAsyncCommit
        }
        // `cp.async.mbarrier.arrive[.noinc].shared{::cta}.b64 [mbar]`: `.noinc` is one
        // asynchronous arrival once this thread's cp.asyncs land; without it the pending count
        // is first raised by one (as ptxas: SYNCS.ARRIVE.TRANS64.RED.A0T1, one expected
        // transaction unit and no arrival, then ARRIVES.LDGSTSBAR.64.TRANSCNT completing it).
        // ASSUME(E-CPASYNC-MBAR): see ASSUMPTIONS.md
        "cp" if mods.len() >= 3 && mods[..3] == ["async", "mbarrier", "arrive"] => {
            let noinc = has("noinc");
            if !mods[3..]
                .iter()
                .all(|m| matches!(*m, "noinc" | "shared" | "shared::cta" | "b64"))
            {
                bail!(lw.err("cp.async.mbarrier.arrive variant not modeled"));
            }
            let addr = lw.addr(op_n(0)?)?;
            if noinc {
                Op::CpAsyncMbarArrive { addr, tx: false }
            } else {
                extra.push(Op::CpAsyncMbarArrive {
                    addr: addr.clone(),
                    tx: true,
                });
                Op::MbarArrive {
                    addr,
                    tx: Some(Val::Imm(1)),
                    remote: false,
                    count: Some(Val::Imm(0)),
                    complete: false,
                    cluster: false,
                }
            }
        }
        // Non-tensor bulk copies (as SASS UBLKCP; DeepEP's tma_load_1d / tma_store_1d):
        // `cp.async.bulk.shared::{cta,cluster}.global.mbarrier::complete_tx::bytes [dst], [src],
        // size, [mbar]` and `cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size`.
        // Lowered as TMA ops with `dims = 0`; `tmap` carries the byte count (exec.rs
        // `bulk_map`), `coords[0]` the global address.
        // `cp.async.bulk.prefetch.L2.global [a], size` (SASS UBLKPF.L2): a cache hint
        // (probes/memhints).
        "cp" if mods == ["async", "bulk", "prefetch", "L2", "global"] => Op::Nop,
        "cp" if mods.len() >= 4
            && mods[..2] == ["async", "bulk"]
            && !has("tensor")
            && !has("prefetch")
            && (mods[2].starts_with("shared") && mods[3] == "global"
                || mods[2] == "global" && mods[3] == "shared::cta") =>
        {
            // Fail closed on every other modifier (multicast, cache hints with their extra
            // operand, cta_group).
            let known = if mods[2] == "global" {
                mods.len() == 5 && mods[4] == "bulk_group"
            } else {
                mods.len() == 5
                    && mods[4] == "mbarrier::complete_tx::bytes"
                    && matches!(mods[2], "shared::cta" | "shared::cluster")
            };
            if !known {
                bail!(lw.err(format!(
                    "cp.async.bulk form .{} not modeled",
                    mods.join(".")
                )));
            }
            if ops.len() != if mods[2] == "global" { 3 } else { 4 } {
                bail!(lw.err("cp.async.bulk: unexpected operands"));
            }
            let reg_of = |a: Addr, what: &str| -> Result<(Val, u32)> {
                let b = a
                    .base
                    .ok_or_else(|| lw.err(format!("cp.async.bulk {what} must be a register")))?;
                Ok((b, a.off as u32))
            };
            let size = lw.val(op_n(2)?)?;
            if mods[2] == "global" {
                if !has("bulk_group") {
                    bail!(lw.err("cp.async.bulk store without .bulk_group not modeled"));
                }
                let g = lw.addr(op_n(0)?)?;
                if g.off != 0 {
                    bail!(lw.err("cp.async.bulk global address offset not modeled"));
                }
                let (src, off) = reg_of(lw.addr(op_n(1)?)?, "source")?;
                Op::TmaStore {
                    dims: 0,
                    src,
                    off,
                    tmap: size,
                    coords: vec![
                        g.base
                            .ok_or_else(|| lw.err("global address must be a register"))?,
                    ],
                    red: false,
                }
            } else {
                if !has("mbarrier::complete_tx::bytes") {
                    bail!(lw.err("cp.async.bulk load without an mbarrier not modeled"));
                }
                let g = lw.addr(op_n(1)?)?;
                if g.off != 0 {
                    bail!(lw.err("cp.async.bulk global address offset not modeled"));
                }
                let (dst, off) = reg_of(lw.addr(op_n(0)?)?, "destination")?;
                let (mbar, mbar_off) = reg_of(lw.addr(op_n(3)?)?, "mbarrier")?;
                Op::TmaLoad {
                    dims: 0,
                    dst,
                    off,
                    mbar,
                    mbar_off,
                    tmap: size,
                    coords: vec![
                        g.base
                            .ok_or_else(|| lw.err("global address must be a register"))?,
                    ],
                    mask: None,
                    pair: false,
                    src_smem: false,
                }
            }
        }
        "cp" if mods == ["async", "bulk", "commit_group"] => Op::BulkCommit,
        "cp" if mods.starts_with(&["async", "bulk", "wait_group"]) => {
            let n = lw.val(op_n(0)?)?;
            let Val::Imm(n) = n else {
                bail!(lw.err("wait_group count must be an immediate"))
            };
            Op::BulkWait {
                n: n as u32,
                read: has("read"),
                sb: None,
                cp: false,
            }
        }
        "cp" if mods.starts_with(&["async", "bulk", "tensor"]) => {
            let dims: u8 = mods[3]
                .trim_end_matches('d')
                .parse()
                .map_err(|_| lw.err("bad tensor dims"))?;
            let store = mods.get(4) == Some(&"global") && has("shared::cta") && has("bulk_group");
            if store {
                // cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group [tmap, {c..}], [src]
                let t = op_n(0)?
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_string();
                let (tm, coords) = t
                    .split_once(',')
                    .ok_or_else(|| lw.err("bad tensor operand"))?;
                let coords: Vec<Val> = coords
                    .trim()
                    .trim_start_matches('{')
                    .trim_end_matches('}')
                    .split(',')
                    .map(|c| lw.val(c))
                    .collect::<Result<_>>()?;
                if coords.len() != dims as usize {
                    bail!(lw.err("coordinate count != dims"));
                }
                let sa = lw.addr(op_n(1)?)?;
                let src = sa
                    .base
                    .ok_or_else(|| lw.err("TMA store source must be a register"))?;
                return Ok(Inst {
                    guard,
                    op: Op::TmaStore {
                        dims,
                        src,
                        off: sa.off as u32,
                        tmap: lw.val(tm)?,
                        coords,
                        red: false,
                    },
                    uniform: false,
                    loc: lw.line,
                    text: text.to_string(),
                });
            }
            if !((has("shared::cluster") || has("shared::cta"))
                && has("global")
                && has("mbarrier::complete_tx::bytes"))
            {
                bail!(lw.err("only global->shared TMA loads and shared::cta->global bulk_group stores are modeled so far"));
            }
            let da = lw.addr(op_n(0)?)?;
            let dst_off = da.off as u32;
            let dst = da
                .base
                .ok_or_else(|| lw.err("TMA dst must be a register"))?;
            let t = op_n(1)?
                .trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .to_string();
            let (tm, coords) = t
                .split_once(',')
                .ok_or_else(|| lw.err("bad tensor operand"))?;
            let coords: Vec<Val> = coords
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .split(',')
                .map(|c| lw.val(c))
                .collect::<Result<_>>()?;
            // `.tile::gather4` (2-D only): the column, then four row coordinates.
            let want = if has("tile::gather4") {
                5
            } else {
                dims as usize
            };
            if has("tile::gather4") && dims != 2 || coords.len() != want {
                bail!(lw.err("coordinate count != dims"));
            }
            let ma = lw.addr(op_n(2)?)?;
            let mbar = ma.base.ok_or_else(|| lw.err("mbar must be a register"))?;
            let mask = if has("multicast::cluster") {
                Some(lw.val(op_n(3)?)?)
            } else {
                None
            };
            Op::TmaLoad {
                dims,
                dst,
                off: dst_off,
                mbar,
                mbar_off: ma.off as u32,
                tmap: lw.val(tm)?,
                coords,
                mask,
                pair: has("cta_group::2"),
                src_smem: false,
            }
        }
        _ => bail!(lw.err(format!("unsupported PTX instruction `{opcode}`"))),
    };
    lw.extra = extra;
    Ok(Inst {
        guard,
        op,
        uniform: false,
        loc: lw.line,
        text: text.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixture() {
        let src = include_str!("../../../fixtures/r0_tma_sum/kernel.ptx");
        let k = parse_kernel(src, None).unwrap();
        assert_eq!(k.name, "r0_tma_sum");
        assert_eq!(k.params.len(), 2);
        assert_eq!(k.params[1].offset, 128);
        assert!(
            k.code
                .iter()
                .any(|i| matches!(i.op, Op::TmaLoad { dims: 2, .. }))
        );
        assert_eq!(k.smem_bytes, 0x400 + 8192 + 8);
    }

    fn one(line: &str) -> Result<Op> {
        let src = format!(
            ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry k()\n{{\n\
             .reg .pred %p<4>;\n.reg .b32 %r<8>;\n{line}\nret;\n}}\n"
        );
        Ok(parse_kernel(&src, None)?.code[0].op.clone())
    }

    #[test]
    fn redux_neg_setp_f32_forms() {
        let red = |l: &str| match one(l).unwrap() {
            Op::Redux { op, .. } => op,
            o => panic!("{l}: {o:?}"),
        };
        assert_eq!(
            red("redux.sync.max.NaN.f32 %r1, %r2, 0xffffffff;"),
            ReduxOp::MaxF {
                abs: false,
                nan: true
            }
        );
        assert_eq!(
            red("redux.sync.min.abs.f32 %r1, %r2, 0xffffffff;"),
            ReduxOp::MinF {
                abs: true,
                nan: false
            }
        );
        assert_eq!(red("redux.sync.add.s32 %r1, %r2, -1;"), ReduxOp::Sum);
        assert_eq!(
            red("redux.sync.max.s32 %r1, %r2, 0xffffffff;"),
            ReduxOp::MaxS
        );
        assert_eq!(red("redux.sync.or.b32 %r1, %r2, 0xffffffff;"), ReduxOp::Or);
        // Fail closed: partial member masks and forms that do not exist.
        assert!(one("redux.sync.max.NaN.f32 %r1, %r2, 0xffff;").is_err());
        assert!(one("redux.sync.add.f32 %r1, %r2, 0xffffffff;").is_err());
        assert!(one("redux.sync.max.NaN.abs.f32 %r1, %r2, 0xffffffff;").is_err());
        assert!(matches!(
            one("neg.f32 %r1, %r2;").unwrap(),
            Op::Fadd {
                neg: [true, false],
                ftz: false,
                ..
            }
        ));
        assert!(one("neg.ftz.f32 %r1, %r2;").is_err());
        assert!(matches!(
            one("setp.leu.f32 %p1, %r1, 0f00000000;").unwrap(),
            Op::Fsetp {
                cmp: FCmp::Leu,
                ftz: false,
                ..
            }
        ));
        assert!(matches!(
            one("setp.gt.ftz.f32 %p1, %r1, %r2;").unwrap(),
            Op::Fsetp {
                cmp: FCmp::Gt,
                ftz: true,
                ..
            }
        ));
        assert!(one("setp.lt.f32 %p1|%p2, %r1, %r2;").is_err());
        assert!(one("setp.lt.and.f32 %p1, %r1, %r2, %p3;").is_err());
    }

    /// Every op one statement lowers to (64-bit registers %rd0..7 declared too).
    fn ops(line: &str) -> Result<Vec<Op>> {
        let src = format!(
            ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry k()\n{{\n\
             .reg .pred %p<4>;\n.reg .b32 %r<8>;\n.reg .b64 %rd<8>;\n{line}\nret;\n}}\n"
        );
        let k = parse_kernel(&src, None)?;
        Ok(k.code[..k.code.len() - 1]
            .iter()
            .map(|i| i.op.clone())
            .collect())
    }

    #[test]
    fn packed_f32x2_forms() {
        let f2 = |l: &str| match &ops(l).unwrap()[..] {
            [Op::Ffma2 { a, b, c, neg, .. }] => (a.1, b.clone(), c.clone(), *neg),
            o => panic!("{l}: {o:?}"),
        };
        let pair = F2Lanes::Pair;
        let (a, b, c, neg) = f2("fma.rn.f32x2 %rd1, %rd2, %rd3, %rd4;");
        assert_eq!((a, b.1, c.1, neg), (pair, pair, pair, [false; 3]));
        // mul: a*b + (-0); add / sub: a*1 + (+-b).
        let (_, _, c, _) = f2("mul.rn.f32x2 %rd1, %rd2, %rd3;");
        assert_eq!(c, (Val::Imm(0x8000_0000), F2Lanes::Bcast));
        let (_, b, c, neg) = f2("add.rn.f32x2 %rd1, %rd2, %rd3;");
        assert_eq!(
            (b, c.1, neg),
            ((Val::Imm(0x3f80_0000), F2Lanes::Bcast), pair, [false; 3])
        );
        let (_, _, _, neg) = f2("sub.rn.f32x2 %rd1, %rd2, %rd3;");
        assert_eq!(neg, [false, false, true]);
        // Fail closed: no rounding modifier (ptxas may contract), other roundings, ftz, sat,
        // immediates.
        for l in [
            "fma.f32x2 %rd1, %rd2, %rd3, %rd4;",
            "mul.f32x2 %rd1, %rd2, %rd3;",
            "add.f32x2 %rd1, %rd2, %rd3;",
            "add.rn.ftz.f32x2 %rd1, %rd2, %rd3;",
            "mul.rn.sat.f32x2 %rd1, %rd2, %rd3;",
            "add.rn.f32x2 %rd1, %rd2, 0;",
        ] {
            assert!(ops(l).is_err(), "{l}");
        }
    }

    #[test]
    fn cvt_x2_vectors_setmaxnreg_forms() {
        assert!(matches!(
            &ops("cvt.rn.bf16x2.f32 %r1, %r2, %r3;").unwrap()[..],
            [Op::F2fpBf16 { .. }]
        ));
        assert!(matches!(
            &ops("cvt.rn.f16x2.f32 %r1, %r2, %r3;").unwrap()[..],
            [Op::F2fpF16 { .. }]
        ));
        assert!(ops("cvt.rz.bf16x2.f32 %r1, %r2, %r3;").is_err());
        assert!(ops("cvt.rn.relu.bf16x2.f32 %r1, %r2, %r3;").is_err());
        // Registers that are not consecutive: moves into the scratch block before a store,
        // out of it after a load.
        match &ops("st.shared.v4.b32 [%r1], {%r5, %r4, %r3, %r2};").unwrap()[..] {
            [
                Op::Mov { .. },
                Op::Mov { .. },
                Op::Mov { .. },
                Op::Mov { .. },
                Op::St {
                    bytes: 16,
                    v: Val::Reg(_, 4),
                    ..
                },
            ] => {}
            o => panic!("{o:?}"),
        }
        match &ops("ld.shared.v2.b32 {%r3, %r1}, [%r2];").unwrap()[..] {
            [
                Op::Ld { n: 2, bytes: 8, .. },
                Op::Mov { .. },
                Op::Mov { .. },
            ] => {}
            o => panic!("{o:?}"),
        }
        match &ops("ld.shared.v2.b64 {%rd1, %rd2}, [%r2];").unwrap()[..] {
            [
                Op::Ld {
                    n: 2, bytes: 16, ..
                },
            ] => {}
            o => panic!("{o:?}"),
        }
        assert!(ops("ld.shared.v4.b64 {%rd1, %rd2, %rd3, %rd4}, [%r2];").is_err());
        assert!(matches!(
            &ops("setmaxnreg.inc.sync.aligned.u32 232;").unwrap()[..],
            [Op::Nop]
        ));
        assert!(matches!(
            &ops("setmaxnreg.dec.sync.aligned.u32 24;").unwrap()[..],
            [Op::Nop]
        ));
        for l in [
            "setmaxnreg.inc.sync.aligned.u32 20;",
            "setmaxnreg.dec.sync.aligned.u32 100;",
            "setmaxnreg.inc.sync.aligned.u32 %r1;",
            "setmaxnreg.inc.u32 64;",
        ] {
            assert!(ops(l).is_err(), "{l}");
        }
    }
}
