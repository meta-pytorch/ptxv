/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Lift parsed LLO listings (`llo.rs`) into the TPU IR. Each mnemonic's operand roles and
//! meaning come from listings and hardware runs (fb/targets/tpu/notes/sim.md); modelling
//! choices are tagged with their assumption IDs.

use std::collections::BTreeMap;
use std::collections::HashMap;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_tpu_ir::*;

use crate::llo;
use crate::llo::RawBundle;
use crate::llo::RawOp;

/// Branch delay of `sbr.rel` in final listings, in bundles.
/// ASSUME(TPU-E-BR-DELAY)
pub const BRANCH_DELAY: u8 = 6;

/// `%s13_s9` -> s9; `%v3_v0` -> v0; original listings: `%s13` -> s#13; SSA-only `%110` -> None.
pub fn reg(tok: &str) -> Option<Reg> {
    let t = tok.strip_prefix('%')?;
    let file_of = |s: &str| -> Option<(File, u32)> {
        let l = s.bytes().take_while(u8::is_ascii_alphabetic).count();
        let (f, n) = s.split_at(l);
        let file = match f {
            "s" => File::S,
            "p" => File::P,
            "v" => File::V,
            "vm" => File::Vm,
            _ => return None,
        };
        Some((file, n.parse().ok()?))
    };
    if let Some((_, phys)) = t.rsplit_once('_') {
        let (file, idx) = file_of(phys)?;
        return Some(Reg { file, idx });
    }
    let (file, idx) = file_of(t)?;
    Some(Reg { file, idx })
}

fn parse_f32(s: &str) -> Option<f32> {
    match s {
        "inf" => Some(f32::INFINITY),
        "-inf" => Some(f32::NEG_INFINITY),
        "nan" => Some(f32::NAN),
        _ => {
            let ok = s
                .bytes()
                .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'-' | b'+' | b'e'));
            if ok && s.bytes().any(|c| c.is_ascii_digit()) {
                s.parse().ok()
            } else {
                None
            }
        }
    }
}

struct Lifter<'a> {
    front: Front,
    allocs: Vec<Alloc>,
    alloc_ids: HashMap<String, u32>,
    regions: &'a BTreeMap<u32, u32>,
    unlowered: Vec<Unlowered>,
}

impl Lifter<'_> {
    fn alloc(&mut self, name: &str, space: Space) -> u32 {
        if let Some(&i) = self.alloc_ids.get(name) {
            return i;
        }
        // Not declared in the original header: a register-allocator spill slot (`13_spill`),
        // given fresh space in the space it is used in.
        let i = self.allocs.len() as u32;
        self.allocs.push(Alloc {
            name: name.to_string(),
            space,
            size: 512,
            fixed: None,
            tag: String::new(),
        });
        self.alloc_ids.insert(name.to_string(), i);
        i
    }

    fn addr(&mut self, s: &str, space: Space) -> Result<Addr> {
        let mut terms = Vec::new();
        for t in s.split('+').map(str::trim).filter(|t| !t.is_empty()) {
            if let Some(a) = t.strip_prefix("#allocation") {
                terms.push(Term::Alloc(self.alloc(a, space)));
            } else if t.starts_with('%') {
                terms.push(Term::Reg(reg(t).ok_or_else(|| anyhow!("register {t}"))?));
            } else if let Some(h) = t.strip_prefix("$0x").or_else(|| t.strip_prefix("0x")) {
                terms.push(Term::Imm(i64::from_str_radix(h, 16)?));
            } else if let Some(d) = t.strip_prefix('$') {
                terms.push(Term::Imm(d.parse()?));
            } else {
                terms.push(Term::Imm(t.parse()?));
            }
        }
        Ok(Addr { terms })
    }

    /// Scalar-context value of a token.
    fn val(&mut self, tok: &str) -> Result<Val> {
        let mut t = tok.trim();
        if let Some(r) = t.strip_prefix("[resolvable:") {
            t = r.split_once(']').map(|x| x.1.trim()).unwrap_or(r);
        }
        if t.starts_with('%') {
            let t = t.split_whitespace().next().unwrap_or(t); // `%s210 [#allocation3]`
            return Ok(Val::Reg(
                reg(t).ok_or_else(|| anyhow!("register token {t}"))?,
            ));
        }
        if t.starts_with("[#allocation") || t.starts_with("#allocation") {
            return Ok(Val::Addr(
                self.addr(t.trim_matches(|c| c == '[' || c == ']'), Space::Smem)?,
            ));
        }
        if let Some(h) = t.strip_prefix("$0x") {
            return Ok(Val::Imm(u64::from_str_radix(h, 16)? as u32));
        }
        if let Some(d) = t.strip_prefix('$') {
            return Ok(Val::Imm(d.parse::<i64>()? as u32));
        }
        if t == "true" || t == "false" {
            return Ok(Val::Imm((t == "true") as u32));
        }
        if let Ok(v) = t.parse::<i64>() {
            return Ok(Val::Imm(v as u32));
        }
        if let Some(h) = t.strip_prefix("0x") {
            return Ok(Val::Imm(u64::from_str_radix(h, 16)? as u32));
        }
        if let Some(h) = t.strip_prefix("-0x") {
            return Ok(Val::Imm((-(i64::from_str_radix(h, 16)?)) as u32));
        }
        if let Some(f) = parse_f32(t) {
            return Ok(Val::Imm(f.to_bits()));
        }
        if t.contains(" [#allocation") {
            bail!("immediate with an allocation annotation {t:?} (meaning unobserved)");
        }
        bail!("scalar operand {t:?}")
    }

    /// `[vmem:[base] sm:$0xff ss:$8]` -> (space name, base text, attrs).
    fn mem(tok: &str) -> Result<(String, String, BTreeMap<String, u32>)> {
        let t = tok.trim();
        let inner = t
            .strip_prefix('[')
            .and_then(|x| x.strip_suffix(']'))
            .ok_or_else(|| anyhow!("memory operand {t:?}"))?;
        let (space, rest) = inner
            .split_once(":[")
            .ok_or_else(|| anyhow!("memory operand {t:?}"))?;
        let (base, attrs) = rest
            .split_once(']')
            .ok_or_else(|| anyhow!("memory operand {t:?}"))?;
        let mut m = BTreeMap::new();
        for a in attrs.split_whitespace() {
            if let Some((k, v)) = a.split_once(':') {
                if v.starts_with('%') {
                    bail!("{k}: from a register (unobserved form)");
                }
                let v = v.trim_start_matches('$');
                let n = match v.strip_prefix("0x") {
                    Some(h) => u32::from_str_radix(h, 16)?,
                    None if v.starts_with('-') => bail!("{k}: negative ({v}, unobserved form)"),
                    None => v.parse().map_err(|_| anyhow!("{k}: value {v:?}"))?,
                };
                m.insert(k.to_string(), n);
            }
        }
        Ok((space.to_string(), base.to_string(), m))
    }

    fn arg(&mut self, o: &RawOp, i: usize) -> Result<Val> {
        let t = o
            .args
            .get(i)
            .ok_or_else(|| anyhow!("missing operand {i}"))?
            .clone();
        self.val(&t)
    }

    fn named(&mut self, o: &RawOp, k: &str) -> Result<Val> {
        let t = o
            .named
            .get(k)
            .ok_or_else(|| anyhow!("missing operand {k}="))?
            .clone();
        self.val(&t)
    }

    fn reg_arg(o: &RawOp, i: usize) -> Result<Reg> {
        let t = o
            .args
            .get(i)
            .ok_or_else(|| anyhow!("missing operand {i}"))?;
        reg(t).ok_or_else(|| anyhow!("register operand {t}"))
    }

    fn bin2(&mut self, o: &RawOp) -> Result<(Val, Val)> {
        Ok((self.arg(o, 0)?, self.arg(o, 1)?))
    }

    fn unit(mn: &str, key: &str) -> u8 {
        mn.find(key)
            .and_then(|i| mn[i + key.len()..].chars().next())
            .and_then(|c| c.to_digit(10))
            .unwrap_or(0) as u8
    }

    fn slot(mn: &str) -> Option<u32> {
        let i = mn.find("mrb[")?;
        mn[i + 4..].split(']').next()?.parse().ok()
    }

    fn lower(&mut self, o: &RawOp) -> Result<Op> {
        let mn = o.mn.as_str();
        let orig = self.front == Front::Original;
        let parts: Vec<&str> = mn.split('.').collect();
        let sbin = |op| -> Option<SBin> { Some(op) };
        let s_op = match mn {
            "sadd.s32" => sbin(SBin::Add),
            "ssub.s32" => sbin(SBin::Sub),
            "smul.u32" | "smul.s32" | "smul.addr" => sbin(SBin::Mul),
            "smulhi.u32" => sbin(SBin::MulHiU),
            "sand.u32" => sbin(SBin::And),
            "sor.u32" => sbin(SBin::Or),
            "sxor.u32" => sbin(SBin::Xor),
            "sshll.u32" => sbin(SBin::Shl),
            "sshrl.u32" => sbin(SBin::Shrl),
            "sshra.s32" => sbin(SBin::Shra),
            "smin.u32" => sbin(SBin::MinU),
            "smax.u32" => sbin(SBin::MaxU),
            "smin.s32" => sbin(SBin::MinS),
            "smax.s32" => sbin(SBin::MaxS),
            "sdiv.u32" => sbin(SBin::DivU),
            "srem.u32" => sbin(SBin::RemU),
            "sdiv.s32" => sbin(SBin::DivS),
            "srem.s32" => sbin(SBin::RemS),
            "scalar_lea.smem" | "scalar_lea.vmem" | "scalar_lea.hbm" | "scalar_lea.sflag" => {
                sbin(SBin::Add)
            }
            _ => None,
        };
        if let Some(op) = s_op {
            let (a, b) = self.bin2(o)?;
            return Ok(Op::SBin { op, a, b });
        }
        let cond = |c: &str| -> Option<Cond> {
            Some(match c {
                "eq" => Cond::Eq,
                "ne" => Cond::Ne,
                "lt" => Cond::Lt,
                "le" => Cond::Le,
                "gt" => Cond::Gt,
                "ge" => Cond::Ge,
                _ => return None,
            })
        };
        let ty = |t: &str| -> Option<CmpTy> {
            Some(match t {
                "s32" => CmpTy::S32,
                "u32" => CmpTy::U32,
                "f32" => CmpTy::F32,
                _ => return None,
            })
        };
        if (parts[0] == "scmp" || parts[0] == "vcmp") && parts.len() >= 3 {
            let (Some(c), Some(t)) = (cond(parts[1]), ty(parts[2])) else {
                bail!("compare {mn}")
            };
            let (a, b) = self.bin2(o)?;
            return Ok(if parts[0] == "scmp" {
                Op::SCmp {
                    cond: c,
                    ty: t,
                    a,
                    b,
                }
            } else {
                Op::VCmp {
                    cond: c,
                    ty: t,
                    a,
                    b,
                }
            });
        }
        let fop = |f: &str| -> Option<FOp> {
            Some(match f {
                "vadd" => FOp::Add,
                "vsub" => FOp::Sub,
                "vmul" => FOp::Mul,
                "vmax" => FOp::Max,
                "vmin" => FOp::Min,
                _ => return None,
            })
        };
        if parts.len() == 2
            && let Some(f) = fop(parts[0])
            && (parts[1] == "f32" || parts[1] == "bf16")
        {
            let (a, b) = self.bin2(o)?;
            return Ok(if parts[1] == "f32" {
                Op::VF32 { op: f, a, b }
            } else {
                Op::VBf16 { op: f, a, b }
            });
        }
        let iop = match mn {
            "vadd.s32" => Some(IOp::Add),
            "vsub.s32" => Some(IOp::Sub),
            "vmul.u32" => Some(IOp::Mul),
            "vand.u32" => Some(IOp::And),
            "vor.u32" => Some(IOp::Or),
            "vxor.u32" => Some(IOp::Xor),
            // shift amounts mod 32 (unobserved for >= 32). ASSUME(TPU-N-VSHIFT)
            "vshll.u32" => Some(IOp::Shl),
            "vshrl.u32" => Some(IOp::Shrl),
            "vshra.s32" => Some(IOp::Shra),
            _ => None,
        };
        if let Some(op) = iop {
            let (a, b) = self.bin2(o)?;
            return Ok(Op::VInt { op, a, b });
        }
        let eupf = |n: &str| -> Option<EupFn> {
            Some(match n {
                "vrcp" => EupFn::Rcp,
                "vexp2" => EupFn::Exp2,
                "vlog2" => EupFn::Log2,
                "vrsqrt" => EupFn::Rsqrt,
                "vtanh" => EupFn::Tanh,
                "vsqrt" => EupFn::Sqrt,
                "vexp" => EupFn::Exp,
                "vlog" => EupFn::Log,
                _ => return None,
            })
        };
        if let Some(f) = eupf(parts[0])
            && parts.get(1) == Some(&"f32")
            && (parts.len() == 2 || (parts.len() == 3 && parts[2] == "pop"))
        {
            return Ok(Op::Eup {
                f,
                a: self.arg(o, 0)?,
                fused_pop: parts.len() == 3,
            });
        }
        if mn.starts_with("dma.") && mn.contains("_to_") && mn != "dma.done.wait" {
            let k = &mn[4..];
            let (s, d) = k.split_once("_to_").ok_or_else(|| anyhow!("dma kind"))?;
            let sp = |x: &str| -> Result<Space> {
                Ok(match x {
                    "hbm" => Space::Hbm,
                    "vmem" => Space::Vmem,
                    "smem" => Space::Smem,
                    _ => bail!("dma space {x}"),
                })
            };
            let kind = DmaKind {
                src: sp(s)?,
                dst: sp(d)?,
            };
            let stride = if o.named.contains_key("src_stride") {
                Some([
                    self.named(o, "src_stride")?,
                    self.named(o, "dst_stride")?,
                    self.named(o, "steps_per_stride")?,
                ])
            } else {
                None
            };
            return Ok(Op::Dma {
                kind,
                src: self.named(o, s)?,
                dst: self.named(o, d)?,
                size: self.named(o, "size_in_granules")?,
                flag: self.named(o, "dst_syncflagno")?,
                stride,
            });
        }
        if parts[0] == "vmatmul" {
            if orig {
                return Ok(Op::OMxuMatmul {
                    unit: Self::unit(mn, "mxu"),
                    a: self.arg(o, 0)?,
                    f32: mn.contains(".f32."),
                });
            }
            let ok = (mn.starts_with("vmatmul.mubr.f32.") || mn.starts_with("vmatmul.mubr.bf16."))
                && mn.contains(".gmra.mrb[");
            if !ok {
                bail!("matmul form {mn}");
            }
            return Ok(Op::MxuMatmul {
                unit: Self::unit(mn, "mxu"),
                a: self.arg(o, 0)?,
                slot: Self::slot(mn).ok_or_else(|| anyhow!("mrb slot"))?,
                f32: mn.contains(".f32."),
                latch: mn.contains(".vlgmr."),
            });
        }
        if mn.starts_with("vpop.f32.mrb[") {
            let unit = Self::unit(mn, "mxu");
            return Ok(if orig {
                Op::OMxuPop { unit }
            } else {
                Op::MxuPop {
                    unit,
                    slot: Self::slot(mn).ok_or_else(|| anyhow!("mrb slot"))?,
                }
            });
        }
        Ok(match mn {
            "int_to_ptr.vmem" | "int_to_ptr.hbm" | "int_to_ptr.smem" | "int_to_ptr.sflag" => {
                if orig {
                    let last = o
                        .args
                        .last()
                        .ok_or_else(|| anyhow!("int_to_ptr operand"))?
                        .clone();
                    Op::Copy(self.val(&last)?)
                } else {
                    Op::Nop
                }
            }
            "sphi" | "vphi" => {
                if orig {
                    Op::Phi {
                        init: self.arg(o, 0)?,
                        back: self.arg(o, 1)?,
                        vector: mn == "vphi",
                    }
                } else {
                    Op::Nop
                }
            }
            "xla_tuple" | "sfence" | "vdelay" | "snop" | "vnop" | "sdelay" | "vdwg.mxu0"
            | "vdwg.mxu1" => Op::Nop,
            m if m.starts_with("inlined_call_operand.") => {
                let i = o
                    .text
                    .find("index: ")
                    .ok_or_else(|| anyhow!("operand index"))?;
                let n: String = o.text[i + 7..]
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                Op::Operand { index: n.parse()? }
            }
            "smov" => Op::SMov(
                self.val(
                    o.args
                        .last()
                        .ok_or_else(|| anyhow!("smov"))?
                        .clone()
                        .as_str(),
                )?,
            ),
            "pmov" => Op::PMov(
                self.val(
                    o.args
                        .last()
                        .ok_or_else(|| anyhow!("pmov"))?
                        .clone()
                        .as_str(),
                )?,
            ),
            "scalar_select" => Op::SSelect {
                p: self.named(o, "predicate")?,
                t: self.named(o, "on_true")?,
                f: self.named(o, "on_false")?,
            },
            "pneg" => Op::PNeg(self.arg(o, 0)?),
            "por" | "pand" | "pnand" | "pnor" | "pxor" => {
                let op = match mn {
                    "por" => PBin::Or,
                    "pand" => PBin::And,
                    "pnand" => PBin::Nand,
                    "pnor" => PBin::Nor,
                    _ => PBin::Xor,
                };
                let (a, b) = self.bin2(o)?;
                Op::PBin { op, a, b }
            }
            "scvt.s32.f32" => Op::SCvtSF(self.arg(o, 0)?),
            "scvt.f32.s32" => Op::SCvtFS(self.arg(o, 0)?),
            "sld" => {
                let (_, base, _) = Self::mem(&o.args[0])?;
                Op::SLd(self.addr(&base, Space::Smem)?)
            }
            "sst" => {
                let (_, base, _) = Self::mem(&o.args[0])?;
                Op::SSt {
                    addr: self.addr(&base, Space::Smem)?,
                    val: self.arg(o, 1)?,
                }
            }
            "shalt.err" => Op::HaltErr {
                msg: o.text.clone(),
            },
            "shalt" => Op::Halt,
            "sbr.rel" => {
                let i = o
                    .text
                    .find("$region")
                    .ok_or_else(|| anyhow!("branch target"))?;
                let n: String = o.text[i + 7..]
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect();
                let r: u32 = n.parse()?;
                let target = *self
                    .regions
                    .get(&r)
                    .ok_or_else(|| anyhow!("branch target region {r} not found"))?;
                // delay and sync stops are filled in by `resolve_branches`
                Op::Branch { target, delay: 0 }
            }
            "vsyncadd" => Op::SyncAdd {
                flag: self.arg(o, 0)?,
                val: self.arg(o, 1)?,
            },
            // `vsyncpa f, k` sets the flag (seen only around the pipeline's own DMA flags).
            // ASSUME(TPU-S-SYNCPA)
            "vsyncpa" | "vsyncset" => Op::SyncSet {
                flag: self.arg(o, 0)?,
                val: self.arg(o, 1)?,
            },
            "vsyncmov" => Op::SyncMov {
                flag: self.arg(o, 0)?,
            },
            "vpop.sfrf" => Op::Pop(Fifo::Sfrf),
            "vwait.ge" | "dma.done.wait" => Op::Wait {
                flag: self.arg(o, 0)?,
                val: self.arg(o, 1)?,
                eq: false,
            },
            "vwait.eq" => Op::Wait {
                flag: self.arg(o, 0)?,
                val: self.arg(o, 1)?,
                eq: true,
            },
            "dma.done" => Op::DmaDone {
                flag: self.arg(o, 0)?,
                val: self.arg(o, 1)?,
            },
            "vsyncadd.remote" => Op::RemoteSyncAdd {
                addr: self.arg(o, 0)?,
                val: self.arg(o, 1)?,
            },
            // The source/destination spaces of `dma.general` are printed in a comment that
            // the parser strips; modelled as vmem-to-vmem (all corpus uses).
            // ASSUME(TPU-L-DMA-GENERAL-DEV)
            "dma.general" => Op::DmaGeneral {
                kind: DmaKind {
                    src: Space::Vmem,
                    dst: Space::Vmem,
                },
                src: self.named(o, "src")?,
                dst: self.named(o, "dst")?,
                size: self.named(o, "size_in_granules")?,
                dflag: self.named(o, "dst_syncflagno")?,
                sflag: self.named(o, "src_syncflagno")?,
                overrides: match o.named.get("overrides") {
                    Some(_) => self.named(o, "overrides")?,
                    None => Val::Imm(0),
                },
            },
            "vld" => {
                let (_, base, attrs) = Self::mem(&o.args[0])?;
                if attrs
                    .keys()
                    .any(|k| !matches!(k.as_str(), "sm" | "ss" | "sps"))
                {
                    bail!("vld attrs {attrs:?}");
                }
                // strided load: `sps` sublanes from each group `ss` rows apart (mm_bf16).
                // ASSUME(TPU-L-VLD-STRIDE)
                let stride = attrs
                    .get("ss")
                    .map(|&ss| (ss, attrs.get("sps").copied().unwrap_or(1)));
                Op::VLd {
                    addr: self.addr(&base, Space::Vmem)?,
                    sm: attrs.get("sm").copied().unwrap_or(0xff) as u8,
                    stride,
                }
            }
            "vst" | "vst.msk" => {
                let (_, base, attrs) = Self::mem(&o.args[0])?;
                if attrs.keys().any(|k| k != "sm") {
                    bail!("vst attrs {attrs:?}");
                }
                let src = match o.named.get("vst_source") {
                    Some(s) => self.val(&s.clone())?,
                    None => self.arg(o, 1)?,
                };
                let vm = if mn == "vst.msk" {
                    Some(
                        reg(o.named.get("vm").ok_or_else(|| anyhow!("vst.msk vm"))?)
                            .ok_or_else(|| anyhow!("vm"))?,
                    )
                } else {
                    None
                };
                Op::VSt {
                    addr: self.addr(&base, Space::Vmem)?,
                    sm: attrs.get("sm").copied().unwrap_or(0xff) as u8,
                    vm,
                    src,
                }
            }
            "vmov" => Op::VMov(
                self.val(
                    o.args
                        .last()
                        .ok_or_else(|| anyhow!("vmov"))?
                        .clone()
                        .as_str(),
                )?,
            ),
            "vstv" => Op::VStv(self.arg(o, 0)?),
            "vpack.c.f32.bf16" | "vpack.c.b32.b16" => {
                let (hi, lo) = self.bin2(o)?;
                Op::VPack {
                    hi,
                    lo,
                    int: mn == "vpack.c.b32.b16",
                }
            }
            "vunpack.c.l.bf16.f32"
            | "vunpack.c.h.bf16.f32"
            | "vunpack.c.u.bf16.f32"
            | "vunpack.c.l.s16.s32"
            | "vunpack.c.u.s16.s32" => Op::VUnpack {
                a: self.arg(o, 0)?,
                high: !mn.starts_with("vunpack.c.l."),
                int: mn.ends_with(".s16.s32"),
            },
            "vunpack.c.l.s8.bf16" | "vunpack.c.u.s8.bf16" => Op::VUnpackS8 {
                a: self.arg(o, 0)?,
                high: mn.starts_with("vunpack.c.u."),
            },
            "vcombine.low" | "vcombine.high" => {
                let (a, b) = self.bin2(o)?;
                Op::VCombine {
                    a,
                    b,
                    high: mn == "vcombine.high",
                }
            }
            "vmand" | "vmor" | "vmxor" => Op::VMask {
                op: match mn {
                    "vmand" => MOp::And,
                    "vmor" => MOp::Or,
                    _ => MOp::Xor,
                },
                a: Self::reg_arg(o, 0)?,
                b: Some(Self::reg_arg(o, 1)?),
            },
            "vmnot" => Op::VMask {
                op: MOp::Not,
                a: Self::reg_arg(o, 0)?,
                b: None,
            },
            "vsel" => Op::VSel {
                m: reg(o.named.get("vm").ok_or_else(|| anyhow!("vsel vm"))?)
                    .ok_or_else(|| anyhow!("vsel vm"))?,
                t: self.named(o, "on_true_vy")?,
                f: self.named(o, "on_false_vx")?,
            },
            "vtrunc.f32" => Op::VTrunc(self.arg(o, 0)?),
            // saturating, NaN -> 0 (unobserved; inputs in range in the corpus). ASSUME(TPU-N-CVT-SAT)
            "vcvt.f32.s32" => Op::VCvtFS(self.arg(o, 0)?),
            "vcvt.s32.f32" => Op::VCvtSF(self.arg(o, 0)?),
            "vcvt.f32.s32.to.zero.pseudo" => Op::VCvtTZ(self.arg(o, 0)?),
            "vpop.eup" => Op::Pop(Fifo::Eup),
            "vlaneseq.b32" => Op::VLaneSeq,
            "vperm.slane" => Op::VPermSlane {
                a: self.arg(o, 0)?,
                pat: self.arg(o, 1)?,
            },
            "vunpack.c.0.s8.s32" | "vunpack.c.1.s8.s32" | "vunpack.c.2.s8.s32"
            | "vunpack.c.3.s8.s32" => Op::VUnpackS8S32 {
                a: self.arg(o, 0)?,
                q: mn.as_bytes()[10] - b'0',
            },
            "vrot.slane" => Op::VRotSlane {
                a: self.arg(o, 0)?,
                k: self.arg(o, 1)?,
            },
            "vrot.lane.b32.xlu0" | "vrot.lane.b32.xlu1" => Op::VRotLane {
                unit: Self::unit(mn, "xlu"),
                a: self.arg(o, 0)?,
                k: self.arg(o, 1)?,
            },
            m if m.contains(".xlane.f32.xlu") => Op::XLane {
                unit: Self::unit(mn, "xlu"),
                op: match parts[0] {
                    "vadd" => XlaneOp::Add,
                    "vmax" => XlaneOp::Max,
                    "vmin" => XlaneOp::Min,
                    _ => bail!("xlane {mn}"),
                },
                a: self.arg(o, 0)?,
            },
            m if m.starts_with("vpop.xlane.xlu")
                || m.starts_with("vpop.permute.xlu")
                || m.starts_with("vpop.trf.xlu") =>
            {
                Op::Pop(Fifo::Xlu(Self::unit(mn, "xlu")))
            }
            m if m.starts_with("vxpose.xlu") => Op::VXpose {
                unit: Self::unit(mn, "xlu"),
                phase: match parts.last().copied() {
                    Some("start") => XposePhase::Start,
                    Some("cont") => XposePhase::Cont,
                    Some("end") => XposePhase::End,
                    _ => bail!("vxpose {mn}"),
                },
                a: self.named(o, "vx")?,
                width: o
                    .named
                    .get("width")
                    .and_then(|w| w.parse().ok())
                    .unwrap_or(128),
            },
            "vcmask" => Op::VCmask(self.arg(o, 0)?),
            "vpush" => Op::VPush(self.arg(o, 0)?),
            "spop" => Op::Pop(Fifo::V2s),
            "vtos" => Op::VTos(self.arg(o, 0)?),
            "vmatprep.subr.bf16.mxu0" | "vmatprep.subr.bf16.mxu1" => {
                let unit = Self::unit(mn, "mxu");
                if orig {
                    Op::OMxuPush {
                        unit,
                        a: self.arg(o, 0)?,
                        hi: true,
                        bf16: true,
                    }
                } else {
                    Op::MxuSubr {
                        unit,
                        a: self.arg(o, 0)?,
                    }
                }
            }
            "vmatpush1.bf16.msra.mxu0"
            | "vmatpush1.bf16.msra.mxu1"
            | "vmatpush3.bf16.msra.mxu0"
            | "vmatpush3.bf16.msra.mxu1"
            | "vmatpush1.bf16.xpose.msra.mxu0"
            | "vmatpush1.bf16.xpose.msra.mxu1"
            | "vmatpush3.bf16.xpose.msra.mxu0"
            | "vmatpush3.bf16.xpose.msra.mxu1" => {
                let unit = Self::unit(mn, "mxu");
                if orig && mn.contains(".xpose") {
                    bail!("original-front .xpose push not modeled");
                } else if orig {
                    Op::OMxuPush {
                        unit,
                        a: self.arg(o, 0)?,
                        hi: false,
                        bf16: true,
                    }
                } else {
                    Op::MxuPush {
                        unit,
                        a: self.arg(o, 0)?,
                    }
                }
            }
            "vmatprep.subr.mxu0"
            | "vmatprep.subr.mxu1"
            | "vmatpush1.msra.mxu0"
            | "vmatpush1.msra.mxu1" => Op::OMxuPush {
                unit: Self::unit(mn, "mxu"),
                a: self.arg(o, 0)?,
                hi: mn.starts_with("vmatprep"),
                bf16: false,
            },
            "vmatprep.mubr.bf16.mxu0"
            | "vmatprep.mubr.bf16.mxu1"
            | "vmatprep.mubr.f32.mxu0"
            | "vmatprep.mubr.f32.mxu1" => Op::MxuPrep {
                unit: Self::unit(mn, "mxu"),
                a: self.arg(o, 0)?,
            },
            _ => bail!("opcode not lowered"),
        })
    }

    fn inst(&mut self, bi: u32, o: &RawOp) -> Inst {
        let guard = o.pred.as_ref().map(|(neg, t)| match t.as_str() {
            "0" => Guard::Const(*neg),
            "1" => Guard::Const(!*neg),
            _ => match reg(t) {
                Some(reg) => Guard::Reg { reg, neg: *neg },
                None => Guard::Const(false),
            },
        });
        let op = match self.lower(o) {
            Ok(op) => op,
            Err(e) => {
                let idx = self.unlowered.len() as u32;
                self.unlowered.push(Unlowered {
                    bundle: bi,
                    mn: o.mn.clone(),
                    text: o.text.clone(),
                    why: e.to_string(),
                });
                Op::Unlowered { idx }
            }
        };
        Inst {
            guard,
            def: o.def.as_deref().and_then(reg),
            op,
            mn: o.mn.clone(),
            text: o.text.clone(),
        }
    }
}

fn space_of(s: &str) -> Space {
    match s {
        "vmem" => Space::Vmem,
        "sflag" => Space::Sflag,
        "hbm" => Space::Hbm,
        _ => Space::Smem,
    }
}

/// Operand declarations (`inlined_call_operand.<space> [shape: T[dims], index: N, kind: K`).
pub fn operand_decls(text: &str) -> Vec<OperandDecl> {
    let mut out: BTreeMap<u32, OperandDecl> = BTreeMap::new();
    let key = "inlined_call_operand.";
    let mut s = text;
    while let Some(i) = s.find(key) {
        let r = &s[i + key.len()..];
        s = r;
        let (space, rest) = match r.strip_prefix("<no memory space>") {
            Some(x) => ("none".to_string(), x),
            None => {
                let n = r.bytes().take_while(|c| c.is_ascii_alphanumeric()).count();
                (r[..n].to_string(), &r[n..])
            }
        };
        let Some(rest) = rest.trim_start().strip_prefix("[shape: ") else {
            continue;
        };
        let Some(br) = rest.find('[') else { continue };
        let dtype = rest[..br].to_string();
        let Some(close) = rest.find(']') else {
            continue;
        };
        let shape = rest[br + 1..close]
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        let tail = &rest[close..];
        let Some(ii) = tail.find("index: ") else {
            continue;
        };
        let idx: String = tail[ii + 7..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        let output = tail
            .find("kind: ")
            .is_some_and(|k| tail[k + 6..].starts_with("output"));
        let Ok(index) = idx.parse() else { continue };
        out.entry(index).or_insert(OperandDecl {
            index,
            space,
            dtype,
            shape,
            output,
        });
    }
    out.into_values().collect()
}

/// Fill in branch delays (final listings). With `enc` (decoded from the executable, keyed by
/// listing bundle address) each branch takes its encoded delay y and its target is checked
/// against imm0. ASSUME(TPU-E-BR-DELAY) Branches without an encoding get the inferred rule:
/// `BRANCH_DELAY` bundles, ended before the first bundle holding a sync-flag op (the compiler
/// never puts one inside a window). ASSUME(TPU-E-BR-DELAY-INFERRED) Original listings branch
/// immediately.
fn resolve_branches(k: &mut Kernel, enc: Option<&BTreeMap<u32, crate::exe::BranchEnc>>) {
    if k.front == Front::Original {
        // original listings: branches immediate, phis read the header-entry values, int_to_ptr
        // copies, MXU results through a FIFO. ASSUME(TPU-L-ORIG-FRONT)
        k.assumptions.push("TPU-L-ORIG-FRONT".into());
        return;
    }
    let n = k.bundles.len();
    let (mut binary, mut inferred) = (0, 0);
    for bi in 0..n {
        for ii in 0..k.bundles[bi].insts.len() {
            let Op::Branch { target, .. } = k.bundles[bi].insts[ii].op else {
                continue;
            };
            let addr = k.bundles[bi].addr;
            let want = k.bundles.get(target as usize).map(|b| b.addr as i64);
            let e = enc.and_then(|e| e.get(&addr)).copied();
            let e = match e {
                Some(e) if want != Some(addr as i64 + e.imm as i64) => {
                    k.notes.push(format!("branch at {addr:#x}: imm0 {:+} disagrees with listing target {want:?}; delay inferred", e.imm));
                    None
                }
                e => e,
            };
            let delay = match e {
                Some(e) => {
                    binary += 1;
                    e.y
                }
                None => {
                    inferred += 1;
                    let mut delay = BRANCH_DELAY;
                    for d in 1..=BRANCH_DELAY {
                        // past the end: the core finishes before the branch resolves
                        let Some(b) = k.bundles.get(bi + d as usize) else {
                            break;
                        };
                        if b.insts
                            .iter()
                            .any(|i| i.op.is_sync() && i.guard != Some(Guard::Const(false)))
                        {
                            delay = d - 1;
                            break;
                        }
                    }
                    delay
                }
            };
            if let Op::Branch { delay: dl, .. } = &mut k.bundles[bi].insts[ii].op {
                *dl = delay;
            }
        }
    }
    k.delay_source = match (binary, inferred) {
        (0, 0) => "none",
        (_, 0) => "binary",
        (0, _) => "inferred",
        _ => "mixed",
    }
    .into();
    if inferred > 0 {
        k.assumptions.push("TPU-E-BR-DELAY-INFERRED".into());
    }
}

fn lift_bundles(
    name: &str,
    front: Front,
    raw: &[RawBundle],
    regions: &BTreeMap<u32, u32>,
    original_text: &str,
    decl_text: &str,
    enc: Option<&BTreeMap<u32, crate::exe::BranchEnc>>,
) -> Kernel {
    let mut allocs = Vec::new();
    let mut alloc_ids = HashMap::new();
    let mut ra = llo::parse_allocs(original_text);
    ra.sort_by_key(|a| a.id);
    for a in ra {
        alloc_ids.insert(a.id.to_string(), allocs.len() as u32);
        allocs.push(Alloc {
            name: a.id.to_string(),
            space: space_of(&a.space),
            size: a.size,
            fixed: if a.fixed { a.offset } else { None },
            tag: a.tag.clone(),
        });
    }
    let mut l = Lifter {
        front,
        allocs,
        alloc_ids,
        regions,
        unlowered: Vec::new(),
    };
    let bundles = raw
        .iter()
        .enumerate()
        .map(|(bi, b)| Bundle {
            insts: b
                .ops
                .iter()
                .map(|o| {
                    let mut i = l.inst(bi as u32, o);
                    // the halt's reason is in the bundle's comment
                    if let Op::HaltErr { msg } = &mut i.op {
                        msg.push_str(" /* ");
                        msg.push_str(&b.comments.join(" "));
                        msg.push_str(" */");
                    }
                    i
                })
                .collect(),
            addr: b.addr,
            depth: b.depth,
        })
        .collect();
    let mut k = Kernel {
        name: name.to_string(),
        front,
        bundles,
        allocs: l.allocs,
        operands: operand_decls(decl_text),
        unlowered: l.unlowered,
        delay_source: "none".into(),
        assumptions: Vec::new(),
        notes: Vec::new(),
    };
    resolve_branches(&mut k, enc);
    k
}

/// Lift a `final_bundles` listing; `original` supplies the allocation header. Branch delays
/// are inferred (see `lift_final_exe`).
pub fn lift_final(name: &str, final_text: &str, original_text: &str) -> Kernel {
    lift_final_exe(name, final_text, original_text, None)
}

/// Lift a `final_bundles` listing with branch encodings decoded from its executable.
pub fn lift_final_exe(
    name: &str,
    final_text: &str,
    original_text: &str,
    enc: Option<&BTreeMap<u32, crate::exe::BranchEnc>>,
) -> Kernel {
    let raw = llo::parse(final_text);
    let mut regions = llo::regions(&raw);
    // A branch to a region no bundle marks (`target = $region9` with no `Start region 9`)
    // goes to the exit bundle (`/* exit bundle: ... */`, tagged PF): the kernel-skip branch of
    // serving and Kilo listings. Checked against imm0 when the executable is available.
    // ASSUME(TPU-L-REGION-EXIT)
    let mut exit_used = Vec::new();
    if let Some(exit) = raw
        .iter()
        .position(|b| b.comments.iter().any(|c| c.contains("exit bundle")))
    {
        for o in raw
            .iter()
            .flat_map(|b| &b.ops)
            .filter(|o| o.mn == "sbr.rel")
        {
            let Some(i) = o.text.find("$region") else {
                continue;
            };
            let n: String = o.text[i + 7..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if let Ok(r) = n.parse::<u32>()
                && !regions.contains_key(&r)
            {
                regions.insert(r, exit as u32);
                exit_used.push(r);
            }
        }
    }
    let mut k = lift_bundles(
        name,
        Front::Final,
        &raw,
        &regions,
        original_text,
        final_text,
        enc,
    );
    if !exit_used.is_empty() {
        k.assumptions.push("TPU-L-REGION-EXIT".into());
        k.notes.push(format!(
            "branch target regions {exit_used:?} unmarked: taken as the exit bundle"
        ));
    }
    k
}

/// Lift an `original` listing.
pub fn lift_original(name: &str, original_text: &str) -> Kernel {
    let (raw, regions) = llo::parse_original(original_text);
    lift_bundles(
        name,
        Front::Original,
        &raw,
        &regions,
        original_text,
        original_text,
        None,
    )
}

/// Opcodes a listing uses that the front-end does not lower, with counts.
pub fn lift_report(k: &Kernel) -> Vec<(String, usize, String)> {
    let mut m: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for u in &k.unlowered {
        let e = m.entry(u.mn.clone()).or_insert((0, u.why.clone()));
        e.0 += 1;
    }
    let mut v: Vec<_> = m.into_iter().map(|(k, (n, w))| (k, n, w)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regs() {
        assert_eq!(
            reg("%s13_s9"),
            Some(Reg {
                file: File::S,
                idx: 9
            })
        );
        assert_eq!(
            reg("%vm17_vm0"),
            Some(Reg {
                file: File::Vm,
                idx: 0
            })
        );
        assert_eq!(
            reg("%v3"),
            Some(Reg {
                file: File::V,
                idx: 3
            })
        );
        assert_eq!(reg("%110"), None);
    }

    #[test]
    fn delay_window_stops_at_sync() {
        let fin = "  0x0   :  { %1 = sbr.rel (%p1_p1) target = $region9 } /* Start region 8 */\n\
                   0x1   :  {}\n\
                   0x2   :  { %2 = vsyncadd [#allocation4], 1 }\n\
                   0x3 PF:  { %3 = vsyncadd [#allocation4], 1 } /* Start/End empty region 9 */\n";
        let k = lift_final(
            "t",
            fin,
            "#allocation4 [shape = 's32[1]{0}', space=sflag, size = 0x4, scoped]\n",
        );
        assert!(k.unlowered.is_empty(), "{:?}", k.unlowered);
        assert_eq!(
            k.bundles[0].insts[0].op,
            Op::Branch {
                target: 3,
                delay: 1
            }
        );
    }
}
