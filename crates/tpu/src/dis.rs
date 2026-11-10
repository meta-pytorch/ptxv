/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! tpu7x TensorCore bundle decoder (port of fb/targets/tpu/tools/tpudis.py): a 64-byte bundle
//! -> listing-like ops, plus the listing comparator (`canon_listing`, `op_match`) that scores it
//! against `final_bundles` listings. Every field is documented with its evidence in
//! fb/targets/tpu/probes/binary/FINDINGS.md. ASSUME(TPU-L-BIN-DECODE)
//!
//! A bundle is a 512-bit little-endian integer X; field (o, w) = (X >> o) & (2^w - 1).
//! Slot map (low .. high): res1 0, res0 11, ext1 22, ext0 47, misc 72..96, vld1 97, vld0 120,
//! vst 143..170, valu3 171, valu2 204, valu1 237, valu0 270, vs3..vs0 303..322, imm5..imm0
//! 323..442, S1 443, S0 467, pred1 496, pred0 501. Slot predication pr: 0 empty, 1 always,
//! 2 pred0, 3 pred1.

use crate::llo;
use crate::llo::RawOp;
use crate::optab;
use crate::optab::Dk;

const MASK32: u64 = 0xFFFF_FFFF;

/// A bundle as eight little-endian 64-bit words.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bits(pub [u64; 8]);

impl Bits {
    pub fn new(raw: &[u8; 64]) -> Self {
        let mut w = [0u64; 8];
        for (i, c) in raw.chunks_exact(8).enumerate() {
            w[i] = u64::from_le_bytes(c.try_into().expect("8 bytes"));
        }
        Bits(w)
    }

    /// Field (o, w), w <= 32.
    pub fn f(&self, o: usize, w: usize) -> u64 {
        let (i, s) = (o / 64, o % 64);
        let mut v = self.0[i] >> s;
        if s + w > 64 && i + 1 < 8 {
            v |= self.0[i + 1] << (64 - s);
        }
        v & ((1u64 << w) - 1)
    }

    pub fn imm(&self, k: usize) -> u64 {
        self.f(423 - 20 * k, 20)
    }

    pub fn vs(&self, k: usize) -> u64 {
        self.f(318 - 5 * k, 5)
    }
}

fn sext20(v: u64) -> i64 {
    if v & 0x80000 != 0 {
        v as i64 - (1 << 20)
    } else {
        v as i64
    }
}

/// One decoded op, in the canonical listing form of tools/tpudis.py.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dec {
    pub slot: &'static str,
    pub mn: String,
    pub dst: Option<String>,
    pub srcs: Vec<String>,
    pub pred: Option<String>,
    /// Slot word (ext: the 23-bit value; others: 0).
    pub word: u64,
}

impl Dec {
    fn new(slot: &'static str, mn: impl Into<String>, pred: Option<String>) -> Self {
        Dec {
            slot,
            mn: mn.into(),
            dst: None,
            srcs: Vec::new(),
            pred,
            word: 0,
        }
    }

    fn dst(mut self, d: impl Into<String>) -> Self {
        self.dst = Some(d.into());
        self
    }

    fn srcs(mut self, s: Vec<String>) -> Self {
        self.srcs = s;
        self
    }
}

/// `dst = mn (pred) srcs` as tpudis.py prints it.
pub fn fmt(o: &Dec) -> String {
    let p = o
        .pred
        .as_ref()
        .map(|p| format!("({p}) "))
        .unwrap_or_default();
    let d = o
        .dst
        .as_ref()
        .map(|d| format!("{d} = "))
        .unwrap_or_default();
    format!("{d}{} {p}{}", o.mn, o.srcs.join(", "))
        .trim_end()
        .to_string()
}

// ---- scalar -----------------------------------------------------------------------------

fn sy_const(y: u64) -> Option<u64> {
    Some(match y {
        46 => 0x100,
        47 => MASK32,
        48..=56 => y - 48,
        57 => 0x10,
        58 => 0x18,
        59 => 0x20,
        60 => 0x30,
        61 => 0x40,
        62 => 0x60,
        63 => 0x80,
        _ => return None,
    })
}

/// ScalarY: Err(reg) or Ok(Some(value)) / Ok(None) (unknown code).
enum Sy {
    Reg(u64),
    Imm(Option<u64>),
}

fn scalar_y(x: &Bits, y: u64) -> Sy {
    if y < 32 {
        return Sy::Reg(y);
    }
    if let Some(c) = sy_const(y) {
        return Sy::Imm(Some(c));
    }
    Sy::Imm(Some(match y {
        32..=35 => x.imm((y - 32) as usize),
        36..=39 => (x.imm((y - 36) as usize) | 0xFFF0_0000) & MASK32,
        40..=43 => (x.imm((y - 40) as usize) << 12) & MASK32,
        44 => ((x.imm(1) & 0xFFFF) << 16) | (x.imm(0) & 0xFFFF),
        45 => ((x.imm(3) & 0xFFFF) << 16) | (x.imm(2) & 0xFFFF),
        _ => return Sy::Imm(None),
    }))
}

fn fmt_y(v: &Sy) -> String {
    match v {
        Sy::Reg(n) => format!("s{n}"),
        Sy::Imm(Some(n)) => n.to_string(),
        Sy::Imm(None) => "?".into(),
    }
}

fn pred_str(x: &Bits, pr: u64) -> Option<String> {
    match pr {
        1 => None,
        0 => Some("never".into()),
        _ => {
            let o = if pr == 2 { 501 } else { 496 };
            Some(format!(
                "{}p{}",
                if x.f(o + 4, 1) != 0 { "!" } else { "" },
                x.f(o, 4)
            ))
        }
    }
}

fn pred_or(v: u64) -> String {
    let (neg, b) = (v >= 16, v & 15);
    let s = match b {
        14 => "1".to_string(),
        15 => "0".to_string(),
        _ => format!("p{b}"),
    };
    format!("{}{s}", if neg { "!" } else { "" })
}

#[derive(Clone, Copy, PartialEq)]
enum SKind {
    Yx,
    Xy,
    Cmp,
    Pred,
    Ld,
    LdXy,
    St,
}

fn sops(op: u64) -> Option<(&'static str, SKind)> {
    use SKind::*;
    Some(match op {
        1 => ("sld", Ld),
        2 => ("sld", LdXy),
        3 => ("sst", St),
        10 => ("sadd.s32", Yx),
        12 => ("ssub.s32", Yx),
        14 => ("sand.u32", Yx),
        15 => ("sor.u32", Yx),
        16 => ("sxor.u32", Yx),
        20 => ("smul.u32", Yx),
        23 => ("sshll.u32", Xy),
        24 => ("sshrl.u32", Xy),
        25 => ("sshra.s32", Xy),
        29 => ("smin.u32", Xy),
        30 => ("scmp.eq.s32.totalorder", Cmp),
        31 => ("scmp.ne.s32.totalorder", Cmp),
        32 => ("scmp.gt.s32.totalorder", Cmp),
        33 => ("scmp.ge.s32.totalorder", Cmp),
        34 => ("scmp.lt.s32.totalorder", Cmp),
        35 => ("scmp.le.s32.totalorder", Cmp),
        38 => ("scmp.lt.u32.totalorder", Cmp),
        40 => ("sc.u32", Cmp),
        41 => ("por", Pred),
        43 => ("scmp.ne.f32.partialorder", Cmp),
        _ => return None,
    })
}

fn sop0(x: u64) -> Option<&'static str> {
    Some(match x {
        0 => "shalt",
        1 => "spop",
        2 => "spop.drf",
        5 => "sbr.rel",
        9 => "sfence",
        11 => "scvt.s32.f32",
        12 => "scvt.f32.s32",
        13 => "smov",
        15 => "sceil.f32",
        16 => "sfloor.f32",
        17 => "vpop.sfrf",
        _ => return None,
    })
}

fn dec_scalar(x: &Bits, base: usize, name: &'static str) -> Option<Dec> {
    let (d, y, xr, op, pr) = (
        x.f(base, 5),
        x.f(base + 5, 6),
        x.f(base + 11, 5),
        x.f(base + 16, 6),
        x.f(base + 22, 2),
    );
    if pr == 0 {
        return None;
    }
    let p = pred_str(x, pr);
    let yv = scalar_y(x, y);
    if op == 0 {
        let mn = sop0(xr).map(str::to_string).unwrap_or(format!("sop0.{xr}"));
        let o = Dec::new(name, mn.clone(), p);
        return Some(match mn.as_str() {
            "smov" | "scvt.s32.f32" | "scvt.f32.s32" | "sceil.f32" | "sfloor.f32" => {
                o.dst(format!("s{d}")).srcs(vec![fmt_y(&yv)])
            }
            "vpop.sfrf" | "spop.drf" | "spop" => o.dst(format!("s{d}")),
            // target = pc + imm0; y = delay (hardware-verified). ASSUME(TPU-E-BR-DELAY)
            "sbr.rel" => o.srcs(vec![
                format!("target={}", sext20(x.imm(0))),
                format!("delay={y}"),
            ]),
            "shalt" if pr != 1 => Dec {
                mn: "shalt.err".into(),
                ..o
            },
            _ => o,
        });
    }
    if op == 21 {
        return Some(
            Dec::new(name, "smulhi.u32", p)
                .dst(format!("s{d}"))
                .srcs(vec![fmt_y(&yv), format!("s{xr}")]),
        );
    }
    if op == 22 {
        return Some(Dec::new(name, "sdivrem.u32", p).srcs(vec![format!("s{xr}"), fmt_y(&yv)]));
    }
    let (mn, kind) = match sops(op) {
        Some((m, k)) => (m.to_string(), k),
        None => (format!("sop{op}"), SKind::Yx),
    };
    let o = Dec::new(name, mn, p);
    Some(match kind {
        SKind::Yx => o
            .dst(format!("s{d}"))
            .srcs(vec![fmt_y(&yv), format!("s{xr}")]),
        SKind::Xy => o
            .dst(format!("s{d}"))
            .srcs(vec![format!("s{xr}"), fmt_y(&yv)]),
        SKind::Cmp => o
            .dst(format!("p{d}"))
            .srcs(vec![format!("s{xr}"), fmt_y(&yv)]),
        SKind::Pred => {
            let (a, b) = (pred_or(xr), pred_or(y));
            let o = o.dst(format!("p{d}"));
            if a.starts_with('!') && b.starts_with('!') {
                if a != b {
                    Dec {
                        mn: "pnand".into(),
                        ..o
                    }
                    .srcs(vec![a[1..].to_string(), b[1..].to_string()])
                } else {
                    Dec {
                        mn: "pneg".into(),
                        ..o
                    }
                    .srcs(vec![a[1..].to_string()])
                }
            } else if a == "1" && b == "1" {
                Dec {
                    mn: "pmov".into(),
                    ..o
                }
                .srcs(vec!["1".into()])
            } else {
                o.srcs(vec![a, b])
            }
        }
        SKind::Ld => o
            .dst(format!("s{d}"))
            .srcs(vec![format!("[smem:[{}]]", fmt_y(&yv))]),
        SKind::LdXy => o
            .dst(format!("s{d}"))
            .srcs(vec![format!("[smem:[{} + s{xr}]]", fmt_y(&yv))]),
        SKind::St => o.srcs(vec![format!("[smem:[{}]] s{xr}", fmt_y(&yv))]),
    })
}

fn mem_name(core: u64, mem: u64) -> Option<&'static str> {
    Some(match (core, mem) {
        (0, 0) => "vmem",
        (0, 1) => "smem",
        (1, 0) => "hbm",
        (1, 3) => "host_iova",
        _ => return None,
    })
}

fn dec_dma_general(x: &Bits) -> Dec {
    let sd = scalar_y(x, x.f(448, 6));
    Dec::new("dma", "dma.general", pred_str(x, x.f(489, 2))).srcs(vec![
        format!("src=s{}", x.f(478, 5)),
        format!("size_in_granules={}", fmt_y(&scalar_y(x, x.f(472, 6)))),
        format!("dst=s{}", x.f(454, 5)),
        format!("dst_syncflagno=s{}", x.vs(1)),
        format!("src_syncflagno=s{}", x.vs(0)),
        format!("stridedescaddr={}", fmt_y(&sd)),
        format!("overrides=s{}", x.vs(2)),
    ])
}

fn dec_dma(x: &Bits) -> Dec {
    let s0op = x.f(483, 6);
    let dst = mem_name(x.f(469, 2), x.f(467, 2))
        .map(str::to_string)
        .unwrap_or(format!("m{}", x.f(467, 5)));
    let src = mem_name(x.f(461, 2), x.f(459, 2))
        .map(str::to_string)
        .unwrap_or(format!("m{}", x.f(459, 4)));
    let th = x.f(464, 3);
    let size = scalar_y(x, x.f(472, 6));
    let flag = scalar_y(x, x.f(448, 6));
    let flag_s = match &flag {
        Sy::Imm(Some(v)) => format!("[{v}]"),
        Sy::Imm(None) => "[None]".into(),
        r => fmt_y(r),
    };
    let mut srcs = vec![
        format!("[thread:${th}]"),
        format!("{src}=s{}", x.f(478, 5)),
        format!("size_in_granules={}", fmt_y(&size)),
        format!("{dst}=s{}", x.f(454, 5)),
        format!("dst_syncflagno={flag_s}"),
    ];
    if s0op == 7 {
        srcs.push(format!("src_stride=s{}", x.vs(0)));
        srcs.push(format!("dst_stride=s{}", x.vs(1)));
        srcs.push(format!("steps_per_stride=s{}", x.vs(2)));
    }
    if dst == "host_iova" {
        srcs.push(format!("host_iova_high=s{}", x.vs(3)));
    }
    Dec::new(
        "dma",
        format!("dma.{src}_to_{dst}"),
        pred_str(x, x.f(489, 2)),
    )
    .srcs(srcs)
}

// ---- vector ALU --------------------------------------------------------------------------

fn vops(op: u64) -> Option<&'static str> {
    Some(match op {
        3 => "vadd.s32",
        4 => "vsub.s32",
        5 => "vmul.u32",
        6 => "vand.u32",
        7 => "vor.u32",
        8 => "vxor.u32",
        9 => "vshll.u32",
        10 => "vshrl.u32",
        11 => "vshra.s32",
        12 => "vadd.f32",
        13 => "vsub.f32",
        14 => "vmul.f32",
        15 => "vmax.f32",
        16 => "vmin.f32",
        22 => "vmov",
        23 => "vrot.slane",
        38 => "vcmp.eq.s32.totalorder",
        40 => "vcmp.gt.s32.totalorder",
        42 => "vcmp.lt.s32.totalorder",
        46 => "vcmp.eq.f32.partialorder",
        50 => "vcmp.lt.f32.partialorder",
        55 => "vmmov",
        _ => return None,
    })
}

fn unary(ys: u64) -> Option<&'static str> {
    Some(match ys {
        1 => "vunpack.c.l.bf16.f32",
        3 => "vceil.f32",
        4 => "vfloor.f32",
        5 => "vcvt.s32.f32",
        6 => "vcvt.f32.s32",
        10 => "vlaneseq.b32",
        _ => return None,
    })
}

fn eup(ys: u64) -> Option<&'static str> {
    Some(match ys {
        16 => "vrsqrt.f32",
        17 => "vexp2.f32",
        18 => "vlog2.f32",
        19 => "vtanh.f32",
        21 => "vrcp.f32",
        _ => return None,
    })
}

fn vector_y(x: &Bits, ysel: u64, yreg: u64) -> String {
    match ysel {
        0 => format!("v{yreg}"),
        1 => "1".into(),
        2 => MASK32.to_string(),
        3 => "0".into(),
        4 => 0x3F80_0000u64.to_string(),
        5 => 0xBF80_0000u64.to_string(),
        6 => 0x3F00_0000u64.to_string(),
        7..=12 => x.imm((ysel - 7) as usize).to_string(),
        13..=18 => ((x.imm((ysel - 13) as usize) | 0xFFF0_0000) & MASK32).to_string(),
        19..=24 => ((x.imm((ysel - 19) as usize) << 12) & MASK32).to_string(),
        25..=27 => {
            let k = 2 * (ysel - 25) as usize;
            (((x.imm(k + 1) & 0xFFFF) << 16) | (x.imm(k) & 0xFFFF)).to_string()
        }
        _ => format!("s{}", x.vs((ysel - 28) as usize)),
    }
}

/// C `%.{prec}g` (Python format spec `.{prec}g`).
fn fmt_g(f: f64, prec: usize) -> String {
    let p = prec.max(1);
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let e = format!("{:.*e}", p - 1, f);
    let (mant, exp) = e.split_once('e').expect("exponent");
    let x: i32 = exp.parse().expect("exponent digits");
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if x < -4 || x >= p as i32 {
        let m = strip(mant);
        format!("{m}e{}{:02}", if x < 0 { '-' } else { '+' }, x.abs())
    } else {
        strip(&format!("{:.*}", (p as i32 - 1 - x) as usize, f))
    }
}

/// Float ops print immediates as floats in the listing (tpudis.fmt_vimm).
pub fn fmt_vimm(mn: &str, s: &str) -> String {
    if !mn.contains(".f32") || s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
        return s.to_string();
    }
    let Ok(v) = s.parse::<u64>() else {
        return s.to_string();
    };
    let f32v = f32::from_bits(v as u32);
    let f = f32v as f64;
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == f.trunc() && f.abs() < 1e15 {
        return format!("{f:.1}");
    }
    for prec in 1..10 {
        let t = fmt_g(f, prec);
        if t.parse::<f32>().map(f32::to_bits) == Ok(f32v.to_bits()) {
            return t;
        }
    }
    format!("{f}")
}

const VALU_BASES: [usize; 4] = [270, 237, 204, 171];

fn table<K: PartialEq + Copy>(t: &[(K, optab::Ent)], k: K) -> Option<optab::Ent> {
    t.iter().find(|e| e.0 == k).map(|e| e.1)
}

fn dec_valu(x: &Bits, base: usize, name: &'static str) -> Option<Dec> {
    let (d, yreg, ysel, xr, op, pr) = (
        x.f(base, 6),
        x.f(base + 6, 6),
        x.f(base + 12, 5),
        x.f(base + 17, 6),
        x.f(base + 23, 8),
        x.f(base + 31, 2),
    );
    if pr == 0 {
        return None;
    }
    let o = Dec::new(name, "", pred_str(x, pr));
    let with = |o: Dec, mn: &str| Dec {
        mn: mn.to_string(),
        ..o
    };
    if op == 55 && ysel == 2 {
        return Some(
            with(o, "vmmov")
                .dst(format!("vm{d}"))
                .srcs(vec!["1".into()]),
        );
    }
    if op == 22 && ysel >= 28 {
        return Some(
            with(o, "vstv")
                .dst(format!("v{d}"))
                .srcs(vec![vector_y(x, ysel, yreg)]),
        );
    }
    if op == 27 && matches!(ysel, 26 | 28 | 30) {
        let acc = VALU_BASES
            .iter()
            .position(|&b| b == base)
            .expect("valu base");
        return Some(if ysel == 30 {
            with(o, &format!("vmovacc.low.even.vacc{acc}")).dst(format!("v{d}"))
        } else {
            let mn = if ysel == 26 {
                "vmaxabs.f32"
            } else {
                "vmaxabs.init.f32"
            };
            with(o, &format!("{mn}.vacc{acc}")).srcs(vec![format!("v{yreg}"), format!("v{xr}")])
        });
    }
    if op == 128 {
        return Some(with(o, "vpack.c.f32.eXmY").dst(format!("v{d}")).srcs(vec![
            format!("vx=v{xr}"),
            format!("vy=v{yreg}"),
            format!("cvtdesc={}", vector_y(x, ysel, 0)),
        ]));
    }
    let fam = matches!(op, 0 | 1 | 2 | 27);
    let ent = if fam {
        table(optab::VALU_FAM, (op as u32, ysel as u32))
            .or_else(|| table(optab::VALU_PRED_FAM, (op as u32, ysel as u32)))
    } else {
        table(optab::VALU_OPS, op as u32).or_else(|| table(optab::VALU_PRED, op as u32))
    };
    if let Some((mn, tmpl, dk)) = ent {
        let yi = vector_y(x, ysel, yreg);
        let y = if tmpl.ends_with("_i") {
            yi
        } else {
            fmt_vimm(mn, &yi)
        };
        let xv = format!("v{xr}");
        let yr = format!("v{yreg}");
        let srcs = match tmpl.trim_end_matches("_i") {
            "yx" => vec![y, xv],
            "xy" => vec![xv, y],
            "x" => vec![xv],
            "y" => vec![y],
            "yrx" => vec![yr, xv],
            "xyr" => vec![xv, yr],
            _ => vec![],
        };
        let o = with(o, mn).srcs(srcs);
        return Some(match dk {
            Some(Dk::V) => o.dst(format!("v{d}")),
            Some(Dk::Vm) => o.dst(format!("vm{d}")),
            None => o,
        });
    }
    if op == 0 {
        return Some(if let Some(m) = eup(ysel) {
            with(o, m).srcs(vec![format!("v{xr}")])
        } else if ysel == 10 {
            with(o, "vlaneseq.b32").dst(format!("v{d}"))
        } else {
            let m = unary(ysel)
                .map(str::to_string)
                .unwrap_or(format!("vunary.{ysel}"));
            with(o, &m)
                .dst(format!("v{d}"))
                .srcs(vec![format!("v{xr}")])
        });
    }
    if op == 27 {
        let m = match ysel {
            7 => "vpack.c.f32.bf16".to_string(),
            20 => "vtrunc.f32".to_string(),
            _ => format!("vop27.{ysel}"),
        };
        return Some(
            with(o, &m)
                .dst(format!("v{d}"))
                .srcs(vec![format!("v{xr}")]),
        );
    }
    if (96..=127).contains(&op) {
        let (y, xv) = (vector_y(x, ysel, yreg), format!("v{xr}"));
        let (t, f) = if op < 112 { (y, xv) } else { (xv, y) };
        return Some(with(o, "vsel").dst(format!("v{d}")).srcs(vec![
            format!("vm=vm{}", op & 15),
            format!("on_true_vy={t}"),
            format!("on_false_vx={f}"),
        ]));
    }
    let mn = vops(op).map(str::to_string).unwrap_or(format!("vop{op}"));
    let dst = if mn.starts_with("vcmp") {
        format!("vm{d}")
    } else {
        format!("v{d}")
    };
    let y = fmt_vimm(&mn, &vector_y(x, ysel, yreg));
    let xf = ["vshll", "vshrl", "vshra", "vcmp", "vrot"]
        .iter()
        .any(|p| mn.starts_with(p));
    let srcs = if xf {
        vec![format!("v{xr}"), y]
    } else {
        vec![y, format!("v{xr}")]
    };
    Some(with(o, &mn).dst(dst).srcs(srcs))
}

// ---- vector load / store ---------------------------------------------------------------

fn mask_str(x: &Bits, m: u64) -> String {
    match m {
        0 => "0xff".into(),
        11 => "0x1".into(),
        12 => "0xf".into(),
        13 => "0xf0".into(),
        14 => "0x3".into(),
        15 => "0x10".into(),
        5..=10 => format!("{:#x}", x.imm((m - 5) as usize)),
        _ => format!("s{}", x.vs((m as usize).wrapping_sub(1) & 3)),
    }
}

fn stride_const(c: u64) -> Option<&'static str> {
    Some(match c {
        11 => "-1",
        12 => "2",
        13 => "4",
        14 => "8",
        15 => "16",
        _ => return None,
    })
}

fn voff(x: &Bits, s: u64) -> String {
    if s <= 6 {
        format!("${:#x}", x.imm((s - 1) as usize))
    } else {
        "?7".into()
    }
}

fn vsrc(x: &Bits, s: u64) -> Option<String> {
    match s {
        0 => None,
        1..=4 => Some(format!("s{}", x.vs((s - 1) as usize))),
        5..=10 => Some(x.imm((s - 5) as usize).to_string()),
        _ => Some(format!("?{s}")),
    }
}

fn addr_str(x: &Bits, b: u64, off: u64) -> String {
    let mut a = format!("s{}", x.vs(b as usize));
    if off != 0 {
        a.push_str(&format!(" + {}", voff(x, off)));
    }
    a
}

fn dec_vld_fields(x: &Bits, base: usize, name: &'static str, pr: u64) -> Option<Dec> {
    let (stride, off, b, m, d, op) = (
        x.f(base, 4),
        x.f(base + 4, 3),
        x.f(base + 7, 2),
        x.f(base + 9, 4),
        x.f(base + 13, 6),
        x.f(base + 19, 2),
    );
    if pr == 0 {
        return None;
    }
    let addr = addr_str(x, b, off);
    if op >= 2 {
        let pat = if (5..=10).contains(&stride) {
            let k = 2 * (stride - 5) as usize;
            format!(
                "${:#x}",
                ((x.imm(k + 1) & 0xFFFF) << 16) | (x.imm(k) & 0xFFFF)
            )
        } else if stride == 12 {
            "$0x32107654".into()
        } else {
            format!("?{stride}")
        };
        return Some(
            Dec::new(name, "vld.sshfl", pred_str(x, pr))
                .dst(format!("v{d}"))
                .srcs(vec![format!(
                    "[vmem:[{addr}] sm:{} pattern:{pat}]",
                    mask_str(x, m)
                )]),
        );
    }
    let extra = stride_str(x, stride);
    Some(
        Dec::new(name, "vld", pred_str(x, pr))
            .dst(format!("v{d}"))
            .srcs(vec![format!(
                "[vmem:[{addr}]{extra} sm:{}]",
                mask_str(x, m)
            )]),
    )
}

fn stride_str(x: &Bits, st: u64) -> String {
    match st {
        5..=10 => format!(" ss:${}", x.imm((st - 5) as usize) & 0xFFFF),
        1..=4 => format!(" ss:s{}", x.vs((st - 1) as usize)),
        _ => stride_const(st)
            .map(|s| format!(" ss:${s}"))
            .unwrap_or_default(),
    }
}

fn dec_vst(x: &Bits) -> Option<Dec> {
    let (off, b, m, src, op, pr) = (
        x.f(147, 3),
        x.f(150, 2),
        x.f(152, 4),
        x.f(156, 6),
        x.f(162, 7),
        x.f(169, 2),
    );
    if pr == 0 {
        return None;
    }
    let p = pred_str(x, pr);
    let mem = format!(
        "[vmem:[{}]{} sm:{}]",
        addr_str(x, b, off),
        stride_str(x, x.f(143, 4)),
        mask_str(x, m)
    );
    if (32..=63).contains(&op) {
        return Some(Dec::new("vst", "vst.msk", p).srcs(vec![
            format!("{mem} vm=vm{}", op & 15),
            format!("vst_source=v{src}"),
        ]));
    }
    let mn = match op {
        2 | 3 => "vst".to_string(),
        12 => "vsetiar.raw.iar0".into(),
        13 => "vsetiar.raw.iar1".into(),
        14 => "vpush".into(),
        15 => "vsetrngseed".into(),
        _ => format!("vstop{op}"),
    };
    if mn != "vst" {
        return Some(Dec::new("vst", mn, p).srcs(vec![format!("v{src}")]));
    }
    Some(Dec::new("vst", "vst", p).srcs(vec![format!("{mem} vst_source=v{src}")]))
}

// ---- vector misc -----------------------------------------------------------------------

/// Vreg read ports (ext slots: port 0..7; misc-slot store: 1..8).
pub const VPORTS: [usize; 9] = [156, 276, 287, 243, 254, 210, 221, 177, 188];

fn dec_misc(x: &Bits) -> Option<Dec> {
    let (opa, sa, sb, opb, pr) = (x.f(76, 5), x.f(81, 4), x.f(85, 6), x.f(91, 4), x.f(95, 2));
    if pr == 0 {
        return None;
    }
    let p = pred_str(x, pr);
    if opb & 0xC == 0 {
        return dec_vld_fields(x, 72, "misc.vld", pr);
    }
    if (8..=11).contains(&opb) {
        let (off, b, m) = (x.f(76, 3), x.f(79, 2), x.f(81, 4));
        let src = x.f(VPORTS[1 + x.f(85, 3) as usize], 6);
        let mem = format!(
            "[vmem:[{}]{} sm:{}]",
            addr_str(x, b, off),
            stride_str(x, x.f(72, 4)),
            mask_str(x, m)
        );
        let (mn, srcs) = if opb >= 10 {
            (
                "vst.msk",
                vec![
                    format!("{mem} vm=vm{}", x.f(162, 4)),
                    format!("vst_source=v{src}"),
                ],
            )
        } else {
            ("vst", vec![format!("{mem} vst_source=v{src}")])
        };
        return Some(Dec::new("misc.vst", mn, p).srcs(srcs));
    }
    if opb == 6 && opa != 7 {
        let mn = match opa {
            0 => "vmmov".to_string(),
            2 => "vmand".into(),
            3 => "vmor".into(),
            4 => "vmxor".into(),
            5 => "vmpackc.low".into(),
            _ => format!("vmop{opa}"),
        };
        let mut srcs = vec![format!("vm{}", x.f(81, 4))];
        if opa != 0 {
            srcs.push(format!("vm{}", x.f(72, 4)));
        }
        return Some(
            Dec::new("misc", mn, p)
                .dst(format!("vm{}", x.f(85, 4)))
                .srcs(srcs),
        );
    }
    let mn = match (opa, opb) {
        (1, 4) => "vsyncadd".to_string(),
        (7, 4) => "vsyncpa".into(),
        (7, 6) => "vsyncmov".into(),
        (19, 5) => "vwait.ge".into(),
        (17, 5) => "vwait.ne".into(),
        (6, 4) => "vtrace".into(),
        (5, 4) => "vsettm".into(),
        (3, 4) => "vdelay".into(),
        (7, 7) => "vrng.u32".into(),
        (1, 7) => "vsyncadd.remote".into(),
        _ => format!("vmisc.{opa}.{opb}"),
    };
    let flag = |s: u64| -> String {
        let v = vsrc(x, s).unwrap_or_else(|| "None".into());
        if s >= 5 { format!("[sflag:{v}]") } else { v }
    };
    let val = |s: u64| -> String {
        match vsrc(x, s) {
            None => "0".into(),
            Some(v) if s >= 5 => ((sext20(v.parse().unwrap_or(0)) as u64) & MASK32).to_string(),
            Some(v) => v,
        }
    };
    let o = Dec::new("misc", mn.clone(), p);
    Some(match mn.as_str() {
        "vsyncadd" | "vsyncpa" | "vwait.ge" | "vwait.ne" => o.srcs(vec![flag(sb), val(sa)]),
        "vsyncadd.remote" => o.srcs(vec![vsrc(x, sb).unwrap_or_else(|| "0".into()), val(sa)]),
        "vsyncmov" => o.srcs(vec![flag(sb)]),
        "vtrace" => {
            if sa == 1 && sb == 1 {
                o.srcs(vec![vsrc(x, 1).unwrap_or_default()])
            } else {
                let num =
                    |s: u64| vsrc(x, s).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) & 0xFFFF;
                o.srcs(vec![((num(sb) << 16) | num(sa)).to_string()])
            }
        }
        "vsettm" => o.srcs(vec![vsrc(x, sa).unwrap_or_else(|| "0".into())]),
        "vrng.u32" => o.dst(format!("v{sb}")),
        _ => o,
    })
}

// ---- vector extended (MXU / XLU) and result --------------------------------------------

fn mm_dt(sub: u64) -> String {
    match sub & 7 {
        0 => "f32".into(),
        1 => "bf16".into(),
        7 => "f8e4m3fn".into(),
        _ => format!("dt{sub}"),
    }
}

fn push_sub(sub: u64) -> String {
    match sub {
        8 => "bf16.msra".into(),
        4 => "f8e4m3.msra".into(),
        9 => "bf16.xpose.msra".into(),
        _ => sub.to_string(),
    }
}

/// Mnemonic of family `fam` with sub-field `sub` (word = fam[22:15] u1[14] sub[13:10]
/// port[9:7] low[6:0]); None for an unknown family.
fn ext_fam(fam: u64, sub: u64) -> Option<String> {
    Some(match fam {
        0x01 => format!("vmatmul.mubr.{}", mm_dt(sub)),
        0x02 => format!("vmatmul.mubr.{}.vlgmr.msra", mm_dt(sub)),
        0x0B => "vmatmul.mubr.msk.bf16".into(),
        0x0C => "vmatmul.mubr.msk.bf16.vlgmr.msra".into(),
        0x37 => match sub {
            4 => "vmatprep.mubr".into(),
            5 => "vmatprep.subr.bf16".into(),
            _ => format!("vmatprep.{sub}"),
        },
        0x38 => format!("vmatpush1.{}", push_sub(sub)),
        0x3A => format!("vmatpush3.{}", push_sub(sub)),
        0xF4 => "vmatprep.mubr.msk.f32".into(),
        0xF5 => "vmatprep.mubr.msk.bf16".into(),
        0xF8 => "vmatprep.subr.msk.bf16".into(),
        0x89 => "vmatpush1.bf16.xpose.msk.msra".into(),
        0xA9 => "vmatpush3.bf16.xpose.msk.msra".into(),
        0x69 | 0x6A => "vset.pattern.permute".into(),
        0x6D => "vxpose.b32.start".into(),
        0x6F => "vxpose.c.b16.start".into(),
        0x73 => "vxpose.b32.cont".into(),
        0x74 => "vxpose.b32.end".into(),
        0x75 => "vperm".into(),
        0x76 => "vrot.lane.b32".into(),
        0x79 => "vbcast.lane.b32".into(),
        0x7D => match sub {
            0 => "vadd.xlane.f32".into(),
            1 => "vmax.xlane.f32".into(),
            2 => "vmin.xlane.f32".into(),
            5 => "vadd.xlane.bf16".into(),
            6 => "vmax.xlane.bf16".into(),
            7 => "vmin.xlane.bf16".into(),
            _ => format!("vxlane.{sub}"),
        },
        _ => return None,
    })
}

fn ext_name(k: u64) -> Option<String> {
    let m = ext_fam(k >> 15, (k >> 10) & 15)?;
    Some(if k & 0x4000 != 0 {
        format!("{m}.u1")
    } else {
        m
    })
}

/// Every mnemonic an ext slot can print (without `.u1`).
fn is_ext_name(mn: &str) -> bool {
    const FAMS: [u64; 25] = [
        0x01, 0x02, 0x0B, 0x0C, 0x37, 0x38, 0x3A, 0xF4, 0xF5, 0xF8, 0x89, 0xA9, 0x69, 0x6A, 0x6D,
        0x6F, 0x73, 0x74, 0x75, 0x76, 0x79, 0x7D, 0x7D, 0x7D, 0x7D,
    ];
    FAMS.iter()
        .any(|&f| (0..16).any(|s| ext_fam(f, s).as_deref() == Some(mn)))
}

fn dec_ext(x: &Bits, base: usize, name: &'static str) -> Option<Dec> {
    let pr = x.f(base + 23, 2);
    if pr == 0 {
        return None;
    }
    let k = x.f(base, 23);
    let mn = ext_name(k).unwrap_or(format!("vext.{k:#x}"));
    let mut o = Dec::new(name, mn, pred_str(x, pr)).srcs(vec![format!(
        "v{}",
        x.f(VPORTS[((k >> 7) & 7) as usize], 6)
    )]);
    o.word = k;
    Some(o)
}

fn dec_res(x: &Bits, base: usize, name: &'static str) -> Option<Dec> {
    let (d, op, pr) = (x.f(base, 6), x.f(base + 6, 3), x.f(base + 9, 2));
    if pr == 0 {
        return None;
    }
    let mn = match op {
        0 => "vpop.eup".to_string(),
        1 => "vpop.xlu".into(),
        2 => "vpop.xlu.u1".into(),
        4 => "vpop.f32.mrb".into(),
        5 => "vpop.f32.mrb.u1".into(),
        _ => format!("vpop.{op}"),
    };
    Some(Dec::new(name, mn, pred_str(x, pr)).dst(format!("v{d}")))
}

/// Decode one bundle.
pub fn decode(raw: &[u8; 64]) -> Vec<Dec> {
    let x = Bits::new(raw);
    let mut ops = Vec::new();
    if x.f(489, 2) != 0 && matches!(x.f(483, 6), 6 | 7) {
        ops.push(dec_dma(&x));
    } else if x.f(489, 2) != 0 && x.f(483, 6) == 8 {
        ops.push(dec_dma_general(&x));
    } else {
        ops.extend(dec_scalar(&x, 467, "s0"));
        ops.extend(dec_scalar(&x, 443, "s1"));
    }
    for (b, n) in [
        (270, "valu0"),
        (237, "valu1"),
        (204, "valu2"),
        (171, "valu3"),
    ] {
        ops.extend(dec_valu(&x, b, n));
    }
    ops.extend(dec_vst(&x));
    ops.extend(dec_vld_fields(&x, 120, "vld0", x.f(141, 2)));
    ops.extend(dec_vld_fields(&x, 97, "vld1", x.f(118, 2)));
    ops.extend(dec_misc(&x));
    ops.extend(dec_ext(&x, 47, "ext0"));
    ops.extend(dec_ext(&x, 22, "ext1"));
    ops.extend(dec_res(&x, 11, "res0"));
    ops.extend(dec_res(&x, 0, "res1"));
    ops
}

/// An empty bundle (no slot occupied).
pub fn is_empty(raw: &[u8; 64]) -> bool {
    raw[..63].iter().all(|&b| b == 0) && raw[63] & !0x02 == 0x50
}

// ---- listing comparator ----------------------------------------------------------------

/// Ops that have no machine encoding.
pub const NO_ENCODING: &[&str] = &[
    "inlined_call_operand",
    "int_to_ptr",
    "compiler-scheduling-barrier",
    "inlined_call",
    "sphi",
    "vphi",
];

pub fn no_encoding(mn: &str) -> bool {
    NO_ENCODING.iter().any(|p| mn.starts_with(p))
}

/// An op as tools/llo.py parses it (positional args, named operands kept in order).
#[derive(Clone, Debug, Default)]
pub struct PyOp {
    pub def: Option<String>,
    pub mn: String,
    pub pred: Option<String>,
    pub args: Vec<String>,
}

fn is_w(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

pub fn py_op(text: &str) -> PyOp {
    let mut op = PyOp::default();
    let mut rest = text;
    // DEF_RE: ^(%[\w.]+)\s*=\s*(.*)$
    if let Some(r) = text.strip_prefix('%') {
        let n = r.bytes().take_while(|&c| is_w(c) || c == b'.').count();
        if n > 0 {
            let after = r[n..].trim_start();
            if let Some(a) = after.strip_prefix('=') {
                op.def = Some(text[..1 + n].to_string());
                rest = a.trim_start();
            }
        }
    }
    let (mn, args) = match rest.split_once(' ') {
        Some((m, a)) => (m, a.trim()),
        None => (rest, ""),
    };
    op.mn = mn.to_string();
    let mut args = args.to_string();
    // ^\((!?)(%[\w]+|0|1)\)\s*,?\s*(.*)$
    if let Some(inner) = args.strip_prefix('(')
        && let Some(close) = inner.find(')')
    {
        let body = &inner[..close];
        let (neg, tok) = match body.strip_prefix('!') {
            Some(t) => ("!", t),
            None => ("", body),
        };
        let ok = tok == "0"
            || tok == "1"
            || (tok.len() > 1 && tok.starts_with('%') && tok[1..].bytes().all(is_w));
        if ok {
            op.pred = Some(format!("{neg}{tok}"));
            let r = inner[close + 1..].trim_start();
            let r = r.strip_prefix(',').unwrap_or(r).trim_start();
            args = r.to_string();
        }
    }
    op.args = if args.is_empty() {
        Vec::new()
    } else {
        llo::split_top(&args, ",")
    };
    op
}

/// `%[\w.]+?_((?:s|p|v|vm)\d+)\b` -> reg, then `%(\d+)` -> `%`.
fn canon_regs(a: &str) -> String {
    let b = a.as_bytes();
    let mut out = String::with_capacity(a.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let tok_len = b[i + 1..]
                .iter()
                .take_while(|&&c| is_w(c) || c == b'.')
                .count();
            let tok = &a[i + 1..i + 1 + tok_len];
            let tb = tok.as_bytes();
            let mut done = false;
            for j in 1..tb.len() {
                if tb[j] != b'_' {
                    continue;
                }
                let r = &tok[j + 1..];
                let rb = r.as_bytes();
                // alternatives s | p | v | vm, then \d+, then a word boundary
                for pre in ["s", "p", "v", "vm"] {
                    if !r.starts_with(pre) {
                        continue;
                    }
                    let nd = rb[pre.len()..]
                        .iter()
                        .take_while(|c| c.is_ascii_digit())
                        .count();
                    let end = pre.len() + nd;
                    if nd > 0 && (end == rb.len() || !is_w(rb[end])) {
                        out.push_str(&r[..end]);
                        out.push_str(&r[end..]);
                        done = true;
                        break;
                    }
                }
                if done {
                    break;
                }
            }
            if done {
                i += 1 + tok_len;
                continue;
            }
            // %(\d+) -> %
            out.push('%');
            let nd = b[i + 1..].iter().take_while(|c| c.is_ascii_digit()).count();
            i += 1 + nd;
            continue;
        }
        let ch = a[i..].chars().next().expect("char boundary");
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// `#allocation\d+(_\w+)?( [+-] \$0x[0-9a-f]+)?` -> `A`.
fn canon_alloc(a: &str) -> String {
    let key = "#allocation";
    let mut out = String::new();
    let mut rest = a;
    while let Some(i) = rest.find(key) {
        let r = &rest[i + key.len()..];
        let rb = r.as_bytes();
        let nd = rb.iter().take_while(|c| c.is_ascii_digit()).count();
        if nd == 0 {
            out.push_str(&rest[..i + key.len()]);
            rest = r;
            continue;
        }
        let mut e = nd;
        if rb.get(e) == Some(&b'_') {
            let nw = rb[e + 1..].iter().take_while(|&&c| is_w(c)).count();
            if nw > 0 {
                e += 1 + nw;
            }
        }
        let t = &r[e..];
        if (t.starts_with(" + $0x") || t.starts_with(" - $0x"))
            && t.as_bytes()
                .get(6)
                .is_some_and(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
        {
            let nh = t.as_bytes()[6..]
                .iter()
                .take_while(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
                .count();
            e += 6 + nh;
        }
        out.push_str(&rest[..i]);
        out.push('A');
        rest = &r[e..];
    }
    out.push_str(rest);
    out
}

fn canon_arg(a: &str) -> String {
    let mut s = canon_alloc(&canon_regs(a));
    // (\S)\s+\[A\]$ -> \1
    if let Some(p) = s.strip_suffix("[A]") {
        let t = p.trim_end();
        if t.len() < p.len() && !t.is_empty() {
            s = t.to_string();
        }
    }
    s.replace("sm:$", "sm:").trim().to_string()
}

fn alias(mn: &str) -> &str {
    match mn {
        "dma.done.wait" => "vwait.ge",
        "scalar_lea.vmem" | "scalar_lea.hbm" | "scalar_lea.smem" | "scalar_lea.sflag" => "sadd.s32",
        "scalar_parameter_address" => "sld",
        "reloc" => "smov",
        m => m,
    }
}

/// Remove every `.{u}{0,1}` for u in mxu/xlu; returns (stripped, saw unit 1).
fn strip_units(mn: &str) -> (String, bool) {
    let mut out = String::new();
    let mut u1 = false;
    let mut i = 0;
    let b = mn.as_bytes();
    while i < b.len() {
        if b[i] == b'.' && (mn[i + 1..].starts_with("mxu") || mn[i + 1..].starts_with("xlu")) {
            let d = b.get(i + 4).copied();
            if matches!(d, Some(b'0') | Some(b'1')) {
                let bound = b.get(i + 5).is_none_or(|&c| !is_w(c));
                if d == Some(b'1') && bound {
                    u1 = true;
                }
                i += 5;
                continue;
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    (out, u1)
}

fn unit1(mn: &str) -> bool {
    strip_units(mn).1
}

/// Canonical listing op (tpudis.canon_listing) with the listing text of each argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Canon {
    pub mn: String,
    pub dst: Option<String>,
    pub args: Vec<String>,
    pub pred: Option<String>,
    /// Listing text behind each canonical argument.
    pub raw: Vec<String>,
}

pub fn canon_listing(op: &PyOp) -> Canon {
    let mn0 = alias(&op.mn).to_string();
    let u1 = if unit1(&mn0) { ".u1" } else { "" };
    let mut mn = if mn0.starts_with("vpop.") && mn0 != "vpop.sfrf" {
        let base = if mn0.contains(".mrb") {
            "vpop.f32.mrb".to_string()
        } else if mn0.contains("xlu") {
            "vpop.xlu".into()
        } else if mn0.contains("eup") {
            "vpop.eup".into()
        } else {
            mn0.clone()
        };
        base + u1
    } else {
        // .gmra.mrb[N]
        let mut m = mn0.clone();
        if let Some(i) = m.find(".gmra.mrb[")
            && let Some(j) = m[i..].find(']')
            && m[i + 10..i + j].bytes().all(|c| c.is_ascii_digit())
            && j > 10
        {
            m.replace_range(i..i + j + 1, "");
        }
        strip_units(&m).0 + u1
    };
    for t in ["vmatprep.mubr.bf16", "vmatprep.mubr.f32"] {
        if let Some(r) = mn.strip_prefix(t) {
            mn = format!("vmatprep.mubr{r}");
            break;
        }
    }
    let mut args = op.args.clone();
    let mut pred = op.pred.clone();
    // predicated move form `(p0, s12)`
    if let Some(a0) = args.first()
        && let Some(inner) = a0.strip_prefix('(').and_then(|r| r.strip_suffix(')'))
        && let Some((p, v)) = inner.split_once(", ")
        && v.len() > 1
        && v.starts_with('%')
        && v[1..].bytes().all(is_w)
    {
        let q = p.strip_prefix('!').unwrap_or(p);
        let q = q.strip_prefix('%').unwrap_or(q);
        let ok = !q.is_empty() && q.bytes().all(is_w) && {
            let nd = q.bytes().rev().take_while(|c| c.is_ascii_digit()).count();
            nd > 0 && q.len() > nd && q.as_bytes()[q.len() - nd - 1] == b'p' && q.len() > nd + 1
        };
        if ok {
            pred = Some(p.to_string());
            args.remove(0);
        }
    }
    if (mn == "sst" || mn == "vst")
        && args.len() == 2
        && (args[0].ends_with(')') || args[1].starts_with("vst_source"))
    {
        args = vec![format!("{} {}", args[0], args[1])];
    }
    let mut out = Vec::new();
    let mut raw = Vec::new();
    for a0 in &args {
        let mut a = a0.clone();
        // \s*\((!?%[\w]+)\)
        if let Some(i) = find_paren_pred(&a) {
            let (s, e, p) = i;
            pred = Some(p);
            a = format!("{}{}", &a[..s], &a[e..]);
        }
        if let Some((lo, hi)) = a.split_once('|') {
            let num = |s: &str| {
                !s.is_empty()
                    && s.bytes()
                        .all(|c| c.is_ascii_digit() || b".e+-".contains(&c))
            };
            if num(lo) && num(hi) {
                a = lo.to_string();
            }
        }
        // ` sps:\$\d+`
        while let Some(i) = a.find(" sps:$") {
            let nd = a.as_bytes()[i + 6..]
                .iter()
                .take_while(|c| c.is_ascii_digit())
                .count();
            if nd == 0 {
                break;
            }
            a.replace_range(i..i + 6 + nd, "");
        }
        raw.push(a0.clone());
        out.push(canon_arg(&a));
    }
    let pred = pred.map(|p| {
        // %[\w]+?_(p\d+) -> \1, then lstrip('%')
        let mut s = String::new();
        let b = p.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%' {
                let n = b[i + 1..].iter().take_while(|&&c| is_w(c)).count();
                let tok = &p[i + 1..i + 1 + n];
                let mut hit = None;
                for j in 1..tok.len() {
                    if tok.as_bytes()[j] == b'_' && tok[j + 1..].starts_with('p') {
                        let nd = tok.as_bytes()[j + 2..]
                            .iter()
                            .take_while(|c| c.is_ascii_digit())
                            .count();
                        if nd > 0 {
                            hit = Some((j, nd));
                            break;
                        }
                    }
                }
                if let Some((j, nd)) = hit {
                    s.push_str(&tok[j + 1..j + 2 + nd]);
                    s.push_str(&tok[j + 2 + nd..]);
                    i += 1 + n;
                    continue;
                }
            }
            s.push(b[i] as char);
            i += 1;
        }
        s.trim_start_matches('%').to_string()
    });
    let dst = op
        .def
        .as_deref()
        .map(canon_arg)
        .filter(|d| !d.starts_with('%'));
    Canon {
        mn,
        dst,
        args: out,
        pred,
        raw,
    }
}

/// First `\s*\((!?%\w+)\)` in `a`: (start, end, predicate).
fn find_paren_pred(a: &str) -> Option<(usize, usize, String)> {
    let b = a.as_bytes();
    for (i, _) in a.match_indices('(') {
        let mut j = i + 1;
        if b.get(j) == Some(&b'!') {
            j += 1;
        }
        if b.get(j) != Some(&b'%') {
            continue;
        }
        let n = b[j + 1..].iter().take_while(|&&c| is_w(c)).count();
        if n == 0 || b.get(j + 1 + n) != Some(&b')') {
            continue;
        }
        let mut s = i;
        while s > 0 && b[s - 1].is_ascii_whitespace() {
            s -= 1;
        }
        return Some((s, j + 2 + n, a[i + 1..j + 1 + n].to_string()));
    }
    None
}

/// Canonical decoded op: `[sflag:N]` -> `[A]`.
pub fn canon_decoded(o: &Dec) -> Canon {
    let c = |a: &String| {
        let mut s = a.clone();
        while let Some(i) = s.find("[sflag:") {
            let r = &s[i + 7..];
            let nd = r.bytes().take_while(|c| c.is_ascii_digit()).count();
            if nd > 0 && r.as_bytes().get(nd) == Some(&b']') {
                s.replace_range(i..i + 8 + nd, "[A]");
            } else {
                break;
            }
        }
        s
    };
    Canon {
        mn: o.mn.clone(),
        dst: o.dst.clone(),
        args: o.srcs.iter().map(c).collect(),
        pred: o.pred.clone(),
        raw: o.srcs.clone(),
    }
}

/// Mnemonics renamed between libtpu 0.0.50 and the nightly libtpu-meta (same encoding).
pub fn build_equiv(mn: &str) -> &str {
    match mn {
        "vexp2.f32" => "vpow2.f32",
        "vpack.c.f32.bf16" => "vpack.c.bf16",
        "vunpack.c.l.bf16.f32" => "vunpack.c.l.bf16",
        "vlaneseq.b32" => "vlaneseq",
        "vmmov" => "vnez",
        "vunpack.c.0.s8.s32" => "vunpack.c.0.s8",
        "vunpack.c.1.s8.s32" => "vunpack.c.1.s8",
        "vunpack.c.2.s8.s32" => "vunpack.c.2.s8",
        "vunpack.c.3.s8.s32" => "vunpack.c.3.s8",
        "vpack.c.b32.b16" => "vpack.c.b16",
        "vpack.c.b16.b8" => "vpack.c.b8",
        "vunpack.c.l.s4.s8" => "vunpack.c.l.s4",
        "vunpack.c.h.s4.s8" | "vunpack.c.u.s4.s8" => "vunpack.c.h.s4",
        "vunpack.c.l.s16.s32" => "vunpack.c.l.b16",
        "vunpack.c.u.s16.s32" => "vunpack.c.h.b16",
        "vunpack.c.u.s8.bf16" => "vunpack.c.h.s8.bf16",
        "vunpack.c.u.bf16.f32" | "vunpack.c.h.bf16.f32" => "vunpack.c.h.bf16",
        "set.rngseed" => "vsetrngseed",
        "vrng" => "vrng.u32",
        m => m,
    }
}

/// Element type the listing carries but the encoding does not.
pub fn untyped(mn: &str) -> String {
    let mut m = mn.to_string();
    for k in ["subr", "mubr"] {
        for t in ["bf16", "f8e4m3fn", "f8e4m3"] {
            let p = format!("vmatprep.{k}.{t}");
            if let Some(r) = m.strip_prefix(&p)
                && (r.is_empty() || r == ".u1")
            {
                m = format!("vmatprep.{k}{r}");
            }
        }
    }
    for p in [
        "vxpose.c.b16.",
        "vxpose.c.b32.",
        "vxpose.b16.",
        "vxpose.b32.",
    ] {
        if let Some(r) = m.strip_prefix(p)
            && (r.starts_with("cont") || r.starts_with("end"))
        {
            m = format!("vxpose.{r}");
            break;
        }
    }
    m
}

fn commutative(mn: &str) -> bool {
    matches!(
        mn,
        "sadd.s32"
            | "sand.u32"
            | "sor.u32"
            | "sxor.u32"
            | "smul.u32"
            | "smin.u32"
            | "vadd.s32"
            | "vadd.f32"
            | "vmul.f32"
            | "vmul.u32"
            | "vand.u32"
            | "vor.u32"
            | "vxor.u32"
            | "vmax.f32"
            | "vmin.f32"
            | "scmp.eq.s32.totalorder"
            | "scmp.ne.s32.totalorder"
    )
}

fn is_int(s: &str) -> bool {
    let t = s.strip_prefix('-').unwrap_or(s);
    !t.is_empty() && t.bytes().all(|c| c.is_ascii_digit())
}

fn int32(s: &str) -> Option<u64> {
    s.parse::<i128>().ok().map(|v| (v & MASK32 as i128) as u64)
}

fn py_float(s: &str) -> Option<f64> {
    let t = s.trim();
    let l = t.to_ascii_lowercase();
    let body = l.trim_start_matches(['+', '-']);
    if matches!(body, "inf" | "infinity" | "nan") {
        return t.parse::<f64>().ok().or(Some(if l.starts_with('-') {
            f64::NEG_INFINITY
        } else if body == "nan" {
            f64::NAN
        } else {
            f64::INFINITY
        }));
    }
    if !t.bytes().any(|c| c.is_ascii_digit()) {
        return None;
    }
    t.parse::<f64>().ok()
}

/// Immediates: listing float literal vs encoded float32 (or bf16 half) bit pattern.
pub fn num_eq(a: &str, b: &str) -> bool {
    if is_int(a) && is_int(b) {
        return int32(a) == int32(b);
    }
    let Some(fa) = py_float(a) else { return false };
    if !is_int(b) {
        return py_float(b).is_some_and(|fb| fa == fb);
    }
    let f = fa as f32;
    if fa.is_finite() && f.is_infinite() {
        return false; // struct.pack overflow
    }
    let bits = f.to_bits() as u64;
    let Some(v) = int32(b) else { return false };
    v == bits || v == bits >> 16 || v == (bits >> 16) * 0x10001
}

/// `a` with symbol `A` placeholders matches `b` (tpudis: any address in the same form).
fn sym_match(a: &str, b: &str) -> bool {
    let parts: Vec<&str> = a.split('A').collect();
    fn alts(s: &str) -> Vec<usize> {
        let b = s.as_bytes();
        let mut out = Vec::new();
        let digits = |i: usize| b[i..].iter().take_while(|c| c.is_ascii_digit()).count();
        let hex = |i: usize| {
            b[i..]
                .iter()
                .take_while(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
                .count()
        };
        if b.first() == Some(&b's') {
            let n = digits(1);
            for k in 1..=n {
                out.push(1 + k);
                let r = &s[1 + k..];
                if r.starts_with(" + $0x") {
                    let h = hex(1 + k + 6);
                    for m in 1..=h {
                        out.push(1 + k + 6 + m);
                    }
                }
            }
        }
        if s.starts_with("$0x") {
            for m in 1..=hex(3) {
                out.push(3 + m);
            }
        }
        for k in 1..=digits(0) {
            out.push(k);
        }
        if s.starts_with('A') {
            out.push(1);
        }
        out
    }
    fn go(parts: &[&str], b: &str) -> bool {
        let Some((first, rest)) = parts.split_first() else {
            return b.is_empty();
        };
        let Some(r) = b.strip_prefix(first) else {
            return false;
        };
        if rest.is_empty() {
            return r.is_empty();
        }
        alts(r).into_iter().any(|n| go(rest, &r[n..]))
    }
    go(&parts, b)
}

fn dma_match(lmn: &str, la: &[String], dmn: &str, da: &[String]) -> bool {
    if lmn != dmn {
        return false;
    }
    if lmn == "dma.general" {
        let dk: Vec<(&str, &str)> = da.iter().filter_map(|a| a.split_once('=')).collect();
        for a in la {
            let (k, v) = a.split_once('=').unwrap_or((a.as_str(), ""));
            if k == "stridedescaddr" {
                continue;
            }
            if let Some((_, dv)) = dk.iter().rev().find(|(dk, _)| *dk == k)
                && *dv != v
            {
                return false;
            }
        }
        return true;
    }
    let lt = la.join(" ");
    let dt = da.join(" ");
    if let Some(i) = lt.find("[thread:$") {
        let n: String = lt[i + 9..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        if !n.is_empty()
            && lt[i + 9 + n.len()..].starts_with(']')
            && !dt.contains(&format!("[thread:${n}]"))
        {
            return false;
        }
    }
    let Some((src, dst)) = lmn[4..].split_once("_to_") else {
        return false;
    };
    let mut want: Vec<(String, String)> = Vec::new();
    for key in [src, dst, "size_in_granules", "src_address", "host_iova_low"] {
        // \b{key}=(s\d+|\d+)
        let mut s = lt.as_str();
        let mut off = 0;
        while let Some(i) = s.find(&format!("{key}=")) {
            let abs = off + i;
            let bound = abs == 0 || !is_w(lt.as_bytes()[abs - 1]);
            let v = &lt[abs + key.len() + 1..];
            let val = if let Some(r) = v.strip_prefix('s') {
                let n = r.bytes().take_while(u8::is_ascii_digit).count();
                (n > 0).then(|| format!("s{}", &r[..n]))
            } else {
                let n = v.bytes().take_while(u8::is_ascii_digit).count();
                (n > 0).then(|| v[..n].to_string())
            };
            if bound && let Some(val) = val {
                want.push((key.to_string(), val));
                break;
            }
            off = abs + 1;
            s = &lt[off..];
        }
    }
    if want.is_empty() {
        let pos: Vec<&str> = la
            .iter()
            .map(|a| a.rsplit("] ").next().unwrap_or(a))
            .collect();
        if pos.len() >= 3 {
            want = vec![
                (src.to_string(), pos[0].to_string()),
                ("size_in_granules".into(), pos[1].to_string()),
                (dst.to_string(), pos[2].to_string()),
            ];
        }
    }
    for (k, v) in &mut want {
        if k == "src_address" {
            *k = src.to_string();
        }
        if k == "host_iova_low" {
            *k = dst.to_string();
        }
        let _ = v;
    }
    // dict semantics: later keys overwrite earlier ones
    let mut uniq: Vec<(String, String)> = Vec::new();
    for (k, v) in want {
        if let Some(e) = uniq.iter_mut().find(|e| e.0 == k) {
            e.1 = v;
        } else {
            uniq.push((k, v));
        }
    }
    uniq.iter().all(|(k, v)| dt.contains(&format!("{k}={v}")))
}

/// Listing op `l` vs decoded op `d` (tpudis.op_match).
pub fn op_match(l: &Canon, d: &Canon) -> bool {
    let dmn = d.mn.as_str();
    let mut lmn = l.mn.clone();
    if lmn != dmn && build_equiv(&lmn) == build_equiv(dmn) {
        lmn = dmn.to_string();
    }
    if lmn != dmn && untyped(&lmn) == untyped(dmn) {
        lmn = dmn.to_string();
    }
    let lp = l.pred.as_deref().filter(|s| !s.is_empty());
    let dp = d.pred.as_deref().filter(|s| !s.is_empty());
    if lp != dp && !(lmn == "shalt.err" && dmn == "shalt.err") {
        let ok = matches!((lp, dp), (Some(a), Some(b)) if a.replace('!', "") == b.replace('!', ""))
            && lmn == "shalt.err";
        if !ok {
            return false;
        }
    }
    if lmn.starts_with("dma.") && dmn.starts_with("dma.") {
        return dma_match(&lmn, &l.args, dmn, &d.args);
    }
    let ld = l.dst.as_deref().filter(|s| !s.is_empty());
    let dd = d.dst.as_deref().filter(|s| !s.is_empty());
    if (lmn == dmn && is_ext_name(dmn.split(".u1").next().unwrap_or(dmn)))
        || (lmn == dmn && dmn.starts_with("vpop.") && dmn != "vpop.sfrf")
    {
        return ld == dd || ld.is_none();
    }
    if lmn != dmn && !(lmn == "shalt.err" && dmn == "shalt") {
        return false;
    }
    if ld != dd && ld.is_some() {
        return false;
    }
    let (la, da) = (&l.args, &d.args);
    if commutative(&lmn) && la.len() == 2 && da.len() == 2 && la[0] == da[1] && la[1] == da[0] {
        return true;
    }
    if la.len() != da.len() {
        return matches!(
            lmn.as_str(),
            "vrng.u32"
                | "vpop.sfrf"
                | "spop.drf"
                | "spop"
                | "sbr.rel"
                | "vsetiar.raw"
                | "sfence"
                | "shalt.err"
                | "vtrace"
        );
    }
    la.iter().zip(da).all(|(a, b)| arg_match(&lmn, a, b))
}

fn arg_match(lmn: &str, a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    if lmn == "sld" && !a.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) {
        return true;
    }
    if a.starts_with("target = $region") && b.starts_with("target=") {
        return true;
    }
    if a == "A" || a == "[A]" {
        let t = b.strip_prefix('[').unwrap_or(b);
        let t = t.strip_suffix(']').unwrap_or(t);
        if t == "A"
            || (!t.is_empty() && t.bytes().all(|c| c.is_ascii_digit()))
            || (t.len() > 1 && t.starts_with('s') && t[1..].bytes().all(|c| c.is_ascii_digit()))
        {
            return true;
        }
    }
    if a.contains('A') && sym_match(a, b) {
        return true;
    }
    if is_int(a) && is_int(b) && int32(a) == int32(b) {
        return true;
    }
    let (ka, va) = a.rsplit_once('=').unwrap_or(("", a));
    let (kb, vb) = b.rsplit_once('=').unwrap_or(("", b));
    ka == kb && num_eq(va, vb)
}

/// Match a bundle's listing ops against its decoded ops (kcov.score): per listing op, the
/// index of the decoded op it matched (scalar_select takes two complementary smov).
pub fn match_bundle(lops: &[Canon], dops: &[Canon]) -> Vec<Option<Vec<usize>>> {
    let mut used = vec![false; dops.len()];
    let mut out = Vec::new();
    for lo in lops {
        if lo.mn == "scalar_select" {
            let hs: Vec<usize> = (0..dops.len())
                .filter(|&i| !used[i] && dops[i].mn == "smov" && dops[i].dst == lo.dst)
                .collect();
            if hs.len() >= 2 {
                used[hs[0]] = true;
                used[hs[1]] = true;
                out.push(Some(vec![hs[0], hs[1]]));
            } else {
                out.push(None);
            }
            continue;
        }
        let hit = (0..dops.len()).find(|&i| !used[i] && op_match(lo, &dops[i]));
        if let Some(i) = hit {
            used[i] = true;
        }
        out.push(hit.map(|i| vec![i]));
    }
    out
}

/// Score of one bundle: (listing ops, matched, decoded ops left over, unmatched mnemonics).
pub fn score_bundle(raw: &[u8; 64], ops: &[RawOp]) -> (usize, usize, usize, Vec<String>) {
    let lops: Vec<Canon> = ops
        .iter()
        .filter(|o| !no_encoding(&o.mn))
        .map(|o| canon_listing(&py_op(&o.text)))
        .collect();
    let dops: Vec<Canon> = decode(raw).iter().map(canon_decoded).collect();
    let m = match_bundle(&lops, &dops);
    let used: usize = m.iter().flatten().map(Vec::len).sum();
    let ok = m.iter().filter(|x| x.is_some()).count();
    let miss = lops
        .iter()
        .zip(&m)
        .filter(|(_, x)| x.is_none())
        .map(|(l, _)| l.mn.clone())
        .collect();
    (lops.len(), ok, dops.len() - used, miss)
}

/// Opcode-multiset signature of a listing bundle (tpumap._sig_listing).
pub fn sig_listing(ops: &[RawOp]) -> Vec<String> {
    let mut v: Vec<String> = ops
        .iter()
        .filter(|o| !no_encoding(&o.mn))
        .map(|o| canon_listing(&py_op(&o.text)).mn)
        .collect();
    v.sort();
    v
}

/// Opcode-multiset signature of an image bundle (tpumap._sig_image).
pub fn sig_image(raw: &[u8; 64]) -> Vec<String> {
    let mut v: Vec<String> = decode(raw).into_iter().map(|o| o.mn).collect();
    v.sort();
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_format() {
        assert_eq!(fmt_g(0.002232143, 7), "0.002232143");
        assert_eq!(fmt_g(1e-30, 1), "1e-30");
        assert_eq!(fmt_g(0.18033688, 8), "0.18033688");
        assert_eq!(fmt_vimm("vmul.f32", "1138753536"), "448.0");
        assert_eq!(fmt_vimm("vmul.f32", "4286578688"), "-inf");
    }

    #[test]
    fn canon() {
        let c = canon_listing(&py_op(
            "%v1845_v13 = vsel vm=%vm11677_vm7, on_true_vy=%v1805_v2, on_false_vx=%v1837_v11",
        ));
        assert_eq!(c.mn, "vsel");
        assert_eq!(c.dst.as_deref(), Some("v13"));
        assert_eq!(c.args, vec!["vm=vm7", "on_true_vy=v2", "on_false_vx=v11"]);
        let c = canon_listing(&py_op(
            "%1745 = vst [vmem:[%s11569_s6 + $0x150] sm:$0xf] vst_source=%v1726_v42",
        ));
        assert_eq!(c.args, vec!["[vmem:[s6 + $0x150] sm:0xf] vst_source=v42"]);
        assert_eq!(c.dst, None);
        let c = canon_listing(&py_op(
            "%892 = vmatmul.mubr.f32.vlgmr.msra.gmra.mrb[0].mxu1 %v421_v59",
        ));
        assert_eq!(c.mn, "vmatmul.mubr.f32.vlgmr.msra.u1");
        let c = canon_listing(&py_op("sst [smem:[#allocation16]] (!%p186_p3), %s187_s7"));
        assert_eq!(c.pred.as_deref(), Some("!p3"));
        assert_eq!(c.args, vec!["[smem:[A]] s7"]);
        assert!(sym_match("[smem:[A]] s7", "[smem:[24]] s7"));
        assert!(num_eq("1.0", "1065353216"));
        assert!(num_eq("-inf", "4286578688"));
    }
}
