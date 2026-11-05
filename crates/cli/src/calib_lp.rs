/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tensor-core calibration for `tcgen05.mma.kind::f8f6f4` (`ptxv calib-mma probes/mma_f8`).
//!
//! For each format pair (and accumulator type) the probe kernel runs on a free GB300 with
//! input sets chosen to separate rounding behaviours (as `calib.rs` does for kind::f16); every
//! member of the `LpMmaModel` family is then scored bit for bit against the hardware output.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use ptxv_case::Case;
use ptxv_ir::Lp;
use ptxv_ir::LpRnd;
use ptxv_sim::lowp;
use ptxv_sim::mma_lp::Fp4Model;
use ptxv_sim::mma_lp::LpMmaModel;
use ptxv_sim::mma_lp::Operand;
use serde::Serialize;

pub(crate) struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn sign(&mut self) -> f64 {
        if self.next() & 1 == 0 { 1.0 } else { -1.0 }
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.unit()).cos()
    }
}

/// A block-scaled kind of the probes/mma_bs kernel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bs {
    /// Kernel entry (one per kind).
    pub entry: &'static str,
    /// K elements per instruction.
    pub kn: usize,
    /// K elements per scale factor.
    pub blk: usize,
    /// Scale-factor format.
    pub sf: Lp,
}

/// One calibration configuration: A / B formats, the D type, and the block-scaled kind.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cfg {
    pub a: Lp,
    pub b: Lp,
    pub f16_d: bool,
    pub bs: Option<Bs>,
}

impl Cfg {
    fn name(&self) -> String {
        let base = format!(
            "{:?}x{:?}->{}",
            self.a,
            self.b,
            if self.f16_d { "f16" } else { "f32" }
        )
        .to_lowercase();
        match self.bs {
            None => base,
            Some(b) => format!("{}:{base}:{:?}", b.entry, b.sf).to_lowercase(),
        }
    }
    /// K elements per instruction.
    fn kn(&self) -> usize {
        self.bs.map_or(32, |b| b.kn)
    }
    /// K elements per 16-byte core-matrix row: 32 for packed FP4 (kind::mxf4 / mxf4nvf4).
    fn epc(&self) -> usize {
        if self.kn() == 64 { 32 } else { 16 }
    }
}

/// kind::f8f6f4 instruction-descriptor format code.
pub(crate) fn f8_code(t: Lp) -> u32 {
    match t {
        Lp::E4m3 => 0,
        Lp::E5m2 => 1,
        Lp::E2m3 => 3,
        Lp::E3m2 => 4,
        Lp::E2m1 => 5,
        _ => panic!("{t:?} is not a kind::f8f6f4 operand format"),
    }
}

/// Encode a value (RN, saturating) into `t`; return the code.
fn enc(t: Lp, v: f64) -> u32 {
    lowp::from_f32((v as f32).to_bits(), t, LpRnd::Rn, 0, true, false)
}

/// A[rows][K] codes for `t`: one per element (K = 128 slots per row).
type Codes = Vec<u32>;

struct Set {
    name: &'static str,
    a: Codes,
    b: Codes,
    /// Formats of the first K slice (the first MMA, idesc0) if not the configuration's.
    first: Option<(Lp, Lp)>,
    /// Block-scaled kinds: scale codes per (row, instruction, block), row-major.
    sc: Option<(Codes, Codes)>,
}

/// Instruction-descriptor format code of an operand under `cfg`'s kind.
fn fmt_code(cfg: Cfg, t: Lp) -> u32 {
    if cfg.kn() == 64 { 1 } else { f8_code(t) }
}

/// Largest finite and smallest subnormal exponents of `t`.
fn range(t: Lp) -> (i32, i32) {
    let f = lowp::fmt(t);
    let maxv = lowp::value(f.max, t);
    (maxv.log2().floor() as i32, 1 - f.bias - f.mbits as i32)
}

fn sets(cfg: Cfg, m: usize, n: usize, k: usize) -> Vec<Set> {
    let mut r = Rng(0x0f8 + f8_code(cfg.a) as u64 * 7 + f8_code(cfg.b) as u64 + cfg.kn() as u64);
    let gen =
        |r: &mut Rng, t: Lp, len: usize, f: &dyn Fn(&mut Rng, usize, i32, i32) -> f64| -> Codes {
            let (emax, emin) = range(t);
            (0..len).map(|i| enc(t, f(r, i, emax, emin))).collect()
        };
    let mut out = Vec::new();
    out.push(Set {
        name: "normal",
        a: gen(&mut r, cfg.a, m * k, &|r, _, _, _| r.normal()),
        b: gen(&mut r, cfg.b, n * k, &|r, _, _, _| r.normal()),
        first: None,
        sc: None,
    });
    // Every code equally likely (NaN / Inf codes replaced by zero): subnormals, the full
    // exponent range, both signs.
    let codes = |r: &mut Rng, t: Lp, len: usize| -> Codes {
        let w = lowp::width(t);
        (0..len)
            .map(|_| {
                let c = (r.next() as u32) & ((1 << w) - 1);
                if lowp::value(c, t).is_finite() { c } else { 0 }
            })
            .collect()
    };
    out.push(Set {
        name: "codes",
        a: codes(&mut r, cfg.a, m * k),
        b: codes(&mut r, cfg.b, n * k),
        first: None,
        sc: None,
    });
    out.push(Set {
        name: "logu",
        a: gen(&mut r, cfg.a, m * k, &|r, _, emax, emin| {
            r.sign() * 2f64.powf(emin as f64 + r.unit() * (emax - emin + 1) as f64)
        }),
        b: gen(&mut r, cfg.b, n * k, &|r, _, emax, emin| {
            r.sign() * 2f64.powf(emin as f64 + r.unit() * (emax - emin + 1) as f64)
        }),
        first: None,
        sc: None,
    });
    // Cancellation: big terms of both signs with small terms in between.
    out.push(Set {
        name: "cancel",
        a: gen(&mut r, cfg.a, m * k, &|r, i, emax, emin| {
            if i % 2 == 0 {
                2f64.powi(emax - (r.next() % 3) as i32)
            } else {
                r.sign() * 2f64.powf(emin as f64 + r.unit() * ((emax - emin) as f64 / 2.0))
            }
        }),
        b: gen(&mut r, cfg.b, n * k, &|r, _, emax, _| {
            r.sign() * 2f64.powi(emax / 2 - (r.next() % 2) as i32)
        }),
        first: None,
        sc: None,
    });
    // One dominant product per 32-wide step plus small ones 2^-e below it (e by row).
    out.push(Set {
        name: "probe_align",
        a: gen(&mut r, cfg.a, m * k, &|_, i, emax, emin| {
            let row = (i / k) as i32;
            if i % 32 == 0 {
                2f64.powi(emax)
            } else {
                2f64.powi((emax - row % (emax - emin + 1)).max(emin))
            }
        }),
        b: gen(&mut r, cfg.b, n * k, &|r, i, emax, _| {
            if (i / k).is_multiple_of(2) {
                1.0
            } else if i % 32 == 0 {
                2f64.powi(emax / 2)
            } else {
                r.sign() * 1.5
            }
        }),
        first: None,
        sc: None,
    });
    // A large accumulator from the first step, then small products.
    out.push(Set {
        name: "probe_acc",
        a: gen(&mut r, cfg.a, m * k, &|_, i, emax, emin| {
            let row = (i / k) as i32;
            if i % k < 32 {
                2f64.powi(emax - row % 3)
            } else {
                2f64.powi((emax - 2 - row % (emax - emin)).max(emin))
            }
        }),
        b: gen(&mut r, cfg.b, n * k, &|r, _, emax, _| {
            r.sign() * 2f64.powi(emax / 2)
        }),
        first: None,
        sc: None,
    });
    // Subnormal alignment: each 32-wide step leads with min-subnormal(A) x max(B), whose
    // field exponent sum exceeds its true exponent sum; the other products sit 20-27 binades
    // below the field sum, where the two alignments truncate differently.
    let (fa, fb) = (lowp::fmt(cfg.a), lowp::fmt(cfg.b));
    let (amax, _) = range(cfg.a);
    let (bmax, _) = range(cfg.b);
    let lead = (1 - fa.bias) + bmax; // field exponent sum of the leading product
    let a_lo: Vec<u32> = (0..m * k)
        .map(|i| {
            if i % 32 == 0 {
                1 // smallest subnormal
            } else {
                enc(cfg.a, r.sign() * 2f64.powi(1 - fa.bias) * (1.0 + r.unit()))
            }
        })
        .collect();
    let b_lo: Vec<u32> = (0..n * k)
        .map(|i| {
            if i % 32 == 0 {
                fb.max
            } else {
                let d = 20 + (i / k) as i32 % 8;
                let x = lead - d - (1 - fa.bias);
                enc(cfg.b, 2f64.powi(x.min(bmax)) * (1.0 + r.unit()))
            }
        })
        .collect();
    let _ = amax;
    out.push(Set {
        name: "probe_sub",
        a: a_lo,
        b: b_lo,
        first: None,
        sc: None,
    });
    // f16 midpoints: 64 + 2^-5 (an f16 tie) +- 2^-18 (below f32 precision there), scaled by
    // the row; a wide first step (e4m3) for narrow formats.
    let mid = |r: &mut Rng, t: Lp, i: usize, row_side: bool| -> u32 {
        let (j, row) = (i % k, (i / k) as i32);
        let sc = 2f64.powi(row % 3);
        let v = match (j, row_side) {
            (0, true) => 8.0 * sc,
            (0, false) => 8.0,
            (1, true) => 2f64.powi(-3) * sc,
            (1, false) => 2f64.powi(-2),
            (2, true) => 2f64.powi(-9) * sc,
            (2, false) => r.sign() * 2f64.powi(-9),
            _ => 0.0,
        };
        enc(t, v)
    };
    out.push(Set {
        name: "probe_mid",
        a: (0..m * k).map(|i| mid(&mut r, cfg.a, i, true)).collect(),
        b: (0..n * k).map(|i| mid(&mut r, cfg.b, i, false)).collect(),
        first: None,
        sc: None,
    });
    // Large accumulators from a wide first step (idesc0: e5m2 for f32 D, e4m3 for f16 D),
    // then this configuration's products: exposes accumulator alignment, per-group rounding
    // and the rounding mode for formats whose own sums are always exact (FP4 / FP6).
    // acc_big: one huge product plus mid-range ones (acc ~2^20..2^28; f16 D: ~2^9..2^14);
    // acc_span: a large and a tiny product (acc bits spanning ~22 binades).
    let ft = if cfg.f16_d { Lp::E4m3 } else { Lp::E5m2 };
    let (b0, b1) = if cfg.f16_d { (4.5, 7.0) } else { (10.0, 13.5) };
    let own = |r: &mut Rng, t: Lp| -> u32 {
        let (emax, emin) = range(t);
        enc(
            t,
            r.sign() * 2f64.powf(emin as f64 + r.unit() * (emax - emin + 1) as f64),
        )
    };
    let big = |r: &mut Rng, t: Lp, i: usize| -> u32 {
        match i % k {
            0 => enc(ft, r.sign() * 2f64.powf(b0 + r.unit() * (b1 - b0))),
            1..32 => enc(ft, r.sign() * 2f64.powf(-6.0 + r.unit() * 12.0)),
            _ => own(r, t),
        }
    };
    out.push(Set {
        name: "acc_big",
        a: (0..m * k).map(|i| big(&mut r, cfg.a, i)).collect(),
        b: (0..n * k).map(|i| big(&mut r, cfg.b, i)).collect(),
        first: Some((ft, ft)),
        sc: None,
    });
    let (s0, s1) = if cfg.f16_d { (2.0, 4.0) } else { (4.0, 6.0) };
    let span = |r: &mut Rng, t: Lp, i: usize| -> u32 {
        match i % k {
            0 => enc(Lp::E4m3, r.sign() * 2f64.powf(s0 + r.unit() * (s1 - s0))),
            1 => enc(Lp::E4m3, r.sign() * 2f64.powf(-6.0 - r.unit() * 3.0)),
            2..32 => 0,
            _ => own(r, t),
        }
    };
    out.push(Set {
        name: "acc_span",
        a: (0..m * k).map(|i| span(&mut r, cfg.a, i)).collect(),
        b: (0..n * k).map(|i| span(&mut r, cfg.b, i)).collect(),
        first: Some((Lp::E4m3, Lp::E4m3)),
        sc: None,
    });
    out
}

/// Input sets for a block-scaled kind: operand sets as for kind::f8f6f4 (without the wide
/// first steps) with scale factors chosen to separate the candidates: narrow (2^+-8 / random
/// normal ue4m3), wide (2^+-30 / every ue4m3 code), blocks of very different scale within an
/// instruction, a large scale on the first instruction only (big accumulator), and large
/// scales on subnormal operands.
fn bs_sets(cfg: Cfg, m: usize, n: usize, k: usize) -> Vec<Set> {
    let bs = cfg.bs.unwrap();
    let nb = bs.kn / bs.blk;
    let mut r = Rng(0xb5 + f8_code(cfg.a) as u64 * 11 + bs.kn as u64 + (bs.sf == Lp::Ue4m3) as u64);
    let base: Vec<Set> = sets(Cfg { bs: None, ..cfg }, m, n, k)
        .into_iter()
        .filter(|s| s.first.is_none())
        .collect();
    // Scale code for a log2 value (ue8m0: exact; ue4m3: nearest, saturating).
    let code = |x: f64| -> u32 {
        if bs.sf == Lp::Ue8m0 {
            (x.round() as i32 + 127).clamp(0, 254) as u32
        } else {
            enc(Lp::Ue4m3, 2f64.powf(x)) & 0x7f
        }
    };
    let rand_code = |r: &mut Rng| -> u32 {
        if bs.sf == Lp::Ue8m0 {
            (r.next() % 255) as u32
        } else {
            let c = (r.next() % 0x7f) as u32;
            c.max(1)
        }
    };
    let (lo, hi) = if bs.sf == Lp::Ue8m0 {
        (-30.0, 30.0)
    } else {
        (-9.0, 8.0)
    };
    let mut out = Vec::new();
    for (i, s) in base.into_iter().enumerate() {
        let kind = i % 5;
        let gen = |r: &mut Rng, rows: usize| -> Codes {
            (0..rows * 4 * nb)
                .map(|j| {
                    let (kc, b) = ((j / nb) % 4, j % nb);
                    match kind {
                        0 => code(r.unit() * 16.0 - 8.0),
                        1 => {
                            if bs.sf == Lp::Ue8m0 {
                                code(lo + r.unit() * (hi - lo))
                            } else {
                                rand_code(r)
                            }
                        }
                        2 => code(if b == 0 { hi * 0.6 } else { lo * 0.3 } + r.unit() * 2.0),
                        3 => code(if kc == 0 { hi * 0.5 } else { lo * 0.4 } + r.unit() * 3.0),
                        _ => code(r.unit() * (hi - lo) / 2.0),
                    }
                })
                .collect()
        };
        let sa = gen(&mut r, m);
        let sb = gen(&mut r, n);
        if s.name == "codes" && bs.sf == Lp::Ue4m3 {
            // ue4m3 extremes: the largest scales (1.75 x 1.75: the product's exponent exceeds
            // the fields' sum) on even blocks, the smallest subnormal on odd blocks, small
            // products: the truncation point depends on how the scale exponent is taken.
            let ext = |rows: usize| -> Codes {
                (0..rows * 4 * nb)
                    .map(|j| {
                        if (j % nb).is_multiple_of(2) {
                            0x7e
                        } else {
                            0x01
                        }
                    })
                    .collect()
            };
            let small = |codes: &Codes| -> Codes { codes.iter().map(|&c| c & 0b1011).collect() };
            out.push(Set {
                name: "ue4m3_edge",
                a: small(&s.a),
                b: small(&s.b),
                first: None,
                sc: Some((ext(m), ext(n))),
            });
        }
        if s.name == "codes" {
            // Scale gaps of 28..43 binades between the blocks of an instruction (rows): where
            // the truncation of the smaller blocks begins.
            let gap = |rows: usize, side_a: bool| -> Codes {
                (0..rows * 4 * nb)
                    .map(|j| {
                        let (row, b) = (j / (4 * nb), j % nb);
                        let g = if side_a { 28 + (row % 16) as i32 } else { 0 };
                        if bs.sf == Lp::Ue8m0 {
                            (127 + 20 - g * (b % 2) as i32) as u32
                        } else {
                            code(4.0 - (g as f64 / 3.0) * (b % 2) as f64)
                        }
                    })
                    .collect()
            };
            out.push(Set {
                name: "scale_gap",
                a: s.a.clone(),
                b: s.b.clone(),
                first: None,
                sc: Some((gap(m, true), gap(n, false))),
            });
        }
        out.push(Set {
            name: s.name,
            a: s.a,
            b: s.b,
            first: None,
            sc: Some((sa, sb)),
        });
    }
    out
}

/// Pack codes into the probe's global layout: per 16-byte core-matrix row, `epc` elements:
/// 16 (FP8: a byte each; FP6: 12 bytes LSB-first + 4 zero bytes; FP4: 8 bytes, low nibble
/// first + 8 zero bytes: what kind::f8f6f4 / mxf8f6f4 read) or 32 packed FP4 (mxf4 kinds).
pub(crate) fn pack_rows(t: Lp, first: Option<Lp>, codes: &[u32], k: usize, epc: usize) -> Vec<u8> {
    let rows = codes.len() / k;
    let bytes_row = k / epc * 16;
    let mut out = vec![0u8; rows * bytes_row];
    for r in 0..rows {
        for c in 0..k / epc {
            let w = lowp::width(if c < 32 / epc { first.unwrap_or(t) } else { t }) as usize;
            let base = r * bytes_row + c * 16;
            let mut bits: u128 = 0;
            for j in 0..epc {
                bits |= (codes[r * k + c * epc + j] as u128) << (w * j);
            }
            for b in 0..16 {
                out[base + b] = if b * 8 < epc * w {
                    (bits >> (8 * b)) as u8
                } else {
                    0
                };
            }
        }
    }
    out
}

/// SFA / SFB words of the probes/mma_bs kernel: scale (row, instruction kc, block b) at byte
/// b (of 4) of TMEM lane row % 32 + 32 w, column 4 kc + row / 32, for the sub-partition w of
/// the D rows that read it: A's row m only in its own (w = m / 32), B's in all four.
pub(crate) fn sf_words(sc: &Codes, rows: usize, nb: usize, all_w: bool) -> Vec<u8> {
    let mut w = vec![0u32; 128 * 16];
    for row in 0..rows {
        for kc in 0..4 {
            for b in 0..nb {
                let c = sc[(row * 4 + kc) * nb + b];
                for sp in 0..4 {
                    if !all_w && sp != row / 32 {
                        continue;
                    }
                    let lane = row % 32 + 32 * sp;
                    w[lane * 16 + 4 * kc + row / 32] |= c << (8 * b);
                }
            }
        }
    }
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One set's operands for the model: codes decoded per element, scales per block.
struct Ops {
    a: Vec<Operand>,
    b: Vec<Operand>,
    sc: Option<(Vec<Operand>, Vec<Operand>)>,
}

/// A calibration candidate: the narrow-kind family, or (K = 64 packed-FP4 kinds) the FP4 one.
#[derive(Clone, Copy, Debug)]
enum Cand {
    Lp(LpMmaModel),
    Fp4(Fp4Model),
}

impl Cand {
    fn name(&self) -> String {
        match self {
            Cand::Lp(m) => m.name(),
            Cand::Fp4(m) => m.name(),
        }
    }
}

/// Outputs `outs` (row-major indices) of D = A * B^T through `kn`-wide steps (the first step
/// overwrites D), per the model.
fn gemm(md: &Cand, cfg: Cfg, o: &Ops, n: usize, k: usize, outs: &[usize]) -> Vec<u32> {
    let kn = cfg.kn();
    let (nb, blk) = cfg.bs.map_or((1, kn), |b| (b.kn / b.blk, b.blk));
    outs.iter()
        .map(|&ix| {
            let (mi, ni) = (ix / n, ix % n);
            let mut acc = 0u32;
            for kc in 0..k / kn {
                let a = &o.a[mi * k + kc * kn..mi * k + kc * kn + kn];
                let b = &o.b[ni * k + kc * kn..ni * k + kc * kn + kn];
                let first = if kc == 0 { None } else { Some(acc) };
                acc = match (&o.sc, md) {
                    (None, Cand::Lp(md)) => md.step(first, a, b, cfg.f16_d),
                    (Some((sa, sb)), Cand::Lp(md)) => {
                        let t: Vec<Operand> = (0..kn)
                            .map(|j| {
                                let bl = j / blk;
                                a[j].mul(&b[j])
                                    .mul(&sa[(mi * 4 + kc) * nb + bl])
                                    .mul(&sb[(ni * 4 + kc) * nb + bl])
                            })
                            .collect();
                        md.step_terms(first, &t, cfg.f16_d)
                    }
                    (Some((sa, sb)), Cand::Fp4(md)) => {
                        let p: Vec<f64> = (0..kn).map(|j| a[j].v * b[j].v).collect();
                        md.step(
                            first,
                            &p,
                            &sa[(mi * 4 + kc) * nb..(mi * 4 + kc) * nb + nb],
                            &sb[(ni * 4 + kc) * nb..(ni * 4 + kc) * nb + nb],
                            blk,
                        )
                    }
                    (None, Cand::Fp4(_)) => unreachable!(),
                };
            }
            acc
        })
        .collect()
}

/// Matches per set of each candidate on outputs `outs`, scored in parallel.
fn score(
    fam: &[Cand],
    cfg: Cfg,
    ops: &[Ops],
    hw: &[Vec<u32>],
    n: usize,
    k: usize,
    outs: &[usize],
) -> Vec<Vec<usize>> {
    let nt = std::thread::available_parallelism()
        .map_or(8, |x| x.get())
        .min(24);
    let mut res = vec![Vec::new(); fam.len()];
    std::thread::scope(|sc| {
        for (ci, chunk) in res.chunks_mut(fam.len().div_ceil(nt)).enumerate() {
            let base = ci * fam.len().div_ceil(nt);
            sc.spawn(move || {
                for (j, r) in chunk.iter_mut().enumerate() {
                    let md = &fam[base + j];
                    *r = ops
                        .iter()
                        .zip(hw)
                        .map(|(o, h)| {
                            gemm(md, cfg, o, n, k, outs)
                                .iter()
                                .zip(outs)
                                .filter(|(x, o)| **x == h[**o])
                                .count()
                        })
                        .collect();
                }
            });
        }
    });
    res
}

#[derive(Serialize)]
struct Score {
    model: String,
    matches: Vec<usize>,
    total: usize,
}

/// Write a copy of the probe case with its idesc parameter replaced.
fn case_with_idesc(
    case: &Case,
    dir: &Path,
    idesc: u32,
    idesc0: u32,
    entry: Option<&str>,
) -> Result<PathBuf> {
    let src = std::fs::read_to_string(case.dir.join("case.toml"))?;
    let mut out = String::new();
    let mut seen = 0;
    for l in src.lines() {
        if l.starts_with("value = ") && seen < 2 {
            out += &format!("value = {:#x}\n", if seen == 0 { idesc } else { idesc0 });
            seen += 1;
        } else if let (Some(e), true) = (entry, l.starts_with("entry = ")) {
            out += &format!("entry = \"{e}\"\n");
        } else if let Some(f) = l.strip_prefix("file = \"") {
            out += &format!("file = \"{}\n", case.dir.join(f).display());
        } else {
            out += l;
            out.push('\n');
        }
    }
    let d = dir.join(format!(
        "case_{}_{idesc:08x}_{idesc0:08x}",
        entry.unwrap_or("")
    ));
    std::fs::create_dir_all(&d)?;
    std::fs::write(d.join("case.toml"), out)?;
    std::fs::copy(case.dir.join(&case.m.ptx), d.join(&case.m.ptx))?;
    Ok(d)
}

pub(crate) fn idesc(cfg: Cfg) -> u32 {
    let common =
        fmt_code(cfg, cfg.a) << 7 | fmt_code(cfg, cfg.b) << 10 | (64 >> 3) << 17 | (128 >> 4) << 24;
    match cfg.bs {
        // Scale-factor ids 0; bit 23: ue8m0 scales.
        Some(b) => common | ((b.sf == Lp::Ue8m0) as u32) << 23,
        None => common | (if cfg.f16_d { 0 } else { 1 } << 4),
    }
}

pub fn run(case: &Case, out: &Option<PathBuf>, only: Option<&str>) -> Result<()> {
    let dir = crate::out_dir(case, out)?.join("calib");
    std::fs::create_dir_all(&dir)?;
    let (m, n) = (128usize, 64usize);
    let (_, cubin) = crate::load(case, "good", "ptx", &dir)?;
    // PTXV_GPU pins the GPU (it must still be free of other processes).
    let free = ptxv_hw::free_gpus()?;
    let pin = std::env::var("PTXV_GPU")
        .ok()
        .and_then(|g| g.parse::<u32>().ok());
    let gpu = match pin {
        Some(g) if free.contains(&g) => g,
        Some(g) => bail!("GPU {g} is not free"),
        None => *free.first().ok_or_else(|| anyhow::anyhow!("no free GPU"))?,
    };
    let exe = std::env::current_exe()?;
    let scaled = case.buffer("SFA").is_ok();
    let c = |a, b, f16_d| Cfg {
        a,
        b,
        f16_d,
        bs: None,
    };
    let cfgs: Vec<Cfg> = if !scaled {
        vec![
            c(Lp::E4m3, Lp::E4m3, false),
            c(Lp::E5m2, Lp::E5m2, false),
            c(Lp::E4m3, Lp::E5m2, false),
            c(Lp::E2m3, Lp::E2m3, false),
            c(Lp::E3m2, Lp::E3m2, false),
            c(Lp::E2m1, Lp::E2m1, false),
            c(Lp::E4m3, Lp::E2m1, false),
            c(Lp::E4m3, Lp::E4m3, true),
            c(Lp::E2m1, Lp::E2m1, true),
        ]
    } else {
        let mx8 = Bs {
            entry: "mma_mxf8",
            kn: 32,
            blk: 32,
            sf: Lp::Ue8m0,
        };
        let mx4 = Bs {
            entry: "mma_mxf4",
            kn: 64,
            blk: 32,
            sf: Lp::Ue8m0,
        };
        let nv4 = |sf| Bs {
            entry: "mma_nvf4",
            kn: 64,
            blk: 16,
            sf,
        };
        let b = |a, bb, bs| Cfg {
            a,
            b: bb,
            f16_d: false,
            bs: Some(bs),
        };
        vec![
            b(Lp::E4m3, Lp::E4m3, mx8),
            b(Lp::E5m2, Lp::E5m2, mx8),
            b(Lp::E2m1, Lp::E2m1, mx8),
            b(Lp::E4m3, Lp::E2m3, mx8),
            b(Lp::E2m1, Lp::E2m1, mx4),
            b(Lp::E2m1, Lp::E2m1, nv4(Lp::Ue8m0)),
            b(Lp::E2m1, Lp::E2m1, nv4(Lp::Ue4m3)),
            // (mma_nvf4_2x with ue4m3 scales: illegal instruction on GB300.)
        ]
    };
    let mut report = Vec::new();
    for cfg in cfgs {
        if only.is_some_and(|o| !cfg.name().contains(o)) {
            continue;
        }
        let (a, b, f16_d) = (cfg.a, cfg.b, cfg.f16_d);
        let k = 4 * cfg.kn();
        let nb = cfg.bs.map_or(1, |x| x.kn / x.blk);
        let sets = if scaled {
            bs_sets(cfg, m, n, k)
        } else {
            sets(cfg, m, n, k)
        };
        let fam: Vec<Cand> = if cfg.kn() == 64 {
            Fp4Model::family().into_iter().map(Cand::Fp4).collect()
        } else {
            LpMmaModel::family()
                .into_iter()
                .filter(|md| f16_d || !md.via32)
                .map(Cand::Lp)
                .collect()
        };
        println!(
            "calib-mma {} on GPU {gpu} (idesc {:#010x}), {} candidates",
            cfg.name(),
            idesc(cfg),
            fam.len()
        );
        let mut hw: Vec<Vec<u32>> = Vec::new();
        for s in &sets {
            let tag = format!("{}.{}", cfg.name().replace(['>', '-', ':'], ""), s.name);
            let pa = dir.join(format!("{tag}.A.bin"));
            let pb = dir.join(format!("{tag}.B.bin"));
            let pd = dir.join(format!("{tag}.D.hw.bin"));
            std::fs::write(&pa, pack_rows(a, s.first.map(|f| f.0), &s.a, k, cfg.epc()))?;
            std::fs::write(&pb, pack_rows(b, s.first.map(|f| f.1), &s.b, k, cfg.epc()))?;
            let mut extra = Vec::new();
            if let Some((sa, sb)) = &s.sc {
                let (psa, psb) = (
                    dir.join(format!("{tag}.SFA.bin")),
                    dir.join(format!("{tag}.SFB.bin")),
                );
                std::fs::write(&psa, sf_words(sa, m, nb, false))?;
                std::fs::write(&psb, sf_words(sb, n, nb, true))?;
                extra = vec![
                    "--input".to_string(),
                    format!("SFA={}", psa.display()),
                    "--input".to_string(),
                    format!("SFB={}", psb.display()),
                ];
            }
            let first = s.first.map(|(fa, fb)| Cfg {
                a: fa,
                b: fb,
                f16_d,
                bs: None,
            });
            let cdir = case_with_idesc(
                case,
                &dir,
                idesc(cfg),
                idesc(first.unwrap_or(cfg)),
                cfg.bs.map(|x| x.entry),
            )?;
            let o = std::process::Command::new(&exe)
                .args([
                    "hw-worker",
                    cdir.to_str().unwrap(),
                    cubin.to_str().unwrap(),
                    "--runs",
                    "3",
                ])
                .args([
                    "--input",
                    &format!("A={}", pa.display()),
                    "--input",
                    &format!("B={}", pb.display()),
                    "--dump",
                    pd.to_str().unwrap(),
                ])
                .args(&extra)
                .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
                .output()?;
            if !o.status.success() {
                bail!("hw run failed: {}", String::from_utf8_lossy(&o.stderr));
            }
            let r: serde_json::Value = serde_json::from_slice(&o.stdout)?;
            if r["distinct"].as_array().map(|a| a.len()) != Some(1) {
                bail!("hardware output is not deterministic for set {}", s.name);
            }
            hw.push(
                std::fs::read(&pd)?
                    .chunks(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            );
        }
        let opnd = |codes: &[u32], t: Lp, first: Option<Lp>| -> Vec<Operand> {
            codes
                .iter()
                .enumerate()
                .map(|(i, &c)| Operand::new(c, if i % k < 32 { first.unwrap_or(t) } else { t }))
                .collect()
        };
        let ops: Vec<Ops> = sets
            .iter()
            .map(|s| Ops {
                a: opnd(&s.a, a, s.first.map(|f| f.0)),
                b: opnd(&s.b, b, s.first.map(|f| f.1)),
                sc: s.sc.as_ref().map(|(sa, sb)| {
                    let sf = cfg.bs.unwrap().sf;
                    (
                        sa.iter().map(|&c| Operand::scale(c, sf)).collect(),
                        sb.iter().map(|&c| Operand::scale(c, sf)).collect(),
                    )
                }),
            })
            .collect();
        // Stage 1: every candidate on a sample of 256 outputs per set; stage 2: the best 64
        // (and every one tied with the best) on all outputs.
        let cand = fam;
        let sample: Vec<usize> = (0..256).map(|i| (i * 2654435761usize) % (m * n)).collect();
        let s1 = score(&cand, cfg, &ops, &hw, n, k, &sample);
        let mut order: Vec<usize> = (0..cand.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(s1[i].iter().sum::<usize>()));
        let best = s1[order[0]].iter().sum::<usize>();
        let top: Vec<Cand> = order
            .iter()
            .enumerate()
            .take_while(|&(j, &i)| j < 64 || s1[i].iter().sum::<usize>() == best)
            .map(|(_, &i)| cand[i])
            .collect();
        let all: Vec<usize> = (0..m * n).collect();
        let s2 = score(&top, cfg, &ops, &hw, n, k, &all);
        let mut scores: Vec<Score> = top
            .iter()
            .zip(s2)
            .map(|(md, matches)| Score {
                model: md.name(),
                matches,
                total: m * n,
            })
            .collect();
        scores.sort_by_key(|s| std::cmp::Reverse(s.matches.iter().sum::<usize>()));
        println!(
            "  {:<34} {}",
            "model",
            sets.iter()
                .map(|s| format!("{:>12}", s.name))
                .collect::<String>()
        );
        for s in scores.iter().take(6) {
            println!(
                "  {:<34} {}",
                s.model,
                s.matches
                    .iter()
                    .map(|x| format!("{:>12}", format!("{x}/{}", s.total)))
                    .collect::<String>()
            );
        }
        let perfect: Vec<String> = scores
            .iter()
            .filter(|s| s.matches.iter().all(|&x| x == s.total))
            .map(|s| s.model.clone())
            .collect();
        println!(
            "  matching every output: {}",
            if perfect.is_empty() {
                "NONE".to_string()
            } else {
                perfect.join(", ")
            }
        );
        report.push((
            cfg.name(),
            perfect,
            scores.into_iter().take(12).collect::<Vec<_>>(),
        ));
    }
    let rp = dir.join(if scaled {
        "calib_bs.json"
    } else {
        "calib_f8.json"
    });
    std::fs::write(&rp, serde_json::to_string_pretty(&report)?)?;
    println!("  written {}", rp.display());
    Ok(())
}
