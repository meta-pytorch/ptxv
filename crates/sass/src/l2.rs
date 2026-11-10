/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! SASS forms probed on GB300 by agent L2 (2026-11-06), lowered ahead of `lower_line`'s main
//! match: FRND, F2F between f64 and f16 / bf16 and between f16 and bf16, RPCMOV.32,
//! UBLKRED.G.S.ADD.F32.RN, VIMNMX / VIMNMX3 .U16x2 / .S16x2, `LEA.HI.X[.SX32]` with a `~Rb`
//! operand, `FCHK P, -Ra, b`. Every modifier outside the probed sets is refused.

use anyhow::Result;
use anyhow::bail;
use ptxv_ir::*;

use super::CubinMeta;
use super::Lw;

/// The forms `lower` takes over from `lower_line` (it refuses unprobed variants of them).
pub(super) fn claims(lw: &Lw, base: &str, mods: &[&str]) -> bool {
    let op = |i: usize| lw.l.operands.get(i).map(|s| s.as_str()).unwrap_or("");
    match base {
        "FRND" | "RPCMOV" | "UBLKRED" | "UTMAREDG" => true,
        "UBLKCP" => mods == ["S", "S"],
        "BMOV" => true,
        "F2FP" => {
            mods.last() == Some(&"UNPACK_B")
                && [".B1", ".B2", ".B3"]
                    .iter()
                    .any(|b| op(1).replace(".reuse", "").ends_with(b))
        }
        "F2F" => {
            mods.len() >= 2
                && matches!(
                    (mods[0], mods[1]),
                    ("BF16" | "F16", "F64")
                        | ("F64", "BF16" | "F16")
                        | ("BF16", "F16")
                        | ("F16", "BF16")
                )
        }
        "VIMNMX" | "VIMNMX3" => mods.iter().any(|m| m.ends_with("16x2") || *m == "RELU"),
        "VIADDMNMX" => mods.contains(&"RELU"),
        "LEA" | "ULEA" if mods.contains(&"HI") => lw.l.operands.iter().any(|o| o.starts_with('~')),
        // `LEA Rd, Pc, Ra, -Rb, s` (a carry-out with a negated b).
        "LEA" => {
            mods.is_empty()
                && lw.l.operands.len() == 5
                && op(1).starts_with('P')
                && op(1) != "PT"
                && op(3).starts_with("-R")
        }
        "FCHK" => op(1).starts_with(['-', '|']) || op(2).starts_with('|'),
        "BREAK" => lw.l.operands.len() == 2,
        "CCTL" => mods.iter().any(|m| m.starts_with("PF")),
        "F2I" => mods.iter().any(|m| matches!(*m, "S16" | "U16")),
        _ => false,
    }
}

fn rnd_of(m: &str) -> Option<Rnd> {
    match m {
        "RZ" | "TRUNC" => Some(Rnd::Rz),
        "RM" | "FLOOR" => Some(Rnd::Rm),
        "RP" | "CEIL" => Some(Rnd::Rp),
        _ => None,
    }
}

/// A 16-bit source `Ra` / `Ra.H1` (or a 32 / 64-bit register): (value, high half).
fn half_src(lw: &Lw, s: &str, bytes: u8) -> Result<(Val, bool)> {
    let s = s.trim().trim_end_matches(".reuse");
    match s.strip_suffix(".H1") {
        Some(r) => Ok((lw.val(r, bytes)?, true)),
        None => Ok((lw.val(s, bytes)?, false)),
    }
}

pub(super) fn lower(lw: &Lw, base: &str, mods: &[&str], meta: &CubinMeta) -> Result<Op> {
    let has = |m: &str| mods.contains(&m);
    Ok(match base {
        // FRND[.F64|.F16|.BF16][.FTZ][.TRUNC|.FLOOR|.CEIL] Rd, Ra (cvt.r{n,z,m,p}i[.ftz]): round
        // to an integral value in the same format; `.FTZ` only on f32 (the probed set). A
        // 16-bit source is the register's low half (no `.H1` form probed). ASSUME(N-FCVT)
        "FRND" => {
            let mut fmt = FFmt::F32;
            let mut rnd = Rnd::Rn;
            let (mut nf, mut nr) = (0, 0);
            for m in mods {
                match *m {
                    "F64" => (fmt, nf) = (FFmt::F64, nf + 1),
                    "F16" => (fmt, nf) = (FFmt::F16, nf + 1),
                    "BF16" => (fmt, nf) = (FFmt::Bf16, nf + 1),
                    "FTZ" => {}
                    m if matches!(m, "TRUNC" | "FLOOR" | "CEIL") => {
                        (rnd, nr) = (rnd_of(m).unwrap(), nr + 1)
                    }
                    m => bail!(lw.err(format!("FRND modifier not modeled: {m}"))),
                }
            }
            if nf > 1 || nr > 1 || (has("FTZ") && fmt != FFmt::F32) || lw.l.operands.len() != 2 {
                bail!(lw.err("FRND form not modeled"));
            }
            Op::FCvt {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, if fmt == FFmt::F64 { 8 } else { 4 })?,
                from: fmt,
                to: fmt,
                rnd,
                ftz: has("FTZ"),
                int: true,
                hi: false,
            }
        }
        // F2F.{BF16,F16}.F64[.RZ|.RM|.RP] (narrowing, rounded), F2F.F64.{BF16,F16} Rd, Ra[.H1]
        // (widening, exact), F2F.BF16.F16[.R*] / F2F.F16.BF16[.R*] Rd, Ra[.H1]. ASSUME(N-FCVT)
        "F2F" => {
            let f = |m: &str| match m {
                "F64" => FFmt::F64,
                "F16" => FFmt::F16,
                _ => FFmt::Bf16,
            };
            let (to, from) = (f(mods[0]), f(mods[1]));
            let rnd = match &mods[2..] {
                [] => Rnd::Rn,
                [m] if to != FFmt::F64 && matches!(*m, "RZ" | "RM" | "RP") => rnd_of(m).unwrap(),
                _ => bail!(lw.err(format!("F2F.{} not modeled", mods.join(".")))),
            };
            if lw.l.operands.len() != 2 {
                bail!(lw.err("F2F form not modeled"));
            }
            let (a, hi) = if from == FFmt::F64 {
                (lw.val(lw.op(1)?, 8)?, false)
            } else {
                half_src(lw, lw.op(1)?, 4)?
            };
            Op::FCvt {
                d: lw.dst(lw.op(0)?)?,
                a,
                from,
                to,
                rnd,
                ftz: false,
                int: false,
                hi,
            }
        }
        // `RPCMOV.32 Rpc.LO, Ra` / `RPCMOV.32 Rd, Rpc.LO`: ptxas parks a register in the low
        // half of the return-PC register under register pressure (CuTe DSL kernels). Modeled
        // as a per-thread register past the kernel's register count, which no instruction of
        // the kernel can name (`check_kernel` refuses kernels with a CALL that could clobber
        // Rpc). GB300: probes/rpcmov. ASSUME(L-RPCMOV)
        "RPCMOV" => {
            if mods != ["32"] || lw.l.operands.len() != 2 {
                bail!(lw.err("RPCMOV form not modeled"));
            }
            let rc = meta.regcount;
            if rc == 0 || rc >= 255 {
                bail!(lw.err(format!(
                    "RPCMOV with register count {rc}: no free register to model Rpc"
                )));
            }
            let rpc = Reg {
                file: File::R,
                idx: rc,
            };
            match (lw.op(0)?, lw.op(1)?) {
                ("Rpc.LO", a) => Op::Mov {
                    d: rpc,
                    a: lw.val(a, 4)?,
                },
                (d, "Rpc.LO") => Op::Mov {
                    d: lw.dst(d)?,
                    a: Val::Reg(rpc, 1),
                },
                _ => bail!(lw.err("RPCMOV operands not modeled (only Rpc.LO)")),
            }
        }
        // `UBLKRED.G.S.ADD.F32.RN [URg], [URs], URn` (cp.reduce.async.bulk.global.shared::cta
        // .bulk_group.add.f32): as UBLKCP.G.S (URn 16-byte units, the issuer's bulk group),
        // each f32 added to the global word. GB300: probes/bulk_red. ASSUME(N-BULK-RED)
        "UBLKRED" => {
            if mods != ["G", "S", "ADD", "F32", "RN"] {
                bail!(lw.err(format!("UBLKRED.{} not modeled", mods.join("."))));
            }
            Op::TmaStore {
                dims: 0,
                src: lw.ureg_at(lw.op(1)?, 0, 1)?,
                off: 0,
                tmap: lw.val(lw.op(2)?, 4)?,
                coords: vec![lw.ureg_at(lw.op(0)?, 0, 2)?],
                red: true,
            }
        }
        // `F2FP[.RELU].<F16|BF16>.E2M1.UNPACK_B Rd, Ra.B1|.B2|.B3` (cvt.rn[.relu].f16x2.e2m1x2
        // of byte k of a register): as the byte-0 form from byte k. Only E2M1 (an x2 source is
        // one byte). GB300: probes/e2m1_bytes. ASSUME(N-CVT-LP)
        "F2FP" => {
            let Some((to, from, super::F2fpKind::Unpack, _, _, relu)) = super::f2fp_lp(mods) else {
                bail!(lw.err(format!(
                    "F2FP.{} with a byte selector not modeled",
                    mods.join(".")
                )));
            };
            if from != Lp::E2m1 || lw.l.operands.len() != 2 {
                bail!(lw.err(format!(
                    "F2FP.{} with a byte selector not modeled",
                    mods.join(".")
                )));
            }
            let o = lw.op(1)?.replace(".reuse", "");
            let (r, b) = o
                .rsplit_once(".B")
                .ok_or_else(|| lw.err("bad byte selector"))?;
            Op::CvtUnpack {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(r, 4)?,
                hi: false,
                from,
                to,
                relu,
                byte: b.parse().map_err(|_| lw.err("bad byte selector"))?,
            }
        }
        // `BMOV.32[.CLEAR] Rd, Bb` / `BMOV.32 Bb, Ra` (Triton: save, clear and restore a
        // BSSY barrier around a region). ASSUME(E-BSYNC)
        "BMOV" => {
            let clear = match mods {
                ["32"] => false,
                ["32", "CLEAR"] => true,
                _ => bail!(lw.err(format!("BMOV.{} not modeled", mods.join(".")))),
            };
            let bar = |s: &str| {
                s.strip_prefix('B')
                    .and_then(|x| x.parse::<u8>().ok())
                    .filter(|&b| b < 16)
            };
            let (o0, o1) = (lw.op(0)?, lw.op(1)?);
            match (bar(o0), bar(o1)) {
                (Some(b), None) if !clear => Op::BMov {
                    b,
                    r: lw.dst(o1)?,
                    to_b: true,
                    clear: false,
                },
                (None, Some(b)) => Op::BMov {
                    b,
                    r: lw.dst(o0)?,
                    to_b: false,
                    clear,
                },
                _ => bail!(lw.err("BMOV operands not modeled")),
            }
        }
        // `UBLKCP.S.S [URd], [URs], URn` (cp.async.bulk.shared::cluster.shared::cta
        // .mbarrier::complete_tx::bytes): URd = the destination's shared::cluster address,
        // URd+1 its mbarrier's; URs = the source in this CTA's shared memory; URn 16-byte units.
        // GB300: probes/bulk_s2s. ASSUME(S-TMA)
        "UBLKCP" => {
            if lw.l.operands.len() != 3 {
                bail!(lw.err("UBLKCP.S.S operands not modeled"));
            }
            let s = lw.op(0)?;
            Op::TmaLoad {
                dims: 0,
                dst: lw.ureg_at(s, 0, 1)?,
                off: 0,
                mbar: lw.ureg_at(s, 1, 1)?,
                mbar_off: 0,
                tmap: lw.val(lw.op(2)?, 4)?,
                coords: vec![lw.ureg_at(lw.op(1)?, 0, 1)?],
                mask: None,
                pair: false,
                src_smem: true,
            }
        }
        // `UTMAREDG.<N>D.ADD [URa], [URb]` (cp.reduce.async.bulk.tensor.<N>d.global.shared::cta
        // .add.tile.bulk_group): as UTMASTG (URa: smem source then N coordinates; URb: the
        // tensor map), each element added to the global one in the map's type (f32 / f16 /
        // bf16; others unmodeled at run time). GB300: probes/tma_red. ASSUME(N-BULK-RED)
        "UTMAREDG" => {
            let dims: u8 = match mods {
                [d, "ADD"] => d
                    .strip_suffix('D')
                    .and_then(|n| n.parse().ok())
                    .filter(|n| (1..=5).contains(n))
                    .ok_or_else(|| lw.err(format!("UTMAREDG.{} not modeled", mods.join("."))))?,
                _ => bail!(lw.err(format!("UTMAREDG.{} not modeled", mods.join(".")))),
            };
            // A third operand `desc[URZ]`: no L2 cache-policy descriptor (the bf16x6 FA4
            // backward); any other descriptor is refused.
            let ok = match lw.l.operands.len() {
                2 => true,
                3 => lw.op(2)?.replace(' ', "") == "desc[URZ]",
                _ => false,
            };
            if !ok {
                bail!(lw.err("UTMAREDG operands not modeled"));
            }
            let t = lw.op(0)?;
            let coords = (0..dims as u16)
                .map(|k| lw.ureg_at(t, 1 + k, 1))
                .collect::<Result<_>>()?;
            Op::TmaStore {
                dims,
                src: lw.ureg_at(t, 0, 1)?,
                off: 0,
                tmap: lw.ureg_at(lw.op(1)?, 0, 2)?,
                coords,
                red: true,
            }
        }
        // VIMNMX.{U16x2,S16x2} d, PT, PT, a, b, sel / VIMNMX3.{U16x2,S16x2} d, a, b, c, sel:
        // per 16-bit lane, sel PT = min, !PT = max (min / max .u16x2 / .s16x2). `.RELU`
        // (signed only: VIMNMX.RELU, VIMNMX.S16x2.RELU) clamps the result at 0. The predicate
        // outputs are refused. GB300: probes/int16x2, probes/imnmx_relu. ASSUME(N-INT16X2)
        "VIMNMX" | "VIMNMX3" => {
            let relu = mods.contains(&"RELU");
            let (signed, x16) = match mods
                .iter()
                .filter(|m| **m != "RELU")
                .copied()
                .collect::<Vec<_>>()[..]
            {
                ["U16x2"] if !relu => (false, true),
                ["S16x2"] => (true, true),
                [] if relu => (true, false),
                _ => bail!(lw.err(format!("{base}.{} not modeled", mods.join(".")))),
            };
            if base == "VIMNMX3" && (relu || !x16) {
                bail!(lw.err(format!("{base}.{} not modeled", mods.join("."))));
            }
            let sel = |s: &str| -> Result<bool> {
                match s {
                    "PT" => Ok(false),
                    "!PT" => Ok(true),
                    _ => bail!(lw.err(format!("{base} with a predicate selector not modeled"))),
                }
            };
            let d = lw.dst(lw.op(0)?)?;
            if base == "VIMNMX" {
                if lw.l.operands.len() != 6 || lw.op(1)? != "PT" || lw.op(2)? != "PT" {
                    bail!(lw.err("VIMNMX predicate outputs not modeled"));
                }
                let b = lw.val(lw.op(4)?, 4)?;
                Op::IntMnmx3 {
                    d,
                    a: lw.val(lw.op(3)?, 4)?,
                    b: b.clone(),
                    c: b,
                    signed,
                    max: sel(lw.op(5)?)?,
                    x16,
                    relu,
                }
            } else {
                if lw.l.operands.len() != 5 {
                    bail!(lw.err("VIMNMX3 form not modeled"));
                }
                Op::IntMnmx3 {
                    d,
                    a: lw.val(lw.op(1)?, 4)?,
                    b: lw.val(lw.op(2)?, 4)?,
                    c: lw.val(lw.op(3)?, 4)?,
                    signed,
                    max: sel(lw.op(4)?)?,
                    x16,
                    relu,
                }
            }
        }
        // `VIADDMNMX.RELU d, a, b, c, sel` (an add fused with min / max.relu.s32): signed
        // min / max of (a + b) mod 2^32 and c, clamped at 0; `-Rb` (a subtraction fused):
        // a - b. GB300: probes/imnmx_relu, probes/iaddmnmx_relu. ASSUME(L-ROSETTA)
        "VIADDMNMX" => {
            let bt = lw.op(2)?.replace(".reuse", "");
            let neg_b = bt.starts_with("-R");
            if mods != ["RELU"]
                || lw.l.operands.len() != 5
                || bt.starts_with('~')
                || bt.starts_with('-') && !neg_b
            {
                bail!(lw.err(format!("VIADDMNMX.{} form not modeled", mods.join("."))));
            }
            Op::AddMinMax {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                b: lw.val(bt.trim_start_matches('-'), 4)?,
                c: lw.val(lw.op(3)?, 4)?,
                signed: true,
                max: match lw.op(4)? {
                    "PT" => false,
                    "!PT" => true,
                    _ => bail!(lw.err("VIADDMNMX with a predicate selector not modeled")),
                },
                neg_b,
                relu: true,
            }
        }
        // `[U]LEA.HI.X.SX32 Rd, Ra, ~Rb, s, P` (the high word of `(sext(a) << s) - y`; the
        // uniform form in cuBLASLt's bf16x9 kernels): b enters
        // complemented, plus the carry-in. Only that form (the one in the corpus) is lowered.
        // GB300: probes/int16x2 (cubins/lea.cubin). ASSUME(L-ROSETTA)
        // `LEA Rd, Pc, Ra, -Rb, s`: (a << s) + ~b + 1 and its carry (the low word of
        // `(sext(a) << s) - y`, before LEA.HI.X.SX32 ~Rb). GB300: probes/int16x2.
        // ASSUME(L-ROSETTA)
        "LEA" if mods.is_empty() => {
            if lw.op(2)?.starts_with(['-', '~']) {
                bail!(lw.err("LEA with two negated operands not modeled"));
            }
            Op::Lea {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(2)?, 4)?,
                b: lw.val(lw.op(3)?.trim_start_matches('-'), 4)?,
                shift: super::hex(lw.op(4)?).ok_or_else(|| lw.err("bad shift"))? as u8,
                cout: Some(lw.pdst(lw.op(1)?)?),
                neg_a: false,
                neg_b: true,
            }
        }
        "LEA" | "ULEA" => {
            if mods != ["HI", "X", "SX32"] || lw.l.operands.len() != 5 {
                bail!(lw.err(format!("LEA.{} with ~Rb not modeled", mods.join("."))));
            }
            let b = lw.op(2)?;
            let Some(b) = b.strip_prefix('~') else {
                bail!(lw.err("LEA.HI.X.SX32: only the b operand may be complemented"));
            };
            if lw.op(1)?.starts_with(['~', '-']) {
                bail!(lw.err("LEA.HI.X.SX32 with a modified a operand not modeled"));
            }
            Op::LeaHi {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                b: lw.val(b, 4)?,
                neg_b: false,
                not_b: true,
                c: Val::Imm(0),
                shift: super::hex(lw.op(3)?).ok_or_else(|| lw.err("bad shift"))? as u8,
                sx32: true,
                cin: Some(lw.pred(lw.op(4)?)?),
                cout: None,
            }
        }
        // `FCHK P, -Ra, b` (div.rn of a negated numerator) and `FCHK P, |Ra|, |Rb|` (Triton's
        // lerp): the predicate ignores the operands' signs (GB300: probes/fchk_neg and
        // probes/fchk_abs equal probes/fchk bit for bit). A negated divisor is refused.
        // ASSUME(N-FCHK)
        "FCHK" => {
            let abs = |s: &str| -> Option<String> {
                s.strip_prefix('|')
                    .and_then(|x| x.strip_suffix('|'))
                    .map(str::to_string)
            };
            let a0 = lw.op(1)?;
            let a = abs(a0).unwrap_or_else(|| a0.trim_start_matches('-').to_string());
            let b0 = lw.op(2)?;
            let b = abs(b0).unwrap_or_else(|| b0.to_string());
            if !mods.is_empty()
                || lw.l.operands.len() != 3
                || a.starts_with(['-', '|'])
                || b.starts_with(['-', '|'])
            {
                bail!(lw.err("FCHK form not modeled"));
            }
            Op::Fchk {
                d: lw.pdst(lw.op(0)?)?,
                a: lw.val(&a, 4)?,
                b: lw.fval(&b)?,
            }
        }
        // `BREAK[.RELIABLE] Pc, Bb`: the lanes where the guard and Pc hold leave barrier b
        // (ptxas pairs it with `BRA Pc, target` under the same guard; S-BRA-COND).
        // ASSUME(E-BSYNC)
        "BREAK" => {
            if !(mods.is_empty() || mods == ["RELIABLE"]) {
                bail!(lw.err(format!("BREAK.{} not modeled", mods.join("."))));
            }
            let b = lw
                .op(1)?
                .strip_prefix('B')
                .and_then(|x| x.parse::<u8>().ok())
                .filter(|&b| b < 16)
                .ok_or_else(|| lw.err("bad convergence barrier"))?;
            Op::Break {
                b,
                cond: Some(lw.pred(lw.op(0)?)?),
            }
        }
        // `CCTL.E.PF1 [Ra]` / `CCTL.E.PF2 [Ra]` (prefetch.global.L1 / .L2): a cache hint, no
        // architectural effect (probes/f2i16 runs both). Other prefetch forms are refused.
        "CCTL" => {
            if !(mods == ["E", "PF1"] || mods == ["E", "PF2"]) || lw.l.operands.len() != 1 {
                bail!(lw.err(format!("CCTL.{} not modeled", mods.join("."))));
            }
            Op::Nop
        }
        // `F2I[.FTZ].{S16,U16}[.TRUNC|.FLOOR|.CEIL].NTZ Rd, Ra` (cvt.r*i[.ftz].{s16,u16}.f32):
        // saturated to the 16-bit range, NaN 0, sign- / zero-extended in Rd (probes/f2i16).
        // ASSUME(N-F2I16)
        "F2I" => {
            let mut signed = None;
            let mut rnd = Rnd::Rn;
            let mut nr = 0;
            for m in mods {
                match *m {
                    "S16" | "U16" if signed.is_none() => signed = Some(*m == "S16"),
                    "TRUNC" | "FLOOR" | "CEIL" => (rnd, nr) = (rnd_of(m).unwrap(), nr + 1),
                    "FTZ" | "NTZ" => {}
                    m => bail!(lw.err(format!("F2I modifier not modeled: {m}"))),
                }
            }
            if nr > 1 || !has("NTZ") || lw.l.operands.len() != 2 {
                bail!(lw.err(format!("F2I.{} not modeled", mods.join("."))));
            }
            Op::F2I {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                signed: signed.unwrap(),
                bits: 16,
                rnd,
                ftz: has("FTZ"),
                f64: false,
            }
        }
        _ => unreachable!("l2::claims"),
    })
}

/// Kernel-level conditions of the forms above. RPCMOV's Rpc model holds if no CALL can run
/// while Rpc holds a parked value. Kernels whose only CALLs go into traps (ptxas's tcgen05
/// guardrail functions start with `BPT.TRAP`) qualify as a whole; otherwise every restore
/// must follow its save in straight-line code (no control transfer and no branch target in
/// between), which no CALL can interrupt (CUTLASS's grouped GEMMs: save and restore four
/// instructions apart). ASSUME(L-RPCMOV)
pub(super) fn check_kernel(code: &[Inst]) -> Result<()> {
    check_bmov(code)?;
    check_rpcmov(code)
}

/// A register `BMOV.32[.CLEAR] Rd, Bb` saves a barrier mask into is only read by
/// `BMOV.32 Bb', Rd` until it is overwritten (code order): the model's mask encoding is never
/// observed by arithmetic. ASSUME(E-BSYNC)
fn check_bmov(code: &[Inst]) -> Result<()> {
    for (i, inst) in code.iter().enumerate() {
        let Op::BMov { r, to_b: false, .. } = inst.op else {
            continue;
        };
        for later in &code[i + 1..] {
            let restore = matches!(later.op, Op::BMov { r: q, to_b: true, .. } if q == r);
            if !restore && later.op.src_regs().contains(&r) {
                bail!(
                    "`{}` saves a convergence barrier into a register that `{}` reads: BMOV's mask encoding is not modeled",
                    inst.text,
                    later.text
                );
            }
            if later.op.dst_regs().contains(&r) {
                break;
            }
        }
    }
    Ok(())
}

fn check_rpcmov(code: &[Inst]) -> Result<()> {
    let is_save = |i: &Inst| i.text.contains("RPCMOV") && i.text.contains("Rpc.LO,");
    let is_restore = |i: &Inst| i.text.contains("RPCMOV") && !is_save(i);
    if !code.iter().any(|i| i.text.contains("RPCMOV")) {
        return Ok(());
    }
    let calls_only_traps = code.iter().all(|i| match &i.op {
        Op::Call { target } => matches!(code.get(*target).map(|t| &t.op), Some(Op::Trap { .. })),
        Op::CallAbs { .. } => false,
        _ => true,
    });
    if calls_only_traps {
        return Ok(());
    }
    let mut targets = std::collections::HashSet::new();
    for i in code {
        match &i.op {
            Op::Bra { target, .. } | Op::Call { target } => {
                targets.insert(*target);
            }
            Op::Brx { targets: t, .. } => targets.extend(t.iter().copied()),
            _ => {}
        }
    }
    let control = |i: &Inst| {
        matches!(
            i.op,
            Op::Bra { .. }
                | Op::Brx { .. }
                | Op::Call { .. }
                | Op::CallAbs { .. }
                | Op::Ret { .. }
                | Op::Exit
                | Op::Trap { .. }
        )
    };
    for (r, i) in code.iter().enumerate() {
        if !is_restore(i) {
            continue;
        }
        let mut k = r;
        let ok = loop {
            if targets.contains(&k) || k == 0 {
                break false;
            }
            k -= 1;
            if is_save(&code[k]) {
                break !(control(&code[k]) || !code[k].guard.is_always());
            }
            if control(&code[k]) {
                break false;
            }
        };
        if !ok || !i.guard.is_always() {
            bail!(
                "sass {:#06x}: RPCMOV restore not preceded by its save in straight-line code, in a kernel with calls (Rpc may be clobbered): not modeled\n    `{}`",
                i.loc,
                i.text
            );
        }
    }
    Ok(())
}
