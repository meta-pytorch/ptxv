/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PTX forms of agent L2's SASS lifts (sass/src/l2.rs), lowered to the same IR ops so the
//! probes replay on both front-ends: `cvt.r{n,z,m,p}i[.ftz]` between equal float types,
//! `cvt` between f64 and f16 / bf16 and between f16 and bf16, `min` / `max` `.u16x2` /
//! `.s16x2`, `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`.

use anyhow::Result;
use anyhow::bail;
use ptxv_ir::*;

use super::Lower;

fn ffmt(t: &str) -> Option<FFmt> {
    match t {
        "f16" => Some(FFmt::F16),
        "bf16" => Some(FFmt::Bf16),
        "f32" => Some(FFmt::F32),
        "f64" => Some(FFmt::F64),
        _ => None,
    }
}

/// The forms `lower` handles (it refuses unmodeled variants of them).
pub(super) fn claims(base: &str, mods: &[&str]) -> bool {
    let n = mods.len();
    match base {
        "cvt" if n >= 2 && matches!(mods[n - 2], "s16" | "u16") && mods[n - 1] == "f32" => mods
            .iter()
            .any(|m| matches!(*m, "rni" | "rzi" | "rmi" | "rpi")),
        "cvt" if n >= 2 => {
            let (to, from) = (mods[n - 2], mods[n - 1]);
            let int = mods
                .iter()
                .any(|m| matches!(*m, "rni" | "rzi" | "rmi" | "rpi"));
            (int && to == from && ffmt(to).is_some())
                || matches!(
                    (to, from),
                    ("bf16" | "f16", "f64")
                        | ("f64", "bf16" | "f16")
                        | ("bf16", "f16")
                        | ("f16", "bf16")
                )
        }
        "min" | "max" => {
            mods.last().is_some_and(|t| matches!(*t, "u16x2" | "s16x2")) || mods.contains(&"relu")
        }
        "cp" => {
            mods.first() == Some(&"reduce")
                || mods
                    == [
                        "async",
                        "bulk",
                        "shared::cluster",
                        "shared::cta",
                        "mbarrier::complete_tx::bytes",
                    ]
        }
        _ => false,
    }
}

pub(super) fn lower(lw: &Lower, base: &str, mods: &[&str], ops: &[String]) -> Result<Op> {
    let op_n = |i: usize| -> Result<&str> {
        ops.get(i)
            .map(|s| s.as_str())
            .ok_or_else(|| lw.err(format!("missing operand {i}")))
    };
    let n = mods.len();
    Ok(match base {
        // cvt.r{n,z,m,p}i[.ftz].{s16,u16}.f32 (SASS F2I[.FTZ].{S16,U16}...NTZ): saturated,
        // NaN 0, extended in the register. ASSUME(N-F2I16)
        "cvt" if matches!(mods[n - 2], "s16" | "u16") => {
            let mut rnd = None;
            for m in &mods[..n - 2] {
                let r = match *m {
                    "rni" => Rnd::Rn,
                    "rzi" => Rnd::Rz,
                    "rmi" => Rnd::Rm,
                    "rpi" => Rnd::Rp,
                    "ftz" => continue,
                    m => bail!(lw.err(format!("cvt modifier .{m} not modeled"))),
                };
                if rnd.replace(r).is_some() {
                    bail!(lw.err("cvt with two roundings"));
                }
            }
            Op::F2I {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                signed: mods[n - 2] == "s16",
                bits: 16,
                rnd: rnd.unwrap(),
                ftz: mods.contains(&"ftz"),
                f64: false,
            }
        }
        // ASSUME(N-FCVT): GB300 probes/fcvt (both front-ends).
        "cvt" => {
            let (to, from) = (ffmt(mods[n - 2]).unwrap(), ffmt(mods[n - 1]).unwrap());
            let mut rnd = None;
            let mut int = false;
            let mut ftz = false;
            for m in &mods[..n - 2] {
                let r = match *m {
                    "rn" | "rni" => Rnd::Rn,
                    "rz" | "rzi" => Rnd::Rz,
                    "rm" | "rmi" => Rnd::Rm,
                    "rp" | "rpi" => Rnd::Rp,
                    "ftz" if to == FFmt::F32 => {
                        ftz = true;
                        continue;
                    }
                    m => bail!(lw.err(format!("cvt modifier .{m} not modeled"))),
                };
                if rnd.replace(r).is_some() {
                    bail!(lw.err("cvt with two roundings"));
                }
                int = m.ends_with('i');
            }
            let widen = to == FFmt::F64 && from != FFmt::F64;
            let rnd = match rnd {
                Some(r) if !widen => r,
                None if widen => Rnd::Rn,
                _ => bail!(lw.err(format!("cvt.{} not modeled", mods.join(".")))),
            };
            if int != (to == from) || ops.len() != 2 {
                bail!(lw.err(format!("cvt.{} not modeled", mods.join("."))));
            }
            Op::FCvt {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                from,
                to,
                rnd,
                ftz,
                int,
                hi: false,
            }
        }
        // min / max [.relu] .u16x2 / .s16x2 / .s32 (SASS VIMNMX[.U16x2|.S16x2][.RELU]; .relu
        // only signed). ASSUME(N-INT16X2)
        "min" | "max" => {
            let relu = mods.contains(&"relu");
            let ok = matches!(
                mods,
                ["u16x2"] | ["s16x2"] | ["relu", "s16x2"] | ["relu", "s32"]
            );
            if !ok || ops.len() != 3 {
                bail!(lw.err(format!("{base}.{} not modeled", mods.join("."))));
            }
            let b = lw.val(op_n(2)?)?;
            Op::IntMnmx3 {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: b.clone(),
                c: b,
                signed: mods[n - 1] != "u16x2",
                max: base == "max",
                x16: mods[n - 1] != "s32",
                relu,
            }
        }
        // `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [g], [s], bytes` (SASS
        // UBLKRED.G.S.ADD.F32.RN): as the bulk store, each f32 added. ASSUME(N-BULK-RED)
        // `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dst], [src],
        // bytes, [mbar]` (SASS UBLKCP.S.S). ASSUME(S-TMA)
        "cp" if mods[0] == "async" => {
            if ops.len() != 4 {
                bail!(lw.err("cp.async.bulk shared->shared: unexpected operands"));
            }
            let reg = |i: usize, what: &str| -> Result<(Val, u32)> {
                let a = lw.addr(op_n(i)?)?;
                let b = a
                    .base
                    .ok_or_else(|| lw.err(format!("cp.async.bulk {what} must be a register")))?;
                Ok((b, a.off as u32))
            };
            let (dst, off) = reg(0, "destination")?;
            let (src, soff) = reg(1, "source")?;
            let (mbar, mbar_off) = reg(3, "mbarrier")?;
            if soff != 0 {
                bail!(lw.err("cp.async.bulk shared source offset not modeled"));
            }
            Op::TmaLoad {
                dims: 0,
                dst,
                off,
                mbar,
                mbar_off,
                tmap: lw.val(op_n(2)?)?,
                coords: vec![src],
                mask: None,
                pair: false,
                src_smem: true,
            }
        }
        // `cp.reduce.async.bulk.tensor.<N>d.global.shared::cta.add.tile.bulk_group
        // [tmap, {c..}], [src]` (SASS UTMAREDG.<N>D.ADD). ASSUME(N-BULK-RED)
        "cp" if mods.get(3) == Some(&"tensor") => {
            let dims: u8 = mods
                .get(4)
                .and_then(|d| d.strip_suffix('d'))
                .and_then(|n| n.parse().ok())
                .filter(|n| (1..=5).contains(n))
                .ok_or_else(|| lw.err("bad tensor dims"))?;
            if mods[5..] != ["global", "shared::cta", "add", "tile", "bulk_group"] || ops.len() != 2
            {
                bail!(lw.err(format!("cp.{} not modeled", mods.join("."))));
            }
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
            Op::TmaStore {
                dims,
                src: sa
                    .base
                    .ok_or_else(|| lw.err("TMA reduction source must be a register"))?,
                off: sa.off as u32,
                tmap: lw.val(tm)?,
                coords,
                red: true,
            }
        }
        "cp" => {
            if mods
                != [
                    "reduce",
                    "async",
                    "bulk",
                    "global",
                    "shared::cta",
                    "bulk_group",
                    "add",
                    "f32",
                ]
                || ops.len() != 3
            {
                bail!(lw.err(format!("cp.{} not modeled", mods.join("."))));
            }
            let g = lw.addr(op_n(0)?)?;
            let s = lw.addr(op_n(1)?)?;
            if g.off != 0 {
                bail!(lw.err("cp.reduce.async.bulk global address offset not modeled"));
            }
            Op::TmaStore {
                dims: 0,
                src: s
                    .base
                    .ok_or_else(|| lw.err("cp.reduce.async.bulk source must be a register"))?,
                off: s.off as u32,
                tmap: lw.val(op_n(2)?)?,
                coords: vec![
                    g.base
                        .ok_or_else(|| lw.err("global address must be a register"))?,
                ],
                red: true,
            }
        }
        _ => unreachable!("l2::claims"),
    })
}
