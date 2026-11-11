/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Replay samples of the GB300 MUFU sweep (fixtures/mufu/samples.json), written by
//! `ptxv mufu-sweep` from the exhaustive dumps of probes/mufu, checked by `cargo test` without
//! a GPU.

use ptxv_ir::MufuFmt;
use ptxv_ir::MufuFunc;
use serde::Deserialize;
use serde::Serialize;

/// The swept ops: name, function, format, PTX op (Some(ftz)) or bare MUFU (None).
pub const OPS: &[(&str, MufuFunc, MufuFmt, Option<bool>)] = &[
    ("ex2.ftz", MufuFunc::Ex2, MufuFmt::F32, None),
    ("ex2", MufuFunc::Ex2, MufuFmt::F32, Some(false)),
    ("rcp.ftz", MufuFunc::Rcp, MufuFmt::F32, None),
    ("rcp", MufuFunc::Rcp, MufuFmt::F32, Some(false)),
    ("rsq.ftz", MufuFunc::Rsq, MufuFmt::F32, None),
    ("rsq", MufuFunc::Rsq, MufuFmt::F32, Some(false)),
    ("lg2.ftz", MufuFunc::Lg2, MufuFmt::F32, None),
    ("lg2", MufuFunc::Lg2, MufuFmt::F32, Some(false)),
    ("sqrt.ftz", MufuFunc::Sqrt, MufuFmt::F32, None),
    ("sqrt", MufuFunc::Sqrt, MufuFmt::F32, Some(false)),
    ("sin.raw", MufuFunc::Sin, MufuFmt::F32, None),
    ("cos.raw", MufuFunc::Cos, MufuFmt::F32, None),
    ("sin", MufuFunc::Sin, MufuFmt::F32, Some(false)),
    ("cos", MufuFunc::Cos, MufuFmt::F32, Some(false)),
    ("tanh", MufuFunc::Tanh, MufuFmt::F32, None),
    ("rcp64h", MufuFunc::Rcp64h, MufuFmt::F32, None),
    ("rsq64h", MufuFunc::Rsq64h, MufuFmt::F32, None),
    ("ex2.f16", MufuFunc::Ex2, MufuFmt::F16, None),
    ("ex2.bf16", MufuFunc::Ex2, MufuFmt::Bf16, None),
    ("tanh.f16", MufuFunc::Tanh, MufuFmt::F16, None),
    ("tanh.bf16", MufuFunc::Tanh, MufuFmt::Bf16, None),
];

pub fn op(name: &str) -> anyhow::Result<(MufuFunc, MufuFmt, Option<bool>)> {
    OPS.iter()
        .find(|o| o.0 == name)
        .map(|o| (o.1, o.2, o.3))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unknown op {name}; one of {}",
                OPS.iter().map(|o| o.0).collect::<Vec<_>>().join(", ")
            )
        })
}

/// The model's output for input `x`, None if not modeled.
pub fn model(func: MufuFunc, fmt: MufuFmt, ptx: Option<bool>, x: u32) -> Option<u32> {
    match ptx {
        None => crate::mufu::mufu(func, fmt, false, x),
        Some(ftz) => crate::mufu::ptx_approx(func, fmt, ftz, x),
    }
}

/// Replay sample of one op (see `Sample::check`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Sample {
    pub op: String,
    /// Special inputs and their recorded outputs.
    pub specials: Vec<(u32, u32)>,
    /// `random_n` pseudo-random inputs (`random_input(i)`): FNV-1a hash of each block of
    /// `RANDOM_BLOCK` recorded outputs, and the first `RAW` outputs verbatim.
    pub random_n: usize,
    pub random_hashes: Vec<String>,
    pub random_raw: Vec<u32>,
    /// FNV-1a hash of each chunk of `CHUNK` outputs of the whole sweep, in input order.
    pub chunk_hashes: Vec<String>,
}

pub const RANDOM_N: usize = 1 << 20;
pub const RANDOM_BLOCK: usize = 4096;
pub const RAW: usize = 1024;
pub const CHUNK: u64 = 1 << 24;

pub fn random_input(i: u64) -> u32 {
    // splitmix64
    let mut z = i.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) as u32
}

pub fn fnv(words: impl Iterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Special f32 inputs: signed zeros, denormal edges, normal edges, infinities, NaNs, every
/// power of two, and the expansion thresholds of the PTX ops.
pub fn specials(bits16: bool) -> Vec<u32> {
    if bits16 {
        return vec![]; // 16-bit forms are replayed exhaustively
    }
    let mut v = vec![
        0x0000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x0080_0001,
        0x00ff_ffff,
        0x3f7f_ffff,
        0x3f80_0000,
        0x3f80_0001,
        0x3fff_ffff,
        0x7e7f_ffff,
        0x7e80_0000,
        0x7e80_0001,
        0x7eff_ffff,
        0x7f00_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0x7f80_0001,
        0x7fbf_ffff,
        0x7fc0_0000,
        0x7fff_ffff,
        0xc2fc_0000, // -126
        0xc2fb_ffff,
        0xc2fc_0001,
        0x4300_0000, // 128
        0x42ff_ffff,
        0x3e80_0000, // 0.25 (a quarter turn)
        0x3f00_0000,
        0x3f40_0000,
    ];
    for e in 0..255u32 {
        v.push(e << 23);
    }
    let neg: Vec<u32> = v.iter().map(|x| x | 0x8000_0000).collect();
    v.extend(neg);
    v.sort();
    v.dedup();
    v
}

impl Sample {
    /// Compare the model with this sample; Err lists the first differences. With `partial`
    /// (an op modeled on part of its domain), inputs the model does not cover are skipped, and
    /// so are the hashed blocks (which mix both); returns the number of inputs compared.
    pub fn check_with(&self, partial: bool) -> Result<usize, String> {
        let (func, fmt, ptx) = op(&self.op).map_err(|e| e.to_string())?;
        let m = |x: u32| model(func, fmt, ptx, x);
        let mut errs = Vec::new();
        let mut n = 0;
        let raw = self.specials.iter().copied().chain(
            self.random_raw
                .iter()
                .enumerate()
                .map(|(i, &y)| (random_input(i as u64), y)),
        );
        for (x, y) in raw {
            match m(x) {
                Some(z) if z == y => n += 1,
                None if partial => {}
                z => errs.push(format!("{:#010x}: hw {y:#010x} model {z:x?}", x)),
            }
        }
        if partial {
            return if errs.is_empty() {
                Ok(n)
            } else {
                errs.truncate(8);
                Err(format!("{}: {}", self.op, errs.join("; ")))
            };
        }
        let mut out = Vec::with_capacity(RANDOM_BLOCK);
        for (b, h) in self.random_hashes.iter().enumerate() {
            out.clear();
            for i in b * RANDOM_BLOCK..(b + 1) * RANDOM_BLOCK {
                match m(random_input(i as u64)) {
                    Some(z) => out.push(z),
                    None => {
                        return Err(format!(
                            "{}: not modeled ({:#010x})",
                            self.op,
                            random_input(i as u64)
                        ));
                    }
                }
            }
            if format!("{:016x}", fnv(out.iter().copied())) != *h {
                errs.push(format!("random block {b} differs"));
            }
            n += RANDOM_BLOCK;
        }
        if errs.is_empty() {
            Ok(n)
        } else {
            errs.truncate(8);
            Err(format!("{}: {}", self.op, errs.join("; ")))
        }
    }

    /// Replay chunks `chunks` of the whole sweep (2^32 inputs; 2^16 for 16-bit forms).
    pub fn check_full(&self, chunks: std::ops::Range<usize>) -> Result<(), String> {
        let (func, fmt, ptx) = op(&self.op).map_err(|e| e.to_string())?;
        let n = total(fmt);
        let per = CHUNK.min(n);
        let bad: Vec<usize> = std::thread::scope(|s| {
            let hs: Vec<_> = chunks
                .map(|c| {
                    s.spawn(move || {
                        let words = (c as u64 * per..(c as u64 + 1) * per)
                            .map(|x| model(func, fmt, ptx, x as u32).unwrap_or(0xdead_beef));
                        (c, format!("{:016x}", fnv(words)))
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap())
                .filter(|(c, h)| self.chunk_hashes.get(*c) != Some(h))
                .map(|(c, _)| c)
                .collect()
        });
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("{}: chunks {bad:?} differ", self.op))
        }
    }
}

pub fn total(fmt: MufuFmt) -> u64 {
    if fmt == MufuFmt::F32 {
        1 << 32
    } else {
        1 << 16
    }
}
