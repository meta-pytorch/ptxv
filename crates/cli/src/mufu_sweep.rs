/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv mufu-sweep`: compare an exhaustive GB300 sweep of one MUFU / approx op (all 2^32 f32
//! inputs, or all 2^16 for the 16-bit forms, as written by probes/mufu/probe) with the
//! simulator's model, measure the hardware's error against the true function, and record the
//! replay sample that `cargo test` checks without a GPU (fixtures/mufu/samples.json).

use std::io::Read;
use std::path::Path;

use anyhow::Result;
use anyhow::bail;
use ptxv_ir::MufuFmt;
use ptxv_ir::MufuFunc;
use ptxv_sim::mufu_sample::*;
use ptxv_sim::num::mufu_true;

/// Error of a hardware output against the true function, in the output's ulps and absolute.
fn err_of(func: MufuFunc, fmt: MufuFmt, ptx: bool, x: u32, y: u32) -> Option<(f64, f64)> {
    // The 64H units' words are f64 high words (their bound is measured in probes/mufu).
    if fmt != MufuFmt::F32 || matches!(func, MufuFunc::Rcp64h | MufuFunc::Rsq64h) {
        return None;
    }
    let xf = f32::from_bits(x) as f64;
    let t = mufu_true(func, ptx, xf);
    let yf = f32::from_bits(y) as f64;
    if !t.is_finite() || !yf.is_finite() || t.abs() < f32::MIN_POSITIVE as f64 {
        return None;
    }
    if t.abs() > f32::MAX as f64 {
        return None;
    }
    let ulp = ptxv_sim::num::ulp_f32(t as f32);
    Some(((yf - t).abs() / ulp, (yf - t).abs()))
}

#[derive(Default, Clone, Copy, Debug)]
struct Stats {
    mismatches: u64,
    first_mismatch: Option<(u32, u32, Option<u32>)>,
    not_modeled: u64,
    max_ulps: (f64, u32),
    /// Over normal inputs only (the domain of the bounds in ASSUMPTIONS.md).
    max_ulps_normal_in: (f64, u32),
    max_abs: (f64, u32),
}

impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.mismatches += o.mismatches;
        self.not_modeled += o.not_modeled;
        if self.first_mismatch.is_none() {
            self.first_mismatch = o.first_mismatch;
        }
        for (a, b) in [
            (&mut self.max_ulps, o.max_ulps),
            (&mut self.max_ulps_normal_in, o.max_ulps_normal_in),
            (&mut self.max_abs, o.max_abs),
        ] {
            if b.0 > a.0 {
                *a = b;
            }
        }
    }
}

fn scan(op: (MufuFunc, MufuFmt, Option<bool>), base: u64, ys: &[u32]) -> Stats {
    let (func, fmt, ptx) = op;
    let mut st = Stats::default();
    for (i, &y) in ys.iter().enumerate() {
        let x = (base + i as u64) as u32;
        let y = if fmt == MufuFmt::F32 { y } else { y & 0xffff };
        match model(func, fmt, ptx, x) {
            Some(z) if z == y => {}
            None => st.not_modeled += 1,
            z => {
                st.mismatches += 1;
                if st.first_mismatch.is_none() {
                    st.first_mismatch = Some((x, y, z));
                }
            }
        }
        if let Some((u, a)) = err_of(func, fmt, ptx.is_some(), x, y) {
            if u > st.max_ulps.0 {
                st.max_ulps = (u, x);
            }
            let e = x >> 23 & 0xff;
            if e != 0 && u > st.max_ulps_normal_in.0 {
                st.max_ulps_normal_in = (u, x);
            }
            if a > st.max_abs.0 {
                st.max_abs = (a, x);
            }
        }
    }
    st
}

pub fn run(name: &str, input: &Path, sample_out: Option<&Path>) -> Result<()> {
    let o = op(name)?;
    let n = total(o.1);
    let mut rd: Box<dyn Read> = if input == Path::new("-") {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(std::fs::File::open(input)?)
    };
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let per = CHUNK.min(n) as usize;
    let mut buf = vec![0u8; per * 4];
    let mut total_st = Stats::default();
    let mut sample = Sample {
        op: name.into(),
        ..Default::default()
    };
    let mut base = 0u64;
    while base < n {
        rd.read_exact(&mut buf)?;
        let ys: Vec<u32> = buf
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .map(|y| if o.1 == MufuFmt::F32 { y } else { y & 0xffff })
            .collect();
        sample
            .chunk_hashes
            .push(format!("{:016x}", fnv(ys.iter().copied())));
        let part = ys.len().div_ceil(threads);
        let sts: Vec<Stats> = std::thread::scope(|s| {
            let hs: Vec<_> = ys
                .chunks(part)
                .enumerate()
                .map(|(j, c)| s.spawn(move || scan(o, base + (j * part) as u64, c)))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for st in &sts {
            total_st.merge(st);
        }
        base += per as u64;
    }
    let show = |(v, x): (f64, u32)| format!("{v:.4} at {x:#010x} ({:e})", f32::from_bits(x));
    let show_abs = |(v, x): (f64, u32)| {
        format!(
            "{:.4} * 2^-24 at {x:#010x} ({:e})",
            v * 2f64.powi(24),
            f32::from_bits(x)
        )
    };
    println!(
        "mufu-sweep {name}: {n} inputs, model mismatches {}{}, not modeled {}",
        total_st.mismatches,
        total_st
            .first_mismatch
            .map(|(x, y, z)| format!(" (first: {x:#010x} hw {y:#010x} model {z:x?})"))
            .unwrap_or_default(),
        total_st.not_modeled
    );
    if o.1 == MufuFmt::F32 {
        println!(
            "  error vs true function (finite normal results): max {} ulp (normal inputs: {}), \
             max abs {}",
            show(total_st.max_ulps),
            show(total_st.max_ulps_normal_in),
            show_abs(total_st.max_abs)
        );
    }
    if let Some(p) = sample_out {
        // The sample needs the hardware outputs for chosen inputs: re-read them by index.
        let get = |x: u32| -> Result<u32> {
            if input == Path::new("-") {
                bail!("--sample needs a dump file, not stdin");
            }
            use std::io::Seek;
            let mut f = std::fs::File::open(input)?;
            f.seek(std::io::SeekFrom::Start(x as u64 * 4))?;
            let mut b = [0u8; 4];
            f.read_exact(&mut b)?;
            let y = u32::from_le_bytes(b);
            Ok(if o.1 == MufuFmt::F32 { y } else { y & 0xffff })
        };
        if o.1 == MufuFmt::F32 {
            for x in specials(false) {
                sample.specials.push((x, get(x)?));
            }
            sample.random_n = RANDOM_N;
            // Sort the reads by offset for locality, then hash in index order.
            let mut idx: Vec<(u32, usize)> =
                (0..RANDOM_N).map(|i| (random_input(i as u64), i)).collect();
            idx.sort();
            let mut ys = vec![0u32; RANDOM_N];
            let mut f = std::io::BufReader::new(std::fs::File::open(input)?);
            let mut pos = 0u64;
            for (x, i) in idx {
                use std::io::Seek;
                let want = x as u64 * 4;
                if want != pos {
                    f.seek(std::io::SeekFrom::Start(want))?;
                }
                let mut b = [0u8; 4];
                f.read_exact(&mut b)?;
                pos = want + 4;
                ys[i] = u32::from_le_bytes(b);
            }
            sample.random_raw = ys[..RAW].to_vec();
            sample.random_hashes = ys
                .chunks(RANDOM_BLOCK)
                .map(|c| format!("{:016x}", fnv(c.iter().copied())))
                .collect();
        }
        // Merge into the samples file.
        let mut all: Vec<Sample> = if p.exists() {
            serde_json::from_str(&std::fs::read_to_string(p)?)?
        } else {
            vec![]
        };
        all.retain(|s| s.op != name);
        all.push(sample);
        all.sort_by_key(|s| OPS.iter().position(|o| o.0 == s.op));
        std::fs::write(p, serde_json::to_string(&all)?)?;
        println!("  sample written to {}", p.display());
    }
    Ok(())
}
