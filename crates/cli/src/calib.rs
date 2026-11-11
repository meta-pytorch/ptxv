/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Tensor-core arithmetic calibration (`ptxv calib-mma`).
//!
//! Runs a tcgen05 GEMM case on a free GB300 with several input distributions chosen to
//! separate candidate rounding behaviours, then scores every member of the `MmaModel`
//! family bit-for-bit against the hardware output. The accepted model is the one that
//! matches every output of every input set. With f32 A / B buffers the case is a TF32 MMA
//! (probes/tmma: K steps of 8, raw f32 words in the registers), and the input conversion to
//! tf32 (truncate / round to nearest even / round half away) is scored with each model.

use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;
use ptxv_case::Case;
use ptxv_case::f32_to_bf16_rn;
use ptxv_case::f64_to_f16_rn;
use ptxv_sim::fp::MmaModel;
use ptxv_sim::fp::Term;
use serde::Serialize;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn sign(&mut self) -> f64 {
        if self.next() & 1 == 0 { 1.0 } else { -1.0 }
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.unit()).cos()
    }
}

/// An input set: A[M][K] and B[N][K] as f32 bit patterns of the input words (bf16 cases:
/// bf16 values, written as their high halves).
struct Set {
    name: &'static str,
    why: &'static str,
    a: Vec<u32>,
    b: Vec<u32>,
}

/// TF32 input conversion candidates: how the unit reads the low 13 mantissa bits of an f32
/// register word. `None`: bf16 inputs (exact).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Conv {
    None,
    Trunc,
    Rne,
    Rna,
}

impl Conv {
    fn apply(self, x: u32) -> u32 {
        let low = x & 0x1fff;
        match self {
            Conv::None => x,
            Conv::Trunc => x & !0x1fff,
            Conv::Rne => {
                let up = low > 0x1000 || (low == 0x1000 && x & 0x2000 != 0);
                (x & !0x1fff).wrapping_add(if up { 0x2000 } else { 0 })
            }
            Conv::Rna => (x & !0x1fff).wrapping_add(if low >= 0x1000 { 0x2000 } else { 0 }),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Conv::None => "",
            Conv::Trunc => "/in-trunc",
            Conv::Rne => "/in-rne",
            Conv::Rna => "/in-rna",
        }
    }
}

/// Operand format of a calibration case (the A buffer's dtype).
#[derive(Clone, Copy, PartialEq)]
enum Fmt {
    Bf16,
    F16,
    Tf32,
}

fn sets(m: usize, n: usize, k: usize, fmt: Fmt, step: usize) -> Vec<Set> {
    // bf16 / f16 inputs: rounded to the format (as f32 words); tf32: f32 words (every
    // mantissa bit random where the value is).
    let bf = move |x: f64| -> u32 {
        match fmt {
            Fmt::Tf32 => (x as f32).to_bits(),
            Fmt::Bf16 => (f32_to_bf16_rn(x as f32) as u32) << 16,
            // Subnormal f16 values flush to zero: the calibration covers normal inputs.
            Fmt::F16 if x.abs() < 2f64.powi(-14) => (0.0f32.copysign(x as f32)).to_bits(),
            Fmt::F16 => (ptxv_sim::fp::f16_to_f64(f64_to_f16_rn(x)) as f32).to_bits(),
        }
    };
    // f16 normals span 2^-14 .. 2^15: smaller exponent ranges keep every input normal.
    let f16 = fmt == Fmt::F16;
    let (tiny, big) = if f16 { (13, 4) } else { (39, 10) };
    let mut out = Vec::new();
    let mut r = Rng(7);
    let gen = |r: &mut Rng, len: usize, f: &dyn Fn(&mut Rng, usize) -> f64| -> Vec<u32> {
        (0..len).map(|i| bf(f(r, i))).collect()
    };
    out.push(Set {
        name: "normal",
        why: "typical data",
        a: gen(&mut r, m * k, &|r, _| r.normal()),
        b: gen(&mut r, n * k, &|r, _| r.normal()),
    });
    out.push(Set {
        name: "logu12",
        why: "magnitudes 2^[-12,12]: alignment shifts up to 48 bits between products",
        a: gen(&mut r, m * k, &|r, _| {
            r.sign() * 2f64.powf(r.unit() * 24.0 - 12.0)
        }),
        b: gen(&mut r, n * k, &|r, _| {
            r.sign() * 2f64.powf(r.unit() * 24.0 - 12.0)
        }),
    });
    out.push(Set {
        name: "logu30",
        why: "magnitudes 2^[-30,30] (f16: 2^[-14,14]): most small products fall below the kept bits",
        a: gen(&mut r, m * k, &|r, _| {
            r.sign() * 2f64.powf(if f16 { r.unit() * 28.0 - 14.0 } else { r.unit() * 60.0 - 30.0 })
        }),
        b: gen(&mut r, n * k, &|r, _| {
            r.sign() * 2f64.powf(if f16 { r.unit() * 28.0 - 14.0 } else { r.unit() * 60.0 - 30.0 })
        }),
    });
    // Cancellation: pairs (x, -x(1+tiny)) so the dot product is a small residual of big terms.
    out.push(Set {
        name: "cancel",
        why: "big terms cancel; the result depends on the low bits of small terms",
        a: gen(&mut r, m * k, &|r, i| {
            if i % 2 == 0 {
                2f64.powf(r.unit() * 8.0)
            } else {
                2f64.powf(r.unit() * 8.0 - if f16 { 14.0 } else { 20.0 }) * r.sign()
            }
        }),
        b: gen(&mut r, n * k, &|r, i| {
            if i % 2 == 0 {
                r.sign()
            } else {
                r.sign() * 2f64.powf(r.unit() * 4.0)
            }
        }),
    });
    // Probe: one dominant product per instruction plus small ones at controlled distance.
    out.push(Set {
        name: "probe_align",
        why: "1 + fifteen products 2^-e (e varies by row): measures how many bits below the max survive",
        a: (0..m * k).map(|i| if i % step == 0 { bf(1.0) } else { bf(2f64.powi(-((i / k) as i32 % (tiny + 1)))) }).collect(),
        b: (0..n * k).map(|i| if (i / k).is_multiple_of(2) { bf(1.0) } else { bf(if i % step == 0 { 1.0 } else { 1.5 }) }).collect(),
    });
    // Probe: large accumulator from the first K chunk, then tiny products.
    out.push(Set {
        name: "probe_acc",
        why: "first instruction leaves a large accumulator; later instructions add tiny products",
        a: (0..m * k)
            .map(|i| {
                if i % k < step {
                    bf(2f64.powi(big + (i / k) as i32 % 8))
                } else {
                    bf(2f64.powi(-((i / k) as i32 % if f16 { 14 } else { 24 })))
                }
            })
            .collect(),
        b: gen(&mut r, n * k, &|r, _| r.sign() * (1.0 + r.unit())),
    });
    out
}

fn models() -> Vec<MmaModel> {
    let mut v = Vec::new();
    for group in [1u8, 2, 4, 8, 16] {
        for frac in std::iter::once(None).chain((23u8..=32).map(Some)) {
            for final_rz in [false, true] {
                for acc_in_group in [true, false] {
                    for trunc_floor in [false, true] {
                        for exp_from_inputs in [false, true] {
                            if frac.is_none() && (trunc_floor || exp_from_inputs) {
                                continue; // irrelevant without truncation
                            }
                            v.push(MmaModel {
                                group,
                                frac_bits: frac,
                                final_rz,
                                acc_in_group,
                                trunc_floor,
                                exp_from_inputs,
                            });
                        }
                    }
                }
            }
        }
    }
    v
}

/// D = A * B^T through MMA steps of `step` products (16 bf16, 8 tf32), as the kernel does
/// (the first step overwrites D).
#[allow(clippy::too_many_arguments)]
fn gemm(
    model: &MmaModel,
    conv: Conv,
    a: &[u32],
    b: &[u32],
    m: usize,
    n: usize,
    k: usize,
    step: usize,
) -> Vec<u32> {
    let mut d = vec![0u32; m * n];
    let f = |x: u32| f32::from_bits(conv.apply(x)) as f64;
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for kc in 0..k / step {
                let terms: Vec<Term> = (0..step)
                    .map(|j| {
                        let kk = kc * step + j;
                        Term::prod(f(a[mi * k + kk]), f(b[ni * k + kk]), model.exp_from_inputs)
                    })
                    .collect();
                acc = model.apply_terms(if kc == 0 { 0.0 } else { acc }, &terms);
            }
            d[mi * n + ni] = acc.to_bits();
        }
    }
    d
}

#[derive(Serialize)]
struct Score {
    model: String,
    matches: Vec<usize>,
    total: usize,
}

pub fn run(case: &Case, out: &Option<PathBuf>, step: Option<usize>) -> Result<()> {
    let dir = crate::out_dir(case, out)?.join("calib");
    std::fs::create_dir_all(&dir)?;
    let (ba, bb) = (case.buffer("A")?, case.buffer("B")?);
    let (m, k, n) = (
        ba.shape[0] as usize,
        ba.shape[1] as usize,
        bb.shape[0] as usize,
    );
    let fmt = match ba.dtype.as_str() {
        "f32" => Fmt::Tf32,
        "f16" => Fmt::F16,
        _ => Fmt::Bf16,
    };
    let tf32 = fmt == Fmt::Tf32;
    let step = step.unwrap_or(if tf32 { 8 } else { 16 });
    if k % step != 0 {
        bail!("K must be a multiple of {step}");
    }
    let convs: &[Conv] = if tf32 {
        &[Conv::Trunc, Conv::Rne, Conv::Rna]
    } else {
        &[Conv::None]
    };
    // A binary case (probes/hmma) runs its own cubin.
    let front = if case.m.cubin.is_some() {
        "sass"
    } else {
        "ptx"
    };
    let (_, cubin) = crate::load(case, "good", front, &dir)?;
    let free = ptxv_hw::free_gpus()?;
    let Some(&gpu) = free.first() else {
        bail!("no free GPU")
    };
    println!(
        "calib-mma: {} on GPU {gpu}, D[{m}x{n}] = A[{m}x{k}] B[{n}x{k}]^T, {} candidate models",
        case.m.name,
        models().len() * convs.len()
    );
    let exe = std::env::current_exe()?;
    let sets = sets(m, n, k, fmt, step);
    let words = |v: &[u32]| -> Vec<u8> {
        match fmt {
            Fmt::Tf32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Fmt::Bf16 => v
                .iter()
                .flat_map(|x| ((x >> 16) as u16).to_le_bytes())
                .collect(),
            Fmt::F16 => v
                .iter()
                .flat_map(|x| f64_to_f16_rn(f32::from_bits(*x) as f64).to_le_bytes())
                .collect(),
        }
    };
    let mut hw: Vec<Vec<u32>> = Vec::new();
    for s in &sets {
        let pa = dir.join(format!("{}.A.bin", s.name));
        let pb = dir.join(format!("{}.B.bin", s.name));
        let pd = dir.join(format!("{}.D.hw.bin", s.name));
        std::fs::write(&pa, words(&s.a))?;
        std::fs::write(&pb, words(&s.b))?;
        let o = std::process::Command::new(&exe)
            .args([
                "hw-worker",
                case.dir.to_str().unwrap(),
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
            .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
            .output()?;
        if !o.status.success() {
            bail!("hw run failed: {}", String::from_utf8_lossy(&o.stderr));
        }
        let r: serde_json::Value = serde_json::from_slice(&o.stdout)?;
        if r["distinct"].as_array().map(|a| a.len()) != Some(1) {
            bail!("hardware output is not deterministic for set {}", s.name);
        }
        let bytes = std::fs::read(&pd)?;
        hw.push(
            bytes
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        );
        println!("  ran {:<12} ({})", s.name, s.why);
    }
    let mut scores = Vec::new();
    for md in models() {
        for &cv in convs {
            let matches: Vec<usize> = sets
                .iter()
                .zip(&hw)
                .map(|(s, h)| {
                    gemm(&md, cv, &s.a, &s.b, m, n, k, step)
                        .iter()
                        .zip(h)
                        .filter(|(x, y)| x == y)
                        .count()
                })
                .collect();
            scores.push(Score {
                model: format!("{}{}", md.name(), cv.name()),
                matches,
                total: m * n,
            });
        }
    }
    scores.sort_by_key(|s| std::cmp::Reverse(s.matches.iter().sum::<usize>()));
    println!(
        "\n  {:<34} {}",
        "model",
        sets.iter()
            .map(|s| format!("{:>12}", s.name))
            .collect::<String>()
    );
    for s in scores.iter().take(12) {
        println!(
            "  {:<34} {}",
            s.model,
            s.matches
                .iter()
                .map(|x| format!("{:>12}", format!("{x}/{}", s.total)))
                .collect::<String>()
        );
    }
    let perfect: Vec<&Score> = scores
        .iter()
        .filter(|s| s.matches.iter().all(|&x| x == s.total))
        .collect();
    println!(
        "\n  models matching every output of every set: {}",
        if perfect.is_empty() {
            "NONE".into()
        } else {
            perfect
                .iter()
                .map(|s| s.model.clone())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    #[derive(Serialize)]
    struct Report<'a> {
        sets: Vec<(&'a str, &'a str)>,
        scores: &'a [Score],
    }
    std::fs::write(
        dir.join("calib.json"),
        serde_json::to_string_pretty(&Report {
            sets: sets.iter().map(|s| (s.name, s.why)).collect(),
            scores: &scores,
        })?,
    )?;
    println!("  written {}", dir.join("calib.json").display());
    Ok(())
}
