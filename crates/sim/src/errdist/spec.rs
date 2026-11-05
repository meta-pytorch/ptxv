/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The errdist input-distribution spec, version 1 (docs/errdist.md §2).
//!
//! A spec gives, per input buffer of the case (by name), an element distribution, optional
//! hierarchical scales along axes, and a quantization recipe that turns real values into the
//! buffer's codes (and its scale buffer's codes). The same spec drives the analytic tiers
//! (exact code PMFs) and every sampled input (Monte Carlo, simulator and GPU validation), so
//! predictions and observations refer to one distribution.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde::Serialize;

use super::special::Rng;
use super::special::{self};

pub const SPEC_VERSION: u32 = 1;

// ------------------------------------------------------------------------------------ formats

/// Number formats a buffer element or code can have.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize
)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    F32,
    Bf16,
    F16,
    E4m3,
    E5m2,
    E2m1,
    Ue8m0,
    Ue4m3,
}

impl Format {
    pub fn lp(self) -> Option<ptxv_ir::Lp> {
        use ptxv_ir::Lp;
        Some(match self {
            Format::E4m3 => Lp::E4m3,
            Format::E5m2 => Lp::E5m2,
            Format::E2m1 => Lp::E2m1,
            Format::Ue8m0 => Lp::Ue8m0,
            Format::Ue4m3 => Lp::Ue4m3,
            Format::Bf16 => Lp::Bf16,
            Format::F16 => Lp::F16,
            Format::F32 => return None,
        })
    }
    pub fn from_lp(t: ptxv_ir::Lp) -> Option<Format> {
        use ptxv_ir::Lp;
        Some(match t {
            Lp::E4m3 => Format::E4m3,
            Lp::E5m2 => Format::E5m2,
            Lp::E2m1 => Format::E2m1,
            Lp::Ue8m0 => Format::Ue8m0,
            Lp::Ue4m3 => Format::Ue4m3,
            Lp::Bf16 => Format::Bf16,
            Lp::F16 => Format::F16,
            _ => return None,
        })
    }
    /// Explicit mantissa bits.
    pub fn mbits(self) -> i32 {
        match self {
            Format::F32 => 23,
            Format::Bf16 => 7,
            Format::F16 => 10,
            Format::E4m3 | Format::Ue4m3 => 3,
            Format::E5m2 => 2,
            Format::E2m1 => 1,
            Format::Ue8m0 => 0,
        }
    }
    /// Exponent of the smallest normal value.
    pub fn emin(self) -> i32 {
        match self {
            Format::F32 | Format::Bf16 => -126,
            Format::F16 => -14,
            Format::E4m3 | Format::Ue4m3 => -6,
            Format::E5m2 => -14,
            Format::E2m1 => 0,
            Format::Ue8m0 => -127,
        }
    }
    /// Largest finite value.
    pub fn max(self) -> f64 {
        match self {
            Format::F32 => f32::MAX as f64,
            Format::Bf16 => f32::from_bits(0x7f7f_0000) as f64,
            Format::F16 => 65504.0,
            Format::E4m3 | Format::Ue4m3 => 448.0,
            Format::E5m2 => 57344.0,
            Format::E2m1 => 6.0,
            Format::Ue8m0 => 2f64.powi(127),
        }
    }
    pub fn signed(self) -> bool {
        !matches!(self, Format::Ue8m0 | Format::Ue4m3)
    }
    /// Has an infinity (overflow without saturation gives it).
    pub fn has_inf(self) -> bool {
        matches!(
            self,
            Format::F32 | Format::Bf16 | Format::F16 | Format::E5m2
        )
    }
    /// Bytes per element in a buffer of this format (e2m1: two per byte, `None`).
    pub fn bytes(self) -> Option<usize> {
        match self {
            Format::F32 => Some(4),
            Format::Bf16 | Format::F16 => Some(2),
            Format::E2m1 => None,
            _ => Some(1),
        }
    }

    /// Round to nearest even onto this format's grid (subnormals included). Returns the
    /// rounded value and whether it overflowed (then: +-max with `sat`, else +-inf).
    pub fn round(self, x: f64, sat: bool) -> (f64, bool) {
        if x == 0.0 || x.is_nan() {
            return (if self.signed() { x } else { x.abs() }, x.is_nan());
        }
        let x = if self.signed() { x } else { x.abs() };
        if self == Format::Ue8m0 {
            // Powers of two only: round the exponent to nearest (ties up) on a log scale is not
            // a recipe anyone uses; ue8m0 values come from `ScaleRule`.
            let e = x.log2().round().clamp(-127.0, 127.0);
            return (2f64.powi(e as i32), false);
        }
        let mb = self.mbits();
        let e = exp_floor(x).max(self.emin());
        let q = 2f64.powi(e - mb);
        let r = (x / q).round_ties_even() * q;
        let max = self.max();
        // The overflow threshold of RNE: max + half its spacing (ties go to the even, larger
        // value, i.e. overflow).
        let half = 2f64.powi(exp_floor(max) - mb - 1);
        if r.abs() > max || (x.abs() >= max + half) {
            let o = if sat || !self.has_inf() {
                max
            } else {
                f64::INFINITY
            };
            return (o.copysign(x), true);
        }
        (r, false)
    }

    /// The code of a value on this format's grid (narrow formats; f32 / bf16 / f16 bits).
    pub fn code(self, v: f64) -> u32 {
        match self {
            Format::F32 => (v as f32).to_bits(),
            Format::Bf16 => (v as f32).to_bits() >> 16,
            Format::F16 => crate::fp::f32_to_f16_rn((v as f32).to_bits()) as u32,
            Format::Ue8m0 => {
                if v == 0.0 {
                    0
                } else {
                    (exp_floor(v) + 127).clamp(0, 254) as u32
                }
            }
            _ => {
                let mb = self.mbits();
                let bias = 1 - self.emin();
                let neg = v.is_sign_negative() && self.signed();
                let a = v.abs();
                let mag = if a == 0.0 {
                    0
                } else {
                    let e = exp_floor(a).max(self.emin());
                    let m = (a / 2f64.powi(e - mb)) as u32; // exact on the grid
                    if m < 1 << mb {
                        m // subnormal
                    } else {
                        (((e + bias) as u32) << mb) | (m - (1 << mb))
                    }
                };
                let w = self.width();
                mag | if neg { 1 << (w - 1) } else { 0 }
            }
        }
    }

    /// The value of a code.
    pub fn value(self, c: u32) -> f64 {
        match self {
            Format::F32 => f32::from_bits(c) as f64,
            Format::Bf16 => f32::from_bits(c << 16) as f64,
            Format::F16 => crate::fp::f16_to_f64(c as u16),
            _ => crate::lowp::value(c, self.lp().unwrap()),
        }
    }

    /// Code width in bits.
    pub fn width(self) -> u32 {
        match self {
            Format::F32 => 32,
            Format::Bf16 | Format::F16 => 16,
            _ => crate::lowp::width(self.lp().unwrap()),
        }
    }

    pub fn parse(s: &str) -> Option<Format> {
        serde_json::from_value(serde_json::Value::String(s.to_lowercase())).ok()
    }
}

/// floor(log2 |x|) for finite nonzero x.
pub fn exp_floor(x: f64) -> i32 {
    let b = x.abs().to_bits();
    let e = ((b >> 52) & 0x7ff) as i32;
    if e == 0 {
        -1074 + (63 - (b & ((1 << 52) - 1)).leading_zeros() as i32)
    } else {
        e - 1023
    }
}

// ------------------------------------------------------------------------------ distributions

/// A scalar distribution. JSON: `{"kind": "normal", "mean": 0, "std": 1}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Dist {
    Normal {
        #[serde(default)]
        mean: f64,
        #[serde(default = "one")]
        std: f64,
    },
    Uniform {
        lo: f64,
        hi: f64,
    },
    Laplace {
        #[serde(default)]
        mean: f64,
        #[serde(default = "one")]
        scale: f64,
    },
    StudentT {
        df: f64,
        #[serde(default)]
        mean: f64,
        #[serde(default = "one")]
        scale: f64,
    },
    /// exp(N(mu, sigma)), with a random sign when `signed`.
    LogNormal {
        #[serde(default)]
        mu: f64,
        #[serde(default = "one")]
        sigma: f64,
        #[serde(default)]
        signed: bool,
    },
    Mixture {
        components: Vec<Component>,
    },
    /// An explicit PMF over values (uniform when `probs` is absent).
    Discrete {
        values: Vec<f64>,
        #[serde(default)]
        probs: Option<Vec<f64>>,
    },
    /// Uniform over the integers lo..=hi.
    Integers {
        lo: i64,
        hi: i64,
    },
    Const {
        value: f64,
    },
    /// The empirical distribution of a captured tensor (i.i.d. resampling of its values).
    Empirical {
        file: String,
        #[serde(default = "f32_fmt")]
        dtype: Format,
    },
    /// A piecewise-uniform density: `counts[i]` over [edges[i], edges[i+1]). Inline or from a
    /// JSON file `{"edges": [...], "counts": [...]}`.
    Histogram {
        #[serde(default)]
        edges: Vec<f64>,
        #[serde(default)]
        counts: Vec<f64>,
        #[serde(default)]
        file: Option<String>,
    },
    /// Codes of a format, given directly (for `quant: codes`): all finite codes uniformly, a
    /// code range lo..=hi, or an explicit list with optional probabilities.
    Codes {
        format: Format,
        #[serde(default)]
        range: Option<[u32; 2]>,
        #[serde(default)]
        values: Option<Vec<u32>>,
        #[serde(default)]
        probs: Option<Vec<f64>>,
    },
    /// a * X + b.
    Affine {
        #[serde(default = "one")]
        a: f64,
        #[serde(default)]
        b: f64,
        dist: Box<Dist>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub weight: f64,
    pub dist: Dist,
}

fn one() -> f64 {
    1.0
}
fn f32_fmt() -> Format {
    Format::F32
}

impl Dist {
    /// Resolve file references (relative to `base`) into inline data; check parameters.
    pub fn resolve(&mut self, base: &std::path::Path) -> Result<(), String> {
        match self {
            Dist::Normal { std, .. } if *std <= 0.0 => return Err("normal: std must be > 0".into()),
            Dist::Uniform { lo, hi } if lo.partial_cmp(&hi) != Some(std::cmp::Ordering::Less) => {
                return Err("uniform: need lo < hi".into());
            }
            Dist::Laplace { scale, .. } if *scale <= 0.0 => return Err("laplace: scale > 0".into()),
            Dist::StudentT { df, scale, .. } if *df <= 0.0 || *scale <= 0.0 => {
                return Err("student_t: df > 0 and scale > 0".into());
            }
            Dist::Mixture { components } => {
                if components.is_empty() || components.iter().any(|c| c.weight < 0.0) {
                    return Err("mixture: nonempty, weights >= 0".into());
                }
                for c in components.iter_mut() {
                    c.dist.resolve(base)?;
                }
            }
            Dist::Discrete { values, probs } => {
                if values.is_empty() {
                    return Err("discrete: no values".into());
                }
                if let Some(p) = probs {
                    if p.len() != values.len() || p.iter().any(|x| *x < 0.0) {
                        return Err("discrete: probs must match values, >= 0".into());
                    }
                }
            }
            Dist::Integers { lo, hi } if *lo > *hi => return Err("integers: lo > hi".into()),
            Dist::Empirical { file, dtype } => {
                let p = base.join(&*file);
                let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                let vals: Vec<f64> = match dtype {
                    Format::F32 => b
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                        .collect(),
                    Format::Bf16 | Format::F16 => b
                        .chunks_exact(2)
                        .map(|c| dtype.value(u16::from_le_bytes([c[0], c[1]]) as u32))
                        .collect(),
                    f => b.iter().map(|&c| f.value(c as u32)).collect(),
                };
                let vals: Vec<f64> = vals.into_iter().filter(|v| v.is_finite()).collect();
                if vals.is_empty() {
                    return Err(format!("{}: no finite values", p.display()));
                }
                *self = Dist::Discrete {
                    values: vals,
                    probs: None,
                };
            }
            Dist::Histogram {
                edges,
                counts,
                file,
            } => {
                if let Some(f) = file.take() {
                    let p = base.join(&f);
                    let s =
                        std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
                    let v: serde_json::Value =
                        serde_json::from_str(&s).map_err(|e| e.to_string())?;
                    *edges =
                        serde_json::from_value(v["edges"].clone()).map_err(|e| e.to_string())?;
                    *counts =
                        serde_json::from_value(v["counts"].clone()).map_err(|e| e.to_string())?;
                }
                if edges.len() != counts.len() + 1
                    || counts.is_empty()
                    || edges
                        .windows(2)
                        .any(|w| w[0].partial_cmp(&w[1]) != Some(std::cmp::Ordering::Less))
                    || counts.iter().any(|c| *c < 0.0)
                    || counts.iter().sum::<f64>() <= 0.0
                {
                    return Err("histogram: need increasing edges, counts.len() + 1 edges".into());
                }
            }
            Dist::Codes {
                format,
                range,
                values,
                probs,
            } => {
                if let (Some(v), Some(p)) = (values.as_ref(), probs.as_ref()) {
                    if v.len() != p.len() {
                        return Err("codes: probs must match values".into());
                    }
                }
                if values.is_none() {
                    let (lo, hi) = match range {
                        Some([a, b]) => (*a, *b),
                        None => (0, (1u32 << format.width()) - 1),
                    };
                    let v: Vec<u32> = (lo..=hi).filter(|&c| format.value(c).is_finite()).collect();
                    if v.is_empty() {
                        return Err("codes: no finite code in range".into());
                    }
                    *values = Some(v);
                    *range = None;
                }
            }
            Dist::Affine { dist, .. } => dist.resolve(base)?,
            _ => {}
        }
        Ok(())
    }

    pub fn sample(&self, r: &mut Rng) -> f64 {
        match self {
            Dist::Normal { mean, std } => mean + std * r.normal(),
            Dist::Uniform { lo, hi } => lo + (hi - lo) * r.unit(),
            Dist::Laplace { mean, scale } => {
                let u = r.open() - 0.5;
                mean - scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
            }
            Dist::StudentT { df, mean, scale } => {
                let z = r.normal();
                let v = 2.0 * r.gamma(df / 2.0);
                mean + scale * z / (v / df).sqrt()
            }
            Dist::LogNormal { mu, sigma, signed } => {
                let x = (mu + sigma * r.normal()).exp();
                if *signed && r.next_u64() & 1 == 1 {
                    -x
                } else {
                    x
                }
            }
            Dist::Mixture { components } => {
                let tot: f64 = components.iter().map(|c| c.weight).sum();
                let mut u = r.unit() * tot;
                for c in components {
                    if u < c.weight {
                        return c.dist.sample(r);
                    }
                    u -= c.weight;
                }
                components.last().unwrap().dist.sample(r)
            }
            Dist::Discrete { values, probs } => values[pick(r, values.len(), probs.as_deref())],
            Dist::Integers { lo, hi } => (*lo + r.below((hi - lo + 1) as u64) as i64) as f64,
            Dist::Const { value } => *value,
            Dist::Empirical { .. } => panic!("unresolved empirical distribution"),
            Dist::Histogram { edges, counts, .. } => {
                let i = pick(r, counts.len(), Some(counts));
                edges[i] + (edges[i + 1] - edges[i]) * r.unit()
            }
            Dist::Codes { values, probs, .. } => {
                let v = values.as_ref().expect("unresolved codes");
                v[pick(r, v.len(), probs.as_deref())] as f64
            }
            Dist::Affine { a, b, dist } => a * dist.sample(r) + b,
        }
    }

    /// P(X <= x) (`incl`) or P(X < x).
    pub fn cdf(&self, x: f64, incl: bool) -> f64 {
        match self {
            Dist::Normal { mean, std } => special::phi((x - mean) / std),
            Dist::Uniform { lo, hi } => ((x - lo) / (hi - lo)).clamp(0.0, 1.0),
            Dist::Laplace { mean, scale } => {
                let z = (x - mean) / scale;
                if z < 0.0 {
                    0.5 * z.exp()
                } else {
                    1.0 - 0.5 * (-z).exp()
                }
            }
            Dist::StudentT { df, mean, scale } => special::student_t_cdf((x - mean) / scale, *df),
            Dist::LogNormal { mu, sigma, signed } => {
                let pos = |t: f64| {
                    if t <= 0.0 {
                        0.0
                    } else {
                        special::phi((t.ln() - mu) / sigma)
                    }
                };
                if *signed {
                    if x >= 0.0 {
                        0.5 + 0.5 * pos(x)
                    } else {
                        0.5 - 0.5 * pos(-x)
                    }
                } else {
                    pos(x)
                }
            }
            Dist::Mixture { components } => {
                let tot: f64 = components.iter().map(|c| c.weight).sum();
                components
                    .iter()
                    .map(|c| c.weight * c.dist.cdf(x, incl))
                    .sum::<f64>()
                    / tot
            }
            Dist::Discrete { values, probs } => {
                let n = values.len() as f64;
                let tot: f64 = probs.as_ref().map_or(n, |p| p.iter().sum());
                values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| if incl { **v <= x } else { **v < x })
                    .map(|(i, _)| probs.as_ref().map_or(1.0, |p| p[i]))
                    .sum::<f64>()
                    / tot
            }
            Dist::Integers { lo, hi } => {
                let n = (hi - lo + 1) as f64;
                let k = if incl { x.floor() } else { x.ceil() - 1.0 };
                ((k - *lo as f64 + 1.0) / n).clamp(0.0, 1.0)
            }
            Dist::Const { value } => {
                if (incl && *value <= x) || *value < x {
                    1.0
                } else {
                    0.0
                }
            }
            Dist::Empirical { .. } => panic!("unresolved empirical distribution"),
            Dist::Histogram { edges, counts, .. } => {
                let tot: f64 = counts.iter().sum();
                let mut s = 0.0;
                for (i, c) in counts.iter().enumerate() {
                    if x >= edges[i + 1] {
                        s += c;
                    } else if x > edges[i] {
                        s += c * (x - edges[i]) / (edges[i + 1] - edges[i]);
                    }
                }
                s / tot
            }
            Dist::Codes { values, probs, .. } => {
                let v = values.as_ref().expect("unresolved codes");
                let tot: f64 = probs.as_ref().map_or(v.len() as f64, |p| p.iter().sum());
                v.iter()
                    .enumerate()
                    .filter(|(_, c)| {
                        if incl {
                            (**c as f64) <= x
                        } else {
                            (**c as f64) < x
                        }
                    })
                    .map(|(i, _)| probs.as_ref().map_or(1.0, |p| p[i]))
                    .sum::<f64>()
                    / tot
            }
            Dist::Affine { a, b, dist } => {
                if *a > 0.0 {
                    dist.cdf((x - b) / a, incl)
                } else if *a < 0.0 {
                    1.0 - dist.cdf((x - b) / a, !incl)
                } else if (incl && *b <= x) || *b < x {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// P(lo <~ X <~ hi) with the given endpoint inclusions.
    pub fn mass(&self, lo: f64, lo_incl: bool, hi: f64, hi_incl: bool) -> f64 {
        (self.cdf(hi, hi_incl) - self.cdf(lo, !lo_incl)).max(0.0)
    }

    /// The atoms (value, probability) of a purely discrete distribution, None otherwise.
    pub fn atoms(&self) -> Option<Vec<(f64, f64)>> {
        let norm = |v: Vec<(f64, f64)>| {
            let t: f64 = v.iter().map(|x| x.1).sum();
            v.into_iter().map(|(a, p)| (a, p / t)).collect::<Vec<_>>()
        };
        match self {
            Dist::Discrete { values, probs } => Some(norm(
                values
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (*v, probs.as_ref().map_or(1.0, |p| p[i])))
                    .collect(),
            )),
            Dist::Integers { lo, hi } => Some(norm((*lo..=*hi).map(|k| (k as f64, 1.0)).collect())),
            Dist::Const { value } => Some(vec![(*value, 1.0)]),
            Dist::Codes { values, probs, .. } => {
                let v = values.as_ref()?;
                Some(norm(
                    v.iter()
                        .enumerate()
                        .map(|(i, c)| (*c as f64, probs.as_ref().map_or(1.0, |p| p[i])))
                        .collect(),
                ))
            }
            Dist::Mixture { components } => {
                let tot: f64 = components.iter().map(|c| c.weight).sum();
                let mut m: BTreeMap<u64, (f64, f64)> = BTreeMap::new();
                for c in components {
                    for (v, p) in c.dist.atoms()? {
                        let e = m.entry(v.to_bits()).or_insert((v, 0.0));
                        e.1 += p * c.weight / tot;
                    }
                }
                Some(m.into_values().collect())
            }
            Dist::Affine { a, b, dist } => Some(
                dist.atoms()?
                    .into_iter()
                    .map(|(v, p)| (a * v + b, p))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// E|X| (numerically for the continuous families), for scale summaries.
    pub fn std_guess(&self) -> f64 {
        let mut r = Rng::new(12345);
        let n = 4096;
        let (mut s, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let x = self.sample(&mut r);
            s += x;
            s2 += x * x;
        }
        (s2 / n as f64 - (s / n as f64).powi(2)).max(0.0).sqrt()
    }
}

fn pick(r: &mut Rng, n: usize, probs: Option<&[f64]>) -> usize {
    match probs {
        None => r.below(n as u64) as usize,
        Some(p) => {
            let tot: f64 = p.iter().sum();
            let mut u = r.unit() * tot;
            for (i, w) in p.iter().enumerate() {
                if u < *w {
                    return i;
                }
                u -= w;
            }
            n - 1
        }
    }
}

// ----------------------------------------------------------------------------- tensor specs

/// One input tensor (a buffer of the case, by name).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorSpec {
    /// Element distribution (before scales and quantization).
    pub dist: Dist,
    /// Hierarchical multipliers: for each, one draw per index along `axis` (per input sample),
    /// multiplying every element with that index.
    #[serde(default)]
    pub scales: Vec<AxisScale>,
    /// Outlier channels: each index along `axis` is an outlier with probability `prob`, its
    /// elements multiplied by `factor` (sugar for a two-point mixture in `scales`).
    #[serde(default)]
    pub outliers: Option<Outliers>,
    /// Real values -> buffer codes. Default: round to nearest even to the buffer's dtype.
    #[serde(default)]
    pub quant: Quant,
    /// Logical shape (row-major) the axes refer to. Default: the buffer's shape (e2m1 codes:
    /// last dimension doubled, two codes per byte).
    #[serde(default)]
    pub shape: Option<Vec<u64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AxisScale {
    pub axis: usize,
    pub dist: Dist,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outliers {
    pub axis: usize,
    pub prob: f64,
    pub factor: f64,
}

/// How real values become the buffer's contents.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Quant {
    /// Round to nearest even to the buffer's float dtype (f32, bf16, f16); overflow is
    /// out-of-domain mass.
    #[default]
    Round,
    /// The distribution draws codes directly (`codes`, `integers`, `discrete` over code
    /// numbers) of format `format` (default: the format the kernel reads the buffer as).
    Codes {
        #[serde(default)]
        format: Option<Format>,
    },
    /// Round to nearest even to a narrow format, saturating (`cvt.rn.satfinite`), no scale.
    Cast { format: Format },
    /// OCP microscaling: blocks of `block` elements along the last logical axis share one
    /// ue8m0 scale stored in buffer `scales`; elements are RNE(x / scale), saturating.
    Mx {
        elem: Format,
        #[serde(default = "block32")]
        block: usize,
        scales: String,
        #[serde(default)]
        scale_rule: ScaleRule,
    },
    /// NVFP4: e2m1 elements, blocks of 16 with a ue4m3 scale RN(amax_b / (6 gs)) in buffer
    /// `scales`; gs = amax_tensor / (448 * 6) when `global_scale` (else 1). The kernel's
    /// output is in units of gs_A gs_B (the global scale is outside the MMA).
    Nvfp4 {
        #[serde(default = "block16")]
        block: usize,
        scales: String,
        #[serde(default)]
        global_scale: bool,
    },
}

fn block32() -> usize {
    32
}
fn block16() -> usize {
    16
}

/// How an MX block's ue8m0 scale follows from its absolute maximum.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScaleRule {
    /// 2^ceil(log2(amax / elem_max)): amax / scale <= elem_max, no saturation (cuBLAS /
    /// `cvt.rp.ue8m0` recipe).
    #[default]
    Ceil,
    /// OCP MX v1.0: 2^(floor(log2 amax) - emax_elem); the largest elements may saturate.
    Floor,
}

impl ScaleRule {
    /// The ue8m0 exponent of a block with absolute maximum `amax` (0: the smallest scale).
    pub fn exponent(self, amax: f64, elem: Format) -> i32 {
        if amax == 0.0 || !amax.is_finite() {
            return -127;
        }
        let e = match self {
            ScaleRule::Ceil => {
                let r = amax / elem.max();
                let f = exp_floor(r);
                if r == 2f64.powi(f) { f } else { f + 1 }
            }
            ScaleRule::Floor => exp_floor(amax) - exp_floor(elem.max()),
        };
        e.clamp(-127, 127)
    }
}

/// Engine settings a spec may carry (CLI flags override them).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Engine {
    /// Monte Carlo inputs (Tier 3) through the extracted numerics DAG.
    #[serde(default)]
    pub mc_inputs: Option<usize>,
    /// Validation: inputs run through the simulator.
    #[serde(default)]
    pub sim_trials: Option<usize>,
    /// Validation: inputs run on GB300 (0: none).
    #[serde(default)]
    pub gpu_trials: Option<usize>,
    /// Report every output's summary, not only the aggregate.
    #[serde(default)]
    pub per_output: bool,
}

/// An input-distribution spec.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub version: u32,
    #[serde(default)]
    pub seed: u64,
    /// Per input buffer (by the case's buffer name). A scale buffer named by a recipe gets no
    /// entry of its own.
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorSpec>,
    /// For input buffers without an entry.
    #[serde(default)]
    pub default: Option<TensorSpec>,
    #[serde(default)]
    pub engine: Engine,
    /// Free-form label echoed in the report.
    #[serde(default)]
    pub name: Option<String>,
}

impl Spec {
    /// Parse JSON (or a serde_json value converted from TOML) and resolve file references
    /// relative to `base`.
    pub fn from_value(v: serde_json::Value, base: &std::path::Path) -> Result<Spec, String> {
        let mut s: Spec = serde_json::from_value(v).map_err(|e| format!("spec: {e}"))?;
        if s.version != SPEC_VERSION {
            return Err(format!(
                "spec version {} (this ptxv reads version {SPEC_VERSION})",
                s.version
            ));
        }
        for (n, t) in s.tensors.iter_mut() {
            t.dist
                .resolve(base)
                .map_err(|e| format!("tensor {n}: {e}"))?;
            for a in t.scales.iter_mut() {
                a.dist
                    .resolve(base)
                    .map_err(|e| format!("tensor {n} scale: {e}"))?;
            }
            if let Some(o) = &t.outliers {
                if !(0.0..=1.0).contains(&o.prob) {
                    return Err(format!("tensor {n}: outliers.prob in [0, 1]"));
                }
            }
        }
        if let Some(t) = s.default.as_mut() {
            t.dist.resolve(base)?;
        }
        Ok(s)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors.get(name).or(self.default.as_ref())
    }

    /// Buffers some recipe fills with scales.
    pub fn scale_buffers(&self) -> Vec<(String, String)> {
        self.tensors
            .iter()
            .filter_map(|(n, t)| match &t.quant {
                Quant::Mx { scales, .. } | Quant::Nvfp4 { scales, .. } => {
                    Some((scales.clone(), n.clone()))
                }
                _ => None,
            })
            .collect()
    }
}

// ------------------------------------------------------------------------------ sampling

/// What the sampler needs to know about one input buffer.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Layout {
    pub name: String,
    /// The case dtype ("f32", "bf16", "f16", "u8", "u16").
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Bytes in the buffer.
    pub bytes: usize,
    /// The format the kernel reads its elements as (byte buffers: from the numerics leaves).
    pub fmt: Format,
    /// Scale blocks of a code buffer: (scale byte, code element indices), derived from which
    /// scale leaf each code leaf is multiplied with in the tensor-core steps.
    pub blocks: Option<Vec<(u64, Vec<u64>)>>,
}

impl Layout {
    /// Logical elements (e2m1: two per byte).
    pub fn elems(&self) -> usize {
        match self.fmt.bytes() {
            None => self.bytes * 2,
            Some(b) if matches!(self.dtype.as_str(), "u8" | "u16") => self.bytes / b,
            Some(_) => self.shape.iter().product::<u64>() as usize,
        }
    }
    fn logical_shape(&self, t: &TensorSpec) -> Vec<u64> {
        if let Some(s) = &t.shape {
            return s.clone();
        }
        let mut s = self.shape.clone();
        if self.fmt == Format::E2m1 {
            if let Some(l) = s.last_mut() {
                *l *= 2;
            }
        }
        if s.iter().product::<u64>() as usize != self.elems() {
            s = vec![self.elems() as u64];
        }
        s
    }
}

/// Counts of sampled events outside the modeled domain.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Ood {
    /// Elements that overflowed their format (saturated or infinite).
    pub overflow: u64,
    /// Elements whose value is NaN or infinite.
    pub nonfinite: u64,
    /// Scale codes that are zero (ue4m3) or NaN.
    pub bad_scale: u64,
    /// Elements sampled.
    pub elements: u64,
}

/// One sampled input: buffer bytes by name, plus the real values before quantization.
pub struct Sample {
    pub bytes: BTreeMap<String, Vec<u8>>,
    pub ood: Ood,
}

fn put_code(buf: &mut [u8], fmt: Format, dtype: &str, i: usize, code: u32) {
    match fmt {
        Format::E2m1 => {
            let b = &mut buf[i / 2];
            let sh = 4 * (i % 2);
            *b = (*b & !(0xf << sh)) | (((code & 0xf) as u8) << sh);
        }
        Format::F32 => buf[4 * i..4 * i + 4].copy_from_slice(&code.to_le_bytes()),
        Format::Bf16 | Format::F16 => {
            buf[2 * i..2 * i + 2].copy_from_slice(&(code as u16).to_le_bytes())
        }
        _ => {
            let _ = dtype;
            buf[i] = code as u8
        }
    }
}

/// The real values of one tensor for one sample (element distribution x axis scales).
fn real_values(t: &TensorSpec, n: usize, shape: &[u64], r: &mut Rng) -> Vec<f64> {
    let mut mult = vec![1.0f64; n];
    let strides: Vec<u64> = (0..shape.len())
        .map(|a| shape[a + 1..].iter().product())
        .collect();
    let mut axes: Vec<(usize, Vec<f64>)> = Vec::new();
    for s in &t.scales {
        let len = *shape.get(s.axis).unwrap_or(&1) as usize;
        axes.push((s.axis, (0..len).map(|_| s.dist.sample(r)).collect()));
    }
    if let Some(o) = &t.outliers {
        let len = *shape.get(o.axis).unwrap_or(&1) as usize;
        axes.push((
            o.axis,
            (0..len)
                .map(|_| if r.unit() < o.prob { o.factor } else { 1.0 })
                .collect(),
        ));
    }
    for (ax, v) in &axes {
        if *ax >= shape.len() {
            continue;
        }
        for (i, m) in mult.iter_mut().enumerate() {
            let idx = (i as u64 / strides[*ax]) % shape[*ax];
            *m *= v[idx as usize];
        }
    }
    (0..n).map(|i| mult[i] * t.dist.sample(r)).collect()
}

/// Draw input sample `idx` for every buffer in `layouts`.
// ASSUME(N-ERRDIST-IID): the sampler draws exactly the spec's law.
pub fn sample(spec: &Spec, layouts: &[Layout], idx: u64) -> Result<Sample, String> {
    let mut out = BTreeMap::new();
    let mut ood = Ood::default();
    let scale_bufs: BTreeMap<String, String> = spec.scale_buffers().into_iter().collect();
    let by_name: BTreeMap<&str, &Layout> = layouts.iter().map(|l| (l.name.as_str(), l)).collect();
    for l in layouts {
        if scale_bufs.contains_key(&l.name) {
            continue; // filled by its recipe
        }
        let t = spec
            .tensor(&l.name)
            .ok_or_else(|| format!("no distribution for input buffer {}", l.name))?;
        let mut r = Rng::stream(spec.seed, idx, &l.name);
        let n = l.elems();
        let shape = l.logical_shape(t);
        let x = real_values(t, n, &shape, &mut r);
        let mut buf = vec![0u8; l.bytes];
        ood.elements += n as u64;
        match &t.quant {
            Quant::Round => {
                let f = match l.dtype.as_str() {
                    "f32" => Format::F32,
                    "bf16" => Format::Bf16,
                    "f16" => Format::F16,
                    d => {
                        return Err(format!(
                            "{}: quant round needs a float buffer, not {d}",
                            l.name
                        ));
                    }
                };
                for (i, v) in x.iter().enumerate() {
                    let (q, of) = f.round(*v, false);
                    ood.overflow += of as u64;
                    ood.nonfinite += !q.is_finite() as u64;
                    put_code(&mut buf, f, &l.dtype, i, f.code(q));
                }
            }
            Quant::Codes { format } => {
                let f = format.unwrap_or(l.fmt);
                for (i, v) in x.iter().enumerate() {
                    if *v < 0.0 || v.fract() != 0.0 || *v >= 2f64.powi(f.width() as i32) {
                        return Err(format!("{}: quant codes: {v} is not a {f:?} code", l.name));
                    }
                    ood.nonfinite += !f.value(*v as u32).is_finite() as u64;
                    put_code(&mut buf, f, &l.dtype, i, *v as u32);
                }
            }
            Quant::Cast { format } => {
                for (i, v) in x.iter().enumerate() {
                    let (q, of) = format.round(*v, true);
                    ood.overflow += of as u64;
                    put_code(&mut buf, *format, &l.dtype, i, format.code(q));
                }
            }
            Quant::Mx {
                elem,
                block,
                scales,
                scale_rule,
            } => {
                let sl = by_name
                    .get(scales.as_str())
                    .ok_or_else(|| format!("{}: scale buffer {scales} is not an input", l.name))?;
                let mut sbuf = vec![0u8; sl.bytes];
                for (sbyte, idxs) in blocks_of(l, *block)? {
                    let amax = idxs
                        .iter()
                        .map(|&i| x[i as usize].abs())
                        .fold(0.0, f64::max);
                    let e = scale_rule.exponent(amax, *elem);
                    sbuf[sbyte as usize] = (e + 127) as u8;
                    let s = 2f64.powi(e);
                    for &i in &idxs {
                        let (q, of) = elem.round(x[i as usize] / s, true);
                        ood.overflow += of as u64;
                        put_code(&mut buf, *elem, &l.dtype, i as usize, elem.code(q));
                    }
                }
                out.insert(scales.clone(), sbuf);
            }
            Quant::Nvfp4 {
                block,
                scales,
                global_scale,
            } => {
                let sl = by_name
                    .get(scales.as_str())
                    .ok_or_else(|| format!("{}: scale buffer {scales} is not an input", l.name))?;
                let mut sbuf = vec![0u8; sl.bytes];
                let gs = if *global_scale {
                    let a = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    if a > 0.0 {
                        ((a / (448.0 * 6.0)) as f32) as f64
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                for (sbyte, idxs) in blocks_of(l, *block)? {
                    let amax = idxs
                        .iter()
                        .map(|&i| x[i as usize].abs())
                        .fold(0.0, f64::max);
                    let (s, of) = Format::Ue4m3.round(amax / (6.0 * gs), true);
                    ood.overflow += of as u64;
                    let sc = Format::Ue4m3.code(s);
                    ood.bad_scale += (sc == 0) as u64;
                    sbuf[sbyte as usize] = sc as u8;
                    let div = if s == 0.0 { 1.0 } else { s * gs };
                    for &i in &idxs {
                        let (q, of) = Format::E2m1.round(x[i as usize] / div, true);
                        ood.overflow += of as u64;
                        put_code(
                            &mut buf,
                            Format::E2m1,
                            &l.dtype,
                            i as usize,
                            Format::E2m1.code(q),
                        );
                    }
                }
                out.insert(scales.clone(), sbuf);
            }
        }
        out.insert(l.name.clone(), buf);
    }
    Ok(Sample { bytes: out, ood })
}

/// A code buffer's scale blocks: from the layout (numerics leaves), else contiguous runs of
/// `block` elements with scale byte = run index.
fn blocks_of(l: &Layout, block: usize) -> Result<Vec<(u64, Vec<u64>)>, String> {
    match &l.blocks {
        Some(b) => {
            if let Some((_, v)) = b.iter().find(|(_, v)| v.len() > block) {
                return Err(format!(
                    "{}: a scale byte covers {} elements, more than the recipe's block {block}",
                    l.name,
                    v.len()
                ));
            }
            Ok(b.clone())
        }
        None => {
            let n = l.elems() as u64;
            Ok((0..n.div_ceil(block as u64))
                .map(|j| {
                    (
                        j,
                        (j * block as u64..((j + 1) * block as u64).min(n)).collect(),
                    )
                })
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_the_hardware_conversions() {
        // Every e4m3 / e2m1 value and midpoint rounds as lowp's encoder (N-CVT-LP) does.
        for (f, t) in [
            (Format::E4m3, ptxv_ir::Lp::E4m3),
            (Format::E2m1, ptxv_ir::Lp::E2m1),
            (Format::E5m2, ptxv_ir::Lp::E5m2),
        ] {
            for c in 0..(1u32 << f.width()) {
                let v = f.value(c);
                if !v.is_finite() {
                    continue;
                }
                assert_eq!(f.code(f.round(v, true).0), c, "{f:?} code {c}");
                for x in [v * 1.0001, v * 0.9999, v * 1.03, v * 0.97, v + 1e-3] {
                    let hw = crate::lowp::from_f32(
                        (x as f32).to_bits(),
                        t,
                        ptxv_ir::LpRnd::Rn,
                        0,
                        true,
                        false,
                    );
                    let ours = f.code(f.round((x as f32) as f64, true).0);
                    assert_eq!(f.value(hw), f.value(ours), "{f:?} {x}");
                }
            }
        }
        // bf16: against the f32 -> bf16 conversion.
        let mut r = Rng::new(1);
        for _ in 0..100_000 {
            let x = (r.normal() * 2f64.powi(r.below(60) as i32 - 30)) as f32;
            let hw = crate::fp::f32_to_bf16_rn(x.to_bits()) as u32;
            assert_eq!(Format::Bf16.code(Format::Bf16.round(x as f64, false).0), hw);
        }
    }

    #[test]
    fn spec_parses_and_samples() {
        let v = serde_json::json!({
            "version": 1, "seed": 7,
            "tensors": {
                "a": {"dist": {"kind": "mixture", "components": [
                    {"weight": 0.9, "dist": {"kind": "normal"}},
                    {"weight": 0.1, "dist": {"kind": "student_t", "df": 3}}]},
                      "scales": [{"axis": 0, "dist": {"kind": "log_normal", "sigma": 0.5}}]},
                "A": {"dist": {"kind": "normal"}, "quant": {"kind": "mx", "elem": "e4m3", "scales": "SFA"}}
            }
        });
        let s = Spec::from_value(v, std::path::Path::new(".")).unwrap();
        let ls = vec![
            Layout {
                name: "a".into(),
                dtype: "bf16".into(),
                shape: vec![4, 8],
                bytes: 64,
                fmt: Format::Bf16,
                blocks: None,
            },
            Layout {
                name: "A".into(),
                dtype: "u8".into(),
                shape: vec![2, 64],
                bytes: 128,
                fmt: Format::E4m3,
                blocks: None,
            },
            Layout {
                name: "SFA".into(),
                dtype: "u8".into(),
                shape: vec![4],
                bytes: 4,
                fmt: Format::Ue8m0,
                blocks: None,
            },
        ];
        let x = sample(&s, &ls, 3).unwrap();
        let y = sample(&s, &ls, 3).unwrap();
        assert_eq!(x.bytes, y.bytes);
        // Each MX block's maximum lands in the top binade of e4m3 (ceil rule: (224, 448]).
        let a = &x.bytes["A"];
        for j in 0..4 {
            let m = (0..32)
                .map(|i| Format::E4m3.value(a[32 * j + i] as u32).abs())
                .fold(0.0, f64::max);
            assert!((224.0..=448.0).contains(&m), "{m}");
        }
        assert!(
            Spec::from_value(serde_json::json!({"version": 2}), std::path::Path::new(".")).is_err()
        );
    }
}
