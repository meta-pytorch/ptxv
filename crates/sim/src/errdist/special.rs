/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Random numbers and special functions for errdist (no external crates).

/// xoshiro256++ seeded through splitmix64.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

pub fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A stable 64-bit hash of a string (FNV-1a), for per-tensor streams.
pub fn hash_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        Rng {
            s: [
                splitmix(&mut x),
                splitmix(&mut x),
                splitmix(&mut x),
                splitmix(&mut x),
            ],
        }
    }
    /// The stream of (seed, sample, name): the same draw whatever order samples are made in.
    pub fn stream(seed: u64, sample: u64, name: &str) -> Rng {
        let mut x =
            seed ^ sample.wrapping_mul(0xd1b5_4a32_d192_ed03) ^ hash_str(name).rotate_left(17);
        Rng::new(splitmix(&mut x))
    }
    pub fn next_u64(&mut self) -> u64 {
        let r = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in (0, 1).
    pub fn open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }
    pub fn normal(&mut self) -> f64 {
        let (a, b) = (self.open(), self.unit());
        (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
    }
    /// Gamma(shape k, scale 1), Marsaglia-Tsang.
    pub fn gamma(&mut self, k: f64) -> f64 {
        if k < 1.0 {
            let u = self.open();
            return self.gamma(k + 1.0) * u.powf(1.0 / k);
        }
        let d = k - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.open();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

const FRAC_2_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// erfc(x), absolute error below ~1e-15: Taylor series of erf for |x| <= 2, continued
/// fraction beyond.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x <= 2.0 {
        let (mut term, mut sum, x2) = (x, x, x * x);
        for n in 1..200 {
            term *= -x2 / n as f64;
            let t = term / (2 * n + 1) as f64;
            sum += t;
            if t.abs() < 1e-17 * sum.abs() {
                break;
            }
        }
        return 1.0 - FRAC_2_SQRT_PI * sum;
    }
    if x > 27.0 {
        return 0.0;
    }
    // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))), Lentz.
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for n in 1..500 {
        let a = n as f64 / 2.0;
        d = x + a * d;
        d = if d.abs() < tiny { tiny } else { d };
        c = x + a / c;
        c = if c.abs() < tiny { tiny } else { c };
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    (-x * x).exp() / (f * std::f64::consts::PI.sqrt())
}

/// Standard normal CDF.
pub fn phi(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal density.
pub fn phi_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (std::f64::consts::TAU).sqrt()
}

/// ln Gamma(x), x > 0 (Lanczos, g = 7, n = 9).
pub fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (std::f64::consts::TAU).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized incomplete beta I_x(a, b) (continued fraction, Numerical Recipes betacf).
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let cf = |a: f64, b: f64, x: f64| -> f64 {
        let tiny = 1e-300;
        let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;
        d = if d.abs() < tiny { tiny } else { d };
        d = 1.0 / d;
        let mut h = d;
        for m in 1..1000 {
            let m = m as f64;
            let m2 = 2.0 * m;
            let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            d = if d.abs() < tiny { tiny } else { d };
            c = 1.0 + aa / c;
            c = if c.abs() < tiny { tiny } else { c };
            d = 1.0 / d;
            h *= d * c;
            let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            d = if d.abs() < tiny { tiny } else { d };
            c = 1.0 + aa / c;
            c = if c.abs() < tiny { tiny } else { c };
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        h
    };
    if x < (a + 1.0) / (a + b + 2.0) {
        lbt.exp() * cf(a, b, x) / a
    } else {
        1.0 - lbt.exp() * cf(b, a, 1.0 - x) / b
    }
}

/// Student-t CDF with `df` degrees of freedom.
pub fn student_t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let tail = 0.5 * beta_inc(df / 2.0, 0.5, x);
    if t > 0.0 { 1.0 - tail } else { tail }
}

/// Kolmogorov distribution: P(sqrt(n) D_n > z) asymptotically.
pub fn kolmogorov_q(z: f64) -> f64 {
    if z < 0.2 {
        return 1.0;
    }
    let mut s = 0.0;
    for k in 1..200 {
        let t = 2.0 * (-1f64).powi(k - 1) * (-2.0 * (k as f64 * z).powi(2)).exp();
        s += t;
        if t.abs() < 1e-17 {
            break;
        }
    }
    s.clamp(0.0, 1.0)
}

/// Clopper-Pearson interval for k successes in n trials at confidence 1 - alpha.
/// Hoeffding's (1963, Theorem 1) two-sided 1 - alpha interval for the mean of n independent
/// variables in [0, 1] whose sample mean is `m`: the q with n KL(m || q) <= ln(2 / alpha), KL
/// the Bernoulli divergence. It holds for any distribution on [0, 1]; for 0 / 1 variables at
/// m = 0 the upper end is Clopper-Pearson's, 1 - (alpha / 2)^(1/n).
pub fn kl_mean_bounds(m: f64, n: usize, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let c = (2.0 / alpha).ln() / n as f64;
    let kl = |p: f64, q: f64| -> f64 {
        let t = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
        t(p, q) + t(1.0 - p, 1.0 - q)
    };
    // KL(m || q) grows away from m on either side: bisect for where it reaches c.
    let solve = |mut a: f64, mut b: f64, below: bool| -> f64 {
        for _ in 0..200 {
            let mid = 0.5 * (a + b);
            if (kl(m, mid) <= c) == below {
                a = mid;
            } else {
                b = mid;
            }
        }
        0.5 * (a + b)
    };
    let hi = if m >= 1.0 || kl(m, 1.0 - 1e-15) <= c {
        1.0
    } else {
        solve(m, 1.0, true)
    };
    let lo = if m <= 0.0 || kl(m, 1e-300) <= c {
        0.0
    } else {
        solve(m, 0.0, true)
    };
    (lo, hi)
}

pub fn clopper_pearson(k: u64, n: u64, alpha: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let inv = |target: f64, a: f64, b: f64| -> f64 {
        // I_x(a, b) = target, bisection.
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if beta_inc(a, b, mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let (k, n) = (k as f64, n as f64);
    let lo = if k == 0.0 {
        0.0
    } else {
        inv(alpha / 2.0, k, n - k + 1.0)
    };
    let hi = if k == n {
        1.0
    } else {
        inv(1.0 - alpha / 2.0, k + 1.0, n - k)
    };
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn special_functions_match_known_values() {
        assert!((phi(1.0) - 0.841_344_746_068_542_9).abs() < 1e-15);
        assert!((phi(-3.0) - 0.001_349_898_031_630_094_6).abs() < 1e-16);
        assert!((erfc(3.0) - 2.209_049_699_858_544e-5).abs() < 1e-18);
        assert!((ln_gamma(10.0) - 362880f64.ln()).abs() < 1e-12);
        assert!((student_t_cdf(2.0, 3.0) - 0.930_337_9).abs() < 1e-6);
        let (lo, hi) = clopper_pearson(5, 100, 0.05);
        assert!((lo - 0.016_431).abs() < 1e-5 && (hi - 0.112_836).abs() < 1e-5);
    }
}
