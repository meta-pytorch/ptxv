/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Numerics hooks of the TPU machine. Every arithmetic result that is not plain data movement
//! goes through `Numerics`, on bit patterns. `Placeholder` is the model of
//! fb/targets/tpu/tools/llosim.py (bitwise equal to it); calibrated models from
//! tools/tpunum.py are ported by implementing the trait (or replacing a method).
//!
//! Status against hardware (probes/corpus): VPU f32/bf16 and conversions exact; EUP and MXU
//! accumulation are placeholders.

/// Canonical NaN of VPU results.
pub const CANON_NAN: u32 = 0x7FC0_0000;

/// Flush an f32 subnormal to signed zero. ASSUME(TPU-N-VPU-FTZ)
#[inline]
pub fn ftz(b: u32) -> u32 {
    if b & 0x7F80_0000 == 0 {
        b & 0x8000_0000
    } else {
        b
    }
}

#[inline]
fn canon(b: u32) -> u32 {
    if f32::from_bits(b).is_nan() {
        CANON_NAN
    } else {
        b
    }
}

#[inline]
pub fn bf16_to_f32(h: u32) -> u32 {
    (h & 0xFFFF) << 16
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VpuOp {
    Add,
    Sub,
    Mul,
    Max,
    Min,
}

pub use ptxv_tpu_ir::EupFn;
pub use ptxv_tpu_ir::XlaneOp;

pub trait Numerics: Send + Sync {
    /// One VPU f32 lane: inputs and result flushed, NaN results canonical, max/min order
    /// -0 < +0 and propagate NaN (probes/corpus ew_f32).
    fn vpu_f32(&self, op: VpuOp, a: u32, b: u32) -> u32 {
        let (x, y) = (f32::from_bits(ftz(a)), f32::from_bits(ftz(b)));
        let r = match op {
            VpuOp::Add => x + y,
            VpuOp::Sub => x - y,
            VpuOp::Mul => x * y,
            VpuOp::Max | VpuOp::Min => {
                if x.is_nan() || y.is_nan() {
                    f32::NAN
                } else {
                    let key = |v: f32| {
                        let i = v.to_bits() as i32;
                        if i < 0 { !(i & 0x7FFF_FFFF) } else { i }
                    };
                    let pick_x = if op == VpuOp::Max {
                        key(x) > key(y)
                    } else {
                        key(x) < key(y)
                    };
                    if pick_x { x } else { y }
                }
            }
        };
        canon(ftz(r.to_bits()))
    }

    /// f32 bits -> bf16 bits (low 16 bits of the result): RNE, subnormal results flushed,
    /// NaN -> 0x7fc0. ASSUME(TPU-N-BF16-RNE)
    fn bf16_round(&self, f: u32) -> u32 {
        let b = ftz(f) as u64;
        let mut r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u32;
        if r & 0x7F80 == 0 {
            r &= 0x8000;
        }
        if f32::from_bits(f).is_nan() {
            r = 0x7FC0;
        }
        r
    }

    /// EUP function of one lane. Placeholder: float64 math, rounded to f32 (div_f32,
    /// trans_f32 differ from hardware). ASSUME(TPU-N-EUP)
    fn eup(&self, f: EupFn, x: u32) -> u32 {
        let v = f32::from_bits(ftz(x)) as f64;
        let r = match f {
            EupFn::Rcp => 1.0 / v,
            EupFn::Exp2 => v.exp2(),
            EupFn::Log2 => v.log2(),
            EupFn::Rsqrt => 1.0 / v.sqrt(),
            EupFn::Tanh => v.tanh(),
            EupFn::Sqrt => v.sqrt(),
            EupFn::Exp => v.exp(),
            EupFn::Log => v.ln(),
        };
        canon(ftz((r as f32).to_bits()))
    }

    /// Cross-lane reduce of one sublane (128 lanes); the result is broadcast by the caller.
    /// Order unobserved (integer data in the corpus): pairwise halving tree in f32.
    /// ASSUME(TPU-N-XLANE-ORDER)
    fn xlane(&self, op: XlaneOp, row: &[u32]) -> u32 {
        let v: Vec<f32> = row.iter().map(|&b| f32::from_bits(ftz(b))).collect();
        match op {
            XlaneOp::Max | XlaneOp::Min => {
                if v.iter().any(|x| x.is_nan()) {
                    return f32::NAN.to_bits();
                }
                let r = v.iter().copied().fold(v[0], |a, b| {
                    if op == XlaneOp::Max {
                        a.max(b)
                    } else {
                        a.min(b)
                    }
                });
                r.to_bits()
            }
            XlaneOp::Add => {
                let mut w = v;
                while w.len() > 1 {
                    let h = w.len() / 2;
                    w = (0..h).map(|i| w[i] + w[i + h]).collect();
                }
                w[0].to_bits()
            }
        }
    }

    /// The f32 LHS of a default-precision matmul as fed to the array: rounded to bf16.
    fn mxu_lhs_f32(&self, x: u32) -> u32 {
        bf16_to_f32(self.bf16_round(x))
    }

    /// One packed-bf16 VPU lane half: `a`, `b` bf16 bits -> bf16 bits. Placeholder: f32 op,
    /// then `bf16_round` (llosim.py).
    fn vpu_bf16(&self, op: VpuOp, a: u32, b: u32) -> u32 {
        self.bf16_round(self.vpu_f32(op, bf16_to_f32(a), bf16_to_f32(b)))
    }

    /// bf16 -> f32 (vunpack): placeholder exact.
    fn bf16_unpack(&self, h: u32) -> u32 {
        bf16_to_f32(h)
    }

    /// An EUP function on one bf16 lane half; `None`: no bitwise model (the op is unmodeled).
    fn eup_bf16(&self, _f: EupFn, _h: u32) -> Option<u32> {
        None
    }

    /// Whether the f32 EUP function has a bitwise model (otherwise the op is unmodeled).
    fn eup_modeled(&self, _f: EupFn) -> bool {
        true
    }

    /// One MXU pass: `a` (m x k bf16 bits) times `w` (k x n bf16 bits), k <= 256 -> m x n f32
    /// bits. Placeholder: `mxu` on the bf16 values.
    fn mxu_pass(&self, a: &[u32], w: &[u32], m: usize, k: usize, n: usize) -> Vec<u32> {
        let lhs: Vec<u32> = a.iter().map(|&h| bf16_to_f32(h)).collect();
        self.mxu(&lhs, w, m, k, n)
    }

    /// The f32 add of MXU result accumulation (vadd.f32 rules).
    fn f32_add(&self, a: u32, b: u32) -> u32 {
        self.vpu_f32(VpuOp::Add, a, b)
    }

    /// `lhs` (m x k, f32 bits, row-major) times `w` (k x n, bf16 bits) -> m x n f32 bits.
    /// Placeholder: exact sums of 8 consecutive k products, each rounded to f32 and
    /// accumulated in k order in f32 (mm_bf16: 587/32768 elements off hardware).
    /// ASSUME(TPU-N-MXU-ACC)
    fn mxu(&self, lhs: &[u32], w: &[u32], m: usize, k: usize, n: usize) -> Vec<u32> {
        let a: Vec<f64> = lhs.iter().map(|&b| f32::from_bits(b) as f64).collect();
        let wf: Vec<f64> = w
            .iter()
            .map(|&b| f32::from_bits(bf16_to_f32(b)) as f64)
            .collect();
        let mut out = vec![0u32; m * n];
        let (mut r, mut acc) = (vec![0f32; n], vec![0f64; n]);
        for i in 0..m {
            r.fill(0.0);
            let mut k0 = 0;
            while k0 < k {
                acc.fill(0.0);
                for kk in k0..(k0 + 8).min(k) {
                    let x = a[i * k + kk];
                    for (s, w) in acc.iter_mut().zip(&wf[kk * n..kk * n + n]) {
                        *s += x * w;
                    }
                }
                for (rj, s) in r.iter_mut().zip(&acc) {
                    *rj = (*rj as f64 + (*s as f32) as f64) as f32;
                }
                k0 += 8;
            }
            for (o, rj) in out[i * n..i * n + n].iter_mut().zip(&r) {
                *o = canon(ftz(rj.to_bits()));
            }
        }
        out
    }
}

/// The tools/llosim.py model (differential oracle).
pub struct Placeholder;

impl Numerics for Placeholder {}

pub static PLACEHOLDER: Placeholder = Placeholder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vpu_rules() {
        let n = &PLACEHOLDER;
        // max(+0, -0) = +0 and max(-0, +0) = +0 (ew_f32 row 0)
        assert_eq!(n.vpu_f32(VpuOp::Max, 0, 0x8000_0000), 0);
        assert_eq!(n.vpu_f32(VpuOp::Max, 0x8000_0000, 0), 0);
        // subnormal inputs flush: 1e-40 * 1 = +0
        assert_eq!(n.vpu_f32(VpuOp::Mul, 1e-40f32.to_bits(), 1f32.to_bits()), 0);
        assert_eq!(
            n.vpu_f32(
                VpuOp::Add,
                f32::INFINITY.to_bits(),
                f32::NEG_INFINITY.to_bits()
            ),
            CANON_NAN
        );
        assert_eq!(n.bf16_round(1f32.to_bits()), 0x3F80);
    }
}

// ------------------------------------------------------------------ calibrated (tools/tpunum.py)

/// f64 value -> bf16 bits: RNE with gradual underflow onto the bf16 grid, then a subnormal result
/// flushed to a signed zero; NaN 0x7fc0; overflow +-Inf. (tpunum `_bf16_ftz_canon`.)
pub fn f64_to_bf16_ftz(x: f64) -> u32 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let sign = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_infinite() {
        return sign | 0x7F80;
    }
    if a < f64::from_bits(0x3810_0000_0000_0000) {
        // below 2^-126: RNE onto multiples of 2^-133; only a round-up to 2^-126 survives the flush
        let q = a * 2f64.powi(133);
        let n = q.round_ties_even();
        return if n >= 128.0 { sign | 0x0080 } else { sign };
    }
    let b = a.to_bits();
    // keep 7 fraction bits of 52: round at bit 45
    let keep = b >> 45;
    let rem = b & ((1u64 << 45) - 1);
    let half = 1u64 << 44;
    let up = rem > half || (rem == half && keep & 1 == 1);
    let r = keep + up as u64; // f64 exponent (11 bits) and 7 fraction bits
    let e64 = (r >> 7) as i64 - 1023;
    if e64 > 127 {
        return sign | 0x7F80;
    }
    let e = (e64 + 127) as u32;
    sign | (e << 7) | (r & 0x7F) as u32
}

fn bf16_val_daz(h: u32) -> f64 {
    let h = h & 0xFFFF;
    if h & 0x7F80 == 0 {
        return if h & 0x8000 != 0 { -0.0 } else { 0.0 };
    }
    f32::from_bits(h << 16) as f64
}

/// One fused group (tpunum `_fused_groups`): exact products, align to the largest product
/// exponent, truncate toward zero at 2^(Emax - 27), exact sum, flush below 2^-126, RNE to f32.
/// A zero sum is +0; Inf/NaN per IEEE with canonical NaN. ASSUME(TPU-N-MXU-SPECIAL)
fn fused_group(p: &[f64]) -> f32 {
    let (mut nan, mut pinf, mut ninf) = (false, false, false);
    let mut emax = i32::MIN;
    for &x in p {
        if x.is_nan() {
            nan = true;
        } else if x == f64::INFINITY {
            pinf = true;
        } else if x == f64::NEG_INFINITY {
            ninf = true;
        } else if x != 0.0 {
            emax = emax.max(((x.abs().to_bits() >> 52) as i32) - 1023);
        }
    }
    if nan || (pinf && ninf) {
        return f32::NAN;
    }
    if pinf {
        return f32::INFINITY;
    }
    if ninf {
        return f32::NEG_INFINITY;
    }
    if emax == i32::MIN {
        return 0.0;
    }
    let w = 27;
    let mut sum = 0f64;
    for &x in p {
        if x.is_finite() && x != 0.0 {
            sum += (x * 2f64.powi(w - emax)).trunc();
        }
    }
    let s = sum * 2f64.powi(emax - w);
    if s == 0.0 {
        return 0.0;
    }
    if s.abs() < 2f64.powi(-126) {
        return if s < 0.0 { -0.0 } else { 0.0 };
    }
    s as f32
}

/// vadd.f32 on values (tpunum `vpu_add_f32`): DAZ inputs, RNE, subnormal result -> signed zero.
fn add_f32v(a: f32, b: f32) -> f32 {
    let f = |x: f32| {
        if x.abs() < f32::MIN_POSITIVE {
            if x.is_sign_negative() { -0.0 } else { 0.0 }
        } else {
            x
        }
    };
    let hi = f(a) as f64 + f(b) as f64;
    let r = hi as f32;
    if hi.abs() < 2f64.powi(-126) {
        return if hi.is_sign_negative() { -0.0 } else { 0.0 };
    }
    r
}

/// The bitwise models of tools/tpunum.py (validated on hardware; notes/numerics.md).
pub struct Calibrated {
    rcp: Vec<i8>,
    rsqrt: Vec<i8>,
}

static RCP_NPZ: &[u8] = include_bytes!("../../../fb/targets/tpu/tools/tpunum_f32_rcp.npz");
static RSQRT_NPZ: &[u8] = include_bytes!("../../../fb/targets/tpu/tools/tpunum_f32_rsqrt.npz");

fn load_diff(npz: &[u8]) -> Vec<i8> {
    let npy = crate::inflate::zip_member(npz, "diff.npy").expect("EUP table");
    let hlen = u16::from_le_bytes([npy[8], npy[9]]) as usize;
    npy[10 + hlen..].iter().map(|&b| b as i8).collect()
}

impl Calibrated {
    fn new() -> Self {
        Calibrated {
            rcp: load_diff(RCP_NPZ),
            rsqrt: load_diff(RSQRT_NPZ),
        }
    }

    /// vrcp.f32: table model (tpunum `vrcp_f32`). ASSUME(TPU-N-EUP-F32)
    fn vrcp_f32(&self, b: u32) -> u32 {
        let s = b & 0x8000_0000;
        let e = ((b >> 23) & 0xFF) as i64;
        let m = (b & 0x7F_FFFF) as i64;
        if e == 255 {
            return if m == 0 { s } else { CANON_NAN };
        }
        if e == 0 {
            return s | 0x7F80_0000;
        }
        let base = ((1.0 / (1.0 + m as f64 / 2f64.powi(23))) as f32).to_bits() as i64;
        let rb = base + self.rcp[m as usize] as i64;
        let re = ((rb >> 23) & 0xFF) - (e - 127);
        let out = if re <= 0 {
            0
        } else if re >= 255 {
            0x7F80_0000
        } else {
            ((re << 23) | (rb & 0x7F_FFFF)) as u32
        };
        out | s
    }

    /// vrsqrt.f32: table model (tpunum `vrsqrt_f32`). ASSUME(TPU-N-EUP-F32)
    fn vrsqrt_f32(&self, b: u32) -> u32 {
        let e = ((b >> 23) & 0xFF) as i64;
        let neg = b >> 31 == 1;
        let frac = (b & 0x7F_FFFF) as i64;
        if e == 255 && frac != 0 {
            return CANON_NAN;
        }
        if e == 0 {
            return if neg { 0xFF80_0000 } else { 0x7F80_0000 };
        }
        if neg {
            return CANON_NAN;
        }
        if e == 255 {
            return 0;
        }
        let ue = e - 127;
        let k = ue.div_euclid(2);
        let ybits = ((ue - 2 * k + 127) << 23) | frac;
        let y = f32::from_bits(ybits as u32) as f64;
        let base = ((1.0 / y.sqrt()) as f32).to_bits() as i64;
        let rb = base + self.rsqrt[(ybits - 0x3F80_0000) as usize] as i64;
        let re = ((rb >> 23) & 0xFF) - k;
        ((re << 23) | (rb & 0x7F_FFFF)) as u32
    }
}

impl Numerics for Calibrated {
    /// vadd/vsub/vmul/vmax/vmin.bf16: exact op on DAZ inputs, one RNE rounding to bf16, FTZ,
    /// canonical NaN; vmul flushes a product below 2^-126 before rounding. ASSUME(TPU-N-VPU-BF16)
    fn vpu_bf16(&self, op: VpuOp, a: u32, b: u32) -> u32 {
        let (x, y) = (bf16_val_daz(a), bf16_val_daz(b));
        let r = match op {
            VpuOp::Add => x + y,
            VpuOp::Sub => x - y,
            VpuOp::Mul => {
                let p = x * y;
                if p.abs() < 2f64.powi(-126) {
                    if p.is_sign_negative() { -0.0 } else { 0.0 }
                } else {
                    p
                }
            }
            VpuOp::Max | VpuOp::Min => {
                if x.is_nan() || y.is_nan() {
                    f64::NAN
                } else if op == VpuOp::Max {
                    if x > y || (x == y && !x.is_sign_negative()) {
                        x
                    } else {
                        y
                    }
                } else if x < y || (x == y && x.is_sign_negative()) {
                    x
                } else {
                    y
                }
            }
        };
        f64_to_bf16_ftz(r)
    }

    /// vunpack bf16 -> f32: exact, bf16 subnormals -> signed zero, canonical NaN. ASSUME(TPU-N-CVT)
    fn bf16_unpack(&self, h: u32) -> u32 {
        let v = bf16_val_daz(h);
        if v.is_nan() {
            CANON_NAN
        } else {
            (v as f32).to_bits()
        }
    }

    fn eup_modeled(&self, f: EupFn) -> bool {
        matches!(f, EupFn::Rcp | EupFn::Rsqrt)
    }

    fn eup(&self, f: EupFn, x: u32) -> u32 {
        match f {
            EupFn::Rcp => self.vrcp_f32(x),
            EupFn::Rsqrt => self.vrsqrt_f32(x),
            _ => CANON_NAN,
        }
    }

    /// bf16 EUP: correctly rounded (f64 reference), DAZ, FTZ, canonical NaN. ASSUME(TPU-N-EUP-BF16)
    fn eup_bf16(&self, f: EupFn, h: u32) -> Option<u32> {
        let x = bf16_val_daz(h);
        let r = match f {
            EupFn::Exp2 => x.exp2(),
            EupFn::Rsqrt => 1.0 / x.sqrt(),
            EupFn::Rcp => 1.0 / x,
            EupFn::Tanh => x.tanh(),
            EupFn::Log2 => x.log2(),
            _ => return None,
        };
        Some(f64_to_bf16_ftz(r))
    }

    /// Cross-lane sum: lanes 8g..8g+7 as ((0+1)+(2+3))+((4+5)+(6+7)), the 16 group sums
    /// accumulated left to right; every node vadd.f32. ASSUME(TPU-N-REDUCE)
    fn xlane(&self, op: XlaneOp, row: &[u32]) -> u32 {
        let v: Vec<f32> = row.iter().map(|&b| f32::from_bits(b)).collect();
        match op {
            XlaneOp::Add => {
                let mut acc: Option<f32> = None;
                for g in 0..16 {
                    let l = &v[8 * g..8 * g + 8];
                    let t = add_f32v(
                        add_f32v(add_f32v(l[0], l[1]), add_f32v(l[2], l[3])),
                        add_f32v(add_f32v(l[4], l[5]), add_f32v(l[6], l[7])),
                    );
                    acc = Some(acc.map_or(t, |a| add_f32v(a, t)));
                }
                let r = acc.expect("16 groups");
                if r.is_nan() { CANON_NAN } else { r.to_bits() }
            }
            XlaneOp::Max | XlaneOp::Min => {
                let op = if op == XlaneOp::Max {
                    VpuOp::Max
                } else {
                    VpuOp::Min
                };
                row.iter()
                    .skip(1)
                    .fold(ftz(row[0]), |a, &b| self.vpu_f32(op, a, b))
            }
        }
    }

    /// One MXU pass (tpunum `mxu_pass`): K padded to 256; group j = products at K
    /// {4j..4j+3} U {128+4j..128+4j+3}, fused; pairs (2j, 2j+1) added; 16 pair sums accumulated
    /// left to right; products: subnormal operands 0, |p| < 2^-126 -> +0, |p| >= 2^128 -> Inf.
    /// ASSUME(TPU-N-MXU-BF16, TPU-N-MXU-SPECIAL)
    fn mxu_pass(&self, a: &[u32], w: &[u32], m: usize, k: usize, n: usize) -> Vec<u32> {
        assert!(k <= 256);
        let av: Vec<f64> = a.iter().map(|&h| bf16_val_daz(h)).collect();
        let wv: Vec<f64> = w.iter().map(|&h| bf16_val_daz(h)).collect();
        let mut out = vec![0u32; m * n];
        let mut p = [0f64; 8];
        for i in 0..m {
            for j in 0..n {
                let mut g = [0f32; 32];
                for (gj, gv) in g.iter_mut().enumerate() {
                    for h in 0..2 {
                        for t in 0..4 {
                            let kk = h * 128 + 4 * gj + t;
                            let mut x = if kk < k {
                                av[i * k + kk] * wv[kk * n + j]
                            } else {
                                0.0
                            };
                            if x.abs() < 2f64.powi(-126) {
                                x = 0.0;
                            } else if x.abs() >= 2f64.powi(128) {
                                x = f64::INFINITY.copysign(x);
                            }
                            p[h * 4 + t] = x;
                        }
                    }
                    *gv = fused_group(&p);
                }
                let mut acc: Option<f32> = None;
                for q in 0..16 {
                    let u = add_f32v(g[2 * q], g[2 * q + 1]);
                    acc = Some(acc.map_or(u, |x| add_f32v(x, u)));
                }
                let r = acc.expect("16 pairs");
                out[i * n + j] = r.to_bits();
            }
        }
        out
    }

    fn f32_add(&self, a: u32, b: u32) -> u32 {
        add_f32v(f32::from_bits(a), f32::from_bits(b)).to_bits()
    }
}

/// The calibrated models (loaded once).
pub fn calibrated() -> &'static Calibrated {
    static C: std::sync::OnceLock<Calibrated> = std::sync::OnceLock::new();
    C.get_or_init(Calibrated::new)
}
