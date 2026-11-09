//! `ptxv check --nonfinite`: NaN / Inf in the outputs, and where they were produced.
//!
//! Two parts:
//! - a scan of output (and input) buffers by element format (`scan`);
//! - births (`record`, from `exec.rs`): an arithmetic instruction whose result is non-finite
//!   while its operands were finite (overflow, 0 / 0, log / rsqrt of 0 or a negative), or NaN
//!   while no operand was NaN (inf - inf, 0 * inf). Moves, selects, min / max and loads are
//!   never births, and an infinite constant operand is an operand like any other: an
//!   intentional -inf (an attention mask, a running max's start) is not a birth, the NaN it
//!   later produces (-inf - -inf) is.
//!
//! Births go to a process-wide sink, off unless `enable` was called: exploration clones the
//! machine per state, and the sink must not make otherwise equal states differ.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use serde::Serialize;

/// Whether births are recorded.
static ON: AtomicBool = AtomicBool::new(false);

static SINK: Mutex<BTreeMap<u32, Birth>> = Mutex::new(BTreeMap::new());

/// One instruction that produced a non-finite value from operands that did not hold one of
/// that kind: the first time it did (operands and result), and how many times.
#[derive(Clone, Debug, Serialize)]
pub struct Birth {
    pub pc: u32,
    /// "inf from finite", "nan from finite" or "nan from inf".
    pub kind: String,
    pub thread: usize,
    pub operands: Vec<String>,
    pub result: String,
    pub count: u64,
}

/// Start recording births (clears the sink).
pub fn enable() {
    SINK.lock().unwrap().clear();
    ON.store(true, Ordering::Relaxed);
}

/// Stop recording; the births so far, by pc.
pub fn take() -> Vec<Birth> {
    ON.store(false, Ordering::Relaxed);
    std::mem::take(&mut *SINK.lock().unwrap())
        .into_values()
        .collect()
}

pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Element formats of a buffer or register lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    F64,
    F32,
    F16,
    Bf16,
    /// fp8 e4m3 (the "fn" variant: NaN only, no infinity).
    E4m3,
    E5m2,
    /// fp4 / fp6 element formats and e8m0 / ue4m3 scales.
    E2m1,
    E2m3,
    E3m2,
    Ue8m0,
    Ue4m3,
}

impl Fmt {
    /// A case buffer dtype's element format (None: not a float buffer).
    pub fn of_dtype(d: &str) -> Option<Fmt> {
        Some(match d {
            "f64" => Fmt::F64,
            "f32" => Fmt::F32,
            "f16" => Fmt::F16,
            "bf16" => Fmt::Bf16,
            "e4m3" | "fp8_e4m3" | "e4m3fn" => Fmt::E4m3,
            "e5m2" | "fp8_e5m2" => Fmt::E5m2,
            "e2m1" | "fp4" | "fp4_e2m1" => Fmt::E2m1,
            "e2m3" => Fmt::E2m3,
            "e3m2" => Fmt::E3m2,
            "ue8m0" | "e8m0" => Fmt::Ue8m0,
            "ue4m3" => Fmt::Ue4m3,
            _ => return None,
        })
    }

    pub fn of_lp(l: ptxv_ir::Lp) -> Fmt {
        use ptxv_ir::Lp;
        match l {
            Lp::E4m3 => Fmt::E4m3,
            Lp::E5m2 => Fmt::E5m2,
            Lp::E2m3 => Fmt::E2m3,
            Lp::E3m2 => Fmt::E3m2,
            Lp::E2m1 => Fmt::E2m1,
            Lp::Ue8m0 => Fmt::Ue8m0,
            Lp::Ue4m3 => Fmt::Ue4m3,
            Lp::F16 => Fmt::F16,
            Lp::Bf16 => Fmt::Bf16,
        }
    }

    /// Bits per element.
    pub fn bits(self) -> u32 {
        match self {
            Fmt::F64 => 64,
            Fmt::F32 => 32,
            Fmt::F16 | Fmt::Bf16 => 16,
            Fmt::E2m1 => 4,
            _ => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Fmt::F64 => "f64",
            Fmt::F32 => "f32",
            Fmt::F16 => "f16",
            Fmt::Bf16 => "bf16",
            Fmt::E4m3 => "e4m3",
            Fmt::E5m2 => "e5m2",
            Fmt::E2m1 => "e2m1",
            Fmt::E2m3 => "e2m3",
            Fmt::E3m2 => "e3m2",
            Fmt::Ue8m0 => "ue8m0",
            Fmt::Ue4m3 => "ue4m3",
        }
    }

    /// The class of one element's bits (low `bits()` bits of `x`).
    pub fn class(self, x: u64) -> Class {
        let c = |inf: bool, nan: bool| {
            if nan {
                Class::Nan
            } else if inf {
                Class::Inf
            } else {
                Class::Finite
            }
        };
        match self {
            Fmt::F64 => {
                let e = (x >> 52) & 0x7ff == 0x7ff;
                c(e, e && x & ((1 << 52) - 1) != 0)
            }
            Fmt::F32 => {
                let e = (x >> 23) & 0xff == 0xff;
                c(e, e && x & 0x7f_ffff != 0)
            }
            Fmt::F16 => {
                let e = x & 0x7c00 == 0x7c00;
                c(e, e && x & 0x3ff != 0)
            }
            Fmt::Bf16 => {
                let e = x & 0x7f80 == 0x7f80;
                c(e, e && x & 0x7f != 0)
            }
            Fmt::E4m3 | Fmt::Ue4m3 => c(false, x & 0x7f == 0x7f),
            Fmt::E5m2 => {
                let e = x & 0x7c == 0x7c;
                c(e, e && x & 3 != 0)
            }
            Fmt::Ue8m0 => c(false, x & 0xff == 0xff),
            Fmt::E2m1 | Fmt::E2m3 | Fmt::E3m2 => Class::Finite,
        }
    }

    /// The value of one element, for messages.
    pub fn show(self, x: u64) -> String {
        match self {
            Fmt::F64 => format!("{:e}", f64::from_bits(x)),
            Fmt::F32 => format!("{:e}", f32::from_bits(x as u32)),
            Fmt::F16 => format!(
                "{:e} (f16 {x:#06x})",
                f32::from_bits(crate::elementwise::f16_to_f32_bits(x as u16))
            ),
            Fmt::Bf16 => format!("{:e} (bf16 {x:#06x})", f32::from_bits((x as u32) << 16)),
            f => match f.class(x) {
                Class::Nan => format!("NaN ({} {x:#04x})", f.name()),
                Class::Inf if x & 0x80 != 0 => format!("-inf ({} {x:#04x})", f.name()),
                Class::Inf => format!("inf ({} {x:#04x})", f.name()),
                Class::Finite => format!("{} {x:#04x}", f.name()),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Finite,
    Inf,
    Nan,
}

/// Non-finite elements of one buffer.
#[derive(Clone, Debug, Default, Serialize)]
pub struct BufScan {
    pub buffer: String,
    pub dtype: String,
    pub elements: u64,
    pub nan: u64,
    pub inf: u64,
    /// The first non-finite element: (index, value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first: Option<(u64, String)>,
}

impl BufScan {
    pub fn nonfinite(&self) -> u64 {
        self.nan + self.inf
    }
}

/// Scan `bytes` (little-endian elements of format `f`; e2m1: two per byte, low nibble first).
pub fn scan(buffer: &str, dtype: &str, f: Fmt, bytes: &[u8]) -> BufScan {
    let mut s = BufScan {
        buffer: buffer.into(),
        dtype: dtype.into(),
        ..Default::default()
    };
    let w = f.bits();
    let n = bytes.len() as u64 * 8 / w as u64;
    s.elements = n;
    for i in 0..n {
        let x = if w >= 8 {
            let b = (w / 8) as usize;
            let o = i as usize * b;
            let mut v = 0u64;
            for (k, &y) in bytes[o..o + b].iter().enumerate() {
                v |= (y as u64) << (8 * k);
            }
            v
        } else {
            (bytes[i as usize / 2] as u64 >> (4 * (i % 2))) & 0xf
        };
        match f.class(x) {
            Class::Finite => continue,
            Class::Inf => s.inf += 1,
            Class::Nan => s.nan += 1,
        }
        if s.first.is_none() {
            s.first = Some((i, f.show(x)));
        }
    }
    s
}

/// One operand or result lane: its format and bits.
pub type Lane = (Fmt, u64);

/// Record a birth at `pc` (thread `t`) if `result` holds a non-finite lane that `operands`
/// do not explain: any non-finite result lane with every operand lane finite, or a NaN
/// result lane with no NaN operand lane.
pub fn record(pc: u32, t: usize, operands: &[Lane], result: &[Lane]) {
    let worst = |v: &[Lane]| v.iter().map(|&(f, x)| f.class(x)).max();
    let (Some(r), oc) = (worst(result), worst(operands).unwrap_or(Class::Finite)) else {
        return;
    };
    let kind = match (oc, r) {
        (Class::Finite, Class::Inf) => "inf from finite",
        (Class::Finite, Class::Nan) => "nan from finite",
        (Class::Inf, Class::Nan) => "nan from inf",
        _ => return,
    };
    let mut s = SINK.lock().unwrap();
    s.entry(pc)
        .and_modify(|b| b.count += 1)
        .or_insert_with(|| Birth {
            pc,
            kind: kind.into(),
            thread: t,
            operands: operands.iter().map(|&(f, x)| f.show(x)).collect(),
            result: result
                .iter()
                .map(|&(f, x)| f.show(x))
                .collect::<Vec<_>>()
                .join(", "),
            count: 1,
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert_eq!(Fmt::F32.class(0x7f80_0000), Class::Inf);
        assert_eq!(Fmt::F32.class(0x7fc0_0000), Class::Nan);
        assert_eq!(Fmt::F32.class(0x7f7f_ffff), Class::Finite);
        assert_eq!(Fmt::F16.class(0x7c00), Class::Inf);
        assert_eq!(Fmt::F16.class(0xfe00), Class::Nan);
        assert_eq!(Fmt::Bf16.class(0xff80), Class::Inf);
        assert_eq!(Fmt::E4m3.class(0x7e), Class::Finite);
        assert_eq!(Fmt::E4m3.class(0xff), Class::Nan);
        assert_eq!(Fmt::E5m2.class(0x7c), Class::Inf);
        assert_eq!(Fmt::E5m2.class(0x7d), Class::Nan);
        assert_eq!(Fmt::E2m1.class(0xf), Class::Finite);
    }

    #[test]
    fn scan_counts() {
        let b: Vec<u8> = [1.0f32, f32::NAN, f32::NEG_INFINITY, 2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let s = scan("y", "f32", Fmt::F32, &b);
        assert_eq!((s.elements, s.nan, s.inf), (4, 1, 1));
        assert_eq!(s.first.as_ref().unwrap().0, 1);
    }
}
