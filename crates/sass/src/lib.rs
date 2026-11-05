/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! SASS front-end: cubin -> `nvdisasm` text -> `ptxv_ir::Kernel`.
//!
//! Every opcode mapping here was derived by compiling a PTX probe with `ptxas` and reading
//! the SASS it produced (the "Rosetta" method, see ROADMAP.md), and is validated by
//! differential runs against GB300. Unknown opcodes or modifiers are hard errors.
//!
//! Notable lowerings (sm_100a / sm_103a):
//! * `SYNCS.EXCH.64 URZ, [mbar], URx`         -> `mbarrier.init` with an encoded state word
//! * `SYNCS.ARRIVE.TRANS64 RZ, [mbar], Rtx`   -> `mbarrier.arrive.expect_tx`
//! * `SYNCS.PHASECHK.TRANS64[.TRYWAIT] P, [mbar], Rp` -> `try_wait.parity`, parity in bit 31
//! * `UTMALDG.Nd [URa], [URb]`  -> TMA load; URa..: smem dst, mbar, N coords; URb: tensor map
//! * `ELECT` + `BRA.U.ANY` loops are how ptxas issues uniform-datapath ops per active thread.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_ir::*;

pub mod elf;
pub mod scoreboard;

pub const RZ: u16 = 255;
/// sm_100 has uniform registers UR0..UR254 (ptxas uses UR63 and UR64 as ordinary
/// registers); URZ is index 255.
pub const URZ: u16 = 255;
pub const PT: u16 = 7;

#[derive(Debug, Clone)]
pub struct SassLine {
    pub addr: u32,
    pub guard: Option<String>,
    pub opcode: String,
    pub operands: Vec<String>,
    pub text: String,
    /// High 64 bits of the encoding (scheduling control bits live in [41, 64)), when the
    /// disassembly was produced with `-hex`.
    pub hi: Option<u64>,
    /// Low 64 bits of the encoding (`-hex`); a 32-bit immediate is bits [32, 64).
    pub lo: Option<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct CubinMeta {
    /// (ordinal, offset, size) from EIATTR_KPARAM_INFO.
    pub params: Vec<(u32, u32, u32)>,
    pub const0_size: u32,
    pub smem_size: u32,
    /// Contents of the kernel's constant banks other than 0 (`.nv.constant<N>.<kernel>`, or
    /// the module's `.nv.constant<N>`), relocated (`elf::Elf::bank`).
    pub banks: HashMap<u8, Vec<u8>>,
    /// Synthetic addresses of the relocated symbols in those banks (address, name).
    pub syms: Vec<(u64, String)>,
}

fn tool(name: &str) -> String {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-13.0".into());
    format!("{cuda}/bin/{name}")
}

/// Run a CUDA binary tool and return its stdout. `nvdisasm` and `cuobjdump` output is
/// memoized per process, keyed by the arguments and the bytes of every file argument: a
/// multi-launch case loads the same multi-megabyte cubin once per launch.
pub fn run_tool(name: &str, args: &[&str]) -> Result<String> {
    use std::hash::Hash;
    use std::hash::Hasher;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    /// (tool path, args, hash and total length of the file arguments' bytes).
    type MemoKey = (String, Vec<String>, u64, usize);
    static MEMO: OnceLock<Mutex<HashMap<MemoKey, String>>> = OnceLock::new();
    let key = if matches!(name, "nvdisasm" | "cuobjdump") {
        let (mut h, mut n) = (std::collections::hash_map::DefaultHasher::new(), 0);
        for a in args {
            if let Ok(b) = std::fs::read(a) {
                b.hash(&mut h);
                n += b.len();
            }
        }
        let args = args.iter().map(|a| a.to_string()).collect();
        Some((tool(name), args, h.finish(), n))
    } else {
        None
    };
    let memo = MEMO.get_or_init(Default::default);
    if let Some(out) = key
        .as_ref()
        .and_then(|k| memo.lock().unwrap().get(k).cloned())
    {
        return Ok(out);
    }
    let out = run_tool_uncached(name, args)?;
    if let Some(k) = key {
        memo.lock().unwrap().insert(k, out.clone());
    }
    Ok(out)
}

fn run_tool_uncached(name: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(tool(name))
        .args(args)
        .output()
        .with_context(|| format!("running {name}"))?;
    if !out.status.success() {
        bail!("{name} failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    Ok(String::from_utf8(out.stdout)?)
}

/// Compile PTX to a cubin with ptxas.
pub fn ptxas(ptx: &Path, cubin: &Path, arch: &str) -> Result<()> {
    run_tool(
        "ptxas",
        &[
            &format!("-arch={arch}"),
            ptx.to_str().unwrap(),
            "-o",
            cubin.to_str().unwrap(),
        ],
    )?;
    Ok(())
}

fn hex(s: &str) -> Option<u32> {
    u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

pub fn read_meta(cubin: &Path, kernel: &str) -> Result<CubinMeta> {
    let elf = run_tool("cuobjdump", &["-elf", cubin.to_str().unwrap()])?;
    let mut m = CubinMeta::default();
    for line in elf.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if let Some(name) = cols.last() {
            if cols.len() >= 4 {
                // Only the section header table (first occurrence); the symbol table
                // later lists the same names with other columns.
                if *name == format!(".nv.constant0.{kernel}") && m.const0_size == 0 {
                    m.const0_size = hex(cols[2]).unwrap_or(0);
                } else if *name == format!(".nv.shared.{kernel}") && m.smem_size == 0 {
                    m.smem_size = hex(cols[2]).unwrap_or(0);
                }
            }
        }
    }
    // KPARAM_INFO blocks inside the kernel's .nv.info section.
    let sect = format!(".nv.info.{kernel}");
    let start = elf.lines().position(|l| l.trim() == sect);
    if let Some(start) = start {
        for line in elf.lines().skip(start + 1) {
            if line.starts_with('.') {
                break;
            }
            if let Some(v) = line.trim().strip_prefix("Value:") {
                if v.contains("Ordinal") && v.contains("Offset") {
                    let field = |k: &str| -> Option<u32> {
                        let i = v.find(k)? + k.len();
                        let rest = v[i..].trim_start().trim_start_matches(':').trim_start();
                        hex(rest.split_whitespace().next()?)
                    };
                    m.params.push((
                        field("Ordinal").unwrap(),
                        field("Offset").unwrap(),
                        field("Size").unwrap(),
                    ));
                }
            }
        }
    }
    m.params.sort();
    let img = elf::Elf::parse(&std::fs::read(cubin)?)?;
    for bank in 1..18u8 {
        if let Some((data, syms)) = img.bank(bank, kernel)? {
            m.banks.insert(bank, data);
            m.syms.extend(syms);
        }
    }
    if m.smem_size == 0 {
        m.smem_size = 0x400;
    }
    Ok(m)
}

/// Parse `nvdisasm -c [-hex]` output for one function.
pub fn parse_disasm(text: &str, kernel: &str) -> Result<(Vec<SassLine>, HashMap<String, u32>)> {
    let mut lines = Vec::new();
    let mut labels = HashMap::new();
    let mut in_fn = false;
    let mut pending_labels: Vec<String> = Vec::new();
    for raw in text.lines() {
        let t = raw.trim();
        if t.starts_with(".text.") && t.ends_with(':') {
            in_fn = t == format!(".text.{kernel}:");
            continue;
        }
        if t.starts_with("//----") {
            // A section header; executable cubins (no `.text.<name>:` label) start the
            // function here and label it `<name>:`.
            in_fn = match t.split_whitespace().nth(1) {
                Some(sec) if sec.starts_with(".text.") => sec == format!(".text.{kernel}"),
                _ => in_fn && !t.contains("SYMBOLS"),
            };
            continue;
        }
        if !in_fn {
            continue;
        }
        // `-hex`: the second line of each instruction holds the high encoding word.
        if let Some(w) = t.strip_prefix("/* 0x").and_then(|r| r.strip_suffix("*/")) {
            if let Some(l) = lines.last_mut() {
                let l: &mut SassLine = l;
                if l.hi.is_none() {
                    l.hi = u64::from_str_radix(w.trim(), 16).ok();
                }
            }
            continue;
        }
        if (t.starts_with(".L_") || t.starts_with('$')) && t.ends_with(':') {
            pending_labels.push(t.trim_end_matches(':').to_string());
            continue;
        }
        let Some(rest) = t.strip_prefix("/*") else {
            continue;
        };
        let Some((a, body)) = rest.split_once("*/") else {
            continue;
        };
        let Some(addr) = hex(a) else { continue };
        for l in pending_labels.drain(..) {
            labels.insert(l, addr);
        }
        let mut halves = body.split("/* 0x");
        let body = halves.next().unwrap();
        let lo = halves
            .next()
            .and_then(|w| u64::from_str_radix(w.trim().trim_end_matches("*/").trim(), 16).ok());
        let body = body.trim().trim_end_matches(';').trim();
        let (guard, body) = if body.starts_with('@') {
            let (g, r) = body.split_once(char::is_whitespace).unwrap();
            (Some(g[1..].to_string()), r.trim())
        } else {
            (None, body)
        };
        let (opcode, ops) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
        let operands = split_ops(ops);
        lines.push(SassLine {
            addr,
            guard,
            opcode: opcode.to_string(),
            operands,
            text: body.to_string(),
            hi: None,
            lo,
        });
    }
    if lines.is_empty() {
        bail!("kernel {kernel} not found in disassembly");
    }
    Ok((lines, labels))
}

fn split_ops(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '[' | '(' | '{' => {
                depth += 1;
                cur.push(c)
            }
            ']' | ')' | '}' => {
                depth -= 1;
                cur.push(c)
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear()
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

struct Lw<'a> {
    l: &'a SassLine,
}

impl<'a> Lw<'a> {
    fn err(&self, m: impl std::fmt::Display) -> anyhow::Error {
        anyhow!("sass {:#06x}: {m}\n    `{}`", self.l.addr, self.l.text)
    }

    fn op(&self, i: usize) -> Result<&str> {
        self.l
            .operands
            .get(i)
            .map(|s| s.as_str())
            .ok_or_else(|| self.err(format!("missing operand {i}")))
    }

    /// Parse a register name (no modifiers): R12, RZ, UR4, URZ.
    fn reg_name(s: &str) -> Option<(File, u16)> {
        let s = s.trim_end_matches(".reuse");
        if s == "RZ" {
            return Some((File::R, RZ));
        }
        if s == "URZ" {
            return Some((File::UR, URZ));
        }
        if let Some(n) = s.strip_prefix("UR") {
            return n.parse().ok().map(|n| (File::UR, n));
        }
        if let Some(n) = s.strip_prefix('R') {
            return n.parse().ok().map(|n| (File::R, n));
        }
        None
    }

    fn dst(&self, s: &str) -> Result<Reg> {
        let s = s.trim_end_matches(".64");
        let (file, idx) = Self::reg_name(s).ok_or_else(|| self.err(format!("bad dst {s}")))?;
        Ok(Reg { file, idx })
    }

    fn pred(&self, s: &str) -> Result<Pred> {
        let (neg, s) = match s.strip_prefix('!') {
            Some(r) => (true, r),
            None => (false, s),
        };
        let reg = match s {
            "PT" | "UPT" => None,
            _ if s.starts_with("UP") => Some(Reg {
                file: File::UP,
                idx: s[2..].parse().map_err(|_| self.err("bad pred"))?,
            }),
            _ if s.starts_with('P') => Some(Reg {
                file: File::P,
                idx: s[1..].parse().map_err(|_| self.err("bad pred"))?,
            }),
            _ => return Err(self.err(format!("bad predicate {s}"))),
        };
        Ok(Pred { reg, neg })
    }

    fn pdst(&self, s: &str) -> Result<Reg> {
        match s {
            "PT" => Ok(Reg {
                file: File::P,
                idx: PT,
            }),
            "UPT" => Ok(Reg {
                file: File::UP,
                idx: PT,
            }),
            _ => {
                let p = self.pred(s)?;
                if p.neg {
                    bail!(self.err("negated predicate destination"));
                }
                Ok(p.reg.unwrap())
            }
        }
    }

    fn is_pred_tok(s: &str) -> bool {
        let s = s.trim_start_matches('!');
        s == "PT"
            || s == "UPT"
            || (s.starts_with('P') && s[1..].parse::<u8>().is_ok())
            || (s.starts_with("UP") && s[2..].parse::<u8>().is_ok())
    }

    /// Source value: register (optionally `.64`), immediate, `c[0x0][off]`, special reg.
    fn val(&self, s: &str, bytes: u8) -> Result<Val> {
        let s = s.trim();
        if let Some(h) = s.strip_prefix("-0x") {
            return u64::from_str_radix(h, 16)
                .map(|v| Val::Imm(v.wrapping_neg() & 0xffff_ffff))
                .map_err(|_| self.err("bad imm"));
        }
        if s.starts_with('-') || s.starts_with('|') || s.starts_with('~') {
            bail!(self.err(format!("operand modifier not modeled: {s}")));
        }
        if let Some(r) = s.strip_prefix("c[0x0][") {
            let off = hex(r.trim_end_matches(']')).ok_or_else(|| self.err("bad cbank"))?;
            return Ok(Val::Const(off, bytes));
        }
        if let Some(sr) = s.strip_prefix("SR_") {
            let v = match sr {
                "TID.X" => SReg::Tid(0),
                "TID.Y" => SReg::Tid(1),
                "TID.Z" => SReg::Tid(2),
                "CTAID.X" => SReg::CtaId(0),
                "CTAID.Y" => SReg::CtaId(1),
                "CTAID.Z" => SReg::CtaId(2),
                "LANEID" => SReg::LaneId,
                "CgaCtaId" => SReg::ClusterCtaRank,
                "CgaSize" => SReg::ClusterNCtaRank,
                "LTMASK" => SReg::LaneMaskLt,
                "LEMASK" => SReg::LaneMaskLe,
                "GEMASK" => SReg::LaneMaskGe,
                "GTMASK" => SReg::LaneMaskGt,
                "EQMASK" => SReg::LaneMaskEq,
                "SWINHI" => SReg::SharedWindowHi,
                _ => bail!(self.err(format!("special register {s} not modeled"))),
            };
            return Ok(Val::SReg(v));
        }
        let (base, n) = match s.strip_suffix(".64") {
            Some(b) => (b, 2u8),
            None => (s, (bytes / 4).max(1)),
        };
        if let Some((file, idx)) = Self::reg_name(base) {
            if idx == RZ && file == File::R || idx == URZ && file == File::UR {
                return Ok(Val::Imm(0));
            }
            return Ok(Val::Reg(Reg { file, idx }, n));
        }
        if let Some(h) = base.strip_prefix("0x") {
            return u64::from_str_radix(h, 16)
                .map(Val::Imm)
                .map_err(|_| self.err("bad imm"));
        }
        if let Ok(v) = base.parse::<i64>() {
            return Ok(Val::Imm(v as u64));
        }
        bail!(self.err(format!("bad operand {s}")))
    }

    /// Source value with an optional `-` (negate; not on hex immediates, which `val` folds) or
    /// `~` (complement) modifier: (value, neg, not).
    fn modval(&self, s: &str, bytes: u8) -> Result<(Val, bool, bool)> {
        let s = s.trim();
        if let Some(r) = s.strip_prefix('~') {
            return Ok((self.val(r, bytes)?, false, true));
        }
        match s.strip_prefix('-').filter(|r| !r.starts_with("0x")) {
            Some(r) => Ok((self.val(r, bytes)?, true, false)),
            None => Ok((self.val(s, bytes)?, false, false)),
        }
    }

    /// f32 source of a float instruction: an immediate prints as a decimal float (`3`,
    /// `1.00000000000000000000e+09`, `-0.5`, `+INF`).
    fn fval(&self, s: &str) -> Result<Val> {
        let s = s.trim();
        if Self::reg_name(s.trim_end_matches(".64")).is_some() || s.starts_with("c[") {
            return self.val(s, 4);
        }
        if s.trim_start_matches(['+', '-']) == "QNAN" {
            // nvdisasm prints every NaN immediate alike: take its bits from the encoding.
            let lo = self
                .l
                .lo
                .ok_or_else(|| self.err("NaN immediate without the encoding"))?;
            return Ok(Val::Imm(lo >> 32));
        }
        let v: f64 = match s.trim_start_matches('+') {
            "INF" => f64::INFINITY,
            "-INF" => f64::NEG_INFINITY,
            x => x
                .parse()
                .map_err(|_| self.err(format!("bad f32 operand {s}")))?,
        };
        let v = v as f32;
        if v as f64 != s.trim_start_matches('+').parse::<f64>().unwrap_or(v as f64) {
            bail!(self.err(format!("f32 immediate {s} is not exact")));
        }
        Ok(Val::Imm(v.to_bits() as u64))
    }

    /// f32 source with `-`, `|..|` modifiers: (value, neg, abs). Immediates carry their sign.
    fn fmodval(&self, s: &str) -> Result<(Val, bool, bool)> {
        let s = s.trim().trim_end_matches(".reuse");
        let (neg, r) = match s.strip_prefix('-') {
            Some(r) if !r.starts_with(|c: char| c.is_ascii_digit() || c == 'I' || c == 'Q') => {
                (true, r)
            }
            _ => (false, s),
        };
        match r.strip_prefix('|').and_then(|x| x.strip_suffix('|')) {
            Some(x) => Ok((self.fval(x)?, neg, true)),
            None => Ok((self.fval(r)?, neg, false)),
        }
    }

    /// An f64 source: a register pair, `c[..]` (8 bytes) or a decimal immediate, with `-x` /
    /// `|x|`.
    fn dmodval(&self, s: &str) -> Result<(Val, bool, bool)> {
        let s = s.trim().trim_end_matches(".reuse");
        let (neg, r) = match s.strip_prefix('-') {
            Some(r) if !r.starts_with(|c: char| c.is_ascii_digit() || c == 'I' || c == 'Q') => {
                (true, r)
            }
            _ => (false, s),
        };
        let (abs, r) = match r.strip_prefix('|').and_then(|x| x.strip_suffix('|')) {
            Some(x) => (true, x),
            None => (false, r),
        };
        let v = if r == "RZ" {
            Val::Imm(0)
        } else if Self::reg_name(r).is_some() {
            let Val::Reg(reg, _) = self.val(r, 4)? else {
                bail!(self.err("bad f64 register"));
            };
            if reg.file == File::R && reg.idx == 255 {
                Val::Imm(0) // RZ
            } else {
                Val::Reg(reg, 2)
            }
        } else if r.starts_with("c[") {
            self.val(r, 8)?
        } else {
            let x: f64 = match r.trim_start_matches('+') {
                "INF" => f64::INFINITY,
                "-INF" => f64::NEG_INFINITY,
                t => t
                    .parse()
                    .map_err(|_| self.err(format!("bad f64 operand {r}")))?,
            };
            Val::Imm(x.to_bits())
        };
        Ok((v, neg, abs))
    }

    /// `[R2+UR8+0x10]`, `[UR9]`, `desc[UR12][R2.64]`.
    fn addr(&self, s: &str) -> Result<Addr> {
        let s = s.trim();
        let inner = if let Some(r) = s.strip_prefix("desc[") {
            let close = r.find(']').ok_or_else(|| self.err("bad desc"))?;
            &r[close + 1..]
        } else {
            s
        };
        let inner = inner
            .strip_prefix('[')
            .and_then(|x| x.strip_suffix(']'))
            .ok_or_else(|| self.err(format!("bad address {s}")))?;
        let mut a = Addr {
            base: None,
            base2: None,
            off: 0,
        };
        for part in inner.split('+') {
            let part = part.trim();
            if let Some(h) = part.strip_prefix("0x") {
                a.off += i64::from_str_radix(h, 16).map_err(|_| self.err("bad offset"))?;
            } else if let Some(h) = part.strip_prefix("-0x") {
                a.off -= i64::from_str_radix(h, 16).map_err(|_| self.err("bad offset"))?;
            } else {
                let v = self.val(part, 4)?;
                if matches!(v, Val::Imm(0)) {
                    continue; // RZ / URZ
                }
                if a.base.is_none() {
                    a.base = Some(v);
                } else {
                    a.base2 = Some(v);
                }
            }
        }
        Ok(a)
    }

    /// `tmem[UR4+0x10]` / `gdesc[UR4]` / `idesc[UR17]` -> (register value, offset).
    fn bracket(&self, s: &str, prefix: &str, n: u8) -> Result<(Val, u32)> {
        let inner = s
            .trim()
            .strip_prefix(prefix)
            .and_then(|x| x.strip_prefix('['))
            .and_then(|x| x.strip_suffix(']'))
            .ok_or_else(|| self.err(format!("expected {prefix}[..], got {s}")))?;
        let (r, off) = match inner.split_once('+') {
            Some((r, o)) => (r, hex(o).ok_or_else(|| self.err("bad offset"))?),
            None => (inner, 0),
        };
        let v = self.val(r, 4 * n)?;
        let v = match v {
            Val::Reg(reg, _) => Val::Reg(reg, n),
            other => other,
        };
        Ok((v, off))
    }

    fn ureg_at(&self, s: &str, k: u16, n: u8) -> Result<Val> {
        let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
        let (file, idx) =
            Self::reg_name(inner).ok_or_else(|| self.err(format!("bad tuple base {s}")))?;
        if file != File::UR {
            bail!(self.err("expected a uniform register tuple"));
        }
        if idx == URZ {
            return Ok(Val::Imm(0));
        }
        Ok(Val::Reg(Reg { file, idx: idx + k }, n))
    }

    /// Skip leading predicate operands (carry/second outputs); they must be PT for now.
    fn skip_preds(&self, from: usize) -> Result<usize> {
        let mut i = from;
        while i < self.l.operands.len() && Self::is_pred_tok(&self.l.operands[i]) {
            if self.l.operands[i] != "PT" && self.l.operands[i] != "UPT" {
                bail!(self.err("predicate side output not modeled yet"));
            }
            i += 1;
        }
        Ok(i)
    }
}

fn parse_half(s: &str) -> Option<u16> {
    let v: f64 = s.trim().parse().ok()?;
    let h = f64_to_f16_exact(v)?;
    Some(h)
}

/// Encode an f64 that is exactly representable in f16 (immediates are always exact).
fn f64_to_f16_exact(v: f64) -> Option<u16> {
    if v == 0.0 {
        return Some(if v.is_sign_negative() { 0x8000 } else { 0 });
    }
    let sign = if v < 0.0 { 0x8000u16 } else { 0 };
    let a = v.abs();
    for bits in 1u16..0x7c00 {
        // Brute force is fine: immediates are rare and the space is tiny.
        if f16_to_f64(bits) == a {
            return Some(sign | bits);
        }
    }
    None
}

pub fn f16_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    s * match e {
        0 => m * 2f64.powi(-24),
        31 => f64::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

// ASSUME(L-ROSETTA): see ASSUMPTIONS.md
fn lower_line(
    l: &SassLine,
    labels: &HashMap<String, u32>,
    addr_to_idx: &HashMap<u32, usize>,
    meta: &CubinMeta,
    plain_exit: Option<usize>,
) -> Result<Inst> {
    let lw = Lw { l };
    let parts: Vec<&str> = l.opcode.split('.').collect();
    let base = parts[0];
    let mods = &parts[1..];
    let has = |m: &str| mods.contains(&m);
    let guard = match &l.guard {
        Some(g) => lw.pred(g)?,
        None => Pred::TRUE,
    };
    let mut uniform = base.starts_with('U') || matches!(base, "S2UR" | "LDCU" | "VOTEU");
    let op = match base {
        "NOP" => Op::Nop,
        // Tensor-map prefetch (prefetch.tensormap): a cache hint.
        "UTMACCTL" if has("PF") => Op::Nop,
        // fence.proxy.tensormap::generic.acquire: ptxas emits `CCTL.E.C.LDCU.IV.DEEP [a]`
        // (drop the constant-cache copy) then `UTMACCTL.IV [a]` (the TMA unit's).
        "UTMACCTL" if mods == ["IV"] => {
            // A 64-bit generic address in a uniform register pair.
            let mut addr = lw.addr(lw.op(0)?)?;
            if let Some(Val::Reg(r, 1)) = addr.base {
                addr.base = Some(Val::Reg(r, 2));
            }
            Op::TmapFence { addr }
        }
        "CCTL" if mods == ["E", "C", "LDCU", "IV", "DEEP"] => Op::Nop,
        // NANOSLEEP(.SYNCS) is a timing hint (try_wait suspend); no semantic effect.
        // ASSUME(E-NANOSLEEP): see ASSUMPTIONS.md
        "NANOSLEEP" => Op::Nop,
        "EXIT" if l.operands.is_empty() => Op::Exit,
        // `@P0 EXIT P2` (torch's reduce_kernel: `if (mult != 0 && threadIdx.x != 0) return`):
        // the threads where both hold exit, as `@P0 BRA P2` branches (ASSUME(S-BRA-COND)).
        // Lowered as that branch to a plain EXIT of the function.
        "EXIT" if l.operands.len() == 1 && mods.is_empty() => {
            let target = plain_exit
                .ok_or_else(|| lw.err("EXIT with a condition, no plain EXIT to branch to"))?;
            return Ok(Inst {
                guard,
                op: Op::Bra {
                    target,
                    any: false,
                    tid0_falls: false,
                    cond: Some(lw.pred(lw.op(0)?)?),
                    div: None,
                },
                uniform: false,
                loc: l.addr,
                text: l.text.clone(),
            });
        }
        // griddepcontrol.wait / launch_dependents (probes/pdl). ASSUME(E-PDL)
        "ACQBULK" => Op::GridDepWait,
        "PREEXIT" => Op::GridDepLaunch,
        "MOV" | "UMOV" => Op::Mov {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
        },
        "LDC" | "LDCU" => {
            let bytes = if has("128") {
                16
            } else if has("64") {
                8
            } else if has("U8") || has("S8") {
                if has("S8") {
                    bail!(lw.err("sign-extending LDC not modeled"));
                }
                1
            } else if has("U16") || has("S16") {
                if has("S16") {
                    bail!(lw.err("sign-extending LDC not modeled"));
                }
                2
            } else {
                4
            };
            // `c[0x2][R6+0xc]`, `c[0x0][UR4+0x2b4]`: another bank, or a register index.
            let src = lw.op(1)?;
            let inner = src
                .strip_prefix("c[0x")
                .and_then(|r| r.split_once("]["))
                .ok_or_else(|| lw.err("bad cbank"))?;
            let bank = u8::from_str_radix(inner.0, 16).map_err(|_| lw.err("bad cbank"))?;
            let idx = inner.1.trim_end_matches(']');
            if bank == 0 && hex(idx).is_some() {
                Op::Mov {
                    d: lw.dst(lw.op(0)?)?,
                    a: lw.val(src, bytes)?,
                }
            } else {
                let (base, off) = match idx.split_once('+') {
                    Some((r, o)) => (lw.val(r, 4)?, hex(o).ok_or_else(|| lw.err("bad cbank"))?),
                    None if hex(idx).is_some() => (Val::Imm(0), hex(idx).unwrap()),
                    None => (lw.val(idx, 4)?, 0),
                };
                let data = if bank == 0 {
                    vec![]
                } else {
                    meta.banks
                        .get(&bank)
                        .cloned()
                        .ok_or_else(|| lw.err(format!("no constant bank {bank} in the cubin")))?
                };
                Op::LdConst {
                    d: lw.dst(lw.op(0)?)?,
                    bank,
                    base,
                    off,
                    bytes,
                    data,
                }
            }
        }
        "S2R" | "S2UR" => Op::Mov {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
        },
        "IADD3" | "UIADD3" => {
            let wide = has("64");
            let d = lw.dst(lw.op(0)?)?;
            // Carry-out predicates follow the destination.
            let mut i = 1;
            let mut cout = Vec::new();
            while i < l.operands.len() && Lw::is_pred_tok(&l.operands[i]) {
                let o = &l.operands[i];
                if o != "PT" && o != "UPT" {
                    if cout.len() != i - 1 {
                        bail!(lw.err("IADD3 second carry-out without the first"));
                    }
                    cout.push(lw.pdst(o)?);
                }
                i += 1;
            }
            let mut neg = [false; 3];
            let mut not = [false; 3];
            let mut v = Vec::new();
            for k in 0..3 {
                let (mut x, n, c) = lw.modval(lw.op(i + k)?, if wide { 8 } else { 4 })?;
                // The immediate of a .64 add is 32 bits, sign-extended (cuBLAS steps a
                // tensor-map pointer back with `UIADD3.64 UR52, ..., -0x800`). ASSUME(L-ROSETTA)
                if let (true, Val::Imm(v)) = (wide, &x) {
                    x = Val::Imm(*v as u32 as i32 as i64 as u64);
                }
                neg[k] = n;
                not[k] = c;
                v.push(x);
            }
            let mut cin = Vec::new();
            for o in &l.operands[i + 3..] {
                cin.push(lw.pred(o)?);
            }
            if !has("X") && !cin.is_empty() {
                bail!(lw.err("carry-in operands without .X"));
            }
            Op::Iadd3 {
                d,
                a: v[0].clone(),
                b: v[1].clone(),
                c: v[2].clone(),
                neg,
                not,
                wide,
                cout,
                cin,
            }
        }
        // `VIADD R0, R0, -UR5` (torch's radix sort): a + (-b), as IADD3 with a negated input.
        "VIADD" if lw.op(2)?.starts_with("-U") || lw.op(2)?.starts_with("-R") => {
            let (b, neg, not) = lw.modval(lw.op(2)?, 4)?;
            Op::Iadd3 {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                b,
                c: Val::Imm(0),
                neg: [false, neg, false],
                not: [false, not, false],
                wide: false,
                cout: vec![],
                cin: vec![],
            }
        }
        "VIADD" => Op::Int {
            op: IntOp::Add,
            bits: 32,
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: lw.val(lw.op(2)?, 4)?,
        },
        // LEA.HI.SX32 d, a, b, s: the high word comes from a sign-extended to 64 bits.
        // LEA.HI[.X][.SX32] d, a, b, [c,] s[, Pcin]
        "LEA" | "ULEA" if has("HI") => {
            let d = lw.dst(lw.op(0)?)?;
            let cout = match lw.op(1)? {
                p if Lw::is_pred_tok(p) && p != "PT" && p != "UPT" => Some(lw.pdst(p)?),
                _ => None,
            };
            let i = if cout.is_some() { 2 } else { lw.skip_preds(1)? };
            let sx32 = has("SX32");
            let n = if sx32 { 3 } else { 4 };
            let sh = hex(lw.op(i + n - 1)?).ok_or_else(|| lw.err("bad shift"))? as u8;
            let cin = match l.operands.get(i + n) {
                Some(p) if has("X") => Some(lw.pred(p)?),
                None if !has("X") => None,
                _ => bail!(lw.err("LEA.HI carry-in operand / .X mismatch")),
            };
            let (b, neg_b) = match lw.op(i + 1)?.strip_prefix('-').filter(|b| b.contains('R')) {
                Some(b) => (b, true),
                None => (lw.op(i + 1)?, false),
            };
            Op::LeaHi {
                d,
                a: lw.val(lw.op(i)?, 4)?,
                b: lw.val(b, 4)?,
                neg_b,
                c: if sx32 {
                    Val::Imm(0)
                } else {
                    lw.val(lw.op(i + 2)?, 4)?
                },
                shift: sh,
                sx32,
                cin,
                cout,
            }
        }
        "IABS" => Op::Int {
            op: IntOp::Abs,
            bits: 32,
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: Val::Imm(0),
        },
        // USGXT.U32 d, a, n: keep the low n bits (zero-extend); without .U32 sign-extend.
        "SGXT" | "USGXT" if !has("W") => Op::Bfe {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            pos: Val::Imm(0),
            len: lw.val(lw.op(2)?, 4)?,
            signed: !has("U32"),
        },
        // I2F[.S64|.U64|.S16|.U16|.S8|.U8]: the source width selects the bits converted.
        "I2F" | "I2FP" if !has("F64") && !has("F16") && !has("FTZ") => Op::I2F {
            d: lw.dst(lw.op(0)?)?,
            hi: lw.op(1)?.trim_end_matches(".reuse").ends_with(".H1"),
            a: lw.val(
                lw.op(1)?
                    .trim_end_matches(".reuse")
                    .trim_end_matches(".H1")
                    .trim_end_matches(".H0"),
                if has("U64") || has("S64") { 8 } else { 4 },
            )?,
            signed: !mods.iter().any(|m| m.starts_with('U')),
            bits: if has("U64") || has("S64") {
                64
            } else if has("U16") || has("S16") {
                16
            } else if has("U8") || has("S8") {
                8
            } else {
                32
            },
            rnd: if has("RP") {
                Rnd::Rp
            } else if has("RM") {
                Rnd::Rm
            } else if has("RZ") {
                Rnd::Rz
            } else {
                Rnd::Rn
            },
        },
        "F2I" if has("TRUNC") && !has("F64") => Op::F2I {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            signed: !has("U32") && !has("U64"),
            bits: if has("U64") || has("S64") { 64 } else { 32 },
        },
        "F2F" if mods == ["F32", "F64"] || mods == ["F64", "F32"] => {
            let to64 = mods[0] == "F64";
            Op::F2F {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, if to64 { 4 } else { 8 })?,
                to64,
                rnd: Rnd::Rn,
            }
        }
        // HADD2.F32 Rd, -RZ, Ra.H0_H0: an f16 -> f32 conversion.
        "HADD2" if mods == ["F32"] && lw.op(1)? == "-RZ" => {
            let src = lw.op(2)?.replace(".reuse", "");
            let src = src.as_str();
            let (r, hi) = match (src.strip_suffix(".H0_H0"), src.strip_suffix(".H1_H1")) {
                (Some(r), _) => (r, false),
                (_, Some(r)) => (r, true),
                _ => bail!(lw.err("HADD2.F32 operand form not modeled")),
            };
            Op::CvtF32F16 {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(r, 4)?,
                hi,
            }
        }
        // `F2F.BF16.F32 Rd, Ra` (cvt.rn.bf16.f32 into a 16-bit register): round to nearest
        // even, upper half zero; unlike `F2FP.BF16.F32.PACK_AB`, a NaN becomes 0x7fff
        // (fixtures/s_ptx_cute_ops = GB300).
        "F2F" if mods == ["F16", "F32"] || mods == ["BF16", "F32"] => Op::F2fF16 {
            d: lw.dst(lw.op(0)?)?,
            a: lw.fval(lw.op(1)?)?,
            bf16: mods[0] == "BF16",
        },
        // Narrow-format F2FP (N-CVT-LP): PACK_AB[_MERGE_C] from two f32, UNPACK_B_MERGE_C from a
        // packed f16x2 / bf16x2, UNPACK_B decodes.
        "F2FP" if f2fp_lp(mods).is_some() => {
            let (to, from, kind, rnd, sat, relu) = f2fp_lp(mods).unwrap();
            let src = |i: usize| -> Result<(Val, bool)> {
                let o = lw.op(i)?.replace(".reuse", "");
                match o.strip_suffix(".H1") {
                    Some(r) => Ok((lw.val(r, 4)?, true)),
                    None => Ok((lw.val(o.strip_suffix(".H0").unwrap_or(&o), 4)?, false)),
                }
            };
            let d = lw.dst(lw.op(0)?)?;
            match kind {
                F2fpKind::Pack { merge } => {
                    let c = if merge || rnd == LpRnd::Rs {
                        src(3)?.0
                    } else {
                        Val::Imm(0)
                    };
                    Op::CvtPack {
                        d,
                        a: lw.fval(lw.op(1)?)?,
                        b: lw.fval(lw.op(2)?)?,
                        c,
                        half: None,
                        to,
                        rnd,
                        sat,
                        relu,
                    }
                }
                F2fpKind::Half => {
                    let (a, hi) = src(1)?;
                    if hi {
                        bail!(lw.err("F2FP.UNPACK_B_MERGE_C with a .H1 source not modeled"));
                    }
                    Op::CvtPack {
                        d,
                        a,
                        b: Val::Imm(0),
                        c: src(2)?.0,
                        half: Some(from),
                        to,
                        rnd,
                        sat,
                        relu,
                    }
                }
                F2fpKind::Unpack => {
                    let (a, hi) = src(1)?;
                    Op::CvtUnpack {
                        d,
                        a,
                        hi,
                        from,
                        to,
                        relu,
                    }
                }
            }
        }
        "F2FP" if mods == ["F16", "F32", "PACK_AB"] => Op::F2fpF16 {
            d: lw.dst(lw.op(0)?)?,
            a: lw.fval(lw.op(1)?)?,
            b: lw.fval(lw.op(2)?)?,
        },
        // FFMA / FMUL [.RZ|.RM|.RP] [.FTZ] [.SAT]. ASSUME(N-FALU)
        "FFMA" | "FMUL"
            if mods
                .iter()
                .all(|m| matches!(*m, "RZ" | "RM" | "RP" | "FTZ" | "SAT")) =>
        {
            let d = lw.dst(lw.op(0)?)?;
            let mut v = Vec::new();
            let (mut neg, mut abs) = ([false; 3], [false; 3]);
            for k in 1..l.operands.len() {
                let (x, n, a) = lw.fmodval(lw.op(k)?)?;
                v.push(x);
                neg[k - 1] = n;
                abs[k - 1] = a;
            }
            if v.len() != if base == "FFMA" { 3 } else { 2 } {
                bail!(lw.err("operand count"));
            }
            Op::Ffma {
                d,
                a: v[0].clone(),
                b: v[1].clone(),
                c: v.get(2).cloned(),
                neg,
                abs,
                rnd: sass_rnd(mods),
                ftz: has("FTZ"),
                sat: has("SAT"),
            }
        }
        // FMNMX d, a, b, p / FMNMX3 d, a, b, c, p: p = PT selects the minimum, !PT the
        // maximum (ptxas: min.f32 / max.f32). ASSUME(N-FALU)
        "FMNMX" | "FMNMX3" if mods.iter().all(|m| matches!(*m, "FTZ" | "NAN")) => {
            let n = if base == "FMNMX3" { 3 } else { 2 };
            if l.operands.len() != n + 2 {
                bail!(lw.err("operand count"));
            }
            let (mut v, mut neg, mut abs) = (Vec::new(), [false; 3], [false; 3]);
            for k in 0..n {
                let (x, ng, ab) = lw.fmodval(lw.op(1 + k)?)?;
                v.push(x);
                neg[k] = ng;
                abs[k] = ab;
            }
            Op::Fmnmx {
                d: lw.dst(lw.op(0)?)?,
                a: v[0].clone(),
                b: v[1].clone(),
                c: v.get(2).cloned(),
                neg,
                abs,
                min: lw.pred(lw.op(n + 1)?)?,
                ftz: has("FTZ"),
                nan: has("NAN"),
            }
        }
        // FSETP.<cmp>[.FTZ].<AND|OR|XOR> Pd, PT, a, b, Pc. ASSUME(N-FALU)
        // DFMA / DMUL / DADD [.RZ|.RM|.RP]: f64 on register pairs. ASSUME(N-FALU)
        "DFMA" | "DMUL" | "DADD"
            if mods.len() <= 1 && mods.iter().all(|m| matches!(*m, "RZ" | "RM" | "RP" | "RN")) =>
        {
            let rnd = sass_rnd(mods);
            let d = lw.dst(lw.op(0)?)?;
            let n = if base == "DFMA" { 3 } else { 2 };
            let mut v = Vec::new();
            let (mut neg, mut abs) = ([false; 3], [false; 3]);
            for k in 0..n {
                let (x, ng, ab) = lw.dmodval(lw.op(k + 1)?)?;
                v.push(x);
                neg[k] = ng;
                abs[k] = ab;
            }
            match base {
                "DFMA" => Op::Dfma {
                    d,
                    a: v[0].clone(),
                    b: v[1].clone(),
                    c: Some(v[2].clone()),
                    neg,
                    abs,
                    rnd,
                },
                "DMUL" => Op::Dfma {
                    d,
                    a: v[0].clone(),
                    b: v[1].clone(),
                    c: None,
                    neg,
                    abs,
                    rnd,
                },
                _ => Op::Dfma {
                    d,
                    a: v[0].clone(),
                    b: Val::Imm(1f64.to_bits()),
                    c: Some(v[1].clone()),
                    neg: [neg[0], false, neg[1]],
                    abs: [abs[0], false, abs[1]],
                    rnd,
                },
            }
        }
        "DSETP" if (mods == ["MIN", "AND"] || mods == ["MAX", "AND"]) && lw.op(4)? == "PT" => {
            Op::DsetpMinMax {
                d: lw.pdst(lw.op(0)?)?,
                q: match lw.op(1)? {
                    "PT" => None,
                    x => Some(lw.pdst(x)?),
                },
                a: lw.dmodval(lw.op(2)?)?.0,
                b: lw.dmodval(lw.op(3)?)?.0,
                max: mods[0] == "MAX",
            }
        }
        "DSETP" if lw.op(1)? == "PT" => {
            let cmp = match mods.first().copied() {
                Some("F") => FCmp::F,
                Some("LT") => FCmp::Lt,
                Some("EQ") => FCmp::Eq,
                Some("LE") => FCmp::Le,
                Some("GT") => FCmp::Gt,
                Some("NE") => FCmp::Ne,
                Some("GE") => FCmp::Ge,
                Some("NUM") => FCmp::Num,
                Some("NAN") => FCmp::Nan,
                Some("LTU") => FCmp::Ltu,
                Some("EQU") => FCmp::Equ,
                Some("LEU") => FCmp::Leu,
                Some("GTU") => FCmp::Gtu,
                Some("NEU") => FCmp::Neu,
                Some("GEU") => FCmp::Geu,
                Some("T") => FCmp::T,
                _ => bail!(lw.err("DSETP comparison not modeled")),
            };
            let bop = match mods.get(1).copied() {
                Some("AND") => BoolOp::And,
                Some("OR") => BoolOp::Or,
                Some("XOR") => BoolOp::Xor,
                _ => bail!(lw.err("DSETP combiner not modeled")),
            };
            if mods.len() != 2 {
                bail!(lw.err("DSETP modifier not modeled"));
            }
            let (a, na, aa) = lw.dmodval(lw.op(2)?)?;
            let (b, nb, ab) = lw.dmodval(lw.op(3)?)?;
            Op::Dsetp {
                d: lw.pdst(lw.op(0)?)?,
                cmp,
                a,
                b,
                neg: [na, nb],
                abs: [aa, ab],
                c: lw.pred(lw.op(4)?)?,
                bop,
            }
        }
        // `HSETP2.BF16_V2.<cmp>.<bop> Pd, PT, Ra.H0_H0, Rb.H1_H1, Pc` or with an immediate pair
        // (`-INF , -INF`): one bf16 lane per operand (both selectors of an operand name the same
        // half). ASSUME(L-ROSETTA): fixtures/s_routing_forms = GB300.
        "HSETP2" if mods.first() == Some(&"BF16_V2") && mods.len() == 3 => {
            let cmp = match mods[1] {
                "LT" => FCmp::Lt,
                "EQ" => FCmp::Eq,
                "LE" => FCmp::Le,
                "GT" => FCmp::Gt,
                "NE" => FCmp::Ne,
                "GE" => FCmp::Ge,
                "NUM" => FCmp::Num,
                "NAN" => FCmp::Nan,
                "LTU" => FCmp::Ltu,
                "EQU" => FCmp::Equ,
                "LEU" => FCmp::Leu,
                "GTU" => FCmp::Gtu,
                "NEU" => FCmp::Neu,
                "GEU" => FCmp::Geu,
                _ => bail!(lw.err("HSETP2 comparison not modeled")),
            };
            let bop = match mods[2] {
                "AND" => BoolOp::And,
                "OR" => BoolOp::Or,
                "XOR" => BoolOp::Xor,
                _ => bail!(lw.err("HSETP2 combiner not modeled")),
            };
            if lw.op(1)? != "PT" {
                bail!(lw.err("HSETP2 second output not modeled"));
            }
            let half = |o: &str| -> Result<(Val, bool)> {
                let o = o.replace(".reuse", "");
                let (r, hi) = match (o.strip_suffix(".H0_H0"), o.strip_suffix(".H1_H1")) {
                    (Some(r), _) => (r.to_string(), false),
                    (_, Some(r)) => (r.to_string(), true),
                    _ => bail!(lw.err(format!("HSETP2 operand {o} not modeled"))),
                };
                Ok((lw.val(&r, 4)?, hi))
            };
            // A bf16 immediate as nvdisasm prints it (`-INF`, `1`, `0.5`).
            let bf_imm = |o: &str| -> Result<u64> {
                let v = lw.fval(o)?;
                match v {
                    Val::Imm(b) if b & 0xffff == 0 => Ok(b >> 16),
                    _ => bail!(lw.err(format!("HSETP2 immediate {o} not modeled"))),
                }
            };
            let (a, ha) = half(lw.op(2)?)?;
            let (b, hb, c) = if lw.l.operands.len() == 6 {
                let (x, y) = (bf_imm(lw.op(3)?)?, bf_imm(lw.op(4)?)?);
                if x != y {
                    bail!(lw.err("HSETP2 immediate pair with distinct halves not modeled"));
                }
                (Val::Imm(x), false, lw.pred(lw.op(5)?)?)
            } else {
                let (b, hb) = half(lw.op(3)?)?;
                (b, hb, lw.pred(lw.op(4)?)?)
            };
            Op::Hsetp2Bf16 {
                d: lw.pdst(lw.op(0)?)?,
                cmp,
                a,
                b,
                hi: [ha, hb],
                c,
                bop,
            }
        }
        "FSETP" | "FSET" if base == "FSETP" || mods.first() == Some(&"BF") => {
            let bf = base == "FSET";
            let mods = if bf { &mods[1..] } else { mods };
            let has = |m: &str| mods.contains(&m);
            let cmp = match mods.first().copied() {
                Some("F") => FCmp::F,
                Some("LT") => FCmp::Lt,
                Some("EQ") => FCmp::Eq,
                Some("LE") => FCmp::Le,
                Some("GT") => FCmp::Gt,
                Some("NE") => FCmp::Ne,
                Some("GE") => FCmp::Ge,
                Some("NUM") => FCmp::Num,
                Some("NAN") => FCmp::Nan,
                Some("LTU") => FCmp::Ltu,
                Some("EQU") => FCmp::Equ,
                Some("LEU") => FCmp::Leu,
                Some("GTU") => FCmp::Gtu,
                Some("NEU") => FCmp::Neu,
                Some("GEU") => FCmp::Geu,
                Some("T") => FCmp::T,
                _ => bail!(lw.err("FSETP comparison not modeled")),
            };
            let bop = if has("AND") {
                BoolOp::And
            } else if has("OR") {
                BoolOp::Or
            } else if has("XOR") {
                BoolOp::Xor
            } else {
                bail!(lw.err("FSETP combiner not modeled"));
            };
            if mods.len() != 2 + has("FTZ") as usize {
                bail!(lw.err("FSETP modifier not modeled"));
            }
            if bf {
                // `FSET.BF.GT.AND R4, R5, R23, PT` (CUTLASS fmha): 1.0f / 0
                // (fixtures/s_lift_k = GB300). ASSUME(L-ROSETTA)
                let (a, na, aa) = lw.fmodval(lw.op(1)?)?;
                let (b, nb, ab) = lw.fmodval(lw.op(2)?)?;
                Op::Fset {
                    d: lw.dst(lw.op(0)?)?,
                    cmp,
                    a,
                    b,
                    neg: [na, nb],
                    abs: [aa, ab],
                    c: lw.pred(lw.op(3)?)?,
                    bop,
                    ftz: has("FTZ"),
                }
            } else {
                if lw.op(1)? != "PT" {
                    bail!(lw.err("FSETP second output not modeled"));
                }
                let (a, na, aa) = lw.fmodval(lw.op(2)?)?;
                let (b, nb, ab) = lw.fmodval(lw.op(3)?)?;
                Op::Fsetp {
                    d: lw.pdst(lw.op(0)?)?,
                    cmp,
                    a,
                    b,
                    neg: [na, nb],
                    abs: [aa, ab],
                    c: lw.pred(lw.op(4)?)?,
                    bop,
                    ftz: has("FTZ"),
                }
            }
        }
        // FSEL d, a, b, p = SEL on f32 bits (immediates print as floats); `|x|` / `-x`
        // operands (torch's GELU: `FSEL R13, |R14|, R13, P0`) are sign-bit operations.
        "FSEL" if mods.is_empty() => {
            let (a, na, aa) = lw.fmodval(lw.op(1)?)?;
            let (b, nb, ab) = lw.fmodval(lw.op(2)?)?;
            if na || aa || nb || ab {
                Op::Fsel {
                    d: lw.dst(lw.op(0)?)?,
                    a,
                    b,
                    p: lw.pred(lw.op(3)?)?,
                    neg: [na, nb],
                    abs: [aa, ab],
                }
            } else {
                Op::Sel {
                    d: lw.dst(lw.op(0)?)?,
                    a,
                    b,
                    p: lw.pred(lw.op(3)?)?,
                }
            }
        }
        // The range check of ptxas's IEEE division (div.rn.f32). ASSUME(N-FCHK)
        // `FCHK P0, R0, 1.4426950216293334961`: an f32 immediate divisor (CUTLASS fmha).
        "FCHK" if mods.is_empty() => Op::Fchk {
            d: lw.pdst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: lw.fval(lw.op(2)?)?,
        },
        "F2FP" if mods == ["BF16", "F32", "PACK_AB"] => Op::F2fpBf16 {
            d: lw.dst(lw.op(0)?)?,
            a: lw.fval(lw.op(1)?)?,
            b: lw.fval(lw.op(2)?)?,
        },
        "BREV" | "UBREV" => Op::Brev {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
        },
        "VIADDMNMX" if mods.iter().all(|m| *m == "U32") => Op::AddMinMax {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: lw.val(lw.op(2)?.trim_start_matches('-'), 4)?,
            neg_b: lw.op(2)?.starts_with("-R"),
            c: lw.val(lw.op(3)?, 4)?,
            signed: !has("U32"),
            max: match lw.op(4)? {
                "PT" => false,
                "!PT" => true,
                _ => bail!(lw.err("VIADDMNMX with a predicate selector not modeled")),
            },
        },
        // `VIMNMX3 R40, R0, UR5, R33, PT` (cuBLAS gemmSN_NN_kernel): selector PT = min, !PT =
        // max, as VIMNMX (fixtures/s_lift_k = GB300). ASSUME(L-ROSETTA)
        "VIMNMX3" | "UVIMNMX3" if mods.iter().all(|m| *m == "U32") && l.operands.len() == 5 => {
            Op::IntMnmx3 {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                b: lw.val(lw.op(2)?, 4)?,
                c: lw.val(lw.op(3)?, 4)?,
                signed: !has("U32"),
                max: match lw.op(4)? {
                    "PT" | "UPT" => false,
                    "!PT" | "!UPT" => true,
                    _ => bail!(lw.err("VIMNMX3 with a predicate selector not modeled")),
                },
            }
        }
        "P2R" if mods.is_empty() && lw.op(1)? == "PR" => Op::P2r {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(2)?, 4)?,
            mask: hex(lw.op(3)?).ok_or_else(|| lw.err("bad mask"))? & 0x7f,
        },
        // ASSUME(N-MUFU): see ASSUMPTIONS.md
        "MUFU" => {
            let func = match mods.first().copied() {
                Some("EX2") => MufuFunc::Ex2,
                Some("RCP") => MufuFunc::Rcp,
                Some("RSQ") => MufuFunc::Rsq,
                Some("LG2") => MufuFunc::Lg2,
                Some("SIN") => MufuFunc::Sin,
                Some("COS") => MufuFunc::Cos,
                Some("SQRT") => MufuFunc::Sqrt,
                Some("TANH") => MufuFunc::Tanh,
                Some("RCP64H") => MufuFunc::Rcp64h,
                Some("RSQ64H") => MufuFunc::Rsq64h,
                _ => bail!(lw.err("MUFU function not modeled")),
            };
            let fmt = match &mods[1..] {
                [] => MufuFmt::F32,
                ["F16"] => MufuFmt::F16,
                ["BF16"] => MufuFmt::Bf16,
                ["F16x2"] => MufuFmt::F16x2,
                ["BF16x2"] => MufuFmt::Bf16x2,
                _ => bail!(lw.err("MUFU format not modeled")),
            };
            let src = lw.op(1)?.trim().trim_end_matches(".reuse");
            let (a, neg, abs, hi) = match src.strip_suffix(".H1").or(src.strip_suffix(".H0")) {
                Some(r) if matches!(fmt, MufuFmt::F16 | MufuFmt::Bf16) => {
                    (lw.val(r, 4)?, false, false, src.ends_with(".H1"))
                }
                Some(_) => bail!(lw.err("MUFU half selector on a 32-bit format")),
                None if fmt == MufuFmt::F32 => {
                    let (a, neg, abs) = lw.fmodval(src)?;
                    (a, neg, abs, false)
                }
                None => (lw.val(src, 4)?, false, false, false),
            };
            Op::Mufu {
                d: lw.dst(lw.op(0)?)?,
                a,
                func,
                fmt,
                hi,
                neg,
                abs,
                ptx: None,
            }
        }
        "STTM" => {
            let (n, half): (usize, Option<u8>) = match mods {
                [x] if x.starts_with('x') => {
                    (x[1..].parse().map_err(|_| lw.err("STTM count"))?, None)
                }
                [s @ ("16dp32bit_t0_t15" | "16dp32bit_t16_t31"), x] if x.starts_with('x') => (
                    x[1..].parse().map_err(|_| lw.err("STTM count"))?,
                    Some(s.ends_with("t31") as u8),
                ),
                _ => bail!(lw.err("STTM shape not modeled yet (only 32x32b, 16x32bx2 .xN)")),
            };
            let (taddr, off) = lw.bracket(lw.op(0)?, "tmem", 1)?;
            let src = lw.op(1)?;
            let v: Vec<Val> = if src == "RZ" {
                vec![Val::Imm(0); n]
            } else {
                let r = lw.dst(src)?;
                (0..n as u16)
                    .map(|i| {
                        Val::Reg(
                            Reg {
                                file: r.file,
                                idx: r.idx + i,
                            },
                            1,
                        )
                    })
                    .collect()
            };
            Op::TmemSt {
                taddr,
                off,
                v,
                tracked: false,
                half,
            }
        }
        // STSM.16.M88.N[.T] [Ra], Rb: stmatrix.m8n8.xN, registers Rb..Rb+N-1.
        // STSM.16.MT88.N: the transposed form.
        "STSM" if has("16") && (has("M88") || has("MT88")) => {
            uniform = false;
            let n: u16 = match mods.last() {
                Some(m) if m.chars().all(|c| c.is_ascii_digit()) => m.parse().unwrap_or(0),
                _ => 1,
            };
            if !matches!(n, 1 | 2 | 4) {
                bail!(lw.err("STSM count"));
            }
            let r = lw.dst(lw.op(1)?)?;
            Op::StMatrix {
                addr: lw.addr(lw.op(0)?)?,
                v: (0..n)
                    .map(|i| {
                        Val::Reg(
                            Reg {
                                file: r.file,
                                idx: r.idx + i,
                            },
                            1,
                        )
                    })
                    .collect(),
                trans: has("T") || has("MT88"),
            }
        }
        "LEA" | "ULEA" => {
            let d = lw.dst(lw.op(0)?)?;
            let (cout, i) = match lw.op(1)? {
                "PT" | "UPT" => (None, 2),
                p if Lw::is_pred_tok(p) => (Some(lw.pdst(p)?), 2),
                _ => (None, 1),
            };
            if l.operands.len() > i + 3 {
                bail!(lw.err("LEA with a second predicate output not modeled"));
            }
            let sh = hex(lw.op(i + 2)?).ok_or_else(|| lw.err("bad shift"))? as u8;
            // LEA d, a, -b, s = (a << s) - b, which is a multiply-add.
            if let Some(b) = lw.op(i + 1)?.strip_prefix('-').filter(|b| b.contains('R')) {
                if sh >= 32 || cout.is_some() {
                    bail!(lw.err("LEA with a negated operand: shift / carry-out not modeled"));
                }
                Op::Imad {
                    d,
                    a: lw.val(lw.op(i)?, 4)?,
                    b: Val::Imm(1 << sh),
                    c: lw.val(b, 4)?,
                    neg_c: true,
                    not_c: false,
                    wide: false,
                    hi: false,
                    signed: false,
                    cin: None,
                    cout: None,
                }
            } else {
                // `LEA Rd, P, -Ra, Rb, s` (flash attention's 64-bit pointer decrement):
                // b + ~(a << s) + 1, with the carry of that add.
                let (a, neg_a) = match lw.op(i)?.strip_prefix('-').filter(|a| a.contains('R')) {
                    Some(a) => (a, true),
                    None => (lw.op(i)?, false),
                };
                Op::Lea {
                    d,
                    a: lw.val(a, 4)?,
                    b: lw.val(lw.op(i + 1)?, 4)?,
                    shift: sh,
                    cout,
                    neg_a,
                }
            }
        }
        "SHF" | "USHF" => {
            let right = has("R");
            let signed = has("S32") || has("S64");
            Op::Shf {
                d: lw.dst(lw.op(0)?)?,
                lo: lw.val(lw.op(1)?, 4)?,
                sh: lw.val(lw.op(2)?, 4)?,
                hi: lw.val(lw.op(3)?, 4)?,
                right,
                signed,
                high: has("HI"),
            }
        }
        "LOP3" | "ULOP3" => {
            // `LOP3.LUT[.PAND] [Pp,] Rd, a, b, c, lut, Pq`: with a predicate output,
            // P = (result != 0) || Pq, or && Pq with `.PAND` (GB300: probes/lop3_pmode).
            // ASSUME(L-LOP3-P): see ASSUMPTIONS.md
            let (p, o) = if Lw::is_pred_tok(lw.op(0)?) {
                (Some(lw.pdst(lw.op(0)?)?), 1)
            } else {
                (None, 0)
            };
            let lut = hex(lw.op(o + 4)?).ok_or_else(|| lw.err("bad lut"))? as u8;
            let pand = match mods {
                ["LUT"] => false,
                ["LUT", "PAND"] => true,
                _ => bail!(lw.err("LOP3 variant not modeled")),
            };
            let pq = match l.operands.get(o + 5).map(|q| q.as_str()) {
                None | Some("!PT" | "!UPT") if !pand => None,
                Some(q) => Some((lw.pred(q)?, pand)),
                None => bail!(lw.err("LOP3.PAND without a predicate operand")),
            };
            Op::Lop3 {
                d: lw.dst(lw.op(o)?)?,
                a: lw.val(lw.op(o + 1)?, 4)?,
                b: lw.val(lw.op(o + 2)?, 4)?,
                c: lw.val(lw.op(o + 3)?, 4)?,
                lut,
                p,
                pq,
            }
        }
        "PLOP3" | "UPLOP3" => {
            if lw.op(1)? != "PT" && lw.op(1)? != "UPT" {
                bail!(lw.err("PLOP3 second output not modeled"));
            }
            Op::Plop3 {
                d: lw.pdst(lw.op(0)?)?,
                a: lw.pred(lw.op(2)?)?,
                b: lw.pred(lw.op(3)?)?,
                c: lw.pred(lw.op(4)?)?,
                lut: hex(lw.op(5)?).ok_or_else(|| lw.err("bad lut"))? as u8,
            }
        }
        "ISETP" | "UISETP" => {
            if !(has("AND") || has("OR") || has("XOR")) {
                bail!(lw.err("only ISETP.*.AND / .OR / .XOR is modeled"));
            }
            let u = has("U32");
            let cmp = match (mods[0], u) {
                ("EQ", _) => Cmp::Eq,
                ("NE", _) => Cmp::Ne,
                ("LT", true) => Cmp::LtU,
                ("LE", true) => Cmp::LeU,
                ("GT", true) => Cmp::GtU,
                ("GE", true) => Cmp::GeU,
                ("LT", false) => Cmp::LtS,
                ("LE", false) => Cmp::LeS,
                ("GT", false) => Cmp::GtS,
                ("GE", false) => Cmp::GeS,
                (c, _) => bail!(lw.err(format!("compare {c}"))),
            };
            if lw.op(1)? != "PT" && lw.op(1)? != "UPT" {
                bail!(lw.err("ISETP second output not modeled"));
            }
            Op::Setp {
                d: lw.pdst(lw.op(0)?)?,
                cmp,
                bits: 32,
                a: lw.val(lw.op(2)?, 4)?,
                b: lw.val(lw.op(3)?, 4)?,
                and: lw.pred(lw.op(4)?)?,
                or: has("OR"),
                xor: has("XOR"),
                ex: if has("EX") {
                    Some(lw.pred(lw.op(5)?)?)
                } else {
                    None
                },
            }
        }
        // `VOTE.EQ Pd, Pa` (vote.sync.uni.pred): all active lanes agree on Pa.
        "VOTEU" | "VOTE" if mods == ["ALL"] || mods == ["EQ"] => {
            uniform = false; // warp-collective, handled by the executor
            if l.operands.len() != 2 {
                bail!(lw.err("VOTE.ALL / VOTE.EQ form not modeled"));
            }
            Op::VoteAll {
                d: lw.pdst(lw.op(0)?)?,
                a: lw.pred(lw.op(1)?)?,
                uni: mods == ["EQ"],
            }
        }
        "ELECT" => {
            let lane = match lw.op(1)? {
                "URZ" => None,
                s => Some(lw.dst(s)?),
            };
            if lw.op(2)? != "PT" {
                bail!(lw.err("ELECT with a membermask predicate not modeled"));
            }
            Op::Elect {
                d: lw.pdst(lw.op(0)?)?,
                lane,
            }
        }
        // IMAD[.WIDE|.HI][.U32][.X] d, [Pcout,] a, b, c[, Pcin]
        "IMAD" | "UIMAD" => {
            let wide = has("WIDE");
            let hi = has("HI");
            let signed = (wide || hi) && !has("U32");
            let d = lw.dst(lw.op(0)?)?;
            let (cout, i) = match lw.op(1)? {
                "PT" | "UPT" => (None, 2),
                p if Lw::is_pred_tok(p) => (Some(lw.pdst(p)?), 2),
                _ => (None, 1),
            };
            if cout.is_some() && (signed || !(wide || hi)) {
                bail!(lw.err("IMAD carry-out only modeled for .WIDE.U32 / .HI.U32"));
            }
            let cbytes = if wide || hi { 8 } else { 4 };
            let cin = if has("X") {
                Some(lw.pred(lw.op(i + 3)?)?)
            } else {
                None
            };
            let (c, neg_c, not_c) = lw.modval(lw.op(i + 2)?, cbytes)?;
            if (neg_c || not_c) && cbytes == 8 {
                bail!(lw.err("negated 64-bit IMAD addend not modeled"));
            }
            Op::Imad {
                d,
                a: lw.val(lw.op(i)?, 4)?,
                b: lw.val(lw.op(i + 1)?, 4)?,
                c,
                neg_c,
                not_c,
                wide,
                hi,
                signed,
                cin,
                cout,
            }
        }
        "HFMA2" => {
            // Operands: d, a, b, c where c may be two f16 immediates "hi, lo".
            let d = lw.dst(lw.op(0)?)?;
            let (neg_a, a_s) = match lw.op(1)?.strip_prefix('-') {
                Some(r) => (true, r),
                None => (false, lw.op(1)?),
            };
            let a = lw.val(a_s, 4)?;
            let b = lw.val(lw.op(2)?, 4)?;
            let c = if l.operands.len() == 5 {
                match (parse_half(lw.op(3)?), parse_half(lw.op(4)?)) {
                    (Some(hi), Some(lo)) => Val::Imm(((hi as u64) << 16) | lo as u64),
                    // `+QNAN` (any NaN prints alike): the pair from the encoding's immediate.
                    _ => Val::Imm(l.lo.ok_or_else(|| lw.err("bad f16 imm"))? >> 32 & 0xffff_ffff),
                }
            } else {
                lw.val(lw.op(3)?, 4)?
            };
            if mods.iter().any(|m| *m != "MMA") {
                bail!(lw.err("HFMA2 modifier not modeled"));
            }
            Op::Hfma2 { d, a, b, c, neg_a }
        }
        "FFMA2" | "FMUL2" | "FADD2" if mods.is_empty() => {
            let f2 = |s: &str| -> Result<(Val, F2Lanes)> {
                let (r, l) = if let Some(r) = s.strip_suffix(".F32x2.HI_LO") {
                    (r, F2Lanes::Pair)
                } else if let Some(r) = s.strip_suffix(".F32x2.LO_HI") {
                    (r, F2Lanes::Swap)
                } else if let Some(r) = s.strip_suffix(".F32") {
                    (r, F2Lanes::Bcast)
                } else if s.parse::<f64>().is_ok() || s.starts_with("0x") || s.contains("INF") {
                    // An f32 immediate (`FMUL2 R4, R2.F32x2.HI_LO, 0.5`), broadcast to both
                    // lanes: its bits are the encoding's word 32..63 (fixtures/c_triton_gelu_bf16
                    // = GB300).
                    let imm =
                        l.lo.ok_or_else(|| lw.err("FFMA2 immediate without encoding"))? >> 32
                            & 0xffff_ffff;
                    return Ok((Val::Imm(imm), F2Lanes::Bcast));
                } else {
                    bail!(lw.err(format!("FFMA2 operand {s} not modeled")))
                };
                let n = if l == F2Lanes::Bcast { 4 } else { 8 };
                Ok((lw.val(r, n)?, l))
            };
            // `-R16.F32`: the operand negated (fixtures/s_lift_k = GB300). ASSUME(L-ROSETTA)
            let mut neg = [false; 3];
            let mut f2n = |k: usize, s: &str| -> Result<(Val, F2Lanes)> {
                let s = match s.strip_prefix('-') {
                    Some(r) if r.starts_with('R') => {
                        neg[k] = true;
                        r
                    }
                    _ => s,
                };
                f2(s)
            };
            if base == "FADD2" {
                // FADD2 d, a, c: a*1 + c, one rounding to nearest per lane: the sum (N-FALU).
                let (d, a, c) = (lw.dst(lw.op(0)?)?, f2n(0, lw.op(1)?)?, f2n(2, lw.op(2)?)?);
                Op::Ffma2 {
                    d,
                    a,
                    b: (Val::Imm(0x3f80_0000), F2Lanes::Bcast),
                    c,
                    neg,
                }
            } else {
                let (d, a, b) = (lw.dst(lw.op(0)?)?, f2n(0, lw.op(1)?)?, f2n(1, lw.op(2)?)?);
                // FMUL2: a*b + (-0), which is the rounded product (round to nearest).
                let c = if base == "FMUL2" {
                    (Val::Imm(0x8000_0000), F2Lanes::Bcast)
                } else {
                    f2n(2, lw.op(3)?)?
                };
                Op::Ffma2 { d, a, b, c, neg }
            }
        }
        // ASSUME(N-FADD, N-FALU): see ASSUMPTIONS.md
        "FADD" => {
            if mods
                .iter()
                .any(|m| !matches!(*m, "RZ" | "RM" | "RP" | "FTZ" | "SAT"))
            {
                bail!(lw.err("FADD modifier not modeled"));
            }
            let (a, na, aa) = lw.fmodval(lw.op(1)?)?;
            let (b, nb, ab) = lw.fmodval(lw.op(2)?)?;
            Op::Fadd {
                d: lw.dst(lw.op(0)?)?,
                a,
                b,
                rnd: sass_rnd(mods),
                neg: [na, nb],
                abs: [aa, ab],
                ftz: has("FTZ"),
                sat: has("SAT"),
            }
        }
        // VIMNMX[.U32] d, PT, PT, a, b, p: p = PT selects the minimum, !PT the maximum (as
        // VIADDMNMX); the two predicate outputs must be unused.
        "VIMNMX" | "UVIMNMX" if mods.iter().all(|m| *m == "U32") => {
            if lw.op(1)?.trim_start_matches('U') != "PT"
                || lw.op(2)?.trim_start_matches('U') != "PT"
            {
                bail!(lw.err("VIMNMX predicate outputs not modeled"));
            }
            let u = has("U32");
            let max = match lw.op(5)? {
                "PT" | "UPT" => false,
                "!PT" | "!UPT" => true,
                _ => bail!(lw.err("VIMNMX with a predicate selector not modeled")),
            };
            Op::Int {
                op: match (u, max) {
                    (true, false) => IntOp::MinU,
                    (true, true) => IntOp::MaxU,
                    (false, false) => IntOp::MinS,
                    (false, true) => IntOp::MaxS,
                },
                bits: 32,
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(3)?, 4)?,
                b: lw.val(lw.op(4)?, 4)?,
            }
        }
        "BMSK" | "UBMSK" if mods.is_empty() || mods == ["W"] => Op::Int {
            op: if has("W") {
                IntOp::BmskWrap
            } else {
                IntOp::Bmsk
            },
            bits: 32,
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: lw.val(lw.op(2)?, 4)?,
        },
        "POPC" | "UPOPC" if mods.is_empty() => Op::Int {
            op: IntOp::Popc,
            bits: 32,
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: Val::Imm(0),
        },
        "R2P" if mods.is_empty() && lw.op(0)? == "PR" => Op::R2p {
            a: lw.val(lw.op(1)?, 4)?,
            mask: hex(lw.op(2)?).ok_or_else(|| lw.err("bad mask"))? & 0x7f,
        },
        "R2UR" => {
            uniform = false; // collective: reads the value from the active lanes
            if l.operands.len() == 3 {
                Op::R2Ur {
                    p: Some(lw.pdst(lw.op(0)?)?),
                    d: lw.dst(lw.op(1)?)?,
                    a: lw.val(lw.op(2)?, 4)?,
                }
            } else {
                Op::R2Ur {
                    p: None,
                    d: lw.dst(lw.op(0)?)?,
                    a: lw.val(lw.op(1)?, 4)?,
                }
            }
        }
        "VOTEU" | "VOTE" if has("ANY") => {
            uniform = false;
            if l.operands.len() == 2 {
                // VOTEU.ANY UPd, Pa: predicate result = any active lane has Pa.
                Op::VoteAny {
                    d: lw.pdst(lw.op(0)?)?,
                    a: lw.pred(lw.op(1)?)?,
                }
            } else {
                if lw.op(1)? != "UPT" && lw.op(1)? != "PT" {
                    bail!(lw.err("VOTE.ANY predicate output not modeled"));
                }
                Op::VoteAny {
                    d: lw.dst(lw.op(0)?)?,
                    a: lw.pred(lw.op(2)?)?,
                }
            }
        }
        "UFLO" | "FLO" if has("U32") => Op::Flo {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            sh: has("SH"),
        },
        // `REDUX[.OR|.XOR|.SUM] URd, Ra` / `CREDUX.MIN|.MAX[.S32] URd, Ra` (redux.sync):
        // reduction over the active lanes into a uniform register (GB300: fixtures/s_warpcoll).
        "REDUX" | "CREDUX" => {
            uniform = false;
            let op = match (base, mods) {
                ("REDUX", []) => ReduxOp::And,
                ("REDUX", ["OR"]) => ReduxOp::Or,
                ("REDUX", ["XOR"]) => ReduxOp::Xor,
                ("REDUX", ["SUM"]) => ReduxOp::Sum,
                ("CREDUX", ["MIN"]) => ReduxOp::MinU,
                ("CREDUX", ["MAX"]) => ReduxOp::MaxU,
                ("CREDUX", ["MIN", "S32"]) => ReduxOp::MinS,
                ("CREDUX", ["MAX", "S32"]) => ReduxOp::MaxS,
                _ => bail!(lw.err(format!("unsupported SASS opcode `{}`", l.opcode))),
            };
            Op::Redux {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                op,
            }
        }
        // `ATOMS.CAST.SPIN P, [Ra], Rc, Rc+1`: CAS (compare Rc, new Rc+1), P = swapped.
        // `ATOMS.CAST.SPIN.64 P, [Ra], Rc, Rc+2`: the 64-bit form (compare Rc:Rc+1, new
        // Rc+2:Rc+3; ptxas's loop for a shared 64-bit min / max).
        "ATOMS" if mods == ["CAST", "SPIN"] || mods == ["CAST", "SPIN", "64"] => {
            let wide = mods.len() == 3;
            let r = lw.dst(lw.op(2)?)?;
            let new = lw.dst(lw.op(3)?)?;
            if new.file != r.file || new.idx != r.idx + if wide { 2 } else { 1 } {
                bail!(lw.err("CAS operands not a register pair"));
            }
            Op::Atom {
                op: AtomOp::CasSpin,
                space: Space::Shared,
                bytes: if wide { 8 } else { 4 },
                addr: lw.addr(lw.op(1)?)?,
                v: Val::Reg(r, 2),
                d: None,
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: Scope::Cta,
                },
                p: Some(lw.pdst(lw.op(0)?)?),
                new: wide.then_some(Val::Reg(new, 2)),
            }
        }
        // `ATOMS.CAS[.64] Rd, [Ra], Rb, Rc` (atom.shared.cas): compare Rb, new Rc, Rd = old.
        // ASSUME(L-ROSETTA): fixtures/s_atom_ext = GB300 (with .CAST.SPIN.64, the 64-bit and
        // MIN / MAX forms of ATOM / ATOMG / REDG below).
        "ATOMS" if mods == ["CAS"] || mods == ["CAS", "64"] => {
            let bytes = if mods.len() == 2 { 8 } else { 4 };
            let d = match lw.op(0)? {
                "RZ" => None,
                x => Some(lw.dst(x)?),
            };
            Op::Atom {
                op: AtomOp::Cas,
                space: Space::Shared,
                bytes,
                addr: lw.addr(lw.op(1)?)?,
                v: lw.val(lw.op(2)?, bytes)?,
                d,
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: Scope::Cta,
                },
                p: None,
                new: Some(lw.val(lw.op(3)?, bytes)?),
            }
        }
        // Generic atomics (`atomicAdd(float*)` on an unknown space):
        // `ATOM.E.ADD.F32.FTZ.RN.STRONG.GPU P, RZ, [Ra.64], Rb` (P = performed: global),
        // `ATOM.E.CAST.SPIN PT, Rd, [Ra], Rc, Rc+1` (Rd = 1 when swapped).
        "ATOM"
            if has("E")
                && (has("F32") && has("ADD") && has("FTZ") && has("RN")
                    || has("CAST") && has("SPIN")) =>
        {
            let cas = has("CAST");
            let p = match lw.op(0)? {
                "PT" => None,
                x => Some(lw.pdst(x)?),
            };
            let d = match lw.op(1)? {
                "RZ" => None,
                x => Some(lw.dst(x)?),
            };
            let mut addr = lw.addr(lw.op(2)?)?;
            if let Some(Val::Reg(r, 1)) = addr.base {
                if r.file == File::R {
                    addr.base = Some(Val::Reg(r, 2));
                }
            }
            Op::Atom {
                op: if cas {
                    AtomOp::CasSpin
                } else {
                    AtomOp::FAddFtz
                },
                space: Space::Generic,
                bytes: 4,
                addr,
                v: if cas {
                    Val::Reg(lw.dst(lw.op(3)?)?, 2)
                } else {
                    lw.val(lw.op(3)?, 4)?
                },
                d,
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: sass_scope(mods).unwrap_or(Scope::Gpu),
                },
                p,
                new: None,
            }
        }
        // `ATOM.E.CAS.STRONG.SYS PT, Rd, [Ra], Rb, Rb+1` (atom.cas.b32 on a generic address:
        // compare Rb, new Rb+1, Rd = old value). The .release / .acquire of atom.acq_rel are
        // the separate MEMBAR / CCTL.IVALL around it (torch symmetric memory's signals).
        // `.64`: compare Rb:Rb+1, new Rc:Rc+1 (`ATOM.E.CAS.64 P2, Rd, [Ra], Rb, Rc`; a generic
        // ATOM's predicate result says the address was global, as for the other generic
        // atomics).
        "ATOM" | "ATOMG" if has("E") && has("CAS") && !mods.iter().any(|m| m.starts_with('F')) => {
            if let Some(m) = mods.iter().find(|m| !atom_mod_known(m)) {
                bail!(lw.err(format!("{base}.CAS: modifier .{m} not modeled")));
            }
            let wide = has("64");
            let bytes: u8 = if wide { 8 } else { 4 };
            let p = match lw.op(0)? {
                "PT" => None,
                x if base == "ATOM" => Some(lw.pdst(x)?),
                _ => bail!(lw.err("CAS with a predicate result not modeled")),
            };
            let d = match lw.op(1)? {
                "RZ" => None,
                x => Some(lw.dst(x)?),
            };
            let mut addr = lw.addr(lw.op(2)?)?;
            if let Some(Val::Reg(r, 1)) = addr.base {
                if r.file == File::R {
                    addr.base = Some(Val::Reg(r, 2));
                }
            }
            let cmp = lw.dst(lw.op(3)?)?;
            let new = lw.dst(lw.op(4)?)?;
            if !wide && (new.file != cmp.file || new.idx != cmp.idx + 1) {
                bail!(lw.err("CAS operands not a register pair"));
            }
            Op::Atom {
                op: AtomOp::Cas,
                space: if base == "ATOM" {
                    Space::Generic
                } else {
                    Space::Global
                },
                bytes,
                addr,
                v: Val::Reg(cmp, 2),
                d,
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: sass_scope(mods).unwrap_or(Scope::Gpu),
                },
                p,
                new: wide.then_some(Val::Reg(new, 2)),
            }
        }
        // Generic integer atomics: `ATOM.E.ADD.STRONG.SYS PT, RZ, [Ra.64], Rb` (DeepEP's
        // barrier signals). The predicate (if not PT) says the address was global.
        "ATOM"
            if has("E")
                && !has("CAS")
                && !mods.iter().any(|m| m.starts_with('F'))
                && (has("ADD")
                    || has("OR")
                    || has("AND")
                    || has("XOR")
                    || has("EXCH")
                    || has("MIN")
                    || has("MAX")) =>
        {
            let op = if has("ADD") {
                AtomOp::Add
            } else if has("OR") {
                AtomOp::Or
            } else if has("AND") {
                AtomOp::And
            } else if has("XOR") {
                AtomOp::Xor
            } else if has("EXCH") {
                AtomOp::Exch
            } else {
                if let Some(m) = mods.iter().find(|m| !atom_mod_known(m)) {
                    bail!(lw.err(format!("ATOM min / max: modifier .{m} not modeled")));
                }
                min_max_op(mods)
            };
            let bytes = if has("64") || has("S64") { 8 } else { 4 };
            let p = match lw.op(0)? {
                "PT" => None,
                x => Some(lw.pdst(x)?),
            };
            let d = match lw.op(1)? {
                "RZ" => None,
                x => Some(lw.dst(x)?),
            };
            let mut addr = lw.addr(lw.op(2)?)?;
            if let Some(Val::Reg(r, 1)) = addr.base {
                if r.file == File::R {
                    addr.base = Some(Val::Reg(r, 2));
                }
            }
            Op::Atom {
                op,
                space: Space::Generic,
                bytes,
                addr,
                v: lw.val(lw.op(3)?, bytes)?,
                d,
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: sass_scope(mods).unwrap_or(Scope::Gpu),
                },
                p,
                new: None,
            }
        }
        "QSPC" if has("E") && (has("S") || has("D")) && lw.op(1)? == "RZ" => {
            let mut addr = lw.addr(lw.op(2)?)?;
            if let Some(Val::Reg(r, 1)) = addr.base {
                if r.file == File::R {
                    addr.base = Some(Val::Reg(r, 2));
                }
            }
            Op::Qspc {
                p: lw.pdst(lw.op(0)?)?,
                addr,
            }
        }
        // `ATOMS.POPC.INC.32 RZ, [Ra]`: ptxas's form of `red.shared.add.u32 [a], 1` (the warp's
        // lanes on one address add their count at once): each active lane adds 1.
        // ASSUME(L-ROSETTA): fixtures/s_routing_forms = GB300.
        "ATOMS" if mods == ["POPC", "INC", "32"] && lw.op(0)? == "RZ" => Op::Atom {
            op: AtomOp::Add,
            space: Space::Shared,
            bytes: 4,
            addr: lw.addr(lw.op(1)?)?,
            v: Val::Imm(1),
            d: None,
            sync: MemSync {
                sem: Sem::Relaxed,
                scope: Scope::Cta,
            },
            p: None,
            new: None,
        },
        "ATOMS" => {
            let op = match mods {
                ["OR"] => AtomOp::Or,
                ["AND"] => AtomOp::And,
                ["XOR"] => AtomOp::Xor,
                ["ADD"] => AtomOp::Add,
                ["EXCH"] => AtomOp::Exch,
                // ASSUME(L-ROSETTA): fixtures/s_lift_k = GB300.
                ["MIN"] | ["MIN", "U32"] => AtomOp::MinU,
                ["MAX"] | ["MAX", "U32"] => AtomOp::MaxU,
                ["MIN", "S32"] => AtomOp::MinS,
                ["MAX", "S32"] => AtomOp::MaxS,
                _ => bail!(lw.err("ATOMS variant not modeled")),
            };
            let d = match lw.op(0)? {
                "RZ" => None,
                x => Some(lw.dst(x)?),
            };
            Op::Atom {
                op,
                space: Space::Shared,
                bytes: 4,
                addr: lw.addr(lw.op(1)?)?,
                v: lw.val(lw.op(2)?, 4)?,
                d,
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: Scope::Cta,
                },
                p: None,
                new: None,
            }
        }
        // `ATOMG.E.ADD.STRONG.GPU PT, Rd, [Ra], Rb` / `REDG.E.ADD.STRONG.SYS [Ra], Rb`. The
        // release / acquire of atom.release / .acquire are separate MEMBAR / CCTL.IVALL.
        "ATOMG" | "REDG" => {
            // `.F32.FTZ.RN`: the fp32 add (Triton's tl.atomic_add on f32).
            let f32_add = has("ADD") && has("F32") && has("FTZ") && has("RN");
            let op = if f32_add {
                AtomOp::FAddFtz
            } else if has("ADD") {
                AtomOp::Add
            } else if has("OR") {
                AtomOp::Or
            } else if has("AND") {
                AtomOp::And
            } else if has("XOR") {
                AtomOp::Xor
            } else if has("EXCH") {
                AtomOp::Exch
            } else if has("MIN") || has("MAX") {
                // `ATOMG.E.MIN[.S32|.64|.S64]`, `REDG.E.MAX.S32` (atom / red .min / .max).
                if let Some(m) = mods.iter().find(|m| !atom_mod_known(m)) {
                    bail!(lw.err(format!("{base} min / max: modifier .{m} not modeled")));
                }
                min_max_op(mods)
            } else {
                bail!(lw.err("ATOMG/REDG variant not modeled"));
            };
            let min_max = matches!(
                op,
                AtomOp::MinU | AtomOp::MaxU | AtomOp::MinS | AtomOp::MaxS
            );
            if mods.iter().any(|m| {
                matches!(*m, "F16x2" | "BF16x2" | "CAS")
                    || (matches!(*m, "S32" | "S64") && !min_max)
                    || (*m == "F32" && !f32_add)
            }) {
                bail!(lw.err("ATOMG/REDG variant not modeled"));
            }
            let bytes = if has("64") || has("S64") { 8 } else { 4 };
            let sync = MemSync {
                sem: Sem::Relaxed,
                scope: sass_scope(mods).unwrap_or(Scope::Gpu),
            };
            let wide_addr = |a: Addr| wide_e_addr(a, has("E"));
            if base == "REDG" {
                Op::Atom {
                    op,
                    space: Space::Global,
                    bytes,
                    addr: wide_addr(lw.addr(lw.op(0)?)?),
                    v: lw.val(lw.op(1)?, bytes)?,
                    d: None,
                    sync,
                    p: None,
                    new: None,
                }
            } else {
                // Operand 0 is a predicate destination (PT here).
                if lw.op(0)? != "PT" {
                    bail!(lw.err("ATOMG with a predicate result not modeled"));
                }
                let d = match lw.op(1)? {
                    "RZ" => None,
                    x => Some(lw.dst(x)?),
                };
                Op::Atom {
                    op,
                    space: Space::Global,
                    bytes,
                    addr: wide_addr(lw.addr(lw.op(2)?)?),
                    v: lw.val(lw.op(3)?, bytes)?,
                    d,
                    sync,
                    p: None,
                    new: None,
                }
            }
        }
        "STS" => {
            let bytes = match mods {
                ["U8"] => 1,
                ["U16"] => 2,
                [] => 4,
                ["64"] => 8,
                ["128"] => 16,
                _ => bail!(lw.err("STS width")),
            };
            // ASSUME(S-SMEM-STRONG): see ASSUMPTIONS.md
            Op::St {
                space: Space::Shared,
                bytes,
                addr: lw.addr(lw.op(0)?)?,
                v: lw.val(lw.op(1)?, bytes)?,
                sync: Some(SMEM_RELAXED),
            }
        }
        // PMTRIG: a performance-monitor trigger, no architectural effect.
        "YIELD" | "DEPBAR" | "PMTRIG" => Op::Nop,
        // ASSUME(E-BSYNC): see ASSUMPTIONS.md
        "BSSY" | "BSYNC" => {
            let b = lw
                .op(0)?
                .strip_prefix('B')
                .and_then(|x| x.parse::<u8>().ok())
                .filter(|&b| b < 16)
                .ok_or_else(|| lw.err("bad convergence barrier"))?;
            if l.opcode.starts_with("BSSY") {
                Op::Bssy { b }
            } else {
                Op::Bsync { b }
            }
        }
        // `BREAK.RELIABLE Bb` (torch's vectorized elementwise loops: lanes past the end leave
        // the unrolled loop's barrier, then branch past its BSYNC). ASSUME(E-BSYNC)
        "BREAK" => {
            let b = lw
                .op(0)?
                .strip_prefix('B')
                .and_then(|x| x.parse::<u8>().ok())
                .filter(|&b| b < 16)
                .ok_or_else(|| lw.err("bad convergence barrier"))?;
            Op::Break { b }
        }
        // Invalidates the sync unit's view after mbarrier words were written with plain
        // stores (ptxas's divergent-path lowering of mbarrier.init). The simulator decodes
        // mbarriers from memory, so this has no further effect.
        "SYNCS" if has("CCTL") && has("IVALL") => Op::Nop, // scheduling / convergence hint; scoreboards are checked in scoreboard.rs
        "WARPSYNC" if mods.is_empty() || mods == ["ALL"] || mods == ["COLLECTIVE"] => {
            if has("COLLECTIVE") {
                // `WARPSYNC.COLLECTIVE Rm, `(end)``: `end` must follow the region's ENDCOLLECTIVE.
                let t = lw.op(1)?;
                let lbl = t
                    .trim_start_matches('`')
                    .trim_start_matches('(')
                    .trim_end_matches(')');
                labels
                    .get(lbl)
                    .ok_or_else(|| lw.err(format!("unknown label {lbl}")))?;
            }
            Op::WarpSync {
                mask: if has("ALL") {
                    Val::Imm(0xffff_ffff)
                } else {
                    lw.val(lw.op(0)?, 4)?
                },
                collective: has("COLLECTIVE"),
            }
        }
        // Region end of `WARPSYNC.COLLECTIVE`; `start` is filled in after lowering.
        "ENDCOLLECTIVE" if mods.is_empty() && l.operands.is_empty() => {
            uniform = false;
            Op::EndCollective { start: usize::MAX }
        }
        // `LEPC R20, `(.L_x_1)``: the code address of the label (a return address).
        "LEPC" => {
            let t = lw.op(1)?;
            let lbl = t
                .trim_start_matches('`')
                .trim_start_matches('(')
                .trim_end_matches(')');
            let a = match labels.get(lbl) {
                Some(a) => *a as u64,
                None => hex(t).ok_or_else(|| lw.err(format!("unknown label {lbl}")))? as u64,
            };
            Op::Mov {
                d: lw.dst(lw.op(0)?)?,
                a: Val::Imm(a),
            }
        }
        // `CALL.ABS.NOINC R14`: a call through an address loaded from a relocated constant bank
        // (CUDA_KERNEL_ASSERT -> `__assertfail`).
        "CALL" if has("ABS") && lw.op(0)?.starts_with('`') => {
            // `CALL.ABS.NOINC `(vprintf)`: a call of a named external function (DeepEP's
            // device asserts print before `BPT.TRAP`).
            uniform = false;
            let name = lw.op(0)?.trim_start_matches('`').trim_start_matches('(');
            let name = name.trim_end_matches(')').to_string();
            const EXT: u64 = 0xffff_0000_0000_0000;
            Op::CallAbs {
                target: Val::Imm(EXT),
                syms: vec![(EXT, name)],
            }
        }
        "CALL" if has("ABS") => {
            uniform = false;
            Op::CallAbs {
                target: lw.val(lw.op(0)?, 8)?,
                syms: meta.syms.clone(),
            }
        }
        "CALL" if has("REL") => {
            let t = lw.op(0)?;
            let lbl = t
                .trim_start_matches('`')
                .trim_start_matches('(')
                .trim_end_matches(')');
            let a = *labels
                .get(lbl)
                .ok_or_else(|| lw.err(format!("unknown label {lbl}")))?;
            uniform = false;
            Op::Call {
                target: *addr_to_idx
                    .get(&a)
                    .ok_or_else(|| lw.err("call target outside function"))?,
            }
        }
        "RET" if has("REL") => Op::Ret {
            addr: lw.val(lw.op(0)?.split_whitespace().next().unwrap_or(""), 4)?,
        },
        "BPT" if has("TRAP") => Op::Trap { why: String::new() },
        "UTCATOMSWS" if has("FIND_AND_SET") => Op::TmemAllocRaw {
            ok: lw.pdst(lw.op(0)?)?,
            d: lw.dst(lw.op(1)?)?,
            units: lw.val(lw.op(2)?, 4)?,
            pair: has("2CTA"),
        },
        "STAS" => {
            // `STAS [Ra.64], Rv`: Ra = shared::cluster address, Ra+1 = its mbarrier.
            if !mods.is_empty() {
                bail!(lw.err("STAS variant not modeled"));
            }
            let a = lw.op(0)?.trim_start_matches('[').trim_end_matches(']');
            let r = a
                .strip_suffix(".64")
                .ok_or_else(|| lw.err("STAS wants [Rx.64]"))?;
            let base = lw.dst(r)?;
            Op::StAsync {
                addr: Val::Reg(base, 1),
                mbar: Val::Reg(
                    Reg {
                        file: base.file,
                        idx: base.idx + 1,
                    },
                    1,
                ),
                v: lw.val(lw.op(1)?, 4)?,
            }
        }
        "UTCATOMSWS" if has("AND") => Op::TmemFreeRaw {
            mask: lw.val(lw.op(1)?, 4)?,
        },
        "UVIRTCOUNT" if has("DEALLOC") && has("SMPOOL") => Op::TmemRelinquish,
        // UTCHMMA = kind::f16; UTCQMMA = kind::f8f6f4, or kind::mxf8f6f4 with a sixth operand
        // `tmem[URs]` (URs, URs+1 = the A, B scale-factor addresses); UTCOMMA = kind::mxf4 /
        // mxf4nvf4 (`.BLOCK16`: scale_vec::4X), always with scale factors.
        "UTCHMMA" | "UTCQMMA" | "UTCOMMA" => {
            let pair = mods.contains(&"2CTA");
            let block16 = mods.contains(&"BLOCK16");
            if mods
                .iter()
                .any(|m| !(*m == "2CTA" || *m == "BLOCK16" && base == "UTCOMMA"))
            {
                bail!(lw.err(format!("{base} variant not modeled yet")));
            }
            let scaled = lw.l.operands.len() == 7;
            let kind = match (base, scaled) {
                ("UTCHMMA", false) => MmaKind::F16,
                ("UTCQMMA", false) => MmaKind::F8f6f4,
                ("UTCQMMA", true) => MmaKind::Mxf8f6f4,
                ("UTCOMMA", true) => MmaKind::Mxf4nvf4 { x4: block16 },
                _ => bail!(lw.err(format!("{base} operand form not modeled"))),
            };
            let scale = if scaled {
                let (s, soff) = lw.bracket(lw.op(5)?, "tmem", 1)?;
                let Val::Reg(r, _) = s else {
                    bail!(lw.err("scale-factor tmem operand must be a register pair"));
                };
                if soff != 0 {
                    bail!(lw.err("tmem offset on the scale factors not modeled"));
                }
                let r1 = Reg {
                    file: r.file,
                    idx: r.idx + 1,
                };
                Some((Val::Reg(r, 1), Val::Reg(r1, 1)))
            } else {
                None
            };
            // `UTCHMMA tmem[URa], gdesc[URb], ...`: A in TMEM.
            let a_tmem = lw.op(0)?.starts_with("tmem[");
            let (adesc, aoff) = if a_tmem {
                lw.bracket(lw.op(0)?, "tmem", 1)?
            } else {
                lw.bracket(lw.op(0)?, "gdesc", 2)?
            };
            if aoff != 0 {
                bail!(lw.err("tmem offset on UTCHMMA A not modeled"));
            }
            let (bdesc, _) = lw.bracket(lw.op(1)?, "gdesc", 2)?;
            let (d, doff) = lw.bracket(lw.op(2)?, "tmem", 1)?;
            let (idesc, _) = lw.bracket(lw.op(4)?, "idesc", 1)?;
            if doff != 0 {
                bail!(lw.err("tmem offset on UTCHMMA not modeled"));
            }
            Op::Mma {
                d,
                adesc,
                bdesc,
                idesc,
                enable_d: lw.pred(lw.op(if scaled { 6 } else { 5 })?)?,
                pair,
                disable: vec![],
                a_tmem,
                kind,
                scale,
            }
        }
        // `UTCCP.T.S[.2CTA].4x32dp128bit tmem[URa+off], gdesc[URd]` = tcgen05.cp
        // [.cta_group::2].32x128b.warpx4 (Rosetta: probes/tc_pair_bs).
        "UTCCP" => {
            let pair = mods == ["T", "S", "2CTA", "4x32dp128bit"];
            if mods != ["T", "S", "4x32dp128bit"] && !pair {
                bail!(lw.err("UTCCP variant not modeled yet"));
            }
            let (taddr, off) = lw.bracket(lw.op(0)?, "tmem", 1)?;
            let (sdesc, _) = lw.bracket(lw.op(1)?, "gdesc", 2)?;
            Op::TcCp {
                taddr,
                off,
                sdesc,
                pair,
            }
        }
        "UTCBAR" => {
            // `UTCBAR [mbar], URZ` or `UTCBAR.2CTA.MULTICAST [mbar], URZ, URmask` (mask unshifted).
            // `.MULTICAST` alone: cta_group::1 (same mask operand, checked against ptxas).
            let multicast = mods == ["2CTA", "MULTICAST"] || mods == ["MULTICAST"];
            if lw.op(1)? != "URZ" || !(mods.is_empty() || multicast) {
                bail!(lw.err("UTCBAR variant not modeled yet"));
            }
            let (m, off) = lw.bracket(&format!("x{}", lw.op(0)?), "x", 1)?;
            Op::TcCommit {
                mbar: m,
                off,
                mask: if multicast {
                    Some(lw.val(lw.op(2)?, 4)?)
                } else {
                    None
                },
            }
        }
        "LDTM" => {
            let mut s16x256b = false;
            let mut s16x128b = false;
            let (n, s16x64b, half): (u8, bool, Option<u8>) = match mods {
                [x] if x.starts_with('x') => (
                    x[1..].parse().map_err(|_| lw.err("LDTM count"))?,
                    false,
                    None,
                ),
                ["16dp256bit"] | ["16dp256bit", _] => {
                    s16x256b = true;
                    let x: u8 = match mods.get(1) {
                        Some(x) => x
                            .strip_prefix('x')
                            .and_then(|x| x.parse().ok())
                            .ok_or_else(|| lw.err("LDTM count"))?,
                        None => 1,
                    };
                    (4 * x, false, None)
                }
                ["16dp128bit"] | ["16dp128bit", _] => {
                    s16x128b = true;
                    let x: u8 = match mods.get(1) {
                        Some(x) => x
                            .strip_prefix('x')
                            .and_then(|x| x.parse().ok())
                            .ok_or_else(|| lw.err("LDTM count"))?,
                        None => 1,
                    };
                    (2 * x, false, None)
                }
                ["16dp64bit", x] if x.starts_with('x') => (
                    x[1..].parse().map_err(|_| lw.err("LDTM count"))?,
                    true,
                    None,
                ),
                [s @ ("16dp32bit_t0_t15" | "16dp32bit_t16_t31"), x] if x.starts_with('x') => (
                    x[1..].parse().map_err(|_| lw.err("LDTM count"))?,
                    false,
                    Some(s.ends_with("t31") as u8),
                ),
                _ => {
                    bail!(lw.err("LDTM shape not modeled yet (only 32x32b, 16x64b, 16x32bx2 .xN)"))
                }
            };
            let (taddr, off) = lw.bracket(lw.op(1)?, "tmem", 1)?;
            Op::TmemLd {
                d: lw.dst(lw.op(0)?)?,
                n,
                taddr,
                off,
                tracked: false,
                s16x64b,
                half,
                s16x256b,
                s16x128b,
            }
        }
        // IDP.2A.{LO,HI}.{U16,S16}.{U8,S8} Rd, Ra, Rb, Rc: dp2a (cuDNN's address arithmetic).
        "IDP" if mods.len() == 4 && mods[0] == "2A" => {
            let ty = |m: &str, w: &str| match m.strip_suffix(w) {
                Some("U") => Ok(false),
                Some("S") => Ok(true),
                _ => Err(lw.err("IDP type")),
            };
            Op::Dp2a {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                b: lw.val(lw.op(2)?, 4)?,
                c: lw.val(lw.op(3)?, 4)?,
                hi: match mods[1] {
                    "LO" => false,
                    "HI" => true,
                    _ => bail!(lw.err("IDP.2A half")),
                },
                a_signed: ty(mods[2], "16")?,
                b_signed: ty(mods[3], "8")?,
            }
        }
        // LDL[.LU] Rd, [R1+off] / STL [R1+off], Rv: local memory (register spills).
        "LDL" | "STL" => {
            let bytes = if has("64") {
                8
            } else if has("128") {
                16
            } else if has("U8") || (base == "STL" && has("S8")) {
                // Zero-extended loads; a store writes the low byte / half either way
                // (fixtures/s_lift_k = GB300). ASSUME(L-ROSETTA)
                1
            } else if has("U16") || (base == "STL" && has("S16")) {
                2
            } else if has("S8") || has("S16") {
                bail!(lw.err("sign-extending local load not modeled"));
            } else {
                4
            };
            if base == "LDL" {
                Op::LdLocal {
                    d: lw.dst(lw.op(0)?)?,
                    bytes,
                    addr: lw.addr(lw.op(1)?)?,
                }
            } else {
                Op::StLocal {
                    bytes,
                    addr: lw.addr(lw.op(0)?)?,
                    v: lw.val(lw.op(1)?, bytes as u8)?,
                }
            }
        }
        // LDSM.16.M88.N / .MT88.N Rd, [Ra]: ldmatrix.m8n8.xN[.trans].b16.
        "LDSM" if has("16") && (has("M88") || has("MT88")) => {
            uniform = false;
            // `LDSM.16.M88` (no count): x1.
            let n: u8 = match mods.last() {
                Some(m) if m.chars().all(|c| c.is_ascii_digit()) => m.parse().unwrap_or(0),
                _ => 1,
            };
            if !matches!(n, 1 | 2 | 4) {
                bail!(lw.err("LDSM count"));
            }
            Op::LdMatrix {
                d: lw.dst(lw.op(0)?)?,
                n,
                addr: lw.addr(lw.op(1)?)?,
                trans: has("MT88"),
            }
        }
        // LDGSTS.E.BYPASS[.LTC128B].128 [Rd+off], desc[UR][Ra.64]: cp.async.cg 16 bytes (no
        // zero-fill / ignore-src operand seen).
        "LDGSTS" => {
            // A third operand: the source-valid predicate; `!PT` = src-size 0, zero fill
            // (Triton's masked cp.async of the prefetch past the last block: its address is
            // out of bounds).
            let src_ok = match l.operands.get(2) {
                None => Pred::TRUE,
                Some(p) => lw.pred(p)?,
            };
            // Size: `.128` / `.64`, none = 32 bits (cp.async.ca 4 / 8 / 16 bytes).
            let bytes = if has("128") {
                16
            } else if has("64") {
                8
            } else if mods
                .iter()
                .any(|m| matches!(*m, "U8" | "S8" | "U16" | "S16"))
            {
                bail!(lw.err("LDGSTS form not modeled (sub-word)"));
            } else {
                4
            };
            Op::CpAsync {
                dst: lw.addr(lw.op(0)?)?,
                src: lw.addr(lw.op(1)?)?,
                bytes,
                src_ok,
            }
        }
        "LDGDEPBAR" => Op::CpAsyncCommit,
        "ARRIVES"
            if mods == ["LDGSTSBAR", "64", "TRANSCNT"] || mods == ["LDGSTSBAR", "64", "ARVCNT"] =>
        {
            Op::CpAsyncMbarArrive {
                addr: lw.addr(lw.op(0)?)?,
                tx: mods[2] == "TRANSCNT",
            }
        }
        // HMMA.16816.F32.BF16 Rd, Ra, Rb, Rc: mma.sync m16n8k16 bf16 x bf16 + f32.
        "HMMA" if mods == ["16816", "F32", "BF16"] || mods == ["1688", "F32", "TF32"] => {
            uniform = false;
            let regs = |s: &str, n: u16| -> Result<Vec<Val>> {
                let s = s.trim_end_matches(".reuse");
                if s == "RZ" {
                    return Ok(vec![]);
                }
                let r = lw.dst(s)?;
                Ok((0..n)
                    .map(|i| {
                        Val::Reg(
                            Reg {
                                file: r.file,
                                idx: r.idx + i,
                            },
                            1,
                        )
                    })
                    .collect())
            };
            let a = regs(lw.op(1)?, 4)?;
            let b = regs(lw.op(2)?, 2)?;
            if a.is_empty() || b.is_empty() {
                bail!(lw.err("HMMA with a zero A/B operand not modeled"));
            }
            Op::Hmma {
                d: lw.dst(lw.op(0)?.trim_end_matches(".reuse"))?,
                a,
                b,
                c: regs(lw.op(3)?, 4)?,
                tf32: mods[0] == "1688",
            }
        }
        "LDS" => {
            let bytes = match mods {
                ["U16"] | ["S16"] => 2,
                ["U8"] => 1,
                [] => 4,
                ["64"] => 8,
                ["128"] => 16,
                _ => bail!(lw.err("LDS width")),
            };
            let d = lw.dst(lw.op(0)?)?;
            // ASSUME(S-SMEM-STRONG): see ASSUMPTIONS.md (`smem_acquires` may make it acquire)
            Op::Ld {
                d,
                n: (bytes / 4).max(1),
                space: Space::Shared,
                bytes,
                addr: lw.addr(lw.op(1)?)?,
                sync: Some(SMEM_RELAXED),
                sext: has("S16") || has("S8"),
            }
        }
        "LDG" if has("ENL2") && has("256") => Op::Ld256 {
            hi: lw.dst(lw.op(0)?)?,
            lo: lw.dst(lw.op(1)?)?,
            addr: lw.addr(lw.op(2)?)?,
            sync: sass_strong(mods),
        },
        "LDG" | "LD" => {
            let bytes = if has("64") {
                8
            } else if has("128") {
                16
            } else if has("U8") || has("S8") {
                1
            } else if has("U16") || has("S16") {
                2
            } else {
                4
            };
            Op::Ld {
                d: lw.dst(lw.op(0)?)?,
                n: (bytes / 4).max(1),
                space: if l.opcode.starts_with("LDG") {
                    Space::Global
                } else {
                    Space::Generic
                },
                bytes,
                addr: wide_e_addr(lw.addr(lw.op(1)?)?, has("E")),
                sync: sass_strong(mods),
                sext: has("S8") || has("S16"),
            }
        }
        // `ST.E [R2.64], R5`: a generic-address store (torch's index_select / embedding), as
        // `LD` is a generic load.
        "STG" | "ST" => {
            let bytes = if has("64") {
                8
            } else if has("128") {
                16
            } else if has("U8") || has("S8") {
                1
            } else if has("U16") || has("S16") {
                2
            } else {
                4
            };
            Op::St {
                space: if base == "STG" {
                    Space::Global
                } else {
                    Space::Generic
                },
                bytes,
                addr: wide_e_addr(lw.addr(lw.op(0)?)?, has("E")),
                v: lw.val(lw.op(1)?, bytes)?,
                sync: sass_strong(mods),
            }
        }
        // BAR.SYNC id[, count] / BAR.ARV id, count (bar.arrive: counts without waiting).
        "BAR" if has("SYNC") || has("ARV") => {
            let arrive = has("ARV");
            let count = match l.operands.get(1) {
                Some(c) => Some(lw.val(c, 4)?),
                None if arrive => bail!(lw.err("BAR.ARV without a thread count")),
                None => None,
            };
            Op::BarSync {
                id: lw.val(lw.op(0)?, 4)?,
                count,
                arrive,
            }
        }
        // mbarrier.inval
        "SYNCS" if has("CCTL") && has("IV") => Op::MbarInval {
            addr: lw.addr(lw.op(0)?)?,
        },
        "SYNCS" if has("EXCH") && has("64") => {
            if lw.op(0)? != "URZ" && lw.op(0)? != "RZ" {
                bail!(lw.err("SYNCS.EXCH with a result not modeled"));
            }
            Op::MbarInit {
                addr: lw.addr(lw.op(1)?)?,
                count: lw.val(lw.op(2)?, 8)?,
                encoded: true,
            }
        }
        "SYNCS" if has("ARRIVE") && has("TRANS64") => {
            if lw.op(0)? != "RZ" && lw.op(0)? != "URZ" {
                bail!(lw.err("SYNCS.ARRIVE returning state not modeled"));
            }
            match mods {
                // Arrive + expect_tx of the byte count in the third operand.
                ["ARRIVE", "TRANS64"] => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: Some(lw.val(lw.op(2)?, 4)?),
                    remote: false,
                    count: None,
                    complete: false,
                },
                // `.RED`: arrive.expect_tx on another CTA's mbarrier (shared::cluster address).
                ["ARRIVE", "TRANS64", "RED"] => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: Some(lw.val(lw.op(2)?, 4)?),
                    remote: true,
                    count: None,
                    complete: false,
                },
                // `.RED.A1T0`: a plain arrival there.
                ["ARRIVE", "TRANS64", "RED", "A1T0"] if lw.op(2)? == "RZ" => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: None,
                    remote: true,
                    count: None,
                    complete: false,
                },
                // `A1T0`: one arrival, no transaction bytes (plain mbarrier.arrive).
                ["ARRIVE", "TRANS64", "A1T0"] if lw.op(2)? == "RZ" => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: None,
                    remote: false,
                    count: None,
                    complete: false,
                },
                // `.A0TR` / `.A0TX`: no arrival; expect / complete the transaction bytes
                // (`mbarrier.expect_tx` / `mbarrier.complete_tx`, checked against ptxas).
                ["ARRIVE", "TRANS64", "RED", z @ ("A0TR" | "A0TX")] => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: Some(lw.val(lw.op(2)?, 4)?),
                    remote: true,
                    count: Some(Val::Imm(0)),
                    complete: *z == "A0TX",
                },
                // `.RED.A0T1`: no arrival, expect one transaction unit; ptxas emits it before
                // `ARRIVES.LDGSTSBAR.64.TRANSCNT` for `cp.async.mbarrier.arrive` (no .noinc),
                // which completes that unit asynchronously (probe in docs/rung3b.md).
                ["ARRIVE", "TRANS64", "RED", "A0T1"] if lw.op(2)? == "RZ" => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: Some(Val::Imm(1)),
                    remote: true,
                    count: Some(Val::Imm(0)),
                    complete: false,
                },
                _ => bail!(lw.err("SYNCS.ARRIVE variant not modeled")),
            }
        }
        // With or without `.TRYWAIT` (the plain form is a non-suspending test_wait; the model
        // treats both alike, E-SPIN).
        // A test whose result is discarded (`TRYWAIT PT, ...`: trtllm-gen polls the next
        // stage's mbarrier ahead of time) changes nothing the model tracks; dropping its
        // acquire only removes happens-before edges. ASSUME(E-SPIN)
        "SYNCS" if has("PHASECHK") && has("TRANS64") && lw.op(0)? == "PT" => Op::Nop,
        "SYNCS" if has("PHASECHK") && has("TRANS64") => Op::MbarTryWait {
            d: lw.pdst(lw.op(0)?)?,
            addr: lw.addr(lw.op(1)?)?,
            parity: lw.val(lw.op(2)?, 4)?,
            bit31: true,
        },
        "FENCE" if has("VIEW") && has("ASYNC") => Op::Fence {
            kind: FenceKind::ProxyAsync,
        },
        // try_cancel without multicast.
        "UGETNEXTWORKID" if mods == ["SELFCAST"] || mods == ["BROADCAST"] => Op::ClcTryCancel {
            resp: lw.ureg_at(lw.op(0)?, 0, 1)?,
            mbar: lw.ureg_at(lw.op(1)?, 0, 1)?,
            multicast: mods == ["BROADCAST"],
        },
        "UTMASTG" => {
            if mods.len() != 1 {
                bail!(lw.err("UTMASTG variant not modeled"));
            }
            let dims: u8 = mods[0]
                .trim_end_matches('D')
                .parse()
                .map_err(|_| lw.err("dims"))?;
            // URa: smem source, then N coordinates; URb: tensor map.
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
            }
        }
        // Closes the bulk group (cp.async.bulk.commit_group); its read scoreboard counts the
        // group until the group's shared-memory reads are done (see `lower`).
        "UTMACMDFLUSH" => Op::BulkCommit,
        // Generic-proxy ordering is covered by DRF-SC (E-DRF).
        // `CS2R Rd, SRZ`: zero the pair Rd:Rd+1 (other special registers are not modeled).
        // `CS2R.32 Rd, SR_x`: a 32-bit special register.
        "CS2R" if has("32") => Op::Mov {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
        },
        // `CS2R Rd, SR_CLOCKLO` (clock64): ASSUME(E-CLOCK) the clock reads 0 every time, so
        // timeout checks (`clock64() - start > N`) never fire.
        "CS2R" if l.operands.len() == 2 && lw.op(1)? == "SR_CLOCKLO" => Op::Mov {
            d: lw.dst(lw.op(0)?)?,
            a: Val::Reg(
                Reg {
                    file: File::R,
                    idx: RZ,
                },
                2,
            ),
        },
        "CS2R" if l.operands.len() == 2 && lw.op(1)? == "SRZ" => Op::Mov {
            d: lw.dst(lw.op(0)?)?,
            a: Val::Reg(
                Reg {
                    file: File::R,
                    idx: RZ,
                },
                2,
            ),
        },
        // The release half of a fence (ptxas: before release stores and in fence.acq_rel); also
        // what a later relaxed cluster arrive releases (E-CLUSTER-BAR). ASSUME(S-SCOPE)
        "MEMBAR" => Op::Fence {
            kind: FenceKind::Release(sass_scope(mods).unwrap_or(Scope::Sys)),
        },
        // L1 invalidate: the acquire half (after acquire loads, after MEMBAR in fence.acq_rel,
        // after the final bulk wait).
        "CCTL" if has("IVALL") => Op::Fence {
            kind: FenceKind::Acquire,
        },
        // Non-tensor bulk copies (cp.async.bulk; DeepEP's tma_load_1d / tma_store_1d):
        // `UBLKCP.S.G [URs], [URg], URn, desc[..]`: global URg:URg+1 -> shared URs, completing
        // the transaction on the mbarrier in URs+1; `UBLKCP.G.S [URg], [URs], URn, desc[..]`:
        // shared -> global in the issuer's bulk group. URn counts 16-byte units. Lowered as
        // TMA ops with `dims = 0`: `tmap` carries the unit count and `coords[0]` the global
        // address (exec.rs builds a 1-D map for them).
        "UBLKCP" if mods == ["S", "G"] => {
            let s = lw.op(0)?;
            Op::TmaLoad {
                dims: 0,
                dst: lw.ureg_at(s, 0, 1)?,
                off: 0,
                mbar: lw.ureg_at(s, 1, 1)?,
                mbar_off: 0,
                tmap: lw.val(lw.op(2)?, 4)?,
                coords: vec![lw.ureg_at(lw.op(1)?, 0, 2)?],
                mask: None,
                pair: false,
            }
        }
        "UBLKCP" if mods == ["G", "S"] => Op::TmaStore {
            dims: 0,
            src: lw.ureg_at(lw.op(1)?, 0, 1)?,
            off: 0,
            tmap: lw.val(lw.op(2)?, 4)?,
            coords: vec![lw.ureg_at(lw.op(0)?, 0, 2)?],
        },
        "UTMALDG" => {
            // `.MULTICAST`: a third operand UR holds the ctaMask. `.2CTA`: the mbarrier may be
            // in the peer CTA (cp.async.bulk.tensor ... .cta_group::2).
            // `.2D.GATHER4` (cp.async.bulk.tensor.2d...tile::gather4): URa+2 is the column,
            // URa+3..URa+6 the four row coordinates (5 coords; exec.rs `complete_tma`).
            let multicast = mods.len() == 2 && mods[1] == "MULTICAST";
            let pair = mods.len() == 2 && mods[1] == "2CTA";
            let gather4 = mods == ["2D", "GATHER4"];
            if mods.len() != 1 && !multicast && !pair && !gather4 {
                bail!(lw.err("UTMALDG variant not modeled yet (im2col)"));
            }
            let dims: u8 = mods[0]
                .trim_end_matches('D')
                .parse()
                .map_err(|_| lw.err("dims"))?;
            let t = lw.op(0)?;
            let coords = (0..dims as u16 + if gather4 { 3 } else { 0 })
                .map(|k| lw.ureg_at(t, 2 + k, 1))
                .collect::<Result<_>>()?;
            Op::TmaLoad {
                dims,
                dst: lw.ureg_at(t, 0, 1)?,
                off: 0,
                mbar: lw.ureg_at(t, 1, 1)?,
                mbar_off: 0,
                tmap: lw.ureg_at(lw.op(1)?, 0, 2)?,
                coords,
                mask: if multicast {
                    Some(lw.val(lw.op(2)?, 4)?)
                } else {
                    None
                },
                pair,
            }
        }
        // barrier.cluster: the arrive is always relaxed in SASS; `.release` is a MEMBAR before it.
        // Every active thread takes part (PTX semantics are per thread), although the
        // instruction is on the uniform datapath.
        "UCGABAR_ARV" => {
            uniform = false;
            Op::ClusterArrive { release: false }
        }
        "UCGABAR_WAIT" => {
            uniform = false;
            Op::ClusterWait
        }
        // Error barriers (drain outstanding memory errors before a cluster arrive): no effect
        // on the memory model.
        "ERRBAR" | "CGAERRBAR" => Op::Nop,
        "SEL" | "USEL" => Op::Sel {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: lw.val(lw.op(2)?, 4)?,
            p: lw.pred(lw.op(3)?)?,
        },
        "PRMT" | "UPRMT" => Op::Prmt {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            sel: lw.val(lw.op(2)?, 4)?,
            b: lw.val(lw.op(3)?, 4)?,
        },
        // `[@P] BRA.DIV URm, `(label)``: warp-collective divergence test (Op::Bra `div`).
        "BRA" if mods == ["DIV"] && l.operands.len() == 2 => {
            let t = lw.op(1)?;
            let lbl = t
                .trim_start_matches('`')
                .trim_start_matches('(')
                .trim_end_matches(')');
            let a = *labels
                .get(lbl)
                .ok_or_else(|| lw.err(format!("unknown label {lbl}")))?;
            let target = *addr_to_idx
                .get(&a)
                .ok_or_else(|| lw.err("label outside function"))?;
            uniform = false;
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: Some(lw.val(lw.op(0)?, 4)?),
            }
        }
        "BRA" => {
            // `BRA.U UP0, `(label)`: the condition is an operand, not a guard.
            // `BRA.U !P0, UPT, `(label)`: a thread predicate and a uniform one, both must hold;
            // only a true uniform predicate is modeled.
            let (cond, t) = if l.operands.len() == 3 && lw.op(1)? == "UPT" {
                (Some(lw.pred(lw.op(0)?)?), lw.op(2)?)
            } else if l.operands.len() == 2 {
                (Some(lw.pred(lw.op(0)?)?), lw.op(1)?)
            } else {
                (None, lw.op(0)?)
            };
            if let Some(c) = cond.filter(|_| !guard.is_always()) {
                // `@P0 BRA P1, target`: taken where both hold. ASSUME(S-BRA-COND)
                let lbl = t
                    .trim_start_matches('`')
                    .trim_start_matches('(')
                    .trim_end_matches(')');
                let a = *labels
                    .get(lbl)
                    .ok_or_else(|| lw.err(format!("unknown label {lbl}")))?;
                let target = *addr_to_idx
                    .get(&a)
                    .ok_or_else(|| lw.err("label outside function"))?;
                if has("U") || has("ANY") {
                    bail!(lw.err("BRA.U / .ANY with a guard and a condition not modeled"));
                }
                return Ok(Inst {
                    guard,
                    op: Op::Bra {
                        target,
                        any: false,
                        tid0_falls: false,
                        cond: Some(c),
                        div: None,
                    },
                    uniform: false,
                    loc: l.addr,
                    text: l.text.clone(),
                });
            }
            if let Some(c) = cond {
                let lbl = t
                    .trim_start_matches('`')
                    .trim_start_matches('(')
                    .trim_end_matches(')');
                let a = *labels
                    .get(lbl)
                    .ok_or_else(|| lw.err(format!("unknown label {lbl}")))?;
                let target = *addr_to_idx
                    .get(&a)
                    .ok_or_else(|| lw.err("label outside function"))?;
                return Ok(Inst {
                    guard: c,
                    op: Op::Bra {
                        target,
                        any: has("ANY"),
                        tid0_falls: false,
                        cond: None,
                        div: None,
                    },
                    uniform: false,
                    loc: l.addr,
                    text: l.text.clone(),
                });
            }
            let lbl = t
                .trim_start_matches('`')
                .trim_start_matches('(')
                .trim_end_matches(')');
            let a = *labels
                .get(lbl)
                .ok_or_else(|| lw.err(format!("unknown label {lbl}")))?;
            let target = *addr_to_idx
                .get(&a)
                .ok_or_else(|| lw.err("label outside function"))?;
            uniform = false;
            Op::Bra {
                target,
                any: has("ANY"),
                tid0_falls: has("NOT_TID0"),
                cond: None,
                div: None,
            }
        }
        // `BRX R4 -0x8bf0 (*"BRANCH_TARGETS .L_x_1,.L_x_2"*)`
        "BRX" | "BRXU" => {
            uniform = false;
            let (head, tl) = l
                .text
                .split_once("BRANCH_TARGETS")
                .ok_or_else(|| lw.err("BRX without BRANCH_TARGETS"))?;
            let mut w = head.split_whitespace().skip(1);
            let base = lw.val(w.next().ok_or_else(|| lw.err("BRX operand"))?, 4)?;
            let rel = w.next().ok_or_else(|| lw.err("BRX offset"))?;
            let rel = match rel.strip_prefix('-') {
                Some(h) => -(hex(h).ok_or_else(|| lw.err("bad offset"))? as i64),
                None => hex(rel).ok_or_else(|| lw.err("bad offset"))? as i64,
            };
            let mut targets = Vec::new();
            for lbl in tl.trim_end_matches([')', '*', '"']).split(',') {
                let lbl = lbl.trim().trim_matches('"');
                let a = *labels
                    .get(lbl)
                    .ok_or_else(|| lw.err(format!("unknown label {lbl}")))?;
                targets.push(
                    *addr_to_idx
                        .get(&a)
                        .ok_or_else(|| lw.err("label outside function"))?,
                );
            }
            Op::Brx {
                base,
                off: l.addr as i64 + 16 + rel,
                targets,
            }
        }
        // setmaxnreg: the register budget does not change what the model executes. The
        // TRY_ALLOC form always succeeds here (on hardware a failure is retried).
        "USETMAXREG" if has("DEALLOC") || (has("ALLOC") && !has("TRY_ALLOC")) => Op::Nop,
        "USETMAXREG" if has("TRY_ALLOC") => Op::Plop3 {
            d: lw.pdst(lw.op(0)?)?,
            a: Pred::TRUE,
            b: Pred::TRUE,
            c: Pred::TRUE,
            lut: 0xff,
        },
        // SHFL.<mode> Pout, Rd, Ra, b, c (PTX shfl.sync operands; the full member mask).
        "SHFL" => {
            uniform = false;
            let p = match lw.op(0)? {
                "PT" => None,
                x => Some(lw.pdst(x)?),
            };
            let mode = match mods {
                ["IDX"] => ShflMode::Idx,
                ["UP"] => ShflMode::Up,
                ["DOWN"] => ShflMode::Down,
                ["BFLY"] => ShflMode::Bfly,
                _ => bail!(lw.err("SHFL mode")),
            };
            Op::Shfl {
                d: lw.dst(lw.op(1)?)?,
                a: lw.val(lw.op(2)?, 4)?,
                lane: lw.val(lw.op(3)?, 4)?,
                clamp: lw.val(lw.op(4)?, 4)?,
                mode,
                p,
            }
        }
        _ => bail!(lw.err(format!("unsupported SASS opcode `{}`", l.opcode))),
    };
    if matches!(op, Op::TmaLoad { .. } | Op::MbarInit { .. })
        && l.operands
            .iter()
            .all(|o| !o.contains('R') || o.contains("UR"))
    {
        uniform = true;
    }
    if matches!(
        op,
        Op::Elect { .. }
            | Op::Bra { .. }
            | Op::Call { .. }
            | Op::TmemLd { .. }
            | Op::WarpSync { .. }
            | Op::Bssy { .. }
            | Op::Bsync { .. }
            | Op::Break { .. }
    ) {
        uniform = false;
    }
    Ok(Inst {
        guard,
        op,
        uniform,
        loc: l.addr,
        text: l.text.clone(),
    })
}

/// Load a kernel from a cubin: disassemble, read metadata, lower.
// ASSUME(L-NVDISASM): see ASSUMPTIONS.md
/// Scope suffix of a SASS memory instruction or MEMBAR (`.SYS`, `.GPU`, `.SM` = CTA; `.CTA`
/// in `MEMBAR.ALL.CTA`).
/// Modifiers of the integer min / max and CAS forms of ATOM / ATOMG / REDG that the lowering
/// models (anything else, e.g. a cache-eviction hint, is rejected).
fn atom_mod_known(m: &str) -> bool {
    matches!(
        m,
        "E" | "CAS"
            | "MIN"
            | "MAX"
            | "U32"
            | "S32"
            | "64"
            | "S64"
            | "STRONG"
            | "GPU"
            | "SYS"
            | "SM"
            | "CTA"
    )
}

/// `.MIN` / `.MAX` of an integer atomic: `.S32` / `.S64` signed, else unsigned.
fn min_max_op(mods: &[&str]) -> AtomOp {
    let signed = mods.iter().any(|m| matches!(*m, "S32" | "S64"));
    match (mods.contains(&"MIN"), signed) {
        (true, false) => AtomOp::MinU,
        (true, true) => AtomOp::MinS,
        (false, false) => AtomOp::MaxU,
        (false, true) => AtomOp::MaxS,
    }
}

fn sass_scope(mods: &[&str]) -> Option<Scope> {
    mods.iter().find_map(|m| match *m {
        "SYS" => Some(Scope::Sys),
        "GPU" => Some(Scope::Gpu),
        "SM" | "CTA" => Some(Scope::Cta),
        _ => None,
    })
}

/// Rounding-mode modifier of a float instruction (none: to nearest even).
fn sass_rnd(mods: &[&str]) -> Rnd {
    if mods.contains(&"RZ") {
        Rnd::Rz
    } else if mods.contains(&"RM") {
        Rnd::Rm
    } else if mods.contains(&"RP") {
        Rnd::Rp
    } else {
        Rnd::Rn
    }
}

/// LDS / STS of the CTA's own shared memory: relaxed strong at CTA scope (S-SMEM-STRONG).
/// ptxas emits the same encoding for weak, relaxed and volatile .cta shared accesses (no
/// field carries strength: docs/smem-order-mapping.md), and GB300 shows no weaker behavior
/// (docs/smem-litmus.md).
const SMEM_RELAXED: MemSync = MemSync {
    sem: Sem::Relaxed,
    scope: Scope::Cta,
};

/// Register-only SASS operations (no memory, synchronization or control effect).
fn alu_only(op: &Op) -> bool {
    matches!(
        op,
        Op::Mov { .. }
            | Op::Int { .. }
            | Op::MulWide { .. }
            | Op::Imad { .. }
            | Op::Iadd3 { .. }
            | Op::CvtInt { .. }
            | Op::Flo { .. }
            | Op::Lea { .. }
            | Op::LeaHi { .. }
            | Op::Pack64 { .. }
            | Op::I2F { .. }
            | Op::F2I { .. }
            | Op::Shf { .. }
            | Op::Lop3 { .. }
            | Op::Sel { .. }
            | Op::Plop3 { .. }
            | Op::Setp { .. }
            | Op::Fadd { .. }
            | Op::Ffma { .. }
            | Op::Ffma2 { .. }
            | Op::Fmnmx { .. }
            | Op::Fsetp { .. }
            | Op::Hsetp2Bf16 { .. }
            | Op::Fset { .. }
            | Op::Fchk { .. }
            | Op::Fsel { .. }
            | Op::Mufu { .. }
            | Op::R2p { .. }
            | Op::Brev { .. }
            | Op::AddMinMax { .. }
            | Op::IntMnmx3 { .. }
            | Op::P2r { .. }
            | Op::Nop
    )
}

/// ASSUME(S-SMEM-STRONG): an LDS whose scoreboard is waited on before the warp issues any
/// later memory, synchronization or control instruction is `ld.acquire.cta` (ptxas's
/// lowering of it: LDS, then a wait on its scoreboard; no cache maintenance at CTA scope).
/// The scan follows straight-line code only; a branch or any non-ALU instruction issued
/// before the wait ends it (relaxed).
fn smem_acquires(lines: &[SassLine], code: &mut [Inst]) {
    for i in 0..code.len() {
        let Op::Ld {
            space: Space::Shared,
            sync: Some(s),
            ..
        } = &code[i].op
        else {
            continue;
        };
        if *s != SMEM_RELAXED || lines[i].opcode.split('.').next() != Some("LDS") {
            continue;
        }
        let Some(sb) = lines[i].hi.and_then(|h| scoreboard::Ctrl::decode(h).wbar) else {
            continue;
        };
        let waited = |j: usize| {
            lines[j]
                .hi
                .is_some_and(|h| scoreboard::Ctrl::decode(h).wait >> sb & 1 == 1)
        };
        let mut acq = false;
        for (j, inst) in code.iter().enumerate().skip(i + 1) {
            if waited(j) {
                acq = true;
                break;
            }
            if !alu_only(&inst.op) {
                break;
            }
        }
        if acq {
            if let Op::Ld { sync, .. } = &mut code[i].op {
                *sync = Some(MemSync {
                    sem: Sem::Acquire,
                    scope: Scope::Cta,
                });
            }
        }
    }
}

/// `.E` on a global / generic access: a 64-bit address, even when nvdisasm prints `[R6]`
/// without `.64` (cuBLAS's cutlass3x SIMT sgemm: `LEA R10, P0, R8, UR4, 0x2; LEA.HI.X R11,
/// ...; STG.E [R10], R48`).
fn wide_e_addr(mut a: Addr, e: bool) -> Addr {
    if e {
        if let Some(Val::Reg(r, 1)) = a.base {
            if r.file == File::R {
                a.base = Some(Val::Reg(r, 2));
            }
        }
    }
    a
}

/// `.STRONG.<scope>` on LDG / STG: a relaxed strong access (acquire / release are separate
/// CCTL.IVALL / MEMBAR instructions).
fn sass_strong(mods: &[&str]) -> Option<MemSync> {
    if !mods.contains(&"STRONG") {
        return None;
    }
    Some(MemSync {
        sem: Sem::Relaxed,
        scope: sass_scope(mods).unwrap_or(Scope::Gpu),
    })
}

pub fn load_cubin(cubin: &Path, kernel: &str) -> Result<Kernel> {
    let dis = run_tool("nvdisasm", &["-c", "-hex", cubin.to_str().unwrap()])?;
    let meta = read_meta(cubin, kernel)?;
    lower(&dis, kernel, &meta)
}

/// One kind of SASS instruction the lifter does not lower: why (the error without its
/// location), how many instructions, and the first full error message.
#[derive(Clone, Debug)]
pub struct Unlowered {
    pub why: String,
    pub count: usize,
    pub example: String,
}

/// Every kind of instruction of `kernel` in `cubin` the lifter does not lower (empty: the
/// kernel lifts). The PTXV_LIFT_REPORT listing, as data.
pub fn lift_report(cubin: &Path, kernel: &str) -> Result<Vec<Unlowered>> {
    let dis = run_tool("nvdisasm", &["-c", "-hex", cubin.to_str().unwrap()])?;
    let meta = read_meta(cubin, kernel)?;
    let (lines, labels) = parse_disasm(&dis, kernel)?;
    let addr_to_idx: HashMap<u32, usize> =
        lines.iter().enumerate().map(|(i, l)| (l.addr, i)).collect();
    let plain_exit = lines
        .iter()
        .position(|l| l.opcode == "EXIT" && l.guard.is_none() && l.operands.is_empty());
    Ok(unlowered_lines(
        &lines,
        &labels,
        &addr_to_idx,
        &meta,
        plain_exit,
    ))
}

fn unlowered_lines(
    lines: &[SassLine],
    labels: &HashMap<String, u32>,
    addr_to_idx: &HashMap<u32, usize>,
    meta: &CubinMeta,
    plain_exit: Option<usize>,
) -> Vec<Unlowered> {
    let mut seen: std::collections::BTreeMap<String, (usize, String)> = Default::default();
    for l in lines {
        if let Err(e) = lower_line(l, labels, addr_to_idx, meta, plain_exit) {
            let first = e.to_string().lines().next().unwrap_or("").to_string();
            let why = first
                .split_once(": ")
                .map_or(first.clone(), |x| x.1.to_string());
            let e2 = seen.entry(why).or_insert((0, e.to_string()));
            e2.0 += 1;
        }
    }
    seen.into_iter()
        .map(|(why, (count, example))| Unlowered {
            why,
            count,
            example,
        })
        .collect()
}

pub fn lower(dis: &str, kernel: &str, meta: &CubinMeta) -> Result<Kernel> {
    let (mut lines, labels) = parse_disasm(dis, kernel)?;
    // Relocated 32-bit halves (`32@lo(X)` / `32@hi(X)`): a label's code address
    // (`FN + .L_x_15@srel`, the return address of a `vprintf` call) or a data symbol (printf
    // format strings, `$str$5`), which only the printf paths of failed device assertions use:
    // address 0 (never dereferenced by the model, which does not run vprintf).
    for l in &mut lines {
        for o in &mut l.operands {
            let (hi, inner) = match (o.strip_prefix("32@lo("), o.strip_prefix("32@hi(")) {
                (Some(r), _) => (false, r),
                (_, Some(r)) => (true, r),
                _ => continue,
            };
            let inner = inner.strip_suffix(')').unwrap_or(inner);
            let v: u64 = match inner.split_once('+').map(|x| x.1.trim()) {
                Some(rest) if rest.ends_with("@srel)") || rest.ends_with("@srel") => {
                    let lbl = rest.trim_end_matches(')').trim_end_matches("@srel");
                    *labels
                        .get(lbl)
                        .ok_or_else(|| anyhow!("sass {:#06x}: unknown label {lbl}", l.addr))?
                        as u64
                }
                _ => 0,
            };
            *o = format!("{:#x}", if hi { v >> 32 } else { v & 0xffff_ffff });
        }
    }
    // Trailing `BRA self` + NOP padding after the last EXIT is ptxas filler; keep it (it is
    // unreachable) so addresses stay aligned with the disassembly.
    let addr_to_idx: HashMap<u32, usize> =
        lines.iter().enumerate().map(|(i, l)| (l.addr, i)).collect();
    let mut code = Vec::with_capacity(lines.len());
    let plain_exit = lines
        .iter()
        .position(|l| l.opcode == "EXIT" && l.guard.is_none() && l.operands.is_empty());
    // PTXV_LIFT_REPORT: list every instruction the model does not lower (a new binary).
    if std::env::var("PTXV_LIFT_REPORT").is_ok() {
        let seen = unlowered_lines(&lines, &labels, &addr_to_idx, meta, plain_exit);
        eprintln!(
            "lift report for {kernel}: {} kinds of instruction not lowered",
            seen.len()
        );
        for u in &seen {
            eprintln!(
                "{:5} {}\n      e.g. {}",
                u.count,
                u.why,
                u.example.replace('\n', " ")
            );
        }
        if !seen.is_empty() {
            bail!("{} kinds of instruction not lowered", seen.len());
        }
    }
    for l in &lines {
        code.push(lower_line(l, &labels, &addr_to_idx, meta, plain_exit)?);
    }
    // ENDCOLLECTIVE closes the nearest preceding WARPSYNC.COLLECTIVE.
    for i in 0..code.len() {
        if matches!(code[i].op, Op::EndCollective { .. }) {
            let start = (0..i)
                .rev()
                .find(|&j| {
                    matches!(
                        code[j].op,
                        Op::WarpSync {
                            collective: true,
                            ..
                        }
                    )
                })
                .ok_or_else(|| {
                    anyhow!(
                        "ENDCOLLECTIVE at {:#x} without WARPSYNC.COLLECTIVE",
                        code[i].loc
                    )
                })?;
            code[i].op = Op::EndCollective { start };
        }
    }
    if lines
        .iter()
        .enumerate()
        .any(|(i, l)| l.addr != (i as u32) * 16)
    {
        bail!("instruction addresses are not contiguous 16-byte slots");
    }
    let mut klabels: Vec<(usize, String)> = labels
        .iter()
        .filter_map(|(n, a)| addr_to_idx.get(a).map(|&i| (i, n.clone())))
        .collect();
    klabels.sort();
    for (i, inst) in code.iter_mut().enumerate() {
        if let Op::Trap { why } = &mut inst.op {
            *why = klabels
                .iter()
                .rev()
                .find(|(j, _)| *j <= i)
                .map(|(_, n)| n.clone())
                .unwrap_or_default();
        }
    }
    // ASSUME(L-OPCLASS): see ASSUMPTIONS.md
    // cp.async.bulk.wait_group[.read] n has no instruction of its own: ptxas waits with
    // `DEPBAR.LE SBk, n` on the read scoreboard of UTMACMDFLUSH (the bulk group counter). Both
    // forms lower to the same wait, which only guarantees the source reads are done.
    let bulk_sbs: Vec<u8> = lines
        .iter()
        .filter(|l| l.opcode == "UTMACMDFLUSH")
        .filter_map(|l| l.hi.and_then(|h| scoreboard::Ctrl::decode(h).rbar))
        .collect();
    // cp.async.wait_group n: `DEPBAR.LE SBk, n` on the write scoreboard of LDGDEPBAR.
    let cp_sbs: Vec<u8> = lines
        .iter()
        .filter(|l| l.opcode == "LDGDEPBAR")
        .filter_map(|l| l.hi.and_then(|h| scoreboard::Ctrl::decode(h).wbar))
        .collect();
    for (i, l) in lines.iter().enumerate() {
        if l.opcode != "DEPBAR.LE" || l.operands.len() != 2 {
            continue;
        }
        let sb = l.operands[0]
            .strip_prefix("SB")
            .and_then(|k| k.parse::<u8>().ok());
        let n = u32::from_str_radix(l.operands[1].trim_start_matches("0x"), 16).ok();
        if let (Some(sb), Some(n)) = (sb, n) {
            if bulk_sbs.contains(&sb) {
                code[i].op = Op::BulkWait { n, read: true };
            } else if cp_sbs.contains(&sb) {
                code[i].op = Op::CpAsyncWait { n };
            }
        }
    }
    // A wait bit on that scoreboard in any instruction's control word is the same wait with
    // n = 0 (cuBLAS's nvjet epilogue: the BAR.SYNC before reusing a TMA-store staging buffer).
    // ASSUME(E-ASYNC-STEP)
    let bulk_read_waits: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            l.hi.map(scoreboard::Ctrl::decode)
                .is_some_and(|c| bulk_sbs.iter().any(|&s| c.wait >> s & 1 != 0))
        })
        .map(|(i, _)| i)
        .collect();
    smem_acquires(&lines, &mut code);
    let static_findings = scoreboard::check(&lines, &code)?;
    let pend = meta.params.iter().map(|p| p.1 + p.2).max().unwrap_or(0);
    let params = meta
        .params
        .iter()
        .map(|&(ord, off, size)| Param {
            name: format!("p{ord}"),
            offset: off,
            size,
        })
        .collect();
    Ok(Kernel {
        name: kernel.to_string(),
        front: Front::Sass,
        code,
        nregs: [0, 256, 64, 8, 8],
        params,
        param_cbank: meta.const0_size - pend,
        smem_bytes: meta.smem_size,
        vreg_names: vec![],
        labels: klabels,
        static_findings,
        bulk_read_waits,
        parts: vec![],
    })
}

enum F2fpKind {
    Pack { merge: bool },
    Half,
    Unpack,
}

/// Decode the modifiers of a narrow-format `F2FP` (N-CVT-LP):
/// `[SATFINITE] [RELU] <to> <F32|F16|BF16> <PACK_AB[_MERGE_C]|UNPACK_B_MERGE_C> [RZ|RP|RS]` or
/// `[RELU] <F16|BF16> <from> UNPACK_B` -> (to, from, kind, rounding, satfinite, relu). f16 / bf16
/// targets only with `.RS` or `.SATFINITE` (the plain RN forms are `F2fpF16` / `F2fpBf16`).
fn f2fp_lp(mods: &[&str]) -> Option<(Lp, Lp, F2fpKind, LpRnd, bool, bool)> {
    let fmt = |m: &str| {
        Some(match m {
            "E4M3" => Lp::E4m3,
            "E5M2" => Lp::E5m2,
            "E2M3" => Lp::E2m3,
            "E3M2" => Lp::E3m2,
            "E2M1" => Lp::E2m1,
            "E8" => Lp::Ue8m0,
            "F16" => Lp::F16,
            "BF16" => Lp::Bf16,
            _ => return None,
        })
    };
    let mut i = 0;
    let sat = mods.get(i) == Some(&"SATFINITE");
    i += sat as usize;
    let relu = mods.get(i) == Some(&"RELU");
    i += relu as usize;
    let to = fmt(mods.get(i)?)?;
    let src = mods.get(i + 1)?;
    let kind = *mods.get(i + 2)?;
    let rnd = match mods.get(i + 3..)? {
        [] => LpRnd::Rn,
        ["RZ"] => LpRnd::Rz,
        ["RP"] => LpRnd::Rp,
        ["RS"] => LpRnd::Rs,
        _ => return None,
    };
    let wide = matches!(to, Lp::F16 | Lp::Bf16);
    match (*src, kind) {
        ("F32", "PACK_AB" | "PACK_AB_MERGE_C") => {
            let merge = kind == "PACK_AB_MERGE_C";
            // f16 / bf16 targets: `.RS`, or RN with `.SATFINITE` (gemm2's epilogue).
            if wide && (!(rnd == LpRnd::Rs || rnd == LpRnd::Rn && sat) || merge) || !wide && !merge
            {
                return None;
            }
            if matches!(rnd, LpRnd::Rz | LpRnd::Rp) != (to == Lp::Ue8m0) {
                return None;
            }
            Some((to, to, F2fpKind::Pack { merge }, rnd, sat, relu))
        }
        ("F16" | "BF16", "UNPACK_B_MERGE_C") if !wide && rnd != LpRnd::Rs => {
            Some((to, fmt(src)?, F2fpKind::Half, rnd, sat, relu))
        }
        (_, "UNPACK_B") if wide && !sat && rnd == LpRnd::Rn => {
            let from = fmt(src)?;
            (!matches!(from, Lp::F16 | Lp::Bf16)).then_some((
                to,
                from,
                F2fpKind::Unpack,
                rnd,
                sat,
                relu,
            ))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_imm() {
        assert_eq!(parse_half("5.9604644775390625e-08"), Some(1));
        assert_eq!(parse_half("1.875"), Some(0x3f80));
        assert_eq!(parse_half("0"), Some(0));
    }
}
