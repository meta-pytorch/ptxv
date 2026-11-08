/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Scoreboard verification: check, instead of trust, the scheduling control bits ptxas puts
//! in every instruction.
//!
//! Volta and later GPUs do not interlock on registers written or read by *variable-latency*
//! instructions (loads, `LDTM`, `S2R`, `SYNCS`, ...). The compiler must give the producer a
//! scoreboard (SB0..SB5) and put a wait on that scoreboard on, or before, the first instruction
//! that touches the register. A missing wait is a silent data race inside one thread: the
//! consumer reads whatever the register held before. This is exactly the guarantee
//! `tcgen05.wait::ld` asks for at the PTX level, and the binary has no other instruction for it.
//!
//! Control word (`hi >> 41`, validated on sm_100a/sm_103a by Rosetta probes, see docs/sass-ctrl.md):
//!
//! | bits    | field | meaning |
//! |---------|-------|---------|
//! | [0, 4)  | stall | cycles before the next instruction issues (fixed-latency hazards) |
//! | 4       | yield | |
//! | [5, 8)  | wbar  | scoreboard incremented at issue, released when the results are written (7 = none) |
//! | [8, 11) | rbar  | scoreboard released when the sources have been read (7 = none) |
//! | [11,17) | wait  | mask of scoreboards that must reach zero *before this instruction issues* |
//! | [17,21) | reuse | operand reuse cache |
//!
//! `DEPBAR.LE SBk, n` waits until at most `n` operations are outstanding on SBk.
//!
//! The analysis is a dataflow fixpoint over the whole CFG (every path, independent of the launch
//! grid and of predicate values). Its state is the set of possibly-outstanding operations, each
//! with the registers it may still own and a lower bound on how many newer operations were
//! issued on its scoreboard (for `DEPBAR.LE`). It reports, for a register R:
//! * RAW: R is read while a variable-latency write of R may be outstanding;
//! * WAW: R is written while such a write may be outstanding;
//! * WAR: R is written while a read of R tracked by a read scoreboard may be outstanding;
//! * a variable-latency instruction whose result has no write scoreboard at all.
//!
//! Not checked here (trusted, see docs/sass-ctrl.md): fixed-latency hazards covered by stall
//! counts, and WAR on variable-latency reads that carry no read scoreboard.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;

use anyhow::Result;
use anyhow::bail;
use ptxv_ir::CtrlPatch;
use ptxv_ir::Inst;
use ptxv_ir::Op;
use ptxv_ir::StaticFinding;

use super::SassLine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ctrl {
    pub stall: u8,
    pub yld: bool,
    pub wbar: Option<u8>,
    pub rbar: Option<u8>,
    pub wait: u8,
    pub reuse: u8,
}

impl Ctrl {
    // ASSUME(L-CTRL): see ASSUMPTIONS.md
    pub fn decode(hi: u64) -> Ctrl {
        let c = hi >> 41;
        let sb = |x: u64| if x == 7 { None } else { Some(x as u8) };
        Ctrl {
            stall: (c & 0xf) as u8,
            yld: (c >> 4) & 1 == 1,
            wbar: sb((c >> 5) & 7),
            rbar: sb((c >> 8) & 7),
            wait: ((c >> 11) & 0x3f) as u8,
            reuse: ((c >> 17) & 0xf) as u8,
        }
    }

    pub fn encode(&self, hi: u64) -> u64 {
        let c = (self.stall as u64 & 0xf)
            | (self.yld as u64) << 4
            | (self.wbar.unwrap_or(7) as u64) << 5
            | (self.rbar.unwrap_or(7) as u64) << 8
            | (self.wait as u64 & 0x3f) << 11
            | (self.reuse as u64 & 0xf) << 17;
        (hi & ((1u64 << 41) - 1)) | c << 41
    }

    pub fn describe(&self) -> String {
        let sb = |x: Option<u8>| x.map(|k| format!("SB{k}")).unwrap_or("-".into());
        let w: Vec<String> = (0..6)
            .filter(|k| self.wait >> k & 1 == 1)
            .map(|k| format!("SB{k}"))
            .collect();
        format!(
            "stall={} wr={} rd={} wait=[{}]",
            self.stall,
            sb(self.wbar),
            sb(self.rbar),
            w.join(",")
        )
    }
}

/// A register as the scoreboard sees it: (file letter, index). Files: b'R', b'U' (uniform),
/// b'P', b'Q' (uniform predicate).
type RegKey = (u8, u16);

fn reg_token(tok: &str) -> Option<(RegKey, bool)> {
    // Returns the register and whether it is the zero/true register (never a hazard).
    let num = |s: &str| s.parse::<u16>().ok();
    if tok == "RZ" || tok == "URZ" || tok == "PT" || tok == "UPT" {
        return Some(((0, 0), true));
    }
    if let Some(n) = tok.strip_prefix("UR").and_then(num) {
        return Some(((b'U', n), false));
    }
    if let Some(n) = tok.strip_prefix("UP").and_then(num) {
        return Some(((b'Q', n), false));
    }
    if let Some(n) = tok.strip_prefix('R').and_then(num) {
        return Some(((b'R', n), false));
    }
    if let Some(n) = tok.strip_prefix('P').and_then(num) {
        return Some(((b'P', n), false));
    }
    None
}

/// All registers mentioned in one operand, with their widths. `width` applies to a bare
/// register operand (data registers of `.64`/`.128` memory ops).
fn operand_regs(op: &str, width: u16, out: &mut Vec<RegKey>) {
    let b = op.as_bytes();
    let mut i = 0;
    let bare = !op.contains('[');
    while i < b.len() {
        let at_boundary =
            i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.');
        if at_boundary && b[i].is_ascii_alphabetic() {
            let mut j = i;
            while j < b.len() && b[j].is_ascii_alphanumeric() {
                j += 1;
            }
            let tok = &op[i..j];
            if let Some((r, zero)) = reg_token(tok) {
                let mut w = 1;
                if op[j..].starts_with(".64") {
                    w = 2;
                } else if bare && r.0 == b'R' {
                    w = width;
                }
                // 64-bit descriptors in brackets: desc[URx], gdesc[URx] (idesc[URx] is 32-bit).
                if op[..i].ends_with("desc[") && !op[..i].ends_with("idesc[") {
                    w = 2;
                }
                if !zero {
                    for k in 0..w {
                        out.push((r.0, r.1 + k));
                    }
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
}

fn is_pred_tok(s: &str) -> bool {
    matches!(reg_token(s), Some(((b'P' | b'Q', _), _))) || s == "PT" || s == "UPT"
}

fn is_plain_reg(s: &str) -> bool {
    let s = s.trim_end_matches(".reuse").trim_end_matches(".64");
    reg_token(s).is_some()
}

/// Opcodes with no register destination: every register operand is a source.
const NO_DEF: &[&str] = &[
    "LDGSTS",
    "ST",
    "STG",
    "STS",
    "STL",
    "STAS",
    "RED",
    "REDG",
    "BRA",
    "BRX",
    "JMP",
    "JMX",
    "BAR",
    "CALL",
    "RET",
    "EXIT",
    "WARPSYNC",
    "DEPBAR",
    "NOP",
    "BSSY",
    "BSYNC",
    "BREAK",
    "UTMALDG",
    "UTMASTG",
    "UTMAPF",
    "UTMACCTL",
    "UTCHMMA",
    "UTCQMMA",
    "UTCOMMA",
    "UTCBAR",
    "NANOSLEEP",
    "YIELD",
    "PMTRIG",
    "BPT",
    "FENCE",
    "MEMBAR",
    "CCTL",
    "UVIRTCOUNT",
    "ERRBAR",
    "ACQBULK",
    "KILL",
];

/// Variable-latency opcodes: their register results are not interlocked and need a write
/// scoreboard. (Every opcode ptxas ever gives a write scoreboard on our corpus, plus the
/// usual memory/transcendental/shuffle units.)
// ASSUME(L-OPCLASS): see ASSUMPTIONS.md
const VAR_LAT: &[&str] = &[
    // `BMOV.32.CLEAR R2, B11` with wr=SB2, its restore `BMOV.32 B11, R2` waiting on SB2.
    "BMOV",
    "LD",
    "LDG",
    "LDGMC",
    "LDS",
    "LDL",
    "LDC",
    "LDCU",
    "LDSM",
    "LDTM",
    "LDGSTS",
    "ATOM",
    "ATOMG",
    "ATOMS",
    "S2R",
    "S2UR",
    "CS2R.32",
    "REDUX",
    "SYNCS",
    "UTCATOMSWS",
    "MUFU",
    "SHFL",
    "MATCH",
    "POPC",
    "FLO",
    "BREV",
    "I2F",
    "F2I",
    "F2F",
    "I2I",
    "FRND",
    "DMUL",
    "DADD",
    "DFMA",
    "QSPC",
    "LDGDEPBAR",
];

/// The f64 pipe: completes in issue order (probes/f64_queue). ASSUME(L-OPCLASS)
const F64_PIPE: &[&str] = &["DADD", "DMUL", "DFMA"];

/// The in-order completion queue of a scoreboard-free write (the cover rule in `check`): the
/// shared-memory queue, the conversion / transcendental unit, POPC / FLO, the f64 pipe (with the
/// conversions that have an f64 operand: ptxas covers `I2F.F64.U64.RZ R8` with no scoreboard
/// by a wait on a later `I2F.F64.RM`'s, not on the `I2F.U16`'s issued between; probes/cvt_i2f
/// = GB300), or the opcode itself. ASSUME(L-OPCLASS)
fn inorder_queue(opcode: &str) -> &str {
    let b = opcode.split('.').next().unwrap_or("");
    match b {
        "LDS" | "SHFL" | "LDSM" | "ATOMS" => "shared",
        "F2I" | "I2F" | "F2F" if opcode.split('.').any(|m| m == "F64") => "f64",
        "MUFU" | "F2I" | "I2F" | "F2F" | "FRND" => "xu",
        "POPC" | "FLO" => "bit",
        "DADD" | "DMUL" | "DFMA" | "DSETP" => "f64",
        b => b,
    }
}

/// Stall cycles after which an f64-pipe result with no write scoreboard is in its register:
/// ptxas for sm_103a pads such results with NOPs (`st=15` x 3, `st=4`) and reads them no
/// sooner than 128 cycles later (Triton layernorm / split-k / pps kernels). ASSUME(L-OPCLASS)
const F64_LAT: u16 = 128;

/// Instructions that read their register operands asynchronously (memory / async units).
/// ptxas only gives some of them a read scoreboard: the units read operands in issue order,
/// so a read scoreboard on a later one covers the earlier ones (validated by the mutation
/// sweep, docs/sass-ctrl.md).
// ASSUME(L-OPREAD, L-OPCLASS): see ASSUMPTIONS.md
const ASYNC_READ: &[&str] = &[
    "LD",
    "LDG",
    "LDGMC",
    "LDS",
    "LDL",
    "LDC",
    "LDSM",
    "LDGSTS",
    "ST",
    "STG",
    "STS",
    "STL",
    "STSM",
    "ATOM",
    "ATOMG",
    "ATOMS",
    "RED",
    "REDG",
    "SYNCS",
    "LDTM",
    "STTM",
    "UTMALDG",
    "UTMASTG",
    "UTMAPF",
    "UTCHMMA",
    "UTCQMMA",
    "UTCOMMA",
    "UTCBAR",
    "UTCATOMSWS",
    "UTCCP",
    "UTCSHIFT",
];

/// Vector memory-pipe instructions whose operand reads a later SHFL cannot overtake.
// ASSUME(L-OPREAD): see ASSUMPTIONS.md
// LDGMC (multimem.ld_reduce): the same LSU queue as LDG (symm_mem _ReduceScatterKernel:
// `LDGMC ... R16, [R8.64+URZ]` then `LDGMC ... R8, [R8.64+UR20]` with no read scoreboard;
// both ranks' outputs = GB300, MN). ASSUME(L-OPREAD)
const MIO_INORDER: &[&str] = &[
    "LD", "LDG", "LDS", "ST", "STG", "STS", "STSM", "LDSM", "ATOMS", "SHFL", "LDGSTS", "LDL",
    "STL", "LDGMC",
];

fn width_of(opcode: &str) -> u16 {
    let mods: Vec<&str> = opcode.split('.').skip(1).collect();
    // LDSM.16.M88.4: four registers; HMMA.16816.F32.*: a 4-register accumulator.
    if opcode.starts_with("LDSM.") {
        // LDSM.8.MT1616[.2]: two registers per 16x16 b8 matrix.
        let per = if mods.contains(&"MT1616") { 2 } else { 1 };
        return per * mods.last().and_then(|n| n.parse().ok()).unwrap_or(1);
    }
    if opcode.starts_with("HMMA.16816.F32") {
        return 4;
    }
    // LDGMC.E.<op>.<type>: the access width (F32x4 / BF16x8: 16 bytes, 64 / F64: 8).
    if opcode.starts_with("LDGMC.") {
        let t = mods.get(2).copied().unwrap_or("");
        let lanes: u16 = t
            .rsplit_once('x')
            .and_then(|(_, n)| n.parse().ok())
            .unwrap_or(1);
        let bits: u16 = match t.split('x').next().unwrap_or("") {
            "64" | "S64" | "F64" => 64,
            "F16" | "BF16" => 16,
            "E4M3" | "E5M2" => 8,
            _ => 32,
        };
        return (bits * lanes / 32).max(1);
    }
    // {LDTM,STTM}.16dp256bit[.xN]: 4N registers; .16dp128bit[.xN]: 2N.
    let per = match mods.first() {
        Some(&"16dp256bit") => 4,
        Some(&"16dp128bit") => 2,
        _ => 0,
    };
    if per > 0 && (opcode.starts_with("LDTM.") || opcode.starts_with("STTM.")) {
        let n = mods
            .iter()
            .find_map(|m| m.strip_prefix('x').and_then(|n| n.parse::<u16>().ok()))
            .unwrap_or(1);
        return per * n;
    }
    if let Some(n) = mods
        .iter()
        .find_map(|m| m.strip_prefix('x').and_then(|n| n.parse::<u16>().ok()))
    {
        return n;
    }
    if mods.contains(&"128") {
        4
    } else if mods.contains(&"64") || mods.contains(&"WIDE") {
        2
    } else {
        1
    }
}

/// (defs, uses) of one instruction.
pub fn def_use(l: &SassLine) -> (Vec<RegKey>, Vec<RegKey>) {
    let base = l.opcode.split('.').next().unwrap_or("");
    let w = width_of(&l.opcode);
    let ops = &l.operands;
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    if let Some(g) = &l.guard {
        operand_regs(g.trim_start_matches('!'), 1, &mut uses);
    }
    // An address operand first (`SYNCS.CCTL.IV [UR7+0xe010]`) is never a destination.
    let ndef = if NO_DEF.contains(&base) || ops.is_empty() || ops[0].starts_with('[') {
        0
    } else if base == "VOTE" || base == "VOTEU" {
        2.min(ops.len())
    } else if base == "FCHK" {
        1 // FCHK Pd, Ra, Rb: a predicate only
    } else if is_pred_tok(&ops[0]) {
        if ops.len() > 2 && is_plain_reg(&ops[1]) && !ops[1].starts_with('!') {
            2
        } else {
            1
        }
    } else {
        1 + ops[1..]
            .iter()
            .take_while(|o| is_pred_tok(o) && !o.starts_with('!'))
            .count()
    };
    // `LDTM.STAT.xN.<op>.F32 Rred, Rd, tmem[..]` (tcgen05.ld.red): Rred and N registers Rd...
    if base == "LDTM" && l.opcode.starts_with("LDTM.STAT.") && ops.len() == 3 {
        operand_regs(&ops[0], 1, &mut defs);
        operand_regs(&ops[1], w, &mut defs);
        operand_regs(&ops[2], 1, &mut uses);
        return (defs, uses);
    }
    // `LDG.E.ENL2.256 Rhi, Rlo, [..]`: two 4-register destinations.
    if base == "LDG" && l.opcode.contains(".ENL2.256") && ops.len() == 3 {
        operand_regs(&ops[0], 4, &mut defs);
        operand_regs(&ops[1], 4, &mut defs);
        operand_regs(&ops[2], 1, &mut uses);
        return (defs, uses);
    }
    let store_like = matches!(
        base,
        "ST" | "STG" | "STS" | "STL" | "RED" | "REDG" | "ATOM" | "ATOMS" | "ATOMG"
    );
    for (i, o) in ops.iter().enumerate() {
        if o.starts_with('`') || o.starts_with('(') {
            continue; // labels
        }
        if i < ndef {
            operand_regs(o, if i == 0 && !store_like { w } else { 1 }, &mut defs);
        } else {
            let dw = if store_like && i == ops.len() - 1 {
                w
            } else if base == "HMMA" && w == 4 {
                // A: 4 registers, B: 2, C: 4.
                [1, 4, 2, 4][i.min(3)]
            } else {
                1
            };
            operand_regs(o, dw, &mut uses);
        }
    }
    if store_like && ndef == 1 {
        // ATOM*.64 returns 64 bits too.
        defs.clear();
        operand_regs(&ops[0], w, &mut defs);
    }
    (defs, uses)
}

/// Outstanding variable-latency operations at a program point (a may-analysis): key
/// (scoreboard, is-write, producer pc) -> what is pending. Scoreboard 6 = none.
type State = BTreeMap<(u8, bool, usize), Pend>;

#[derive(Clone, Debug, PartialEq)]
struct Pend {
    /// Registers possibly still pending.
    regs: BTreeSet<RegKey>,
    /// Minimum number of newer operations issued on the same scoreboard since (saturating).
    newer: u8,
    /// (A memory-queue read) also complete once any of these writes (scoreboard, pc) of a
    /// later memory-queue instruction completed: the queue is in order (on every path). A
    /// scoreboard with `RCOV` set is a later memory-queue instruction's *read* scoreboard.
    cov: BTreeSet<(u8, usize)>,
    /// The producer's guard predicate has not been redefined since (on every path).
    gok: bool,
    /// Minimum stall cycles issued since the producer (saturating; on every path).
    cyc: u16,
}

impl Pend {
    fn new() -> Pend {
        Pend {
            regs: BTreeSet::new(),
            newer: 0,
            cov: BTreeSet::new(),
            gok: true,
            cyc: 0,
        }
    }
}

const NO_SB: u8 = 6;
/// Tags a `Pend::cov` entry as a read scoreboard (else a write scoreboard).
const RCOV: u8 = 0x80;
/// Asynchronous read with no read scoreboard yet; attached to the next one that has one.
const UNTRACKED: u8 = 7;

fn join(into: &mut State, from: &State) -> bool {
    let mut changed = false;
    for (k, f) in from {
        match into.get_mut(k) {
            None => {
                into.insert(*k, f.clone());
                changed = true;
            }
            Some(p) => {
                for x in &f.regs {
                    changed |= p.regs.insert(*x);
                }
                if f.newer < p.newer {
                    p.newer = f.newer;
                    changed = true;
                }
                let cov: BTreeSet<_> = p.cov.intersection(&f.cov).copied().collect();
                if cov != p.cov {
                    p.cov = cov;
                    changed = true;
                }
                if p.gok && !f.gok {
                    p.gok = false;
                    changed = true;
                }
                if f.cyc < p.cyc {
                    p.cyc = f.cyc;
                    changed = true;
                }
            }
        }
    }
    changed
}

fn reg_name(r: RegKey) -> String {
    match r.0 {
        b'R' => format!("R{}", r.1),
        b'U' => format!("UR{}", r.1),
        b'P' => format!("P{}", r.1),
        _ => format!("UP{}", r.1),
    }
}

// ASSUME(L-STALL, L-MEMSB): see ASSUMPTIONS.md
pub fn check(lines: &[SassLine], code: &[Inst]) -> Result<Vec<StaticFinding>> {
    if lines.iter().any(|l| l.hi.is_none()) {
        bail!("scoreboard check needs `nvdisasm -hex` encodings");
    }
    let ctrl: Vec<Ctrl> = lines.iter().map(|l| Ctrl::decode(l.hi.unwrap())).collect();
    let du: Vec<_> = lines.iter().map(def_use).collect();
    let ret_sites: Vec<usize> = code
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i.op, Op::Call { .. }))
        .map(|(i, _)| i + 1)
        .collect();
    let succ = |pc: usize| -> Vec<usize> {
        let i = &code[pc];
        let next = if pc + 1 < code.len() {
            vec![pc + 1]
        } else {
            vec![]
        };
        match &i.op {
            Op::Bra {
                target,
                tid0_falls: false,
                div: None,
                ..
            } if i.guard.is_always() => vec![*target],
            Op::Bra { target, .. } => [vec![*target], next].concat(),
            Op::Exit | Op::Trap { .. } if i.guard.is_always() => vec![],
            Op::Call { target } => vec![*target],
            Op::Ret { .. } => ret_sites.clone(),
            Op::Brx { targets, .. } => targets.clone(),
            _ => next,
        }
    };
    // `DEPBAR.LE SBk, n`: at most n operations outstanding on SBk; `DEPBAR {5,4,3}`: none on
    // each listed scoreboard; `DEPBAR.LE SBk, n, {4,3}`: both.
    let mut depbar: Vec<Vec<(u8, u8)>> = vec![vec![]; lines.len()];
    for (pc, l) in lines.iter().enumerate() {
        if !l.opcode.starts_with("DEPBAR") {
            continue;
        }
        if l.opcode == "DEPBAR" && l.operands.len() == 1 && l.operands[0].starts_with('{') {
            for k in l.operands[0]
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
            {
                let sb = k.trim().parse::<u8>().map_err(|_| {
                    anyhow::anyhow!("sass {:#06x}: bad DEPBAR `{}`", l.addr, l.text)
                })?;
                depbar[pc].push((sb, 0));
            }
            continue;
        }
        let sb = l
            .operands
            .first()
            .and_then(|o| o.strip_prefix("SB"))
            .and_then(|k| k.parse::<u8>().ok());
        let n = l
            .operands
            .get(1)
            .and_then(|o| u8::from_str_radix(o.trim_start_matches("0x"), 16).ok());
        let (true, Some(sb), Some(n), 2..=3) = (l.opcode == "DEPBAR.LE", sb, n, l.operands.len())
        else {
            bail!("sass {:#06x}: unsupported DEPBAR form `{}`", l.addr, l.text)
        };
        depbar[pc].push((sb, n));
        if let Some(list) = l.operands.get(2) {
            let bad = || anyhow::anyhow!("sass {:#06x}: bad DEPBAR `{}`", l.addr, l.text);
            let list = list
                .strip_prefix('{')
                .and_then(|x| x.strip_suffix('}'))
                .ok_or_else(bad)?;
            for k in list.split(',') {
                depbar[pc].push((k.trim().parse::<u8>().map_err(|_| bad())?, 0));
            }
        }
    }
    let mut input: Vec<Option<State>> = vec![None; lines.len()];
    input[0] = Some(State::new());
    let mut work: Vec<usize> = vec![0];
    let mut found: HashMap<String, StaticFinding> = HashMap::new();
    while let Some(pc) = work.pop() {
        let mut st = input[pc].clone().unwrap();
        let c = &ctrl[pc];
        let l = &lines[pc];
        // Waits happen before issue. A completed write also means its own instruction has
        // read its operands.
        let done: BTreeSet<usize> = st
            .keys()
            .filter(|k| k.0 < NO_SB && k.1 && c.wait >> k.0 & 1 == 1)
            .map(|k| k.2)
            .collect();
        let done_w: BTreeSet<(u8, usize)> = st
            .keys()
            .filter(|k| k.0 < NO_SB && c.wait >> k.0 & 1 == 1)
            .map(|k| (if k.1 { k.0 } else { k.0 | RCOV }, k.2))
            .collect();
        st.retain(|k, v| {
            if !k.1 && v.cov.iter().any(|w| done_w.contains(w)) {
                false
            } else if k.0 == UNTRACKED {
                !done.contains(&k.2)
            } else {
                k.0 == NO_SB || c.wait >> k.0 & 1 == 0
            }
        });
        for &(sb, n) in &depbar[pc] {
            // Untracked reads are not counted by DEPBAR and stay pending (conservative).
            st.retain(|k, v| k.0 != sb || v.newer < n);
        }
        let (defs, uses) = &du[pc];
        // A shuffle reads and writes through the same in-order memory queue as the loads and
        // stores before it, so it cannot overwrite a register they have not read yet (rung
        // 3a: `STG ..., R21` with no read scoreboard, then `SHFL.BFLY PT, R21, ...`).
        // ASSUME(L-OPREAD): see ASSUMPTIONS.md
        if l.opcode.split('.').next() == Some("SHFL") {
            st.retain(|k, _| {
                k.0 != UNTRACKED
                    || !MIO_INORDER.contains(&lines[k.2].opcode.split('.').next().unwrap_or(""))
            });
        }
        let cguard = l.guard.as_deref();
        for (&(sb, write, ppc), pend) in &st {
            // `@P LDG R10` then `@!P MOV R10`: never both (P unchanged since).
            let excl = pend.gok
                && match (lines[ppc].guard.as_deref(), cguard) {
                    (Some(a), Some(b)) => {
                        a.trim_start_matches('!') == b.trim_start_matches('!')
                            && a.starts_with('!') != b.starts_with('!')
                    }
                    _ => false,
                };
            if excl {
                continue;
            }
            // A shuffle reads its operands in the queue, after the results of the shared loads
            // issued before it have landed (the row reduction above). ASSUME(L-OPREAD)
            let queued = write
                && sb == NO_SB
                && l.opcode.split('.').next() == Some("SHFL")
                && lines[ppc].opcode.split('.').next() == Some("LDS");
            // A load writes its destination when its data returns, after the earlier
            // instructions of the same in-order memory queue read their operands (torch's
            // reduce: `LDG.E R23, [R12.64]` with no read scoreboard, then `LDG.E R12, [R14.64]`).
            // ASSUME(L-OPREAD)
            let base_of = |i: usize| lines[i].opcode.split('.').next().unwrap_or("");
            // Same for a local load after cp.async (flash attention: `LDGSTS ..., [R2.64]`, its
            // read covered by a later LDGSTS's read scoreboard, then `LDL R2` with no wait).
            // Whether or not the earlier read is tracked by a scoreboard (s_sm80: `LDG.E R4,
            // [R32.64+0x150]` covered by a later read scoreboard, then `LDG R33` with no wait;
            // `STG ..., R5` then `LDSM R4`).
            // A shuffle too: it writes its result after passing the same queue (Inductor's
            // persistent RMSNorm: `@!P0 STG.E.128 [R32.64], R40` with rd=SB0, then
            // `SHFL.BFLY R41, R0, 0x10` with no wait; = GB300, docs/compile-vs-eager.md).
            // Global atomics read their operands in the same queue ahead of a later load's
            // write (ptxas: `ATOMG.E.EXCH PT, RZ, [R32], R8` with no read scoreboard, then
            // `LDG.E R8` with no wait; CUTLASS mx8 grouped GEMM's tensor-map update; ptxas only,
            // not probed).
            // The f64 pipe reads and writes in issue order: a later DADD / DMUL / DFMA cannot
            // overwrite a register an earlier one has not read (ptxas: `DADD R8, -R8, R10` with
            // rd=SB0, then `DMUL R10, R10, R10` with no wait; probes/f64_queue = GB300).
            // TMEM loads likewise write after the earlier TMEM stores and loads of the warp
            // read their registers (the TMEM queue).
            let queued_war = !write
                && (matches!(
                    base_of(pc),
                    "LDG" | "LDS" | "LD" | "LDL" | "LDSM" | "SHFL" | "LDGMC"
                ) && (MIO_INORDER.contains(&base_of(ppc))
                    || matches!(base_of(ppc), "ATOMG" | "ATOM" | "REDG" | "RED"))
                    || F64_PIPE.contains(&base_of(pc)) && F64_PIPE.contains(&base_of(ppc))
                    || base_of(pc) == "LDTM" && matches!(base_of(ppc), "STTM" | "LDTM"));
            // `LDTM.16dp32bit_t0_t15` then `_t16_t31` into the same registers: each writes only
            // its own half of the warp's lanes (PTX `.16x32bx2`, split by ptxas).
            let halves = |a: &str, b: &str| {
                let h = |s: &str| {
                    s.contains(".16dp32bit_t0_t15") as u8
                        | (s.contains(".16dp32bit_t16_t31") as u8) << 1
                };
                h(a) != 0 && h(b) != 0 && h(a) != h(b)
            };
            let other_half = write && halves(&lines[ppc].opcode, &l.opcode);
            for &r in &pend.regs {
                // The pending access is a read: the queued load may read the register too
                // (torch's one-shot all-reduce: `LDG.E.64 R8, [R6.64]` .. `LDG.E.64 R6,
                // [R6.64+0x18]`, its own address read also in queue order).
                if queued_war && defs.contains(&r) {
                    continue;
                }
                if other_half && defs.contains(&r) && !uses.contains(&r) {
                    continue;
                }
                let kind = if write && uses.contains(&r) && !queued {
                    if sb == NO_SB { "nosb" } else { "raw" }
                } else if defs.contains(&r) {
                    if write {
                        "waw"
                    } else if sb == UNTRACKED {
                        "war-nosb"
                    } else {
                        "war"
                    }
                } else {
                    continue;
                };
                // One finding per (hazard kind, producer); later consumers are listed in it.
                let sig = format!("sb:{kind}:{ppc}");
                if let Some(f) = found.get_mut(&sig) {
                    if !f.pcs.contains(&pc) {
                        f.pcs.push(pc);
                    }
                    continue;
                }
                let prod = &lines[ppc];
                let msg = if kind == "war-nosb" {
                    format!(
                        "SCOREBOARD WAR: {} overwritten at sass:{:#06x} `{}` while sass:{:#06x} `{}` may not have read it yet (asynchronous read with no read scoreboard, and none on a later read in the same unit)",
                        reg_name(r),
                        l.addr,
                        l.text,
                        prod.addr,
                        prod.text
                    )
                } else if kind == "nosb" {
                    format!(
                        "SCOREBOARD: {} read at sass:{:#06x} `{}`, but its producer sass:{:#06x} `{}` is variable-latency and has no write scoreboard",
                        reg_name(r),
                        l.addr,
                        l.text,
                        prod.addr,
                        prod.text
                    )
                } else {
                    let what = if kind == "raw" { "read" } else { "overwritten" };
                    let by = if write { "write" } else { "read" };
                    let why = if sb == NO_SB {
                        "it has no scoreboard".to_string()
                    } else {
                        format!("SB{sb}) may be outstanding: no wait on SB{sb} on some path (")
                    };
                    format!(
                        "SCOREBOARD {}: {} {what} at sass:{:#06x} `{}` while the {by} by sass:{:#06x} `{}` ({why}{})",
                        kind.to_uppercase(),
                        reg_name(r),
                        l.addr,
                        l.text,
                        prod.addr,
                        prod.text,
                        if sb == NO_SB {
                            String::new()
                        } else {
                            c.describe()
                        }
                    )
                };
                found.insert(
                    sig.clone(),
                    StaticFinding {
                        kind: "violation".into(),
                        sig,
                        msg,
                        pcs: vec![ppc, pc],
                    },
                );
            }
        }
        // A hazard is reported once; registers overwritten here are no longer pending (this
        // also keeps one bug from cascading).
        for p in st.values_mut() {
            p.regs.retain(|r| !defs.contains(r));
        }
        // A redefined predicate invalidates the guards that name it.
        for (k, p) in st.iter_mut() {
            if let Some(g) = &lines[k.2].guard {
                let mut gr = Vec::new();
                operand_regs(g.trim_start_matches('!'), 1, &mut gr);
                if gr.iter().any(|r| defs.contains(r)) {
                    p.gok = false;
                }
            }
        }
        let base = l.opcode.split('.').next().unwrap_or("");
        let var = VAR_LAT.contains(&base);
        let issue = |st: &mut State, sb: u8, write: bool, regs: BTreeSet<RegKey>| {
            for (k, v) in st.iter_mut() {
                if k.0 == sb {
                    v.newer = v.newer.saturating_add(1);
                }
            }
            // A previous instance of the same producer still pending merges with this one
            // (conservatively: its guard may differ, and it is not covered).
            let fresh = !st.contains_key(&(sb, write, pc));
            let e = st.entry((sb, write, pc)).or_insert_with(Pend::new);
            e.regs.extend(regs);
            e.newer = 0;
            e.cyc = 0;
            e.gok &= fresh;
            if !fresh {
                e.cov.clear();
            }
        };
        let mio = MIO_INORDER.contains(&base);
        // A memory instruction's operands are read before its result is written: once its
        // write scoreboard completes, so did its reads (torch's layer_norm: `LDG.E.64 R18,
        // [R18.64]` with wr=SB2, a wait on SB2, then R18 overwritten; no read scoreboard).
        // ASSUME(L-OPREAD): see ASSUMPTIONS.md
        let own_cov = |st: &mut State, rsb: u8| {
            if let (Some(wsb), Some(e)) = (c.wbar, st.get_mut(&(rsb, false, pc))) {
                e.cov.insert((wsb, pc));
            }
        };
        if let Some(sb) = c.wbar {
            // TMEM loads complete in issue order: ptxas gives only the last of a run of LDTMs a
            // write scoreboard and waits on it before using the earlier ones' registers too
            // (rung 2e). An earlier untracked LDTM is covered by this one's scoreboard. Same
            // for shuffles (rung 3a: `SHFL.BFLY` with no scoreboard, its result read after a
            // wait on a later `SHFL.DOWN`'s).
            // ASSUME(L-OPREAD): see ASSUMPTIONS.md
            // Shuffles and shared loads share the queue (rung 4, Triton row reduction:
            // `@!P1 LDS R3` with no scoreboard, `SHFL.BFLY R4, R3` with wr=SB0, then R3 read
            // after a wait on SB0).
            // Likewise `LDSM` (flash attention: `LDSM.16.MT88.4` with no scoreboard, its
            // registers used after a wait on a later `LDSM`'s).
            // And `S2UR` (`S2UR UR6, SR_CgaCtaId` with no scoreboard, read after a wait on a later
            // `S2UR`'s).
            // The conversion / transcendental unit is one queue too (radix sort: `MUFU.RCP R17`
            // with no scoreboard, read after a wait on a later `F2I.FTZ.U32.TRUNC`'s).
            // And value-returning shared atomics (`ATOMS.MIN R12` with no scoreboard, R12 read
            // after a wait on the next `ATOMS.MAX.S32 R13`'s; fixtures/s_atom_ext).
            // And `POPC` (ptxas for sm_103a: `POPC R6, R14` with no scoreboard, R6 read after a
            // wait on a later `POPC R17`'s; probes/int_misc = GB300 500/500 runs, the CuTe
            // DietGpu decode kernels).
            // And `FLO` with `POPC` (`FLO.U32 R0` with no scoreboard, R0 read after a wait on a
            // later `POPC`'s; probes/flo_popc = GB300 500/500 runs, DeepEP hybrid).
            // `FRND` is in the conversion queue too (Triton's mxint8 KV quantizer: `MUFU.RCP R17`
            // with no scoreboard, read after a wait on a later `FRND R16`'s; capture = GB300).
            // And `REDUX` (Triton's MoE `_select_kernel`: 20 `REDUX.SUM.S32 URd` with no write
            // scoreboard, each read after waits on later REDUXes' scoreboards; the capture's
            // output = GB300).
            // And the f64 pipe (`DFMA R2, -R12, R4, 1` with no scoreboard, R2 read after a wait
            // on a later `DADD R6`'s; probes/f64_queue = GB300, Triton's split-k LSE), DSETP
            // included (`DADD R6` read after a wait on a later `DSETP`'s; probes/f64_dsetp).
            if matches!(
                base,
                "LDTM"
                    | "SHFL"
                    | "LDS"
                    | "LDSM"
                    | "ATOMS"
                    | "MUFU"
                    | "S2UR"
                    | "F2I"
                    | "I2F"
                    | "F2F"
                    | "FRND"
                    | "POPC"
                    | "FLO"
                    | "REDUX"
                    | "DADD"
                    | "DMUL"
                    | "DFMA"
                    | "DSETP"
            ) {
                let mine = inorder_queue(&l.opcode);
                let ldtm: Vec<_> = st
                    .keys()
                    .filter(|k| k.0 == NO_SB && k.1 && inorder_queue(&lines[k.2].opcode) == mine)
                    .cloned()
                    .collect();
                for k in ldtm {
                    let p = st.remove(&k).unwrap();
                    let e = st.entry((sb, true, k.2)).or_insert_with(Pend::new);
                    e.regs.extend(p.regs);
                    e.gok &= p.gok;
                }
            }
            // TMEM stores read their registers in the TMEM queue before a later TMEM load's
            // result lands (flash attention: `STTM ..., R36` with no read scoreboard, an
            // `LDTM` with wr=SB1 waited on, then R36 overwritten). ASSUME(L-OPREAD)
            if base == "LDTM" {
                for (k, v) in st.iter_mut() {
                    if !k.1 && matches!(lines[k.2].opcode.split('.').next(), Some("STTM" | "LDTM"))
                    {
                        v.cov.insert((sb, pc));
                    }
                }
            }
            // Memory-queue reads issued before this write are done once it completes (rung 3a:
            // `STS [..], R24` with rd=SB1, `LDS R7` with wr=SB2, a wait on SB2, then R24
            // overwritten with no wait on SB1).
            if mio {
                for (k, v) in st.iter_mut() {
                    if !k.1
                        && MIO_INORDER.contains(&lines[k.2].opcode.split('.').next().unwrap_or(""))
                    {
                        v.cov.insert((sb, pc));
                    }
                }
            }
            if let (Some(rsb), "ATOM") = (c.rbar, base) {
                // A generic ATOM with a result register and a predicate (`ATOM.E.CAS.64 P0,
                // R8, [R22], R8, R10` with wr=SB5 rd=SB0, then `@P0 BRA` waiting SB0 only):
                // the predicate on the read scoreboard as below, the register on the write one.
                // ASSUME(L-OPREAD)
                let (p, r): (BTreeSet<RegKey>, BTreeSet<RegKey>) =
                    defs.iter().partition(|k| matches!(k.0, b'P' | b'Q'));
                if !p.is_empty() {
                    issue(&mut st, rsb, true, p);
                }
                issue(&mut st, sb, true, r);
            } else {
                issue(&mut st, sb, true, defs.iter().cloned().collect());
            }
        } else if let (true, Some(rsb), "ATOM") = (var && !defs.is_empty(), c.rbar, base) {
            // A generic ATOM's predicate result (`P` = performed: the address was global) is
            // known once its operands are read: ptxas waits on its read scoreboard (index_add:
            // `ATOM.E.ADD.F32... P2, RZ, [R8.64], R13` with rd=SB0, then `@P2 BRA` waiting SB0).
            // ASSUME(L-OPREAD)
            let (p, r): (BTreeSet<RegKey>, BTreeSet<RegKey>) =
                defs.iter().partition(|k| matches!(k.0, b'P' | b'Q'));
            issue(&mut st, rsb, true, p);
            if !r.is_empty() {
                issue(&mut st, NO_SB, true, r);
            }
        } else if var && !defs.is_empty() {
            issue(&mut st, NO_SB, true, defs.iter().cloned().collect());
        }
        // The guard predicate is consumed at issue; read scoreboards cover operands.
        let mut g = Vec::new();
        if let Some(gp) = &l.guard {
            operand_regs(gp.trim_start_matches('!'), 1, &mut g);
        }
        let reads: BTreeSet<RegKey> = uses.iter().filter(|r| !g.contains(r)).cloned().collect();
        // Vector instructions read uniform operands at issue (sweep: `LDTM tmem[UR4]` then
        // `R2UR UR4` with no wait; ptxas: `STG desc[UR6]` with rd=SB0 then `S2UR UR6` with no
        // wait); uniform-unit instructions (UTMA*, UTC*) read their URs late but predicate
        // operands at issue (ptxas, CuTeDSL GEMM: `UTCHMMA ..., UP1` with rd=SB2, then
        // `UISETP UP1` with no wait, and the wait on SB2 placed right before `ULEA UR48`).
        // ASSUME(L-OPREAD): see ASSUMPTIONS.md
        let uniform_unit = base.starts_with('U');
        let reads: BTreeSet<RegKey> = reads
            .into_iter()
            .filter(|r| r.0 == b'R' || uniform_unit && r.0 == b'U')
            .collect();
        if let (Some(sb), true) = (c.rbar, mio) {
            // Memory-queue reads issued before this one complete no later than it does: its
            // read scoreboard covers them, tracked or not (GB300: probes/sb_mio_order, an
            // `STS [Rt]` with no read scoreboard, then `LDG` and `STS [Rt+0x400]` with rd=SB4,
            // then `LDC Rt` waiting on SB4 only: 0 of 1000 runs wrong; with the scoreboard on
            // the first STS instead, every run loses the second STS's address; quack
            // rmsnorm_bwd N=256). Only where this instruction executes whenever the earlier
            // one did: unguarded, or the same guard not redefined since (a guarded-off
            // instruction may release its scoreboard at once). ASSUME(L-SB-ORDER)
            for (k, v) in st.iter_mut() {
                let same_guard = match (l.guard.as_deref(), lines[k.2].guard.as_deref()) {
                    (None, _) => true,
                    (Some(a), Some(b)) => a == b && v.gok,
                    _ => false,
                };
                if !k.1
                    && same_guard
                    && MIO_INORDER.contains(&lines[k.2].opcode.split('.').next().unwrap_or(""))
                {
                    v.cov.insert((sb | RCOV, pc));
                }
            }
        }
        if let Some(sb) = c.rbar {
            // In-order operand reads: earlier untracked reads of the same unit complete no
            // later than this one. Only the same unit: `SYNCS.ARRIVE.TRANS64 ..., R2` with no
            // read scoreboard is covered by a later `SYNCS.ARRIVE ..., R2` with rd=SB1 (ptxas
            // waits on SB1 before overwriting R2), not by a `UTMALDG` in between with rd=SB0,
            // which reads only uniform registers (Triton grouped fp8 GEMM). ASSUME(L-OPREAD)
            fn unit(b: &str) -> &str {
                if MIO_INORDER.contains(&b) {
                    "mio"
                } else if matches!(b, "ATOMG" | "ATOM" | "REDG" | "RED") {
                    "gatom"
                } else if matches!(b, "UTCHMMA" | "UTCQMMA" | "UTCOMMA" | "UTCBAR") {
                    // The tcgen05 issue queue (FA4 forward: `UTCBAR [UR61]` with no read
                    // scoreboard covered by the next `UTCHMMA`'s rd=SB2, waited on before UR61
                    // is rewritten; fixtures/c_fa4_causal = GB300).
                    "utc"
                } else {
                    b
                }
            }
            let untracked: Vec<_> = st
                .keys()
                .filter(|k| {
                    k.0 == UNTRACKED
                        && unit(lines[k.2].opcode.split('.').next().unwrap_or("")) == unit(base)
                })
                .cloned()
                .collect();
            for k in untracked {
                let p = st.remove(&k).unwrap();
                let e = st.entry((sb, false, k.2)).or_insert_with(Pend::new);
                e.regs.extend(p.regs);
                e.cov = p.cov;
                e.gok &= p.gok;
            }
            issue(&mut st, sb, false, reads);
            own_cov(&mut st, sb);
        } else if ASYNC_READ.contains(&base) && !reads.is_empty() {
            issue(&mut st, UNTRACKED, false, reads);
            own_cov(&mut st, UNTRACKED);
        }
        // Entries with no registers left only matter through the counts already applied, and
        // writes through their completion: a wait on a write scoreboard also completes its
        // instruction's operand reads (`ATOMG.E.EXCH PT, RZ, [R32], R6` with wr=SB4 and no
        // read scoreboard, a wait on SB4, then R32 overwritten; CUTLASS grouped wgrad).
        // ASSUME(L-OPREAD)
        st.retain(|k, v| !v.regs.is_empty() || k.1 && k.0 < NO_SB);
        // f64 results with no scoreboard are complete F64_LAT stall cycles after issue.
        // ASSUME(L-OPCLASS)
        for v in st.values_mut() {
            v.cyc = v.cyc.saturating_add(c.stall as u16);
        }
        st.retain(|k, v| {
            !(k.0 == NO_SB
                && k.1
                && v.cyc >= F64_LAT
                && F64_PIPE.contains(&lines[k.2].opcode.split('.').next().unwrap_or("")))
        });
        for s in succ(pc) {
            let changed = match &mut input[s] {
                None => {
                    input[s] = Some(st.clone());
                    true
                }
                Some(t) => join(t, &st),
            };
            if changed {
                work.push(s);
            }
        }
    }
    for f in found.values_mut() {
        if f.pcs.len() > 2 {
            f.pcs[2..].sort();
            let more: Vec<String> = f.pcs[2..]
                .iter()
                .map(|&p| format!("{:#06x}", lines[p].addr))
                .collect();
            f.msg += &format!("; also at {}", more.join(" "));
        }
    }
    let mut v: Vec<StaticFinding> = found.into_values().collect();
    v.sort_by(|a, b| (a.pcs[1], a.pcs[0]).cmp(&(b.pcs[1], b.pcs[0])));
    Ok(v)
}

/// File offset of section `name` in an ELF64 little-endian image.
fn elf_section(img: &[u8], name: &str) -> Result<(usize, usize)> {
    let u16_at = |o: usize| u16::from_le_bytes(img[o..o + 2].try_into().unwrap()) as usize;
    let u32_at = |o: usize| u32::from_le_bytes(img[o..o + 4].try_into().unwrap()) as usize;
    let u64_at = |o: usize| u64::from_le_bytes(img[o..o + 8].try_into().unwrap()) as usize;
    if img.len() < 64 || &img[..4] != b"\x7fELF" || img[4] != 2 || img[5] != 1 {
        bail!("not an ELF64 little-endian cubin");
    }
    let (shoff, shentsize, shnum, shstrndx) =
        (u64_at(0x28), u16_at(0x3a), u16_at(0x3c), u16_at(0x3e));
    let sh = |i: usize| shoff + i * shentsize;
    let stroff = u64_at(sh(shstrndx) + 0x18);
    for i in 0..shnum {
        let n = stroff + u32_at(sh(i));
        let end = n + img[n..].iter().position(|&b| b == 0).unwrap_or(0);
        if &img[n..end] == name.as_bytes() {
            return Ok((u64_at(sh(i) + 0x18), u64_at(sh(i) + 0x20)));
        }
    }
    bail!("section {name} not found")
}

/// Patch a cubin file in place; returns a log of the edits.
pub fn patch_file(
    cubin: &std::path::Path,
    kernel: &str,
    patches: &[CtrlPatch],
) -> Result<Vec<String>> {
    let mut img = std::fs::read(cubin)?;
    let dis = super::run_tool("nvdisasm", &["-c", "-hex", cubin.to_str().unwrap()])?;
    // Patches may name another kernel of the module (`CtrlPatch::kernel`).
    let mut log = Vec::new();
    let kernels: std::collections::BTreeSet<&str> = patches
        .iter()
        .map(|p| p.kernel.as_deref().unwrap_or(kernel))
        .collect();
    for k in kernels {
        let ps: Vec<CtrlPatch> = patches
            .iter()
            .filter(|p| p.kernel.as_deref().unwrap_or(kernel) == k)
            .cloned()
            .collect();
        log.extend(patch_cubin(&mut img, &dis, k, &ps)?);
    }
    std::fs::write(cubin, img)?;
    // Instruction edits: the result must disassemble as the case says.
    if patches.iter().any(|p| p.expect.is_some()) {
        let dis = super::run_tool("nvdisasm", &["-c", "-hex", cubin.to_str().unwrap()])?;
        for p in patches {
            let Some(want) = &p.expect else { continue };
            let (lines, _) = super::parse_disasm(&dis, p.kernel.as_deref().unwrap_or(kernel))?;
            let a =
                p.at.strip_prefix("@0x")
                    .and_then(|a| u32::from_str_radix(a, 16).ok());
            let l = lines
                .iter()
                .find(|l| Some(l.addr) == a || a.is_none() && l.text.contains(&p.at))
                .ok_or_else(|| anyhow::anyhow!("patch {}: instruction gone", p.at))?;
            let full = match &l.guard {
                Some(g) => format!("@{g} {}", l.text),
                None => l.text.clone(),
            };
            if full != *want {
                bail!(
                    "patch {}: disassembles as `{full}`, expected `{want}`",
                    p.at
                );
            }
        }
    }
    Ok(log)
}

/// Apply control-bit patches to a cubin (binary-only mutants: the PTX is untouched).
pub fn patch_cubin(
    img: &mut [u8],
    dis_hex: &str,
    kernel: &str,
    patches: &[CtrlPatch],
) -> Result<Vec<String>> {
    let (lines, _) = super::parse_disasm(dis_hex, kernel)?;
    let (off, size) = elf_section(img, &format!(".text.{kernel}"))?;
    let mut log = Vec::new();
    for p in patches {
        let addr =
            p.at.strip_prefix("@0x")
                .and_then(|a| u32::from_str_radix(a, 16).ok());
        let l = lines
            .iter()
            .find(|l| {
                if let Some(a) = addr {
                    l.addr == a
                } else {
                    l.text.contains(&p.at)
                }
            })
            .ok_or_else(|| anyhow::anyhow!("patch: no instruction matches `{}`", p.at))?;
        let o = off + l.addr as usize + 8;
        if l.addr as usize + 16 > size {
            bail!("patch: address outside .text");
        }
        let mut hi = u64::from_le_bytes(img[o..o + 8].try_into().unwrap());
        if Some(hi) != l.hi {
            bail!("patch: encoding mismatch at {:#06x}", l.addr);
        }
        let ol = off + l.addr as usize;
        let mut lo = u64::from_le_bytes(img[ol..ol + 8].try_into().unwrap());
        if p.nop {
            // The kernel's own padding NOP (ptxas ends every kernel with them).
            let n = lines
                .iter()
                .rev()
                .find(|x| x.opcode == "NOP" && x.guard.is_none())
                .ok_or_else(|| anyhow::anyhow!("patch: no NOP in {kernel}"))?;
            let (nlo, nhi) = (n.lo.unwrap_or(0), n.hi.unwrap_or(0));
            lo = nlo;
            hi = (nhi & ((1u64 << 41) - 1)) | (hi & !((1u64 << 41) - 1));
        }
        for &[start, width, value] in &p.bits {
            if start + width > 128 || width == 0 || width > 64 {
                bail!("patch: bad bit field [{start}, {width}]");
            }
            for k in 0..width {
                let b = start + k;
                let v = (value >> k) & 1;
                let w = if b < 64 { &mut lo } else { &mut hi };
                let sh = b % 64;
                *w = (*w & !(1u64 << sh)) | (v << sh);
            }
        }
        img[ol..ol + 8].copy_from_slice(&lo.to_le_bytes());
        let mut c = Ctrl::decode(hi);
        let before = c.describe();
        let sb = |x: u8| if x == 7 { None } else { Some(x) };
        if let Some(w) = p.wait {
            c.wait = w;
        }
        if let Some(w) = p.wbar {
            c.wbar = sb(w);
        }
        if let Some(r) = p.rbar {
            c.rbar = sb(r);
        }
        if let Some(s) = p.stall {
            c.stall = s;
        }
        img[o..o + 8].copy_from_slice(&c.encode(hi).to_le_bytes());
        log.push(format!(
            "sass:{:#06x} `{}`: {before} -> {}",
            l.addr,
            l.text,
            c.describe()
        ));
    }
    Ok(log)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str) -> SassLine {
        let (opcode, ops) = text.split_once(' ').unwrap_or((text, ""));
        SassLine {
            addr: 0,
            guard: None,
            opcode: opcode.into(),
            operands: super::super::split_ops(ops),
            text: text.into(),
            hi: None,
            lo: None,
        }
    }

    #[test]
    fn ctrl_roundtrip() {
        // LDTM.x16 R36, tmem[UR4]   (sm_103a, ptxas 13.0): wr SB1, no wait
        let c = Ctrl::decode(0x000e620008240000);
        assert_eq!((c.wbar, c.rbar, c.wait, c.stall), (Some(1), None, 0, 1));
        // LDTM.x16 R20, tmem[UR4+0x10]: wr SB0, waits SB1
        let c = Ctrl::decode(0x002e220008240000);
        assert_eq!((c.wbar, c.wait), (Some(0), 0b10));
        assert_eq!(c.encode(0x002e220008240000), 0x002e220008240000);
    }

    #[test]
    fn def_use_forms() {
        let r = |f: u8, i: u16| (f, i);
        assert_eq!(
            def_use(&line("LDTM.x16 R36, tmem[UR4]")),
            ((36..52).map(|i| r(b'R', i)).collect(), vec![r(b'U', 4)])
        );
        assert_eq!(
            def_use(&line("STG.E desc[UR14][R2.64+0x4], R37")).1,
            vec![
                r(b'U', 14),
                r(b'U', 15),
                r(b'R', 2),
                r(b'R', 3),
                r(b'R', 37)
            ]
        );
        assert_eq!(
            def_use(&line("ISETP.NE.AND P2, PT, R0, RZ, PT")),
            (vec![r(b'P', 2)], vec![r(b'R', 0)])
        );
        assert_eq!(
            def_use(&line("UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR4, UR4")),
            (vec![r(b'Q', 0), r(b'U', 4)], vec![r(b'U', 4)])
        );
        assert_eq!(
            def_use(&line("LDTM.STAT.x4.MAX.F32 R37, R4, tmem[UR6]")),
            (
                vec![r(b'R', 37), r(b'R', 4), r(b'R', 5), r(b'R', 6), r(b'R', 7)],
                vec![r(b'U', 6)]
            )
        );
        assert_eq!(
            def_use(&line("PLOP3.LUT P1, P0, !P0, PT, !P1, 0x80, 0xc8")),
            (vec![r(b'P', 1), r(b'P', 0)], vec![r(b'P', 0), r(b'P', 1)])
        );
        assert_eq!(def_use(&line("R2P PR, R0.B1, 0x7f")).1, vec![r(b'R', 0)]);
        assert_eq!(
            def_use(&line("PLOP3.LUT P0, PT, P1, PT, PT, 0x8, 0x0")),
            (vec![r(b'P', 0)], vec![r(b'P', 1)])
        );
        assert_eq!(
            def_use(&line("IADD3 R2, P0, PT, R4, R5, RZ")),
            (vec![r(b'R', 2), r(b'P', 0)], vec![r(b'R', 4), r(b'R', 5)])
        );
        assert_eq!(
            def_use(&line("IMAD.WIDE.U32 R2, R2, 0x100, R4")).0,
            vec![r(b'R', 2), r(b'R', 3)]
        );
        assert_eq!(
            def_use(&line("LDC.64 R4, c[0x0][0x480]")).0,
            vec![r(b'R', 4), r(b'R', 5)]
        );
        assert_eq!(
            def_use(&line("VOTE.ANY R5, PT, P0")),
            (vec![r(b'R', 5)], vec![r(b'P', 0)])
        );
    }
}
