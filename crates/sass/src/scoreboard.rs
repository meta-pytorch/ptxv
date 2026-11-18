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

/// Is `base` variable-latency in a module for SM version `sm` (ELF `e_flags`; 0: unknown,
/// treated as sm_103a)? ptxas 13.0 for sm_100 / sm_100a / sm_100f gives DADD / DMUL / DFMA no
/// write scoreboard and covers their results with stall counts alone (`DFMA R4, -R6, R2, 1`
/// st=8, then `DFMA R4, R4, R4, R4`), where for sm_103a it pads them with NOPs or scoreboards
/// them. This models a native sm_100 GPU: GB300 reschedules such modules at load when it must
/// (probes/f64_arch: the loaded image scoreboards every f64 op, whatever the cubin's stalls),
/// so a verdict on an sm_100 cubin is marked as not a GB300 verdict (`Kernel::static_image`).
/// The rule follows the module's SM, not the toolkit note's `-arch` (`CubinMeta::target`).
/// This is the only rule keyed on the SM. ASSUME(L-OPCLASS)
fn var_lat(base: &str, sm: u32) -> bool {
    VAR_LAT.contains(&base) && !(sm == 100 && F64_PIPE.contains(&base))
}

/// The in-order completion queue of a scoreboard-free write (the cover rule in `check`): the
/// shared-memory queue, POPC / FLO, the f64 pipe (with the conversions that have an f64
/// operand: ptxas covers `I2F.F64.U64.RZ R8` with no scoreboard by a wait on a later
/// `I2F.F64.RM`'s, not on the `I2F.U16`'s issued between; probes/cvt_i2f = GB300), or the
/// opcode itself. MUFU, F2I, I2F, F2F and FRND each keep their own: GB300 reads the stale
/// register when a later op of another of them carries the scoreboard (probes/xu_queue COVER,
/// an F2I covered by a later MUFU's, MISS 96; probes/frnd_queue COVER, a MUFU by a later
/// FRND's, MISS 82; 1000 / 1000 runs each, as their positive controls). `hw` (the GB300 rules)
/// adds no queue here: a pair joins only with a probe behind it. ASSUME(L-OPCLASS)
fn inorder_queue(opcode: &str, hw: bool, broad: bool) -> &str {
    let b = opcode.split('.').next().unwrap_or("");
    let f64_op = opcode.split('.').any(|m| m == "F64");
    match b {
        // The shared-memory queue across opcodes at the gb300 level only, and only where it is
        // measured (E-HW-SB-QUEUE: probes/shared_queue COVER, an LDS covered by a later SHFL's
        // write scoreboard, GB300 1000 / 1000 correct; not for an earlier SHFL: `check_rules`,
        // MISS 106). LDSM joins it at the gb300 level for the pairs probes/smem_queue_w1
        // measured (`hw_cover_measured`), else in the broad rules only, as ATOMS (captures:
        // fixtures/s_atom_ext; `HW_UNMEASURED_SIG`).
        "LDS" | "SHFL" if hw => "shared",
        // LDSM too, for the measured pairs only (`hw_cover_measured`: probes/smem_queue_w1).
        "LDSM" if hw => "shared",
        "LDSM" | "ATOMS" if broad => "shared",
        // The conversions: one queue per full opcode at the strict level, per base opcode under
        // the GB300 rules, where `conv_cover` decides which pairs (a later 32-bit F2I's write
        // scoreboard does not cover an earlier `F2I.S64`: GB300 wrong in 1000 / 1000, its
        // own-scoreboard reference correct, MISS 130). Those with an f64 operand: one queue per
        // opcode at the gb300 level (probes/cvt_i2f: `I2F.F64.U64.RZ` covered by a later
        // `I2F.F64.RM`, ptxas's schedule), covering there only the measured pairs
        // (`hw_cover_measured`, MISS 170), with the f64 pipe in the broad rules only.
        "F2I" | "I2F" | "F2F" if f64_op && broad => "f64",
        "F2I" if f64_op && hw => "F2I.F64",
        "I2F" if f64_op && hw => "I2F.F64",
        "F2F" if f64_op && hw => "F2F.F64",
        "F2I" | "I2F" | "F2F" if !hw => opcode,
        // MUFU and the conversions share no queue at either level: a later MUFU's write
        // scoreboard does not cover an earlier F2I (probes/xu_queue COVER: GB300 wrong in 1000
        // of 1000 runs; MISS 96). POPC / FLO and the f64 pipe are queues across opcodes at the
        // gb300 level, but no cross-opcode pair of them is measured with a discriminating
        // control (`hw_cover_measured`: the broad rules only, MISS 170).
        "POPC" | "FLO" if hw => "bit",
        "DADD" | "DMUL" | "DFMA" | "DSETP" if hw => "f64",
        // FRND under the GB300 rules (`check_rules`): one queue per operand width. A later f32
        // FRND's write scoreboard does not cover an earlier FRND.F64 (probes/frnd64 COVER: GB300
        // wrong in 1000 / 1000 runs at stalls 2, 4, 8, as its NOWAIT control; MISS 156). Measured
        // within a width: FRND by FRND.FLOOR, FRND.F64 by FRND.F64.FLOOR (probes/frnd_same, 0 /
        // 1000, the NOWAIT controls 1000 / 1000).
        "FRND" if hw && f64_op => "FRND.F64",
        b => b,
    }
}

/// Whether the gb300 rules' write cover of an earlier untracked `earlier` by a later `later`
/// (one `inorder_queue` at the gb300 level) lies inside a window measured on GB300 with a
/// discriminating control in the same layout (E-HW-SB-QUEUE). A pair outside is covered in the
/// broad rules only (`HW_UNMEASURED_SIG`, a contract at every level; MISS 170). A later probe
/// widens a window here. ASSUME(L-OPCLASS)
/// `gap`: the stall cycles from `earlier` to `later` (minimum, maximum over the paths);
/// `later_guarded`: `later` has a guard predicate.
fn hw_cover_measured(earlier: &str, later: &str, gap: (u16, u16), later_guarded: bool) -> bool {
    let base = |o: &str| o.split('.').next().unwrap_or("").to_string();
    let f64_op = |o: &str| o.split('.').any(|m| m == "F64");
    let (be, bl) = (base(earlier), base(later));
    // The strict rules' own queue (one opcode; their orders hold at every level), and FRND
    // within one operand width (probes/frnd_same: FRND by FRND.FLOOR, FRND.F64 by
    // FRND.F64.FLOOR, 0 / 1000, NOWAIT controls 1000 / 1000).
    if inorder_queue(earlier, false, false) == inorder_queue(later, false, false) {
        return true;
    }
    match (be.as_str(), bl.as_str()) {
        // probes/shared_queue COVER, lds128_shfl, sq_guard: an LDS by a later SHFL.
        ("LDS", "SHFL") => true,
        // probes/smem_queue_w1 (one warp: the conflict-free later op completes in 24 stall
        // cycles, the conflicted earlier one after 72) and smem_queue (8 warps): a
        // bank-conflicted `LDSM.16.M88.4` covered by a later 32-bit `LDS` or `SHFL.BFLY`, a
        // 32-bit `LDS` by a later `LDSM.16.M88.4` / `LDSM.16.MT88.4`, issued 2 to 8 stall
        // cycles after it, unguarded: COVER 0 / 1000 at gaps 2, 4, 8, NOWAIT and the reader
        // delayed past the later op's completion (DELAY) 1000 / 1000, the MUFU control 1000 /
        // 1000. (An LDSM by a later LDSM is the strict rules' queue.)
        ("LDSM", "LDS" | "SHFL") | ("LDS", "LDSM") => {
            matches!(
                (earlier, later),
                ("LDSM.16.M88.4", "LDS" | "SHFL.BFLY")
                    | ("LDS", "LDSM.16.M88.4" | "LDSM.16.MT88.4")
            ) && gap.0 >= 2
                && gap.1 <= 8
                && !later_guarded
        }
        // REDUX ops crossed (probes/redux_queue COVER, redux_ops AND / SUM.S32: 0 / 1000,
        // NOWAIT controls 1000 / 1000).
        ("REDUX", "REDUX") => true,
        // Conversions with no f64 operand: `conv_cover` holds the measured pairs.
        ("F2I" | "I2F" | "F2F", _) if !f64_op(earlier) && !f64_op(later) => true,
        // f64 conversions, one queue per base opcode: only the measured pairs (probes/f2i64_f64
        // COVER, i2f64_s16 COVER: 0 / 1000 at stalls 2, 4, 8, NOWAIT 1000 / 1000). The other
        // pairs, F2F.F64.F32 / F2F.F32.F64 included, are unmeasured.
        _ => matches!(
            (earlier, later),
            ("F2I.S64.F64.TRUNC", "F2I.F64.TRUNC") | ("I2F.F64.S16", "I2F.F64")
        ),
        // Not measured (each no discriminating control, RVS2 P2): FLO by a later POPC
        // (probes/flo_popc: one run of ptxas's schedule), the f64 pipe across DADD / DMUL /
        // DFMA / DSETP (f64_queue, f64_dsetp: `good` only).
    }
}

/// Whether the gb300 rules' queued WAR (`queued_unit`: a later op of an in-order unit
/// writing a register an earlier one of its queue reads, no wait) lies in a measured window.
/// None is: probes/xu_war's F2F.F64.F32 pair has no discriminating control (RVS2 P2, MISS
/// 170); the order is in the broad rules only. ASSUME(L-OPREAD)
fn hw_queued_war_measured(_reader: &str, _writer: &str) -> bool {
    false
}

/// Whether the gb300 rules' "UGETNEXTWORKID reads its URs at issue" is measured: no
/// (probes/clc_war has no layout where a late read would show, RVS2 P2, MISS 170); in the
/// broad rules only. ASSUME(L-OPREAD)
fn hw_ugetnextworkid_measured() -> bool {
    false
}

/// The later memory-queue op's scoreboard that covers an earlier op's register read: its read
/// scoreboard (R10: `RCOV`, and an untracked read attached to it) or its write scoreboard (R11).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Cover {
    Read,
    Write,
}

/// The upper end of the measured gap windows (stall cycles from the earlier op to the later
/// one, every path), probes/sb_mio_gap: the windows are 0 / 1000 from 4 to 2048, and the
/// controls in the same layouts (saturated memory pipe) wrong in every run up to 1024; at 2048
/// no control is wrong (an STS's read never stays pending that long there), so 2048 is no
/// measured window. Gaps beyond: the broad rules only. The static gap is a lower bound on the
/// real one (waits and contention only add cycles), so a window also stands for later real issue
/// than measured: 0 / 1000 at every nominal gap to 2048 (real to about 3101) and the in-order
/// queue support that (ASSUMPTIONS.md).
const HW_QUEUE_GAP_MAX: u16 = 1024;

/// Whether the gb300 rules' memory-queue cover of an earlier op's register read by a later op
/// of another opcode lies in a window measured on GB300 with a discriminating control in the
/// same layout (E-HW-SB-QUEUE, MISS 170 by the user's decision of 2026-11-12), the later op
/// unguarded and issued 4 to `HW_QUEUE_GAP_MAX` (1024) stall cycles after the earlier one on
/// every path (`gap`). probes/sb_mio_order (gaps 4 to 22) and probes/sb_mio_gap (gaps 4, 8, 16,
/// 22, 32, 45, 64, 128, 256, 512, 1024; 1184 CTAs x 256 threads, 64 warps per SM, shared memory
/// scrubbed; full occupancy alone and with a saturated memory pipe): 0 / 1000 runs wrong in each
/// window, and each control (the same window with no wait) wrong in every run of the saturated
/// layout. Measured through the later op's read and write scoreboards: a 32-bit `STS` by an
/// `LDS` (rd_on_lds, wcov), an `LDS.128` (cov_l128), an `LDG.E` (cov_ldg) or a `SHFL.BFLY`
/// (cov_shfl); an `STS.64` by an `LDS.64` (cov_s64_l64); an `STS.128` by an `LDS.128` or an
/// `LDS` (cov_s128_l128, cov_s128_lds); a 32-bit `STG.E` by an `LDS` (cov_stg_lds; each with
/// `_rd`). Through its write scoreboard only (`Cover::Write`): an `STS.64` / `STS.128` by a
/// `SHFL.BFLY` (c_s64_shfl, c_s128_shfl), an `STS.128` by an `LDSM.16.M88.4` or a `SHFL.IDX`
/// (c_s128_ldsm4, c_s128_sidx), an `STS` by a `SHFL.IDX` (c_s_sidx), an `STS.U8` / `STS.U16` by
/// an `LDS` (c_u8_lds, c_u16_lds), an `STG.E` by a `SHFL.BFLY`, an `LDG.E` or an `LDS.128`
/// (c_stg_*), an `STSM.16.M88.4` by an `LDS.128` or an `LDS` (c_stsm_*). Not measured, so the
/// broad rules only (`HW_UNMEASURED_SIG`): every other width and opcode pair (STG.E.128,
/// LDSM.16.MT88.4, ...), other gaps, global atomics across opcodes, and TMA stores' and
/// tcgen05 ops' reads attached to a later read scoreboard of their unit (captures only). An
/// earlier op of the later one's own base opcode is the strict rules' queue (`kb == base`).
/// ASSUME(L-OPREAD)
fn hw_mio_cover_measured(
    earlier: &str,
    later: &str,
    cover: Cover,
    gap: (u16, u16),
    later_guarded: bool,
) -> bool {
    let both = matches!(
        (earlier, later),
        ("STS", "LDS" | "LDS.128" | "LDG.E" | "SHFL.BFLY")
            | ("STS.64", "LDS.64")
            | ("STS.128", "LDS.128" | "LDS")
            | ("STG.E", "LDS")
    );
    let write = matches!(
        (earlier, later),
        ("STS", "SHFL.IDX")
            | ("STS.64", "SHFL.BFLY")
            | ("STS.128", "SHFL.BFLY" | "LDSM.16.M88.4" | "SHFL.IDX")
            | ("STS.U8" | "STS.U16", "LDS")
            | ("STG.E", "SHFL.BFLY" | "LDG.E" | "LDS.128")
            | ("STSM.16.M88.4", "LDS.128" | "LDS")
    );
    let pair = both || write && cover == Cover::Write;
    pair && gap.0 >= 4 && gap.1 <= HW_QUEUE_GAP_MAX && !later_guarded
}

/// Whether the gb300 rules' queued WAR across opcodes (R12: a load or shuffle writes a register
/// an earlier memory-queue op of another opcode reads, with no wait) lies in a measured window
/// (E-HW-SB-QUEUE, MISS 170, the user's decision of 2026-11-12): both unguarded, the writer
/// issued 4 to `HW_QUEUE_GAP_MAX` (1024) stall cycles after the reader on every path (`gap`).
/// probes/sb_mio_order (gaps 4 to 22) and probes/sb_mio_gap (gaps 4 to 1024, as for
/// `hw_mio_cover_measured`): 0 / 1000 each, the controls (an ALU write in the slot at the same
/// gap) wrong in every run of the saturated layout. Measured on the reader's address and data
/// registers: an `LDS` / `LDG.E` writing a 32-bit STS's (qwar_*, dq_lds, dq_ldg), a `SHFL.BFLY`
/// (a_s_shfl, dq_shfl); an `LDS.64` / `SHFL.BFLY` writing an `STS.64`'s (a_s64_*, dq_s64_*); an
/// `LDS.128` / `LDS` / `SHFL.BFLY` writing an `STS.128`'s (a_s128_*, dq_s128_*, d_s128_lds); an
/// `LDS` writing an `STS.U8` / `STS.U16`'s (a_u8 / a_u16, d_u8 / d_u16); a `SHFL.BFLY` writing a
/// 32-bit `STG.E`'s (a_stg_shfl, dq_stg_shfl). On the data registers only (`data`): an
/// `LDSM.16.M88`, `LDS.128` or `SHFL.IDX` writing an STS's (dq_ldsm, dq_q_l128, d_s_sidx); an
/// `LDSM.16.M88.4` / `SHFL.IDX` writing an `STS.128`'s (dq_s128_ldsm4, d_s128_sidx); an `LDS`,
/// `LDG.E` or `LDS.128` writing an `STG.E`'s (dq_stg_lds, d_stg_ldg, d_stg_l128); an
/// `LDS.128`, `LDSM.16.M88.4` or `LDS` writing an `STSM.16.M88.4`'s (d_stsm_*); an `LDS`
/// writing an `ATOMS.EXCH`'s 4 cycles after it (sb_mio_order dq_atoms_lds). Other pairs, widths
/// and gaps: the broad rules only. ASSUME(L-OPREAD)
fn hw_load_war_measured(reader: &str, writer: &str, data: bool, gap: (u16, u16)) -> bool {
    let within = |hi: u16| gap.0 >= 4 && gap.1 <= hi;
    match (reader, writer) {
        ("STS", "LDS" | "LDG.E" | "SHFL.BFLY")
        | ("STS.64", "LDS.64" | "SHFL.BFLY")
        | ("STS.128", "LDS.128" | "LDS" | "SHFL.BFLY")
        | ("STS.U8" | "STS.U16", "LDS")
        | ("STG.E", "SHFL.BFLY") => within(HW_QUEUE_GAP_MAX),
        ("STS", "LDSM.16.M88" | "LDS.128" | "SHFL.IDX")
        | ("STS.128", "LDSM.16.M88.4" | "SHFL.IDX")
        | ("STG.E", "LDS" | "LDG.E" | "LDS.128")
        | ("STSM.16.M88.4", "LDS.128" | "LDSM.16.M88.4" | "LDS") => {
            data && within(HW_QUEUE_GAP_MAX)
        }
        ("ATOMS.EXCH", "LDS") => data && within(4),
        _ => false,
    }
}

/// Whether register `r` is a data operand of memory instruction `l` (not in its address).
fn data_operand(l: &SassLine, r: RegKey) -> bool {
    let mut addr = Vec::new();
    for o in l.operands.iter().filter(|o| o.contains('[')) {
        operand_regs(o, 1, &mut addr);
    }
    !addr.contains(&r)
}

/// The in-order queue of a unit whose operand reads and result writes both follow issue order
/// (a later op of the same queue cannot write a register before an earlier one read it): the
/// conversion / transcendental queues and the f64 pipe. ASSUME(L-OPCLASS)
fn queued_unit(opcode: &str, hw: bool, broad: bool) -> Option<&str> {
    // At the gb300 level, and there only in the measured windows (`hw_queued_war_measured`:
    // none; probes/xu_war has no discriminating control, MISS 170).
    if !hw {
        return None;
    }
    let q = inorder_queue(opcode, true, broad);
    let b = opcode.split('.').next().unwrap_or("");
    (matches!(b, "MUFU" | "F2I" | "I2F" | "F2F") || q == "f64").then_some(q)
}

/// Whether a later conversion `later`'s write scoreboard covers an earlier untracked one
/// `earlier` of the same base opcode (no f64 operand), under the GB300 rules. Never an earlier
/// one with a 64-bit integer operand by a later one without (probes/cvt64_queue F2I64_COVER:
/// GB300 wrong in 1000 / 1000; MISS 130). Measured: the same full opcode, and an earlier 32-bit
/// F2I by a later `F2I.S64` (F2I32_COVER: 0 / 1000 wrong); other pairs in the broad rules only.
fn conv_cover(earlier: &str, later: &str, broad: bool) -> bool {
    let w64 = |o: &str| o.split('.').any(|m| m == "S64" || m == "U64");
    if w64(earlier) && !w64(later) {
        return false;
    }
    // An I2F from an 8- or 16-bit integer (no f64 operand) is covered by no later conversion,
    // its own full opcode included: a later I2F.S16's write scoreboard does not order an
    // I2F.S8's write (probes/i2f8_queue COVER, GB300 wrong at gaps of 9 to 15 stall cycles),
    // nor a later identical I2F's (probes/i2f8_same SAME_COVER / S16_COVER, wrong in 159 to 599
    // of 1000 runs; MISS 155). The f64 form keeps its queue (I2F.F64.S16 by a later I2F.F64:
    // RVS2's i2f64_s16, 0 / 1000, its NOWAIT control 1000 / 1000).
    if narrow_i2f(earlier) {
        return false;
    }
    earlier == later
        || broad
        || earlier.split('.').next() == Some("F2I") && !w64(earlier) && w64(later)
}

/// An I2F from an 8- or 16-bit integer with no f64 operand (`I2F.S8`, `I2F.U16.RZ`, ...): the
/// class probes/i2f8_queue and i2f8_same measure (`conv_cover`: no later conversion covers it).
fn narrow_i2f(opcode: &str) -> bool {
    let mut m = opcode.split('.');
    m.next() == Some("I2F")
        && !opcode.split('.').any(|x| x == "F64")
        && m.any(|x| matches!(x, "S8" | "U8" | "S16" | "U16"))
}

/// Stall cycles from which a read (or a WAW overwrite) of an f64-pipe result (DADD / DMUL /
/// DFMA) with no write scoreboard that nothing covers rests on the pipe's latency (`sb:lat:`, a
/// contract no level discharges); sooner, a violation (MISS 151). No level trusts the latency:
/// ptxas for sm_103a reads such results 128 counted cycles after issue (Triton layernorm /
/// split-k / pps kernels), which GB300 refutes. GB300 (a LOP3 reader, D = stall cycles from the
/// DADD): wrong up to D = 416 with the f64 pipe saturated at full occupancy (probes/f64_lat_sat,
/// the _Q layout: 2 / 1000 runs at 384, 1 / 1000 at 416; 1000 / 1000 up to 352), 0 / 1000 at every
/// 32-step D from 448 to 640 and at 704 in every layout (672 not run; 0 / 10000 at 448 and 480 in
/// the _Q layout); up to 320 at full occupancy without the saturating pool (f64_lat_occ), 160
/// with one CTA (f64_lat). 624 is 1.5 times the worst wrong D (the user's rule: at or above the
/// first D clean in the worst layout, 448, and 1.5 times the worst wrong one). ASSUME(L-OPCLASS)
const F64_LAT: u16 = 624;

/// Stall cycles from which a read (or a WAW overwrite) of an `I2F.S8` result with no write
/// scoreboard that nothing covers rests on the conversion's latency (`sb:lat:`, a contract no
/// level discharges); sooner, a violation (MISS 186). GB300 (an FMUL reader, D = stall cycles
/// from the I2F): wrong up to D = 33 with the conversion pipe saturated at full occupancy
/// (probes/i2f8_lat_sat: 883 / 1000 runs at 33), 0 / 1000 at every D from 34 to 39 and at 40,
/// 47, 48, 49, 50 there; up to 24 at full occupancy (i2f8_lat_occ), 22 with one CTA (i2f8_lat); a
/// WAW lost up to 16 with one CTA (i2f8_lat WAW). 50 is the first D at or above 1.5 times the
/// worst wrong D (the user's T = 48 rule on a basis of 32; 33 was found wrong since).
/// ASSUME(L-OPCLASS)
const CONV_LAT_CYC: u16 = 50;

/// The conversions whose result latency a GB300 probe measured (`CONV_LAT_CYC`): `I2F.S8` (f32
/// from a signed byte, probes/i2f8_lat, i2f8_lat_occ, i2f8_lat_sat). Other narrow forms
/// (`I2F.S16`, `I2F.U8`, `I2F.F16.S16`, `I2F.BF16.S16`, `I2F.F16.U8`, ...) have no measured
/// window: a late uncovered read of their result is a violation (MISS 186). ASSUME(L-OPCLASS)
fn lat_measured(opcode: &str) -> bool {
    opcode == "I2F.S8"
}

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
    // The bulk copy reads its operands late too (MISS 131), and so do the remote shared
    // store, the TMA reduction and the bulk prefetch: ptxas gives each a read scoreboard
    // (`STAS [R2.64], R0` rd=SB1, `UTMAREDG.2D.ADD [UR8], [UR6]` rd=SB2, `UBLKPF.L2 [UR4], UR9`
    // rd=SB1; MISS 157, probes/async_read_ops).
    "UBLKCP",
    "STAS",
    "UTMAREDG",
    "UBLKPF",
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

/// Whether an instruction's register destinations are 64-bit pairs, and which of its sources
/// (by index among the operands after the destinations) are, where the opcode's modifiers say so
/// without a `.64` operand suffix (MISS 181: a read of only the high register of a DADD result
/// went unchecked; GB300 wrong in 1000 / 1000 runs, probes/f64_lat K0_D8_NOWAIT). A bare register
/// operand of width 2 stands for Rn and Rn+1 (`operand_regs`). ASSUME(L-OPCLASS)
fn wide64(opcode: &str) -> (bool, Box<dyn Fn(usize) -> bool>) {
    let mut m = opcode.split('.');
    let base = m.next().unwrap_or("");
    let mods: Vec<&str> = m.collect();
    let has = |x: &str| mods.contains(&x);
    let int64 = has("S64") || has("U64");
    let all = |d: bool| -> (bool, Box<dyn Fn(usize) -> bool>) { (d, Box::new(|_| true)) };
    let none = |d: bool| -> (bool, Box<dyn Fn(usize) -> bool>) { (d, Box::new(|_| false)) };
    match base {
        // The f64 pipe: every register operand a pair (DSETP: a predicate result).
        "DADD" | "DMUL" | "DFMA" | "DMNMX" => all(true),
        "DSETP" | "DSET" => all(false),
        "FRND" if has("F64") => all(true),
        // F2F.<dst>.<src>: the first type modifier is the destination's.
        "F2F" => {
            let ty: Vec<&str> = mods
                .iter()
                .copied()
                .filter(|x| matches!(*x, "F16" | "BF16" | "F32" | "F64" | "E4M3" | "E5M2" | "TF32"))
                .collect();
            let src = ty.get(1) == Some(&"F64");
            (ty.first() == Some(&"F64"), Box::new(move |_| src))
        }
        // I2F: an F64 is the destination, a 64-bit integer the source; F2I the reverse.
        "I2F" => (has("F64"), Box::new(move |_| int64)),
        "F2I" => {
            let src = has("F64");
            (int64, Box::new(move |_| src))
        }
        // f64 / 64-bit atomics: the returned value and the data operands.
        "ATOM" | "ATOMG" | "ATOMS" | "RED" | "REDG"
            if has("64") || has("F64") || int64 =>
        {
            all(true)
        }
        // IMAD.WIDE Rd, Ra, Rb, Rc: Rd and Rc pairs.
        "IMAD" if has("WIDE") => (true, Box::new(|i| i == 2)),
        _ => none(false),
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
    // An address operand first (`SYNCS.CCTL.IV [UR7+0xe010]`, a TMEM address: `UTCCP ...
    // tmem[UR23+0x104], gdesc[UR4]`, `STTM.x16 tmem[UR6], R36`) is never a destination (before,
    // its uniform register counted as written; MISS 181 made the read visible).
    let addr_first = ops.first().is_some_and(|o| {
        o.starts_with('[') || ["tmem[", "gdesc[", "desc["].iter().any(|p| o.starts_with(p))
    });
    let ndef = if NO_DEF.contains(&base) || ops.is_empty() || addr_first {
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
    // `LDG.E.ENL2.256 Rhi, Rlo, [..][, mask]`: two 4-register destinations, or with a mask
    // as many registers per half as it selects words there.
    if base == "LDG" && l.opcode.contains(".256") && (ops.len() == 3 || ops.len() == 4) {
        let mask = ops
            .get(3)
            .and_then(|m| m.trim().strip_prefix("0x"))
            .and_then(|h| u8::from_str_radix(h, 16).ok())
            .unwrap_or(0xff);
        operand_regs(&ops[0], (mask >> 4).count_ones() as u16, &mut defs);
        operand_regs(&ops[1], (mask & 0xf).count_ones() as u16, &mut defs);
        operand_regs(&ops[2], 1, &mut uses);
        return (defs, uses);
    }
    let store_like = matches!(
        base,
        "ST" | "STG" | "STS" | "STL" | "RED" | "REDG" | "ATOM" | "ATOMS" | "ATOMG"
    );
    // 64-bit register operands written without a `.64` suffix (MISS 181): f64 results and
    // sources are register pairs, as are 64-bit integer conversion operands and f64 / 64-bit
    // atomic data.
    let (d64, s64) = wide64(&l.opcode);
    for (i, o) in ops.iter().enumerate() {
        if o.starts_with('`') || o.starts_with('(') {
            continue; // labels
        }
        if i < ndef {
            let dw = if d64 { 2 } else if i == 0 && !store_like { w } else { 1 };
            operand_regs(o, dw, &mut defs);
        } else {
            let dw = if s64(i - ndef) {
                2
            } else if store_like && i == ops.len() - 1 {
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
        operand_regs(&ops[0], if d64 { 2 } else { w }, &mut defs);
    }
    (defs, uses)
}

/// The scoreboard key of a lifted register, None for the zero / true registers (RZ, URZ, PT,
/// UPT) and virtual ones.
fn lifted_key(r: ptxv_ir::Reg) -> Option<RegKey> {
    use ptxv_ir::File;
    match r.file {
        File::R if r.idx < crate::RZ => Some((b'R', r.idx)),
        File::UR if r.idx < crate::URZ => Some((b'U', r.idx)),
        File::P if r.idx != 7 => Some((b'P', r.idx)),
        File::UP if r.idx != 7 => Some((b'Q', r.idx)),
        _ => None,
    }
}

/// (defs, uses) of one instruction: `def_use`'s, joined with every register its lift writes /
/// reads (`Op::dst_regs` / `src_regs`, which span each operand's 32-bit slots). The lift is the
/// one source of operand widths: a register group the text form does not spell out (a uniform
/// pair `LDCU.64 UR6`, `STG.E.ENL2.256`'s eight data registers, `REDG.E.ADD.F32x4`'s four, a TMA
/// op's uniform group: MISS 101, 181) is tracked whole; `def_use` keeps what the lift does not
/// model (an instruction lowered to a no-op still names its registers). ASSUME(L-OPREAD)
fn def_use_lifted(l: &SassLine, op: Option<&Op>) -> (Vec<RegKey>, Vec<RegKey>) {
    let (mut defs, mut uses) = def_use(l);
    if let Some(op) = op {
        for (regs, out) in [(op.dst_regs(), &mut defs), (op.src_regs(), &mut uses)] {
            for k in regs.into_iter().filter_map(lifted_key) {
                if !out.contains(&k) {
                    out.push(k);
                }
            }
        }
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
    /// Maximum stall cycles issued since the producer over the paths, capped at `CMAX_CAP` (a
    /// gap window's upper end: `hw_cover_measured`).
    cmax: u16,
}

/// Cap of `Pend::cmax` (every window it bounds is shorter; the cap keeps loops' fixpoint short).
const CMAX_CAP: u16 = HW_QUEUE_GAP_MAX + 1;

impl Pend {
    fn new() -> Pend {
        Pend {
            regs: BTreeSet::new(),
            newer: 0,
            cov: BTreeSet::new(),
            gok: true,
            cyc: 0,
            cmax: 0,
        }
    }
}

const NO_SB: u8 = 6;
/// Tags a `Pend::cov` entry as a read scoreboard (else a write scoreboard).
const RCOV: u8 = 0x80;
/// Asynchronous read with no read scoreboard yet; attached to the next one that has one.
const UNTRACKED: u8 = 7;

/// Moves an earlier pending operation `p` onto a later instruction's scoreboard (its entry `k`:
/// an untracked read or an unscoreboarded result attached to it), called right after that
/// instruction's issue. The entry completes with it, the newest operation on the scoreboard:
/// `newer` 0, also over an entry already there (the minimum; MISS 264). The gap counters keep
/// counting from the earlier operation (the minimum / maximum with an entry already there), so
/// the windows measured from it stay measured from it, not from the attach (MISS 250). Its guard
/// holds and a cover applies only if they did for both.
fn attach(st: &mut State, k: (u8, bool, usize), p: Pend) {
    let fresh = !st.contains_key(&k);
    let e = st.entry(k).or_insert_with(Pend::new);
    e.newer = 0;
    if fresh {
        (e.cyc, e.cmax, e.cov) = (p.cyc, p.cmax, p.cov);
    } else {
        e.cyc = e.cyc.min(p.cyc);
        e.cmax = e.cmax.max(p.cmax);
        e.cov = e.cov.intersection(&p.cov).copied().collect();
    }
    e.gok &= p.gok;
    e.regs.extend(p.regs);
}

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
                if f.cmax > p.cmax {
                    p.cmax = f.cmax;
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

/// `DEPBAR.LE SBk, n`: at most n operations outstanding on SBk; `DEPBAR {5,4,3}`: none on
/// each listed scoreboard; `DEPBAR.LE SBk, n, {4,3}`: both. (scoreboard, n) pairs; empty for
/// any other instruction.
pub fn depbar_waits(l: &SassLine) -> Result<Vec<(u8, u8)>> {
    let mut out = vec![];
    if !l.opcode.starts_with("DEPBAR") {
        return Ok(out);
    }
    let bad = || anyhow::anyhow!("sass {:#06x}: bad DEPBAR `{}`", l.addr, l.text);
    if l.opcode == "DEPBAR" && l.operands.len() == 1 && l.operands[0].starts_with('{') {
        for k in l.operands[0]
            .trim_matches(|c| c == '{' || c == '}')
            .split(',')
        {
            out.push((k.trim().parse::<u8>().map_err(|_| bad())?, 0));
        }
        return Ok(out);
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
    out.push((sb, n));
    if let Some(list) = l.operands.get(2) {
        let list = list
            .strip_prefix('{')
            .and_then(|x| x.strip_suffix('}'))
            .ok_or_else(bad)?;
        for k in list.split(',') {
            out.push((k.trim().parse::<u8>().map_err(|_| bad())?, 0));
        }
    }
    Ok(out)
}

/// Signature prefix of a hazard that only the GB300 queue rules (E-HW-SB-QUEUE: FRND within a
/// width, the measured conversion pairs, an LDS by a later SHFL, the measured memory-queue
/// covers and queued load WARs across opcodes, `REDUX` in order) clear: a strict finding that
/// the gb300 profile discharges.
pub const HW_QUEUE_SIG: &str = "sb-hwq:";

/// Signature prefix of a hazard that only the broad queue rules clear: orders ptxas's schedules
/// and kernel captures rely on but no GB300 probe measured (LDSM / ATOMS in the shared-memory
/// queue, f64 conversions in the f64 pipe and the unmeasured f64 conversion pairs, POPC / FLO
/// and the f64 pipe across opcodes, the conversion / f64 units' operand-read order,
/// UGETNEXTWORKID's reads at issue, STTM reads before a later LDTM, the memory-queue covers and
/// queued load WARs across opcodes outside their measured windows, TMA stores' and tcgen05 ops'
/// untracked reads covered by a later read scoreboard of their unit; MISS 170). A contract
/// finding at every level: the gb300 profile does not discharge it.
pub const HW_UNMEASURED_SIG: &str = "sb-unmeasured:";

/// The static scoreboard check (`check_rules`) under the strict rules, the GB300 queue rules
/// measured on GB300 (E-HW-SB-QUEUE) and the broad rules: a hazard the measured rules clear
/// is a strict contract finding with the `HW_QUEUE_SIG` prefix (the gb300 profile discharges
/// it); one only the broad rules clear a contract with the `HW_UNMEASURED_SIG` prefix (no
/// level discharges it); the rest as found.
pub fn check(lines: &[SassLine], code: &[Inst], sm: u32) -> Result<Vec<StaticFinding>> {
    let strict = check_rules(lines, code, sm, false, false)?;
    let hw = check_rules(lines, code, sm, true, false)?;
    let broad = check_rules(lines, code, sm, true, true)?;
    // The levels are compared per hazard, keyed by (producer, consumer, register) (P1 of the
    // RVS2 review: comparing `sb:kind:producer` signatures mislabelled reads the strict rules
    // prove safe). A hazard the strict rules do not find is none: their orders hold at every
    // level, and a broader level can only lose a cover there (its first-match move onto
    // another queue member). One the strict rules find is a violation if the gb300 and broad
    // rules find it too (an `sb:lat:` latency contract stays a contract), an `sb-unmeasured:`
    // contract if only the broad rules clear it (no level discharges it), and an `sb-hwq:`
    // contract if the gb300 rules clear it (the gb300 profile discharges it).
    let mut tagged = Vec::new();
    for (k, h) in strict {
        let short = format!("{}:{}", h.kind, h.ppc);
        let (sig, kind, note) = if !hw.contains_key(&k) {
            (
                format!("{HW_QUEUE_SIG}{short}"),
                "contract",
                " (cleared by GB300's queue rules only, E-HW-SB-QUEUE: measured on GB300, a bound that holds at the gb300 profile)",
            )
        } else if !broad.contains_key(&k) {
            (
                format!("{HW_UNMEASURED_SIG}{short}"),
                "contract",
                " (cleared only by queue orders ptxas relies on that no GB300 probe measured, E-HW-SB-QUEUE: a contract at every level)",
            )
        } else if h.kind == "lat" {
            (format!("sb:{short}"), "contract", "")
        } else {
            (format!("sb:{short}"), "violation", "")
        };
        tagged.push((sig, kind.to_string(), note.to_string(), h));
    }
    Ok(aggregate(lines, tagged))
}

// ASSUME(L-STALL, L-MEMSB): see ASSUMPTIONS.md
fn check_rules(
    lines: &[SassLine],
    code: &[Inst],
    sm: u32,
    hw: bool,
    broad: bool,
) -> Result<HashMap<(usize, usize, RegKey), Hz>> {
    if lines.iter().any(|l| l.hi.is_none()) {
        bail!("scoreboard check needs `nvdisasm -hex` encodings");
    }
    let ctrl: Vec<Ctrl> = lines.iter().map(|l| Ctrl::decode(l.hi.unwrap())).collect();
    let du: Vec<_> = lines
        .iter()
        .enumerate()
        .map(|(pc, l)| def_use_lifted(l, code.get(pc).map(|i| &i.op)))
        .collect();
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
    let depbar: Vec<Vec<(u8, u8)>> = lines.iter().map(depbar_waits).collect::<Result<_>>()?;
    let mut input: Vec<Option<State>> = vec![None; lines.len()];
    input[0] = Some(State::new());
    let mut work: Vec<usize> = vec![0];
    // Every hazard, keyed by (producer pc, consumer pc, register): `check` compares the levels
    // on that key (a hazard's kind may differ between levels, and one producer can have
    // several hazards).
    let mut found: HashMap<(usize, usize, RegKey), Hz> = HashMap::new();
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
        // No SHFL clears earlier untracked memory-queue reads: GB300 rewrites the register
        // before an earlier STS reads it, with a SHFL between and with an ALU MOV in its slot
        // alike (1000 / 1000 runs wrong each; MISS 128, probes/sb_mio_order rvs_shfl_retain).
        // A SHFL's own write after them is `queued_war`'s.
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
            // A SHFL reads its operands at issue, not after an earlier LDS's result lands
            // (GB300: wrong in 1000 / 1000 runs at stalls 4 and 8, as its ALU-reader control;
            // MISS 129, probes/shared_queue RVS_QREAD_LDS_*; probes/shfl_qread for SHFL after
            // SHFL).
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
            // Each conversion / transcendental opcode likewise within itself (no queue across
            // MUFU, F2I, I2F, F2F: MISS 96), and the conversions with an f64 operand with the
            // f64 pipe (`inorder_queue`): CuTe top-k, ptxas for sm_103a and
            // sm_100a, `F2F.F64.F32 R98, R6` with rd=SB1, then `F2F.F64.F32 R6, R19` with no
            // wait on SB1 (probes/xu_war; ptxas's own schedules give such reads no scoreboard).
            // In the broad rules only: xu_war has no discriminating control
            // (`hw_queued_war_measured`, MISS 170).
            // TMEM loads likewise write after the earlier TMEM stores and loads of the warp
            // read their registers (the TMEM queue).
            // Across opcodes at the gb300 level in its measured windows only (R12,
            // `hw_load_war_measured`: probes/sb_mio_order and sb_mio_gap), in the broad rules
            // otherwise; the same opcode's at strict.
            let load_war = !write
                && matches!(
                    base_of(pc),
                    "LDG" | "LDS" | "LD" | "LDL" | "LDSM" | "SHFL" | "LDGMC"
                )
                && (MIO_INORDER.contains(&base_of(ppc))
                    || matches!(base_of(ppc), "ATOMG" | "ATOM" | "REDG" | "RED"));
            let load_war_r = |r: RegKey| {
                load_war
                    && (base_of(pc) == base_of(ppc)
                        || broad
                        || hw
                            && hw_load_war_measured(
                                &lines[ppc].opcode,
                                &lines[pc].opcode,
                                data_operand(&lines[ppc], r),
                                (pend.cyc, pend.cmax),
                            ))
            };
            let queued_war = !write
                && (queued_unit(&lines[pc].opcode, hw, broad)
                        == Some(inorder_queue(&lines[ppc].opcode, true, broad))
                        && (lines[pc].opcode == lines[ppc].opcode
                            || broad
                            || !matches!(base_of(pc), "F2I" | "I2F" | "F2F"))
                        && (broad || hw_queued_war_measured(&lines[ppc].opcode, &lines[pc].opcode))
                    || base_of(pc) == "LDTM"
                        && (base_of(ppc) == "LDTM" || base_of(ppc) == "STTM" && broad));
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
            // Shared loads of a warp write back in issue order: a later `LDS` may overwrite a
            // register an earlier one has not written yet, and its value is the one left
            // (ptxas 13.0 / 13.3: `LDS.128 R4, [R0]` then `LDS R6, [R0+0x10]` with no wait;
            // GB300: probes/lds_waw, a 32-way bank-conflicted `LDS.128` then a broadcast `LDS`
            // into its third register, 0 of 26.2M threads keep the earlier value, while an ALU
            // write in its place is overwritten by the late `LDS.128` in every thread).
            // Only where the later load runs whenever the earlier did (unguarded, or the same
            // guard unchanged): a guarded-off LDS writes nothing. ASSUME(L-OPREAD)
            let lds_waw = write
                && base_of(pc) == "LDS"
                && base_of(ppc) == "LDS"
                && (cguard.is_none() || pend.gok && lines[ppc].guard.as_deref() == cguard);
            for &r in &pend.regs {
                // The pending access is a read: the queued load may read the register too
                // (torch's one-shot all-reduce: `LDG.E.64 R8, [R6.64]` .. `LDG.E.64 R6,
                // [R6.64+0x18]`, its own address read also in queue order).
                if (queued_war || load_war_r(r)) && defs.contains(&r) {
                    continue;
                }
                if other_half && defs.contains(&r) && !uses.contains(&r) {
                    continue;
                }
                if lds_waw && defs.contains(&r) && !uses.contains(&r) {
                    continue;
                }
                let kind = if write && uses.contains(&r) {
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
                // An unscoreboarded conversion read later than any distance measured wrong:
                // ptxas relies on its latency, which the check cannot bound (L-OPCLASS): a
                // contract naming the assumption, not a violation (the captured Triton
                // dequantize_kv: `I2F.S8 R12` read 33 stall cycles later).
                // Only the measured form (`lat_measured`: I2F.S8); other conversions stay
                // violations (xu_queue COVER_ST: an F2I read 17 cycles later behind a MUFU,
                // GB300 wrong in every run; the narrow forms no probe measured, MISS 186).
                // A WAW on its destination likewise (the later write must land after the
                // conversion's: the same latency, decided by the user 2026-11-12): sooner than
                // CONV_LAT_CYC a violation, from it a contract.
                // An f64-pipe result likewise from F64_LAT (MISS 151: no level trusts ptxas's
                // 128 cycles; sooner a violation).
                let f64 = F64_PIPE.contains(&lines[ppc].opcode.split('.').next().unwrap_or(""));
                let latency = (kind == "nosb" || kind == "waw" && sb == NO_SB)
                    && (lat_measured(&lines[ppc].opcode) && pend.cyc >= CONV_LAT_CYC
                        || f64 && pend.cyc >= F64_LAT);
                let kind = if latency { "lat" } else { kind };
                if found.contains_key(&(ppc, pc, r)) {
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
                } else if kind == "lat" {
                    let why = if f64 {
                        format!("ptxas relies on the f64 pipe's latency, which no level bounds (GB300: outstanding up to 416 stall cycles with the f64 pipe saturated at full occupancy, 0 / 1000 at 448 to 640 and 704, probes/f64_lat_sat; sooner than {F64_LAT} a violation; L-OPCLASS)")
                    } else {
                        format!("ptxas relies on the conversion's latency, which no level bounds (GB300: outstanding up to 33 stall cycles with the conversion pipe saturated at full occupancy, 0 / 1000 from 34 to 50, probes/i2f8_lat_sat; sooner than {CONV_LAT_CYC} a violation; L-OPCLASS)")
                    };
                    format!(
                        "SCOREBOARD LATENCY: {} {} at sass:{:#06x} `{}` {} stall cycles after its producer sass:{:#06x} `{}`, which has no write scoreboard: {why}",
                        reg_name(r),
                        if uses.contains(&r) { "read" } else { "overwritten" },
                        l.addr,
                        l.text,
                        pend.cyc,
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
                    (ppc, pc, r),
                    Hz {
                        kind,
                        ppc,
                        pc,
                        msg,
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
        let var = var_lat(base, sm);
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
            e.cmax = 0;
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
            // Unscoreboarded results this one's write scoreboard covers: attached after its
            // issue (`attach`).
            let mut ldtm = Vec::new();
            // TMEM loads complete in issue order: ptxas gives only the last of a run of LDTMs a
            // write scoreboard and waits on it before using the earlier ones' registers too
            // (rung 2e). An earlier untracked LDTM is covered by this one's scoreboard. Same
            // for shuffles (rung 3a: `SHFL.BFLY` with no scoreboard, its result read after a
            // wait on a later `SHFL.DOWN`'s).
            // ASSUME(L-OPREAD): see ASSUMPTIONS.md
            // Shuffles and shared loads share the queue in one direction, at the gb300 level
            // (rung 4, Triton row reduction: `@!P1 LDS R3` with no scoreboard, `SHFL.BFLY R4,
            // R3` with wr=SB0, then R3 read after a wait on SB0; probes/shared_queue COVER):
            // an earlier SHFL is not covered by a later LDS (REV_COVER, MISS 106).
            // Likewise `LDSM` (flash attention: `LDSM.16.MT88.4` with no scoreboard, its
            // registers used after a wait on a later `LDSM`'s).
            // And `S2UR` (`S2UR UR6, SR_CgaCtaId` with no scoreboard, read after a wait on a later
            // `S2UR`'s).
            // MUFU and the conversions are no queue across opcodes (`inorder_queue`, MISS 96):
            // ptxas's radix sort reads a `MUFU.RCP R17` with no scoreboard after a wait on a
            // later `F2I.FTZ.U32.TRUNC`'s, but the reverse order is refuted on GB300
            // (probes/xu_queue COVER) and no probe backs this one.
            // And value-returning shared atomics (`ATOMS.MIN R12` with no scoreboard, R12 read
            // after a wait on the next `ATOMS.MAX.S32 R13`'s; fixtures/s_atom_ext).
            // And `POPC` (ptxas for sm_103a: `POPC R6, R14` with no scoreboard, R6 read after a
            // wait on a later `POPC R17`'s; probes/int_misc: ptxas's own schedule, GB300's output = the model, no positive control; the CuTe
            // DietGpu decode kernels).
            // And `FLO` with `POPC` in the broad rules only (`FLO.U32 R0` with no scoreboard, R0
            // read after a wait on a later `POPC`'s; probes/flo_popc: ptxas's own schedule, one
            // recorded GB300 run = the model, no positive control; DeepEP hybrid; MISS 170).
            // Nor `FRND` with MUFU (probes/frnd_queue: GB300 reads a
            // `MUFU.RCP` result before it lands after a wait on a later `FRND`'s scoreboard).
            // `REDUX` completes in issue order under the GB300 rules only (Triton's MoE
            // `_select_kernel`: 20 `REDUX.SUM.S32 URd` with no write scoreboard, each read after
            // waits on later REDUXes' scoreboards; probes/redux_queue COVER = GB300 1000 / 1000,
            // its positive control wrong 1000 / 1000: a measured null, E-HW-SB-QUEUE).
            // And the f64 pipe, across DADD / DMUL / DFMA / DSETP in the broad rules only
            // (`DFMA R2, -R12, R4, 1` with no scoreboard, R2 read after a wait on a later `DADD
            // R6`'s; probes/f64_queue = GB300, Triton's split-k LSE; `DADD R6` read after a wait
            // on a later `DSETP`'s, probes/f64_dsetp: `good` runs only, no control; MISS 170).
            // REDUX only under the GB300 rules (`hw`, E-HW-SB-QUEUE: a capture, no probe).
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
                    | "POPC"
                    | "FLO"
                    | "DADD"
                    | "DMUL"
                    | "DFMA"
                    | "DSETP"
            ) || hw && matches!(base, "REDUX" | "FRND")
            {
                let mine = inorder_queue(&l.opcode, hw, broad);
                // An untracked SHFL result is covered by a later SHFL's write scoreboard only,
                // at either level: GB300 reads it stale after a wait on a later LDS's
                // (probes/shared_queue REV_COVER, 1000 / 1000 runs wrong, as its positive
                // control; MISS 106).
                let shfl = |o: &str| o.split('.').next() == Some("SHFL");
                ldtm = st
                    .keys()
                    .filter(|k| {
                        k.0 == NO_SB
                            && k.1
                            && inorder_queue(&lines[k.2].opcode, hw, broad) == mine
                            // A gb300 pair only inside its measured window (MISS 170).
                            && (!hw
                                || broad
                                || hw_cover_measured(
                                    &lines[k.2].opcode,
                                    &l.opcode,
                                    (st[*k].cyc, st[*k].cmax),
                                    l.guard.is_some(),
                                ))
                            && (!shfl(&lines[k.2].opcode) || shfl(base))
                            && (!matches!(base, "F2I" | "I2F" | "F2F")
                                || l.opcode.split('.').any(|m| m == "F64")
                                || conv_cover(&lines[k.2].opcode, &l.opcode, broad))
                            // An earlier POPC by a later FLO: the broad rules only (measured:
                            // FLO by a later POPC, probes/flo_popc; POPC by POPC, int_misc).
                            && (broad
                                || lines[k.2].opcode.split('.').next() != Some("POPC")
                                || base != "FLO")
                    })
                    .cloned()
                    .collect();
            }
            // TMEM stores read their registers in the TMEM queue before a later TMEM load's
            // result lands (flash attention: `STTM ..., R36` with no read scoreboard, an
            // `LDTM` with wr=SB1 waited on, then R36 overwritten). ASSUME(L-OPREAD)
            if base == "LDTM" {
                for (k, v) in st.iter_mut() {
                    let kb = lines[k.2].opcode.split('.').next().unwrap_or("");
                    // An STTM's (another opcode's) in the broad rules only (captures; no GB300
                    // probe with a discriminating control: `HW_UNMEASURED_SIG`).
                    if !k.1 && (kb == "LDTM" || kb == "STTM" && broad) {
                        v.cov.insert((sb, pc));
                    }
                }
            }
            // Memory-queue reads issued before this write are done once it completes (rung 3a:
            // `STS [..], R24` with rd=SB1, `LDS R7` with wr=SB2, a wait on SB2, then R24
            // overwritten with no wait on SB1).
            // Another opcode's reads at the gb300 level in its measured windows only (R11,
            // `hw_mio_cover_measured` with `Cover::Write`: probes/sb_mio_order wcov, sb_mio_gap
            // cov_* / c_*), in the broad rules otherwise; the same opcode's at strict.
            if mio {
                for (k, v) in st.iter_mut() {
                    let kb = lines[k.2].opcode.split('.').next().unwrap_or("");
                    let measured = || {
                        hw_mio_cover_measured(
                            &lines[k.2].opcode,
                            &l.opcode,
                            Cover::Write,
                            (v.cyc, v.cmax),
                            l.guard.is_some(),
                        )
                    };
                    if !k.1 && MIO_INORDER.contains(&kb) && (kb == base || broad || hw && measured())
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
            for k in ldtm {
                let p = st.remove(&k).unwrap();
                attach(&mut st, (sb, true, k.2), p);
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
        // UGETNEXTWORKID (clusterlaunchcontrol.try_cancel) is the broad rules' exception: it consumes its
        // address URs at issue (GB300, probes/clc_war: the response address register rewritten
        // by the next instruction, with or without a read scoreboard on UGETNEXTWORKID, never
        // moves the response; ptxas rewrites UR8 / UR9 after it with no wait on its rd
        // scoreboard in TLX CLC kernels). ASSUME(L-OPREAD): see ASSUMPTIONS.md
        // (Measured: no; the broad rules only, MISS 170.)
        let uniform_unit = base.starts_with('U')
            && (base != "UGETNEXTWORKID" || !(broad || hw && hw_ugetnextworkid_measured()));
        let reads: BTreeSet<RegKey> = reads
            .into_iter()
            .filter(|r| r.0 == b'R' || uniform_unit && r.0 == b'U')
            .collect();
        // At the gb300 level only (E-HW-SB-QUEUE: probes/sb_mio_order, stg_guard_war), and
        // across opcodes only in the measured window (R10, MISS 170).
        if let (Some(sb), true) = (c.rbar, mio && hw) {
            // Memory-queue reads issued before this one complete no later than it does: its
            // read scoreboard covers them, tracked or not (GB300: probes/sb_mio_order, an
            // `STS [Rt]` with no read scoreboard, then `LDG` and `STS [Rt+0x400]` with rd=SB4,
            // then `LDC Rt` waiting on SB4 only: 0 of 1000 runs wrong; with the scoreboard on
            // the first STS instead, every run loses the second STS's address; quack
            // rmsnorm_bwd N=256). Only where this instruction executes whenever the earlier
            // one did: unguarded, or the same guard not redefined since (a guarded-off
            // instruction may release its scoreboard at once), except between global stores:
            // a guarded-off `STG`'s read scoreboard is released in queue order too (GB300:
            // probes/stg_guard_war, `STG.E.64 [R2.64], R14` with no read scoreboard, then
            // `@P0 STG` with rd=SB0 and P0 false, then R2 or R14 rewritten after a wait on SB0
            // only: 0 of 5.2M threads wrong; without the wait every thread stores the rewritten
            // value; Triton TBE forward). ASSUME(L-SB-ORDER)
            let stg = |b: &str| b.split('.').next() == Some("STG");
            for (k, v) in st.iter_mut() {
                let same_guard = match (l.guard.as_deref(), lines[k.2].guard.as_deref()) {
                    (None, _) => true,
                    (Some(a), Some(b)) => a == b && v.gok,
                    _ => false,
                } || stg(base) && stg(&lines[k.2].opcode);
                // Another opcode's in its measured windows only (R10, `hw_mio_cover_measured` with
                // `Cover::Read`: sb_mio_order rd_on_lds, sb_mio_gap cov_*_rd), in the broad rules
                // otherwise; the same base opcode's as the strict rules' attachment below (a
                // scoreboard never waited on changes nothing in the unit's order).
                let kb = lines[k.2].opcode.split('.').next().unwrap_or("");
                if !k.1
                    && same_guard
                    && MIO_INORDER.contains(&kb)
                    && (kb == base
                        || broad
                        || hw_mio_cover_measured(
                            &lines[k.2].opcode,
                            &l.opcode,
                            Cover::Read,
                            (v.cyc, v.cmax),
                            l.guard.is_some(),
                        ))
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
                } else if matches!(b, "UTCHMMA" | "UTCQMMA" | "UTCOMMA" | "UTCBAR" | "UTCCP") {
                    // The tcgen05 issue queue (FA4 forward: `UTCBAR [UR61]` with no read
                    // scoreboard covered by the next `UTCHMMA`'s rd=SB2, waited on before UR61
                    // is rewritten; fixtures/c_fa4_causal = GB300). `UTCCP` too: the GB300
                    // driver's own sm_103 image of a DeepGEMM sm_100 GEMM issues
                    // `UTCCP.T.S.4x32dp128bit ..., gdesc[UR12]` with no read scoreboard and
                    // rewrites UR13 after waiting only on a later `UTCQMMA`'s rd scoreboard;
                    // fixtures/r3_cute_mxf8 with its UTCCPs' read scoreboards removed = GB300
                    // 1000/1000 (`utccp_in_mma_issue_order`).
                    "utc"
                } else {
                    b
                }
            }
            // A TMA store's or a tcgen05 op's untracked reads attach to a later read
            // scoreboard of its unit in the broad rules only (R13, MISS 170: captures, and
            // UTCCP's 1000 / 1000 runs and probes/utmastg_queue have no positive control). So do
            // another opcode's of the memory queue, but for the measured window at the gb300
            // level (R10, `hw_mio_cover_measured` with `Cover::Read`: probes/sb_mio_order
            // rd_on_lds, sb_mio_gap rd_on_lds and cov_*_rd) and global atomics across opcodes
            // (not measured). The same opcode's stay at the
            // strict level: the read-order analogue of a same-opcode completion queue, which
            // ptxas's schedules rest on throughout (it gives only the last store of a run a read
            // scoreboard and rewrites the earlier ones' registers after waiting on it; GB300:
            // sb_mio_order inorder).
            let untracked: Vec<_> = st
                .iter()
                .filter(|(k, v)| {
                    let ub = unit(lines[k.2].opcode.split('.').next().unwrap_or(""));
                    let kb = lines[k.2].opcode.split('.').next().unwrap_or("");
                    k.0 == UNTRACKED
                        && ub == unit(base)
                        && (broad
                            || !matches!(ub, "UTMASTG" | "utc") && kb == base
                            || hw
                                && ub == "mio"
                                && hw_mio_cover_measured(
                                    &lines[k.2].opcode,
                                    &l.opcode,
                                    Cover::Read,
                                    (v.cyc, v.cmax),
                                    l.guard.is_some(),
                                ))
                })
                .map(|(k, _)| *k)
                .collect();
            issue(&mut st, sb, false, reads);
            for k in untracked {
                let p = st.remove(&k).unwrap();
                attach(&mut st, (sb, false, k.2), p);
            }
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
        for v in st.values_mut() {
            v.cyc = v.cyc.saturating_add(c.stall as u16);
            v.cmax = v.cmax.saturating_add(c.stall as u16).min(CMAX_CAP);
        }
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
    Ok(found)
}

/// One hazard found by `check_rules`: its kind (`raw`, `nosb`, `lat`, ...), producer and
/// consumer pcs and message.
#[derive(Clone, Debug)]
struct Hz {
    kind: &'static str,
    ppc: usize,
    pc: usize,
    msg: String,
}

/// One finding per (signature, producer), its consumers listed: `sig` and `kind` per hazard.
fn aggregate(lines: &[SassLine], hz: Vec<(String, String, String, Hz)>) -> Vec<StaticFinding> {
    let mut by: BTreeMap<String, Vec<(String, String, Hz)>> = BTreeMap::new();
    for (sig, kind, note, h) in hz {
        by.entry(sig).or_default().push((kind, note, h));
    }
    let mut v = Vec::new();
    for (sig, mut hs) in by {
        hs.sort_by_key(|(_, _, h)| h.pc);
        let (kind, note, first) = hs[0].clone();
        let mut pcs = vec![first.ppc];
        for (_, _, h) in &hs {
            if !pcs[1..].contains(&h.pc) {
                pcs.push(h.pc);
            }
        }
        let mut msg = first.msg.clone();
        if pcs.len() > 2 {
            let more: Vec<String> = pcs[2..]
                .iter()
                .map(|&p| format!("{:#06x}", lines[p].addr))
                .collect();
            msg += &format!("; also at {}", more.join(" "));
        }
        msg += &note;
        v.push(StaticFinding {
            kind,
            sig,
            msg,
            pcs,
        });
    }
    v.sort_by(|a, b| (a.pcs[1], a.pcs[0]).cmp(&(b.pcs[1], b.pcs[0])));
    v
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

    /// MISS 181 (RVS3 H1..H3): the scoreboard tracks every register the lift reads or writes,
    /// each operand's whole register group (`def_use_lifted`: the lift is the one source of
    /// widths). Every form below that lifts: its lifted registers are all tracked; the forms
    /// the text alone under-counted, as GB300 measured them (rvs3 repros: `LDCU.64 UR6` read at
    /// UR7, STG.E.ENL2.256's R5 / R9 / R11 overwritten, wrong 1000 / 1000) and by the ISA.
    #[test]
    fn lift_widths_are_tracked() {
        let regs = |f: u8, r: std::ops::Range<u16>| r.map(|i| (f, i)).collect::<Vec<_>>();
        let mut lifted = 0;
        for t in WIDTH_FORMS {
            let l = line(t);
            let Ok(op) = crate::lower_text(t, None) else {
                continue; // not modeled: such a kernel has no verdict
            };
            lifted += 1;
            let (d, u) = def_use_lifted(&l, Some(&op));
            for k in op.dst_regs().into_iter().filter_map(lifted_key) {
                assert!(d.contains(&k), "{t}: def {k:?} untracked: {d:?}");
            }
            for k in op.src_regs().into_iter().filter_map(lifted_key) {
                assert!(u.contains(&k), "{t}: use {k:?} untracked: {u:?}");
            }
        }
        assert!(lifted >= 50, "{lifted}");
        let du = |t: &str| def_use_lifted(&line(t), Some(&crate::lower_text(t, None).unwrap()));
        let has = |got: &[RegKey], want: &[RegKey], t: &str| {
            for k in want {
                assert!(got.contains(k), "{t}: {k:?} missing from {got:?}");
            }
        };
        for (t, defs, uses) in [
            // H1: uniform register pairs / quads.
            ("LDCU.64 UR4, c[0x0][0x390]", regs(b'U', 4..6), vec![]),
            ("LDCU.128 UR4, c[0x0][0x390]", regs(b'U', 4..8), vec![]),
            ("DADD R4, R2, UR6", regs(b'R', 4..6), regs(b'U', 6..8)),
            ("DSETP.MAX.AND P0, P2, R4, UR4, PT", vec![], regs(b'U', 4..6)),
            ("IMAD.WIDE.U32 R2, R0, 0x4, UR6", regs(b'R', 2..4), regs(b'U', 6..8)),
            ("UIMAD.WIDE.U32 UR4, UR6, UR7, UR8", regs(b'U', 4..6), regs(b'U', 8..10)),
            ("UIADD3.64 UR4, UPT, UPT, UR4, 0x8, URZ", regs(b'U', 4..6), regs(b'U', 4..6)),
            // H2: STG.E.ENL2.256's eight data registers.
            ("STG.E.ENL2.256 desc[UR4][R2.64], R8, R4", vec![], regs(b'R', 4..12)),
            // H3: f32 / bf16 vector atomics' data and result.
            ("REDG.E.ADD.F32x4.FTZ.RN.STRONG.GPU desc[UR8][R2.64], R4", vec![], regs(b'R', 4..8)),
            ("ATOMG.E.ADD.F32x4.FTZ.RN.STRONG.GPU PT, R8, desc[UR8][R2.64], R4", regs(b'R', 8..12), regs(b'R', 4..8)),
            // RVS3 R1: BF16x4's 8 bytes of data (probes/atom_bf16x4's forms).
            ("REDG.E.ADD.BF16x4.RN.STRONG.GPU desc[UR4][R4.64+0x4000], R10", vec![], regs(b'R', 10..12)),
            ("ATOMG.E.ADD.BF16x4.RN.STRONG.GPU PT, R8, desc[UR4][R4.64], R8", regs(b'R', 8..10), regs(b'R', 8..10)),
            // STAS.64 data, an `[Rn]` address with .E (64-bit), CS2R's pair.
            ("STAS.64 [R2.64], R4", vec![], regs(b'R', 4..6)),
            ("ATOMG.E.CAS.64.STRONG.GPU PT, R10, [R26], R8, R10", regs(b'R', 10..12), regs(b'R', 26..28)),
            ("CS2R R4, SRZ", regs(b'R', 4..6), vec![]),
            ("MATCH.ANY.U64 R11, R6", vec![], regs(b'R', 6..8)),
            ("HMMA.1688.F32.BF16 R12, R18, R0, R12", regs(b'R', 12..16), regs(b'R', 12..16)),
        ] {
            let (d, u) = du(t);
            has(&d, &defs, t);
            has(&u, &uses, t);
        }
        // A TMEM / descriptor address first is read, never written.
        for t in ["UTCCP.T.S.4x32dp128bit tmem[UR23+0x104], gdesc[UR4]", "STTM.x16 tmem[UR6], R36"] {
            let (d, u) = du(t);
            assert!(d.is_empty(), "{t}: {d:?}");
            assert!(u.contains(&(b'U', 23)) || u.contains(&(b'U', 6)), "{t}: {u:?}");
        }
    }
    const WIDTH_FORMS: &[&str] = &[
        "LDCU.64 UR4, c[0x0][0x390]",
        "LDCU.128 UR4, c[0x0][0x390]",
        "UMOV UR6, UR5",
        "DSETP.MAX.AND P0, P2, R4, UR4, PT",
        "DADD R4, R2, UR6",
        "DFMA R4, R2, UR6, R8",
        "DMUL R4, R2, c[0x0][0x380]",
        "IMAD.WIDE.U32 R2, R0, 0x4, UR6",
        "IMAD.WIDE.U32 R26, P1, R14, R9, R10",
        "IMAD.WIDE.U32.X R10, R15, R9, R28, P1",
        "IMAD.WIDE R8, R4, R3, RZ",
        "UIMAD.WIDE.U32 UR4, UR6, UR7, UR8",
        "UIMAD.WIDE UR4, UR6, 0x4, UR8",
        "ATOMG.E.CAS.64.STRONG.GPU PT, R10, [R26], R8, R10",
        "ATOMG.E.CAS.128.STRONG.GPU PT, R8, desc[UR8][R26.64], R12, R16",
        "ATOMG.E.EXCH.128.STRONG.GPU PT, R8, desc[UR8][R26.64], R12",
        "ATOMS.CAST.SPIN.64 P1, [R13], R8, R10",
        "ATOMS.CAS.64 R14, [R31+0x8], R12, R14",
        "ATOMG.E.ADD.F64.RN.STRONG.GPU PT, R28, desc[UR8][R26.64+0x8], R24",
        "REDG.E.ADD.F64.RN.STRONG.GPU desc[UR8][R26.64+0x10], R28",
        "ATOMG.E.ADD.64.STRONG.GPU PT, R8, desc[UR8][R8.64], R14",
        "ATOM.E.ADD.64.STRONG.GPU PT, R8, [R8.64], R14",
        "ATOM.E.CAS.64.STRONG.GPU PT, R8, [R8.64], R14, R16",
        "DSETP.MIN.AND P2, P3, R8, R10, PT",
        "DSETP.GT.AND P2, PT, -|R18|, R8, PT",
        "F2F.F16.F64 R4, R4",
        "F2F.F64.F16 R10, R4",
        "F2F.BF16.F64 R4, R4",
        "F2F.F32.F64.RZ R4, R6",
        "F2I.U64.TRUNC R14, R14",
        "F2I.S64.F64.TRUNC R10, R30",
        "F2I.U64.F64.TRUNC R10, R30",
        "I2F.F64.S64 R30, R12",
        "I2F.F64.U64.RZ R8, R4",
        "I2F.S64 R34, R12",
        "I2F.U64 R34, R12",
        "I2F.F16.S64 R34, R12",
        "DMMA.884 R4, R8, R10, R4",
        "DMMA.16816 R4, R8, R12, R16",
        "HMMA.16816.F16 R4, R8, R12, R4",
        "HMMA.1688.F32 R4, R8, R12, R4",
        "HMMA.16816.F32.BF16 R4, R8, R12, R4",
        "IMMA.16832.S8.S8 R4, R8, R12, R4",
        "REDG.E.ADD.F32x4.FTZ.RN.STRONG.GPU desc[UR8][R2.64], R4",
        "ATOMG.E.ADD.F32x4.FTZ.RN.STRONG.GPU PT, R8, desc[UR8][R2.64], R4",
        "REDG.E.ADD.BF16x4.RN.STRONG.GPU desc[UR4][R4.64+0x4000], R10",
        "ATOMG.E.ADD.BF16x4.RN.STRONG.GPU PT, R8, desc[UR4][R4.64], R8",
        "REDG.E.ADD.F16x8.RN.STRONG.GPU desc[UR8][R2.64], R4",
        "LDS.U.128 R4, [R0]",
        "STS.128 [R0], R4",
        "LDG.E.128 R4, desc[UR4][R2.64]",
        "STG.E.128 desc[UR4][R2.64], R4",
        "LDG.E.ENL2.256 R8, R4, desc[UR4][R2.64]",
        "STG.E.ENL2.256 desc[UR4][R2.64], R8, R4",
        "LDSM.16.M88.4 R4, [R0]",
        "LDSM.16.MT88.2 R4, [R0]",
        "R2UR.64 UR4, R2",
        "R2UR UR4, R2",
        "MOV.64 R4, R2",
        "IADD.64 R4, R2, R6",
        "IADD.64 R4, R2, 0x1",
        "IMNMX.64 R4, R2, R6, PT",
        "ISETP.GE.U32.64.AND P0, PT, R2, R4, PT",
        "UIADD3.64 UR4, UPT, UPT, UR4, 0x8, URZ",
        "UMOV.64 UR4, UR6",
        "SHFL.IDX PT, R4, R5, R6, 0x1f",
        "POPC R30, R30",
        "FRND.F64.TRUNC R4, R2",
        "FRND.F64 R4, R2",
        "FRND R4, R2",
        "DMNMX R4, R2, R6, !PT",
        "MUFU.RSQ64H R15, R11",
        "DADD R4, R2.reuse, R6",
        "F2F.F64.F64 R4, R2",
        "I2I.S64.S32 R4, R2",
        "F2FP.F16.F32.PACK_AB R4, R2, R3",
        "LDGSTS.E.BYPASS.128 [R2], desc[UR4][R4.64]",
        "STAS.64 [R2.64], R4",
        "LDTM.x2 R4, tmem[UR4]",
        "HMMA.1688.F32.TF32 R36, R44, R58, R36",
        "HMMA.1688.F32.BF16 R4, R68, R72, R4",
        "STSM.16.M88.4 [R11], R4",
        "STTM.x16 tmem[UR6], R36",
        "FMUL2 R20, R20.F32x2.HI_LO, R0.F32",
        "P2R R21, PR, RZ, 0x20",
        "LDG.E R2, [R30]",
        "CALL.ABS.NOINC R2",
        "IMAD.HI.U32 R22, P1, R25, R29, R22",
        "SYNCS.EXCH.64 URZ, [UR4+0x90], UR6",
    ];

    /// The gb300 rules' measured windows (E-HW-SB-QUEUE, MISS 170): exactly the measured pairs,
    /// gaps, widths and guards.
    #[test]
    fn hw_measured_windows() {
        let m = |e, l, gap, g| hw_cover_measured(e, l, gap, g);
        // probes/smem_queue_w1: the pairs at gaps 2..8, the later op unguarded.
        for (e, l) in [
            ("LDSM.16.M88.4", "LDS"),
            ("LDSM.16.M88.4", "SHFL.BFLY"),
            ("LDS", "LDSM.16.M88.4"),
            ("LDS", "LDSM.16.MT88.4"),
        ] {
            assert!(m(e, l, (2, 8), false), "{e} by {l}");
            assert!(!m(e, l, (1, 8), false), "{e} by {l} gap 1");
            assert!(!m(e, l, (2, 9), false), "{e} by {l} gap 9");
            assert!(!m(e, l, (2, 8), true), "{e} by {l} guarded");
        }
        for (e, l) in [
            ("LDSM.16.M88.4", "LDS.128"),
            ("LDSM.16.M88.4", "LDS.U16"),
            ("LDSM.16.M88.2", "LDS"),
            ("LDSM.16.MT88.4", "LDS"),
            ("LDSM.16.M88.4", "SHFL.IDX"),
            ("LDS.128", "LDSM.16.M88.4"),
            ("LDS", "LDSM.16.M88.2"),
            ("LDSM.16.M88.4", "ATOMS.ADD"),
        ] {
            assert!(!m(e, l, (2, 8), false), "{e} by {l}");
        }
        // The strict queue (LDSM by LDSM, LDS by LDS) and an LDS by a later SHFL: any gap.
        assert!(m("LDSM.16.M88.4", "LDSM.16.MT88.4", (1, 64), true));
        assert!(m("LDS", "SHFL.DOWN", (1, 64), false));
        // MISS 170: not measured.
        assert!(!m("FLO.U32", "POPC", (2, 2), false));
        for (e, l) in [("DADD", "DMUL"), ("DFMA", "DADD"), ("DADD", "DSETP.GT.AND")] {
            assert!(!m(e, l, (2, 2), false), "{e} by {l}");
        }
        assert!(!m("F2F.F64.F32", "F2F.F32.F64", (2, 2), false));
        assert!(!m("I2F.F64.U64.RZ", "I2F.F64.RM", (2, 2), false));
        assert!(!m("F2I.F64.TRUNC", "F2I.S64.F64.TRUNC", (2, 2), false));
        // Measured (probes/f2i64_f64, i2f64_s16; frnd_same; redux_ops; cvt64_queue).
        assert!(m("F2I.S64.F64.TRUNC", "F2I.F64.TRUNC", (2, 2), false));
        assert!(m("I2F.F64.S16", "I2F.F64", (2, 2), false));
        assert!(m("FRND.F64", "FRND.F64.FLOOR", (2, 2), false));
        assert!(m("REDUX.AND", "REDUX.SUM.S32", (2, 2), false));
        assert!(m("F2I.TRUNC.NTZ", "F2I.S64.TRUNC", (2, 2), false));
        assert!(!hw_queued_war_measured("F2F.F64.F32", "F2F.F64.F32"));
        assert!(!hw_queued_war_measured("MUFU.EX2", "MUFU.RCP"));
        assert!(!hw_ugetnextworkid_measured());
        // R10 / R11 (MISS 170, the user's decision 2026-11-12; probes/sb_mio_order and
        // sb_mio_gap): the measured pairs 4 to 1024 stall cycles after the earlier op on every
        // path, unguarded; through both scoreboards or the write one only; nothing else.
        let c = hw_mio_cover_measured;
        let both = [
            ("STS", "LDS"),
            ("STS", "LDS.128"),
            ("STS", "LDG.E"),
            ("STS", "SHFL.BFLY"),
            ("STS.64", "LDS.64"),
            ("STS.128", "LDS.128"),
            ("STS.128", "LDS"),
            ("STG.E", "LDS"),
        ];
        let write = [
            ("STS", "SHFL.IDX"),
            ("STS.64", "SHFL.BFLY"),
            ("STS.128", "SHFL.BFLY"),
            ("STS.128", "LDSM.16.M88.4"),
            ("STS.128", "SHFL.IDX"),
            ("STS.U8", "LDS"),
            ("STS.U16", "LDS"),
            ("STG.E", "SHFL.BFLY"),
            ("STG.E", "LDG.E"),
            ("STG.E", "LDS.128"),
            ("STSM.16.M88.4", "LDS.128"),
            ("STSM.16.M88.4", "LDS"),
        ];
        for (e, l) in both.into_iter().chain(write) {
            let k = Cover::Write;
            assert!(c(e, l, k, (4, 1024), false) && c(e, l, k, (11, 11), false), "{e} by {l}");
            assert!(!c(e, l, k, (3, 4), false) && !c(e, l, k, (4, 1025), false), "{e} by {l}");
            assert!(!c(e, l, k, (4, 4), true), "{e} by {l}");
            let r = c(e, l, Cover::Read, (4, 1024), false);
            assert_eq!(r, both.contains(&(e, l)), "{e} by {l}: read scoreboard");
        }
        for (e, l) in [
            ("STS", "LDS.64"),
            ("STS.128", "LDSM.16.MT88.4"),
            ("STG.E.128", "LDS"),
            ("STG.E.64", "LDS.64"),
            ("STSM.16.M88.2", "LDS"),
            ("STS.U8", "LDS.128"),
            ("LDS", "SHFL.BFLY"),
            ("UTMASTG.2D", "UTMASTG.2D"),
            ("UTCCP.T.S.4x32dp128bit", "UTCQMMA"),
        ] {
            assert!(!c(e, l, Cover::Write, (4, 4), false), "{e} by {l}");
        }
        // The gap's upper end is the maximum over the paths, capped above the windows' end (a
        // loop's unbounded gap reads as past every window).
        let (cap, end) = (CMAX_CAP, HW_QUEUE_GAP_MAX);
        assert!(cap > end, "Pend::cmax capped at {cap}, inside the windows (to {end})");
        // R12: the measured pairs 4 to 1024 stall cycles after the reader, address and data
        // registers or the data ones only; an ATOMS.EXCH's data by an LDS 4 cycles after it.
        let w = hw_load_war_measured;
        let addr = [
            ("STS", "LDS"),
            ("STS", "LDG.E"),
            ("STS", "SHFL.BFLY"),
            ("STS.64", "LDS.64"),
            ("STS.64", "SHFL.BFLY"),
            ("STS.128", "LDS.128"),
            ("STS.128", "LDS"),
            ("STS.128", "SHFL.BFLY"),
            ("STS.U8", "LDS"),
            ("STS.U16", "LDS"),
            ("STG.E", "SHFL.BFLY"),
        ];
        let data = [
            ("STS", "LDSM.16.M88"),
            ("STS", "LDS.128"),
            ("STS", "SHFL.IDX"),
            ("STS.128", "LDSM.16.M88.4"),
            ("STS.128", "SHFL.IDX"),
            ("STG.E", "LDS"),
            ("STG.E", "LDG.E"),
            ("STG.E", "LDS.128"),
            ("STSM.16.M88.4", "LDS.128"),
            ("STSM.16.M88.4", "LDSM.16.M88.4"),
            ("STSM.16.M88.4", "LDS"),
        ];
        for (r, wr) in addr.into_iter().chain(data) {
            assert!(w(r, wr, true, (4, 1024)) && w(r, wr, true, (22, 22)), "{r} by {wr}");
            assert!(!w(r, wr, true, (4, 1025)) && !w(r, wr, true, (3, 4)), "{r} by {wr}");
            assert_eq!(w(r, wr, false, (8, 8)), addr.contains(&(r, wr)), "{r} by {wr}: address");
        }
        assert!(w("ATOMS.EXCH", "LDS", true, (4, 4)) && !w("ATOMS.EXCH", "LDS", false, (4, 4)));
        assert!(!w("ATOMS.EXCH", "LDS", true, (4, 5)));
        for (r, wr) in [
            ("STS", "LDS.64"),
            ("STG.E.128", "LDS"),
            ("STG.E.128", "SHFL.BFLY"),
            ("STS.128", "LDSM.16.MT88.4"),
            ("STSM.16.M88.4", "SHFL.BFLY"),
            ("LDS", "SHFL.BFLY"),
            ("ATOMG.E.ADD", "LDG.E"),
        ] {
            assert!(!w(r, wr, true, (4, 4)), "{r} by {wr}");
        }
        // The narrow I2F class (no cover, `conv_cover`): not the f64 form. The latency
        // contract (`CONV_LAT_CYC`): the measured form only (MISS 186).
        assert!(narrow_i2f("I2F.S8") && narrow_i2f("I2F.U16.RZ"));
        assert!(lat_measured("I2F.S8"));
        for o in ["I2F.S16", "I2F.U8", "I2F.F16.S16", "I2F.BF16.S16", "I2F.F16.U8", "I2F.F64.S16"] {
            assert!(!lat_measured(o), "{o}");
        }
        assert!(!narrow_i2f("I2F.F64.S16") && !narrow_i2f("I2F.S32") && !narrow_i2f("F2I.S16"));
    }

    /// Findings of `check` (sm_103a) on a straight-line or branching program of
    /// (text, wbar, rbar, wait, stall) lines 0x10 apart; the label `.L_J` is line `j`.
    fn check_prog(prog: &[(&str, u8, u8, u8, u8)], j: usize) -> Vec<StaticFinding> {
        let mut lines = Vec::new();
        let labels: HashMap<String, u32> = [(".L_J".to_string(), (j * 16) as u32)].into();
        let mut a2i = HashMap::new();
        for (i, &(t, w, r, wait, stall)) in prog.iter().enumerate() {
            let (guard, body) = match t.strip_prefix('@') {
                Some(x) => {
                    let (g, b) = x.split_once(' ').unwrap();
                    (Some(g.to_string()), b)
                }
                None => (None, t),
            };
            let (opcode, ops) = body.split_once(' ').unwrap_or((body, ""));
            let c = Ctrl {
                stall,
                yld: false,
                wbar: (w < 7).then_some(w),
                rbar: (r < 7).then_some(r),
                wait,
                reuse: 0,
            };
            lines.push(SassLine {
                addr: (i * 16) as u32,
                guard,
                opcode: opcode.into(),
                operands: crate::split_ops(ops),
                text: body.into(),
                hi: Some(c.encode(0)),
                lo: Some(0),
            });
            a2i.insert((i * 16) as u32, i);
        }
        let code: Vec<_> = lines
            .iter()
            .map(|l| {
                crate::lower_line(l, &labels, &a2i, &crate::CubinMeta::default(), &[]).unwrap()
            })
            .collect();
        check(&lines, &code, 103).unwrap()
    }

    /// `attach` onto an entry already there: the minimum gap, the maximum gap, the guard if both
    /// held, the covers both have, the attaching op's DEPBAR count (0), both entries' registers;
    /// onto none: the earlier op's gaps, guard and covers.
    #[test]
    fn attach_merges_an_existing_entry() {
        let r = |i: u16| (b'R', i);
        let k = (4, false, 7);
        let p = Pend {
            regs: [r(2)].into(),
            newer: 5,
            cov: [(2, 5), (0, 9)].into(),
            gok: false,
            cyc: 5,
            cmax: 30,
        };
        let mut st: State = [(
            k,
            Pend {
                regs: [r(1)].into(),
                newer: 3,
                cov: [(1, 3), (2, 5)].into(),
                gok: true,
                cyc: 10,
                cmax: 20,
            },
        )]
        .into();
        attach(&mut st, k, p.clone());
        let e = &st[&k];
        assert_eq!((e.cyc, e.cmax, e.gok, e.newer), (5, 30, false, 0));
        assert_eq!(e.cov, [(2, 5)].into());
        assert_eq!(e.regs, [r(1), r(2)].into());
        let mut st = State::new();
        attach(&mut st, k, p.clone());
        let e = &st[&k];
        assert_eq!((e.cyc, e.cmax, e.gok, e.newer), (5, 30, false, 0));
        assert_eq!(e.cov, p.cov);
        let mut st: State = [(k, Pend::new())].into();
        attach(&mut st, k, Pend { gok: true, ..p });
        assert!(st[&k].gok && st[&k].cov.is_empty());
    }

    /// MISS 264 (a): an untracked STS read attached to a later STS's read scoreboard SB4
    /// completes with that STS, the newest op on SB4: `DEPBAR.LE SB4, 0x1` (satisfied at once
    /// with only that STS outstanding) does not complete it.
    #[test]
    fn attach_newer_is_the_attaching_ops() {
        const N: u8 = 7;
        let p = |dep: &'static str| {
            vec![
                ("STS [R2], R3", N, N, 0, 4),
                ("STS [R8], R9", N, 4, 0, 4),
                (dep, N, N, 0, 4),
                ("MOV R3, R7", N, N, 0, 4),
                ("EXIT", N, N, 0, 4),
            ]
        };
        let nop = check_prog(&p("NOP"), 0);
        let le1 = check_prog(&p("DEPBAR.LE SB4, 0x1"), 0);
        let le0 = check_prog(&p("DEPBAR.LE SB4, 0x0"), 0);
        assert!(!nop.is_empty(), "control: the WAR with no DEPBAR");
        assert!(le0.is_empty(), "control: DEPBAR.LE SB4, 0x0 completes it: {le0:?}");
        assert!(!le1.is_empty(), "DEPBAR.LE SB4, 0x1 with only the attaching STS on SB4");
        // One more SB4 op after the attaching STS: DEPBAR.LE SB4, 0x1 completes it.
        let mut q = p("DEPBAR.LE SB4, 0x1");
        q.insert(2, ("STS [R10], R11", N, 4, 0, 4));
        let le1b = check_prog(&q, 0);
        assert!(le1b.is_empty(), "{le1b:?}");
    }

    /// MISS 264 (b): path X attaches the untracked STS to SB4 three SB4 ops before the DEPBAR,
    /// path Y at the last of them; at the join the entry's count is path Y's (the minimum), so
    /// `DEPBAR.LE SB4, 0x2` keeps path Y's hazard.
    #[test]
    fn attach_newer_merge_takes_the_minimum() {
        const N: u8 = 7;
        let p = |br: &'static str| {
            vec![
                ("STS [R2], R3", N, N, 0, 4),
                (br, N, N, 0, 4),
                ("STS [R8], R9", N, 4, 0, 4),
                ("STS [R10], R11", N, 4, 0, 4),
                ("STS [R12], R13", N, 4, 0, 4),
                ("STS [R14], R15", N, 4, 0, 4),
                ("DEPBAR.LE SB4, 0x2", N, N, 0, 4),
                ("MOV R3, R7", N, N, 0, 4),
                ("EXIT", N, N, 0, 4),
            ]
        };
        let y = check_prog(&p("BRA `(.L_J)"), 5);
        let x = check_prog(&p("NOP"), 5);
        let xy = check_prog(&p("@P0 BRA `(.L_J)"), 5);
        assert!(!y.is_empty(), "control: path Y alone keeps the hazard");
        assert!(x.is_empty(), "control: path X alone clears it: {x:?}");
        assert!(!xy.is_empty(), "the join of X and Y keeps path Y's hazard");
    }

    /// The gap window's upper end is the maximum over the paths (`Pend::cmax`), its lower end
    /// the minimum (`Pend::cyc`).
    #[test]
    fn join_gap_bounds() {
        let pend = |cyc, cmax| Pend {
            cyc,
            cmax,
            ..Pend::new()
        };
        let mut a: State = [((NO_SB, true, 0), pend(3, 3))].into_iter().collect();
        let b: State = [((NO_SB, true, 0), pend(10, 10))].into_iter().collect();
        assert!(join(&mut a, &b));
        assert_eq!((a[&(NO_SB, true, 0)].cyc, a[&(NO_SB, true, 0)].cmax), (3, 10));
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
        // MISS 181: 64-bit operands without a `.64` suffix.
        let rr = |v: &[u16]| v.iter().map(|&i| r(b'R', i)).collect::<Vec<_>>();
        assert_eq!(def_use(&line("DADD R4, R2, R6")), (rr(&[4, 5]), rr(&[2, 3, 6, 7])));
        assert_eq!(def_use(&line("DFMA R4, R2, R6, R8")).1, rr(&[2, 3, 6, 7, 8, 9]));
        assert_eq!(def_use(&line("DSETP.GT.AND P0, PT, R4, R6, PT")).1, rr(&[4, 5, 6, 7]));
        assert_eq!(def_use(&line("F2F.F64.F32 R4, R2")), (rr(&[4, 5]), rr(&[2])));
        assert_eq!(def_use(&line("F2F.F32.F64 R4, R2")), (rr(&[4]), rr(&[2, 3])));
        assert_eq!(def_use(&line("I2F.F64.S16 R2, R0")), (rr(&[2, 3]), rr(&[0])));
        assert_eq!(def_use(&line("I2F.S64 R12, R6")), (rr(&[12]), rr(&[6, 7])));
        assert_eq!(def_use(&line("F2I.S64.F64.TRUNC R2, R4")), (rr(&[2, 3]), rr(&[4, 5])));
        assert_eq!(def_use(&line("F2I.F64.TRUNC R5, R4")), (rr(&[5]), rr(&[4, 5])));
        assert_eq!(def_use(&line("FRND.F64 R4, R2")), (rr(&[4, 5]), rr(&[2, 3])));
        assert_eq!(def_use(&line("IMAD.WIDE.U32 R2, R3, 0x50, R18")), (rr(&[2, 3]), rr(&[3, 18, 19])));
        assert_eq!(def_use(&line("MUFU.RCP64H R5, R3")), (rr(&[5]), rr(&[3])));
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
        // Modified register operands are read like plain ones (probes/f2i_neg,
        // probes/lea_hi_not).
        assert_eq!(
            def_use(&line("F2I.U32.TRUNC.NTZ R38, -R38")),
            (vec![r(b'R', 38)], vec![r(b'R', 38)])
        );
        assert_eq!(
            def_use(&line("LEA.HI.X R54, R52, ~R55, RZ, 0x7, P0")),
            (
                vec![r(b'R', 54)],
                vec![r(b'R', 52), r(b'R', 55), r(b'P', 0)]
            )
        );
    }
}
