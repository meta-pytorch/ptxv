/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! PTX front-end: text -> `ptxv_ir::Kernel`.
//!
//! The parser is generic over PTX statement syntax (guard, dotted opcode, operand list), so
//! it accepts the whole ISA grammatically. Lowering is per opcode and fails loudly on
//! anything it does not model yet; we never skip an instruction.

use std::collections::HashMap;
use std::collections::HashSet;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_ir::*;

mod l2;

/// ptxas puts user shared memory after a 1 KiB system-reserved prefix on Blackwell; we
/// mirror that so PTX and SASS reports show the same smem offsets.
pub const SMEM_RESERVED: u32 = 0x400;

#[derive(Debug, Clone)]
struct Stmt {
    line: u32,
    labels: Vec<String>,
    text: String,
}

/// (opcode, operands) of a statement, its guard dropped.
fn stmt_parts(text: &str) -> (&str, Vec<&str>) {
    let t = text.trim().trim_end_matches(';').trim();
    let t = match t.strip_prefix('@') {
        Some(r) => r.split_once(char::is_whitespace).map_or("", |x| x.1).trim(),
        None => t,
    };
    let (op, rest) = t.split_once(char::is_whitespace).unwrap_or((t, ""));
    let ops = rest
        .split(',')
        .map(|x| x.trim().trim_matches(|c| c == '{' || c == '}').trim())
        .filter(|x| !x.is_empty())
        .collect();
    (op, ops)
}

/// Destinations of `mul.f16` / `mul.f16x2` without `.rn` that ptxas cannot contract into an
/// FMA: every use is a multiplicand of `mul` / `fma`, a store or a conversion (Triton's
/// int4 / fp8 KV dequant: `mul.f16 %rs230, %rs6, %rs229` feeding `fma.rn.f16` as a). Any
/// other use (an add / sub, an fma addend, a move ptxas could propagate) leaves the mul
/// refused.
/// `mov.b64 {%r1, %r2}, %rd3` (an unpack: the vector is the destination), not a pack.
fn is_unpack(st: &Stmt, op: &str) -> bool {
    st.text
        .split_once(op)
        .is_some_and(|(_, rest)| rest.trim_start().starts_with('{'))
}

/// Whether every use of `r`, through at most `depth` packs / unpacks (`mov.b32 %r, {%h1,
/// %h2}`, `mov.b64 {%r1, %r2}, %rd`), is one ptxas cannot contract a mul into: a store, a
/// conversion, an `mma` operand, a multiplicand of `mul` / `fma`, or a `prmt` / `shfl.sync`
/// source or `selp` data operand whose result is such, a `stmatrix` / `tcgen05.st` source (Triton's bf16 dequant: plain
/// `mul.bf16`s packed into `.b32`, then `.b64`, unpacked into an `mma.sync` B operand;
/// probes/bf16_sink; the NaN-override `selp`, `stmatrix` and `tcgen05.st`: probes/bf16_selp).
fn sink_only(insts: &[&Stmt], r: &str, depth: u32) -> bool {
    if depth == 0 {
        return false;
    }
    insts.iter().all(|st| {
        let (op, ops) = stmt_parts(&st.text);
        let base = op.split('.').next().unwrap_or("");
        let unpack = base == "mov" && is_unpack(st, op);
        ops.iter().enumerate().all(|(i, o)| {
            let def =
                (i == 0 || unpack && i + 1 < ops.len()) && !matches!(base, "st" | "red" | "atom");
            if def || *o != r {
                return true;
            }
            match base {
                "st" | "stmatrix" | "cvt" | "mma" | "mul" => true,
                "tcgen05" => op.starts_with("tcgen05.st."),
                "fma" => i == 1 || i == 2,
                // A select between the product and another value (Triton's NaN override).
                "selp" if i == 1 || i == 2 => sink_only(insts, ops[0], depth - 1),
                "mov" if unpack => ops[..i]
                    .iter()
                    .all(|h| *h == "_" || sink_only(insts, h, depth - 1)),
                "mov" if st.text.contains('{') => sink_only(insts, ops[0], depth - 1),
                // A byte permute of a packed product (CUTLASS / Triton fragment shuffles).
                "prmt" if i == 1 || i == 2 => sink_only(insts, ops[0], depth - 1),
                // A shuffled value (the source operand of shfl.sync).
                "shfl" if i == 1 && !ops[0].contains('|') => sink_only(insts, ops[0], depth - 1),
                _ => false,
            }
        })
    })
}

fn uncontractible_f16_muls(insts: &[&Stmt]) -> HashSet<String> {
    let mut cand: HashSet<String> = HashSet::new();
    let mut x2: HashSet<String> = HashSet::new();
    for st in insts {
        let (op, ops) = stmt_parts(&st.text);
        if op.starts_with("mul.")
            && [".f16", ".f16x2", ".bf16", ".bf16x2", ".f32x2"]
                .iter()
                .any(|t| op.ends_with(t))
            && !op.split('.').any(|m| m == "rn")
        {
            if let Some(d) = ops.first() {
                cand.insert(d.to_string());
                if op.ends_with(".f32x2") {
                    x2.insert(d.to_string());
                }
            }
        }
    }
    // The statements reading `r` (not defining it), with the operand index.
    let readers = |r: &str| -> Vec<(&Stmt, usize)> {
        let mut out = Vec::new();
        for st in insts {
            let (op, ops) = stmt_parts(&st.text);
            let base = op.split('.').next().unwrap_or("");
            let unpack = base == "mov" && is_unpack(st, op);
            for (i, o) in ops.iter().enumerate() {
                let def = (i == 0 || unpack && i + 1 < ops.len())
                    && !matches!(base, "st" | "red" | "atom");
                if !def && *o == r {
                    out.push((*st, i));
                }
            }
        }
        out
    };
    // A pack `mov.b64 %p, {lo, hi}`'s destination.
    let pack_dst = |st: &Stmt| -> Option<String> {
        let (op, ops) = stmt_parts(&st.text);
        (op.starts_with("mov.") && !is_unpack(st, op) && st.text.contains('{'))
            .then(|| ops[0].to_string())
    };
    // f32x2: ptxas keeps an FMUL2 whose halves are unpacked apart from the scalar ops reading
    // them (FMUL2 + FADD; FFMA with the half as its addend), and an FFMA2 reading the product
    // as its addend apart from it (probes/f32x2_unpack). A half repacked into a vector is
    // followed once more: in the other lane (Triton's `mov.b64 {_, hi}, p; mov.b64 v, {hi,
    // x}; add.f32x2`: FMUL2 + FADD2) the vector may be read by anything but a repack; in its
    // own lane only by multiplicands, fma operands, stores, conversions or unpacks whose
    // halves are not repacked.
    let pack_ok = |p: &str, crossed: bool| {
        readers(p).iter().all(|&(st, i)| {
            let (op, ops) = stmt_parts(&st.text);
            let no_repack = |hs: &[&str]| {
                hs.iter()
                    .all(|h| readers(h).iter().all(|&(s2, _)| pack_dst(s2).is_none()))
            };
            match op.split('.').next().unwrap_or("") {
                "mul" | "fma" | "st" | "cvt" => true,
                "mov" if is_unpack(st, op) && i + 1 == ops.len() => no_repack(&ops[..i]),
                "add" | "sub" if crossed => true,
                _ => false,
            }
        })
    };
    // Whether `o` may hold a lane of product `prod`: a copy, a select, or an unpack of `prod`
    // or of a register some plain `mov` defines.
    let from_prod = |o: &str, prod: &str| {
        let copied = |s: &str| {
            insts.iter().any(|st| {
                let (op, ops) = stmt_parts(&st.text);
                op.starts_with("mov.") && !st.text.contains('{') && ops.first() == Some(&s)
            })
        };
        insts.iter().any(|st| {
            let (op, ops) = stmt_parts(&st.text);
            let base = op.split('.').next().unwrap_or("");
            if base == "mov" && is_unpack(st, op) {
                let src = ops[ops.len() - 1];
                ops[..ops.len() - 1].contains(&o) && (src == prod || copied(src))
            } else {
                matches!(base, "mov" | "selp") && ops.first() == Some(&o)
            }
        })
    };
    // Half `h` of product `prod`, from lane `lane` of the unpack. Repacked into its own lane
    // next to a value not from `prod`, it is no longer the product (ptxas keeps FMUL2 and the
    // FADD2 apart, as for the other lane; the identity repack `{p.lo, p.hi}` contracts:
    // probes/f32x2_repack).
    let half_ok = |h: &str, lane: usize, prod: &str| {
        readers(h).iter().all(|&(st, i)| match pack_dst(st) {
            None => true,
            Some(p) => {
                let (_, ops) = stmt_parts(&st.text);
                let other = ops.get(if i == 1 { 2 } else { 1 }).copied().unwrap_or("");
                ops.len() == 3 && pack_ok(&p, i - 1 != lane || !from_prod(other, prod))
            }
        })
    };
    for st in insts {
        let (op, ops) = stmt_parts(&st.text);
        let base = op.split('.').next().unwrap_or("");
        let is_def = |i: usize| i == 0 && !matches!(base, "st" | "red" | "atom");
        for (i, o) in ops.iter().enumerate() {
            if is_def(i) || !cand.contains(*o) {
                continue;
            }
            let safe = match base {
                "mul" => true,
                "fma" => i == 1 || i == 2 || i == 3 && x2.contains(*o),
                "st" | "stmatrix" | "cvt" => true,
                // Integer reads of the product's bits (Triton's stochastic rounding).
                "shr" | "shl" | "and" | "or" | "xor" => true,
                "tcgen05" => op.starts_with("tcgen05.st."),
                "selp" if i == 1 || i == 2 => sink_only(insts, ops[0], 6),
                // `mov.b32 %r13, {%h10, %h10}`: a pack whose result is only stored;
                // `mov.b64 {%r64, %r65}, %rd15`: an unpack whose halves are only stored or
                // converted (Triton's f32x2 RMSNorm before cvt.rn.bf16x2.f32).
                "mov" if is_unpack(st, op) && i + 1 == ops.len() && i > 0 && x2.contains(*o) => ops
                    [..i]
                    .iter()
                    .enumerate()
                    .all(|(l, r)| *r == "_" || half_ok(r, l, o)),
                "mov" if is_unpack(st, op) && i + 1 == ops.len() && i > 0 => {
                    ops[..i].iter().all(|r| *r == "_" || sink_only(insts, r, 6))
                }
                "mov" => st.text.contains('{') && sink_only(insts, ops[0], 6),
                // An mma operand; a byte permute or shuffle whose result is only such uses.
                "mma" => true,
                "prmt" if i == 1 || i == 2 => sink_only(insts, ops[0], 6),
                "shfl" if i == 1 && !ops[0].contains('|') => sink_only(insts, ops[0], 6),
                _ => false,
            };
            if !safe {
                cand.remove(*o);
            }
        }
    }
    cand
}

/// A plain `add` / `sub` `.f32` / `.f64` that ptxas contracts with the plain `mul` producing
/// one of its operands (PTX-FMAD): the mul's two operands and the add operand it fills.
#[derive(Clone, Debug)]
struct Fmad {
    a: String,
    b: String,
    slot: usize,
    /// The product reaches the add through a `neg` (FFMA -a, b, c).
    neg: bool,
}

/// `%name` tokens of an operand string.
fn reg_tokens(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let j = (i + 1..b.len())
                .find(|&j| !(b[j].is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'$'))
                .unwrap_or(b.len());
            out.push(&s[i..j]);
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// Destinations and sources of a statement (register tokens): a leading `{..}` list or the
/// first operand is the destination unless it is an address (`[..]`: stores, reductions);
/// the guard predicate is a source.
fn stmt_defs_reads(text: &str) -> (Vec<&str>, Vec<&str>) {
    let t = text.trim().trim_end_matches(';').trim();
    let (guard, t) = match t.strip_prefix('@') {
        Some(r) => r.split_once(char::is_whitespace).unwrap_or((r, "")),
        None => ("", t),
    };
    let rest = t
        .split_once(char::is_whitespace)
        .map_or("", |x| x.1)
        .trim_start();
    let (dst, src) = if rest.starts_with('[') {
        ("", rest)
    } else if rest.starts_with('{') {
        let e = rest.find('}').map_or(rest.len(), |e| e + 1);
        (&rest[..e], &rest[e..])
    } else {
        rest.split_once(',').unwrap_or((rest, ""))
    };
    let mut reads = reg_tokens(guard);
    reads.extend(reg_tokens(src));
    (reg_tokens(dst), reads)
}

/// PTX-FMAD: ptxas 13.0's default (`--fmad=true`) contraction of a float `mul` without a
/// rounding modifier into the `add` / `sub`s without one that read it (probes/fmad,
/// probes/fmad_multi). All or nothing per mul: when every live reader of the product is such
/// an add / sub (`.sat` allowed) of the same type and `.ftz` setting, the product in one
/// operand, ptxas contracts it into each (one FFMA / DFMA per reader, the mul disappears);
/// readers through a plain copy (`mov`) or a `neg` (FFMA -a, b, c) count as the reader they
/// feed. Any other live reader (a store, a mul or fma operand, a compare, a conversion, an
/// `add.rn`, an add of another `.ftz` setting, `add p, p`, a copy or `neg` read by such, a
/// reader in another block) keeps the FMUL and every reader rounds the product. Dead readers
/// (ptxas removes them first) do not count. An add / sub of two contracted products folds the
/// one that has no other reader (both: the first operand's); the other mul then keeps its
/// FMUL for all its readers. Readers are the statements the mul's definition reaches: every
/// reader of the register when it has one definition, else those before the next
/// redefinition in the block (a definition reaching the block's end is not followed).
/// Returns the contracted adds by statement index, and statements refused because ptxas may
/// contract them in a way not modeled (a contracting reader in another block, before the mul,
/// under a guard or after the mul's operands change while no other reader keeps the FMUL; an
/// add of two products both read elsewhere too; a guarded mul; ...).
fn fmad_pairs(
    insts: &[&Stmt],
    brx_tables: &HashMap<String, Vec<String>>,
) -> (HashMap<usize, Fmad>, HashMap<usize, String>) {
    let mut pairs = HashMap::new();
    let mut refuse: HashMap<usize, String> = HashMap::new();
    let n = insts.len();
    // (base, type, ftz, sat, has rounding) of a float add / sub / mul.
    fn fop(op: &str) -> Option<(&str, &str, bool, bool, bool)> {
        let mut it = op.split('.');
        let base = it.next()?;
        if !matches!(base, "add" | "sub" | "mul") {
            return None;
        }
        let mods: Vec<&str> = it.collect();
        let ty = *mods.last()?;
        if !matches!(ty, "f32" | "f64" | "f16") {
            return None;
        }
        let rnd = mods.iter().any(|m| matches!(*m, "rn" | "rz" | "rm" | "rp"));
        Some((base, ty, mods.contains(&"ftz"), mods.contains(&"sat"), rnd))
    }
    let guard_of = |t: &str| -> String {
        let t = t.trim();
        if t.starts_with('@') {
            t.split_whitespace().next().unwrap_or("").to_string()
        } else {
            String::new()
        }
    };
    let parts: Vec<(&str, Vec<&str>)> = insts.iter().map(|s| stmt_parts(&s.text)).collect();
    let dr: Vec<(Vec<&str>, Vec<&str>)> = insts.iter().map(|s| stmt_defs_reads(&s.text)).collect();
    let guards: Vec<String> = insts.iter().map(|s| guard_of(&s.text)).collect();
    // Basic blocks: a label some branch or `.branchtargets` list names starts one (Triton's
    // `$L__tmp` debug labels do not); a branch, return, exit or call ends one.
    let mut named: HashSet<&str> = brx_tables.values().flatten().map(|s| s.as_str()).collect();
    for st in insts {
        named.extend(
            st.text
                .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.')))
                .filter(|w| !w.is_empty()),
        );
    }
    let mut block = Vec::with_capacity(n);
    let mut b = 0usize;
    for (i, st) in insts.iter().enumerate() {
        if i > 0 && st.labels.iter().any(|l| named.contains(l.as_str())) {
            b += 1;
        }
        block.push(b);
        let base = parts[i].0.split('.').next().unwrap_or("");
        if matches!(base, "bra" | "brx" | "ret" | "exit" | "call" | "trap") {
            b += 1;
        }
    }
    let mut ndefs: HashMap<&str, usize> = HashMap::new();
    let mut readers_of: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, (d, r)) in dr.iter().enumerate() {
        for x in d {
            *ndefs.entry(x).or_default() += 1;
        }
        for x in r {
            let v = readers_of.entry(x).or_default();
            if v.last() != Some(&i) {
                v.push(i);
            }
        }
    }
    // Liveness by register name (an over-approximation: a statement counted dead is dead).
    // Roots: statements with effects beyond their destinations.
    let pure = |op: &str| {
        matches!(
            op.split('.').next().unwrap_or(""),
            "add"
                | "sub"
                | "mul"
                | "mad"
                | "fma"
                | "neg"
                | "abs"
                | "min"
                | "max"
                | "mov"
                | "cvt"
                | "cvta"
                | "selp"
                | "setp"
                | "set"
                | "ex2"
                | "lg2"
                | "rcp"
                | "sqrt"
                | "rsqrt"
                | "sin"
                | "cos"
                | "tanh"
                | "div"
                | "rem"
                | "and"
                | "or"
                | "xor"
                | "not"
                | "cnot"
                | "shl"
                | "shr"
                | "shf"
                | "prmt"
                | "bfe"
                | "bfi"
                | "brev"
                | "popc"
                | "clz"
                | "lop3"
                | "copysign"
                | "testp"
                | "mul24"
                | "mad24"
                | "sad"
                | "slct"
                | "dp4a"
                | "dp2a"
        )
    };
    let mut live: Vec<bool> = (0..n)
        .map(|i| !pure(parts[i].0) || dr[i].0.is_empty())
        .collect();
    let mut live_regs: HashSet<&str> = HashSet::new();
    let mut work: Vec<usize> = (0..n).filter(|&i| live[i]).collect();
    let mut definers: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, (d, _)) in dr.iter().enumerate() {
        for x in d {
            definers.entry(x).or_default().push(i);
        }
    }
    while let Some(i) = work.pop() {
        for &r in &dr[i].1 {
            if live_regs.insert(r) {
                for &k in definers.get(r).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if !live[k] {
                        live[k] = true;
                        work.push(k);
                    }
                }
            }
        }
    }
    // The statements definition `i` of `r` reaches in its block, and whether it may reach past
    // the block's end (a definition among several that is not redefined in its block).
    let reach = |i: usize, r: &str| -> (Vec<usize>, bool) {
        if ndefs.get(r) == Some(&1) {
            return (
                readers_of
                    .get(r)
                    .map(|v| v.iter().copied().filter(|&j| j != i).collect())
                    .unwrap_or_default(),
                false,
            );
        }
        let mut out = Vec::new();
        for j in i + 1..n {
            if block[j] != block[i] {
                return (out, true);
            }
            if dr[j].1.contains(&r) {
                out.push(j);
            }
            if dr[j].0.contains(&r) && (guards[j].is_empty() || guards[j] == guards[i]) {
                return (out, false);
            }
        }
        (out, true)
    };
    // Per mul: its contracting readers (add, product slot, product negated), or why not.
    enum Use {
        Contract(Vec<(usize, usize, bool)>),
        Keep,
        Unknown(String),
    }
    let decide = |i: usize| -> Option<Use> {
        let (op, ops) = &parts[i];
        let ("mul", ty, ftz, _, false) = fop(op)? else {
            return None;
        };
        let d = *dr[i].0.first()?;
        let srcs: Vec<&str> = ops[1..].iter().flat_map(|o| reg_tokens(o)).collect();
        let mut out = Vec::new();
        let (mut keep, mut unknown, mut any_add) = (false, None::<String>, false);
        // (defining statement, register, negated, depth)
        let mut stack = vec![(i, d, false, 0u32)];
        while let Some((k, r, neg, depth)) = stack.pop() {
            let (rs, escapes) = reach(k, r);
            if escapes {
                // Live past its block (the register has other definitions): its readers there
                // are not known. A reader in the block that keeps the mul still decides it.
                unknown.get_or_insert("plain add / sub, and read or written elsewhere too".into());
                any_add |= readers_of.get(r).is_some_and(|v| {
                    v.iter().any(|&j| {
                        matches!(fop(parts[j].0), Some(("add" | "sub", ..)))
                            || parts[j].0.starts_with("neg.")
                            || parts[j].0.starts_with("mov.")
                    })
                });
            }
            for j in rs {
                if !live[j] {
                    continue;
                }
                let (jop, jops) = &parts[j];
                let jbase = jop.split('.').next().unwrap_or("");
                let same = block[j] == block[i] && j > i;
                let in_slot: Vec<usize> = (1..=2)
                    .filter(|&s| jops.get(s).is_some_and(|x| x.trim() == r))
                    .collect();
                match fop(jop) {
                    Some(("add" | "sub", aty, aftz, _, arnd)) => {
                        any_add = true;
                        if arnd || aty != ty || aftz != ftz || in_slot.len() != 1 {
                            keep = true;
                        } else if !same {
                            unknown.get_or_insert("plain add / sub in another block or before it (ptxas contracts across blocks after if-conversion or loop unrolling, not modeled)".into());
                        } else if !guards[j].is_empty() {
                            unknown.get_or_insert("plain add / sub under a guard".into());
                        } else if (i + 1..j).any(|k2| dr[k2].0.iter().any(|x| srcs.contains(x))) {
                            unknown
                                .get_or_insert("plain add / sub after its operands change".into());
                        } else {
                            out.push((j, in_slot[0], neg));
                        }
                    }
                    _ if depth < 4
                        && jops.len() == 2
                        && jops[1].trim() == r
                        && !insts[j].text.contains('{')
                        && (*jop == format!("neg.{ty}") && !ftz
                            || matches!(jbase, "mov")
                                && (jop.ends_with(ty)
                                    || jop.ends_with(match ty {
                                        "f16" => "b16",
                                        "f32" => "b32",
                                        _ => "b64",
                                    }))) =>
                    {
                        any_add |= jbase == "neg";
                        if ty == "f16" {
                            // Not followed: refuse the adds it reaches (the Unknown arm below).
                            any_add = true;
                            unknown
                                .get_or_insert("copy / neg of an f16 product (not probed)".into());
                        } else if !guards[j].is_empty() || !same {
                            unknown.get_or_insert(
                                "copy / neg in another block or under a guard".into(),
                            );
                        } else {
                            let nd = dr[j].0.first().copied().unwrap_or("");
                            stack.push((j, nd, neg ^ (jbase == "neg"), depth + 1));
                        }
                    }
                    _ => keep = true,
                }
            }
        }
        if !any_add {
            return None;
        }
        Some(if keep {
            Use::Keep
        } else if let Some(w) = unknown {
            Use::Unknown(w)
        } else if !guards[i].is_empty() {
            Use::Unknown("plain add / sub of a guarded mul".into())
        } else {
            Use::Contract(out)
        })
    };
    let mut cand: HashMap<usize, Vec<(usize, usize, bool)>> = HashMap::new();
    for i in 0..n {
        let Some(u) = decide(i) else { continue };
        let (op, _) = &parts[i];
        let d = dr[i].0.first().copied().unwrap_or("");
        match u {
            Use::Contract(v) => {
                cand.insert(i, v);
            }
            Use::Keep => {}
            Use::Unknown(w) => {
                // Refuse the readers that could contract: the adds / subs and negs reading it.
                let msg = format!("{op} {d} read by a {w}: ptxas may contract it (PTX-FMAD)");
                let mut stack = vec![(i, d, 0u32)];
                while let Some((k, r, depth)) = stack.pop() {
                    let (rs, escapes) = reach(k, r);
                    let rs = if escapes {
                        readers_of.get(r).cloned().unwrap_or_default()
                    } else {
                        rs
                    };
                    for j in rs {
                        if j == k || !live[j] {
                            continue;
                        }
                        let (jop, jops) = &parts[j];
                        let jb = jop.split('.').next().unwrap_or("");
                        if matches!(jb, "add" | "sub") && fop(jop).is_some() {
                            refuse.entry(j).or_insert_with(|| msg.clone());
                        } else if depth < 4
                            && matches!(jb, "neg" | "mov")
                            && jops.len() == 2
                            && jops[1].trim() == r
                        {
                            if jb == "neg" {
                                refuse.entry(j).or_insert_with(|| msg.clone());
                            }
                            if let Some(&nd) = dr[j].0.first() {
                                stack.push((j, nd, depth + 1));
                            }
                        }
                    }
                }
            }
        }
    }
    // Adds reading two contracted products: fold the one with no other reader (both: the
    // first operand's); the other mul keeps its FMUL. Repeated until nothing changes.
    loop {
        let mut by_add: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (&i, v) in &cand {
            for &(j, slot, _) in v {
                by_add.entry(j).or_default().push((i, slot));
            }
        }
        let mut drop = None;
        for (&j, ms) in &by_add {
            if ms.len() < 2 {
                continue;
            }
            let single = |i: usize| cand[&i].len() == 1;
            if parts[j].0.ends_with(".f16") {
                let (op, _) = &parts[j];
                refuse.insert(j, format!("{op} of two plain f16 mul products (not probed): ptxas may contract it (PTX-FMAD)"));
                for &(i, _) in ms {
                    cand.remove(&i);
                }
                drop = Some(usize::MAX);
                break;
            }
            let (m1, m2) = if ms[0].1 == 1 {
                (ms[0].0, ms[1].0)
            } else {
                (ms[1].0, ms[0].0)
            };
            drop = Some(match (single(m1), single(m2)) {
                (true, _) => m2,
                (false, true) => m1,
                (false, false) => {
                    let (op, _) = &parts[j];
                    refuse.insert(
                        j,
                        format!("{op} of two plain mul products both read elsewhere too: ptxas may contract it (PTX-FMAD)"),
                    );
                    cand.remove(&m1);
                    m2
                }
            });
            break;
        }
        match drop {
            Some(usize::MAX) => {}
            Some(i) => {
                cand.remove(&i);
            }
            None => break,
        }
    }
    for (i, v) in cand {
        let ops = &parts[i].1;
        for (j, slot, neg) in v {
            pairs.insert(
                j,
                Fmad {
                    a: ops[1].to_string(),
                    b: ops[2].to_string(),
                    slot,
                    neg,
                },
            );
        }
    }
    for j in refuse.keys() {
        pairs.remove(j);
    }
    (pairs, refuse)
}

/// PTX-FMAD for `.f32x2` (probes/fmad2): ptxas 13.0 contracts a `mul.f32x2` into every
/// `add` / `sub.f32x2` reading it when all its readers are such adds with the same rounding
/// mode (none = `.rn`) and `.ftz` setting, the product in one operand (FFMA2, FFMA2.RZ, ...;
/// the mul disappears), whatever the rounding modifier; with any other reader (a store, an
/// fma or mul operand, an unpack, an add of another mode, `add p, p`) it contracts none
/// (FMUL2 + FADD2). An add of two such muls folds its first operand's when the second has no
/// other reader. A label no branch names (`$L__tmp`) does not end the block. Returns the
/// contracted adds, statements refused (a contracting reader in another block, under a guard,
/// after the mul's operands change, before the mul, reading two such muls of which the second
/// has other readers, or next to a reader of a kind not probed), and the muls whose readers
/// are all probed kinds other than unpacks (their own value is rounded once whatever ptxas
/// does).
fn fmad2_pairs(
    insts: &[&Stmt],
    brx_tables: &HashMap<String, Vec<String>>,
) -> (
    HashMap<usize, Fmad>,
    HashMap<usize, String>,
    HashSet<String>,
) {
    let mut pairs: HashMap<usize, Fmad> = HashMap::new();
    let mut refuse: HashMap<usize, String> = HashMap::new();
    let mut decided: HashSet<String> = HashSet::new();
    // (base, rounding, ftz) of an f32x2 op.
    fn x2(op: &str) -> Option<(&str, &str, bool)> {
        let mut it = op.split('.');
        let base = it.next()?;
        let mods: Vec<&str> = it.collect();
        if mods.last() != Some(&"f32x2") {
            return None;
        }
        let rnd = mods
            .iter()
            .find(|m| matches!(**m, "rn" | "rz" | "rm" | "rp"))
            .copied()
            .unwrap_or("rn");
        Some((base, rnd, mods.contains(&"ftz")))
    }
    let guarded = |t: &str| t.trim().starts_with('@');
    // Labels no instruction or `.branchtargets` list names (Triton's `$L__tmp` debug labels)
    // do not start a block.
    let mut named: HashSet<&str> = brx_tables.values().flatten().map(|s| s.as_str()).collect();
    for st in insts {
        named.extend(
            st.text
                .split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.')))
                .filter(|w| !w.is_empty()),
        );
    }
    let target = |s: &Stmt| s.labels.iter().any(|l| named.contains(l.as_str()));
    let mut occ: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, st) in insts.iter().enumerate() {
        for r in reg_tokens(&st.text) {
            let v = occ.entry(r).or_default();
            if v.last() != Some(&i) {
                v.push(i);
            }
        }
    }
    // Contraction candidates by add: the fold, and whether the add is the mul's only reader.
    let mut cands: HashMap<usize, Vec<(Fmad, bool)>> = HashMap::new();
    for (i, st) in insts.iter().enumerate() {
        let (op, ops) = stmt_parts(&st.text);
        let Some(("mul", rnd, ftz)) = x2(op) else {
            continue;
        };
        let Some(&d) = ops.first() else { continue };
        let mut q: Vec<(usize, usize)> = Vec::new();
        let (mut other, mut unknown, mut unpacked) = (false, false, false);
        for &j in occ[d].iter().filter(|&&j| j != i) {
            let (o, jo) = stmt_parts(&insts[j].text);
            let base = o.split('.').next().unwrap_or("");
            let unpack = base == "mov" && is_unpack(insts[j], o);
            let defs = if unpack {
                &jo[..jo.len().saturating_sub(1)]
            } else if matches!(base, "st" | "red" | "atom") {
                &jo[..0]
            } else {
                &jo[..jo.len().min(1)]
            };
            if defs.iter().any(|x| reg_tokens(x).contains(&d)) {
                unknown = true;
                continue;
            }
            match x2(o) {
                Some(("add" | "sub", r2, f2)) if r2 == rnd && f2 == ftz => {
                    let slots: Vec<usize> = (1..=2)
                        .filter(|&k| jo.get(k).is_some_and(|x| x.trim() == d))
                        .collect();
                    match slots[..] {
                        [k] => q.push((j, k)),
                        [_, _] => other = true,
                        _ => unknown = true,
                    }
                }
                Some(("add" | "sub" | "fma" | "mul", _, _)) => other = true,
                _ if base == "st" => other = true,
                _ if unpack && jo.last().is_some_and(|x| x.trim() == d) => {
                    other = true;
                    // Halves that only scalar ops read (no copy, select, permute, shuffle or
                    // repack that could carry one back into a vector): ptxas keeps the FMUL2
                    // apart from them (probes/f32x2_unpack) and from the add / sub.f32x2
                    // reading the product next to the unpack (probes/f32x2_unpack_add).
                    let scalar_only = jo[..jo.len() - 1].iter().map(|h| h.trim()).all(|h| {
                        h == "_"
                            || occ.get(h).is_some_and(|v| {
                                v.iter().filter(|&&k| k != j).all(|&k| {
                                    let (ko, kops) = stmt_parts(&insts[k].text);
                                    let kb = ko.split('.').next().unwrap_or("");
                                    let kdef = !matches!(kb, "st" | "red" | "atom")
                                        && kops.first().is_some_and(|x| reg_tokens(x).contains(&h));
                                    !kdef && !matches!(kb, "mov" | "selp" | "prmt" | "shfl")
                                })
                            })
                    });
                    unpacked |= !scalar_only;
                }
                _ => unknown = true,
            }
        }
        // Halves unpacked into anything but scalar ops: whether ptxas contracts through them is
        // `uncontractible_f16_muls`.
        if !unknown && !unpacked {
            decided.insert(d.to_string());
        }
        if q.is_empty() || other && !unknown {
            continue;
        }
        let why = |w: &str| {
            format!("{op} {d} read by a same-mode add / sub {w}: ptxas may contract it (PTX-FMAD)")
        };
        // `.ftz` pairs: ptxas 13.1 / 13.2 contract `mul.ftz` + `add.ftz` only without a rounding
        // modifier (FFMA2.FTZ; with `.rn`: FMUL2.FTZ + FADD2.FTZ), unlike 13.0's non-ftz rule
        // above, and 13.0's ftz behavior is unprobed: refused.
        if ftz {
            for &(j, _) in &q {
                refuse.insert(
                    j,
                    why("with .ftz (contraction depends on the ptxas version)"),
                );
            }
            continue;
        }
        if unknown {
            for &(j, _) in &q {
                refuse.insert(j, why("and by an instruction of a kind not probed"));
            }
            continue;
        }
        let srcs: Vec<&str> = ops[1..].iter().flat_map(|o| reg_tokens(o)).collect();
        let ok = |j: usize| -> bool {
            if j <= i || guarded(&st.text) || guarded(&insts[j].text) {
                return false;
            }
            let crosses = insts[i + 1..=j].iter().any(|s| target(s))
                || insts[i + 1..j].iter().any(|s| {
                    let b = stmt_parts(&s.text).0.split('.').next().unwrap_or("");
                    matches!(b, "bra" | "brx" | "ret" | "exit" | "call")
                });
            let redef = insts[i + 1..j].iter().any(|s| {
                let (o, so) = stmt_parts(&s.text);
                let b = o.split('.').next().unwrap_or("");
                let def = if is_unpack(s, o) || s.text.contains('|') {
                    so[..so.len().saturating_sub(1)].join(",")
                } else if matches!(b, "st" | "red" | "prefetch" | "bar" | "barrier") {
                    String::new()
                } else {
                    so.first().map(|x| x.to_string()).unwrap_or_default()
                };
                reg_tokens(&def).iter().any(|r| srcs.contains(r))
            });
            !crosses && !redef
        };
        if !q.iter().all(|&(j, _)| ok(j)) {
            for &(j, _) in &q {
                refuse.insert(
                    j,
                    why("in another block, under a guard, before it, or after its operands change"),
                );
            }
            continue;
        }
        let only = q.len() == 1;
        for (j, slot) in q {
            let f = Fmad {
                a: ops[1].to_string(),
                b: ops[2].to_string(),
                slot,
                neg: false,
            };
            cands.entry(j).or_default().push((f, only));
        }
    }
    // An add / sub of two such muls: ptxas folds the first operand's, the second staying an
    // FMUL2 (probes/fmad2), when the second mul has no other reader; otherwise refused.
    for (j, mut cs) in cands {
        match cs.len() {
            1 => {
                pairs.insert(j, cs.pop().unwrap().0);
            }
            _ if cs.iter().any(|(f, only)| f.slot == 2 && *only) => {
                let f = cs.into_iter().find(|(f, _)| f.slot == 1).unwrap().0;
                pairs.insert(j, f);
            }
            _ => {
                refuse.insert(
                    j,
                    "add / sub.f32x2 of two contractible mul.f32x2, the second read elsewhere too \
                     (which one ptxas folds is not modeled; PTX-FMAD)"
                        .into(),
                );
            }
        }
    }
    for j in refuse.keys() {
        pairs.remove(j);
    }
    (pairs, refuse, decided)
}

/// PTX-FMAD for `.f32x2`: plain `mul.f32x2` destinations that no `add` / `sub.f32x2` can read,
/// whichever definition of the register reaches it (register names, so loop-carried and
/// redefined registers are covered): every statement naming the register reads it as a
/// multiplicand or fma operand (FFMA2 cannot absorb a product as its addend:
/// probes/f32x2_unpack), stores it, selects it (`selp`: FMUL2, FSEL, then FADD2 for an add of
/// the selection: probes/f32x2_sinks), copies it with `mov` (the copy is followed), or unpacks
/// it into halves (ptxas keeps FMUL2 apart from scalar readers: probes/f32x2_unpack); a
/// vector repacked from a half, or from a copy of one, must pass the same test.
/// ptxas has nothing to contract such a mul into: it is one FMUL2 (.rn).
fn x2_no_add_readers(insts: &[&Stmt]) -> HashSet<String> {
    let mut readers: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    // Scalar register -> the vectors packed from it / its plain copies.
    let mut packs: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut copies: HashMap<&str, Vec<&str>> = HashMap::new();
    for (j, st) in insts.iter().enumerate() {
        let (op, ops) = stmt_parts(&st.text);
        let base = op.split('.').next().unwrap_or("");
        let unpack = base == "mov" && is_unpack(st, op);
        if base == "mov" && !unpack && st.text.contains('{') {
            for o in &ops[1..] {
                for r in reg_tokens(o) {
                    packs.entry(r).or_default().push(ops[0]);
                }
            }
        } else if base == "mov" && !unpack && ops.len() == 2 {
            for r in reg_tokens(ops[1]) {
                copies.entry(r).or_default().push(ops[0]);
            }
        }
        for (i, o) in ops.iter().enumerate() {
            let def =
                (i == 0 || unpack && i + 1 < ops.len()) && !matches!(base, "st" | "red" | "atom");
            if !def {
                for r in reg_tokens(o) {
                    readers.entry(r).or_default().push((j, i));
                }
            }
        }
    }
    let ok = |d: &str| -> bool {
        let mut seen: HashSet<String> = HashSet::new();
        let mut stack = vec![d.to_string()];
        while let Some(r) = stack.pop() {
            if seen.len() > 64 {
                return false;
            }
            if !seen.insert(r.clone()) {
                continue;
            }
            for &(j, i) in readers.get(r.as_str()).map(|v| v.as_slice()).unwrap_or(&[]) {
                let st = insts[j];
                let (op, ops) = stmt_parts(&st.text);
                let base = op.split('.').next().unwrap_or("");
                let fine = match base {
                    "mul" | "fma" if op.ends_with(".f32x2") => true,
                    "st" => true,
                    "selp" => i == 1 || i == 2,
                    // Halves: scalar readers keep FMUL2 apart (probes/f32x2_unpack); a vector
                    // repacked from a half or a copy of it is followed like the product (the
                    // identity repack contracts: probes/f32x2_repack).
                    "mov" if is_unpack(st, op) => {
                        if i + 1 != ops.len() {
                            false
                        } else {
                            let mut hs: Vec<&str> =
                                ops[..i].iter().copied().filter(|h| *h != "_").collect();
                            let mut hseen: HashSet<&str> = HashSet::new();
                            while let Some(h) = hs.pop() {
                                if !hseen.insert(h) {
                                    continue;
                                }
                                hs.extend(copies.get(h).map(|v| v.as_slice()).unwrap_or(&[]));
                                for v in packs.get(h).map(|v| v.as_slice()).unwrap_or(&[]) {
                                    stack.push(v.to_string());
                                }
                            }
                            true
                        }
                    }
                    "mov" if !st.text.contains('{') && ops.len() == 2 => {
                        stack.push(ops[0].to_string());
                        true
                    }
                    _ => false,
                };
                if !fine {
                    return false;
                }
            }
        }
        true
    };
    insts
        .iter()
        .filter_map(|st| {
            let (op, ops) = stmt_parts(&st.text);
            (op == "mul.f32x2")
                .then(|| ops.first().map(|d| d.to_string()))
                .flatten()
        })
        .filter(|d| ok(d))
        .collect()
}

/// Destinations of float `mul`s without `.rn` (ptxas may contract them into an FMA with an
/// add that reads them).
fn loose_mul_dsts(insts: &[&Stmt]) -> HashSet<String> {
    insts
        .iter()
        .filter_map(|st| {
            let (op, ops) = stmt_parts(&st.text);
            let f = op.ends_with(".f16")
                || op.ends_with(".f16x2")
                || op.ends_with(".bf16")
                || op.ends_with(".bf16x2")
                || op.ends_with(".f32x2")
                || op.ends_with(".f32")
                || op.ends_with(".f64");
            (op.starts_with("mul.") && f && !op.split('.').any(|m| m == "rn"))
                .then(|| ops.first().map(|d| d.to_string()))
                .flatten()
        })
        .collect()
}

/// `call.uni (retval0), vprintf, (param0, param1)` (DeepEP's timeout prints): the call becomes
/// `__vprintf` (no effect, as the SASS front's `CALL.ABS.NOINC `(vprintf)`; ASSUME(E-CLOCK)),
/// the `st.param` of its arguments `__nop`, and the `ld.param` of its result a move of the
/// undefined register (S-INIT: the count printf returns is not modeled, a use is reported).
/// `call.uni __assertfail, (p0, .., p4)` (CUDA / Triton device asserts) likewise becomes
/// `__assertfail`, the SASS front's `CALL.ABS` to it: the grid is aborted (a trap).
fn vprintf_calls(stmts: &mut [Stmt]) {
    let names = |s: &str| -> Vec<String> {
        s.trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split(',')
            .map(|x| x.trim().to_string())
            .filter(|x| !x.is_empty())
            .collect()
    };
    let calls: Vec<usize> = (0..stmts.len())
        .filter(|&i| {
            let (op, _) = stmt_parts(&stmts[i].text);
            op.starts_with("call")
                && (stmts[i].text.contains("vprintf") || stmts[i].text.contains("__assertfail"))
        })
        .collect();
    for i in calls {
        let t = stmts[i].text.trim().to_string();
        // `call.uni (ret), vprintf, (p0, p1)`: ret and the argument names.
        let parts: Vec<&str> = t.splitn(2, char::is_whitespace).collect();
        let rest = parts.get(1).copied().unwrap_or("");
        let (ret, after) = match rest.trim_start().strip_prefix('(') {
            Some(r) => match r.split_once(')') {
                Some((a, b)) => (names(a), b),
                None => continue,
            },
            None => (vec![], rest),
        };
        let after = after.trim().trim_start_matches(',').trim();
        let (callee, rest, nargs, max_ret) = if let Some(a) = after.strip_prefix("vprintf") {
            ("__vprintf", a, 2, 1)
        } else if let Some(a) = after.strip_prefix("__assertfail") {
            ("__assertfail", a, 5, 0)
        } else {
            continue;
        };
        let args = names(rest.trim().trim_start_matches(','));
        if t.starts_with('@') || ret.len() > max_ret || args.len() != nargs {
            continue; // left as is: refused by the lowering
        }
        stmts[i].text = callee.to_string();
        for st in stmts[i.saturating_sub(16)..i].iter_mut() {
            let (op, ops) = stmt_parts(&st.text);
            if op.starts_with("st.param")
                && ops.first().is_some_and(|a| {
                    let a = a
                        .trim()
                        .trim_start_matches('[')
                        .trim_end_matches(']')
                        .trim();
                    args.iter().any(|p| p == a)
                })
            {
                st.text = "__nop".to_string();
            }
        }
        if let Some(r) = ret.first() {
            let end = (i + 4).min(stmts.len());
            for st in stmts[i + 1..end].iter_mut() {
                let (op, ops) = stmt_parts(&st.text);
                if op.starts_with("ld.param")
                    && ops.get(1).is_some_and(|a| {
                        a.trim()
                            .trim_start_matches('[')
                            .trim_end_matches(']')
                            .trim()
                            == r
                    })
                {
                    let d = ops[0].trim().to_string();
                    st.text = format!("mov.b32 {d}, %__undef");
                }
            }
        }
    }
}

/// RN(1 / c) for a normal f32 `c` (exact integer arithmetic; `c`'s reciprocal is normal for
/// 2^-126 <= |c| <= 2^126).
fn recip_rn_f32(c: u32) -> u32 {
    let s = c & 0x8000_0000;
    let e = ((c >> 23) & 0xff) as i32;
    let m = (c & 0x7f_ffff | 0x80_0000) as u128; // c = m * 2^(e - 150)
    // 1 / c = 2^(150 - e) / m; q = floor(2^64 / m) has 41 or 42 bits.
    let num = 1u128 << 64;
    let (mut q, r) = (num / m, num % m);
    let bits = 128 - q.leading_zeros() as i32;
    let sh = bits - 24;
    let low = q & ((1u128 << sh) - 1);
    q >>= sh;
    let half = 1u128 << (sh - 1);
    if low > half || low == half && (r != 0 || q & 1 == 1) {
        q += 1;
    }
    // value = q * 2^(sh - 64 + 150 - e); q in [2^23, 2^24] (2^24 after a carry).
    let mut exp = sh - 64 + 150 - e + 23;
    if q == 1 << 24 {
        q >>= 1;
        exp += 1;
    }
    s | ((exp + 127) as u32) << 23 | (q as u32 & 0x7f_ffff)
}

/// ptxas folds `div.full` / `div.approx [.ftz] .f32 d, a, C` with a constant divisor (an
/// immediate, or a register whose only definition is a `mov` of one that dominates the
/// division) into `mul [.ftz] .f32 d, a, RN(1/C)` for 2^-126 <= |C| <= 2^126, C != +-1 (then
/// a plain mul, contracted into a following add as any other: PTX-FMAD); C = 1 is `a` (`.ftz`:
/// a + -0), C = -1 is -a + -0 (probes/div_const). The statements are rewritten so; other
/// constants (the scaled range) stay divisions. ASSUME(PTX-FMAD)
fn div_by_const(stmts: &mut [Stmt]) {
    // Registers with exactly one definition, an unguarded `mov` of an f32 literal.
    let mut defs: HashMap<String, (usize, Option<u32>)> = HashMap::new();
    let mut ndefs: HashMap<String, usize> = HashMap::new();
    for (i, st) in stmts.iter().enumerate() {
        let t = st.text.trim();
        if t.is_empty() || t.starts_with('.') {
            continue;
        }
        let (op, ops) = stmt_parts(t);
        let base = op.split('.').next().unwrap_or("");
        if matches!(base, "st" | "red" | "prefetch" | "bar" | "barrier" | "cp") {
            continue;
        }
        let Some(d0) = ops.first() else { continue };
        // The destination text: a `{a, b}` vector, `p|q`, or the first operand.
        let body = t
            .strip_prefix('@')
            .map_or(t, |r| r.split_once(char::is_whitespace).map_or("", |x| x.1))
            .trim();
        let rest = body
            .split_once(char::is_whitespace)
            .map_or("", |x| x.1)
            .trim_start();
        let dtoks = if rest.starts_with('{') {
            reg_tokens(rest.split('}').next().unwrap_or(""))
        } else if t.contains('|') {
            ops.iter().flat_map(|o| reg_tokens(o)).collect::<Vec<_>>()
        } else {
            reg_tokens(d0)
        };
        for r in dtoks {
            *ndefs.entry(r.to_string()).or_default() += 1;
            let lit = (matches!(op, "mov.b32" | "mov.f32" | "mov.u32")
                && !t.starts_with('@')
                && ops.len() == 2)
                .then(|| {
                    let x = ops[1].trim();
                    parse_int(x)
                        .filter(|_| x.starts_with("0f") || x.starts_with("0F"))
                        .map(|v| v as u32)
                        .or_else(|| float_lit(x, Some(32)).flatten().map(|v| v as u32))
                })
                .flatten();
            defs.insert(r.to_string(), (i, lit));
        }
    }
    // Branch sources per label (a `brx` table target is reachable from anywhere).
    let mut srcs: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, st) in stmts.iter().enumerate() {
        let (op, ops) = stmt_parts(&st.text);
        if op.starts_with("bra") {
            if let Some(l) = ops.last() {
                srcs.entry(l.trim().to_string()).or_default().push(i);
            }
        } else if st.text.contains(".branchtargets") {
            for l in st
                .text
                .split(".branchtargets")
                .nth(1)
                .unwrap_or("")
                .split(',')
            {
                srcs.entry(l.trim().trim_end_matches(';').to_string())
                    .or_default()
                    .push(usize::MAX);
            }
        }
    }
    // `def` dominates `use_` when control enters (def, use_] only through def's fallthrough.
    let dominates = |def: usize, use_: usize| -> bool {
        def < use_
            && stmts[def + 1..=use_].iter().all(|s| {
                s.labels.iter().all(|l| {
                    srcs.get(l)
                        .is_none_or(|v| v.iter().all(|&b| b >= def && b < use_))
                })
            })
    };
    let mut rewrites = Vec::new();
    for (i, st) in stmts.iter().enumerate() {
        let t = st.text.trim().to_string();
        let (guard, body) = match t.strip_prefix('@') {
            Some(r) => {
                let (g, b) = r.split_once(char::is_whitespace).unwrap_or((r, ""));
                (format!("@{g} "), b.trim().to_string())
            }
            None => (String::new(), t.clone()),
        };
        let (op, ops) = stmt_parts(&body);
        let ftz = match op {
            "div.full.f32" | "div.approx.f32" => false,
            "div.full.ftz.f32" | "div.approx.ftz.f32" => true,
            _ => continue,
        };
        if ops.len() != 3 {
            continue;
        }
        let b = ops[2].trim();
        let c = if b.starts_with("0f") || b.starts_with("0F") {
            parse_int(b).map(|v| v as u32)
        } else if b.starts_with('%') {
            match (defs.get(b), ndefs.get(b)) {
                (Some(&(di, Some(v))), Some(1)) if dominates(di, i) => Some(v),
                _ => None,
            }
        } else {
            float_lit(b, Some(32)).flatten().map(|v| v as u32)
        };
        let Some(c) = c else { continue };
        let (d, a) = (ops[0].trim(), ops[1].trim());
        let e = (c >> 23) & 0xff;
        let f = if ftz { ".ftz" } else { "" };
        let new = match c {
            0x3f80_0000 if ftz => format!("add.rn.ftz.f32 {d}, {a}, 0f80000000"),
            0x3f80_0000 => format!("mov.b32 {d}, {a}"),
            0xbf80_0000 => format!("sub.rn{f}.f32 {d}, 0f80000000, {a}"),
            _ if (1..=252).contains(&e) || c & 0x7fff_ffff == 0x7e80_0000 => {
                format!("mul{f}.f32 {d}, {a}, 0f{:08X}", recip_rn_f32(c))
            }
            _ => continue,
        };
        rewrites.push((i, format!("{guard}{new}")));
    }
    for (i, t) in rewrites {
        stmts[i].text = t;
    }
}

/// `createpolicy` destinations used only as the cache-policy operand of `.L2::cache_hint`
/// instructions (their value is then irrelevant to the model).
fn hint_only_policies(insts: &[&Stmt]) -> HashSet<String> {
    let mut cand: HashSet<String> = HashSet::new();
    for st in insts {
        let (op, ops) = stmt_parts(&st.text);
        if op.starts_with("createpolicy.") {
            if let Some(d) = ops.first() {
                cand.insert(d.to_string());
            }
        }
    }
    for st in insts {
        let (op, ops) = stmt_parts(&st.text);
        if op.starts_with("createpolicy.") {
            continue;
        }
        let hint = op.split('.').any(|m| m == "L2::cache_hint");
        let base = op.split('.').next().unwrap_or("");
        let def0 = !matches!(base, "st" | "red" | "atom" | "cp" | "prefetch");
        for (i, o) in ops.iter().enumerate() {
            // A plain definition (`mov.u64 %rd2, 0x0` before the createpolicy) is no use.
            if i == 0 && def0 && !st.text.contains('{') {
                continue;
            }
            if cand.contains(*o) && !(hint && i + 1 == ops.len()) {
                cand.remove(*o);
            }
        }
    }
    cand
}

/// Remove comments, keeping newlines so line numbers survive.
fn strip_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            i += 2;
            while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                if b[i] == b'\n' {
                    out.push('\n');
                }
                i += 1;
            }
            i += 2;
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn line_of(src: &str, byte: usize) -> u32 {
    src[..byte].bytes().filter(|&c| c == b'\n').count() as u32 + 1
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || c == '$' || c == '%'
}

/// Split the body into statements terminated by ';', peeling off leading labels.
fn split_statements(src: &str, start: usize, end: usize) -> Vec<Stmt> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_start = start;
    let mut labels = Vec::new();
    let body = &src[start..end];
    let mut idx = 0;
    let bytes = body.as_bytes();
    while idx < bytes.len() {
        let c = bytes[idx] as char;
        if c == ';' {
            let t = cur.trim().to_string();
            if !t.is_empty() || !labels.is_empty() {
                let lead = cur.len() - cur.trim_start().len();
                out.push(Stmt {
                    line: line_of(src, cur_start + lead),
                    labels: std::mem::take(&mut labels),
                    text: t,
                });
            }
            cur.clear();
            cur_start = start + idx + 1;
        } else if c == ':' && !cur.trim().is_empty() {
            // A label is `IDENT:` at statement start, not part of `::`.
            let t = cur.trim();
            let next = bytes.get(idx + 1).map(|&b| b as char);
            let prev_colon = idx > 0 && bytes[idx - 1] == b':';
            if next != Some(':')
                && !prev_colon
                && t.chars().next().is_some_and(is_ident_start)
                && t.chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
            {
                labels.push(t.to_string());
                cur.clear();
                cur_start = start + idx + 1;
            } else {
                cur.push(c);
            }
        } else {
            if cur.is_empty() {
                cur_start = start + idx;
            }
            cur.push(c);
        }
        idx += 1;
    }
    out
}

/// Split operands at top-level commas.
fn split_operands(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '[' | '{' => {
                depth += 1;
                cur.push(c)
            }
            ']' | '}' => {
                depth -= 1;
                cur.push(c)
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// A float literal as the instruction's float type `ty` (32 / 64): `Some(None)` outside an
/// `.f32` / `.f64` instruction, `None` if `s` is not one that needs converting. PTX evaluates
/// decimal literals (`1.0`, `20.`, `1e-3`, `.5`) as f64; ptxas rounds them to nearest even
/// for an .f32 instruction and converts `0d` (f64 bits) the same way; `0f` stays its raw 32
/// bits, zero-extended in an .f64 instruction (GB300, probes/float_lit), `0d` in .f64 too.
/// ASSUME(L-ROSETTA)
fn float_lit(s: &str, ty: Option<u32>) -> Option<Option<u64>> {
    let (neg, b) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let hex = |p: &str| {
        b.strip_prefix(p)
            .or_else(|| b.strip_prefix(&p.to_uppercase()))
            .and_then(|h| u64::from_str_radix(h, 16).ok())
    };
    let v: f64 = if let Some(h) = hex("0d") {
        if ty != Some(32) {
            return None;
        }
        f64::from_bits(h)
    } else if hex("0f").is_some() {
        return None;
    } else {
        let dec = !b.is_empty()
            && b.bytes()
                .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
            && b.bytes()
                .next()
                .is_some_and(|c| c.is_ascii_digit() || c == b'.')
            && (b.contains('.') || b.contains(['e', 'E']))
            && !b.starts_with("0x");
        if !dec {
            return None;
        }
        b.parse::<f64>().ok()?
    };
    let v = if neg { -v } else { v };
    Some(match ty {
        Some(32) => Some((v as f32).to_bits() as u64),
        Some(64) => Some(v.to_bits()),
        _ => None,
    })
}

/// Generic base of the local window (the SASS front's `c[0x0][0x2f8]`; S-CBANK).
const LOCAL_GENERIC_BASE: u64 = 0x7d00_0000_0000;

fn parse_int(s: &str) -> Option<u64> {
    let s = s.trim();
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let s = s.trim_end_matches('U');
    let v = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(h, 16).ok()?
    } else if let Some(h) = s.strip_prefix("0f").or_else(|| s.strip_prefix("0F")) {
        u64::from_str_radix(h, 16).ok()? // f32 bit pattern
    } else if let Some(h) = s.strip_prefix("0d").or_else(|| s.strip_prefix("0D")) {
        u64::from_str_radix(h, 16).ok()?
    } else {
        s.parse::<u64>().ok()?
    };
    Some(if neg { v.wrapping_neg() } else { v })
}

struct Lower<'a> {
    regs: &'a HashMap<String, Reg>,
    wide_regs: &'a HashSet<String>,
    /// Module `.global` symbols: index into `Kernel::globals`, or why it is not modeled.
    global_syms: &'a HashMap<String, Result<u32, String>>,
    labels: &'a HashMap<String, usize>,
    /// Shared symbols -> shared-window address; param symbols -> param-space offset.
    shared_syms: &'a HashMap<String, u32>,
    /// `.local` arrays of the kernel (`__local_depot0`): offsets in the thread's local frame.
    local_syms: &'a HashMap<String, u32>,
    param_syms: &'a HashMap<String, u32>,
    line: u32,
    text: &'a str,
    /// Ops to run after the lowered one, under the same guard: a PTX instruction that has no
    /// single IR op (`cp.async.mbarrier.arrive`, `vote.sync.any`, `mad.lo.s64`, ...).
    extra: Vec<Op>,
    /// Hidden scratch register for those expansions (allocated only if a statement needs it:
    /// `%__x64`).
    tmp_v: Option<Reg>,
    /// A hidden register nothing writes (`%__undef`): the indeterminate data of `discard`.
    undef_v: Option<Reg>,
    /// Two hidden predicates (`min` / `max.f64`: DSETP.MIN/MAX's select and both-NaN).
    tmp_p: Option<Reg>,
    /// Ops to run before the lowered one, under the same guard (moves into `vec_v`).
    pre: Vec<Op>,
    /// `mul.f16[x2]` / `mul.f32x2` destinations ptxas cannot contract
    /// (`uncontractible_f16_muls`).
    f16_mul_ok: &'a HashSet<String>,
    /// Destinations of float muls without `.rn` (`loose_mul_dsts`).
    loose_muls: &'a HashSet<String>,
    /// `mul.f32x2` destinations whose readers are all kinds `fmad2_pairs` models.
    x2_decided: &'a HashSet<String>,
    /// The statement's float type when its last type modifier is `.f32` / `.f64`: the type
    /// float literals convert to (`float_lit`).
    fimm: Option<u32>,
    /// This statement's PTX-FMAD contraction (`fmad_pairs`), or why it is refused.
    fmad: Option<&'a Fmad>,
    fmad_refuse: Option<&'a String>,
    /// `createpolicy` destinations used only as cache hints (`hint_only_policies`).
    hint_policies: &'a HashSet<String>,
    /// `.branchtargets` lists by their label (`brx.idx`).
    brx_tables: &'a HashMap<String, Vec<String>>,
    /// Hidden consecutive scratch registers for vector operands whose registers are not
    /// consecutive (`st.shared.v4.b32 [a], {%r262, %r261, %r260, %r259}`): `%__vec0..7`.
    vec_v: Option<Reg>,
}

/// Most elements of a vector operand (`.v8`).
const VEC_SCRATCH: u16 = 8;

/// Does this statement have a `{...}` vector operand of an `ld` / `st` whose registers are not
/// consecutive (it needs the vector scratch registers)?
fn needs_vec_scratch(text: &str, regs: &HashMap<String, Reg>) -> bool {
    let t = text.trim();
    let t = match t.strip_prefix('@') {
        Some(r) => r.split_once(char::is_whitespace).map_or("", |x| x.1).trim(),
        None => t,
    };
    // tensormap.replace's read-modify-write holds the map word in `%__vec0`.
    if t.starts_with("tensormap.") {
        return true;
    }
    // tcgen05.mma `[taddr + off]` and try_cancel `[saddr + off]` operands add the offset into
    // a scratch register.
    if t.starts_with("tcgen05.mma") || t.starts_with("clusterlaunchcontrol.try_cancel") {
        return t.split('[').skip(1).any(|x| {
            x.split(']').next().is_some_and(|a| {
                a.contains('+') && !a.trim_end().ends_with("+ 0") && !a.trim_end().ends_with("+0")
            })
        });
    }
    if !(t.starts_with("ld.")
        || t.starts_with("st.")
        || t.starts_with("red.global.add.v")
        || t.starts_with("ldmatrix.")
        || t.starts_with("mma."))
    {
        return false;
    }
    let Some((_, rest)) = t.split_once('{') else {
        return false;
    };
    let Some((inner, _)) = rest.split_once('}') else {
        return false;
    };
    let rs: Vec<Option<&Reg>> = inner.split(',').map(|x| regs.get(x.trim())).collect();
    rs.len() > 1
        && rs.iter().enumerate().any(|(i, r)| match (r, rs[0]) {
            (Some(r), Some(r0)) => r.file != r0.file || r.idx != r0.idx + i as u16,
            _ => true,
        })
}

/// Does this statement's expansion need the hidden scratch register?
fn needs_scratch(text: &str) -> bool {
    let t = text.trim();
    let t = match t.strip_prefix('@') {
        Some(r) => r.split_once(char::is_whitespace).map_or("", |x| x.1).trim(),
        None => t,
    };
    t.starts_with("mad.lo.s64")
        || t.starts_with("tensormap.")
        || t.starts_with("cvt.rna.tf32.f32")
        || t.starts_with("cvt.rn.sat.f16.f32")
        || t.starts_with("sqrt.rn.f32")
        || t.starts_with("sqrt.rn.ftz.f32")
        || t.starts_with("div.rn.ftz.f32")
        || (t.starts_with("min.") || t.starts_with("max."))
            && t.split_whitespace()
                .next()
                .is_some_and(|o| o.ends_with(".f64"))
        || t.starts_with("mad.lo.u64")
        || t.starts_with("mad.wide.s16")
        || t.starts_with("mad.wide.u16")
        || t.starts_with("popc.b64")
        || t.starts_with("mov.b32") && t.contains('{')
        || (t.starts_with("atom.") || t.starts_with("red.")) && t.contains(".v2.bf16")
}

impl<'a> Lower<'a> {
    /// A `.sync` membermask operand: None for the full warp (`0xffffffff`, `-1`), else the
    /// immediate or register naming part of it (E-WARPCOLL; probes/redux_mask,
    /// probes/shfl_mask). An empty or wider immediate, or a predicate, is refused.
    fn member_mask(&self, s: &str) -> Result<Option<Val>> {
        let s = s.trim();
        match parse_int(s) {
            _ if s == "-1" => Ok(None),
            Some(m) if m == u32::MAX as u64 || m == u64::MAX => Ok(None),
            Some(m) if m <= u32::MAX as u64 && m != 0 => Ok(Some(Val::Imm(m))),
            None if self.reg(s).is_ok_and(|r| r.file != File::P) => Ok(Some(self.val(s)?)),
            _ => bail!(self.err("membermask form not modeled")),
        }
    }

    fn scratch_v(&self) -> Result<Reg> {
        self.tmp_v
            .ok_or_else(|| self.err("internal: no scratch register"))
    }

    fn err(&self, msg: impl std::fmt::Display) -> anyhow::Error {
        anyhow!("ptx line {}: {msg}\n    `{}`", self.line, self.text)
    }

    fn reg(&self, s: &str) -> Result<Reg> {
        // A one-element vector `{ %r1 }` (Triton: `ld.global.b32 { %r1 }, [..]`).
        let s = match s.trim().strip_prefix('{').and_then(|x| x.strip_suffix('}')) {
            Some(x) if !x.contains(',') => x.trim(),
            _ => s,
        };
        self.regs
            .get(s)
            .copied()
            .ok_or_else(|| self.err(format!("unknown register {s}")))
    }

    fn sreg(s: &str) -> Option<SReg> {
        let dim = |c: &str| match c {
            "x" => Some(0u8),
            "y" => Some(1),
            "z" => Some(2),
            _ => None,
        };
        let (base, d) = s
            .split_once('.')
            .map(|(a, b)| (a, Some(b)))
            .unwrap_or((s, None));
        Some(match (base, d) {
            ("%tid", Some(c)) => SReg::Tid(dim(c)?),
            ("%ntid", Some(c)) => SReg::NTid(dim(c)?),
            ("%ctaid", Some(c)) => SReg::CtaId(dim(c)?),
            ("%nctaid", Some(c)) => SReg::NCtaId(dim(c)?),
            ("%laneid", None) => SReg::LaneId,
            ("%warpid", None) => SReg::WarpId,
            ("%cluster_ctarank", None) => SReg::ClusterCtaRank,
            ("%cluster_nctarank", None) => SReg::ClusterNCtaRank,
            // probes/cluster_sreg.
            ("%cluster_ctaid", Some(c)) => SReg::ClusterCtaId(dim(c)?),
            ("%cluster_nctaid", Some(c)) => SReg::ClusterNCtaId(dim(c)?),
            ("%clusterid", Some(c)) => SReg::ClusterId(dim(c)?),
            ("%nclusterid", Some(c)) => SReg::NClusterId(dim(c)?),
            ("%lanemask_lt", None) => SReg::LaneMaskLt,
            ("%lanemask_le", None) => SReg::LaneMaskLe,
            ("%lanemask_ge", None) => SReg::LaneMaskGe,
            ("%lanemask_gt", None) => SReg::LaneMaskGt,
            ("%lanemask_eq", None) => SReg::LaneMaskEq,
            _ => return None,
        })
    }

    fn val(&self, s: &str) -> Result<Val> {
        let s = s.trim();
        let s = match s.strip_prefix('{').and_then(|x| x.strip_suffix('}')) {
            Some(x) if !x.contains(',') => x.trim(),
            _ => s,
        };
        if let Some(sr) = Self::sreg(s) {
            return Ok(Val::SReg(sr));
        }
        // ASSUME(E-CLOCK): clock reads are 0, as on the SASS front (timeouts never fire).
        if matches!(
            s,
            "%clock"
                | "%clock_hi"
                | "%clock64"
                | "%globaltimer"
                | "%globaltimer_lo"
                | "%globaltimer_hi"
        ) {
            return Ok(Val::Imm(0));
        }
        // Registers declared without `%` (`{.reg .s64 a; ...}` inline asm, renamed `a__sN`).
        if s.starts_with('%') || self.regs.contains_key(s) {
            return Ok(Val::Reg(self.reg(s)?, 1));
        }
        if let Some(v) = float_lit(s, self.fimm) {
            return v.map(Val::Imm).ok_or_else(|| {
                self.err(format!(
                    "float literal `{s}` outside an .f32 / .f64 instruction not modeled"
                ))
            });
        }
        if let Some(v) = parse_int(s) {
            return Ok(Val::Imm(v));
        }
        if let Some(g) = self.global_syms.get(s) {
            let i = g.clone().map_err(|e| self.err(e))?;
            return Ok(Val::SReg(SReg::GlobalAddr(i)));
        }
        if let Some(&a) = self.shared_syms.get(s) {
            return Ok(Val::SReg(SReg::SharedAddr(a)));
        }
        if let Some(&a) = self.local_syms.get(s) {
            return Ok(Val::Imm(a as u64));
        }
        if let Some(&a) = self.param_syms.get(s) {
            return Ok(Val::Imm(a as u64));
        }
        Err(self.err(format!("bad operand `{s}`")))
    }

    /// Predicate operand, optionally negated (`%p3`, `!%p3`).
    fn pred_src(&self, s: &str) -> Result<Pred> {
        let s = s.trim();
        let (neg, s) = match s.strip_prefix('!') {
            Some(r) => (true, r),
            None => (false, s),
        };
        Ok(Pred {
            reg: Some(self.pred_dst(s)?),
            neg,
        })
    }

    /// An `ld` / `st` vector operand `{%r16, %r17, ...}` -> (first register, count, moves).
    /// Registers that are not consecutive go through the vector scratch registers: the moves
    /// into them run before a store (`pre`), out of them after a load (`extra`).
    fn reg_vec_mem(&self, s: &str, load: bool) -> Result<(Reg, u8, Vec<Op>)> {
        if let Ok((r, n)) = self.reg_vec(s) {
            return Ok((r, n, vec![]));
        }
        let mut moves = Vec::new();
        let inner = s.trim().trim_start_matches('{').trim_end_matches('}');
        // Store sources may be immediates (`st.global.v2.b32 [a], {0, %r120}`).
        let vals: Vec<Val> = inner
            .split(',')
            .map(|r| match self.reg(r.trim()) {
                Ok(r) => Ok(Val::Reg(r, 1)),
                Err(e) if load => Err(e),
                Err(_) => self.val(r.trim()),
            })
            .collect::<Result<_>>()?;
        let base = self
            .vec_v
            .ok_or_else(|| self.err("internal: no vector scratch registers"))?;
        let bad = |v: &Val| {
            !matches!(v, Val::Reg(r, _) if r.file == File::V || false) && !matches!(v, Val::Imm(_))
        };
        if vals.len() > VEC_SCRATCH as usize || vals.iter().any(bad) {
            return Err(self.err("vector operand registers must be consecutive"));
        }
        for (i, v) in vals.iter().enumerate() {
            let tmp = Reg {
                file: File::V,
                idx: base.idx + i as u16,
            };
            moves.push(match (load, v) {
                (true, Val::Reg(r, _)) => Op::Mov {
                    d: *r,
                    a: Val::Reg(tmp, 1),
                },
                _ => Op::Mov {
                    d: tmp,
                    a: v.clone(),
                },
            });
        }
        Ok((base, vals.len() as u8, moves))
    }

    /// `{%r16, %r17, ...}` -> (first register, count); registers must be consecutive.
    fn reg_vec(&self, s: &str) -> Result<(Reg, u8)> {
        // A vector register (`.reg .v4 .s32 abcd`) named whole: its components.
        let t = s.trim();
        if !t.starts_with('{') && self.regs.contains_key(&format!("{t}.y")) {
            let n = if self.regs.contains_key(&format!("{t}.w")) {
                4
            } else {
                2
            };
            let list: Vec<String> = ["x", "y", "z", "w"][..n]
                .iter()
                .map(|c| format!("{t}.{c}"))
                .collect();
            return self.reg_vec(&format!("{{{}}}", list.join(", ")));
        }
        let inner = s.trim().trim_start_matches('{').trim_end_matches('}');
        let regs: Vec<Reg> = inner
            .split(',')
            .map(|r| self.reg(r.trim()))
            .collect::<Result<_>>()?;
        for (i, r) in regs.iter().enumerate() {
            if r.idx != regs[0].idx + i as u16 || r.file != regs[0].file {
                return Err(self.err("vector operand registers must be consecutive"));
            }
        }
        Ok((regs[0], regs.len() as u8))
    }

    fn pred_dst(&self, s: &str) -> Result<Reg> {
        let r = self.reg(s.trim())?;
        if r.file != File::P {
            return Err(self.err(format!("{s} is not a predicate")));
        }
        Ok(r)
    }

    fn dst(&self, s: &str) -> Result<Reg> {
        self.reg(s.trim())
    }

    /// `[base]`, `[base+off]`, `[sym]`, `[sym+off]`, `[imm]`.
    fn addr(&self, s: &str) -> Result<Addr> {
        let inner = s
            .trim()
            .strip_prefix('[')
            .and_then(|x| x.strip_suffix(']'))
            .ok_or_else(|| self.err(format!("expected [address], got `{s}`")))?
            .trim();
        let (b, off) = if let Some(p) = inner.rfind(['+', '-']).filter(|&p| p > 0) {
            let o =
                parse_int(&inner[p..].replace('+', "")).ok_or_else(|| self.err("bad offset"))?;
            // `[%rd17+-2097152]`: a `+` before the signed offset.
            let b = inner[..p].trim();
            (b.strip_suffix('+').unwrap_or(b).trim(), o as i64)
        } else {
            (inner, 0)
        };
        if b.starts_with('%') || self.regs.contains_key(b) {
            return Ok(Addr {
                base: Some(Val::Reg(self.reg(b)?, 1)),
                base2: None,
                off,
            });
        }
        if let Some(g) = self.global_syms.get(b) {
            let i = g.clone().map_err(|e| self.err(e))?;
            return Ok(Addr {
                base: Some(Val::SReg(SReg::GlobalAddr(i))),
                base2: None,
                off,
            });
        }
        if let Some(&a) = self.shared_syms.get(b) {
            return Ok(Addr {
                base: Some(Val::SReg(SReg::SharedAddr(a))),
                base2: None,
                off,
            });
        }
        if let Some(&a) = self.param_syms.get(b) {
            return Ok(Addr {
                base: None,
                base2: None,
                off: a as i64 + off,
            });
        }
        if let Some(v) = parse_int(b) {
            return Ok(Addr {
                base: None,
                base2: None,
                off: v as i64 + off,
            });
        }
        Err(self.err(format!("bad address `{s}`")))
    }
}

/// Narrow-format `cvt` forms (N-CVT-LP): `.<f8|f6|f4|ue8m0>x2` from f32 pairs or f16x2 /
/// bf16x2, the `.rs` x4 / f16x2 / bf16x2 packs, and the f16x2 / bf16x2 decodes. None: not a
/// narrow form.
fn cvt_lp(lw: &Lower, mods: &[&str], ops: &[String]) -> Result<Option<Op>> {
    // `cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 d, a, scale` (PTX ISA 9.1): the two
    // s2f6 codes of the .b16 `a` times the two ue8m0 scales of the .b16 `scale` (N-CVT-LP;
    // ptxas: F2FP.SATFINITE.BF16.S2_6.UNPACK_B).
    if mods == ["rn", "satfinite", "scaled::n2::ue8m0", "bf16x2", "s2f6x2"] {
        if ops.len() != 3 {
            bail!(lw.err("cvt .s2f6x2: expected d, a, scale"));
        }
        return Ok(Some(Op::CvtS2f6x2 {
            d: lw.dst(ops[0].trim())?,
            a: lw.val(ops[1].trim())?,
            hi: false,
            scale: lw.val(ops[2].trim())?,
        }));
    }
    let n = mods.len();
    let (to, from) = (mods[n - 2], mods[n - 1]);
    let has = |m: &str| mods.contains(&m);
    let lp = |t: &str| -> Option<(Lp, u8)> {
        let (base, x) = t.split_at(t.len().checked_sub(2)?);
        let x = x.strip_prefix('x')?.parse::<u8>().ok()?;
        Some((
            match base {
                "e4m3" => Lp::E4m3,
                "e5m2" => Lp::E5m2,
                "e2m3" => Lp::E2m3,
                "e3m2" => Lp::E3m2,
                "e2m1" => Lp::E2m1,
                "ue8m0" => Lp::Ue8m0,
                "f16" => Lp::F16,
                "bf16" => Lp::Bf16,
                _ => return None,
            },
            x,
        ))
    };
    let rnd = if has("rn") {
        LpRnd::Rn
    } else if has("rz") {
        LpRnd::Rz
    } else if has("rp") {
        LpRnd::Rp
    } else if has("rs") {
        LpRnd::Rs
    } else {
        return Ok(None);
    };
    let (sat, relu) = (has("satfinite"), has("relu"));
    let op = |i: usize| -> Result<&str> {
        ops.get(i)
            .map(|s| s.as_str())
            .ok_or_else(|| lw.err(format!("cvt: missing operand {i}")))
    };
    Ok(Some(match (lp(to), from) {
        // Encodes from an f32 pair: d = {a (high lane), b}.
        (Some((t, 2)), "f32")
            if !matches!(t, Lp::F16 | Lp::Bf16) || rnd == LpRnd::Rs || rnd == LpRnd::Rn && sat =>
        {
            let rs = rnd == LpRnd::Rs;
            Op::CvtPack {
                d: lw.dst(op(0)?)?,
                a: lw.val(op(1)?)?,
                b: lw.val(op(2)?)?,
                c: if rs { lw.val(op(3)?)? } else { Val::Imm(0) },
                half: None,
                to: t,
                rnd,
                sat,
                relu,
            }
        }
        (Some((t, 4)), "f32") if rnd == LpRnd::Rs => {
            let v: Vec<Val> = op(1)?
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| lw.val(x.trim()))
                .collect::<Result<_>>()?;
            let src: [Val; 4] = v
                .try_into()
                .map_err(|_| lw.err("cvt .x4 wants four sources"))?;
            Op::CvtRsX4 {
                d: lw.dst(op(0)?)?,
                src,
                rbits: lw.val(op(2)?)?,
                to: t,
                relu,
            }
        }
        (Some((t, 2)), "f16x2" | "bf16x2") if !matches!(t, Lp::F16 | Lp::Bf16) => Op::CvtPack {
            d: lw.dst(op(0)?)?,
            a: lw.val(op(1)?)?,
            b: Val::Imm(0),
            c: Val::Imm(0),
            half: Some(if from == "f16x2" { Lp::F16 } else { Lp::Bf16 }),
            to: t,
            rnd,
            sat,
            relu,
        },
        // Decodes: d = {high code, low code} as f16x2 / bf16x2.
        (Some((t @ (Lp::F16 | Lp::Bf16), 2)), _) => match lp(from) {
            Some((f, 2)) if !matches!(f, Lp::F16 | Lp::Bf16) && rnd == LpRnd::Rn => Op::CvtUnpack {
                d: lw.dst(op(0)?)?,
                a: lw.val(op(1)?)?,
                hi: false,
                from: f,
                to: t,
                relu,
                byte: 0,
            },
            _ => return Ok(None),
        },
        _ => return Ok(None),
    }))
}

fn type_bits(t: &str) -> Option<u8> {
    Some(match t {
        "b8" | "u8" | "s8" => 8,
        "b16" | "u16" | "s16" | "f16" | "bf16" => 16,
        "b32" | "u32" | "s32" | "f32" => 32,
        "b64" | "u64" | "s64" | "f64" => 64,
        "b128" => 128,
        "pred" => 1,
        _ => return None,
    })
}

fn space_of(m: &str) -> Option<Space> {
    Some(match m {
        "shared" | "shared::cta" => Space::Shared,
        "shared::cluster" => Space::SharedCluster,
        "global" => Space::Global,
        "param" => Space::Param,
        _ => return None,
    })
}

fn scope_of(mods: &[&str]) -> Option<Scope> {
    mods.iter().find_map(|m| match *m {
        "cta" => Some(Scope::Cta),
        "cluster" => Some(Scope::Cluster),
        "gpu" => Some(Scope::Gpu),
        "sys" => Some(Scope::Sys),
        _ => None,
    })
}

/// Memory semantics of an ld / st / atom / red. `.volatile` is relaxed at system scope (PTX
/// ISA: volatile = relaxed.sys for memory-model purposes); `atom` / `red` default to
/// relaxed.gpu. ASSUME(S-SCOPE)
fn mem_sync(mods: &[&str], rmw: bool) -> Result<Option<MemSync>> {
    let sem = mods.iter().find_map(|m| match *m {
        "relaxed" => Some(Sem::Relaxed),
        "acquire" => Some(Sem::Acquire),
        "release" => Some(Sem::Release),
        "acq_rel" => Some(Sem::AcqRel),
        _ => None,
    });
    if mods.contains(&"volatile") {
        return Ok(Some(MemSync {
            sem: Sem::Relaxed,
            scope: Scope::Sys,
        }));
    }
    // ld/st.mmio.relaxed.sys: a strong relaxed system-scope access (SASS STG.E.64.MMIO.SYS,
    // probes/memhints).
    if mods.contains(&"mmio") {
        if sem != Some(Sem::Relaxed) || scope_of(mods) != Some(Scope::Sys) || rmw {
            bail!("only ld/st.mmio.relaxed.sys is modeled");
        }
        return Ok(Some(MemSync {
            sem: Sem::Relaxed,
            scope: Scope::Sys,
        }));
    }
    Ok(match (sem, rmw) {
        (None, false) => None,
        (sem, _) => Some(MemSync {
            sem: sem.unwrap_or(Sem::Relaxed),
            scope: scope_of(mods).unwrap_or(Scope::Gpu),
        }),
    })
}

/// Inline-asm scopes (`{ .reg .pred p; L: ... }` inside an entry body, as Triton emits for
/// every try_wait) are flattened: labels and registers declared in a scope are renamed
/// `name__sN` within it, and the braces become spaces, so lines keep their positions. A scope
/// brace is one at statement position (after `;`, a scope brace, a label, or a directive line
/// such as `.loc`); vector operand braces, also those starting a continuation line of an
/// instruction (`mma.sync...f32\n {%r1, ..},`), are not.
fn flatten_scopes(src: &str) -> String {
    let b = src.as_bytes();
    // Pass 1: find scope braces (depth >= 2: inside an entry body).
    let mut depth = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    let mut scopes: Vec<(usize, usize)> = Vec::new();
    let mut prev = b';';
    // First character of the statement in progress (None: at statement position).
    let mut pending: Option<u8> = None;
    // An array initializer (`.global .b8 _$_str[11] = {95, 95, ...};`, Triton's assert
    // strings) is not a scope.
    let mut in_init = false;
    for (i, &c) in b.iter().enumerate() {
        let stmt_pos = pending.is_none();
        if in_init {
            in_init = c != b'}';
            if !c.is_ascii_whitespace() {
                prev = c;
            }
            continue;
        }
        match c {
            b'{' if prev == b'=' => in_init = true,
            b'{' if depth == 0 || stmt_pos => {
                depth += 1;
                stack.push(i);
                pending = None;
                prev = c;
                continue;
            }
            b'}' if stmt_pos && !stack.is_empty() => {
                let open = stack.pop().unwrap();
                if depth >= 2 {
                    scopes.push((open, i));
                }
                depth -= 1;
                pending = None;
                prev = c;
                continue;
            }
            _ => {}
        }
        let label_colon = c == b':' && b.get(i + 1) != Some(&b':') && (i == 0 || b[i - 1] != b':');
        if c == b';' || label_colon {
            pending = None;
        } else if c == b'\n' && pending == Some(b'.') {
            // A directive line without `;` (`.loc 1 2 3`, `.maxntid 128, 1, 1`) ends here.
            pending = None;
        } else if !c.is_ascii_whitespace() && pending.is_none() {
            pending = Some(c);
        }
        if !c.is_ascii_whitespace() {
            prev = c;
        }
    }
    if scopes.is_empty() {
        return src.to_string();
    }
    let mut out: Vec<u8> = b.to_vec();
    /// A scope's brace offsets and its (old, new) name pairs.
    type ScopeRenames = (usize, usize, Vec<(String, String)>);
    let mut renames: Vec<ScopeRenames> = Vec::new();
    for (n, &(o, c)) in scopes.iter().enumerate() {
        let mut body = b[o + 1..c].to_vec();
        for &(inner_open, inner_close) in &scopes {
            if o < inner_open && inner_close < c {
                for byte in &mut body[inner_open - o - 1..inner_close - o] {
                    if *byte != b'\n' {
                        *byte = b' ';
                    }
                }
            }
        }
        let body = String::from_utf8(body).unwrap();
        let mut names = Vec::new();
        for st in body.split(';') {
            let t = st.trim();
            // labels: `IDENT:` at statement start (possibly several)
            let mut rest = t;
            while let Some((l, r)) = rest.split_once(':') {
                let l = l.trim();
                if !l.is_empty()
                    && l.chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '$')
                    && !r.starts_with(':')
                {
                    names.push((l.to_string(), format!("{l}__s{n}")));
                    rest = r;
                } else {
                    break;
                }
            }
            if let Some(r) = rest.trim().strip_prefix(".reg") {
                for tok in r.split([' ', '\t', '\n', ',']) {
                    let tok = tok.trim();
                    if !tok.is_empty() && !tok.starts_with('.') {
                        let (base, count) = match tok.split_once('<') {
                            Some((b, c)) => {
                                (b, c.trim_end_matches('>').trim().parse::<usize>().ok())
                            }
                            None => (tok, None),
                        };
                        names.push((base.to_string(), format!("{base}__s{n}")));
                        // `.reg .u32 %tmp32_<1>` declares %tmp32_0 (renamed %tmp32___sN0, as
                        // the declaration `%tmp32___sN<1>` expands): its uses are renamed too,
                        // so a scoped `%p<1>` no longer aliases the body's %p0.
                        for i in 0..count.unwrap_or(0) {
                            names.push((format!("{base}{i}"), format!("{base}__s{n}{i}")));
                        }
                    }
                }
            }
        }
        let map = names;
        renames.push((o, c, map));
        out[o] = b' ';
        out[c] = b' ';
    }
    // Resolve each token against its innermost declaring scope using original offsets.
    let mut result = Vec::with_capacity(out.len());
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_alphabetic() || matches!(b[i], b'_' | b'$' | b'%') {
            let start = i;
            while i < b.len()
                && (b[i].is_ascii_alphanumeric() || matches!(b[i], b'_' | b'$' | b'%'))
            {
                i += 1;
            }
            let word = &src[start..i];
            let replacement = if start > 0 && b[start - 1] == b'.' {
                None
            } else {
                renames
                    .iter()
                    .filter(|(o, c, _)| *o < start && start < *c)
                    .find_map(|(_, _, map)| map.iter().find(|(name, _)| name == word))
            };
            result
                .extend_from_slice(replacement.map_or(&out[start..i], |(_, name)| name.as_bytes()));
        } else {
            result.push(out[i]);
            i += 1;
        }
    }
    String::from_utf8(result).unwrap()
}

/// `[.align A] .type name[N]` of a shared declaration: (name, alignment, bytes). `name[]`
/// (extern) has size 0.
fn shared_decl(rest: &str) -> Result<(String, u32, u32)> {
    let toks: Vec<&str> = rest.split_whitespace().collect();
    let mut align = 1u32;
    let mut elem = 1u32;
    let mut size = 0u32;
    let mut name = String::new();
    let mut i = 0;
    while i < toks.len() {
        if toks[i] == ".align" {
            align = toks[i + 1].parse()?;
            i += 1;
        } else if let Some(b) = toks[i].strip_prefix('.').and_then(type_bits) {
            elem = (b as u32 / 8).max(1);
        } else if let Some((n, c)) = toks[i].split_once('[') {
            name = n.to_string();
            let c = c.trim_end_matches(']');
            size = if c.is_empty() {
                0
            } else {
                elem * c.parse::<u32>()?
            };
        } else {
            name = toks[i].to_string();
            size = elem;
        }
        i += 1;
    }
    Ok((name, align, size))
}

/// Parse a PTX module and lower the named entry (or the only entry).
pub fn parse_kernel(src: &str, entry: Option<&str>) -> Result<Kernel> {
    // `.loc` / `.file` debug directives end at the newline, not at a `;`: drop them (keeping
    // the newline) so they do not swallow the next statement.
    let src: String = strip_comments(src)
        .lines()
        .map(|l| {
            let t = l.trim_start();
            // (`.local` declarations stay: only `.loc` itself.)
            let dir = |d: &str| {
                t.strip_prefix(d)
                    .is_some_and(|r| r.is_empty() || r.starts_with(char::is_whitespace))
            };
            if dir(".loc") || dir(".file") { "" } else { l }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let src = flatten_scopes(&src);
    let mut pos = 0;
    let (name, params_src, body_start, body_end, entry_at) = loop {
        let at = src[pos..]
            .find(".entry")
            .ok_or_else(|| anyhow!("no .entry found"))?
            + pos;
        let after = &src[at + 6..];
        let name_end = after.find(['(', ' ', '\n', '\t']).unwrap_or(after.len());
        let mut name = after[..name_end].trim().to_string();
        if name.is_empty() {
            name = after
                .trim_start()
                .split(['(', ' ', '\n'])
                .next()
                .unwrap_or("")
                .to_string();
        }
        let lp = at
            + 6
            + after
                .find('(')
                .ok_or_else(|| anyhow!("entry without params"))?;
        let rp = lp
            + src[lp..]
                .find(')')
                .ok_or_else(|| anyhow!("unterminated params"))?;
        let lb = rp
            + src[rp..]
                .find('{')
                .ok_or_else(|| anyhow!("entry without body"))?;
        let mut depth = 0;
        let mut rb = lb;
        for (i, c) in src[lb..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        rb = lb + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if entry.is_none_or(|e| e == name) {
            break (name, src[lp + 1..rp].to_string(), lb + 1, rb, at);
        }
        pos = rb;
    };

    // Parameters: `.param .align A .b8 name[N]` / `.param .u64 name`.
    let mut params = Vec::new();
    let mut param_syms = HashMap::new();
    let mut off = 0u32;
    for p in params_src
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        let toks: Vec<&str> = p.split_whitespace().collect();
        let mut align = 0u32;
        let mut elem = 0u32;
        let mut name = String::new();
        let mut count = 1u32;
        let mut i = 0;
        // `.param .u64 .ptr .global .align 1 p`: after `.ptr`, the state space and alignment
        // describe the pointee, not the parameter.
        let mut ptr = false;
        while i < toks.len() {
            let t = toks[i];
            if t == ".ptr" {
                ptr = true;
            } else if ptr && matches!(t, ".global" | ".shared" | ".const" | ".local") {
            } else if t == ".align" {
                if !ptr {
                    align = toks[i + 1].parse()?;
                }
                i += 1;
            } else if let Some(b) = t.strip_prefix('.').and_then(type_bits) {
                elem = (b as u32 / 8).max(1);
            } else if t != ".param" {
                if let Some((n, c)) = t.split_once('[') {
                    name = n.to_string();
                    count = c.trim_end_matches(']').parse()?;
                } else {
                    name = t.to_string();
                }
            }
            i += 1;
        }
        let size = elem * count;
        let align = if align == 0 { elem } else { align };
        off = off.div_ceil(align) * align;
        param_syms.insert(name.clone(), off);
        params.push(Param {
            name,
            offset: off,
            size,
        });
        off += size;
    }

    let mut stmts = split_statements(&src, body_start, body_end);
    div_by_const(&mut stmts);
    vprintf_calls(&mut stmts);

    // Pass 1: declarations, labels.
    let mut regs = HashMap::new();
    // Registers declared 64 bits wide (`.reg .b64` / `.s64` / `.u64`): a signed narrower load
    // into one sign-extends to 64 bits.
    let mut wide_regs: HashSet<String> = HashSet::new();
    let mut vreg_names = Vec::new();
    let mut npred = 0u16;
    let mut shared_syms = HashMap::new();
    let mut local_syms = HashMap::new();
    let mut local_bytes = 0u32;
    let mut shared_syms_pre: Vec<(String, u32)> = Vec::new();
    let mut smem = SMEM_RESERVED;
    // Module-scope shared variables (before the entry). `.extern .shared x[]` is the dynamic
    // shared memory window: it starts after all static shared memory.
    let mut externs = Vec::new();
    // Module-scope `.global` variables (DeepEP's `epSyncer`, printf's `$str`): one
    // `GlobalSym` each (initializer bytes, or None: the launch's S-GLOBAL-INIT buffer).
    let mut globals: Vec<GlobalSym> = Vec::new();
    let mut global_syms: HashMap<String, Result<u32, String>> = HashMap::new();
    for line in src[..entry_at].split([';', '\n']) {
        let t = line.trim();
        let g = t
            .trim_start_matches(".visible")
            .trim_start()
            .trim_start_matches(".extern")
            .trim_start()
            .trim_start_matches(".weak")
            .trim_start();
        if let Some(rest) = g.strip_prefix(".global ") {
            let (decl, init) = match rest.split_once('=') {
                Some((d, i)) => (d, Some(i.trim())),
                None => (rest, None),
            };
            let (name, _align, size) = shared_decl(decl)?;
            let elem = decl
                .split_whitespace()
                .find_map(|x| x.strip_prefix('.').and_then(type_bits))
                .map_or(1, |b| (b as usize / 8).max(1));
            let bytes = match init {
                None => Ok(None),
                Some(i) => {
                    let vals: Option<Vec<u64>> = i
                        .trim_start_matches('{')
                        .trim_end_matches('}')
                        .split(',')
                        .map(|x| parse_int(x.trim()))
                        .collect();
                    match vals {
                        Some(v) if v.len() * elem <= size as usize => {
                            let mut b: Vec<u8> = v
                                .iter()
                                .flat_map(|x| x.to_le_bytes()[..elem].to_vec())
                                .collect();
                            b.resize(size as usize, 0);
                            Ok(Some(b))
                        }
                        _ => Err(format!("global {name}: initializer `{i}` not modeled")),
                    }
                }
            };
            match bytes {
                Ok(init) => {
                    global_syms.insert(name.clone(), Ok(globals.len() as u32));
                    globals.push(GlobalSym {
                        name,
                        size: size as u64,
                        init,
                    });
                }
                Err(e) => {
                    global_syms.insert(name, Err(e));
                }
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix(".extern") {
            if let Some(rest) = rest.trim().strip_prefix(".shared") {
                let (name, align, _) = shared_decl(rest)?;
                externs.push((name, align));
            }
        } else if let Some(rest) = t.strip_prefix(".shared") {
            let (name, align, size) = shared_decl(rest)?;
            smem = smem.div_ceil(align) * align;
            shared_syms_pre.push((name, smem));
            smem += size;
        }
    }
    let mut labels = HashMap::new();
    let mut insts_src: Vec<&Stmt> = Vec::new();
    let mut brx_tables: HashMap<String, Vec<String>> = HashMap::new();
    for st in &stmts {
        for l in &st.labels {
            labels.insert(l.clone(), insts_src.len());
        }
        let t = st.text.as_str();
        if t.is_empty() {
            continue;
        }
        if let Some(rest) = t.strip_prefix(".reg") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            // `.reg .v4 .s32 abcd;` (an inline-asm vector register): consecutive slots named
            // `abcd.x`, `.y`, `.z`, `.w`; `abcd` itself is the first.
            if let Some(n) = toks
                .first()
                .and_then(|t| t.strip_prefix(".v"))
                .and_then(|n| n.parse::<usize>().ok())
            {
                if !matches!(n, 2 | 4) || toks.len() < 3 || toks[1] == ".pred" {
                    bail!(
                        "ptx line {}: vector register declaration not modeled",
                        st.line
                    );
                }
                for nm in toks[2..].join(" ").split(',') {
                    let nm = nm.trim();
                    if nm.contains('<') {
                        bail!("ptx line {}: vector register ranges not modeled", st.line);
                    }
                    for (k, c) in ["x", "y", "z", "w"][..n].iter().enumerate() {
                        vreg_names.push(format!("{nm}.{c}"));
                        let r = Reg {
                            file: File::V,
                            idx: (vreg_names.len() - 1) as u16,
                        };
                        if matches!(toks[1], ".b64" | ".s64" | ".u64") {
                            wide_regs.insert(format!("{nm}.{c}"));
                        }
                        if k == 0 {
                            regs.insert(nm.to_string(), r);
                        }
                        regs.insert(format!("{nm}.{c}"), r);
                    }
                }
                continue;
            }
            let is_pred = toks.first() == Some(&".pred");
            // A .b128 register takes two consecutive 64-bit slots (low, high).
            let is_b128 = toks.first() == Some(&".b128");
            for decl in toks[1..].join(" ").split(',') {
                let decl = decl.trim();
                let names: Vec<String> = if let Some((base, n)) = decl.split_once('<') {
                    let n: u32 = n.trim_end_matches('>').parse()?;
                    (0..n).map(|i| format!("{base}{i}")).collect()
                } else {
                    vec![decl.to_string()]
                };
                for nm in names {
                    let r = if is_pred {
                        npred += 1;
                        Reg {
                            file: File::P,
                            idx: npred - 1,
                        }
                    } else {
                        vreg_names.push(nm.clone());
                        let r = Reg {
                            file: File::V,
                            idx: (vreg_names.len() - 1) as u16,
                        };
                        if is_b128 {
                            vreg_names.push(format!("{nm}.hi"));
                            regs.insert(
                                format!("{nm}.hi"),
                                Reg {
                                    file: File::V,
                                    idx: (vreg_names.len() - 1) as u16,
                                },
                            );
                        }
                        r
                    };
                    if matches!(toks.first(), Some(&".b64" | &".s64" | &".u64")) {
                        wide_regs.insert(nm.clone());
                    }
                    regs.insert(nm, r);
                }
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix(".shared") {
            let (name, align, size) = shared_decl(rest)?;
            smem = smem.div_ceil(align) * align;
            shared_syms.insert(name, smem);
            smem += size;
            continue;
        }
        // `.local .align A .b8 __local_depot0[N]` (the compiler's stack frame, addressed through
        // `%SPL`): per-thread local memory, as SASS LDL / STL (offsets from 0).
        if let Some(rest) = t.strip_prefix(".local") {
            let (name, align, size) = shared_decl(rest)?;
            local_bytes = local_bytes.div_ceil(align) * align;
            local_syms.insert(name, local_bytes);
            local_bytes += size;
            continue;
        }
        // `$L_brx_0: .branchtargets L1, L2, ...;`: the target list of `brx.idx i, $L_brx_0`.
        if let Some(rest) = t.strip_prefix(".branchtargets") {
            let names: Vec<String> = rest.split(',').map(|x| x.trim().to_string()).collect();
            for l in &st.labels {
                brx_tables.insert(l.clone(), names.clone());
            }
            continue;
        }
        if t.starts_with('.') {
            continue; // other directives (.maxntid, .pragma, ...) do not affect semantics here
        }
        insts_src.push(st);
    }

    for (n, a) in shared_syms_pre {
        shared_syms.insert(n, a);
    }
    for (n, align) in externs {
        smem = smem.div_ceil(align.max(16)) * align.max(16);
        shared_syms.insert(n, smem);
    }

    // Scratch registers for multi-op expansions.
    let mut tmp_v = None;
    if insts_src.iter().any(|st| needs_scratch(&st.text)) {
        vreg_names.push("%__x64".to_string());
        tmp_v = Some(Reg {
            file: File::V,
            idx: (vreg_names.len() - 1) as u16,
        });
    }
    let mut undef_v = None;
    if insts_src
        .iter()
        .any(|st| stmt_parts(&st.text).0.starts_with("discard.") || st.text == "__vprintf")
    {
        vreg_names.push("%__undef".to_string());
        undef_v = Some(Reg {
            file: File::V,
            idx: (vreg_names.len() - 1) as u16,
        });
        regs.insert("%__undef".to_string(), undef_v.unwrap());
    }
    let mut tmp_p = None;
    if insts_src.iter().any(|st| {
        let op = stmt_parts(&st.text).0;
        (op.starts_with("min.") || op.starts_with("max.")) && op.ends_with(".f64")
    }) {
        tmp_p = Some(Reg {
            file: File::P,
            idx: npred,
        });
        npred += 2;
    }
    let mut vec_v = None;
    if insts_src
        .iter()
        .any(|st| needs_vec_scratch(&st.text, &regs))
    {
        vec_v = Some(Reg {
            file: File::V,
            idx: vreg_names.len() as u16,
        });
        for i in 0..VEC_SCRATCH {
            vreg_names.push(format!("%__vec{i}"));
        }
    }
    // The numerics track the high half of a 64-bit register under `idx | 0x8000`
    // (sim::num::V_HI): indices stay below it.
    if vreg_names.len() > 0x8000 {
        bail!("kernel {name}: more than 32768 virtual registers");
    }

    let f16_mul_ok = uncontractible_f16_muls(&insts_src);
    let loose_muls = loose_mul_dsts(&insts_src);
    let (mut fmad, mut fmad_refuse) = fmad_pairs(&insts_src, &brx_tables);
    let (fmad2, fmad2_refuse, mut x2_decided) = fmad2_pairs(&insts_src, &brx_tables);
    x2_decided.extend(x2_no_add_readers(&insts_src));
    fmad.extend(fmad2);
    fmad_refuse.extend(fmad2_refuse);
    let hint_policies = hint_only_policies(&insts_src);
    // Pass 2: instructions. A statement may lower to several ops (`Lower::extra`); branch
    // targets and labels are statement indices until remapped below.
    let mut code = Vec::new();
    let mut start = Vec::with_capacity(insts_src.len() + 1);
    // PTXV_LIFT_REPORT: list every statement the front-end does not lower (grouped by reason
    // and opcode) instead of stopping at the first, as the SASS front does.
    let report = std::env::var("PTXV_LIFT_REPORT").is_ok();
    let mut unlowered: Vec<(String, usize, String)> = Vec::new();
    for (si, st) in insts_src.iter().enumerate() {
        let mut lw = Lower {
            regs: &regs,
            wide_regs: &wide_regs,
            global_syms: &global_syms,
            labels: &labels,
            shared_syms: &shared_syms,
            local_syms: &local_syms,
            param_syms: &param_syms,
            line: st.line,
            text: &st.text,
            extra: Vec::new(),
            tmp_v,
            undef_v,
            tmp_p,
            pre: Vec::new(),
            vec_v,
            f16_mul_ok: &f16_mul_ok,
            loose_muls: &loose_muls,
            x2_decided: &x2_decided,
            fimm: None,
            fmad: fmad.get(&si),
            fmad_refuse: fmad_refuse.get(&si),
            hint_policies: &hint_policies,
            brx_tables: &brx_tables,
        };
        let inst = match lower_inst(&mut lw) {
            Ok(i) => i,
            Err(e) if report => {
                let msg = format!("{e:#}");
                let first = msg.lines().next().unwrap_or("").to_string();
                let why = match first.split_once(": ") {
                    Some((l, r)) if l.starts_with("ptx line ") => r.to_string(),
                    _ => first,
                };
                let opc = st
                    .text
                    .split_whitespace()
                    .find(|t| !t.starts_with('@'))
                    .unwrap_or("");
                let key = format!("{why} [{opc}]");
                match unlowered.iter_mut().find(|u| u.0 == key) {
                    Some(u) => u.1 += 1,
                    None => unlowered.push((key, 1, st.text.replace('\n', " "))),
                }
                start.push(code.len());
                continue;
            }
            Err(e) => return Err(e).with_context(|| format!("in kernel {name}")),
        };
        start.push(code.len());
        let (guard, loc, text) = (inst.guard, inst.loc, inst.text.clone());
        let mut ops = lw.pre.clone();
        ops.push(inst.op.clone());
        ops.extend(lw.extra.iter().cloned());
        // Every op of an expansion runs under the statement's guard, so none but the last
        // may write it.
        if let Some(g) = guard.reg {
            if ops[..ops.len() - 1]
                .iter()
                .any(|o| o.dst_regs().contains(&g))
            {
                return Err(lw.err("expansion writes its own guard predicate"))
                    .with_context(|| format!("in kernel {name}"));
            }
        }
        for op in lw.pre {
            code.push(Inst {
                guard,
                op,
                uniform: false,
                loc,
                text: text.clone(),
            });
        }
        code.push(inst);
        for op in lw.extra {
            code.push(Inst {
                guard,
                op,
                uniform: false,
                loc,
                text: text.clone(),
            });
        }
    }
    if report {
        eprintln!(
            "lift report for {name}: {} kinds of instruction not lowered",
            unlowered.len()
        );
        for (why, n, ex) in &unlowered {
            eprintln!("{n:5} {why}\n      e.g. {ex}");
        }
        if !unlowered.is_empty() {
            bail!(
                "in kernel {name}: {} kinds of instruction not lowered",
                unlowered.len()
            );
        }
    }
    start.push(code.len());
    if code.len() != insts_src.len() {
        for i in code.iter_mut() {
            if let Op::Bra { target, .. } = &mut i.op {
                *target = start[*target];
            }
            if let Op::Brx { targets, .. } = &mut i.op {
                targets.iter_mut().for_each(|t| *t = start[*t]);
            }
        }
        for v in labels.values_mut() {
            *v = start[*v];
        }
    }

    // `.reqntid x[, y[, z]]` / `.maxntid ...` between the parameter list and the body.
    let ntid = {
        let head = &src[entry_at..body_start];
        let head = head.rsplit_once(')').map_or(head, |x| x.1);
        let dir = |d: &str| -> Option<[u32; 3]> {
            let i = head.find(d)? + d.len();
            let rest = &head[i..];
            let end = rest.find(['.', '{', '\n']).unwrap_or(rest.len());
            let v: Vec<u32> = rest[..end]
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            (!v.is_empty() && v.len() <= 3).then(|| {
                [
                    v[0],
                    v.get(1).copied().unwrap_or(1),
                    v.get(2).copied().unwrap_or(1),
                ]
            })
        };
        dir(".reqntid")
            .map(|dims| Ntid { req: true, dims })
            .or_else(|| dir(".maxntid").map(|dims| Ntid { req: false, dims }))
    };
    Ok(Kernel {
        name,
        front: Front::Ptx,
        code,
        nregs: [vreg_names.len() as u16, 0, 0, npred, 0],
        params,
        param_cbank: 0x380,
        smem_bytes: smem,
        vreg_names,
        labels: {
            let mut l: Vec<(usize, String)> = labels.iter().map(|(n, i)| (*i, n.clone())).collect();
            l.sort();
            l
        },
        static_findings: vec![],
        bulk_read_waits: vec![],
        bulk_commit_sb: vec![],
        parts: vec![],
        ntid,
        global_relocs: (0..globals.len() as u32)
            .map(|i| GlobalReloc {
                bank: GLOBAL_RELOC_BANK,
                off: 8 * i,
                global: i,
                addend: 0,
            })
            .collect(),
        globals,
        static_image: None,
    })
}

fn lower_inst(lw: &mut Lower) -> Result<Inst> {
    let text = lw.text.trim();
    let (guard, rest) = if let Some(r) = text.strip_prefix('@') {
        let (g, rest) = r
            .split_once(char::is_whitespace)
            .ok_or_else(|| lw.err("bad guard"))?;
        let (neg, g) = match g.strip_prefix('!') {
            Some(x) => (true, x),
            None => (false, g),
        };
        (
            Pred {
                reg: Some(lw.pred_dst(g)?),
                neg,
            },
            rest.trim(),
        )
    } else {
        (Pred::TRUE, text)
    };
    let (opcode, ops_src) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let ops = split_operands(ops_src);
    let parts: Vec<&str> = opcode.split('.').collect();
    let base = parts[0];
    let mods = &parts[1..];
    let has = |m: &str| mods.contains(&m);
    if let Some(w) = lw.fmad_refuse {
        bail!(lw.err(format!("{w}; not modeled")));
    }
    lw.fimm = match mods.last() {
        Some(&"f32") => Some(32),
        Some(&"f64") => Some(64),
        _ => None,
    };
    let ty = mods
        .iter()
        .rev()
        .find_map(|m| type_bits(m).map(|b| (b, *m)));
    let bits = ty.map(|t| t.0).unwrap_or(32);
    let tname = ty.map(|t| t.1).unwrap_or("b32");
    let mut extra: Vec<Op> = Vec::new();
    // Ops to run before the lowered one (moved into `lw.pre` at the end).
    let mut pre: Vec<Op> = Vec::new();
    let op_n = |i: usize| -> Result<&str> {
        ops.get(i)
            .map(|s| s.as_str())
            .ok_or_else(|| lw.err(format!("missing operand {i}")))
    };

    // The 16-bit float type of an arithmetic op (`bf16x2` / `f16x2` are not in `type_bits`).
    let ht = mods.last().copied().unwrap_or("");
    let half_t = matches!(ht, "bf16" | "bf16x2" | "f16" | "f16x2");
    let op = match base {
        // Agent L2's forms (l2.rs): cvt.r*i between equal float types, cvt f64 <-> f16 /
        // bf16 and f16 <-> bf16, min / max .u16x2 / .s16x2, cp.reduce.async.bulk .add.f32.
        _ if l2::claims(base, mods) => l2::lower(lw, base, mods, &ops)?,
        // Packed / scalar bf16 arithmetic as the SASS ptxas emits (`HFMA2.BF16_V2`, `HMUL2`,
        // `HADD2`; probes/bf16x2, both front-ends): `fma.rn[.relu]`, `mul.rn`, `add.rn`,
        // `sub.rn` .bf16x2 / .bf16 and `neg.bf16x2` / `.bf16` (-0*0 - x: a NaN becomes 0x7fff).
        // Without `.rn` ptxas may contract a mul and an add into one HFMA2: refused, as are
        // immediates, `.ftz` / `.sat` / `.oob` and f16 arithmetic. ASSUME(N-BF16X2)
        // Mixed precision `fma` / `add` / `sub` `.{rn,rz,rm,rp}[.sat].f32.bf16`: SASS FHFMA.BF16 /
        // FHADD.BF16 (probes/bf16_ext, both front-ends). ASSUME(N-BF16X2) `.f32.f16`: FHFMA /
        // FHADD without .BF16 (probes/fhadd). ASSUME(N-FHADD)
        "add" | "sub" | "fma"
            if matches!(ht, "bf16" | "f16") && mods.len() >= 2 && mods[mods.len() - 2] == "f32" =>
        {
            let f16 = ht == "f16";
            let rest = &mods[..mods.len() - 2];
            let rnd = match rest.iter().find(|m| **m != "sat") {
                Some(&"rn") => Rnd::Rn,
                Some(&"rz") => Rnd::Rz,
                Some(&"rm") => Rnd::Rm,
                Some(&"rp") => Rnd::Rp,
                _ => bail!(lw.err(format!("{base}.f32.{ht} needs a rounding mode"))),
            };
            if rest.len() > 2 || rest.len() == 2 && !rest.contains(&"sat") {
                bail!(lw.err(format!(
                    "{base}.f32.{ht} with modifiers {rest:?} not modeled"
                )));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.f32.{ht} immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            let (a, b, c) = if base == "fma" {
                (reg(1)?, reg(2)?, reg(3)?)
            } else {
                (
                    reg(1)?,
                    Val::Imm(if f16 { 0x3c00 } else { 0x3f80 }),
                    reg(2)?,
                )
            };
            Op::FfmaBf16 {
                d,
                a,
                b,
                c,
                hi: [false; 2],
                neg: [false, false, base == "sub"],
                rnd,
                sat: has("sat"),
                f16,
            }
        }
        // `min` / `max` [.NaN] [.xorsign.abs] `.bf16x2` / `.bf16`: SASS HMNMX2.BF16_V2
        // (probes/bf16_ext). ASSUME(N-BF16X2)
        "min" | "max" if matches!(ht, "bf16" | "bf16x2") => {
            let rest = &mods[..mods.len() - 1];
            if rest
                .iter()
                .any(|m| !matches!(*m, "NaN" | "xorsign" | "abs"))
                || has("xorsign") != has("abs")
            {
                bail!(lw.err(format!("{base}.{ht} with modifiers {rest:?} not modeled")));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.{ht} immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            let op = Op::Hmnmx2Bf16 {
                d,
                a: reg(1)?,
                b: reg(2)?,
                sel: [HSel::Pair; 2],
                min: Pred {
                    reg: None,
                    neg: base == "max",
                },
                nan: has("NaN"),
                xorsign: has("xorsign"),
            };
            if ht == "bf16" {
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
            }
            op
        }
        // f16 arithmetic as ptxas emits it (SASS `HFMA2` / `HMUL2` / `HADD2`, probes/f16_arith):
        // `fma.rn`, `mul.rn`, `add.rn`, `sub.rn` `.f16` / `.f16x2`, and `mul.f16[x2]` without
        // `.rn` where no use lets ptxas contract it (`uncontractible_f16_muls`). Register
        // operands only; `.ftz` / `.sat` / `.relu` are refused. ASSUME(L-ROSETTA)
        "add" | "sub" | "mul" | "fma" if matches!(ht, "f16" | "f16x2") => {
            let rest = &mods[..mods.len() - 1];
            let d_name = op_n(0)?.trim();
            // Scalar .f16 without `.rn`: fmad_pairs decides the contraction (PTX-FMAD); a plain
            // mul or add it does not contract (or refuse) rounds once.
            let scalar_plain =
                ht == "f16" && rest.is_empty() && matches!(base, "mul" | "add" | "sub");
            let ok = rest == ["rn"]
                || base == "mul" && rest.is_empty() && lw.f16_mul_ok.contains(d_name)
                || scalar_plain;
            if !ok {
                bail!(lw.err(format!(
                    "{base}.{ht} with modifiers {rest:?} not modeled (only .rn, or a mul ptxas cannot contract)"
                )));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.{ht} immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            // PTX-FMAD: the contracted add / sub is one HFMA2 (probes/fmad_f16).
            // ASSUME(PTX-FMAD)
            if let (true, "add" | "sub", Some(f)) = (scalar_plain, base, lw.fmad) {
                let sub = base == "sub";
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
                Op::Hfma2 {
                    d,
                    a: lw.val(&f.a)?,
                    b: lw.val(&f.b)?,
                    c: reg(3 - f.slot)?,
                    abs_a: false,
                    neg_a: (sub && f.slot == 2) ^ f.neg,
                    neg_c: sub && f.slot == 1,
                    sel: [HSel::Pair; 3],
                }
            } else {
                let (a, b, c, neg_c) = match base {
                    "mul" => (reg(1)?, reg(2)?, Val::Imm(0x8000_8000), false),
                    "add" | "sub" => (reg(1)?, Val::Imm(0x3c00_3c00), reg(2)?, base == "sub"),
                    _ => (reg(1)?, reg(2)?, reg(3)?, false),
                };
                let op = Op::Hfma2 {
                    d,
                    a,
                    b,
                    c,
                    abs_a: false,
                    neg_a: false,
                    neg_c,
                    sel: [HSel::Pair; 3],
                };
                if ht == "f16" {
                    extra.push(Op::Int {
                        op: IntOp::And,
                        bits: 32,
                        d,
                        a: Val::Reg(d, 1),
                        b: Val::Imm(0xffff),
                    });
                }
                op
            }
        }
        // `abs` / `neg` `.f16` / `.f16x2` (SASS `HADD2 d, |R|, -RZ.H0_H0` / `HADD2 d, -R,
        // -RZ.H0_H0`): |x| or -x plus -0 (a NaN becomes the canonical 0x7fff;
        // probes/f16_absneg). ASSUME(L-ROSETTA)
        "abs" | "neg" if matches!(ht, "f16" | "f16x2") && mods.len() == 1 => {
            let d = lw.dst(op_n(0)?)?;
            let a = lw.val(op_n(1)?)?;
            if !matches!(a, Val::Reg(..)) {
                bail!(lw.err(format!("{base}.{ht} immediate operand not modeled")));
            }
            if ht == "f16" {
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
            }
            Op::Hfma2 {
                d,
                a,
                b: Val::Imm(0x3c00_3c00),
                c: Val::Imm(0x8000_8000),
                abs_a: base == "abs",
                neg_a: base == "neg",
                neg_c: false,
                sel: [HSel::Pair; 3],
            }
        }
        // `abs.bf16x2` / `.bf16`: -0*0 + |x| (SASS `HFMA2.BF16_V2 d, -RZ.H0_H0, RZ.H0_H0, |R|`;
        // a NaN becomes 0x7fff).
        "add" | "sub" | "mul" | "fma" | "neg" | "abs" if matches!(ht, "bf16" | "bf16x2") => {
            let rest = &mods[..mods.len() - 1];
            // No rounding modifier: .rn where ptxas cannot contract, as for f32x2 (a mul no use
            // of which lets ptxas contract it; an add / sub of no loose mul; probes/bf16_norn).
            let plain_ok = rest.is_empty()
                && match base {
                    "mul" => lw.f16_mul_ok.contains(op_n(0)?.trim()),
                    "add" | "sub" => ![op_n(1)?, op_n(2)?]
                        .iter()
                        .any(|o| lw.loose_muls.contains(o.trim())),
                    _ => false,
                };
            let ok = match base {
                "neg" | "abs" => rest.is_empty(),
                "fma" => rest == ["rn"] || rest == ["rn", "relu"],
                _ => rest == ["rn"] || plain_ok,
            };
            if !ok {
                bail!(lw.err(format!(
                    "{base}.{ht} with modifiers {rest:?} not modeled (only .rn, fma .rn.relu)"
                )));
            }
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err(format!("{base}.{ht} immediate operand not modeled")));
                }
                Ok(v)
            };
            let d = lw.dst(op_n(0)?)?;
            let (a, b, c, neg) = match base {
                "mul" => (reg(1)?, reg(2)?, Val::Imm(0x8000_8000), [false; 3]),
                "add" | "sub" => (
                    reg(1)?,
                    Val::Imm(0x3f80_3f80),
                    reg(2)?,
                    [false, false, base == "sub"],
                ),
                "neg" => (
                    Val::Imm(0x8000_8000),
                    Val::Imm(0),
                    reg(1)?,
                    [false, false, true],
                ),
                "abs" => (Val::Imm(0x8000_8000), Val::Imm(0), reg(1)?, [false; 3]),
                _ => (reg(1)?, reg(2)?, reg(3)?, [false; 3]),
            };
            let op = Op::Hfma2Bf16 {
                d,
                a,
                b,
                c,
                sel: [HSel::Pair; 3],
                neg,
                abs: [false, false, base == "abs"],
                relu: has("relu"),
            };
            if ht == "bf16" {
                // A 16-bit register: the high lane (computed from the operands' high bits)
                // cleared.
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
            }
            op
        }
        "add" | "sub" | "mul" | "fma" | "neg" | "abs" | "min" | "max" if half_t => {
            bail!(lw.err(format!("{base}.{ht} not modeled")))
        }
        // `mov.b128 q, {lo, hi}` / `mov.b128 {lo, hi}, q`: a .b128 register is two 64-bit
        // slots (CuTe DSL packs a cluster-launch-control response loaded as .v2.b64 for
        // clusterlaunchcontrol.query_cancel).
        "mov" if bits == 128 => {
            let (vec, q) = if op_n(1)?.trim().starts_with('{') {
                (op_n(1)?, op_n(0)?)
            } else {
                (op_n(0)?, op_n(1)?)
            };
            let halves: Vec<&str> = vec
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| x.trim())
                .collect();
            if halves.len() != 2 || !vec.trim().starts_with('{') {
                bail!(lw.err("only mov.b128 q, {lo, hi} and mov.b128 {lo, hi}, q are modeled"));
            }
            let q = q.trim();
            let (qlo, qhi) = (lw.reg(q)?, lw.reg(&format!("{q}.hi"))?);
            let pack = op_n(1)?.trim().starts_with('{');
            let mut ops = Vec::new();
            for (h, qr) in halves.iter().zip([qlo, qhi]) {
                if pack {
                    ops.push(Op::Mov {
                        d: qr,
                        a: lw.val(h)?,
                    });
                } else if *h != "_" {
                    ops.push(Op::Mov {
                        d: lw.dst(h)?,
                        a: Val::Reg(qr, 1),
                    });
                }
            }
            if ops.is_empty() {
                bail!(lw.err("mov.b128 {_, _} has no destination"));
            }
            let last = ops.pop().unwrap();
            pre.extend(ops);
            last
        }
        "mov" if op_n(0)?.trim().starts_with('{') => {
            let halves: Vec<&str> = op_n(0)?
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| x.trim())
                .collect();
            if !(halves.len() == 2 && matches!(bits, 16 | 32 | 64)
                || halves.len() == 4 && bits == 32)
            {
                bail!(lw.err(
                    "only mov.b64 {lo, hi} / mov.b32 {lo, hi} / mov.b32 {b0, b1, b2, b3} / mov.b16 {lo, hi} unpacking is modeled"
                ));
            }
            let r = |x: &str| -> Result<Option<Reg>> {
                if x == "_" {
                    Ok(None)
                } else {
                    lw.dst(x).map(Some)
                }
            };
            if bits <= 32 {
                // `mov.b32 {h0, h1}, r`: two 16-bit halves, as PRMT with zero upper bytes.
                let a = lw.val(op_n(1)?)?;
                let half = |d: Reg, sel: u64| Op::Prmt {
                    d,
                    a: a.clone(),
                    sel: Val::Imm(sel),
                    b: Val::Imm(0),
                };
                // `mov.b32 {b0, b1, b2, b3}, r`: four bytes, each zero-extended.
                // `mov.b16 {b0, b1}, h`: two bytes (probes/mov_pack).
                let sels: &[u64] = if halves.len() == 4 {
                    &[0x4440, 0x4441, 0x4442, 0x4443]
                } else if bits == 16 {
                    &[0x4440, 0x4441]
                } else {
                    &[0x4410, 0x4432]
                };
                let mut ops: Vec<Op> = halves
                    .iter()
                    .zip(sels)
                    .map(|(h, &sel)| Ok((r(h)?, sel)))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .filter_map(|(d, sel)| d.map(|d| half(d, sel)))
                    .collect();
                if ops.is_empty() {
                    bail!(lw.err("mov.b32 {_, _} has no destination"));
                }
                let first = ops.remove(0);
                extra.extend(ops);
                first
            } else {
                Op::Unpack64 {
                    lo: r(halves[0])?,
                    hi: r(halves[1])?,
                    a: lw.val(op_n(1)?)?,
                }
            }
        }
        "mov" if op_n(1)?.trim().starts_with('{') => {
            let halves: Vec<&str> = op_n(1)?
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| x.trim())
                .collect();
            if !(halves.len() == 2 && matches!(bits, 16 | 32 | 64)
                || halves.len() == 4 && bits == 32)
            {
                bail!(lw.err(
                    "only mov.b64 / mov.b32 / mov.b16 %d, {lo, hi} and mov.b32 %d, {b0, b1, b2, b3} packing is modeled"
                ));
            }
            if halves.len() == 4 {
                // `mov.b32 r, {b0, b1, b2, b3}`: the low byte of each, b0 lowest; three PRMTs
                // through the scratch register (probes/mov_pack).
                let t = lw
                    .tmp_v
                    .ok_or_else(|| lw.err("internal: no scratch register"))?;
                let tv = Val::Reg(t, 1);
                let b = |i: usize| lw.val(halves[i]);
                pre.push(Op::Prmt {
                    d: t,
                    a: b(0)?,
                    sel: Val::Imm(0x0040),
                    b: b(1)?,
                });
                pre.push(Op::Prmt {
                    d: t,
                    a: tv.clone(),
                    sel: Val::Imm(0x0410),
                    b: b(2)?,
                });
                Op::Prmt {
                    d: lw.dst(op_n(0)?)?,
                    a: tv,
                    sel: Val::Imm(0x4210),
                    b: b(3)?,
                }
            } else if bits == 16 {
                // `mov.b16 h, {lo, hi}`: the low byte of each, upper half cleared.
                let d = lw.dst(op_n(0)?)?;
                extra.push(Op::Int {
                    op: IntOp::And,
                    bits: 32,
                    d,
                    a: Val::Reg(d, 1),
                    b: Val::Imm(0xffff),
                });
                Op::Prmt {
                    d,
                    a: lw.val(halves[0])?,
                    sel: Val::Imm(0x0040),
                    b: lw.val(halves[1])?,
                }
            } else if bits == 32 {
                // `mov.b32 r, {h0, h1}`: r = h1 << 16 | h0 (PRMT bytes a0 a1 b0 b1).
                Op::Prmt {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(halves[0])?,
                    sel: Val::Imm(0x5410),
                    b: lw.val(halves[1])?,
                }
            } else {
                Op::Pack64 {
                    d: lw.dst(op_n(0)?)?,
                    lo: lw.val(halves[0])?,
                    hi: lw.val(halves[1])?,
                }
            }
        }
        "mov" if !has("pred") => Op::Mov {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
        },
        // Packed f32x2 (sm_100+): `fma.rn.f32x2`, `mul.rn.f32x2`, `add.rn.f32x2`, `sub.rn
        // .f32x2` on .b64 registers, lane 0 in the low 32 bits. ptxas emits FFMA2 / FMUL2 /
        // FADD2: one rounding per lane, subnormals kept, canonical NaN (fixtures/s_ptx_packed
        // = GB300; `.rz` / `.rm` / `.rp`: FFMA2.RZ etc., probes/bf16_ext). ptxas contracts a mul
        // read only by adds / subs of its rounding mode, modifiers or not (`fmad2_pairs`,
        // PTX-FMAD). Without a rounding mode otherwise: accepted as .rn only for a mul no
        // use of which lets ptxas contract it (`uncontractible_f16_muls`) and for an add / sub
        // that reads no such mul (probes/f32x2_norn = GB300). `.ftz`: FFMA2.FTZ etc., inputs
        // and results flushed per lane (probes/f32x2_ftz = GB200); a `.ftz` mul read by a
        // same-mode add is refused (`fmad2_pairs`). ptxas rejects `.sat`. ASSUME(L-ROSETTA)
        "add" | "sub" | "mul" | "fma" if mods.last() == Some(&"f32x2") => {
            let ftz = mods.contains(&"ftz");
            let rest: Vec<&str> = mods[..mods.len() - 1]
                .iter()
                .copied()
                .filter(|m| *m != "ftz")
                .collect();
            let rest = &rest[..];
            // A mul whose readers fmad2_pairs decides (contracted or not, its own value is
            // rounded once), an add / sub it contracts, or one reading no undecided loose mul.
            let plain_ok = rest.is_empty()
                && match base {
                    "mul" => {
                        let d = op_n(0)?.trim();
                        lw.f16_mul_ok.contains(d) || lw.x2_decided.contains(d)
                    }
                    "add" | "sub" => {
                        lw.fmad.is_some()
                            || ![op_n(1)?, op_n(2)?].iter().any(|o| {
                                let o = o.trim();
                                lw.loose_muls.contains(o) && !lw.x2_decided.contains(o)
                            })
                    }
                    _ => false,
                };
            let rnd = match rest {
                [] if plain_ok => Rnd::Rn,
                ["rn"] => Rnd::Rn,
                ["rz"] => Rnd::Rz,
                ["rm"] => Rnd::Rm,
                ["rp"] => Rnd::Rp,
                _ => bail!(lw.err(format!(
                    "{base}.f32x2 with modifiers {rest:?} not modeled (one of .rn / .rz / .rm / .rp, optional .ftz)"
                ))),
            };
            let reg = |s: &str| -> Result<(Val, F2Lanes)> {
                match lw.val(s)? {
                    v @ Val::Reg(..) => Ok((v, F2Lanes::Pair)),
                    _ => bail!(lw.err(format!("{base}.f32x2 operand `{s}` must be a register"))),
                }
            };
            let d = lw.dst(op_n(0)?)?;
            let (a, b) = (reg(op_n(1)?)?, reg(op_n(2)?)?);
            let one = (Val::Imm(0x3f80_0000), F2Lanes::Bcast);
            match base {
                // PTX-FMAD (fmad2_pairs): one FFMA2 in the add's rounding mode, a sub negating
                // the product (slot 2) or the addend. ASSUME(PTX-FMAD)
                "add" | "sub" if lw.fmad.is_some() => {
                    let f = lw.fmad.unwrap();
                    let sub = base == "sub";
                    Op::Ffma2 {
                        d,
                        a: reg(&f.a)?,
                        b: reg(&f.b)?,
                        c: reg(op_n(3 - f.slot)?)?,
                        neg: [sub && f.slot == 2, false, sub && f.slot == 1],
                        abs: [false; 3],
                        rnd,
                        ftz,
                    }
                }
                // a*1 + b: the rounded sum (as FADD2); sub negates b.
                "add" | "sub" => Op::Ffma2 {
                    d,
                    a,
                    b: one,
                    c: b,
                    neg: [false, false, base == "sub"],
                    abs: [false; 3],
                    rnd,
                    ftz,
                },
                // a*b + (-0): the rounded product (as FMUL2; +0 under .rm).
                "mul" => Op::Ffma2 {
                    d,
                    a,
                    b,
                    c: (
                        Val::Imm(if rnd == Rnd::Rm { 0 } else { 0x8000_0000 }),
                        F2Lanes::Bcast,
                    ),
                    neg: [false; 3],
                    abs: [false; 3],
                    rnd,
                    ftz,
                },
                _ => Op::Ffma2 {
                    d,
                    a,
                    b,
                    c: reg(op_n(3)?)?,
                    neg: [false; 3],
                    abs: [false; 3],
                    rnd,
                    ftz,
                },
            }
        }
        // PTX-FMAD: ptxas contracts the plain mul producing operand `slot` into this plain
        // add / sub: one FFMA / DFMA, a sub negating the product (slot 2) or the addend.
        // ASSUME(PTX-FMAD)
        "add" | "sub" if matches!(tname, "f32" | "f64") && lw.fmad.is_some() => {
            let f = lw.fmad.unwrap();
            let other = lw.val(op_n(3 - f.slot)?)?;
            let (a, b) = (lw.val(&f.a)?, lw.val(&f.b)?);
            let neg = [
                (base == "sub" && f.slot == 2) ^ f.neg,
                false,
                base == "sub" && f.slot == 1,
            ];
            let d = lw.dst(op_n(0)?)?;
            if tname == "f64" {
                Op::Dfma {
                    d,
                    a,
                    b,
                    c: Some(other),
                    neg,
                    abs: [false; 3],
                    rnd: Rnd::Rn,
                }
            } else {
                Op::Ffma {
                    d,
                    a,
                    b,
                    c: Some(other),
                    neg,
                    abs: [false; 3],
                    rnd: Rnd::Rn,
                    ftz: has("ftz"),
                    sat: has("sat"),
                }
            }
        }
        // f64 arithmetic: as ptxas lowers it (DADD a, -b for sub; DMUL; DFMA).
        "add" | "sub" | "mul" | "fma" if tname == "f64" => {
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            let (a, b) = (lw.val(op_n(1)?)?, lw.val(op_n(2)?)?);
            let d = lw.dst(op_n(0)?)?;
            match base {
                "add" | "sub" => Op::Dfma {
                    d,
                    a,
                    b: Val::Imm(1f64.to_bits()),
                    c: Some(b),
                    neg: [false, false, base == "sub"],
                    abs: [false; 3],
                    rnd,
                },
                "mul" => Op::Dfma {
                    d,
                    a,
                    b,
                    c: None,
                    neg: [false; 3],
                    abs: [false; 3],
                    rnd,
                },
                _ => Op::Dfma {
                    d,
                    a,
                    b,
                    c: Some(lw.val(op_n(3)?)?),
                    neg: [false; 3],
                    abs: [false; 3],
                    rnd,
                },
            }
        }
        "div" | "sqrt" if tname == "f64" && ["rn", "rz", "rm", "rp"].iter().any(|m| has(m)) => {
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            Op::DivSqrt64 {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: if base == "div" {
                    Some(lw.val(op_n(2)?)?)
                } else {
                    None
                },
                rnd,
            }
        }
        "add" | "sub" if tname.starts_with('f') => {
            if bits != 32 {
                bail!(lw.err("only add.f32 / sub.f32 are modeled so far"));
            }
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            Op::Fadd {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                rnd,
                neg: [false, base == "sub"],
                abs: [false; 2],
                ftz: has("ftz"),
                sat: has("sat"),
            }
        }
        // mul.f32 / fma.rn.f32 (one rounding; mul is the rounded product).
        "mul" | "fma" if tname == "f32" && !has("wide") && !has("hi") => {
            let rnd = if has("rz") {
                Rnd::Rz
            } else if has("rm") {
                Rnd::Rm
            } else if has("rp") {
                Rnd::Rp
            } else {
                Rnd::Rn
            };
            Op::Ffma {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                c: if base == "fma" {
                    Some(lw.val(op_n(3)?)?)
                } else {
                    None
                },
                neg: [false; 3],
                abs: [false; 3],
                rnd,
                ftz: has("ftz"),
                sat: has("sat"),
            }
        }
        // ASSUME(N-MUFU): see ASSUMPTIONS.md
        // rsqrt.approx.f64 (no .ftz): ptxas's refinement and slow path (probes/rsqrt_f64).
        "rsqrt" if mods == ["approx", "f64"] => Op::Rsqrt64Approx {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
        },
        "ex2" | "rcp" | "rsqrt" | "lg2" | "sin" | "cos" | "tanh" | "sqrt" if has("approx") => {
            let func = match base {
                "ex2" => MufuFunc::Ex2,
                "rcp" => MufuFunc::Rcp,
                "rsqrt" => MufuFunc::Rsq,
                "lg2" => MufuFunc::Lg2,
                "sin" => MufuFunc::Sin,
                "cos" => MufuFunc::Cos,
                "tanh" => MufuFunc::Tanh,
                _ => MufuFunc::Sqrt,
            };
            let fmt = match mods.last().copied() {
                Some("f32") => MufuFmt::F32,
                Some("f16") => MufuFmt::F16,
                Some("bf16") => MufuFmt::Bf16,
                Some("f16x2") => MufuFmt::F16x2,
                Some("bf16x2") => MufuFmt::Bf16x2,
                // `rsqrt.approx.f64` (no .ftz) adds a Newton step and a slow path: not modeled.
                Some("f64") if has("ftz") && matches!(base, "rcp" | "rsqrt") => MufuFmt::F64,
                _ => bail!(lw.err(format!("{base}.approx of this type is not modeled"))),
            };
            let func = match (func, fmt) {
                (MufuFunc::Rcp, MufuFmt::F64) => MufuFunc::Rcp64h,
                (MufuFunc::Rsq, MufuFmt::F64) => MufuFunc::Rsq64h,
                (f, MufuFmt::F32 | MufuFmt::F64) => f,
                (f @ (MufuFunc::Ex2 | MufuFunc::Tanh), _) => f,
                _ => bail!(lw.err(format!("{base}.approx of this type is not modeled"))),
            };
            Op::Mufu {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                func,
                fmt,
                hi: false,
                neg: false,
                abs: false,
                ptx: Some(has("ftz")),
            }
        }
        // Predicate logic: and/or/xor/not/mov on .pred (a = 0xF0, b = 0xCC LUT convention).
        "and" | "or" | "xor" | "not" | "mov" if has("pred") => {
            let d = lw.pred_dst(op_n(0)?)?;
            let src = |i: usize| -> Result<Pred> {
                let o = op_n(i)?;
                match parse_int(o) {
                    Some(0) => Ok(Pred {
                        reg: None,
                        neg: true,
                    }),
                    Some(_) => Ok(Pred::TRUE),
                    None => lw.pred_src(o),
                }
            };
            let a = src(1)?;
            let (b, lut) = match base {
                "and" => (src(2)?, 0xc0),
                "or" => (src(2)?, 0xfc),
                "xor" => (src(2)?, 0x3c),
                "not" => (Pred::TRUE, 0x0f),
                _ => (Pred::TRUE, 0xf0),
            };
            Op::Plop3 {
                d,
                a,
                b,
                c: Pred::TRUE,
                lut,
                d2: None,
            }
        }
        "selp" => {
            // 32-bit selp of an immediate: the register gets 32 bits (exec keeps all 64 bits
            // of a virtual register, for selp.b64).
            let w = |v: Val| match v {
                Val::Imm(x) if bits <= 32 => Val::Imm(x & 0xffff_ffff),
                v => v,
            };
            Op::Sel {
                d: lw.dst(op_n(0)?)?,
                a: w(lw.val(op_n(1)?)?),
                b: w(lw.val(op_n(2)?)?),
                p: lw.pred_src(op_n(3)?)?,
            }
        }
        // `bfi.b32` (ptxas: PRMT / BMSK / SHF / LOP3; probes/bfi_clz). `.b64` is not modeled.
        "bfi" if mods == ["b32"] => Op::Bfi {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            pos: lw.val(op_n(3)?)?,
            len: lw.val(op_n(4)?)?,
        },
        // `clz.b32 d, a` = 31 - bfind.u32 a (ptxas: FLO.U32 + IADD3 -d, 0x1f; 32 for 0;
        // probes/bfi_clz). `.b64` is not modeled.
        "clz" if mods == ["b32"] => {
            let d = lw.dst(op_n(0)?)?;
            extra.push(Op::Int {
                op: IntOp::Sub,
                bits: 32,
                d,
                a: Val::Imm(31),
                b: Val::Reg(d, 1),
            });
            Op::Flo {
                d,
                a: lw.val(op_n(1)?)?,
                sh: false,
            }
        }
        "bfe" if matches!(tname, "s32" | "u32" | "s64" | "u64") && mods.len() == 1 => Op::Bfe {
            bits,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            pos: lw.val(op_n(2)?)?,
            len: lw.val(op_n(3)?)?,
            signed: tname.starts_with('s'),
        },
        // min.f32 / max.f32 [.ftz] [.NaN] (2 or 3 sources), as SASS FMNMX / FMNMX3; `.abs`
        // (three sources): FMNMX3 |a|, |b|, |c|; `.xorsign.abs` (two sources): FMNMX.XORSIGN
        // (probes/fmnmx_xorsign). ASSUME(N-XORSIGN)
        // min / max.f64 as ptxas: DSETP.MIN/MAX.AND p, q, a, b (p: a is the min / max, -0
        // below +0, a NaN loses; q: both NaN), a 64-bit select, and with q b's NaN quieted
        // (high word | 0x80000) (probes/f64_misc).
        "min" | "max" if mods == ["f64"] => {
            let p = lw
                .tmp_p
                .ok_or_else(|| lw.err("internal: no scratch predicates"))?;
            let q = Reg {
                file: File::P,
                idx: p.idx + 1,
            };
            let tmp = lw.scratch_v()?;
            let d = lw.dst(op_n(0)?)?;
            let (a, b) = (lw.val(op_n(1)?)?, lw.val(op_n(2)?)?);
            pre.push(Op::DsetpMinMax {
                d: p,
                q: Some(q),
                a: a.clone(),
                b: b.clone(),
                max: base == "max",
            });
            pre.push(Op::Sel {
                d,
                a,
                b,
                p: Pred {
                    reg: Some(p),
                    neg: false,
                },
            });
            pre.push(Op::Int {
                op: IntOp::Or,
                bits: 64,
                d: tmp,
                a: Val::Reg(d, 2),
                b: Val::Imm(0x0008_0000_0000_0000),
            });
            Op::Sel {
                d,
                a: Val::Reg(tmp, 2),
                b: Val::Reg(d, 2),
                p: Pred {
                    reg: Some(q),
                    neg: false,
                },
            }
        }
        // neg / abs.f64 (ptxas: DADD d, -RZ, -a / DADD d, -RZ, |a|; probes/f64_misc).
        "neg" | "abs" if mods == ["f64"] => Op::Dfma {
            d: lw.dst(op_n(0)?)?,
            a: Val::Imm(0),
            b: Val::Imm(1f64.to_bits()),
            c: Some(lw.val(op_n(1)?)?),
            neg: [true, false, base == "neg"],
            abs: [false, false, base == "abs"],
            rnd: Rnd::Rn,
        },
        "min" | "max" if tname == "f32" => {
            let n = (1..4).take_while(|&i| op_n(i).is_ok()).count();
            if let Some(m) = mods
                .iter()
                .find(|m| !matches!(**m, "ftz" | "NaN" | "xorsign" | "abs" | "f32"))
            {
                bail!(lw.err(format!("{base}.f32: modifier .{m} not modeled")));
            }
            let xorsign = has("xorsign");
            // `.abs` alone (three sources; ptxas: FMNMX3 |a|, |b|, |c|, probes/fmnmx3_abs).
            let abs3 = has("abs") && !xorsign && n == 3;
            if !abs3 && (xorsign != has("abs") || (xorsign && n != 2)) {
                bail!(lw.err(format!("{base}.f32: .xorsign.abs form not modeled")));
            }
            Op::Fmnmx {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                c: if n == 3 {
                    Some(lw.val(op_n(3)?)?)
                } else {
                    None
                },
                neg: [false; 3],
                abs: [abs3; 3],
                min: Pred {
                    reg: None,
                    neg: base == "max",
                },
                ftz: has("ftz"),
                nan: has("NaN"),
                xorsign,
            }
        }
        // `not.b16` / `.b32` / `.b64`: x XOR all ones of the width.
        "not" if matches!(tname, "b16" | "b32" | "b64") => Op::Int {
            op: IntOp::Xor,
            bits,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            }),
        },
        // `add.u16x2` / `add.s16x2` (SASS VIADD.16x2; probes/viadd16x2): two 16-bit lanes,
        // each modulo 2^16. The .sat form (s16x2 only) is not modeled.
        "add" if matches!(mods, ["u16x2" | "s16x2"]) => Op::Int {
            op: IntOp::Add16x2,
            bits: 32,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        // `popc.b64` (ptxas: a POPC per half and an add; probes/popc64): popc of the low word,
        // plus popc of the high word through the scratch register.
        "popc" if mods == ["b64"] => {
            let tmp = lw.scratch_v()?;
            let d = lw.dst(op_n(0)?)?;
            let a = lw.val(op_n(1)?)?;
            pre.push(Op::Int {
                op: IntOp::ShrU,
                bits: 64,
                d: tmp,
                a: a.clone(),
                b: Val::Imm(32),
            });
            pre.push(Op::Int {
                op: IntOp::Popc,
                bits: 32,
                d: tmp,
                a: Val::Reg(tmp, 1),
                b: Val::Imm(0),
            });
            extra.push(Op::Int {
                op: IntOp::Add,
                bits: 32,
                d,
                a: Val::Reg(d, 1),
                b: Val::Reg(tmp, 1),
            });
            Op::Int {
                op: IntOp::Popc,
                bits: 32,
                d,
                a,
                b: Val::Imm(0),
            }
        }
        // `popc.b32` (SASS POPC; probes/int_misc).
        "popc" if mods == ["b32"] => Op::Int {
            op: IntOp::Popc,
            bits: 32,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(0),
        },
        // `bfind[.shiftamt].u32` (SASS FLO.U32[.SH]; probes/flo_popc): 0xffffffff for 0.
        // `.s32` / `.u64` are not modeled.
        "bfind" if mods == ["u32"] || mods == ["shiftamt", "u32"] => Op::Flo {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            sh: mods.len() == 2,
        },
        "bmsk" => Op::Int {
            op: if has("wrap") {
                IntOp::BmskWrap
            } else {
                IntOp::Bmsk
            },
            bits: 32,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        "dp2a" => Op::Dp2a {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            c: lw.val(op_n(3)?)?,
            hi: has("hi"),
            a_signed: mods.get(mods.len().wrapping_sub(2)) == Some(&"s32"),
            b_signed: mods.last() == Some(&"s32"),
        },
        // div.full.f32 (Triton's softmax / layer norm divide): ptxas's RCP expansion.
        "div" if matches!(mods, ["full" | "rn", "f32"]) => Op::DivFull {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            rn: has("rn"),
        },
        // `brev.b32 d, a` (SASS BREV; probes/ptx_misc2).
        "brev" if mods == ["b32"] => Op::Brev {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
        },
        // rcp.{rn,rz,rm,rp}.f64 d, a = div.<rnd>.f64 d, 1.0, a (probes/ptx_misc2).
        "rcp"
            if mods.len() == 2
                && mods[1] == "f64"
                && matches!(mods[0], "rn" | "rz" | "rm" | "rp") =>
        {
            Op::DivSqrt64 {
                d: lw.dst(op_n(0)?)?,
                a: Val::Imm(1f64.to_bits()),
                b: Some(lw.val(op_n(1)?)?),
                rnd: match mods[0] {
                    "rz" => Rnd::Rz,
                    "rm" => Rnd::Rm,
                    "rp" => Rnd::Rp,
                    _ => Rnd::Rn,
                },
            }
        }
        // sqrt.rn.f32 d, a: the IEEE square root (ptxas: MUFU.RSQ, a Newton step and a slow path),
        // as the f64 square root rounded to f32 (exact: 53 >= 2 * 24 + 2 bits, no double-
        // rounding error), through the scratch register; NaN results 0x7fffffff
        // (probes/sqrt_rn: GB300 = IEEE on 4096 patterns).
        "sqrt" if mods == ["rn", "f32"] => {
            let tmp = lw.scratch_v()?;
            pre.push(Op::F2F {
                d: tmp,
                a: lw.val(op_n(1)?)?,
                to64: true,
                rnd: Rnd::Rn,
            });
            pre.push(Op::DivSqrt64 {
                d: tmp,
                a: Val::Reg(tmp, 2),
                b: None,
                rnd: Rnd::Rn,
            });
            pre.push(Op::F2F {
                d: tmp,
                a: Val::Reg(tmp, 2),
                to64: false,
                rnd: Rnd::Rn,
            });
            // + (-0): the identity except on a NaN, which becomes the canonical 0x7fffffff
            // (GB300's result for negative and NaN inputs).
            Op::Fadd {
                d: lw.dst(op_n(0)?)?,
                a: Val::Reg(tmp, 1),
                b: Val::Imm(0x8000_0000),
                rnd: Rnd::Rn,
                neg: [false; 2],
                abs: [false; 2],
                ftz: false,
                sat: false,
            }
        }
        // div.rn.ftz.f32 / sqrt.rn.ftz.f32: the IEEE result of the operands flushed (subnormals
        // to a zero of their sign), the result flushed too (probes/ftz_div_sqrt). Flushes are
        // FADD.FTZ x, -RZ (also canonicalizing NaN, which the quotient / root does anyway).
        "div" | "sqrt" if mods == ["rn", "ftz", "f32"] => {
            let flush = |d: Reg, a: Val| Op::Fadd {
                d,
                a,
                b: Val::Imm(0x8000_0000),
                rnd: Rnd::Rn,
                neg: [false; 2],
                abs: [false; 2],
                ftz: true,
                sat: false,
            };
            let tmp = lw.scratch_v()?;
            let d = lw.dst(op_n(0)?)?;
            pre.push(flush(tmp, lw.val(op_n(1)?)?));
            if base == "div" {
                // d may alias a (already in tmp) or b.
                pre.push(flush(d, lw.val(op_n(2)?)?));
                pre.push(Op::DivFull {
                    d,
                    a: Val::Reg(tmp, 1),
                    b: Val::Reg(d, 1),
                    rn: true,
                });
            } else {
                pre.push(Op::F2F {
                    d: tmp,
                    a: Val::Reg(tmp, 1),
                    to64: true,
                    rnd: Rnd::Rn,
                });
                pre.push(Op::DivSqrt64 {
                    d: tmp,
                    a: Val::Reg(tmp, 2),
                    b: None,
                    rnd: Rnd::Rn,
                });
                pre.push(Op::F2F {
                    d,
                    a: Val::Reg(tmp, 2),
                    to64: false,
                    rnd: Rnd::Rn,
                });
            }
            flush(d, Val::Reg(d, 1))
        }
        // rcp.rn.f32 d, a: the IEEE 1 / a (ptxas: the div.rn.f32 expansion; probes/rcp_rn).
        "rcp" if mods == ["rn", "f32"] => Op::DivFull {
            d: lw.dst(op_n(0)?)?,
            a: Val::Imm(0x3f80_0000),
            b: lw.val(op_n(1)?)?,
            rn: true,
        },
        "div" | "rem" | "min" | "max" if !tname.starts_with('f') => {
            let s = tname.starts_with('s');
            let op = match (base, s) {
                ("div", false) => IntOp::DivU,
                ("div", true) => IntOp::DivS,
                ("rem", false) => IntOp::RemU,
                ("rem", true) => IntOp::RemS,
                ("min", false) => IntOp::MinU,
                ("min", true) => IntOp::MinS,
                ("max", false) => IntOp::MaxU,
                _ => IntOp::MaxS,
            };
            Op::Int {
                op,
                bits,
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
            }
        }
        "shfl" => {
            let mode = if has("idx") {
                ShflMode::Idx
            } else if has("up") {
                ShflMode::Up
            } else if has("down") {
                ShflMode::Down
            } else if has("bfly") {
                ShflMode::Bfly
            } else {
                bail!(lw.err("shfl mode"));
            };
            if !has("sync") {
                bail!(lw.err("only shfl.sync is modeled"));
            }
            let mask = lw.member_mask(op_n(4)?)?;
            // `d|p`: the in-range predicate.
            let (dd, pp) = match op_n(0)?.split_once('|') {
                Some((d, p)) => (d.trim(), Some(lw.pred_dst(p.trim())?)),
                None => (op_n(0)?, None),
            };
            Op::Shfl {
                p: pp,
                d: lw.dst(dd)?,
                a: lw.val(op_n(1)?)?,
                lane: lw.val(op_n(2)?)?,
                clamp: lw.val(op_n(3)?)?,
                mode,
                mask,
            }
        }
        // `ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16 {d..}, [a]` (SASS LDSM).
        // `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0..3}, {a0..3}, {b0, b1},
        // {c0..3}` / `m16n8k8.row.col.f32.tf32.tf32.f32` (SASS HMMA.16816.F32.BF16 /
        // HMMA.1688.F32.TF32, `Op::Hmma`; probes/mma_sync_ptx).
        "mma" => {
            // (tf32, A / B registers); m16n8k8 bf16: HMMA.1688.F32.BF16 (probes/bmma8).
            let f16 = mods
                == [
                    "sync", "aligned", "m16n8k16", "row", "col", "f32", "f16", "f16", "f32",
                ];
            let (tf32, s8, na, nb) = match mods {
                [
                    "sync",
                    "aligned",
                    "m16n8k16",
                    "row",
                    "col",
                    "f32",
                    "bf16",
                    "bf16",
                    "f32",
                ] => (false, false, 4, 2),
                // HMMA.16816.F32 (probes/hmma_f16).
                _ if f16 => (false, false, 4, 2),
                [
                    "sync",
                    "aligned",
                    "m16n8k8",
                    "row",
                    "col",
                    "f32",
                    "bf16",
                    "bf16",
                    "f32",
                ] => (false, false, 2, 1),
                [
                    "sync",
                    "aligned",
                    "m16n8k8",
                    "row",
                    "col",
                    "f32",
                    "tf32",
                    "tf32",
                    "f32",
                ] => (true, false, 4, 2),
                // IMMA.16832.S8.S8.SAT (probes/imma_s8).
                [
                    "sync",
                    "aligned",
                    "m16n8k32",
                    "row",
                    "col",
                    "satfinite",
                    "s32",
                    "s8",
                    "s8",
                    "s32",
                ] => (false, true, 4, 2),
                _ => bail!(lw.err(
                    "only mma.sync.aligned.{m16n8k16,m16n8k8}.row.col.f32.bf16.bf16.f32 / \
                     m16n8k16.row.col.f32.f16.f16.f32 / m16n8k8.row.col.f32.tf32.tf32.f32 / \
                     m16n8k32.row.col.satfinite.s32.s8.s8.s32 are modeled"
                )),
            };
            let list = |i: usize, n: usize| -> Result<Vec<Val>> {
                let v: Vec<Val> = op_n(i)?
                    .trim()
                    .trim_start_matches('{')
                    .trim_end_matches('}')
                    .split(',')
                    .map(|r| match lw.val(r)? {
                        v @ Val::Reg(..) => Ok(v),
                        _ => bail!(lw.err("mma operand must be a register")),
                    })
                    .collect::<Result<_>>()?;
                if v.len() != n {
                    bail!(lw.err("mma: unexpected fragment size"));
                }
                Ok(v)
            };
            let (a, b, c) = (list(1, na)?, list(2, nb)?, list(3, 4)?);
            let (d, n, moves) = lw.reg_vec_mem(op_n(0)?, true)?;
            if n != 4 {
                bail!(lw.err("mma: unexpected fragment size"));
            }
            extra.extend(moves);
            Op::Hmma {
                d,
                a,
                b,
                c,
                tf32,
                s8,
                f16,
            }
        }
        // `movmatrix.sync.aligned.m8n8.trans.b16 d, a` (SASS MOVM.16.MT88; probes/movm).
        "movmatrix" => {
            if mods != ["sync", "aligned", "m8n8", "trans", "b16"] {
                bail!(lw.err("only movmatrix.sync.aligned.m8n8.trans.b16 is modeled"));
            }
            Op::MovMatrix {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
            }
        }
        // `ldmatrix.sync.aligned.m16n16.x{1,2}.trans.shared.b8` (FA4 / CuTe fp8): two registers
        // per matrix, as SASS LDSM.8.MT1616[.2] (probes/ldsm_b8). ASSUME(L-ROSETTA)
        "ldmatrix" if has("m16n16") => {
            if !(has("b8") && has("trans"))
                || mods.iter().any(|m| {
                    !matches!(
                        *m,
                        "sync" | "aligned" | "m16n16" | "x1" | "x2" | "trans" | "shared" | "b8"
                    )
                })
            {
                bail!(lw.err("only ldmatrix.sync.aligned.m16n16.x{1,2}.trans.shared.b8 is modeled of the m16n16 forms"));
            }
            let num: u8 = if has("x2") {
                2
            } else if has("x1") {
                1
            } else {
                bail!(lw.err("missing .xN"))
            };
            let (d, n, moves) = lw.reg_vec_mem(op_n(0)?, true)?;
            if n != 2 * num {
                bail!(lw.err("register count does not match .xN (two per matrix)"));
            }
            extra.extend(moves);
            Op::LdMatrix {
                d,
                n,
                addr: lw.addr(op_n(1)?)?,
                trans: true,
                b8: true,
            }
        }
        "ldmatrix" => {
            if !(has("m8n8") && has("b16"))
                || mods.iter().any(|m| {
                    !matches!(
                        *m,
                        "sync"
                            | "aligned"
                            | "m8n8"
                            | "x1"
                            | "x2"
                            | "x4"
                            | "trans"
                            | "shared"
                            | "b16"
                    )
                })
            {
                bail!(
                    lw.err(
                        "only ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16 is modeled"
                    )
                );
            }
            let num: u8 = mods
                .iter()
                .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse().ok()))
                .ok_or_else(|| lw.err("missing .xN"))?;
            let (d, n, moves) = lw.reg_vec_mem(op_n(0)?, true)?;
            if n != num {
                bail!(lw.err("register count does not match .xN"));
            }
            extra.extend(moves);
            Op::LdMatrix {
                d,
                n,
                addr: lw.addr(op_n(1)?)?,
                trans: has("trans"),
                b8: false,
            }
        }
        "stmatrix" => {
            // `.m16n8.trans.b8` as SASS STSM.8.MT168 (probes/stsm_b8).
            let b8 = has("m16n8") && has("b8") && has("trans");
            if !(has("m8n8") && has("b16")) && !b8 {
                bail!(lw.err("only stmatrix.m8n8.b16 and .m16n8.trans.b8 are modeled"));
            }
            let num: usize = mods
                .iter()
                .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse().ok()))
                .ok_or_else(|| lw.err("missing .xN"))?;
            let v: Vec<Val> = op_n(1)?
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .split(',')
                .map(|r| lw.val(r))
                .collect::<Result<_>>()?;
            if v.len() != num {
                bail!(lw.err("register count does not match .xN"));
            }
            Op::StMatrix {
                addr: lw.addr(op_n(0)?)?,
                v,
                trans: has("trans"),
                b8,
            }
        }
        // `lop3.b32 d, a, b, c, immLut` (PTX ISA: the LUT is indexed as F(0xF0, 0xCC, 0xAA)
        // of a, b, c, SASS LOP3.LUT's convention). The predicate-output form `d|p` and the
        // `.or` / `.and` BoolOp forms are refused.
        "lop3" if mods == ["b32"] => {
            if ops.len() != 5 || ops_src.contains('|') {
                bail!(lw.err("only lop3.b32 d, a, b, c, immLut is modeled"));
            }
            let lut = parse_int(op_n(4)?.trim())
                .filter(|&v| v <= 0xff)
                .ok_or_else(|| lw.err("lop3.b32: immLut must be an immediate 0..255"))?;
            Op::Lop3 {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                c: lw.val(op_n(3)?)?,
                lut: lut as u8,
                p: None,
                pq: None,
            }
        }
        "add" | "sub" | "and" | "or" | "xor" | "shl" | "shr" => {
            if has("sat") && (!matches!(base, "add" | "sub") || tname != "s32") {
                bail!(lw.err("integer saturation is modeled only for add/sub.sat.s32"));
            }
            let op = match base {
                "add" if has("sat") => IntOp::AddSatS,
                "sub" if has("sat") => IntOp::SubSatS,
                "add" => IntOp::Add,
                "sub" => IntOp::Sub,
                "and" => IntOp::And,
                "or" => IntOp::Or,
                "xor" => IntOp::Xor,
                "shl" => IntOp::Shl,
                _ if tname.starts_with('s') => IntOp::ShrS,
                _ => IntOp::ShrU,
            };
            Op::Int {
                op,
                bits,
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
            }
        }
        // `neg.f32 d, a` (no .ftz): what ptxas emits, `FADD d, -a, -RZ` (exact for every non-NaN
        // value, -0 + -0 = -0 included; a NaN gives the canonical NaN; fixtures/e_ptx_redux).
        "neg" if tname == "f32" && !has("ftz") => Op::Fadd {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(0x8000_0000),
            rnd: Rnd::Rn,
            neg: [true, false],
            abs: [false; 2],
            ftz: false,
            sat: false,
        },
        // `abs[.ftz].f32 d, a`: what ptxas emits, `FADD[.FTZ] d, |a|, -RZ` (a NaN gives the
        // canonical NaN; probes/abs_f32 = GB300).
        "abs" if tname == "f32" && mods.iter().all(|m| matches!(*m, "ftz" | "f32")) => Op::Fadd {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(0x8000_0000),
            rnd: Rnd::Rn,
            neg: [false, false],
            abs: [true, false],
            ftz: has("ftz"),
            sat: false,
        },
        // `abs.s32 d, a` (IABS; abs(INT_MIN) = INT_MIN).
        "abs" if mods == ["s32"] || mods == ["s64"] => Op::Int {
            op: IntOp::Abs,
            bits: if mods == ["s64"] { 64 } else { 32 },
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(0),
        },
        // `shf.{l,r}.{clamp,wrap}.b32 d, a, b, c`: funnel shift of b:a by c, clamped to 32 or
        // mod 32 (GB300: fixtures/s_shf). ASSUME(L-SHF)
        "shf" => {
            let wrap = match (has("clamp"), has("wrap")) {
                (true, false) => false,
                (false, true) => true,
                _ => bail!(lw.err("shf needs exactly one of .clamp / .wrap")),
            };
            if has("l") == has("r") || bits != 32 {
                bail!(lw.err("shf form not modeled"));
            }
            Op::Shf {
                d: lw.dst(op_n(0)?)?,
                lo: lw.val(op_n(1)?)?,
                hi: lw.val(op_n(2)?)?,
                sh: lw.val(op_n(3)?)?,
                right: has("r"),
                signed: false,
                high: has("l"),
                wide: false,
                wrap,
            }
        }
        // `neg.s32 d, a` = 0 - a.
        "neg" if tname.starts_with('s') => Op::Int {
            op: IntOp::Sub,
            bits,
            d: lw.dst(op_n(0)?)?,
            a: Val::Imm(0),
            b: lw.val(op_n(1)?)?,
        },
        "mul" if has("wide") => {
            if !matches!(mods, ["wide", "s16" | "u16" | "s32" | "u32"]) {
                bail!(lw.err("mul.wide is modeled only for 16- or 32-bit integer sources"));
            }
            Op::MulWide {
                bits,
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                signed: tname.starts_with('s'),
            }
        }
        "mul" if has("hi") => Op::Int {
            op: if tname.starts_with('s') {
                IntOp::MulHiS
            } else {
                IntOp::MulHiU
            },
            bits,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        "mad" if has("lo") && bits == 32 => Op::Imad {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            c: lw.val(op_n(3)?)?,
            neg_c: false,
            not_c: false,
            wide: false,
            hi: false,
            signed: false,
            cin: None,
            cout: None,
        },
        // `mad.lo.{s,u}64 d, a, b, c` (CuTeDSL's 64-bit address arithmetic): d = a*b + c mod
        // 2^64, through a scratch register (d may alias c).
        "mad" if has("lo") && bits == 64 && !tname.starts_with('f') => {
            if !matches!(mods, ["lo", "s64" | "u64"]) {
                bail!(lw.err("mad: only mad.lo.{s64,u64} is modeled in 64 bits"));
            }
            let tmp = lw.scratch_v()?;
            extra.push(Op::Int {
                op: IntOp::Add,
                bits: 64,
                d: lw.dst(op_n(0)?)?,
                a: Val::Reg(tmp, 1),
                b: lw.val(op_n(3)?)?,
            });
            Op::Int {
                op: IntOp::Mul,
                bits: 64,
                d: tmp,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
            }
        }
        // `mad.wide.{s,u}16 d, a, imm, c`: d = ext16(a) * ext16(imm) + c in 32 bits (the 16x16
        // product is exact), a extended into the scratch register (probes/mad_wide16). A
        // register b is refused.
        "mad" if has("wide") && bits == 16 => {
            let signed = tname == "s16";
            let ext = |v: u64| {
                if signed {
                    v as u16 as i16 as i32 as u32
                } else {
                    v as u16 as u32
                }
            };
            let b = match lw.val(op_n(2)?)? {
                Val::Imm(v) => Val::Imm(ext(v) as u64),
                _ => bail!(lw.err("mad.wide.{s,u}16 with a register multiplier not modeled")),
            };
            let tmp = lw.scratch_v()?;
            pre.push(Op::CvtInt {
                d: tmp,
                a: lw.val(op_n(1)?)?,
                from_bits: 16,
                to_bits: 32,
                signed,
            });
            Op::Imad {
                d: lw.dst(op_n(0)?)?,
                a: Val::Reg(tmp, 1),
                b,
                c: lw.val(op_n(3)?)?,
                neg_c: false,
                not_c: false,
                wide: false,
                hi: false,
                signed: false,
                cin: None,
                cout: None,
            }
        }
        // `mad.wide.s32 d, a, b, c` (Triton's address arithmetic): 32x32 -> 64 plus a 64-bit c.
        "mad" if has("wide") && bits == 32 => Op::Imad {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            c: lw.val(op_n(3)?)?,
            neg_c: false,
            not_c: false,
            wide: true,
            hi: false,
            signed: tname.starts_with('s'),
            cin: None,
            cout: None,
        },
        "prefetch" => Op::Nop, // a cache hint
        // `prmt.b32 d, a, b, c` (default mode): SASS `PRMT d, a, c, b` (probes/prmt_ptx).
        "prmt" if mods == ["b32"] => Op::Prmt {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            sel: lw.val(op_n(3)?)?,
            b: lw.val(op_n(2)?)?,
        },
        // `createpolicy.fractional...b64 d[, f]`: a cache-policy descriptor. Its value matters
        // only to the cache; accepted when every use is the policy operand of an
        // `.L2::cache_hint` instruction (then a zero placeholder). ASSUME(L-PTX-HINT)
        "createpolicy" if has("fractional") && has("b64") => {
            let d = op_n(0)?.trim();
            if !lw.hint_policies.contains(d) {
                bail!(lw.err("createpolicy whose value is used as data not modeled"));
            }
            Op::Mov {
                d: lw.dst(d)?,
                a: Val::Imm(0),
            }
        }
        "mul" if has("lo") || !tname.starts_with('f') => Op::Int {
            op: IntOp::Mul,
            bits,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
        },
        // setp.<cmp>[.ftz].f32 p, a, b (one predicate, no combining operand) as SASS
        // `FSETP.<cmp>[.FTZ].AND p, PT, a, b, PT`: the ordered compares are false and the `u`
        // ones true when an operand is NaN.
        // `setp.<cmp>.f16 / .bf16 p, a, b` and `.f16x2 / .bf16x2 p|q, a, b` (SASS HSETP2
        // [.BF16_V2], probes/hsetp2): per lane the exact f32 widening compared; no `.ftz`, no
        // combining predicate.
        // `set.<cmp>.{u32,f16x2,bf16x2}.{f16x2,bf16x2} d, a, b` (SASS HSET2[.BF16_V2][.BF]):
        // per half 0xffff / 0 (u32) or 1.0 / 0 in the operand format (probes/hset2).
        "set" if matches!(ht, "f16x2" | "bf16x2") && mods.len() == 3 && ops.len() == 3 => {
            let cmp = match mods[0] {
                "eq" => FCmp::Eq,
                "ne" => FCmp::Ne,
                "lt" => FCmp::Lt,
                "le" => FCmp::Le,
                "gt" => FCmp::Gt,
                "ge" => FCmp::Ge,
                "equ" => FCmp::Equ,
                "neu" => FCmp::Neu,
                "ltu" => FCmp::Ltu,
                "leu" => FCmp::Leu,
                "gtu" => FCmp::Gtu,
                "geu" => FCmp::Geu,
                "num" => FCmp::Num,
                "nan" => FCmp::Nan,
                c => bail!(lw.err(format!("unsupported set compare {c}"))),
            };
            let bf = match mods[1] {
                "u32" => false,
                t if t == ht => true,
                _ => bail!(lw.err("set: destination type not modeled")),
            };
            let reg = |i: usize| -> Result<Val> {
                match lw.val(op_n(i)?)? {
                    v @ Val::Reg(..) => Ok(v),
                    _ => bail!(lw.err("set half immediate operand not modeled")),
                }
            };
            Op::Hset2 {
                d: lw.dst(op_n(0)?)?,
                cmp,
                a: reg(1)?,
                b: reg(2)?,
                sel: [HSel::Pair; 2],
                neg: [false; 2],
                abs: [false; 2],
                c: Pred::TRUE,
                bop: BoolOp::And,
                f16: ht == "f16x2",
                bf,
            }
        }
        "setp" if matches!(ht, "f16" | "f16x2" | "bf16" | "bf16x2") => {
            let cmp = match mods[0] {
                "eq" => FCmp::Eq,
                "ne" => FCmp::Ne,
                "lt" => FCmp::Lt,
                "le" => FCmp::Le,
                "gt" => FCmp::Gt,
                "ge" => FCmp::Ge,
                "equ" => FCmp::Equ,
                "neu" => FCmp::Neu,
                "ltu" => FCmp::Ltu,
                "leu" => FCmp::Leu,
                "gtu" => FCmp::Gtu,
                "geu" => FCmp::Geu,
                "num" => FCmp::Num,
                "nan" => FCmp::Nan,
                c => bail!(lw.err(format!("unsupported setp compare {c}"))),
            };
            if mods.len() != 2 || ops.len() != 3 {
                bail!(lw.err(format!(
                    "setp.{} form not modeled (no .ftz, no combining predicate)",
                    mods.join(".")
                )));
            }
            let x2 = ht.ends_with("x2");
            let (d, q) = match op_n(0)?.split_once('|') {
                Some((p, q)) if x2 => (lw.pred_dst(p.trim())?, Some(lw.pred_dst(q.trim())?)),
                None if !x2 => (lw.pred_dst(op_n(0)?)?, None),
                _ => bail!(lw.err("setp half destination form")),
            };
            let reg = |i: usize| -> Result<Val> {
                let v = lw.val(op_n(i)?)?;
                if !matches!(v, Val::Reg(..)) {
                    bail!(lw.err("setp half immediate operand not modeled"));
                }
                Ok(v)
            };
            let sel = if x2 { HSel::Pair } else { HSel::H0 };
            Op::Hsetp2Bf16 {
                d,
                q,
                cmp,
                a: reg(1)?,
                b: reg(2)?,
                sel: [sel; 2],
                neg: [false; 2],
                abs: [false; 2],
                c: Pred::TRUE,
                bop: BoolOp::And,
                f16: ht.starts_with("f16"),
            }
        }
        // setp.<cmp>.f32 (FSETP) / .f64 (DSETP: ptxas's f64 compare, no .ftz).
        "setp" if matches!(tname, "f32" | "f64") => {
            let cmp = match mods[0] {
                "eq" => FCmp::Eq,
                "ne" => FCmp::Ne,
                "lt" => FCmp::Lt,
                "le" => FCmp::Le,
                "gt" => FCmp::Gt,
                "ge" => FCmp::Ge,
                "equ" => FCmp::Equ,
                "neu" => FCmp::Neu,
                "ltu" => FCmp::Ltu,
                "leu" => FCmp::Leu,
                "gtu" => FCmp::Gtu,
                "geu" => FCmp::Geu,
                "num" => FCmp::Num,
                "nan" => FCmp::Nan,
                c => bail!(lw.err(format!("unsupported setp compare {c}"))),
            };
            if !matches!(mods.len(), 2 | 3)
                || mods.len() == 3 && (mods[1] != "ftz" || tname == "f64")
            {
                bail!(lw.err(format!("setp.{tname} modifiers not modeled")));
            }
            if op_n(0)?.contains('|') || op_n(3).is_ok() {
                bail!(lw.err(format!(
                    "setp.{tname} with p|q or a combining predicate not modeled"
                )));
            }
            let (d, a, b) = (lw.pred_dst(op_n(0)?)?, lw.val(op_n(1)?)?, lw.val(op_n(2)?)?);
            if tname == "f64" {
                Op::Dsetp {
                    d,
                    cmp,
                    a,
                    b,
                    neg: [false; 2],
                    abs: [false; 2],
                    c: Pred::TRUE,
                    bop: BoolOp::And,
                }
            } else {
                Op::Fsetp {
                    d,
                    cmp,
                    a,
                    b,
                    neg: [false; 2],
                    abs: [false; 2],
                    c: Pred::TRUE,
                    bop: BoolOp::And,
                    ftz: has("ftz"),
                }
            }
        }
        "setp" => {
            let s = tname.starts_with('s');
            // ptxas also takes the type before the comparison (`setp.u32.eq`, an inline-asm
            // spelling in CuTe DSL kernels).
            let c0 = match mods {
                [t, c, ..] if type_bits(t).is_some() && !t.starts_with('f') => *c,
                _ => mods[0],
            };
            let cmp = match (c0, s) {
                ("eq", _) => Cmp::Eq,
                ("ne", _) => Cmp::Ne,
                ("lt", false) | ("lo", _) => Cmp::LtU,
                ("le", false) | ("ls", _) => Cmp::LeU,
                ("gt", false) | ("hi", _) => Cmp::GtU,
                ("ge", false) | ("hs", _) => Cmp::GeU,
                ("lt", true) => Cmp::LtS,
                ("le", true) => Cmp::LeS,
                ("gt", true) => Cmp::GtS,
                ("ge", true) => Cmp::GeS,
                (c, _) => bail!(lw.err(format!("unsupported setp compare {c}"))),
            };
            if tname.starts_with('f') {
                bail!(lw.err("float setp not modeled yet"));
            }
            if op_n(0)?.contains('|') {
                bail!(lw.err("setp with p|q not modeled"));
            }
            // `setp.CmpOp.BoolOp d, a, b, c`: d = (a CmpOp b) BoolOp c.
            let comb = ["and", "or", "xor"].into_iter().find(|m| mods.contains(m));
            let (and, or, xor) = match comb {
                None => {
                    if op_n(3).is_ok() {
                        bail!(lw.err("setp with a fourth operand and no boolean operation"));
                    }
                    (Pred::TRUE, false, false)
                }
                Some(c) => (lw.pred_src(op_n(3)?)?, c == "or", c == "xor"),
            };
            Op::Setp {
                d: lw.pred_dst(op_n(0)?)?,
                cmp,
                bits,
                a: lw.val(op_n(1)?)?,
                b: lw.val(op_n(2)?)?,
                and,
                or,
                xor,
                ex: None,
            }
        }
        // `discard.global.L2 [a], 128` (SASS `CCTL.E.RML2 [Ra]`): the 128-byte line's contents
        // become indeterminate (PTX: a weak write of an unstable indeterminate value), modeled
        // as weak stores of a register nothing writes: later loads give undefined data
        // (S-INIT), and the write is race-checked. ASSUME(S-INIT)
        "discard" if mods == ["global", "L2"] => {
            if parse_int(op_n(1)?.trim()) != Some(128) {
                bail!(lw.err("discard size must be 128"));
            }
            let a = lw.addr(op_n(0)?)?;
            let u = lw
                .undef_v
                .ok_or_else(|| lw.err("internal: no undefined register"))?;
            let st = |i: i64| Op::St {
                space: Space::Global,
                bytes: 8,
                addr: Addr {
                    off: a.off + 8 * i,
                    ..a.clone()
                },
                v: Val::Reg(u, 1),
                sync: None,
            };
            for i in 0..15 {
                lw.pre.push(st(i));
            }
            st(15)
        }
        // `copysign.f32 d, a, b`: b with the sign bit of a (ptxas: LOP3.LUT d, a, 0x80000000, b,
        // 0xb8 or equivalent; a pure bit operation, NaNs included: probes/copysign).
        "copysign" if mods == ["f32"] => Op::Lop3 {
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: lw.val(op_n(2)?)?,
            c: Val::Imm(0x8000_0000),
            // (a & c) | (b & !c)
            lut: 0xe4,
            p: None,
            pq: None,
        },
        // `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
        // [g], [s], 128`: warp-collective copy of the 128-byte map to global memory, lane l
        // word l (ptxas: `LDS` + `ATOMG.E.EXCH.STRONG.GPU` per lane), as release stores; the
        // reader's fence.proxy.tensormap::generic.acquire then sees it (S-TMAP).
        "tensormap"
            if mods
                == [
                    "cp_fenceproxy",
                    "global",
                    "shared::cta",
                    "tensormap::generic",
                    "release",
                    "gpu",
                    "sync",
                    "aligned",
                ] =>
        {
            if parse_int(op_n(2)?.trim()) != Some(128) {
                bail!(lw.err("tensormap.cp_fenceproxy size must be 128"));
            }
            let (g, sm) = (lw.addr(op_n(0)?)?, lw.addr(op_n(1)?)?);
            let x = lw.scratch_v()?;
            let w = Reg {
                file: File::V,
                idx: lw
                    .vec_v
                    .ok_or_else(|| lw.err("internal: no vector scratch"))?
                    .idx,
            };
            lw.pre.push(Op::Int {
                op: IntOp::Shl,
                bits: 32,
                d: x,
                a: Val::SReg(SReg::LaneId),
                b: Val::Imm(2),
            });
            if sm.base2.is_some() || g.base2.is_some() {
                bail!(lw.err("tensormap.cp_fenceproxy address form not modeled"));
            }
            lw.pre.push(Op::Ld {
                d: w,
                n: 1,
                space: Space::Shared,
                bytes: 4,
                addr: Addr {
                    base2: Some(Val::Reg(x, 1)),
                    ..sm
                },
                sync: None,
                sext: false,
            });
            // One collective copy: the warp's lanes leave it together, each having the whole
            // map's writes ordered before its own later accesses (as `bar.warp.sync -1`).
            extra.push(Op::WarpSync {
                mask: Val::Imm(0xffff_ffff),
                collective: false,
                all: false,
            });
            Op::St {
                space: Space::Global,
                bytes: 4,
                addr: Addr {
                    base2: Some(Val::Reg(x, 1)),
                    ..g
                },
                v: Val::Reg(w, 1),
                sync: Some(MemSync {
                    sem: Sem::Release,
                    scope: Scope::Gpu,
                }),
            }
        }
        // `tensormap.replace.tile.<field>.shared::cta.b1024.<b32|b64> [tm], [ord,] v`: as ptxas
        // 13.0 lowers it, a read-modify-write of the field's bits in the 128-byte map in shared
        // memory (LDS, LOP3 0xb8, STS; probes/tmap_replace). Layout (byte offsets): 0x00
        // global_address (64 bits); 0x08: rank bits 4..6, elemtype 7..10 (+ bit 16 for 11 / 12),
        // interleave 11..12, swizzle 13..14 (mode 4: bit 22), fill 15; 0x0c + 4 i: global_stride
        // [i] >> 4 (low 32 bits), 0x1c bits 4 i..4 i + 3: its bits 36..39; 0x20 + 4 i:
        // global_dim[i] - 1; 0x34 bits 3 i..3 i + 2: element_stride[i] - 1; byte 0x37 + i:
        // box_dim[i] - 1. ASSUME(S-TMAP)
        "tensormap" if mods.first() == Some(&"replace") => {
            let field = mods.get(2).copied().unwrap_or("");
            if mods.get(1) != Some(&"tile")
                || mods.get(3) != Some(&"shared::cta")
                || mods.get(4) != Some(&"b1024")
                || mods.len() != 6
            {
                bail!(
                    lw.err("tensormap.replace form not modeled (only .tile .shared::cta .b1024)")
                );
            }
            let base = lw.addr(op_n(0)?)?;
            let at = |off: i64| Addr {
                base: base.base.clone(),
                base2: base.base2.clone(),
                off: base.off + off,
            };
            let indexed = matches!(
                field,
                "box_dim" | "global_dim" | "global_stride" | "element_stride"
            );
            let ord = if indexed {
                match parse_int(op_n(1)?.trim()) {
                    Some(o @ 0..=4) => o as i64,
                    _ => bail!(lw.err("tensormap.replace ordinal must be an immediate 0..4")),
                }
            } else {
                0
            };
            let vs = op_n(if indexed { 2 } else { 1 })?;
            let v = lw.val(vs)?;
            let imm = parse_int(vs.trim());
            let x = lw.scratch_v()?;
            let w = Reg {
                file: File::V,
                idx: lw
                    .vec_v
                    .ok_or_else(|| lw.err("internal: no vector scratch"))?
                    .idx,
            };
            let ld = |off: i64| Op::Ld {
                d: w,
                n: 1,
                space: Space::Shared,
                bytes: 4,
                addr: at(off),
                sync: None,
                sext: false,
            };
            let int = |op: IntOp, bits: u8, a: Val, b: u64| Op::Int {
                op,
                bits,
                d: x,
                a,
                b: Val::Imm(b),
            };
            // (old & !mask) | (new & mask) into word `off`; `new` already shifted into place.
            let merge = |lw: &mut Lower, off: i64, new: Val, mask: u32| {
                lw.pre.push(ld(off));
                lw.pre.push(Op::Lop3 {
                    d: w,
                    a: Val::Reg(w, 1),
                    b: Val::Imm(mask as u64),
                    c: new,
                    lut: 0xb8,
                    p: None,
                    pq: None,
                });
                Op::St {
                    space: Space::Shared,
                    bytes: 4,
                    addr: at(off),
                    v: Val::Reg(w, 1),
                    sync: None,
                }
            };
            let xv = Val::Reg(x, 1);
            let enum_v = |max: u64| -> Result<u64> {
                match imm {
                    Some(i) if i <= max => Ok(i),
                    _ => bail!(lw.err(format!(
                        "tensormap.replace .{field} needs an immediate 0..{max}"
                    ))),
                }
            };
            match (field, mods[5]) {
                ("global_address", "b64") => Op::St {
                    space: Space::Shared,
                    bytes: 8,
                    addr: at(0),
                    v,
                    sync: None,
                },
                ("rank", "b32") => {
                    lw.pre.push(int(IntOp::Shl, 32, v, 4));
                    merge(lw, 8, xv, 0x70)
                }
                ("elemtype", "b32") => {
                    let e = enum_v(15)?;
                    let (code, hi) = match e {
                        0..=10 => (e, false),
                        11 => (7, true),
                        12 => (8, true),
                        _ => (e - 2, false),
                    };
                    if hi {
                        lw.pre.push(ld(8));
                        lw.pre.push(Op::Int {
                            op: IntOp::Or,
                            bits: 32,
                            d: w,
                            a: Val::Reg(w, 1),
                            b: Val::Imm(0x1_0000),
                        });
                        lw.pre.push(Op::St {
                            space: Space::Shared,
                            bytes: 4,
                            addr: at(8),
                            v: Val::Reg(w, 1),
                            sync: None,
                        });
                    }
                    merge(lw, 8, Val::Imm(code << 7), 0x780)
                }
                ("interleave_layout", "b32") => {
                    let e = enum_v(2)?;
                    merge(lw, 8, Val::Imm(e << 11), 0x1800)
                }
                ("swizzle_mode", "b32") => {
                    let e = enum_v(4)?;
                    let bits = if e == 4 { 0x40_0000 } else { e << 13 };
                    merge(lw, 8, Val::Imm(bits), 0x40_6000)
                }
                ("fill_mode", "b32") => {
                    let e = enum_v(1)?;
                    merge(lw, 8, Val::Imm(e << 15), 0x8000)
                }
                ("global_dim", "b32") => {
                    lw.pre.push(int(IntOp::Add, 32, v, 0xffff_ffff));
                    Op::St {
                        space: Space::Shared,
                        bytes: 4,
                        addr: at(0x20 + 4 * ord),
                        v: xv,
                        sync: None,
                    }
                }
                ("box_dim", "b32") | ("element_stride", "b32") => {
                    let (off, sh, mask) = if field == "box_dim" {
                        let b = 0x37 + ord;
                        (b & !3, 8 * (b & 3) as u64, 0xffu32 << (8 * (b & 3)))
                    } else {
                        (0x34, 3 * ord as u64, 7u32 << (3 * ord))
                    };
                    lw.pre.push(int(IntOp::Add, 32, v, 0xffff_ffff));
                    lw.pre.push(int(IntOp::Shl, 32, xv.clone(), sh));
                    merge(lw, off, xv, mask)
                }
                ("global_stride", "b64") => {
                    lw.pre.push(int(IntOp::ShrU, 64, v.clone(), 4));
                    lw.pre.push(Op::St {
                        space: Space::Shared,
                        bytes: 4,
                        addr: at(0xc + 4 * ord),
                        v: xv.clone(),
                        sync: None,
                    });
                    lw.pre.push(int(IntOp::ShrU, 64, v, 36));
                    lw.pre.push(int(IntOp::Shl, 32, xv.clone(), 4 * ord as u64));
                    merge(lw, 0x1c, xv, 0xfu32 << (4 * ord))
                }
                _ => bail!(lw.err(format!(
                    "tensormap.replace .{field}.{} not modeled",
                    mods[5]
                ))),
            }
        }
        // `brx.idx[.uni] i, $L`: jump to entry i of the `.branchtargets` list labelled $L
        // (an index outside it is an error: PTX leaves it undefined). Per thread, as `bra`.
        "brx" if has("idx") && mods.iter().all(|m| matches!(*m, "idx" | "uni")) => {
            let l = op_n(1)?.trim();
            let names = lw
                .brx_tables
                .get(l)
                .ok_or_else(|| lw.err(format!("brx.idx: no .branchtargets labelled {l}")))?;
            let mut targets = Vec::with_capacity(names.len());
            for n in names {
                targets.push(
                    *lw.labels
                        .get(n.as_str())
                        .ok_or_else(|| lw.err(format!("unknown label {n}")))?,
                );
            }
            Op::Brx {
                base: lw.val(op_n(0)?)?,
                off: 0,
                targets,
            }
        }
        "bra" => {
            let l = op_n(0)?;
            let target = *lw
                .labels
                .get(l)
                .ok_or_else(|| lw.err(format!("unknown label {l}")))?;
            Op::Bra {
                target,
                any: false,
                tid0_falls: false,
                cond: None,
                div: None,
            }
        }
        "redux" => {
            // redux.sync.{add,min,max}.{u32,s32} / .{and,or,xor}.b32 /
            // .{min,max}[.abs][.NaN].f32 d, a, membermask (full warp only): the SASS
            // REDUX / CREDUX forms ptxas emits for them (mslk decode: redux.sync.max.NaN.f32 is
            // CREDUX.MAX.F32.NAN), with every member lane receiving the result.
            // ASSUME(L-ROSETTA): fixtures/e_ptx_redux = GB300.
            // Any other membermask (a register, Triton / CUDA C++ `mov.b32 %r92, -1`, or a
            // partial immediate): the redux waits for the live lanes it names and reduces over
            // them; executing lanes that differ from the mask are reported (probes/redux_mask
            // = GB300). ASSUME(E-WARPCOLL)
            let mask = match parse_int(op_n(2)?) {
                _ if op_n(2)?.trim() == "-1" => None,
                Some(m) if m == u32::MAX as u64 => None,
                Some(m) if m <= u32::MAX as u64 && m != 0 => Some(Val::Imm(m)),
                None if lw.reg(op_n(2)?.trim()).is_ok_and(|r| r.file != File::P) => {
                    Some(lw.val(op_n(2)?)?)
                }
                _ => bail!(lw.err("redux.sync membermask form not modeled")),
            };
            let op = match mods {
                ["sync", "add", "u32" | "s32"] => ReduxOp::Sum,
                ["sync", "and", "b32"] => ReduxOp::And,
                ["sync", "or", "b32"] => ReduxOp::Or,
                ["sync", "xor", "b32"] => ReduxOp::Xor,
                ["sync", "min", "u32"] => ReduxOp::MinU,
                ["sync", "max", "u32"] => ReduxOp::MaxU,
                ["sync", "min", "s32"] => ReduxOp::MinS,
                ["sync", "max", "s32"] => ReduxOp::MaxS,
                ["sync", m @ ("min" | "max"), rest @ .., "f32"]
                    if matches!(rest, [] | ["abs"] | ["NaN"] | ["abs", "NaN"]) =>
                {
                    let (abs, nan) = (rest.contains(&"abs"), rest.contains(&"NaN"));
                    if *m == "min" {
                        ReduxOp::MinF { abs, nan }
                    } else {
                        ReduxOp::MaxF { abs, nan }
                    }
                }
                _ => bail!(lw.err("redux.sync form not modeled")),
            };
            Op::Redux {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                op,
                mask,
            }
        }
        // `match.any.sync.b32` / `.b64 d, a, membermask` (SASS MATCH.ANY[.U64]): full warp only,
        // as vote (probes/match_any).
        "match" if matches!(mods, ["any", "sync", "b32" | "b64"]) => {
            if parse_int(op_n(2)?) != Some(u32::MAX as u64) && parse_int(op_n(2)?) != Some(u64::MAX)
            {
                bail!(lw.err("match.any.sync with a partial membermask not modeled"));
            }
            Op::MatchAny {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                wide: has("b64"),
            }
        }
        "vote" => {
            // vote.sync.{all,any,uni}.pred d, a, membermask / vote.sync.ballot.b32 d, a,
            // membermask (full warp only; fixtures/l1_coreader's warp-level poll, DeepEP
            // barrier_block's `VOTE.ALL`, CuTeDSL's `vote.sync.any.pred`). any = !all(!a).
            if !matches!(
                mods,
                ["sync", "all" | "any" | "uni", "pred"] | ["sync", "ballot", "b32"]
            ) {
                bail!(
                    lw.err(
                        "only vote.sync.{all,any,uni}.pred and vote.sync.ballot.b32 are modeled"
                    )
                );
            }
            let mask = lw.member_mask(op_n(2)?)?;
            let a = lw.pred_src(op_n(1)?)?;
            let not_a = Pred {
                reg: a.reg,
                neg: !a.neg,
            };
            if has("ballot") {
                Op::VoteAny {
                    d: lw.dst(op_n(0)?)?,
                    a,
                    mask,
                }
            } else {
                let d = lw.pred_dst(op_n(0)?)?;
                let pd = Pred {
                    reg: Some(d),
                    neg: false,
                };
                if has("all") || has("uni") {
                    Op::VoteAll {
                        d,
                        a,
                        uni: has("uni"),
                        mask,
                    }
                } else if has("any") {
                    extra.push(Op::Plop3 {
                        d,
                        a: pd,
                        b: Pred::TRUE,
                        c: Pred::TRUE,
                        lut: 0x0f,
                        d2: None,
                    });
                    Op::VoteAll {
                        d,
                        a: not_a,
                        uni: false,
                        mask,
                    }
                } else {
                    bail!(lw.err("vote.sync mode not modeled"));
                }
            }
        }
        "elect" => {
            // elect.sync d|p, membermask. Only the full-warp mask is modeled (the leader is
            // chosen among the active threads executing it); a register mask is checked when
            // executed (E-ELECT).
            let mask = match op_n(1)? {
                "0xffffffff" | "-1" => None,
                m if m.starts_with('%') => Some(lw.val(m)?),
                _ => bail!(lw.err("elect.sync with a partial membermask not modeled")),
            };
            let (d, p) = op_n(0)?
                .split_once('|')
                .ok_or_else(|| lw.err("elect.sync wants d|p"))?;
            let lane = if d.trim() == "_" {
                None
            } else {
                Some(lw.dst(d.trim())?)
            };
            Op::Elect {
                d: lw.pred_dst(p.trim())?,
                lane,
                mask,
            }
        }
        "ret" | "exit" => Op::Exit,
        // `trap` aborts the grid (as SASS BPT.TRAP; `__assertfail` lowers to a call PTX
        // keeps symbolic, so only explicit traps are seen here).
        "trap" => Op::Trap { why: String::new() },
        // A timing hint only: it cannot change which schedules are possible.
        // ASSUME(E-NANOSLEEP): see ASSUMPTIONS.md
        "nanosleep" => Op::Nop,
        // setmaxnreg.{inc,dec}.sync.aligned.u32 N (ptxas: USETMAXREG): only the register budget
        // changes. N must be an immediate in 24..=256, a multiple of 8. ASSUME(E-SETMAXNREG)
        "setmaxnreg" => {
            if !matches!(mods, ["inc" | "dec", "sync", "aligned", "u32"]) {
                bail!(lw.err(format!("setmaxnreg form .{} not modeled", mods.join("."))));
            }
            match parse_int(op_n(0)?) {
                Some(n) if (24..=256).contains(&n) && n % 8 == 0 && ops.len() == 1 => Op::Nop,
                _ => bail!(lw.err("setmaxnreg wants one immediate in 24..=256, a multiple of 8")),
            }
        }
        // `cvta.local.u64 d, a` (DeepEP: `%SP` from `%SPL`, only ever a vprintf argument
        // pointer): the generic address of the local window, as the SASS front's
        // `c[0x0][0x2f8]` (unmapped: a generic access through it is reported). ASSUME(S-CBANK)
        // The pre-pass's rewrites (vprintf_calls).
        "__nop" => Op::Nop,
        name @ ("__vprintf" | "__assertfail") => {
            const EXT: u64 = 0xffff_0000_0000_0000;
            let sym = if name == "__vprintf" {
                "vprintf"
            } else {
                "__assertfail"
            };
            Op::CallAbs {
                target: Val::Imm(EXT),
                syms: vec![(EXT, sym.to_string())],
            }
        }
        "cvta" if has("local") && !has("to") && has("u64") => Op::Int {
            op: IntOp::Add,
            bits: 64,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(LOCAL_GENERIC_BASE),
        },
        // `cvta.to.local.u64 d, a` (nvcc: the local address of a stack buffer whose generic
        // address `%SP + k` was taken, e.g. for a printf argument list): the inverse of the arm
        // above. A generic address outside the local window is undefined in PTX; here it
        // gives an offset of 2^32 or more, which ld / st .local reject. ASSUME(S-CBANK)
        "cvta" if has("local") && has("to") && has("u64") => Op::Int {
            op: IntOp::Sub,
            bits: 64,
            d: lw.dst(op_n(0)?)?,
            a: lw.val(op_n(1)?)?,
            b: Val::Imm(LOCAL_GENERIC_BASE),
        },
        "cvta" => {
            let to = has("to");
            let space = mods
                .iter()
                .find_map(|m| space_of(m))
                .ok_or_else(|| lw.err("cvta without a space"))?;
            Op::Cvta {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                space,
                to_generic: !to,
            }
        }
        // `cvt.rna.tf32.f32 d, a`: round to 19 bits, ties away (ptxas: a finite a gets + 0x1000,
        // then & 0xffffe000; Inf / NaN are only masked, so a NaN with no payload above bit 12
        // becomes Inf: probes/cvt_tf32). The finite test as integer ops: exponent != 0xff.
        // `cvt.rn / .rz [.relu] [.satfinite] .tf32.f32` (SASS F2FP [.SATFINITE] [.RELU]
        // .TF32.F32.PACK_B [.RZ]; probes/cvt_tf32_rn).
        "cvt"
            if mods.len() >= 3
                && mods[mods.len() - 2..] == ["tf32", "f32"]
                && matches!(mods[0], "rn" | "rz")
                && mods[1..mods.len() - 2]
                    .iter()
                    .all(|m| matches!(*m, "relu" | "satfinite")) =>
        {
            let mid = &mods[1..mods.len() - 2];
            if mid.len() > 1 && mid[0] == mid[1] {
                bail!(lw.err("cvt.tf32 modifier repeated"));
            }
            Op::CvtTf32 {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                rz: mods[0] == "rz",
                sat: mid.contains(&"satfinite"),
                relu: mid.contains(&"relu"),
            }
        }
        "cvt" if mods == ["rna", "tf32", "f32"] => {
            let a = lw.val(op_n(1)?)?;
            let d = lw.dst(op_n(0)?)?;
            let x = lw.scratch_v()?;
            let xv = Val::Reg(x, 1);
            let int = |op: IntOp, d: Reg, a: Val, b: Val| Op::Int {
                op,
                bits: 32,
                d,
                a,
                b,
            };
            for (i, (op, b)) in [
                (IntOp::ShrU, 23u64),
                (IntOp::And, 0xff),
                (IntOp::Add, 1),
                (IntOp::ShrU, 8),
                (IntOp::Xor, 1),
                (IntOp::Shl, 12),
            ]
            .into_iter()
            .enumerate()
            {
                let src = if i == 0 { a.clone() } else { xv.clone() };
                lw.pre.push(int(op, x, src, Val::Imm(b)));
            }
            lw.pre.push(int(IntOp::Add, d, a, xv));
            int(IntOp::And, d, Val::Reg(d, 1), Val::Imm(0xffff_e000))
        }
        "cvt" => {
            let n = mods.len();
            let (to, from) = (mods[n - 2], mods[n - 1]);
            let int = |t: &str| t.starts_with('u') || t.starts_with('s') || t.starts_with('b');
            if to == "f32" && from == "bf16" {
                Op::CvtF32Bf16 {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                }
            } else if to == "f32" && from == "f16" {
                // cvt.f32.f16 (SASS HADD2.F32 d, -RZ, a.H0_H0): exact.
                Op::CvtF32F16 {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    hi: false,
                }
            } else if to == "f16" && from == "f32" && has("rn") {
                if mods != ["rn", "f16", "f32"] && mods != ["rn", "sat", "f16", "f32"] {
                    bail!(lw.err("cvt.rn.f16.f32 modifier not modeled"));
                }
                let mut value = lw.val(op_n(1)?)?;
                if has("sat") {
                    let tmp = lw.scratch_v()?;
                    pre.push(Op::Fadd {
                        d: tmp,
                        a: value,
                        b: Val::Imm(0),
                        rnd: Rnd::Rn,
                        neg: [false; 2],
                        abs: [false; 2],
                        ftz: false,
                        sat: true,
                    });
                    value = Val::Reg(tmp, 1);
                }
                // cvt.rn.f16.f32 (SASS F2FP.F16.F32.PACK_AB Rd, RZ, Ra).
                Op::F2fpF16 {
                    d: lw.dst(op_n(0)?)?,
                    a: Val::Imm(0),
                    b: value,
                    rz: false,
                }
            } else if mods == ["rn", "bf16x2", "f32"]
                || mods == ["rn", "f16x2", "f32"]
                || mods == ["rz", "f16x2", "f32"]
            {
                // cvt.rn.{bf16x2,f16x2}.f32 d, a, b: d = {a (high half), b}, each rounded to
                // nearest even; ptxas emits F2FP.{BF16,F16}.F32.PACK_AB d, a, b
                // (fixtures/s_ptx_packed = GB300). cvt.rz.f16x2.f32: F2FP...PACK_AB.RZ
                // (probes/f2fp_rz, N-F2FP-RZ). .relu / .satfinite / other roundings are the
                // narrow-format path (cvt_lp) or refused. ASSUME(L-ROSETTA)
                let (d, a, b) = (lw.dst(op_n(0)?)?, lw.val(op_n(1)?)?, lw.val(op_n(2)?)?);
                if to == "bf16x2" {
                    Op::F2fpBf16 { d, a, b }
                } else {
                    Op::F2fpF16 {
                        d,
                        a,
                        b,
                        rz: has("rz"),
                    }
                }
            } else if to == "bf16" && from == "f32" && has("rn") {
                if mods != ["rn", "bf16", "f32"] {
                    bail!(lw.err("cvt.rn.bf16.f32 modifier not modeled"));
                }
                // cvt.rn.bf16.f32 (SASS F2FP.BF16.F32.PACK_AB Rd, RZ, Ra).
                Op::F2fpBf16 {
                    d: lw.dst(op_n(0)?)?,
                    a: Val::Imm(0),
                    b: lw.val(op_n(1)?)?,
                }
            } else if matches!(to, "f32" | "f64")
                && matches!(from, "s16" | "u16" | "s32" | "u32" | "s64" | "u64")
                && n <= 3
            {
                // cvt[.{rn,rz,rm,rp}].{f32,f64}.{s,u}{16,32,64} (SASS I2FP.F32.S32 / I2F[.F64]
                // [.S16|.U16|.S64|.U64][.RZ|.RM|.RP]; probes/cvt_i2f). ptxas requires the
                // rounding modifier.
                let bits: u8 = from[1..].parse().unwrap();
                let rnd = match mods[0] {
                    "rz" => Rnd::Rz,
                    "rm" => Rnd::Rm,
                    "rp" => Rnd::Rp,
                    "rn" => Rnd::Rn,
                    _ => bail!(lw.err("cvt to a float from an integer needs a rounding modifier")),
                };
                Op::I2F {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    signed: from.starts_with('s'),
                    rnd,
                    bits,
                    hi: false,
                    f64: to == "f64",
                }
            } else if from == "f32"
                && matches!(to, "s32" | "u32" | "s64" | "u64")
                && matches!(mods[0], "rzi" | "rni" | "rmi" | "rpi")
                && (n == 3 || n == 4 && mods[1] == "ftz")
            {
                // cvt.r{z,n,m,p}i[.ftz].{s,u}{32,64}.f32 (SASS F2I[.FTZ][.U32|.S64|.U64]
                // [.TRUNC|.FLOOR|.CEIL][.NTZ]; fixtures/s_f2i, probes/cvt_misc): round to an
                // integer, saturate, NaN 0.
                Op::F2I {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    signed: to.starts_with('s'),
                    bits: type_bits(to).unwrap(),
                    rnd: match mods[0] {
                        "rzi" => Rnd::Rz,
                        "rni" => Rnd::Rn,
                        "rmi" => Rnd::Rm,
                        _ => Rnd::Rp,
                    },
                    ftz: has("ftz"),
                    f64: false,
                }
            } else if matches!(mods, [_, "f32", "f64"] if matches!(mods[0], "rn" | "rz" | "rm" | "rp"))
                || mods == ["f64", "f32"]
            {
                // cvt.r{n,z,m,p}.f32.f64 / cvt.f64.f32 (SASS F2F.F32.F64[.RZ|.RM|.RP] /
                // F2F.F64.F32; probes/cvt_misc).
                Op::F2F {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    to64: to == "f64",
                    rnd: match mods[0] {
                        "rz" => Rnd::Rz,
                        "rm" => Rnd::Rm,
                        "rp" => Rnd::Rp,
                        _ => Rnd::Rn,
                    },
                }
            } else if mods == ["sat", "f32", "f32"] || mods == ["ftz", "sat", "f32", "f32"] {
                // cvt[.ftz].sat.f32.f32 (SASS FADD[.FTZ].SAT d, RZ, a; probes/cvt_misc,
                // probes/cvt_ftz): clamp to [0, 1], NaN 0, subnormals 0 under .ftz.
                Op::Fadd {
                    d: lw.dst(op_n(0)?)?,
                    a: Val::Imm(0),
                    b: lw.val(op_n(1)?)?,
                    rnd: Rnd::Rn,
                    neg: [false; 2],
                    abs: [false; 2],
                    ftz: has("ftz"),
                    sat: true,
                }
            } else if mods == ["ftz", "f32", "f32"] {
                // cvt.ftz.f32.f32 (SASS FADD.FTZ d, a, -RZ; probes/cvt_ftz): subnormals to
                // zero of their sign, -0 kept.
                Op::Fadd {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    b: Val::Imm(0x8000_0000),
                    rnd: Rnd::Rn,
                    neg: [false; 2],
                    abs: [false; 2],
                    ftz: true,
                    sat: false,
                }
            } else if from == "f64"
                && matches!(to, "s32" | "u32" | "s64" | "u64")
                && n == 3
                && matches!(mods[0], "rzi" | "rni" | "rmi" | "rpi")
            {
                // cvt.r{z,n,m,p}i.{s,u}{32,64}.f64 (SASS F2I[.S64|.U64|.U32].F64[.TRUNC|.FLOOR|
                // .CEIL]). ASSUME(N-F2I-F64)
                Op::F2I {
                    d: lw.dst(op_n(0)?)?,
                    a: lw.val(op_n(1)?)?,
                    signed: to.starts_with('s'),
                    bits: type_bits(to).unwrap(),
                    rnd: match mods[0] {
                        "rzi" => Rnd::Rz,
                        "rni" => Rnd::Rn,
                        "rmi" => Rnd::Rm,
                        _ => Rnd::Rp,
                    },
                    ftz: false,
                    f64: true,
                }
            } else if let Some(op) = cvt_lp(lw, mods, &ops)? {
                op
            } else if int(to) && int(from) && n == 2 {
                let (fb, tb) = (type_bits(from).unwrap(), type_bits(to).unwrap());
                if tb < fb && to.starts_with('s') {
                    // Narrowing to a signed type (`cvt.s8.s32 %rs, %r`): the low tb bits,
                    // sign-extended to the destination register (PTX ISA; ptxas I2F.S8 for
                    // a following cvt.rn.f32.s16; probes/cvt_s8_narrow).
                    Op::CvtInt {
                        d: lw.dst(op_n(0)?)?,
                        a: lw.val(op_n(1)?)?,
                        from_bits: tb,
                        to_bits: tb.max(32),
                        signed: true,
                    }
                } else {
                    Op::CvtInt {
                        d: lw.dst(op_n(0)?)?,
                        a: lw.val(op_n(1)?)?,
                        from_bits: fb,
                        to_bits: tb,
                        signed: from.starts_with('s'),
                    }
                }
            } else {
                bail!(lw.err("this cvt is not modeled yet"))
            }
        }
        "tcgen05" => {
            let sub = mods.first().copied().unwrap_or("");
            // ASSUME(E-PAIR): cta_group::2 alloc / dealloc / relinquish act on each CTA's own
            // TMEM (both CTAs of the pair issue them, the same allocation in each).
            let pair = has("cta_group::2");
            let cg1 = || -> Result<()> { Ok(()) };
            match sub {
                "alloc" => {
                    cg1()?;
                    Op::TmemAlloc {
                        dst: lw.addr(op_n(0)?)?,
                        ncols: lw.val(op_n(1)?)?,
                    }
                }
                "relinquish_alloc_permit" => {
                    cg1()?;
                    Op::TmemRelinquish
                }
                "dealloc" => {
                    cg1()?;
                    Op::TmemDealloc {
                        taddr: lw.val(op_n(0)?)?,
                        ncols: lw.val(op_n(1)?)?,
                    }
                }
                "mma" => {
                    cg1()?;
                    let vec = |x: &str| mods.contains(&x);
                    // kind::tf32 is UTCHMMA too: the instruction descriptor's A / B format
                    // (2) tells it from kind::f16 (N-MMA).
                    let kind = if has("kind::f16") || has("kind::tf32") {
                        MmaKind::F16
                    } else if has("kind::f8f6f4") {
                        MmaKind::F8f6f4
                    } else if has("kind::mxf8f6f4")
                        && !vec("scale_vec::2X")
                        && !vec("scale_vec::4X")
                    {
                        MmaKind::Mxf8f6f4
                    } else if has("kind::mxf4") && !vec("scale_vec::1X") && !vec("scale_vec::4X") {
                        MmaKind::Mxf4
                    } else if has("kind::mxf4nvf4") && !vec("scale_vec::1X") {
                        MmaKind::Mxf4nvf4 {
                            x4: vec("scale_vec::4X") || vec("block16"),
                        }
                    } else {
                        bail!(lw.err(
                            "tcgen05.mma kind not modeled (f16, tf32, f8f6f4, mxf8f6f4, mxf4, mxf4nvf4)"
                        ))
                    };
                    let scaled = !matches!(kind, MmaKind::F16 | MmaKind::F8f6f4);
                    if scaled != has("block_scale") || has("ws") || has("sp") {
                        bail!(lw.err("tcgen05.mma variant not modeled"));
                    }
                    // Operands (PTX ISA): [d], a, bdesc, idesc, ([scale_a], [scale_b],)
                    // ({disable-output-lane},) enable-input-d (, scale-input-d). The lane mask is a
                    // `{..}` vector; scale-input-d an immediate (kind::f16 / tf32 only).
                    let base = if scaled { 6 } else { 4 };
                    let has_mask = ops.get(base).is_some_and(|o| o.trim().starts_with('{'));
                    let en_i = base + has_mask as usize;
                    if !(ops.len() == en_i + 1 || ops.len() == en_i + 2) {
                        bail!(lw.err("tcgen05.mma: unexpected operand count"));
                    }
                    // D = A*B + D * 2^-scale_d (0..15; kind::f16 / tf32, the sim refuses other
                    // kinds and scaled values that are not normal or zero).
                    let scale_d: u8 = if ops.len() == en_i + 2 {
                        let sc = parse_int(op_n(en_i + 1)?.trim()).ok_or_else(|| {
                            lw.err("tcgen05.mma scale-input-d must be an immediate")
                        })?;
                        if scaled || !(0..=15).contains(&sc) {
                            bail!(lw.err("tcgen05.mma scale-input-d not allowed here"));
                        }
                        sc as u8
                    } else {
                        0
                    };
                    let scale = if scaled {
                        let t = |i: usize| -> Result<Val> {
                            let a = lw.addr(op_n(i)?)?;
                            if a.off != 0 {
                                bail!(lw.err("tcgen05.mma scale address offset not modeled"));
                            }
                            a.base
                                .ok_or_else(|| lw.err("tmem address must be a register"))
                        };
                        Some((t(4)?, t(5)?))
                    } else {
                        None
                    };
                    let disable: Vec<Val> = if has_mask {
                        op_n(base)?
                            .trim()
                            .trim_matches(|c| c == '{' || c == '}')
                            .split(',')
                            .map(|x| lw.val(x.trim()))
                            .collect::<Result<_>>()?
                    } else {
                        vec![]
                    };
                    // `[taddr + off]`: the offset adds to the TMEM address word (as SASS
                    // `tmem[URa+off]` and tcgen05.cp), computed into a vector scratch register
                    // (probes/tc_mma_off).
                    let mut tmem_addr = |s: &str, slot: u16| -> Result<Val> {
                        let ta = lw.addr(s)?;
                        let base = ta
                            .base
                            .ok_or_else(|| lw.err("tmem address must be a register"))?;
                        if ta.off == 0 {
                            return Ok(base);
                        }
                        let off = u32::try_from(ta.off)
                            .map_err(|_| lw.err("tcgen05.mma: negative TMEM address offset"))?;
                        let v = lw
                            .vec_v
                            .ok_or_else(|| lw.err("internal: no vector scratch registers"))?;
                        let tmp = Reg {
                            file: File::V,
                            idx: v.idx + slot,
                        };
                        pre.push(Op::Int {
                            op: IntOp::Add,
                            bits: 32,
                            d: tmp,
                            a: base,
                            b: Val::Imm(off as u64),
                        });
                        Ok(Val::Reg(tmp, 1))
                    };
                    let d = tmem_addr(op_n(0)?, 0)?;
                    // `[a]`: A in TMEM.
                    let a_tmem = op_n(1)?.trim().starts_with('[');
                    let adesc = if a_tmem {
                        tmem_addr(op_n(1)?, 1)?
                    } else {
                        lw.val(op_n(1)?)?
                    };
                    Op::Mma {
                        d,
                        adesc,
                        bdesc: lw.val(op_n(2)?)?,
                        idesc: lw.val(op_n(3)?)?,
                        // An immediate enable-input-d (CuTeDSL: `..., idesc, 0`): 0 overwrites
                        // D, 1 accumulates (probes/tc_mma_imm).
                        enable_d: match op_n(en_i)?.trim() {
                            "0" => Pred {
                                reg: None,
                                neg: true,
                            },
                            "1" => Pred::TRUE,
                            s => lw.pred_src(s)?,
                        },
                        pair,
                        disable,
                        a_tmem,
                        kind,
                        scale,
                        scale_d,
                    }
                }
                "cp" => {
                    // ASSUME(S-TCCP): see ASSUMPTIONS.md
                    if !has("32x128b")
                        || !has("warpx4")
                        || mods.iter().any(|m| m.starts_with("b8x16"))
                    {
                        bail!(lw.err(
                            "tcgen05.cp variant not modeled (32x128b.warpx4, no decompression)"
                        ));
                    }
                    let ta = lw.addr(op_n(0)?)?;
                    // `[taddr+off]`: the offset adds to the TMEM address word, as SASS
                    // `tmem[URa+off]`.
                    let off = u32::try_from(ta.off)
                        .map_err(|_| lw.err("tcgen05.cp: negative TMEM address offset"))?;
                    Op::TcCp {
                        taddr: ta
                            .base
                            .ok_or_else(|| lw.err("tmem address must be a register"))?,
                        off,
                        sdesc: lw.val(op_n(1)?)?,
                        pair,
                    }
                }
                "commit" => {
                    cg1()?;
                    let ma = lw.addr(op_n(0)?)?;
                    let a = ma
                        .base
                        .ok_or_else(|| lw.err("mbarrier address must be a register"))?;
                    let mask = if has("multicast::cluster") {
                        Some(lw.val(op_n(1)?)?)
                    } else {
                        None
                    };
                    Op::TcCommit {
                        mbar: a,
                        off: ma.off as u32,
                        mask,
                    }
                }
                "ld" => {
                    let x2 = has("16x32bx2");
                    if !has("32x32b") && !has("16x64b") && !has("16x128b") && !has("16x256b") && !x2
                    {
                        bail!(lw.err(
                            "only the 32x32b, 16x64b, 16x128b, 16x256b and 16x32bx2 tcgen05.ld shapes are modeled"
                        ));
                    }
                    // `tcgen05.ld.red...{min,max}[.abs][.NaN].f32 {d..}, red, [a][, split]`:
                    // also the min / max of the loaded values into `red` (SASS LDTM.STAT;
                    // probes/tmem_ldred, probes/tmem_ldred16 on both front-ends).
                    let is_red = has("red");
                    let red = if is_red {
                        if !has("f32") || has("min") == has("max") || !(has("32x32b") || x2) {
                            bail!(lw.err("tcgen05.ld.red form not modeled"));
                        }
                        let (abs, nan) = (has("abs"), has("NaN"));
                        let op = if has("min") {
                            ReduxOp::MinF { abs, nan }
                        } else {
                            ReduxOp::MaxF { abs, nan }
                        };
                        Some((lw.dst(op_n(1)?)?, op))
                    } else {
                        None
                    };
                    let ops: Vec<&str> = if is_red {
                        ops.iter()
                            .enumerate()
                            .filter(|(i, _)| *i != 1)
                            .map(|(_, o)| o.as_str())
                            .collect()
                    } else {
                        ops.iter().map(|o| o.as_str()).collect()
                    };
                    let op_n = |i: usize| -> Result<&str> {
                        ops.get(i)
                            .copied()
                            .ok_or_else(|| lw.err(format!("missing operand {i}")))
                    };
                    if x2 != (ops.len() == 3) || ops.len() > 3 {
                        bail!(lw.err("tcgen05.ld: unexpected operands"));
                    }
                    // Destination registers need not be consecutive (CuTe DSL lists them in
                    // descending order): the load writes each listed register.
                    let list: Vec<Reg> = op_n(0)?
                        .trim()
                        .trim_matches(|c| c == '{' || c == '}')
                        .split(',')
                        .map(|r| lw.reg(r.trim()))
                        .collect::<Result<_>>()?;
                    if list.is_empty() || list.len() > 255 || list.iter().any(|r| r.file != File::V)
                    {
                        bail!(lw.err("tcgen05.ld destinations must be 1..255 registers"));
                    }
                    let (d, n) = (list[0], list.len() as u8);
                    let consecutive = list
                        .iter()
                        .enumerate()
                        .all(|(i, r)| r.idx == d.idx + i as u16);
                    let regs = (!consecutive).then_some(list);
                    let num: u8 = mods
                        .iter()
                        .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse().ok()))
                        .ok_or_else(|| lw.err("missing .xN"))?;
                    let per = if has("16x256b") {
                        4
                    } else if has("16x128b") {
                        2
                    } else {
                        1
                    };
                    if num as u16 * per != n as u16 {
                        bail!(lw.err("register count does not match .xN"));
                    }
                    let a = lw.addr(op_n(1)?)?;
                    let taddr = a
                        .base
                        .ok_or_else(|| lw.err("tmem address must be a register"))?;
                    let ld = |half: Option<u8>, off: u32| Op::TmemLd {
                        d,
                        n,
                        taddr: taddr.clone(),
                        off,
                        tracked: true,
                        s16x64b: has("16x64b"),
                        half,
                        s16x256b: has("16x256b"),
                        s16x128b: has("16x128b"),
                        red,
                        regs: regs.clone(),
                    };
                    if x2 {
                        // `.16x32bx2 d, [a], split`: lanes 0..15 at a, lanes 16..31 at
                        // a + split columns (SASS LDTM.16dp32bit_t0_t15 + _t16_t31;
                        // probes/tmem_x2). ASSUME(S-TMEM)
                        let split = parse_int(op_n(2)?.trim())
                            .ok_or_else(|| lw.err("bad split offset"))?
                            as u32;
                        extra.push(ld(Some(1), (a.off as u32).wrapping_add(split)));
                        ld(Some(0), a.off as u32)
                    } else {
                        ld(None, a.off as u32)
                    }
                }
                "wait::ld" => Op::TmemWaitLd,
                "wait::st" => Op::TmemWaitSt,
                "st" => {
                    let x2 = has("16x32bx2");
                    if !has("32x32b") && !has("16x256b") && !has("16x128b") && !x2 {
                        bail!(lw.err(
                            "only the 32x32b / 16x256b / 16x128b / 16x32bx2 tcgen05.st shapes are modeled so far"
                        ));
                    }
                    if ops.len() != 2 + x2 as usize {
                        bail!(lw.err("tcgen05.st: unexpected operands"));
                    }
                    let num: usize = mods
                        .iter()
                        .find_map(|m| m.strip_prefix('x').and_then(|x| x.parse::<usize>().ok()))
                        .ok_or_else(|| lw.err("missing .xN"))?
                        * if has("16x256b") {
                            4
                        } else if has("16x128b") {
                            2
                        } else {
                            1
                        };
                    let v: Vec<Val> = op_n(1 + x2 as usize)?
                        .trim()
                        .trim_start_matches('{')
                        .trim_end_matches('}')
                        .split(',')
                        .map(|r| lw.val(r))
                        .collect::<Result<_>>()?;
                    if v.len() != num {
                        bail!(lw.err("register count does not match .xN"));
                    }
                    let a = lw.addr(op_n(0)?)?;
                    let taddr = a
                        .base
                        .ok_or_else(|| lw.err("tmem address must be a register"))?;
                    let st = |half: Option<u8>, off: u32| Op::TmemSt {
                        taddr: taddr.clone(),
                        off,
                        v: v.clone(),
                        tracked: true,
                        half,
                        s16x256b: has("16x256b"),
                        s16x128b: has("16x128b"),
                    };
                    if x2 {
                        // `.16x32bx2 [a], split, {v}`: as tcgen05.ld.16x32bx2.
                        // ASSUME(S-TMEM)
                        let split = parse_int(op_n(1)?.trim())
                            .ok_or_else(|| lw.err("bad split offset"))?
                            as u32;
                        extra.push(st(Some(1), (a.off as u32).wrapping_add(split)));
                        st(Some(0), a.off as u32)
                    } else {
                        st(None, a.off as u32)
                    }
                }
                "fence::before_thread_sync" => Op::Fence {
                    kind: FenceKind::TcBefore,
                },
                "fence::after_thread_sync" => Op::Fence {
                    kind: FenceKind::TcAfter,
                },
                _ => bail!(lw.err(format!("tcgen05.{sub} is not modeled yet"))),
            }
        }
        // st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [addr], v, [mbar]: a store
        // into (another CTA's) shared memory that completes 4 transaction bytes on the
        // mbarrier there (SASS STAS). Lowering it as a plain st dropped the complete_tx.
        "st" if has("async") => {
            let vn = mods
                .iter()
                .find_map(|m| m.strip_prefix('v').and_then(|x| x.parse::<u32>().ok()))
                .unwrap_or(1);
            let bytes = bits as u32 / 8 * vn;
            if !(has("mbarrier::complete_tx::bytes") && has("shared::cluster"))
                || !(bits == 32 || bits == 64)
                || !matches!(bytes, 4 | 8 | 16)
                || has("release")
                || has("mmio")
            {
                bail!(lw.err(
                    "only st.async.shared::cluster.mbarrier::complete_tx::bytes of 4, 8 or 16 bytes is modeled"
                ));
            }
            let reg = |s: &str| -> Result<Val> {
                let a = lw.addr(s)?;
                if a.off != 0 || a.base2.is_some() {
                    bail!(lw.err("st.async wants register addresses"));
                }
                a.base
                    .ok_or_else(|| lw.err("st.async wants register addresses"))
            };
            let v = if vn > 1 {
                let (r, k) = lw.reg_vec(op_n(1)?)?;
                if k as u32 != vn {
                    bail!(lw.err("vector length does not match .vN"));
                }
                Val::Reg(r, k)
            } else {
                lw.val(op_n(1)?)?
            };
            Op::StAsync {
                addr: reg(op_n(0)?)?,
                v,
                mbar: reg(op_n(2)?)?,
                bytes: bytes as u8,
            }
        }
        // `ld` / `st` `.local` (scalar, up to 32 bits) through a `.local` array's address: the
        // thread's local memory, as SASS LDL / STL. ASSUME(L-ROSETTA)
        "ld" | "st" if has("local") => {
            // .v2 / .v4: one access per element at its offset (probes/local_vec); a 64-bit
            // element is one virtual register (both words).
            let vn = mods
                .iter()
                .find_map(|m| m.strip_prefix('v').and_then(|n| n.parse::<u32>().ok()));
            if bits > 64 || vn.is_some_and(|n| !matches!(n, 2 | 4)) {
                bail!(lw.err("only ld / st .local of up to 64 bits, .v2 / .v4, are modeled"));
            }
            if let Some(n) = vn {
                if base == "ld" && tname.starts_with('s') && bits < 32 {
                    bail!(lw.err("sign-extending ld.local not modeled"));
                }
                let bytes = (bits / 8).max(1) as u32;
                let list = op_n(if base == "ld" { 0 } else { 1 })?
                    .trim()
                    .trim_matches(|c| c == '{' || c == '}')
                    .split(',')
                    .map(|x| x.trim().to_string())
                    .collect::<Vec<_>>();
                if list.len() != n as usize {
                    bail!(lw.err("vector length does not match .vN"));
                }
                let a = lw.addr(op_n(if base == "ld" { 1 } else { 0 })?)?;
                let mut ops = Vec::new();
                for (i, e) in list.iter().enumerate() {
                    let addr = Addr {
                        off: a.off + (i as u32 * bytes) as i64,
                        ..a.clone()
                    };
                    ops.push(if base == "ld" {
                        Op::LdLocal {
                            d: lw.dst(e)?,
                            bytes,
                            addr,
                        }
                    } else {
                        Op::StLocal {
                            bytes,
                            addr,
                            v: lw.val(e)?,
                        }
                    });
                }
                let last = ops.pop().unwrap();
                pre.extend(ops);
                last
            } else {
                if base == "ld" && tname.starts_with('s') && bits < 32 {
                    bail!(lw.err("sign-extending ld.local not modeled"));
                }
                if let Some(m) = mods
                    .iter()
                    .find(|m| type_bits(m).is_none() && **m != "local")
                {
                    bail!(lw.err(format!("{base}.local: modifier .{m} not modeled")));
                }
                let bytes = (bits / 8).max(1) as u32;
                if base == "ld" {
                    Op::LdLocal {
                        d: lw.dst(op_n(0)?)?,
                        bytes,
                        addr: lw.addr(op_n(1)?)?,
                    }
                } else {
                    Op::StLocal {
                        bytes,
                        addr: lw.addr(op_n(0)?)?,
                        v: lw.val(op_n(1)?)?,
                    }
                }
            }
        }
        // ASSUME(S-MULTIMEM): multimem.st / .red lower to a global St / Atom (as ptxas lowers
        // them to STG / REDG); the simulator sends them to every bound location when the
        // address is a multicast one. multimem.ld_reduce is its own op.
        "multimem" => {
            let kind = mods.first().copied().unwrap_or("");
            if !matches!(kind, "ld_reduce" | "st" | "red") {
                bail!(lw.err(format!("multimem.{kind} not modeled")));
            }
            let rest = &mods[1..];
            if let Some(m) = rest.iter().find(|m| {
                !matches!(
                    **m,
                    "weak"
                        | "relaxed"
                        | "acquire"
                        | "release"
                        | "cta"
                        | "cluster"
                        | "gpu"
                        | "sys"
                        | "global"
                        | "add"
                        | "min"
                        | "max"
                        | "and"
                        | "or"
                        | "xor"
                        | "acc::f32"
                        | "acc::f16"
                        | "v2"
                        | "v4"
                        | "v8"
                        | "b32"
                        | "b64"
                        | "u32"
                        | "u64"
                        | "s32"
                        | "s64"
                        | "f32"
                        | "f64"
                        | "f16"
                        | "f16x2"
                        | "bf16"
                        | "bf16x2"
                        | "e4m3"
                        | "e4m3x2"
                        | "e4m3x4"
                        | "e5m2"
                        | "e5m2x2"
                        | "e5m2x4"
                )
            }) {
                bail!(lw.err(format!("multimem.{kind}: modifier .{m} not modeled")));
            }
            let tyname = *rest.last().unwrap_or(&"");
            let (ety, ebytes): (McTy, u8) = match tyname {
                "u32" | "b32" => (McTy::U32, 4),
                "s32" => (McTy::S32, 4),
                "u64" | "b64" => (McTy::U64, 8),
                "s64" => (McTy::S64, 8),
                "f32" => (McTy::F32, 4),
                "f64" => (McTy::F64, 8),
                "f16" => (McTy::F16, 2),
                "f16x2" => (McTy::F16, 4),
                "bf16" => (McTy::Bf16, 2),
                "bf16x2" => (McTy::Bf16, 4),
                "e4m3" => (McTy::E4m3, 1),
                "e4m3x2" => (McTy::E4m3, 2),
                "e4m3x4" => (McTy::E4m3, 4),
                "e5m2" => (McTy::E5m2, 1),
                "e5m2x2" => (McTy::E5m2, 2),
                "e5m2x4" => (McTy::E5m2, 4),
                _ => bail!(lw.err("multimem: missing type")),
            };
            let vec_n = rest
                .iter()
                .find_map(|m| m.strip_prefix('v').and_then(|x| x.parse::<u8>().ok()))
                .unwrap_or(1);
            let bytes = ebytes * vec_n;
            if !matches!(bytes, 4 | 8 | 16) {
                bail!(lw.err("multimem: only 32 / 64 / 128-bit accesses are modeled"));
            }
            let has_r = |m: &str| rest.contains(&m);
            let sem = if has_r("acquire") {
                Some(Sem::Acquire)
            } else if has_r("release") {
                Some(Sem::Release)
            } else if has_r("relaxed") {
                Some(Sem::Relaxed)
            } else {
                None
            };
            let scope = scope_of(rest);
            // Defaults (PTX ISA): ld_reduce / st weak, red relaxed at .sys.
            let sync = match (kind, sem) {
                ("red", s) => Some(MemSync {
                    sem: s.unwrap_or(Sem::Relaxed),
                    scope: scope.unwrap_or(Scope::Sys),
                }),
                (_, None) => None,
                (_, Some(s)) => Some(MemSync {
                    sem: s,
                    scope: scope.unwrap_or(Scope::Gpu),
                }),
            };
            if let Some(m) = sync {
                if (kind == "ld_reduce" && m.releases()) || (kind != "ld_reduce" && m.acquires()) {
                    bail!(lw.err("invalid memory semantics for this multimem operation"));
                }
            }
            let op = if has_r("add") {
                McOp::Add
            } else if has_r("min") {
                McOp::Min
            } else if has_r("max") {
                McOp::Max
            } else if has_r("and") {
                McOp::And
            } else if has_r("or") {
                McOp::Or
            } else if has_r("xor") {
                McOp::Xor
            } else {
                McOp::Add
            };
            let regs = (bytes / 4).max(1);
            match kind {
                "ld_reduce" => {
                    let wide = ety.lane_bytes() == 8;
                    let (d, k, moves) = if wide {
                        (lw.dst(op_n(0)?)?, 1, vec![])
                    } else if regs > 1 {
                        lw.reg_vec_mem(op_n(0)?, true)?
                    } else {
                        (
                            lw.dst(op_n(0)?.trim_matches(|c| c == '{' || c == '}'))?,
                            1,
                            vec![],
                        )
                    };
                    if !wide && k != regs {
                        bail!(lw.err(
                            "multimem.ld_reduce: destination count does not match the access"
                        ));
                    }
                    let addr = lw.addr(op_n(1)?)?;
                    lw.extra = moves;
                    Op::LdReduce {
                        d,
                        n: k,
                        bytes,
                        addr,
                        op,
                        ty: ety,
                        acc: has_r("acc::f32") || has_r("acc::f16"),
                        sync,
                    }
                }
                "st" => {
                    let wide = ety.lane_bytes() == 8;
                    let addr = lw.addr(op_n(0)?)?;
                    let v = if !wide && regs > 1 {
                        let (r, k, moves) = lw.reg_vec_mem(op_n(1)?, false)?;
                        if k != regs {
                            bail!(lw.err("multimem.st: source count does not match the access"));
                        }
                        lw.pre = moves;
                        Val::Reg(r, k)
                    } else {
                        lw.val(op_n(1)?.trim_matches(|c| c == '{' || c == '}'))?
                    };
                    Op::St {
                        space: Space::Global,
                        bytes,
                        addr,
                        v,
                        sync,
                    }
                }
                _ => {
                    // multimem.red: the integer reductions and the scalar f32 add (as SASS
                    // REDG .F32.FTZ.RN) are modeled; vector / 16-bit float adds are not.
                    let aop = match (op, ety, vec_n) {
                        (McOp::Add, McTy::F32, 1) => AtomOp::FAddFtz,
                        (
                            _,
                            McTy::F32
                            | McTy::F64
                            | McTy::F16
                            | McTy::Bf16
                            | McTy::E4m3
                            | McTy::E5m2,
                            _,
                        ) => {
                            bail!(lw.err("multimem.red on this float type / vector not modeled"))
                        }
                        (McOp::Add, _, 1) => AtomOp::Add,
                        (McOp::And, _, 1) => AtomOp::And,
                        (McOp::Or, _, 1) => AtomOp::Or,
                        (McOp::Xor, _, 1) => AtomOp::Xor,
                        (McOp::Min, McTy::S32 | McTy::S64, 1) => AtomOp::MinS,
                        (McOp::Max, McTy::S32 | McTy::S64, 1) => AtomOp::MaxS,
                        (McOp::Min, _, 1) => AtomOp::MinU,
                        (McOp::Max, _, 1) => AtomOp::MaxU,
                        _ => bail!(lw.err("multimem.red with a vector not modeled")),
                    };
                    Op::Atom {
                        p: None,
                        op: aop,
                        space: Space::Global,
                        bytes,
                        addr: lw.addr(op_n(0)?)?,
                        v: lw.val(op_n(1)?)?,
                        d: None,
                        sync: sync.unwrap(),
                        new: None,
                    }
                }
            }
        }
        "ld" | "st" => {
            let space = mods
                .iter()
                .find_map(|m| space_of(m))
                .unwrap_or(Space::Generic);
            let bytes = (bits / 8).max(1);
            if bytes > 8 && base == "st" {
                bail!(lw.err("st.b128 not modeled yet"));
            }
            // Vectors: consecutive registers only (`{%r5, %r6, %r7, %r8}`), one element per
            // register.
            let vec_n = mods
                .iter()
                .find_map(|m| m.strip_prefix('v').and_then(|x| x.parse::<u8>().ok()));
            if let Some(n) = vec_n {
                // 64-bit elements: `.v2.b64` (one PTX register each), and the 32-byte
                // `.global.v4.b64` (SASS LDG / STG.E.ENL2.256; probes/ldg256_mask).
                if bits > 64 || bits == 64 && !(n == 2 || n == 4 && space == Space::Global) {
                    bail!(lw.err(
                        "only vector elements up to 32 bits, .v2.b64 and .global.v4.b64 are modeled"
                    ));
                }
                let sync = mem_sync(mods, false).map_err(|e| lw.err(e.to_string()))?;
                let opnd = if base == "ld" { op_n(0)? } else { op_n(1)? };
                let (r, k, moves) = lw.reg_vec_mem(opnd, base == "ld")?;
                if k != n {
                    bail!(lw.err("vector length does not match .vN"));
                }
                let bytes = (bits / 8).max(1) * n;
                let inst = Inst {
                    guard,
                    op: if base == "ld" {
                        Op::Ld {
                            d: r,
                            n,
                            space,
                            bytes,
                            addr: lw.addr(op_n(1)?)?,
                            sync,
                            sext: tname.starts_with('s') && bits < 32,
                        }
                    } else {
                        Op::St {
                            space,
                            bytes,
                            addr: lw.addr(op_n(0)?)?,
                            v: Val::Reg(r, n),
                            sync,
                        }
                    },
                    uniform: false,
                    loc: lw.line,
                    text: text.to_string(),
                };
                if base == "ld" {
                    lw.extra = moves;
                } else {
                    lw.pre = moves;
                }
                return Ok(inst);
            }
            let sync = mem_sync(mods, false).map_err(|e| lw.err(e.to_string()))?;
            if let Some(m) = sync {
                if (base == "ld" && m.releases()) || (base == "st" && m.acquires()) {
                    bail!(lw.err("invalid memory semantics for this access"));
                }
            }
            if base == "ld" {
                let d = lw.dst(op_n(0)?)?;
                // `ld.s32 %rd, [a]` (a 64-bit register): sign-extended to 64 bits (PTX ISA;
                // Triton's `ld.param.s32 %rd11` of an ignore index; probes/ld_sext64).
                if tname.starts_with('s') && bits < 64 && lw.wide_regs.contains(op_n(0)?.trim()) {
                    extra.push(Op::CvtInt {
                        d,
                        a: Val::Reg(d, 1),
                        from_bits: bits,
                        to_bits: 64,
                        signed: true,
                    });
                }
                Op::Ld {
                    d,
                    n: 1,
                    space,
                    bytes,
                    addr: lw.addr(op_n(1)?)?,
                    sync,
                    sext: tname.starts_with('s') && bits < 32,
                }
            } else {
                Op::St {
                    space,
                    bytes,
                    addr: lw.addr(op_n(0)?)?,
                    v: lw.val(op_n(1)?)?,
                    sync,
                }
            }
        }
        // `atom` / `red` `.add.noftz` `.bf16x2` / `.v2.bf16` / `.bf16` on global memory: per-lane
        // bf16 adds (SASS `ATOMG` / `REDG .E.ADD.BF16x2.RN`; the scalar form is a CAS loop with
        // `HADD2.BF16_V2`). `.v2.bf16` packs its two halves through the scratch register.
        // probes/bf16x2, both front-ends. ASSUME(N-BF16X2)
        "atom" | "red" if has("noftz") => {
            let bf = match ht {
                "bf16x2" => true,
                "bf16" => false,
                _ => bail!(lw.err(format!("{base}.noftz.{ht} not modeled"))),
            };
            if let Some(m) = mods.iter().find(|m| {
                !matches!(
                    **m,
                    "add"
                        | "noftz"
                        | "bf16"
                        | "bf16x2"
                        | "v2"
                        | "relaxed"
                        | "acquire"
                        | "release"
                        | "acq_rel"
                        | "cta"
                        | "cluster"
                        | "gpu"
                        | "sys"
                        | "global"
                )
            }) {
                bail!(lw.err(format!("{base}: modifier .{m} not modeled")));
            }
            if !has("add") || !has("global") || (has("v2") && bf) {
                bail!(lw.err(format!("{base}.noftz form not modeled")));
            }
            let pair = bf || has("v2");
            let sync = mem_sync(mods, true)
                .map_err(|e| lw.err(e.to_string()))?
                .unwrap();
            let (d, a, v) = if base == "atom" {
                (Some(op_n(0)?), op_n(1)?, op_n(2)?)
            } else {
                (None, op_n(0)?, op_n(1)?)
            };
            let halves = |x: &str| -> Option<Vec<String>> {
                let inner = x.trim().strip_prefix('{')?.strip_suffix('}')?;
                Some(inner.split(',').map(|h| h.trim().to_string()).collect())
            };
            let (v, d) = if has("v2") {
                let tmp = lw.scratch_v()?;
                let (Some(vh), dh) = (halves(v), d.map(halves)) else {
                    bail!(lw.err("v2.bf16 operands must be {a, b} vectors"));
                };
                if vh.len() != 2
                    || dh
                        .as_ref()
                        .is_some_and(|x| x.as_ref().is_none_or(|x| x.len() != 2))
                {
                    bail!(lw.err("v2.bf16 operands must be {a, b} vectors"));
                }
                lw.pre.push(Op::Prmt {
                    d: tmp,
                    a: lw.val(&vh[0])?,
                    sel: Val::Imm(0x5410),
                    b: lw.val(&vh[1])?,
                });
                if let Some(Some(dh)) = dh {
                    for (h, sel) in [(&dh[0], 0x4410), (&dh[1], 0x4432)] {
                        if h != "_" {
                            extra.push(Op::Prmt {
                                d: lw.dst(h)?,
                                a: Val::Reg(tmp, 1),
                                sel: Val::Imm(sel),
                                b: Val::Imm(0),
                            });
                        }
                    }
                    (Val::Reg(tmp, 1), Some(tmp))
                } else {
                    (Val::Reg(tmp, 1), None)
                }
            } else {
                (lw.val(v)?, d.map(|x| lw.dst(x)).transpose()?)
            };
            Op::Atom {
                p: None,
                op: if pair {
                    AtomOp::AddBf16x2
                } else {
                    AtomOp::AddBf16
                },
                space: Space::Global,
                bytes: if pair { 4 } else { 2 },
                addr: lw.addr(a)?,
                v,
                d,
                sync,
                new: None,
            }
        }
        // red.global.add.v2 / .v4.f32 [a], {..} (SASS REDG.E.ADD.F32x2 / F32x4.FTZ.RN) and
        // atom.global.add.v2 / .v4.f32 {..}, [a], {..} (ATOMG; Triton's tl.atomic_add on a
        // vectorized f32 tensor, `atom.global.gpu.acq_rel.add.v2.f32`): one FADD.FTZ atomic per
        // element, each on its own; an atom returns each element's old value (probes/red_vec,
        // probes/atom_vec).
        "atom" | "red"
            if (has("v2") || has("v4"))
                && has("add")
                && has("f32")
                && has("global")
                && mods.iter().all(|m| {
                    matches!(
                        *m,
                        "global"
                            | "add"
                            | "f32"
                            | "v2"
                            | "v4"
                            | "relaxed"
                            | "release"
                            | "cta"
                            | "cluster"
                            | "gpu"
                            | "sys"
                    ) || (base == "atom" && matches!(*m, "acquire" | "acq_rel"))
                }) =>
        {
            let n = if has("v4") { 4 } else { 2 };
            let ret = base == "atom";
            let (r, cnt, moves) = lw.reg_vec_mem(op_n(ret as usize + 1)?, false)?;
            if cnt != n {
                bail!(lw.err(format!("{base} vector operand count")));
            }
            pre.extend(moves);
            let d = if ret {
                // The old values come back through the same scratch registers as the sources
                // when the destinations are not consecutive: element k reads source k before
                // it writes destination k.
                let (dr, dcnt, back) = lw.reg_vec_mem(op_n(0)?, true)?;
                if dcnt != n {
                    bail!(lw.err("atom vector operand count"));
                }
                // Fail closed: a destination that overlaps a later source element.
                if back.is_empty()
                    && dr.file == r.file
                    && dr.idx != r.idx
                    && dr.idx < r.idx + n as u16
                    && r.idx < dr.idx + n as u16
                {
                    bail!(lw.err("atom vector destination overlapping its sources not modeled"));
                }
                extra.extend(back);
                Some(dr)
            } else {
                None
            };
            Op::Atom {
                p: None,
                op: AtomOp::FAddFtz,
                space: Space::Global,
                bytes: 4 * n,
                addr: lw.addr(op_n(ret as usize)?)?,
                v: Val::Reg(r, 1),
                d,
                sync: mem_sync(mods, true)
                    .map_err(|e| lw.err(e.to_string()))?
                    .unwrap(),
                new: None,
            }
        }
        "atom" | "red" => {
            // Fail closed: every modifier must be one modeled here (no `.noftz`, cache hints,
            // vectors or 16-bit types).
            if let Some(m) = mods.iter().find(|m| {
                !matches!(
                    **m,
                    "add"
                        | "or"
                        | "and"
                        | "xor"
                        | "exch"
                        | "cas"
                        | "min"
                        | "max"
                        | "inc"
                        | "b32"
                        | "b64"
                        | "u32"
                        | "u64"
                        | "s32"
                        | "s64"
                        | "f32"
                        | "f64"
                        | "relaxed"
                        | "acquire"
                        | "release"
                        | "acq_rel"
                        | "cta"
                        | "cluster"
                        | "gpu"
                        | "sys"
                        | "global"
                        | "shared"
                        | "shared::cta"
                        | "shared::cluster"
                        | "async"
                )
            }) {
                bail!(lw.err(format!("{base}: modifier .{m} not modeled")));
            }
            let space = mods
                .iter()
                .find_map(|m| space_of(m))
                .unwrap_or(Space::Generic);
            // `red.async.release.{gpu,sys}.global.add.{u32,s32,u64,s64} [a], b` (ptxas 13.0: a
            // per-lane ELECT loop of MEMBAR.ALL.CTA; UREDGR.E.ADD.NA.STRONG.<scope>; DEPBAR.LE
            // SB, 0, so the reduction has completed before the next instruction): a red of that
            // semantics and scope (probes/redasync_litmus: the release orders the thread's
            // earlier global stores on GB300). ASSUME(S-RED-ASYNC). The PTX ISA defines no other
            // `.global` form (`.relaxed` needs `.cluster` scope, `.shared::cluster` and an
            // mbarrier; the other operations are `.shared::cluster` only), and ptxas 13.0 - 13.2
            // accept them without a diagnostic and emit the same UREDGR.E.ADD (GB300 and GB200
            // add; probes/red_async_ops), so they are refused.
            if has("async")
                && (base != "red"
                    || space != Space::Global
                    || !has("add")
                    || !matches!(tname, "u32" | "s32" | "u64" | "s64")
                    || !(has("gpu") || has("sys"))
                    || !has("release"))
            {
                bail!(lw.err(
                    "red.async: only .release .gpu / .sys .global .add .u32 / .s32 / .u64 / .s64 \
                     is defined for .global (ptxas compiles the other forms to the release add)"
                ));
            }
            let op = if has("add") && tname == "f32" {
                AtomOp::FAddFtz
            } else if has("add") && tname == "f64" {
                AtomOp::FAdd64
            } else if has("add") {
                AtomOp::Add
            } else if has("or") {
                AtomOp::Or
            } else if has("and") {
                AtomOp::And
            } else if has("xor") {
                AtomOp::Xor
            } else if has("exch") && base == "atom" {
                AtomOp::Exch
            } else if has("cas") && base == "atom" {
                AtomOp::Cas
            } else if (has("inc") || has("dec")) && tname == "u32" {
                if has("inc") { AtomOp::Inc } else { AtomOp::Dec }
            } else if (has("min") || has("max")) && matches!(tname, "u32" | "s32" | "u64" | "s64") {
                // Integer min / max (CUTLASS / CuTeDSL `smem_fmax`: a float max through
                // `red.max.s32` for non-negative and `red.min.u32` for negative values).
                match (has("min"), tname.starts_with('s')) {
                    (true, false) => AtomOp::MinU,
                    (true, true) => AtomOp::MinS,
                    (false, false) => AtomOp::MaxU,
                    (false, true) => AtomOp::MaxS,
                }
            } else {
                bail!(lw.err("this atom/red operation is not modeled yet"));
            };
            if (tname.starts_with('f') && !matches!(op, AtomOp::FAddFtz | AtomOp::FAdd64))
                || (bits != 32 && bits != 64)
            {
                bail!(lw.err("only 32/64-bit integer atom/red are modeled"));
            }
            let sync = mem_sync(mods, true)
                .map_err(|e| lw.err(e.to_string()))?
                .unwrap();
            let (d, a, v) = if base == "atom" {
                (Some(lw.dst(op_n(0)?)?), op_n(1)?, op_n(2)?)
            } else {
                (None, op_n(0)?, op_n(1)?)
            };
            // `atom.cas d, [a], b, c`: compare b, new c.
            let new = if op == AtomOp::Cas {
                Some(lw.val(op_n(3)?)?)
            } else {
                None
            };
            Op::Atom {
                p: None,
                op,
                space,
                bytes: (bits / 8),
                addr: lw.addr(a)?,
                v: lw.val(v)?,
                d,
                sync,
                new,
            }
        }
        "barrier" if has("cluster") => {
            if has("arrive") {
                if has("relaxed") && has("release") {
                    bail!(lw.err("barrier.cluster.arrive cannot be both relaxed and release"));
                }
                Op::ClusterArrive {
                    release: !has("relaxed"),
                }
            } else if has("wait") {
                if has("relaxed") {
                    bail!(lw.err("barrier.cluster.wait.relaxed not modeled"));
                }
                Op::ClusterWait
            } else {
                bail!(lw.err("only barrier.cluster.arrive / wait are modeled"));
            }
        }
        "clusterlaunchcontrol" if has("try_cancel") => {
            // `.multicast::cluster::all`: every CTA of the cluster gets the response and the
            // completion at the same offsets (SASS UGETNEXTWORKID.BROADCAST, probes/clc_mcast).
            let multicast = mods.contains(&"multicast::cluster::all");
            if !(has("async") && has("shared::cta") && has("mbarrier::complete_tx::bytes"))
                || mods
                    .iter()
                    .any(|m| m.starts_with("multicast") && *m != "multicast::cluster::all")
            {
                bail!(
                    lw.err("only try_cancel to shared::cta [.multicast::cluster::all] is modeled")
                );
            }
            // `[reg+off]` operands (CuTe DSL: `[%r23+432]`): the offset adds into a vector
            // scratch register. Before 2026-11-09 the offset was dropped, so the response landed
            // `off` bytes early and the load after the wait read memory nothing wrote.
            let mut saddr = |s: &str, slot: u16| -> Result<Val> {
                let a = lw.addr(s)?;
                let base = a.base.ok_or_else(|| lw.err("register address"))?;
                if a.off == 0 {
                    return Ok(base);
                }
                let off = u32::try_from(a.off)
                    .map_err(|_| lw.err("try_cancel: negative address offset"))?;
                let v = lw
                    .vec_v
                    .ok_or_else(|| lw.err("internal: no vector scratch registers"))?;
                let tmp = Reg {
                    file: File::V,
                    idx: v.idx + slot,
                };
                pre.push(Op::Int {
                    op: IntOp::Add,
                    bits: 32,
                    d: tmp,
                    a: base,
                    b: Val::Imm(off as u64),
                });
                Ok(Val::Reg(tmp, 1))
            };
            let resp = saddr(op_n(0)?, 0)?;
            let mbar = saddr(op_n(1)?, 1)?;
            Op::ClcTryCancel {
                resp,
                mbar,
                multicast,
            }
        }
        // `get_first_ctaid.v4.b32.b128 {x, y, z, _}, q`: one query per named element (the
        // fourth is 0; ptxas: MOV / LOP3 / SHF of the response words, RZ).
        "clusterlaunchcontrol" if has("query_cancel") && has("get_first_ctaid") && has("v4") => {
            let q = lw.reg(op_n(1)?.trim())?;
            let elems: Vec<&str> = op_n(0)?
                .trim()
                .trim_matches(|c| c == '{' || c == '}')
                .split(',')
                .map(|x| x.trim())
                .collect();
            if elems.len() != 4 {
                bail!(lw.err("get_first_ctaid.v4 takes four destinations"));
            }
            let mut ops = Vec::new();
            for (k, e) in elems.iter().enumerate() {
                if *e == "_" {
                    continue;
                }
                ops.push(if k < 3 {
                    Op::ClcQuery {
                        d: lw.dst(e)?,
                        q,
                        ctaid: Some(k as u8),
                    }
                } else {
                    Op::Mov {
                        d: lw.dst(e)?,
                        a: Val::Imm(0),
                    }
                });
            }
            let last = ops
                .pop()
                .ok_or_else(|| lw.err("get_first_ctaid.v4 has no destination"))?;
            pre.extend(ops);
            last
        }
        "clusterlaunchcontrol" if has("query_cancel") => {
            // GB300 response (ptxas's decoding): word 0 ctaid.x, word 1 ctaid.y | ctaid.z << 16,
            // bit 0 of word 2 is_canceled (E-CLC).
            let ctaid = if has("is_canceled") {
                None
            } else if has("get_first_ctaid::x") {
                Some(0)
            } else if has("get_first_ctaid::y") {
                Some(1)
            } else if has("get_first_ctaid::z") {
                Some(2)
            } else {
                bail!(lw.err("this query_cancel form is not modeled"));
            };
            Op::ClcQuery {
                d: if ctaid.is_none() {
                    lw.pred_dst(op_n(0)?)?
                } else {
                    lw.dst(op_n(0)?)?
                },
                q: lw.reg(op_n(1)?.trim())?,
                ctaid,
            }
        }
        "mapa" => {
            // mapa.u64 without a space: a generic address in the shared window (S-SWIN).
            if !has("shared::cluster") && !(mods.len() == 1 && has("u64")) {
                bail!(lw.err("only mapa.shared::cluster / generic mapa.u64 are modeled"));
            }
            Op::Mapa {
                d: lw.dst(op_n(0)?)?,
                a: lw.val(op_n(1)?)?,
                rank: lw.val(op_n(2)?)?,
            }
        }
        // bar{.cta}.sync / barrier{.cta}.sync{.aligned} (also bare `bar` / `barrier.cta`), and
        // bar{.cta}.arrive / barrier{.cta}.arrive{.aligned}, which always takes a count.
        // `bar.warp.sync membermask` (WARPSYNC).
        "bar" if mods == ["warp", "sync"] => Op::WarpSync {
            mask: lw.val(op_n(0)?)?,
            collective: false,
            all: false,
        },
        // `bar.red.popc.u32 d, a{, b}, {!}c` / `bar.red.{and,or}.pred p, a{, b}, {!}c` (SASS
        // BAR.RED + B2R.RESULT; probes/bar_red). ASSUME(E-BAR-RED)
        "bar" | "barrier" if has("red") => {
            let known = |m: &&str| {
                matches!(
                    *m,
                    "red" | "cta" | "aligned" | "popc" | "and" | "or" | "u32" | "pred"
                )
            };
            let (op, popc) = match (
                has("popc") && has("u32"),
                has("and"),
                has("or"),
                has("pred"),
            ) {
                (true, false, false, false) => (BarRedOp::Popc, true),
                (false, true, false, true) => (BarRedOp::And, false),
                (false, false, true, true) => (BarRedOp::Or, false),
                _ => bail!(lw.err("bar.red form not modeled")),
            };
            if !mods.iter().all(known) {
                bail!(lw.err("bar.red modifier not modeled"));
            }
            let (count, c) = match ops.len() {
                3 => (None, op_n(2)?),
                4 => (Some(lw.val(op_n(2)?)?), op_n(3)?),
                _ => bail!(lw.err("bar.red operands")),
            };
            let d = op_n(0)?;
            extra.push(Op::BarResult {
                d: if popc { Some(lw.dst(d)?) } else { None },
                p: if popc { None } else { Some(lw.pred_dst(d)?) },
            });
            Op::BarSync {
                id: lw.val(op_n(1)?)?,
                count,
                arrive: false,
                packed: false,
                aligned: base == "bar" || has("aligned"),
                red: Some((op, lw.pred_src(c)?)),
            }
        }
        "bar" | "barrier" => {
            let arrive = has("arrive");
            let known = |m: &&str| matches!(*m, "sync" | "arrive" | "cta" | "aligned");
            if !mods.iter().all(known) || (arrive && has("sync")) {
                bail!(lw.err("only bar.sync / bar.arrive are modeled so far"));
            }
            let id = lw.val(op_n(0)?)?;
            let count = match ops.get(1) {
                Some(c) => Some(lw.val(c)?),
                None if arrive => bail!(lw.err("bar.arrive requires a thread count")),
                None => None,
            };
            // `bar.*` is `barrier.*.aligned`.
            let aligned = base == "bar" || has("aligned");
            Op::BarSync {
                id,
                count,
                arrive,
                packed: false,
                aligned,
                red: None,
            }
        }
        // ASSUME(E-PDL): see ASSUMPTIONS.md
        "griddepcontrol" => match mods.first().copied() {
            Some("wait") => Op::GridDepWait,
            Some("launch_dependents") => Op::GridDepLaunch,
            _ => bail!(lw.err("unknown griddepcontrol form")),
        },
        // `membar.cta` / `.gl` / `.sys` = `fence.sc.cta` / `.gpu` / `.sys` (PTX ISA; ptxas emits
        // the same MEMBAR.SC.<scope> [+ ERRBAR + CCTL.IVALL]), modeled as fence.sc is
        // (probes/fence_scopes). ASSUME(S-SCOPE)
        "membar" => Op::Fence {
            kind: FenceKind::Membar(match mods {
                ["cta"] => Scope::Cta,
                ["gl"] => Scope::Gpu,
                ["sys"] => Scope::Sys,
                _ => bail!(lw.err("this membar form is not modeled")),
            }),
        },
        "fence" => {
            if has("mbarrier_init") {
                Op::Fence {
                    kind: FenceKind::MbarInit,
                }
            } else if has("acq_rel") || has("sc") {
                // ASSUME(S-SCOPE): fence.sc is treated as fence.acq_rel (no SC order is
                // modeled beyond what the interleaving gives).
                // The scope is required (PTX ISA `fence{.sem}.scope`): no default that could be
                // stronger than what the kernel asked for.
                Op::Fence {
                    kind: FenceKind::Membar(
                        scope_of(mods).ok_or_else(|| lw.err("fence without a scope"))?,
                    ),
                }
            } else if mods.first() == Some(&"release") {
                // fence.release.<scope> (PTX 8.6; ptxas: MEMBAR at the scope, as SASS lifts it).
                Op::Fence {
                    kind: FenceKind::Release(scope_of(mods).unwrap_or(Scope::Sys)),
                }
            } else if mods.first() == Some(&"acquire") {
                // fence.acquire.sys (ptxas: CCTL.IVALL): FenceKind::Acquire. `.cta` / `.cluster`
                // / `.gpu` (ptxas: nothing / CCTL.IVALL / CCTL.IVALL): an acquire of that scope
                // only (probes/fence_scopes). ASSUME(S-SCOPE)
                if mods.len() != 2 {
                    bail!(lw.err("this fence.acquire form is not modeled"));
                }
                match scope_of(mods).ok_or_else(|| lw.err("fence.acquire without a scope"))? {
                    Scope::Sys => Op::Fence {
                        kind: FenceKind::Acquire,
                    },
                    sc => Op::Fence {
                        kind: FenceKind::AcquireAt(sc),
                    },
                }
            } else if mods.first() == Some(&"proxy") && has("async") {
                // `fence.proxy.async` with no state space orders all of them (ptxas: MEMBAR.ALL.GPU
                // + FENCE.VIEW.ASYNC.S); `.global` orders global memory only, which no proxy
                // check models, so it never covers a shared-memory check. ASSUME(S-PROXY)
                match &mods[1..] {
                    ["async"] | ["async", "shared::cta" | "shared::cluster"] => Op::Fence {
                        kind: FenceKind::ProxyAsync,
                    },
                    ["async", "global"] => Op::Fence {
                        kind: FenceKind::ProxyAsyncGlobal,
                    },
                    _ => bail!(lw.err("this fence.proxy form is not modeled")),
                }
            } else if mods.len() == 4
                && mods[..3] == ["proxy", "tensormap::generic", "acquire"]
                && matches!(mods[3], "cta" | "cluster" | "gpu" | "sys")
            {
                // fence.proxy.tensormap::generic.acquire.<scope> [a], 128 (ptxas: CCTL.E.C.LDCU
                // .IV.DEEP + UTMACCTL.IV [a], lifted as TmapFence on the SASS front). ASSUME(S-TMAP)
                if parse_int(op_n(1)?.trim()) != Some(128) {
                    bail!(lw.err("fence.proxy.tensormap size must be 128"));
                }
                Op::TmapFence {
                    addr: lw.addr(op_n(0)?)?,
                }
            } else if mods == ["proxy", "alias"] {
                // ASSUME(S-MC-ALIAS): a bidirectional proxy fence (no release / acquire of
                // its own).
                Op::Fence {
                    kind: FenceKind::ProxyAlias,
                }
            } else {
                bail!(lw.err("this fence is not modeled yet"))
            }
        }
        "mbarrier" => match mods.first().copied() {
            Some("inval") => Op::MbarInval {
                addr: lw.addr(op_n(0)?)?,
            },
            Some("init") => Op::MbarInit {
                addr: lw.addr(op_n(0)?)?,
                count: lw.val(op_n(1)?)?,
                encoded: false,
            },
            Some("arrive") => {
                // ASSUME(S-SCOPE): .release.cta / .release.cluster are not distinguished.
                let remote = has("shared::cluster");
                let tx = if has("expect_tx") {
                    Some(lw.val(op_n(2)?)?)
                } else {
                    None
                };
                let count = if !has("expect_tx") && ops.len() > 2 {
                    Some(lw.val(op_n(2)?)?)
                } else {
                    None
                };
                Op::MbarArrive {
                    addr: lw.addr(op_n(1)?)?,
                    tx,
                    remote,
                    count,
                    complete: false,
                    cluster: has("release") && has("cluster"),
                }
            }
            // `mbarrier.expect_tx` / `complete_tx [.relaxed] [.cta | .cluster] .shared{::cta,
            // ::cluster}.b64 [a], tx`: no arrival, tx added to / taken from the tx-count, relaxed
            // (ptxas: SYNCS.ARRIVE.TRANS64.RED.A0TR / .A0TX; S-MBAR).
            Some(m @ ("expect_tx" | "complete_tx")) => {
                if let Some(x) = mods[1..].iter().find(|x| {
                    !matches!(
                        **x,
                        "relaxed"
                            | "cta"
                            | "cluster"
                            | "shared"
                            | "shared::cta"
                            | "shared::cluster"
                            | "b64"
                    )
                }) {
                    bail!(lw.err(format!("mbarrier.{m}: modifier .{x} not modeled")));
                }
                Op::MbarArrive {
                    addr: lw.addr(op_n(0)?)?,
                    tx: Some(lw.val(op_n(1)?)?),
                    remote: has("shared::cluster"),
                    count: Some(Val::Imm(0)),
                    complete: m == "complete_tx",
                    cluster: false,
                }
            }
            // test_wait: the non-suspending form, alike for the model (as SASS PHASECHK without
            // .TRYWAIT; E-SPIN).
            Some("try_wait" | "test_wait") if has("parity") => Op::MbarTryWait {
                d: lw.pred_dst(op_n(0)?)?,
                addr: lw.addr(op_n(1)?)?,
                parity: lw.val(op_n(2)?)?,
                bit31: false,
            },
            _ => bail!(lw.err("this mbarrier op is not modeled yet")),
        },
        // Non-bulk cp.async (SASS LDGSTS / LDGDEPBAR / DEPBAR / ARRIVES.LDGSTSBAR):
        // `cp.async.{ca,cg}.shared{::cta}.global[.L2::...] [dst], [src], cp-size{, src-size |
        // ignore-src}{, cache-policy}`. src-size must be 0 (zero fill) or cp-size; ignore-src
        // is a predicate (true: zero fill).
        "cp" if mods.len() >= 4
            && mods[0] == "async"
            && matches!(mods[1], "ca" | "cg")
            && matches!(mods[2], "shared" | "shared::cta")
            && mods[3] == "global" =>
        {
            // ASSUME(L-PTX-HINT): `.L2::cache_hint` (with its cache-policy operand) and the
            // `.L2::{64,128,256}B` prefetch sizes are performance hints only.
            if let Some(m) = mods[4..]
                .iter()
                .find(|m| !matches!(**m, "L2::cache_hint" | "L2::64B" | "L2::128B" | "L2::256B"))
            {
                bail!(lw.err(format!("cp.async: modifier .{m} not modeled")));
            }
            let size =
                parse_int(op_n(2)?).ok_or_else(|| lw.err("cp.async size must be an immediate"))?;
            if !matches!(size, 4 | 8 | 16) || (mods[1] == "cg" && size != 16) {
                bail!(lw.err("cp.async size must be 4, 8 or 16 (16 for .cg)"));
            }
            let hint = has("L2::cache_hint");
            let n_extra = ops.len() - 3 - hint as usize;
            // The fourth operand: src-size (immediate or register; the first src-size bytes
            // are copied, the rest zero-filled; probes/cpasync_srcsize = GB300) or the
            // ignore-src predicate.
            let mut src_size = None;
            let src_ok = match n_extra {
                0 => Pred::TRUE,
                1 => {
                    let o = op_n(3)?.trim();
                    match parse_int(o) {
                        Some(0) => Pred {
                            reg: None,
                            neg: true,
                        },
                        Some(n) if n == size => Pred::TRUE,
                        Some(n) if n < size => {
                            src_size = Some(Val::Imm(n));
                            Pred::TRUE
                        }
                        Some(_) => bail!(lw.err("cp.async src-size larger than its cp-size")),
                        None => match lw.reg(o.trim_start_matches('!')) {
                            Ok(r) if r.file == File::P => {
                                let p = lw.pred_src(o)?;
                                Pred {
                                    reg: p.reg,
                                    neg: !p.neg,
                                }
                            }
                            _ => {
                                src_size = Some(lw.val(o)?);
                                Pred::TRUE
                            }
                        },
                    }
                }
                _ => bail!(lw.err("cp.async: unexpected operands")),
            };
            Op::CpAsync {
                dst: lw.addr(op_n(0)?)?,
                src: lw.addr(op_n(1)?)?,
                bytes: size as u32,
                src_ok,
                src_size,
                zfill: false,
            }
        }
        "cp" if mods == ["async", "commit_group"] => Op::CpAsyncCommit,
        "cp" if mods == ["async", "wait_group"] => {
            let n = parse_int(op_n(0)?)
                .ok_or_else(|| lw.err("wait_group count must be an immediate"))?;
            Op::CpAsyncWait { n: n as u32 }
        }
        // `cp.async.wait_all` = commit_group + wait_group 0.
        "cp" if mods == ["async", "wait_all"] => {
            extra.push(Op::CpAsyncWait { n: 0 });
            Op::CpAsyncCommit
        }
        // `cp.async.mbarrier.arrive[.noinc].shared{::cta}.b64 [mbar]`: `.noinc` is one
        // asynchronous arrival once this thread's cp.asyncs land; without it the pending count
        // is first raised by one (as ptxas: SYNCS.ARRIVE.TRANS64.RED.A0T1, one expected
        // transaction unit and no arrival, then ARRIVES.LDGSTSBAR.64.TRANSCNT completing it).
        // ASSUME(E-CPASYNC-MBAR): see ASSUMPTIONS.md
        "cp" if mods.len() >= 3 && mods[..3] == ["async", "mbarrier", "arrive"] => {
            let noinc = has("noinc");
            if !mods[3..]
                .iter()
                .all(|m| matches!(*m, "noinc" | "shared" | "shared::cta" | "b64"))
            {
                bail!(lw.err("cp.async.mbarrier.arrive variant not modeled"));
            }
            let addr = lw.addr(op_n(0)?)?;
            if noinc {
                Op::CpAsyncMbarArrive { addr, tx: false }
            } else {
                extra.push(Op::CpAsyncMbarArrive {
                    addr: addr.clone(),
                    tx: true,
                });
                Op::MbarArrive {
                    addr,
                    tx: Some(Val::Imm(1)),
                    remote: false,
                    count: Some(Val::Imm(0)),
                    complete: false,
                    cluster: false,
                }
            }
        }
        // Non-tensor bulk copies (as SASS UBLKCP; DeepEP's tma_load_1d / tma_store_1d):
        // `cp.async.bulk.shared::{cta,cluster}.global.mbarrier::complete_tx::bytes [dst], [src],
        // size, [mbar]` and `cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size`.
        // Lowered as TMA ops with `dims = 0`; `tmap` carries the byte count (exec.rs
        // `bulk_map`), `coords[0]` the global address.
        // `cp.async.bulk.prefetch.L2.global [a], size` (SASS UBLKPF.L2): a cache hint
        // (probes/memhints).
        "cp" if mods == ["async", "bulk", "prefetch", "L2", "global"] => Op::Nop,
        // `cp.async.bulk.prefetch.tensor.<n>d.L2.global[.tile][.L2::cache_hint] [tmap,
        // {coords}][, policy]` (SASS UTMAPF.L2.<n>D): a cache hint (probes/tma_prefetch).
        "cp" if mods.len() >= 6
            && mods[..3] == ["async", "bulk", "prefetch"]
            && mods[3] == "tensor"
            && matches!(mods[4], "1d" | "2d" | "3d" | "4d" | "5d")
            && mods[5..]
                .iter()
                .all(|m| matches!(*m, "L2" | "global" | "tile" | "L2::cache_hint"))
            && has("L2")
            && has("global") =>
        {
            Op::Nop
        }
        "cp" if mods.len() >= 4
            && mods[..2] == ["async", "bulk"]
            && !has("tensor")
            && !has("prefetch")
            && (mods[2].starts_with("shared") && mods[3] == "global"
                || mods[2] == "global" && mods[3] == "shared::cta") =>
        {
            // ASSUME(L-PTX-HINT): a trailing `.L2::cache_hint` and its cache-policy operand
            // (last) are a performance hint only. Fail closed on every other modifier
            // (multicast, cta_group).
            let hint = mods.len() == 6 && mods[5] == "L2::cache_hint";
            let known = mods.len() == 5 + hint as usize
                && if mods[2] == "global" {
                    mods[4] == "bulk_group"
                } else {
                    mods[4] == "mbarrier::complete_tx::bytes"
                        && matches!(mods[2], "shared::cta" | "shared::cluster")
                };
            if !known {
                bail!(lw.err(format!(
                    "cp.async.bulk form .{} not modeled",
                    mods.join(".")
                )));
            }
            if ops.len() != if mods[2] == "global" { 3 } else { 4 } + hint as usize {
                bail!(lw.err("cp.async.bulk: unexpected operands"));
            }
            let reg_of = |a: Addr, what: &str| -> Result<(Val, u32)> {
                let b = a
                    .base
                    .ok_or_else(|| lw.err(format!("cp.async.bulk {what} must be a register")))?;
                Ok((b, a.off as u32))
            };
            let size = lw.val(op_n(2)?)?;
            if mods[2] == "global" {
                if !has("bulk_group") {
                    bail!(lw.err("cp.async.bulk store without .bulk_group not modeled"));
                }
                let g = lw.addr(op_n(0)?)?;
                if g.off != 0 {
                    bail!(lw.err("cp.async.bulk global address offset not modeled"));
                }
                let (src, off) = reg_of(lw.addr(op_n(1)?)?, "source")?;
                Op::TmaStore {
                    dims: 0,
                    src,
                    off,
                    tmap: size,
                    coords: vec![
                        g.base
                            .ok_or_else(|| lw.err("global address must be a register"))?,
                    ],
                    red: false,
                }
            } else {
                if !has("mbarrier::complete_tx::bytes") {
                    bail!(lw.err("cp.async.bulk load without an mbarrier not modeled"));
                }
                let g = lw.addr(op_n(1)?)?;
                if g.off != 0 {
                    bail!(lw.err("cp.async.bulk global address offset not modeled"));
                }
                let (dst, off) = reg_of(lw.addr(op_n(0)?)?, "destination")?;
                let (mbar, mbar_off) = reg_of(lw.addr(op_n(3)?)?, "mbarrier")?;
                Op::TmaLoad {
                    dims: 0,
                    dst,
                    off,
                    mbar,
                    mbar_off,
                    tmap: size,
                    coords: vec![
                        g.base
                            .ok_or_else(|| lw.err("global address must be a register"))?,
                    ],
                    mask: None,
                    pair: false,
                    src_smem: false,
                }
            }
        }
        "cp" if mods == ["async", "bulk", "commit_group"] => Op::BulkCommit,
        "cp" if mods.starts_with(&["async", "bulk", "wait_group"]) => {
            let n = lw.val(op_n(0)?)?;
            let Val::Imm(n) = n else {
                bail!(lw.err("wait_group count must be an immediate"))
            };
            Op::BulkWait {
                n: n as u32,
                read: has("read"),
                sb: None,
                cp: false,
            }
        }
        "cp" if mods.starts_with(&["async", "bulk", "tensor"]) => {
            let dims: u8 = mods[3]
                .trim_end_matches('d')
                .parse()
                .map_err(|_| lw.err("bad tensor dims"))?;
            // Fail closed: `.im2col` (and its `::w` forms) gathers through per-pixel offsets
            // that a tiled load does not compute; it, `.tile::scatter4` and any other modifier
            // outside the modeled tiled forms are refused, never lowered as a tiled copy.
            if let Some(m) = mods[4..].iter().find(|m| {
                !matches!(
                    **m,
                    "global"
                        | "shared::cta"
                        | "shared::cluster"
                        | "mbarrier::complete_tx::bytes"
                        | "tile"
                        | "tile::gather4"
                        | "multicast::cluster"
                        | "cta_group::1"
                        | "cta_group::2"
                        | "L2::cache_hint"
                        | "bulk_group"
                )
            }) {
                bail!(lw.err(format!("cp.async.bulk.tensor: modifier .{m} not modeled")));
            }
            let store = mods.get(4) == Some(&"global") && has("shared::cta") && has("bulk_group");
            if store {
                // cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group [tmap, {c..}], [src]
                let t = op_n(0)?
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .to_string();
                let (tm, coords) = t
                    .split_once(',')
                    .ok_or_else(|| lw.err("bad tensor operand"))?;
                let coords: Vec<Val> = coords
                    .trim()
                    .trim_start_matches('{')
                    .trim_end_matches('}')
                    .split(',')
                    .map(|c| lw.val(c))
                    .collect::<Result<_>>()?;
                if coords.len() != dims as usize {
                    bail!(lw.err("coordinate count != dims"));
                }
                let sa = lw.addr(op_n(1)?)?;
                let src = sa
                    .base
                    .ok_or_else(|| lw.err("TMA store source must be a register"))?;
                return Ok(Inst {
                    guard,
                    op: Op::TmaStore {
                        dims,
                        src,
                        off: sa.off as u32,
                        tmap: lw.val(tm)?,
                        coords,
                        red: false,
                    },
                    uniform: false,
                    loc: lw.line,
                    text: text.to_string(),
                });
            }
            if !((has("shared::cluster") || has("shared::cta"))
                && has("global")
                && has("mbarrier::complete_tx::bytes"))
            {
                bail!(lw.err("only global->shared TMA loads and shared::cta->global bulk_group stores are modeled so far"));
            }
            let da = lw.addr(op_n(0)?)?;
            let dst_off = da.off as u32;
            let dst = da
                .base
                .ok_or_else(|| lw.err("TMA dst must be a register"))?;
            let t = op_n(1)?
                .trim()
                .trim_start_matches('[')
                .trim_end_matches(']')
                .to_string();
            let (tm, coords) = t
                .split_once(',')
                .ok_or_else(|| lw.err("bad tensor operand"))?;
            let coords: Vec<Val> = coords
                .trim()
                .trim_start_matches('{')
                .trim_end_matches('}')
                .split(',')
                .map(|c| lw.val(c))
                .collect::<Result<_>>()?;
            // `.tile::gather4` (2-D only): the column, then four row coordinates.
            let want = if has("tile::gather4") {
                5
            } else {
                dims as usize
            };
            if has("tile::gather4") && dims != 2 || coords.len() != want {
                bail!(lw.err("coordinate count != dims"));
            }
            let ma = lw.addr(op_n(2)?)?;
            let mbar = ma.base.ok_or_else(|| lw.err("mbar must be a register"))?;
            let mask = if has("multicast::cluster") {
                Some(lw.val(op_n(3)?)?)
            } else {
                None
            };
            Op::TmaLoad {
                dims,
                dst,
                off: dst_off,
                mbar,
                mbar_off: ma.off as u32,
                tmap: lw.val(tm)?,
                coords,
                mask,
                pair: has("cta_group::2"),
                src_smem: false,
            }
        }
        _ => bail!(lw.err(format!("unsupported PTX instruction `{opcode}`"))),
    };
    lw.extra = extra;
    lw.pre.extend(pre);
    Ok(Inst {
        guard,
        op,
        uniform: false,
        loc: lw.line,
        text: text.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixture() {
        let src = include_str!("../../../fixtures/r0_tma_sum/kernel.ptx");
        let k = parse_kernel(src, None).unwrap();
        assert_eq!(k.name, "r0_tma_sum");
        assert_eq!(k.params.len(), 2);
        assert_eq!(k.params[1].offset, 128);
        assert!(
            k.code
                .iter()
                .any(|i| matches!(i.op, Op::TmaLoad { dims: 2, .. }))
        );
        assert_eq!(k.smem_bytes, 0x400 + 8192 + 8);
    }

    fn one(line: &str) -> Result<Op> {
        let src = format!(
            ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry k()\n{{\n\
             .reg .pred %p<4>;\n.reg .b32 %r<8>;\n{line}\nret;\n}}\n"
        );
        Ok(parse_kernel(&src, None)?.code[0].op.clone())
    }

    #[test]
    fn redux_neg_setp_f32_forms() {
        let red = |l: &str| match one(l).unwrap() {
            Op::Redux { op, .. } => op,
            o => panic!("{l}: {o:?}"),
        };
        assert_eq!(
            red("redux.sync.max.NaN.f32 %r1, %r2, 0xffffffff;"),
            ReduxOp::MaxF {
                abs: false,
                nan: true
            }
        );
        assert_eq!(
            red("redux.sync.min.abs.f32 %r1, %r2, 0xffffffff;"),
            ReduxOp::MinF {
                abs: true,
                nan: false
            }
        );
        assert_eq!(red("redux.sync.add.s32 %r1, %r2, -1;"), ReduxOp::Sum);
        assert_eq!(
            red("redux.sync.max.s32 %r1, %r2, 0xffffffff;"),
            ReduxOp::MaxS
        );
        assert_eq!(red("redux.sync.or.b32 %r1, %r2, 0xffffffff;"), ReduxOp::Or);
        // Fail closed: forms that do not exist, an empty membermask.
        assert!(one("redux.sync.max.NaN.u32 %r1, %r2, 0xffff;").is_err());
        assert!(one("redux.sync.add.f32 %r1, %r2, 0xffffffff;").is_err());
        assert!(one("redux.sync.add.s32 %r1, %r2, 0;").is_err());
        // A register or partial immediate membermask: the redux over the lanes it names.
        assert!(matches!(
            &ops("redux.sync.add.s32 %r1, %r2, %r3;").unwrap()[..],
            [Op::Redux {
                op: ReduxOp::Sum,
                mask: Some(Val::Reg(..)),
                ..
            }]
        ));
        assert!(matches!(
            &ops("redux.sync.add.s32 %r1, %r2, 0xff;").unwrap()[..],
            [Op::Redux {
                mask: Some(Val::Imm(0xff)),
                ..
            }]
        ));
        assert!(matches!(
            one("redux.sync.add.s32 %r1, %r2, 0xffffffff;").unwrap(),
            Op::Redux { mask: None, .. }
        ));
        // createpolicy used only as a cache hint: a placeholder; used as data: refused.
        assert!(ops("createpolicy.fractional.L2::evict_first.b64 %rd2, 1.0;\nld.global.L2::cache_hint.b32 %r1, [%rd1], %rd2;").is_ok());
        assert!(ops("createpolicy.fractional.L2::evict_first.b64 %rd2, 1.0;\nst.global.b64 [%rd1], %rd2;").is_err());
        assert!(matches!(
            &ops("prmt.b32 %r1, %r2, %r3, 0x7632U;").unwrap()[..],
            [Op::Prmt { .. }]
        ));
        // A vector store with an immediate element: moves into the scratch block.
        assert!(matches!(
            &ops("st.global.v2.b32 [%rd1], {0, %r2};").unwrap()[..],
            [
                Op::Mov { a: Val::Imm(0), .. },
                Op::Mov { .. },
                Op::St { bytes: 8, .. }
            ]
        ));
        assert!(one("redux.sync.max.NaN.abs.f32 %r1, %r2, 0xffffffff;").is_err());
        assert!(matches!(
            one("neg.f32 %r1, %r2;").unwrap(),
            Op::Fadd {
                neg: [true, false],
                ftz: false,
                ..
            }
        ));
        assert!(one("neg.ftz.f32 %r1, %r2;").is_err());
        assert!(matches!(
            one("setp.leu.f32 %p1, %r1, 0f00000000;").unwrap(),
            Op::Fsetp {
                cmp: FCmp::Leu,
                ftz: false,
                ..
            }
        ));
        assert!(matches!(
            one("setp.gt.ftz.f32 %p1, %r1, %r2;").unwrap(),
            Op::Fsetp {
                cmp: FCmp::Gt,
                ftz: true,
                ..
            }
        ));
        assert!(one("setp.lt.f32 %p1|%p2, %r1, %r2;").is_err());
        assert!(one("setp.lt.and.f32 %p1, %r1, %r2, %p3;").is_err());
    }

    /// Every op one statement lowers to (64-bit registers %rd0..7 declared too).
    fn ops(line: &str) -> Result<Vec<Op>> {
        let src = format!(
            ".version 9.0\n.target sm_103a\n.address_size 64\n.visible .entry k()\n{{\n\
             .reg .pred %p<4>;\n.reg .b32 %r<8>;\n.reg .b64 %rd<8>;\n{line}\nret;\n}}\n"
        );
        let k = parse_kernel(&src, None)?;
        Ok(k.code[..k.code.len() - 1]
            .iter()
            .map(|i| i.op.clone())
            .collect())
    }

    #[test]
    fn fmad_contraction() {
        // PTX-FMAD: a plain mul read once by a plain add / sub in its block is one FFMA. The
        // add's result is stored (a dead reader does not count).
        let last = |l: &str| {
            let st = if l.contains(".f64") {
                "st.global.f64 [%rd1], %rd4;"
            } else {
                "st.global.f32 [%rd1], %r4;"
            };
            let v = ops(&format!("{l}\n{st}")).unwrap();
            v[v.len() - 2].clone()
        };
        match last("mul.f32 %r1, %r2, %r3;\nadd.f32 %r4, %r5, %r1;") {
            Op::Ffma {
                c: Some(Val::Reg(..)),
                neg,
                sat: false,
                ..
            } => assert_eq!(neg, [false; 3]),
            o => panic!("{o:?}"),
        }
        match last("mul.f32 %r1, %r2, %r3;\nsub.f32 %r4, %r5, %r1;") {
            Op::Ffma { neg, .. } => assert_eq!(neg, [true, false, false]),
            o => panic!("{o:?}"),
        }
        match last("mul.f32 %r1, %r2, %r3;\nsub.sat.f32 %r4, %r1, %r5;") {
            Op::Ffma { neg, sat: true, .. } => assert_eq!(neg, [false, false, true]),
            o => panic!("{o:?}"),
        }
        assert!(matches!(
            last("mul.f64 %rd1, %rd2, %rd3;\nadd.f64 %rd4, %rd1, %rd5;"),
            Op::Dfma {
                c: Some(Val::Reg(..)),
                ..
            }
        ));
        // Kept apart: a rounding modifier on either, .ftz on one only.
        for l in [
            "mul.f32 %r1, %r2, %r3;\nadd.rn.f32 %r4, %r1, %r5;",
            "mul.rn.f32 %r1, %r2, %r3;\nadd.f32 %r4, %r1, %r5;",
            "mul.ftz.f32 %r1, %r2, %r3;\nadd.f32 %r4, %r1, %r5;",
        ] {
            assert!(matches!(last(l), Op::Fadd { .. }), "{l}");
        }
        // All or nothing per mul (probes/fmad_multi): every reader an add: an FFMA each; a
        // live store, mul or compare reader: the adds round the product; dead readers ignored.
        let all = |l: &str| ops(l).unwrap();
        let ffmas = |v: &[Op]| {
            v.iter()
                .filter(|o| matches!(o, Op::Ffma { c: Some(_), .. }))
                .count()
        };
        assert_eq!(
            ffmas(&all(
                "mul.f32 %r1, %r2, %r3;\nadd.f32 %r4, %r1, %r5;\nsub.f32 %r6, %r5, %r1;\nst.global.f32 [%rd1], %r4;\nst.global.f32 [%rd1], %r6;"
            )),
            2
        );
        assert_eq!(
            ffmas(&all(
                "mul.f32 %r1, %r2, %r3;\nadd.f32 %r4, %r1, %r5;\nst.global.f32 [%rd1], %r1;\nst.global.f32 [%rd1], %r4;"
            )),
            0
        );
        assert_eq!(
            ffmas(&all(
                "mul.f32 %r1, %r2, %r3;\nadd.f32 %r4, %r1, %r5;\nsetp.gt.f32 %p1, %r1, %r5;\nst.global.f32 [%rd1], %r4;"
            )),
            1
        );
        assert_eq!(
            ffmas(&all(
                "mul.f32 %r1, %r2, %r3;\nmov.f32 %r6, %r1;\nadd.f32 %r4, %r6, %r5;\nst.global.f32 [%rd1], %r4;"
            )),
            1
        );
        match last("mul.f32 %r1, %r2, %r3;\nneg.f32 %r6, %r1;\nadd.f32 %r4, %r6, %r5;") {
            Op::Ffma { neg, .. } => assert_eq!(neg, [true, false, false]),
            o => panic!("{o:?}"),
        }
        // An add of two products folds the first operand's, or the one with no other reader.
        match last("mul.f32 %r1, %r2, %r3;\nmul.f32 %r6, %r2, %r5;\nadd.f32 %r4, %r1, %r6;") {
            Op::Ffma {
                a: Val::Reg(a, _),
                c: Some(Val::Reg(c, _)),
                ..
            } => assert!(a.idx == 2 && c.idx == 6),
            o => panic!("{o:?}"),
        }
        let v = all(
            "mul.f32 %r1, %r2, %r3;\nmul.f32 %r6, %r2, %r5;\nadd.f32 %r4, %r1, %r6;\nadd.f32 %r7, %r1, %r5;\nst.global.f32 [%rd1], %r4;\nst.global.f32 [%rd1], %r7;",
        );
        assert!(
            matches!(
                v[2],
                Op::Ffma {
                    c: Some(Val::Reg(Reg { idx: 1, .. }, _)),
                    ..
                }
            ),
            "{v:?}"
        );
        assert!(matches!(v[3], Op::Fadd { .. }), "{v:?}");
        // Refused: ambiguous or unmodeled contractions.
        for l in [
            "mul.f32 %r1, %r2, %r3;\nmul.f32 %r6, %r2, %r5;\nadd.f32 %r4, %r1, %r6;\nadd.f32 %r7, %r1, %r5;\nadd.f32 %r0, %r6, %r5;\nst.global.v2.f32 [%rd1], {%r7, %r0};",
            "@%p1 mul.f32 %r1, %r2, %r3;\n@%p1 add.f32 %r4, %r1, %r5;",
            "mul.f32 %r1, %r2, %r3;\nmov.f32 %r2, 0f3F800000;\nadd.f32 %r4, %r1, %r5;",
            "mul.f32 %r1, %r2, %r3;\n@%p1 bra L1;\nL1:\nadd.f32 %r4, %r1, %r5;",
            // A definition among several live past its block, read by an add there.
            "mov.f32 %r1, 0f00000000;\nmul.f32 %r1, %r1, %r3;\nbra.uni L1;\nL1:\nadd.f32 %r4, %r5, %r1;",
        ] {
            assert!(
                ops(&format!("{l}\nst.global.f32 [%rd1], %r4;")).is_err(),
                "{l}"
            );
        }
    }

    #[test]
    fn fmad_contraction_f16() {
        // PTX-FMAD for scalar .f16 (probes/fmad_f16): HFMA2 into every add / sub reader; a
        // converted product keeps HMUL2 + HADD2; copies, negs and two-product adds refused.
        let all = |l: &str| {
            ops(&format!(
                ".reg .b16 %h<8>;\n{l}\nst.global.b16 [%rd1], %h4;\nst.global.b16 [%rd1], %h5;"
            ))
        };
        let hfmas = |v: &[Op]| {
            v.iter()
                .filter(|o| {
                    matches!(
                        o,
                        Op::Hfma2 {
                            b: Val::Reg(..),
                            c: Val::Reg(..),
                            ..
                        }
                    )
                })
                .count()
        };
        let v =
            all("mul.f16 %h1, %h2, %h3;\nadd.f16 %h4, %h6, %h1;\nsub.f16 %h5, %h1, %h6;").unwrap();
        assert_eq!(hfmas(&v), 2, "{v:?}");
        assert!(
            matches!(
                v[v.len() - 4],
                Op::Hfma2 {
                    neg_a: false,
                    neg_c: true,
                    ..
                }
            ),
            "{v:?}"
        );
        let v = all("mul.f16 %h1, %h2, %h3;\nsub.f16 %h4, %h6, %h1;\nmov.b16 %h5, %h6;").unwrap();
        assert!(
            v.iter().any(|o| matches!(
                o,
                Op::Hfma2 {
                    neg_a: true,
                    neg_c: false,
                    ..
                }
            )),
            "{v:?}"
        );
        let v = all("mul.f16 %h1, %h2, %h3;\nadd.f16 %h4, %h1, %h6;\ncvt.f32.f16 %r1, %h1;\nmov.b16 %h5, %h6;\nst.global.f32 [%rd1], %r1;").unwrap();
        assert_eq!(hfmas(&v), 0, "{v:?}");
        for l in [
            "mul.f16 %h1, %h2, %h3;\nmov.b16 %h7, %h1;\nadd.f16 %h4, %h7, %h6;\nmov.b16 %h5, %h6;",
            "mul.f16 %h1, %h2, %h3;\nneg.f16 %h7, %h1;\nadd.f16 %h4, %h7, %h6;\nmov.b16 %h5, %h6;",
            "mul.f16 %h1, %h2, %h3;\nmul.f16 %h7, %h2, %h6;\nadd.f16 %h4, %h1, %h7;\nmov.b16 %h5, %h6;",
        ] {
            assert!(all(l).is_err(), "{l}");
        }
    }

    #[test]
    fn vector_register() {
        // `.reg .v4 .s32 abcd;` in an inline-asm scope: components, and the whole as a vector.
        let v = ops("{\n.reg .v4 .s32 abcd;\nmov.s32 abcd.x, %r1;\nmov.s32 abcd.y, %r2;\nmov.s32 abcd.z, %r3;\nmov.s32 abcd.w, %r4;\nst.global.v4.s32 [%rd1], abcd;\n}").unwrap();
        let regs: Vec<Reg> = v[..4].iter().map(|o| o.dst_regs()[0]).collect();
        assert!(regs.windows(2).all(|w| w[1].idx == w[0].idx + 1), "{v:?}");
        assert!(ops("{\n.reg .v3 .s32 abc;\n}").is_err());
    }

    #[test]
    fn max_abs_three_sources() {
        // `max.NaN.abs.f32 d, a, b, c` = FMNMX3.NAN |a|, |b|, |c|; two sources need .xorsign.
        let v = ops("max.NaN.abs.f32 %r1, %r2, %r3, %r4;").unwrap();
        assert!(
            matches!(
                v[0],
                Op::Fmnmx {
                    c: Some(_),
                    abs: [true, true, true],
                    nan: true,
                    xorsign: false,
                    ..
                }
            ),
            "{v:?}"
        );
        assert!(ops("max.abs.f32 %r1, %r2, %r3;").is_err());
    }

    /// `red.async.release.{gpu|sys}.global.add.{u32|s32|u64|s64}`: an atomic add, a release at
    /// that scope. Forms the PTX ISA does not define on `.global` (`.relaxed`, ops other than
    /// add: ptxas silently encodes them as the release's ADD), `.mmio` and other scopes are
    /// refused.
    #[test]
    fn red_async_global_forms() {
        let first = |l: &str| ops(l).map(|v| v[0].clone());
        let atom = |l: &str| match first(l).unwrap() {
            Op::Atom {
                op: AtomOp::Add,
                space: Space::Global,
                bytes,
                d: None,
                sync,
                ..
            } => (bytes, sync),
            o => panic!("{l}: {o:?}"),
        };
        let ms = |sem, scope| MemSync { sem, scope };
        assert_eq!(
            atom("red.async.release.gpu.global.add.s32 [%rd1], %r1;"),
            (4, ms(Sem::Release, Scope::Gpu))
        );
        assert_eq!(
            atom("red.async.release.gpu.global.add.u32 [%rd1+8], %r1;"),
            (4, ms(Sem::Release, Scope::Gpu))
        );
        assert_eq!(
            atom("red.async.release.sys.global.add.u64 [%rd1], %rd2;"),
            (8, ms(Sem::Release, Scope::Sys))
        );
        assert_eq!(
            atom("red.async.release.gpu.global.add.s64 [%rd1], %rd2;"),
            (8, ms(Sem::Release, Scope::Gpu))
        );
        for bad in [
            "red.async.relaxed.gpu.global.add.u32 [%rd1], %r1;",
            "red.async.relaxed.sys.global.add.u64 [%rd1], %rd2;",
            "red.async.release.cluster.global.add.u32 [%rd1], %r1;",
            "red.async.mmio.release.gpu.global.add.u32 [%rd1], %r1;",
            "red.async.release.gpu.global.add.f32 [%rd1], %r1;",
            "red.async.release.gpu.global.min.s32 [%rd1], %r1;",
            "red.async.release.gpu.global.inc.u32 [%rd1], %r1;",
            "red.async.release.gpu.global.or.b32 [%rd1], %r1;",
            "red.async.mmio.release.sys.global.add.u32 [%rd1], %r1;",
            "red.async.release.cta.global.add.s32 [%rd1], %r1;",
            "red.async.global.add.s32 [%rd1], %r1;",
        ] {
            assert!(first(bad).is_err(), "{bad}");
        }
    }

    /// `.ftz` on `fma` / `add` / `sub` / `mul .f32x2` sets `ftz` (FFMA2.FTZ, FADD2.FTZ[.RM],
    /// FMUL2.FTZ); a `.ftz` mul read by a same-mode add is refused, since whether ptxas
    /// contracts it depends on its version (13.1: only without a rounding modifier).
    #[test]
    fn packed_f32x2_ftz() {
        let f = |l: &str| match &ops(l).unwrap()[..] {
            [.., Op::Ffma2 { rnd, ftz, neg, .. }] => (*rnd, *ftz, *neg),
            o => panic!("{l}: {o:?}"),
        };
        assert_eq!(
            f("fma.rn.ftz.f32x2 %rd1, %rd2, %rd3, %rd4;"),
            (Rnd::Rn, true, [false; 3])
        );
        assert_eq!(
            f("add.rm.ftz.f32x2 %rd1, %rd2, %rd3;"),
            (Rnd::Rm, true, [false; 3])
        );
        assert_eq!(
            f("sub.rn.ftz.f32x2 %rd1, %rd2, %rd3;"),
            (Rnd::Rn, true, [false, false, true])
        );
        assert_eq!(
            f("mul.rz.ftz.f32x2 %rd1, %rd2, %rd3;"),
            (Rnd::Rz, true, [false; 3])
        );
        assert_eq!(
            f("fma.rn.f32x2 %rd1, %rd2, %rd3, %rd4;"),
            (Rnd::Rn, false, [false; 3])
        );
        // Kept apart by ptxas 13.1 (FMUL2.FTZ + FADD2.FTZ) or contracted (no rounding
        // modifier): either way refused.
        assert!(
            ops("mul.rn.ftz.f32x2 %rd1, %rd2, %rd3;\nadd.rn.ftz.f32x2 %rd4, %rd1, %rd3;").is_err()
        );
        assert!(ops("mul.ftz.f32x2 %rd1, %rd2, %rd3;\nadd.ftz.f32x2 %rd4, %rd1, %rd3;").is_err());
        // Different ftz settings never contract (FMUL2.FTZ + FADD2).
        assert!(ops("mul.rn.ftz.f32x2 %rd1, %rd2, %rd3;\nadd.rn.f32x2 %rd4, %rd1, %rd3;").is_ok());
        assert!(ops("fma.rn.ftz.sat.f32x2 %rd1, %rd2, %rd3, %rd4;").is_err());
    }

    #[test]
    fn red_async_and_tensor_reduce_forms() {
        // red.async.release.gpu.global.add = a release red at gpu scope; other operations are
        // refused (ptxas 13.0 compiles them to an add).
        let v = ops("red.async.release.gpu.global.add.s32 [%rd1], %r1;").unwrap();
        assert!(
            matches!(
                &v[0],
                Op::Atom {
                    op: AtomOp::Add,
                    space: Space::Global,
                    sync: MemSync {
                        sem: Sem::Release,
                        scope: Scope::Gpu
                    },
                    d: None,
                    ..
                }
            ),
            "{v:?}"
        );
        for l in [
            "red.async.relaxed.sys.global.add.u64 [%rd1], %rd2;",
            "red.async.release.gpu.global.min.s32 [%rd1], %r1;",
            "red.async.release.gpu.global.or.b32 [%rd1], %r1;",
            "atom.async.release.gpu.global.add.s32 %r2, [%rd1], %r1;",
            "red.async.release.gpu.global.add.f32 [%rd1], %r1;",
        ] {
            assert!(ops(l).is_err(), "{l}");
        }
        // Tensor add-reductions without `.tile` and with an L2 cache hint.
        let a = ops("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%rd1, {%r1, %r2}], [%r3];").unwrap();
        let b = ops("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%rd1, {%r1, %r2}], [%r3];").unwrap();
        let c = ops("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group.L2::cache_hint [%rd1, {%r1, %r2}], [%r3], %rd2;").unwrap();
        assert!(
            a == b
                && b == c
                && matches!(
                    a[0],
                    Op::TmaStore {
                        red: true,
                        dims: 2,
                        ..
                    }
                ),
            "{a:?}"
        );
        assert!(ops("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group.L2::cache_hint [%rd1, {%r1, %r2}], [%r3];").is_err());
    }

    #[test]
    fn tcgen05_mma_operand_order() {
        // [d], a, b, idesc, ({mask},) enable-input-d (, scale-input-d): `..., 0, 1` overwrites D
        // (enable 0; the scale of an unread accumulator is moot); a nonzero scale with a live
        // enable scales the accumulator; the mask is a braced vector.
        let mma = |tail: &str| {
            ops(&format!(
                "tcgen05.mma.cta_group::1.kind::f16 [%r1], %rd1, %rd2, %r2, {tail};"
            ))
        };
        let en = |v: &[Op]| match v.last().unwrap() {
            Op::Mma {
                enable_d,
                disable,
                scale_d,
                ..
            } => (*enable_d, disable.len(), *scale_d),
            o => panic!("{o:?}"),
        };
        assert_eq!(
            en(&mma("0, 1").unwrap()),
            (
                Pred {
                    reg: None,
                    neg: true
                },
                0,
                1
            )
        );
        assert_eq!(en(&mma("1, 0").unwrap()), (Pred::TRUE, 0, 0));
        assert_eq!(en(&mma("1, 3").unwrap()), (Pred::TRUE, 0, 3));
        assert_eq!(en(&mma("%p1, 15").unwrap()).2, 15);
        assert!(mma("1, 16").is_err());
        assert!(mma("1, %r9").is_err());
        assert_eq!(
            en(&mma("{%r3, %r4, %r5, %r6}, 1").unwrap()),
            (Pred::TRUE, 4, 0)
        );
        assert_eq!(en(&mma("{%r3, %r4, %r5, %r6}, 0, 2").unwrap()).1, 4);
    }

    #[test]
    fn tma_im2col_refused() {
        assert!(ops("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r2, %r3, %r4, %r5}], [%r6], {%rs1, %rs2};").is_err());
        assert!(ops("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r1], [%rd1, {%r2, %r3}], [%r6];").is_ok());
    }

    #[test]
    fn setp_type_before_compare() {
        // `setp.u32.eq` (ptxas accepts the type first) = `setp.eq.u32`.
        assert_eq!(
            ops("setp.u32.eq %p1, %r1, 3;").unwrap(),
            ops("setp.eq.u32 %p1, %r1, 3;").unwrap()
        );
        assert_eq!(
            ops("setp.s32.lt %p1, %r1, %r2;").unwrap(),
            ops("setp.lt.s32 %p1, %r1, %r2;").unwrap()
        );
    }

    #[test]
    fn scoped_names_not_modifiers() {
        let s = flatten_scopes(
            "{\nmov.b32 %r9, 0;\n{ .reg .b32 lo; mad.lo.u32 lo, %r1, %r2, %r3; mov.b32 %r4, lo; }\n}",
        );
        assert!(s.contains("mad.lo.u32 lo__s0,"), "{s}");
        assert!(s.contains("%r4, lo__s0;"), "{s}");
        // A scoped range declaration renames its expanded names.
        let s = flatten_scopes(
            "{\nmov.b32 %r9, 0;\n{ .reg .u32 %t_<2>; .reg .pred %p<1>; setp.eq.u32 %p0, %t_1, 0; }\n}",
        );
        assert!(s.contains("%t___s0<2>") && s.contains("%p__s0<1>"), "{s}");
        assert!(s.contains("setp.eq.u32 %p__s00, %t___s01, 0"), "{s}");
    }

    /// recip_rn_f32 = RN(1 / c) (ptxas's folded div.full / div.approx constants).
    #[test]
    fn recip_constants() {
        for (c, r) in [
            (0x4040_0000u32, 0x3eaa_aaabu32),
            (0x3fb8_aa3b, 0x3f31_7218),
            (0x4b00_0000, 0x3400_0000),
            (0x7e80_0000, 0x0080_0000),
            (0x0080_0000, 0x7e80_0000),
            (0x4120_0000, 0x3dcc_cccd),
            (0xc0e0_0000, 0xbe12_4925),
            (0x3dcc_cccd, 0x4120_0000),
        ] {
            assert_eq!(super::recip_rn_f32(c), r, "{c:#x}");
        }
        // Against f64 (1 / c is never within 2^-53 relative of an f32 tie here).
        let mut x = 0x1234_5678u32;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let c = x & 0x807f_ffff | (1 + x % 252) << 23;
            let want = ((1.0 / f32::from_bits(c) as f64) as f32).to_bits();
            assert_eq!(super::recip_rn_f32(c), want, "{c:#x}");
        }
    }

    #[test]
    fn float_literals() {
        assert_eq!(float_lit("0.1", Some(32)), Some(Some(0x3dcc_cccd)));
        assert_eq!(float_lit("20.", Some(32)), Some(Some(0x41a0_0000)));
        assert_eq!(float_lit("-2.5", Some(32)), Some(Some(0xc020_0000)));
        assert_eq!(float_lit(".5e1", Some(32)), Some(Some(0x40a0_0000)));
        assert_eq!(
            float_lit("1.00000017881393432617", Some(32)),
            Some(Some(0x3f80_0002))
        );
        assert_eq!(
            float_lit("0.1", Some(64)),
            Some(Some(0x3fb9_9999_9999_999a))
        );
        assert_eq!(
            float_lit("0d3FB999999999999A", Some(32)),
            Some(Some(0x3dcc_cccd))
        );
        assert_eq!(float_lit("0f3DCCCCCD", Some(64)), None);
        assert_eq!(float_lit("0d3FB999999999999A", Some(64)), None);
        assert_eq!(float_lit("1.0", None), Some(None));
        assert_eq!(float_lit("12", Some(32)), None);
        assert_eq!(float_lit("0x10", Some(32)), None);
        assert!(ops("add.f32 %r1, %r2, 1.5;").is_ok());
        assert!(ops("add.s32 %r1, %r2, 1.5;").is_err());
    }

    #[test]
    fn packed_f32x2_forms() {
        let f2 = |l: &str| match &ops(l).unwrap()[..] {
            [Op::Ffma2 { a, b, c, neg, .. }] => (a.1, b.clone(), c.clone(), *neg),
            o => panic!("{l}: {o:?}"),
        };
        let pair = F2Lanes::Pair;
        let (a, b, c, neg) = f2("fma.rn.f32x2 %rd1, %rd2, %rd3, %rd4;");
        assert_eq!((a, b.1, c.1, neg), (pair, pair, pair, [false; 3]));
        // mul: a*b + (-0); add / sub: a*1 + (+-b).
        let (_, _, c, _) = f2("mul.rn.f32x2 %rd1, %rd2, %rd3;");
        assert_eq!(c, (Val::Imm(0x8000_0000), F2Lanes::Bcast));
        let (_, b, c, neg) = f2("add.rn.f32x2 %rd1, %rd2, %rd3;");
        assert_eq!(
            (b, c.1, neg),
            ((Val::Imm(0x3f80_0000), F2Lanes::Bcast), pair, [false; 3])
        );
        let (_, _, _, neg) = f2("sub.rn.f32x2 %rd1, %rd2, %rd3;");
        assert_eq!(neg, [false, false, true]);
        // No rounding modifier: .rn where ptxas cannot contract (a lone mul; an add of no
        // loose mul), refused where it can (a mul feeding an add, that add, fma).
        assert!(ops("mul.f32x2 %rd1, %rd2, %rd3;").is_ok());
        assert!(ops("add.f32x2 %rd1, %rd2, %rd3;").is_ok());
        assert!(ops("mul.f32x2 %rd1, %rd2, %rd3;\nmul.f32x2 %rd4, %rd1, %rd3;").is_ok());
        // PTX-FMAD (fmad2_pairs): a mul read only by adds / subs of its rounding mode (none =
        // .rn) is contracted into each, explicit .rn included; any other reader: none.
        let fused = |l: &str| match ops(l).unwrap().last() {
            Some(Op::Ffma2 { b, neg, rnd, .. }) if b.0 != Val::Imm(0x3f80_0000) => {
                Some((*neg, *rnd))
            }
            Some(Op::Ffma2 { .. }) => None,
            o => panic!("{l}: {o:?}"),
        };
        let no = [false; 3];
        for (l, want) in [
            (
                "mul.f32x2 %rd1, %rd2, %rd3;\nadd.rn.f32x2 %rd4, %rd1, %rd3;",
                Some((no, Rnd::Rn)),
            ),
            (
                "mul.rn.f32x2 %rd1, %rd2, %rd3;\nadd.f32x2 %rd4, %rd1, %rd3;",
                Some((no, Rnd::Rn)),
            ),
            (
                "mul.rn.f32x2 %rd1, %rd2, %rd3;\nadd.rn.f32x2 %rd4, %rd3, %rd1;",
                Some((no, Rnd::Rn)),
            ),
            (
                "mul.f32x2 %rd1, %rd2, %rd3;\nsub.f32x2 %rd4, %rd3, %rd1;",
                Some(([true, false, false], Rnd::Rn)),
            ),
            (
                "mul.rn.f32x2 %rd1, %rd2, %rd3;\nsub.rn.f32x2 %rd4, %rd1, %rd3;",
                Some(([false, false, true], Rnd::Rn)),
            ),
            (
                "mul.rz.f32x2 %rd1, %rd2, %rd3;\nadd.rz.f32x2 %rd4, %rd1, %rd3;",
                Some((no, Rnd::Rz)),
            ),
            (
                "mul.rn.f32x2 %rd1, %rd2, %rd3;\nadd.rz.f32x2 %rd4, %rd1, %rd3;",
                None,
            ),
            (
                "mul.rn.f32x2 %rd1, %rd2, %rd3;\nadd.rn.f32x2 %rd4, %rd1, %rd1;",
                None,
            ),
            (
                "mul.rn.f32x2 %rd1, %rd2, %rd3;\nst.global.b64 [%rd7], %rd1;\nadd.rn.f32x2 %rd4, %rd1, %rd3;",
                None,
            ),
            (
                "mul.f32x2 %rd1, %rd2, %rd3;\nmul.f32x2 %rd5, %rd1, %rd3;\nadd.f32x2 %rd4, %rd1, %rd3;",
                None,
            ),
        ] {
            assert_eq!(fused(l), want, "{l}");
        }
        let two = "mul.rn.f32x2 %rd1, %rd2, %rd3;\nadd.rn.f32x2 %rd4, %rd1, %rd3;\nadd.rn.f32x2 %rd5, %rd1, %rd2;";
        assert!(matches!(
            &ops(two).unwrap()[..],
            [
                _,
                Op::Ffma2 {
                    neg: [false, false, false],
                    ..
                },
                Op::Ffma2 { .. }
            ]
        ));
        // Two such muls: the first operand's is folded (b = %rd3 here, %rd2 swapped).
        let two_mul = |l: &str| match ops(l).unwrap().last() {
            Some(Op::Ffma2 { b, c, .. }) => (b.0.clone(), c.0.clone()),
            o => panic!("{l}: {o:?}"),
        };
        let (b1, c1) = two_mul(
            "mul.rn.f32x2 %rd1, %rd2, %rd3;\nmul.rn.f32x2 %rd5, %rd2, %rd2;\nadd.rn.f32x2 %rd4, %rd1, %rd5;",
        );
        let (b2, c2) = two_mul(
            "mul.rn.f32x2 %rd1, %rd2, %rd3;\nmul.rn.f32x2 %rd5, %rd2, %rd2;\nadd.rn.f32x2 %rd4, %rd5, %rd1;",
        );
        assert!(b1 != b2 && c1 != c2, "{b1:?} {b2:?}");
        // A debug label nothing branches to does not end the block.
        assert!(
            fused("mul.rn.f32x2 %rd1, %rd2, %rd3;\n$L__tmp1:\nadd.rn.f32x2 %rd4, %rd1, %rd3;")
                .is_some()
        );
        for l in [
            "mul.rn.f32x2 %rd1, %rd2, %rd3;\nmul.rn.f32x2 %rd5, %rd2, %rd2;\nadd.rn.f32x2 %rd4, %rd1, %rd5;\nadd.rn.f32x2 %rd6, %rd5, %rd3;",
            "mul.rn.f32x2 %rd1, %rd2, %rd3;\n@%p1 bra $L__tmp1;\n$L__tmp1:\nadd.rn.f32x2 %rd4, %rd1, %rd3;",
            "mul.rn.f32x2 %rd1, %rd2, %rd3;\n@%p1 add.rn.f32x2 %rd4, %rd1, %rd3;",
            "mul.rn.f32x2 %rd1, %rd2, %rd3;\nmov.b64 %rd2, 0;\nadd.rn.f32x2 %rd4, %rd1, %rd3;",
            "mul.rn.f32x2 %rd1, %rd2, %rd3;\nadd.rn.f32x2 %rd4, %rd1, %rd3;\nselp.b64 %rd5, %rd1, %rd3, %p1;",
        ] {
            assert!(ops(l).is_err(), "{l}");
        }
        // Unpacked halves read by scalar ops, an fma addend, a cross-lane repack into an add.
        assert!(
            ops("mul.f32x2 %rd1, %rd2, %rd3;\nmov.b64 {%r1, %r2}, %rd1;\nadd.f32 %r3, %r1, %r2;")
                .is_ok()
        );
        assert!(ops("mul.f32x2 %rd1, %rd2, %rd3;\nfma.rn.f32x2 %rd4, %rd2, %rd2, %rd1;").is_ok());
        assert!(
            ops(
                "mul.f32x2 %rd1, %rd2, %rd3;\nmov.b64 {_, %r2}, %rd1;\nmov.b64 %rd4, {%r2, %r3};\n\
             add.f32x2 %rd5, %rd4, %rd2;"
            )
            .is_ok()
        );
        // Same lane next to a value not from the product: no contraction (probes/f32x2_repack).
        assert!(
            ops(
                "mul.f32x2 %rd1, %rd2, %rd3;\nmov.b64 {%r1, _}, %rd1;\nmov.b64 %rd4, {%r1, %r3};\n\
             add.f32x2 %rd5, %rd4, %rd2;"
            )
            .is_ok()
        );
        // Halves repacked into a vector no add / sub.f32x2 reads (a loop-carried accumulator
        // scaled each iteration): the mul is FMUL2.
        assert!(
            ops(
                "mul.f32x2 %rd1, %rd1, %rd3;\nmov.b64 {%r1, %r2}, %rd1;\nmov.b32 %r3, %r2;\n\
             mov.b64 %rd1, {%r1, %r3};\nmul.f32x2 %rd5, %rd1, %rd2;\nst.global.b64 [%rd6], %rd5;"
            )
            .is_ok()
        );
        // .. but read by an add through the repack: refused.
        assert!(
            ops(
                "mul.f32x2 %rd1, %rd1, %rd3;\nmov.b64 {%r1, %r2}, %rd1;\nmov.b32 %r3, %r2;\n\
             mov.b64 %rd1, {%r1, %r3};\nadd.f32x2 %rd5, %rd1, %rd2;\nst.global.b64 [%rd6], %rd5;"
            )
            .is_err()
        );
        // The identity repack (directly, through a copy, or a second unpack) contracts: refused.
        for l in [
            "mov.b64 {%r1, %r2}, %rd1;\nmov.b64 %rd4, {%r1, %r2};",
            "mov.b64 {%r1, %r2}, %rd1;\nmov.b32 %r3, %r2;\nmov.b64 %rd4, {%r1, %r3};",
            "mov.b64 {%r1, _}, %rd1;\nmov.b64 {_, %r2}, %rd1;\nmov.b64 %rd4, {%r1, %r2};",
        ] {
            let k = format!("mul.f32x2 %rd1, %rd2, %rd3;\n{l}\nadd.f32x2 %rd5, %rd4, %rd2;");
            assert!(ops(&k).is_err(), "{k}");
        }
        // Scopes: after a directive line without `;`, and not a continuation operand list.
        assert!(ops(".loc 1 2 3\n{ .reg .b32 a; mov.b32 a, %r1; mov.b32 %r2, a; }").is_ok());
        assert!(ops("mov.b64\n {%r1, %r2}, %rd1;\n{ .reg .b32 a; mov.b32 a, %r1; }").is_ok());
        assert!(ops("mov.b32 %r1, %lanemask_lt;\nld.global.u32 %r2, [%rd1+-8];").is_ok());
        // bf16x2 / bf16: the same rule.
        assert!(ops("sub.bf16x2 %r1, %r2, %r3;").is_ok());
        assert!(ops("mul.bf16x2 %r1, %r2, %r3;\nst.global.b32 [%rd1], %r1;").is_ok());
        assert!(ops("mul.bf16x2 %r1, %r2, %r3;\nadd.bf16x2 %r4, %r1, %r3;").is_err());
        assert!(ops("mul.rn.bf16x2 %r1, %r2, %r3;\nadd.bf16x2 %r4, %r1, %r3;").is_ok());
        assert!(ops("fma.bf16x2 %r1, %r2, %r3, %r4;").is_err());
        // Through packs, prmt, shfl into mma / cvt: accepted; a pack into an add: refused.
        assert!(
            ops(
                ".reg .b16 %rs<8>;\nmul.bf16 %rs1, %rs2, %rs3;\nmov.b32 %r1, {%rs1, %rs4};\n\
             prmt.b32 %r2, %r1, %r3, 0x5410;\nmov.b32 {%rs5, %rs6}, %r2;\ncvt.f32.bf16 %r4, %rs5;"
            )
            .is_ok()
        );
        assert!(
            ops(
                ".reg .b16 %rs<8>;\nmul.bf16 %rs1, %rs2, %rs3;\nmov.b32 %r1, {%rs1, %rs4};\n\
             add.bf16x2 %r2, %r1, %r3;"
            )
            .is_err()
        );
        // A pack of two plain muls that is only stored (Triton bf16 scale).
        assert!(
            ops(
                ".reg .b16 %rs<8>;\nmul.bf16 %rs1, %rs2, %rs3;\nmul.bf16 %rs4, %rs5, %rs6;\n\
             mov.b32 %r1, {%rs1, %rs4};\nst.global.b32 [%rd1], %r1;"
            )
            .is_ok()
        );
        // Fail closed: other roundings, sat, immediates (.ftz: `packed_f32x2_ftz`).
        for l in [
            "fma.f32x2 %rd1, %rd2, %rd3, %rd4;",
            "mul.rn.sat.f32x2 %rd1, %rd2, %rd3;",
            "add.rn.f32x2 %rd1, %rd2, 0;",
        ] {
            assert!(
                ops(&format!("{l}\nst.global.f32 [%rd1], %r4;")).is_err(),
                "{l}"
            );
        }
    }

    #[test]
    fn cvt_x2_vectors_setmaxnreg_forms() {
        assert!(matches!(
            &ops("cvt.rn.bf16x2.f32 %r1, %r2, %r3;").unwrap()[..],
            [Op::F2fpBf16 { .. }]
        ));
        assert!(matches!(
            &ops("cvt.rn.f16x2.f32 %r1, %r2, %r3;").unwrap()[..],
            [Op::F2fpF16 { .. }]
        ));
        assert!(matches!(
            &ops("cvt.rz.f16x2.f32 %r1, %r2, %r3;").unwrap()[..],
            [Op::F2fpF16 { rz: true, .. }]
        ));
        assert!(ops("cvt.rz.bf16x2.f32 %r1, %r2, %r3;").is_err());
        assert!(ops("cvt.rn.relu.bf16x2.f32 %r1, %r2, %r3;").is_err());
        // Registers that are not consecutive: moves into the scratch block before a store,
        // out of it after a load.
        match &ops("st.shared.v4.b32 [%r1], {%r5, %r4, %r3, %r2};").unwrap()[..] {
            [
                Op::Mov { .. },
                Op::Mov { .. },
                Op::Mov { .. },
                Op::Mov { .. },
                Op::St {
                    bytes: 16,
                    v: Val::Reg(_, 4),
                    ..
                },
            ] => {}
            o => panic!("{o:?}"),
        }
        match &ops("ld.shared.v2.b32 {%r3, %r1}, [%r2];").unwrap()[..] {
            [
                Op::Ld { n: 2, bytes: 8, .. },
                Op::Mov { .. },
                Op::Mov { .. },
            ] => {}
            o => panic!("{o:?}"),
        }
        match &ops("ld.shared.v2.b64 {%rd1, %rd2}, [%r2];").unwrap()[..] {
            [
                Op::Ld {
                    n: 2, bytes: 16, ..
                },
            ] => {}
            o => panic!("{o:?}"),
        }
        assert!(ops("ld.shared.v4.b64 {%rd1, %rd2, %rd3, %rd4}, [%r2];").is_err());
        assert!(matches!(
            &ops("setmaxnreg.inc.sync.aligned.u32 232;").unwrap()[..],
            [Op::Nop]
        ));
        assert!(matches!(
            &ops("setmaxnreg.dec.sync.aligned.u32 24;").unwrap()[..],
            [Op::Nop]
        ));
        for l in [
            "setmaxnreg.inc.sync.aligned.u32 20;",
            "setmaxnreg.dec.sync.aligned.u32 100;",
            "setmaxnreg.inc.sync.aligned.u32 %r1;",
            "setmaxnreg.inc.u32 64;",
        ] {
            assert!(
                ops(&format!("{l}\nst.global.f32 [%rd1], %r4;")).is_err(),
                "{l}"
            );
        }
    }
}
