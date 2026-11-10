/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Binary front-end: the machine code of a serialized executable (the cubin analogue) placed
//! against its listings (port of tools/tpumap.py and the content placement of tools/kpairs.py),
//! scored op by op (`dis::op_match`), and lifted to the same TPU IR as `lift::lift_final`.
//!
//! Lifting renders every decoded op back into listing syntax and runs the listing lifter on
//! the result, so both fronts share one lowering. Registers, immediates, predicates, opcodes
//! and branch delays come from the machine code. ASSUME(TPU-L-BIN-DECODE) What the encoding does
//! not hold comes from the matched listing op: allocation and region symbols, the MRB slot of
//! a pop, the vmatprep dtype, the vxpose width, ext-op operands other than the vreg port,
//! halt messages and the operand-free pseudo ops. ASSUME(TPU-L-BIN-LISTING-INFO) Ops whose decode
//! disagrees with the listing are rendered from the decode and reported in the notes; listing
//! ops the decode lacks entirely keep their listing text. ASSUME(TPU-L-BIN-FALLBACK)

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_tpu_ir::Kernel;

use crate::dis;
use crate::dis::Canon;
use crate::dis::Dec;
use crate::exe;
use crate::exe::BranchEnc;
use crate::llo;
use crate::llo::RawBundle;
use crate::llo::RawOp;

/// Loader bundles before the entry bundle; the TLP listing starts at PROLOGUE + 1.
pub const PROLOGUE: usize = exe::PROLOGUE;

/// One listing program set (TLP, late init/fini, kernels whose dump id lies between this TLP
/// id and the next).
#[derive(Default, Clone)]
pub struct ProgramSet {
    pub tlp: Vec<RawBundle>,
    pub init: Vec<RawBundle>,
    pub fini: Vec<RawBundle>,
    pub kernels: Vec<(String, Vec<RawBundle>)>,
}

/// `<id>-<name>-<pass>-final_bundles.txt` -> (id, name).
fn split_name(f: &str) -> Option<(u64, String)> {
    let stem = f.strip_suffix("-final_bundles.txt")?;
    if f.contains("schedule-analysis") || f.contains("static-per-bundle") {
        return None;
    }
    let dash = stem.find('-')?;
    let id = stem[..dash].parse().ok()?;
    let rest = &stem[dash + 1..];
    let k = rest.rfind('-')?;
    if rest[k + 1..].is_empty() || !rest[k + 1..].bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((id, rest[..k].replace(['<', '>'], "")))
}

fn listing_files(dir: &Path) -> Result<Vec<(u64, String, std::path::PathBuf)>> {
    let mut v = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let f = p
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_string();
        if let Some((id, name)) = split_name(&f) {
            v.push((id, name, p));
        }
    }
    v.sort();
    Ok(v)
}

/// Program sets of a listing directory (tpumap.groups).
pub fn program_sets(dir: &Path) -> Result<Vec<ProgramSet>> {
    let parsed = listing_files(dir)?;
    let mut tlp: Vec<u64> = parsed
        .iter()
        .filter(|x| x.1 == "TLP")
        .map(|x| x.0)
        .collect();
    tlp.dedup();
    let fixed = |n: &str| matches!(n, "TLP" | "late-initialization" | "late-finalization");
    let mut out = Vec::new();
    for (k, &tid) in tlp.iter().enumerate() {
        let nxt = tlp.get(k + 1).copied().unwrap_or(u64::MAX);
        let mut s = ProgramSet::default();
        for (id, name, p) in &parsed {
            if *id == tid || (tid < *id && *id < nxt && !fixed(name)) {
                let bl = llo::parse(&std::fs::read_to_string(p)?);
                match name.as_str() {
                    "TLP" => s.tlp = bl,
                    "late-initialization" => s.init = bl,
                    "late-finalization" => s.fini = bl,
                    _ => s.kernels.push((name.clone(), bl)),
                }
            }
        }
        out.push(s);
    }
    Ok(out)
}

fn real_ops(b: &RawBundle) -> usize {
    b.ops.iter().filter(|o| !dis::no_encoding(&o.mn)).count()
}

fn is_call(b: &RawBundle) -> bool {
    b.ops.len() == 1 && b.ops[0].mn == "inlined_call"
}

fn hash_sig(v: &[String]) -> u64 {
    use std::hash::Hash;
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// Listing bundles in image order: (source listing, bundle). With the image `bs`, each TLP
/// call with arguments takes the kernel whose first 64 bundles best match the image there
/// (opcode multisets; unused kernels win ties); without it, dump order (tpumap.expand).
fn expand<'a>(
    s: &'a ProgramSet,
    bs: Option<&[[u8; 64]]>,
    start: usize,
) -> Vec<(&'a str, &'a RawBundle)> {
    let calls: Vec<usize> = (0..s.tlp.len()).filter(|&i| is_call(&s.tlp[i])).collect();
    let noarg: Vec<usize> = calls
        .iter()
        .copied()
        .filter(|&i| s.tlp[i].ops[0].args.is_empty())
        .collect();
    let witharg: Vec<usize> = calls
        .iter()
        .copied()
        .filter(|&i| !s.tlp[i].ops[0].args.is_empty())
        .collect();
    let mut order: BTreeMap<usize, (&str, &Vec<RawBundle>)> = BTreeMap::new();
    if let Some(&i) = noarg.first() {
        order.insert(i, ("init", &s.init));
    }
    if noarg.len() > 1 {
        order.insert(*noarg.last().expect("nonempty"), ("fini", &s.fini));
    }
    if bs.is_none() {
        for (&i, (name, bl)) in witharg.iter().zip(&s.kernels) {
            order.insert(i, (name.as_str(), bl));
        }
    }
    let ksig: Vec<Vec<u64>> = s
        .kernels
        .iter()
        .map(|(_, bl)| {
            bl.iter()
                .take(64)
                .map(|b| hash_sig(&dis::sig_listing(&b.ops)))
                .collect()
        })
        .collect();
    let mut remaining: Vec<bool> = vec![true; s.kernels.len()];
    let mut seq: Vec<(&str, &RawBundle)> = Vec::new();
    for (i, b) in s.tlp.iter().enumerate() {
        if let Some(bs) = bs
            && !order.contains_key(&i)
            && witharg.contains(&i)
        {
            let pos = start + seq.len();
            let mut best: Option<((f64, bool, i64), usize)> = None;
            for (k, sig) in ksig.iter().enumerate() {
                let n = sig.len().min(bs.len().saturating_sub(pos));
                let hit = (0..n)
                    .filter(|&j| sig[j] == hash_sig(&dis::sig_image(&bs[pos + j])))
                    .count();
                let key = (hit as f64 / n.max(1) as f64, remaining[k], -(k as i64));
                if best.as_ref().is_none_or(|b| key > b.0) {
                    best = Some((key, k));
                }
            }
            if let Some((_, k)) = best {
                remaining[k] = false;
                order.insert(i, (s.kernels[k].0.as_str(), &s.kernels[k].1));
            }
        }
        match order.get(&i) {
            Some((src, bl)) => seq.extend(bl.iter().map(|x| (*src, x))),
            None => seq.push(("TLP", b)),
        }
    }
    seq
}

/// A listing bundle paired with its machine bundle.
#[derive(Clone)]
pub struct Placed {
    pub src: String,
    pub bundle: RawBundle,
    pub raw: [u8; 64],
}

/// Align a listing directory with a code image (tpumap.align): the program set with the
/// lowest empty/non-empty disagreement rate; kernels placed by content when there are several.
/// ASSUME(TPU-L-BIN-PLACE)
pub fn align(img: &[[u8; 64]], dir: &Path) -> Result<Vec<Placed>> {
    let start = PROLOGUE + 1;
    let sets = program_sets(dir)?;
    let mut best: Option<(f64, usize)> = None;
    for (si, s) in sets.iter().enumerate() {
        let seq = expand(s, None, start);
        if start + seq.len() > img.len() {
            continue;
        }
        let bad = seq
            .iter()
            .enumerate()
            .filter(|(i, (_, b))| dis::is_empty(&img[start + i]) != (real_ops(b) == 0))
            .count();
        let r = bad as f64 / seq.len().max(1) as f64;
        if best.is_none_or(|b| r < b.0) {
            best = Some((r, si));
        }
    }
    let Some((_, si)) = best else {
        return Ok(Vec::new());
    };
    let s = &sets[si];
    let seq = expand(s, (s.kernels.len() > 1).then_some(img), start);
    Ok(seq
        .into_iter()
        .enumerate()
        .filter(|(i, _)| start + i < img.len())
        .map(|(i, (src, b))| Placed {
            src: src.to_string(),
            bundle: b.clone(),
            raw: img[start + i],
        })
        .collect())
}

/// Kernel listings of a Kilo/Pallas capture directory (`llo/`): name -> file, for names
/// containing one of the kernel keywords of tools/kpairs.py.
pub fn kernel_listings(dir: &Path) -> Result<BTreeMap<String, std::path::PathBuf>> {
    const KEYS: [&str; 7] = ["kilo", "topk", "rpa", "RPA", "gmm", "moe", "MoE"];
    Ok(listing_files(dir)?
        .into_iter()
        .filter(|(_, n, _)| KEYS.iter().any(|k| n.contains(k)))
        .map(|(_, n, p)| (n, p))
        .collect())
}

/// Image signatures for content placement.
pub fn image_sigs(img: &[[u8; 64]]) -> Vec<u64> {
    img.iter().map(|b| hash_sig(&dis::sig_image(b))).collect()
}

/// Offset of a kernel listing in the image by content (every 4th of its first 400 bundles;
/// ties go to the larger offset, as tools/kpairs.py) and the probe score (hits, probes).
pub fn place_by_content(bl: &[RawBundle], img_sig: &[u64]) -> (usize, usize, usize) {
    let sig: Vec<u64> = bl
        .iter()
        .take(400)
        .map(|b| hash_sig(&dis::sig_listing(&b.ops)))
        .collect();
    let ks: Vec<usize> = (0..sig.len()).step_by(4).collect();
    let n = img_sig.len().saturating_sub(bl.len()) + 1;
    let mut best = (0usize, 0usize);
    for o in 0..n.max(1) {
        let hit = ks
            .iter()
            .filter(|&&k| img_sig.get(o + k).is_some_and(|&s| s == sig[k]))
            .count();
        if (hit, o) >= best {
            best = (hit, o);
        }
    }
    (best.1, best.0, ks.len())
}

/// Bundles of kernel `name`: from the TLP alignment when it places the kernel, else by
/// content search.
pub fn place_kernel(img: &[[u8; 64]], dir: &Path, name: &str) -> Result<Vec<Placed>> {
    let al = align(img, dir).unwrap_or_default();
    let first = al.iter().position(|p| p.src == name);
    if let Some(i) = first {
        let v: Vec<Placed> = al[i..]
            .iter()
            .take_while(|p| p.src == name)
            .cloned()
            .collect();
        return Ok(v);
    }
    let f = listing_files(dir)?
        .into_iter()
        .rfind(|(_, n, _)| n == name)
        .ok_or_else(|| anyhow!("no final_bundles listing named {name} in {}", dir.display()))?;
    let bl = llo::parse(&std::fs::read_to_string(&f.2)?);
    let (o, hit, n) = place_by_content(&bl, &image_sigs(img));
    if hit * 2 < n {
        bail!("{name}: content placement probe {hit}/{n} (no confident location in the image)");
    }
    Ok(bl
        .into_iter()
        .enumerate()
        .filter(|(j, _)| o + j < img.len())
        .map(|(j, b)| Placed {
            src: name.to_string(),
            bundle: b,
            raw: img[o + j],
        })
        .collect())
}

// ---- rendering decoded ops in listing syntax ------------------------------------------

fn is_reg_tok(t: &str) -> bool {
    for p in ["vm", "v", "s", "p"] {
        if let Some(r) = t.strip_prefix(p)
            && !r.is_empty()
            && r.bytes().all(|c| c.is_ascii_digit())
        {
            return true;
        }
    }
    false
}

/// Canonical operand text -> listing syntax: registers get `%`, `[sflag:N]` becomes `$N`.
fn listing_syntax(a: &str) -> String {
    let mut s = a.to_string();
    while let Some(i) = s.find("[sflag:") {
        let Some(j) = s[i..].find(']') else { break };
        let n = s[i + 7..i + j].to_string();
        s.replace_range(i..i + j + 1, &format!("${n}"));
    }
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 4);
    let mut i = 0;
    while i < b.len() {
        let w = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
        if w(b[i]) && (i == 0 || !w(b[i - 1]) && b[i - 1] != b'%' && b[i - 1] != b'$') {
            let n = b[i..].iter().take_while(|&&c| w(c)).count();
            let t = &s[i..i + n];
            if is_reg_tok(t) {
                out.push('%');
            }
            out.push_str(t);
            i += n;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

fn pred_syntax(p: Option<&str>) -> String {
    match p {
        None => String::new(),
        Some(p) => match p.strip_prefix('!') {
            Some(q) => format!("(!%{q})"),
            None => format!("(%{p})"),
        },
    }
}

/// Listing spelling the lifter knows for a decoded mnemonic (decode-only ops).
fn lifter_spelling(d: &Dec) -> String {
    let (base, u) = match d.mn.strip_suffix(".u1") {
        Some(b) => (b, 1),
        None => (d.mn.as_str(), 0),
    };
    let low = d.word & 0x7F;
    match base {
        "vpow2.f32" => "vexp2.f32".into(),
        "vpack.c.bf16" => "vpack.c.f32.bf16".into(),
        "vunpack.c.l.bf16" => "vunpack.c.l.bf16.f32".into(),
        "vunpack.c.h.bf16" => "vunpack.c.h.bf16.f32".into(),
        "vlaneseq" => "vlaneseq.b32".into(),
        "vpop.xlu" => format!("vpop.xlane.xlu{u}"),
        // the MRB slot of a pop is not encoded: left unresolved (unlowered)
        "vpop.f32.mrb" => format!("vpop.f32.mrb[?].mxu{u}"),
        m if m.starts_with("vmatmul.mubr.") => format!("{m}.gmra.mrb[{}].mxu{u}", 2 * low),
        "vmatprep.mubr" => format!("vmatprep.mubr.bf16.mxu{u}"),
        m if m.starts_with("vmatprep.") || m.starts_with("vmatpush") => format!("{m}.mxu{u}"),
        m if m.starts_with("vrot.lane.") || m.contains(".xlane.") || m == "vperm" => {
            format!("{m}.xlu{u}")
        }
        m if m.starts_with("vxpose.") => format!("vxpose.xlu{u}.{}", &m[7..]),
        m => m.to_string(),
    }
}

fn is_symbolic(canon: &str) -> bool {
    canon.contains('A') || canon.contains('%') || canon.starts_with("target = $region")
}

/// Counters of one lift.
#[derive(Default, Clone, Debug)]
pub struct LiftStats {
    pub bundles: usize,
    /// Ops rendered from the machine code (matched and decode-only).
    pub decoded: usize,
    /// Decoded ops with no listing op (rendered from the decode alone).
    pub decode_only: usize,
    /// Operands / mnemonic parts taken from the listing.
    pub listing_info: usize,
    /// Listing ops without encoding (pseudo ops) carried over.
    pub pseudo: usize,
    /// Listing ops the decode disagrees with or lacks.
    pub disagree: Vec<String>,
}

fn ext_like(c: &Canon) -> bool {
    c.mn.starts_with("vmat")
        || c.mn.starts_with("vxpose")
        || c.mn.starts_with("vrot.lane")
        || c.mn.contains(".xlane")
        || c.mn.starts_with("vperm")
        || c.mn.starts_with("vset.pattern")
        || c.mn.starts_with("vbcast")
}

/// Render decoded op `d` matched with listing op `l` (canonical `lc`).
fn render_matched(d: &Dec, dc: &Canon, l: &RawOp, lc: &Canon, st: &mut LiftStats) -> String {
    let def = match &d.dst {
        Some(r) => format!("%{r} = "),
        None => l
            .def
            .as_ref()
            .map(|x| format!("{x} = "))
            .unwrap_or_default(),
    };
    let pred = pred_syntax(d.pred.as_deref());
    let mn = l.mn.as_str();
    if dis::canon_listing(&dis::py_op(&format!("{mn} x"))).mn != d.mn {
        st.listing_info += 1; // mnemonic spelling: units, MRB slot, dtype, build alias
    }
    let py = dis::py_op(&l.text);
    let mut largs: Vec<String> = py.args.clone();
    if py.pred.is_none()
        && let Some(a0) = largs.first()
        && a0.starts_with('(')
        && a0.ends_with(')')
        && a0.contains(", %")
    {
        largs.remove(0); // `(p, %old)` predicated-move form: the predicate is rendered
    }
    let args: Vec<String> = if mn.starts_with("dma.") {
        render_dma(dc, &largs, st)
    } else if ext_like(lc) || lc.mn.starts_with("vpop.") {
        // operands other than the vreg port come from the listing
        st.listing_info += 1;
        let v = dc.args.first().cloned();
        let mut done = false;
        largs
            .iter()
            .map(|a| {
                let c = dis::canon_listing(&dis::py_op(&format!("m {a}")))
                    .args
                    .join(",");
                if !done
                    && let Some(v) = &v
                    && c.starts_with('v')
                    && is_reg_tok(&c)
                {
                    done = true;
                    return format!("%{v}");
                }
                a.clone()
            })
            .collect()
    } else if lc.args.len() == dc.args.len() {
        let swapped = lc.args.len() == 2
            && lc.args[0] == dc.args[1]
            && lc.args[1] == dc.args[0]
            && lc.args[0] != lc.args[1];
        (0..dc.args.len())
            .map(|i| {
                if !swapped && is_symbolic(&lc.args[i]) {
                    st.listing_info += 1;
                    lc.raw[i].clone()
                } else if !swapped
                    && lc.mn == "sld"
                    && lc.args[i].bytes().all(|c| c.is_ascii_digit())
                {
                    st.listing_info += 1; // scalar_parameter_address N
                    lc.raw[i].clone()
                } else {
                    listing_syntax(&dc.args[i])
                }
            })
            .collect()
    } else {
        // sbr.rel, vtrace, sfence, pops: operands the encoding prints differently
        st.listing_info += 1;
        lc.raw.clone()
    };
    let args = args.join(", ");
    let sep = if pred.is_empty() || args.is_empty() {
        ""
    } else {
        ", "
    };
    format!("{def}{mn} {pred}{sep}{args}")
        .trim_end()
        .to_string()
}

/// DMA operands: listing names, machine values for every encoded non-symbolic operand.
fn render_dma(dc: &Canon, largs: &[String], st: &mut LiftStats) -> Vec<String> {
    let dmap: HashMap<&str, &str> = dc.args.iter().filter_map(|a| a.split_once('=')).collect();
    let (src, dst) = dc.mn[4..].split_once("_to_").unwrap_or(("", ""));
    largs
        .iter()
        .map(|a| {
            let Some((k, v)) = a.split_once('=') else {
                return a.clone();
            };
            let key = match k {
                "src_address" => src,
                "host_iova_low" => dst,
                _ => k,
            };
            let vc = dis::canon_listing(&dis::py_op(&format!("m {v}")))
                .args
                .join(",");
            match dmap.get(key) {
                Some(dv) if !is_symbolic(&vc) => format!("{k}={}", listing_syntax(dv)),
                _ => {
                    st.listing_info += 1;
                    a.clone()
                }
            }
        })
        .collect()
}

/// Render a decoded op without a listing counterpart.
fn render_decoded(d: &Dec) -> String {
    let def = d
        .dst
        .as_ref()
        .map(|r| format!("%{r} = "))
        .unwrap_or_default();
    let pred = pred_syntax(d.pred.as_deref());
    let mn = lifter_spelling(d);
    let args: Vec<String> = if d.mn.starts_with("vxpose.") {
        vec![format!(
            "vx=%{}",
            d.srcs.first().map(String::as_str).unwrap_or("v0")
        )]
    } else if d.mn == "sbr.rel" {
        let t: i64 = d.srcs[0].trim_start_matches("target=").parse().unwrap_or(0);
        vec![format!("target = $region@{t}")]
    } else {
        d.srcs.iter().map(|a| listing_syntax(a)).collect()
    };
    let args = args.join(", ");
    let sep = if pred.is_empty() || args.is_empty() {
        ""
    } else {
        ", "
    };
    format!("{def}{mn} {pred}{sep}{args}")
        .trim_end()
        .to_string()
}

/// Rendered listing text of one placed bundle plus its branch encoding.
fn render_bundle(p: &Placed, st: &mut LiftStats) -> (Vec<String>, Option<BranchEnc>) {
    let decs = dis::decode(&p.raw);
    let dcs: Vec<Canon> = decs.iter().map(dis::canon_decoded).collect();
    let enc_ops: Vec<&RawOp> = p
        .bundle
        .ops
        .iter()
        .filter(|o| !dis::no_encoding(&o.mn))
        .collect();
    let lcs: Vec<Canon> = enc_ops
        .iter()
        .map(|o| dis::canon_listing(&dis::py_op(&o.text)))
        .collect();
    let m = dis::match_bundle(&lcs, &dcs);
    let mut used = vec![false; decs.len()];
    for x in m.iter().flatten() {
        for &i in x {
            used[i] = true;
        }
    }
    let mut out = Vec::new();
    let mut ei = 0;
    for o in &p.bundle.ops {
        if dis::no_encoding(&o.mn) {
            st.pseudo += 1;
            out.push(o.text.clone());
            continue;
        }
        let (lc, hit) = (&lcs[ei], &m[ei]);
        ei += 1;
        match hit {
            Some(ix) if ix.len() == 2 => {
                // scalar_select: two complementary predicated smov in the machine code
                for &i in ix {
                    st.decoded += 1;
                    out.push(render_decoded(&decs[i]));
                }
            }
            Some(ix) => {
                st.decoded += 1;
                out.push(render_matched(&decs[ix[0]], &dcs[ix[0]], o, lc, st));
            }
            None => {
                // a leftover decoded op with the same mnemonic: the machine's version
                let alt = (0..decs.len()).find(|&i| {
                    !used[i]
                        && dis::untyped(dis::build_equiv(&dcs[i].mn))
                            == dis::untyped(dis::build_equiv(&lc.mn))
                });
                match alt {
                    Some(i) => {
                        used[i] = true;
                        st.decoded += 1;
                        st.disagree.push(format!(
                            "{:#x}: listing `{}` vs decoded `{}`",
                            p.bundle.addr,
                            o.text,
                            dis::fmt(&decs[i])
                        ));
                        out.push(render_matched(&decs[i], &dcs[i], o, lc, st));
                    }
                    None => {
                        st.disagree.push(format!(
                            "{:#x}: listing `{}` not in the machine code; listing text kept",
                            p.bundle.addr, o.text
                        ));
                        out.push(o.text.clone());
                    }
                }
            }
        }
    }
    for (i, d) in decs.iter().enumerate() {
        if !used[i] {
            st.decoded += 1;
            st.decode_only += 1;
            out.push(render_decoded(d));
        }
    }
    let enc = decs.iter().find(|d| d.mn == "sbr.rel").map(|d| BranchEnc {
        y: d.srcs[1].trim_start_matches("delay=").parse().unwrap_or(0),
        imm: d.srcs[0].trim_start_matches("target=").parse().unwrap_or(0),
    });
    (out, enc)
}

fn bundle_line(b: &RawBundle, ops: &[String], extra_comment: Option<&str>) -> String {
    let mut s = format!(
        "  {:#x} {}:  {}{{ {} }}",
        b.addr,
        b.tag.as_deref().unwrap_or(""),
        ">".repeat(b.depth as usize),
        ops.join("  ;;  ")
    );
    for c in b.comments.iter().map(String::as_str).chain(extra_comment) {
        s.push_str(&format!(" /* {c} */"));
    }
    s
}

/// The listing context of a binary lift.
pub struct ListingCtx<'a> {
    pub llo_dir: &'a Path,
    /// Listing name of the kernel (`vec_loop.1`).
    pub kernel: &'a str,
    pub final_text: &'a str,
    pub original_text: &'a str,
}

/// The rendered `final_bundles`-syntax text of a kernel's machine code, its branch encodings
/// (listing address -> encoding) and the lift counters.
pub fn binary_listing(
    exe_bytes: &[u8],
    ctx: &ListingCtx,
) -> Result<(String, BTreeMap<u32, BranchEnc>, LiftStats)> {
    let img = exe::code_image(exe_bytes)?;
    let placed = place_kernel(&img, ctx.llo_dir, ctx.kernel)?;
    let want = llo::parse(ctx.final_text);
    if placed.len() != want.len() {
        bail!(
            "{}: placed {} bundles, listing has {}",
            ctx.kernel,
            placed.len(),
            want.len()
        );
    }
    let mut st = LiftStats::default();
    let mut lines = Vec::new();
    let mut enc = BTreeMap::new();
    for p in &placed {
        let (ops, e) = render_bundle(p, &mut st);
        if let Some(e) = e {
            enc.insert(p.bundle.addr, e);
        }
        lines.push(bundle_line(&p.bundle, &ops, None));
        st.bundles += 1;
    }
    Ok((lines.join("\n") + "\n", enc, st))
}

/// Lift the machine code of kernel `ctx.kernel` in executable `exe_bytes` to the TPU IR
/// (same IR as `lift::lift_final`). Notes carry the lift counters and every listing/decode
/// disagreement.
pub fn lift_binary(exe_bytes: &[u8], ctx: &ListingCtx) -> Result<Kernel> {
    let (text, enc, st) = binary_listing(exe_bytes, ctx)?;
    let mut k = crate::lift::lift_final_exe(ctx.kernel, &text, ctx.original_text, Some(&enc));
    k.assumptions.push("TPU-L-BIN-DECODE".into());
    if st.listing_info + st.pseudo > 0 {
        k.assumptions.push("TPU-L-BIN-LISTING-INFO".into());
    }
    if !st.disagree.is_empty() {
        k.assumptions.push("TPU-L-BIN-FALLBACK".into());
    }
    k.notes.push(format!(
        "binary front: {} bundles, {} ops from the machine code ({} without a listing op), {} listing-sourced parts, {} pseudo ops, {} disagreements",
        st.bundles,
        st.decoded,
        st.decode_only,
        st.listing_info,
        st.pseudo,
        st.disagree.len()
    ));
    k.notes.extend(st.disagree.iter().take(20).cloned());
    Ok(k)
}

/// Lift a corpus kernel directory's executable.
pub fn lift_kernel_dir(kd: &crate::corpus::KernelDir) -> Result<Kernel> {
    let exe = kd
        .exe
        .as_ref()
        .ok_or_else(|| anyhow!("{}: no executable", kd.name))?;
    let name = exe::listing_name(&kd.final_path).ok_or_else(|| anyhow!("listing name"))?;
    let bytes = std::fs::read(exe)?;
    let mut k = lift_binary(
        &bytes,
        &ListingCtx {
            llo_dir: &kd.llo_dir,
            kernel: &name,
            final_text: &kd.final_text,
            original_text: &kd.original_text,
        },
    )?;
    k.name = kd.name.clone();
    Ok(k)
}

/// Binary-only rendering of a whole image (no listing): entry bundle onwards, branch targets
/// as synthetic regions. Symbols, operand declarations and pop slots are absent (unmodeled).
pub fn image_listing(exe_bytes: &[u8]) -> Result<String> {
    let img = exe::code_image(exe_bytes)?;
    let start = PROLOGUE + 1;
    let end = img
        .iter()
        .rposition(|b| !dis::is_empty(b))
        .map_or(start, |i| i + 1);
    let mut targets = BTreeMap::new();
    let mut rows = Vec::new();
    for (i, raw) in img.iter().enumerate().take(end).skip(start) {
        let decs = dis::decode(raw);
        let addr = (i - start) as u32;
        let mut ops = Vec::new();
        for d in &decs {
            let mut t = render_decoded(d);
            if d.mn == "sbr.rel" {
                let off: i64 = d.srcs[0].trim_start_matches("target=").parse().unwrap_or(0);
                let tgt = addr as i64 + off;
                targets.insert(tgt, ());
                t = t.replace(&format!("$region@{off}"), &format!("$region{tgt}"));
            }
            ops.push(t);
        }
        rows.push((addr, ops));
    }
    Ok(rows
        .iter()
        .map(|(a, ops)| {
            let b = RawBundle {
                addr: *a,
                ..Default::default()
            };
            let c = targets
                .contains_key(&(*a as i64))
                .then(|| format!("Start region {a}"));
            bundle_line(&b, ops, c.as_deref())
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n")
}

/// Run the binary front of a corpus kernel once and compare with the hardware outputs
/// (`corpus::run_kernel` for the machine code).
pub fn run_binary(
    kd: &crate::corpus::KernelDir,
    policy: crate::machine::DmaPolicy,
    num: &'static dyn crate::num::Numerics,
) -> Result<crate::corpus::RunResult> {
    use crate::corpus::OutCmp;
    use crate::corpus::RunResult;
    let kernel = std::sync::Arc::new(lift_kernel_dir(kd)?);
    let mut m = kd.machine(kernel.clone(), num)?;
    let mut res = RunResult {
        kernel: kd.name.clone(),
        front: "binary".into(),
        status: "ok".into(),
        error: String::new(),
        bundles: kernel.bundles.len(),
        steps: 0,
        exact: None,
        outputs: Vec::new(),
        arrays: Vec::new(),
    };
    let r = m.run(policy);
    res.steps = m.cores.iter().map(|c| c.steps).sum();
    if let Err(e) = r {
        res.status = e.status().into();
        res.error = e.to_string().chars().take(300).collect();
        return Ok(res);
    }
    let outs = kd.outputs(&m)?;
    let mut exact = true;
    let mut per_core = vec![0usize; kd.ncores()];
    for (c, got) in &outs {
        let oi = per_core[*c];
        per_core[*c] += 1;
        let (want, nondet) = kd.expected(*c, oi)?;
        let bad = got
            .data
            .iter()
            .zip(&want.data)
            .filter(|(a, b)| a != b)
            .count()
            + got.data.len().abs_diff(want.data.len());
        exact &= bad == 0 || nondet;
        res.outputs.push(OutCmp {
            core: *c,
            out: oi,
            mismatch: bad,
            n: want.data.len(),
            nondet,
        });
    }
    res.exact = Some(exact);
    res.arrays = outs;
    Ok(res)
}

// ---- scoring --------------------------------------------------------------------------

/// Decoder agreement over a set of placed bundles (unique (bundle, listing text) pairs).
#[derive(Default, Clone, Debug)]
pub struct Score {
    pub bundles: usize,
    pub bundles_exact: usize,
    pub ops: usize,
    pub ops_matched: usize,
    pub extra: usize,
    pub miss: BTreeMap<String, usize>,
}

impl Score {
    pub fn add(&mut self, o: &Score) {
        self.bundles += o.bundles;
        self.bundles_exact += o.bundles_exact;
        self.ops += o.ops;
        self.ops_matched += o.ops_matched;
        self.extra += o.extra;
        for (k, v) in &o.miss {
            *self.miss.entry(k.clone()).or_default() += v;
        }
    }

    pub fn pct(&self) -> f64 {
        100.0 * self.ops_matched as f64 / self.ops.max(1) as f64
    }

    /// Top unmatched mnemonics.
    pub fn top_miss(&self, n: usize) -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = self.miss.iter().map(|(k, v)| (k.clone(), *v)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }
}

/// Deduplication key of a scored pair.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Dedup {
    /// Bytes + canonical listing ops of encoded ops (tpudis.py --test).
    Canonical,
    /// Bytes + op texts (tools/kpairs.py).
    Text,
}

/// Score placed pairs; `seen` (shared across calls) deduplicates by `mode`.
pub fn score(
    pairs: &[([u8; 64], &RawBundle)],
    seen: &mut std::collections::HashSet<([u8; 64], Vec<String>)>,
    mode: Dedup,
) -> Score {
    let mut s = Score::default();
    for (raw, b) in pairs {
        let ops: Vec<String> = match mode {
            Dedup::Text => b.ops.iter().map(|o| o.text.clone()).collect(),
            Dedup::Canonical => b
                .ops
                .iter()
                .filter(|o| !dis::no_encoding(&o.mn))
                .map(|o| {
                    let c = dis::canon_listing(&dis::py_op(&o.text));
                    format!("{:?}", (c.mn, c.dst, c.args, c.pred))
                })
                .collect(),
        };
        let key = (*raw, ops);
        if !seen.insert(key) {
            continue;
        }
        let (n, ok, extra, miss) = dis::score_bundle(raw, &b.ops);
        s.bundles += 1;
        s.ops += n;
        s.ops_matched += ok;
        s.extra += extra;
        if ok == n && extra == 0 {
            s.bundles_exact += 1;
        }
        for m in miss {
            *s.miss.entry(m).or_default() += 1;
        }
    }
    s
}

/// Score a differential-corpus directory (`rt/<v>.bin` + `llo/<v>/`), as `tpudis.py --test`.
pub fn score_corpus_dir(dir: &Path) -> Result<Score> {
    let mut total = Score::default();
    let mut seen = std::collections::HashSet::new();
    let mut rts: Vec<_> = std::fs::read_dir(dir.join("rt"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect();
    rts.sort();
    for rt in rts {
        let v = rt
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        let Ok(img) = std::fs::read(&rt)
            .map_err(anyhow::Error::from)
            .and_then(|b| exe::code_image(&b))
        else {
            continue;
        };
        let Ok(al) = align(&img, &dir.join("llo").join(&v)) else {
            continue;
        };
        let pairs: Vec<([u8; 64], &RawBundle)> = al.iter().map(|p| (p.raw, &p.bundle)).collect();
        total.add(&score(&pairs, &mut seen, Dedup::Canonical));
    }
    Ok(total)
}

/// Score a Kilo/Pallas capture directory (`exe.rt.bin` or `exe0.rt.bin` + `llo/`): every
/// kernel listing placed by content (tools/kpairs.py), unique pairs per directory.
pub fn score_capture_dir(dir: &Path) -> Result<Score> {
    let rt = ["exe.rt.bin", "exe0.rt.bin"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("{}: no executable", dir.display()))?;
    let img = exe::code_image(&std::fs::read(rt)?)?;
    let sigs = image_sigs(&img);
    let mut seen = std::collections::HashSet::new();
    let mut total = Score::default();
    for (_, f) in kernel_listings(&dir.join("llo"))? {
        let bl = llo::parse(&std::fs::read_to_string(&f)?);
        let (o, _, _) = place_by_content(&bl, &sigs);
        let pairs: Vec<([u8; 64], &RawBundle)> = bl
            .iter()
            .enumerate()
            .filter(|(j, _)| o + j < img.len())
            .map(|(j, b)| (img[o + j], b))
            .collect();
        total.add(&score(&pairs, &mut seen, Dedup::Text));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syntax() {
        assert_eq!(
            listing_syntax("[vmem:[s6 + $0x70] sm:0xf] vst_source=v43"),
            "[vmem:[%s6 + $0x70] sm:0xf] vst_source=%v43"
        );
        assert_eq!(listing_syntax("vm=vm7"), "vm=%vm7");
        assert_eq!(listing_syntax("[sflag:5]"), "$5");
        assert_eq!(pred_syntax(Some("!p3")), "(!%p3)");
        assert_eq!(
            split_name("17-vec_loop.1-02-final_bundles.txt"),
            Some((17, "vec_loop.1".into()))
        );
        assert_eq!(
            split_name("17-TLP-01-schedule-analysis_final_bundles.txt"),
            None
        );
    }
}
