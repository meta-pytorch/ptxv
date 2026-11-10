/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Branch delays from a serialized TPU executable (port of tools/tpuexe.py, tpumap.py and the
//! sbr.rel decode of tpubranch.py; layout in fb/targets/tpu/probes/binary/FINDINGS.md).
//!
//! The listing never prints a branch's delay; the machine code does: `sbr.rel` (scalar op 0,
//! x = 5) carries y (6 bits at S0 bit 472 / S1 bit 448) = number of bundles executed after the
//! branch before it redirects, and imm0 (20 bits at 423, signed) = target - pc.
//! ASSUME(TPU-E-BR-DELAY)

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;

use crate::llo;
use crate::llo::RawBundle;

const BLOCK: usize = 1 << 16;
/// Loader/prologue bundles before the entry bundle (libtpu 0.0.50).
pub const PROLOGUE: usize = 416;

fn varint(b: &[u8], i: &mut usize) -> Result<u64> {
    let (mut r, mut s) = (0u64, 0u32);
    loop {
        let c = *b.get(*i).ok_or_else(|| anyhow!("varint overrun"))?;
        *i += 1;
        if s < 64 {
            r |= ((c & 0x7F) as u64) << s;
        }
        s += 7;
        if c < 0x80 {
            return Ok(r);
        }
    }
}

/// Length-delimited fields `(field, bytes)` of a protobuf message (other wire types skipped);
/// errors if the bytes are not a well-formed message.
fn pb_fields(b: &[u8]) -> Result<Vec<(u64, &[u8])>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let key = varint(b, &mut i)?;
        let (f, wt) = (key >> 3, key & 7);
        if f == 0 {
            bail!("field 0");
        }
        match wt {
            0 => {
                varint(b, &mut i)?;
            }
            1 => i += 8,
            2 => {
                let n = varint(b, &mut i)? as usize;
                if i + n > b.len() {
                    bail!("overrun");
                }
                out.push((f, &b[i..i + n]));
                i += n;
            }
            5 => i += 4,
            _ => bail!("wire type {wt}"),
        }
    }
    if i != b.len() {
        bail!("trailing bytes");
    }
    Ok(out)
}

fn pb_get<'a>(m: &'a [u8], path: &[u64]) -> Result<Vec<&'a [u8]>> {
    let mut cur = vec![m];
    for &f in path {
        let mut nxt = Vec::new();
        for x in cur {
            nxt.extend(
                pb_fields(x)?
                    .into_iter()
                    .filter(|(g, _)| *g == f)
                    .map(|(_, v)| v),
            );
        }
        cur = nxt;
    }
    Ok(cur)
}

fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().expect("8 bytes"))
}

fn riegeli_start(b: &[u8]) -> Result<usize> {
    let mut i = b.windows(9).position(|w| w == b"pjrt_ifrt").unwrap_or(0);
    while i + 64 <= b.len() {
        if u64_at(b, i + 8) == 0 && u64_at(b, i + 16) == 0x40 && b[i + 48] == 0x73 {
            return Ok(i);
        }
        i += 1;
    }
    bail!("no Riegeli signature found")
}

/// Records of the Riegeli file inside a serialized executable (uncompressed simple chunks).
fn records(b: &[u8]) -> Result<Vec<Vec<u8>>> {
    let start = riegeli_start(b)?;
    let f = &b[start..];
    let mut l = Vec::with_capacity(f.len());
    let mut blk = 0;
    while blk < f.len() {
        let end = (blk + BLOCK).min(f.len());
        if blk + 24 < end {
            l.extend_from_slice(&f[blk + 24..end]);
        }
        blk += BLOCK;
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i + 40 <= l.len() {
        let size = u64_at(&l, i + 8) as usize;
        let tn = u64_at(&l, i + 24);
        let ctype = (tn & 0xFF) as u8;
        let data = &l[i + 40..(i + 40 + size).min(l.len())];
        if ctype == 0x72 && size > 0 {
            if data[0] != 0 {
                bail!("compressed Riegeli chunk not supported");
            }
            let mut j = 1;
            let sl = varint(data, &mut j)? as usize;
            let mut k = j;
            let mut sizes = Vec::new();
            while k < j + sl {
                sizes.push(varint(data, &mut k)? as usize);
            }
            let mut pos = j + sl;
            for n in sizes {
                out.push(data[pos..(pos + n).min(data.len())].to_vec());
                pos += n;
            }
        } else if !matches!(ctype, 0x73 | 0x72 | 0x6D | 0x70) {
            break;
        }
        i += 40 + size;
    }
    Ok(out)
}

/// The code image: 64-byte bundles of ds_executable.core_program.memory_segments[0].
pub fn code_bundles(exe: &[u8]) -> Result<Vec<[u8; 64]>> {
    for r in records(exe)? {
        // TpuExecutableProto.1 ds_executable .1 core_program .8 memory_segments .3 initialized_data
        if let Ok(segs) = pb_get(&r, &[1, 1, 8, 3])
            && let Some(s) = segs.first()
        {
            return Ok(s
                .chunks_exact(64)
                .map(|c| c.try_into().expect("64 bytes"))
                .collect());
        }
    }
    bail!("no code segment found")
}

/// Varint-length-prefixed protos (libtpu-meta nightly layout: no Riegeli file; message 1 is the
/// core program).
fn framed_messages(b: &[u8]) -> Result<Vec<&[u8]>> {
    let (mut out, mut i) = (Vec::new(), 0);
    while i < b.len() {
        let n = varint(b, &mut i)? as usize;
        if i + n > b.len() {
            bail!("not a framed message sequence");
        }
        out.push(&b[i..i + n]);
        i += n;
    }
    Ok(out)
}

/// The code image of a serialized executable of either libtpu build: 0.0.50 (pjrt_ifrt header
/// and Riegeli, field path 1.1.8.3) or the libtpu-meta nightly (framed protos, path 8.3 of the
/// core program). Port of tools/tpuexe.py `bundles`.
pub fn code_image(exe: &[u8]) -> Result<Vec<[u8; 64]>> {
    if riegeli_start(exe).is_ok() {
        return code_bundles(exe);
    }
    for m in framed_messages(exe)?.into_iter().skip(1) {
        let Ok(segs) = pb_get(m, &[8, 3]) else {
            continue;
        };
        if let Some(s) = segs
            .into_iter()
            .find(|s| s.len() % 64 == 0 && s.len() >= 64 * 16)
        {
            return Ok(s
                .chunks_exact(64)
                .map(|c| c.try_into().expect("64 bytes"))
                .collect());
        }
    }
    bail!("no code segment found")
}

// ------------------------------------------------------------------ listing alignment

const PSEUDO: &[&str] = &[
    "inlined_call_operand",
    "int_to_ptr",
    "compiler-scheduling-barrier",
    "inlined_call",
    "sphi",
    "vphi",
];

fn real_ops(b: &RawBundle) -> usize {
    b.ops
        .iter()
        .filter(|o| !PSEUDO.iter().any(|p| o.mn.starts_with(p)))
        .count()
}

fn empty(raw: &[u8; 64]) -> bool {
    raw[..63].iter().all(|&x| x == 0) && raw[63] & !0x02 == 0x50
}

fn is_call(b: &RawBundle) -> bool {
    b.ops.len() == 1 && b.ops[0].mn == "inlined_call"
}

#[derive(Default)]
struct ProgramSet {
    tlp: Vec<RawBundle>,
    init: Vec<RawBundle>,
    fini: Vec<RawBundle>,
    kernels: Vec<(String, Vec<RawBundle>)>,
}

/// Program sets of a listing directory: TLP/late-* (dumped as `<late-*>`) sharing an id plus the kernels whose id
/// lies between that id and the next set's.
fn groups(dir: &Path) -> Result<Vec<ProgramSet>> {
    let mut parsed: Vec<(u64, String, std::path::PathBuf)> = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        let f = p
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_string();
        if !f.ends_with("-final_bundles.txt")
            || f.contains("schedule-analysis")
            || f.contains("static-per-bundle")
        {
            continue;
        }
        let stem = &f[..f.len() - "-final_bundles.txt".len()];
        let Some(dash) = stem.find('-') else { continue };
        let Ok(id) = stem[..dash].parse::<u64>() else {
            continue;
        };
        let rest = &stem[dash + 1..];
        let Some(k) = rest.rfind('-') else { continue };
        if !rest[k + 1..].bytes().all(|c| c.is_ascii_digit()) {
            continue;
        }
        parsed.push((id, rest[..k].replace(['<', '>'], ""), p));
    }
    parsed.sort();
    let mut tlp: Vec<u64> = parsed
        .iter()
        .filter(|x| x.1 == "TLP")
        .map(|x| x.0)
        .collect();
    tlp.dedup();
    let mut out = Vec::new();
    for (k, &tid) in tlp.iter().enumerate() {
        let nxt = tlp.get(k + 1).copied().unwrap_or(u64::MAX);
        let mut s = ProgramSet::default();
        for (id, name, p) in &parsed {
            let fixed = matches!(
                name.as_str(),
                "TLP" | "late-initialization" | "late-finalization"
            );
            if *id == tid || (tid < *id && *id < nxt && !fixed) {
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

/// The TLP with each `inlined_call` bundle replaced by the program it calls:
/// [(source listing, bundle)].
fn expand(s: &ProgramSet) -> Vec<(&str, &RawBundle)> {
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
    for (&i, (name, bl)) in witharg.iter().zip(&s.kernels) {
        order.insert(i, (name.as_str(), bl));
    }
    let mut seq = Vec::new();
    for (i, b) in s.tlp.iter().enumerate() {
        match order.get(&i) {
            Some((src, bl)) => seq.extend(bl.iter().map(|x| (*src, x))),
            None => seq.push(("TLP", b)),
        }
    }
    seq
}

fn field(x: &[u8; 64], o: usize, w: usize) -> u64 {
    let mut v = 0u64;
    for k in 0..w {
        let bit = o + k;
        v |= (((x[bit / 8] >> (bit % 8)) & 1) as u64) << k;
    }
    v
}

/// One `sbr.rel` decoded from the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchEnc {
    /// Bundles executed after the branch before it redirects.
    pub y: u8,
    /// Target bundle offset from the branch.
    pub imm: i32,
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for p in permutations(n - 1) {
        for i in 0..=p.len() {
            let mut q = p.clone();
            q.insert(i, n - 1);
            out.push(q);
        }
    }
    out.sort();
    out
}

/// ASSUME(TPU-L-EXE-ALIGN)
/// Branch encodings of listing `kernel` (its bundle address -> encoding), from executable
/// `exe` aligned with the listing set in `llo_dir`. The alignment tries every program set
/// and every call order of its kernels (dump order is not always call order: remote_sig) and
/// keeps the one with the fewest empty/non-empty disagreements. Errors unless the kernel's own
/// bundles align exactly. Only bundles holding an `sbr.rel` in the listing are reported.
pub fn branch_delays(exe: &[u8], llo_dir: &Path, kernel: &str) -> Result<BTreeMap<u32, BranchEnc>> {
    let bs = code_image(exe)?;
    let start = PROLOGUE + 1;
    let mut sets = groups(llo_dir)?;
    let mut best: Option<(usize, usize, Vec<(String, RawBundle, [u8; 64])>)> = None;
    for s in &mut sets {
        if !s.kernels.iter().any(|(n, _)| n == kernel) {
            continue;
        }
        let orig = std::mem::take(&mut s.kernels);
        let perms = if orig.len() <= 6 {
            permutations(orig.len())
        } else {
            vec![(0..orig.len()).collect()]
        };
        for perm in perms {
            s.kernels = perm.iter().map(|&i| orig[i].clone()).collect();
            let seq = expand(s);
            if start + seq.len() > bs.len() {
                continue;
            }
            let mis = |(i, (_, b)): (usize, &(&str, &RawBundle))| {
                empty(&bs[start + i]) != (real_ops(b) == 0)
            };
            let bad = seq.iter().enumerate().filter(|x| mis(*x)).count();
            if best.as_ref().is_none_or(|b| bad < b.0) {
                let kbad = seq
                    .iter()
                    .enumerate()
                    .filter(|x| x.1.0 == kernel && mis(*x))
                    .count();
                let pairs = seq
                    .iter()
                    .enumerate()
                    .map(|(i, (src, b))| (src.to_string(), (*b).clone(), bs[start + i]))
                    .collect();
                best = Some((bad, kbad, pairs));
            }
        }
        s.kernels = orig;
    }
    let (_, kbad, pairs) =
        best.ok_or_else(|| anyhow!("no program set containing {kernel} fits the image"))?;
    if kbad > 0 {
        bail!("{kernel}: {kbad} bundles disagree with the image (empty vs non-empty)");
    }
    let mut out = BTreeMap::new();
    for (src, b, raw) in &pairs {
        if src != kernel || !b.ops.iter().any(|o| o.mn == "sbr.rel") {
            continue;
        }
        for base in [467usize, 443] {
            if field(raw, base + 22, 2) != 0
                && field(raw, base + 16, 6) == 0
                && field(raw, base + 11, 5) == 5
            {
                let y = field(raw, base + 5, 6) as u8;
                let i = field(raw, 423, 20) as i32;
                let imm = if i & 0x80000 != 0 { i - (1 << 20) } else { i };
                out.insert(b.addr, BranchEnc { y, imm });
            }
        }
    }
    Ok(out)
}

/// Listing name (`vec_loop.1`) of the final_bundles file `path`.
pub fn listing_name(path: &Path) -> Option<String> {
    let f = path.file_name()?.to_str()?;
    let stem = f.strip_suffix("-final_bundles.txt")?;
    let rest = &stem[stem.find('-')? + 1..];
    Some(rest[..rest.rfind('-')?].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perms() {
        assert_eq!(permutations(3).len(), 6);
        assert_eq!(permutations(0), vec![Vec::<usize>::new()]);
    }
}
