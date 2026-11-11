/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `read_meta` parses the cubin's ELF itself (`.nv.info` records, section sizes, the symbol
//! table). Cross-check: on every kernel of every committed probe and fixture cubin it gives what
//! `cuobjdump -elf` / `-res-usage` print (the parsing `read_meta` did before; the register
//! count of a non-entry function is its EIATTR_REGCOUNT record, which `-res-usage` omits).

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hex(s: &str) -> Option<u32> {
    u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// (const0 size, shared size, params, ntid (req, dims), register count) from cuobjdump.
type Meta = (
    u32,
    u32,
    Vec<(u32, u32, u32)>,
    Option<(bool, [u32; 3])>,
    u16,
);

/// The two cuobjdump listings of a cubin, each run once per cubin, with indexes over them so
/// that every kernel's fields are found without rescanning the whole listing per kernel.
struct Dump {
    elf: String,
    ru: Option<String>,
    /// Section name -> the first nonzero size printed for it (section-header lines).
    sizes: HashMap<String, u32>,
    /// `.nv.info.<kernel>` -> index of the first line equal to it (trimmed).
    info: HashMap<String, usize>,
    /// Kernel -> the first parsable `register count:` after `function: <kernel>(`.
    rec: HashMap<String, u16>,
    /// Kernel -> the REG: value of the line after its first `Function <kernel>:` in -res-usage.
    reg: HashMap<String, u16>,
}

impl Dump {
    fn new(cubin: &Path) -> Option<Dump> {
        let c = cubin.to_str().unwrap();
        let elf = ptxv_sass::run_tool("cuobjdump", &["-elf", c]).ok()?;
        let ru = ptxv_sass::run_tool("cuobjdump", &["-res-usage", c]).ok();
        let (mut sizes, mut info, mut rec) = (HashMap::new(), HashMap::new(), HashMap::new());
        for (n, line) in elf.lines().enumerate() {
            let cols: Vec<&str> = line.split_whitespace().collect();
            if let (Some(name), true) = (cols.last(), cols.len() >= 4) {
                let v = hex(cols[2]).unwrap_or(0);
                if v != 0 {
                    sizes.entry(name.to_string()).or_insert(v);
                }
            }
            let t = line.trim();
            if t.starts_with(".nv.info.") {
                info.entry(t.to_string()).or_insert(n);
            }
            // `function: <name>(...register count: N`: the first occurrence of each name on
            // the line, as `str::split_once` finds it.
            let mut seen: Vec<&str> = Vec::new();
            let mut at = 0;
            while let Some(i) = line[at..].find("function: ") {
                let start = at + i + "function: ".len();
                at = start;
                let Some(open) = line[start..].find('(') else {
                    break;
                };
                let name = &line[start..start + open];
                if seen.contains(&name) {
                    continue;
                }
                seen.push(name);
                let rest = &line[start + open + 1..];
                let r = rest
                    .split_once("register count:")
                    .and_then(|(_, n)| n.trim().parse::<u16>().ok());
                if let Some(r) = r {
                    rec.entry(name.to_string()).or_insert(r);
                }
            }
        }
        let mut reg = HashMap::new();
        if let Some(ru) = &ru {
            let mut it = ru.lines().peekable();
            while let Some(l) = it.next() {
                let Some(k) = l
                    .trim()
                    .strip_prefix("Function ")
                    .and_then(|k| k.strip_suffix(':'))
                else {
                    continue;
                };
                let r = it
                    .peek()
                    .and_then(|n| n.trim().strip_prefix("REG:"))
                    .and_then(|n| n.split_whitespace().next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                reg.entry(k.to_string()).or_insert(r);
            }
        }
        Some(Dump {
            elf,
            ru,
            sizes,
            info,
            rec,
            reg,
        })
    }
}

/// `kernel`'s fields from the cubin's cuobjdump listings.
fn via_cuobjdump(c: &str, d: &Dump, kernel: &str) -> Option<Meta> {
    // A name the indexes cannot key exactly (they split at the first '('): scan the listings.
    if kernel.contains('(') || kernel.contains("function: ") {
        return via_cuobjdump_scan(c, &d.elf, d.ru.as_deref(), kernel);
    }
    let c0 = d
        .sizes
        .get(&format!(".nv.constant0.{kernel}"))
        .copied()
        .unwrap_or(0);
    let sm = d
        .sizes
        .get(&format!(".nv.shared.{kernel}"))
        .copied()
        .unwrap_or(0);
    let start = d.info.get(&format!(".nv.info.{kernel}")).copied();
    let (params, ntid) = info_block(&d.elf, start)?;
    let rec = d.rec.get(kernel).copied();
    let regs = d.reg.get(kernel).copied().unwrap_or(0);
    Some(finish(c, kernel, c0, sm, params, ntid, rec, regs))
}

/// The params and ntid of the `.nv.info.<kernel>` block starting at line `start`; None when a
/// parameter record does not parse (the kernel is skipped).
type Info = (Vec<(u32, u32, u32)>, Option<(bool, [u32; 3])>);
fn info_block(elf: &str, start: Option<usize>) -> Option<Info> {
    let (mut params, mut ntid) = (vec![], None);
    if let Some(start) = start {
        let mut attr = String::new();
        for line in elf.lines().skip(start + 1) {
            if line.starts_with('.') {
                break;
            }
            if let Some(a) = line.trim().strip_prefix("Attribute:") {
                attr = a.trim().to_string();
            }
            if let Some(v) = line.trim().strip_prefix("Value:") {
                if attr == "EIATTR_REQNTID" || attr == "EIATTR_MAX_THREADS" {
                    let d: Vec<u32> = v.split_whitespace().filter_map(hex).collect();
                    if let [x, y, z] = d[..] {
                        ntid = Some((attr == "EIATTR_REQNTID", [x, y, z]));
                    }
                }
                if v.contains("Ordinal") && v.contains("Offset") {
                    let field = |k: &str| -> Option<u32> {
                        let i = v.find(k)? + k.len();
                        let rest = v[i..].trim_start().trim_start_matches(':').trim_start();
                        hex(rest.split_whitespace().next()?)
                    };
                    params.push((field("Ordinal")?, field("Offset")?, field("Size")?));
                }
            }
        }
    }
    params.sort();
    Some((params, ntid))
}

/// The register count: the EIATTR_REGCOUNT record `-elf` prints (`rec`). `-res-usage`
/// (`regs`) agrees on kernels but prints REG:0 for non-entry functions that have a record.
#[allow(clippy::too_many_arguments)]
fn finish(
    c: &str,
    kernel: &str,
    c0: u32,
    sm: u32,
    params: Vec<(u32, u32, u32)>,
    ntid: Option<(bool, [u32; 3])>,
    rec: Option<u16>,
    mut regs: u16,
) -> Meta {
    if let Some(r) = rec {
        assert!(
            regs == 0 || regs == r,
            "{c} {kernel}: -res-usage REG:{regs}, record {r}"
        );
        regs = r;
    }
    (c0, if sm == 0 { 0x400 } else { sm }, params, ntid, regs)
}

/// `via_cuobjdump` by scanning the listings (no indexes).
fn via_cuobjdump_scan(c: &str, elf: &str, ru: Option<&str>, kernel: &str) -> Option<Meta> {
    let (mut c0, mut sm) = (0, 0);
    for line in elf.lines() {
        let cols: Vec<&str> = line.split_whitespace().collect();
        if let (Some(name), true) = (cols.last(), cols.len() >= 4) {
            if *name == format!(".nv.constant0.{kernel}") && c0 == 0 {
                c0 = hex(cols[2]).unwrap_or(0);
            } else if *name == format!(".nv.shared.{kernel}") && sm == 0 {
                sm = hex(cols[2]).unwrap_or(0);
            }
        }
    }
    let sect = format!(".nv.info.{kernel}");
    let start = elf.lines().position(|l| l.trim() == sect);
    let (params, ntid) = info_block(elf, start)?;
    let tag = format!("function: {kernel}(");
    let rec: Option<u16> = elf.lines().find_map(|l| {
        let (_, rest) = l.split_once(&tag)?;
        rest.split_once("register count:")?.1.trim().parse().ok()
    });
    let mut regs = 0;
    if let Some(ru) = ru {
        let mut it = ru.lines();
        while let Some(l) = it.next() {
            if l.trim() == format!("Function {kernel}:") {
                regs = it
                    .next()
                    .and_then(|n| n.trim().strip_prefix("REG:"))
                    .and_then(|n| n.split_whitespace().next())
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(0);
                break;
            }
        }
    }
    Some(finish(c, kernel, c0, sm, params, ntid, rec, regs))
}

fn cubins(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.is_dir() {
            cubins(&p, out);
        } else if p.extension().is_some_and(|x| x == "cubin") {
            out.push(p);
        }
    }
}

#[test]
#[ignore = "heavy: cuobjdump on 377 cubins, 9.4k kernels compared; 30 s"]
fn read_meta_matches_cuobjdump() {
    let mut all = vec![];
    cubins(&root().join("probes"), &mut all);
    cubins(&root().join("fixtures"), &mut all);
    all.sort();
    let (mut kernels, mut skipped) = (0, 0);
    for c in &all {
        let img = ptxv_sass::elf::Elf::parse(&std::fs::read(c).unwrap()).unwrap();
        // cuobjdump runs once per cubin (not per kernel); a cubin it cannot list skips every
        // kernel in it.
        let dump = Dump::new(c);
        for s in &img.sections {
            let Some(k) = s.name.strip_prefix(".text.") else {
                continue;
            };
            let Some(want) = dump
                .as_ref()
                .and_then(|d| via_cuobjdump(c.to_str().unwrap(), d, k))
            else {
                skipped += 1;
                continue;
            };
            let m = ptxv_sass::read_meta(c, k).unwrap();
            let got = (
                m.const0_size,
                m.smem_size,
                m.params.clone(),
                m.ntid.map(|n| (n.req, n.dims)),
                m.regcount,
            );
            assert_eq!(got, want, "{} {k}", c.display());
            kernels += 1;
        }
    }
    assert!(
        kernels >= 300 && skipped == 0,
        "{kernels} kernels, {skipped} skipped"
    );
}
