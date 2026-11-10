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

fn via_cuobjdump(cubin: &Path, kernel: &str) -> Option<Meta> {
    let c = cubin.to_str().unwrap();
    let elf = ptxv_sass::run_tool("cuobjdump", &["-elf", c]).ok()?;
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
    let (mut params, mut ntid) = (vec![], None);
    let sect = format!(".nv.info.{kernel}");
    if let Some(start) = elf.lines().position(|l| l.trim() == sect) {
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
    // The register count: the EIATTR_REGCOUNT record `-elf` prints. `-res-usage` agrees on
    // kernels but prints REG:0 for non-entry functions that have a record (checked below).
    let tag = format!("function: {kernel}(");
    let rec: Option<u16> = elf.lines().find_map(|l| {
        let (_, rest) = l.split_once(&tag)?;
        rest.split_once("register count:")?.1.trim().parse().ok()
    });
    let mut regs = 0;
    if let Ok(ru) = ptxv_sass::run_tool("cuobjdump", &["-res-usage", c]) {
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
    if let Some(r) = rec {
        assert!(
            regs == 0 || regs == r,
            "{c} {kernel}: -res-usage REG:{regs}, record {r}"
        );
        regs = r;
    }
    Some((c0, if sm == 0 { 0x400 } else { sm }, params, ntid, regs))
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
fn read_meta_matches_cuobjdump() {
    let mut all = vec![];
    cubins(&root().join("probes"), &mut all);
    cubins(&root().join("fixtures"), &mut all);
    all.sort();
    let (mut kernels, mut skipped) = (0, 0);
    for c in &all {
        let img = ptxv_sass::elf::Elf::parse(&std::fs::read(c).unwrap()).unwrap();
        for s in &img.sections {
            let Some(k) = s.name.strip_prefix(".text.") else {
                continue;
            };
            let Some(want) = via_cuobjdump(c, k) else {
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
