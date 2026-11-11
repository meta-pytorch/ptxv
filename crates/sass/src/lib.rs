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
mod modset;
mod undis;
/// The disassembly of an encoding nvdisasm 13.0 prints nothing for, if `undis.rs` knows it.
pub use undis::decode as decode_unprinted;
mod l2;
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
    /// EIATTR_REQNTID / EIATTR_MAX_THREADS of the kernel.
    pub ntid: Option<Ntid>,
    /// The kernel's register count (EIATTR_REGCOUNT in `.nv.info`); 0 if unknown.
    pub regcount: u16,
    /// Data globals whose addresses those banks hold, and the words holding them.
    pub globals: Vec<ptxv_ir::GlobalSym>,
    pub global_relocs: Vec<ptxv_ir::GlobalReloc>,
    /// SM version of the module (ELF `e_flags` bits 8..16: 100 for sm_100 / sm_100a / sm_100f,
    /// 103 for sm_103a); 0 when unknown. Selects arch-specific opcode latencies (scoreboard.rs).
    /// This, not `target`, selects the hazard rules, and marks an sm_100 module's verdict as
    /// not a GB300 verdict: GB300 may reschedule such a module at load (probes/f64_arch).
    /// ASSUME(L-OPCLASS)
    pub sm: u32,
    /// The compile target as given to ptxas (`-arch sm_100f`), from the toolkit note
    /// (`.note.nv.tkinfo`): the only place a cubin records the arch-specific (`a`) or family
    /// (`f`) suffix, since `e_flags` is the same for sm_100, sm_100a and sm_100f. Empty when
    /// the note is absent.
    pub target: String,
    /// The kernel's code (`.text.<kernel>`), for the slots nvdisasm prints nothing for
    /// (`undis.rs`); empty when unknown.
    pub text: Vec<u8>,
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

/// Compile PTX to a cubin with ptxas (the toolkit `ptxas_with` picks for the module).
pub fn ptxas(ptx: &Path, cubin: &Path, arch: &str) -> Result<()> {
    ptxas_with(ptx, cubin, arch, None, &[]).map(|_| ())
}

/// `major.minor` of a version string ("13.3", "13.3.73", "V13.3.73").
fn major_minor(s: &str) -> Option<(u32, u32)> {
    let mut it = s.trim().trim_start_matches(['V', 'v']).split('.');
    Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
}

/// The CUDA toolkit version of a ptxas binary (`--version`: "release 13.3, V13.3.73"), memoized.
pub fn ptxas_version(path: &str) -> Option<(u32, u32)> {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    type Memo = Mutex<HashMap<String, Option<(u32, u32)>>>;
    static MEMO: OnceLock<Memo> = OnceLock::new();
    let memo = MEMO.get_or_init(Default::default);
    if let Some(v) = memo.lock().unwrap().get(path) {
        return *v;
    }
    let v = Command::new(path)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).to_string();
            let i = s.find("release ")?;
            major_minor(s[i + 8..].split(',').next()?)
        });
    memo.lock().unwrap().insert(path.to_string(), v);
    v
}

/// The PTX ISA of a module (its `.version`).
pub fn ptx_isa(ptx: &str) -> Option<(u32, u32)> {
    let l = ptx
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with(".version "))?;
    major_minor(
        l.trim_start_matches(".version ")
            .split_whitespace()
            .next()?,
    )
}

/// The first CUDA toolkit whose ptxas accepts PTX ISA `isa`: ISA 9.x came with CUDA 13.x and
/// 8.x with 12.x (8.8 and later: 12.8 and later).
pub fn toolkit_for_isa(isa: (u32, u32)) -> (u32, u32) {
    match isa {
        (9, m) => (13, m),
        (8, m) => (12, m.min(9)),
        (a, b) if a > 9 => (a + 4, b),
        _ => (0, 0),
    }
}

/// Installed toolkits: PTXV_TOOLKITS is a directory of `<major.minor>/bin/ptxas` trees (the
/// toolkit registry; a version directory may be a symlink to a CUDA install). Sorted by version.
pub fn toolkits() -> Vec<((u32, u32), String)> {
    let Ok(root) = std::env::var("PTXV_TOOLKITS") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let ptxas = e.path().join("bin/ptxas");
        if let (Some(v), true) = (major_minor(&name), ptxas.exists()) {
            out.push((v, ptxas.to_string_lossy().to_string()));
        }
    }
    out.sort();
    out
}

/// The registry toolkits whose ptxas runs here and reports its directory's version: a ptxas
/// built for another architecture (a registry shared by aarch64 and x86_64 hosts) or a
/// mislabeled directory is skipped, never run or credited with a version it is not.
pub fn usable_toolkits() -> Vec<((u32, u32), String)> {
    toolkits()
        .into_iter()
        .filter(|(v, p)| ptxas_version(p) == Some(*v))
        .collect()
}

/// The ptxas for a module: the registry toolkit equal to `want` (the toolkit that built the
/// executed cubin, `major.minor`) if installed; else the default ptxas (`$CUDA_HOME/bin`) when
/// it accepts the module's PTX ISA, else the oldest registry toolkit that does. Returns the path
/// and the chosen toolkit version.
pub fn select_ptxas(isa: Option<(u32, u32)>, want: Option<&str>) -> (String, Option<(u32, u32)>) {
    let default = tool("ptxas");
    let dv = ptxas_version(&default);
    let reg = usable_toolkits();
    if let Some(w) = want.and_then(major_minor) {
        if dv == Some(w) {
            return (default, dv);
        }
        if let Some((v, p)) = reg.iter().find(|(v, _)| *v == w) {
            return (p.clone(), Some(*v));
        }
    }
    let need = isa.map(toolkit_for_isa).unwrap_or((0, 0));
    if dv.is_none_or(|d| d >= need) {
        return (default, dv);
    }
    match reg.iter().find(|(v, _)| *v >= need) {
        Some((v, p)) => (p.clone(), Some(*v)),
        None => (default, dv),
    }
}

/// Why no installed ptxas can compile a module of PTX ISA `isa`, when the one `select_ptxas`
/// picked (version `chosen`) is older than the first toolkit that accepts it; None when it is
/// new enough or its version is unknown (ptxas then reports for itself). `reg`: the usable
/// registry toolkits; `skipped`: registry versions whose ptxas does not run here.
pub fn ptxas_too_old(
    isa: (u32, u32),
    chosen: Option<(u32, u32)>,
    reg: &[((u32, u32), String)],
    skipped: &[(u32, u32)],
) -> Option<String> {
    let need = toolkit_for_isa(isa);
    let (c, d) = chosen?;
    if (c, d) >= need {
        return None;
    }
    let have: Vec<String> = reg.iter().map(|((a, b), _)| format!("{a}.{b}")).collect();
    let bad: Vec<String> = skipped.iter().map(|(a, b)| format!("{a}.{b}")).collect();
    let (n0, n1) = need;
    let bad = if bad.is_empty() {
        String::new()
    } else {
        format!(
            "; skipped, ptxas does not run here or reports another version: [{}]",
            bad.join(", ")
        )
    };
    Some(format!(
        "the module is PTX ISA {}.{}, which needs ptxas {n0}.{n1} or later; the newest usable \
         is {c}.{d} (default $CUDA_HOME/bin/ptxas, and the PTXV_TOOLKITS registry: [{}]{bad}): \
         install toolkit {n0}.{n1} for this host as <PTXV_TOOLKITS>/{n0}.{n1}/bin/ptxas",
        isa.0,
        isa.1,
        have.join(", "),
    ))
}

/// The registry toolkit (PTXV_TOOLKITS) `select_ptxas` picks for a module, None when it picks
/// the default ptxas (whose version is ambient, not a property of the case).
pub fn registry_toolkit(ptx_text: &str, want: Option<&str>) -> Option<(u32, u32)> {
    let (path, v) = select_ptxas(ptx_isa(ptx_text), want);
    (path != tool("ptxas")).then_some(v).flatten()
}

/// Compile PTX to a cubin with the ptxas `select_ptxas` picks for it (`want`: the toolkit that
/// built the executed cubin; `extra`: options the toolchain passed that the toolkit note does
/// not record, a case's `ptxas_args`). Returns the toolkit version used ("13.3"), when known.
pub fn ptxas_with(
    ptx: &Path,
    cubin: &Path,
    arch: &str,
    want: Option<&str>,
    extra: &[String],
) -> Result<Option<String>> {
    let text = std::fs::read_to_string(ptx).unwrap_or_default();
    let isa = ptx_isa(&text);
    let (path, v) = select_ptxas(isa, want);
    let usable = usable_toolkits();
    let skipped: Vec<_> = toolkits()
        .into_iter()
        .map(|(v, _)| v)
        .filter(|v| !usable.iter().any(|(u, _)| u == v))
        .collect();
    if let Some(why) = isa.and_then(|i| ptxas_too_old(i, v, &usable, &skipped)) {
        bail!("{why}");
    }
    let out = Command::new(&path)
        .args(extra)
        .args([
            &format!("-arch={arch}"),
            ptx.to_str().unwrap(),
            "-o",
            cubin.to_str().unwrap(),
        ])
        .output()
        .with_context(|| format!("running {path}"))?;
    if !out.status.success() {
        bail!(
            "ptxas ({path}) failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(v.map(|(a, b)| format!("{a}.{b}")))
}

#[cfg(test)]
mod toolkit_tests {
    use super::*;

    #[test]
    fn ptx_isa_maps_to_the_first_toolkit_that_accepts_it() {
        assert_eq!(ptx_isa(".version 9.3\n.target sm_103a\n"), Some((9, 3)));
        assert_eq!(ptx_isa("// x\n  .version 8.8\n"), Some((8, 8)));
        assert_eq!(toolkit_for_isa((9, 0)), (13, 0));
        assert_eq!(toolkit_for_isa((9, 3)), (13, 3));
        assert_eq!(toolkit_for_isa((8, 8)), (12, 8));
        assert_eq!(major_minor("V13.3.73"), Some((13, 3)));
        assert_eq!(major_minor("13.1"), Some((13, 1)));
    }

    #[test]
    fn a_module_newer_than_every_installed_ptxas_is_refused_by_name() {
        let reg = vec![((13, 0), "/r/13.0/bin/ptxas".to_string())];
        let why = ptxas_too_old((9, 3), Some((13, 0)), &reg, &[(13, 3)]).unwrap();
        assert!(why.contains("PTX ISA 9.3, which needs ptxas 13.3"), "{why}");
        assert!(
            why.contains("[13.0]") && why.contains("PTXV_TOOLKITS"),
            "{why}"
        );
        assert!(
            why.contains("does not run here or reports another version: [13.3]"),
            "{why}"
        );
        assert_eq!(ptxas_too_old((9, 3), Some((13, 3)), &reg, &[]), None);
        assert_eq!(ptxas_too_old((9, 0), Some((13, 3)), &reg, &[]), None);
        assert_eq!(ptxas_too_old((9, 3), None, &reg, &[]), None);
    }

    #[test]
    fn registry_lists_version_dirs_with_a_ptxas() {
        let d = std::env::temp_dir().join(format!("ptxv-toolkits-{}", std::process::id()));
        for v in ["13.3", "13.1", "notaversion"] {
            std::fs::create_dir_all(d.join(v).join("bin")).unwrap();
            std::fs::write(d.join(v).join("bin/ptxas"), "").unwrap();
        }
        std::fs::create_dir_all(d.join("13.2")).unwrap(); // no bin/ptxas: not a toolkit
        // SAFETY: tests of this module do not read PTXV_TOOLKITS concurrently.
        unsafe { std::env::set_var("PTXV_TOOLKITS", &d) };
        let vs: Vec<_> = toolkits().into_iter().map(|(v, _)| v).collect();
        unsafe { std::env::remove_var("PTXV_TOOLKITS") };
        assert_eq!(vs, vec![(13, 1), (13, 3)]);
    }
}

/// The bytes of `.text.<kernel>` of a cubin (its machine code), if present.
pub fn kernel_text(cubin: &[u8], kernel: &str) -> Option<Vec<u8>> {
    let e = elf::Elf::parse(cubin).ok()?;
    let i = e.find(&format!(".text.{kernel}"))?;
    Some(e.sections[i].data.clone())
}

/// The CUDA toolkit that built a cubin ("13.3"), from its toolkit note (`.note.nv.tkinfo`:
/// "Cuda compilation tools, release 13.3, V13.3.73").
pub fn cubin_toolkit(cubin: &[u8]) -> Option<String> {
    let e = elf::Elf::parse(cubin).ok()?;
    let note = String::from_utf8_lossy(&e.sections[e.find(".note.nv.tkinfo")?].data).to_string();
    let i = note.find("release ")?;
    let (a, b) = major_minor(note[i + 8..].split(',').next()?)?;
    Some(format!("{a}.{b}"))
}

fn hex(s: &str) -> Option<u32> {
    u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

/// SM version in an ELF64 cubin's `e_flags` (bits 8..16; 0 for anything else).
pub fn elf_sm(b: &[u8]) -> u32 {
    if b.len() < 0x34 || b[..4] != *b"\x7fELF" || b[4] != 2 {
        return 0;
    }
    (u32::from_le_bytes([b[0x30], b[0x31], b[0x32], b[0x33]]) >> 8) & 0xff
}

/// The compile target in a cubin's toolkit note (`.note.nv.tkinfo`): the `sm_*` operand of
/// the recorded `-arch` / `--gpu-name` option (`"sm_100f"`, `"sm_103a"`).
pub fn tk_target(img: &elf::Elf) -> Option<String> {
    let note = &img.sections[img.find(".note.nv.tkinfo")?].data;
    let text = String::from_utf8_lossy(note);
    let mut words = text.split(|c: char| c == '\0' || c == '=' || c.is_whitespace());
    while let Some(w) = words.next() {
        if matches!(w, "-arch" | "--gpu-name") {
            if let Some(t) = words.find(|t| !t.is_empty()) {
                if t.starts_with("sm_") {
                    return Some(t.to_string());
                }
            }
        }
    }
    None
}

/// The tool that wrote a cubin's toolkit note (`.note.nv.tkinfo`): `"ptxas"`, or Triton's
/// bundled `"ptxas-blackwell"`, which runs with options the note does not record.
pub fn tk_producer(cubin: &[u8]) -> Option<String> {
    let img = elf::Elf::parse(cubin).ok()?;
    let note = &img.sections[img.find(".note.nv.tkinfo")?].data;
    note.split(|&c| !(0x20..0x7f).contains(&c))
        .filter(|w| w.len() >= 3)
        .map(|w| String::from_utf8_lossy(w).into_owned())
        .find(|w| w.starts_with("ptxas") || w.starts_with("nvcc") || w.starts_with("nvrtc"))
}

/// The records of a `.nv.info[.<kernel>]` section: (attribute, value bytes). Each record is a
/// format byte, an attribute byte, then for EIFMT_SVAL (4) a 16-bit size and that many bytes,
/// otherwise a 16-bit value.
pub fn nv_info(data: &[u8]) -> Vec<(u8, &[u8])> {
    let mut out = Vec::new();
    let mut o = 0;
    while o + 4 <= data.len() {
        let (fmt, attr) = (data[o], data[o + 1]);
        let (val, next) = if fmt == 4 {
            let n = u16::from_le_bytes([data[o + 2], data[o + 3]]) as usize;
            (data.get(o + 4..o + 4 + n), o + 4 + n)
        } else {
            (data.get(o + 2..o + 4), o + 4)
        };
        match val {
            Some(v) => out.push((attr, v)),
            None => break,
        }
        o = next;
    }
    out
}

const EIATTR_MAX_THREADS: u8 = 0x05;
const EIATTR_REQNTID: u8 = 0x10;
const EIATTR_KPARAM_INFO: u8 = 0x17;
const EIATTR_KPARAM_INFO_V2: u8 = 0x45;
const EIATTR_REGCOUNT: u8 = 0x2f;

/// The kernel's metadata, read from the cubin's ELF directly (sections, `.nv.info` records,
/// symbol table), so post-processed cubins that `cuobjdump` rejects ("Invalid ELF") load too.
/// Checked against `cuobjdump -elf` / `-res-usage` on every probe and fixture cubin
/// (`read_meta_matches_cuobjdump`).
pub fn read_meta(cubin: &Path, kernel: &str) -> Result<CubinMeta> {
    let bytes = std::fs::read(cubin)?;
    let img = elf::Elf::parse(&bytes)?;
    let mut m = CubinMeta {
        sm: elf_sm(&bytes),
        target: tk_target(&img).unwrap_or_default(),
        ..Default::default()
    };
    let size = |name: String| img.find(&name).map(|i| img.sections[i].size as u32);
    m.const0_size = size(format!(".nv.constant0.{kernel}")).unwrap_or(0);
    m.smem_size = size(format!(".nv.shared.{kernel}")).unwrap_or(0);
    let w = |v: &[u8], i: usize| -> u32 {
        v.get(4 * i..4 * i + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .unwrap_or(0)
    };
    if let Some(i) = img.find(&format!(".nv.info.{kernel}")) {
        for (attr, v) in nv_info(&img.sections[i].data) {
            match attr {
                // index u32, ordinal u16, offset u16, then the size in bits 18..32 (V1), or
                // in bits 0..16 (V2: parameters over 16 KiB, trtllm-gen's 17 KiB struct; the
                // other bits were 0 in every cubin seen, anything else is refused).
                EIATTR_KPARAM_INFO | EIATTR_KPARAM_INFO_V2 if v.len() >= 12 => {
                    let size = if attr == EIATTR_KPARAM_INFO {
                        (w(v, 2) >> 18) & 0x3fff
                    } else if w(v, 2) >> 16 == 0 {
                        w(v, 2)
                    } else {
                        bail!(
                            "{kernel}: EIATTR_KPARAM_INFO_V2 word {:#x} not modeled",
                            w(v, 2)
                        );
                    };
                    m.params.push((
                        u16::from_le_bytes([v[4], v[5]]) as u32,
                        u16::from_le_bytes([v[6], v[7]]) as u32,
                        size,
                    ))
                }
                EIATTR_REQNTID | EIATTR_MAX_THREADS if v.len() >= 12 => {
                    m.ntid = Some(Ntid {
                        req: attr == EIATTR_REQNTID,
                        dims: [w(v, 0), w(v, 1), w(v, 2)],
                    })
                }
                _ => {}
            }
        }
    }
    m.params.sort();
    if let Some(i) = img.find(&format!(".text.{kernel}")) {
        m.text = img.sections[i].data.clone();
    }
    // EIATTR_REGCOUNT in `.nv.info`: (symbol index, register count) per function.
    if let Some(i) = img.find(".nv.info") {
        for (attr, v) in nv_info(&img.sections[i].data) {
            if attr == EIATTR_REGCOUNT
                && v.len() >= 8
                && img.symbol(w(v, 0) as usize).is_ok_and(|s| s.name == kernel)
            {
                m.regcount = w(v, 1) as u16;
            }
        }
    }
    for bank in 1..18u8 {
        if let Some(cb) = img.bank(bank, kernel)? {
            m.banks.insert(bank, cb.data);
            m.syms.extend(cb.syms);
            for (off, g, addend) in cb.data_relocs {
                let i = match m.globals.iter().position(|x| x.name == g.name) {
                    Some(i) => i,
                    None => {
                        m.globals.push(ptxv_ir::GlobalSym {
                            name: g.name,
                            size: g.size,
                            init: g.init,
                        });
                        m.globals.len() - 1
                    }
                };
                m.global_relocs.push(ptxv_ir::GlobalReloc {
                    bank,
                    off,
                    global: i as u32,
                    addend,
                });
            }
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
        lines.push(sass_line(addr, body, lo, None));
    }
    if lines.is_empty() {
        bail!("kernel {kernel} not found in disassembly");
    }
    Ok((lines, labels))
}

/// One instruction from its printed text (`[@g] OPCODE operands [;]`).
fn sass_line(addr: u32, body: &str, lo: Option<u64>, hi: Option<u64>) -> SassLine {
    let body = body.trim().trim_end_matches(';').trim();
    let (guard, body) = if body.starts_with('@') {
        let (g, r) = body.split_once(char::is_whitespace).unwrap();
        (Some(g[1..].to_string()), r.trim())
    } else {
        (None, body)
    };
    let (opcode, ops) = body.split_once(char::is_whitespace).unwrap_or((body, ""));
    SassLine {
        addr,
        guard,
        opcode: opcode.to_string(),
        operands: split_ops(ops),
        text: body.to_string(),
        hi,
        lo,
    }
}

/// Fill the 16-byte slots nvdisasm printed nothing for with the encodings `undis::decode`
/// knows, read from the kernel's code; other holes stay (the kernel is refused later).
fn fill_unprinted(lines: &mut Vec<SassLine>, text: &[u8]) {
    let have: std::collections::HashSet<u32> = lines.iter().map(|l| l.addr).collect();
    // Scan the whole text, not only up to the last printed line: a trailing slot can be
    // unprinted too. Padding (zero bytes) does not decode.
    let mut added = false;
    for o in (0..text.len() / 16 * 16).step_by(16) {
        if have.contains(&(o as u32)) {
            continue;
        }
        let w = |i: usize| u64::from_le_bytes(text[o + i..o + i + 8].try_into().unwrap());
        let (lo, hi) = (w(0), w(8));
        if let Some(t) = undis::decode(lo, hi) {
            lines.push(sass_line(o as u32, &t, Some(lo), Some(hi)));
            added = true;
        }
    }
    if added {
        lines.sort_by_key(|l| l.addr);
    }
}

/// A relative `BRA`'s target from its encoding (opcode 0x947): the signed word offset is bits
/// [16, 24) and [34, 64) of the low word, counted from the next instruction. Fitted on the
/// 243 `BRA` of ssa2's MXINT8 G32 decode kernel that nvdisasm 13.3 labels (all agree); `None`
/// when bits [24, 34) are not zero (a form not seen).
fn bra_target(addr: u32, lo: u64) -> Option<u32> {
    if lo & 0xfff != 0x947 || (lo >> 24) & 0x3ff != 0 {
        return None;
    }
    let v = ((lo >> 16) & 0xff) | ((lo >> 34) << 8);
    let off = ((v << 26) as i64 >> 26) * 4;
    u32::try_from(addr as i64 + 16 + off).ok()
}

/// Labels of branch targets nvdisasm did not print: nvdisasm 13.0 prints neither an
/// instruction it cannot decode (ptxas 13.3's `F2FP.SATFINITE.BF16.S2_6`, filled from its
/// encoding by `fill_unprinted`) nor a label on it, so a `BRA` there names an unknown label.
/// Each such label gets the target decoded from the branch's encoding, provided the decoding
/// agrees with every label nvdisasm did print and the target is an instruction; otherwise
/// nothing is added (the kernel is refused as before).
fn recover_branch_labels(lines: &[SassLine], labels: &mut HashMap<String, u32>) {
    let have: std::collections::HashSet<u32> = lines.iter().map(|l| l.addr).collect();
    let lbl_of = |l: &SassLine| -> Option<String> {
        if l.opcode != "BRA" && !l.opcode.starts_with("BRA.") {
            return None;
        }
        let t = l.operands.last()?;
        let t = t.strip_prefix("`(")?.strip_suffix(')')?;
        Some(t.to_string())
    };
    let mut add = Vec::new();
    for l in lines {
        let (Some(lbl), Some(lo)) = (lbl_of(l), l.lo) else {
            continue;
        };
        // Forms the decoder does not know (`BRA.DIV`, `BRA.U` with a uniform predicate, ...)
        // give no evidence either way: a known label is left alone, an unknown one stays
        // unknown (and the kernel is refused there as before).
        let t = bra_target(l.addr, lo);
        match (labels.get(&lbl), t) {
            (Some(&a), Some(t)) if t != a => return,
            (None, Some(t)) if have.contains(&t) => add.push((lbl, t)),
            _ => {}
        }
    }
    for (lbl, t) in add {
        if labels.get(&lbl).is_some_and(|&a| a != t) {
            return;
        }
        labels.insert(lbl, t);
    }
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
            // `R2.reuse.SIGN` / `UR6.SIGN` (PLOP3 inputs; ptxas's add.sat.s32 / sub.sat.s32):
            // the register's sign bit as a predicate (`Machine::prd`; probes/int_sat).
            _ if s.ends_with(".SIGN") => {
                let r = s.trim_end_matches(".SIGN").trim_end_matches(".reuse");
                match Self::reg_name(r) {
                    Some((file @ (File::R | File::UR), idx)) => Some(Reg { file, idx }),
                    _ => return Err(self.err(format!("bad predicate {s}"))),
                }
            }
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
                // `R.SIGN` / `UR.SIGN` are predicate inputs only (MISS 23).
                match p.reg {
                    Some(
                        r @ Reg {
                            file: File::P | File::UP,
                            ..
                        },
                    ) => Ok(r),
                    _ => bail!(self.err(format!("bad predicate destination {s}"))),
                }
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
            // 32-bit clock and global-timer halves (`S2UR` / `S2R` / `CS2R.32`; %clock,
            // %clock_hi, %globaltimer_lo / _hi): read as 0 like the 64-bit forms, a value the
            // model never uses to order anything, so timeouts never fire. ASSUME(E-CLOCK)
            match sr {
                "CLOCKLO" | "GLOBALTIMERLO" => return Ok(Val::SReg(SReg::ClockLo32)),
                "CLOCKHI" | "GLOBALTIMERHI" => return Ok(Val::SReg(SReg::ClockHi32)),
                _ => {}
            }
            let v = match sr {
                "TID.X" => SReg::Tid(0),
                "TID.Y" => SReg::Tid(1),
                "TID.Z" => SReg::Tid(2),
                "CTAID.X" => SReg::CtaId(0),
                "CTAID.Y" => SReg::CtaId(1),
                "CTAID.Z" => SReg::CtaId(2),
                "LANEID" => SReg::LaneId,
                "CgaCtaId" => SReg::ClusterCtaRank,
                "CgaSize" => SReg::CgaShape,
                "LTMASK" => SReg::LaneMaskLt,
                "LEMASK" => SReg::LaneMaskLe,
                "GEMASK" => SReg::LaneMaskGe,
                "GTMASK" => SReg::LaneMaskGt,
                "EQMASK" => SReg::LaneMaskEq,
                "SWINHI" => SReg::SharedWindowHi,
                // `%warpid` (ptxas extracts it from SR_VIRTID): the warp's SM slot, whose value
                // the simulator leaves undefined, so the exact field layout does not matter.
                // ASSUME(E-WARPID): see ASSUMPTIONS.md
                "VIRTID" => SReg::WarpId,
                // %smid (probes/smid). ASSUME(E-SMID)
                "VIRTUALSMID" => SReg::SmId,
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
        // `[R2.U32+UR6]` / `[RZ.U32+UR6]` (CuTe DSL dist kernels' generic stores): a 32-bit
        // register, zero-extended, plus the 64-bit uniform pair UR6:UR7. ASSUME(L-ROSETTA)
        let mut zext = false;
        for part in inner.split('+') {
            let mut part = part.trim();
            if let Some(r) = part.strip_suffix(".U32") {
                if !r.starts_with('R') || zext {
                    return Err(self.err(format!("bad address {s}")));
                }
                zext = true;
                part = r;
            }
            if let Some(h) = part.strip_prefix("0x") {
                a.off += i64::from_str_radix(h, 16).map_err(|_| self.err("bad offset"))?;
            } else if let Some(h) = part.strip_prefix("-0x") {
                a.off -= i64::from_str_radix(h, 16).map_err(|_| self.err("bad offset"))?;
            } else {
                // `[RZ.U32+UR6]`: the zero register as a 32-bit base (an address in the
                // uniform register alone; ptxas's STG.E.64.MMIO.SYS doorbell form).
                let part = part
                    .strip_suffix(".U32")
                    .filter(|p| *p == "RZ")
                    .unwrap_or(part);
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
        if zext {
            let mut wide = 0;
            for b in [&mut a.base, &mut a.base2].into_iter().flatten() {
                if let Val::Reg(r, 1) = *b {
                    if r.file == File::UR {
                        *b = Val::Reg(r, 2);
                        wide += 1;
                    }
                }
            }
            if wide != 1 {
                return Err(self.err(format!("address {s}: .U32 needs one uniform base")));
            }
        }
        Ok(a)
    }

    /// `tmem[UR4+0x10]` / `gdesc[UR4]` / `idesc[UR17]` -> (register value, offset).
    fn bracket(&self, s: &str, prefix: &str, n: u8) -> Result<(Val, u32)> {
        self.bracket_neg(s, prefix, n, false)
    }

    /// `bracket`; `neg`: a negative offset (`tmem[UR5+-0x10]`) is allowed, as its 32-bit
    /// two's complement (for executors that add the offset with wrapping arithmetic).
    fn bracket_neg(&self, s: &str, prefix: &str, n: u8, neg: bool) -> Result<(Val, u32)> {
        let inner = s
            .trim()
            .strip_prefix(prefix)
            .and_then(|x| x.strip_prefix('['))
            .and_then(|x| x.strip_suffix(']'))
            .ok_or_else(|| self.err(format!("expected {prefix}[..], got {s}")))?;
        // `tmem[UR5+-0x10]`: a negative offset, added to the 32-bit TMEM address like a
        // positive one (ptxas folds `taddr - 16` into it; GB300: probes/tmem_ldred).
        let (r, off) = match inner.split_once('+') {
            Some((r, o)) => match o.strip_prefix('-') {
                Some(n) if neg => (
                    r,
                    hex(n).ok_or_else(|| self.err("bad offset"))?.wrapping_neg(),
                ),
                _ => (r, hex(o).ok_or_else(|| self.err("bad offset"))?),
            },
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

/// A half immediate pair printed `hi, lo` (f16, or bf16 when `bf16`): its 32 bits from the
/// encoding's word 32..63, checked against the printed values. ASSUME(L-ROSETTA)
fn half_imm_pair(lo: Option<u64>, toks: [&str; 2], bf16: bool) -> std::result::Result<u32, String> {
    let bits = (lo.ok_or("half immediate without encoding")? >> 32 & 0xffff_ffff) as u32;
    for (tok, half) in [(toks[0], bits >> 16), (toks[1], bits & 0xffff)] {
        let t = tok.trim().trim_start_matches('+');
        let ok = if bf16 {
            let w = f32::from_bits(half << 16);
            match t {
                "QNAN" | "-QNAN" => w.is_nan(),
                "INF" => w == f32::INFINITY,
                "-INF" => w == f32::NEG_INFINITY,
                x => x
                    .parse::<f64>()
                    .is_ok_and(|p| p == w as f64 || (p as f32) == w),
            }
        } else {
            match t {
                "QNAN" | "-QNAN" => half & 0x7c00 == 0x7c00 && half & 0x3ff != 0,
                "INF" => half == 0x7c00,
                "-INF" => half == 0xfc00,
                x => parse_half(x) == Some(half as u16),
            }
        };
        if !ok {
            return Err(format!(
                "half immediate {tok} is not the encoding's {half:#06x}"
            ));
        }
    }
    Ok(bits)
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

/// The operand-less EXITs of a function: (line index, guard).
fn exit_targets(lines: &[SassLine]) -> Vec<(usize, Option<String>)> {
    lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.opcode == "EXIT" && l.operands.is_empty())
        .map(|(i, l)| (i, l.guard.clone()))
        .collect()
}

// ASSUME(L-ROSETTA): see ASSUMPTIONS.md
fn lower_line(
    l: &SassLine,
    labels: &HashMap<String, u32>,
    addr_to_idx: &HashMap<u32, usize>,
    meta: &CubinMeta,
    exits: &[(usize, Option<String>)],
) -> Result<Inst> {
    let lw = Lw { l };
    let parts: Vec<&str> = l.opcode.split('.').collect();
    let base = parts[0];
    let mods = &parts[1..];
    let has = |m: &str| mods.contains(&m);
    // Fail closed on modifiers: a token outside the audited set of the opcode is refused, so
    // no arm can silently ignore a modifier it does not know (VIADD.16x2, STG.256).
    // ASSUME(L-ROSETTA)
    if let Some(m) = unaudited_mod(base, mods) {
        bail!(lw.err(format!(
            "modifier .{m} of {base} not modeled (not in the audited set, modset.rs)"
        )));
    }
    let guard = match &l.guard {
        Some(g) => lw.pred(g)?,
        None => Pred::TRUE,
    };
    let mut uniform = base.starts_with('U') || matches!(base, "S2UR" | "LDCU" | "VOTEU");
    let op = match base {
        // FRND, F2F f64 <-> f16 / bf16 and f16 <-> bf16, RPCMOV, UBLKRED, VIMNMX[3] 16x2,
        // LEA.HI.X ~Rb, FCHK -Ra (agent L2's forms, l2.rs).
        _ if l2::claims(&lw, base, mods) => l2::lower(&lw, base, mods, meta)?,
        "NOP" => Op::Nop,
        // `MOVM.16.MT88 Rd, Ra` (movmatrix.sync.aligned.m8n8.trans.b16; probes/movm).
        "MOVM" if mods == ["16", "MT88"] => {
            uniform = false;
            Op::MovMatrix {
                d: lw.dst(lw.op(0)?.trim_end_matches(".reuse"))?,
                a: lw.val(lw.op(1)?.trim_end_matches(".reuse"), 4)?,
            }
        }
        // Tensor-map prefetch (prefetch.tensormap): a cache hint.
        "UTMACCTL" if has("PF") => Op::Nop,
        // `UBLKPF.L2 [URa], URb` (cp.async.bulk.prefetch.L2): a cache hint (probes/memhints).
        "UBLKPF" if mods == ["L2"] => Op::Nop,
        // `UTMAPF.L2.<n>D [URa], [URb], desc[..]` (cp.async.bulk.prefetch.tensor tile mode): a
        // cache hint (probes/tma_prefetch).
        "UTMAPF" if has("L2") && mods.len() == 2 => Op::Nop,
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
        // `CCTL.E.RML2 [Ra]` (discard.global.L2 [a], 128): the 128-byte line's contents become
        // indeterminate, a weak store of undefined data over the line, as on the PTX front:
        // one 128-byte store of a register nothing writes (V255, outside the R file's RZ
        // alias), so race checks, footprints and S-INIT see it as any store. ASSUME(S-INIT)
        "CCTL" if mods == ["E", "RML2"] && l.operands.len() == 1 => Op::St {
            space: Space::Global,
            bytes: 128,
            addr: wide_e_addr(lw.addr(lw.op(0)?)?, true),
            v: Val::Reg(
                Reg {
                    file: File::V,
                    idx: 255,
                },
                1,
            ),
            sync: None,
        },
        // NANOSLEEP(.SYNCS) is a timing hint (try_wait suspend); no semantic effect.
        // ASSUME(E-NANOSLEEP): see ASSUMPTIONS.md
        "NANOSLEEP" => Op::Nop,
        "EXIT" if l.operands.is_empty() => Op::Exit,
        // `@P0 EXIT P2` (torch's reduce_kernel: `if (mult != 0 && threadIdx.x != 0) return`):
        // the threads where both hold exit, as `@P0 BRA P2` branches (ASSUME(S-BRA-COND)).
        // Lowered as that branch to a plain EXIT of the function.
        "EXIT" if l.operands.len() == 1 && mods.is_empty() => {
            // The target: a plain EXIT, else an EXIT with this instruction's own guard (Triton's
            // `@!P0 EXIT P1` ... `@!P0 EXIT`): the branching lanes reach it directly, with the
            // guard still true there.
            let target = exits
                .iter()
                .find(|e| e.1.is_none())
                .or_else(|| exits.iter().find(|e| e.1.is_some() && e.1 == l.guard))
                .map(|e| e.0)
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
                // `c[0x0][UR5+-0x1c80]`: a negative offset, added mod 2^32 (probes/cbank_neg).
                let soff = |o: &str| match o.strip_prefix('-') {
                    Some(h) => hex(h).map(u32::wrapping_neg),
                    None => hex(o),
                };
                let (base, off) = match idx.split_once('+') {
                    Some((r, o)) => (lw.val(r, 4)?, soff(o).ok_or_else(|| lw.err("bad cbank"))?),
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
        // `VIADD.16x2 d, a, b` (add.u16x2 / add.s16x2; probes/viadd16x2): two 16-bit lanes,
        // each modulo 2^16, b a register or a 32-bit immediate. Fail closed: a negated or
        // inverted operand of .16x2, and any other VIADD modifier.
        "VIADD"
            if mods == ["16x2"]
                && lw.l.operands.len() == 3
                && !lw.l.operands[1..].iter().any(|o| o.starts_with(['-', '~', '|'])) =>
        {
            Op::Int {
                op: IntOp::Add16x2,
                bits: 32,
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                b: lw.val(lw.op(2)?, 4)?,
            }
        }
        // `VIADD.16x2 d, a, -b` (ptxas 13.3's sub.u16x2 / sub.s16x2): each lane a - b modulo
        // 2^16 (probes/sub16x2).
        "VIADD"
            if mods == ["16x2"]
                && lw.l.operands.len() == 3
                && !lw.l.operands[1].starts_with(['-', '~', '|'])
                && (lw.l.operands[2].starts_with("-R") || lw.l.operands[2].starts_with("-U")) =>
        {
            Op::Int {
                op: IntOp::Sub16x2,
                bits: 32,
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                b: lw.val(&lw.op(2)?[1..], 4)?,
            }
        }
        "VIADD" if !mods.is_empty() => {
            bail!(lw.err(format!("VIADD.{} not modeled", mods.join("."))))
        }
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
            let (b, not_b) = match b.strip_prefix('~').filter(|b| b.contains('R')) {
                Some(b) => (b, true),
                None => (b, false),
            };
            Op::LeaHi {
                d,
                a: lw.val(lw.op(i)?, 4)?,
                b: lw.val(b, 4)?,
                neg_b,
                not_b,
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
            bits: 32,
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            pos: Val::Imm(0),
            len: lw.val(lw.op(2)?, 4)?,
            signed: !has("U32"),
        },
        // I2F[.S64|.U64|.S16|.U16|.S8|.U8]: the source width selects the bits converted.
        // I2F[.F64][.S64|.U64|.U32|.S16|.U16|.S8|.U8][.RZ|.RM|.RP] (I2FP: the same); `.F64`:
        // an f64 result (libdevice sinf's large-argument reduction: `I2F.F64.S64`; GB300:
        // fixtures/s_f2i).
        "I2F" | "I2FP"
            if !has("F16")
                && !has("FTZ")
                && mods.iter().all(|m| {
                    matches!(
                        *m,
                        "F64"
                            | "F32"
                            | "S64"
                            | "U64"
                            | "S32"
                            | "U32"
                            | "S16"
                            | "U16"
                            | "S8"
                            | "U8"
                            | "RZ"
                            | "RM"
                            | "RP"
                    )
                }) =>
        {
            Op::I2F {
                f64: has("F64"),
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
            }
        }
        // F2I[.FTZ][.U32|.S64|.U64][.TRUNC|.FLOOR|.CEIL][.NTZ] Rd, Ra (cvt.r{n,z,m,p}i.s32.f32
        // ...): an f32 source rounded to an integer, saturated, NaN 0 (32-bit) / 2^63 (64-bit).
        // `.FTZ` and `.NTZ` do not change the integer (a subnormal rounds to 0 or +-1 only with
        // FLOOR / CEIL, see the exec arm); GB300: fixtures/s_lift_k, s_f2i.
        "F2I" if !has("F64") && !has("F16") => {
            if let Some(m) = mods.iter().find(|m| {
                !matches!(
                    **m,
                    "FTZ" | "NTZ" | "U32" | "S32" | "U64" | "S64" | "TRUNC" | "FLOOR" | "CEIL"
                )
            }) {
                bail!(lw.err(format!("F2I modifier not modeled: {m}")));
            }
            let rnd = match (has("TRUNC"), has("FLOOR"), has("CEIL")) {
                (true, false, false) => Rnd::Rz,
                (false, true, false) => Rnd::Rm,
                (false, false, true) => Rnd::Rp,
                (false, false, false) => Rnd::Rn,
                _ => bail!(lw.err("F2I with two roundings")),
            };
            // `F2I[.FTZ][.U32][.TRUNC|.FLOOR|.CEIL].NTZ Rd, -Ra` (ptxas folds `neg.f32` into
            // the conversion): the source's sign bit is flipped first, so the rounding and the
            // saturation act on -a (`F2I.FLOOR` gives floor(-a), `F2I.U32` 0 for a > 0) and a
            // NaN of either sign still gives 0. Only a data register on the 32-bit `.NTZ` forms;
            // `|Ra|`, `-|Ra|` and a negated source of any other kind stay refused.
            // GB300: probes/f2i_neg (all 16 forms). ASSUME(L-ROSETTA)
            let src = lw.op(1)?;
            let (src, neg) = match src.strip_prefix('-').filter(|r| !r.starts_with("0x")) {
                Some(r) => {
                    let data = matches!(Lw::reg_name(r), Some((File::R, i)) if i != RZ);
                    if !data
                        || !has("NTZ")
                        || has("S32")
                        || has("S64")
                        || has("U64")
                        || l.operands.len() != 2
                    {
                        bail!(lw.err(format!(
                            "F2I with the source {src} not modeled (only -R on the 32-bit .NTZ forms)"
                        )));
                    }
                    (r, true)
                }
                None => (src, false),
            };
            Op::F2I {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(src, 4)?,
                signed: !has("U32") && !has("U64"),
                bits: if has("U64") || has("S64") { 64 } else { 32 },
                rnd,
                ftz: has("FTZ"),
                f64: false,
                neg,
            }
        }
        // F2I[.S64|.U64|.U32].F64[.TRUNC|.FLOOR|.CEIL] Rd, Ra (cvt.r{z,n,m,p}i.{s,u}{32,64}.f64,
        // torch's cross entropy): an f64 source (register pair) rounded to an integer,
        // saturated, NaN 2^31 (32-bit) / 2^63 (64-bit). Other modifiers (.FTZ, .NTZ, .S32
        // spelled out, F16 sources) are not modeled. GB300: fixtures/s_f2i_f64, all 16 forms.
        // ASSUME(N-F2I-F64)
        "F2I" if has("F64") && !has("F16") => {
            if let Some(m) = mods.iter().find(|m| {
                !matches!(
                    **m,
                    "F64" | "U32" | "U64" | "S64" | "TRUNC" | "FLOOR" | "CEIL"
                )
            }) {
                bail!(lw.err(format!("F2I.F64 modifier not modeled: {m}")));
            }
            if mods.last() == Some(&"F64")
                && mods.len() > 1
                && !matches!(mods[0], "U32" | "U64" | "S64")
                || mods
                    .iter()
                    .filter(|m| matches!(**m, "U32" | "U64" | "S64"))
                    .count()
                    > 1
            {
                bail!(lw.err("F2I.F64 form not modeled"));
            }
            let rnd = match (has("TRUNC"), has("FLOOR"), has("CEIL")) {
                (true, false, false) => Rnd::Rz,
                (false, true, false) => Rnd::Rm,
                (false, false, true) => Rnd::Rp,
                (false, false, false) => Rnd::Rn,
                _ => bail!(lw.err("F2I with two roundings")),
            };
            Op::F2I {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 8)?,
                signed: !has("U32") && !has("U64"),
                bits: if has("U64") || has("S64") { 64 } else { 32 },
                rnd,
                ftz: false,
                f64: true,
                neg: false,
            }
        }
        // F2F.F32.F64[.RZ|.RM|.RP] (cvt.r{n,z,m,p}.f32.f64; probes/cvt_misc), F2F.F64.F32.
        "F2F"
            if mods == ["F64", "F32"]
                || mods.len() <= 3
                    && mods[..2] == ["F32", "F64"]
                    && mods[2..].iter().all(|m| matches!(*m, "RZ" | "RM" | "RP")) =>
        {
            let to64 = mods[0] == "F64";
            Op::F2F {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, if to64 { 4 } else { 8 })?,
                to64,
                rnd: match mods.get(2) {
                    Some(&"RZ") => Rnd::Rz,
                    Some(&"RM") => Rnd::Rm,
                    Some(&"RP") => Rnd::Rp,
                    _ => Rnd::Rn,
                },
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
        // `F2FP.SATFINITE.BF16.S2_6.UNPACK_B Rd, Ra[.H1], Rb.H0 | imm.H0`
        // (cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2; N-CVT-LP, probes/s2f6_cvt). Only
        // printed by `undis.rs` (nvdisasm 13.0 skips the encoding). The scale is the low half.
        "F2FP" if mods == ["SATFINITE", "BF16", "S2_6", "UNPACK_B"] => {
            let a = lw.op(1)?.replace(".reuse", "");
            let (a, hi) = match a.strip_suffix(".H1") {
                Some(r) => (r.to_string(), true),
                None => (a, false),
            };
            let sc = lw.op(2)?.replace(".reuse", "");
            let Some(sc) = sc.strip_suffix(".H0") else {
                bail!(lw.err("F2FP.S2_6 scale other than the low half (.H0) not modeled"));
            };
            Op::CvtS2f6x2 {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(&a, 4)?,
                hi,
                scale: lw.val(sc, 4)?,
            }
        }
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
                    // ptxas 13.3's bf16 decodes carry a ue8m0x2 scale pair: the immediate
                    // 0x7f7f.H0 (2^0 in both lanes) without `.scaled::n2::ue8m0`, a register
                    // with it. The decode op has no scale, so only the identity is lifted
                    // (MISS 61: a register scale was ignored; probes/cvt_lp_scaled). N-CVT-LP
                    let ops = &lw.l.operands;
                    if ops.len() > 3
                        || ops.get(2).is_some_and(|s| {
                            !s.replace(".reuse", "")
                                .strip_suffix(".H0")
                                .is_some_and(|v| matches!(lw.fval(v), Ok(Val::Imm(0x7f7f))))
                        })
                    {
                        bail!(lw.err("F2FP.UNPACK_B scale other than 0x7f7f.H0 not modeled"));
                    }
                    Op::CvtUnpack {
                        d,
                        a,
                        hi,
                        from,
                        to,
                        relu,
                        byte: 0,
                    }
                }
            }
        }
        // `F2FP [.SATFINITE] [.RELU] .TF32.F32.PACK_B [.RZ] Rd, Ra` (cvt.rn / .rz [.relu]
        // [.satfinite] .tf32.f32; probes/cvt_tf32_rn).
        "F2FP" if f2fp_tf32(mods).is_some() => {
            let (rz, sat, relu) = f2fp_tf32(mods).unwrap();
            Op::CvtTf32 {
                d: lw.dst(lw.op(0)?)?,
                a: lw.fval(lw.op(1)?)?,
                rz,
                sat,
                relu,
            }
        }
        // `.RZ`: toward zero (cvt.rz.f16x2.f32, probes/f2fp_rz). ASSUME(N-F2FP-RZ)
        "F2FP" if mods == ["F16", "F32", "PACK_AB"] || mods == ["F16", "F32", "PACK_AB", "RZ"] => {
            Op::F2fpF16 {
                d: lw.dst(lw.op(0)?)?,
                a: lw.fval(lw.op(1)?)?,
                b: lw.fval(lw.op(2)?)?,
                rz: mods.len() == 4,
            }
        }
        // FMUL.{D2,D4,D8,M2,M4,M8} [.FTZ] [.SAT]: the product scaled by 2^-1..-3 / 2^1..3 with one
        // rounding (ptxas folds a non-.rn mul.f32 times 0.125 .. 8 into it; probes/fmul_scale,
        // MISS 55). Round to nearest only (the directed forms are not probed). ASSUME(N-FALU)
        "FMUL"
            if mods
                .iter()
                .any(|m| matches!(*m, "D2" | "D4" | "D8" | "M2" | "M4" | "M8"))
                && mods.iter().all(|m| {
                    matches!(*m, "D2" | "D4" | "D8" | "M2" | "M4" | "M8" | "FTZ" | "SAT")
                })
                && mods
                    .iter()
                    .filter(|m| matches!(**m, "D2" | "D4" | "D8" | "M2" | "M4" | "M8"))
                    .count()
                    == 1 =>
        {
            let scale = if has("D2") {
                -1
            } else if has("D4") {
                -2
            } else if has("D8") {
                -3
            } else if has("M2") {
                1
            } else if has("M4") {
                2
            } else {
                3
            };
            if l.operands.len() != 3 {
                bail!(lw.err("operand count"));
            }
            let (a, na, aa) = lw.fmodval(lw.op(1)?)?;
            let (b, nb, ab) = lw.fmodval(lw.op(2)?)?;
            Op::FmulScale {
                d: lw.dst(lw.op(0)?)?,
                a,
                b,
                neg: [na, nb],
                abs: [aa, ab],
                scale,
                ftz: has("FTZ"),
                sat: has("SAT"),
            }
        }
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
        // `FMNMX[.FTZ][.NAN].XORSIGN d, |a|, |b|, p` (min / max .xorsign.abs.f32): the result
        // takes sign(a) XOR sign(b) of the operands as stored, whatever the |x| printed
        // (probes/fmnmx_xorsign: `|UR6|` holding -448 gives the flipped sign). Only the forms
        // ptxas emits are accepted: |register| or |uniform register|, or a non-negative
        // immediate. ASSUME(N-XORSIGN)
        "FMNMX" | "FMNMX3" if mods.iter().all(|m| matches!(*m, "FTZ" | "NAN" | "XORSIGN")) => {
            let n = if base == "FMNMX3" { 3 } else { 2 };
            if l.operands.len() != n + 2 {
                bail!(lw.err("operand count"));
            }
            let xorsign = has("XORSIGN");
            if xorsign && n != 2 {
                bail!(lw.err("FMNMX3.XORSIGN not modeled"));
            }
            let (mut v, mut neg, mut abs) = (Vec::new(), [false; 3], [false; 3]);
            for k in 0..n {
                let (x, ng, ab) = lw.fmodval(lw.op(1 + k)?)?;
                if xorsign {
                    let ok = match &x {
                        Val::Imm(c) => !ng && !ab && c & 0x8000_0000 == 0,
                        _ => !ng && ab,
                    };
                    if !ok {
                        bail!(lw.err(format!(
                            "FMNMX.XORSIGN operand {} not modeled (|register| or a non-negative immediate)",
                            lw.op(1 + k)?
                        )));
                    }
                }
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
                xorsign,
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
        // `DSETP.<cmp>.<AND|OR|XOR> Pd, PT, a, b, Pc`: Pd = (a cmp b) op Pc on f64 pairs
        // (fixtures/s_f64; `.LE` / `.LTU` with each combiner: probes/dsetp_ltu_le). The gate
        // admits no `.EQU` / `.NUM` / `.F` / `.T`. ASSUME(N-FALU)
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
        // `HSETP2[.BF16_V2].<cmp>.<bop> Pd, Pq, a, b, Pc` (`setp.<cmp>.f16x2` / `.bf16x2` and the
        // scalar forms): per lane, the halves (f16 or bf16) widened exactly to f32, compared;
        // Pd = lane 0 bop Pc, Pq = lane 1 bop Pc. Operands: `[-][|]R[|][.reuse][.H0_H0|.H1_H1]`,
        // RZ, or an immediate pair printed `hi, lo` (bits from the encoding, checked against the
        // print). ASSUME(L-ROSETTA): fixtures/s_routing_forms, probes/hsetp2 = GB300.
        // `HSET2[.BF16_V2][.BF].<cmp>.<bop> Rd, a, b, Pc`: the same comparison as a register
        // (probes/hset2).
        "HSETP2" | "HSET2" if {
            let m = if mods.first() == Some(&"BF16_V2") { &mods[1..] } else { mods };
            let m = if base == "HSET2" && m.first() == Some(&"BF") { &m[1..] } else { m };
            m.len() == 2
        } =>
        {
            let bf16 = mods.first() == Some(&"BF16_V2");
            let m = if bf16 { &mods[1..] } else { mods };
            let set = base == "HSET2";
            let bf = set && m.first() == Some(&"BF");
            let m = if bf { &m[1..] } else { m };
            let cmp = match m[0] {
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
            let bop = match m[1] {
                "AND" => BoolOp::And,
                "OR" => BoolOp::Or,
                "XOR" => BoolOp::Xor,
                _ => bail!(lw.err("HSETP2 combiner not modeled")),
            };
            let (d, q) = if set {
                (lw.dst(lw.op(0)?)?, None)
            } else {
                (
                    lw.pdst(lw.op(0)?)?,
                    match lw.op(1)? {
                        "PT" => None,
                        p => Some(lw.pdst(p)?),
                    },
                )
            };
            let first = if set { 1 } else { 2 };
            let toks: Vec<String> = (first..l.operands.len() - 1)
                .map(|i| lw.op(i).map(|x| x.replace(".reuse", "")))
                .collect::<Result<_>>()?;
            let c = lw.pred(lw.op(l.operands.len() - 1)?)?;
            let is_reg = |x: &str| {
                let r = x.trim_start_matches('-').trim_start_matches('|');
                let r = r.split(['.', '|']).next().unwrap_or(r);
                r == "RZ" || Lw::reg_name(r).is_some()
            };
            let imm_at = match toks.len() {
                2 => None,
                3 if !is_reg(&toks[1]) && !is_reg(&toks[2]) => Some(1),
                _ => bail!(lw.err("HSETP2 operand count")),
            };
            let mut ops: Vec<(Val, HSel, bool, bool)> = Vec::new();
            let mut i = 0;
            while i < toks.len() {
                if Some(ops.len()) == imm_at {
                    let bits = (l.lo.ok_or_else(|| lw.err("HSETP2 immediate without encoding"))?
                        >> 32
                        & 0xffff_ffff) as u32;
                    for (tok, half) in [(&toks[i], bits >> 16), (&toks[i + 1], bits & 0xffff)] {
                        let t = tok.trim().trim_start_matches('+');
                        let ok = if bf16 {
                            let w = f32::from_bits(half << 16);
                            match t {
                                "QNAN" | "-QNAN" => w.is_nan(),
                                "INF" => w == f32::INFINITY,
                                "-INF" => w == f32::NEG_INFINITY,
                                x => x.parse::<f64>().is_ok_and(|p| p == w as f64 || (p as f32) == w),
                            }
                        } else {
                            match t {
                                "QNAN" | "-QNAN" => half & 0x7c00 == 0x7c00 && half & 0x3ff != 0,
                                "INF" => half == 0x7c00,
                                "-INF" => half == 0xfc00,
                                x => parse_half(x) == Some(half as u16),
                            }
                        };
                        if !ok {
                            bail!(lw.err(format!("HSETP2 immediate {tok} is not the encoding's {half:#06x}")));
                        }
                    }
                    ops.push((Val::Imm(bits as u64), HSel::Pair, false, false));
                    i += 2;
                    continue;
                }
                let t = toks[i].as_str();
                let (neg, t) = match t.strip_prefix('-') {
                    Some(r) => (true, r),
                    None => (false, t),
                };
                let (r, sel) = if let Some(r) = t.strip_suffix(".H0_H0") {
                    (r, HSel::H0)
                } else if let Some(r) = t.strip_suffix(".H1_H1") {
                    (r, HSel::H1)
                } else {
                    (t, HSel::Pair)
                };
                let (abs, r) = match r.strip_prefix('|').and_then(|x| x.strip_suffix('|')) {
                    Some(x) => (true, x),
                    None => (false, r),
                };
                if r.contains('.') || r.contains('|') || !is_reg(r) {
                    bail!(lw.err(format!("HSETP2 operand {} not modeled", toks[i])));
                }
                let v = if r == "RZ" { Val::Imm(0) } else { lw.val(r, 4)? };
                ops.push((v, sel, neg, abs));
                i += 1;
            }
            if set {
                Op::Hset2 {
                    d,
                    cmp,
                    a: ops[0].0.clone(),
                    b: ops[1].0.clone(),
                    sel: [ops[0].1, ops[1].1],
                    neg: [ops[0].2, ops[1].2],
                    abs: [ops[0].3, ops[1].3],
                    c,
                    bop,
                    f16: !bf16,
                    bf,
                }
            } else {
                Op::Hsetp2Bf16 {
                    d,
                    q,
                    cmp,
                    a: ops[0].0.clone(),
                    b: ops[1].0.clone(),
                    sel: [ops[0].1, ops[1].1],
                    neg: [ops[0].2, ops[1].2],
                    abs: [ops[0].3, ops[1].3],
                    c,
                    bop,
                    f16: !bf16,
                }
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
                // (fixtures/s_lift_k = GB300). `FSET.BF.NAN.AND Rd, a, b, Pc`: 1.0f when an
                // operand is a NaN and Pc holds (probes/fset_nan). ASSUME(L-ROSETTA)
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
        // `-Rb` / `-URb`: a subtraction fused (a - b). A negated operand of any other kind
        // (`-c[..]`) is refused; a negative immediate is parsed as such. Before LM's fix the
        // `-URb` form lost its sign (q_split_seqlens_kernel: max(c0 - rows, 0) simulated as
        // c0 + rows; PTX front and GB300 agree on c0 - rows). ASSUME(L-ROSETTA)
        "VIADDMNMX" if mods.iter().all(|m| *m == "U32") => Op::AddMinMax {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(1)?, 4)?,
            b: {
                let bt = lw.op(2)?.replace(".reuse", "");
                match bt.strip_prefix('-') {
                    Some(r) if r.starts_with('R') || r.starts_with("UR") => lw.val(r, 4)?,
                    Some(r) if r.starts_with(|c: char| c.is_ascii_digit()) => lw.val(&bt, 4)?,
                    Some(_) => bail!(lw.err("VIADDMNMX negated operand form not modeled")),
                    None => lw.val(&bt, 4)?,
                }
            },
            neg_b: lw.op(2)?.starts_with("-R") || lw.op(2)?.starts_with("-UR"),
            c: lw.val(lw.op(3)?, 4)?,
            signed: !has("U32"),
            max: match lw.op(4)? {
                "PT" => false,
                "!PT" => true,
                _ => bail!(lw.err("VIADDMNMX with a predicate selector not modeled")),
            },
            relu: false,
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
                x16: false,
                relu: false,
            }
        }
        "P2R" if mods.is_empty() && lw.op(1)? == "PR" => Op::P2r {
            d: lw.dst(lw.op(0)?)?,
            a: lw.val(lw.op(2)?, 4)?,
            mask: hex(lw.op(3)?).ok_or_else(|| lw.err("bad mask"))? & 0x7f,
            uniform: false,
            shift: 0,
        },
        // `UP2UR URd, UPR, URa, mask` (CuTe DSL kernels materializing uniform flags): P2R on
        // the uniform predicates. `.B3` (byte 3; ptxas's save of UPR around a fence sequence,
        // mask 0xff with bit 7 = UPT) places the bits in that byte. ASSUME(L-ROSETTA)
        "UP2UR"
            if matches!(mods, [] | ["B1"] | ["B2"] | ["B3"]) && lw.op(1)? == "UPR" =>
        {
            let m = hex(lw.op(3)?).ok_or_else(|| lw.err("bad mask"))?;
            // Bit 7 (UPT) only in the byte forms, whose byte goes back through UR2UP's
            // UP0..UP6.
            if m & !0xff != 0 || (mods.is_empty() && m & 0x80 != 0) {
                bail!(lw.err(format!("UP2UR mask {m:#x} not modeled")));
            }
            let shift = match mods {
                ["B1"] => 8,
                ["B2"] => 16,
                ["B3"] => 24,
                _ => 0,
            };
            Op::P2r {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(2)?, 4)?,
                mask: m,
                uniform: true,
                shift,
            }
        }
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
            // shape: 0 = 32x32b / 16x32bx2, 1 = 16x128b, 2 = 16x256b (probes/tmem_st16).
            let (n, half, shape): (usize, Option<u8>, u8) = match mods {
                // Bare `STTM tmem[..], R` is tcgen05.st.32x32b.x1 (probes/bf16_selp).
                [] => (1, None, 0),
                [x] if x.starts_with('x') => {
                    (x[1..].parse().map_err(|_| lw.err("STTM count"))?, None, 0)
                }
                [s @ ("16dp32bit_t0_t15" | "16dp32bit_t16_t31"), x] if x.starts_with('x') => (
                    x[1..].parse().map_err(|_| lw.err("STTM count"))?,
                    Some(s.ends_with("t31") as u8),
                    0,
                ),
                [s @ ("16dp256bit" | "16dp128bit"), x @ ..] if x.len() <= 1 => {
                    let x: usize = match x.first() {
                        Some(x) => x
                            .strip_prefix('x')
                            .and_then(|x| x.parse().ok())
                            .ok_or_else(|| lw.err("STTM count"))?,
                        None => 1,
                    };
                    if *s == "16dp256bit" {
                        (4 * x, None, 2)
                    } else {
                        (2 * x, None, 1)
                    }
                }
                _ => bail!(lw.err(
                    "STTM shape not modeled yet (only 32x32b, 16x32bx2, 16x128b, 16x256b .xN)"
                )),
            };
            // A negative offset (`STTM.x32 tmem[UR7+-0x60]`), as LDTM (probes/tmem_stneg).
            let (taddr, off) = lw.bracket_neg(lw.op(0)?, "tmem", 1, true)?;
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
                s16x256b: shape == 2,
                s16x128b: shape == 1,
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
                b8: false,
            }
        }
        // STSM.8.MT168[.N] [Ra], Rb: stmatrix.m16n8.xN.trans.b8, registers Rb..Rb+N-1.
        // probes/stsm_b8 = GB300. ASSUME(L-ROSETTA)
        "STSM" if mods.first() == Some(&"8") && has("MT168") => {
            uniform = false;
            let n: u16 = match mods {
                ["8", "MT168"] => 1,
                ["8", "MT168", "2"] => 2,
                ["8", "MT168", "4"] => 4,
                _ => bail!(lw.err(format!("{} not modeled", l.opcode))),
            };
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
                trans: true,
                b8: true,
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
                    neg_b: false,
                }
            }
        }
        // `SHF.{L,R}[.W].{U32,S32,U64,S64}[.HI] d, lo, sh, hi`: funnel shift of hi:lo by the u32
        // `sh`, clamped to 32 (.U32 / .S32) or 64 (.U64 / .S64), or mod 32 with .W (GB300:
        // fixtures/s_shf, every amount class incl. >= 64 and >= 2^31). ASSUME(L-SHF)
        "SHF" | "USHF" => {
            let (mut dir, mut ty, mut wrap, mut high) = (None, None, false, false);
            for m in mods {
                match *m {
                    "L" | "R" if dir.is_none() => dir = Some(*m == "R"),
                    "U32" | "S32" | "U64" | "S64" if ty.is_none() => ty = Some(*m),
                    "W" => wrap = true,
                    "HI" => high = true,
                    _ => bail!(lw.err("SHF modifier not modeled")),
                }
            }
            let (Some(right), Some(ty)) = (dir, ty) else {
                bail!(lw.err("SHF without a direction or type"));
            };
            let wide = ty.ends_with("64");
            if wrap && wide {
                bail!(lw.err("SHF.W with a 64-bit type not modeled"));
            }
            Op::Shf {
                d: lw.dst(lw.op(0)?)?,
                lo: lw.val(lw.op(1)?, 4)?,
                sh: lw.val(lw.op(2)?, 4)?,
                hi: lw.val(lw.op(3)?, 4)?,
                right,
                signed: ty.starts_with('S'),
                high,
                wide,
                wrap,
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
        // `PLOP3.LUT Pd, Pd2, a, b, c, lut, lut2`: Pd = lut(a, b, c), Pd2 = lut2(a, b, c)
        // (GB300: fixtures/s_lift_l, the block-sparsity kernel's all / any `P1, P0, !P0, PT,
        // !P1, 0x80, 0xc8`). With Pd2 = PT, ptxas fills lut2 with anything.
        "PLOP3" | "UPLOP3" => {
            if !mods.is_empty() && mods != ["LUT"] || l.operands.len() != 7 {
                bail!(lw.err("PLOP3 form not modeled"));
            }
            let lut2 = hex(lw.op(6)?).ok_or_else(|| lw.err("bad lut"))? as u8;
            let d2 = match lw.op(1)? {
                "PT" | "UPT" => None,
                p => Some((lw.pdst(p)?, lut2)),
            };
            Op::Plop3 {
                d: lw.pdst(lw.op(0)?)?,
                a: lw.pred(lw.op(2)?)?,
                b: lw.pred(lw.op(3)?)?,
                c: lw.pred(lw.op(4)?)?,
                lut: hex(lw.op(5)?).ok_or_else(|| lw.err("bad lut"))? as u8,
                d2,
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
                mask: None,
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
                mask: None,
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
            // `IMAD.WIDE[.U32] d, a, b, -c`: the product minus the 64-bit c, mod 2^64
            // (probes/int_misc = GB300); with a carry-in / carry-out, or `~c`, not modeled.
            // ASSUME(L-ROSETTA)
            if cbytes == 8 && (not_c || neg_c && (cout.is_some() || cin.is_some() || hi)) {
                bail!(lw.err("negated 64-bit IMAD addend not modeled with this form"));
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
        // Packed bf16: `HMUL2.BF16_V2 d, a, b` (a*b + -0), `HADD2.BF16_V2 d, a, b` (a*1 + b),
        // `HFMA2.BF16_V2[.RELU] d, a, b, c`. Operands: `[-]R[.H0_H0|.H1_H1]`, `[-]RZ[...]`, a
        // uniform register, or an immediate pair printed `hi, lo` (bits from the encoding).
        // The table's other modifiers (.FTZ, .SAT, .FMZ, .OOB, .F32 with BF16_V2) are refused.
        // ASSUME(N-BF16X2): probes/bf16x2.
        "HMUL2" | "HADD2" | "HFMA2" if mods.first() == Some(&"BF16_V2") => {
            let relu = match &mods[1..] {
                [] => false,
                ["RELU"] if base == "HFMA2" => true,
                m => bail!(lw.err(format!("{base}.BF16_V2 modifier .{} not modeled", m.join(".")))),
            };
            let n = if base == "HFMA2" { 3 } else { 2 };
            // Source tokens: one per operand, an immediate pair taking two.
            let toks: Vec<String> = (1..l.operands.len())
                .map(|i| lw.op(i).map(|x| x.replace(".reuse", "")))
                .collect::<Result<_>>()?;
            let is_reg = |x: &str| {
                let r = x.trim_start_matches('-');
                let r = r.split('.').next().unwrap_or(r);
                r == "RZ" || r == "URZ" || Lw::reg_name(r).is_some()
            };
            let imm_at = match (toks.len(), n) {
                (2, 2) | (3, 3) => None,
                (3, 2) if !is_reg(&toks[1]) && !is_reg(&toks[2]) => Some(1),
                (4, 3) if !is_reg(&toks[1]) && !is_reg(&toks[2]) => Some(1),
                (4, 3) if !is_reg(&toks[2]) && !is_reg(&toks[3]) => Some(2),
                _ => bail!(lw.err("operand count")),
            };
            let mut ops: Vec<(Val, HSel, bool, bool)> = Vec::new();
            let mut i = 0;
            while i < toks.len() {
                if Some(ops.len()) == imm_at {
                    let bits = (l.lo.ok_or_else(|| lw.err("bf16 immediate without encoding"))?
                        >> 32
                        & 0xffff_ffff) as u32;
                    // The printed values must be the encoding's halves (hi first).
                    for (tok, half) in [(&toks[i], bits >> 16), (&toks[i + 1], bits & 0xffff)] {
                        let w = f32::from_bits(half << 16);
                        let ok = match tok.trim_start_matches('+') {
                            "QNAN" | "-QNAN" => w.is_nan(),
                            "INF" => w == f32::INFINITY,
                            "-INF" => w == f32::NEG_INFINITY,
                            x => x.parse::<f64>().is_ok_and(|p| p == w as f64 || (p as f32) == w),
                        };
                        if !ok {
                            bail!(lw.err(format!("bf16 immediate {tok} is not the encoding's {half:#06x}")));
                        }
                    }
                    ops.push((Val::Imm(bits as u64), HSel::Pair, false, false));
                    i += 2;
                    continue;
                }
                let t = toks[i].as_str();
                let (neg, t) = match t.strip_prefix('-') {
                    Some(r) => (true, r),
                    None => (false, t),
                };
                let (r, sel) = if let Some(r) = t.strip_suffix(".H0_H0") {
                    (r, HSel::H0)
                } else if let Some(r) = t.strip_suffix(".H1_H1") {
                    (r, HSel::H1)
                } else {
                    (t, HSel::Pair)
                };
                // `|R|` / `|R|.H0_H0`: the halves' sign bits cleared (Triton's bf16 abs).
                let (abs, r) = match r.strip_prefix('|').and_then(|x| x.strip_suffix('|')) {
                    Some(x) => (true, x),
                    None => (false, r),
                };
                if r.contains('.') || r.contains('|') || !is_reg(r) {
                    bail!(lw.err(format!("{base}.BF16_V2 operand {} not modeled", toks[i])));
                }
                let v = if r == "RZ" || r == "URZ" {
                    Val::Imm(0)
                } else {
                    lw.val(r, 4)?
                };
                ops.push((v, sel, neg, abs));
                i += 1;
            }
            let one = (Val::Imm(0x3f80_3f80), HSel::Pair, false, false);
            let mzero = (Val::Imm(0x8000_8000), HSel::Pair, false, false);
            let [a, b, c] = match base {
                "HMUL2" => [ops[0].clone(), ops[1].clone(), mzero],
                "HADD2" => [ops[0].clone(), one, ops[1].clone()],
                _ => [ops[0].clone(), ops[1].clone(), ops[2].clone()],
            };
            Op::Hfma2Bf16 {
                d: lw.dst(lw.op(0)?)?,
                a: a.0,
                b: b.0,
                c: c.0,
                sel: [a.1, b.1, c.1],
                neg: [a.2, b.2, c.2],
                abs: [a.3, b.3, c.3],
                relu,
            }
        }
        // `FHFMA.BF16[.RZ|.RM|.RP][.SAT] d, a[.H1], b[.H1], [-]c` / `FHADD.BF16[...] d, a[.H1],
        // [-]c` (fma / add / sub `.f32.bf16`): bf16 halves times / plus an f32. Only the forms
        // probes/bf16_ext covers: register operands, a sign flip on c only. ASSUME(N-BF16X2)
        // Without `.BF16` the halves are f16 (`.f32.f16`, probes/fhadd). ASSUME(N-FHADD)
        "FHFMA" | "FHADD"
            if mods
                .iter()
                .skip(usize::from(mods.first() == Some(&"BF16")))
                .all(|m| matches!(*m, "RZ" | "RM" | "RP" | "SAT"))
                && mods.iter().filter(|m| matches!(**m, "RZ" | "RM" | "RP")).count() <= 1
                && mods.iter().filter(|m| **m == "SAT").count() <= 1 =>
        {
            let f16 = mods.first() != Some(&"BF16");
            let n = if base == "FHFMA" { 3 } else { 2 };
            if l.operands.len() != n + 1 {
                bail!(lw.err("operand count"));
            }
            let half = |k: usize| -> Result<(Val, bool)> {
                let t = lw.op(k)?.replace(".reuse", "");
                let (r, hi) = match (t.strip_suffix(".H1"), t.strip_suffix(".H0")) {
                    (Some(r), _) => (r.to_string(), true),
                    (_, Some(r)) => (r.to_string(), false),
                    _ => (t.clone(), false),
                };
                if r.contains(['-', '|', '.']) || Lw::reg_name(&r).is_none() {
                    bail!(lw.err(format!("{base}.BF16 operand {t} not modeled")));
                }
                Ok((lw.val(&r, 4)?, hi))
            };
            let (a, ahi) = half(1)?;
            let (b, bhi) = if n == 3 {
                half(2)?
            } else {
                (Val::Imm(if f16 { 0x3c00 } else { 0x3f80 }), false)
            };
            let ct = lw.op(n)?.replace(".reuse", "");
            let (negc, cr) = match ct.strip_prefix('-') {
                Some(r) => (true, r.to_string()),
                None => (false, ct.clone()),
            };
            if cr.contains(['|', '.']) || (cr != "RZ" && Lw::reg_name(&cr).is_none()) {
                bail!(lw.err(format!("{base}.BF16 operand {ct} not modeled")));
            }
            let c = if cr == "RZ" { Val::Imm(0) } else { lw.val(&cr, 4)? };
            Op::FfmaBf16 {
                d: lw.dst(lw.op(0)?)?,
                a,
                b,
                c,
                hi: [ahi, bhi],
                neg: [false, false, negc],
                rnd: sass_rnd(mods),
                sat: has("SAT"),
                f16,
            }
        }
        // `HMNMX2.BF16_V2[.NAN][.XORSIGN] d, a[.Hx_Hx], b[.Hx_Hx], p` (min / max .bf16x2 /
        // .bf16): `.XORSIGN` takes `|a|`, `|b|` (signs from the stored operands, as
        // FMNMX.XORSIGN); register operands only. ASSUME(N-BF16X2)
        "HMNMX2"
            if mods.first() == Some(&"BF16_V2")
                && mods[1..].iter().all(|m| matches!(*m, "NAN" | "XORSIGN")) =>
        {
            // b may be a bf16 immediate pair (`HMNMX2.BF16_V2 R0, R0, +INF , +INF , PT`).
            let imm = l.operands.len() == 5;
            if l.operands.len() != 4 && !imm {
                bail!(lw.err("operand count"));
            }
            let xorsign = has("XORSIGN");
            let src = |k: usize| -> Result<(Val, HSel)> {
                let t = lw.op(k)?.replace(".reuse", "");
                if t == "RZ" && !xorsign {
                    return Ok((Val::Imm(0), HSel::Pair));
                }
                let (abs, r) = match t.strip_prefix('|') {
                    Some(x) => match x.split_once('|') {
                        Some((r, rest)) => (true, format!("{r}{rest}")),
                        None => bail!(lw.err(format!("HMNMX2 operand {t} not modeled"))),
                    },
                    None => (false, t.clone()),
                };
                let (r, sel) = if let Some(r) = r.strip_suffix(".H0_H0") {
                    (r.to_string(), HSel::H0)
                } else if let Some(r) = r.strip_suffix(".H1_H1") {
                    (r.to_string(), HSel::H1)
                } else {
                    (r, HSel::Pair)
                };
                if abs != xorsign || r.contains(['-', '.']) || Lw::reg_name(&r).is_none() {
                    bail!(lw.err(format!("HMNMX2.BF16_V2 operand {t} not modeled")));
                }
                Ok((lw.val(&r, 4)?, sel))
            };
            let (a, sa) = src(1)?;
            let (b, sb) = if imm {
                if xorsign {
                    bail!(lw.err("HMNMX2.XORSIGN with an immediate not modeled"));
                }
                let bits = half_imm_pair(l.lo, [lw.op(2)?, lw.op(3)?], true)
                    .map_err(|e| lw.err(e))?;
                (Val::Imm(bits as u64), HSel::Pair)
            } else {
                src(2)?
            };
            Op::Hmnmx2Bf16 {
                d: lw.dst(lw.op(0)?)?,
                a,
                b,
                sel: [sa, sb],
                min: lw.pred(lw.op(if imm { 4 } else { 3 })?)?,
                nan: has("NAN"),
                xorsign,
            }
        }
        // `QMUL4.E4M3x4.E2M1x4.E4M3x4 d, Ra.N3210 | Ra.N7654, Rb.B0` (trtllm-gen FMHA's e2m1 KV
        // dequantization): four e2m1 nibbles of a times the e4m3 byte 0 of b, rounded to e4m3.
        // Only the byte-0 scale is probed (probes/qmul4 = GB300). ASSUME(N-QMUL4)
        "QMUL4" if mods == ["E4M3x4", "E2M1x4", "E4M3x4"] => {
            if l.operands.len() != 3 {
                bail!(lw.err("operand count"));
            }
            let a = lw.op(1)?.replace(".reuse", "");
            let b = lw.op(2)?.replace(".reuse", "");
            let (ra, hi) = match (a.strip_suffix(".N3210"), a.strip_suffix(".N7654")) {
                (Some(r), _) => (r.to_string(), false),
                (_, Some(r)) => (r.to_string(), true),
                _ => bail!(lw.err(format!("QMUL4 operand {a} not modeled"))),
            };
            let Some(rb) = b.strip_suffix(".B0") else {
                bail!(lw.err(format!("QMUL4 operand {b} not modeled")));
            };
            if Lw::reg_name(&ra).is_none() || Lw::reg_name(rb).is_none() {
                bail!(lw.err("QMUL4 with a non-register operand not modeled"));
            }
            Op::Qmul4 {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(&ra, 4)?,
                b: lw.val(rb, 4)?,
                hi,
                bsel: 0,
            }
        }
        // `HMNMX2[.NAN][.XORSIGN] d, a[.Hx_Hx], b[.Hx_Hx], p` (min / max [.NaN] [.xorsign.abs]
        // .f16x2 / .f16). Register operands `[|]R[|]`: under `.XORSIGN` both are `|R|` (the
        // signs still from the stored values, as FMNMX.XORSIGN); otherwise `|R|` clears the
        // sign (ptxas's fold of abs.f16x2). b may be an f16 immediate pair (`65504, 65504`),
        // non-negative under `.XORSIGN`. probes/hmnmx2 = GB300. ASSUME(N-HMNMX2)
        "HMNMX2" if mods.iter().all(|m| matches!(*m, "NAN" | "XORSIGN")) => {
            let imm = l.operands.len() == 5;
            if l.operands.len() != 4 && !imm {
                bail!(lw.err("operand count"));
            }
            let xorsign = has("XORSIGN");
            let src = |k: usize| -> Result<(Val, HSel, bool)> {
                let t = lw.op(k)?.replace(".reuse", "");
                let (abs, r) = match t.strip_prefix('|') {
                    Some(x) => match x.split_once('|') {
                        Some((r, rest)) => (true, format!("{r}{rest}")),
                        None => bail!(lw.err(format!("HMNMX2 operand {t} not modeled"))),
                    },
                    None => (false, t.clone()),
                };
                let (r, sel) = if let Some(r) = r.strip_suffix(".H0_H0") {
                    (r.to_string(), HSel::H0)
                } else if let Some(r) = r.strip_suffix(".H1_H1") {
                    (r.to_string(), HSel::H1)
                } else {
                    (r, HSel::Pair)
                };
                if (xorsign && !abs) || r.contains(['-', '.']) || Lw::reg_name(&r).is_none() {
                    bail!(lw.err(format!("HMNMX2 operand {t} not modeled")));
                }
                Ok((lw.val(&r, 4)?, sel, abs && !xorsign))
            };
            let (a, sa, aa) = src(1)?;
            let (b, sb, ab) = if imm {
                let bits = half_imm_pair(l.lo, [lw.op(2)?, lw.op(3)?], false)
                    .map_err(|e| lw.err(e))?;
                if xorsign && bits & 0x8000_8000 != 0 {
                    bail!(lw.err("HMNMX2.XORSIGN with a negative immediate not modeled"));
                }
                (Val::Imm(bits as u64), HSel::Pair, false)
            } else {
                src(2)?
            };
            Op::Hmnmx2F16 {
                d: lw.dst(lw.op(0)?)?,
                a,
                b,
                sel: [sa, sb],
                abs: [aa, ab],
                min: lw.pred(lw.op(if imm { 4 } else { 3 })?)?,
                nan: has("NAN"),
                xorsign,
            }
        }
        // `MATCH.ANY[.U64] Rd, Ra` (match.any.sync.b32 / .b64; ptxas emits WARPSYNC first for a
        // partial membermask): per lane the mask of active lanes holding the same value.
        // probes/match_any = GB300.
        "MATCH" if mods == ["ANY"] || mods == ["ANY", "U64"] => {
            uniform = false;
            let wide = mods.len() == 2;
            Op::MatchAny {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, if wide { 8 } else { 4 })?,
                wide,
            }
        }
        // `VHMNMX[.BF16_V2][.NAN] d, a, b, c, p`: three-input half min / max (ptxas fuses two
        // HMNMX2). Operands `[|]R[|][.reuse][.H0_H0|.H1_H1]`. probes/vhmnmx = GB300.
        "VHMNMX" if mods.iter().all(|m| matches!(*m, "BF16_V2" | "NAN")) => {
            if l.operands.len() != 5 {
                bail!(lw.err("operand count"));
            }
            let src = |k: usize| -> Result<(Val, HSel, bool)> {
                let t = lw.op(k)?.replace(".reuse", "");
                let (r, sel) = if let Some(r) = t.strip_suffix(".H0_H0") {
                    (r, HSel::H0)
                } else if let Some(r) = t.strip_suffix(".H1_H1") {
                    (r, HSel::H1)
                } else {
                    (t.as_str(), HSel::Pair)
                };
                let (abs, r) = match r.strip_prefix('|').and_then(|x| x.strip_suffix('|')) {
                    Some(x) => (true, x),
                    None => (false, r),
                };
                if r == "RZ" {
                    return Ok((Val::Imm(0), sel, abs));
                }
                if r.contains(['-', '.', '|']) || Lw::reg_name(r).is_none() {
                    bail!(lw.err(format!("VHMNMX operand {t} not modeled")));
                }
                Ok((lw.val(r, 4)?, sel, abs))
            };
            let (a, b, c) = (src(1)?, src(2)?, src(3)?);
            Op::Vhmnmx {
                d: lw.dst(lw.op(0)?)?,
                a: a.0,
                b: b.0,
                c: c.0,
                sel: [a.1, b.1, c.1],
                abs: [a.2, b.2, c.2],
                min: lw.pred(lw.op(4)?)?,
                nan: has("NAN"),
                f16: !has("BF16_V2"),
            }
        }
        // f16x2: `HFMA2[.MMA] d, a, b, c`, `HMUL2 d, a, b` (a*b + -0), `HADD2 d, a, c` (a*1 + c).
        // Operands: `[-]R[.reuse][.H0_H0|.H1_H1]`, RZ, or an f16 immediate pair printed `hi, lo`
        // (bits from the encoding, checked against the print). A sign flip only on a and c.
        // probes/f16_arith = GB300. ASSUME(L-ROSETTA)
        "HFMA2" | "HMUL2" | "HADD2" if mods.iter().all(|m| *m == "MMA") => {
            let n = if base == "HFMA2" { 3 } else { 2 };
            let toks: Vec<String> = (1..l.operands.len())
                .map(|i| lw.op(i).map(|x| x.replace(".reuse", "")))
                .collect::<Result<_>>()?;
            let is_reg = |x: &str| {
                let r = x.trim_start_matches('-');
                let r = r.split('.').next().unwrap_or(r);
                r == "RZ" || Lw::reg_name(r).is_some()
            };
            let imm_at = match (toks.len(), n) {
                (2, 2) | (3, 3) => None,
                (3, 2) if !is_reg(&toks[1]) && !is_reg(&toks[2]) => Some(1),
                (4, 3) if !is_reg(&toks[1]) && !is_reg(&toks[2]) => Some(1),
                (4, 3) if !is_reg(&toks[2]) && !is_reg(&toks[3]) => Some(2),
                _ => bail!(lw.err("operand count")),
            };
            let mut ops: Vec<(Val, HSel, bool)> = Vec::new();
            let mut abs_a = false;
            let mut i = 0;
            while i < toks.len() {
                if Some(ops.len()) == imm_at {
                    let bits = (l.lo.ok_or_else(|| lw.err("f16 immediate without encoding"))? >> 32
                        & 0xffff_ffff) as u32;
                    for (tok, half) in [(&toks[i], bits >> 16), (&toks[i + 1], bits & 0xffff)] {
                        let ok = match tok.trim_start_matches('+') {
                            "QNAN" | "-QNAN" => half & 0x7c00 == 0x7c00 && half & 0x3ff != 0,
                            "INF" => half == 0x7c00,
                            "-INF" => half == 0xfc00,
                            x => parse_half(x) == Some(half as u16),
                        };
                        if !ok {
                            bail!(lw.err(format!("f16 immediate {tok} is not the encoding's {half:#06x}")));
                        }
                    }
                    ops.push((Val::Imm(bits as u64), HSel::Pair, false));
                    i += 2;
                    continue;
                }
                let t = toks[i].as_str();
                let (neg, t) = match t.strip_prefix('-') {
                    Some(r) => (true, r),
                    None => (false, t),
                };
                // `|R|` / `|R|.H0_H0` (a only): the absolute value (PTX abs.f16x2;
                // probes/f16_absneg).
                let stripped;
                let t = match t.strip_prefix('|').and_then(|x| x.split_once('|')) {
                    Some((r, rest)) if ops.is_empty() && !neg => {
                        abs_a = true;
                        stripped = format!("{r}{rest}");
                        stripped.as_str()
                    }
                    _ => t,
                };
                let (r, sel) = if let Some(r) = t.strip_suffix(".H0_H0") {
                    (r, HSel::H0)
                } else if let Some(r) = t.strip_suffix(".H1_H1") {
                    (r, HSel::H1)
                } else {
                    (t, HSel::Pair)
                };
                if r.contains('.') || r.contains('|') || !is_reg(r) {
                    bail!(lw.err(format!("{base} operand {} not modeled", toks[i])));
                }
                let v = if r == "RZ" { Val::Imm(0) } else { lw.val(r, 4)? };
                ops.push((v, sel, neg));
                i += 1;
            }
            let one = (Val::Imm(0x3c00_3c00), HSel::Pair, false);
            let mzero = (Val::Imm(0x8000_8000), HSel::Pair, false);
            let [a, b, c] = match base {
                "HMUL2" => [ops[0].clone(), ops[1].clone(), mzero],
                "HADD2" => [ops[0].clone(), one, ops[1].clone()],
                _ => [ops[0].clone(), ops[1].clone(), ops[2].clone()],
            };
            if b.2 {
                bail!(lw.err(format!("{base} with a negated b not modeled")));
            }
            Op::Hfma2 {
                d: lw.dst(lw.op(0)?)?,
                a: a.0,
                b: b.0,
                c: c.0,
                abs_a,
                neg_a: a.2,
                neg_c: c.2,
                sel: [a.1, b.1, c.1],
            }
        }
        // `.RZ` / `.RM` / `.RP`: one rounding in that mode (probes/bf16_ext = GB300). `.FTZ`:
        // subnormal inputs and results flushed to signed zero per lane (probes/f32x2_ftz =
        // GB200).
        "FFMA2" | "FMUL2" | "FADD2"
            if mods.iter().filter(|m| **m != "FTZ").count() <= 1
                && mods.iter().filter(|m| **m == "FTZ").count() <= 1
                && mods.iter().all(|m| matches!(*m, "RZ" | "RM" | "RP" | "FTZ")) =>
        {
            let rnd = sass_rnd(mods);
            let ftz = mods.contains(&"FTZ");
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
            // `-R16.F32`, `-UR4.F32`: the operand negated (fixtures/s_lift_k, probes/ffma2_ur =
            // GB300). ASSUME(L-ROSETTA)
            // `|R20|.F32x2.HI_LO`: the absolute value (probes/ffma2_ur = GB300).
            let mut neg = [false; 3];
            let mut abs = [false; 3];
            let mut f2n = |k: usize, s: &str| -> Result<(Val, F2Lanes)> {
                let s = match s.strip_prefix('-') {
                    Some(r) if r.starts_with('R') || r.starts_with("UR") || r.starts_with('|') => {
                        neg[k] = true;
                        r
                    }
                    _ => s,
                };
                let s = match s.strip_prefix('|').and_then(|r| r.split_once('|')) {
                    Some((r, rest)) => {
                        abs[k] = true;
                        format!("{r}{rest}")
                    }
                    None => s.to_string(),
                };
                f2(&s)
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
                    abs,
                    rnd,
                    ftz,
                }
            } else {
                let (d, a, b) = (lw.dst(lw.op(0)?)?, f2n(0, lw.op(1)?)?, f2n(1, lw.op(2)?)?);
                // FMUL2: a*b + (-0), the rounded product (an exact zero product keeps its
                // sign; under .RM the additive identity is +0 instead).
                let c = if base == "FMUL2" {
                    let z = if rnd == Rnd::Rm { 0 } else { 0x8000_0000 };
                    (Val::Imm(z), F2Lanes::Bcast)
                } else {
                    f2n(2, lw.op(3)?)?
                };
                Op::Ffma2 {
                    d,
                    a,
                    b,
                    c,
                    neg,
                    abs,
                    rnd,
                    ftz,
                }
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
        // `R2P PR, Ra[.B1|.B2|.B3], mask`: byte k of Ra to P0..P6 (FA4's mask bit tests;
        // GB300: fixtures/s_lift_l).
        "R2P" if mods.is_empty() && lw.op(0)? == "PR" => {
            let src = lw.op(1)?.trim_end_matches(".reuse");
            let (src, shift) = match src.rsplit_once(".B") {
                Some((r, k)) if matches!(k, "0" | "1" | "2" | "3") => {
                    (r, 8 * k.parse::<u8>().unwrap())
                }
                Some(_) => bail!(lw.err(format!("R2P operand selector not modeled: {src}"))),
                None => (src, 0),
            };
            let mask = hex(lw.op(2)?).ok_or_else(|| lw.err("bad mask"))?;
            if mask > 0x7f {
                bail!(lw.err(format!("R2P mask {mask:#x} beyond P6 not modeled")));
            }
            Op::R2p {
                a: lw.val(src, 4)?,
                mask,
                shift,
                uniform: false,
            }
        }
        // `UR2UP UPR, URa.Bk[, mask]`: restores what `UP2UR.Bk` saved (ptxas's fence.proxy.alias
        // sequence). No mask operand: UP0..UP6. ASSUME(L-ROSETTA)
        "UR2UP" if mods.is_empty() && lw.op(0)? == "UPR" => {
            let src = lw.op(1)?;
            let (src, shift) = match src.rsplit_once(".B") {
                Some((r, k)) if matches!(k, "0" | "1" | "2" | "3") => {
                    (r, 8 * k.parse::<u8>().unwrap())
                }
                Some(_) => bail!(lw.err(format!("UR2UP operand selector not modeled: {src}"))),
                None => (src, 0),
            };
            let mask = match l.operands.len() {
                2 => 0x7f,
                3 => hex(lw.op(2)?).ok_or_else(|| lw.err("bad mask"))? & 0x7f,
                _ => bail!(lw.err("UR2UP operands not modeled")),
            };
            Op::R2p {
                a: lw.val(src, 4)?,
                mask,
                shift,
                uniform: true,
            }
        }
        "R2UR" => {
            uniform = false; // collective: reads the value from the active lanes
            if l.operands.len() == 3 {
                Op::R2Ur {
                    p: Some(lw.pdst(lw.op(0)?)?),
                    d: lw.dst(lw.op(1)?)?,
                    a: lw.val(lw.op(2)?, 4)?,
                    or: has("OR"),
                }
            } else {
                if has("OR") {
                    bail!(lw.err("R2UR.OR without a predicate"));
                }
                Op::R2Ur {
                    p: None,
                    d: lw.dst(lw.op(0)?)?,
                    a: lw.val(lw.op(1)?, 4)?,
                    or: false,
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
                    mask: None,
                }
            } else {
                if lw.op(1)? != "UPT" && lw.op(1)? != "PT" {
                    bail!(lw.err("VOTE.ANY predicate output not modeled"));
                }
                Op::VoteAny {
                    d: lw.dst(lw.op(0)?)?,
                    a: lw.pred(lw.op(2)?)?,
                    mask: None,
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
                // `.S32` (redux.sync.add.s32): the same wrapping sum (probes/redux_s32,
                // with partial masks). The other `.S32` forms of the table are refused.
                ("REDUX", ["SUM"]) | ("REDUX", ["SUM", "S32"]) => ReduxOp::Sum,
                ("CREDUX", ["MIN"]) => ReduxOp::MinU,
                ("CREDUX", ["MAX"]) => ReduxOp::MaxU,
                ("CREDUX", ["MIN", "S32"]) => ReduxOp::MinS,
                ("CREDUX", ["MAX", "S32"]) => ReduxOp::MaxS,
                // `CREDUX.MIN|.MAX|.MINABS|.MAXABS.F32[.NAN]` (redux.sync f32 forms; GB300:
                // fixtures/s_lift_l).
                ("CREDUX", [m, "F32", rest @ ..])
                    if matches!(*m, "MIN" | "MAX" | "MINABS" | "MAXABS")
                        && (rest.is_empty() || rest == ["NAN"]) =>
                {
                    let (abs, nan) = (m.ends_with("ABS"), rest == ["NAN"]);
                    if m.starts_with("MIN") {
                        ReduxOp::MinF { abs, nan }
                    } else {
                        ReduxOp::MaxF { abs, nan }
                    }
                }
                _ => bail!(lw.err(format!("unsupported SASS opcode `{}`", l.opcode))),
            };
            Op::Redux {
                d: lw.dst(lw.op(0)?)?,
                a: lw.val(lw.op(1)?, 4)?,
                op,
                mask: None,
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
                    scope: sass_scope(mods).unwrap_or(Scope::Cta),
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
            // A 32-bit compare and new value in two unrelated registers (`ATOMG.E.CAS PT, R20,
            // [R4], R26, R20`: ptxas's CAS loop for a scalar bf16 atom.add, probes/bf16x2 =
            // GB300): compare Rb, new Rc. ASSUME(L-ROSETTA)
            let pair = new.file == cmp.file && new.idx == cmp.idx + 1;
            Op::Atom {
                op: AtomOp::Cas,
                space: if base == "ATOM" {
                    Space::Generic
                } else {
                    Space::Global
                },
                bytes,
                addr,
                v: Val::Reg(cmp, if wide || pair { 2 } else { 1 }),
                d,
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: sass_scope(mods).unwrap_or(Scope::Cta),
                },
                p,
                new: if wide {
                    Some(Val::Reg(new, 2))
                } else if pair {
                    None
                } else {
                    Some(Val::Reg(new, 1))
                },
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
                    scope: sass_scope(mods).unwrap_or(Scope::Cta),
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
            // Fail closed: the modifier set (besides .E, .STRONG and the scope) must be one of
            // the modeled forms (`atomg_op`).
            let Some((op, bytes)) = atomg_op(mods) else {
                bail!(lw.err(format!(
                    "ATOMG/REDG variant not modeled: .{}",
                    mods.join(".")
                )));
            };
            let sync = MemSync {
                sem: Sem::Relaxed,
                scope: sass_scope(mods).unwrap_or(Scope::Cta),
            };
            let wide_addr = |a: Addr| wide_e_addr(a, has("E"));
            let vector = matches!(op, AtomOp::FAddFtz | AtomOp::AddBf16x2) && bytes > 4;
            if base == "REDG" {
                Op::Atom {
                    op,
                    space: Space::Global,
                    bytes,
                    addr: wide_addr(lw.addr(lw.op(0)?)?),
                    // A vector red: the first of its consecutive data registers.
                    v: lw.val(lw.op(1)?, if vector { 4 } else { bytes })?,
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
                // A vector ATOMG (`.F32x2` / `.F32x4`): Rd and Rb are the first of their
                // consecutive registers, one FADD.FTZ atomic per element (probes/atom_vec).
                let d = match lw.op(1)? {
                    "RZ" => None,
                    x => Some(lw.dst(x)?),
                };
                Op::Atom {
                    op,
                    space: Space::Global,
                    bytes,
                    addr: wide_addr(lw.addr(lw.op(2)?)?),
                    v: lw.val(lw.op(3)?, if vector { 4 } else { bytes })?,
                    d,
                    sync,
                    p: None,
                    new: None,
                }
            }
        }
        // `UREDGR.E.ADD.NA[.S32 | .64].STRONG.GPU|SYS [URa], URb` (ptxas 13.0's red.async
        // .global, issued for one elected lane per BRA.U.ANY loop iteration after MEMBAR.ALL.CTA
        // and waited for by DEPBAR): a red of that scope that releases the issuing thread's
        // earlier accesses (probes/redasync_litmus: with UREDGR nothing was stale; MEMBAR.ALL.CTA
        // + REDG gave stale reads). Executed once per warp (uniform datapath). ASSUME(S-RED-ASYNC)
        "UREDGR" => {
            let core: Vec<&str> = mods
                .iter()
                .copied()
                .filter(|m| !matches!(*m, "E" | "NA" | "STRONG" | "GPU" | "SYS"))
                .collect();
            let bytes = match core.as_slice() {
                ["ADD"] | ["ADD", "S32"] | ["ADD", "U32"] => 4,
                ["ADD", "64"] => 8,
                _ => bail!(lw.err(format!("UREDGR variant not modeled: .{}", mods.join(".")))),
            };
            let scope = match (has("GPU"), has("SYS")) {
                (true, false) => Scope::Gpu,
                (false, true) => Scope::Sys,
                _ => bail!(lw.err("UREDGR without .GPU / .SYS not modeled")),
            };
            if !has("STRONG") || !has("E") {
                bail!(lw.err("UREDGR without .E.STRONG not modeled"));
            }
            Op::Atom {
                op: AtomOp::Add,
                space: Space::Global,
                bytes,
                addr: wide_e_addr(lw.addr(lw.op(0)?)?, true),
                v: lw.val(lw.op(1)?, bytes)?,
                d: None,
                sync: MemSync {
                    sem: Sem::Release,
                    scope,
                },
                p: None,
                new: None,
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
            Op::Break { b, cond: None }
        }
        // Invalidates the sync unit's view after mbarrier words were written with plain
        // stores (ptxas's divergent-path lowering of mbarrier.init). The simulator decodes
        // mbarriers from memory, so this has no further effect.
        "SYNCS" if has("CCTL") && has("IVALL") => Op::Nop, // scheduling / convergence hint; scoreboards are checked in scoreboard.rs
        "WARPSYNC"
            if mods.is_empty()
                || mods == ["ALL"]
                || mods == ["COLLECTIVE"]
                || mods == ["COLLECTIVE", "ALL"] =>
        {
            let all = mods == ["COLLECTIVE", "ALL"];
            if all && l.operands.len() != 1 {
                bail!(lw.err("WARPSYNC.COLLECTIVE.ALL form not modeled"));
            }
            if has("COLLECTIVE") {
                // `WARPSYNC.COLLECTIVE Rm, `(end)``: `end` must follow the region's ENDCOLLECTIVE.
                let t = lw.op(if all { 0 } else { 1 })?;
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
                all,
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
            // `STAS[.64|.128] [Ra.64], Rv`: Ra = shared::cluster address, Ra+1 = its mbarrier;
            // Rv.. = 1, 2 or 4 data registers.
            let bytes: u8 = match mods {
                [] => 4,
                ["64"] => 8,
                ["128"] => 16,
                _ => bail!(lw.err("STAS variant not modeled")),
            };
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
                v: lw.val(lw.op(1)?, bytes)?,
                bytes,
            }
        }
        "UTCATOMSWS" if has("AND") => Op::TmemFreeRaw {
            mask: lw.val(lw.op(1)?, 4)?,
        },
        "UVIRTCOUNT" if has("DEALLOC") && has("SMPOOL") => Op::TmemRelinquish,
        // UTCHMMA = kind::f16; UTCQMMA = kind::f8f6f4, or kind::mxf8f6f4 with a sixth operand
        // `tmem[URs]` (URs, URs+1 = the A, B scale-factor addresses); UTCOMMA = kind::mxf4 /
        // mxf4nvf4 (`.BLOCK16`: scale_vec::4X), always with scale factors. ptxas 13.1 spells
        // the same `kind::mxf4nvf4.block_scale.block16` UTCOMMA.4X on sm_100a and
        // UTCOMMA.BLOCK16 on sm_103a (probes/tc_pair_bs/sm100 = GB200).
        "UTCHMMA" | "UTCQMMA" | "UTCOMMA" => {
            let pair = mods.contains(&"2CTA");
            let block16 = mods.contains(&"BLOCK16") || mods.contains(&"4X");
            if mods.iter().any(|m| {
                !(*m == "2CTA" || matches!(*m, "BLOCK16" | "4X") && base == "UTCOMMA")
            }) || mods.contains(&"BLOCK16") && mods.contains(&"4X")
            {
                bail!(lw.err(format!("{base} variant not modeled yet")));
            }
            // `UTCHMMA ..., idesc[..], UPd, s`: a trailing immediate is scale-input-d (D = A B +
            // D * 2^-s; probes/mma_scale_d). Block-scaled kinds have 7 operands instead.
            let scale_d = match (base, lw.l.operands.get(6)) {
                ("UTCHMMA", Some(s)) if s.starts_with("0x") => {
                    let v = hex(s).filter(|&v| (1..16).contains(&v)).ok_or_else(|| {
                        lw.err(format!("UTCHMMA scale-input-d {s} not modeled"))
                    })?;
                    v as u8
                }
                _ => 0,
            };
            let scaled = lw.l.operands.len() == 7 && scale_d == 0;
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
            // `gdesc[URa].A_KEEP` / `.A_REUSE.A_KEEP` / `.A_REUSE` = PTX `.collector::a::fill`
            // / `use` / `lastuse` (ptxas 13.0; probes/tc_collector). ASSUME(S-TC-COLLECTOR)
            let a_full = lw.op(0)?;
            let (a_op, a_sfx) = match a_full.rfind(']') {
                Some(i) => a_full.split_at(i + 1),
                None => (a_full, ""),
            };
            let coll = match a_sfx {
                "" => MmaCollector::Discard,
                ".A_KEEP" => MmaCollector::Fill,
                ".A_REUSE.A_KEEP" => MmaCollector::Use,
                ".A_REUSE" => MmaCollector::LastUse,
                x => bail!(lw.err(format!("{base} A operand suffix {x} not modeled"))),
            };
            let a_tmem = a_op.starts_with("tmem[");
            let (adesc, aoff) = if a_tmem {
                lw.bracket(a_op, "tmem", 1)?
            } else {
                lw.bracket(a_op, "gdesc", 2)?
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
                scale_d,
                coll,
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
            // `LDTM.STAT.xN.MIN|MAX[ABS].F32[.NAN] Rred, Rd, tmem[..]` (tcgen05.ld.red, 32x32b
            // only): Rred = min / max of the N values loaded into Rd.. (FA4's row max).
            // Also `.16dp32bit_t0_t15` / `_t16_t31` (tcgen05.ld.red.16x32bx2, FA4 decode;
            // probes/tmem_ldred16).
            let stat_shape = match mods {
                ["STAT", s @ ("16dp32bit_t0_t15" | "16dp32bit_t16_t31"), ..] => Some(*s),
                _ => None,
            };
            let mods_owned: Vec<&str>;
            let mods = match stat_shape {
                Some(_) => {
                    mods_owned = [&mods[..1], &mods[2..]].concat();
                    &mods_owned[..]
                }
                None => mods,
            };
            let (red_op, mods, vop) = match mods {
                ["STAT", x, m, "F32", rest @ ..]
                    if x.starts_with('x')
                        && matches!(*m, "MIN" | "MAX" | "MINABS" | "MAXABS")
                        && (rest.is_empty() || rest == ["NAN"]) =>
                {
                    let (abs, nan) = (m.ends_with("ABS"), rest == ["NAN"]);
                    let op = if m.starts_with("MIN") {
                        ReduxOp::MinF { abs, nan }
                    } else {
                        ReduxOp::MaxF { abs, nan }
                    };
                    (Some(op), &mods[1..2], 1)
                }
                ["STAT", ..] => bail!(lw.err(format!("LDTM.STAT form not modeled: {}", l.opcode))),
                _ => (None, mods, 0),
            };
            let shaped: Vec<&str>;
            let mods = match stat_shape {
                Some(s) => {
                    shaped = [&[s][..], mods].concat();
                    &shaped[..]
                }
                None => mods,
            };
            let red = match red_op {
                // The reduced value discarded (RZ): a plain load.
                Some(_) if lw.op(0)? == "RZ" => None,
                Some(op) => Some((lw.dst(lw.op(0)?)?, op)),
                None => None,
            };
            let (n, s16x64b, half): (u8, bool, Option<u8>) = match mods {
                // Bare `LDTM R, tmem[..]` is tcgen05.ld.32x32b.x1 (probes/bf16_selp).
                [] => (1, false, None),
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
            if red_op.is_some() && lw.op(vop)? == "RZ" {
                bail!(lw.err("LDTM.STAT with the loaded values discarded (RZ) not modeled"));
            }
            let (taddr, off) = lw.bracket_neg(lw.op(vop + 1)?, "tmem", 1, true)?;
            Op::TmemLd {
                d: lw.dst(lw.op(vop)?)?,
                n,
                taddr,
                off,
                tracked: false,
                s16x64b,
                half,
                s16x256b,
                s16x128b,
                red,
                regs: None,
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
        // LDSM.8.MT1616[.2] Rd, [Ra]: ldmatrix.m16n16.x{1,2}.trans.b8 (FA4 fp8 decode), two
        // registers per matrix. GB300: probes/ldsm_b8. ASSUME(L-ROSETTA)
        "LDSM" if mods.first() == Some(&"8") && has("MT1616") => {
            let n: u8 = match mods {
                ["8", "MT1616"] => 2,
                ["8", "MT1616", "2"] => 4,
                _ => bail!(lw.err(format!("{} not modeled", l.opcode))),
            };
            Op::LdMatrix {
                d: lw.dst(lw.op(0)?)?,
                n,
                addr: lw.addr(lw.op(1)?)?,
                trans: true,
                b8: true,
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
                b8: false,
            }
        }
        // LDGSTS.E[.BYPASS][.LTC128B][.128|.64][.ZFILL] [Rd+off], desc[UR][Ra.64][, P]:
        // cp.async 16 / 8 / 4 bytes.
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
                src_size: None,
                // `.ZFILL`: a partial src-size s as source address + bytes - s
                // (probes/cpasync_srcsize = GB300).
                zfill: has("ZFILL"),
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
        // HMMA.1688.F32.BF16 (mma.sync m16n8k8 bf16): A two registers, B one (probes/bmma8).
        "HMMA" | "IMMA"
            if mods == ["16816", "F32", "BF16"]
                || mods == ["1688", "F32", "TF32"]
                || mods == ["1688", "F32", "BF16"]
                || mods == ["16816", "F32"]
                || base == "IMMA" && mods == ["16832", "S8", "S8", "SAT"] =>
        {
            uniform = false;
            let regs = |s: &str, n: u16| -> Result<Vec<Val>> {
                let s = s.trim_end_matches(".reuse");
                // IMMA's operand layouts (`R88.ROW`, `R80.COL`), as the instruction requires.
                let s = s
                    .trim_end_matches(".ROW")
                    .trim_end_matches(".COL")
                    .trim_end_matches(".reuse");
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
            let bf16_k8 = mods == ["1688", "F32", "BF16"];
            let a = regs(lw.op(1)?, if bf16_k8 { 2 } else { 4 })?;
            let b = regs(lw.op(2)?, if bf16_k8 { 1 } else { 2 })?;
            if a.is_empty() || b.is_empty() {
                bail!(lw.err("HMMA with a zero A/B operand not modeled"));
            }
            Op::Hmma {
                d: lw.dst(lw.op(0)?.trim_end_matches(".reuse"))?,
                a,
                b,
                c: regs(lw.op(3)?, 4)?,
                tf32: mods.get(2) == Some(&"TF32"),
                s8: base == "IMMA",
                f16: base == "HMMA" && mods == ["16816", "F32"],
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
                nc: false,
            }
        }
        // 32-byte accesses (`ld` / `st.global.v8.b32`, probes/ldst256). ENL2 / ELL2 / EFL2 is
        // the L2 eviction policy every 256-bit global access carries (normal, PTX
        // `.L2::evict_last`, `.L2::evict_first`; probes/ldst256_l2): no ordering effect. A .256
        // form without one, or with RZ data, fails closed.
        "LDG" | "STG" if has("256") && !(has("ENL2") || has("ELL2") || has("EFL2")) => {
            bail!(lw.err("LDG / STG.256 without an L2 eviction policy not modeled"))
        }
        // `LDGMC.E.<op>.<type>[.RN].STRONG.SYS Rd, [Ra.64+URb]` / `[RZ.U32+URb]`:
        // multimem.ld_reduce (S-MULTIMEM; probes/multimem/forms_sass.txt). `HPADD`: the
        // `.acc::f32` / `.acc::f16` add. ptxas emits `.STRONG.SYS` for every PTX semantics
        // (`.weak` included); acquire adds a CCTL.IVALL after it.
        "LDGMC" => {
            let op = match mods.get(1).copied() {
                Some("ADD") | Some("HPADD") => McOp::Add,
                Some("MIN") => McOp::Min,
                Some("MAX") => McOp::Max,
                Some("AND") => McOp::And,
                Some("OR") => McOp::Or,
                Some("XOR") => McOp::Xor,
                _ => bail!(lw.err("LDGMC operation not modeled")),
            };
            let (ty, bytes): (McTy, u8) = match mods.get(2).copied() {
                Some("32") => (McTy::U32, 4),
                Some("S32") => (McTy::S32, 4),
                Some("64") => (McTy::U64, 8),
                Some("S64") => (McTy::S64, 8),
                Some("F32") => (McTy::F32, 4),
                Some("F32x2") => (McTy::F32, 8),
                Some("F32x4") => (McTy::F32, 16),
                Some("F64") => (McTy::F64, 8),
                Some("F16x2") => (McTy::F16, 4),
                Some("F16x4") => (McTy::F16, 8),
                Some("F16x8") => (McTy::F16, 16),
                Some("BF16x2") => (McTy::Bf16, 4),
                Some("BF16x4") => (McTy::Bf16, 8),
                Some("BF16x8") => (McTy::Bf16, 16),
                Some("E4M3x4") => (McTy::E4m3, 4),
                Some("E4M3x8") => (McTy::E4m3, 8),
                Some("E4M3x16") => (McTy::E4m3, 16),
                Some("E5M2x4") => (McTy::E5m2, 4),
                Some("E5M2x8") => (McTy::E5m2, 8),
                Some("E5M2x16") => (McTy::E5m2, 16),
                _ => bail!(lw.err("LDGMC type not modeled")),
            };
            if mods.first() != Some(&"E")
                || !mods[3..]
                    .iter()
                    .all(|m| matches!(*m, "RN" | "STRONG" | "SYS"))
                || !has("STRONG")
                || !has("SYS")
            {
                bail!(lw.err("LDGMC modifiers not modeled"));
            }
            let a = lw.op(1)?;
            let inner = a
                .trim()
                .strip_prefix('[')
                .and_then(|x| x.strip_suffix(']'))
                .ok_or_else(|| lw.err("bad LDGMC address"))?;
            let mut addr = Addr {
                base: None,
                base2: None,
                off: 0,
            };
            for part in inner.split('+').map(str::trim) {
                if part == "RZ.U32" || part == "URZ" || part == "RZ" {
                    continue;
                }
                if let Some(h) = part.strip_prefix("0x") {
                    addr.off += i64::from_str_radix(h, 16).map_err(|_| lw.err("bad offset"))?;
                    continue;
                }
                if let Some(h) = part.strip_prefix("-0x") {
                    addr.off -= i64::from_str_radix(h, 16).map_err(|_| lw.err("bad offset"))?;
                    continue;
                }
                if part.ends_with(".U32") {
                    bail!(lw.err("LDGMC with a 32-bit register address not modeled"));
                }
                let r = lw.dst(part.trim_end_matches(".64"))?;
                let v = Val::Reg(r, 2);
                if addr.base.is_none() {
                    addr.base = Some(v);
                } else {
                    addr.base2 = Some(v);
                }
            }
            Op::LdReduce {
                d: lw.dst(lw.op(0)?)?,
                n: (bytes / 4).max(1),
                bytes,
                addr,
                op,
                ty,
                acc: mods[1] == "HPADD",
                sync: Some(MemSync {
                    sem: Sem::Relaxed,
                    scope: Scope::Sys,
                }),
            }
        }
        // `LDG.E.ENL2.256 Rhi, Rlo, [a], mask`: only the mask's words, each half's compacted
        // from its base register (RZ for a half with none); probes/ldg256_mask.
        "LDG" if has("256") => {
            let mask = match lw.l.operands.get(3) {
                None => 0xff,
                Some(m) => {
                    let m = m.trim();
                    let v = m
                        .strip_prefix("0x")
                        .and_then(|h| u8::from_str_radix(h, 16).ok())
                        .ok_or_else(|| lw.err("bad LDG.256 mask"))?;
                    if v == 0 || lw.l.operands.len() > 4 {
                        bail!(lw.err("LDG.256 mask form not modeled"));
                    }
                    v
                }
            };
            let (hi, lo) = (lw.dst(lw.op(0)?)?, lw.dst(lw.op(1)?)?);
            // A half's base is RZ exactly when the mask selects none of its words.
            for (r, m) in [(lo, mask & 0xf), (hi, mask >> 4)] {
                if (r.file == File::R && r.idx == RZ) != (m == 0) {
                    bail!(lw.err("LDG.256 destination / mask combination not modeled"));
                }
            }
            Op::Ld256 {
                hi,
                lo,
                addr: wide_e_addr(lw.addr(lw.op(2)?)?, has("E")),
                sync: sass_strong(mods),
                mask,
                nc: has("CONSTANT"),
            }
        }
        "STG" if has("256") => {
            if lw.op(1)? == "RZ" || lw.op(2)? == "RZ" {
                bail!(lw.err("STG.256 with RZ data not modeled"));
            }
            Op::St256 {
                lo: lw.dst(lw.op(1)?)?,
                hi: lw.dst(lw.op(2)?)?,
                addr: wide_e_addr(lw.addr(lw.op(0)?)?, has("E")),
                sync: sass_strong(mods),
            }
        }
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
                // `.CONSTANT`: ld.global.nc (S-NC).
                nc: has("CONSTANT"),
            }
        }
        "STG" if has("MMIO") && (!has("SYS") || has("STRONG")) => {
            bail!(lw.err("STG.MMIO other than .MMIO.SYS not modeled"))
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
        // `BAR.RED.{POPC,AND,OR}[.DEFER_BLOCKING] id[, count], {!}P` and its result
        // `B2R.RESULT Rd[, Pd]` (bar.red; probes/bar_red). ASSUME(E-BAR-RED)
        "BAR" if has("RED") => {
            let op = match (has("POPC"), has("AND"), has("OR")) {
                (true, false, false) => BarRedOp::Popc,
                (false, true, false) => BarRedOp::And,
                (false, false, true) => BarRedOp::Or,
                _ => bail!(lw.err("BAR.RED form not modeled")),
            };
            let n = l.operands.len();
            if !(2..=3).contains(&n) {
                bail!(lw.err("BAR.RED operands not modeled"));
            }
            let id = lw.val(lw.op(0)?, 4)?;
            let count = if n == 3 { Some(lw.val(lw.op(1)?, 4)?) } else { None };
            if !matches!(id, Val::Imm(_)) || count.as_ref().is_some_and(|c| !matches!(c, Val::Imm(_))) {
                bail!(lw.err("BAR.RED with register operands not modeled"));
            }
            Op::BarSync {
                id,
                count,
                arrive: false,
                packed: false,
                aligned: true,
                red: Some((op, lw.pred(lw.op(n - 1)?)?)),
            }
        }
        "B2R" if mods == ["RESULT"] => {
            let d = lw.op(0)?;
            Op::BarResult {
                d: if d == "RZ" { None } else { Some(lw.dst(d)?) },
                p: match l.operands.get(1) {
                    None => None,
                    Some(p) if p == "PT" => None,
                    Some(p) => Some(lw.pdst(p)?),
                },
            }
        }
        "BAR" if has("SYNC") || has("ARV") => {
            let arrive = has("ARV");
            let count = match l.operands.get(1) {
                Some(c) => Some(lw.val(c, 4)?),
                None if arrive => bail!(lw.err("BAR.ARV without a thread count")),
                None => None,
            };
            let id = lw.val(lw.op(0)?, 4)?;
            // `BAR.SYNC R4, R4`: ptxas's register form (`barrier.sync %r, %r`, and the
            // divergent path of a non-aligned barrier.sync): R4 = count << 16 | id & 0xf.
            let packed = match (&id, &count) {
                (Val::Reg(a, _), Some(Val::Reg(b, _))) if a == b => true,
                (Val::Reg(..), Some(Val::Reg(..))) => {
                    bail!(lw.err("BAR with different id and count registers not modeled"))
                }
                _ => false,
            };
            Op::BarSync {
                id,
                count,
                arrive,
                packed,
                // Except in a collective region (`aligned_bars`).
                aligned: true,
                red: None,
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
                    scope: Scope::Cluster,
                    release: false,
                },
                // `.RED`: arrive.expect_tx on another CTA's mbarrier (shared::cluster address).
                ["ARRIVE", "TRANS64", "RED"] => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: Some(lw.val(lw.op(2)?, 4)?),
                    remote: true,
                    count: None,
                    complete: false,
                    scope: Scope::Cluster,
                    release: false,
                },
                // `.RED.A1T0`: a plain arrival there.
                ["ARRIVE", "TRANS64", "RED", "A1T0"] if lw.op(2)? == "RZ" => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: None,
                    remote: true,
                    count: None,
                    complete: false,
                    scope: Scope::Cluster,
                    release: false,
                },
                // `A1T0`: one arrival, no transaction bytes (plain mbarrier.arrive).
                ["ARRIVE", "TRANS64", "A1T0"] if lw.op(2)? == "RZ" => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: None,
                    remote: false,
                    count: None,
                    complete: false,
                    scope: Scope::Cluster,
                    release: false,
                },
                // `.ART0` / `.RED.ART0`: the arrival count in the third operand, no transaction
                // bytes (`mbarrier.arrive [a], count` on a shared::cta / shared::cluster
                // address; probes/mbar_count).
                ["ARRIVE", "TRANS64", "ART0"] | ["ARRIVE", "TRANS64", "RED", "ART0"] => {
                    Op::MbarArrive {
                        addr: lw.addr(lw.op(1)?)?,
                        tx: None,
                        remote: mods.len() == 4,
                        count: Some(lw.val(lw.op(2)?, 4)?),
                        complete: false,
                        scope: Scope::Cluster,
                        release: false,
                    }
                }
                // `.A0TR` / `.A0TX`: no arrival; expect / complete the transaction bytes
                // (`mbarrier.expect_tx` / `mbarrier.complete_tx`, checked against ptxas).
                ["ARRIVE", "TRANS64", "RED", z @ ("A0TR" | "A0TX")] => Op::MbarArrive {
                    addr: lw.addr(lw.op(1)?)?,
                    tx: Some(lw.val(lw.op(2)?, 4)?),
                    remote: true,
                    count: Some(Val::Imm(0)),
                    complete: *z == "A0TX",
                    scope: Scope::Cluster,
                    release: false,
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
                    scope: Scope::Cluster,
                    release: false,
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
            // ASSUME(S-MBAR-SCOPE): no scope bits; ptxas emits PHASECHK for every scope.
            scope: Scope::Cluster,
            acquire: true,
        },
        // Proxy fences by state space (ptxas, sm_103a): `fence.proxy.async[.shared::cta|
        // .shared::cluster]` is MEMBAR.ALL.<CTA|GPU> + `FENCE.VIEW.ASYNC.S`,
        // `fence.proxy.async.global` is `FENCE.VIEW.ASYNC.G`, and `tcgen05.wait::st` is
        // `FENCE.VIEW.ASYNC.T`. Each orders only its own state space. ASSUME(S-PROXY)
        "FENCE" if mods == ["VIEW", "ASYNC", "S"] => Op::Fence {
            kind: FenceKind::ProxyAsync,
        },
        "FENCE" if mods == ["VIEW", "ASYNC", "G"] => Op::Fence {
            kind: FenceKind::ProxyAsyncGlobal,
        },
        "FENCE" if mods == ["VIEW", "ASYNC", "T"] => Op::TmemWaitSt,
        // try_cancel without multicast.
        "UGETNEXTWORKID" if mods == ["SELFCAST"] || mods == ["BROADCAST"] => Op::ClcTryCancel {
            resp: lw.ureg_at(lw.op(0)?, 0, 1)?,
            mbar: lw.ureg_at(lw.op(1)?, 0, 1)?,
            multicast: mods == ["BROADCAST"],
        },
        "UTMASTG" => {
            // `UTMASTG.<n>D.IM2COL [URa], [URb]` (cp.async.bulk.tensor...im2col_no_offs): the
            // coordinates get a trailing 0 for the (absent) im2col offsets (sim im2col.rs).
            let im2col = mods.len() == 2 && mods[1] == "IM2COL";
            if mods.len() != 1 && !im2col {
                bail!(lw.err("UTMASTG variant not modeled"));
            }
            let dims: u8 = mods[0]
                .trim_end_matches('D')
                .parse()
                .map_err(|_| lw.err("dims"))?;
            // URa: smem source, then N coordinates; URb: tensor map.
            let t = lw.op(0)?;
            let mut coords: Vec<Val> = (0..dims as u16)
                .map(|k| lw.ureg_at(t, 1 + k, 1))
                .collect::<Result<_>>()?;
            if im2col {
                coords.push(Val::Imm(0));
            }
            Op::TmaStore {
                dims,
                src: lw.ureg_at(t, 0, 1)?,
                off: 0,
                tmap: lw.ureg_at(lw.op(1)?, 0, 2)?,
                coords,
                red: false,
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
        // `CS2R Rd, SR_CLOCKLO` (clock64) / `SR_GLOBALTIMERLO` (%globaltimer: torch's
        // symmetric-memory barrier timeout): ASSUME(E-CLOCK) the clock reads 0 every time, so
        // timeout checks (`clock64() - start > N`) never fire.
        "CS2R"
            if l.operands.len() == 2 && matches!(lw.op(1)?, "SR_CLOCKLO" | "SR_GLOBALTIMERLO") =>
        {
            Op::Mov {
                d: lw.dst(lw.op(0)?)?,
                a: Val::SReg(SReg::Clock),
            }
        }
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
            kind: FenceKind::Release(sass_scope(mods).ok_or_else(|| {
                lw.err("MEMBAR without a scope not modeled (a default scope could be stronger than the hardware)")
            })?),
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
                src_smem: false,
            }
        }
        "UBLKCP" if mods == ["G", "S"] => Op::TmaStore {
            dims: 0,
            src: lw.ureg_at(lw.op(1)?, 0, 1)?,
            off: 0,
            tmap: lw.val(lw.op(2)?, 4)?,
            coords: vec![lw.ureg_at(lw.op(0)?, 0, 2)?],
            red: false,
        },
        "UTMALDG" => {
            // `.MULTICAST`: a third operand UR holds the ctaMask. `.2CTA`: the mbarrier may be
            // in the peer CTA (cp.async.bulk.tensor ... .cta_group::2).
            // `.2D.GATHER4` (cp.async.bulk.tensor.2d...tile::gather4): URa+2 is the column,
            // URa+3..URa+6 the four row coordinates (5 coords; exec.rs `complete_tma`).
            // `.MULTICAST.2CTA`: both (cuBLAS's 2-CTA nvjet kernels, M = N >= 512).
            // `.2D.GATHER4.MULTICAST[.2CTA]`: a multicast gather4 (trtllm-gen routed GEMM).
            // `.<n>D.IM2COL[.2CTA] [URa], [URb], URo[, desc[URp]]` (cp.async.bulk.tensor
            // ...im2col): URo holds the packed im2col offsets, appended to the coordinates (sim
            // im2col.rs); the desc[] cache-policy operand is a hint, dropped (L-PTX-HINT).
            let im2col = mods.get(1) == Some(&"IM2COL");
            let mods_i: Vec<&str> = mods.iter().copied().filter(|m| *m != "IM2COL").collect();
            let mods = &mods_i[..];
            if im2col && !(mods.len() == 1 || mods[1..] == ["2CTA"]) {
                bail!(lw.err(format!("UTMALDG form not modeled: {}", l.opcode)));
            }
            // `.2D.GATHER4` combines with `.MULTICAST` / `.2CTA` as a tile load does
            // (probes/tma_g4_mcast = GB300).
            if mods.is_empty() {
                bail!(lw.err(format!("UTMALDG form not modeled: {}", l.opcode)));
            }
            let gather4 = mods.len() >= 2 && mods[..2] == ["2D", "GATHER4"];
            let rest = &mods[if gather4 { 2 } else { 1 }..];
            let multicast = rest.first() == Some(&"MULTICAST");
            let pair = rest == ["2CTA"] || rest == ["MULTICAST", "2CTA"];
            let known = rest.is_empty()
                || rest == ["MULTICAST"]
                || rest == ["2CTA"]
                || rest == ["MULTICAST", "2CTA"];
            if !known || !mods[0].ends_with('D') || im2col && gather4 {
                bail!(lw.err(format!("UTMALDG form not modeled: {}", l.opcode)));
            }
            let dims: u8 = mods[0]
                .trim_end_matches('D')
                .parse()
                .map_err(|_| lw.err("dims"))?;
            let t = lw.op(0)?;
            let mut coords: Vec<Val> = (0..dims as u16 + if gather4 { 3 } else { 0 })
                .map(|k| lw.ureg_at(t, 2 + k, 1))
                .collect::<Result<_>>()?;
            if im2col {
                if dims < 3 || !(l.operands.len() == 3 || l.operands.len() == 4 && lw.op(3)?.starts_with("desc[")) {
                    bail!(lw.err(format!("UTMALDG.IM2COL operand form not modeled: {}", l.text)));
                }
                coords.push(lw.val(lw.op(2)?, 4)?);
            }
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
                src_smem: false,
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
        "ERRBAR" => Op::Nop,
        // ASSUME(S-REMOTE-INFLIGHT): waits for the thread's outstanding remote (DSMEM)
        // operations; ptxas emits it after MEMBAR.ALL.GPU for cluster-scope releases.
        "CGAERRBAR" => Op::Fence {
            kind: FenceKind::CgaErrBar,
        },
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
        "BRA" if (mods == ["DIV"] || mods == ["CONV"]) && l.operands.len() == 2 => {
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
                any: has("CONV"),
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
        // ASSUME(E-SETMAXNREG)
        "USETMAXREG" if has("DEALLOC") || (has("ALLOC") && !has("TRY_ALLOC")) => Op::Nop,
        "USETMAXREG" if has("TRY_ALLOC") => Op::Plop3 {
            d: lw.pdst(lw.op(0)?)?,
            a: Pred::TRUE,
            b: Pred::TRUE,
            c: Pred::TRUE,
            lut: 0xff,
            d2: None,
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
                mask: None,
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
            | Op::BMov { .. }
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

/// The operation and width of `ATOMG` / `REDG` with modifiers `mods`, None for a form not
/// modeled. Besides `.E`, `.STRONG` and the scope: `.ADD` / `.OR` / `.AND` / `.XOR` / `.EXCH`
/// [`.64`] (integer; `.ADD.S32` / `.U32` add alike: the sum wraps either way, probes/redux_s32),
/// `.MIN` / `.MAX` [`.S32` | `.U32` | `.64` | `.S64`], `.ADD.F32.FTZ.RN` (Triton's
/// tl.atomic_add on f32), `.ADD.BF16x2.RN` (per-lane bf16 add, probes/bf16x2, N-BF16X2),
/// `.ADD.F32x2` / `.F32x4.FTZ.RN` (per-element f32 adds), `.ADD.BF16x4.RN` (two BF16x2 adds)
/// and `.ADD.F64.RN`. Everything else in
/// the sm_103a table (`.F16x2`, `.F16x4` / `.x8`, `.CAS` handled elsewhere) is refused.
fn atomg_op(mods: &[&str]) -> Option<(AtomOp, u8)> {
    let mut core: Vec<&str> = mods
        .iter()
        .copied()
        .filter(|m| !matches!(*m, "E" | "STRONG" | "GPU" | "SYS" | "SM" | "CTA"))
        .collect();
    core.sort_unstable();
    let int = |op: AtomOp| -> Option<(AtomOp, u8)> {
        match core.len() {
            1 => Some((op, 4)),
            2 if core.contains(&"64") => Some((op, 8)),
            2 if core.contains(&"U32") => Some((op, 4)),
            _ => None,
        }
    };
    match core.as_slice() {
        ["ADD", "S32"] => Some((AtomOp::Add, 4)),
        ["ADD", "F32", "FTZ", "RN"] => Some((AtomOp::FAddFtz, 4)),
        // A vector atom / red (one FADD.FTZ atomic per element; probes/red_vec, probes/atom_vec).
        ["ADD", "F32x2", "FTZ", "RN"] => Some((AtomOp::FAddFtz, 8)),
        ["ADD", "F32x4", "FTZ", "RN"] => Some((AtomOp::FAddFtz, 16)),
        ["ADD", "F64", "RN"] => Some((AtomOp::FAdd64, 8)),
        ["ADD", "BF16x2", "RN"] => Some((AtomOp::AddBf16x2, 4)),
        // `.v4.bf16` (Triton's bf16 tl.atomic_add on 4 lanes): one BF16x2 atomic per 4-byte
        // half (probes/atom_bf16x4, N-BF16X2).
        ["ADD", "BF16x4", "RN"] => Some((AtomOp::AddBf16x2, 8)),
        [..] if core.contains(&"ADD") => int(AtomOp::Add),
        [..] if core.contains(&"OR") => int(AtomOp::Or),
        [..] if core.contains(&"AND") => int(AtomOp::And),
        [..] if core.contains(&"XOR") => int(AtomOp::Xor),
        [..] if core.contains(&"EXCH") => int(AtomOp::Exch),
        // `atom.inc.u32` / `atom.dec.u32` (a CuTe grid barrier's arrival counter): 32-bit only.
        ["INC"] | ["INC", "U32"] => Some((AtomOp::Inc, 4)),
        ["DEC"] | ["DEC", "U32"] => Some((AtomOp::Dec, 4)),
        [..] if core.contains(&"MIN") || core.contains(&"MAX") => {
            let width = core
                .iter()
                .filter(|m| !matches!(**m, "MIN" | "MAX"))
                .copied()
                .collect::<Vec<_>>();
            let bytes = match width.as_slice() {
                [] | ["U32"] | ["S32"] => 4,
                ["64"] | ["S64"] => 8,
                _ => return None,
            };
            (core.len() == width.len() + 1).then(|| (min_max_op(&core), bytes))
        }
        _ => None,
    }
}

/// The audited modifier tokens per base opcode (`modset.rs`), sorted by opcode.
#[doc(hidden)]
pub fn audited_mods() -> &'static [(&'static str, &'static [&'static str])] {
    modset::AUDITED_MODS
}

/// The first modifier of `base` outside `modset::AUDITED_MODS` (None: all audited).
fn unaudited_mod<'a>(base: &str, mods: &[&'a str]) -> Option<&'a str> {
    let ok: &[&str] = modset::AUDITED_MODS
        .binary_search_by(|(b, _)| (*b).cmp(base))
        .map(|i| modset::AUDITED_MODS[i].1)
        .unwrap_or(&[]);
    mods.iter().find(|m| !ok.contains(m)).copied()
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

/// The scope token of a strong access or atomic. ptxas always prints one on global and
/// generic atomics and `.STRONG` accesses (corpus audit); where it is absent (ATOM.E.CAST.SPIN,
/// the shared-memory leg of a generic atomic) callers take `Scope::Cta`, the weakest, so a
/// default never makes the model stronger than the hardware. ASSUME(S-SCOPE)
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
    // `[R.U32+UR]`: the uniform pair is the 64-bit part, the register stays 32-bit.
    let ur64 = [&a.base, &a.base2]
        .into_iter()
        .flatten()
        .any(|b| matches!(b, Val::Reg(r, 2) if r.file == File::UR));
    if e && !ur64 {
        if let Some(Val::Reg(r, 1)) = a.base {
            // `[RZ.U32+UR28]` (the zero base dropped): the uniform pair holds the address.
            if r.file == File::R || r.file == File::UR && a.base2.is_none() {
                a.base = Some(Val::Reg(r, 2));
            }
        }
    }
    a
}

/// `.STRONG.<scope>` on LDG / STG: a relaxed strong access (acquire / release are separate
/// CCTL.IVALL / MEMBAR instructions).
fn sass_strong(mods: &[&str]) -> Option<MemSync> {
    // `STG.E.64.MMIO.SYS`: st.mmio.relaxed.sys, a strong relaxed .sys store (probes/memhints;
    // the STG arm refuses MMIO without SYS).
    if mods.contains(&"MMIO") {
        return Some(MemSync {
            sem: Sem::Relaxed,
            scope: Scope::Sys,
        });
    }
    if !mods.contains(&"STRONG") {
        return None;
    }
    Some(MemSync {
        sem: Sem::Relaxed,
        scope: sass_scope(mods).unwrap_or(Scope::Cta),
    })
}

/// ASSUME(E-ALIGNED): every SASS `BAR` is executed by the whole (live) warp at once. ptxas
/// compiles PTX `bar.sync` / `barrier.sync.aligned` to `[WARPSYNC.ALL;] BAR.SYNC`, predicated
/// per branch when the PTX branches (it assumes convergence), and the non-aligned
/// `barrier.sync` to `BRA.DIV` around the same `BAR` with a `WARPSYNC.COLLECTIVE` slow path
/// (sm_103a, ptxas 13.0). A `BAR` inside a collective region (the slow path) is therefore
/// the non-aligned form; any other is aligned. GB300: two halves of a warp at two `bar.sync`
/// hang, at two `barrier.sync` complete (fixtures/e_aligned_bar AB1 / AB2, `ptxv probe`).
fn aligned_bars(code: &mut [Inst]) {
    let regions: Vec<(usize, usize)> = code
        .iter()
        .enumerate()
        .filter_map(|(i, x)| match x.op {
            Op::EndCollective { start } => Some((start, i)),
            _ => None,
        })
        .collect();
    for (s, e) in regions {
        for x in &mut code[s.min(e)..=e] {
            if let Op::BarSync { aligned, .. } = &mut x.op {
                *aligned = false;
            }
        }
    }
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
    let (mut lines, mut labels) = parse_disasm(&dis, kernel)?;
    fill_unprinted(&mut lines, &meta.text);
    recover_branch_labels(&lines, &mut labels);
    let addr_to_idx: HashMap<u32, usize> =
        lines.iter().enumerate().map(|(i, l)| (l.addr, i)).collect();
    let exits = exit_targets(&lines);
    Ok(unlowered_lines(
        &lines,
        &labels,
        &addr_to_idx,
        &meta,
        &exits,
    ))
}

fn unlowered_lines(
    lines: &[SassLine],
    labels: &HashMap<String, u32>,
    addr_to_idx: &HashMap<u32, usize>,
    meta: &CubinMeta,
    exits: &[(usize, Option<String>)],
) -> Vec<Unlowered> {
    let mut seen: std::collections::BTreeMap<String, (usize, String)> = Default::default();
    for l in lines {
        if let Err(e) = lower_line(l, labels, addr_to_idx, meta, exits) {
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
    let (mut lines, mut labels) = parse_disasm(dis, kernel)?;
    fill_unprinted(&mut lines, &meta.text);
    recover_branch_labels(&lines, &mut labels);
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
    let exits = exit_targets(&lines);
    // PTXV_LIFT_REPORT: list every instruction the model does not lower (a new binary).
    if std::env::var("PTXV_LIFT_REPORT").is_ok() {
        let seen = unlowered_lines(&lines, &labels, &addr_to_idx, meta, &exits);
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
        code.push(lower_line(l, &labels, &addr_to_idx, meta, &exits)?);
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
    if let Some((i, l)) = lines
        .iter()
        .enumerate()
        .find(|(i, l)| l.addr != (*i as u32) * 16)
    {
        // nvdisasm prints nothing for an encoding it does not know (a cubin from a newer
        // ptxas: CuTeDSL's bundled one): the slot is missing from the listing.
        bail!(
            "instruction addresses are not contiguous 16-byte slots: nothing printed for {:#06x}..{:#06x} (an encoding this nvdisasm does not know)",
            i * 16,
            l.addr
        );
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
    // ptxas reuses scoreboards: one kernel may count cp.async groups and bulk groups on the
    // same SBk in different places (mslk's FA4 decode: LDGDEPBAR's and UTMACMDFLUSH's are
    // both SB0). Which kind a `DEPBAR.LE SBk` waits for is decided per instruction, by the
    // kinds that may be outstanding on SBk when it is reached; both: not modeled.
    let reach = depbar_kinds(&lines, &code);
    for (i, l) in lines.iter().enumerate() {
        // A trailing `{list}` (scoreboards drained to zero) is not an async-group wait here:
        // groups counted on a listed scoreboard stay outstanding (conservative).
        if l.opcode != "DEPBAR.LE" || !(2..=3).contains(&l.operands.len()) {
            continue;
        }
        let sb = l.operands[0]
            .strip_prefix("SB")
            .and_then(|k| k.parse::<u8>().ok());
        let n = u32::from_str_radix(l.operands[1].trim_start_matches("0x"), 16).ok();
        if let (Some(sb), Some(n)) = (sb, n) {
            let kinds = reach[i].map_or(0, |r| r[sb as usize & 7]);
            match kinds {
                DEP_CP => code[i].op = Op::CpAsyncWait { n },
                DEP_BULK => {
                    code[i].op = Op::BulkWait {
                        n,
                        read: true,
                        sb: Some(sb),
                        cp: false,
                    }
                }
                0 if bulk_sbs.contains(&sb) => {
                    code[i].op = Op::BulkWait {
                        n,
                        read: true,
                        sb: Some(sb),
                        cp: false,
                    }
                }
                // Both kinds on some path (paths the analysis cannot tell apart, e.g. per
                // warp, or a kernel that commits both on one scoreboard): the scoreboard
                // counts both kinds, so at most n of the two together stay outstanding, which
                // implies at most n of each; the model completes only the latter (weaker, so
                // sound; exact for n = 0). ASSUME(E-CPASYNC)
                k if k == DEP_CP | DEP_BULK => {
                    code[i].op = Op::BulkWait {
                        n,
                        read: true,
                        sb: Some(sb),
                        cp: true,
                    }
                }
                0 if cp_sbs.contains(&sb) => code[i].op = Op::CpAsyncWait { n },
                0 => {}
                _ => bail!(
                    "sass {:#06x}: `{}` may wait for cp.async groups and bulk groups on SB{sb} (not modeled)",
                    l.addr,
                    l.text
                ),
            }
        }
    }
    // A wait bit on that scoreboard in any instruction's control word is the same wait with
    // n = 0 (cuBLAS's nvjet epilogue: the BAR.SYNC before reusing a TMA-store staging buffer),
    // for the groups committed on that scoreboard only (`bulk_commit_sb`): CUTLASS's grouped
    // GEMM flushes its stores on SB0 and waits on SB3 for its own MEMBAR / LDTM / UTMASTG
    // operand reads, while other flushes of the kernel use SB3 (soundness-harness miss 3).
    // ASSUME(E-ASYNC-STEP)
    let bulk_mask = bulk_sbs.iter().fold(0u8, |m, &s| m | 1 << s);
    let bulk_read_waits: Vec<(usize, u8)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            let w = l.hi.map(scoreboard::Ctrl::decode)?.wait & bulk_mask;
            (w != 0).then_some((i, w))
        })
        .collect();
    let bulk_commit_sb: Vec<(usize, u8)> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.opcode == "UTMACMDFLUSH")
        .filter_map(|(i, l)| Some((i, l.hi.and_then(|h| scoreboard::Ctrl::decode(h).rbar)?)))
        .collect();
    aligned_bars(&mut code);
    smem_acquires(&lines, &mut code);
    l2::check_kernel(&code)?;
    let static_findings = scoreboard::check(&lines, &code, meta.sm)?;
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
        bulk_commit_sb,
        parts: vec![],
        ntid: meta.ntid,
        globals: meta.globals.clone(),
        global_relocs: meta.global_relocs.clone(),
        // On GB300 the driver may rewrite an sm_100 / sm_100f module into an sm_103 image at
        // load: the same instructions, with its own control words and NOP padding
        // (probes/f64_arch's sm100f build: the same image for any input control bits;
        // probes/mma_bs's: the cubin as built). The lift's semantics carry over; hazard and
        // scoreboard verdicts do not. ASSUME(L-OPCLASS)
        static_image: (meta.sm == 100).then(|| {
            format!(
                "static sm_100 image ({}); control bits and timing not the executed ones (the GB300 driver may reschedule sm_100 modules at load: take hazard verdicts from the captured image)",
                if meta.target.is_empty() { "sm_100" } else { &meta.target }
            )
        }),
    })
}

const DEP_CP: u8 = 1;
const DEP_BULK: u8 = 2;

/// For each instruction, the kinds of asynchronous group (`DEP_CP`: LDGDEPBAR's write
/// scoreboard, `DEP_BULK`: UTMACMDFLUSH's read scoreboard) that may be outstanding on each
/// scoreboard when it is reached (None: unreachable). A `DEPBAR.LE SBk, 0` clears SBk.
fn depbar_kinds(lines: &[SassLine], code: &[Inst]) -> Vec<Option<[u8; 8]>> {
    let ret_sites: Vec<usize> = code
        .iter()
        .enumerate()
        .filter(|(_, i)| matches!(i.op, Op::Call { .. } | Op::CallAbs { .. }))
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
            Op::Call { target } => [vec![*target], next].concat(),
            Op::Ret { .. } => ret_sites.clone(),
            Op::Brx { targets, .. } => targets.clone(),
            _ => next,
        }
    };
    let mut input: Vec<Option<[u8; 8]>> = vec![None; lines.len()];
    if lines.is_empty() {
        return input;
    }
    input[0] = Some([0; 8]);
    let mut work = vec![0usize];
    while let Some(pc) = work.pop() {
        let mut st = input[pc].unwrap();
        let l = &lines[pc];
        let c = l.hi.map(scoreboard::Ctrl::decode);
        match l.opcode.as_str() {
            "LDGDEPBAR" => {
                if let Some(sb) = c.and_then(|c| c.wbar) {
                    st[sb as usize & 7] |= DEP_CP;
                }
            }
            "UTMACMDFLUSH" => {
                if let Some(sb) = c.and_then(|c| c.rbar) {
                    st[sb as usize & 7] |= DEP_BULK;
                }
            }
            "DEPBAR.LE"
                if l.guard.is_none() && l.operands.get(1).map(|s| s.as_str()) == Some("0x0") =>
            {
                if let Some(sb) = l.operands[0]
                    .strip_prefix("SB")
                    .and_then(|k| k.parse::<usize>().ok())
                {
                    st[sb & 7] = 0;
                }
            }
            _ => {}
        }
        for q in succ(pc) {
            if q >= input.len() {
                continue;
            }
            let new = match input[q] {
                None => st,
                Some(old) => std::array::from_fn(|k| old[k] | st[k]),
            };
            if input[q] != Some(new) {
                input[q] = Some(new);
                work.push(q);
            }
        }
    }
    input
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
/// `F2FP [.SATFINITE] [.RELU] .TF32.F32.PACK_B [.RZ]` -> (rz, sat, relu).
fn f2fp_tf32(mods: &[&str]) -> Option<(bool, bool, bool)> {
    let mut i = 0;
    let sat = mods.first() == Some(&"SATFINITE");
    i += sat as usize;
    let relu = mods.get(i) == Some(&"RELU");
    i += relu as usize;
    match mods.get(i..)? {
        ["TF32", "F32", "PACK_B"] => Some((false, sat, relu)),
        ["TF32", "F32", "PACK_B", "RZ"] => Some((true, sat, relu)),
        _ => None,
    }
}

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

/// Lower one disassembled instruction on its own (no labels; `enc`: the encoding's low word,
/// for immediate forms). For audits and tests of the lifter's accepted forms.
#[doc(hidden)]
pub fn lower_text(text: &str, enc: Option<u64>) -> Result<Op> {
    let body = text.trim().trim_end_matches(';').trim();
    let (guard, body) = match body.strip_prefix('@') {
        Some(r) => {
            let (g, r) = r.split_once(' ').unwrap_or((r, ""));
            (Some(g.to_string()), r.trim())
        }
        None => (None, body),
    };
    let (opcode, ops) = body.split_once(' ').unwrap_or((body, ""));
    let l = SassLine {
        addr: 0,
        guard,
        opcode: opcode.into(),
        operands: split_ops(ops),
        text: body.into(),
        hi: None,
        lo: enc,
    };
    let mut labels = HashMap::new();
    labels.insert(".L_x_1".to_string(), 0u32);
    let mut a2i = HashMap::new();
    a2i.insert(0u32, 0usize);
    lower_line(&l, &labels, &a2i, &CubinMeta::default(), &[]).map(|i| i.op)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unprinted known encoding in the last slot of the text is decoded too (the scan
    /// covers the whole text, not only up to the last printed line); zero padding is not.
    #[test]
    fn fill_unprinted_reaches_trailing_slots() {
        let (lo, hi) = (0x0000002c0002763eu64, 0x190fe40002083417u64);
        let mut text = vec![0u8; 48];
        text[16..24].copy_from_slice(&lo.to_le_bytes());
        text[24..32].copy_from_slice(&hi.to_le_bytes());
        let mut lines = vec![sass_line(0, "EXIT", None, None)];
        fill_unprinted(&mut lines, &text);
        let addrs: Vec<u32> = lines.iter().map(|l| l.addr).collect();
        assert_eq!(addrs, vec![0, 16]);
        assert!(lines[1].opcode.starts_with("F2FP.SATFINITE.BF16.S2_6"));
    }

    /// A `BRA` whose target nvdisasm 13.0 did not print (an S2_6 conversion) gets its label from
    /// the encoding; encodings from ssa2's MXINT8 G32 decode kernel (nvdisasm 13.3 labels).
    #[test]
    fn recover_branch_labels_from_encoding() {
        assert_eq!(bra_target(0x7830, 0x0000000000081947), Some(0x7860));
        assert_eq!(bra_target(0xd250, 0xffffffa400807947), Some(0x7860));
        assert_eq!(
            bra_target(0xd830, 0xffffffe000447947),
            Some(0xd830 + 16 - 7920)
        );
        assert_eq!(bra_target(0xda00, 0xfffffffc00fc7947), Some(0xda00));
        assert_eq!(bra_target(0x0, 0x0000000001000947), None);
        let (lo, hi) = (0x0000002c0002763eu64, 0x190fe40002083417u64);
        let mut text = vec![0u8; 64];
        text[32..40].copy_from_slice(&lo.to_le_bytes());
        text[40..48].copy_from_slice(&hi.to_le_bytes());
        let mut lines = vec![
            sass_line(0, "@P1 BRA `(.L_x_1)", Some(0x0000000000041947), None),
            sass_line(16, "BRA `(.L_x_0)", Some(0xfffffffc00fc7947), None),
            sass_line(48, "EXIT", None, None),
        ];
        let mut labels = HashMap::from([(".L_x_0".to_string(), 16u32)]);
        fill_unprinted(&mut lines, &text);
        recover_branch_labels(&lines, &mut labels);
        assert_eq!(labels.get(".L_x_1"), Some(&32));
        // A form the decoder does not know (nonzero bits [24, 34)) next to a known label
        // changes nothing.
        let mut lines2 = lines.clone();
        lines2.push(sass_line(
            64,
            "BRA.DIV UR4, `(.L_x_0)",
            Some(0x0000000001000947),
            None,
        ));
        let mut labels = HashMap::from([(".L_x_0".to_string(), 16u32)]);
        recover_branch_labels(&lines2, &mut labels);
        assert_eq!(labels.get(".L_x_1"), Some(&32));
        // A printed label the encoding contradicts: nothing is recovered (fail closed).
        let mut labels = HashMap::from([(".L_x_0".to_string(), 48u32)]);
        recover_branch_labels(&lines, &mut labels);
        assert_eq!(labels.get(".L_x_1"), None);
    }

    /// FFMA2 / FMUL2 / FADD2 .FTZ (with an optional rounding mode) set `ftz`; two rounding
    /// modes or an unknown modifier are refused.
    #[test]
    fn packed_f32_ftz_forms() {
        let f = |l: &str| match lower_text(l, None).unwrap() {
            Op::Ffma2 { rnd, ftz, neg, .. } => (rnd, ftz, neg),
            o => panic!("{l}: {o:?}"),
        };
        assert_eq!(
            f("FFMA2.FTZ R4, R4.F32x2.HI_LO, R6.F32x2.HI_LO, R8.F32x2.HI_LO"),
            (Rnd::Rn, true, [false; 3])
        );
        assert_eq!(
            f("FADD2.FTZ.RM R4, R4.F32x2.HI_LO, R6.F32x2.HI_LO"),
            (Rnd::Rm, true, [false; 3])
        );
        assert_eq!(
            f("FADD2.FTZ R4, R4.F32x2.HI_LO, -R6.F32x2.HI_LO"),
            (Rnd::Rn, true, [false, false, true])
        );
        assert_eq!(
            f("FMUL2.FTZ.RZ R4, R4.F32x2.HI_LO, R6.F32x2.HI_LO"),
            (Rnd::Rz, true, [false; 3])
        );
        assert_eq!(
            f("FFMA2 R4, R4.F32x2.HI_LO, R6.F32x2.HI_LO, R8.F32x2.HI_LO"),
            (Rnd::Rn, false, [false; 3])
        );
        for bad in [
            "FFMA2.FTZ.RZ.RM R4, R4.F32x2.HI_LO, R6.F32x2.HI_LO, R8.F32x2.HI_LO",
            "FFMA2.SAT R4, R4.F32x2.HI_LO, R6.F32x2.HI_LO, R8.F32x2.HI_LO",
            "FADD2.FTZ.FTZ R4, R4.F32x2.HI_LO, R6.F32x2.HI_LO",
        ] {
            assert!(lower_text(bad, None).is_err(), "{bad}");
        }
    }

    /// UTCOMMA.4X (sm_100a) = UTCOMMA.BLOCK16 (sm_103a): kind::mxf4nvf4 scale_vec::4X.
    #[test]
    fn utcomma_4x_is_block16() {
        let kind = |l: &str| match lower_text(l, None).unwrap() {
            Op::Mma { kind, pair, .. } => (kind, pair),
            o => panic!("{l}: {o:?}"),
        };
        let x4 = MmaKind::Mxf4nvf4 { x4: true };
        assert_eq!(
            kind(
                "UTCOMMA.2CTA.4X gdesc[UR10], gdesc[UR18], tmem[UR40], tmem[UR42], idesc[UR43], tmem[UR44], UPT"
            ),
            (x4, true)
        );
        assert_eq!(
            kind(
                "UTCOMMA.2CTA.BLOCK16 gdesc[UR10], gdesc[UR18], tmem[UR40], tmem[UR42], idesc[UR43], tmem[UR44], UPT"
            ),
            (x4, true)
        );
        assert_eq!(
            kind("UTCOMMA gdesc[UR10], gdesc[UR18], tmem[UR40], tmem[UR42], idesc[UR43], tmem[UR44], UPT").0,
            MmaKind::Mxf4nvf4 { x4: false }
        );
        for bad in [
            "UTCOMMA.2CTA.4X.BLOCK16 gdesc[UR10], gdesc[UR18], tmem[UR40], tmem[UR42], idesc[UR43], tmem[UR44], UPT",
            "UTCQMMA.4X gdesc[UR10], gdesc[UR18], tmem[UR40], tmem[UR42], idesc[UR43], tmem[UR44], UPT",
        ] {
            assert!(lower_text(bad, None).is_err(), "{bad}");
        }
    }

    /// The forms lifted for the mslk / msl_ops gaps, and the neighbouring ones the probes do
    /// not cover, which must be refused (fail closed).
    #[test]
    fn lift_l_forms() {
        let ok = [
            "HMUL2.BF16_V2 R11, R11, R6.H0_H0",
            "HADD2.BF16_V2 R8, R13, R12.H0_H0",
            "HADD2.BF16_V2 R13, R4.reuse, -R5.reuse",
            "HFMA2.BF16_V2 R17, R20, R24, -RZ",
            "HFMA2.BF16_V2 R12, -RZ.H0_H0, RZ.H0_H0, -R11",
            "HFMA2.BF16_V2.RELU R18, R4.reuse, R5, R6.reuse",
            "FMNMX.XORSIGN R24, |R4|.reuse, |R5|.reuse, PT",
            "FMNMX.FTZ.NAN.XORSIGN R9, |R0|, |R9|, !PT",
            "FMNMX.XORSIGN R19, |R0|.reuse, |UR6|, PT",
            "REDUX.SUM.S32 UR5, R10",
            "REDG.E.ADD.BF16x2.RN.STRONG.GPU desc[UR6][R4.64], R11",
            "ATOMG.E.ADD.BF16x2.RN.STRONG.GPU PT, R16, desc[UR4][R2.64+0x5000], R4",
            "ATOMG.E.ADD.S32.STRONG.GPU PT, R27, desc[UR12][R26.64], R29",
            "REDG.E.ADD.S32.STRONG.GPU desc[UR4][R2.64+0x200], R9",
            "REDG.E.ADD.F64.RN.STRONG.GPU desc[UR6][R4.64], R10",
            "REDG.E.ADD.F32x2.FTZ.RN.STRONG.GPU desc[UR6][R4.64], R10",
            "REDG.E.ADD.F32x4.FTZ.RN.STRONG.GPU desc[UR6][R4.64], R8",
            "FHFMA.BF16.RZ R19, R4.reuse.H1, R5.reuse.H1, R6.reuse",
            "FHFMA.BF16.SAT R25, R4.reuse, R5, R7.reuse",
            "FHADD.BF16.RZ R24, R4.H1, -R7",
            "FHADD R8, R3, -R6.reuse",
            "FHADD.RZ.SAT R15, R2, -R3",
            "FHFMA R14, R2.reuse, R2.H1, R3.reuse",
            "HMNMX2.BF16_V2 R18, R4.reuse.H0_H0, R5.reuse.H1_H1, PT",
            "HMNMX2.BF16_V2.NAN.XORSIGN R16, |R4|, |R5|, PT",
            "HMNMX2.XORSIGN R0, |R2|.reuse.H0_H0, |R3|.reuse.H1_H1, !PT",
            "HMNMX2.NAN.XORSIGN R10, |R2|.reuse, |R3|.reuse, PT",
            "HMNMX2 R8, R8, |R7|, !PT",
            "HMNMX2.NAN R6, R2.reuse, R3.reuse, PT",
            "HMNMX2 R18, R2.H1_H1, R3.H0_H0, PT",
            "STSM.8.MT168.4 [R17+0x25400], R12",
            "STSM.8.MT168 [R3], R16",
            "QMUL4.E4M3x4.E2M1x4.E4M3x4 R8, R2.reuse.N3210, R11.reuse.B0",
            "QMUL4.E4M3x4.E2M1x4.E4M3x4 R9, R2.N7654, R11.B0",
            "VIADDMNMX.RELU R135, R0, -R135, 0x80, PT",
            "VIADDMNMX.RELU R9, R11.reuse, -R9, R0, PT",
            "FFMA2.RM R10, R6.reuse.F32x2.HI_LO, R4.reuse.F32x2.HI_LO, R6.reuse.F32x2.LO_HI",
            "FADD2.RP R12, R6.F32x2.HI_LO, -R4.F32x2.HI_LO",
            "IMAD.WIDE R28, R21, 0x2, -R26",
            "IMAD.WIDE.U32 R10, R13, 0x6, -R2",
            "UP2UR UR37, UPR, URZ, 0x1",
            "ATOMG.E.CAS.STRONG.GPU PT, R20, [R4], R26, R20",
            "HFMA2.BF16_V2 R7, -RZ.H0_H0, RZ.H0_H0, |R2|",
            "HFMA2.BF16_V2 R0, -RZ.H0_H0, RZ.H0_H0, |R2|.reuse.H1_H1",
            "HFMA2 R4, R7.reuse.H1_H1, R0.reuse.H0_H0, R15.reuse.H0_H0",
            "MATCH.ANY.U64 R5, R10",
            "VHMNMX R4, |R4|.H0_H0, |R4|.H1_H1, |R5|.H0_H0, !PT",
            "HSETP2.BF16_V2.NUM.AND P4, P6, R48, RZ.H0_H0, PT",
            "UBLKPF.L2 [UR8], UR6",
            "VHMNMX.BF16_V2 R0, R0, RZ.H0_H0, R12, !PT",
            "R2UR.OR P0, UR9, R13",
            "STG.E.64.MMIO.SYS [R28+0x50], R4",
            "STG.E.NA.128 desc[UR4][R4.64], R8",
            "LDG.E.NA.U8.STRONG.SYS R26, desc[UR4][R12.64]",
            "ST.E.EF.128 desc[UR4][R28.64+0x40], R16",
            "ATOM.E.MAX.64.STRONG.SM P0, R4, desc[UR4][R28.64], R6",
            "ATOMG.E.ADD.F32x2.FTZ.RN.STRONG.GPU PT, R2, desc[UR6][R4.64], R10",
        ];
        for l in ok {
            lower1(l).unwrap_or_else(|e| panic!("{l}: {e}"));
        }
        let refused = [
            "STG.E.64.MMIO R1, R2",
            "STG.E.64.MMIO.STRONG.SYS [R2], R4",
            "R2UR.OR UR9, R13",
            "HADD2.BF16_V2.FTZ R1, R2, R3",
            "HADD2.BF16_V2.SAT R1, R2, R3",
            "HFMA2.BF16_V2.FMZ R1, R2, R3, R4",
            "HFMA2.BF16_V2.OOB R1, R2, R3, R4",
            "HMUL2.BF16_V2 R1, |R2, R3",
            "HMUL2.BF16_V2 R1, R2.H1_H0, R3",
            "FMNMX.XORSIGN R1, R2, |R3|, PT",
            "FMNMX.XORSIGN R1, -|R2|, |R3|, PT",
            "FMNMX3.XORSIGN R1, |R2|, |R3|, |R4|, PT",
            "REDUX.S32 UR5, R10",
            "REDUX.OR.S32 UR5, R10",
            "REDUX.MIN UR5, R10",
            "REDG.E.ADD.F16x2.RN.STRONG.GPU desc[UR6][R4.64], R11",
            "REDG.E.INC.STRONG.GPU desc[UR6][R4.64], R10",
            "ATOMG.E.ADD.F16x4.GPU.RN.STRONG PT, R1, desc[UR6][R4.64], R10",
            "FHFMA.BF16 R1, -R2, R3, R4",
            "FHFMA.BF16.RZ.RM R1, R2, R3, R4",
            "FHADD.RZ.RP R1, R2, R3",
            "FHADD.SAT.SAT R1, R2, R3",
            "FHFMA.BF16.BF16 R1, R2, R3, R4",
            "HMNMX2.BF16_V2 R1, |R2|, |R3|, PT",
            "HMNMX2.BF16_V2.XORSIGN R1, R2, R3, PT",
            "HMNMX2.XORSIGN R1, R2, |R3|, PT",
            "HMNMX2 R1, -R2, R3, PT",
            "HMNMX2.FTZ R1, R2, R3, PT",
            "HMNMX2 R1, R2.H1_H0, R3, PT",
            "STSM.8.MT168.8 [R3], R16",
            "STSM.8.MT168.4.T [R3], R16",
            "QMUL4.E4M3x4.E2M1x4.E4M3x4 R8, R2.N3210, R11.B1",
            "QMUL4.E4M3x4.E2M1x4.E4M3x4 R8, R2.N5432, R11.B0",
            "QMUL4.E4M3x4.E2M1x4.E2M1x4 R8, R2.N3210, R11.B0",
            "VIADDMNMX.RELU R1, R0, ~R2, 0x80, PT",
            "VIADDMNMX.RELU R1, -R0, R2, 0x80, PT",
            "FFMA2.RZ.RM R10, R6.F32x2.HI_LO, R4.F32x2.HI_LO, R6.F32x2.HI_LO",
            "IMAD.WIDE R28, R21, 0x2, ~R26",
            "IMAD.WIDE.U32.X R10, R13, 0x6, -R2, P0",
            "UP2UR UR37, UPR, URZ, 0x80",
        ];
        for l in refused {
            assert!(lower1(l).is_err(), "{l} must be refused");
        }
    }

    /// Masked LDG.256: accepted when each half's base is RZ exactly when its mask is empty.
    #[test]
    fn ldg256_mask_forms() {
        for l in [
            "LDG.E.ENL2.256 R2, RZ, desc[UR4][R2.64], 0xc0",
            "LDG.E.ENL2.256 RZ, R8, desc[UR4][R2.64], 0x3",
            "LDG.E.ENL2.256 R4, R2, desc[UR4][R2.64], 0x3c",
        ] {
            assert!(lower1(l).is_ok(), "{l}");
        }
        for l in [
            "LDG.E.ENL2.256 R2, R4, desc[UR4][R2.64], 0xc0",
            "LDG.E.ENL2.256 RZ, R4, desc[UR4][R2.64], 0x30",
            "LDG.E.ENL2.256 R2, R4, desc[UR4][R2.64], 0x0",
        ] {
            assert!(lower1(l).is_err(), "{l} must be refused");
        }
    }

    #[test]
    fn viaddmnmx_negated_uniform() {
        // q_split_seqlens_kernel: `VIADDMNMX R13, R3, -UR8, RZ, !PT` = max(R3 - UR8, 0).
        for (l, neg) in [
            ("VIADDMNMX R13, R3, -UR8, RZ, !PT", true),
            ("VIADDMNMX R13, R3, -R8, RZ, !PT", true),
            ("VIADDMNMX R13, R3, UR8, RZ, !PT", false),
        ] {
            match lower1(l).unwrap() {
                Op::AddMinMax { neg_b, .. } => assert_eq!(neg_b, neg, "{l}"),
                o => panic!("{l}: {o:?}"),
            }
        }
        assert!(lower1("VIADDMNMX R13, R3, -c[0x0][0x10], RZ, !PT").is_err());
    }

    #[test]
    fn half_imm() {
        assert_eq!(parse_half("5.9604644775390625e-08"), Some(1));
        assert_eq!(parse_half("1.875"), Some(0x3f80));
        assert_eq!(parse_half("0"), Some(0));
    }

    /// Lower one disassembled instruction (no labels, no encoding).
    /// UTMAREDG with an L2 cache-policy descriptor (`.L2::cache_hint`, the cascade WGRAD's
    /// evict-first fp32 reduce-add) lowers like the hint-free form: a reducing TmaStore with the
    /// same source, map and coordinates. Other third operands stay refused.
    #[test]
    fn utmaredg_cache_policy_desc() {
        let plain = lower1("UTMAREDG.3D.ADD [UR20], [UR24]").unwrap();
        for l in [
            "UTMAREDG.3D.ADD [UR20], [UR24], desc[URZ]",
            "UTMAREDG.3D.ADD [UR20], [UR24], desc[UR26]",
        ] {
            let op = lower1(l).unwrap_or_else(|e| panic!("{l}: {e}"));
            assert_eq!(format!("{op:?}"), format!("{plain:?}"), "{l}");
            assert!(
                matches!(
                    op,
                    Op::TmaStore {
                        red: true,
                        dims: 3,
                        ..
                    }
                ),
                "{l}"
            );
        }
        for l in [
            "UTMAREDG.3D.ADD [UR20], [UR24], UR26",
            "UTMAREDG.3D.ADD [UR20], [UR24], desc[UR26], UR4",
        ] {
            assert!(lower1(l).is_err(), "{l} must be refused");
        }
    }

    /// ASSUME(E-CLOCK): `CS2R Rd, SR_CLOCKLO` / `SR_GLOBALTIMERLO` move the 64-bit clock
    /// special register (0 unless PTXV_CLOCK_STEP is set) into the register pair.
    #[test]
    fn cs2r_clock_lowers_to_clock_sreg() {
        for l in ["CS2R R4, SR_CLOCKLO", "CS2R R6, SR_GLOBALTIMERLO"] {
            match lower1(l).unwrap_or_else(|e| panic!("{l}: {e}")) {
                Op::Mov {
                    a: Val::SReg(SReg::Clock),
                    ..
                } => {}
                o => panic!("{l}: {o:?}"),
            }
        }
    }

    fn lower1(text: &str) -> Result<Op> {
        let body = text.trim();
        let (guard, body) = match body.strip_prefix('@') {
            Some(r) => {
                let (g, r) = r.split_once(' ').unwrap();
                (Some(g.to_string()), r.trim())
            }
            None => (None, body),
        };
        let (opcode, ops) = body.split_once(' ').unwrap_or((body, ""));
        let l = SassLine {
            addr: 0,
            guard,
            opcode: opcode.into(),
            operands: split_ops(ops),
            text: body.into(),
            hi: None,
            lo: None,
        };
        let mut labels = HashMap::new();
        labels.insert(".L_x_1".to_string(), 0u32);
        let mut a2i = HashMap::new();
        a2i.insert(0u32, 0usize);
        lower_line(&l, &labels, &a2i, &CubinMeta::default(), &[]).map(|i| i.op)
    }

    fn r(idx: u16) -> Reg {
        Reg { file: File::R, idx }
    }

    /// `DEPBAR.LE SB0` after a cp.async commit and after a bulk commit on the same scoreboard
    /// (control words from mslk's FA4 decode): each gets its own kind.
    #[test]
    fn depbar_kinds_per_instruction() {
        let mk = |addr: u32, text: &str, hi: u64| {
            let (opcode, ops) = text.split_once(' ').unwrap_or((text, ""));
            SassLine {
                addr,
                guard: None,
                opcode: opcode.into(),
                operands: split_ops(ops),
                text: text.into(),
                hi: Some(hi),
                lo: None,
            }
        };
        let lines = vec![
            mk(0x00, "LDGDEPBAR", 0x000e240000000000),
            mk(0x10, "DEPBAR.LE SB0, 0x0", 0x000fc80000000000),
            mk(0x20, "UTMACMDFLUSH", 0x0001e20000000000),
            mk(0x30, "DEPBAR.LE SB0, 0x0", 0x000fc80000000000),
            mk(0x40, "EXIT", 0x000fea0003800000),
        ];
        let code: Vec<Inst> = lines
            .iter()
            .map(|l| {
                lower_line(
                    l,
                    &HashMap::new(),
                    &HashMap::new(),
                    &CubinMeta::default(),
                    &[],
                )
                .unwrap()
            })
            .collect();
        let k = depbar_kinds(&lines, &code);
        assert_eq!(k[1].unwrap()[0], DEP_CP);
        assert_eq!(k[3].unwrap()[0], DEP_BULK);
    }

    #[test]
    fn shf_forms() {
        match lower1("SHF.R.U32.HI R36, RZ, R36, R132.reuse").unwrap() {
            Op::Shf {
                right: true,
                signed: false,
                high: true,
                wide: false,
                wrap: false,
                ..
            } => {}
            o => panic!("{o:?}"),
        }
        assert!(matches!(
            lower1("SHF.L.U64.HI R29, R10, R8, R11").unwrap(),
            Op::Shf {
                right: false,
                high: true,
                wide: true,
                wrap: false,
                ..
            }
        ));
        assert!(matches!(
            lower1("SHF.R.W.U32 R21, R10, R8, R11").unwrap(),
            Op::Shf {
                wrap: true,
                wide: false,
                ..
            }
        ));
        assert!(lower1("SHF.R.W.U64 R21, R10, R8, R11").is_err());
        assert!(lower1("SHF.R.U32.XYZ R21, R10, R8, R11").is_err());
        assert!(lower1("SHF.U32 R21, R10, R8, R11").is_err());
        assert!(matches!(
            lower1("F2I.S64.F64.TRUNC R28, R4").unwrap(),
            Op::F2I {
                f64: true,
                signed: true,
                bits: 64,
                rnd: Rnd::Rz,
                ..
            }
        ));
        assert!(matches!(
            lower1("F2I.F64.FLOOR R2, R4").unwrap(),
            Op::F2I {
                f64: true,
                signed: true,
                bits: 32,
                rnd: Rnd::Rm,
                ..
            }
        ));
        assert!(lower1("F2I.FTZ.S64.F64.TRUNC R28, R4").is_err());
        assert!(lower1("F2I.S64.F64.TRUNC.NTZ R28, R4").is_err());
        assert!(lower1("F2I.S64.U32.F64.TRUNC R28, R4").is_err());
    }

    /// `LEA.HI.X Rd, Ra, ~Rb, RZ, s, P` (probes/lea_hi_not): exactly that shape; the operand
    /// kinds the probe does not cover stay refused.
    #[test]
    fn lea_hi_x_complemented() {
        for (l, s) in [
            ("LEA.HI.X R54, R52, ~R55, RZ, 0x7, P0", 7),
            ("LEA.HI.X R11, R0.reuse, ~R3.reuse, RZ, 0x1, P1", 1),
            ("LEA.HI.X R25, R0, ~R3, RZ, 0x1e, P6", 30),
        ] {
            match lower1(l).unwrap_or_else(|e| panic!("{l}: {e}")) {
                Op::LeaHi {
                    a: Val::Reg(_, 1),
                    b: Val::Reg(_, 1),
                    c: Val::Imm(0),
                    shift,
                    sx32: false,
                    neg_b: false,
                    not_b: true,
                    cin: Some(Pred { neg: false, .. }),
                    cout: None,
                    ..
                } => assert_eq!(shift, s, "{l}"),
                o => panic!("{l}: {o:?}"),
            }
        }
        for l in [
            "LEA.HI.X R54, R52, ~R55, R53, 0x7, P0",
            "LEA.HI.X R54, R52, ~R55, RZ, 0x0, P0",
            "LEA.HI.X R54, R52, ~R55, RZ, 0x1f, P0",
            "LEA.HI.X R54, R52, ~R55, RZ, 0x7, !P0",
            "LEA.HI.X R54, R52, ~R55, RZ, 0x7, PT",
            "LEA.HI.X R54, R52, ~R55, RZ, 0x7, UP0",
            "LEA.HI.X R54, R52, ~R55, RZ, 0x7",
            "LEA.HI.X R54, R52, ~UR5, RZ, 0x7, P0",
            "LEA.HI.X R54, R52, ~RZ, RZ, 0x7, P0",
            "LEA.HI.X R54, R52, ~0x5, RZ, 0x7, P0",
            "LEA.HI.X R54, UR4, ~R55, RZ, 0x7, P0",
            "LEA.HI.X R54, RZ, ~R55, RZ, 0x7, P0",
            "LEA.HI.X R54, 0x4, ~R55, RZ, 0x7, P0",
            "LEA.HI.X R54, ~R52, R55, RZ, 0x7, P0",
            "LEA.HI.X R54, ~R52, ~R55, RZ, 0x7, P0",
            "LEA.HI.X R54, P1, R52, ~R55, RZ, 0x7, P0",
            "LEA.HI R54, R52, ~R55, RZ, 0x7",
            "ULEA.HI.X UR4, UR5, ~UR6, URZ, 0x7, UP0",
        ] {
            assert!(lower1(l).is_err(), "{l} must be refused");
        }
    }

    /// `F2I ... Rd, -Ra` (probes/f2i_neg): a negated data register on the 32-bit `.NTZ` forms
    /// only; every other modified source stays refused.
    #[test]
    fn f2i_negated_source() {
        for (l, signed, rnd, ftz) in [
            ("F2I.TRUNC.NTZ R0, -R0", true, Rnd::Rz, false),
            ("F2I.U32.TRUNC.NTZ R38, -R38", false, Rnd::Rz, false),
            ("F2I.NTZ R9, -R4", true, Rnd::Rn, false),
            ("F2I.U32.FLOOR.NTZ R21, -R4.reuse", false, Rnd::Rm, false),
            ("F2I.FTZ.CEIL.NTZ R29, -R4", true, Rnd::Rp, true),
            ("F2I.FTZ.U32.NTZ R0, -R4", false, Rnd::Rn, true),
        ] {
            match lower1(l).unwrap_or_else(|e| panic!("{l}: {e}")) {
                Op::F2I {
                    a: Val::Reg(_, 1),
                    neg: true,
                    bits: 32,
                    f64: false,
                    signed: s,
                    rnd: r,
                    ftz: f,
                    ..
                } => assert_eq!((s, r, f), (signed, rnd, ftz), "{l}"),
                o => panic!("{l}: {o:?}"),
            }
        }
        assert!(matches!(
            lower1("F2I.TRUNC.NTZ R0, R0").unwrap(),
            Op::F2I { neg: false, .. }
        ));
        for l in [
            "F2I.TRUNC.NTZ R0, |R0|",
            "F2I.TRUNC.NTZ R0, -|R0|",
            "F2I.TRUNC.NTZ R0, ~R0",
            "F2I.TRUNC.NTZ R0, -RZ",
            "F2I.TRUNC.NTZ R0, -UR4",
            "F2I.TRUNC.NTZ R0, -c[0x0][0x380]",
            "F2I.TRUNC.NTZ R0, -R0.H1",
            "F2I.TRUNC.NTZ R0, -R0, R1",
            "F2I.TRUNC R0, -R0",
            "F2I.S32.TRUNC.NTZ R0, -R0",
            "F2I.S64.TRUNC R0, -R2",
            "F2I.U64.TRUNC R0, -R2",
            "F2I.S64.TRUNC.NTZ R0, -R2",
            "F2I.F64.TRUNC R0, -R2",
            "F2I.U32.F64.TRUNC R0, -R2",
            "F2I.S16.TRUNC.NTZ R0, -R2",
            "F2I.U16.TRUNC.NTZ R0, -R2",
        ] {
            assert!(lower1(l).is_err(), "{l} must be refused");
        }
    }

    #[test]
    fn real_kernel_forms() {
        assert!(matches!(
            lower1("CREDUX.MAX.F32.NAN UR10, R12").unwrap(),
            Op::Redux {
                op: ReduxOp::MaxF {
                    abs: false,
                    nan: true
                },
                ..
            }
        ));
        assert!(matches!(
            lower1("CREDUX.MINABS.F32 UR7, R2.reuse").unwrap(),
            Op::Redux {
                op: ReduxOp::MinF {
                    abs: true,
                    nan: false
                },
                ..
            }
        ));
        assert!(lower1("CREDUX.MAX.F64 UR7, R2").is_err());
        assert!(lower1("CREDUX.MAX.F32.FTZ UR7, R2").is_err());
        match lower1("PLOP3.LUT P1, P0, !P0, PT, !P1, 0x80, 0xc8").unwrap() {
            Op::Plop3 {
                lut,
                d2: Some((d2, lut2)),
                ..
            } => {
                assert_eq!((lut, lut2, d2.idx), (0x80, 0xc8, 0));
            }
            o => panic!("{o:?}"),
        }
        assert!(matches!(
            lower1("PLOP3.LUT P0, PT, P1, PT, PT, 0x8, 0x80").unwrap(),
            Op::Plop3 { d2: None, .. }
        ));
        assert!(matches!(
            lower1("R2P PR, R0.B2, 0x7f").unwrap(),
            Op::R2p {
                shift: 16,
                mask: 0x7f,
                ..
            }
        ));
        assert!(lower1("R2P PR, R0.H1, 0x7f").is_err());
        assert!(lower1("R2P PR, R0, 0xff").is_err());
        match lower1("LDTM.x8 R4, tmem[UR5+-0x10]").unwrap() {
            Op::TmemLd {
                off,
                n: 8,
                red: None,
                ..
            } => assert_eq!(off, 0xffff_fff0),
            o => panic!("{o:?}"),
        }
        match lower1("LDTM.STAT.x32.MAX.F32 R37, R4, tmem[UR4+0x20]").unwrap() {
            Op::TmemLd {
                d,
                n: 32,
                off: 0x20,
                red:
                    Some((
                        rd,
                        ReduxOp::MaxF {
                            abs: false,
                            nan: false,
                        },
                    )),
                ..
            } => {
                assert_eq!((d, rd), (r(4), r(37)));
            }
            o => panic!("{o:?}"),
        }
        assert!(matches!(
            lower1("LDTM.STAT.x32.MAX.F32 RZ, R4, tmem[UR4]").unwrap(),
            Op::TmemLd {
                red: None,
                n: 32,
                ..
            }
        ));
        assert!(lower1("LDTM.STAT.x32.MAX.F32 R4, RZ, tmem[UR4]").is_err());
        assert!(lower1("LDTM.STAT.16dp256bit.x8.MAX.F32 R4, R8, tmem[UR4]").is_err());
        assert!(lower1("LDTM.STAT.16dp32bit_t0_t15.x8.MAX.F32 R4, R8, tmem[UR4]").is_ok());
        // Negative offsets on stores too (probes/tmem_stneg): the 32-bit two's complement.
        assert!(matches!(
            lower1("STTM.x8 tmem[UR5+-0x10], R4").unwrap(),
            Op::TmemSt {
                off: 0xffff_fff0,
                ..
            }
        ));
        match lower1("BAR.SYNC.DEFER_BLOCKING R4, R4").unwrap() {
            Op::BarSync {
                packed: true,
                arrive: false,
                ..
            } => {}
            o => panic!("{o:?}"),
        }
        assert!(lower1("BAR.SYNC.DEFER_BLOCKING R4, R5").is_err());
        assert!(matches!(
            lower1("BAR.SYNC.DEFER_BLOCKING R4, 0x40").unwrap(),
            Op::BarSync { packed: false, .. }
        ));
        assert!(matches!(
            lower1("WARPSYNC.COLLECTIVE.ALL `(.L_x_1)").unwrap(),
            Op::WarpSync {
                collective: true,
                all: true,
                ..
            }
        ));
        assert!(matches!(
            lower1("UTMALDG.3D.MULTICAST.2CTA [UR8], [UR10], UR4").unwrap(),
            Op::TmaLoad {
                mask: Some(_),
                pair: true,
                dims: 3,
                ..
            }
        ));
        assert!(matches!(
            lower1("UTMALDG.2D.GATHER4.MULTICAST.2CTA [UR32], [UR58], UR66").unwrap(),
            Op::TmaLoad {
                mask: Some(_),
                pair: true,
                dims: 2,
                ref coords,
                ..
            } if coords.len() == 5
        ));
        assert!(lower1("UTMALDG.2D.GATHER4.2CTA.MULTICAST [UR32], [UR58], UR66").is_err());
        assert!(lower1("UTMALDG.3D.GATHER4 [UR32], [UR58]").is_err());
        assert!(matches!(
            lower1("LDCU UR8, c[0x0][UR5+-0x1c80]").unwrap(),
            Op::LdConst {
                bank: 0,
                off: 0xffff_e380,
                bytes: 4,
                ..
            }
        ));
        assert!(lower1("LDCU UR8, c[0x0][UR5+-0xzz]").is_err());
        assert!(lower1("UTMALDG.3D.IM2COL [UR8], [UR10]").is_err());
        assert!(matches!(
            lower1("UISETP.EQ.U32.XOR UP0, UPT, UR4, 0xc, !UP1").unwrap(),
            Op::Setp { xor: true, .. }
        ));
        assert!(matches!(
            lower1("LEA.HI R5, R6, -R5, RZ, 0x1c").unwrap(),
            Op::LeaHi {
                neg_b: true,
                shift: 0x1c,
                ..
            }
        ));
        assert!(matches!(
            lower1("LEA.HI.X.SX32 R0, R2, ~R3, 0x1, P2").unwrap(),
            Op::LeaHi {
                not_b: true,
                neg_b: false,
                sx32: true,
                cin: Some(_),
                ..
            }
        ));
        assert!(matches!(lower1("UBLKPF.L2 [UR6], UR4").unwrap(), Op::Nop));
    }
}
