/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Kernel directories of the probe corpus (fb/targets/tpu/probes/*/out): `k/<name>/meta.json`
//! with `.npy` inputs and hardware outputs, listings under `llo/<name>/`. Loads a kernel into a
//! `TpuMachine`, runs it and compares the outputs bit for bit (the `run_kernel` of
//! tools/llosim.py).

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_tpu_ir::Front;
use ptxv_tpu_ir::Kernel;
use ptxv_tpu_ir::OperandDecl;
use serde::Serialize;

use crate::data;
use crate::data::Array;
use crate::layout;
use crate::layout::Layout;
use crate::lift;
use crate::machine::DmaPolicy;
use crate::machine::TpuMachine;
use crate::num::Numerics;

/// Two-core kernels: logical device ids of mesh positions 0 and 1.
pub const DEVPAIRS: &[(&str, (u32, u32))] = &[
    ("remote_ring", (0, 1)),
    ("remote_ici", (0, 2)),
    ("remote_sig", (0, 1)),
];

/// The `final_bundles` listing of kernel `name` (highest pass number) and its `original`.
pub fn find_listings(llo_dir: &Path, name: &str) -> Result<(PathBuf, PathBuf)> {
    let mut fins: Vec<PathBuf> = std::fs::read_dir(llo_dir)
        .with_context(|| format!("{}", llo_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let f = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
            f.contains(&format!("-{name}."))
                && f.ends_with("final_bundles.txt")
                && !f.contains("schedule-analysis")
        })
        .collect();
    fins.sort();
    let fin = fins
        .pop()
        .ok_or_else(|| anyhow!("no final_bundles for {name} in {}", llo_dir.display()))?;
    let f = fin.to_string_lossy().to_string();
    let cut = f.rfind("-final_bundles.txt").expect("suffix");
    let orig = PathBuf::from(format!("{}-00-original.txt", &f[..cut - 3]));
    Ok((fin, orig))
}

/// A kernel directory resolved to its listings and metadata.
pub struct KernelDir {
    pub name: String,
    pub dir: PathBuf,
    pub meta: serde_json::Value,
    pub final_text: String,
    pub original_text: String,
    pub pair: Option<(u32, u32)>,
    pub llo_dir: PathBuf,
    pub final_path: PathBuf,
    /// Serialized executable (`<out>/rt/<name>.bin`), source of the branch delays.
    pub exe: Option<PathBuf>,
    /// Listing mode (`from_listing`): devices 0..n run the program, inputs are zeros, no
    /// expected outputs.
    pub listing_devices: Option<usize>,
    /// Operand layouts of the kernel's custom call in the HLO dumps next to the listings
    /// (inputs, outputs), or why they could not be read. ASSUME(TPU-L-OPERAND-LAYOUT)
    pub layouts: std::result::Result<(Vec<Layout>, Vec<Layout>), String>,
}

impl KernelDir {
    pub fn open(kdir: &Path) -> Result<Self> {
        let dir = std::fs::canonicalize(kdir).with_context(|| format!("{}", kdir.display()))?;
        let name = dir
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or_else(|| anyhow!("kernel dir name"))?
            .to_string();
        let root = dir
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| anyhow!("kernel dir layout"))?;
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
        let llo_dir = root.join("llo").join(&name);
        let (fin, orig) = find_listings(&llo_dir, &name)?;
        // two-device kernels: logical device ids from meta.json `devices` (`TPU_<id>(...)`)
        let ids: Vec<u32> = meta["devices"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|d| {
                        d.as_str()?
                            .strip_prefix("TPU_")?
                            .split('(')
                            .next()?
                            .parse()
                            .ok()
                    })
                    .collect()
            })
            .unwrap_or_default();
        let pair = match ids[..] {
            [a, b] => Some((a, b)),
            _ => DEVPAIRS.iter().find(|(n, _)| *n == name).map(|x| x.1),
        };
        let exe = [
            root.join("rt").join(format!("{name}.bin")),
            dir.join("exe.bin"),
        ]
        .into_iter()
        .find(|p| p.exists());
        Ok(KernelDir {
            listing_devices: None,
            layouts: layout::kernel_layouts(&llo_dir, &name).map_err(|e| format!("{e:#}")),
            llo_dir,
            exe,
            name,
            dir,
            meta,
            final_text: std::fs::read_to_string(&fin)?,
            original_text: std::fs::read_to_string(&orig)?,
            final_path: fin,
            pair,
        })
    }

    /// Lift one front. Final listings take branch delays from the executable when the
    /// kernel directory has one (falling back to the inferred rule with a note).
    pub fn lift(&self, front: Front) -> Kernel {
        match front {
            Front::Final => {
                let mut note = None;
                let enc = self.exe.as_ref().and_then(|p| {
                    let listing = crate::exe::listing_name(&self.final_path)?;
                    let r = std::fs::read(p)
                        .map_err(anyhow::Error::from)
                        .and_then(|b| crate::exe::branch_delays(&b, &self.llo_dir, &listing));
                    match r {
                        Ok(m) => Some(m),
                        Err(e) => {
                            note = Some(format!("executable {}: {e}", p.display()));
                            None
                        }
                    }
                });
                let mut k = lift::lift_final_exe(
                    &self.name,
                    &self.final_text,
                    &self.original_text,
                    enc.as_ref(),
                );
                k.notes.extend(note);
                k
            }
            Front::Original => lift::lift_original(&self.name, &self.original_text),
        }
    }

    /// A bare listing (`*-final_bundles.txt`, with its `-00-original.txt` next to it): zero
    /// inputs, `ndev` devices (logical ids 0..ndev) when the program talks to other cores.
    pub fn from_listing(final_path: &Path, exe: Option<PathBuf>, ndev: usize) -> Result<Self> {
        let final_path = std::fs::canonicalize(final_path)?;
        let llo_dir = final_path
            .parent()
            .ok_or_else(|| anyhow!("listing dir"))?
            .to_path_buf();
        let name = crate::exe::listing_name(&final_path)
            .ok_or_else(|| anyhow!("not a final_bundles listing"))?;
        let f = final_path.to_string_lossy().to_string();
        let cut = f.rfind("-final_bundles.txt").expect("suffix");
        let orig = PathBuf::from(format!("{}-00-original.txt", &f[..cut - 3]));
        let final_text = std::fs::read_to_string(&final_path)?;
        let remote = final_text.contains("vsyncadd.remote") || final_text.contains("dma.general");
        Ok(KernelDir {
            layouts: layout::kernel_layouts(&llo_dir, &name).map_err(|e| format!("{e:#}")),
            name,
            dir: llo_dir.clone(),
            meta: serde_json::Value::Null,
            original_text: std::fs::read_to_string(&orig).unwrap_or_default(),
            final_text,
            pair: None,
            llo_dir,
            final_path,
            exe,
            listing_devices: Some(if remote { ndev.max(1) } else { 1 }),
        })
    }

    pub fn ncores(&self) -> usize {
        match (self.pair, self.listing_devices) {
            (Some(_), _) => 2,
            (None, Some(n)) => n,
            _ => 1,
        }
    }

    /// (chip, core in chip) -> mesh position. ASSUME(TPU-L-DEVICE-IDS)
    pub fn devmap(&self) -> HashMap<(u32, u32), usize> {
        match self.pair {
            Some((a, b)) => [(0usize, a), (1, b)]
                .into_iter()
                .map(|(pos, d)| ((d >> 1, d & 1), pos))
                .collect(),
            None => (0..self.listing_devices.unwrap_or(1) as u32)
                .map(|d| ((d >> 1, d & 1), d as usize))
                .collect(),
        }
    }

    fn input(&self, core: usize, idx: u32) -> Result<Array> {
        if self.listing_devices.is_some() {
            bail!("listing mode has no input files");
        }
        if self.pair.is_some() {
            return data::read_npy(&self.dir.join(format!("d{core}_in{idx}.npy")));
        }
        let ent = &self.meta["inputs"][idx as usize];
        if let Some(g) = ent.get("gen") {
            if g["kind"] != "mod_iota" {
                bail!("input generator {g}");
            }
            let m = g["mod"].as_u64().ok_or_else(|| anyhow!("mod"))?;
            let shape: Vec<u64> = g["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("shape"))?
                .iter()
                .filter_map(|x| x.as_u64())
                .collect();
            let n: u64 = shape.iter().product();
            let (dt, descr) = match g["dtype"].as_str() {
                Some("float32") => (0, "<f4"),
                Some("int32") => (1, "<i4"),
                d => bail!("generator dtype {d:?}"),
            };
            let data = (0..n)
                .map(|i| {
                    if dt == 0 {
                        ((i % m) as f32).to_bits()
                    } else {
                        (i % m) as u32
                    }
                })
                .collect();
            return Ok(Array {
                shape,
                itemsize: 4,
                descr: descr.into(),
                data,
            });
        }
        data::read_npy(&self.dir.join(format!("in{idx}.npy")))
    }

    /// The HLO layout of operand `d`; an operand whose layout cannot be read is unmodeled.
    pub fn layout(&self, d: &OperandDecl) -> Result<&Layout> {
        let ls = self
            .layouts
            .as_ref()
            .map_err(|e| anyhow!("unmodeled operand layout: {e}"))?;
        layout::for_decl(ls, d).map_err(|e| anyhow!("unmodeled operand layout: {e:#}"))
    }

    /// A machine with the kernel loaded and its operands placed.
    pub fn machine(&self, kernel: Arc<Kernel>, num: &'static dyn Numerics) -> Result<TpuMachine> {
        let mut m = TpuMachine::new(kernel.clone(), self.ncores(), self.devmap(), num);
        for c in 0..self.ncores() {
            for d in &kernel.operands {
                if self.listing_devices.is_some() {
                    // zeros; element types the host side does not model are sized as 16-bit
                    let isz = data::dtype(&d.dtype).map(|x| x.0).unwrap_or(2);
                    let isz = if d.space == "smem" || d.shape.len() == 1 {
                        4
                    } else {
                        isz
                    };
                    if d.space == "none" {
                        m.place(c, d, &[], Some(0))?;
                    } else {
                        // zero inputs: the layout only sizes the buffer; without an HLO dump the
                        // default tiling is used (listing mode only)
                        let n = match self.layout(d) {
                            Ok(l) => layout::n_words(l, isz)?,
                            Err(_) => data::n_words(&d.shape, isz, &d.space),
                        };
                        m.place(c, d, &vec![0; n], None)?;
                    }
                    continue;
                }
                let (isz, descr) = data::dtype(&d.dtype)?;
                let arr = if d.output {
                    Array::zeros(&d.shape, isz, descr)
                } else {
                    self.input(c, d.index)?
                };
                if d.space == "none" {
                    m.place(c, d, &[], arr.data.first().copied())?;
                    continue;
                }
                let words = layout::to_words(&arr, self.layout(d)?)?;
                m.place(c, d, &words, None)?;
            }
        }
        Ok(m)
    }

    /// Output arrays of each core, in operand order.
    pub fn outputs(&self, m: &TpuMachine) -> Result<Vec<(usize, Array)>> {
        let mut out = Vec::new();
        for p in m.placed.iter().filter(|p| p.decl.output) {
            let (isz, descr) = data::dtype(&p.decl.dtype)?;
            let w = m.operand_words(p)?;
            let a = match self.layout(&p.decl) {
                Ok(l) => layout::from_words(&w, l, isz, descr)?,
                Err(_) if self.listing_devices.is_some() => {
                    data::from_words(&w, &p.decl.shape, isz, descr, &p.space)
                }
                Err(e) => return Err(e),
            };
            out.push((p.core, a));
        }
        Ok(out)
    }

    /// Hardware output `oi` of core `c` and whether it is declared nondeterministic.
    pub fn expected(&self, c: usize, oi: usize) -> Result<(Array, bool)> {
        let stem = if self.pair.is_some() {
            format!("d{c}_out{oi}")
        } else {
            format!("out{oi}")
        };
        let nd = self.meta["nondeterministic_outputs"]
            .as_array()
            .is_some_and(|a| a.iter().any(|x| x.as_str() == Some(&stem)));
        Ok((data::read_npy(&self.dir.join(format!("{stem}.npy")))?, nd))
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct OutCmp {
    pub core: usize,
    pub out: usize,
    pub mismatch: usize,
    pub n: usize,
    pub nondet: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunResult {
    pub kernel: String,
    pub front: String,
    pub status: String,
    pub error: String,
    pub bundles: usize,
    pub steps: u64,
    pub exact: Option<bool>,
    pub outputs: Vec<OutCmp>,
    #[serde(skip)]
    pub arrays: Vec<(usize, Array)>,
}

/// Lift, run with `policy` and compare against the hardware outputs.
pub fn run_kernel(
    kd: &KernelDir,
    front: Front,
    policy: DmaPolicy,
    num: &'static dyn Numerics,
) -> Result<RunResult> {
    let kernel = Arc::new(kd.lift(front));
    if kd.listing_devices.is_none()
        && let Some(e) = kernel
            .operands
            .iter()
            .filter(|d| d.space != "none")
            .find_map(|d| kd.layout(d).err())
    {
        return Ok(RunResult {
            kernel: kd.name.clone(),
            front: format!("{front:?}").to_lowercase(),
            status: "Unimpl".into(),
            error: e.to_string(),
            bundles: kernel.bundles.len(),
            steps: 0,
            exact: None,
            outputs: Vec::new(),
            arrays: Vec::new(),
        });
    }
    let mut m = kd.machine(kernel.clone(), num)?;
    let mut res = RunResult {
        kernel: kd.name.clone(),
        front: format!("{front:?}").to_lowercase(),
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
        if bad > 0 && !nondet {
            exact = false;
        }
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
