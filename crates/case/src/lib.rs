/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Test cases: a kernel plus everything needed to launch it, and its mutant variants.
//!
//! A case is a directory with `case.toml` and the kernel source. Until `ptxv-capture` exists
//! (ROADMAP rung 2), the manifest plays the role of a `.ptxcap`: launch dims, buffers, and
//! parameters, including tensor maps described by their encode arguments.
//!
//! Variants are textual edits on lines carrying a `// @<marker>` comment, so the good kernel
//! and each mutant differ by exactly one visible line.

use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub description: String,
    /// The kernel source; empty for a binary-only case (`cubin`).
    #[serde(default)]
    pub ptx: String,
    /// Binary-only case (a library kernel, no PTX): the cubin, verified through the SASS
    /// front-end only.
    #[serde(default)]
    pub cubin: Option<String>,
    pub entry: String,
    #[serde(default = "default_arch")]
    pub arch: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic shared memory bytes (`.extern .shared x[]`), as the launch passes it.
    #[serde(default)]
    pub dynamic_smem: u32,
    /// Thread-block cluster shape (the launch attribute / `.reqnctapercluster`).
    /// None: no cluster (`%is_explicit_cluster` = 0).
    #[serde(default)]
    pub cluster: Option<[u32; 3]>,
    pub outputs: Vec<String>,
    /// Groups of output buffers that must hold identical bytes at the end (replicas, e.g. the
    /// ranks of an all-reduce). A difference is a "divergent" finding (N4).
    #[serde(default)]
    pub replicas: Vec<Vec<String>>,
    #[serde(default)]
    pub spec: Option<String>,
    /// Tensor-core arithmetic model name (default: the calibrated one).
    #[serde(default)]
    pub mma_model: Option<String>,
    /// Kernel completion as a fence (E-EXIT-FENCE): "sys" / "gpu" make each thread's EXIT a
    /// `fence.acq_rel` at that scope, so a relaxed observation of a peer's release is acquired
    /// by the time the next kernel on the GPU starts. None (default): the strict PTX model,
    /// where only an acquire fence does that. `PTXV_EXIT_FENCE=none|gpu|sys` overrides it.
    #[serde(default)]
    pub exit_fence: Option<String>,
    /// E-POLL-ACQ (rung 4b, opt-in): a thread leaving a poll loop (E-SPIN) acquires what its
    /// polls observed, as if the loop ended with an acquire fence. DeepEP's `barrier_block`
    /// exits on relaxed `ld.volatile` polls with no fence. `PTXV_POLL_ACQUIRE=0|1` overrides.
    /// REFUTED on GB300: takes effect only with an explicit opt-in (`assume_refuted`).
    #[serde(default)]
    pub poll_acquire: bool,
    /// Refuted assumptions (ASSUMPTIONS.md status "refuted") the case may rely on, e.g.
    /// `["E-POLL-ACQ"]`. Without it, a case that sets `poll_acquire` is refused. Verdicts name
    /// the opt-in. `ptxv check --assume-refuted ID` / `PTXV_ASSUME_REFUTED=ID,..` add to it.
    #[serde(default)]
    pub assume_refuted: Vec<String>,
    /// E-SMEM-CUMUL (rung 4b, opt-in): a SASS shared-memory store releases, at CTA scope,
    /// what its thread acquired from others (not its own earlier writes). DeepEP combine relays
    /// acquired channel tails to its warps through `st.volatile.shared` with no fence.
    /// `PTXV_SMEM_CUMULATIVE=0|1` overrides.
    #[serde(default)]
    pub smem_cumulative: bool,
    /// E-HW-L1 (opt-in): model GB300's non-coherent per-SM L1 (crates/sim/src/l1.rs): weak
    /// global loads can hit stale lines; reported as "stale". `PTXV_HW_L1=0|1` overrides.
    #[serde(default)]
    pub hw_l1: bool,
    /// E-L1-CORES (opt-in, implies `hw_l1`): launches that are not stream-ordered may have CTAs
    /// on the same SM and share its L1 lines; the SM placement is a nondeterministic choice
    /// (crates/sim/src/l1_cores.rs). `PTXV_L1_CORES=0|1` overrides.
    #[serde(default)]
    pub l1_coresident: bool,
    /// E-L1-HAVOC (opt-in, implies `hw_l1`): any agent outside the case may fill any line of a
    /// case buffer on any SM at any time; a weak load that such a fill could make stale is
    /// reported (`stale-l1-havoc`). `PTXV_L1_HAVOC=0|1` overrides.
    #[serde(default)]
    pub l1_havoc: bool,
    /// E-RESIDENCY (opt-in): at most this many CTAs of the case's kernels are resident per GPU
    /// at once; CTAs (whole clusters) become resident in any order as slots free up
    /// (crates/sim/src/residency.rs). None: every CTA is resident (the default model).
    /// `PTXV_RESIDENCY=N` overrides; `ptxv check --residency auto` searches the minimum.
    #[serde(default)]
    pub residency: Option<u32>,
    /// The launch is cooperative (`cuLaunchCooperativeKernel`): the driver guarantees that
    /// every CTA is co-resident, or refuses the launch. Only used to report whether a needed
    /// residency is guaranteed.
    #[serde(default)]
    pub cooperative: bool,
    /// E-TMEM-COTENANT (opt-in): an adversarial co-tenant shares each SM's tensor memory
    /// (crates/sim/src/tenant.rs). `PTXV_TMEM_COTENANT=0|1` overrides.
    #[serde(default)]
    pub tmem_cotenant: bool,
    /// E-TMEM-PERMIT in co-tenant mode: a CTA's allocation permit blocks a co-tenant's
    /// allocations until it relinquishes or exits (default true; false: columns only).
    #[serde(default)]
    pub tmem_permit: Option<bool>,
    /// E-READ-DONE (rung 4c, opt-in): a read whose value an instruction of its thread has used
    /// (or an async read whose completion was awaited) is ordered, for write-after-read
    /// purposes, before every later write that depends on a strong store the thread made after
    /// it (no load buffering of completed reads). DeepEP's ring head is a relaxed store after
    /// the receivers consumed their slots. `PTXV_READ_DONE=0|1` overrides.
    #[serde(default)]
    pub read_done: bool,
    pub buffers: Vec<Buffer>,
    pub params: Vec<ParamSpec>,
    /// Further kernels launched after the first one on the same stream, in order (same PTX
    /// file, same buffers). Rung 1e: programmatic dependent launch.
    #[serde(default)]
    pub launches: Vec<NextLaunch>,
    pub variants: Vec<Variant>,
}

/// A kernel launched after the previous one on the same stream.
#[derive(Debug, Clone, Deserialize)]
pub struct NextLaunch {
    pub entry: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    #[serde(default)]
    pub cluster: Option<[u32; 3]>,
    pub params: Vec<ParamSpec>,
    /// Launched with programmatic stream serialization (cudaLaunchAttribute
    /// ProgrammaticStreamSerialization): it may start before the previous kernel completes.
    #[serde(default)]
    pub pdl: bool,
    #[serde(default)]
    pub dynamic_smem: u32,
    /// GPU (index into the GPUs the case runs on). A kernel on another GPU than every earlier
    /// one is launched independently: nothing orders it with them (rung 1f).
    #[serde(default)]
    pub device: u32,
    /// Binary-only case: the cubin defining this kernel when it is not the manifest's
    /// (`tools/capture.py`: one op's kernels may come from torch, cuBLAS and cuDNN modules).
    #[serde(default)]
    pub cubin: Option<String>,
    /// Stream order override (rung 4b): the index (in `all_launches` order) of the launch
    /// this one waits for instead of the previous launch on its GPU; -1: none (a dropped
    /// event wait between streams). Set by a variant's `after`.
    #[serde(default)]
    pub after: Option<i64>,
}

/// One kernel launch of a case (the first is the manifest's top level).
#[derive(Debug, Clone)]
pub struct LaunchSpec<'a> {
    pub entry: &'a str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub cluster: Option<[u32; 3]>,
    pub params: &'a [ParamSpec],
    pub pdl: bool,
    pub device: u32,
    pub dynamic_smem: u32,
    /// Its own cubin (`NextLaunch::cubin`); None: the case's.
    pub cubin: Option<&'a str>,
    /// `NextLaunch::after`.
    pub after: Option<i64>,
}

impl Manifest {
    /// Number of GPUs the case's kernels run on.
    pub fn ndevices(&self) -> u32 {
        1 + self.launches.iter().map(|l| l.device).max().unwrap_or(0)
    }

    /// Every kernel launch of the case, in stream order.
    pub fn all_launches(&self) -> Vec<LaunchSpec<'_>> {
        let mut v = vec![LaunchSpec {
            entry: &self.entry,
            grid: self.grid,
            block: self.block,
            cluster: self.cluster,
            params: &self.params,
            pdl: false,
            device: 0,
            dynamic_smem: self.dynamic_smem,
            cubin: None,
            after: None,
        }];
        for l in &self.launches {
            v.push(LaunchSpec {
                entry: &l.entry,
                grid: l.grid,
                block: l.block,
                cluster: l.cluster,
                params: &l.params,
                pdl: l.pdl,
                device: l.device,
                dynamic_smem: l.dynamic_smem,
                cubin: l.cubin.as_deref(),
                after: l.after,
            });
        }
        v
    }
}

fn default_arch() -> String {
    "sm_103a".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct Buffer {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    /// "normal" | "uniform" | "zeros" | "nan" | "iota" | "bits" (random bit patterns) |
    /// "file" (the exact bytes of `file`, e.g. a captured tensor)
    pub init: String,
    /// Raw contents for `init = "file"` (relative to the case directory; resolved by
    /// `Case::load`).
    #[serde(default)]
    pub file: Option<String>,
    /// Output buffers: the bytes the GPU produced when the case was captured (a file
    /// relative to the case directory), compared bitwise with the simulator by `check`.
    #[serde(default)]
    pub expect: Option<String>,
    /// Byte offset in the buffer where `expect` starts (a tensor inside a larger allocation).
    #[serde(default)]
    pub expect_offset: u64,
    #[serde(default)]
    pub seed: u64,
    /// GPU whose memory holds it (the others reach it as peer memory over NVLink).
    #[serde(default)]
    pub device: u32,
    /// (element index, value) pairs that override `init` (equivalence witnesses).
    #[serde(default)]
    pub values: Vec<(u64, f64)>,
    /// Pointer words stored in the buffer: a device-side pointer table (torch symmetric
    /// memory passes `T** buffer_ptrs_dev`, one peer pointer per rank). Each is filled with the
    /// named buffer's address (+ `byte_offset`) once addresses are assigned, in the simulator
    /// and on hardware (tools/capture_dist.py).
    #[serde(default)]
    pub ptrs: Vec<BufferPtr>,
}

/// An 8-byte pointer to `buffer` + `byte_offset`, stored at byte `offset` of a buffer.
#[derive(Debug, Clone, Deserialize)]
pub struct BufferPtr {
    pub offset: u64,
    pub buffer: String,
    #[serde(default)]
    pub byte_offset: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ParamSpec {
    /// Pointer to a buffer (passed as a 64-bit global address), `byte_offset` bytes in.
    Ptr {
        buffer: String,
        #[serde(default, rename = "byte_offset")]
        offset: u64,
    },
    /// A 128-byte CUtensorMap built with cuTensorMapEncodeTiled over `buffer`
    /// (row-major; dims are innermost-first like the PTX coordinates).
    Tensormap {
        buffer: String,
        r#box: Vec<u32>,
        /// Override dims (elements, innermost first); default = buffer shape reversed.
        #[serde(default)]
        dims: Option<Vec<u64>>,
        /// Override byte strides of dims 1..; default = row-major strides.
        #[serde(default)]
        strides: Option<Vec<u64>>,
        /// Bytes from the buffer's start to the tensor's (a map over a tensor inside a larger
        /// allocation).
        #[serde(default, rename = "byte_offset")]
        offset: u64,
        /// Shared-memory swizzle: "none" (default), "32B", "64B", "128B".
        #[serde(default)]
        swizzle: Option<String>,
        /// Hardware runs only: the descriptor bytes as captured (hex, up to 128 bytes; the
        /// first 8 are replaced by the buffer's address). For frameworks that build
        /// descriptors with bits `cuTensorMapEncodeTiled` does not set; the model uses the
        /// geometry above, which the capture matched against these bytes.
        #[serde(default)]
        raw: Option<String>,
        /// 4-bit elements, two per byte (`CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`; `raw` required
        /// for hardware runs): `dims`, `box` and the buffer are given in bytes, and TMA
        /// coordinate 0 counts elements (it must be even). ASSUME(S-TMA)
        #[serde(default)]
        elem4: bool,
    },
    U32 {
        value: u32,
    },
    U64 {
        value: u64,
    },
    /// Raw parameter bytes (hex, little-endian as passed), e.g. a by-value struct captured
    /// from a JIT framework's launch (probes/launchcap). `fields` are pointers and tensor maps
    /// inside it (a library's packed argument block), filled in per run like top-level ones.
    Bytes {
        hex: String,
        #[serde(default)]
        fields: Vec<ParamField>,
    },
}

/// A `ptr` or `tensormap` parameter at byte `offset` inside a `bytes` parameter.
#[derive(Debug, Clone, Deserialize)]
pub struct ParamField {
    pub offset: u32,
    #[serde(flatten)]
    pub spec: ParamSpec,
}

/// Decode a `Bytes` parameter.
pub fn param_bytes(hex: &str) -> Result<Vec<u8>> {
    if !hex.len().is_multiple_of(2) {
        bail!("odd-length hex parameter");
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| anyhow!("bad hex parameter: {e}"))
        })
        .collect()
}

#[derive(Debug, Clone, Deserialize)]
pub struct Variant {
    pub id: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub marker: Option<String>,
    #[serde(default)]
    pub find: Option<String>,
    /// Replacement text; the special value "<delete>" removes the marked line.
    #[serde(default)]
    pub replace: Option<String>,
    /// Further edits, applied in order after the first (a variant that changes two places).
    #[serde(default)]
    pub edits: Vec<Edit>,
    /// Expected verdict classes, e.g. ["clean"], ["deadlock"], ["race"].
    pub expect: Vec<String>,
    /// Expected classes for the SASS front-end when they differ from the PTX front-end
    /// (PTX-level undefined behavior that this ptxas happens to compile safely).
    #[serde(default)]
    pub expect_sass: Option<Vec<String>>,
    /// Binary-only mutant: edits of SASS control bits applied to the compiled cubin (the PTX
    /// is the base kernel's). See crates/sass/src/scoreboard.rs.
    #[serde(default)]
    pub patch: Vec<ptxv_ir::CtrlPatch>,
    /// How the bug behaves on hardware (documentation).
    #[serde(default)]
    pub hw_note: String,
    /// Stream-order mutants (rung 4b): `[launch, after]` pairs setting `NextLaunch::after`
    /// (indices in `all_launches` order; after = -1: ordered after nothing). Applied by
    /// `Case::for_variant`.
    #[serde(default)]
    pub after: Vec<[i64; 2]>,
    /// E-RESIDENCY: the expected minimum number of co-resident CTAs per GPU for which every
    /// residency at or above it is deadlock-free (`residency::analyze`); 0: deadlocks even
    /// with every CTA resident.
    #[serde(default)]
    pub min_residency: Option<u32>,
    /// E-TMEM-COTENANT: the expected verdict classes in co-tenant mode (`tenant::check`).
    #[serde(default)]
    pub expect_cotenant: Option<Vec<String>>,
}

/// One textual edit on the lines tagged `// <marker>`.
#[derive(Debug, Clone, Deserialize)]
pub struct Edit {
    pub marker: String,
    #[serde(default)]
    pub find: Option<String>,
    pub replace: String,
}

#[derive(Debug, Clone)]
pub struct Case {
    pub dir: PathBuf,
    pub m: Manifest,
}

impl Buffer {
    /// The initial bytes with the pointer words (`ptrs`) filled in; `base_of` gives a
    /// buffer's address.
    pub fn init_bytes_with(&self, base_of: impl Fn(&str) -> Result<u64>) -> Result<Vec<u8>> {
        let mut b = self.init_bytes();
        for p in &self.ptrs {
            let o = p.offset as usize;
            let w = b
                .get_mut(o..o + 8)
                .ok_or_else(|| anyhow!("buffer {}: pointer at {o} past its end", self.name))?;
            w.copy_from_slice(&(base_of(&p.buffer)? + p.byte_offset).to_le_bytes());
        }
        Ok(b)
    }

    pub fn elem_bytes(&self) -> usize {
        match self.dtype.as_str() {
            "bf16" | "f16" | "u16" | "i16" => 2,
            "f32" | "u32" | "i32" => 4,
            "f64" | "u64" | "i64" => 8,
            "u8" | "i8" | "e4m3" => 1,
            d => panic!("unknown dtype {d}"),
        }
    }
    pub fn elems(&self) -> usize {
        self.shape.iter().product::<u64>() as usize
    }
    pub fn bytes(&self) -> usize {
        self.elems() * self.elem_bytes()
    }

    /// Deterministic contents (identical in the simulator and on the GPU).
    pub fn init_bytes(&self) -> Vec<u8> {
        let n = self.elems();
        let mut rng = SplitMix(self.seed.wrapping_add(0x9e37_79b9_7f4a_7c15));
        if self.init == "file" {
            let p = self
                .file
                .as_deref()
                .expect("init = \"file\" without `file`");
            let b = std::fs::read(p).unwrap_or_else(|e| panic!("buffer {}: {p}: {e}", self.name));
            assert_eq!(
                b.len(),
                self.bytes(),
                "buffer {}: {p} has the wrong size",
                self.name
            );
            return b;
        }
        if self.init == "bits" {
            // Uniformly random bit patterns (integer / ALU tests).
            let mut out: Vec<u8> = (0..self.bytes().div_ceil(8))
                .flat_map(|_| rng.next().to_le_bytes())
                .collect();
            out.truncate(self.bytes());
            return out;
        }
        let mut vals = Vec::with_capacity(n);
        for i in 0..n {
            let v = match self.init.as_str() {
                "normal" => rng.normal(),
                "uniform" => rng.unit() * 2.0 - 1.0,
                "zeros" => 0.0,
                "nan" => f64::NAN,
                "iota" => i as f64,
                s => panic!("unknown init {s}"),
            };
            vals.push(v);
        }
        for &(i, v) in &self.values {
            vals[i as usize] = v;
        }
        let mut out = Vec::with_capacity(self.bytes());
        for v in vals {
            match self.dtype.as_str() {
                "f32" => out.extend_from_slice(&(v as f32).to_bits().to_le_bytes()),
                "bf16" => out.extend_from_slice(&f32_to_bf16_rn(v as f32).to_le_bytes()),
                "u32" => out.extend_from_slice(&(v as u32).to_le_bytes()),
                "u64" => out.extend_from_slice(&(v as u64).to_le_bytes()),
                "i32" => out.extend_from_slice(&(v as i32).to_le_bytes()),
                "i64" => out.extend_from_slice(&(v as i64).to_le_bytes()),
                "u8" => out.push(v as u8),
                "u16" => out.extend_from_slice(&(v as u16).to_le_bytes()),
                "f16" => out.extend_from_slice(&f64_to_f16_rn(v).to_le_bytes()),
                d => panic!("init for dtype {d} not implemented"),
            }
        }
        out
    }
}

/// f32 -> bf16 with round-to-nearest-even (NaN stays NaN).
/// The f16 nearest to the f32 nearest to `v` (ties to even; overflow to infinity).
pub fn f64_to_f16_rn(v: f64) -> u16 {
    let f = v as f32;
    if f.is_nan() {
        return 0x7e00;
    }
    let s = if f.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = f.abs() as f64;
    if a >= 65520.0 {
        return s | 0x7c00;
    }
    // Spacing of f16 at a: 2^-24 below 2^-14, else 2^(e - 10).
    let e = if a < 2f64.powi(-14) {
        -14
    } else {
        a.log2().floor() as i32
    };
    let e = if a >= 2f64.powi(-14) && 2f64.powi(e) > a {
        e - 1
    } else {
        e
    };
    let q = 2f64.powi(e.max(-14) - 10);
    let m = (a / q).round_ties_even();
    let r = m * q;
    // Encode r exactly (a value of the f16 lattice).
    if r < 2f64.powi(-14) {
        return s | (r / 2f64.powi(-24)) as u16;
    }
    let er = r.log2().floor() as i32;
    let er = if 2f64.powi(er) > r { er - 1 } else { er };
    let mant = (r / 2f64.powi(er) - 1.0) * 1024.0;
    s | (((er + 15) as u16) << 10) | mant as u16
}

pub fn f32_to_bf16_rn(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    let lsb = (b >> 16) & 1;
    ((b.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

impl Variant {
    pub fn expect_for(&self, front: &str) -> Vec<String> {
        match (front, &self.expect_sass) {
            ("sass", Some(e)) => e.clone(),
            _ => self.expect.clone(),
        }
    }
}

/// Dims (innermost first) and byte strides (dims 1..) of a tensor map over a buffer.
/// Swizzle span in bytes (0 = none) and CUtensorMapSwizzle enum value.
pub fn swizzle_of(s: &Option<String>) -> anyhow::Result<(u32, u32)> {
    Ok(match s.as_deref().unwrap_or("none") {
        "none" => (0, 0),
        "32B" => (32, 1),
        "64B" => (64, 2),
        "128B" => (128, 3),
        x => anyhow::bail!("unknown swizzle {x}"),
    })
}

pub fn tmap_geometry(
    b: &Buffer,
    dims: &Option<Vec<u64>>,
    strides: &Option<Vec<u64>>,
) -> (Vec<u64>, Vec<u64>) {
    let d: Vec<u64> = dims
        .clone()
        .unwrap_or_else(|| b.shape.iter().rev().copied().collect());
    let s = strides.clone().unwrap_or_else(|| {
        let mut v = Vec::new();
        let mut acc = b.elem_bytes() as u64;
        for x in &d[..d.len() - 1] {
            acc *= x;
            v.push(acc);
        }
        v
    });
    (d, s)
}

impl Case {
    pub fn load(dir: impl AsRef<Path>) -> Result<Case> {
        let dir = dir.as_ref().to_path_buf();
        let p = dir.join("case.toml");
        let m: Manifest = toml::from_str(
            &std::fs::read_to_string(&p).with_context(|| format!("{}", p.display()))?,
        )
        .with_context(|| format!("parsing {}", p.display()))?;
        let mut m = m;
        for b in &mut m.buffers {
            for f in [&mut b.file, &mut b.expect].into_iter().flatten() {
                *f = dir.join(&*f).to_string_lossy().into_owned();
            }
        }
        Ok(Case { dir, m })
    }

    /// The case as variant `id` runs it: its launch-order edits (`Variant::after`) applied.
    pub fn for_variant(&self, id: &str) -> Result<Case> {
        let mut c = self.clone();
        for &[l, a] in &self.variant(id)?.after {
            if l < 1 || l as usize > c.m.launches.len() || a >= l || a < -1 {
                bail!("variant {id}: bad after = [{l}, {a}]");
            }
            c.m.launches[l as usize - 1].after = Some(a);
        }
        Ok(c)
    }

    pub fn variant(&self, id: &str) -> Result<&Variant> {
        self.m
            .variants
            .iter()
            .find(|v| v.id == id)
            .ok_or_else(|| anyhow!("no variant {id}"))
    }

    /// Compare a terminal output (the output buffers concatenated in `outputs` order) with
    /// the captured GPU bytes (`expect`). One line per output buffer that has them; empty when
    /// none has.
    pub fn expect_report(&self, output: &[u8]) -> Result<Vec<String>> {
        let mut lines = Vec::new();
        let mut at = 0;
        for o in &self.m.outputs {
            let b = self.buffer(o)?;
            let got = output
                .get(at..at + b.bytes())
                .ok_or_else(|| anyhow!("output shorter than the output buffers"))?;
            at += b.bytes();
            let Some(p) = &b.expect else { continue };
            let want = std::fs::read(p).with_context(|| p.clone())?;
            let off = b.expect_offset as usize;
            let got = got
                .get(off..off + want.len())
                .ok_or_else(|| anyhow!("{o}: expected bytes run past the buffer"))?;
            let diff: Vec<usize> = (0..want.len()).filter(|&i| got[i] != want[i]).collect();
            lines.push(match diff.first() {
                None => format!(
                    "{o}: bitwise equal to the captured GPU output ({} bytes)",
                    want.len()
                ),
                Some(&i) => format!(
                    "{o}: MISMATCH vs the captured GPU output: {} of {} bytes differ, first at byte {} (sim {:#04x}, gpu {:#04x})",
                    diff.len(),
                    want.len(),
                    off + i,
                    got[i],
                    want[i]
                ),
            });
        }
        Ok(lines)
    }

    pub fn buffer(&self, name: &str) -> Result<&Buffer> {
        self.m
            .buffers
            .iter()
            .find(|b| b.name == name)
            .ok_or_else(|| anyhow!("no buffer {name}"))
    }

    /// PTX text of a variant (the good kernel for variants without a marker).
    pub fn variant_ptx(&self, id: &str) -> Result<String> {
        let mut src = std::fs::read_to_string(self.dir.join(&self.m.ptx))?;
        let v = self.variant(id)?;
        if let Some(marker) = &v.marker {
            src = apply_edit(&src, id, marker, v.find.as_deref(), v.replace.as_deref())?;
        }
        for e in &v.edits {
            src = apply_edit(&src, id, &e.marker, e.find.as_deref(), Some(&e.replace))?;
        }
        Ok(src)
    }
}

/// Replace `find` with `replace` on the one line tagged `// <marker>`, or delete every tagged
/// line (`replace = "<delete>"`, e.g. a wait loop).
fn apply_edit(
    src: &str,
    id: &str,
    marker: &str,
    find: Option<&str>,
    replace: Option<&str>,
) -> Result<String> {
    let tag = format!("// {marker}");
    let mut hits = 0;
    let mut out = String::new();
    for line in src.lines() {
        if line.contains(&tag) {
            hits += 1;
            match replace {
                Some("<delete>") => continue,
                Some(r) => {
                    let f = find.ok_or_else(|| anyhow!("variant {id}: replace without find"))?;
                    if !line.contains(f) {
                        bail!("variant {id}: `{f}` not found on marked line");
                    }
                    out.push_str(&line.replacen(f, r, 1));
                }
                None => bail!("variant {id}: marker without replace"),
            }
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    let ok = if replace == Some("<delete>") {
        hits >= 1
    } else {
        hits == 1
    };
    if !ok {
        bail!("variant {id}: marker {marker} found {hits} times");
    }
    Ok(out)
}

/// Third-party (vendor) kernel binaries are not part of the open-source export; they are
/// listed, with how to regenerate them, in `fixtures/VENDOR_FIXTURES.toml`. Returns true (and
/// says why on stderr) when a fixture the test needs is listed there and one of its files is
/// missing, so the test can skip: `if skip_missing_vendor_files(&fixtures, &["x"]) { return; }`.
/// Each item is a fixture name, or a path (relative to the directory holding `fixtures`) to a
/// case.toml / equiv.toml / errdist spec, whose own fixture and every `../name` it references
/// (recursively) count as needed.
pub fn skip_missing_vendor_files(fixtures: &Path, needs: &[&str]) -> bool {
    #[derive(Deserialize)]
    struct Entry {
        name: String,
        files: Vec<String>,
    }
    #[derive(Deserialize)]
    struct VendorManifest {
        fixture: Vec<Entry>,
    }
    let Ok(text) = std::fs::read_to_string(fixtures.join("VENDOR_FIXTURES.toml")) else {
        return false;
    };
    let m: VendorManifest = match toml::from_str(&text) {
        Ok(m) => m,
        Err(e) => panic!("fixtures/VENDOR_FIXTURES.toml: {e}"),
    };
    // Fixture names needed, following `"../name` references of the given files.
    let root = fixtures.parent().unwrap_or(Path::new("."));
    let mut names: Vec<String> = Vec::new();
    let mut todo: Vec<PathBuf> = Vec::new();
    for n in needs {
        if n.ends_with(".toml") {
            todo.push(root.join(n));
        } else {
            names.push(n.to_string());
            todo.push(fixtures.join(n).join("case.toml"));
        }
    }
    let mut seen = std::collections::HashSet::new();
    while let Some(p) = todo.pop() {
        if !seen.insert(p.clone()) {
            continue;
        }
        if let Some(d) = p.parent().and_then(|d| d.file_name()) {
            names.push(d.to_string_lossy().into_owned());
        }
        let Ok(t) = std::fs::read_to_string(&p) else {
            continue;
        };
        for part in t.split("\"../").skip(1) {
            let name: String = part.chars().take_while(|c| !matches!(c, '/' | '"')).collect();
            if !name.is_empty() {
                todo.push(fixtures.join(&name).join("case.toml"));
            }
        }
    }
    let mut missing = Vec::new();
    for e in m.fixture.iter().filter(|e| names.contains(&e.name)) {
        for f in &e.files {
            if !fixtures.join(&e.name).join(f).is_file() {
                missing.push(format!("{}/{f}", e.name));
            }
        }
    }
    if missing.is_empty() {
        return false;
    }
    eprintln!(
        "SKIP: vendor fixture files not present: {} (regenerate: python3 tools/regen_fixture.py --missing)",
        missing.join(", ")
    );
    true
}
