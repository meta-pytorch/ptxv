/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Run a case on real GPUs through a minimal CUDA driver-API binding.
//!
//! Used for differential conformance (simulator vs GB300, bitwise) and for measuring how
//! often a bug shows up on hardware. Hangs are handled by the caller, which runs this in a
//! child process with a timeout (see `ptxv hw`).
//!
//! GPU policy: callers pick only GPUs without compute processes that ptxv did not start
//! (`free_gpus`).

#![allow(non_camel_case_types)]

use std::collections::BTreeMap;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;

use anyhow::Result;
use anyhow::bail;
use ptxv_case::Case;
use ptxv_case::ParamSpec;
use serde::Serialize;

type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;

#[repr(C)]
struct CUlaunchAttribute {
    id: u32,
    pad: [u8; 4],
    value: [u64; 8],
}

#[repr(C)]
struct CUlaunchConfig {
    grid: [u32; 3],
    block: [u32; 3],
    smem: u32,
    stream: *mut c_void,
    attrs: *mut CUlaunchAttribute,
    nattrs: u32,
}

/// CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
const ATTR_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;
const ATTR_CLUSTER_DIMENSION: u32 = 4;
/// CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION (cuda.h).
const ATTR_PREFERRED_CLUSTER_DIMENSION: u32 = 11;
/// CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH: nonzero when the cubin fixes the cluster shape.
const FUNC_REQUIRED_CLUSTER_WIDTH: i32 = 11;

#[repr(C, align(128))]
#[derive(Clone, Copy)]
pub struct CUtensorMap(pub [u64; 16]);

// The CUDA driver is loaded at the first hardware call (dlopen), not linked: the verifier
// itself runs on machines without a GPU driver (CPU cluster nodes).
macro_rules! cuda_fns {
    ($(fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> CUresult;)*) => {
        #[allow(non_snake_case)]
        struct Cuda {
            $($name: unsafe extern "C" fn($($ty),*) -> CUresult,)*
        }
        fn cuda() -> &'static Cuda {
            static C: std::sync::OnceLock<Cuda> = std::sync::OnceLock::new();
            C.get_or_init(|| unsafe {
                extern "C" {
                    fn dlopen(name: *const c_char, flags: i32) -> *mut c_void;
                    fn dlsym(h: *mut c_void, name: *const c_char) -> *mut c_void;
                }
                const RTLD_NOW: i32 = 2;
                let h = dlopen(c"libcuda.so.1".as_ptr(), RTLD_NOW);
                assert!(!h.is_null(), "libcuda.so.1 not found: no GPU driver on this machine");
                Cuda {
                    $($name: {
                        let f = dlsym(h, concat!(stringify!($name), "\0").as_ptr() as *const c_char);
                        assert!(!f.is_null(), concat!("libcuda has no ", stringify!($name)));
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($ty),*) -> CUresult>(f)
                    },)*
                }
            })
        }
        $(
            #[allow(non_snake_case, clippy::too_many_arguments)]
            unsafe fn $name($($arg: $ty),*) -> CUresult {
                unsafe { (cuda().$name)($($arg),*) }
            }
        )*
    };
}

cuda_fns! {
    fn cuInit(flags: u32) -> CUresult;
    fn cuDeviceGet(d: *mut CUdevice, ordinal: i32) -> CUresult;
    fn cuDevicePrimaryCtxRetain(ctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    fn cuModuleLoadData(m: *mut CUmodule, image: *const c_void) -> CUresult;
    fn cuModuleGetFunction(f: *mut CUfunction, m: CUmodule, name: *const c_char) -> CUresult;
    fn cuModuleGetGlobal_v2(
        p: *mut CUdeviceptr,
        bytes: *mut usize,
        m: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    fn cuMemAlloc_v2(p: *mut CUdeviceptr, bytes: usize) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, n: usize) -> CUresult;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, n: usize) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        smem: u32,
        stream: *mut c_void,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuLaunchKernelEx(
        cfg: *const CUlaunchConfig,
        f: CUfunction,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    fn cuCtxSynchronize() -> CUresult;
    fn cuCtxEnablePeerAccess(peer: CUcontext, flags: u32) -> CUresult;
    fn cuDeviceGetAttribute(v: *mut i32, attr: i32, dev: CUdevice) -> CUresult;
    fn cuFuncSetAttribute(f: CUfunction, attr: i32, v: i32) -> CUresult;
    fn cuFuncGetAttribute(v: *mut i32, attr: i32, f: CUfunction) -> CUresult;
    fn cuGetErrorString(e: CUresult, s: *mut *const c_char) -> CUresult;
    fn cuTensorMapEncodeTiled(
        tm: *mut CUtensorMap,
        dtype: u32,
        rank: u32,
        addr: *mut c_void,
        dims: *const u64,
        strides: *const u64,
        boxd: *const u32,
        estrides: *const u32,
        interleave: u32,
        swizzle: u32,
        l2: u32,
        oob: u32,
    ) -> CUresult;
}

fn ck(r: CUresult, what: &str) -> Result<()> {
    if r == 0 {
        return Ok(());
    }
    let mut s: *const c_char = std::ptr::null();
    unsafe { cuGetErrorString(r, &mut s) };
    let msg = if s.is_null() {
        format!("error {r}")
    } else {
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    };
    bail!("{what}: {msg}")
}

fn tmap_dtype(d: &str) -> Result<u32> {
    Ok(match d {
        "u8" | "e4m3" => 0,
        "u16" => 1,
        "u32" => 2,
        "i32" => 3,
        "f16" => 6,
        "f32" => 7,
        "bf16" => 9,
        _ => bail!("no tensor-map dtype for {d}"),
    })
}

#[derive(Debug, Serialize)]
pub struct HwResult {
    pub gpu_visible: String,
    pub runs: usize,
    /// Runs whose output differs from `expected` (if given).
    pub mismatches: usize,
    /// Distinct outputs observed: (count, number of differing f32 words vs expected).
    pub distinct: Vec<(usize, usize)>,
    /// Indices of the first (up to 20) mismatching runs.
    pub mismatch_runs: Vec<usize>,
    pub scrubbed: bool,
    #[serde(skip)]
    pub first_output: Vec<u8>,
}

/// Launch `image` (cubin bytes or NUL-terminated PTX) `runs` times. Must run in a process
/// whose CUDA_VISIBLE_DEVICES selects one free GPU.
/// Fills every SM's shared memory with a NaN pattern, so a kernel that reads shared memory
/// before writing it sees poison instead of the previous launch's data.
const SCRUB_PTX: &str = r#"
.version 8.0
.target sm_90
.address_size 64
.extern .shared .align 16 .b8 dsmem[];
.visible .entry ptxv_scrub(.param .u32 nbytes)
{
    .reg .b32 %r<8>;
    .reg .pred %p<2>;
    ld.param.u32 %r0, [nbytes];
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    mov.u32 %r3, dsmem;
    mov.u32 %r5, 0x7fc17fc1;
L:
    setp.ge.u32 %p0, %r2, %r0;
    @%p0 bra DONE;
    add.u32 %r4, %r3, %r2;
    st.shared.u32 [%r4], %r5;
    add.u32 %r2, %r2, 4096;
    bra L;
DONE:
    ret;
}
"#;

pub fn run(
    case: &Case,
    image: &[u8],
    runs: usize,
    expected: Option<&[u8]>,
    scrub: bool,
) -> Result<HwResult> {
    run_with(case, image, runs, expected, scrub, &BTreeMap::new())
}

/// Like `run`, with some buffers' initial contents replaced (calibration inputs).
pub fn run_with(
    case: &Case,
    image: &[u8],
    runs: usize,
    expected: Option<&[u8]>,
    scrub: bool,
    inputs: &BTreeMap<String, Vec<u8>>,
) -> Result<HwResult> {
    let m = &case.m;
    unsafe {
        ck(cuInit(0), "cuInit")?;
        // One context per GPU of the case (device i = the i-th visible GPU), with peer access
        // between all of them (rung 1f: NVLink peer memory).
        let mut devs = Vec::new();
        let mut ctxs = Vec::new();
        for i in 0..m.ndevices() {
            let mut d = 0;
            ck(cuDeviceGet(&mut d, i as i32), "cuDeviceGet")?;
            let mut c = std::ptr::null_mut();
            ck(
                cuDevicePrimaryCtxRetain(&mut c, d),
                "cuDevicePrimaryCtxRetain",
            )?;
            devs.push(d);
            ctxs.push(c);
        }
        for (i, &c) in ctxs.iter().enumerate() {
            ck(cuCtxSetCurrent(c), "cuCtxSetCurrent")?;
            for (j, &peer) in ctxs.iter().enumerate() {
                let r = if i == j {
                    0
                } else {
                    cuCtxEnablePeerAccess(peer, 0)
                };
                if r != 0 && r != 704 {
                    // 704: CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                    ck(r, "cuCtxEnablePeerAccess")?;
                }
            }
        }
        let dev = devs[0];
        let set = |d: u32| ck(cuCtxSetCurrent(ctxs[d as usize]), "cuCtxSetCurrent");
        let mut modules = Vec::new();
        for &c in &ctxs {
            ck(cuCtxSetCurrent(c), "cuCtxSetCurrent")?;
            let mut module = std::ptr::null_mut();
            ck(
                cuModuleLoadData(&mut module, image.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
            modules.push(module);
        }
        let launches = m.all_launches();
        let mut fs = Vec::new();
        let mut cluster_attr = Vec::new();
        // Launches with their own cubin (kernels from several library modules): one module
        // per (cubin, device), kept alive with the others.
        let mut own: BTreeMap<(String, u32), CUmodule> = BTreeMap::new();
        // Module data globals each launch's constant banks name (S-GLOBAL-INIT):
        // (module key (None: the case's), device, global).
        let main_cubin = m.cubin.as_deref();
        let mut referenced: Vec<(Option<String>, u32, ptxv_sass::elf::DataSym)> = Vec::new();
        for l in &launches {
            set(l.device)?;
            let mut module = modules[l.device as usize];
            let key = l.cubin.filter(|c| Some(*c) != main_cubin);
            let globals = match key {
                None => ptxv_sass::elf::Elf::parse(image)
                    .and_then(|e| e.kernel_globals(l.entry))
                    .unwrap_or_default(),
                Some(c) => ptxv_sass::elf::Elf::parse(&std::fs::read(case.dir.join(c))?)
                    .and_then(|e| e.kernel_globals(l.entry))
                    .unwrap_or_default(),
            };
            for g in globals {
                let k = (key.map(String::from), l.device);
                if !referenced
                    .iter()
                    .any(|(c, d, x)| (c.clone(), *d) == k && x.name == g.name)
                {
                    referenced.push((k.0, k.1, g));
                }
            }
            if let Some(c) = l.cubin {
                let key = (c.to_string(), l.device);
                if !own.contains_key(&key) {
                    // A mutant's patched copy (`{variant}.{cubin}`, cli `load_with`).
                    let patched = std::env::var("PTXV_PATCHED_PREFIX")
                        .ok()
                        .map(|p| std::path::PathBuf::from(format!("{p}.{c}")))
                        .filter(|p| p.exists());
                    let img = std::fs::read(patched.unwrap_or_else(|| case.dir.join(c)))?;
                    let mut md = std::ptr::null_mut();
                    ck(
                        cuModuleLoadData(&mut md, img.as_ptr() as *const c_void),
                        "cuModuleLoadData",
                    )?;
                    own.insert(key.clone(), md);
                }
                module = own[&key];
            }
            let mut f = std::ptr::null_mut();
            let name = CString::new(l.entry)?;
            ck(
                cuModuleGetFunction(&mut f, module, name.as_ptr()),
                "cuModuleGetFunction",
            )?;
            if l.dynamic_smem > 48 * 1024 {
                // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                ck(
                    cuFuncSetAttribute(f, 8, l.dynamic_smem as i32),
                    "dynamic smem",
                )?;
            }
            // A cluster shape the cubin does not fix (cuBLAS: set at launch) goes in a launch
            // attribute.
            let mut req = 0;
            ck(
                cuFuncGetAttribute(&mut req, FUNC_REQUIRED_CLUSTER_WIDTH, f),
                "cluster attribute",
            )?;
            cluster_attr.push(req == 0 && l.cluster.is_some_and(|c| c != [1, 1, 1]));
            fs.push(f);
        }

        if !m.multicast.is_empty() {
            bail!(
                "case has NVLS multicast objects ({}): not supported by the hardware runner",
                m.multicast.len()
            );
        }
        // The module holding a global: the case's (per device) or a launch's own cubin.
        let module_of = |c: Option<&str>, d: u32| -> Result<CUmodule> {
            match c.filter(|c| Some(*c) != main_cubin) {
                None => Ok(modules[d as usize]),
                Some(c) => own.get(&(c.to_string(), d)).copied().ok_or_else(|| {
                    anyhow::anyhow!("global in cubin {c} on GPU {d}: no launch loads that module")
                }),
            }
        };
        let global_ptr = |md: CUmodule, name: &str| -> Result<(CUdeviceptr, usize)> {
            let (mut p, mut n) = (0, 0);
            let cname = CString::new(name)?;
            ck(
                cuModuleGetGlobal_v2(&mut p, &mut n, md, cname.as_ptr()),
                &format!("cuModuleGetGlobal({name})"),
            )?;
            Ok((p, n))
        };
        let mut dptr: BTreeMap<String, (CUdeviceptr, Vec<u8>)> = BTreeMap::new();
        for b in &m.buffers {
            let init = match inputs.get(&b.name) {
                Some(v) => {
                    if v.len() != b.bytes() {
                        bail!(
                            "input override for {} has {} bytes, expected {}",
                            b.name,
                            v.len(),
                            b.bytes()
                        );
                    }
                    v.clone()
                }
                None => b.init_bytes(),
            };
            let mut p = 0;
            set(b.device)?;
            if let Some(sym) = &b.symbol {
                // The module's own global (S-GLOBAL-INIT), rewritten with `init` every run.
                let (gp, n) = global_ptr(module_of(b.cubin.as_deref(), b.device)?, sym)?;
                if init.len() > n {
                    bail!(
                        "buffer {} ({} bytes) backs global {sym} of {n} bytes",
                        b.name,
                        init.len()
                    );
                }
                p = gp;
            } else {
                ck(cuMemAlloc_v2(&mut p, init.len().max(1)), "cuMemAlloc")?;
            }
            ck(
                cuMemcpyHtoD_v2(p, init.as_ptr() as *const c_void, init.len()),
                "HtoD",
            )?;
            dptr.insert(b.name.clone(), (p, init));
        }
        // Globals the case does not declare start every run from their ELF initial value, as
        // in the simulator (S-GLOBAL-INIT), not from what the previous run left.
        for (c, d, g) in &referenced {
            if m.global_buffer(c.as_deref(), &g.name, *d).is_some() {
                continue;
            }
            set(*d)?;
            let (p, _) = global_ptr(module_of(c.as_deref(), *d)?, &g.name)?;
            let init = g.init.clone().unwrap_or_else(|| vec![0; g.size as usize]);
            let mut name = format!("global:{}", g.name);
            if let Some(c) = c {
                name += &format!("[{c}]");
            }
            if *d != 0 {
                name += &format!("@gpu{d}");
            }
            dptr.insert(name, (p, init));
        }
        // Pointer tables (`ptrs`): filled in with the device addresses just assigned.
        for b in m.buffers.iter().filter(|b| !b.ptrs.is_empty()) {
            let addr = |n: &str| {
                dptr.get(n)
                    .map(|x| x.0)
                    .ok_or_else(|| anyhow::anyhow!("no buffer {n}"))
            };
            let mut init = dptr[&b.name].1.clone();
            for q in &b.ptrs {
                let o = q.offset as usize;
                init[o..o + 8].copy_from_slice(&(addr(&q.buffer)? + q.byte_offset).to_le_bytes());
            }
            let p = dptr[&b.name].0;
            set(b.device)?;
            ck(
                cuMemcpyHtoD_v2(p, init.as_ptr() as *const c_void, init.len()),
                "HtoD",
            )?;
            dptr.insert(b.name.clone(), (p, init));
        }
        // Parameter storage must outlive the launch; Box keeps addresses stable.
        let mut storage: Vec<Box<dyn std::any::Any>> = Vec::new();
        let mut all_args: Vec<Vec<*mut c_void>> = Vec::new();
        for l in &launches {
            let mut args: Vec<*mut c_void> = Vec::new();
            for spec in l.params {
                match spec {
                    ParamSpec::Ptr { buffer, offset } => {
                        let b = Box::new(dptr[buffer].0 + offset);
                        args.push(&*b as *const u64 as *mut c_void);
                        storage.push(b);
                    }
                    ParamSpec::U32 { value } => {
                        let b = Box::new(*value);
                        args.push(&*b as *const u32 as *mut c_void);
                        storage.push(b);
                    }
                    ParamSpec::U64 { value } => {
                        let b = Box::new(*value);
                        args.push(&*b as *const u64 as *mut c_void);
                        storage.push(b);
                    }
                    ParamSpec::Bytes { hex, fields } => {
                        // u64 backing keeps the bytes 8-aligned.
                        let mut raw = ptxv_case::param_bytes(hex)?;
                        for f in fields {
                            let o = f.offset as usize;
                            match &f.spec {
                                ParamSpec::Ptr { buffer, offset } => raw[o..o + 8]
                                    .copy_from_slice(&(dptr[buffer].0 + offset).to_le_bytes()),
                                ParamSpec::Tensormap {
                                    buffer,
                                    raw: Some(r),
                                    offset,
                                    ..
                                } => {
                                    let b = ptxv_case::param_bytes(r)?;
                                    raw[o..o + b.len()].copy_from_slice(&b);
                                    raw[o..o + 8]
                                        .copy_from_slice(&(dptr[buffer].0 + offset).to_le_bytes());
                                }
                                _ => bail!("bytes fields: ptr, or tensormap with raw bytes"),
                            }
                        }
                        let mut b = vec![0u64; raw.len().div_ceil(8)].into_boxed_slice();
                        for (i, x) in raw.iter().enumerate() {
                            b[i / 8] |= (*x as u64) << (8 * (i % 8));
                        }
                        args.push(b.as_ptr() as *mut c_void);
                        storage.push(Box::new(b));
                    }
                    ParamSpec::Tensormap {
                        buffer,
                        r#box,
                        dims,
                        strides,
                        swizzle,
                        raw,
                        offset,
                        elem4,
                    } => {
                        if *elem4 && raw.is_none() {
                            bail!("a 4-bit tensor map (elem4) needs its raw bytes");
                        }
                        let swz = ptxv_case::swizzle_of(swizzle)?.1;
                        let buf = case.buffer(buffer)?;
                        let (dims, strides) = ptxv_case::tmap_geometry(buf, dims, strides);
                        let estr = vec![1u32; dims.len()];
                        let mut tm = Box::new(CUtensorMap([0; 16]));
                        if let Some(raw) = raw {
                            let b = ptxv_case::param_bytes(raw)?;
                            if b.len() > 128 || b.len() < 8 {
                                bail!("raw tensor map must be 8..128 bytes");
                            }
                            for (i, x) in b.iter().enumerate() {
                                tm.0[i / 8] |= (*x as u64) << (8 * (i % 8));
                            }
                            tm.0[0] = dptr[buffer].0 + offset;
                            args.push(&*tm as *const CUtensorMap as *mut c_void);
                            storage.push(tm);
                            continue;
                        }
                        ck(
                            cuTensorMapEncodeTiled(
                                &mut *tm,
                                tmap_dtype(&buf.dtype)?,
                                dims.len() as u32,
                                (dptr[buffer].0 + offset) as *mut c_void,
                                dims.as_ptr(),
                                strides.as_ptr(),
                                r#box.as_ptr(),
                                estr.as_ptr(),
                                0,
                                swz,
                                0,
                                0,
                            ),
                            "cuTensorMapEncodeTiled",
                        )?;
                        args.push(&*tm as *const CUtensorMap as *mut c_void);
                        storage.push(tm);
                    }
                }
            }
            all_args.push(args);
        }

        let mut res = HwResult {
            gpu_visible: std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
            runs: 0,
            mismatches: 0,
            distinct: vec![],
            mismatch_runs: vec![],
            scrubbed: scrub,
            first_output: vec![],
        };
        // Scrubber: one 1024-thread CTA per SM slot with the maximum dynamic smem.
        let mut scrub_f: CUfunction = std::ptr::null_mut();
        let (mut nsm, mut smem_max) = (0i32, 0i32);
        set(0)?; // the scrubber runs on GPU 0 only
        if scrub {
            let src = CString::new(SCRUB_PTX)?;
            let mut sm = std::ptr::null_mut();
            ck(
                cuModuleLoadData(&mut sm, src.as_ptr() as *const c_void),
                "load scrubber",
            )?;
            let n = CString::new("ptxv_scrub")?;
            ck(
                cuModuleGetFunction(&mut scrub_f, sm, n.as_ptr()),
                "scrubber fn",
            )?;
            ck(cuDeviceGetAttribute(&mut nsm, 16, dev), "SM count")?; // MULTIPROCESSOR_COUNT
            ck(cuDeviceGetAttribute(&mut smem_max, 97, dev), "smem optin")?; // MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
            ck(cuFuncSetAttribute(scrub_f, 8, smem_max), "scrubber smem")?; // MAX_DYNAMIC_SHARED_SIZE_BYTES
        }
        let mut seen: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
        for _ in 0..runs {
            // Every buffer, not just the outputs: a kernel may write its inputs (r1e resets
            // one for the next step).
            for (p, init) in dptr.values() {
                ck(
                    cuMemcpyHtoD_v2(*p, init.as_ptr() as *const c_void, init.len()),
                    "reset buffer",
                )?;
            }
            if scrub {
                set(0)?;
                let mut nb = smem_max as u32;
                let mut a = [&mut nb as *mut u32 as *mut c_void];
                ck(
                    cuLaunchKernel(
                        scrub_f,
                        nsm as u32 * 2,
                        1,
                        1,
                        1024,
                        1,
                        1,
                        smem_max as u32,
                        std::ptr::null_mut(),
                        a.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                    "scrub",
                )?;
            }
            // Every kernel of the case, in order on the default stream; a PDL launch carries
            // the programmatic-stream-serialization attribute (E-PDL).
            for (i, l) in launches.iter().enumerate() {
                set(l.device)?;
                let mut attrs = Vec::new();
                if l.pdl {
                    attrs.push(CUlaunchAttribute {
                        id: ATTR_PROGRAMMATIC_STREAM_SERIALIZATION,
                        pad: [0; 4],
                        value: [1, 0, 0, 0, 0, 0, 0, 0],
                    });
                }
                if cluster_attr[i] {
                    let [x, y, z] = l.cluster.unwrap();
                    attrs.push(CUlaunchAttribute {
                        id: ATTR_CLUSTER_DIMENSION,
                        pad: [0; 4],
                        value: [x as u64 | (y as u64) << 32, z as u64, 0, 0, 0, 0, 0, 0],
                    });
                    // The captured launch's preferred cluster dimension (S-CGA-SHAPE).
                    if let Some([x, y, z]) = l.preferred_cluster {
                        attrs.push(CUlaunchAttribute {
                            id: ATTR_PREFERRED_CLUSTER_DIMENSION,
                            pad: [0; 4],
                            value: [x as u64 | (y as u64) << 32, z as u64, 0, 0, 0, 0, 0, 0],
                        });
                    }
                }
                let cfg = CUlaunchConfig {
                    grid: l.grid,
                    block: l.block,
                    smem: l.dynamic_smem,
                    stream: std::ptr::null_mut(),
                    attrs: attrs.as_mut_ptr(),
                    nattrs: attrs.len() as u32,
                };
                ck(
                    cuLaunchKernelEx(&cfg, fs[i], all_args[i].as_mut_ptr(), std::ptr::null_mut()),
                    "cuLaunchKernelEx",
                )?;
            }
            for d in (0..ctxs.len() as u32).rev() {
                set(d)?;
                ck(cuCtxSynchronize(), "cuCtxSynchronize")?;
            }
            let mut out = Vec::new();
            for o in &m.outputs {
                let (p, init) = &dptr[o];
                let mut host = vec![0u8; init.len()];
                ck(
                    cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut c_void, *p, host.len()),
                    "DtoH",
                )?;
                out.extend(host);
            }
            if res.runs == 0 {
                res.first_output = out.clone();
            }
            res.runs += 1;
            if let Some(e) = expected {
                if e != out.as_slice() {
                    if res.mismatch_runs.len() < 20 {
                        res.mismatch_runs.push(res.runs - 1);
                    }
                    res.mismatches += 1;
                }
            }
            *seen.entry(out).or_default() += 1;
        }
        // PTXV_HW_DUMP=<dir>: every distinct output to <dir>/hw_<FNV-1a of the bytes>.bin
        // (witness replays: compare with a simulated schedule's output).
        let dump = std::env::var_os("PTXV_HW_DUMP").map(std::path::PathBuf::from);
        for (o, c) in seen {
            if let Some(d) = &dump {
                let h = o.iter().fold(0xcbf29ce484222325u64, |h, x| {
                    (h ^ *x as u64).wrapping_mul(0x100000001b3)
                });
                std::fs::create_dir_all(d)
                    .and_then(|_| std::fs::write(d.join(format!("hw_{h:016x}.bin")), &o))
                    .map_err(|e| anyhow::anyhow!("PTXV_HW_DUMP: {e}"))?;
                eprintln!("hw output {h:016x}: {c} runs");
            }
            let diff = match expected {
                Some(e) => o.chunks(4).zip(e.chunks(4)).filter(|(a, b)| a != b).count(),
                None => 0,
            };
            res.distinct.push((c, diff));
        }
        res.distinct.sort_by_key(|d| std::cmp::Reverse(d.0));
        drop(storage);
        Ok(res)
    }
}

/// GPUs with no compute processes on them right now.
/// This process's ancestors (parent, grandparent, ...), from /proc.
fn ancestor_pids() -> Vec<u32> {
    let mut out = Vec::new();
    let mut pid = std::process::id();
    while pid > 1 && out.len() < 64 {
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
            break;
        };
        // Field 4 (ppid) follows the parenthesized command name, which may contain spaces.
        let Some(ppid) = stat
            .rsplit_once(')')
            .and_then(|(_, rest)| rest.split_whitespace().nth(1))
            .and_then(|x| x.parse().ok())
        else {
            break;
        };
        out.push(ppid);
        pid = ppid;
    }
    out
}

pub fn free_gpus() -> Result<Vec<u32>> {
    let q = |args: &[&str]| -> Result<String> {
        let o = std::process::Command::new("nvidia-smi")
            .args(args)
            .output()?;
        Ok(String::from_utf8(o.stdout)?)
    };
    let gpus = q(&["--query-gpu=index,uuid", "--format=csv,noheader"])?;
    let busy = q(&["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])?;
    // A GPU used only by one of our ancestors (e.g. the Python program that captured the
    // kernels and called ptxv) counts as free: that context is ours, not a foreign job.
    let ours = ancestor_pids();
    let busy: Vec<&str> = busy
        .lines()
        .filter_map(|l| {
            let (u, pid) = l.split_once(',')?;
            let pid: u32 = pid.trim().parse().ok()?;
            (!ours.contains(&pid)).then_some(u.trim())
        })
        .collect();
    let all: Vec<(u32, String)> = gpus
        .lines()
        .filter_map(|l| {
            let (i, u) = l.split_once(',')?;
            Some((i.trim().parse().ok()?, u.trim().to_string()))
        })
        .collect();
    let allowed = allowed_gpus(
        std::env::var("PTXV_GPUS").ok().as_deref(),
        std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref(),
        &all,
    );
    Ok(all
        .iter()
        .filter(|(_, u)| !busy.contains(&u.as_str()))
        .map(|(i, _)| *i)
        .filter(|i| allowed.as_ref().is_none_or(|a| a.contains(i)))
        .collect())
}

/// The GPUs (nvidia-smi indices) `ptxv` may pick, from `PTXV_GPUS` (`3` or `0,3`: indices),
/// else from `CUDA_VISIBLE_DEVICES` read as the CUDA runtime reads it: a comma-separated list
/// of indices or `GPU-` UUID prefixes, ending at the first entry that names no device (an
/// empty or all-invalid list allows none). The worker is launched with
/// `CUDA_VISIBLE_DEVICES` set to the chosen index, so the caller's setting must restrict the
/// choice, not be overridden by it. None: no restriction. Index order is nvidia-smi's (PCI
/// bus), which is the runtime's on a node of identical GPUs.
pub fn allowed_gpus(
    ptxv_gpus: Option<&str>,
    cuda_visible: Option<&str>,
    all: &[(u32, String)],
) -> Option<Vec<u32>> {
    if let Some(v) = ptxv_gpus {
        return Some(v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    }
    let v = cuda_visible?;
    let mut out = Vec::new();
    for e in v.split(',').map(str::trim) {
        let hit = if let Ok(i) = e.parse::<u32>() {
            all.iter().find(|(j, _)| *j == i)
        } else if e.starts_with("GPU-") {
            let m: Vec<_> = all.iter().filter(|(_, u)| u.starts_with(e)).collect();
            if m.len() == 1 { Some(m[0]) } else { None }
        } else {
            None
        };
        match hit {
            Some((i, _)) if !out.contains(i) => out.push(*i),
            Some(_) => {}
            None => break,
        }
    }
    Some(out)
}

#[cfg(test)]
mod gpu_env_tests {
    use super::allowed_gpus;

    #[test]
    fn cuda_visible_devices_restricts_the_choice() {
        let all: Vec<(u32, String)> = (0..4).map(|i| (i, format!("GPU-{i}a{i}b-0000"))).collect();
        // PTXV_GPUS wins.
        assert_eq!(allowed_gpus(Some("3"), Some("2"), &all), Some(vec![3]));
        // Unset both: no restriction.
        assert_eq!(allowed_gpus(None, None, &all), None);
        // Indices, UUID prefixes, stop at the first invalid entry.
        assert_eq!(allowed_gpus(None, Some("3"), &all), Some(vec![3]));
        assert_eq!(allowed_gpus(None, Some("3,1"), &all), Some(vec![3, 1]));
        assert_eq!(allowed_gpus(None, Some("GPU-2a2"), &all), Some(vec![2]));
        assert_eq!(allowed_gpus(None, Some("1,7,3"), &all), Some(vec![1]));
        assert_eq!(allowed_gpus(None, Some("GPU-"), &all), Some(vec![]));
        assert_eq!(allowed_gpus(None, Some(""), &all), Some(vec![]));
        assert_eq!(allowed_gpus(None, Some("-1"), &all), Some(vec![]));
    }
}
