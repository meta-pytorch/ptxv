/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Numerics of packed f32x2 values in 64-bit PTX registers: each half is a lane with its own
//! node (`num::lane_reg`), through ld.v2.b64, mov.b64 {lo, hi} packing and unpacking, a plain
//! 64-bit mov, fma.rn.f32x2 in the tracked x*1 + 0 form, and st.b64.

use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::exec::Loc;
use ptxv_sim::machine::Machine;

const PTX: &str = r#"
.version 9.0
.target sm_103a
.address_size 64
.visible .entry lanes(.param .u64 x_p, .param .u64 out_p)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;
    ld.param.u64 %rd1, [x_p];
    ld.param.u64 %rd2, [out_p];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    ld.global.v2.b64 {%rd3, %rd4}, [%rd1];
    mov.b32 %r1, 0f3F800000;
    mov.b32 %r2, 0f00000000;
    mov.b64 %rd5, {%r1, %r1};
    mov.b64 %rd6, {%r2, %r2};
    fma.rn.f32x2 %rd7, %rd3, %rd5, %rd6;
    mov.b64 {%r3, %r4}, %rd7;
    st.global.b32 [%rd2], %r4;
    st.global.b32 [%rd2+4], %r3;
    mov.b64 %rd8, %rd4;
    st.global.b64 [%rd2+8], %rd8;
    mov.b64 {_, %r5}, %rd4;
    mov.b64 %rd9, {%r5, %r3};
    st.global.b64 [%rd2+16], %rd9;
    ret;
}
"#;

const CASE: &str = r#"
name = "lanes"
description = "f32x2 lanes of 64-bit PTX registers in the numerics"
ptx = "kernel.ptx"
entry = "lanes"
grid = [1, 1, 1]
block = [32, 1, 1]
outputs = ["out"]

[[buffers]]
name = "x"
dtype = "f32"
shape = [4]
init = "zeros"
values = [[0, 1.5], [1, -2.0], [2, 3.1415927410125732], [3, -0.75]]

[[buffers]]
name = "out"
dtype = "f32"
shape = [6]
init = "zeros"

[[params]]
kind = "ptr"
buffer = "x"

[[params]]
kind = "ptr"
buffer = "out"

[[variants]]
id = "good"
expect = ["clean"]
"#;

#[test]
fn f32x2_lanes_keep_their_nodes() {
    let dir = std::env::temp_dir().join(format!("ptxv_lanes_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("kernel.ptx"), PTX).unwrap();
    std::fs::write(dir.join("case.toml"), CASE).unwrap();
    let case = Case::load(&dir).unwrap();
    let k = ptxv_ptx::parse_kernel(PTX, Some("lanes")).unwrap();
    let launch = build_launch(&case, &k).unwrap();
    let mut m = Machine::new(Arc::new(k), Arc::new(launch));
    let mut ns = ptxv_sim::num::NumState::with_mma_eps(m.launch.mma_model.eps());
    let (xi, oi) = (0, m.launch.outputs[0]);
    ns.f32_inputs.insert(xi);
    ns.taint.taint_alloc(xi, 16);
    ns.not_inputs.insert(oi);
    m.num = Some(Box::new(ns));
    let (m, _, ok) = ptxv_sim::explore::run_once(m);
    assert!(ok);
    let ns = m.num.as_ref().unwrap();
    let x = [1.5f32, -2.0, std::f32::consts::PI, -0.75];
    let out = |w: u64| {
        let n = ns
            .mem_node(Loc::Global {
                alloc: oi,
                off: 4 * w,
            })
            .unwrap_or_else(|| panic!("out[{w}] has no node"));
        let nd = &ns.nodes[n as usize];
        (f32::from_bits(nd.hw), ptxv_sim::exact::approx(&nd.exact))
    };
    // out = {x1, x0} (fma lanes, swapped by the unpack), {x2, x3} (mov), {x3, x0} (repack).
    for (w, want) in [x[1], x[0], x[2], x[3], x[3], x[0]].into_iter().enumerate() {
        let (hw, ex) = out(w as u64);
        assert_eq!(hw, want, "out[{w}] value");
        assert_eq!(ex, want as f64, "out[{w}] exact value");
        assert!(
            ns.taint
                .memory_unit_input_derived((1, oi as u64, 4 * w as u64))
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
