/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! E-SPIN poll-loop shapes (spin::poll_loops) on both front-ends: loops that are polls must be
//! accepted (a refusal makes the model step every iteration), and loops whose failed
//! iterations leave state behind must be refused.

use ptxv_ir::Kernel;

const HDR: &str = ".version 9.0
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 p, .param .u32 n)
{
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .pred %p<8>;
    ld.param.u64 %rd1, [p];
    cvta.to.global.u64 %rd1, %rd1;
    ld.param.u32 %r10, [n];
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd2, %r1, 4;
    add.u64 %rd3, %rd1, %rd2;
";
const FTR: &str = "    st.global.u32 [%rd3+128], %r1;
    ret;
}
";

fn kernels(name: &str, body: &str) -> Vec<(&'static str, Kernel)> {
    let src = format!("{HDR}{body}\n{FTR}");
    let dir = std::env::temp_dir().join(format!("ptxv-spin-shapes-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("k.ptx");
    std::fs::write(&p, &src).unwrap();
    let c = dir.join("k.cubin");
    ptxv_sass::ptxas(&p, &c, "sm_103a").unwrap();
    vec![
        ("ptx", ptxv_ptx::parse_kernel(&src, Some("k")).unwrap()),
        ("sass", ptxv_sass::load_cubin(&c, "k").unwrap()),
    ]
}

type Shape = Result<(usize, usize), String>;

fn shapes(name: &str, body: &str) -> Vec<(&'static str, Shape)> {
    kernels(name, body)
        .into_iter()
        .map(|(f, k)| {
            let v = ptxv_sim::spin::poll_loops(&k.code, k.front);
            assert_eq!(v.len(), 1, "{name}/{f}: {v:?}");
            (f, v.into_iter().next().unwrap().1)
        })
        .collect()
}

#[test]
fn polls_are_accepted() {
    for (name, body) in [
        (
            "basic",
            "L: ld.acquire.gpu.global.u32 %r5, [%rd1];
    setp.ne.u32 %p1, %r5, 1;
    @%p1 bra L;",
        ),
        // The signal-pad channel bug's scratch case: ptxas re-materializes the poll address at the
        // loop head (LDC.64 R2; LDG.E.STRONG.SYS R2, [R2.64+0x14]): invariant, not carried.
        (
            "remat",
            "    add.u64 %rd4, %rd1, 20;
L: ld.acquire.sys.global.u32 %r5, [%rd4];
    setp.lt.s32 %p1, %r5, 1;
    @%p1 bra L;",
        ),
        (
            "bit",
            "L: ld.acquire.gpu.global.u32 %r5, [%rd1];
    and.b32 %r6, %r5, 4;
    setp.eq.u32 %p1, %r6, 0;
    @%p1 bra L;",
        ),
        (
            "u64",
            "L: ld.acquire.gpu.global.u64 %rd5, [%rd1];
    cvt.u64.u32 %rd6, %r10;
    setp.lt.u64 %p1, %rd5, %rd6;
    @%p1 bra L;",
        ),
        (
            "invariant_head",
            "L: add.u32 %r9, %r1, 1;
    ld.acquire.gpu.global.u32 %r5, [%rd1];
    setp.lt.u32 %p1, %r5, %r9;
    @%p1 bra L;",
        ),
        // A head copy of the previous load that nothing reads after the loop.
        (
            "dead_copy",
            "    mov.u32 %r20, 0;
L: mov.u32 %r21, %r20;
    ld.acquire.gpu.global.u32 %r20, [%rd1];
    setp.ne.u32 %p1, %r20, 1;
    @%p1 bra L;",
        ),
    ] {
        for (f, r) in shapes(name, body) {
            assert!(r.is_ok(), "{name}/{f}: {r:?}");
        }
    }
}

#[test]
fn carried_state_is_refused() {
    for (name, body) in [
        // Waits for a change: compares with the previous iteration's load.
        (
            "change",
            "    ld.relaxed.gpu.global.u32 %r5, [%rd1];
L: mov.u32 %r7, %r5;
    ld.acquire.gpu.global.u32 %r5, [%rd1];
    setp.eq.u32 %p1, %r5, %r7;
    @%p1 bra L;",
        ),
        // An iteration counter read after the loop.
        (
            "counter",
            "    mov.u32 %r20, 0;
L: ld.acquire.gpu.global.u32 %r5, [%rd1];
    add.u32 %r20, %r20, 1;
    setp.eq.u32 %p1, %r5, 0;
    @%p1 bra L;
    add.u32 %r1, %r1, %r20;",
        ),
        // The head copies the previous iteration's load, which is read after the loop: the
        // model (which runs the head once) would leave the pre-loop value there. Accepted
        // before (a miss).
        (
            "live_copy",
            "    mov.u32 %r20, 0;
L: mov.u32 %r21, %r20;
    ld.acquire.gpu.global.u32 %r20, [%rd1];
    setp.ne.u32 %p1, %r20, 1;
    @%p1 bra L;
    add.u32 %r1, %r1, %r21;",
        ),
    ] {
        for (f, r) in shapes(name, body) {
            let e = r.expect_err(&format!("{name}/{f} accepted"));
            assert!(e.contains("carried"), "{name}/{f}: {e}");
        }
    }
}
