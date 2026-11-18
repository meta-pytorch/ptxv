/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::Addr;
use ptxv_ir::File;
use ptxv_ir::Front;
use ptxv_ir::Inst;
use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Space;
use ptxv_ir::Val;
use ptxv_memory_write::memory_safety::AccessKind;
use ptxv_memory_write::memory_safety::Instruction;
use ptxv_memory_write::memory_safety::Program;
use ptxv_memory_write::memory_safety::Request;
use serde::Serialize;

use crate::machine::Launch;
use crate::strict_memory_globals::specialize_code;
use crate::strict_memory_launch::aliases_present;
use crate::strict_memory_launch::dimensions_product;
use crate::strict_memory_launch::project_allocations;
use crate::strict_memory_launch::replicate_code;
use crate::strict_memory_launch::validate_outputs;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Refusal {
    Front,
    Kernel,
    Launch,
    Shape,
    Instruction(usize),
    Address(usize),
    Width(usize),
    Register(usize),
    Allocation(usize),
}

pub fn constant_address(address: &Addr) -> Option<u64> {
    let first = match address.base {
        None => 0,
        Some(Val::Imm(value)) => value,
        _ => return None,
    };
    let second = match address.base2 {
        None => 0,
        Some(Val::Imm(value)) => value,
        _ => return None,
    };
    Some(
        (address.off as u64)
            .wrapping_add(first)
            .wrapping_add(second),
    )
}

fn width_supported(bytes: u8) -> bool {
    matches!(bytes, 1 | 2 | 4 | 8)
}

pub fn project_instruction(
    instruction: &Inst,
    nregs_v: u16,
    pc: usize,
    last: bool,
) -> Result<Option<Instruction>, Refusal> {
    if !instruction.guard.is_always() || instruction.uniform {
        return Err(Refusal::Instruction(pc));
    }
    let access = match &instruction.op {
        Op::Ld {
            d,
            n,
            space: Space::Global,
            bytes,
            addr,
            sync: None,
            nc: false,
            ..
        } => {
            if *n != 1 || !width_supported(*bytes) {
                return Err(Refusal::Width(pc));
            }
            if d.file != File::V || d.idx >= nregs_v {
                return Err(Refusal::Register(pc));
            }
            let address = match constant_address(addr) {
                Some(address) => address,
                None => return Err(Refusal::Address(pc)),
            };
            Request {
                address,
                width: *bytes as u64,
                alignment: *bytes as u64,
                kind: AccessKind::Load,
            }
        }
        Op::St {
            space: Space::Global,
            bytes,
            addr,
            v: Val::Imm(_),
            sync: None,
        } => {
            if !width_supported(*bytes) {
                return Err(Refusal::Width(pc));
            }
            let address = match constant_address(addr) {
                Some(address) => address,
                None => return Err(Refusal::Address(pc)),
            };
            Request {
                address,
                width: *bytes as u64,
                alignment: *bytes as u64,
                kind: AccessKind::Store { defined: true },
            }
        }
        Op::Nop => return Ok(Some(Instruction::Skip)),
        Op::Exit if last => {
            return Ok(Some(Instruction::Halt));
        }
        _ => return Err(Refusal::Instruction(pc)),
    };
    Ok(Some(Instruction::Access(access)))
}

pub fn project_code(code: &[Inst], nregs_v: u16) -> Result<Vec<Instruction>, Refusal> {
    let length = code.len();
    if length == 0 || !matches!(code[length - 1].op, Op::Exit) {
        return Err(Refusal::Kernel);
    }
    let mut instructions = Vec::new();
    let mut failure = None;
    let mut pc = 0;
    while pc < length && failure.is_none() {
        match project_instruction(&code[pc], nregs_v, pc, pc + 1 == length) {
            Ok(Some(projected)) => instructions.push(projected),
            Ok(None) => {}
            Err(reason) => failure = Some(reason),
        }
        pc += 1;
    }
    match failure {
        Some(reason) => Err(reason),
        None => Ok(instructions),
    }
}

pub fn validate_launch(kernel: &Kernel, launch: &Launch) -> Result<usize, Refusal> {
    if kernel.front != Front::Ptx {
        return Err(Refusal::Front);
    }
    if kernel.params.len() != 0
        || kernel.parts.len() != 0
        || kernel.static_findings.len() != 0
        || kernel.bulk_read_waits.len() != 0
        || kernel.bulk_commit_sb.len() != 0
        || kernel.ldtm_sb.len() != 0
        || kernel.ldtm_waits.len() != 0
        || kernel.static_image.is_some()
        || kernel.cluster.is_some()
        || kernel.code.len() == 0
    {
        return Err(Refusal::Kernel);
    }
    let env = &launch.env;
    if launch.grids.len() != 1
        || launch.notes.len() != 0
        || launch.perm.len() != 0
        || launch.mcast.len() != 0
        || launch.mc_alias_hw
        || launch.views.len() != 0
        || launch.spaces.len() != 0
        || aliases_present(&launch.aliased)
        || launch.tensormaps.len() != 0
        || launch.replicas.len() != 0
        || launch.exit_fence.is_some()
        || launch.poll_acquire
        || launch.smem_cumulative
        || launch.hw_l1
        || launch.l1_cores
        || launch.l1_havoc
        || launch.read_done
        || launch.remote_fifo
        || launch.hw_bsync_coll
        || launch.hw_bsync_yield
        || launch.hw_sts_lowest
        || launch.hw_stg_lowest
        || launch.hw_split_ur
        || launch.hw_order
        || launch.tc_fence_implied
        || launch.async_mb_dest
        || launch.refuted_opt_in.len() != 0
        || launch.nic.is_some()
        || env.residency.is_some()
        || env.residency_device.is_some()
        || env.cooperative
        || env.tmem_cotenant
        || env.tmem_share.len() != 0
    {
        return Err(Refusal::Launch);
    }
    let grid = &launch.grids[0];
    if grid.cluster != [1, 1, 1]
        || grid.explicit_cluster
        || grid.preferred.is_some()
        || grid.cga_shape != 0
        || grid.params.len() != 0
        || grid.entry != 0
        || grid.cta0 != 0
        || grid.pdl
        || grid.device != 0
        || grid.after.is_some()
        || grid.waits.len() != 0
    {
        return Err(Refusal::Shape);
    }
    let ctas = match dimensions_product(grid.grid) {
        Some(count) => count,
        None => return Err(Refusal::Shape),
    };
    let block_threads = match dimensions_product(grid.block) {
        Some(count) => count,
        None => return Err(Refusal::Shape),
    };
    if ctas > 65_535 || block_threads > 1024 || grid.block[2] > 64 {
        return Err(Refusal::Shape);
    }
    let count = match ctas.checked_mul(block_threads) {
        Some(count) => count,
        None => return Err(Refusal::Shape),
    };
    if count > 65_536 {
        return Err(Refusal::Shape);
    }
    if let Some(ntid) = kernel.ntid {
        let limit = match dimensions_product(ntid.dims) {
            Some(limit) => limit,
            None => return Err(Refusal::Shape),
        };
        if limit > 1024
            || (ntid.req && ntid.dims != grid.block)
            || (!ntid.req && block_threads > limit)
        {
            return Err(Refusal::Shape);
        }
    }
    match validate_outputs(launch) {
        Ok(()) => {}
        Err(reason) => return Err(reason),
    }
    Ok(count)
}

pub fn project(kernel: &Kernel, launch: &Launch) -> Result<Program, Refusal> {
    let count = match validate_launch(kernel, launch) {
        Ok(count) => count,
        Err(reason) => return Err(reason),
    };
    let grid = &launch.grids[0];
    let allocations = match project_allocations(launch) {
        Ok(allocations) => allocations,
        Err(reason) => return Err(reason),
    };
    let specialized = match specialize_code(&kernel.code, &grid.cbank_relocs) {
        Ok(instructions) => instructions,
        Err(reason) => return Err(reason),
    };
    let instructions = match project_code(&specialized, kernel.nregs[File::V as usize]) {
        Ok(instructions) => instructions,
        Err(reason) => return Err(reason),
    };
    let threads = replicate_code(&instructions, count);
    Ok(Program {
        allocations,
        threads,
    })
}

pub fn verify_certificate(
    kernel: &Kernel,
    launch: &Launch,
    certificate: &[ptxv_memory_write::memory_safety::State],
) -> bool {
    match project(kernel, launch) {
        Ok(program) => ptxv_memory_write::memory_safety::verify(&program, certificate),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use ptxv_memory_write::memory_safety;
    use serde_json::json;

    use super::*;
    use crate::machine::Machine;

    fn input(body: &str, threads: u32, undefined: bool) -> (Kernel, Launch) {
        let text = format!(
            ".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry k() {{\n.reg .b64 %r<2>;\n{body}\nret;\n}}"
        );
        input_source(&text, threads, undefined)
    }

    fn input_source(text: &str, threads: u32, undefined: bool) -> (Kernel, Launch) {
        let kernel = ptxv_ptx::parse_kernel(text, Some("k")).unwrap();
        let case = ptxv_case::Case {
            dir: PathBuf::new(),
            m: serde_json::from_value(json!({
                "name":"strict-memory-admission", "description":"strict memory safety controls",
                "entry":"k", "grid":[1,1,1], "block":[threads,1,1], "outputs":[],
                "buffers":[{"name":"x","dtype":"u64","shape":[2],"init":"zeros"}],
                "params":[], "variants":[]
            }))
            .unwrap(),
        };
        let mut launch = crate::build_launch(&case, &kernel).unwrap();
        launch.allocs[0].base = 4096;
        launch.allocs[0].undef = undefined;
        launch.allocs[0].dead = undefined;
        (kernel, launch)
    }

    fn certificate(program: &Program) -> Option<Vec<memory_safety::State>> {
        let initial = memory_safety::initial(program)?;
        let mut states = vec![initial];
        let mut cursor = 0;
        while cursor < states.len() {
            let next = memory_safety::successors(program, &states[cursor])?;
            for state in next {
                if state.bad {
                    return None;
                }
                if !states.contains(&state) {
                    states.push(state);
                }
            }
            cursor += 1;
            assert!(
                states.len() < 1000,
                "small control exceeded its certificate budget"
            );
        }
        Some(states)
    }

    fn strict(program: &Program) -> bool {
        match certificate(program) {
            Some(states) => memory_safety::verify(program, &states),
            None => false,
        }
    }

    #[test]
    fn compiler_valid_static_global_program_uses_actual_launch_memory() {
        let text = ".version 8.8\n.target sm_103a\n.address_size 64\n.global .align 8 .u64 g[2] = {0,0};\n.visible .entry k() {\n.reg .b64 %r<2>;\nst.global.u64 [g],7;\nld.global.u64 %r0,[g+8];\nret;\n}";
        let (kernel, launch) = input_source(text, 1, false);
        assert!(!kernel.globals.is_empty());
        assert!(!kernel.global_relocs.is_empty());
        assert!(!launch.grids[0].cbank_relocs.is_empty());
        let program = project(&kernel, &launch).expect("static globals should be admitted");
        let states = certificate(&program).expect("defined static-global program is safe");
        assert!(verify_certificate(&kernel, &launch, &states));
        assert!(!verify_certificate(&kernel, &launch, &[]));
        let native =
            crate::explore::explore(Machine::new(Arc::new(kernel), Arc::new(launch)), 1000);
        assert!(native.complete);
        assert_eq!(native.classes(), ["clean"]);
    }

    #[test]
    fn parsed_global_programs_have_nonvacuous_strict_certificates() {
        for (body, threads) in [
            ("ld.global.u64 %r0,[4096];", 2),
            ("st.global.u64 [4096],7; ld.global.u64 %r0,[4096];", 1),
        ] {
            let (kernel, launch) = input(body, threads, false);
            let program = project(&kernel, &launch).unwrap();
            assert_eq!(program.threads.len(), threads as usize);
            assert!(strict(&program));
            let native = crate::explore::explore(
                Machine::new(Arc::new(kernel), Arc::new(launch)),
                1000,
            );
            assert!(native.complete);
            assert_eq!(native.classes(), ["clean"]);
        }
    }

    #[test]
    fn strict_certificate_rejects_dead_undefined_reads_and_same_value_races() {
        for (body, threads, undefined) in [
            ("ld.global.u64 %r0,[4096];", 1, true),
            ("st.global.u64 [4096],0;", 2, false),
        ] {
            let (kernel, launch) = input(body, threads, undefined);
            let program = project(&kernel, &launch).unwrap();
            assert!(!strict(&program));
            let native = crate::explore::explore(
                Machine::new(Arc::new(kernel), Arc::new(launch)),
                1000,
            );
            assert!(native.complete);
            assert_eq!(native.classes(), ["clean"]);
        }
    }

    #[test]
    fn strict_certificate_rejects_misalignment_and_out_of_bounds() {
        for body in ["ld.global.u64 %r0,[4097];", "ld.global.u64 %r0,[4112];"] {
            let (kernel, launch) = input(body, 1, false);
            assert!(!strict(&project(&kernel, &launch).unwrap()));
        }
    }

    #[test]
    fn unsupported_instructions_and_nonconstant_addresses_are_refused() {
        for body in [
            "atom.global.add.u32 %r0,[4096],1;",
            "ld.global.u64 %r0,[%r1];",
            "ld.global.acquire.gpu.u64 %r0,[4096];",
            "ld.global.nc.u64 %r0,[4096];",
            "mov.u64 %r0,4096; ld.global.u64 %r1,[%r0];",
        ] {
            let (kernel, launch) = input(body, 1, false);
            assert!(project(&kernel, &launch).is_err(), "{body}");
        }
    }

    #[test]
    fn guards_trailing_instructions_and_environment_features_are_refused() {
        let (kernel, launch) = input("ld.global.u64 %r0,[4096];", 1, false);
        let mut guarded = kernel.clone();
        guarded.code[0].guard.neg = true;
        assert!(project(&guarded, &launch).is_err());
        let mut trailing = kernel.clone();
        trailing.code.push(trailing.code[0].clone());
        assert!(project(&trailing, &launch).is_err());
        let mut external = launch.clone();
        external.l1_havoc = true;
        assert!(project(&kernel, &external).is_err());
        let mut alias = launch.clone();
        alias.aliased[0] = true;
        assert!(project(&kernel, &alias).is_err());
    }

    #[test]
    fn grid_dependencies_and_tensor_scoreboards_are_refused() {
        let (kernel, launch) = input("ld.global.u64 %r0,[4096];", 1, false);
        let program = project(&kernel, &launch).unwrap();
        let states = certificate(&program).unwrap();
        assert!(verify_certificate(&kernel, &launch, &states));

        for dependency in [0, 1] {
            let mut waiting = launch.clone();
            waiting.grids[0].waits.push(dependency);
            assert!(matches!(project(&kernel, &waiting), Err(Refusal::Shape)));
            assert!(!verify_certificate(&kernel, &waiting, &states));
        }

        let mut scoreboard = kernel.clone();
        scoreboard.ldtm_sb.push((0, 0));
        assert!(matches!(
            project(&scoreboard, &launch),
            Err(Refusal::Kernel)
        ));
        assert!(!verify_certificate(&scoreboard, &launch, &states));

        let mut scoreboard_wait = kernel.clone();
        scoreboard_wait.ldtm_waits.push((0, 1, vec![]));
        assert!(matches!(
            project(&scoreboard_wait, &launch),
            Err(Refusal::Kernel)
        ));
        assert!(!verify_certificate(&scoreboard_wait, &launch, &states));
    }

    #[test]
    fn readback_requires_valid_initially_defined_output_allocations() {
        let (kernel, mut launch) = input("ld.global.u64 %r0,[4096];", 1, false);
        launch.outputs.push(0);
        assert!(strict(&project(&kernel, &launch).unwrap()));
        launch.outputs.push(1);
        assert!(project(&kernel, &launch).is_err());
        let (kernel, mut launch) = input("st.global.u64 [4096],0;", 1, true);
        launch.outputs.push(0);
        assert!(project(&kernel, &launch).is_err());
    }

    #[test]
    fn malformed_dimensions_are_refused_without_overflow() {
        let (mut kernel, mut launch) = input("ld.global.u64 %r0,[4096];", 1, false);
        launch.grids[0].block = [u32::MAX; 3];
        assert!(project(&kernel, &launch).is_err());
        launch.grids[0].block = [1, 1, 1];
        kernel.ntid = Some(ptxv_ir::Ntid {
            req: false,
            dims: [u32::MAX; 3],
        });
        assert!(project(&kernel, &launch).is_err());
    }

    #[test]
    fn whole_code_projection_preserves_accesses_and_source_pcs() {
        let (mut kernel, _) = input("ld.global.u64 %r0,[4096];", 1, false);
        let mut nop = kernel.code[0].clone();
        nop.op = Op::Nop;
        nop.text = "typed IR Nop".into();
        kernel.code.insert(0, nop.clone());
        kernel.code.insert(2, nop);
        let projected = project_code(&kernel.code, kernel.nregs[File::V as usize]).unwrap();
        assert_eq!(projected.len(), kernel.code.len());
        assert!(matches!(projected[0], Instruction::Skip));
        assert!(matches!(
            projected[1],
            Instruction::Access(Request {
                kind: AccessKind::Load,
                ..
            })
        ));
        assert!(matches!(projected[2], Instruction::Skip));
        assert!(matches!(projected[3], Instruction::Halt));
        let (kernel, _) = input("ret; ld.global.u64 %r0,[4096];", 1, false);
        assert!(project_code(&kernel.code, kernel.nregs[File::V as usize]).is_err());
        assert!(project_code(&[], 0).is_err());
    }

    #[test]
    fn constant_address_matches_native_modular_sum() {
        let address = Addr {
            base: Some(Val::Imm(u64::MAX)),
            base2: Some(Val::Imm(3)),
            off: -1,
        };
        assert_eq!(constant_address(&address), Some(1));
        assert_eq!(
            constant_address(&Addr {
                base: Some(Val::Const(0, 8)),
                ..address
            }),
            None
        );
    }
}
