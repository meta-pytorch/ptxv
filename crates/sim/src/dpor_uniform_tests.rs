/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ptxv_ir::File;
use ptxv_ir::Reg;

use crate::machine::Footprint;
use crate::machine::Machine;

pub(crate) fn machine(init: &str, peer: &str, victim: &[&str], threads: u32) -> Machine {
    let mut instructions = vec![
        "S2UR UR0, SR_CTAID.X",
        init,
        "S2R R0, SR_LANEID",
        "ISETP.EQ.U32.AND P0, PT, R0, RZ, PT",
        "@P0 BRA `(.L_victim)",
        peer,
        "EXIT",
    ];
    instructions.extend_from_slice(victim);
    instructions.extend(["BPT.TRAP 0x1", "EXIT"]);
    let exit = instructions.len() - 1;
    let mut text = String::from(".text.k:\n");
    for (pc, instruction) in instructions.iter().enumerate() {
        if pc == 7 {
            text.push_str(".L_victim:\n");
        }
        if pc == exit {
            text.push_str(".L_exit:\n");
        }
        let control = ptxv_sass::scoreboard::Ctrl {
            stall: 15,
            yld: false,
            wbar: matches!(pc, 0 | 2).then_some(0),
            rbar: None,
            wait: if matches!(pc, 1 | 3) { 1 } else { 0 },
            reuse: 0,
        };
        text.push_str(&format!(
            "/*{:04x}*/ {instruction} ;\n/* 0x{:016x} */\n",
            pc * 16,
            control.encode(0)
        ));
    }
    let kernel = ptxv_sass::lower(&text, "k", &ptxv_sass::CubinMeta::default()).unwrap();
    assert!(
        kernel.static_findings.is_empty(),
        "{:?}",
        kernel.static_findings
    );
    let manifest = serde_json::from_value(serde_json::json!({
        "name":"uniform-dependency", "description":"shared warp register dependencies",
        "entry":"k", "grid":[1,1,1], "block":[threads,1,1], "outputs":[],
        "buffers":[], "params":[], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: std::path::PathBuf::new(),
        m: manifest,
    };
    let launch = crate::build_launch(&case, &kernel).unwrap();
    Machine::new(Arc::new(kernel), Arc::new(launch))
}

#[test]
fn uniform_peer_writes_preserve_reachable_traps() {
    for (name, init, peer, victim) in [
        (
            "UP",
            "UISETP.EQ.U32.AND UP0, UPT, UR0, URZ, UPT",
            "UISETP.NE.U32.AND UP0, UPT, URZ, URZ, UPT",
            vec!["@UP0 BRA `(.L_exit)"],
        ),
        (
            "negated UP",
            "UISETP.NE.U32.AND UP0, UPT, UR0, URZ, UPT",
            "UISETP.EQ.U32.AND UP0, UPT, URZ, URZ, UPT",
            vec!["@!UP0 BRA `(.L_exit)"],
        ),
        (
            "UR compare",
            "NOP",
            "UMOV UR0, 0x1",
            vec![
                "ISETP.EQ.U32.AND P1, PT, UR0, URZ, PT",
                "@P1 BRA `(.L_exit)",
            ],
        ),
        (
            "captured UR",
            "NOP",
            "UMOV UR0, 0x1",
            vec![
                "MOV R2, UR0",
                "ISETP.EQ.U32.AND P1, PT, R2, RZ, PT",
                "@P1 BRA `(.L_exit)",
            ],
        ),
    ] {
        let mut machine = machine(init, peer, &victim, 32);
        for pc in 0..5 {
            machine.exec_one(&(0..32).collect::<Vec<_>>(), pc).unwrap();
        }
        let stateful = crate::explore::explore(machine.clone(), 10000);
        let dpor = crate::dpor::explore_dpor(machine, 10000);
        eprintln!(
            "{name}: stateful states={}, complete={}, classes={:?}; DPOR states={}, complete={}, classes={:?}",
            stateful.states,
            stateful.complete,
            stateful.classes(),
            dpor.states,
            dpor.complete,
            dpor.classes()
        );

        assert!(dpor.complete && stateful.complete);
        assert!(
            dpor.classes().iter().any(|kind| kind == "trap"),
            "{name}: {:?}",
            dpor.findings
        );
        assert_eq!(stateful.classes(), dpor.classes(), "{name}");
    }
}

#[test]
fn private_peer_writes_do_not_change_victim_registers() {
    for (init, peer, victim) in [
        (
            "MOV R3, 0x1",
            "MOV R3, 0x0",
            vec!["ISETP.NE.U32.AND P1, PT, R3, RZ, PT", "@P1 BRA `(.L_exit)"],
        ),
        (
            "ISETP.EQ.U32.AND P1, PT, RZ, RZ, PT",
            "ISETP.NE.U32.AND P1, PT, RZ, RZ, PT",
            vec!["@P1 BRA `(.L_exit)"],
        ),
    ] {
        let mut machine = machine(init, peer, &victim, 32);
        for pc in 0..5 {
            machine.exec_one(&(0..32).collect::<Vec<_>>(), pc).unwrap();
        }
        let stateful = crate::explore::explore(machine.clone(), 10000);
        let dpor = crate::dpor::explore_dpor(machine, 10000);
        eprintln!(
            "private: stateful states={}, complete={}, classes={:?}; DPOR states={}, complete={}, classes={:?}",
            stateful.states,
            stateful.complete,
            stateful.classes(),
            dpor.states,
            dpor.complete,
            dpor.classes()
        );
        assert!(dpor.complete && stateful.complete);
        assert_eq!(dpor.classes(), ["clean"]);
        assert_eq!(stateful.classes(), dpor.classes());
    }
}

#[test]
fn native_uniform_write_keys_match_canonical_owner_aliases() {
    let mut machine = machine(
        "UMOV UR0, 0x1",
        "UMOV UR0, 0x0",
        &["@UP0 BRA `(.L_exit)"],
        64,
    );
    let warp_ids: Vec<_> = machine.threads.iter().map(|thread| thread.warp).collect();
    let owners = ptxv_ir::register_owner::uniform_owners(&warp_ids, machine.warps.len()).unwrap();
    for file in [File::UR, File::UP] {
        let register = Reg { file, idx: 0 };
        for peer in [1, 32] {
            machine.set_footprint(Some(Box::new(Footprint::default())));
            machine.wr(0, register, 1, 1);
            let mut first = *machine.take_footprint().unwrap();
            first.normalize();
            machine.set_footprint(Some(Box::new(Footprint::default())));
            machine.wr(peer, register, 0, 1);
            let mut second = *machine.take_footprint().unwrap();
            second.normalize();
            let alias = ptxv_ir::register_owner::canonical_key(0, &owners, register)
                == ptxv_ir::register_owner::canonical_key(peer, &owners, register);
            assert_eq!(first.conflicts(&second), alias, "{file:?} peer{peer}");
        }
    }
}
