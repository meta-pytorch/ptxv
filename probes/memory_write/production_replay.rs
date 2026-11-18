/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Test-only adapter compiled inside an archived production `num` module.

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use serde_json::json;

use super::*;
use crate::halfin::Half;
use crate::halfin::Src;
use crate::machine::tmem_idx;

const WIDTH: u64 = 32;

fn origin(value: Option<Src>) -> Value {
    match value {
        None => Value::Null,
        Some(Src::Zero) => json!([0, 0, 0, 0]),
        Some(Src::E { alloc, elem, b }) => json!([1, alloc, elem, b]),
        Some(Src::N { node, b }) => json!([2, node, 0, b]),
        Some(Src::E7 { alloc, elem }) => json!([3, alloc, elem, 0]),
    }
}

fn machine() -> Machine {
    machine_for_body("")
}

fn machine_for_body(body: &str) -> Machine {
    machine_for_body_threads(body, 1)
}

fn machine_for_body_threads(body: &str, threads: u32) -> Machine {
    let ptx = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input, .param .u64 output, .param .align 128 .b8 tensor[128]) {{
  .shared .align 16 .b8 scratch[64];
  .shared .align 8 .b64 barrier;
  .local .align 4 .b8 spill[16];
  .reg .b32 %r<8>;
  .reg .b64 %p<2>;
  .reg .pred %q;
  ld.param.u64 %p0, [input];
  ld.param.u64 %p1, [output];
  {body}
  ret;
}}
"#
    );
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&ptx, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"production-write-replay", "description":"central write adapter",
        "entry":"k", "grid":[1,1,1], "block":[threads,1,1], "outputs":["out"],
        "buffers":[
            {"name":"x","dtype":"bf16","shape":[32],"init":"zeros"},
            {"name":"out","dtype":"u8","shape":[64],"init":"zeros"}
        ],
        "params":[{"kind":"ptr","buffer":"x"},{"kind":"ptr","buffer":"out"},
            {"kind":"tensormap","buffer":"x","box":[8]}],
        "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    let mut m = Machine::new(kernel, launch);
    let mut ns = NumState::default();
    ns.half.track = true;
    ns.half.inputs.insert(0, Half::Bf16);
    ns.not_inputs.insert(1);
    m.num = Some(Box::new(ns));
    m
}

fn loc(space: u8, allocation: u64, offset: u64) -> Loc {
    match space {
        0 => Loc::Shared {
            cta: allocation as u16,
            off: offset as u32,
        },
        1 => Loc::Global {
            alloc: allocation as usize,
            off: offset,
        },
        2 => Loc::Param {
            grid: allocation as u16,
            off: offset as u32,
        },
        _ => panic!("not a byte-addressed production location: {space}"),
    }
}

fn row(m: &Machine, space: u8, allocation: u64, offset: u64) -> Value {
    let ns = m.num.as_ref().unwrap();
    let key = (space, allocation, offset);
    let (raw_origin, global_written) = ns.half.production_replay_raw(key);
    let (value, defined, written, byte_origin, word_key) = match space {
        0 | 1 => {
            let mem = if space == 0 {
                &m.smem[allocation as usize]
            } else {
                &m.gmem[allocation as usize]
            };
            let shadow = mem.sh.get(offset as usize);
            let value = *mem.bytes.get(offset as usize);
            let byte_origin = m.half_mem(loc(space, allocation, offset), 1, Some(&[value]))[0];
            (
                value,
                (space == 1 || shadow.w.is_some()) && !shadow.undef,
                if space == 1 && ns.half.is_on() {
                    global_written
                } else {
                    shadow.w.is_some()
                },
                byte_origin,
                Some(key),
            )
        }
        2 => {
            let value = m.launch.grids[allocation as usize].params[offset as usize];
            (value, true, false, raw_origin, Some(key))
        }
        3 => {
            let mem = &m.tmem[allocation as usize];
            let index = offset as usize / 4;
            let written = *mem.init.get(index) == 1;
            (
                ((*mem.words.get(index) >> (8 * (offset % 4))) & 255) as u8,
                written && !mem.sh.get(index).undef,
                written,
                raw_origin,
                offset
                    .is_multiple_of(4)
                    .then_some((3, allocation, offset / 4)),
            )
        }
        4 => {
            let local = &m.threads[allocation as usize].local;
            let word = local
                .binary_search_by_key(&((offset & !3) as u32), |(address, _)| *address)
                .ok()
                .map(|index| local[index].1);
            let shift = 8 * (offset % 4);
            (
                word.map_or(0, |word| (word >> shift) as u8),
                word.is_some_and(|word| (word >> (32 + shift)) & 255 == 0),
                word.is_some(),
                raw_origin,
                Some(key),
            )
        }
        _ => panic!("unknown space"),
    };
    json!({"key":[space,allocation,offset],"value":value,"origin":origin(byte_origin),
        "tainted":ns.taint.memory_unit_input_derived(key),"defined":defined,"written":written,
        "word":word_key.and_then(|key| ns.mem.get(&key).copied()).map(|node| history(ns, node))})
}

fn history(ns: &NumState, node: u32) -> Value {
    json!([
        u8::from(matches!(ns.nodes[node as usize].kind, NodeKind::Opaque)),
        node
    ])
}

fn markers(m: &Machine) -> Value {
    let ns = m.num.as_ref().unwrap();
    let mut numeric: Vec<_> = ns.written.iter().copied().collect();
    numeric.sort();
    let half: Vec<_> = (0..m.gmem.len())
        .flat_map(|allocation| {
            (0..64).filter_map(move |offset| {
                ns.half
                    .production_replay_raw((1, allocation as u64, offset))
                    .1
                    .then_some((allocation, offset))
            })
        })
        .collect();
    json!({"numeric_written":numeric,"half_written":half})
}

fn rows(m: &Machine) -> Vec<Value> {
    [
        (0, 0, WIDTH),
        (1, 0, WIDTH),
        (1, 1, WIDTH),
        (2, 0, 16),
        (3, 0, 128),
        (4, 0, WIDTH),
    ]
    .into_iter()
    .flat_map(|(space, allocation, width)| {
        (0..width).map(move |offset| row(m, space, allocation, offset))
    })
    .chain((1024..1024 + WIDTH).map(|offset| row(m, 0, 0, offset)))
    .collect()
}

fn payload(rows: &[Value]) -> Vec<Value> {
    rows.iter()
        .map(|row| {
            json!({"value":row["value"],"origin":row["origin"],
        "tainted":row["tainted"],"defined":row["defined"],"word":row["word"]})
        })
        .collect()
}

fn event(key: (u8, u64, u64), payload: Vec<Value>) -> Value {
    json!({"kind":"write","key":[key.0,key.1,key.2],"payload":payload})
}

fn write_bytes(m: &mut Machine, key: (u8, u64, u64), data: &[u8], undefined: u64) {
    m.wundef = u128::from(undefined);
    let vc = m.threads[0].vc.clone();
    m.access(0, &vc, 0, loc(key.0, key.1, key.2), data.len(), Some(data));
    m.wundef = 0;
}

fn seed(
    m: &mut Machine,
    space: u8,
    allocation: u64,
    offset: u64,
    bytes: &[u8],
    tainted: bool,
) {
    write_bytes(m, (space, allocation, offset), bytes, 0);
    let origins: Vec<_> = (0..bytes.len())
        .map(|byte| {
            Some(match byte % 4 {
                0 => Src::Zero,
                1 => Src::E {
                    alloc: 0,
                    elem: byte as u64,
                    b: 0,
                },
                2 => Src::N { node: 7, b: 0 },
                _ => Src::E7 {
                    alloc: 0,
                    elem: byte as u64,
                },
            })
        })
        .collect();
    m.half_mem_set((space, allocation, offset), &origins);
    let ns = m.num.as_mut().unwrap();
    ns.taint.assign_memory(
        space,
        allocation,
        offset,
        offset + bytes.len() as u64,
        tainted,
    );
    for word in (0..bytes.len() / 4).map(|word| 4 * word) {
        let bits = u32::from_le_bytes(bytes[word..word + 4].try_into().unwrap());
        let node = ns.const_leaf(bits, 0);
        ns.mem
            .insert((space, allocation, offset + word as u64), node);
    }
}

fn copy_case(
    name: &str,
    source: (u8, u64, u64),
    destination: (u8, u64, u64),
    width: usize,
    tainted: bool,
    overlap: bool,
    frozen: bool,
    undefined: u64,
) -> Value {
    let mut m = machine();
    let bytes: Vec<_> = (0..16).map(|byte| 17 + byte).collect();
    seed(&mut m, source.0, source.1, source.2, &bytes, tainted);
    if !overlap {
        seed(
            &mut m,
            destination.0,
            destination.1,
            destination.2,
            &[250; 16],
            !tainted,
        );
    }
    if undefined != 0 {
        for byte in 0..width {
            let mem = if source.0 == 0 {
                &mut m.smem[source.1 as usize]
            } else {
                &mut m.gmem[source.1 as usize]
            };
            mem.sh.get_mut(source.2 as usize + byte).undef = undefined >> byte & 1 != 0;
        }
    }
    let data: Vec<u8> = (0..width)
        .map(|byte| {
            row(&m, source.0, source.1, source.2 + byte as u64)["value"]
                .as_u64()
                .unwrap() as u8
        })
        .collect();
    let snapshot = m.num_read_snapshot(Some(loc(source.0, source.1, source.2)), &data, 0);
    let captured: Vec<_> = (0..width)
        .map(|byte| row(&m, source.0, source.1, source.2 + byte as u64))
        .collect();
    let mut events = Vec::new();
    let initial = rows(&m);
    let initial_markers = markers(&m);
    if frozen {
        let zeros = vec![0; width];
        write_bytes(&mut m, source, &zeros, 0);
        let zero_snapshot = m.num_read_snapshot(None, &zeros, 0);
        m.num_copy_snapshot(
            loc(source.0, source.1, source.2),
            &zeros,
            zero_snapshot.as_ref(),
            0,
        );
        let overwritten: Vec<_> = (0..width).map(|_| json!({"value":0,"origin":[0,0,0,0],"tainted":false,"defined":true,"word":null})).collect();
        events.push(event(source, overwritten));
    }
    write_bytes(&mut m, destination, &data, undefined);
    m.num_copy_snapshot(
        loc(destination.0, destination.1, destination.2),
        &data,
        snapshot.as_ref(),
        0,
    );
    events.push(event(destination, payload(&captured)));
    json!({"name":name,"cells":initial,"events":events,"expected":rows(&m),"markers_before":initial_markers,"markers_after":markers(&m)})
}

fn store_case(space: u8, offset: u64, width: usize, undefined: u64, tainted: bool) -> Value {
    let mut m = machine();
    let allocation = if space == 1 { 1 } else { 0 };
    seed(&mut m, space, allocation, 0, &[250; 32], false);
    let reg = Reg {
        file: File::R,
        idx: 0,
    };
    let value = Val::Reg(reg, 1);
    let bytes: Vec<u8> = (0..width).map(|byte| 50 + byte as u8).collect();
    let origins = [
        Some(Src::Zero),
        Some(Src::E {
            alloc: 0,
            elem: 2,
            b: 1,
        }),
        None,
        Some(Src::E7 { alloc: 0, elem: 3 }),
    ];
    m.half_set_reg(0, reg, Some(origins));
    let ns = m.num.as_mut().unwrap();
    if tainted {
        ns.taint.reg.insert(ns.taint.reg_key(0, reg));
    }
    let node = ns.const_leaf(u32::from_le_bytes([50, 51, 52, 53]), 0);
    ns.set(0, reg, Some(node));
    let node_history = history(ns, node);
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let capture = m.num_store_origins(0, &value, width as u8, None);
    let node_start = m.num.as_ref().unwrap().nodes.len();
    let source: Vec<_> = (0..width).map(|byte| json!({"value":bytes[byte],"origin":origin(origins[byte % 4]),
        "tainted":tainted,"defined":undefined >> byte & 1 == 0,"word":if width == 4 && byte == 0 { Some(node_history.clone()) } else { None }})).collect();
    write_bytes(&mut m, (space, allocation, offset), &bytes, undefined);
    m.num_st(
        0,
        &value,
        loc(space, allocation, offset),
        &bytes,
        &capture,
        0,
    );
    let ns = m.num.as_ref().unwrap();
    let mut effect = event((space, allocation, offset), source);
    let updates: Vec<_> = (offset & !3..offset + width as u64)
        .step_by(4)
        .filter_map(|start| {
            ns.mem
                .get(&(space, allocation, start))
                .copied()
                .filter(|node| *node as usize >= node_start)
                .map(|node| json!({"key":[space,allocation,start],"history":history(ns,node)}))
        })
        .collect();
    effect["word_updates"] = json!(updates);
    json!({"name":format!("store-space-{space}-off-{offset}-width-{width}-undef-{undefined}-taint-{tainted}"),
        "cells":initial,"events":[effect],"expected":rows(&m),"markers_before":initial_markers,"markers_after":markers(&m)})
}

fn local_or_tmem_case(space: u8, undefined: bool) -> Value {
    let mut m = machine();
    let reg = Reg {
        file: File::R,
        idx: 0,
    };
    let value = Val::Reg(reg, 1);
    let bits = 0x44332211;
    let origins = [
        Some(Src::Zero),
        None,
        Some(Src::E {
            alloc: 0,
            elem: 1,
            b: 1,
        }),
        Some(Src::E7 { alloc: 0, elem: 2 }),
    ];
    m.half_set_reg(0, reg, Some(origins));
    let ns = m.num.as_mut().unwrap();
    let node = ns.const_leaf(bits, 0);
    ns.set(0, reg, Some(node));
    let node_history = history(ns, node);
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let offset = 4;
    if space == 3 {
        let index = tmem_idx(1, 0);
        *m.tmem[0].words.get_mut(index) = bits;
        *m.tmem[0].init.get_mut(index) = 1;
        m.tmem[0].sh.get_mut(index).undef = undefined;
        m.num_tmem_st(0, &value, 0, 1, 0);
    } else {
        m.threads[0].make_mut().local.push((
            offset,
            bits as u64 | if undefined { 0xffff_ffff_0000_0000 } else { 0 },
        ));
        m.num_local_st(0, &value, offset);
    }
    let source: Vec<_> = (0..4).map(|byte| json!({"value":bits.to_le_bytes()[byte],"origin":origin(origins[byte]),
        "tainted":false,"defined":!undefined,"word":if byte == 0 { Some(node_history.clone()) } else { None }})).collect();
    json!({"name":format!("helper-store-space-{space}-undefined-{undefined}"),"cells":initial,
        "events":[event((space,0,offset as u64),source)],"expected":rows(&m),
        "markers_before":initial_markers,"markers_after":markers(&m)})
}

fn scheduler_case(name: &str, body: &str, writes: &[(u8, u64, u64, usize)]) -> Value {
    let m = machine_for_body(body);
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let (m, findings, complete) = crate::explore::run_once(m);
    assert!(complete, "{name}: incomplete: {findings:?}");
    assert!(findings.is_empty(), "{name}: {findings:?}");
    let events: Vec<_> = writes.iter().map(|&(space, allocation, offset, width)| {
        let mut effect = event((space, allocation, offset), (0..width).map(|_| json!({
            "value":0,"origin":if name.starts_with("atomic") { Value::Null } else { json!([0,0,0,0]) },
            "tainted":false,"defined":true,"word":null
        })).collect());
        if name.contains("f32") {
            let ns = m.num.as_ref().unwrap();
            let node = ns.nodes.iter().rposition(|node| matches!(node.kind,NodeKind::AtomAdd(..))).unwrap() as u32;
            effect["word_updates"] = json!([{"key":[space,allocation,offset],"history":history(ns,node)}]);
        }
        effect
    }).collect();
    json!({"name":format!("scheduler-{name}"),"cells":initial,"events":events,"expected":rows(&m),
        "markers_before":initial_markers,"markers_after":markers(&m)})
}

fn check_markers(case: &Value) {
    for (plane, per_word) in [("numeric_written", true), ("half_written", false)] {
        let mut expected: BTreeSet<(u64, u64)> =
            serde_json::from_value(case["markers_before"][plane].clone()).unwrap();
        for effect in case["events"].as_array().unwrap() {
            if plane == "half_written" && case["half_tracking"] == false {
                continue;
            }
            let key = effect["key"].as_array().unwrap();
            if key[0] != 1 {
                continue;
            }
            let allocation = key[1].as_u64().unwrap();
            let off = key[2].as_u64().unwrap();
            if effect["kind"] == "numeric_forget" {
                if per_word {
                    expected.insert((allocation, off));
                }
                continue;
            }
            let width = effect["payload"].as_array().unwrap().len() as u64;
            for byte in off..off + width {
                expected.insert((allocation, if per_word { byte & !3 } else { byte }));
            }
        }
        let actual: BTreeSet<(u64, u64)> =
            serde_json::from_value(case["markers_after"][plane].clone()).unwrap();
        assert_eq!(expected, actual, "{}: {plane}", case["name"]);
    }
}

fn cas_case(success: bool, width: usize) -> Value {
    let mut m = machine_for_body(&format!(
        "atom.global.cas.b{} {}, [%p0], {}, 34;",
        8 * width,
        if width == 8 { "%p1" } else { "%r0" },
        if success { 17 } else { 16 }
    ));
    let old = 17u64.to_le_bytes();
    Arc::make_mut(&mut m.launch).allocs[0].init[..width].copy_from_slice(&old[..width]);
    for (offset, byte) in old[..width].iter().copied().enumerate() {
        *m.gmem[0].bytes.get_mut(offset) = byte;
    }
    let ns = m.num.as_mut().unwrap();
    let node = ns.const_leaf(17, 0);
    ns.mem.insert((1, 0, 0), node);
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let (m, findings, complete) = crate::explore::run_once(m);
    assert!(complete, "cas success={success}: {findings:?}");
    assert!(findings.is_empty(), "cas success={success}: {findings:?}");
    let events: Vec<_> = if success {
        vec![event((1,0,0),34u64.to_le_bytes()[..width].iter().copied().map(|byte| json!({"value":byte,"origin":null,"tainted":false,"defined":true,"word":null})).collect())]
    } else {
        vec![json!({"kind":"numeric_forget","key":[1,0,0]})]
    };
    json!({"name":format!("scheduler-cas-width-{width}-success-{success}"),"cells":initial,"events":events,"expected":rows(&m),
        "markers_before":initial_markers,"markers_after":markers(&m)})
}

fn tracking_disabled_case() -> Value {
    let mut m = machine_for_body("st.global.u32 [%p1], 0;");
    let half = &mut m.num.as_mut().unwrap().half;
    half.track = false;
    half.inputs.clear();
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let (m, findings, complete) = crate::explore::run_once(m);
    assert!(
        complete && findings.is_empty(),
        "tracking disabled: {findings:?}"
    );
    let source = (0..4)
        .map(|_| json!({"value":0,"origin":null,"tainted":false,"defined":true,"word":null}))
        .collect();
    json!({"name":"scheduler-half-tracking-disabled","half_tracking":false,"numeric_tracking":true,"cells":initial,
        "events":[event((1,1,0),source)],"expected":rows(&m),"markers_before":initial_markers,"markers_after":markers(&m)})
}

fn wide_atomic_boundary() -> Value {
    let mut m = machine_for_body(
        "ld.global.u32 %r1, [%p0+4]; atom.global.cas.b64 %p1, [%p0], 17, 4294967330;",
    );
    let old = 17u64.to_le_bytes();
    Arc::make_mut(&mut m.launch).allocs[0].init[..8].copy_from_slice(&old);
    for (offset, byte) in old.into_iter().enumerate() {
        *m.gmem[0].bytes.get_mut(offset) = byte;
    }
    m.num.as_mut().unwrap().f32_inputs.insert(0);
    for _ in 0..10 {
        if matches!(m.k.code[m.threads[0].pc as usize].op, Op::Atom { .. }) {
            break;
        }
        let action = m.actions()[0].0.clone();
        m.apply(&action);
    }
    assert!(matches!(
        m.k.code[m.threads[0].pc as usize].op,
        Op::Atom { .. }
    ));
    assert!(m.num.as_ref().unwrap().mem.contains_key(&(1, 0, 4)));
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let (m, findings, complete) = crate::explore::run_once(m);
    assert!(
        complete && findings.is_empty(),
        "wide CAS boundary: {findings:?}"
    );
    if std::env::var("PTXV_ALLOW_KNOWN_WIDE_ATOMIC_BUG").as_deref() == Ok("1") {
        let node = m.num.as_ref().unwrap().mem[&(1, 0, 4)];
        assert_eq!(m.num.as_ref().unwrap().nodes[node as usize].hw, 0);
    } else {
        assert!(
            !m.num.as_ref().unwrap().mem.contains_key(&(1, 0, 4)),
            "wide CAS retained a stale numeric history"
        );
    }
    assert_eq!(*m.gmem[0].bytes.get(4), 1);
    let source = 4294967330u64
        .to_le_bytes()
        .into_iter()
        .map(|byte| json!({"value":byte,"origin":null,"tainted":false,"defined":true,"word":null}))
        .collect();
    json!({"name":"reachable-wide-cas-stale-upper-history","cells":initial,"events":[event((1,0,0),source)],"expected":rows(&m),
        "markers_before":initial_markers,"markers_after":markers(&m)})
}

fn scheduler_tmem_case() -> Value {
    let mut m = machine_for_body_threads(
        "mov.u32 %r0, 0; mov.u32 %r3, scratch; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r3], 32; bar.sync 0; ld.shared.u32 %r4, [%r3]; tcgen05.fence::after_thread_sync; tcgen05.st.sync.aligned.32x32b.x1.b32 [%r4], {%r0}; tcgen05.wait::st.sync.aligned; tcgen05.fence::before_thread_sync; bar.sync 0; tcgen05.fence::after_thread_sync; tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r4, 32;",
        32,
    );
    let initial = rows(&m);
    let initial_markers = markers(&m);
    for _ in 0..100 {
        if matches!(
            m.k.code[m.threads[0].pc as usize].op,
            Op::TmemDealloc { .. }
        ) {
            break;
        }
        let action = m.actions()[0].0.clone();
        m.apply(&action);
    }
    assert!(matches!(
        m.k.code[m.threads[0].pc as usize].op,
        Op::TmemDealloc { .. }
    ));
    let after = rows(&m);
    let after_markers = markers(&m);
    let (_, findings, complete) = crate::explore::run_once(m);
    assert!(
        complete && findings.is_empty(),
        "TMEM scheduler store: {findings:?}"
    );
    let zeros = |width, source| {
        (0..width)
            .map(|_| json!({"value":0,"origin":origin(source),"tainted":false,"defined":true,"word":null}))
            .collect()
    };
    json!({"name":"scheduler-tmem-store-before-deallocation","cells":initial,
        "events":[event((0,0,1024),zeros(4,None)),event((3,0,0),zeros(128,Some(Src::Zero)))],"expected":after,
        "markers_before":initial_markers,"markers_after":after_markers})
}

fn scheduler_copy_case(name: &str, body: &str, frozen: bool) -> Value {
    let m = machine_for_body(body);
    let captured: Vec<_> = (0..16).map(|offset| row(&m, 1, 0, offset)).collect();
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let (m, findings, complete) = crate::explore::run_once(m);
    assert!(complete, "{name}: incomplete: {findings:?}");
    assert!(findings.is_empty(), "{name}: {findings:?}");
    let mut events = Vec::new();
    if frozen {
        events.push(event(
            (1, 0, 0),
            (0..4)
                .map(|_| {
                    json!({"value":0,"origin":[0,0,0,0],
            "tainted":false,"defined":true,"word":null})
                })
                .collect(),
        ));
    }
    events.push(event((0, 0, 1024), payload(&captured)));
    json!({"name":format!("scheduler-{name}"),"cells":initial,"events":events,"expected":rows(&m),
        "markers_before":initial_markers,"markers_after":markers(&m)})
}

fn bulk_case(reduce: bool) -> Value {
    let body = format!(
        "mov.u32 %r1, scratch; st.shared.u32 [%r1], 0; st.shared.u32 [%r1+4], 0; st.shared.u32 [%r1+8], 0; st.shared.u32 [%r1+12], 0; fence.proxy.async.shared::cta; {} [%p1], [%r1], 16; cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;",
        if reduce {
            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
        } else {
            "cp.async.bulk.global.shared::cta.bulk_group"
        }
    );
    let m = machine_for_body(&body);
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let (m, findings, complete) = crate::explore::run_once(m);
    assert!(complete, "bulk reduce={reduce}: {findings:?}");
    assert!(findings.is_empty(), "bulk reduce={reduce}: {findings:?}");
    let zeros: Vec<_> = (0..16)
        .map(|_| json!({"value":0,"origin":[0,0,0,0],"tainted":false,"defined":true,"word":null}))
        .collect();
    let shared = event((0, 0, 1024), zeros.clone());
    let mut destination = event((1, 1, 0), zeros);
    if reduce {
        for byte in destination["payload"].as_array_mut().unwrap() {
            byte["origin"] = Value::Null;
        }
        let ns = m.num.as_ref().unwrap();
        let nodes: Vec<_> = ns
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(node, value)| {
                matches!(value.kind, NodeKind::AtomAdd(..)).then_some(node as u32)
            })
            .collect();
        assert_eq!(nodes.len(), 4);
        destination["word_updates"] = json!(
            nodes
                .into_iter()
                .enumerate()
                .map(|(word, node)| json!({"key":[1,1,4*word],"history":history(ns,node)}))
                .collect::<Vec<_>>()
        );
    }
    json!({"name":format!("scheduler-bulk-reduce-{reduce}"),"cells":initial,"events":[shared,destination],"expected":rows(&m),
        "markers_before":initial_markers,"markers_after":markers(&m)})
}

fn tma_case(zero: bool) -> Value {
    let body = format!(
        "mov.u32 %r1, scratch; mov.u32 %r2, barrier; mbarrier.init.shared::cta.b64 [%r2], 1; mbarrier.arrive.expect_tx.shared::cta.b64 %p1, [%r2], 16; mov.b64 %p0, tensor; cvta.param.u64 %p0, %p0; cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%p0, {{{}}}], [%r2]; WAIT: mbarrier.try_wait.parity.shared::cta.b64 %q, [%r2], 0; @!%q bra WAIT;",
        if zero { 32 } else { 0 }
    );
    let m = machine_for_body(&body);
    let initial = rows(&m);
    let initial_markers = markers(&m);
    let captured: Vec<_> = (0..16).map(|offset| row(&m, 1, 0, offset)).collect();
    let source = if zero {
        (0..16).map(|_| json!({"value":0,"origin":[0,0,0,0],"tainted":false,"defined":true,"word":null})).collect()
    } else {
        payload(&captured)
    };
    let (m, findings, complete) = crate::explore::run_once(m);
    assert!(complete, "tma zero={zero}: {findings:?}");
    assert!(findings.is_empty(), "tma zero={zero}: {findings:?}");
    json!({"name":format!("scheduler-tma-zero-{zero}"),"cells":initial,"events":[event((0,0,1024),source)],"expected":rows(&m),
        "markers_before":initial_markers,"markers_after":markers(&m)})
}

#[test]
fn export_production_replay() {
    let output =
        std::env::var("PTXV_PRODUCTION_REPLAY_OUTPUT").expect("set PTXV_PRODUCTION_REPLAY_OUTPUT");
    let mut cases = Vec::new();
    for (from, to) in [
        ((1, 0, 0), (0, 0, 8)),
        ((0, 0, 0), (1, 1, 8)),
        ((1, 0, 0), (1, 1, 8)),
        ((0, 0, 0), (0, 0, 16)),
    ] {
        for width in [0, 4, 8, 16] {
            for tainted in [false, true] {
                cases.push(copy_case(
                    &format!("copy-{from:?}-{to:?}-{width}-{tainted}"),
                    from,
                    to,
                    width,
                    tainted,
                    false,
                    false,
                    0,
                ));
            }
        }
    }
    cases.push(copy_case(
        "overlap-forward",
        (0, 0, 0),
        (0, 0, 4),
        12,
        true,
        true,
        false,
        0,
    ));
    cases.push(copy_case(
        "overlap-backward",
        (0, 0, 4),
        (0, 0, 0),
        12,
        false,
        true,
        false,
        0,
    ));
    cases.push(copy_case(
        "async-frozen-source",
        (1, 0, 0),
        (0, 0, 16),
        8,
        true,
        false,
        true,
        0,
    ));
    cases.push(copy_case(
        "copy-undefined-byte",
        (0, 0, 0),
        (1, 1, 8),
        8,
        true,
        false,
        false,
        0b1010,
    ));
    for space in [0, 1] {
        for (offset, width) in [(0, 4), (4, 4), (1, 1), (3, 2), (7, 3)] {
            cases.push(store_case(space, offset, width, 0, false));
        }
        cases.push(store_case(space, 8, 4, 0b1010, false));
        cases.push(store_case(space, 8, 4, 0, true));
        cases.push(store_case(space, 3, 2, 0b01, true));
    }
    for space in [3, 4] {
        for undefined in [false, true] {
            cases.push(local_or_tmem_case(space, undefined));
        }
    }
    cases.push(scheduler_case(
        "global-store",
        "st.global.u32 [%p1], 0;",
        &[(1, 1, 0, 4)],
    ));
    cases.push(scheduler_case(
        "shared-store",
        "st.shared.u32 [scratch], 0;",
        &[(0, 0, 1024, 4)],
    ));
    cases.push(scheduler_case(
        "generic-store",
        "st.u32 [%p1], 0;",
        &[(1, 1, 0, 4)],
    ));
    cases.push(scheduler_case(
        "narrow-store",
        "st.global.u16 [%p1+2], 0;",
        &[(1, 1, 2, 2)],
    ));
    cases.push(scheduler_case(
        "local-store",
        "mov.u32 %r0, 4; st.local.u32 [%r0], 0;",
        &[(4, 0, 4, 4)],
    ));
    cases.push(scheduler_copy_case("cp-async", "cp.async.ca.shared.global [scratch], [%p0], 16; cp.async.commit_group; cp.async.wait_group 0;", false));
    cases.push(scheduler_copy_case("cp-async-frozen", "cp.async.ca.shared.global [scratch], [%p0], 16; st.global.u32 [%p0], 0; cp.async.commit_group; cp.async.wait_group 0;", true));
    cases.push(scheduler_case(
        "atomic-exchange",
        "atom.global.exch.b32 %r0, [%p1], 0;",
        &[(1, 1, 0, 4)],
    ));
    cases.push(scheduler_case(
        "atomic-add-integer",
        "atom.global.add.u32 %r0, [%p1], 0;",
        &[(1, 1, 0, 4)],
    ));
    cases.push(scheduler_case(
        "atomic-reduction-integer",
        "red.global.add.u32 [%p1], 0;",
        &[(1, 1, 0, 4)],
    ));
    cases.push(scheduler_case(
        "atomic-add-f32",
        "atom.global.add.f32 %r0, [%p1], 0f00000000;",
        &[(1, 1, 0, 4)],
    ));
    cases.push(scheduler_case(
        "atomic-reduction-f32",
        "red.global.add.f32 [%p1], 0f00000000;",
        &[(1, 1, 0, 4)],
    ));
    cases.push(scheduler_case(
        "atomic-exchange-64",
        "atom.global.exch.b64 %p0, [%p1], 0;",
        &[(1, 1, 0, 8)],
    ));
    cases.push(scheduler_case("cp-async-zero-capture", "cp.async.ca.shared.global [scratch], [%p0], 16, 0; cp.async.commit_group; cp.async.wait_group 0;", &[(0,0,1024,16)]));
    cases.push(bulk_case(false));
    cases.push(bulk_case(true));
    cases.push(tma_case(false));
    cases.push(tma_case(true));
    for width in [4, 8] {
        cases.push(cas_case(true, width));
        cases.push(cas_case(false, width));
    }
    cases.push(tracking_disabled_case());
    cases.push(scheduler_tmem_case());
    for case in &mut cases {
        if case.get("half_tracking").is_none() {
            case["half_tracking"] = json!(true);
        }
        case["numeric_tracking"] = json!(true);
        check_markers(case);
        for state in ["cells", "expected"] {
            for cell in case[state].as_array().unwrap() {
                if !cell["word"].is_null() {
                    assert_eq!(
                        cell["key"][2].as_u64().unwrap() % 4,
                        0,
                        "{}: production aligned-history invariant",
                        case["name"]
                    );
                }
            }
        }
    }
    let boundary = copy_case(
        "unreachable-unaligned-history-boundary",
        (0, 0, 1),
        (0, 0, 4),
        4,
        false,
        true,
        false,
        0,
    );
    std::fs::write(
        format!("{output}.unaligned.json"),
        serde_json::to_vec_pretty(&json!({"cases":[boundary]})).unwrap(),
    )
    .unwrap();
    let mut wide = wide_atomic_boundary();
    wide["half_tracking"] = json!(true);
    wide["numeric_tracking"] = json!(true);
    check_markers(&wide);
    std::fs::write(
        format!("{output}.wide-atomic.json"),
        serde_json::to_vec_pretty(&json!({"cases":[wide.clone()]})).unwrap(),
    )
    .unwrap();
    if std::env::var("PTXV_ALLOW_KNOWN_WIDE_ATOMIC_BUG").as_deref() != Ok("1") {
        cases.push(wide);
    }
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&json!({"cases":cases})).unwrap(),
    )
    .unwrap();
    println!(
        "exported {} real production memory-effect cases to {output}",
        cases.len()
    );
}
