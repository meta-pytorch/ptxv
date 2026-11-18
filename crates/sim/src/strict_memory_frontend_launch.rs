/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_case::Buffer;
use ptxv_case::Manifest;
use ptxv_memory_write::execution_core;
use ptxv_memory_write::machine_registers::LaneRegisters;
use ptxv_memory_write::value_flow;

use crate::strict_memory_frontend::token_equals;
use crate::strict_memory_frontend_parse::SourceProgram;
use crate::strict_memory_frontend_parse::option_value;
use crate::strict_memory_frontend_parse::parse_source;

pub fn bytes_equal(left: &[u8], right: &[u8]) -> bool {
    let mut same = left.len() == right.len();
    let mut index = 0;
    while same && index < left.len() {
        same = left[index] == right[index];
        index += 1;
    }
    same
}

pub fn manifest_supported(manifest: &Manifest) -> bool {
    manifest.cubin.is_none()
        && manifest.executed_cubin.is_none()
        && manifest.dynamic_smem == 0
        && manifest.cluster.is_none()
        && manifest.preferred_cluster.is_none()
        && (manifest.replicas.len() == 0)
        && manifest.spec.is_none()
        && manifest.mma_model.is_none()
        && manifest.exit_fence.is_none()
        && !manifest.poll_acquire
        && (manifest.assume_refuted.len() == 0)
        && !manifest.smem_cumulative
        && !manifest.hw_l1
        && !manifest.l1_coresident
        && !manifest.l1_havoc
        && manifest.residency.is_none()
        && !manifest.cooperative
        && !manifest.tmem_cotenant
        && !manifest.read_done
        && !manifest.remote_fifo
        && (manifest.domains.len() == 0)
        && (manifest.nic_qps.len() == 0)
        && (manifest.nic_mrs.len() == 0)
        && (manifest.multicast.len() == 0)
        && manifest.mc_alias.is_none()
        && (manifest.views.len() == 0)
        && (manifest.spaces.len() == 0)
        && manifest.alias_xspace.is_none()
        && (manifest.params.len() == 0)
        && (manifest.launches.len() == 0)
        && (manifest.permuted.len() == 0)
}

pub fn dimensions_product(dimensions: [u32; 3]) -> Option<u64> {
    if dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] == 0 {
        return None;
    }
    match (dimensions[0] as u64).checked_mul(dimensions[1] as u64) {
        Some(first) => first.checked_mul(dimensions[2] as u64),
        None => None,
    }
}

pub fn coordinates(index: u64, dimensions: [u32; 3]) -> Option<[u32; 3]> {
    let count = option_value!(dimensions_product(dimensions));
    if index >= count {
        return None;
    }
    let x = index % dimensions[0] as u64;
    let remainder = index / dimensions[0] as u64;
    let y = remainder % dimensions[1] as u64;
    let z = remainder / dimensions[1] as u64;
    Some([x as u32, y as u32, z as u32])
}

pub fn bitmap_words(count: u16) -> usize {
    count as usize / 64 + if count % 64 == 0 { 0 } else { 1 }
}

pub fn empty_registers(virtuals: u16, predicates: u16) -> value_flow::Registers {
    let mut regs = Vec::new();
    let mut pointers = Vec::new();
    let mut index = 0usize;
    while index < virtuals as usize {
        regs.push(0);
        pointers.push(None);
        index += 1;
    }
    let mut written = Vec::new();
    index = 0;
    while index < bitmap_words(virtuals) {
        written.push(0);
        index += 1;
    }
    let mut preds = Vec::new();
    let mut pdef = Vec::new();
    index = 0;
    while index < bitmap_words(predicates) {
        preds.push(0);
        pdef.push(0);
        index += 1;
    }
    value_flow::Registers {
        raw: LaneRegisters {
            regs,
            preds,
            pdef,
            written,
            vund: Vec::new(),
        },
        pointers,
    }
}

pub fn initial_threads(
    manifest: &Manifest,
    virtuals: u16,
    predicates: u16,
) -> Option<Vec<execution_core::ThreadState>> {
    if manifest.grid[0] > 2_147_483_647
        || manifest.grid[1] > 65_535
        || manifest.grid[2] > 65_535
        || manifest.block[0] > 1024
        || manifest.block[1] > 1024
        || manifest.block[2] > 64
    {
        return None;
    }
    let blocks = option_value!(dimensions_product(manifest.grid));
    let block_threads = option_value!(dimensions_product(manifest.block));
    let count = option_value!(blocks.checked_mul(block_threads));
    if count > 1_048_576 || block_threads > 1024 {
        return None;
    }
    let words = virtuals as u64 + predicates as u64 + 1;
    if option_value!(count.checked_mul(words)) > 1_048_576 {
        return None;
    }
    let mut threads = Vec::new();
    let mut index = 0;
    let mut valid = true;
    while valid && index < count {
        match (
            coordinates(index / block_threads, manifest.grid),
            coordinates(index % block_threads, manifest.block),
        ) {
            (Some(ctaid), Some(tid)) => {
                threads.push(execution_core::ThreadState {
                    pc: 0,
                    running: true,
                    registers: empty_registers(virtuals, predicates),
                    ids: value_flow::ThreadIds {
                        tid,
                        ctaid,
                        ntid: manifest.block,
                        nctaid: manifest.grid,
                        lane: ((index % block_threads) % 32) as u32,
                    },
                });
            }
            _ => valid = false,
        }
        index += 1;
    }
    if valid { Some(threads) } else { None }
}

pub fn dtype_bytes(dtype: &[u8]) -> Option<u64> {
    if bytes_equal(dtype, b"u8") || bytes_equal(dtype, b"i8") {
        Some(1)
    } else if bytes_equal(dtype, b"u16") || bytes_equal(dtype, b"i16") {
        Some(2)
    } else if bytes_equal(dtype, b"u32") || bytes_equal(dtype, b"i32") {
        Some(4)
    } else if bytes_equal(dtype, b"u64") || bytes_equal(dtype, b"i64") {
        Some(8)
    } else {
        None
    }
}

pub fn buffer_extent(buffer: &Buffer) -> Option<u64> {
    let mut bytes = dtype_bytes(buffer.dtype.as_bytes());
    let mut index = 0;
    while bytes.is_some() && index < buffer.shape.len() {
        bytes = match bytes {
            Some(before) => before.checked_mul(buffer.shape[index]),
            None => None,
        };
        index += 1;
    }
    bytes
}

pub fn buffer_supported(buffer: &Buffer) -> bool {
    buffer.device == 0
        && buffer.symbol.is_some()
        && buffer.cubin.is_none()
        && (!buffer.scratch || bytes_equal(buffer.init.as_bytes(), b"undef"))
        && buffer.file.is_none()
        && buffer.expect.is_none()
        && buffer.expect_offset == 0
        && (buffer.expect_ignore.len() == 0)
        && (buffer.values.len() == 0)
        && (buffer.ptrs.len() == 0)
        && buffer.invariant.is_none()
        && (bytes_equal(buffer.init.as_bytes(), b"zeros")
            || bytes_equal(buffer.init.as_bytes(), b"undef"))
}

pub fn global_named(source: &[u8], program: &SourceProgram, symbol: &[u8]) -> Option<usize> {
    let mut found = usize::MAX;
    let mut unique = true;
    let mut index = 0;
    while unique && index < program.scope.globals.len() {
        let name = program.scope.globals[index].name;
        if token_equals(source, name, symbol) {
            unique = found == usize::MAX;
            found = index;
        }
        index += 1;
    }
    if unique && found != usize::MAX {
        Some(found)
    } else {
        None
    }
}

pub fn buffer_object(source: &[u8], program: &SourceProgram, buffer: &Buffer) -> Option<usize> {
    match &buffer.symbol {
        Some(symbol) => global_named(source, program, symbol.as_bytes()),
        None => None,
    }
}

pub fn buffer_index(
    source: &[u8],
    program: &SourceProgram,
    manifest: &Manifest,
    object: usize,
) -> Option<Option<usize>> {
    let mut found = usize::MAX;
    let mut valid = true;
    let mut index = 0;
    while valid && index < manifest.buffers.len() {
        match buffer_object(source, program, &manifest.buffers[index]) {
            Some(target) => {
                if target == object {
                    valid = found == usize::MAX;
                    found = index;
                }
            }
            None => valid = false,
        }
        index += 1;
    }
    if !valid {
        None
    } else if found == usize::MAX {
        Some(None)
    } else {
        Some(Some(found))
    }
}

pub fn initial_allocation(
    source: &[u8],
    program: &SourceProgram,
    manifest: &Manifest,
    object: usize,
) -> Option<execution_core::AllocationBytes> {
    if object >= program.scope.globals.len() {
        return None;
    }
    let backing = option_value!(buffer_index(source, program, manifest, object));
    let original = &program.scope.globals[object].bytes;
    if original.len() > 67_108_864 {
        return None;
    }
    let mut explicit = true;
    let mut initialized = true;
    if let Some(index) = backing {
        let buffer = &manifest.buffers[index];
        if !buffer_supported(buffer)
            || option_value!(buffer_extent(buffer)) != original.len() as u64
        {
            return None;
        }
        initialized = bytes_equal(buffer.init.as_bytes(), b"zeros");
        explicit = false;
    }
    let mut bytes = Vec::new();
    let mut defined = Vec::new();
    let mut index = 0;
    while index < original.len() {
        bytes.push(if explicit { original[index] } else { 0 });
        defined.push(initialized);
        index += 1;
    }
    Some(execution_core::AllocationBytes {
        bytes,
        defined,
        live: true,
    })
}

pub struct InitializedObject {
    pub layout: value_flow::AllocationLayout,
    pub allocation: execution_core::AllocationBytes,
    pub next: u64,
    pub total: u64,
}

pub fn initial_object(
    source: &[u8],
    program: &SourceProgram,
    manifest: &Manifest,
    object: usize,
    cursor: u64,
    total: u64,
) -> Option<InitializedObject> {
    if object >= program.scope.globals.len() {
        return None;
    }
    let global = &program.scope.globals[object];
    let alignment = global.alignment;
    if alignment == 0 {
        return None;
    }
    let bytes = global.bytes.len() as u64;
    let rounded = option_value!(cursor.checked_add(alignment - 1));
    let base = rounded / alignment * alignment;
    let allocation = option_value!(initial_allocation(source, program, manifest, object));
    let next = option_value!(base.checked_add(bytes));
    let total = option_value!(total.checked_add(bytes));
    if total > 67_108_864 {
        return None;
    }
    Some(InitializedObject {
        layout: value_flow::AllocationLayout {
            base,
            bytes,
            alignment,
        },
        allocation,
        next,
        total,
    })
}

pub fn initial_allocations(
    source: &[u8],
    program: &SourceProgram,
    manifest: &Manifest,
) -> Option<(
    Vec<value_flow::AllocationLayout>,
    Vec<execution_core::AllocationBytes>,
)> {
    let mut layouts = Vec::new();
    let mut allocations = Vec::new();
    let mut position = 0usize;
    let mut base = 0x7000_0000_0000u64;
    let mut valid = true;
    let mut total = 0u64;
    while valid && position < program.scope.globals.len() {
        match initial_object(source, program, manifest, position, base, total) {
            Some(initialized) => {
                layouts.push(initialized.layout);
                allocations.push(initialized.allocation);
                base = initialized.next;
                total = initialized.total;
            }
            None => valid = false,
        }
        position += 1;
    }
    if valid {
        Some((layouts, allocations))
    } else {
        None
    }
}

pub fn resolve_output(
    source: &[u8],
    program: &SourceProgram,
    manifest: &Manifest,
    name: &[u8],
) -> Option<usize> {
    let mut found = usize::MAX;
    let mut unique = true;
    let mut index = 0;
    while unique && index < manifest.buffers.len() {
        if bytes_equal(manifest.buffers[index].name.as_bytes(), name) {
            unique = found == usize::MAX;
            match buffer_object(source, program, &manifest.buffers[index]) {
                Some(object) => found = object,
                None => unique = false,
            }
        }
        index += 1;
    }
    if unique && found != usize::MAX {
        Some(found)
    } else {
        None
    }
}

pub fn outputs(source: &[u8], program: &SourceProgram, manifest: &Manifest) -> Option<Vec<usize>> {
    let mut outputs = Vec::new();
    let mut valid = true;
    let mut index = 0;
    while valid && index < manifest.outputs.len() {
        match resolve_output(
            source,
            program,
            manifest,
            manifest.outputs[index].as_bytes(),
        ) {
            Some(object) => outputs.push(object),
            None => valid = false,
        }
        index += 1;
    }
    if valid { Some(outputs) } else { None }
}

pub fn buffer_extent_matches(buffer: &Buffer, bytes: u64) -> bool {
    match buffer_extent(buffer) {
        Some(extent) => extent == bytes,
        None => false,
    }
}

pub fn buffer_name_fresh(manifest: &Manifest, index: usize) -> bool {
    let mut earlier = 0;
    let mut valid = true;
    while valid && earlier < index {
        valid = !bytes_equal(
            manifest.buffers[index].name.as_bytes(),
            manifest.buffers[earlier].name.as_bytes(),
        );
        earlier += 1;
    }
    valid
}

pub fn buffer_matches(source: &[u8], program: &SourceProgram, buffer: &Buffer) -> bool {
    if !buffer_supported(buffer) {
        return false;
    }
    match buffer_object(source, program, buffer) {
        Some(object) => {
            buffer_extent_matches(buffer, program.scope.globals[object].bytes.len() as u64)
        }
        None => false,
    }
}

pub fn buffers_supported(source: &[u8], program: &SourceProgram, manifest: &Manifest) -> bool {
    let mut valid = true;
    let mut index = 0;
    while valid && index < manifest.buffers.len() {
        valid = buffer_matches(source, program, &manifest.buffers[index])
            && buffer_name_fresh(manifest, index);
        index += 1;
    }
    valid
}

pub fn build_program(
    source: &[u8],
    entry: &[u8],
    manifest: &Manifest,
) -> Option<execution_core::Program> {
    if source.len() > 1_048_576
        || !manifest_supported(manifest)
        || !bytes_equal(entry, manifest.entry.as_bytes())
    {
        return None;
    }
    let parsed = option_value!(parse_source(source, entry));
    if !buffers_supported(source, &parsed, manifest) {
        return None;
    }
    let threads = option_value!(initial_threads(
        manifest,
        parsed.scope.virtual_count,
        parsed.scope.predicate_count,
    ));
    let (layouts, allocations) = option_value!(initial_allocations(source, &parsed, manifest));
    let outputs = option_value!(outputs(source, &parsed, manifest));
    Some(execution_core::Program {
        code: parsed.code,
        layouts,
        initial: execution_core::State {
            threads,
            allocations,
        },
        outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(body: &str) -> String {
        format!(
            ".version 9.0 .target sm_103a .address_size 64 .global .align 8 .u64 x[2]={{11,23}}; .visible .entry k() {{ .reg .b32 %r<4>; .reg .b64 %rd<4>; .reg .pred %p<8>; {body} }}"
        )
    }

    fn manifest(source: &str) -> Manifest {
        serde_json::from_value(serde_json::json!({
            "name": "strict", "description": "source-owned construction", "ptx": source,
            "entry": "k", "grid": [1,1,1], "block": [1,1,1], "outputs": [],
            "buffers": [], "params": [], "variants": []
        }))
        .unwrap()
    }

    fn backing(name: &str, symbol: &str, elements: u64, init: &str) -> Buffer {
        serde_json::from_value(serde_json::json!({
            "name": name, "dtype": "u64", "shape": [elements], "init": init, "symbol": symbol
        }))
        .unwrap()
    }

    #[test]
    fn object_construction_rejects_wrapping_placements_and_preserves_literal_bytes() {
        let raw = source("ret;");
        let input = manifest(&raw);
        let mut parsed = parse_source(raw.as_bytes(), b"k").unwrap();
        let object = initial_object(raw.as_bytes(), &parsed, &input, 0, 3, 0).unwrap();
        assert_eq!(object.layout.base, 8);
        assert_eq!(object.layout.bytes, 16);
        assert_eq!(object.layout.alignment, 8);
        assert_eq!(object.next, 24);
        assert_eq!(object.total, 16);
        assert_eq!(&object.allocation.bytes[..8], &11u64.to_le_bytes());
        assert_eq!(&object.allocation.bytes[8..], &23u64.to_le_bytes());
        assert!(object.allocation.defined.iter().all(|defined| *defined));
        assert!(initial_object(raw.as_bytes(), &parsed, &input, 0, u64::MAX, 0).is_none());
        assert!(initial_object(raw.as_bytes(), &parsed, &input, 0, u64::MAX - 7, 0).is_none());
        assert!(initial_object(raw.as_bytes(), &parsed, &input, 0, 0, 67_108_848).is_some());
        assert!(initial_object(raw.as_bytes(), &parsed, &input, 0, 0, 67_108_849).is_none());
        parsed.scope.globals[0].alignment = 0;
        assert!(initial_object(raw.as_bytes(), &parsed, &input, 0, 0, 0).is_none());
    }

    #[test]
    fn source_constructs_the_actual_native_executable_program() {
        let raw =
            source("mov.u64 %rd0,x; ld.global.u64 %rd1,[%rd0+8]; st.global.u64 [%rd0],%rd1; ret;");
        let mut selected = manifest(&raw);
        selected.ptx = "kernel.ptx".into();
        selected.variants = serde_json::from_value(serde_json::json!([
            {"id":"selected", "expect":["clean"], "replace":"irrelevant-unapplied-metadata"}
        ]))
        .unwrap();
        let program = build_program(raw.as_bytes(), b"k", &selected).unwrap();
        let mut state = execution_core::initial_state(&program).unwrap();
        while !execution_core::terminal(&state) {
            execution_core::execute(&program, &mut state, 0).unwrap();
        }
        assert_eq!(&state.allocations[0].bytes[..8], &23u64.to_le_bytes());
        assert!(state.allocations[0].defined.iter().all(|defined| *defined));
    }

    #[test]
    fn every_three_dimensional_actor_is_present_and_lane_is_cta_linear() {
        let raw = source("ret;");
        let mut input = manifest(&raw);
        input.grid = [2, 3, 2];
        input.block = [3, 2, 3];
        let program = build_program(raw.as_bytes(), b"k", &input).unwrap();
        let ids: std::collections::HashSet<_> = program
            .initial
            .threads
            .iter()
            .map(|thread| (thread.ids.ctaid, thread.ids.tid))
            .collect();
        assert_eq!(ids.len(), 12 * 18);
        assert_eq!(program.initial.threads.len(), 12 * 18);
        for thread in &program.initial.threads {
            let t = thread.ids;
            assert!(t.tid[0] < 3 && t.tid[1] < 2 && t.tid[2] < 3);
            assert!(t.ctaid[0] < 2 && t.ctaid[1] < 3 && t.ctaid[2] < 2);
            assert_eq!(t.lane, (t.tid[0] + 3 * (t.tid[1] + 2 * t.tid[2])) % 32);
            assert_eq!(t.ntid, input.block);
            assert_eq!(t.nctaid, input.grid);
        }
        assert_eq!(program.initial.threads[17].ids.tid, [2, 1, 2]);
        assert_eq!(program.initial.threads[18].ids.ctaid, [1, 0, 0]);
        assert_eq!(program.initial.threads[18].ids.tid, [0, 0, 0]);
        assert!(execution_core::initial_state(&program).is_ok());
        input.grid = [1, 1, 1];
        input.block = [1, 1, 1024];
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
        input.block = [1, 1, 1];
        input.grid = [1, 65_536, 1];
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
        input.block[0] = 0;
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
    }

    #[test]
    fn captured_backing_cannot_widen_a_declared_global_object() {
        let raw = source("ret;");
        let mut input = manifest(&raw);
        input.buffers.push(backing("captured", "x", 3, "zeros"));
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
        input.buffers[0].shape = vec![2];
        input.outputs.push("captured".into());
        let program = build_program(raw.as_bytes(), b"k", &input).unwrap();
        assert_eq!(program.outputs, [0]);
        assert_eq!(program.layouts[0].bytes, 16);
        assert_eq!(program.initial.allocations[0].bytes, vec![0; 16]);
        input.buffers[0].init = "undef".into();
        input.buffers[0].scratch = true;
        let program = build_program(raw.as_bytes(), b"k", &input).unwrap();
        assert!(
            program.initial.allocations[0]
                .defined
                .iter()
                .all(|defined| !defined)
        );
        input.outputs[0] = "missing".into();
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
    }

    #[test]
    fn synthetic_cross_object_escape_remains_an_out_of_bounds_fault() {
        let raw = source("ld.global.u64 %rd0,[x+4294967296]; ret;")
            .replace(".visible", ".global .align 8 .u64 y[1]={99}; .visible");
        let mut program = build_program(raw.as_bytes(), b"k", &manifest(&raw)).unwrap();
        program.layouts[1].base = program.layouts[0].base + 4294967296;
        let mut state = execution_core::initial_state(&program).unwrap();
        assert_eq!(
            execution_core::execute(&program, &mut state, 0),
            Err(execution_core::Fault::OutOfBounds)
        );
    }

    #[test]
    fn declared_alignment_is_preserved_when_synthetic_base_is_more_aligned() {
        let raw = source("ld.global.u64 %rd0,[x]; ret;").replace(
            ".align 8 .u64 x[2]={11,23}",
            ".align 4 .u32 x[4]={11,0,23,0}",
        );
        let program = build_program(raw.as_bytes(), b"k", &manifest(&raw)).unwrap();
        assert_eq!(program.layouts[0].base % 8, 0);
        assert_eq!(program.layouts[0].alignment, 4);
        let mut state = execution_core::initial_state(&program).unwrap();
        assert_eq!(
            execution_core::execute(&program, &mut state, 0),
            Err(execution_core::Fault::Misaligned)
        );
    }

    #[test]
    fn refuses_unknown_initializers_entry_mismatch_and_unmodeled_launches() {
        let raw = source("ret;");
        let mut input = manifest(&raw);
        input.buffers.push(backing("captured", "x", 2, "bits"));
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
        input.buffers.clear();
        input.entry = "different_entry".into();
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
        input.entry = "k".into();
        input.poll_acquire = true;
        assert!(build_program(raw.as_bytes(), b"k", &input).is_none());
    }
}
