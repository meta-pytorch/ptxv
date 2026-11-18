#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in the repository root.
"""Extract actual IR memory-effect producers and check their independent contracts."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "lean_model/instruction_effects"
RUST = [
    "crates/ir/src/atomic_cancel.rs",
    "crates/ir/src/atomic_result_tests.rs",
    "crates/ir/src/atomic_result.rs",
    "crates/ir/src/guard_certificate.rs",
    "crates/ir/src/lib.rs",
    "crates/ir/src/register_owner.rs",
    "crates/ir/src/register_effect.rs",
    "crates/ir/src/read_window.rs",
    "crates/ir/src/store_window.rs",
    "crates/ir/src/masked_load.rs",
    "crates/ir/src/payload.rs",
    "crates/ir/src/metadata_delivery.rs",
    "crates/ir/src/comparison_select.rs",
    "crates/ir/src/destination_rows.rs",
    "crates/ir/src/control_flow.rs",
    "crates/ir/src/control_transfer.rs",
    "crates/ir/src/predicate_read.rs",
    "crates/ir/src/analysis_knowledge.rs",
]
MODULES = [
    "AtomicCancelWord",
    "AtomicCancelShape",
    "AtomicCancelRefinement",
    "AtomicCancelReads",
    "AtomicCancelCaller",
    "AtomicCancelScope",
    "AtomicCancelWrites",
    "AtomicCancelSuffix",
    "AtomicCancelSuffixPath",
    "AtomicResult",
    "OwnerModel",
    "OwnerRefinement",
    "OwnerComparison",
    "OwnerPredicate",
    "OwnerComparisonPlan",
    "StoreSourceRoles",
    "StoreRegisterMembership",
    "StoreDeferredRefinement",
    "StoreNarrowSinkRefinement",
    "StoreSourceRoleProducer",
    "Model",
    "SpanRefinement",
    "Refinement",
    "AddressRefinement",
    "Bounds",
    "MemoryRefinement",
    "AtomicRefinement",
    "CasRefinement",
    "MaskRefinement",
    "ProducerModel",
    "AtomicSourcesRefinement",
    "ControlDataRefinement",
    "ReadRefinement",
    "WriteRefinement",
    "NarrowRead",
    "StoreSourceRefinement",
    "ExecutionCoverage",
    "StoreWindowModel",
    "StoreWindowRefinement",
    "StoreWindowProducer",
]
MODULES += [
    "ControlFlowModel",
    "ControlFlowPrimitives",
    "ControlFlowRange",
    "ControlFlowRows",
    "ControlFlowRefinement",
    "ControlFlowGraphModel",
    "ControlFlowLookup",
    "ControlFlowGraphRefinement",
]
MODULES += ["ControlFlowStorage", "BarrierSeed", "BarrierSeedRows"]
MODULES += [
    "DestinationKeys",
    "DestinationRows",
    "DestinationRowsTotal",
    "PartialSuccess",
]
MODULES += ["GraphSeedSource"]
MODULES += ["PredicateRead", "PredicateKnowledge"]
MODULES += [
    "LookaheadControlModel",
    "LookaheadControlFalseGuard",
    "LookaheadControlFallthrough",
    "LookaheadControlGuardLifting",
]
MODULES += ["OwnerTemporal", "AnalysisKnowledge"]
MODULES += [
    "ComparisonSelectionModel",
    "ComparisonSelectionRefinement",
    "ComparisonSelectionProducer",
    "ComparisonSelectionOperands",
    "ComparisonSelectionPair",
    "ComparisonSelectionSoundness",
    "ComparisonRealMax",
]
MODULES += [
    "SelectionProgram/Code/Types",
    "SelectionProgram/Code/Funs",
    "SelectionProgram/Model",
    "SelectionProgram/Shape",
    "SelectionProgram/Semantics",
    "SelectionProgram/MachineControl",
    "SelectionProgram/Composition",
]
MODULES += [
    "SelectionEvaluator/Model",
    "SelectionEvaluator/Refinement",
    "SelectionEvaluator/Observations",
    "SelectionEvaluator/Composition",
]
MODULES += [
    "SelectionBindings/Model",
    "SelectionBindings/Reads",
    "SelectionBindings/Lookup",
    "SelectionBindings/Wrappers",
    "SelectionBindings/Execution",
    "SelectionBindings/Bounds",
    "SelectionBindings/Composition",
]
MODULES += [
    "SelectionDriver/Model",
    "SelectionDriver/Refinement",
    "SelectionDriver/Selected",
    "SelectionDriver/Observations",
    "SelectionDriver/Composition",
]
MODULES += [
    "SelectionBound/Model",
    "SelectionBound/Eligibility",
    "SelectionBound/Reads",
    "SelectionBound/Semantics",
    "SelectionBound/Driver",
    "SelectionBound/Composition",
]
MODULES += [
    "GuardCertificateAssignment",
    "GuardCertificateCheck",
    "GuardCertificateComparison",
    "GuardCertificateControl",
    "GuardCertificateDecision",
    "GuardCertificateFacts",
    "GuardCertificateFrame",
    "GuardCertificateLookup",
    "GuardCertificateMove",
    "GuardCertificateOperands",
    "GuardCertificatePaths",
    "GuardCertificateSpan",
    "GuardCertificateTransfer",
    "GuardCertificateValidation",
    "GuardCertificateWords",
    "GuardCertificateWritten",
    "GuardCertificatePublication",
    "GuardCertificatePublicationSource",
    "GuardCertificateRowsMerge",
    "GuardCertificateRowsProtocol",
    "GuardCertificateRowsPublish",
    "GuardCertificateRowsTruth",
    "GuardCertificateDefinedAssignment",
    "GuardCertificateDefinedFacts",
    "GuardCertificateDefinedPaths",
    "GuardCertificateDefinedPublication",
    "GuardCertificateDefinedTransfer",
    "GuardCertificateUnknownGuard",
]
MODULES += [
    "IEEEComparison/Model",
    "IEEEComparison/Magnitude",
    "IEEEComparison/Representation",
    "IEEEComparison/Facts",
    "IEEEComparison/Comparison",
    "IEEEComparison/Keys",
    "IEEEComparison/Refinement",
    "IEEEComparison/Primitive",
    "IEEEComparison/Composition",
]
MODULES += [
    "SelectionHardware/Model",
    "SelectionHardware/Refinement",
    "SelectionHardware/Composition",
]
MODULES += [
    "MetadataDeliveryModel",
    "MetadataThreads",
    "MetadataDeliveryLoops",
    "MetadataDeliveryDestinations",
    "MetadataDeliveryProducer",
    "MetadataDeliveryFrame",
    "MetadataDeliveryProperties",
]
MODULES += [
    "PayloadAssembly",
    "PayloadByteRefinement",
    "PayloadInitialization",
    "PayloadModel",
    "PayloadProducer",
    "PayloadReadCoverage",
    "PayloadSupportModel",
    "PayloadSupportRefinement",
    "PayloadValidity",
    "PayloadWindows",
    "St256Model",
    "St256Producer",
    "St256Refinement",
    "PayloadFailure",
    "PayloadFailureProducer",
]
MODULES += [
    "MaskedLoadBit",
    "MaskedLoadFrame",
    "MaskedLoadLoop",
    "MaskedLoadModel",
    "MaskedLoadProducer",
    "MaskedLoadWord",
]
MODULES += ["AnalysisReadAdmission", "AnalysisReadComposition"]
MODULES += ["LookaheadControlReturnValidation"]
MODULES += ["StoreSyntheticDiscard"]
LEAN = (
    ["lakefile.lean", "lean-toolchain", "Instruction.lean", "Audit.lean"]
    + [f"Instruction/{name}.lean" for name in MODULES]
    + ["Instruction/Code/Types.lean", "Instruction/Code/Funs.lean"]
)
ENTRIES = [
    "atomic_cancel::cancelled_return",
    "atomic_cancel::certificate_candidate",
    "atomic_cancel::cancels",
    "atomic_cancel::linear_private",
    "atomic_cancel::reads_return",
    "atomic_cancel::overwrites_return",
    "atomic_cancel::plain_atom",
    "atomic_result::local_order_monitored",
    "atomic_result::observed",
    "register_owner::uniform_owners",
    "register_owner::canonical_key",
    "register_owner::record_aliases",
    "register_owner::normalize_predicate",
    "register_owner::comparison_plan",
    "store_window::metadata_sources",
    "register_effect::memory_reads",
    "register_effect::memory_reads_for_front",
    "register_effect::memory_writes",
    "register_effect::memory_data_reads",
    "register_effect::memory_control_reads",
    "read_window::narrow",
    "store_window::sources",
    "masked_load::execute",
    "payload::from_operation",
    "payload::from_st256",
    "metadata_delivery::execute",
    "comparison_select::capture",
    "comparison_select::from_pair",
    "comparison_select::from_operation",
    "comparison_select::operations",
    "comparison_select::evaluate",
    "comparison_select::real_max",
    "destination_rows::collect",
    "destination_rows::keys",
    "control_flow::successors",
    "control_flow::graph_successors",
    "control_transfer::from_operation",
    "control_transfer::with_fallthrough",
    "predicate_read::evaluate",
    "register_owner::is_thread_private",
    "analysis_knowledge::predicate",
    "analysis_knowledge::value_is_thread_private",
    "control_flow::barrier_seed",
    "control_flow::barrier_seeds",
    "control_flow::graph_and_seeds",
]
ENTRIES += [
    "guard_certificate::validate",
    "guard_certificate::post_value",
    "guard_certificate::decision",
    "guard_certificate::frame",
    "guard_certificate::merge_grid",
    "guard_certificate::branch_fold",
    "guard_certificate::merge_rows",
    "guard_certificate::publish",
]
ENTRIES += [
    "analysis_knowledge::reads_are_thread_private",
    "analysis_knowledge::direct_reads_are_thread_private",
]
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
MUTATIONS = [
    (
        "atomic-cancellation-accepts-add",
        "src/atomic_cancel.rs",
        "op: IntOp::Sub | IntOp::Xor",
        "op: IntOp::Add | IntOp::Xor",
        "Instruction.AtomicCancelRefinement",
    ),
    (
        "store-defers-control-register",
        "src/store_window.rs",
        "if !has_register(control, register) {",
        "if true {",
        "Instruction.StoreDeferredRefinement",
    ),
    (
        "store-narrows-control-register",
        "src/store_window.rs",
        "register.file == File::V || has_register(control, register)",
        "register.file == File::V",
        "Instruction.StoreNarrowSinkRefinement",
    ),
    (
        "store-half-mask-uses-byte-mask",
        "src/store_window.rs",
        "Some((*first, 65535))",
        "Some((*first, 255))",
        "Instruction.StoreNarrowSinkRefinement",
    ),
    (
        "load-forgot-vector-count",
        "src/register_effect.rs",
        "let count = load_count(d.file, *bytes, *n);",
        "let count = load_count(d.file, *bytes, 1);",
        "Instruction.Refinement",
    ),
    (
        "store-forgot-vector-elements",
        "src/register_effect.rs",
        "address(&mut result, addr);\n            store_sources(&mut result, v, *bytes, false);",
        "address(&mut result, addr);\n            slots(&mut result, v);",
        "Instruction.MemoryRefinement",
    ),
    (
        "async-store-uses-plain-width",
        "src/register_effect.rs",
        "slots(&mut result, mbar);\n            store_sources(&mut result, v, *bytes, true);",
        "slots(&mut result, mbar);\n            store_sources(&mut result, v, *bytes, false);",
        "Instruction.MemoryRefinement",
    ),
    (
        "store-window-uses-physical-stride",
        "src/store_window.rs",
        "let start = index as u32 * stride;",
        "let start = index as u32 * 4;",
        "Instruction.StoreWindowRefinement",
    ),
    (
        "cas-forgot-implicit-replacement",
        "src/register_effect.rs",
        "implicit_cas_sources(result, bytes, v);",
        "implicit_cas_sources(result, 0, v);",
        "Instruction.AtomicSourcesRefinement",
    ),
    (
        "narrow-conversion-ignores-high-half",
        "src/read_window.rs",
        "=> Some((a, if *hi { 16 } else { 0 }, 16)),",
        "=> Some((a, if *hi { 0 } else { 16 }, 16)),",
        "Instruction.NarrowRead",
    ),
    (
        "masked-load-ignores-mask",
        "src/register_effect.rs",
        "let (low, high) = selected_halves(*mask);",
        "let (low, high) = selected_halves(255);",
        "Instruction.MaskRefinement",
    ),
]

MUTATIONS += [
    (
        "uniform_predicate_uses_lane_owner",
        "src/register_owner.rs",
        "File::UR | File::UP if thread < owners.len()",
        "File::UR if thread < owners.len()",
        "Instruction.OwnerRefinement",
    ),
    (
        "repeated_warp_uses_current_thread",
        "src/register_owner.rs",
        "Some(owner) => owner,",
        "Some(_owner) => thread,",
        "Instruction.OwnerRefinement",
    ),
    (
        "register_file_namespace_erased",
        "src/register_owner.rs",
        "(owner, register.file as u8, register.idx)",
        "(owner, 0, register.idx)",
        "Instruction.OwnerRefinement",
    ),
    (
        "leader_id_narrows_to_eight_bits",
        "src/register_owner.rs",
        "first[warp] = Some(thread);",
        "first[warp] = Some(thread as u8 as u32);",
        "Instruction.OwnerRefinement",
    ),
    (
        "first_invalid_warp_is_accepted",
        "src/register_owner.rs",
        "if warp >= warp_count {",
        "if warp > warp_count {",
        "Instruction.OwnerRefinement",
    ),
    (
        "record_file_namespace_ignored",
        "src/register_owner.rs",
        "if register.file as u8 != record.1 || register.idx != record.2 {",
        "if register.idx != record.2 {",
        "Instruction.OwnerRefinement",
    ),
    (
        "record_index_ignored",
        "src/register_owner.rs",
        "if register.file as u8 != record.1 || register.idx != record.2 {",
        "if register.file as u8 != record.1 {",
        "Instruction.OwnerRefinement",
    ),
    (
        "cached_record_uses_lane_owner",
        "src/register_owner.rs",
        "let previous = canonical_key(record.0 as usize, owners, register);",
        "let previous = (record.0, register.file as u8, register.idx);",
        "Instruction.OwnerRefinement",
    ),
    (
        "cached_record_thread_narrows_to_eight_bits",
        "src/register_owner.rs",
        "canonical_key(record.0 as usize, owners, register)",
        "canonical_key(record.0 as u8 as usize, owners, register)",
        "Instruction.OwnerRefinement",
    ),
]

MUTATIONS += [
    (
        "guarded_brx_fallthrough_erased",
        "src/control_flow.rs",
        "append(&mut result, targets);\n            if !always {",
        "append(&mut result, targets);\n            if false {",
        "Instruction.ControlFlowRefinement",
    ),
    (
        "secondary_condition_erased",
        "src/control_flow.rs",
        "if always && !*tid0_falls && cond.is_none() && div.is_none() {",
        "if always && !*tid0_falls && div.is_none() {",
        "Instruction.ControlFlowRefinement",
    ),
    (
        "false_fold_branch_taken",
        "src/control_flow.rs",
        "Some(false) => append_next(&mut result, &context),",
        "Some(false) => result.push(*target),",
        "Instruction.ControlFlowRefinement",
    ),
    (
        "guarded_return_fallthrough_erased",
        "src/control_flow.rs",
        "if !always {\n"
        "                append_next(&mut result, &context);\n"
        "            }\n"
        "        }\n"
        "        Op::Brx",
        "if false {\n"
        "                append_next(&mut result, &context);\n"
        "            }\n"
        "        }\n"
        "        Op::Brx",
        "Instruction.ControlFlowRefinement",
    ),
    (
        "brx_target_erased",
        "src/control_flow.rs",
        "append(&mut result, targets);",
        "let _ = targets; append(&mut result, &[]);",
        "Instruction.ControlFlowRefinement",
    ),
    (
        "graph_guard_replaced_true",
        "src/control_flow.rs",
        "            code[pc].guard,",
        "            Pred::TRUE,",
        "Instruction.ControlFlowGraphRefinement",
    ),
]

MUTATIONS += [
    (
        "ptx_predicate_seven_is_erased",
        "src/register_owner.rs",
        "front == crate::Front::Sass && register.idx == 7",
        "front == crate::Front::Ptx && register.idx == 7",
        "Instruction.OwnerPredicate",
    ),
    (
        "wrong_uniform_predicate_is_erased",
        "src/register_owner.rs",
        "crate::File::UP => register.idx == 7,",
        "crate::File::UP => register.idx == 6,",
        "Instruction.OwnerPredicate",
    ),
    (
        "constant_predicate_negation_is_flipped",
        "src/register_owner.rs",
        "reg: None,\n            neg: predicate.neg,",
        "reg: None,\n            neg: !predicate.neg,",
        "Instruction.OwnerPredicate",
    ),
    (
        "constant_predicate_retains_register_evidence",
        "src/register_owner.rs",
        "reg: None,\n            neg: predicate.neg,",
        "reg: predicate.reg,\n            neg: predicate.neg,",
        "Instruction.OwnerPredicate",
    ),
    (
        "ignored_predicate_write_is_remembered",
        "src/register_owner.rs",
        "&& normalized.reg.is_some(),",
        "&& (normalized.reg.is_some() || destination.idx == 7),",
        "Instruction.OwnerComparisonPlan",
    ),
    (
        "nonplain_comparison_is_remembered",
        "src/register_owner.rs",
        "remember: plain\n",
        "remember: (!plain || plain)\n",
        "Instruction.OwnerComparisonPlan",
    ),
    (
        "comparison_record_file_is_erased",
        "src/register_owner.rs",
        "destination.file as u8, destination.idx",
        "0, destination.idx",
        "Instruction.OwnerComparisonPlan",
    ),
    (
        "comparison_creator_narrows_to_eight_bits",
        "src/register_owner.rs",
        "key: (thread as u32,",
        "key: ((thread as u8) as u32,",
        "Instruction.OwnerComparisonPlan",
    ),
    (
        "nonpredicate_write_is_remembered",
        "src/register_owner.rs",
        "&& matches!(destination.file, crate::File::P | crate::File::UP)",
        "&& (matches!(destination.file, crate::File::P | crate::File::UP) || plain)",
        "Instruction.OwnerComparisonPlan",
    ),
]

MUTATIONS += [
    (
        "highest_barrier_omitted",
        "src/control_flow.rs",
        "*value < 32",
        "*value < 31",
        "Instruction.BarrierSeed",
    ),
    (
        "unknown_barrier_ignored",
        "src/control_flow.rs",
        "Op::BarSync { .. } | Op::CallAbs { .. } => u32::MAX,",
        "Op::BarSync { .. } | Op::CallAbs { .. } => 0,",
        "Instruction.BarrierSeed",
    ),
]

MUTATIONS += [
    (
        "collector_uses_legacy_on_supported",
        "src/destination_rows.rs",
        "Some(registers) => registers,",
        "Some(_) => fallback.row(operation),",
        "Instruction.DestinationRows",
    ),
    (
        "collector_erases_selected_row",
        "src/destination_rows.rs",
        "Some(registers) => registers,",
        "Some(_) => Vec::new(),",
        "Instruction.DestinationRows",
    ),
    (
        "raw_ir_erases_file",
        "src/destination_rows.rs",
        "            row.push(crate::register_owner::canonical_key(\n"
        "                0,\n"
        "                &[],\n"
        "                rows[pc][position],\n"
        "            ));",
        "            let (owner, _, index) = crate::register_owner::canonical_key(0, &[], "
        "rows[pc][position]);\n"
        "            row.push((owner, 0, index));",
        "Instruction.DestinationKeys",
    ),
]

MUTATIONS += [
    (
        "source_omits_seeds",
        "src/control_flow.rs",
        "let seeds = barrier_seeds(code);",
        "let seeds = Vec::new();",
        "Instruction.GraphSeedSource",
    ),
    (
        "source_drops_successor_rows",
        "src/control_flow.rs",
        "let successors = graph_successors(code, cut, fold, returns);",
        "let successors = Vec::new();",
        "Instruction.GraphSeedSource",
    ),
]

MUTATIONS += [
    (
        "constant_base_inverted",
        "src/predicate_read.rs",
        "None => Some(true),",
        "None => Some(false),",
        "Instruction.PredicateRead",
    ),
    (
        "predicate_negation_ignored",
        "src/predicate_read.rs",
        "known.map(|value| value != normalized.neg)",
        "known.map(|value| value)",
        "Instruction.PredicateRead",
    ),
    (
        "unknown_predicate_assumed_true",
        "src/predicate_read.rs",
        "known.map(|value| value != normalized.neg)",
        "known.or(Some(true)).map(|value| value != normalized.neg)",
        "Instruction.PredicateRead",
    ),
    (
        "architectural_normalization_bypassed",
        "src/predicate_read.rs",
        "let normalized = crate::register_owner::normalize_predicate(front, predicate);",
        "let normalized = predicate;",
        "Instruction.PredicateRead",
    ),
]

MUTATIONS += [
    (
        "false_guard_decodes_target",
        "src/control_transfer.rs",
        "AbstractValue::Known(0) => Some(Outcome::Next),",
        "AbstractValue::Known(0) => taken_operation(context, operation, reader),",
        "Instruction.LookaheadControlFalseGuard",
    ),
    (
        "unknown_guard_drops_fallthrough",
        "src/control_transfer.rs",
        "AbstractValue::UniformUnknown => with_fallthrough(taken, context.pc),",
        "AbstractValue::UniformUnknown => taken,",
        "Instruction.LookaheadControlGuardLifting",
    ),
    (
        "fork_appends_the_wrong_pc",
        "src/control_transfer.rs",
        "Outcome::Fork(mut targets) => {\n            targets.push(next);",
        "Outcome::Fork(mut targets) => {\n            targets.push(0);",
        "Instruction.LookaheadControlFallthrough",
    ),
]

MUTATIONS += [
    (
        "shared_up_accepted",
        "src/register_owner.rs",
        "matches!(register.file, File::V | File::R | File::P)",
        "matches!(register.file, File::V | File::R | File::P | File::UP)",
        "Instruction.OwnerTemporal",
    ),
    (
        "predicate_shared_filter_bypassed",
        "src/analysis_knowledge.rs",
        "if !crate::register_owner::is_thread_private(register) {",
        "if false {",
        "Instruction.AnalysisKnowledge",
    ),
    (
        "operand_shared_filter_bypassed",
        "src/analysis_knowledge.rs",
        "Val::Reg(register, _) => crate::register_owner::is_thread_private(*register),",
        "Val::Reg(_, _) => true,",
        "Instruction.AnalysisKnowledge",
    ),
    (
        "constant_normalization_bypassed",
        "src/analysis_knowledge.rs",
        "let normalized = crate::register_owner::normalize_predicate(front, value);",
        "let normalized = value;",
        "Instruction.AnalysisKnowledge",
    ),
]

MUTATIONS += [
    (
        "real-max-classifies-minimum-as-maximum",
        "src/comparison_select.rs",
        "FCmp::Gt | FCmp::Ge | FCmp::Gtu | FCmp::Geu => plan.negate == plan.swap,",
        "FCmp::Gt | FCmp::Ge | FCmp::Gtu | FCmp::Geu => plan.negate != plan.swap,",
        "Instruction.ComparisonRealMax",
    ),
    (
        "real-max-accepts-equality-mode",
        "src/comparison_select.rs",
        "FCmp::Lt | FCmp::Le | FCmp::Ltu | FCmp::Leu => plan.negate != plan.swap,\n        _ => false,",
        "FCmp::Lt | FCmp::Le | FCmp::Ltu | FCmp::Leu => plan.negate != plan.swap,\n        FCmp::Eq => true,\n        _ => false,",
        "Instruction.ComparisonRealMax",
    ),
]

MUTATIONS += [
    (
        "selection-changes-actual-comparison",
        "src/comparison_select.rs",
        "comparison: *cmp,",
        "comparison: FCmp::Gt,",
        "Instruction.ComparisonSelectionRefinement",
    ),
    (
        "selection-ignores-predicate-negation",
        "src/comparison_select.rs",
        "comparator.compare(plan.comparison, x, y) != plan.negate",
        "comparator.compare(plan.comparison, x, y) != false",
        "Instruction.ComparisonSelectionRefinement",
    ),
    (
        "selection-pair-ignores-operand-swap",
        "src/comparison_select.rs",
        "let (first, second) = if plan.swap { (y, x) } else { (x, y) };",
        "let (first, second) = if plan.swap { (x, y) } else { (x, y) };",
        "Instruction.ComparisonSelectionRefinement",
    ),
    (
        "selection-reads-wrong-capture-source",
        "src/comparison_select.rs",
        "let Some(right) = reader.node_of(b) else {",
        "let Some(right) = reader.node_of(a) else {",
        "Instruction.ComparisonSelectionOperands",
    ),
    (
        "selection-reads-wrong-selector-source",
        "src/comparison_select.rs",
        "let Some(second) = reader.node_of(b) else {",
        "let Some(second) = reader.node_of(a) else {",
        "Instruction.ComparisonSelectionPair",
    ),
    (
        "selection-drops-actual-selector-negation",
        "src/comparison_select.rs",
        "negate: p.neg,",
        "negate: false,",
        "Instruction.ComparisonSelectionPair",
    ),
    (
        "selection-aliases-predicate-files",
        "src/comparison_select.rs",
        "left.file as u8 == right.file as u8 && left.idx == right.idx",
        "left.idx == right.idx",
        "Instruction.ComparisonSelectionOperands",
    ),
    (
        "selection-accepts-unsupported-ftz",
        "src/comparison_select.rs",
        "bop: BoolOp::And,\n            ftz: false,\n            ..",
        "bop: BoolOp::And,\n            ftz: _,\n            ..",
        "Instruction.ComparisonSelectionRefinement",
    ),
    (
        "selection-reverses-forward-operands",
        "src/comparison_select.rs",
        "let swap = if first == x && second == y {\n        false",
        "let swap = if first == x && second == y {\n        true",
        "Instruction.ComparisonSelectionPair",
    ),
]


MUTATIONS += [
    (
        "guard-edge-ignores-post",
        "src/guard_certificate.rs",
        "        let fact = incoming[position];\n        if !crate::register_owner::is_thread_private(fact.register)\n            || !same_value(post_value(context, fact.register), Some(fact.value))\n        {\n            return false;\n",
        "        let fact = incoming[position];\n        if !crate::register_owner::is_thread_private(fact.register)\n            || (!same_value(post_value(context, fact.register), Some(fact.value)) && false)\n        {\n            return false;\n",
        "Instruction.GuardCertificateCheck",
    ),
    (
        "guard-edge-accepts-shared",
        "src/guard_certificate.rs",
        "    while position < incoming.len() {\n        let fact = incoming[position];\n        if !crate::register_owner::is_thread_private(fact.register)\n            || !same_value(post_value(context, fact.register), Some(fact.value))\n        {\n",
        "    while position < incoming.len() {\n        let fact = incoming[position];\n        if false && !crate::register_owner::is_thread_private(fact.register)\n            || !same_value(post_value(context, fact.register), Some(fact.value))\n        {\n",
        "Instruction.GuardCertificateCheck",
    ),
    (
        "guard-successors-skips-rows",
        "src/guard_certificate.rs",
        "    successors: &[usize],\n) -> bool {\n    let mut position = 0;\n    while position < successors.len() {\n        let target = successors[position];\n",
        "    successors: &[usize],\n) -> bool {\n    let mut position = successors.len();\n    while position < successors.len() {\n        let target = successors[position];\n",
        "Instruction.GuardCertificateCheck",
    ),
    (
        "guard-entry-accepts-facts",
        "src/guard_certificate.rs",
        "    }\n    match &certificate.states[certificate.entry] {\n        Some(facts) if empty(facts) => {}\n        _ => return None,\n    }\n",
        "    }\n    match &certificate.states[certificate.entry] {\n        Some(facts) if !empty(facts) => {}\n        _ => return None,\n    }\n",
        "Instruction.GuardCertificateValidation",
    ),
    (
        "guard-move-preserves-upper-bits",
        "src/guard_certificate.rs",
        "            value\n        } else {\n            value & 0xffff_ffff\n        },\n    })\n",
        "            value\n        } else {\n            value\n        },\n    })\n",
        "Instruction.GuardCertificateMove",
    ),
    (
        "guard-compare-signed-as-unsigned",
        "src/guard_certificate.rs",
        "        Cmp::GtU => x > y,\n        Cmp::GeU => x >= y,\n        Cmp::LtS => sx < sy,\n        Cmp::LeS => sx <= sy,\n        Cmp::GtS => sx > sy,\n",
        "        Cmp::GtU => x > y,\n        Cmp::GeU => x >= y,\n        Cmp::LtS => x < y,\n        Cmp::LeS => sx <= sy,\n        Cmp::GtS => sx > sy,\n",
        "Instruction.GuardCertificateWords",
    ),
    (
        "guard-move-omits-second-word",
        "src/guard_certificate.rs",
        "    }\n    match operand {\n        Val::Reg(_, count) => (*count as u16).max(1),\n        Val::Const(_, bytes) => (*bytes as u16 / 4).max(1),\n        _ => 1,\n",
        "    }\n    match operand {\n        Val::Reg(_, _) => 1,\n        Val::Const(_, bytes) => (*bytes as u16 / 4).max(1),\n        _ => 1,\n",
        "Instruction.GuardCertificateSpan",
    ),
    (
        "guard-validation-prunes-unknown-edge",
        "src/guard_certificate.rs",
        "                    count: certificate.code.len(),\n                    cut: 0,\n                    fold: known,\n                    returns,\n                },\n",
        "                    count: certificate.code.len(),\n                    cut: 0,\n                    fold: Some(true),\n                    returns,\n                },\n",
        "Instruction.GuardCertificateValidation",
    ),
    (
        "guard-merge-trusts-disagreeing-grid",
        "src/guard_certificate.rs",
        "        (_, None) => previous,\n        (None, Some(value)) => Some(value),\n        (Some(Some(old)), Some(Some(value))) if old == value => Some(Some(value)),\n        _ => Some(None),\n    }\n",
        "        (_, None) => previous,\n        (None, Some(value)) => Some(value),\n        (Some(Some(old)), Some(Some(value))) if true => Some(Some(value)),\n        _ => Some(None),\n    }\n",
        "Instruction.GuardCertificatePublicationSource",
    ),
    (
        "guard-publication-accepts-collective-branch",
        "src/guard_certificate.rs",
        "        (\n            Op::Bra {\n                any: false,\n                tid0_falls: false,\n                cond: None,\n",
        "        (\n            Op::Bra {\n                any: _,\n                tid0_falls: false,\n                cond: None,\n",
        "Instruction.GuardCertificatePublicationSource",
    ),
    (
        "guard-merge-rows-skips-first-pc",
        "src/guard_certificate.rs",
        "        return false;\n    }\n    let mut position = 0;\n    while position < previous.len() {\n        previous[position] = merge_grid(previous[position], incoming[position]);\n",
        "        return false;\n    }\n    let mut position = 1;\n    while position < previous.len() {\n        previous[position] = merge_grid(previous[position], incoming[position]);\n",
        "Instruction.GuardCertificateRowsMerge",
    ),
    (
        "guard-merge-rows-overwrites-pc-zero",
        "src/guard_certificate.rs",
        "    let mut position = 0;\n    while position < previous.len() {\n        previous[position] = merge_grid(previous[position], incoming[position]);\n        position += 1;\n    }\n",
        "    let mut position = 0;\n    while position < previous.len() {\n        previous[0] = merge_grid(previous[position], incoming[position]);\n        position += 1;\n    }\n",
        "Instruction.GuardCertificateRowsMerge",
    ),
    (
        "guard-publication-reuses-first-observation",
        "src/guard_certificate.rs",
        "    let mut position = 0;\n    while position < code.len() {\n        output.push(branch_fold(&code[position], observations[position]));\n        position += 1;\n    }\n",
        "    let mut position = 0;\n    while position < code.len() {\n        output.push(branch_fold(&code[position], observations[0]));\n        position += 1;\n    }\n",
        "Instruction.GuardCertificateRowsPublish",
    ),
    (
        "guard-unknown-guard-restores-equal-bits",
        "src/guard_certificate.rs",
        "        _ => None,\n    };\n    if matches!(guard, Some(true)) {\n        assigned\n    } else {\n",
        "        _ => None,\n    };\n    if matches!(guard, Some(true)) || same_value(assigned, old) {\n        assigned\n    } else {\n",
        "Instruction.GuardCertificateUnknownGuard",
    ),
    (
        "guard-undefined-operand-becomes-zero",
        "src/guard_certificate.rs",
        "                Some(0)\n            } else {\n                lookup(facts, *register)\n            }\n        }\n",
        "                Some(0)\n            } else {\n                lookup(facts, *register).or(Some(0))\n            }\n        }\n",
        "Instruction.GuardCertificateDefinedFacts",
    ),
    (
        "guard-paired-constant-write-omits-high-word",
        "src/guard_certificate.rs",
        "    match operand {\n        Val::Reg(_, count) => (*count as u16).max(1),\n        Val::Const(_, bytes) => (*bytes as u16 / 4).max(1),\n        _ => 1,\n    }\n",
        "    match operand {\n        Val::Reg(_, count) => (*count as u16).max(1),\n        Val::Const(_, _bytes) => 1,\n        _ => 1,\n    }\n",
        "Instruction.GuardCertificateSpan",
    ),
]

MUTATIONS += [
    (
        "metadata-omits-destination-history",
        "src/metadata_delivery.rs",
        "history(destinations[index], touched, active, guarded, writer);\n        index += 1;",
        "history(destinations[index], touched, active, guarded, writer);\n        index += 2;",
        "Instruction.MetadataDeliveryDestinations",
    ),
    (
        "metadata-clears-bits-after-taint",
        "src/metadata_delivery.rs",
        "clear_bits(destination, touched, active, guarded, writer);\n"
        "        active_taint(destination, active, guarded, writer);",
        "active_taint(destination, active, guarded, writer);\n"
        "        clear_bits(destination, touched, active, guarded, writer);",
        "Instruction.MetadataDeliveryDestinations",
    ),
    (
        "metadata-omits-guard-only-force",
        "src/metadata_delivery.rs",
        "guard_taint(destination, active, guarded, writer);",
        "guard_taint(destination, active, active, writer);",
        "Instruction.MetadataDeliveryDestinations",
    ),
    (
        "metadata-misclassifies-load-capture",
        "src/metadata_delivery.rs",
        "matches!(operation, Op::Ld { .. })",
        "matches!(operation, Op::Ld { bytes: 0, .. })",
        "Instruction.MetadataDeliveryProducer",
    ),
    (
        "metadata-omits-guard-only-touched",
        "src/metadata_delivery.rs",
        "if !contains(active, thread) {",
        "if !contains(active, thread) && !contains(guarded, thread) {",
        "Instruction.MetadataThreads",
    ),
    (
        "metadata-empty-writes-skip-preparation",
        "src/metadata_delivery.rs",
        "let touched = touched_threads(active, guarded);",
        "if destinations.len() == 0 { return false; }\n"
        "    let touched = touched_threads(active, guarded);",
        "Instruction.MetadataDeliveryProducer",
    ),
    (
        "metadata-unsupported-delivers-callback",
        "src/metadata_delivery.rs",
        "let Some(destinations) = register_effect::memory_writes(operation) else {\n"
        "        return false;",
        "let Some(destinations) = register_effect::memory_writes(operation) else {\n"
        "        writer.apply(Action::PrepareSources { is_load: false });\n"
        "        return false;",
        "Instruction.MetadataDeliveryProducer",
    ),
    (
        "metadata-prepares-before-history",
        "src/metadata_delivery.rs",
        "histories(&destinations, &touched, active, guarded, writer);\n"
        "    let is_load = is_load(operation);\n"
        "    writer.apply(Action::PrepareSources { is_load });",
        "let is_load = is_load(operation);\n"
        "    writer.apply(Action::PrepareSources { is_load });\n"
        "    histories(&destinations, &touched, active, guarded, writer);",
        "Instruction.MetadataDeliveryProducer",
    ),
    (
        "metadata-queries-another-thread",
        "src/metadata_delivery.rs",
        "let source_tainted = writer.source_tainted(thread, destination);",
        "let source_tainted = writer.source_tainted(0, destination);",
        "Instruction.MetadataDeliveryLoops",
    ),
]

MUTATIONS += [
    (
        "payload-nonzero-padding",
        "src/payload.rs",
        "output.push(0);",
        "output.push(1);",
        "Instruction.PayloadInitialization",
    ),
    (
        "payload-repeats-low-byte",
        "src/payload.rs",
        "value >> (8 * byte)",
        "value >> 0",
        "Instruction.PayloadByteRefinement",
    ),
    (
        "payload-omits-final-window",
        "src/payload.rs",
        "while index < windows.len() {",
        "while index + 1 < windows.len() {",
        "Instruction.PayloadAssembly",
    ),
    (
        "payload-accepts-overwide-element",
        "src/payload.rs",
        "bytes / elements <= 8",
        "bytes / elements <= 16",
        "Instruction.PayloadSupportRefinement",
    ),
    (
        "payload-accepts-sass-virtual",
        "src/payload.rs",
        "Front::Sass => file != File::V,",
        "Front::Sass => true,",
        "Instruction.PayloadSupportRefinement",
    ),
    (
        "st256-repeats-low-base",
        "src/payload.rs",
        "let first = if index < 4 { *lo } else { *hi };",
        "let first = if index < 3 { *lo } else { *hi };",
        "Instruction.St256Refinement",
    ),
    (
        "st256-restarts-high-byte-offset",
        "src/payload.rs",
        "4 * index as u32,",
        "4 * (index % 4) as u32,",
        "Instruction.St256Refinement",
    ),
    (
        "st256-omits-final-word",
        "src/payload.rs",
        "while index < 8 {",
        "while index < 7 {",
        "Instruction.St256Refinement",
    ),
    (
        "st256-copies-eight-bytes",
        "src/payload.rs",
        "            4,\n        ));",
        "            8,\n        ));",
        "Instruction.St256Refinement",
    ),
    (
        "payload-discards-read-error",
        "src/payload.rs",
        "Err(error) => return Err(error),",
        "Err(error) => { if index == 0 { return Err(error); } return Ok(output); },",
        "Instruction.PayloadFailure",
    ),
]

MUTATIONS += [
    (
        "masked-loop-ignores-mask",
        "src/masked_load.rs",
        "half(*lo, *mask, 0, data, writer);",
        "half(*lo, 255, 0, data, writer);",
        "Instruction.MaskedLoadProducer",
    ),
    (
        "masked-loop-leaves-destination-holes",
        "src/masked_load.rs",
        "idx: first.idx + rank,",
        "idx: first.idx + rank * 2,",
        "Instruction.MaskedLoadLoop",
    ),
    (
        "masked-word-reverses-byte-order",
        "src/masked_load.rs",
        "u32::from_le_bytes([",
        "u32::from_be_bytes([",
        "Instruction.MaskedLoadWord",
    ),
    (
        "masked-loop-doubles-half-start",
        "src/masked_load.rs",
        "let source = start + index;",
        "let source = start + start;",
        "Instruction.MaskedLoadLoop",
    ),
    (
        "masked-loop-omits-last-source",
        "src/masked_load.rs",
        "writer.write(destination, value, source * 4);",
        "if source != 7 { writer.write(destination, value, source * 4); }",
        "Instruction.MaskedLoadLoop",
    ),
]


MUTATIONS += [
    (
        "atomic-result-forgot-predicate",
        "src/atomic_result.rs",
        "word.is_some()\n        || crate::",
        "word.is_some()\n        || false && crate::",
        "Instruction.AtomicResult",
    ),
    (
        "atomic-result-counts-ignored-predicate",
        "src/atomic_result.rs",
        ".reg\n        .is_some()",
        ".reg\n        .is_some() || predicate.is_some()",
        "Instruction.AtomicResult",
    ),
    (
        "atomic-result-forgot-unused-cas",
        "src/atomic_result.rs",
        "matches!(operation, AtomOp::Cas | AtomOp::CasSpin)",
        "matches!(operation, AtomOp::CasSpin)",
        "Instruction.AtomicResult",
    ),
    (
        "atomic-result-forgot-unused-cas-spin",
        "src/atomic_result.rs",
        "matches!(operation, AtomOp::Cas | AtomOp::CasSpin)",
        "matches!(operation, AtomOp::Cas)",
        "Instruction.AtomicResult",
    ),
]


EXCHANGE_LOCAL_POLICY_CONTROLS = {
    "atomic-result-forgot-unused-cas": "AtomOp::CasSpin | AtomOp::Exch",
    "atomic-result-forgot-unused-cas-spin": "AtomOp::Cas | AtomOp::Exch",
}
MUTATIONS = [
    (
        name,
        source,
        (
            "AtomOp::Cas | AtomOp::CasSpin | AtomOp::Exch"
            if name in EXCHANGE_LOCAL_POLICY_CONTROLS
            else old
        ),
        EXCHANGE_LOCAL_POLICY_CONTROLS.get(name, new),
        proof,
    )
    for name, source, old, new, proof in MUTATIONS
]
MUTATIONS += [
    (
        "atomic-result-forgot-unused-exchange",
        "src/atomic_result.rs",
        "AtomOp::Cas | AtomOp::CasSpin | AtomOp::Exch",
        "AtomOp::Cas | AtomOp::CasSpin",
        "Instruction.AtomicResult",
    )
]


MUTATIONS += [
    (
        "exit-data-read-scan-skips-first",
        "src/analysis_knowledge.rs",
        "let mut index = 0;",
        "let mut index = 1;",
        "Instruction.AnalysisReadAdmission",
    ),
    (
        "exit-data-direct-operand-ignored",
        "src/analysis_knowledge.rs",
        "Some(value_is_thread_private(a))",
        "Some(value_is_thread_private(a) || true)",
        "Instruction.AnalysisReadComposition",
    ),
]


MUTATIONS += [
    (
        "lookahead-return-empty-table-accepted",
        "src/control_transfer.rs",
        "Some(targets) => targets.contains(&target),",
        "Some(targets) => targets.is_empty() || targets.contains(&target),",
        "Instruction.LookaheadControlReturnValidation",
    ),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(sim_sources=()) -> dict[str, str]:
    paths = [ROOT / name for name in RUST] + [PROJECT / name for name in LEAN]
    paths += [
        ROOT / name
        for name in [
            "Cargo.toml",
            "Cargo.lock",
            "crates/ir/Cargo.toml",
            "crates/sim/src/num.rs",
            "crates/sim/src/seg.rs",
            "crates/sim/src/exec.rs",
            "crates/sim/tests/masked_load_effects.rs",
            "crates/sim/tests/memory_payload.rs",
            "crates/sim/tests/st256_payload.rs",
            "crates/sim/src/explore.rs",
            "crates/sass/src/lib.rs",
            "probes/instruction_effects/check.py",
            "lean_model/instruction_effects/README.md",
            "probes/instruction_effects/Cargo.lock",
            "probes/memory_write/check.py",
            "lean_model/generated_lean.py",
        ]
    ]
    paths += [ROOT / name for name in sim_sources]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def load_core():
    spec = importlib.util.spec_from_file_location(
        "memory_effect_checker", ROOT / "probes/memory_write/check.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the memory-effect checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_selection_program():
    spec = importlib.util.spec_from_file_location(
        "selection_program_checker",
        ROOT / "probes/instruction_effects/selection_program.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import simulator-constructor checker")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_ir(destination: Path) -> None:
    (destination / "src").mkdir(parents=True)
    for source in RUST:
        shutil.copyfile(ROOT / source, destination / "src" / Path(source).name)
    workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
    package = workspace["workspace"]["package"]
    serde = workspace["workspace"]["dependencies"]["serde"]
    if isinstance(serde, str):
        serde = {"version": serde}
    dependency = ", ".join(
        f"{key} = {json.dumps(value)}" for key, value in serde.items()
    )
    (destination / "Cargo.toml").write_text(
        '[package]\nname = "ptxv-ir"\n'
        f"version = {json.dumps(package['version'])}\nedition = {json.dumps(package['edition'])}\n"
        f"[dependencies]\nserde = {{ {dependency} }}\n[workspace]\n"
    )
    shutil.copyfile(
        ROOT / "probes/instruction_effects/Cargo.lock", destination / "Cargo.lock"
    )


def extract(core, tools, crate: Path, out: Path, env) -> Path:
    command = [tools["charon"], "cargo", "--preset=aeneas", "--extract-opaque-bodies"]
    for entry in ENTRIES:
        command += ["--start-from", f"ptxv_ir::{entry}"]
    command += [
        "--opaque",
        "core",
        "--opaque",
        "std",
        "--opaque",
        "alloc",
        "--include",
        "core::option",
        "--include",
        "core::result",
        "--include",
        "core::ops::control_flow::ControlFlow",
        "--error-on-warnings",
        "--abort-on-error",
        "--dest-file",
        str(out / "Effects.llbc"),
        "--",
        "--offline",
        "--locked",
        "--lib",
    ]
    core.run(command, crate, out / "extract.log", env)
    fresh = out / "fresh"
    fresh.mkdir()
    core.run(
        [
            tools["aeneas"],
            "-backend",
            "lean",
            "-namespace",
            "PtxvIr",
            "-dest",
            str(fresh),
            "-subdir",
            "Instruction/Code",
            "-split-files",
            "-abort-on-error",
            "-no-progress-bar",
            "-all-computable",
            str(out / "Effects.llbc"),
        ],
        crate,
        out / "translate.log",
        env,
    )
    generated = fresh / "Instruction/Code"
    if sorted(path.name for path in generated.iterdir()) != ["Funs.lean", "Types.lean"]:
        raise RuntimeError("Unexpected generated files, possibly an external stub")
    core.LEAN_EXPORT.publish(generated, out / "raw-generated")
    return generated


def negative_controls(
    core, tools, crate: Path, project: Path, out: Path, env, selected=None
) -> dict:
    controls = {}
    for name, source_file, old, new, theorem in MUTATIONS:
        if selected and name not in selected:
            continue
        work = out / "mutations" / name
        mutant = work / "ir"
        proof = work / "lean"
        shutil.copytree(crate, mutant)
        shutil.copytree(project, proof, symlinks=True)
        source = mutant / source_file
        original = source.read_text()
        if original.count(old) != 1:
            raise RuntimeError(f"Ambiguous mutation target: {name}")
        source.write_text(original.replace(old, new))
        status = core.run(
            ["cargo", "test", "--offline", "--locked", "--lib"],
            mutant,
            work / "rust-tests.log",
            env,
            expect_success=False,
        )
        if (
            status == 0
            or "test result: FAILED" not in (work / "rust-tests.log").read_text()
        ):
            raise RuntimeError(f"Mutation escaped Rust regression tests: {name}")
        generated = extract(core, tools, mutant, work, env)
        for path in generated.iterdir():
            shutil.copyfile(path, proof / "Instruction/Code" / path.name)
        core.run(
            [tools["lake"], "build", "Instruction.Code.Funs"],
            proof,
            work / "code.log",
            env,
        )
        rejected = core.run(
            [tools["lake"], "build", theorem],
            proof,
            work / "proof.log",
            env,
            expect_success=False,
        )
        if (
            rejected == 0
            or f"error: {theorem.replace('.', '/')}.lean:"
            not in (work / "proof.log").read_text()
        ):
            raise RuntimeError(f"Mutation escaped the theorem: {name}")
        proof_log = (work / "proof.log").read_text()
        if any(
            marker in proof_log
            for marker in (
                "Unknown identifier",
                "Unknown constant",
                "failed to create thread",
            )
        ):
            raise RuntimeError(f"Unrelated source proof failure: {name}")
        controls[name] = {
            "source_sha256": digest(source),
            "rust_exit": status,
            "generated_build": True,
            "proof_exit": rejected,
            "theorem": theorem,
        }
    return controls


def main() -> None:
    selection_program = load_selection_program()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument(
        "--control",
        action="append",
        choices=[item[0] for item in MUTATIONS]
        + [item[0] for item in selection_program.MUTATIONS],
    )
    args = parser.parse_args()
    if args.control and not args.negative_controls:
        parser.error("--control requires --negative-controls")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "receipt.json"
    receipt.unlink(missing_ok=True)
    tools = json.loads(args.tools.read_text())
    core = load_core()
    env = core.verification_environment(out, tools)
    env["CARGO_HOME"] = os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PTXV_EXCLUDE_GPUS"] = "0,1,2,3,4,5,6,7"
    sim_out = out / "sim"
    sim_out.mkdir()
    sim_workspace, sim_sources = selection_program.source_archive(sim_out)
    before = hashes(sim_sources)
    for name in sim_sources:
        archived = sim_workspace / name
        if (
            name != Path("Cargo.lock")
            and archived.is_file()
            and digest(archived) != before[str(name)]
        ):
            raise RuntimeError(f"Simulator source changed while archiving: {name}")
    tool_hashes = {
        name: digest(Path(tools[name])) for name in ["charon", "aeneas", "lake"]
    }
    backend_hash = digest(Path(tools["backend"]) / "lake-manifest.json")
    crate = out / "ir"
    archive_ir(crate)
    core.run(
        ["cargo", "test", "--offline", "--locked", "--lib"],
        crate,
        out / "rust-tests.log",
        env,
    )
    generated = extract(core, tools, crate, out, env)
    sim_env = selection_program.native_environment(core, tools, sim_out, sim_workspace)
    sim_generated = selection_program.extract(
        core, tools, sim_workspace, sim_out, sim_env
    )
    for generated_files, subdir in [
        (generated, "Instruction/Code"),
        (sim_generated, selection_program.SUBDIR),
    ]:
        for generated_file in generated_files.iterdir():
            target = PROJECT / subdir / generated_file.name
            if args.regenerate:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(generated_file, target)
            elif generated_file.read_bytes() != target.read_bytes():
                raise RuntimeError(f"Stale generated file: {target}")
    if args.regenerate:
        after = hashes(sim_sources)
        generated_names = {
            "lean_model/instruction_effects/Instruction/Code/Types.lean",
            "lean_model/instruction_effects/Instruction/Code/Funs.lean",
            "lean_model/instruction_effects/Instruction/SelectionProgram/Code/Types.lean",
            "lean_model/instruction_effects/Instruction/SelectionProgram/Code/Funs.lean",
        }
        if {
            name: value for name, value in before.items() if name not in generated_names
        } != {
            name: value for name, value in after.items() if name not in generated_names
        }:
            raise RuntimeError("Sources changed during extraction")
        before = after
    project = out / "lean"
    for name in LEAN:
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target)
        if re.search(
            r"\b(sorry|admit|axiom|native_decide|bv_decide)\b", target.read_text()
        ):
            raise RuntimeError(f"Forbidden proof shortcut: {target}")
    core.configure(project, Path(tools["backend"]))
    manifest_path = project / "lake-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["name"] = "ptxv_instruction_effects"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    core.run(
        [tools["lake"], "build", "Instruction"], project, out / "lean-build.log", env
    )
    core.run(
        [tools["lake"], "env", "lean", "-j4", "Audit.lean"],
        project,
        out / "axiom-audit.log",
        env,
    )
    audit = (out / "axiom-audit.log").read_text()
    for group in re.findall(r"depends on axioms: \[([^\]]*)\]", audit):
        unexpected = {
            name.strip() for name in group.split(",") if name.strip()
        } - ALLOWED_AXIOMS
        if unexpected:
            raise RuntimeError(f"Unexpected axioms: {unexpected}")
    controls = (
        negative_controls(
            core, tools, crate, project, out, env, set(args.control or [])
        )
        if args.negative_controls
        else {}
    )
    sim_controls = {item[0] for item in selection_program.MUTATIONS}
    if args.negative_controls and (
        not args.control or sim_controls.intersection(args.control)
    ):
        controls.update(
            selection_program.negative_controls(
                core,
                tools,
                sim_workspace,
                project,
                sim_out,
                sim_env,
                set(args.control or []),
            )
        )
    if hashes(sim_sources) != before:
        raise RuntimeError("Sources changed during verification")
    if tool_hashes != {name: digest(Path(tools[name])) for name in tool_hashes}:
        raise RuntimeError("Verification tools changed during verification")
    if backend_hash != digest(Path(tools["backend"]) / "lake-manifest.json"):
        raise RuntimeError("Backend dependencies changed during verification")
    receipt.write_text(
        json.dumps(
            {
                "source_sha256": before,
                "generated_export": json.loads(
                    (out / "raw-generated/export.json").read_text()
                ),
                "selection_program_generated_export": json.loads(
                    (sim_out / "raw-generated/export.json").read_text()
                ),
                "tool_sha256": tool_hashes,
                "backend_manifest_sha256": backend_hash,
                "fresh_extraction": True,
                "sim_constructor_extraction": True,
                "sim_evaluator_extraction": True,
                "sim_bindings_extraction": True,
                "sim_driver_extraction": True,
                "sim_bound_extraction": True,
                "sim_ieee_comparison_extraction": True,
                "sim_hardware_extraction": True,
                "sim_extraction_roots": [
                    "ptxv_sim::seg::comparison_program_parts",
                    "ptxv_sim::seg::comparison_instruction",
                    "ptxv_sim::seg::binding::read_sval",
                    "ptxv_sim::seg::binding::register_binding",
                    "ptxv_sim::seg::binding::predicate_binding",
                    "ptxv_sim::seg::binding::register_bits",
                    "ptxv_sim::seg::binding::read_predicate",
                    "ptxv_sim::seg::program::execute",
                    "ptxv_sim::seg::bound",
                    "ptxv_sim::fp::f32_cmp",
                    "ptxv_sim::seg::hardware::evaluate",
                ],
                "sim_constructor_root": "ptxv_sim::seg::comparison_program_parts",
                "sim_source_inventory": sorted(map(str, sim_sources)),
                "rust_tests": True,
                "lean_build": True,
                "axiom_audit": True,
                "allowed_axioms": sorted(ALLOWED_AXIOMS),
                "negative_controls": controls,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Verified actual instruction effects; receipt: {receipt}")


MUTATIONS += [
    (
        "sass-discard-omits-repeated-payload",
        "src/store_window.rs",
        "return repeated_windows(first, bytes);",
        "return repeated_windows(first, 0);",
        "Instruction.StoreSyntheticDiscard",
    ),
]


if __name__ == "__main__":
    main()
