#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in the repository root.
"""Extract the Rust memory-effects crate, check its proofs, and record freshness."""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

LOGGER: logging.Logger = logging.getLogger(__name__)
PROBE = Path(__file__).resolve().parent
ROOT = PROBE.parent.parent
LEAN = ROOT / "lean_model/memory_write"
EXPORT_SPEC = importlib.util.spec_from_file_location(
    "generated_lean", ROOT / "lean_model/generated_lean.py"
)
if EXPORT_SPEC is None or EXPORT_SPEC.loader is None:
    raise RuntimeError("Cannot load generated Lean export policy")
LEAN_EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(LEAN_EXPORT)
LEAN_FILES = [
    "MemoryWrite/MachineRecorderModel.lean",
    "MemoryWrite/MachineRecorderRefinement.lean",
    "MemoryWrite/MachineRecorderFinish.lean",
    "MemoryWrite/MachineRecorderObservation.lean",
    "MemoryWrite/MachineFootprint.lean",
    "MemoryWrite/MachineFootprintMerge.lean",
    "MemoryWrite/MachineFootprintContexts.lean",
    "MemoryWrite/MachineActionModel.lean",
    "MemoryWrite/MachineActionBasic.lean",
    "MemoryWrite/MachineActionFinish.lean",
    "MemoryWrite/MachineActionMutations.lean",
    "MemoryWrite/MachineActionContracts.lean",
    "MemoryWrite/MachineRegisters.lean",
    "MemoryWrite/MachineRegistersCoupling.lean",
    "MemoryWrite/MachineRegistersWords.lean",
    "MemoryWrite/MachineRegistersUniform.lean",
    "MemoryWrite/MachineRegistersPoison.lean",
    "MemoryWrite/MachineRegistersFill.lean",
    "MemoryWrite/MachineRegistersVirtual.lean",
    "MemoryWrite/MachineRegistersVirtualMap.lean",
    "MemoryWrite/MachineRegistersDefinedness.lean",
    "MemoryWrite/MachineWarp.lean",
    "MemoryWrite/MachineControl.lean",
    "MemoryWrite/EffectsNamespaces.lean",
    "MemoryWrite/EffectsConstructors.lean",
    "MemoryWrite/EffectsComposition.lean",
    "MemoryWrite/EffectsFold.lean",
    "MemoryWrite/EffectsInvalidation.lean",
    "MemoryWrite/EffectsSticky.lean",
    "MemoryWrite/ConvergenceBarrierFootprint.lean",
    "MemoryWrite/BarrierBatchWitness.lean",
    "MemoryWrite/BarrierBatchBudget.lean",
    "MemoryWrite/BarrierBatchModel.lean",
    "ArrivalWalkRetryWitness.lean",
    "lean-toolchain",
    "lakefile.lean",
    "MemoryWrite.lean",
    "MemoryWrite/ArrivalWalkModel.lean",
    "MemoryWrite/ArrivalWalkFrontier.lean",
    "MemoryWrite/ArrivalWalkChildren.lean",
    "MemoryWrite/ArrivalWalkCallbacks.lean",
    "MemoryWrite/ArrivalWalkRestart.lean",
    "MemoryWrite/ArrivalWalkLoop.lean",
    "MemoryWrite/ArrivalWalkRefinement.lean",
    "MemoryWrite/ArrivalWalkDependencies.lean",
    "MemoryWrite/Model.lean",
    "MemoryWrite/Source.lean",
    "MemoryWrite/Representation.lean",
    "MemoryWrite/Theorem.lean",
    "MemoryWrite/Snapshot.lean",
    "MemoryWrite/Address.lean",
    "MemoryWrite/Storage.lean",
    "MemoryWrite/Effects.lean",
    "MemoryWrite/Production.lean",
    "MemoryWrite/Transfers.lean",
    "MemoryWrite/Module.lean",
    "MemoryWrite/Adapter.lean",
    "MemoryWrite/Producers.lean",
    "MemoryWrite/Consumers.lean",
    "MemoryWrite/WellFormed.lean",
    "MemoryWrite/AsyncSafety.lean",
    "MemoryWrite/Footprint.lean",
    "MemoryWrite/Commutation.lean",
    "MemoryWrite/RustSpan.lean",
    "MemoryWrite/RustCapture.lean",
    "MemoryWrite/RustRefinement.lean",
    "MemoryWrite/RustEngine.lean",
    "MemoryWrite/RustCopies.lean",
    "MemoryWrite/RustWords.lean",
    "MemoryWrite/RustNumeric.lean",
    "MemoryWrite/RustAtomic.lean",
    "MemoryWrite/AtomicInteger.lean",
    "MemoryWrite/AtomicGlobalObserver.lean",
    "MemoryWrite/AtomicExecution.lean",
    "MemoryWrite/AtomicRequestLog.lean",
    "MemoryWrite/AtomicBytes.lean",
    "MemoryWrite/AtomicCompareExchange.lean",
    "MemoryWrite/AtomicRefinement.lean",
    "MemoryWrite/AtomicComponentOrder.lean",
    "MemoryWrite/AtomicConfluence.lean",
    "MemoryWrite/RustAtomicConfluence.lean",
    "MemoryWrite/AtomicComponentPlan.lean",
    "MemoryWrite/AtomicComponentTrace.lean",
    "MemoryWrite/RustAtomicComponentPlan.lean",
    "MemoryWrite/RustAtomicComponentTrace.lean",
    "MemoryWrite/RustAtomicPacked.lean",
    "MemoryWrite/RustAtomicScalar.lean",
    "MemoryWrite/AtomicLaneCoverage.lean",
    "MemoryWrite/RustAtomicReturningCoverage.lean",
    "MemoryWrite/AtomicReturningCoverage.lean",
    "MemoryWrite/RustAtomicLocalCoverage.lean",
    "MemoryWrite/AtomicLocalCoverage.lean",
    "MemoryWrite/RustAtomicRedCoverage.lean",
    "MemoryWrite/AtomicRedCoverage.lean",
    "MemoryWrite/RustAtomicLaneScan.lean",
    "MemoryWrite/AtomicLaneScan.lean",
    "MemoryWrite/RustAtomicScalarFloat.lean",
    "MemoryWrite/AtomicScalarFloat.lean",
    "MemoryWrite/AtomicHistorySafety.lean",
    "MemoryWrite/AsyncHistorySafety.lean",
    "MemoryWrite/CheckedControlSafety.lean",
    "MemoryWrite/WriteConsumerSafety.lean",
    "MemoryWrite/MarkerConsumerSafety.lean",
    "MemoryWrite/SymbolicState.lean",
    "MemoryWrite/SymbolicStateRefinement.lean",
    "MemoryWrite/MetadataPlanModel.lean",
    "MemoryWrite/MetadataPlanRefinement.lean",
    "MemoryWrite/AnalysisCache.lean",
    "MemoryWrite/AnalysisCacheRefinement.lean",
    "MemoryWrite/AnalysisCacheFootprint.lean",
    "MemoryWrite/AnalysisCacheReplay.lean",
    "MemoryWrite/AnalysisCacheMayWrite.lean",
    "MemoryWrite/GraphSource.lean",
    "MemoryWrite/RawSourceRows.lean",
    "MemoryWrite/SourceRows.lean",
    "MemoryWrite/BarrierReachMerge.lean",
    "MemoryWrite/BarrierReachModel.lean",
    "MemoryWrite/BarrierReachPaths.lean",
    "MemoryWrite/BarrierReachScan.lean",
    "MemoryWrite/BarrierReachTotal.lean",
    "MemoryWrite/RegisterGraphPublication.lean",
    "MemoryWrite/RegisterIndexBudget.lean",
    "MemoryWrite/RegisterIndexBuildProgress.lean",
    "MemoryWrite/RegisterIndexBuildTotal.lean",
    "MemoryWrite/RegisterIndexEnsureTotal.lean",
    "MemoryWrite/RegisterIndexInternTotal.lean",
    "MemoryWrite/RegisterIndexRowProgress.lean",
    "MemoryWrite/RegisterIndexRowTotal.lean",
    "MemoryWrite/MayWriteAnalysisTotal.lean",
    "MemoryWrite/MayWriteCompactTotal.lean",
    "MemoryWrite/MayWriteFiniteFacts.lean",
    "MemoryWrite/MayWriteInitializationProgress.lean",
    "MemoryWrite/MayWriteMainTotal.lean",
    "MemoryWrite/MayWriteMergeProgress.lean",
    "MemoryWrite/MayWriteRowOrder.lean",
    "MemoryWrite/MayWriteScanTotal.lean",
    "MemoryWrite/MayWriteSuccessorRank.lean",
    "MemoryWrite/MayWriteSuccessorTotal.lean",
    "MemoryWrite/MayWriteWordProgress.lean",
    "MemoryWrite/TotalPrimitives.lean",
    "MemoryWrite/MayWriteMainProgress.lean",
    "MemoryWrite/MayWriteQueryProgress.lean",
    "MemoryWrite/MayWriteQueryTotal.lean",
    "MemoryWrite/RegisterIndexAllocation.lean",
    "MemoryWrite/RegisterIndexBuildGraph.lean",
    "MemoryWrite/RegisterIndexBuildModel.lean",
    "MemoryWrite/RegisterIndexBuildRow.lean",
    "MemoryWrite/RegisterIndexEnsurePage.lean",
    "MemoryWrite/RegisterIndexIntern.lean",
    "MemoryWrite/RegisterIndexInternModel.lean",
    "MemoryWrite/RegisterIndexKey.lean",
    "MemoryWrite/RegisterIndexLookup.lean",
    "MemoryWrite/RegisterIndexModel.lean",
    "MemoryWrite/RegisterIndexPageInsertion.lean",
    "MemoryWrite/RegisterIndexPath.lean",
    "MemoryWrite/RegisterIndexSearch.lean",
    "MemoryWrite/RegisterIndexSlots.lean",
    "MemoryWrite/AnalysisCacheSoundness.lean",
    "MemoryWrite/MayWriteAllocation.lean",
    "MemoryWrite/MayWriteAnalysis.lean",
    "MemoryWrite/MayWriteBounds.lean",
    "MemoryWrite/MayWriteCompact.lean",
    "MemoryWrite/MayWriteCopyIncoming.lean",
    "MemoryWrite/MayWriteEnsurePc.lean",
    "MemoryWrite/MayWriteInitialized.lean",
    "MemoryWrite/MayWriteMainLoop.lean",
    "MemoryWrite/MayWriteMergeEffect.lean",
    "MemoryWrite/MayWriteOutgoing.lean",
    "MemoryWrite/MayWritePageAddition.lean",
    "MemoryWrite/MayWritePaths.lean",
    "MemoryWrite/MayWritePropagation.lean",
    "MemoryWrite/MayWriteQuery.lean",
    "MemoryWrite/MayWriteQueue.lean",
    "MemoryWrite/MayWriteResources.lean",
    "MemoryWrite/MayWriteRowMerge.lean",
    "MemoryWrite/MayWriteScan.lean",
    "MemoryWrite/MayWriteSearch.lean",
    "MemoryWrite/MayWriteSetup.lean",
    "MemoryWrite/MayWriteSlots.lean",
    "MemoryWrite/MayWriteStorage.lean",
    "MemoryWrite/MayWriteStorageInvariant.lean",
    "MemoryWrite/MayWriteWordCount.lean",
    "MemoryWrite/MayWriteWords.lean",
    "MemoryWrite/MayWriteWorklist.lean",
    "MemoryWrite/PartialLoops.lean",
    "MemoryWrite/SuccessfulPrimitives.lean",
    "MemoryWrite/Definedness.lean",
    "MemoryWrite/DefinednessRefinement.lean",
    "MemoryWrite/BarrierTransition.lean",
    "MemoryWrite/BarrierPublication.lean",
    "MemoryWrite/BarrierTransitionRefinement.lean",
    "MemoryWrite/BarrierStaging.lean",
    "MemoryWrite/BarrierClockMove.lean",
    "MemoryWrite/BarrierPhaseKey.lean",
    "MemoryWrite/BarrierPolling.lean",
    "MemoryWrite/BarrierWaitPolicy.lean",
    "MemoryWrite/RustBarrierWaitPolicy.lean",
    "MemoryWrite/SparseClock.lean",
    "MemoryWrite/RustSparseClock.lean",
    "MemoryWrite/ClockBulk.lean",
    "MemoryWrite/ClockFilter.lean",
    "MemoryWrite/ClockMeet.lean",
    "MemoryWrite/ClockReadRelease.lean",
    "MemoryWrite/ClockTick.lean",
    "MemoryWrite/ClockUpdates.lean",
    "MemoryWrite/RustClockBulk.lean",
    "MemoryWrite/RustClockFilter.lean",
    "MemoryWrite/RustClockMeet.lean",
    "MemoryWrite/RustClockProducerHistory.lean",
    "MemoryWrite/RustClockProducerPublication.lean",
    "MemoryWrite/RustClockReadRelease.lean",
    "MemoryWrite/RustClockTick.lean",
    "MemoryWrite/RustClockUpdates.lean",
    "MemoryWrite/RustClockMerge.lean",
    "MemoryWrite/RustClockCover.lean",
    "MemoryWrite/RustClockScan.lean",
    "MemoryWrite/RustClockJoin.lean",
    "MemoryWrite/RustClockPublication.lean",
    "MemoryWrite/BarrierReduction.lean",
    "MemoryWrite/BarrierUnitArrival.lean",
    "MemoryWrite/BarrierReductionRefinement.lean",
    "MemoryWrite/ExecutionReduction.lean",
    "MemoryWrite/ExecutionReductionRefinement.lean",
    "MemoryWrite/StoreWord.lean",
    "MemoryWrite/StoreWordRefinement.lean",
    "Audit.lean",
    "Tests.lean",
    "FullTests.lean",
    "RustTests.lean",
    "Replay.lean",
    "MemoryWrite/RustFootprint.lean",
]
GENERATED = ["MemoryWrite/Code/Types.lean", "MemoryWrite/Code/Funs.lean"]
RUST_FILES = ["Cargo.toml", "Cargo.lock"]
RUST_CRATE = ROOT / "crates/memory-write"
RUST_SOURCES = [
    "src/machine_action.rs",
    "src/machine_footprint.rs",
    "src/machine_recorder.rs",
    "src/machine_registers.rs",
    "src/machine_control.rs",
    "src/machine_warp.rs",
    "src/effects.rs",
    "src/arrival_walk.rs",
    "Cargo.toml",
    "src/lib.rs",
    "src/barrier.rs",
    "src/clock.rs",
    "src/cache.rs",
    "src/cache_graph.rs",
    "src/register_index.rs",
    "src/source_rows.rs",
    "src/graph_source.rs",
    "src/barrier_reach.rs",
    "src/checked.rs",
    "src/definedness.rs",
    "src/engine.rs",
    "src/words.rs",
    "src/numeric.rs",
    "src/atomic.rs",
    "src/atomic_bytes.rs",
    "src/atomic_components.rs",
    "src/atomic_packed.rs",
    "src/atomic_scalar.rs",
    "src/atomic_confluence.rs",
    "src/atomic_integer.rs",
    "src/atomic_exec.rs",
    "src/symbolic.rs",
    "src/execution.rs",
    "src/reduction.rs",
    "src/store.rs",
    "src/footprint.rs",
]


MUTATIONS: dict[str, tuple[str, str, str, str]] = {
    "footprint_skips_wrong_cursor": (
        "src/footprint.rs",
        "RustFootprint",
        "if key_before(x, y) {\n            i += 1;",
        "if key_before(x, y) {\n            j += 1;",
    ),
    "footprint_compares_start_instead_of_end": (
        "src/footprint.rs",
        "RustFootprint",
        "} else if x.3 <= y.2 {",
        "} else if x.2 <= y.2 {",
    ),
    "footprint_discards_overlap": (
        "src/footprint.rs",
        "RustFootprint",
        "return true;",
        "return false;",
    ),
    "footprint_empty_uniform_resource": (
        "src/footprint.rs",
        "RustFootprint",
        "pub fn shared_register_range(warp: u32) -> Range {\n    (17, warp, 0, 1)",
        "pub fn shared_register_range(warp: u32) -> Range {\n    (17, warp, 0, 0)",
    ),
    "barrier_policy_ignores_blocking_override": (
        "src/barrier.rs",
        "RustBarrierWaitPolicy",
        "if is_peek && !blocking_override {",
        "if is_peek {",
    ),
    "barrier_policy_blocks_a_valid_pending_peek": (
        "src/barrier.rs",
        "RustBarrierWaitPolicy",
        "Ok(None) => nonblocking,",
        "Ok(None) => false,",
    ),
    "barrier_policy_repeats_a_diagnosed_error": (
        "src/barrier.rs",
        "RustBarrierWaitPolicy",
        "other => wait_can_progress(other, reported, pc),",
        "other => matches!(other, Err(_)) || wait_can_progress(other, reported, pc),",
    ),
    "barrier_policy_forces_an_invalid_query_to_fail": (
        "src/barrier.rs",
        "RustBarrierWaitPolicy",
        "let valid = matches!(result, Ok(_));",
        "let valid = true;",
    ),
    "clock_missing_actor_is_nonzero": (
        "src/clock.rs",
        "RustSparseClock",
        "if found { base[index].1 } else { 0 }",
        "if found { base[index].1 } else { 1 }",
    ),
    "clock_overlay_loses_larger_own": (
        "src/clock.rs",
        "RustSparseClock",
        "result[index].1 = result[index].1.max(own.1);",
        "result[index].1 = result[index].1.min(own.1);",
    ),
    "clock_join_uses_minimum": (
        "src/clock.rs",
        "RustClockMerge",
        "result.push((a.0, a.1.max(b.1)));",
        "result.push((a.0, a.1.min(b.1)));",
    ),
    "clock_shared_base_keeps_uncovered_own": (
        "src/clock.rs",
        "RustClockJoin",
        "get(left, left_own, right_own.0) >= right_own.1",
        "get(left, left_own, right_own.0) <= right_own.1",
    ),
    "clock_empty_constructor_is_nonzero": (
        "src/clock.rs",
        "RustSparseClock",
        "(Vec::new(), (0, 0))",
        "(Vec::new(), (0, 1))",
    ),
    "barrier_wait_error_is_mistaken_for_blocking": (
        "src/barrier.rs",
        "BarrierPolling",
        "Err(_) => match reported {\n            Some(previous) => previous != pc,\n            None => true,\n        },",
        "Err(_) => false,",
    ),
    "barrier_wait_diagnostic_is_not_recorded": (
        "src/barrier.rs",
        "BarrierPolling",
        "*reported = Some(pc);",
        "*reported = None;",
    ),
    "barrier_poll_acquires_the_other_parity": (
        "src/barrier.rs",
        "BarrierPolling",
        "Some(&completed[0])",
        "Some(&completed[1])",
    ),
    "barrier_next_arrival_key_carries_into_the_next_object": (
        "src/barrier.rs",
        "BarrierPhaseKey",
        "phase_key(offset, outcome.after.phase)",
        "phase_key(offset, outcome.intermediate.phase) + 1",
    ),
    "barrier_phase_key_leaks_unmasked_phase_bits": (
        "src/barrier.rs",
        "BarrierPhaseKey",
        "offset as u64 * 0x10_0000 + (phase % 0x10_0000) as u64",
        "offset as u64 * 0x10_0000 + phase as u64",
    ),
    "barrier_release_overwrites_the_wrong_slot": (
        "src/barrier.rs",
        "BarrierClockMove",
        "Some(0) => completed[0] = std::mem::replace(current, empty),",
        "Some(0) => completed[1] = std::mem::replace(current, empty),",
    ),
    "barrier_release_keeps_the_previous_done_as_current": (
        "src/barrier.rs",
        "BarrierClockMove",
        "Some(0) => completed[0] = std::mem::replace(current, empty),",
        "Some(0) => std::mem::swap(current, &mut completed[0]),",
    ),
    "barrier_begin_resets_pending_before_publication": (
        "src/barrier.rs",
        "BarrierStaging",
        "*pending = outcome.intermediate.pending;",
        "*pending = outcome.after.pending;",
    ),
    "barrier_finish_keeps_the_old_phase": (
        "src/barrier.rs",
        "BarrierStaging",
        "*phase = outcome.after.phase;",
        "*phase = outcome.intermediate.phase;",
    ),
    "unit_arrival_ignores_last_pending_with_transaction_debt": (
        "src/barrier.rs",
        "BarrierUnitArrival",
        "pending_before <= 1",
        "pending_before <= 0",
    ),
    "barrier_transaction_membership_is_not_written": (
        "src/barrier.rs",
        "BarrierTransitionRefinement",
        "=> value != 0,",
        "=> value != value,",
    ),
    "barrier_publishes_to_the_next_parity": (
        "src/barrier.rs",
        "BarrierTransitionRefinement",
        "Some(outcome.intermediate.phase % 2)",
        "Some(outcome.after.phase % 2)",
    ),
    "barrier_phase_does_not_advance": (
        "src/barrier.rs",
        "BarrierTransitionRefinement",
        "phase: state.phase.wrapping_add(1),",
        "phase: state.phase,",
    ),
    "barrier_reset_uses_zero_instead_of_expected": (
        "src/barrier.rs",
        "BarrierTransitionRefinement",
        "pending: state.expected,",
        "pending: 0,",
    ),
    "barrier_overarrival_discards_transaction_update": (
        "src/barrier.rs",
        "BarrierTransitionRefinement",
        "return rejected(state, Fault::OverArrival);",
        "return rejected(Counters { transactions: 0, ..state }, Fault::OverArrival);",
    ),
    "barrier_negative_transaction_debt_completes": (
        "src/barrier.rs",
        "BarrierTransitionRefinement",
        "state.transactions == 0",
        "state.transactions <= 0",
    ),
    "program_generation_erased": (
        "src/cache.rs",
        "AnalysisCacheRefinement",
        "        generation,\n",
        "        generation: 0,\n",
    ),
    "exhausted_generation_recycled": (
        "src/cache.rs",
        "AnalysisCacheRefinement",
        "if current == u64::MAX {\n        None",
        "if current == u64::MAX {\n        Some(0)",
    ),
    "failed_observation_becomes_zero": (
        "src/cache.rs",
        "AnalysisCacheRefinement",
        "match (recorded, current) {\n        (None, None) => true,\n        (Some(left), Some(right)) => left == right,\n        _ => false,\n    }",
        "recorded.unwrap_or(0) == current.unwrap_or(0)",
    ),
    "replay_forgets_prior_mismatch": (
        "src/cache.rs",
        "AnalysisCacheReplay",
        "matches: replay.matches && same_observation(recorded, current)",
        "matches: same_observation(recorded, current)",
    ),
    "replay_ignores_initial_key": (
        "src/cache.rs",
        "AnalysisCacheReplay",
        "matches: same_key(recorded, current)",
        "matches: true",
    ),
    "replay_accepts_unconditionally": (
        "src/cache.rs",
        "AnalysisCacheReplay",
        "if replay.matches {",
        "if replay.matches || true {",
    ),
    "selection_ignores_half_offset": (
        "src/definedness.rs",
        "DefinednessRefinement",
        "let selected = mask >> start;",
        "let selected = mask;",
    ),
    "narrow_conversion_reads_whole_word": (
        "src/definedness.rs",
        "DefinednessRefinement",
        "if width >= 64 {",
        "if width >= 64 || width == 16 {",
    ),
    "store_poison_is_whole_word": (
        "src/definedness.rs",
        "DefinednessRefinement",
        "if selected_undefined(mask, 8 * byte, 8) {",
        "if mask != 0 {",
    ),
    "ordered_write_retains_undefinedness": (
        "src/definedness.rs",
        "DefinednessRefinement",
        "source || unordered && previous",
        "source || previous",
    ),
    "vector_store_ignores_byte_offset": (
        "src/definedness.rs",
        "DefinednessRefinement",
        "byte_mask(source, count) << first",
        "byte_mask(source, count)",
    ),
    "vector_store_forgets_previous_registers": (
        "src/definedness.rs",
        "DefinednessRefinement",
        "previous | (byte_mask(source, count) << first)",
        "byte_mask(source, count) << first",
    ),
    "barrier_rejects_exact_arrival_count": (
        "src/reduction.rs",
        "BarrierReductionRefinement",
        "if pending < arrivals {",
        "if pending <= arrivals {",
    ),
    "barrier_count_decrements_only_one": (
        "src/reduction.rs",
        "BarrierReductionRefinement",
        "Some(pending - arrivals)",
        "Some(pending - 1)",
    ),
    "barrier_completion_ignores_transaction_debt": (
        "src/reduction.rs",
        "BarrierReductionRefinement",
        "pending == 0 && transaction_count_is_zero",
        "pending == 0 || transaction_count_is_zero",
    ),
    "plain_arrival_pending_dependency_is_dropped": (
        "src/reduction.rs",
        "BarrierReductionRefinement",
        "arrivals > 0",
        "arrivals > 1",
    ),
    "disabled_allocation_blocks": (
        "src/reduction.rs",
        "BarrierReductionRefinement",
        "AllocationRequest::Disabled | AllocationRequest::Invalid => true,",
        "AllocationRequest::Disabled => false,\n        AllocationRequest::Invalid => true,",
    ),
    "opaque_store_claims_known_history": (
        "src/store.rs",
        "StoreWordRefinement",
        "cell.word = Some(WordHistory::Opaque(id));",
        "cell.word = Some(WordHistory::Known(id));",
    ),
    "store_reverses_first_source_byte": (
        "src/store.rs",
        "StoreWordRefinement",
        "payload.push(payload_byte(bytes[0], source, word));",
        "payload.push(payload_byte(bytes[3], source, word));",
    ),
    "store_ignores_high_address_dependency": (
        "src/store.rs",
        "StoreWordRefinement",
        "let decision = control(input.guard, input.address_low, input.address_high);",
        "let decision = control(input.guard, input.address_low, Metadata { tainted: false, ..input.address_high });",
    ),
    "store_invents_missing_source_history": (
        "src/store.rs",
        "StoreWordRefinement",
        "        None => None,",
        "        None => Some(WordHistory::Known(0)),",
    ),
    "input_derived_zero_becomes_constant": (
        "src/symbolic.rs",
        "SymbolicStateRefinement",
        "if input_derived {",
        "if input_derived && false {",
    ),
    "disabled_dependent_guard_preserves_history": (
        "src/symbolic.rs",
        "SymbolicStateRefinement",
        "if guard_dependent {",
        "if guard_dependent && enabled {",
    ),
    "inactive_guard_keeps_stale_history": (
        "src/symbolic.rs",
        "MetadataPlanRefinement",
        "clear_history: touched && (guard_dependent || !hook_wrote),",
        "clear_history: active && (guard_dependent || !hook_wrote),",
    ),
    "guard_does_not_force_taint": (
        "src/symbolic.rs",
        "MetadataPlanRefinement",
        "force_taint: guard_dependent,",
        "force_taint: false,",
    ),
    "active_taint_complemented": (
        "src/symbolic.rs",
        "MetadataPlanRefinement",
        "active_taint: if active { Some(source_tainted) } else { None },",
        "active_taint: if active { Some(!source_tainted) } else { None },",
    ),
    "inactive_guard_keeps_bit_cache": (
        "src/symbolic.rs",
        "MetadataPlanRefinement",
        "clear_bits: touched,",
        "clear_bits: active,",
    ),
    "node_id_narrows_to_u8": (
        "src/symbolic.rs",
        "SymbolicStateRefinement",
        "Ok(index as u32)",
        "Ok(index as u8 as u32)",
    ),
    "predicate_namespace_is_dropped": (
        "src/symbolic.rs",
        "SymbolicStateRefinement",
        "        file,\n",
        "        file: 0,\n",
    ),
    "poll_ignores_uniform_registers": (
        "src/execution.rs",
        "ExecutionReductionRefinement",
        "&& same_words(before.uniform, after.uniform)",
        "&& true",
    ),
    "poll_ignores_scalar_definedness": (
        "src/execution.rs",
        "ExecutionReductionRefinement",
        "&& same_words(before.scalar_defined, after.scalar_defined)",
        "&& true",
    ),
    "poll_forgets_intermediate_store_change": (
        "src/execution.rs",
        "ExecutionReductionRefinement",
        "memory_unchanged: audit.memory_unchanged && same_memory(before, after),",
        "memory_unchanged: same_memory(before, after),",
    ),
    "poll_ignores_memory_definedness": (
        "src/execution.rs",
        "ExecutionReductionRefinement",
        "|| before[index].defined != after[index].defined",
        "|| false",
    ),
    "exact_start_only": (
        "src/lib.rs",
        "Source",
        "!payload.is_empty() && address < end && start < address + 4",
        "!payload.is_empty() && address == start",
    ),
    "missing_written_marker": (
        "src/lib.rs",
        "Source",
        "cell.written = true;",
        "cell.written = false;",
    ),
    "captured_zero_clears_taint": (
        "src/lib.rs",
        "Source",
        "cell.tainted = byte.tainted;",
        "cell.tainted = byte.tainted && byte.value != 0;",
    ),
    "snapshot_clears_taint": (
        "src/lib.rs",
        "Source",
        "tainted: cell.tainted,",
        "tainted: false,",
    ),
    "rewrite_keeps_stale_half_origin": (
        "src/lib.rs",
        "Source",
        "cell.origin = byte.origin;",
        "if byte.origin.is_some() { cell.origin = byte.origin; }",
    ),
    "snapshot_invents_zero_origin": (
        "src/lib.rs",
        "Source",
        "origin: cell.origin,",
        "origin: Some(Origin::Zero),",
    ),
    "snapshot_invents_defined_bytes": (
        "src/lib.rs",
        "Source",
        "defined: cell.defined,",
        "defined: true,",
    ),
    "snapshot_promotes_opaque_history": (
        "src/lib.rs",
        "Source",
        "word: cell.word,",
        "word: match cell.word { Some(WordHistory::Opaque(id)) => Some(WordHistory::Known(id)), history => history },",
    ),
    "copy_retains_destination_history": (
        "src/lib.rs",
        "Source",
        "cells[index] = cell_after_write(cells[index], write, payload);",
        "let old_word = cells[index].word; cells[index] = cell_after_write(cells[index], write, payload); cells[index].word = old_word;",
    ),
    "dependent_guard_accepted": (
        "src/checked.rs",
        "RustRefinement",
        "if write.guard_dependent {",
        "if false {",
    ),
    "captured_false_dependent_guard_accepted": (
        "src/checked.rs",
        "RustRefinement",
        "if write.guard_dependent {",
        "if write.guard_dependent && write.enabled {",
    ),
    "missing_span_validation": (
        "src/checked.rs",
        "RustRefinement",
        "if !span_fits(cells, write.destination, write.payload.len()) {",
        "if false {",
    ),
    "half_tracker_coupled_to_numeric": (
        "src/engine.rs",
        "RustEngine",
        "if track_half {\n            half.push(",
        "if track_numeric {\n            half.push(",
    ),
    "postword_history_dropped": (
        "src/words.rs",
        "RustWords",
        "cell.word = patch.history;",
        "cell.word = None;",
    ),
    "numeric_forget_marks_half": (
        "src/numeric.rs",
        "RustNumeric",
        "state.numeric_written.push(key);",
        "state.half_written.push(key);",
    ),
    "atomic_width_truncated": (
        "src/atomic.rs",
        "RustAtomic",
        "let end = offset as u128 + bytes as u128;",
        "let end = offset as u128 + if bytes > 4 { 4 } else { bytes as u128 };",
    ),
    "atomic_history_start_only": (
        "src/atomic.rs",
        "RustAtomic",
        "history_start: if offset < 3 { 0 } else { offset - 3 },",
        "history_start: offset,",
    ),
    "failed_cas_uses_write_span": (
        "src/atomic.rs",
        "RustAtomic",
        "if !did_write {",
        "if !did_write && bytes == 0 {",
    ),
    "integer_signed_minimum_uses_unsigned_order": (
        "src/atomic_integer.rs",
        "AtomicInteger",
        "if (old as i32) < (value as i32) {",
        "if old < value {",
    ),
    "cas_uses_upper_compare_bits": (
        "src/atomic_integer.rs",
        "AtomicInteger",
        "let compare = input.compare as u32;",
        "let compare = (input.compare >> 32) as u32;",
    ),
    "failed_cas_writes": (
        "src/atomic_exec.rs",
        "AtomicExecution",
        "let did_write = !cas || result.swapped;",
        "let did_write = !cas || result.swapped || cas;",
    ),
    "cas_returns_replacement": (
        "src/atomic_exec.rs",
        "AtomicExecution",
        "        } else {\n            result.old\n        },\n        return_words:",
        "        } else {\n            result.replacement\n        },\n        return_words:",
    ),
    "spin_predicate_always_succeeds": (
        "src/atomic_exec.rs",
        "AtomicExecution",
        "predicate: !spin || result.swapped,",
        "predicate: true,",
    ),
    "failed_cas_advances_epoch": (
        "src/atomic_exec.rs",
        "AtomicExecution",
        "advance_epoch: did_write,",
        "advance_epoch: true,",
    ),
    "fp64_uses_word_width": (
        "src/atomic_exec.rs",
        "AtomicExecution",
        "Operation::AddF64 => bytes == 8,",
        "Operation::AddF64 => bytes == 4,",
    ),
    "packed_atomic_reuses_low_lane": (
        "src/atomic_exec.rs",
        "AtomicExecution",
        "let high = F::add_bf16((input.old >> 16) as u16, (input.value >> 16) as u16) as u64;",
        "let high = F::add_bf16(input.old as u16, input.value as u16) as u64;",
    ),
    "atomic_decode_reverses_bytes": (
        "src/atomic_bytes.rs",
        "AtomicBytes",
        "u16::from_le_bytes([bytes[0], bytes[1]])",
        "u16::from_le_bytes([bytes[1], bytes[0]])",
    ),
}

MUTATIONS.update(
    {
        "compact_pc_offset_permuted": (
            "src/cache_graph.rs",
            "MayWriteCompact",
            "pc: entry.page * PAGE_SIZE + offset,",
            "pc: entry.page * PAGE_SIZE + (offset ^ 1),",
        ),
        "fresh_empty_successor_not_queued": (
            "src/cache_graph.rs",
            "MayWritePropagation",
            "if fresh || changed {",
            "if (fresh && false) || changed {",
        ),
        "instruction_write_bit_erased": (
            "src/cache_graph.rs",
            "MayWriteOutgoing",
            "out[register / 64] |= 1 << (register % 64);",
            "out[register / 64] |= 0 << (register % 64);",
        ),
        "merge_intersection_loses_path_write": (
            "src/cache_graph.rs",
            "MayWriteWords",
            "let joined = target[index] | source[index];",
            "let joined = target[index] & source[index];",
        ),
        "query_bit_result_inverted": (
            "src/cache_graph.rs",
            "MayWriteQuery",
            "words[register / 64] >> (register % 64) & 1 == 1",
            "words[register / 64] >> (register % 64) & 1 == 0",
        ),
        "stopping_gate_inverted": (
            "src/cache_graph.rs",
            "MayWriteMainLoop",
            "if graph.reach[pc] & bit == 0 {",
            "if graph.reach[pc] & bit != 0 {",
        ),
    }
)

MUTATIONS.update(
    {
        "fresh_dense_id_reused": (
            "src/register_index.rs",
            "RegisterIndexIntern",
            "let index = self.count;",
            "let index = 0;",
        ),
        "fresh_slot_not_published": (
            "src/register_index.rs",
            "RegisterIndexIntern",
            "self.pages[position].slots[offset] = Some(index);",
            "self.pages[position].slots[offset] = None;",
        ),
        "graph_write_id_erased": (
            "src/register_index.rs",
            "RegisterIndexBuildRow",
            "row.push(index.intern(destinations[pc][position]));",
            "let _ = index.intern(destinations[pc][position]); row.push(0);",
        ),
        "lookup_key_index_erased": (
            "src/register_index.rs",
            "RegisterIndexLookup",
            "let page = page_key(key);\n"
            "        let position = page_position(&self.pages, page);",
            "let page = page_key(RegisterKey { index: 0, ..key });\n"
            "        let position = page_position(&self.pages, page);",
        ),
        "register_file_page_erased": (
            "src/register_index.rs",
            "RegisterIndexKey",
            "u64::from(key.file) * 1024",
            "u64::from(key.file) * 0",
        ),
        "register_index_page_erased": (
            "src/register_index.rs",
            "RegisterIndexKey",
            "u64::from(key.index / 64)",
            "u64::from(key.index / 64) * 0",
        ),
    }
)

MUTATIONS.update(
    {
        "enqueue_without_progress": (
            "src/cache_graph.rs",
            "MayWritePropagation",
            "if fresh || changed {",
            "if true {",
        ),
        "find_pc_stops_advancing": (
            "src/cache_graph.rs",
            "MayWriteQueryProgress",
            "if rows[middle].pc < pc {\n            left = middle + 1;",
            "if rows[middle].pc < pc {\n            left = middle;",
        ),
        "pop_without_decrease": (
            "src/cache_graph.rs",
            "MayWriteMainProgress",
            "pending -= 1;",
            "pending -= 0;",
        ),
        "query_shift_can_reach_64": (
            "src/cache_graph.rs",
            "MayWriteQueryProgress",
            "words[register / 64] >> (register % 64) & 1 == 1",
            "words[register / 64] >> (register % 128) & 1 == 1",
        ),
    }
)

MUTATIONS.update(
    {
        "changed_update_ignored": (
            "src/barrier_reach.rs",
            "BarrierReachScan",
            "            changed = true;",
            "            changed = pc == 0;",
        ),
        "iteration_inverts_progress": (
            "src/barrier_reach.rs",
            "BarrierReachTotal",
            "changed = scan(successors, &mut reach);",
            "changed = !scan(successors, &mut reach);",
        ),
        "iteration_never_stops": (
            "src/barrier_reach.rs",
            "BarrierReachTotal",
            "changed = scan(successors, &mut reach);",
            "changed = scan(successors, &mut reach) || reach.len() != 0;",
        ),
        "iteration_stops_early": (
            "src/barrier_reach.rs",
            "BarrierReachTotal",
            "changed = scan(successors, &mut reach);",
            "let _ = scan(successors, &mut reach);\n        changed = false;",
        ),
        "pc_cursor_stalls": (
            "src/register_index.rs",
            "RegisterIndexBuildProgress",
            "pc += 1;",
            "pc += pc;",
        ),
        "published_count_erased": (
            "src/register_index.rs",
            "RegisterGraphPublication",
            "graph.register_count = index.len();",
            "graph.register_count = 0;",
        ),
        "published_reach_erased": (
            "src/register_index.rs",
            "RegisterGraphPublication",
            "graph.writes = writes;",
            "graph.reach = Vec::new();\n    graph.writes = writes;",
        ),
        "published_writes_erased": (
            "src/register_index.rs",
            "RegisterGraphPublication",
            "graph.writes = writes;",
            "graph.writes = Vec::new();",
        ),
        "row_cursor_stalls": (
            "src/register_index.rs",
            "RegisterIndexRowProgress",
            "position += 1;",
            "position += position;",
        ),
        "union_replaced_intersection": (
            "src/barrier_reach.rs",
            "BarrierReachMerge",
            "merged |= reach[successors[position]];",
            "merged &= reach[successors[position]];",
        ),
    }
)

MUTATIONS.update(
    {
        "raw_core_erases_file": (
            "src/source_rows.rs",
            "RawSourceRows",
            "crate::symbolic::register_key(owner, file, index)",
            "crate::symbolic::register_key(owner, 0, index)",
        ),
        "raw_core_erases_owner": (
            "src/source_rows.rs",
            "RawSourceRows",
            "crate::symbolic::register_key(owner, file, index)",
            "crate::symbolic::register_key(0, file, index)",
        ),
        "raw_core_masks_index": (
            "src/source_rows.rs",
            "RawSourceRows",
            "crate::symbolic::register_key(owner, file, index)",
            "crate::symbolic::register_key(owner, file, index & 0x7fff)",
        ),
        "source_key_cursor_stalls": (
            "src/source_rows.rs",
            "SourceRows",
            "position += 1;",
            "position += position;",
        ),
        "source_row_cursor_stalls": (
            "src/source_rows.rs",
            "SourceRows",
            "pc += 1;",
            "pc += pc;",
        ),
    }
)

MUTATIONS.update(
    {
        "constructor_drops_successor_rows": (
            "src/graph_source.rs",
            "GraphSource",
            "succ: successors,",
            "succ: Vec::new(),",
        ),
        "constructor_mismatches_seeds": (
            "src/graph_source.rs",
            "GraphSource",
            "let reach = barrier_reach::propagate(&successors, seeds);",
            "let reach = barrier_reach::propagate(&successors, vec![0; seeds.len()]);",
        ),
        "constructor_omits_propagation": (
            "src/graph_source.rs",
            "GraphSource",
            "let reach = barrier_reach::propagate(&successors, seeds);",
            "let reach = seeds;",
        ),
        "constructor_overwrites_reach": (
            "src/graph_source.rs",
            "GraphSource",
            "        reach,\n        succ:",
            "        reach: Vec::new(),\n        succ:",
        ),
    }
)

MUTATIONS.update(
    {
        "clock_bulk_skips_alternate_owns": (
            "src/clock.rs",
            "RustClockBulk",
            "overlay_entry(&mut result, actor, stamp);\n        index += 1;",
            "overlay_entry(&mut result, actor, stamp);\n        index += 2;",
        ),
        "clock_bulk_skips_first_own": (
            "src/clock.rs",
            "RustClockBulk",
            "let mut result = base.to_vec();\n    let mut index = 0;",
            "let mut result = base.to_vec();\n    let mut index = 1;",
        ),
        "clock_bulk_swaps_actor_and_stamp": (
            "src/clock.rs",
            "RustClockBulk",
            "overlay_entry(&mut result, actor, stamp);",
            "overlay_entry(&mut result, stamp, actor);",
        ),
        "clock_empty_constructor_is_nonzero": (
            "src/clock.rs",
            "RustSparseClock",
            "(Vec::new(), (0, 0))",
            "(Vec::new(), (0, 1))",
        ),
        "clock_filter_drops_the_lower_bound_own": (
            "src/clock.rs",
            "RustClockFilter",
            "if own.0 >= lower {",
            "if own.0 > lower {",
        ),
        "clock_filter_import_changes_the_own_entry": (
            "src/clock.rs",
            "RustClockFilter",
            "(entries, (0, 0))",
            "(entries, (0, 1))",
        ),
        "clock_filter_inverts_empty_selection": (
            "src/clock.rs",
            "RustClockFilter",
            "if selected_len == 0 {",
            "if selected_len != 0 {",
        ),
        "clock_filter_inverts_the_decision": (
            "src/clock.rs",
            "RustClockFilter",
            "if keep.keep_entry(entry) {",
            "if !keep.keep_entry(entry) {",
        ),
        "clock_filter_keeps_known_timestamps": (
            "src/clock.rs",
            "RustClockFilter",
            "get(self.base, self.own, actor) < stamp",
            "get(self.base, self.own, actor) <= stamp",
        ),
        "clock_join_uses_minimum": (
            "src/clock.rs",
            "RustClockMerge",
            "result.push((a.0, a.1.max(b.1)));",
            "result.push((a.0, a.1.min(b.1)));",
        ),
        "clock_meet_advances_wrong_side": (
            "src/clock.rs",
            "RustClockMeet",
            "if a.0 < b.0 {\n            i += 1;",
            "if a.0 < b.0 {\n            j += 1;",
        ),
        "clock_meet_discards_positive_entries": (
            "src/clock.rs",
            "RustClockMeet",
            "if stamp > 0 {",
            "if stamp == 0 {",
        ),
        "clock_meet_keeps_larger_timestamp": (
            "src/clock.rs",
            "RustClockMeet",
            "let stamp = a.1.min(b.1);",
            "let stamp = a.1.max(b.1);",
        ),
        "clock_missing_actor_is_nonzero": (
            "src/clock.rs",
            "RustSparseClock",
            "if found { base[index].1 } else { 0 }",
            "if found { base[index].1 } else { 1 }",
        ),
        "clock_overlay_loses_larger_own": (
            "src/clock.rs",
            "RustSparseClock",
            "result[index].1 = result[index].1.max(stamp);",
            "result[index].1 = result[index].1.min(stamp);",
        ),
        "clock_read_release_drops_fresh_clock": (
            "src/clock.rs",
            "RustClockReadRelease",
            "fresh.push((read_actor(actor), stamp));",
            "fresh.push((read_actor(actor), 0));",
        ),
        "clock_read_release_drops_inherited_clock": (
            "src/clock.rs",
            "RustClockReadRelease",
            "inherited.push((actor, stamp));",
            "inherited.push((actor, 0));",
        ),
        "clock_read_release_loses_actor_flag": (
            "src/clock.rs",
            "RustClockReadRelease",
            "actor + READ_BIT",
            "actor",
        ),
        "clock_read_release_misroutes_own": (
            "src/clock.rs",
            "RustClockReadRelease",
            "overlay_entry(&mut result, read_actor(reader), done);",
            "overlay_entry(&mut result, read_actor(done), done);",
        ),
        "clock_shared_base_keeps_uncovered_own": (
            "src/clock.rs",
            "RustClockJoin",
            "get(left, left_own, right_own.0) >= right_own.1",
            "get(left, left_own, right_own.0) <= right_own.1",
        ),
        "clock_tick_does_not_increment": (
            "src/clock.rs",
            "RustClockTick",
            "let next = get(base, *own, actor) + 1;",
            "let next = get(base, *own, actor) + 0;",
        ),
        "clock_tick_injects_a_clock_when_clearing_own": (
            "src/clock.rs",
            "RustClockTick",
            "*own = (0, 0);",
            "*own = (0, 1);",
        ),
        "clock_tick_relocates_the_wrong_actor": (
            "src/clock.rs",
            "RustClockTick",
            "if own.1 != 0 && own.0 != actor {",
            "if own.1 != 0 && own.0 == actor {",
        ),
        "clock_tick_single_swaps_actor_and_timestamp": (
            "src/clock.rs",
            "RustClockTick",
            "base.push((actor, clock));",
            "base.push((clock, actor));",
        ),
        "clock_updates_coalesces_distinct_actors": (
            "src/clock.rs",
            "RustClockUpdates",
            "if result[last].0 == entry.0 {",
            "if result[last].0 != entry.0 {",
        ),
        "clock_updates_coalesces_with_minimum": (
            "src/clock.rs",
            "RustClockUpdates",
            "result[last].1 = result[last].1.max(entry.1);",
            "result[last].1 = result[last].1.min(entry.1);",
        ),
        "clock_updates_keeps_nonempty_own": (
            "src/clock.rs",
            "RustClockUpdates",
            "(merge_sorted_updates(&base, updates), (0, 0))",
            "(merge_sorted_updates(&base, updates), (0, 1))",
        ),
        "clock_updates_reverses_stream_order": (
            "src/clock.rs",
            "RustClockUpdates",
            "} else if a.0 < b.0 {\n                push_coalesced(&mut result, a);",
            "} else if a.0 > b.0 {\n                push_coalesced(&mut result, a);",
        ),
        "clock_updates_suffix_duplicates_boundary": (
            "src/clock.rs",
            "RustClockUpdates",
            "result.extend_from_slice(&suffix[1..length]);",
            "result.extend_from_slice(&suffix[0..length]);",
        ),
    }
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


MUTATIONS.update(
    {
        "atomic_component_choice_missing": (
            "src/atomic_components.rs",
            "RustAtomicComponentPlan",
            "result.push(Component::Second);",
            "result.push(Component::First);",
        ),
        "atomic_component_retires_another": (
            "src/atomic_components.rs",
            "RustAtomicComponentPlan",
            "Component::Second => plan.second = false,",
            "Component::Second => plan.first = false,",
        ),
        "atomic_component_source_reselected": (
            "src/atomic_components.rs",
            "RustAtomicComponentPlan",
            "Component::Second => input.second,",
            "Component::Second => input.first,",
        ),
        "atomic_component_address_skips_word": (
            "src/atomic_components.rs",
            "RustAtomicComponentPlan",
            "Component::Second => 4,",
            "Component::Second => 8,",
        ),
        "atomic_vector_ticket_eligible": (
            "src/atomic_components.rs",
            "RustAtomicComponentPlan",
            "!(float_add && (bytes == 8 || bytes == 16))",
            "!(float_add && bytes == 8)",
        ),
    }
)


MUTATIONS.update(
    {
        "atomic_log_uses_comparison_operand": (
            "src/atomic_exec.rs",
            "AtomicRequestLog",
            "Some(request.value as u32)",
            "Some(request.compare as u32)",
        ),
        "atomic_arithmetic_uses_comparison_operand": (
            "src/atomic_exec.rs",
            "AtomicRequestLog",
            "value: request.value,",
            "value: request.compare,",
        ),
        "atomic_log_accepts_unsupported_width": (
            "src/atomic_exec.rs",
            "AtomicRequestLog",
            "matches!(request.op, Operation::AddF32Ftz) && request.bytes == 4",
            "matches!(request.op, Operation::AddF32Ftz)",
        ),
    }
)


MUTATIONS.update(
    {
        "confluence_checks_only_initial_commutation": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "!commutes_layer::<F>(alphabet, domain)",
            "!commutes_layer::<F>(alphabet, &layers[0])",
        ),
        "confluence_omits_prefix_closure": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "if depth + 1 < layers.len() && !extends_layer::<F>(alphabet, states, &layers[depth + 1]) {",
            "if false && depth + 1 < layers.len() && !extends_layer::<F>(alphabet, states, &layers[depth + 1]) {",
        ),
        "confluence_omits_initial_membership": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "!contains(&layers[0], start)",
            "false && !contains(&layers[0], start)",
        ),
        "confluence_omits_operand_membership": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "!subset(values, alphabet)",
            "false && !subset(values, alphabet)",
        ),
        "confluence_omits_layer_domain_membership": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "if !subset(states, domain) {",
            "if false && !subset(states, domain) {",
        ),
        "confluence_omits_reference_equality": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "    reference::<F>(values, start) == want\n}",
            "    reference::<F>(values, start) == reference::<F>(values, start)\n}",
        ),
        "confluence_accepts_oversized_layer": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "if states.len() > 4096 {",
            "if states.len() > 4097 {",
        ),
        "confluence_accepts_oversized_domain": (
            "src/atomic_confluence.rs",
            "RustAtomicConfluence",
            "domain.len() > 4096",
            "domain.len() > 4097",
        ),
    }
)


MUTATIONS.update(
    {
        "packed_transition_uses_the_first_half_source": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "source: crate::atomic_components::frozen_operand(input, component),",
            "source: input.first,",
        ),
        "packed_upper_uses_lower_operand": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "Some((word / 65536) as u16)",
            "Some(word as u16)",
        ),
        "packed_upper_uses_wrong_offset": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "Component::Second => Some(2),",
            "Component::Second => Some(4),",
        ),
        "packed_return_shifts_by_one_byte": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "(high as u32) * 65536",
            "(high as u32) * 256",
        ),
        "packed_first_return_loses_other_half": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "returns.low = old;",
            "returns.low = old;\n            returns.high = 0;",
        ),
        "packed_undefined_mask_uses_wrong_half": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "if returns.high_undefined { u16::MAX } else { 0 }",
            "if returns.low_undefined { u16::MAX } else { 0 }",
        ),
        "packed_definedness_ignores_selected_mask": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "Some(mask) => Some(mask == 0),",
            "Some(_mask) => Some(true),",
        ),
        "packed_ticket_forcing_accepts_components": (
            "src/atomic_packed.rs",
            "RustAtomicPacked",
            "!packed_bf16 || bytes != 4",
            "!packed_bf16 || bytes != 8",
        ),
    }
)


MUTATIONS.update(
    {
        "scalar_plan_accepts_the_wrong_width": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "        && bytes == 4\n        && supported_operation(operation)",
            "        && bytes == 8\n        && supported_operation(operation)",
        ),
        "scalar_plan_accepts_sass": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "if ptx\n",
            "if !ptx\n",
        ),
        "scalar_plan_issues_the_wrong_component": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "first: true,\n            second: false,",
            "first: false,\n            second: true,",
        ),
        "scalar_claim_ignores_the_forced_actor": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "Some(forced) => actor == forced,",
            "Some(_forced) => true,",
        ),
        "scalar_empty_guard_is_blocked": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "    !active || ready\n}",
            "    active && ready\n}",
        ),
        "scalar_ticket_accepts_unsupported_bundled_lanes": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "actor == other_actor || warp != other_warp || pc != other_pc || continued",
            "actor == other_actor || warp == other_warp || pc != other_pc || continued",
        ),
    }
)


MUTATIONS.update(
    {
        "scalar_unhandled_overlap_is_accepted_clean": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "&& same_warp && (same_word || shared_result)",
            "&& same_warp && (!same_word && !shared_result)",
        ),
        "scalar_unhandled_guard_rejects_disjoint_words": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "(same_word || shared_result)",
            "(true)",
        ),
        "scalar_frozen_request_replaces_the_actual_operation": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "        op: operation,\n        bytes: 4,",
            "        op: Operation::Integer(IntegerOp::Add),\n        bytes: 4,",
        ),
        "scalar_frozen_request_replaces_the_latched_source": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "value: source.captured as u64,",
            "value: 0,",
        ),
        "scalar_exchange_replacement_depends_on_undefined_old": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "Operation::Integer(IntegerOp::Exch) => source_defined,",
            "Operation::Integer(IntegerOp::Exch) => old_defined && source_defined,",
        ),
    }
)


MUTATIONS.update(
    {
        "scalar_float_plan_selects_f64_instead_of_f32": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "        Operation::AddF32Ftz\n            | Operation::Integer(",
            "        Operation::AddF64\n            | Operation::Integer(",
        ),
    }
)


MUTATIONS.update(
    {
        "scalar_uncontinued_pair_is_silenced": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "    ptx && result_used && !continued && same_warp",
            "    ptx && result_used && continued && same_warp",
        ),
        "scalar_pair_reports_only_returnless_accesses": (
            "src/atomic_scalar.rs",
            "RustAtomicScalar",
            "    ptx && result_used && !continued && same_warp",
            "    ptx && !result_used && !continued && same_warp",
        ),
    }
)


MUTATIONS.update(
    {
        "lane_scan_omits_every_suffix": (
            "src/atomic_scalar.rs",
            "RustAtomicLaneScan",
            "            index + 1,",
            "            claims.len(),",
        ),
        "lane_scan_skips_alternate_suffix_claims": (
            "src/atomic_scalar.rs",
            "RustAtomicLaneScan",
            "        index += 1;\n    }\n    false\n}\n\npub fn unmodeled_claims",
            "        index += 2;\n    }\n    false\n}\n\npub fn unmodeled_claims",
        ),
        "lane_scan_reverses_word_offsets": (
            "src/atomic_scalar.rs",
            "RustAtomicLaneScan",
            "left.2 == right.2 && left.3 == right.3;",
            "left.2 == right.2 && left.3 != right.3;",
        ),
        "lane_scan_conflates_different_warps": (
            "src/atomic_scalar.rs",
            "RustAtomicLaneScan",
            "        left.0 == right.0,",
            "        true,",
        ),
    }
)


MUTATIONS.update(
    {
        "returned_word_coverage_keeps_a_frontend_exemption": (
            "src/atomic_scalar.rs",
            "RustAtomicReturningCoverage",
            "unmodeled_claims(true, true, continued, shared_result, claims)",
            "unmodeled_claims(false, true, continued, shared_result, claims)",
        ),
        "returned_word_coverage_forgets_the_observed_return": (
            "src/atomic_scalar.rs",
            "RustAtomicReturningCoverage",
            "unmodeled_claims(true, true, continued, shared_result, claims)",
            "unmodeled_claims(true, false, continued, shared_result, claims)",
        ),
        "returned_word_coverage_reverses_continuation": (
            "src/atomic_scalar.rs",
            "RustAtomicReturningCoverage",
            "unmodeled_claims(true, true, continued, shared_result, claims)",
            "unmodeled_claims(true, true, !continued, shared_result, claims)",
        ),
    }
)


MUTATIONS.update(
    {
        "local_atomic_coverage_defers_every_certificate_monitor": (
            "src/atomic_scalar.rs",
            "RustAtomicLocalCoverage",
            "(monitored || scalar_red && !certificate_collection) && conflict",
            "(monitored || scalar_red) && !certificate_collection && conflict",
        ),
        "local_atomic_coverage_drops_ordinary_red": (
            "src/atomic_scalar.rs",
            "RustAtomicLocalCoverage",
            "(monitored || scalar_red && !certificate_collection) && conflict",
            "monitored && conflict",
        ),
        "local_atomic_coverage_drops_monitored_returns_and_cas": (
            "src/atomic_scalar.rs",
            "RustAtomicLocalCoverage",
            "(monitored || scalar_red && !certificate_collection) && conflict",
            "scalar_red && !certificate_collection && conflict",
        ),
        "local_atomic_coverage_forgets_shared_return_owners": (
            "src/atomic_scalar.rs",
            "RustAtomicLocalCoverage",
            "returned_word_claims_unmodeled(continued, monitored && shared_result, claims)",
            "returned_word_claims_unmodeled(continued, false, claims)",
        ),
        "local_atomic_coverage_reverses_continuation": (
            "src/atomic_scalar.rs",
            "RustAtomicLocalCoverage",
            "returned_word_claims_unmodeled(continued, monitored && shared_result, claims)",
            "returned_word_claims_unmodeled(!continued, monitored && shared_result, claims)",
        ),
    }
)


def source_hashes() -> dict[str, str]:
    files = [
        PROBE / path
        for path in RUST_FILES
        + ["check.py", "production_check.py", "production_replay.rs", "coverage.json"]
    ]
    files += [
        PROBE / "controls/ProgressWitness.lean",
        PROBE / "controls/AuditWitness.lean",
    ]
    files += [PROBE / "clock-producers.json", ROOT / "lean_model/generated_lean.py"]
    files += [RUST_CRATE / path for path in RUST_SOURCES]
    files += [
        ROOT / path
        for path in [
            "Cargo.lock",
            "crates/sim/Cargo.toml",
            "crates/sim/src/num.rs",
            "crates/sim/src/exec.rs",
            "crates/sim/src/exec/atomic_components.rs",
            "crates/sim/src/atomic_component_tests.rs",
            "crates/sim/src/atomic_packed_tests.rs",
            "crates/sim/src/atomic_scalar_order_tests.rs",
            "crates/sim/src/atomic_lane_coverage_tests.rs",
            "crates/sim/src/atomic_sass_order_tests.rs",
            "crates/sim/src/atomic_red_coverage_tests.rs",
            "crates/sim/src/atomic_order_cases/unused_cas.ptx",
            "crates/sim/src/atomic_order_cases/max_spin_single.ptx",
            "crates/sim/src/atomic_order_cases/max_spin_private.ptx",
            "crates/sim/src/atomic_order_cases/max_spin.ptx",
            "crates/sim/src/atomic_predicate_coverage_tests.rs",
            "crates/sim/src/atomic_order_cases/unused_exch.ptx",
            "crates/sim/src/atomic_exchange_coverage_tests.rs",
            "crates/ir/src/register_owner.rs",
            "crates/ir/src/lib.rs",
            "crates/ir/src/atomic_result.rs",
            "crates/sim/src/atomic_log_tests.rs",
            "crates/sim/src/dpor.rs",
            "crates/sim/src/wellsync.rs",
            "crates/sim/src/wellsync/fadd.rs",
            "crates/sim/src/wellsync/ticket.rs",
            "crates/sim/src/coverage.rs",
            "crates/sim/src/memory_effect.rs",
            "crates/sim/src/halfin.rs",
            "crates/sim/src/elementwise.rs",
            "crates/sim/src/equiv.rs",
            "crates/sim/src/spin.rs",
            "crates/sim/src/explore.rs",
            "crates/sim/src/bar_program.rs",
            "crates/sim/src/machine.rs",
            "crates/sim/src/reduction_tests.rs",
            "crates/sim/tests/fixtures.rs",
            "fixtures/mbar_plain_generation_order/kernel.ptx",
            "fixtures/mbar_transaction_generation_order/kernel.ptx",
            "fixtures/mbar_transaction_generation_order/case.toml",
            "fixtures/mbar_wait_uninitialized/kernel.ptx",
            "fixtures/mbar_wait_uninitialized/case.toml",
            "fixtures/mbar_peek_uninitialized/kernel.ptx",
            "fixtures/mbar_peek_uninitialized/case.toml",
            "fixtures/mbar_forced_peek_uninitialized/kernel.ptx",
            "fixtures/mbar_forced_peek_uninitialized/case.toml",
            "fixtures/mbar_plain_generation_order/case.toml",
            "crates/sim/src/wellsync/ring.rs",
            "crates/sim/src/atomic_memory_tests.rs",
            "crates/sim/src/atomic_exec.rs",
            "crates/sim/src/lib.rs",
            "crates/ptx/src/lib.rs",
        ]
    ]
    files += [LEAN / path for path in LEAN_FILES + GENERATED]
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


MUTATIONS.update(
    {
        "arrival_walk_next_drops_successor": (
            "src/arrival_walk.rs",
            "ArrivalWalkLoop",
            "                    Expansion::Next(state) => {\n"
            "                        if !pending.push(state) {\n"
            "                            return true;\n"
            "                        }\n"
            "                    }",
            "                    Expansion::Next(_state) => {}",
        ),
        "arrival_walk_fresh_reversed": (
            "src/arrival_walk.rs",
            "ArrivalWalkLoop",
            "if !walker.fresh(&state) {",
            "if walker.fresh(&state) {",
        ),
        "arrival_walk_exhaustion_returns_false": (
            "src/arrival_walk.rs",
            "ArrivalWalkLoop",
            "        }\n    }\n    true\n}\n\nimpl<State> Default",
            "        }\n    }\n    false\n}\n\nimpl<State> Default",
        ),
        "arrival_walk_budget_never_decreases": (
            "src/arrival_walk.rs",
            "ArrivalWalkLoop",
            "budget -= 1;",
            "budget -= 0;",
        ),
        "arrival_walk_retry_is_skipped": (
            "src/arrival_walk.rs",
            "ArrivalWalkLoop",
            "if started && !walker.retry() {",
            "if started {",
        ),
        "arrival_walk_dependency_scan_skips_first": (
            "src/arrival_walk.rs",
            "ArrivalWalkDependencies",
            "let mut index = 0;",
            "let mut index = 1;",
        ),
    }
)


MUTATIONS.update(
    {
        "convergence_barrier_aliases_uniform_register_namespace": (
            "src/footprint.rs",
            "ConvergenceBarrierFootprint",
            "(19, warp, start, start + 1)",
            "(17, warp, start, start + 1)",
        )
    }
)


MUTATIONS.update(
    {
        "effects-fault-discards-prefix": (
            "src/effects.rs",
            "EffectsComposition",
            "        Effect::Fault => {\n            state.invalidated = true;",
            "        Effect::Fault => {\n"
            "            state.summary.writes = Vec::new();\n"
            "            state.summary.releases = Vec::new();\n"
            "            state.invalidated = true;",
        ),
        "effects-convergence-aliases-uniform": (
            "src/effects.rs",
            "EffectsNamespaces",
            "        ResourceSpace::ConvergenceBarrier => 19,",
            "        ResourceSpace::ConvergenceBarrier => 17,",
        ),
    }
)


def run(
    command: Sequence[str | Path],
    cwd: Path,
    log: Path,
    env: dict[str, str],
    expect_success: bool = True,
) -> int:
    LOGGER.info("Running %s; log: %s", command[0], log)
    with log.open("w") as output:
        result = subprocess.run(
            [
                "prlimit",
                "--as=51539607552",
                "--data=51539607552",
                "--",
                *map(str, command),
            ],
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    if expect_success and result.returncode:
        tail = "\n".join(log.read_text().splitlines()[-30:])
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{tail}")
    return result.returncode


def configure(project: Path, backend: Path) -> None:
    manifest = json.loads((backend / "lake-manifest.json").read_text())
    manifest["name"] = "ptxv_memory_write"
    for package in manifest["packages"]:
        package["inherited"] = True
    manifest["packages"] = [p for p in manifest["packages"] if p["name"] != "aeneas"]
    manifest["packages"].append(
        {
            "type": "path",
            "name": "aeneas",
            "dir": str(backend),
            "manifestFile": "lake-manifest.json",
            "configFile": "lakefile.lean",
            "inherited": False,
        }
    )
    (project / ".aeneas-path").write_text(str(backend) + "\n")
    (project / "lake-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    packages = project / ".lake/packages"
    packages.mkdir(parents=True, exist_ok=True)
    for entry in (backend / ".lake/packages").iterdir():
        target = packages / entry.name
        if target.is_symlink() or target.exists():
            if target.resolve() != entry.resolve():
                raise RuntimeError(f"Dependency path mismatch: {target}")
        else:
            target.symlink_to(entry.resolve(), target_is_directory=True)


def extract(
    crate: Path,
    destination: Path,
    tools: dict[str, str],
    env: dict[str, str],
    logs: Path,
) -> None:
    llbc = logs / "MemoryWrite.llbc"
    run(
        [
            tools["charon"],
            "cargo",
            "--preset=aeneas",
            "--extract-opaque-bodies",
            "--start-from",
            "ptxv_memory_write::machine_action::new",
            "--start-from",
            "ptxv_memory_write::machine_action::is_recording",
            "--start-from",
            "ptxv_memory_write::machine_action::snapshot",
            "--start-from",
            "ptxv_memory_write::machine_action::set_footprint",
            "--start-from",
            "ptxv_memory_write::machine_action::take_footprint",
            "--start-from",
            "ptxv_memory_write::machine_action::begin",
            "--start-from",
            "ptxv_memory_write::machine_action::apply_route",
            "--start-from",
            "ptxv_memory_write::machine_action::record_access",
            "--start-from",
            "ptxv_memory_write::machine_action::record_release",
            "--start-from",
            "ptxv_memory_write::machine_action::record_acquire",
            "--start-from",
            "ptxv_memory_write::machine_action::record_fault",
            "--start-from",
            "ptxv_memory_write::machine_action::record_finding",
            "--start-from",
            "ptxv_memory_write::machine_action::record_arrival",
            "--start-from",
            "ptxv_memory_write::machine_action::finish",
            "--start-from",
            "ptxv_memory_write::machine_action::write_uniform",
            "--start-from",
            "ptxv_memory_write::machine_action::write_mask",
            "--start-from",
            "ptxv_memory_write::machine_footprint::append_range",
            "--start-from",
            "ptxv_memory_write::machine_footprint::Footprint::add",
            "--start-from",
            "ptxv_memory_write::machine_footprint::merge_summary",
            "--start-from",
            "ptxv_memory_write::machine_footprint::begin_context",
            "--start-from",
            "ptxv_memory_write::machine_footprint::record_context",
            "--start-from",
            "ptxv_memory_write::machine_footprint::finish_context",
            "--start-from",
            "ptxv_memory_write::machine_footprint::mode",
            "--start-from",
            "ptxv_memory_write::machine_footprint::has_membership",
            "--start-from",
            "ptxv_memory_write::machine_footprint::Footprint::conflicts",
            "--start-from",
            "ptxv_memory_write::machine_recorder::new",
            "--start-from",
            "ptxv_memory_write::machine_recorder::fresh_snapshot",
            "--start-from",
            "ptxv_memory_write::machine_recorder::begin",
            "--start-from",
            "ptxv_memory_write::machine_recorder::route",
            "--start-from",
            "ptxv_memory_write::machine_recorder::record_access",
            "--start-from",
            "ptxv_memory_write::machine_recorder::record_release",
            "--start-from",
            "ptxv_memory_write::machine_recorder::record_acquire",
            "--start-from",
            "ptxv_memory_write::machine_recorder::record_fault",
            "--start-from",
            "ptxv_memory_write::machine_recorder::record_finding",
            "--start-from",
            "ptxv_memory_write::machine_recorder::record_arrival",
            "--start-from",
            "ptxv_memory_write::machine_recorder::finish",
            "--start-from",
            "ptxv_memory_write::machine_registers::uniform_access",
            "--start-from",
            "ptxv_memory_write::machine_warp::mask_access",
            "--start-from",
            "ptxv_memory_write::machine_control::set_pc",
            "--start-from",
            "ptxv_memory_write::machine_control::advance",
            "--start-from",
            "ptxv_memory_write::machine_control::set_status",
            "--start-from",
            "ptxv_memory_write::machine_control::apply_flow",
            "--start-from",
            "ptxv_memory_write::machine_control::resume",
            "--start-from",
            "ptxv_memory_write::machine_control::release",
            "--start-from",
            "ptxv_memory_write::machine_control::take_acquire",
            "--start-from",
            "ptxv_memory_write::machine_warp::mask_effect",
            "--start-from",
            "ptxv_memory_write::machine_warp::write_mask",
            "--start-from",
            "ptxv_memory_write::machine_warp::commit_async",
            "--start-from",
            "ptxv_memory_write::machine_registers::set_word",
            "--start-from",
            "ptxv_memory_write::machine_registers::slot_undefined",
            "--start-from",
            "ptxv_memory_write::machine_registers::set_bit",
            "--start-from",
            "ptxv_memory_write::machine_registers::set_virtual_undefined",
            "--start-from",
            "ptxv_memory_write::machine_registers::write_virtual",
            "--start-from",
            "ptxv_memory_write::machine_registers::write_scalar",
            "--start-from",
            "ptxv_memory_write::machine_registers::write_predicate",
            "--start-from",
            "ptxv_memory_write::machine_registers::uniform_effect",
            "--start-from",
            "ptxv_memory_write::machine_registers::write_uniform",
            "--start-from",
            "ptxv_memory_write::machine_registers::write_uniform_predicate",
            "--start-from",
            "ptxv_memory_write::machine_registers::poison_scalar",
            "--start-from",
            "ptxv_memory_write::machine_registers::poison_predicate",
            "--start-from",
            "ptxv_memory_write::machine_registers::poison_uniform",
            "--start-from",
            "ptxv_memory_write::machine_registers::poison_uniform_predicate",
            "--start-from",
            "ptxv_memory_write::machine_registers::fill_scalar_if_unwritten",
            "--start-from",
            "ptxv_memory_write::machine_registers::fill_uniform_if_unwritten",
            "--start-from",
            "ptxv_memory_write::atomic_exec::resolve_old",
            "--start-from",
            "ptxv_memory_write::atomic_exec::fadd_operand",
            "--start-from",
            "ptxv_memory_write::atomic_components::ticket_access_eligible",
            "--start-from",
            "ptxv_memory_write::atomic_components::choices",
            "--start-from",
            "ptxv_memory_write::atomic_components::begin_component",
            "--start-from",
            "ptxv_memory_write::atomic_components::frozen_operand",
            "--start-from",
            "ptxv_memory_write::atomic_components::frozen_address",
            "--start-from",
            "ptxv_memory_write::atomic_components::vector_plan",
            "--start-from",
            "ptxv_memory_write::atomic_components::pending",
            "--start-from",
            "ptxv_memory_write::atomic_components::finish",
            "--start-from",
            "ptxv_memory_write::atomic_components::retired",
            "--start-from",
            "ptxv_memory_write::atomic_components::byte_offset",
            "--start-from",
            "ptxv_memory_write::apply_copy_effect",
            "--start-from",
            "ptxv_memory_write::footprint::conflicts",
            "--start-from",
            "ptxv_memory_write::footprint::shared_register_range",
            "--start-from",
            "ptxv_memory_write::footprint::convergence_barrier_range",
            "--start-from",
            "ptxv_memory_write::cache::next_generation",
            "--start-from",
            "ptxv_memory_write::cache::make_key",
            "--start-from",
            "ptxv_memory_write::cache::same_key",
            "--start-from",
            "ptxv_memory_write::cache::same_observation",
            "--start-from",
            "ptxv_memory_write::cache::begin_replay",
            "--start-from",
            "ptxv_memory_write::cache::observe",
            "--start-from",
            "ptxv_memory_write::cache::replay_possible",
            "--start-from",
            "ptxv_memory_write::cache::finish_replay",
            "--start-from",
            "ptxv_memory_write::cache_graph::valid_graph",
            "--start-from",
            "ptxv_memory_write::register_index::build_writes",
            "--start-from",
            "ptxv_memory_write::register_index::publish_graph",
            "--start-from",
            "ptxv_memory_write::source_rows::from_raw",
            "--start-from",
            "ptxv_memory_write::graph_source::build",
            "--start-from",
            "ptxv_memory_write::barrier_reach::propagate",
            "--start-from",
            "ptxv_memory_write::register_index::RegisterIndex::get",
            "--start-from",
            "ptxv_memory_write::register_index::RegisterIndex::len",
            "--start-from",
            "ptxv_memory_write::register_index::RegisterIndex::is_empty",
            "--start-from",
            "ptxv_memory_write::cache_graph::may_write_sets",
            "--start-from",
            "ptxv_memory_write::cache_graph::may_at",
            "--start-from",
            "ptxv_memory_write::definedness::selected_undefined",
            "--start-from",
            "ptxv_memory_write::definedness::byte_mask",
            "--start-from",
            "ptxv_memory_write::definedness::merge_byte_mask",
            "--start-from",
            "ptxv_memory_write::definedness::written_undefined",
            "--start-from",
            "ptxv_memory_write::store::materialize_opaque",
            "--start-from",
            "ptxv_memory_write::checked::checked_write",
            "--start-from",
            "ptxv_memory_write::checked::capture_span",
            "--start-from",
            "ptxv_memory_write::engine::apply",
            "--start-from",
            "ptxv_memory_write::numeric::forget_numeric",
            "--start-from",
            "ptxv_memory_write::atomic::atomic_footprint",
            "--start-from",
            "ptxv_memory_write::atomic_bytes::decode",
            "--start-from",
            "ptxv_memory_write::atomic_bytes::encode",
            "--start-from",
            "ptxv_memory_write::atomic_integer::eval",
            "--start-from",
            "ptxv_memory_write::atomic_exec::execute",
            "--start-from",
            "ptxv_memory_write::symbolic::register_key",
            "--start-from",
            "ptxv_memory_write::symbolic::checked_node_id",
            "--start-from",
            "ptxv_memory_write::symbolic::classify_missing",
            "--start-from",
            "ptxv_memory_write::symbolic::prepare",
            "--start-from",
            "ptxv_memory_write::symbolic::assign_metadata",
            "--start-from",
            "ptxv_memory_write::symbolic::destination_plan",
            "--start-from",
            "ptxv_memory_write::symbolic::binary",
            "--start-from",
            "ptxv_memory_write::symbolic::control",
            "--start-from",
            "ptxv_memory_write::store::prepare_word",
            "--start-from",
            "ptxv_memory_write::store::commit_word",
            "--start-from",
            "ptxv_memory_write::execution::same_words",
            "--start-from",
            "ptxv_memory_write::execution::same_undefined",
            "--start-from",
            "ptxv_memory_write::execution::same_registers",
            "--start-from",
            "ptxv_memory_write::execution::same_memory",
            "--start-from",
            "ptxv_memory_write::execution::begin_audit",
            "--start-from",
            "ptxv_memory_write::execution::record_store",
            "--start-from",
            "ptxv_memory_write::execution::accepts",
            "--start-from",
            "ptxv_memory_write::reduction::arrival_pending",
            "--start-from",
            "ptxv_memory_write::reduction::phase_completes",
            "--start-from",
            "ptxv_memory_write::reduction::arrival_writes_pending",
            "--start-from",
            "ptxv_memory_write::reduction::allocation_ready",
            "--start-from",
            "ptxv_memory_write::clock::empty",
            "--start-from",
            "ptxv_memory_write::clock::single",
            "--start-from",
            "ptxv_memory_write::clock::relocate",
            "--start-from",
            "ptxv_memory_write::clock::install_relocation",
            "--start-from",
            "ptxv_memory_write::clock::tick_own",
            "--start-from",
            "ptxv_memory_write::clock::overlay_entry",
            "--start-from",
            "ptxv_memory_write::clock::join_shared",
            "--start-from",
            "ptxv_memory_write::clock::meet_entries",
            "--start-from",
            "ptxv_memory_write::clock::meet_clock",
            "--start-from",
            "ptxv_memory_write::clock::filter_entries",
            "--start-from",
            "ptxv_memory_write::clock::uncovered_clock",
            "--start-from",
            "ptxv_memory_write::clock::read_release_clock",
            "--start-from",
            "ptxv_memory_write::clock::tail",
            "--start-from",
            "ptxv_memory_write::clock::from_sorted",
            "--start-from",
            "ptxv_memory_write::clock::join_sorted_updates",
            "--start-from",
            "ptxv_memory_write::clock::locate",
            "--start-from",
            "ptxv_memory_write::clock::lookup",
            "--start-from",
            "ptxv_memory_write::clock::get",
            "--start-from",
            "ptxv_memory_write::clock::entries",
            "--start-from",
            "ptxv_memory_write::clock::merge_entries",
            "--start-from",
            "ptxv_memory_write::clock::join_plan",
            "--start-from",
            "ptxv_memory_write::clock::merge_clock",
            "--start-from",
            "ptxv_memory_write::barrier::transition",
            "--start-from",
            "ptxv_memory_write::barrier::writes_phase_membership",
            "--start-from",
            "ptxv_memory_write::barrier::batch_writes_phase_membership",
            "--start-from",
            "ptxv_memory_write::barrier::next_arrival_key",
            "--start-from",
            "ptxv_memory_write::barrier::phase_key",
            "--start-from",
            "ptxv_memory_write::barrier::publication_parity",
            "--start-from",
            "ptxv_memory_write::barrier::poll_release",
            "--start-from",
            "ptxv_memory_write::barrier::mark_wait_error",
            "--start-from",
            "ptxv_memory_write::barrier::clear_wait_error",
            "--start-from",
            "ptxv_memory_write::barrier::wait_can_progress",
            "--start-from",
            "ptxv_memory_write::barrier::wait_mode",
            "--start-from",
            "ptxv_memory_write::barrier::wait_policy",
            "--start-from",
            "ptxv_memory_write::barrier::begin_update",
            "--start-from",
            "ptxv_memory_write::barrier::publish_release",
            "--start-from",
            "ptxv_memory_write::barrier::finish_update",
            "--start-from",
            "ptxv_memory_write::atomic_confluence::check",
            "--start-from",
            "ptxv_memory_write::atomic_packed::plan",
            "--start-from",
            "ptxv_memory_write::atomic_packed::offset",
            "--start-from",
            "ptxv_memory_write::atomic_packed::operand",
            "--start-from",
            "ptxv_memory_write::atomic_packed::operand_defined",
            "--start-from",
            "ptxv_memory_write::atomic_packed::ticket_access_eligible",
            "--start-from",
            "ptxv_memory_write::atomic_packed::assemble",
            "--start-from",
            "ptxv_memory_write::atomic_packed::empty_returns",
            "--start-from",
            "ptxv_memory_write::atomic_packed::record",
            "--start-from",
            "ptxv_memory_write::atomic_packed::returned",
            "--start-from",
            "ptxv_memory_write::atomic_packed::undefined_mask",
            "--start-from",
            "ptxv_memory_write::atomic_packed::begin_component",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::plan",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::claim_allowed",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::ticket_progress",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::ticket_pair_eligible",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::supported_operation",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::request",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::replacement_defined",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::bundled_order_unmodeled",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::unmodeled_claims",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::returned_word_claims_unmodeled",
            "--start-from",
            "ptxv_memory_write::atomic_scalar::local_claims_unmodeled",
            "--start-from",
            "ptxv_memory_write::arrival_walk::may_arrive",
            "--start-from",
            "ptxv_memory_write::arrival_walk::has_conflict",
            "--start-from",
            "ptxv_memory_write::barrier::arrival_batch",
            "--start-from",
            "ptxv_memory_write::barrier::invalidate_arrival_batch",
            "--start-from",
            "ptxv_memory_write::barrier::record_arrival_batch",
            "--start-from",
            "ptxv_memory_write::barrier::arrival_batch_writes",
            "--start-from",
            "ptxv_memory_write::effects::begin",
            "--start-from",
            "ptxv_memory_write::effects::resource_tag",
            "--start-from",
            "ptxv_memory_write::effects::resource_range",
            "--start-from",
            "ptxv_memory_write::effects::uniform_register",
            "--start-from",
            "ptxv_memory_write::effects::convergence_barrier",
            "--start-from",
            "ptxv_memory_write::effects::included",
            "--start-from",
            "ptxv_memory_write::effects::poisons_arrivals",
            "--start-from",
            "ptxv_memory_write::effects::record_arrival",
            "--start-from",
            "ptxv_memory_write::effects::apply",
            "--start-from",
            "ptxv_memory_write::effects::fold_from",
            "--start-from",
            "ptxv_memory_write::effects::finish",
            "--start-from",
            "ptxv_memory_write::effects::fold",
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
            "--include",
            "core::ops::range::Range",
            "--error-on-warnings",
            "--abort-on-error",
            "--dest-file",
            llbc,
            "--",
            "--offline",
            "--locked",
            "--lib",
        ],
        crate,
        logs / "extract.log",
        env,
    )
    run(
        [
            tools["aeneas"],
            "-backend",
            "lean",
            "-namespace",
            "PtxvMemoryWrite",
            "-dest",
            destination,
            "-subdir",
            "MemoryWrite/Code",
            "-split-files",
            "-abort-on-error",
            "-no-progress-bar",
            "-all-computable",
            llbc,
        ],
        crate,
        logs / "translate.log",
        env,
    )
    LEAN_EXPORT.publish(destination / "MemoryWrite/Code", logs / "raw-generated")
    for name in GENERATED:
        text = (destination / name).read_text()
        if "sorry" in text or "axiom " in text or "external" in text.lower():
            raise RuntimeError(f"Unexpected stub in generated source: {name}")


def prepare_mutant(work: Path, source: str, old: str, new: str) -> tuple[Path, Path]:
    original = (RUST_CRATE / source).read_text()
    if original.count(old) != 1:
        raise RuntimeError(f"Mutation target is not unique: {work.name}")
    crate, project = work / "rust", work / "lean"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    project.mkdir(parents=True, exist_ok=True)
    (crate / "Cargo.toml").write_text(
        (RUST_CRATE / "Cargo.toml").read_text() + "\n[workspace]\n"
    )
    (crate / "Cargo.lock").write_bytes((PROBE / "Cargo.lock").read_bytes())
    for path in RUST_SOURCES:
        if path.startswith("src/"):
            (crate / path).write_bytes((RUST_CRATE / path).read_bytes())
    (crate / source).write_text(original.replace(old, new))
    for path in LEAN_FILES:
        target = project / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((LEAN / path).read_bytes())
    return crate, project


def run_native_mutation(work: Path, crate: Path, env: dict[str, str]) -> str:
    command = ["cargo", "test", "--offline", "--locked"]
    log = work / "rust-tests.log"
    if work.name not in {
        "enqueue_without_progress",
        "arrival_walk_budget_never_decreases",
        "find_pc_stops_advancing",
        "iteration_never_stops",
        "iteration_inverts_progress",
        "source_key_cursor_stalls",
        "source_row_cursor_stalls",
        "row_cursor_stalls",
        "pc_cursor_stalls",
    }:
        status = run(command, crate, log, env, expect_success=False)
        if status == 0 or "test result: FAILED" not in log.read_text():
            raise RuntimeError(
                f"Mutation escaped the Rust regression tests: {work.name}"
            )
        return "test_failure"
    run(command + ["--no-run"], crate, work / "rust-compile.log", env)
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                "prlimit",
                "--as=51539607552",
                "--data=51539607552",
                "--",
                *command,
            ],
            cwd=crate,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            status = process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            output.write(
                "\nCONTROL: native tests exceeded the eight-second progress budget\n"
            )
            return "termination_timeout"
    if status == 0 or "test result: FAILED" not in log.read_text():
        raise RuntimeError(f"Mutation escaped the Rust regression tests: {work.name}")
    return "test_failure"


def check_progress_witness(
    work: Path, project: Path, tools: dict[str, str], env: dict[str, str]
) -> None:
    for name in ["ProgressWitness.lean", "AuditWitness.lean"]:
        shutil.copyfile(PROBE / "controls" / name, project / name)
    lakefile = project / "lakefile.lean"
    lakefile.write_text(lakefile.read_text() + "\nlean_lib ProgressWitness\n")
    run(
        [tools["lake"], "build", "ProgressWitness"],
        project,
        work / "progress-witness.log",
        env,
    )
    run(
        [tools["lake"], "env", "lean", "-j4", "AuditWitness.lean"],
        project,
        work / "progress-witness-audit.log",
        env,
    )


def check_mutant(
    work: Path, proof: str, tools: dict[str, str], env: dict[str, str]
) -> str:
    crate, project = work / "rust", work / "lean"
    native_result = run_native_mutation(work, crate, env)
    configure(project, Path(tools["backend"]))
    lakefile = project / "lakefile.lean"
    configuration = lakefile.read_text()
    arguments = 'moreLeanArgs := #["-j4"]'
    if configuration.count(arguments) != 1:
        raise RuntimeError("Expected one negative proof argument configuration")
    lakefile.write_text(
        configuration.replace(arguments, 'moreLeanArgs := #["-j4", "-DmaxErrors=1"]')
    )
    extract(crate, project, tools, env, work)
    run(
        [tools["lake"], "build", "MemoryWrite.Code.Funs"],
        project,
        work / "code.log",
        env,
    )
    status = run(
        [tools["lake"], "build", f"MemoryWrite.{proof}"],
        project,
        work / "proof.log",
        env,
        expect_success=False,
    )
    proof_log = (work / "proof.log").read_text()
    if status == 0 or f"error: MemoryWrite/{proof}.lean:" not in proof_log:
        raise RuntimeError(f"Mutation escaped the source proof: {work.name}")
    if "Unknown identifier" in proof_log or "Unknown constant" in proof_log:
        raise RuntimeError(f"Unrelated source proof failure: {work.name}")
    if work.name == "iteration_inverts_progress":
        check_progress_witness(work, project, tools, env)
    if work.name == "arrival_walk_retry_is_skipped":
        run(
            [tools["lake"], "env", "lean", "-j4", "ArrivalWalkRetryWitness.lean"],
            project,
            work / "arrival-retry-witness.log",
            env,
        )
    return native_result


def negative_controls(
    out: Path, tools: dict[str, str], env: dict[str, str]
) -> dict[str, dict[str, Any]]:
    results = {}
    for name, (source, proof, old, new) in MUTATIONS.items():
        work = out / "negative" / name
        prepare_mutant(work, source, old, new)
        native_result = check_mutant(work, proof, tools, env)
        results[name] = {
            "rust_tests_rejected": native_result == "test_failure",
            "native_termination_rejected": native_result == "termination_timeout",
            "native_outcome": native_result,
            "independent_arrival_retry_witness_audited": name
            == "arrival_walk_retry_is_skipped",
            "independent_progress_witness_audited": name
            == "iteration_inverts_progress",
            "extraction_compiles": True,
            "source_proof_rejected": True,
        }
        LOGGER.info("Negative control rejected: %s", name)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--production-source", type=Path)
    parser.add_argument("--production-revision")
    parser.add_argument("--production-target", type=Path)
    parser.add_argument("--production-cargo-lock", type=Path)
    args = parser.parse_args()
    if bool(args.production_source) != bool(args.production_revision):
        parser.error(
            "--production-source and --production-revision are required together"
        )
    if (
        args.production_target or args.production_cargo_lock
    ) and not args.production_source:
        parser.error("Production target/lock options require --production-source")
    return args


def verification_environment(out: Path, tools: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(out / "target")
    env["CARGO_NET_OFFLINE"] = "true"
    env["PATH"] = str(Path(tools["lake"]).parent) + os.pathsep + env["PATH"]
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PTXV_EXCLUDE_GPUS"] = "0,1,2,3,4,5,6,7"
    return env


def check_extraction(
    out: Path, tools: dict[str, str], env: dict[str, str], regenerate: bool
) -> dict[str, str]:
    original = source_hashes()
    fresh = out / "fresh"
    fresh.mkdir(exist_ok=True)
    extract(PROBE, fresh, tools, env, out)
    for path in GENERATED:
        expected, actual = LEAN / path, fresh / path
        if regenerate:
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(actual, expected)
        elif expected.read_bytes() != actual.read_bytes():
            raise RuntimeError(f"Stale extraction: {path}; rerun with --regenerate")
    before = source_hashes()
    generated = {str((LEAN / path).relative_to(ROOT)) for path in GENERATED}
    if any(
        value != before[path]
        for path, value in original.items()
        if path not in generated
    ):
        raise RuntimeError(
            "Source or proofs changed during extraction; no receipt written"
        )
    return before


def stage_lean_package(out: Path, tools: dict[str, str]) -> Path:
    project = out / "lean"
    project.mkdir(parents=True, exist_ok=False)
    for name in LEAN_FILES + GENERATED:
        target = project / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LEAN / name, target)
    configure(project, Path(tools["backend"]))
    return project


def check_module(
    out: Path, tools: dict[str, str], env: dict[str, str], project: Path | None = None
) -> None:
    project = LEAN if project is None else project
    run(["cargo", "test", "--offline", "--locked"], PROBE, out / "rust-tests.log", env)
    run([tools["lake"], "build"], project, out / "lean-build.log", env)
    for source, log in [
        ("Audit.lean", "audit.log"),
        ("Tests.lean", "lean-tests.log"),
        ("FullTests.lean", "full-tests.log"),
        ("RustTests.lean", "rust-lean-tests.log"),
    ]:
        run([tools["lake"], "env", "lean", "-j4", source], project, out / log, env)


def check_production(
    args: argparse.Namespace, out: Path, env: dict[str, str]
) -> dict[str, Any] | None:
    if not args.production_source:
        return None
    work = out / "production"
    command = [
        "python3",
        PROBE / "production_check.py",
        "--source",
        args.production_source.resolve(),
        "--revision",
        args.production_revision,
        "--work",
        work,
        "--target-dir",
        (args.production_target or out / "production-target").resolve(),
        "--lean-project",
        LEAN,
    ]
    if args.production_cargo_lock:
        command += ["--cargo-lock", args.production_cargo_lock.resolve()]
    run(command, ROOT, out / "production-check.log", env)
    return json.loads((work / "receipt.json").read_text())


def check_freshness(
    before: dict[str, str],
    tools: dict[str, str],
    tool_hashes: dict[str, str],
    backend_manifest_hash: str,
) -> None:
    if before != source_hashes():
        raise RuntimeError(
            "Source or proofs changed during verification; no receipt written"
        )
    if tool_hashes != {key: digest(Path(tools[key])) for key in tool_hashes}:
        raise RuntimeError(
            "A verification tool changed during verification; no receipt written"
        )
    if backend_manifest_hash != digest(Path(tools["backend"]) / "lake-manifest.json"):
        raise RuntimeError(
            "Backend dependencies changed during verification; no receipt written"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tools = json.loads(args.tools.read_text())
    for key in ["charon", "aeneas", "lake", "backend"]:
        tools[key] = str(Path(tools[key]).resolve(strict=True))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "receipt.json"
    receipt.unlink(missing_ok=True)
    env = verification_environment(out, tools)
    configure(LEAN, Path(tools["backend"]))
    tool_hashes = {
        key: digest(Path(tools[key])) for key in ["charon", "aeneas", "lake"]
    }
    backend_hash = digest(Path(tools["backend"]) / "lake-manifest.json")
    before = check_extraction(out, tools, env, args.regenerate)
    project = stage_lean_package(out, tools)
    check_module(out, tools, env, project)
    controls = negative_controls(out, tools, env) if args.negative_controls else {}
    production = check_production(args, out, env)
    check_freshness(before, tools, tool_hashes, backend_hash)
    receipt.write_text(
        json.dumps(
            {
                "source_sha256": before,
                "tools": tools,
                "tool_sha256": tool_hashes,
                "backend_manifest_sha256": backend_hash,
                "fresh_extraction": True,
                "generated_export": json.loads(
                    (out / "raw-generated/export.json").read_text()
                ),
                "rust_tests": True,
                "lean_build": True,
                "lean_package": "lean",
                "lean_package_sha256": {
                    name: digest(project / name) for name in LEAN_FILES + GENERATED
                },
                "axiom_audit": True,
                "lean_tests": True,
                "negative_controls": controls,
                "production_replay": production,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    LOGGER.info(
        "Verified extraction, proofs, tests and axiom audit; receipt: %s", receipt
    )


MUTATIONS.update(
    {
        "barrier_batch_admits_oversubscribed_pair": (
            "src/barrier.rs",
            "BarrierBatchWitness",
            "batch.arrivals > batch.pending / 2",
            "batch.arrivals > batch.pending",
        ),
    }
)


if __name__ == "__main__":
    main()
