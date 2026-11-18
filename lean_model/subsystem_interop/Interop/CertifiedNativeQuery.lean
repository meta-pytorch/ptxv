/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop.CertificateDynamicPaths

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.GuardCertificate Instruction.DestinationRows
open MemoryWrite.MayWrite MemoryWrite.RegisterIndex MemoryWrite.BarrierReach
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.ControlFlow

theorem certified_native_queries_catch_dynamic_path_writes
    {C F : Type} (constants : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (bank : U32 → U8 → Option U64)
    (fold : alloc.vec.Vec (Option Bool)) (checked : CertifiedFolds constants certificate bank fold)
    (reader : destination_rows.LegacyDestinations F) (fallback : F)
    (input : KernelInputs certificate.code certificate.returns)
    (legacy : LegacyReturns reader fallback certificate.code.val)
    (keyStorage : ∀ rows : alloc.vec.Vec (alloc.vec.Vec Reg),
      destination_rows.collect reader certificate.code fallback = .ok rows → BuildFits (programRows rows).val) :
    ∃ before rows keys index graph,
      graphFromSource certificate.code 0#u8 (alloc.vec.Vec.deref fold) certificate.returns = .ok before ∧
      destination_rows.collect reader certificate.code fallback = .ok rows ∧
      keysFromSource rows = .ok keys ∧
      PtxvMemoryWrite.register_index.publish_graph before (alloc.vec.Vec.deref keys) = .ok (index, graph) ∧
      ∀ (mask : U32) (bit : Bit), HasBit mask bit → ∀ (start : Usize), start.val < certificate.code.val.length →
        AnalysisFits graph → ∃ sets,
          PtxvMemoryWrite.cache_graph.may_write_sets graph start mask = .ok sets ∧
          ∀ (initial : PartialRegisters), ExecutionDefined certificate bank start.val initial →
          ∀ (pc : Usize) (registers : PartialRegisters) (register : Reg),
            DynamicMemoryWrittenBefore certificate bank start.val initial pc.val registers register →
            DynamicToBarrier certificate bank bit pc.val registers →
            Std.bind (programKeyFromSource register) (fun identity =>
              Std.bind (PtxvMemoryWrite.register_index.RegisterIndex.get index identity)
                (fun dense => PtxvMemoryWrite.cache_graph.may_at sets pc dense)) = .ok true := by
  obtain ⟨before, rows, keys, index, graph, sourceOk, rowsOk, keysOk, published, queries⟩ :=
    native_kernel_publication_analysis_and_queries_return_true reader fallback certificate.code 0#u8
      (alloc.vec.Vec.deref fold) certificate.returns input legacy keyStorage
  refine ⟨before, rows, keys, index, graph, sourceOk, rowsOk, keysOk, published, ?_⟩
  intro mask bit requested start inside fits
  obtain ⟨sets, analysis, answers⟩ := queries mask bit requested start inside fits
  refine ⟨sets, analysis, ?_⟩
  intro initial entry pc registers register written gate
  apply answers pc register
  · simpa only [alloc.vec.Vec.deref, Slice.from_val] using
      written.raw_written constants certificate bank fold checked graph mask start.val pc.val initial registers register entry
  · simpa only [alloc.vec.Vec.deref, Slice.from_val] using
      dynamic_to_barrier_is_static constants certificate bank fold checked graph mask bit pc.val registers
        ((written.path certificate bank start.val pc.val initial registers register).execution certificate bank
          start.val pc.val initial registers entry) gate

end Instruction.ControlFlow
