/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadAssembly
import Instruction.PayloadSupportRefinement
import Instruction.PayloadValidity
import Instruction.PayloadReadCoverage

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem payload_assemble_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (windows : Slice (Reg × U32 × U32))
    (total : U8) (reader : Reader) (values : Key → U64)
    (hvalid : payloadWindowsValid windows.val total.val)
    (hread : ∀ register, key register ∈ windows.val.map windowSource →
      readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄) :
    payload.assemble readerInst windows total reader ⦃ result =>
      byteObservation result = .Ok (payloadBytes total.val (windows.val.map byteWindow)
        (fun key => (values key).val)) ⦄ := by
  unfold payload.assemble
  step with payload_initialize_spec total (alloc.vec.Vec.new U8) 0#u8 (by simp)
    (by simp [alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val])
    as ⟨output, houtput⟩
  step with payload_assemble_loop_spec readerInst windows reader values total.val
    (by scalar_tac) hvalid hread output 0#usize (by simp)
    (by
      have hempty : payloadBytes total.val [] (fun key => (values key).val) =
          List.replicate total.val 0 := by
        unfold payloadBytes
        have hzero : payloadByte [] (fun key => (values key).val) = fun _ => 0 := by
          funext index
          simp [payloadByte]
        rw [hzero]
        simp
      change output.val.map UScalar.val = payloadBytes total.val [] (fun key => (values key).val)
      rw [hempty]
      exact houtput)
    as ⟨result, hresult⟩
  exact hresult

@[step] theorem actual_store_payload_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (front : Front) (operation : Op)
    (reader : Reader) (values : Key → U64)
    (hindex : windowsBounded (storeWindows front operation))
    (hread : ∀ register, key register ∈ (memoryReads operation).getD [] →
      readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄) :
    payload.from_operation readerInst front operation reader ⦃ result =>
      result.map byteObservation = (supportedPayloadTotal front operation).map fun total =>
        .Ok (payloadBytes total (storeWindows front operation) (fun key => (values key).val)) ⦄ := by
  unfold payload.from_operation
  step with payload_supported_total_spec front operation as ⟨total, htotal⟩
  cases total with
  | none => simp [← htotal]
  | some total =>
    simp only [Option.map_some] at htotal
    have hsupported : supportedPayloadTotal front operation = some total.val := htotal.symm
    step with store_windows_spec front operation hindex as ⟨windows, hwindows⟩
    have hvalid := operation_windows_valid front operation total.val hsupported windows.val hwindows
    have hreader : ∀ register, key register ∈ windows.val.map windowSource →
        readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄ := by
      intro register hmember
      apply hread register
      rcases List.mem_map.mp hmember with ⟨window, hm, hk⟩
      have hwindow : byteWindow window ∈ storeWindows front operation := by
        rw [← hwindows]
        exact List.mem_map.mpr ⟨window, hm, rfl⟩
      have hr := store_window_declared_read_coverage front operation (byteWindow window) hwindow
      change windowSource window ∈ (memoryReads operation).getD [] at hr
      rw [hk] at hr
      exact hr
    step with payload_assemble_spec readerInst (alloc.vec.Vec.deref windows) total reader values
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hvalid)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hreader) as ⟨result, hresult⟩
    simp only [alloc.vec.Vec.deref, Slice.from_val] at hresult
    simp [hresult, hsupported, hwindows]

end Instruction
