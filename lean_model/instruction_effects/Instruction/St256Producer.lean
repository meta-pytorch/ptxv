/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.St256Refinement
import Instruction.PayloadProducer

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem actual_st256_payload_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (operation : Op)
    (reader : Reader) (values : Key → U64)
    (hbounded : ∀ windows, st256Windows operation = some windows → windowsBounded windows)
    (hread : ∀ register, key register ∈ (memoryReads operation).getD [] →
      readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄) :
    payload.from_st256 readerInst operation reader ⦃ result =>
      result.map byteObservation = (st256Windows operation).map fun windows =>
        .Ok (payloadBytes 32 windows (fun register => (values register).val)) ⦄ := by
  unfold payload.from_st256
  step with st256_sources_spec operation hbounded as ⟨windows, hwindows⟩
  cases windows with
  | none => simp [← hwindows]
  | some windows =>
    have hvalid := actual_st256_windows_valid operation windows.val hwindows
    have hmodel : windows.val.map byteWindow = (st256Windows operation).getD [] := by
      rw [← hwindows]
      rfl
    have hreader : ∀ register, key register ∈ windows.val.map windowSource →
        readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄ := by
      intro register hmember
      apply hread register
      rcases List.mem_map.mp hmember with ⟨window, hm, hk⟩
      have hw : byteWindow window ∈ (st256Windows operation).getD [] := by
        rw [← hmodel]
        exact List.mem_map.mpr ⟨window, hm, rfl⟩
      have hr := actual_st256_window_declared_read operation (byteWindow window) hw
      change windowSource window ∈ (memoryReads operation).getD [] at hr
      rw [hk] at hr
      exact hr
    step with payload_assemble_spec readerInst (alloc.vec.Vec.deref windows) 32#u8 reader values
      (by simpa [alloc.vec.Vec.deref, Slice.from_val] using hvalid)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hreader) as ⟨result, hresult⟩
    simp only [alloc.vec.Vec.deref, Slice.from_val] at hresult
    simp [hresult, ← hwindows]

end Instruction
