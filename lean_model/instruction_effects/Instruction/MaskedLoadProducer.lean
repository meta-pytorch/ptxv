/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MaskedLoadLoop

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem masked_load_half_spec {Writer State : Type}
    (writerInst : masked_load.RegisterWriter Writer) (first : Reg) (mask start : U8)
    (data : Array U8 32#usize) (writer : Writer) (observe : Writer → State)
    (valid : Writer → Prop) (transition : State → LoadObservation → State)
    (hvalid : valid writer) (hstart : start.val + 4 ≤ 8)
    (hbounded : Bounded (consecutive first (halfWords mask start.val).length))
    (hwrite : LoadWriterContract writerInst observe valid transition
      (selectedLoadHalf first mask start.val (loadedMemoryWord data))) :
    masked_load.half writerInst first mask start data writer ⦃ updated =>
      valid updated ∧ observe updated =
        (selectedLoadHalf first mask start.val (loadedMemoryWord data)).foldl transition (observe writer) ⦄ := by
  unfold masked_load.half
  step with masked_load_half_loop_spec writerInst first mask start data writer 0#u16 0#u8
    (observe writer) observe valid transition hvalid hstart (by decide)
    (by simp) hbounded hwrite (by simp) as ⟨updated, hupdatedValid, hupdated⟩
  exact ⟨hupdatedValid, hupdated⟩

@[step] theorem actual_masked_load_execute_spec {Writer State : Type}
    (writerInst : masked_load.RegisterWriter Writer) (operation : Op)
    (data : Array U8 32#usize) (writer : Writer) (observe : Writer → State)
    (valid : Writer → Prop) (transition : State → LoadObservation → State)
    (hvalid : valid writer)
    (hbounded : Bounded ((maskedLoadObservations operation (loadedMemoryWord data)).map
      LoadObservation.destination))
    (hwrite : LoadWriterContract writerInst observe valid transition
      (maskedLoadObservations operation (loadedMemoryWord data))) :
    masked_load.execute writerInst operation data writer ⦃ updated =>
      valid updated ∧ observe updated =
        (maskedLoadObservations operation (loadedMemoryWord data)).foldl transition (observe writer) ⦄ := by
  unfold masked_load.execute
  cases operation <;> try (simpa [maskedLoadObservations] using hvalid)
  rename_i high low address sync mask nc
  have hparts : Bounded (consecutive low (halfWords mask 0).length) ∧
      Bounded (consecutive high (halfWords mask 4).length) := by
    simpa only [masked_load_destinations, maskedWrites, bounded_append] using hbounded
  have hlow : LoadWriterContract writerInst observe valid transition
      (selectedLoadHalf low mask 0 (loadedMemoryWord data)) := by
    intro current hc destination value sourceByte hm
    apply hwrite current hc destination value sourceByte
    exact List.mem_append.mpr (Or.inl hm)
  have hhigh : LoadWriterContract writerInst observe valid transition
      (selectedLoadHalf high mask 4 (loadedMemoryWord data)) := by
    intro current hc destination value sourceByte hm
    apply hwrite current hc destination value sourceByte
    exact List.mem_append.mpr (Or.inr hm)
  step with masked_load_half_spec writerInst low mask 0#u8 data writer observe valid transition
    hvalid (by decide) hparts.1 hlow as ⟨middle, hmiddleValid, hmiddle⟩
  step with masked_load_half_spec writerInst high mask 4#u8 data middle observe valid transition
    hmiddleValid (by decide) hparts.2 hhigh as ⟨updated, hupdatedValid, hupdated⟩
  rw [hmiddle] at hupdated
  exact ⟨hupdatedValid, by simpa only [maskedLoadObservations, List.foldl_append] using hupdated⟩

end Instruction
