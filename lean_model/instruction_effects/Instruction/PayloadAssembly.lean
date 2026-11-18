/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadByteRefinement
import Instruction.PayloadInitialization
import Instruction.PayloadWindows
import Instruction.StoreDeferredRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def byteObservation {Error : Type} : core.result.Result (alloc.vec.Vec U8) Error →
    core.result.Result (List Nat) Error
  | .Ok bytes => .Ok (bytes.val.map UScalar.val)
  | .Err error => .Err error

def payloadWindowsValid (windows : List (Reg × U32 × U32)) (total : Nat) : Prop :=
  ∀ index (hi : index < windows.length),
    (windows[index]'hi).2.2.val ≤ 8 ∧
    (∀ byte < (windows[index]'hi).2.2.val,
      (windows[index]'hi).2.1.val + byte < total) ∧
    beforeWindow ((windows.take index).map byteWindow) (byteWindow (windows[index]'hi))

@[step] theorem payload_assemble_loop_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (windows : Slice (Reg × U32 × U32))
    (reader : Reader) (values : Key → U64) (total : Nat)
    (hsize : total ≤ 255) (hvalid : payloadWindowsValid windows.val total)
    (hread : ∀ register, key register ∈ windows.val.map windowSource →
      readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄)
    (output : alloc.vec.Vec U8) (index : Usize) (hindex : index.val ≤ windows.val.length)
    (houtput : output.val.map UScalar.val =
      payloadBytes total ((windows.val.take index.val).map byteWindow) (fun key => (values key).val)) :
    payload.assemble_loop1 readerInst windows reader output index ⦃ result =>
      byteObservation result = .Ok (payloadBytes total (windows.val.map byteWindow)
        (fun key => (values key).val)) ⦄ := by
  unfold payload.assemble_loop1
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U8 × Usize => windows.val.length - state.2.val)
    (fun state : alloc.vec.Vec U8 × Usize => state.2.val ≤ windows.val.length ∧
      state.1.val.map UScalar.val = payloadBytes total
        ((windows.val.take state.2.val).map byteWindow) (fun key => (values key).val))
  · rintro ⟨current, index⟩ ⟨hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold payload.assemble_loop1.body
    by_cases hlt : index.val < windows.val.length
    · have htest : index < Slice.len windows := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨window, hwindow⟩
      rcases window with ⟨register, first, count⟩
      have hentry : (register, first, count) = windows.val[index.val]'hlt := hwindow
      have hreg : key register ∈ windows.val.map windowSource := by
        apply List.mem_map.mpr
        refine ⟨windows.val[index.val]'hlt, List.getElem_mem hlt, ?_⟩
        rw [← hentry]
        rfl
      step with hread register hreg as ⟨result, hresult⟩
      rw [hresult]
      have hproperties := hvalid index.val hlt
      rw [← hentry] at hproperties
      obtain ⟨hcount, hwithin, hbefore⟩ := hproperties
      have hlength : current.val.length = total := by
        have := congrArg List.length hcurrent
        simpa [payloadBytes] using this
      step with payload_write_window_spec current first count (values (key register)) 0#u32 current
        hcount (by simp) (by omega) (by simpa [hlength] using hwithin)
        (by simp) as ⟨written, hwritten⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_⟩
      · scalar_tac
      · rw [hwritten, hcurrent]
        have hn : next.val = index.val + 1 := by scalar_tac
        rw [hn, List.take_succ_eq_append_getElem hlt, List.map_append]
        simp only [List.map_cons, List.map_nil]
        rw [← hentry]
        rw [payload_bytes_append total _ _ _ hbefore]
        rfl
      · scalar_tac
    · have htest : ¬index < Slice.len windows := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok, byteObservation]
      have heq : index.val = windows.val.length := by omega
      simpa [heq] using congrArg (core.result.Result.Ok (E := Error)) hcurrent
  · exact ⟨hindex, houtput⟩

end Instruction
