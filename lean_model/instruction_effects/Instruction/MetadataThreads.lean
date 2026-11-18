/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MetadataDeliveryModel
import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def threadNumbers (threads : List Usize) : List Nat := threads.map (·.val)

@[simp] theorem thread_numbers_member (threads : List Usize) (thread : Usize) :
    thread.val ∈ threadNumbers threads ↔ thread ∈ threads := by
  simp only [threadNumbers, List.mem_map]
  constructor
  · rintro ⟨other, hm, heq⟩
    exact UScalar.eq_of_val_eq heq ▸ hm
  · intro hm
    exact ⟨thread, hm, rfl⟩

@[step] theorem metadata_contains_loop_spec (threads : Slice Usize) (thread index : Usize)
    (hindex : index.val ≤ threads.val.length)
    (hprior : thread ∉ threads.val.take index.val) :
    metadata_delivery.contains_loop threads thread index ⦃ result => result = decide (thread ∈ threads.val) ⦄ := by
  unfold metadata_delivery.contains_loop
  apply loop.spec_decr_nat (fun index : Usize => threads.val.length - index.val)
    (fun index : Usize => index.val ≤ threads.val.length ∧ thread ∉ threads.val.take index.val)
  · intro current ⟨hbound, hnot⟩
    unfold metadata_delivery.contains_loop.body
    by_cases hlt : current.val < threads.val.length
    · have htest : current < threads.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨value, hvalue⟩
      by_cases heq : value = thread
      · simp only [heq, ite_true]
        have hm : thread ∈ threads.val := by
          have hg : threads.val[current.val] ∈ threads.val := List.getElem_mem hlt
          grind
        simp [hm]
      · simp only [heq, ite_false]
        step as ⟨next, hnext⟩; try scalar_tac
        have hn : next.val = current.val + 1 := by scalar_tac
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        rw [hn, List.take_succ_eq_append_getElem hlt]
        simp only [List.mem_append, List.mem_singleton, not_or]
        exact ⟨hnot, by simpa [hvalue] using Ne.symm heq⟩
    · have htest : ¬current < threads.len := by scalar_tac
      have hi : current.val = threads.val.length := by omega
      have hn : thread ∉ threads.val := by simpa [hi] using hnot
      simp [htest, hn]
  · exact ⟨hindex, hprior⟩

@[step] theorem metadata_contains_spec (threads : Slice Usize) (thread : Usize) :
    metadata_delivery.contains threads thread ⦃ result => result = decide (thread ∈ threads.val) ⦄ := by
  unfold metadata_delivery.contains
  apply metadata_contains_loop_spec
  · scalar_tac
  · simp

@[step] theorem metadata_touched_loop_spec (active guarded : Slice Usize)
    (touched : alloc.vec.Vec Usize) (index : Usize)
    (hindex : index.val ≤ guarded.val.length)
    (hsize : active.val.length + guarded.val.length ≤ Usize.max)
    (htouched : touched.val = active.val ++ (guarded.val.take index.val).filter
      (fun thread => thread ∉ active.val)) :
    metadata_delivery.touched_threads_loop active guarded touched index ⦃ result =>
      result.val = active.val ++ guarded.val.filter (fun thread => thread ∉ active.val) ⦄ := by
  unfold metadata_delivery.touched_threads_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec Usize × Usize => guarded.val.length - state.2.val)
    (fun state : alloc.vec.Vec Usize × Usize => state.2.val ≤ guarded.val.length ∧
      state.1.val = active.val ++ (guarded.val.take state.2.val).filter
        (fun thread => thread ∉ active.val))
  · rintro ⟨current, position⟩ ⟨hbound, hcurrent⟩
    dsimp only at hbound hcurrent
    unfold metadata_delivery.touched_threads_loop.body
    by_cases hlt : position.val < guarded.val.length
    · have htest : position < guarded.len := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨thread, hthread⟩
      have hthread : thread = guarded.val[position.val]'hlt := hthread
      step with metadata_contains_spec active thread as ⟨found, hfound⟩
      rw [hfound]
      by_cases hm : thread ∈ active.val
      · simp only [hm, decide_true, ite_true, bind_ok]
        step as ⟨next, hnext⟩; try scalar_tac
        have hn : next.val = position.val + 1 := by scalar_tac
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        rw [hn, List.take_succ_eq_append_getElem hlt, List.filter_append]
        have hf : ([guarded.val[position.val]'hlt].filter fun t => t ∉ active.val) = [] := by
          rw [← hthread]
          simp [hm]
        rw [hf, List.append_nil]
        exact hcurrent
      · simp only [hm, decide_false, Bool.false_eq_true, ite_false]
        have hlength : current.val.length < Usize.max := by
          rw [hcurrent, List.length_append]
          have hf := List.length_filter_le (fun thread => decide (thread ∉ active.val))
            (guarded.val.take position.val)
          have ht := List.length_take_le position.val guarded.val
          omega
        step with alloc.vec.Vec.push_spec current thread hlength as ⟨updated, hupdated⟩
        step as ⟨next, hnext⟩; try scalar_tac
        have hn : next.val = position.val + 1 := by scalar_tac
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        rw [hn, List.take_succ_eq_append_getElem hlt, List.filter_append]
        have hf : ([guarded.val[position.val]'hlt].filter fun t => t ∉ active.val) = [thread] := by
          rw [← hthread]
          simp [hm]
        rw [hf]
        simpa only [hcurrent, List.append_assoc] using hupdated
    · have htest : ¬position < guarded.len := by scalar_tac
      have hi : position.val = guarded.val.length := by omega
      simpa [htest, hi] using hcurrent
  · exact ⟨hindex, htouched⟩

@[step] theorem metadata_touched_spec (active guarded : Slice Usize)
    (hsize : active.val.length + guarded.val.length ≤ Usize.max) :
    metadata_delivery.touched_threads active guarded ⦃ result =>
      threadNumbers result.val = metadataTouched (threadNumbers active.val) (threadNumbers guarded.val) ⦄ := by
  unfold metadata_delivery.touched_threads
  step as ⟨touched, htouched⟩
  step with metadata_touched_loop_spec active guarded touched 0#usize (by scalar_tac) hsize
    (by change touched.slice.val = _; simpa using congrArg Slice.val htouched.symm) as ⟨result, hresult⟩
  have hmap : threadNumbers (guarded.val.filter fun thread => thread ∉ active.val) =
      (threadNumbers guarded.val).filter fun thread => thread ∉ threadNumbers active.val := by
    change List.map (fun t : Usize => t.val) _ = List.filter _ (List.map _ _)
    rw [List.filter_map]
    have hp : (fun t : Usize => decide (t ∉ active.val)) =
        (fun t : Nat => decide (t ∉ threadNumbers active.val)) ∘ (fun t : Usize => t.val) := by
      funext thread
      change decide (thread ∉ active.val) = decide (thread.val ∉ threadNumbers active.val)
      simp only [thread_numbers_member]
    rw [hp]
  change threadNumbers result.val = metadataTouched (threadNumbers active.val) (threadNumbers guarded.val)
  rw [hresult]
  unfold metadataTouched
  have happend : threadNumbers (active.val ++ guarded.val.filter (fun thread => thread ∉ active.val)) =
      threadNumbers active.val ++ threadNumbers (guarded.val.filter (fun thread => thread ∉ active.val)) := by
    simp only [threadNumbers, List.map_append]
  rw [happend, hmap]

end Instruction
