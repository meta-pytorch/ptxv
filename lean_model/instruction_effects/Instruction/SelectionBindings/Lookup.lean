/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBindings.Reads

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegBinding

open SegProgram

@[step] theorem source_lookup_loop (bindings : Slice Binding) (register : PtxvSim.ptxv_ir.Reg)
    (start : Usize) (bounded : start.val ≤ bindings.val.length) :
    PtxvSim.seg.binding.register_binding_loop bindings register start ⦃ source =>
      source = lookup (bindings.val.drop start.val) register ⦄ := by
  classical
  unfold PtxvSim.seg.binding.register_binding_loop
  apply loop.spec_decr_nat
    (fun index : Usize => bindings.val.length - index.val)
    (fun index : Usize => index.val ≤ bindings.val.length ∧
      lookup (bindings.val.drop index.val) register = lookup (bindings.val.drop start.val) register)
  · intro index invariant
    rcases invariant with ⟨hbound, hlookup⟩
    unfold PtxvSim.seg.binding.register_binding_loop.body
    by_cases within : index.val < bindings.val.length
    · have htest : index < Slice.len bindings := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨entry, hentry⟩
      rcases entry with ⟨candidate, source⟩
      have hhead := lookup_drop bindings.val register index.val within
      rw [← hentry] at hhead
      step with source_file_equality candidate.file register.file as ⟨equal, hequal⟩
      by_cases sameFile : candidate.file = register.file
      · have hb : equal = true := hequal.mpr sameFile
        simp only [hb, ite_true]
        by_cases sameIndex : candidate.idx = register.idx
        · simp only [sameIndex, ite_true, WP.spec_ok]
          have sameKey := (register_key_exact candidate register).mpr ⟨sameFile, sameIndex⟩
          simp only [sameKey, ite_true] at hhead
          exact hhead.symm.trans hlookup
        · simp only [sameIndex, ite_false]
          have differentKey : registerKey candidate ≠ registerKey register :=
            fun h => sameIndex ((register_key_exact candidate register).mp h).2
          simp only [differentKey, ite_false] at hhead
          step as ⟨next, hnext⟩; try scalar_tac
      · have hb : equal = false := by cases equal <;> simp_all
        simp only [hb, Bool.false_eq_true, ite_false]
        have differentKey : registerKey candidate ≠ registerKey register :=
          fun h => sameFile ((register_key_exact candidate register).mp h).1
        simp only [differentKey, ite_false] at hhead
        step as ⟨next, hnext⟩; try scalar_tac
    · have htest : ¬ index < Slice.len bindings := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have hend : bindings.val.drop index.val = [] := List.drop_eq_nil_iff.mpr (by omega)
      simpa [hend, lookup] using hlookup
  · exact ⟨bounded, rfl⟩

@[step] theorem source_register_binding (bindings : Slice Binding) (register : PtxvSim.ptxv_ir.Reg) :
    PtxvSim.seg.binding.register_binding bindings register ⦃ source =>
      source = lookup bindings.val register ⦄ := by
  unfold PtxvSim.seg.binding.register_binding
  step with source_lookup_loop bindings register 0#usize (by simp) as ⟨source, hsource⟩
  simpa using hsource

@[step] theorem source_predicate_binding (predicate : PtxvSim.ptxv_ir.Pred) (bindings : Slice Binding) :
    PtxvSim.seg.binding.predicate_binding predicate bindings ⦃ source =>
      source = predicateSource predicate bindings.val ⦄ := by
  unfold PtxvSim.seg.binding.predicate_binding
  cases hreg : predicate.reg with
  | none => simp [predicateSource, hreg]
  | some register =>
    step with source_register_binding bindings register as ⟨source, hsource⟩
    simp [predicateSource, hreg, hsource]

end Instruction.SegBinding
