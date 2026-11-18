/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchModel
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source

set_option autoImplicit false
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def outputValid (launch : machine.Launch) (output : Usize) : Prop :=
  ∃ allocation, launch.allocs.val[output.val]? = some allocation ∧ allocation.undef = false

theorem source_outputs_loop (launch : machine.Launch)
    (failure : Option strict_memory_admission.Refusal) (index : Usize)
    (computed : strict_memory_launch.validate_outputs_loop launch failure index = .ok none) :
    failure = none ∧ ∀ output ∈ launch.outputs.val.drop index.val, outputValid launch output := by
  generalize fuelEq : launch.outputs.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing failure index with
  | h fuel ih =>
    have unroll : strict_memory_launch.validate_outputs_loop launch failure index = (do
        let result ← strict_memory_launch.validate_outputs_loop.body launch failure index
        match result with
        | .cont pair => strict_memory_launch.validate_outputs_loop launch pair.1 pair.2
        | .done result => .ok result) := by
      unfold strict_memory_launch.validate_outputs_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done result => rfl
    rw [unroll] at computed
    by_cases inside : index.val < launch.outputs.val.length
    · have test : index < alloc.vec.Vec.len launch.outputs := by scalar_tac
      cases failure with
      | some reason =>
        simp [strict_memory_launch.validate_outputs_loop.body, test] at computed
      | none =>
        have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
          have := alloc.vec.Vec.len_ineq launch.outputs
          scalar_tac
        let after : Usize := Usize.ofNatCore (index.val + 1) bound
        have afterVal : after.val = index.val + 1 := by simp [after]
        have increment := source_usize_increment index bound
        let output := launch.outputs.val[index.val]
        have outputRead : alloc.vec.Vec.index_usize launch.outputs index = .ok output := by
          simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside, output]
        simp only [strict_memory_launch.validate_outputs_loop.body, test, ite_true,
          core.option.Option.is_none, Option.isNone, alloc.vec.Vec.index_slice_index, outputRead, bind_ok] at computed
        by_cases outside : output >= alloc.vec.Vec.len launch.allocs
        · simp only [outside, ite_true, bind_ok, increment] at computed
          have tail := ih (launch.outputs.val.length - after.val) (by omega)
            (some strict_memory_admission.Refusal.Launch) after computed rfl
          cases tail.1
        · have allocationInside : output.val < launch.allocs.val.length := by scalar_tac
          let allocation := launch.allocs.val[output.val]
          have allocationRead : alloc.vec.Vec.index_usize launch.allocs output = .ok allocation := by
            simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem allocationInside, allocation]
          simp only [outside, ite_false, allocationRead, bind_ok] at computed
          cases defined : allocation.undef with
          | true =>
            simp only [defined, ite_true, bind_ok, increment] at computed
            have tail := ih (launch.outputs.val.length - after.val) (by omega)
              (some (.Allocation output)) after computed rfl
            cases tail.1
          | false =>
            simp only [defined, Bool.false_eq_true, ite_false, bind_ok, increment] at computed
            have tail := ih (launch.outputs.val.length - after.val) (by omega) none after computed rfl
            refine ⟨rfl, ?_⟩
            rw [List.drop_eq_getElem_cons inside]
            intro selected member
            simp only [List.mem_cons] at member
            rcases member with equal | member
            · subst selected
              exact ⟨allocation, List.getElem?_eq_getElem allocationInside, defined⟩
            · apply tail.2 selected
              simpa only [afterVal] using member
    · have test : ¬index < alloc.vec.Vec.len launch.outputs := by scalar_tac
      simp only [strict_memory_launch.validate_outputs_loop.body, test, ite_false, bind_ok,
        ok.injEq] at computed
      refine ⟨computed, ?_⟩
      simp [List.drop_eq_nil_of_le (show launch.outputs.val.length ≤ index.val by omega)]

theorem source_validate_outputs (launch : machine.Launch)
    (computed : strict_memory_launch.validate_outputs launch = .ok (.Ok ())) : outputsDefined launch := by
  unfold strict_memory_launch.validate_outputs at computed
  obtain ⟨failure, ran, finished⟩ := source_bind_success _ _ _ computed
  cases failure with
  | some reason => simp at finished
  | none =>
    have valid := (source_outputs_loop launch none 0#usize ran).2
    simpa [outputsDefined, outputValid] using valid

end MemorySafetyIR
