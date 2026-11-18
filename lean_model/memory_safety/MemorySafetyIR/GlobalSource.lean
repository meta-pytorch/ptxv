/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.GlobalValueSource
import MemorySafetyIR.GlobalSpecialization
import MemorySafetyIR.CodeProjection

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem source_global_address_matches_context (index : U32) (bindings : Slice GlobalBinding) :
    strict_memory_globals.global_address index bindings = .ok (globalAddress bindings.val index) := by
  rw [source_global_address_meaning]
  rfl

theorem source_global_operand_matches_context (operand : Option ptxv_ir.Val)
    (bindings : Slice GlobalBinding) :
    strict_memory_globals.address_value operand bindings = .ok (globalOperand bindings.val operand) := by
  rw [source_global_operand]
  rfl

theorem source_global_resolve_matches_context (operand : ptxv_ir.Addr)
    (bindings : Slice GlobalBinding) :
    strict_memory_globals.resolve_address operand bindings =
      .ok (globalResolvedAddress bindings.val operand) := by
  rw [source_resolve_global_address]
  rfl

theorem source_specialized_instruction_exact
    (instruction specialized : ptxv_ir.Inst) (bindings : Slice GlobalBinding) (pc : Usize)
    (accepted : strict_memory_globals.specialize_instruction instruction bindings pc =
      .ok (.Ok specialized)) : SpecializesInstruction bindings.val instruction specialized := by
  rcases instruction with ⟨guard, operation, uniform, loc, text⟩
  cases operation with
  | Ld destination count space bytes operand sync sext nc =>
    simp only [strict_memory_globals.specialize_instruction] at accepted
    split at accepted
    · cases space <;> simp at accepted
      rw [source_global_resolve_matches_context] at accepted
      simp only [bind_ok] at accepted
      cases resolved : globalResolvedAddress bindings.val operand with
      | none => simp [resolved] at accepted
      | some bits =>
        simp [resolved] at accepted
        cases accepted
        exact ⟨rfl, .load destination bytes operand sync sext nc bits resolved⟩
    · simp at accepted
  | St space bytes operand value sync =>
    cases space <;> cases value <;>
      simp [strict_memory_globals.specialize_instruction] at accepted
    rw [source_global_resolve_matches_context] at accepted
    simp only [bind_ok] at accepted
    cases resolved : globalResolvedAddress bindings.val operand with
    | none => simp [resolved] at accepted
    | some bits =>
      simp [resolved] at accepted
      cases accepted
      exact ⟨rfl, .store bytes operand _ sync bits resolved⟩
  | Nop =>
    simp [strict_memory_globals.specialize_instruction] at accepted
    cases accepted
    exact ⟨rfl, .nop⟩
  | Exit =>
    simp [strict_memory_globals.specialize_instruction] at accepted
    cases accepted
    exact ⟨rfl, .exit⟩
  | _ => simp [strict_memory_globals.specialize_instruction] at accepted

private theorem bind_success {α β : Type} (computation : Result α) (next : α → Result β)
    (value : β) (accepted : Aeneas.Std.bind computation next = .ok value) :
    ∃ input, computation = .ok input ∧ next input = .ok value := by
  cases computation with
  | ret input => exact ⟨input, rfl, by simpa only [bind_ok] using accepted⟩
  | vis effect continuation => simp only [bind_vis, vis_not_ok] at accepted
  | div => simp only [bind_div, div_not_ok] at accepted

private theorem push_success {α : Type} (values next : alloc.vec.Vec α) (value : α)
    (computed : alloc.vec.Vec.push values value = .ok next) :
    next.val = values.val ++ [value] := by
  unfold alloc.vec.Vec.push at computed
  dsimp only at computed
  split at computed
  · simp only [ok.injEq] at computed
    rw [← computed]
    simp [List.concat_eq_append]
  · simp at computed

private theorem usize_increment (index : Usize)
    (bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits) :
    (index + 1#usize : Result Usize) = .ok (Usize.ofNatCore (index.val + 1) bound) := by
  have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
  simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
    UScalar.check_bounds, platformBound, Result.ofOption]
  rfl

private theorem specialization_unroll (code : Slice ptxv_ir.Inst) (bindings : Slice GlobalBinding)
    (accumulator : alloc.vec.Vec ptxv_ir.Inst)
    (failure : Option strict_memory_admission.Refusal) (pc : Usize) :
    strict_memory_globals.specialize_code_loop code bindings accumulator failure pc = (do
      let outcome ← strict_memory_globals.specialize_code_loop.body code bindings accumulator failure pc
      match outcome with
      | .cont (next, reason, index) =>
        strict_memory_globals.specialize_code_loop code bindings next reason index
      | .done output => .ok output) := by
  unfold strict_memory_globals.specialize_code_loop
  rw [loop.eq_def]
  congr 1
  funext outcome
  cases outcome with
  | cont triple => rcases triple with ⟨_, _, _⟩; rfl
  | done output => rfl

private theorem specialization_failed (code : Slice ptxv_ir.Inst) (bindings : Slice GlobalBinding)
    (accumulator : alloc.vec.Vec ptxv_ir.Inst)
    (reason : strict_memory_admission.Refusal) (pc : Usize) :
    strict_memory_globals.specialize_code_loop code bindings accumulator (some reason) pc =
      .ok (accumulator, some reason) := by
  rw [specialization_unroll]
  by_cases inside : pc < Slice.len code <;>
    simp [strict_memory_globals.specialize_code_loop.body, inside, core.option.Option.is_none]

theorem source_specialize_code_loop (code : Slice ptxv_ir.Inst) (bindings : Slice GlobalBinding)
    (pc : Usize) (accumulator result : alloc.vec.Vec ptxv_ir.Inst)
    (computed : strict_memory_globals.specialize_code_loop code bindings
      accumulator none pc = .ok (result, none)) :
    ∃ tail, result.val = accumulator.val ++ tail ∧
      List.Forall₂ (SpecializesInstruction bindings.val) (code.val.drop pc.val) tail := by
  generalize fuelEq : code.val.length - pc.val = fuel
  induction fuel using Nat.strong_induction_on generalizing pc accumulator with
  | h fuel ih =>
    rw [specialization_unroll] at computed
    by_cases inside : pc.val < code.val.length
    · have test : pc < Slice.len code := by scalar_tac
      have bound : pc.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := code.property
        scalar_tac
      let nextPc : Usize := Usize.ofNatCore (pc.val + 1) bound
      have nextVal : nextPc.val = pc.val + 1 := by simp [nextPc]
      have increment : (pc + 1#usize : Result Usize) = .ok nextPc := usize_increment pc bound
      have read : Slice.index_usize code pc = .ok code.val[pc.val] := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside]
        rfl
      simp only [strict_memory_globals.specialize_code_loop.body, test, ite_true,
        core.option.Option.is_none, Option.isNone, read, increment, bind_ok] at computed
      try simp only [Std.bind_assoc] at computed
      obtain ⟨outcome, admitted, computed⟩ := bind_success _ _ _ computed
      cases outcome with
      | Err reason => simp [specialization_failed, Aeneas.Std.uncurry] at computed
      | Ok target =>
        simp only [Std.bind_assoc, bind_ok] at computed
        obtain ⟨pushed, pushedEq, computed⟩ := bind_success _ _ _ computed
        simp only [Aeneas.Std.uncurry, bind_ok] at computed
        obtain ⟨tail, values, related⟩ := ih (code.val.length - nextPc.val) (by omega)
          nextPc pushed computed rfl
        have pushedVal := push_success accumulator pushed target pushedEq
        refine ⟨target :: tail, ?_, ?_⟩
        · simpa only [pushedVal, List.append_assoc, List.singleton_append] using values
        · rw [List.drop_eq_getElem_cons inside]
          exact .cons (source_specialized_instruction_exact _ target bindings pc admitted)
            (by simpa only [nextVal] using related)
    · have test : ¬pc < Slice.len code := by scalar_tac
      simp only [strict_memory_globals.specialize_code_loop.body, test, ite_false,
        bind_ok, ok.injEq, Prod.mk.injEq] at computed
      rcases computed with ⟨rfl, _⟩
      refine ⟨[], by simp, ?_⟩
      rw [List.drop_eq_nil_of_le (show code.val.length ≤ pc.val by omega)]
      exact .nil

theorem source_specialize_code_exact
    (code : Slice ptxv_ir.Inst) (bindings : Slice GlobalBinding)
    (specialized : alloc.vec.Vec ptxv_ir.Inst)
    (accepted : strict_memory_globals.specialize_code code bindings = .ok (.Ok specialized)) :
    List.Forall₂ (SpecializesInstruction bindings.val) code.val specialized.val := by
  unfold strict_memory_globals.specialize_code at accepted
  obtain ⟨outcome, loopEq, accepted⟩ := bind_success _ _ _ accepted
  rcases outcome with ⟨result, reason⟩
  cases reason with
  | some reason => simp [Aeneas.Std.uncurry] at accepted
  | none =>
    simp only [Aeneas.Std.uncurry, ok.injEq, core.result.Result.Ok.injEq] at accepted
    subst result
    obtain ⟨tail, values, related⟩ := source_specialize_code_loop code bindings 0#usize
      (alloc.vec.Vec.new ptxv_ir.Inst) specialized loopEq
    have same : specialized.val = tail := by simpa [alloc.vec.Vec.new] using values
    simpa [same] using related

theorem source_global_code_interpretation_exact
    (code : Slice ptxv_ir.Inst) (bindings : Slice GlobalBinding) (nregs : U16)
    (specialized : alloc.vec.Vec ptxv_ir.Inst)
    (projected : alloc.vec.Vec CheckedInstruction)
    (specialization : strict_memory_globals.specialize_code code bindings =
      .ok (.Ok specialized))
    (projection : strict_memory_admission.project_code specialized.slice nregs =
      .ok (.Ok projected)) :
    List.Forall₂ (fun instruction result => ∀ valuation,
      interpretInstruction (bindGlobals bindings.val valuation) instruction = result)
      code.val (projected.val.map checkedInstructionMemoryModel) := by
  exact specialized_code_interpretation_exact bindings.val code.val specialized.val projected.val
    (source_specialize_code_exact code bindings specialized specialization)
    (source_project_code_exact specialized.slice nregs projected projection)

end MemorySafetyIR
