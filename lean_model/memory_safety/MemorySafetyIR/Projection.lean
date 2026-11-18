/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.Footprint

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192

namespace MemorySafetyIR

abbrev CheckedInstruction := ptxv_memory_write.memory_safety.Instruction

theorem source_projected_instruction_present
    (instruction : ptxv_ir.Inst) (nregs : U16) (pc : Usize) (last : Bool)
    (projected : Option CheckedInstruction)
    (accepted : strict_memory_admission.project_instruction instruction nregs pc last =
      .ok (.Ok projected)) : ∃ result, projected = some result := by
  cases projected with
  | some result => exact ⟨result, rfl⟩
  | none =>
    obtain ⟨guardReg, guardNeg, uniform⟩ :=
      source_admission_requires_true_nonuniform_guard instruction nregs pc last _ accepted
    have classified := source_admission_refuses_every_other_opcode instruction nregs pc last _ accepted
    rcases instruction with ⟨⟨greg, gneg⟩, operation, uni, loc, text⟩
    dsimp only at guardReg guardNeg uniform classified
    subst greg
    subst gneg
    subst uni
    cases classified with
    | load destination count space bytes operand sync sext nc =>
      cases space <;> cases sync <;> cases nc <;>
        simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
          source_scalar_width, source_nonvirtual_file,
          source_constant_address, lift] at accepted
      all_goals
        split_ifs at accepted <;> try simp at accepted
        cases resolved : constantAddress operand <;> simp [resolved] at accepted
    | store space bytes operand source sync =>
      cases source <;> cases space <;> cases sync <;>
        simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
          source_scalar_width, source_constant_address, lift] at accepted
      all_goals
        split_ifs at accepted <;> try simp at accepted
        cases resolved : constantAddress operand <;> simp [resolved] at accepted
    | nop => simp [strict_memory_admission.project_instruction, ptxv_ir.Pred.is_always,
         ] at accepted
    | exit => cases last <;> simp [strict_memory_admission.project_instruction,
        ptxv_ir.Pred.is_always,  ] at accepted

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

def Projected (nregs : U16) (source : ptxv_ir.Inst) (target : CheckedInstruction) : Prop :=
  ∃ pc last, strict_memory_admission.project_instruction source nregs pc last = .ok (.Ok (some target))

private theorem projection_unroll (code : Slice ptxv_ir.Inst) (nregs : U16)
    (length : Usize) (accumulator : alloc.vec.Vec CheckedInstruction)
    (failure : Option strict_memory_admission.Refusal) (pc : Usize) :
    strict_memory_admission.project_code_loop code nregs length accumulator failure pc = (do
      let outcome ← strict_memory_admission.project_code_loop.body code nregs length accumulator failure pc
      match outcome with
      | .cont (next, reason, index) =>
        strict_memory_admission.project_code_loop code nregs length next reason index
      | .done output => .ok output) := by
  unfold strict_memory_admission.project_code_loop
  rw [loop.eq_def]
  congr 1
  funext outcome
  cases outcome with
  | cont triple => rcases triple with ⟨_, _, _⟩; rfl
  | done output => rfl

private theorem projection_failed (code : Slice ptxv_ir.Inst) (nregs : U16)
    (length : Usize) (accumulator : alloc.vec.Vec CheckedInstruction)
    (reason : strict_memory_admission.Refusal) (pc : Usize) :
    strict_memory_admission.project_code_loop code nregs length accumulator (some reason) pc =
      .ok (accumulator, some reason) := by
  rw [projection_unroll]
  by_cases inside : pc < length <;>
    simp [strict_memory_admission.project_code_loop.body, inside, core.option.Option.is_none]

theorem source_project_code_loop (code : Slice ptxv_ir.Inst) (nregs : U16)
    (pc : Usize) (accumulator result : alloc.vec.Vec CheckedInstruction)
    (computed : strict_memory_admission.project_code_loop code nregs (Slice.len code)
      accumulator none pc = .ok (result, none)) :
    ∃ tail, result.val = accumulator.val ++ tail ∧
      List.Forall₂ (Projected nregs) (code.val.drop pc.val) tail := by
  generalize fuelEq : code.val.length - pc.val = fuel
  induction fuel using Nat.strong_induction_on generalizing pc accumulator with
  | h fuel ih =>
    rw [projection_unroll] at computed
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
      simp only [strict_memory_admission.project_code_loop.body, test, ite_true,
        core.option.Option.is_none, Option.isNone, read, increment, bind_ok] at computed
      try simp only [Std.bind_assoc] at computed
      obtain ⟨outcome, admitted, computed⟩ := bind_success _ _ _ computed
      cases outcome with
      | Err reason => simp [projection_failed] at computed
      | Ok projected =>
        obtain ⟨target, rfl⟩ := source_projected_instruction_present _ _ _ _ projected admitted
        dsimp only at computed
        simp only [Std.bind_assoc, bind_ok] at computed
        obtain ⟨pushed, pushedEq, computed⟩ := bind_success _ _ _ computed
        simp only [Aeneas.Std.uncurry, bind_ok] at computed
        obtain ⟨tail, values, related⟩ := ih (code.val.length - nextPc.val) (by omega)
          nextPc pushed computed rfl
        have pushedVal := push_success accumulator pushed target pushedEq
        refine ⟨target :: tail, ?_, ?_⟩
        · simpa only [pushedVal, List.append_assoc, List.singleton_append] using values
        · rw [List.drop_eq_getElem_cons inside]
          exact .cons ⟨pc, _, admitted⟩ (by simpa only [nextVal] using related)
    · have test : ¬pc < Slice.len code := by scalar_tac
      simp only [strict_memory_admission.project_code_loop.body, test, ite_false,
        bind_ok, ok.injEq, Prod.mk.injEq] at computed
      rcases computed with ⟨rfl, _⟩
      refine ⟨[], by simp, ?_⟩
      rw [List.drop_eq_nil_of_le (show code.val.length ≤ pc.val by omega)]
      exact .nil

theorem source_project_code_preserves_every_instruction
    (code : Slice ptxv_ir.Inst) (nregs : U16)
    (projected : alloc.vec.Vec CheckedInstruction)
    (accepted : strict_memory_admission.project_code code nregs = .ok (.Ok projected)) :
    List.Forall₂ (Projected nregs) code.val projected.val := by
  unfold strict_memory_admission.project_code at accepted
  dsimp only at accepted
  split at accepted
  · simp at accepted
  · obtain ⟨beforeLast, subtraction, accepted⟩ := bind_success _ _ _ accepted
    obtain ⟨lastInstruction, read, accepted⟩ := bind_success _ _ _ accepted
    obtain ⟨isExit, lastCheck, accepted⟩ := bind_success _ _ _ accepted
    cases isExit with
    | false => simp at accepted
    | true =>
      obtain ⟨outcome, loopEq, accepted⟩ := bind_success _ _ _ accepted
      rcases outcome with ⟨result, reason⟩
      cases reason with
      | some reason => simp at accepted
      | none =>
        simp only [Aeneas.Std.uncurry, ok.injEq, core.result.Result.Ok.injEq] at accepted
        subst result
        obtain ⟨tail, values, related⟩ := source_project_code_loop code nregs 0#usize
          (alloc.vec.Vec.new CheckedInstruction) projected loopEq
        have same : projected.val = tail := by simpa [alloc.vec.Vec.new] using values
        simpa [same] using related

theorem source_project_code_preserves_source_pcs
    (code : Slice ptxv_ir.Inst) (nregs : U16)
    (projected : alloc.vec.Vec CheckedInstruction)
    (accepted : strict_memory_admission.project_code code nregs = .ok (.Ok projected)) :
    code.val.length = projected.val.length :=
  (source_project_code_preserves_every_instruction code nregs projected accepted).length_eq

end MemorySafetyIR
