/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchRestrictions
import MemorySafetyIR.SourceLaunchAliases
import MemorySafetyIR.SourceLaunchDimensions
import MemorySafetyIR.SourceLaunchOutputs
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 16000000
set_option maxRecDepth 16384
set_option Elab.async false
set_option debug.skipKernelTC false
set_option backward.isDefEq.respectTransparency false

namespace MemorySafetyIR

private theorem successful_guard {α : Type} (condition : Prop) [Decidable condition]
    (reason : strict_memory_admission.Refusal)
    (next : Result (core.result.Result α strict_memory_admission.Refusal)) (out : α)
    (accepted : (if condition then .ok (.Err reason) else next) = .ok (.Ok out)) :
    ¬condition ∧ next = .ok (.Ok out) := by
  by_cases h : condition <;> simp_all

private theorem vector_empty {α : Type} (values : alloc.vec.Vec α)
    (guard : ¬(alloc.vec.Vec.len values != 0#usize) = true) : values.val = [] := by
  simpa using guard

private theorem option_none {α : Type} (value : Option α)
    (guard : ¬core.option.Option.is_some value = true) : value = none := by
  cases value <;> simp_all [core.option.Option.is_some]

private theorem boolean_false (value : Bool) (guard : ¬value = true) : value = false := by
  simpa using guard

private theorem front_ptx (value : ptxv_ir.Front) (result : Bool)
    (read : core.cmp.PartialEq.ne.trait_default ptxv_ir.Front.Insts.CoreCmpPartialEqFront value .Ptx = .ok result)
    (guard : ¬result = true) : value = .Ptx := by
  cases value <;> simp_all [core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default,
    ptxv_ir.Front.Insts.CoreCmpPartialEqFront.eq, ptxv_ir.Front.read_discriminant]

private theorem successful_outputs (launch : machine.Launch) (produced expected : Usize)
    (accepted : (do
      let result ← strict_memory_launch.validate_outputs launch
      match result with
      | .Ok _ => ok (core.result.Result.Ok produced)
      | .Err reason => ok (core.result.Result.Err reason)) = ok (core.result.Result.Ok expected)) :
    produced = expected ∧ strict_memory_launch.validate_outputs launch = .ok (.Ok ()) := by
  obtain ⟨result, computed, accepted⟩ := source_bind_success _ _ _ accepted
  cases result with
  | Err reason => simp at accepted
  | Ok unitValue =>
    cases unitValue
    exact ⟨by simpa only [ok.injEq, core.result.Result.Ok.injEq] using accepted, computed⟩

open Lean Elab Tactic Meta in
private def launchGuardHead (hypothesis : Syntax) (expected : Name) : TacticM Unit := withMainContext do
  let declaration ← (← getFVarId hypothesis).getDecl
  let arguments := (← instantiateMVars declaration.type).getAppArgs
  let mut left := arguments[1]!.headBeta
  while left.isLet do
    left := left.letBody!.instantiate1 left.letValue! |>.headBeta
  unless left.getAppFn.isConstOf expected do
    throwError "launch prefix has a different head"

open Lean Elab Tactic Meta in
elab "peel_launch_guard " hypothesis:ident : tactic => do
  launchGuardHead hypothesis ``ite
  evalTactic (← `(tactic| (
    obtain ⟨rejectedGuard, nextAccepted⟩ := successful_guard _ _ _ _ $hypothesis
    clear $hypothesis
    have $hypothesis := nextAccepted
    clear nextAccepted)))

open Lean Elab Tactic Meta in
elab "peel_launch_bind " hypothesis:ident value:ident computed:ident : tactic => do
  launchGuardHead hypothesis ``Aeneas.Std.bind
  evalTactic (← `(tactic| (
    obtain ⟨$value, $computed, nextAccepted⟩ := source_bind_success _ _ _ $hypothesis
    clear $hypothesis
    have $hypothesis := nextAccepted
    clear nextAccepted)))

def ValidatedLaunchTrace (_kernel : ptxv_ir.Kernel) (launch : machine.Launch) (count : Usize) : Prop :=
  ∃ grid ctas block,
    alloc.vec.Vec.index_usize launch.grids 0#usize = .ok grid ∧
    strict_memory_launch.dimensions_product grid.grid = .ok (some ctas) ∧
    strict_memory_launch.dimensions_product grid.block = .ok (some block) ∧
    Usize.checked_mul ctas block = some count ∧
    strict_memory_launch.validate_outputs launch = .ok (.Ok ()) ∧
    launch.grids.val.length = 1 ∧ launch.nic = none

theorem source_validate_launch_trace (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (count : Usize)
    (accepted : strict_memory_admission.validate_launch kernel launch = .ok (.Ok count)) :
    ValidatedLaunchTrace kernel launch count ∧ LaunchRestrictions kernel launch := by
  unfold strict_memory_admission.validate_launch at accepted
  peel_launch_bind accepted front frontRead
  repeat' peel_launch_guard accepted
  peel_launch_bind accepted aliased aliasesRead
  repeat' peel_launch_guard accepted
  peel_launch_bind accepted grid gridRead
  peel_launch_bind accepted cluster clusterRead
  repeat' peel_launch_guard accepted
  peel_launch_bind accepted gridProduct gridProductRead
  obtain ⟨ctas, rfl⟩ : ∃ value, gridProduct = some value := by
    cases gridProduct with
    | none => simp at accepted
    | some value => exact ⟨value, rfl⟩
  dsimp only at accepted
  peel_launch_bind accepted blockProduct blockProductRead
  obtain ⟨block, rfl⟩ : ∃ value, blockProduct = some value := by
    cases blockProduct with
    | none => simp at accepted
    | some value => exact ⟨value, rfl⟩
  dsimp only at accepted
  repeat' peel_launch_guard accepted
  peel_launch_bind accepted z zRead
  repeat' peel_launch_guard accepted
  peel_launch_bind accepted product productRead
  obtain ⟨total, rfl⟩ : ∃ value, product = some value := by
    cases product with
    | none => simp at accepted
    | some value => exact ⟨value, rfl⟩
  dsimp only at accepted
  repeat' peel_launch_guard accepted
  have singleGrid : launch.grids.val.length = 1 := by scalar_tac
  have noLdtmScoreboards : kernel.ldtm_sb.val = [] := vector_empty _ (by assumption)
  have noLdtmWaits : kernel.ldtm_waits.val = [] := vector_empty _ (by assumption)
  have noGridWaits : grid.waits.val = [] := vector_empty _ (by assumption)
  have gridRestriction : GridRestrictions grid := by
    constructor
    all_goals first
      | exact noGridWaits
      | exact vector_empty _ (by assumption)
      | exact option_none _ (by assumption)
      | exact boolean_false _ (by assumption)
      | scalar_tac
  have gridsEq : launch.grids.val = [grid] := by
    have read : alloc.vec.Vec.index_usize launch.grids 0#usize = .ok grid := by
      simpa only [alloc.vec.Vec.index_slice_index] using gridRead
    have selected := vec_index_success launch.grids 0#usize grid read
    obtain ⟨first, equal⟩ := List.length_eq_one_iff.mp singleGrid
    have firstEq : first = grid := by simpa [equal] using selected
    simpa only [firstEq] using equal
  have restrictions : LaunchRestrictions kernel launch := by
    constructor
    · constructor
      · exact front_ptx kernel.front front frontRead (by assumption)
      all_goals first
        | exact noLdtmScoreboards
        | exact noLdtmWaits
        | exact vector_empty _ (by assumption)
        | exact option_none _ (by assumption)
        | (simpa using (show ¬alloc.vec.Vec.len kernel.code = 0#usize from by assumption))
    · exact singleGrid
    · intro selected member
      have same : selected = grid := by simpa only [gridsEq, List.mem_singleton] using member
      simpa only [same] using gridRestriction
    all_goals first
      | exact vector_empty _ (by assumption)
      | exact option_none _ (by assumption)
      | exact boolean_false _ (by assumption)
      | (have noAlias : aliased = false := boolean_false _ (by assumption)
         subst aliased
         simpa [alloc.vec.Vec.deref] using source_aliases_present_false launch.aliased.deref aliasesRead)
  have close : ((do
      let result ← strict_memory_launch.validate_outputs launch
      match result with
      | .Ok _ => ok (core.result.Result.Ok total)
      | .Err reason => ok (core.result.Result.Err reason)) = ok (core.result.Result.Ok count)) →
      ValidatedLaunchTrace kernel launch count ∧ LaunchRestrictions kernel launch := by
    intro outputAccepted
    obtain ⟨rfl, outputsRead⟩ := successful_outputs launch total count outputAccepted
    refine ⟨⟨grid, ctas, block, ?_, gridProductRead, blockProductRead, ?_, outputsRead,
      singleGrid, restrictions.noNic⟩, restrictions⟩
    · simpa only [alloc.vec.Vec.index_slice_index] using gridRead
    · simpa only [lift, ok.injEq] using productRead
  cases hntid : kernel.ntid with
  | none =>
    simp only [hntid] at accepted
    exact close accepted
  | some ntid =>
    simp only [hntid] at accepted
    peel_launch_bind accepted limitOption limitRead
    obtain ⟨limit, rfl⟩ : ∃ value, limitOption = some value := by
      cases limitOption with
      | none => simp at accepted
      | some value => exact ⟨value, rfl⟩
    dsimp only at accepted
    peel_launch_guard accepted
    by_cases required : ntid.req = true
    · simp only [required, ite_true] at accepted
      peel_launch_bind accepted sameBlock sameBlockRead
      peel_launch_guard accepted
      exact close accepted
    · simp only [required] at accepted
      peel_launch_guard accepted
      exact close accepted

end MemorySafetyIR
