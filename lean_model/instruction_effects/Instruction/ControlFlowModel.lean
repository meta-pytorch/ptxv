/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs

open Aeneas Aeneas.Std

set_option autoImplicit false

namespace Instruction.ControlFlow

structure Context where
  pc : Nat
  count : Nat
  cut : U8
  fold : Option Bool
  returns : Option (List Nat)

def always (guard : PtxvIr.Pred) : Bool := guard.reg.isNone && !guard.neg

def next (context : Context) : List Nat :=
  if context.pc + 1 < context.count then [context.pc + 1] else []

def returnTargets (context : Context) : List Nat := context.returns.getD (List.range context.count)

def successors (operation : PtxvIr.Op) (guard : PtxvIr.Pred) (context : Context) : List Nat :=
  match operation with
  | .ClusterWait => if context.cut &&& 1#u8 ≠ 0#u8 then [] else next context
  | .BarSync _ none false _ _ _ =>
      if context.cut &&& 2#u8 ≠ 0#u8 && always guard then [] else next context
  | .Bra target any tid0 all condition divergent =>
      if always guard && !tid0 && condition.isNone && divergent.isNone then [target.val]
      else if !any && !all && !tid0 && condition.isNone && divergent.isNone then
        match context.fold with
        | some true => [target.val]
        | some false => next context
        | none => target.val :: next context
      else target.val :: next context
  | .Call target => target.val :: next context
  | .Ret _ => returnTargets context ++ if always guard then [] else next context
  | .Brx _ _ targets => targets.val.map UScalar.val ++ if always guard then [] else next context
  | .Exit | .Trap _ => if always guard then [] else next context
  | _ => next context

def Admissible (operation : PtxvIr.Op) (guard : PtxvIr.Pred) (context : Context) (destination : Nat) : Prop :=
  destination ∈ successors operation guard context

theorem guarded_brx_fallthrough (base : PtxvIr.Val) (offset : I64) (targets : alloc.vec.Vec Usize)
    (guard : PtxvIr.Pred) (context : Context) (conditional : always guard = false)
    (inside : context.pc + 1 < context.count) :
    Admissible (.Brx base offset targets) guard context (context.pc + 1) := by
  simp only [Admissible, successors, conditional, Bool.false_eq_true, ite_false,
    next, inside, ite_true, List.mem_append, List.mem_singleton, or_true]

theorem conditioned_branch_fallthrough (target : Usize) (condition : PtxvIr.Pred)
    (guard : PtxvIr.Pred) (context : Context) (inside : context.pc + 1 < context.count) :
    Admissible (.Bra target false false false (some condition) none) guard context (context.pc + 1) := by
  simp [Admissible, successors, next, inside]

theorem collective_branch_does_not_use_lane_fold (target : Usize) (guard : PtxvIr.Pred)
    (context : Context) :
    successors (.Bra target false false true none none) guard context =
      if always guard then [target.val] else target.val :: next context := by
  simp [successors]

theorem branch_targets_are_admissible (base : PtxvIr.Val) (offset : I64)
    (targets : alloc.vec.Vec Usize) (guard : PtxvIr.Pred) (context : Context)
    (destination : Usize) (member : destination ∈ targets.val) :
    Admissible (.Brx base offset targets) guard context destination.val := by
  apply List.mem_append_left
  exact List.mem_map.mpr ⟨destination, member, rfl⟩

end Instruction.ControlFlow
