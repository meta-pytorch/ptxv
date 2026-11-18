/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateValidation
import Instruction.GuardCertificateDecision
import Instruction.ControlFlowRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.GuardCertificate

open Instruction.ControlFlow

def semanticEdges (operation : Op) (guard : Pred) (truth : Bool) (context : Context) : List Nat :=
  successors operation (effectiveGuard guard (some truth)) {context with fold := some truth}

def checkedEdges (operation : Op) (guard : Pred) (known : Option Bool) (context : Context) : List Nat :=
  successors operation (effectiveGuard guard known) {context with fold := known}

theorem unknown_edges_cover_truth (operation : Op) (guard : Pred) (truth : Bool)
    (context : Context) (destination : Nat)
    (structural : always guard = true → truth = true)
    (member : destination ∈ semanticEdges operation guard truth context) :
    destination ∈ checkedEdges operation guard none context := by
  cases operation <;>
    simp only [semanticEdges, checkedEdges, successors, effectiveGuard] at member ⊢
  all_goals try exact member
  case Bra target any tid0 all condition divergent =>
    cases truth <;> cases condition <;> cases divergent <;> cases any <;> cases tid0 <;> cases all <;>
      (by_cases present : always guard = true) <;>
      simp_all [Pred.TRUE, next]
    all_goals split at member <;> simp_all [always]
  case Ret address =>
    cases truth <;> (by_cases present : always guard = true) <;>
      simp_all [always, Pred.TRUE, next, returnTargets]
  case Brx base offset targets =>
    cases truth <;> (by_cases present : always guard = true) <;>
      simp_all [always, Pred.TRUE, next, returnTargets]
  case Exit | Trap =>
    cases truth <;> (by_cases present : always guard = true) <;>
      simp_all [always, Pred.TRUE, next, returnTargets]
  case BarSync id count arrive packed aligned reduction =>
    cases count <;> cases arrive <;> cases truth <;>
      simp_all [always, Pred.TRUE, next]
    all_goals split at member <;> simp_all [always]

theorem checked_edges_cover_truth (operation : Op) (guard : Pred) (truth : Bool)
    (known : Option Bool) (context : Context) (destination : Nat)
    (structural : always guard = true → truth = true)
    (compatible : ∀ observed, known = some observed → observed = truth)
    (member : destination ∈ semanticEdges operation guard truth context) :
    destination ∈ checkedEdges operation guard known context := by
  cases known with
  | none => exact unknown_edges_cover_truth operation guard truth context destination structural member
  | some observed =>
    have same := compatible observed rfl
    subst observed
    exact member

theorem structural_guard_true (front : Front) (registers : Reg → U64) (guard : Pred)
    (structural : always guard = true) :
    RegisterOwner.predicateRead front (fun register => registers register != 0#u64) guard = true := by
  cases guard with
  | mk register neg =>
    cases register <;> cases neg <;>
      simp_all [always, RegisterOwner.predicateRead]

end Instruction.GuardCertificate
