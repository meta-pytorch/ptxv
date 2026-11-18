/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateDefinedPublication
import Interop.NativeGraphPublication

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.GuardCertificate
open MemoryWrite.MayWrite MemoryWrite.BarrierReach
set_option autoImplicit false
set_option maxHeartbeats 7000000
namespace Instruction.ControlFlow

structure CertifiedFolds {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (bank : U32 → U8 → Option U64)
    (fold : alloc.vec.Vec (Option Bool)) where
  observations : alloc.vec.Vec Observation
  accepted : guard_certificate.validate reader certificate = .ok (some observations)
  rowsFit : RowsFit certificate
  framesFit : FramesFit certificate
  constantsSound : ConstantsSound reader certificate.constants bank
  rows : List (Slice Observation)
  before : Slice Observation
  after : Slice Observation
  merged : MergeRowsCalls rows before after
  member : alloc.vec.Vec.deref observations ∈ rows
  published : guard_certificate.publish certificate.code after = .ok (some fold)
  initiallyEmpty : ∀ pc < certificate.code.val.length, before[pc]! = none

theorem CertifiedFolds.known_defined {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (bank : U32 → U8 → Option U64)
    (fold : alloc.vec.Vec (Option Bool)) (checked : CertifiedFolds reader certificate bank fold)
    (pc : Nat) (registers : PartialRegisters) (execution : ExecutionDefined certificate bank pc registers)
    (instruction : Inst) (instructionAt : certificate.code.val[pc]? = some instruction)
    (truth : Bool) (known : fold.val[pc]?.getD none = some truth) :
    partialPredicate certificate.front registers instruction.guard = some truth := by
  have inside := (List.getElem?_eq_some_iff.mp instructionAt).1
  have outputAt : fold.val[pc]? = some (some truth) := by
    cases observed : fold.val[pc]? <;> simp_all
  exact published_rows_guard_defined reader certificate checked.observations bank checked.accepted
    checked.rowsFit checked.framesFit checked.constantsSound checked.rows checked.before checked.after
    checked.merged checked.member fold checked.published pc (checked.initiallyEmpty pc inside) registers
    execution instruction instructionAt truth outputAt

theorem semantic_edges_in_folded_cfg (operation : Op) (guard : Pred) (truth : Bool)
    (known : Option Bool) (context : Context) (destination : Nat)
    (structural : always guard = true → truth = true)
    (compatible : ∀ observed, known = some observed → observed = truth)
    (edge : destination ∈ semanticEdges operation guard truth context) :
    destination ∈ successors operation guard {context with fold := known} := by
  cases operation <;>
    simp only [semanticEdges, successors, effectiveGuard] at edge ⊢
  all_goals try exact edge
  case Bra target any tid0 all condition divergent =>
    cases known with
    | none =>
      cases truth <;> cases condition <;> cases divergent <;> cases any <;> cases tid0 <;> cases all <;>
        (by_cases present : always guard = true) <;> simp_all [Pred.TRUE, always, next]
      all_goals split <;> simp_all
    | some observed =>
      have same := compatible observed rfl
      subst observed
      cases truth <;> cases condition <;> cases divergent <;> cases any <;> cases tid0 <;> cases all <;>
        (by_cases present : always guard = true) <;> simp_all [Pred.TRUE, always, next]
      all_goals split <;> simp_all
  case Ret address | Brx | Exit | Trap =>
    cases truth <;> (by_cases present : always guard = true) <;>
      simp_all [always, Pred.TRUE, next, returnTargets]
  case BarSync id count arrive packed aligned reduction =>
    cases count <;> cases arrive <;> cases truth <;>
      simp_all [always, Pred.TRUE, next]

structure DynamicStep {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) (pc : Nat) (before : PartialRegisters)
    (destination : Nat) (after : PartialRegisters) where
  instruction : Inst
  instructionAt : certificate.code.val[pc]? = some instruction
  post : StepPostDefined certificate.front instruction bank before after
  truth : Bool
  choice : ∀ observed, partialPredicate certificate.front before instruction.guard = some observed → truth = observed
  edge : destination ∈ semanticEdges instruction.op instruction.guard truth (nodeContext certificate pc)

theorem DynamicStep.execution {C : Type} (certificate : guard_certificate.Certificate C)
    (bank : U32 → U8 → Option U64) (pc destination : Nat) (before after : PartialRegisters)
    (step : DynamicStep certificate bank pc before destination after)
    (execution : ExecutionDefined certificate bank pc before) :
    ExecutionDefined certificate bank destination after :=
  .own execution step.instructionAt step.post step.truth step.choice step.edge

theorem CertifiedFolds.dynamic_edge {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C) (bank : U32 → U8 → Option U64)
    (fold : alloc.vec.Vec (Option Bool)) (checked : CertifiedFolds reader certificate bank fold)
    (graph : PtxvMemoryWrite.cache_graph.MayWriteGraph) (mask : U32)
    (pc destination : Nat) (before after : PartialRegisters)
    (execution : ExecutionDefined certificate bank pc before)
    (step : DynamicStep certificate bank pc before destination after) :
    (rawCfgView graph certificate.code.val 0#u8 fold.val certificate.returns.val mask).edge pc destination := by
  refine ⟨step.instruction, step.instructionAt, ?_⟩
  have compatible : ∀ observed, fold.val[pc]?.getD none = some observed → observed = step.truth := by
    intro observed present
    exact (step.choice observed (checked.known_defined reader certificate bank fold pc before execution
      step.instruction step.instructionAt observed present)).symm
  have structural : always step.instruction.guard = true → step.truth = true := by
    intro present
    have parts : step.instruction.guard.reg = none ∧ step.instruction.guard.neg = false := by
      simpa [always] using present
    exact step.choice true (by simp [partialPredicate, Instruction.PredicateRead.meaning, parts.1, parts.2])
  have included := semantic_edges_in_folded_cfg step.instruction.op step.instruction.guard step.truth
    (fold.val[pc]?.getD none) (nodeContext certificate pc) destination structural compatible step.edge
  simpa only [Admissible, nodeContext, rowContext, returnsAt, Option.map_map, Function.comp_def,
    alloc.vec.Vec.deref, Slice.from_val] using included

end Instruction.ControlFlow
