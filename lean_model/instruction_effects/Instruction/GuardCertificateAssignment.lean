/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateMove
import Instruction.GuardCertificateComparison

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.GuardCertificate

def AssignmentTruth (front : Front) (operation : Op) (registers : Reg → U64)
    (bank : U32 → U8 → Option U64) (fact : Fact) : Prop :=
  match operation with
  | .Mov destination operand =>
    (destination.file = .V ∨ destination.file = .R ∧ destination.idx.val < 255) ∧
    scalarMove destination operand = true ∧
    ∃ value, operandValue registers bank operand = some value ∧
      fact.register = destination ∧ fact.value.val = moveWord destination value
  | .Setp destination comparison bits left right and or false none =>
    destination.file = .P ∧ ¬(front = .Sass ∧ destination.idx = 7#u16) ∧
    ComparisonTruth {front, destination, comparison, bits, and, or} (left, right) registers bank fact
  | _ => False

theorem source_assignment_truth {C : Type} (reader : guard_certificate.Constants C)
    (front : Front) (instruction : Inst) (facts : Slice Fact) (constants : C)
    (registers : Reg → U64) (bank : U32 → U8 → Option U64)
    (sound : FactsSound facts.val registers) (constantSound : ConstantsSound reader constants bank)
    (fact : Fact)
    (returned : guard_certificate.assignment reader front instruction facts constants = .ok (some fact)) :
    AssignmentTruth front instruction.op registers bank fact := by
  cases instruction with
  | mk guard operation uniform loc text =>
    cases operation <;> try (solve | simp [guard_certificate.assignment] at returned)
    case Mov destination operand =>
      exact source_move_assignment_sound reader destination operand facts constants registers bank
        sound constantSound fact returned
    case Setp destination comparison bits left right and or xor extra =>
      cases extra with
      | some _ => simp [guard_certificate.assignment] at returned
      | none =>
        cases xor with
        | true => simp [guard_certificate.assignment] at returned
        | false =>
          exact source_comparison_assignment_sound reader
            {front, destination, comparison, bits, and, or} (left, right)
            facts constants registers bank sound constantSound fact returned

end Instruction.GuardCertificate
