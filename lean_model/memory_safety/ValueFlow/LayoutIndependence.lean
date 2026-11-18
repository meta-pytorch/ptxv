/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.Semantics

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option debug.skipKernelTC false

namespace ValueFlow

structure WordRelated (left right : Word) : Prop where
  undefined : left.undefined = right.undefined
  pointer : left.pointer = right.pointer
  scalar : left.pointer = none → left.bits = right.bits

structure BanksRelated (left right : RegisterBank) : Prop where
  predicates : left.predicate = right.predicate
  words : ∀ index,
    match left.virtual index, right.virtual index with
    | none, none => True
    | some left, some right => WordRelated left right
    | _, _ => False

theorem read_word_rebase (left right : Word) (related : WordRelated left right) (bits : Nat) :
    readWord bits left = readWord bits right := by
  rcases left with ⟨leftBits, leftUndefined, leftPointer⟩
  rcases right with ⟨rightBits, rightUndefined, rightPointer⟩
  rcases related with ⟨undefined, pointer, scalar⟩
  dsimp only at undefined pointer scalar
  subst rightUndefined rightPointer
  cases leftPointer with
  | none =>
    have same := scalar rfl
    subst rightBits
    rfl
  | some pointer => rfl

theorem read_guard_rebase (left right : RegisterBank) (related : BanksRelated left right)
    (guard : Guard) : readGuard left guard = readGuard right guard := by
  simp only [readGuard, related.predicates]

theorem read_operand_rebase (left right : RegisterBank) (related : BanksRelated left right)
    (ids : ThreadIds) (bits : Nat) (operand : Operand) :
    readOperand left ids bits operand = readOperand right ids bits operand := by
  cases operand with
  | immediate _ | pointer _ | special _ _ => rfl
  | virtual index =>
    have wordRelation := related.words index
    cases l : left.virtual index <;> cases r : right.virtual index <;>
      simp only [l, r] at wordRelation
    · simp only [readOperand, l, r, Option.bind_eq_bind, Option.bind_none]
    · simp only [readOperand, l, r, Option.bind_eq_bind, Option.bind_some,
        read_word_rebase _ _ wordRelation]

theorem address_rebase (left right : RegisterBank) (related : BanksRelated left right)
    (ids : ThreadIds) (location : Address) :
    address left ids location = address right ids location := by
  simp only [address, read_operand_rebase left right related]

theorem evaluate_instruction_rebase (left right : RegisterBank) (related : BanksRelated left right)
    (ids : ThreadIds) (instruction : Instruction) :
    evaluateInstruction left ids instruction = evaluateInstruction right ids instruction := by
  cases instruction <;>
    simp only [evaluateInstruction, read_operand_rebase left right related,
      address_rebase left right related]

theorem evaluate_rebase (left right : RegisterBank) (related : BanksRelated left right)
    (ids : ThreadIds) (instruction : GuardedInstruction) :
    evaluate left ids instruction = evaluate right ids instruction := by
  simp only [evaluate, read_guard_rebase left right related,
    evaluate_instruction_rebase left right related]

end ValueFlow
