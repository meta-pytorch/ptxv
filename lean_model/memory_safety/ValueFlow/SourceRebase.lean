/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceModel
import ValueFlow.LayoutIndependence

open Aeneas Aeneas.Std

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option debug.skipKernelTC false

namespace ValueFlow.Source

structure RegistersRelated (left right : V.Registers) : Prop where
  virtualLength : left.raw.regs.val.length = right.raw.regs.val.length
  pointerTags : left.pointers = right.pointers
  written : left.raw.written = right.raw.written
  undefined : left.raw.vund = right.raw.vund
  predicates : left.raw.preds = right.raw.preds
  predicateDefinitions : left.raw.pdef = right.raw.pdef
  scalars : ∀ index : Nat,
    left.pointers.val[index]?.getD none = none →
    left.raw.regs.val[index]?.getD 0#u64 = right.raw.regs.val[index]?.getD 0#u64

theorem source_banks_related (left right : V.Registers) (related : RegistersRelated left right) :
    BanksRelated (bankModel left) (bankModel right) := by
  constructor
  · funext index
    simp only [bankModel, predicateModel, related.predicates, related.predicateDefinitions]
  · intro index
    simp only [bankModel, wordModel, related.virtualLength, related.pointerTags, related.written,
      related.undefined]
    by_cases shape : index < right.raw.regs.val.length ∧
        index < right.pointers.val.length ∧ index / 64 < right.raw.written.val.length
    · simp only [shape]
      by_cases written : (right.raw.written.val[index / 64]?.getD 0#u64).val /
          2 ^ (index % 64) % 2 = 1
      · simp only [written, ite_true]
        refine ⟨rfl, rfl, ?_⟩
        intro untagged
        have noPointer : right.pointers.val[index]?.getD none = none := by
          cases pointer : right.pointers.val[index]?.getD none <;> simp_all
        apply congrArg UScalar.val
        apply related.scalars index
        simpa only [related.pointerTags] using noPointer
      · simp only [written, ite_false, and_self, ite_true]
    · simp only [shape, ite_false]

theorem source_evaluate_rebase (left right : V.Registers) (related : RegistersRelated left right)
    (ids : V.ThreadIds) (instruction : V.GuardedInstruction) :
    evaluate (bankModel left) (idsModel ids) (guardedModel instruction) =
      evaluate (bankModel right) (idsModel ids) (guardedModel instruction) :=
  evaluate_rebase _ _ (source_banks_related left right related) _ _

end ValueFlow.Source
