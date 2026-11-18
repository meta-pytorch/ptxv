/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace MemoryWrite.AtomicComponentOrder

structure ComponentHistory where
  oldA : Nat
  oldB : Nat
  final : Nat
  deriving DecidableEq

def addComponent (aFirst : Bool) (a b : Nat) : ComponentHistory :=
  if aFirst then ⟨0, a, a + b⟩ else ⟨b, 0, b + a⟩

structure PackedHistory where
  oldA : Nat × Nat
  oldB : Nat × Nat
  final : Nat × Nat
  deriving DecidableEq

def pack (lo hi : ComponentHistory) : PackedHistory :=
  ⟨(lo.oldA, hi.oldA), (lo.oldB, hi.oldB), (lo.final, hi.final)⟩

def jointOrder (aFirst : Bool) : PackedHistory :=
  pack (addComponent aFirst 1 2) (addComponent aFirst 10 20)

def mixedOrder : PackedHistory :=
  pack (addComponent true 1 2) (addComponent false 10 20)

theorem mixed_order_observations :
    mixedOrder.oldA = (0, 20) ∧ mixedOrder.oldB = (1, 0) ∧
      mixedOrder.final = (3, 30) := by
  decide

theorem final_alone_cannot_certify_atomic_returns :
    (∀ aFirst, mixedOrder.final = (jointOrder aFirst).final) ∧
      (∀ aFirst, mixedOrder ≠ jointOrder aFirst) := by
  constructor <;> intro aFirst <;> cases aFirst <;> decide

end MemoryWrite.AtomicComponentOrder
