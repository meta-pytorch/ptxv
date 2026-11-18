/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.Semantics

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace ValueFlow

theorem read_word_scalar_bound (word : Word) (bits value : Nat)
    (computed : readWord bits word = some (.scalar value)) : value < 2 ^ bits := by
  unfold readWord at computed
  split at computed
  · cases pointer : word.pointer <;> simp only [pointer] at computed
    · simp only [Option.some.injEq, Value.scalar.injEq] at computed
      subst value
      exact Nat.mod_lt _ (by positivity)
    · split at computed <;> simp_all
  · simp_all

theorem read_operand_scalar_bound (bank : RegisterBank) (ids : ThreadIds) (bits value : Nat)
    (operand : Operand) (computed : readOperand bank ids bits operand = some (.scalar value)) :
    value < 2 ^ bits := by
  unfold readOperand at computed
  split at computed
  · cases operand with
    | immediate raw =>
      simp only [Option.some.injEq, Value.scalar.injEq] at computed
      subst value
      exact Nat.mod_lt _ (by positivity)
    | virtual index =>
      cases word : bank.virtual index <;>
        simp only [word, Option.bind_eq_bind, Option.bind_some, Option.bind_none] at computed
      next => simp_all
      next actual => exact read_word_scalar_bound actual bits value computed
    | pointer pointer => split at computed <;> simp_all
    | special kind axis =>
      cases found : readSpecial ids kind axis <;>
        simp only [found, Option.map_some, Option.map_none] at computed
      next => simp_all
      next raw =>
        simp only [Option.some.injEq, Value.scalar.injEq] at computed
        subst value
        exact Nat.mod_lt _ (by positivity)
  · simp_all

end ValueFlow
