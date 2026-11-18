/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicExecution
import MemoryWrite.RustAtomicComponentPlan

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.AtomicRequestLog

def inputOf (request : atomic_exec.AtomicRequest) (old : U64) : atomic_exec.AtomicInput :=
  { op := request.op, bytes := request.bytes, old, value := request.value, compare := request.compare }

def faddOperand (request : atomic_exec.AtomicRequest) : Option U32 :=
  match request.op with
  | .AddF32Ftz => if request.bytes = 4#u8 then some (UScalar.cast .U32 request.value) else none
  | _ => none

@[step] theorem rust_resolve_old (request : atomic_exec.AtomicRequest) (old : U64) :
    atomic_exec.resolve_old request old ⦃ input => input = inputOf request old ⦄ := by
  simp [atomic_exec.resolve_old, inputOf]

@[step] theorem rust_fadd_operand (request : atomic_exec.AtomicRequest) :
    atomic_exec.fadd_operand request ⦃ logged => logged = faddOperand request ⦄ := by
  cases request with
  | mk op bytes value compare =>
    cases op <;> by_cases width : bytes = 4#u8 <;>
      simp [atomic_exec.fadd_operand, faddOperand, width, lift, bind_ok]

@[step] theorem source_fadd_log_matches_actual_compute {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (arithmetic : AtomicFloatContract implementation semantics)
    (value compare old : U64) :
    (do
      let request : atomic_exec.AtomicRequest := ⟨.AddF32Ftz, 4#u8, value, compare⟩
      let input ← atomic_exec.resolve_old request old
      let logged ← atomic_exec.fadd_operand request
      let result ← atomic_exec.execute implementation input
      ok (logged, result))
      ⦃ result => result.1 = some (UScalar.cast .U32 value) ∧
        ∃ output, result.2 = .Ok output ∧ output.replacement = UScalar.cast .U64
          (semantics.addF32Ftz (UScalar.cast .U32 old) (UScalar.cast .U32 value)) ⦄ := by
  step with rust_resolve_old ⟨.AddF32Ftz, 4#u8, value, compare⟩ old as ⟨input, inputEq⟩
  subst input
  step with rust_fadd_operand ⟨.AddF32Ftz, 4#u8, value, compare⟩ as ⟨logged, loggedEq⟩
  step with rust_atomic_float_execute_spec implementation semantics arithmetic
    (inputOf ⟨.AddF32Ftz, 4#u8, value, compare⟩ old) (by intro op; simp [inputOf])
    as ⟨result, resultEq⟩
  simp only [inputOf, AtomicSupported] at resultEq
  simp only [loggedEq, faddOperand, ↓reduceIte]
  refine ⟨trivial, _, resultEq, ?_⟩
  simp [atomicFinishModel, atomicFloatResult]

@[step] theorem source_component_log_is_frozen (input : atomic_components.FrozenOperands)
    (plan : atomic_components.Plan) (component : atomic_components.Component) :
    (do
      let selected ← atomic_components.begin_component input plan component
      match selected with
      | none => ok none
      | some selected =>
        let logged ← atomic_exec.fadd_operand
          ⟨.AddF32Ftz, 4#u8, UScalar.cast .U64 selected.source.captured, 0#u64⟩
        ok (some (selected.source, logged)))
      ⦃ result => ∀ source logged, result = some (source, logged) →
        logged = some source.captured ∧
        source = AtomicComponentPlan.frozenOperand input (AtomicComponentPlan.componentValue component) ⦄ := by
  step with AtomicComponentPlan.rust_begin_component input plan component as ⟨selected, remaining, frozen⟩
  cases selected with
  | none => simp
  | some selected =>
    step with rust_fadd_operand
      ⟨.AddF32Ftz, 4#u8, UScalar.cast .U64 selected.source.captured, 0#u64⟩
      as ⟨logged, loggedEq⟩
    simp only [Option.some.injEq, Prod.mk.injEq]
    intro source actual equal
    obtain ⟨sourceEq, loggedSame⟩ := equal
    subst source
    subst actual
    refine ⟨?_, (frozen selected rfl).1⟩
    simp only [faddOperand, ↓reduceIte] at loggedEq
    rw [loggedEq]
    congr 1
    scalar_tac

end MemoryWrite.AtomicRequestLog
