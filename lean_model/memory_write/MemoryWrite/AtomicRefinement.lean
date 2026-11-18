/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicExecution
import MemoryWrite.AtomicInteger

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite

def atomicIntegerView (input : atomic_exec.AtomicInput) (op : atomic_integer.IntegerOp) :
    atomic_integer.IntegerInput :=
  ⟨op, input.bytes, input.old, input.value, input.compare⟩

def atomicComputedModel (semantics : AtomicFloatSemantics) (input : atomic_exec.AtomicInput) :
    core.result.Result atomic_integer.IntegerResult atomic_integer.AtomicError :=
  match input.op with
  | .Integer op => integerModel (atomicIntegerView input op)
  | _ => if AtomicSupported input.op input.bytes then .Ok (atomicFloatResult semantics input)
    else .Err .UnsupportedWidth

def atomicExecutionModel (semantics : AtomicFloatSemantics) (input : atomic_exec.AtomicInput) :
    core.result.Result atomic_exec.AtomicOutcome atomic_integer.AtomicError :=
  match atomicComputedModel semantics input with
  | .Ok computed => .Ok (atomicFinishModel input computed)
  | .Err error => .Err error

theorem atomic_integer_supported_iff (op : atomic_integer.IntegerOp) (bytes : U8) :
    AtomicSupported (.Integer op) bytes ↔ bytes = 4#u8 ∨
      (bytes = 8#u8 ∧ op ≠ .Inc ∧ op ≠ .Dec) := by
  cases op <;> simp [AtomicSupported]

theorem integerModel_rejects (input : atomic_exec.AtomicInput) (op : atomic_integer.IntegerOp)
    (hwidth : ¬AtomicSupported (.Integer op) input.bytes) :
    integerModel (atomicIntegerView input op) = .Err .UnsupportedWidth := by
  have hn := (not_congr (atomic_integer_supported_iff op input.bytes)).mp hwidth
  simp [integerModel, atomicIntegerView, hn]

theorem rust_atomic_integer_execute_spec {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (input : atomic_exec.AtomicInput) (op : atomic_integer.IntegerOp)
    (hoperation : input.op = .Integer op) :
    atomic_exec.execute implementation input
      ⦃ result => result = atomicExecutionModel semantics input ⦄ := by
  by_cases hwidth : AtomicSupported input.op input.bytes
  · unfold atomic_exec.execute
    apply WP.spec_bind (rust_atomic_validate_spec input.op input.bytes)
    intro validated hvalidated
    rw [ite_eq_left hwidth] at hvalidated
    rw [hvalidated, hoperation]
    apply WP.spec_bind (rust_integer_eval_spec (atomicIntegerView input op))
    intro computed hcomputed
    rw [hcomputed]
    cases hmodel : integerModel (atomicIntegerView input op) with
    | Err error => simp [atomicExecutionModel, atomicComputedModel, hoperation, hmodel]
    | Ok value =>
      apply WP.spec_bind (rust_atomic_finish_spec input value)
      intro output houtput
      simpa only [WP.spec_ok, atomicExecutionModel, atomicComputedModel, hoperation, hmodel]
        using congrArg core.result.Result.Ok houtput
  · have hreject := rust_atomic_execute_rejects implementation input hwidth
    have hn : ¬AtomicSupported (.Integer op) input.bytes := by simpa [hoperation] using hwidth
    simpa [atomicExecutionModel, atomicComputedModel, hoperation, integerModel_rejects input op hn]
      using hreject

@[step]
theorem rust_atomic_execute_spec {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (hfloat : AtomicFloatContract implementation semantics) (input : atomic_exec.AtomicInput) :
    atomic_exec.execute implementation input
      ⦃ result => result = atomicExecutionModel semantics input ⦄ := by
  cases hop : input.op with
  | Integer op => exact rust_atomic_integer_execute_spec implementation semantics input op hop
  | AddF32Ftz | AddF64 | AddBf16 | AddBf16x2 =>
    have hnot : ∀ op, input.op ≠ .Integer op := by intro op; simp [hop]
    have hsource := rust_atomic_float_execute_spec implementation semantics hfloat input hnot
    have hmodel : atomicExecutionModel semantics input =
        if AtomicSupported input.op input.bytes then
          .Ok (atomicFinishModel input (atomicFloatResult semantics input))
        else .Err .UnsupportedWidth := by
      by_cases hwidth : AtomicSupported input.op input.bytes <;>
        have hw := hwidth <;> rw [hop] at hw <;>
        simp [atomicExecutionModel, atomicComputedModel, hop, hw]
    simpa only [hmodel] using hsource

theorem atomicComputedModel_success (semantics : AtomicFloatSemantics)
    (input : atomic_exec.AtomicInput) (hwidth : AtomicSupported input.op input.bytes) :
    ∃ computed, atomicComputedModel semantics input = .Ok computed := by
  cases hop : input.op with
  | Integer op =>
    have hw := (atomic_integer_supported_iff op input.bytes).mp (by simpa [hop] using hwidth)
    simp [atomicComputedModel, hop, integerModel, atomicIntegerView, hw]
  | AddF32Ftz | AddF64 | AddBf16 | AddBf16x2 =>
    refine ⟨atomicFloatResult semantics input, ?_⟩
    have hw := hwidth
    rw [hop] at hw
    simp [atomicComputedModel, hop, hw]

structure AtomicOutcomeControls (input : atomic_exec.AtomicInput)
    (outcome : atomic_exec.AtomicOutcome) : Prop where
  width : outcome.bytes = input.bytes
  writes : outcome.did_write = (!atomicIsCas input.op || outcome.swapped)
  epoch : outcome.advance_epoch = outcome.did_write
  returned : outcome.returned =
    if atomicIsSpin input.op then if outcome.swapped then 1#u64 else 0#u64 else outcome.old
  returnWidth : outcome.return_words =
    if atomicIsSpin input.op then 1#u8 else if input.bytes = 8#u8 then 2#u8 else 1#u8
  predicate : outcome.predicate = if atomicIsSpin input.op then outcome.swapped else true

theorem atomicFinishModel_controls (input : atomic_exec.AtomicInput)
    (computed : atomic_integer.IntegerResult) :
    AtomicOutcomeControls input (atomicFinishModel input computed) := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem rust_atomic_execute_progress {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (hfloat : AtomicFloatContract implementation semantics) (input : atomic_exec.AtomicInput)
    (hwidth : AtomicSupported input.op input.bytes) :
    ∃ outcome, atomic_exec.execute implementation input = .ok (.Ok outcome) ∧
      atomicExecutionModel semantics input = .Ok outcome ∧ AtomicOutcomeControls input outcome := by
  obtain ⟨computed, hcomputed⟩ := atomicComputedModel_success semantics input hwidth
  obtain ⟨actual, hactual, hmodel⟩ := spec_returns _ _
    (rust_atomic_execute_spec implementation semantics hfloat input)
  have hexpected : atomicExecutionModel semantics input = .Ok (atomicFinishModel input computed) := by
    simp [atomicExecutionModel, hcomputed]
  refine ⟨atomicFinishModel input computed, ?_, hexpected, atomicFinishModel_controls input computed⟩
  simpa only [hmodel, hexpected] using hactual

theorem rust_atomic_execute_footprint {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (hfloat : AtomicFloatContract implementation semantics) (input : atomic_exec.AtomicInput)
    (offset : U64) (hwidth : AtomicSupported input.op input.bytes) :
    ∃ outcome effect,
      atomic_exec.execute implementation input = .ok (.Ok outcome) ∧
      atomicExecutionModel semantics input = .Ok outcome ∧ AtomicOutcomeControls input outcome ∧
      atomic.atomic_footprint offset outcome.bytes outcome.did_write = .ok effect ∧
      AtomicFootprintBounds offset input.bytes (!atomicIsCas input.op || outcome.swapped) effect ∧
      (∀ word, AtomicHistoryMember effect word ↔
        if (!atomicIsCas input.op || outcome.swapped) = true then
          input.bytes.val > 0 ∧ word.val < offset.val + input.bytes.val ∧ offset.val < word.val + 4
        else word = offset) ∧
      (∀ word, AtomicMarkerMember effect word ↔
        if (!atomicIsCas input.op || outcome.swapped) = true then
          input.bytes.val > 0 ∧ word.val % 4 = 0 ∧ word.val < offset.val + input.bytes.val ∧
            offset.val < word.val + 4
        else word = offset) := by
  obtain ⟨outcome, hactual, hmodel, hcontrols⟩ :=
    rust_atomic_execute_progress implementation semantics hfloat input hwidth
  obtain ⟨effect, heffect, hbounds, hhistory, hmarkers⟩ :=
    rust_atomic_footprint_spec offset outcome.bytes outcome.did_write
  refine ⟨outcome, effect, hactual, hmodel, hcontrols, heffect, ?_, ?_, ?_⟩
  · simpa only [hcontrols.width, hcontrols.writes] using hbounds
  · simpa only [hcontrols.width, hcontrols.writes] using hhistory
  · simpa only [hcontrols.width, hcontrols.writes] using hmarkers

end MemoryWrite
