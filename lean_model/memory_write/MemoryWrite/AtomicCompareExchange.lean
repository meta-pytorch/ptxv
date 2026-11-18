/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicRefinement
import MemoryWrite.RustNumeric

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite

def atomicCasResultModel (input : atomic_exec.AtomicInput) : atomic_integer.IntegerResult :=
  let old := integerNormalize input.bytes input.old
  let value := integerNormalize input.bytes input.value
  let compare := integerNormalize input.bytes input.compare
  ⟨old, if old = compare then value else old, decide (old = compare)⟩

theorem integerModel_cas_exact (input : atomic_exec.AtomicInput) (op : atomic_integer.IntegerOp)
    (hcas : op = .Cas ∨ op = .CasSpin) (hwidth : input.bytes = 4#u8 ∨ input.bytes = 8#u8) :
    integerModel (atomicIntegerView input op) = .Ok (atomicCasResultModel input) := by
  have hvalid : input.bytes = 4#u8 ∨
      (input.bytes = 8#u8 ∧ op ≠ .Inc ∧ op ≠ .Dec) := by
    rcases hcas with rfl | rfl <;> simp_all
  unfold integerModel
  simp only [atomicIntegerView, hvalid, ite_true]
  by_cases hequal : integerNormalize input.bytes input.old =
      integerNormalize input.bytes input.compare
  · rcases hcas with rfl | rfl <;>
      simp [integerUpdate, integerIsCas, atomicCasResultModel, hequal]
  · have hneval : (integerNormalize input.bytes input.old).val ≠
        (integerNormalize input.bytes input.compare).val :=
      fun h => hequal (UScalar.eq_of_val_eq h)
    rcases hcas with rfl | rfl <;>
      simp [integerUpdate, integerIsCas, atomicCasResultModel, hequal, hneval]

structure AtomicCompareExchangeContract (input : atomic_exec.AtomicInput)
    (outcome : atomic_exec.AtomicOutcome) : Prop where
  old : outcome.old = integerNormalize input.bytes input.old
  replacement : outcome.replacement =
    if integerNormalize input.bytes input.old = integerNormalize input.bytes input.compare then
      integerNormalize input.bytes input.value else integerNormalize input.bytes input.old
  swapped : outcome.swapped =
    decide (integerNormalize input.bytes input.old = integerNormalize input.bytes input.compare)
  controls : AtomicOutcomeControls input outcome

private def unusedFloatSemantics : AtomicFloatSemantics :=
  ⟨fun old _ => old, fun old _ => old, fun old _ => old⟩

theorem rust_atomic_compare_exchange_progress {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (input : atomic_exec.AtomicInput)
    (op : atomic_integer.IntegerOp) (hoperation : input.op = .Integer op)
    (hcas : op = .Cas ∨ op = .CasSpin) (hwidth : input.bytes = 4#u8 ∨ input.bytes = 8#u8) :
    ∃ outcome, atomic_exec.execute implementation input = .ok (.Ok outcome) ∧
      AtomicCompareExchangeContract input outcome := by
  obtain ⟨actual, hactual, hmodel⟩ := spec_returns _ _
    (rust_atomic_integer_execute_spec implementation unusedFloatSemantics input op hoperation)
  have hexpected : atomicExecutionModel unusedFloatSemantics input =
      .Ok (atomicFinishModel input (atomicCasResultModel input)) := by
    simp [atomicExecutionModel, atomicComputedModel, hoperation,
      integerModel_cas_exact input op hcas hwidth]
  refine ⟨atomicFinishModel input (atomicCasResultModel input), ?_, ?_⟩
  · simpa only [hmodel, hexpected] using hactual
  · exact ⟨rfl, rfl, rfl, atomicFinishModel_controls input (atomicCasResultModel input)⟩

theorem rust_failed_cas_numeric_frame {F Input : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (input : atomic_exec.AtomicInput)
    (hoperation : input.op = .Integer .Cas) (hwidth : input.bytes = 4#u8 ∨ input.bytes = 8#u8)
    (hcompare : integerNormalize input.bytes input.old ≠ integerNormalize input.bytes input.compare)
    (sem : Semantics Input) (memory : Memory Input) (before : Storage)
    (state : engine.Engine) (key : Key) (hcells : state.cells.slice = before.toCells)
    (hcapacity : state.track_numeric = true → key.space = 1#u8 →
      state.numeric_written.val.length < Usize.max)
    (hstate : Storage.Represents sem memory before) :
    ∃ outcome next,
      atomic_exec.execute implementation input = .ok (.Ok outcome) ∧
      AtomicCompareExchangeContract input outcome ∧ outcome.did_write = false ∧
      numeric.forget_numeric state key = .ok next ∧ RustNumericRefines before state key next ∧
      Represents sem memory next.cells.slice ∧
      next.cells.slice.val.map numericByteView = state.cells.slice.val.map numericByteView ∧
      next.half_written = state.half_written := by
  obtain ⟨outcome, hactual, hcontract⟩ := rust_atomic_compare_exchange_progress implementation
    input .Cas hoperation (Or.inl rfl) hwidth
  have hswap : outcome.swapped = false := by simpa [hcompare] using hcontract.swapped
  have hwrite : outcome.did_write = false := by
    simpa [hoperation, atomicIsCas, hswap] using hcontract.controls.writes
  obtain ⟨next, hnext, hrefines, hrep, _, hhalf⟩ :=
    rust_forget_numeric_preserves_representation sem memory before state key hcells hcapacity hstate
  exact ⟨outcome, next, hactual, hcontract, hwrite, hnext, hrefines, hrep,
    hrefines.byte_frame before state next key hcells, hhalf⟩

end MemoryWrite
