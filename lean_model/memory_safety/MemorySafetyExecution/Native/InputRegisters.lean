/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputWriterShape
import ValueFlow.SourceDependencyMemory

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

private theorem related_render_total (left right : V.Value) (layouts : Slice value_flow.AllocationLayout)
    (tainted : Bool) (bits : U64)
    (related : ValuesRelated tainted (valueModel left) (valueModel right))
    (computed : value_flow.render_value left layouts = .ok (.Ok bits)) :
    ∃ other, value_flow.render_value right layouts = .ok (.Ok other) := by
  cases tainted with
  | false =>
    have values : valueModel left = valueModel right := by simpa only [ValuesRelated, Bool.false_eq_true, ite_false] using related
    cases left with
    | Scalar a =>
      cases right with
      | Scalar b => exact ⟨b, rfl⟩
      | Pointer b => cases values
    | Pointer a =>
      cases right with
      | Scalar b => cases values
      | Pointer b =>
        have pointers : a = b := by
          have mapped : pointerModel a = pointerModel b := by simpa only [valueModel, Value.pointer.injEq] using values
          have allocation := UScalar.eq_of_val_eq (congrArg ValueFlow.Pointer.allocation mapped)
          have offset := UScalar.eq_of_val_eq (congrArg ValueFlow.Pointer.offset mapped)
          cases a; cases b; simp_all
        subst b
        exact ⟨bits, computed⟩
  | true =>
    obtain ⟨a, b, _, rightValue⟩ := related
    cases right with
    | Scalar b => exact ⟨b, rfl⟩
    | Pointer b => cases rightValue

theorem source_virtual_write_inputs (dependencies : DependencyRegisters) (left right leftAfter : V.Registers)
    (index : U16) (shadowIndex : Usize) (metadata : Metadata) (leftValue rightValue : V.Value)
    (layouts : Slice value_flow.AllocationLayout)
    (related : RegistersInputsRelated dependencies left right)
    (shadowIndexEq : shadowIndex.val = index.val)
    (defined : metadata.defined = true)
    (values : ValuesRelated metadata.tainted (valueModel leftValue) (valueModel rightValue))
    (written : value_flow.write_value left index leftValue layouts = .ok (.Ok (), leftAfter)) :
    ∃ rightAfter, value_flow.write_value right index rightValue layouts = .ok (.Ok (), rightAfter) ∧
      RegistersInputsRelated { dependencies with virtuals := dependencies.virtuals.set shadowIndex metadata }
        leftAfter rightAfter := by
  obtain ⟨components⟩ := source_write_value_components left leftAfter index leftValue layouts written
  have rightBounds : index.val < right.raw.regs.val.length ∧ index.val < right.pointers.val.length ∧
      index.val / 64 < right.raw.written.val.length := by
    simpa only [←related.shape.virtualLength, ←related.shape.pointers, ←related.shape.written] using components.bounds
  have rightCanonical : ValueFlow.Storage.MachineRegistersVirtualMap.Canonical right.raw.vund.val := by
    simpa only [related.shape.undefined] using related.shape.canonical
  obtain ⟨bits, rendered⟩ := related_render_total leftValue rightValue layouts metadata.tainted components.bits values components.rendered
  obtain ⟨rightAfter, rightWritten⟩ := source_write_value_total right index rightValue layouts bits rightCanonical rightBounds rendered
  refine ⟨rightAfter, rightWritten, ?_⟩
  have shape := input_virtual_write_shape left right leftAfter rightAfter index leftValue rightValue layouts
    metadata.tainted related.shape written rightWritten values
  have shadowInside : shadowIndex.val < dependencies.virtuals.val.length := by
    have metadataLength := related.virtualMetadataLength
    have bounded := components.bounds.1
    omega
  have wordsLength : leftAfter.raw.regs.val.length = left.raw.regs.val.length := by
    have rawSpec := ValueFlow.Storage.MachineRegistersVirtual.source_write_virtual left.raw index components.bits 0#u64
      components.bounds.1 components.bounds.2.2
      (canonical_room left.raw.vund.val related.shape.canonical)
    rw [components.written] at rawSpec
    have words : components.raw.regs.val = left.raw.regs.val.set index.val components.bits := by
      have meaning := rawSpec
      simp only [WP.spec_ok] at meaning
      exact meaning.1
    simp only [components.afterEq, words, List.length_set]
  have predicatesLength : leftAfter.raw.preds = left.raw.preds := by
    have effect := source_write_value_success left leftAfter index leftValue layouts related.shape.canonical written
    obtain ⟨bits, rendered, effect⟩ := effect
    have rawSpec := ValueFlow.Storage.MachineRegistersVirtual.source_write_virtual left.raw index components.bits 0#u64
      components.bounds.1 components.bounds.2.2 (canonical_room left.raw.vund.val related.shape.canonical)
    rw [components.written] at rawSpec
    simp only [WP.spec_ok] at rawSpec
    simpa only [components.afterEq] using rawSpec.2.2.1
  refine ⟨shape, ?_, ?_, ?_⟩
  · simpa only [alloc.vec.Vec.set_val_eq, List.length_set, wordsLength] using related.virtualMetadataLength
  · simpa only [predicatesLength] using related.predicateMetadataLength
  · rw [dependencies_model_virtual_set dependencies shadowIndex metadata shadowInside defined]
    change InputBanksRelated (virtualDependencies (dependenciesModel dependencies) shadowIndex.val metadata.tainted) _ _
    rw [shadowIndexEq]
    exact source_value_writes_preserve_inputs _ left right leftAfter rightAfter index leftValue rightValue
      layouts layouts metadata.tainted related.banks related.shape.canonical rightCanonical written rightWritten values

theorem source_predicate_write_inputs (dependencies : DependencyRegisters) (left right leftAfter : V.Registers)
    (index : U16) (shadowIndex : Usize) (metadata : Metadata) (leftValue rightValue : Bool)
    (related : RegistersInputsRelated dependencies left right)
    (shadowIndexEq : shadowIndex.val = index.val)
    (defined : metadata.defined = true)
    (values : metadata.tainted = false → leftValue = rightValue)
    (written : value_flow.write_predicate left index leftValue = .ok (.Ok (), leftAfter)) :
    ∃ rightAfter, value_flow.write_predicate right index rightValue = .ok (.Ok (), rightAfter) ∧
      RegistersInputsRelated { dependencies with predicates := dependencies.predicates.set shadowIndex metadata }
        leftAfter rightAfter := by
  obtain ⟨bounds, raw, rawWritten, afterEq⟩ := source_write_predicate_components left leftAfter index leftValue written
  have rightBounds : index.val / 64 < right.raw.preds.val.length ∧ index.val / 64 < right.raw.pdef.val.length := by
    simpa only [←related.shape.predicateLength, ←related.shape.definitions] using bounds
  obtain ⟨rightAfter, rightWritten⟩ := source_write_predicate_total right index rightValue rightBounds
  have rawSpec := ValueFlow.Storage.MachineRegisters.source_write_predicate left.raw index leftValue true bounds.1 bounds.2
  rw [rawWritten] at rawSpec
  simp only [WP.spec_ok] at rawSpec
  have wordsLength : leftAfter.raw.regs = left.raw.regs := by simpa only [afterEq] using rawSpec.1
  have predicatesLength : leftAfter.raw.preds.val.length = left.raw.preds.val.length := by
    simp only [afterEq, rawSpec.2.2.2.1, List.length_set]
  have shadowInside : shadowIndex.val < dependencies.predicates.val.length := by
    have length := related.predicateMetadataLength
    have bounded := bounds.1
    omega
  refine ⟨rightAfter, rightWritten, input_predicate_write_shape left right leftAfter rightAfter index
    leftValue rightValue related.shape written rightWritten, ?_, ?_, ?_⟩
  · simpa only [wordsLength] using related.virtualMetadataLength
  · simpa only [alloc.vec.Vec.set_val_eq, List.length_set, predicatesLength] using related.predicateMetadataLength
  · rw [dependencies_model_predicate_set dependencies shadowIndex metadata shadowInside defined]
    change InputBanksRelated (predicateDependencies (dependenciesModel dependencies) shadowIndex.val metadata.tainted) _ _
    rw [shadowIndexEq]
    exact source_predicate_writes_preserve_inputs _ left right leftAfter rightAfter index leftValue rightValue
      metadata.tainted related.banks written rightWritten values

end MemorySafetyExecution.Native
