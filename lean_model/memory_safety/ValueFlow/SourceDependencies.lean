/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceEvaluate
import ValueFlow.SourceDependencyModel
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

@[step]
theorem source_dependency_metadata (tainted defined : Bool) :
    value_flow_dependencies.metadata tainted defined
      ⦃ result => result = ⟨0#u32, none, tainted, defined⟩ ⦄ := by
  simp [value_flow_dependencies.metadata, symbolic.prepare]

@[step]
theorem source_dependency_join (left right : Metadata) :
    value_flow_dependencies.join left right ⦃ result =>
      result.tainted = (left.tainted || right.tainted) ∧
      result.defined = (left.defined && right.defined) ⦄ := by
  unfold value_flow_dependencies.join symbolic.binary
  unfold symbolic.binary_bits
  dsimp only
  cases left.tainted <;> cases left.defined <;> simp

theorem metadata_dependency_found (values : List Metadata) (index : Nat) (metadata : Metadata)
    (found : values[index]? = some metadata) (defined : metadata.defined = true) :
    metadataDependency values index = some metadata.tainted := by
  simp only [metadataDependency, found, Option.bind_eq_bind, Option.bind_some, defined, ite_true]

@[step]
theorem source_dependency_read (registers : DependencyRegisters) (operand : V.Operand) :
    value_flow_dependencies.read registers operand ⦃ result => ∀ metadata,
      result = .Ok metadata → metadata.defined = true ∧
      operandDependency (dependenciesModel registers) (operandModel operand) = some metadata.tainted ⦄ := by
  unfold value_flow_dependencies.read
  cases operand with
  | Immediate value =>
      step as ⟨known, knownEq⟩
      subst known
      simp [operandModel, operandDependency]
  | Pointer pointer =>
      step as ⟨known, knownEq⟩
      subst known
      simp [operandModel, operandDependency]
  | Special special =>
      step as ⟨known, knownEq⟩
      subst known
      simp [operandModel, operandDependency]
  | Virtual index =>
      step with UScalar.cast_inBounds_spec .Usize index (by scalar_tac) as ⟨converted, convertedEq⟩
      split
      next outside => simp
      next inside =>
        step with UScalar.cast_inBounds_spec .Usize index (by scalar_tac) as ⟨convertedAgain, againEq⟩
        step as ⟨value, valueAt⟩
        split
        next defined =>
          simp only [WP.spec_ok, core.result.Result.Ok.injEq]
          intro metadata same
          subst metadata
          refine ⟨defined, ?_⟩
          simp only [operandModel, operandDependency, dependenciesModel]
          apply metadata_dependency_found
          · rw [List.getElem?_eq_getElem (by scalar_tac)]
            exact congrArg some (by simpa only [againEq] using valueAt.symm)
          · exact defined
        next undefined => simp

@[step]
theorem source_dependency_guard (registers : DependencyRegisters) (instruction : V.GuardedInstruction) :
    value_flow_dependencies.guard registers instruction ⦃ result => result = .Ok () →
      ∀ index, (guardedModel instruction).guard.predicate = some index →
        (dependenciesModel registers).predicate index = some false ⦄ := by
  unfold value_flow_dependencies.guard
  cases predicate : instruction.guard.predicate with
  | none =>
      step as ⟨known, knownEq⟩
      subst known
      simp [symbolic.control, guardedModel, guardModel, predicate]
  | some index =>
      step with UScalar.cast_inBounds_spec .Usize index (by scalar_tac) as ⟨converted, convertedEq⟩
      split
      next outside => simp
      next inside =>
        step with UScalar.cast_inBounds_spec .Usize index (by scalar_tac) as ⟨convertedAgain, againEq⟩
        step as ⟨value, valueAt⟩
        split
        next defined =>
          step as ⟨known, knownEq⟩
          subst known
          unfold symbolic.control
          simp only [Bool.false_eq_true, ite_false, bind_ok]
          split
          next tainted => simp
          next untainted =>
            simp only [WP.spec_ok]
            intro accepted otherIndex same
            have indexEq : otherIndex = index.val := by
              simpa only [guardedModel, guardModel, predicate, Option.map_some, Option.some.injEq] using same.symm
            subst otherIndex
            have notTainted : value.tainted = false := Bool.eq_false_iff.mpr untainted
            simpa only [dependenciesModel, notTainted] using
              metadata_dependency_found registers.predicates.val index.val value
                (by rw [List.getElem?_eq_getElem (by scalar_tac)]
                    exact congrArg some (by simpa only [againEq] using valueAt.symm)) defined
        next undefined => simp

@[step]
theorem source_dependency_pair (registers : DependencyRegisters) (left right : V.Operand) :
    value_flow_dependencies.pair registers left right ⦃ result => ∀ metadata,
      result = .Ok metadata → metadata.defined = true ∧
      PairTracked (dependenciesModel registers) (operandModel left) (operandModel right) metadata.tainted ⦄ := by
  unfold value_flow_dependencies.pair
  step as ⟨leftResult, leftMeaning⟩
  cases leftResult with
  | Err error => simp
  | Ok a =>
      have aMeaning := leftMeaning a rfl
      step as ⟨rightResult, rightMeaning⟩
      cases rightResult with
      | Err error => simp
      | Ok b =>
          have bMeaning := rightMeaning b rfl
          step as ⟨result, resultTainted, resultDefined⟩
          simp only [WP.spec_ok, core.result.Result.Ok.injEq]
          intro metadata same
          subst metadata
          refine ⟨?_, a.tainted, b.tainted, aMeaning.2, bMeaning.2, resultTainted⟩
          simp only [resultDefined, aMeaning.1, bMeaning.1, Bool.true_and]

@[step]
theorem source_dependency_address (registers : DependencyRegisters) (location : V.Address) :
    value_flow_dependencies.address registers location ⦃ result => result = .Ok () →
      AddressTracked (dependenciesModel registers) (addressModel location) ⦄ := by
  unfold value_flow_dependencies.address
  step as ⟨baseResult, baseMeaning⟩
  cases baseResult with
  | Err error => simp
  | Ok base =>
      have baseRead := baseMeaning base rfl
      step as ⟨indexResult, indexMeaning⟩
      cases indexResult with
      | Err error => simp
      | Ok index =>
          have indexRead := indexMeaning index rfl
          step as ⟨known, knownEq⟩
          subst known
          unfold symbolic.control
          cases baseTaint : base.tainted <;> cases indexTaint : index.tainted <;>
            simp_all [AddressTracked, addressModel]

def MetadataPrefix (loaded : Slice Metadata) (index : Nat) (field : Metadata → Bool) (value : Bool) : Prop :=
  ∀ position metadata, position < index → loaded.val[position]? = some metadata → field metadata = value

theorem metadata_prefix_extend (loaded : Slice Metadata) (index : Nat) (metadata : Metadata)
    (field : Metadata → Bool) (value : Bool)
    (before : MetadataPrefix loaded index field value)
    (selected : loaded.val[index]? = some metadata) (agrees : field metadata = value) :
    MetadataPrefix loaded (index + 1) field value := by
  intro position other bound found
  by_cases previous : position < index
  · exact before position other previous found
  · have equal : position = index := by omega
    subst position
    rw [selected] at found
    cases found
    exact agrees

@[step]
theorem source_dependency_loaded_loop (loaded : Slice Metadata) :
    value_flow_dependencies.loaded_metadata_loop loaded ⟨0#u32, none, false, true⟩ 0#usize
      ⦃ result =>
        (result.defined = true → MetadataPrefix loaded loaded.val.length (·.defined) true) ∧
        (result.tainted = false → MetadataPrefix loaded loaded.val.length (·.tainted) false) ⦄ := by
  unfold value_flow_dependencies.loaded_metadata_loop
  apply loop.spec_decr_nat
    (fun current : Metadata × Usize => loaded.val.length - current.2.val)
    (fun current : Metadata × Usize => current.2.val ≤ loaded.val.length ∧
      (current.1.defined = true → MetadataPrefix loaded current.2.val (·.defined) true) ∧
      (current.1.tainted = false → MetadataPrefix loaded current.2.val (·.tainted) false))
  · intro current invariant
    rcases current with ⟨result, cursor⟩
    obtain ⟨cursorBound, definedPrefix, taintedPrefix⟩ := invariant
    dsimp only at cursorBound definedPrefix taintedPrefix
    unfold value_flow_dependencies.loaded_metadata_loop.body
    dsimp only
    by_cases inside : cursor.val < loaded.val.length
    · have sourceInside : cursor < Slice.len loaded := by scalar_tac
      simp only [sourceInside, ite_true]
      step as ⟨selected, selectedAt⟩
      have selectedFound : loaded.val[cursor.val]? = some selected := by
        rw [List.getElem?_eq_getElem inside]
        exact congrArg some selectedAt.symm
      step as ⟨joined, joinedTaint, joinedDefined⟩
      step as ⟨next, nextEq⟩
      refine ⟨by scalar_tac, ?_, ?_, by scalar_tac⟩
      · intro defined
        have both : result.defined = true ∧ selected.defined = true := by
          rw [joinedDefined] at defined
          simpa only [Bool.and_eq_true] using defined
        have extended := metadata_prefix_extend loaded cursor.val selected (·.defined) true
          (definedPrefix both.1) selectedFound both.2
        simpa only [nextEq] using extended
      · intro untainted
        have neither : result.tainted = false ∧ selected.tainted = false := by
          rw [joinedTaint] at untainted
          simpa only [Bool.or_eq_false_iff] using untainted
        have extended := metadata_prefix_extend loaded cursor.val selected (·.tainted) false
          (taintedPrefix neither.1) selectedFound neither.2
        simpa only [nextEq] using extended
    · have sourceOutside : ¬cursor < Slice.len loaded := by scalar_tac
      have finished : cursor.val = loaded.val.length := by omega
      simp only [sourceOutside, ite_false, WP.spec_ok]
      simpa only [finished] using And.intro definedPrefix taintedPrefix
  · simp [MetadataPrefix]

@[step]
theorem source_dependency_loaded (loaded : Slice Metadata) (bytes : U8) :
    value_flow_dependencies.loaded_metadata loaded bytes ⦃ result => ∀ metadata,
      result = .Ok metadata → metadata.defined = true ∧
      (metadata.tainted = false → ∀ value ∈ loaded.val, value.tainted = false) ⦄ := by
  unfold value_flow_dependencies.loaded_metadata
  step with UScalar.cast_inBounds_spec .Usize bytes (by scalar_tac) as ⟨width, widthEq⟩
  split
  next wrongShape => simp
  next shape =>
    step
    split
    next empty => simp
    next nonempty =>
      step as ⟨initial, initialEq⟩
      subst initial
      step as ⟨result, resultDefined, resultTaint⟩
      split
      next defined =>
        simp only [WP.spec_ok, core.result.Result.Ok.injEq]
        intro metadata equal
        subst metadata
        refine ⟨defined, ?_⟩
        intro untainted value member
        obtain ⟨index, inside, found⟩ := List.mem_iff_getElem.mp member
        exact resultTaint untainted index value inside
          (by rw [List.getElem?_eq_getElem inside]; exact congrArg some found)
      next undefined => simp

def TransferPost (registers : DependencyRegisters) (instruction : V.GuardedInstruction)
    (action : V.Action) (result : core.result.Result DependencyTransfer value_flow_dependencies.DependencyError) : Prop :=
  ∀ transferred, result = .Ok transferred →
    TransferCondition (dependenciesModel registers) (guardedModel instruction) (actionModel action)
      (transferTaint transferred) ∧ TransferDefined transferred ∧ TransferMatches action transferred

theorem source_virtual_finish (registers : DependencyRegisters) (instruction : V.GuardedInstruction)
    (destination : U16) (value : V.Value) (metadata : Metadata)
    (guardTracked : ∀ index, (guardedModel instruction).guard.predicate = some index →
      (dependenciesModel registers).predicate index = some false)
    (defined : metadata.defined = true)
    (body : PointerIndependent metadata.tainted (valueModel value) →
      InstructionTransfer (dependenciesModel registers) (instructionModel instruction.instruction)
        (.writeVirtual destination.val (valueModel value)) metadata.tainted) :
    (if metadata.tainted then do
      let isPointer ← (match value with | .Scalar _ => .ok false | .Pointer _ => .ok true : Result Bool)
      if isPointer then .ok (.Err value_flow_dependencies.DependencyError.Address)
      else .ok (.Ok (.Virtual destination metadata))
    else .ok (.Ok (.Virtual destination metadata)) :
      Result (core.result.Result DependencyTransfer value_flow_dependencies.DependencyError))
      ⦃ result => TransferPost registers instruction (.WriteVirtual destination value) result ⦄ := by
  by_cases tainted : metadata.tainted = true
  · cases value with
    | Scalar bits =>
        simp only [tainted, ite_true, bind_ok, Bool.false_eq_true, ite_false, WP.spec_ok,
          TransferPost, core.result.Result.Ok.injEq]
        intro transferred same
        subst transferred
        refine ⟨⟨guardTracked, ?_⟩, defined, rfl⟩
        apply body
        exact Or.inr ⟨bits.val, rfl⟩
    | Pointer pointer => simp [tainted, TransferPost]
  · have untainted : metadata.tainted = false := Bool.eq_false_iff.mpr tainted
    simp only [untainted, Bool.false_eq_true, ite_false, WP.spec_ok, TransferPost,
      core.result.Result.Ok.injEq]
    intro transferred same
    subst transferred
    refine ⟨⟨guardTracked, ?_⟩, defined, rfl⟩
    exact body (Or.inl untainted)

@[step]
theorem source_dependency_transfer (instruction : V.GuardedInstruction) (action : V.Action)
    (registers : DependencyRegisters) (loaded : Slice Metadata) :
    value_flow_dependencies.transfer instruction action registers loaded
      ⦃ result => TransferPost registers instruction action result ⦄ := by
  unfold value_flow_dependencies.transfer
  step as ⟨guardResult, guardMeaning⟩
  cases guardResult with
  | Err error => simp [TransferPost]
  | Ok unit =>
      have guardTracked := guardMeaning rfl
      cases action with
      | Next =>
          simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq]
          intro transferred same
          subst transferred
          exact ⟨⟨guardTracked, .next _⟩, trivial, trivial⟩
      | Jump target =>
          cases sourceInstruction : instruction.instruction <;>
            simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq, reduceCtorEq,
              false_implies, implies_true]
          intro transferred same
          subst transferred
          exact ⟨⟨guardTracked, by simpa only [guardedModel, sourceInstruction, instructionModel, actionModel, transferTaint]
            using InstructionTransfer.jump (dependencies := dependenciesModel registers) _ _⟩, trivial, trivial⟩
      | Halt =>
          cases sourceInstruction : instruction.instruction <;>
            simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq, reduceCtorEq,
              false_implies, implies_true]
          intro transferred same
          subst transferred
          exact ⟨⟨guardTracked, by simpa only [guardedModel, sourceInstruction, instructionModel, actionModel, transferTaint]
            using InstructionTransfer.halt (dependencies := dependenciesModel registers)⟩, trivial, trivial⟩
      | WriteVirtual destination value =>
          cases sourceInstruction : instruction.instruction <;>
            simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq, reduceCtorEq,
              false_implies, implies_true]
          all_goals
            step as ⟨sourceResult, sourceMeaning⟩
            cases sourceResult with
            | Err error => simp [TransferPost]
            | Ok metadata =>
              have meaning := sourceMeaning metadata rfl
              have body : PointerIndependent metadata.tainted (valueModel value) →
                  InstructionTransfer (dependenciesModel registers) (instructionModel instruction.instruction)
                    (.writeVirtual destination.val (valueModel value)) metadata.tainted := by
                intro pointerIndependent
                first
                | simpa only [instructionModel, sourceInstruction] using
                    InstructionTransfer.move (dependencies := dependenciesModel registers) _ _ _ _ _ _ meaning.2 pointerIndependent
                | simpa only [instructionModel, sourceInstruction] using
                    InstructionTransfer.binary (dependencies := dependenciesModel registers) _ _ _ _ _ _ _ _ meaning.2 pointerIndependent
                | simpa only [instructionModel, sourceInstruction] using
                    InstructionTransfer.multiplyWide (dependencies := dependenciesModel registers) _ _ _ _ _ _ meaning.2 pointerIndependent
                | simpa only [instructionModel, sourceInstruction] using
                    InstructionTransfer.convert (dependencies := dependenciesModel registers) _ _ _ _ _ _ _ meaning.2 pointerIndependent
              cases value <;> simpa only [TransferPost] using
                source_virtual_finish registers instruction destination _ metadata guardTracked meaning.1 body
      | WritePredicate destination value =>
          cases sourceInstruction : instruction.instruction <;>
            simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq, reduceCtorEq,
              false_implies, implies_true]
          step as ⟨sourceResult, sourceMeaning⟩
          cases sourceResult with
          | Err error => simp [TransferPost]
          | Ok metadata =>
              have meaning := sourceMeaning metadata rfl
              simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq]
              intro transferred same
              subst transferred
              refine ⟨⟨guardTracked, ?_⟩, meaning.1, rfl⟩
              simpa only [guardedModel, sourceInstruction, instructionModel, actionModel, transferTaint] using
                InstructionTransfer.compare (dependencies := dependenciesModel registers) _ _ _ _ _ _ _ _ meaning.2
      | Load destination bits bytes signed pointer =>
          cases sourceInstruction : instruction.instruction <;>
            simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq, reduceCtorEq,
              false_implies, implies_true]
          step as ⟨addressResult, addressMeaning⟩
          cases addressResult with
          | Err error => simp [TransferPost]
          | Ok unit =>
              have addressTracked := addressMeaning rfl
              step as ⟨metadataResult, metadataMeaning⟩
              cases metadataResult with
              | Err error => simp [TransferPost]
              | Ok metadata =>
                  have meaning := metadataMeaning metadata rfl
                  simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq]
                  intro transferred same
                  subst transferred
                  refine ⟨⟨guardTracked, ?_⟩, meaning.1, rfl⟩
                  simpa only [guardedModel, sourceInstruction, instructionModel, actionModel, transferTaint] using
                    InstructionTransfer.load (dependencies := dependenciesModel registers) _ _ _ _ _ _ _ _ _ _ _ addressTracked
      | Store bytes pointer value =>
          cases sourceInstruction : instruction.instruction <;>
            simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq, reduceCtorEq,
              false_implies, implies_true]
          step as ⟨addressResult, addressMeaning⟩
          cases addressResult with
          | Err error => simp [TransferPost]
          | Ok unit =>
              have addressTracked := addressMeaning rfl
              step as ⟨metadataResult, metadataMeaning⟩
              cases metadataResult with
              | Err error => simp [TransferPost]
              | Ok metadata =>
                  have meaning := metadataMeaning metadata rfl
                  simp only [WP.spec_ok, TransferPost, core.result.Result.Ok.injEq]
                  intro transferred same
                  subst transferred
                  refine ⟨⟨guardTracked, ?_⟩, meaning.1, trivial⟩
                  simpa only [guardedModel, sourceInstruction, instructionModel, actionModel, transferTaint] using
                    InstructionTransfer.store (dependencies := dependenciesModel registers) _ _ _ _ _ _ _ addressTracked meaning.2

theorem source_transfer_success (instruction : V.GuardedInstruction) (action : V.Action)
    (registers : DependencyRegisters) (loaded : Slice Metadata) (transferred : DependencyTransfer)
    (computed : value_flow_dependencies.transfer instruction action registers loaded = .ok (.Ok transferred)) :
    TransferCondition (dependenciesModel registers) (guardedModel instruction) (actionModel action)
      (transferTaint transferred) ∧ TransferDefined transferred ∧ TransferMatches action transferred := by
  have specification := source_dependency_transfer instruction action registers loaded
  rw [computed] at specification
  have meaning : TransferPost registers instruction action (.Ok transferred) := by
    simpa only [WP.spec_ok] using specification
  exact meaning transferred rfl

theorem source_evaluate_inputs (left right : V.Registers) (ids : V.ThreadIds)
    (instruction : V.GuardedInstruction) (action : V.Action)
    (metadata : DependencyRegisters) (loaded : Slice Metadata) (transferred : DependencyTransfer)
    (related : InputBanksRelated (dependenciesModel metadata) (bankModel left) (bankModel right))
    (transferComputed : value_flow_dependencies.transfer instruction action metadata loaded = .ok (.Ok transferred))
    (computed : value_flow.evaluate left ids instruction = .ok (.Ok action)) :
    ∃ other, value_flow.evaluate right ids instruction = .ok (.Ok other) ∧
      ActionsRelated (transferTaint transferred) (actionModel action) (actionModel other) := by
  have checked := source_transfer_success instruction action metadata loaded transferred transferComputed
  have capturedMeaning := source_evaluate left ids instruction
  rw [computed] at capturedMeaning
  have pureComputed : evaluate (bankModel left) (idsModel ids) (guardedModel instruction) = some (actionModel action) := by
    have meaning : some (actionModel action) =
        evaluate (bankModel left) (idsModel ids) (guardedModel instruction) := by
      simpa only [WP.spec_ok, outcomeModel] using capturedMeaning
    exact meaning.symm
  obtain ⟨otherAction, otherComputed, actions⟩ :=
    evaluate_inputs (dependenciesModel metadata) (bankModel left) (bankModel right) related (idsModel ids)
      (guardedModel instruction) (actionModel action) (transferTaint transferred) checked.1 pureComputed
  obtain ⟨otherResult, otherCall, otherMeaning⟩ :=
    MemoryWrite.spec_returns _ _ (source_evaluate right ids instruction)
  rw [otherComputed] at otherMeaning
  cases otherResult with
  | Err error => simp [outcomeModel] at otherMeaning
  | Ok other =>
      simp only [outcomeModel, Option.some.injEq] at otherMeaning
      exact ⟨other, otherCall, otherMeaning ▸ actions⟩

end ValueFlow.Source
