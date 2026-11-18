/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceBufferExtent
import FrontendProgram.GlobalReferences
import FrontendProgram.SourceLoop

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.isDefEq.respectTransparency false
set_option backward.split false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

def GlobalNamesInjective (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram) : Prop :=
  ∀ (i j : Nat) (left right : Global), parsed.scope.globals.val[i]? = some left →
    parsed.scope.globals.val[j]? = some right → nameBytes source left.name = nameBytes source right.name → i = j

theorem token_text_bytes (source : List U8) (token : SourceToken) (expected : List U8)
    (meaning : TokenText source token.start.val token.end.val expected) :
    nameBytes source token = expected := by
  obtain ⟨ordered, bound, length, points⟩ := meaning
  apply List.ext_getElem?
  intro index
  by_cases inside : index < expected.length
  · simpa only [nameBytes, List.getElem?_take, length, inside, ite_true, List.getElem?_drop] using points index inside
  · simp only [nameBytes, List.getElem?_take, length, inside, ite_false, List.getElem?_eq_none (by omega : expected.length ≤ index)]

private def GlobalFound (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (symbol : Slice U8) (found : Usize) : Prop :=
  found ≠ core.num.Usize.MAX → ∃ global, parsed.scope.globals.val[found.val]? = some global ∧
    nameBytes source.val global.name = symbol.val

private theorem global_named_loop (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (symbol : Slice U8) (found index : Usize) (unique : Bool) (output : Usize × Bool)
    (bound : index.val ≤ parsed.scope.globals.val.length)
    (prior : GlobalFound source parsed symbol found)
    (computed : strict_memory_frontend_launch.global_named_loop source parsed symbol found unique index = .ok output) :
    output.2 = true → GlobalFound source parsed symbol output.1 := by
  unfold strict_memory_frontend_launch.global_named_loop at computed
  apply source_loop_success
    (fun (current : Usize × Bool × Usize) => strict_memory_frontend_launch.global_named_loop.body source parsed symbol current.1 current.2.1 current.2.2)
    (fun current => parsed.scope.globals.val.length - current.2.2.val)
    (fun current => current.2.2.val ≤ parsed.scope.globals.val.length ∧
      (current.2.1 = true → GlobalFound source parsed symbol current.1))
    (fun result => result.2 = true → GlobalFound source parsed symbol result.1)
    ?_ (found, unique, index) output ⟨bound, fun _ => prior⟩ computed
  rintro ⟨currentFound, currentUnique, cursor⟩ flow ⟨cursorBound, invariant⟩ bodyRun
  cases currentUnique with
  | false =>
    simp only [strict_memory_frontend_launch.global_named_loop.body, Bool.false_eq_true, ite_false, ok.injEq] at bodyRun
    subst flow
    simp
  | true =>
    by_cases inside : cursor.val < parsed.scope.globals.val.length
    · have test : cursor < alloc.vec.Vec.len parsed.scope.globals := by scalar_tac
      let global := parsed.scope.globals.val[cursor.val]
      have read : alloc.vec.Vec.index_usize parsed.scope.globals cursor = .ok global := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside, global]
      have incrementBound : cursor.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq parsed.scope.globals
        scalar_tac
      let after := Usize.ofNatCore (cursor.val + 1) incrementBound
      have afterVal : after.val = cursor.val + 1 := by simp [after]
      have increment := source_usize_increment cursor incrementBound
      simp only [strict_memory_frontend_launch.global_named_loop.body, ite_true, test,
        alloc.vec.Vec.index_slice_index, read, bind_ok] at bodyRun
      obtain ⟨same, sameRun, remainder⟩ := source_bind_success _ _ _ bodyRun
      cases same with
      | false =>
        simp only [Bool.false_eq_true, ite_false, bind_ok, increment, ok.injEq] at remainder
        change Result.ok (.cont (currentFound, true, after)) = .ok flow at remainder
        have flowEq := (by simpa only [ok.injEq] using remainder : _ = flow)
        subst flow
        exact ⟨⟨by scalar_tac, invariant⟩, by scalar_tac⟩
      | true =>
        simp only [ite_true, bind_ok, increment, ok.injEq] at remainder
        change Result.ok (.cont (cursor, decide (currentFound = core.num.Usize.MAX), after)) = .ok flow at remainder
        have flowEq := (by simpa only [ok.injEq] using remainder : _ = flow)
        subst flow
        refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
        intro accepted different
        exact ⟨global, List.getElem?_eq_getElem inside,
          token_text_bytes source.val global.name symbol.val (source_token_equals source symbol global.name sameRun)⟩
    · have test : ¬cursor < alloc.vec.Vec.len parsed.scope.globals := by scalar_tac
      simp only [strict_memory_frontend_launch.global_named_loop.body, ite_true, test, ite_false, ok.injEq] at bodyRun
      subst flow
      exact invariant

theorem source_global_named (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (symbol : Slice U8) (object : Usize)
    (computed : strict_memory_frontend_launch.global_named source parsed symbol = .ok (some object)) :
    ∃ global, parsed.scope.globals.val[object.val]? = some global ∧ nameBytes source.val global.name = symbol.val := by
  unfold strict_memory_frontend_launch.global_named at computed
  obtain ⟨result, loopRun, remainder⟩ := source_bind_success _ _ _ computed
  obtain ⟨found, unique⟩ := result
  cases unique with
  | false => simp at remainder
  | true =>
    change (if (found != core.num.Usize.MAX) = true then Result.ok (some found) else .ok none) = .ok (some object) at remainder
    simp only [bne_iff_ne] at remainder
    by_cases different : found ≠ core.num.Usize.MAX
    · have equal : found = object := by simpa [different] using remainder
      subst found
      exact global_named_loop source parsed symbol core.num.Usize.MAX 0#usize true (object, true)
        (by simp) (by intro impossible; exact False.elim (impossible rfl)) loopRun rfl different
    · simp only [bne_iff_ne, different, ite_false] at remainder
      simp at remainder

theorem source_buffer_object (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (buffer : Buffer) (object : Usize)
    (computed : strict_memory_frontend_launch.buffer_object source parsed buffer = .ok (some object)) :
    BacksObject source.val parsed object.val buffer := by
  unfold strict_memory_frontend_launch.buffer_object at computed
  cases symbolEq : buffer.symbol with
  | none => simp only [symbolEq] at computed; simp at computed
  | some symbol =>
    simp only [symbolEq] at computed
    obtain ⟨bytes, bytesRun, namedRun⟩ := source_bind_success _ _ _ computed
    obtain ⟨global, globalAt, same⟩ := source_global_named source parsed bytes object namedRun
    exact ⟨global, symbol, globalAt, symbolEq, same.trans (source_string_bytes symbol bytes bytesRun)⟩


theorem backs_object_unique (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (names : GlobalNamesInjective source parsed) (buffer : Buffer) (left right : Nat)
    (leftBacking : BacksObject source parsed left buffer) (rightBacking : BacksObject source parsed right buffer) :
    left = right := by
  obtain ⟨leftGlobal, leftSymbol, leftAt, leftName, leftText⟩ := leftBacking
  obtain ⟨rightGlobal, rightSymbol, rightAt, rightName, rightText⟩ := rightBacking
  have symbols : leftSymbol = rightSymbol := Option.some.inj (leftName.symm.trans rightName)
  exact names left right leftGlobal rightGlobal leftAt rightAt (leftText.trans (symbols ▸ rightText.symm))

private def BufferFound (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Nat) (limit : Nat) (found : Usize) : Prop :=
  (found ≠ core.num.Usize.MAX → ∃ buffer, found.val < limit ∧
    manifest.buffers.val[found.val]? = some buffer ∧ BacksObject source parsed object buffer) ∧
  (∀ index buffer, index < limit → manifest.buffers.val[index]? = some buffer →
    BacksObject source parsed object buffer → found ≠ core.num.Usize.MAX ∧ index = found.val)

private theorem buffer_index_loop (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Usize) (names : GlobalNamesInjective source.val parsed)
    (found index : Usize) (valid : Bool) (output : Usize × Bool)
    (bound : index.val ≤ manifest.buffers.val.length)
    (prior : valid = true → BufferFound source.val parsed manifest object.val index.val found)
    (computed : strict_memory_frontend_launch.buffer_index_loop source parsed manifest object found valid index = .ok output) :
    output.2 = true → BufferFound source.val parsed manifest object.val manifest.buffers.val.length output.1 := by
  unfold strict_memory_frontend_launch.buffer_index_loop at computed
  apply source_loop_success
    (fun (current : Usize × Bool × Usize) => strict_memory_frontend_launch.buffer_index_loop.body source parsed manifest object current.1 current.2.1 current.2.2)
    (fun current => manifest.buffers.val.length - current.2.2.val)
    (fun current => current.2.2.val ≤ manifest.buffers.val.length ∧
      (current.2.1 = true → BufferFound source.val parsed manifest object.val current.2.2.val current.1))
    (fun result => result.2 = true → BufferFound source.val parsed manifest object.val manifest.buffers.val.length result.1)
    ?_ (found, valid, index) output ⟨bound, prior⟩ computed
  rintro ⟨currentFound, currentValid, cursor⟩ flow ⟨cursorBound, invariant⟩ bodyRun
  cases currentValid with
  | false =>
    simp only [strict_memory_frontend_launch.buffer_index_loop.body, Bool.false_eq_true, ite_false, ok.injEq] at bodyRun
    subst flow
    simp
  | true =>
    have prior := invariant rfl
    change BufferFound source.val parsed manifest object.val cursor.val currentFound at prior
    by_cases inside : cursor.val < manifest.buffers.val.length
    · have test : cursor < alloc.vec.Vec.len manifest.buffers := by scalar_tac
      let buffer := manifest.buffers.val[cursor.val]
      have read : alloc.vec.Vec.index_usize manifest.buffers cursor = .ok buffer := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside, buffer]
      have incrementBound : cursor.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq manifest.buffers
        scalar_tac
      let after := Usize.ofNatCore (cursor.val + 1) incrementBound
      have afterVal : after.val = cursor.val + 1 := by simp [after]
      have increment := source_usize_increment cursor incrementBound
      simp only [strict_memory_frontend_launch.buffer_index_loop.body, ite_true, test,
        alloc.vec.Vec.index_slice_index, read, bind_ok] at bodyRun
      obtain ⟨targetOption, targetRun, remainder⟩ := source_bind_success _ _ _ bodyRun
      cases targetOption with
      | none =>
        simp only [bind_ok, increment, ok.injEq] at remainder
        change Result.ok (.cont (currentFound, false, after)) = .ok flow at remainder
        have flowEq := (by simpa only [ok.injEq] using remainder : _ = flow)
        subst flow
        exact ⟨⟨by scalar_tac, by simp⟩, by scalar_tac⟩
      | some target =>
        have targetBacking := source_buffer_object source parsed buffer target targetRun
        by_cases same : target = object
        · simp only [same, ite_true, bind_ok, increment, ok.injEq] at remainder
          change Result.ok (.cont (cursor, decide (currentFound = core.num.Usize.MAX), after)) = .ok flow at remainder
          have flowEq := (by simpa only [ok.injEq] using remainder : _ = flow)
          subst flow
          refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
          intro accepted
          change BufferFound source.val parsed manifest object.val after.val cursor
          have empty : currentFound = core.num.Usize.MAX := of_decide_eq_true accepted
          have selected : cursor ≠ core.num.Usize.MAX := by scalar_tac
          refine ⟨?_, ?_⟩
          · intro _
            exact ⟨buffer, by omega, List.getElem?_eq_getElem inside, by simpa only [same] using targetBacking⟩
          · intro query other lower atOther backs
            refine ⟨selected, ?_⟩
            by_cases earlier : query < cursor.val
            · exact False.elim ((prior.2 query other earlier atOther backs).1 empty)
            · omega
        · simp only [same, ite_false, bind_ok, increment, ok.injEq] at remainder
          change Result.ok (.cont (currentFound, true, after)) = .ok flow at remainder
          have flowEq := (by simpa only [ok.injEq] using remainder : _ = flow)
          subst flow
          refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
          intro _
          change BufferFound source.val parsed manifest object.val after.val currentFound
          refine ⟨?_, ?_⟩
          · intro selected
            obtain ⟨old, lower, oldAt, oldBacking⟩ := prior.1 selected
            exact ⟨old, by omega, oldAt, oldBacking⟩
          · intro query other lower atOther backs
            by_cases earlier : query < cursor.val
            · exact prior.2 query other earlier atOther backs
            · have current : query = cursor.val := by omega
              subst query
              have otherEq : other = buffer := by
                have atBuffer : manifest.buffers.val[cursor.val]? = some buffer := List.getElem?_eq_getElem inside
                exact Option.some.inj (atOther.symm.trans atBuffer)
              subst other
              have targetEq := backs_object_unique source.val parsed names buffer target.val object.val targetBacking backs
              exact False.elim (same (UScalar.eq_of_val_eq targetEq))
    · have test : ¬cursor < alloc.vec.Vec.len manifest.buffers := by scalar_tac
      simp only [strict_memory_frontend_launch.buffer_index_loop.body, ite_true, test, ite_false, ok.injEq] at bodyRun
      subst flow
      have complete : cursor.val = manifest.buffers.val.length := by scalar_tac
      simpa only [complete] using invariant

theorem source_buffer_index (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Usize) (selected : Option Usize)
    (names : GlobalNamesInjective source.val parsed)
    (computed : strict_memory_frontend_launch.buffer_index source parsed manifest object = .ok (some selected)) :
    (selected = none → ∀ buffer ∈ manifest.buffers.val, ¬BacksObject source.val parsed object.val buffer) ∧
    (∀ index, selected = some index → ∃ buffer, manifest.buffers.val[index.val]? = some buffer ∧
      BacksObject source.val parsed object.val buffer ∧ ∀ query other, manifest.buffers.val[query]? = some other →
        BacksObject source.val parsed object.val other → query = index.val) := by
  unfold strict_memory_frontend_launch.buffer_index at computed
  obtain ⟨result, loopRun, remainder⟩ := source_bind_success _ _ _ computed
  obtain ⟨found, valid⟩ := result
  cases valid with
  | false => simp at remainder
  | true =>
    change (if found = core.num.Usize.MAX then Result.ok (some none) else .ok (some (some found))) = .ok (some selected) at remainder
    have initial : BufferFound source.val parsed manifest object.val 0 core.num.Usize.MAX := by
      constructor
      · intro impossible; exact False.elim (impossible rfl)
      · intro index buffer impossible; omega
    have meaning := buffer_index_loop source parsed manifest object names core.num.Usize.MAX 0#usize true (found,true)
      (by simp) (by intro _; exact initial) loopRun rfl
    by_cases empty : found = core.num.Usize.MAX
    · have selectedEq : selected = none := by simpa only [empty, ite_true, ok.injEq, Option.some.injEq] using remainder.symm
      subst selected
      constructor
      · intro _ buffer member backs
        obtain ⟨index, atBuffer⟩ := List.mem_iff_getElem?.mp member
        have inside : index < manifest.buffers.val.length := List.getElem?_eq_some_iff.mp atBuffer |>.1
        exact (meaning.2 index buffer inside atBuffer backs).1 empty
      · intro index impossible; cases impossible
    · have selectedEq : selected = some found := by simpa only [empty, ite_false, ok.injEq, Option.some.injEq] using remainder.symm
      subst selected
      constructor
      · intro impossible; cases impossible
      · intro index same
        cases same
        obtain ⟨buffer, lower, atBuffer, backs⟩ := meaning.1 empty
        refine ⟨buffer, atBuffer, backs, ?_⟩
        intro query other atOther otherBacking
        have inside : query < manifest.buffers.val.length := (List.getElem?_eq_some_iff.mp atOther).1
        exact (meaning.2 query other inside atOther otherBacking).2

end PtxFrontendProgram
