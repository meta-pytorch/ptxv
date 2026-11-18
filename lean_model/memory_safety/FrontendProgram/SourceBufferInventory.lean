/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceBufferMapping

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

private theorem option_none {α : Type} (value : Option α) :
    core.option.Option.is_none value = true ↔ value = none := by
  cases value <;> simp [core.option.Option.is_none]

private theorem option_some {α : Type} (value : Option α) :
    core.option.Option.is_some value = true ↔ value ≠ none := by
  cases value <;> simp [core.option.Option.is_some]

private theorem empty_vector {α : Type} (value : alloc.vec.Vec α) :
    alloc.vec.Vec.len value = 0#usize ↔ value.val = [] := by
  constructor
  · intro empty
    cases h : value.val with
    | nil => rfl
    | cons first rest => scalar_tac
  · intro empty
    scalar_tac

private theorem compared_bytes (left right : Slice U8) (same : Bool)
    (computed : strict_memory_frontend_launch.bytes_equal left right = .ok same) :
    same = decide (left.val = right.val) := by
  have specification := source_bytes_equal left right
  rw [computed] at specification
  simpa only [WP.spec_ok] using specification

theorem source_buffer_supported (buffer : Buffer)
    (computed : strict_memory_frontend_launch.buffer_supported buffer = .ok true) :
    BufferSupported buffer := by
  unfold strict_memory_frontend_launch.buffer_supported at computed
  dsimp only at computed
  repeat' first
    | contradiction
    | simp only [lift, bind_ok, ok.injEq, Bool.false_eq_true] at computed
    | split at computed
  all_goals obtain ⟨bytes, readRun, computed⟩ := source_bind_success _ _ _ computed
  all_goals have readMeaning := source_string_bytes buffer.init bytes readRun
  all_goals try (
    try simp only [lift, bind_ok] at computed
    obtain ⟨same, sameRun, computed⟩ := source_bind_success _ _ _ computed
    have compared := compared_bytes _ _ same sameRun
    rw [readMeaning] at compared
    first
      | change same = decide (utf8 buffer.init = utf8 "zeros") at compared
      | change same = decide (utf8 buffer.init = utf8 "undef") at compared
    simp only [utf8_eq_iff] at compared
    cases same
    all_goals simp at compared
    all_goals repeat' first
      | contradiction
      | simp only [lift, bind_ok, ok.injEq, Bool.false_eq_true] at computed
      | split at computed)
  all_goals try (
    try simp only [lift, bind_ok] at computed
    obtain ⟨same, sameRun, computed⟩ := source_bind_success _ _ _ computed
    have compared := compared_bytes _ _ same sameRun
    rw [readMeaning] at compared
    first
      | change same = decide (utf8 buffer.init = utf8 "zeros") at compared
      | change same = decide (utf8 buffer.init = utf8 "undef") at compared
    simp only [utf8_eq_iff] at compared
    cases same
    all_goals simp at compared
    all_goals repeat' first
      | contradiction
      | simp only [lift, bind_ok, ok.injEq, Bool.false_eq_true] at computed
      | split at computed)
  all_goals try (
    try simp only [lift, bind_ok] at computed
    obtain ⟨same, sameRun, computed⟩ := source_bind_success _ _ _ computed
    have compared := compared_bytes _ _ same sameRun
    rw [readMeaning] at compared
    first
      | change same = decide (utf8 buffer.init = utf8 "zeros") at compared
      | change same = decide (utf8 buffer.init = utf8 "undef") at compared
    simp only [utf8_eq_iff] at compared
    cases same
    all_goals simp at compared
    all_goals repeat' first
      | contradiction
      | simp only [lift, bind_ok, ok.injEq, Bool.false_eq_true] at computed
      | split at computed)
  all_goals try (
    have finalCompare := compared_bytes _ _ true computed
    rw [readMeaning] at finalCompare
    change true = decide (utf8 buffer.init = utf8 "undef") at finalCompare
    simp only [utf8_eq_iff] at finalCompare
    have finalInit : buffer.init = "undef" := by simpa using finalCompare
    clear computed)
  all_goals simp only [option_none, option_some, empty_vector] at *
  all_goals unfold BufferSupported
  all_goals repeat' apply And.intro
  all_goals first
    | assumption
    | scalar_tac
    | (intro scratch; first | assumption | contradiction)
    | exact Or.inl (by assumption)
    | exact Or.inr (by assumption)


theorem source_buffer_extent_matches (buffer : Buffer) (bytes : U64)
    (computed : strict_memory_frontend_launch.buffer_extent_matches buffer bytes = .ok true) :
    BufferExtent buffer bytes.val := by
  unfold strict_memory_frontend_launch.buffer_extent_matches at computed
  obtain ⟨extent, extentRun, remainder⟩ := source_bind_success _ _ _ computed
  cases extent with
  | none => simp at remainder
  | some extent =>
    have equal : extent = bytes := by simpa using remainder
    subst extent
    exact source_buffer_extent buffer bytes extentRun

theorem source_buffer_matches (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (buffer : Buffer)
    (computed : strict_memory_frontend_launch.buffer_matches source parsed buffer = .ok true) :
    BufferSupported buffer ∧ ∃ object global, parsed.scope.globals.val[object]? = some global ∧
      BacksObject source.val parsed object buffer ∧ BufferExtent buffer global.bytes.val.length := by
  unfold strict_memory_frontend_launch.buffer_matches at computed
  obtain ⟨supported, supportedRun, remainder⟩ := source_bind_success _ _ _ computed
  cases supported with
  | false => simp at remainder
  | true =>
    have supported := source_buffer_supported buffer supportedRun
    simp only [ite_true] at remainder
    obtain ⟨objectOption, objectRun, remainder⟩ := source_bind_success _ _ _ remainder
    cases objectOption with
    | none => simp at remainder
    | some object =>
      have backing := source_buffer_object source parsed buffer object objectRun
      obtain ⟨global, globalRun, remainder⟩ := source_bind_success _ _ _ remainder
      have atGlobal := vec_index_success parsed.scope.globals object global globalRun
      have castBound : (alloc.vec.Vec.len global.bytes).val ≤ UScalar.max .U64 := by
        have bound := (alloc.vec.Vec.len global.bytes).hBounds
        rcases Usize.bounds_eq with width | width <;> simp_scalar at width ⊢ <;> omega
      have cast := UScalar.cast_inBounds_spec .U64 (alloc.vec.Vec.len global.bytes) castBound
      obtain ⟨bytes, bytesRun, remainder⟩ := source_bind_success _ _ _ remainder
      have byteValue : bytes.val = global.bytes.val.length := by
        rw [bytesRun] at cast
        simpa only [WP.spec_ok, alloc.vec.Vec.len_val] using cast
      have extent := source_buffer_extent_matches buffer bytes remainder
      exact ⟨supported, object.val, global, atGlobal, backing, by simpa only [byteValue] using extent⟩


private def NamesFresh (manifest : Manifest) (index limit : Nat) : Prop :=
  ∀ earlier left right, earlier < limit → manifest.buffers.val[index]? = some left →
    manifest.buffers.val[earlier]? = some right → left.name ≠ right.name

theorem source_buffer_name_fresh (manifest : Manifest) (index : Usize)
    (indexBound : index.val < manifest.buffers.val.length)
    (computed : strict_memory_frontend_launch.buffer_name_fresh manifest index = .ok true) :
    NamesFresh manifest index.val index.val := by
  unfold strict_memory_frontend_launch.buffer_name_fresh
    strict_memory_frontend_launch.buffer_name_fresh_loop at computed
  apply source_loop_success
    (fun (current : Usize × Bool) => strict_memory_frontend_launch.buffer_name_fresh_loop.body manifest index current.1 current.2)
    (fun current => index.val - current.1.val)
    (fun current => current.1.val ≤ index.val ∧ (current.2 = true → NamesFresh manifest index.val current.1.val))
    (fun result => result = true → NamesFresh manifest index.val index.val)
    ?_ (0#usize, true) true ⟨by simp, by intro _ earlier left right impossible; change earlier < 0 at impossible; omega⟩ computed rfl
  rintro ⟨cursor, valid⟩ flow ⟨cursorBound, invariant⟩ bodyRun
  cases valid with
  | false =>
    simp only [strict_memory_frontend_launch.buffer_name_fresh_loop.body, Bool.false_eq_true, ite_false, ok.injEq] at bodyRun
    subst flow
    simp
  | true =>
    have prior := invariant rfl
    change NamesFresh manifest index.val cursor.val at prior
    by_cases inside : cursor.val < index.val
    · have test : cursor < index := by scalar_tac
      have cursorInside : cursor.val < manifest.buffers.val.length := by omega
      let selected := manifest.buffers.val[index.val]
      let earlier := manifest.buffers.val[cursor.val]
      have readSelected : alloc.vec.Vec.index_usize manifest.buffers index = .ok selected := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem indexBound, selected]
      have readEarlier : alloc.vec.Vec.index_usize manifest.buffers cursor = .ok earlier := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem cursorInside, earlier]
      simp only [strict_memory_frontend_launch.buffer_name_fresh_loop.body, ite_true, test,
        alloc.vec.Vec.index_slice_index, readSelected, readEarlier, bind_ok] at bodyRun
      obtain ⟨leftBytes, leftRun, remainder⟩ := source_bind_success _ _ _ bodyRun
      obtain ⟨rightBytes, rightRun, remainder⟩ := source_bind_success _ _ _ remainder
      obtain ⟨same, sameRun, remainder⟩ := source_bind_success _ _ _ remainder
      have sameSpec := source_bytes_equal leftBytes rightBytes
      rw [sameRun] at sameSpec
      have sameMeaning : same = decide (selected.name = earlier.name) := by
        simpa only [WP.spec_ok, source_string_bytes selected.name leftBytes leftRun,
          source_string_bytes earlier.name rightBytes rightRun, utf8_eq_iff] using sameSpec
      obtain ⟨after, increment, remainder⟩ := source_bind_success _ _ _ remainder
      have incrementBound : cursor.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq manifest.buffers
        scalar_tac
      have incrementSpec := source_usize_increment cursor incrementBound
      have afterVal : after.val = cursor.val + 1 := by
        rw [incrementSpec] at increment
        have equal : after = Usize.ofNatCore (cursor.val + 1) incrementBound := by simpa only [ok.injEq] using increment.symm
        simp [equal]
      change Result.ok (.cont (after, !same)) = .ok flow at remainder
      have flowEq : .cont (after, !same) = flow := by simpa only [ok.injEq] using remainder
      subst flow
      refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
      intro accepted
      change NamesFresh manifest index.val after.val
      have different : selected.name ≠ earlier.name := by
        rw [sameMeaning] at accepted
        simpa using accepted
      intro query left right lower leftAt rightAt
      by_cases before : query < cursor.val
      · exact prior query left right before leftAt rightAt
      · have current : query = cursor.val := by omega
        subst query
        have leftEq : left = selected := Option.some.inj (leftAt.symm.trans (List.getElem?_eq_getElem indexBound))
        have rightEq : right = earlier := Option.some.inj (rightAt.symm.trans (List.getElem?_eq_getElem cursorInside))
        simpa only [leftEq, rightEq] using different
    · have test : ¬cursor < index := by scalar_tac
      simp only [strict_memory_frontend_launch.buffer_name_fresh_loop.body, ite_true, test, ite_false, ok.injEq] at bodyRun
      subst flow
      have complete : cursor.val = index.val := by scalar_tac
      simpa only [complete] using invariant


def BufferMatchesMeaning (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram) (buffer : Buffer) : Prop :=
  BufferSupported buffer ∧ ∃ object global, parsed.scope.globals.val[object]? = some global ∧
    BacksObject source parsed object buffer ∧ BufferExtent buffer global.bytes.val.length

private def BuffersPrefix (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (limit : Nat) : Prop :=
  (∀ index buffer, index < limit → manifest.buffers.val[index]? = some buffer → BufferMatchesMeaning source parsed buffer) ∧
  (∀ i j left right, i < j → j < limit → manifest.buffers.val[i]? = some left →
    manifest.buffers.val[j]? = some right → left.name ≠ right.name)

private theorem buffers_supported_loop (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (valid : Bool) (index : Usize) (output : Bool)
    (bound : index.val ≤ manifest.buffers.val.length)
    (prior : valid = true → BuffersPrefix source.val parsed manifest index.val)
    (computed : strict_memory_frontend_launch.buffers_supported_loop source parsed manifest valid index = .ok output) :
    output = true → BuffersPrefix source.val parsed manifest manifest.buffers.val.length := by
  unfold strict_memory_frontend_launch.buffers_supported_loop at computed
  apply source_loop_success
    (fun (current : Bool × Usize) => strict_memory_frontend_launch.buffers_supported_loop.body source parsed manifest current.1 current.2)
    (fun current => manifest.buffers.val.length - current.2.val)
    (fun current => current.2.val ≤ manifest.buffers.val.length ∧
      (current.1 = true → BuffersPrefix source.val parsed manifest current.2.val))
    (fun result => result = true → BuffersPrefix source.val parsed manifest manifest.buffers.val.length)
    ?_ (valid, index) output ⟨bound, prior⟩ computed
  rintro ⟨valid, cursor⟩ flow ⟨cursorBound, invariant⟩ bodyRun
  cases valid with
  | false =>
    simp only [strict_memory_frontend_launch.buffers_supported_loop.body, Bool.false_eq_true, ite_false, ok.injEq] at bodyRun
    subst flow
    simp
  | true =>
    have prior := invariant rfl
    change BuffersPrefix source.val parsed manifest cursor.val at prior
    by_cases inside : cursor.val < manifest.buffers.val.length
    · have test : cursor < alloc.vec.Vec.len manifest.buffers := by scalar_tac
      let buffer := manifest.buffers.val[cursor.val]
      have read : alloc.vec.Vec.index_usize manifest.buffers cursor = .ok buffer := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside, buffer]
      simp only [strict_memory_frontend_launch.buffers_supported_loop.body, ite_true, test,
        alloc.vec.Vec.index_slice_index, read, bind_ok] at bodyRun
      obtain ⟨matched, matchedRun, remainder⟩ := source_bind_success _ _ _ bodyRun
      obtain ⟨fresh, freshRun, remainder⟩ := source_bind_success _ _ _ remainder
      obtain ⟨after, increment, remainder⟩ := source_bind_success _ _ _ remainder
      have incrementBound : cursor.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq manifest.buffers
        scalar_tac
      have incrementSpec := source_usize_increment cursor incrementBound
      have afterVal : after.val = cursor.val + 1 := by
        rw [incrementSpec] at increment
        have equal : after = Usize.ofNatCore (cursor.val + 1) incrementBound := by simpa only [ok.injEq] using increment.symm
        simp [equal]
      change Result.ok (.cont (fresh, after)) = .ok flow at remainder
      have flowEq : .cont (fresh, after) = flow := by simpa only [ok.injEq] using remainder
      subst flow
      refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
      intro accepted
      change fresh = true at accepted
      change BuffersPrefix source.val parsed manifest after.val
      have matching : matched = true := by
        cases matched with
        | false =>
          have rejected : fresh = false := by simpa only [Bool.false_eq_true, ite_false, ok.injEq] using freshRun.symm
          exact False.elim (Bool.false_ne_true (rejected.symm.trans accepted))
        | true => rfl
      have meaning := source_buffer_matches source parsed buffer (matching ▸ matchedRun)
      have freshMeaning := source_buffer_name_fresh manifest cursor inside (by simpa only [matching, ite_true, accepted] using freshRun)
      refine ⟨?_, ?_⟩
      · intro query other lower atOther
        by_cases earlier : query < cursor.val
        · exact prior.1 query other earlier atOther
        · have current : query = cursor.val := by omega
          subst query
          have otherEq : other = buffer := Option.some.inj (atOther.symm.trans (List.getElem?_eq_getElem inside))
          simpa only [BufferMatchesMeaning, otherEq] using meaning
      · intro i j left right order lower leftAt rightAt
        by_cases earlier : j < cursor.val
        · exact prior.2 i j left right order earlier leftAt rightAt
        · have current : j = cursor.val := by omega
          subst j
          exact Ne.symm (freshMeaning i right left order rightAt leftAt)
    · have test : ¬cursor < alloc.vec.Vec.len manifest.buffers := by scalar_tac
      simp only [strict_memory_frontend_launch.buffers_supported_loop.body, ite_true, test, ite_false, ok.injEq] at bodyRun
      subst flow
      have complete : cursor.val = manifest.buffers.val.length := by scalar_tac
      simpa only [complete] using invariant

theorem source_buffers_supported (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest)
    (computed : strict_memory_frontend_launch.buffers_supported source parsed manifest = .ok true) :
    (∀ buffer ∈ manifest.buffers.val, BufferMatchesMeaning source.val parsed buffer) ∧
      manifest.buffers.val.Pairwise (fun left right => left.name ≠ right.name) := by
  have initial : BuffersPrefix source.val parsed manifest 0 := by
    constructor
    · intro index buffer impossible; omega
    · intro i j left right order impossible; omega
  have meaning := buffers_supported_loop source parsed manifest true 0#usize true
    (by simp) (by intro _; exact initial) computed rfl
  refine ⟨?_, ?_⟩
  · intro buffer member
    obtain ⟨index, atBuffer⟩ := List.mem_iff_getElem?.mp member
    exact meaning.1 index buffer (List.getElem?_eq_some_iff.mp atBuffer).1 atBuffer
  · apply List.pairwise_iff_getElem.mpr
    intro i j leftBound rightBound order
    exact meaning.2 i j _ _ order rightBound (List.getElem?_eq_getElem leftBound) (List.getElem?_eq_getElem rightBound)

end PtxFrontendProgram
