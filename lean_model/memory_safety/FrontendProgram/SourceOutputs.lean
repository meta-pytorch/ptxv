/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceLoop
import FrontendProgram.SourceBytes
import FrontendProgram.SourceBufferMapping

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

private def OutputWitness (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (label : List U8) (object : Usize) : Prop :=
  ∃ buffer ∈ manifest.buffers.val, utf8 buffer.name = label ∧ BacksObject source parsed object.val buffer

private theorem resolve_output_loop (source label : Slice U8)
    (parsed : strict_memory_frontend_parse.SourceProgram) (manifest : Manifest)
    (mapping : ∀ buffer object, strict_memory_frontend_launch.buffer_object source parsed buffer = .ok (some object) →
      BacksObject source.val parsed object.val buffer)
    (found : Usize) (unique : Bool) (index : Usize)
    (bound : index.val ≤ manifest.buffers.val.length)
    (invariant : unique = true → found = core.num.Usize.MAX ∨ OutputWitness source.val parsed manifest label.val found)
    (result : Usize × Bool)
    (computed : strict_memory_frontend_launch.resolve_output_loop source parsed manifest label found unique index = .ok result) :
    result.2 = true → result.1 = core.num.Usize.MAX ∨ OutputWitness source.val parsed manifest label.val result.1 := by
  unfold strict_memory_frontend_launch.resolve_output_loop at computed
  apply source_loop_success _ (fun current => manifest.buffers.val.length-current.2.2.val)
    (fun current => current.2.2.val ≤ manifest.buffers.val.length ∧
      (current.2.1 = true → current.1 = core.num.Usize.MAX ∨ OutputWitness source.val parsed manifest label.val current.1))
    (fun output => output.2 = true → output.1 = core.num.Usize.MAX ∨ OutputWitness source.val parsed manifest label.val output.1)
    ?_ (found,unique,index) result ⟨bound,invariant⟩ computed
  rintro ⟨found,unique,index⟩ flow ⟨bound,invariant⟩ computed
  unfold strict_memory_frontend_launch.resolve_output_loop.body at computed
  cases unique with
  | false =>
    simp only [Bool.false_eq_true,ite_false,ok.injEq] at computed
    subst flow
    simp
  | true =>
    simp only [ite_true] at computed
    by_cases inside : index < alloc.vec.Vec.len manifest.buffers
    · simp only [inside,ite_true] at computed
      obtain ⟨buffer,read,computed⟩ := source_bind_success _ _ _ computed
      have lookup := vec_index_success manifest.buffers index buffer read
      have member : buffer ∈ manifest.buffers.val := List.mem_of_getElem? lookup
      obtain ⟨bytes,encoded,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨same,compared,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨pair,selected,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨next,increment,computed⟩ := source_bind_success _ _ _ computed
      have nextVal : next.val = index.val+1 := by
        have room := alloc.vec.Vec.len_ineq manifest.buffers
        have law := Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac)
        rw [increment] at law
        simpa only [WP.spec_ok,UScalar.ofNatCore_val_eq] using law
      simp only [ok.injEq] at computed
      subst flow
      refine ⟨⟨by scalar_tac,?_⟩,by scalar_tac⟩
      intro isUnique
      cases same with
      | false =>
        simp only [Bool.false_eq_true,ite_false,ok.injEq] at selected
        subst pair
        exact invariant rfl
      | true =>
        simp only [ite_true] at selected
        obtain ⟨object,objectComputed,selected⟩ := source_bind_success _ _ _ selected
        cases object with
        | none =>
          simp only [ok.injEq] at selected
          subst pair
          cases isUnique
        | some object =>
          simp only [ok.injEq] at selected
          subst pair
          right
          have law := source_bytes_equal bytes label
          rw [compared] at law
          have equalBytes : bytes.val = label.val := by simpa only [WP.spec_ok, Bool.true_eq, decide_eq_true_eq] using law
          refine ⟨buffer,member,?_,mapping buffer object objectComputed⟩
          rw [←source_string_bytes buffer.name bytes encoded]
          exact equalBytes
    · simp only [inside,ite_false,ok.injEq] at computed
      subst flow
      exact invariant

theorem source_resolve_output_from_mapping (source label : Slice U8)
    (parsed : strict_memory_frontend_parse.SourceProgram) (manifest : Manifest) (object : Usize)
    (mapping : ∀ buffer object, strict_memory_frontend_launch.buffer_object source parsed buffer = .ok (some object) →
      BacksObject source.val parsed object.val buffer)
    (computed : strict_memory_frontend_launch.resolve_output source parsed manifest label = .ok (some object)) :
    OutputWitness source.val parsed manifest label.val object := by
  unfold strict_memory_frontend_launch.resolve_output at computed
  obtain ⟨result,walked,computed⟩ := source_bind_success _ _ _ computed
  have law := resolve_output_loop source label parsed manifest mapping core.num.Usize.MAX true 0#usize
    (by simp) (by intro _; exact Or.inl rfl) result walked
  rcases result with ⟨found,unique⟩
  change (if unique = true then (if (found != core.num.Usize.MAX) = true then Result.ok (some found) else Result.ok none) else Result.ok none) = Result.ok (some object) at computed
  cases unique with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    by_cases missing : found = core.num.Usize.MAX
    · simp [missing] at computed
    · have same : found = object := by simpa [missing] using computed
      subst object
      exact (law rfl).resolve_left missing

private def OutputPrefix (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (count : Nat) (outputs : List Usize) : Prop :=
  outputs.length=count ∧ ∀ (position : Nat) (object : Usize) (label : String), outputs[position]?=some object →
    manifest.outputs.val[position]?=some label →
    ∃ buffer ∈ manifest.buffers.val, buffer.name=label ∧ BacksObject source parsed object.val buffer

private theorem output_prefix_append (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (count : Nat) (outputs : List Usize) (object : Usize) (label : String)
    (inventory : OutputPrefix source parsed manifest count outputs)
    (labelAt : manifest.outputs.val[count]?=some label)
    (witness : OutputWitness source parsed manifest (utf8 label) object) :
    OutputPrefix source parsed manifest (count+1) (outputs++[object]) := by
  refine ⟨by simpa using inventory.1,?_⟩
  have length : outputs.length=count := inventory.1
  intro position selected selectedLabel lookup selectedLabelAt
  by_cases inside : position < outputs.length
  · apply inventory.2 position selected selectedLabel _ selectedLabelAt
    simpa only [List.getElem?_append,inside,ite_true] using lookup
  · have tail : [object][position-outputs.length]?=some selected := by
      simpa only [List.getElem?_append,inside,ite_false] using lookup
    cases difference : position-outputs.length with
    | zero =>
      have atEnd : position=count := by omega
      have selectedEq : object=selected := by simpa only [difference,List.getElem?_cons_zero,Option.some.injEq] using tail
      have labelEq : label=selectedLabel := by simpa only [atEnd,labelAt,Option.some.injEq] using selectedLabelAt
      obtain ⟨buffer,member,named,backs⟩ := witness
      exact ⟨buffer,member,(utf8_injective named).trans labelEq,by simpa only [←selectedEq] using backs⟩
    | succ rest => simp only [difference,List.getElem?_cons_succ,List.getElem?_nil] at tail; cases tail

private theorem outputs_loop (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest)
    (mapping : ∀ buffer object, strict_memory_frontend_launch.buffer_object source parsed buffer = .ok (some object) →
      BacksObject source.val parsed object.val buffer)
    (outputs : alloc.vec.Vec Usize) (valid : Bool) (index : Usize)
    (bound : index.val ≤ manifest.outputs.val.length)
    (inventory : valid=true → OutputPrefix source.val parsed manifest index.val outputs.val)
    (result : alloc.vec.Vec Usize × Bool)
    (computed : strict_memory_frontend_launch.outputs_loop source parsed manifest outputs valid index = .ok result) :
    result.2=true → OutputsMeaning source.val parsed manifest result.1.val := by
  unfold strict_memory_frontend_launch.outputs_loop at computed
  apply source_loop_success _ (fun current => manifest.outputs.val.length-current.2.2.val)
    (fun current => current.2.2.val ≤ manifest.outputs.val.length ∧
      (current.2.1=true → OutputPrefix source.val parsed manifest current.2.2.val current.1.val))
    (fun output => output.2=true → OutputsMeaning source.val parsed manifest output.1.val)
    ?_ (outputs,valid,index) result ⟨bound,inventory⟩ computed
  rintro ⟨outputs,valid,index⟩ flow ⟨bound,inventory⟩ computed
  unfold strict_memory_frontend_launch.outputs_loop.body at computed
  cases valid with
  | false =>
    simp only [Bool.false_eq_true,ite_false,ok.injEq] at computed
    subst flow
    simp
  | true =>
    simp only [ite_true] at computed
    by_cases inside : index < alloc.vec.Vec.len manifest.outputs
    · simp only [inside,ite_true] at computed
      obtain ⟨label,read,computed⟩ := source_bind_success _ _ _ computed
      have labelAt := vec_index_success manifest.outputs index label read
      obtain ⟨bytes,encoded,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨resolved,resolvedComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨pair,appended,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨next,increment,computed⟩ := source_bind_success _ _ _ computed
      have nextVal : next.val=index.val+1 := by
        have room := alloc.vec.Vec.len_ineq manifest.outputs
        have law := Usize.add_spec (x:=index) (y:=1#usize) (by scalar_tac)
        rw [increment] at law
        simpa only [WP.spec_ok,UScalar.ofNatCore_val_eq] using law
      simp only [ok.injEq] at computed
      subst flow
      refine ⟨⟨by scalar_tac,?_⟩,by scalar_tac⟩
      intro isValid
      cases resolved with
      | none =>
        simp only [ok.injEq] at appended
        subst pair
        cases isValid
      | some object =>
        obtain ⟨nextOutputs,pushed,appended⟩ := source_bind_success _ _ _ appended
        have appendedEq := vec_push_success outputs nextOutputs object pushed
        simp only [ok.injEq] at appended
        subst pair
        rw [nextVal,appendedEq]
        apply output_prefix_append _ _ _ _ _ _ _ (inventory rfl) labelAt
        have witness := source_resolve_output_from_mapping source bytes parsed manifest object mapping resolvedComputed
        simpa only [source_string_bytes label bytes encoded] using witness
    · simp only [inside,ite_false,ok.injEq] at computed
      subst flow
      intro _
      have atEnd : index.val=manifest.outputs.val.length := by scalar_tac
      exact ⟨by simpa only [atEnd] using (inventory rfl).1,(inventory rfl).2⟩

theorem source_outputs_from_mapping (source : Slice U8)
    (parsed : strict_memory_frontend_parse.SourceProgram) (manifest : Manifest)
    (outputs : alloc.vec.Vec Usize)
    (mapping : ∀ buffer object, strict_memory_frontend_launch.buffer_object source parsed buffer = .ok (some object) →
      BacksObject source.val parsed object.val buffer)
    (computed : strict_memory_frontend_launch.outputs source parsed manifest = .ok (some outputs)) :
    OutputsMeaning source.val parsed manifest outputs.val := by
  unfold strict_memory_frontend_launch.outputs at computed
  obtain ⟨result,walked,computed⟩ := source_bind_success _ _ _ computed
  have emptyPrefix : OutputPrefix source.val parsed manifest 0 (alloc.vec.Vec.new Usize).val := by
    refine ⟨by simp,?_⟩
    intro position object label impossible
    simp at impossible
  have law := outputs_loop source parsed manifest mapping (alloc.vec.Vec.new Usize) true 0#usize
    (by simp) (by intro _; exact emptyPrefix) result walked
  rcases result with ⟨built,valid⟩
  change (if valid=true then Result.ok (some built) else Result.ok none)=Result.ok (some outputs) at computed
  cases valid with
  | false => simp at computed
  | true =>
    have equal : built=outputs := by simpa using computed
    subst outputs
    exact law rfl

theorem source_outputs (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (outputs : alloc.vec.Vec Usize)
    (computed : strict_memory_frontend_launch.outputs source parsed manifest = .ok (some outputs)) :
    OutputsMeaning source.val parsed manifest outputs.val :=
  source_outputs_from_mapping source parsed manifest outputs
    (source_buffer_object source parsed) computed

end PtxFrontendProgram
