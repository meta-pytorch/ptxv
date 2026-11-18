/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceInitialBytes
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

def UniqueBacking (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Nat) : Prop :=
  ∀ (i j : Nat) (left right : Buffer), manifest.buffers.val[i]?=some left →
    manifest.buffers.val[j]?=some right → BacksObject source parsed object left →
    BacksObject source parsed object right → i=j

def BufferIndexMeaning (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Nat) (selected : Option Usize) : Prop :=
  (selected=none → ∀ buffer ∈ manifest.buffers.val, ¬BacksObject source parsed object buffer) ∧
  (∀ index, selected=some index → ∃ buffer, manifest.buffers.val[index.val]?=some buffer ∧
    BacksObject source parsed object buffer ∧
    ∀ j other, manifest.buffers.val[j]?=some other → BacksObject source parsed object other → j=index.val)

private theorem index_meaning_unique (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Nat) (selected : Option Usize)
    (meaning : BufferIndexMeaning source parsed manifest object selected) :
    UniqueBacking source parsed manifest object := by
  intro i j left right leftAt rightAt leftBacks rightBacks
  cases selected with
  | none => exact False.elim (meaning.1 rfl left (List.mem_of_getElem? leftAt) leftBacks)
  | some index =>
    obtain ⟨buffer,_,_,unique⟩ := meaning.2 index rfl
    exact (unique i left leftAt leftBacks).trans (unique j right rightAt rightBacks).symm

theorem source_initial_allocation_from_mapping (source : Slice U8)
    (parsed : strict_memory_frontend_parse.SourceProgram) (manifest : Manifest)
    (object : Usize) (allocation : Allocation)
    (mapping : ∀ selected, strict_memory_frontend_launch.buffer_index source parsed manifest object = .ok (some selected) →
      BufferIndexMeaning source.val parsed manifest object.val selected)
    (computed : strict_memory_frontend_launch.initial_allocation source parsed manifest object = .ok (some allocation)) :
    InitialAllocation source.val parsed manifest object.val allocation ∧
      UniqueBacking source.val parsed manifest object.val := by
  unfold strict_memory_frontend_launch.initial_allocation at computed
  by_cases outside : object >= alloc.vec.Vec.len parsed.scope.globals
  · simp [outside] at computed
  · simp only [outside,ite_false] at computed
    obtain ⟨selected,indexComputed,computed⟩ := source_bind_success _ _ _ computed
    cases selected with
    | none => simp at computed
    | some selected =>
      have meaning := mapping selected indexComputed
      have unique := index_meaning_unique source.val parsed manifest object.val selected meaning
      obtain ⟨global,read,computed⟩ := source_bind_success _ _ _ computed
      have globalAt := vec_index_success parsed.scope.globals object global read
      by_cases tooLarge : alloc.vec.Vec.len global.bytes > 67108864#usize
      · simp [tooLarge] at computed
      · simp only [tooLarge,ite_false] at computed
        have room : global.bytes.val.length < Usize.max := by scalar_tac
        cases selected with
        | none =>
          obtain ⟨pair,walked,computed⟩ := source_bind_success _ _ _ computed
          have law := source_copy_initial_bytes global.bytes (alloc.vec.Vec.new U8)
            (alloc.vec.Vec.new Bool) 0#usize room (by simp) (by simp) (by simp)
          rw [walked] at law
          simp only [WP.spec_ok] at law
          rcases pair with ⟨bytes,defined⟩
          have equal : ({ bytes,defined,live := true } : Allocation) = allocation := by simpa using computed
          subst allocation
          refine ⟨⟨global,globalAt,rfl,Or.inl ⟨?_,law.1,law.2⟩⟩,unique⟩
          rintro ⟨buffer,member,backs⟩
          exact meaning.1 rfl buffer member backs
        | some index =>
          obtain ⟨buffer,bufferRead,computed⟩ := source_bind_success _ _ _ computed
          have bufferAt := vec_index_success manifest.buffers index buffer bufferRead
          obtain ⟨supported,_,computed⟩ := source_bind_success _ _ _ computed
          cases supported with
          | false => simp at computed
          | true =>
            simp only [ite_true] at computed
            obtain ⟨extent,_,computed⟩ := source_bind_success _ _ _ computed
            cases extent with
            | none => simp at computed
            | some extent =>
              simp only [lift,bind_ok] at computed
              by_cases unequal : extent != UScalar.cast UScalarTy.U64 (alloc.vec.Vec.len global.bytes)
              · simp [unequal] at computed
              · simp only [unequal,ite_false] at computed
                obtain ⟨initBytes,encoded,computed⟩ := source_bind_success _ _ _ computed
                obtain ⟨initialized,compared,computed⟩ := source_bind_success _ _ _ computed
                have initEq := source_string_bytes buffer.init initBytes encoded
                let zeroBytes : Slice U8 := Array.to_slice (Array.make 5#usize [122#u8,101#u8,114#u8,111#u8,115#u8])
                have zeroEq : zeroBytes.val = utf8 "zeros" := by simp [zeroBytes,Array.to_slice,Array.make,utf8]; decide
                have compareLaw := source_bytes_equal initBytes zeroBytes
                rw [compared] at compareLaw
                have initializedEq : initialized = (buffer.init == "zeros") := by
                  rw [Bool.beq_eq_decide_eq]
                  simpa only [WP.spec_ok,initEq,zeroEq,utf8_eq_iff] using compareLaw
                obtain ⟨pair,walked,computed⟩ := source_bind_success _ _ _ computed
                have law := source_zero_initial_bytes global.bytes (alloc.vec.Vec.new U8) initialized
                  (alloc.vec.Vec.new Bool) 0#usize room (by simp) (by simp) (by simp)
                rw [walked] at law
                simp only [WP.spec_ok] at law
                rcases pair with ⟨bytes,defined⟩
                have equal : ({ bytes,defined,live := true } : Allocation) = allocation := by simpa using computed
                subst allocation
                obtain ⟨selectedBuffer,selectedAt,backs,_⟩ := meaning.2 index rfl
                have same : selectedBuffer=buffer := by simpa only [bufferAt,Option.some.injEq] using selectedAt.symm
                subst selectedBuffer
                refine ⟨⟨global,globalAt,rfl,Or.inr ⟨buffer,List.mem_of_getElem? bufferAt,backs,law.1,?_⟩⟩,unique⟩
                simpa only [initializedEq] using law.2

theorem source_initial_allocation (source : Slice U8)
    (parsed : strict_memory_frontend_parse.SourceProgram) (manifest : Manifest)
    (object : Usize) (allocation : Allocation) (names : GlobalNamesInjective source.val parsed)
    (computed : strict_memory_frontend_launch.initial_allocation source parsed manifest object = .ok (some allocation)) :
    InitialAllocation source.val parsed manifest object.val allocation ∧
      UniqueBacking source.val parsed manifest object.val :=
  source_initial_allocation_from_mapping source parsed manifest object allocation
    (fun selected accepted => source_buffer_index source parsed manifest object selected names accepted) computed

end PtxFrontendProgram
