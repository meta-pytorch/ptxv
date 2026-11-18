/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Names

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false

namespace PtxFrontendProgram

local instance (byte : U8) : Decidable (SourceByte byte) := by
  unfold SourceByte
  infer_instance

theorem source_byte_supported (byte : U8) :
    strict_memory_frontend.source_byte_supported byte = .ok (decide (SourceByte byte)) := by
  generalize expectedEq : decide (SourceByte byte) = expected
  unfold strict_memory_frontend.source_byte_supported
  repeat' split
  all_goals apply congrArg (fun value : Bool => (ok value : Result Bool))
  all_goals rw [←expectedEq]
  all_goals apply Bool.eq_iff_iff.mpr
  all_goals simp_all [SourceByte]
  have low : ¬9 ≤ byte.val := by omega
  have high : ¬32 ≤ byte.val := by omega
  simpa [low, high] using expectedEq.symm

private theorem source_text_false (source : Slice U8) (index : Usize) :
    strict_memory_frontend.source_text_supported_loop source false index=.ok false := by
  unfold strict_memory_frontend.source_text_supported_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend.source_text_supported_loop.body,bind_ok]

private theorem source_text_loop (source : Slice U8) (index : Usize)
    (computed : strict_memory_frontend.source_text_supported_loop source true index=.ok true) :
    ∀ position byte,index.val≤position → source.val[position]?=some byte → SourceByte byte := by
  generalize fuelEq : source.val.length-index.val=fuel
  induction fuel using Nat.strong_induction_on generalizing index with
  | h fuel ih =>
    have unroll : strict_memory_frontend.source_text_supported_loop source true index=(do
        let outcome ← strict_memory_frontend.source_text_supported_loop.body source true index
        match outcome with
        | .cont pair => strict_memory_frontend.source_text_supported_loop source pair.1 pair.2
        | .done value => .ok value) := by
      unfold strict_memory_frontend.source_text_supported_loop
      rw [loop.eq_def]
      congr 1
      funext outcome
      cases outcome with
      | cont pair => cases pair;rfl
      | done value => rfl
    rw [unroll] at computed
    by_cases inside : index.val<source.val.length
    · have test : index<Slice.len source := by scalar_tac
      let byte := source.val[index.val]
      have read : Slice.index_usize source index=.ok byte := by
        simp [Slice.index_usize,List.getElem?_eq_getElem inside,byte];rfl
      have incrementBound : index.val+1<2^UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let after := Usize.ofNatCore (index.val+1) incrementBound
      have afterVal : after.val=index.val+1 := by simp [after]
      have increment := source_usize_increment index incrementBound
      simp only [strict_memory_frontend.source_text_supported_loop.body,
        ite_true,test,read,bind_ok,source_byte_supported,Std.bind_assoc] at computed
      by_cases good : SourceByte byte
      · simp only [good,decide_true,increment,bind_ok] at computed
        change strict_memory_frontend.source_text_supported_loop source true after=.ok true at computed
        have tail:=ih (source.val.length-after.val) (by omega) after computed rfl
        intro position selected lower found
        by_cases samePosition : position=index.val
        · subst position
          have actual : source.val[index.val]?=some byte:=List.getElem?_eq_getElem inside
          have selectedEq : selected=byte := Option.some.inj (found.symm.trans actual)
          subst selected
          exact good
        · exact tail position selected (by omega) found
      · simp only [good,decide_false,increment,bind_ok] at computed
        change strict_memory_frontend.source_text_supported_loop source false after=.ok true at computed
        rw [source_text_false] at computed
        simp at computed
    · intro position byte lower found
      have := (List.getElem?_eq_some_iff.mp found).1
      omega

theorem source_text_supported (source : Slice U8)
    (computed : strict_memory_frontend.source_text_supported source=.ok true) :
    SourceText source.val := by
  unfold strict_memory_frontend.source_text_supported at computed
  intro byte member
  obtain ⟨position,found⟩:=List.mem_iff_getElem?.mp member
  exact source_text_loop source 0#usize computed position byte (by simp) found

theorem source_text_nonzero_ascii (source : List U8) (valid : SourceText source)
    (byte : U8) (member : byte∈source) : 0<byte.val ∧ byte.val<128 := by
  have accepted:=valid byte member
  unfold SourceByte at accepted
  omega

end PtxFrontendProgram
