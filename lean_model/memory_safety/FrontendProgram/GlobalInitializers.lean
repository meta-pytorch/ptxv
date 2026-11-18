/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.OpcodeTables
import FrontendProgram.NamesDecision
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

private theorem parse_loop_success {α β : Type} (body : α → Result (ControlFlow α β))
    (measure : α → Nat) (invariant : α → Prop) (post : β → Prop)
    (bodySound : ∀ before flow, invariant before → body before = .ok flow →
      match flow with
      | .cont after => invariant after ∧ measure after < measure before
      | .done output => post output)
    (before : α) (output : β) (valid : invariant before)
    (computed : loop body before = .ok output) : post output := by
  generalize measured : measure before = count
  induction count using Nat.strong_induction_on generalizing before with
  | h count ih =>
    rw [loop.eq_def] at computed
    obtain ⟨flow,first,remaining⟩ := source_bind_success _ _ _ computed
    have sound := bodySound before flow valid first
    cases flow with
    | done result =>
      have equal : result = output := by simpa only [ok.injEq] using remaining
      simpa only [equal] using sound
    | cont after => exact ih (measure after) (by omega) after sound.1 remaining rfl

theorem little_endian_length (width value : Nat) : (littleEndian width value).length = width := by
  simp [littleEndian]

theorem source_initializer_byte (value : U64) (byte shift : U8) (shifted : U64) (result : U8)
    (inside : byte.val < 8)
    (multiply : (byte * 8#u8 : Result U8) = .ok shift)
    (shiftRun : (value >>> shift : Result U64) = .ok shifted)
    (castRun : lift (UScalar.cast .U8 shifted) = .ok result) :
    result.val = value.val / 256 ^ byte.val % 256 := by
  have multiplied := UScalar.mul_spec (x := byte) (y := 8#u8) (by scalar_tac)
  rw [multiply] at multiplied
  have shiftVal : shift.val = byte.val * 8 := by simpa [WP.spec_ok] using multiplied
  have shiftedSpec := UScalar.ShiftRight_spec value shift (by scalar_tac)
  rw [shiftRun] at shiftedSpec
  have actualVal := (show shifted.val = value.val >>> shift.val ∧ shifted.bv = value.bv >>> shift.val from by simpa only [WP.spec_ok] using shiftedSpec).1
  have castEq : result = UScalar.cast .U8 shifted := by simpa only [lift, ok.injEq] using castRun.symm
  have power : 256 ^ byte.val = 2 ^ (byte.val * 8) := by
    rw [Nat.mul_comm byte.val 8, Nat.pow_mul]
  rw [castEq, UScalar.cast_val_eq, actualVal, Nat.shiftRight_eq_div_pow, shiftVal, ← power]
  rfl

theorem source_initializer_bytes (width : U8) (before after : alloc.vec.Vec U8) (value : U64)
    (widthBound : width.val ≤ 8)
    (computed : strict_memory_frontend_parse.parse_global_loop0_loop0 width before true value 0#u8 = .ok after) :
    after.val = before.val ++ littleEndian width.val value.val := by
  unfold strict_memory_frontend_parse.parse_global_loop0_loop0 at computed
  apply parse_loop_success
    (fun (current : alloc.vec.Vec U8 × U8) => strict_memory_frontend_parse.parse_global_loop0_loop0.body width true value current.1 current.2)
    (fun current => width.val - current.2.val)
    (fun current => current.2.val ≤ width.val ∧ current.1.val = before.val ++ (littleEndian width.val value.val).take current.2.val)
    (fun output => output.val = before.val ++ littleEndian width.val value.val)
    ?_ (before,0#u8) after ⟨by simp, by simp⟩ computed
  rintro ⟨bytes, cursor⟩ flow ⟨cursorBound, invariant⟩ bodyRun
  change cursor.val ≤ width.val at cursorBound
  change bytes.val = before.val ++ (littleEndian width.val value.val).take cursor.val at invariant
  by_cases inside : cursor.val < width.val
  · have test : cursor < width := by scalar_tac
    simp only [strict_memory_frontend_parse.parse_global_loop0_loop0.body, ite_true, test] at bodyRun
    obtain ⟨shift, multiply, remainder⟩ := source_bind_success _ _ _ bodyRun
    obtain ⟨shifted, shiftedRun, remainder⟩ := source_bind_success _ _ _ remainder
    obtain ⟨byte, castRun, remainder⟩ := source_bind_success _ _ _ remainder
    obtain ⟨updated, pushRun, remainder⟩ := source_bind_success _ _ _ remainder
    obtain ⟨next, increment, remainder⟩ := source_bind_success _ _ _ remainder
    have byteMeaning := source_initializer_byte value cursor shift shifted byte (by omega) multiply shiftedRun castRun
    have pushMeaning := vec_push_success bytes updated byte pushRun
    have incrementSpec := UScalar.add_spec (x := cursor) (y := 1#u8) (by scalar_tac)
    rw [increment] at incrementSpec
    have nextVal : next.val = cursor.val + 1 := by simpa [WP.spec_ok] using incrementSpec
    change Result.ok (.cont (updated, next)) = .ok flow at remainder
    have flowEq : .cont (updated, next) = flow := by simpa only [ok.injEq] using remainder
    subst flow
    refine ⟨⟨by scalar_tac, ?_⟩, by scalar_tac⟩
    change updated.val = before.val ++ (littleEndian width.val value.val).take next.val
    rw [pushMeaning, invariant, nextVal, List.append_assoc]
    congr 1
    have indexBound : cursor.val < (littleEndian width.val value.val).length := by simpa only [little_endian_length] using inside
    rw [List.take_succ_eq_append_getElem indexBound]
    congr 2
    apply UScalar.eq_of_val_eq
    simpa only [littleEndian, List.getElem_map, List.getElem_range, UScalar.ofNatCore_val_eq] using byteMeaning
  · have test : ¬cursor < width := by scalar_tac
    simp only [strict_memory_frontend_parse.parse_global_loop0_loop0.body, ite_true, test, ite_false, ok.injEq] at bodyRun
    subst flow
    have complete : cursor.val = width.val := by omega
    have taken : (littleEndian width.val value.val).take width.val = littleEndian width.val value.val := List.take_of_length_le (by simp [little_endian_length])
    simpa only [complete, taken] using invariant


def InitializerWidth (width : U8) : Prop :=
  width.val = 1 ∨ width.val = 2 ∨ width.val = 4 ∨ width.val = 8

theorem source_initializer_value_fits (width : U8) (value : U64)
    (supported : InitializerWidth width)
    (computed : (if width != 8#u8 then do
      let shift ← width * 8#u8
      let limit ← (1#u64 <<< shift : Result U64)
      if value >= limit then .ok false else .ok true
      else .ok true) = .ok true) : value.val < 256 ^ width.val := by
  rcases supported with one | two | four | eight
  · have equal : width = 1#u8 := by scalar_tac
    subst width
    have multiply : (1#u8 * 8#u8 : Result U8) = .ok 8#u8 := by
      obtain ⟨actual, run, meaning⟩ := MemoryWrite.spec_returns _ _ (UScalar.mul_spec (x := 1#u8) (y := 8#u8) (by scalar_tac))
      have equal : actual = 8#u8 := by scalar_tac
      simpa only [equal] using run
    have shifted : (1#u64 <<< 8#u8 : Result U64) = .ok 256#u64 := by
      obtain ⟨actual, run, meaning⟩ := MemoryWrite.spec_returns _ _ (U64.ShiftLeft_spec 1#u64 8#u8 (by scalar_tac))
      have equal : actual = 256#u64 := by
        apply UScalar.eq_of_val_eq
        simpa [U64.size, U64.numBits] using meaning.1
      simpa only [equal] using run
    have differs : (1#u8 != 8#u8) = true := by decide
    simp only [differs, ite_true, multiply, bind_tc_ok, shifted] at computed
    by_cases outside : value >= 256#u64
    · simp [outside] at computed
    · scalar_tac
  · have equal : width = 2#u8 := by scalar_tac
    subst width
    have multiply : (2#u8 * 8#u8 : Result U8) = .ok 16#u8 := by
      obtain ⟨actual, run, meaning⟩ := MemoryWrite.spec_returns _ _ (UScalar.mul_spec (x := 2#u8) (y := 8#u8) (by scalar_tac))
      have equal : actual = 16#u8 := by scalar_tac
      simpa only [equal] using run
    have shifted : (1#u64 <<< 16#u8 : Result U64) = .ok 65536#u64 := by
      obtain ⟨actual, run, meaning⟩ := MemoryWrite.spec_returns _ _ (U64.ShiftLeft_spec 1#u64 16#u8 (by scalar_tac))
      have equal : actual = 65536#u64 := by
        apply UScalar.eq_of_val_eq
        simpa [U64.size, U64.numBits] using meaning.1
      simpa only [equal] using run
    have differs : (2#u8 != 8#u8) = true := by decide
    simp only [differs, ite_true, multiply, bind_tc_ok, shifted] at computed
    by_cases outside : value >= 65536#u64
    · simp [outside] at computed
    · scalar_tac
  · have equal : width = 4#u8 := by scalar_tac
    subst width
    have multiply : (4#u8 * 8#u8 : Result U8) = .ok 32#u8 := by
      obtain ⟨actual, run, meaning⟩ := MemoryWrite.spec_returns _ _ (UScalar.mul_spec (x := 4#u8) (y := 8#u8) (by scalar_tac))
      have equal : actual = 32#u8 := by scalar_tac
      simpa only [equal] using run
    have shifted : (1#u64 <<< 32#u8 : Result U64) = .ok 4294967296#u64 := by
      obtain ⟨actual, run, meaning⟩ := MemoryWrite.spec_returns _ _ (U64.ShiftLeft_spec 1#u64 32#u8 (by scalar_tac))
      have equal : actual = 4294967296#u64 := by
        apply UScalar.eq_of_val_eq
        simpa [U64.size, U64.numBits] using meaning.1
      simpa only [equal] using run
    have differs : (4#u8 != 8#u8) = true := by decide
    simp only [differs, ite_true, multiply, bind_tc_ok, shifted] at computed
    by_cases outside : value >= 4294967296#u64
    · simp [outside] at computed
    · scalar_tac
  · have equal : width = 8#u8 := by scalar_tac
    subst width
    scalar_tac

private theorem initializer_loop_invalid (source : Slice U8) (tokens : Slice SourceToken)
    (width : U8) (count : U64) (after : Usize) (bytes : alloc.vec.Vec U8) (element : U64) :
    strict_memory_frontend_parse.parse_global_loop0 source tokens width count after bytes element false = .ok (after,bytes,false) := by
  unfold strict_memory_frontend_parse.parse_global_loop0
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.parse_global_loop0.body]


theorem source_initializer_values (source : Slice U8) (tokens : Slice SourceToken)
    (width : U8) (count : U64) (start finish : Usize)
    (before after : alloc.vec.Vec U8) (element : U64)
    (supported : InitializerWidth width) (remaining : element.val < count.val)
    (computed : strict_memory_frontend_parse.parse_global_loop0 source tokens width count
      start before element true = .ok (finish,after,true)) :
    ∃ values, InitializerSyntax source.val tokens.val width.val start.val values finish.val ∧
      values.length = count.val-element.val ∧
      after.val = before.val ++ values.flatMap (littleEndian width.val) := by
  generalize measured : count.val-element.val = amount
  induction amount using Nat.strong_induction_on generalizing start before element with
  | h amount ih =>
    unfold strict_memory_frontend_parse.parse_global_loop0 at computed
    rw [loop.eq_def] at computed
    obtain ⟨flow,bodyRun,rest⟩ := source_bind_success _ _ _ computed
    have test : element < count := by scalar_tac
    simp only [strict_memory_frontend_parse.parse_global_loop0.body,ite_true,test] at bodyRun
    obtain ⟨parsed,parseRun,bodyRun⟩ := source_bind_success _ _ _ bodyRun
    cases parsed with
    | none =>
      simp only [ok.injEq] at bodyRun
      subst flow
      change strict_memory_frontend_parse.parse_global_loop0 source tokens width count start before element false = _ at rest
      rw [initializer_loop_invalid] at rest
      simp at rest
    | some pair =>
      rcases pair with ⟨value,next⟩
      obtain ⟨valid,validRun,bodyRun⟩ := source_bind_success _ _ _ bodyRun
      obtain ⟨updated,bytesRun,bodyRun⟩ := source_bind_success _ _ _ bodyRun
      obtain ⟨following,increment,bodyRun⟩ := source_bind_success _ _ _ bodyRun
      have incrementSpec := UScalar.add_spec (x:=element) (y:=1#u64) (by scalar_tac)
      rw [increment] at incrementSpec
      have followingVal : following.val=element.val+1 := by simpa [WP.spec_ok] using incrementSpec
      have number := source_unsigned source tokens start next value parseRun
      have widthBound : width.val ≤ 8 := by rcases supported with h|h|h|h <;>omega
      by_cases more : following < count
      · simp only [more,ite_true,lift,bind_ok] at bodyRun
        obtain ⟨comma,commaRun,bodyRun⟩ := source_bind_success _ _ _ bodyRun
        cases comma with
        | none =>
          simp only [ok.injEq] at bodyRun
          subst flow
          change strict_memory_frontend_parse.parse_global_loop0 source tokens width count next updated following false = _ at rest
          rw [initializer_loop_invalid] at rest
          simp at rest
        | some nextStart =>
          simp only [ok.injEq] at bodyRun
          subst flow
          change strict_memory_frontend_parse.parse_global_loop0 source tokens width count nextStart updated following valid = _ at rest
          cases valid with
          | false => rw [initializer_loop_invalid] at rest; simp at rest
          | true =>
            have fits := source_initializer_value_fits width value supported validRun
            have bytesMeaning := source_initializer_bytes width before updated value widthBound bytesRun
            have smaller : count.val-following.val < amount := by scalar_tac
            obtain ⟨values,grammar,length,bytes⟩ := ih (count.val-following.val) smaller nextStart updated following (by scalar_tac) rest rfl
            obtain ⟨commaNext,commaToken,commaFound,commaMeaning⟩ := source_take source tokens next nextStart _ commaRun
            have commaSyntax : TokenAt source.val tokens.val (start.val+1) "," := by
              have literal : utf8 ","=[44#u8] := by decide
              refine ⟨commaToken,?_,?_⟩
              · simpa only [number.1] using commaFound
              · simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning
            refine ⟨value.val::values,?_,?_,?_⟩
            · apply InitializerSyntax.cons number.2 fits commaSyntax
              have location : nextStart.val=start.val+2 := by omega
              simpa only [location] using grammar
            · simp only [List.length_cons]; omega
            · simpa only [List.flatMap_cons,bytesMeaning,List.append_assoc] using bytes
      · simp only [more,ite_false,ok.injEq] at bodyRun
        subst flow
        change strict_memory_frontend_parse.parse_global_loop0 source tokens width count next updated following valid = _ at rest
        cases valid with
        | false => rw [initializer_loop_invalid] at rest; simp at rest
        | true =>
          have fits := source_initializer_value_fits width value supported validRun
          have bytesMeaning := source_initializer_bytes width before updated value widthBound bytesRun
          unfold strict_memory_frontend_parse.parse_global_loop0 at rest
          rw [loop.eq_def] at rest
          simp only [strict_memory_frontend_parse.parse_global_loop0.body,ite_true,more,ite_false,bind_ok,ok.injEq,Prod.mk.injEq] at rest
          rcases rest with ⟨rfl,rfl,_⟩
          refine ⟨[value.val],?_,?_,?_⟩
          · simpa only [number.1] using InitializerSyntax.single number.2 fits
          · simp only [List.length_singleton]; scalar_tac
          · simpa only [List.flatMap_singleton] using bytesMeaning

end PtxFrontendProgram
