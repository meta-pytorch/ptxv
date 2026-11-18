/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.OpcodeTables

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

theorem source_load_type (source : Slice U8) (token : SourceToken) (bytes : U8) (signed : Bool)
    (computed : strict_memory_frontend_parse.memory_type source token true=.ok (some (bytes,signed))) :
    ∃ text,(text,.Load bytes signed)∈opcodeNames ∧ TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.memory_type at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq,reduceCtorEq] at computed
  all_goals rcases computed with ⟨rfl,rfl⟩
  all_goals rename_i matched
  all_goals
    have meaning := source_token_equals source _ token matched
    first
    | solve
      | refine ⟨"ld.global.u8",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.u8"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 56#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.u16",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.u16"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 49#u8, 54#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.u32",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.u32"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.b32",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.b32"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 98#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.u64",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.u64"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.b64",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.b64"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 98#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.s8",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.s8"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 115#u8, 56#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.s16",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.s16"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 115#u8, 49#u8, 54#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"ld.global.s32",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "ld.global.s32"=[108#u8, 100#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 115#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

theorem source_store_type (source : Slice U8) (token : SourceToken) (bytes : U8) (signed : Bool)
    (computed : strict_memory_frontend_parse.memory_type source token false=.ok (some (bytes,signed))) :
    signed=false ∧ ∃ text,(text,.Store bytes)∈opcodeNames ∧ TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.memory_type at computed
  repeat' first
    | fail_if_no_progress simp only [lift,bind_ok,ite_true,Bool.false_eq_true,ite_false] at computed
    | peel_frontend_bool computed
  all_goals simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq,reduceCtorEq] at computed
  all_goals rcases computed with ⟨rfl,rfl⟩
  all_goals refine ⟨rfl,?_⟩
  all_goals rename_i matched
  all_goals
    have meaning := source_token_equals source _ token matched
    first
    | solve
      | refine ⟨"st.global.u8",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.u8"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 56#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"st.global.b8",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.b8"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 98#u8, 56#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"st.global.u16",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.u16"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 49#u8, 54#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"st.global.b16",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.b16"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 98#u8, 49#u8, 54#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"st.global.u32",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.u32"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"st.global.b32",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.b32"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 98#u8, 51#u8, 50#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"st.global.u64",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.u64"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | solve
      | refine ⟨"st.global.b64",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "st.global.b64"=[115#u8, 116#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 98#u8, 54#u8, 52#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

private theorem binary_opcode (source : Slice U8) (token : SourceToken) (bits : U8) (operation : ptxv_memory_write.value_flow.BinaryOperation)
    (computed : strict_memory_frontend_parse.parse_binary source token=.ok (some (bits,operation))) :
    ∃ text,(text,.Binary bits operation)∈opcodeNames ∧ TokenSpelling source.val token text := by
  obtain ⟨text,listed,meaning⟩ := source_binary source token bits operation computed
  have mapped : (text,strict_memory_frontend_parse.Opcode.Binary bits operation)∈binaryNames.map (fun (text,bits,operation) => (text,strict_memory_frontend_parse.Opcode.Binary bits operation)) :=
    List.mem_map.mpr ⟨(text,bits,operation),listed,rfl⟩
  refine ⟨text,?_,meaning⟩
  simp only [opcodeNames,List.mem_append]
  tauto

private theorem compare_opcode (source : Slice U8) (token : SourceToken) (bits : U8) (comparison : ptxv_memory_write.value_flow.Comparison)
    (computed : strict_memory_frontend_parse.parse_compare source token=.ok (some (bits,comparison))) :
    ∃ text,(text,.Compare bits comparison)∈opcodeNames ∧ TokenSpelling source.val token text := by
  obtain ⟨text,listed,meaning⟩ := source_compare source token bits comparison computed
  have mapped : (text,strict_memory_frontend_parse.Opcode.Compare bits comparison)∈comparisonNames.map (fun (text,bits,comparison) => (text,strict_memory_frontend_parse.Opcode.Compare bits comparison)) :=
    List.mem_map.mpr ⟨(text,bits,comparison),listed,rfl⟩
  refine ⟨text,?_,meaning⟩
  simp only [opcodeNames,List.mem_append]
  tauto

theorem source_classify (source : Slice U8) (token : SourceToken) (opcode : Opcode)
    (computed : strict_memory_frontend_parse.classify source token=.ok (some opcode)) :
    ∃ text,(text,opcode)∈opcodeNames ∧ TokenSpelling source.val token text := by
  unfold strict_memory_frontend_parse.classify at computed
  simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
  obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
  cases matched with
  | true =>
    simp only [ite_true,ok.injEq,Option.some.injEq] at computed
    cases computed
    have meaning := source_token_equals source _ token matchedComputed
    refine ⟨"ret",?_,?_⟩
    · simp [opcodeNames]
    · have literal : utf8 "ret"=[114#u8, 101#u8, 116#u8] := by decide
      simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
  | false =>
    simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
    obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
    cases matched with
    | true =>
      simp only [ite_true,ok.injEq,Option.some.injEq] at computed
      cases computed
      have meaning := source_token_equals source _ token matchedComputed
      refine ⟨"exit",?_,?_⟩
      · simp [opcodeNames]
      · have literal : utf8 "exit"=[101#u8, 120#u8, 105#u8, 116#u8] := by decide
        simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
    | false =>
      simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
      obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
      cases matched with
      | true =>
        simp only [ite_true,ok.injEq,Option.some.injEq] at computed
        cases computed
        have meaning := source_token_equals source _ token matchedComputed
        refine ⟨"bra",?_,?_⟩
        · simp [opcodeNames]
        · have literal : utf8 "bra"=[98#u8, 114#u8, 97#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
      | false =>
        simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
        obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
        cases matched with
        | true =>
          simp only [ite_true,ok.injEq,Option.some.injEq] at computed
          cases computed
          have meaning := source_token_equals source _ token matchedComputed
          refine ⟨"mov.u32",?_,?_⟩
          · simp [opcodeNames]
          · have literal : utf8 "mov.u32"=[109#u8, 111#u8, 118#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
            simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
        | false =>
          simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
          obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
          cases matched with
          | true =>
            simp only [ite_true,ok.injEq,Option.some.injEq] at computed
            cases computed
            have meaning := source_token_equals source _ token matchedComputed
            refine ⟨"mov.b32",?_,?_⟩
            · simp [opcodeNames]
            · have literal : utf8 "mov.b32"=[109#u8, 111#u8, 118#u8, 46#u8, 98#u8, 51#u8, 50#u8] := by decide
              simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
          | false =>
            simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
            obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
            cases matched with
            | true =>
              simp only [ite_true,ok.injEq,Option.some.injEq] at computed
              cases computed
              have meaning := source_token_equals source _ token matchedComputed
              refine ⟨"mov.u64",?_,?_⟩
              · simp [opcodeNames]
              · have literal : utf8 "mov.u64"=[109#u8, 111#u8, 118#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
                simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
            | false =>
              simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
              obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
              cases matched with
              | true =>
                simp only [ite_true,ok.injEq,Option.some.injEq] at computed
                cases computed
                have meaning := source_token_equals source _ token matchedComputed
                refine ⟨"mov.b64",?_,?_⟩
                · simp [opcodeNames]
                · have literal : utf8 "mov.b64"=[109#u8, 111#u8, 118#u8, 46#u8, 98#u8, 54#u8, 52#u8] := by decide
                  simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
              | false =>
                simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
                obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
                cases matched with
                | true =>
                  simp only [ite_true,ok.injEq,Option.some.injEq] at computed
                  cases computed
                  have meaning := source_token_equals source _ token matchedComputed
                  refine ⟨"cvta.global.u64",?_,?_⟩
                  · simp [opcodeNames]
                  · have literal : utf8 "cvta.global.u64"=[99#u8, 118#u8, 116#u8, 97#u8, 46#u8, 103#u8, 108#u8, 111#u8, 98#u8, 97#u8, 108#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
                    simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
                | false =>
                  simp only [Bool.false_eq_true,ite_false] at computed
                  obtain ⟨binary,binaryComputed,computed⟩ := source_bind_success _ _ _ computed
                  cases binary with
                  | some pair =>
                    rcases pair with ⟨bits,operation⟩
                    change ok (some (.Binary bits operation : Opcode))=ok (some opcode) at computed
                    simp only [ok.injEq,Option.some.injEq] at computed
                    cases computed
                    exact binary_opcode source token bits operation binaryComputed
                  | none =>
                    simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
                    obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
                    cases matched with
                    | true =>
                      simp only [ite_true,ok.injEq,Option.some.injEq] at computed
                      cases computed
                      have meaning := source_token_equals source _ token matchedComputed
                      refine ⟨"mul.wide.u32",?_,?_⟩
                      · simp [opcodeNames]
                      · have literal : utf8 "mul.wide.u32"=[109#u8, 117#u8, 108#u8, 46#u8, 119#u8, 105#u8, 100#u8, 101#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
                        simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
                    | false =>
                      simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
                      obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
                      cases matched with
                      | true =>
                        simp only [ite_true,ok.injEq,Option.some.injEq] at computed
                        cases computed
                        have meaning := source_token_equals source _ token matchedComputed
                        refine ⟨"cvt.u64.u32",?_,?_⟩
                        · simp [opcodeNames]
                        · have literal : utf8 "cvt.u64.u32"=[99#u8, 118#u8, 116#u8, 46#u8, 117#u8, 54#u8, 52#u8, 46#u8, 117#u8, 51#u8, 50#u8] := by decide
                          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
                      | false =>
                        simp only [lift,bind_ok,Bool.false_eq_true,ite_false] at computed
                        obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
                        cases matched with
                        | true =>
                          simp only [ite_true,ok.injEq,Option.some.injEq] at computed
                          cases computed
                          have meaning := source_token_equals source _ token matchedComputed
                          refine ⟨"cvt.u32.u64",?_,?_⟩
                          · simp [opcodeNames]
                          · have literal : utf8 "cvt.u32.u64"=[99#u8, 118#u8, 116#u8, 46#u8, 117#u8, 51#u8, 50#u8, 46#u8, 117#u8, 54#u8, 52#u8] := by decide
                            simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning
                        | false =>
                          simp only [Bool.false_eq_true,ite_false] at computed
                          obtain ⟨comparison,comparisonComputed,computed⟩ := source_bind_success _ _ _ computed
                          cases comparison with
                          | some pair =>
                            rcases pair with ⟨bits,operation⟩
                            change ok (some (.Compare bits operation : Opcode))=ok (some opcode) at computed
                            simp only [ok.injEq,Option.some.injEq] at computed
                            cases computed
                            exact compare_opcode source token bits operation comparisonComputed
                          | none =>
                            obtain ⟨load,loadComputed,computed⟩ := source_bind_success _ _ _ computed
                            cases load with
                            | some pair =>
                              rcases pair with ⟨bytes,signed⟩
                              change ok (some (.Load bytes signed : Opcode))=ok (some opcode) at computed
                              simp only [ok.injEq,Option.some.injEq] at computed
                              cases computed
                              exact source_load_type source token bytes signed loadComputed
                            | none =>
                              obtain ⟨store,storeComputed,computed⟩ := source_bind_success _ _ _ computed
                              cases store with
                              | some pair =>
                                rcases pair with ⟨bytes,signed⟩
                                change ok (some (.Store bytes : Opcode))=ok (some opcode) at computed
                                simp only [ok.injEq,Option.some.injEq] at computed
                                cases computed
                                exact (source_store_type source token bytes signed storeComputed).2
                              | none =>
                                simp at computed

end PtxFrontendProgram
