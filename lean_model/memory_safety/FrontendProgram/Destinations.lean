/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.RegisterLookup
import FrontendProgram.Header

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

theorem source_parse_destination (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (bits : U8) (predicate : Bool) (destination : U16) (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_destination source tokens position scope bits predicate=.ok (some (destination,finish))) :
    RegisterAt source.val scope tokens.val position.val bits.val predicate destination ∧
    TokenAt source.val tokens.val (position.val+1) "," ∧ finish.val=position.val+2 := by
  unfold strict_memory_frontend_parse.parse_destination at computed
  dsimp only at computed
  split at computed
  · simp at computed
  · obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
    obtain ⟨register,registerComputed,computed⟩ := source_bind_success _ _ _ computed
    cases register with
    | none => simp at computed
    | some index =>
      obtain ⟨commaPosition,commaAdded,computed⟩ := source_bind_success _ _ _ computed
      simp only [lift,bind_ok] at computed
      obtain ⟨after,commaComputed,computed⟩ := source_bind_success _ _ _ computed
      cases after with
      | none => simp at computed
      | some next =>
        simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
        rcases computed with ⟨rfl,rfl⟩
        have commaEq := source_usize_add_value position 1#usize commaPosition commaAdded
        obtain ⟨finishEq,comma,found,meaning⟩ := source_take source tokens commaPosition next _ commaComputed
        refine ⟨⟨token,source_slice_index tokens position token tokenComputed,
          source_lookup_register source scope token bits predicate index counts registerComputed⟩,?_,by scalar_tac⟩
        refine ⟨comma,?_,?_⟩
        · simpa [commaEq] using found
        · have literal : utf8 ","=[44#u8] := by decide
          simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning

end PtxFrontendProgram
