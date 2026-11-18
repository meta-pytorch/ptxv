/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.AddressOffsets
import FrontendProgram.Operands

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

theorem source_parse_address_tail (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (index : Operand) (offset : I64) (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_address_tail source tokens position scope=.ok (some (index,offset,finish))) :
    AddressTail source.val scope tokens.val position.val finish.val index offset.val := by
  unfold strict_memory_frontend_parse.parse_address_tail at computed
  obtain ⟨offsetResult,offsetComputed,computed⟩ := source_bind_success _ _ _ computed
  cases offsetResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨encoded,next⟩
    change ok (some (.Immediate 0#u64,encoded,next))=ok (some (index,offset,finish)) at computed
    simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
    rcases computed with ⟨rfl,rfl,rfl⟩
    exact AddressTail.offset (source_parse_address_offset source tokens position next encoded offsetComputed)

theorem source_parse_address (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (address : ptxv_memory_write.value_flow.Address) (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_address source tokens position scope=.ok (some (address,finish))) :
    AddressSyntax source.val scope tokens.val position.val finish.val address := by
  unfold strict_memory_frontend_parse.parse_address at computed
  simp only [lift,bind_ok] at computed
  obtain ⟨openResult,openComputed,computed⟩ := source_bind_success _ _ _ computed
  cases openResult with
  | none => simp at computed
  | some basePosition =>
    dsimp only at computed
    split at computed
    · simp at computed
    · obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨baseResult,baseComputed,computed⟩ := source_bind_success _ _ _ computed
      cases baseResult with
      | none => simp at computed
      | some base =>
        obtain ⟨tailPosition,tailAdded,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨tailResult,tailComputed,computed⟩ := source_bind_success _ _ _ computed
        cases tailResult with
        | none => simp at computed
        | some triple =>
          rcases triple with ⟨index,offset,closePosition⟩
          simp only [lift,bind_ok] at computed
          obtain ⟨closeResult,closeComputed,computed⟩ := source_bind_success _ _ _ computed
          cases closeResult with
          | none => simp at computed
          | some next =>
            simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
            rcases computed with ⟨rfl,rfl⟩
            obtain ⟨baseEq,openToken,openFound,openMeaning⟩ := source_take source tokens position basePosition _ openComputed
            obtain ⟨finishEq,closeToken,closeFound,closeMeaning⟩ := source_take source tokens closePosition next _ closeComputed
            have tailEq := source_usize_add_value basePosition 1#usize tailPosition tailAdded
            have tailPositionEq : tailPosition.val=position.val+2 := by scalar_tac
            apply AddressSyntax.address offset
            · have literal : utf8 "["=[91#u8] := by decide
              exact ⟨openToken,openFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using openMeaning⟩
            · simpa [baseEq] using source_slice_index tokens basePosition token tokenComputed
            · exact source_operand source scope token 64#u8 base counts baseComputed
            · simpa [tailPositionEq] using source_parse_address_tail source tokens tailPosition closePosition scope index offset counts tailComputed
            · rfl
            · have literal : utf8 "]"=[93#u8] := by decide
              exact ⟨closeToken,closeFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using closeMeaning⟩
            · exact finishEq

end PtxFrontendProgram
