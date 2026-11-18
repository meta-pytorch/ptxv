/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.LabelFreshness
import FrontendProgram.Header

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

abbrev CollectContinue := alloc.vec.Vec Label × Usize × U32 × Bool × Bool
abbrev CollectDone := alloc.vec.Vec Label × Bool × Bool

def countCodeToken (source : Slice U8) (tokens : Slice SourceToken) (labels : alloc.vec.Vec Label)
    (position next : Usize) (pc : U32) : Result (ControlFlow CollectContinue CollectDone) := do
  let spelling := Array.to_slice (Array.make 1#usize [59#u8])
  let matched ← strict_memory_frontend_parse.matches source tokens position spelling
  let (nextPc,valid) ← if matched then do
    match U32.checked_add pc 1#u32 with
    | none => .ok (pc,false)
    | some nextPc => .ok (nextPc,true)
    else .ok (pc,true)
  .ok (.cont (labels,next,nextPc,matched,valid))

def countCodeTokenAdded (source : Slice U8) (tokens : Slice SourceToken) (labels : alloc.vec.Vec Label)
    (position : Usize) (pc : U32) : Result (ControlFlow CollectContinue CollectDone) := do
  let spelling := Array.to_slice (Array.make 1#usize [59#u8])
  let matched ← strict_memory_frontend_parse.matches source tokens position spelling
  let (nextPc,valid) ← if matched then do
    match U32.checked_add pc 1#u32 with
    | none => .ok (pc,false)
    | some nextPc => .ok (nextPc,true)
    else .ok (pc,true)
  let next ← position+1#usize
  .ok (.cont (labels,next,nextPc,matched,valid))

theorem count_code_token_shape (source : Slice U8) (tokens : Slice SourceToken) (labels : alloc.vec.Vec Label)
    (position next : Usize) (pc : U32) (outcome : ControlFlow CollectContinue CollectDone)
    (computed : countCodeToken source tokens labels position next pc=.ok outcome) :
    ∃ nextPc matched valid,outcome=.cont (labels,next,nextPc,matched,valid) := by
  unfold countCodeToken at computed
  obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨pair,countComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨nextPc,valid⟩
  have same : (ControlFlow.cont (labels,next,nextPc,matched,valid) : ControlFlow CollectContinue CollectDone)=outcome := by simpa using computed
  exact ⟨nextPc,matched,valid,same.symm⟩

theorem count_code_token_added_shape (source : Slice U8) (tokens : Slice SourceToken) (labels : alloc.vec.Vec Label)
    (position : Usize) (pc : U32) (outcome : ControlFlow CollectContinue CollectDone)
    (computed : countCodeTokenAdded source tokens labels position pc=.ok outcome) :
    ∃ next nextPc matched valid,outcome=.cont (labels,next,nextPc,matched,valid) ∧ next.val=position.val+1 := by
  unfold countCodeTokenAdded at computed
  obtain ⟨matched,matchedComputed,computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨pair,countComputed,computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨nextPc,valid⟩
  obtain ⟨next,nextAdded,computed⟩ := source_bind_success _ _ _ computed
  simp only [ok.injEq] at computed
  exact ⟨next,nextPc,matched,valid,computed.symm,by simpa using source_usize_add_value position 1#usize next nextAdded⟩

end PtxFrontendProgram
