/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Resolution
import MemorySafetyExecution.Native.RaceSafety

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

abbrev Layouts := alloc.vec.Vec value_flow.AllocationLayout

def SeparatedLayouts (layouts : Layouts) : Prop :=
  ∀ (firstIndex secondIndex : Nat) (first second : value_flow.AllocationLayout),
    layouts.val[firstIndex]?=some first → layouts.val[secondIndex]?=some second →
    firstIndex≠secondIndex → first.base.val+first.bytes.val≤second.base.val ∨
      second.base.val+second.bytes.val≤first.base.val

structure RequestPlacement (leftLayouts rightLayouts : Layouts) (left right : memory_safety.Request) where
  origin : Nat
  offset : Nat
  leftLayout : value_flow.AllocationLayout
  rightLayout : value_flow.AllocationLayout
  leftSelected : leftLayouts.val[origin]?=some leftLayout
  rightSelected : rightLayouts.val[origin]?=some rightLayout
  objectBytes : leftLayout.bytes=rightLayout.bytes
  objectAlignment : leftLayout.alignment=rightLayout.alignment
  leftAddress : left.address.val=leftLayout.base.val+offset
  rightAddress : right.address.val=rightLayout.base.val+offset
  width : left.width=right.width
  alignment : left.alignment=right.alignment
  kind : left.kind=right.kind
  positive : 0<left.width.val
  objectBound : offset+left.width.val≤leftLayout.bytes.val
  alignmentPositive : 0 < left.alignment.val
  alignmentObject : left.alignment.val ∣ leftLayout.alignment.val
  alignmentOffset : left.alignment.val ∣ offset

inductive EventPlacement (leftLayouts rightLayouts : Layouts) : memory_safety.Event → memory_safety.Event → Prop where
  | control : EventPlacement leftLayouts rightLayouts .Control .Control
  | access (actor : Usize) (left right : memory_safety.Request)
      (placement : RequestPlacement leftLayouts rightLayouts left right) :
      EventPlacement leftLayouts rightLayouts (.Access actor left) (.Access actor right)

def eventActor : memory_safety.Event → Option Nat
  | .Access actor _ => some actor.val
  | .Free actor _ => some actor.val
  | .Control => none

def eventRequest : memory_safety.Event → Option memory_safety.Request
  | .Access _ request => some request
  | _ => none

theorem placement_event_actor {leftLayouts rightLayouts : Layouts} {left right : memory_safety.Event}
    (placement : EventPlacement leftLayouts rightLayouts left right) : eventActor left=eventActor right := by
  cases placement <;> rfl

theorem placement_event_request {leftLayouts rightLayouts : Layouts} {left right : memory_safety.Event}
    (placement : EventPlacement leftLayouts rightLayouts left right) :
    ∀ request, eventRequest left=some request →
      ∃ other, eventRequest right=some other ∧ Nonempty (RequestPlacement leftLayouts rightLayouts request other) := by
  cases placement with
  | control => intro request selected; cases selected
  | access actor request other related =>
    intro selected found
    have same : request=selected := by simpa only [eventRequest,Option.some.injEq] using found
    subst selected
    exact ⟨other,rfl,⟨related⟩⟩

theorem placement_request_overlap (leftLayouts rightLayouts : Layouts)
    (first otherFirst second otherSecond : memory_safety.Request)
    (firstPlacement : RequestPlacement leftLayouts rightLayouts first otherFirst)
    (secondPlacement : RequestPlacement leftLayouts rightLayouts second otherSecond)
    (leftSeparated : SeparatedLayouts leftLayouts) (rightSeparated : SeparatedLayouts rightLayouts) :
    MemorySafety.Overlap first.address.val first.width.val second.address.val second.width.val ↔
      MemorySafety.Overlap otherFirst.address.val otherFirst.width.val otherSecond.address.val otherSecond.width.val := by
  by_cases sameOrigin : firstPlacement.origin=secondPlacement.origin
  · have sameLeft : firstPlacement.leftLayout=secondPlacement.leftLayout := by
      have selected:=firstPlacement.leftSelected
      rw [sameOrigin,secondPlacement.leftSelected] at selected
      simpa only [Option.some.injEq] using selected.symm
    have sameRight : firstPlacement.rightLayout=secondPlacement.rightLayout := by
      have selected:=firstPlacement.rightSelected
      rw [sameOrigin,secondPlacement.rightSelected] at selected
      simpa only [Option.some.injEq] using selected.symm
    unfold MemorySafety.Overlap
    rw [firstPlacement.leftAddress,secondPlacement.leftAddress,firstPlacement.rightAddress,secondPlacement.rightAddress,
      ←firstPlacement.width,←secondPlacement.width,sameLeft,sameRight]
    omega
  · have leftDisjoint:=leftSeparated firstPlacement.origin secondPlacement.origin firstPlacement.leftLayout
      secondPlacement.leftLayout firstPlacement.leftSelected secondPlacement.leftSelected sameOrigin
    have rightDisjoint:=rightSeparated firstPlacement.origin secondPlacement.origin firstPlacement.rightLayout
      secondPlacement.rightLayout firstPlacement.rightSelected secondPlacement.rightSelected sameOrigin
    have leftNot : ¬MemorySafety.Overlap first.address.val first.width.val second.address.val second.width.val := by
      unfold MemorySafety.Overlap
      have firstBound:=firstPlacement.objectBound
      have secondBound:=secondPlacement.objectBound
      rw [firstPlacement.leftAddress,secondPlacement.leftAddress]
      rcases leftDisjoint with before|after <;> omega
    have rightNot : ¬MemorySafety.Overlap otherFirst.address.val otherFirst.width.val otherSecond.address.val otherSecond.width.val := by
      unfold MemorySafety.Overlap
      have firstBound:=firstPlacement.objectBound
      have secondBound:=secondPlacement.objectBound
      have firstBytes:=firstPlacement.objectBytes
      have secondBytes:=secondPlacement.objectBytes
      rw [firstPlacement.rightAddress,secondPlacement.rightAddress,←firstPlacement.width,←secondPlacement.width]
      rw [firstBytes] at firstBound
      rw [secondBytes] at secondBound
      rcases rightDisjoint with before|after <;> omega
    exact iff_of_false leftNot rightNot

theorem placement_access_conflict (leftLayouts rightLayouts : Layouts)
    (firstActor secondActor : Usize) (first otherFirst second otherSecond : memory_safety.Request)
    (firstPlacement : RequestPlacement leftLayouts rightLayouts first otherFirst)
    (secondPlacement : RequestPlacement leftLayouts rightLayouts second otherSecond)
    (leftSeparated : SeparatedLayouts leftLayouts) (rightSeparated : SeparatedLayouts rightLayouts) :
    MemorySafety.Conflict (accessRecord firstActor first) (accessRecord secondActor second) ↔
      MemorySafety.Conflict (accessRecord firstActor otherFirst) (accessRecord secondActor otherSecond) := by
  have overlaps:=placement_request_overlap leftLayouts rightLayouts first otherFirst second otherSecond
    firstPlacement secondPlacement leftSeparated rightSeparated
  cases firstKind:first.kind <;> cases secondKind:second.kind <;>
    simp only [MemorySafety.Conflict,accessRecord,firstKind,secondKind,
      ←firstPlacement.kind,←secondPlacement.kind] <;> exact and_congr_right (fun _ => and_congr_right (fun _ => overlaps))

theorem placement_request_alignment (leftLayouts rightLayouts : Layouts)
    (left right : memory_safety.Request)
    (placement : RequestPlacement leftLayouts rightLayouts left right)
    (legal : ValueFlow.LegalPlacement
      {bytes:=placement.rightLayout.bytes.val,alignment:=placement.rightLayout.alignment.val}
      placement.rightLayout.base.val) :
    0 < right.alignment.val ∧ right.alignment.val ∣ right.address.val := by
  rw [←placement.alignment,placement.rightAddress]
  refine ⟨placement.alignmentPositive, Nat.dvd_add ?_ placement.alignmentOffset⟩
  exact dvd_trans (by rw [←placement.objectAlignment]; exact placement.alignmentObject) legal.1

end MemorySafetyExecution.Native
