/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementEvents
import MemorySafetyExecution.TraceSafety

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open MemorySafetyExecution

def RecordPlacement (leftLayouts rightLayouts : Layouts) (left right : MemorySafety.AccessRecord) : Prop :=
  ∃ actor first second,left=accessRecord actor first ∧ right=accessRecord actor second ∧
    Nonempty (RequestPlacement leftLayouts rightLayouts first second)

private theorem record_model (actor : Usize) (request : memory_safety.Request) :
    MemorySafety.accessRecord actor.val (MemorySafety.Source.requestModel request)=accessRecord actor request := by
  cases kind : request.kind <;> simp [MemorySafety.accessRecord,MemorySafety.Source.requestModel,
    MemorySafety.Source.kindModel,MemorySafety.AccessKind.isWrite,accessRecord,kind]

private theorem forall₂_member_left {α β : Type} {relation : α→β→Prop} {left : List α} {right : List β}
    (related : List.Forall₂ relation left right) (item : β) (member : item∈right) :
    ∃ before∈left,relation before item := by
  induction related with
  | nil => cases member
  | @cons first second left right pair tail ih =>
    rcases List.mem_cons.mp member with same|later
    · subst item
      exact ⟨first,List.mem_cons_self,pair⟩
    · obtain ⟨before,inside,related⟩:=ih later
      exact ⟨before,List.mem_cons_of_mem _ inside,related⟩

private theorem pairwise_transport {α β : Type} {relation : α→β→Prop} {left : List α} {right : List β}
    (related : List.Forall₂ relation left right) (leftPair : α→α→Prop) (rightPair : β→β→Prop)
    (transfer : ∀ a b c d,relation a c → relation b d → leftPair a b → rightPair c d)
    (pairs : left.Pairwise leftPair) : right.Pairwise rightPair := by
  induction related with
  | nil => exact .nil
  | @cons first second left right pair tail ih =>
    obtain ⟨all,remaining⟩:=List.pairwise_cons.mp pairs
    apply List.pairwise_cons.mpr
    refine ⟨?_,ih remaining⟩
    intro other member
    obtain ⟨before,inside,related⟩:=forall₂_member_left tail other member
    exact transfer first before second other pair related (all before inside)

theorem placement_event_records (leftLayouts rightLayouts : Layouts)
    {left right : List memory_safety.Event}
    (related : List.Forall₂ (EventPlacement leftLayouts rightLayouts) left right) :
    List.Forall₂ (RecordPlacement leftLayouts rightLayouts)
      ((left.map MemorySafety.Source.eventModel).flatMap readWriteRecords)
      ((right.map MemorySafety.Source.eventModel).flatMap readWriteRecords) := by
  induction related with
  | nil => exact .nil
  | @cons first second left right pair tail ih =>
    cases pair with
    | control => simpa only [List.map_cons,List.flatMap_cons,MemorySafety.Source.eventModel,
        readWriteRecords,List.nil_append] using ih
    | access actor first second request =>
      simp only [List.map_cons,List.flatMap_cons,MemorySafety.Source.eventModel,readWriteRecords,
        List.singleton_append]
      exact .cons ⟨actor,first,second,record_model actor first,record_model actor second,⟨request⟩⟩ ih

theorem placement_trace_no_conflicts (leftLayouts rightLayouts : Layouts)
    (leftSeparated : SeparatedLayouts leftLayouts) (rightSeparated : SeparatedLayouts rightLayouts)
    {left right : List memory_safety.Event}
    (related : List.Forall₂ (EventPlacement leftLayouts rightLayouts) left right)
    (safe : ((left.map MemorySafety.Source.eventModel).flatMap readWriteRecords).Pairwise
      (fun before after=>¬MemorySafety.Conflict before after)) :
    ((right.map MemorySafety.Source.eventModel).flatMap readWriteRecords).Pairwise
      (fun before after=>¬MemorySafety.Conflict before after) := by
  apply pairwise_transport (placement_event_records leftLayouts rightLayouts related) _ _ ?_ safe
  intro first second otherFirst otherSecond firstRelated secondRelated clean
  obtain ⟨firstActor,firstRequest,otherFirstRequest,rfl,rfl,⟨firstPlacement⟩⟩:=firstRelated
  obtain ⟨secondActor,secondRequest,otherSecondRequest,rfl,rfl,⟨secondPlacement⟩⟩:=secondRelated
  exact fun conflict=>clean ((placement_access_conflict leftLayouts rightLayouts firstActor secondActor
    firstRequest otherFirstRequest secondRequest otherSecondRequest firstPlacement secondPlacement
    leftSeparated rightSeparated).mpr conflict)

end MemorySafetyExecution.Native
