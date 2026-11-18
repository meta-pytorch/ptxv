/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceObject
import FrontendProgram.SourceLoop

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

private theorem append_single_lookup {α : Type} (values : List α) (last value : α) (index : Nat)
    (lookup : (values++[last])[index]?=some value) :
    (values[index]?=some value ∧ index<values.length) ∨ (index=values.length ∧ last=value) := by
  by_cases inside : index<values.length
  · left
    exact ⟨by simpa only [List.getElem?_append,inside,ite_true] using lookup,inside⟩
  · have tail : [last][index-values.length]?=some value := by
      simpa only [List.getElem?_append,inside,ite_false] using lookup
    cases difference : index-values.length with
    | zero =>
      right
      exact ⟨by omega,by simpa only [difference,List.getElem?_cons_zero,Option.some.injEq] using tail⟩
    | succ rest => simp only [difference,List.getElem?_cons_succ,List.getElem?_nil] at tail; cases tail

private def LayoutForGlobal (global : Global) (layout : Layout) : Prop :=
  layout.bytes.val=global.bytes.val.length ∧ layout.alignment=global.alignment ∧
    global.alignment.val ∣ layout.base.val ∧ layout.base.val+layout.bytes.val≤2^64

private def DisjointLayouts (left right : Layout) : Prop :=
  left.base.val+left.bytes.val≤right.base.val ∨ right.base.val+right.bytes.val≤left.base.val

private structure AllocationPrefix (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (count cursor : Nat) (layouts : List Layout) (allocations : List Allocation) : Prop where
  layoutsLength : layouts.length=count
  allocationsLength : allocations.length=count
  layoutEntries : ∀ (object : Nat) (layout : Layout), layouts[object]?=some layout →
    ∃ global,parsed.scope.globals.val[object]?=some global ∧ LayoutForGlobal global layout
  disjoint : layouts.Pairwise DisjointLayouts
  below : ∀ layout∈layouts,layout.base.val+layout.bytes.val≤cursor
  allocationEntries : ∀ (object : Nat) (allocation : Allocation),allocations[object]?=some allocation →
    InitialAllocation source parsed manifest object allocation
  unique : ∀ object,object<count → UniqueBacking source parsed manifest object

private theorem allocation_prefix_append (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (count cursor : Nat) (layouts : List Layout) (allocations : List Allocation)
    (layout : Layout) (allocation : Allocation) (global : Global) (next : Nat)
    (inventory : AllocationPrefix source parsed manifest count cursor layouts allocations)
    (globalAt : parsed.scope.globals.val[count]?=some global)
    (layoutMeaning : LayoutForGlobal global layout) (afterCursor : cursor≤layout.base.val)
    (nextEq : next=layout.base.val+layout.bytes.val)
    (allocationMeaning : InitialAllocation source parsed manifest count allocation)
    (unique : UniqueBacking source parsed manifest count) :
    AllocationPrefix source parsed manifest (count+1) next (layouts++[layout]) (allocations++[allocation]) := by
  refine ⟨by simpa using inventory.layoutsLength,by simpa using inventory.allocationsLength,?_,?_,?_,?_,?_⟩
  · intro object selected lookup
    rcases append_single_lookup layouts layout selected object lookup with prior|last
    · exact inventory.layoutEntries object selected prior.1
    · rcases last with ⟨atEnd,same⟩
      subst selected
      exact ⟨global,by simpa only [atEnd,inventory.layoutsLength] using globalAt,layoutMeaning⟩
  · apply List.pairwise_append.mpr
    refine ⟨inventory.disjoint,by simp,?_⟩
    intro left leftMember right rightMember
    have same : right=layout := by simpa only [List.mem_singleton] using rightMember
    subst right
    exact Or.inl (Nat.le_trans (inventory.below left leftMember) afterCursor)
  · intro selected member
    rcases List.mem_append.mp member with old|last
    · have before:=inventory.below selected old
      omega
    · have same : selected=layout := by simpa only [List.mem_singleton] using last
      subst selected
      omega
  · intro object selected lookup
    rcases append_single_lookup allocations allocation selected object lookup with prior|last
    · exact inventory.allocationEntries object selected prior.1
    · rcases last with ⟨atEnd,same⟩
      subst selected
      simpa only [atEnd,inventory.allocationsLength] using allocationMeaning
  · intro object inside
    by_cases old : object<count
    · exact inventory.unique object old
    · have same : object=count := by omega
      simpa only [same] using unique

private theorem initial_allocations_loop (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (names : GlobalNamesInjective source.val parsed)
    (layouts : alloc.vec.Vec Layout) (allocations : alloc.vec.Vec Allocation) (position : Usize)
    (base : U64) (valid : Bool) (total : U64)
    (bound : position.val≤parsed.scope.globals.val.length)
    (inventory : valid=true → AllocationPrefix source.val parsed manifest position.val base.val layouts.val allocations.val)
    (output : alloc.vec.Vec Layout × alloc.vec.Vec Allocation × Bool)
    (computed : strict_memory_frontend_launch.initial_allocations_loop source parsed manifest layouts allocations position base valid total=.ok output) :
    output.2.2=true → ∃ cursor,AllocationPrefix source.val parsed manifest parsed.scope.globals.val.length cursor output.1.val output.2.1.val := by
  unfold strict_memory_frontend_launch.initial_allocations_loop at computed
  apply source_loop_success _ (fun current=>parsed.scope.globals.val.length-current.2.2.1.val)
    (fun current=>current.2.2.1.val≤parsed.scope.globals.val.length ∧
      (current.2.2.2.2.1=true → AllocationPrefix source.val parsed manifest current.2.2.1.val current.2.2.2.1.val current.1.val current.2.1.val))
    (fun output=>output.2.2=true → ∃ cursor,AllocationPrefix source.val parsed manifest parsed.scope.globals.val.length cursor output.1.val output.2.1.val)
    ?_ (layouts,allocations,position,base,valid,total) output ⟨bound,inventory⟩ computed
  rintro ⟨layouts,allocations,position,base,valid,total⟩ flow ⟨bound,inventory⟩ computed
  unfold strict_memory_frontend_launch.initial_allocations_loop.body at computed
  cases valid with
  | false =>
    simp only [Bool.false_eq_true,ite_false,ok.injEq] at computed
    subst flow
    simp
  | true =>
    simp only [ite_true] at computed
    by_cases inside : position < alloc.vec.Vec.len parsed.scope.globals
    · simp only [inside,ite_true] at computed
      obtain ⟨initialized,initializedComputed,computed⟩:=source_bind_success _ _ _ computed
      obtain ⟨parts,assembled,computed⟩:=source_bind_success _ _ _ computed
      obtain ⟨nextPosition,increment,computed⟩:=source_bind_success _ _ _ computed
      have nextPositionVal : nextPosition.val=position.val+1 := by
        have room:=alloc.vec.Vec.len_ineq parsed.scope.globals
        have law:=Usize.add_spec (x:=position) (y:=1#usize) (by scalar_tac)
        rw [increment] at law
        simpa only [WP.spec_ok,UScalar.ofNatCore_val_eq] using law
      simp only [ok.injEq] at computed
      subst flow
      refine ⟨⟨by scalar_tac,?_⟩,by scalar_tac⟩
      intro isValid
      cases initialized with
      | none =>
        simp only [ok.injEq] at assembled
        subst parts
        cases isValid
      | some initialized =>
        obtain ⟨nextLayouts,layoutsPushed,assembled⟩:=source_bind_success _ _ _ assembled
        obtain ⟨nextAllocations,allocationsPushed,assembled⟩:=source_bind_success _ _ _ assembled
        have layoutsEq:=vec_push_success layouts nextLayouts initialized.layout layoutsPushed
        have allocationsEq:=vec_push_success allocations nextAllocations initialized.allocation allocationsPushed
        simp only [ok.injEq] at assembled
        subst parts
        obtain ⟨global,globalAt,bytesEq,alignmentEq,positive,divides,before,nextEq,nextBound,totalEq,totalBound,allocationMeaning,unique⟩:=
          source_initial_object source parsed manifest position base total initialized names initializedComputed
        rw [nextPositionVal,layoutsEq,allocationsEq]
        apply allocation_prefix_append _ _ _ _ _ _ _ _ _ global _ (inventory rfl) globalAt
          ⟨bytesEq,alignmentEq,divides,?_⟩ before nextEq allocationMeaning unique
        rw [←nextEq]
        have representable := initialized.next.hBounds
        scalar_tac
    · simp only [inside,ite_false,ok.injEq] at computed
      subst flow
      intro _
      have atEnd : position.val=parsed.scope.globals.val.length := by scalar_tac
      exact ⟨base.val,by simpa only [atEnd] using inventory rfl⟩

private theorem allocation_prefix_complete (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (cursor : Nat) (layouts : List Layout) (allocations : List Allocation)
    (inventory : AllocationPrefix source parsed manifest parsed.scope.globals.val.length cursor layouts allocations) :
    LayoutsMeaning parsed.scope.globals.val layouts ∧ allocations.length=parsed.scope.globals.val.length ∧
      (∀ (object : Nat) (allocation : Allocation),allocations[object]?=some allocation → InitialAllocation source parsed manifest object allocation) ∧
      ∀ object,object<parsed.scope.globals.val.length → UniqueBacking source parsed manifest object := by
  refine ⟨⟨inventory.layoutsLength,?_,inventory.disjoint⟩,inventory.allocationsLength,inventory.allocationEntries,inventory.unique⟩
  intro object global globalAt
  have inside : object<layouts.length := by
    rw [inventory.layoutsLength]
    exact (List.getElem?_eq_some_iff.mp globalAt).1
  let layout:=layouts[object]
  have layoutAt : layouts[object]?=some layout:=List.getElem?_eq_getElem inside
  obtain ⟨actual,actualAt,meaning⟩:=inventory.layoutEntries object layout layoutAt
  have same : actual=global := by simpa only [globalAt,Option.some.injEq] using actualAt.symm
  subst actual
  exact ⟨layout,layoutAt,meaning⟩

theorem source_initial_allocations (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (layouts : alloc.vec.Vec Layout) (allocations : alloc.vec.Vec Allocation)
    (names : GlobalNamesInjective source.val parsed)
    (computed : strict_memory_frontend_launch.initial_allocations source parsed manifest=.ok (some (layouts,allocations))) :
    LayoutsMeaning parsed.scope.globals.val layouts.val ∧ allocations.val.length=parsed.scope.globals.val.length ∧
      (∀ (object : Nat) (allocation : Allocation),allocations.val[object]?=some allocation → InitialAllocation source.val parsed manifest object allocation) ∧
      ∀ object,object<parsed.scope.globals.val.length → UniqueBacking source.val parsed manifest object := by
  unfold strict_memory_frontend_launch.initial_allocations at computed
  obtain ⟨result,walked,computed⟩:=source_bind_success _ _ _ computed
  have empty : AllocationPrefix source.val parsed manifest 0 123145302310912
      (alloc.vec.Vec.new Layout).val (alloc.vec.Vec.new Allocation).val := by
    refine ⟨by simp,by simp,?_,by simp,?_,?_,?_⟩
    · intro object layout impossible; simp at impossible
    · intro layout impossible; simp at impossible
    · intro object allocation impossible; simp at impossible
    · intro object impossible; omega
  have law:=initial_allocations_loop source parsed manifest names (alloc.vec.Vec.new Layout)
    (alloc.vec.Vec.new Allocation) 0#usize 123145302310912#u64 true 0#u64 (by simp)
    (by intro _; exact empty) result walked
  rcases result with ⟨builtLayouts,builtAllocations,valid⟩
  change (if valid=true then Result.ok (some (builtLayouts,builtAllocations)) else Result.ok none)=Result.ok (some (layouts,allocations)) at computed
  cases valid with
  | false => simp at computed
  | true =>
    have equal : builtLayouts=layouts ∧ builtAllocations=allocations := by simpa only [ite_true,ok.injEq,Option.some.injEq,Prod.mk.injEq] using computed
    rcases equal with ⟨rfl,rfl⟩
    obtain ⟨cursor,inventory⟩:=law rfl
    exact allocation_prefix_complete source.val parsed manifest cursor builtLayouts.val builtAllocations.val inventory

end PtxFrontendProgram
