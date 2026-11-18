/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Instantiation
import FrontendProgram.GrammarGlobalNames
import FrontendProgram.SourceBufferMapping

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

private theorem initial_registers_unique (virtuals predicates : Nat)
    (left right : ptxv_memory_write.value_flow.Registers)
    (l : InitialRegisters virtuals predicates left) (r : InitialRegisters virtuals predicates right) : left=right := by
  obtain ⟨lr,lp,lw,lpr,lpd,lu⟩:=l
  obtain ⟨rr,rp,rw,rpr,rpd,ru⟩:=r
  have regs := lr.trans rr.symm
  have pointers := lp.trans rp.symm
  have written := lw.trans rw.symm
  have preds := lpr.trans rpr.symm
  have pdef := lpd.trans rpd.symm
  have vund := lu.trans ru.symm
  cases left with
  | mk leftRaw leftPointers =>
    cases right with
    | mk rightRaw rightPointers =>
      cases leftRaw
      cases rightRaw
      simp_all only [alloc.vec.Vec.eq_iff, ptxv_memory_write.machine_registers.LaneRegisters.mk.injEq,
        ptxv_memory_write.value_flow.Registers.mk.injEq]
      simp

private theorem dimensions_injective (left right : Array U32 3#usize)
    (same : dimensionsModel left=dimensionsModel right) : left=right := by
  obtain ⟨lx,ly,lz,ll⟩:=dimensions_three left
  obtain ⟨rx,ry,rz,rl⟩:=dimensions_three right
  have x:=congrArg PtxFrontend.Dimensions.x same
  have y:=congrArg PtxFrontend.Dimensions.y same
  have z:=congrArg PtxFrontend.Dimensions.z same
  simp only [dimensionsModel,ll,rl,List.getD_cons_zero,List.getD_cons_succ] at x y z
  have xe : lx=rx := by scalar_tac
  have ye : ly=ry := by scalar_tac
  have ze : lz=rz := by scalar_tac
  apply (Array.eq_iff left right).mpr
  simp [ll,rl,xe,ye,ze]

private theorem initial_thread_unique (manifest : Manifest) (virtuals predicates index : Nat)
    (left right : ptxv_memory_write.execution_core.ThreadState)
    (l : InitialThread manifest virtuals predicates index left)
    (r : InitialThread manifest virtuals predicates index right) : left=right := by
  obtain ⟨lpc,lrunning,lregs,lntid,lnctaid,lctaid,ltid,llane⟩:=l
  obtain ⟨rpc,rrunning,rregs,rntid,rnctaid,rctaid,rtid,rlane⟩:=r
  have pc : left.pc=right.pc := by scalar_tac
  have running:=lrunning.trans rrunning.symm
  have registers:=initial_registers_unique virtuals predicates left.registers right.registers lregs rregs
  have ntid:=lntid.trans rntid.symm
  have nctaid:=lnctaid.trans rnctaid.symm
  have ctaid:=dimensions_injective left.ids.ctaid right.ids.ctaid (lctaid.trans rctaid.symm)
  have tid:=dimensions_injective left.ids.tid right.ids.tid (ltid.trans rtid.symm)
  have lane : left.ids.lane=right.ids.lane := by scalar_tac
  cases left with
  | mk lp lr lreg lids =>
    cases right with
    | mk rp rr rreg rids =>
      cases lids
      cases rids
      simp_all

private theorem threads_unique (manifest : Manifest) (virtuals predicates : Nat)
    (left right : List ptxv_memory_write.execution_core.ThreadState)
    (l : ThreadsMeaning manifest virtuals predicates left)
    (r : ThreadsMeaning manifest virtuals predicates right) : left=right := by
  have length:=l.2.2.1.trans r.2.2.1.symm
  apply List.ext_getElem?
  intro index
  cases lat : left[index]? with
  | none =>
    have lower:=List.getElem?_eq_none_iff.mp lat
    exact (List.getElem?_eq_none (by omega : right.length≤ index)).symm
  | some lt =>
    have bound:=(List.getElem?_eq_some_iff.mp lat).1
    have rbound : index<right.length := by omega
    have rat:=List.getElem?_eq_getElem rbound
    rw [rat]
    congr 1
    exact initial_thread_unique manifest virtuals predicates index lt right[index] (l.2.2.2 index lt lat) (r.2.2.2 index right[index] rat)

private theorem backing_buffer_unique (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (inventory : BufferInventory source parsed manifest) (object : Nat)
    (left right : Buffer) (lm : left∈manifest.buffers.val) (rm : right∈manifest.buffers.val)
    (lb : BacksObject source parsed object left) (rb : BacksObject source parsed object right) : left=right := by
  obtain ⟨i,li⟩:=List.mem_iff_getElem?.mp lm
  obtain ⟨j,rj⟩:=List.mem_iff_getElem?.mp rm
  have ij:=inventory.2.2 i j object left right li rj lb rb
  subst j
  exact Option.some.inj (li.symm.trans rj)

private theorem initial_allocation_unique (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (inventory : BufferInventory source parsed manifest) (object : Nat)
    (left right : Allocation)
    (l : InitialAllocation source parsed manifest object left)
    (r : InitialAllocation source parsed manifest object right) : left=right := by
  obtain ⟨lg,lat,llive,li⟩:=l
  obtain ⟨rg,rat,rlive,ri⟩:=r
  have globals : lg=rg := Option.some.inj (lat.symm.trans rat)
  subst rg
  have bytes : left.bytes.val=right.bytes.val := by
    rcases li with lit|⟨lb,lm,lbk,lbytes,lflags⟩
    · rcases ri with rit|⟨rb,rm,rbk,rbytes,rflags⟩
      · exact lit.2.1.trans rit.2.1.symm
      · exact False.elim (lit.1 ⟨rb,rm,rbk⟩)
    · rcases ri with rit|⟨rb,rm,rbk,rbytes,rflags⟩
      · exact False.elim (rit.1 ⟨lb,lm,lbk⟩)
      · exact lbytes.trans rbytes.symm
  have flags : left.defined.val=right.defined.val := by
    rcases li with lit|⟨lb,lm,lbk,lbytes,lflags⟩
    · rcases ri with rit|⟨rb,rm,rbk,rbytes,rflags⟩
      · exact lit.2.2.trans rit.2.2.symm
      · exact False.elim (lit.1 ⟨rb,rm,rbk⟩)
    · rcases ri with rit|⟨rb,rm,rbk,rbytes,rflags⟩
      · exact False.elim (rit.1 ⟨lb,lm,lbk⟩)
      · have buffers:=backing_buffer_unique source parsed manifest inventory object lb rb lm rm lbk rbk
        subst rb
        exact lflags.trans rflags.symm
  cases left
  cases right
  simp_all only [alloc.vec.Vec.eq_iff, ptxv_memory_write.execution_core.AllocationBytes.mk.injEq]
  simp

private theorem named_buffer_unique (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (inventory : BufferInventory source parsed manifest)
    (left right : Buffer) (lm : left∈manifest.buffers.val) (rm : right∈manifest.buffers.val)
    (same : left.name=right.name) : left=right := by
  obtain ⟨i,li⟩:=List.mem_iff_getElem?.mp lm
  obtain ⟨j,rj⟩:=List.mem_iff_getElem?.mp rm
  have ib:=(List.getElem?_eq_some_iff.mp li).1
  have jb:=(List.getElem?_eq_some_iff.mp rj).1
  by_cases indices : i=j
  · subst j
    exact Option.some.inj (li.symm.trans rj)
  · rcases Nat.lt_or_gt_of_ne indices with before|after
    · have excluded:=inventory.2.1.rel_getElem_of_lt ib jb before
      rw [(List.getElem?_eq_some_iff.mp li).2,(List.getElem?_eq_some_iff.mp rj).2] at excluded
      exact False.elim (excluded same)
    · have excluded:=inventory.2.1.rel_getElem_of_lt jb ib after
      rw [(List.getElem?_eq_some_iff.mp rj).2,(List.getElem?_eq_some_iff.mp li).2] at excluded
      exact False.elim (excluded same.symm)

private theorem outputs_unique (source : List U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (names : GlobalNamesInjective source parsed) (inventory : BufferInventory source parsed manifest)
    (left right : List Usize) (l : OutputsMeaning source parsed manifest left)
    (r : OutputsMeaning source parsed manifest right) : left=right := by
  have length:=l.1.trans r.1.symm
  have mlength:=l.1
  apply List.ext_getElem?
  intro index
  cases lat : left[index]? with
  | none =>
    have lower:=List.getElem?_eq_none_iff.mp lat
    exact (List.getElem?_eq_none (by omega : right.length≤ index)).symm
  | some lo =>
    have bound:=(List.getElem?_eq_some_iff.mp lat).1
    have rbound : index<right.length := by omega
    have mbound : index<manifest.outputs.val.length := by omega
    have rat:=List.getElem?_eq_getElem rbound
    have mat:=List.getElem?_eq_getElem mbound
    obtain ⟨lb,lm,ln,lbk⟩:=l.2 index lo manifest.outputs.val[index] lat mat
    obtain ⟨rb,rm,rn,rbk⟩:=r.2 index right[index] manifest.outputs.val[index] rat mat
    have buffers:=named_buffer_unique source parsed manifest inventory lb rb lm rm (ln.trans rn.symm)
    subst rb
    have same:=backs_object_unique source parsed names lb lo.val right[index].val lbk rbk
    have equal : lo=right[index] := by scalar_tac
    rw [rat,equal]

private theorem construction_initial_unique (source entry : List U8) (manifest : Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (left right : Program)
    (l : ConstructionMeaning source entry manifest parsed left)
    (r : ConstructionMeaning source entry manifest parsed right) : left.initial=right.initial := by
  have threadsList:=threads_unique manifest parsed.scope.virtual_count.val parsed.scope.predicate_count.val
    left.initial.threads.val right.initial.threads.val l.2.2.2.2.2.2.2.1 r.2.2.2.2.2.2.2.1
  have threads : left.initial.threads=right.initial.threads := by simpa only [alloc.vec.Vec.eq_iff] using threadsList
  have lengths:=l.2.2.2.2.2.1.trans r.2.2.2.2.2.1.symm
  have allocationList : left.initial.allocations.val=right.initial.allocations.val := by
    apply List.ext_getElem?
    intro index
    cases lat : left.initial.allocations.val[index]? with
    | none =>
      have lower:=List.getElem?_eq_none_iff.mp lat
      exact (List.getElem?_eq_none (by omega : right.initial.allocations.val.length≤ index)).symm
    | some la =>
      have bound:=(List.getElem?_eq_some_iff.mp lat).1
      have rbound : index<right.initial.allocations.val.length := by omega
      have rat:=List.getElem?_eq_getElem rbound
      have equal:=initial_allocation_unique source parsed manifest l.2.2.1 index la right.initial.allocations.val[index]
        (l.2.2.2.2.2.2.1 index la lat) (r.2.2.2.2.2.2.1 index right.initial.allocations.val[index] rat)
      rw [rat,equal]
  have allocations : left.initial.allocations=right.initial.allocations := by simpa only [alloc.vec.Vec.eq_iff] using allocationList
  cases li : left.initial
  cases ri : right.initial
  simp_all

theorem construction_memory_instance (source entry : List U8) (manifest : Manifest)
    (parsed : strict_memory_frontend_parse.SourceProgram) (original alternate : Program)
    (grammar : ModuleMeaning source entry parsed)
    (left : ConstructionMeaning source entry manifest parsed original)
    (right : ConstructionMeaning source entry manifest parsed alternate) :
    MemorySafetyExecution.Native.MemoryInstance original alternate := by
  have l:=left.2.2.2.2.1
  have r:=right.2.2.2.2.1
  have initial:=construction_initial_unique source entry manifest parsed original alternate left right
  have names : GlobalNamesInjective source parsed := module_global_names_injective source entry parsed grammar
  have outputs:=outputs_unique source parsed manifest names left.2.2.1 original.outputs.val alternate.outputs.val
    left.2.2.2.2.2.2.2.2 right.2.2.2.2.2.2.2.2
  refine ⟨left.2.2.2.1.trans right.2.2.2.1.symm,?_,?_,?_⟩
  · simpa only [alloc.vec.Vec.eq_iff] using outputs
  · refine ⟨l.1.trans r.1.symm,?_,r.2.2⟩
    intro index old fresh oldAt freshAt
    have bound:=(List.getElem?_eq_some_iff.mp oldAt).1
    have globalBound : index<parsed.scope.globals.val.length := by simpa only [l.1] using bound
    have globalAt:=List.getElem?_eq_getElem globalBound
    obtain ⟨ol,olat,obytes,oalign,_,_⟩:=l.2.1 index parsed.scope.globals.val[index] globalAt
    obtain ⟨nl,nlat,nbytes,nalign,aligned,ending⟩:=r.2.1 index parsed.scope.globals.val[index] globalAt
    have oe : ol=old := Option.some.inj (olat.symm.trans oldAt)
    have ne : nl=fresh := Option.some.inj (nlat.symm.trans freshAt)
    subst ol
    subst nl
    refine ⟨?_,nalign.trans oalign.symm,?_⟩
    · scalar_tac
    · exact ⟨by simpa only [nalign] using aligned,ending⟩
  · rw [←initial]
    refine ⟨rfl,rfl,?_⟩
    intro index old fresh oldAt freshAt
    have same : old=fresh := Option.some.inj (oldAt.symm.trans freshAt)
    subst fresh
    exact ⟨rfl,rfl,rfl⟩

end PtxFrontendProgram
