/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceAllocation

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

private theorem rounded_up (cursor alignment : Nat) (positive : 0<alignment) :
    cursor≤((cursor+alignment-1)/alignment)*alignment := by
  have remainder:=Nat.mod_lt (cursor+alignment-1) positive
  have division:=Nat.mod_add_div (cursor+alignment-1) alignment
  rw [Nat.mul_comm alignment] at division
  omega

theorem source_initial_object (source : Slice U8) (parsed : strict_memory_frontend_parse.SourceProgram)
    (manifest : Manifest) (object : Usize) (cursor total : U64)
    (initialized : strict_memory_frontend_launch.InitializedObject)
    (names : GlobalNamesInjective source.val parsed)
    (computed : strict_memory_frontend_launch.initial_object source parsed manifest object cursor total=.ok (some initialized)) :
    ∃ global, parsed.scope.globals.val[object.val]?=some global ∧
      initialized.layout.bytes.val=global.bytes.val.length ∧
      initialized.layout.alignment=global.alignment ∧
      0<global.alignment.val ∧ global.alignment.val ∣ initialized.layout.base.val ∧
      cursor.val ≤ initialized.layout.base.val ∧
      initialized.next.val=initialized.layout.base.val+initialized.layout.bytes.val ∧
      initialized.next.val≤U64.max ∧
      initialized.total.val=total.val+initialized.layout.bytes.val ∧ initialized.total.val≤67108864 ∧
      InitialAllocation source.val parsed manifest object.val initialized.allocation ∧
      UniqueBacking source.val parsed manifest object.val := by
  unfold strict_memory_frontend_launch.initial_object at computed
  by_cases outside : object>=alloc.vec.Vec.len parsed.scope.globals
  · simp [outside] at computed
  · simp only [outside,ite_false] at computed
    obtain ⟨global,read,computed⟩:=source_bind_success _ _ _ computed
    have globalAt:=vec_index_success parsed.scope.globals object global read
    by_cases zero : global.alignment=0#u64
    · simp [zero] at computed
    · simp only [zero,ite_false,lift,bind_ok] at computed
      have positive : 0<global.alignment.val := by scalar_tac
      obtain ⟨margin,marginComputed,computed⟩:=source_bind_success _ _ _ computed
      have marginLaw:=U64.sub_spec (x:=global.alignment) (y:=1#u64) (by scalar_tac)
      rw [marginComputed] at marginLaw
      simp only [WP.spec_ok] at marginLaw
      have marginVal : margin.val=global.alignment.val-1 := by
        simpa only [WP.spec_ok,UScalar.ofNatCore_val_eq] using marginLaw.1
      cases checkedRound : U64.checked_add cursor margin with
      | none => simp [checkedRound] at computed
      | some rounded =>
        simp only [checkedRound] at computed
        have roundedLaw:=U64.checked_add_bv_spec cursor margin
        rw [checkedRound] at roundedLaw
        have roundedVal : rounded.val=cursor.val+global.alignment.val-1 := by
          have equal:=roundedLaw.2.1
          rw [marginVal] at equal
          omega
        obtain ⟨quotient,divided,computed⟩:=source_bind_success _ _ _ computed
        have quotientLaw:=U64.div_spec rounded (y:=global.alignment) (by omega)
        rw [divided] at quotientLaw
        have quotientVal : quotient.val=rounded.val/global.alignment.val := by simpa only [WP.spec_ok] using quotientLaw
        obtain ⟨base,multiplied,computed⟩:=source_bind_success _ _ _ computed
        have multiplyLaw:=UScalar.mul_equiv quotient global.alignment
        change UScalar.mul quotient global.alignment=.ok base at multiplied
        rw [multiplied] at multiplyLaw
        have baseVal : base.val=quotient.val*global.alignment.val := by simpa using multiplyLaw.2.1
        obtain ⟨allocation,allocationComputed,computed⟩:=source_bind_success _ _ _ computed
        cases allocation with
        | none => simp at computed
        | some allocation =>
          have allocationMeaning:=source_initial_allocation source parsed manifest object allocation names allocationComputed
          let bytes : U64:=UScalar.cast UScalarTy.U64 (alloc.vec.Vec.len global.bytes)
          have bytesVal : bytes.val=global.bytes.val.length := by
            have castBound : (alloc.vec.Vec.len global.bytes).val ≤ UScalar.max .U64 := by
              have bound := (alloc.vec.Vec.len global.bytes).hBounds
              rcases Usize.bounds_eq with width | width <;> simp_scalar at width ⊢ <;> omega
            have cast := UScalar.cast_inBounds_spec .U64 (alloc.vec.Vec.len global.bytes) castBound
            simpa only [lift,WP.spec_ok,alloc.vec.Vec.len_val] using cast
          cases checkedNext : U64.checked_add base bytes with
          | none => simp [bytes,checkedNext] at computed
          | some next =>
            simp only [show UScalar.cast UScalarTy.U64 (alloc.vec.Vec.len global.bytes)=bytes from rfl,checkedNext] at computed
            have nextLaw:=U64.checked_add_bv_spec base bytes
            rw [checkedNext] at nextLaw
            cases checkedTotal : U64.checked_add total bytes with
            | none => simp [checkedTotal] at computed
            | some size =>
              simp only [checkedTotal] at computed
              have totalLaw:=U64.checked_add_bv_spec total bytes
              rw [checkedTotal] at totalLaw
              by_cases tooLarge : size>67108864#u64
              · simp [tooLarge] at computed
              · simp only [tooLarge,ite_false] at computed
                have same : ({layout:={base,bytes,alignment:=global.alignment},allocation,next,total:=size}:
                    strict_memory_frontend_launch.InitializedObject)=initialized := by simpa only [ok.injEq,Option.some.injEq] using computed
                subst initialized
                refine ⟨global,globalAt,bytesVal,rfl,positive,?_,?_,nextLaw.2.1,?_,totalLaw.2.1,?_,allocationMeaning.1,allocationMeaning.2⟩
                · exact ⟨quotient.val,by rw [baseVal,Nat.mul_comm]⟩
                · rw [baseVal,quotientVal,roundedVal]
                  exact rounded_up cursor.val global.alignment.val positive
                · scalar_tac
                · scalar_tac

end PtxFrontendProgram
