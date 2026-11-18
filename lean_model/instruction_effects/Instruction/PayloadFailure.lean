/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadProducer

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem payload_assemble_loop_failure_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (windows : Slice (Reg × U32 × U32))
    (reader : Reader) (values : Key → U64) (total failed : Nat) (error : Error)
    (hsize : total ≤ 255) (hvalid : payloadWindowsValid windows.val total)
    (hfailed : failed < windows.val.length)
    (hreadBefore : ∀ index (hi : index < windows.val.length), index < failed →
      readerInst.read reader (windows.val[index]'hi).1 ⦃ result =>
        result = .Ok (values (key (windows.val[index]'hi).1)) ⦄)
    (hreadFailed : readerInst.read reader (windows.val[failed]'hfailed).1 ⦃ result =>
      result = .Err error ⦄)
    (output : alloc.vec.Vec U8) (index : Usize)
    (hindex : index.val ≤ failed) (hlength : output.val.length = total) :
    payload.assemble_loop1 readerInst windows reader output index ⦃ result =>
      result = .Err error ⦄ := by
  unfold payload.assemble_loop1
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U8 × Usize => failed + 1 - state.2.val)
    (fun state : alloc.vec.Vec U8 × Usize => state.2.val ≤ failed ∧ state.1.val.length = total)
  · rintro ⟨current, index⟩ ⟨hbound, hcurrentLength⟩
    dsimp only at hbound hcurrentLength
    unfold payload.assemble_loop1.body
    have hlt : index.val < windows.val.length := by omega
    have htest : index < Slice.len windows := by scalar_tac
    simp only [htest, ite_true]
    step as ⟨window, hwindow⟩
    rcases window with ⟨register, first, count⟩
    have hentry : (register, first, count) = windows.val[index.val]'hlt := hwindow
    by_cases hprevious : index.val < failed
    · have hreader : readerInst.read reader register ⦃ result =>
          result = .Ok (values (key register)) ⦄ := by
        simpa only [← hentry] using hreadBefore index.val hlt hprevious
      step with hreader as ⟨result, hresult⟩
      rw [hresult]
      have hproperties := hvalid index.val hlt
      rw [← hentry] at hproperties
      obtain ⟨hcount, hwithin, hbefore⟩ := hproperties
      step with payload_write_window_spec current first count (values (key register)) 0#u32 current
        hcount (by simp) (by omega) (by simpa [hcurrentLength] using hwithin)
        (by simp) as ⟨written, hwritten⟩
      have hwrittenLength : written.val.length = total := by
        have := congrArg List.length hwritten
        simpa [overwriteBytes, hcurrentLength] using this
      step as ⟨next, hnext⟩; try scalar_tac
    · have heq : index.val = failed := by omega
      have hreader : readerInst.read reader register ⦃ result => result = .Err error ⦄ := by
        simpa only [← heq, ← hentry] using hreadFailed
      step with hreader as ⟨result, hresult⟩
      rw [hresult]
      simp
  · exact ⟨hindex, hlength⟩

@[step] theorem payload_assemble_failure_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (windows : Slice (Reg × U32 × U32))
    (total : U8) (reader : Reader) (values : Key → U64) (failed : Nat) (error : Error)
    (hvalid : payloadWindowsValid windows.val total.val)
    (hfailed : failed < windows.val.length)
    (hreadBefore : ∀ index (hi : index < windows.val.length), index < failed →
      readerInst.read reader (windows.val[index]'hi).1 ⦃ result =>
        result = .Ok (values (key (windows.val[index]'hi).1)) ⦄)
    (hreadFailed : readerInst.read reader (windows.val[failed]'hfailed).1 ⦃ result =>
      result = .Err error ⦄) :
    payload.assemble readerInst windows total reader ⦃ result => result = .Err error ⦄ := by
  unfold payload.assemble
  step with payload_initialize_spec total (alloc.vec.Vec.new U8) 0#u8 (by simp)
    (by simp [alloc.vec.Vec.new, alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val])
    as ⟨output, houtput⟩
  have hlength : output.val.length = total.val := by
    have := congrArg List.length houtput
    simpa using this
  step with payload_assemble_loop_failure_spec readerInst windows reader values total.val failed error
    (by scalar_tac) hvalid hfailed hreadBefore hreadFailed output 0#usize (by simp) hlength as ⟨result, hresult⟩
  exact hresult

end Instruction
