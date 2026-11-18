/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadFailure
import Instruction.St256Producer

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

theorem byte_window_at (windows : List (Reg × U32 × U32)) (model : List ByteWindow)
    (hshape : windows.map byteWindow = model) (index : Nat)
    (hi : index < windows.length) (hm : index < model.length) :
    byteWindow (windows[index]'hi) = model[index]'hm := by
  have hget := congrArg (fun entries : List ByteWindow => entries[index]?) hshape
  rw [getElem?_pos (windows.map byteWindow) index (by simpa using hi)] at hget
  rw [getElem?_pos model index hm] at hget
  simpa using Option.some.inj hget

@[step] theorem actual_store_payload_failure_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (front : Front) (operation : Op)
    (reader : Reader) (values : Key → U64) (failed : Nat) (error : Error)
    (hbounded : windowsBounded (storeWindows front operation))
    (hsupported : supportedPayloadTotal front operation ≠ none)
    (hfailed : failed < (storeWindows front operation).length)
    (hreadBefore : ∀ index (hi : index < (storeWindows front operation).length),
      index < failed → ∀ register,
      key register = ((storeWindows front operation)[index]'hi).source →
      readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄)
    (hreadFailed : ∀ register,
      key register = ((storeWindows front operation)[failed]'hfailed).source →
      readerInst.read reader register ⦃ result => result = .Err error ⦄) :
    payload.from_operation readerInst front operation reader ⦃ result =>
      result = some (.Err error) ⦄ := by
  unfold payload.from_operation
  step with payload_supported_total_spec front operation as ⟨total, htotal⟩
  cases total with
  | none => exact False.elim (hsupported htotal.symm)
  | some total =>
    simp only [Option.map_some] at htotal
    step with store_windows_spec front operation hbounded as ⟨windows, hshape⟩
    have hvalid := operation_windows_valid front operation total.val htotal.symm windows.val hshape
    have hlength : windows.val.length = (storeWindows front operation).length := by
      have := congrArg List.length hshape
      simpa using this
    have hrawFailed : failed < windows.val.length := by omega
    have hbefore : ∀ index (hi : index < windows.val.length), index < failed →
        readerInst.read reader (windows.val[index]'hi).1 ⦃ result =>
          result = .Ok (values (key (windows.val[index]'hi).1)) ⦄ := by
      intro index hi hp
      have hm : index < (storeWindows front operation).length := by omega
      exact hreadBefore index hm hp _ (congrArg ByteWindow.source
        (byte_window_at windows.val _ hshape index hi hm))
    have hfailure : readerInst.read reader (windows.val[failed]'hrawFailed).1 ⦃ result =>
        result = .Err error ⦄ :=
      hreadFailed _ (congrArg ByteWindow.source
        (byte_window_at windows.val _ hshape failed hrawFailed hfailed))
    step with payload_assemble_failure_spec readerInst (alloc.vec.Vec.deref windows)
      total reader values failed error
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hvalid)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hrawFailed)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hbefore)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hfailure) as ⟨result, hresult⟩
    simp [hresult]

@[step] theorem actual_st256_payload_failure_spec {Reader Error : Type}
    (readerInst : payload.RegisterReader Reader Error) (operation : Op)
    (reader : Reader) (values : Key → U64) (failed : Nat) (error : Error)
    (hbounded : ∀ windows, st256Windows operation = some windows → windowsBounded windows)
    (hfailed : failed < ((st256Windows operation).getD []).length)
    (hreadBefore : ∀ index (hi : index < ((st256Windows operation).getD []).length),
      index < failed → ∀ register,
      key register = (((st256Windows operation).getD [])[index]'hi).source →
      readerInst.read reader register ⦃ result => result = .Ok (values (key register)) ⦄)
    (hreadFailed : ∀ register,
      key register = (((st256Windows operation).getD [])[failed]'hfailed).source →
      readerInst.read reader register ⦃ result => result = .Err error ⦄) :
    payload.from_st256 readerInst operation reader ⦃ result =>
      result = some (.Err error) ⦄ := by
  unfold payload.from_st256
  step with st256_sources_spec operation hbounded as ⟨windows, hwindows⟩
  cases windows with
  | none => simp [← hwindows] at hfailed
  | some windows =>
    have hvalid := actual_st256_windows_valid operation windows.val hwindows
    have hmodel : windows.val.map byteWindow = (st256Windows operation).getD [] := by
      rw [← hwindows]
      rfl
    have hlength : windows.val.length = ((st256Windows operation).getD []).length := by
      have := congrArg List.length hmodel
      simpa using this
    have hrawFailed : failed < windows.val.length := by omega
    have hbefore : ∀ index (hi : index < windows.val.length), index < failed →
        readerInst.read reader (windows.val[index]'hi).1 ⦃ result =>
          result = .Ok (values (key (windows.val[index]'hi).1)) ⦄ := by
      intro index hi hp
      have hm : index < ((st256Windows operation).getD []).length := by omega
      exact hreadBefore index hm hp _ (congrArg ByteWindow.source
        (byte_window_at windows.val _ hmodel index hi hm))
    have hfailure : readerInst.read reader (windows.val[failed]'hrawFailed).1 ⦃ result =>
        result = .Err error ⦄ :=
      hreadFailed _ (congrArg ByteWindow.source
        (byte_window_at windows.val _ hmodel failed hrawFailed hfailed))
    step with payload_assemble_failure_spec readerInst (alloc.vec.Vec.deref windows)
      32#u8 reader values failed error
      (by simpa [alloc.vec.Vec.deref, Slice.from_val] using hvalid)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hrawFailed)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hbefore)
      (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using hfailure) as ⟨result, hresult⟩
    simp [hresult]

end Instruction
