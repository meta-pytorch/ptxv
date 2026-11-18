/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreDeferredRefinement
import Instruction.StoreNarrowSinkRefinement
import Instruction.StoreWindowProducer

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

@[step] theorem store_metadata_sources_spec (front : Front) (operation : Op)
    (hwindowsBound : windowsBounded (storeWindows front operation))
    (hcontrolBound : optionBounded (controlReads operation)) :
    store_window.metadata_sources front operation ⦃ result =>
      result.1.val.map byteWindow = storeWindows front operation ∧
      result.2.1.val.map key = deferredStoreData front operation ∧
      narrowStoreObservation result.2.2 = narrowStoreSink operation ⦄ := by
  unfold store_window.metadata_sources
  step with store_windows_spec front operation hwindowsBound as ⟨windows, hwindows⟩
  have hfinish (control : alloc.vec.Vec Reg)
      (hcontrol : control.val.map key = storeControlReads operation) :
      (do
        let deferred ← store_window.deferred_registers (alloc.vec.Vec.deref windows) (alloc.vec.Vec.deref control)
        let narrow ← store_window.narrow_sink operation (alloc.vec.Vec.deref control)
        ok (windows, deferred, narrow)) ⦃ result =>
          result.1.val.map byteWindow = storeWindows front operation ∧
          result.2.1.val.map key = deferredStoreData front operation ∧
          narrowStoreObservation result.2.2 = narrowStoreSink operation ⦄ := by
    step with deferred_registers_spec (alloc.vec.Vec.deref windows) (alloc.vec.Vec.deref control)
      as ⟨deferred, hdeferred, hlength⟩
    step with narrow_store_sink_spec operation (alloc.vec.Vec.deref control) as ⟨narrow, hnarrow⟩
    refine ⟨hwindows, ?_, ?_⟩
    · rw [hdeferred]
      simp only [alloc.vec.Vec.deref, Slice.from_val]
      change filterWindowSources windows.val (control.val.map key) = deferredStoreData front operation
      rw [hcontrol]
      unfold filterWindowSources deferredStoreData
      rw [← hwindows]
      have hsource : windowSource = ByteWindow.source ∘ byteWindow := by
        funext entry
        rfl
      rw [List.map_map, ← hsource]
    · rw [hnarrow]
      simp only [alloc.vec.Vec.deref, Slice.from_val]
      change narrowWithControls operation (control.val.map key) = narrowStoreSink operation
      rw [hcontrol]
      rfl
  step with control_reads_spec operation hcontrolBound as ⟨control, hcontrol⟩
  cases control with
  | none =>
    simp only [registerObservation] at hcontrol
    have hc : (alloc.vec.Vec.new Reg).val.map key = storeControlReads operation := by
      simp [storeControlReads, ← hcontrol, empty_register_vector]
    simpa only [bind_ok] using hfinish (alloc.vec.Vec.new Reg) hc
  | some registers =>
    simp only [registerObservation] at hcontrol
    have hc : registers.val.map key = storeControlReads operation := by
      simp [storeControlReads, ← hcontrol]
    simpa only [bind_ok] using hfinish registers hc

theorem actual_store_deferred_excludes_control (front : Front) (operation : Op)
    (hwindowsBound : windowsBounded (storeWindows front operation))
    (hcontrolBound : optionBounded (controlReads operation)) :
    store_window.metadata_sources front operation ⦃ result =>
      ∀ reg, reg ∈ storeControlReads operation → reg ∉ result.2.1.val.map key ⦄ := by
  step with store_metadata_sources_spec front operation hwindowsBound hcontrolBound
    as ⟨result, hwindows, hdeferred, hnarrow⟩
  rename_i reg hcontrol
  rw [hdeferred]
  exact deferred_store_data_excludes_control front operation reg hcontrol

theorem actual_store_narrowing_excludes_control (front : Front) (operation : Op)
    (hwindowsBound : windowsBounded (storeWindows front operation))
    (hcontrolBound : optionBounded (controlReads operation)) :
    store_window.metadata_sources front operation ⦃ result =>
      ∀ source mask, narrowStoreObservation result.2.2 = some (source, mask) →
        source ∉ storeControlReads operation ⦄ := by
  step with store_metadata_sources_spec front operation hwindowsBound hcontrolBound
    as ⟨result, hwindows, hdeferred, hnarrow⟩
  rename_i source mask hsource
  rw [hnarrow] at hsource
  exact narrow_store_sink_excludes_control operation source mask hsource

theorem synthetic_sass_discard_metadata_total (index : U16) (address : Addr)
    (haddress : Bounded (addressReads address))
    (hsource : key ⟨.V, index⟩ ∉ addressReads address) :
    store_window.metadata_sources .Sass
      (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) none) ⦃ result =>
      result.1.val.map byteWindow = repeatedWindows ⟨.V, index⟩ 16 128 ∧
      result.2.1.val.map key = (List.range 16).map (fun _ => key ⟨.V, index⟩) ∧
      narrowStoreObservation result.2.2 = none ⦄ := by
  have hmodel : storeWindows .Sass
      (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) none) =
      repeatedWindows ⟨.V, index⟩ 16 128 := by
    simp [storeWindows, storeDescriptor]
  have hbounded : windowsBounded (storeWindows .Sass
      (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) none)) := by
    rw [hmodel]
    intro register hregister
    rcases List.mem_map.mp hregister with ⟨window, hwindow, rfl⟩
    rcases List.mem_map.mp hwindow with ⟨position, hposition, rfl⟩
    dsimp [key]
    scalar_tac
  step with store_metadata_sources_spec .Sass
    (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) none) hbounded
    (by simpa [optionBounded, controlReads] using haddress)
    as ⟨result, hwindows, hdeferred, hnarrow⟩
  refine ⟨hwindows.trans hmodel, ?_, ?_⟩
  · rw [hdeferred]
    simp [deferredStoreData, hmodel, storeControlReads, controlReads,
      repeatedWindows, List.map_map, Function.comp_def, hsource]
  · simpa [narrowStoreSink, narrowStoreDescriptor] using hnarrow

end Instruction
