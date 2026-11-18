/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerModel
import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace RegisterOwner

def keyView (key : U32 × U8 × U16) : Key := ⟨key.1.val, key.2.1.val, key.2.2.val⟩

@[step]
theorem rust_initialize_leaders (count : Usize) (first : alloc.vec.Vec (Option U32))
    (hbound : first.val.length ≤ count.val)
    (hempty : first.val = List.replicate first.val.length none) :
    register_owner.uniform_owners_loop0 count first
      ⦃ result => result.val = List.replicate count.val none ⦄ := by
  unfold register_owner.uniform_owners_loop0
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (Option U32) => count.val - current.val.length)
    (fun current => current.val.length ≤ count.val ∧
      current.val = List.replicate current.val.length none)
  · clear first hbound hempty
    intro current hcurrent
    obtain ⟨hbound, hempty⟩ := hcurrent
    unfold register_owner.uniform_owners_loop0.body
    dsimp only
    split
    · rename_i hlt
      have hlength : current.val.length < count.val := by clear hempty; scalar_tac
      step with alloc.vec.Vec.push_spec current none (by clear hempty; scalar_tac) as ⟨next, hnext⟩
      constructor
      · simp only [hnext, List.length_append, List.length_singleton]; omega
      · simp only [hnext, List.length_append, List.length_singleton]
        rw [List.replicate_succ', ←hempty]
        exact ⟨rfl, by omega⟩
    · rename_i hend
      have hlength : current.val.length = count.val := by clear hempty; scalar_tac
      simpa [hlength] using hempty
  · exact ⟨hbound, hempty⟩

@[step]
theorem rust_canonical_key (thread : Usize) (owners : Slice U32) (register : Reg) :
    register_owner.canonical_key thread owners register ⦃ result =>
      keyView result = canonicalKey thread.val (owners.val.map UScalar.val) register ⦄ := by
  rcases register with ⟨file, index⟩
  cases file <;>
    simp only [register_owner.canonical_key, lift, bind_ok]
  all_goals
    try split
    all_goals
      try step
      simp_all [keyView, canonicalKey, shared, fileTag, UScalar.cast_val_eq,
        read_discriminant, File.read_discriminant]
  all_goals
    have hbound : thread.val < owners.val.length := by scalar_tac
    simp [List.getElem?_eq_getElem hbound]

@[step]
theorem rust_owner_for_warp (first : alloc.vec.Vec (Option U32))
    (warp : Usize) (thread : U32) (bound : warp.val < first.val.length) :
    register_owner.owner_for_warp first warp thread ⦃ result =>
      match first.val[warp.val] with
      | none => result.1 = thread ∧ result.2.val = first.val.set warp.val (some thread)
      | some owner => result.1 = owner ∧ result.2.val = first.val ⦄ := by
  unfold register_owner.owner_for_warp
  step as ⟨slot, hslot⟩
  rw [hslot]
  split
  · step as ⟨old, back, hold, hback⟩
    simp_all [alloc.vec.Vec.set]
  · simp_all

@[step]
theorem rust_extend_leaders (prior : List U32)
    (first : alloc.vec.Vec (Option U32)) (warp : U32) (index : Usize) (thread : U32)
    (same_index : index.val = warp.val) (bound : warp.val < first.val.length)
    (sound : LeadersSound prior first.val) (fresh : thread.val = prior.length) :
    register_owner.owner_for_warp first index thread ⦃ result =>
      result.1.val = (if warp ∈ prior then prior.idxOf warp else prior.length) ∧
      LeadersSound (prior ++ [warp]) result.2.val ∧
      result.2.val.length = first.val.length ⦄ := by
  step with rust_owner_for_warp first index thread (by omega) as ⟨result, hresult⟩
  simp only [same_index] at hresult
  cases slot : first.val[warp.val] with
  | none =>
    simp only [slot] at hresult
    obtain ⟨owner, changed⟩ := hresult
    have hnew := leaders_new prior first.val warp thread bound sound slot fresh
    simp only [owner, changed]
    exact ⟨by simp [hnew.1, fresh], hnew.2, by simp⟩
  | some previous =>
    simp only [slot] at hresult
    obtain ⟨owner, unchanged⟩ := hresult
    have hexisting := leaders_existing prior first.val warp previous bound sound slot
    simp only [owner, unchanged]
    exact ⟨by simp [hexisting.1, hexisting.2.1], hexisting.2.2, True.intro⟩

@[step]
theorem rust_owner_loop (warps : Slice U32) (count : Usize)
    (first : alloc.vec.Vec (Option U32)) (owners : alloc.vec.Vec U32) (thread : Usize)
    (size_bound : warps.val.length ≤ 2 ^ 32 - 1)
    (initial : LoopInvariant warps count first owners thread) :
    register_owner.uniform_owners_loop1 warps count first owners thread
      ⦃ OwnerLoopOutcome warps.val count.val ⦄ := by
  unfold register_owner.uniform_owners_loop1
  apply loop.spec_decr_nat
    (fun state => warps.val.length - state.2.2.val)
    (fun state => LoopInvariant warps count state.1 state.2.1 state.2.2)
  · clear first owners thread initial
    intro state invariant
    rcases state with ⟨first, owners, thread⟩
    unfold register_owner.uniform_owners_loop1.body
    dsimp only
    split
    · rename_i before_end
      have within : thread.val < warps.val.length := by scalar_tac
      step as ⟨warp, hwarp⟩
      simp only [lift, bind_ok]
      have index_exact : (UScalar.cast .Usize warp).val = warp.val := by scalar_tac
      split
      · rename_i invalid
        have bad : count.val ≤ warp.val := by scalar_tac
        simp only [WP.spec_ok, OwnerLoopOutcome]
        exact ⟨warp, by rw [hwarp]; exact List.getElem_mem within, bad⟩
      · rename_i valid
        have warp_bound : warp.val < first.val.length := by
          have := invariant.first_length
          scalar_tac
        have fresh : (UScalar.cast .U32 thread).val = (warps.val.take thread.val).length := by
          simp only [UScalar.cast_val_eq, List.length_take_of_le invariant.thread_bound]
          apply Nat.mod_eq_of_lt
          scalar_tac
        step with rust_extend_leaders (warps.val.take thread.val) first warp
          (UScalar.cast .Usize warp) (UScalar.cast .U32 thread)
          index_exact warp_bound invariant.leaders_sound fresh as
            ⟨owner, next_first, owner_sound, next_leaders, same_length⟩
        step with alloc.vec.Vec.push_spec owners owner
          (by have := invariant.owners_length; scalar_tac) as ⟨next_owners, next_owners_eq⟩
        step as ⟨next_thread, next_thread_eq⟩
        have next : next_thread.val = thread.val + 1 := by scalar_tac
        have appended : warps.val.take next_thread.val = warps.val.take thread.val ++ [warp] := by
          rw [next, List.take_succ_eq_append_getElem within, ←hwarp]
        constructor
        · constructor
          · omega
          · exact same_length.trans invariant.first_length
          · simp only [next_owners_eq, List.length_append, List.length_singleton]
            rw [invariant.owners_length, next]
          · rw [next_owners_eq, List.map_append, invariant.owners_sound,
              appended, owner_table_append]
            simp [owner_sound]
          · rw [appended]
            exact next_leaders
          · rw [appended]
            intro item member
            rcases List.mem_append.mp member with prior | added
            · exact invariant.valid_prefix item prior
            · have item_eq : item = warp := by simpa using added
              subst item
              scalar_tac
        · omega
    · rename_i finished
      have at_end : thread.val = warps.val.length := by
        have := invariant.thread_bound
        scalar_tac
      simp only [WP.spec_ok, OwnerLoopOutcome]
      constructor
      · simpa [at_end] using invariant.owners_sound
      · simpa [at_end] using invariant.valid_prefix
  · exact initial

theorem u32_max_to_usize : (UScalar.cast .Usize core.num.U32.MAX).val = 4294967295 := by
  simp only [UScalar.cast_val_eq, core.num.U32.MAX, UScalar.ofNat,
    UScalar.ofNatCore_val_eq, U32.rMax]
  apply Nat.mod_eq_of_lt
  scalar_tac

@[step]
theorem rust_uniform_owners (warps : Slice U32) (count : Usize) :
    register_owner.uniform_owners warps count ⦃ OwnerOutcome warps.val count.val ⦄ := by
  unfold register_owner.uniform_owners
  simp only [lift, bind_ok]
  split
  · rename_i too_many_threads
    have large : 4294967295 < warps.val.length := by
      have limit := u32_max_to_usize
      scalar_tac
    simp [WP.spec_ok, OwnerOutcome, large]
  · rename_i thread_bound
    have size_bound : warps.val.length ≤ 4294967295 := by
      have limit := u32_max_to_usize
      scalar_tac
    split
    · rename_i too_many_warps
      have large : warps.val.length < count.val := by scalar_tac
      simp [WP.spec_ok, OwnerOutcome, large, Nat.not_lt.mpr size_bound]
    · rename_i count_bound
      step with rust_initialize_leaders count (alloc.vec.Vec.new (Option U32))
        (by simp [alloc.vec.Vec.new]) (by simp [alloc.vec.Vec.new]) as ⟨first, initialized⟩
      have initial : LoopInvariant warps count first (alloc.vec.Vec.new U32) 0#usize := by
        constructor
        · simp
        · simp [initialized]
        · simp [alloc.vec.Vec.new]
        · simp [alloc.vec.Vec.new, ownerTable]
        · simpa [initialized] using empty_leaders_sound count.val
        · simp [ValidWarps]
      step with rust_owner_loop warps count first (alloc.vec.Vec.new U32) 0#usize
        size_bound initial as ⟨result, sound⟩
      have bounded : count.val ≤ warps.val.length := by scalar_tac
      simpa [OwnerOutcome, Nat.not_lt.mpr size_bound, Nat.not_lt.mpr bounded] using sound

@[step]
theorem rust_uniform_owners_valid (warps : Slice U32) (count : Usize)
    (size_bound : warps.val.length ≤ 4294967295)
    (count_bound : count.val ≤ warps.val.length)
    (valid : ValidWarps warps.val count.val) :
    register_owner.uniform_owners warps count ⦃ result =>
      ∃ owners, result = core.result.Result.Ok owners ∧
        owners.val.map UScalar.val = ownerTable warps.val ⦄ := by
  step with rust_uniform_owners warps count as ⟨result, sound⟩
  have exact_outcome : OwnerLoopOutcome warps.val count.val result := by
    simpa [OwnerOutcome, Nat.not_lt.mpr size_bound, Nat.not_lt.mpr count_bound] using sound
  cases result with
  | Ok owners => exact ⟨owners, rfl, exact_outcome.1⟩
  | Err error =>
    cases error with
    | TooManyThreads => exact False.elim exact_outcome
    | TooManyWarps => exact False.elim exact_outcome
    | InvalidWarp =>
      obtain ⟨warp, member, bad⟩ := exact_outcome
      have := valid warp member
      omega

theorem canonical_uniform_alias_iff (warps : List U32) (owners : List Nat)
    (table : owners = ownerTable warps) (register : Reg) (uniform : shared register.file = true)
    (left right : Nat) (hl : left < warps.length) (hr : right < warps.length) :
    canonicalKey left owners register = canonicalKey right owners register ↔
      warps[left] = warps[right] := by
  subst owners
  have left_bound : left < (ownerTable warps).length := by simpa using hl
  have right_bound : right < (ownerTable warps).length := by simpa using hr
  simp only [canonicalKey, uniform, ite_true,
    List.getElem?_eq_getElem left_bound, List.getElem?_eq_getElem right_bound,
    Option.getD_some, Key.mk.injEq, and_self, and_true]
  exact owner_table_alias_iff warps left right hl hr

theorem canonical_lane_alias_iff (owners : List Nat) (register : Reg)
    (lane : shared register.file = false) (left right : Nat)
    (hl : left < 2 ^ 32) (hr : right < 2 ^ 32) :
    canonicalKey left owners register = canonicalKey right owners register ↔ left = right := by
  norm_num at hl hr
  simp [canonicalKey, lane, Nat.mod_eq_of_lt hl, Nat.mod_eq_of_lt hr, Key.mk.injEq]

theorem file_tag_discriminant (file : File) :
    (IScalar.hcast .U8 (read_discriminant file)).val = fileTag file := by
  cases file <;> simp [fileTag, read_discriminant, File.read_discriminant]

theorem u32_cast_to_usize (value : U32) : (UScalar.cast .Usize value).val = value.val := by
  rw [UScalar.cast_val_eq]
  apply Nat.mod_eq_of_lt
  scalar_tac

@[step]
theorem rust_record_aliases (thread : Usize) (owners : Slice U32) (register : Reg)
    (record : U32 × U8 × U16) :
    register_owner.record_aliases thread owners register record ⦃ result =>
      result = decide (canonicalKey thread.val (owners.val.map UScalar.val) register =
        recordKey (owners.val.map UScalar.val) record) ⦄ := by
  rcases record with ⟨writer, file, index⟩
  unfold register_owner.record_aliases
  simp only [lift, bind_ok]
  change (if IScalar.hcast .U8 (read_discriminant register.file) != file then ok false
    else if register.idx != index then ok false
    else do
      let current ← register_owner.canonical_key thread owners register
      let previous ← register_owner.canonical_key (UScalar.cast .Usize writer) owners register
      ok (decide (current.1 = previous.1))) ⦃ result =>
        result = decide (canonicalKey thread.val (owners.val.map UScalar.val) register =
          recordKey (owners.val.map UScalar.val) (writer, file, index)) ⦄
  have file_view := file_tag_discriminant register.file
  split
  · rename_i different
    simp only [WP.spec_ok]
    symm
    rw [decide_eq_false_iff_not]
    intro equal
    have same_file := congrArg Key.file equal
    simp only [canonicalKey, recordKey] at same_file
    scalar_tac
  · rename_i same_file
    have same_file_view : fileTag register.file = file.val := by scalar_tac
    split
    · rename_i different
      simp only [WP.spec_ok]
      symm
      rw [decide_eq_false_iff_not]
      intro equal
      have same_index := congrArg Key.index equal
      simp only [canonicalKey, recordKey] at same_index
      scalar_tac
    · rename_i same_index
      have same_index_view : register.idx.val = index.val := by scalar_tac
      step with rust_canonical_key thread owners register as ⟨current, current_view⟩
      step with rust_canonical_key (UScalar.cast .Usize writer) owners register as
        ⟨previous, previous_view⟩
      rw [record_key_agrees _ (writer, file, index) register same_file_view same_index_view]
      have current_owner := congrArg Key.owner current_view
      have previous_owner := congrArg Key.owner previous_view
      simp only [keyView, u32_cast_to_usize] at current_owner previous_owner
      simp only [canonicalKey, Key.mk.injEq, and_true]
      simp only [canonicalKey] at current_owner previous_owner
      rw [←current_owner, ←previous_owner]
      simp
      exact ⟨fun h => congrArg UScalar.val h, UScalar.eq_of_val_eq⟩

end RegisterOwner
