/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceBankBits

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

open Storage.MachineRegistersVirtualMap

 theorem fold_undefined_absent (entries : List (U16 × U64)) (query accumulated : Nat)
    (absent : ∀ entry ∈ entries, entry.1.val ≠ query) :
    entries.foldl (fun mask entry => if entry.1.val = query then mask ||| entry.2.val else mask)
      accumulated = accumulated := by
  induction entries generalizing accumulated with
  | nil => rfl
  | cons entry rest ih =>
    simp only [List.foldl_cons, absent entry (by simp), ite_false]
    apply ih
    intro other member
    exact absent other (by simp [member])

 theorem undefined_lookup (entries : List (U16 × U64)) (query : U16)
    (sorted : Sorted entries) :
    undefinedBits entries query.val = ((lookup entries query).map UScalar.val).getD 0 := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
    obtain ⟨head, tail⟩ := List.pairwise_cons.mp sorted
    by_cases selected : entry.1 = query
    · have absent : ∀ other ∈ rest, other.1.val ≠ query.val := by
        intro other member same
        have lt := head other member
        rw [selected, same] at lt
        exact (Nat.lt_irrefl _ lt)
      simp only [undefinedBits, List.foldl_cons, selected, ite_true, Nat.zero_or]
      rw [fold_undefined_absent rest query.val entry.2.val absent]
      simp [lookup, selected]
    · have unequal : entry.1.val ≠ query.val := by
        intro equal
        exact selected (UScalar.eq_of_val_eq equal)
      simpa only [undefinedBits, List.foldl_cons, unequal, ite_false, lookup, selected] using ih tail

 theorem undefined_large (entries : List (U16 × U64)) (query : Nat)
    (large : UScalar.max .U16 < query) : undefinedBits entries query = 0 := by
  apply fold_undefined_absent
  intro entry _ same
  have bounded : entry.1.val ≤ UScalar.max .U16 := by scalar_tac
  omega

 theorem undefined_lookup_ext (left right : List (U16 × U64))
    (leftSorted : Sorted left) (rightSorted : Sorted right) (except : U16)
    (same : ∀ query : U16, query ≠ except → lookup left query = lookup right query)
    (query : Nat) (distinct : query ≠ except.val) :
    undefinedBits left query = undefinedBits right query := by
  by_cases representable : query ≤ UScalar.max .U16
  · let key : U16 := UScalar.ofNat query (by scalar_tac)
    have keyVal : key.val = query := by scalar_tac
    have keyNe : key ≠ except := by intro equal; apply distinct; simpa [keyVal] using congrArg UScalar.val equal
    rw [←keyVal, undefined_lookup left key leftSorted, undefined_lookup right key rightSorted, same key keyNe]
  · rw [undefined_large left query (by omega), undefined_large right query (by omega)]

private theorem sorted_length_bound (entries : List (U16 × U64)) (lower upper : Nat)
    (sorted : Sorted entries)
    (range : ∀ entry ∈ entries, lower ≤ entry.1.val ∧ entry.1.val < upper) :
    entries.length ≤ upper - lower := by
  induction entries generalizing lower with
  | nil => simp
  | cons entry rest ih =>
    obtain ⟨head, tail⟩ := List.pairwise_cons.mp sorted
    have first := range entry (by simp)
    have restBound := ih (entry.1.val + 1) tail (by
      intro other member
      have higher := head other member
      have within := range other (by simp [member])
      omega)
    simp only [List.length_cons]
    omega

theorem canonical_room (entries : List (U16 × U64)) (canonical : Canonical entries) :
    entries.length < Usize.max := by
  have bounded := sorted_length_bound entries 0 65536 canonical.1 (by
    intro entry _
    have limit : entry.1.val < 65536 := by scalar_tac
    omega)
  rcases Usize.bounds_eq with narrow | wide <;> scalar_tac

end ValueFlow.Source
