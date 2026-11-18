/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicExecution

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.AtomicConfluence

variable {α : Type}

def PrefixCommutation (add : α → α → α) (start : α) (values : List α) : Prop :=
  ∀ (prior : List α), (∀ value ∈ prior, value ∈ values) → prior.length + 2 ≤ values.length →
    ∀ first ∈ values, ∀ second ∈ values,
      add (add (prior.foldl add start) first) second =
        add (add (prior.foldl add start) second) first

/-- Local commutation at every admissible prior suffices; no associativity is assumed. -/
theorem fold_permutation (add : α → α → α) (start : α) (values : List α)
    (commute : PrefixCommutation add start values) (left right : List α)
    (permutation : left.Perm right) (prior : List α)
    (members : ∀ value ∈ prior ++ left, value ∈ values)
    (length : prior.length + left.length ≤ values.length) :
    left.foldl add (prior.foldl add start) = right.foldl add (prior.foldl add start) := by
  induction permutation using List.Perm.recOnSwap' generalizing prior with
  | nil => rfl
  | @cons first left right permutation ih =>
    simpa only [List.foldl_cons, List.foldl_append, List.foldl_nil] using
      ih (prior ++ [first]) (by simpa [List.append_assoc] using members) (by simp at *; omega)
  | @swap' first second left right permutation ih =>
    have hfirst : first ∈ values := members first (by simp)
    have hsecond : second ∈ values := members second (by simp)
    have hprefix : ∀ value ∈ prior, value ∈ values := by
      intro value member; exact members value (by simp [member])
    have equality := commute prior hprefix (by simp at length; omega) second hsecond first hfirst
    simp only [List.foldl_cons]
    rw [equality]
    simpa only [List.foldl_append, List.foldl_cons, List.foldl_nil] using
      ih (prior ++ [first, second]) (by
          intro value member
          apply members value
          simp only [List.mem_append, List.mem_cons] at member ⊢
          tauto)
        (by simp at *; omega)
  | @trans left middle right first second ihFirst ihSecond =>
    exact (ihFirst prior members length).trans (ihSecond prior (by
      intro value member
      rcases List.mem_append.mp member with hprefix | hmiddle
      · exact members value (List.mem_append_left _ hprefix)
      · exact members value (List.mem_append_right _ (first.mem_iff.mpr hmiddle)))
      (by simpa only [first.length_eq] using length))

structure LayerContract (add : α → α → α) (start : α) (values : List α)
    (layers : List (List α)) : Prop where
  length : layers.length = values.length - 1
  initial : 2 ≤ values.length → start ∈ layers[0]!
  closure : ∀ depth, depth + 1 < layers.length →
    ∀ state ∈ layers[depth]!, ∀ value ∈ values, add state value ∈ layers[depth + 1]!
  commute : ∀ depth, depth < layers.length →
    ∀ state ∈ layers[depth]!, ∀ first ∈ values, ∀ second ∈ values,
      add (add state first) second = add (add state second) first

theorem prefix_in_layer (add : α → α → α) (start : α) (values : List α)
    (layers : List (List α)) (contract : LayerContract add start values layers)
    (prior : List α) (members : ∀ value ∈ prior, value ∈ values)
    (length : prior.length < layers.length) : prior.foldl add start ∈ layers[prior.length]! := by
  induction prior using List.reverseRecOn with
  | nil =>
    simpa using contract.initial (by have := contract.length; simp at length; omega)
  | append_singleton prior value ih =>
    have shorter : prior.length < layers.length := by simp at length; omega
    have prefixMembers : ∀ x ∈ prior, x ∈ values := by
      intro x hx; exact members x (by simp [hx])
    have valueMember : value ∈ values := members value (by simp)
    have closed := contract.closure prior.length (by simp at length; omega)
      (prior.foldl add start) (ih prefixMembers shorter) value valueMember
    simpa only [List.length_append, List.length_singleton, List.foldl_append,
      List.foldl_cons, List.foldl_nil] using closed

theorem layers_imply_prefix_commutation (add : α → α → α) (start : α) (values : List α)
    (layers : List (List α)) (contract : LayerContract add start values layers) :
    PrefixCommutation add start values := by
  intro prior members length first hfirst second hsecond
  have prefixLength : prior.length < layers.length := by have := contract.length; omega
  exact contract.commute prior.length prefixLength (prior.foldl add start)
    (prefix_in_layer add start values layers contract prior members prefixLength)
    first hfirst second hsecond

theorem all_permutations_match (add : α → α → α) (start : α) (values : List α)
    (layers : List (List α)) (contract : LayerContract add start values layers)
    (order : List α) (permutation : values.Perm order) :
    order.foldl add start = values.foldl add start := by
  symm
  simpa using fold_permutation add start values
    (layers_imply_prefix_commutation add start values layers contract)
    values order permutation [] (by simp) (by simp)

end MemoryWrite.AtomicConfluence
