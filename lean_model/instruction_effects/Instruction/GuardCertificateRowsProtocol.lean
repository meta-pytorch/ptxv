/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateRowsMerge
import Instruction.GuardCertificateRowsPublish

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction.GuardCertificate

theorem merge_rows_mismatched (previous incoming : Slice Observation)
    (different : previous.length ≠ incoming.length) :
    guard_certificate.merge_rows previous incoming = .ok (false, previous) := by
  have unequal : Slice.len previous ≠ Slice.len incoming := by
    simpa only [ne_eq, UScalar.eq_equiv, Slice.len_val] using different
  simp [guard_certificate.merge_rows, bne, unequal]

theorem successful_merge_rows (previous incoming after : Slice Observation)
    (returned : guard_certificate.merge_rows previous incoming = .ok (true, after)) :
    previous.length = incoming.length ∧ RowMerged previous incoming after := by
  by_cases lengths : previous.length = incoming.length
  · have meaning := rust_merge_rows previous incoming lengths
    rw [returned, WP.spec_ok] at meaning
    exact ⟨lengths, meaning.2⟩
  · rw [merge_rows_mismatched previous incoming lengths] at returned
    simp at returned

theorem publish_mismatched (code : Slice Inst) (observations : Slice Observation)
    (different : code.length ≠ observations.length) :
    guard_certificate.publish code observations = .ok none := by
  have unequal : Slice.len code ≠ Slice.len observations := by
    simpa only [ne_eq, UScalar.eq_equiv, Slice.len_val] using different
  simp [guard_certificate.publish, bne, unequal]

theorem successful_publish (code : Slice Inst) (observations : Slice Observation)
    (output : alloc.vec.Vec (Option Bool))
    (returned : guard_certificate.publish code observations = .ok (some output)) :
    code.length = observations.length ∧ RowsPublished code observations output := by
  by_cases lengths : code.length = observations.length
  · have meaning := rust_publish code observations lengths
    rw [returned, WP.spec_ok] at meaning
    obtain ⟨actual, same, property⟩ := meaning
    have equal : actual = output := by simpa using same.symm
    subst actual
    exact ⟨lengths, property⟩
  · rw [publish_mismatched code observations lengths] at returned
    simp at returned

inductive MergeRowsCalls : List (Slice Observation) → Slice Observation → Slice Observation → Prop
  | nil (before : Slice Observation) : MergeRowsCalls [] before before
  | cons (row : Slice Observation) (rest : List (Slice Observation))
      (before middle after : Slice Observation) :
      guard_certificate.merge_rows before row = .ok (true, middle) →
      MergeRowsCalls rest middle after →
      MergeRowsCalls (row :: rest) before after

def rowObservations (rows : List (Slice Observation)) (pc : Nat) : List Observation :=
  rows.map fun row => row[pc]!

theorem merge_calls_meaning (rows : List (Slice Observation)) (before after : Slice Observation)
    (calls : MergeRowsCalls rows before after) :
    after.length = before.length ∧ ∀ pc < before.length,
      after[pc]! = (rowObservations rows pc).foldl mergeObservation before[pc]! := by
  induction calls with
  | nil before => exact ⟨rfl, by simp [rowObservations]⟩
  | cons row rest before middle after returned tail ih =>
    have head := (successful_merge_rows before row middle returned).2
    refine ⟨ih.1.trans head.1, ?_⟩
    intro pc inside
    have middleLength := head.1
    have middleInside : pc < middle.length := by omega
    rw [ih.2 pc middleInside, head.2 pc inside]
    rfl

theorem merged_rows_known (rows : List (Slice Observation)) (before after : Slice Observation)
    (calls : MergeRowsCalls rows before after) (pc : Nat) (inside : pc < before.length)
    (initial : before[pc]! = none) (truth : Bool) (known : after[pc]! = some (some truth))
    (row : Slice Observation) (member : row ∈ rows) (current : Option Bool)
    (reached : row[pc]! = some current) : current = some truth := by
  have result := merge_calls_meaning rows before after calls
  have aggregate : (rowObservations rows pc).foldl mergeObservation none = some (some truth) := by
    rw [←initial, ←result.2 pc inside]
    exact known
  apply known_merged_observation (rowObservations rows pc) truth aggregate current
  exact List.mem_map.mpr ⟨row, member, reached⟩

end Instruction.GuardCertificate
