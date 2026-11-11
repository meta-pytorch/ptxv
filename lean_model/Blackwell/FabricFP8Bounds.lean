/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricFP8
import FabricDirectedBounds

namespace FabricFP8Bounds
open FabricFP8

def selected (e4 : Bool) (x : Nat) : Nat := Id.run do
  let mut lower := 0
  for code in [:maximum e4+1] do
    if unit e4 code ≤ x then lower := code
  return lower

theorem selected_fold (e4 : Bool) (x : Nat) : selected e4 x =
    (List.range (maximum e4+1)).foldl (fun lower code => if unit e4 code ≤ x then code else lower) 0 := by
  have step : ∀ code lower, (if unit e4 code ≤ x then
      (pure (ForInStep.yield code) : Id (ForInStep Nat)) else pure (ForInStep.yield lower)) =
      pure (ForInStep.yield (if unit e4 code ≤ x then code else lower)) := by
    intro code lower
    split <;> rfl
  simp only [selected, Std.Legacy.Range.forIn_eq_forIn_range']
  simp_rw [step]
  simp [List.range_eq_range']

#print axioms selected_fold
end FabricFP8Bounds

namespace FabricFP8Bounds
open FabricFP8

def increasing (e4 : Bool) (code : Nat) : Bool := unit e4 code < unit e4 (code+1)
theorem e4_increasing : (List.range 127).all (increasing true) = true := by decide +kernel
theorem e5_increasing : (List.range 124).all (increasing false) = true := by decide +kernel

theorem next_gt (e4 : Bool) (code : Nat) (hc : code ≤ maximum e4) : unit e4 code < unit e4 (code+1) := by
  cases e4
  · change code ≤ 123 at hc
    have h := List.all_eq_true.mp e5_increasing code (List.mem_range.mpr (by omega))
    simpa [increasing] using h
  · change code ≤ 126 at hc
    have h := List.all_eq_true.mp e4_increasing code (List.mem_range.mpr (by omega))
    simpa [increasing] using h

theorem unit_le (e4 : Bool) (a b : Nat) (hab : a ≤ b) (hb : b ≤ maximum e4+1) :
    unit e4 a ≤ unit e4 b := by
  induction hab with
  | refl => exact Nat.le_refl _
  | @step b hab ih =>
    exact Nat.le_trans (ih (by omega)) (Nat.le_of_lt (next_gt e4 b (by omega)))

theorem test_cut (e4 : Bool) (x code i : Nat)
    (hi : i ≤ maximum e4) (hc : code ≤ maximum e4)
    (hx : unit e4 code ≤ x) (hx' : x < unit e4 (code+1)) :
    (unit e4 i ≤ x) ↔ i ≤ code := by
  constructor
  · intro h
    by_contra hnot
    have hle := unit_le e4 (code+1) i (by omega) (by omega)
    omega
  · intro h
    exact Nat.le_trans (unit_le e4 i code h (by omega)) hx

theorem fold_cut (n code : Nat) :
    (List.range n).foldl (fun lower i => if i ≤ code then i else lower) 0 = min code (n-1) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.foldl_append]
    simp only [List.foldl_cons, List.foldl_nil, ih]
    split <;> omega

theorem fold_congr (xs : List Nat) (f g : Nat → Nat → Nat)
    (h : ∀ a i, i ∈ xs → f a i = g a i) (a : Nat) : xs.foldl f a = xs.foldl g a := by
  induction xs generalizing a with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [h a x (by simp)]
    apply ih
    intro a i hi
    exact h a i (by simp [hi])

theorem selected_interval (e4 : Bool) (x code : Nat)
    (hc : code ≤ maximum e4)
    (hx : unit e4 code ≤ x) (hx' : x < unit e4 (code+1)) : selected e4 x = code := by
  rw [selected_fold]
  have congruence : (List.range (maximum e4+1)).foldl
      (fun lower i => if unit e4 i ≤ x then i else lower) 0 =
      (List.range (maximum e4+1)).foldl
      (fun lower i => if i ≤ code then i else lower) 0 := by
    apply fold_congr
    intro a i hi
    simp only [test_cut e4 x code i (by simpa using List.mem_range.mp hi) hc hx hx']
  rw [congruence, fold_cut]
  simp only [Nat.add_sub_cancel, Nat.min_eq_left hc]

#print axioms unit_le
#print axioms selected_interval
end FabricFP8Bounds

namespace FabricFP8Bounds
open FabricFP8

theorem unit_zero (e4 : Bool) : unit e4 0 = 0 := by cases e4 <;> decide

theorem exists_interval (e4 : Bool) (x n : Nat) (hx : x < unit e4 n) :
    ∃ code, code < n ∧ unit e4 code ≤ x ∧ x < unit e4 (code+1) := by
  induction n with
  | zero => rw [unit_zero] at hx; omega
  | succ n ih =>
    by_cases hn : x < unit e4 n
    · obtain ⟨code, hc, hl, hu⟩ := ih hn
      exact ⟨code, by omega, hl, hu⟩
    · exact ⟨n, by omega, by omega, hx⟩

theorem output_interval (e4 : Bool) (totals : Array Accum) (lane code : Nat) (total : Int)
    (ht : totals[lane]! = .finite total) (hc : code ≤ maximum e4)
    (hx : unit e4 code ≤ total.natAbs) (hx' : total.natAbs < unit e4 (code+1)) :
    output e4 totals lane = (if total < 0 then 128 else 0) +
      (if unit e4 code = total.natAbs then code else
        choose code (total.natAbs-unit e4 code) (unit e4 (code+1)-unit e4 code)
          (entropy e4 totals lane) (2^(7*width e4))) := by
  have hmax : total.natAbs < unit e4 (maximum e4+1) :=
    Nat.lt_of_lt_of_le hx' (unit_le e4 (code+1) (maximum e4+1) (by omega) (by omega))
  simp only [output, ht, Nat.not_le.mpr hmax, ↓reduceIte]
  change (if unit e4 (selected e4 total.natAbs) = total.natAbs then
      (if total < 0 then 128 else 0) + selected e4 total.natAbs else
        (if total < 0 then 128 else 0) + choose (selected e4 total.natAbs) (total.natAbs-unit e4 (selected e4 total.natAbs))
          (unit e4 (selected e4 total.natAbs+1)-unit e4 (selected e4 total.natAbs))
          (entropy e4 totals lane) (2^(7*width e4))) = _
  rw [selected_interval e4 total.natAbs code hc hx hx']
  split <;> rfl

#print axioms exists_interval
#print axioms output_interval
end FabricFP8Bounds

namespace FabricFP8Bounds
open FabricFP8 FabricDirectedBounds

theorem model_bound (e4 : Bool) (totals : Array Accum) (lane : Nat) (total : Int)
    (ht : totals[lane]! = .finite total)
    (hfinite : total.natAbs < unit e4 (maximum e4+1)) :
    ∃ y, y ≤ maximum e4+1 ∧ output e4 totals lane = (if total < 0 then 128 else 0)+y ∧
      denominator e4 * Int.natAbs ((unit e4 y : Int)-total.natAbs) ≤ numerator e4 * total.natAbs := by
  obtain ⟨code, hc, hx, hx'⟩ := exists_interval e4 total.natAbs (maximum e4+1) hfinite
  have hcode : code ≤ maximum e4 := by omega
  let y := if unit e4 code = total.natAbs then code else
    choose code (total.natAbs-unit e4 code) (unit e4 (code+1)-unit e4 code)
      (entropy e4 totals lane) (2^(7*width e4))
  have hy : y = code ∨ y = code+1 := by
    dsimp [y]
    split
    · exact Or.inl rfl
    · exact choose_adjacent _ _ _ _ _
  refine ⟨y, ?_, output_interval e4 totals lane code total ht hcode hx hx', ?_⟩
  · rcases hy with h | h <;> omega
  · by_cases heq : unit e4 code = total.natAbs
    · simp [y, heq]
    · have hstrict : unit e4 code < total.natAbs := by omega
      apply sharp_directed_relative e4 code total.natAbs (unit e4 y) hcode hstrict hx'
      exact hy.imp (congrArg (unit e4)) (congrArg (unit e4))

#print axioms model_bound
end FabricFP8Bounds

namespace FabricFP8Bounds
open FabricFP8 FabricDirectedBounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem decode_agreement (e4 : Bool) (word : Nat) (hw : word < 256) :
    (Model.ofNatBits word : Model (scalarFormat e4)).toRat? =
      (decode e4 word).map (fun x => (x : Rat) / (if e4 then 512 else 65536)) := by
  cases e4
  · have h := List.all_eq_true.mp e5_decode_agreement word (List.mem_range.mpr hw)
    simpa [decodeAgrees] using h
  · have h := List.all_eq_true.mp e4_decode_agreement word (List.mem_range.mpr hw)
    simpa [decodeAgrees] using h

theorem code_small (e4 : Bool) (y : Nat) (hy : y ≤ maximum e4) : y < 128 := by
  cases e4 <;> simp_all [maximum] <;> omega

theorem decode_signed (e4 neg : Bool) (y : Nat) (hy : y ≤ maximum e4) :
    decode e4 ((if neg then 128 else 0)+y) =
      some ((if neg then -1 else 1) * (unit e4 y : Int)) := by
  have hs := code_small e4 y hy
  cases neg
  · simp [decode, Nat.mod_eq_of_lt hs, show ¬ y ≥ 256 by omega,
      show ¬ y ≥ 128 by omega, show ¬ y > maximum e4 by omega]
  · simp [decode, Nat.mod_eq_of_lt hs, show ¬ 128+y ≥ 256 by omega,
      show ¬ y > maximum e4 by omega]

#print axioms decode_agreement
#print axioms decode_signed
end FabricFP8Bounds

namespace FabricFP8Bounds
open FabricFP8 FabricDirectedBounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem floatlib_model_bound (e4 : Bool) (totals : Array Accum) (lane : Nat) (total : Int)
    (ht : totals[lane]! = .finite total)
    (hfinite : total.natAbs < unit e4 (maximum e4+1))
    (hout : output e4 totals lane % 128 ≤ maximum e4) :
    ∃ value : Int,
      (Model.ofNatBits (output e4 totals lane) : Model (scalarFormat e4)).toRat? =
        some ((value : Rat)/(if e4 then 512 else 65536)) ∧
      denominator e4 * (value-total).natAbs ≤ numerator e4 * total.natAbs := by
  obtain ⟨y, hy, ho, hbound⟩ := model_bound e4 totals lane total ht hfinite
  have hs : y < 128 := by cases e4 <;> simp_all [maximum] <;> omega
  have hsign : (if total < 0 then 128 else 0)+y < 256 := by split <;> omega
  have hmod : ((if total < 0 then 128 else 0)+y)%128 = y := by
    split <;> simp [Nat.mod_eq_of_lt hs]
  have hfy : y ≤ maximum e4 := by rw [ho, hmod] at hout; exact hout
  let value : Int := if total < 0 then -(unit e4 y : Int) else (unit e4 y : Int)
  have hd : decode e4 (output e4 totals lane) = some value := by
    have h := decode_signed e4 (decide (total < 0)) y hfy
    rw [ho]
    simpa [value] using h
  refine ⟨value, ?_, ?_⟩
  · rw [decode_agreement e4 (output e4 totals lane) (by rw [ho]; exact hsign), hd]
    rfl
  · have hdist : (value-total).natAbs = Int.natAbs ((unit e4 y : Int)-total.natAbs) := by
      dsimp [value]
      split <;> omega
    rw [hdist]
    exact hbound

#print axioms floatlib_model_bound
end FabricFP8Bounds
