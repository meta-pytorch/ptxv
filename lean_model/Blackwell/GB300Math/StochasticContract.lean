/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticSigned
import GB300Math.StochasticSaturation
import GB300Math.StochasticPacked
namespace GB300Math.StochasticContract
open StochasticSm103 StochasticValue StochasticNormal StochasticFinite StochasticSaturation

def positiveError (f : Format) (e t : Nat) (input output : Rat) : Prop :=
  (e ≤ 127-bias f ∧ |output-input| < StochasticUnderflow.spacing f) ∨
    ∃ j : Nat, 1 ≤ j ∧ e+bias f = j+127 ∧
      ((j*2^(fractionBits f) ≤ maximum f ∧
          t ≤ (maximum f-j*2^(fractionBits f))*2^(23-fractionBits f) ∧
          |output-input| < (2:Rat)^((j:Int)-bias f)/(2:Rat)^(fractionBits f)) ∨
        (maximum f ≤ j*2^(fractionBits f)+t/2^(23-fractionBits f) ∧ output=maximumValue f))

theorem positive_saturated_contract (f : Format) (relu : Bool) (e t r : Nat)
    (he : e ≤ 254) (ht : t < 8388608) :
    ∃ input output : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t) = some input ∧
      finiteValue f (lane f relu true (e*8388608+t) r) = some output ∧
      positiveError f e t input output := by
  rcases finite_domain_partition f e t with hu | ⟨j,hj,hb,hd⟩
  · obtain ⟨input,hi,ho,herr⟩ := StochasticUnderflow.full_error f relu true e t r hu ht
    exact ⟨input,_,hi,ho,Or.inl ⟨hu,herr⟩⟩
  · have hb' : bias f ≤ 127 := by cases f <;> decide
    have he' : 1 ≤ e := by omega
    rcases hd with ⟨hbase,hcap⟩ | hmax
    · obtain ⟨input,hi,ho,herr⟩ := finite_full_error f relu true e j t r he' he hj hb hbase hcap ht
      exact ⟨input,_,hi,ho,Or.inr ⟨j,hj,hb,Or.inl ⟨hbase,hcap,herr⟩⟩⟩
    · refine ⟨_,maximumValue f,fp32_input e t he' he ht,
        saturated_value f relu e j t r he' he hj hb ht hmax,?_⟩
      exact Or.inr ⟨j,hj,hb,Or.inr ⟨hmax,rfl⟩⟩

theorem positive_lane_bounded (f : Format) (e t r : Nat)
    (he : e ≤ 254) (ht : t < 8388608) :
    lane f false true (e*8388608+t) r ≤ maximum f := by
  have hx32 : e*8388608+t < 4294967296 := by omega
  have hs : (e*8388608+t)/2147483648 = 0 := by omega
  have heq : (e*8388608+t)/8388608%256 = e := by omega
  have hne : e ≠ 255 := by omega
  simp only [lane,Nat.mod_eq_of_lt hx32,hs,heq,hne,decide_false,Bool.false_and,
    Bool.and_false,Bool.false_eq_true,↓reduceIte,zero_mul,zero_add,Bool.not_true]
  exact Nat.min_le_right _ _

theorem negative_saturated_contract (f : Format) (e t r : Nat)
    (he : e ≤ 254) (ht : t < 8388608) :
    ∃ input output : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t+2147483648) = some (-input) ∧
      finiteValue f (lane f false true (e*8388608+t+2147483648) r) = some (-output) ∧
      positiveError f e t input output := by
  obtain ⟨input,output,hi,ho,herr⟩ := positive_saturated_contract f false e t r he ht
  refine ⟨input,output,?_,?_,herr⟩
  · exact StochasticSigned.rational_negative 8 23 127 0x7f7fffff _ input (by omega) (by decide) hi
  · exact StochasticSigned.finite_negative_value f true _ r output (by omega)
      (positive_lane_bounded f e t r he ht) ho

theorem packed_positive_saturated_contract (f : Format) (relu : Bool) (a b c d entropy i e t : Nat)
    (hi : i < StochasticPacked.laneCount f) (he : e ≤ 254) (ht : t < 8388608)
    (hx : StochasticPacked.inputLane a b c d i = e*8388608+t) :
    ∃ input output : Rat,
      rationalValue 8 23 127 0x7f7fffff (e*8388608+t) = some input ∧
      finiteValue f ((packed f relu true a b c d entropy /
        2^(StochasticPacked.laneWidth f*i)) % 2^(StochasticPacked.laneWidth f)) = some output ∧
      positiveError f e t input output := by
  rw [StochasticPacked.packed_lane f relu true a b c d entropy i hi,hx]
  exact positive_saturated_contract f relu e t (entropyLane f entropy i) he ht

#print axioms positive_saturated_contract
#print axioms positive_lane_bounded
#print axioms negative_saturated_contract
#print axioms packed_positive_saturated_contract
end GB300Math.StochasticContract
