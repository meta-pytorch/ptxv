/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhCentralData
import SqrtBounds
namespace GB300Math.TanhCentralInteger
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def coeff (e j:Nat) : Int×Int×Int := (TanhTables.binades[e]!).2[j]!
def scale (e:Nat) := 184-min (e+119) 126
def cost (e j:Nat) : Rat :=
  let (_a,b,c):=coeff e j
  (|((b:Int):Rat)| *2^17+|((c:Int):Rat)| *(2*131071+1+4718593))/2^(scale e)

theorem costs : ∀e:Fin 11,∀j:Fin 16,cost e.val j.val≤1/400000 := by decide +kernel

theorem square_difference (k:Nat) (hk:k<2^17) (r:ℝ) (hr:0≤r ∧ r≤1) :
    0≤((k:ℝ)+r)^2-(Fp64Seed.squareTruncated k 19:ℝ) ∧
    ((k:ℝ)+r)^2-(Fp64Seed.squareTruncated k 19:ℝ)≤2*131071+1+4718593 := by
  have hs : Fp64Seed.squareTruncated k 19≤k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hd:=SqrtBounds.square_drop_bound k
  have hsub : ((k*k-Fp64Seed.squareTruncated k 19:Nat):ℝ)≤4718593 := by exact_mod_cast hd
  rw [Nat.cast_sub hs,Nat.cast_mul] at hsub
  have hsr : (Fp64Seed.squareTruncated k 19:ℝ)≤(k:ℝ)*(k:ℝ) := by exact_mod_cast hs
  have hkr : (k:ℝ)≤131071 := by exact_mod_cast (show k≤131071 by omega)
  have hkp : (0:ℝ)≤k := Nat.cast_nonneg _
  have hrr:=mul_nonneg hr.1 (show 0≤1-r by linarith)
  have hpr:=mul_le_mul_of_nonneg_left hr.2 hkp
  constructor <;> nlinarith

theorem interpolation_error (a b c:Int) (f k:Nat) (hk:k<2^17) (r:ℝ) (hr:0≤r ∧ r≤1) :
    |((a*2^34+b*(k:Int)*2^17+c*(Fp64Seed.squareTruncated k 19:Int):Int):ℝ)/2^f-
       ((a:ℝ)*2^34+(b:ℝ)*((k:ℝ)+r)*2^17+(c:ℝ)*((k:ℝ)+r)^2)/2^f|≤
      (|(b:ℝ)| *2^17+|(c:ℝ)| *(2*131071+1+4718593))/2^f := by
  have hd:=square_difference k hk r hr
  have hi : ((a*2^34+b*(k:Int)*2^17+c*(Fp64Seed.squareTruncated k 19:Int):Int):ℝ)-
      ((a:ℝ)*2^34+(b:ℝ)*((k:ℝ)+r)*2^17+(c:ℝ)*((k:ℝ)+r)^2)=
      -((b:ℝ)*r*2^17+(c:ℝ)*(((k:ℝ)+r)^2-(Fp64Seed.squareTruncated k 19:ℝ))) := by push_cast;ring
  rw [←sub_div,hi,abs_div,abs_neg,abs_of_pos (by positivity : (0:ℝ)<2^f)]
  apply div_le_div_of_nonneg_right _ (by positivity)
  have h:=abs_add_le ((b:ℝ)*r*2^17) ((c:ℝ)*(((k:ℝ)+r)^2-(Fp64Seed.squareTruncated k 19:ℝ)))
  rw [abs_mul,abs_mul,abs_mul,abs_of_nonneg hr.1,abs_of_pos (by positivity : (0:ℝ)<2^17),abs_of_nonneg hd.1] at h
  have h1:=mul_le_mul_of_nonneg_left hr.2 (abs_nonneg (b:ℝ))
  have h2:=mul_le_mul_of_nonneg_left hd.2 (abs_nonneg (c:ℝ))
  nlinarith

#print axioms costs
#print axioms square_difference
#print axioms interpolation_error
end GB300Math.TanhCentralInteger
