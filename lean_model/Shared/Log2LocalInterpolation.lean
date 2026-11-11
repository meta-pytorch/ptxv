/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2Interpolation
import Log2TightCells
namespace GB300Math.Log2LocalInterpolation
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

theorem facts (index : Nat) (hi : index<2^23) :
    let (a,b,c) := MufuTables.lg2.coefficients[index/131072]!
    let k : Int := (index%131072:Nat)
    let n : Int := a*2^34+857088*4096+b*k*2^22+c*k*k
    let z := Mufu.interpolate MufuTables.lg2 index;
    -(2^24:Int)≤z*2^24-n ∧
      z*2^24-n≤-c*(Log2TightCells.dropBound (index/16384):Nat) := by
  let segment : Fin 64 := ⟨index/131072,by omega⟩
  rcases ht : MufuTables.lg2.coefficients[segment.val]! with ⟨a,b,c⟩
  have hr := Log2Interpolation.ranges_valid segment
  rw [ht] at hr
  have hc : c≤0 := hr.2.2.2.2.2.1
  let k := index%131072
  let square := Fp64Seed.squareTruncated k 19
  let drop := k*k-square
  let limit := Log2TightCells.dropBound (index/16384)
  have hs : square≤k*k := by unfold square Fp64Seed.squareTruncated;exact Nat.sub_le _ _
  have hdrop : drop≤limit := by
    apply Log2Drop.square_drop
    dsimp [k]
    omega
  have hd : 0≤(drop:Int) ∧ (drop:Int)≤(limit:Int) := ⟨Int.natCast_nonneg _,by exact_mod_cast hdrop⟩
  have hsquare : (square:Int)=(k:Int)*(k:Int)-(drop:Int) := by
    have h : drop+square=k*k := Nat.sub_add_cancel hs
    have hc : (drop:Int)+(square:Int)=(k:Int)*(k:Int) := by exact_mod_cast h
    omega
  let n := a*2^34+857088*4096+b*(k:Int)*2^22+c*(k:Int)*(k:Int)
  let full := n-c*(drop:Int)
  have hz : Mufu.interpolate MufuTables.lg2 index=full/2^24 := by
    change ((MufuTables.lg2.coefficients[index/131072]!).1*4194304+857088+
      (MufuTables.lg2.coefficients[index/131072]!).2.1*(k:Int)*1024+
      ((MufuTables.lg2.coefficients[index/131072]!).2.2*(square:Int))/4096)/4096=full/16777216
    rw [show index/131072=segment.val by rfl,ht]
    dsimp only
    rw [Log2Interpolation.floor_join,hsquare]
    congr 1
    dsimp [full,n]
    ring
  have hrem : 0≤full%2^24 ∧ full%2^24≤2^24-1 := by omega
  have hid : full/2^24*2^24=full-full%2^24 := by omega
  have hl := mul_nonneg (show 0≤-c by omega) hd.1
  have hh := mul_le_mul_of_nonneg_left hd.2 (show 0≤-c by omega)
  change _≤_ ∧ _≤_
  rw [hz]
  change -2^24≤full/2^24*2^24-n ∧ full/2^24*2^24-n≤-c*(limit:Int)
  dsimp [full] at *
  constructor <;> nlinarith only [hid,hrem.1,hrem.2,hl,hh]

#print axioms facts
end GB300Math.Log2LocalInterpolation
