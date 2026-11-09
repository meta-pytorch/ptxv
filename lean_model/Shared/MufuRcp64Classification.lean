import MufuRcp64Tail
import MufuRcp64NormalSign
import MufuRcp64SignedCore
set_option maxRecDepth 12000
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange
open MufuRcp32Bounds

abbrev hasSmallRcp64Error (e f : Nat) : Prop := e < 2044 ∨ e=2044 ∧ f=0

theorem positive_normal_classification (e : Fin 2046) (f : Fin (2^20)) (low : Fin (2^32)) :
    let w := ((1+e.val)*2^20+f.val)*2^32+low.val
    ∃ x y : Rat,
      (Model.ofNatBits w : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.evaluate false w) : Model FloatFormat.binary64).toRat? = some y ∧
      (if hasSmallRcp64Error e.val f.val then
        |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 9/(2^23 : ℝ)
       else |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 1) := by
  dsimp only
  by_cases hlo : e.val < 2044
  · obtain ⟨x,y,hx,hy,herr⟩ := positive_full_scaled_error ⟨e.val,hlo⟩ f low
    exact ⟨x,y,hx,hy,by simpa [hasSmallRcp64Error,hlo] using herr⟩
  · by_cases heq : e.val=2044 ∧ f.val=0
    · have hw : ((1+e.val)*2^20+f.val)*2^32+low.val = 2045*2^52+low.val := by rw [heq.1,heq.2];ring
      obtain ⟨x,y,hx,hy,herr,hbound⟩ := boundary_full_error low
      refine ⟨x,y,?_,?_,?_⟩
      · simpa only [hw] using hx
      · simpa only [hw] using hy
      · have hc : hasSmallRcp64Error e.val f.val := Or.inr heq
        rw [ite_eq_left hc]
        exact le_trans (le_of_lt hbound) (by norm_num)
    · let x : Rat := (((2^20+f.val)*2^32+low.val : Nat) : Rat)*(2 : Rat)^((e.val : Int)-1074)
      have hx : 0 < x := by dsimp [x];positivity
      refine ⟨x,0,decode_full_scaled e f low,?_,?_⟩
      · rw [tail_full_bits e f low (by omega) heq]
        decide +kernel
      · have hc : ¬hasSmallRcp64Error e.val f.val := by intro h;rcases h with h|h;exact hlo h;exact heq h
        rw [ite_eq_right hc]
        simp only [Rat.cast_zero]
        apply flushed_relative_error
        have hxr : (0 : ℝ) < (x : ℝ) := by exact_mod_cast hx
        exact ne_of_gt hxr

theorem negative_normal_classification (e : Fin 2046) (f : Fin (2^20)) (low : Fin (2^32)) :
    let w := ((1+e.val)*2^20+f.val)*2^32+low.val
    ∃ x y : Rat,
      (Model.ofNatBits (w+2^63) : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.evaluate false (w+2^63)) : Model FloatFormat.binary64).toRat? = some y ∧
      (if hasSmallRcp64Error e.val f.val then
        |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 9/(2^23 : ℝ)
       else |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 1) := by
  dsimp only
  obtain ⟨x,y,hx,hy,herr⟩ := positive_normal_classification e f low
  refine ⟨-x,-y,Rcp64Signed.signed_decode _ _ hx,?_,?_⟩
  · rw [full_rcp_normal_sign]
    exact Rcp64Signed.signed_decode _ _ hy
  · simpa only [Rat.cast_neg,one_div,inv_neg,neg_sub_neg,abs_sub_comm,abs_neg] using herr

theorem normal_domain_counts :
    2*2046*2^52 = 18428729675200069632 ∧
    2*2044*2^52+2*2^32 = 18410715285280522240 ∧
    2*(2*2^52-2^32) = 18014389919547392 ∧
    18410715285280522240+18014389919547392 = 18428729675200069632 := by decide +kernel

#print axioms positive_normal_classification
#print axioms negative_normal_classification
#print axioms normal_domain_counts
end GB300Math.Rcp64Bounds
