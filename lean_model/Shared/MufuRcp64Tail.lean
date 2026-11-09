import MufuRcp64ScaleAccuracy
import MufuRcp32Tail
set_option maxRecDepth 12000
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange
open MufuRcp32Bounds

theorem seed_maximum_iff_zero (f : Fin (2^20)) :
    Fp64Seed.interpolate false (f.val*8) = 2^24 ↔ f.val = 0 := by
  let i : Fin (2^23) := ⟨f.val*8,by have hf:=f.isLt;omega⟩
  rw [same_interpolation]
  have h := maximum_nat_mantissa_iff_zero i
  simpa [i] using h

theorem top_exponents_fields (w e f n : Nat)
    (hw : w < 2^32) (hs : w &&& 0x80000000 = 0)
    (he : w/2^20%2048 = e) (hf : w%2^20 = f)
    (hn : Fp64Seed.interpolate false (f*8) = n) (ee : e = 2045 ∨ e = 2046) :
    Fp64Seed.highWord false w = if e=2045 ∧ n=2^24 then 2^20 else 0 := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [hs,he,hf]
  have hne : ¬e=2047 := by omega
  have hnz : ¬e=0 := by omega
  simp only [Bool.false_eq_true,false_and,ite_false,Nat.zero_mul,Nat.zero_add,hne,hnz,Nat.zero_or]
  rw [hn]
  clear hn he hf hs hw
  rcases ee with ee | ee <;> subst e
  · by_cases h : n=2^24
    · subst n
      norm_num
    · simp only [h,ite_false,and_false]
      norm_num
  · by_cases h : n=2^24
    · subst n
      norm_num
    · simp only [h,ite_false,and_false]
      norm_num

theorem high_penultimate (f : Fin (2^20)) :
    Fp64Seed.highWord false (2045*2^20+f.val) = if f.val=0 then 2^20 else 0 := by
  have hf:=f.isLt
  have hs : (2045*2^20+f.val) &&& 0x80000000 = 0 := sign_zero_of_lt (2045*2^20+f.val) (by omega)
  rw [top_exponents_fields _ 2045 f.val _ (by omega) hs (by omega) (by omega) rfl (Or.inl rfl)]
  simp only [true_and,seed_maximum_iff_zero]

theorem high_highest (f : Fin (2^20)) :
    Fp64Seed.highWord false (2046*2^20+f.val) = 0 := by
  have hf:=f.isLt
  have hs : (2046*2^20+f.val) &&& 0x80000000 = 0 := sign_zero_of_lt (2046*2^20+f.val) (by omega)
  rw [top_exponents_fields _ 2046 f.val _ (by omega) hs (by omega) (by omega) rfl (Or.inr rfl)]
  simp

theorem boundary_full_bits (low : Fin (2^32)) :
    Fp64Seed.evaluate false (2045*2^52+low.val) = 2^52 := by
  have hl:=low.isLt
  have hh : (2045*2^52+low.val)/2^32 = 2045*2^20 := by omega
  unfold Fp64Seed.evaluate
  rw [hh]
  have h := high_penultimate (0 : Fin (2^20))
  simp only [Fin.val_zero,Nat.add_zero,ite_true] at h
  rw [h]
  decide +kernel

theorem tail_full_bits (e : Fin 2046) (f : Fin (2^20)) (low : Fin (2^32))
    (large : 2044 ≤ e.val) (not_boundary : ¬(e.val=2044 ∧ f.val=0)) :
    Fp64Seed.evaluate false (((1+e.val)*2^20+f.val)*2^32+low.val) = 0 := by
  have he:=e.isLt
  have hl:=low.isLt
  have hh : (((1+e.val)*2^20+f.val)*2^32+low.val)/2^32 = (1+e.val)*2^20+f.val := by omega
  unfold Fp64Seed.evaluate
  rw [hh]
  have cases : e.val=2044 ∨ e.val=2045 := by omega
  rcases cases with ee | ee
  · rw [ee]
    change Fp64Seed.highWord false (2045*2^20+f.val)*2^32=0
    rw [high_penultimate]
    have hf : f.val ≠ 0 := by omega
    simp [hf]
  · rw [ee]
    change Fp64Seed.highWord false (2046*2^20+f.val)*2^32=0
    rw [high_highest,Nat.zero_mul]

theorem boundary_real_error (low : Nat) :
    let x : ℝ := ((2^52+low : Nat) : ℝ)*(2 : ℝ)^(970 : Int)
    let y : ℝ := (2 : ℝ)^(-1022 : Int)
    |y-1/x|/|1/x| = (low : ℝ)/2^52 := by
  dsimp only
  have hx : ((2^52+low : Nat) : ℝ)*(2 : ℝ)^(970 : Int) ≠ 0 := by positivity
  rw [relative_eq_product _ _ hx]
  have product : (((2^52+low : Nat) : ℝ)*(2 : ℝ)^(970 : Int))*(2 : ℝ)^(-1022 : Int)-1 = (low : ℝ)/2^52 := by
    calc
      _ = ((2^52+low : Nat) : ℝ)*((2 : ℝ)^(970 : Int)*(2 : ℝ)^(-1022 : Int))-1 := by ring
      _ = _ := by
        rw [←zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
        norm_num
        ring
  rw [product,abs_of_nonneg (by positivity)]

theorem boundary_full_error (low : Fin (2^32)) :
    ∃ x y : Rat,
      (Model.ofNatBits (2045*2^52+low.val) : Model FloatFormat.binary64).toRat? = some x ∧
      (Model.ofNatBits (Fp64Seed.evaluate false (2045*2^52+low.val)) : Model FloatFormat.binary64).toRat? = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = (low.val : ℝ)/2^52 ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| < 1/(2^20 : ℝ) := by
  refine ⟨((2^52+low.val : Nat) : Rat)*(2 : Rat)^970,(2 : Rat)^(-1022 : Int),?_,?_,?_,?_⟩
  · have d := decode_full_scaled (2044 : Fin 2046) (0 : Fin (2^20)) low
    norm_num at d ⊢
    exact d
  · rw [boundary_full_bits]
    decide +kernel
  · have h:=boundary_real_error low.val
    norm_num at h ⊢
    exact h
  · have h:=boundary_real_error low.val
    norm_num at h ⊢
    rw [h]
    have hl : (low.val : ℝ) < 2^32 := by exact_mod_cast low.isLt
    nlinarith

#print axioms seed_maximum_iff_zero
#print axioms top_exponents_fields
#print axioms high_penultimate
#print axioms high_highest
#print axioms boundary_full_bits
#print axioms tail_full_bits
#print axioms boundary_real_error
#print axioms boundary_full_error
end GB300Math.Rcp64Bounds
