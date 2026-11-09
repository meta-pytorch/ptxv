import MufuRcp64FullOutput
set_option maxRecDepth 8000
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange
open MufuRcp32Bounds

theorem high_word_scaled_fields (w f n e : Nat)
    (hw : w < 2^32) (hs : w &&& 0x80000000 = 0)
    (he : w/2^20%2048 = 1+e) (hf : w%2^20 = f)
    (hn : Fp64Seed.interpolate false (f*8) = n) (ee : e < 2044) :
    Fp64Seed.highWord false w =
      ((if n = 2^24 then 2045 else 2044)-e)*2^20 ||| (n%2^23)/8 := by
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hw]
  dsimp only
  rw [hs,he,hf]
  simp only [Bool.false_eq_true,false_and,ite_false,Nat.zero_mul,Nat.zero_add]
  rw [hn]
  have hne : ¬1+e = 2047 := by omega
  have hnz : ¬1+e = 0 := by omega
  simp only [hne,hnz,false_and,ite_false,Nat.zero_or]
  by_cases h : n = 2^24
  · simp only [ite_eq_left h]
    have hp : ¬(2046-((1+e : Nat) : Int)-0) ≤ 0 := by omega
    rw [ite_eq_right hp]
    have hh : (2046-((1+e : Nat) : Int)-0).toNat = 2045-e := by omega
    rw [hh]
  · simp only [ite_eq_right h]
    have hp : ¬(2046-((1+e : Nat) : Int)-1) ≤ 0 := by omega
    rw [ite_eq_right hp]
    have hh : (2046-((1+e : Nat) : Int)-1).toNat = 2044-e := by omega
    rw [hh]

theorem high_word_scaled (e : Fin 2044) (f : Fin (2^20)) :
    Fp64Seed.highWord false ((1+e.val)*2^20+f.val) =
      let n := Fp64Seed.interpolate false (f.val*8)
      ((if n = 2^24 then 2045 else 2044)-e.val)*2^20 + (n%2^23)/8 := by
  have he := e.isLt
  have hf := f.isLt
  have hw : (1+e.val)*2^20+f.val < 2^32 := by omega
  have hs : ((1+e.val)*2^20+f.val) &&& 0x80000000 = 0 := by
    exact sign_zero_of_lt ((1+e.val)*2^20+f.val) (by omega)
  have hx : ((1+e.val)*2^20+f.val)/2^20%2048 = 1+e.val := by omega
  have hm : ((1+e.val)*2^20+f.val)%2^20 = f.val := by omega
  generalize hn : Fp64Seed.interpolate false (f.val*8) = n
  have frac : (n%2^23)/8 < 2^20 := by omega
  have hor (a : Nat) : a*2^20 ||| ((n%2^23)/8) =
      a*2^20 + ((n%2^23)/8) := by
    simpa only [Nat.mul_comm] using (Nat.two_pow_add_eq_or_of_lt frac a).symm
  rw [high_word_scaled_fields _ _ _ _ hw hs hx hm hn he]
  exact hor _

theorem decode_scaled_max (e : Fin 2044) :
    (Model.ofNatBits (((2045-e.val)*2^20)*2^32) : Model FloatFormat.binary64).toRat? =
      some ((2 : Rat)^21 * (2 : Rat)^(1001-(e.val : Int))) := by
  have he := e.isLt
  let eo : Fin 2046 := ⟨2044-e.val,by omega⟩
  have d := decode_normal_high eo (0 : Fin (2^20))
  simp only [Fin.val_zero,Nat.add_zero,Nat.cast_pow,Nat.cast_ofNat] at d
  have hw : (1+eo.val)*2^20 = (2045-e.val)*2^20 := by dsimp [eo]; omega
  rw [hw] at d
  apply d.trans
  apply congrArg some
  have hp : (eo.val : Int)-1042 = (1001-(e.val : Int))+1 := by dsimp [eo]; omega
  rw [hp,zpow_add₀ (by norm_num : (2 : Rat) ≠ 0),zpow_one]
  ring

theorem decode_scaled_other (e : Fin 2044) (n : Nat) (lo : 2^23 ≤ n) (hi : n < 2^24) :
    (Model.ofNatBits (((2044-e.val)*2^20+(n%2^23)/8)*2^32) : Model FloatFormat.binary64).toRat? =
      some (((n/8 : Nat) : Rat)*(2 : Rat)^(1001-(e.val : Int))) := by
  have he := e.isLt
  let eo : Fin 2046 := ⟨2043-e.val,by omega⟩
  let fo : Fin (2^20) := ⟨(n%2^23)/8,by omega⟩
  have d := decode_normal_high eo fo
  have hw : (1+eo.val)*2^20+fo.val = (2044-e.val)*2^20+(n%2^23)/8 := by dsimp [eo,fo]; omega
  rw [hw] at d
  apply d.trans
  apply congrArg some
  have hp : (eo.val : Int)-1042 = 1001-(e.val : Int) := by dsimp [eo]; omega
  have hm : 2^20+fo.val = n/8 := by dsimp [fo]; omega
  rw [hp,hm]

theorem decode_scaled_seed (e : Fin 2044) (n : Nat) (lo : 2^23 ≤ n) (hi : n ≤ 2^24) :
    (Model.ofNatBits ((((if n = 2^24 then 2045 else 2044)-e.val)*2^20+(n%2^23)/8)*2^32) : Model FloatFormat.binary64).toRat? =
      some (((n/8 : Nat) : Rat)*(2 : Rat)^(1001-(e.val : Int))) := by
  by_cases h : n = 2^24
  · rw [ite_eq_left h]
    have hm : (n%2^23)/8 = 0 := by omega
    have hq : n/8 = 2^21 := by omega
    rw [hm,hq,Nat.add_zero]
    exact decode_scaled_max e
  · rw [ite_eq_right h]
    exact decode_scaled_other e n lo (by omega)

#print axioms high_word_scaled_fields
#print axioms high_word_scaled
#print axioms decode_scaled_max
#print axioms decode_scaled_other
#print axioms decode_scaled_seed
end GB300Math.Rcp64Bounds
