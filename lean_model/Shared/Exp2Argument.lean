import Exp2Decode
namespace GB300Math.Exp2Argument
open Exp2Boundary
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

noncomputable def magnitudeValue (word : Nat) : ℝ :=
  if Mufu.exponent word=0 then (Mufu.fraction word:ℝ)*(2:ℝ)^(-149:Int)
  else ((Mufu.fraction word:ℝ)+2^23)*(2:ℝ)^((Mufu.exponent word:Int)-150)

theorem low_fixed (word : Nat) (he : 0<Mufu.exponent word ∧ Mufu.exponent word<127) :
    positiveFixed word=((Mufu.fraction word:Int)+2^23)/2^(127-Mufu.exponent word) := by
  have hf : Mufu.fraction word<2^23 := by unfold Mufu.fraction;omega
  simp only [positiveFixed,show Mufu.exponent word≠0 by omega,ite_false,show ¬Mufu.exponent word≥127 by omega]
  split_ifs with hsmall
  · rfl
  · have hp : (2^24:Int)≤2^(127-Mufu.exponent word) := pow_le_pow_right₀ (by norm_num) (by omega)
    symm
    exact Int.ediv_eq_zero_of_lt (by omega) (by omega)

theorem positive_error (word : Nat) :
    0≤ magnitudeValue word-(positiveFixed word:ℝ)/2^23 ∧
    magnitudeValue word-(positiveFixed word:ℝ)/2^23≤1/2^23 := by
  have hf : Mufu.fraction word<2^23 := by unfold Mufu.fraction;omega
  have hfr : (Mufu.fraction word:ℝ)≤2^23 := by exact_mod_cast (show Mufu.fraction word≤2^23 by omega)
  by_cases hz : Mufu.exponent word=0
  · simp only [magnitudeValue,positiveFixed,hz,ite_true,Int.cast_zero,zero_div,sub_zero]
    constructor
    · positivity
    · norm_num;nlinarith
  by_cases hh : 127≤Mufu.exponent word
  · rw [positive_fixed_high word hh]
    simp only [magnitudeValue,hz,ite_false]
    have hp : (Mufu.exponent word:Int)-150=((Mufu.exponent word-127:Nat):Int)-23 := by omega
    rw [hp,zpow_sub₀ (by norm_num),zpow_natCast]
    push_cast
    norm_num
    constructor <;> nlinarith
  · rw [low_fixed word (by omega)]
    simp only [magnitudeValue,hz,ite_false]
    let m:Int:=(Mufu.fraction word:Int)+2^23
    let d:Int:=2^(127-Mufu.exponent word)
    have hd : 0<d := by dsimp [d];positivity
    have hrem : 0≤ m%d ∧ m%d<d := ⟨Int.emod_nonneg _ (by omega),Int.emod_lt_of_pos _ hd⟩
    have hid : m%d+d*(m/d)=m := Int.emod_add_mul_ediv m d
    have hrr : 0≤((m%d:Int):ℝ) ∧ ((m%d:Int):ℝ)≤(d:ℝ) := by exact_mod_cast (show 0≤ m%d ∧ m%d≤d by omega)
    have hidr : ((m%d:Int):ℝ)+(d:ℝ)*((m/d:Int):ℝ)=(m:ℝ) := by exact_mod_cast hid
    have hdr : (0:ℝ)<d := by exact_mod_cast hd
    have hp : (Mufu.exponent word:Int)-150=-((127-Mufu.exponent word:Nat):Int)-23 := by omega
    have hs : (2:ℝ)^((Mufu.exponent word:Int)-150)=1/((d:ℝ)*2^23) := by
      rw [hp,zpow_sub₀ (by norm_num),zpow_neg,zpow_natCast]
      dsimp [d]
      push_cast
      norm_num
      ring
    rw [hs]
    have hmc : (m:ℝ)=(Mufu.fraction word:ℝ)+2^23 := by dsimp [m];push_cast;norm_num
    rw [←hmc]
    change 0≤(m:ℝ)*(1/((d:ℝ)*2^23))-((m/d:Int):ℝ)/2^23 ∧ _
    have heq : (m:ℝ)*(1/((d:ℝ)*2^23))-((m/d:Int):ℝ)/2^23=((m%d:Int):ℝ)/((d:ℝ)*2^23) := by
      field_simp
      nlinarith only [hidr]
    rw [heq]
    constructor
    · exact div_nonneg hrr.1 (by positivity)
    · apply (div_le_iff₀ (by positivity : 0<(d:ℝ)*2^23)).mpr
      nlinarith

theorem argument_error (word : Nat) (hw : word<2^32) :
    |(signedFixed word:ℝ)/2^23-Exp2Bounds.finiteValue word|≤1/2^23 := by
  have hp:=positive_error word
  have hs:=sign_zero_iff word hw
  have hv : Exp2Bounds.finiteValue word=(if Mufu.sign word=0 then 1 else -1)*magnitudeValue word := by
    unfold Exp2Bounds.finiteValue magnitudeValue
    simp only [←hs]
    split_ifs <;> ring
  rw [hv]
  unfold signedFixed
  by_cases hn : Mufu.sign word=0
  · simp only [hn,ne_eq,not_true_eq_false,ite_false,ite_true,one_mul]
    exact abs_le.mpr ⟨by linarith,by linarith⟩
  · simp only [hn,ne_eq,not_false_eq_true,ite_true,ite_false]
    split_ifs <;> push_cast <;> exact abs_le.mpr ⟨by linarith,by linarith⟩

theorem high_exact (word : Nat) (he : 127≤Mufu.exponent word) :
    magnitudeValue word=(positiveFixed word:ℝ)/2^23 := by
  rw [positive_fixed_high word he]
  simp only [magnitudeValue,show Mufu.exponent word≠0 by omega,ite_false]
  have hp : (Mufu.exponent word:Int)-150=((Mufu.exponent word-127:Nat):Int)-23 := by omega
  rw [hp,zpow_sub₀ (by norm_num),zpow_natCast]
  push_cast
  norm_num
  ring

theorem nonzero_floor (word : Nat) (hw : word<2^32) (hn : signedFixed word≠0) :
    (signedFixed word:ℝ)/2^23≤Exp2Bounds.finiteValue word := by
  have hp:=positive_error word
  have hs:=sign_zero_iff word hw
  have hv : Exp2Bounds.finiteValue word=(if Mufu.sign word=0 then 1 else -1)*magnitudeValue word := by
    unfold Exp2Bounds.finiteValue magnitudeValue
    simp only [←hs]
    split_ifs <;> ring
  rw [hv]
  have hn' : positiveFixed word≠0 := by
    intro h
    apply hn
    simp only [signedFixed,h,Int.zero_emod,ne_eq,not_true_eq_false,ite_false,neg_zero,sub_zero]
    split_ifs <;> rfl
  by_cases hsign : Mufu.sign word=0
  · simp only [signedFixed,hsign,ne_eq,not_true_eq_false,ite_false,ite_true,one_mul]
    linarith
  · simp only [signedFixed,hsign,ne_eq,not_false_eq_true,ite_true,ite_false]
    by_cases hrem : positiveFixed word%2^23=0
    · have hh : 127≤Mufu.exponent word := by
        by_contra hlow
        have hr:=positive_fixed_low word (by omega)
        have hz:=fixed_nonnegative word
        omega
      have hExact:=high_exact word hh
      simp only [hrem,not_true_eq_false,ite_false,sub_zero,Int.cast_neg]
      linarith
    · simp only [hrem,not_false_eq_true,ite_true]
      push_cast
      linarith

#print axioms high_exact
#print axioms nonzero_floor
#print axioms low_fixed
#print axioms positive_error
#print axioms argument_error
end GB300Math.Exp2Argument
