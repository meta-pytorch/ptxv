import GB300Math.StochasticFinite
namespace GB300Math.StochasticSaturation
open StochasticSm103 StochasticValue StochasticNormal

theorem full_normal_bits (f : Format) (relu satfinite : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127) (ht : t < 8388608) :
    lane f relu satfinite (e*8388608+t) r =
      min (j*2^(fractionBits f)+roundMantissa t r (23-fractionBits f) (entropyBits f))
        (if isPair f && !satfinite then maximum f+1 else maximum f) := by
  have hx : (e*8388608+t)%4294967296 = e*8388608+t := by omega
  have hxe : (e*8388608+t)/8388608%256 = e := by omega
  have hf : (e*8388608+t)%8388608 = t := by omega
  have hs : (e*8388608+t)/2147483648 = 0 := by omega
  have hmaxe : max e 1 = e := by omega
  have htexp : (e:Int)-127+bias f = (j:Int) := by omega
  have hdrop : max 0 (1-(j:Int)) = 0 := by omega
  have hbase : max 0 ((j:Int)-1) = ((j-1:Nat):Int) := by omega
  have hne : e ≠ 0 := by omega
  have hne255 : e ≠ 255 := by omega
  cases f <;> cases relu <;> cases satfinite <;>
    norm_num only [bias, maximum, fractionBits, parameters, entropyBits, isPair] at hb htexp ⊢ <;>
    norm_num only [lane, parameters, isPair] <;>
    simp only [hx, hxe, hf, hs, hmaxe, htexp, hdrop, hbase, hne, hne255] <;>
    clear hx hxe hf hs hmaxe htexp hdrop hbase hne hne255 hb he he' <;>
    norm_num [roundMantissa] <;> omega

theorem round_above_floor (m r d b : Nat) : m/2^d ≤ roundMantissa m r d b := by
  unfold roundMantissa
  apply Nat.div_le_div_right
  omega

theorem saturated_normal_bits (f : Format) (relu : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127) (ht : t < 8388608)
    (hmax : maximum f ≤ j*2^(fractionBits f)+t/2^(23-fractionBits f)) :
    lane f relu true (e*8388608+t) r = maximum f := by
  rw [full_normal_bits f relu true e j t r he he' hj hb ht]
  simp only [Bool.not_true, Bool.and_false, Bool.false_eq_true, ↓reduceIte]
  apply Nat.min_eq_right
  have h := round_above_floor t r (23-fractionBits f) (entropyBits f)
  omega

#print axioms full_normal_bits
#print axioms round_above_floor
#print axioms saturated_normal_bits


theorem finite_domain_partition (f : Format) (e t : Nat) :
    e ≤ 127-bias f ∨ ∃ j : Nat, 1 ≤ j ∧ e+bias f = j+127 ∧
      ((j*2^(fractionBits f) ≤ maximum f ∧
          t ≤ (maximum f-j*2^(fractionBits f))*2^(23-fractionBits f)) ∨
        maximum f ≤ j*2^(fractionBits f)+t/2^(23-fractionBits f)) := by
  by_cases he : e ≤ 127-bias f
  · exact Or.inl he
  · right
    have hb : bias f ≤ 127 := by cases f <;> decide
    let j := e+bias f-127
    have hj : 1 ≤ j := by dsimp [j]; omega
    have hsum : e+bias f = j+127 := by dsimp [j]; omega
    refine ⟨j,hj,hsum,?_⟩
    by_cases hbase : j*2^(fractionBits f) ≤ maximum f
    · by_cases hcap : t ≤ (maximum f-j*2^(fractionBits f))*2^(23-fractionBits f)
      · exact Or.inl ⟨hbase,hcap⟩
      · right
        have hd : maximum f-j*2^(fractionBits f) ≤ t/2^(23-fractionBits f) := by
          apply (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).mpr
          omega
        omega
    · right
      exact Nat.le_trans (Nat.le_of_lt (Nat.lt_of_not_ge hbase)) (Nat.le_add_right _ _)

def maximumValue (f : Format) : Rat :=
  (1+((maximum f%2^(fractionBits f):Nat):Rat)/(2:Rat)^(fractionBits f))*
    (2:Rat)^(((maximum f/2^(fractionBits f):Nat):Int)-bias f)

theorem maximum_value (f : Format) : finiteValue f (maximum f) = some (maximumValue f) := by
  have hrepr : (maximum f/2^(fractionBits f))*2^(fractionBits f)+
      maximum f%2^(fractionBits f) = maximum f := by
    have h := Nat.mod_add_div (maximum f) (2^(fractionBits f))
    rw [Nat.mul_comm] at h
    omega
  have hj : 0 < maximum f/2^(fractionBits f) := by cases f <;> decide
  have hq : maximum f%2^(fractionBits f) ≤ 2^(fractionBits f) :=
    Nat.le_of_lt (Nat.mod_lt _ (Nat.two_pow_pos _))
  have h := StochasticFinite.finite_code_value f _ _ hj (by omega) hq
  rw [hrepr] at h
  exact h

theorem saturated_value (f : Format) (relu : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127) (ht : t < 8388608)
    (hmax : maximum f ≤ j*2^(fractionBits f)+t/2^(23-fractionBits f)) :
    finiteValue f (lane f relu true (e*8388608+t) r) = some (maximumValue f) := by
  rw [saturated_normal_bits f relu e j t r he he' hj hb ht hmax]
  exact maximum_value f

theorem unsaturated_overflow (f : Format) (hf : isPair f = true) (relu : Bool) (e j t r : Nat)
    (he : 1 ≤ e) (he' : e ≤ 254) (hj : 1 ≤ j)
    (hb : e+bias f = j+127) (ht : t < 8388608)
    (hmax : maximum f+1 ≤ j*2^(fractionBits f)+t/2^(23-fractionBits f)) :
    lane f relu false (e*8388608+t) r = maximum f+1 := by
  rw [full_normal_bits f relu false e j t r he he' hj hb ht]
  simp only [hf, Bool.not_false, Bool.and_true, ↓reduceIte]
  apply Nat.min_eq_right
  have h := round_above_floor t r (23-fractionBits f) (entropyBits f)
  omega

#print axioms finite_domain_partition
#print axioms maximum_value
#print axioms saturated_value
#print axioms unsaturated_overflow
end GB300Math.StochasticSaturation
