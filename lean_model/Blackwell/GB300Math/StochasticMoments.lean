import GB300Math.StochasticCorrelation
namespace GB300Math.StochasticMoments
open StochasticSm103 StochasticValue StochasticPacked StochasticProbability StochasticCorrelation
open scoped BigOperators

def last (n : Nat) (hn : 0 < n) : Fin n := ⟨n-1,by omega⟩
def rareError (n : Nat) (hn : 0 < n) (r : Fin n) : Rat :=
  (if r = last n hn then 1 else 0) - 1/(n:Rat)

theorem rare_mean_zero (n : Nat) (hn : 0 < n) :
    (∑ r : Fin n, rareError n hn r) / (n:Rat) = 0 := by
  have hn' : (n:Rat) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  simp [rareError, Finset.sum_sub_distrib, hn']

theorem rare_square_mean (n : Nat) (hn : 0 < n) :
    (∑ r : Fin n, (rareError n hn r)^2) / (n:Rat) =
      ((n:Rat)-1)/(n:Rat)^2 := by
  have hn' : (n:Rat) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  have hpoint : ∀ r : Fin n, (rareError n hn r)^2 =
      (if r = last n hn then 1 else 0) * (1-2/(n:Rat)) + 1/(n:Rat)^2 := by
    intro r
    by_cases h : r = last n hn
    · simp [rareError,h]; ring
    · simp [rareError,h]
  simp_rw [hpoint]
  simp only [Finset.sum_add_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  simp
  field_simp
  ring

theorem coupled_square_mean (n : Nat) (hn : 0 < n) (u : Rat) :
    (∑ r : Fin n, (u*rareError n hn r + u*rareError n hn r)^2) / (n:Rat) =
      4*u^2*((n:Rat)-1)/(n:Rat)^2 := by
  have heq : ∀ r : Fin n, (u*rareError n hn r + u*rareError n hn r)^2 =
      4*u^2*(rareError n hn r)^2 := by intro r; ring
  simp_rw [heq]
  rw [← Finset.mul_sum]
  rw [mul_div_assoc, rare_square_mean n hn]
  ring

theorem rare_sum_zero (n : Nat) (hn : 0 < n) :
    (∑ r : Fin n, rareError n hn r) = 0 := by
  have h := rare_mean_zero n hn
  have hn' : (n:Rat) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  exact (div_eq_zero_iff).mp h |>.resolve_right hn'

theorem independent_square_mean (n : Nat) (hn : 0 < n) (u : Rat) :
    (∑ r : Fin n, ∑ s : Fin n, (u*rareError n hn r + u*rareError n hn s)^2) /
        (n:Rat)^2 = 2*u^2*((n:Rat)-1)/(n:Rat)^2 := by
  have hn' : (n:Rat) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  have hm := rare_square_mean n hn
  have hs : (∑ r : Fin n, (rareError n hn r)^2) = ((n:Rat)-1)/(n:Rat) := by
    apply (div_left_inj' hn').mp
    rw [hm]
    field_simp
  have hp : ∀ r s : Fin n, (u*rareError n hn r + u*rareError n hn s)^2 =
      u^2*(rareError n hn r)^2 +
      2*u^2*rareError n hn r*rareError n hn s + u^2*(rareError n hn s)^2 := by
    intros; ring
  simp_rw [hp]
  simp only [Finset.sum_add_distrib,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  simp_rw [← Finset.mul_sum]
  rw [rare_sum_zero n hn]
  simp only [mul_zero, Finset.sum_const_zero, add_zero]
  rw [hs]
  field_simp
  ring

theorem small_quantum_round (f : Format) (hf : isPair f = false) (r : Nat) :
    roundMantissa (quantum f) r (23-fractionBits f) (entropyBits f) =
      if r%65536 = 65535 then 1 else 0 := by
  cases f <;> simp only [isPair, Bool.true_eq_false] at hf
  all_goals norm_num [quantum,fractionBits,entropyBits,parameters,isPair,roundMantissa]
  all_goals split <;> omega

theorem reverse_last_iff (r : Fin 65536) :
    reverseBits 16 r.val = 65535 ↔ r.val = 65535 := by
  constructor
  · intro h
    exact reverseBits_injective_bounded 16 r.val 65535 r.isLt (by decide)
      (h.trans (reverseBits_all_ones 16).symm)
  · intro h
    rw [h]
    exact reverseBits_all_ones 16

def ulp (f : Format) : Rat := 1/(2:Rat)^(fractionBits f)
def sourceValue (f : Format) : Rat := 1+(quantum f:Rat)/8388608

theorem source_quantum_value (f : Format) (hf : isPair f = false) :
    sourceValue f = 1+ulp f/65536 := by
  cases f <;> simp only [isPair, Bool.true_eq_false] at hf
  all_goals norm_num [sourceValue,ulp,quantum,fractionBits,entropyBits,parameters,isPair]

theorem conversion_error (f : Format) (hf : isPair f = false) (r : Fin 65536) :
    unitOutput f (quantum f) r.val - sourceValue f =
      ulp f*rareError 65536 (by decide) r := by
  rw [unitOutput, small_quantum_round f hf r.val, Nat.mod_eq_of_lt r.isLt,
    source_quantum_value f hf]
  have heq : (r = last 65536 (by decide)) ↔ r.val = 65535 := by
    simp [last, Fin.ext_iff]
  by_cases h : r.val = 65535 <;> simp [rareError,ulp,heq,h] <;> ring

theorem reversed_output_equal (f : Format) (hf : isPair f = false) (r : Fin 65536) :
    unitOutput f (quantum f) (reverseBits 16 r.val) = unitOutput f (quantum f) r.val := by
  simp only [unitOutput,small_quantum_round f hf,
    Nat.mod_eq_of_lt (reverseBits_bound 16 r.val),Nat.mod_eq_of_lt r.isLt, reverse_last_iff r]

def coupledError (f : Format) (r : Fin 65536) : Rat :=
  unitOutput f (quantum f) r.val + unitOutput f (quantum f) (reverseBits 16 r.val) -
    2*sourceValue f

theorem coupled_error (f : Format) (hf : isPair f = false) (r : Fin 65536) :
    coupledError f r = ulp f*rareError 65536 (by decide) r +
      ulp f*rareError 65536 (by decide) r := by
  unfold coupledError
  rw [reversed_output_equal f hf r]
  have h := conversion_error f hf r
  linarith

theorem coupled_variance (f : Format) (hf : isPair f = false) :
    (∑ r : Fin 65536, (coupledError f r)^2) / 65536 =
      4*(ulp f)^2*65535/4294967296 := by
  simp_rw [coupled_error f hf]
  have h := coupled_square_mean 65536 (by decide) (ulp f)
  simp only [Nat.cast_ofNat] at h
  rw [show (65536:Rat)-1=65535 by norm_num, show (65536:Rat)^2=4294967296 by norm_num] at h
  exact h

theorem packed_pair_values (f : Format) (hf : isPair f = false) (relu : Bool)
    (r : Fin 65536) (group : Nat) (hg : group < 2) :
    finiteValue f (packedCode f relu r.val group (2*group)) =
        some (unitOutput f (quantum f) r.val) ∧
      finiteValue f (packedCode f relu r.val group (2*group+1)) =
        some (unitOutput f (quantum f) r.val) := by
  have hc : laneCount f = 4 := by simp [laneCount,hf]
  have ht : quantum f < 8388608 := by
    cases f <;> decide
  unfold packedCode
  rw [packed_lane f relu true _ _ _ _ _ _ (by omega),
    packed_lane f relu true _ _ _ _ _ _ (by omega)]
  simp only [inputLane,ite_self]
  rw [even_entropy f hf r.val group r.isLt hg,odd_entropy f hf r.val group r.isLt hg,
    unit_conversion_value f relu true (quantum f) r.val ht,
    unit_conversion_value f relu true (quantum f) (reverseBits 16 r.val) ht,
    reversed_output_equal f hf r]
  exact ⟨rfl,rfl⟩

def independentError (f : Format) (r s : Fin 65536) : Rat :=
  unitOutput f (quantum f) r.val + unitOutput f (quantum f) s.val - 2*sourceValue f

theorem independent_error (f : Format) (hf : isPair f = false) (r s : Fin 65536) :
    independentError f r s = ulp f*rareError 65536 (by decide) r +
      ulp f*rareError 65536 (by decide) s := by
  unfold independentError
  have hr := conversion_error f hf r
  have hs := conversion_error f hf s
  linarith

theorem independent_variance (f : Format) (hf : isPair f = false) :
    (∑ r : Fin 65536, ∑ s : Fin 65536, (independentError f r s)^2) / 4294967296 =
      2*(ulp f)^2*65535/4294967296 := by
  simp_rw [independent_error f hf]
  have h := independent_square_mean 65536 (by decide) (ulp f)
  simp only [Nat.cast_ofNat] at h
  rw [show (65536:Rat)-1=65535 by norm_num, show (65536:Rat)^2=4294967296 by norm_num] at h
  exact h

theorem coupled_twice_independent (f : Format) (hf : isPair f = false) :
    (∑ r : Fin 65536, (coupledError f r)^2) / 65536 =
      2 * ((∑ r : Fin 65536, ∑ s : Fin 65536, (independentError f r s)^2) / 4294967296) := by
  rw [coupled_variance f hf,independent_variance f hf]
  ring

theorem coupled_mean_zero (f : Format) (hf : isPair f = false) :
    (∑ r : Fin 65536, coupledError f r) / 65536 = 0 := by
  simp_rw [coupled_error f hf]
  simp [Finset.sum_add_distrib,← Finset.mul_sum,rare_sum_zero]

#print axioms packed_pair_values
#print axioms independent_error
#print axioms independent_variance
#print axioms coupled_twice_independent
#print axioms coupled_mean_zero
#print axioms rare_sum_zero
#print axioms independent_square_mean
#print axioms small_quantum_round
#print axioms reverse_last_iff
#print axioms source_quantum_value
#print axioms conversion_error
#print axioms reversed_output_equal
#print axioms coupled_error
#print axioms coupled_variance
#print axioms rare_mean_zero
#print axioms rare_square_mean
#print axioms coupled_square_mean
end GB300Math.StochasticMoments
