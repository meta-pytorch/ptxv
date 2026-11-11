import GB300Math.StochasticSm103
import Mathlib.Algebra.Order.Floor.BigOperators
import Mathlib.Tactic

open scoped BigOperators

namespace GB300Math.StochasticProbability

open StochasticSm103

def quantize (m r n q : Nat) : Nat := (m + r*q) / (n*q)

theorem uniform_sum (m n q : Nat) (hn : 0 < n) (hq : 0 < q) :
    (∑ r ∈ Finset.range n, quantize m r n q) = m / q := by
  have hh : (∑ r ∈ Finset.range n, (m / q + r) / n) = m / q := by
    have h := Int.sum_range_add_ediv (m / q : Nat) hn
    exact_mod_cast h
  rw [← hh]
  apply Finset.sum_congr rfl
  intro r _
  unfold quantize
  rw [Nat.mul_comm n q, ← Nat.div_div_eq_div_mul, Nat.add_mul_div_right _ _ hq]

theorem uniform_mean_numerator (m n q : Nat) (hn : 0 < n) (hq : 0 < q) :
    q * (∑ r ∈ Finset.range n, quantize m r n q) = m - m % q := by
  rw [uniform_sum m n q hn hq]
  have h := Nat.mod_add_div m q
  omega

theorem finite_entropy_bias (m n q : Nat) (hn : 0 < n) (hq : 0 < q) :
    q * (∑ r ∈ Finset.range n, quantize m r n q) ≤ m ∧
    m < q * (∑ r ∈ Finset.range n, quantize m r n q) + q := by
  rw [uniform_mean_numerator m n q hn hq]
  have h := Nat.mod_lt m hq
  have hle := Nat.mod_le m q
  omega

theorem exactly_unbiased_iff (m n q : Nat) (hn : 0 < n) (hq : 0 < q) :
    q * (∑ r ∈ Finset.range n, quantize m r n q) = m ↔ q ∣ m := by
  rw [uniform_mean_numerator m n q hn hq, Nat.dvd_iff_mod_eq_zero]
  have hle := Nat.mod_le m q
  omega

theorem model_mean (m drop bits : Nat) (h : bits ≤ drop) :
    2^(drop-bits) * (∑ r ∈ Finset.range (2^bits), roundMantissa m r drop bits) =
      m - m % 2^(drop-bits) := by
  have hpow : 2^drop = 2^bits * 2^(drop-bits) := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have heq : (∑ r ∈ Finset.range (2^bits), roundMantissa m r drop bits) =
      ∑ r ∈ Finset.range (2^bits), quantize m r (2^bits) (2^(drop-bits)) := by
    apply Finset.sum_congr rfl
    intro r hr
    have hr := Finset.mem_range.mp hr
    simp only [roundMantissa, quantize, Nat.mod_eq_of_lt hr, hpow]
  rw [heq]
  exact uniform_mean_numerator m _ _ (Nat.two_pow_pos _) (Nat.two_pow_pos _)

theorem reverseBits_injective_bounded (n x y : Nat) (hx : x < 2^n) (hy : y < 2^n)
    (h : reverseBits n x = reverseBits n y) : x = y := by
  induction n generalizing x y with
  | zero => simp at hx hy; omega
  | succ n ih =>
    have hbx := reverseBits_bound n (x/2)
    have hby := reverseBits_bound n (y/2)
    have hpow : 2^(n+1) = 2^n * 2 := Nat.pow_succ _ _
    rw [hpow] at hx hy
    have hx2 : x/2 < 2^n := by omega
    have hy2 : y/2 < 2^n := by omega
    have hxmod : x%2 = 0 ∨ x%2 = 1 := by omega
    have hymod : y%2 = 0 ∨ y%2 = 1 := by omega
    have hparts : x%2 = y%2 ∧ reverseBits n (x/2) = reverseBits n (y/2) := by
      rcases hxmod with hxmod | hxmod <;>
        rcases hymod with hymod | hymod <;>
        simp only [reverseBits, hxmod, hymod, Nat.zero_mul, Nat.one_mul, Nat.zero_add,
          true_and] at h ⊢ <;>
        omega
    have hdiv := ih (x/2) (y/2) hx2 hy2 hparts.2
    omega

def reverseFin (n : Nat) (x : Fin (2^n)) : Fin (2^n) :=
  ⟨reverseBits n x.val, reverseBits_bound n x.val⟩

theorem reverseFin_bijective (n : Nat) : Function.Bijective (reverseFin n) := by
  apply (Fintype.bijective_iff_injective_and_card _).mpr
  constructor
  · intro x y h
    apply Fin.ext
    exact reverseBits_injective_bounded n x.val y.val x.isLt y.isLt
      (congrArg Fin.val h)
  · rfl

theorem reversed_uniform_sum (n : Nat) (f : Fin (2^n) → Nat) :
    (∑ r, f (reverseFin n r)) = ∑ r, f r :=
  (reverseFin_bijective n).sum_comp f

theorem reverseBits_all_ones (n : Nat) : reverseBits n (2^n-1) = 2^n-1 := by
  induction n with
  | zero => simp [reverseBits]
  | succ n ih =>
    have hp := Nat.two_pow_pos n
    have hpow : 2^(n+1) = 2^n * 2 := Nat.pow_succ _ _
    have hm : (2^(n+1)-1)%2 = 1 := by rw [hpow]; omega
    have hd : (2^(n+1)-1)/2 = 2^n-1 := by rw [hpow]; omega
    rw [reverseBits, hm, hd, ih, hpow]
    omega

theorem shared_maximum_event (n r : Nat) :
    (r = 2^n-1 ∧ reverseBits n r = 2^n-1) ↔ r = 2^n-1 := by
  constructor
  · exact And.left
  · intro h
    exact ⟨h, h ▸ reverseBits_all_ones n⟩

theorem shared_maximum_count (n : Nat) :
    (Finset.univ.filter fun r : Fin (2^n) =>
      r.val = 2^n-1 ∧ reverseBits n r.val = 2^n-1).card = 1 := by
  let last : Fin (2^n) := ⟨2^n-1, by have := Nat.two_pow_pos n; omega⟩
  have hs : (Finset.univ.filter fun r : Fin (2^n) =>
      r.val = 2^n-1 ∧ reverseBits n r.val = 2^n-1) = {last} := by
    ext r
    simp [shared_maximum_event, Fin.ext_iff, last]
  rw [hs]
  simp

theorem unit_residual_up_iff (k n q r : Nat) (hn : 0 < n) (hq : 0 < q)
    (hr : r < n) : quantize ((k*n+1)*q) r n q = k+1 ↔ r = n-1 := by
  unfold quantize
  rw [← Nat.add_mul, Nat.mul_comm n q, ← Nat.div_div_eq_div_mul,
    Nat.mul_div_cancel _ hq]
  rw [show k*n+1+r = n*k+(1+r) by ring,
    Nat.mul_add_div hn]
  by_cases heq : 1+r = n
  · rw [heq, Nat.div_self hn]
    omega
  · have hlt : 1+r < n := by omega
    rw [Nat.div_eq_of_lt hlt]
    omega

theorem e4m3_subquantum_erased (r : Nat) :
    lane .e4m3 false true 0x3f80000f r = 56 := by
  norm_num [lane, parameters, isPair, roundMantissa]
  omega

theorem e4m3_unit_quantum_up_iff (r : Nat) :
    lane .e4m3 false true 0x3f800010 r = 57 ↔ r % 65536 = 65535 := by
  norm_num [lane, parameters, isPair, roundMantissa]
  omega

theorem e4m3_coupled_roundup (r : Nat) (hr : r < 65536) :
    (lane .e4m3 false true 0x3f800010 r = 57 ∧
      lane .e4m3 false true 0x3f800010 (reverseBits 16 r) = 57) ↔ r = 65535 := by
  have hb : reverseBits 16 r < 65536 := reverseBits_bound 16 r
  rw [e4m3_unit_quantum_up_iff, e4m3_unit_quantum_up_iff,
    Nat.mod_eq_of_lt hr, Nat.mod_eq_of_lt hb]
  exact shared_maximum_event 16 r

#print axioms uniform_sum
#print axioms uniform_mean_numerator
#print axioms finite_entropy_bias
#print axioms exactly_unbiased_iff
#print axioms model_mean
#print axioms reverseFin_bijective
#print axioms reversed_uniform_sum
#print axioms shared_maximum_event
#print axioms shared_maximum_count
#print axioms unit_residual_up_iff
#print axioms e4m3_subquantum_erased
#print axioms e4m3_unit_quantum_up_iff
#print axioms e4m3_coupled_roundup

end GB300Math.StochasticProbability
