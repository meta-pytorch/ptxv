import GB300Math.StochasticPacked
namespace GB300Math.StochasticCorrelation
open StochasticSm103 StochasticProbability StochasticValue StochasticPacked

theorem small_quantum_up_iff (f : Format) (hf : isPair f = false) (relu : Bool) (r : Nat) :
    lane f relu true (0x3f800000+quantum f) r = unitBase f+1 ↔ r%65536 = 65535 := by
  cases f <;> simp only [isPair, Bool.true_eq_false] at hf
  all_goals cases relu <;> norm_num [isPair, lane, parameters, quantum, fractionBits, entropyBits, unitBase, roundMantissa]
  all_goals omega

theorem small_subquantum (f : Format) (hf : isPair f = false) (relu : Bool) (t r : Nat)
    (ht : t < quantum f) : lane f relu true (0x3f800000+t) r = unitBase f := by
  have ht' : t < 8388608 := by
    cases f <;> norm_num [quantum, fractionBits, entropyBits, parameters, isPair] at ht <;> omega
  rw [unit_bitvalue f relu true t r ht']
  cases f <;> simp only [isPair, Bool.true_eq_false] at hf
  all_goals norm_num [quantum, fractionBits, entropyBits, parameters, isPair, roundMantissa] at ht ⊢
  all_goals omega

def groupBits (f : Format) (r group : Nat) : Nat :=
  if f == .e2m1 then (r%256+(r/256)*65536)*2^(8*group) else r*2^(16*group)

theorem even_entropy (f : Format) (hf : isPair f = false) (r group : Nat)
    (hr : r < 65536) (hg : group < 2) :
    entropyLane f (groupBits f r group) (2*group) = r := by
  cases f <;> norm_num [isPair] at hf <;> interval_cases group <;>
    norm_num [entropyLane, groupBits, isPair] <;> omega

theorem odd_entropy (f : Format) (hf : isPair f = false) (r group : Nat)
    (hr : r < 65536) (hg : group < 2) :
    entropyLane f (groupBits f r group) (2*group+1) = reverseBits 16 r := by
  cases f <;> norm_num [isPair] at hf <;> interval_cases group <;>
    norm_num [entropyLane, groupBits, isPair] <;>
    congr 1 <;> omega

theorem paired_up_iff (f : Format) (hf : isPair f = false) (relu : Bool) (r group : Nat)
    (hr : r < 65536) (hg : group < 2) :
    (lane f relu true (0x3f800000+quantum f) (entropyLane f (groupBits f r group) (2*group)) = unitBase f+1 ∧
      lane f relu true (0x3f800000+quantum f) (entropyLane f (groupBits f r group) (2*group+1)) = unitBase f+1) ↔ r=65535 := by
  rw [even_entropy f hf r group hr hg, odd_entropy f hf r group hr hg,
    small_quantum_up_iff f hf relu r, small_quantum_up_iff f hf relu (reverseBits 16 r),
    Nat.mod_eq_of_lt hr, Nat.mod_eq_of_lt (reverseBits_bound 16 r)]
  exact shared_maximum_event 16 r

#print axioms small_quantum_up_iff
#print axioms small_subquantum
#print axioms even_entropy
#print axioms odd_entropy
#print axioms paired_up_iff


def packedCode (f : Format) (relu : Bool) (r group laneIndex : Nat) : Nat :=
  let x := 0x3f800000+quantum f
  (packed f relu true x x x x (groupBits f r group) / 2^(laneWidth f*laneIndex)) % 2^(laneWidth f)

theorem packed_pair_up_iff (f : Format) (hf : isPair f = false) (relu : Bool) (r group : Nat)
    (hr : r < 65536) (hg : group < 2) :
    (packedCode f relu r group (2*group) = unitBase f+1 ∧
      packedCode f relu r group (2*group+1) = unitBase f+1) ↔ r=65535 := by
  have hc : laneCount f = 4 := by simp [laneCount,hf]
  unfold packedCode
  rw [packed_lane f relu true _ _ _ _ _ _ (by omega),
    packed_lane f relu true _ _ _ _ _ _ (by omega)]
  simp only [inputLane, ite_self]
  exact paired_up_iff f hf relu r group hr hg

theorem packed_pair_up_count (f : Format) (hf : isPair f = false) (relu : Bool) (group : Nat)
    (hg : group < 2) :
    (Finset.univ.filter fun r : Fin 65536 =>
      packedCode f relu r.val group (2*group) = unitBase f+1 ∧
      packedCode f relu r.val group (2*group+1) = unitBase f+1).card = 1 := by
  let last : Fin 65536 := ⟨65535,by decide⟩
  have hs : (Finset.univ.filter fun r : Fin 65536 =>
      packedCode f relu r.val group (2*group) = unitBase f+1 ∧
      packedCode f relu r.val group (2*group+1) = unitBase f+1) = {last} := by
    ext r
    simp [packed_pair_up_iff f hf relu r.val group r.isLt hg, Fin.ext_iff, last]
  rw [hs]
  simp

theorem reverse_uniform_rational_sum (n : Nat) (f : Fin (2^n) → Rat) :
    (∑ r, f (reverseFin n r)) = ∑ r, f r :=
  (reverseFin_bijective n).sum_comp f

#print axioms packed_pair_up_iff
#print axioms packed_pair_up_count
#print axioms reverse_uniform_rational_sum
end GB300Math.StochasticCorrelation
