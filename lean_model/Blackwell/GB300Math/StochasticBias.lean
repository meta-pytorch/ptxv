import GB300Math.StochasticFinite
import GB300Math.StochasticUnderflow
namespace GB300Math.StochasticBias
open StochasticSm103 StochasticValue StochasticNormal StochasticFinite

theorem normal_pair_unbiased (f : Format) (hf : isPair f = true) (j t : Nat) :
    normalMean f j t = (1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f) := by
  rw [normal_mean]
  have hq : quantum f = 1 := by cases f <;> simp_all [isPair,quantum,fractionBits,entropyBits,parameters]
  rw [hq,Nat.mod_one,Nat.sub_zero]

theorem bf16_subnormal_unbiased (e t : Nat) (he : e ≤ 127-(parameters .bf16).2.2.1) :
    StochasticUnderflow.mean .bf16 e t =
      (StochasticUnderflow.sourceMantissa e t:Rat)*StochasticUnderflow.sourceScale e := by
  have he' : e = 0 := by norm_num [parameters] at he; omega
  subst e
  rw [StochasticUnderflow.exact_mean .bf16 0 t (by decide)]
  have hq : StochasticUnderflow.quantum .bf16 0 = 1 := by decide
  rw [hq,Nat.mod_one,Nat.sub_zero]

theorem normal_mean_toward_zero (f : Format) (j t : Nat) :
    normalMean f j t ≤ (1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f) := by
  rw [normal_mean]
  have ht : ((t-t%quantum f:Nat):Rat) ≤ (t:Rat) := by exact_mod_cast Nat.sub_le t (t%quantum f)
  have hp : 0 ≤ (2:Rat)^((j:Int)-bias f) := by positivity
  apply mul_le_mul_of_nonneg_right _ hp
  linarith

theorem normal_mean_interval (f : Format) (j t : Nat) :
    (1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f) -
        ((2:Rat)^((j:Int)-bias f)/(2:Rat)^(fractionBits f))/(2:Rat)^(entropyBits f) < normalMean f j t ∧
      normalMean f j t ≤ (1+(t:Rat)/8388608)*(2:Rat)^((j:Int)-bias f) := by
  have h := normal_mean_error_bound f j t
  rw [abs_lt] at h
  exact ⟨by linarith,normal_mean_toward_zero f j t⟩

theorem subnormal_mean_toward_zero (f : Format) (e t : Nat)
    (he : e ≤ 127-(parameters f).2.2.1) :
    StochasticUnderflow.mean f e t ≤
      (StochasticUnderflow.sourceMantissa e t:Rat)*StochasticUnderflow.sourceScale e := by
  rw [StochasticUnderflow.exact_mean f e t he]
  have ht : ((StochasticUnderflow.sourceMantissa e t-
      StochasticUnderflow.sourceMantissa e t%StochasticUnderflow.quantum f e:Nat):Rat) ≤
      (StochasticUnderflow.sourceMantissa e t:Rat) := by exact_mod_cast Nat.sub_le _ _
  exact mul_le_mul_of_nonneg_right ht (by unfold StochasticUnderflow.sourceScale; positivity)

#print axioms normal_pair_unbiased
#print axioms bf16_subnormal_unbiased
#print axioms normal_mean_toward_zero
#print axioms normal_mean_interval
#print axioms subnormal_mean_toward_zero
end GB300Math.StochasticBias
