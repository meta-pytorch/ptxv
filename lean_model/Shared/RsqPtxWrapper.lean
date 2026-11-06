import Approx
import RsqPtxBounds

namespace GB300Math.RsqPtxBounds

theorem normal_adapter (ftz : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    Approx.evaluate "rsq" ftz (exponent*2^23+fraction) =
      some (Mufu.rsq (exponent*2^23+fraction)) := by
  have ht : ¬Mufu.magnitude (exponent*2^23+fraction) < 0x00800000 := by
    dsimp [Mufu.magnitude]
    omega
  norm_num at ht
  cases ftz
  · simp [Approx.evaluate,ht]
  · change some (Mufu.rsq ((exponent*2^23+fraction)%2^32)) = _
    rw [Nat.mod_eq_of_lt (show exponent*2^23+fraction < 2^32 by omega)]

theorem adapter_ptx_relative_error (ftz : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    ∃ output, Approx.evaluate "rsq" ftz (exponent*2^23+fraction) = some output ∧
      |SqrtBounds.positiveNormalValue output-
        1/Real.sqrt (SqrtBounds.positiveNormalValue (exponent*2^23+fraction))| ≤
        ((2 : ℝ)^(-(229 : ℝ)/10))/Real.sqrt (SqrtBounds.positiveNormalValue (exponent*2^23+fraction)) := by
  exact ⟨_,normal_adapter ftz exponent fraction he hf,rsq_bits_ptx_relative_error exponent fraction he hf⟩

#print axioms normal_adapter
#print axioms adapter_ptx_relative_error
end GB300Math.RsqPtxBounds
