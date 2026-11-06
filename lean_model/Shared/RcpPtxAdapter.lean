import Approx
import RcpPtxValues
set_option maxRecDepth 10000
namespace GB300Math.RcpPtxBounds
open RcpBounds

theorem rcp_ftz_adapter (word : Nat) (hw : word < 2^32) :
    Approx.evaluate "rcp" true word = some (Mufu.rcp word) := by
  have hs : "rcp".endsWith ".f16" = false := by decide +kernel
  have hw' : word < 4294967296 := by omega
  simp [Approx.evaluate, Mufu.evaluate, Mufu.evaluate32, hs, Nat.mod_eq_of_lt hw']

theorem rcp_ftz_one_ulp (negative : Bool) (exponent fraction output : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23)
    (hout : Approx.evaluate "rcp" true
      ((if negative then 2^31 else 0)+exponent*2^23+fraction) = some output) :
    let word := (if negative then 2^31 else 0)+exponent*2^23+fraction
    |normalValue output-1/normalValue word| ≤ reciprocalUlp exponent fraction := by
  have hw : (if negative then 2^31 else 0)+exponent*2^23+fraction < 2^32 := by
    cases negative <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> omega
  rw [rcp_ftz_adapter _ hw] at hout
  cases Option.some.inj hout
  exact rcp_signed_one_ulp negative exponent fraction he hf

#print axioms rcp_ftz_adapter
#print axioms rcp_ftz_one_ulp
end GB300Math.RcpPtxBounds
