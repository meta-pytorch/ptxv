import SqrtBounds

set_option maxRecDepth 1000000
set_option maxHeartbeats 10000000

namespace GB300Math.FixedToWord

theorem decode (lead mant : Nat) (hl : lead ≤ 30)
    (hm : 2^23 ≤ mant ∧ mant < 2^24) :
    RcpBounds.positiveNormalValue ((lead+91)*2^23+mant%2^23) =
      (mant:ℚ)*(2:ℚ)^((lead:Int)-59) := by
  rw [RcpBounds.normal_decode (lead+91) (mant%2^23) (by omega) (by omega)]
  have hmod : mant%2^23 = mant-2^23 := by omega
  rw [hmod,Nat.cast_sub hm.1]
  push_cast
  norm_num
  ring_nf
  simp

theorem fixed_error (mag : Nat) (hm : 4096 ≤ mag ∧ mag < 2^31) :
    0 ≤ (mag:ℚ)/2^36-RcpBounds.positiveNormalValue (Mufu.fixedToWord mag 36) ∧
    (mag:ℚ)/2^36-RcpBounds.positiveNormalValue (Mufu.fixedToWord mag 36) ≤ 1/2^29 := by
  have hn : mag ≠ 0 := by omega
  have ha := Nat.log2_self_le hn
  have hb := Nat.lt_log2_self (n := mag)
  have hlow : 12 ≤ mag.log2 := (Nat.le_log2 hn).mpr (by omega)
  have hhigh : mag.log2 ≤ 30 := by have h := (Nat.log2_lt hn).mpr hm.2;omega
  generalize hlead : mag.log2 = lead at *
  have hexp : ((lead:Int)-36+127).toNat = lead+91 := by omega
  have hmnt : 2^23 ≤ (if lead ≥ 23 then mag/2^(lead-23) else mag*2^(23-lead)) ∧
      (if lead ≥ 23 then mag/2^(lead-23) else mag*2^(23-lead)) < 2^24 := by
    interval_cases lead <;> norm_num at * <;> omega
  unfold Mufu.fixedToWord
  simp only [hlead]
  rw [hexp,decode lead _ hhigh hmnt]
  have hdiv0 := Nat.div_mul_le_self mag (2^(lead-23))
  have hdiv1 : mag < mag/2^(lead-23)*2^(lead-23)+2^(lead-23) := by
    have hpos : 0 < 2^(lead-23) := by positivity
    have he := Nat.mod_add_div mag (2^(lead-23))
    have hr := Nat.mod_lt mag hpos
    nlinarith only [he,hr]
  have hd0 : ((mag/2^(lead-23) : Nat):ℚ)*(2^(lead-23):Nat) ≤ mag := by exact_mod_cast hdiv0
  have hd1 : (mag:ℚ) < ((mag/2^(lead-23):Nat):ℚ)*(2^(lead-23):Nat)+(2^(lead-23):Nat) := by exact_mod_cast hdiv1
  interval_cases lead <;> norm_num at * <;> constructor <;> linarith


theorem positive_normal (mag : Nat) (hm : 4096 ≤ mag ∧ mag < 2^31) :
    let word := Mufu.fixedToWord mag 36
    0 < word ∧ word < 2^31 ∧ 1 ≤ Mufu.exponent word ∧ Mufu.exponent word ≤ 254 := by
  have hn : mag ≠ 0 := by omega
  have hlo : 12 ≤ mag.log2 := (Nat.le_log2 hn).mpr (by omega)
  have hhi : mag.log2 ≤ 30 := by have h := (Nat.log2_lt hn).mpr hm.2;omega
  have he : ((mag.log2:Int)-36+127).toNat = mag.log2+91 := by omega
  dsimp [Mufu.fixedToWord,Mufu.exponent]
  rw [he]
  omega

#print axioms positive_normal
#print axioms fixed_error
end GB300Math.FixedToWord
