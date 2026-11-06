import RcpPtxScale
import RcpBounds
set_option maxRecDepth 10000
namespace GB300Math.RcpPtxBounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem mul_power_two_overflow (negative : Bool) (e f p : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23)
    (hp : 1 ≤ p ∧ p ≤ 254) (hs : 382 ≤ e+p) :
    Model.mul (Model.ofFields FloatFormat.binary32 negative e f)
      (Model.ofFields FloatFormat.binary32 false p 0) =
      if negative then Model.negInf FloatFormat.binary32 else Model.posInf FloatFormat.binary32 := by
  rw [Model.Proof.mul_eq_spec]
  unfold Model.Spec.mul
  rw [decode_normal negative e f he hf, decode_normal false p 0 hp (by norm_num)]
  simp only [Bool.xor_false, Nat.add_zero] at *
  have hn : 2^23+f ≠ 0 := by omega
  simp
  rw [Model.roundDyadic_eq_roundDyadicImpl]
  have hlog : ((8388608+f)*8388608).log2 = 46 := by
    apply (Nat.log2_eq_iff (by omega)).mpr
    constructor <;> omega
  have hround := FloatLib.Numerics.roundShiftRightEven_mul_two_pow_of_le (2^23+f) 23 23 (by omega)
  norm_num at hround
  have hexp : (46 : Int)+((e : Int)-150+((p : Int)-150)) = (e : Int)+p-254 := by omega
  have hfmt : FloatFormat.binary32.isIEEE = true := by decide
  have hmin : FloatFormat.ieeeMinNormalExponent FloatFormat.binary32 = -126 := by rfl
  have hmax : FloatFormat.ieeeMaxNormalExponent FloatFormat.binary32 = 127 := by rfl
  have hwidth : FloatFormat.binary32.fracWidth = 23 := by rfl
  have hbias : FloatFormat.binary32.bias = 127 := by rfl
  have hlow : ¬(e : Int)+p-254 < -126 := by omega
  have hhigh : 127 < (e : Int)+p-254 := by omega
  have hcarry : 8388608+f ≠ 16777216 := by omega
  norm_num [Model.roundDyadicImpl, Model.ieeeRoundDyadicImpl,
    hfmt, hmin, hmax, hwidth, hbias, hlog, hround, hexp, hlow, hhigh, hcarry, Model.pow2]

theorem mul_power_two_overflow_word_left (negative : Bool) (e f p : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23)
    (hp : 1 ≤ p ∧ p ≤ 254) (hs : 382 ≤ e+p) :
    Approx.mul (p*2^23) ((if negative then 2^31 else 0)+e*2^23+f) =
      (if negative then 2^31 else 0)+0x7f800000 := by
  have hleft : Approx.asFloat (p*2^23) = Model.ofFields FloatFormat.binary32 false p 0 := by
    simpa using (fields_float false p 0 (by omega) (by norm_num)).symm
  change Approx.asWord (Model.mul _ _) = _
  rw [← fields_float negative e f (by omega) hf,hleft]
  rw [mul_normal_comm false negative p 0 e f hp (by norm_num) he hf]
  rw [mul_power_two_overflow negative e f p he hf hp hs]
  cases negative <;> decide +kernel

theorem rcp_small_positive_range (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 22) (hf : f < 2^23) :
    ∃ r m, 231 ≤ r ∧ r ≤ 253 ∧ m < 2^23 ∧
      Mufu.rcp (e*2^23+f) = r*2^23+m := by
  rw [RcpBounds.output_word_scaled e f (by omega) hf]
  split
  · refine ⟨254-e,0,by omega,by omega,by norm_num,?_⟩
    exact (Nat.add_zero _).symm
  · refine ⟨253-e,(Mufu.interpolate MufuTables.rcp f).toNat%2^23,
      by omega,by omega,Nat.mod_lt _ (by norm_num),rfl⟩

theorem rcp_small_normal_range (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 22) (hf : f < 2^23) :
    ∃ r m, 231 ≤ r ∧ r ≤ 253 ∧ m < 2^23 ∧
      Mufu.rcp ((if negative then 2^31 else 0)+e*2^23+f) =
        (if negative then 2^31 else 0)+r*2^23+m := by
  obtain ⟨r,m,hr0,hr1,hm,ho⟩ := rcp_small_positive_range e f he hf
  refine ⟨r,m,hr0,hr1,hm,?_⟩
  cases negative
  · simp only [Bool.false_eq_true, ↓reduceIte, Nat.zero_add]
    exact ho
  · simp only [↓reduceIte]
    exact (congrArg Mufu.rcp (Nat.add_assoc (2^31) (e*2^23) f)).trans
      ((RcpBounds.reciprocal_sign (e*2^23+f) (by omega)).trans
        ((congrArg (fun z => 2^31+z) ho).trans (Nat.add_assoc (2^31) (r*2^23) m).symm))

theorem reciprocal_subnormal_overflow (negative : Bool) (f : Nat)
    (hf0 : 0 < f) (hf : f ≤ 2^21) :
    Approx.evaluate "rcp" false ((if negative then 2^31 else 0)+f) =
      some ((if negative then 2^31 else 0)+0x7f800000) := by
  have hout : ∃ r m, 231 ≤ r ∧ r ≤ 253 ∧ m < 2^23 ∧
      Mufu.rcp ((if negative then 2^31 else 0)+(f.log2+2)*2^23+(f*2^(23-f.log2)-2^23)) =
        (if negative then 2^31 else 0)+r*2^23+m := by
    by_cases heq : f = 2^21
    · subst f
      refine ⟨231, 0, by norm_num, by norm_num, by norm_num, ?_⟩
      cases negative <;> decide +kernel
    have hlead : f.log2 ≤ 20 := by
      have h := (Nat.log2_lt (by omega : f ≠ 0)).mpr (by omega : f < 2^21)
      omega
    have hm := normalized_mantissa_range f (by omega) (by omega)
    exact rcp_small_normal_range negative
      (f.log2+2) (f*2^(23-f.log2)-2^23) (by omega) (by omega)
  obtain ⟨r, m, hr0, hr1, hm0, hout⟩ := hout
  have hmag : Mufu.magnitude ((if negative then 2^31 else 0)+f) = f := by
    cases negative <;> simp only [Mufu.magnitude, Bool.false_eq_true, ite_false, ite_true] <;> omega
  have heval : Approx.evaluate "rcp" false ((if negative then 2^31 else 0)+f) =
      some (Approx.mul 0x4b800000
        (Mufu.rcp (Approx.mul ((if negative then 2^31 else 0)+f) 0x4b800000))) := by
    norm_num only at hmag
    simp [Approx.evaluate, hmag, show ¬f > 0x7e800000 by omega, show f < 0x00800000 by omega]
  rw [heval,mul_subnormal_normalize_word negative f (by omega) (by omega),hout]
  have h := mul_power_two_overflow_word_left negative r m 151
    (by omega) hm0 (by norm_num) (by omega)
  simpa using congrArg some h

theorem subnormal_overflow_ideal (f : Nat) (hf0 : 0 < f) (hf : f ≤ 2^21) :
    (2 : ℚ)^128 ≤ 1 / ((f : ℚ) / 2^149) := by
  have hpos : 0 < (f : ℚ) / 2^149 := by positivity
  apply (le_div_iff₀ hpos).mpr
  have hbound : (f : ℚ) ≤ 2^21 := by exact_mod_cast hf
  norm_num
  linarith

#print axioms subnormal_overflow_ideal
#print axioms mul_power_two_overflow_word_left
#print axioms reciprocal_subnormal_overflow
end GB300Math.RcpPtxBounds
