import FloatLib.Floats.Formats.BinaryInterchange.Configured
import MufuRcp32Accuracy
namespace GB300Math.MufuRcp32Bounds
open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

def float32 (word : Nat) : Option Rat :=
  Model.toRat? (Model.ofNatBits word : Model FloatFormat.binary32)

theorem fields (e : Fin 2) (f : Fin (2^23)) :
    let x : Model FloatFormat.binary32 := Model.ofNatBits ((126+e.val)*2^23+f.val)
    x.expField = 126+e.val ∧ x.fracField = f.val ∧ x.signBit = false := by
  have he := e.isLt
  have hf := f.isLt
  have hw : (126+e.val)*2^23+f.val < 2^32 := by omega
  have hn : (126+e.val)*2^23+f.val < 2^31 := by omega
  constructor
  · change (((BitVec.ofNat 32 ((126+e.val)*2^23+f.val) >>> 23) &&& (255 : BitVec 32)).toNat) = _
    simp only [BitVec.toNat_and, BitVec.toNat_ushiftRight, BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hw]
    rw [Nat.shiftRight_eq_div_pow]
    change ((((126+e.val)*2^23+f.val) / 2^23) &&& (2^8-1)) = _
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  · constructor
    · change ((BitVec.ofNat 32 ((126+e.val)*2^23+f.val) &&& (8388607 : BitVec 32)).toNat) = _
      simp only [BitVec.toNat_and, BitVec.toNat_ofNat]
      rw [Nat.mod_eq_of_lt hw]
      change (((126+e.val)*2^23+f.val) &&& (2^23-1)) = _
      rw [Nat.and_two_pow_sub_one_eq_mod]
      omega
    · rw [Model.signBit_eq_msb]
      change (BitVec.ofNat 32 ((126+e.val)*2^23+f.val)).msb = false
      rw [BitVec.msb_eq_decide]
      simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hw]
      simpa using hn
theorem decode_fields (x : Model FloatFormat.binary32) (e : Fin 2) (f : Fin (2^23))
    (he : x.expField = 126+e.val) (hf : x.fracField = f.val) (hs : x.signBit = false) :
    Model.toRat? x = some (((2^23+f.val : Nat) : Rat) / 2^(24-e.val)) := by
  have hieee : FloatFormat.binary32.isIEEE = true := by decide
  simp only [Model.toRat?, Model.toDyadic?, hieee, ite_true,
    Model.ieeeToDyadic?, Model.IEEE.isNaN, Model.IEEE.isInf, he, hf, hs]
  fin_cases e <;> norm_num [FloatFormat.expAllOnesNat, FloatFormat.ieeeNormalMantissaExpOffset,
    Model.pow2, FloatLib.Numerics.Dyadic.toRat, FloatFormat.binary32,
    FloatFormat.bias, FloatLib.Numerics.Dyadic.signedSignificand] <;> ring

theorem float32_binade (e : Fin 2) (f : Fin (2^23)) :
    float32 ((126+e.val)*2^23+f.val) =
      some (((2^23+f.val : Nat) : Rat) / 2^(24-e.val)) := by
  have h := fields e f
  exact decode_fields _ e f h.1 h.2.1 h.2.2
theorem encoded_rcp (f : Fin (2^23)) :
  Mufu.rcp (127*2^23+f.val) =
    let r := (Mufu.interpolate MufuTables.rcp f.val).toNat
    if r = 2^24 then 127*2^23 else 126*2^23+r%2^23 := by
  have hf := f.isLt
  have hm : Mufu.magnitude (127*2^23+f.val) = 127*2^23+f.val := by
    unfold Mufu.magnitude
    apply Nat.mod_eq_of_lt
    omega
  have he : Mufu.exponent (127*2^23+f.val) = 127 := by unfold Mufu.exponent; omega
  have hfrac : Mufu.fraction (127*2^23+f.val) = f.val := by unfold Mufu.fraction; omega
  have hs : Mufu.sign (127*2^23+f.val) = 0 := by
    unfold Mufu.sign
    change (127*2^23+f.val) &&& 2^31 = 0
    rw [Nat.and_two_pow]
    rw [Nat.testBit_lt_two_pow (show 127*2^23+f.val < 2^31 by omega)]
    rfl
  have hn : ¬127*2^23+f.val > 0x7f800000 := by omega
  simp only [Mufu.rcp, hm, he, hs, hfrac, hn, ite_false]
  generalize (Mufu.interpolate MufuTables.rcp f.val).toNat = r
  by_cases h : r = 16777216 <;> norm_num [h]
theorem encoded_output_value (f : Fin (2^23)) :
    float32 (Mufu.rcp (127*2^23+f.val)) =
      some ((Mufu.interpolate MufuTables.rcp f.val : Rat)/2^24) := by
  have hr := mantissa_range f
  let r := (Mufu.interpolate MufuTables.rcp f.val).toNat
  have hn : 0 ≤ Mufu.interpolate MufuTables.rcp f.val := by omega
  have hcast : (r : Int) = Mufu.interpolate MufuTables.rcp f.val := Int.toNat_of_nonneg hn
  have rlo : 2^23 ≤ r := by omega
  have rhi : r ≤ 2^24 := by omega
  rw [encoded_rcp]
  change float32 (if r = 2^24 then 127*2^23 else 126*2^23+r%2^23) = _
  by_cases h : r = 2^24
  · rw [ite_eq_left h]
    have d := float32_binade (1 : Fin 2) (0 : Fin (2^23))
    have hi : Mufu.interpolate MufuTables.rcp f.val = 2^24 := by omega
    norm_num at d
    norm_num [hi]
    exact d
  · rw [ite_eq_right h]
    have d := float32_binade (0 : Fin 2) (⟨r%2^23, Nat.mod_lt _ (by norm_num)⟩ : Fin (2^23))
    have hid : 2^23+r%2^23 = r := by omega
    change float32 (126*2^23+r%2^23) = some (((2^23+r%2^23 : Nat):Rat)/2^24) at d
    rw [hid] at d
    have hc : (r : Rat) = (Mufu.interpolate MufuTables.rcp f.val : Rat) := by exact_mod_cast hcast
    simpa only [hc] using d

theorem normalized_floatlib_error (f : Fin (2^23)) :
    ∃ x y : Rat, float32 (127*2^23+f.val) = some x ∧
      float32 (Mufu.rcp (127*2^23+f.val)) = some y ∧
      |(y : ℝ)-1/(x : ℝ)| / |1/(x : ℝ)| ≤ 1/(2^23 : ℝ) := by
  refine ⟨((2^23+f.val : Nat):Rat)/2^23,
    (Mufu.interpolate MufuTables.rcp f.val : Rat)/2^24, ?_, encoded_output_value f, ?_⟩
  · exact float32_binade (1 : Fin 2) f
  · have h := normalized_relative_error f
    norm_num [normalizedInput, normalizedOutput] at h ⊢
    exact h

theorem error_witness :
    Mufu.rcp 0x3fecfdfd = 0x3f0a4424 ∧
    (2^23+(0x6cfdfd : Int))*Mufu.interpolate MufuTables.rcp 0x6cfdfd-2^47 = -13833324 := by
  decide +kernel

theorem half_unit_bound_is_false :
    ¬ ∀ f : Fin (2^23),
      |normalizedOutput f.val-1/normalizedInput f.val| / |1/normalizedInput f.val| ≤
        1/(2^24 : ℝ) := by
  intro h
  have q := h ⟨0x6cfdfd, by decide⟩
  have r : Mufu.interpolate MufuTables.rcp 0x6cfdfd = 9061412 := by decide +kernel
  norm_num [normalizedInput, normalizedOutput, r] at q

#print axioms fields
#print axioms decode_fields
#print axioms float32_binade
#print axioms encoded_rcp
#print axioms encoded_output_value
#print axioms normalized_floatlib_error
#print axioms error_witness
#print axioms half_unit_bound_is_false
end GB300Math.MufuRcp32Bounds
