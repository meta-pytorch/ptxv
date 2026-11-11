import FabricF32RoundBounds
namespace FabricF32Decode
open FloatLib.Floats.Formats.BinaryInterchange
open GB300Math.TensorAlignmentLoss

theorem dyadic_units (d : FloatLib.Numerics.Dyadic) (he : -149 ≤ d.exponent) :
    d.toReal =
      ((if d.negative then -(d.significand*2^(d.exponent+149).toNat : Int)
        else d.significand*2^(d.exponent+149).toNat : Int) : ℝ) / 2^149 := by
  have hnat : ((d.exponent+149).toNat : Int) = d.exponent+149 := Int.toNat_of_nonneg (by omega)
  have hp : (2:ℝ) ^ (d.exponent+149).toNat = (2:ℝ)^d.exponent * 2^149 := by
    rw [← zpow_natCast, hnat, zpow_add₀ (by norm_num)]
    norm_num
  cases hs : d.negative <;>
    simp [FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand, hs, hp] <;> ring

theorem units_value (word : Nat) (value : Int) (hu : FabricF32.units word = some value) :
    (Model.ofNatBits word : Model (.ieee 8 23)).toReal = (value:ℝ)/2^149 := by
  by_cases hw : word < 4294967296
  · cases hd : (Model.ofNatBits word : Model (.ieee 8 23)).toDyadic? with
    | none => simp [FabricF32.units, hw, hd] at hu
    | some d =>
      by_cases he : d.exponent ≥ -149
      · simp [FabricF32.units, hw, hd, he] at hu
        rw [Model.toReal_eq, hd]
        simp only
        rw [← hu]
        exact dyadic_units d he
      · simp [FabricF32.units, hw, hd, he] at hu
  · simp [FabricF32.units, hw] at hu

theorem sum_units_value (words : List Nat)
    (valid : ∀ w ∈ words, ∃ v, FabricF32.units w = some v) :
    (words.map fun w => (Model.ofNatBits w : Model (.ieee 8 23)).toReal).sum =
      (exactSum (words.map fun w => (FabricF32.units w).getD 0) : ℝ)/2^149 := by
  induction words with
  | nil => simp [exactSum]
  | cons w ws ih =>
    obtain ⟨v,hv⟩ := valid w (by simp)
    have ht : ∀ t ∈ ws, ∃ v, FabricF32.units t = some v := by
      intro t ht
      exact valid t (by simp [ht])
    simp only [List.map_cons, List.sum_cons]
    rw [units_value w v hv, ih ht]
    simp [exactSum, hv, add_div]

theorem decoded_model_error (words : List Nat)
    (finite_inputs : ∀ w ∈ words, w % 2^31 < 0x7f800000)
    (valid : ∀ w ∈ words, ∃ v, FabricF32.units w = some v)
    (finite_output : Model.isFinite (Model.ofNatBits (FabricF32.predict words) : Model (.ieee 8 23)) = true) :
    let values := words.map fun w => (FabricF32.units w).getD 0
    |Model.toReal (Model.ofNatBits (FabricF32.predict words) : Model (.ieee 8 23)) -
        (words.map fun w => (Model.ofNatBits w : Model (.ieee 8 23)).toReal).sum| ≤
      (FabricF32RoundBounds.loss words values : ℝ)/(2^149 : ℝ) +
      FloatLib.Floats.Formats.Flocq.ulp FloatLib.Numerics.binaryRadix (Model.fexpOf (.ieee 8 23))
        ((FabricF32RoundBounds.acc words values : ℝ)/(2^149 : ℝ))/2 := by
  rw [sum_units_value words valid]
  exact FabricF32RoundBounds.finite_model_error words finite_inputs finite_output

theorem exponent_lower_bound (x : Model (.ieee 8 23)) (d : FloatLib.Numerics.Dyadic)
    (hd : x.toDyadic? = some d) : -149 ≤ d.exponent := by
  simp only [Model.toDyadic?, show (FloatFormat.ieee 8 23).isIEEE = true from rfl,
    ↓reduceIte] at hd
  unfold Model.ieeeToDyadic? at hd
  split at hd
  · contradiction
  · dsimp only at hd
    split at hd
    · split at hd
      · cases Option.some.inj hd
        change -149 ≤ (0:Int)
        omega
      · cases Option.some.inj hd
        change -149 ≤ (-149:Int)
        omega
    · rename_i he
      cases Option.some.inj hd
      change -149 ≤ (x.expField : Int)-150
      have hpos : x.expField ≠ 0 := by simpa using he
      omega

theorem units_defined (word : Nat) (hw : word < 4294967296)
    (hf : Model.isFinite (Model.ofNatBits word : Model (.ieee 8 23)) = true) :
    ∃ v, FabricF32.units word = some v := by
  obtain ⟨d,hd⟩ := Model.exists_toDyadic?_of_isFinite hf
  have he := exponent_lower_bound _ d hd
  simp [FabricF32.units, hw, hd, he]

theorem exp_word (word : Nat) :
    (Model.ofNatBits word : Model (.ieee 8 23)).expField = word%2147483648/8388608 := by
  rw [Model.expField_eq_expFieldImpl_apply]
  change ((word % 4294967296) >>> 23) &&& (2^8-1) = word%2147483648/8388608
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]
  omega

theorem finite_raw (word : Nat) (hf : word % 2147483648 < 2139095040) :
    Model.isFinite (Model.ofNatBits word : Model (.ieee 8 23)) = true := by
  change ((Model.ofNatBits word : Model (.ieee 8 23)).expField != 255) = true
  rw [exp_word]
  simp only [bne_iff_ne]
  omega

theorem finite_word_error (words : List Nat)
    (valid : ∀ w ∈ words, w < 4294967296 ∧ w % 2147483648 < 2139095040)
    (finite_output : Model.isFinite (Model.ofNatBits (FabricF32.predict words) : Model (.ieee 8 23)) = true) :
    let values := words.map fun w => (FabricF32.units w).getD 0
    |Model.toReal (Model.ofNatBits (FabricF32.predict words) : Model (.ieee 8 23)) -
        (words.map fun w => (Model.ofNatBits w : Model (.ieee 8 23)).toReal).sum| ≤
      (FabricF32RoundBounds.loss words values : ℝ)/(2^149 : ℝ) +
      FloatLib.Floats.Formats.Flocq.ulp FloatLib.Numerics.binaryRadix (Model.fexpOf (.ieee 8 23))
        ((FabricF32RoundBounds.acc words values : ℝ)/(2^149 : ℝ))/2 := by
  apply decoded_model_error words (fun w hw => (valid w hw).2) _ finite_output
  intro w hw
  exact units_defined w (valid w hw).1 (finite_raw w (valid w hw).2)

theorem infinity_rejected : FabricF32.units 2139095040 = none := by decide +kernel
theorem overflow_word_rejected : FabricF32.units 4294967296 = none := by decide +kernel

#print axioms finite_word_error
#print axioms infinity_rejected
#print axioms overflow_word_rejected
#print axioms exp_word
#print axioms finite_raw
#print axioms exponent_lower_bound
#print axioms units_defined
#print axioms sum_units_value
#print axioms decoded_model_error
#print axioms dyadic_units
#print axioms units_value
end FabricF32Decode
