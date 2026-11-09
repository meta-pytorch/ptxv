import FabricBF16RoundBounds
namespace FabricBF16Decode
open FloatLib.Floats.Formats.BinaryInterchange
open GB300Math.TensorAlignmentLoss

theorem dyadic_units (d : FloatLib.Numerics.Dyadic) (he : -133 ≤ d.exponent) :
    d.toReal =
      ((if d.negative then -(d.significand*2^(d.exponent+133).toNat : Int)
        else d.significand*2^(d.exponent+133).toNat : Int) : ℝ) / 2^133 := by
  have hnat : ((d.exponent+133).toNat : Int) = d.exponent+133 := Int.toNat_of_nonneg (by omega)
  have hp : (2:ℝ) ^ (d.exponent+133).toNat = (2:ℝ)^d.exponent * 2^133 := by
    rw [← zpow_natCast, hnat, zpow_add₀ (by norm_num)]
    norm_num
  cases hs : d.negative <;>
    simp [FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand, hs, hp] <;> ring

theorem units_value (word : Nat) (value : Int) (hu : FabricBF16Block.units word = some value) :
    (Model.ofNatBits word : Model FloatFormat.bfloat16).toReal = (value:ℝ)/2^133 := by
  by_cases hw : word < 65536
  · cases hd : (Model.ofNatBits word : Model FloatFormat.bfloat16).toDyadic? with
    | none => simp [FabricBF16Block.units, hw, hd] at hu
    | some d =>
      by_cases he : d.exponent ≥ -133
      · simp [FabricBF16Block.units, hw, hd, he] at hu
        rw [Model.toReal_eq, hd]
        simp only
        rw [← hu]
        exact dyadic_units d he
      · simp [FabricBF16Block.units, hw, hd, he] at hu
  · simp [FabricBF16Block.units, hw] at hu

theorem sum_units_value (words : List Nat)
    (valid : ∀ w ∈ words, ∃ v, FabricBF16Block.units w = some v) :
    (words.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16).toReal).sum =
      (exactSum (words.map fun w => (FabricBF16Block.units w).getD 0) : ℝ)/2^133 := by
  induction words with
  | nil => simp [exactSum]
  | cons w ws ih =>
    obtain ⟨v,hv⟩ := valid w (by simp)
    have ht : ∀ t ∈ ws, ∃ v, FabricBF16Block.units t = some v := by
      intro t ht
      exact valid t (by simp [ht])
    simp only [List.map_cons, List.sum_cons]
    rw [units_value w v hv, ih ht]
    simp [exactSum, hv, add_div]

theorem decoded_model_error (words : List Nat)
    (finite_inputs : ∀ w ∈ words, w % 32768 < 32640)
    (valid : ∀ w ∈ words, ∃ v, FabricBF16Block.units w = some v)
    (finite_output : Model.isFinite (Model.ofNatBits (FabricBF16Block.output #[FabricBF16Block.accumulate words] 0 false) : Model FloatFormat.bfloat16) = true) :
    let values := words.map fun w => (FabricBF16Block.units w).getD 0
    |Model.toReal (Model.ofNatBits (FabricBF16Block.output #[FabricBF16Block.accumulate words] 0 false) : Model FloatFormat.bfloat16) -
        (words.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16).toReal).sum| ≤
      (FabricBF16RoundBounds.loss words values : ℝ)/(2^133 : ℝ) +
      FloatLib.Floats.Formats.Flocq.ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.bfloat16)
        ((FabricBF16RoundBounds.acc words values : ℝ)/(2^133 : ℝ))/2 := by
  rw [sum_units_value words valid]
  exact FabricBF16RoundBounds.finite_model_error words finite_inputs finite_output

theorem exponent_lower_bound (x : Model FloatFormat.bfloat16) (d : FloatLib.Numerics.Dyadic)
    (hd : x.toDyadic? = some d) : -133 ≤ d.exponent := by
  simp only [Model.toDyadic?, show FloatFormat.bfloat16.isIEEE = true from rfl,
    ↓reduceIte] at hd
  unfold Model.ieeeToDyadic? at hd
  split at hd
  · contradiction
  · dsimp only at hd
    split at hd
    · split at hd
      · cases Option.some.inj hd
        change -133 ≤ (0:Int)
        omega
      · cases Option.some.inj hd
        change -133 ≤ (-133:Int)
        omega
    · rename_i he
      cases Option.some.inj hd
      change -133 ≤ (x.expField : Int)-134
      have hpos : x.expField ≠ 0 := by simpa using he
      omega

theorem units_defined (word : Nat) (hw : word < 65536)
    (hf : Model.isFinite (Model.ofNatBits word : Model FloatFormat.bfloat16) = true) :
    ∃ v, FabricBF16Block.units word = some v := by
  obtain ⟨d,hd⟩ := Model.exists_toDyadic?_of_isFinite hf
  have he := exponent_lower_bound _ d hd
  simp [FabricBF16Block.units, hw, hd, he]

theorem exp_word (word : Nat) :
    (Model.ofNatBits word : Model FloatFormat.bfloat16).expField = word%32768/128 := by
  rw [Model.expField_eq_expFieldImpl_apply]
  change ((word % 65536) >>> 7) &&& (2^8-1) = word%32768/128
  rw [Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]
  omega

theorem finite_raw (word : Nat) (hf : word % 32768 < 32640) :
    Model.isFinite (Model.ofNatBits word : Model FloatFormat.bfloat16) = true := by
  change ((Model.ofNatBits word : Model FloatFormat.bfloat16).expField != 255) = true
  rw [exp_word]
  simp only [bne_iff_ne]
  omega

theorem finite_word_error (words : List Nat)
    (valid : ∀ w ∈ words, w < 65536 ∧ w % 32768 < 32640)
    (finite_output : Model.isFinite (Model.ofNatBits (FabricBF16Block.output #[FabricBF16Block.accumulate words] 0 false) : Model FloatFormat.bfloat16) = true) :
    let values := words.map fun w => (FabricBF16Block.units w).getD 0
    |Model.toReal (Model.ofNatBits (FabricBF16Block.output #[FabricBF16Block.accumulate words] 0 false) : Model FloatFormat.bfloat16) -
        (words.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16).toReal).sum| ≤
      (FabricBF16RoundBounds.loss words values : ℝ)/(2^133 : ℝ) +
      FloatLib.Floats.Formats.Flocq.ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.bfloat16)
        ((FabricBF16RoundBounds.acc words values : ℝ)/(2^133 : ℝ))/2 := by
  apply decoded_model_error words (fun w hw => (valid w hw).2) _ finite_output
  intro w hw
  exact units_defined w (valid w hw).1 (finite_raw w (valid w hw).2)

theorem infinity_rejected : FabricBF16Block.units 32640 = none := by decide +kernel
theorem overflow_word_rejected : FabricBF16Block.units 65536 = none := by decide +kernel

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
theorem word_frac (word : Nat) :
    (Model.ofNatBits word : Model FloatFormat.bfloat16).fracField = word%128 := by
  rw [Model.fracField_eq_fracFieldImpl_apply]
  change (word % 65536) &&& (2^7-1) = word%128
  rw [Nat.and_two_pow_sub_one_eq_mod]
  omega

theorem word_sign (word : Nat) (hw : word < 65536) :
    (Model.ofNatBits word : Model FloatFormat.bfloat16).signBit = decide (32768 ≤ word) := by
  rw [Model.signBit_eq_signBitImpl_apply]
  change (word%65536).testBit 15 = decide (32768≤word)
  rw [Nat.mod_eq_of_lt hw, Nat.testBit_eq_decide_div_mod_eq]
  congr 1
  apply propext
  change word/32768%2=1 ↔32768≤word
  omega

theorem units_canonical (word : Nat) (hw : word<65536) (hm : word%32768<32640) :
    FabricBF16Block.units word=some (if 32768≤word then -(FabricBF16Block.unit (word%32768) : Int) else (FabricBF16Block.unit (word%32768) : Int)) := by
  have hs := word_sign word hw
  have hf := FabricBF16Decode.exp_word word
  have hfrac : word%128=(word%32768)%128 := by omega
  have he : word%32768/128 ≠ 255 := by omega
  simp only [FabricBF16Block.units,hw,↓reduceIte]
  simp only [Model.toDyadic?,show FloatFormat.bfloat16.isIEEE=true from rfl,↓reduceIte,
    Model.ieeeToDyadic?,Model.IEEE.isNaN,Model.IEEE.isInf]
  simp only [hf,word_frac,hfrac,hs,show FloatFormat.bfloat16.expAllOnesNat=255 from rfl,
    show FloatFormat.bfloat16.ieeeMinSubnormalExponent=(-133:Int) from rfl,
    show FloatFormat.bfloat16.ieeeNormalMantissaExpOffset=134 from rfl,
    show Model.pow2 FloatFormat.bfloat16.fracWidth=128 from rfl,
    beq_eq_false_iff_ne.mpr he,Bool.false_and,Bool.false_or,Bool.false_eq_true,↓reduceIte]
  by_cases hz : word%32768/128=0
  · by_cases hfz : word%32768%128=0
    · have hwz : word%32768=0 := by omega
      simp [FabricBF16Block.unit,hwz]
    · have hfz1 : word%128 ≠ 0 := by simpa using hfz
      simp [hz,hfz1,FabricBF16Block.unit]
  · have hepos : 1≤word%32768/128 := by omega
    have hex : ((word%32768/128:Int)-134)+133=(word%32768/128:Int)-1 := by omega
    have heok : (word%32768/128:Int)-134 ≥ -133 := by omega
    have hnat : ((word%32768/128:Int)-1).toNat=word%32768/128-1 := by omega
    simp [hz,heok,hex,hnat,FabricBF16Block.unit,add_comm]

#print axioms word_frac
#print axioms word_sign
#print axioms units_canonical
end FabricBF16Decode
