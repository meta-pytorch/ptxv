import FabricF64Decode
import FloatLib.Floats.Formats.Flocq.Theory.Error.Bounds

set_option exponentiation.threshold 2048
namespace FabricF64Bounds
open FabricF64 FloatLib.Floats.Formats.BinaryInterchange FloatLib.Floats.Formats.Flocq

def asDyadic (value : Int) : FloatLib.Numerics.Dyadic :=
  {negative := value < 0, significand := value.natAbs, exponent := -1074}

theorem dyadic_value (value : Int) : (asDyadic value).toReal = (value : ℝ)/2^1074 := by
  cases value <;>
    simp [asDyadic, FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand,
      zpow_neg, div_eq_mul_inv]

theorem round_word (value : Int) :
    (Model.ofNatBits (FabricF64.round value) : Model FloatFormat.binary64) =
      Model.roundDyadic FloatFormat.binary64 (asDyadic value) := by
  unfold FabricF64.round
  exact Model.ofNatBits_toNatBits _

theorem round_error (value : Int)
    (finite : (Model.ofNatBits (FabricF64.round value) : Model FloatFormat.binary64).isFinite = true) :
    |(Model.ofNatBits (FabricF64.round value) : Model FloatFormat.binary64).toReal -
        (value : ℝ)/2^1074| ≤
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.binary64)
        ((value : ℝ)/2^1074)/2 := by
  rw [round_word] at finite ⊢
  rw [Model.toReal_roundDyadic_eq_roundAt _ (by decide) _ finite, dyadic_value]
  exact error_bound_ulp (rnd := nearestEven) _

theorem truncate_loss (shift : Nat) (value : Int) :
    (FabricF64.truncate shift value-value).natAbs ≤ 2^shift-1 := by
  have hp := Nat.two_pow_pos shift
  cases value with
  | ofNat n =>
    have hd := Nat.mod_lt n hp
    have he : n%2^shift+n/2^shift*2^shift=n := by
      simpa [Nat.mul_comm] using Nat.mod_add_div n (2^shift)
    change (FabricF64.truncate shift (n:Int)-(n:Int)).natAbs ≤ 2^shift-1
    have hs : ¬ (n:Int)<0 := by omega
    simp only [FabricF64.truncate,Int.natAbs_natCast,hs,↓reduceIte]
    have hc : ((n%2^shift:Nat):Int)+((n/2^shift*2^shift:Nat):Int)=(n:Int) := by exact_mod_cast he
    have hr : ((n/2^shift*2^shift:Nat):Int)-(n:Int) = -((n%2^shift:Nat):Int) := by omega
    rw [hr,Int.natAbs_neg,Int.natAbs_natCast]
    omega
  | negSucc n =>
    have hd := Nat.mod_lt (n+1) hp
    have he : (n+1)%2^shift+(n+1)/2^shift*2^shift=n+1 := by
      simpa [Nat.mul_comm] using Nat.mod_add_div (n+1) (2^shift)
    simp only [FabricF64.truncate,Int.natAbs_negSucc,Int.negSucc_lt_zero,↓reduceIte]
    have hc : (((n+1)%2^shift:Nat):Int)+(((n+1)/2^shift*2^shift:Nat):Int)=((n+1:Nat):Int) := by exact_mod_cast he
    have hr : -(((n+1)/2^shift*2^shift:Nat):Int)-Int.negSucc n = (((n+1)%2^shift:Nat):Int) := by omega
    rw [hr,Int.natAbs_natCast]
    omega

theorem sum_loss (shift : Nat) (values : List Int) :
    ((values.map (FabricF64.truncate shift)).sum-values.sum).natAbs ≤ values.length*(2^shift-1) := by
  induction values with
  | nil => simp
  | cons v vs ih =>
    have ht := truncate_loss shift v
    have h := Int.natAbs_add_le (FabricF64.truncate shift v-v) ((vs.map (FabricF64.truncate shift)).sum-vs.sum)
    have he : (FabricF64.truncate shift v+(vs.map (FabricF64.truncate shift)).sum)-(v+vs.sum)=
        (FabricF64.truncate shift v-v)+((vs.map (FabricF64.truncate shift)).sum-vs.sum) := by omega
    simp only [List.map_cons,List.sum_cons,List.length_cons]
    rw [he]
    calc
      _ ≤ _ := h
      _ ≤ (2^shift-1)+vs.length*(2^shift-1) := Nat.add_le_add ht ih
      _ = _ := by ring

theorem flush_loss (shift : Nat) (value : Int) :
    (flush shift value-value).natAbs ≤ 2^shift-1 := by
  unfold flush
  split
  · simp only [Int.zero_sub,Int.natAbs_neg];omega
  · simp

def loss (words : List Nat) : Nat :=
  words.length*(2^fineShift words-1)+(2^coarseShift words-1)

def accumulator (words : List Nat) : Int := flush (coarseShift words) (finiteSum words)

theorem accumulator_loss (words : List Nat) :
    (accumulator words-(words.map fun w => (units w).getD 0).sum).natAbs ≤ loss words := by
  have hs := sum_loss (fineShift words) (words.map fun w => (units w).getD 0)
  have hf := flush_loss (coarseShift words) (finiteSum words)
  have hb : (finiteSum words-(words.map fun w => (units w).getD 0).sum).natAbs ≤
      words.length*(2^fineShift words-1) := by
    simpa [finiteSum,List.map_map,Function.comp_def] using hs
  have he : accumulator words-(words.map fun w => (units w).getD 0).sum =
      (flush (coarseShift words) (finiteSum words)-finiteSum words)+
      (finiteSum words-(words.map fun w => (units w).getD 0).sum) := by dsimp [accumulator];omega
  rw [he]
  exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add hf hb) |>.trans_eq (Nat.add_comm _ _)

theorem accumulator_real_error (words : List Nat) :
    |(accumulator words:ℝ)/2^1074-
      (((words.map fun w => (units w).getD 0).sum:Int):ℝ)/2^1074| ≤ (loss words:ℝ)/2^1074 := by
  have h := accumulator_loss words
  have hr : |((accumulator words-(words.map fun w => (units w).getD 0).sum:Int):ℝ)| ≤ (loss words:ℝ) := by
    have hn : (((accumulator words-(words.map fun w => (units w).getD 0).sum).natAbs:Nat):ℝ)≤(loss words:ℝ) := by
      exact_mod_cast h
    simpa using hn
  rw [Int.cast_sub] at hr
  rw [←sub_div,abs_div,abs_of_pos (by positivity : (0:ℝ)<2^1074)]
  exact div_le_div_of_nonneg_right hr (by positivity)

theorem exp_word (word : Nat) :
    (Model.ofNatBits word : Model FloatFormat.binary64).expField=word%2^63/2^52 := by
  rw [Model.expField_eq_expFieldImpl_apply]
  change ((word%2^64)>>>52)&&&(2^11-1)=word%2^63/2^52
  rw [Nat.and_two_pow_sub_one_eq_mod,Nat.shiftRight_eq_div_pow]
  omega

theorem finite_raw (word : Nat) (hf : word%2^63<2047*2^52) :
    (Model.ofNatBits word : Model FloatFormat.binary64).isFinite=true := by
  change ((Model.ofNatBits word : Model FloatFormat.binary64).expField != 2047)=true
  rw [exp_word]
  simp only [bne_iff_ne]
  omega

theorem predict_finite (words : List Nat) (valid : ∀ w∈words,w%2^63<2047*2^52) :
    predict words = FabricF64.round (accumulator words) := by
  have hn : ∀ w ∈ words, ¬ 2047*2^52 < w%2^63 := by
    intro w hw
    have := valid w hw
    omega
  have hi : ∀ w ∈ words, ¬ w%2^63=2047*2^52 := by
    intro w hw
    have := valid w hw
    omega
  have hnf : words.filter (fun w => decide (w%2^63>2047*2^52))=[] := by
    simp only [List.filter_eq_nil_iff,Bool.not_eq_true,decide_eq_false_iff_not]
    exact hn
  have hif : words.filter (fun w => decide (w%2^63=2047*2^52))=[] := by
    simp only [List.filter_eq_nil_iff,Bool.not_eq_true,decide_eq_false_iff_not]
    exact hi
  change words.filter (fun w => decide (w%9223372036854775808>9218868437227405312))=[] at hnf
  change words.filter (fun w => decide (w%9223372036854775808=9218868437227405312))=[] at hif
  simp [predict,hnf,hif,accumulator]

theorem decoded_model_error (words : List Nat)
    (valid : ∀ w∈words,w<2^64 ∧ w%2^63<2047*2^52)
    (finite : (Model.ofNatBits (predict words) : Model FloatFormat.binary64).isFinite=true) :
    |(Model.ofNatBits (predict words) : Model FloatFormat.binary64).toReal-
      (words.map fun w => (Model.ofNatBits w : Model FloatFormat.binary64).toReal).sum| ≤
      (loss words:ℝ)/2^1074+
      ulp FloatLib.Numerics.binaryRadix (Model.fexpOf FloatFormat.binary64)
        ((accumulator words:ℝ)/2^1074)/2 := by
  have he := predict_finite words (fun w hw => (valid w hw).2)
  rw [he] at finite ⊢
  have hu : ∀ w∈words,∃v,units w=some v := by
    intro w hw
    exact FabricF64Decode.units_defined w (valid w hw).1 (finite_raw w (valid w hw).2)
  rw [FabricF64Decode.sum_units_value words hu]
  have hr := round_error (accumulator words) finite
  have ha := accumulator_real_error words
  have ht := abs_sub_le
    (Model.ofNatBits (FabricF64.round (accumulator words)) : Model FloatFormat.binary64).toReal
    ((accumulator words:ℝ)/2^1074)
    ((((words.map fun w => (units w).getD 0).sum:Int):ℝ)/2^1074)
  linarith

#print axioms dyadic_value
#print axioms round_word
#print axioms round_error
#print axioms truncate_loss
#print axioms sum_loss
#print axioms flush_loss
#print axioms accumulator_loss
#print axioms accumulator_real_error
#print axioms exp_word
#print axioms finite_raw
#print axioms predict_finite
#print axioms decoded_model_error
end FabricF64Bounds
