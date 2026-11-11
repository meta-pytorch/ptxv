/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricBF16Decode
set_option exponentiation.threshold 1024
namespace FabricBF16StochasticBounds
open FabricBF16Block

def code (x : Nat) : Nat := 128*(x.log2-7)+x/2^(x.log2-7)

theorem selected_small (x : Nat) (hx : x < 256) :
    code x = x ∧ unit (code x) = x := by
  have hl : x.log2 ≤ 7 := by
    by_cases h : x = 0
    · simp [h]
    · have := (Nat.log2_lt (k := 8) h).mpr hx
      omega
  simp only [code, Nat.sub_eq_zero_of_le hl, pow_zero, Nat.div_one, Nat.mul_zero, Nat.zero_add]
  have hd : x/128 = 0 ∨ x/128 = 1 := by omega
  rcases hd with h | h <;> simp [unit, h] <;> omega

theorem selected_normal (x : Nat) (hx : 256 ≤ x) :
    let s := x.log2-7
    let c := code x
    c/128 = s+1 ∧ c%128 = x/2^s-128 ∧
    unit c ≤ x ∧ x < unit (c+1) ∧ unit (c+1)-unit c = 2^s := by
  have hnonzero : x ≠ 0 := by omega
  have hl : 8 ≤ x.log2 := (Nat.le_log2 hnonzero).mpr hx
  let s := x.log2-7
  have he : x.log2 = s+7 := by dsimp [s]; omega
  have hp : 0 < 2^s := Nat.two_pow_pos _
  have hb := (Nat.log2_eq_iff hnonzero).mp (rfl : x.log2=x.log2)
  rw [he] at hb
  have hlo : 128*2^s ≤ x := by simpa [Nat.pow_add, Nat.mul_comm] using hb.1
  have hhi : x < 256*2^s := by simpa [show s+7+1=s+8 by omega, Nat.pow_add, Nat.mul_comm] using hb.2
  have hqlo : 128 ≤ x/2^s := (Nat.le_div_iff_mul_le hp).mpr hlo
  have hqhi : x/2^s < 256 := (Nat.div_lt_iff_lt_mul hp).mpr hhi
  have hdiv : code x/128 = s+1 := by dsimp [code]; change (128*s+x/2^s)/128=s+1; omega
  have hmod : code x%128 = x/2^s-128 := by dsimp [code]; change (128*s+x/2^s)%128=x/2^s-128; omega
  have hu : unit (code x) = x/2^s*2^s := by
    simp only [unit,hdiv,hmod,show ¬s+1=0 by omega,↓reduceIte,Nat.add_sub_cancel]
    congr 1
    omega
  have hun : unit (code x+1) = (x/2^s+1)*2^s := by
    by_cases h : x/2^s = 255
    · have hd : (code x+1)/128=s+2 := by omega
      have hm : (code x+1)%128=0 := by omega
      simp only [unit,hd,hm,show ¬s+2=0 by omega,↓reduceIte,Nat.zero_add]
      rw [show s+2-1=s+1 by omega,Nat.pow_succ,h]
      omega
    · have hd : (code x+1)/128=s+1 := by omega
      have hm : (code x+1)%128=x/2^s-127 := by omega
      simp only [unit,hd,hm,show ¬s+1=0 by omega,↓reduceIte,Nat.add_sub_cancel]
      congr 1
      omega
  have hrem := Nat.mod_lt x hp
  have hsplit := Nat.mod_add_div x (2^s)
  dsimp only
  change code x/128=s+1 ∧ code x%128=x/2^s-128 ∧ _
  rw [hu,hun]
  refine ⟨hdiv,hmod,Nat.div_mul_le_self _ _,?_,?_⟩
  · nlinarith
  · rw [Nat.add_mul, Nat.one_mul, Nat.add_sub_cancel_left]

theorem selected_interval (x : Nat) :
    unit (code x) ≤ x ∧ x < unit (code x+1) ∧
    unit (code x+1)-unit (code x) = 2^(x.log2-7) := by
  by_cases hx : x < 256
  · obtain ⟨hc,hu⟩ := selected_small x hx
    have hn : unit (code x+1) = x+1 := by
      rw [hc]
      by_cases he : x = 255
      · simp [he,unit]
      · simpa only [(selected_small (x+1) (by omega)).1] using (selected_small (x+1) (by omega)).2
    have hl : x.log2 ≤ 7 := by
      by_cases hz : x=0
      · simp [hz]
      · have := (Nat.log2_lt (k:=8) hz).mpr hx
        omega
    rw [hu,hn,Nat.sub_eq_zero_of_le hl]
    simp
  · exact (selected_normal x (by omega)).2.2

theorem code_finite (x : Nat) (hx : x < unit 32640) : code x < 32640 := by
  by_cases hs : x < 256
  · rw [(selected_small x hs).1]
    omega
  · have hc := (selected_normal x (by omega)).1
    have hn : x ≠ 0 := by omega
    have hl : x.log2 < 261 := (Nat.log2_lt hn).mpr hx
    omega

theorem output_adjacent (states : Array Accum) (lane : Nat) (value : Int)
    (hv : states[lane]! = .finite value) (hx : value.natAbs < unit 32640) :
    ∃ y, y ≤ 32640 ∧
      output states lane true = (if value < 0 then 32768 else 0)+y ∧
      (y=code value.natAbs ∨ y=code value.natAbs+1) ∧
      (value.natAbs=unit (code value.natAbs) → y=code value.natAbs) := by
  have hc := code_finite value.natAbs hx
  by_cases he : value.natAbs=unit (code value.natAbs)
  · refine ⟨code value.natAbs,by omega,?_,Or.inl rfl,by simp⟩
    simp only [output,hv,Bool.not_true,Bool.false_eq_true,↓reduceIte,Nat.not_le.mpr hx]
    change (if value.natAbs=unit (code value.natAbs) then _ else _) = _
    rw [ite_eq_left he]
    rfl
  · let bump := if (value.natAbs-unit (code value.natAbs))*2^42+
        entropy states lane*(unit (code value.natAbs+1)-unit (code value.natAbs)) ≥
        (unit (code value.natAbs+1)-unit (code value.natAbs))*2^42 then 1 else 0
    have hb : bump=0 ∨ bump=1 := by dsimp [bump]; split <;> simp
    refine ⟨code value.natAbs+bump,by omega,?_,by omega,by simp [he]⟩
    simp only [output,hv,Bool.not_true,Bool.false_eq_true,↓reduceIte,Nat.not_le.mpr hx]
    change (if value.natAbs=unit (code value.natAbs) then _ else _) = _
    rw [ite_eq_right he]
    change (if value<0 then 32768 else 0)+code value.natAbs+bump = _
    omega

theorem adjacent_error (x y : Nat)
    (hy : y=code x ∨ y=code x+1)
    (hexact : x=unit (code x) → y=code x) :
    (Int.natAbs ((unit y:Int)-x)) < 2^(x.log2-7) := by
  obtain ⟨hl,hu,hgap⟩ := selected_interval x
  by_cases he : x=unit (code x)
  · rw [hexact he,←he]
    simp
  · rcases hy with rfl | rfl <;> omega

theorem adjacent_relative (x y : Nat)
    (hy : y=code x ∨ y=code x+1)
    (hexact : x=unit (code x) → y=code x) :
    128*(Int.natAbs ((unit y:Int)-x)) ≤ x := by
  by_cases hs : x < 256
  · have hu := (selected_small x hs).2
    rw [hexact hu.symm,hu]
    simp
  · have he := adjacent_error x y hy hexact
    have hl : 128*2^(x.log2-7) ≤ x := by
      have hn : x≠0 := by omega
      have hh : 7 ≤ x.log2 := by
        have := (Nat.le_log2 (k:=8) hn).mpr (show 2^8≤x by omega)
        omega
      have hp := Nat.log2_self_le hn
      rw [show x.log2=x.log2-7+7 by omega,Nat.pow_add] at hp
      simpa [Nat.mul_comm] using hp
    omega

open FloatLib.Floats.Formats.BinaryInterchange

theorem infinity_nonfinite (negative : Bool) :
    Model.isFinite (Model.ofNatBits ((if negative then 32768 else 0)+32640) :
      Model FloatFormat.bfloat16)=false := by cases negative <;> decide +kernel

theorem finite_accumulator (states : Array Accum) (lane : Nat) (value : Int)
    (hv : states[lane]! = .finite value)
    (hf : Model.isFinite (Model.ofNatBits (output states lane true) : Model FloatFormat.bfloat16)=true) :
    value.natAbs < unit 32640 := by
  by_contra h
  have ho : output states lane true = (if value<0 then 32768 else 0)+32640 := by
    simp [output,hv,show unit 32640≤value.natAbs by omega]
  rw [ho] at hf
  have hbad := infinity_nonfinite (decide (value<0))
  simp only [decide_eq_true_eq] at hbad
  rw [hbad] at hf
  contradiction

theorem finite_output_code (states : Array Accum) (lane y : Nat) (value : Int)
    (hy : y≤32640)
    (ho : output states lane true = (if value<0 then 32768 else 0)+y)
    (hf : Model.isFinite (Model.ofNatBits (output states lane true) : Model FloatFormat.bfloat16)=true) :
    y<32640 := by
  by_contra h
  have he : y=32640 := by omega
  rw [he] at ho
  rw [ho] at hf
  have hbad := infinity_nonfinite (decide (value<0))
  simp only [decide_eq_true_eq] at hbad
  rw [hbad] at hf
  contradiction

theorem signed_units (value : Int) (y : Nat) (hy : y<32640) :
    units ((if value<0 then 32768 else 0)+y) =
      some (if value<0 then -(unit y : Int) else (unit y : Int)) := by
  have hw : (if value<0 then 32768 else 0)+y<65536 := by split <;> omega
  have hm : ((if value<0 then 32768 else 0)+y)%32768=y := by
    split <;> simp [Nat.mod_eq_of_lt (show y<32768 by omega)]
  rw [FabricBF16Decode.units_canonical _ hw (by rw [hm];exact hy),hm]
  split <;> simp_all [show y<32768 by omega]

theorem decoded_round_error (states : Array Accum) (lane : Nat) (value : Int)
    (hv : states[lane]! = .finite value)
    (hf : Model.isFinite (Model.ofNatBits (output states lane true) : Model FloatFormat.bfloat16)=true) :
    |Model.toReal (Model.ofNatBits (output states lane true) : Model FloatFormat.bfloat16) -
      (value:ℝ)/2^133| ≤ |(value:ℝ)/2^133|/128 := by
  have hx := finite_accumulator states lane value hv hf
  obtain ⟨y,hy,ho,hadj,hexact⟩ := output_adjacent states lane value hv hx
  have hyf := finite_output_code states lane y value hy ho hf
  let v : Int := if value<0 then -(unit y : Int) else (unit y : Int)
  have hu : units (output states lane true)=some v := by rw [ho];exact signed_units value y hyf
  rw [FabricBF16Decode.units_value _ v hu]
  have hb := adjacent_relative value.natAbs y hadj hexact
  have hdist : (v-value).natAbs=((unit y:Int)-value.natAbs).natAbs := by
    dsimp [v]
    split <;> omega
  have hi : 128*(v-value).natAbs≤value.natAbs := by rw [hdist];exact hb
  have hr : (128:ℝ)*|((v-value:Int):ℝ)|≤|(value:ℝ)| := by
    have hn : (128:ℝ)*(v-value).natAbs≤(value.natAbs:ℝ) := by exact_mod_cast hi
    simpa using hn
  rw [Int.cast_sub] at hr
  rw [←sub_div,abs_div,abs_div,abs_of_pos (by positivity : (0:ℝ)<2^133)]
  rw [div_le_div_iff₀ (by positivity) (by norm_num)]
  nlinarith

theorem decoded_model_error (words : List Nat) (states : Array Accum) (lane : Nat)
    (valid : ∀ w∈words, w<65536 ∧ w%32768<32640)
    (hs : states[lane]! = accumulate words)
    (hf : Model.isFinite (Model.ofNatBits (output states lane true) : Model FloatFormat.bfloat16)=true) :
    let values := words.map fun w => (units w).getD 0
    |Model.toReal (Model.ofNatBits (output states lane true) : Model FloatFormat.bfloat16) -
      (words.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16).toReal).sum| ≤
      (FabricBF16RoundBounds.loss words values : ℝ)/2^133 +
      |(FabricBF16RoundBounds.acc words values : ℝ)/2^133|/128 := by
  let values := words.map fun w => (units w).getD 0
  have hv : states[lane]! = .finite (FabricBF16RoundBounds.acc words values) := by
    rw [hs]
    exact FabricBF16RoundBounds.accumulate_finite words (fun w hw => (valid w hw).2)
  have hr := decoded_round_error states lane (FabricBF16RoundBounds.acc words values) hv hf
  have ha := FabricBF16RoundBounds.accumulator_real_error words values
  have hu : ∀ w∈words, ∃ v,units w=some v := by
    intro w hw
    exact FabricBF16Decode.units_defined w (valid w hw).1 (FabricBF16Decode.finite_raw w (valid w hw).2)
  rw [FabricBF16Decode.sum_units_value words hu]
  have ht := abs_sub_le
    (Model.toReal (Model.ofNatBits (output states lane true) : Model FloatFormat.bfloat16))
    ((FabricBF16RoundBounds.acc words values : ℝ)/2^133)
    ((GB300Math.TensorAlignmentLoss.exactSum values : ℝ)/2^133)
  dsimp [values] at *
  linarith

#print axioms selected_small
#print axioms selected_normal
#print axioms selected_interval
#print axioms code_finite
#print axioms output_adjacent
#print axioms adjacent_error
#print axioms adjacent_relative
#print axioms infinity_nonfinite
#print axioms finite_accumulator
#print axioms finite_output_code
#print axioms signed_units
#print axioms decoded_round_error
#print axioms decoded_model_error
end FabricBF16StochasticBounds
