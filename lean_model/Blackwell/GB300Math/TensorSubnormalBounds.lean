import GB300Math.TensorEncodingBounds
namespace GB300Math.TensorSubnormalBounds
open GB300Math.Tensor GB300Math.TensorEncodingBounds

theorem shifted_magnitude_lt (n sh : Int) (bits : Nat) (_hn : n ≠ 0)
    (bound : (n.natAbs.log2 : Int)+sh < bits) :
    (shiftTruncate n sh).natAbs < 2^bits := by
  have hu := Nat.lt_log2_self (n := n.natAbs)
  rewrite [abs_shift]
  split
  next h =>
    have he : n.natAbs.log2+1+sh.toNat ≤ bits := by omega
    have hx := Nat.mul_lt_mul_of_pos_right hu (Nat.two_pow_pos sh.toNat)
    have hp : 2^(n.natAbs.log2+1+sh.toNat) ≤ 2^bits := Nat.pow_le_pow_right (by decide) he
    exact Nat.lt_of_lt_of_le (by simpa only [← Nat.pow_add] using hx) hp
  next h =>
    apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos (-sh).toNat)).mpr
    have he : n.natAbs.log2+1 ≤ bits+(-sh).toNat := by omega
    have hp : 2^(n.natAbs.log2+1) ≤ 2^(bits+(-sh).toNat) := Nat.pow_le_pow_right (by decide) he
    simpa only [Nat.pow_add] using Nat.lt_of_lt_of_le hu hp

theorem decode_subnormal (negative : Bool) (m : Nat) (hm : m < 2^23) :
    decode ((if negative then 2^31 else 0)+m) 23 8 =
      ⟨if negative then -(m : Int) else m,-149⟩ := by
  cases negative
  · have he : m/8388608%256=0 := by omega
    have hs : m/2147483648%2=0 := by omega
    simp [decode,he,hs,Nat.mod_eq_of_lt hm]
    omega
  · have he : (2147483648+m)/8388608%256=0 := by omega
    have hs : m/2147483648=0 := by omega
    have hf : (2147483648+m)%8388608=m := by omega
    simp [decode,he,hs,hf]
    omega

theorem subnormal_encoding (n e : Int) (hn : n ≠ 0)
    (bound : e+(n.natAbs.log2 : Int) < -126) :
    encodeRz ⟨n,e⟩ =
      if shiftTruncate n (e+149) = 0 then 0
      else (if n<0 then 2^31 else 0)+(shiftTruncate n (e+149)).natAbs := by
  have hm := shifted_magnitude_lt n (e+149) 23 hn (by omega)
  have hmax : max ((n.natAbs.log2 : Int)-23) (-149-e) = -149-e := by omega
  have hs : -(-149-e)=e+149 := by omega
  simp only [encodeRz,hn,ite_false,hmax,hs]
  by_cases hz : shiftTruncate n (e+149)=0
  · simp [hz]
  · have hnz : (shiftTruncate n (e+149)).natAbs ≠ 0 := by omega
    have hl : (shiftTruncate n (e+149)).natAbs.log2 < 23 := (Nat.log2_lt hnz).mpr hm
    have he : e+(-149-e)+((shiftTruncate n (e+149)).natAbs.log2 : Int)+127 ≤ 0 := by omega
    simp [hz,hnz,show ¬ e+(-149-e)+((shiftTruncate n (e+149)).natAbs.log2 : Int)+127 ≥ 255 by omega,he]

theorem decode_encode_subnormal (n e : Int) (hn : n ≠ 0)
    (bound : e+(n.natAbs.log2 : Int) < -126) :
    decode (encodeRz ⟨n,e⟩) 23 8 = ⟨shiftTruncate n (e+149),-149⟩ := by
  rewrite [subnormal_encoding n e hn bound]
  have hm := shifted_magnitude_lt n (e+149) 23 hn (by omega)
  split
  next hz =>
    simp only [hz, decode, Nat.reducePow, Nat.reduceDiv, Nat.reduceMod, Nat.reduceAdd, ↓reduceIte]
    congr 1 <;> omega
  next hz =>
    have he : (if n<0 then 2147483648 else 0) = if decide (n<0) then 2^31 else 0 := by simp
    rewrite [he,decode_subnormal _ _ hm]
    congr 1
    by_cases h : n<0 <;> simp [shiftTruncate,h]

def sourceUnits (n e : Int) : Int := n*(2^(e+149).toNat : Nat)
def resultUnits (n e : Int) : Int :=
  (decode (encodeRz ⟨n,e⟩) 23 8).significand*(2^(-149-e).toNat : Nat)
def budget (e : Int) : Int := (2^(-149-e).toNat-1 : Nat)

theorem subnormal_encoder_error (n e : Int) (hn : n ≠ 0)
    (bound : e+(n.natAbs.log2 : Int) < -126) :
    -budget e ≤ sourceUnits n e-resultUnits n e ∧
      sourceUnits n e-resultUnits n e ≤ budget e := by
  unfold sourceUnits resultUnits budget
  rewrite [decode_encode_subnormal n e hn bound]
  dsimp only
  by_cases h : e+149 ≥ 0
  · have h0 : (-149-e).toNat=0 := by omega
    have hs : e+149=((e+149).toNat : Int) := by omega
    rw [h0]
    simp only [Nat.pow_zero, Int.natCast_one, Int.mul_one, Nat.sub_self, Int.natCast_zero, Int.neg_zero]
    rw [hs, positive_shift]
    simp only [Int.toNat_natCast, Int.sub_self, Int.le_refl, and_self]
  · have h0 : (e+149).toNat=0 := by omega
    have hs : e+149=-(((-149-e).toNat : Nat) : Int) := by omega
    rw [h0]
    simp only [Nat.pow_zero, Int.natCast_one, Int.mul_one]
    rw [hs]
    exact TensorAlignmentLoss.shiftTruncate_loss n (-149-e).toNat

theorem aligned_subnormal_encoder_error (xs : List Int) (step : Nat) (e : Int)
    (positive : 0 < step)
    (hn : TensorAlignmentLoss.truncatedSum xs step ≠ 0)
    (bound : e+((TensorAlignmentLoss.truncatedSum xs step).natAbs.log2 : Int) < -126) :
    let n := TensorAlignmentLoss.truncatedSum xs step;
    let expansion : Int := (2^(e+149).toNat : Nat);
    let totalBudget : Int := ((xs.length*(step-1) : Nat) : Int)*expansion + budget e;
    -totalBudget ≤ TensorAlignmentLoss.exactSum xs*expansion-resultUnits n e ∧
      TensorAlignmentLoss.exactSum xs*expansion-resultUnits n e ≤ totalBudget := by
  dsimp only
  let n := TensorAlignmentLoss.truncatedSum xs step
  have ha := TensorAlignmentLoss.sum_truncation_loss xs step positive
  change -((xs.length*(step-1) : Nat) : Int) ≤ TensorAlignmentLoss.exactSum xs-n ∧
    TensorAlignmentLoss.exactSum xs-n ≤ ((xs.length*(step-1) : Nat) : Int) at ha
  have hr := subnormal_encoder_error n e hn bound
  have hp : 0 ≤ (2^(e+149).toNat : Nat) := Nat.zero_le _
  have hp' : 0 ≤ ((2^(e+149).toNat : Nat) : Int) := Int.ofNat_le.mpr hp
  have hlo := Int.mul_le_mul_of_nonneg_right ha.1 hp'
  have hhi := Int.mul_le_mul_of_nonneg_right ha.2 hp'
  simp only [Int.sub_mul,Int.neg_mul] at hlo hhi
  unfold sourceUnits at hr
  change -(((xs.length*(step-1) : Nat) : Int)*(2^(e+149).toNat : Nat)+budget e) ≤
      TensorAlignmentLoss.exactSum xs*(2^(e+149).toNat : Nat)-resultUnits n e ∧
    TensorAlignmentLoss.exactSum xs*(2^(e+149).toNat : Nat)-resultUnits n e ≤
      ((xs.length*(step-1) : Nat) : Int)*(2^(e+149).toNat : Nat)+budget e
  omega

theorem encode_zero (e : Int) : encodeRz ⟨0,e⟩ = 0 := by simp [encodeRz]

theorem encode_overflow (n e : Int) (hn : n ≠ 0)
    (bound : 128 ≤ e+(n.natAbs.log2 : Int)) :
    encodeRz ⟨n,e⟩ = (if n<0 then 2^31 else 0)+0x7f800000 := by
  have hn' : n.natAbs ≠ 0 := by omega
  have hm := normalized_mantissa_bounds n.natAbs hn'
  have hlog := normalized_mantissa_log n.natAbs hn'
  have hmax : max ((n.natAbs.log2 : Int)-23) (-149-e) = (n.natAbs.log2 : Int)-23 := by omega
  have hneg : -((n.natAbs.log2 : Int)-23)=23-(n.natAbs.log2 : Int) := by omega
  have hnonzero : normalizedMantissa n.natAbs ≠ 0 := by omega
  have hhigh : e+((n.natAbs.log2 : Int)-23)+23+127 ≥ 255 := by omega
  simp [encodeRz,hn,hmax,hneg,normalized_shift,hnonzero,hlog,hhigh]

theorem encoding_cases (n e : Int) : n=0 ∨
    e+(n.natAbs.log2 : Int)< -126 ∨
    (-126 ≤ e+(n.natAbs.log2 : Int) ∧ e+(n.natAbs.log2 : Int) ≤ 127) ∨
    128 ≤ e+(n.natAbs.log2 : Int) := by omega

theorem positive_subnormal_budget_attained :
    sourceUnits 3 (-151)-resultUnits 3 (-151)=budget (-151) ∧
      encodeRz ⟨3,-151⟩=0 := by decide +kernel

theorem negative_subnormal_budget_attained :
    sourceUnits (-3) (-151)-resultUnits (-3) (-151)= -budget (-151) ∧
      encodeRz ⟨-3,-151⟩=0 := by decide +kernel

theorem aligned_zero_error (xs : List Int) (step : Nat) (e : Int)
    (positive : 0 < step) (zero : TensorAlignmentLoss.truncatedSum xs step=0) :
    encodeRz ⟨TensorAlignmentLoss.truncatedSum xs step,e⟩=0 ∧
    -((xs.length*(step-1) : Nat) : Int) ≤ TensorAlignmentLoss.exactSum xs ∧
      TensorAlignmentLoss.exactSum xs ≤ ((xs.length*(step-1) : Nat) : Int) := by
  have h := TensorAlignmentLoss.sum_truncation_loss xs step positive
  simp only [zero,Int.sub_zero] at h
  exact ⟨by rw [zero,encode_zero],h⟩

#print axioms aligned_zero_error

#print axioms aligned_subnormal_encoder_error
#print axioms encode_zero
#print axioms encode_overflow
#print axioms encoding_cases
#print axioms positive_subnormal_budget_attained
#print axioms negative_subnormal_budget_attained

#print axioms shifted_magnitude_lt
#print axioms decode_subnormal
#print axioms subnormal_encoding
#print axioms decode_encode_subnormal
#print axioms subnormal_encoder_error
end GB300Math.TensorSubnormalBounds
