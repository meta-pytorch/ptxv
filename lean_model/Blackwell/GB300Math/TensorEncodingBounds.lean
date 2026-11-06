import GB300Math.TensorDyadicAlignment
namespace GB300Math.TensorEncodingBounds
open GB300Math.Tensor

def normalizedMantissa (n : Nat) : Nat :=
  if n.log2 ≥ 23 then n / 2^(n.log2-23) else n*2^(23-n.log2)

theorem normalized_mantissa_bounds (n : Nat) (hn : n ≠ 0) :
    2^23 ≤ normalizedMantissa n ∧ normalizedMantissa n < 2^24 := by
  have hl := Nat.log2_self_le hn
  have hu := Nat.lt_log2_self (n := n)
  unfold normalizedMantissa
  split
  next h =>
    have e0 : n.log2 = 23+(n.log2-23) := by omega
    have e1 : n.log2+1 = 24+(n.log2-23) := by omega
    constructor
    · apply (Nat.le_div_iff_mul_le (Nat.two_pow_pos _)).mpr
      simpa only [← Nat.pow_add, ← e0] using hl
    · apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr
      simpa only [← Nat.pow_add, ← e1] using hu
  next h =>
    have e0 : n.log2+(23-n.log2)=23 := by omega
    have e1 : n.log2+1+(23-n.log2)=24 := by omega
    have hh0 := Nat.mul_le_mul_right (2^(23-n.log2)) hl
    have hh1 := Nat.mul_lt_mul_of_pos_right hu (Nat.two_pow_pos (23-n.log2))
    constructor
    · simpa only [← Nat.pow_add, e0] using hh0
    · simpa only [← Nat.pow_add, e1] using hh1

theorem normalized_mantissa_log (n : Nat) (hn : n ≠ 0) :
    (normalizedMantissa n).log2=23 := by
  have h := normalized_mantissa_bounds n hn
  exact (Nat.log2_eq_iff (by have : 0 < 2^23 := Nat.two_pow_pos 23; omega)).mpr h

def packNormal (negative : Bool) (m e : Nat) : Nat :=
  (if negative then 2^31 else 0) + e*2^23 + m%2^23

theorem decode_normal (negative : Bool) (m e : Nat)
    (hm : 2^23 ≤ m ∧ m < 2^24) (he : 0 < e ∧ e < 255) :
    decode (packNormal negative m e) 23 8 =
      ⟨if negative then -(m : Int) else m, (e : Int)-150⟩ := by
  have hmax : max 1 e=e := by omega
  have he0 : e ≠ 0 := by omega
  cases negative
  · have hex : (e*8388608 + m%8388608)/8388608%256=e := by omega
    have hs : (e*8388608 + m%8388608)/2147483648%2=0 := by omega
    have hf : (e*8388608 + m%8388608)%8388608+8388608=m := by omega
    simp [packNormal, decode, hex, hs, he0]
    congr 1 <;> omega
  · have hex : (2147483648+e*8388608 + m%8388608)/8388608%256=e := by omega
    have hs : (2147483648+e*8388608 + m%8388608)/2147483648%2=1 := by omega
    have hf : (2147483648+e*8388608 + m%8388608)%8388608+8388608=m := by omega
    simp [packNormal, decode, hex, hs, he0]
    congr 1 <;> omega

theorem abs_shift (n sh : Int) :
    (shiftTruncate n sh).natAbs =
      if sh ≥ 0 then n.natAbs*2^sh.toNat else n.natAbs/2^(-sh).toNat := by
  unfold shiftTruncate
  split <;> split <;> simp only [Int.natAbs_neg, Int.natAbs_natCast]

theorem normalized_shift (n : Int) :
    (shiftTruncate n (23-(n.natAbs.log2 : Int))).natAbs = normalizedMantissa n.natAbs := by
  rewrite [abs_shift]
  unfold normalizedMantissa
  by_cases h : n.natAbs.log2 ≥ 23
  · by_cases he : n.natAbs.log2=23
    · simp [he]
    · have hs : ¬ 23-(n.natAbs.log2 : Int) ≥ 0 := by omega
      have hx : (-(23-(n.natAbs.log2 : Int))).toNat = n.natAbs.log2-23 := by omega
      simp only [h, hs, hx, ite_true, ite_false]
  · have hs : 23-(n.natAbs.log2 : Int) ≥ 0 := by omega
    have hx : (23-(n.natAbs.log2 : Int)).toNat=23-n.natAbs.log2 := by omega
    simp only [h, hs, hx, ite_true, ite_false]

theorem encode_normal_eq (n e : Int) (hn : n ≠ 0)
    (range : -126 ≤ e+(n.natAbs.log2 : Int) ∧ e+(n.natAbs.log2 : Int) ≤ 127) :
    encodeRz ⟨n,e⟩ = packNormal (decide (n<0)) (normalizedMantissa n.natAbs)
      (e+(n.natAbs.log2 : Int)+127).toNat := by
  have hn' : n.natAbs ≠ 0 := by omega
  have hm := normalized_mantissa_bounds n.natAbs hn'
  have hlog := normalized_mantissa_log n.natAbs hn'
  have hmax : max ((n.natAbs.log2 : Int)-23) (-149-e) = (n.natAbs.log2 : Int)-23 := by omega
  have hneg : -((n.natAbs.log2 : Int)-23)=23-(n.natAbs.log2 : Int) := by omega
  have heq : e+((n.natAbs.log2 : Int)-23)+23+127=e+(n.natAbs.log2 : Int)+127 := by omega
  have hnonzero : normalizedMantissa n.natAbs ≠ 0 := by omega
  have hhigh : ¬ e+(n.natAbs.log2 : Int)+127 ≥ 255 := by omega
  have hlow : ¬ e+(n.natAbs.log2 : Int)+127 ≤ 0 := by omega
  simp [encodeRz, hn, hmax, hneg, normalized_shift, hnonzero, hlog, heq,
    hhigh, hlow, packNormal, decide_eq_true_eq]

theorem decode_encode_normal (n e : Int) (hn : n ≠ 0)
    (range : -126 ≤ e+(n.natAbs.log2 : Int) ∧ e+(n.natAbs.log2 : Int) ≤ 127) :
    decode (encodeRz ⟨n,e⟩) 23 8 =
      ⟨shiftTruncate n (23-(n.natAbs.log2 : Int)), e+(n.natAbs.log2 : Int)-23⟩ := by
  have hn' : n.natAbs ≠ 0 := by omega
  have hm := normalized_mantissa_bounds n.natAbs hn'
  have he : 0 < (e+(n.natAbs.log2 : Int)+127).toNat ∧
      (e+(n.natAbs.log2 : Int)+127).toNat < 255 := by omega
  rewrite [encode_normal_eq n e hn range, decode_normal _ _ _ hm he]
  have hs : (if decide (n<0) then -(normalizedMantissa n.natAbs : Int) else
      (normalizedMantissa n.natAbs : Int)) = shiftTruncate n (23-(n.natAbs.log2 : Int)) := by
    rewrite [← normalized_shift]
    by_cases h : n<0 <;> simp [shiftTruncate, h]
  rewrite [hs]
  congr 1
  omega

theorem positive_shift (n : Int) (k : Nat) :
    shiftTruncate n k = n*(2^k : Nat) := by
  simp only [shiftTruncate, show (k : Int) ≥ 0 by omega, ite_true,
    Int.toNat_natCast, Int.natCast_mul]
  by_cases h : n<0
  · simp only [h, ite_true, Int.ofNat_natAbs_of_nonpos (by omega : n ≤ 0), Int.neg_mul, Int.neg_neg]
  · simp only [h, ite_false, Int.natAbs_of_nonneg (by omega : 0 ≤ n)]

def originalUnits (n : Int) : Int := n*(2^((23-(n.natAbs.log2 : Int)).toNat) : Nat)
def encodedUnits (n e : Int) : Int :=
  (decode (encodeRz ⟨n,e⟩) 23 8).significand*(2^(((n.natAbs.log2 : Int)-23).toNat) : Nat)
def roundingBudget (n : Int) : Int := (2^(((n.natAbs.log2 : Int)-23).toNat)-1 : Nat)

theorem normal_encoder_error (n e : Int) (hn : n ≠ 0)
    (range : -126 ≤ e+(n.natAbs.log2 : Int) ∧ e+(n.natAbs.log2 : Int) ≤ 127) :
    -roundingBudget n ≤ originalUnits n-encodedUnits n e ∧
      originalUnits n-encodedUnits n e ≤ roundingBudget n := by
  unfold originalUnits encodedUnits roundingBudget
  rewrite [decode_encode_normal n e hn range]
  dsimp only
  by_cases h : n.natAbs.log2 ≥ 23
  · have h0 : (23-(n.natAbs.log2 : Int)).toNat=0 := by omega
    have h1 : ((n.natAbs.log2 : Int)-23).toNat=n.natAbs.log2-23 := by omega
    have h2 : 23-(n.natAbs.log2 : Int)=-((n.natAbs.log2-23 : Nat) : Int) := by omega
    simpa [h0, h1, h2] using TensorAlignmentLoss.shiftTruncate_loss n (n.natAbs.log2-23)
  · have h0 : ((n.natAbs.log2 : Int)-23).toNat=0 := by omega
    have h1 : (23-(n.natAbs.log2 : Int)).toNat=23-n.natAbs.log2 := by omega
    have h2 : 23-(n.natAbs.log2 : Int)=((23-n.natAbs.log2 : Nat) : Int) := by omega
    simp [h0, h2, positive_shift]

def expansion (n : Int) : Int := (2^((23-(n.natAbs.log2 : Int)).toNat) : Nat)

theorem aligned_normal_encoder_error (xs : List Int) (step : Nat) (e : Int)
    (positive : 0 < step)
    (hn : TensorAlignmentLoss.truncatedSum xs step ≠ 0)
    (range : -126 ≤ e+((TensorAlignmentLoss.truncatedSum xs step).natAbs.log2 : Int) ∧
      e+((TensorAlignmentLoss.truncatedSum xs step).natAbs.log2 : Int) ≤ 127) :
    let n := TensorAlignmentLoss.truncatedSum xs step;
    let budget : Int := ((xs.length*(step-1) : Nat) : Int)*expansion n + roundingBudget n;
    -budget ≤ TensorAlignmentLoss.exactSum xs*expansion n - encodedUnits n e ∧
      TensorAlignmentLoss.exactSum xs*expansion n - encodedUnits n e ≤ budget := by
  dsimp only
  let n := TensorAlignmentLoss.truncatedSum xs step
  have ha := TensorAlignmentLoss.sum_truncation_loss xs step positive
  change -((xs.length*(step-1) : Nat) : Int) ≤ TensorAlignmentLoss.exactSum xs-n ∧
    TensorAlignmentLoss.exactSum xs-n ≤ ((xs.length*(step-1) : Nat) : Int) at ha
  have hr := normal_encoder_error n e hn range
  have hp : 0 ≤ expansion n := by
    unfold expansion
    exact Int.ofNat_le.mpr (Nat.zero_le _)
  have hlo := Int.mul_le_mul_of_nonneg_right ha.1 hp
  have hhi := Int.mul_le_mul_of_nonneg_right ha.2 hp
  simp only [Int.sub_mul, Int.neg_mul] at hlo hhi
  change -roundingBudget n ≤ n*expansion n-encodedUnits n e ∧
    n*expansion n-encodedUnits n e ≤ roundingBudget n at hr
  change -(((xs.length*(step-1) : Nat) : Int)*expansion n + roundingBudget n) ≤
      TensorAlignmentLoss.exactSum xs*expansion n-encodedUnits n e ∧
    TensorAlignmentLoss.exactSum xs*expansion n-encodedUnits n e ≤
      ((xs.length*(step-1) : Nat) : Int)*expansion n + roundingBudget n
  omega

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000
theorem positive_rounding_budget_attained :
    originalUnits 16777217 - encodedUnits 16777217 (-24) = roundingBudget 16777217 ∧
    encodeRz ⟨16777217,-24⟩ = 0x3f800000 := by decide +kernel
theorem negative_rounding_budget_attained :
    originalUnits (-16777217) - encodedUnits (-16777217) (-24) = -roundingBudget (-16777217) ∧
    encodeRz ⟨-16777217,-24⟩ = 0xbf800000 := by decide +kernel

#print axioms normalized_mantissa_bounds
#print axioms normalized_mantissa_log
#print axioms decode_normal
#print axioms abs_shift
#print axioms normalized_shift
#print axioms encode_normal_eq
#print axioms decode_encode_normal
#print axioms positive_shift
#print axioms normal_encoder_error
#print axioms aligned_normal_encoder_error
#print axioms positive_rounding_budget_attained
#print axioms negative_rounding_budget_attained
end GB300Math.TensorEncodingBounds
