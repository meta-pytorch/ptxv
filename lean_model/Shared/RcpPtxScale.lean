import Approx
import Mathlib.Tactic
import FloatLib.Floats.Formats.BinaryInterchange.Model.Fields.Proof
import FloatLib.Numerics.ShiftRightJam.Proof
import FloatLib.Numerics.Bitwise
set_option maxRecDepth 10000
namespace GB300Math.RcpPtxBounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem decode_normal (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23) :
    Model.toDyadic? (Model.ofFields FloatFormat.binary32 negative e f) =
      some ⟨negative, 2^23+f, (e : Int)-150⟩ := by
  have h := Model.toDyadic?_ofFields_normal FloatFormat.binary32 negative e f
    (by omega) (by change e < 255; omega) hf
  have h' : Model.toDyadic? (Model.ofFields FloatFormat.binary32 negative e f) =
      some ⟨negative, 8388608+f, (e : Int)-127-23⟩ := h
  have hexp : (e : Int)-127-23 = (e : Int)-150 := by omega
  norm_num only [hexp] at h' ⊢
  exact h'

theorem mul_power_two_normal (negative : Bool) (e f p : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23)
    (hp : 1 ≤ p ∧ p ≤ 254) (hs : 128 ≤ e+p ∧ e+p ≤ 381) :
    Model.mul (Model.ofFields FloatFormat.binary32 negative e f)
      (Model.ofFields FloatFormat.binary32 false p 0) =
      Model.ofFields FloatFormat.binary32 negative (e+p-127) f := by
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
  have hhigh : ¬127 < (e : Int)+p-254 := by omega
  have hcarry : 8388608+f ≠ 16777216 := by omega
  have hout : ((e : Int)+p-254+127).toNat = e+p-127 := by omega
  norm_num [Model.roundDyadicImpl, Model.ieeeRoundDyadicImpl,
    hfmt, hmin, hmax, hwidth, hbias, hlog, hround, hexp, hlow, hhigh, hcarry, hout, Model.pow2]


theorem mul_subnormal_normalize (negative : Bool) (f : Nat)
    (hf0 : f ≠ 0) (hf : f < 2^23) :
    Model.mul (Model.ofFields FloatFormat.binary32 negative 0 f)
      (Model.ofFields FloatFormat.binary32 false 151 0) =
      Model.ofFields FloatFormat.binary32 negative (f.log2+2)
        (f*2^(23-f.log2)-2^23) := by
  have hd := Model.toDyadic?_ofFields_subnormal FloatFormat.binary32 negative f hf0 hf
  have hd' : Model.toDyadic? (Model.ofFields FloatFormat.binary32 negative 0 f) =
      some ⟨negative, f, -149⟩ := hd
  rw [Model.Proof.mul_eq_spec]
  unfold Model.Spec.mul
  rw [hd', decode_normal false 151 0 (by norm_num) (by norm_num)]
  simp [hf0]
  rw [Model.roundDyadic_eq_roundDyadicImpl]
  have hlead : f.log2 ≤ 22 := by
    have h := (Nat.log2_lt hf0).mpr hf
    omega
  have hlog : (f*8388608).log2 = f.log2+23 := by
    have h := Nat.log2_shiftLeft_of_ne_zero f 23 hf0
    simpa [Nat.shiftLeft_eq] using h
  have hround := FloatLib.Numerics.roundShiftRightEven_mul_two_pow_of_le f 23 f.log2 (by omega)
  norm_num at hround
  have hm0 : f*2^(23-f.log2) ≠ 0 := Nat.mul_ne_zero hf0 (by positivity)
  have hmlog : (f*2^(23-f.log2)).log2 = 23 := by
    have h := Nat.log2_shiftLeft_of_ne_zero f (23-f.log2) hf0
    rw [Nat.shiftLeft_eq] at h
    omega
  have hmlo := Nat.log2_self_le hm0
  have hmhi := Nat.lt_log2_self (n := f*2^(23-f.log2))
  rw [hmlog] at hmlo hmhi
  have hfmt : FloatFormat.binary32.isIEEE = true := by decide
  have hmin : FloatFormat.ieeeMinNormalExponent FloatFormat.binary32 = -126 := by rfl
  have hmax : FloatFormat.ieeeMaxNormalExponent FloatFormat.binary32 = 127 := by rfl
  have hwidth : FloatFormat.binary32.fracWidth = 23 := by rfl
  have hbias : FloatFormat.binary32.bias = 127 := by rfl
  have hexp : ((f.log2+23 : Nat) : Int)-148 = (f.log2 : Int)-125 := by omega
  have hlow : ¬(f.log2 : Int)-125 < -126 := by omega
  have hhigh : ¬127 < (f.log2 : Int)-125 := by omega
  have hcarry : f*2^(23-f.log2) ≠ 16777216 := by omega
  have hout : ((f.log2 : Int)-125+127).toNat = f.log2+2 := by omega
  have hlo' : ¬(f.log2 : Int)+23 < 22 := by omega
  have hhi' : ¬275 < (f.log2 : Int)+23 := by omega
  have hout' : ((f.log2 : Int)+23 + -148+127).toNat = f.log2+2 := by omega
  norm_num [Model.roundDyadicImpl, Model.ieeeRoundDyadicImpl,
    hlo', hhi', hout', hf0, hfmt, hmin, hmax, hwidth, hbias, hlog, hround, hexp, hlow, hhigh, hcarry, hout, Model.pow2]

theorem fields_bits (negative : Bool) (e f : Nat)
    (he : e < 256) (hf : f < 2^23) :
    Model.toNatBits (Model.ofFields FloatFormat.binary32 negative e f) =
      (if negative then 2^31 else 0)+e*2^23+f := by
  rw [Model.ofFields_eq_ofFieldsImpl]
  change ((if negative then 1 <<< 31 else 0) ||| ((e &&& 255) <<< 23) ||| (f &&& 8388607)) = _
  have hem : e &&& 255 = e := by
    exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 8) he
  have hfm : f &&& 8388607 = f := by
    exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 23) hf
  rw [hem,hfm,Nat.or_assoc,← Nat.shiftLeft_add_eq_or_of_lt hf]
  rw [Nat.shiftLeft_eq]
  cases negative
  · simp [Nat.shiftLeft_eq]
  · simp only [↓reduceIte]
    have hx : e*2^23+f < 2^31 := by omega
    have h := Nat.two_pow_add_eq_or_of_lt hx 1
    simpa [Nat.shiftLeft_eq,Nat.add_assoc] using h.symm

theorem fields_float (negative : Bool) (e f : Nat)
    (he : e < 256) (hf : f < 2^23) :
    Model.ofFields FloatFormat.binary32 negative e f =
      Approx.asFloat ((if negative then 2^31 else 0)+e*2^23+f) := by
  rw [← fields_bits negative e f he hf]
  exact (Model.ofNatBits_toNatBits _).symm

theorem fields_word (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23) :
    Approx.asWord (Model.ofFields FloatFormat.binary32 negative e f) =
      (if negative then 2^31 else 0)+e*2^23+f := by
  have hn := Model.isNaN_eq_false_of_isFinite_eq_true _
    (Model.isFinite_eq_true_of_toDyadic?_some (decode_normal negative e f he hf))
  simp only [Approx.asWord,hn,Bool.false_eq_true,↓reduceIte]
  exact fields_bits negative e f (by omega) hf

theorem mul_power_two_normal_word (negative : Bool) (e f p : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23)
    (hp : 1 ≤ p ∧ p ≤ 254) (hs : 128 ≤ e+p ∧ e+p ≤ 381) :
    Approx.mul ((if negative then 2^31 else 0)+e*2^23+f) (p*2^23) =
      (if negative then 2^31 else 0)+(e+p-127)*2^23+f := by
  have hright : Approx.asFloat (p*2^23) = Model.ofFields FloatFormat.binary32 false p 0 := by
    simpa using (fields_float false p 0 (by omega) (by norm_num)).symm
  change Approx.asWord (Model.mul _ _) = _
  rw [← fields_float negative e f (by omega) hf,hright]
  rw [mul_power_two_normal negative e f p he hf hp hs]
  exact fields_word negative (e+p-127) f (by omega) hf

theorem normalized_mantissa_range (f : Nat) (hf0 : f ≠ 0) (hf : f < 2^23) :
    2^23 ≤ f*2^(23-f.log2) ∧ f*2^(23-f.log2) < 2^24 := by
  have hlead : f.log2 ≤ 22 := by
    have h := (Nat.log2_lt hf0).mpr hf
    omega
  have hm0 : f*2^(23-f.log2) ≠ 0 := Nat.mul_ne_zero hf0 (by positivity)
  have hmlog : (f*2^(23-f.log2)).log2 = 23 := by
    have h := Nat.log2_shiftLeft_of_ne_zero f (23-f.log2) hf0
    rw [Nat.shiftLeft_eq] at h
    omega
  have hmlo := Nat.log2_self_le hm0
  have hmhi := Nat.lt_log2_self (n := f*2^(23-f.log2))
  rw [hmlog] at hmlo hmhi
  exact ⟨hmlo, hmhi⟩

theorem mul_subnormal_normalize_word (negative : Bool) (f : Nat)
    (hf0 : f ≠ 0) (hf : f < 2^23) :
    Approx.mul ((if negative then 2^31 else 0)+f) 0x4b800000 =
      (if negative then 2^31 else 0)+(f.log2+2)*2^23+(f*2^(23-f.log2)-2^23) := by
  have hright : Approx.asFloat 0x4b800000 = Model.ofFields FloatFormat.binary32 false 151 0 := by
    simpa using (fields_float false 151 0 (by norm_num) (by norm_num)).symm
  have hleft : Approx.asFloat ((if negative then 2^31 else 0)+f) =
      Model.ofFields FloatFormat.binary32 negative 0 f := by
    simpa using (fields_float negative 0 f (by norm_num) hf).symm
  change Approx.asWord (Model.mul _ _) = _
  rw [hleft,hright,mul_subnormal_normalize negative f hf0 hf]
  have hlead : f.log2 ≤ 22 := by
    have h := (Nat.log2_lt hf0).mpr hf
    omega
  have hm := normalized_mantissa_range f hf0 hf
  exact fields_word negative (f.log2+2) (f*2^(23-f.log2)-2^23) (by omega) (by omega)

theorem mul_normal_comm (nx ny : Bool) (ex fx ey fy : Nat)
    (hx : 1 ≤ ex ∧ ex ≤ 254) (hfx : fx < 2^23)
    (hy : 1 ≤ ey ∧ ey ≤ 254) (hfy : fy < 2^23) :
    Model.mul (Model.ofFields FloatFormat.binary32 nx ex fx)
      (Model.ofFields FloatFormat.binary32 ny ey fy) =
    Model.mul (Model.ofFields FloatFormat.binary32 ny ey fy)
      (Model.ofFields FloatFormat.binary32 nx ex fx) := by
  rw [Model.Proof.mul_eq_spec,Model.Proof.mul_eq_spec]
  unfold Model.Spec.mul
  rw [decode_normal nx ex fx hx hfx, decode_normal ny ey fy hy hfy]
  simp [Nat.mul_comm,Int.add_comm,Bool.xor_comm,Bool.or_comm]

theorem mul_power_two_normal_word_left (negative : Bool) (e f p : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23)
    (hp : 1 ≤ p ∧ p ≤ 254) (hs : 128 ≤ e+p ∧ e+p ≤ 381) :
    Approx.mul (p*2^23) ((if negative then 2^31 else 0)+e*2^23+f) =
      (if negative then 2^31 else 0)+(e+p-127)*2^23+f := by
  have hleft : Approx.asFloat (p*2^23) = Model.ofFields FloatFormat.binary32 false p 0 := by
    simpa using (fields_float false p 0 (by omega) (by norm_num)).symm
  change Approx.asWord (Model.mul _ _) = _
  rw [← fields_float negative e f (by omega) hf,hleft]
  rw [mul_normal_comm false negative p 0 e f hp (by norm_num) he hf]
  rw [mul_power_two_normal negative e f p he hf hp hs]
  exact fields_word negative (e+p-127) f (by omega) hf

theorem mul_one_normal (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23) :
    let word := (if negative then 2^31 else 0)+e*2^23+f
    Approx.mul word 0x3f800000 = word ∧ Approx.mul 0x3f800000 word = word := by
  have h0 := mul_power_two_normal_word negative e f 127 he hf (by omega) (by omega)
  have h1 := mul_power_two_normal_word_left negative e f 127 he hf (by omega) (by omega)
  simpa using And.intro h0 h1

#print axioms mul_one_normal
#print axioms mul_subnormal_normalize_word
#print axioms fields_bits
#print axioms mul_power_two_normal_word
#print axioms mul_power_two_normal
#print axioms mul_subnormal_normalize
end GB300Math.RcpPtxBounds
