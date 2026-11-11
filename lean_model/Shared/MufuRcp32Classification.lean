import MufuRcp32Tail
import MufuRcp32Signed
namespace GB300Math.MufuRcp32Bounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem positive_finite_tail_bits (w : Nat)
    (lo : 0x7e800000 < w) (hi : w < 0x7f800000) : Mufu.rcp w = 0 := by
  have hq : w / 2^23 = 253 ∨ w / 2^23 = 254 := by omega
  cases hq with
  | inl hq =>
    let f : Fin (2^23) := ⟨w % 2^23, Nat.mod_lt _ (by norm_num)⟩
    have hw : w = 253*2^23+f.val := by dsimp [f]; omega
    have hf : f.val ≠ 0 := by omega
    rw [hw, penultimate_normal_exponent]
    simp [hf]
  | inr hq =>
    have he : Mufu.exponent w = 254 := by unfold Mufu.exponent; omega
    have hm : Mufu.magnitude w ≤ 0x7f800000 := by
      unfold Mufu.magnitude
      rw [Nat.mod_eq_of_lt (show w < 2^31 by omega)]
      omega
    rw [highest_normal_exponent_flushes w he hm]
    exact sign_zero_of_lt w (by omega)

theorem finite_tail_floatlib_error (w : Nat)
    (lo : 0x7e800000 < w) (hi : w < 0x7f800000) :
    ∃ x : Rat,
      (Model.ofNatBits w : Model FloatFormat.binary32).toRat? = some x ∧
      0 < x ∧
      (Model.ofNatBits (Mufu.rcp w) : Model FloatFormat.binary32).toRat? = some 0 ∧
      |(0 : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 1 := by
  let e : Fin 254 := ⟨w / 2^23-1, by omega⟩
  let f : Fin (2^23) := ⟨w % 2^23, Nat.mod_lt _ (by norm_num)⟩
  have hw : w = (1+e.val)*2^23+f.val := by dsimp [e,f]; omega
  let x : Rat := ((2^23+f.val : Nat) : Rat)*(2 : Rat)^((e.val : Int)-149)
  have hx : 0 < x := by dsimp [x]; positivity
  refine ⟨x, ?_, hx, ?_, ?_⟩
  · rw [hw]
    exact decode_normal e f
  · rw [positive_finite_tail_bits w lo hi]
    decide +kernel
  · apply flushed_relative_error
    have hxr : (0 : ℝ) < (x : ℝ) := by exact_mod_cast hx
    exact ne_of_gt hxr

theorem negative_finite_tail_floatlib_error (w : Nat)
    (lo : 0x7e800000 < w) (hi : w < 0x7f800000) :
    ∃ x : Rat,
      (Model.ofNatBits (w+2^31) : Model FloatFormat.binary32).toRat? = some x ∧
      x < 0 ∧
      Mufu.rcp (w+2^31) = 0x80000000 ∧
      (Model.ofNatBits (Mufu.rcp (w+2^31)) : Model FloatFormat.binary32).toRat? = some 0 ∧
      |(0 : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 1 := by
  obtain ⟨x,hx,hpos,hout,herr⟩ := finite_tail_floatlib_error w lo hi
  have hbits : Mufu.rcp (w+2^31) = 0x80000000 := by
    rw [Nat.add_comm w, RcpSigned.rcp_sign w (by omega), positive_finite_tail_bits w lo hi]
    rfl
  refine ⟨-x, RcpSigned.signed_decode w x hx, by linarith, hbits, ?_, ?_⟩
  · rw [hbits]
    decide +kernel
  · simpa only [Rat.cast_neg, one_div, inv_neg, zero_sub, neg_neg, abs_neg] using herr

theorem boundary_power_exact :
    Mufu.rcp 0x7e800000 = 0x00800000 ∧
    (Model.ofNatBits 0x7e800000 : Model FloatFormat.binary32).toRat? = some ((2 : Rat)^126) ∧
    (Model.ofNatBits 0x00800000 : Model FloatFormat.binary32).toRat? = some (1/(2 : Rat)^126) := by
  decide +kernel

theorem boundary_power_floatlib_error :
    ∃ x y : Rat,
      (Model.ofNatBits 0x7e800000 : Model FloatFormat.binary32).toRat? = some x ∧
      (Model.ofNatBits (Mufu.rcp 0x7e800000) : Model FloatFormat.binary32).toRat? = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 0 := by
  refine ⟨(2 : Rat)^126, 1/(2 : Rat)^126, boundary_power_exact.2.1, ?_, ?_⟩
  · rw [boundary_power_exact.1]
    exact boundary_power_exact.2.2
  · norm_num

theorem negative_boundary_power_floatlib_error :
    ∃ x y : Rat,
      (Model.ofNatBits 0xfe800000 : Model FloatFormat.binary32).toRat? = some x ∧
      (Model.ofNatBits (Mufu.rcp 0xfe800000) : Model FloatFormat.binary32).toRat? = some y ∧
      |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 0 := by
  obtain ⟨x,y,hx,hy,herr⟩ := boundary_power_floatlib_error
  refine ⟨-x,-y,RcpSigned.signed_decode _ _ hx,?_,?_⟩
  · have hs := RcpSigned.rcp_sign 0x7e800000 (by omega)
    rw [Nat.add_comm (2^31) (Mufu.rcp 0x7e800000)] at hs
    change Mufu.rcp 0xfe800000 = _ at hs
    rw [hs]
    exact RcpSigned.signed_decode _ _ hy
  · simpa only [Rat.cast_neg, one_div, inv_neg, neg_sub_neg, abs_sub_comm, abs_neg] using herr

abbrev hasSmallError (e f : Nat) : Prop := e < 252 ∨ e = 252 ∧ f = 0

theorem positive_normal_classification (e : Fin 254) (f : Fin (2^23)) :
    ∃ x y : Rat,
      (Model.ofNatBits ((1+e.val)*2^23+f.val) : Model FloatFormat.binary32).toRat? = some x ∧
      (Model.ofNatBits (Mufu.rcp ((1+e.val)*2^23+f.val)) : Model FloatFormat.binary32).toRat? = some y ∧
      (if hasSmallError e.val f.val then
        |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 1/(2^23 : ℝ)
       else |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 1) := by
  by_cases hlo : e.val < 252
  · obtain ⟨x,y,hx,hy,herr⟩ := positive_normal_floatlib_error ⟨e.val,hlo⟩ f
    exact ⟨x,y,hx,hy,by simpa [hasSmallError,hlo] using herr⟩
  · by_cases heq : e.val = 252 ∧ f.val = 0
    · have hw : (1+e.val)*2^23+f.val = 0x7e800000 := by omega
      obtain ⟨x,y,hx,hy,herr⟩ := boundary_power_floatlib_error
      refine ⟨x,y,?_,?_,?_⟩
      · simpa only [hw] using hx
      · simpa only [hw] using hy
      · have hc : hasSmallError e.val f.val := Or.inr heq
        rw [ite_eq_left hc, herr]
        norm_num
    · have he := e.isLt
      have hf := f.isLt
      have lo : 0x7e800000 < (1+e.val)*2^23+f.val := by omega
      have hi : (1+e.val)*2^23+f.val < 0x7f800000 := by omega
      obtain ⟨x,hx,hpos,hy,herr⟩ := finite_tail_floatlib_error _ lo hi
      refine ⟨x,0,hx,hy,?_⟩
      have hc : ¬hasSmallError e.val f.val := by
        intro h
        rcases h with h | h
        · exact hlo h
        · exact heq h
      rw [ite_eq_right hc]
      simpa only [Rat.cast_zero] using herr

theorem negative_normal_classification (e : Fin 254) (f : Fin (2^23)) :
    let w := (1+e.val)*2^23+f.val
    ∃ x y : Rat,
      (Model.ofNatBits (w+2^31) : Model FloatFormat.binary32).toRat? = some x ∧
      (Model.ofNatBits (Mufu.rcp (w+2^31)) : Model FloatFormat.binary32).toRat? = some y ∧
      (if hasSmallError e.val f.val then
        |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| ≤ 1/(2^23 : ℝ)
       else |(y : ℝ)-1/(x : ℝ)|/|1/(x : ℝ)| = 1) := by
  dsimp only
  obtain ⟨x,y,hx,hy,herr⟩ := positive_normal_classification e f
  refine ⟨-x,-y,RcpSigned.signed_decode _ _ hx,?_,?_⟩
  · have he := e.isLt
    have hf := f.isLt
    have hs := RcpSigned.rcp_sign ((1+e.val)*2^23+f.val) (by omega)
    rw [Nat.add_comm (2^31) _, Nat.add_comm (2^31) _] at hs
    rw [hs]
    exact RcpSigned.signed_decode _ _ hy
  · simpa only [Rat.cast_neg, one_div, inv_neg, neg_sub_neg, abs_sub_comm, abs_neg] using herr

theorem normal_domain_counts :
    2*254*2^23 = 4261412864 ∧
    2*252*2^23+2 = 4227858434 ∧
    2*(2*2^23-1) = 33554430 ∧
    4227858434+33554430 = 4261412864 := by decide +kernel

theorem low_exponent_input_flushes (w : Nat) (hw : w < 2^23) :
    Mufu.rcp w = 0x7f800000 := by
  have he : Mufu.exponent w = 0 := by unfold Mufu.exponent; omega
  have hs : Mufu.sign w = 0 := sign_zero_of_lt w (by omega)
  have hm : ¬Mufu.magnitude w > 0x7f800000 := by
    unfold Mufu.magnitude
    rw [Nat.mod_eq_of_lt (show w < 2^31 by omega)]
    omega
  unfold Mufu.rcp
  rw [he,hs]
  simp only [hm,ite_false,ite_true,Nat.zero_add]

theorem low_exponent_negative_input_flushes (w : Nat) (hw : w < 2^23) :
    Mufu.rcp (w+2^31) = 0xff800000 := by
  rw [Nat.add_comm w, RcpSigned.rcp_sign w (by omega), low_exponent_input_flushes w hw]
  rfl

theorem nan_input_result (w : Nat) (hn : Mufu.magnitude w > 0x7f800000) :
    Mufu.rcp w = Mufu.canonicalNaN := by
  unfold Mufu.rcp
  rw [ite_eq_left hn]

theorem reciprocal_infinity_results :
    Mufu.rcp 0x7f800000 = 0 ∧ Mufu.rcp 0xff800000 = 0x80000000 := by
  decide +kernel

#print axioms low_exponent_input_flushes
#print axioms low_exponent_negative_input_flushes
#print axioms nan_input_result
#print axioms reciprocal_infinity_results
#print axioms positive_normal_classification
#print axioms negative_normal_classification
#print axioms normal_domain_counts
#print axioms boundary_power_exact
#print axioms boundary_power_floatlib_error
#print axioms negative_boundary_power_floatlib_error
#print axioms positive_finite_tail_bits
#print axioms finite_tail_floatlib_error
#print axioms negative_finite_tail_floatlib_error
end GB300Math.MufuRcp32Bounds
