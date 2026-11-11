import RcpPtxExtreme
import RcpPtxRounding
set_option maxRecDepth 10000
namespace GB300Math.RcpPtxBounds
open RcpBounds

theorem interpolate_top_only_zero (f : Nat) (hf : f < 2^23)
    (hz : (Mufu.interpolate MufuTables.rcp f).toNat = 2^24) : f = 0 := by
  have h := interpolate_error f hf
  dsimp only at h
  have hzi : Mufu.interpolate MufuTables.rcp f = (2^24 : Int) := by omega
  rw [hzi] at h
  nlinarith [Int.natCast_nonneg f]

theorem rcp_high_core_form (e f : Nat) (he : 253 ≤ e ∧ e ≤ 254)
    (hf : f < 2^23) (hlarge : e = 254 ∨ f ≠ 0) :
    ∃ eo fo, (1 ≤ eo ∧ eo ≤ 2) ∧ fo < 2^23 ∧
      Mufu.rcp ((e-2)*2^23+f) = eo*2^23+fo := by
  rw [output_word_scaled (e-2) f (by omega) hf]
  by_cases hz : (Mufu.interpolate MufuTables.rcp f).toNat = 2^24
  · have hf0 := interpolate_top_only_zero f hf hz
    have he254 : e = 254 := by omega
    exact ⟨2,0,by omega,by norm_num,by simp [hz,he254]⟩
  · norm_num at hz
    exact ⟨253-(e-2),(Mufu.interpolate MufuTables.rcp f).toNat%2^23,
      by omega,by omega,by simp [hz]⟩

theorem quarter_low_value_error (e f : Nat) (he : 1 ≤ e ∧ e ≤ 2)
    (hf : f < 2^23) :
    |(FloatLib.Numerics.roundShiftRightEven (2^23+f) (3-e) : ℚ)/2^149 -
      positiveNormalValue (e*2^23+f)/4| ≤ 1/2^150 := by
  rw [normal_decode e f (by omega) hf]
  have h := rounded_subnormal_error (2^23+f) (3-e)
  have he0 := he.1
  have he1 := he.2
  interval_cases e <;> norm_num at h ⊢ <;> convert h using 1 <;> ring_nf

theorem high_seed_error (e f : Nat) (he : 253 ≤ e ∧ e ≤ 254)
    (hf : f < 2^23) :
    |positiveNormalValue (Mufu.rcp ((e-2)*2^23+f)) -
      4/positiveNormalValue (e*2^23+f)| ≤
        if e = 253 then 1/2^148 else 1/2^149 := by
  have h := rcp_bits_scaled_absolute_error (e-2) f (by omega) hf
  rw [input_decode_scaled (e-2) f (by omega) hf] at h
  rw [input_decode_scaled e f (by omega) hf]
  have he0 := he.1
  have he1 := he.2
  interval_cases e <;> norm_num [exponentScale] at h ⊢ <;>
    convert h using 1 <;> ring_nf


theorem high_nonftz_positive_bound (e f : Nat) (he : 253 ≤ e ∧ e ≤ 254)
    (hf : f < 2^23) (hlarge : e = 254 ∨ f ≠ 0) :
    ∃ r : Nat, (0 < r ∧ r ≤ 2^23) ∧
      Approx.evaluate "rcp" false (e*2^23+f) = some r ∧
      |(r : ℚ)/2^149-1/positiveNormalValue (e*2^23+f)| ≤ 1/2^149 := by
  obtain ⟨eo,fo,heo,hfo,hcore⟩ := rcp_high_core_form e f he hf hlarge
  let r := FloatLib.Numerics.roundShiftRightEven (2^23+fo) (3-eo)
  have hr := quarter_round_range fo (3-eo) hfo (by omega)
  have hi : Approx.mul (e*2^23+f) 0x3e800000 = (e-2)*2^23+f := by
    have h := mul_power_two_normal_word false e f 125 (by omega) hf (by omega) (by omega)
    simpa [show e+125-127 = e-2 by omega] using h
  have hbig : Mufu.magnitude (e*2^23+f) > 0x7e800000 := by
    dsimp [Mufu.magnitude]
    omega
  have ho : Approx.evaluate "rcp" false (e*2^23+f) = some r := by
    change (let s := if Mufu.magnitude (e*2^23+f) > 0x7e800000 then 0x3e800000
                   else if Mufu.magnitude (e*2^23+f) < 0x00800000 then 0x4b800000 else 0x3f800000
          some (Approx.mul s (Mufu.rcp (Approx.mul (e*2^23+f) s)))) = some r
    dsimp only
    rw [ite_eq_left hbig,hi,hcore]
    simpa [r] using congrArg some (mul_quarter_low_word_left false eo fo heo hfo)
  refine ⟨r,hr,ho,?_⟩
  have hs := high_seed_error e f he hf
  have hround := quarter_low_value_error eo fo heo hfo
  rw [← hcore] at hround
  change |(r : ℚ)/2^149-positiveNormalValue (Mufu.rcp ((e-2)*2^23+f))/4| ≤ _ at hround
  by_cases he253 : e = 253
  · rw [ite_eq_left he253] at hs
    apply quarter_error_exponent253 _ _ _ _ hround
    simpa [div_eq_mul_inv,mul_assoc] using hs
  · rw [ite_eq_right he253] at hs
    have h := quarter_error_exponent254 ((r : ℚ)/2^149)
      (positiveNormalValue (Mufu.rcp ((e-2)*2^23+f)))
      (1/positiveNormalValue (e*2^23+f)) (by simpa [div_eq_mul_inv,mul_assoc] using hs) hround
    calc
      _ ≤ 3/2^151 := h
      _ ≤ _ := by norm_num


theorem high_nonftz_sign (negative : Bool) (e f : Nat)
    (he : 253 ≤ e ∧ e ≤ 254) (hf : f < 2^23) (hlarge : e = 254 ∨ f ≠ 0) :
    Approx.evaluate "rcp" false ((if negative then 2^31 else 0)+e*2^23+f) =
      (Approx.evaluate "rcp" false (e*2^23+f)).map
        (fun r => (if negative then 2^31 else 0)+r) := by
  obtain ⟨eo,fo,heo,hfo,hcore⟩ := rcp_high_core_form e f he hf hlarge
  have adapter (neg : Bool) :
      Approx.evaluate "rcp" false ((if neg then 2^31 else 0)+e*2^23+f) =
        some ((if neg then 2^31 else 0)+FloatLib.Numerics.roundShiftRightEven (2^23+fo) (3-eo)) := by
    let s := if neg then 2^31 else 0
    have hi : Approx.mul (s+e*2^23+f) 0x3e800000 = s+(e-2)*2^23+f := by
      have h := mul_power_two_normal_word neg e f 125 (by omega) hf (by omega) (by omega)
      simpa [s,show e+125-127 = e-2 by omega] using h
    have hbig : Mufu.magnitude (s+e*2^23+f) > 0x7e800000 := by
      dsimp [s,Mufu.magnitude]
      cases neg <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
    have hc : Mufu.rcp (s+(e-2)*2^23+f) = s+eo*2^23+fo := by
      cases neg
      · simpa [s] using hcore
      · dsimp [s]
        have hsign := reciprocal_sign ((e-2)*2^23+f) (by omega)
        rw [hcore] at hsign
        simpa [Nat.add_assoc] using hsign
    change (let scale := if Mufu.magnitude (s+e*2^23+f) > 0x7e800000 then 0x3e800000
                   else if Mufu.magnitude (s+e*2^23+f) < 0x00800000 then 0x4b800000 else 0x3f800000
          some (Approx.mul scale (Mufu.rcp (Approx.mul (s+e*2^23+f) scale)))) = _
    dsimp only
    rw [ite_eq_left hbig,hi,hc]
    simpa [s] using congrArg some (mul_quarter_low_word_left neg eo fo heo hfo)
  rw [adapter negative]
  have hp := adapter false
  simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add] at hp
  rw [hp]
  rfl

theorem high_boundary_adapter (negative : Bool) :
    Approx.evaluate "rcp" false ((if negative then 2^31 else 0)+0x7e800000) =
      some ((if negative then 2^31 else 0)+0x00800000) := by
  let s := if negative then 2^31 else 0
  have hi : Approx.mul (s+0x7e800000) 0x3f800000 = s+0x7e800000 := by
    simpa [s] using (mul_one_normal negative 253 0 (by norm_num) (by norm_num)).1
  have ho : Approx.mul 0x3f800000 (s+0x00800000) = s+0x00800000 := by
    simpa [s] using (mul_one_normal negative 1 0 (by norm_num) (by norm_num)).2
  have hc : Mufu.rcp (s+0x7e800000) = s+0x00800000 := by
    dsimp [s]
    cases negative <;>
      simp only [Mufu.rcp,Mufu.interpolate,Fp64Seed.squareTruncated,Std.Legacy.Range.forIn_eq_forIn_range'] <;>
      decide +kernel
  have hm : Mufu.magnitude (s+0x7e800000) = 0x7e800000 := by
    dsimp [s,Mufu.magnitude]
    cases negative <;> decide
  change (let scale := if Mufu.magnitude (s+0x7e800000) > 0x7e800000 then 0x3e800000
                   else if Mufu.magnitude (s+0x7e800000) < 0x00800000 then 0x4b800000 else 0x3f800000
          some (Approx.mul scale (Mufu.rcp (Approx.mul (s+0x7e800000) scale)))) = _
  dsimp only
  rw [hm]
  norm_num only
  simp only [↓reduceIte]
  rw [hi,hc,ho]
  rfl

theorem high_nonftz_signed_bound (negative : Bool) (e f : Nat)
    (he : 253 ≤ e ∧ e ≤ 254) (hf : f < 2^23) :
    ∃ r : Nat, (0 < r ∧ r ≤ 2^23) ∧
      Approx.evaluate "rcp" false ((if negative then 2^31 else 0)+e*2^23+f) =
        some ((if negative then 2^31 else 0)+r) ∧
      |(if negative then -(r : ℚ)/2^149 else (r : ℚ)/2^149)-
        1/normalValue ((if negative then 2^31 else 0)+e*2^23+f)| ≤ 1/2^149 := by
  by_cases hl : e = 254 ∨ f ≠ 0
  · obtain ⟨r,hr,ho,hb⟩ := high_nonftz_positive_bound e f he hf hl
    refine ⟨r,hr,?_,?_⟩
    · rw [high_nonftz_sign negative e f he hf hl,ho]
      rfl
    · have hv := value_sign (e*2^23+f) (by omega)
      cases negative
      · simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add]
        rw [hv.1]
        exact hb
      · simp only [↓reduceIte]
        rw [Nat.add_assoc,hv.2]
        simp only [div_neg,neg_div,sub_neg_eq_add]
        rw [show -((r : ℚ)/2^149)+1/positiveNormalValue (e*2^23+f) =
          -((r : ℚ)/2^149-1/positiveNormalValue (e*2^23+f)) by ring,abs_neg]
        exact hb
  · have he253 : e = 253 := by omega
    have hf0 : f = 0 := by omega
    subst e
    subst f
    refine ⟨2^23,by norm_num,?_,?_⟩
    · simpa using high_boundary_adapter negative
    · have hv := value_sign (253*2^23) (by norm_num)
      have hx := normal_decode 253 0 (by norm_num) (by norm_num)
      cases negative <;> simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add,Nat.add_zero]
      · rw [hv.1]
        norm_num at hx ⊢
        rw [hx]
        norm_num
      · rw [hv.2]
        norm_num at hx ⊢
        rw [hx]
        norm_num

#print axioms high_nonftz_signed_bound
end GB300Math.RcpPtxBounds
