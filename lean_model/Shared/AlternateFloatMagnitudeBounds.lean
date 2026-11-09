import AlternateFloatEncodeRoundTrip
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 4096
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem nearestEven_le_of_le (m d p : Nat) (hm : m ≤ p*2^d) :
    nearestEven m d ≤ p := by
  by_cases h : m = p*2^d
  · subst m
    by_cases hd : d=0
    · subst d; simp [nearestEven]
    · simp [nearestEven, hd, Nat.ne_of_lt (Nat.two_pow_pos (d-1))]
  · apply nearestEven_le_of_lt; omega

theorem nearest_relative_normalized (p m d : Nat) (hp : 0<p) (hd : 0<d)
    (hm : p*2^d ≤ m) :
    |((nearestEven m d * 2^d : Nat) : ℚ)/unit - (m : ℚ)/unit| ≤
      ((m : ℚ)/unit)/(2*(p : ℚ)+1) := by
  have h := nearest_normal_relative_bound p m d hp hd hm
  have hu : (0 : ℚ)<unit := by norm_num [unit,grid]
  have hpq : (0 : ℚ)<2*(p : ℚ)+1 := by positivity
  have he : |((nearestEven m d * 2^d : Nat) : ℚ) - m| ≤ (m : ℚ)/(2*(p : ℚ)+1) := by
    apply (le_div_iff₀ hpq).mpr
    nlinarith
  calc
    _ = |((nearestEven m d * 2^d : Nat) : ℚ)-(m : ℚ)| / unit := by rw [← sub_div, abs_div, abs_of_pos hu]
    _ ≤ ((m : ℚ)/(2*(p : ℚ)+1))/unit := (div_le_div_iff_of_pos_right hu).mpr he
    _ = _ := by ring

theorem e4m3_normal_raw_in_range (m : Nat) (hm : m ≠ 0)
    (hk : 506 ≤ m.log2) (hmax : m ≤ 448*unit) :
    (m.log2-506)*8 + nearestEven m (m.log2-3) ≤ 126 := by
  have hlo := Nat.log2_self_le hm
  have hlog : m.log2 ≤ 520 := by
    have hlt : m < 2^521 := by norm_num [unit,grid] at hmax ⊢; omega
    have := (Nat.log2_lt hm).mpr hlt
    omega
  obtain ⟨_,hqh⟩ := rounded_significand_range m 3 hm (by omega)
  norm_num at hqh
  by_cases he : m.log2=520
  · have hq : nearestEven m 517 ≤ 14 := nearestEven_le_of_le m 517 14 (by norm_num [unit,grid] at hmax ⊢; omega)
    simp only [he]; norm_num; omega
  · omega

theorem e5m2_normal_raw_in_range (m : Nat) (hm : m ≠ 0)
    (hk : 498 ≤ m.log2) (hmax : m ≤ 57344*unit) :
    (m.log2-498)*4 + nearestEven m (m.log2-2) ≤ 123 := by
  have hlog : m.log2 ≤ 527 := by
    have hlt : m < 2^528 := by norm_num [unit,grid] at hmax ⊢; omega
    have := (Nat.log2_lt hm).mpr hlt
    omega
  obtain ⟨_,hqh⟩ := rounded_significand_range m 2 hm (by omega)
  norm_num at hqh
  by_cases he : m.log2=527
  · have hq : nearestEven m 525 ≤ 7 := nearestEven_le_of_le m 525 7 (by norm_num [unit,grid] at hmax ⊢; omega)
    simp only [he]; norm_num; omega
  · omega

theorem e4m3_encode_tight_bound (m : Nat) (s : Bool)
    (hmax : m ≤ 448*unit) :
    ∃ n : Nat, decode "e4m3" (encodeFinite "e4m3" false s m) = some (.finite s n) ∧
      |(n : ℚ)/unit-(m : ℚ)/unit| ≤ max (((m : ℚ)/unit)/17) (1/1024) := by
  by_cases hk : m.log2 ≤ 505
  · refine ⟨nearestEven m 503 * 2^503, e4m3_encode_subnormal_roundtrip m s hk, ?_⟩
    have h := nearestEven_rational_half_quantum m 503 (by omega)
    have hc : (2^(503-1) : ℚ)/unit = 1/1024 := by norm_num [unit,grid]
    rw [hc] at h
    exact le_trans h (le_max_right _ _)
  · have hm : m ≠ 0 := by intro hz; subst m; simp at hk
    have hlo := Nat.log2_self_le hm
    have hmgrid : 8*2^(m.log2-3) ≤ m := by
      have hpow : 8*2^(m.log2-3) = 2^m.log2 := by
        calc
          _ = 2^3*2^(m.log2-3) := rfl
          _ = 2^(3+(m.log2-3)) := (Nat.pow_add _ _ _).symm
          _ = _ := by congr 1; omega
      rwa [hpow]
    refine ⟨nearestEven m (m.log2-3)*2^(m.log2-3),
      e4m3_encode_normal_roundtrip m s hm (by omega) (e4m3_normal_raw_in_range m hm (by omega) hmax), ?_⟩
    have h := nearest_relative_normalized 8 m (m.log2-3) (by omega) (by omega) hmgrid
    norm_num at h
    apply le_trans ?_ (le_max_left _ _)
    simpa only [Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat] using h

theorem e5m2_encode_tight_bound (m : Nat) (s : Bool)
    (hmax : m ≤ 57344*unit) :
    ∃ n : Nat, decode "e5m2" (encodeFinite "e5m2" false s m) = some (.finite s n) ∧
      |(n : ℚ)/unit-(m : ℚ)/unit| ≤ max (((m : ℚ)/unit)/9) (1/131072) := by
  by_cases hk : m.log2 ≤ 497
  · refine ⟨nearestEven m 496 * 2^496, e5m2_encode_subnormal_roundtrip m s hk, ?_⟩
    have h := nearestEven_rational_half_quantum m 496 (by omega)
    have hc : (2^(496-1) : ℚ)/unit = 1/131072 := by norm_num [unit,grid]
    rw [hc] at h
    exact le_trans h (le_max_right _ _)
  · have hm : m ≠ 0 := by intro hz; subst m; simp at hk
    have hlo := Nat.log2_self_le hm
    have hmgrid : 4*2^(m.log2-2) ≤ m := by
      have hpow : 4*2^(m.log2-2) = 2^m.log2 := by
        calc
          _ = 2^2*2^(m.log2-2) := rfl
          _ = 2^(2+(m.log2-2)) := (Nat.pow_add _ _ _).symm
          _ = _ := by congr 1; omega
      rwa [hpow]
    refine ⟨nearestEven m (m.log2-2)*2^(m.log2-2),
      e5m2_encode_normal_roundtrip m s hm (by omega) (e5m2_normal_raw_in_range m hm (by omega) hmax), ?_⟩
    have h := nearest_relative_normalized 4 m (m.log2-2) (by omega) (by omega) hmgrid
    norm_num at h
    apply le_trans ?_ (le_max_left _ _)
    simpa only [Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat] using h

#print axioms nearestEven_le_of_le
#print axioms nearest_relative_normalized
#print axioms e4m3_normal_raw_in_range
#print axioms e5m2_normal_raw_in_range
#print axioms e4m3_encode_tight_bound
#print axioms e5m2_encode_tight_bound
end GB300Math.AlternateFloatFloatLib
