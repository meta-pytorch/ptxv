import AlternateFloatMagnitudeBounds
set_option maxHeartbeats 0
set_option maxRecDepth 100000
set_option exponentiation.threshold 4096
namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

theorem encodeFinite_lt256 (format : String) (sat s : Bool) (m : Nat) :
    encodeFinite format sat s m < 256 := by
  unfold encodeFinite
  dsimp only
  split_ifs <;> omega

theorem e4m3_canonical_decode : ∀ b : Fin 256,
    decode "e4m3" (if b.val % 128 = 127 then 127 else b.val) = decode "e4m3" b.val := by decide +kernel

theorem e4m3_encode_decode (s : Bool) (m : Nat) :
    decode "e4m3" (encode "e4m3" false (.finite s m)) =
      decode "e4m3" (encodeFinite "e4m3" false s m) := by
  have h := e4m3_canonical_decode ⟨encodeFinite "e4m3" false s m, encodeFinite_lt256 _ _ _ _⟩
  by_cases hn : encodeFinite "e4m3" false s m % 128 = 127
  · simpa [encode, encodeRaw, hn] using h
  · simpa [encode, encodeRaw, hn] using h

theorem signed_abs_magnitude (s : Bool) (m : Nat) :
    |(if s then -1 else 1)*(m : ℚ)/unit| = (m : ℚ)/unit := by
  have hm : (0 : ℚ) ≤ (m : ℚ)/unit := by positivity
  cases s <;> simp [neg_div, abs_of_nonneg hm]

theorem signed_abs_difference (s : Bool) (m n : Nat) :
    |(if s then -1 else 1)*(n : ℚ)/unit - (if s then -1 else 1)*(m : ℚ)/unit| =
      |(n : ℚ)/unit-(m : ℚ)/unit| := by
  cases s
  · simp
  · simp only [ite_true, neg_one_mul, neg_div]
    rw [neg_sub_neg, abs_sub_comm]

theorem e4m3_value_rounding_bound (s : Bool) (m : Nat) (hmax : m ≤ 448*unit) :
    ∃ y : ℚ, rational (decode "e4m3" (encode "e4m3" false (.finite s m))) = some y ∧
      |y - (if s then -1 else 1)*(m : ℚ)/unit| ≤
        max (|(if s then -1 else 1)*(m : ℚ)/unit|/17) (1/1024) := by
  obtain ⟨n,hd,herr⟩ := e4m3_encode_tight_bound m s hmax
  refine ⟨(if s then -1 else 1)*(n : ℚ)/unit, ?_, ?_⟩
  · rw [e4m3_encode_decode, hd]; rfl
  · rw [signed_abs_magnitude, signed_abs_difference]; exact herr

theorem e5m2_value_rounding_bound (s : Bool) (m : Nat) (hmax : m ≤ 57344*unit) :
    ∃ y : ℚ, rational (decode "e5m2" (encode "e5m2" false (.finite s m))) = some y ∧
      |y - (if s then -1 else 1)*(m : ℚ)/unit| ≤
        max (|(if s then -1 else 1)*(m : ℚ)/unit|/9) (1/131072) := by
  obtain ⟨n,hd,herr⟩ := e5m2_encode_tight_bound m s hmax
  refine ⟨(if s then -1 else 1)*(n : ℚ)/unit, ?_, ?_⟩
  · simpa [encode, encodeRaw, hd, rational]
  · rw [signed_abs_magnitude, signed_abs_difference]; exact herr

theorem e4m3_fma_rounding_bound (sa sb sc : Bool) (a b c : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0)
    (s : Bool) (m : Nat)
    (hv : add (mul (.finite sa a) (.finite sb b)) (.finite sc c) = .finite s m)
    (hmax : m ≤ 448*unit) :
    let ideal : ℚ := ((if sa then -1 else 1)*(a : ℚ)/unit)*
      ((if sb then -1 else 1)*(b : ℚ)/unit)+(if sc then -1 else 1)*(c : ℚ)/unit
    ∃ y : ℚ, rational (decode "e4m3" (encode "e4m3" false
      (add (mul (.finite sa a) (.finite sb b)) (.finite sc c)))) = some y ∧
      |y-ideal| ≤ max (|ideal|/17) (1/1024) := by
  have hex := finite_fma_exact sa sb sc a b c ha hb
  rw [hv] at hex
  have hi := Option.some.inj hex
  obtain ⟨y,hy,herr⟩ := e4m3_value_rounding_bound s m hmax
  refine ⟨y, ?_, ?_⟩
  · rwa [hv]
  · simpa only [hi] using herr

theorem e5m2_fma_rounding_bound (sa sb sc : Bool) (a b c : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0)
    (s : Bool) (m : Nat)
    (hv : add (mul (.finite sa a) (.finite sb b)) (.finite sc c) = .finite s m)
    (hmax : m ≤ 57344*unit) :
    let ideal : ℚ := ((if sa then -1 else 1)*(a : ℚ)/unit)*
      ((if sb then -1 else 1)*(b : ℚ)/unit)+(if sc then -1 else 1)*(c : ℚ)/unit
    ∃ y : ℚ, rational (decode "e5m2" (encode "e5m2" false
      (add (mul (.finite sa a) (.finite sb b)) (.finite sc c)))) = some y ∧
      |y-ideal| ≤ max (|ideal|/9) (1/131072) := by
  have hex := finite_fma_exact sa sb sc a b c ha hb
  rw [hv] at hex
  have hi := Option.some.inj hex
  obtain ⟨y,hy,herr⟩ := e5m2_value_rounding_bound s m hmax
  refine ⟨y, ?_, ?_⟩
  · rwa [hv]
  · simpa only [hi] using herr

theorem e4m3_fma_ideal_bound (sa sb sc : Bool) (a b c : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0)
    (hmax : |((if sa then -1 else 1)*(a : ℚ)/unit)*
      ((if sb then -1 else 1)*(b : ℚ)/unit)+(if sc then -1 else 1)*(c : ℚ)/unit| ≤ 448) :
    let ideal : ℚ := ((if sa then -1 else 1)*(a : ℚ)/unit)*
      ((if sb then -1 else 1)*(b : ℚ)/unit)+(if sc then -1 else 1)*(c : ℚ)/unit
    ∃ y : ℚ, rational (decode "e4m3" (encode "e4m3" false
      (add (mul (.finite sa a) (.finite sb b)) (.finite sc c)))) = some y ∧
      |y-ideal| ≤ max (|ideal|/17) (1/1024) := by
  let z := signed (sa != sb) (a*b/unit)+signed sc c
  let s : Bool := if z=0 then (sa != sb) && sc else z<0
  have hv : add (mul (.finite sa a) (.finite sb b)) (.finite sc c) = .finite s z.natAbs := rfl
  have hex := finite_fma_exact sa sb sc a b c ha hb
  rw [hv] at hex
  have hi := Option.some.inj hex
  have hu : (0 : ℚ)<unit := by norm_num [unit,grid]
  have habs := signed_abs_magnitude s z.natAbs
  rw [hi] at habs
  have hm : (z.natAbs : ℚ)/unit ≤ 448 := by rw [← habs]; exact hmax
  have hmul := (div_le_iff₀ hu).mp hm
  have hbound : z.natAbs ≤ 448*unit := by exact_mod_cast hmul
  exact e4m3_fma_rounding_bound sa sb sc a b c ha hb s z.natAbs hv hbound

theorem e5m2_fma_ideal_bound (sa sb sc : Bool) (a b c : Nat)
    (ha : a % 2^385 = 0) (hb : b % 2^385 = 0)
    (hmax : |((if sa then -1 else 1)*(a : ℚ)/unit)*
      ((if sb then -1 else 1)*(b : ℚ)/unit)+(if sc then -1 else 1)*(c : ℚ)/unit| ≤ 57344) :
    let ideal : ℚ := ((if sa then -1 else 1)*(a : ℚ)/unit)*
      ((if sb then -1 else 1)*(b : ℚ)/unit)+(if sc then -1 else 1)*(c : ℚ)/unit
    ∃ y : ℚ, rational (decode "e5m2" (encode "e5m2" false
      (add (mul (.finite sa a) (.finite sb b)) (.finite sc c)))) = some y ∧
      |y-ideal| ≤ max (|ideal|/9) (1/131072) := by
  let z := signed (sa != sb) (a*b/unit)+signed sc c
  let s : Bool := if z=0 then (sa != sb) && sc else z<0
  have hv : add (mul (.finite sa a) (.finite sb b)) (.finite sc c) = .finite s z.natAbs := rfl
  have hex := finite_fma_exact sa sb sc a b c ha hb
  rw [hv] at hex
  have hi := Option.some.inj hex
  have hu : (0 : ℚ)<unit := by norm_num [unit,grid]
  have habs := signed_abs_magnitude s z.natAbs
  rw [hi] at habs
  have hm : (z.natAbs : ℚ)/unit ≤ 57344 := by rw [← habs]; exact hmax
  have hmul := (div_le_iff₀ hu).mp hm
  have hbound : z.natAbs ≤ 57344*unit := by exact_mod_cast hmul
  exact e5m2_fma_rounding_bound sa sb sc a b c ha hb s z.natAbs hv hbound

#print axioms encodeFinite_lt256
#print axioms e4m3_canonical_decode
#print axioms e4m3_encode_decode
#print axioms signed_abs_magnitude
#print axioms signed_abs_difference
#print axioms e4m3_value_rounding_bound
#print axioms e5m2_value_rounding_bound
#print axioms e4m3_fma_rounding_bound
#print axioms e5m2_fma_rounding_bound
#print axioms e4m3_fma_ideal_bound
#print axioms e5m2_fma_ideal_bound
end GB300Math.AlternateFloatFloatLib
