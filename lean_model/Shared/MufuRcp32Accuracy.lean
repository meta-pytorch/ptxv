import MufuRcp32Universal

namespace GB300Math.MufuRcp32Bounds

noncomputable def normalizedInput (f : Nat) : ℝ := (2^23+(f : ℝ))/2^23
noncomputable def normalizedOutput (f : Nat) : ℝ := (Mufu.interpolate MufuTables.rcp f : ℝ)/2^24

theorem relative_eq_product (x y : ℝ) (hx : x ≠ 0) :
    |y-1/x| / |1/x| = |x*y-1| := by
  rw [← abs_div]
  congr 1
  field_simp

theorem normalized_relative_error (f : Fin (2^23)) :
    |normalizedOutput f.val - 1/normalizedInput f.val| / |1/normalizedInput f.val| ≤
      1/(2^23 : ℝ) := by
  have input_positive : 0 < normalizedInput f.val := by
    dsimp [normalizedInput]
    positivity
  rw [relative_eq_product _ _ (ne_of_gt input_positive)]
  have h := normalized_product_bound f
  have low : -(2^24 : ℝ) ≤
      (2^23+(f.val : ℝ))*(Mufu.interpolate MufuTables.rcp f.val : ℝ)-2^47 := by
    exact_mod_cast h.1
  have high : (2^23+(f.val : ℝ))*(Mufu.interpolate MufuTables.rcp f.val : ℝ)-2^47 ≤
      (2^24 : ℝ) := by exact_mod_cast h.2
  have identity : normalizedInput f.val * normalizedOutput f.val-1 =
      ((2^23+(f.val : ℝ))*(Mufu.interpolate MufuTables.rcp f.val : ℝ)-2^47)/2^47 := by
    dsimp [normalizedInput, normalizedOutput]
    ring
  rw [identity, abs_div]
  have habs := abs_le.mpr ⟨low,high⟩
  norm_num at habs ⊢
  linarith

theorem zero_interpolation : Mufu.interpolate MufuTables.rcp 0 = 2^24 := by
  decide +kernel

theorem mantissa_range (f : Fin (2^23)) :
    (2^23 : Int) ≤ Mufu.interpolate MufuTables.rcp f.val ∧
    Mufu.interpolate MufuTables.rcp f.val ≤ 2^24 := by
  have h := normalized_product_bound f
  have hf : (0 : Int) ≤ f.val := by positivity
  have hfmax : (f.val : Int) ≤ 2^23-1 := by have h := f.isLt; omega
  let r := Mufu.interpolate MufuTables.rcp f.val
  constructor
  · by_contra hh
    have hr : r ≤ 2^23-1 := by dsimp [r]; omega
    have hm : 0 ≤ (2^23 : Int)+f.val := by positivity
    have bound := mul_le_mul_of_nonneg_left hr hm
    nlinarith [h.1]
  · by_cases hz : f.val = 0
    · rw [hz, zero_interpolation]
    · have hpos : (1 : Int) ≤ f.val := by exact_mod_cast (Nat.pos_of_ne_zero hz)
      by_contra hh
      have hr : 2^24+1 ≤ r := by dsimp [r]; omega
      have hm : 0 ≤ (2^23 : Int)+f.val := by positivity
      have bound := mul_le_mul_of_nonneg_left hr hm
      nlinarith [h.2]

#print axioms relative_eq_product
#print axioms normalized_relative_error
#print axioms zero_interpolation
#print axioms mantissa_range

end GB300Math.MufuRcp32Bounds
