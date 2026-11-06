import MufuRcp32Scaling
namespace GB300Math.MufuRcp32Bounds

theorem maximum_mantissa_iff_zero (f : Fin (2^23)) :
    Mufu.interpolate MufuTables.rcp f.val = 2^24 ↔ f.val = 0 := by
  constructor
  · intro hr
    have hi := (normalized_product_bound f).2
    rw [hr] at hi
    have hf : f.val ≤ 1 := by nlinarith
    by_cases hz : f.val = 0
    · exact hz
    · have hone : f.val = 1 := by omega
      have hn : Mufu.interpolate MufuTables.rcp 1 ≠ 2^24 := by decide +kernel
      rw [hone] at hr
      exact False.elim (hn hr)
  · intro hz
    rw [hz]
    exact zero_interpolation

theorem maximum_nat_mantissa_iff_zero (f : Fin (2^23)) :
    (Mufu.interpolate MufuTables.rcp f.val).toNat = 2^24 ↔ f.val = 0 := by
  have hr := mantissa_range f
  have hn : 0 ≤ Mufu.interpolate MufuTables.rcp f.val := le_trans (by norm_num) hr.1
  have hc := Int.toNat_of_nonneg hn
  constructor
  · intro h
    apply (maximum_mantissa_iff_zero f).mp
    omega
  · intro h
    have hm := (maximum_mantissa_iff_zero f).mpr h
    rw [hm]
    decide

theorem penultimate_normal_exponent (f : Fin (2^23)) :
    Mufu.rcp (253*2^23+f.val) = if f.val = 0 then 0x00800000 else 0 := by
  have hf := f.isLt
  have hx : Mufu.exponent (253*2^23+f.val) = 253 := by unfold Mufu.exponent;omega
  have hm : ¬Mufu.magnitude (253*2^23+f.val) > 0x7f800000 := by
    unfold Mufu.magnitude
    rw [Nat.mod_eq_of_lt (show 253*2^23+f.val < 2^31 by omega)]
    omega
  have hs : Mufu.sign (253*2^23+f.val) = 0 := sign_zero_of_lt _ (by omega)
  have hfrac : Mufu.fraction (253*2^23+f.val) = f.val := by unfold Mufu.fraction;omega
  unfold Mufu.rcp
  rw [hx,hs,hfrac]
  simp only [hm, ite_false]
  by_cases hz : f.val = 0
  · norm_num [hz, zero_interpolation]
  · have hh : (Mufu.interpolate MufuTables.rcp f.val).toNat ≠ 2^24 := by
      intro h
      exact hz ((maximum_nat_mantissa_iff_zero f).mp h)
    norm_num at hh
    simp only [hh, ite_false]
    norm_num [hz]

#print axioms maximum_mantissa_iff_zero
#print axioms maximum_nat_mantissa_iff_zero
#print axioms penultimate_normal_exponent
end GB300Math.MufuRcp32Bounds
