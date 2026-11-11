import ScaledSmallFloatLibAccuracy

namespace GB300Math.ScaledSmallAccuracy

theorem e4m3_half_subnormal_ideal :
    ideal "e4m3" ⟨4,3,7,126⟩ 1 2 = some ((2 : ℚ)^(-134 : Int)) := by
  decide +kernel

theorem e4m3_half_subnormal_rounds_even :
    decoded "e4m3" ⟨4,3,7,126⟩ 1 2 = some 0 := by
  decide +kernel

theorem e5m2_half_subnormal_ideal :
    ideal "e5m2" ⟨5,2,15,123⟩ 1 9 = some ((2 : ℚ)^(-134 : Int)) := by
  decide +kernel

theorem e5m2_half_subnormal_rounds_even :
    decoded "e5m2" ⟨5,2,15,123⟩ 1 9 = some 0 := by
  decide +kernel

theorem fp8_bound_is_attained :
    |(0 : ℚ) - (2 : ℚ)^(-134 : Int)| = (2 : ℚ)^(-134 : Int) := by
  decide +kernel

#print axioms e4m3_half_subnormal_ideal
#print axioms e4m3_half_subnormal_rounds_even
#print axioms e5m2_half_subnormal_ideal
#print axioms e5m2_half_subnormal_rounds_even
#print axioms fp8_bound_is_attained

end GB300Math.ScaledSmallAccuracy
