import ScaledConversionAccuracy

set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000000

namespace GB300Math.ScaledConversionAccuracy

private theorem e2m1_modifiers_block0 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e2m1" ⟨2,1,1,7⟩ 0 (0+code.val) scale.val relu sat := by decide +kernel

theorem e2m1_all_modifiers_accuracy : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e2m1" ⟨2,1,1,7⟩ 0 code.val scale.val relu sat := by
  intro code scale relu sat
  have hc := code.isLt
  have hm : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hd : code.val / 16 ≤ 0 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m1_modifiers_block0 ⟨code.val % 16, hm⟩ scale relu sat

#print axioms e2m1_all_modifiers_accuracy

end GB300Math.ScaledConversionAccuracy
