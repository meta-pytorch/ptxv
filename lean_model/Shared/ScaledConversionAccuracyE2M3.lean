import ScaledConversionAccuracy

set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000000

namespace GB300Math.ScaledConversionAccuracy

private theorem e2m3_modifiers_block0 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e2m3" ⟨2,3,1,31⟩ 0 (0+code.val) scale.val relu sat := by decide +kernel

private theorem e2m3_modifiers_block1 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e2m3" ⟨2,3,1,31⟩ 0 (16+code.val) scale.val relu sat := by decide +kernel

private theorem e2m3_modifiers_block2 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e2m3" ⟨2,3,1,31⟩ 0 (32+code.val) scale.val relu sat := by decide +kernel

private theorem e2m3_modifiers_block3 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e2m3" ⟨2,3,1,31⟩ 0 (48+code.val) scale.val relu sat := by decide +kernel

theorem e2m3_all_modifiers_accuracy : ∀ code : Fin 64, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e2m3" ⟨2,3,1,31⟩ 0 code.val scale.val relu sat := by
  intro code scale relu sat
  have hc := code.isLt
  have hm : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hd : code.val / 16 ≤ 3 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_modifiers_block0 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 16 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_modifiers_block1 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 32 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_modifiers_block2 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 48 + code.val % 16 = code.val := by omega
    simpa only [heq] using e2m3_modifiers_block3 ⟨code.val % 16, hm⟩ scale relu sat

#print axioms e2m3_all_modifiers_accuracy

end GB300Math.ScaledConversionAccuracy
