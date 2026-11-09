import ScaledConversionAccuracy

set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000000

namespace GB300Math.ScaledConversionAccuracy

private theorem e5m2_modifiers_block0 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (0+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block1 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (16+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block2 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (32+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block3 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (48+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block4 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (64+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block5 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (80+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block6 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (96+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block7 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (112+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block8 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (128+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block9 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (144+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block10 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (160+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block11 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (176+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block12 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (192+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block13 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (208+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block14 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (224+code.val) scale.val relu sat := by decide +kernel

private theorem e5m2_modifiers_block15 : ∀ code : Fin 16, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) (240+code.val) scale.val relu sat := by decide +kernel

theorem e5m2_all_modifiers_accuracy : ∀ code : Fin 256, ∀ scale : Fin 255, ∀ relu sat : Bool,
    accurate "e5m2" ⟨5,2,15,123⟩ ((2 : ℚ)^(-134 : Int)) code.val scale.val relu sat := by
  intro code scale relu sat
  have hc := code.isLt
  have hm : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hd : code.val / 16 ≤ 15 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block0 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 16 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block1 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 32 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block2 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 48 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block3 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 64 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block4 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 80 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block5 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 96 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block6 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 112 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block7 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 128 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block8 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 144 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block9 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 160 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block10 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 176 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block11 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 192 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block12 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 208 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block13 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 224 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block14 ⟨code.val % 16, hm⟩ scale relu sat
  · have heq : 240 + code.val % 16 = code.val := by omega
    simpa only [heq] using e5m2_modifiers_block15 ⟨code.val % 16, hm⟩ scale relu sat

#print axioms e5m2_all_modifiers_accuracy

end GB300Math.ScaledConversionAccuracy
