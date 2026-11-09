import GB300Math.Mufu

set_option maxRecDepth 10000

namespace GB300Math.MufuProperties

theorem sqrt_drop_left : Mufu.sqrt 0x3f9fffff = 0x3f8f1bbd := by decide +kernel

theorem sqrt_drop_right : Mufu.sqrt 0x3fa00000 = 0x3f8f1bbc := by decide +kernel

theorem sqrt_adjacent_input_drop :
    0x3f9fffff + 1 = (0x3fa00000 : Nat) ∧
    Mufu.sqrt 0x3f9fffff = Mufu.sqrt 0x3fa00000 + 1 := by
  rw [sqrt_drop_left, sqrt_drop_right]
  decide +kernel

theorem sqrt_not_monotone_on_positive_normals :
    ¬ (∀ a b : Nat, 0x00800000 ≤ a → a ≤ b → b < 0x7f800000 →
      Mufu.sqrt a ≤ Mufu.sqrt b) := by
  intro h
  have hd := h 0x3f9fffff 0x3fa00000 (by decide) (by decide) (by decide)
  rw [sqrt_drop_left, sqrt_drop_right] at hd
  omega

theorem tanh_adjacent_input_drop :
    Mufu.tanh 0x3d7fffff = some 0x3d7faac1 ∧
    Mufu.tanh 0x3d800000 = some 0x3d7faac0 := by decide +kernel

#print axioms sqrt_drop_left
#print axioms sqrt_drop_right
#print axioms sqrt_adjacent_input_drop
#print axioms sqrt_not_monotone_on_positive_normals
#print axioms tanh_adjacent_input_drop

end GB300Math.MufuProperties
