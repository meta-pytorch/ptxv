import GB300Math.TextureSeamless
import GB300Math.StochasticValue

namespace GB300Math.TextureSeamlessIdeal

open TextureSeamless

def value (bits : Nat) : Rat :=
  (StochasticValue.rationalValue 8 23 127 0x7f7fffff bits).getD 0

theorem corner_basis_value :
    value (weighted #[86,85,85] #[0,1,2] #[0x3f800000,0,0]) = 86/256 := by
  rw [corner_differs_from_thirds]
  decide +kernel

theorem corner_basis_error :
    |value (weighted #[86,85,85] #[0,1,2] #[0x3f800000,0,0]) - 1/3| = 1/384 := by
  rw [corner_basis_value]
  norm_num

end GB300Math.TextureSeamlessIdeal
#print axioms GB300Math.TextureSeamlessIdeal.corner_basis_value
#print axioms GB300Math.TextureSeamlessIdeal.corner_basis_error
