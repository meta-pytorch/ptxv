import GB300Math.TensorBlackwellUnderflow

namespace GB300Math.TensorShift

def sourceRow (steps row : Nat) : Nat :=
  row / 32 * 32 + min 31 (row % 32 + steps)

theorem source_row_bounded (steps row : Nat) (hr : row < 128) :
    sourceRow steps row < 128 := by
  simp only [sourceRow]
  have := Nat.min_le_left 31 (row % 32 + steps)
  omega

theorem last_row_preserved (steps row : Nat) (hr : row % 32 = 31) :
    sourceRow steps row = row := by
  simp only [sourceRow, hr]
  have : min 31 (31 + steps) = 31 := Nat.min_eq_left (by omega)
  rw [this]
  omega

theorem shifts_compose (a b row : Nat) :
    sourceRow a (sourceRow b row) = sourceRow (a + b) row := by
  simp only [sourceRow]
  have h := Nat.mod_lt row (by decide : 0 < 32)
  omega

#print axioms source_row_bounded
#print axioms last_row_preserved
#print axioms shifts_compose

end GB300Math.TensorShift
