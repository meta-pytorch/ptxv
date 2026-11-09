import Std

namespace GB300Math.TensorTransposeLayout

def offset (rows row k : Nat) : Nat :=
  row % 8 * 2 + row / 8 * 128 + k % 8 * 16 + k / 8 * rows * 16

theorem matrixA_roundtrip (row k : Nat) (hr : row < 128) :
    offset 128 row k / 128 % 16 * 8 + offset 128 row k / 2 % 8 = row ∧
    offset 128 row k / 2048 * 8 + offset 128 row k / 16 % 8 = k := by
  unfold offset
  omega

theorem matrixB_roundtrip (row k : Nat) (hr : row < 8) :
    offset 8 row k / 2 % 8 = row ∧
    offset 8 row k / 128 * 8 + offset 8 row k / 16 % 8 = k := by
  unfold offset
  omega

theorem matrixA_bounds (row k : Nat) (hr : row < 128) (hk : k < 16) :
    offset 128 row k + 2 ≤ 4096 ∧ offset 128 row k % 2 = 0 := by
  unfold offset
  omega

theorem matrixB_bounds (row k : Nat) (hr : row < 8) (hk : k < 16) :
    offset 8 row k + 2 ≤ 256 ∧ offset 8 row k % 2 = 0 := by
  unfold offset
  omega

#print axioms matrixA_roundtrip
#print axioms matrixB_roundtrip
#print axioms matrixA_bounds
#print axioms matrixB_bounds

end GB300Math.TensorTransposeLayout
