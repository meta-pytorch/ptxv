import GB300Math.TensorSparseSmall

namespace GB300Math.TensorSparseSmallLayout

def wordIndex (row j : Nat) : Nat := row * 8 + j / 16
def nibble (metadata : Array Nat) (row j : Nat) : Nat :=
  metadata[wordIndex row j]! / 2^(4*((j/2)%8)) % 16

def dot (afmt bfmt : String) (a b metadata : Array Nat) (row c : Nat) : Option Nat := do
  if a.size != 32 || b.size != 64 || metadata.size != 1024 || row ≥ 128 then none else do
    let indices ← (List.range 32).mapM fun j =>
      TensorSparseSmall.selectedIndex j (nibble metadata row j)
    TensorSmallBlackwell.dot afmt bfmt a (indices.toArray.map fun i => b[i]!) c

theorem word_index_bounded (row j : Nat) (hr : row < 128) (hj : j < 32) :
    wordIndex row j < 1024 := by
  unfold wordIndex
  omega

theorem only_first_two_columns (row j : Nat) (hj : j < 32) :
    wordIndex row j = row*8 ∨ wordIndex row j = row*8+1 := by
  unfold wordIndex
  omega

theorem nibble_bounded (metadata : Array Nat) (row j : Nat) :
    nibble metadata row j < 16 := by
  unfold nibble
  omega

theorem last_group : wordIndex 127 31 = 1017 ∧ 4*((31/2)%8) = 28 := by decide

theorem cutlass_word_column (row group : Nat) (hg : group < 16) :
    ((row*2097152 + 4*group) / 32) % 512 = group / 8 := by omega

theorem cutlass_word_row (row group : Nat) (hg : group < 16) :
    (row*2097152 + 4*group) / 2097152 = row := by omega

theorem cutlass_nibble_shift (row group : Nat) :
    (row*2097152 + 4*group) % 32 = 4*(group%8) := by omega

#print axioms word_index_bounded
#print axioms only_first_two_columns
#print axioms nibble_bounded
#print axioms last_group
#print axioms cutlass_word_column
#print axioms cutlass_word_row
#print axioms cutlass_nibble_shift
end GB300Math.TensorSparseSmallLayout
