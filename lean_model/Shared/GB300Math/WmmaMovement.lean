import Std

namespace GB300Math.WmmaMovement

def stride (m n size layout pad : Nat) : Nat :=
  (if layout = 0 then n else m) + pad * 32 / size

def byte (input : Array Nat) (offset : Nat) : Nat :=
  input[offset / 4]! / 2^(8*(offset % 4)) % 256

def value (m n size a b pad : Nat) (input : Array Nat) (offset : Nat) : Nat :=
  let outStride := stride m n size b pad
  let element := offset / size
  let row := if b = 0 then element / outStride else element % outStride
  let col := if b = 0 then element % outStride else element / outStride
  if row < m ∧ col < n then
    let inStride := stride m n size a pad
    let index := if a = 0 then row*inStride+col else col*inStride+row
    byte input (index*size + offset % size)
  else
    0xdeadbeef / 2^(8*(offset % 4)) % 256

def word (m n size a b pad : Nat) (input : Array Nat) (index : Nat) : Nat :=
  (List.range 4).foldl (fun acc j => acc + value m n size a b pad input (index*4+j)*2^(8*j)) 0

theorem byte_bound (input : Array Nat) (offset : Nat) : byte input offset < 256 := by
  exact Nat.mod_lt _ (by decide)

theorem value_bound (m n size a b pad : Nat) (input : Array Nat) (offset : Nat) :
    value m n size a b pad input offset < 256 := by
  simp only [value]
  split <;> split
  all_goals first | exact byte_bound _ _ | exact Nat.mod_lt _ (by decide)

theorem dense_stride_row (m n size : Nat) : stride m n size 0 0 = n := by
  simp [stride]

theorem dense_stride_col (m n size : Nat) : stride m n size 1 0 = m := by
  simp [stride]

#print axioms byte_bound
#print axioms value_bound
#print axioms dense_stride_row
#print axioms dense_stride_col
end GB300Math.WmmaMovement
