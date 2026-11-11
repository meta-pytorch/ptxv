import Std

namespace GB300Math.TensorInteger

structure Mode where
  bits : Nat
  m : Nat
  k : Nat
  signedA : Bool
  signedB : Bool
  saturate : Bool
  xorPopc : Bool

def mode (name : String) : Option Mode := do
  let parts := name.splitOn "_"
  unless parts.length = 4 ∨ parts.length = 5 do none
  let a := parts[0]!
  let b := parts[1]!
  let m ← parts[2]!.toNat?
  let k ← parts[3]!.toNat?
  let bits ← (a.drop 1).toString.toNat?
  unless bits ∈ [1,4,8] do none
  unless (m = 8 ∧ k = 128 / bits) ∨ (m = 16 ∧ k ∈ [128 / bits,256 / bits]) do none
  let sat := parts.length = 5
  if sat then unless parts[4]! = "sat" ∧ bits ≠ 1 do none
  if bits = 1 then
    unless a = "b1" ∧ (b = "and" ∨ b = "xor") do none
  else
    unless (a = s!"s{bits}" ∨ a = s!"u{bits}") ∧ (b = s!"s{bits}" ∨ b = s!"u{bits}") do none
  return ⟨bits,m,k,a.startsWith "s",b.startsWith "s",sat,b = "xor"⟩

def signed (x bits : Nat) : Int :=
  let y := x % 2^bits
  if y < 2^(bits-1) then y else (y : Int)-2^bits

def dot (config : Mode) (a b : Array Nat) (c : Nat) : Nat := Id.run do
  let mut total := signed c 32
  for i in [:a.size] do
    let x := a[i]! % 2^config.bits
    let y := b[i]! % 2^config.bits
    if config.bits = 1 then
      total := total + (if config.xorPopc then x ^^^ y else x &&& y : Nat)
    else
      let vx := if config.signedA then signed x config.bits else (x : Int)
      let vy := if config.signedB then signed y config.bits else (y : Int)
      total := total + vx*vy
    if config.saturate ∧ (i+1) % (128/config.bits) = 0 then
      total := max (-(2^31)) (min (2^31-1) total)
  if config.saturate then total := max (-(2^31)) (min (2^31-1) total)
  return (total % 2^32).toNat

theorem signed_nibble : signed 15 4 = -1 := by decide +kernel
theorem mixed_signed_product : dot ⟨4,8,32,true,false,false,false⟩ #[15] #[15] 0 = 0xfffffff1 := by decide +kernel
theorem upper_input_bits_ignored : dot ⟨4,8,32,true,false,false,false⟩ #[0xffffffff] #[0xffffffff] 0 = 0xfffffff1 := by decide +kernel
theorem saturation_within_short_block : dot ⟨8,16,32,true,true,true,false⟩ #[1,255] #[1,1] 0x7fffffff = 0x7fffffff := by decide +kernel
theorem positive_saturation : dot ⟨8,16,32,false,false,true,false⟩ #[1] #[1] 0x7fffffff = 0x7fffffff := by decide +kernel
theorem binary_wraps : dot ⟨1,8,128,false,false,false,true⟩ #[1] #[0] 0x7fffffff = 0x80000000 := by decide +kernel
theorem binary_equal_xor : dot ⟨1,8,128,false,false,false,true⟩ #[1,0,1] #[1,0,1] 0 = 0 := by decide +kernel

theorem long_int8_saturates_between_blocks :
    dot ⟨8,16,32,true,true,true,false⟩
      (#[1] ++ Array.replicate 15 0 ++ #[255] ++ Array.replicate 15 0)
      (Array.replicate 32 1) 0x7fffffff = 0x7ffffffe := by decide +kernel

theorem long_int4_saturates_between_blocks :
    dot ⟨4,16,64,true,true,true,false⟩
      (#[1] ++ Array.replicate 31 0 ++ #[15] ++ Array.replicate 31 0)
      (Array.replicate 64 1) 0x7fffffff = 0x7ffffffe := by decide +kernel

#print axioms long_int8_saturates_between_blocks
#print axioms long_int4_saturates_between_blocks

#print axioms signed_nibble
#print axioms mixed_signed_product
#print axioms upper_input_bits_ignored
#print axioms saturation_within_short_block
#print axioms positive_saturation
#print axioms binary_wraps
#print axioms binary_equal_xor

def word (bytes : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut result := 0
  for j in [:4] do result := result + bytes[offset+j]!.toNat * 2^(8*j)
  return result

end GB300Math.TensorInteger

open GB300Math.TensorInteger

def main (args : List String) : IO UInt32 := do
  let [name,input,output] := args | throw (IO.userError "Expected mode input output")
  let some config := mode name | throw (IO.userError s!"Unsupported mode: {name}")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  let stride := ((config.m+8)*config.k+config.m*8)*4
  let outStride := config.m*8*4
  unless xs.size > 0 ∧ xs.size % stride = 0 ∧ ys.size = xs.size/stride*outStride do
    throw (IO.userError "Invalid full-matrix capture lengths")
  let mut bad := 0
  for sample in [:xs.size/stride] do
    let base := sample*stride
    for row in [:config.m] do
      let a := (Array.range config.k).map (fun j => word xs (base+4*(row*config.k+j)))
      for col in [:8] do
        let b := (Array.range config.k).map (fun j => word xs (base+4*(config.m*config.k+col*config.k+j)))
        let c := word xs (base+4*((config.m+8)*config.k+row*8+col))
        let expected := dot config a b c
        let actual := word ys (sample*outStride+4*(row*8+col))
        if expected ≠ actual then
          if bad < 3 then IO.println s!"Mismatch {name} matrix={sample} row={row} col={col} model={expected} hardware={actual}"
          bad := bad+1
  IO.println s!"result {name} cells={xs.size/stride*config.m*8} mismatches={bad}"
  return if bad = 0 then 0 else 1
