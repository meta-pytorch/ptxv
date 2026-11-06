import Std.Tactic

namespace GB300Math.S2F6

def roundEven (n d : Nat) : Nat :=
  let q := n / d
  let r := n % d
  if 2 * r > d ∨ (2 * r = d ∧ q % 2 = 1) then q + 1 else q

theorem roundEven_error (n d : Nat) (hd : 0 < d) :
    2*n ≤ (2*roundEven n d + 1)*d ∧ 2*roundEven n d*d ≤ 2*n+d := by
  have hmod := Nat.mod_lt n hd
  have hdiv := Nat.mod_add_div n d
  unfold roundEven
  dsimp only
  split <;>
    simp only [Nat.add_mul, Nat.mul_add, Nat.mul_assoc, Nat.mul_comm (n / d) d] <;>
    omega

def fromFloat32 (bits : Nat) : Nat :=
  let x := bits % 2^32
  let negative := x ≥ 2^31
  let exponent := x / 2^23 % 256
  let fraction := x % 2^23
  if exponent = 255 ∧ fraction ≠ 0 then 127
  else
    let magnitude :=
      if exponent = 255 then 128
      else if exponent = 0 then roundEven fraction (2^143)
      else if exponent ≥ 144 then (2^23 + fraction) * 2^(exponent - 144)
      else roundEven (2^23 + fraction) (2^(144 - exponent))
    if negative then (256 - min 128 magnitude) % 256 else min 127 magnitude

def packed (a b : Nat) : Nat := 256 * fromFloat32 a + fromFloat32 b

theorem fromFloat32_range (bits : Nat) : fromFloat32 bits < 256 := by
  unfold fromFloat32
  dsimp only
  split
  · omega
  · split
    · exact Nat.mod_lt _ (by decide)
    · omega

theorem packed_range (a b : Nat) : packed a b < 65536 := by
  have ha := fromFloat32_range a
  have hb := fromFloat32_range b
  unfold packed
  omega

example : fromFloat32 0x7fc00000 = 127 := by decide
example : fromFloat32 0xff800000 = 128 := by decide
example : fromFloat32 0x3f800000 = 64 := by decide
example : fromFloat32 0xbf800000 = 192 := by decide
example : fromFloat32 0x3c000000 = 0 := by decide
example : fromFloat32 0x3cc00000 = 2 := by decide

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2^(8*j)
  return word

end GB300Math.S2F6

open GB300Math.S2F6

def main (args : List String) : IO UInt32 := do
  let [input, output] := args | throw (IO.userError "Expected input and output paths")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
    throw (IO.userError "Invalid S2F6 capture sizes")
  let mut mismatches := 0
  for i in [:xs.size / 32] do
    let a := readWord xs (32*i)
    let b := readWord xs (32*i+8)
    let model := packed a b
    let actual := readWord ys (8*i)
    if model ≠ actual then
      if mismatches < 3 then
        IO.println s!"witness row={i} a={a} b={b} model={model} hardware={actual}"
      mismatches := mismatches + 1
  IO.println s!"result s2f6_rn cases={xs.size / 32} mismatches={mismatches}"
  return if mismatches = 0 then 0 else 1
