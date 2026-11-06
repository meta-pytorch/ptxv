import Std

namespace GB300Math.IntegerCarry

def base : Nat := 4294967296

def wrap (bits : Nat) (x : Int) : Nat := (x % (2 ^ bits : Nat)).toNat

def signed32 (x : Nat) : Int := if x < 2 ^ 31 then x else (x : Int) - base

def evaluate (op : String) (signed : Bool) (shift a b c d : Nat) : Option (Nat × Nat) := do
  if a ≥ 2 ^ 64 || b ≥ 2 ^ 64 || c ≥ 2 ^ 64 || d > 1 then none else do
    let al := a % base
    let ah := a / base
    let bl := b % base
    let bh := b / base
    let cl := c % base
    let ch := c / base
    if op = "add" then
      let low := al + bl + d
      let high := ah + bh + low / base
      return (low % base + high % base * base, low / base + high / base * base)
    else if op = "sub" then
      let low : Int := (al : Int) - bl - d
      let borrow := if low < 0 then 1 else 0
      let high : Int := (ah : Int) - bh - borrow
      return (wrap 32 low + wrap 32 high * base,
        (if low < 0 then base - 1 else 0) + (if high < 0 then base - 1 else 0) * base)
    else if op = "lea64" then
      if shift ≥ 32 then none else return ((a * 2 ^ shift + b) % 2 ^ 64, 0)
    else if op = "lea" then
      if shift ≥ 32 then none else do
        let value : Int := if signed then signed32 al else al
        let shifted := wrap 64 (value * 2 ^ shift)
        let carry := (shifted % base + bl) / base
        return ((shifted + b) % 2 ^ 64, carry)
    else
      let x : Int := if signed then signed32 al else al
      let y : Int := if signed then signed32 bl else bl
      let product := x * y
      let lowProduct := wrap 32 product
      let highProduct := wrap 32 (product / base)
      match op with
      | "mad_chain" =>
        let low := lowProduct + cl + d
        let high := highProduct + ch + low / base
        return (low % base + high % base * base, low / base + high / base * base)
      | "mad_low" => return ((lowProduct + cl + d) % base, 0)
      | "mad_high" => return ((highProduct + cl + d) % base, 0)
      | "mad_wide" => return (wrap 64 (product + c), 0)
      | _ => none

theorem add_carries_both_limbs :
    evaluate "add" false 0 0xffffffffffffffff 0 0 1 = some (0, 0x100000001) := by decide

theorem subtraction_borrows_both_limbs :
    evaluate "sub" false 0 0 0 0 1 = some (0xffffffffffffffff, 0xffffffffffffffff) := by decide

theorem signed_address_wraps :
    evaluate "lea" true 4 0xffffffff 15 0 0 = some (0xffffffffffffffff, 0) := by decide

theorem multiply_carry_crosses_low_limb :
    evaluate "mad_chain" false 0 0xffffffff 0xffffffff 0xffffffffffffffff 1 =
      some (0xfffffffe00000001, 0x100000001) := by decide

theorem wide_address_uses_high_word :
    evaluate "lea64" false 4 0x100000001 1 0 0 = some (0x1000000011, 0) := by decide

#print axioms wide_address_uses_high_word
#print axioms add_carries_both_limbs
#print axioms subtraction_borrows_both_limbs
#print axioms signed_address_wraps
#print axioms multiply_carry_crosses_low_limb

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.IntegerCarry

open GB300Math.IntegerCarry

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [op, mode, parameter, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    unless mode = "0" || mode = "1" do throw (IO.userError "Invalid signedness")
    let some shift := parameter.toNat? | throw (IO.userError "Invalid shift")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 2 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let d := readWord xs (32 * i + 24)
      let some model := evaluate op (mode = "1") shift a b c d |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := (readWord ys (16 * i), readWord ys (16 * i + 8))
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {op} signed={mode} shift={shift} row={i} a={a} b={b} c={c} d={d} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
