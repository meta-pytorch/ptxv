import Std

namespace GB300Math.UniformCarry

def base : Nat := 4294967296

def encode (total : Nat) : Nat × Nat :=
  (total % base, (if total ≥ base then 1 else 0) + base * (if total ≥ 2 * base then 1 else 0))

def evaluate (parameter a b c d : Nat) : Option (Nat × Nat) := do
  if parameter ≥ 32 then
    if parameter ≥ 36 || d > 3 then none else do
      let operand := fun (x : Nat) => if parameter ≥ 34 && x % base ≥ 2^31 then ((x % base : Nat) : Int) - base else (x % base : Nat)
      let carry := (d / 2) ^^^ (parameter % 2)
      return (((operand a * operand b + c % base + carry) % base).toNat, 0)
  else
    let complements := parameter % 8
    if d > 3 || parameter ≥ 32 || complements = 3 || complements = 7 then none else do
      let operand := fun (value position : Nat) =>
        let low := value % base
        if complements / 2 ^ position % 2 = 1 then (base - 1) - low else low
      let carries := (d % 4) ^^^ (parameter / 8)
      return encode (operand a 0 + operand b 1 + operand c 2 + carries % 2 + carries / 2)

theorem reconstruct_total (total : Nat) (h : total < 3 * base) :
    total = (encode total).1 + base * ((encode total).2 % base + (encode total).2 / base) := by
  unfold base at h
  by_cases h1 : 4294967296 ≤ total <;> by_cases h2 : 8589934592 ≤ total <;>
    simp [encode, base, h1, h2] <;> omega

theorem carries_are_ordered (total : Nat) :
    (encode total).2 / base ≤ (encode total).2 % base := by
  by_cases h1 : 4294967296 ≤ total <;> by_cases h2 : 8589934592 ≤ total <;>
    simp [encode, base, h1, h2] <;> omega

theorem both_input_carries_add :
    evaluate 0 0 0 0 3 = some (2, 0) := by decide

theorem complements_are_not_negations :
    evaluate 1 0 0 0 0 = some (0xffffffff, 0) := by decide

theorem output_carries_are_thresholds :
    evaluate 0 0xffffffff 0xffffffff 0xffffffff 0 = some (0xfffffffd, 0x100000001) := by decide

theorem inverted_carry_inputs :
    evaluate 24 0 0 0 0 = some (2, 0) := by decide

theorem mad_carry_wrap : evaluate 32 1 1 0xfffffffe 2 = some (0, 0) := by decide
theorem mad_inverted_carry : evaluate 33 0 0 0 0 = some (1, 0) := by decide
#print axioms mad_carry_wrap
#print axioms mad_inverted_carry

#print axioms reconstruct_total
#print axioms carries_are_ordered
#print axioms both_input_carries_add
#print axioms complements_are_not_negations
#print axioms output_carries_are_thresholds
#print axioms inverted_carry_inputs

def readWord (data : ByteArray) (offset count : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:count] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.UniformCarry

open GB300Math.UniformCarry

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [parameter, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some param := parameter.toNat? | throw (IO.userError "Invalid instruction parameter")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 16] do
      let a := readWord xs (16 * i) 4
      let b := readWord xs (16 * i + 4) 4
      let c := readWord xs (16 * i + 8) 4
      let d := readWord xs (16 * i + 12) 4
      let some model := evaluate param a b c d |
        throw (IO.userError s!"Unsupported parameter: {param}")
      let actual := (readWord ys (16 * i) 8, readWord ys (16 * i + 8) 8)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness parameter={param} row={i} a={a} b={b} c={c} d={d} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 16} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
