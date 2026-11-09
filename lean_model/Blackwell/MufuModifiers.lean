import GB300Math.Mufu
import GB300Math.Fp64Seed
import GB300Math.MufuSm103

namespace GB300Math.MufuModifiers

def modify (width modifier input : Nat) : Nat :=
  let word := input % 2 ^ width
  let value := if modifier / 2 % 2 = 1 then word % 2 ^ (width - 1) else word
  if modifier % 2 = 1 then value ^^^ 2 ^ (width - 1) else value

def evaluate (precision op : String) (modifier input : Nat) : Option Nat := do
  if modifier ≥ 4 then none else do
    if precision = "f16" then
      Mufu.evaluate (op ++ ".f16") (modify 16 modifier input)
    else if precision = "bf16" then
      MufuSm103.evaluateBf16 op (modify 16 modifier input)
    else if precision = "f32" then
      Mufu.evaluate op (modify 32 modifier input)
    else if precision = "f64h" then
      if op = "rcp64h" then some (Fp64Seed.highWord false (modify 32 modifier input))
      else if op = "rsq64h" then some (Fp64Seed.highWord true (modify 32 modifier input))
      else none
    else none

theorem absolute_half_ignores_upper : modify 16 2 0x1234bc00 = 0x3c00 := by decide
theorem negative_zero_reciprocal : evaluate "f32" "rcp" 1 0 = some 0xff800000 := by decide
theorem absolute_rsqrt_negative : evaluate "f32" "rsq" 2 0xbf800000 = some 0x3f800000 := by decide
theorem bf16_abs_then_neg : modify 16 3 0x1234bf80 = 0xbf80 := by decide
theorem bf16_absolute_rsqrt : evaluate "bf16" "rsq" 2 0xbf80 = some 0x3f80 := by decide
theorem bf16_negative_zero_reciprocal : evaluate "bf16" "rcp" 1 0 = some 0xff80 := by decide
#print axioms absolute_half_ignores_upper
#print axioms negative_zero_reciprocal
#print axioms absolute_rsqrt_negative
#print axioms bf16_abs_then_neg
#print axioms bf16_absolute_rsqrt
#print axioms bf16_negative_zero_reciprocal

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.MufuModifiers

open GB300Math.MufuModifiers

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, op, modsText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mods := modsText.toNat? | throw (IO.userError "Invalid modifiers")
    let some expected := errorsText.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 8 = 0 ∧ ys.size = xs.size do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 8] do
      let a := readWord xs (8 * i)
      let some model := evaluate precision op mods a |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {op} mods={mods} row={i} a={a} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 8} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
