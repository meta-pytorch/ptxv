import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.PackedConversion
open FloatLib.Floats.Formats.BinaryInterchange

def lane (fmt : FloatFormat) (mode : Model.IEEERoundingMode) (word : Nat) : Nat :=
  let result := Model.castWithRounding FloatFormat.binary32 fmt (Model.ofNatBits word) mode
  if Model.isNaN result then 0x7fff else Model.toNatBits result

def evaluate (precision op : String) (flags parameter a b _c : Nat) : Option Nat := do
  if precision != "f32" || flags > 1 || parameter != 0 then none else do
    let fmt ← if op = "f16" then some FloatFormat.binary16 else if op = "bf16" then some FloatFormat.bfloat16 else none
    let mode := if flags = 0 then Model.IEEERoundingMode.nearestEven else .towardZero
    return 65536 * lane fmt mode a + lane fmt mode b

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem pack_source_order : evaluate "f32" "f16" 0 0 0x3f800000 0x40000000 0 = some 0x3c004000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem quiet_canonical_nan : lane FloatFormat.bfloat16 .nearestEven 0xff800001 = 0x7fff := by decide

#print axioms pack_source_order
#print axioms quiet_canonical_nan

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.PackedConversion

open GB300Math.PackedConversion

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, op, flagsText, modsText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some flags := flagsText.toNat? | throw (IO.userError "Invalid flags")
    let some mods := modsText.toNat? | throw (IO.userError "Invalid modifiers")
    let some expected := errorsText.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let some model := evaluate precision op flags mods a b c |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {op} flags={flags} mods={mods} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
