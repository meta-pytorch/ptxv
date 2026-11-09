import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.F2FPExtra
open FloatLib.Floats.Formats.BinaryInterchange

def lane (fmt : FloatFormat) (mode : Model.IEEERoundingMode) (flags word : Nat) : Nat :=
  let result := Model.castWithRounding FloatFormat.binary32 fmt (Model.ofNatBits word) mode
  let value := Model.toNatBits result
  let magnitude := value % 32768
  let limit := if fmt = FloatFormat.binary16 then 0x7c00 else 0x7f80
  if Model.isNaN result then 0x7fff
  else if flags % 2 = 1 && value >= 32768 then 0
  else if flags / 2 = 1 && magnitude >= limit then value / 32768 * 32768 + limit - 1
  else value

def evaluate (precision mode packing : String) (flags a b c : Nat) : Option Nat := do
  if flags > 3 then none else do
    let fmt ← if precision = "f16" then some FloatFormat.binary16 else if precision = "bf16" then some FloatFormat.bfloat16 else none
    let rounding ← if mode = "rn" then some Model.IEEERoundingMode.nearestEven else if mode = "rz" then some .towardZero else none
    let high ← if packing = "pack" then some (lane fmt rounding flags a) else if packing = "merge" then some (c % 65536) else none
    return lane fmt rounding flags b + high * 65536 + c % 2 ^ 32 * 2 ^ 32

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem merge_uses_low_C : evaluate "f16" "rn" "merge" 0 0 0x3f800000 0x12345678 = some 0x1234567856783c00 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem rectified_negative_zero : lane FloatFormat.binary16 .nearestEven 1 0x80000000 = 0 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem saturation_preserves_nan : lane FloatFormat.bfloat16 .nearestEven 3 0xff800001 = 0x7fff := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem saturates_infinity : lane FloatFormat.binary16 .nearestEven 2 0x7f800000 = 0x7bff := by decide

#print axioms merge_uses_low_C
#print axioms rectified_negative_zero
#print axioms saturation_preserves_nan
#print axioms saturates_infinity

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.F2FPExtra

open GB300Math.F2FPExtra

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, mode, flagsText, packing, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some flags := flagsText.toNat? | throw (IO.userError "Invalid flags")
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
      let some model := evaluate precision mode packing flags a b c |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {mode} flags={flags} packing={packing} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
