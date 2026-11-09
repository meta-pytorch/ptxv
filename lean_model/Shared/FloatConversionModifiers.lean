import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.FloatConversionModifiers
open FloatLib.Floats.Formats.BinaryInterchange

def format (name : String) : Option FloatFormat :=
  match name with
  | "f16" => some FloatFormat.binary16
  | "f32" => some FloatFormat.binary32
  | "f64" => some FloatFormat.binary64
  | _ => none

def rounding (name : String) : Option Model.IEEERoundingMode :=
  match name with
  | "rn" => some .nearestEven
  | "rz" => some .towardZero
  | "ru" => some .towardPositiveInfinity
  | "rd" => some .towardNegativeInfinity
  | _ => none

def modify (width modifier input : Nat) : Nat :=
  let sign := 2^(width-1)
  let bits := input % (2*sign)
  let x := if modifier/2%2=1 then bits%sign else bits
  if modifier%2=1 then x ^^^ sign else x

def evaluate (src dst mode : String) (modifier input : Nat) : Option Nat := do
  let s ← format src
  let d ← format dst
  let r ← rounding mode
  let bits := modify s.bitWidth modifier input
  let value : Model s := Model.ofNatBits bits
  if Model.isNaN value then
    if src = "f64" || dst = "f64" then
      let raw := input % 2^s.bitWidth
      let payload := raw % 2^s.fracWidth
      let converted := if s.fracWidth ≥ d.fracWidth then payload / 2^(s.fracWidth-d.fracWidth)
        else payload * 2^(d.fracWidth-s.fracWidth)
      return raw / 2^(s.bitWidth-1) * 2^(d.bitWidth-1) + (2*d.exponentBias+1) * 2^d.fracWidth +
        (converted ||| 2^(d.fracWidth-1))
    return 2^(d.bitWidth-1)-1
  return Model.toNatBits (Model.castWithRounding s d value r)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem half_negative_zero_to_double : evaluate "f16" "f64" "rn" 1 0 = some 0x8000000000000000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem nan_forwarding_ignores_negation : evaluate "f16" "f64" "rn" 1 0x7c01 = some 0x7ff8040000000000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem direct_double_to_half_rounds_up : evaluate "f64" "f16" "rn" 0 0x3ff0020000000001 = some 0x3c01 := by decide

#print axioms nan_forwarding_ignores_negation
#print axioms direct_double_to_half_rounds_up
#print axioms half_negative_zero_to_double

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.FloatConversionModifiers

open GB300Math.FloatConversionModifiers

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [src, dst, mode, modsText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mods := modsText.toNat? | throw (IO.userError "Invalid modifiers")
    let some expected := errorsText.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let some model := evaluate src dst mode mods a |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {src} {dst} {mode} mods={mods} row={i} a={a} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
