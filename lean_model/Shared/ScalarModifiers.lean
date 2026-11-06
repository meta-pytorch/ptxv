import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.ScalarModifiers

open FloatLib.Floats.Formats.BinaryInterchange

def rounding : String → Option Model.IEEERoundingMode
  | "rn" => some .nearestEven
  | "rz" => some .towardZero
  | "ru" => some .towardPositiveInfinity
  | "rd" => some .towardNegativeInfinity
  | _ => none

def modify (width modifier input : Nat) : Nat :=
  let word := input % 2 ^ width
  let value := if modifier / 2 % 2 = 1 then word % 2 ^ (width - 1) else word
  if modifier % 2 = 1 then value ^^^ 2 ^ (width - 1) else value

def flush (x : Nat) : Nat :=
  if x % 0x80000000 < 0x00800000 then x &&& 0x80000000 else x

def extended : FloatFormat := FloatFormat.ieee 11 23

def arithmetic (fmt : FloatFormat) (op : String) (mode : Model.IEEERoundingMode)
    (a b c : Model fmt) : Option (Model fmt) :=
  match op with
  | "add" => some (Model.addWithRounding mode a b)
  | "mul" => some (Model.mulWithRounding mode a b)
  | "fma" => some (Model.fmaWithRounding mode a b c)
  | _ => none

def evaluate (precision op : String) (mode : Model.IEEERoundingMode)
    (flags modifiers a b c : Nat) : Option Nat := do
  if modifiers ≥ 64 || flags ≥ 4 then none else do
    let fmt ← match precision with
      | "f32" => some FloatFormat.binary32
      | "f64" => some FloatFormat.binary64
      | _ => none
    if precision = "f64" && flags != 0 then none else do
      let rawInputs := if op = "fma" then [b, c, a] else [b, a]
      let a := modify fmt.bitWidth (modifiers % 4) a
      let b := modify fmt.bitWidth (modifiers / 4 % 4) b
      let c := modify fmt.bitWidth (modifiers / 16) c
      let ftz := flags % 2 = 1
      let sat := flags / 2 = 1
      let calculation := if ftz then extended else fmt
      let load := fun x => Model.castWithRounding fmt calculation
        (Model.ofNatBits (if ftz then flush x else x)) .nearestEven
      let result ← arithmetic calculation op mode (load a) (load b) (load c)
      let bits := Model.toNatBits result
      let output := if ftz ∧ bits % 2^34 < (1023 - 126) * 2^23 then
          bits / 2^34 * 2^31
        else Model.toNatBits (Model.castWithRounding calculation fmt result mode)
      if sat then
        if Model.isNaN result ∨ output ≥ 0x80000000 then return 0
        return min output 0x3f800000
      if !Model.isNaN result then return output
      if precision = "f32" then return 0x7fffffff
      match rawInputs.find? (fun x => Model.isNaN (Model.ofNatBits x : Model fmt)) with
      | some nan => return nan ||| 0x0008000000000000
      | none => return 0xfff8000000000000

theorem absolute_negative_zero : modify 32 2 0x80000000 = 0 := by decide

theorem negative_absolute_nan : modify 64 3 0x7ff0000000000001 = 0xfff0000000000001 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem absolute_operand_changes_cancellation :
    evaluate "f32" "add" .nearestEven 0 2 0xbf800000 0x3f800000 0 = some 0x40000000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem saturated_negative_zero :
    evaluate "f32" "mul" .nearestEven 2 1 0 0x3f800000 0 = some 0 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem native_nan_ignores_sign_modifier :
    evaluate "f64" "add" .nearestEven 0 12 0 0x7ff0000000000123 0 =
      some 0x7ff8000000000123 := by decide

#print axioms native_nan_ignores_sign_modifier
#print axioms absolute_negative_zero
#print axioms negative_absolute_nan
#print axioms absolute_operand_changes_cancellation
#print axioms saturated_negative_zero

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.ScalarModifiers

open GB300Math.ScalarModifiers

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, op, modeName, flagsText, modsText, input, output, errorsText] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mode := rounding modeName | throw (IO.userError "Invalid rounding mode")
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
      let some model := evaluate precision op mode flags mods a b c |
        throw (IO.userError s!"Unsupported: {line}")
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {precision} {op} {modeName} flags={flags} mods={mods} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
