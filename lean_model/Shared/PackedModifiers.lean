import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.PackedModifiers
open FloatLib.Floats.Formats.BinaryInterchange

def modify (modifier input : Nat) : Nat :=
  let word := input % 65536
  let value := if modifier / 2 % 2 = 1 then word % 32768 else word
  if modifier % 2 = 1 then value ^^^ 32768 else value

def lane (parameter index input : Nat) : Nat :=
  let selection := parameter / 4
  let selected := if selection = 0 then input / 2 ^ (16 * index)
    else if selection = 1 then input else input / 65536
  modify (parameter % 4) selected

def evaluateLane (precision op : String) (flags a b c : Nat) : Option Nat := do
  let fmt ← if precision = "f16" then some FloatFormat.binary16
    else if precision = "bf16" then some FloatFormat.bfloat16 else none
  let fraction := if precision = "f16" then 10 else 7
  let emin := if precision = "f16" then 14 else 126
  let ftz := flags % 2 = 1
  let calculation := if ftz then (if precision = "f16" then FloatFormat.ieee 11 10 else FloatFormat.ieee 11 7) else fmt
  let load := fun x =>
    let raw := if ftz ∧ x % 32768 < 2 ^ fraction then x &&& 32768 else x
    Model.castWithRounding fmt calculation (Model.ofNatBits raw) .nearestEven
  let x := load a; let y := load b; let z := load c
  let result ← match op with
    | "add" => some (Model.addWithRounding .nearestEven x y)
    | "mul" => some (Model.mulWithRounding .nearestEven x y)
    | "fma" | "mma" => some (Model.fmaWithRounding .nearestEven x y z)
    | _ => none
  let bits := Model.toNatBits result
  let output := if ftz ∧ bits % 2^(11+fraction) < (1023 - emin) * 2^fraction then
      bits / 2^(11+fraction) * 32768
    else Model.toNatBits (Model.castWithRounding calculation fmt result .nearestEven)
  if flags / 2 = 1 then
    if Model.isNaN result ∨ output ≥ 32768 then return 0
    return min output (if precision = "f16" then 0x3c00 else 0x3f80)
  if Model.isNaN result then return 0x7fff
  if flags / 2 = 2 ∧ output ≥ 32768 then return 0
  return output

def evaluate (precision op : String) (flags parameter a b c : Nat) : Option Nat := do
  if flags ≥ 6 || parameter ≥ 1728 then none else do
    let low ← evaluateLane precision op flags (lane (parameter%12) 0 a)
      (lane (parameter/12%12) 0 b) (lane (parameter/144) 0 c)
    let high ← evaluateLane precision op flags (lane (parameter%12) 1 a)
      (lane (parameter/12%12) 1 b) (lane (parameter/144) 1 c)
    return low + 65536 * high

theorem broadcast_high : lane 8 0 0x40003c00 = 0x4000 ∧ lane 8 1 0x40003c00 = 0x4000 := by decide
theorem absolute_lane : lane 2 1 0xbc003c00 = 0x3c00 := by decide
set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem saturated_broadcast : evaluate "f16" "add" 2 8 0x40003c00 0 0 = some 0x3c003c00 := by decide
#print axioms broadcast_high
#print axioms absolute_lane
#print axioms saturated_broadcast

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.PackedModifiers

open GB300Math.PackedModifiers

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
