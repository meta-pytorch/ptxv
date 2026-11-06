import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.Atomic

open FloatLib.Floats.Formats.BinaryInterchange

def extended : FloatFormat := FloatFormat.ieee 11 23

def flush32 (bits : Nat) : Nat :=
  if bits % 0x80000000 < 0x00800000 then bits &&& 0x80000000 else bits

def add (format : FloatFormat) (global : Bool) (a b : Nat) : Nat := Id.run do
  let x : Model format := Model.ofNatBits a
  let y : Model format := Model.ofNatBits b
  if format.bitWidth = 64 ∧ (Model.isNaN x ∨ Model.isNaN y) then
    if global then return if Model.isNaN y then b else a
    return (if Model.isNaN x then a else b) ||| 0x0008000000000000
  let ftz := format.bitWidth = 32 ∧ global
  let workFormat := if ftz then extended else format
  let load := fun v => Model.castWithRounding format workFormat
    (Model.ofNatBits (if ftz then flush32 v else v)) .nearestEven
  let value := Model.addWithRounding .nearestEven (load a) (load b)
  if Model.isNaN value then
    return if format.bitWidth = 64 then 0xfff8000000000000
      else 2^(format.bitWidth-1)-1
  let bits := Model.toNatBits value
  if ftz ∧ bits % 2^34 < (1023-126) * 2^23 then return (bits / 2^34) * 2^31
  return Model.toNatBits (Model.castWithRounding workFormat format value .nearestEven)

def format (precision : String) : Option FloatFormat :=
  match precision.replace "x2" "" with
  | "f32" => some FloatFormat.binary32
  | "f64" => some FloatFormat.binary64
  | "f16" => some FloatFormat.binary16
  | "bf16" => some FloatFormat.bfloat16
  | _ => none

def evaluate (precision : String) (global : Bool) (a b : Nat) : Option Nat := do
  let fmt ← format precision
  if precision.endsWith "x2" then
    return add fmt global (a % 65536) (b % 65536) +
      65536 * add fmt global (a / 65536) (b / 65536)
  return add fmt global a b

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem global_preserves_signaling_nan :
    add FloatFormat.binary64 true 0x3ff0000000000000 0x7ff0000000000001 =
      0x7ff0000000000001 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem global_incoming_nan_has_priority :
    add FloatFormat.binary64 true 0x7ff0000000000001 0xfff0000000000002 =
      0xfff0000000000002 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem shared_quiets_memory_nan_first :
    add FloatFormat.binary64 false 0x7ff0000000000001 0xfff0000000000002 =
      0x7ff8000000000001 := by decide

#print axioms global_preserves_signaling_nan
#print axioms global_incoming_nan_has_priority
#print axioms shared_quiets_memory_nan_first

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word

end GB300Math.Atomic

open GB300Math.Atomic

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected atomic case manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [precision, space, mode, input, output] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed atomic case: {line}")
    unless space = "global" ∨ space = "shared" do throw (IO.userError s!"Invalid space: {space}")
    unless mode ∈ ["return", "reduce", "final"] do throw (IO.userError s!"Invalid mode: {mode}")
    let some fmt := format precision | throw (IO.userError s!"Invalid precision {precision}")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    let stride := if mode = "final" then 32 else 16
    let outStride := if mode = "final" then 8 else 16
    unless xs.size > 0 ∧ xs.size % stride = 0 ∧ ys.size = xs.size / stride * outStride do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / stride] do
      let a := readWord xs (stride*i)
      let b := readWord xs (stride*i+8)
      let some value := evaluate precision (space = "global") a b |
        throw (IO.userError s!"Unsupported case: {line}")
      let actual := readWord ys (outStride*i + outStride-8)
      let oldMatches := mode = "final" ∨ readWord ys (outStride*i) =
        (if mode = "return" then a % 2^(if precision.endsWith "x2" then 32 else fmt.bitWidth) else 0)
      if value ≠ actual ∨ !oldMatches then
        if mismatches < 4 then
          IO.println s!"witness {precision} {space} {mode} row={i} a={a} b={b} model={value} actual={actual} old_matches={decide oldMatches}"
        mismatches := mismatches+1
    IO.println s!"result {precision} {space} {mode} cases={xs.size / stride} mismatches={mismatches}"
    (← IO.getStdout).flush
    failures := failures+mismatches
  return if failures = 0 then 0 else 1
