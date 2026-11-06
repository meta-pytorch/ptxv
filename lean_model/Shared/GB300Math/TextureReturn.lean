import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.TextureReturn
open FloatLib.Floats.Formats.BinaryInterchange

def half (bits : Nat) : Nat :=
  let x : Model FloatFormat.binary32 := Model.ofNatBits bits
  if Model.isNaN x then 0x7fff
  else Model.toNatBits (Model.castWithRounding _ FloatFormat.binary16 x .nearestEven)

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem signed_zero : half 0x80000000 = 0x8000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem smallest_subnormal : half 0x33800000 = 1 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem tie_to_zero : half 0x33000000 = 0 := by decide

theorem canonical_nan : half 0xff800001 = 0x7fff := by decide

def word (bytes : ByteArray) (index : Nat) : Nat :=
  (List.range 4).foldl (fun x i => x + bytes[4*index+i]!.toNat * 2^(8*i)) 0

end GB300Math.TextureReturn

#print axioms GB300Math.TextureReturn.signed_zero
#print axioms GB300Math.TextureReturn.smallest_subnormal
#print axioms GB300Math.TextureReturn.tie_to_zero
#print axioms GB300Math.TextureReturn.canonical_nan

def main (args : List String) : IO UInt32 := do
  let [input, output] := args | throw (IO.userError "Expected FP32 input and FP16 output files")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 4 = 0 && xs.size = ys.size do
    throw (IO.userError "Invalid capture sizes")
  let mut failures := 0
  for i in [:xs.size / 4] do
    let x := GB300Math.TextureReturn.word xs i
    let y := GB300Math.TextureReturn.word ys i
    unless y < 65536 do throw (IO.userError "Output exceeds half width")
    let expected := GB300Math.TextureReturn.half x
    if y != expected then
      if failures < 4 then IO.println s!"witness index={i} input={x} expected={expected} actual={y}"
      failures := failures + 1
  IO.println s!"result count={xs.size / 4} mismatches={failures}"
  return if failures = 0 then 0 else 1
