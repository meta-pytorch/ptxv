import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.TensorDoubleZero
open FloatLib.Floats.Formats.BinaryInterchange

def rounding (mode : String) : Model.IEEERoundingMode :=
  match mode with
  | "rz" => .towardZero
  | "rm" => .towardNegativeInfinity
  | "rp" => .towardPositiveInfinity
  | _ => .nearestEven

def fma (mode : String) (a b c : Nat) : Nat :=
  let fmt := FloatFormat.binary64
  let x : Model fmt := Model.ofNatBits a
  let y : Model fmt := Model.ofNatBits b
  let z : Model fmt := Model.ofNatBits c
  let value := Model.fmaWithRounding (rounding mode) x y z
  if !Model.isNaN value then Model.toNatBits value else
  match [y,z,x].find? Model.isNaN with
  | some nan => Model.toNatBits nan ||| 0x0008000000000000
  | none => 0xfff8000000000000

def dot (mode : String) (a b : Array Nat) (c : Nat) : Nat := Id.run do
  let mut word := c
  for i in [:a.size] do word := fma mode a[i]! b[i]! word
  return word

def modify (mode bits : Nat) : Nat :=
  if bits % 2^63 > 0x7ff0000000000000 then bits else
  let value := if mode / 2 % 2 = 1 then bits % 2^63 else bits
  if mode % 2 = 1 then value ^^^ 2^63 else value

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem modifiers_preserve_nan_payload :
    modify 3 0xfff0000000000011 = 0xfff0000000000011 ∧
    modify 2 0xfff0000000000011 = 0xfff0000000000011 := by decide +kernel

theorem accumulator_modified_once :
    dot "rn" #[0x3ff0000000000000,0x3ff0000000000000,0x3ff0000000000000,0x3ff0000000000000]
      (Array.replicate 4 0x3ff0000000000000) (modify 1 0x3ff0000000000000) =
      0x4008000000000000 := by decide +kernel

theorem zero_fragment_directed_sign :
    dot "rm" (Array.replicate 4 (modify 1 0)) (Array.replicate 4 0) 0 = 0x8000000000000000 ∧
    dot "rn" (Array.replicate 4 (modify 1 0)) (Array.replicate 4 0) 0 = 0 := by decide +kernel

theorem zero_fragment_infinity_invalid :
    dot "rn" (Array.replicate 4 0) (Array.replicate 4 0x7ff0000000000000) 0 =
      0xfff8000000000000 := by decide +kernel

#print axioms zero_fragment_directed_sign
#print axioms zero_fragment_infinity_invalid
#print axioms modifiers_preserve_nan_payload
#print axioms accumulator_modified_once

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word
end GB300Math.TensorDoubleZero
open GB300Math.TensorDoubleZero

def main (args : List String) : IO UInt32 := do
  let [mode,mods,zero,input,output,expected] := args | throw (IO.userError "Expected mode inputs outputs expected-mismatches")
  unless mode ∈ ["rn","rz","rm","rp"] do throw (IO.userError "Invalid rounding mode")
  let some mods := mods.toNat? | throw (IO.userError "Invalid operand modifiers")
  let some zero := zero.toNat? | throw (IO.userError "Invalid zero-register mask")
  unless zero > 0 ∧ zero < 8 do throw (IO.userError "Zero-register mask out of domain")
  unless mods < 64 do throw (IO.userError "Operand modifiers out of range")
  let some expected := expected.toNat? | throw (IO.userError "Invalid expected mismatch count")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 1024 = 0 ∧ ys.size = xs.size / 2 do
    throw (IO.userError "Incorrect 8x4 A, 4x8 B, 8x8 C capture extent")
  let mut errors := 0
  for sample in [:xs.size/1024] do
    let base := sample*1024
    for row in [:8] do
      let a := (Array.range 4).map (fun k => modify (mods % 4) (if zero % 2 = 1 then 0 else readWord xs (base+8*(row*4+k))))
      for col in [:8] do
        let b := (Array.range 4).map (fun k => modify (mods / 4 % 4) (if zero / 2 % 2 = 1 then 0 else readWord xs (base+8*(32+col*4+k))))
        let c := modify (mods / 16) (if zero / 4 % 2 = 1 then 0 else readWord xs (base+8*(64+row*8+col)))
        let model := dot mode a b c
        let actual := readWord ys ((sample*64+row*8+col)*8)
        if model ≠ actual then
          if errors < 8 then IO.println s!"witness sample={sample} row={row} col={col} a={a} b={b} c={c} model={model} actual={actual}"
          errors := errors+1
  IO.println s!"result mode={mode} cells={xs.size/1024*64} mismatches={errors} expected={expected}"
  return if errors = expected then 0 else 1
