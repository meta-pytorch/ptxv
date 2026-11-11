/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureRatio

namespace GB300Math.TextureBCAlpha

def signedByte (n : Nat) : Int := max (-127) (if n ≥ 128 then (n : Int)-256 else n)

def coefficient (signed eight : Bool) (index : Nat) : Nat :=
  if signed then
    if eight then [0,258,36,72,114,144,186,222][index]!
    else [0,258,48,96,162,210,0,258][index]!
  else
    if eight then [0,257,36,72,113,144,185,221][index]!
    else [0,257,48,96,161,209,0,257][index]!

def reversed (eight : Bool) (index : Nat) : Bool :=
  if eight then index == 1 || index == 4 || index == 6 || index == 7
  else index == 1 || index == 4 || index == 5

def extension (n : Int) : Int := if n ≥ 64 then 1 else if n ≤ -64 then -1 else 0

def scalar (signed : Bool) (aByte bByte index : Nat) : Nat :=
  let a : Int := if signed then signedByte aByte else aByte
  let b : Int := if signed then signedByte bByte else bByte
  let denominator : Nat := if signed then 32767 else 65535
  let value : Int := if a ≤ b && index = 6 then (if signed then -32767 else 0)
    else if a ≤ b && index = 7 then denominator else
      let weight := coefficient signed (a > b) index
      let coefficientSum := if signed then 258 else 257
      (coefficientSum-(weight : Int))*a+weight*b +
        if signed then extension (if reversed (a > b) index then b else a) else 0
  let magnitude := min value.natAbs denominator
  (if value < 0 then 2147483648 else 0) + TextureRatio.ratioBits magnitude denominator

def pixel (signed bc5 : Bool) (low high location component : Nat) : Nat :=
  let block := if component = 0 then low else high
  if component = 0 || (component = 1 && bc5) then
    scalar signed (block%256) (block/256%256) (block/2^(16+3*location)%8)
  else 0

theorem coefficient_bound : ∀ signed eight : Bool, ∀ i : Fin 8,
    coefficient signed eight i.val ≤ if signed then 258 else 257 := by decide

theorem extension_bound (n : Int) : -1 ≤ extension n ∧ extension n ≤ 1 := by
  unfold extension
  split <;> (try split) <;> omega

theorem signed_min_alias : signedByte 128 = signedByte 129 := by decide

theorem minus64_endpoint : scalar true 192 0 0 = 0xbf010302 := by decide

theorem unsigned_equal_zero : ∀ i : Fin 6, scalar false 0 0 i.val = 0 := by decide

theorem signed_inserted_extremes :
    scalar true 0 0 6 = 0xbf800000 ∧ scalar true 0 0 7 = 0x3f800000 := by decide

def word (data : ByteArray) (offset size : Nat) : Nat :=
  (List.range size).foldl (fun n i => n+data[offset+i]!.toNat*2^(8*i)) 0
end GB300Math.TextureBCAlpha

#print axioms GB300Math.TextureBCAlpha.coefficient_bound
#print axioms GB300Math.TextureBCAlpha.extension_bound
#print axioms GB300Math.TextureBCAlpha.signed_min_alias
#print axioms GB300Math.TextureBCAlpha.minus64_endpoint
#print axioms GB300Math.TextureBCAlpha.unsigned_equal_zero
#print axioms GB300Math.TextureBCAlpha.signed_inserted_extremes

def main (args : List String) : IO UInt32 := do
  let [format,input,output] := args | throw (IO.userError "Expected BC4/BC5 format, blocks and output")
  unless format ∈ ["bc4u","bc4s","bc5u","bc5s"] do throw (IO.userError "Invalid BC4/BC5 format")
  let signed := format = "bc4s" || format = "bc5s"
  let bc5 := format = "bc5u" || format = "bc5s"
  let size := if bc5 then 16 else 8
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  let blockEdge := if xs.size = 4096*size then 64 else 256
  let edge := blockEdge*4
  unless xs.size = blockEdge*blockEdge*size && ys.size = edge*edge*16 do
    throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for block in [:blockEdge*blockEdge] do
    let low := GB300Math.TextureBCAlpha.word xs (block*size) 8
    let high := if bc5 then GB300Math.TextureBCAlpha.word xs (block*size+8) 8 else 0
    for location in [:16] do
      let pixel := (block/blockEdge*4+location/4)*edge+block%blockEdge*4+location%4
      for component in [:4] do
        let expected := GB300Math.TextureBCAlpha.pixel signed bc5 low high location component
        let actual := GB300Math.TextureBCAlpha.word ys ((pixel*4+component)*4) 4
        if actual != expected then
          if bad < 4 then IO.println s!"witness block={block} pixel={location} component={component} expected={expected} actual={actual}"
          bad := bad+1
  IO.println s!"result count={edge*edge*4} mismatches={bad}"
  return if bad = 0 then 0 else 1
