import GB300Math.TextureRatio

namespace GB300Math.TexturePlanes

def readLE (bytes : ByteArray) (offset count : Nat) : Nat :=
  (List.range count).foldl (fun v i => v + bytes[offset+i]!.toNat * 2^(8*i)) 0

def coordinate (bits : Nat) : Nat :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  m / 2^(150-max e 1)

def convert (bits : Nat) (raw : Bool) (value : Nat) : Nat :=
  if raw then value else TextureRatio.ratioBits value (2^bits-1)

theorem raw_preserves (bits value : Nat) : convert bits true value = value := rfl
theorem normalized_zero (bits : Nat) : convert bits false 0 = 0 := by
  simp [convert, TextureRatio.ratioBits]
theorem normalized_u8_max : convert 8 false 255 = 0x3f800000 := by decide
theorem normalized_u16_max : convert 16 false 65535 = 0x3f800000 := by decide
#print axioms raw_preserves
#print axioms normalized_zero
#print axioms normalized_u8_max
#print axioms normalized_u16_max
end GB300Math.TexturePlanes
open GB300Math.TexturePlanes

def main (args : List String) : IO UInt32 := do
  let [width, height, channels, bits, mode, input, texels, output] := args |
    throw (IO.userError "Expected width height channels bits mode input texels output")
  let some width := width.toNat? | throw (IO.userError "Invalid width")
  let some height := height.toNat? | throw (IO.userError "Invalid height")
  let some channels := channels.toNat? | throw (IO.userError "Invalid channels")
  let some bits := bits.toNat? | throw (IO.userError "Invalid bits")
  unless width > 0 && height > 0 && width ≤ 256 && height ≤ 256 && channels ∈ [1,2] && bits ∈ [8,16] && mode ∈ ["normalized","raw","relation"] do
    throw (IO.userError "Unsupported plane configuration")
  let xs ← IO.FS.readBinFile input
  let ts ← IO.FS.readBinFile texels
  let ys ← IO.FS.readBinFile output
  if mode = "relation" then
    unless xs.size > 0 && xs.size = ys.size do throw (IO.userError "Invalid relation extent")
    let mut failures := 0
    for i in [:xs.size/4] do
      if readLE xs (i*4) 4 != readLE ys (i*4) 4 then failures := failures+1
    IO.println s!"result count={xs.size/4} mismatches={failures}"
    return if failures = 0 then 0 else 1
  unless xs.size > 0 && xs.size % 8 = 0 && ys.size = xs.size*2 && ts.size = width*height*channels*(bits/8) do
    throw (IO.userError "Invalid capture extent")
  let mut failures := 0
  for i in [:xs.size/8] do
    let xb := readLE xs (i*8) 4
    let yb := readLE xs (i*8+4) 4
    let x := coordinate xb
    let y := coordinate yb
    unless xb / 2^31 = 0 && yb / 2^31 = 0 && xb / 2^23 % 256 ≤ 134 && yb / 2^23 % 256 ≤ 134 && x < width && y < height do
      throw (IO.userError "Expected finite in-bounds nonnegative coordinates")
    for lane in [:4] do
      let value := if lane < channels then readLE ts (((y*width+x)*channels+lane)*(bits/8)) (bits/8) else 0
      let expected := convert bits (mode = "raw") value
      let actual := readLE ys ((i*4+lane)*4) 4
      if expected != actual then
        if failures < 5 then IO.println s!"witness row={i} lane={lane} expected={expected} actual={actual}"
        failures := failures+1
  IO.println s!"result count={xs.size/2} mismatches={failures}"
  return if failures = 0 then 0 else 1
