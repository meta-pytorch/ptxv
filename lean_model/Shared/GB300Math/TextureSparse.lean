import Std

namespace GB300Math.TextureSparse

def word (bytes : ByteArray) (index : Nat) : Nat :=
  (List.range 4).foldl (fun v i => v + bytes[index*4+i]!.toNat * 2^(8*i)) 0

def signed (bits : Nat) : Int := if bits < 2^31 then bits else (bits : Int)-2^32

def outside (extent : Nat) (coordinate offset : Int) : Bool :=
  coordinate+offset < 0 || coordinate+offset ≥ extent

end GB300Math.TextureSparse

open GB300Math.TextureSparse

def main (args : List String) : IO UInt32 := do
  let [geometry, ctype, component, input, texels, output] := args |
    throw (IO.userError "Expected geometry, coordinate type, component, inputs, texels, outputs")
  unless geometry ∈ ["2d","3d","a2d"] && ctype ∈ ["s32","f32"] do
    throw (IO.userError "Unsupported sparse configuration")
  let some component := component.toInt? | throw (IO.userError "Invalid component")
  unless component ≥ -1 && component < 4 && (component < 0 || (ctype = "f32" && geometry != "3d")) do
    throw (IO.userError "Unsupported gather")
  let d := if geometry = "3d" then 3 else 1
  let layers := if geometry = "a2d" then 3 else 1
  let xs ← IO.FS.readBinFile input
  let ts ← IO.FS.readBinFile texels
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 32 = 0 && xs.size = ys.size && ts.size = 7*5*d*layers*16 do
    throw (IO.userError "Invalid sparse capture extent")
  let mut failures := 0
  for i in [:xs.size/32] do
    unless word xs (8*i+3) < layers do throw (IO.userError "Layer outside tested domain")
    let mut integerOob := false
    for axis in [:3] do
      let coordinate := word xs (8*i+axis)
      let offset := signed (word xs (8*i+4+axis))
      unless -8 ≤ offset && offset ≤ 7 do throw (IO.userError "Offset outside PTX domain")
      unless ctype = "s32" || coordinate / 2^23 % 256 < 255 do
        throw (IO.userError "Nonfinite coordinate outside this fixture")
      let extent := if axis = 0 then 7 else if axis = 1 then 5 else d
      if axis < 2 || d > 1 then
        integerOob := integerOob || outside extent (signed coordinate) offset
    for lane in [:8] do
      let expected := if lane = 4 && ctype = "s32" && integerOob then 1 else 0
      let actual := word ys (8*i+lane)
      if expected != actual then
        if failures < 5 then IO.println s!"witness row={i} lane={lane} expected={expected} actual={actual}"
        failures := failures+1
  IO.println s!"result count={xs.size/4} mismatches={failures}"
  return if failures = 0 then 0 else 1
