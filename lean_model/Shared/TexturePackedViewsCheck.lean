import GB300Math.TexturePackedViews
open GB300Math.TexturePackedViews

def main (args : List String) : IO UInt32 := do
  let [family, width, height, channels, bits, mode, input, texels, output] := args |
    throw (IO.userError "Expected family width height channels bits mode input texels output")
  let some width := width.toNat? | throw (IO.userError "Invalid width")
  let some height := height.toNat? | throw (IO.userError "Invalid height")
  let some channels := channels.toNat? | throw (IO.userError "Invalid channels")
  let some bits := bits.toNat? | throw (IO.userError "Invalid bits")
  unless width = 64 && height = 64 && family ∈ ["yuy2","y210","y216","ayuv","y410","y416","packed422","packed444"] && channels ∈ [1,2,4] && bits ∈ [8,16] && mode ∈ ["normalized","raw","relation"] do
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
      let value := if lane < channels then readLE ts (((linearIndex (family ∈ ["yuy2","packed422"]) x y)*channels+lane)*(bits/8)) (bits/8) else 0
      let expected := convert bits (mode = "raw") value
      let actual := readLE ys ((i*4+lane)*4) 4
      if expected != actual then
        if failures < 5 then IO.println s!"witness row={i} lane={lane} expected={expected} actual={actual}"
        failures := failures+1
  IO.println s!"result count={xs.size/2} mismatches={failures}"
  return if failures = 0 then 0 else 1
