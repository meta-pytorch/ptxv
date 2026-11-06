import GB300Math.TextureDepthCompare

open GB300Math.TextureDepthCompare

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [depthPath,inputPath,outputPath,functionText,linearText,expectedText] := args |
    throw (IO.userError "Expected depths inputs outputs function linear expected_mismatches")
  let [some function,some linear,some expected] :=
      [functionText,linearText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric argument")
  unless function < 8 && linear < 2 do
    throw (IO.userError "Unsupported compare function/filter")
  let depthBytes ← IO.FS.readBinFile depthPath
  let inputs ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  unless depthBytes.size = 1024 && inputs.size % 16 = 0 &&
      output.size = (inputs.size/16+64)*4 do
    throw (IO.userError "Invalid buffer extent")
  let depths := (List.range 256).toArray.map (readWord depthBytes)
  unless depths.all (fun d => d ≤ 0x3f800000) do
    throw (IO.userError "Depth must be finite nonnegative Float32 in [0,1]")
  let count := inputs.size/16
  let mut mismatches := 0
  for index in [:count+64] do
    let predicted ← if index < count then do
      let some x := coordinate (readWord inputs (index*4)) |
        throw (IO.userError "Coordinate outside the explicit 1/64 grid")
      let some y := coordinate (readWord inputs (index*4+1)) |
        throw (IO.userError "Coordinate outside the explicit 1/64 grid")
      pure (sample depths x y function (readWord inputs (index*4+2)) (linear == 1))
    else pure 0xdeadbeef
    let observed := readWord output index
    if predicted != observed then
      if mismatches < 8 then
        IO.println s!"difference index={index} predicted={predicted} observed={observed}"
      mismatches := mismatches+1
  IO.println s!"records={count} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
