import GB300Math.TextureDepthGeometry

open GB300Math.TextureDepthGeometry

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [inputPath,outputPath,geometryText,functionText,expectedText] := args |
    throw (IO.userError "Expected inputs outputs geometry function expected_mismatches")
  let [some geometry,some function,some expected] :=
      [geometryText,functionText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric argument")
  unless function < 8 && geometry < 5 do throw (IO.userError "Unsupported descriptor")
  let inputs ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  unless inputs.size % 32 = 0 && output.size = (inputs.size/32+64)*4 do
    throw (IO.userError "Invalid extent")
  let count := inputs.size/32
  let mut mismatches := 0
  for index in [:count+64] do
    let predicted ← if index < count then do
      let some depth := payload geometry (readWord inputs (index*8))
          (readWord inputs (index*8+1)) (readWord inputs (index*8+2))
          (readWord inputs (index*8+3)) |
        throw (IO.userError "Input outside the stated geometry fixture")
      pure (result function (readWord inputs (index*8+4)) depth)
    else pure 0xdeadbeef
    let observed := readWord output index
    if predicted != observed then
      if mismatches < 8 then
        IO.println s!"difference index={index} predicted={predicted} observed={observed}"
      mismatches := mismatches+1
  IO.println s!"records={count} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
