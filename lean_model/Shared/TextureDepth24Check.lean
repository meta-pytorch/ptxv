import GB300Math.TextureDepth24

open GB300Math.TextureDepth24

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [codePath,inputPath,outputPath,functionText,expectedText] := args |
    throw (IO.userError "Expected codes inputs outputs function expected_mismatches")
  let [some function,some expected] := [functionText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric argument")
  unless function < 8 do throw (IO.userError "Unsupported comparison")
  let codes ← IO.FS.readBinFile codePath
  unless codes.size = 65536*4 do throw (IO.userError "Invalid code extent")
  let inputs ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  unless inputs.size % 8 = 0 && output.size = (inputs.size/8+64)*4 do
    throw (IO.userError "Invalid extent")
  let count := inputs.size/8
  let mut mismatches := 0
  for index in [:count+64] do
    let predicted ← if index < count then do
      let location := readWord inputs (index*2)
      unless location < 65536 do throw (IO.userError "Invalid texture index")
      let depth := readWord codes location
      unless depth < 2^24 do throw (IO.userError "Invalid D24 code")
      pure (compare function (readWord inputs (index*2+1)) depth)
    else pure 0xcafebabe
    let observed := readWord output index
    if predicted != observed then
      if mismatches < 8 then
        IO.println s!"difference index={index} predicted={predicted} observed={observed}"
      mismatches := mismatches+1
  IO.println s!"records={count} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
