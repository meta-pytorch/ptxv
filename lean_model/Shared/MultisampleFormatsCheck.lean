import GB300Math.MultisampleFormats

open GB300Math.MultisampleFormats

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [inputPath,outputPath,channelsText,formatText,expectedText] := args |
    throw (IO.userError "Expected input output channels format expected_mismatches")
  let [some channels,some format,some expected] := [channelsText,formatText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric argument")
  unless (channels = 1 || channels = 4) && format < 3 do throw (IO.userError "Unsupported format")
  let inputs ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  unless inputs.size % 16 = 0 && output.size = inputs.size+256 do throw (IO.userError "Invalid extent")
  let count := inputs.size/4
  let mut mismatches := 0
  for index in [:count+64] do
    let predicted := if index < count then lane channels format (index%4) (readWord inputs index)
      else 0xdeadbeef
    if readWord output index != predicted then mismatches := mismatches+1
  IO.println s!"records={count} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
