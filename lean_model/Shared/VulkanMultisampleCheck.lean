import GB300Math.VulkanMultisample

open GB300Math.VulkanMultisample

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [dataPath,queryPath,outputPath,wText,hText,lText,sText,expectedText] := args |
    throw (IO.userError "Expected data queries output width height layers samples expected")
  let [some width,some height,some layers,some samples,some expected] :=
      [wText,hText,lText,sText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid integer")
  unless width > 0 && height > 0 && layers > 0 && (samples = 2 || samples = 4 || samples = 8) do
    throw (IO.userError "Invalid dimensions")
  let data ← IO.FS.readBinFile dataPath
  let queries ← IO.FS.readBinFile queryPath
  let output ← IO.FS.readBinFile outputPath
  unless data.size = width*height*layers*samples*16 && queries.size % 16 = 0 &&
      output.size = queries.size+256 do throw (IO.userError "Invalid extent")
  let mut mismatches := 0
  for i in [:queries.size/16] do
    let sample := word queries (4*i)
    let layer := word queries (4*i+1)
    let x := word queries (4*i+2)
    let y := word queries (4*i+3)
    unless sample < samples && layer < layers && x < width && y < height do
      throw (IO.userError "Query out of bounds")
    let index := position width height samples sample layer x y
    for component in [:4] do
      if word data (4*index+component) != word output (4*i+component) then
        mismatches := mismatches+1
  for i in [:64] do
    if word output (queries.size/4+i) != 0xdeadbeef then mismatches := mismatches+1
  IO.println s!"records={queries.size/4} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
