import GB300Math.VulkanDepthFilter

open GB300Math.VulkanDepthFilter

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [dataPath,queryPath,outputPath,modeText,functionText,kindText,dxText,dyText,expectedText] := args |
    throw (IO.userError "Expected data queries output mode function kind dx dy expected")
  let [some mode,some function,some kind,some expected] :=
      [modeText,functionText,kindText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid unsigned integer")
  let [some dx,some dy] := [dxText,dyText].map String.toInt? |
    throw (IO.userError "Invalid signed integer")
  unless mode < 4 && function < 8 && kind < 2 && -8 ≤ dx && dx ≤ 7 && -8 ≤ dy && dy ≤ 7 do
    throw (IO.userError "Invalid descriptor")
  let data ← IO.FS.readBinFile dataPath
  let queries ← IO.FS.readBinFile queryPath
  let output ← IO.FS.readBinFile outputPath
  unless data.size = 1024 && queries.size % 16 = 0 && output.size = queries.size+256 do
    throw (IO.userError "Invalid extent")
  let depths := (Array.range 256).map (word data)
  let mut mismatches := 0
  for i in [:queries.size/16] do
    let x := word queries (4*i)
    let y := word queries (4*i+1)
    unless x/2^23%256 < 255 && y/2^23%256 < 255 do
      throw (IO.userError "Nonfinite coordinate")
    let predicted := sample depths x y mode function (word queries (4*i+2)) kind dx dy
    for component in [:4] do
      if word output (4*i+component) != predicted[component]! then
        if mismatches < 8 then IO.println s!"difference row={i} component={component} predicted={predicted[component]!} actual={word output (4*i+component)}"
        mismatches := mismatches+1
  for i in [:64] do
    if word output (queries.size/4+i) != 0xdeadbeef then mismatches := mismatches+1
  IO.println s!"records={queries.size/4} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
