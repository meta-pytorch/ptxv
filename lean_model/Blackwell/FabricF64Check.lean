import FabricF64

def read64 (data : ByteArray) (index : Nat) : Nat :=
  (List.range 8).foldr (fun j word => data[index+j]!.toNat+256*word) 0

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [label,teamText,a,b,c,d,out,expectedText] := row.splitOn "\t" |
      throw (IO.userError "Malformed manifest")
    let some team := teamText.toNat? | throw (IO.userError "Invalid team")
    let some expected := expectedText.toNat? | throw (IO.userError "Invalid expected count")
    unless team ∈ [2,4] do throw (IO.userError "Unsupported team")
    let inputs ← [a,b,c,d].toArray.mapM fun path => IO.FS.readBinFile (System.FilePath.mk path)
    let actual ← IO.FS.readBinFile out
    let count := inputs[0]!.size/16
    unless count>0 ∧ inputs.all (fun x=>x.size=count*16) ∧ actual.size=count*16 do
      throw (IO.userError "Invalid input or output size")
    let mut mismatches := 0
    for index in [:count] do
      let words := (List.range team).map fun rank => read64 inputs[rank]! (index*16)
      let predicted := FabricF64.predict words
      let observed := read64 actual (index*16)
      if predicted≠observed then
        if mismatches<8 then IO.println s!"witness {label} row={index} predicted={predicted} actual={observed}"
        mismatches := mismatches+1
      if read64 actual (index*16+8)≠0 then mismatches := mismatches+1
    IO.println s!"result {label} checked={count} sentinels={count} mismatches={mismatches} expected={expected}"
    if mismatches≠expected then failed := failed+1
  return if failed=0 then 0 else 1
