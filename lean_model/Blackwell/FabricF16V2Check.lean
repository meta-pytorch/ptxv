import FabricF16V2

def read16 (data : ByteArray) (index : Nat) : Nat :=
  data[index]!.toNat+256*data[index+1]!.toNat

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [label,mode,teamText,liveText,a,b,c,d,out,expectedText] := row.splitOn "\t" |
      throw (IO.userError "Malformed manifest")
    let some team := teamText.toNat? | throw (IO.userError "Invalid team")
    let some live := liveText.toNat? | throw (IO.userError "Invalid lane count")
    let some expected := expectedText.toNat? | throw (IO.userError "Invalid expected count")
    unless mode ∈ ["rne","sr"] do throw (IO.userError "Invalid rounding mode")
    unless team ∈ [2,4] ∧ live ∈ [1,2,4,8] do throw (IO.userError "Unsupported capture domain")
    let inputs ← [a,b,c,d].toArray.mapM fun path => IO.FS.readBinFile (System.FilePath.mk path)
    let actual ← IO.FS.readBinFile out
    let count := inputs[0]!.size/16
    unless count>0 ∧ inputs.all (fun x=>x.size=count*16) ∧ actual.size=count*16 do
      throw (IO.userError "Invalid input or output size")
    let mut checked := 0
    let mut sentinels := 0
    let mut mismatches := 0
    for index in [:count] do
      let words := (Array.range team).map fun rank =>
        (Array.range 8).map fun lane => read16 inputs[rank]! (index*16+2*lane)
      let predictions := FabricF16V2.predict words (mode="sr")
      for lane in [:8] do
        let observed := read16 actual (index*16+2*lane)
        if lane<live then
          let predicted := predictions[lane]!
          checked := checked+1
          if predicted≠observed then
            if mismatches<8 then IO.println s!"witness {label} row={index} lane={lane} predicted={predicted} actual={observed}"
            mismatches := mismatches+1
        else
          sentinels := sentinels+1
          if observed≠0 then mismatches := mismatches+1
    IO.println s!"result {label} checked={checked} sentinels={sentinels} mismatches={mismatches} expected={expected}"
    if mismatches≠expected then failed := failed+1
  return if failed=0 then 0 else 1
