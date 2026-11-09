import FabricF32

def read32 (data : ByteArray) (index : Nat) : Nat :=
  data[index]!.toNat + 256*data[index+1]!.toNat +
    65536*data[index+2]!.toNat + 16777216*data[index+3]!.toNat

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [label, teamText, vecText, a, b, c, d, output, expectedText] := row.splitOn "\t"
      | throw (IO.userError "Malformed manifest")
    let some team := teamText.toNat? | throw (IO.userError "Invalid team")
    let some vec := vecText.toNat? | throw (IO.userError "Invalid vector")
    let some expected := expectedText.toNat? | throw (IO.userError "Invalid expected count")
    unless team ∈ [2,4] ∧ vec ∈ [1,2,4] do throw (IO.userError "Unsupported domain")
    let inputs ← [a,b,c,d].toArray.mapM fun path => IO.FS.readBinFile (System.FilePath.mk path)
    let actual ← IO.FS.readBinFile output
    let count := inputs[0]!.size/16
    unless count > 0 ∧ inputs.all (fun x => x.size = count*16) ∧ actual.size = count*16 do
      throw (IO.userError "Invalid byte extent")
    let mut checked := 0
    let mut sentinels := 0
    let mut mismatches := 0
    for index in [:count] do
      for lane in [:vec] do
        let words := (List.range team).map fun rank => read32 inputs[rank]! (index*16+4*lane)
        let predicted := FabricF32.predict words
        let observed := read32 actual (index*16+4*lane)
        checked := checked+1
        if predicted ≠ observed then
          if mismatches < 8 then IO.println s!"witness {label} row={index} lane={lane} predicted={predicted} actual={observed} inputs={words}"
          mismatches := mismatches+1
      for byte in [4*vec:16] do
        sentinels := sentinels+1
        if actual[index*16+byte]! ≠ 0 then mismatches := mismatches+1
    IO.println s!"result {label} checked={checked} sentinels={sentinels} mismatches={mismatches} expected={expected}"
    if mismatches ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
