import FabricFP8
import FabricBF16
import FabricBF16V2

def read16 (data : ByteArray) (index : Nat) : Nat :=
  data[index]!.toNat + 256 * data[index + 1]!.toNat

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [label, fmt, mode, layout, teamText, liveText, a, b, c, d, out, expectedText] :=
        row.splitOn "\t" | throw (IO.userError "Malformed manifest")
    let some team := teamText.toNat? | throw (IO.userError "Invalid team")
    let some live := liveText.toNat? | throw (IO.userError "Invalid live byte count")
    let some expected := expectedText.toNat? | throw (IO.userError "Invalid expected count")
    unless fmt ∈ ["e4m3", "e5m2", "bf16", "bf16-flushed"] ∧ mode ∈ ["rne", "sr"] ∧
        layout ∈ ["contiguous", "padded64", "natural32"] ∧ team ∈ [2, 4] ∧
        live ∈ [4, 16] ∧ (layout ≠ "natural32" ∨ live = 4) do
      throw (IO.userError "Unsupported capture domain")
    let inputs ← [a, b, c, d].toArray.mapM fun path =>
      IO.FS.readBinFile (System.FilePath.mk path)
    let actual ← IO.FS.readBinFile out
    let stride := if layout = "contiguous" then 16 else 256
    let count := inputs[0]!.size / stride
    unless count > 0 ∧ inputs.all (fun x => x.size = count * stride) ∧
        actual.size = count * 16 do
      throw (IO.userError "Invalid input or output size")
    let mut checked := 0
    let mut sentinels := 0
    let mut mismatches := 0
    for index in [:count] do
      let offset := if layout = "contiguous" then 0
        else if layout = "padded64" then 64 * (index % 4)
        else 4 * (index % 32)
      let base := 8 * (offset / 8)
      let bf16 := fmt ∈ ["bf16", "bf16-flushed"]
      let size := if bf16 then 2 else 1
      let mut predictions := #[]
      for group in [:if bf16 then 1 else (live + 7) / 8] do
        let words := (Array.range team).map fun rank => (Array.range 8).map fun lane =>
          let address := index * stride + base + group * 8 + size * lane
          if bf16 then read16 inputs[rank]! address else (inputs[rank]!)[address]!.toNat
        let values ← if bf16 then pure (if fmt = "bf16-flushed" then FabricBF16V2.predict words (mode = "sr")
            else FabricBF16Block.predict words (mode = "sr"))
          else (FabricFP8.predict (fmt = "e4m3") words).mapM fun value =>
            match value with
            | some v => pure v
            | none => throw (IO.userError "Valid byte inputs rejected by FP8 model")
        predictions := predictions ++ values
      for lane in [:live / size] do
        let predicted := predictions[(offset % 8) / size + lane]!
        let observed := if bf16 then read16 actual (index * 16 + 2 * lane)
          else actual[index * 16 + lane]!.toNat
        checked := checked + 1
        if predicted ≠ observed then
          if mismatches < 8 then
            IO.println s!"witness {label} row={index} lane={lane} predicted={predicted} actual={observed}"
          mismatches := mismatches + 1
      for byte in [live:16] do
        sentinels := sentinels + 1
        if actual[index * 16 + byte]! ≠ 0 then mismatches := mismatches + 1
    IO.println s!"result {label} checked={checked} sentinels={sentinels} mismatches={mismatches} expected={expected}"
    if mismatches ≠ expected then failed := failed + 1
  return if failed = 0 then 0 else 1
