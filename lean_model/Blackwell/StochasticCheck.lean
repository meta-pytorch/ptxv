import GB300Math.Stochastic

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

def main (args : List String) : IO UInt32 := do
  let [input, f16, bf16] := args | throw (IO.userError "Expected input fp16-output bf16-output")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile f16
  let zs ← IO.FS.readBinFile bf16
  unless ys.size > 0 ∧ ys.size % 8 = 0 ∧ zs.size = ys.size ∧ xs.size ≥ ys.size * 4 do
    throw (IO.userError "Invalid capture lengths")
  for i in [:ys.size / 8] do
    let x := readWord xs (32 * i)
    let y := readWord xs (32 * i + 8)
    for (bf, outputs) in [(false, ys), (true, zs)] do
      let expected := GB300Math.Stochastic.pair bf x y (x / 2 ^ 32)
      let actual := readWord outputs (8 * i)
      unless expected = actual do
        throw (IO.userError s!"Mismatch: row={i} bf16={bf} model={expected} hardware={actual}")
  IO.println s!"Both stochastic pair models match {ys.size / 8} records each"
  return 0
