import GB300Math.StochasticSm103

open GB300Math.StochasticSm103

def readWord (data : ByteArray) (offset bytes : Nat) : Nat := Id.run do
  let mut value := 0
  for j in [:bytes] do value := value + data[offset + j]!.toNat * 2^(8*j)
  return value

def main (args : List String) : IO UInt32 := do
  let [format, relu, finite, input, output] := args
    | throw (IO.userError "Expected format relu satfinite input output")
  let f ← match format with
    | "f16" => pure Format.f16
    | "bf16" => pure Format.bf16
    | "e4m3" => pure Format.e4m3
    | "e5m2" => pure Format.e5m2
    | "e2m3" => pure Format.e2m3
    | "e3m2" => pure Format.e3m2
    | "e2m1" => pure Format.e2m1
    | _ => throw (IO.userError "Unknown stochastic format")
  unless (relu == "0" || relu == "1") && (finite == "0" || finite == "1") do
    throw (IO.userError "Modifiers must be 0 or 1")
  unless isPair f || finite == "1" do
    throw (IO.userError "Small stochastic formats require satfinite")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 32 = 0 && ys.size = xs.size / 4 do
    throw (IO.userError "Expected four uint64 input words per uint64 output")
  let mut mismatches := 0
  for i in [:xs.size / 32] do
    let a := readWord xs (32*i) 8
    let model := packed f (relu == "1") (finite == "1") a
      (readWord xs (32*i+8) 8) (readWord xs (32*i+16) 8)
      (readWord xs (32*i+24) 8) (a / 2^32)
    let actual := readWord ys (8*i) 8
    if model ≠ actual then
      if mismatches < 5 then IO.println s!"witness row={i} model={model} actual={actual}"
      mismatches := mismatches + 1
  IO.println s!"result {format} relu={relu} satfinite={finite} cases={xs.size / 32} lane_results={xs.size / 32 * (if isPair f then 2 else 4)} mismatches={mismatches}"
  return if mismatches = 0 then 0 else 1
