import GB300Math.TextureSeamless
open GB300Math.TextureSeamless

def main (args : List String) : IO UInt32 := do
  let [layers,component,filtering,seamless,input,texels,output] := args |
    throw (IO.userError "Expected layers component filtering seamless input texels output")
  let some layers := layers.toNat? | throw (IO.userError "Invalid layers")
  let some component := component.toInt? | throw (IO.userError "Invalid component")
  let some filtering := filtering.toNat? | throw (IO.userError "Invalid filtering")
  let some seamless := seamless.toNat? | throw (IO.userError "Invalid seamless")
  unless layers > 0 && layers ≤ 256 && component ≥ -1 && component < 4 && filtering < 2 && seamless < 2 do
    throw (IO.userError "Unsupported seamless configuration")
  let xs ← IO.FS.readBinFile input
  let ts ← IO.FS.readBinFile texels
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size%16 = 0 && ys.size = xs.size*2 && ts.size = layers*6*8*8*16 do
    throw (IO.userError "Invalid seamless capture extent")
  let mut failures := 0
  for i in [:xs.size/16] do
    let raw := (Array.range 3).map (fun j => word xs (i*4+j))
    let v := raw.map coordinate
    unless raw.all dyadic && v.all (fun x => x.natAbs ≤ 1024) && v.any (fun x => x.natAbs = 1024) do
      throw (IO.userError "Expected exact normalized dyadic direction")
    let layer := word xs (i*4+3)
    unless layer < layers do throw (IO.userError "Invalid cube layer")
    let f := face v
    let (s,t) := project v
    for lane in [:8] do
      let expected := if lane ≥ 4 then if lane = 4 then 1 else 0 else
        if component ≥ 0 then
          let x := ((s+1024)*8-1024)/2048+(if lane ∈ [1,2] then 1 else 0)
          let y := ((t+1024)*8-1024)/2048+(if lane ∈ [0,1] then 1 else 0)
          gather ts layer f x y component.toNat (seamless = 1)
        else if filtering = 1 then linear ts layer f s t lane (seamless = 1)
        else canonical (fetch ts layer f ((s+1024)*8/2048) ((t+1024)*8/2048) lane)
      let actual := word ys (i*8+lane)
      if expected != actual then
        if failures < 8 then IO.println s!"witness row={i} lane={lane} expected={expected} actual={actual}"
        failures := failures+1
  IO.println s!"result count={xs.size/2} mismatches={failures}"
  return if failures = 0 then 0 else 1
