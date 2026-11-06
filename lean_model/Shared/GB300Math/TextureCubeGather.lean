import Std
namespace GB300Math.TextureCubeGather

def word (bytes : ByteArray) (index : Nat) : Nat :=
  (List.range 4).foldl (fun v i => v + bytes[index*4+i]!.toNat * 2^(8*i)) 0

def dyadic (bits : Nat) : Bool :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  e ≤ 127 && m % 2^(142-max e 1) = 0

def coordinate (bits : Nat) : Int :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  let v : Nat := m / 2^(142-max e 1)
  if bits / 2^31 = 1 then -(v : Int) else v

def face (v : Array Int) : Nat :=
  let axis := if v[2]!.natAbs = 256 then 2 else if v[1]!.natAbs = 256 then 1 else 0
  2*axis + if v[axis]! < 0 then 1 else 0

def project (v : Array Int) : Int × Int :=
  match face v with
  | 0 => (-v[2]!, -v[1]!)
  | 1 => (v[2]!, -v[1]!)
  | 2 => (v[0]!, v[2]!)
  | 3 => (v[0]!, -v[2]!)
  | 4 => (v[0]!, -v[1]!)
  | _ => (-v[0]!, -v[1]!)

def index (a : Int) (delta : Nat) : Nat :=
  min 7 (((a+256)*8-256)/512+(delta : Int)).toNat

theorem z_tie : face #[256,256,256] = 4 := by decide
theorem y_tie : face #[256,-256,0] = 3 := by decide
theorem edge_clamps : index (-256) 0 = 0 ∧ index 256 1 = 7 := by decide
#print axioms z_tie
#print axioms y_tie
#print axioms edge_clamps
end GB300Math.TextureCubeGather
open GB300Math.TextureCubeGather

def main (args : List String) : IO UInt32 := do
  let [layered, component, input, texels, output] := args |
    throw (IO.userError "Expected layered component inputs texels outputs")
  let some layered := layered.toNat? | throw (IO.userError "Invalid layered flag")
  let some component := component.toNat? | throw (IO.userError "Invalid component")
  unless layered < 2 && component < 4 do throw (IO.userError "Unsupported configuration")
  let xs ← IO.FS.readBinFile input
  let ts ← IO.FS.readBinFile texels
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 16 = 0 && ys.size = xs.size*2 && ts.size = 8*8*6*(layered+1)*16 do
    throw (IO.userError "Invalid capture extent")
  let mut failures := 0
  for i in [:xs.size/16] do
    let raw := (List.range 3).toArray.map (fun j => word xs (i*4+j))
    let v := raw.map coordinate
    unless raw.all dyadic && v.all (fun x => x.natAbs ≤ 256) && v.any (fun x => x.natAbs = 256) do
      throw (IO.userError "Expected exact normalized dyadic direction")
    let layer := word xs (i*4+3)
    unless layer ≤ layered do throw (IO.userError "Invalid cube layer")
    let (a,b) := project v
    for lane in [:8] do
      let expected := if lane ≥ 4 then if lane = 4 then 1 else 0 else
        let x := index a (if lane ∈ [1,2] then 1 else 0)
        let y := index b (if lane ∈ [0,1] then 1 else 0)
        word ts ((x+8*(y+8*(face v+layer*6)))*4+component)
      let actual := word ys (i*8+lane)
      if expected != actual then
        if failures < 5 then IO.println s!"witness row={i} lane={lane} expected={expected} actual={actual}"
        failures := failures+1
  IO.println s!"result count={xs.size/2} mismatches={failures}"
  return if failures = 0 then 0 else 1
