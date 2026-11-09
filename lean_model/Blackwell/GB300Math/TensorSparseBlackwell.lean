import GB300Math.TensorBlackwellUnderflow

namespace GB300Math.TensorSparseBlackwell

def metadataAddress (row group : Nat) : Nat × Nat :=
  ((row / 16 * 16 + row % 8 + group / 4 * 8) * 8,
    row / 8 % 2 * 4 + group % 4)

def selectedIndex (tf32 : Bool) (j code : Nat) : Option Nat :=
  if tf32 then
    if code = 4 then some (2*j)
    else if code = 14 then some (2*j+1)
    else none
  else if code = 4 ∨ code = 8 ∨ code = 9 ∨ code = 12 ∨ code = 13 ∨ code = 14 then
    some (j / 2 * 4 + code / 2^(2*(j%2)) % 4)
  else none

def evaluate (fmt : String) (a b metadata : Array Nat) (c row : Nat) : Option Nat := do
  let tf32 := fmt = "tf32"
  let k ← match fmt with
    | "f16" | "bf16" => some 32
    | "tf32" => some 16
    | _ => none
  if a.size != k/2 || b.size != k || metadata.size != 1024 || row ≥ 128 then none
  else do
    let selected ← (List.range (k/2)).mapM fun j => do
      let group := if tf32 then j else j/2
      let (address, nibble) := metadataAddress row group
      let code := metadata[address]! / 2^(4*nibble) % 16
      let index ← selectedIndex tf32 j code
      some b[index]!
    TensorBlackwellUnderflow.evaluate fmt a selected.toArray c true

theorem metadata_address_bounded (row group : Nat) (hr : row < 128) (hg : group < 8) :
    (metadataAddress row group).1 < 1024 ∧ (metadataAddress row group).2 < 8 := by
  simp [metadataAddress]
  omega

theorem selected_index_bounded (tf32 : Bool) (j code index : Nat)
    (hj : j < if tf32 then 8 else 16) (h : selectedIndex tf32 j code = some index) :
    index < if tf32 then 16 else 32 := by
  unfold selectedIndex at h
  split at h <;> split at h <;> simp_all
  all_goals omega

theorem unsupported_metadata_rejected : selectedIndex false 0 6 = none ∧
    selectedIndex true 0 0 = none := by decide

#print axioms metadata_address_bounded
#print axioms selected_index_bounded
#print axioms unsupported_metadata_rejected

end GB300Math.TensorSparseBlackwell
