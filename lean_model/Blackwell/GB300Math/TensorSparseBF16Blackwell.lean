/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorLegacyBlackwell

namespace GB300Math.TensorSparseBF16Blackwell
open GB300Math.SparseTensorExtra

def index (code slot : Nat) : Nat :=
  if slot % 2 = 0 then min (code % 4) 2 else max (code / 4) 1

def position (m : Mode) (metadata : Array Nat) (row j : Nat) : Nat :=
  let group := j / 2
  let lane := 4*(row%8)+group/4+m.selector*(m.k/16)
  let nibble := 4*(row/8)+group%4
  let code := metadata[lane]! / 2^(4*nibble) % 16
  group*4+index code j

def evaluate (m : Mode) (a b c e : Array Nat) : Option (Array Nat) := do
  if m.a != "bf16" || m.b != "bf16" || m.acc != "f32" ||
      !validInput m a b c e then none else
    let mut out := #[]
    for row in [:16] do
      let av := (Array.range (m.k/2)).map fun j => a[row*(m.k/2)+j]!
      for col in [:8] do
        let bv := (Array.range (m.k/2)).map fun j => b[col*m.k+position m e row j]!
        out := out.push (GB300Math.TensorLegacyBlackwell.dot "bf16" "f32" av bv c[row*8+col]!)
    return out

theorem selected_index_bound (code slot : Nat) (h : code < 16) : index code slot < 4 := by
  simp only [index]
  split <;> omega

theorem ordered_encodings_agree :
    (#[4,8,9,12,13,14].all fun code =>
      index code 0 == code % 4 && index code 1 == code / 4) = true := by decide +kernel

theorem legacy_basis_discrepancy :
    index 1 0 = 1 ∧ index 1 1 = 1 ∧ index 3 0 = 2 ∧ index 11 1 = 2 := by decide +kernel

#print axioms selected_index_bound
#print axioms ordered_encodings_agree
#print axioms legacy_basis_discrepancy
end GB300Math.TensorSparseBF16Blackwell
