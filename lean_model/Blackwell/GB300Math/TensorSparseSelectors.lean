/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorSparseBlackwell

namespace GB300Math.TensorSparseSelectors

def metadataAddress (selector row group : Nat) : Nat × Nat :=
  let (address, nibble) := TensorSparseBlackwell.metadataAddress row group
  (address + selector, nibble)

def selectMetadata (metadata : Array Nat) (selector : Nat) : Array Nat :=
  (List.range 1024).toArray.map fun i => metadata[i / 8 * 8 + selector]!

def evaluate (fmt : String) (a b metadata : Array Nat) (c row selector : Nat) : Option Nat :=
  if selector > 1 || metadata.size != 1024 then none else
    TensorSparseBlackwell.evaluate fmt a b (selectMetadata metadata selector) c row

theorem metadata_address_bounded (selector row group : Nat)
    (hs : selector < 2) (hr : row < 128) (hg : group < 8) :
    (metadataAddress selector row group).1 < 1024 ∧
    (metadataAddress selector row group).2 < 8 := by
  simp [metadataAddress, TensorSparseBlackwell.metadataAddress]
  omega

theorem unsupported_selector_rejected (fmt : String) (a b m : Array Nat) (c r s : Nat)
    (hs : 1 < s) : evaluate fmt a b m c r s = none := by
  simp [evaluate, hs]

#print axioms metadata_address_bounded
#print axioms unsupported_selector_rejected

end GB300Math.TensorSparseSelectors
