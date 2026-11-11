/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2TightCells
namespace GB300Math.Log2RelativeCells
open Log2Cells Log2TightCells
set_option maxHeartbeats 0
set_option maxRecDepth 1000000

def rightIdeal (j : Nat) : Rat := if j<511 then center (j+1)+2/1000000000000 else 1
def upper (j : Nat) : Rat :=
  (2-rightIdeal j)/2^22-1/2^23-1/2^36-4/1000000000-correction j
abbrev RelativeCell (j : Nat) : Prop :=
  let (a,b,c) := differences j
  a≤upper j ∧ a+b/2≤upper j ∧ a+b+c≤upper j

theorem cells_relative : ∀ j : Fin 512,RelativeCell j.val := by decide +kernel
#print axioms cells_relative
end GB300Math.Log2RelativeCells
