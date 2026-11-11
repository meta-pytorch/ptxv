/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Log2CellData
import Log2Drop
namespace GB300Math.Log2TightCells
open Log2Cells
set_option maxRecDepth 1000000
set_option maxHeartbeats 0

def dropBound (j : Nat) : Nat := Log2Drop.bound ((j%8+1)*16384-1)
def correction (j : Nat) : Rat :=
  let c : Int := (MufuTables.lg2.coefficients[j / 8]!).2.2;
  -(c : Rat) * (dropBound j : Rat) / 2 ^ 60
abbrev Tight (j : Nat) : Prop :=
  let (a,b,c) := differences j;
  -173/1000000000+1/2^36≤a ∧ a≤173/1000000000-correction j ∧
  -173/1000000000+1/2^36≤a+b/2 ∧ a+b/2≤173/1000000000-correction j ∧
  -173/1000000000+1/2^36≤a+b+c ∧ a+b+c≤173/1000000000-correction j

theorem cells_tight : ∀ j : Fin 512,Tight j.val := by decide +kernel
#print axioms cells_tight
end GB300Math.Log2TightCells
