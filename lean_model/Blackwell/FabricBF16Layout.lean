/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricBF16Permutation

namespace FabricBF16Layout
open FabricBF16Block

theorem rne_local (xs ys : Array Accum) (i j : Nat) (h : xs[i]! = ys[j]!) :
    output xs i false = output ys j false := by
  cases hv : ys[j]! <;> simp [output,h,hv]

def original : Array (Array Nat) :=
  #[#[0x3f80,0x3fc0,0x3f80,0,0,0,0,0],#[0x3b80,0,0,0,0,0,0,0]]

def swapped : Array (Array Nat) := original.map fun row =>
  #[row[0]!,row[2]!,row[1]!,row[3]!,row[4]!,row[5]!,row[6]!,row[7]!]

theorem fixed_target : (original.toList.map fun row => row[0]!) =
    (swapped.toList.map fun row => row[0]!) := by decide +kernel

theorem inputs_permuted : ∀ rank : Fin 2,
    original[rank]!.toList.Perm swapped[rank]!.toList := by decide +kernel

theorem original_output : (predict original true)[0]! = 0x3f81 := by decide +kernel

theorem swapped_output : (predict swapped true)[0]! = 0x3f80 := by decide +kernel

theorem rne_unchanged : (predict original false)[0]! = 0x3f80 ∧
    (predict swapped false)[0]! = 0x3f80 := by decide +kernel

theorem sr_not_lane_equivariant : (predict original true)[0]! ≠
    (predict swapped true)[0]! := by rw [original_output,swapped_output];decide +kernel

#print axioms rne_local
#print axioms fixed_target
#print axioms inputs_permuted
#print axioms original_output
#print axioms swapped_output
#print axioms rne_unchanged
#print axioms sr_not_lane_equivariant
end FabricBF16Layout
