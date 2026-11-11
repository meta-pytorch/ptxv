/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF32Properties
namespace FabricF32Permutation
open FabricF32 GB300Math.TensorAlignmentLoss

theorem filter_empty {xs ys : List Nat} (hp : xs.Perm ys) (p : Nat → Bool) :
    (xs.filter p).isEmpty = (ys.filter p).isEmpty := by
  have h := (hp.filter p).length_eq
  cases hxs : xs.filter p <;> cases hys : ys.filter p <;> simp_all

theorem nan_fold {xs ys : List Nat} (hp : xs.Perm ys) :
    xs.foldl (fun p w => max p (w ||| 0x400000)) 0 =
      ys.foldl (fun p w => max p (w ||| 0x400000)) 0 := by
  exact hp.foldl_eq' (by intros; omega) 0

theorem predict_perm {xs ys : List Nat} (hp : xs.Perm ys) : predict xs = predict ys := by
  have hn := hp.filter (fun w => decide (w % 2^31 > 0x7f800000))
  have hi := hp.filter (fun w => decide (w % 2^31 = 0x7f800000))
  have hne := filter_empty hp (fun w => decide (w % 2^31 > 0x7f800000))
  have hie := filter_empty hp (fun w => decide (w % 2^31 = 0x7f800000))
  have hneg := hi.any_eq (f := fun w => decide (w ≥ 2^31))
  have hpos := hi.any_eq (f := fun w => decide (w < 2^31))
  have ha : aligned xs (xs.map fun w => (units w).getD 0) =
      aligned ys (ys.map fun w => (units w).getD 0) := by
    unfold aligned
    rw [FabricF32Properties.fineShift_perm hp,
      FabricF32Properties.truncatedSum_perm (hp.map (fun w => (units w).getD 0))]
  unfold predict
  dsimp only
  rw [hne, hie, nan_fold hn, hneg, hpos, ha, FabricF32Properties.coarseShift_perm hp]

#print axioms filter_empty
#print axioms nan_fold
#print axioms predict_perm
end FabricF32Permutation
