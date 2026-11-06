import FabricBF16

namespace FabricBF16Permutation
open FabricBF16Block GB300Math.TensorAlignmentLoss

theorem maxExponent_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    maxExponent xs = maxExponent ys := by
  unfold maxExponent
  exact hp.foldr_eq' (by intros; omega) 0

theorem latticeShift_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    latticeShift xs = latticeShift ys := by simp [latticeShift, maxExponent_perm hp]

theorem truncatedSum_perm {xs ys : List Int} (hp : xs.Perm ys) (step : Nat) :
    truncatedSum xs step = truncatedSum ys step := by
  unfold truncatedSum
  exact hp.foldr_eq' (by intros; omega) 0

theorem filter_empty {xs ys : List Nat} (hp : xs.Perm ys) (p : Nat → Bool) :
    (xs.filter p).isEmpty = (ys.filter p).isEmpty := by
  have h := (hp.filter p).length_eq
  cases hxs : xs.filter p <;> cases hys : ys.filter p <;> simp_all

theorem nan_payload_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    xs.foldl (fun p w => max p (w%64)) 0 =
      ys.foldl (fun p w => max p (w%64)) 0 := by
  exact hp.foldl_eq' (by intros; omega) 0

theorem accumulate_perm {xs ys : List Nat} (hp : xs.Perm ys) :
    accumulate xs = accumulate ys := by
  let isnan := fun w => decide (w%32768>32640)
  let isinf := fun w => decide (w%32768=32640)
  let negative := fun w => decide (w≥32768)
  have hn := hp.filter isnan
  have hi := hp.filter isinf
  have hne := filter_empty hp isnan
  have hie := filter_empty hp isinf
  have hnneg := hn.any_eq (f := negative)
  have hneg := hi.any_eq (f := negative)
  have hpos := hi.any_eq (f := fun w => decide (w<32768))
  have ha : aligned xs (xs.map fun w => (units w).getD 0) =
      aligned ys (ys.map fun w => (units w).getD 0) := by
    unfold aligned
    rw [latticeShift_perm hp,truncatedSum_perm (hp.map (fun w => (units w).getD 0))]
  have hnfiltered := nan_payload_perm (hn.filter negative)
  have hnall := nan_payload_perm hn
  unfold accumulate
  dsimp only
  change (if !(xs.filter isnan).isEmpty then _ else _) = _
  rw [hne,hie,hnneg,hneg,hpos,ha]
  split
  · split <;> simp_all [isnan,negative,isinf] <;> rfl
  · rfl

#print axioms maxExponent_perm
#print axioms latticeShift_perm
#print axioms truncatedSum_perm
#print axioms filter_empty
#print axioms nan_payload_perm
#print axioms accumulate_perm
theorem state_array_perm (xs ys : Array (Array Nat)) (hp : xs.toList.Perm ys.toList) (group : Nat) :
    (Array.range 4).map (fun lane => accumulate (xs.toList.map fun rank => rank[group*4+lane]!)) =
      (Array.range 4).map (fun lane => accumulate (ys.toList.map fun rank => rank[group*4+lane]!)) := by
  apply Array.map_congr_left
  intro lane h
  exact accumulate_perm (hp.map (fun rank => rank[group*4+lane]!))

theorem predict_perm (xs ys : Array (Array Nat)) (hp : xs.toList.Perm ys.toList) (sr : Bool) :
    predict xs sr = predict ys sr := by
  unfold predict
  simp_rw [state_array_perm xs ys hp]

#print axioms state_array_perm
#print axioms predict_perm
end FabricBF16Permutation
