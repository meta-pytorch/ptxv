import GB300Math.TensorError

namespace GB300Math.TensorGroupError
open GB300Math.Tensor GB300Math.TensorError

theorem flattened_sum (groups : List (List Int)) :
    groups.flatten.sum = (groups.map List.sum).sum := by
  induction groups with
  | nil => rfl
  | cons xs groups ih =>
    simpa only [List.flatten_cons, List.sum_append, List.map_cons, List.sum_cons] using
      congrArg (xs.sum + ·) ih

theorem grouped_alignment_residual (groups : List (List Int)) (s : Nat) :
    (groups.flatten.sum -
      ((groups.map List.sum).map (fun x => shiftTruncate x (-(s:Int)))).sum * (2^s:Int)).natAbs
      ≤ groups.length * (2^s-1) := by
  rw [flattened_sum]
  simpa only [List.length_map] using sum_alignment_residual (groups.map List.sum) s

theorem grouped_two_stage_residual (groups : List (List Int)) (s t : Nat) :
    let aligned := ((groups.map List.sum).map (fun x => shiftTruncate x (-(s:Int)))).sum
    (groups.flatten.sum - shiftTruncate aligned (-(t:Int))*(2^(s+t):Int)).natAbs ≤
      groups.length*(2^s-1) + (2^t-1)*2^s := by
  dsimp only
  rw [flattened_sum]
  simpa only [List.length_map] using two_stage_alignment_residual (groups.map List.sum) s t

theorem grouping_changes_alignment :
    shiftTruncate (-160) (-4) + shiftTruncate 150 (-4) = -1 ∧
    shiftTruncate ((-160)+150) (-4) = 0 := by decide +kernel

#print axioms flattened_sum
#print axioms grouped_alignment_residual
#print axioms grouped_two_stage_residual
#print axioms grouping_changes_alignment
end GB300Math.TensorGroupError
