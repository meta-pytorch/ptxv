import FabricF32
namespace FabricF32Bounds
open FabricF32 GB300Math.TensorAlignmentLoss

theorem flush_loss_sharp (shift : Nat) (value : Int) :
    (flush shift value-value).natAbs ≤ 2^shift-1 := by
  unfold flush
  split
  next h =>
    simp only [Int.zero_sub, Int.natAbs_neg]
    omega
  next h => simp

theorem accumulator_loss (words : List Nat) (values : List Int) :
    (flush (coarseShift words) (aligned words values)-exactSum values).natAbs ≤
      values.length*(2^fineShift words-1)+(2^coarseShift words-1) := by
  have ht := sum_truncation_loss values (2^fineShift words) (Nat.two_pow_pos _)
  have hf := flush_loss_sharp (coarseShift words) (aligned words values)
  have hb : (aligned words values-exactSum values).natAbs ≤
      values.length*(2^fineShift words-1) := by
    dsimp [aligned]
    omega
  have he : flush (coarseShift words) (aligned words values)-exactSum values =
      (flush (coarseShift words) (aligned words values)-aligned words values)+
      (aligned words values-exactSum values) := by omega
  rw [he]
  exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add hf hb) |>.trans_eq (Nat.add_comm _ _)

#print axioms flush_loss_sharp
#print axioms accumulator_loss
end FabricF32Bounds
