import Log2Relative
import Log2PtxRange
import Log2Classification
namespace GB300Math.Log2Complete
set_option maxHeartbeats 3000000

theorem raw_outside_relative (w : Nat)
    (hw : 2^23 ≤ w ∧ w < 255*2^23)
    (ho : w < 126*2^23 ∨ 128*2^23 ≤ w) :
    ∃ x y : Rat, Root32FloatLib.value w = some x ∧
      Root32FloatLib.value (Mufu.lg2 w) = some y ∧ 0 < x ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22 := by
  let e : Fin 254 := ⟨w/2^23-1, by omega⟩
  let f : Fin (2^23) := ⟨w%2^23, by omega⟩
  have hword : (1+e.val)*2^23+f.val=w := by dsimp [e,f]; omega
  have h := Log2Relative.decoded_relative e f (by dsimp [e]; omega)
  rwa [hword] at h

theorem documented_contract (w : Nat) (hw : 2^23 ≤ w ∧ w < 255*2^23) :
    ∃ x y : Rat, Root32FloatLib.value w = some x ∧
      Root32FloatLib.value (Mufu.lg2 w) = some y ∧ 0 < x ∧
      (if 126*2^23 ≤ w ∧ w < 128*2^23
       then |(y:ℝ)-Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22
       else |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22) := by
  split_ifs with hnear
  · exact Log2PtxRange.raw_near_absolute_error w hnear
  · exact raw_outside_relative w hw (by omega)

theorem domain_counts :
    254*2^23 = (2130706432:Nat) ∧ 2*2^23 = (16777216:Nat) ∧
      252*2^23 = (2113929216:Nat) := by decide +kernel

#print axioms raw_outside_relative
#print axioms documented_contract
#print axioms domain_counts
end GB300Math.Log2Complete
