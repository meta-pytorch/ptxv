import Log2PtxContract
import Log2Subnormal
namespace GB300Math.Log2PtxAll
open Root32FloatLib
set_option maxHeartbeats 300000
set_option maxRecDepth 16000

theorem normal_evaluate (w : Nat) (hw : 2^23 ≤ w ∧ w < 255*2^23) :
    Approx.evaluate "lg2" false w = some (Mufu.lg2 w) := by
  have ht : ¬ Mufu.magnitude w < 0x00800000 := by
    unfold Mufu.magnitude
    omega
  simp only [Approx.evaluate,String.reduceEq,Bool.false_eq_true,or_self,ite_false,ht]

theorem documented_contract (w : Nat) (hw : 0 < w ∧ w < 255*2^23) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "lg2" false w = some out ∧
      value w = some x ∧ value out = some y ∧
      (if 126*2^23 < w ∧ w < 128*2^23
       then |(y:ℝ)-Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22
       else |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2| ≤ 1/2^22) := by
  by_cases hs : w < 2^23
  · obtain ⟨out,x,y,hop,hx,hy,herr⟩ := Log2Subnormal.relative_bound ⟨w,hs⟩ hw.1
    refine ⟨out,x,y,hop,hx,hy,?_⟩
    split_ifs with hnear
    · omega
    · exact herr
  · have hn : 2^23 ≤ w ∧ w < 255*2^23 := by omega
    obtain ⟨x,y,hx,hy,_,herr⟩ := Log2PtxContract.documented_open_contract w hn
    exact ⟨Mufu.lg2 w,x,y,normal_evaluate w hn,hx,hy,herr⟩

#print axioms normal_evaluate
#print axioms documented_contract
end GB300Math.Log2PtxAll
