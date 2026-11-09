import RsqBounds

set_option maxRecDepth 1000000
set_option maxHeartbeats 0

namespace GB300Math.RsqPtxBounds
open RsqBounds

def residualBound : Int := 17*2^45-289*2^17

abbrev CellBound (p : Parameters) (left : Int) : Prop :=
  let (q0,q1,q2,q3,q4,q5) := taylor p left
  0 ≤ p.c ∧ p.b+262142*p.c ≤ 0 ∧ 0 < p.m ∧ 0 ≤ p.step ∧
  2^23*2^34 ≤ numerator p 131071-p.c*4718593 ∧ p.a < (2^24+1)*2^34 ∧
  -residualBound*2^68+2*(p.m+p.step*(left+2047))*numerator p left*loss p ≤
    lower q0 q1 q2 q3 q4 q5 2047 ∧
  upper q0 q1 q2 q3 q4 q5 2047 ≤ residualBound*2^68

abbrev ExactCell (p : Parameters) (left : Nat) : Prop :=
  ∀ offset : Fin 2048,
    let k := left+offset.val
    let z := (p.a+p.b*(k : Int)+p.c*(Fp64Seed.squareTruncated k 19 : Int))/2^34;
    -residualBound ≤ (p.m+p.step*(k : Int))*z^2-2^71 ∧
    (p.m+p.step*(k : Int))*z^2-2^71 ≤ residualBound

theorem cells_valid : ∀ segment : Fin 128, ∀ cell : Fin 64,
    CellBound (parameters segment.val) ((cell.val : Int)*2048) ∨
    ExactCell (parameters segment.val) (cell.val*2048) := by decide +kernel

#print axioms cells_valid
end GB300Math.RsqPtxBounds
