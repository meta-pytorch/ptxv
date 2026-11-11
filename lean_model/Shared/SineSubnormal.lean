import SineArgument
namespace GB300Math.SineSubnormal
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxHeartbeats 3000000

theorem fields (f : Fin (2^23)) :
    let x := (Model.ofNatBits f.val : Model FloatFormat.binary32)
    x.expField=0 ∧ x.fracField=f.val ∧ x.signBit=false := by
  have hf := f.isLt
  have hw : f.val<2^32 := by omega
  have hn : f.val<2^31 := by omega
  constructor
  · change (((BitVec.ofNat 32 f.val >>> 23) &&& (255:BitVec 32)).toNat)=0
    simp only [BitVec.toNat_and,BitVec.toNat_ushiftRight,BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hw,Nat.shiftRight_eq_div_pow]
    change ((f.val/2^23) &&& (2^8-1))=0
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  constructor
  · change ((BitVec.ofNat 32 f.val &&& (8388607:BitVec 32)).toNat)=f.val
    simp only [BitVec.toNat_and,BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hw]
    change (f.val &&& (2^23-1))=f.val
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  · rw [Model.signBit_eq_msb]
    change (BitVec.ofNat 32 f.val).msb=false
    rw [BitVec.msb_eq_decide]
    simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt hw]
    simpa using hn

theorem decode (f : Fin (2^23)) :
    (Model.ofNatBits f.val : Model FloatFormat.binary32).toRat?=some ((f.val:Rat)/2^149) := by
  have h := fields f
  have hieee : FloatFormat.binary32.isIEEE=true := by decide
  have hall : (0 == FloatFormat.binary32.expAllOnesNat)=false := by decide
  have hexp : FloatFormat.binary32.ieeeMinSubnormalExponent = -149 := by decide
  simp only [Model.toRat?,Model.toDyadic?,hieee,ite_true,Model.ieeeToDyadic?,
    Model.IEEE.isNaN,Model.IEEE.isInf,h.1,h.2.1,h.2.2,hall,hexp]
  by_cases hz : f.val=0
  all_goals simp [hz,FloatLib.Numerics.Dyadic.toRat,FloatLib.Numerics.Dyadic.signedSignificand]
  all_goals ring

theorem reduction (f : Fin (2^23)) :
    let q := SineArgument.rawPhase 0 (2^23+f.val)
    ∃ x r : Rat, ∃ n : Nat,
      (Model.ofNatBits f.val : Model FloatFormat.binary32).toRat?=some x ∧
      x=(n:Rat)+(q:Rat)/2^25+r ∧ 0≤r ∧ r≤1/2^25 := by
  have hf := f.isLt
  have hfr : (f.val:Rat) ≤ 2^23 := by exact_mod_cast (show f.val ≤ 2^23 by omega)
  refine ⟨(f.val:Rat)/2^149,(f.val:Rat)/2^149,0,decode f,?_,by positivity,?_⟩
  · simp [SineArgument.rawPhase]
  · norm_num
    nlinarith
#print axioms decode
#print axioms reduction
end GB300Math.SineSubnormal
