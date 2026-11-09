import Exp2Boundary
import SineSubnormal
namespace GB300Math.Exp2Decode
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

theorem positive (word : Nat) (hw : word<0x7f800000) :
    ∃ x:Rat,(Model.ofNatBits word:Model FloatFormat.binary32).toRat?=some x ∧
      (x:ℝ)=Exp2Bounds.finiteValue word := by
  have hsign : word<2^31 := by omega
  have hf : Mufu.fraction word<2^23 := by unfold Mufu.fraction;omega
  have he : Mufu.exponent word<255 := by unfold Mufu.exponent;omega
  have hword : word=Mufu.exponent word*2^23+Mufu.fraction word := by
    unfold Mufu.exponent Mufu.fraction;omega
  by_cases hz : Mufu.exponent word=0
  · let f:Fin (2^23):=⟨Mufu.fraction word,hf⟩
    have hwf : word=f.val := by dsimp [f];omega
    refine ⟨(f.val:Rat)/2^149,?_,?_⟩
    · rw [hwf];exact SineSubnormal.decode f
    · simp only [Exp2Bounds.finiteValue,hsign,ite_true,hz,one_mul]
      dsimp [f]
      push_cast
      norm_num
      ring
  · let e:Fin 254:=⟨Mufu.exponent word-1,by omega⟩
    let f:Fin (2^23):=⟨Mufu.fraction word,hf⟩
    have heq : 1+e.val=Mufu.exponent word := by dsimp [e];omega
    have hpow : (e.val:Int)-149=(Mufu.exponent word:Int)-150 := by dsimp [e];omega
    have hd:=MufuRcp32Bounds.decode_normal e f
    rw [heq,hpow,←hword] at hd
    refine ⟨((2^23+f.val:Nat):Rat)*(2:Rat)^((Mufu.exponent word:Int)-150),hd,?_⟩
    simp only [Exp2Bounds.finiteValue,hsign,ite_true,hz,ite_false,one_mul]
    dsimp [f]
    push_cast
    norm_num

theorem finite (word : Nat) (hw : word<2^32) (hf : Mufu.magnitude word<0x7f800000) :
    ∃ x:Rat,(Model.ofNatBits word:Model FloatFormat.binary32).toRat?=some x ∧
      (x:ℝ)=Exp2Bounds.finiteValue word := by
  by_cases hs : word<2^31
  · apply positive
    simpa only [Mufu.magnitude,Nat.mod_eq_of_lt hs] using hf
  · let mag:=word%2^31
    have hm : mag<0x7f800000 := hf
    have hw' : word=mag+2^31 := by dsimp [mag];omega
    obtain ⟨x,hx,hv⟩:=positive mag hm
    refine ⟨-x,?_,?_⟩
    · rw [hw'];exact RcpSigned.signed_decode mag x hx
    · have hm' : mag<2^31 := by omega
      have he : Mufu.exponent word=Mufu.exponent mag := by unfold Mufu.exponent;dsimp [mag];omega
      have hfr : Mufu.fraction word=Mufu.fraction mag := by unfold Mufu.fraction;dsimp [mag];omega
      rw [Rat.cast_neg,hv]
      simp only [Exp2Bounds.finiteValue,hs,hm',ite_true,ite_false,he,hfr]
      ring

theorem packed (be : Int) (mant : Nat) (he : 1≤be ∧ be≤254) (hm : (2^23:Nat)≤ mant ∧ mant<2^24) :
    ∃ y:Rat,(Model.ofNatBits (Exp2Boundary.pack be mant):Model FloatFormat.binary32).toRat?=some y ∧
      (y:ℝ)=(mant:ℝ)/2^23*(2:ℝ)^(be-127) := by
  let e:Fin 254:=⟨be.toNat-1,by omega⟩
  let f:Fin (2^23):=⟨mant%2^23,by omega⟩
  have heq : 1+e.val=be.toNat := by dsimp [e];omega
  have hpow : (e.val:Int)-149=be-150 := by dsimp [e];omega
  have hmant : 2^23+f.val=mant := by dsimp [f];omega
  have hd:=MufuRcp32Bounds.decode_normal e f
  rw [heq,hpow,hmant] at hd
  refine ⟨(mant:Rat)*(2:Rat)^(be-150),?_,?_⟩
  · simpa only [Exp2Boundary.pack,show ¬be≤0 by omega,show ¬be≥255 by omega,ite_false] using hd
  · push_cast
    have hp : be-150=(be-127)-23 := by omega
    rw [hp,zpow_sub₀ (by norm_num),zpow_ofNat]
    ring

#print axioms positive
#print axioms finite
#print axioms packed
end GB300Math.Exp2Decode
