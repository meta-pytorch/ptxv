import RcpPtxScale
import Root32Classification
namespace GB300Math.RootPtxScale
open FloatLib.Floats.Formats.BinaryInterchange
open Root32FloatLib
set_option maxRecDepth 10000
set_option maxHeartbeats 6000000

theorem normal_value (e f : Nat) (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23) :
    value (e*2^23+f)=some ((2^23+(f:Rat))*2^((e:Int)-150)) := by
  have h := MufuRcp32Bounds.decode_normal ⟨e-1,by omega⟩ ⟨f,hf⟩
  have heq : 1+(e-1)=e := by omega
  have hi : ((e-1:Nat):Int)-149=(e:Int)-150 := by omega
  convert h using 1 <;> norm_num [heq,hi,value]

theorem scaled_value (e f p : Nat) (he : 1 ≤ e ∧ e ≤ 254) (hf : f < 2^23)
    (hp : 1 ≤ p ∧ p ≤ 254) (hs : 128 ≤ e+p ∧ e+p ≤ 381) :
    value (Approx.mul (e*2^23+f) (p*2^23)) =
      some (((2^23+(f:Rat))*2^((e:Int)-150))*2^((p:Int)-127)) := by
  have h := RcpPtxBounds.mul_power_two_normal_word false e f p he hf hp hs
  simp only [Bool.false_eq_true,ite_false,Nat.zero_add] at h
  rw [h,normal_value (e+p-127) f (by omega) hf]
  apply congrArg some
  rw [mul_assoc,← zpow_add₀ (by norm_num : (2:Rat) ≠ 0)]
  congr 2
  omega

theorem scaled_word_value (w p : Nat)
    (hw : 2^23 ≤ w ∧ w < 255*2^23)
    (hp : 1 ≤ p ∧ p ≤ 254) (hs : 128 ≤ w/2^23+p ∧ w/2^23+p ≤ 381) :
    ∃ y : Rat, value w=some y ∧
      value (Approx.mul w (p*2^23))=some (y*2^((p:Int)-127)) := by
  have he : 1 ≤ w/2^23 ∧ w/2^23 ≤ 254 := by omega
  have hf : w%2^23 < 2^23 := by omega
  have hid : w/2^23*2^23+w%2^23=w := by omega
  refine ⟨(2^23+((w%2^23:Nat):Rat))*2^(((w/2^23:Nat):Int)-150),?_,?_⟩
  · simpa only [hid] using normal_value (w/2^23) (w%2^23) he hf
  · simpa only [hid] using scaled_value (w/2^23) (w%2^23) p he hf hp hs

def normalized (f : Nat) : Nat := (f.log2+2)*2^23+(f*2^(23-f.log2)-2^23)

theorem normalize (f : Fin (2^23)) (hf : 0 < f.val) :
    Approx.mul f.val 0x4b800000=normalized f.val ∧
    value f.val=some ((f.val:Rat)/2^149) ∧
    value (normalized f.val)=some (((f.val:Rat)/2^149)*2^24) ∧
    2*2^23 ≤ normalized f.val ∧ normalized f.val < 25*2^23 := by
  have hnorm := RcpPtxBounds.mul_subnormal_normalize_word false f.val (by omega) f.isLt
  simp only [Bool.false_eq_true,ite_false,Nat.zero_add] at hnorm
  have hm := RcpPtxBounds.normalized_mantissa_range f.val (by omega) f.isLt
  have hl : f.val.log2 ≤ 22 := by have h := (Nat.log2_lt (by omega : f.val ≠ 0)).mpr f.isLt;omega
  refine ⟨hnorm,SineSubnormal.decode f,?_,?_,?_⟩
  · unfold normalized
    rw [normal_value (f.val.log2+2) (f.val*2^(23-f.val.log2)-2^23) (by omega) (by omega)]
    congr 1
    have hmant : 2^23+((f.val*2^(23-f.val.log2)-2^23:Nat):Rat) = f.val*2^(23-f.val.log2) := by exact_mod_cast (show 2^23+(f.val*2^(23-f.val.log2)-2^23)=f.val*2^(23-f.val.log2) by omega)
    rw [hmant]
    generalize hh : f.val.log2=l at *
    interval_cases l <;> norm_num <;> ring
  · unfold normalized;omega
  · unfold normalized;omega

#print axioms normal_value
#print axioms scaled_value
#print axioms scaled_word_value
#print axioms normalize
end GB300Math.RootPtxScale
