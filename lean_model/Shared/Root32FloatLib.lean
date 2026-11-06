import SqrtTightBounds
import RsqPtxValues
import MufuRcp32Signed
namespace GB300Math.Root32FloatLib
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 10000
set_option maxHeartbeats 3000000

def value (w : Nat) : Option Rat := (Model.ofNatBits w : Model FloatFormat.binary32).toRat?

theorem decode_normal (w : Nat) (hw : 2^23 ≤ w ∧ w < 255*2^23) :
    value w = some (RcpBounds.positiveNormalValue w) := by
  let e : Fin 254 := ⟨w/2^23-1,by omega⟩
  let f : Fin (2^23) := ⟨w%2^23,by omega⟩
  have hword : (1+e.val)*2^23+f.val=w := by dsimp [e,f];omega
  have h := MufuRcp32Bounds.decode_normal e f
  rw [hword] at h
  have hmanual := RcpBounds.normal_decode (1+e.val) f.val (by have:=e.isLt;omega) f.isLt
  rw [hword] at hmanual
  unfold value
  rw [h,hmanual]
  push_cast
  congr 3
  ring

theorem sqrt_range (e f : Nat) (he : 1≤e ∧ e≤254) (hf : f<2^23) :
    64*2^23≤Mufu.sqrt (e*2^23+f) ∧ Mufu.sqrt (e*2^23+f)<191*2^23 := by
  have ht : -63≤SqrtBounds.scaleExponent e ∧ SqrtBounds.scaleExponent e≤63 := by
    unfold SqrtBounds.scaleExponent
    omega
  rw [SqrtBounds.output_word_scaled e f he hf]
  omega

theorem sqrt_decoded (e : Fin 254) (f : Fin (2^23)) :
    ∃ x y : Rat,
      value ((1+e.val)*2^23+f.val)=some x ∧
      value (Mufu.sqrt ((1+e.val)*2^23+f.val))=some y ∧
      |(y:ℝ)-Real.sqrt (x:ℝ)|≤SqrtBounds.outputScale (1+e.val)/2^23 := by
  have he : 1≤1+e.val ∧ 1+e.val≤254 := by have:=e.isLt;omega
  have hw : 2^23≤(1+e.val)*2^23+f.val ∧ (1+e.val)*2^23+f.val<255*2^23 := by have:=f.isLt;omega
  have ho := sqrt_range (1+e.val) f.val he f.isLt
  refine ⟨_,_,decode_normal _ hw,decode_normal _ (by omega),?_⟩
  exact SqrtTightBounds.sqrt_bits_scaled_absolute_error (1+e.val) f.val he f.isLt

theorem rsq_decoded (e : Fin 254) (f : Fin (2^23)) :
    ∃ x y : Rat,
      value ((1+e.val)*2^23+f.val)=some x ∧
      value (Mufu.rsq ((1+e.val)*2^23+f.val))=some y ∧
      |(y:ℝ)-1/Real.sqrt (x:ℝ)|≤((17:ℝ)/2^27)/Real.sqrt (x:ℝ) := by
  have he : 1≤1+e.val ∧ 1+e.val≤254 := by have:=e.isLt;omega
  have hw : 2^23≤(1+e.val)*2^23+f.val ∧ (1+e.val)*2^23+f.val<255*2^23 := by have:=f.isLt;omega
  have ho := RsqPtxBounds.output_normal_range (1+e.val) f.val he f.isLt
  refine ⟨_,_,decode_normal _ hw,decode_normal _ (by omega),?_⟩
  exact RsqPtxBounds.rsq_bits_relative_error (1+e.val) f.val he f.isLt

#print axioms decode_normal
#print axioms sqrt_range
#print axioms sqrt_decoded
#print axioms rsq_decoded
end GB300Math.Root32FloatLib
