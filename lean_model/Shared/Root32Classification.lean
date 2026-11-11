import Root32FloatLib
import SineSubnormal
namespace GB300Math.Root32FloatLib
set_option maxRecDepth 10000
set_option maxHeartbeats 3000000

theorem normal_positive (e : Fin 254) (f : Fin (2^23)) :
    0 < (RcpBounds.positiveNormalValue ((1+e.val)*2^23+f.val) : ℝ) := by
  rw [RcpBounds.normal_decode _ _ (by have:=e.isLt;omega) f.isLt]
  push_cast
  positivity

theorem sqrt_relative (e : Fin 254) (f : Fin (2^23)) :
    ∃ x y : Rat,
      value ((1+e.val)*2^23+f.val)=some x ∧
      value (Mufu.sqrt ((1+e.val)*2^23+f.val))=some y ∧
      |(y:ℝ)-Real.sqrt (x:ℝ)|/|Real.sqrt (x:ℝ)|≤1/2^23 := by
  have he : 1≤1+e.val ∧ 1+e.val≤254 := by have:=e.isLt;omega
  obtain ⟨x,y,hx,hy,hbound⟩ := sqrt_decoded e f
  have hdecode := decode_normal ((1+e.val)*2^23+f.val) (by have:=f.isLt;omega)
  have hxeq : x=RcpBounds.positiveNormalValue ((1+e.val)*2^23+f.val) := by rw [hx] at hdecode;exact Option.some.inj hdecode
  have hxpos : 0<(x:ℝ) := by rw [hxeq];exact normal_positive e f
  have hscale : SqrtBounds.outputScale (1+e.val) ≤ Real.sqrt (x:ℝ) := by
    rw [hxeq]
    change SqrtBounds.outputScale _≤Real.sqrt (SqrtBounds.positiveNormalValue _)
    rw [SqrtBounds.input_decode_scaled _ _ (by omega) f.isLt,
      Real.sqrt_mul (by unfold SqrtBounds.normalizedInput;positivity),
      Real.sqrt_sq (by unfold SqrtBounds.outputScale;positivity)]
    have hnorm : 1≤SqrtBounds.normalizedInput (SqrtBounds.mantissaIndex (1+e.val) f.val) := by
      unfold SqrtBounds.normalizedInput
      have hp : (1:ℝ)≤2^(SqrtBounds.mantissaIndex (1+e.val) f.val/2^23) := one_le_pow₀ (by norm_num)
      have hm : (0:ℝ)≤((SqrtBounds.mantissaIndex (1+e.val) f.val%2^23:Nat):ℝ) := Nat.cast_nonneg _
      have ha : 1≤(2^23+((SqrtBounds.mantissaIndex (1+e.val) f.val%2^23:Nat):ℝ))/2^23 := by norm_num only [Nat.reducePow] at hm ⊢; nlinarith
      simpa using mul_le_mul ha hp (by norm_num : (0:ℝ)≤1) (by linarith)
    have hsqrt : 1≤Real.sqrt (SqrtBounds.normalizedInput (SqrtBounds.mantissaIndex (1+e.val) f.val)) := by
      exact (Real.one_le_sqrt).mpr hnorm
    have hp := mul_le_mul_of_nonneg_right hsqrt (show 0≤SqrtBounds.outputScale (1+e.val) by unfold SqrtBounds.outputScale;positivity)
    simpa using hp
  refine ⟨x,y,hx,hy,?_⟩
  rw [abs_of_pos (Real.sqrt_pos.mpr hxpos)]
  apply (div_le_iff₀ (Real.sqrt_pos.mpr hxpos)).mpr
  nlinarith

theorem rsq_relative (e : Fin 254) (f : Fin (2^23)) :
    ∃ x y : Rat,
      value ((1+e.val)*2^23+f.val)=some x ∧
      value (Mufu.rsq ((1+e.val)*2^23+f.val))=some y ∧
      |(y:ℝ)-1/Real.sqrt (x:ℝ)|/|1/Real.sqrt (x:ℝ)|≤17/2^27 := by
  obtain ⟨x,y,hx,hy,hbound⟩ := rsq_decoded e f
  have hdecode := decode_normal ((1+e.val)*2^23+f.val) (by have:=e.isLt;have:=f.isLt;omega)
  have hxeq : x=RcpBounds.positiveNormalValue ((1+e.val)*2^23+f.val) := by rw [hx] at hdecode;exact Option.some.inj hdecode
  have hxpos : 0<(x:ℝ) := by rw [hxeq];exact normal_positive e f
  have hs := Real.sqrt_pos.mpr hxpos
  refine ⟨x,y,hx,hy,?_⟩
  rw [abs_of_pos (by positivity : 0<(1:ℝ)/Real.sqrt (x:ℝ))]
  apply (div_le_iff₀ (by positivity : 0<(1:ℝ)/Real.sqrt (x:ℝ))).mpr
  simpa only [div_eq_mul_inv,one_mul] using hbound

theorem zero_exponent (negative : Bool) (f : Fin (2^23)) :
    Mufu.sqrt ((if negative then 2^31 else 0)+f.val)=(if negative then 2^31 else 0) ∧
    Mufu.rsq ((if negative then 2^31 else 0)+f.val)=(if negative then 2^31 else 0)+0x7f800000 := by
  have hf:=f.isLt
  have he : Mufu.exponent ((if negative then 2^31 else 0)+f.val)=0 := by
    cases negative <;> simp only [Bool.false_eq_true,ite_false,ite_true,Nat.zero_add] <;> unfold Mufu.exponent <;> omega
  have hm : ¬Mufu.magnitude ((if negative then 2^31 else 0)+f.val)>0x7f800000 := by
    cases negative <;> simp only [Bool.false_eq_true,ite_false,ite_true,Nat.zero_add] <;> unfold Mufu.magnitude <;> omega
  have hs : Mufu.sign ((if negative then 2^31 else 0)+f.val)=(if negative then 2^31 else 0) := by
    cases negative
    · simpa using RcpSigned.sign_zero f.val (by omega)
    · simpa using RcpSigned.sign_set f.val (by omega)
  simp only [Mufu.sqrt,Mufu.rsq,hm,ite_false,he,ite_true,hs,eq_self,and_self]

theorem positive_subnormal_sqrt_error (f : Fin (2^23)) (hf : 0<f.val) :
    ∃ x : Rat, value f.val=some x ∧ value (Mufu.sqrt f.val)=some 0 ∧
      |(0:ℝ)-Real.sqrt (x:ℝ)|/|Real.sqrt (x:ℝ)|=1 := by
  have h := zero_exponent false f
  simp only [Bool.false_eq_true,ite_false,Nat.zero_add] at h
  refine ⟨_,SineSubnormal.decode f,?_,?_⟩
  · rw [h.1];decide +kernel
  · have hx : 0<((f.val:Rat)/2^149:ℝ) := by exact_mod_cast (show 0<(f.val:Rat)/2^149 by positivity)
    rw [zero_sub,abs_neg]
    push_cast
    exact div_self (ne_of_gt (abs_pos.mpr (ne_of_gt (Real.sqrt_pos.mpr hx))))

theorem negative_normal (e : Fin 254) (f : Fin (2^23)) :
    Mufu.sqrt (2^31+(1+e.val)*2^23+f.val)=0x7fffffff ∧
    Mufu.rsq (2^31+(1+e.val)*2^23+f.val)=0x7fffffff := by
  have he:=e.isLt
  have hf:=f.isLt
  have hx : Mufu.exponent (2^31+(1+e.val)*2^23+f.val)=1+e.val := by unfold Mufu.exponent;omega
  have hs : Mufu.sign (2^31+(1+e.val)*2^23+f.val)=2^31 := by
    have h := RcpSigned.sign_set ((1+e.val)*2^23+f.val) (by omega)
    simpa only [Nat.add_assoc] using h
  have hm : ¬Mufu.magnitude (2^31+(1+e.val)*2^23+f.val)>0x7f800000 := by unfold Mufu.magnitude;omega
  simp only [Mufu.sqrt,Mufu.rsq,hm,ite_false,hx,hs,Mufu.canonicalNaN]
  simp

theorem special_bits :
    Mufu.sqrt 0x7f800000=0x7f800000 ∧ Mufu.rsq 0x7f800000=0 ∧
    Mufu.sqrt 0xff800000=0x7fffffff ∧ Mufu.rsq 0xff800000=0x7fffffff := by decide +kernel

theorem nan_bits (w : Nat) (hn : Mufu.magnitude w>0x7f800000) :
    Mufu.sqrt w=0x7fffffff ∧ Mufu.rsq w=0x7fffffff := by
  simp [Mufu.sqrt,Mufu.rsq,hn,Mufu.canonicalNaN]

theorem positive_subnormal_rsq_nonfinite (f : Fin (2^23)) :
    value (Mufu.rsq f.val)=none := by
  have h := (zero_exponent false f).2
  simp only [Bool.false_eq_true,ite_false,Nat.zero_add] at h
  rw [h]
  decide +kernel

theorem positive_normal_count : 254*2^23=2130706432 := by decide +kernel

#print axioms negative_normal
#print axioms special_bits
#print axioms nan_bits
#print axioms positive_subnormal_rsq_nonfinite
#print axioms positive_normal_count

#print axioms normal_positive
#print axioms sqrt_relative
#print axioms rsq_relative
#print axioms zero_exponent
#print axioms positive_subnormal_sqrt_error
end GB300Math.Root32FloatLib
