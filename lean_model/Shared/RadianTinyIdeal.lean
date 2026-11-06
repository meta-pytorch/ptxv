import RadianTinyMul
namespace GB300Math.RadianTinyIdeal
open FloatLib.Floats.Formats.BinaryInterchange
open Root32FloatLib
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000

theorem sincos_small (negative : Bool) (w : Nat) (hw : w<2^24) :
    Mufu.sincos ((if negative then 2^31 else 0)+w) false=(if negative then 2^31 else 0) ∧
    Mufu.sincos ((if negative then 2^31 else 0)+w) true=0x3f800000 := by
  let e : Fin 255 := ⟨w/2^23,by omega⟩
  let f : Fin (2^23) := ⟨w%2^23,by omega⟩
  have hwrd : SineModelBridge.word negative e f=(if negative then 2^31 else 0)+w := by
    dsimp [SineModelBridge.word,e,f];omega
  have shape:=SineModelBridge.kernel_shape negative e f
  rw [hwrd] at shape
  have hh : w/2^23=0 ∨ w/2^23=1 := by omega
  have hp : SineModelBridge.phase e f=0 := by
    change SineArgument.rawPhase (w/2^23) (2^23+f.val)=0
    rcases hh with hh|hh <;> rw [hh] <;> norm_num [SineArgument.rawPhase]
  rw [hp] at shape
  have hvalues : SinePhaseResult.phaseWord 0 negative=(if negative then 2^31 else 0) ∧
      SinePhaseResult.phaseWord ((0+2^23)%2^25) false=0x3f800000 := by
    cases negative <;> decide +kernel
  exact ⟨shape.1.trans hvalues.1,shape.2.trans hvalues.2⟩

theorem positive_input (e f : Nat) (he:e<4) (hf:f<2^23) :
    ∃x:Rat,value (e*2^23+f)=some x ∧ 0≤x ∧ x<1/2^123 := by
  by_cases he0:e=0
  · subst e
    refine ⟨(f:Rat)/2^149,?_,by positivity,?_⟩
    · simpa [value] using SineSubnormal.decode ⟨f,hf⟩
    · have hfr : (f:Rat)<2^23 := by exact_mod_cast hf
      norm_num only [Nat.cast_ofNat] at *
      linarith only[hfr]
  · have hlo : 1≤e := by omega
    refine ⟨(2^23+(f:Rat))*2^((e:Int)-150),RootPtxScale.normal_value e f (by omega) hf,?_,?_⟩
    · positivity
    · have hfr : (f:Rat)<2^23 := by exact_mod_cast hf
      interval_cases e <;> norm_num only [Int.reduceSub,zpow_neg,zpow_natCast,Nat.cast_ofNat,pow_succ] <;> linarith only[hfr]

theorem input_decode (negative : Bool) (e f : Nat) (he:e<4) (hf:f<2^23) :
    ∃x:Rat,value ((if negative then 2^31 else 0)+(e*2^23+f))=some x ∧ |(x:ℝ)|<1/2^123 := by
  obtain ⟨x,hx,hx0,hx1⟩:=positive_input e f he hf
  have hxr : |(x:ℝ)|<1/2^123 := by
    rw [abs_of_nonneg (Rat.cast_nonneg.mpr hx0)]
    have hcast := (Rat.cast_lt (K:=ℝ)).mpr hx1
    norm_num only [Rat.cast_div,Rat.cast_one,Rat.cast_pow,Rat.cast_ofNat] at hcast
    norm_num only [pow_succ]
    exact hcast
  cases negative
  · refine ⟨x,?_,hxr⟩
    simpa using hx
  · refine ⟨-x,?_,?_⟩
    · have h:=RcpSigned.signed_decode (e*2^23+f) x hx
      simpa [value,Nat.add_comm] using h
    · simpa using hxr

theorem decoded_small_bound (negative : Bool) (e f : Nat) (he:e<4) (hf:f<2^23) :
    ∃x:Rat,value ((if negative then 2^31 else 0)+(e*2^23+f))=some x ∧
      value (Mufu.sincos (Approx.mul ((if negative then 2^31 else 0)+(e*2^23+f)) 0x3e22f983 true) false)=some 0 ∧
      value (Mufu.sincos (Approx.mul ((if negative then 2^31 else 0)+(e*2^23+f)) 0x3e22f983 true) true)=some 1 ∧
      |(0:ℝ)-Real.sin (x:ℝ)|<1/2^123 ∧ |(1:ℝ)-Real.cos (x:ℝ)|<1/2^123 := by
  obtain ⟨x,hx,hbound⟩:=input_decode negative e f he hf
  obtain ⟨w,hw,hword⟩:=RadianTinyMul.mul_small negative e f he hf
  have hs:=sincos_small negative w hw
  rw [hword,hs.1,hs.2]
  have hz : value (if negative then 2^31 else 0)=some 0 := by
    have hp : value 0=some 0 := by simpa [value] using SineSubnormal.decode ⟨0,by decide⟩
    cases negative
    · exact hp
    · have h:=RcpSigned.signed_decode 0 0 hp
      simpa [value] using h
  have hone : value 0x3f800000=some 1 := by
    have h:=RootPtxScale.normal_value 127 0 (by omega) (by decide)
    norm_num at h
    exact h
  refine ⟨x,hx,hz,hone,?_,?_⟩
  · have h:=Real.abs_sin_sub_sin_le 0 (x:ℝ)
    simp only [Real.sin_zero,zero_sub,abs_neg] at h ⊢
    exact h.trans_lt hbound
  · have h:=Real.abs_cos_sub_cos_le 0 (x:ℝ)
    simp only [Real.cos_zero,zero_sub,abs_neg] at h
    exact h.trans_lt hbound

#print axioms sincos_small
#print axioms positive_input
#print axioms input_decode
#print axioms decoded_small_bound
end GB300Math.RadianTinyIdeal
