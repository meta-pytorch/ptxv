import Log2SubShape
import Log2SubAddDecode
namespace GB300Math.Log2Subnormal
open Root32FloatLib RootPtxScale
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

theorem relative_bound (f : Fin (2^23)) (hf : 0<f.val) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "lg2" false f.val=some out ∧
      value f.val=some x ∧ value out=some y ∧
      |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|/|Real.log (x:ℝ)/Real.log 2|≤1/2^22 := by
  obtain ⟨hop,hx,hn,hlo,hhi⟩:=normalize f hf
  generalize hnorm : normalized f.val=n at *
  obtain ⟨xn,yn,hxn,hyn,hxp,herr⟩:=Log2Uniform.raw_positive_normal_error n (by omega)
  have hxid : xn=(f.val:Rat)/2^149*2^24 := by rw [hn] at hxn;exact (Option.some.inj hxn).symm
  obtain ⟨ff,hff,hshape⟩:=Log2SubShape.output_shape n ⟨hlo,hhi⟩
  have hp:=RootPtxScale.normal_value 133 ff (by omega) hff
  have hs:=RcpSigned.signed_decode (133*2^23+ff) ((2^23+(ff:Rat))*2^((133:Int)-150)) hp
  have hbits : value (2^31+133*2^23+ff)=some (-((2^23+(ff:Rat))/2^17)) := by
    change value (133*2^23+ff+2^31)=some (-((2^23+(ff:Rat))*2^((133:Int)-150))) at hs
    rw [show 133*2^23+ff+2^31=2^31+133*2^23+ff by omega] at hs
    have hpow : (2:Rat)^((133:Int)-150)=1/2^17 := by norm_num
    rw [hpow,mul_one_div] at hs
    exact hs
  rw [hshape,hbits] at hyn
  have hyid : yn= -((2^23+(ff:Rat))/2^17) := Option.some.inj hyn.symm
  have hdecode := Log2SubAdd.decoded_add ff hff
  generalize hout : Log2SubAdd.result ff=out at hdecode
  obtain ⟨y,hy,hadd⟩ := hdecode
  let x:Rat:=(f.val:Rat)/2^149
  have hxpos : 0<(x:ℝ) := by dsimp[x];push_cast;positivity
  have hxlog : Real.log ((xn:ℝ))/Real.log 2=Real.log (x:ℝ)/Real.log 2+24 := by
    rw [hxid]
    change Real.log ((x*2^24:Rat):ℝ)/Real.log 2=_
    push_cast
    rw [Real.log_mul (ne_of_gt hxpos) (by positivity),Real.log_pow]
    have hlog2 : Real.log (2:ℝ)≠0 := ne_of_gt (Real.log_pos (by norm_num))
    field_simp
    norm_num; ring
  have ha : |(y:ℝ)-((yn:ℝ)-24)|≤1/2^16 := by
    have hyreal : (yn:ℝ)= -((2^23+(ff:ℝ))/2^17) := by
      have h := congrArg (fun q:Rat => (q:ℝ)) hyid
      norm_num only [Rat.cast_neg,Rat.cast_div,Rat.cast_add,Rat.cast_pow,Rat.cast_natCast,Rat.cast_ofNat] at h
      convert h using 1; norm_num
    rw [hyreal]
    convert hadd using 1; ring
  have he : |(y:ℝ)-Real.log (x:ℝ)/Real.log 2|≤1/2^16+1/125000 := by
    have hc:=abs_sub_le (y:ℝ) ((yn:ℝ)-24) (Real.log (x:ℝ)/Real.log 2)
    rw [hxlog] at herr
    have heq : |(yn:ℝ)-24-Real.log (x:ℝ)/Real.log 2|=|(yn:ℝ)-(Real.log (x:ℝ)/Real.log 2+24)| := by rw [sub_sub,add_comm (24:ℝ)]
    rw [heq] at hc
    linarith only[hc,ha,herr]
  have hxhi : (x:ℝ)≤(2:ℝ)^(-126:Int) := by
    dsimp[x];push_cast
    norm_num
    have hfr : (f.val:ℝ)<8388608 := by exact_mod_cast f.isLt
    linarith only[hfr]
  have hl2 : 0<Real.log (2:ℝ) := Real.log_pos (by norm_num)
  have hl:=Real.log_le_log hxpos hxhi
  rw [Real.log_zpow] at hl
  have hquot : Real.log (x:ℝ)/Real.log 2≤ -126 := by
    apply (div_le_iff₀ hl2).mpr
    norm_num at hl
    linarith only[hl]
  have hden : 126≤|Real.log (x:ℝ)/Real.log 2| := by
    rw [abs_of_nonpos (by linarith only[hquot])]
    linarith only[hquot]
  refine ⟨out,x,y,?_,hx,hy,?_⟩
  · have ht : Mufu.magnitude f.val<0x00800000 := by unfold Mufu.magnitude;have:=f.isLt;omega
    simp only [Approx.evaluate,String.reduceEq,Bool.false_eq_true,or_self,ite_false,ht,ite_true,hop,hshape]
    change some (Log2SubAdd.result ff)=some out
    rw [hout]
  · apply (div_le_iff₀ (by linarith only[hden])).mpr
    nlinarith only[he,hden]

#print axioms relative_bound
end GB300Math.Log2Subnormal
