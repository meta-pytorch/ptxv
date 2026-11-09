import RadianScale
namespace GB300Math.RadianScale
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 8000000
theorem mul_radian_negative (e f : Nat) (he : 4 ≤ e ∧ e ≤ 254) (hf : f < 2^23) :
    let p := product f
    let l := p.log2
    Approx.mul (2^31+(e*2^23+f)) 0x3e22f983 true =
      2^31+((l+e-49)*2^23+(p/2^(l-23)-2^23)) := by
  dsimp only
  have hr := product_range f hf
  have hleft : Approx.asFloat (2^31+(e*2^23+f))=Model.ofFields FloatFormat.binary32 true e f := by
    simpa [Nat.add_assoc] using (RcpPtxBounds.fields_float true e f (by omega) hf).symm
  have hright : Approx.asFloat 0x3e22f983=Model.ofFields FloatFormat.binary32 false 124 2292099 := by
    simpa using (RcpPtxBounds.fields_float false 124 2292099 (by norm_num) (by norm_num)).symm
  change Approx.asWord (Model.mulWithRounding .towardZero _ _)=_
  rw [hleft,hright,Model.mulWithRounding_eq_of_toDyadic?_some .towardZero (by decide)
    (RcpPtxBounds.decode_normal true e f (by omega) hf)
    (RcpPtxBounds.decode_normal false 124 2292099 (by norm_num) (by norm_num))]
  have hp : (2^23+f)*10680707=product f := rfl
  have heq : ((e:Int)-150)+(124-150)=(e:Int)-176 := by omega
  simp only [Model.roundDyadicWithRounding,Model.roundDyadicTowardZero,Bool.xor_false]
  norm_num only [Nat.reducePow,Nat.reduceAdd,Nat.cast_ofNat] at heq ⊢
  rw [heq]
  change Approx.asWord (if product f==0 then Model.zero _ true else
    Model.roundDyadicMagnitudeDown FloatFormat.binary32 true (product f) ((e:Int)-176))=_
  have hn : (product f==0)=false := by simp;omega
  rw [hn]
  simp only [Bool.false_eq_true,ite_false]
  have hlog0 := Nat.log2_self_le (by omega : product f ≠ 0)
  have hlog1 := Nat.lt_log2_self (n:=product f)
  generalize hl : (product f).log2=l at *
  have hl0 : 46 ≤ l := hr.2.2.1
  have hl1 : l ≤ 47 := hr.2.2.2
  have hmant : 2^23 ≤ product f/2^(l-23) ∧ product f/2^(l-23) < 2^24 := by
    interval_cases l <;> norm_num at * <;> omega
  have hout : 1 ≤ l+e-49 ∧ l+e-49 ≤ 254 := by omega
  have hlow : ¬(l:Int)+((e:Int)-176)< -126 := by omega
  have hhigh : ¬127<(l:Int)+((e:Int)-176) := by omega
  have hsub : ¬(l:Int)+((e:Int)-176)< -149 := by omega
  have henc : ((l:Int)+((e:Int)-176)+127).toNat=l+e-49 := by omega
  have hlead : ¬l<23 := by omega
  have hemax : ¬l+e-49>254 := by omega
  have hfmax : ¬(l+e-49=254 ∧ product f/2^(l-23)-8388608>8388607) := by omega
  have hlead' : 23 ≤ l := by omega
  simp only [Model.roundDyadicMagnitudeDown,hl,
    show FloatFormat.binary32.fracWidth=23 from rfl,
    show FloatFormat.binary32.maxNormalExponent=127 from rfl,
    show FloatFormat.binary32.minNormalExponent= -126 from rfl,
    show FloatFormat.binary32.minSubnormalExponent= -149 from rfl,
    show FloatFormat.binary32.exponentBias=127 from rfl,
    show FloatFormat.binary32.maxFiniteExpField=254 from rfl,
    show FloatFormat.binary32.maxFiniteFracField=8388607 from rfl,
    hlead',ite_true,
    show Model.pow2 23=8388608 from rfl]
  change Approx.asWord (if (l:Int)+((e:Int)-176)>127 then _ else if (l:Int)+((e:Int)-176)< -149 then _ else if (l:Int)+((e:Int)-176)< -126 then _ else _) = _
  rw [ite_eq_right hhigh,ite_eq_right hsub,ite_eq_right hlow]
  change (Int.ofNat l+((e:Int)-176)+Int.ofNat 127).toNat=l+e-49 at henc
  rw [henc]
  have hsh : Nat.shiftRight (product f) (l-23)=product f/2^(l-23) := Nat.shiftRight_eq_div_pow _ _
  rw [hsh]
  have hfmax' : ¬product f/2^(l-23)-8388608>8388607 := by omega
  simp only [hemax,hfmax',decide_false,Bool.and_false,Bool.or_self,Bool.false_eq_true,ite_false]
  have hfrac : product f/2^(l-23)-2^23<2^23 := by omega
  have result := RcpPtxBounds.fields_word true (l+e-49) (product f/2^(l-23)-2^23) hout hfrac
  simpa only [Nat.reducePow,ite_true,Nat.add_assoc] using result

#print axioms mul_radian_negative
end GB300Math.RadianScale
