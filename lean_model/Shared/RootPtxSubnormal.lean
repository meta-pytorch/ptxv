import RootPtxScale
namespace GB300Math.RootPtxSubnormal
open RootPtxScale Root32FloatLib
set_option maxRecDepth 10000
set_option maxHeartbeats 6000000

theorem relative_scale (y a c : ℝ) (hc : c ≠ 0) :
    |y*c-a*c|/|a*c|=|y-a|/|a| := by
  rw [← sub_mul,abs_mul,abs_mul]
  exact mul_div_mul_right _ _ (abs_ne_zero.mpr hc)

theorem ptx_sqrt_subnormal (f : Fin (2^23)) (hf : 0 < f.val) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "sqrt" false f.val=some out ∧
      value f.val=some x ∧ value out=some y ∧
      |(y:ℝ)-Real.sqrt (x:ℝ)|/|Real.sqrt (x:ℝ)|≤1/2^23 := by
  obtain ⟨hword,hx,hn,hlo,hhi⟩ := normalize f hf
  generalize hnorm : normalized f.val=n at *
  have hne : 1 ≤ n/2^23 ∧ n/2^23 ≤ 254 := by omega
  have hnf : n%2^23<2^23 := by omega
  let e : Fin 254 := ⟨n/2^23-1,by omega⟩
  let frac : Fin (2^23) := ⟨n%2^23,hnf⟩
  have hnword : (1+e.val)*2^23+frac.val=n := by dsimp [e,frac];omega
  obtain ⟨x,yn,hxin,hyn,hbound⟩ := sqrt_relative e frac
  rw [hnword] at hxin hyn
  have hxeq : x=(f.val:Rat)/2^149*2^24 := by rw [hn] at hxin;exact (Option.some.inj hxin).symm
  have horange : 64*2^23≤Mufu.sqrt n ∧ Mufu.sqrt n<191*2^23 := by
    have h := sqrt_range (n/2^23) (n%2^23) hne hnf
    have nh : n/2^23*2^23+n%2^23=n := by simpa only [Nat.mul_comm] using Nat.div_add_mod n (2^23)
    rwa [nh] at h
  obtain ⟨ys,hys,hscaled⟩ := scaled_word_value (Mufu.sqrt n) 115 (by omega) (by omega) (by omega)
  have hyeq : ys=yn := by rw [hyn] at hys;exact (Option.some.inj hys).symm
  rw [hyeq] at hscaled
  norm_num at hscaled
  refine ⟨Approx.mul (Mufu.sqrt n) 0x39800000,(f.val:Rat)/2^149,yn/4096,?_,hx,?_,?_⟩
  · unfold Approx.evaluate
    have ht : Mufu.magnitude f.val<0x00800000 := by unfold Mufu.magnitude;have:=f.isLt;omega
    simp only [String.reduceEq,Bool.false_eq_true,ite_false,or_self,ht,ite_true,hword]
  · simpa [div_eq_mul_inv] using hscaled
  have hroot : Real.sqrt (x:ℝ)=Real.sqrt ((f.val:Rat)/2^149:ℝ)*4096 := by
    rw [hxeq]
    push_cast
    rw [Real.sqrt_mul (by positivity)]
    norm_num
  rw [hroot] at hbound
  have hrel := relative_scale yn (Real.sqrt ((f.val:Rat)/2^149:ℝ)*4096) (1/4096) (by norm_num)
  have hc : (Real.sqrt ((f.val:Rat)/2^149:ℝ)*4096)*(1/4096)=Real.sqrt ((f.val:Rat)/2^149:ℝ) := by ring
  rw [hc] at hrel
  push_cast at hrel ⊢
  simpa only [div_eq_mul_inv,one_mul] using hrel.trans_le hbound

theorem ptx_rsq_subnormal (f : Fin (2^23)) (hf : 0 < f.val) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "rsq" false f.val=some out ∧
      value f.val=some x ∧ value out=some y ∧
      |(y:ℝ)-1/Real.sqrt (x:ℝ)|/|1/Real.sqrt (x:ℝ)|≤17/2^27 := by
  obtain ⟨hword,hx,hn,hlo,hhi⟩ := normalize f hf
  generalize hnorm : normalized f.val=n at *
  have hne : 1 ≤ n/2^23 ∧ n/2^23 ≤ 254 := by omega
  have hnf : n%2^23<2^23 := by omega
  let e : Fin 254 := ⟨n/2^23-1,by omega⟩
  let frac : Fin (2^23) := ⟨n%2^23,hnf⟩
  have hnword : (1+e.val)*2^23+frac.val=n := by dsimp [e,frac];omega
  obtain ⟨x,yn,hxin,hyn,hbound⟩ := rsq_relative e frac
  rw [hnword] at hxin hyn
  have hxeq : x=(f.val:Rat)/2^149*2^24 := by rw [hn] at hxin;exact (Option.some.inj hxin).symm
  have horange := RsqPtxBounds.output_normal_range (n/2^23) (n%2^23) hne hnf
  have nh : n/2^23*2^23+n%2^23=n := by simpa only [Nat.mul_comm] using Nat.div_add_mod n (2^23)
  rw [nh] at horange
  obtain ⟨ys,hys,hscaled⟩ := scaled_word_value (Mufu.rsq n) 139 (by omega) (by omega) (by omega)
  have hyeq : ys=yn := by rw [hyn] at hys;exact (Option.some.inj hys).symm
  rw [hyeq] at hscaled
  norm_num at hscaled
  refine ⟨Approx.mul (Mufu.rsq n) 0x45800000,(f.val:Rat)/2^149,yn*4096,?_,hx,hscaled,?_⟩
  · unfold Approx.evaluate
    have ht : Mufu.magnitude f.val<0x00800000 := by unfold Mufu.magnitude;have:=f.isLt;omega
    simp only [String.reduceEq,Bool.false_eq_true,ite_false,or_self,ht,ite_true,hword]
  have hroot : Real.sqrt (x:ℝ)=Real.sqrt ((f.val:Rat)/2^149:ℝ)*4096 := by
    rw [hxeq]
    push_cast
    rw [Real.sqrt_mul (by positivity)]
    norm_num
  rw [hroot] at hbound
  have hrel := relative_scale yn (1/(Real.sqrt ((f.val:Rat)/2^149:ℝ)*4096)) 4096 (by norm_num)
  have hc : (1/(Real.sqrt ((f.val:Rat)/2^149:ℝ)*4096))*4096=1/Real.sqrt ((f.val:Rat)/2^149:ℝ) := by field_simp
  rw [hc] at hrel
  push_cast at hrel ⊢
  exact hrel.trans_le hbound

#print axioms relative_scale
#print axioms ptx_sqrt_subnormal
#print axioms ptx_rsq_subnormal
end GB300Math.RootPtxSubnormal

namespace GB300Math.RootPtxSubnormal
open RootPtxScale Root32FloatLib

theorem ptx_sqrt_normal (e : Fin 254) (f : Fin (2^23)) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "sqrt" false ((1+e.val)*2^23+f.val)=some out ∧
      value ((1+e.val)*2^23+f.val)=some x ∧ value out=some y ∧
      |(y:ℝ)-Real.sqrt (x:ℝ)|/|Real.sqrt (x:ℝ)|≤1/2^23 := by
  obtain ⟨x,y,hx,hy,hb⟩ := sqrt_relative e f
  refine ⟨Mufu.sqrt ((1+e.val)*2^23+f.val),x,y,?_,hx,hy,hb⟩
  have ht : ¬Mufu.magnitude ((1+e.val)*2^23+f.val)<0x00800000 := by
    unfold Mufu.magnitude
    have := e.isLt
    have := f.isLt
    omega
  simp only [Approx.evaluate,String.reduceEq,Bool.false_eq_true,or_self,ite_false,ht]

theorem ptx_rsq_normal (e : Fin 254) (f : Fin (2^23)) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "rsq" false ((1+e.val)*2^23+f.val)=some out ∧
      value ((1+e.val)*2^23+f.val)=some x ∧ value out=some y ∧
      |(y:ℝ)-1/Real.sqrt (x:ℝ)|/|1/Real.sqrt (x:ℝ)|≤17/2^27 := by
  obtain ⟨x,y,hx,hy,hb⟩ := rsq_relative e f
  refine ⟨Mufu.rsq ((1+e.val)*2^23+f.val),x,y,?_,hx,hy,hb⟩
  have ht : ¬Mufu.magnitude ((1+e.val)*2^23+f.val)<0x00800000 := by
    unfold Mufu.magnitude
    have := e.isLt
    have := f.isLt
    omega
  simp only [Approx.evaluate,String.reduceEq,Bool.false_eq_true,or_self,ite_false,ht]

theorem ptx_sqrt_positive_finite (w : Nat) (hw : 0 < w ∧ w < 0x7f800000) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "sqrt" false w=some out ∧ value w=some x ∧ value out=some y ∧
      |(y:ℝ)-Real.sqrt (x:ℝ)|/|Real.sqrt (x:ℝ)|≤1/2^23 := by
  by_cases ht : w < 2^23
  · exact ptx_sqrt_subnormal ⟨w,ht⟩ hw.1
  · let e : Fin 254 := ⟨w/2^23-1,by omega⟩
    let f : Fin (2^23) := ⟨w%2^23,by omega⟩
    have hid : (1+e.val)*2^23+f.val=w := by dsimp [e,f];omega
    simpa only [hid] using ptx_sqrt_normal e f

theorem ptx_rsq_positive_finite (w : Nat) (hw : 0 < w ∧ w < 0x7f800000) :
    ∃ out : Nat, ∃ x y : Rat,
      Approx.evaluate "rsq" false w=some out ∧ value w=some x ∧ value out=some y ∧
      |(y:ℝ)-1/Real.sqrt (x:ℝ)|/|1/Real.sqrt (x:ℝ)|≤17/2^27 := by
  by_cases ht : w < 2^23
  · exact ptx_rsq_subnormal ⟨w,ht⟩ hw.1
  · let e : Fin 254 := ⟨w/2^23-1,by omega⟩
    let f : Fin (2^23) := ⟨w%2^23,by omega⟩
    have hid : (1+e.val)*2^23+f.val=w := by dsimp [e,f];omega
    simpa only [hid] using ptx_rsq_normal e f

theorem positive_finite_count : 255*2^23-1=2139095039 := by decide +kernel

#print axioms ptx_sqrt_normal
#print axioms ptx_rsq_normal
#print axioms ptx_sqrt_positive_finite
#print axioms ptx_rsq_positive_finite
#print axioms positive_finite_count
end GB300Math.RootPtxSubnormal

namespace GB300Math.RootPtxSubnormal

theorem smallest_subnormal_bits :
    Approx.evaluate "sqrt" false 1=some 0x1a3504f3 ∧
    Approx.evaluate "rsq" false 1=some 0x64b504f2 ∧
    Approx.evaluate "sqrt" true 1=some 0 ∧
    Approx.evaluate "rsq" true 1=some 0x7f800000 := by decide +kernel

theorem smallest_negative_subnormal_bits :
    Approx.evaluate "sqrt" false 0x80000001=some 0x7fffffff ∧
    Approx.evaluate "rsq" false 0x80000001=some 0x7fffffff ∧
    Approx.evaluate "sqrt" true 0x80000001=some 0x80000000 ∧
    Approx.evaluate "rsq" true 0x80000001=some 0xff800000 := by decide +kernel

#print axioms smallest_subnormal_bits
#print axioms smallest_negative_subnormal_bits
end GB300Math.RootPtxSubnormal
