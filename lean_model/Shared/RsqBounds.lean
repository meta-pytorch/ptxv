import RsqCells

set_option maxRecDepth 1000000
set_option maxHeartbeats 10000000

namespace GB300Math.RsqBounds

theorem cell_error (p : Parameters) (left k dropped : Int)
    (hcell : CellBound p left) (hl : 0 ≤ left ∧ left+2047 ≤ 131071)
    (hw : left ≤ k ∧ k ≤ left+2047)
    (hd : 0 ≤ dropped ∧ dropped ≤ 4718593) :
    let z := (p.a+p.b*k+p.c*(k*k-dropped))/2^34
    2^23 ≤ z ∧ z ≤ 2^24 ∧
    (p.m+p.step*k)*(z-2)^2 ≤ 2^71 ∧ 2^71 ≤ (p.m+p.step*k)*(z+2)^2 := by
  rcases ht : taylor p left with ⟨q0,q1,q2,q3,q4,q5⟩
  simp only [CellBound,ht] at hcell
  rcases hcell with ⟨hc,hb,hm,hs,hnlo,hnhi,hmin,hlo,hhi⟩
  have hbox := polynomial_box q0 q1 q2 q3 q4 q5 (k-left) 2047 (by omega) (by omega)
  have heq : q0+q1*(k-left)+q2*(k-left)^2+q3*(k-left)^3+q4*(k-left)^4+q5*(k-left)^5 =
      (p.m+p.step*k)*(numerator p k)^2-2^139 := by
    have vals := ht
    simp only [taylor,Prod.mk.injEq] at vals
    rcases vals with ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩
    dsimp [numerator]
    ring
  rw [heq] at hbox
  let q := numerator p k
  let n := p.a+p.b*k+p.c*(k*k-dropped)
  let z := n/2^34
  let delta := p.c*dropped+n%2^34
  let m := p.m+p.step*k
  have hqlo := numerator_decreasing p k 131071 hc hb (by omega) (by omega) (by omega)
  have hqhi := numerator_decreasing p 0 k hc hb (by omega) (by omega) (by omega)
  have hqleft := numerator_decreasing p left k hc hb hl.1 hw.1 (by omega)
  have hqright := numerator_decreasing p k (left+2047) hc hb (by omega) hw.2 hl.2
  have hd0 := mul_nonneg hc hd.1
  have hd1 := mul_le_mul_of_nonneg_left hd.2 hc
  have hr : 0 ≤ n%2^34 ∧ n%2^34 ≤ 2^34-1 := by omega
  have hdelta : 0 ≤ delta ∧ delta ≤ loss p := by dsimp [delta,loss];omega
  have hn : z*2^34 = q-delta := by
    have hv : n = z*2^34+n%2^34 := by dsimp [z];omega
    dsimp [q,n,delta,numerator] at *
    nlinarith only [hv]
  have hzb : 2^23 ≤ z ∧ z ≤ 2^24 := by
    dsimp [numerator,q] at *
    constructor <;> omega
  have hzmin : minimum p left ≤ z := by
    dsimp [minimum]
    omega
  have hmlo : 0 < p.m+p.step*left ∧ p.m+p.step*left ≤ m := by
    have ha := mul_nonneg hs hl.1
    have hb := mul_le_mul_of_nonneg_left hw.1 hs
    dsimp [m];omega
  have hmhi : m ≤ p.m+p.step*(left+2047) := by
    dsimp [m]
    have h := mul_le_mul_of_nonneg_left hw.2 hs
    omega
  have hq0 : 0 ≤ q := by omega
  have hqleft0 : 0 ≤ numerator p left := by omega
  have hloss0 : 0 ≤ q^2-(q-delta)^2 := by
    have hp := mul_nonneg hdelta.1 (show 0 ≤ 2*q-delta by omega)
    nlinarith only [hp]
  have hloss1 : q^2-(q-delta)^2 ≤ 2*q*loss p := by
    have hp := mul_nonneg (show 0 ≤ 2*q by omega) (show 0 ≤ loss p-delta by omega)
    nlinarith [sq_nonneg delta]
  have hscale0 := mul_nonneg (show 0 ≤ m by omega) hloss0
  have hscale1 := mul_le_mul_of_nonneg_left hloss1 (show 0 ≤ m by omega)
  have hproduct := mul_le_mul hmhi hqleft hq0 (show 0 ≤ p.m+p.step*(left+2047) by omega)
  have hproduct2 := mul_le_mul_of_nonneg_right hproduct (show 0 ≤ 2*loss p by omega)
  have hres : -(p.m+p.step*left)*(4*minimum p left-4) ≤ m*z^2-2^71 ∧
      m*z^2-2^71 ≤ (p.m+p.step*left)*(4*minimum p left-4) := by
    have hid : (m*z^2-2^71)*2^68 = (m*q^2-2^139)-m*(q^2-(q-delta)^2) := by
      rw [show (2 : Int)^68 = (2^34)^2 by norm_num]
      calc
        _ = m*(z*2^34)^2-2^139 := by ring
        _ = _ := by rw [hn];ring
    dsimp [m,q] at hid hscale0 hscale1 hproduct2 ⊢
    constructor <;> nlinarith only [hid,hbox.1,hbox.2,hlo,hhi,hscale0,hscale1,hproduct2]
  have hbound := mul_le_mul hmlo.2 (show 4*minimum p left-4 ≤ 4*z-4 by omega)
    (show 0 ≤ 4*minimum p left-4 by omega) (show 0 ≤ m by omega)
  change 2^23 ≤ z ∧ z ≤ 2^24 ∧ _
  refine ⟨hzb.1,hzb.2,?_,?_⟩
  · nlinarith only [hres.2,hbound]
  · have hp := mul_nonneg (show 0 ≤ m by omega) (show (0 : Int) ≤ 8 by omega)
    nlinarith only [hres.1,hbound,hp]

theorem interpolate_error (index : Nat) (hi : index < 2^24) :
    let z := Mufu.interpolate MufuTables.rsq index
    let m : Int := (2^23+index%2^23)*2^(index/2^23)
    2^23 ≤ z ∧ z ≤ 2^24 ∧ m*(z-2)^2 ≤ 2^71 ∧ 2^71 ≤ m*(z+2)^2 := by
  let segment : Fin 128 := ⟨index/131072,by omega⟩
  let k := index%131072
  let cell : Fin 64 := ⟨k/2048,by dsimp [k];omega⟩
  let drop := k*k-Fp64Seed.squareTruncated k 19
  have hs : Fp64Seed.squareTruncated k 19 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hd : 0 ≤ (drop : Int) ∧ (drop : Int) ≤ 4718593 := by
    constructor
    · exact Int.natCast_nonneg _
    · exact_mod_cast SqrtBounds.square_drop_bound k
  have hl : 0 ≤ (cell.val : Int)*2048 ∧ (cell.val : Int)*2048+2047 ≤ 131071 := by
    have := cell.isLt
    omega
  have hw : (cell.val : Int)*2048 ≤ (k : Int) ∧ (k : Int) ≤ (cell.val : Int)*2048+2047 := by
    dsimp [cell];omega
  have h := cell_error (parameters segment.val) ((cell.val : Int)*2048) k drop
    (cells_valid segment cell) hl hw hd
  have hsquare : (k : Int)*(k : Int)-(drop : Int) = Fp64Seed.squareTruncated k 19 := by
    have hh := Nat.sub_add_cancel hs
    have hc : (drop : Int)+(Fp64Seed.squareTruncated k 19 : Int) = (k : Int)*(k : Int) := by
      exact_mod_cast hh
    omega
  have hpar : segment.val/64 = index/2^23 := by dsimp [segment];omega
  have hrem : (segment.val%64 : Int)*131072+(k : Int) = index%2^23 := by
    dsimp [segment,k];omega
  have hm : (parameters segment.val).m+(parameters segment.val).step*(k : Int) =
      (2^23+(index%2^23 : Nat))*2^(index/2^23) := by
    dsimp [parameters]
    rw [hpar]
    push_cast
    calc
      _ = (8388608+((segment.val%64 : Int)*131072+(k : Int)))*2^(index/2^23) := by ring
      _ = _ := by rw [hrem];norm_num
  dsimp only at h ⊢
  rw [hm,hsquare] at h
  convert h using 1 <;>
    dsimp [Mufu.interpolate,MufuTables.rsq,parameters,segment,k] <;> norm_num <;> ring_nf

noncomputable def normalizedOutput (index : Nat) : ℝ :=
  if index = 0 then 1 else (Mufu.interpolate MufuTables.rsq index : ℝ)/2^24

theorem reciprocal_sqrt_interval (x y e : ℝ) (hx : 0 < x) (he : 0 ≤ e)
    (hy : e ≤ y) (hlo : x*(y-e)^2 ≤ 1) (hhi : 1 ≤ x*(y+e)^2) :
    |y-1/Real.sqrt x| ≤ e := by
  have hs : 0 < Real.sqrt x := Real.sqrt_pos.mpr hx
  have hsq := Real.sq_sqrt hx.le
  have hleft : (Real.sqrt x*(y-e))^2 = x*(y-e)^2 := by rw [mul_pow,hsq]
  have hright : (Real.sqrt x*(y+e))^2 = x*(y+e)^2 := by rw [mul_pow,hsq]
  have hln : 0 ≤ Real.sqrt x*(y-e) := mul_nonneg hs.le (by linarith)
  have hrn : 0 ≤ Real.sqrt x*(y+e) := mul_nonneg hs.le (by linarith)
  have hl : Real.sqrt x*(y-e) ≤ 1 := by nlinarith only [hleft,hlo,hln]
  have hr : 1 ≤ Real.sqrt x*(y+e) := by nlinarith only [hright,hhi,hrn]
  have hl' : y-e ≤ 1/Real.sqrt x := (le_div_iff₀ hs).mpr (by nlinarith only [hl])
  have hr' : 1/Real.sqrt x ≤ y+e := (div_le_iff₀ hs).mpr (by nlinarith only [hr])
  exact abs_le.mpr ⟨by linarith,by linarith⟩

theorem rsq_absolute_error (index : Nat) (hi : index < 2^24) :
    |normalizedOutput index-1/Real.sqrt (SqrtBounds.normalizedInput index)| ≤ 1/2^23 := by
  by_cases hz : index = 0
  · simp [normalizedOutput,hz,SqrtBounds.normalizedInput]
  have h := interpolate_error index hi
  dsimp only at h
  have hlo := (Int.cast_le (R := ℝ)).mpr h.2.2.1
  have hhi := (Int.cast_le (R := ℝ)).mpr h.2.2.2
  push_cast at hlo hhi
  have hm : (index : Int)%8388608 = ((index%8388608 : Nat) : Int) := by omega
  rw [hm] at hlo hhi
  simp only [Int.cast_natCast] at hlo hhi
  have hy : (1 : ℝ)/2 ≤ normalizedOutput index := by
    dsimp [normalizedOutput]
    rw [ite_eq_right hz]
    have hr : (2 : ℝ)^23 ≤ (Mufu.interpolate MufuTables.rsq index : ℝ) := by exact_mod_cast h.1
    norm_num at hr ⊢
    linarith
  have hl : SqrtBounds.normalizedInput index*(normalizedOutput index-1/2^23)^2 ≤ 1 := by
    dsimp [SqrtBounds.normalizedInput,normalizedOutput]
    rw [ite_eq_right hz]
    norm_num at hlo ⊢
    nlinarith only [hlo]
  have hh : 1 ≤ SqrtBounds.normalizedInput index*(normalizedOutput index+1/2^23)^2 := by
    dsimp [SqrtBounds.normalizedInput,normalizedOutput]
    rw [ite_eq_right hz]
    norm_num at hhi ⊢
    nlinarith only [hhi]
  apply reciprocal_sqrt_interval _ _ _ _ (by positivity) (by norm_num at hy ⊢;linarith) hl hh
  dsimp [SqrtBounds.normalizedInput]
  positivity

open SqrtBounds (positiveNormalValue scaleExponent mantissaIndex outputScale)

theorem output_word_scaled (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    Mufu.rsq (exponent*2^23+fraction) =
      if mantissaIndex exponent fraction = 0 then (127-scaleExponent exponent).toNat*2^23
      else
        let z := (Mufu.interpolate MufuTables.rsq (mantissaIndex exponent fraction)).toNat
        ((if z = 2^24 then 127 else 126)-scaleExponent exponent).toNat*2^23+z%2^23 := by
  have hex : Mufu.exponent (exponent*2^23+fraction) = exponent := by dsimp [Mufu.exponent];omega
  have hfr : Mufu.fraction (exponent*2^23+fraction) = fraction := by dsimp [Mufu.fraction];omega
  have hs := RcpBounds.small_sign (exponent*2^23+fraction) (by omega)
  have hmag : ¬Mufu.magnitude (exponent*2^23+fraction) > 0x7f800000 := by dsimp [Mufu.magnitude];omega
  have hr : Mufu.rootIndex (exponent*2^23+fraction) =
      (mantissaIndex exponent fraction,scaleExponent exponent) := by
    unfold Mufu.rootIndex
    rw [hex,hfr]
    rfl
  rw [Mufu.rsq,ite_eq_right hmag]
  simp only [hex,hs]
  rw [ite_eq_right (show exponent ≠ 0 by omega),
    ite_eq_right (show ¬(0 : Nat) ≠ 0 by simp),ite_eq_right (show exponent ≠ 255 by omega)]
  rw [hr]

theorem output_decode_scaled (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    positiveNormalValue (Mufu.rsq (exponent*2^23+fraction))*outputScale exponent =
      normalizedOutput (mantissaIndex exponent fraction) := by
  have hp := SqrtBounds.index_properties exponent fraction hf
  have ht : -63 ≤ scaleExponent exponent ∧ scaleExponent exponent ≤ 63 := by dsimp [scaleExponent];omega
  rw [output_word_scaled exponent fraction he hf]
  by_cases hi : mantissaIndex exponent fraction = 0
  · rw [ite_eq_left hi]
    have hdec := RcpBounds.normal_decode (127-scaleExponent exponent).toNat 0 (by omega) (by norm_num)
    simp only [Nat.add_zero] at hdec
    unfold positiveNormalValue
    rw [hdec]
    push_cast
    have hc : (((127-scaleExponent exponent).toNat : Nat) : Int) = 127-scaleExponent exponent := by omega
    rw [hc]
    dsimp [outputScale]
    rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    rw [show 127-scaleExponent exponent-150+scaleExponent exponent = (-23 : Int) by omega]
    norm_num [normalizedOutput,hi]
  · rw [ite_eq_right hi]
    let z := (Mufu.interpolate MufuTables.rsq (mantissaIndex exponent fraction)).toNat
    have hz := interpolate_error (mantissaIndex exponent fraction) hp.1
    have hzb : 2^23 ≤ z ∧ z ≤ 2^24 := by dsimp [z];omega
    have hzi : (z : Int) = Mufu.interpolate MufuTables.rsq (mantissaIndex exponent fraction) := by dsimp [z];omega
    change positiveNormalValue (((if z = 2^24 then 127 else 126)-scaleExponent exponent).toNat*2^23+z%2^23)*_ = _
    by_cases htop : z = 2^24
    · rw [ite_eq_left htop]
      have hm : z%2^23 = 0 := by omega
      rw [hm,Nat.add_zero]
      have hdec := RcpBounds.normal_decode (127-scaleExponent exponent).toNat 0 (by omega) (by norm_num)
      simp only [Nat.add_zero] at hdec
      unfold positiveNormalValue
      rw [hdec]
      push_cast
      have hc : (((127-scaleExponent exponent).toNat : Nat) : Int) = 127-scaleExponent exponent := by omega
      rw [hc]
      dsimp [outputScale]
      rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      have hv : Mufu.interpolate MufuTables.rsq (mantissaIndex exponent fraction) = 2^24 := by omega
      rw [show 127-scaleExponent exponent-150+scaleExponent exponent = (-23 : Int) by omega]
      norm_num [normalizedOutput,hi,hv]
    · rw [ite_eq_right htop]
      unfold positiveNormalValue
      rw [RcpBounds.normal_decode _ _ (by omega) (by omega)]
      have hm : z%2^23 = z-2^23 := by omega
      have hsub : ((z-2^23 : Nat) : ℚ) = (z : ℚ)-2^23 := by rw [Nat.cast_sub hzb.1];norm_num
      rw [hm,hsub]
      push_cast
      have hc : (((126-scaleExponent exponent).toNat : Nat) : Int) = 126-scaleExponent exponent := by omega
      rw [hc]
      dsimp [outputScale]
      rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
      have hzr : (Mufu.interpolate MufuTables.rsq (mantissaIndex exponent fraction) : ℝ) = z := by exact_mod_cast hzi.symm
      rw [show 126-scaleExponent exponent-150+scaleExponent exponent = (-24 : Int) by omega]
      norm_num [normalizedOutput,hi,hzr]
      ring

theorem rsq_bits_scaled_absolute_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rsq (exponent*2^23+fraction))-
      1/Real.sqrt (positiveNormalValue (exponent*2^23+fraction))| ≤
      1/(2^23*outputScale exponent) := by
  have hp := SqrtBounds.index_properties exponent fraction hf
  have hs : 0 < outputScale exponent := by unfold outputScale;positivity
  have ho := output_decode_scaled exponent fraction he hf
  have hxo : positiveNormalValue (Mufu.rsq (exponent*2^23+fraction)) =
      normalizedOutput (mantissaIndex exponent fraction)/outputScale exponent := by
    apply (eq_div_iff hs.ne').mpr
    exact ho
  rw [hxo,SqrtBounds.input_decode_scaled exponent fraction (by omega) hf,
    Real.sqrt_mul (show 0 ≤ SqrtBounds.normalizedInput (mantissaIndex exponent fraction) by
      unfold SqrtBounds.normalizedInput;positivity),Real.sqrt_sq hs.le]
  rw [show 1/(Real.sqrt (SqrtBounds.normalizedInput (mantissaIndex exponent fraction))*outputScale exponent) =
    (1/Real.sqrt (SqrtBounds.normalizedInput (mantissaIndex exponent fraction)))/outputScale exponent by ring,
    ← sub_div,abs_div,abs_of_pos hs]
  simpa only [div_div] using div_le_div_of_nonneg_right (rsq_absolute_error _ hp.1) hs.le

#print axioms cells_valid
#print axioms rsq_absolute_error
#print axioms rsq_bits_scaled_absolute_error
end GB300Math.RsqBounds
