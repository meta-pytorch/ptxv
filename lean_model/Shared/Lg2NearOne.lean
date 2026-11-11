/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SqrtBounds
import FixedToWord
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

set_option maxRecDepth 10000000
set_option maxHeartbeats 5000000

namespace GB300Math.Lg2NearOne

def lnTwo : ℚ := 69314718056/100000000000

theorem log_two_bound : |Real.log 2-(lnTwo : ℝ)| ≤ 1/10000000000 := by
  have h := Real.sum_range_sub_log_div_le (x := (1:ℝ)/3) (by norm_num) 11
  norm_num [Finset.sum_range_succ,lnTwo] at h ⊢
  rcases abs_le.mp h with ⟨hl,hu⟩
  exact abs_le.mpr ⟨by linarith,by linarith⟩

noncomputable def logPolynomial (t : ℝ) : ℝ := t-t^2/2+t^3/3-t^4/4

theorem log_near_one (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 1/64) :
    |Real.log (1+t)-logPolynomial t| ≤ 1/1000000000 := by
  have h := Real.abs_log_sub_add_sum_range_le (x := -t)
    (by rw [abs_neg,abs_of_nonneg ht.1];linarith [ht.2]) 4
  rw [abs_neg,abs_of_nonneg ht.1] at h
  have hpow : t^5 ≤ (1/64:ℝ)^5 := pow_le_pow_left₀ ht.1 ht.2 5
  have hrem : t^5/(1-t) ≤ 1/1000000000 := by
    apply (div_le_iff₀ (by linarith [ht.2])).mpr
    norm_num at hpow ⊢
    nlinarith only [hpow,ht.2]
  have he : (∑ i ∈ Finset.range 4, (-t)^(i+1)/(i+1)) + Real.log (1-(-t)) =
      Real.log (1+t)-logPolynomial t := by
    norm_num [Finset.sum_range_succ,logPolynomial]
    ring
  rw [he] at h
  exact h.trans hrem

def poly (k : ℚ) : ℚ :=
  (7*2^34+857088*4096+47272*k*2^22-11632*k^2)/2^60

def residualCoefficients : ℚ × ℚ × ℚ × ℚ × ℚ :=
  ((7*2^34+857088*4096)/2^60*lnTwo,
   47272*2^22/2^60*lnTwo-1/2^23,
   -11632/2^60*lnTwo+1/(2*2^46),-1/(3*2^69),1/(4*2^92))

def taylor (left : ℚ) : ℚ × ℚ × ℚ × ℚ × ℚ :=
  let (a,b,c,d,e) := residualCoefficients
  (a+b*left+c*left^2+d*left^3+e*left^4,
   b+2*c*left+3*d*left^2+4*e*left^3,c+3*d*left+6*e*left^2,d+4*e*left,e)

def lower (a b c d e width : ℚ) : ℚ :=
  a+min 0 (b*width)+min 0 (c*width^2)+min 0 (d*width^3)+min 0 (e*width^4)
def upper (a b c d e width : ℚ) : ℚ :=
  a+max 0 (b*width)+max 0 (c*width^2)+max 0 (d*width^3)+max 0 (e*width^4)

abbrev CellValid (cell : Nat) : Prop :=
  let (a,b,c,d,e) := taylor (cell*8192);
  -(1/10000000:ℚ) ≤ lower a b c d e 8191 ∧ upper a b c d e 8191 ≤ 1/10000000

set_option maxHeartbeats 0 in
theorem cells_valid : ∀ cell : Fin 16, CellValid cell.val := by
  intro cell
  fin_cases cell <;> norm_num [CellValid,taylor,residualCoefficients,lnTwo,lower,upper]

theorem term_box (c x width : ℝ) (hx : 0 ≤ x) (hw : x ≤ width) :
    min 0 (c*width) ≤ c*x ∧ c*x ≤ max 0 (c*width) := by
  by_cases hc : 0 ≤ c
  · exact ⟨(min_le_left _ _).trans (mul_nonneg hc hx),
      (mul_le_mul_of_nonneg_left hw hc).trans (le_max_right _ _)⟩
  · exact ⟨(min_le_right _ _).trans (mul_le_mul_of_nonpos_left hw (by linarith)),
      (mul_nonpos_of_nonpos_of_nonneg (by linarith) hx).trans (le_max_left _ _)⟩

theorem polynomial_box (a b c d e width x : ℚ) (hx : 0 ≤ x) (hw : x ≤ width) :
    lower a b c d e width ≤ a+b*x+c*x^2+d*x^3+e*x^4 ∧
    a+b*x+c*x^2+d*x^3+e*x^4 ≤ upper a b c d e width := by
  have h1 := term_box (b:ℝ) x width (by exact_mod_cast hx) (by exact_mod_cast hw)
  have h2 := term_box (c:ℝ) (x^2) (width^2) (by positivity) (by exact_mod_cast pow_le_pow_left₀ hx hw 2)
  have h3 := term_box (d:ℝ) (x^3) (width^3) (by positivity) (by exact_mod_cast pow_le_pow_left₀ hx hw 3)
  have h4 := term_box (e:ℝ) (x^4) (width^4) (by positivity) (by exact_mod_cast pow_le_pow_left₀ hx hw 4)
  have h1' : min 0 (b*width) ≤ b*x ∧ b*x ≤ max 0 (b*width) := by exact_mod_cast h1
  have h2' : min 0 (c*width^2) ≤ c*x^2 ∧ c*x^2 ≤ max 0 (c*width^2) := by exact_mod_cast h2
  have h3' : min 0 (d*width^3) ≤ d*x^3 ∧ d*x^3 ≤ max 0 (d*width^3) := by exact_mod_cast h3
  have h4' : min 0 (e*width^4) ≤ e*x^4 ∧ e*x^4 ≤ max 0 (e*width^4) := by exact_mod_cast h4
  dsimp [lower,upper]
  constructor <;> linarith only [h1'.1,h1'.2,h2'.1,h2'.2,h3'.1,h3'.2,h4'.1,h4'.2]

theorem poly_error (k : Nat) (hk : k < 131072) :
    |(poly k : ℝ)*(lnTwo:ℝ)-logPolynomial ((k:ℝ)/2^23)| ≤ 1/10000000 := by
  let cell : Fin 16 := ⟨k/8192,by omega⟩
  let left : ℚ := (cell.val:ℚ)*8192
  have ht : 0 ≤ (k:ℚ)-left ∧ (k:ℚ)-left ≤ 8191 := by
    have h : 0 ≤ (k:Int)-(cell.val:Int)*8192 ∧ (k:Int)-(cell.val:Int)*8192 ≤ 8191 := by dsimp [cell];omega
    dsimp only [left]
    exact_mod_cast h
  rcases hcoeff : taylor left with ⟨a,b,c,d,e⟩
  have hc := cells_valid cell
  change let (a,b,c,d,e) := taylor left; -(1/10000000:ℚ) ≤ lower a b c d e 8191 ∧ upper a b c d e 8191 ≤ 1/10000000 at hc
  rw [hcoeff] at hc
  have hb := polynomial_box a b c d e 8191 ((k:ℚ)-left) ht.1 ht.2
  have hid : a+b*((k:ℚ)-left)+c*((k:ℚ)-left)^2+d*((k:ℚ)-left)^3+e*((k:ℚ)-left)^4 =
      poly k*lnTwo-((k:ℚ)/2^23-((k:ℚ)/2^23)^2/2+((k:ℚ)/2^23)^3/3-((k:ℚ)/2^23)^4/4) := by
    simp only [taylor,Prod.mk.injEq] at hcoeff
    rcases hcoeff with ⟨rfl,rfl,rfl,rfl,rfl⟩
    norm_num [poly,residualCoefficients]
    ring
  have hq := abs_le.mpr ⟨hc.1.trans hb.1,hb.2.trans hc.2⟩
  rw [hid] at hq
  have hreal : ((|poly k*lnTwo-((k:ℚ)/2^23-((k:ℚ)/2^23)^2/2+((k:ℚ)/2^23)^3/3-((k:ℚ)/2^23)^4/4)|:ℚ):ℝ) ≤ ((1/10000000:ℚ):ℝ) := by exact_mod_cast hq
  push_cast at hreal
  exact hreal


theorem scalar_facts (k s : Nat) (hk : k < 131072)
    (hs : s ≤ k*k) (hdrop : k*k-s ≤ 4718593) :
    let z : Int := (30217216+48406528*k+(-11632*(s:Int))/4096)/4096
    4096 ≤ z ∧ z < 2^31 ∧ |(z:ℚ)/2^36-poly k| ≤ 1/20000000 := by
  let drop := k*k-s
  have hd : 0 ≤ (drop:Int) ∧ (drop:Int) ≤ 4718593 := by
    exact ⟨by positivity,by exact_mod_cast hdrop⟩
  have hsq : (s:Int) = (k:Int)^2-(drop:Int) := by
    have he : (drop:Int)+(s:Int) = (k:Int)*(k:Int) := by exact_mod_cast Nat.sub_add_cancel hs
    nlinarith only [he]
  let z : Int := (30217216+48406528*k+(-11632*(s:Int))/4096)/4096
  let inner : Int := 30217216+48406528*k+(-11632*(s:Int))/4096
  have hz : z = inner/4096 := rfl
  let r0 : Int := (-11632*(s:Int))%4096
  let r1 : Int := inner%4096
  have hr0 : 0 ≤ r0 ∧ r0 ≤ 4095 := by dsimp [r0];omega
  have hr1 : 0 ≤ r1 ∧ r1 ≤ 4095 := by dsimp [r1];omega
  have hdiv0 : -11632*(s:Int) = ((-11632*(s:Int))/4096)*4096+r0 := by dsimp [r0];omega
  have hdiv1 : inner = z*4096+r1 := by rw [hz];dsimp [r1];omega
  have he : z*2^24 = 7*2^34+857088*4096+47272*(k:Int)*2^22-11632*(k:Int)^2+
      11632*(drop:Int)-r0-r1*4096 := by
    dsimp [inner] at hdiv1
    linear_combination -hdiv0-4096*hdiv1-11632*hsq
  have hki : 0 ≤ (k:Int) ∧ (k:Int) ≤ 131071 := ⟨Nat.cast_nonneg _,by exact_mod_cast (Nat.le_of_lt_succ hk)⟩
  have hinc : 0 ≤ (k:Int)*(47272*2^22-11632*(k:Int)) := mul_nonneg hki.1 (by norm_num;linarith only [hki.2])
  have hzlo : 4096 ≤ z := by nlinarith only [he,hinc,hd.1,hr0.2,hr1.2]
  have hzhi : z < 2^31 := by nlinarith only [he,hki.2,hd.2,hr0.1,hr1.1,sq_nonneg (k:Int)]
  refine ⟨hzlo,hzhi,?_⟩
  have heq : (z:ℚ)*2^24 = 7*2^34+857088*4096+47272*(k:ℚ)*2^22-11632*(k:ℚ)^2+
      11632*(drop:ℚ)-(r0:ℚ)-(r1:ℚ)*4096 := by
    have h := congrArg (fun x : Int => (x:ℚ)) he
    push_cast at h
    norm_num at h ⊢
    exact h
  have herror : -(1/20000000:ℚ) ≤ (11632*(drop:ℚ)-(r0:ℚ)-(r1:ℚ)*4096)/2^60 ∧
      (11632*(drop:ℚ)-(r0:ℚ)-(r1:ℚ)*4096)/2^60 ≤ 1/20000000 := by
    have hdq : 0 ≤ (drop:ℚ) ∧ (drop:ℚ) ≤ 4718593 := by exact_mod_cast hd
    have h0q : 0 ≤ (r0:ℚ) ∧ (r0:ℚ) ≤ 4095 := by exact_mod_cast hr0
    have h1q : 0 ≤ (r1:ℚ) ∧ (r1:ℚ) ≤ 4095 := by exact_mod_cast hr1
    constructor <;> norm_num <;> linarith only [hdq.1,hdq.2,h0q.1,h0q.2,h1q.1,h1q.2]
  have hid : (z:ℚ)/2^36-poly k = (11632*(drop:ℚ)-(r0:ℚ)-(r1:ℚ)*4096)/2^60 := by
    dsimp [poly]
    linear_combination heq / 2^60
  change |(z:ℚ)/2^36-poly k| ≤ _
  rw [hid]
  exact abs_le.mpr herror


theorem interpolation_facts (k : Nat) (hk : k < 131072) :
    4096 ≤ Mufu.interpolate MufuTables.lg2 k ∧
    Mufu.interpolate MufuTables.lg2 k < 2^31 ∧
    |(Mufu.interpolate MufuTables.lg2 k : ℚ)/2^36-poly k| ≤ 1/20000000 := by
  have hz : Mufu.interpolate MufuTables.lg2 k =
      (30217216+48406528*(k:Int)+(-11632*(Fp64Seed.squareTruncated k 19:Int))/4096)/4096 := by
    dsimp [Mufu.interpolate,MufuTables.lg2]
    rw [show k/131072=0 by omega,show k%131072=k by omega]
    have hkm : (k:Int)%131072=k := by omega
    norm_num [hkm]
    congr 1
    ring
  rw [hz]
  exact scalar_facts k _ hk (by unfold Fp64Seed.squareTruncated;exact Nat.sub_le _ _) (SqrtBounds.square_drop_bound k)

theorem ideal_error (k : Nat) (hk : k < 131072) :
    |(Mufu.interpolate MufuTables.lg2 k : ℝ)/2^36-
      Real.log (1+(k:ℝ)/2^23)/Real.log 2| ≤ 1/5000000 := by
  let y : ℝ := (Mufu.interpolate MufuTables.lg2 k : ℝ)/2^36
  let p : ℝ := poly k
  let q : ℝ := lnTwo
  let t : ℝ := (k:ℝ)/2^23
  let l := Real.log (1+t)
  let L := Real.log 2
  have hf := interpolation_facts k hk
  have hy0 : 0 ≤ y := by
    dsimp [y]
    have h : (0:ℝ) ≤ (Mufu.interpolate MufuTables.lg2 k:Int) := by exact_mod_cast (show (0:Int) ≤ Mufu.interpolate MufuTables.lg2 k by omega)
    exact div_nonneg h (by norm_num)
  have hy1 : y ≤ 1/32 := by
    have h : (Mufu.interpolate MufuTables.lg2 k:ℝ) ≤ 2^31 := by exact_mod_cast (le_of_lt hf.2.1)
    dsimp [y];norm_num at h ⊢;linarith
  have herr : |y-p| ≤ 1/20000000 := by
    have h : ((|(Mufu.interpolate MufuTables.lg2 k:ℚ)/2^36-poly k|:ℚ):ℝ) ≤ ((1/20000000:ℚ):ℝ) := by exact_mod_cast hf.2.2
    push_cast at h
    exact h
  have ht : 0 ≤ t ∧ t ≤ 1/64 := by
    have h : (k:ℝ) ≤ 131072 := by exact_mod_cast (show k ≤ 131072 by omega)
    dsimp [t]
    norm_num only [Nat.reducePow]
    exact ⟨by positivity,by linarith only [h]⟩
  have hlog := log_near_one t ht
  have hpoly := poly_error k hk
  have hln := log_two_bound
  have hq0 : 0 ≤ q := by norm_num [q,lnTwo]
  have hq1 : q ≤ 7/10 := by norm_num [q,lnTwo]
  have hL : 69/100 ≤ L := by
    have hh := (abs_le.mp hln).1
    norm_num [lnTwo] at hh
    dsimp [L];linarith
  have h1 : |(y-p)*q| ≤ 7/200000000 := by
    rw [abs_mul,abs_of_nonneg hq0]
    have h := mul_le_mul herr hq1 hq0 (by norm_num : (0:ℝ) ≤ 1/20000000)
    norm_num at h ⊢
    exact h
  have h2 : |y*(L-q)| ≤ 1/320000000000 := by
    rw [abs_mul,abs_of_nonneg hy0]
    have h := mul_le_mul hy1 hln (abs_nonneg _) (by norm_num : (0:ℝ) ≤ 1/32)
    norm_num at h ⊢
    exact h
  have hident : y*L-l = (y-p)*q+y*(L-q)+(p*q-logPolynomial t)+(logPolynomial t-l) := by ring
  have hn : |y*L-l| ≤ 69/500000000 := by
    have a := abs_le.mp h1
    have b := abs_le.mp h2
    have c := abs_le.mp hpoly
    have d := abs_le.mp hlog
    change -(1/10000000:ℝ) ≤ p*q-logPolynomial t ∧ p*q-logPolynomial t ≤ 1/10000000 at c
    change -(1/1000000000:ℝ) ≤ l-logPolynomial t ∧ l-logPolynomial t ≤ 1/1000000000 at d
    exact abs_le.mpr ⟨by linarith only [hident,a.1,b.1,c.1,d.2],by linarith only [hident,a.2,b.2,c.2,d.1]⟩
  have hLp : 0 < L := by linarith
  have hid : y-l/L = (y*L-l)/L := by field_simp
  change |y-l/L| ≤ _
  rw [hid,abs_div,abs_of_pos hLp]
  apply (div_le_iff₀ hLp).mpr
  nlinarith only [hn,hL]

noncomputable def value (word : Nat) : ℝ :=
  if word = 0 then 0 else (RcpBounds.positiveNormalValue word : ℝ)

 theorem returned_bits (k : Nat) (hk : 0 < k ∧ k < 131072) :
    Mufu.lg2 (0x3f800000+k) = Mufu.fixedToWord (Mufu.interpolate MufuTables.lg2 k).toNat 36 := by
  have hf := interpolation_facts k hk.2
  have he : Mufu.exponent (0x3f800000+k) = 127 := by dsimp [Mufu.exponent];omega
  have hm : Mufu.fraction (0x3f800000+k) = k := by dsimp [Mufu.fraction];omega
  have hs := RcpBounds.small_sign (0x3f800000+k) (by omega)
  have hmag : ¬Mufu.magnitude (0x3f800000+k) > 0x7f800000 := by dsimp [Mufu.magnitude];omega
  simp only [Mufu.lg2,he,hm,hs,ite_eq_right hmag]
  norm_num [show k ≠ 0 by omega,show (0:Int) < Mufu.interpolate MufuTables.lg2 k by omega]


theorem returned_positive_normal (k : Nat) (hk : 0 < k ∧ k < 131072) :
    let word := Mufu.lg2 (0x3f800000+k)
    0 < word ∧ word < 2^31 ∧ 1 ≤ Mufu.exponent word ∧ Mufu.exponent word ≤ 254 := by
  have h := interpolation_facts k hk.2
  rw [returned_bits k hk]
  exact FixedToWord.positive_normal _ (by omega)

 theorem returned_error (k : Nat) (hk : k < 131072) :
    |value (Mufu.lg2 (0x3f800000+k))-Real.log (1+(k:ℝ)/2^23)/Real.log 2| ≤ 1/2^22 := by
  by_cases hz : k = 0
  · subst k
    norm_num [Mufu.lg2_one,value]
  have hf := interpolation_facts k hk
  let z := (Mufu.interpolate MufuTables.lg2 k).toNat
  have hzcast : (z:Int) = Mufu.interpolate MufuTables.lg2 k := by dsimp [z];omega
  have hzb : 4096 ≤ z ∧ z < 2^31 := by omega
  have hfix := FixedToWord.fixed_error z hzb
  have hid := ideal_error k hk
  have hnz : Mufu.fixedToWord z 36 ≠ 0 := ne_of_gt (FixedToWord.positive_normal z hzb).1
  rw [returned_bits k ⟨by omega,hk⟩]
  change |value (Mufu.fixedToWord z 36)-_| ≤ _
  rw [value,ite_eq_right hnz]
  have hfreal : 0 ≤ (z:ℝ)/2^36-(RcpBounds.positiveNormalValue (Mufu.fixedToWord z 36):ℝ) ∧
      (z:ℝ)/2^36-(RcpBounds.positiveNormalValue (Mufu.fixedToWord z 36):ℝ) ≤ 1/2^29 := by
    have ha : (0:ℝ) ≤ (((z:ℚ)/2^36-RcpBounds.positiveNormalValue (Mufu.fixedToWord z 36)):ℝ) := by exact_mod_cast hfix.1
    have hb : ((((z:ℚ)/2^36-RcpBounds.positiveNormalValue (Mufu.fixedToWord z 36)):ℚ):ℝ) ≤ ((1/2^29:ℚ):ℝ) := by exact_mod_cast hfix.2
    push_cast at ha hb
    exact ⟨ha,hb⟩
  have heq : (Mufu.interpolate MufuTables.lg2 k:ℝ) = z := by exact_mod_cast hzcast.symm
  rw [heq] at hid
  have hi := abs_le.mp hid
  exact abs_le.mpr ⟨by linarith only [hfreal.2,hi.1],by linarith only [hfreal.1,hi.2]⟩


theorem input_value (k : Nat) (hk : k < 131072) :
    (RcpBounds.positiveNormalValue (0x3f800000+k):ℝ) = 1+(k:ℝ)/2^23 := by
  have he : Mufu.exponent (0x3f800000+k) = 127 := by dsimp [Mufu.exponent];omega
  have hm : Mufu.fraction (0x3f800000+k) = k := by dsimp [Mufu.fraction];omega
  dsimp [RcpBounds.positiveNormalValue]
  rw [he,hm]
  norm_num
  ring

theorem bit_pattern_ideal_error (word : Nat)
    (hw : 0x3f800000 ≤ word ∧ word < 0x3f820000) :
    |value (Mufu.lg2 word)-Real.log (RcpBounds.positiveNormalValue word:ℝ)/Real.log 2| ≤ 1/2^22 := by
  let k := word-0x3f800000
  have hk : k < 131072 := by dsimp [k];omega
  have he : word = 0x3f800000+k := by dsimp [k];omega
  rw [he,input_value k hk]
  exact returned_error k hk


theorem witness_bits : Mufu.lg2 0x3f800001 = 0x3495f600 := by decide +kernel

theorem witness_value : value (Mufu.lg2 0x3f800001) = (19195:ℝ)/2^36 := by
  rw [witness_bits]
  norm_num [value,RcpBounds.positiveNormalValue,Mufu.exponent,Mufu.fraction]

theorem witness_ideal_range :
    0 < Real.log (1+(1:ℝ)/2^23)/Real.log 2 ∧
    Real.log (1+(1:ℝ)/2^23)/Real.log 2 ≤ 100/(69*2^23) := by
  have hlp : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hxp : 0 < Real.log (1+(1:ℝ)/2^23) := Real.log_pos (by norm_num)
  have hlo : 69/100 ≤ Real.log 2 := by
    have h := (abs_le.mp log_two_bound).1
    norm_num [lnTwo] at h
    linarith
  have hhi := Real.log_le_sub_one_of_pos (x := 1+(1:ℝ)/2^23) (by norm_num)
  refine ⟨div_pos hxp hlp,?_⟩
  apply (div_le_iff₀ hlp).mpr
  norm_num at hhi ⊢
  nlinarith only [hlo,hhi]

theorem witness_absolute_error :
    (1:ℝ)/2^24 < |value (Mufu.lg2 0x3f800001)-Real.log (1+(1:ℝ)/2^23)/Real.log 2| := by
  rw [witness_value]
  have h := witness_ideal_range.2
  have hl : 0 < (19195:ℝ)/2^36-Real.log (1+(1:ℝ)/2^23)/Real.log 2 := by linarith
  rw [abs_of_pos hl]
  linarith

theorem witness_relative_error :
    (1:ℝ)/2 < |value (Mufu.lg2 0x3f800001)-Real.log (1+(1:ℝ)/2^23)/Real.log 2| /
      (Real.log (1+(1:ℝ)/2^23)/Real.log 2) := by
  rw [witness_value]
  have h := witness_ideal_range
  have hl : 0 < (19195:ℝ)/2^36-Real.log (1+(1:ℝ)/2^23)/Real.log 2 := by linarith [h.2]
  rw [abs_of_pos hl]
  apply (lt_div_iff₀ h.1).mpr
  linarith [h.2]

#print axioms input_value
#print axioms bit_pattern_ideal_error
#print axioms interpolation_facts
#print axioms ideal_error
#print axioms returned_positive_normal
#print axioms returned_error
#print axioms witness_bits
#print axioms witness_absolute_error
#print axioms witness_relative_error
#print axioms log_two_bound
#print axioms log_near_one
#print axioms poly_error
end GB300Math.Lg2NearOne
