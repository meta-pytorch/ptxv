import RadianError
import RadianNegative
namespace GB300Math.RadianSigned
open FloatLib.Floats.Formats.BinaryInterchange
open RadianScale Root32FloatLib
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000

theorem negative_bound (e f : Nat) (he : 4≤e ∧ e≤254) (hf:f<2^23) :
    ∃ x ys yc : Rat,
      value (2^31+(e*2^23+f))=some x ∧
      value (Mufu.sincos (Approx.mul (2^31+(e*2^23+f)) 0x3e22f983 true) false)=some ys ∧
      value (Mufu.sincos (Approx.mul (2^31+(e*2^23+f)) 0x3e22f983 true) true)=some yc ∧
      |(ys:ℝ)-Real.sin (x:ℝ)|≤1/2^20+|(x:ℝ)|/2^22 ∧
      |(yc:ℝ)-Real.cos (x:ℝ)|≤1/2^20+|(x:ℝ)|/2^22 := by
  obtain ⟨x,y,hx,hy,hxp,he0,he1⟩:=RadianError.decoded_scale e f he hf
  have hr:=product_range f hf
  have hlo:=Nat.log2_self_le (by omega : product f≠0)
  have hhi:=Nat.lt_log2_self (n:=product f)
  generalize hl : (product f).log2=l at *
  have hl0 : 46≤l := by omega
  have hl1 : l≤47 := by omega
  have hm : 2^23≤product f/2^(l-23) ∧ product f/2^(l-23)<2^24 := by
    interval_cases l <;> norm_num at * <;> omega
  have hpword:=mul_radian_normal e f he hf
  have hnword:=mul_radian_negative e f he hf
  dsimp only at hpword hnword
  rw [hl] at hpword hnword
  have hop : Approx.mul (2^31+(e*2^23+f)) 0x3e22f983 true=
      2^31+Approx.mul (e*2^23+f) 0x3e22f983 true := by rw [hpword,hnword]
  have hw : Approx.mul (2^31+(e*2^23+f)) 0x3e22f983 true<2^32 := by rw [hnword];omega
  have hexp : Mufu.exponent (Approx.mul (2^31+(e*2^23+f)) 0x3e22f983 true)<255 := by rw [hnword];unfold Mufu.exponent;omega
  have hxn : value (2^31+(e*2^23+f))=some (-x) := by
    have h:=RcpSigned.signed_decode (e*2^23+f) x hx
    simpa [value,Nat.add_comm] using h
  have hyn : value (Approx.mul (2^31+(e*2^23+f)) 0x3e22f983 true)=some (-y) := by
    rw [hop]
    have h:=RcpSigned.signed_decode (Approx.mul (e*2^23+f) 0x3e22f983 true) y hy
    simpa [value,Nat.add_comm] using h
  obtain ⟨y',ys,yc,hy',hys,hyc,hs,hc⟩:=SineUniform.raw_finite_bound _ hw hexp
  have hyid : y'= -y := by have h : some y'=some (-y) := hy'.symm.trans hyn;exact Option.some.inj h
  subst y'
  have hphase:=RadianError.phase_error (x:ℝ) y (Rat.cast_nonneg.mpr hxp.le) he0 he1
  have hphase' : |2*Real.pi*(-(y:ℝ))-(-(x:ℝ))|≤|(x:ℝ)|/2^22 := by
    have habs : |(x:ℝ)|=(x:ℝ) := abs_of_nonneg (Rat.cast_nonneg.mpr hxp.le)
    rw [show 2*Real.pi*(-(y:ℝ))-(-(x:ℝ))= -(2*Real.pi*y-x) by ring,abs_neg,habs]
    exact hphase
  refine ⟨-x,ys,yc,hxn,hys,hyc,?_,?_⟩
  · have hdiff:=(Real.abs_sin_sub_sin_le (2*Real.pi*(-(y:ℝ))) (-(x:ℝ))).trans hphase'
    have htri:= (abs_sub_le (ys:ℝ) (Real.sin (2*Real.pi*(-(y:ℝ)))) (Real.sin (-(x:ℝ)))).trans
      (add_le_add (by simpa using hs) hdiff)
    simpa using htri
  · have hdiff:=(Real.abs_cos_sub_cos_le (2*Real.pi*(-(y:ℝ))) (-(x:ℝ))).trans hphase'
    have htri:= (abs_sub_le (yc:ℝ) (Real.cos (2*Real.pi*(-(y:ℝ)))) (Real.cos (-(x:ℝ)))).trans
      (add_le_add (by simpa using hc) hdiff)
    simpa using htri

def word (negative : Bool) (e f : Nat) : Nat := (if negative then 2^31 else 0)+(e*2^23+f)

theorem operational_bound (negative : Bool) (e f : Nat) (he : 4≤e ∧ e≤254) (hf:f<2^23) :
    ∃ x ys yc : Rat,
      value (word negative e f)=some x ∧
      Approx.evaluate "sin" false (word negative e f)=some (Mufu.sincos (Approx.mul (word negative e f) 0x3e22f983 true) false) ∧
      Approx.evaluate "cos" false (word negative e f)=some (Mufu.sincos (Approx.mul (word negative e f) 0x3e22f983 true) true) ∧
      value (Mufu.sincos (Approx.mul (word negative e f) 0x3e22f983 true) false)=some ys ∧
      value (Mufu.sincos (Approx.mul (word negative e f) 0x3e22f983 true) true)=some yc ∧
      |(ys:ℝ)-Real.sin (x:ℝ)|≤1/2^20+|(x:ℝ)|/2^22 ∧
      |(yc:ℝ)-Real.cos (x:ℝ)|≤1/2^20+|(x:ℝ)|/2^22 := by
  have hs : ∀w, Approx.evaluate "sin" false w=some (Mufu.sincos (Approx.mul w 0x3e22f983 true) false) := by intro w;rfl
  have hc : ∀w, Approx.evaluate "cos" false w=some (Mufu.sincos (Approx.mul w 0x3e22f983 true) true) := by intro w;rfl
  cases negative
  · obtain ⟨x,ys,yc,hx,hys,hyc,hse,hce⟩:=RadianError.normal_bound e f he hf
    obtain ⟨x',y,hx',_,hxp,_,_⟩:=RadianError.decoded_scale e f he hf
    have hid : x'=x := Option.some.inj (hx'.symm.trans hx)
    subst x'
    have hax : |(x:ℝ)|=(x:ℝ) := abs_of_nonneg (Rat.cast_nonneg.mpr hxp.le)
    refine ⟨x,ys,yc,?_,hs _,hc _,?_,?_,?_,?_⟩
    · simpa only [word,Bool.false_eq_true,ite_false,Nat.zero_add] using hx
    · simpa only [word,Bool.false_eq_true,ite_false,Nat.zero_add] using hys
    · simpa only [word,Bool.false_eq_true,ite_false,Nat.zero_add] using hyc
    · rwa [hax]
    · rwa [hax]
  · obtain ⟨x,ys,yc,hx,hys,hyc,hse,hce⟩:=negative_bound e f he hf
    refine ⟨x,ys,yc,?_,hs _,hc _,?_,?_,hse,hce⟩
    · simpa only [word,ite_true] using hx
    · simpa only [word,ite_true] using hys
    · simpa only [word,ite_true] using hyc

theorem covered_word_count : 2*(254-4+1)*2^23=4211081216 := by decide +kernel
#print axioms negative_bound
#print axioms operational_bound
#print axioms covered_word_count
end GB300Math.RadianSigned
