/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RadianScale
import SineUniform
namespace GB300Math.RadianError
open FloatLib.Floats.Formats.BinaryInterchange
open RadianScale Root32FloatLib
set_option maxRecDepth 16000
set_option maxHeartbeats 6000000

theorem quantization (p l : Nat) (hl : 46≤l ∧ l≤47)
    (hp : 2^l≤p ∧ p<2^(l+1)) :
    0≤(p:ℝ)/2^26-(p/2^(l-23):Nat)*(2:ℝ)^((l:Int)-49) ∧
    (p:ℝ)/2^26-(p/2^(l-23):Nat)*(2:ℝ)^((l:Int)-49)≤((p:ℝ)/2^26)/2^23 := by
  obtain ⟨hl0,hl1⟩ := hl
  interval_cases l
  · have hd : p/8388608*8388608≤p ∧ p<(p/8388608+1)*8388608 := by omega
    have hd0 : (p/8388608:Nat)*8388608≤(p:ℝ) := by exact_mod_cast hd.1
    have hd1 : (p:ℝ)<((p/8388608:Nat)+1)*8388608 := by exact_mod_cast hd.2
    have hp0 : (2:ℝ)^46≤p := by exact_mod_cast hp.1
    norm_num at *
    have hpr : (70368744177664:ℝ)≤(p:ℝ) := by exact_mod_cast hp.1
    constructor <;> nlinarith only [hd0,hd1,hpr]
  · have hd : p/16777216*16777216≤p ∧ p<(p/16777216+1)*16777216 := by omega
    have hd0 : (p/16777216:Nat)*16777216≤(p:ℝ) := by exact_mod_cast hd.1
    have hd1 : (p:ℝ)<((p/16777216:Nat)+1)*16777216 := by exact_mod_cast hd.2
    have hp0 : (2:ℝ)^47≤p := by exact_mod_cast hp.1
    norm_num at *
    have hpr : (140737488355328:ℝ)≤(p:ℝ) := by exact_mod_cast hp.1
    constructor <;> nlinarith only [hd0,hd1,hpr]

theorem decoded_scale (e f : Nat) (he : 4≤e ∧ e≤254) (hf:f<2^23) :
    ∃ x y : Rat,
      value (e*2^23+f)=some x ∧
      value (Approx.mul (e*2^23+f) 0x3e22f983 true)=some y ∧
      0<x ∧ 0≤(x:ℝ)*(10680707/2^26)-(y:ℝ) ∧
      (x:ℝ)*(10680707/2^26)-(y:ℝ)≤((x:ℝ)*(10680707/2^26))/2^23 := by
  have hr:=product_range f hf
  have hl0:=Nat.log2_self_le (by omega : product f≠0)
  have hl1:=Nat.lt_log2_self (n:=product f)
  generalize hh : (product f).log2=l at *
  have hlo : 46≤l := by omega
  have hhi : l≤47 := by omega
  have hm : 2^23≤product f/2^(l-23) ∧ product f/2^(l-23)<2^24 := by
    interval_cases l <;> norm_num at * <;> omega
  have hout : 1≤l+e-49 ∧ l+e-49≤254 := by omega
  have hfrac : product f/2^(l-23)-2^23<2^23 := by omega
  have hin:=RootPtxScale.normal_value e f (by omega) hf
  have hop:=mul_radian_normal e f he hf
  dsimp only at hop
  rw [hh] at hop
  have hd:=RootPtxScale.normal_value (l+e-49) (product f/2^(l-23)-2^23) hout hfrac
  have hmant : 2^23+((product f/2^(l-23)-2^23:Nat):Rat)=(product f/2^(l-23):Nat) := by exact_mod_cast (show 2^23+(product f/2^(l-23)-2^23)=product f/2^(l-23) by omega)
  rw [hmant] at hd
  let x:Rat:=(2^23+(f:Rat))*2^((e:Int)-150)
  let y:Rat:=(product f/2^(l-23):Nat)*2^(((l+e-49:Nat):Int)-150)
  refine ⟨x,y,hin,?_,?_,?_,?_⟩
  · rw [hop];exact hd
  · change 0<(2^23+(f:Rat))*2^((e:Int)-150)
    exact mul_pos (by have h : (0:Rat)≤f := Nat.cast_nonneg f;norm_num;nlinarith only[h]) (zpow_pos (by norm_num) _)
  all_goals
    have hq:=quantization (product f) l (by omega) ⟨hl0,hl1⟩
    have ht : 0<(2:ℝ)^((e:Int)-150) := zpow_pos (by norm_num) _
    have hexp : ((l+e-49:Nat):Int)-150=((e:Int)-150)+((l:Int)-49) := by omega
    have hy : (y:ℝ)=(product f/2^(l-23):Nat)*(2:ℝ)^((l:Int)-49)*(2:ℝ)^((e:Int)-150) := by
      dsimp [y]
      push_cast
      rw [hexp,zpow_add₀ (by norm_num : (2:ℝ)≠0)]
      ring
    have hx : (x:ℝ)*(10680707/2^26)=((product f:ℝ)/2^26)*(2:ℝ)^((e:Int)-150) := by
      dsimp [x,product];push_cast;ring
    rw [hx,hy]
  · nlinarith only [mul_nonneg hq.1 ht.le]
  · nlinarith only [mul_le_mul_of_nonneg_right hq.2 ht.le]

theorem phase_error (x y : ℝ) (hx : 0≤x)
    (h0 : 0≤x*(10680707/2^26)-y)
    (h1 : x*(10680707/2^26)-y≤(x*(10680707/2^26))/2^23) :
    |2*Real.pi*y-x|≤x/2^22 := by
  have hp0:=Real.pi_gt_d20
  have hp1:=Real.pi_lt_d20
  have hmul : 0≤2*Real.pi := by positivity
  have hc0 : 1-1/2^22≤2*Real.pi*(10680707/2^26)*(1-1/2^23) := by nlinarith only [hp0]
  have hc1 : 2*Real.pi*(10680707/2^26)≤1 := by nlinarith only [hp1]
  have hy0 : x*(10680707/2^26)*(1-1/2^23)≤y := by nlinarith only [h1]
  have hy1 : y≤x*(10680707/2^26) := by linarith only [h0]
  have hlo:=mul_le_mul_of_nonneg_left hc0 hx
  have hhi:=mul_le_mul_of_nonneg_left hc1 hx
  have hlom:=mul_le_mul_of_nonneg_left hy0 hmul
  have hhim:=mul_le_mul_of_nonneg_left hy1 hmul
  rw [abs_le]
  constructor <;> nlinarith only [hlo,hhi,hlom,hhim]

theorem normal_bound (e f : Nat) (he : 4≤e ∧ e≤254) (hf:f<2^23) :
    ∃ x ys yc : Rat,
      value (e*2^23+f)=some x ∧
      value (Mufu.sincos (Approx.mul (e*2^23+f) 0x3e22f983 true) false)=some ys ∧
      value (Mufu.sincos (Approx.mul (e*2^23+f) 0x3e22f983 true) true)=some yc ∧
      |(ys:ℝ)-Real.sin (x:ℝ)|≤1/2^20+(x:ℝ)/2^22 ∧
      |(yc:ℝ)-Real.cos (x:ℝ)|≤1/2^20+(x:ℝ)/2^22 := by
  obtain ⟨x,y,hx,hy,hxp,he0,he1⟩:=decoded_scale e f he hf
  have hr:=product_range f hf
  have hlo:=Nat.log2_self_le (by omega : product f≠0)
  have hhi:=Nat.lt_log2_self (n:=product f)
  generalize hl : (product f).log2=l at *
  have hl0 : 46≤l := by omega
  have hl1 : l≤47 := by omega
  have hm : 2^23≤product f/2^(l-23) ∧ product f/2^(l-23)<2^24 := by
    interval_cases l <;> norm_num at * <;> omega
  have hword:=mul_radian_normal e f he hf
  dsimp only at hword
  rw [hl] at hword
  have hw : Approx.mul (e*2^23+f) 0x3e22f983 true<2^32 := by rw [hword];omega
  have hexp : Mufu.exponent (Approx.mul (e*2^23+f) 0x3e22f983 true)<255 := by rw [hword];unfold Mufu.exponent;omega
  obtain ⟨y',ys,yc,hy',hys,hyc,hs,hc⟩:=SineUniform.raw_finite_bound _ hw hexp
  have hyid : y'=y := by have h : some y'=some y := hy'.symm.trans hy;exact Option.some.inj h
  subst y'
  have hphase:=phase_error (x:ℝ) y (Rat.cast_nonneg.mpr hxp.le) he0 he1
  refine ⟨x,ys,yc,hx,hys,hyc,?_,?_⟩
  · have hdiff:=(Real.abs_sin_sub_sin_le (2*Real.pi*y) x).trans hphase
    exact (abs_sub_le (ys:ℝ) (Real.sin (2*Real.pi*y)) (Real.sin x)).trans (add_le_add hs hdiff)
  · have hdiff:=(Real.abs_cos_sub_cos_le (2*Real.pi*y) x).trans hphase
    exact (abs_sub_le (yc:ℝ) (Real.cos (2*Real.pi*y)) (Real.cos x)).trans (add_le_add hc hdiff)

#print axioms quantization
#print axioms decoded_scale
#print axioms phase_error
#print axioms normal_bound
end GB300Math.RadianError
