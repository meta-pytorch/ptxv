/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhCentralInteger
import TanhCentralPolynomial
namespace GB300Math.TanhCentralIndex
open TanhCentralCells TanhCentralInteger
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def discard (e:Nat) := (TanhTables.binades[e]!).1

theorem geometry (e:Fin 11) (f:Nat) (hf:f<2^23) :
    0<stride e.val ∧ f/stride e.val<count e.val ∧ count e.val≤512 ∧
    (f/stride e.val*stride e.val/2^(discard e.val))/2^17=(f/2^(discard e.val))/2^17 ∧
    (f/2^(discard e.val))%2^17=
      (f/stride e.val*stride e.val/2^(discard e.val))%2^17+(f%stride e.val)/2^(discard e.val) ∧
    f%2^(discard e.val)=(f%stride e.val)%2^(discard e.val) := by
  fin_cases e <;> norm_num [stride,count,discard,TanhTables.binades] <;> omega

theorem parameters (e:Fin 11) :
    0<2^(discard e.val) ∧ 58≤scale e.val ∧ scale e.val≤65 := by
  fin_cases e <;> norm_num [discard,scale,TanhTables.binades]

noncomputable def position (e f:Nat) : ℝ := ((f%stride e:Nat):ℝ)/(stride e:ℝ)
noncomputable def remainder (e f:Nat) : ℝ := ((f%2^(discard e):Nat):ℝ)/(2^(discard e):ℝ)

theorem position_range (e:Fin 11) (f:Nat) (hf:f<2^23) :
    0≤position e.val f ∧ position e.val f≤1 := by
  have h:=geometry e f hf
  have hp : (0:ℝ)<stride e.val := by exact_mod_cast h.1
  have hm : ((f%stride e.val:Nat):ℝ)<stride e.val := by exact_mod_cast Nat.mod_lt f h.1
  unfold position
  exact ⟨by positivity,(div_le_one hp).mpr hm.le⟩

theorem remainder_range (e f:Nat) : 0≤remainder e f ∧ remainder e f≤1 := by
  have hp : (0:ℝ)<2^(discard e) := by positivity
  have hm : ((f%2^(discard e):Nat):ℝ)<2^(discard e) := by exact_mod_cast Nat.mod_lt f (Nat.two_pow_pos _)
  unfold remainder
  exact ⟨by positivity,(div_le_one hp).mpr hm.le⟩

theorem input_identity (e:Fin 11) (f:Nat) (hf:f<2^23) :
    (start e.val (f/stride e.val):ℝ)+(step e.val:ℝ)*position e.val f=
      (2^23+(f:ℝ))*(2:ℝ)^((e.val:Int)-31) := by
  have hs:= (geometry e f hf).1
  have hd : ((f/stride e.val:Nat):ℝ)*(stride e.val:ℝ)+((f%stride e.val:Nat):ℝ)=(f:ℝ) := by
    exact_mod_cast (by simpa [Nat.mul_comm] using Nat.div_add_mod f (stride e.val))
  unfold start step position
  push_cast
  field_simp
  nlinarith only [hd]

theorem local_identity (e:Fin 11) (f:Nat) (hf:f<2^23) :
    (((f/2^(discard e.val))%2^17:Nat):ℝ)+remainder e.val f=
      (((f/stride e.val*stride e.val/2^(discard e.val))%2^17:Nat):ℝ)+
      (stride e.val:ℝ)/(2^(discard e.val):ℝ)*position e.val f := by
  have hg:=geometry e f hf
  have hk:=hg.2.2.2.2.1
  have hmod:=hg.2.2.2.2.2
  have hdiv : (((f%stride e.val)/2^(discard e.val):Nat):ℝ)*(2^(discard e.val):ℝ)+
      (((f%stride e.val)%2^(discard e.val):Nat):ℝ)=((f%stride e.val:Nat):ℝ) := by
    exact_mod_cast (by simpa [Nat.mul_comm] using Nat.div_add_mod (f%stride e.val) (2^(discard e.val)))
  have hstride : (stride e.val:ℝ)≠0 := by exact_mod_cast (ne_of_gt hg.1)
  unfold remainder position
  rw [hk,hmod,Nat.cast_add]
  field_simp
  nlinarith only [hdiv]

theorem native_identity (e:Fin 11) (f:Nat) (hf:f<2^23) :
    let k:Nat:=(f/2^(discard e.val))%2^17
    let (a,b,c):=coeff e.val ((f/2^(discard e.val))/2^17)
    TanhCentralPolynomial.nativePoly e.val (f/stride e.val) (position e.val f)=
      ((a:ℝ)*2^34+(b:ℝ)*((k:ℝ)+remainder e.val f)*2^17+
        (c:ℝ)*((k:ℝ)+remainder e.val f)^2)/2^(scale e.val) := by
  dsimp only
  have hg:=geometry e f hf
  have hi:=local_identity e f hf
  rcases hh:TanhTables.binades[e.val]! with ⟨dd,cs⟩
  have hdd:discard e.val=dd := by simp [discard,hh]
  have hcoeff (j:Nat):cs[j]! =coeff e.val j := by simp [coeff,hh]
  unfold TanhCentralPolynomial.nativePoly nativeCoefficients
  simp only [hh]
  rw [←hdd,hg.2.2.2.1,hcoeff]
  rcases hc:coeff e.val ((f/2^(discard e.val))/2^17) with ⟨a,b,c⟩
  dsimp only
  simp only [Rat.cast_add,Rat.cast_mul,Rat.cast_div,Rat.cast_pow,Rat.cast_intCast,Rat.cast_natCast,Rat.cast_one,Rat.cast_ofNat]
  rw [hi]
  have he : scale e.val = 34+(150-min (e.val+119) 126) := by unfold scale;omega
  rw [he]
  simp only [pow_add]
  ring

#print axioms geometry
#print axioms parameters
#print axioms position_range
#print axioms remainder_range
#print axioms input_identity
#print axioms local_identity
#print axioms native_identity
end GB300Math.TanhCentralIndex
