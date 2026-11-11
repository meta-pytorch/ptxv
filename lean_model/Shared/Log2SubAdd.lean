/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import RadianTiny
import Log2Uniform
namespace GB300Math.Log2SubAdd
open FloatLib.Floats.Formats.BinaryInterchange
open Root32FloatLib
set_option maxHeartbeats 500000
set_option maxRecDepth 16000

def result (f : Nat) : Nat := Approx.asWord (Model.addWithRounding .nearestEven
  (Approx.asFloat (2^31+133*2^23+f)) (Approx.asFloat 0xc1c00000))

theorem exact_dyadic (f : Nat) (hf : f<2^23) :
    result f=Approx.asWord (Model.roundDyadicGeneral FloatFormat.binary32
      ⟨true,4*(2^23+f)+12582912,(-19:Int)⟩) := by
  have hl : Approx.asFloat (2^31+133*2^23+f)=Model.ofFields FloatFormat.binary32 true 133 f := by
    simpa using (RcpPtxBounds.fields_float true 133 f (by norm_num) hf).symm
  have hr : Approx.asFloat 0xc1c00000=Model.ofFields FloatFormat.binary32 true 131 4194304 := by
    simpa using (RcpPtxBounds.fields_float true 131 4194304 (by norm_num) (by norm_num)).symm
  unfold result
  rw [hl,hr,Model.addWithRounding,Model.Proof.add_eq_spec,Model.Spec.add,
    RcpPtxBounds.decode_normal true 133 f (by norm_num) hf,
    RcpPtxBounds.decode_normal true 131 4194304 (by norm_num) (by norm_num)]
  simp only []
  rw [←Model.roundDyadicGeneral_eq_roundDyadic]
  congr 2
  simp only [Model.addDyadic,FloatLib.Numerics.Dyadic.add,FloatLib.Numerics.Dyadic.addFields]
  norm_num only [Nat.reducePow,Nat.reduceAdd,Nat.cast_ofNat,Int.reduceSub]
  have hm : (8388608+f==0)=false := by simp
  simp only [hm,Bool.false_eq_true,ite_false]
  simp [Nat.shiftLeft_eq]
  have hz : -((8388608+(f:Int))*4)+ -12582912≠0 := by omega
  have hn : -((8388608+(f:Int))*4)<12582912 := by omega
  have ha : (-((8388608+(f:Int))*4)+ -12582912).natAbs=4*(8388608+f)+12582912 := by omega
  simp only [hz,ite_false,hn,decide_true,ha]

def amplitude (f : Nat) : Nat := 4*(2^23+f)+12582912

theorem pack_normal (l r : Nat) (hl : 25≤l ∧ l≤26) (hr : 2^23≤r ∧ r<2^24) :
    Model.packRoundedNormal FloatFormat.binary32 true (Model.nativeOverflow FloatFormat.binary32 true)
      ((l:Int)+(-19)) r=Model.ofFields FloatFormat.binary32 true (l+108) (r-2^23) := by
  have hc : (r==2^24)=false := by simp;omega
  have ho : ¬127<(l:Int)+(-19) := by omega
  have he : ((l:Int)+(-19)+127).toNat=l+108 := by omega
  have hmax : ¬l+108>254 := by omega
  have hf : ¬r-2^23>8388607 := by omega
  simp only [Model.packRoundedNormal,
    show FloatFormat.binary32.fracWidth=23 from rfl,
    show FloatFormat.binary32.maxNormalExponent=127 from rfl,
    show FloatFormat.binary32.exponentBias=127 from rfl,
    show FloatFormat.binary32.maxFiniteExpField=254 from rfl,
    show FloatFormat.binary32.maxFiniteFracField=8388607 from rfl,
    show Model.pow2 (23+1)=2^24 from rfl,
    show Model.pow2 23=2^23 from rfl,hc,Bool.false_eq_true,ite_false,ho]
  norm_num only [Nat.cast_ofNat]
  rw [he]
  norm_num only [Nat.reducePow] at hf
  simp only [hmax,hf,decide_false,Bool.and_false,Bool.or_self,Bool.false_eq_true,ite_false]

theorem rounded_data (f : Nat) (hf : f<2^23) :
    let n:=amplitude f
    let l:=n.log2
    let r:=FloatLib.Numerics.roundShiftRightEven n (l-23)
    25≤l ∧ l≤26 ∧ 2^23≤r ∧ r<2^24 ∧
    result f=2^31+(l+108)*2^23+(r-2^23) ∧
    n/2^(l-23)≤r ∧ r≤n/2^(l-23)+1 := by
  dsimp only
  have hn : 2^25≤amplitude f ∧ amplitude f<79691776 := by dsimp [amplitude];omega
  have hlog0 : 25≤(amplitude f).log2 := (Nat.le_log2 (by omega)).mpr hn.1
  have hlog1 : (amplitude f).log2<27 := (Nat.log2_lt (by omega)).mpr (by omega)
  have hlow:=Nat.log2_self_le (n:=amplitude f) (by omega)
  have hhigh:=Nat.lt_log2_self (n:=amplitude f)
  generalize hl:(amplitude f).log2=l at *
  let r:=FloatLib.Numerics.roundShiftRightEven (amplitude f) (l-23)
  have hr0 : amplitude f/2^(l-23)≤r := by
    simpa only [RadianTiny.shift] using FloatLib.Numerics.shiftRight_le_roundShiftRightEven (amplitude f) (l-23)
  have hr1 : r≤amplitude f/2^(l-23)+1 := by
    simpa only [RadianTiny.shift] using FloatLib.Numerics.roundShiftRightEven_le_shiftRight_add1 (amplitude f) (l-23)
  have hr : 2^23≤r ∧ r<2^24 := by
    rcases (show l=25 ∨ l=26 by omega) with he|he
    · have exactr : r=amplitude f/4 := by
        dsimp [r]
        rw [he,FloatLib.Numerics.roundShiftRightEven_eq_roundQuotientEven]
        norm_num [FloatLib.Numerics.roundQuotientEven,amplitude]
      rw [exactr]
      norm_num [he] at hlow hhigh
      omega
    · norm_num [he] at hr0 hr1 hlow hhigh
      omega
  refine ⟨hlog0,by omega,hr.1,hr.2,?_,hr0,hr1⟩
  rw [exact_dyadic f hf]
  change Approx.asWord (Model.roundDyadicGeneral FloatFormat.binary32 ⟨true,amplitude f,-19⟩)=_
  have hn0 : (amplitude f==0)=false := by simp;omega
  have hexp : ((l:Int)+(-19)+127).toNat=l+108 := by omega
  have he0 : ¬127<(l:Int)+(-19) := by omega
  have he1 : ¬(l:Int)+(-19)< -126 := by omega
  have hlead : 23≤l := by omega
  have carry : (r==2^24)=false := by simp;omega
  have hfrac : r-2^23<2^23 := by omega
  have hcarry : ¬l+108>254 := by omega
  have hlast : ¬r-2^23>8388607 := by omega
  simp only [Model.roundDyadicGeneral,hn0,Bool.false_eq_true,ite_false,hl,
    show FloatFormat.binary32.maxNormalExponent=127 from rfl,
    show FloatFormat.binary32.minNormalExponent= -126 from rfl,
    show FloatFormat.binary32.fracWidth=23 from rfl,
    hlead,ite_true]
  norm_num only [Nat.cast_ofNat]
  rw [ite_eq_right he0,ite_eq_right he1]
  change Approx.asWord (Model.packRoundedNormal FloatFormat.binary32 true _ ((l:Int)+(-19)) r)=_
  rw [pack_normal l r ⟨hlog0,by omega⟩ hr]
  exact RadianTiny.fields_word true (l+108) (r-2^23) (by omega) hfrac

#print axioms exact_dyadic
#print axioms pack_normal
#print axioms rounded_data
end GB300Math.Log2SubAdd
