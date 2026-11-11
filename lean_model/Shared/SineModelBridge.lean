/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SinePhaseResult
import SineSubnormal
namespace GB300Math.SineModelBridge
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 4000000

def word (negative : Bool) (e : Fin 255) (f : Fin (2^23)) :=
  (if negative then 2^31 else 0)+(e.val*2^23+f.val)
def phase (e : Fin 255) (f : Fin (2^23)) := SineArgument.rawPhase e.val (2^23+f.val)

theorem parameter_fields (negative : Bool) (e : Fin 255) (f : Fin (2^23)) :
    Mufu.exponent (word negative e f)=e.val ∧
    Mufu.fraction (word negative e f)=f.val ∧
    Mufu.sign (word negative e f)=(if negative then 2^31 else 0) := by
  have he:=e.isLt
  have hf:=f.isLt
  have hw : e.val*2^23+f.val<2^31 := by omega
  cases negative
  · simp only [word,Bool.false_eq_true,ite_false,Nat.zero_add]
    exact ⟨by unfold Mufu.exponent;omega,by unfold Mufu.fraction;omega,RcpSigned.sign_zero _ hw⟩
  · simp only [word,ite_true]
    exact ⟨by unfold Mufu.exponent;omega,by unfold Mufu.fraction;omega,RcpSigned.sign_set _ hw⟩

theorem kernel_shape (negative : Bool) (e : Fin 255) (f : Fin (2^23)) :
    Mufu.sincos (word negative e f) false=SinePhaseResult.phaseWord (phase e f) negative ∧
    Mufu.sincos (word negative e f) true=SinePhaseResult.phaseWord ((phase e f+2^23)%2^25) false := by
  have h := parameter_fields negative e f
  have he : e.val≠255 := by have:=e.isLt;omega
  unfold Mufu.sincos
  rw [h.1,h.2.1,h.2.2]
  simp only [he,ite_false]
  change _=_ ∧ _=_
  simp only [phase,SineArgument.rawPhase,SinePhaseResult.phaseWord,
    SineTableBounds.magnitudeWord,SinePhase.tableIndex]
  cases negative <;> simp [Nat.add_comm]

theorem positive_reduction (e : Fin 255) (f : Fin (2^23)) :
    ∃ x r : Rat, ∃ n : Nat,
      (Model.ofNatBits (word false e f) : Model FloatFormat.binary32).toRat?=some x ∧
      x=(n:Rat)+(phase e f:Rat)/2^25+r ∧ 0≤r ∧ r≤1/2^25 := by
  by_cases hz : e.val=0
  · have h := SineSubnormal.reduction f
    simpa [word,phase,hz] using h
  · let ei : Fin 254 := ⟨e.val-1,by have:=e.isLt;omega⟩
    have he : 1+ei.val=e.val := by dsimp [ei];omega
    have h := SineArgument.positive_normal ei f
    simp only [he] at h
    simpa [word,phase] using h

#print axioms kernel_shape
#print axioms positive_reduction
end GB300Math.SineModelBridge
