/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureGradientAxis

namespace GB300Math.TextureAnisotropyTaps

set_option maxRecDepth 100000
set_option maxHeartbeats 0

def reciprocalTable : Array Nat := #[256,255,255,254,253,253,252,251,251,250,249,248,248,247,246,246,245,244,244,243,243,242,241,241,240,239,239,238,237,237,236,235,235,234,233,233,232,232,231,230,230,229,228,228,227,227,226,225,225,224,224,223,222,222,221,221,220,219,219,218,218,217,216,216,215,215,214,214,213,212,212,211,211,210,210,209,208,208,207,207,206,206,205,204,204,203,203,202,202,201,201,200,200,199,198,198,197,197,196,196,195,195,194,194,193,193,192,192,191,191,190,190,189,189,188,188,187,186,186,185,185,184,184,183,183,182,182,182,181,181,180,180,179,179,178,178,177,177,176,176,175,175,174,174,173,173,172,172,171,171,171,170,170,169,169,168,168,167,167,166,166,166,165,165,164,164,163,163,162,162,162,161,161,160,160,159,159,159,158,158,157,157,156,156,156,155,155,154,154,153,153,153,152,152,151,151,151,150,150,149,149,149,148,148,147,147,147,146,146,145,145,145,144,144,143,143,143,142,142,141,141,141,140,140,140,139,139,138,138,138,137,137,137,136,136,135,135,135,134,134,134,133,133,133,132,132,132,131,131,130,130,130,129,129,129,128]

structure Tap where
  offset : Int
  weight : Nat
  deriving Repr, Inhabited

def logarithm (gradient : Nat) : Nat :=
  let e := gradient/2^23%256
  let m := gradient%2^23
  if e ≤ 119 then
    if e = 119 then TextureGradientAxis.logTable[m/2^15]! else 0
  else (e-119)*256+TextureGradientAxis.logTable[m/2^15]!

def configuration (q : Nat) : Nat × Nat :=
  let exponent := q/256
  let groupCount := 2^min 2 exponent
  let weight := reciprocalTable[q%256]! / 2^(exponent-min 2 exponent)
  if groupCount = 2 ∧ weight = 128 then (4,256) else (groupCount,weight)

def taps (q : Nat) (negative : Bool) : Nat × List Tap :=
  if q = 0 then (1,[⟨0,256⟩])
  else if q < 256 then
    let r := reciprocalTable[q]!
    let offset := 2*((256+(if negative then 0 else 1)-r)/2)
    (2,[⟨-(offset:Int),256⟩,⟨offset,256⟩])
  else
    let (groups,weight) := configuration q
    let count := 2*((groups*256+2*weight-1)/(2*weight))
    let edge := (groups*256-(count-2)*weight)/2
    (groups, (List.range count).map fun (i : Nat) =>
      ⟨128*(2*(i:Int)-(count:Int)+1), if i=0 ∨ i+1=count then edge else weight⟩)

def upperWeight (weight fraction : Nat) (vertical : Bool) : Nat :=
  (weight*fraction+(if vertical then 127 else 128))/256

def contribution (weight fraction : Nat) (vertical : Bool) (low high : Int) : Int :=
  let upper := upperWeight weight fraction vertical
  ((weight-upper:Int)*low)+(upper:Int)*high

theorem upperWeight_bounded (w f : Nat) (vertical : Bool) (hf : f ≤ 256) :
    upperWeight w f vertical ≤ w := by
  have hm := Nat.mul_le_mul_left w hf
  unfold upperWeight
  split <;> omega

theorem horizontal_rounding (n : Nat) :
    -127 ≤ (256*((n+128)/256):Int)-(n:Int) ∧
    (256*((n+128)/256):Int)-(n:Int) ≤ 128 := by omega

theorem vertical_rounding (n : Nat) :
    -128 ≤ (256*((n+127)/256):Int)-(n:Int) ∧
    (256*((n+127)/256):Int)-(n:Int) ≤ 127 := by omega

theorem constant_preserved (w f : Nat) (vertical : Bool) (c : Int) (hf : f ≤ 256) :
    contribution w f vertical c c = (w:Int)*c := by
  have hu := upperWeight_bounded w f vertical hf
  unfold contribution
  rw [← Int.add_mul]
  congr 1
  omega

theorem weights_sum : ∀ q : Fin 1025, ∀ negative : Bool,
    ((taps q negative).2.map Tap.weight).foldl (·+·) 0 = (taps q negative).1*256 := by
  decide

theorem table_bounds : ∀ q : Fin 256,
    128 ≤ reciprocalTable[q]! ∧ reciprocalTable[q]! ≤ 256 := by decide

theorem log3_witness : logarithm 0x3c400000 = 406 := by decide

theorem horizontal_drift_witness :
    ((taps 406 false).2.map fun t => contribution t.weight 128 false
      (t.offset/256) (t.offset/256+1)).foldl (·+·) 0 = 2 := by decide

theorem vertical_drift_witness :
    ((taps 406 false).2.map fun t => contribution t.weight 128 true
      (t.offset/256) (t.offset/256+1)).foldl (·+·) 0 = -2 := by decide

#print axioms upperWeight_bounded
#print axioms horizontal_rounding
#print axioms vertical_rounding
#print axioms constant_preserved
#print axioms weights_sum
#print axioms table_bounds
#print axioms log3_witness
#print axioms horizontal_drift_witness
#print axioms vertical_drift_witness
end GB300Math.TextureAnisotropyTaps
