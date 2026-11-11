/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyArithmetic
import GB300Math.TextureAnisotropyOriented

namespace GB300Math.TextureClampOrder

set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def lower : Array (Nat × Nat) :=
  #[(187657461,4),(1889344616,8),(3754884185,124),(4181450926,484)]

def upperLate : Array (Nat × Nat) :=
  let p := TextureAnisotropyOriented.plane 101 37 62
  #[(993940539,p.1*4),(993940539,p.2.1*4),
    (1961333120,p.2.2.1*4),(1961333120,p.2.2.2*4)]

def upperEarly : Array (Nat × Nat) :=
  let p := TextureAnisotropyOriented.plane 101 0 62
  #[(993940539,p.1*4),(1961333120,p.2.2.1*4)]

theorem exact_merged_weights (w x y : Nat) (hx : x≤256) :
    w*(256-x)*(256-y)+w*x*(256-y)=w*256*(256-y) ∧
    w*(256-x)*y+w*x*y=w*256*y := by
  constructor <;> rw [←Nat.add_mul,←Nat.mul_add,Nat.sub_add_cancel hx]

theorem late_weights :
    let p := TextureAnisotropyOriented.plane 101 37 62
    (p.1+p.2.1)*4=304 ∧ (p.2.2.1+p.2.2.2)*4=100 := by decide +kernel

theorem early_weights :
    let p := TextureAnisotropyOriented.plane 101 0 62
    p.1*4=308 ∧ p.2.2.1*4=96 := by decide +kernel

theorem mass_preserved :
    (upperLate.map Prod.snd).sum=(upperEarly.map Prod.snd).sum ∧
    (upperLate.map Prod.snd).sum=404 := by decide +kernel

theorem late_output :
    TextureAnisotropyArithmetic.interpolate #[lower,upperLate] 4 false=4172383174 := by
  decide +kernel

theorem early_output :
    TextureAnisotropyArithmetic.interpolate #[lower,upperEarly] 4 false=4172383406 := by
  decide +kernel

theorem outputs_differ :
    TextureAnisotropyArithmetic.interpolate #[lower,upperLate] 4 false ≠
    TextureAnisotropyArithmetic.interpolate #[lower,upperEarly] 4 false := by
  rw [late_output,early_output]
  decide

theorem output_word_distance :
    TextureAnisotropyArithmetic.interpolate #[lower,upperEarly] 4 false-
      TextureAnisotropyArithmetic.interpolate #[lower,upperLate] 4 false=232 := by
  rw [late_output,early_output]

#print axioms exact_merged_weights
#print axioms late_weights
#print axioms early_weights
#print axioms mass_preserved
#print axioms late_output
#print axioms early_output
#print axioms outputs_differ
#print axioms output_word_distance

end GB300Math.TextureClampOrder
