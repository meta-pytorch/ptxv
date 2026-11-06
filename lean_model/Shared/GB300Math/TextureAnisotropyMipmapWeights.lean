import GB300Math.TextureAnisotropyTaps

namespace GB300Math.TextureAnisotropyMipmapWeights

set_option maxRecDepth 100000
set_option maxHeartbeats 0

def roundMass (fraction mass : Nat) : Nat := (fraction*mass+128)/256

def scaleFrom (fraction start : Nat) : List Nat → List Nat
  | [] => []
  | w::ws => (roundMass fraction (start+w)-roundMass fraction start)::
      scaleFrom fraction (start+w) ws

def scaleWeights (fraction : Nat) (weights : List Nat) : List Nat :=
  scaleFrom fraction 0 weights

theorem round_monotone (f a b : Nat) (h : a≤b) : roundMass f a≤roundMass f b := by
  apply Nat.div_le_div_right
  exact Nat.add_le_add_right (Nat.mul_le_mul_left f h) 128

theorem sum_scaleFrom (f start : Nat) (weights : List Nat) :
    (scaleFrom f start weights).sum =
      roundMass f (start+weights.sum)-roundMass f start := by
  induction weights generalizing start with
  | nil => simp [scaleFrom]
  | cons w ws ih =>
    have h0 := round_monotone f start (start+w) (by omega)
    have h1 := round_monotone f (start+w) (start+w+ws.sum) (by omega)
    simp only [scaleFrom,List.sum_cons,ih]
    rw [show start+(w+ws.sum)=start+w+ws.sum by omega]
    omega

theorem mass_conservation (f groups : Nat) (weights : List Nat)
    (h : weights.sum=groups*256) :
    (scaleWeights f weights).sum=f*groups := by
  rw [scaleWeights,sum_scaleFrom,h]
  simp only [Nat.zero_add,roundMass,Nat.mul_zero,Nat.zero_add,Nat.reduceDiv,Nat.sub_zero]
  rw [← Nat.mul_assoc,Nat.add_div]
  simp
  decide

theorem prefix_error (n : Int) :
    -127 ≤ ((n+128)/256)*256-n ∧ ((n+128)/256)*256-n ≤ 128 := by
  have h0 := Int.emod_nonneg (n+128) (by decide : (256:Int)≠0)
  have h1 := Int.emod_lt_of_pos (n+128) (by decide : (0:Int)<256)
  rw [Int.emod_def,Int.mul_comm 256 ((n+128)/256)] at h0 h1
  omega

theorem coefficient_error (a b : Int) :
    -255 ≤ (((b+128)/256)-((a+128)/256))*256-(b-a) ∧
    (((b+128)/256)-((a+128)/256))*256-(b-a) ≤ 255 := by
  have h0 := prefix_error a
  have h1 := prefix_error b
  simp only [Int.sub_mul]
  omega

theorem coefficient_error_attained :
    (((383+128)/256)-((128+128)/256))*256-(383-128 : Int) = -255 ∧
    (((384+128)/256)-((127+128)/256))*256-(384-127 : Int) = 255 := by
  decide +kernel

def halfStep (j : Nat) : Nat :=
  (j+1)/4-if j=367 ∨ j=371 ∨ j=467 ∨ j=511 then 1 else 0

theorem half_step_bucket_bound : ∀ k : Fin 256,
    let j := 256+k.val
    j+1-4*halfStep j ≤ 4 ∧ 4*halfStep j-j ≤ 4 := by decide +kernel

theorem asymmetric_tie : scaleWeights 24 [16,240,240,16]=[2,22,23,1] := by
  decide +kernel

theorem cumulative_differs_from_independent :
    scaleWeights 11 [16,248,248,248,248,16] = [1,10,11,11,10,1] := by
  decide +kernel

#print axioms round_monotone
#print axioms sum_scaleFrom
#print axioms mass_conservation
#print axioms prefix_error
#print axioms coefficient_error
#print axioms coefficient_error_attained
#print axioms half_step_bucket_bound
#print axioms asymmetric_tie
#print axioms cumulative_differs_from_independent

end GB300Math.TextureAnisotropyMipmapWeights
