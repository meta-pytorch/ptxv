import GB300Math.TextureMipmap

namespace GB300Math.TextureAnisotropyBlend

def fraction (u aniso : Nat) : Int :=
  if aniso=1 then u else
  if aniso=2 then ((3*u/2 : Nat):Int)-63 else
  if aniso=4 then ((u+u/2+(u+3)/4 : Nat):Int)-95 else
  2*(u:Int)-128

def remap (q aniso : Nat) : Nat :=
  q/256*256 + (max 0 (min 256 (fraction (q%256) aniso))).toNat

def bound (aniso : Nat) : Nat :=
  if aniso=1 then 0 else if aniso=2 then 43 else if aniso=4 then 55 else 64

theorem fraction_error (u aniso : Nat) (hu : u<256) :
    -(bound aniso : Int) ≤ max 0 (min 256 (fraction u aniso)) - u ∧
    max 0 (min 256 (fraction u aniso)) - u ≤ bound aniso := by
  unfold fraction bound
  split <;> (try split) <;> (try split) <;> omega

theorem remap_error (q aniso : Nat) :
    -(bound aniso : Int) ≤ (remap q aniso : Int)-q ∧
    (remap q aniso : Int)-q ≤ bound aniso := by
  have h := fraction_error (q%256) aniso (Nat.mod_lt _ (by decide : 0<256))
  unfold remap
  omega

theorem small_endpoints : remap 213 2 = 256 ∧ remap 201 4 = 256 := by decide +kernel

theorem endpoints : remap 64 8 = 0 ∧ remap 192 8 = 256 := by decide +kernel

theorem isotropic (q : Nat) : remap q 1 = q := by
  unfold remap fraction
  omega

end GB300Math.TextureAnisotropyBlend
