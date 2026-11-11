import Std

namespace GB300Math.TextureMipmap

def exponent (v : Nat) : Nat := v / 2^23 % 256

def q8 (v : Nat) : Int :=
  let e := exponent v
  if e = 0 ∨ (e = 255 ∧ v % 2^23 ≠ 0) then 0 else
  let sign : Int := if v ≥ 2^31 then -1 else 1
  if e = 255 then sign * 2^40 else
  let m := v % 2^23 + 2^23
  let shift := (e : Int) - 142
  sign * (if shift ≥ 0 then m * 2^shift.toNat else m / 2^(-shift).toNat : Nat)

def encodeQ8 (n : Nat) : Nat :=
  if n = 0 then 0 else
  let k := n.log2
  (k + 119) * 2^23 + (n * 2^(23-k)) % 2^23

def sample (lod levels : Nat) : Nat :=
  encodeQ8 (256 + min ((levels-1)*256) (q8 lod).toNat)

theorem sample_one_level (lod : Nat) : sample lod 1 = 0x3f800000 := by
  simp only [sample, Nat.sub_self, Nat.zero_mul, Nat.zero_min, Nat.add_zero]
  decide +kernel

theorem subnormal_lod_zero (v : Nat) (h : exponent v = 0) : q8 v = 0 := by
  simp [q8, h]

theorem nan_lod_zero (v : Nat) (he : exponent v = 255) (hm : v % 2^23 ≠ 0) :
    q8 v = 0 := by simp [q8, he, hm]

theorem position_bounded (lod levels : Nat) :
    min ((levels-1)*256) (q8 lod).toNat ≤ (levels-1)*256 := Nat.min_le_left _ _

#print axioms sample_one_level
#print axioms subnormal_lod_zero
#print axioms nan_lod_zero
#print axioms position_bounded

end GB300Math.TextureMipmap
