import GB300Math.TextureGradientAxis

namespace GB300Math.TextureGradient1D

def clean (v : Nat) : Nat := if v%2^31 > 0x7f800000 then 0 else v%2^31

def exponent (v : Nat) : Nat := v/2^23%256

def aligned (v e : Nat) : Nat :=
  if exponent v = 0 then 0 else (v%2^23+2^23)/2^(14+e-exponent v)

def norm (a b : Nat) : Nat × Nat :=
  let e := max (exponent a) (exponent b)
  let x := aligned a e
  let y := aligned b e
  let shift := if x+y ≥ 1024 then 1 else 0
  let sum := (x+y)/2^shift*2^shift
  let divisor := 2^(3+shift)
  let product := (sum*11+divisor/2)/divisor*divisor
  (max (max (16*x) (16*y)) product, e)

def position (a b : Nat) : Nat :=
  let a := clean a
  let b := clean b
  if a = 0x7f800000 ∨ b = 0x7f800000 then 2048 else
    let (r,e) := norm a b
    if r = 0 then 0 else
      let top := r.log2
      let index := r*256/2^top%256
      min 2048 (((e:Int)-132+(top:Int))*256+
        (TextureGradientAxis.logTable[index]!:Int)).toNat

def sample (a b : Nat) : Nat := TextureMipmap.encodeQ8 (256+position a b)

theorem norm_symmetric (a b : Nat) : norm a b = norm b a := by
  simp only [norm, Nat.max_comm (exponent b) (exponent a), Nat.add_comm]
  rw [Nat.max_comm (16 * aligned a (max (exponent a) (exponent b)))
    (16 * aligned b (max (exponent a) (exponent b)))]

theorem position_symmetric (a b : Nat) : position a b = position b a := by
  simp only [position, norm_symmetric (clean a) (clean b), or_comm]

theorem position_bounded (a b : Nat) : position a b ≤ 2048 := by
  unfold position
  dsimp only
  split
  · exact Nat.le_refl _
  · split
    · exact Nat.zero_le _
    · exact Nat.min_le_left _ _

#print axioms norm_symmetric
#print axioms position_symmetric
#print axioms position_bounded
end GB300Math.TextureGradient1D
