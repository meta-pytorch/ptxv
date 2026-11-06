import GB300Math.TextureDepthPCF

namespace GB300Math.TextureDepthPCF

theorem coefficients_bounds (x y : Nat) (_hx : x ≤ 256) (hy : y ≤ 256) :
    ((256-x)*(256-y)+128)/256 ≤ 256-x ∧ (x*y+128)/256 ≤ x := by
  have h1 := Nat.mul_le_mul_left (256-x) (Nat.sub_le 256 y)
  have h2 := Nat.mul_le_mul_left x hy
  omega

theorem interpolation_error (x y : Nat) (hx : x ≤ 256) (hy : y ≤ 256)
    (a b c d : Bool) :
    -256 ≤ 256*(blend x y a b c d : Int)-idealNumerator x y a b c d ∧
    256*(blend x y a b c d : Int)-idealNumerator x y a b c d ≤ 256 := by
  have hb := coefficients_bounds x y hx hy
  have hl : -127 ≤ 256*((((256-x)*(256-y)+128)/256 : Nat) : Int) - ((256-x)*(256-y) : Nat) ∧
      256*((((256-x)*(256-y)+128)/256 : Nat) : Int) - ((256-x)*(256-y) : Nat) ≤ 128 := by omega
  have hh : -127 ≤ 256*(((x*y+128)/256 : Nat) : Int) - (x*y : Nat) ∧
      256*(((x*y+128)/256 : Nat) : Int) - (x*y : Nat) ≤ 128 := by omega
  simp only [Int.natCast_mul, Int.natCast_sub hx, Int.natCast_sub hy,
    Int.natCast_add, Int.natCast_ediv] at hl hh
  cases a <;> cases b <;> cases c <;> cases d <;>
    simp [blend, plane, idealNumerator, Bool.toNat, Int.natCast_sub hb.1,
      Int.natCast_sub hb.2, Int.natCast_sub hx, Int.natCast_sub hy] <;> omega

theorem interpolation_bound_attained :
    256*(blend 1 128 true false false true : Int) -
      idealNumerator 1 128 true false false true = 256 := by decide +kernel

end GB300Math.TextureDepthPCF
