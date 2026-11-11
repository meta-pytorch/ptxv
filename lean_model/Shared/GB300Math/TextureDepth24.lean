import GB300Math.TextureDepthNormalized

namespace GB300Math.TextureDepth24

open TextureDepthCompare

def nearestEven (numerator denominator : Nat) : Nat :=
  let q := numerator / denominator
  let r := numerator % denominator
  q + if 2*r > denominator || (2*r == denominator && q % 2 == 1) then 1 else 0

def reference (word : Nat) : Nat :=
  if word / 2^31 = 1 || isNaN word then 0
  else if word ≥ 0x3f800000 then 0xffffff
  else if word / 2^23 = 0 then 0
  else
    let mantissa := 2^23 + word % 2^23
    let denominator := 2^(150-word/2^23)
    if word < 0x3d800000 then
      (2*mantissa*2^24 + denominator) / (2*denominator)
    else nearestEven (mantissa*0xffffff) denominator

def compare (function word depth : Nat) : Nat :=
  if TextureDepthNormalized.cmp function (reference word) depth then 0x3f800000 else 0

theorem nearestEven_error (n d : Nat) (hd : 0 < d) :
    -((d : Nat) : Int) ≤ 2*((nearestEven n d : Nat) : Int)*d - 2*(n : Int) ∧
    2*((nearestEven n d : Nat) : Int)*d - 2*(n : Int) ≤ d := by
  have h := Nat.mod_add_div n d
  have hr := Nat.mod_lt n hd
  have hi := congrArg (fun x : Nat => (x : Int)) h
  simp only [Int.natCast_add, Int.natCast_mul] at hi
  rw [Int.mul_comm (d : Int) ((n/d : Nat) : Int)] at hi
  simp only [nearestEven]
  split
  all_goals
    rename_i hc
    simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
    simp only [Int.natCast_add, Int.natCast_one, Int.natCast_zero, Int.add_mul,
      Int.mul_add, Int.zero_mul, Int.mul_one, Int.mul_zero, Int.mul_assoc]
    omega

theorem halfUp_error (n d : Nat) (hd : 0 < d) :
    -((d : Nat) : Int) < 2*(((2*n+d)/(2*d) : Nat) : Int)*d - 2*(n : Int) ∧
    2*(((2*n+d)/(2*d) : Nat) : Int)*d - 2*(n : Int) ≤ d := by
  have h := Nat.mod_add_div (2*n+d) (2*d)
  have hr := Nat.mod_lt (2*n+d) (by omega : 0 < 2*d)
  have hi := congrArg (fun x : Nat => (x : Int)) h
  simp only [Int.natCast_add, Int.natCast_mul] at hi
  change (((2*n+d) % (2*d) : Nat) : Int) + (2*(d : Int))*(((2*n+d)/(2*d) : Nat) : Int) = 2*(n : Int)+d at hi
  simp only [Int.mul_assoc, Int.mul_comm] at *
  omega

theorem first_positive_tie : reference 0x33000000 = 1 := by decide +kernel

theorem below_first_positive_tie : reference 0x32ffffff = 0 := by decide +kernel

theorem half_tie_even : reference 0x3f000000 = 0x800000 := by decide +kernel

theorem half_successor : reference 0x3f000001 = 0x800000 := by decide +kernel

theorem branch_boundary : reference 0x3d800000 = 0x100000 := by decide +kernel

theorem zero_nan : reference 0x7fc00000 = 0 := by decide +kernel

theorem positive_infinity : reference 0x7f800000 = 0xffffff := by decide +kernel

theorem negative_infinity : reference 0xff800000 = 0 := by decide +kernel

end GB300Math.TextureDepth24
