import GB300Math.XorSign

namespace GB300Math.XorSign

theorem magnitude_lt_sign (f : Format) (ftz : Bool) (a : Nat) :
    magnitude f ftz a < 2^(f.width-1) := by
  unfold magnitude
  dsimp only
  split
  · exact Nat.two_pow_pos _
  · exact Nat.mod_lt _ (Nat.two_pow_pos _)

theorem lane_non_nan (f : Format) (maximum propagate ftz : Bool) (a b : Nat)
    (ha : magnitude f ftz a ≤ 2^(f.width-1)-2^f.fraction)
    (hb : magnitude f ftz b ≤ 2^(f.width-1)-2^f.fraction) :
    lane f maximum propagate ftz a b =
      (if maximum then max (magnitude f ftz a) (magnitude f ftz b)
       else min (magnitude f ftz a) (magnitude f ftz b)) +
      (((a/2^(f.width-1))%2+(b/2^(f.width-1))%2)%2)*2^(f.width-1) := by
  simp [lane, Nat.not_lt.mpr ha, Nat.not_lt.mpr hb]

theorem selected_magnitude_lt (f : Format) (maximum ftz : Bool) (a b : Nat) :
    (if maximum then max (magnitude f ftz a) (magnitude f ftz b)
     else min (magnitude f ftz a) (magnitude f ftz b)) < 2^(f.width-1) := by
  have ha := magnitude_lt_sign f ftz a
  have hb := magnitude_lt_sign f ftz b
  split
  · exact Nat.max_lt.mpr ⟨ha, hb⟩
  · exact Nat.lt_of_le_of_lt (Nat.min_le_left _ _) ha

theorem non_nan_lane_sign (f : Format) (maximum propagate ftz : Bool) (a b : Nat)
    (ha : magnitude f ftz a ≤ 2^(f.width-1)-2^f.fraction)
    (hb : magnitude f ftz b ≤ 2^(f.width-1)-2^f.fraction) :
    lane f maximum propagate ftz a b / 2^(f.width-1) =
      ((a/2^(f.width-1))%2+(b/2^(f.width-1))%2)%2 := by
  rw [lane_non_nan f maximum propagate ftz a b ha hb]
  rw [Nat.add_mul_div_right _ _ (Nat.two_pow_pos _)]
  rw [Nat.div_eq_of_lt (selected_magnitude_lt f maximum ftz a b)]
  simp

theorem non_nan_lane_magnitude (f : Format) (maximum propagate ftz : Bool) (a b : Nat)
    (ha : magnitude f ftz a ≤ 2^(f.width-1)-2^f.fraction)
    (hb : magnitude f ftz b ≤ 2^(f.width-1)-2^f.fraction) :
    lane f maximum propagate ftz a b % 2^(f.width-1) =
      if maximum then max (magnitude f ftz a) (magnitude f ftz b)
      else min (magnitude f ftz a) (magnitude f ftz b) := by
  rw [lane_non_nan f maximum propagate ftz a b ha hb]
  rw [Nat.add_mul_mod_self_right]
  exact Nat.mod_eq_of_lt (selected_magnitude_lt f maximum ftz a b)

theorem one_nan_left (f : Format) (maximum ftz : Bool) (a b : Nat)
    (ha : magnitude f ftz a > 2^(f.width-1)-2^f.fraction)
    (hb : magnitude f ftz b ≤ 2^(f.width-1)-2^f.fraction) :
    lane f maximum false ftz a b = magnitude f ftz b +
      (((a/2^(f.width-1))%2+(b/2^(f.width-1))%2)%2)*2^(f.width-1) := by
  simp [lane, ha, Nat.not_lt.mpr hb]

theorem one_nan_right (f : Format) (maximum ftz : Bool) (a b : Nat)
    (ha : magnitude f ftz a ≤ 2^(f.width-1)-2^f.fraction)
    (hb : magnitude f ftz b > 2^(f.width-1)-2^f.fraction) :
    lane f maximum false ftz a b = magnitude f ftz a +
      (((a/2^(f.width-1))%2+(b/2^(f.width-1))%2)%2)*2^(f.width-1) := by
  simp [lane, Nat.not_lt.mpr ha, hb]

theorem both_nan (f : Format) (maximum propagate ftz : Bool) (a b : Nat)
    (ha : magnitude f ftz a > 2^(f.width-1)-2^f.fraction)
    (hb : magnitude f ftz b > 2^(f.width-1)-2^f.fraction) :
    lane f maximum propagate ftz a b = 2^(f.width-1)-1 := by
  simp [lane, ha, hb]

theorem propagate_nan (f : Format) (maximum ftz : Bool) (a b : Nat)
    (h : magnitude f ftz a > 2^(f.width-1)-2^f.fraction ∨
         magnitude f ftz b > 2^(f.width-1)-2^f.fraction) :
    lane f maximum true ftz a b = 2^(f.width-1)-1 := by
  rcases h with ha | hb
  · simp [lane, ha]
  · simp [lane, hb]

theorem magnitude16_mod (fraction : Nat) (ftz : Bool) (a : Nat) :
    magnitude ⟨16, fraction⟩ ftz (a % 65536) = magnitude ⟨16, fraction⟩ ftz a := by
  unfold magnitude
  dsimp only
  rw [Nat.mod_mod_of_dvd a (by decide : 32768 ∣ 65536)]

theorem lane16_mod (fraction : Nat) (maximum propagate ftz : Bool) (a b : Nat) :
    lane ⟨16, fraction⟩ maximum propagate ftz (a % 65536) (b % 65536) =
      lane ⟨16, fraction⟩ maximum propagate ftz a b := by
  have ha : a % 65536 / 32768 = a / 32768 % 2 := Nat.mod_mul_right_div_self a 32768 2
  have hb : b % 65536 / 32768 = b / 32768 % 2 := Nat.mod_mul_right_div_self b 32768 2
  simp only [lane, magnitude16_mod]
  simp only [ha, hb, Nat.mod_mod]


theorem lane16_lt (fraction : Nat) (maximum propagate ftz : Bool) (a b : Nat) :
    lane ⟨16, fraction⟩ maximum propagate ftz a b < 65536 := by
  have ha := magnitude_lt_sign (⟨16, fraction⟩ : Format) ftz a
  have hb := magnitude_lt_sign (⟨16, fraction⟩ : Format) ftz b
  change magnitude ⟨16, fraction⟩ ftz a < 32768 at ha
  change magnitude ⟨16, fraction⟩ ftz b < 32768 at hb
  have hc := Nat.mod_lt ((a / 32768)%2+(b/32768)%2) (by decide : 0 < 2)
  simp only [lane]
  split
  · decide
  · split
    · omega
    · split
      · omega
      · split
        · have hm := Nat.max_lt.mpr ⟨ha, hb⟩
          omega
        · have hm := Nat.lt_of_le_of_lt (Nat.min_le_left (magnitude ⟨16, fraction⟩ ftz a) (magnitude ⟨16, fraction⟩ ftz b)) ha
          omega


theorem packed_f16x2 (maximum propagate ftz : Bool) (a b : Nat) :
    evaluate "f16x2" (if maximum then "max" else "min") ftz propagate a b =
      some (lane ⟨16, 10⟩ maximum propagate ftz (a % 65536) (b % 65536) +
        65536 * lane ⟨16, 10⟩ maximum propagate ftz (a / 65536 % 65536) (b / 65536 % 65536)) := by
  cases maximum <;> simp [evaluate, format, lane16_mod]

theorem packed_bf16x2 (maximum propagate : Bool) (a b : Nat) :
    evaluate "bf16x2" (if maximum then "max" else "min") false propagate a b =
      some (lane ⟨16, 7⟩ maximum propagate false (a % 65536) (b % 65536) +
        65536 * lane ⟨16, 7⟩ maximum propagate false (a / 65536 % 65536) (b / 65536 % 65536)) := by
  cases maximum <;> simp [evaluate, format, lane16_mod]

theorem packed_low_projection (fraction : Nat) (maximum propagate ftz : Bool)
    (alo ahi blo bhi : Nat) :
    (lane ⟨16, fraction⟩ maximum propagate ftz alo blo +
      65536 * lane ⟨16, fraction⟩ maximum propagate ftz ahi bhi) % 65536 =
    lane ⟨16, fraction⟩ maximum propagate ftz alo blo := by
  rw [Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt (lane16_lt fraction maximum propagate ftz alo blo)

theorem packed_high_projection (fraction : Nat) (maximum propagate ftz : Bool)
    (alo ahi blo bhi : Nat) :
    (lane ⟨16, fraction⟩ maximum propagate ftz alo blo +
      65536 * lane ⟨16, fraction⟩ maximum propagate ftz ahi bhi) / 65536 =
    lane ⟨16, fraction⟩ maximum propagate ftz ahi bhi := by
  rw [Nat.add_mul_div_left _ _ (by decide : 0 < 65536)]
  rw [Nat.div_eq_of_lt (lane16_lt fraction maximum propagate ftz alo blo)]
  simp


#print axioms magnitude_lt_sign
#print axioms lane_non_nan
#print axioms selected_magnitude_lt
#print axioms non_nan_lane_sign
#print axioms non_nan_lane_magnitude

#print axioms one_nan_left
#print axioms one_nan_right
#print axioms both_nan
#print axioms propagate_nan
#print axioms magnitude16_mod
#print axioms lane16_mod

#print axioms lane16_lt

#print axioms packed_f16x2
#print axioms packed_bf16x2
#print axioms packed_low_projection
#print axioms packed_high_projection

end GB300Math.XorSign
