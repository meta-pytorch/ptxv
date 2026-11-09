import Std

namespace GB300Math.VulkanMultisample

def position (width height samples sample layer x y : Nat) : Nat :=
  ((layer * height + y) * width + x) * samples + sample

theorem sample_recovery (width height samples sample layer x y : Nat)
    (h : sample < samples) :
    position width height samples sample layer x y % samples = sample := by
  simp [position, Nat.add_mod, Nat.mod_eq_of_lt h]

theorem pixel_recovery (width height samples sample layer x y : Nat)
    (h : sample < samples) :
    position width height samples sample layer x y / samples =
      (layer * height + y) * width + x := by
  have hs : 0 < samples := by omega
  simp only [position, Nat.add_comm (((layer * height + y) * width + x) * samples) sample,
    Nat.add_mul_div_right _ _ hs, Nat.div_eq_of_lt h, Nat.zero_add]

theorem pair_injective (base a b c d : Nat) (hb : b < base) (hd : d < base)
    (h : a * base + b = c * base + d) : a = c ∧ b = d := by
  have hp : 0 < base := by omega
  have hm := congrArg (· % base) h
  simp [Nat.add_mod, Nat.mod_eq_of_lt hb, Nat.mod_eq_of_lt hd] at hm
  have hq := congrArg (· / base) h
  rw [Nat.add_comm (a*base) b, Nat.add_comm (c*base) d,
    Nat.add_mul_div_right _ _ hp, Nat.add_mul_div_right _ _ hp,
    Nat.div_eq_of_lt hb, Nat.div_eq_of_lt hd] at hq
  exact ⟨by simpa using hq, hm⟩

theorem position_injective (width height samples sample layer x y sample' layer' x' y' : Nat)
    (hs : sample < samples) (hs' : sample' < samples)
    (hx : x < width) (hx' : x' < width) (hy : y < height) (hy' : y' < height)
    (h : position width height samples sample layer x y =
      position width height samples sample' layer' x' y') :
    sample = sample' ∧ layer = layer' ∧ x = x' ∧ y = y' := by
  have h1 := pair_injective samples _ _ _ _ hs hs' h
  have h2 := pair_injective width _ _ _ _ hx hx' h1.1
  have h3 := pair_injective height _ _ _ _ hy hy' h2.1
  exact ⟨h1.2, h3.1, h2.2, h3.2⟩

theorem order_is_distinguishable :
    position 5 3 4 0 1 1 2 = 104 ∧
    position 5 3 4 1 0 1 2 = 45 := by decide +kernel

end GB300Math.VulkanMultisample
