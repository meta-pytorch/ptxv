import PackedIntegerNew

namespace GB300Math.PackedIntegerNew

theorem encode_bound (width : Nat) (value : Int) : encode width value < 2^width := by
  have hp : (0 : Int) < (2^width : Nat) := by exact_mod_cast Nat.two_pow_pos width
  have hn : ((2^width : Nat) : Int) ≠ 0 := by omega
  exact (Int.toNat_lt (Int.emod_nonneg value hn)).2 (Int.emod_lt_of_pos value hp)

theorem lane_bound (operation : String) (signed relu : Bool) (width a b : Nat) :
    lane operation signed relu width a b < 2^width := by
  exact encode_bound width _

theorem lower_lane_preserved (operation : String) (signed relu : Bool) (a b c : Nat)
    (h : operation ≠ "pack") :
    evaluate operation signed relu 16 a b c % 65536 = lane operation signed relu 16 a b := by
  have bound := lane_bound operation signed relu 16 a b
  simp only [evaluate, h, ↓reduceIte, show (16 : Nat) ≠ 32 by decide]
  omega

theorem lane_ignores_high_bits (operation : String) (signed relu : Bool) (a b highA highB : Nat) :
    lane operation signed relu 16 (a % 65536 + highA*65536) (b % 65536 + highB*65536) =
      lane operation signed relu 16 a b := by
  simp [lane, decode, Nat.add_mod]

theorem output_bound (operation : String) (signed relu : Bool) (a b c : Nat)
    (h : operation ≠ "pack") :
    evaluate operation signed relu 16 a b c < 2^32 := by
  have low := lane_bound operation signed relu 16 a b
  have high := lane_bound operation signed relu 16 (a/65536) (b/65536)
  simp only [evaluate, h, ↓reduceIte, show (16 : Nat) ≠ 32 by decide]
  omega

theorem saturation_value_preserved (signed : Bool) (width : Nat) (value : Int)
    (hw : width ∈ [2,4,8,16]) :
    decode signed width (encode width (clamp signed width value)) = clamp signed width value := by
  simp at hw
  rcases hw with rfl | rfl | rfl | rfl <;> cases signed <;>
    simp [decode, encode, clamp] <;> (try split) <;> omega

theorem packing_preserves_payload (signed : Bool) (width a b c : Nat)
    (hw : width ∈ [2,4,8,16]) :
    pack signed width a b c / 2^(2*width) = c % 2^(32-2*width) := by
  have ha := encode_bound width (clamp signed width (decode true 32 a))
  have hb := encode_bound width (clamp signed width (decode true 32 b))
  simp at hw
  rcases hw with rfl | rfl | rfl | rfl <;> simp only [pack] <;> omega

#print axioms encode_bound
#print axioms lower_lane_preserved
#print axioms lane_ignores_high_bits
#print axioms output_bound
#print axioms saturation_value_preserved
#print axioms packing_preserves_payload

end GB300Math.PackedIntegerNew
