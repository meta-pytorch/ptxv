import Std.Tactic

namespace GB300Math.ConversionTransitionScan

theorem partition_covers (word : Nat) (h : word < 2^32) :
    word / 65536 < 65536 ∧ word % 65536 < 65536 ∧
      word / 65536 * 65536 + word % 65536 = word := by
  omega

theorem partition_no_unsigned_overflow (part offset : Nat)
    (hp : part < 65536) (ho : offset < 65536) :
    part * 65536 + offset < 2^32 := by
  omega

theorem partition_unique (p q i j : Nat) (hi : i < 65536) (hj : j < 65536)
    (h : p * 65536 + i = q * 65536 + j) : p = q ∧ i = j := by
  omega

theorem complement_covers (word : Nat) (h : word < 2^32) :
    2^32 - 1 - word < 2^32 ∧ 2^32 - 1 - (2^32 - 1 - word) = word := by
  omega

#print axioms partition_covers
#print axioms partition_no_unsigned_overflow
#print axioms partition_unique
#print axioms complement_covers

end GB300Math.ConversionTransitionScan
