import GB300Math.Mufu

set_option maxRecDepth 100000
set_option maxHeartbeats 0

namespace GB300Math.MufuProperties

theorem sqrt_normalized_segment_boundaries : ∀ segment : Fin 128,
    (Mufu.sqrt (0x3f800000 + segment.val * 131072) <
      Mufu.sqrt (0x3f800000 + segment.val * 131072 - 1)) ↔
      segment.val = 16 ∨ segment.val = 23 ∨ segment.val = 24 ∨ segment.val = 46 := by
  decide +kernel

theorem sqrt_scaled_boundary_drops : ∀ exponent : Fin 127,
    ∀ fraction : Fin 4,
    let boundary := [2097152, 3014656, 3145728, 6029312][fraction.val]!
    let word := (2 * exponent.val + 1) * 2^23 + boundary
    Mufu.sqrt (word - 1) = Mufu.sqrt word + 1 := by decide +kernel

#print axioms sqrt_normalized_segment_boundaries
#print axioms sqrt_scaled_boundary_drops

end GB300Math.MufuProperties
