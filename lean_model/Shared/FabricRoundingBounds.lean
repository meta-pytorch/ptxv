import FabricRounding
import Mathlib.Tactic

set_option maxRecDepth 100000
set_option maxHeartbeats 0

namespace GB300Math.FabricRoundingBounds
open FloatLib.Floats.Formats.BinaryInterchange
open GB300Math.FabricRounding

def value (word : Nat) : Option Rat :=
  Model.toRat? (Model.ofNatBits word : Model FloatFormat.bfloat16)

theorem input_values : value 0xbfd8 = some (-27/16) ∧ value 0x4343 = some 195 := by
  decide +kernel

theorem output_values : value 0x4341 = some 193 ∧ value 0x4342 = some 194 := by
  decide +kernel

theorem nearest_result : candidates FloatFormat.bfloat16 [0xbfd8, 0x4343] true = some [0x4341] := by
  decide +kernel

theorem enclosing_results : candidates FloatFormat.bfloat16 [0xbfd8, 0x4343] false = some [0x4341, 0x4342] := by
  decide +kernel

theorem exact_errors :
    |(193 : Rat) - (-27/16 + 195)| = 5/16 ∧
    |(194 : Rat) - (-27/16 + 195)| = 11/16 ∧
    (11/16 : Rat) / (5/16) = 11/5 := by
  norm_num

theorem wider_than_half_ulp : (1/2 : Rat) < 11/16 ∧ (11/16 : Rat) < 1 := by
  norm_num

theorem neighbor_choice_bound (word : Nat)
    (h : word ∈ [0x4341, 0x4342]) :
    ∃ result : Rat, value word = some result ∧ |result - (-27/16 + 195)| ≤ 11/16 := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with h | h
  · subst word
    exact ⟨193, output_values.1, by norm_num⟩
  · subst word
    exact ⟨194, output_values.2, by norm_num⟩

#print axioms input_values
#print axioms output_values
#print axioms nearest_result
#print axioms enclosing_results
#print axioms exact_errors
#print axioms wider_than_half_ulp
#print axioms neighbor_choice_bound
end GB300Math.FabricRoundingBounds
