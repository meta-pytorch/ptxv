import GB300Math.TextureGradient
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace GB300Math.TextureGradientBounds
open TextureGradient
def Certificate (i q : Nat) : Prop :=
  2^(4096+2*q) ≤ 8*(256+i)^512 ∧
  (257+i)^512 ≤ 8*2^(4096+2*q)
instance (i q : Nat) : Decidable (Certificate i q) := by unfold Certificate; infer_instance
def valid (i : Nat) : Bool := decide (Certificate i logTable[i]!)
theorem all_checked : (List.range 256).all valid = true := by decide +kernel
theorem all_certificates (i : Nat) (hi : i ∈ List.range 256) :
    Certificate i logTable[i]! := by
  have h := List.all_eq_true.mp all_checked i hi
  simpa [valid] using h
#print axioms all_checked
#print axioms all_certificates
end GB300Math.TextureGradientBounds
