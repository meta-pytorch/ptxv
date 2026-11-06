import GB300Math.TensorScaledBlackwell

namespace GB300Math.TensorScaledCancellation
open GB300Math.Tensor GB300Math.TensorScaledIdeal

def a : Array Nat := (Array.range 96).map fun j =>
  if j = 32 ∨ j = 48 then 1 else if j = 33 ∨ j = 49 then 2
  else if j = 64 ∨ j = 65 then 10 else 0

def b : Array Nat := (Array.range 96).map fun j =>
  if j = 32 ∨ j = 48 ∨ j = 64 then 1 else if j = 33 ∨ j = 49 then 3
  else if j = 65 then 5 else 0

def ratio (x : GB300Math.Tensor.Dyadic) : Rat :=
  if x.exponent ≥ 0 then (x.significand : Rat) * 2^x.exponent.toNat
  else (x.significand : Rat) / 2^(-x.exponent).toNat

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

theorem ideal_cancels :
    (exactDot false a b #[254,254,254] #[254,219,219] 0).map ratio = some 0 := by
  decide +kernel

theorem selected_model_overflows :
    TensorScaledBlackwell.evaluate false a b #[254,254,254] #[254,219,219] 0 (-10000)
      = some 0xff800000 := by
  decide +kernel

theorem removing_unused_large_scale_restores_zero :
    TensorScaledBlackwell.evaluate false a b #[219,254,254] #[219,219,219] 0 (-10000)
      = some 0 := by
  decide +kernel

#print axioms ideal_cancels
#print axioms selected_model_overflows
#print axioms removing_unused_large_scale_restores_zero
end GB300Math.TensorScaledCancellation
