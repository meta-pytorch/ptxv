import GB300Math.TensorLegacyBlackwell
import GB300Math.TensorSmallBlackwell
import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.TensorLegacyFP8
open GB300Math.Tensor
open FloatLib.Floats.Formats.BinaryInterchange

def widen (fmt : String) (x : Nat) : Nat :=
  if TensorSmallBlackwell.nan fmt x then 0x7fff
  else if TensorSmallBlackwell.infinity fmt x then (if x ≥ 128 then 32768 else 0) + 0x7c00
  else if x % 128 = 0 then (if x ≥ 128 then 32768 else 0)
  else
    let value := if fmt = "e4m3" then decode x 3 4 else decode x 2 5
    SparseTensorExtra.roundHalf value.significand value.exponent

def add (half : Bool) (a b : Nat) : Nat :=
  let fmt := if half then FloatFormat.binary16 else FloatFormat.binary32
  let value := Model.addWithRounding .nearestEven
    (Model.ofNatBits a : Model fmt) (Model.ofNatBits b)
  if Model.isNaN value then (if half then 0x7fff else 0x7fffffff)
  else Model.toNatBits value

def dot (afmt bfmt : String) (half : Bool) (a b : Array Nat) (c : Nat) : Option Nat := do
  if !(afmt = "e4m3" || afmt = "e5m2") || !(bfmt = "e4m3" || bfmt = "e5m2") ||
      !(a.size = 16 || a.size = 32) || b.size != a.size ||
      !(a++b).all (· < 256) || c ≥ (if half then 65536 else 2^32) then none else
  let av := a.map (widen afmt)
  let bv := b.map (widen bfmt)
  let low := (Array.range a.size).filter (fun j => j%4 < 2)
  let high := (Array.range a.size).filter (fun j => j%4 ≥ 2)
  let native := fun aa bb cc =>
    if half then TensorLegacyBlackwell.halfDot aa bb cc
    else TensorBlackwell.alignedDot aa bb cc 10 5 true
  let first := native (low.map (av[·]!)) (low.map (bv[·]!)) 0
  let second := native (high.map (av[·]!)) (high.map (bv[·]!)) first
  return add half c second

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem widening_preserves_negative_zero : widen "e4m3" 128 = 32768 := by decide +kernel

theorem fp8_largest_finite_widens : widen "e4m3" 126 = 0x5f00 := by decide +kernel

theorem unit_dot : dot "e4m3" "e5m2" false
    (Array.replicate 16 56) (Array.replicate 16 60) 0 = some 0x41800000 := by decide +kernel

#print axioms widening_preserves_negative_zero
#print axioms fp8_largest_finite_widens
#print axioms unit_dot
end GB300Math.TensorLegacyFP8
