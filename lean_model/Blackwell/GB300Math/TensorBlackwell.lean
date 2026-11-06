import GB300Math.TensorCore

namespace GB300Math.TensorBlackwell
open GB300Math.Tensor

def alignedDot (a b : Array Nat) (c fw ew : Nat)
    (clampSubnormalAccumulator : Bool) : Nat := Id.run do
  let accumulator := decode c 23 8
  let mut nan := isNaN c 23 8
  let mut positiveInf := isInf c 23 8 && c < 2^31
  let mut negativeInf := isInf c 23 8 && c ≥ 2^31
  let accumulatorTop := if accumulator.significand = 0 then -10000
    else if clampSubnormalAccumulator then max (-126) (inputTop accumulator)
    else inputTop accumulator
  let mut terms : Array (GB300Math.Tensor.Dyadic × Int) := #[(accumulator, accumulatorTop)]
  for i in [:a.size] do
    let ax := a[i]!
    let bx := b[i]!
    nan := nan || isNaN ax fw ew || isNaN bx fw ew ||
      (isInf ax fw ew && isZero bx fw ew) || (isZero ax fw ew && isInf bx fw ew)
    if isInf ax fw ew || isInf bx fw ew then
      let negative := ax / 2^(fw+ew) % 2 != bx / 2^(fw+ew) % 2
      positiveInf := positiveInf || !negative
      negativeInf := negativeInf || negative
    let x := decode ax fw ew
    let y := decode bx fw ew
    terms := terms.push (⟨x.significand*y.significand, x.exponent+y.exponent⟩,
      if x.significand = 0 ∨ y.significand = 0 then -10000
      else x.exponent+y.exponent+2*fw)
  if nan || (positiveInf && negativeInf) then return 0x7fffffff
  if positiveInf || negativeInf then
    return (if negativeInf then 2^31 else 0) + 0x7f800000
  let top := terms.foldl (fun s t => max s t.2) (-10000)
  let scale := top - 25
  let total := terms.foldl
    (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-scale)) 0
  return encodeRz ⟨total, scale⟩

def evaluate (fmt : String) (a b : Array Nat) (c : Nat)
    (clampSubnormalAccumulator : Bool) : Option Nat := do
  let (fw, ew, k) ← match fmt with
    | "f16" => some (10,5,16)
    | "bf16" => some (7,8,16)
    | "tf32" => some (23,8,8)
    | _ => none
  if a.size != k || b.size != k || c ≥ 2^32 then none
  else if !(a ++ b).all (fun x => x < 2^(fw+ew+1)) then none
  else if fmt = "tf32" && !(a ++ b).all (fun x => x % 2^13 = 0) then none
  else some (alignedDot a b c fw ew clampSubnormalAccumulator)

theorem alignment_truncation_bound (n shift : Nat) :
    n / 2^shift * 2^shift ≤ n ∧
    n < n / 2^shift * 2^shift + 2^shift := by
  have h := Nat.mod_add_div n (2^shift)
  have hr := Nat.mod_lt n (Nat.two_pow_pos shift)
  rw [Nat.mul_comm (2^shift) (n / 2^shift)] at h
  omega

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem alignment_preserves_residual :
    alignedDot #[0x3c00,0xbc00,0x0c00] #[0x3c00,0x3c00,0x0800] 0 10 5 false =
      0x33000000 := by
  decide +kernel

theorem subnormal_alignment_candidates_differ :
    alignedDot #[0x9980] #[0x1980] 1 7 8 false = 0 ∧
    alignedDot #[0x9980] #[0x1980] 1 7 8 true = 1 := by
  decide +kernel

theorem invalid_operand_extent_rejected :
    evaluate "f16" #[0] #[] 0 false = none := by decide +kernel

#print axioms alignment_truncation_bound
#print axioms alignment_preserves_residual
#print axioms subnormal_alignment_candidates_differ
#print axioms invalid_operand_extent_rejected

end GB300Math.TensorBlackwell
