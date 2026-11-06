import GB300Math.TensorBlackwell

namespace GB300Math.TensorMixed16
open GB300Math.Tensor

def alignedDot (a b : Array Nat) (c afw aew bfw bew : Nat) : Nat := Id.run do
  let accumulator := decode c 23 8
  let mut nan := isNaN c 23 8
  let mut positiveInf := isInf c 23 8 && c < 2^31
  let mut negativeInf := isInf c 23 8 && c ≥ 2^31
  let accumulatorTop := if accumulator.significand = 0 then -10000
    else max (-126) (inputTop accumulator)
  let mut terms : Array (GB300Math.Tensor.Dyadic × Int) := #[(accumulator, accumulatorTop)]
  for i in [:a.size] do
    let ax := a[i]!
    let bx := b[i]!
    nan := nan || isNaN ax afw aew || isNaN bx bfw bew ||
      (isInf ax afw aew && isZero bx bfw bew) || (isZero ax afw aew && isInf bx bfw bew)
    if isInf ax afw aew || isInf bx bfw bew then
      let negative := ax / 2^(afw+aew) % 2 != bx / 2^(bfw+bew) % 2
      positiveInf := positiveInf || !negative
      negativeInf := negativeInf || negative
    let x := decode ax afw aew
    let y := decode bx bfw bew
    terms := terms.push (⟨x.significand*y.significand, x.exponent+y.exponent⟩,
      if x.significand = 0 ∨ y.significand = 0 then -10000
      else x.exponent+y.exponent+afw+bfw)
  if nan || (positiveInf && negativeInf) then return 0x7fffffff
  if positiveInf || negativeInf then
    return (if negativeInf then 2^31 else 0) + 0x7f800000
  let top := terms.foldl (fun s t => max s t.2) (-10000)
  let scale := max (top - 25) (-158)
  let total := terms.foldl
    (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-scale)) 0
  return encodeRz ⟨total, scale⟩

def evaluate (afmt bfmt : String) (a b : Array Nat) (c : Nat) : Option Nat := do
  let (afw, aew) ← match afmt with
    | "f16" => some (10,5)
    | "bf16" => some (7,8)
    | _ => none
  let (bfw, bew) ← match bfmt with
    | "f16" => some (10,5)
    | "bf16" => some (7,8)
    | _ => none
  if a.size != 16 || b.size != 16 || c ≥ 2^32 then none
  else if !(a ++ b).all (fun x => x < 2^16) then none
  else some (alignedDot a b c afw aew bfw bew)

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem mixed_unit_product :
    alignedDot #[0x3c00] #[0x3f80] 0 10 5 7 8 = 0x3f800000 ∧
    alignedDot #[0x3f80] #[0x3c00] 0 7 8 10 5 = 0x3f800000 := by decide +kernel

theorem wrong_type_changes_value :
    alignedDot #[0x3c00] #[0x3f80] 0 10 5 10 5 = 0x3ff00000 := by decide +kernel

theorem invalid_operand_extent_rejected :
    evaluate "f16" "bf16" #[0] #[] 0 = none := by decide +kernel

#print axioms mixed_unit_product
#print axioms wrong_type_changes_value
#print axioms invalid_operand_extent_rejected

end GB300Math.TensorMixed16
