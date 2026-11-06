import GB300Math.TensorSmallBlackwell
import GB300Math.SparseTensorExtra

namespace GB300Math.TensorSmallHalf
open GB300Math.Tensor GB300Math.TensorSmallBlackwell

def dot (afmt bfmt : String) (a b : Array Nat) (c : Nat) (viaF32 : Bool := false) : Option Nat := do
  let (af, ae) ← format afmt
  let (bf, be) ← format bfmt
  if a.size != 32 || b.size != 32 || c ≥ 2^16 ||
      !a.all (· < 2^(af+ae+1)) || !b.all (· < 2^(bf+be+1)) then none else
  some (Id.run do
    let accumulator := decode c 10 5
    let mut isNan := isNaN c 10 5
    let mut positiveInf := isInf c 10 5 && c < 2^15
    let mut negativeInf := isInf c 10 5 && c ≥ 2^15
    let ctop := if accumulator.significand = 0 then -10000 else max (-14) (inputTop accumulator)
    let mut terms : Array (GB300Math.Tensor.Dyadic × Int) := #[(accumulator,ctop)]
    for i in [:32] do
      let ax := a[i]!
      let bx := b[i]!
      let ai := infinity afmt ax
      let bi := infinity bfmt bx
      isNan := isNan || nan afmt ax || nan bfmt bx ||
        (ai && isZero bx bf be) || (bi && isZero ax af ae)
      if ai || bi then
        let negative := ax / 2^(af+ae) != bx / 2^(bf+be)
        positiveInf := positiveInf || !negative
        negativeInf := negativeInf || negative
      let x := decode ax af ae
      let y := decode bx bf be
      let top := if x.significand = 0 || y.significand = 0 then -10000
        else x.exponent+y.exponent+af+bf
      terms := terms.push (⟨x.significand*y.significand,x.exponent+y.exponent⟩,top)
    if isNan || (positiveInf && negativeInf) then return 0x7fff
    if positiveInf || negativeInf then return (if negativeInf then 2^15 else 0)+0x7c00
    let top := terms.foldl (fun s t => max s t.2) (-10000)
    let exponent := top-25
    let coefficient := terms.foldl
      (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-exponent)) 0
    if viaF32 then
      let intermediate := encodeRz ⟨coefficient,exponent⟩
      let decoded := decode intermediate 23 8
      return SparseTensorExtra.roundHalf decoded.significand decoded.exponent
    return SparseTensorExtra.roundHalf coefficient exponent)

set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

def witnessA := #[40,12] ++ Array.replicate 30 0
def witnessB := #[36,8] ++ Array.replicate 30 0

theorem direct_rounding_witness :
    dot "e5m2" "e5m2" witnessA witnessB 0x3c00 false = some 0x3c01 := by decide +kernel

theorem double_rounding_witness :
    dot "e5m2" "e5m2" witnessA witnessB 0x3c00 true = some 0x3c00 := by decide +kernel

#print axioms direct_rounding_witness
#print axioms double_rounding_witness
end GB300Math.TensorSmallHalf
