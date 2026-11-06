import GB300Math.TensorSmallBlackwell

namespace GB300Math.TensorScaledSmall
open GB300Math.Tensor
open GB300Math.TensorSmallBlackwell (format nan infinity)

def dot (afmt bfmt : String) (a b : Array Nat) (c sa sb : Nat) (margin : Option Nat) : Option Nat := do
  let (af, ae) ← format afmt
  let (bf, be) ← format bfmt
  if sa ≥ 256 || sb ≥ 256 || a.size != 32 || b.size != 32 || c ≥ 2^32 ||
      !a.all (· < 2^(af+ae+1)) || !b.all (· < 2^(bf+be+1)) then none else
  some (Id.run do
    let accumulator := decode c 23 8
    let mut isNan := isNaN c 23 8 || sa == 255 || sb == 255
    let scale : Int := (sa : Int) + (sb : Int) - 254
    let mut positiveInf := isInf c 23 8 && c < 2^31
    let mut negativeInf := isInf c 23 8 && c ≥ 2^31
    let ctop := if accumulator.significand = 0 then -10000 else max (-126) (inputTop accumulator)
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
        else x.exponent+y.exponent+af+bf+scale
      terms := terms.push (⟨x.significand*y.significand,x.exponent+y.exponent+scale⟩,top)
    if isNan || (positiveInf && negativeInf) then return 0x7fffffff
    if positiveInf || negativeInf then return (if negativeInf then 2^31 else 0)+0x7f800000
    let top := terms.foldl (fun s t => max s t.2) (-10000)
    let exponent := match margin with
      | some n => max (top-(n:Int)) (-158)
      | none => terms.foldl (fun e t => min e t.1.exponent) 0
    let coefficient := terms.foldl
      (fun s t => s + shiftTruncate t.1.significand (t.1.exponent-exponent)) 0
    return encodeRz ⟨coefficient,exponent⟩)

set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

theorem unit_scale_dot :
    dot "e4m3" "e2m1" (Array.replicate 32 56) (Array.replicate 32 2) 0 127 127 (some 25) =
      some 0x42000000 := by decide +kernel

theorem independent_scales_multiply :
    dot "e4m3" "e2m1" (Array.replicate 32 56) (Array.replicate 32 2) 0 128 128 (some 25) =
      some 0x43000000 := by decide +kernel

theorem scale_nan_propagates :
    dot "e2m1" "e2m1" (Array.replicate 32 0) (Array.replicate 32 0) 0 255 127 none =
      some 0x7fffffff := by decide +kernel

#print axioms unit_scale_dot
#print axioms independent_scales_multiply
#print axioms scale_nan_propagates
end GB300Math.TensorScaledSmall
