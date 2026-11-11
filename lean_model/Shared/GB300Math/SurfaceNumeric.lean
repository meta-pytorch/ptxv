import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.SurfaceNumeric
open FloatLib.Floats.Formats.BinaryInterchange

def signed (width bits : Nat) : Int :=
  if bits % 2^width < 2^(width-1) then bits % 2^width
  else (bits % 2^width : Int) - 2^width

def encode (width : Nat) (value : Int) : Nat := (value % (2^width : Nat)).toNat

def half (bits : Nat) : Nat :=
  let x : Model FloatFormat.binary32 := Model.ofNatBits bits
  if Model.isNaN x then
    (bits / 2^31) * 32768 + 31744 + max 1 (bits % 2^23 / 8192)
  else Model.toNatBits (Model.castWithRounding _ FloatFormat.binary16 x .towardZero)

def fixedMagnitude (precision bits : Nat) : Nat :=
  let magnitude := bits % 2^31
  let exponent := magnitude / 2^23
  if magnitude > 0x7f800000 then 0
  else if exponent ≥ 127 then 2^precision
  else if exponent = 0 then 0
  else (2^23 + magnitude % 2^23) / 2^(150-exponent-precision)

def quantize (precision maximum fixed : Nat) : Nat :=
  (fixed * maximum + 2^(precision-1) - 1) / 2^precision

def normalized (width : Nat) (isSigned : Bool) (bits : Nat) : Nat :=
  let negative := bits / 2^31 = 1
  let magnitudeBits := if isSigned then width-1 else width
  let value := quantize (magnitudeBits+4) (2^magnitudeBits-1)
    (fixedMagnitude (magnitudeBits+4) bits)
  if negative then if isSigned then encode width (-(value : Int)) else 0
  else value

def store (kind : String) (bits : Nat) : Nat :=
  match kind with
  | "u8" => min bits 255
  | "u16" => min bits 65535
  | "s8" => encode 8 (max (-128) (min 127 (signed 32 bits)))
  | "s16" => encode 16 (max (-32768) (min 32767 (signed 32 bits)))
  | "f16" => half bits
  | "unorm8" => normalized 8 false bits
  | "unorm16" => normalized 16 false bits
  | "snorm8" => normalized 8 true bits
  | "snorm16" => normalized 16 true bits
  | "unorm10" => normalized 10 false bits
  | "unorm2" => min 3 ((normalized 10 false bits + 128) / 320)
  | _ => bits

def reduce (width : Nat) (isSigned : Bool) (operation : String) (a b : Nat) : Nat :=
  let less := if isSigned then signed width a < signed width b else a < b
  match operation with
  | "add" => (a+b) % 2^width
  | "min" => if less then a else b
  | "max" => if less then b else a
  | "and" => a &&& b
  | "or" => a ||| b
  | _ => 0

theorem round_error (numerator halfScale : Nat) (h : 0 < halfScale) :
    let q := (numerator + halfScale - 1) / (2*halfScale)
    ((q * (2*halfScale) : Nat) : Int) - numerator ≤ halfScale ∧
    (numerator : Int) - ((q * (2*halfScale) : Nat) : Int) ≤ halfScale := by
  have hm := Nat.mod_lt (numerator + halfScale - 1) (by omega : 0 < 2*halfScale)
  have hd := Nat.mod_add_div (numerator + halfScale - 1) (2*halfScale)
  simp only at *
  rw [Nat.mul_comm (2*halfScale)] at hd
  omega

theorem unorm8_half : store "unorm8" 0x3f000000 = 127 := by decide
theorem unorm16_half : store "unorm16" 0x3f000000 = 32767 := by decide
theorem snorm8_negative_half : store "snorm8" 0xbf000000 = 193 := by decide
theorem normalized_nan : store "unorm16" 0x7fffffff = 0 := by decide
theorem snorm_negative_infinity : store "snorm8" 0xff800000 = 129 := by decide
set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem half_finite_overflow : half 0x7f7fffff = 0x7bff := by decide
theorem half_signaling_nan : half 0xff800001 = 0xfc01 := by decide

def word (bytes : ByteArray) (width index : Nat) : Nat :=
  (List.range width).foldl (fun x i => x + bytes[width*index+i]!.toNat * 2^(8*i)) 0

end GB300Math.SurfaceNumeric

#print axioms GB300Math.SurfaceNumeric.round_error
#print axioms GB300Math.SurfaceNumeric.unorm8_half
#print axioms GB300Math.SurfaceNumeric.unorm16_half
#print axioms GB300Math.SurfaceNumeric.snorm8_negative_half
#print axioms GB300Math.SurfaceNumeric.normalized_nan
#print axioms GB300Math.SurfaceNumeric.snorm_negative_infinity
#print axioms GB300Math.SurfaceNumeric.half_finite_overflow
#print axioms GB300Math.SurfaceNumeric.half_signaling_nan

def main (args : List String) : IO UInt32 := do
  let [kind, widthText, signedText, input, initial, output] := args |
    throw (IO.userError "Expected kind, width, signed flag, input, initial, output")
  let some width := widthText.toNat? | throw (IO.userError "Invalid width")
  unless width ∈ [8,16,32,64] do throw (IO.userError "Unsupported width")
  unless signedText ∈ ["0", "1"] do throw (IO.userError "Invalid signed flag")
  let xs ← IO.FS.readBinFile input
  let cs ← IO.FS.readBinFile initial
  let ys ← IO.FS.readBinFile output
  let reduction := kind ∈ ["add", "min", "max", "and", "or"]
  unless reduction || kind ∈ ["u8", "u16", "u32", "s8", "s16", "s32", "f16", "f32",
    "unorm8", "unorm16", "snorm8", "snorm16", "unorm10", "unorm2"] do
    throw (IO.userError "Unknown operation")
  let inWidth := if reduction then width/8 else 4
  unless ys.size > 0 && ys.size % (width/8) = 0 do
    throw (IO.userError "Invalid output shape")
  let count := ys.size / (width/8)
  unless xs.size ≥ count*inWidth && (!reduction || cs.size ≥ count*inWidth) do
    throw (IO.userError "Invalid input shape")
  let mut failures := 0
  for i in [:count] do
    let x := GB300Math.SurfaceNumeric.word xs inWidth i
    let y := GB300Math.SurfaceNumeric.word ys (width/8) i
    let expected := if reduction then
      GB300Math.SurfaceNumeric.reduce width (signedText == "1") kind
        (GB300Math.SurfaceNumeric.word cs inWidth i) x
      else GB300Math.SurfaceNumeric.store kind x
    if y != expected then
      if failures < 4 then IO.println s!"witness index={i} input={x} expected={expected} actual={y}"
      failures := failures+1
  IO.println s!"result count={count} mismatches={failures}"
  return if failures = 0 then 0 else 1
