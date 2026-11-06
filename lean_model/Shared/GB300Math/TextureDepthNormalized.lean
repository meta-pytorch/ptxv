import GB300Math.TextureDepthCompare

namespace GB300Math.TextureDepthNormalized

open TextureDepthCompare

def fixed20 (word : Nat) : Nat :=
  if word / 2^31 = 1 || isNaN word then 0
  else if word ≥ 0x3f800000 then 2^20
  else if word / 2^23 = 0 then 0
  else (2^23 + word % 2^23) / 2^(130-word/2^23)

def quantize (value : Nat) : Nat :=
  (2*value*65535 + 2^20-1) / 2^21

def reference (word : Nat) : Nat := quantize (fixed20 word)

def threshold (code : Nat) : Nat :=
  if code = 0 then 0 else (2*code-1)*2^19 / 65535 + 1

def cmp (function value depth : Nat) : Bool :=
  match function with
  | 0 => false
  | 1 => value < depth
  | 2 => value == depth
  | 3 => value ≤ depth
  | 4 => value > depth
  | 5 => value != depth
  | 6 => value ≥ depth
  | _ => true

def compare (function word depth : Nat) : Nat :=
  if cmp function (reference word) depth then 0x3f800000 else 0

theorem quantize_bounded (value : Nat) (h : value ≤ 2^20) :
    quantize value ≤ 65535 := by
  unfold quantize
  omega

theorem threshold_characterization (code value : Nat)
    (hc : 0 < code) (hv : value ≤ 2^20) :
    threshold code ≤ value ↔ code ≤ quantize value := by
  simp only [threshold, Nat.ne_of_gt hc, ↓reduceIte, quantize]
  omega

theorem threshold_strict (a b : Nat) (ha : 0 < a) (hab : a < b) :
    threshold a < threshold b := by
  simp only [threshold, Nat.ne_of_gt ha, Nat.ne_of_gt (by omega : 0 < b), ↓reduceIte]
  omega

theorem reference_error (units : Nat) :
    let result := quantize (units / 2^129);
    -((2^148 + 65535*2^129 : Nat) : Int) <
      (result : Int)*2^149 - (units : Int)*65535 ∧
    (result : Int)*2^149 - (units : Int)*65535 ≤ (2^148 : Int) := by
  simp only [quantize]
  omega

theorem half_tie_down : reference 0x3f000000 = 32767 := by decide +kernel

theorem half_threshold : reference 0x3f000010 = 32768 := by decide +kernel

theorem zero_nan : reference 0x7fc00000 = 0 := by decide +kernel

theorem positive_infinity : reference 0x7f800000 = 65535 := by decide +kernel

theorem negative_infinity : reference 0xff800000 = 0 := by decide +kernel

end GB300Math.TextureDepthNormalized
