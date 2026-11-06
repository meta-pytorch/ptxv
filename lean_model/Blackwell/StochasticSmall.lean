import Std.Tactic

namespace GB300Math.StochasticSmall

inductive Format where
  | e4m3 | e5m2 | e2m3 | e3m2 | e2m1
  deriving DecidableEq, Repr

def parameters : Format → Nat × Nat × Nat × Nat
  | .e4m3 => (4, 3, 7, 126)
  | .e5m2 => (5, 2, 15, 123)
  | .e2m3 => (2, 3, 1, 31)
  | .e3m2 => (3, 2, 3, 31)
  | .e2m1 => (2, 1, 1, 7)

def reverseBits : Nat → Nat → Nat
  | 0, _ => 0
  | n + 1, x => (x % 2) * 2^n + reverseBits n (x / 2)

theorem reverseBits_bound (n x : Nat) : reverseBits n x < 2^n := by
  induction n generalizing x with
  | zero => simp [reverseBits]
  | succ n ih =>
    have h := ih (x / 2)
    have hx : x % 2 = 0 ∨ x % 2 = 1 := by omega
    rcases hx with hx | hx <;> simp [reverseBits, hx, Nat.pow_succ] <;> omega

def entropyLane (f : Format) (bits lane : Nat) : Nat :=
  let r := if f == .e2m1 then
      let shifted := bits / 2^(8*(lane / 2))
      shifted % 256 + (shifted / 65536 % 256) * 256
    else bits / 2^(16*(lane / 2)) % 65536
  if lane % 2 = 0 then r else reverseBits 16 r

def roundWithNoise (mantissa noise step : Nat) : Nat := (mantissa + noise) / step

theorem roundWithNoise_exact_grid (q noise step : Nat) (hn : noise < step) :
    roundWithNoise (q * step) noise step = q := by
  unfold roundWithNoise
  rw [Nat.mul_comm q, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hn, Nat.add_zero]

theorem roundWithNoise_error (mantissa noise step : Nat) (hn : noise < step) :
    roundWithNoise mantissa noise step * step < mantissa + step ∧
      mantissa < roundWithNoise mantissa noise step * step + step := by
  have hRemainder := Nat.mod_lt (mantissa + noise) (show 0 < step by omega)
  have hDivision := Nat.mod_add_div (mantissa + noise) step
  rw [Nat.mul_comm step] at hDivision
  unfold roundWithNoise
  omega

def roundMantissa (mantissa entropy drop : Nat) : Nat :=
  roundWithNoise mantissa ((entropy % 65536) * 2^(drop - 16)) (2^drop)

theorem noise_bound (entropy drop : Nat) (h : 16 ≤ drop) :
    (entropy % 65536) * 2^(drop - 16) < 2^drop := by
  have hpow : 2^drop = 65536 * 2^(drop - 16) := by
    calc
      2^drop = 2^(16 + (drop - 16)) := by congr 1 <;> omega
      _ = 65536 * 2^(drop - 16) := by rw [Nat.pow_add]
  rw [hpow]
  exact Nat.mul_lt_mul_of_pos_right (Nat.mod_lt _ (by decide)) (Nat.two_pow_pos _)

theorem roundMantissa_exact_grid (q entropy drop : Nat) (h : 16 ≤ drop) :
    roundMantissa (q * 2^drop) entropy drop = q :=
  roundWithNoise_exact_grid q _ _ (noise_bound entropy drop h)

theorem roundMantissa_error (mantissa entropy drop : Nat) (h : 16 ≤ drop) :
    roundMantissa mantissa entropy drop * 2^drop < mantissa + 2^drop ∧
      mantissa < roundMantissa mantissa entropy drop * 2^drop + 2^drop :=
  roundWithNoise_error mantissa _ _ (noise_bound entropy drop h)

def lane (f : Format) (input entropy : Nat) : Nat :=
  let (exponentBits, fractionBits, bias, limit) := parameters f
  let x := input % 2^32
  let sign := (x / 2^31) * 2^(exponentBits + fractionBits)
  let exponent := x / 2^23 % 256
  let fraction := x % 2^23
  if exponent = 255 then
    if fraction ≠ 0 then
      if f == .e4m3 || f == .e5m2 then 127 else limit
    else sign + limit
  else
    let mantissa := fraction + if exponent = 0 then 0 else 2^23
    let targetExponent : Int := ((max exponent 1 : Nat) : Int) - 127 + bias
    let drop := 23 - fractionBits + (max 0 (1 - targetExponent)).toNat
    let value := (max 0 (targetExponent - 1)).toNat * 2^fractionBits +
      roundMantissa mantissa entropy drop
    sign + min value limit

def packed (f : Format) (a b c d entropy : Nat) : Nat :=
  let width := if f == .e2m1 then 4 else 8
  lane f a (entropyLane f entropy 0) +
    lane f b (entropyLane f entropy 1) * 2^width +
    lane f c (entropyLane f entropy 2) * 2^(2*width) +
    lane f d (entropyLane f entropy 3) * 2^(3*width)

example : entropyLane .e4m3 0x89abcdef 1 = 0xf7b3 := by decide
example : entropyLane .e2m1 0x89abcdef 0 = 0xabef := by decide
example : entropyLane .e2m1 0x89abcdef 2 = 0x89cd := by decide
example : lane .e2m1 0xffc00001 0 = 7 := by decide
example : lane .e4m3 0x80000000 65535 = 128 := by decide
example : lane .e5m2 0x7f800000 0 = 123 := by decide

def readWord (data : ByteArray) (offset bytes : Nat) : Nat := Id.run do
  let mut value := 0
  for j in [:bytes] do value := value + data[offset + j]!.toNat * 2^(8*j)
  return value

end GB300Math.StochasticSmall

open GB300Math.StochasticSmall

def main (args : List String) : IO UInt32 := do
  let [format, input, output] := args
    | throw (IO.userError "Expected format input output")
  let f ← match format with
    | "e4m3" => pure Format.e4m3
    | "e5m2" => pure Format.e5m2
    | "e2m3" => pure Format.e2m3
    | "e3m2" => pure Format.e3m2
    | "e2m1" => pure Format.e2m1
    | _ => throw (IO.userError "Unknown stochastic format")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 32 = 0 && ys.size = xs.size / 4 do
    throw (IO.userError "Expected four uint64 input words per uint64 output")
  let mut mismatches := 0
  for i in [:xs.size / 32] do
    let a := readWord xs (32*i) 8
    let model := packed f a (readWord xs (32*i+8) 8)
      (readWord xs (32*i+16) 8) (readWord xs (32*i+24) 8) (a / 2^32)
    let actual := readWord ys (8*i) 8
    if model ≠ actual then
      if mismatches < 3 then IO.println s!"witness row={i} model={model} actual={actual}"
      mismatches := mismatches + 1
  IO.println s!"result {format} cases={xs.size / 32} lane_results={xs.size / 8} mismatches={mismatches}"
  return if mismatches = 0 then 0 else 1
