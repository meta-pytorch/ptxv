import FloatLib.Floats.Formats.BinaryInterchange.Configured
import Mathlib.Tactic.IntervalCases
import SmallConversions

set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000000

namespace GB300Math.S2F6FloatLibAccuracy

open FloatLib.Floats GB300Math.SmallConversions

def ideal (word scale : Nat) : ℚ :=
  let signed : Int := if word < 128 then word else (word : Int)-256
  (signed : ℚ) * (2 : ℚ)^((scale : Int)-133)

def decoded (word scale : Nat) : Option ℚ :=
  let bits := upS2F6 false false true word scale
  ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits bits : ExecFloat.Binary 8 7)

def expected (word scale : Nat) : Option ℚ :=
  if word = 128 ∧ scale = 254 then none else some (ideal word scale)

private theorem block_0 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (0+low.val) scale.val = expected (0+low.val) scale.val := by decide +kernel

private theorem block_1 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (16+low.val) scale.val = expected (16+low.val) scale.val := by decide +kernel

private theorem block_2 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (32+low.val) scale.val = expected (32+low.val) scale.val := by decide +kernel

private theorem block_3 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (48+low.val) scale.val = expected (48+low.val) scale.val := by decide +kernel

private theorem block_4 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (64+low.val) scale.val = expected (64+low.val) scale.val := by decide +kernel

private theorem block_5 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (80+low.val) scale.val = expected (80+low.val) scale.val := by decide +kernel

private theorem block_6 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (96+low.val) scale.val = expected (96+low.val) scale.val := by decide +kernel

private theorem block_7 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (112+low.val) scale.val = expected (112+low.val) scale.val := by decide +kernel

private theorem block_8 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (128+low.val) scale.val = expected (128+low.val) scale.val := by decide +kernel

private theorem block_9 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (144+low.val) scale.val = expected (144+low.val) scale.val := by decide +kernel

private theorem block_10 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (160+low.val) scale.val = expected (160+low.val) scale.val := by decide +kernel

private theorem block_11 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (176+low.val) scale.val = expected (176+low.val) scale.val := by decide +kernel

private theorem block_12 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (192+low.val) scale.val = expected (192+low.val) scale.val := by decide +kernel

private theorem block_13 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (208+low.val) scale.val = expected (208+low.val) scale.val := by decide +kernel

private theorem block_14 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (224+low.val) scale.val = expected (224+low.val) scale.val := by decide +kernel

private theorem block_15 : ∀ low : Fin 16, ∀ scale : Fin 255,
    decoded (240+low.val) scale.val = expected (240+low.val) scale.val := by decide +kernel

theorem scaled_exact_except_overflow : ∀ code : Fin 256, ∀ scale : Fin 255,
    decoded code.val scale.val = expected code.val scale.val := by
  intro code scale
  have hcode := code.isLt
  have hlow : code.val % 16 < 16 := Nat.mod_lt _ (by decide)
  have hdiv : code.val / 16 ≤ 15 := by omega
  interval_cases h : code.val / 16
  · have heq : 0 + code.val % 16 = code.val := by omega
    have proof := block_0 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 16 + code.val % 16 = code.val := by omega
    have proof := block_1 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 32 + code.val % 16 = code.val := by omega
    have proof := block_2 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 48 + code.val % 16 = code.val := by omega
    have proof := block_3 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 64 + code.val % 16 = code.val := by omega
    have proof := block_4 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 80 + code.val % 16 = code.val := by omega
    have proof := block_5 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 96 + code.val % 16 = code.val := by omega
    have proof := block_6 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 112 + code.val % 16 = code.val := by omega
    have proof := block_7 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 128 + code.val % 16 = code.val := by omega
    have proof := block_8 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 144 + code.val % 16 = code.val := by omega
    have proof := block_9 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 160 + code.val % 16 = code.val := by omega
    have proof := block_10 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 176 + code.val % 16 = code.val := by omega
    have proof := block_11 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 192 + code.val % 16 = code.val := by omega
    have proof := block_12 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 208 + code.val % 16 = code.val := by omega
    have proof := block_13 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 224 + code.val % 16 = code.val := by omega
    have proof := block_14 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof
  · have heq : 240 + code.val % 16 = code.val := by omega
    have proof := block_15 ⟨code.val % 16, hlow⟩ scale
    simpa only [heq] using proof

theorem finite_scaled_exact (code : Fin 256) (scale : Fin 255)
    (finite : code.val ≠ 128 ∨ scale.val ≠ 254) :
    decoded code.val scale.val = some (ideal code.val scale.val) := by
  have impossible : ¬(code.val = 128 ∧ scale.val = 254) := by omega
  simpa [expected, impossible] using scaled_exact_except_overflow code scale

theorem overflow_output : upS2F6 false false true 128 254 = 0xff80 := by decide

theorem invalid_scale_nan : ∀ code : Fin 256,
    upS2F6 false false true code.val 255 = 0x7fff := by decide +kernel

#print axioms scaled_exact_except_overflow
#print axioms finite_scaled_exact
#print axioms overflow_output
#print axioms invalid_scale_nan

end GB300Math.S2F6FloatLibAccuracy
