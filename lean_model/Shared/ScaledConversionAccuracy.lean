import FloatLib.Floats.Formats.BinaryInterchange.Configured
import Mathlib.Tactic.IntervalCases
import SmallConversions

set_option maxRecDepth 1000000
set_option maxHeartbeats 2000000000

namespace GB300Math.ScaledConversionAccuracy
open FloatLib.Floats GB300Math.SmallConversions

def finiteIdeal (name : String) (fmt : Format) (word scale : Nat) : Option ℚ :=
  let exponent := word / 2^fmt.fractionBits % 2^fmt.exponentBits
  let fraction := word % 2^fmt.fractionBits
  if (name = "e4m3" ∧ exponent = 15 ∧ fraction = 7) ∨
      (name = "e5m2" ∧ exponent = 31) then none else
    let negative := word / 2^(fmt.fractionBits+fmt.exponentBits) % 2 = 1
    let mantissa := fraction + if exponent = 0 then 0 else 2^fmt.fractionBits
    let power : Int := (max exponent 1 : Int) - (fmt.bias : Int) -
      (fmt.fractionBits : Int) + (scale : Int) - 127
    some ((if negative then -1 else 1) * (mantissa : ℚ) * (2 : ℚ)^power)

def maxFinite : ℚ := 255 * (2 : ℚ)^120

def specifiedIdeal (relu sat : Bool) (value : ℚ) : ℚ :=
  let rectified := if relu then max 0 value else value
  if sat then max (-maxFinite) (min maxFinite rectified) else rectified

def decoded (name : String) (fmt : Format) (word scale : Nat) (relu sat : Bool) : Option ℚ :=
  ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits
    (upSmall name fmt true relu sat true word scale) : ExecFloat.Binary 8 7)

def accurate (name : String) (fmt : Format) (bound : ℚ) (word scale : Nat)
    (relu sat : Bool) : Prop :=
  match finiteIdeal name fmt word scale with
  | none => True
  | some source =>
    let exact := specifiedIdeal relu sat source
    if |exact| > maxFinite then True else
      match decoded name fmt word scale relu sat with
      | none => False
      | some actual => |actual-exact| ≤ bound ∧
        (|exact| ≥ (2 : ℚ)^(-126 : Int) → actual = exact)

instance (name : String) (fmt : Format) (bound : ℚ) (word scale : Nat)
    (relu sat : Bool) : Decidable (accurate name fmt bound word scale relu sat) := by
  unfold accurate
  split
  · infer_instance
  · dsimp
    split
    · infer_instance
    · split <;> infer_instance

end GB300Math.ScaledConversionAccuracy
