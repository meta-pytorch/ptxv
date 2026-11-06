import FloatLib.Floats.Formats.BinaryInterchange.Configured
import MufuReciprocalBounds

namespace GB300Math.MufuReciprocalBounds

open FloatLib.Floats

def rationalValue (bf16 : Bool) (word : Nat) : ℚ :=
  let (numerator, denominator) := dyadic bf16 word
  let magnitude := (numerator : ℚ) / denominator
  if word / 32768 % 2 = 1 then -magnitude else magnitude

def floatLibValue (bf16 : Bool) (word : Nat) : Option ℚ :=
  if bf16 then
    ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits word : ExecFloat.Binary 8 7)
  else
    ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits word : ExecFloat.Binary 5 10)

def decodeMatches (bf16 : Bool) (word : Nat) : Bool :=
  if word % 32768 < infinity bf16 then
    floatLibValue bf16 word == some (rationalValue bf16 word)
  else floatLibValue bf16 word == none

theorem decoded_finite (bf16 : Bool) (word : Nat)
    (finite : word % 32768 < infinity bf16)
    (checked : decodeMatches bf16 word = true) :
    floatLibValue bf16 word = some (rationalValue bf16 word) := by
  simpa [decodeMatches, finite] using checked

theorem decoded_nonfinite (bf16 : Bool) (word : Nat)
    (nonfinite : infinity bf16 ≤ word % 32768)
    (checked : decodeMatches bf16 word = true) :
    floatLibValue bf16 word = none := by
  simpa [decodeMatches, not_lt.mpr nonfinite] using checked

#print axioms decoded_finite
#print axioms decoded_nonfinite

end GB300Math.MufuReciprocalBounds
