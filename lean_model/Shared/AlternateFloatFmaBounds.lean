import AlternateFloatPairBounds
namespace GB300Math.AlternateFloat
def tripleBound (dst : String) (a b c : Nat) : Bool :=
  match decode dst a, decode dst b, decode dst c with
  | some x, some y, some z =>
    let ideal := add (mul x y) z
    match gridInteger ideal, decode dst (encode dst false ideal) with
    | some v, some output =>
      match gridInteger output with
      | some rounded =>
        let factor := if dst = "e4m3" then 17 else 9
        let floor := if dst = "e4m3" then 17*2^502 else 9*2^495
        decide (factor * (rounded-v).natAbs ≤ max v.natAbs floor)
      | _ => true
    | _, _ => true
  | _, _, _ => true
end GB300Math.AlternateFloat
