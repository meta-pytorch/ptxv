import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.FabricRounding
open FloatLib.Floats.Formats.BinaryInterchange

def candidates (fmt : FloatFormat) (inputs : List Nat) (rne : Bool) : Option (List Nat) := do
  unless inputs.all (fun word => word < 2^fmt.bitWidth) do none
  let values := inputs.toArray.map fun x => (Model.ofNatBits x : Model fmt)
  unless values.all Model.isFinite do none
  let modes : List Model.IEEERoundingMode := if rne then [.nearestEven]
    else [.towardNegativeInfinity, .towardPositiveInfinity]
  let rounded := modes.map fun mode =>
    let word := Model.toNatBits (Model.sum fmt values mode)
    if word % 2^(fmt.bitWidth-1) = 0 then 0 else word
  unless rounded.all (fun word => Model.isFinite (Model.ofNatBits word : Model fmt)) do none
  return rounded

end GB300Math.FabricRounding
