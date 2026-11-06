import Log2CellData
import Log2TightCells
namespace GB300Math.Log2OneSidedCells
open Log2TightCells
open Log2Cells
set_option maxRecDepth 1000000
set_option maxHeartbeats 0

abbrev OneSided (j : Nat) : Prop :=
  let (a,b,c) := differences j;
  -54/1000000000+1/2^36≤a ∧ a≤173/1000000000-correction j ∧
  -54/1000000000+1/2^36≤a+b/2 ∧ a+b/2≤173/1000000000-correction j ∧
  -54/1000000000+1/2^36≤a+b+c ∧ a+b+c≤173/1000000000-correction j

theorem cells_one_sided : ∀ j : Fin 512,OneSided j.val := by decide +kernel
#print axioms cells_one_sided
end GB300Math.Log2OneSidedCells
