import GB300Math.TextureAnisotropyOriented
namespace GB300Math.TextureAnisotropyOriented
set_option maxRecDepth 100000
set_option maxHeartbeats 0

def cellLow (a : Nat) : Nat := max 1 (2*a-2)
def cellHigh (a : Nat) : Nat := 2*a+2

theorem cell_endpoint_enclosures : ∀ i j : Fin 16,
    let a := 2*i.val+1
    let b := 2*j.val+1
    max a b ≥ 17 →
    let v := directionTable[a]![b]!
    (8*v-37)^2*(cellLow a^2+cellHigh b^2) ≤ 512^2*cellLow a^2 ∧
    512^2*cellHigh a^2 ≤ (8*v+37)^2*(cellHigh a^2+cellLow b^2) := by decide

theorem direction_rounding_witness : directionTable[17]![7]! = 59 ∧
    directionTable[7]![17]! = 24 := by decide

theorem four_sixty_fourths_too_small :
    (24+4)^2*(16^2+8^2) < 64^2*8^2 := by decide

#print axioms cell_endpoint_enclosures
#print axioms direction_rounding_witness
#print axioms four_sixty_fourths_too_small
end GB300Math.TextureAnisotropyOriented
