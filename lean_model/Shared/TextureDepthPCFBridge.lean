import TextureLinearCoordinates
import GB300Math.TextureDepthPCF

namespace GB300Math.TextureDepthPCF

theorem coordinate_binding (word : Nat) :
    coordinate word = TextureLinearCoordinates.coordinate word 4 true := by
  have h (e : Int) : e - 142 + ((4 : Nat) : Int) = e - 138 := by omega
  simp only [coordinate, TextureLinearCoordinates.coordinate, ite_true, h]

theorem weight_binding (x y : Nat) :
    plane x y = TextureLinearCoordinates.weightPlane x y 256 := by
  have h : (x*256+128)/256 = x := by omega
  simp only [plane, TextureLinearCoordinates.weightPlane, h]

#print axioms coordinate_binding
#print axioms weight_binding

end GB300Math.TextureDepthPCF
