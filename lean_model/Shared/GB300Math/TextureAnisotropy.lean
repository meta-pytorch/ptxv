import Std
import GB300Math.MultisampleFetch

namespace GB300Math.TextureAnisotropy

def lod (major minor logAnisotropy : Int) : Int :=
  max 0 (min 8 (max (min major minor) (max major minor-logAnisotropy)))

def levelWord (major minor logAnisotropy : Int) : Nat :=
  MultisampleFetch.encode (lod major minor logAnisotropy+1).toNat + 16*2^23

def pointWord (checker : Bool) (x y channel : Nat) : Nat :=
  if !checker ∨ channel=0 then 0x3f800000
  else if (if channel=1 then x%2 else if channel=2 then y%2 else (x+y)%2)=0 then 0
  else 0x3f800000

theorem lod_range (major minor logAnisotropy : Int) :
    0 ≤ lod major minor logAnisotropy ∧ lod major minor logAnisotropy ≤ 8 := by
  unfold lod
  omega

theorem more_anisotropy (major minor a b : Int) (hab : a ≤ b) :
    lod major minor b ≤ lod major minor a := by
  unfold lod
  omega

theorem isotropic (major minor : Int) :
    lod major minor 0 = max 0 (min 8 (max major minor)) := by
  unfold lod
  omega

theorem four_level_difference : levelWord 6 0 0 = 0x40e00000 ∧
    levelWord 6 0 4 = 0x40400000 := by decide +kernel

end GB300Math.TextureAnisotropy
