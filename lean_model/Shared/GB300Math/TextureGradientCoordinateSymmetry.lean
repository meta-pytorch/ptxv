import GB300Math.TextureGradient3D

namespace GB300Math.TextureGradientCoordinateSymmetry
open TextureGradient3D

theorem positive_unit_lod :
    position 0x4b800000 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 0 := by decide +kernel

theorem negative_unit_lod :
    position 0x4b800000 0x3e800000 0xbf000000 0xbf800000 0 0 0 0 0 = 1280 := by decide +kernel

theorem positive_unit_sample :
    sample 0x4b800000 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 0x3f800000 := by decide +kernel

theorem negative_unit_sample :
    sample 0x4b800000 0x3e800000 0xbf000000 0xbf800000 0 0 0 0 0 = 0x40c00000 := by decide +kernel

theorem sign_reversal_maximum_gap :
    position 0x4b800000 0x3e800000 0xbf000000 0xbf800000 0 0 0 0 0 -
    position 0x4b800000 0x3e800000 0xbf000000 0x3f800000 0 0 0 0 0 = 1280 := by
  rw [positive_unit_lod, negative_unit_lod]

#print axioms positive_unit_lod
#print axioms negative_unit_lod
#print axioms positive_unit_sample
#print axioms negative_unit_sample
#print axioms sign_reversal_maximum_gap
end GB300Math.TextureGradientCoordinateSymmetry
