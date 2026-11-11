import GB300Math.TextureMipmap

namespace GB300Math.TextureGradientAxis

def logTable : Array Nat := #[0,2,4,5,6,8,9,11,12,13,15,16,18,19,20,22,23,24,26,27,28,30,31,32,34,35,36,38,39,40,42,43,44,45,47,48,49,50,52,53,54,55,57,58,59,60,62,63,64,65,66,68,69,70,71,72,74,75,76,77,78,79,81,82,83,84,85,86,88,89,90,91,92,93,94,95,97,98,99,100,101,102,103,104,105,106,107,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,135,136,137,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,162,163,164,165,166,167,168,169,170,171,172,173,173,174,175,176,177,178,179,180,181,181,182,183,184,185,186,187,188,188,189,190,191,192,193,194,195,195,196,197,198,199,200,200,201,202,203,204,205,205,206,207,208,209,210,210,211,212,213,214,214,215,216,217,218,218,219,220,221,222,222,223,224,225,226,226,227,228,229,229,230,231,232,233,233,234,235,236,236,237,238,239,239,240,241,242,242,243,244,245,245,246,247,248,248,249,250,251,251,252,253,253,254,255,255]

def position (v : Nat) : Nat :=
  let e : Nat := v/2^23%256
  let m : Nat := v%2^23
  if e = 0 ∨ (e = 255 ∧ m ≠ 0) then 0 else
    min 2048 (((e:Int)-119)*256+(logTable[m/2^15]!:Int)).toNat

def sample (v : Nat) : Nat := TextureMipmap.encodeQ8 (256+position v)

theorem position_bounded (v : Nat) : position v ≤ 2048 := by
  unfold position
  dsimp only
  split
  · exact Nat.zero_le _
  · exact Nat.min_le_left _ _

theorem nan_zero_lod (v : Nat) (he : v/2^23%256 = 255) (hm : v%2^23 ≠ 0) :
    position v = 0 := by simp [position, he, hm]

theorem subnormal_zero_lod (v : Nat) (he : v/2^23%256 = 0) :
    position v = 0 := by simp [position, he]

#print axioms position_bounded
#print axioms nan_zero_lod
#print axioms subnormal_zero_lod
end GB300Math.TextureGradientAxis
