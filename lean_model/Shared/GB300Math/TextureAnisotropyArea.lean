import GB300Math.TextureGradientAxis

namespace GB300Math.TextureAnisotropyArea

def omitted (x y : Nat) : Nat :=
  (x%2)*(y%64) + (x/2%2)*(y%32)*2 +
  (x/4%2)*(y%16)*4 + (x/8%2)*(y%8)*8 +
  (x/16%2)*(y%4)*16 + (x/32%2)*(y%2)*32

def product (x y : Nat) : Nat := (x*y-omitted x y+448)/1024

def significand (v : Nat) : Nat := 512+v%2^23/2^14

def scale (v : Nat) : Int := (v/2^23%256:Int)-136

def logarithm (s : Nat) (e : Int) : Int :=
  let k := s.log2
  (e+k)*256 + (TextureGradientAxis.logTable[s*256/2^k-256]!:Int)

def position (a b anisotropy : Nat) : Nat :=
  let x := significand a
  let y := significand b
  let major := max (logarithm x (scale a)) (logarithm y (scale b))
  let area := logarithm (product x y) (scale a+scale b+10)
  let minor := min major (area-major)
  let capped := max minor (major-(anisotropy.log2:Int)*256)
  min 2048 (2048+capped).toNat

def sample (a b anisotropy : Nat) : Nat :=
  TextureMipmap.encodeQ8 (256+position a b anisotropy)

theorem omitted_bound (x y : Nat) : omitted x y ≤ 321 := by
  have b0 := Nat.mod_lt x (by decide : 0<2)
  have b1 := Nat.mod_lt (x/2) (by decide : 0<2)
  have b2 := Nat.mod_lt (x/4) (by decide : 0<2)
  have b3 := Nat.mod_lt (x/8) (by decide : 0<2)
  have b4 := Nat.mod_lt (x/16) (by decide : 0<2)
  have b5 := Nat.mod_lt (x/32) (by decide : 0<2)
  have h0 := Nat.mul_le_mul (show x%2≤1 by omega) (show y%64≤63 by omega)
  have h1 := Nat.mul_le_mul (show x/2%2≤1 by omega) (show y%32≤31 by omega)
  have h2 := Nat.mul_le_mul (show x/4%2≤1 by omega) (show y%16≤15 by omega)
  have h3 := Nat.mul_le_mul (show x/8%2≤1 by omega) (show y%8≤7 by omega)
  have h4 := Nat.mul_le_mul (show x/16%2≤1 by omega) (show y%4≤3 by omega)
  have h5 := Nat.mul_le_mul (show x/32%2≤1 by omega) (show y%2≤1 by omega)
  unfold omitted
  omega

theorem product_error (x y : Nat) (hx : 512≤x) (hy : 512≤y) :
    -896 ≤ (1024*product x y:Int)-(x*y:Int) ∧
    (1024*product x y:Int)-(x*y:Int) ≤ 448 := by
  have h := omitted_bound x y
  have hp := Nat.mul_le_mul hx hy
  have hd := Nat.div_add_mod (x*y-omitted x y+448) 1024
  have hm := Nat.mod_lt (x*y-omitted x y+448) (by decide : 0<1024)
  unfold product
  omega

theorem significand_bounds (v : Nat) : 512≤significand v ∧ significand v≤1023 := by
  unfold significand
  omega

theorem position_bounded (a b anisotropy : Nat) : position a b anisotropy ≤ 2048 := by
  unfold position
  exact Nat.min_le_left _ _

theorem product_not_nearest :
    (1024*product 575 639:Int)-(575*639:Int) = -833 := by decide

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 0 in
theorem omitted_mod (x y : Nat) : omitted x y%64 = x*y%64 := by
  have finite : ∀ a b : Fin 64, omitted a.val b.val%64 = a.val*b.val%64 := by decide
  have x0 : x%64%2=x%2 := by omega
  have x1 : x%64/2%2=x/2%2 := by omega
  have x2 : x%64/4%2=x/4%2 := by omega
  have x3 : x%64/8%2=x/8%2 := by omega
  have x4 : x%64/16%2=x/16%2 := by omega
  have x5 : x%64/32%2=x/32%2 := by omega
  have y0 : y%64%64=y%64 := by omega
  have y1 : y%64%32=y%32 := by omega
  have y2 : y%64%16=y%16 := by omega
  have y3 : y%64%8=y%8 := by omega
  have y4 : y%64%4=y%4 := by omega
  have y5 : y%64%2=y%2 := by omega
  have periodic : omitted (x%64) (y%64)=omitted x y := by
    simp only [omitted, x0,x1,x2,x3,x4,x5,y0,y1,y2,y3,y4,y5]
  have h := finite ⟨x%64,by omega⟩ ⟨y%64,by omega⟩
  dsimp only at h
  rw [periodic, ← Nat.mul_mod] at h
  exact h

theorem product_tight_error (x y : Nat) (hx : 512≤x) (hy : 512≤y) :
    -833 ≤ (1024*product x y:Int)-(x*y:Int) ∧
    (1024*product x y:Int)-(x*y:Int) ≤ 448 := by
  have h := omitted_bound x y
  have hp := Nat.mul_le_mul hx hy
  have he := omitted_mod x y
  have hr : (x*y-omitted x y+448)%64=0 := by omega
  have hd := Nat.div_add_mod (x*y-omitted x y+448) 1024
  have hm := Nat.mod_lt (x*y-omitted x y+448) (by decide : 0<1024)
  unfold product
  omega

#print axioms omitted_bound
#print axioms product_error
#print axioms significand_bounds
#print axioms position_bounded
#print axioms product_not_nearest
#print axioms omitted_mod
#print axioms product_tight_error
end GB300Math.TextureAnisotropyArea
